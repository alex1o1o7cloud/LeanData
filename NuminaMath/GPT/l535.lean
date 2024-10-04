import Mathlib

namespace probability_both_red_is_1_10_probability_one_white_one_red_is_3_5_l535_535727

-- Definition of the problem in Lean 4
def num_white_balls : ℕ := 3
def num_red_balls : ℕ := 2
def total_balls : ℕ := num_white_balls + num_red_balls
def possible_pairs : finset (finset ℕ) := 
  ((finset.range total_balls).powerset.filter (λ s, s.card = 2))

-- Draw 2 balls
def both_red : finset ℕ := {3, 4} -- pairs {4, 5} in zero-indexed as {3, 4}
def one_white_one_red : finset (finset ℕ) :=
  ({(0,3), (0,4), (1,3), (1,4), (2,3), (2,4): finset (ℕ)}).map (function.uncurry finset.singleton)

-- Probabilities
def P_both_red := both_red.card.to_rat / possible_pairs.card.to_rat
def P_one_white_one_red := one_white_one_red.card.to_rat / possible_pairs.card.to_rat

theorem probability_both_red_is_1_10 :
  P_both_red = 1 / 10 := sorry

theorem probability_one_white_one_red_is_3_5 :
  P_one_white_one_red = 3 / 5 := sorry

end probability_both_red_is_1_10_probability_one_white_one_red_is_3_5_l535_535727


namespace problem_statement_l535_535605

theorem problem_statement (a b : ℝ) 
  (h1 : 30 ^ a = 4) 
  (h2 : 30 ^ b = 9) : 
  18 ^ ((1 - a - b) / (2 * (1 - b))) = 5 / 6 := 
sorry

end problem_statement_l535_535605


namespace problem_solution_l535_535310

open Real

noncomputable def f (x : ℝ) : ℝ := log x

theorem problem_solution (a b : ℝ) (ha : 0 < a) (hb : a < b) :
  let p := f (sqrt (a * b))
  let q := f ((a + b) / 2)
  let r := (f a + f b) / 2
  in p = r ∧ p < q :=
by
  sorry

end problem_solution_l535_535310


namespace original_price_color_TV_l535_535057

theorem original_price_color_TV (x : ℝ) 
  (h : 1.12 * x - x = 144) : 
  x = 1200 :=
sorry

end original_price_color_TV_l535_535057


namespace find_t_l535_535886

open Real

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

noncomputable def curve (x y t : ℝ) : Prop := y = 3 * |x - t|

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (positive_coords : x > 0 ∧ y > 0)

def distance (P Q : Point) : ℝ :=
  sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

theorem find_t (m n s p t k : ℝ) (A B : Point)
  (hA : curve A.x A.y t)
  (hB : curve B.x B.y t)
  (h1 : m > 0) (h2 : n > 0) (h3 : s > 0) (h4 : p > 0)
  (hO : ∀ (x y : ℝ), circle x y → (∃ (P : Point), P.x = x ∧ P.y = y))
  (hratio : ∀ (P : Point) (x1 y1 : ℝ), circle x1 y1
    → distance P A / distance P B = k)
  (hk : k > 1) :
  t = 1 :=
sorry

end find_t_l535_535886


namespace min_value_of_expression_l535_535420

theorem min_value_of_expression (x y : ℝ) : 
  ∃ x y, 2 * x^2 + 3 * y^2 - 8 * x + 12 * y + 40 = 20 := 
sorry

end min_value_of_expression_l535_535420


namespace binary_to_octal_conversion_l535_535105

theorem binary_to_octal_conversion (n : ℕ) (hn : n = 0b101110) : nat.to_octal n = 0o56 :=
by
  sorry

end binary_to_octal_conversion_l535_535105


namespace smallest_number_exceeding_triangle_perimeter_l535_535754

theorem smallest_number_exceeding_triangle_perimeter (a b : ℕ) (a_eq_7 : a = 7) (b_eq_21 : b = 21) :
  ∃ P : ℕ, (∀ c : ℝ, 14 < c ∧ c < 28 → a + b + c < P) ∧ P = 56 := by
  sorry

end smallest_number_exceeding_triangle_perimeter_l535_535754


namespace linear_regression_test_l535_535228

theorem linear_regression_test (x y : ℕ → ℝ) (b : ℝ)
  (h_sum_x : ∑ i in finset.range 10, x i = 20)
  (h_sum_y : ∑ i in finset.range 10, y i = 30)
  (h_regression : ∀ i, y i = -3 + b * x i) :
  b = 3 :=
sorry

end linear_regression_test_l535_535228


namespace incorrect_statement_among_ABCD_l535_535028

theorem incorrect_statement_among_ABCD :
  ¬ (∀ a b : ℝ, a < b ∧ b < 0 → (
    ∀ x y : ℝ, x = a ∧ y = b → (¬ ∃ c : ℝ, c^2 = x) ∧ (¬ ∃ d : ℝ, d^2 = y))) :=
begin
  sorry
end

end incorrect_statement_among_ABCD_l535_535028


namespace asymptotes_of_hyperbola_l535_535841

theorem asymptotes_of_hyperbola (x y : ℝ) :
  (x ^ 2 / 4 - y ^ 2 / 9 = -1) →
  (y = (3 / 2) * x ∨ y = -(3 / 2) * x) :=
sorry

end asymptotes_of_hyperbola_l535_535841


namespace insphere_midpoints_coplanar_l535_535298

noncomputable def isInPlane (p1 p2 p3 p4 : Point) : Prop := sorry

variable {Point : Type}

theorem insphere_midpoints_coplanar {A B C D O Mbc Mad Mac Mbd : Point}
  (hO_center : isIncenter O A B C D)
  (h_sum_areas : area A B C + area A B D = area C D A + area C D B)
  (hMbc : midpoint B C Mbc)
  (hMad : midpoint A D Mad)
  (hMac : midpoint A C Mac)
  (hMbd : midpoint B D Mbd) :
  isInPlane O Mbc Mad Mac := sorry

end insphere_midpoints_coplanar_l535_535298


namespace factorization_l535_535525

-- Prove that the given polynomial expression equals (1 + x)^n
theorem factorization (x : ℝ) (n : ℕ) : 
  (1 + x) + x * (1 + x) + x * (1 + x)^2 + ∑ (k : ℕ) in finset.range (n - 1), x * (1 + x) ^ (k + 2) = (1 + x) ^ n := 
by
  sorry

end factorization_l535_535525


namespace sum_of_abs_values_eq_12_l535_535954

theorem sum_of_abs_values_eq_12 (a b c d : ℝ) (h : 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) :
  abs a + abs b + abs c + abs d = 12 := sorry

end sum_of_abs_values_eq_12_l535_535954


namespace complement_A_l535_535227

noncomputable def U : Set ℝ := Set.univ

noncomputable def A : Set ℝ := {x | (x + 2) / x < 0}

theorem complement_A : (U \ A) = {x | x ≥ 0 ∨ x ≤ -2} :=
by
  ext
  split
  case mp =>
    intro hx
    simp [U, A] at hx
    by_cases h1 : x = 0
    case inl =>
      exfalso
      simp [h1] at hx
    case inr =>
      simp at hx
      have : (x + 2) / x >= 0, from not_lt.mp hx
      cases lt_or_gt_of_ne h1 with h2 h3
      case inl =>
        right
        linarith [hx]
      case inr =>
        simp at this
        left
        linarith
  case mpr =>
    intro hx
    simp [U, A]
    cases hx
    case inl =>
      have : x + 2 > 0, from add_pos_of_pos_of_nonneg hx zero_le_two
      have : x > 0, from best_of_scalar_field_of_div_pos admin_hx
      contrapose hx
      intro h
      suffices : false, from absurd this not_false,
      linarith
    case inr =>
      have : x + 2 < 0, from add_neg_of_neg_of_pos_of_lt hx not_sorry_zero_le_neg.so
      have : x < 0, from best_of_scalar_field_of_div_ltp admin_hx
      simp at this
      linarith

end complement_A_l535_535227


namespace intersection_points_isosceles_triangle_l535_535585

-- Defining the line and the circle
def line (x y : ℝ) : Prop := 2 * x + y + 4 = 0
def circle (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 1 = 0

-- Coordinates of A
def A : ℝ × ℝ := (-11 / 5, 2 / 5)

-- Coordinates of B
def B : ℝ × ℝ := (-3, 2)

-- Possible coordinates of D making ABD an isosceles triangle
def D1 : ℝ × ℝ := (-5, 0)
def D2 : ℝ × ℝ := (-11 / 5 + 2 * Real.sqrt 19 / 5, 0)
def D3 : ℝ × ℝ := (-11 / 5 - 2 * Real.sqrt 19 / 5, 0)

theorem intersection_points :
  line A.1 A.2 ∧ circle A.1 A.2 ∧ 
  line B.1 B.2 ∧ circle B.1 B.2 :=
sorry

theorem isosceles_triangle (D : ℝ × ℝ) :
  D = D1 ∨ D = D2 ∨ D = D3 :=
sorry

end intersection_points_isosceles_triangle_l535_535585


namespace meeting_point_2015th_l535_535173

-- Define the parameters of the problem
variables (A B C D : Type)
variables (x y t : ℝ) -- Speeds and the initial time delay

-- State the problem as a theorem
theorem meeting_point_2015th (start_times_differ : t > 0)
                            (speeds_pos : x > 0 ∧ y > 0)
                            (pattern : ∀ n : ℕ, (odd n → (meeting_point n = C)) ∧ (even n → (meeting_point n = D)))
                            (n = 2015) :
  meeting_point n = C :=
  sorry

end meeting_point_2015th_l535_535173


namespace length_of_major_axis_l535_535197

-- Definitions based on the given conditions
def ellipse (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ (∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
def foci_distance (c : ℝ) : Prop := c = sqrt 3
def perpendicular_chord (a b : ℝ) (AB : ℝ) : Prop := AB = 4 ∧ 2 * sqrt (a^2 - b^2) = 2 * sqrt 3

-- Main problem statement
theorem length_of_major_axis (a b : ℝ) (h1 : ellipse a b) (h2 : foci_distance (sqrt (a^2 - b^2)))
  (h3 : perpendicular_chord a b 4) : 2 * a = 6 :=
sorry

end length_of_major_axis_l535_535197


namespace kite_area_l535_535339

-- Given conditions
variables {A B C D : Type} -- points representing the vertices of the kite
variables (AB AD BC CD AC BD : ℝ) -- lengths

-- Definitions restricted to the conditions provided
def is_kite (AB AD BC CD AC BD : ℝ) := AB = AD ∧ BC = CD ∧ AC = 30 ∧ (BD^2 = (25 * 7))

-- Lean statement we want to prove
theorem kite_area (h : is_kite 15 15 20 20 30 BD) : ½ * AC * BD = 150 * Real.sqrt 7 :=
sorry

end kite_area_l535_535339


namespace ratio_of_length_to_width_of_field_is_two_to_one_l535_535370

-- Definitions based on conditions
def lengthOfField : ℕ := 80
def widthOfField (field_area pond_area : ℕ) : ℕ := field_area / lengthOfField
def pondSideLength : ℕ := 8
def pondArea : ℕ := pondSideLength * pondSideLength
def fieldArea : ℕ := pondArea * 50
def lengthMultipleOfWidth (length width : ℕ) := ∃ k : ℕ, length = k * width

-- Main statement to prove the ratio of length to width is 2:1
theorem ratio_of_length_to_width_of_field_is_two_to_one :
  lengthMultipleOfWidth lengthOfField (widthOfField fieldArea pondArea) →
  lengthOfField = 2 * (widthOfField fieldArea pondArea) :=
by
  -- Conditions
  have h1 : pondSideLength = 8 := rfl
  have h2 : pondArea = pondSideLength * pondSideLength := rfl
  have h3 : fieldArea = pondArea * 50 := rfl
  have h4 : lengthOfField = 80 := rfl
  sorry

end ratio_of_length_to_width_of_field_is_two_to_one_l535_535370


namespace num_int_values_n_terminated_l535_535152

theorem num_int_values_n_terminated (N : ℕ) (hN1 : 1 ≤ N) (hN2 : N ≤ 500) :
  ∃ n : ℕ, n = 10 ∧ ∀ k, 0 ≤ k → k < n → ∃ (m : ℕ), N = m * 49 :=
sorry

end num_int_values_n_terminated_l535_535152


namespace f_2013_value_l535_535254

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, x ≠ 1 → f (2 * x + 1) + g (3 - x) = x
axiom h2 : ∀ x : ℝ, x ≠ 1 → f ((3 * x + 5) / (x + 1)) + 2 * g ((2 * x + 1) / (x + 1)) = x / (x + 1)

theorem f_2013_value : f 2013 = 1010 / 1007 :=
by
  sorry

end f_2013_value_l535_535254


namespace paving_stone_width_l535_535401

theorem paving_stone_width :
  let courtyard_length := 70
  let courtyard_width := 16.5
  let num_paving_stones := 231
  let paving_stone_length := 2.5
  let courtyard_area := courtyard_length * courtyard_width
  let total_area_covered := courtyard_area
  let paving_stone_width := total_area_covered / (paving_stone_length * num_paving_stones)
  paving_stone_width = 2 :=
by
  sorry

end paving_stone_width_l535_535401


namespace brenda_spay_cats_l535_535829

theorem brenda_spay_cats (c d : ℕ) (h1 : c + d = 21) (h2 : d = 2 * c) : c = 7 :=
sorry

end brenda_spay_cats_l535_535829


namespace areas_equal_l535_535986

open EuclideanGeometry

-- Define points and triangle
variables (A B C E M N : Point)
variable [noncollinear A B C]
variable (triangleABC : Triangle A B C)
variable (angleBisectorAE : AngleBisector triangleABC (Line.mk A E))

-- Define the circumcircles
variables (circumcircleAEB : Circle (Triangle.mk A E B))
variables (circumcircleAEC : Circle (Triangle.mk A E C))

-- Define the intersection points
variables (intersectionN_on_AC : N ∈ Circle.intersect_side circumcircleAEC A C)
variables (intersectionM_on_AB : M ∈ Circle.intersect_side circumcircleAEB A B)

-- The main theorem to prove
theorem areas_equal (h: cn_includes_conditions) :
  area (Triangle.mk B M E) = area (Triangle.mk C N E) :=
sorry

end areas_equal_l535_535986


namespace path_count_outside_square_l535_535452

theorem path_count_outside_square :
  let paths := {p : ℕ → ℤ × ℤ // ∀ n, p (n + 1) = (p n.1 + 1, p n.2) ∨ p (n + 1) = (p n.1, p n.2 + 1) ∧
                                               p 0 = (-5, -5) ∧ p 20 = (5, 5)};
  let valid_path := λ p, ∀ n, ¬(-3 ≤ (p n).1 ∧ (p n).1 ≤ 3 ∧ -3 ≤ (p n).2 ∧ (p n).2 ≤ 3);
  finset.card {p ∈ paths // valid_path p} = 4252 :=
sorry

end path_count_outside_square_l535_535452


namespace domain_f_2x_minus_1_l535_535220

theorem domain_f_2x_minus_1 (f : ℝ → ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → 2 ≤ x + 1 ∧ x + 1 ≤ 3) → 
  (∀ z, 2 ≤ 2 * z - 1 ∧ 2 * z - 1 ≤ 3 → ∃ x, 3/2 ≤ x ∧ x ≤ 2 ∧ 2 * x - 1 = z) := 
sorry

end domain_f_2x_minus_1_l535_535220


namespace jane_reaches_R_after_4_tosses_l535_535332

/-- Define the probability of reaching dot R after four coin tosses -/
def probability_reach_R : ℚ := 3 / 8

/-- Define the main problem as a theorem statement -/
theorem jane_reaches_R_after_4_tosses :
  ∀ (start : string) (moves : ℕ → string),
  start = "A" ∧
    (∀ toss, (moves toss = "U" ∨ moves toss = "R")) ∧
    (∀ toss, (∃ seq, seq.length = 4 ∧ list.count seq "U" = 2 ∧ list.count seq "R" = 2)) →
  probability_reach_R = 3 / 8 :=
sorry

end jane_reaches_R_after_4_tosses_l535_535332


namespace batsman_average_after_12th_innings_l535_535030

theorem batsman_average_after_12th_innings 
  (A : ℕ) 
  (total_runs_11_innings : ℕ := 11 * A) 
  (new_average : ℕ := A + 2) 
  (total_runs_12_innings : ℕ := total_runs_11_innings + 92) 
  (increased_average_after_12 : 12 * new_average = total_runs_12_innings) 
  : new_average = 70 := 
by
  -- skipping proof
  sorry

end batsman_average_after_12th_innings_l535_535030


namespace intersection_point_P_condition_on_a_equation_of_line_l535_535931

def line1 (x y : ℝ) : Prop := x - 2 * y + 4 = 0
def line2 (x y : ℝ) : Prop := 4 * x + 3 * y + 5 = 0
def pointA : ℝ × ℝ := (-1, -2)
def pointP : ℝ × ℝ := (-2, 1)

theorem intersection_point_P : ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ (x, y) = pointP := by
  sorry

theorem condition_on_a (a : ℝ) : ¬ (a = -2 ∨ a = -1 ∨ a = 8 / 3) ↔ 
  ¬ (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ (a * x + 2 * y - 6 = 0)) := by
  sorry

theorem equation_of_line (k : ℝ) : 
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ ((k * (x + 2) - y + k) = 0 ∨ (x + 2 = 0))) 
  ∧ (abs (k + 2 - 2 * k + 1) / sqrt (k^2 + 1) = 1) ↔ (k = -4/3 ∨ k = 0) := by
  sorry

end intersection_point_P_condition_on_a_equation_of_line_l535_535931


namespace period_and_max_of_f_find_b_l535_535937

-- First, define the basic settings and function for part I.
def m : ℝ × ℝ := (1, real.sqrt 3)
def n (x : ℝ) : ℝ × ℝ := (real.sin x, real.cos x)
def f (x : ℝ) : ℝ := m.1 * n x.1 + m.2 * n x.2

-- Part I: Prove the smallest positive period and maximum value of f(x)
theorem period_and_max_of_f : (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ x, f x ≤ 2 ∧ ∃ x₀, f x₀ = 2) :=
  sorry

-- Part II: Given, then prove b in the acute triangle
def c : ℝ := real.sqrt 6
def cos_B : ℝ := 1 / 3
def f_C : ℝ := real.sqrt 3

theorem find_b (a b c : ℝ) (A B C : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (h_cos_B : real.cos B = cos_B) (hC_eq : f C = f_C) :
  b = 8 / 3 :=
  sorry

end period_and_max_of_f_find_b_l535_535937


namespace jellyfish_cost_l535_535818

theorem jellyfish_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : J = 20 := by
  sorry

end jellyfish_cost_l535_535818


namespace contrapositive_of_even_sum_l535_535706

def Even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem contrapositive_of_even_sum (a b : ℕ) :
  (Even a ∧ Even b → Even (a + b)) ↔ 
  (¬Even (a + b) → ¬(Even a ∧ Even b)) := by
  sorry

end contrapositive_of_even_sum_l535_535706


namespace original_price_l535_535764

theorem original_price (P : ℝ) (h : 0.75 * (0.75 * P) = 17) : P = 30.22 :=
by
  sorry

end original_price_l535_535764


namespace product_form_l535_535640

theorem product_form (b a : ℤ) :
  (10 * b + a) * (10 * b + 10 - a) = 100 * b * (b + 1) + a * (10 - a) := 
sorry

end product_form_l535_535640


namespace volume_of_prism_l535_535703

theorem volume_of_prism
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 40)
  (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := 
by
  sorry

end volume_of_prism_l535_535703


namespace monotonic_intervals_max_min_on_interval_l535_535916

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

theorem monotonic_intervals :
  (∀ x, x < -1 → f' x > 0) ∧ (∀ x, x > 1 → f' x > 0) ∧ (∀ x, -1 < x ∧ x < 1 → f' x < 0) :=
sorry

theorem max_min_on_interval :
  ∃ x_max x_min, 
  x_max ∈ Set.Icc (-3 : ℝ) 3 ∧ x_min ∈ Set.Icc (-3 : ℝ) 3 ∧ 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧ 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x_min ≤ f x) ∧ 
  f x_max = 59 ∧ f x_min = -49 :=
sorry

end monotonic_intervals_max_min_on_interval_l535_535916


namespace driving_speed_l535_535991

variable (total_distance : ℝ) (break_time : ℝ) (total_trip_time : ℝ)

theorem driving_speed (h1 : total_distance = 480)
                      (h2 : break_time = 1)
                      (h3 : total_trip_time = 9) : 
  (total_distance / (total_trip_time - break_time)) = 60 :=
by
  sorry

end driving_speed_l535_535991


namespace johns_weekly_earnings_l535_535645

-- Define conditions
def days_off_per_week : ℕ := 3
def streaming_hours_per_day : ℕ := 4
def earnings_per_hour : ℕ := 10

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the question as a theorem
theorem johns_weekly_earnings :
  let days_streaming_per_week := days_in_week - days_off_per_week,
      weekly_streaming_hours := days_streaming_per_week * streaming_hours_per_day,
      weekly_earnings := weekly_streaming_hours * earnings_per_hour
  in weekly_earnings = 160 := by
  -- Proof is omitted with 'sorry'
  sorry

end johns_weekly_earnings_l535_535645


namespace count_non_squares_or_cubes_in_200_l535_535947

theorem count_non_squares_or_cubes_in_200 :
  let total_numbers := 200
  let count_perfect_squares := 14
  let count_perfect_cubes := 5
  let count_sixth_powers := 2
  total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers) = 183 :=
by
  let total_numbers := 200
  let count_perfect_squares := 14
  let count_perfect_cubes := 5
  let count_sixth_powers := 2
  have h1 : total_numbers = 200 := rfl
  have h2 : count_perfect_squares = 14 := rfl
  have h3 : count_perfect_cubes = 5 := rfl
  have h4 : count_sixth_powers = 2 := rfl
  show total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers) = 183
  calc
    total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers)
        = 200 - (14 + 5 - 2) : by rw [h1, h2, h3, h4]
    ... = 200 - 17 : by norm_num
    ... = 183 : by norm_num

end count_non_squares_or_cubes_in_200_l535_535947


namespace box_cost_is_550_l535_535353

noncomputable def cost_of_dryer_sheets (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                        (sheets_per_box : ℕ) (annual_savings : ℝ) : ℝ :=
  let sheets_per_week := loads_per_week * sheets_per_load
  let sheets_per_year := sheets_per_week * 52
  let boxes_per_year := sheets_per_year / sheets_per_box
  annual_savings / boxes_per_year

theorem box_cost_is_550 (h1 : 4 = 4)
                        (h2 : 1 = 1)
                        (h3 : 104 = 104)
                        (h4 : 11 = 11) :
  cost_of_dryer_sheets 4 1 104 11 = 5.50 :=
by
  sorry

end box_cost_is_550_l535_535353


namespace total_cookies_l535_535491

variables (Chris Kenny Glenn Dan Anne : ℕ)

-- Chris has one third as many cookies as Kenny
def condition1 : Prop := Chris = Kenny / 3
-- Glenn has four times as many cookies as Chris
def condition2 : Prop := Glenn = 4 * Chris
-- Glenn has 24 cookies
def condition3 : Prop := Glenn = 24
-- Dan has twice as many cookies as Chris and Kenny combined
def condition4 : Prop := Dan = 2 * (Chris + Kenny)
-- Anne has half as many cookies as Kenny
def condition5 : Prop := Anne = Kenny / 2

-- Prove that the total number of cookies is 105
theorem total_cookies (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) : 
  Chris + Kenny + Glenn + Dan + Anne = 105 := 
by 
  sorry

end total_cookies_l535_535491


namespace dart_landings_unique_lists_l535_535484

theorem dart_landings_unique_lists : 
  let num_darts := 5
  let num_boards := 4
    (∃ lists : List (List ℕ), 
      (∀ l ∈ lists, l.sorted (≥) ∧ list_sum l = num_darts ∧ length l = num_boards) 
        ∧ lists.length = 6)
  :=
sorry

end dart_landings_unique_lists_l535_535484


namespace log_a_2024_approx_6_l535_535108

def diamondsuit (a b : ℝ) : ℝ := a^(Real.log 5 b)
def heartsuit (a b : ℝ) : ℝ := a^(1 / Real.log 5 b)
def a : ℕ → ℝ
| 4 := heartsuit 4 3
| n := diamondsuit (heartsuit n (n - 1)) (a (n - 1))

theorem log_a_2024_approx_6 :
  Real.log 5 (a 2024) ≈ 6 := sorry

end log_a_2024_approx_6_l535_535108


namespace speed_excluding_stoppages_l535_535850

-- Conditions
def speed_with_stoppages := 33 -- kmph
def stoppage_time_per_hour := 16 -- minutes

-- Conversion of conditions to statements
def running_time_per_hour := 60 - stoppage_time_per_hour -- minutes
def running_time_in_hours := running_time_per_hour / 60 -- hours

-- Proof Statement
theorem speed_excluding_stoppages : 
  (speed_with_stoppages = 33) → (stoppage_time_per_hour = 16) → (75 = 33 / (44 / 60)) :=
by
  intros h1 h2
  sorry

end speed_excluding_stoppages_l535_535850


namespace triangle_problem_l535_535275

-- Given an acute triangle ABC with opposite sides a, b, and c,
-- and given √3a = 2c sin(A),
-- find C and the perimeter of the triangle given further conditions.
theorem triangle_problem
  (ABC : Type*)
  [triangle ABC]
  {a b c : ℝ}
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (acute : is_acute_triangle ABC)
  (opposite_sides : opposite_sides ABC a b c)
  (cond1 : √3 * a = 2 * c * sin (angle A))
  (cond2 : c = √7)
  (cond3 : a * b = 6) :
  (angle C = π / 3) ∧ ((a + b + c) = 5 + √7) := sorry

end triangle_problem_l535_535275


namespace equation_of_ellipse_equation_of_line_l535_535200

section
variables {a b: ℝ}
variables (A : ℝ × ℝ) (O : ℝ × ℝ) (e : ℝ) (slope_AF : ℝ) (area_OPQ : ℝ)

-- Conditions
def point_A := A = (0, -2)
def point_O := O = (0, 0)
def ellipse_eccentricity := e = (√3) / 2
def slope_of_AF := slope_AF = (2 * √3) / 3
def area_OPQ_eq_one := area_OPQ = 1

-- Declaration of the Ellipse and line equations
def ellipse_general := ∀ (a b : ℝ) (x y : ℝ),
  (a > 0 ∧ b > 0 ∧ a > b) →
  (e = c / a) →
  (c > 0) →
  (c = √3) →
  ((x^2) / a^2 + (y^2) / b^2 = 1)

-- Proving the equation of the ellipse
theorem equation_of_ellipse (h : ellipse_general a b 0 (-2)) : (a = 2) → (b = 1) → (∀ (x y : ℝ), (x^2) / 4 + y^2 = 1) :=
by
  sorry

-- Declaration of the line equation
def line_pq := ∀ (k : ℝ), y = k * x - 2

-- Proving the equation of line l
theorem equation_of_line (h : line_pq k) : (area_OPQ_eq_one) →
  (k = √7 / 2 ∨ k = -√7 / 2) →
  (∀ (x : ℝ), y = ((√7 / 2) * x) - 2 ∨ y = (-(√7 / 2) * x) - 2) :=
by
  sorry
end

end equation_of_ellipse_equation_of_line_l535_535200


namespace equal_degrees_l535_535297

variable {V : Type} [Fintype V]

structure Graph (V : Type) :=
  (adj : V → V → Prop)
  (sym : ∀ v w, adj v w = adj w v)
  (loopless : ∀ v, ¬ adj v v) -- Simple graph

variable (G : Graph V)

-- The conditions of the problem
variables (n : ℕ) (partition : Finset V → Prop)
          (cond : ∀ (V1 V2 : Finset V), V1.card = n → V2.card = n →
                    (∀ v, v ∉ V1 → v ∈ V2) →
                    (∑ v in V1, ∑ w in V1, if (G.adj v w) then 1 else 0 = ∑ v in V2, ∑ w in V2, if (G.adj v w) then 1 else 0))

-- Define the degree of a vertex
def degree (G : Graph V) (v : V) : ℕ :=
∑ w, if G.adj v w then 1 else 0

-- Main statement to prove
theorem equal_degrees (G : Graph V) [Fintype V] (h : 2 * n = Fintype.card V)
    (cond : ∀ (V1 V2 : Finset V), V1.card = n → V2.card = n →
             (∀ v, v ∉ V1 → v ∈ V2) →
            (∑ v in V1, ∑ w in V1, if G.adj v w then 1 else 0 = ∑ v in V2, ∑ w in V2, if G.adj v w then 1 else 0)) :
    (∀ v w : V, degree G v = degree G w) :=
sorry

end equal_degrees_l535_535297


namespace correct_statement_D_l535_535758

noncomputable theory

variables {R : Type*} [AddCommGroup R] [Module ℝ R]

-- Definitions
def is_unit_vector (v : R) : Prop := ∥v∥ = 1

theorem correct_statement_D (m n : ℝ) (a : R) 
  (hm : m ≠ 0) (hn : n ≠ 0) (ha : a ≠ 0) :
  (m + n) • a = m • a + n • a :=
by
  sorry

end correct_statement_D_l535_535758


namespace gear_rotation_possible_l535_535620

theorem gear_rotation_possible (n : ℕ) : 
  (n % 2 = 0) ↔ can_rotate(n) :=
sorry

-- Auxiliary definition that explains the gearing system
def can_rotate (n : ℕ) : Prop :=
  if n % 2 = 0 then
    True
  else
    False

end gear_rotation_possible_l535_535620


namespace num_valid_n_values_l535_535509

noncomputable def count_valid_factors (lower upper : ℕ) : ℕ :=
  (Finset.range (upper + 1)).filter (λ n : ℕ, 
    ∃ a b : ℤ, a + b = 3 ∧ a * b = -n).count

theorem num_valid_n_values :
  count_valid_factors 1 2000 = 2 := 
by sorry

end num_valid_n_values_l535_535509


namespace largest_band_members_l535_535797

theorem largest_band_members
  (p q m : ℕ)
  (h1 : p * q + 3 = m)
  (h2 : (q + 1) * (p + 2) = m)
  (h3 : m < 120) :
  m = 119 :=
sorry

end largest_band_members_l535_535797


namespace daily_profit_9080_l535_535467

theorem daily_profit_9080 (num_employees : Nat) (shirts_per_employee_per_day : Nat) (hours_per_shift : Nat) (wage_per_hour : Nat) (bonus_per_shirt : Nat) (shirt_sale_price : Nat) (nonemployee_expenses : Nat) :
  num_employees = 20 →
  shirts_per_employee_per_day = 20 →
  hours_per_shift = 8 →
  wage_per_hour = 12 →
  bonus_per_shirt = 5 →
  shirt_sale_price = 35 →
  nonemployee_expenses = 1000 →
  (num_employees * shirts_per_employee_per_day * shirt_sale_price) - ((num_employees * (hours_per_shift * wage_per_hour + shirts_per_employee_per_day * bonus_per_shirt)) + nonemployee_expenses) = 9080 := 
by
  intros
  sorry

end daily_profit_9080_l535_535467


namespace Mary_sheep_remaining_l535_535324

theorem Mary_sheep_remaining : 
  let initial_sheep := 1200 in
  let sheep_to_sister := initial_sheep / 4 in
  let sheep_after_sister := initial_sheep - sheep_to_sister in
  let sheep_to_brother := sheep_after_sister / 3 in
  let sheep_after_brother := sheep_after_sister - sheep_to_brother in
  let sheep_to_cousin := sheep_after_brother / 6 in
  let sheep_after_cousin := sheep_after_brother - sheep_to_cousin in
  sheep_after_cousin = 500 :=
by
  sorry

end Mary_sheep_remaining_l535_535324


namespace pieces_on_grid_l535_535967

theorem pieces_on_grid :
  ∃ (placements : Finset (Fin 4 → Fin 4)), 
  (∀ (f ∈ placements), 
    (∀ i j, i ≠ j → f i ≠ f j) ∧          -- each row and column unique constraint
    (∀ i j, i ≠ j → f i - i ≠ f j - j) ∧   -- one piece per diagonal, main diagonal constraint
    (∀ i j, i ≠ j → f i + i ≠ f j + j)) ∧  -- one piece per diagonal, anti-diagonal constraint
  placements.card = 8 := 
sorry

end pieces_on_grid_l535_535967


namespace other_carton_holds_one_racket_l535_535702

theorem other_carton_holds_one_racket :
  ∃ x : ℕ, (38 * 2 + 24 * x = 100) ↔ (x = 1) := 
by
  existsi 1
  rw [mul_add, mul_add]
  norm_num

end other_carton_holds_one_racket_l535_535702


namespace terminating_decimal_count_l535_535145

theorem terminating_decimal_count : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = 49 * k)}.card = 10 :=
by
  sorry

end terminating_decimal_count_l535_535145


namespace valid_fractions_l535_535508

theorem valid_fractions :
  ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ (1 ≤ z ∧ z ≤ 9) ∧
  (10 * x + y) % (10 * y + z) = 0 ∧ (10 * x + y) / (10 * y + z) = x / z :=
sorry

end valid_fractions_l535_535508


namespace ratio_FQ_HQ_l535_535743

-- Define the given conditions:
variables {FQ HQ : ℝ}
constant EQ : ℝ := 5
constant GQ : ℝ := 10

-- Using the Power of a Point theorem:
axiom power_of_a_point_theorem : EQ * FQ = GQ * HQ

-- State the theorem to prove:
theorem ratio_FQ_HQ : FQ / HQ = 2 := by
  sorry

end ratio_FQ_HQ_l535_535743


namespace number_of_lines_with_angle_greater_than_30_degrees_l535_535687

-- Definition of next integer points on the curve
def next_integer_points (x y : ℝ) : Prop :=
  ∃ (k : ℤ), x = k ∧ y = real.sqrt (9 - x^2)

-- Check if a point is on the curve
def on_curve (x y : ℝ) : Prop := y = real.sqrt (9 - x^2)

-- Calculate slope between two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  if p2.1 - p1.1 = 0 then 0 else (p2.2 - p1.2) / (p2.1 - p1.1)

-- Angles between points
def angle_greater_than_30_degrees (p1 p2 : ℝ × ℝ) : Prop :=
  abs (slope p1 p2) > real.tan (real.pi / 6)

theorem number_of_lines_with_angle_greater_than_30_degrees :
  ∃ (l : ℕ), l = 2 ∧
    ∀ (p1 p2 : ℝ × ℝ),
      (on_curve p1.1 p1.2) ∧ (on_curve p2.1 p2.2) ∧
      (int.floor p1.1 = p1.1) ∧ (int.floor p2.1 = p2.1) ∧
      (int.floor p1.2 = p1.2) ∧ (int.floor p2.2 = p2.2) →
      angle_greater_than_30_degrees p1 p2 :=
begin
  sorry
end

end number_of_lines_with_angle_greater_than_30_degrees_l535_535687


namespace leap_years_count_l535_535794

theorem leap_years_count : 
  let valid_years := {y : ℕ | y % 100 = 0 ∧ 1996 ≤ y ∧ y ≤ 4096 ∧ (y % 1000 = 300 ∨ y % 1000 = 700)} in
  valid_years.to_finset.card = 4 :=
by
  sorry

end leap_years_count_l535_535794


namespace find_m_l535_535555

theorem find_m (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) ∧ (m - 1 ≠ 0) → m = -1 :=
by
  sorry

end find_m_l535_535555


namespace to_shin_young_wins_both_games_l535_535427

/-
  Define the possible moves in rock-paper-scissors.
-/
inductive Move
| rock | paper | scissors

/-
  Define the result of a game given the moves of both players.
  True indicates Shin Young wins.
-/
def win (sy_move ja_move : Move) : Bool :=
  match sy_move, ja_move with
  | Move.rock, Move.scissors => true
  | Move.paper, Move.rock => true
  | Move.scissors, Move.paper => true
  | _, _ => false

/-
  Define the number of cases where Shin-Young wins both games.
-/
def shin_young_wins_twice : Nat :=
  let possible_moves := [Move.rock, Move.paper, Move.scissors]
  let win_cases := finset.product (finset.univ : finset Move) (finset.univ : finset Move)
  let valid_cases := finset.filter (fun (g1, g2) => win g1.1 g1.2 && win g2.1 g2.2) (finset.product win_cases win_cases)
  valid_cases.card

/-
  Theorem to prove the number of sequences where Shin-Young wins both times is 9.
-/
theorem shin_young_wins_both_games : shin_young_wins_twice = 9 := by
  sorry

end to_shin_young_wins_both_games_l535_535427


namespace max_f_eq_4_monotonic_increase_interval_l535_535933

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem max_f_eq_4 (x : ℝ) : ∀ x : ℝ, f x ≤ 4 := 
by
  sorry

theorem monotonic_increase_interval (k : ℤ) : ∀ x : ℝ, (k * Real.pi - Real.pi / 4 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4) ↔ 
  (0 ≤ Real.sin (2 * x) ∧ Real.sin (2 * x) ≤ 1) :=
by
  sorry

end max_f_eq_4_monotonic_increase_interval_l535_535933


namespace leah_spending_fraction_l535_535650

noncomputable def fraction_spent_on_milkshake (earnings remaining_in_wallet lost_money : ℕ) : ℚ :=
  earnings - 2 * remaining_in_wallet - lost_money

theorem leah_spending_fraction (f remaining_in_wallet lost_money : ℚ) (earnings : ℕ) :
  earnings = 28 → remaining_in_wallet = 1 → lost_money = 11 → (f * earnings = 28 - 12) →
  f = 1 / 7 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end leah_spending_fraction_l535_535650


namespace a2019_lt_half_lt_a2018_l535_535505

noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := a n - (a n) ^ 2 / 2019

theorem a2019_lt_half_lt_a2018 : a 2019 < 1 / 2 ∧ 1 / 2 < a 2018 :=
by sorry

end a2019_lt_half_lt_a2018_l535_535505


namespace closest_value_sqrt_diff_l535_535528

noncomputable def closest_value (x : ℝ) (y : ℝ) : ℝ :=
  let diff := Real.sqrt x - Real.sqrt y in
  if abs (diff - 0.152) <= abs (diff - 0.157) ∧ abs (diff - 0.152) <= abs (diff - 0.158) ∧ abs (diff - 0.152) <= abs (diff - 0.160) ∧ abs (diff - 0.152) <= abs (diff - 0.166) then 0.152
  else if abs (diff - 0.157) <= abs (diff - 0.158) ∧ abs (diff - 0.157) <= abs (diff - 0.160) ∧ abs (diff - 0.157) <= abs (diff - 0.166) then 0.157
  else if abs (diff - 0.158) <= abs (diff - 0.160) ∧ abs (diff - 0.158) <= abs (diff - 0.166) then 0.158
  else if abs (diff - 0.160) <= abs (diff - 0.166) then 0.160
  else 0.166

theorem closest_value_sqrt_diff :
  closest_value 91 88 = 0.158 := by
  sorry

end closest_value_sqrt_diff_l535_535528


namespace find_magnitude_of_b_l535_535935

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_magnitude_of_b (b : ℝ × ℝ) (h_angle : vector_dot (1, 1) b = -vector_magnitude b)
  (h_distance : vector_magnitude ((1, 1).1 - 2 * b.1, (1, 1).2 - 2 * b.2) = real.sqrt 10) :
  vector_magnitude b = 2 :=
sorry

end find_magnitude_of_b_l535_535935


namespace petya_wins_or_not_l535_535825

theorem petya_wins_or_not (n : ℕ) (h : n ≥ 3) : (n ≠ 4) ↔ petya_wins n :=
begin
  sorry
end

end petya_wins_or_not_l535_535825


namespace interval_of_increase_a_equal_3_range_of_a_decreasing_in_0_half_l535_535910

-- Define the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x + 1 - Real.log x

-- Statement for part (I)
theorem interval_of_increase_a_equal_3 :
  let f' (x : ℝ) : ℝ := -2 * x + 3 - (1 / x)
  (∀ x, f' x > 0 → (x ∈ Ioo (1/2 : ℝ) 1)) :=
sorry

-- Statement for part (II)
theorem range_of_a_decreasing_in_0_half :
  let f' (x : ℝ) (a : ℝ) : ℝ := -2 * x + a - (1 / x)
  (∀ x ∈ Ioo (0 : ℝ) (1/2 : ℝ), f' x ≤ 0) → (∀ a, a ≤ 3) :=
sorry

end interval_of_increase_a_equal_3_range_of_a_decreasing_in_0_half_l535_535910


namespace part_a_part_b_l535_535511

-- Define polynomials and conditions
def poly1 (p q r : ℤ) := (x^3 - x^2 - 6*x + 2) * (x^3 + p * x^2 + q * x + r)
def poly2 (c d e f p q r s : ℤ) := (x^4 + c * x^3 + d * x^2 + e * x + f) * (x^4 + p * x^3 + q * x^2 + r * x + s)

-- Part (a): Prove the coefficients p = 1, q = -6, r = -2 lead to no odd powers of x
theorem part_a (p q r : ℤ) : (poly1 p q r).coeff 1 = 0 → (poly1 p q r).coeff 3 = 0 → (poly1 p q r).coeff 5 = 0 → (p = 1) ∧ (q = -6) ∧ (r = -2) :=
by sorry

-- Part (b): Prove the coefficients p = -c, q = d, r = -e, s = f given K ≠ 0
theorem part_b (c d e f p q r s : ℤ) (h : c^2 * f - c * d * e + e^2 ≠ 0) :
  (poly2 c d e f p q r s).coeff 1 = 0 → (poly2 c d e f p q r s).coeff 3 = 0 → (poly2 c d e f p q r s).coeff 5 = 0 →
  (poly2 c d e f p q r s).coeff 7 = 0 → (p = -c) ∧ (q = d) ∧ (r = -e) ∧ (s = f) :=
by sorry

end part_a_part_b_l535_535511


namespace total_volume_of_all_cubes_l535_535490

def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume (count : ℕ) (side_length : ℕ) : ℕ := count * (cube_volume side_length)

theorem total_volume_of_all_cubes :
  total_volume 4 3 + total_volume 3 4 = 300 :=
by
  sorry

end total_volume_of_all_cubes_l535_535490


namespace simplify_expression_l535_535953

theorem simplify_expression (θ : ℝ) (h : θ ∈ Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  sqrt (1 - sin (2 * θ)) - sqrt (1 + sin (2 * θ)) = 2 * sin θ :=
sorry

end simplify_expression_l535_535953


namespace max_abc_value_l535_535309

theorem max_abc_value (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_equation : a * b + c = (a + c) * (b + c))
  (h_sum : a + b + c = 2) : abc ≤ 1/27 :=
by sorry

end max_abc_value_l535_535309


namespace minimum_sum_of_distances_l535_535198

-- Definitions for the lines and the parabola
def l1 (P : ℝ × ℝ) : Prop := 4 * P.1 - 3 * P.2 + 6 = 0
def l2 (P : ℝ × ℝ) : Prop := P.1 = -1
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

-- Statement of the proof problem
theorem minimum_sum_of_distances : 
  let d (P : ℝ × ℝ) := dist P (1, 0) + if l2 P then 0 else abs (P.1 + 1) in
  ∃ P : ℝ × ℝ, parabola P ∧ d P = 2 :=
sorry

end minimum_sum_of_distances_l535_535198


namespace least_value_x_y_z_l535_535444

theorem least_value_x_y_z (x y z : ℕ) (hx : x = 4 * y) (hy : y = 7 * z) (hz : 0 < z) : x - y - z = 19 :=
by
  -- placeholder for actual proof
  sorry

end least_value_x_y_z_l535_535444


namespace cost_of_calf_l535_535464

theorem cost_of_calf (C : ℝ) (total_cost : ℝ) (cow_to_calf_ratio : ℝ) :
  total_cost = 990 ∧ cow_to_calf_ratio = 8 ∧ total_cost = C + 8 * C → C = 110 := by
  sorry

end cost_of_calf_l535_535464


namespace least_number_divisible_l535_535020

theorem least_number_divisible (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 9 = 4) (h3 : n % 18 = 4) : n = 130 := sorry

end least_number_divisible_l535_535020


namespace meeting_point_2015_is_C_l535_535169

-- Given definitions based on conditions
variable (x y : ℝ) -- Speeds of the motorcycle and the cyclist
variable (A B C D : Point) -- Points on segment AB
variable (meetings : ℕ → Point) -- Function representing the meeting point sequence

-- Conditions stating the alternating meeting pattern
axiom start_at_A (n : ℕ) : meetings (2 * n + 1) = C
axiom start_at_B (n : ℕ) : meetings (2 * n + 2) = D

-- The theorem statement to be proved
theorem meeting_point_2015_is_C : meetings 2015 = C := sorry

end meeting_point_2015_is_C_l535_535169


namespace comparison_of_M_and_N_l535_535190

theorem comparison_of_M_and_N (a b t : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : t > 0) :
  let M := a / b in
  let N := (a + t) / (b + t) in
  M > N :=
by
  let M := a / b
  let N := (a + t) / (b + t)
  have hMN : M - N = t * (a - b) / (b * (b + t)) by sorry
  sorry

end comparison_of_M_and_N_l535_535190


namespace min_value_correct_max_value_correct_l535_535982

noncomputable def min_value_expr : ℕ :=
  ∑ n in finset.range 50, (n + 1) * (101 - (n + 1))

noncomputable def max_value_expr : ℕ :=
  ∑ n in finset.range 50, (2 * n + 1) * (2 * n + 2)

theorem min_value_correct : min_value_expr = 85850 := by
  sorry

theorem max_value_correct : max_value_expr = 169150 := by
  sorry

end min_value_correct_max_value_correct_l535_535982


namespace regression_lines_common_point_l535_535400

variables {α : Type*} [linear_ordered_field α]

/-- 
  Two students A and B conducted 10 and 15 experiments, respectively. 
  Both used the least squares method to obtain regression lines l₁ and l₂. 
  The average observed value of x is s and the average observed value of y is t.
  We need to prove that point (s, t) is common to both regression lines l₁ and l₂.
-/

variables 
  {l1 l2 : α → α} -- regression lines l1 and l2
  {s t : α}       -- average of x and y respectively

/-- The point (s, t) lies on both regression lines l₁ and l₂. -/
theorem regression_lines_common_point 
  (h1 : l1 s = t) (h2 : l2 s = t) : 
  (∃ point : α × α, point = (s, t) ∧ l1 point.fst = point.snd ∧ l2 point.fst = point.snd) :=
by {
  use (s, t),
  split; try { refl },
  split;
  { assumption }
}

end regression_lines_common_point_l535_535400


namespace total_cost_750_candies_l535_535456

def candy_cost (candies : ℕ) (cost_per_box : ℕ) (candies_per_box : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
  let boxes := candies / candies_per_box
  let total_cost := boxes * cost_per_box
  if candies > discount_threshold then
    (1 - discount_rate) * total_cost
  else
    total_cost

theorem total_cost_750_candies :
  candy_cost 750 8 30 500 0.1 = 180 :=
by sorry

end total_cost_750_candies_l535_535456


namespace polynomial_roots_abs_sum_l535_535651

theorem polynomial_roots_abs_sum (a b c : ℝ) (w : ℂ)
  (h₁ : ∀ z : ℂ, (z = w + 3*complex.I ∨ z = w + 9*complex.I ∨ z = 2*w - 4) →
    (z = 0 → False)) :
  let P := λ z : ℂ, z^3 + a*z^2 + b*z + c in
  |a + b + c| = 136 :=
by sorry

end polynomial_roots_abs_sum_l535_535651


namespace smallest_x_for_three_digit_product_l535_535377

theorem smallest_x_for_three_digit_product : ∃ x : ℕ, (27 * x >= 100) ∧ (∀ y < x, 27 * y < 100) :=
by
  sorry

end smallest_x_for_three_digit_product_l535_535377


namespace balance_difference_calculation_l535_535821

-- Define the compound interest formula for the given conditions
noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r) ^ t

-- Define the given conditions
def angela_initial_amount : ℝ := 9000
def angela_interest_rate : ℝ := 0.055
def bob_initial_amount : ℝ := 11000
def bob_interest_rate : ℝ := 0.065
def years : ℝ := 25

-- Define the final balances
def angela_final_balance : ℝ := compound_interest angela_initial_amount angela_interest_rate years
def bob_final_balance : ℝ := compound_interest bob_initial_amount bob_interest_rate years

-- Define the positive difference between their balances
def balance_difference : ℝ := abs (bob_final_balance - angela_final_balance)

-- Theorem statement
theorem balance_difference_calculation : balance_difference = 13486 :=
by sorry

end balance_difference_calculation_l535_535821


namespace triangle_center_distance_l535_535051

noncomputable def isosceles_triangle (a : ℝ) := {α β : ℝ // α = β ∧ α + β + 45 = 180}
noncomputable def circumscribed_circle (R : ℝ) := {O : ℝ}
noncomputable def inscribed_circle (r: ℝ) (d_VX : ℝ) := {X : ℝ // d_VX = 4}
noncomputable def incenter_distance (d_XI : ℝ) := {I : ℝ}

theorem triangle_center_distance :
  ∃ (d_XI : ℝ), 
    ∀ (α β : ℝ) (a : ℝ) (R r d_VX : ℝ), 
      isosceles_triangle (a) →
      circumscribed_circle (R) →
      inscribed_circle (r) (d_VX) →
      incenter_distance (d_XI) →
      d_XI = ? :=
by 
  sorry

end triangle_center_distance_l535_535051


namespace total_animals_correct_l535_535001

variables (num_legs total_animals : ℕ) (num_cows : ℕ := 6) (num_cow_legs_per_cow : ℕ := 4) (num_duck_legs_per_duck : ℕ := 2)

-- Define the total number of legs
def total_legs := 42

-- Define the number of cow legs
def cow_legs := num_cows * num_cow_legs_per_cow

-- Define the remaining legs (which are duck legs)
def duck_legs := total_legs - cow_legs

-- Define the number of ducks
def num_ducks := duck_legs / num_duck_legs_per_duck

-- Define the total number of animals (cows + ducks)
def total_animals := num_cows + num_ducks

theorem total_animals_correct : total_animals = 15 := by
  sorry

end total_animals_correct_l535_535001


namespace jane_bear_production_increase_l535_535294

theorem jane_bear_production_increase
  (B H : ℕ) -- B = bears per week, H = hours per week without assistant
  (P : ℕ) -- P = percentage increase in bears per week when she works with an assistant
  (h1 : Jane makes some percentage more bears per week with an assistant)
  (h2 : Jane works 10 percent fewer hours each week with an assistant)
  (h3 : Having an assistant increases Jane's output of toy bears per hour by 100 percent) :
  P = 80 :=
by
  sorry

end jane_bear_production_increase_l535_535294


namespace monotonic_intervals_extreme_values_on_interval_l535_535922

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

theorem monotonic_intervals:
  (∀ x : ℝ, x < -1 → 0 < (f' x)) ∧ 
  (∀ x : ℝ, x > 1 → 0 < (f' x)) ∧ 
  (∀ x : ℝ, -1 < x ∧ x < 1 → (f' x) < 0) := 
by {
  sorry
}

theorem extreme_values_on_interval : 
  (∃ x_min x_max, -3 ≤ x_min ∧ x_min ≤ 3 ∧ -3 ≤ x_max ∧ x_max ≤ 3 ∧ 
   f x_min = -49 ∧ f x_max = 59) :=
by {
  exists 3, -3,
  sorry
}

end monotonic_intervals_extreme_values_on_interval_l535_535922


namespace length_of_nylon_cord_l535_535459

-- Definitions based on the conditions
def tree : ℝ := 0 -- Tree as the center point (assuming a 0 for simplicity)
def distance_ran : ℝ := 30 -- Dog ran approximately 30 feet

-- The theorem to prove
theorem length_of_nylon_cord : (distance_ran / 2) = 15 := by
  -- Assuming the dog ran along the diameter of the circle
  -- and the length of the cord is the radius of that circle.
  sorry

end length_of_nylon_cord_l535_535459


namespace no_integers_negative_l535_535955

theorem no_integers_negative (a b c d : ℤ) (h : 5 ^ a + 5 ^ b = 2 ^ c + 2 ^ d + 17) : 
  (a < 0) → (b < 0) → (c < 0) → (d < 0) → false :=
sorry

end no_integers_negative_l535_535955


namespace integer_solutions_of_inequality_l535_535368

theorem integer_solutions_of_inequality (x : ℤ) : 
  (-4 < 1 - 3 * (x: ℤ) ∧ 1 - 3 * (x: ℤ) ≤ 4) ↔ (x = -1 ∨ x = 0 ∨ x = 1) := 
by 
  sorry

end integer_solutions_of_inequality_l535_535368


namespace correct_rotation_identifies_C_l535_535622

def arrow : Type := string

structure Square :=
  (top_left : arrow)
  (top_right : arrow)
  (bottom_left : arrow)
  (bottom_right : arrow)

def rotate_90_clockwise (s : Square) : Square :=
  { top_left := s.bottom_left,
    top_right := s.top_left,
    bottom_left := s.bottom_right,
    bottom_right := s.top_right }

def original_config := Square.mk "Up" "Right" "Down" "Left"

def rotated_config := Square.mk "Right" "Down" "Left" "Up"

theorem correct_rotation_identifies_C :
  rotate_90_clockwise original_config = rotated_config :=
  sorry

end correct_rotation_identifies_C_l535_535622


namespace average_of_consecutive_integers_l535_535536

theorem average_of_consecutive_integers (a b : ℤ) (h1 : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) :
  let b := a + 2 in
  let avg_of_b := ((a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 5 in
  avg_of_b = a + 4 :=
by
  sorry

end average_of_consecutive_integers_l535_535536


namespace find_value_of_X_l535_535958

theorem find_value_of_X :
  let X_initial := 5
  let S_initial := 0
  let X_increment := 3
  let target_sum := 15000
  let X := X_initial + X_increment * 56
  2 * target_sum ≥ 3 * 57 * 57 + 7 * 57 →
  X = 173 :=
by
  sorry

end find_value_of_X_l535_535958


namespace smallest_term_seq_a_l535_535285

open Real

def seq_a (n : ℕ) : ℝ := n^2 - 9 * n - 100

theorem smallest_term_seq_a :
  ∃ n, n ∈ {4, 5} ∧ (∀ m ∈ ({1, 2, 3, 4, 5, ..., 100}), seq_a n ≤ seq_a m) :=
sorry

end smallest_term_seq_a_l535_535285


namespace Z_4_3_eq_37_l535_535838

def Z (a b : ℕ) : ℕ :=
  a^2 + a * b + b^2

theorem Z_4_3_eq_37 : Z 4 3 = 37 :=
  by
    sorry

end Z_4_3_eq_37_l535_535838


namespace square_projection_exists_l535_535733

structure Point :=
(x y : Real)

structure Line :=
(a b c : Real) -- Line equation ax + by + c = 0

def is_on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

theorem square_projection_exists (P : Point) (l : Line) :
  ∃ (A B C D : Point), 
  is_on_line A l ∧ 
  is_on_line B l ∧
  (A.x + B.x) / 2 = P.x ∧ 
  (A.y + B.y) / 2 = P.y ∧ 
  (A.x = B.x ∨ A.y = B.y) ∧ -- assuming one of the sides lies along the line
  (C.x + D.x) / 2 = P.x ∧ 
  (C.y + D.y) / 2 = P.y ∧ 
  C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B :=
sorry

end square_projection_exists_l535_535733


namespace probability_of_two_red_two_green_l535_535455

def red_balls : ℕ := 10
def green_balls : ℕ := 8
def total_balls : ℕ := red_balls + green_balls
def drawn_balls : ℕ := 4

def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def prob_two_red_two_green : ℚ :=
  (combination red_balls 2 * combination green_balls 2 : ℚ) / combination total_balls drawn_balls

theorem probability_of_two_red_two_green :
  prob_two_red_two_green = 7 / 17 := 
sorry

end probability_of_two_red_two_green_l535_535455


namespace line_c_intersects_one_of_a_or_b_l535_535963

-- Definitions of the entities
variables (a b c : line) (α β : plane)
def skew_lines (a b : line) : Prop := ¬ ∃ p, p ∈ a ∧ p ∈ b
def line_in_plane (a : line) (α : plane) : Prop := ∀ p, p ∈ a → p ∈ α
def plane_intersection_line (α β : plane) (c : line) : Prop := ∀ p, p ∈ c ↔ p ∈ α ∧ p ∈ β

-- Given conditions in the problem
variable (h₁ : skew_lines a b)
variable (h₂ : line_in_plane a α)
variable (h₃ : line_in_plane b β)
variable (h₄ : plane_intersection_line α β c)

-- Theorem to be proved
theorem line_c_intersects_one_of_a_or_b (h₁ : skew_lines a b) 
                                        (h₂ : line_in_plane a α) 
                                        (h₃ : line_in_plane b β) 
                                        (h₄ : plane_intersection_line α β c) : 
                                        (∃ p, p ∈ c ∧ p ∈ a) ∨ (∃ p, p ∈ c ∧ p ∈ b) :=
sorry

end line_c_intersects_one_of_a_or_b_l535_535963


namespace permutations_behind_Alice_l535_535712

theorem permutations_behind_Alice (n : ℕ) (h : n = 7) : 
  (Nat.factorial n) = 5040 :=
by
  rw [h]
  rw [Nat.factorial]
  sorry

end permutations_behind_Alice_l535_535712


namespace tan_twopi_minus_alpha_l535_535185

-- An auxiliary lemma to use inside the main statement
lemma log8_inverse (x : ℝ) : real.log x / real.log 8 = log x / log 8 :=
by sorry

-- The main theorem statement
theorem tan_twopi_minus_alpha (alpha : ℝ) 
  (h₀ : sin (π - alpha) = real.log (1 / 4) / real.log 8)
  (h₁ : alpha ∈ set.Ioo (-π / 2) 0) : 
  real.tan (2 * π - alpha) = 2 * real.sqrt 5 / 5 := 
by sorry

end tan_twopi_minus_alpha_l535_535185


namespace volume_of_cube_l535_535392

-- Define the conditions
def surface_area (a : ℝ) : ℝ := 6 * a^2
def side_length (a : ℝ) (SA : ℝ) : Prop := SA = 6 * a^2
def volume (a : ℝ) : ℝ := a^3

-- State the theorem
theorem volume_of_cube (a : ℝ) (SA : surface_area a = 150) : volume a = 125 := 
sorry

end volume_of_cube_l535_535392


namespace circus_accommodation_l535_535735

theorem circus_accommodation : 246 * 4 = 984 := by
  sorry

end circus_accommodation_l535_535735


namespace midpoints_parallel_l535_535765

theorem midpoints_parallel (A B C L M N: Point)
  (h₁: SimilarTriangle ABC LMN)
  (h₂: AC = BC)
  (h₃: LN = MN)
  (h₄: AL = BM) :
  let D := midpoint A B
  let E := midpoint L M
  parallel (lineSegment D E) (lineSegment C N) := 
sorry

end midpoints_parallel_l535_535765


namespace total_water_in_bucket_l535_535783

noncomputable def initial_gallons : ℝ := 3
noncomputable def added_gallons_1 : ℝ := 6.8
noncomputable def liters_to_gallons (liters : ℝ) : ℝ := liters / 3.78541
noncomputable def quart_to_gallons (quarts : ℝ) : ℝ := quarts / 4
noncomputable def added_gallons_2 : ℝ := liters_to_gallons 10
noncomputable def added_gallons_3 : ℝ := quart_to_gallons 4

noncomputable def total_gallons : ℝ :=
  initial_gallons + added_gallons_1 + added_gallons_2 + added_gallons_3

theorem total_water_in_bucket :
  abs (total_gallons - 13.44) < 0.01 :=
by
  -- convert amounts and perform arithmetic operations
  sorry

end total_water_in_bucket_l535_535783


namespace least_prime_triangle_angle_l535_535623

-- Definitions from conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∈ finset.range (n - 2 + 1), m + 2 = n ∨ n % (m + 2) ≠ 0

variable (a b c : ℕ)

-- The Lean statement to prove the main condition
theorem least_prime_triangle_angle (h_prime_a : is_prime a) (h_prime_b : is_prime b) (h_prime_c : is_prime c)
    (h_sum_angles : a + b + c = 180) (h_order : a > b ∧ b > c) : c = 3 :=
  sorry

end least_prime_triangle_angle_l535_535623


namespace number_of_street_trees_l535_535761

-- Definitions from conditions
def road_length : ℕ := 1500
def interval_distance : ℕ := 25

-- The statement to prove
theorem number_of_street_trees : (road_length / interval_distance) + 1 = 61 := 
by
  unfold road_length
  unfold interval_distance
  sorry

end number_of_street_trees_l535_535761


namespace cone_height_l535_535788

-- Define the volume of the cone
def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Define the specific case where the volume is provided
def cone_spec_volume : ℝ := 16384 * π

-- Define the vertex angle condition leading to h = r
def is_right_isosceles (h r : ℝ) : Prop := h = r

-- Prove that the height h equals the cube root of 49152
theorem cone_height (r h : ℝ) (h_eq_r : is_right_isosceles h r) (vol_eq : cone_volume r h = cone_spec_volume) : h = Real.cbrt 49152 :=
by
  sorry

end cone_height_l535_535788


namespace sin_70_eq_1_minus_2k_squared_l535_535188

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * k^2 :=
by
  sorry

end sin_70_eq_1_minus_2k_squared_l535_535188


namespace find_p_l535_535679

noncomputable def f : ℂ := 10
noncomputable def w : ℂ := -10 + 250 * Complex.i
noncomputable def p : ℂ := 799 + 25 * Complex.i

theorem find_p : f * p - w = 8000 := 
by
  sorry

end find_p_l535_535679


namespace number_of_years_passed_l535_535849

theorem number_of_years_passed (A F : ℝ) (r n : ℝ)
  (hA : A = 64000) 
  (hF : F = 87111.11111111112)
  (hr : r = 7/6) 
  (h_growth : F = A * r^n) : n ≈ 2 :=
by 
  rw [hA, hF, hr] at h_growth
  have h : (7 / 6 : ℝ) ≠ 0 := by norm_num
  rw ←div_eq_iff_eq_mul (-1) h_growth
  exact sorry

end number_of_years_passed_l535_535849


namespace ratio_BP_PE_l535_535985

noncomputable def triangle_ABC (A B C D E P : Point) (AB AC BC : ℝ) : Prop :=
AB = 8 ∧ AC = 6 ∧ BC = 4 ∧
AngleBisector A B D ∧
AngleBisector B A E ∧
Intersects A B P ∧
Intersects B E P

theorem ratio_BP_PE (A B C D E P : Point)
  (h : triangle_ABC A B C D E P) : ratio BP PE = 4 :=
sorry

end ratio_BP_PE_l535_535985


namespace quadrilateral_circumcircles_concurrent_l535_535090

theorem quadrilateral_circumcircles_concurrent
  (A B C D E F P : Type)
  [quadrilateral ABCD]
  (intersect_ext_sides : intersect_extensions_opposite_sides ABCD E F)
  (circumcircle_ABF : on_circumcircle P A B F)
  (circumcircle_ADE : on_circumcircle P A D E)
  (circumcircle_BCE : on_circumcircle P B C E)
  (circumcircle_CDF : on_circumcircle P C D F) :
  on_circumcircle P A B F ∧
  on_circumcircle P A D E ∧
  on_circumcircle P B C E ∧
  on_circumcircle P C D F :=
by
  sorry

end quadrilateral_circumcircles_concurrent_l535_535090


namespace prism_volume_l535_535069

open Real

theorem prism_volume :
  ∃ (a b c : ℝ), a * b = 15 ∧ b * c = 10 ∧ c * a = 30 ∧ a * b * c = 30 * sqrt 5 :=
by
  sorry

end prism_volume_l535_535069


namespace correct_statement_B_l535_535796

variable (NaturalSelection :: directed_changes_gene_frequency : Prop)
variable (ChangesInGeneFrequency :: leads_to_evolution : Prop)
variable (IndividualsEvolution :: population_evolves : Prop)
variable (GeographicalIsolation :: leads_to_reproductive_isolation : Prop)

theorem correct_statement_B : ChangesInGeneFrequency =
  leads_to_evolution → True := by sorry

end correct_statement_B_l535_535796


namespace total_people_in_class_l535_535445

-- Define the number of people based on their interests
def likes_both: Nat := 5
def only_baseball: Nat := 2
def only_football: Nat := 3
def likes_neither: Nat := 6

-- Define the total number of people in the class
def total_people := likes_both + only_baseball + only_football + likes_neither

-- Theorem statement
theorem total_people_in_class : total_people = 16 :=
by
  -- Proof is skipped
  sorry

end total_people_in_class_l535_535445


namespace distance_geologists_probability_l535_535280

theorem distance_geologists_probability :
  let speed := 4 -- km/h
  let n_roads := 6
  let travel_time := 1 -- hour
  let distance_traveled := speed * travel_time -- km
  let distance_threshold := 6 -- km
  let n_outcomes := n_roads * n_roads
  let favorable_outcomes := 18 -- determined from the solution steps
  let probability := favorable_outcomes / n_outcomes
  probability = 0.5 := by
  sorry

end distance_geologists_probability_l535_535280


namespace find_quadruples_l535_535127

open Nat

theorem find_quadruples (a b p n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
    (h : a^3 + b^3 = p^n) :
    (∃ k, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3 * k + 1) ∨
    (∃ k, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3 * k + 2) ∨
    (∃ k, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3 * k + 2) :=
sorry

end find_quadruples_l535_535127


namespace cone_to_sphere_altitude_ratio_l535_535068

def volume_sphere (r : ℝ) := (4 / 3) * π * r^3
def volume_cone (r h : ℝ) := (1 / 3) * π * r^2 * h

theorem cone_to_sphere_altitude_ratio (r h : ℝ) (hvol : volume_cone r h = (1 / 3) * volume_sphere r) :
  h / r = 4 / 3 :=
by
  sorry

end cone_to_sphere_altitude_ratio_l535_535068


namespace dodecagon_diagonals_intersect_probability_l535_535775

theorem dodecagon_diagonals_intersect_probability :
  ∀ (dodecagon : Type) [regular_polygon dodecagon (12 : nat)],
  let diagonals_count := 54 in
  let intersecting_diagonals_count := 495 in
  (intersecting_diagonals_count : ℚ) / (binom diagonals_count 2) = 15 / 43 :=
by
  intros
  have diagonals : 54 := 54
  have intersections : 495 := 495
  have total_pairs_diagonals : 1431 := binom 54 2
  have probability : ℚ := intersections / total_pairs_diagonals
  rw [Q.to_rat_eq]
  apply Q.eq_of_rat_eq
  norm_num
  exact 15 / 43
  sorry

end dodecagon_diagonals_intersect_probability_l535_535775


namespace meeting_point_2015_is_C_l535_535172

-- Given definitions based on conditions
variable (x y : ℝ) -- Speeds of the motorcycle and the cyclist
variable (A B C D : Point) -- Points on segment AB
variable (meetings : ℕ → Point) -- Function representing the meeting point sequence

-- Conditions stating the alternating meeting pattern
axiom start_at_A (n : ℕ) : meetings (2 * n + 1) = C
axiom start_at_B (n : ℕ) : meetings (2 * n + 2) = D

-- The theorem statement to be proved
theorem meeting_point_2015_is_C : meetings 2015 = C := sorry

end meeting_point_2015_is_C_l535_535172


namespace area_PQRS_l535_535695

open_locale classical
open_locale real

variables {XYZ : Type*} {PQRS : Type*} (X Y Z P Q R S : XYZ)

-- Geometrical properties and given conditions
variables [geometry XYZ]
variables (h_triangle : is_triangle X Y Z)
variables (h_rectangle : is_rectangle P Q R S)
variables (PQ_on_XZ : on_Line PQ XZ)
variables (altitude_Y_to_XZ : altitude Y XZ = 8)
variables (length_XZ : length XZ = 12)
variables (PQ_eq_one_third_PR : length PQ = (1 / 3) * length PR)

-- Statement to prove
theorem area_PQRS :
  area P Q R S = 6912 / 121 :=
sorry

end area_PQRS_l535_535695


namespace number_of_4_digit_palindromes_l535_535451

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_valid_4_digit_palindrome (n : ℕ) : Prop :=
  is_palindrome n ∧ is_4_digit n

theorem number_of_4_digit_palindromes : 
  {n : ℕ | is_valid_4_digit_palindrome n}.to_finset.card = 90 :=
by
  sorry

end number_of_4_digit_palindromes_l535_535451


namespace sin_negative_if_third_or_fourth_quadrant_l535_535034

theorem sin_negative_if_third_or_fourth_quadrant (α : ℝ) :
  (sin α < 0) ↔ (∃ n : ℤ, π < α + 2 * n * π ∧ α + 2 * n * π < 2 * π) ∨ (3 * π < α + 2 * n * π ∧ α + 2 * n * π < 4 * π) :=
by
  sorry

end sin_negative_if_third_or_fourth_quadrant_l535_535034


namespace count_negative_numbers_l535_535261

theorem count_negative_numbers : 
  let expr1 := -2^2
  let expr2 := (-2)^2
  let expr3 := -(-2)
  let expr4 := -|-2|
  1 ≤ expr1 ∧ expr1 < 0 ∧
  0 ≤ expr2 ∧ 0 < expr2 ∧ 
  0 ≤ expr3 ∧ 0 < expr3 ∧ 
  1 ≤ expr4 ∧ expr4 < 0 ∧
  (expr1 < 0) + (expr2 < 0) + (expr3 < 0) + (expr4 < 0) = 2 := 
by
  sorry

end count_negative_numbers_l535_535261


namespace vasya_max_sum_l535_535016

noncomputable def maxSumSequence : ℕ :=
  1 + 2 + 3 + 4 + 6 + 8 + 11 + 13 + 14 + 17

theorem vasya_max_sum : maxSumSequence = 165 :=
by
  -- We need to verify the sequence:
  -- 1, 2, 3, 4, 6, 8, 11, 13, 14, 17
  -- satisfies all conditions:
  -- 1. Each number must be greater than all previously named numbers.
  -- 2. Cannot be the sum of any two previously named numbers.
  -- And the sum is equal to 165.
  let sequence := [1, 2, 3, 4, 6, 8, 11, 13, 14, 17]
  have h1 : ∀ i j, i < j → sequence[i] < sequence[j], from sorry  -- By sequence construction
  have h2 : ∀ i j k, i < j ∧ j < k → sequence[k] ≠ sequence[i] + sequence[j], from sorry -- By checking each pair
  have h_sum : List.sum sequence = 165, from rfl -- The computed sum matches
  sorry

end vasya_max_sum_l535_535016


namespace values_of_x_solve_ggx_eq_7_l535_535252

noncomputable def g (x : ℝ) : ℝ :=
if x ≥ -3 then x^2 - 9 else x + 4

theorem values_of_x_solve_ggx_eq_7 : 
  {x : ℝ | g (g x) = 7}.finite.to_finset.card = 4 :=
by
  sorry

end values_of_x_solve_ggx_eq_7_l535_535252


namespace smallest_period_pi_max_value_min_value_l535_535581

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

open Real

theorem smallest_period_pi : ∀ x, f (x + π) = f x := by
  unfold f
  intros
  sorry

theorem max_value : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≤ 1 + sqrt 2 := by
  unfold f
  intros
  sorry

theorem min_value : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≥ 0 := by
  unfold f
  intros
  sorry

end smallest_period_pi_max_value_min_value_l535_535581


namespace correct_statements_l535_535928

-- Define the geometric series
def geometric_series (a r : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := geometric_series n a r * r

-- Prove the conditions associated with the given geometric series
theorem correct_statements (a r : ℝ) (h_a : a = 1) (h_r : r = 1 / 3) :
  let S := ∑' n, geometric_series a r n in
  -- Statement 1: The sum increases without limit
  ¬ (tendsto S at_top at_top) ∧ 
  -- Statement 2: The sum is exactly 2
  ¬ (S = 2) ∧
  -- Statement 3: The difference between any term of the sequence and zero can be made less than any positive quantity no matter how small
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |geometric_series a r n - 0| < ε) ∧
  -- Statement 4: The difference between the sum and 3/2 can be made less than any positive quantity no matter how small
  (∀ ε > 0, |S - (3/2)| < ε) ∧
  -- Statement 5: The sum approaches a limit
  (∃ l : ℝ, tendsto S at_top (𝓝 l)) :=
by
  sorry

end correct_statements_l535_535928


namespace committee_selection_l535_535544

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_selection :
  let total_combinations := binomial 14 4
  let restricted_combinations := binomial 12 2
  let valid_combinations := total_combinations - restricted_combinations
  valid_combinations = 935 :=
by
  let total_combinations := binomial 14 4
  let restricted_combinations := binomial 12 2
  let valid_combinations := total_combinations - restricted_combinations
  exact Eq.refl 935
  sorry

end committee_selection_l535_535544


namespace neg_p_implies_neg_q_l535_535887

variables {x : ℝ}

def condition_p (x : ℝ) : Prop := |x + 1| > 2
def condition_q (x : ℝ) : Prop := 5 * x - 6 > x^2
def neg_p (x : ℝ) : Prop := |x + 1| ≤ 2
def neg_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

theorem neg_p_implies_neg_q : (∀ x, neg_p x → neg_q x) :=
by 
  -- Proof is skipped according to the instructions
  sorry

end neg_p_implies_neg_q_l535_535887


namespace determine_m_range_l535_535549

-- Define propositions P and Q
def P (t : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) = 1)
def Q (t m : ℝ) : Prop := 1 - m < t ∧ t < 1 + m ∧ m > 0

-- Define negation of propositions
def notP (t : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) ≠ 1)
def notQ (t m : ℝ) : Prop := ¬ (1 - m < t ∧ t < 1 + m)

-- Main problem: Determine the range of m where notP -> notQ is a sufficient but not necessary condition
theorem determine_m_range {m : ℝ} : (∃ t : ℝ, notP t → notQ t m) ↔ (0 < m ∧ m ≤ 3) := by
  sorry

end determine_m_range_l535_535549


namespace angle_A_l535_535884

open Real

noncomputable def equilateral_triangle_side_length : ℝ := 1
noncomputable def PA_perpendicular_distance : ℝ := sqrt 6 / 4

theorem angle_A'C_AB_eq_90_degrees (A B C P A' D O : Point)
  (h_eq_triangle : equilateral_triangle A B C equilateral_triangle_side_length)
  (h_PA_perpendicular : Perpendicular P A (Plane A B C))
  (h_PA_length : dist P A = PA_perpendicular_distance)
  (h_A'_reflection : reflection A' A (Plane P B C)) :
  angle (A', C, AB) = 90 :=
by
  sorry

end angle_A_l535_535884


namespace sequence_a_13_l535_535590

noncomputable def sequence_a (n : ℕ) : ℕ :=
if n = 1 then 0 else sequence_a (n - 1) + 2 * sqrt (sequence_a (n - 1) + 1) + 1

theorem sequence_a_13 : sequence_a 13 = 168 :=
sorry

end sequence_a_13_l535_535590


namespace solution_condition1_solution_condition2_solution_condition3_solution_condition4_l535_535815

-- Define the conditions
def Condition1 : Prop :=
  ∃ (total_population box1 box2 sampled : Nat),
  total_population = 30 ∧ box1 = 21 ∧ box2 = 9 ∧ sampled = 10

def Condition2 : Prop :=
  ∃ (total_population produced_by_A produced_by_B sampled : Nat),
  total_population = 30 ∧ produced_by_A = 21 ∧ produced_by_B = 9 ∧ sampled = 10

def Condition3 : Prop :=
  ∃ (total_population sampled : Nat),
  total_population = 300 ∧ sampled = 10

def Condition4 : Prop :=
  ∃ (total_population sampled : Nat),
  total_population = 300 ∧ sampled = 50

-- Define the appropriate methods
def LotteryMethod : Prop := ∃ method : String, method = "Lottery method"
def StratifiedSampling : Prop := ∃ method : String, method = "Stratified sampling"
def RandomNumberMethod : Prop := ∃ method : String, method = "Random number method"
def SystematicSampling : Prop := ∃ method : String, method = "Systematic sampling"

-- Statements to prove the appropriate methods for each condition
theorem solution_condition1 : Condition1 → LotteryMethod := by sorry
theorem solution_condition2 : Condition2 → StratifiedSampling := by sorry
theorem solution_condition3 : Condition3 → RandomNumberMethod := by sorry
theorem solution_condition4 : Condition4 → SystematicSampling := by sorry

end solution_condition1_solution_condition2_solution_condition3_solution_condition4_l535_535815


namespace range_of_f_l535_535380

def f (x : ℝ) := x^2 + 1

theorem range_of_f : set.range f = set.Ici 1 :=
by
  sorry

end range_of_f_l535_535380


namespace find_number_l535_535710

theorem find_number (n : ℕ) : gcd 30 n = 10 ∧ 70 ≤ n ∧ n ≤ 80 ∧ 200 ≤ lcm 30 n ∧ lcm 30 n ≤ 300 → (n = 70 ∨ n = 80) :=
sorry

end find_number_l535_535710


namespace x_ne_y_l535_535236

def x_seq : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := x_seq (n+1) + 2 * x_seq n

def y_seq : ℕ → ℕ
| 0     := 1
| 1     := 7
| (n+2) := 2 * y_seq (n+1) + 3 * y_seq n

theorem x_ne_y {m n : ℕ} (hm : m > 0) (hn : n > 0) : x_seq m ≠ y_seq n :=
by sorry

end x_ne_y_l535_535236


namespace find_missing_part_l535_535037

variable (x y : ℚ) -- Using rationals as the base field for generality.

theorem find_missing_part :
  2 * x * (-3 * x^2 * y) = -6 * x^3 * y := 
by
  sorry

end find_missing_part_l535_535037


namespace arrange_books_l535_535075

theorem arrange_books :
  let total_books := 9
  let algebra_books := 4
  let calculus_books := 5
  algebra_books + calculus_books = total_books →
  Nat.choose total_books algebra_books = 126 :=
by
  intro h
  rw [Nat.choose_eq_factorial_div_factorial (total_books - algebra_books)]
  sorry

end arrange_books_l535_535075


namespace correct_substitution_l535_535757

theorem correct_substitution (x y : ℝ) 
  (h1 : y = 1 - x) 
  (h2 : x - 2 * y = 4) : x - 2 + 2 * x = 4 :=
by
  sorry

end correct_substitution_l535_535757


namespace asymptotes_of_hyperbola_min_focal_distance_l535_535130

theorem asymptotes_of_hyperbola_min_focal_distance :
  ∀ (x y m : ℝ),
  (m = 1 → 
   (∀ x y : ℝ, (x^2 / (m^2 + 8) - y^2 / (6 - 2 * m) = 1) → 
   (y = 2/3 * x ∨ y = -2/3 * x))) := 
  sorry

end asymptotes_of_hyperbola_min_focal_distance_l535_535130


namespace terminating_decimal_count_l535_535144

theorem terminating_decimal_count : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = 49 * k)}.card = 10 :=
by
  sorry

end terminating_decimal_count_l535_535144


namespace find_shorter_diagonal_length_l535_535570

noncomputable def length_of_shorter_diagonal (a b : ℝ^3) (angle : ℝ) (ha : ‖a‖ = 2) (hb : ‖b‖ = 4) (h_angle : angle = π / 3) : ℝ :=
  let dot_product := ‖a‖ * ‖b‖ * Real.cos angle
  let length_squared := ‖a‖^2 + ‖b‖^2 - 2 * dot_product
  Real.sqrt length_squared

theorem find_shorter_diagonal_length (a b : ℝ^3) (angle : ℝ) (ha : ‖a‖ = 2) (hb : ‖b‖ = 4) (h_angle : angle = π / 3) :
  length_of_shorter_diagonal a b angle ha hb h_angle = 2 * Real.sqrt 3 := by
  sorry

#eval find_shorter_diagonal_length (λ i, if i = 0 then 2 else 0) (λ i, if i = 1 then 4 else 0) (π / 3) rfl rfl rfl

end find_shorter_diagonal_length_l535_535570


namespace plot_length_l535_535471

noncomputable def plot_length_conditions (b l : ℝ) : Prop :=
  l = b + 24 ∧
  let P := 2 * l + 2 * b in
  let P_section := P / 2 in
  26.50 * P_section + 30 * P_section = 5300

theorem plot_length : 
  ∃ l : ℝ, ∀ b : ℝ, plot_length_conditions b l → abs (l - 58.9) < 0.01 :=
sorry

end plot_length_l535_535471


namespace problem_statement_l535_535225

-- Define the arithmetic sequence and sum conditions
def S_n : ℕ → ℕ := sorry  -- You will define the actual sum function here when the proof is written
def a_n : ℕ → ℕ := fun n => n  -- By part 1, a_n = n
def b_n (n : ℕ) : ℝ := 1 / (a_n n * a_n (n + 2))
def T_n (n : ℕ) : ℝ := (Finset.range n).sum (λ k => b_n k)

-- Provided conditions
axiom S_3_eq_6 : S_n 3 = 6
axiom S_5_eq_15 : S_n 5 = 15

-- Lean statement to prove
theorem problem_statement (n : ℕ) (S_n_def : ∀ n, S_n n = n * (n + 1) / 2) :
  (∀ m, T_n m ≥ 1 / 3) ∧ (∀ m, T_n m < 3 / 4) :=
by
  -- This is where you would implement the proof for the theorem
  sorry

end problem_statement_l535_535225


namespace circle_eqn_tangent_line_no_intersect_circle_perp_line_intersect_circle_l535_535978

-- Step (Ⅰ): Prove the equation of the circle
theorem circle_eqn_tangent
  (C : (ℝ × ℝ)) (h1 : C = (1, -2))
  (tangent_line : ℝ → ℝ → ℝ)
  (h2 : ∀ x y, tangent_line x y = x + y + 3 * sqrt 2 + 1)
  (tangent_condition : ∀ x y, tangent_line x y = 0 → (x - 1)^2 + (y + 2)^2 = 9) :
  (x - 1)^2 + (y + 2)^2 = 9 :=
by sorry

-- Step (Ⅱ): Prove the range of values for k
theorem line_no_intersect_circle
  (C : (ℝ × ℝ)) (h1 : C = (1, -2))
  (circle_eqn : ℝ → ℝ → ℝ)
  (h3 : ∀ x y, circle_eqn x y = (x - 1)^2 + (y + 2)^2 - 9)
  (k : ℝ)
  (line_eqn : ℝ → ℝ → ℝ)
  (h4 : ∀ x y, line_eqn x y = y - k * x - 1)
  (no_intersect_condition : ∀ x y, line_eqn x y ≠ 0 → ∃ r, circle_eqn (C.1) (C.2) < (x - C.1)^2 + (y - C.2)^2) :
  0 < k ∧ k < 3 / 4 :=
by sorry

-- Step (Ⅲ): Prove the values of m
theorem perp_line_intersect_circle
  (C : (ℝ × ℝ)) (h1 : C = (1, -2))
  (circle_eqn : ℝ → ℝ → ℝ)
  (h3 : ∀ x y, circle_eqn x y = (x - 1)^2 + (y + 2)^2 - 9)
  (m : ℝ)
  (line_eqn : ℝ → ℝ → ℝ)
  (h5 : ∀ x y, line_eqn x y = y - x - m)
  (intersect_condition : ∃ x₁ y₁ x₂ y₂, line_eqn x₁ y₁ = 0 ∧ circle_eqn x₁ y₁ = 0 ∧ line_eqn x₂ y₂ = 0 ∧ circle_eqn x₂ y₂ = 0 ∧ ((x₁ - C.1) * (x₂ - C.1) + (y₁ - C.2) * (y₂ - C.2)) = 0) :
  m = 1 ∨ m = -4 :=
by sorry

end circle_eqn_tangent_line_no_intersect_circle_perp_line_intersect_circle_l535_535978


namespace ambulance_ride_cost_l535_535996

theorem ambulance_ride_cost 
  (total_bill : ℝ) 
  (medication_percentage overnight_percentage : ℝ) 
  (food_cost : ℝ)
  (H1 : total_bill = 5000)
  (H2 : medication_percentage = 0.50)
  (H3 : overnight_percentage = 0.25)
  (H4 : food_cost = 175) : 
  let medication_cost := medication_percentage * total_bill in
  let remaining_after_medication := total_bill - medication_cost in
  let overnight_cost := overnight_percentage * remaining_after_medication in
  let remaining_after_overnight := remaining_after_medication - overnight_cost in
  let remaining_after_food := remaining_after_overnight - food_cost in 
  remaining_after_food = 1700 :=
by sorry

end ambulance_ride_cost_l535_535996


namespace johns_weekly_earnings_l535_535646

-- Define conditions
def days_off_per_week : ℕ := 3
def streaming_hours_per_day : ℕ := 4
def earnings_per_hour : ℕ := 10

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the question as a theorem
theorem johns_weekly_earnings :
  let days_streaming_per_week := days_in_week - days_off_per_week,
      weekly_streaming_hours := days_streaming_per_week * streaming_hours_per_day,
      weekly_earnings := weekly_streaming_hours * earnings_per_hour
  in weekly_earnings = 160 := by
  -- Proof is omitted with 'sorry'
  sorry

end johns_weekly_earnings_l535_535646


namespace meeting_point_2015th_l535_535174

-- Define the parameters of the problem
variables (A B C D : Type)
variables (x y t : ℝ) -- Speeds and the initial time delay

-- State the problem as a theorem
theorem meeting_point_2015th (start_times_differ : t > 0)
                            (speeds_pos : x > 0 ∧ y > 0)
                            (pattern : ∀ n : ℕ, (odd n → (meeting_point n = C)) ∧ (even n → (meeting_point n = D)))
                            (n = 2015) :
  meeting_point n = C :=
  sorry

end meeting_point_2015th_l535_535174


namespace find_angle_A_and_area_of_triangle_l535_535964

theorem find_angle_A_and_area_of_triangle 
  (A B C a b c : ℝ) 
  (h1 : a = 2) 
  (h2 : b + c = 2 * Real.sqrt 3) 
  (h3 : Real.sin B * Real.sin C - Real.cos B * Real.cos C = 1 / 2)
  : A = Real.pi / 3 ∧ 
    let bc := (b * c) in 
    let area := (1 / 2) * bc * Real.sin A in 
    area = (2 * Real.sqrt 3) / 3 := 
by 
  sorry

end find_angle_A_and_area_of_triangle_l535_535964


namespace petya_wins_n_ne_4_l535_535827

theorem petya_wins_n_ne_4 (n : ℕ) (h : n ≥ 3) :
  (∀ k, k ∈ {1, ..., n - 1} -> 
  (∃ i, (empty_glasses[i] ∧ empty_glasses[i+1])) ∨
  (∃ j, (full_glasses[j] ∧ full_glasses[j+1])) →
  (Vasya_moves (one empty one full)) →
  player_cannot_move_loses) ↔ (n ≠ 4) :=
sorry

end petya_wins_n_ne_4_l535_535827


namespace meeting_point_2015_is_C_l535_535168

-- Given definitions based on conditions
variable (x y : ℝ) -- Speeds of the motorcycle and the cyclist
variable (A B C D : Point) -- Points on segment AB
variable (meetings : ℕ → Point) -- Function representing the meeting point sequence

-- Conditions stating the alternating meeting pattern
axiom start_at_A (n : ℕ) : meetings (2 * n + 1) = C
axiom start_at_B (n : ℕ) : meetings (2 * n + 2) = D

-- The theorem statement to be proved
theorem meeting_point_2015_is_C : meetings 2015 = C := sorry

end meeting_point_2015_is_C_l535_535168


namespace max_expression_value_l535_535981

theorem max_expression_value (a b c d e : ℕ)
  (h : {a, b, c, d, e} = {0, 1, 2, 3, 4}) :
  e * c^a + b - d ≤ 39 :=
by sorry

end max_expression_value_l535_535981


namespace isosceles_triangle_DMB_l535_535522

-- Definitions for segment lengths based on their ratios.
variables {A B C D O M : Type*}
variables [LinearOrderedField A B C D O M]
variable (A B C D O M: A)

-- Definitions for the lengths and ratios
def equal_segments : A = C := sorry
def segment_ratio_AO_OB : O = 1/3 * A / (2/3 * A) := sorry
def segment_ratio_CO_OD : O = 1/3 * C / (2/3 * C) := sorry
def intersection_point : ∀ P, ∃ M, on_line A D P ∧ on_line B C P := sorry

-- Now the proof statement based on given definitions.
theorem isosceles_triangle_DMB :
  equal_segments →
  segment_ratio_AO_OB →
  segment_ratio_CO_OD →
  intersection_point O →
  is_isosceles D M B :=
sorry

end isosceles_triangle_DMB_l535_535522


namespace average_price_of_towels_l535_535079

-- Definitions based on conditions
def cost_towel1 : ℕ := 3 * 100
def cost_towel2 : ℕ := 5 * 150
def cost_towel3 : ℕ := 2 * 600
def total_cost : ℕ := cost_towel1 + cost_towel2 + cost_towel3
def total_towels : ℕ := 3 + 5 + 2
def average_price : ℕ := total_cost / total_towels

-- Statement to be proved
theorem average_price_of_towels :
  average_price = 225 :=
by
  sorry

end average_price_of_towels_l535_535079


namespace complex_magnitude_l535_535313

theorem complex_magnitude (i : ℂ) (hi : i^2 = -1) : 
  |((1 + i)^13 - (1 - i)^13)| = 128 := 
by 
  sorry

end complex_magnitude_l535_535313


namespace correct_statement_l535_535432

-- Definition of terms
def term_coeff (t : Type) [Ring t] := t
def term_var (v : Type) := v -> Nat

-- The degree of a term is defined as the sum of the exponents of all its variables.
def degree (t : Type) [Ring t] (a : term_var t) (e : t -> Nat) : Nat := 
  e a 

-- A monomial is an algebraic expression with only one term.
def is_monomial (t : Type) [Ring t] (l : List t) : Prop := 
  l.length = 1

-- The coefficient of a term is the numerical part of the term.
def coefficient {t : Type} [Ring t] (a : t) : t :=
  a

-- The proof problem translated from the identified conditions and conclusion.
theorem correct_statement : 
  degree Int id (λ a, if a = 1 then 2 else if a = 2 then 1 else 0) = 3 :=
begin
  sorry
end

end correct_statement_l535_535432


namespace ellipse_equation_line_AC_l535_535708

noncomputable def ellipse_eq (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def foci_distance (a c : ℝ) : Prop := 
  a - c = 1 ∧ a + c = 3

noncomputable def b_value (a c b : ℝ) : Prop :=
  b = Real.sqrt (a^2 - c^2)

noncomputable def rhombus_on_line (m : ℝ) : Prop := 
  7 * (2 * m / 7) + 1 - 7 * (3 * m / 7) = 0

theorem ellipse_equation (a b c : ℝ) (h1 : foci_distance a c) (h2 : b_value a c b) :
  ellipse_eq x y a b :=
sorry

theorem line_AC (a b c x y x1 y1 x2 y2 : ℝ) 
  (h1 : ellipse_eq x1 y1 a b)
  (h2 : ellipse_eq x2 y2 a b)
  (h3 : 7 * x1 - 7 * y1 + 1 = 0)
  (h4 : 7 * x2 - 7 * y2 + 1 = 0)
  (h5 : rhombus_on_line y) :
  x + y + 1 = 0 :=
sorry

end ellipse_equation_line_AC_l535_535708


namespace weight_greater_than_half_pow_n_l535_535852

variable {n : ℕ}
variable {p : ℕ → ℝ}

theorem weight_greater_than_half_pow_n (h1 : ∀ i, p i ≥ 0)
    (h2 : ∑ i in finset.range n, p i = 1) : 
    ∃ i < n, p i > 1 / 2 ^ i :=
sorry

end weight_greater_than_half_pow_n_l535_535852


namespace Griffin_Hailey_passes_l535_535239

theorem Griffin_Hailey_passes
  (v_G : ℝ) (v_H : ℝ) (r_G : ℝ) (r_H : ℝ) (t : ℝ)
  (C_G : real.pi * 2 * r_G = 100 * real.pi)
  (C_H : real.pi * 2 * r_H = 90 * real.pi)
  (ω_G : v_G / (100 * real.pi) * 2 * real.pi = 5.2)
  (ω_H : v_H / (90 * real.pi) * 2 * real.pi ≈ 6.89)
  : ⌊(t * (ω_G + ω_H) / (2 * real.pi))⌋ = 86 :=
sorry

end Griffin_Hailey_passes_l535_535239


namespace least_number_to_add_l535_535425

theorem least_number_to_add (x : ℕ) : (1021 + x) % 25 = 0 ↔ x = 4 := 
by 
  sorry

end least_number_to_add_l535_535425


namespace num_int_values_n_terminated_l535_535154

theorem num_int_values_n_terminated (N : ℕ) (hN1 : 1 ≤ N) (hN2 : N ≤ 500) :
  ∃ n : ℕ, n = 10 ∧ ∀ k, 0 ≤ k → k < n → ∃ (m : ℕ), N = m * 49 :=
sorry

end num_int_values_n_terminated_l535_535154


namespace all_terms_are_integers_l535_535719

def x : ℕ → ℝ
| 0       := 1
| (n + 1) := (3 * x n + real.sqrt (5 * (x n) ^ 2 - 4)) / 2

theorem all_terms_are_integers : ∀ n, ∃ k : ℤ, x n = k :=
by sorry -- This part refers to the base case and induction step described in the solution.

end all_terms_are_integers_l535_535719


namespace dasha_strip_dimensions_l535_535503

theorem dasha_strip_dimensions (a b c : ℕ) (h_sum : 2 * a * (b + c) - a * a = 43) : 
  a = 1 ∧ b + c = 22 := 
begin 
  sorry 
end

end dasha_strip_dimensions_l535_535503


namespace coefficient_of_x5_expansion_l535_535281

theorem coefficient_of_x5_expansion (a : ℝ) (h : (a - x) * (2 + x) ^ 6 = polynomial (coeff' (x ^ 5)) = 12) : a = 6 :=
sorry

end coefficient_of_x5_expansion_l535_535281


namespace find_fx_neg_l535_535895

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x

theorem find_fx_neg (h1 : odd_function f) (h2 : f_nonneg f) : 
  ∀ x : ℝ, x < 0 → f x = -x^2 - 2*x := 
by
  sorry

end find_fx_neg_l535_535895


namespace no_such_polynomials_a_b_c_l535_535338

noncomputable theory

def condition (f g : Polynomial ℝ) (a b c : ℕ) :=
  (a > 1) ∧ (b > 1) ∧ (c > 1) ∧
  (f ≠ 0) ∧ (g ≠ 0) ∧ 
  (¬ isRoot f 1) ∧ (¬ isRoot g 1) ∧
  (∀ z : ℂ, isRoot f z → Im z = 0) ∧ (∀ z : ℂ, isRoot g z → Im z = 0) ∧
  (f^a + g^b = Polynomial.X - 1)^c

theorem no_such_polynomials_a_b_c :
  ¬ ∃ (a b c : ℕ) (f g : Polynomial ℝ), condition f g a b c :=
sorry

end no_such_polynomials_a_b_c_l535_535338


namespace matchbox_positioning_possible_l535_535777

theorem matchbox_positioning_possible :
  ∃ (matches : Fin 3 → Prop) (matchbox : Prop),
    (∀ i, ¬ matches i.head_touches_table) ∧
    (∀ i j, i ≠ j → ¬ matches i.head_touches matches j) ∧
    (∀ i, ¬ matches i.head_touches matchbox) :=
sorry

end matchbox_positioning_possible_l535_535777


namespace total_bathing_suits_l535_535454

theorem total_bathing_suits (men_women_bathing_suits : ℕ) (men_bathing_suits : ℕ) (women_bathing_suits : ℕ):
  men_bathing_suits = 14797 → women_bathing_suits = 4969 → men_bathing_suits + women_bathing_suits = 19766 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end total_bathing_suits_l535_535454


namespace problem_l535_535661

set_option pp.generalizedFieldNotations false

-- Given definitions
variables {Line Plane : Type}
variable l : Line
variable m : Line
variable n : Line
variable α : Plane
variable β : Plane

-- Definitions of conditions
def non_coincident (x y z : Line) : Prop := x ≠ y ∧ y ≠ z ∧ z ≠ x
def perpendicular (x y : Plane) : Prop := ∀ (l : Line), l ∈ x → l ∉ y
def subset_line_in_plane (l : Line) (α : Plane) : Prop := l ∈ α
def parallel (l : Line) (β : Plane) : Prop := ∃ (m : Line), m ∈ β ∧ l ∥ m

-- Defining the propositions
def propA {α β : Plane} {l : Line} (h1 : l ∈ α) (h2 : l ∥ β) (h3 : l ⊥ α) : Prop := α ⊥ β

-- Lean statement
theorem problem (h1 : non_coincident l m n)
               (h2 : perpendicular α β)
               (h3 : subset_line_in_plane l α)
               (h4 : subset_line_in_plane m α)
               (h5 : l ⊥ α)
               (h6 : l ∥ β) :
  α ⊥ β :=
by {
  apply h2, -- this uses the condition α ⊥ β from the definitions
  sorry -- skips the proof
}

end problem_l535_535661


namespace pop_cd_price_l535_535730

-- Define the cost of each type of CD
def rock_and_roll_cd_price : ℕ := 5
def dance_cd_price : ℕ := 3
def country_cd_price : ℕ := 7

-- Define the number of each type of CD Julia wants to buy
def num_cds : ℕ := 4

-- Julia's budget details
def julia_budget : ℕ := 75
def julia_shortfall : ℕ := 25

-- Calculate the total cost of rock, dance, and country CDs
def cost_rock_and_roll_cds := num_cds * rock_and_roll_cd_price
def cost_dance_cds := num_cds * dance_cd_price
def cost_country_cds := num_cds * country_cd_price

-- Calculate the total amount Julia needs
def total_required := julia_budget + julia_shortfall

-- Calculate the cost of pop CDs
def expected_pop_cd_price := 10

theorem pop_cd_price :
  let total_rock_dance_country_cost := cost_rock_and_roll_cds + cost_dance_cds + cost_country_cds in
  let total_pop_cd_cost := total_required - total_rock_dance_country_cost in
  total_pop_cd_cost / num_cds = expected_pop_cd_price :=
by {
  sorry -- Proof is omitted as requested.
}

end pop_cd_price_l535_535730


namespace brenda_spay_cats_l535_535828

theorem brenda_spay_cats (c d : ℕ) (h1 : c + d = 21) (h2 : d = 2 * c) : c = 7 :=
sorry

end brenda_spay_cats_l535_535828


namespace total_crackers_l535_535499

-- Definitions based on conditions
def boxes_darren_bought : ℕ := 4
def crackers_per_box : ℕ := 24
def boxes_calvin_bought : ℕ := (2 * boxes_darren_bought) - 1

-- The statement to prove
theorem total_crackers (boxes_darren_bought = 4) (crackers_per_box = 24) : 
  (boxes_darren_bought * crackers_per_box) + (boxes_calvin_bought * crackers_per_box) = 264 := 
by 
  sorry

end total_crackers_l535_535499


namespace george_borrow_amount_l535_535866

-- Define the conditions
def initial_fee_rate : ℝ := 0.05
def doubling_rate : ℝ := 2
def total_weeks : ℕ := 2
def total_fee : ℝ := 15

-- Define the problem statement
theorem george_borrow_amount : 
  ∃ (P : ℝ), (initial_fee_rate * P + initial_fee_rate * doubling_rate * P = total_fee) ∧ P = 100 :=
by
  -- Statement only, proof is skipped
  sorry

end george_borrow_amount_l535_535866


namespace center_of_symmetry_of_polar_curve_l535_535633

theorem center_of_symmetry_of_polar_curve :
  ∃ (r θ : ℝ), (r = 1) ∧ (θ = π / 2) ∧
  (∀ (ρ θ₀ : ℝ), (ρ = 2 * sin θ₀) → 
  (ρ^2 = 2 * ρ * sin θ₀ → 
  (let x := ρ * cos θ₀,
       y := ρ * sin θ₀ 
   in (x^2 + (y - 1)^2 = 1))) :=
begin
  sorry
end

end center_of_symmetry_of_polar_curve_l535_535633


namespace count_not_squares_or_cubes_l535_535944

theorem count_not_squares_or_cubes (n : ℕ) : 
  let total := 200 in
  let perfect_squares := 14 in
  let perfect_cubes := 5 in
  let perfect_sixth_powers := 2 in
  let squares_or_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers in
  let count_not_squares_or_cubes := total - squares_or_cubes in
  n = count_not_squares_or_cubes :=
by
  let total := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let perfect_sixth_powers := 2
  let squares_or_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let count_not_squares_or_cubes := total - squares_or_cubes
  show _ from sorry

end count_not_squares_or_cubes_l535_535944


namespace meeting_point_2015th_l535_535167

-- Definitions for the conditions
def motorist_speed (x : ℝ) := x
def cyclist_speed (y : ℝ) := y
def initial_delay (t : ℝ) := t 
def first_meeting_point := C
def second_meeting_point := D

-- The main proof problem statement
theorem meeting_point_2015th
  (x y t : ℝ) -- speeds of the motorist and cyclist and the initial delay
  (C D : Point) -- points C and D on the segment AB where meetings occur
  (pattern_alternation : ∀ n: ℤ, n > 0 → ((n % 2 = 1) → n-th_meeting_point = C) ∧ ((n % 2 = 0) → n-th_meeting_point = D))
  (P_A_B_cycle : ∀ n: ℕ, (P → A ∨ B → C ∨ A → B ∨ D → P) holds for each meeting): 
  2015-th_meeting_point = C :=
by
  sorry

end meeting_point_2015th_l535_535167


namespace first_player_wins_l535_535436

/-- In a game where players take turns moving a token, with the k-th move shifting the 
token by 2^(k−1) cells, and the player who cannot make the next valid move loses, 
the first player has a winning strategy regardless of the opponent's moves. -/
theorem first_player_wins
  (game_moves : Nat → Nat)
  (initial_board_size : Nat) :
  (∀ k, game_moves k = 2^(k-1)) →
  (∃ n, ∑ i in finset.range n, game_moves i < initial_board_size) →
  (∃ k, game_moves k ≥ initial_board_size) →
  ∀ player, player ≠ "First" → ∃ move, move ∈ game_moves → (move ≠ 2^(initial_board_size - 1)) :=
sorry

end first_player_wins_l535_535436


namespace other_factor_form_l535_535050

theorem other_factor_form (w : ℕ) (h_pos : 0 < w)
  (h_factors : 936 * w % (2^5 * 3^3) = 0)
  (h_min_w : w = 144) :
  ∃ x : ℕ, 12^x = 12^2 :=
by
  use 2
  sorry

end other_factor_form_l535_535050


namespace gadgets_selling_prices_and_total_amount_l535_535465

def cost_price_mobile : ℕ := 16000
def cost_price_laptop : ℕ := 25000
def cost_price_camera : ℕ := 18000

def loss_percentage_mobile : ℕ := 20
def gain_percentage_laptop : ℕ := 15
def loss_percentage_camera : ℕ := 10

def selling_price_mobile : ℕ := cost_price_mobile - (cost_price_mobile * loss_percentage_mobile / 100)
def selling_price_laptop : ℕ := cost_price_laptop + (cost_price_laptop * gain_percentage_laptop / 100)
def selling_price_camera : ℕ := cost_price_camera - (cost_price_camera * loss_percentage_camera / 100)

def total_amount_received : ℕ := selling_price_mobile + selling_price_laptop + selling_price_camera

theorem gadgets_selling_prices_and_total_amount :
  selling_price_mobile = 12800 ∧
  selling_price_laptop = 28750 ∧
  selling_price_camera = 16200 ∧
  total_amount_received = 57750 := by
  sorry

end gadgets_selling_prices_and_total_amount_l535_535465


namespace distance_between_towns_l535_535843

theorem distance_between_towns 
  (x : ℝ) 
  (h1 : x / 100 - x / 110 = 0.15) : 
  x = 165 := 
by 
  sorry

end distance_between_towns_l535_535843


namespace count_terminating_decimals_l535_535140

theorem count_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 500) : 
  (nat.floor (500 / 49) = 10) := 
by
  sorry

end count_terminating_decimals_l535_535140


namespace tangent_line_through_P_F_monotonic_intervals_compare_logs_l535_535914

open Real

-- Define the function f(x)
def f (x : ℝ) := exp x

-- Define the inverse function g(x)
noncomputable def g (x : ℝ) := log x

-- Define the point P
def P : ℝ × ℝ := (-1, 0)

-- Define the function F(x)
noncomputable def F (x : ℝ) := x / log x

-- Theorem statement for problem (1)
theorem tangent_line_through_P : 
  ∃ (m : ℝ), (P.snd = m * P.fst + (exp m)) ∧
             (∀ x : ℝ, exp x = x * exp m + (exp m)) → 
             m = 0 :=
sorry

-- Theorem stating problem (2) - monotonic intervals and minimum value
theorem F_monotonic_intervals :
  (∀ x, (x > 0 ∧ x < exp 1) → deriv (λ x, F x) x < 0) ∧
  (∀ x, (x > exp 1) → deriv (λ x, F x) x > 0) ∧
  (F (exp 1) = exp 1) :=
sorry

-- Theorem comparing the values
theorem compare_logs :
  sqrt 2 * log (sqrt 3) > sqrt 3 * log (sqrt 2) :=
sorry

end tangent_line_through_P_F_monotonic_intervals_compare_logs_l535_535914


namespace statistics_no_increase_l535_535674

theorem statistics_no_increase 
  (scores : List ℝ := [32, 35, 37, 38, 41, 43, 45, 48, 52, 55]) 
  (new_score : ℝ := 34) :
  ¬ (∃ stat_increases : Prop, 
     (let range_increases := (let min_score_before := 32 in 
                              let max_score_before := 55 in 
                              let range_before := max_score_before - min_score_before in 
                              let min_score_after := min min_score_before new_score in 
                              let max_score_after := max max_score_before new_score in 
                              let range_after := max_score_after - min_score_after in 
                              range_after > range_before))
     ∨ (let median_increases := (let sorted_scores_before := scores in 
                                 let median_before := 42 in 
                                 let sorted_scores_after := List.sort (List.insert new_score sorted_scores_before) in 
                                 let median_after := 41 in 
                                 median_after > median_before))
     ∨ (let mean_increases := (let sum_before := 426 in 
                               let count_before := 10 in 
                               let mean_before := sum_before / count_before in 
                               let sum_after := sum_before + new_score in 
                               let count_after := count_before + 1 in 
                               let mean_after := sum_after / count_after in 
                               mean_after > mean_before))
     ∨ (let mode_increases := false) 
     ∨ (let midrange_increases := (let min_score_before := 32 in 
                                   let max_score_before := 55 in 
                                   let midrange_before := (min_score_before + max_score_before) / 2 in 
                                   let min_score_after := min min_score_before new_score in 
                                   let max_score_after := max max_score_before new_score in 
                                   let midrange_after := (min_score_after + max_score_after) / 2 in 
                                   midrange_after > midrange_before))) then 
  sorry

end statistics_no_increase_l535_535674


namespace part_1_part_2_l535_535199

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0],
    ![-1, 1]]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2],
    ![0, 3]]

def C : Matrix (Fin 2) (Fin 2) ℤ :=
  A ⬝ B

def line_l1 (x y : ℤ) : Prop :=
  x + y = 0

def transformation (A : Matrix (Fin 2) (Fin 2) ℤ) (v : Fin 2 → ℤ) : Fin 2 → ℤ :=
  λ i, Fin.sum_univ (λ j, A i j * v j)

theorem part_1 : C = ![![1, 2],
                       ![-1, 1]] := by
  sorry

theorem part_2 (x y : ℤ) (h : line_l1 x y) :
  let v := ![x, y]
  let v' := transformation C v
  line_l2 (v' 0) (v' 1) := by
  sorry

def line_l2 (x' y' : ℤ) : Prop :=
  2 * x' - y' = 0

#check part_1
#check part_2

end part_1_part_2_l535_535199


namespace perp_line_parallel_plane_perp_line_l535_535208

variable {Line : Type} {Plane : Type}
variable (a b : Line) (α β : Plane)
variable (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (parallel_lines : Line → Line → Prop)

-- Conditions
variable (non_coincident_lines : ¬(a = b))
variable (non_coincident_planes : ¬(α = β))
variable (a_perp_α : perpendicular a α)
variable (b_par_α : parallel b α)

-- Prove
theorem perp_line_parallel_plane_perp_line :
  perpendicular a α ∧ parallel b α → parallel_lines a b :=
sorry

end perp_line_parallel_plane_perp_line_l535_535208


namespace inscribed_sphere_radius_l535_535474

theorem inscribed_sphere_radius (b d : ℝ) : 
  (b * Real.sqrt d - b = 15 * (Real.sqrt 5 - 1) / 4) → 
  b + d = 11.75 :=
by
  intro h
  sorry

end inscribed_sphere_radius_l535_535474


namespace mean_median_mode_eq_x_l535_535713

theorem mean_median_mode_eq_x (x : ℕ) :
  (70 + 110 + x + 45 + 55 + 210 + 95 + 85 + 130) / 9 = x ∧
  list.median [45, 55, 70, 85, 95, 100, 110, 130, 210] = some x ∧
  list.mode [70, 110, x, 45, 55, 210, 95, 85, 130] = some x
  ↔ x = 100 := 
by {
  -- Placeholder for proof
  sorry
}

end mean_median_mode_eq_x_l535_535713


namespace isosceles_triangle_sides_20cm_isosceles_triangle_with_base_4cm_l535_535078

-- Part 1
theorem isosceles_triangle_sides_20cm (base leg : ℕ) (h_leg_eq : leg = base - 2) (h_eq : base + 2 * leg = 20) :
  {a // a = 6 ∧ a = leg} ∧ {b // b = 6 ∧ b = leg} ∧ {c // c = 8 ∧ c = base} :=
by
  sorry

-- Part 2
theorem isosceles_triangle_with_base_4cm (leg base : ℕ) (h_base_eq : base = 4) (h_total : 2 * leg + base = 20) :
  (leg = 8) ∧ (base = 4) :=
by
  sorry

end isosceles_triangle_sides_20cm_isosceles_triangle_with_base_4cm_l535_535078


namespace rhombus_diagonals_ratio_l535_535472

theorem rhombus_diagonals_ratio (a b d1 d2 : ℝ) 
  (h1: a > 0) (h2: b > 0)
  (h3: d1 = 2 * (a / Real.cos θ))
  (h4: d2 = 2 * (b / Real.cos θ)) :
  d1 / d2 = a / b := 
sorry

end rhombus_diagonals_ratio_l535_535472


namespace coeff_of_x6_in_expansion_l535_535507

theorem coeff_of_x6_in_expansion : 
  (∃ (c : ℤ), c = (finset.range 7).sum (λ r, (((-1)^r) * (nat.choose 6 r) * (1 : ℤ) ^ (6 - r))) ∧ c = -20) :=
begin
  sorry
end

end coeff_of_x6_in_expansion_l535_535507


namespace find_angle_B_l535_535966

variable (a b c A B C : ℝ)

-- Assuming all the necessary conditions and givens
axiom triangle_condition1 : a * (Real.sin B * Real.cos C) + c * (Real.sin B * Real.cos A) = (1 / 2) * b
axiom triangle_condition2 : a > b

-- We need to prove B = π / 6
theorem find_angle_B : B = π / 6 :=
by
  sorry

end find_angle_B_l535_535966


namespace count_non_squares_or_cubes_in_200_l535_535945

theorem count_non_squares_or_cubes_in_200 :
  let total_numbers := 200
  let count_perfect_squares := 14
  let count_perfect_cubes := 5
  let count_sixth_powers := 2
  total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers) = 183 :=
by
  let total_numbers := 200
  let count_perfect_squares := 14
  let count_perfect_cubes := 5
  let count_sixth_powers := 2
  have h1 : total_numbers = 200 := rfl
  have h2 : count_perfect_squares = 14 := rfl
  have h3 : count_perfect_cubes = 5 := rfl
  have h4 : count_sixth_powers = 2 := rfl
  show total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers) = 183
  calc
    total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers)
        = 200 - (14 + 5 - 2) : by rw [h1, h2, h3, h4]
    ... = 200 - 17 : by norm_num
    ... = 183 : by norm_num

end count_non_squares_or_cubes_in_200_l535_535945


namespace meeting_point_2015th_l535_535165

-- Definitions for the conditions
def motorist_speed (x : ℝ) := x
def cyclist_speed (y : ℝ) := y
def initial_delay (t : ℝ) := t 
def first_meeting_point := C
def second_meeting_point := D

-- The main proof problem statement
theorem meeting_point_2015th
  (x y t : ℝ) -- speeds of the motorist and cyclist and the initial delay
  (C D : Point) -- points C and D on the segment AB where meetings occur
  (pattern_alternation : ∀ n: ℤ, n > 0 → ((n % 2 = 1) → n-th_meeting_point = C) ∧ ((n % 2 = 0) → n-th_meeting_point = D))
  (P_A_B_cycle : ∀ n: ℕ, (P → A ∨ B → C ∨ A → B ∨ D → P) holds for each meeting): 
  2015-th_meeting_point = C :=
by
  sorry

end meeting_point_2015th_l535_535165


namespace meeting_point_2015th_l535_535164

-- Definitions for the conditions
def motorist_speed (x : ℝ) := x
def cyclist_speed (y : ℝ) := y
def initial_delay (t : ℝ) := t 
def first_meeting_point := C
def second_meeting_point := D

-- The main proof problem statement
theorem meeting_point_2015th
  (x y t : ℝ) -- speeds of the motorist and cyclist and the initial delay
  (C D : Point) -- points C and D on the segment AB where meetings occur
  (pattern_alternation : ∀ n: ℤ, n > 0 → ((n % 2 = 1) → n-th_meeting_point = C) ∧ ((n % 2 = 0) → n-th_meeting_point = D))
  (P_A_B_cycle : ∀ n: ℕ, (P → A ∨ B → C ∨ A → B ∨ D → P) holds for each meeting): 
  2015-th_meeting_point = C :=
by
  sorry

end meeting_point_2015th_l535_535164


namespace ratio_of_distances_in_tetrahedron_l535_535303

theorem ratio_of_distances_in_tetrahedron (A B C D E : Point)
  (h_tetrahedron : regular_tetrahedron A B C D)
  (hE_inside : inside_tetrahedron E A B C D) :
  let s := sum_of_distances_to_faces E A B C D
  let S := sum_of_distances_to_edges E A B C D
  in s / S = 2 * Real.sqrt 2 / 3 :=
by
  sorry

end ratio_of_distances_in_tetrahedron_l535_535303


namespace sara_initial_quarters_l535_535346

theorem sara_initial_quarters (total_quarters: ℕ) (dad_gave: ℕ) (initial_quarters: ℕ) 
  (h1: dad_gave = 49) (h2: total_quarters = 70) (h3: total_quarters = initial_quarters + dad_gave) :
  initial_quarters = 21 := 
by {
  sorry
}

end sara_initial_quarters_l535_535346


namespace matt_age_three_years_ago_l535_535043

theorem matt_age_three_years_ago (james_age_three_years_ago : ℕ) (age_difference : ℕ) (future_factor : ℕ) :
  james_age_three_years_ago = 27 →
  age_difference = 3 →
  future_factor = 2 →
  ∃ matt_age_now : ℕ,
  james_age_now: ℕ,
    james_age_now = james_age_three_years_ago + age_difference ∧
    (matt_age_now + 5) = future_factor * (james_age_now + 5) ∧
    matt_age_now = 65 :=
by
  sorry

end matt_age_three_years_ago_l535_535043


namespace perimeter_of_rectangle_l535_535800

-- Define the side of the larger square and the central square
variables {y : ℝ}

-- Define the conditions
def large_square_side := 8 * y
def central_square_side := 3 * y

-- Definition of the perimeter of the rectangle
def rectangle_perimeter (x : ℝ) := 2 * (3 * x + (8 * x - 3 * x))

-- Statement to prove
theorem perimeter_of_rectangle (h1 : large_square_side = 8 * y) 
                              (h2 : central_square_side = 3 * y) : 
    rectangle_perimeter y = 16 * y :=
by
  -- The exact proof is not required; "sorry" is used here to skip the proof.
  sorry

end perimeter_of_rectangle_l535_535800


namespace find_angle_of_inclination_l535_535724

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := 2 * x + 2 * y - 5 = 0

-- Define the angle of inclination function
def tan_inv (m : ℝ) : ℝ :=
if m = -1 then 135 else sorry -- Simplified for this purpose

theorem find_angle_of_inclination (x y : ℝ) (h : line_eq x y) : 
  let m := -1 in tan_inv m = 135 :=
by
  -- The necessary proof steps will be added here later
  sorry

end find_angle_of_inclination_l535_535724


namespace James_beat_old_record_by_296_points_l535_535644

def touchdowns_per_game := 4
def points_per_touchdown := 6
def number_of_games := 15
def two_point_conversions := 6
def points_per_two_point_conversion := 2
def field_goals := 8
def points_per_field_goal := 3
def extra_point_attempts := 20
def points_per_extra_point := 1
def consecutive_touchdowns := 3
def games_with_consecutive_touchdowns := 5
def bonus_multiplier := 2
def old_record := 300

def James_points : ℕ :=
  (touchdowns_per_game * number_of_games * points_per_touchdown) + 
  ((consecutive_touchdowns * games_with_consecutive_touchdowns) * points_per_touchdown * bonus_multiplier) +
  (two_point_conversions * points_per_two_point_conversion) +
  (field_goals * points_per_field_goal) +
  (extra_point_attempts * points_per_extra_point)

def points_above_old_record := James_points - old_record

theorem James_beat_old_record_by_296_points : points_above_old_record = 296 := by
  -- here would be the proof
  sorry

end James_beat_old_record_by_296_points_l535_535644


namespace probability_blue_draw_l535_535792

noncomputable def probability_all_blue (num_red num_blue num_white num_green : ℕ) (draws : ℕ) : ℚ :=
let total_marbles := num_red + num_blue + num_white + num_green in
if h : draws = 3 ∧ num_blue >= 3 then
  (num_blue / total_marbles) * ((num_blue - 1) / (total_marbles - 1)) * ((num_blue - 2) / (total_marbles - 2))
else 0

theorem probability_blue_draw :
  probability_all_blue 4 5 8 3 3 = 1 / 114 :=
by sorry

end probability_blue_draw_l535_535792


namespace value_of_x_l535_535424

theorem value_of_x (x : ℤ) : (3000 + x) ^ 2 = x ^ 2 → x = -1500 := 
by
  sorry

end value_of_x_l535_535424


namespace non_neg_int_solutions_l535_535526

theorem non_neg_int_solutions :
  (∀ m n k : ℕ, 2 * m + 3 * n = k ^ 2 →
    (m = 0 ∧ n = 1 ∧ k = 2) ∨
    (m = 3 ∧ n = 0 ∧ k = 3) ∨
    (m = 4 ∧ n = 2 ∧ k = 5)) :=
by
  intro m n k h
  -- outline proof steps here
  sorry

end non_neg_int_solutions_l535_535526


namespace simplify_fraction_l535_535523

variable (a x b : ℝ)
variable (h_b_gt0 : b > 0)

theorem simplify_fraction (a x b : ℝ) (h_b_gt0 : b > 0) :
  (sqrt b * (sqrt (a^2 + x^2) - (x^2 - a^2) / sqrt (a^2 + x^2)) / (b * (a^2 + x^2))) = 
  (2 * a^2 * sqrt b) / (b * (a^2 + x^2) ^ (3 / 2)) := 
by
  sorry

end simplify_fraction_l535_535523


namespace angle_DTQ_is_right_angle_l535_535410

/-- \(ABCD\) is a square where \(P\) and \(Q\) are on sides \(AB\) and \(BC\) respectively with \(BP = BQ\).
A perpendicular line from \(B\) to \(PC\) meets at \(T\). Prove that \( \angle DTQ = 90^\circ \). -/
theorem angle_DTQ_is_right_angle
  (A B C D P Q T : Type) [square A B C D]
  (hP : P ∈ segment A B)
  (hQ : Q ∈ segment B C)
  (hBP : BP = BQ)
  (hT : is_perpendicular B PC)
  : angle D T Q = 90 := 
sorry

end angle_DTQ_is_right_angle_l535_535410


namespace compare_a_b_c_l535_535209

noncomputable def a : ℝ := Real.ln 2
noncomputable def b : ℝ := Real.sqrt Real.pi
noncomputable def c : ℝ := Real.logb (1/2) Real.exp 1

theorem compare_a_b_c : b > a ∧ a > c := 
by
  have ha : 0 < a := Real.ln_pos (by norm_num)
  have ha_le_1 : a < 1 := Real.ln_lt_one_of_lt (by norm_num)
  have hb : b > 1 := Real.sqrt_lt Real.pi (by norm_num)
  have hc : c < 0 := Real.logb_neg (one_div_pos.2 (by norm_num)) (by norm_num)
  exact ⟨hb, lt_trans ha_le_1 hc⟩

end compare_a_b_c_l535_535209


namespace problem_part_i_problem_part_ii_l535_535593

variables {R : Type*} [Field R] (A B C : R × R)

-- Conditions
def is_triangle (A B C : R × R) : Prop :=
A ≠ B ∧ B ≠ C ∧ C ≠ A

def line_through (P Q : R × R) (a b c : R) : Prop :=
a * P.1 + b * P.2 + c = 0 ∧ a * Q.1 + b * Q.2 + c = 0

-- Definitions for this specific problem
def A := (4 : R, 0 : R)
def B := (8 : R, 10 : R)
def C := (0 : R, 6 : R)

-- Problem statements
def eqn_line_through_A_parallel_to_BC : Prop :=
line_through A (0, -8) 1 2 (-8)

def eqn_lines_through_B_equidistant_from_A_and_C : Prop :=
line_through B (8, 10) 3 (-2) (-4) ∧ line_through B (8, 10) 3 2 (-44)

theorem problem_part_i : eqn_line_through_A_parallel_to_BC A B C :=
sorry

theorem problem_part_ii : eqn_lines_through_B_equidistant_from_A_and_C A B C :=
sorry

end problem_part_i_problem_part_ii_l535_535593


namespace angle_degrees_l535_535962

-- Define the conditions
def sides_parallel (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ = θ₂ ∨ (θ₁ + θ₂ = 180)

def angle_relation (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ = 3 * θ₂ - 20 ∨ θ₂ = 3 * θ₁ - 20

-- Statement of the problem
theorem angle_degrees (θ₁ θ₂ : ℝ) (h_parallel : sides_parallel θ₁ θ₂) (h_relation : angle_relation θ₁ θ₂) :
  (θ₁ = 10 ∧ θ₂ = 10) ∨ (θ₁ = 50 ∧ θ₂ = 130) ∨ (θ₁ = 130 ∧ θ₂ = 50) ∨ θ₁ + θ₂ = 180 ∧ (θ₁ = 3 * θ₂ - 20 ∨ θ₂ = 3 * θ₁ - 20) :=
by sorry

end angle_degrees_l535_535962


namespace exists_triangle_3_equal_similar_not_exists_triangle_5_equal_similar_scalene_triangle_unequal_similar_equilateral_triangle_not_unequal_l535_535115

-- Proof Problem 1
theorem exists_triangle_3_equal_similar :
  ∃ (Δ : Type) [is_triangle Δ], 
  ∃ (parts : List Δ),
  parts.length = 3 ∧
  (∀ part ∈ parts, is_similar Δ part) ∧ 
  (∀ part1 part2 ∈ parts, part1 = part2) := sorry

-- Proof Problem 2
theorem not_exists_triangle_5_equal_similar :
  ¬ ∃ (Δ : Type) [is_triangle Δ], 
  ∃ (parts : List Δ),
  parts.length = 5 ∧ 
  (∀ part ∈ parts, is_similar Δ part) ∧ 
  (∀ part1 part2 ∈ parts, part1 = part2) := sorry

-- Proof Problem 3
theorem scalene_triangle_unequal_similar (Δ : Type) [is_triangle Δ] [is_scalene Δ] :
  ∃ (parts : List Δ),
  (∀ part ∈ parts, is_similar Δ part) ∧ 
  (∃ part1 part2 ∈ parts, part1 ≠ part2) := sorry

-- Proof Problem 4
theorem equilateral_triangle_not_unequal (Δ : Type) [is_triangle Δ] [is_equilateral Δ] :
  ¬ ∃ (parts : List Δ),
  (∀ part ∈ parts, is_equilateral part ∧ is_similar Δ part) ∧ 
  (∃ part1 part2 ∈ parts, part1 ≠ part2) := sorry

end exists_triangle_3_equal_similar_not_exists_triangle_5_equal_similar_scalene_triangle_unequal_similar_equilateral_triangle_not_unequal_l535_535115


namespace circulation_along_path_l535_535488

noncomputable def circulation (A : ℝ → ℝ → ℝ × ℝ) (path : ℝ → ℝ × ℝ) : ℝ :=
  ∫ (λ t, (A (path t).1 (path t).2).fst * (path t).1 + (A (path t).1 (path t).2).snd * (path t).2) 0 2 * π

def field (r θ : ℝ) : ℝ × ℝ := (2 * r, (R + r) * sin θ)

def path (φ : ℝ) : ℝ × ℝ := (R, π / 2)

theorem circulation_along_path :
  circulation field path = 4 * π * R ^ 2 :=
sorry

end circulation_along_path_l535_535488


namespace PE_eq_PF_l535_535876

open EuclideanGeometry

-- Definitions and conditions
variables {A B C D E F P O : Point}

-- A cyclic quadrilateral
def isCyclicQuadrilateral (A B C D : Point) : Prop :=
  ∃ (O : Point), isCircle O A B C ∧ isCircle O A D C

-- Diagonal intersection at point P
def diagonalsIntersectAtP (A B C D P : Point) : Prop :=
  intersection (line A C) (line B D) = some P

-- Perpendicular EF to PO with intersections at E and F
def perpendicularEFPO (E F P O : Point) : Prop :=
  perp (line E F) (line P O) ∧ onLine P (line P O) ∧ onLine E (line A B) ∧ onLine F (line C D)

-- Example the statement to be proven
theorem PE_eq_PF (h1 : isCyclicQuadrilateral A B C D)
  (h2 : diagonalsIntersectAtP A B C D P)
  (h3 : perpendicularEFPO E F P O) : dist P E = dist P F :=
sorry

end PE_eq_PF_l535_535876


namespace bookish_campus_reading_l535_535071

-- Definitions of the variables involved in the problem
def reading_hours : List ℝ := [4, 5, 5, 6, 10]

def mean (l: List ℝ) : ℝ := l.sum / l.length

def variance (l: List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

-- The proof statement
theorem bookish_campus_reading :
  mean reading_hours = 6 ∧ variance reading_hours = 4.4 :=
by
  sorry

end bookish_campus_reading_l535_535071


namespace meeting_point_2015_l535_535179

/-- 
A motorist starts at point A, and a cyclist starts at point B. They travel towards each other and 
meet for the first time at point C. After meeting, they turn around and travel back to their starting 
points and continue this pattern of meeting, turning around, and traveling back to their starting points. 
Prove that their 2015th meeting point is at point C.
-/
theorem meeting_point_2015 
  (A B C D : Type) 
  (x y t : ℕ)
  (odd_meeting : ∀ n : ℕ, (2 * n + 1) % 2 = 1) : 
  ∃ n, (n = 2015) → odd_meeting n = 1 → (n % 2 = 1 → (C = "C")) := 
sorry

end meeting_point_2015_l535_535179


namespace train_passing_time_l535_535404

theorem train_passing_time:
  (speed_faster speed_slower distance : ℝ) 
  (km_to_m_per_sec : ℝ) 
  (h1 : speed_faster = 46) 
  (h2 : speed_slower = 36) 
  (h3 : distance = 200) 
  (h4 : km_to_m_per_sec = 5/18) :
  (400 / ((speed_faster - speed_slower) * km_to_m_per_sec)) = 144 :=
by sorry

end train_passing_time_l535_535404


namespace triangle_ABC_properties_l535_535639

theorem triangle_ABC_properties
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 6)
  (h2 : b * Real.sin ((B + C) / 2) = a * Real.sin B)
  (hG : ∃ M, M = "centroid" ∧ (∀ AM, AM = 2 * Real.sqrt 3 → "AM_length")) 
  (hI : ∃ M, M = "incenter" ∧ (∀ AD, AD = 3 * Real.sqrt 3 → "AD_length")) :
    A = Real.pi / 3 ∧ 1 / 2 * b * c * Real.sin A = 9 * Real.sqrt 3 := by 
  sorry

end triangle_ABC_properties_l535_535639


namespace total_amount_is_69_l535_535807

-- Define the total amount paid for the work as X
def total_amount_paid (X : ℝ) : Prop := 
  let B_payment := 12
  let portion_of_B_work := 4 / 23
  (portion_of_B_work * X = B_payment)

theorem total_amount_is_69 : ∃ X : ℝ, total_amount_paid X ∧ X = 69 := by
  sorry

end total_amount_is_69_l535_535807


namespace probability_at_0_2_after_4_moves_eq_l535_535795

noncomputable def probability_upwards : ℝ := 1 / 4

noncomputable def total_moves : ℕ := 4

noncomputable def moves_up : ℕ := 3

noncomputable def moves_down : ℕ := 1 -- Because 3 movements upwards and 1 downwards make total 4 moves.

theorem probability_at_0_2_after_4_moves_eq :
  let p := probability_upwards in 
  let n := total_moves in 
  let k_up := moves_up in 
  let k_down := moves_down in 
  (n.choose k_up) * (p ^ k_up) * ((1 - p) ^ k_down) = 3 / 64 :=
by
  sorry

end probability_at_0_2_after_4_moves_eq_l535_535795


namespace area_inner_quadrilateral_eq_ninth_l535_535885

open Set

-- Define a convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  convex_hull ℝ {A, B, C, D} = {p | p ∈ affine_span ℝ {A, B, C, D}}

-- Main statement to be proved
theorem area_inner_quadrilateral_eq_ninth (A B C D : ℝ × ℝ)
  (h : is_convex_quadrilateral A B C D) :
  ∃ E1 E2 F1 F2 G1 G2 H1 H2 : ℝ × ℝ,
    (E1 ∈ line_map A B '' (Icc (1/3) (2/3))) ∧ 
    (E2 ∈ line_map A B '' (Icc (2/3) 1)) ∧ 
    (F1 ∈ line_map B C '' (Icc (1/3) (2/3))) ∧ 
    (F2 ∈ line_map B C '' (Icc (2/3) 1)) ∧
    (G1 ∈ line_map C D '' (Icc (1/3) (2/3))) ∧ 
    (G2 ∈ line_map C D '' (Icc (2/3) 1)) ∧
    (H1 ∈ line_map D A '' (Icc (1/3) (2/3))) ∧ 
    (H2 ∈ line_map D A '' (Icc (2/3) 1)) ∧
    let PQRS_area := convex_hull ℝ {intersect_lines (A, E1) (C, G1), intersect_lines (A, E2) (C, G2),
                                       intersect_lines (B, F1) (D, H1), intersect_lines (B, F2) (D, H2)} in
    (area PQRS_area / area (convex_hull ℝ {A, B, C, D}) = 1 / 9) :=
sorry

end area_inner_quadrilateral_eq_ninth_l535_535885


namespace relationship_of_s_and_t_l535_535806

variables (a b c : ℝ)
def area (a b c : ℝ) := 1 / 4
def circumradius (a b c : ℝ) := 1
def s (a b c : ℝ) := real.sqrt a + real.sqrt b + real.sqrt c
def t (a b c : ℝ) := 1 / a + 1 / b + 1 / c

theorem relationship_of_s_and_t 
  (h_area : area a b c = 1 / 4) 
  (h_circumradius : circumradius a b c = 1) : 
  s a b c < t a b c := 
sorry

end relationship_of_s_and_t_l535_535806


namespace A_2010_eq_1_l535_535191

-- Definitions and conditions
def seq (n : ℕ) : ℚ
| 0     := 3
| (n+1) := (seq n - 1) / seq n

def A_n (n : ℕ) : ℚ :=
  (Finset.range n).prod (λ k, seq k)

-- Statement of the proof problem
theorem A_2010_eq_1 : A_n 2010 = 1 := sorry

end A_2010_eq_1_l535_535191


namespace file_size_l535_535995

-- Definitions based on conditions
def upload_speed : ℕ := 8 -- megabytes per minute
def upload_time : ℕ := 20 -- minutes

-- Goal to prove
theorem file_size:
  (upload_speed * upload_time = 160) :=
by sorry

end file_size_l535_535995


namespace number_of_1s_and_2s_l535_535060

theorem number_of_1s_and_2s {N : ℕ} (digits : Fin N → Fin 2) 
  (h_len : 100 = N) 
  (h_ev_twos : ∀ i j : Fin N, digits i = 1 → digits j = 1 → even (|i - j|))
  (h_sum : (digits.sum : ℕ) % 3 = 0) :
  ∃ n1 n2 : ℕ, n1 + n2 = 100 ∧  n1 = 98 ∧ n2 = 2 :=
by 
  sorry

end number_of_1s_and_2s_l535_535060


namespace probability_neither_prime_nor_composite_l535_535959

open Real

-- Define the set of numbers from 1 to 95
def numbers : Set ℕ := {n | 1 ≤ n ∧ n ≤ 95}

-- Define the predicate for a number being neither prime nor composite
def is_neither_prime_nor_composite (n : ℕ) : Prop :=
  n = 1

-- Define the subset of numbers that are neither prime nor composite
def neither_prime_nor_composite : Set ℕ := {n ∈ numbers | is_neither_prime_nor_composite n}

-- The main theorem stating the probability
theorem probability_neither_prime_nor_composite :
  (neither_prime_nor_composite.to_finset.card : ℝ) / (numbers.to_finset.card : ℝ) = 1 / 95 :=
by
  -- Proof is omitted
  sorry

end probability_neither_prime_nor_composite_l535_535959


namespace measure_angle_C_perimeter_triangle_l535_535267

-- Definitions
variable (a b c : ℝ)
variable (A B C : ℝ)  -- angles in radians

-- Acute triangle with specific relation
variable (h1 : 0 < A ∧ A < π/2)
variable (h2 : 0 < B ∧ B < π/2)
variable (h3 : 0 < C ∧ C < π/2)
variable (h4 : a = c*sin A*2/sqrt 3)  -- Given condition

-- Part 1: Prove the measure of angle C is π/3
theorem measure_angle_C (h4 : a = 2 * c * sin A / sqrt 3): C = π/3 :=
sorry

-- Additional conditions for Part 2
variable (h5 : c = sqrt 7)
variable (h6 : a * b = 6)

-- Part 2: Prove the perimeter of the triangle
theorem perimeter_triangle (h4 : a = 2 * c * sin A / sqrt 3)
                           (h5 : c = sqrt 7)
                           (h6 : a * b = 6) :
  a + b + c = 5 + sqrt 7 :=
sorry

end measure_angle_C_perimeter_triangle_l535_535267


namespace odd_binomial_coeff_count_l535_535692

theorem odd_binomial_coeff_count (n : ℕ) : 
  (finset.sum (finset.range (2^n + 1)) 
              (λ u, finset.sum (finset.range (u + 1)) 
                              (λ v, if (nat.choose u v) % 2 = 1 then 1 else 0))) = 3^n :=
  sorry

end odd_binomial_coeff_count_l535_535692


namespace dot_product_self_eq_nine_l535_535248

-- Definitions
variables (u : ℝ^3) (norm_u : ‖u‖ = 3)

-- The theorem statement
theorem dot_product_self_eq_nine : u • u = 9 :=
by
  sorry

end dot_product_self_eq_nine_l535_535248


namespace length_of_other_train_l535_535453

-- Defining constants and conditions
def length_first_train : ℝ := 220
def speed_first_train_kmh : ℝ := 120
def speed_second_train_kmh : ℝ := 80
def crossing_time : ℝ := 9
def speed_conversion_factor : ℝ := 1000 / 3600 -- conversion from km/h to m/s

-- Convert speeds from km/h to m/s
def speed_first_train_ms : ℝ := speed_first_train_kmh * speed_conversion_factor
def speed_second_train_ms : ℝ := speed_second_train_kmh * speed_conversion_factor

-- Calculate relative speed
def relative_speed : ℝ := speed_first_train_ms + speed_second_train_ms

-- Calculate total distance covered in the given crossing time
def total_distance : ℝ := relative_speed * crossing_time

-- Define the length of the first train
def length_of_first_train : ℝ := 220

-- Prove the length of the other train
theorem length_of_other_train : 
  220 + 279.95 = 220 + (relative_speed * crossing_time - 220) :=
by
  -- We should prove the theorem here.
  sorry

end length_of_other_train_l535_535453


namespace polyhedron_volume_correct_l535_535983

-- Definitions and conditions from the problem
def is_isosceles_right_triangle (x : ℝ) (y : ℝ) (z : ℝ) : Prop :=
  (x = y ∧ x^2 + y^2 = z^2)

def is_rectangle (x : ℝ) (y : ℝ) : Prop :=
  (x > 0 ∧ y > 0)

def right_triangle_area (b : ℝ) (h : ℝ) : ℝ :=
  (1/2) * b * h

-- Given conditions
def A_isosceles_right : Prop := is_isosceles_right_triangle 1 1 (real.sqrt 2)
def E_isosceles_right : Prop := is_isosceles_right_triangle 1 1 (real.sqrt 2)
def F_isosceles_right : Prop := is_isosceles_right_triangle 1 1 (real.sqrt 2)

def B_rectangle : Prop := is_rectangle 1 2
def C_rectangle : Prop := is_rectangle 1 2
def D_rectangle : Prop := is_rectangle 1 2

def G_right_triangle : Prop := is_isosceles_right_triangle 2 1 (real.sqrt 5)

-- Volume calculations
def prism_volume : ℝ := 1 * 2 * 2

def tetrahedron_volume : ℝ :=
  (1/3) * right_triangle_area 2 1 * 2

def polyhedron_volume : ℝ := prism_volume - tetrahedron_volume

-- Theorem statement
theorem polyhedron_volume_correct : 
  A_isosceles_right ∧ E_isosceles_right ∧ F_isosceles_right ∧
  B_rectangle ∧ C_rectangle ∧ D_rectangle ∧ G_right_triangle →
  polyhedron_volume = 10 / 3 :=
by run_cmd sorry

end polyhedron_volume_correct_l535_535983


namespace matt_current_age_is_65_l535_535045

variable (matt_age james_age : ℕ)

def james_current_age := 30
def james_age_in_5_years := james_current_age + 5
def matt_age_in_5_years := 2 * james_age_in_5_years
def matt_current_age := matt_age_in_5_years - 5

theorem matt_current_age_is_65 : matt_current_age = 65 := 
by
  -- sorry is here to skip the proof.
  sorry

end matt_current_age_is_65_l535_535045


namespace log_sequence_product_eq_eight_l535_535514

-- Define the product of interest
def log_sequence_product : ℝ :=
  (List.range (256 - 2)).map (λ n, Real.log (n + 3) / Real.log (n + 2)).prod

-- State the theorem to be proven
theorem log_sequence_product_eq_eight : log_sequence_product = 8 := by
  sorry

end log_sequence_product_eq_eight_l535_535514


namespace radius_of_larger_circle_is_25_over_3_l535_535403

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := (5 / 2) * r 

theorem radius_of_larger_circle_is_25_over_3
  (rAB rBD : ℝ)
  (h_ratio : 2 * rBD = 5 * rBD / 2)
  (h_ab : rAB = 8)
  (h_tangent : ∀ rBD, (5 * rBD / 2 - 8) ^ 2 = 64 + rBD ^ 2) :
  radius_of_larger_circle (10 / 3) = 25 / 3 :=
  by
  sorry

end radius_of_larger_circle_is_25_over_3_l535_535403


namespace proof_ellipse_equation_proof_existence_of_k_l535_535908

noncomputable def ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  (∃ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) ∧ 
  (b = 1) ∧ 
  (a^2 - 1) / a^2 = (√6 / 3)^2 ∧ 
  a^2 = 3

noncomputable def existence_of_k (k : ℝ) (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
  (∃ k : ℝ, 
    (y1 = k * x1 + 2) ∧ 
    (y2 = k * x2 + 2) ∧ 
    (1 + 3 * k^2) * x1^2 + 12 * k * x1 * x2 + 9 = 0 ∧
    ((6 * k^2 - 6) > 0) ∧ 
    k ≠ 0 ∧
    (1 + k^2) * (x1 * x2) + (2 * k - 1) * (x1 + x2) + 5 = 0 ∧
    k = -7 / 6
  )

theorem proof_ellipse_equation : ∃ a b : ℝ, a > b ∧ b > 0 ∧ ellipse_equation a b sorry sorry := sorry

theorem proof_existence_of_k : ∃ k : ℝ, k ≠ 0 ∧ -7 / 6 = k ∧ 
  existence_of_k k (√3) 1 sorry sorry := sorry

end proof_ellipse_equation_proof_existence_of_k_l535_535908


namespace base_unit_digit_l535_535139

def unit_digit (n : ℕ) : ℕ := n % 10

theorem base_unit_digit (x : ℕ) :
  unit_digit ((x^41) * (41^14) * (14^87) * (87^76)) = 4 →
  unit_digit x = 1 :=
by
  sorry

end base_unit_digit_l535_535139


namespace exponential_increasing_exponential_decreasing_l535_535337

-- Define the exponential function
def exponential (a : ℝ) (x : ℝ) : ℝ := a^x

-- Prove that the function y = a^x is increasing for a > 1
theorem exponential_increasing (a : ℝ) (x : ℝ) (h : 1 < a) : 
    (0 < exp (log a * x)) := 
by {
    sorry
}

-- Prove that the function y = a^x is decreasing for 0 < a < 1
theorem exponential_decreasing (a : ℝ) (x : ℝ) (h : 0 < a) (h' : a < 1) : 
    (0 < exp (log a * x) -> 0 > log a) := 
by {
    sorry
}

end exponential_increasing_exponential_decreasing_l535_535337


namespace remaining_episodes_for_all_series_l535_535760

def total_episodes (seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  seasons * episodes_per_season

def watched_episodes (total : ℕ) (fraction_watched : ℚ) : ℕ :=
  (fraction_watched * total).to_nat

def remaining_episodes (total : ℕ) (watched : ℕ) : ℕ :=
  total - watched

theorem remaining_episodes_for_all_series :
  let total1 := total_episodes 18 25;
  let total2 := total_episodes 10 30;
  let total3 := total_episodes 8 20;
  let watched1 := watched_episodes total1 (2/5 : ℚ);
  let watched2 := watched_episodes total2 (3/5 : ℚ);
  let watched3 := watched_episodes total3 (1/4 : ℚ);
  remaining_episodes total1 watched1 +
  remaining_episodes total2 watched2 +
  remaining_episodes total3 watched3 = 510 :=
by sorry

end remaining_episodes_for_all_series_l535_535760


namespace union_M_N_l535_535890

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by
  sorry

end union_M_N_l535_535890


namespace anna_chocolates_l535_535088

theorem anna_chocolates : ∃ (n : ℕ), (5 * 2^(n-1) > 200) ∧ n = 7 :=
by
  sorry

end anna_chocolates_l535_535088


namespace terminating_decimal_count_l535_535146

theorem terminating_decimal_count : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = 49 * k)}.card = 10 :=
by
  sorry

end terminating_decimal_count_l535_535146


namespace _l535_535551

noncomputable def pentagon_theorem (A B C D E : Point)
  (hConvex : ConvexPentagon A B C D E)
  (hEqualSides : AB = BC ∧ BC = CD)
  (hAngles1 : ∠EAB = ∠BCD)
  (hAngles2 : ∠EDC = ∠CBA) :
  ∃ H : Point, PerpendicularFrom E BC ∧ Intersects H AC ∧ Intersects H BD := 
sorry

end _l535_535551


namespace not_p_sufficient_not_necessary_q_l535_535927

variable (m : ℝ)
def p := m ≥ 1/4
def q := ∃ x : ℝ, x^2 + x + m = 0

theorem not_p_sufficient_not_necessary_q : (¬ p) ↔ (sufficient_condition (¬ p) q ∧ ¬ necessary_condition (¬ p) q) :=
by
  sorry

end not_p_sufficient_not_necessary_q_l535_535927


namespace find_tabitha_age_l535_535124

-- Define the conditions
variable (age_started : ℕ) (colors_started : ℕ) (years_future : ℕ) (future_colors : ℕ)

-- Let's specify the given problem's conditions:
axiom h1 : age_started = 15          -- Tabitha started at age 15
axiom h2 : colors_started = 2        -- with 2 colors
axiom h3 : years_future = 3          -- in three years
axiom h4 : future_colors = 8         -- she will have 8 different colors

-- The proof problem we need to state:
theorem find_tabitha_age : ∃ age_now : ℕ, age_now = age_started + (future_colors - colors_started) - years_future := by
  sorry

end find_tabitha_age_l535_535124


namespace matt_current_age_is_65_l535_535046

variable (matt_age james_age : ℕ)

def james_current_age := 30
def james_age_in_5_years := james_current_age + 5
def matt_age_in_5_years := 2 * james_age_in_5_years
def matt_current_age := matt_age_in_5_years - 5

theorem matt_current_age_is_65 : matt_current_age = 65 := 
by
  -- sorry is here to skip the proof.
  sorry

end matt_current_age_is_65_l535_535046


namespace smallest_root_of_g_l535_535138

noncomputable def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem smallest_root_of_g : ∀ x : ℝ, g x = 0 → x = - Real.sqrt (3 / 7) :=
by
  sorry

end smallest_root_of_g_l535_535138


namespace limit_sin_x_minus_x_over_x3_limit_ln_1_plus_x_minus_x_plus_x2_over_2_over_x3_l535_535406

-- Part (a)
theorem limit_sin_x_minus_x_over_x3 :
  tendsto (λ x : ℝ, (sin x - x) / x^3) (𝓝 0) (𝓝 (-1 / 6)) :=
sorry

-- Part (b) 
theorem limit_ln_1_plus_x_minus_x_plus_x2_over_2_over_x3 :
  tendsto (λ x : ℝ, (log (1 + x) - x + x^2 / 2) / x^3) (𝓝 0) (𝓝 (1 / 3)) :=
sorry

end limit_sin_x_minus_x_over_x3_limit_ln_1_plus_x_minus_x_plus_x2_over_2_over_x3_l535_535406


namespace sequence_1005th_term_l535_535495

-- Definitions based on conditions
def first_term : ℚ := sorry
def second_term : ℚ := 10
def third_term : ℚ := 4 * first_term - (1:ℚ)
def fourth_term : ℚ := 4 * first_term + (1:ℚ)

-- Common difference
def common_difference : ℚ := (fourth_term - third_term)

-- Arithmetic sequence term calculation
def nth_term (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n-1) * d

-- Theorem statement
theorem sequence_1005th_term : nth_term first_term common_difference 1005 = 5480 := sorry

end sequence_1005th_term_l535_535495


namespace pigeonhole_principle_divisible_non_empty_subset_divisible_l535_535770

-- Problem 1 Statement
theorem pigeonhole_principle_divisible (n : ℕ) (a : Fin (n + 1) → ℤ) : 
  ∃ (i j : Fin (n + 1)), i ≠ j ∧ (a i - a j) % n = 0 := 
sorry

-- Problem 2 Statement
theorem non_empty_subset_divisible (n : ℕ) (S : Fin n → ℤ) :
  ∃ (t : Finset (Fin n)), t.nonempty ∧ (∑ i in t, S i) % n = 0 :=
sorry

end pigeonhole_principle_divisible_non_empty_subset_divisible_l535_535770


namespace increasing_even_func_l535_535082

open Real

theorem increasing_even_func :
  ∃ (f : ℝ → ℝ), f = λ x, |sin x| ∧ (∀ x y, 0 < x ∧ x < y ∧ y < π / 2 → f x < f y) ∧ (∀ x, f x = f (-x)) :=
by
  existsi (λ x : ℝ, abs (sin x))
  split
  . refl
  . split
  . intros x y hx
    cases hx with hx1 hy
    apply sin_lt_sin
    linarith
    all_goals
    linarith
  . intros x
    dsimp
    rw abs_neg
    sorry

end increasing_even_func_l535_535082


namespace terminating_decimal_count_l535_535147

theorem terminating_decimal_count : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = 49 * k)}.card = 10 :=
by
  sorry

end terminating_decimal_count_l535_535147


namespace triangle_ABC_properties_l535_535288

theorem triangle_ABC_properties
  (A B C : ℝ) (a b c : ℝ)
  (h_a_lt_b : a < b)
  (h_b_lt_c : b < c)
  (h_sin_A : Real.sin A = (Real.sqrt 3 * a) / (2 * b))
  (h_a_value : a = 2)
  (h_b_value : b = Real.sqrt 7) :
  B = Real.pi / 3 ∧
  c = 3 ∧
  let area := (1 / 2) * a * c * Real.sin B in
  area = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end triangle_ABC_properties_l535_535288


namespace square_into_three_identical_squares_l535_535497

theorem square_into_three_identical_squares (a s : ℝ) :
  (s = sqrt 3 * a) ↔ (s ^ 2 = 3 * a ^ 2) := 
sorry

end square_into_three_identical_squares_l535_535497


namespace num_pos_integers_with_nonzero_hundredths_l535_535540

theorem num_pos_integers_with_nonzero_hundredths :
  ∃ S : finset ℕ, (∀ n ∈ S, (0 < n ∧ n ≤ 100) ∧ ∃ k : ℕ, ∃ m : ℕ, n = 2^k * 5^m) ∧
  (∀ n ∈ S, (n = 1 ∨ n = 2 ∨ n = 5 ∨ n = 10) → false) ∧
  S.card = 11 :=
sorry

end num_pos_integers_with_nonzero_hundredths_l535_535540


namespace max_abs_sum_l535_535642

theorem max_abs_sum (a b c : ℝ) (h : ∀ x, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  |a| + |b| + |c| ≤ 3 :=
sorry

end max_abs_sum_l535_535642


namespace simplify_expression_l535_535351

theorem simplify_expression : 
  let x := 2
  let y := -1 / 2
  (2 * x^2 + (-x^2 - 2 * x * y + 2 * y^2) - 3 * (x^2 - x * y + 2 * y^2)) = -10 := by
  sorry

end simplify_expression_l535_535351


namespace parabola_with_focus_at_circle_center_and_vertex_origin_l535_535530

theorem parabola_with_focus_at_circle_center_and_vertex_origin :
  (∃ (p : ℝ), p = 2 ∧ ∀ (x y : ℝ), y^2 = 4 * p * x) :=
begin
  sorry
end

end parabola_with_focus_at_circle_center_and_vertex_origin_l535_535530


namespace meeting_2015th_at_C_l535_535161

-- Conditions Definitions
variable (A B C D P : Type)
variable (x y t : ℝ)  -- speeds and starting time difference
variable (mw cyclist : ℝ → ℝ)  -- paths of motorist and cyclist

-- Proof statement
theorem meeting_2015th_at_C 
(Given_meeting_pattern: ∀ n : ℕ, odd n → (mw (n * (x + y))) = C):
  (mw (2015 * (x + y))) = C := 
by 
  sorry  -- Proof omitted

end meeting_2015th_at_C_l535_535161


namespace count_distinct_four_digit_numbers_divisible_by_3_ending_in_45_l535_535240

/--
Prove that the total number of distinct four-digit numbers that end with 45 and 
are divisible by 3 is 27.
-/
theorem count_distinct_four_digit_numbers_divisible_by_3_ending_in_45 :
  ∃ n : ℕ, n = 27 ∧ 
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → 
  (∃ k : ℕ, a + b + 9 = 3 * k) → 
  (10 * (10 * a + b) + 45) = 1000 * a + 100 * b + 45 → 
  1000 * a + 100 * b + 45 = n := sorry

end count_distinct_four_digit_numbers_divisible_by_3_ending_in_45_l535_535240


namespace johnny_paths_unique_l535_535331

-- Defining points in the set S
def points (i j : ℕ) : Prop := (i ≤ 1) ∧ (j ≤ 5)

-- Defining Johnny's movement constraints
def valid_point (p: ℕ × ℕ) : Prop := 
  ∃ i j, p = (i, j) ∧ points i j

def path_crosses_itself (path: List (ℕ × ℕ)) : Prop := 
  ∃ i j, i ≠ j ∧ path.nth i = path.nth j

-- The main problem statement
theorem johnny_paths_unique :
  ∃ n, n = 252 ∧ 
  (∀ (path: List (ℕ × ℕ)),
    path.length = 12 ∧ -- Johnny must visit each point
    path.head = (0, 0) ∧ -- Starting point
    path.last = (5, 1) ∧ -- Ending point
    ∀ p ∈ path, valid_point p ∧ -- All points in path must be valid points in S
    ¬(path_crosses_itself path) -- Path must not cross itself
  ) →
  (path_combinations (List (ℕ × _)) n = 252) :=
sorry

end johnny_paths_unique_l535_535331


namespace total_points_scored_l535_535116

-- Definitions based on the conditions
def three_point_shots := 13
def two_point_shots := 20
def free_throws := 5
def missed_free_throws := 2
def points_per_three_point_shot := 3
def points_per_two_point_shot := 2
def points_per_free_throw := 1
def penalty_per_missed_free_throw := 1

-- Main statement proving the total points James scored
theorem total_points_scored :
  three_point_shots * points_per_three_point_shot +
  two_point_shots * points_per_two_point_shot +
  free_throws * points_per_free_throw -
  missed_free_throws * penalty_per_missed_free_throw = 82 :=
by
  sorry

end total_points_scored_l535_535116


namespace impossible_to_arrange_1000_segments_l535_535988

theorem impossible_to_arrange_1000_segments :
  ∀ (segments : fin 1000 → set (ℝ × ℝ)), 
  (∀ i, ∃ j ≠ i, ∃ k ≠ i, (segments i).endpoints ⊆ set.interior (segments j) ∧ (segments i).endpoints ⊆ set.interior (segments k)) → False := 
by
  sorry

end impossible_to_arrange_1000_segments_l535_535988


namespace probability_different_cars_l535_535721

theorem probability_different_cars : 
  let cars := {A, B, C} in
  let choices := cars × cars in
  let different_choices := { (a, b) | (a, b) ∈ choices ∧ a ≠ b } in
  (Finset.card different_choices : ℚ) / (Finset.card choices : ℚ) = 2 / 3 :=
by
  sorry

end probability_different_cars_l535_535721


namespace monotonic_intervals_extreme_values_on_interval_l535_535921

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

theorem monotonic_intervals:
  (∀ x : ℝ, x < -1 → 0 < (f' x)) ∧ 
  (∀ x : ℝ, x > 1 → 0 < (f' x)) ∧ 
  (∀ x : ℝ, -1 < x ∧ x < 1 → (f' x) < 0) := 
by {
  sorry
}

theorem extreme_values_on_interval : 
  (∃ x_min x_max, -3 ≤ x_min ∧ x_min ≤ 3 ∧ -3 ≤ x_max ∧ x_max ≤ 3 ∧ 
   f x_min = -49 ∧ f x_max = 59) :=
by {
  exists 3, -3,
  sorry
}

end monotonic_intervals_extreme_values_on_interval_l535_535921


namespace minimum_value_of_expression_l535_535195

noncomputable def monotonic_function_property
    (f : ℝ → ℝ)
    (h_monotonic : ∀ x y, (x ≤ y → f x ≤ f y) ∨ (x ≥ y → f x ≥ f y))
    (h_additive : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
    (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : f a + f (2 * b - 1) = 0): Prop :=
    (1 : ℝ) / a + 8 / b = 25

theorem minimum_value_of_expression 
    (f : ℝ → ℝ)
    (h_monotonic : ∀ x y, (x ≤ y → f x ≤ f y) ∨ (x ≥ y → f x ≥ f y))
    (h_additive : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
    (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : f a + f (2 * b - 1) = 0) :
    (1 : ℝ) / a + 8 / b = 25 := 
sorry

end minimum_value_of_expression_l535_535195


namespace triangle_area_interval_l535_535805

theorem triangle_area_interval (s : ℝ) :
  10 ≤ (s - 1)^(3 / 2) ∧ (s - 1)^(3 / 2) ≤ 50 → (5.64 ≤ s ∧ s ≤ 18.32) :=
by
  sorry

end triangle_area_interval_l535_535805


namespace min_a2_k2b2_l535_535659

variable (a b t k : ℝ)
variable (hk : 0 < k)
variable (h : a + k * b = t)

theorem min_a2_k2b2 (a b t k : ℝ) (hk : 0 < k) (h : a + k * b = t) :
  a^2 + (k * b)^2 ≥ (1 + k^2) * (t^2) / ((1 + k)^2) :=
sorry

end min_a2_k2b2_l535_535659


namespace find_a_l535_535366

theorem find_a (a : ℝ) (a_pos : 0 < a + 1) (hyperbola_foci_eq_ellipse_foci : 
    ∀ x y : ℝ, (x^2 / (a+1) - y^2 = 1) ∧ (x^2 / 4 + y^2 / a^2 = 1)) : 
  a = 1 := 
begin
  sorry
end

end find_a_l535_535366


namespace area_of_triangle_l535_535656

open Matrix

def a : Matrix (Fin 2) (Fin 1) ℤ := ![![4], ![-1]]
def b : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![5]]

theorem area_of_triangle : (abs (a 0 0 * b 1 0 - a 1 0 * b 0 0) : ℚ) / 2 = 23 / 2 :=
by
  -- To be proved (using :ℚ for the cast to rational for division)
  sorry

end area_of_triangle_l535_535656


namespace meeting_2015th_at_C_l535_535160

-- Conditions Definitions
variable (A B C D P : Type)
variable (x y t : ℝ)  -- speeds and starting time difference
variable (mw cyclist : ℝ → ℝ)  -- paths of motorist and cyclist

-- Proof statement
theorem meeting_2015th_at_C 
(Given_meeting_pattern: ∀ n : ℕ, odd n → (mw (n * (x + y))) = C):
  (mw (2015 * (x + y))) = C := 
by 
  sorry  -- Proof omitted

end meeting_2015th_at_C_l535_535160


namespace smallest_circle_area_l535_535669

theorem smallest_circle_area (A B C : Point) (CA CB AB : ℝ)
  (h1 : dist C A = 5) (h2 : dist C B = 5) (h3 : dist A B = 8) :
  let ω : Circle := circumscribed_circle A B C in
  4 * π = 16 * π :=
sorry

end smallest_circle_area_l535_535669


namespace dylan_ice_cubes_l535_535117

-- Definitions based on conditions
def trays := 2
def spaces_per_tray := 12
def total_tray_ice := trays * spaces_per_tray
def pitcher_multiplier := 2

-- The statement to be proven
theorem dylan_ice_cubes (x : ℕ) : x + pitcher_multiplier * x = total_tray_ice → x = 8 :=
by {
  sorry
}

end dylan_ice_cubes_l535_535117


namespace meeting_point_2015th_l535_535163

-- Definitions for the conditions
def motorist_speed (x : ℝ) := x
def cyclist_speed (y : ℝ) := y
def initial_delay (t : ℝ) := t 
def first_meeting_point := C
def second_meeting_point := D

-- The main proof problem statement
theorem meeting_point_2015th
  (x y t : ℝ) -- speeds of the motorist and cyclist and the initial delay
  (C D : Point) -- points C and D on the segment AB where meetings occur
  (pattern_alternation : ∀ n: ℤ, n > 0 → ((n % 2 = 1) → n-th_meeting_point = C) ∧ ((n % 2 = 0) → n-th_meeting_point = D))
  (P_A_B_cycle : ∀ n: ℕ, (P → A ∨ B → C ∨ A → B ∨ D → P) holds for each meeting): 
  2015-th_meeting_point = C :=
by
  sorry

end meeting_point_2015th_l535_535163


namespace tangent_lengths_l535_535745

noncomputable def internal_tangent_length (r1 r2 d : ℝ) : ℝ :=
  Real.sqrt (d^2 - (r1 + r2)^2)

noncomputable def external_tangent_length (r1 r2 d : ℝ) : ℝ :=
  Real.sqrt (d^2 - (r1 - r2)^2)

theorem tangent_lengths (r1 r2 d : ℝ) (h_r1 : r1 = 8) (h_r2 : r2 = 10) (h_d : d = 50) :
  internal_tangent_length r1 r2 d = 46.67 ∧ external_tangent_length r1 r2 d = 49.96 :=
by
  sorry

end tangent_lengths_l535_535745


namespace constant_term_in_expansion_l535_535282

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem constant_term_in_expansion :
  let poly := (2 * (x^2) - (1 / x))^6
  let T (r : ℕ) := ((-1)^r) * (2^(6-r)) * (binomial_coefficient 6 r) * (x^(12 - 3*r))
  let const_term := T 4
  const_term = 60 := by
sorry

end constant_term_in_expansion_l535_535282


namespace correct_propositions_count_l535_535689

theorem correct_propositions_count (a b c : ℝ) :
  (a < b → a + c < b + c) ∧
  (a + c < b + c → a < b) ∧
  (a ≥ b → a + c ≥ b + c) ∧
  (a + c ≥ b + c → a ≥ b) :=
begin
  split,
  { -- Prove the original proposition
    intros h,
    linarith, },
  split,
  { -- Prove the converse
    intros h,
    linarith, },
  split,
  { -- Prove the inverse
    intros h,
    linarith, },
  { -- Prove the contrapositive
    intros h,
    linarith, },
end

end correct_propositions_count_l535_535689


namespace geom_seq_common_ratio_l535_535221

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem geom_seq_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (H_geo : ∀ n : ℕ, a (n + 1) = a n * q)
  (H_inc : ∀ m n : ℕ, m < n → a m < a n)
  (H_pos : a 1 > 0)
  (H_eq : 2 * (a 3 + a 5) = 5 * a 4) :
  q = 2 :=
sorry

end geom_seq_common_ratio_l535_535221


namespace independence_test_confidence_l535_535284

theorem independence_test_confidence 
  (X Y : Type) (H0 : ¬ R X Y) (P_K2_ge_6_635 : P(K^2 ≥ 6.635) ≈ 0.01) :
  (1 - 0.01 > 0.99) ∧ (R X Y) :=
sorry

end independence_test_confidence_l535_535284


namespace locus_of_N_l535_535091

theorem locus_of_N
  (F : Point)
  (l : Line)
  (p : ℝ)
  (h_pos : p > 0)
  (M : Point)
  (h_M_on_l : M ∈ l)
  (N : Point)
  (h_N_on_MF_extension : ∃ k : ℝ, k > 1 ∧ N = M + k * (F - M))
  (h_condition : dist F N / dist M N = 1 / dist M F) :
  let e := 1 / p in
  if h_e_gt_one : e > 1 then
    ∃ l', N = hyperbola_right_of (vertical_line l')
  else if h_e_eq_one : e = 1 then
    ∃ l', N = parabola_right_of (vertical_line l')
  else if h_e_lt_one : 0 < e ∧ e < 1 then
    ∃ l', N = ellipse_right_of (vertical_line l')
  else false :=
by
  sorry

end locus_of_N_l535_535091


namespace win_probability_l535_535376

theorem win_probability (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose = 3 / 8) :=
by
  -- Provide the proof here if needed, but skip it
  sorry

end win_probability_l535_535376


namespace machine_input_number_l535_535604

theorem machine_input_number (x : ℤ) :
  (x + 15 - 6 = 35) → x = 26 :=
begin
  intro h,
  sorry,
end

end machine_input_number_l535_535604


namespace f_monotonic_intervals_f_max_min_on_interval_l535_535919

-- Define the function
def f (x : ℝ) : ℝ := 3 * x ^ 3 - 9 * x + 5

-- Statement for Part 1: Monotonic Intervals
theorem f_monotonic_intervals : 
  (∀ x, x < -1 → monotone_increasing f x) ∧
  (∀ x, x > 1 → monotone_increasing f x) ∧
  (∀ x, -1 < x ∧ x < 1 → monotone_decreasing f x) := 
sorry

-- Statement for Part 2: Maximum and Minimum Values on [-3, 3]
theorem f_max_min_on_interval : 
  is_max_on f 3 (-3, 3) ∧ 
  ∀ x, x ∈ [-3, 3] → f x ≤ 59 ∧
  is_min_on f (-3) (-3, 3) ∧ 
  ∀ x, x ∈ [-3, 3] → f x ≥ -49 := 
sorry

end f_monotonic_intervals_f_max_min_on_interval_l535_535919


namespace std_dev_of_normal_distribution_l535_535704

theorem std_dev_of_normal_distribution (μ σ : ℝ) (h1: μ = 14.5) (h2: μ - 2 * σ = 11.5) : σ = 1.5 := 
by 
  sorry

end std_dev_of_normal_distribution_l535_535704


namespace translate_cos_to_sin_l535_535006

def translate_cos_to_sin_eq_sin_shifted (x : ℝ) : Prop :=
  let cos_fn : ℝ → ℝ := λ x, Real.cos (2*x - Real.pi / 4)
  let sin_fn : ℝ → ℝ := λ x, Real.sin (2*x)
  ∀ x, cos_fn x = sin_fn (x + Real.pi / 8)

theorem translate_cos_to_sin :
  translate_cos_to_sin_eq_sin_shifted :=
by
  sorry

end translate_cos_to_sin_l535_535006


namespace square_based_pyramid_edges_l535_535458

-- Define a structure for a square-based pyramid
structure SquareBasedPyramid :=
  (base_edges : ℕ)
  (apex_to_base_edges : ℕ)
  (total_edges : ℕ)

-- Assume the conditions provided
axiom base_edges_count : SquareBasedPyramid → base_edges = 4
axiom apex_to_base_edges_count : SquareBasedPyramid → apex_to_base_edges = 4

-- The theorem statement to prove
theorem square_based_pyramid_edges : ∀ (p : SquareBasedPyramid),
  p.total_edges = p.base_edges + p.apex_to_base_edges → 
  p.total_edges = 8 :=
by
  intro p
  have h_base : p.base_edges = 4 := base_edges_count p
  have h_apex : p.apex_to_base_edges = 4 := apex_to_base_edges_count p
  sorry -- Proof goes here

end square_based_pyramid_edges_l535_535458


namespace meeting_point_2015_l535_535178

/-- 
A motorist starts at point A, and a cyclist starts at point B. They travel towards each other and 
meet for the first time at point C. After meeting, they turn around and travel back to their starting 
points and continue this pattern of meeting, turning around, and traveling back to their starting points. 
Prove that their 2015th meeting point is at point C.
-/
theorem meeting_point_2015 
  (A B C D : Type) 
  (x y t : ℕ)
  (odd_meeting : ∀ n : ℕ, (2 * n + 1) % 2 = 1) : 
  ∃ n, (n = 2015) → odd_meeting n = 1 → (n % 2 = 1 → (C = "C")) := 
sorry

end meeting_point_2015_l535_535178


namespace determine_function_l535_535839

theorem determine_function {f : ℝ → ℝ} :
  (∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by
  sorry

end determine_function_l535_535839


namespace quadratic_root_is_zero_then_m_neg_one_l535_535557

theorem quadratic_root_is_zero_then_m_neg_one (m : ℝ) (h_eq : (m-1) * 0^2 + 2 * 0 + m^2 - 1 = 0) : m = -1 := by
  sorry

end quadratic_root_is_zero_then_m_neg_one_l535_535557


namespace two_numbers_product_l535_535388

theorem two_numbers_product (x y : ℕ) 
  (h1 : x + y = 90) 
  (h2 : x - y = 10) : x * y = 2000 :=
by
  sorry

end two_numbers_product_l535_535388


namespace sum_rounded_to_nearest_whole_l535_535479

def mixed_number1 : ℚ := 53 + 3 / 8
def mixed_number2 : ℚ := 27 + 11 / 16
def sum_mixed_numbers : ℚ := mixed_number1 + mixed_number2

theorem sum_rounded_to_nearest_whole :
  Real.floor (sum_mixed_numbers + 1 / 2) = 81 := 
by
  -- convert mixed numbers to improper fractions
  have h1 : mixed_number1 = 427 / 8 := by norm_num
  have h2 : mixed_number2 = 443 / 16 := by norm_num
  
  -- find common denominator and add fractions
  have h3 : sum_mixed_numbers = 1297 / 16 := by norm_num

  -- convert to decimal and round
  have h4 : (1297 : ℚ) / 16 = 81.0625 := by norm_num

  -- apply rounding logic
  have h5 : Real.floor (81.0625 + 1 / 2) = 81 := by norm_num
  exact h5

end sum_rounded_to_nearest_whole_l535_535479


namespace parabola_has_property_l535_535925

noncomputable def parabola_property (p : ℝ) : Prop :=
  ∃ (A B M : ℝ × ℝ),
    (p > 0) ∧
    (M = (1, 0)) ∧
    (A.2 = 0) ∧ -- A lies on the directrix which is y = 0
    (B.1, B.2) ≠ M ∧ -- B is on the parabola
    (B.2^2 = 2 * p * B.1) ∧
    (B.2 = sqrt 3 * B.1 - sqrt 3) ∧ -- line AB equation y = sqrt(3)x - sqrt(3)
    ((A.1 + B.1) / 2 = M.1) ∧
    (by sorry : (A, M, B) is collinear) ∧
    (p = 2)

theorem parabola_has_property : ∀ (p : ℝ), parabola_property p :=
by sorry

end parabola_has_property_l535_535925


namespace isosceles_triangle_angle_bisector_inequality_l535_535766

theorem isosceles_triangle_angle_bisector_inequality
  (A B C L : Point) 
  (AB AC : Segment) 
  (base : Segment)
  (h_iso : is_isosceles_triangle A B C AB AC base)
  (h_bis : is_angle_bisector_of A C B L) 
  : length (Segment.mk C L) < 2 * length (Segment.mk B L) :=
sorry

end isosceles_triangle_angle_bisector_inequality_l535_535766


namespace find_triangle_sides_and_inradius_l535_535613

theorem find_triangle_sides_and_inradius (A B : ℝ) (a b c : ℝ) (hab_cos : (cos A / cos B) = b / a)
  (hab_ratio : b / a = 4 / 3) (hc : c = 10) (h_sum_angles : A + B = Real.pi / 2) :
  a = 6 ∧ b = 8 ∧ (a + b - c) / 2 = 2 :=
by
  sorry

end find_triangle_sides_and_inradius_l535_535613


namespace count_integer_triplets_l535_535856

open BigOperators

theorem count_integer_triplets :
  ∃ (triplets : Finset (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ triplets → Nat.lcm a (Nat.lcm b c) = 20000 ∧ Nat.gcd a (Nat.gcd b c) = 20) ∧ 
  Finset.card triplets = 56 := by
  sorry

end count_integer_triplets_l535_535856


namespace minimum_concerts_to_listen_l535_535482

/-- There are six musicians and at each concert, some musicians perform while others listen.
The minimum number of concerts needed for each of the six musicians to listen to all the other
musicians at least once is 4. -/
theorem minimum_concerts_to_listen (n : ℕ) (performers : finset (fin 6)) (i : fin 6) :
  n = 4 := 
sorry

end minimum_concerts_to_listen_l535_535482


namespace cube_diagonal_length_l535_535054

theorem cube_diagonal_length (V A : ℝ) (hV : V = 384) (hA : A = 384) : 
  ∃ d : ℝ, d = 8 * Real.sqrt 3 :=
by
  sorry

end cube_diagonal_length_l535_535054


namespace region_area_l535_535652

variable {θ : Real}
variable {R : ℝ}
variable {r : ℝ}

noncomputable def largestPossibleArea (θ : Real) (hθ1 : θ > Real.pi / 2) (hθ2 : θ < Real.pi) (hθ3 : Real.sin θ = 3 / 5) : ℝ :=
  let d : ℝ := 1
  let area : ℝ := Real.pi * (R^2 - r^2)
  have hr : R - r = d := by sorry
  have hP : R^2 - r^2 = (d/2)^2 := by sorry
  let rSquaredTerm := (d/2)^2
  have area_eq : area = Real.pi * rSquaredTerm := by sorry
  have final_area : area = Real.pi / 4 := by
    rw [area_eq, rSquaredTerm]
    simp
  final_area

theorem region_area (θ : Real) (hθ1 : θ > Real.pi / 2) (hθ2 : θ < Real.pi) (hθ3 : Real.sin θ = 3 / 5) : largestPossibleArea θ hθ1 hθ2 hθ3 = Real.pi / 4 := by
  sorry

end region_area_l535_535652


namespace odd_function_value_at_neg_one_l535_535573

theorem odd_function_value_at_neg_one :
  (∀ x, f (-x) = -f x) →
  (∀ x, 0 < x → f x = x^2 + 2/x) →
  f (-1) = -3 :=
by
  intros h_odd h_def_pos
  sorry

end odd_function_value_at_neg_one_l535_535573


namespace find_A_when_dividing_14_l535_535023

theorem find_A_when_dividing_14 {
  A : ℕ
  (h : 14 = (A * 3) + 2)
} : A = 4 :=
by {
  sorry
}

end find_A_when_dividing_14_l535_535023


namespace median_mean_difference_l535_535415

noncomputable def data_set : List ℕ := [12, 41, 44, 48, 47, 53, 60, 62, 56, 32, 23, 25, 31]

def calculate_median (lst : List ℕ) : ℚ :=
  let sorted_lst := List.sort (≤) lst
  let n := sorted_lst.length
  if n % 2 = 1 then ↑(sorted_lst.get! (n / 2))
  else (↑(sorted_lst.get! (n / 2 - 1)) + ↑(sorted_lst.get! (n / 2))) / 2

def calculate_mean (lst : List ℕ) : ℚ :=
  let sum := lst.foldl (fun acc x => acc + x) 0
  ↑sum / ↑lst.length

theorem median_mean_difference : (calculate_median data_set) - (calculate_mean data_set) = 38 / 13 :=
by
  sorry

end median_mean_difference_l535_535415


namespace verandah_area_l535_535365

theorem verandah_area (room_length room_width verandah_width : ℕ)
  (h1 : room_length = 15) (h2 : room_width = 12) (h3 : verandah_width = 2) :
  let total_length := room_length + 2 * verandah_width,
      total_width := room_width + 2 * verandah_width,
      total_area := total_length * total_width,
      room_area := room_length * room_width in
  total_area - room_area = 124 := 
by
  sorry

end verandah_area_l535_535365


namespace translation_theorem_l535_535008

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + θ)
noncomputable def g (θ : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x - 2 * φ + θ)

theorem translation_theorem
  (θ φ : ℝ)
  (hθ1 : |θ| < Real.pi / 2)
  (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi)
  (hf : f θ 0 = 1 / 2)
  (hg : g θ φ 0 = 1 / 2) :
  φ = 2 * Real.pi / 3 :=
sorry

end translation_theorem_l535_535008


namespace sufficient_p_wages_l535_535802

variable (S P Q : ℕ)

theorem sufficient_p_wages (h1 : S = 40 * Q) (h2 : S = 15 * (P + Q))  :
  ∃ D : ℕ, S = D * P ∧ D = 24 := 
by
  use 24
  sorry

end sufficient_p_wages_l535_535802


namespace exists_good_set_l535_535664

noncomputable def M : Finset ℕ := Finset.range 21 \ {0}

noncomputable def f (S : Finset ℕ) (h9 : S.card = 9) : ℕ :=
  sorry -- Assume the existence of some function f.

theorem exists_good_set :
  ∃ T : Finset ℕ, T.card = 10 ∧ ∀ k : ℕ, k ∈ T → (f (T \ {k}) (by rw [Finset.card_erase_of_mem]; exact Nat.card_eq 9 (by assumption)) ≠ k) :=
begin
  sorry
end

end exists_good_set_l535_535664


namespace biased_coin_probability_l535_535780

open ProbabilityTheory

noncomputable def biasedCoin (p_heads : ℝ) (p_tails : ℝ) : List ℕ → ℝ
| []     := 1
| (0 :: flips) := p_heads * biasedCoin p_heads p_tails flips
| (1 :: flips) := p_tails * biasedCoin p_heads p_tails flips

theorem biased_coin_probability (p_heads : ℝ) (p_tails : ℝ) :
  (p_heads = 0.3) → (p_tails = 0.7) →
  biasedCoin p_heads p_tails ([1, 1, 1, 0] ++ [0, 1, 0] ++ [1, 0, 1, 1]) = 0.3087 :=
by
  intros h_heads h_tails
  sorry

end biased_coin_probability_l535_535780


namespace triangle_problem_l535_535273

-- Given an acute triangle ABC with opposite sides a, b, and c,
-- and given √3a = 2c sin(A),
-- find C and the perimeter of the triangle given further conditions.
theorem triangle_problem
  (ABC : Type*)
  [triangle ABC]
  {a b c : ℝ}
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (acute : is_acute_triangle ABC)
  (opposite_sides : opposite_sides ABC a b c)
  (cond1 : √3 * a = 2 * c * sin (angle A))
  (cond2 : c = √7)
  (cond3 : a * b = 6) :
  (angle C = π / 3) ∧ ((a + b + c) = 5 + √7) := sorry

end triangle_problem_l535_535273


namespace lucy_average_speed_l535_535678

-- Definitions related to the conditions
def distance_1 : ℕ := 420
def time_1 : ℕ := 7
def distance_2 : ℕ := 480
def time_2 : ℕ := 8

-- Main statement to prove
theorem lucy_average_speed :
  let total_distance := distance_1 + distance_2 in
  let total_time := time_1 + time_2 in
  let average_speed := total_distance / total_time in
  average_speed = 60 :=
by
  sorry

end lucy_average_speed_l535_535678


namespace number_of_strings_is_multiple_of_3_l535_535705

theorem number_of_strings_is_multiple_of_3 (N : ℕ) :
  (∀ (avg_total avg_one_third avg_two_third : ℚ), 
    avg_total = 80 ∧ avg_one_third = 70 ∧ avg_two_third = 85 →
    (∃ k : ℕ, N = 3 * k)) :=
by
  intros avg_total avg_one_third avg_two_third h
  sorry

end number_of_strings_is_multiple_of_3_l535_535705


namespace meeting_point_2015th_l535_535176

-- Define the parameters of the problem
variables (A B C D : Type)
variables (x y t : ℝ) -- Speeds and the initial time delay

-- State the problem as a theorem
theorem meeting_point_2015th (start_times_differ : t > 0)
                            (speeds_pos : x > 0 ∧ y > 0)
                            (pattern : ∀ n : ℕ, (odd n → (meeting_point n = C)) ∧ (even n → (meeting_point n = D)))
                            (n = 2015) :
  meeting_point n = C :=
  sorry

end meeting_point_2015th_l535_535176


namespace odd_function_l535_535905

def f (x : ℝ) : ℝ :=
  if x > 0 then
    x^3 + x + 1
  else if x < 0 then
    x^3 + x - 1
  else 
    0

theorem odd_function (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
begin
  intros x h,
  have h_neg : f (-x) = -f x, from h_odd x,
  have h_nonpos : f x = -f (-x), {
    rw [h_neg, h_pos (-x)],
    simp at *,
    sorry
  },
  sorry
end

end odd_function_l535_535905


namespace max_turtle_paths_l535_535778

-- Defining the grid and its points
inductive Point : Type
| A | B | C | D | E | F | K | L | M

open Point

-- Defining the paths between points (Orthogonal and Diagonal)
inductive Path : Type
| orthogonal : Point → Point → Path
| diagonal : Point → Point → Path

open Path

-- Check if a path is orthogonal
def is_orthogonal : Path → Bool
| (orthogonal _ _) => true
| _               => false

-- Define possible orthogonal paths
def orthogonal_paths : List Path :=
[orthogonal A B, orthogonal B C, orthogonal D E, orthogonal E F, orthogonal K L, orthogonal L M,
 orthogonal A D, orthogonal B E, orthogonal C F, orthogonal D K, orthogonal E L, orthogonal F M]

-- Define possible diagonal paths
def diagonal_paths : List Path :=
[diagonal A E, diagonal B F, diagonal D L, diagonal E M,
 diagonal C E, diagonal B D, diagonal F L, diagonal E K]

-- Maximum number of alternating paths the turtle can traverse without repetition
def turtle_max_paths (paths : List Path) : Bool :=
(paths.length ≤ 17) && paths.enum.all (λ (i, p),
  (i % 2 = 0) → is_orthogonal p = true ∧ (i % 2 = 1) → is_orthogonal p = false)

theorem max_turtle_paths : ∃ (paths : List Path), 
  paths ⊂ (orthogonal_paths ++ diagonal_paths) ∧
  turtle_max_paths paths ∧ 
  paths.length = 17 :=
by
  sorry

end max_turtle_paths_l535_535778


namespace spend_on_video_games_l535_535602

def total_allowance : ℝ := 50
def fraction_books : ℝ := 1 / 2
def fraction_toys : ℝ := 1 / 4
def fraction_snacks : ℝ := 1 / 10

theorem spend_on_video_games : 
  (total_allowance - 
    (fraction_books * total_allowance + 
     fraction_toys * total_allowance + 
     fraction_snacks * total_allowance)) = 7.5 := 
by
  sorry

end spend_on_video_games_l535_535602


namespace matt_age_three_years_ago_l535_535041

theorem matt_age_three_years_ago (james_age_three_years_ago : ℕ) (age_difference : ℕ) (future_factor : ℕ) :
  james_age_three_years_ago = 27 →
  age_difference = 3 →
  future_factor = 2 →
  ∃ matt_age_now : ℕ,
  james_age_now: ℕ,
    james_age_now = james_age_three_years_ago + age_difference ∧
    (matt_age_now + 5) = future_factor * (james_age_now + 5) ∧
    matt_age_now = 65 :=
by
  sorry

end matt_age_three_years_ago_l535_535041


namespace find_A_cos_alpha_beta_l535_535582

-- Define the function f
def f (x : ℝ) (A : ℝ) := A * Real.cos (x / 4 + Real.pi / 6)

-- a) First proof problem: proving that A = 2
theorem find_A (A : ℝ) 
  (h : f (Real.pi / 3) A = Real.sqrt 2) :
  A = 2 :=
  by 
  sorry

-- b) Second proof problem: proving that cos(α + β) = -13/85
theorem cos_alpha_beta (α β : ℝ) 
  (h_alpha : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (h_beta : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h1 : f (4 * α + 4 * Real.pi / 3) 2 = -30 / 17)
  (h2 : f (4 * β - 2 * Real.pi / 3) 2 = 8 / 5) :
  Real.cos (α + β) = -13 / 85 :=
  by 
  sorry

end find_A_cos_alpha_beta_l535_535582


namespace multiples_of_7_properties_l535_535729

-- Define the range of natural numbers within 50
def nat_within_50 : Set ℕ := { n | n ≤ 50 }

-- Define the multiples of 7 within 50
def multiples_of_7_within_50 : Set ℕ := { n | n ∈ nat_within_50 ∧ 7 ∣ n }

-- Prove the desired properties
theorem multiples_of_7_properties :
  (multiples_of_7_within_50.card = 7) ∧
  (7 ∈ multiples_of_7_within_50) ∧
  (49 ∈ multiples_of_7_within_50) :=
  by
  sorry

end multiples_of_7_properties_l535_535729


namespace find_m_l535_535592

open Set

variable {α : Type} [DecidableEq α]

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + (m - 1) = 0}

theorem find_m (m : ℝ) (hA : A = {1, 2}) (hB_in_AuB : ∀ x ∈ B m, x ∈ A) :
  m = 3 :=
by
  sorry

end find_m_l535_535592


namespace det_rotation_matrix_l535_535316

variable (α : ℝ)

def rotation_matrix (α : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos α, -Real.sin α], ![Real.sin α, Real.cos α]]

theorem det_rotation_matrix (α : ℝ) : det (rotation_matrix α) = 1 := 
by
  -- The proof steps are omitted here.
  sorry

end det_rotation_matrix_l535_535316


namespace max_intersections_50_segments_l535_535357

theorem max_intersections_50_segments :
  let X_points := 10
  let Y_points := 5
  let segments := X_points * Y_points
  let intersections := (X_points choose 2) * (Y_points choose 2)
    in intersections = 450 :=
by
  -- Definitions based on problem conditions
  let X_points := 10
  let Y_points := 5
  let segments := X_points * Y_points
  let intersections := (X_points choose 2) * (Y_points choose 2)
  
  -- Target statement
  have h_intersections : intersections = 450 := sorry,

  -- Conclusion
  exact h_intersections

end max_intersections_50_segments_l535_535357


namespace find_angle_measure_l535_535817

theorem find_angle_measure (x : ℝ) (hx : 90 - x + 40 = (1 / 2) * (180 - x)) : x = 80 :=
by
  sorry

end find_angle_measure_l535_535817


namespace benjamin_skating_time_l535_535250

theorem benjamin_skating_time (distance speed : ℕ) (h_dist : distance = 150) (h_speed : speed = 12) :
  (distance / speed : ℚ) = 12.5 :=
by
  rw [h_dist, h_speed]
  norm_num
  sorry

end benjamin_skating_time_l535_535250


namespace equivalentOverallReductionA_equivalentOverallReductionB_equivalentOverallReductionC_l535_535396

def originalPriceA : ℝ := 600
def reductionsA : list ℝ := [0.20, 0.15, 0.10]
def expectedReductionA : ℝ := 38.8

def originalPriceB : ℝ := 800
def reductionsB : list ℝ := [0.25, 0.20, 0.05]
def expectedReductionB : ℝ := 43

def originalPriceC : ℝ := 1000
def reductionsC : list ℝ := [0.30, 0.10, 0.25]
def expectedReductionC : ℝ := 52.75

theorem equivalentOverallReductionA :
  calculateOverallPercentageReduction originalPriceA reductionsA = expectedReductionA := by
  sorry

theorem equivalentOverallReductionB :
  calculateOverallPercentageReduction originalPriceB reductionsB = expectedReductionB := by
  sorry

theorem equivalentOverallReductionC :
  calculateOverallPercentageReduction originalPriceC reductionsC = expectedReductionC := by
  sorry

-- Function to calculate the overall percentage reduction based on the original price and a list of reductions
noncomputable def calculateOverallPercentageReduction (originalPrice : ℝ) (reductions : list ℝ) : ℝ :=
  let finalPrice := reductions.foldl (λ price reduction => price * (1 - reduction)) originalPrice
  ((originalPrice - finalPrice) / originalPrice) * 100

end equivalentOverallReductionA_equivalentOverallReductionB_equivalentOverallReductionC_l535_535396


namespace Matthew_friends_count_l535_535326

-- Conditions as definitions
def total_crackers := 36
def crackers_per_friend := 6.5

-- Question to prove: how many friends did Matthew give crackers to given the conditions
def number_of_friends := Nat.floor (total_crackers / crackers_per_friend)

-- Proof statement
theorem Matthew_friends_count : number_of_friends = 5 := by
  sorry

end Matthew_friends_count_l535_535326


namespace find_fahrenheit_l535_535015

variable (F : ℝ)
variable (C : ℝ)

theorem find_fahrenheit (h : C = 40) (h' : C = 5 / 9 * (F - 32)) : F = 104 := by
  sorry

end find_fahrenheit_l535_535015


namespace tetrahedron_conditions_l535_535561

theorem tetrahedron_conditions (k : ℕ) (a : ℝ) (h_k : k ≥ 0 ∧ k ≤ 5) : 
  match k with 
  | 1 => 0 < a ∧ a < Real.sqrt 3
  | 2 => 0 < a ∧ a < (Real.sqrt 6 + Real.sqrt 2) / 2
  | 3 => a > 0
  | 4 => a > (Real.sqrt 6 - Real.sqrt 2) / 2
  | 5 => a > Real.sqrt 3 / 3
  | _ => False
  end :=
by
  sorry

end tetrahedron_conditions_l535_535561


namespace xy_sum_of_squares_l535_535952

theorem xy_sum_of_squares (x y : ℝ) (h1 : x - y = 18) (h2 : x + y = 22) : x^2 + y^2 = 404 := by
  sorry

end xy_sum_of_squares_l535_535952


namespace toothpick_count_l535_535405

theorem toothpick_count (length width : ℕ) (h_len : length = 20) (h_width : width = 10) : 
  2 * (length * (width + 1) + width * (length + 1)) = 430 :=
by
  sorry

end toothpick_count_l535_535405


namespace day_100_days_after_l535_535492

def DayOfWeek := ℕ
def Thursday : DayOfWeek := 4
def Saturday : DayOfWeek := 6

-- Given: birthday_day = Thursday
-- Question: What day of the week will it be 100 days after her birthday?
-- Expected answer to prove: Saturday

theorem day_100_days_after (d : DayOfWeek) (birthday_day : d = Thursday) : 
  (d + 100) % 7 = Saturday := 
by
  sorry

end day_100_days_after_l535_535492


namespace radius_calculation_l535_535375

noncomputable def radius_of_circle : ℝ :=
  let center_x := -19 / 2 in
  let radius := real.sqrt ((center_x - 2) ^ 2 + 5 ^ 2) / 2 in
  radius

theorem radius_calculation:
  ∃ (center_x : ℝ), 
  (real.dist (center_x, 0) (2, 5) = real.dist (center_x, 0) (3, 1)) ∧
    (radius_of_circle = real.sqrt 629 / 2) :=
by
  use -19 / 2
  simp [radius_of_circle, real.dist_eq]
  sorry

end radius_calculation_l535_535375


namespace probability_exactly_five_blue_marbles_l535_535335

noncomputable def prob_five_blue_marbles : ℚ :=
  (nat.choose 8 5) * ((2/3)^5) * ((1/3)^3)

theorem probability_exactly_five_blue_marbles :
  (prob_five_blue_marbles).to_real ≈ 0.272 :=
begin
  sorry
end

end probability_exactly_five_blue_marbles_l535_535335


namespace oates_reunion_attendees_l535_535483

theorem oates_reunion_attendees (total_guests hall_attendees both_attendees : ℕ)
  (h1 : total_guests = 150)
  (h2 : hall_attendees = 52)
  (h3 : both_attendees = 28) : 
  let oates_attendees := total_guests - hall_attendees + both_attendees in
  oates_attendees = 126 :=
by
  -- Define oates_attendees based on given conditions
  let oates_attendees := total_guests - (hall_attendees - both_attendees)
  -- Prove that oates_attendees equals to 126
  have : oates_attendees = 150 - (52 - 28) := by
    dsimp [oates_attendees]
    rw [h1, h2, h3]
  have : oates_attendees = 150 - 24 := by
    rw [h1, h2, h3]
  rw this
  exact eq.refl 126

end oates_reunion_attendees_l535_535483


namespace sequence_a10_eq_92_l535_535635

-- Define the sequence recurrence relation and initial condition
def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n ≥ 1, a (n + 1) = a n + 2 * n

-- Prove that a_10 = 92
theorem sequence_a10_eq_92 (a : ℕ → ℕ) (h : sequence a) : a 10 = 92 :=
begin
  sorry
end

end sequence_a10_eq_92_l535_535635


namespace fixed_point_at_5_75_l535_535539

-- Defining the function
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 + k * x - 5 * k

-- Stating the theorem that the graph passes through the fixed point (5, 75)
theorem fixed_point_at_5_75 (k : ℝ) : quadratic_function k 5 = 75 := by
  sorry

end fixed_point_at_5_75_l535_535539


namespace number_of_unique_hyperbolas_l535_535859

def binomial_coefficient (m n : ℕ) : ℕ :=
  Nat.choose m n

theorem number_of_unique_hyperbolas :
  let valid_coeffs := {C_mn | ∃ (m n : ℕ), 1 ≤ n ∧ n ≤ m ∧ m ≤ 5 ∧ C_mn = binomial_coefficient m n ∧ C_mn > 1}
  let unique_coeffs := valid_coeffs.to_finset
  unique_coeffs.card = 6 :=
by
  sorry

end number_of_unique_hyperbolas_l535_535859


namespace number_of_1s_and_2s_l535_535061

theorem number_of_1s_and_2s {N : ℕ} (digits : Fin N → Fin 2) 
  (h_len : 100 = N) 
  (h_ev_twos : ∀ i j : Fin N, digits i = 1 → digits j = 1 → even (|i - j|))
  (h_sum : (digits.sum : ℕ) % 3 = 0) :
  ∃ n1 n2 : ℕ, n1 + n2 = 100 ∧  n1 = 98 ∧ n2 = 2 :=
by 
  sorry

end number_of_1s_and_2s_l535_535061


namespace proof_true_proposition_l535_535564

open Classical

def P : Prop := ∀ x : ℝ, x^2 ≥ 0
def Q : Prop := ∃ x : ℚ, x^2 ≠ 3
def true_proposition (p q : Prop) := p ∨ ¬q

theorem proof_true_proposition : P ∧ ¬Q → true_proposition P Q :=
by
  intro h
  sorry

end proof_true_proposition_l535_535564


namespace sequence_converges_l535_535867

open Real

theorem sequence_converges (x : ℕ → ℝ) (h₀ : ∀ n, x (n + 1) = 1 + x n - 0.5 * (x n) ^ 2) (h₁ : 1 < x 1 ∧ x 1 < 2) :
  ∀ n ≥ 3, |x n - sqrt 2| < 2 ^ (-n : ℝ) :=
by
  sorry

end sequence_converges_l535_535867


namespace find_m_correct_l535_535798

noncomputable def find_m (Q : Point) (B : List Point) (m : ℝ) : Prop :=
  let circle_area := 4 * Real.pi
  let radius := 2
  let area_sector_B1B2 := Real.pi / 3
  let area_region_B1B2 := 1 / 8
  let area_triangle_B1B2 := area_sector_B1B2 - area_region_B1B2 * circle_area
  let area_sector_B4B5 := Real.pi / 3
  let area_region_B4B5 := 1 / 10
  let area_triangle_B4B5 := area_sector_B4B5 - area_region_B4B5 * circle_area
  let area_sector_B9B10 := Real.pi / 3
  let area_region_B9B10 := 4 / 15 - Real.sqrt 3 / m
  let area_triangle_B9B10 := area_sector_B9B10 - area_region_B9B10 * circle_area
  m = 3

theorem find_m_correct (Q : Point) (B : List Point) : find_m Q B 3 :=
by
  unfold find_m
  sorry

end find_m_correct_l535_535798


namespace max_area_triangle_ABC_l535_535888

noncomputable def OA : ℝ := 4
noncomputable def OB : ℝ := 3
noncomputable def OC : ℝ := 2
noncomputable def dot_product_OB_OC : ℝ := 3

theorem max_area_triangle_ABC :
  ∃ A B C : ℝ×ℝ, 
    (dist (0, 0) A = OA) ∧
    (dist (0, 0) B = OB) ∧
    (dist (0, 0) C = OC) ∧
    ((B.1 * C.1 + B.2 * C.2) = dot_product_OB_OC) ∧
    (area (A, B, C) = 2 * sqrt 7 + 3 * sqrt 3 / 2) :=
sorry

end max_area_triangle_ABC_l535_535888


namespace triangle_ratio_constant_l535_535883

theorem triangle_ratio_constant
  (A B C D : Type)
  [inner_product_space ℝ D]
  [plane ℝ D]
  [has_angle ℝ D]
  [has_measure ℝ D]
  (eq_triangle : triangle ℝ D A B C)
  (D_in_triangle : point_in_interior ℝ D D eq_triangle)
  (angle_condition : angle_eq ℝ D (angle A D B) (angle A C B + 90))
  (length_condition : length A C ∙ length B D = length A D ∙ length B C) :
  ∃ k : ℝ, ∀ eq_triangle_A B C D : Type,
  ℝeq_triangle_ABeq_triangle_CD_neq(k) eq_triangle_AB_eq_eq_triangle_CD_neq(k)  :=
begin
  -- Proof steps will be skipped here.
  sorry
end

end triangle_ratio_constant_l535_535883


namespace primes_digit_3_count_l535_535244

open Nat 

def primes_with_digit_3_under_50 : List ℕ := [3, 13, 23, 33, 43]

/-- The number of primes less than 50 with 3 as the ones digit is 4. -/
theorem primes_digit_3_count :
  (primes_with_digit_3_under_50.filter (λ n, Prime n)).length = 4 :=
by 
  sorry

end primes_digit_3_count_l535_535244


namespace CF_length_l535_535382

-- Definitions of points and lengths
def AB : ℝ := 12
def BC : ℝ := 6
def area_rect (l w : ℝ) : ℝ := l * w

-- Conditions
def center_S (l w : ℝ) : (ℝ × ℝ) := (l / 2, w / 2)
def area_BCFS (rect_area : ℝ) : ℝ := rect_area / 3

-- Main theorem to prove CF = 4 cm
theorem CF_length (F : ℝ) (h1 : F ∈ (0 .. BC)) 
  (h2 : F = CF) 
  (h3 : BCFS_area = area_BCFS (area_rect AB BC)) : 
  F = 4 := 
sorry

end CF_length_l535_535382


namespace problem_1_problem_2_problem_3_l535_535589

-- Define the sequence a_n
def a : ℕ → ℝ
| 0     := 3
| (n+1) := c * a n + m

noncomputable def b (n : ℕ) : ℝ := 1 / (a n - 1)

noncomputable def S (n : ℕ) : ℝ := (finset.range n).sum (λ i, b (i + 1))

theorem problem_1 (h₀ : c = 1) (h₁ : m = 1) : ∀ n, a n = n + 2 :=
sorry

theorem problem_2 (h₀ : c = 2) (h₁ : m = -1) : 
  (∃ r a₀, ∀ n, (a n - 1) = r^n * a₀) :=
sorry

theorem problem_3 (h₀ : c = 2) (h₁ : m = -1) :
  ∀ n, S n < 1 :=
sorry

end problem_1_problem_2_problem_3_l535_535589


namespace no_convex_polyhedron_with_odd_sections_l535_535518

-- Define a convex polyhedron and its properties.
structure ConvexPolyhedron where
  vertices : Finset Point  -- A finite set of points
  edges : Set (Point × Point)  -- A set of edges
  faces : Set (Set Point)  -- A set of faces
  is_convex : Convex polyhedron  -- Convex condition
  nonempty : vertices.Nonempty  -- Ensure there is at least one vertex
  -- Other properties defining the polyhedron...

-- Define the property that any section of the polyhedron by a plane that doesn't pass through a vertex is a polygon with an odd number of sides.
def odd_sided_sections (P : ConvexPolyhedron) : Prop :=
  ∀ (σ : Plane), (∀ v ∈ P.vertices, v ∉ σ) →
  ∃ (poly : Polygon), poly.sides % 2 = 1 ∧ (P ∩ σ) = poly

-- The theorem statement confirming the non-existence of such a polyhedron.
theorem no_convex_polyhedron_with_odd_sections :
  ¬ ∃ (P : ConvexPolyhedron), odd_sided_sections P :=
by
  sorry

end no_convex_polyhedron_with_odd_sections_l535_535518


namespace sin_eq_log_has_three_real_roots_l535_535374

-- Definitions of the functions
def y1 (x : ℝ) : ℝ := Real.sin x
def y2 (x : ℝ) : ℝ := Real.log x

-- Statement of the problem
theorem sin_eq_log_has_three_real_roots
  (h_dom1 : ∀ x : ℝ, true) -- Domain of sin x is all real numbers
  (h_dom2 : ∀ x : ℝ, x > 0 → true) -- Domain of log x is x > 0
  : ∃ x1 x2 x3 : ℝ, (y1 x1 = y2 x1) ∧ (y1 x2 = y2 x2) ∧ (y1 x3 = y2 x3) ∧ 
    (x1 ≠ x2) ∧ (x1 ≠ x3) ∧ (x2 ≠ x3) := sorry

end sin_eq_log_has_three_real_roots_l535_535374


namespace line_equation_passing_through_points_l535_535595

theorem line_equation_passing_through_points 
  (a₁ b₁ a₂ b₂ : ℝ)
  (h1 : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h2 : 2 * a₂ + 3 * b₂ + 1 = 0)
  (h3 : ∀ (x y : ℝ), (x, y) = (2, 3) → a₁ * x + b₁ * y + 1 = 0 ∧ a₂ * x + b₂ * y + 1 = 0) :
  (∀ (x y : ℝ), (2 * x + 3 * y + 1 = 0) ↔ 
                (a₁ = x ∧ b₁ = y) ∨ (a₂ = x ∧ b₂ = y)) :=
by
  sorry

end line_equation_passing_through_points_l535_535595


namespace complex_expression_value_l535_535219

noncomputable def z1 : ℂ := 2 - Complex.i
noncomputable def z2 : ℂ := -Complex.i

theorem complex_expression_value :
  (z1 / z2 + Complex.abs z2) = 2 + 2 * Complex.i := by
  sorry

end complex_expression_value_l535_535219


namespace find_missing_number_l535_535372

theorem find_missing_number (x : ℕ) : 
  (1 + 22 + 23 + 24 + 25 + 26 + x + 2) / 8 = 20 → x = 37 := by
  sorry

end find_missing_number_l535_535372


namespace tile_probability_eq_five_over_eleven_l535_535845
noncomputable def probabilityTileInPirate : ℚ := 
  let probability_unique_letters := [('P', 1), ('R', 1), ('O', 1), ('B', 1), ('A', 1), ('I', 1), ('L', 1), ('T', 1), ('Y', 1)]
  let pirate_letters := ['P', 'I', 'R', 'A', 'T', 'E']
  let letters_in_pirate := probability_unique_letters.filter (λ x => x.fst ∈ pirate_letters)
  let count_letters_in_pirate := letters_in_pirate.length
  let total_letters := 11
  return ratMk count_letters_in_pirate total_letters

theorem tile_probability_eq_five_over_eleven : probabilityTileInPirate = 5 / 11 := by
  sorry

end tile_probability_eq_five_over_eleven_l535_535845


namespace all_points_same_value_l535_535093

theorem all_points_same_value {f : ℤ × ℤ → ℕ}
  (h : ∀ x y : ℤ, f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4) :
  ∃ k : ℕ, ∀ x y : ℤ, f (x, y) = k :=
sorry

end all_points_same_value_l535_535093


namespace odd_function_solution_l535_535902

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_solution (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
by
  sorry

end odd_function_solution_l535_535902


namespace real_solution_count_l535_535136
open Real

theorem real_solution_count :
  (∃ x : ℝ, (5 * x / (x ^ 2 + 2 * x + 4) + 7 * x / (x ^ 2 - 7 * x + 4) = -5 / 3)) →
  (number_of_real_solutions (λ x, 5 * x / (x ^ 2 + 2 * x + 4) + 7 * x / (x ^ 2 - 7 * x + 4) + 5 / 3) = 2) :=
begin
  sorry
end

end real_solution_count_l535_535136


namespace interest_difference_l535_535022

-- Define the principal, rate, and time
def principal : ℕ := 1000
def rate : ℝ := 0.10
def time : ℕ := 4

-- Define the simple interest calculation
def simple_interest (P : ℕ) (R : ℝ) (T : ℕ) : ℝ :=
  P * R * T

-- Define the compound interest calculation
def compound_interest (P : ℕ) (R : ℝ) (T : ℕ) : ℝ :=
  P * (1 + R)^T - P

-- The main proposition to prove the difference equals $64.10
theorem interest_difference :
  let SI := simple_interest principal rate time in
  let CI := compound_interest principal rate time in
  CI - SI = 64.10 :=
by
  -- calculation details and proof will go here when implemented
  sorry

end interest_difference_l535_535022


namespace count_non_squares_or_cubes_in_200_l535_535946

theorem count_non_squares_or_cubes_in_200 :
  let total_numbers := 200
  let count_perfect_squares := 14
  let count_perfect_cubes := 5
  let count_sixth_powers := 2
  total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers) = 183 :=
by
  let total_numbers := 200
  let count_perfect_squares := 14
  let count_perfect_cubes := 5
  let count_sixth_powers := 2
  have h1 : total_numbers = 200 := rfl
  have h2 : count_perfect_squares = 14 := rfl
  have h3 : count_perfect_cubes = 5 := rfl
  have h4 : count_sixth_powers = 2 := rfl
  show total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers) = 183
  calc
    total_numbers - (count_perfect_squares + count_perfect_cubes - count_sixth_powers)
        = 200 - (14 + 5 - 2) : by rw [h1, h2, h3, h4]
    ... = 200 - 17 : by norm_num
    ... = 183 : by norm_num

end count_non_squares_or_cubes_in_200_l535_535946


namespace volleyball_ranking_l535_535971

-- Define type for place
inductive Place where
  | first : Place
  | second : Place
  | third : Place

-- Define type for teams
inductive Team where
  | A : Team
  | B : Team
  | C : Team

open Place Team

-- Given conditions as hypotheses
def LiMing_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p first A ∨ p third A) ∧ (p first B ∨ p third B) ∧ 
  ¬ (p first A ∧ p third A) ∧ ¬ (p first B ∧ p third B)

def ZhangHua_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p third A ∨ p first C) ∧ (p third A ∨ p first A) ∧ 
  ¬ (p third A ∧ p first A) ∧ ¬ (p first C ∧ p third C)

def WangQiang_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p second C ∨ p third B) ∧ (p second C ∨ p third C) ∧ 
  ¬ (p second C ∧ p third C) ∧ ¬ (p third B ∧ p second B)

-- Final proof problem
theorem volleyball_ranking (p : Place → Team → Prop) :
    (LiMing_prediction_half_correct p) →
    (ZhangHua_prediction_half_correct p) →
    (WangQiang_prediction_half_correct p) →
    p first C ∧ p second A ∧ p third B :=
  by
    sorry

end volleyball_ranking_l535_535971


namespace angle_bcd_is_90_degrees_l535_535618

theorem angle_bcd_is_90_degrees (circle : Type) 
  (diameter EB : circle)
  (parallels : ∀ (DC AB ED DF : circle), 
                EB ∥ DC ∧ 
                AB ∥ ED ∧ 
                DF ∥ AB ∧ 
                DF ∥ ED) 
  (angles_ratio : ∃ (AEB ABE : ℝ), 
                   3 * AEB = 4 * ABE ∧ 
                   AEB + ABE = 90) : 
  ∠BCD = 90 := 
by 
  sorry

end angle_bcd_is_90_degrees_l535_535618


namespace man_half_age_in_12_years_l535_535056

variables (M F Y : ℕ)

def man's_age := (2/5 : ℚ) * F
def father_age := 60
def years_to_half_age := ∃ (Y : ℕ), (man's_age + Y : ℚ) = (1/2 : ℚ) * (father_age + Y)

theorem man_half_age_in_12_years : years_to_half_age man's_age father's_age :=
by
  let F := 60
  let M := (2/5 : ℚ) * F
  use 12
  have h1 : M = 24 := by norm_num
  have h2 : (M + 12 : ℚ) = (1/2 : ℚ) * (F + 12) := by norm_num
  exact h2

end man_half_age_in_12_years_l535_535056


namespace probability_rolling_less_than_4_l535_535756

theorem probability_rolling_less_than_4 (h : ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 8) : 
  (if p : ∃ x, x < 4 ∧ x ∈ (finset.range 8 \ {0}) then ↑(finset.filter (λ x, x < 4) (finset.range 8 \ {0})).card / ↑((finset.range 8 \ {0}).card) = (3 : ℚ) / 8 
  else 0) := sorry

end probability_rolling_less_than_4_l535_535756


namespace number_of_men_in_first_group_l535_535956

-- Define the conditions
def first_group_men_colors_48m_in_2_days (M : ℕ) : Prop :=
  work_per_day M = 24

def six_men_colors_36m_in_1_day : Prop :=
  work_per_day 6 = 36

-- Helper function to represent work done per day
def work_per_day (workers : ℕ) : ℕ :=
  if workers = 6 then 36 
  else (if workers = M then 24 else 0) -- Assuming M is defined elsewhere

-- Define the main statement to prove
theorem number_of_men_in_first_group (M : ℕ) (h1 : first_group_men_colors_48m_in_2_days M) (h2 : six_men_colors_36m_in_1_day) :
  M = 9 := by
  sorry

end number_of_men_in_first_group_l535_535956


namespace eugene_used_4_boxes_of_toothpicks_l535_535848

theorem eugene_used_4_boxes_of_toothpicks:
  ∀ (cards_per_box cards_used cards_total toothpicks_per_card toothpicks_per_box : ℕ) 
  (cards_total_eq : cards_total = 52) 
  (cards_used_eq : cards_used = cards_total - 23)
  (toothpicks_per_card_eq : toothpicks_per_card = 64) 
  (toothpicks_per_box_eq : toothpicks_per_box = 550)
  (cards_per_box_eq : cards_per_box = (toothpicks_per_card * cards_used) / toothpicks_per_box),
  ⌈cards_per_box⌉ = 4 :=
by
  assume cards_per_box cards_used cards_total toothpicks_per_card toothpicks_per_box
         cards_total_eq cards_used_eq toothpicks_per_card_eq toothpicks_per_box_eq cards_per_box_eq
  sorry

end eugene_used_4_boxes_of_toothpicks_l535_535848


namespace tennis_player_games_l535_535077

theorem tennis_player_games (b : ℕ → ℕ) (h1 : ∀ k, b k ≥ k) (h2 : ∀ k, b k ≤ 12 * (k / 7)) :
  ∃ i j : ℕ, i < j ∧ b j - b i = 20 :=
by
  sorry

end tennis_player_games_l535_535077


namespace count_negative_numbers_l535_535260

-- Define the evaluations
def eval_expr1 := - (2^2)
def eval_expr2 := (-2) ^ 2
def eval_expr3 := - (-2)
def eval_expr4 := - (abs (-2))

-- Define the negativity checks
def is_negative (x : ℤ) : Prop := x < 0

-- Prove the number of negative results
theorem count_negative_numbers :
  (∑ b in [eval_expr1, eval_expr2, eval_expr3, eval_expr4].map is_negative, if b then 1 else 0) = 2 :=
by sorry

end count_negative_numbers_l535_535260


namespace tangent_slope_at_P_is_120_degrees_l535_535723

noncomputable def tangent_slope_angle : ℝ :=
  let f : ℝ → ℝ := λ x, x^3 - real.sqrt 3 * x + 1
  let f' := deriv f
  let slope := f' 0
  real.arctan slope

theorem tangent_slope_at_P_is_120_degrees :
  tangent_slope_angle = 120 :=
by
  let f : ℝ → ℝ := λ x, x^3 - real.sqrt 3 * x + 1
  let f' := deriv f
  let slope := f' 0
  have slope_eq : slope = -real.sqrt 3, sorry
  have θ := real.arctan slope
  have θ_eq : θ = 120, sorry
  exact θ_eq

end tangent_slope_at_P_is_120_degrees_l535_535723


namespace hyperbola_eccentricity_is_2_l535_535707

-- Define the hyperbola equation and its parameters
noncomputable def hyperbola_eq : (ℝ × ℝ) → Prop := 
  λ (x y), x^2 - (y^2 / 3) = 1

-- Define the parameters for the hyperbola
def a : ℝ := 1
def b : ℝ := sqrt 3
def c : ℝ := 2

-- Define the eccentricity of the hyperbola
def eccentricity (c a : ℝ) : ℝ := c / a

-- Prove that the eccentricity of the given hyperbola is 2
theorem hyperbola_eccentricity_is_2 : eccentricity c a = 2 := sorry

end hyperbola_eccentricity_is_2_l535_535707


namespace correct_choice_is_B_l535_535433

/-
# This Lean 4 statement corresponds to the problem asking
# for the correct choice among four statements.
-/

variable (s1 s2 s3 s4 : Prop)

-- Conditions as extracted from the problem statement and solution
axiom h_s1 : ¬ s1
axiom h_s2 : ¬ s2
axiom h_s3 : s3
axiom h_s4 : s4

-- Prove that the correct choice is B, i.e., (s3 and s4 are correct)
theorem correct_choice_is_B : (s3 ∧ s4) ∧ (¬ s1 ∧ ¬ s2) := by
  exact ⟨⟨h_s3, h_s4⟩, ⟨h_s1, h_s2⟩⟩

end correct_choice_is_B_l535_535433


namespace winning_probability_is_correct_l535_535968

def idioms : List String := ["意气风发", "风平浪静", "心猿意马", "信马由缰", "气壮山河", "信口开河"]

def win_condition (idiom1 idiom2 : String) : Bool :=
  ∃ ch, ch ∈ idiom1 ∧ ch ∈ idiom2

def total_possible_outcomes : ℕ := Nat.choose 6 2

def successful_outcomes : ℕ :=
  List.length (List.filter (λ (pair : String × String), win_condition pair.fst pair.snd)
    [(idioms[0], idioms[1]), (idioms[0], idioms[2]), (idioms[0], idioms[3]), 
     (idioms[0], idioms[4]), (idioms[0], idioms[5]), (idioms[1], idioms[2]),
     (idioms[1], idioms[3]), (idioms[1], idioms[4]), (idioms[1], idioms[5]), 
     (idioms[2], idioms[3]), (idioms[2], idioms[4]), (idioms[2], idioms[5]),
     (idioms[3], idioms[4]), (idioms[3], idioms[5]), (idioms[4], idioms[5])])

noncomputable def probability_of_winning : ℚ :=
  (successful_outcomes : ℚ) / (total_possible_outcomes : ℚ)

theorem winning_probability_is_correct :
  probability_of_winning = 2 / 5 := by
  sorry

end winning_probability_is_correct_l535_535968


namespace find_f_minus_half_l535_535896

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def function_definition (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x = 4^x

-- Theorem statement
theorem find_f_minus_half {f : ℝ → ℝ}
  (h_odd : is_odd_function f)
  (h_def : function_definition f) :
  f (-1/2) = -2 :=
by
  sorry

end find_f_minus_half_l535_535896


namespace sufficient_but_not_necessary_l535_535194

variable {a b : ℝ}

theorem sufficient_but_not_necessary (ha : a > 0) (hb : b > 0) : 
  (ab > 1) → (a + b > 2) ∧ ¬ (a + b > 2 → ab > 1) :=
by
  sorry

end sufficient_but_not_necessary_l535_535194


namespace arrangement_of_books_l535_535074

theorem arrangement_of_books : 
  (∑ s in (finset.range 10).powerset.filter (λ s, s.card = 4), 1) = 126 :=
sorry

end arrangement_of_books_l535_535074


namespace smallest_munificence_monic_cubic_polynomial_l535_535109

-- Define munificence for a polynomial p(x) as the maximum absolute value of p(x) 
-- on the interval -1 ≤ x ≤ 1
def munificence (p : ℝ → ℝ) : ℝ :=
  Real.Sup (set.image (λ x, abs (p x)) (set.Icc (-1 : ℝ) 1))

-- Define a monic cubic polynomial of the form p(x) = x^3 + px + q
def monic_cubic (p q : ℝ) : ℝ → ℝ :=
  λ x, x^3 + p * x + q

-- The smallest possible munificence for the monic cubic polynomial p(x) = x^3 + px + q
theorem smallest_munificence_monic_cubic_polynomial :
  ∃ p q : ℝ, ∀ x ∈ set.Icc (-1 : ℝ) 1, munificence (monic_cubic p q) = 1 :=
sorry

end smallest_munificence_monic_cubic_polynomial_l535_535109


namespace monotonically_increasing_range_of_a_l535_535913

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 2 * x + 3

theorem monotonically_increasing_range_of_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ -Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2 :=
by
  sorry

end monotonically_increasing_range_of_a_l535_535913


namespace unique_a_l535_535584

noncomputable def f (x : ℝ) : ℝ := x^2 - x

noncomputable def g : ℝ → ℝ 
| x := if x < 1 then -x^3 + (2+1)*x^2 - 2*x else Real.log x

theorem unique_a (a : ℝ) (h : ∀ x > 0, f x ≥ g x) : a = 2 :=
by
  have hx_pos_1 : ∀ x ≥ 1, f x ≥ Real.log x := sorry
  have hx_pos_2 : ∀ x, 0 < x ∧ x < 1 → (f x ≥ -x^3 + (a + 1)x^2 - ax) := sorry
  have unique_a_eq_2 : a = 2 := sorry
  exact unique_a_eq_2

end unique_a_l535_535584


namespace no_distributive_laws_hold_l535_535307

def has_star (R : Type*) := R → R → R

noncomputable def star_fn (a b : ℝ) : ℝ := a + 2 * b

def law_I (x y z : ℝ) : Prop := x \# (y + z) = (x \# y) + (x \# z)
def law_II (x y z : ℝ) : Prop := x + (y \# z) = (x + y) \# (x + z)
def law_III (x y z : ℝ) : Prop := x \# (y \# z) = (x \# y) \# (x \# z)

theorem no_distributive_laws_hold : 
  ¬(∀ x y z : ℝ, law_I x y z) ∧ ¬(∀ x y z : ℝ, law_II x y z) ∧ ¬(∀ x y z : ℝ, law_III x y z) :=
by 
  sorry

end no_distributive_laws_hold_l535_535307


namespace contribution_proof_l535_535449

theorem contribution_proof (total : ℕ) (a_months b_months : ℕ) (a_total b_total a_received b_received : ℕ) :
  total = 3400 →
  a_months = 12 →
  b_months = 16 →
  a_received = 2070 →
  b_received = 1920 →
  (∃ (a_contributed b_contributed : ℕ), a_contributed = 1800 ∧ b_contributed = 1600) :=
by
  sorry

end contribution_proof_l535_535449


namespace principal_correct_l535_535137

def interest_rate (year : ℕ) : ℚ :=
  match year with
  | 1 => 0.03
  | 2 => 0.05
  | 3 => 0.04
  | 4 => 0.06
  | 5 => 0.05
  | _ => 0

noncomputable def principal_amount (total_amount : ℚ) : ℚ :=
  let amounts := (List.range' 1 5).foldr (λ year acc, acc / (1 + interest_rate year)) total_amount
  amounts / (1 + interest_rate 1)

theorem principal_correct :
  principal_amount 3000 = 2396.43 := 
by sorry

end principal_correct_l535_535137


namespace EFGH_perimeter_l535_535342

noncomputable def perimeter_rectangle_EFGH (WE EX WY XZ : ℕ) : Rat :=
  let WX := Real.sqrt (WE ^ 2 + EX ^ 2)
  let p := 15232
  let q := 100
  p / q

theorem EFGH_perimeter :
  let WE := 12
  let EX := 16
  let WY := 24
  let XZ := 32
  perimeter_rectangle_EFGH WE EX WY XZ = 15232 / 100 :=
by
  sorry

end EFGH_perimeter_l535_535342


namespace bisect_A2_C2_by_BO_l535_535624

variables {A B C A1 C1 A2 C2 O: Type}
variables [Triangle A B C] (hAcute : acute_angled A B C)
variables (hAltitudeA : is_altitude A A1 B C) (hAltitudeC : is_altitude C C1 A B)
variables (hSymmetricA2 : symmetric_point A1 (midpoint B C) A2)
variables (hSymmetricC2 : symmetric_point C1 (midpoint A B) C2)
variables (hCircumcenter : is_circumcenter O A B C)

theorem bisect_A2_C2_by_BO (B : Type) (O : Type) (A2 : Type) (C2 : Type) :
  bisect_line_segment B O A2 C2 :=
sorry

end bisect_A2_C2_by_BO_l535_535624


namespace competition_order_l535_535972

variable (A B C D : ℕ)

-- Conditions as given in the problem
axiom cond1 : B + D = 2 * A
axiom cond2 : A + C < B + D
axiom cond3 : A < B + C

-- The desired proof statement
theorem competition_order : D > B ∧ B > A ∧ A > C :=
by
  sorry

end competition_order_l535_535972


namespace construct_equilateral_triangle_l535_535553

noncomputable theory

open EuclideanGeometry

variables {A P : Point} (e : Line)

def exist_equilateral_triangle (A P : Point) (e : Line) : Prop :=
  ∃ (B C : Point) (O : Point),
    equilateral_triangle A B C ∧
    lies_on_line P (line_through B C) ∧
    lies_on_line O e ∧
    circumcenter O A B C

theorem construct_equilateral_triangle : 
  ∀ (e : Line) (A P : Point),
  (∃ (B C : Point),
    equilateral_triangle A B C ∧
    lies_on_line P (line_through B C) ∧
    lies_on_line (circumcenter O A B C) e) :=
sorry

end construct_equilateral_triangle_l535_535553


namespace odd_function_solution_l535_535900

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_solution (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
by
  sorry

end odd_function_solution_l535_535900


namespace circle_center_radius_sum_l535_535304

theorem circle_center_radius_sum:
  let D := {p : ℝ × ℝ | p.1^2 + 8 * p.1 + 18 * p.2 + 98 = -p.2^2 - 6 * p.1} in
  let a := -7 in
  let b := -9 in
  let r := 4 * Real.sqrt 2 in
  (∀ p ∈ D, ((p.1 + 7)^2 + (p.2 + 9)^2 = 32)) →
  a + b + r = -16 + 4 * Real.sqrt 2 :=
by
  intro D a b r h
  -- Define the center and radius of the circle
  have h_center : ∀ p ∈ D, ((p.1 + 7)^2 + (p.2 + 9)^2 = 32), from h
  sorry

end circle_center_radius_sum_l535_535304


namespace range_of_G_l535_535422

def G (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem range_of_G : set.range G = 𝓝[𝓘(ℝ, ℝ)] (-2: ℝ) (2: ℝ) :=
  sorry

end range_of_G_l535_535422


namespace person_time_to_walk_without_walkway_l535_535065

def time_to_walk_without_walkway 
  (walkway_length : ℝ) 
  (time_with_walkway : ℝ) 
  (time_against_walkway : ℝ) 
  (correct_time : ℝ) : Prop :=
  ∃ (vp vw : ℝ), 
    ((vp + vw) * time_with_walkway = walkway_length) ∧ 
    ((vp - vw) * time_against_walkway = walkway_length) ∧ 
     correct_time = walkway_length / vp

theorem person_time_to_walk_without_walkway : 
  time_to_walk_without_walkway 120 40 160 64 :=
sorry

end person_time_to_walk_without_walkway_l535_535065


namespace relationship_between_x_plus_one_and_ex_l535_535872

theorem relationship_between_x_plus_one_and_ex (x : ℝ) : x + 1 ≤ Real.exp x :=
sorry

end relationship_between_x_plus_one_and_ex_l535_535872


namespace florist_has_56_roses_l535_535460

def initial_roses := 50
def roses_sold := 15
def roses_picked := 21

theorem florist_has_56_roses (r0 rs rp : ℕ) (h1 : r0 = initial_roses) (h2 : rs = roses_sold) (h3 : rp = roses_picked) : 
  r0 - rs + rp = 56 :=
by sorry

end florist_has_56_roses_l535_535460


namespace count_terminating_decimals_l535_535143

theorem count_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 500) : 
  (nat.floor (500 / 49) = 10) := 
by
  sorry

end count_terminating_decimals_l535_535143


namespace median_and_range_of_temperatures_l535_535263

theorem median_and_range_of_temperatures :
  let temps := [12, 9, 10, 6, 11, 12, 17],
      sorted_temps := List.sort temps,
      median := sorted_temps.get (sorted_temps.length / 2),
      range := sorted_temps.get (sorted_temps.length - 1) - sorted_temps.get 0
  in
  median = 11 ∧ range = 11 :=
by
  sorry

end median_and_range_of_temperatures_l535_535263


namespace meeting_point_2015_is_C_l535_535171

-- Given definitions based on conditions
variable (x y : ℝ) -- Speeds of the motorcycle and the cyclist
variable (A B C D : Point) -- Points on segment AB
variable (meetings : ℕ → Point) -- Function representing the meeting point sequence

-- Conditions stating the alternating meeting pattern
axiom start_at_A (n : ℕ) : meetings (2 * n + 1) = C
axiom start_at_B (n : ℕ) : meetings (2 * n + 2) = D

-- The theorem statement to be proved
theorem meeting_point_2015_is_C : meetings 2015 = C := sorry

end meeting_point_2015_is_C_l535_535171


namespace sum_all_products_eq_l535_535157

def group1 : List ℚ := [3/4, 3/20] -- Using 0.15 as 3/20 to work with rationals
def group2 : List ℚ := [4, 2/3]
def group3 : List ℚ := [3/5, 6/5] -- Using 1.2 as 6/5 to work with rationals

def allProducts (a b c : List ℚ) : List ℚ :=
  List.bind a (fun x =>
  List.bind b (fun y =>
  List.map (fun z => x * y * z) c))

theorem sum_all_products_eq :
  (allProducts group1 group2 group3).sum = 7.56 := by
  sorry

end sum_all_products_eq_l535_535157


namespace exists_point_P_l535_535462

variables (G : Type) [graph G]
variables (n k : ℕ)
variable (no_triangles : ∀ (P Q R : G), ¬ (P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ edge P Q ∧ edge Q R ∧ edge R P)) 

theorem exists_point_P (G : Type) [graph G] (n k : ℕ) (no_triangles : ∀ (P Q R : G), ¬ (P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ edge P Q ∧ edge Q R ∧ edge R P)) :
  ∃ (P : G), ∃ (H : (G.edges.card ≤ k ∧ n ≥ 3)), (∑' Q ∉ adj P, G.adj_edges.card) ≤ k * (1 - (4 * k) / (n^2)) := 
sorry

end exists_point_P_l535_535462


namespace day_142_2003_is_Thursday_l535_535957

-- Define the days of the week in a custom data type
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

-- Define the function to compute the day of the week for any given day number
def dayOfWeek (start_day : DayOfWeek) (days_after : ℕ) : DayOfWeek :=
  match start_day with
  | Sunday    => DayOfWeek.casesOn (days_after % 7)
  | Monday    => DayOfWeek.casesOn (1 + days_after % 7)
  | Tuesday   => DayOfWeek.casesOn (2 + days_after % 7)
  | Wednesday => DayOfWeek.casesOn (3 + days_after % 7)
  | Thursday  => DayOfWeek.casesOn (4 + days_after % 7)
  | Friday    => DayOfWeek.casesOn (5 + days_after % 7)
  | Saturday  => DayOfWeek.casesOn (6 + days_after % 7)
  end

-- Given conditions:
def day_15_2003 := Wednesday

-- Proof goal:
theorem day_142_2003_is_Thursday : dayOfWeek day_15_2003 (142 - 15) = Thursday :=
by
  -- Let's add a sorry for now as this is only the statement, not the actual proof
  sorry

end day_142_2003_is_Thursday_l535_535957


namespace transform_line_eq_l535_535232

section
variables (a : ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![a, 1]])
variables (α : Vector (Fin 2) ℝ := ![1, 1])
variables (λ : ℝ := 3)

-- Condition: One of the eigenvalues of M is 3 and its corresponding eigenvector is α.
variable (eigenvalue_eigenvector_condition : M.mulVec α = λ • α)

-- Finding a under the given eigenvalue and eigenvector condition
lemma find_a : a = 2 :=
by
  have h : M.mulVec α = ![1 + 2, a + 1] := rfl
  rw [eigenvalue_eigenvector_condition, Matrix.mulVec] at h
  finish

-- Updated matrix M with a = 2
noncomputable def Matrix_updated : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![2, 1]]

variable (l1_condition : x1 + 2 * y1 + 1 = 0) -- Condition: Line l1
variable (transformation : Matrix_updated.mulVec ![x1, y1] = ![x2, y2]) -- Condition: transformation by M

-- Prove the resultant line equation l2 is x+1=0
theorem transform_line_eq : x2 + 1 = 0 :=
sorry

end

end transform_line_eq_l535_535232


namespace odd_function_l535_535904

def f (x : ℝ) : ℝ :=
  if x > 0 then
    x^3 + x + 1
  else if x < 0 then
    x^3 + x - 1
  else 
    0

theorem odd_function (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
begin
  intros x h,
  have h_neg : f (-x) = -f x, from h_odd x,
  have h_nonpos : f x = -f (-x), {
    rw [h_neg, h_pos (-x)],
    simp at *,
    sorry
  },
  sorry
end

end odd_function_l535_535904


namespace problem_solution_l535_535673

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10

theorem problem_solution : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 0 :=
by
  sorry

end problem_solution_l535_535673


namespace simplify_expression_l535_535350

theorem simplify_expression (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - a^2) / (b^2 - a^2) =
  (a^3 - 3 * a * b^2 + 2 * b^3) / (a * b * (b + a)) :=
by
  sorry

end simplify_expression_l535_535350


namespace david_marks_in_physics_l535_535504

theorem david_marks_in_physics : 
  ∀ (P : ℝ), 
  let english := 72 
  let mathematics := 60 
  let chemistry := 62 
  let biology := 84 
  let average_marks := 62.6 
  let num_subjects := 5 
  let total_marks := average_marks * num_subjects 
  let known_marks := english + mathematics + chemistry + biology 
  total_marks - known_marks = P → P = 35 :=
by
  sorry

end david_marks_in_physics_l535_535504


namespace alberto_spent_more_l535_535809

-- Define the expenses of Alberto and Samara
def alberto_expenses : ℕ := 2457
def samara_oil_expense : ℕ := 25
def samara_tire_expense : ℕ := 467
def samara_detailing_expense : ℕ := 79
def samara_total_expenses : ℕ := samara_oil_expense + samara_tire_expense + samara_detailing_expense

-- State the theorem to prove the difference in expenses
theorem alberto_spent_more :
  alberto_expenses - samara_total_expenses = 1886 := by
  sorry

end alberto_spent_more_l535_535809


namespace factorial_division_l535_535494

theorem factorial_division : (50! / 48!) = 2450 := by
  sorry

end factorial_division_l535_535494


namespace number_of_solutions_l535_535842

theorem number_of_solutions : 
  ∃ n : ℕ, 
  n = 90 ∧ 
  ∀ θ : ℝ, 
  0 < θ ∧ θ < 3 * Real.pi → 
  tan (7 * Real.pi * Real.cos θ) = cot (3 * Real.pi * Real.sin θ) → 
  n = 90 := 
sorry

end number_of_solutions_l535_535842


namespace meeting_point_2015th_l535_535177

-- Define the parameters of the problem
variables (A B C D : Type)
variables (x y t : ℝ) -- Speeds and the initial time delay

-- State the problem as a theorem
theorem meeting_point_2015th (start_times_differ : t > 0)
                            (speeds_pos : x > 0 ∧ y > 0)
                            (pattern : ∀ n : ℕ, (odd n → (meeting_point n = C)) ∧ (even n → (meeting_point n = D)))
                            (n = 2015) :
  meeting_point n = C :=
  sorry

end meeting_point_2015th_l535_535177


namespace total_crackers_l535_535501

-- Define the conditions
def boxes_Darren := 4
def crackers_per_box := 24
def boxes_Calvin := 2 * boxes_Darren - 1

-- Define the mathematical proof problem
theorem total_crackers : 
  let total_Darren := boxes_Darren * crackers_per_box
  let total_Calvin := boxes_Calvin * crackers_per_box
  total_Darren + total_Calvin = 264 :=
by
  sorry

end total_crackers_l535_535501


namespace find_other_number_l535_535369

theorem find_other_number (lcm_ab hcf_ab : ℕ) (A : ℕ) (h_lcm: Nat.lcm A (B) = lcm_ab)
  (h_hcf : Nat.gcd A (B) = hcf_ab) (h_a : A = 48) (h_lcm_value: lcm_ab = 192) (h_hcf_value: hcf_ab = 16) :
  B = 64 :=
by
  sorry

end find_other_number_l535_535369


namespace meeting_2015th_at_C_l535_535158

-- Conditions Definitions
variable (A B C D P : Type)
variable (x y t : ℝ)  -- speeds and starting time difference
variable (mw cyclist : ℝ → ℝ)  -- paths of motorist and cyclist

-- Proof statement
theorem meeting_2015th_at_C 
(Given_meeting_pattern: ∀ n : ℕ, odd n → (mw (n * (x + y))) = C):
  (mw (2015 * (x + y))) = C := 
by 
  sorry  -- Proof omitted

end meeting_2015th_at_C_l535_535158


namespace num_integers_satisfying_abs_leq_bound_l535_535242

theorem num_integers_satisfying_abs_leq_bound : ∃ n : ℕ, n = 19 ∧ ∀ x : ℤ, |x| ≤ 3 * Real.sqrt 10 → (x ≥ -9 ∧ x ≤ 9) := by
  sorry

end num_integers_satisfying_abs_leq_bound_l535_535242


namespace jelly_cost_l535_535322

-- Definitions based on the given conditions.
def cost_of_sandwich (B J : ℕ) : ℕ := 3 * B + 6 * J

def total_cost (N B J : ℕ) : ℕ := N * (cost_of_sandwich B J)

-- Given conditions as Lean definitions.
variables (N B J : ℕ)
variables (hN_pos : N > 1)
variables (h_total_cost : total_cost N B J = 336)

-- The goal is to prove the cost of jelly in dollars.
theorem jelly_cost (N B J : ℕ) (hN_pos : N > 1) (h_total_cost : total_cost N B J = 336) : 
  (N * J * 6) / 100 = 2.10 := 
sorry

end jelly_cost_l535_535322


namespace smallest_n_fig2_exists_not_possible_n25_fig4_not_possible_n39_fig4_smallest_n_fig4_exists_l535_535767

-- Part (a)
theorem smallest_n_fig2_exists : ∃ n : ℕ, (∀ (a b : ℕ) (H : a ≠ b) (H1 : a b ≤ n) 
  (connected : is_connected a b Fig2), coprime (a - b) n) ∧ n = 4 :=
begin
  sorry,
end

-- Part (b)
theorem not_possible_n25_fig4 : ¬ (∃ (arrangement : list ℕ), length arrangement = number_of_circles Fig4 ∧ 
  all_numbers_in_arrangement_bounded_by_n 25 arrangement ∧ 
  arrangement_satisfies_properties 25 arrangement) :=
begin
  sorry,
end

-- Part (c)
theorem not_possible_n39_fig4 : ¬ (∃ (arrangement : list ℕ), length arrangement = number_of_circles Fig4 ∧ 
  all_numbers_in_arrangement_bounded_by_n 39 arrangement ∧ 
  arrangement_satisfies_properties 39 arrangement) :=
begin
  sorry,
end

-- Part (d)
theorem smallest_n_fig4_exists : ∃ n : ℕ, (∀ (a b : ℕ) (H : a ≠ b) (H1 : a b ≤ n) 
  (connected : is_connected a b Fig4), coprime (a - b) n) ∧ n = 105 :=
begin
  sorry,
end

end smallest_n_fig2_exists_not_possible_n25_fig4_not_possible_n39_fig4_smallest_n_fig4_exists_l535_535767


namespace perp_iff_sum_squares_eq_l535_535932

variables {V : Type*} [InnerProductSpace ℝ V]

/-- Necessary and sufficient condition for two vectors a and b to be perpendicular -/
theorem perp_iff_sum_squares_eq (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a + b) ⬝ (a + b) = ∥a∥^2 + ∥b∥^2 ↔ a ⬝ b = 0 :=
by
  sorry

end perp_iff_sum_squares_eq_l535_535932


namespace find_tabitha_age_l535_535123

-- Define the conditions
variable (age_started : ℕ) (colors_started : ℕ) (years_future : ℕ) (future_colors : ℕ)

-- Let's specify the given problem's conditions:
axiom h1 : age_started = 15          -- Tabitha started at age 15
axiom h2 : colors_started = 2        -- with 2 colors
axiom h3 : years_future = 3          -- in three years
axiom h4 : future_colors = 8         -- she will have 8 different colors

-- The proof problem we need to state:
theorem find_tabitha_age : ∃ age_now : ℕ, age_now = age_started + (future_colors - colors_started) - years_future := by
  sorry

end find_tabitha_age_l535_535123


namespace trig_positive_values_l535_535429

theorem trig_positive_values :
  (sin (305 * real.pi / 180) * cos (460 * real.pi / 180) > 0) ∧ 
  (cos (378 * real.pi / 180) * sin (1100 * real.pi / 180) > 0) ∧ 
  (tan (188 * real.pi / 180) * cos (158 * real.pi / 180) ≤ 0) ∧ 
  (tan (400 * real.pi / 180) * tan (470 * real.pi / 180) ≤ 0) :=
by
  -- Proof is omitted
  sorry

end trig_positive_values_l535_535429


namespace count_odd_numbers_greater_than_1000_l535_535156

-- Define the conditions
def digits : set ℕ := {0, 1, 2, 3, 4}
def valid_digit (d : ℕ) : Prop := d ∈ digits

-- Define a digit chosen without repetition, greater than 1000, odd
def is_odd (n : ℕ) : Prop := n % 2 = 1
def has_unique_digits (n : ℕ) : Prop := (to_list n.to_string).nodup
def greater_than_1000 (n : ℕ) : Prop := n > 1000

-- Define the problem statement
theorem count_odd_numbers_greater_than_1000 : 
  (∑ n in (finset.range 10000), 
    if greater_than_1000 n ∧ is_odd n ∧ has_unique_digits n 
    then 1 else 0) = 72 :=
  sorry

end count_odd_numbers_greater_than_1000_l535_535156


namespace suzie_reads_pages_hour_l535_535677

-- Declaration of the variables and conditions
variables (S : ℕ) -- S is the number of pages Suzie reads in an hour
variables (L : ℕ) -- L is the number of pages Liza reads in an hour

-- Conditions given in the problem
def reads_per_hour_Liza : L = 20 := sorry
def reads_more_pages : L * 3 = S * 3 + 15 := sorry

-- The statement we want to prove:
theorem suzie_reads_pages_hour : S = 15 :=
by
  -- Proof steps needed here (omitted due to the instruction)
  sorry

end suzie_reads_pages_hour_l535_535677


namespace ineq_triangle_inscribed_circle_l535_535804

theorem ineq_triangle_inscribed_circle (a b c r : ℝ) (s : ℝ) 
  (h1 : s = (a + b + c) / 2) (h2 : r = sqrt ((s * (s - a) * (s - b) * (s - c)) / s)) :
  (1 / (s - a)^2 + 1 / (s - b)^2 + 1 / (s - c)^2) ≥ 1 / r^2 :=
by
  sorry

end ineq_triangle_inscribed_circle_l535_535804


namespace least_number_to_subtract_l535_535132

-- Define the given number, 724946
def given_number : ℕ := 724946

-- Define the prime numbers 37 and 53
def prime_37 : ℕ := 37
def prime_53 : ℕ := 53

-- Define the LCM of 37 and 53, which is 37 * 53 = 1961
def lcm_37_and_53 : ℕ := prime_37 * prime_53

-- The problem statement in Lean
theorem least_number_to_subtract :
  ∃ (least_subtract : ℕ), given_number - least_subtract = 723789 ∧ (723789 % prime_37 = 0 ∧ 723789 % prime_53 = 0) := 
begin
  -- We assert that the least number to subtract is 1157
  use 1157,
  split,
  { -- Show that 724946 - 1157 = 723789
    exact rfl,
  },
  { -- Show that 723789 is divisible by both 37 and 53
    split,
    { -- 723789 is divisible by 37
      exact rfl,
    },
    { -- 723789 is divisible by 53
      exact rfl,
    }
  }
end

end least_number_to_subtract_l535_535132


namespace nolan_money_left_l535_535329

variable (m b : ℝ)
variable (h : (1 / 3) * m = (1 / 4) * b)

theorem nolan_money_left :
  (m - b) / m = 0 :=
by 
  have : b = (4 / 3) * m := by rw [← h, div_eq_mul_inv, ←mul_assoc, (show (1 : ℝ) / 3 * 3 = 1, by norm_num), mul_one]
  rw [this, (show (m - (4 / 3) * m) / m = 0, by ring)]
  sorry

end nolan_money_left_l535_535329


namespace distinct_points_count_l535_535610

def A : set ℕ := {1, 2, 3}
def B : set ℕ := {1, 4, 5, 6}

noncomputable def num_distinct_points : ℕ :=
  let coords := (A × B) ∪ (B × A)
  coords.card - ({(1, 1)} : set (ℕ × ℕ)).card

theorem distinct_points_count : num_distinct_points = 23 :=
sorry

end distinct_points_count_l535_535610


namespace meet_opposite_in_6_seconds_l535_535742

noncomputable def meet_in_opposite_direction (s : ℝ) (t₁ t₂ : ℝ) (h1 : t₂ = t₁ + 5) 
(h2 : 30 * (s / t₁ - s / t₂) = s) : ℝ :=
let speed₁ := s / t₁
let speed₂ := s / t₂
let relative_speed := speed₁ + speed₂ in
s / relative_speed

theorem meet_opposite_in_6_seconds : ∀ (s t₁ t₂ : ℝ), t₂ = t₁ + 5 → 30 * (s / t₁ - s / t₂) = s → 
  meet_in_opposite_direction s t₁ t₂ = 6 :=
by
  intros s t₁ t₂ h1 h2
  have ht₁ : t₁ = 10 := sorry
  have ht₂ : t₂ = 15 := sorry
  rw [meet_in_opposite_direction]
  rw [h1, h2]
  suffices relative_speed: (s / 10 + s / 15) = s / 6 by rw [<-relative_speed]; norm_num
  sorry

end meet_opposite_in_6_seconds_l535_535742


namespace limit_of_p_n_is_tenth_l535_535438

noncomputable def p_n (n : ℕ) : ℝ := sorry -- Definition of p_n needs precise formulation.

def tends_to_tenth_as_n_infty (p : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (p n - 1/10) < ε

theorem limit_of_p_n_is_tenth : tends_to_tenth_as_n_infty p_n := sorry

end limit_of_p_n_is_tenth_l535_535438


namespace prove_options_l535_535759

-- Define the problem conditions
def point_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 5
def point_on_line (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = 1
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define specific conditions
def point_P1 : ℝ × ℝ := (1, 3)
def point_P2 : ℝ × ℝ := (2, 1)
def point_P3 : ℝ × ℝ := (1, 1)
def circle_eq : ℝ × ℝ := (x, y)
def line_eq : ℝ × ℝ := (2*x, y-5)

-- Define the target tangent line
def tangent_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the line and its direction vector
def line_direction_vector (a b c : ℝ) (v : ℝ × ℝ) : Prop := v = (1, 2)

-- State the theorem to prove both conditions
theorem prove_options :
  (point_on_circle 2 1) ∧ 
  (tangent_line 2 1) ∧ 
  (line_direction_vector 2 (-1) (-1) (1, 2)) :=
by
  sorry

end prove_options_l535_535759


namespace abc_less_than_one_l535_535688

variables {a b c : ℝ}

theorem abc_less_than_one (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1: a^2 < b) (h2: b^2 < c) (h3: c^2 < a) : a < 1 ∧ b < 1 ∧ c < 1 := by
  sorry

end abc_less_than_one_l535_535688


namespace ellipse_properties_l535_535201

variable (a b k : ℝ)

theorem ellipse_properties (h₁: a > 0) (h₂: b > 0) (h₃: a > b)
  (hA: (-2 : ℝ), 0) ∈ ({p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1})
  (hP: (1 : ℝ), 3 / 2 ∈ ({p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1})) :
  ((a = 2) ∧ (b = sqrt 3) ∧ (c = sqrt (a^2 - b^2)) ∧ (ecc = c / a) ∧ 
   (a = 2) ∧ (b = sqrt 3)) ∧ (ecc = 1 / 2) ∧ (k = 3 / 2) :=
by
  sorry

end ellipse_properties_l535_535201


namespace find_m_l535_535256

theorem find_m (x : ℝ) (m : ℝ) (h : ∃ x, (x - 2) ≠ 0 ∧ (4 - 2 * x) ≠ 0 ∧ (3 / (x - 2) + 1 = m / (4 - 2 * x))) : m = -6 :=
by
  sorry

end find_m_l535_535256


namespace geometric_sequence_quadratic_roots_l535_535907

theorem geometric_sequence_quadratic_roots
    (a b : ℝ)
    (h_geometric : ∃ q : ℝ, b = 2 * q ∧ a = 2 * q^2) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + (1 / 3) = 0 ∧ a * x2^2 + b * x2 + (1 / 3) = 0) :=
by
  sorry

end geometric_sequence_quadratic_roots_l535_535907


namespace difference_max_min_is_7_l535_535997

-- Define the number of times Kale mowed his lawn during each season
def timesSpring : ℕ := 8
def timesSummer : ℕ := 5
def timesFall : ℕ := 12

-- Statement to prove
theorem difference_max_min_is_7 : 
  (max timesSpring (max timesSummer timesFall)) - (min timesSpring (min timesSummer timesFall)) = 7 :=
by
  -- Proof would go here
  sorry

end difference_max_min_is_7_l535_535997


namespace find_matrix_l535_535880

noncomputable def M := λ (a b c d : ℝ), matrix.vec_cons
  (vector.vec_cons a (vector.vec_cons b vector.nil))
  (matrix.vec_cons (vector.vec_cons c (vector.vec_cons d vector.nil)) matrix.nil)

theorem find_matrix (a b c d: ℝ) : 
  (M a b c d) * (λ i, if i = 0 then 1 else -1) = (λ i, if i = 0 then 1 else -1) ∧
  (M a b c d) * (λ i, if i = 0 then 1 else 1) = (λ i, if i = 0 then 3 else 1) ↔
  (M a b c d) = λ _ _, (matrix.vec_cons (vector.vec_cons 2 (vector.vec_cons 1 vector.nil))
                              (matrix.vec_cons (vector.vec_cons 0 (vector.vec_cons 1 vector.nil)) matrix.nil)) :=
by sorry

end find_matrix_l535_535880


namespace f_2_eq_1_l535_535579

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 1

theorem f_2_eq_1 (a b : ℝ) (h : f a b (-2) = 1) : f a b 2 = 1 :=
by {
  sorry 
}

end f_2_eq_1_l535_535579


namespace profit_at_marked_price_l535_535608

-- Definitions from conditions
variable (x : ℝ) -- cost price
variable (mp : ℝ) -- marked price

-- Conditions
def condition1 := (1 + 0.2) * x = 0.8 * mp

-- Question rephrased to equivalent proof
theorem profit_at_marked_price : condition1 → (mp - x) / x = 0.5 := 
by
  intro h
  -- the proof goes here
  sorry

end profit_at_marked_price_l535_535608


namespace min_possible_value_of_max_intersecting_sets_l535_535874

noncomputable def min_intersecting_sets (sets : Fin 11 → Finset ℕ) : ℕ :=
  -- Define the minimum value satisfying the proof, initialize it to a high number
  Finset.sup (Finset.range 11) (λ i, sets i ∩ sets ((i+1) % 11))

theorem min_possible_value_of_max_intersecting_sets :
  ∃ (sets : Fin 11 → Finset ℕ) (h : ∀ i j, i < j → (sets i ∩ sets j).nonempty),
    (min_intersecting_sets sets = 4) := 
sorry

end min_possible_value_of_max_intersecting_sets_l535_535874


namespace total_crackers_l535_535500

-- Definitions based on conditions
def boxes_darren_bought : ℕ := 4
def crackers_per_box : ℕ := 24
def boxes_calvin_bought : ℕ := (2 * boxes_darren_bought) - 1

-- The statement to prove
theorem total_crackers (boxes_darren_bought = 4) (crackers_per_box = 24) : 
  (boxes_darren_bought * crackers_per_box) + (boxes_calvin_bought * crackers_per_box) = 264 := 
by 
  sorry

end total_crackers_l535_535500


namespace min_value_of_x_l535_535951

----------------
-- Definitions --
----------------

def x_positive (x : ℝ) : Prop := x > 0
def log_inequality (x : ℝ) : Prop := log x ≥ log 3 + log (sqrt x)

--------------------
-- Proof Statement --
--------------------

theorem min_value_of_x (x : ℝ) (h_pos : x_positive x) (h_ineq : log_inequality x) : x ≥ 9 := 
by
  sorry

end min_value_of_x_l535_535951


namespace area_of_hexagon_l535_535877

structure HexagonalPolygon :=
(P Q R S T U V : Point)
(PQ QR UT TU : ℝ)
(h_p1 : PQ = 8)
(h_p2 : QR = 10)
(h_p3 : UT = 7)
(h_p4 : TU = 3)
(rect_PQRV : IsRectangle P Q R V)
(rect_VUT : IsRectangle V U T)

def area_PQRSTU (hp : HexagonalPolygon) : ℝ := 
hp.PQ * hp.QR - (hp.VU * hp.VT)

theorem area_of_hexagon (hp : HexagonalPolygon) : 
  hp.area_PQRSTU = 65 := sorry

end area_of_hexagon_l535_535877


namespace limit_x_y_part_a_limit_x_y_part_b_l535_535312

noncomputable def recurrence_x (x y : ℕ → ℝ) (n : ℕ) : ℝ := (x n + y n) / 2
noncomputable def recurrence_y (x y : ℕ → ℝ) (n : ℕ) : ℝ := real.sqrt ((recurrence_x x y n) * (y n))

-- Theorem statements
theorem limit_x_y_part_a (x0 y0 : ℝ) (h : 0 ≤ x0 ∧ x0 < y0) :
  let x := λ n, if n = 0 then x0 else recurrence_x x y (n-1),
      y := λ n, if n = 0 then y0 else recurrence_y x y (n-1)
  in ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (x n - (sqrt (y0^2 - x0^2) / real.arccos (x0 / y0))) < ε ∧
                          abs (y n - (sqrt (y0^2 - x0^2) / real.arccos (x0 / y0))) < ε :=
begin
  sorry
end

theorem limit_x_y_part_b (x0 y0 : ℝ) (h : 0 < y0 ∧ y0 < x0) :
  let x := λ n, if n = 0 then x0 else recurrence_x x y (n-1),
      y := λ n, if n = 0 then y0 else recurrence_y x y (n-1)
  in ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (x n - (sqrt (x0^2 - y0^2) / real.arcosh (x0 / y0))) < ε ∧
                          abs (y n - (sqrt (x0^2 - y0^2) / real.arcosh (x0 / y0))) < ε :=
begin
  sorry
end

end limit_x_y_part_a_limit_x_y_part_b_l535_535312


namespace spinner_probability_C_l535_535787

noncomputable def A := 1 / 4
noncomputable def B := 1 / 3
noncomputable def D := 1 / 6

theorem spinner_probability_C :
  let C := 1 - A - B - D in
  C = 1 / 4 :=
by
  let x := 1 - A - B - D
  have eq_x : x = 1 / 4 := sorry
  exact eq_x

end spinner_probability_C_l535_535787


namespace max_binomial_coefficient_sum_reciprocals_ineq_l535_535989

variables {m n : ℕ}
variable {a : ℕ → ℝ}

-- Given that {a} is an arithmetic sequence and the coefficients of the first three terms of the expansion
-- of (1 + x/2)^m are a_1, a_2, and a_3 respectively.
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)

theorem max_binomial_coefficient (hm : m > 0)
  (h1 : a 1 = 1) (h2 : a 2 = m / 2) (h3 : a 3 = m * (m - 1) / 8) (ha : is_arithmetic_seq a) :
  (m = 8 → max_coeff_term = (35 / 8 : ℝ) * (2 / m)^4) :=
by
  sorry

theorem sum_reciprocals_ineq (h_arith: is_arithmetic_seq a)
  (h_a : ∀ n, a n = 3 * n - 2) 
  (hn : n ≥ 2) :
  (∑ i in finset.range ((n^2) - n + 1), 1 / a (n + i)) > 1 / 3 :=
by
  sorry

end max_binomial_coefficient_sum_reciprocals_ineq_l535_535989


namespace evenly_spaced_even_probability_find_p_plus_q_l535_535736

noncomputable def evenly_spaced_probability (a n : ℕ) : ℚ :=
if (1 ≤ a) ∧ (a + n ≤ 20) ∧ (a + 2 * n ≤ 20) then
  2^(-a:ℤ) * 2^(-(a + n):ℤ) * 2^(-(a + 2 * n):ℤ)
else
  0

noncomputable def total_probability : ℚ :=
∑ a in finset.range 19,
∑ n in finset.range (((20 - a) / 2) + 1),
  evenly_spaced_probability a n

theorem evenly_spaced_even_probability : total_probability = 6 / 3280 :=
sorry

theorem find_p_plus_q : 6 + 3280 = 3286 :=
sorry

end evenly_spaced_even_probability_find_p_plus_q_l535_535736


namespace nonagon_diagonal_intersection_points_l535_535811

def number_of_intersection_points (n : ℕ) : ℕ :=
  nat.choose n 4

theorem nonagon_diagonal_intersection_points :
  number_of_intersection_points 9 = 126 :=
by {
  unfold number_of_intersection_points,
  norm_num,
  sorry
}

end nonagon_diagonal_intersection_points_l535_535811


namespace part1_part2_part3_l535_535447

-- Definitions and assumptions
def N (D : Set (ℤ × ℤ)) : ℕ := sorry

def regionA (n : ℕ) : Set (ℤ × ℤ) :=
  {p | ∃ k, 1 ≤ k ∧ k ≤ n ∧ p.snd = k^2 ∧ p.fst = k }

def regionB (n : ℕ) : Set (ℤ × ℤ) :=
  {p | ∃ k, 1 ≤ k ∧ k ≤ n ∧ p.snd = k^2 ∧ p.fst = k } ∪
  {p | ∃ k, k = 1 ∧ 1 ≤ p.snd ∧ p.snd ≤ n^2 }

-- Proof problem 1
theorem part1 (n : ℕ) (hn : 1 < n) : 
  N (regionA n) = (n * (n + 1) * (2 * n + 1)) / 6 :=
sorry

-- Proof problem 2
theorem part2 (n : ℕ) (hn : 1 < n) : 
  N (regionB n) = (4 * n^3 - 3 * n^2 + 5 * n) / 6 :=
sorry

-- Proof problem 3
theorem part3 (n : ℕ) (hn : 1 < n) : 
  (∑ k in Finset.range (n^2 + 1), int.floor (Real.sqrt k)) = (4 * n^3 - 3 * n^2 + 5 * n) / 6 :=
sorry

end part1_part2_part3_l535_535447


namespace valid_N_count_l535_535485

theorem valid_N_count : ∃ N : Finset ℕ, 
  (∀ n ∈ N, 
    100 ≤ n ∧ n < 1000 ∧
    let N_4 := (64 * (n / 64 % 4) + 16 * (n / 16 % 4) + 4 * (n / 4 % 4) + (n % 4)) in
    let N_7 := (49 * (n / 49 % 7) + 7 * (n / 7 % 7) + (n % 7)) in
    (N_4 + N_7) % 1000 = (2 * n) % 1000)
  ∧ N.card = 15 := 
sorry

end valid_N_count_l535_535485


namespace part_I_part_II_l535_535586

noncomputable theory

open Real

def parabola (p : ℝ) : set (ℝ × ℝ) := {point ∣ ∃ y, point = (y^2/(2*p), y)}

def distance (point1 point2 : ℝ × ℝ) : ℝ :=
  sqrt ((point1.1 - point2.1)^2 + (point1.2 - point2.2)^2)

theorem part_I (p : ℝ) (p_pos : 0 < p) (M : ℝ × ℝ) (hM : M ∈ parabola p)
  (F := (p, 0)) (hMF : distance M F = 4) :
  p = 2 := by
  sorry

def intersects (p k : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ x1 y1 x2 y2, A = (x1, y1) ∧ B = (x2, y2) ∧
    y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧
    y1 = k * (x1 - 4) ∧ y2 = k * (x2 - 4)

theorem part_II (A B O : ℝ × ℝ) (hA : A ∈ parabola 2)
  (hB : B ∈ parabola 2) (hO : O = (0,0)) (k : ℝ)
  (line_l : intersects 2 k A B) :
  let S := 1/2 * abs (A.1 * B.2 - A.2 * B.1)
  in S ≥ 16 := by
  sorry

end part_I_part_II_l535_535586


namespace external_angle_regular_octagon_l535_535348

-- Assume G is a generic octagon and consider its properties
variables (ABCDEGFH : Type) [regular_octagon ABCDEFGH]

-- Define the main theorem
theorem external_angle_regular_octagon (o: regular_octagon ABCDEFPGH) 
  (A B C : Point)
  (hAB: is_side A B o)
  (hBC: is_side B C o)
  (P : Point)
  (hAP: lies_on_ray A P hAB)
  (hCP: lies_on_ray C P hBC) :
  measure_angle A P B = 45 :=
sorry

end external_angle_regular_octagon_l535_535348


namespace train_cross_pole_time_l535_535987

-- Define the conditions
def train_length : ℝ := 800
def train_speed_kmh : ℝ := 180
def train_speed_ms : ℝ := train_speed_kmh * (5/18)

-- Statement of the proof problem
theorem train_cross_pole_time : (train_length / train_speed_ms) = 16 := 
by
  -- Skipping the proof
  sorry

end train_cross_pole_time_l535_535987


namespace money_left_l535_535466

theorem money_left 
  (salary : ℝ)
  (spent_on_food : ℝ)
  (spent_on_rent : ℝ)
  (spent_on_clothes : ℝ)
  (total_spent : ℝ)
  (money_left : ℝ)
  (h_salary : salary = 170000)
  (h_food : spent_on_food = salary * (1 / 5))
  (h_rent : spent_on_rent = salary * (1 / 10))
  (h_clothes : spent_on_clothes = salary * (3 / 5))
  (h_total_spent : total_spent = spent_on_food + spent_on_rent + spent_on_clothes)
  (h_money_left : money_left = salary - total_spent) :
  money_left = 17000 :=
by
  sorry

end money_left_l535_535466


namespace maria_travel_fraction_l535_535844

variable (x : ℝ)

theorem maria_travel_fraction (h1 : 400 - x * 400 - (1/4) * (400 - x * 400) = 150)
    : x = 1 / 2 :=
  by
  rw [sub_eq_iff_eq_add] at h1,
  sorry

end maria_travel_fraction_l535_535844


namespace find_a_l535_535541

theorem find_a (a b : ℝ) (r s t : ℝ) 
  (h_poly : 7 * r^3 + 3 * a * r^2 + 6 * b * r + 2 * a = 0)
  (h_roots : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h_pos_roots : r > 0 ∧ s > 0 ∧ t > 0)
  (h_log_sum : Real.log 243 = 5 * Real.log 3)
  (h_sum_squares : r^2 + s^2 + t^2 = 49) :
  a = -850.5 :=
by
  sorry

end find_a_l535_535541


namespace ned_price_per_game_l535_535328

def number_of_games : Nat := 15
def non_working_games : Nat := 6
def total_earnings : Nat := 63
def number_of_working_games : Nat := number_of_games - non_working_games
def price_per_working_game : Nat := total_earnings / number_of_working_games

theorem ned_price_per_game : price_per_working_game = 7 :=
by
  sorry

end ned_price_per_game_l535_535328


namespace compute_zeta6_sum_l535_535657

open Complex

def zeta1 : ℂ := -- define or assume values according to the conditions
def zeta2 : ℂ := -- define or assume values according to the conditions
def zeta3 : ℂ := -- define or assume values according to the conditions

def condition1 := (zeta1 + zeta2 + zeta3 = 2)
def condition2 := (zeta1^2 + zeta2^2 + zeta3^2 = 5)
def condition3 := (zeta1^3 + zeta2^3 + zeta3^3 = 14)

theorem compute_zeta6_sum (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  zeta1^6 + zeta2^6 + zeta3^6 = 128.75 :=
by
  sorry

end compute_zeta6_sum_l535_535657


namespace range_of_a_l535_535587

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ (a < -2 ∨ a > 2) :=
by
  sorry

end range_of_a_l535_535587


namespace not_necessarily_divisible_by_36_l535_535701

theorem not_necessarily_divisible_by_36 (k : ℤ) :
  let n := k * (k + 1) * (k + 2) in
  (n % 11 = 0) → ¬ ∀ m : ℤ, n % 36 = 0 :=
sorry

end not_necessarily_divisible_by_36_l535_535701


namespace statement_A_statement_B_statement_C_statement_D_l535_535542

-- Define the function f
def f (x : ℝ) : ℝ := 4 * sin (2 * x + π / 3)

-- Statements to prove
theorem statement_A : ∀ x : ℝ, f x = 4 * cos (2 * x - π / 6) := sorry

theorem statement_B : ¬(∀ x : ℝ, f (x - π / 6) = f (- (x - π / 6))) := sorry

theorem statement_C : ∀ x, 0 ≤ x ∧ x ≤ π / 3 → f x ≤ 4 := sorry

theorem statement_D : ∀ x : ℝ, f (x + π / 12) = f (- (x + π / 12)) := sorry

end statement_A_statement_B_statement_C_statement_D_l535_535542


namespace f_monotonic_intervals_f_max_min_on_interval_l535_535918

-- Define the function
def f (x : ℝ) : ℝ := 3 * x ^ 3 - 9 * x + 5

-- Statement for Part 1: Monotonic Intervals
theorem f_monotonic_intervals : 
  (∀ x, x < -1 → monotone_increasing f x) ∧
  (∀ x, x > 1 → monotone_increasing f x) ∧
  (∀ x, -1 < x ∧ x < 1 → monotone_decreasing f x) := 
sorry

-- Statement for Part 2: Maximum and Minimum Values on [-3, 3]
theorem f_max_min_on_interval : 
  is_max_on f 3 (-3, 3) ∧ 
  ∀ x, x ∈ [-3, 3] → f x ≤ 59 ∧
  is_min_on f (-3) (-3, 3) ∧ 
  ∀ x, x ∈ [-3, 3] → f x ≥ -49 := 
sorry

end f_monotonic_intervals_f_max_min_on_interval_l535_535918


namespace fraction_identity_l535_535762

theorem fraction_identity :
  (\((3 * 5 : ℚ) / (9 * 11) * ((7 * 9 * 11) / (3 * 5 * 7))) = 1 := by sorry

end fraction_identity_l535_535762


namespace coefficients_square_l535_535670

def a : ℕ → ℕ
| 0      := 1
| 1      := 1
| 2      := 2
| (n+3)  := a n + a (n+1) + a (n+2)

theorem coefficients_square (n : ℕ) (h : n = 1 ∨ n = 9) : a (n-1) = n^2 :=
by by_cases h₁ : n = 1; 
   by_cases h₂ : n = 9; 
   try { rewrite h₁; exact dec_trivial }; 
   try { rewrite h₂; exact dec_trivial }; 
   try { rewrite h₁ in h; contradiction };
   try { rewrite h₂ in h; contradiction };
   sorry

end coefficients_square_l535_535670


namespace max_area_triangle_l535_535965

-- Definitions from conditions
variables {A B C : ℝ} -- angles of triangle
variables {a b c : ℝ} -- corresponding sides opposite to angles A, B, C

-- Condition: b = 4
axiom H1 : b = 4

-- Posing the problem in terms of triangle area and conditions
theorem max_area_triangle (H : (2 * a - c) / b = (Math.cos C) / (Math.cos B))
: let K := (1 / 2) * a * c * Math.sin B in K ≤ 4 * Real.sqrt 3 :=
by
  -- Skipping the proof with sorry.
  sorry

end max_area_triangle_l535_535965


namespace geometric_sequence_T_n_sum_l535_535559

open Nat

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Define the given sequence and the sum recurrence relation
axiom seq_condition : a 1 = 1
axiom sum_condition : ∀ n : ℕ, 1 ≤ n → S (n + 1) - 2 * S n - n - 1 = 0

-- Define the transformed sequences and the sums
def a_plus_one (n : ℕ) := a n + 1
def b (n : ℕ) := n * a n
def T (n : ℕ) := ∑ i in range n, b (i + 1)

-- Question (I): Prove that the sequence {a_n + 1} is a geometric sequence with first term 2 and common ratio 2
theorem geometric_sequence (n : ℕ) :
  ∃ (r : ℝ), r = 2 ∧ (a_plus_one a 1 = 2) ∧ (∀ n, 1 ≤ n → a_plus_one a (n + 1) = r * a_plus_one a n) :=
sorry

-- Question (II): Prove that T_n = (n-1)2^(n+1) - (n(n+1))/2 + 2
noncomputable def T_n_formula (n : ℕ) :=
  (n - 1) * 2^(n + 1) - (n * (n + 1) / 2) + 2

theorem T_n_sum (n : ℕ) :
  T a n = T_n_formula n :=
sorry

end geometric_sequence_T_n_sum_l535_535559


namespace dodecagon_diagonals_intersect_probability_l535_535774

theorem dodecagon_diagonals_intersect_probability :
  ∀ (dodecagon : Type) [regular_polygon dodecagon (12 : nat)],
  let diagonals_count := 54 in
  let intersecting_diagonals_count := 495 in
  (intersecting_diagonals_count : ℚ) / (binom diagonals_count 2) = 15 / 43 :=
by
  intros
  have diagonals : 54 := 54
  have intersections : 495 := 495
  have total_pairs_diagonals : 1431 := binom 54 2
  have probability : ℚ := intersections / total_pairs_diagonals
  rw [Q.to_rat_eq]
  apply Q.eq_of_rat_eq
  norm_num
  exact 15 / 43
  sorry

end dodecagon_diagonals_intersect_probability_l535_535774


namespace election_votes_l535_535031

variable (V : ℝ)

theorem election_votes (h1 : 0.70 * V - 0.30 * V = 192) : V = 480 :=
by
  sorry

end election_votes_l535_535031


namespace part1_part2_l535_535107

def custom_op (a b : ℝ) : ℝ :=
  if a >= b then a * b - a else a * b + b 

theorem part1 :
  custom_op (3 - Real.sqrt 3) (Real.sqrt 3) = 4 * Real.sqrt 3 - 3 :=
by sorry

theorem part2 (x : ℝ) :
  custom_op (2 * x) (x + 1) = 6 → (x = Real.sqrt 3 ∨ x = -5 / 2) :=
by sorry

end part1_part2_l535_535107


namespace complementary_implication_mutually_exclusive_not_implication_l535_535027

-- Definitions of probability space and events
def prob_space (Ω : Type) := Ω → ℝ

noncomputable def P {Ω : Type} (A B : Ω → Prop) [measure_theory.measure_space Ω] := 
  measure_theory.measure_of (measure_theory.outer_measure.of_function (λ s, ∑' i, ∥s i∥))

-- Given conditions
def complementary {Ω : Type} (A B : Ω → Prop) [measure_theory.measure_space Ω] :=
  P A + P B = 1

def mutually_exclusive {Ω : Type} (A B : Ω → Prop) [measure_theory.measure_space Ω] :=
  P (λ ω, A ω ∧ B ω)= 0

-- Theorem statement
theorem complementary_implication {Ω : Type} (A B : Ω → Prop) 
  [measure_theory.measure_space Ω] :
  complementary A B → mutually_exclusive A B :=
sorry

theorem mutually_exclusive_not_implication {Ω : Type} (A B : Ω → Prop) 
  [measure_theory.measure_space Ω] : 
  mutually_exclusive A B → ¬ (complementary A B) :=
sorry

end complementary_implication_mutually_exclusive_not_implication_l535_535027


namespace find_x_plus_y_l535_535215

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 :=
by
  sorry

end find_x_plus_y_l535_535215


namespace path_length_l535_535384

theorem path_length (P Q : ℝ) (n : ℕ) (a : ℕ → ℝ) 
  (h1 : P + Q = 73)
  (h2 : ∑ i in finset.range n, a i = 73) :
  ∑ i in finset.range n, (3 * a i) = 219 :=
  sorry

end path_length_l535_535384


namespace mary_marbles_l535_535325

theorem mary_marbles (initial_marbles gave_away : ℝ) (initial_marbles = 9.0) (gave_away = 3.0) : 
  (initial_marbles - gave_away = 6.0) :=
sorry

end mary_marbles_l535_535325


namespace abs_ineq_one_abs_ineq_two_l535_535699

-- First proof problem: |x-1| + |x+3| < 6 implies -4 < x < 2
theorem abs_ineq_one (x : ℝ) : |x - 1| + |x + 3| < 6 → -4 < x ∧ x < 2 :=
by
  sorry

-- Second proof problem: 1 < |3x-2| < 4 implies -2/3 ≤ x < 1/3 or 1 < x ≤ 2
theorem abs_ineq_two (x : ℝ) : 1 < |3 * x - 2| ∧ |3 * x - 2| < 4 → (-2/3) ≤ x ∧ x < (1/3) ∨ 1 < x ∧ x ≤ 2 :=
by
  sorry

end abs_ineq_one_abs_ineq_two_l535_535699


namespace total_weight_of_bad_carrots_is_correct_l535_535100

variable (carol_carrots : ℕ) (carol_avg_weight : ℕ) (mom_carrots : ℕ) (mom_avg_weight : ℕ)
variable (carol_bad_pct : ℝ) (mom_bad_pct : ℝ)

-- Define the total number of carrots Carol and her mom picked
def carol_total_weight : ℕ := carol_carrots * carol_avg_weight
def mom_total_weight : ℕ := mom_carrots * mom_avg_weight

-- Define the number of bad carrots each picked, rounded down to nearest integer
def carol_bad_carrots : ℕ := int.floor (carol_carrots * carol_bad_pct).to_int
def mom_bad_carrots : ℕ := int.floor (mom_carrots * mom_bad_pct).to_int

-- Define the weight of bad carrots each picked
def carol_bad_weight : ℕ := carol_bad_carrots * carol_avg_weight
def mom_bad_weight : ℕ := mom_bad_carrots * mom_avg_weight

def total_bad_weight : ℕ := carol_bad_weight + mom_bad_weight

theorem total_weight_of_bad_carrots_is_correct :
  carol_carrots = 35 → carol_avg_weight = 90 →
  mom_carrots = 28 → mom_avg_weight = 80 →
  carol_bad_pct = 0.12 → mom_bad_pct = 0.08 →
  total_bad_weight carol_carrots carol_avg_weight mom_carrots mom_avg_weight carol_bad_pct mom_bad_pct = 520 :=
by
  intros
  sorry

end total_weight_of_bad_carrots_is_correct_l535_535100


namespace evaluate_expression_l535_535752

theorem evaluate_expression : 4 * 5 - 3 + 2^3 - 3 * 2 = 19 := by
  sorry

end evaluate_expression_l535_535752


namespace angle_BCD_is_130_degrees_l535_535984

open Real

noncomputable def angle_BCD_degrees (A B C D E : Point) 
  (h1 : diameter E B)
  (h2 : is_parallel (line_segment E B) (line_segment D C))
  (h3 : is_parallel (line_segment A B) (line_segment E D))
  (h4 : ∃ k, angle_degrees A E B = 4 * k ∧ angle_degrees A B E = 5 * k) : ℝ :=
  130

theorem angle_BCD_is_130_degrees (A B C D E : Point) 
  (h1 : diameter E B)
  (h2 : is_parallel (line_segment E B) (line_segment D C))
  (h3 : is_parallel (line_segment A B) (line_segment E D))
  (h4 : ∃ k, angle_degrees A E B = 4 * k ∧ angle_degrees A B E = 5 * k) : 
  angle_degrees B C D = angle_BCD_degrees A B C D E h1 h2 h3 h4 :=
  by
    sorry

end angle_BCD_is_130_degrees_l535_535984


namespace domain_h_l535_535416

def h (x : ℝ) : ℝ := (2 * x - 3) / (x - 5)

theorem domain_h : ∀ x : ℝ, x ∈ (Set.univ \ {5}) ↔ h x ∈ ℝ ∧ x ≠ 5 := by
  intros x
  sorry

end domain_h_l535_535416


namespace num_girls_went_to_spa_l535_535791

-- Define the condition that each girl has 20 nails
def nails_per_girl : ℕ := 20

-- Define the total number of nails polished
def total_nails_polished : ℕ := 40

-- Define the number of girls
def number_of_girls : ℕ := total_nails_polished / nails_per_girl

-- The theorem we want to prove
theorem num_girls_went_to_spa : number_of_girls = 2 :=
by
  unfold number_of_girls
  unfold total_nails_polished
  unfold nails_per_girl
  sorry

end num_girls_went_to_spa_l535_535791


namespace beetle_probability_l535_535048

def grid_width : ℕ := 10
def grid_height : ℕ := 10

def is_horizontal_edge (y : ℕ) : Prop :=
  y = 0 ∨ y = grid_height

def is_vertical_edge (x : ℕ) : Prop :=
  x = 0 ∨ x = grid_width

noncomputable def P (x y : ℕ) : ℝ :=
  if is_horizontal_edge y then 1
  else if is_vertical_edge x then 0
  else sorry  -- Recursive relation and detailed setup skipped

theorem beetle_probability : P 3 4 = 0.6 :=
sorry

end beetle_probability_l535_535048


namespace product_of_all_real_x_satisfying_eq_l535_535381

theorem product_of_all_real_x_satisfying_eq :
  (∀ x : ℝ, (2 * x + 4) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) →
  let quadratic_eq := (1 : ℝ) * x^2 + 19 * x + 4 = 0
  (∃ roots : set ℝ, roots.prod = 4) :=
by
  sorry

end product_of_all_real_x_satisfying_eq_l535_535381


namespace pure_imaginary_square_l535_535246

noncomputable def is_pure_imaginary_number (z : ℂ) : Prop :=
z.re = 0

theorem pure_imaginary_square (x : ℝ) (h : is_pure_imaginary_number ((x:ℂ) + complex.i) ^ 2) : x = 1 ∨ x = -1 :=
sorry

end pure_imaginary_square_l535_535246


namespace arithmetic_seq_5a1_plus_a7_l535_535562

def arithmetic_seq_condition (a : ℕ → ℚ) : Prop :=
∃ d : ℚ, ∀ n : ℕ, a (n+1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℚ) (S n : ℚ) : Prop :=
S = ∑ i in finset.range n, a i

theorem arithmetic_seq_5a1_plus_a7 (a : ℕ → ℚ) (d : ℚ) :
  arithmetic_seq_condition a →
  sum_of_first_n_terms a 6 3 →
  5 * a 0 + a 6 = 12 :=
by
  intros h1 h2
  sorry

end arithmetic_seq_5a1_plus_a7_l535_535562


namespace find_b_l535_535858

noncomputable def general_quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_b (a c : ℝ) (y1 y2 : ℝ) :
  y1 = general_quadratic a 3 c 2 →
  y2 = general_quadratic a 3 c (-2) →
  y1 - y2 = 12 →
  3 = 3 :=
by
  intros h1 h2 h3
  sorry

end find_b_l535_535858


namespace sin_70_eq_1_minus_2k_squared_l535_535189

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * k^2 :=
by
  sorry

end sin_70_eq_1_minus_2k_squared_l535_535189


namespace terminating_fraction_count_l535_535151

theorem terminating_fraction_count :
  (∃ n_values : Finset ℕ, (∀ n ∈ n_values, 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = k * 49)) ∧ n_values.card = 10) :=
by
  -- Placeholder for the proof, does not contribute to the conditions-direct definitions.
  sorry

end terminating_fraction_count_l535_535151


namespace sister_sandcastle_height_l535_535094

theorem sister_sandcastle_height (miki_height : ℝ)
                                (height_diff : ℝ)
                                (h_miki : miki_height = 0.8333333333333334)
                                (h_diff : height_diff = 0.3333333333333333) :
  miki_height - height_diff = 0.5 :=
by
  sorry

end sister_sandcastle_height_l535_535094


namespace Jinwoo_pages_per_day_l535_535731

theorem Jinwoo_pages_per_day (total_pages : ℕ) (days_in_week : ℕ) :
  total_pages = 220 → days_in_week = 7 → (total_pages + days_in_week - 1) / days_in_week = 32 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end Jinwoo_pages_per_day_l535_535731


namespace find_marks_in_biology_l535_535106

/-- 
David's marks in various subjects and his average marks are given.
This statement proves David's marks in Biology assuming the conditions provided.
--/
theorem find_marks_in_biology
  (english : ℕ) (math : ℕ) (physics : ℕ) (chemistry : ℕ) (avg_marks : ℕ)
  (total_subjects : ℕ)
  (h_english : english = 91)
  (h_math : math = 65)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 67)
  (h_avg_marks : avg_marks = 78)
  (h_total_subjects : total_subjects = 5)
  : ∃ (biology : ℕ), biology = 85 := 
by
  sorry

end find_marks_in_biology_l535_535106


namespace matchstick_move_equiv_l535_535682

-- Original incorrect equation
def original_eqn : Prop := 3 ≠ 11 + 1 - 2 - 7

-- Condition after moving one matchstick (hypothesize):
def rearranged_eqn : Prop := 8 = 4 + 4 

-- Proving the revised proposition after a single matchstick movement:
theorem matchstick_move_equiv : original_eqn -> rearranged_eqn :=
by 
  -- Assuming original_eqn condition
  intro h,
  -- Proof required in practical scenario but stating theorem only for problem context.
  sorry

end matchstick_move_equiv_l535_535682


namespace percentage_error_is_98_82_l535_535801

theorem percentage_error_is_98_82 :
  let x := 12 in
  let correct_val := (x * (5 / 3)) - 3 in
  let incorrect_val := (x * (3 / 5)) - 7 in
  abs ((correct_val - incorrect_val) / correct_val) * 100 = 98.82 := by
  let x := 12
  let correct_val := (x * (5 / 3)) - 3
  let incorrect_val := (x * (3 / 5)) - 7
  simp only [correct_val, incorrect_val]
  norm_num
  sorry

end percentage_error_is_98_82_l535_535801


namespace at_most_one_triangle_l535_535726

-- Definitions from the conditions
def City : Type := ℕ
constant cities : Fin 101 → City
constant Airline : Type := ℕ
constant airlines : Fin 99 → Airline
constant connected_by : City → City → Airline → Prop

-- Axioms from the conditions
axiom city_count : ∃ (city_list : Fin 101 → City), ∀ c, c ∈ city_list
axiom airline_count : ∃ (airline_list : Fin 99 → Airline), ∀ a, a ∈ airline_list
axiom flight_existence : ∀ (c1 c2 : City), c1 ≠ c2 → ∃ a, connected_by c1 c2 a
axiom all_airlines_depart : ∀ (c : City), ∀ a, ∃ c2, connected_by c c2 a ∧ c ≠ c2

-- Theorem to prove
theorem at_most_one_triangle :
  ∀ (c1 c2 c3 : City) (a : Airline),
  (connected_by c1 c2 a ∧ connected_by c2 c3 a ∧ connected_by c3 c1 a) →
  ∀ (d1 d2 d3 : City) (b : Airline),
  (connected_by d1 d2 b ∧ connected_by d2 d3 b ∧ connected_by d3 d1 b) →
  (c1 = d1 ∧ c2 = d2 ∧ c3 = d3 ∧ a = b) ∨
  (c1 = d1 ∧ c2 = d3 ∧ c3 = d2 ∧ a = b) ∨
  (c1 = d2 ∧ c2 = d3 ∧ c3 = d1 ∧ a = b) ∨
  (c1 = d2 ∧ c2 = d1 ∧ c3 = d3 ∧ a = b) ∨
  (c1 = d3 ∧ c2 = d1 ∧ c3 = d2 ∧ a = b) ∨
  (c1 = d3 ∧ c2 = d2 ∧ c3 = d1 ∧ a = b) :=
sorry

end at_most_one_triangle_l535_535726


namespace problem_1_problem_2_l535_535230

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) - abs (x + 2)

-- Prove the inequality f(x, 1) ≤ -x has the solution set x ≤ -3 or -1 ≤ x ≤ 3:
theorem problem_1 {x : ℝ} : f x 1 ≤ -x ↔ x ≤ -3 ∨ (-1 ≤ x ∧ x ≤ 3) := 
by sorry

-- Prove the range of values for a such that f(x, a) ≤ a^2 + 1 always holds true:
theorem problem_2 {a : ℝ} : (∀ x, f x a ≤ a^2 + 1) ↔ (a ≤ -Real.sqrt 2 ∨ Real.sqrt 2 ≤ a) := 
by sorry

end problem_1_problem_2_l535_535230


namespace divide_rectangle_into_five_l535_535517

theorem divide_rectangle_into_five (L W : ℝ) :
  ∃ (R1 R2 R3 R4 R5 : set (ℝ × ℝ)),
  (R1 ∪ R2 ∪ R3 ∪ R4 ∪ R5 = set.univ) ∧
  (∀ i j, i ≠ j → ¬∃ R, R ∪ (if i < j then R1 else R2) = R1 ∪ if i < j then R2 else R1) := 
sorry

end divide_rectangle_into_five_l535_535517


namespace average_of_consecutive_integers_l535_535537

theorem average_of_consecutive_integers (a b : ℤ) (h1 : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) :
  let b := a + 2 in
  let avg_of_b := ((a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 5 in
  avg_of_b = a + 4 :=
by
  sorry

end average_of_consecutive_integers_l535_535537


namespace ordered_pair_l535_535024

-- Definitions
def P (x : ℝ) := x^4 - 8 * x^3 + 20 * x^2 - 34 * x + 15
def D (k : ℝ) (x : ℝ) := x^2 - 3 * x + k
def R (a : ℝ) (x : ℝ) := x + a

-- Hypothesis
def condition (k a : ℝ) : Prop := ∀ x : ℝ, P x % D k x = R a x

-- Theorem
theorem ordered_pair (k a : ℝ) (h : condition k a) : (k, a) = (5, 15) := 
  sorry

end ordered_pair_l535_535024


namespace average_age_of_family_l535_535055

theorem average_age_of_family :
  let num_grandparents := 2
  let num_parents := 2
  let num_grandchildren := 3
  let avg_age_grandparents := 64
  let avg_age_parents := 39
  let avg_age_grandchildren := 6
  let total_age_grandparents := avg_age_grandparents * num_grandparents
  let total_age_parents := avg_age_parents * num_parents
  let total_age_grandchildren := avg_age_grandchildren * num_grandchildren
  let total_age_family := total_age_grandparents + total_age_parents + total_age_grandchildren
  let num_family_members := num_grandparents + num_parents + num_grandchildren
  let avg_age_family := total_age_family / num_family_members
  avg_age_family = 32 := 
  by 
  repeat { sorry }

end average_age_of_family_l535_535055


namespace tangent_circles_distance_l535_535930

theorem tangent_circles_distance :
  ∀ {O₁ O₂ : Type*} [MetricSpace O₁] [MetricSpace O₂] (dist : O₁ → O₂ → ℝ) (radius_O₁ : ℝ) (radius_O₂ : ℝ),
  (radius_O₁ = 3) → (radius_O₂ = 2) → (dist O₁ O₂ = radius_O₁ + radius_O₂ ∨ dist O₁ O₂ = |radius_O₁ - radius_O₂|) →
  dist O₁ O₂ = 5 ∨ dist O₁ O₂ = 1 :=
begin
  intros O₁ O₂ _ _ dist radius_O₁ radius_O₂ h₁ h₂ h,
  rw [h₁, h₂], -- Replacing radii with 3 and 2 respectively
  exact h,
end

end tangent_circles_distance_l535_535930


namespace remainder_of_98_pow_50_mod_50_l535_535423

theorem remainder_of_98_pow_50_mod_50 : (98 ^ 50) % 50 = 0 := by
  sorry

end remainder_of_98_pow_50_mod_50_l535_535423


namespace not_right_triangle_l535_535290

theorem not_right_triangle (A B C : ℝ) (hA : A + B = 180 - C) 
  (hB : A = B / 2 ∧ A = C / 3) 
  (hC : A = B / 2 ∧ B = C / 1.5) 
  (hD : A = 2 * B ∧ A = 3 * C):
  (C ≠ 90) :=
by {
  sorry
}

end not_right_triangle_l535_535290


namespace neg_of_proposition_l535_535588

variable (a : ℝ)

def proposition := ∀ x : ℝ, 0 < a^x

theorem neg_of_proposition (h₀ : 0 < a) (h₁ : a ≠ 1) : ¬proposition a ↔ ∃ x : ℝ, a^x ≤ 0 :=
by
  sorry

end neg_of_proposition_l535_535588


namespace sale_in_third_month_l535_535790

def grocer_sales (s1 s2 s4 s5 s6 : ℕ) (average : ℕ) (num_months : ℕ) (total_sales : ℕ) : Prop :=
  s1 = 5266 ∧ s2 = 5768 ∧ s4 = 5678 ∧ s5 = 6029 ∧ s6 = 4937 ∧ average = 5600 ∧ num_months = 6 ∧ total_sales = average * num_months

theorem sale_in_third_month
  (s1 s2 s4 s5 s6 total_sales : ℕ)
  (h : grocer_sales s1 s2 s4 s5 s6 5600 6 total_sales) :
  ∃ s3 : ℕ, total_sales - (s1 + s2 + s4 + s5 + s6) = s3 ∧ s3 = 5922 := 
by {
  sorry
}

end sale_in_third_month_l535_535790


namespace sum_of_elements_l535_535387

theorem sum_of_elements (S : Finset ℤ) (h_card : S.card = 4) 
    (h_sum_subsets : ∑ s in S.powerset, s.sum id = 2008) : S.sum id = 251 := 
by
  sorry

end sum_of_elements_l535_535387


namespace triangle_nonexistence_l535_535716

theorem triangle_nonexistence (a b : ℝ) (A : ℝ) (h₁ : a = 4) (h₂ : b = 5 * Real.sqrt 2) (h₃ : A = Real.pi / 4) :
  ∃ (n : ℕ), n = 0 ∧ ∀ (C : Triangle), ¬ (C.a = a ∧ C.b = b ∧ C.A = A) :=
by
  sorry

end triangle_nonexistence_l535_535716


namespace find_a7_l535_535226

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (n : ℕ)

-- Condition 1: The sequence {a_n} is geometric with all positive terms.
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

-- Condition 2: a₄ * a₁₀ = 16
axiom geo_seq_condition : is_geometric_sequence a r ∧ a 4 * a 10 = 16

-- The goal to prove
theorem find_a7 : (is_geometric_sequence a r ∧ a 4 * a 10 = 16) → a 7 = 4 :=
by {
  sorry
}

end find_a7_l535_535226


namespace find_d_div_a_l535_535717
noncomputable def quad_to_square_form (x : ℝ) : ℝ :=
  x^2 + 1500 * x + 1800

theorem find_d_div_a : 
  ∃ (a d : ℝ), (∀ x : ℝ, quad_to_square_form x = (x + a)^2 + d) 
  ∧ a = 750 
  ∧ d = -560700 
  ∧ d / a = -560700 / 750 := 
sorry

end find_d_div_a_l535_535717


namespace simplify_f_find_f_given_cos_l535_535213

-- Define the conditions
def is_third_quadrant (α : Real) : Prop :=
  π < α ∧ α < (3 * π) / 2

def f (α : Real) : Real :=
  (sin (α - π / 2) * cos ((3 / 2) * π + α) * tan (π - α)) / 
  (tan (-α - π) * sin (-α - π))

-- Prove the two results
theorem simplify_f (α : Real) (h : is_third_quadrant α) : 
  f α = -cos α := 
by
  sorry

theorem find_f_given_cos (α : Real) (h1 : is_third_quadrant α) 
  (h2 : cos (α - 3 / 2 * π) = 3 / 5) : 
  f α = 3 / 5 := 
by
  sorry

end simplify_f_find_f_given_cos_l535_535213


namespace log_div_inequality_l535_535098

theorem log_div_inequality :
  (log 1 - log 25) / 100 = -20 :=
sorry

end log_div_inequality_l535_535098


namespace angle_between_vectors_is_90_l535_535853

def vector_u : ℝ × ℝ := (3, -4)
def vector_v : ℝ × ℝ := (4, 3)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

theorem angle_between_vectors_is_90 :
  let θ := Real.arccos (dot_product vector_u vector_v / (magnitude vector_u * magnitude vector_v))
  θ = Real.pi / 2 :=
by
  sorry

end angle_between_vectors_is_90_l535_535853


namespace trailing_zeroes_53_plus_54_fact_l535_535601

-- Definition to count trailing zeroes of a factorial
def count_trailing_zeroes (n : ℕ) : ℕ :=
  (nat.div n 5) + (nat.div n 25) + (nat.div n 125) -- function could be extended for higher powers if necessary

-- Proof statement
theorem trailing_zeroes_53_plus_54_fact :
  count_trailing_zeroes 53 + count_trailing_zeroes 54 = 12 := sorry

end trailing_zeroes_53_plus_54_fact_l535_535601


namespace fill_cistern_time_l535_535457

-- Define the rates of the taps
def rateA := (1 : ℚ) / 3  -- Tap A fills 1 cistern in 3 hours (rate is 1/3 per hour)
def rateB := -(1 : ℚ) / 6  -- Tap B empties 1 cistern in 6 hours (rate is -1/6 per hour)
def rateC := (1 : ℚ) / 2  -- Tap C fills 1 cistern in 2 hours (rate is 1/2 per hour)

-- Define the combined rate
def combinedRate := rateA + rateB + rateC

-- The time to fill the cistern when all taps are opened simultaneously
def timeToFill := 1 / combinedRate

-- The theorem stating that the time to fill the cistern is 1.5 hours
theorem fill_cistern_time : timeToFill = (3 : ℚ) / 2 := by
  sorry  -- The proof is omitted as per the instructions

end fill_cistern_time_l535_535457


namespace sum_of_squares_of_roots_of_quadratic_l535_535378

noncomputable def sum_of_squares_of_roots (a b c : ℚ) (ha : a ≠ 0) : ℚ :=
  let x1_plus_x2 := -b / a
  let x1_times_x2 := c / a
  (x1_plus_x2 ^ 2) - 2 * x1_times_x2

theorem sum_of_squares_of_roots_of_quadratic :
  sum_of_squares_of_roots 10 15 (-20) 10 ≠ 0 = 25 / 4 :=
by
  unfold sum_of_squares_of_roots
  calc
  (-15 / 10) ^ 2 - 2 * (-20 / 10) = (-(3 / 2) ^ 2) : by sorry
    ... = ((-(3 / 2)) ^ 2 : by sorry
    ... = (9 / 4) : by sorry
    ... = (-2 * -2) : by sorry
    ... = (4)      : by sorry
    ... = (9 / 4 + 4) : by sorry
    ... = (25 / 4) : by sorry //-- result

end sum_of_squares_of_roots_of_quadratic_l535_535378


namespace parallel_lines_condition_iff_l535_535231

def line_parallel (a : ℝ) : Prop :=
  let l1_slope := -1 / -a
  let l2_slope := -(a - 1) / -12
  l1_slope = l2_slope

theorem parallel_lines_condition_iff (a : ℝ) :
  (a = 4) ↔ line_parallel a := by
  sorry

end parallel_lines_condition_iff_l535_535231


namespace stratified_sampling_group_a_count_l535_535067

theorem stratified_sampling_group_a_count
  (total_cities : ℕ)
  (group_a_count : ℕ)
  (group_b_count : ℕ)
  (group_c_count : ℕ)
  (selected_count : ℕ)
  (h_total : total_cities = 24)
  (h_group_a : group_a_count = 4)
  (h_group_b : group_b_count = 12)
  (h_group_c : group_c_count = 8)
  (h_selected : selected_count = 6) :
  (group_a_count * selected_count) / total_cities = 1 := 
by { rw [h_total, h_group_a, h_selected], norm_num }

end stratified_sampling_group_a_count_l535_535067


namespace proof_problem_l535_535658

/-- Let {a_n} be an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ d : ℝ, a (n+1) - a n = d ∧ a (m+1) - a m = d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(1 + n) * (a 1 + a n) / 2

theorem proof_problem (a : ℕ → ℝ) 
  (h_seq : is_arithmetic_sequence a)
  (h_pos : a 1 > 0)
  (h_cond : a 1 + a 9 = a 4) :
  a 6 = 0 ∧
  sum_arithmetic_sequence a 11 = 0 ∧
  sum_arithmetic_sequence a 5 = sum_arithmetic_sequence a 6 ∧
  ∀ n, n ≠ 7 → sum_arithmetic_sequence a 7 ≤ sum_arithmetic_sequence a n := 
by 
  sorry

end proof_problem_l535_535658


namespace final_books_is_correct_l535_535408

def initial_books : ℝ := 35.5
def books_bought : ℝ := 12.3
def books_given_to_friends : ℝ := 7.2
def books_donated : ℝ := 20.8

theorem final_books_is_correct :
  (initial_books + books_bought - books_given_to_friends - books_donated) = 19.8 := by
  sorry

end final_books_is_correct_l535_535408


namespace num_int_values_n_terminated_l535_535155

theorem num_int_values_n_terminated (N : ℕ) (hN1 : 1 ≤ N) (hN2 : N ≤ 500) :
  ∃ n : ℕ, n = 10 ∧ ∀ k, 0 ≤ k → k < n → ∃ (m : ℕ), N = m * 49 :=
sorry

end num_int_values_n_terminated_l535_535155


namespace number_of_solutions_l535_535600

theorem number_of_solutions :
  let count := (Set.toFinset { p : ℤ × ℤ | let a := p.1, b := p.2 in a^2 + b^2 < 25 ∧ a^2 + b^2 < 10*a ∧ a^2 + b^2 < 10*b }).card
  in count = 9 := by
  sorry

end number_of_solutions_l535_535600


namespace max_m_value_min_value_expression_l535_535347

-- Define the conditions for the inequality where the solution is the entire real line
theorem max_m_value (x m : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
sorry

-- Define the conditions for a, b, c > 0 and their sum equal to 1
-- and prove the minimum value of 4a^2 + 9b^2 + c^2
theorem min_value_expression (a b c : ℝ) (hpos1 : a > 0) (hpos2 : b > 0) (hpos3 : c > 0) (hsum : a + b + c = 1) :
  4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧ (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 → a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
sorry

end max_m_value_min_value_expression_l535_535347


namespace M_necessary_for_N_l535_535305

def M (x : ℝ) : Prop := -1 < x ∧ x < 3
def N (x : ℝ) : Prop := 0 < x ∧ x < 3

theorem M_necessary_for_N : (∀ a : ℝ, N a → M a) ∧ (∃ b : ℝ, M b ∧ ¬N b) :=
by sorry

end M_necessary_for_N_l535_535305


namespace calculate_difference_of_squares_l535_535831

theorem calculate_difference_of_squares : 403^2 - 397^2 = 4800 := by
  have h : ∀ a b : ℕ, a^2 - b^2 = (a + b) * (a - b) := by
    intro a b
    exact (nat.pow_two_sub_pow_two a b).symm
  rw h
  norm_num
  exact rfl

end calculate_difference_of_squares_l535_535831


namespace sum_solutions_eq_32_l535_535301

theorem sum_solutions_eq_32 :
  ∃ (n : ℕ) (xs ys : fin n → ℤ), 
    (∀ i, |(xs i) - 2| = 3 * |(ys i) - 8| ∧ |(xs i) - 8| = |(ys i) - 2| ) ∧ 
    ((finset.univ : finset (fin n)).sum (λ i, xs i + ys i) = 32) :=
sorry

end sum_solutions_eq_32_l535_535301


namespace total_crackers_l535_535502

-- Define the conditions
def boxes_Darren := 4
def crackers_per_box := 24
def boxes_Calvin := 2 * boxes_Darren - 1

-- Define the mathematical proof problem
theorem total_crackers : 
  let total_Darren := boxes_Darren * crackers_per_box
  let total_Calvin := boxes_Calvin * crackers_per_box
  total_Darren + total_Calvin = 264 :=
by
  sorry

end total_crackers_l535_535502


namespace no_polynomials_exist_l535_535114

theorem no_polynomials_exist (P Q : ℝ → ℝ → ℝ) :
  (∀ x y : ℝ, (x + y) * P x y + (2 * x - y - 3) * Q x y = x^2 + y^2) → (∃ x y : ℝ, (x + y) * P x y + (2 * x - y - 3) * Q x y ≠ x^2 + y^2) :=
begin
  sorry
end

end no_polynomials_exist_l535_535114


namespace sum_powers_of_i_l535_535101

noncomputable def sum_of_powers_of_i : ℂ :=
  let i := complex.I in
  (finset.range 604).sum (λ n, i ^ n)

theorem sum_powers_of_i : sum_of_powers_of_i = 0 := by
  sorry

end sum_powers_of_i_l535_535101


namespace value_of_expression_l535_535873

variable (x1 x2 : ℝ)

def sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 3
def product_roots (x1 x2 : ℝ) : Prop := x1 * x2 = -4

theorem value_of_expression (h1 : sum_roots x1 x2) (h2 : product_roots x1 x2) : 
  x1^2 - 4*x1 - x2 + 2*x1*x2 = -7 :=
by sorry

end value_of_expression_l535_535873


namespace average_of_consecutive_sequences_l535_535534

theorem average_of_consecutive_sequences (a b : ℕ) (h : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) : 
    ((b + (b+1) + (b+2) + (b+3) + (b+4)) / 5) = a + 4 :=
by
  sorry

end average_of_consecutive_sequences_l535_535534


namespace third_largest_number_l535_535345

theorem third_largest_number (a b c : ℕ) (h1 : a = 1) (h2 : b = 6) (h3 : c = 8) : 
  third_largest (list.map (λ ds, ds.2.head * 100 + ds.2.tail.head! * 10 + ds.2.tail.tail.head!) ((list.permutations [1,6,8]).map (λ l, (l, l)))) = 681 :=
by {sorry}

end third_largest_number_l535_535345


namespace perimeter_DEF_leq_half_perimeter_ABC_l535_535087

theorem perimeter_DEF_leq_half_perimeter_ABC
  (A B C D E F H : Type*)
  [IsAcuteTriangle A B C]
  [IsOrthocenter H A B C]
  [IsFeetOfAltitudes D E F A B C] :
  (perimeter DEF <= (perimeter ABC) / 2) :=
sorry

end perimeter_DEF_leq_half_perimeter_ABC_l535_535087


namespace perpendicular_condition_l535_535253

def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + 2 * y - 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := 3 * x - a * y + 1

def perpendicular_lines (a : ℝ) : Prop := 
  ∀ (x y : ℝ), line1 a x y = 0 → line2 a x y = 0 → 3 * a - 2 * a = 0 

theorem perpendicular_condition (a : ℝ) (h : perpendicular_lines a) : a = 0 := sorry

end perpendicular_condition_l535_535253


namespace min_bottles_l535_535018

theorem min_bottles (a b : ℕ) (h1 : a > b) (h2 : b > 1) : 
  ∃ x : ℕ, x = Nat.ceil (a - a / b) := sorry

end min_bottles_l535_535018


namespace probability_1_lt_X_le_3_l535_535906

def P (X : ℕ → ℝ) (k : ℕ) : ℝ := if h : k ∈ {1, 2, 3, 4} then k / 10 else 0

theorem probability_1_lt_X_le_3 (X : ℕ → ℝ) (h : ∀ k, k ∈ {1, 2, 3, 4} → X k = k / 10) :
  (X 2 + X 3) = 1 / 2 :=
by 
  sorry

end probability_1_lt_X_le_3_l535_535906


namespace solution_set_f_gt_linearity_l535_535572

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_deriv : ℝ → ℝ := sorry

-- Conditions
axiom domain_f : ∀ x, x ∈ (ℝ : Type)
axiom f_at_minus_one : f (-1) = 2
axiom f_deriv_gt_two : ∀ x, f_deriv x > 2

-- Proof Problem
theorem solution_set_f_gt_linearity : ∀ x, x > -1 → f(x) > 2 * x + 4 := 
sorry

end solution_set_f_gt_linearity_l535_535572


namespace coefficient_of_x3_l535_535019

def poly1 : Polynomial ℤ := Polynomial.of_finsupp $ finsupp.of_list [(5, 1), (3, -4), (2, 3), (1, -2), (0, 5)]
def poly2 : Polynomial ℤ := Polynomial.of_finsupp $ finsupp.of_list [(2, 3), (1, -2), (0, 4)]
def poly3 : Polynomial ℤ := Polynomial.of_finsupp $ finsupp.of_list [(1, -1), (0, 1)]

theorem coefficient_of_x3 :
  (poly1 * poly2 * poly3).coeff 3 = -3 := by
  sorry

end coefficient_of_x3_l535_535019


namespace least_number_condition_l535_535419

theorem least_number_condition (n : ℕ) (k : ℕ) :
  ((∀ p ∈ {2, 3, 4, 5, 6, 7}, n % p = 2 % p) ∧ (n % 9 = 0)) ↔ n = 2102 := by
  sorry

end least_number_condition_l535_535419


namespace alternating_sum_12000_terms_l535_535021

theorem alternating_sum_12000_terms : (∑ i in Finset.range 12000, (-1)^(i+1) * (i + 1)) = 6000 := by
  sorry

end alternating_sum_12000_terms_l535_535021


namespace polynomial_sum_of_squares_l535_535336

theorem polynomial_sum_of_squares
  (P : Polynomial ℝ)
  (h1 : ∀ x : ℝ, 0 ≤ P.eval x) :
  ∃ (Q : ℕ → Polynomial ℝ) (n : ℕ), P = (Finset.range n).sum (λ i, (Q i)^2) :=
begin
  sorry
end

end polynomial_sum_of_squares_l535_535336


namespace tom_jerry_age_ratio_l535_535737

-- Definitions representing the conditions in the problem
variable (t j x : ℕ)

-- Condition 1: Three years ago, Tom was three times as old as Jerry
def condition1 : Prop := t - 3 = 3 * (j - 3)

-- Condition 2: Four years before that, Tom was five times as old as Jerry
def condition2 : Prop := t - 7 = 5 * (j - 7)

-- Question: In how many years will the ratio of their ages be 3:2,
-- asserting that the answer is 21
def ageRatioInYears : Prop := (t + x) / (j + x) = 3 / 2 → x = 21

-- The proposition we need to prove
theorem tom_jerry_age_ratio (h1 : condition1 t j) (h2 : condition2 t j) : ageRatioInYears t j x := 
  sorry
  
end tom_jerry_age_ratio_l535_535737


namespace digit_removal_ways_l535_535715

theorem digit_removal_ways :
  let digits := {1, 1, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8}
  let mandatory_removals := {6, 7, 8}
  let remaining_after_mandatory_removals := {1, 1, 1, 1, 2, 3, 4, 4, 5}
  ∃ (ways : Nat), ways = 60 :=
by
  let digits := {1, 1, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8}
  let mandatory_removals := {6, 7, 8}
  let remaining_after_mandatory_removals := {1, 1, 1, 1, 2, 3, 4, 4, 5}
  have ways : Nat := 5 * 3 * 2 * 2
  exact ⟨ways, rfl⟩

end digit_removal_ways_l535_535715


namespace kw_percent_combined_assets_l535_535493

def assets_price_relation (Ax Ay : ℝ) : Prop :=
  1.60 * Ax = 2 * Ay

def debt (Ax : ℝ) : ℝ :=
  0.20 * Ax

def stocks (Ay : ℝ) : ℝ :=
  0.30 * Ay

def combined_assets (Ax Ay : ℝ) : ℝ :=
  Ax + Ay - debt Ax + stocks Ay

def KW_price (KW Ay : ℝ) : ℝ :=
  2 * Ay

def combined_assets_percent (Ax Ay : ℝ) (KW : ℝ) : ℝ :=
  KW / combined_assets Ax Ay * 100

theorem kw_percent_combined_assets (Ax Ay : ℝ) (KW : ℝ)
  (h1 : assets_price_relation Ax Ay) 
  (h2 : KW = KW_price KW Ay) :
  abs (combined_assets_percent Ax Ay  KW - 86.96) < 0.01 :=
by
  sorry

end kw_percent_combined_assets_l535_535493


namespace symmetric_point_over_y_axis_l535_535286

def reflect_over_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2, -p.3)

theorem symmetric_point_over_y_axis :
  reflect_over_y_axis (2, -3, 5) = (-2, -3, -5) :=
by
  sorry

end symmetric_point_over_y_axis_l535_535286


namespace f_positive_l535_535532

-- Definition of the function f(x)
def f (x : ℝ) : ℝ :=
  2 * x - (3 * x / 2) * Real.log (Real.exp 1 - 1 / (3 * x))

-- Theorem statement for the given problem
theorem f_positive (x : ℝ) (h : x > 1 / (3 * Real.exp 1)) :
  f x > 0 :=
sorry

end f_positive_l535_535532


namespace complex_quadrant_l535_535361

def complex_z := ((-1 : ℝ) + (1: ℝ) * complex.I) / (1 + (1: ℝ) * complex.I - 1 : ℂ)

theorem complex_quadrant :
  complex.re complex_z < 0 ∧ complex.im complex_z > 0 :=
by
  sorry

end complex_quadrant_l535_535361


namespace parametric_line_slope_l535_535924

open Real

def param_line (t : ℝ) : ℝ × ℝ :=
  (3 + 4 * t, 4 - 5 * t)

theorem parametric_line_slope :
  ∃ m : ℝ, ∀ t : ℝ, m = -5/4 :=
begin
  use -5/4,
  intro t,
  sorry

end parametric_line_slope_l535_535924


namespace range_of_list_l535_535771

   def list := [7, 13, 4, 9, 6]
   def range (l : List ℕ) : ℕ := l.maximum - l.minimum

   theorem range_of_list : range list = 9 := by
     sorry
   
end range_of_list_l535_535771


namespace distance_squared_sum_eq_constant_l535_535654

open Real

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C Q : V)

def G' := (1 / 4 : ℝ) • A + (1 / 4 : ℝ) • B + (1 / 2 : ℝ) • C

theorem distance_squared_sum_eq_constant {k' : ℝ} :
  dist Q A ^ 2 + dist Q B ^ 2 + dist Q C ^ 2 = k' * dist Q G' ^ 2 + dist G' A ^ 2 + dist G' B ^ 2 + dist G' C ^ 2 →
  k' = 4 :=
by sorry

end distance_squared_sum_eq_constant_l535_535654


namespace correct_statements_l535_535431

-- Define the individual statements
def statement_1 (x : ℝ) : Prop := irrational x → (∀ n : ℕ, x ≠ (n : ℚ)) ∧ ∃ (s : ℕ → ℤ), ∀ n, (x ≠ s n / 10^n)
def statement_2 : Prop := ∀ x : ℝ, ∃ p : ℝ, x = p
def statement_3 : Prop := (∃ x y z w : ℝ, irrational x ∧ irrational y ∧ irrational z ∧ irrational w ∧ 1 < x ∧ x < 3 ∧ 1 < y ∧ y < 3 ∧ 1 < z ∧ z < 3 ∧ 1 < w ∧ w < 3 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ w ∧ w ≠ x ∧ x ≠ z ∧ y ≠ w) ∧ (¬ ∃ a b c d e f g h : ℝ, irrational a ∧ irrational b ∧ irrational c ∧ irrational d ∧ irrational e ∧ irrational f ∧ irrational g ∧ irrational h ∧ 1 < a ∧ a < 3 ∧ 1 < b ∧ b < 3 ∧ 1 < c ∧ c < 3 ∧ 1 < d ∧ d < 3 ∧ 1 < e ∧ e < 3 ∧ 1 < f ∧ f < 3 ∧ 1 < g ∧ g < 3 ∧ 1 < h ∧ h < 3)
def statement_4 : Prop := ∀ x, (1.495 < x ∧ x < 1.505) → ¬ (x = 1.50)
def statement_5 (a b : ℝ) : Prop := (a ≠ b ∧ a = -b) → (a / b = -1)

-- The main proof statement
theorem correct_statements : statement_1 ∧ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_4 ∧ ¬ statement_5 :=
by sorry

end correct_statements_l535_535431


namespace tan_nine_pi_over_four_l535_535125

theorem tan_nine_pi_over_four : 
  let pi := Real.pi in
  let tan := Real.tan in
  ∀ θ : ℝ, tan θ = tan (θ + 2 * pi) →
  (tan (45 * (π / 180)) = 1) →
  (∃ θ : ℝ, θ = 9 * (π / 4) ∧ tan θ = 1) := by
  intros h_period h_fundamental
  exists 9 * (π / 4)
  split
  · rfl
  · sorry

end tan_nine_pi_over_four_l535_535125


namespace sum_sequence_l535_535881

noncomputable def a : ℕ → ℚ
| 0       := 1
| (n + 1) := 3 * (a n) + 1

noncomputable def S_n (n : ℕ) : ℚ :=
  (1/4) * (3^(n + 1) - 2 * (n + 1) - 3)

theorem sum_sequence (n : ℕ) : 
  (∑ i in Finset.range (n + 1), a i) = S_n n :=
sorry

end sum_sequence_l535_535881


namespace barbara_initial_candies_l535_535096

noncomputable def initialCandies (used left: ℝ) := used + left

theorem barbara_initial_candies (used left: ℝ) (h_used: used = 9.0) (h_left: left = 9) : initialCandies used left = 18 := 
by
  rw [h_used, h_left]
  norm_num
  sorry

end barbara_initial_candies_l535_535096


namespace tabitha_current_age_l535_535121

noncomputable def tabithaAge (currentColors : ℕ) (yearsPassed : ℕ) (startAge : ℕ) (futureYears : ℕ) (futureColors : ℕ) : Prop :=
  (currentColors = (futureColors - futureYears)) ∧
  (yearsPassed = (currentColors - 2)) ∧
  (yearsPassed + startAge = 18)

theorem tabitha_current_age : tabithaAge 5 3 15 3 8 := 
by
  unfold tabithaAge
  split
  all_goals {simp}
  sorry

end tabitha_current_age_l535_535121


namespace find_perimeter_ABCD_l535_535341

open Real

def RhombusInscribedInRectangle (ABCD : Type) :=
  ∃ (P Q R S : Point) (PB BQ PR QS : ℝ),
  (PB = 15) ∧ (BQ = 20) ∧ (PR = 30) ∧ (QS = 40) ∧
  inscribed_rhombus PQRS ABCD ABC PQRS PQ 30 40 ABCD contains_rect ABCD.

theorem find_perimeter_ABCD :
  ∃ m n : ℕ, coprime m n ∧ m + n = 677 :=
by
  sorry

end find_perimeter_ABCD_l535_535341


namespace gcd_five_ten_div_four_l535_535417

def five_factorial := (5! : ℕ)
def ten_div_four_factorial := (10! / 4! : ℕ)

theorem gcd_five_ten_div_four :
  Nat.gcd five_factorial ten_div_four_factorial = 120 := by
  sorry

end gcd_five_ten_div_four_l535_535417


namespace probability_of_BEE_l535_535395

theorem probability_of_BEE : 
  let cards := ["E", "E", "B"]
  and arrangements := [("B", "E", "E"), ("E", "B", "E"), ("E", "E", "B")]
  in
  (countp (λ s => s = ("B", "E", "E")) arrangements : ℚ) / (arrangements.length : ℚ) = 1 / 3 :=
by
  sorry

end probability_of_BEE_l535_535395


namespace margarita_run_distance_l535_535637

theorem margarita_run_distance :
  ∀ (Ricciana_run Ricciana_jump Margarita_jump Margarita_total_distance : ℕ),
    Ricciana_run = 20 →
    Ricciana_jump = 4 →
    Margarita_jump = 2 * Ricciana_jump - 1 →
    Margarita_total_distance = Ricciana_run + Ricciana_jump + 1 →
    Margarita_run = Margarita_total_distance - Margarita_jump →
    Margarita_run = 18 :=
by
  intros Ricciana_run Ricciana_jump Margarita_jump Margarita_total_distance
  rintros ⟨h1, h2, h3, h4⟩
  sorry

end margarita_run_distance_l535_535637


namespace tom_finishes_in_6_years_l535_535740

variable (BS_time : ℕ) (PhD_time : ℕ) (fraction : ℚ)

def total_program_time := BS_time + PhD_time

noncomputable def tom_completion_time := total_program_time * fraction

theorem tom_finishes_in_6_years :
  BS_time = 3 ∧ PhD_time = 5 ∧ fraction = (3/4 : ℚ) → tom_completion_time BS_time PhD_time fraction = 6 := by
  sorry

end tom_finishes_in_6_years_l535_535740


namespace intersecting_diagonals_probability_l535_535772

def probability_of_intersecting_diagonals_inside_dodecagon : ℚ :=
  let total_points := 12
  let total_segments := (total_points.choose 2)
  let sides := 12
  let diagonals := total_segments - sides
  let ways_to_choose_2_diagonals := (diagonals.choose 2)
  let ways_to_choose_4_points := (total_points.choose 4)
  let probability := (ways_to_choose_4_points : ℚ) / (ways_to_choose_2_diagonals : ℚ)
  probability

theorem intersecting_diagonals_probability (H : probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477) : 
  probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477 :=
  by
  sorry

end intersecting_diagonals_probability_l535_535772


namespace probability_neither_prime_nor_composite_lemma_l535_535960

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def neither_prime_nor_composite (n : ℕ) : Prop :=
  ¬ is_prime n ∧ ¬ is_composite n

def probability_of_neither_prime_nor_composite (n : ℕ) : ℚ :=
  if 1 ≤ n ∧ n ≤ 97 then 1 / 97 else 0

theorem probability_neither_prime_nor_composite_lemma :
  probability_of_neither_prime_nor_composite 1 = 1 / 97 := by
  sorry

end probability_neither_prime_nor_composite_lemma_l535_535960


namespace cost_of_12_roll_package_is_correct_l535_535784

variable (cost_per_roll_package : ℝ)
variable (individual_cost_per_roll : ℝ := 1)
variable (number_of_rolls : ℕ := 12)
variable (percent_savings : ℝ := 0.25)

-- The definition of the total cost of the package
def total_cost_package := number_of_rolls * (individual_cost_per_roll - (percent_savings * individual_cost_per_roll))

-- The goal is to prove that the total cost of the package is $9
theorem cost_of_12_roll_package_is_correct : total_cost_package = 9 := 
by
  sorry

end cost_of_12_roll_package_is_correct_l535_535784


namespace count_terminating_decimals_l535_535142

theorem count_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 500) : 
  (nat.floor (500 / 49) = 10) := 
by
  sorry

end count_terminating_decimals_l535_535142


namespace xy_div_eq_one_third_l535_535578

theorem xy_div_eq_one_third (x y z : ℝ) 
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : x + y + z = 21)
  (h4 : y / z = 6) : 
  x / y = 1 / 3 :=
by
  sorry

end xy_div_eq_one_third_l535_535578


namespace object_horizontal_speed_l535_535820

theorem object_horizontal_speed
(h_initial_height : ∃ h : ℝ, h = 100)
(h_downward_acceleration : ∃ a : ℝ, a = 32)
(h_horizontal_distance : ∃ d : ℝ, d = 200)
(h_time : ∃ t : ℝ, t = 2)
(h_conversion : ∃ m : ℝ, m = 5280) :
  ∃ s : ℝ, s ≈ 68.18 :=
by
  sorry

end object_horizontal_speed_l535_535820


namespace smallest_positive_z_l535_535354

theorem smallest_positive_z (x z : ℝ) (k : ℤ) 
  (cos_x_zero : cos x = 0) 
  (sin_xz_sqrt_3_2 : sin (x + z) = sqrt 3 / 2) : 
  ∃ z, (z > 0) ∧ (z = π / 6) :=
by 
  have h1 : x = π / 2 ∨ x = 3 * π / 2, from sorry,
  have h2 : (x + z = π / 3 + 2 * π * k) ∨ (x + z = 2 * π / 3 + 2 * π * k), from sorry,
  sorry

end smallest_positive_z_l535_535354


namespace find_angle_C_find_perimeter_l535_535270

-- Definitions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Opposite sides to angles A, B, C respectively
variables (acute : A < π/2 ∧ B < π/2 ∧ C < π/2) -- Acute triangle condition
variables (eq1 : sqrt 3 * a = 2 * c * sin A) -- Given condition
variables (c_val : c = sqrt 7) (ab_val: a * b = 6) -- Given values for Part 2

-- Proof statements
theorem find_angle_C (h : acute) (h1 : eq1) : C = π / 3 :=
sorry

theorem find_perimeter (h : acute) (h1 : eq1) (h2 : c_val) (h3 : ab_val) : a + b + c = 5 + sqrt 7 :=
sorry

end find_angle_C_find_perimeter_l535_535270


namespace rectangular_equation_of_curve_range_of_x_minus_2y_l535_535224

-- Define the polar equation and conditions of curve C
def polar_equation (ρ θ : ℝ) : Prop := 7 * ρ^2 - ρ^2 * real.cos (2 * θ) - 24 = 0

-- Define the rectangular equation of the curve C from polar to rectangular coordinates
theorem rectangular_equation_of_curve (x y : ℝ) (h : ∃ ρ θ, ρ = real.sqrt (x^2 + y^2) ∧ θ = real.atan2 y x ∧ polar_equation ρ θ) :
  x^2 / 4 + y^2 / 3 = 1 :=
sorry

-- Define the point (x, y) on curve C and find the range of x - 2y
theorem range_of_x_minus_2y (x y : ℝ) (h : ∃ ρ θ, ρ = real.sqrt (x^2 + y^2) ∧ θ = real.atan2 y x ∧ polar_equation ρ θ) :
  -4 ≤ x - 2 * y ∧ x - 2 * y ≤ 4 :=
sorry

end rectangular_equation_of_curve_range_of_x_minus_2y_l535_535224


namespace inequality_system_solution_l535_535700

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 1 > 2 * (x + 1) ∧ (x + 2) / 3 > x - 2) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end inequality_system_solution_l535_535700


namespace largest_possible_median_l535_535418

theorem largest_possible_median (x : ℕ) (hx : x > 0) : 
  let s := [x, 2 * x, 4 * x, 3, 2, 5].sort in s[2] = 5 ∨ s[3] = 5 :=
by
  sorry

end largest_possible_median_l535_535418


namespace right_triangle_hypotenuse_l535_535373

-- Define the context and conditions of the problem
variable (a b : ℝ)
axiom median_condition_1 : b^2 + (a / 2)^2 = 36
axiom median_condition_2 : a^2 + (b / 2)^2 = 36

-- Define and prove the theorem
theorem right_triangle_hypotenuse : (∃ (a b : ℝ), (b^2 + (a / 2)^2 = 36) ∧ (a^2 + (b / 2)^2 = 36) ∧ (2 * real.sqrt (a^2 + b^2) = 2 * real.sqrt 57.6)) :=
by
  use a,
  use b,
  exact ⟨median_condition_1, median_condition_2, sorry⟩

end right_triangle_hypotenuse_l535_535373


namespace angle_between_vectors_l535_535212

section
variables (a b : ℝ) (vec_a vec_b : EuclideanSpace ℝ (Fin 3))
variables (h1 : ∥vec_a∥ = 1) (h2 : ∥vec_b∥ = 2) (h3 : ∀ i, vec_a i + vec_b i = 0) (h4 : vec_a ⬝ vec_b = 0)

theorem angle_between_vectors : ∃ θ : ℝ, θ = 120 :=
by
  sorry
end

end angle_between_vectors_l535_535212


namespace petya_wins_n_ne_4_l535_535826

theorem petya_wins_n_ne_4 (n : ℕ) (h : n ≥ 3) :
  (∀ k, k ∈ {1, ..., n - 1} -> 
  (∃ i, (empty_glasses[i] ∧ empty_glasses[i+1])) ∨
  (∃ j, (full_glasses[j] ∧ full_glasses[j+1])) →
  (Vasya_moves (one empty one full)) →
  player_cannot_move_loses) ↔ (n ≠ 4) :=
sorry

end petya_wins_n_ne_4_l535_535826


namespace total_profit_calculation_l535_535296

theorem total_profit_calculation :
  ∀ (P : ℝ), 
    let john_investment := 18000 * 12,
    let rose_investment := 12000 * 9,
    let tom_investment := 9000 * 8,
    let total_investment := john_investment + rose_investment + tom_investment,
    let rose_share := (rose_investment / total_investment) * P,
    let tom_share := (tom_investment / total_investment) * P,
    rose_share - tom_share = 370 →
    P = 4070 :=
sorry

end total_profit_calculation_l535_535296


namespace curve_conversion_min_AB_distance_l535_535233

-- Definitions for Question 1
def polar_to_cartesian (ρ α : ℝ) : ℝ × ℝ := 
  (ρ * real.cos α, ρ * real.sin α)

def Curve_polar (ρ α : ℝ) : Prop := 
  ρ * (real.sin α)^2 = 2 * (ρ * real.cos α)

def Curve_cartesian (x y : ℝ) : Prop := 
  y^2 = 2 * x

theorem curve_conversion (ρ α : ℝ) (h : Curve_polar ρ α) : 
  Curve_cartesian (ρ * real.cos α) (ρ * real.sin α) :=
begin
  sorry  -- proof omitted
end

-- Definitions for Question 2
def Line_parametric (t θ : ℝ) : ℝ × ℝ :=
  (1/2 + t * real.cos θ, t * real.sin θ)

def Curve (x y : ℝ) : Prop :=
  y^2 = 2 * x

def intersection_points (θ : ℝ) (hθ : 0 < θ ∧ θ < real.pi) (A B : ℝ × ℝ) : Prop :=
  ∃ t1 t2 : ℝ, 
    A = Line_parametric t1 θ ∧ B = Line_parametric t2 θ ∧
    Curve (A.1) (A.2) ∧ Curve (B.1) (B.2)

theorem min_AB_distance (θ : ℝ) (hθ : 0 < θ ∧ θ < real.pi) (A B : ℝ × ℝ) (h : intersection_points θ hθ A B) :
  ∃ m : ℝ, (∀ θ' (hθ' : 0 < θ' ∧ θ' < real.pi) (A B : ℝ × ℝ), intersection_points θ' hθ' A B → |A.1 - B.1| = m) ∧
  m = 2 :=
by sorry

end curve_conversion_min_AB_distance_l535_535233


namespace sum_of_solutions_l535_535660

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 2

-- Define the inverse function of f
def f_inv (x : ℝ) : ℝ := (x - 2) / 3

-- Statement of the problem
theorem sum_of_solutions :
  ∑ x in {x | f_inv x = f (2 * x)}, x = -8 / 17 :=
by
  sorry

end sum_of_solutions_l535_535660


namespace count_not_squares_or_cubes_l535_535943

theorem count_not_squares_or_cubes (n : ℕ) : 
  let total := 200 in
  let perfect_squares := 14 in
  let perfect_cubes := 5 in
  let perfect_sixth_powers := 2 in
  let squares_or_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers in
  let count_not_squares_or_cubes := total - squares_or_cubes in
  n = count_not_squares_or_cubes :=
by
  let total := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let perfect_sixth_powers := 2
  let squares_or_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let count_not_squares_or_cubes := total - squares_or_cubes
  show _ from sorry

end count_not_squares_or_cubes_l535_535943


namespace ratio_of_dolls_l535_535515

-- Definitions used in Lean 4 statement directly appear in the conditions
variable (I : ℕ) -- the number of dolls Ivy has
variable (Dina_dolls : ℕ := 60) -- Dina has 60 dolls
variable (Ivy_collectors : ℕ := 20) -- Ivy has 20 collector edition dolls

-- Condition based on given problem
axiom Ivy_collectors_condition : (2 / 3 : ℚ) * I = 20

-- Lean 4 statement for the proof problem
theorem ratio_of_dolls (h : 3 * Ivy_collectors = 2 * I) : Dina_dolls / I = 2 := by
  sorry

end ratio_of_dolls_l535_535515


namespace monotonic_increasing_g_l535_535871

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x - sin x ^ 2

noncomputable def g (x : ℝ) : ℝ := f (x - π / 6) + 1 / 2

theorem monotonic_increasing_g : monotone_on g (set.Icc 0 (π / 3)) :=
sorry

end monotonic_increasing_g_l535_535871


namespace sum_difference_l535_535755

open Nat

-- Definitions of sequences for odd and even integers
def odd_seq (n : ℕ) : ℕ := 2 * n - 1
def even_seq (n : ℕ) : ℕ := 2 * n

-- Main theorem to prove the sum of differences
theorem sum_difference :
  (finset.range 50).sum (λ n, even_seq (n + 1)) - (finset.range 50).sum (λ n, odd_seq (n + 1)) = 50 :=
by
  sorry

end sum_difference_l535_535755


namespace linear_function_decreasing_y_l535_535709

theorem linear_function_decreasing_y (x1 y1 y2 : ℝ) :
  y1 = -2 * x1 - 7 → y2 = -2 * (x1 - 1) - 7 → y1 < y2 := by
  intros h1 h2
  sorry

end linear_function_decreasing_y_l535_535709


namespace complement_N_in_M_l535_535237

def M : Set ℤ := {-1, 0, 1}

def N : Set ℝ := {x | ∃ k : ℤ, x = Real.cos (k * Real.pi)}

theorem complement_N_in_M :
  (M \ N) = {0} :=
sorry

end complement_N_in_M_l535_535237


namespace angle_B_range_l535_535287

def range_of_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  (0 < B ∧ B ≤ Real.pi / 3)

theorem angle_B_range
  (a b c A B C : ℝ)
  (h1 : b^2 = a * c)
  (h2 : A + B + C = π)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : c > 0)
  (h6 : a + b > c)
  (h7 : a + c > b)
  (h8 : b + c > a) :
  range_of_angle_B a b c A B C :=
sorry

end angle_B_range_l535_535287


namespace triangle_problem_l535_535274

-- Given an acute triangle ABC with opposite sides a, b, and c,
-- and given √3a = 2c sin(A),
-- find C and the perimeter of the triangle given further conditions.
theorem triangle_problem
  (ABC : Type*)
  [triangle ABC]
  {a b c : ℝ}
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (acute : is_acute_triangle ABC)
  (opposite_sides : opposite_sides ABC a b c)
  (cond1 : √3 * a = 2 * c * sin (angle A))
  (cond2 : c = √7)
  (cond3 : a * b = 6) :
  (angle C = π / 3) ∧ ((a + b + c) = 5 + √7) := sorry

end triangle_problem_l535_535274


namespace arithmetic_sequence_S5_l535_535306

-- Define the variables and conditions for the arithmetic sequence
variable {a : ℕ → ℤ} -- the arithmetic sequence terms
variable {S : ℕ → ℤ} -- the sum of the first n terms
variable {d : ℤ} -- the common difference

-- Given conditions
def a2_eq_1 : a 2 = 1 := by sorry
def a4_eq_5 : a 4 = 5 := by sorry

-- Definition of the sum of the first n terms of an arithmetic sequence
def S_def (n : ℕ) : ℤ :=
  ∑ k in Finset.range n, a (k + 1)

-- Assertion to be proved
theorem arithmetic_sequence_S5 : S 5 = 15 :=
by
  -- Given arithmetic sequence properties and the derived common difference
  have d_def : d = (a 4 - a 2) / 2 := by sorry
  have a1_def : a 1 = a 2 - d := by sorry

  -- Calculation using the sum formula for the arithmetic sequence
  have S5_cal : S 5 = 5 * (a 1) + (5 * 4 / 2) * d := by sorry

  -- Substituting the values
  calc
    S 5 = 5 * (a 2 - d) + (10) * d := by rw [a1_def, S5_cal] -- Since 5 * 4 / 2 = 10
       ... = 5 * (1 - d) + 10 * d  := by rw [a2_eq_1]
       ... = 5 * 1 - 5 * d + 10 * d := by ring
       ... = 5 + 5 * d := by ring
       ... = 15 := by rw [d_def]; ring -- Plugging in the calculated value of d

end arithmetic_sequence_S5_l535_535306


namespace count_basic_events_probability_diff_three_probability_product_six_l535_535013

-- Define the sample space for two dice rolls
def sample_space : Finset (ℕ × ℕ) :=
  Finset.univ.product Finset.univ

-- Define the event "difference is 3"
def event_diff_three (x y : ℕ) : Prop := abs (x - y) = 3

-- Define the event "product is 6"
def event_product_six (x y : ℕ) : Prop := x * y = 6

-- Proof statement for problem 1: There are 36 different basic events
theorem count_basic_events : sample_space.card = 36 :=
  sorry

-- Proof statement for problem 2: The probability of "the difference between the numbers facing up is 3" is 1/6
theorem probability_diff_three : 
  (Finset.filter (λ (p : ℕ × ℕ), event_diff_three p.1 p.2) sample_space).card / sample_space.card = 1/6 :=
  sorry

-- Proof statement for problem 3: The probability of "the product of the numbers facing up is 6" is 1/9
theorem probability_product_six : 
  (Finset.filter (λ (p : ℕ × ℕ), event_product_six p.1 p.2) sample_space).card / sample_space.card = 1/9 :=
  sorry

end count_basic_events_probability_diff_three_probability_product_six_l535_535013


namespace dogwood_trees_after_work_l535_535283

theorem dogwood_trees_after_work 
  (trees_part1 : ℝ) (trees_part2 : ℝ) (trees_part3 : ℝ)
  (trees_cut : ℝ) (trees_planted : ℝ)
  (h1 : trees_part1 = 5.0) (h2 : trees_part2 = 4.0) (h3 : trees_part3 = 6.0)
  (h_cut : trees_cut = 7.0) (h_planted : trees_planted = 3.0) :
  trees_part1 + trees_part2 + trees_part3 - trees_cut + trees_planted = 11.0 :=
by
  sorry

end dogwood_trees_after_work_l535_535283


namespace incorrect_corresponding_angles_l535_535477

theorem incorrect_corresponding_angles (l1 l2 l3 : Line) 
(H1 : ∀ a b, (a ∈ l1 ∧ b ∈ l2 ∧ Intersection(l1, l2))) 
(H2 : Vertical_angles_equal(l1, l2)) 
(H3 : Adjacent_angles_not_necessarily_complementary(l1, l2, l3)) 
(H4 : If_alternate_interior_angles_equal_then_corresponding_angles_equal(l1, l2, l3))
: ¬ Corresponding_angles_equal(l1, l2, l3) :=
sorry

end incorrect_corresponding_angles_l535_535477


namespace remainder_correct_l535_535510

noncomputable def dividend := 3 * X^5 + 4 * X^3 - 2 * X^2 + 5 * X - 8
noncomputable def divisor := X^2 + 3 * X + 2
noncomputable def remainder := 66 * X - 22

theorem remainder_correct : ∃ q : Polynomial ℝ, dividend = divisor * q + remainder := 
by 
  sorry

end remainder_correct_l535_535510


namespace find_y_l535_535711

-- Definitions for the conditions
def has_24_factors (y : ℕ) : Prop :=
  ∏ (p: ℕ) in (nat.factors y).to_finset, (p.count (nat.factors y) + 1) = 24

def is_factor (x y: ℕ) : Prop := y % x = 0

-- The main statement to prove
theorem find_y (y : ℕ) (h1: has_24_factors y) (h2: is_factor 18 y) (h3: is_factor 20 y) : y = 360 :=
sorry

end find_y_l535_535711


namespace converse_statement_b_false_l535_535308

/--
Given three lines \(a\), \(b\), \(c\) in space, and two planes \(\alpha\), \(\beta\) in space.

Statement A: If \(c \perp \alpha\) and \(c \perp \beta\), then \(\alpha \parallel \beta\).

Statement B: If \(b \subset \alpha\) and \(b \perp \beta\), then \(\alpha \perp \beta\).

Statement C: If \(b \subset \alpha\), \(c\) is the projection of \(a\) on \(\alpha\), and \(b \perp c\), then \(a \perp b\).

Statement D: If \(b \subset \alpha\) and \(c \nsubseteq \alpha\) and \(c \parallel \alpha\), then \(b \parallel c\).

Prove that the converse of Statement B is false.
-/
theorem converse_statement_b_false (a b c : Line) (alpha beta : Plane) :
  (b \subset alpha) ∧ (b \perp beta) → (alpha \perp beta) → False := by
  sorry

end converse_statement_b_false_l535_535308


namespace distance_center_to_line_l535_535769

def circle_polar (ρ θ: ℝ) : Prop := ρ^2 + 2 * ρ * cos θ - 3 = 0
def line_polar (ρ θ : ℝ) : Prop := ρ * cos θ + ρ * sin θ - 7 = 0

theorem distance_center_to_line :
  (∃ ρ θ, circle_polar ρ θ) →
  (∃ ρ θ, line_polar ρ θ) →
  ∃ ρ θ, ρ * cos θ + ρ * sin θ - 7 = 0 →
  let center := (-1 : ℝ, 0 : ℝ) in
  let line_func := (λ (x y : ℝ), x + y - 7) in
  abs (center.1 + center.2 - 7) / sqrt 2 = 4 * sqrt 2 :=
by
  sorry

end distance_center_to_line_l535_535769


namespace sin_alpha_value_l535_535950

open Real

theorem sin_alpha_value (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : cos (α + π / 4) = 1 / 3) :
  sin α = (4 - sqrt 2) / 6 :=
sorry

end sin_alpha_value_l535_535950


namespace ratio_of_puzzle_times_l535_535741

def total_time := 70
def warmup_time := 10
def remaining_puzzles := 60 / 2

theorem ratio_of_puzzle_times : (remaining_puzzles / warmup_time) = 3 := by
  -- Given Conditions
  have H1 : 70 = 10 + 2 * (60 / 2) := by sorry
  -- Simplification and Calculation
  have H2 : (remaining_puzzles = 30) := by sorry
  -- Ratio Calculation
  have ratio_calculation: (30 / 10) = 3 := by sorry
  exact ratio_calculation

end ratio_of_puzzle_times_l535_535741


namespace total_fertilizer_used_l535_535461

def daily_fertilizer := 3
def num_days := 12
def extra_final_day := 6

theorem total_fertilizer_used : 
    (daily_fertilizer * num_days + (daily_fertilizer + extra_final_day)) = 45 :=
by
  sorry

end total_fertilizer_used_l535_535461


namespace solve_abs_equation_l535_535128

theorem solve_abs_equation (x : ℝ) (h : | x + 1 | = 2 * x + 4) :
  x = -5 / 3 := by
  sorry

end solve_abs_equation_l535_535128


namespace volume_of_region_l535_535513

theorem volume_of_region :
  {x y z : ℝ // x + y + z ≤ 12 / 2 ∧ x + y - z ≤ 12 / 2 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0} ⊆
  {p : ℝ^3 // p.1 + p.2 ≤ 6 ∧ p.3 ≤ 6} →
  volume_of_region (λ p, p.1 + p.2 + p.3 ≤ 12 / 2 ∧ p.1 + p.2 - p.3 ≤ 12 / 2 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.3 ≥ 0) 
   = 108 :=
sorry

end volume_of_region_l535_535513


namespace f_2014_value_l535_535552

noncomputable def f : ℝ → ℝ := sorry

theorem f_2014_value (h₁ : ∀ x : ℝ, f(x + 6) + f(x) = 0)
                      (h₂ : ∀ x : ℝ, f(x - 1) = f(-(x - 1) + 2))
                      (h₃ : f 2 = 4) :
                      f 2014 = -4 := sorry

end f_2014_value_l535_535552


namespace smallest_argument_complex_l535_535812

theorem smallest_argument_complex:
  ∃ Z : ℂ, |Z - 25 * complex.I| ≤ 15 ∧ Z = 12 + 16 * complex.I := 
sorry

end smallest_argument_complex_l535_535812


namespace range_of_m_l535_535574

theorem range_of_m (m : ℝ) (x y : ℝ)
  (h1 : x + y - 3 * m = 0)
  (h2 : 2 * x - y + 2 * m - 1 = 0)
  (h3 : x > 0)
  (h4 : y < 0) : 
  -1 < m ∧ m < 1/8 := 
sorry

end range_of_m_l535_535574


namespace primes_count_l535_535862

open Int

theorem primes_count (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ r s : ℤ, ∀ x : ℤ, (x^3 - x + 2) % p = ((x - r)^2 * (x - s)) % p := 
  by
    sorry

end primes_count_l535_535862


namespace det_B_eq_one_l535_535314

-- We assume that a, d are real numbers and define the matrix B
variables (a d : ℝ)
def B : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]

-- We can state the problem as follows
theorem det_B_eq_one (h : B a d + B a d⁻¹ = 0) : Matrix.det (B a d) = 1 := by
  sorry

end det_B_eq_one_l535_535314


namespace probability_all_same_flips_l535_535007

noncomputable def four_same_flips_probability : ℚ := 
  (∑' n : ℕ, if n > 0 then (1/2)^(4*n) else 0)

theorem probability_all_same_flips : 
  four_same_flips_probability = 1 / 15 := 
sorry

end probability_all_same_flips_l535_535007


namespace sin_C_and_area_of_triangle_l535_535638

theorem sin_C_and_area_of_triangle (a b c : ℕ) (A B C : ℝ) 
  (h1 : cos A = 5 / 13) 
  (h2 : tan (B / 2) + cot (B / 2) = 10 / 3) 
  (h3 : c = 21) 
  : sin C = 63 / 65 ∧ (1/2) * a * c * sin B = 126 := 
by sorry

end sin_C_and_area_of_triangle_l535_535638


namespace problem_incorrect_proposition_C_correct_proposition_A_correct_proposition_B_correct_proposition_D_l535_535816

-- Definitions for propositions expressed as predicates
variables (m n : Line) (α β : Plane)

-- Translate propositions into Lean terms
def proposition_A : Prop := (m ∉ α ∧ n ∈ α ∧ m ∥ n) → m ∥ α
def proposition_B : Prop := (m ⊥ α ∧ m ∈ β) → α ⊥ β
def proposition_C : Prop := (m ∈ α ∧ n ∈ β ∧ α ∥ β) → m ∥ n
def proposition_D : Prop := (m ⊥ α ∧ m ⊥ β) → α ∥ β

-- Define the proof problem
theorem problem_incorrect_proposition_C : ¬proposition_C :=
begin
  -- Proposition C is incorrect, as m and n could be either parallel or skew
  sorry
end

-- Check other propositions are correct
theorem correct_proposition_A : proposition_A :=
by sorry

theorem correct_proposition_B : proposition_B :=
by sorry

theorem correct_proposition_D : proposition_D :=
by sorry

end problem_incorrect_proposition_C_correct_proposition_A_correct_proposition_B_correct_proposition_D_l535_535816


namespace number_of_ones_and_twos_in_N_l535_535059

def is_natural (n : ℕ) : Prop := n ≥ 0

def is_hundred_digit_number (N : ℕ → ℕ) : Prop :=
  ∑ i in range 100, (N i = 1 ∨ N i = 2)

def even_number_between_twos (N : ℕ → ℕ) : Prop :=
  ∀ i j, i < j → N i = 2 → N j = 2 → ∑ k in (range (i+1)).filter (λ k, k < j), (N k = 1 ∨ N k = 2) % 2 = 0

def divisible_by_three (N : ℕ → ℕ) : Prop :=
  (∑ i in range 100, N i) % 3 = 0

def number_of_ones_and_twos (N : ℕ → ℕ) : ℕ × ℕ :=
  (∑ i in range 100, if N i = 1 then 1 else 0, ∑ i in range 100, if N i = 2 then 1 else 0)

theorem number_of_ones_and_twos_in_N (N : ℕ → ℕ) (h⁄1 : is_hundred_digit_number N)
  (h⁄2 : even_number_between_twos N) (h⁄3 : divisible_by_three N) : number_of_ones_and_twos N = (98, 2) :=
sorry

end number_of_ones_and_twos_in_N_l535_535059


namespace max_min_sum_s_l535_535202

theorem max_min_sum_s {x y : ℝ} (h : x^2 + y^2 = 1) : 
  let S := x^4 + x * y + y^4 
  in (max (S) + min (S)) = 9 / 8 :=
sorry

end max_min_sum_s_l535_535202


namespace total_votes_cast_l535_535265

variable (V : ℕ)

def james_votes := 0.005 * V
def winning_threshold := 0.50 * V
def additional_votes_needed := 991

theorem total_votes_cast :
  james_votes V + additional_votes_needed > winning_threshold V →
  V = 2004 :=
by
  sorry

end total_votes_cast_l535_535265


namespace arrangements_number_l535_535389

-- Definitions
variable (Students : Fin 4)
variable podium : Fin 1
variable sweep : Fin 1
variables mop1 mop2 : Fin 2

-- Problem
theorem arrangements_number : 
  ∃ arrangements : ℕ, arrangements = 12 := by
  sorry

end arrangements_number_l535_535389


namespace angle_sum_l535_535206

noncomputable def theta_phi_sum : Real :=
  let θ : Real := Real.atan (3 / 4)
  let φ : Real := Real.asin (3 / 5)
  θ + φ

theorem angle_sum (θ φ : Real) 
  (hθ : 0 < θ ∧ θ < π / 2)
  (hφ : 0 < φ ∧ φ < π / 2)
  (h1 : Real.tan θ = 3 / 4)
  (h2 : Real.sin φ = 3 / 5) :
  θ + φ = Real.atan (24 / 7) :=
by
  sorry

end angle_sum_l535_535206


namespace find_missing_number_l535_535612

def average (l : List ℕ) : ℚ := l.sum / l.length

theorem find_missing_number : 
  ∃ x : ℕ, 
    average [744, 745, 747, 748, 749, 752, 752, 753, 755, x] = 750 :=
sorry

end find_missing_number_l535_535612


namespace salary_increase_l535_535327

theorem salary_increase :
  let S : ℝ := 1
  let rate : ℝ := 1.12
  let n : ℝ := 5
  ((rate ^ n) - 1) * 100 ≈ 76 := by
  sorry

end salary_increase_l535_535327


namespace fundraiser_goal_eq_750_l535_535383

def bronze_donations := 10 * 25
def silver_donations := 7 * 50
def gold_donations   := 1 * 100
def total_collected  := bronze_donations + silver_donations + gold_donations
def amount_needed    := 50
def total_goal       := total_collected + amount_needed

theorem fundraiser_goal_eq_750 : total_goal = 750 :=
by
  sorry

end fundraiser_goal_eq_750_l535_535383


namespace rhombus_in_triangle_l535_535033

theorem rhombus_in_triangle 
  (A B C M K N : Point)
  (m n : ℝ)
  (h_rhombus : rhombus AMKN)
  (h_triangle : triangle ABC)
  (h_shared_angle : angle BAC = angle NAM)
  (h_diagonal_ratio : divides_vertex_side K A M C 2 3)
  (h_diagonals_length : length (diagonal AM) = m ∧ length (diagonal KN) = n):
  (side_length AB = (5 / 6) * sqrt (m^2 + n^2) ∧ 
   side_length AC = (5 / 4) * sqrt (m^2 + n^2)) :=
sorry

end rhombus_in_triangle_l535_535033


namespace remaining_payment_is_correct_l535_535443

-- Define the total cost of the product
def total_cost (d : ℝ) := d / 0.10

-- Define the deposit
def deposit : ℝ := 150

-- Define the remaining amount to be paid
def remaining_amount (t : ℝ) (d : ℝ) := t - d

-- Theorem statement
theorem remaining_payment_is_correct :
  remaining_amount (total_cost deposit) deposit = 1350 := by
  sorry

end remaining_payment_is_correct_l535_535443


namespace count_non_squares_or_cubes_l535_535941

theorem count_non_squares_or_cubes (n : ℕ) (h₀ : 1 ≤ n ∧ n ≤ 200) : 
  ∃ c, c = 182 ∧ 
  (∃ k, k^2 = n ∨ ∃ m, m^3 = n) → false :=
by
  sorry

end count_non_squares_or_cubes_l535_535941


namespace total_visible_surface_area_of_stack_l535_535846

theorem total_visible_surface_area_of_stack : 
  let side_lengths := [9, 8, 7, 6, 5, 4, 3, 1]
  let visible_surface_area (s: ℕ) (above: Bool := false) : ℕ :=
    if above then 5 * s^2
    else if s == 1 then 4 * s^2 else 5 * s^2
  in 
  (visible_surface_area 9) + (visible_surface_area 8) + (visible_surface_area 7) + (visible_surface_area 6) + 
  (visible_surface_area 5) + (visible_surface_area 4) + (visible_surface_area 3) + (visible_surface_area 1 true) = 1408 :=
by
  sorry

end total_visible_surface_area_of_stack_l535_535846


namespace increase_by_fraction_l535_535412

theorem increase_by_fraction (original_value : ℕ) (fraction : ℚ) : original_value = 120 → fraction = 5/6 → original_value + original_value * fraction = 220 :=
by
  intros h1 h2
  sorry

end increase_by_fraction_l535_535412


namespace tan_alpha_minus_pi_over_4_l535_535868

variable {α β : ℝ}

theorem tan_alpha_minus_pi_over_4 (h1 : tan (α + β) = 2 / 5) (h2 : tan (β + π / 4) = 1 / 4) :
  tan (α - π / 4) = 3 / 22 := by
  sorry

end tan_alpha_minus_pi_over_4_l535_535868


namespace equal_segments_l535_535291

noncomputable def incircle_touches_at (A B C D : Point) (O : Circle) (incircle : Incircle) : Prop :=
  -- Definition representing the incircle touching BC at D in triangle ABC with center O.
  incircle.touches BC D ∧ incircle = Circle.mk O

noncomputable def diameter_of_incircle (D E : Point) (O : Circle) : Prop :=
  -- Definition representing DE as the diameter of incircle with center O.
  diameter DE O

noncomputable def line_intersects (A E F : Point) (BC : Line) : Prop :=
  -- Definition representing the line AE intersects BC at F.
  intersection A E BC = F

theorem equal_segments (A B C D E F : Point) (O : Circle) (incircle : Incircle) (BC : Line) :
  incircle_touches_at A B C D O incircle → 
  diameter_of_incircle D E O →
  line_intersects A E F BC →
  seg_length B D = seg_length C F :=
begin
  intros h1 h2 h3,
  sorry -- Proof goes here
end

end equal_segments_l535_535291


namespace same_face_probability_correct_l535_535603

-- Define the number of sides on the dice
def sides_20 := 20
def sides_16 := 16

-- Define the number of colored sides for each dice category
def maroon_20 := 5
def teal_20 := 8
def cyan_20 := 6
def sparkly_20 := 1

def maroon_16 := 4
def teal_16 := 6
def cyan_16 := 5
def sparkly_16 := 1

-- Define the probabilities of each color matching
def prob_maroon : ℚ := (maroon_20 / sides_20) * (maroon_16 / sides_16)
def prob_teal : ℚ := (teal_20 / sides_20) * (teal_16 / sides_16)
def prob_cyan : ℚ := (cyan_20 / sides_20) * (cyan_16 / sides_16)
def prob_sparkly : ℚ := (sparkly_20 / sides_20) * (sparkly_16 / sides_16)

-- Define the total probability of same face
def prob_same_face := prob_maroon + prob_teal + prob_cyan + prob_sparkly

-- The theorem we need to prove
theorem same_face_probability_correct : 
  prob_same_face = 99 / 320 :=
by
  sorry

end same_face_probability_correct_l535_535603


namespace distinct_multiples_of_5_l535_535938

theorem distinct_multiples_of_5 : 
  (finset.card {n | 
    ∃ x y z w, (x, y, z, w) ∈ {(2, 5, 5, 0), (2, 5, 0, 5), (5, 2, 5, 0), (5, 2, 0, 5), 
                                (5, 5, 2, 0), (5, 0, 2, 5), (2, 0, 5, 5), (0, 2, 5, 5), 
                                (0, 5, 2, 5), (5, 0, 5, 2), (0, 5, 5, 2), (5, 5, 0, 2)},
    (n = 1000 * x + 100 * y + 10 * z + w) ∧ (w = 0 ∨ w = 5)
  }) = 7 :=
sorry

end distinct_multiples_of_5_l535_535938


namespace equivalent_statements_l535_535434

theorem equivalent_statements (P Q : Prop) : (¬P → Q) ↔ (¬Q → P) :=
by
  sorry

end equivalent_statements_l535_535434


namespace legs_in_park_l535_535632

def total_legs (dogs cats birds spiders missing_dog_legs missing_cat_legs missing_spider_legs : ℕ) : ℕ :=
  let dog_legs := dogs * 4
  let cat_legs := cats * 4
  let bird_legs := birds * 2
  let spider_legs := spiders * 8
  dog_legs + cat_legs + bird_legs + spider_legs - missing_dog_legs - missing_cat_legs - missing_spider_legs

theorem legs_in_park 
(dogs cats birds spiders missing_dog_legs missing_cat_legs missing_spider_legs : ℕ)
(h_dogs : dogs = 109) (h_cats : cats = 37) (h_birds : birds = 52) (h_spiders : spiders = 19)
(h_missing_dog_legs : missing_dog_legs = 4) (h_missing_cat_legs : missing_cat_legs = 3) 
(h_missing_spider_legs : missing_spider_legs = 4) :
  total_legs dogs cats birds spiders missing_dog_legs missing_cat_legs missing_spider_legs = 829 := 
  by 
  rw [h_dogs, h_cats, h_birds, h_spiders, h_missing_dog_legs, h_missing_cat_legs, h_missing_spider_legs]
  simp [total_legs]
  sorry

end legs_in_park_l535_535632


namespace count_negative_numbers_l535_535262

theorem count_negative_numbers : 
  let expr1 := -2^2
  let expr2 := (-2)^2
  let expr3 := -(-2)
  let expr4 := -|-2|
  1 ≤ expr1 ∧ expr1 < 0 ∧
  0 ≤ expr2 ∧ 0 < expr2 ∧ 
  0 ≤ expr3 ∧ 0 < expr3 ∧ 
  1 ≤ expr4 ∧ expr4 < 0 ∧
  (expr1 < 0) + (expr2 < 0) + (expr3 < 0) + (expr4 < 0) = 2 := 
by
  sorry

end count_negative_numbers_l535_535262


namespace problem_l535_535875

theorem problem (a b c : ℂ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 6) :
  (a - 1)^(2023) + (b - 1)^(2023) + (c - 1)^(2023) = 0 :=
by
  sorry

end problem_l535_535875


namespace monotonic_intervals_extreme_values_on_interval_l535_535923

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

theorem monotonic_intervals:
  (∀ x : ℝ, x < -1 → 0 < (f' x)) ∧ 
  (∀ x : ℝ, x > 1 → 0 < (f' x)) ∧ 
  (∀ x : ℝ, -1 < x ∧ x < 1 → (f' x) < 0) := 
by {
  sorry
}

theorem extreme_values_on_interval : 
  (∃ x_min x_max, -3 ≤ x_min ∧ x_min ≤ 3 ∧ -3 ≤ x_max ∧ x_max ≤ 3 ∧ 
   f x_min = -49 ∧ f x_max = 59) :=
by {
  exists 3, -3,
  sorry
}

end monotonic_intervals_extreme_values_on_interval_l535_535923


namespace paper_cut_square_l535_535330

noncomputable def proof_paper_cut_square : Prop :=
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ ((2 * x - 2 = 2 - x) ∨ (2 * (2 * x - 2) = 2 - x)) ∧ (x = 1.2 ∨ x = 1.5)

theorem paper_cut_square : proof_paper_cut_square :=
sorry

end paper_cut_square_l535_535330


namespace geometric_sequence_fifth_term_l535_535631

theorem geometric_sequence_fifth_term (a₁ : ℕ) (q : ℕ) (h₁ : a₁ = 1) (hq : q = 2) : 
  let a₅ := a₁ * q^(5-1)
  in a₅ = 16 := 
by
  let a₅ := a₁ * q^(5-1)
  have h : a₅ = 1 * 2^4, by
    rw [h₁, hq]
  rw [←h]
  exact rfl

end geometric_sequence_fifth_term_l535_535631


namespace ellen_final_legos_l535_535521

def initial_legos : ℕ := 2080
def rank_3_legos : ℕ := 17
def additional_legos : ℕ := Int.floor (0.045 * initial_legos.toReal).toNat
def final_legos : ℕ := initial_legos + rank_3_legos + additional_legos

theorem ellen_final_legos : final_legos = 2190 := by
  show final_legos = 2080 + 17 + 93
  sorry

end ellen_final_legos_l535_535521


namespace yard_area_l535_535295

theorem yard_area (posts : Nat) (spacing : Real) (longer_factor : Nat) (shorter_side_posts longer_side_posts : Nat)
  (h1 : posts = 24)
  (h2 : spacing = 3)
  (h3 : longer_factor = 3)
  (h4 : 2 * (shorter_side_posts + longer_side_posts) = posts - 4)
  (h5 : longer_side_posts = 3 * shorter_side_posts + 2) :
  (spacing * (shorter_side_posts - 1)) * (spacing * (longer_side_posts - 1)) = 144 :=
by
  sorry

end yard_area_l535_535295


namespace number_of_integers_between_sqrt10_sqrt40_l535_535243

theorem number_of_integers_between_sqrt10_sqrt40 : 
  ∃ (n : ℕ), n = 3 ∧ ∀ k : ℕ, 4 ≤ k ∧ k ≤ 6 → k ∈ { 4, 5, 6 } :=
by
  sorry

end number_of_integers_between_sqrt10_sqrt40_l535_535243


namespace norm_b_in_range_l535_535892

-- Definitions of vectors, dot product, and norms
variable {V : Type*} [inner_product_space ℝ V]

-- Given that \(\overset{→}{a}\) is a unit vector on a plane
variable (a b : V)
variable (h_unit : ∥a∥ = 1)

-- \(\overset{→}{b}\) satisfies \(\overset{→}{b}·(\overset{→}{a}-\overset{→}{b})=0\).
variable (h_condition : ⟪b, a - b⟫ = 0)

-- Prove that \(|\overset{→}{b}| ∈ [0, 1]\).
theorem norm_b_in_range : ∥b∥ ≤ 1 :=
sorry

end norm_b_in_range_l535_535892


namespace number_of_correct_statements_l535_535393

def statement_1 (u : ℕ → ℝ) (h : ∃ n, u n = 0) : ¬ is_geometric u :=
  sorry

def statement_2 (q : ℝ) (h : q ∈ set.univ) : not (q = 0 → is_geometric (λ n, q^n)) :=
  sorry

def statement_3 (a b c : ℝ) (h : b^2 = a * c) : ¬ (is_geometric_seq a b c) :=
  sorry

def statement_4 (u : ℕ → ℝ) (h : forall n, u n = u 0) : (is_geometric u ∧ u 0 ≠ 0 → ∀ n, u n = 1) :=
  sorry

def count_correct_statements : ℕ :=
if statement_1 u h ∧ statement_2 q h ∧ statement_3 a b c h ∧ statement_4 u h then 1 else 0

theorem number_of_correct_statements : count_correct_statements = 1 :=
  sorry

end number_of_correct_statements_l535_535393


namespace percentage_blue_pigment_correct_l535_535785

/-- A certain dark blue paint contains some percent blue pigment (B) and 
    60 percent red pigment by weight. A certain green paint contains the same 
    percent blue pigment (B) and 60 percent yellow pigment. When these paints 
    are mixed to produce a brown paint, it contains 40 percent blue pigment, 
    and the brown paint weighs 10 grams. The red pigment contributes 3 grams 
    of that weight. Prove that the percentage of blue pigment (B) in the dark 
    blue and green paints is 20%. -/
noncomputable def percentage_blue_pigment_in_paints : Prop :=
  let B : ℝ := 0.20 in
  let D : ℝ := 5 in
  let G : ℝ := 10 - D in
  (D * 0.6 = 3) ∧ 
  (D + G = 10) ∧ 
  (B * D + B * G = 10 * 0.4) →
  (B = 0.20)

theorem percentage_blue_pigment_correct :
  percentage_blue_pigment_in_paints := by
  sorry

end percentage_blue_pigment_correct_l535_535785


namespace cube_sides_count_l535_535440

theorem cube_sides_count (n : ℕ) (paint_colors : Fin n → Prop)
  (unique_painting_conditions : ∀ f : Fin n → Fin 6, ∃! (g : Fin n → Fin 6), 
  IsValidPainting g ∧ IsRotationEquivalent f g)
  (num_paintings : num_distinct_paintings(f : Fin n → Fin 6, g : Fin n → Fin 6) = 30) :
  n = 6 := 
sorry

def IsValidPainting (f : Fin n → Fin 6) : Prop :=
 sorry

def IsRotationEquivalent (f g : Fin n → Fin 6) : Prop :=
 sorry

def num_distinct_paintings : ℕ := 
 sorry

end cube_sides_count_l535_535440


namespace tan_theta_pure_imaginary_l535_535606

theorem tan_theta_pure_imaginary (θ : Real) :
  let z := Complex.ofReal (Real.sin θ - 3/5) + Complex.i * Complex.ofReal (Real.cos θ - 4/5)
  z.im ≠ 0 → z.re = 0 → θ ≠ π/2 → Real.tan θ = -3/4 :=
by
  intro z _ Im_ne_0 Re_eq_0 theta_ne_pi_over_2
  have : Real.sin θ = 3/5 
  have : Real.cos θ = -4/5
  sorry

end tan_theta_pure_imaginary_l535_535606


namespace problem_statement_l535_535580

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem problem_statement :
  f(2) = 4 / 5 ∧
  f(1 / 2) = 1 / 5 ∧
  f(3) = 9 / 10 ∧
  f(1 / 3) = 1 / 10 ∧
  ∀ x : ℝ, x ≠ 0 → f(x) + f(1 / x) = 1 ∧
  f(1) + ∑ i in (finset.range 2013).map (λ i : ℕ, i + 2), (f(i) + f(1 / i)) = 4025 / 2 :=
by
  sorry

end problem_statement_l535_535580


namespace tabitha_current_age_l535_535119

noncomputable def tabithaAge (currentColors : ℕ) (yearsPassed : ℕ) (startAge : ℕ) (futureYears : ℕ) (futureColors : ℕ) : Prop :=
  (currentColors = (futureColors - futureYears)) ∧
  (yearsPassed = (currentColors - 2)) ∧
  (yearsPassed + startAge = 18)

theorem tabitha_current_age : tabithaAge 5 3 15 3 8 := 
by
  unfold tabithaAge
  split
  all_goals {simp}
  sorry

end tabitha_current_age_l535_535119


namespace unique_irrational_sum_l535_535697

-- Definition of the problem
def sequence_expression (α : ℝ) : Prop := 
  ∃ a : ℕ → ℕ, (∀ n : ℕ, a n < a (n+1)) ∧
  α = ∑' n, (-1)^(n+1) / (∏ i in Finset.range (n+1), (a i))

-- Given the conditions, we need to prove:
theorem unique_irrational_sum (α : ℝ) (hα : α ∈ Ioo 0 1) :
  sequence_expression α ∧ (∀ {a b : ℕ → ℕ}, 
  (∀ n : ℕ, a n < a (n+1)) ∧ (∀ n : ℕ, b n < b (n+1)) → 
  α = ∑' n, (-1)^(n+1) / (∏ i in Finset.range (n+1), (a i)) → 
  α = ∑' n, (-1)^(n+1) / (∏ i in Finset.range (n+1), (b i)) → 
  a = b) :=
sorry

end unique_irrational_sum_l535_535697


namespace cyclic_sum_inequality_l535_535550

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  ( ( (a - b) * (a - c) / (a + b + c) ) + 
    ( (b - c) * (b - d) / (b + c + d) ) + 
    ( (c - d) * (c - a) / (c + d + a) ) + 
    ( (d - a) * (d - b) / (d + a + b) ) ) ≥ 0 := 
by
  sorry

end cyclic_sum_inequality_l535_535550


namespace planes_perpendicular_parallel_l535_535814

-- Let α, β, γ be planes
variables {α β γ : Plane}

-- Define parallel and perpendicular relations
def parallel (π1 π2 : Plane) : Prop := ∀ (l : Line), l ∈ π1 → l ∈ π2
def perpendicular (π1 π2 : Plane) : Prop := ∀ (l1 l2 : Line), l1 ∈ π1 → l2 ∈ π2 → (l1 ⊥ l2)

theorem planes_perpendicular_parallel {α β γ : Plane}
  (h1 : perpendicular α β) (h2: parallel α γ) : perpendicular β γ :=
sorry

end planes_perpendicular_parallel_l535_535814


namespace count_ways_to_sum_2020_as_1s_and_2s_l535_535506

theorem count_ways_to_sum_2020_as_1s_and_2s : ∃ n, (∀ x y : ℕ, 4 * x + 5 * y = 2020 → x + y = n) → n = 102 :=
by
-- Mathematics proof needed.
sorry

end count_ways_to_sum_2020_as_1s_and_2s_l535_535506


namespace sum_of_digits_of_greatest_prime_factor_of_16777_l535_535386

noncomputable def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_greatest_prime_factor_of_16777 :
  ∑ p in nat.prime_divisors 16777, if nat.prime_divisors 16777 = {11} then digits_sum p = 2 :=
by 
  sorry

end sum_of_digits_of_greatest_prime_factor_of_16777_l535_535386


namespace find_m_l535_535218

open Nat

theorem find_m : ∃ m : ℝ, m > 0 ∧ (binom 5 2 * m^2 - binom 5 1 * m = 30) ∧ m = 2 :=
by
  use 2
  split
  { linarith }  -- proof for m > 0
  split
  { norm_num [binom], linarith } -- proof for the binomial theorem part
  { rfl } -- proof that m = 2
  sorry

end find_m_l535_535218


namespace find_analytical_expression_of_f_l535_535569

-- Define the function f and the condition it needs to satisfy
variable (f : ℝ → ℝ)
variable (hf : ∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1))

-- State the objective to prove
theorem find_analytical_expression_of_f : 
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x / (1 + x) := by
  sorry

end find_analytical_expression_of_f_l535_535569


namespace slope_of_tangent_at_neg5_l535_535363

variable {f : ℝ → ℝ}

-- Given conditions
axiom even_function : ∀ x, f(-x) = f(x)
axiom derivable_function : ∀ x, ∃ f', f' = deriv f x
axiom derivative_at_1 : deriv f 1 = -2
axiom periodic_function : ∀ x, f(x + 2) = f(x - 2) 

-- Proof statement
theorem slope_of_tangent_at_neg5 : deriv f (-5) = 2 := by
  sorry

end slope_of_tangent_at_neg5_l535_535363


namespace number_of_boxes_l535_535596

def magazines : ℕ := 63
def magazines_per_box : ℕ := 9

theorem number_of_boxes : magazines / magazines_per_box = 7 :=
by 
  sorry

end number_of_boxes_l535_535596


namespace like_terms_implies_a_plus_2b_eq_3_l535_535949

theorem like_terms_implies_a_plus_2b_eq_3 (a b : ℤ) (h1 : 2 * a + b = 6) (h2 : a - b = 3) : a + 2 * b = 3 :=
sorry

end like_terms_implies_a_plus_2b_eq_3_l535_535949


namespace smallest_integer_n_for_rotation_matrix_l535_535533

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem smallest_integer_n_for_rotation_matrix :
  ∃ n : ℕ, 0 < n ∧ (rotation_matrix (140 * Real.pi / 180)) ^ n = 1 ∧ (∀ m : ℕ, 0 < m ∧ (rotation_matrix (140 * Real.pi / 180)) ^ m = 1 → n ≤ m) := 
begin 
  sorry 
end

end smallest_integer_n_for_rotation_matrix_l535_535533


namespace quotient_correct_l535_535421

theorem quotient_correct (dividend divisor remainder quotient : ℕ)
  (h1 : dividend = 149) (h2 : divisor = 16) (h3 : remainder = 5) (h4 : dividend - remainder = 144) (h5 : 144 / divisor = quotient) : quotient = 9 :=
by
  rw [h1, h2, h3] at h4,
  rw h4 at h5,
  exact h5

end quotient_correct_l535_535421


namespace length_of_platform_l535_535779

-- Defining the given constants
def train_length : ℝ := 750.0
def time_to_cross_platform : ℝ := 97.0
def time_to_cross_pole : ℝ := 90.0
def speed_of_train : ℝ := train_length / time_to_cross_pole

-- Defining the equation to find the platform length
def platform_length : ℝ := 808.33 - train_length

theorem length_of_platform :
  let L := (speed_of_train * time_to_cross_platform) - train_length in
  L = 58.33 :=
  by
    -- All the steps to find L based on given conditions would go here
    sorry

end length_of_platform_l535_535779


namespace length_of_BC_l535_535480

theorem length_of_BC (a : ℝ) :
  (∀ B C, B = (-a, 2 * a^2) ∧ C = (a, 2 * a^2) ∧ ∃ A, A = (0, 0) ∧ 
    (distance B C = 2 * a) ∧ (area_of_triangle A B C = 128)) →
  (2 * a = 8) :=
by
  sorry

end length_of_BC_l535_535480


namespace count_non_squares_or_cubes_l535_535940

theorem count_non_squares_or_cubes (n : ℕ) (h₀ : 1 ≤ n ∧ n ≤ 200) : 
  ∃ c, c = 182 ∧ 
  (∃ k, k^2 = n ∨ ∃ m, m^3 = n) → false :=
by
  sorry

end count_non_squares_or_cubes_l535_535940


namespace no_equilateral_triangle_l535_535833

noncomputable def trihedral_angle (S A B C : Type) :=
  S ≠ A ∧ S ≠ B ∧ S ≠ C ∧
  ∠SBC < 60 ∧
  (S ⟶ A).perpendicular (plane S B C)

theorem no_equilateral_triangle (S A B C : Type) 
  (h : trihedral_angle S A B C) :
  ∀ (P : plane), ¬ (section_ABC_eq_equilateral P S A B C) :=
by
  sorry

/-- The angle at $SBC$ is less than $60^\circ$, and $SA$ is perpendicular to the plane $SBC$, 
    resulting in no equilateral triangle as a cross-section. -/

end no_equilateral_triangle_l535_535833


namespace find_x_l535_535864

theorem find_x (x : ℚ) : (3 : ℚ)^(3*x^2 - 8*x + 3) = (3 : ℚ)^(3*x^2 + 4*x - 5) -> x = 2 / 3 := 
by
  sorry

end find_x_l535_535864


namespace zero_count_in_interval_l535_535571

variables {R : Type*} [MetricSpace R] [NormedField R]

-- Define the function f and its properties
def f (x : ℝ) : ℝ := sorry

-- Given properties of f
axiom odd_function : ∀ x, f (-x) = - (f x)
axiom periodic_function : ∀ x, f (x + 3) = f x
axiom f_2_zero : f 2 = 0

-- Main theorem statement
theorem zero_count_in_interval : (set_of (λ x, f x = 0) ∩ set.Ioo 0 6).card = 2 :=
sorry

end zero_count_in_interval_l535_535571


namespace find_expression_l535_535035

theorem find_expression (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end find_expression_l535_535035


namespace volume_calculation_l535_535835

def volume_of_box : ℕ := 3 * 4 * 5
def volume_within_one_unit : ℚ := (462 + 40 * Real.pi) / 3
def m := 462
def n := 40
def p := 3

theorem volume_calculation :
  volume_of_box = 60 ∧
  volume_within_one_unit = 154 + (40 * Real.pi) / 3 ∧
  coprime n p →
  m + n + p = 505 :=
by
  intros
  sorry

end volume_calculation_l535_535835


namespace min_value_correct_l535_535855

noncomputable def min_value (x y : ℝ) : ℝ :=
x * y / (x^2 + y^2)

theorem min_value_correct :
  ∃ x y : ℝ,
    (2 / 5 : ℝ) ≤ x ∧ x ≤ (1 / 2 : ℝ) ∧
    (1 / 3 : ℝ) ≤ y ∧ y ≤ (3 / 8 : ℝ) ∧
    min_value x y = (6 / 13 : ℝ) :=
by sorry

end min_value_correct_l535_535855


namespace max_three_term_arithm_progressions_l535_535134

-- Define the main problem
theorem max_three_term_arithm_progressions (n : ℕ) (hn : n ≥ 3) :
  let f : ℕ → ℕ := λ k, (k-1) * (k-1) / 2
  ∃ seq : Finₓ n → ℝ, Monotone seq →
  (∃ count : ℕ, count = f n) := 
sorry

end max_three_term_arithm_progressions_l535_535134


namespace stream_current_rate_l535_535793

theorem stream_current_rate (r w : ℝ) : (
  (18 / (r + w) + 6 = 18 / (r - w)) ∧ 
  (18 / (3 * r + w) + 2 = 18 / (3 * r - w))
) → w = 6 := 
by {
  sorry
}

end stream_current_rate_l535_535793


namespace wen_family_theater_cost_l535_535397

theorem wen_family_theater_cost :
  (∃ (regular senior child : ℝ),
    senior = 7.50 ∧
    regular = senior / 0.80 ∧
    child = regular * 0.60 ∧
    let total_before_discount := 2 * senior + 2 * regular + 2 * child in
    total_before_discount * 0.90 = 40.50) := 
sorry

end wen_family_theater_cost_l535_535397


namespace distance_from_X_to_BC_l535_535277

-- Define point A, B, C, D
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (s, 0)
def C : ℝ × ℝ := (s, s)
def D : ℝ × ℝ := (0, s)

-- Define quarter circles
def arc_centered_at_A (p : ℝ × ℝ) : Prop := p.1 ^ 2 + p.2 ^ 2 = s ^ 2
def arc_centered_at_D (p : ℝ × ℝ) : Prop := p.1 ^ 2 + (p.2 - s) ^ 2 = s ^ 2

-- Define X as the intersection of the quarter circles
def X : ℝ × ℝ := (s * sqrt 3 / 2, s / 2)

-- The distance from X to side BC, which is the line y = s
def distance_to_BC (p : ℝ × ℝ) : ℝ := s - p.2

theorem distance_from_X_to_BC (s : ℝ) (h1 : arc_centered_at_A X) (h2 : arc_centered_at_D X) :
  distance_to_BC X = s / 2 := by
  sorry

end distance_from_X_to_BC_l535_535277


namespace darts_final_score_is_600_l535_535970

def bullseye_points : ℕ := 50

def first_dart_points (bullseye : ℕ) : ℕ := 3 * bullseye

def second_dart_points : ℕ := 0

def third_dart_points (bullseye : ℕ) : ℕ := bullseye / 2

def fourth_dart_points (bullseye : ℕ) : ℕ := 2 * bullseye

def total_points_before_fifth (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def fifth_dart_points (bullseye : ℕ) (previous_total : ℕ) : ℕ :=
  bullseye + previous_total

def final_score (d1 d2 d3 d4 d5 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4 + d5

theorem darts_final_score_is_600 :
  final_score
    (first_dart_points bullseye_points)
    second_dart_points
    (third_dart_points bullseye_points)
    (fourth_dart_points bullseye_points)
    (fifth_dart_points bullseye_points (total_points_before_fifth
      (first_dart_points bullseye_points)
      second_dart_points
      (third_dart_points bullseye_points)
      (fourth_dart_points bullseye_points))) = 600 :=
  sorry

end darts_final_score_is_600_l535_535970


namespace city_divided_1014_districts_l535_535616

-- Define the Squares and Streets
variable (Square : Type) [Fintype Square]
variable (Street : Square → Square → Prop)

-- Conditions: Each square has exactly two one-way streets departing from it
axiom (H1: ∀ s : Square, ∃! t u : Square, (Street s t ∧ Street s u))

-- Define the target property to prove
def can_be_divided_into_1014_districts (Square : Type) [Fintype Square] (Street : Square → Square → Prop) : Prop :=
  ∃ (districts : Square → Fin 1014),
    (∀ s1 s2 : Square, districts s1 = districts s2 → (¬ Street s1 s2 ∧ ¬ Street s2 s1)) ∧
    (∀ d1 d2 : Fin 1014, ∀ s1 s2 : Square, districts s1 = d1 → districts s2 = d2 → s1 ≠ s2 →
      Street s1 s2 → ¬ Street s2 s1)

-- Main theorem to prove
theorem city_divided_1014_districts
  (Square : Type) [Fintype Square]
  (Street : Square → Square → Prop)
  (H1 : ∀ s : Square, ∃! t u : Square, (Street s t ∧ Street s u)) :
  can_be_divided_into_1014_districts Square Street :=
sorry

end city_divided_1014_districts_l535_535616


namespace james_calories_burned_per_week_l535_535293

variable (calories_per_hour_walking : ℕ) (calories_per_hour_dancing : ℕ)
variable (sessions_per_day : ℕ) (hours_per_session : ℕ)
variable (days_per_week : ℕ)

def total_calories_burned_per_week := 
  2 * calories_per_hour_walking = calories_per_hour_dancing → 
  sessions_per_day = 2 →
  hours_per_session = 0.5 →
  days_per_week = 4 →
  calories_per_hour_walking = 300 →
  calories_per_hour_dancing * (sessions_per_day * hours_per_session : ℕ) * days_per_week = 2400

theorem james_calories_burned_per_week :
  total_calories_burned_per_week 300 600 2 0.5 4 := by
  sorry

end james_calories_burned_per_week_l535_535293


namespace odd_function_l535_535903

def f (x : ℝ) : ℝ :=
  if x > 0 then
    x^3 + x + 1
  else if x < 0 then
    x^3 + x - 1
  else 
    0

theorem odd_function (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
begin
  intros x h,
  have h_neg : f (-x) = -f x, from h_odd x,
  have h_nonpos : f x = -f (-x), {
    rw [h_neg, h_pos (-x)],
    simp at *,
    sorry
  },
  sorry
end

end odd_function_l535_535903


namespace distribution_schemes_l535_535684

theorem distribution_schemes (classes students : ℕ) 
  (h_classes : classes = 3) 
  (h_students : students = 5) 
  (h_max_students_per_class : ∀ c : ℕ, c < classes → c ≤ 2) : 
  distribution_schemes classes students = 90 := 
by
  sorry

end distribution_schemes_l535_535684


namespace original_price_l535_535321

theorem original_price (P : ℝ) (H : 5 * 12 = 60 ∧ 60 * 1.85 + 9 = 60 * P) : P = 2 :=
by 
  have h0 : 5 * 12 = 60 := by linarith
  have h1 : 60 * 1.85 + 9 = 60 * P := H.2
  linarith

end original_price_l535_535321


namespace table_tennis_possible_outcomes_l535_535748

-- Two people are playing a table tennis match. The first to win 3 games wins the match.
-- The match continues until a winner is determined.
-- Considering all possible outcomes (different numbers of wins and losses for each player are considered different outcomes),
-- prove that there are a total of 30 possible outcomes.

theorem table_tennis_possible_outcomes : 
  ∃ total_outcomes : ℕ, total_outcomes = 30 := 
by
  -- We need to prove that the total number of possible outcomes is 30
  sorry

end table_tennis_possible_outcomes_l535_535748


namespace odd_function_solution_l535_535901

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_solution (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
by
  sorry

end odd_function_solution_l535_535901


namespace prove_f_2002_l535_535667

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x)

lemma f_iterative_property (x : ℝ) : f (f (f x)) = x :=
  by {
      -- Original proof simplification would be here
      sorry
  }

lemma periodic_application_to_2002 (x : ℝ) (n : ℕ) (h_periodic : ∀ x, f (f (f x)) = x) : 
  f^[n] x = if n % 3 = 0 then x else if n % 3 = 1 then f x else f (f x) :=
  by {
      -- Original proof simplification would be here
      sorry
  }

theorem prove_f_2002 (x : ℝ) : f^[2002] 2002 = 2001 / 2002 :=
  begin
    have h : f (2002) = -1 / 2001 := by {
      -- Original computation simplification would be here
      sorry
    },
    have h2 : f (-1 / 2001) = 2001 / 2002 := by {
      -- Original computation simplification would be here
      sorry
    },
    show f^[2002] 2002 = 2001 / 2002,
    {
      rw ← periodic_application_to_2002 2002 2002 f_iterative_property,
      simp [nat.mod_eq_of_lt (nat.lt_add_one_of_le (by decide))]
    } 
  end

end prove_f_2002_l535_535667


namespace orange_bows_l535_535264

theorem orange_bows (n : ℕ) 
  (purple_fraction : n * 1 / 4)
  (yellow_fraction : n * 1 / 3)
  (orange_fraction : n * 1 / 12)
  (black_bows : 40)
  (black_fraction : n * 1 / 3 = 40) : 
  n * 1 / 12 = 10 :=
begin
  sorry
end

end orange_bows_l535_535264


namespace sin_70_given_sin_10_l535_535186

theorem sin_70_given_sin_10 (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by 
  sorry

end sin_70_given_sin_10_l535_535186


namespace length_of_BA_is_sqrt_557_l535_535854

-- Define the given conditions
def AD : ℝ := 6
def DC : ℝ := 11
def CB : ℝ := 6
def AC : ℝ := 14

-- Define the theorem statement
theorem length_of_BA_is_sqrt_557 (x : ℝ) (H1 : AD = 6) (H2 : DC = 11) (H3 : CB = 6) (H4 : AC = 14) :
  x = Real.sqrt 557 :=
  sorry

end length_of_BA_is_sqrt_557_l535_535854


namespace matt_current_age_is_65_l535_535044

variable (matt_age james_age : ℕ)

def james_current_age := 30
def james_age_in_5_years := james_current_age + 5
def matt_age_in_5_years := 2 * james_age_in_5_years
def matt_current_age := matt_age_in_5_years - 5

theorem matt_current_age_is_65 : matt_current_age = 65 := 
by
  -- sorry is here to skip the proof.
  sorry

end matt_current_age_is_65_l535_535044


namespace triangle_side_length_l535_535289

theorem triangle_side_length
  (A B C : ℝ) (a b c : ℝ)
  (h1 : ∠C = 2 * ∠A)
  (h2 : a = 36)
  (h3 : c = 64) :
  b = 52 / 27 := 
sorry

end triangle_side_length_l535_535289


namespace intersection_of_sets_l535_535238

def set_M : Set ℝ := { x | x >= 2 }
def set_N : Set ℝ := { x | -1 <= x ∧ x <= 3 }
def set_intersection : Set ℝ := { x | 2 <= x ∧ x <= 3 }

theorem intersection_of_sets : (set_M ∩ set_N) = set_intersection := by
  sorry

end intersection_of_sets_l535_535238


namespace third_consecutive_odd_integers_is_fifteen_l535_535776

theorem third_consecutive_odd_integers_is_fifteen :
  ∃ x : ℤ, (x % 2 = 1 ∧ (x + 2) % 2 = 1 ∧ (x + 4) % 2 = 1) ∧ (x + 2 + (x + 4) = x + 17) → (x + 4 = 15) :=
by
  sorry

end third_consecutive_odd_integers_is_fifteen_l535_535776


namespace terminating_fraction_count_l535_535150

theorem terminating_fraction_count :
  (∃ n_values : Finset ℕ, (∀ n ∈ n_values, 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = k * 49)) ∧ n_values.card = 10) :=
by
  -- Placeholder for the proof, does not contribute to the conditions-direct definitions.
  sorry

end terminating_fraction_count_l535_535150


namespace shaded_area_after_50_iterations_l535_535686

variable (BM MI : ℝ) (n : ℕ)

noncomputable def area_triangle (BM MI : ℝ) : ℝ :=
  (1 / 2) * BM * MI

noncomputable def total_shaded_area (initial_area : ℝ) (n : ℕ) : ℝ :=
  let a := initial_area / 4
  let r := 1 / 4
  a * (1 - r^n) / (1 - r)

theorem shaded_area_after_50_iterations : BM = 12 → MI = 12 → total_shaded_area (area_triangle BM MI) 50 = 24 :=
by
  intros hBM hMI
  rw [hBM, hMI]
  have h_initial_area : area_triangle 12 12 = 72 := rfl
  simp [h_initial_area, area_triangle, total_shaded_area]
  sorry

end shaded_area_after_50_iterations_l535_535686


namespace derivative_cos_is_odd_function_derivative_exp_is_not_odd_nor_even_derivative_ln_is_not_odd_nor_even_derivative_a_pow_x_is_not_odd_nor_even_l535_535813

-- Definitions of the functions and their derivatives
def f_cos (x : ℝ) : ℝ := cos x
def derivative_f_cos (x : ℝ) : ℝ := -sin x

def f_exp (x : ℝ) : ℝ := exp x
def derivative_f_exp (x : ℝ) : ℝ := exp x

def f_ln (x : ℝ) : ℝ := log x
def derivative_f_ln (x : ℝ) : ℝ := 1 / x

def f_a_pow_x (a x : ℝ) : ℝ := a^x
def derivative_f_a_pow_x (a x : ℝ) : ℝ := a^x * log a

-- Statement that the derivative of cos x is an odd function
theorem derivative_cos_is_odd_function : 
  (∀ x : ℝ, derivative_f_cos (-x) = -derivative_f_cos x) := by 
  sorry

-- Placeholder for additional assertions about other functions
theorem derivative_exp_is_not_odd_nor_even : 
  (∀ x : ℝ, derivative_f_exp (-x) ≠ -derivative_f_exp x) ∧ 
  (∀ x : ℝ, derivative_f_exp (-x) ≠ derivative_f_exp x) := by 
  sorry

theorem derivative_ln_is_not_odd_nor_even : 
  (∀ x : ℝ, x > 0 → derivative_f_ln (-x) ≠ -derivative_f_ln x) ∧ 
  (∀ x : ℝ, x > 0 → derivative_f_ln (-x) ≠ derivative_f_ln x) := by 
  sorry

theorem derivative_a_pow_x_is_not_odd_nor_even (a : ℝ) : 
  (∀ x : ℝ, derivative_f_a_pow_x a (-x) ≠ -derivative_f_a_pow_x a x) ∧ 
  (∀ x : ℝ, derivative_f_a_pow_x a (-x) ≠ derivative_f_a_pow_x a x) := by 
  sorry

end derivative_cos_is_odd_function_derivative_exp_is_not_odd_nor_even_derivative_ln_is_not_odd_nor_even_derivative_a_pow_x_is_not_odd_nor_even_l535_535813


namespace number_of_zeros_of_g_l535_535911

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else Real.log2 x

noncomputable def g (x : ℝ) : ℝ :=
  f (f x) + 1

def zero_count := (4 : ℕ)

theorem number_of_zeros_of_g : 
  (∃ x1 x2 x3 x4 : ℝ, g x1 = 0 ∧ g x2 = 0 ∧ g x3 = 0 ∧ g x4 = 0 ∧ 
  (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4)) →
  ∀ n : ℕ, (n = 4) := 
  sorry

end number_of_zeros_of_g_l535_535911


namespace probability_product_divisible_by_14_l535_535520

theorem probability_product_divisible_by_14 :
  let faces := {1, 2, 3, 4, 5, 6}
  let fair_die_roll := (faces × faces).uniform_probability
  let sum_roll (r : faces × faces) : ℕ := r.1 + r.2
  let three_rolls := vector (sum_roll fair_die_roll) 3
  let condition_14 (scores : vector ℕ 3) : Prop := scores.data.product % 14 = 0
  ∑ r in three_rolls, if condition_14 r then fair_die_roll r else 0 = 1/3
:= sorry

end probability_product_divisible_by_14_l535_535520


namespace john_weekly_earnings_l535_535648

/-- John takes 3 days off of streaming per week. 
    John streams for 4 hours at a time on the days he does stream.
    John makes $10 an hour.
    Prove that John makes $160 a week. -/

theorem john_weekly_earnings (days_off : ℕ) (hours_per_day : ℕ) (wage_per_hour : ℕ) 
  (h_days_off : days_off = 3) (h_hours_per_day : hours_per_day = 4) 
  (h_wage_per_hour : wage_per_hour = 10) : 
  7 - days_off * hours_per_day * wage_per_hour = 160 := by
  sorry

end john_weekly_earnings_l535_535648


namespace meeting_point_2015th_l535_535175

-- Define the parameters of the problem
variables (A B C D : Type)
variables (x y t : ℝ) -- Speeds and the initial time delay

-- State the problem as a theorem
theorem meeting_point_2015th (start_times_differ : t > 0)
                            (speeds_pos : x > 0 ∧ y > 0)
                            (pattern : ∀ n : ℕ, (odd n → (meeting_point n = C)) ∧ (even n → (meeting_point n = D)))
                            (n = 2015) :
  meeting_point n = C :=
  sorry

end meeting_point_2015th_l535_535175


namespace min_value_of_m_l535_535311

noncomputable def g (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2
noncomputable def h (x : ℝ) := (Real.exp (-x) - Real.exp x) / 2

theorem min_value_of_m (m : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → m * g x + h x ≥ 0) → m ≥ (Real.exp 2 - 1) / (Real.exp 2 + 1) :=
by
  intro h
  have key_ineq : ∀ x, -1 ≤ x ∧ x ≤ 1 → m ≥ 1 - 2 / (Real.exp (2 * x) + 1) := sorry
  sorry

end min_value_of_m_l535_535311


namespace ln_3_approx_l535_535524

noncomputable def maclaurin_series_ln : ℕ → ℝ
| 0       := 1
| (n + 1) := if n % 2 = 0 then 0 else (-1 : ℝ)^((n+1)/2) * 2 / (n + 1)

theorem ln_3_approx :
  abs (∑ n in Finset.range 10, maclaurin_series_ln n * (1 / 2)^(n + 1) - (∑ n in Finset.range 10, (maclaurin_series_ln n * (1 / 2)^(n + 1)) * 2) - ln 3 ) < 0.001 :=
sorry

end ln_3_approx_l535_535524


namespace player_F_always_wins_l535_535749

theorem player_F_always_wins (cards : Fin 9 → ℕ)
  (unique_cards : ∀ i j, i ≠ j → cards i ≠ cards j) :
  let F_positions := [0, 2];
      S_positions := [0, 6];
      t1 := cards 0 + cards 8;
      t2 := cards 1 + cards 7;
      t3 := cards 2 + cards 6;
      t4 := cards 3 + cards 5 in
  — Strategy and game play omitted for brevity —
  (sum (F_positions.map cards) > sum (S_positions.map cards)) :=
sorry

end player_F_always_wins_l535_535749


namespace ratio_of_a_to_b_l535_535630

variables (AB DC : Points)

axiom parallel : parallel AB DC
axiom AB_eq_b : AB = b
axiom CD_eq_a : CD = a
axiom a_lt_b : a < b

def area_trapezium_ABCD (S : ℝ) : Prop := area ABCD = S
def area_triangle_BOC (S : ℝ) : Prop := area (BOC) = 2 * S / 9

theorem ratio_of_a_to_b (S : ℝ) :
  area_trapezium_ABCD S →
  area_triangle_BOC S →
  a / b = 1 / 2 :=
by
  intros
  -- Proof goes here
  sorry

end ratio_of_a_to_b_l535_535630


namespace num_elements_in_set_S_l535_535655

theorem num_elements_in_set_S :
  let S := {n : ℕ | n > 1 ∧ IsRepeatingDecimal (1 / n) 10}
  card S = 23 :=
by
  have h₁ : 10 ^ 10 - 1 = (11 : ℕ) * (3 : ℕ) ^ 2 * (101 : ℕ) * (10201 : ℕ) := by sorry,
  have h₂ : 24 = ∑ d in divisors (10 ^ 10 - 1), 1 := by sorry,
  have h₃ : S = {n : ℕ | n ∣ (10 ^ 10 - 1) ∧ n > 1} := by sorry,
  exact sorry

end num_elements_in_set_S_l535_535655


namespace ellipse_equation_line_PM_l535_535577

open Real

/-- Given an ellipse with the equation x²/a² + y²/b² = 1 where a > b > 0, 
and the eccentricity e = 1/2, along with the point (√3, -√3/2) on the ellipse,
prove the equation of the ellipse is x²/4 + y²/3 = 1. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (e : ℝ) (he : e = 1 / 2) (p : ℝ × ℝ) (hp : p = (√3, -√3 / 2)) 
    (h : (p.1 ^ 2) / a^2 + (p.2 ^ 2) / b^2 = 1) : 
    a^2 = 4 ∧ b^2 = 3 :=
  sorry

/-- Given an ellipse with focus at F and the line l through F intersecting the ellipse
at points M(x1, y1) and N(x2, y2). If point P is symmetric to N with respect to the x-axis,
prove that line PM always passes through the fixed point (4,0). -/
theorem line_PM (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (p : ℝ × ℝ) (hp : p = (√3, -√3 / 2)) 
    (h : (p.1 ^ 2) / a^2 + (p.2 ^ 2) / b^2 = 1) 
    (x1 y1 x2 y2 : ℝ) 
    (hx : y2 = -y2)
    (focus : ℝ × ℝ) 
    (hf : focus = (√(a^2 - b^2), 0)) 
    (line : ℝ → ℝ) 
    (hline : ∀ y, line y = y * (y1 - y2) / (x1 - x2) + y1) :
    ∃ fx fy : ℝ, fx = 4 ∧ fy = 0 :=
  sorry

end ellipse_equation_line_PM_l535_535577


namespace triangle_area_half_l535_535014

theorem triangle_area_half (AB AC BC : ℝ) (h₁ : AB = 8) (h₂ : AC = BC) (h₃ : AC * AC = AB * AB / 2) (h₄ : AC = BC) : 
  (1 / 2) * (1 / 2 * AB * AB) = 16 :=
  by
  sorry

end triangle_area_half_l535_535014


namespace min_x_y_l535_535894

theorem min_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) : x + y ≥ 4 :=
by
  sorry

end min_x_y_l535_535894


namespace equal_length_day_l535_535619

-- Conditions
def bulrush_length (n : ℕ) : ℝ :=
  3 * (1 - (1 / 2) ^ n) / (1 - 1 / 2)

def reed_length (n : ℕ) : ℝ :=
  (2 ^ n - 1) / (2 - 1)

noncomputable def lg (x : ℝ) : ℝ :=
  if x = 2 then 0.3010
  else if x = 3 then 0.4771
  else sorry  -- Assume definitions for specific logarithm values provided

-- Proof statement
theorem equal_length_day : 
  let n := 1 + lg 3 / lg 2 in 
  Float.round n = 2.6 := 
by
  sorry

end equal_length_day_l535_535619


namespace rancher_no_cows_l535_535437

theorem rancher_no_cows (s c : ℕ) (h1 : 30 * s + 31 * c = 1200) 
  (h2 : 15 ≤ s) (h3 : s ≤ 35) : c = 0 :=
by
  sorry

end rancher_no_cows_l535_535437


namespace sum_of_powers_of_two_l535_535099

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 :=
by
  sorry

end sum_of_powers_of_two_l535_535099


namespace meeting_2015th_at_C_l535_535162

-- Conditions Definitions
variable (A B C D P : Type)
variable (x y t : ℝ)  -- speeds and starting time difference
variable (mw cyclist : ℝ → ℝ)  -- paths of motorist and cyclist

-- Proof statement
theorem meeting_2015th_at_C 
(Given_meeting_pattern: ∀ n : ℕ, odd n → (mw (n * (x + y))) = C):
  (mw (2015 * (x + y))) = C := 
by 
  sorry  -- Proof omitted

end meeting_2015th_at_C_l535_535162


namespace simplify_expression_l535_535349

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : 
  2 * (1 - (2 * (1 - (1 + (2 * (1 - x)))))) = 8 * x - 10 := 
by sorry

end simplify_expression_l535_535349


namespace meeting_point_2015_l535_535180

/-- 
A motorist starts at point A, and a cyclist starts at point B. They travel towards each other and 
meet for the first time at point C. After meeting, they turn around and travel back to their starting 
points and continue this pattern of meeting, turning around, and traveling back to their starting points. 
Prove that their 2015th meeting point is at point C.
-/
theorem meeting_point_2015 
  (A B C D : Type) 
  (x y t : ℕ)
  (odd_meeting : ∀ n : ℕ, (2 * n + 1) % 2 = 1) : 
  ∃ n, (n = 2015) → odd_meeting n = 1 → (n % 2 = 1 → (C = "C")) := 
sorry

end meeting_point_2015_l535_535180


namespace number_of_ways_people_can_stand_l535_535005

def people := ["A", "B", "C"]
def steps : ℕ := 7

-- Each step can accommodate at most 2 people and positions on the same step are indistinguishable
def max_people_per_step : ℕ := 2

theorem number_of_ways_people_can_stand : 
  ∃ (ways : ℕ), ways = 336 ∧ 
  (ways = (number_of_ways_to_place_people_on_steps people steps max_people_per_step)) :=
sorry

end number_of_ways_people_can_stand_l535_535005


namespace max_length_XY_l535_535803

theorem max_length_XY (p : ℝ) : 
  let s := p / 2 in
  let a_max := p / 4 in
  let XY := (a_max * (p - 2 * a_max)) / p in
  XY = p / 8 :=
by
  -- placeholder for the proof
  sorry

end max_length_XY_l535_535803


namespace pascal_triangle_fifth_number_twentieth_row_l535_535111

theorem pascal_triangle_fifth_number_twentieth_row : 
  (Nat.choose 20 4) = 4845 :=
by
  sorry

end pascal_triangle_fifth_number_twentieth_row_l535_535111


namespace difficult_minus_easy_l535_535822

variables (S1 S2 S3 S12 S13 S23 S123 : Nat)

def total_problems_solved : Prop := S1 + S2 + S3 + S12 + S13 + S23 + S123 = 100
def anton_solved : Prop := S1 + S12 + S13 + S123 = 60
def artem_solved : Prop := S2 + S12 + S23 + S123 = 60
def vera_solved : Prop := S3 + S13 + S23 + S123 = 60

def number_of_difficult_problems := S1 + S2 + S3
def number_of_easy_problems := S123

theorem difficult_minus_easy :
  total_problems_solved ∧ anton_solved ∧ artem_solved ∧ vera_solved →
  number_of_difficult_problems - number_of_easy_problems = 20 :=
sorry

end difficult_minus_easy_l535_535822


namespace meeting_point_2015_is_C_l535_535170

-- Given definitions based on conditions
variable (x y : ℝ) -- Speeds of the motorcycle and the cyclist
variable (A B C D : Point) -- Points on segment AB
variable (meetings : ℕ → Point) -- Function representing the meeting point sequence

-- Conditions stating the alternating meeting pattern
axiom start_at_A (n : ℕ) : meetings (2 * n + 1) = C
axiom start_at_B (n : ℕ) : meetings (2 * n + 2) = D

-- The theorem statement to be proved
theorem meeting_point_2015_is_C : meetings 2015 = C := sorry

end meeting_point_2015_is_C_l535_535170


namespace no_primes_in_range_l535_535861

theorem no_primes_in_range (n : ℕ) (h : 2 < n) :
  ∀ p ∈ finset.filter nat.prime (finset.Ico (n.factorial + 3) (n.factorial + 2 * n + 1)), false :=
by
  sorry

end no_primes_in_range_l535_535861


namespace intersecting_diagonals_probability_l535_535773

def probability_of_intersecting_diagonals_inside_dodecagon : ℚ :=
  let total_points := 12
  let total_segments := (total_points.choose 2)
  let sides := 12
  let diagonals := total_segments - sides
  let ways_to_choose_2_diagonals := (diagonals.choose 2)
  let ways_to_choose_4_points := (total_points.choose 4)
  let probability := (ways_to_choose_4_points : ℚ) / (ways_to_choose_2_diagonals : ℚ)
  probability

theorem intersecting_diagonals_probability (H : probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477) : 
  probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477 :=
  by
  sorry

end intersecting_diagonals_probability_l535_535773


namespace ratio_BC_CD_l535_535693

-- Conditions definitions
variables {A B C D F : Type}
variables (quadrilateral : Type)
variables (right_angle_C : quadrilateral)
variables (right_angle_D : quadrilateral)
variables (triangle_BCD_sim_DAB : quadrilateral)
variables (BC_CD_inequality : quadrilateral)
variables (point_F_interior : quadrilateral)
variables (triangle_BCD_sim_DCF : quadrilateral)
variables (area_BAF_13_times_DCF : quadrilateral)

-- Main theorem statement
theorem ratio_BC_CD : BC / CD = 2 + sqrt 3 :=
by {
  sorry
}

end ratio_BC_CD_l535_535693


namespace selectedParticipants_correct_l535_535000

-- Define the random number table portion used in the problem
def randomNumTable := [
  [12, 56, 85, 99, 26, 96, 96, 68, 27, 31, 05, 03, 72, 93, 15, 57, 12, 10, 14, 21, 88, 26, 49, 81, 76]
]

-- Define the conditions
def totalStudents := 247
def selectedStudentsCount := 4
def startingIndexRow := 4
def startingIndexCol := 9
def startingNumber := randomNumTable[0][8]

-- Define the expected selected participants' numbers
def expectedParticipants := [050, 121, 014, 218]

-- The Lean statement that needs to be proved
theorem selectedParticipants_correct : expectedParticipants = [050, 121, 014, 218] := by
  sorry

end selectedParticipants_correct_l535_535000


namespace number_of_ordered_pairs_eq_one_l535_535599

theorem number_of_ordered_pairs_eq_one :
  ∃ (p n : ℕ), p.prime ∧ n > 0 ∧ (1 + p)^n = 1 + p * n + n^p ∧ 
  (∀ (p' n' : ℕ), p'.prime ∧ n' > 0 ∧ (1 + p')^n' = 1 + p' * n' + n'^p' → (p' = p ∧ n' = n)) :=
sorry

end number_of_ordered_pairs_eq_one_l535_535599


namespace find_nonSunday_date_l535_535969

-- Define what it means for a date to never be a Sunday in any month of a certain year
def neverSunday (d : ℕ) : Prop :=
  ∀m y, ¬ (Date.mk y m d).dayOfWeek = sunday

-- Define the problem statement as a theorem
theorem find_nonSunday_date : 
  neverSunday 31 := 
sorry

end find_nonSunday_date_l535_535969


namespace max_min_f_l535_535583

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem max_min_f : 
  ∃ max_val min_val,
  max_val = f 0 ∧ min_val = f (Real.pi / 2) ∧
  ∀ x ∈ set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ max_val ∧ min_val ≤ f x :=
by
  -- Proof required here
  sorry

end max_min_f_l535_535583


namespace matt_age_three_years_ago_l535_535042

theorem matt_age_three_years_ago (james_age_three_years_ago : ℕ) (age_difference : ℕ) (future_factor : ℕ) :
  james_age_three_years_ago = 27 →
  age_difference = 3 →
  future_factor = 2 →
  ∃ matt_age_now : ℕ,
  james_age_now: ℕ,
    james_age_now = james_age_three_years_ago + age_difference ∧
    (matt_age_now + 5) = future_factor * (james_age_now + 5) ∧
    matt_age_now = 65 :=
by
  sorry

end matt_age_three_years_ago_l535_535042


namespace total_erasers_l535_535643

def cases : ℕ := 7
def boxes_per_case : ℕ := 12
def erasers_per_box : ℕ := 25

theorem total_erasers : cases * boxes_per_case * erasers_per_box = 2100 := by
  sorry

end total_erasers_l535_535643


namespace correct_statement_l535_535435

theorem correct_statement : 
  (∀ x : ℝ, (x < 0 → x^2 > x)) ∧
  (¬ ∀ x : ℝ, (x^2 > 0 → x > 0)) ∧
  (¬ ∀ x : ℝ, (x^2 > x → x > 0)) ∧
  (¬ ∀ x : ℝ, (x^2 > x → x < 0)) ∧
  (¬ ∀ x : ℝ, (x < 1 → x^2 < x)) :=
by
  sorry

end correct_statement_l535_535435


namespace intersection_complement_is_correct_l535_535591

open Set

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2 * x - 3}
noncomputable def N : Set ℝ := {x | -5 ≤ x ∧ x ≤ 2}

theorem intersection_complement_is_correct :
  M ∩ (Set.univ \ N) = {x : ℝ | x > 2} := 
begin
  sorry
end

end intersection_complement_is_correct_l535_535591


namespace boat_speed_in_still_water_l535_535977

namespace BoatSpeed

variables (V_b V_s : ℝ)

def condition1 : Prop := V_b + V_s = 15
def condition2 : Prop := V_b - V_s = 5

theorem boat_speed_in_still_water (h1 : condition1 V_b V_s) (h2 : condition2 V_b V_s) : V_b = 10 :=
by
  sorry

end BoatSpeed

end boat_speed_in_still_water_l535_535977


namespace sequence_properties_l535_535560

def sequence (n : ℕ) : ℚ 
| 0 => 2 
| (n + 1) => 2 - 1 / (sequence n)

theorem sequence_properties : 
  (∀ n : ℕ, n > 0 → 
    (1 : ℚ) / (sequence n - 1) = 1 + (1 : ℚ) / (sequence (n - 1) - 1))) ∧
  (∀ n : ℕ, n > 0 → sequence n > 1) ∧
  ¬ (∃ n : ℕ, n > 0 ∧ sequence n = 2 * sequence (2 * n)) ∧
  (∀ ε > 0, ∃ n₀ : ℕ, ∀ n : ℕ, n > n₀ → |(sequence (n + 1)) - (sequence n)| < ε) := 
sorry

end sequence_properties_l535_535560


namespace tan_diff_eqn_l535_535893

theorem tan_diff_eqn (α : ℝ) (h : Real.tan α = 2) : Real.tan (α - 3 * Real.pi / 4) = -3 := 
by 
  sorry

end tan_diff_eqn_l535_535893


namespace multiply_add_square_l535_535830

theorem multiply_add_square : 15 * 28 + 42 * 15 + 15^2 = 1275 :=
by
  sorry

end multiply_add_square_l535_535830


namespace sum_extrema_values_l535_535315

noncomputable def g (x : ℝ) : ℝ := 
  |x - 5| + |x - 7| - |2x - 12| + |3x - 21|

theorem sum_extrema_values : ∃ (max_val min_val : ℝ), max_val = 10 ∧ min_val = 1 ∧ 
  (∀ x, 5 ≤ x ∧ x ≤ 10 → min_val ≤ g x ∧ g x ≤ max_val) ∧ 
  (max_val + min_val = 11) :=
sorry

end sum_extrema_values_l535_535315


namespace minimize_acme_cost_l535_535478

theorem minimize_acme_cost (x : ℕ) : 75 + 12 * x < 16 * x → x = 19 :=
by
  intro h
  sorry

end minimize_acme_cost_l535_535478


namespace coeff_x2_expansion_l535_535629

theorem coeff_x2_expansion : 
  let expr := (x^2 + (1 / x^2) - 2) ^ 4
  in coeff (expr.expand) (Monomial.mk 2) = -56 :=
by sorry

end coeff_x2_expansion_l535_535629


namespace arrangement_of_books_l535_535073

theorem arrangement_of_books : 
  (∑ s in (finset.range 10).powerset.filter (λ s, s.card = 4), 1) = 126 :=
sorry

end arrangement_of_books_l535_535073


namespace closed_curve_area_is_correct_l535_535360

noncomputable def enclosed_area (s : ℝ) : ℝ :=
  let sector_area := 3 * π
  let pentagon_area := (9 / 4) * Real.sqrt(5 * (5 + 2 * Real.sqrt 5))
  sector_area + pentagon_area

theorem closed_curve_area_is_correct :
  enclosed_area 3 = (9 / 4) * Real.sqrt(5 * (5 + 2 * Real.sqrt 5)) + 3 * π :=
by sorry

end closed_curve_area_is_correct_l535_535360


namespace find_function_l535_535912

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x+1) = x^2 - 1) : ∀ x : ℝ, f(x) = x^2 - 2x :=
sorry

end find_function_l535_535912


namespace total_amount_received_is_1465_l535_535994

-- defining the conditions
def principal_1 : ℝ := 4000
def principal_2 : ℝ := 8200
def rate_1 : ℝ := 0.11
def rate_2 : ℝ := rate_1 + 0.015

-- defining the interest from each account
def interest_1 := principal_1 * rate_1
def interest_2 := principal_2 * rate_2

-- stating the total amount received
def total_received := interest_1 + interest_2

-- proving the total amount received
theorem total_amount_received_is_1465 : total_received = 1465 := by
  -- proof goes here
  sorry

end total_amount_received_is_1465_l535_535994


namespace central_angle_radians_l535_535558

-- Definitions for the problem
variables (r θ : ℝ)
def arc_length (r θ : ℝ) := r * θ
def sector_area (r θ : ℝ) := (1/2) * r^2 * θ

-- Conditions
axiom arc_length_eq_four : arc_length r θ = 4
axiom sector_area_eq_four : sector_area r θ = 4

-- Proof statement
theorem central_angle_radians : θ = 2 :=
by
  sorry

end central_angle_radians_l535_535558


namespace hundred_a_plus_b_l535_535671

theorem hundred_a_plus_b (a b : ℝ)
  (h1 : (∀ x : ℝ, (x = -a ∨ x = -b ∨ x = -10) ∧ x ≠ -4 → (x+a) * (x+b) * (x+10) = 0)
  (h2 : (∀ x : ℝ, x = -4 → (x+2*a) * (x+4) * (x+7) = 0 ∧ x ≠ -b ∧ x ≠ -10))
  : 100 * a + b = 207 :=
sorry

end hundred_a_plus_b_l535_535671


namespace find_k_l535_535113

theorem find_k (x k : ℝ) :
  (∀ x, x ∈ Set.Ioo (-4 : ℝ) 3 ↔ x * (x^2 - 9) < k) → k = 0 :=
  by
  sorry

end find_k_l535_535113


namespace inequality_proof_problem_l535_535317

open Real

theorem inequality_proof_problem {m n : ℕ} {a : ℕ → ℝ} (h_pos : ∀ i, 0 < a i) (h_m_pos : 0 < m) (h_n_pos : 0 < n)  (h_n_m_diff : 3 ≤ n - m) :
  ∏ i in finset.range (n - m), (a i ^ n - a i ^ m + n - m) ≥ (finset.sum (finset.range (n - m)) (λ i, a i)) ^ (n - m) :=
sorry

end inequality_proof_problem_l535_535317


namespace slope_angle_y_eq_neg1_l535_535722

theorem slope_angle_y_eq_neg1 : (∃ line : ℝ → ℝ, ∀ y: ℝ, line y = -1 → ∃ θ : ℝ, θ = 0) :=
by
  -- Sorry is used to skip the proof.
  sorry

end slope_angle_y_eq_neg1_l535_535722


namespace polynomial_p_l535_535364

variable {a b c : ℝ}

theorem polynomial_p (a b c : ℝ) : 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) * 2 :=
by
  sorry

end polynomial_p_l535_535364


namespace number_of_routes_10_minutes_l535_535980

def M (n : ℕ) : ℕ × ℕ := 
if n = 0 then (1, 0) 
else let (mA, mB) := M (n-1) in (mB, mA + mB)

theorem number_of_routes_10_minutes : (M 10).fst = 34 :=
sorry

end number_of_routes_10_minutes_l535_535980


namespace f_monotonic_intervals_f_max_min_on_interval_l535_535920

-- Define the function
def f (x : ℝ) : ℝ := 3 * x ^ 3 - 9 * x + 5

-- Statement for Part 1: Monotonic Intervals
theorem f_monotonic_intervals : 
  (∀ x, x < -1 → monotone_increasing f x) ∧
  (∀ x, x > 1 → monotone_increasing f x) ∧
  (∀ x, -1 < x ∧ x < 1 → monotone_decreasing f x) := 
sorry

-- Statement for Part 2: Maximum and Minimum Values on [-3, 3]
theorem f_max_min_on_interval : 
  is_max_on f 3 (-3, 3) ∧ 
  ∀ x, x ∈ [-3, 3] → f x ≤ 59 ∧
  is_min_on f (-3) (-3, 3) ∧ 
  ∀ x, x ∈ [-3, 3] → f x ≥ -49 := 
sorry

end f_monotonic_intervals_f_max_min_on_interval_l535_535920


namespace λ_value_correct_l535_535576

-- Define the conditions and the required proof statement

structure Point where
  x : ℝ
  y : ℝ

def d : Point := ⟨1, -1⟩

def A : Point := ⟨1, 1⟩
def B : Point := ⟨-1, 8⟩

def A1 : Point -- Foot of perpendicular from A to line l
def B1 : Point -- Foot of perpendicular from B to line l

def vector_sub (p1 p2 : Point) : Point := 
  ⟨p1.x - p2.x, p1.y - p2.y⟩

noncomputable def λ : ℝ :=
  let AB := vector_sub B A
  (AB.x * d.x + AB.y * d.y) / (d.x * d.x + d.y * d.y)

theorem λ_value_correct : λ = -9/2 := by
  -- Proof goes here
  sorry

end λ_value_correct_l535_535576


namespace average_price_per_book_is_13_45_l535_535694

noncomputable def total_amount_spent := 600 + 240 + 180 + 325
noncomputable def total_number_of_books := 40 + 20 + 15 + 25
noncomputable def average_price_per_book := total_amount_spent / total_number_of_books

theorem average_price_per_book_is_13_45 :
  average_price_per_book = 13.45 := by
  sorry

end average_price_per_book_is_13_45_l535_535694


namespace two_integers_divide_2_pow_96_minus_1_l535_535216

theorem two_integers_divide_2_pow_96_minus_1 : 
  ∃ a b : ℕ, (60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧ a ≠ b ∧ a ∣ (2^96 - 1) ∧ b ∣ (2^96 - 1) ∧ a = 63 ∧ b = 65) := 
sorry

end two_integers_divide_2_pow_96_minus_1_l535_535216


namespace billiard_trajectory_forms_regular_1998_gon_l535_535781

variables {A1 A2 ... A1998 : ℝ} 

noncomputable def is_regular_1998_gon (points : list ℝ) : Prop :=
  ∀ (i j k : ℕ), is_regular_polygon points 1998 i j k

noncomputable def angle_of_reflection (α β : ℝ) : Prop :=
  α = β

theorem billiard_trajectory_forms_regular_1998_gon 
  (A1 A2 ... A1998 : ℝ) 
  (midpoint : ℝ) 
  (trajectory : list ℝ) 
  (h1 : is_polygon trajectory)
  (h2 : starts_at_midpoint_of_A1A2 trajectory midpoint)
  (h3 : bounces_off_regular_sides trajectory [A2, A3, ... , A1998, A1])
  (h4 : ∀ i, angle_of_reflection (incident_angle trajectory i) (reflected_angle trajectory i))
  (h5 : returns_to_start trajectory) :
  is_regular_1998_gon trajectory := 
sorry

end billiard_trajectory_forms_regular_1998_gon_l535_535781


namespace sum_4501st_4052nd_digit_sequence_l535_535072

theorem sum_4501st_4052nd_digit_sequence : 
  let sequence := λ (n : ℕ), 
    ((List.range (n + 1)).map (λ m, List.repeat m m)).join
  in (sequence 4501) + (sequence 4052) = 9 := by
  sorry

end sum_4501st_4052nd_digit_sequence_l535_535072


namespace evaluate_expression_l535_535118

theorem evaluate_expression : 
  (196 * (1 / 17 - 1 / 21) + 361 * (1 / 21 - 1 / 13) + 529 * (1 / 13 - 1 / 17)) /
    (14 * (1 / 17 - 1 / 21) + 19 * (1 / 21 - 1 / 13) + 23 * (1 / 13 - 1 / 17)) = 56 :=
by
  sorry

end evaluate_expression_l535_535118


namespace time_for_250_m_distance_l535_535010

-- Defining Velocity functions for the two bodies.
def v1 (t : ℝ) : ℝ := 6 * t^2 + 4 * t
def v2 (t : ℝ) : ℝ := 4 * t

-- Defining the displacement functions as integrals of the velocity functions.
def s1 (t : ℝ) : ℝ := ∫ (x : ℝ) in 0..t, v1 x
def s2 (t : ℝ) : ℝ := ∫ (x : ℝ) in 0..t, v2 x

-- The goal is to prove that the time when the distance between s1 and s2 is 250 meters is 5 seconds.
theorem time_for_250_m_distance : ∃ t : ℝ, s1 t - s2 t = 250 ∧ t = 5 :=
by
  -- proof steps would go here
  sorry

end time_for_250_m_distance_l535_535010


namespace digit_at_tens_place_is_eight_l535_535407

-- Definitions for conditions
def digits : List ℕ := [1, 2, 3, 5, 8]

def is_valid_number (n : ℕ) : Prop :=
  let ds := (Nat.digits 10 n)
  ds.length = 5 ∧
  ∀ d, d ∈ ds → d ∈ digits ∧ list.count ds d = 1 ∧
  (n % 2 = 0)

-- The smallest valid even number using the digits 1, 2, 3, 5, and 8
noncomputable def smallest_even_number : ℕ :=
  List.minimum' (List.filter is_valid_number (List.permutations digits)).map (λ ds, List.foldl (λ n d, n * 10 + d) 0 ds)

-- The proof goal in Lean
theorem digit_at_tens_place_is_eight :
  Nat.digits 10 smallest_even_number !! 1 = some 8 :=
sorry

end digit_at_tens_place_is_eight_l535_535407


namespace rainfall_second_week_l535_535519

noncomputable def total_rainfall : ℝ := 45
noncomputable def first_week_rainfall (x : ℝ) : ℝ := x
noncomputable def second_week_rainfall (x : ℝ) : ℝ := 1.5 * x
noncomputable def third_week_rainfall (x : ℝ) : ℝ := 0.5 * (1.5 * x)

noncomputable def rainfall_equation (x : ℝ) : Prop :=
  first_week_rainfall x + second_week_rainfall x + third_week_rainfall x = total_rainfall

theorem rainfall_second_week (x : ℝ) (h : rainfall_equation x) : second_week_rainfall x ≈ 20.77 :=
begin
  sorry
end

end rainfall_second_week_l535_535519


namespace complex_number_parts_l535_535718

theorem complex_number_parts (z : ℂ) (i_sq : ℂ) :
  (i_sq = -1) → (z = i_sq + complex.i) → ((z.re = -1) ∧ (z.im = 1)) :=
by
  intro h1 h2
  rw h1 at h2
  rw [Complex.add_re, Complex.add_im, Complex.i_re, Complex.i_im] at h2
  exact ⟨by simp [h2], by simp [h2]⟩

end complex_number_parts_l535_535718


namespace monotonic_intervals_max_min_on_interval_l535_535915

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

theorem monotonic_intervals :
  (∀ x, x < -1 → f' x > 0) ∧ (∀ x, x > 1 → f' x > 0) ∧ (∀ x, -1 < x ∧ x < 1 → f' x < 0) :=
sorry

theorem max_min_on_interval :
  ∃ x_max x_min, 
  x_max ∈ Set.Icc (-3 : ℝ) 3 ∧ x_min ∈ Set.Icc (-3 : ℝ) 3 ∧ 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧ 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x_min ≤ f x) ∧ 
  f x_max = 59 ∧ f x_min = -49 :=
sorry

end monotonic_intervals_max_min_on_interval_l535_535915


namespace proof_problem_l535_535052

variables {n : ℕ} {r d : ℝ} {P : ℝ} 
variables {a : Fin n → ℝ} {XA : Fin n → ℝ}

-- Conditions
def conditions (r d P : ℝ) (a : Fin n → ℝ) (XA : Fin n → ℝ) :=
  (∀ i, XA i = d^2 + r^2 + 2 * (d * r)) ∧  -- This combines XAi² = d² + r² + 2(XO · OAi) assuming the dot product is zero
  (P = ∑ i, a i) ∧                           -- Perimeter is the sum of the side lengths a_i
  (∑ i, a i * r = 0)                        -- Summation condition with weighted vectors of side lengths, simplified for the zero result

-- Theorem to prove 
theorem proof_problem 
  (h : conditions r d P a XA) : 
  ∑ i, a i * (XA i)^2 = P * (r^2 + d^2) :=
by sorry

end proof_problem_l535_535052


namespace work_completion_l535_535080

theorem work_completion (A B : ℝ → ℝ) (h1 : ∀ t, A t = B t) (h3 : A 4 + B 4 = 1) : B 1 = 1/2 :=
by {
  sorry
}

end work_completion_l535_535080


namespace billy_age_is_45_l535_535097

variable (Billy_age Joe_age : ℕ)

-- Given conditions
def condition1 := Billy_age = 3 * Joe_age
def condition2 := Billy_age + Joe_age = 60
def condition3 := Billy_age > 60 / 2

-- Prove Billy's age is 45
theorem billy_age_is_45 (h1 : condition1 Billy_age Joe_age) (h2 : condition2 Billy_age Joe_age) (h3 : condition3 Billy_age) : Billy_age = 45 :=
by
  sorry

end billy_age_is_45_l535_535097


namespace number_of_twos_l535_535017

noncomputable def sequence_property (n : ℕ) : ℕ :=
  ⌊(Real.sqrt 2 - 1) * n + 1 - Real.sqrt 2 / 2⌋

theorem number_of_twos (n : ℕ) : 
  ∀ S : ℕ → ℕ, 
  (S 0 = 1) ∧ 
  (∀ m, S m = 2 → S (m + 1) ≠ 2) ∧ 
  (∀ m, (S m = 1 ∧ S (m + 1) = 1) → S (m + 2) ≠ 1) → 
  (number_of_twos_n_elements S n = sequence_property n) :=
sorry

end number_of_twos_l535_535017


namespace solve_for_x_l535_535251

theorem solve_for_x (x : ℤ) (h : (-1) * 2 * x * 4 = 24) : x = -3 := by
  sorry

end solve_for_x_l535_535251


namespace max_product_geom_seq_l535_535621

theorem max_product_geom_seq (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (a_pos : ∀ n, 0 < a n)
  (h_a2 : a 2 = 8)
  (h_a4 : 16 * (a 4)^2 = a 1 * a 5) :
  ∃ n, (∀ m, m ≠ 3 → (a 1)^m * q^((m * (m - 1)) / 2) < (a 1)^3 * q^((3 * 2) / 2)) :=
sorry

end max_product_geom_seq_l535_535621


namespace tom_finishes_in_6_years_l535_535739

variable (BS_time : ℕ) (PhD_time : ℕ) (fraction : ℚ)

def total_program_time := BS_time + PhD_time

noncomputable def tom_completion_time := total_program_time * fraction

theorem tom_finishes_in_6_years :
  BS_time = 3 ∧ PhD_time = 5 ∧ fraction = (3/4 : ℚ) → tom_completion_time BS_time PhD_time fraction = 6 := by
  sorry

end tom_finishes_in_6_years_l535_535739


namespace volume_cone_270_sector_div_pi_eq_1125_sqrt7_l535_535053

noncomputable def volume_cone_divided_by_pi (r h: ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h / π

theorem volume_cone_270_sector_div_pi_eq_1125_sqrt7
  (R : ℝ) (theta : ℝ) (r : ℝ) (s : ℝ) (h : ℝ)
  (h0 : R = 20)
  (h1 : theta = 270)
  (h2 : s = R)
  (h3 : r = 15) 
  (h4 : h = 5 * Real.sqrt 7) :
  volume_cone_divided_by_pi r h = 1125 * Real.sqrt 7 :=
by
  sorry

end volume_cone_270_sector_div_pi_eq_1125_sqrt7_l535_535053


namespace mark_age_proof_l535_535399

-- Define the variables for Mark's and Aaron's ages
variables (M A X : ℕ)

-- Given conditions
def mark_age := 28
def condition1 := M - 3 = 3 * (A - 3) + 1
def condition2 := M + 4 = 2 * (A + 4) + X

theorem mark_age_proof :
  mark_age = 28 → -- Mark is 28
  (∃ A, condition1) → -- Three years ago, Mark's age was 1 year more than thrice Aaron's age
  (∃ X, condition2) → -- Four years from now, Mark's age will be some years more than twice Aaron's age
  X = 2 := -- The targeted proof statement
by
  sorry -- Proof placeholder

end mark_age_proof_l535_535399


namespace least_fly_distance_l535_535473

theorem least_fly_distance 
    (r : ℝ) (h : ℝ) 
    (a : ℝ) (b : ℝ) 
    (l : ℝ)
    (h₁ : r = 600)
    (h₂ : h = 200 * Real.sqrt 7)
    (h₃ : a = 125)
    (h₄ : b = 375 * Real.sqrt 2)
    (h₅ : l = Real.sqrt (r^2 + h^2)) :
    Real.sqrt (a^2 + b^2) = 125 * Real.sqrt 19 :=
by
  rw [←h₁, ←h₂, ←h₃, ←h₄, ←h₅]
  srw sorry

end least_fly_distance_l535_535473


namespace regular_decagon_area_l535_535205

theorem regular_decagon_area (JKLMPQRST : set ℝ) (is_regular_decagon : is_regular_decagon JKLMPQRST)
    (A J C : ℝ) (hAJ : A ≠ J) (hJC : J ≠ C) (hAJ1 : dist A J = 1) (hJC1 : dist J C = 1) :
    ∃ area : ℝ, area = 3.5 + 4 * Real.sqrt 2 :=
by sorry

end regular_decagon_area_l535_535205


namespace inequality_proof_l535_535869

theorem inequality_proof (a b c : ℝ) (hab : a > b) : a * |c| ≥ b * |c| := by
  sorry

end inequality_proof_l535_535869


namespace find_k_l535_535936

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (2, -3)

def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k (k : ℝ) :
  is_perpendicular (k • vector_a - 2 • vector_b) vector_a ↔ k = -1 :=
sorry

end find_k_l535_535936


namespace arrangements_teachers_next_to_each_other_l535_535609

noncomputable def count_arrangements :
  {S T : ℕ} → (students : Fin S) → (teachers : Fin T) → ℕ
  | 4, 3, students, teachers => sorry

theorem arrangements_teachers_next_to_each_other :
  count_arrangements (Fin 4) (Fin 3) = 2880 :=
sorry

end arrangements_teachers_next_to_each_other_l535_535609


namespace james_pitbull_count_l535_535292

-- Defining the conditions
def husky_count : ℕ := 5
def retriever_count : ℕ := 4
def retriever_pups_per_retriever (husky_pups_per_husky : ℕ) : ℕ := husky_pups_per_husky + 2
def husky_pups := husky_count * 3
def retriever_pups := retriever_count * (retriever_pups_per_retriever 3)
def pitbull_pups (P : ℕ) : ℕ := P * 3
def total_pups (P : ℕ) : ℕ := husky_pups + retriever_pups + pitbull_pups P
def total_adults (P : ℕ) : ℕ := husky_count + retriever_count + P
def condition (P : ℕ) : Prop := total_pups P = total_adults P + 30

-- The proof objective
theorem james_pitbull_count : ∃ P : ℕ, condition P → P = 2 := by
  sorry

end james_pitbull_count_l535_535292


namespace egyptians_panamanians_l535_535003

-- Given: n + m = 12 and (n(n-1))/2 + (m(m-1))/2 = 31 and n > m
-- Prove: n = 7 and m = 5

theorem egyptians_panamanians (n m : ℕ) (h1 : n + m = 12) (h2 : n > m) 
(h3 : n * (n - 1) / 2 + m * (m - 1) / 2 = 31) :
  n = 7 ∧ m = 5 := 
by
  sorry

end egyptians_panamanians_l535_535003


namespace height_min_surface_area_l535_535676

def height_of_box (x : ℝ) : ℝ := x + 4

def surface_area (x : ℝ) : ℝ := 2 * x^2 + 4 * x * (x + 4)

theorem height_min_surface_area :
  ∀ x : ℝ, surface_area x ≥ 150 → x ≥ 5 → height_of_box x = 9 :=
by
  intros x h1 h2
  sorry

end height_min_surface_area_l535_535676


namespace find_q_symmetric_seq_3_fold_exists_symmetric_sequence_with_Pm_fold_bound_on_am_l535_535211

-- Definition of a symmetric sequence
def m_fold_symmetric (m : ℕ) (seq : ℕ → ℝ) : Prop :=
  ∀ d ∈ finset.range (2 * m - 1), seq (2 * m - d) = seq d

-- Question (1)
theorem find_q_symmetric_seq_3_fold (q : ℝ) :
  (∀ n, n > 0 → ∃ (seq : ℕ → ℝ), seq n = q ^ n ∧ m_fold_symmetric 3 seq) →
  q = 0 ∨ q = 1 ∨ q = -1 :=
sorry

-- Question (2)
theorem exists_symmetric_sequence_with_Pm_fold (P : ℕ) (h_P : P > 0) :
  ∃ (seq : ℕ → ℝ), ∀ m : ℕ, m > 0 →
    m_fold_symmetric (P * m) seq ∧ (finset.univ.image seq).card = P + 1 :=
sorry

-- Question (3)
theorem bound_on_am (a : ℕ → ℕ) (x : ℕ → ℝ) (k : ℕ)
  (h_a : ∀ i, a i > 0) (h_incr : ∀ i j, i < j → a i < a j) 
  (h_symmetric : ∀ m, m > 0 → m_fold_symmetric (a m) x ∧ 
    (finset.univ.image x).card ≤ k):
  ∀ m : ℕ, m > 0 → a m ≤ 2^(m-1) * (k-1) + 1 :=
sorry

end find_q_symmetric_seq_3_fold_exists_symmetric_sequence_with_Pm_fold_bound_on_am_l535_535211


namespace value_of_x_plus_y_squared_l535_535193

theorem value_of_x_plus_y_squared (x y : ℝ) 
  (h₁ : x^2 + y^2 = 20) 
  (h₂ : x * y = 6) : 
  (x + y)^2 = 32 :=
by
  sorry

end value_of_x_plus_y_squared_l535_535193


namespace num_int_values_n_terminated_l535_535153

theorem num_int_values_n_terminated (N : ℕ) (hN1 : 1 ≤ N) (hN2 : N ≤ 500) :
  ∃ n : ℕ, n = 10 ∧ ∀ k, 0 ≤ k → k < n → ∃ (m : ℕ), N = m * 49 :=
sorry

end num_int_values_n_terminated_l535_535153


namespace problem_statement_l535_535899

open Real

noncomputable def pressure_at_height (p0 : ℝ) (k : ℝ) (h : ℝ) : ℝ :=
  p0 * exp (-k * h)

variable (p0 : ℝ) (k : ℝ := 10⁻⁴) (elevation_levels : List (ℝ × ℝ))

theorem problem_statement 
  (h1 : ℝ) (p1 : ℝ) (h2_min h2_max h3_min h3_max: ℝ) (p2 p3 : ℝ) :
  (4000 ≤ h1) →
  (h2_min ≤ h2_max ∧ h2_min = 1000 ∧ h2_max = 2000) →
  (h3_min ≤ h3_max ∧ h3_min = 200 ∧ h3_max = 1000) →
  (ln p0 - ln p1 = k * h1) →
  (ln p0 - ln p2 = k * h2_min) →
  (ln p0 - ln p3 = k * h3_max) →
  (p1 ≤ p0 / exp 0.4) ∧
  (p0 > p3) ∧
  (p3 ≤ exp 0.18 * p2) :=
by
  sorry

end problem_statement_l535_535899


namespace john_weekly_earnings_l535_535647

/-- John takes 3 days off of streaming per week. 
    John streams for 4 hours at a time on the days he does stream.
    John makes $10 an hour.
    Prove that John makes $160 a week. -/

theorem john_weekly_earnings (days_off : ℕ) (hours_per_day : ℕ) (wage_per_hour : ℕ) 
  (h_days_off : days_off = 3) (h_hours_per_day : hours_per_day = 4) 
  (h_wage_per_hour : wage_per_hour = 10) : 
  7 - days_off * hours_per_day * wage_per_hour = 160 := by
  sorry

end john_weekly_earnings_l535_535647


namespace polynomial_root_fraction_l535_535103

theorem polynomial_root_fraction (p q r s : ℝ) (h : p ≠ 0) 
    (h1 : p * 4^3 + q * 4^2 + r * 4 + s = 0)
    (h2 : p * (-3)^3 + q * (-3)^2 + r * (-3) + s = 0) : 
    (q + r) / p = -13 :=
by
  sorry

end polynomial_root_fraction_l535_535103


namespace front_wheel_marker_tangency_point_l535_535004

-- Definition of given constants and conditions
def circle_radius : ℝ := 30
def front_wheel_radius : ℝ := 15
def back_wheel_radius : ℝ := 5

-- Circumference calculations
def circle_circumference : ℝ := 2 * Real.pi * circle_radius
def front_wheel_circumference : ℝ := 2 * Real.pi * front_wheel_radius

-- Mathematical translation of the problem's solution
theorem front_wheel_marker_tangency_point :
  ∃ position_on_pathway : ℝ, 
  position_on_pathway = (circle_radius : ℝ) / (front_wheel_radius : ℝ) * Real.pi :=
sorry

end front_wheel_marker_tangency_point_l535_535004


namespace anya_kolya_apples_l535_535823

theorem anya_kolya_apples (A K : ℕ) (h1 : A = (K * 100) / (A + K)) (h2 : K = (A * 100) / (A + K)) : A = 50 ∧ K = 50 :=
sorry

end anya_kolya_apples_l535_535823


namespace quadratic_solution_identity_l535_535379

theorem quadratic_solution_identity {a b c : ℝ} (h1 : a ≠ 0) (h2 : a * (1 : ℝ)^2 + b * (1 : ℝ) + c = 0) : 
  a + b + c = 0 :=
sorry

end quadratic_solution_identity_l535_535379


namespace value_of_m_monotonicity_find_a_and_r_l535_535217

noncomputable def basic_conditions (a : ℝ) (m : ℝ) :=
  a > 1 ∧ a ≠ 1 ∧ f(x) = log a ((1 - m * x) / (x - 1))

theorem value_of_m (a : ℝ) : (basic_conditions a m) → m = -1 :=
by
  sorry

theorem monotonicity (a : ℝ) (m : ℝ) : (basic_conditions a m) → (m = -1) →  ∀ x > 1, decreasing_on (λ x, log a ((1 + x) / ( x - 1))) (set.Ioi 1) :=
by
  sorry

theorem find_a_and_r (a r : ℝ) : (basic_conditions a (-1)) → range (λ x, log a ((x + 1) / (x - 1))) = Ioi 1 → a = 2 + real.sqrt 3 ∧ r = 1 :=
by
  sorry

end value_of_m_monotonicity_find_a_and_r_l535_535217


namespace intersection_right_complement_l535_535203

open Set

def A := {x : ℝ | x - 1 ≥ 0}
def B := {x : ℝ | 3 / x ≤ 1}

theorem intersection_right_complement :
  A ∩ (compl B) = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_right_complement_l535_535203


namespace length_y_rounded_is_32_l535_535696

-- Condition 1: Seven identical rectangles
def number_of_rectangles := 7

-- Condition 2: Arrangement in three rows
def middle_row_contains_3_rectangles := True
def top_and_bottom_row_contains_2_rectangles_each := True

-- Condition 3: Area of the larger rectangle is 4900
def area_ABCD := 4900

-- Given these conditions, we need to prove the length y, rounded to the nearest integer, is 32
theorem length_y_rounded_is_32 : 
  (∃ (y : ℝ), (let h := (2/3) * y in 7 * ((2/3) * y * y) = 4900) ∧ (Real.sqrt 1050).round = 32) := 
  sorry

end length_y_rounded_is_32_l535_535696


namespace false_proposition_of_quadratic_l535_535879

theorem false_proposition_of_quadratic
  (a : ℝ) (h0 : a ≠ 0)
  (h1 : ¬(5 = a * (1/2)^2 + (-a^2 - 1) * (1/2) + a))
  (h2 : (a^2 + 1) / (2 * a) > 0)
  (h3 : (0, a) = (0, x) ∧ x > 0)
  (h4 : ∀ x : ℝ, a * x^2 + (-a^2 - 1) * x + a ≤ 0) :
  false :=
sorry

end false_proposition_of_quadratic_l535_535879


namespace profit_calculation_l535_535340

def initial_outlay : ℕ := 10000
def cost_first_300_set : ℕ := 20
def cost_beyond_300_set : ℕ := 15
def price_first_400_set : ℕ := 50
def price_beyond_400_set : ℕ := 45
def total_sets : ℕ := 800
def first_300_sets : ℕ := 300
def first_400_sets : ℕ := 400

theorem profit_calculation :
  let manufacturing_cost := initial_outlay + (first_300_sets * cost_first_300_set) + ((total_sets - first_300_sets) * cost_beyond_300_set) in
  let revenue := (first_400_sets * price_first_400_set) + ((total_sets - first_400_sets) * price_beyond_400_set) in
  let profit := revenue - manufacturing_cost in
  profit = 14500 :=
by
  sorry

end profit_calculation_l535_535340


namespace quadratic_with_given_means_l535_535898

theorem quadratic_with_given_means (a b : ℝ) 
  (h1 : (a + b) / 2 = 6) 
  (h2 : real.sqrt (a * b) = 5) : 
  (∀ x : ℝ, x^2 - (a + b) * x + a * b = 0 ↔ x^2 - 12 * x + 25 = 0) :=
by 
  sorry

end quadratic_with_given_means_l535_535898


namespace two_digit_numbers_condition_l535_535750

theorem two_digit_numbers_condition :
  ∃ (x y : ℕ), x > y ∧ x < 100 ∧ y < 100 ∧ x - y = 56 ∧ (x^2 % 100) = (y^2 % 100) ∧
  ((x = 78 ∧ y = 22) ∨ (x = 22 ∧ y = 78)) :=
by sorry

end two_digit_numbers_condition_l535_535750


namespace matrix_multiplication_correct_l535_535102

def matA := Matrix.of ![![2, -3], ![1, 2]]
def matB := Matrix.of ![![4], ![-6]]
def matC := Matrix.of ![![26], ![-8]]

theorem matrix_multiplication_correct : matA.mul matB = matC := 
by sorry

end matrix_multiplication_correct_l535_535102


namespace inverse_mod_53_l535_535891

theorem inverse_mod_53 (h : 17 * 13 % 53 = 1) : 36 * 40 % 53 = 1 :=
by
  -- Given condition: 17 * 13 % 53 = 1
  -- Derived condition: (-17) * -13 % 53 = 1 which is equivalent to 17 * 13 % 53 = 1
  -- So we need to find: 36 * x % 53 = 1 where x = -13 % 53 => x = 40
  sorry

end inverse_mod_53_l535_535891


namespace polygon_sides_of_set_T_l535_535672

noncomputable def T (b : ℝ) : set (ℝ × ℝ) :=
  {p | b ≤ p.1 ∧ p.1 ≤ 3 * b ∧ b ≤ p.2 ∧ p.2 <= 3 * b ∧ p.1 + p.2 >= 2 * b ∧ p.2 <= p.1 / 2 + b ∧ p.1 <= p.2 / 2 + b}

theorem polygon_sides_of_set_T (b : ℝ) (hb : 0 < b) : 
  ∃ (n : ℕ), n = 3 ∧ ∃ (vertices : ℕ → ℝ × ℝ),
    (∀ i, i < n → vertices i ∈ T b) ∧ 
    set.pairwise (disjoint on (λ i, segment (vertices i) (vertices (i+1) % n))) := 
sorry

end polygon_sides_of_set_T_l535_535672


namespace polygon_area_correct_l535_535414

-- Define the coordinates of the vertices
def vertex1 := (2, 1)
def vertex2 := (4, 3)
def vertex3 := (6, 1)
def vertex4 := (4, 6)

-- Define a function to calculate the area using the Shoelace Theorem
noncomputable def shoelace_area (vertices : List (ℕ × ℕ)) : ℚ :=
  let xys := vertices ++ [vertices.head!]
  let sum1 := (xys.zip (xys.tail!)).map (fun ((x1, y1), (x2, y2)) => x1 * y2)
  let sum2 := (xys.zip (xys.tail!)).map (fun ((x1, y1), (x2, y2)) => y1 * x2)
  (sum1.sum - sum2.sum : ℚ) / 2

-- Instantiate the specific vertices
def polygon := [vertex1, vertex2, vertex3, vertex4]

-- The theorem statement
theorem polygon_area_correct : shoelace_area polygon = 6 := by
  sorry

end polygon_area_correct_l535_535414


namespace impossible_to_have_equal_checkers_l535_535047

theorem impossible_to_have_equal_checkers (L : set (fin 3 × fin 3) -> Prop) : 
  (∀ n, ∃ k, ∀ (c : fin 3 × fin 3), (n ≠ 0 ∧ ∀ move : set (fin 3 × fin 3), L move → (∀ cell ∈ move, c = k)) → 
  ∀ (c1 c2 : fin 3 × fin 3), c1 ≠ c2 → n = 0) := sorry

end impossible_to_have_equal_checkers_l535_535047


namespace monotonic_intervals_max_min_on_interval_l535_535917

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

theorem monotonic_intervals :
  (∀ x, x < -1 → f' x > 0) ∧ (∀ x, x > 1 → f' x > 0) ∧ (∀ x, -1 < x ∧ x < 1 → f' x < 0) :=
sorry

theorem max_min_on_interval :
  ∃ x_max x_min, 
  x_max ∈ Set.Icc (-3 : ℝ) 3 ∧ x_min ∈ Set.Icc (-3 : ℝ) 3 ∧ 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧ 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x_min ≤ f x) ∧ 
  f x_max = 59 ∧ f x_min = -49 :=
sorry

end monotonic_intervals_max_min_on_interval_l535_535917


namespace find_d_l535_535371

noncomputable def vector_d : ℝ × ℝ :=
  if h : ∃ (d : ℝ × ℝ),
    let ⟨a, b⟩ := d in
    (a^2 + (5 * a / 6)^2 = 1) ∧
    (λ x y : ℝ, y = (5 * x - 7) / 6) (3 + a) (1 + 5 * a / 6) = true 
  then classical.some h
  else (0, 0)

theorem find_d :
  vector_d = (sqrt (36 / 61), 5 * sqrt (36 / 61) / 6) ∨
  vector_d = (-sqrt (36 / 61), -5 * sqrt (36 / 61) / 6) :=
sorry

end find_d_l535_535371


namespace number_of_valid_pairs_l535_535948

def valid_pairs : List (ℕ × ℕ) := 
[(7, 21), (8, 12), (9, 9), (12, 6)]

theorem number_of_valid_pairs :
  (count_pairs (valid_pairs, λ (m, n), (6 / m) + (3 / n) = 1)) = 4 := 
sorry

end number_of_valid_pairs_l535_535948


namespace find_hypotenuse_l535_535266

-- Let PQR be a right triangle with the right angle at PQR.
-- Let PQ be one leg of the triangle and QS be the angle bisector from Q to PR.
variables (P Q R S : Point)
variables (PQ QR PR QS : ℝ)

-- Definitions from the conditions
def is_right_triangle (P Q R : Point) : Prop :=
  ∠ P Q R = π / 2

def length_PQ := 12
def length_QS := 6 * sqrt 5

-- Statement to be proved
theorem find_hypotenuse (h1 : is_right_triangle P Q R)
                        (h2 : PQ = length_PQ)
                        (h3 : QS = length_QS) :
  QR = 20 :=
begin
  sorry
end

end find_hypotenuse_l535_535266


namespace max_parking_cars_l535_535062

-- Definitions and conditions
def grid_size : ℕ := 7
def total_cells : ℕ := grid_size * grid_size
def gate_cells_removed : ℕ := total_cells - 1
def max_cars : ℕ := 28

-- The structure of the parking grid and conditions for car placement
structure ParkingGrid where
  width : ℕ
  height : ℕ
  gate : (ℕ × ℕ)
  car_positions : List (ℕ × ℕ)
  valid_parking : List (ℕ × ℕ) → Prop

-- Define the parking grid with appropriate conditions
def flower_city_parking : ParkingGrid := {
  width := grid_size,
  height := grid_size,
  gate := (0, 0), -- The corner cell acting as a gate
  car_positions := [],
  valid_parking := λ cars, 
    cars.length ≤ gate_cells_removed ∧
    (∀ car ∈ cars, ∃ path : List (ℕ × ℕ), 
      path.head = car ∧ path.last = flower_city_parking.gate ∧
      (∀ (a b : ℕ × ℕ), (a, b) ∈ path.zip path.tail →
        ((a.1 = b.1 ∧ abs (a.2 - b.2) = 1) ∨ (a.2 = b.2 ∧ abs (a.1 - b.1) = 1)) ∧
        (b ∉ cars ∨ b = flower_city_parking.gate)))
}

theorem max_parking_cars : ∃ pg : ParkingGrid, 
  pg.width = grid_size ∧ 
  pg.height = grid_size ∧ 
  pg.gate = (0, 0) ∧ 
  pg.valid_parking pg.car_positions ∧ 
  pg.car_positions.length = max_cars := by
  -- Proof to be completed
  sorry

end max_parking_cars_l535_535062


namespace find_x_l535_535204

def vec_a : ℝ × ℝ × ℝ := (-2, 1, 3)
def vec_b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

theorem find_x (x : ℝ) (h : (-2) * 1 + 1 * x + 3 * (-1) = 0) : x = 5 :=
by
  sorry

end find_x_l535_535204


namespace min_value_frac_ineq_l535_535257

theorem min_value_frac_ineq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  ∃ x, x = (1/a) + (2/b) ∧ x ≥ 9 :=
sorry

end min_value_frac_ineq_l535_535257


namespace parabola_solution_l535_535469

theorem parabola_solution (a b : ℝ) : 
  (∃ y : ℝ, y = 2^2 + 2 * a + b ∧ y = 20) ∧ 
  (∃ y : ℝ, y = (-2)^2 + (-2) * a + b ∧ y = 0) ∧ 
  b = (0^2 + 0 * a + b) → 
  a = 5 ∧ b = 6 := 
by {
  sorry
}

end parabola_solution_l535_535469


namespace shorten_to_sixth_power_l535_535411

theorem shorten_to_sixth_power (x n m p q r : ℕ) (h1 : x > 1000000)
  (h2 : x / 10 = n^2)
  (h3 : n^2 / 10 = m^3)
  (h4 : m^3 / 10 = p^4)
  (h5 : p^4 / 10 = q^5) :
  q^5 / 10 = r^6 :=
sorry

end shorten_to_sixth_power_l535_535411


namespace nearest_sum_f_eq_2004_l535_535789
-- Import all necessary libraries in one go

-- Define the given conditions and the problem statement
noncomputable def f (x : ℝ) : ℝ := sorry

theorem nearest_sum_f_eq_2004 :
  (∀ x : ℝ, x ≠ 0 → 3 * f x + f (1 / x) = 15 * x + 8) →
  (nearest_int (sum { x : ℝ | f x = 2004 }) = 356) :=
begin
  sorry
end

end nearest_sum_f_eq_2004_l535_535789


namespace cyclic_quadrilateral_l535_535691

theorem cyclic_quadrilateral (A B C D : Point) 
    (angle_A : angle A B D + angle C A D = 180)
    (angle_B : angle B C A + angle D B C = 180) : 
    ∃ O : Circle, Circumscribed O A B C D :=
by sorry

end cyclic_quadrilateral_l535_535691


namespace min_area_of_right_triangle_l535_535799

theorem min_area_of_right_triangle : 
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧
                  nat.gcd a b = 1 ∧
                  nat.gcd a c = 1 ∧
                  nat.gcd b c = 1 ∧
                  ∃ n : ℕ, n * n = (a * b * (a + b + c) / a + b + c) ∧
                  (∀ x y z : ℕ, x^2 + y^2 = z^2 ∧
                                 nat.gcd x y = 1 ∧
                                 nat.gcd x z = 1 ∧
                                 nat.gcd y z = 1 ∧
                                 ∃ m : ℕ, m * m = (x * y * (x + y + z) / x + y + z)  
                                 → (x * y / 2 ≥ 24))

end min_area_of_right_triangle_l535_535799


namespace at_least_three_entries_remain_infinitely_many_choices_for_three_entries_remain_l535_535299

-- Part (a)
theorem at_least_three_entries_remain
  (n : ℕ) (x : Fin n → ℕ)
  (H_distinct : ∀ i j : Fin n, i ≠ j → x i ≠ x j)
  (Hx1 : x 0 = 1) : 
  ∃ (remaining : Fin n → ℕ), set.size ({remaining k | k < n} : set ℕ) ≥ 3 := 
sorry

-- Part (b)
theorem infinitely_many_choices_for_three_entries_remain :
  ∃ f : ℕ → ℕ, (∀ n : ℕ, set.size ({f k | k < n} : set ℕ) = 3) ∧
  set.infinite ({n : ℕ | ∀ x : Fin n → ℕ, (Hx1 : x 0 = 1) → 
  set.size ({remaining k | k < n} : set ℕ) = 3}) := 
sorry

end at_least_three_entries_remain_infinitely_many_choices_for_three_entries_remain_l535_535299


namespace largest_abs_value_among_four_l535_535084

theorem largest_abs_value_among_four : ∀ (a b c d : ℤ), (a = 4) → (b = -5) → (c = 0) → (d = -1) → abs b > abs a ∧ abs b > abs c ∧ abs b > abs d :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  simp
  sorry

end largest_abs_value_among_four_l535_535084


namespace characters_per_day_l535_535402

-- Definitions based on conditions
def chars_total_older : ℕ := 8000
def chars_total_younger : ℕ := 6000
def chars_per_day_diff : ℕ := 100

-- Define the main theorem
theorem characters_per_day (x : ℕ) :
  chars_total_older / x = chars_total_younger / (x - chars_per_day_diff) := 
sorry

end characters_per_day_l535_535402


namespace number_of_ones_and_twos_in_N_l535_535058

def is_natural (n : ℕ) : Prop := n ≥ 0

def is_hundred_digit_number (N : ℕ → ℕ) : Prop :=
  ∑ i in range 100, (N i = 1 ∨ N i = 2)

def even_number_between_twos (N : ℕ → ℕ) : Prop :=
  ∀ i j, i < j → N i = 2 → N j = 2 → ∑ k in (range (i+1)).filter (λ k, k < j), (N k = 1 ∨ N k = 2) % 2 = 0

def divisible_by_three (N : ℕ → ℕ) : Prop :=
  (∑ i in range 100, N i) % 3 = 0

def number_of_ones_and_twos (N : ℕ → ℕ) : ℕ × ℕ :=
  (∑ i in range 100, if N i = 1 then 1 else 0, ∑ i in range 100, if N i = 2 then 1 else 0)

theorem number_of_ones_and_twos_in_N (N : ℕ → ℕ) (h⁄1 : is_hundred_digit_number N)
  (h⁄2 : even_number_between_twos N) (h⁄3 : divisible_by_three N) : number_of_ones_and_twos N = (98, 2) :=
sorry

end number_of_ones_and_twos_in_N_l535_535058


namespace measure_angle_C_perimeter_triangle_l535_535269

-- Definitions
variable (a b c : ℝ)
variable (A B C : ℝ)  -- angles in radians

-- Acute triangle with specific relation
variable (h1 : 0 < A ∧ A < π/2)
variable (h2 : 0 < B ∧ B < π/2)
variable (h3 : 0 < C ∧ C < π/2)
variable (h4 : a = c*sin A*2/sqrt 3)  -- Given condition

-- Part 1: Prove the measure of angle C is π/3
theorem measure_angle_C (h4 : a = 2 * c * sin A / sqrt 3): C = π/3 :=
sorry

-- Additional conditions for Part 2
variable (h5 : c = sqrt 7)
variable (h6 : a * b = 6)

-- Part 2: Prove the perimeter of the triangle
theorem perimeter_triangle (h4 : a = 2 * c * sin A / sqrt 3)
                           (h5 : c = sqrt 7)
                           (h6 : a * b = 6) :
  a + b + c = 5 + sqrt 7 :=
sorry

end measure_angle_C_perimeter_triangle_l535_535269


namespace inequalities_hold_l535_535512

theorem inequalities_hold 
  (a b c x y z : ℝ) 
  (hx : x^2 < a) 
  (hy : y^2 < b) 
  (hz : z^2 < c) :
  xy + yz + zx < a + b + c ∧ 
  x^4 + y^4 + z^4 < a^2 + b^2 + c^2 ∧ 
  x^3y^3z^3 < abc := 
by
  sorry

end inequalities_hold_l535_535512


namespace alice_sum_2004_incorrect_alice_sum_2005_incorrect_bob_sum_1396_incorrect_l535_535409

/- Defining the problem conditions -/
def card_pair_sum (n : ℕ) : ℕ := 2 * n + 1 -- Sum of two consecutive integers

/- a) Proof that Alice's sum of 2004 is incorrect -/
theorem alice_sum_2004_incorrect (cards : ℕ → ℕ) (h : ∃ (selected : fin 21 → ℕ), 
  (∑ i, cards (selected i)) = 2004) :
  false :=
sorry

/- b) Proof that Alice's sum of 2005 is incorrect -/
theorem alice_sum_2005_incorrect (cards : ℕ → ℕ) (h : ∃ (selected : fin 21 → ℕ), 
  (∑ i, cards (selected i)) = 2005) :
  false :=
sorry

/- c) Proof that Bob's sum of 1396 is incorrect given Alice's valid sum of 2003 -/
theorem bob_sum_1396_incorrect (cards : ℕ → ℕ)
  (h_alice : ∃ (selected_alice : fin 21 → ℕ), 
    (∑ i, cards (selected_alice i)) = 2003)
  (h_bob : ∃ (selected_bob : fin 20 → ℕ), 
    (∑ i, cards (selected_bob i)) = 1396) :
  false :=
sorry

end alice_sum_2004_incorrect_alice_sum_2005_incorrect_bob_sum_1396_incorrect_l535_535409


namespace lowest_price_scheme_l535_535786

variable (a : ℝ) (h_pos : a > 0)

def scheme_a_price := a * 1.10 * 0.90
def scheme_b_price := a * 0.90 * 1.10
def scheme_c_price := a * 1.15 * 0.85
def scheme_d_price := a * 1.20 * 0.80

theorem lowest_price_scheme :
  scheme_d_price a h_pos < scheme_a_price a h_pos ∧
  scheme_d_price a h_pos < scheme_b_price a h_pos ∧
  scheme_d_price a h_pos < scheme_c_price a h_pos := by
  sorry

end lowest_price_scheme_l535_535786


namespace number_of_admissible_pairs_l535_535245

def is_admissible_pair (S T : set ℕ) (n : ℕ) : Prop :=
  (∀ s ∈ S, s > T.card) ∧ (∀ t ∈ T, t > S.card)

example : ∀ S T : set ℕ, S ⊆ {1, 2, ..., 10} → T ⊆ {1, 2, ..., 10} → is_admissible_pair S T 10 → (S = ∅ ∧ T = ∅) :=
by
  sorry

theorem number_of_admissible_pairs : 
  finset.card {p : finset (finset ℕ × finset ℕ) | ∃ S T : finset ℕ, p = ⟨S, T⟩ ∧ S ⊆ {1, 2, ..., 10} ∧ T ⊆ {1, 2, ..., 10} ∧ is_admissible_pair S T 10} = 1 :=
by
  sorry

end number_of_admissible_pairs_l535_535245


namespace csc_sec_arithmetic_sequence_l535_535247

theorem csc_sec_arithmetic_sequence (x d : ℝ) (h₁ : cos x = sin x + d) (h₂ : tan x = sin x + 2 * d) :
  csc x ^ 4 - sec x ^ 4 = (1 / (16 * d ^ 4)) - (1 / (1 - 4 * d ^ 2) ^ 2) :=
by
  sorry

end csc_sec_arithmetic_sequence_l535_535247


namespace number_multiplies_xz_l535_535249

theorem number_multiplies_xz (x y z w A B : ℝ) (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  A * B = 4 :=
sorry

end number_multiplies_xz_l535_535249


namespace quadrants_cos_sin_identity_l535_535563

theorem quadrants_cos_sin_identity (α : ℝ) 
  (h1 : π < α ∧ α < 2 * π)  -- α in the fourth quadrant
  (h2 : Real.cos α = 3 / 5) :
  (1 + Real.sqrt 2 * Real.cos (2 * α - π / 4)) / 
  (Real.sin (α + π / 2)) = -2 / 5 :=
by
  sorry

end quadrants_cos_sin_identity_l535_535563


namespace necessary_not_sufficient_l535_535897

variable (a l : Type) [Line a] [Line l] (β : Type) [Plane β]

axiom line_in_plane (a : Line) (β : Plane) : a ∈ β

def perpendicular_to_line (l a : Line) := ∀ (p : Point), p ∈ l → p ∈ a → ⟪p⟫ = 90 -- assuming ⟪p⟫ is the angle measure function

def perpendicular_to_plane (l : Line) (β : Plane) := ∀ (a : Line), a ∈ β → perpendicular_to_line l a

theorem necessary_not_sufficient 
    (h1: line_in_plane a β) 
    (h2: perpendicular_to_line l a) :
    ∃ β, perpendicular_to_plane l β ↔ perpendicular_to_line l a :=
sorry

end necessary_not_sufficient_l535_535897


namespace sam_drove_distance_l535_535323

theorem sam_drove_distance (m_distance : ℕ) (m_time : ℕ) (s_time : ℕ) (s_distance : ℕ)
  (m_distance_eq : m_distance = 120) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  s_distance = (m_distance / m_time) * s_time :=
by
  sorry

end sam_drove_distance_l535_535323


namespace points_H_within_triangle_A1B1C1_l535_535882

-- Define the points and heights in the given geometry
variables (A B C P H : Point)
variables (h_a h_b h_c : ℝ)
variables (S_ABC S_PBC : ℝ)

-- Conditions of the given problem
variable (condition_1 : ∀ (P : Point), ∃ (H : Point), H.isProjectionOf P onto (Plane ABC) ∧
                                          smallestHeightHeight PABC H.PH)

-- Define the regions A1B1C1
def Region_A1B1C1 (A B C : Point) (h_a h_b h_c : ℝ) : Set Point := 
  {H : Point | (dist H lineBC < h_a) ∧ (dist H lineAC < h_b) ∧ (dist H lineAB < h_c)}

-- The set of points H
def Set_of_points_H (A1 B1 C1 : Point) : Set Point :=
  {H : Point | H inside_of_triangle A1 B1 C1}

-- Theorem statement (proof not included)
theorem points_H_within_triangle_A1B1C1 : 
  ∀ (P : Point), ∃ (H : Point), Set_of_points_H (Region_A1B1C1 A B C h_a h_b h_c) := 
      sorry

end points_H_within_triangle_A1B1C1_l535_535882


namespace complete_work_together_in_days_l535_535993

-- Define the work rates for John, Rose, and Michael
def johnWorkRate : ℚ := 1 / 10
def roseWorkRate : ℚ := 1 / 40
def michaelWorkRate : ℚ := 1 / 20

-- Define the combined work rate when they work together
def combinedWorkRate : ℚ := johnWorkRate + roseWorkRate + michaelWorkRate

-- Define the total work to be done
def totalWork : ℚ := 1

-- Calculate the total number of days required to complete the work together
def totalDays : ℚ := totalWork / combinedWorkRate

-- Theorem to prove the total days is 40/7
theorem complete_work_together_in_days : totalDays = 40 / 7 :=
by
  -- Following steps would be the complete proofs if required
  rw [totalDays, totalWork, combinedWorkRate, johnWorkRate, roseWorkRate, michaelWorkRate]
  sorry

end complete_work_together_in_days_l535_535993


namespace total_amount_l535_535029

-- Definitions directly derived from the conditions in the problem
variable (you_spent friend_spent : ℕ)
variable (h1 : friend_spent = you_spent + 1)
variable (h2 : friend_spent = 8)

-- The goal is to prove that the total amount spent on lunch is $15
theorem total_amount : you_spent + friend_spent = 15 := by
  sorry

end total_amount_l535_535029


namespace cost_of_jam_l535_535847

theorem cost_of_jam (N B J H : ℕ) (h : N > 1) (cost_eq : N * (6 * B + 7 * J + 4 * H) = 462) : 7 * J * N = 462 :=
by
  sorry

end cost_of_jam_l535_535847


namespace find_number_l535_535725

-- Given conditions:
def sum_and_square (n : ℕ) : Prop := n^2 + n = 252
def is_factor (n d : ℕ) : Prop := d % n = 0

-- Equivalent proof problem statement
theorem find_number : ∃ n : ℕ, sum_and_square n ∧ is_factor n 180 ∧ n > 0 ∧ n = 14 :=
by
  sorry

end find_number_l535_535725


namespace find_tabitha_age_l535_535122

-- Define the conditions
variable (age_started : ℕ) (colors_started : ℕ) (years_future : ℕ) (future_colors : ℕ)

-- Let's specify the given problem's conditions:
axiom h1 : age_started = 15          -- Tabitha started at age 15
axiom h2 : colors_started = 2        -- with 2 colors
axiom h3 : years_future = 3          -- in three years
axiom h4 : future_colors = 8         -- she will have 8 different colors

-- The proof problem we need to state:
theorem find_tabitha_age : ∃ age_now : ℕ, age_now = age_started + (future_colors - colors_started) - years_future := by
  sorry

end find_tabitha_age_l535_535122


namespace find_missing_part_l535_535038

variable (x y : ℚ) -- Using rationals as the base field for generality.

theorem find_missing_part :
  2 * x * (-3 * x^2 * y) = -6 * x^3 * y := 
by
  sorry

end find_missing_part_l535_535038


namespace right_angle_clackers_proof_l535_535680

-- Let's define the conditions
def full_circle_clackers : ℕ := 600
def right_angle_fraction : ℚ := 1 / 4

-- Define the question we are proving
def clackers_in_right_angle : ℕ := full_circle_clackers * ↑right_angle_fraction

-- Main theorem statement
theorem right_angle_clackers_proof :
  clackers_in_right_angle = 150 :=
by
  -- We state that clackers in a right angle is 150
  sorry

end right_angle_clackers_proof_l535_535680


namespace sequence_term_l535_535863

open Int

-- Define the sequence {S_n} as stated in the problem
def S (n : ℕ) : ℤ := 2 * n^2 - 3 * n

-- Define the sequence {a_n} as the finite difference of {S_n}
def a (n : ℕ) : ℤ := if n = 1 then -1 else S n - S (n - 1)

-- The theorem statement
theorem sequence_term (n : ℕ) (hn : n > 0) : a n = 4 * n - 5 :=
by sorry

end sequence_term_l535_535863


namespace overall_average_marks_l535_535276

theorem overall_average_marks (n P : ℕ) (P_avg F_avg : ℕ) (H_n : n = 120) (H_P : P = 100) (H_P_avg : P_avg = 39) (H_F_avg : F_avg = 15) :
  (P_avg * P + F_avg * (n - P)) / n = 35 := 
by
  sorry

end overall_average_marks_l535_535276


namespace problem_I3_1_l535_535214

noncomputable def a : ℝ := 1

theorem problem_I3_1 (x : ℝ) (h1 : 2^(x + 1) = 8^((1 / x) - (1 / 3))) (hx : 0 < x) : x = a :=
-- sorry to skip the proof
sorry

end problem_I3_1_l535_535214


namespace current_rate_of_daal_l535_535498

theorem current_rate_of_daal 
  (previous_rate : ℝ)
  (reduced_consumption : ℝ)
  (fixed_expenditure : ℝ)
  (previous_rate_eq : previous_rate = 16)
  (reduced_consumption_eq : reduced_consumption = 0.8)
  (fixed_expenditure_eq : fixed_expenditure = previous_rate) :
  let R := fixed_expenditure / reduced_consumption in
  R = 20 := 
by 
  have h1 : fixed_expenditure / reduced_consumption = 20, from sorry
  exact h1

end current_rate_of_daal_l535_535498


namespace M_trajectory_parabola_l535_535627

-- Define the cube and the points
structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A B C D : Point)
(A1 B1 C1 D1 : Point)

-- Define the plane equation and distance function
def distance (P Q : Point) : ℝ :=
((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)^(1/2)

def distance_point_to_plane (P : Point) (A B C : Point) : ℝ :=
let normal := Point.mk (A.y * (B.z - C.z) + B.y * (C.z - A.z) + C.y * (A.z - B.z))
                      (A.z * (B.x - C.x) + B.z * (C.x - A.x) + C.z * (A.x - B.x))
                      (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) in
(abs (normal.x * P.x + normal.y * P.y + normal.z * P.z) 
     / ((normal.x^2 + normal.y^2 + normal.z^2)^(1/2)))

-- Mathematical statement of the problem
theorem M_trajectory_parabola (cube : Cube) (M : Point) :
  M.x ∈ set.Icc cube.B.x cube.C1.x ∧
  M.y ∈ set.Icc cube.B.y cube.C1.y ∧
  M.z ∈ set.Icc cube.B1.z cube.C1.z ∧
  distance M cube.B = distance_point_to_plane M cube.C cube.D cube.C1 →
  ∃ p : parabola, p.trajectory ∈ face BCC1B1 :=
sorry

end M_trajectory_parabola_l535_535627


namespace area_inequalities_in_triangle_l535_535448

theorem area_inequalities_in_triangle
  (ABC : Triangle)
  (acute_angled : ABC.isAcute)
  (P : Point)
  (inside_triangle : P ∈ ABC.inside)
  (D E F : Point)
  (AP_ext : Line.through P intersects ABC.oppositeSide A D)
  (BP_ext : Line.through P intersects ABC.oppositeSide B E)
  (CP_ext : Line.through P intersects ABC.oppositeSide C F)
  (S_df : Area(ABC.triangle DEF) = Area P)
  (H : Point := ABC.orthocenter)
  (I : Point := ABC.incenter)
  (G : Point := ABC.centroid)
  (S' : NonnegativeReal)
  (S'_def : S' = Area(triangleByIncircle ABC))
  :
  Area(ABC.triangle H) ≤ S' ∧ S' ≤ Area(ABC.triangle I) ∧ Area(ABC.triangle I) ≤ Area(ABC.triangle G) := 
sorry

end area_inequalities_in_triangle_l535_535448


namespace first_comparison_second_comparison_l535_535025

theorem first_comparison (x y : ℕ) (h1 : x = 2^40) (h2 : y = 3^28) : x < y := 
by sorry

theorem second_comparison (a b : ℕ) (h3 : a = 31^11) (h4 : b = 17^14) : a < b := 
by sorry

end first_comparison_second_comparison_l535_535025


namespace trig_identity_l535_535548

theorem trig_identity (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 2 / 3) : 
  Real.cos (2 * α + Real.pi / 3) = -1 / 9 :=
by
  sorry

end trig_identity_l535_535548


namespace find_n_l535_535617

theorem find_n
  (c d : ℝ)
  (H1 : 450 * c + 300 * d = 300 * c + 375 * d)
  (H2 : ∃ t1 t2 t3 : ℝ, t1 = 4 ∧ t2 = 1 ∧ t3 = n ∧ 75 * 4 * (c + d) = 900 * c + t3 * d)
  : n = 600 / 7 :=
by
  sorry

end find_n_l535_535617


namespace median_sum_to_zero_l535_535668

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C A₁ B₁ C₁ : V}

/-- Median vectors of the triangle -/
def medianA (A B C : V) := (A + B + C) / 3
def medianB (A B C : V) := (A + B + C) / 3
def medianC (A B C : V) := (A + B + C) / 3

theorem median_sum_to_zero (hA₁: A₁ = (B + C) / 2) (hB₁: B₁ = (A + C) / 2) (hC₁: C₁ = (A + B) / 2):
  ((A₁ - A) + (B₁ - B) + (C₁ - C)) = 0 :=
sorry

end median_sum_to_zero_l535_535668


namespace egyptian_method_percentage_error_l535_535428

theorem egyptian_method_percentage_error :
  let a := 6
  let b := 4
  let c := 20
  let h := Real.sqrt (c^2 - ((a - b) / 2)^2)
  let S := ((a + b) / 2) * h
  let S1 := ((a + b) * c) / 2
  let percentage_error := abs ((20 / Real.sqrt 399) - 1) * 100
  percentage_error = abs ((20 / Real.sqrt 399) - 1) * 100 := by
  sorry

end egyptian_method_percentage_error_l535_535428


namespace diane_faster_than_rhonda_l535_535343

theorem diane_faster_than_rhonda :
  ∀ (rhonda_time sally_time diane_time total_time : ℕ), 
  rhonda_time = 24 →
  sally_time = rhonda_time + 2 →
  total_time = 71 →
  total_time = rhonda_time + sally_time + diane_time →
  (rhonda_time - diane_time) = 3 :=
by
  intros rhonda_time sally_time diane_time total_time
  intros h_rhonda h_sally h_total h_sum
  sorry

end diane_faster_than_rhonda_l535_535343


namespace laura_change_l535_535999

-- Define the cost of a pair of pants and a shirt.
def cost_of_pants := 54
def cost_of_shirts := 33

-- Define the number of pants and shirts Laura bought.
def num_pants := 2
def num_shirts := 4

-- Define the amount Laura gave to the cashier.
def amount_given := 250

-- Calculate the total cost.
def total_cost := num_pants * cost_of_pants + num_shirts * cost_of_shirts

-- Define the expected change.
def expected_change := 10

-- The main theorem stating the problem and its solution.
theorem laura_change :
  amount_given - total_cost = expected_change :=
by
  sorry

end laura_change_l535_535999


namespace field_trip_student_count_l535_535394

theorem field_trip_student_count
  (boys : ℕ)
  (girls_initial_more : ℕ → ℕ) :
  boys = 8 →
  girls_initial_more boys = boys + 2 →
  ∀ (girls_missing : ℕ), girls_missing = 2 → (boys + girls_initial_more boys = 18) :=
begin
  intros boys_eq more_girls_eq girls_missing_eq,
  rw boys_eq at *,
  rw more_girls_eq at *,
  rw girls_missing_eq,
  -- We now have 8 boys and 10 girls initially.
  have num_girls_initial : ℕ := 10,
  have num_total_students := 8 + num_girls_initial,
  exact num_total_students,
end

end field_trip_student_count_l535_535394


namespace probability_of_average_5_is_1_9_l535_535865

-- Definitions given in the problem
def numbers_set : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def total_possibilities : ℕ := 36

-- Prove that the probability of selecting any seven distinct numbers from
-- the set {1, 2, 3, 4, 5, 6, 7, 8, 9} such that their average is 5 is 1/9
theorem probability_of_average_5_is_1_9 :
  (∃ count, 
    count = 4 ∧
    probability = count / total_possibilities ∧
    probability = 1/9) :=
by {
  sorry
}

end probability_of_average_5_is_1_9_l535_535865


namespace no_congruent_disjoint_union_l535_535837

noncomputable def disk : set (ℝ × ℝ) := 
{p | let (x, y) := p in x^2 + y^2 ≤ 1}

theorem no_congruent_disjoint_union :
  ∀ (A B : set (ℝ × ℝ)), 
  (∀ p, p ∈ A → ∃ q, q ∈ B ∧ (∃ f : (ℝ × ℝ) → (ℝ × ℝ), isometry f ∧ p = f q)) →
  A ∩ B = ∅ →
  A ∪ B = disk →
  false :=
by { intros A B h_congruence h_disjoint h_union, sorry }

end no_congruent_disjoint_union_l535_535837


namespace find_m_find_n_l535_535934

-- Definition of the vectors and the dot product function
def vec_a (x : ℝ) (m : ℝ) : ℝ × ℝ := (2 * Real.sin x , m)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.sin x + Real.cos x , 1)
def f (x : ℝ) (m : ℝ) : ℝ :=
  let a := vec_a x m
  let b := vec_b x
  a.1 * b.1 + a.2 * b.2

-- The given condition: maximum value of f(x) is √2
axiom max_value_of_f (m : ℝ) : ∃ x : ℝ, f x m = Real.sqrt 2

-- Question 1: Find the value of m
theorem find_m (m : ℝ) : max_value_of_f m → m = -1 := 
by
  sorry

-- Helper function to represent the translated and reflected function
def translated_f (x n : ℝ) (m : ℝ) : ℝ :=
  f (x + n) m

-- Condition for translated function being even (symmetry about y-axis)
axiom even_function_condition (n : ℝ) (m : ℝ) : 
  ∃ x : ℝ, translated_f x n m = Real.sqrt 2 * Real.sin (2 * (x + n) - Real.pi / 4)

-- Question 2: Find the minimum value of n
theorem find_n (n : ℝ) : even_function_condition n -1 → n = 3 * Real.pi / 8 := 
by
  sorry

end find_m_find_n_l535_535934


namespace proof_probability_l535_535439

variable (Xavier_succeeds : Prop)
variable (Yvonne_succeeds : Prop)
variable (Zelda_succeeds : Prop)

variable [decidable_pred Xavier_succeeds]
variable [decidable_pred Yvonne_succeeds]
variable [decidable_pred Zelda_succeeds]

def probability_Xavier_succeeds : ℚ := 1 / 6
def probability_Yvonne_succeeds : ℚ := 1 / 2
def probability_Zelda_succeeds : ℚ := 5 / 8

def probability_Xavier_Yvonne_not_Zelda : ℚ :=
  probability_Xavier_succeeds * probability_Yvonne_succeeds * (1 - probability_Zelda_succeeds)

theorem proof_probability :
  probability_Xavier_Yvonne_not_Zelda = 1 / 32 :=
by
  unfold probability_Xavier_succeeds probability_Yvonne_succeeds probability_Zelda_succeeds probability_Xavier_Yvonne_not_Zelda
  sorry

end proof_probability_l535_535439


namespace parametric_to_standard_max_distance_to_line_l535_535278

-- Given parametric equations for line l
def line_param_eqn (t : ℝ) : ℝ × ℝ := 
  (-1 + (Real.sqrt 2) / 2 * t, 2 + (Real.sqrt 2) / 2 * t)

-- Given parametric equations for curve C
def curve_param_eqn (θ : ℝ) : ℝ × ℝ := 
  (4 * Real.cos θ, Real.cos (2 * θ))

-- Standard form of the curve C
def curve_standard_eqn (x : ℝ) : ℝ := 
  x^2 / 8 - 1

-- Theorem 1: Parametric to standard form
theorem parametric_to_standard (θ : ℝ) : 
  curve_param_eqn θ = (x, curve_standard_eqn x) ∧ x ∈ [-4, 4] := sorry

-- Distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  Real.abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

-- Parameters for line l
def line_params : ℝ × ℝ × ℝ := (1, -1, 3)

-- Theorem 2: Maximum distance from a point on curve C to line l
theorem max_distance_to_line (θ : ℝ) : 
  ∃ θ_max, distance_point_to_line (curve_param_eqn θ_max) 1 (-1) 3 = 3 * Real.sqrt 2 := sorry

end parametric_to_standard_max_distance_to_line_l535_535278


namespace compute_difference_of_squares_l535_535834

theorem compute_difference_of_squares :
  let a := 23
  let b := 12
  (a + b) ^ 2 - (a - b) ^ 2 = 1104 := by
sorry

end compute_difference_of_squares_l535_535834


namespace arrange_books_l535_535076

theorem arrange_books :
  let total_books := 9
  let algebra_books := 4
  let calculus_books := 5
  algebra_books + calculus_books = total_books →
  Nat.choose total_books algebra_books = 126 :=
by
  intro h
  rw [Nat.choose_eq_factorial_div_factorial (total_books - algebra_books)]
  sorry

end arrange_books_l535_535076


namespace Isabel_total_problems_l535_535641

theorem Isabel_total_problems :
  let math_pages := 2
  let reading_pages := 4
  let science_pages := 3
  let history_pages := 1
  let problems_per_math_page := 5
  let problems_per_reading_page := 5
  let problems_per_science_page := 7
  let problems_per_history_page := 10
  let total_math_problems := math_pages * problems_per_math_page
  let total_reading_problems := reading_pages * problems_per_reading_page
  let total_science_problems := science_pages * problems_per_science_page
  let total_history_problems := history_pages * problems_per_history_page
  let total_problems := total_math_problems + total_reading_problems + total_science_problems + total_history_problems
  total_problems = 61 := by
  sorry

end Isabel_total_problems_l535_535641


namespace a_2017_l535_535568

noncomputable def sequence_a : ℕ → ℚ
| 0     := 0
| (n+1) := if n = 0 then 1 else -1 / (1 + sequence_a n)

lemma a_1 : sequence_a 1 = 1 := rfl

lemma recurrence (n : ℕ) : sequence_a (n + 2) = -1 / (1 + sequence_a (n + 1)) :=
  by { cases n; simp [sequence_a], }

lemma periodicity (n : ℕ) : sequence_a (n + 3) = sequence_a n :=
by {
  intro n,
  induction n using Nat.strong_induction_on with n IH,
  cases n with n,
  { rw [sequence_a, sequence_a, sequence_a] },
  { cases n with n,
    { rw [sequence_a, sequence_a] },
    { specialize IH n.succ (Nat.lt_succ_of_le (le_refl _)),
      rw [sequence_a, sequence_a, IH],
      norm_num } }
}

theorem a_2017 : sequence_a 2017 = 1 :=
by {
  have p := periodicity 671,
  simp at p,
  exact p,
}

end a_2017_l535_535568


namespace count_four_digit_no_5_no_6_l535_535241

theorem count_four_digit_no_5_no_6 : 
  (count_digits_no_5_no_6 4) = 3584 :=
sorry

/-- Counts the number of n-digit numbers with no digits being 5 or 6. -/
def count_digits_no_5_no_6 (n : ℕ) : ℕ :=
if n = 4 then 
  7 * (8 ^ 3)
else 
  0 -- Here we specify only for 4-digit numbers, otherwise we return 0

end count_four_digit_no_5_no_6_l535_535241


namespace largest_divisor_of_product_of_consecutive_odd_integers_l535_535665

theorem largest_divisor_of_product_of_consecutive_odd_integers :
  ∀ (a b c d : ℕ), (a % 2 = 1) ∧ (b = a + 2) ∧ (c = b + 2) ∧ (d = c + 2) 
  → ∃ (k : ℕ), is_greatest (λ n, n ∣ (a * b * c * d)) k ∧ k = 3 :=
by sorry

end largest_divisor_of_product_of_consecutive_odd_integers_l535_535665


namespace eccentricity_range_l535_535663

noncomputable def ellipse_eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  Set ℝ := {e : ℝ | e = (real.sqrt (a^2 - b^2)) / a ∧ (∃ P : ℝ × ℝ, P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧
  let F1 := (-real.sqrt (a^2 - b^2), 0)
      F2 := (real.sqrt (a^2 - b^2), 0)
  in angle P F1 P F2 = π / 2) } 

theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  ∀ e ∈ ellipse_eccentricity_range a b h, e ∈ Set.Ico (real.sqrt 2 / 2) 1 :=
begin
  sorry
end

end eccentricity_range_l535_535663


namespace denmark_pizza_topping_combinations_l535_535110

theorem denmark_pizza_topping_combinations :
  let cheese_options := 3
  let meat_options := 4
  let vegetable_options := 5
  let pepperoni := 1  -- The specific meat option that is pepperoni
  let peppers := 1    -- The specific vegetable option that is peppers
  let total_combinations := (cheese_options * meat_options * vegetable_options) -

    -- Subtracting combinations where pepperoni excludes peppers:
    (cheese_options * pepperoni * (vegetable_options - peppers)) + 

    -- Adding combinations where pepperoni is not chosen:
    (cheese_options * (meat_options - pepperoni) * vegetable_options)
  in 
  total_combinations = 57 :=
by
  sorry

end denmark_pizza_topping_combinations_l535_535110


namespace intersection_value_l535_535634

noncomputable def cartesian_coords (ρ θ : ℝ) : ℝ × ℝ :=
(ρ * Real.cos θ, ρ * Real.sin θ)

def parametric_C1 (t : ℝ) : ℝ × ℝ :=
(-2 + 3 * t, -2 + 4 * t)

def cartesian_C1 (x y : ℝ) : Prop :=
4 * x - 3 * y + 6 = 0

def cartesian_C2 (x y : ℝ) : Prop :=
x^2 + y^2 - 4 * x = 0

def P := cartesian_coords (2 * Real.sqrt 2) (-3 * Real.pi / 4)

def distance (A B : ℝ × ℝ) : ℝ :=
Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

lemma parametric_to_cartesian (t : ℝ) : cartesian_C2 (-2 + 3 * t) (-2 + 4 * t) :=
begin
  sorry -- This step involves showing that substituted coordinates satisfy the equation
end

lemma intersection_t_values (t1 t2 : ℝ) : t1 + t2 = 16 / 5 ∧ t1 * t2 = 80 / 5 :=
begin
  sorry -- This step involves solving the quadratic equation from substitution
end

def value_of_expr (P A B : ℝ × ℝ) (t1 t2 : ℝ) := (1 / distance P A) + (1 / distance P B)

theorem intersection_value :
  ∃ t1 t2 : ℝ, (intersection_t_values t1 t2) → value_of_expr P (parametric_C1 t1) (parametric_C1 t2) = 1 / 5 :=
begin
  sorry -- We would show here that the given distances yield the required value
end

end intersection_value_l535_535634


namespace max_value_of_function_l535_535135

noncomputable def max_value_function (t : ℝ) : ℝ :=
if t < -2 then -(t + 2)^2 + 1 else
if -2 ≤ t ∧ t ≤ -1 then 1 else
-(t + 1)^2 + 1

theorem max_value_of_function (t : ℝ) (x : ℝ) (h : x ∈ set.Icc t (t + 1)) :
  ∃ y, y = -x^2 - 2*x ∧
    (y ≤ max_value_function t) :=
by
  sorry

end max_value_of_function_l535_535135


namespace time_to_destination_l535_535441

theorem time_to_destination (speed_ratio : ℕ) (mr_harris_time : ℕ) 
  (distance_multiple : ℕ) (h1 : speed_ratio = 3) 
  (h2 : mr_harris_time = 3) 
  (h3 : distance_multiple = 5) : 
  (mr_harris_time / speed_ratio) * distance_multiple = 5 := by
  sorry

end time_to_destination_l535_535441


namespace prime_factor_computation_l535_535527

def log_base_10 (n : ℕ) := real.log n / real.log 10 

def count_prime_factors (n : ℕ) : ℕ :=
-- dummy implementation, assume this computes the number of not necessarily distinct prime factors.
sorry

theorem prime_factor_computation :
  ∃ a b : ℕ,
    a > 0 ∧ b > 0 ∧
    (3 * log_base_10 a + 2 * log_base_10 (Nat.gcd a b) = 90) ∧
    (log_base_10 b + 2 * log_base_10 (Nat.lcm a b) = 310) ∧
    (let p := count_prime_factors a in
     let q := count_prime_factors b in
     4 * p + 3 * q = 800) :=
sorry

end prime_factor_computation_l535_535527


namespace count_terminating_decimals_l535_535141

theorem count_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 500) : 
  (nat.floor (500 / 49) = 10) := 
by
  sorry

end count_terminating_decimals_l535_535141


namespace prob_3_right_letters_l535_535002

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def derangement (n : ℕ) : ℕ :=
  factorial n * (1 - (1 / 1.factorial) + (1 / 2.factorial) - (1 / 3.factorial) + (1 / 4.factorial) - ... + (if n % 2 = 0 then 1 / (n.factorial) else -1 / (n.factorial)))

theorem prob_3_right_letters : (binomial 7 3 * derangement 4) / (factorial 7) = 1 / 16 :=
by
  sorry

end prob_3_right_letters_l535_535002


namespace count_non_squares_or_cubes_l535_535939

theorem count_non_squares_or_cubes (n : ℕ) (h₀ : 1 ≤ n ∧ n ≤ 200) : 
  ∃ c, c = 182 ∧ 
  (∃ k, k^2 = n ∨ ∃ m, m^3 = n) → false :=
by
  sorry

end count_non_squares_or_cubes_l535_535939


namespace find_possible_y_values_l535_535355

-- Given conditions, defining the polynomial P(x)
def P (x : ℝ) (a b c : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + a * x + c

-- The set of possible y values identified
def possible_y_values : set ℝ :=
  { y | y = π / 8 ∨ y = 3 * π / 8 ∨ y = 5 * π / 8 ∨ y = 7 * π / 8 ∨
         y = π / 7 ∨ y = 2 * π / 7 ∨ y = 3 * π / 7 ∨ y = 4 * π / 7 ∨ 
         y = 5 * π / 7 ∨ y = 6 * π / 7 ∨
         y = π / 9 ∨ y = 2 * π / 9 ∨ y = π / 3 ∨ y = 4 * π / 9 ∨ 
         y = 5 * π / 9 ∨ y = 2 * π / 3 ∨ y = 7 * π / 9 ∨ y = 8 * π / 9 }

-- Problem statement to prove
theorem find_possible_y_values (a b c : ℝ) :
  (∀ x ∈ set.univ, P x a b c = 0 → (x = real.tan y ∨ x = real.tan (2 * y) ∨ x = real.tan (3 * y)) → y ∈ Icc 0 π) →
  (y ∈ possible_y_values) :=
sorry

end find_possible_y_values_l535_535355


namespace simplify_expression_l535_535092

variable (a b c : ℝ)
variable (h_a : a ≠ 3) (h_b : b ≠ 4) (h_c : c ≠ 5)

theorem simplify_expression : 
  (\(a - 3) / (5 - c) * (b - 4) / (3 - a) * (c - 5) / (4 - b\)) = -1 :=
sorry

end simplify_expression_l535_535092


namespace remainder_is_constant_when_b_is_minus_five_halves_l535_535857

noncomputable def poly_div_remainder (p d : Polynomial ℝ) : Polynomial ℝ × Polynomial ℝ :=
Polynomial.divModByMonicAux p d

theorem remainder_is_constant_when_b_is_minus_five_halves 
  (p d : Polynomial ℝ) (b : ℝ) :
  p = Polynomial.C 12 * X ^ 4 + Polynomial.C (-5) * X ^ 3 + Polynomial.C b * X ^ 2 + Polynomial.C (-4) * X + Polynomial.C 8 →
  d = Polynomial.C 3 * X ^ 2 + Polynomial.C (-2) * X + Polynomial.C 1 →
  (poly_div_remainder p d).snd = Polynomial.C c :=
  sorry

end remainder_is_constant_when_b_is_minus_five_halves_l535_535857


namespace eighteen_letter_arrangement_l535_535597

theorem eighteen_letter_arrangement :
  (∑ k in Finset.range 7, (Nat.choose 6 k)^3) = 
  ∑ k in Finset.range 7, (Nat.choose 6 k)^3 := 
sorry

end eighteen_letter_arrangement_l535_535597


namespace belt_length_sufficient_l535_535398

theorem belt_length_sufficient (r O_1O_2 O_1O_3 O_3_plane : ℝ) 
(O_1O_2_eq : O_1O_2 = 12) (O_1O_3_eq : O_1O_3 = 10) (O_3_plane_eq : O_3_plane = 8) (r_eq : r = 2) : 
(∃ L₁ L₂, L₁ = 32 + 4 * Real.pi ∧ L₂ = 22 + 2 * Real.sqrt 97 + 4 * Real.pi ∧ 
L₁ ≠ 54 ∧ L₂ > 54) := 
by 
  sorry

end belt_length_sufficient_l535_535398


namespace kareem_largest_l535_535649

def jose_final : ℕ :=
  let start := 15
  let minus_two := start - 2
  let triple := minus_two * 3
  triple + 5

def thuy_final : ℕ :=
  let start := 15
  let triple := start * 3
  let minus_two := triple - 2
  minus_two + 5

def kareem_final : ℕ :=
  let start := 15
  let minus_two := start - 2
  let add_five := minus_two + 5
  add_five * 3

theorem kareem_largest : kareem_final > jose_final ∧ kareem_final > thuy_final := by
  sorry

end kareem_largest_l535_535649


namespace sequence_increasing_range_lambda_l535_535235

theorem sequence_increasing_range_lambda (λ : ℝ) (a_n : ℕ → ℝ) :
  (∀ n : ℕ, 0 < n → a_n n = n^2 - 2 * λ * n + 1) →
  (∀ n : ℕ, 0 < n → a_n (n + 1) > a_n n) ↔ λ < 3 / 2 :=
sorry

end sequence_increasing_range_lambda_l535_535235


namespace problem_statement_l535_535302

def A (x : ℝ) : ℝ := 3 * real.sqrt x
def B (x : ℝ) : ℝ := x ^ 3

theorem problem_statement : A (B (A (B (A (B 2))))) = 792 * real.root 4 (6) :=
by
    sorry

end problem_statement_l535_535302


namespace common_sum_of_matrix_l535_535751

theorem common_sum_of_matrix :
  let S := (1 / 2 : ℝ) * 25 * (10 + 34)
  let adjusted_total := S + 10
  let common_sum := adjusted_total / 6
  common_sum = 93.33 :=
by
  sorry

end common_sum_of_matrix_l535_535751


namespace meeting_2015th_at_C_l535_535159

-- Conditions Definitions
variable (A B C D P : Type)
variable (x y t : ℝ)  -- speeds and starting time difference
variable (mw cyclist : ℝ → ℝ)  -- paths of motorist and cyclist

-- Proof statement
theorem meeting_2015th_at_C 
(Given_meeting_pattern: ∀ n : ℕ, odd n → (mw (n * (x + y))) = C):
  (mw (2015 * (x + y))) = C := 
by 
  sorry  -- Proof omitted

end meeting_2015th_at_C_l535_535159


namespace truncated_pyramid_angle_l535_535747

theorem truncated_pyramid_angle
  (α β : ℝ) 
  (hypα : 0 < α ∧ α < (π / 2)) 
  (hypβ : 0 < β ∧ β < π) : 
  ∃ θ : ℝ, θ = arctan (tan α / cos (β / 2)) :=
by
  use arctan (tan α / cos (β / 2))
  sorry

end truncated_pyramid_angle_l535_535747


namespace ellipse_l535_535819

noncomputable def ellipse_parametric (t : ℝ) : ℝ × ℝ :=
  ( (3 * (Real.cos t - 2)) / (3 - Real.sin t), 
    (4 * (Real.sin t - 3)) / (3 - Real.sin t) )

def ellipse_standard_form (x y : ℝ) : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ :=
  sorry  -- Placeholder for coefficients calculation

theorem ellipse(params : ℝ → ℝ × ℝ) :
  ∃ (A B C D E F : ℤ),
  (∀ x y, params ((A : ℝ) * x^2 + (B : ℝ) * x * y + (C : ℝ) * y^2 + (D : ℝ) * x + (E : ℝ) * y + (F : ℝ) = 0))
  ∧ Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D)) (Int.natAbs E)) (Int.natAbs F) = 1 :=
exists.intro A (exists.intro B (exists.intro C (exists.intro D (exists.intro E (exists.intro F,
begin
  sorry  -- Placeholder for the proof
end)))))

end ellipse_l535_535819


namespace survey_method_is_census_l535_535714

-- Define the conditions
def high_accuracy : Prop := true -- placeholder for actual condition
def great_importance : Prop := true -- placeholder for actual condition

-- Define the method used for the survey
def survey_method (high_accuracy : Prop) (great_importance : Prop) : String :=
  if high_accuracy ∧ great_importance then "Census" else "Other"

-- Proof problem statement
theorem survey_method_is_census : high_accuracy → great_importance → survey_method high_accuracy great_importance = "Census" :=
by
  sorry

end survey_method_is_census_l535_535714


namespace students_attend_Purum_Elementary_School_l535_535728
open Nat

theorem students_attend_Purum_Elementary_School (P N : ℕ) 
  (h1 : P + N = 41) (h2 : P = N + 3) : P = 22 :=
sorry

end students_attend_Purum_Elementary_School_l535_535728


namespace meeting_point_2015_l535_535181

/-- 
A motorist starts at point A, and a cyclist starts at point B. They travel towards each other and 
meet for the first time at point C. After meeting, they turn around and travel back to their starting 
points and continue this pattern of meeting, turning around, and traveling back to their starting points. 
Prove that their 2015th meeting point is at point C.
-/
theorem meeting_point_2015 
  (A B C D : Type) 
  (x y t : ℕ)
  (odd_meeting : ∀ n : ℕ, (2 * n + 1) % 2 = 1) : 
  ∃ n, (n = 2015) → odd_meeting n = 1 → (n % 2 = 1 → (C = "C")) := 
sorry

end meeting_point_2015_l535_535181


namespace sin_double_angle_sub_pi_over_4_l535_535546

theorem sin_double_angle_sub_pi_over_4 (x : ℝ) :
  (sin x = (sqrt 5 - 1) / 2) → sin (2 * (x - π / 4)) = 2 - sqrt 5 :=
by
  intro h
  sorry

end sin_double_angle_sub_pi_over_4_l535_535546


namespace inequality_proof_l535_535666

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (h_sum : sqrt a + sqrt b + sqrt c = 3)

theorem inequality_proof (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (h_sum : sqrt a + sqrt b + sqrt c = 3) : 
    (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3 / 2 := 
by 
  sorry

end inequality_proof_l535_535666


namespace two_digit_number_probability_div_by_2_or_3_l535_535081

theorem two_digit_number_probability_div_by_2_or_3 : 
  let total_count := 90 in
  let even_count := 45 in
  let div_by_3_count := 30 in
  let div_by_6_count := 15 in
  let favorable_count := even_count + div_by_3_count - div_by_6_count in
  (favorable_count / total_count : ℝ) = 2 / 3 :=
by
  let total_count := 90
  let even_count := 45
  let div_by_3_count := 30
  let div_by_6_count := 15
  let favorable_count := even_count + div_by_3_count - div_by_6_count
  show (favorable_count / total_count : ℝ) = 2 / 3
  sorry

end two_digit_number_probability_div_by_2_or_3_l535_535081


namespace inequality_true_l535_535192

variable {x y z : ℝ}

theorem inequality_true 
  (h1 : x > y)
  (h2 : y > z)
  (h3 : x + y + z = 0) : 
  xy > xz := 
sorry

end inequality_true_l535_535192


namespace min_value_ge_9_l535_535318

noncomputable theory

open Real

theorem min_value_ge_9 (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 3^a * 3^b = 3) : 
  (1 / a) + (4 / b) ≥ 9 :=
by
  sorry

end min_value_ge_9_l535_535318


namespace initial_acidic_liquid_quantity_l535_535426

theorem initial_acidic_liquid_quantity
  (A : ℝ) -- initial quantity of the acidic liquid in liters
  (W : ℝ) -- quantity of water to be removed in liters
  (h1 : W = 6)
  (h2 : (0.40 * A) = 0.60 * (A - W)) : 
  A = 18 :=
by sorry

end initial_acidic_liquid_quantity_l535_535426


namespace polynomial_coeff_l535_535926

noncomputable def a_coeffs := {a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ}

theorem polynomial_coeff 
    (x : ℝ)
    (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) 
    (h : x^2 + 2*x^(11) = a_0 + a_1*(x+1) + a_2*(x+1)^2 + a_3*(x+1)^3 + a_4*(x+1)^4 + a_5*(x+1)^5 + a_6*(x+1)^6 + a_7*(x+1)^7 + a_8*(x+1)^8 + a_9*(x+1)^9 + a_10*(x+1)^10 + a_11*(x+1)^11) :
  a_10 = -22 :=
by
  sorry

end polynomial_coeff_l535_535926


namespace rectangle_I_A_I_B_I_C_I_D_l535_535662

variables {A B C D S : Type} [Point A] [Point B] [Point C] [Point D]

def cyclic_quadrilateral (A B C D : Type) [Point A] [Point B] [Point C] [Point D] : Prop :=
  ∃ circumcircle : Circle, A ∈ circumcircle ∧ B ∈ circumcircle ∧ C ∈ circumcircle ∧ D ∈ circumcircle

def incenter (P Q R : Type) [Point P] [Point Q] [Point R] : Type :=
  let incenter : Point := sorry
  incenter

variables (I_A I_B I_C I_D : Type)
          [Point I_A] [Point I_B] [Point I_C] [Point I_D]
          [Incenter I_A B C D] [Incenter I_B A D C] [Incenter I_C D A B] [Incenter I_D B A C]

theorem rectangle_I_A_I_B_I_C_I_D
  (h_cyclic : cyclic_quadrilateral A B C D)
  (h_I_A : I_A = incenter B C D)
  (h_I_B : I_B = incenter A D C)
  (h_I_C : I_C = incenter D A B)
  (h_I_D : I_D = incenter B A C) : 
  is_rectangle I_A I_B I_C I_D :=
sorry

end rectangle_I_A_I_B_I_C_I_D_l535_535662


namespace point_on_exp_graph_cos_value_l535_535961

theorem point_on_exp_graph_cos_value (n : ℝ) (h : 3^n = 3) : Real.cos (π / (3 * n)) = 1 / 2 := 
by 
  sorry

end point_on_exp_graph_cos_value_l535_535961


namespace boxes_left_to_sell_l535_535990

def sales_goal : ℕ := 150
def first_customer : ℕ := 5
def second_customer : ℕ := 4 * first_customer
def third_customer : ℕ := second_customer / 2
def fourth_customer : ℕ := 3 * third_customer
def fifth_customer : ℕ := 10
def total_sold : ℕ := first_customer + second_customer + third_customer + fourth_customer + fifth_customer

theorem boxes_left_to_sell : sales_goal - total_sold = 75 := by
  sorry

end boxes_left_to_sell_l535_535990


namespace smallest_integer_y_l535_535753

theorem smallest_integer_y (y : ℤ) (h : 3 - 5 * y < 23) : -3 ≥ y :=
by {
  sorry
}

end smallest_integer_y_l535_535753


namespace geometric_sequence_form_l535_535975

-- Definitions for sequences and common difference/ratio
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ (m n : ℕ), a n = a m + (n - m) * d

def isGeometricSeq (b : ℕ → ℝ) (q : ℝ) :=
  ∀ (m n : ℕ), b n = b m * q ^ (n - m)

-- Problem statement: given an arithmetic sequence, find the form of the corresponding geometric sequence
theorem geometric_sequence_form
  (b : ℕ → ℝ) (q : ℝ) (m n : ℕ) (b_m : ℝ) (q_pos : q > 0) :
  (∀ (m n : ℕ), b n = b m * q ^ (n - m)) :=
sorry

end geometric_sequence_form_l535_535975


namespace circumcircles_intersect_on_angle_bisector_l535_535196

-- Given a triangle ABC.
variables {A B C A1 C1 : Point}

-- Definitions for the conditions given
def is_extension (X Y Z : Point) : Prop := collinear X Y Z ∧ dist X Z > dist Y Z

-- The following definitions suppose we have a formal geometry setup
-- Definition for points A1 and C1 being on the extensions of AB and CB beyond B respectively
def extension_condition (A B C A1 C1 : Point) : Prop :=
  is_extension A B A1 ∧ is_extension B C C1 ∧ dist A C = dist A1 C ∧ dist A C = dist C1 C

-- Definition for circumcircle of a triangle
def on_circumcircle (P Q R X : Point) : Prop :=
  dist P X * dist Q X = dist X R * dist P X ∧ dist P X * dist X R = dist Q X * dist R X

-- Statement to prove: circumcircles of triangles ABA1 and CBC1 intersect on angle bisector of angle B.
theorem circumcircles_intersect_on_angle_bisector
  (h : extension_condition A B C A1 C1) :
  ∃ I : Point, 
    (on_circumcircle A B A1 I ∧ on_circumcircle C B C1 I) ∧ 
    is_angle_bisector_of ∠ABC I :=
sorry

end circumcircles_intersect_on_angle_bisector_l535_535196


namespace cubic_difference_l535_535567

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 59) : a^3 - b^3 = 448 :=
by
  sorry

end cubic_difference_l535_535567


namespace arrangement_plans_l535_535333

theorem arrangement_plans (graduates_universities : fin 6 → fin 3) 
  (grouping : fin 3 → fin 6) :
  (∀ (i : fin 3), (graduates_universities (grouping i * 2) ≠ graduates_universities (grouping i * 2 + 1))) →
  ∃ (arrangement_plans : fin 6 → fin 3), 
  finset.card finset.univ = 48 := 
sorry

end arrangement_plans_l535_535333


namespace clique_partition_l535_535086

variables (G : Type) [Graph G] (V : set G) (is_clique : set G → Prop) (largest_clique : is_clique → ℕ)

-- Conditions
variable (even_largest_clique : ∃ (clique : set G), is_clique clique ∧ largest_clique clique mod 2 = 0)

-- Question to prove
theorem clique_partition (G : Type) [Graph G] (V : set G) (is_clique : set G → Prop) (largest_clique : set G → ℕ)
  (even_largest_clique : ∃ (clique : set G), is_clique clique ∧ largest_clique clique mod 2 = 0) :
  ∃ (A B : set G), A ∪ B = V ∧ A ∩ B = ∅ ∧ (∃ a, is_clique a ∧ largest_clique a = largest_clique (V \ A)) :=
begin
  sorry
end

end clique_partition_l535_535086


namespace y_over_x_in_range_l535_535300

open Real

noncomputable def given_real_condition (x y : ℝ) : Prop :=
  x ≥ sqrt 2021 ∧ (cbrt (x + sqrt 2021) + cbrt (x - sqrt 2021) = cbrt y)

theorem y_over_x_in_range (x y : ℝ) (h : given_real_condition x y) : (2 ≤ y / x) ∧ (y / x < 8) :=
sorry

end y_over_x_in_range_l535_535300


namespace area_complement_set_A_eq_4pi_l535_535594

noncomputable def U : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y)}

noncomputable def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 1) * Real.cos a + y * Real.sin a = 2}

theorem area_complement_set_A_eq_4pi (a : ℝ) : 
  ∃ (complement_A : Set (ℝ × ℝ)), 
    complement_A = {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 1) ^ 2 + y ^ 2 = 4} ∧
    ∀ (S : Set (ℝ × ℝ)), S = complement_A → measure_theory.measure.univ S = 4 * Real.pi :=
sorry

end area_complement_set_A_eq_4pi_l535_535594


namespace theresa_crayons_count_l535_535734

noncomputable def crayons_teresa (initial_teresa_crayons : Nat) 
                                 (initial_janice_crayons : Nat) 
                                 (shared_with_nancy : Nat)
                                 (given_to_mark : Nat)
                                 (received_from_nancy : Nat) : Nat := 
  initial_teresa_crayons + received_from_nancy

theorem theresa_crayons_count : crayons_teresa 32 12 (12 / 2) 3 8 = 40 := by
  -- Given: Theresa initially has 32 crayons.
  -- Janice initially has 12 crayons.
  -- Janice shares half of her crayons with Nancy: 12 / 2 = 6 crayons.
  -- Janice gives 3 crayons to Mark.
  -- Theresa receives 8 crayons from Nancy.
  -- Therefore: Theresa will have 32 + 8 = 40 crayons.
  sorry

end theresa_crayons_count_l535_535734


namespace range_of_m_l535_535545

-- Define the floor function [x]
def floor (x : ℝ) : ℤ := int.floor x

-- Define set A 
def A : set ℝ := {y | ∃ x : ℝ, y = x - (floor x)}

-- Define set B parameterized by m
def B (m : ℝ) : set ℝ := {y | 0 ≤ y ∧ y ≤ m}

-- The theorem we need to prove
theorem range_of_m : ∀ m, (A ⊆ B m) → (1 ≤ m) :=
by
  assume m,
  sorry

end range_of_m_l535_535545


namespace boat_speed_in_still_water_eq_16_l535_535782

theorem boat_speed_in_still_water_eq_16 (stream_rate : ℝ) (time_downstream : ℝ) (distance_downstream : ℝ) (V_b : ℝ) 
(h1 : stream_rate = 5) (h2 : time_downstream = 6) (h3 : distance_downstream = 126) : 
  V_b = 16 :=
by sorry

end boat_speed_in_still_water_eq_16_l535_535782


namespace sqrt_three_irrational_sqrt_three_infinite_non_repeating_l535_535112

theorem sqrt_three_irrational : irrational (sqrt 3) :=
sorry

theorem sqrt_three_infinite_non_repeating :
  ∀ (x : ℝ), (x = sqrt 3) → (¬ (∃ r : ℚ, (x : ℝ) = r)) ∧ (non_repeating (decimal_expansion x)) :=
by
  intro x hx
  have h : irrational (sqrt 3) := sqrt_three_irrational
  split
  {
    -- Proof that √3 is not a rational number
    sorry
  }
  {
    -- Proof that the decimal expansion of √3 is non-repeating
    sorry
  }

end sqrt_three_irrational_sqrt_three_infinite_non_repeating_l535_535112


namespace limit_seq_eq_e_neg33_l535_535489

noncomputable def lim_seq : ℕ → ℝ := λ n, ( (n - 10) / (n + 1) )^(3 * n + 1)

theorem limit_seq_eq_e_neg33 : 
  (tendsto (λ n, lim_seq n) at_top (𝓝 (real.exp (-33)))) :=
by sorry

end limit_seq_eq_e_neg33_l535_535489


namespace hyperbola_asymptote_l535_535255

theorem hyperbola_asymptote (a b : ℝ) (h1 : ∃ e: ℝ, e = 3 ∧ e = real.sqrt (1 + (b^2 / a^2))) :
  ∀ y x : ℝ, y = (±(real.sqrt 2) / 4) * x :=
begin
  sorry
end

end hyperbola_asymptote_l535_535255


namespace complex_imaginary_part_simplified_l535_535229

theorem complex_imaginary_part_simplified : Im (i / (1 - i)) = 1 / 2 :=
by
  sorry

end complex_imaginary_part_simplified_l535_535229


namespace center_of_circle_l535_535840

theorem center_of_circle : ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 → (1, 1) = (1, 1) :=
by
  intros x y h
  sorry

end center_of_circle_l535_535840


namespace expression_of_y_l535_535543

theorem expression_of_y (x y : ℝ) (h : x - y / 2 = 1) : y = 2 * x - 2 :=
sorry

end expression_of_y_l535_535543


namespace order_of_abc_l535_535566

-- Define the variables a, b, and c
noncomputable def a : ℝ := sqrt 5 - sqrt 3
noncomputable def b : ℝ := sqrt 3 - 1
noncomputable def c : ℝ := sqrt 7 - sqrt 5

-- Theorem statement to prove the desired order
theorem order_of_abc : b > a ∧ a > c := by 
  sorry

end order_of_abc_l535_535566


namespace total_pencils_l535_535390

def initial_pencils : ℕ := 2
def added_pencils : ℕ := 3

theorem total_pencils : initial_pencils + added_pencils = 5 :=
by {
  exact eq.refl 5
}

end total_pencils_l535_535390


namespace find_m_l535_535554

theorem find_m (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) ∧ (m - 1 ≠ 0) → m = -1 :=
by
  sorry

end find_m_l535_535554


namespace number_of_proper_subsets_of_set_A_l535_535929

def set_A : Finset ℕ := {1, 2, 3}

theorem number_of_proper_subsets_of_set_A :
  (set_A.powerset.filter (λ s, s ≠ set_A)).card = 7 := 
sorry

end number_of_proper_subsets_of_set_A_l535_535929


namespace chord_line_midpoint_l535_535909

def is_midpoint (p mid end1 end2 : ℝ × ℝ) : Prop :=
  (p.1 = (end1.1 + end2.1) / 2) ∧ (p.2 = (end1.2 + end2.2) / 2)

def on_ellipse (end : ℝ × ℝ) : Prop :=
  (end.1 ^ 2) / 36 + (end.2 ^ 2) / 9 = 1

theorem chord_line_midpoint (A B : ℝ × ℝ) 
(hA : on_ellipse A) (hB : on_ellipse B) 
(mid : ℝ × ℝ) (hM : mid = (1, 1)) (hMid : is_midpoint mid mid A B) :
  ∃ k : ℝ, k = -1 / 4 ∧ (∀ x y : ℝ, y - 1 = k * (x - 1) ↔ ((x, y) = (p : ℝ × ℝ) := (mid)) :=
begin
  sorry
end

end chord_line_midpoint_l535_535909


namespace first_discount_percentage_l535_535720

variable (x : ℝ)

-- Hypotheses: listed price, final price, and second discount
def listed_price : ℝ := 510
def final_price : ℝ := 381.48
def second_discount : ℝ := 0.15

-- Expression for the price after first discount
def price_after_first_discount (x : ℝ) := listed_price * (1 - x / 100)

-- Expression for the final price after both discounts
def final_price_expression (x : ℝ) := price_after_first_discount x * (1 - second_discount)

-- Theorem to prove the first discount percentage is 12%
theorem first_discount_percentage :
  final_price_expression x = final_price -> x = 12 :=
by
  simp [price_after_first_discount, final_price_expression]
  sorry

end first_discount_percentage_l535_535720


namespace swimming_speed_eq_l535_535356

theorem swimming_speed_eq (S R H : ℝ) (h1 : R = 9) (h2 : H = 5) (h3 : H = (2 * S * R) / (S + R)) :
  S = 45 / 13 :=
by
  sorry

end swimming_speed_eq_l535_535356


namespace find_square_sum_l535_535358

variable (a b c : ℝ)

def arithmetic_mean_condition : Prop := (a + b + c) / 3 = 9
def geometric_mean_condition : Prop := Real.cbrt (a * b * c) = 6
def harmonic_mean_condition : Prop := 3 / ((1 / a) + (1 / b) + (1 / c)) = 4

theorem find_square_sum (ha : arithmetic_mean_condition a b c) (hg : geometric_mean_condition a b c) (hh : harmonic_mean_condition a b c) : 
  a^2 + b^2 + c^2 = 405 := 
by 
  sorry

end find_square_sum_l535_535358


namespace hyperbola_eccentricity_l535_535575

theorem hyperbola_eccentricity 
  (a b k m : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (A B N : ℝ × ℝ) 
  (h1 : N.1 = (k * m / (1 - k^2)))
  (h2 : A.1 = (-a * m / (k * a + b)))
  (h3 : B.1 = (a * m / (b - k * a)))
  (hm : N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  sqrt (1 + b^2 / a^2) = sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l535_535575


namespace total_votes_cast_l535_535974

theorem total_votes_cast (total_votes : ℕ) (brenda_votes : ℕ) (percentage_brenda : ℚ) 
  (h1 : brenda_votes = 40) (h2 : percentage_brenda = 0.25) 
  (h3 : brenda_votes = percentage_brenda * total_votes) : total_votes = 160 := 
by sorry

end total_votes_cast_l535_535974


namespace probability_ge_two_l535_535064

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_ge_two :
  let p := 0.6
  let n := 3
  let X (k : ℕ) := binomial_probability n k p
  in X 2 + X 3 = 81 / 125 :=
by sorry

end probability_ge_two_l535_535064


namespace malvina_card_sum_l535_535992

theorem malvina_card_sum :
  (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ 
  (MalvinaVal = sin x ∨ MalvinaVal = cos x ∨ MalvinaVal = tan x) ∧ 
  (PaulinaVal = cos x ∨ PaulinaVal = tan x ∨ PaulinaVal = sin x) ∧ 
  (GeorginaVal = cos x ∨ GeorginaVal = tan x ∨ GeorginaVal = sin x) ∧ 
  (PaulinaVal = GeorginaVal) ∧ (MalvinaVal ≠ PaulinaVal)) →
  MalvinaVal + (if MalvinaVal = cos x then sin x else cos x) + (if MalvinaVal = cos x then tan x else sin x) = (1 + Real.sqrt 5) / 2 :=
sorry

end malvina_card_sum_l535_535992


namespace volume_after_increase_l535_535470

variable (l w h : ℕ)
variable (V S E : ℕ)

noncomputable def original_volume : ℕ := l * w * h
noncomputable def surface_sum : ℕ := (l * w) + (w * h) + (h * l)
noncomputable def edge_sum : ℕ := l + w + h

theorem volume_after_increase (h_volume : original_volume l w h = 5400)
  (h_surface : surface_sum l w h = 1176)
  (h_edge : edge_sum l w h = 60) : 
  (l + 1) * (w + 1) * (h + 1) = 6637 := sorry

end volume_after_increase_l535_535470


namespace functional_equation_solution_l535_535126

noncomputable def function_nat_nat (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, f (x + y) = f x + f y

theorem functional_equation_solution :
  ∀ f : ℕ → ℕ, function_nat_nat f → ∃ a : ℕ, ∀ x : ℕ, f x = a * x :=
by
  sorry

end functional_equation_solution_l535_535126


namespace fish_tank_capacity_l535_535095

theorem fish_tank_capacity :
  (∃ (r : ℕ) (t : ℕ) (additional : ℕ), r = 1 / 20 ∧ t = 6 * 60 ∧ additional = 32 ∧
    (t / 20) + additional = 50) :=
begin
  sorry
end

end fish_tank_capacity_l535_535095


namespace max_sum_of_squares_of_sides_l535_535450

theorem max_sum_of_squares_of_sides (R : ℝ) (A B C : Point) 
  (h_circle : Circle R (Triangle Inscribed A B C)) : 
  ∃ (ABC : Triangle A B C), 
    (AB² + BC² + CA² ≤ 9 * R²) 
    ∧ (AB² + BC² + CA² = 9 * R² → Equilateral ABC) := sorry

end max_sum_of_squares_of_sides_l535_535450


namespace marble_probability_calculation_l535_535344

noncomputable def probability_five_blue_marbles :=
  let blue_prob := 12 / 20   -- probability of drawing a blue marble
  let red_prob := 8 / 20     -- probability of drawing a red marble
  let specific_sequence_prob := (blue_prob ^ 5) * (red_prob ^ 3)
  let combinations := Nat.choose 8 5
  let total_probability := combinations * specific_sequence_prob
  total_probability

theorem marble_probability_calculation :
  (probability_five_blue_marbles).round = 0.279 := sorry

end marble_probability_calculation_l535_535344


namespace petya_wins_or_not_l535_535824

theorem petya_wins_or_not (n : ℕ) (h : n ≥ 3) : (n ≠ 4) ↔ petya_wins n :=
begin
  sorry
end

end petya_wins_or_not_l535_535824


namespace cosine_angle_AG_BC_l535_535009

variable (A B C D G : Type)
variable [inner_product_space ℝ A]

-- Define the relevant points or vectors
variables (AB AG BC CA : A)
variables (D : A)
variables (AD DG : A)

-- Define the relevant properties
axiom midpoint_D : D = (1/2) • AG
axiom AB_eq_1 : ∥AB∥ = 1
axiom AG_eq_1 : ∥AG∥ = 1
axiom BC_eq_10 : ∥BC∥ = 10
axiom CA_eq_sqrt51 : ∥CA∥ = real.sqrt 51
axiom dot_product_condition : ⟪AB, AD⟫ + ⟪CA, AG⟫ = -2

-- Define the cosine calculation
noncomputable def cosine_AG_BC : ℝ := 
let φ := real.angle AG BC in
real.cos φ

-- The theorem to prove
theorem cosine_angle_AG_BC :
  cosine_AG_BC A B C D G AB AG BC CA AD DG = -2 / 5 :=
sorry

end cosine_angle_AG_BC_l535_535009


namespace probability_X_equals_one_l535_535258

noncomputable def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_X_equals_one (n : ℕ) (p : ℝ) (h1 : p = 0.6) (h2 : E : ℝ := n * p) (h3 : E = 3) :
  binomial_pmf n p 1 = 3 * (0.4 ^ 4) :=
begin
  sorry
end

end probability_X_equals_one_l535_535258


namespace tabitha_current_age_l535_535120

noncomputable def tabithaAge (currentColors : ℕ) (yearsPassed : ℕ) (startAge : ℕ) (futureYears : ℕ) (futureColors : ℕ) : Prop :=
  (currentColors = (futureColors - futureYears)) ∧
  (yearsPassed = (currentColors - 2)) ∧
  (yearsPassed + startAge = 18)

theorem tabitha_current_age : tabithaAge 5 3 15 3 8 := 
by
  unfold tabithaAge
  split
  all_goals {simp}
  sorry

end tabitha_current_age_l535_535120


namespace triangle_area_formed_by_lines_l535_535487

def line1 := { p : ℝ × ℝ | p.2 = p.1 - 4 }
def line2 := { p : ℝ × ℝ | p.2 = -p.1 - 4 }
def x_axis := { p : ℝ × ℝ | p.2 = 0 }

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_formed_by_lines :
  ∃ (A B C : ℝ × ℝ), A ∈ line1 ∧ A ∈ line2 ∧ B ∈ line1 ∧ B ∈ x_axis ∧ C ∈ line2 ∧ C ∈ x_axis ∧ 
  triangle_area A B C = 8 :=
by
  sorry

end triangle_area_formed_by_lines_l535_535487


namespace number_of_schools_l535_535626

-- Define the conditions
def is_median (a : ℕ) (n : ℕ) : Prop := 2 * a - 1 = n
def high_team_score (a b c : ℕ) : Prop := a > b ∧ a > c
def ranks (b c : ℕ) : Prop := b = 39 ∧ c = 67

-- Define the main problem
theorem number_of_schools (a n b c : ℕ) :
  is_median a n →
  high_team_score a b c →
  ranks b c →
  34 ≤ a ∧ a < 39 →
  2 * a ≡ 1 [MOD 3] →
  (n = 67 → a = 35) →
  (∀ m : ℕ, n = 3 * m + 1) →
  m = 23 :=
by
  sorry

end number_of_schools_l535_535626


namespace mila_total_distance_l535_535681

/-- Mila's car consumes a gallon of gas every 40 miles, her full gas tank holds 16 gallons, starting with a full tank, she drove 400 miles, then refueled with 10 gallons, 
and upon arriving at her destination her gas tank was a third full.
Prove that the total distance Mila drove that day is 826 miles. -/
theorem mila_total_distance (consumption_per_mile : ℝ) (tank_capacity : ℝ) (initial_drive : ℝ) (refuel_amount : ℝ) (final_fraction : ℝ)
  (consumption_per_mile_def : consumption_per_mile = 1 / 40)
  (tank_capacity_def : tank_capacity = 16)
  (initial_drive_def : initial_drive = 400)
  (refuel_amount_def : refuel_amount = 10)
  (final_fraction_def : final_fraction = 1 / 3) :
  ∃ total_distance : ℝ, total_distance = 826 :=
by
  sorry

end mila_total_distance_l535_535681


namespace find_pairs_l535_535851

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
by
  sorry

end find_pairs_l535_535851


namespace problem_one_problem_two_problem_three_l535_535832

-- 1. Prove that the given expression evaluates to 1.
theorem problem_one : (-1)^2 + (Real.pi - 3.14)^0 + 2 * Real.sin (Float.pi / 3) + abs (1 - Real.sqrt 3) - Real.sqrt 12 = 1 :=
by sorry

-- 2. Prove that the solutions to the quadratic equation are 1 and 2.
theorem problem_two (x : ℝ) : (x - 2) * (x - 1) = 0 ↔ x = 1 ∨ x = 2 :=
by sorry

-- 3. Prove that for x = 1, the given expression evaluates to -1.
theorem problem_three (x : ℝ) (h₁ : x ≤ 2) (h₂ : x ≠ 0) (h₃ : x ≠ 2) : 
  (x + 2 + 4 / (x - 2)) / (x^3 / (x^2 - 4 * x + 4)) = -1 :=
by 
  have h₄ : x = 1 := sorry 
  rw h₄
  sorry

end problem_one_problem_two_problem_three_l535_535832


namespace value_of_b_l535_535878

noncomputable def function_bounds := 
  ∃ (k b : ℝ), (∀ (x : ℝ), (-3 ≤ x ∧ x ≤ 1) → (-1 ≤ k * x + b ∧ k * x + b ≤ 8)) ∧ (b = 5 / 4 ∨ b = 23 / 4)

theorem value_of_b : function_bounds :=
by
  sorry

end value_of_b_l535_535878


namespace meeting_point_2015th_l535_535166

-- Definitions for the conditions
def motorist_speed (x : ℝ) := x
def cyclist_speed (y : ℝ) := y
def initial_delay (t : ℝ) := t 
def first_meeting_point := C
def second_meeting_point := D

-- The main proof problem statement
theorem meeting_point_2015th
  (x y t : ℝ) -- speeds of the motorist and cyclist and the initial delay
  (C D : Point) -- points C and D on the segment AB where meetings occur
  (pattern_alternation : ∀ n: ℤ, n > 0 → ((n % 2 = 1) → n-th_meeting_point = C) ∧ ((n % 2 = 0) → n-th_meeting_point = D))
  (P_A_B_cycle : ∀ n: ℕ, (P → A ∨ B → C ∨ A → B ∨ D → P) holds for each meeting): 
  2015-th_meeting_point = C :=
by
  sorry

end meeting_point_2015th_l535_535166


namespace faster_train_speed_l535_535012

theorem faster_train_speed (dist_between_stations : ℕ) (extra_distance : ℕ) (slower_speed : ℕ) 
  (dist_between_stations_eq : dist_between_stations = 444)
  (extra_distance_eq : extra_distance = 60) 
  (slower_speed_eq : slower_speed = 16) :
  ∃ (faster_speed : ℕ), faster_speed = 21 := by
  sorry

end faster_train_speed_l535_535012


namespace unique_line_through_point_and_equal_intercepts_l535_535732

theorem unique_line_through_point_and_equal_intercepts :
  ∃! l : ℝ → ℝ → Prop, -- there exists a unique line
  (∃ b : ℝ, ∃ a : ℝ, l = λ x y, y = (-1/a)*x + b ∧ a = b) ∧ -- which has equal x and y intercepts
  l 0 5 := -- and passes through point (0, 5)
sorry

end unique_line_through_point_and_equal_intercepts_l535_535732


namespace max_turtles_move_indefinitely_l535_535976

-- Definitions based on the problem conditions
def board_size : Type := (fin 101) × (fin 99)
def is_adjacent (a b : board_size) : Prop :=
  (a.1 = b.1 ∧ (a.2 + 1 = b.2 ∨ a.2 = b.2 + 1)) ∨ 
  (a.2 = b.2 ∧ (a.1 + 1 = b.1 ∨ a.1 = b.1 + 1))

def turtle_moves (pos time : board_size → nat) : Prop :=
  ∀ t, ∃ a b : board_size, is_adjacent (pos t) (pos (t + 1)) ∧ 
                             ((pos t).1 ≠ (pos (t + 1)).1 ∨ (pos t).2 ≠ (pos (t + 1)).2)

-- Statement of the problem to prove
theorem max_turtles_move_indefinitely : 
  ∃ (pos : nat → board_size → nat), 
  (∀ t, ∃ a : fin 101, ∃ b : fin 99, (pos t (a, b) = a ∧ pos t (a, b) = b)) ∧
  (∀ t, ∀ a b, (pos t (a, b) ≠ pos t (a + 1, b)) ∧ (pos t (a, b) ≠ pos t (a, b + 1))) →
  (∃ n : nat, 0 < n ∧ n ≤ 9800) := 
sorry

end max_turtles_move_indefinitely_l535_535976


namespace quadrilateral_similarity_bisector_intersection_l535_535183

namespace Quadrilaterals

-- Define the setup for cyclic quadrilateral ABCD
variables {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α]
variables (A B C D : α) (AC BD : α)

-- Conditions: ABCD is a cyclic quadrilateral
def cyclic_quadrilateral (A B C D : α) : Prop := sorry

-- Perpendiculars dropped from vertices to diagonals AC and BD
def perpendicular (P Q R : α) : Prop := sorry

-- The feet of the perpendiculars from A, B, C, D to AC and BD
variables (A1 B1 C1 D1 : α)

-- Proving the similarity of quadrilaterals A1B1C1D1 and ABCD
theorem quadrilateral_similarity 
  (h_cyclic : cyclic_quadrilateral A B C D)
  (h_perp_A : perpendicular A A1 BD)
  (h_perp_B : perpendicular B B1 BD)
  (h_perp_C : perpendicular C C1 AC)
  (h_perp_D : perpendicular D D1 AC) :
  similar_quadrilateral A1 B1 C1 D1 A B C D := 
sorry

-- Proving the diagonal intersection property for A1B1C1D1
theorem bisector_intersection
  (h_cyclic : cyclic_quadrilateral A B C D)
  (h_perp_A : perpendicular A A1 BD)
  (h_perp_B : perpendicular B B1 BD)
  (h_perp_C : perpendicular C C1 AC)
  (h_perp_D : perpendicular D D1 AC)
  (h_similar : similar_quadrilateral A1 B1 C1 D1 A B C D) :
  diagonal_bisectors_pass_intersections A1 B1 C1 D1 A B C D :=
sorry

end Quadrilaterals

end quadrilateral_similarity_bisector_intersection_l535_535183


namespace find_angle_C_find_perimeter_l535_535271

-- Definitions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Opposite sides to angles A, B, C respectively
variables (acute : A < π/2 ∧ B < π/2 ∧ C < π/2) -- Acute triangle condition
variables (eq1 : sqrt 3 * a = 2 * c * sin A) -- Given condition
variables (c_val : c = sqrt 7) (ab_val: a * b = 6) -- Given values for Part 2

-- Proof statements
theorem find_angle_C (h : acute) (h1 : eq1) : C = π / 3 :=
sorry

theorem find_perimeter (h : acute) (h1 : eq1) (h2 : c_val) (h3 : ab_val) : a + b + c = 5 + sqrt 7 :=
sorry

end find_angle_C_find_perimeter_l535_535271


namespace first_group_people_count_l535_535352

theorem first_group_people_count (P : ℕ) (W : ℕ) 
  (h1 : P * 3 * W = 3 * W) 
  (h2 : 8 * 3 * W = 8 * W) : 
  P = 3 :=
by
  sorry

end first_group_people_count_l535_535352


namespace terminating_fraction_count_l535_535148

theorem terminating_fraction_count :
  (∃ n_values : Finset ℕ, (∀ n ∈ n_values, 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = k * 49)) ∧ n_values.card = 10) :=
by
  -- Placeholder for the proof, does not contribute to the conditions-direct definitions.
  sorry

end terminating_fraction_count_l535_535148


namespace find_expression_l535_535036

theorem find_expression (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end find_expression_l535_535036


namespace cesaro_sum_extension_l535_535538

noncomputable def cesaro_sum (B : List ℝ) (n : ℕ) : ℝ := 
  (List.sum (List.scanl (+) 0 B).drop 1) / n

theorem cesaro_sum_extension (B : List ℝ) (hB_len : B.length = 150) (hB_cesaro : cesaro_sum B 150 = 1200) :
  cesaro_sum (2 :: B) 151 = 1194 := by
  -- Proof omitted
  sorry

end cesaro_sum_extension_l535_535538


namespace problem_statement_l535_535234

variables {a b c x y z : ℝ}

-- conditions
axiom h1 : x = a / (2 * b + 3 * c)
axiom h2 : y = 2 * b / (3 * c + a)
axiom h3 : z = 3 * c / (a + 2 * b)

-- theorem to prove
theorem problem_statement : (x / (1 + x)) + (y / (1 + y)) + (z / (1 + z)) = 1 :=
by {
    rw [h1, h2, h3],
    -- Add detailed calculations directly from the solution
    have hx := a / (2 * b + 3 * c),
    have hy := 2 * b / (3 * c + a),
    have hz := 3 * c / (a + 2 * b),
    rw [hx, hy, hz],
    simp,
    sorry
}

end problem_statement_l535_535234


namespace terminating_fraction_count_l535_535149

theorem terminating_fraction_count :
  (∃ n_values : Finset ℕ, (∀ n ∈ n_values, 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = k * 49)) ∧ n_values.card = 10) :=
by
  -- Placeholder for the proof, does not contribute to the conditions-direct definitions.
  sorry

end terminating_fraction_count_l535_535149


namespace ratio_proof_l535_535446

-- Define the conditions
def ratio_condition (x : ℕ) : Prop :=
  15 * 10 = x

-- Lean statement that expresses the proof problem
theorem ratio_proof : ∃ x : ℕ, ratio_condition x ∧ x = 150 :=
by
  use 150
  split
  · exact rfl
  · exact rfl
  sorry

end ratio_proof_l535_535446


namespace collinear_P_E_F_l535_535089

open_locale classical

variables {α : Type*} [plane_geometry α]

-- Given definitions and assumptions
variables (A B C D P Q E F : α)
variable (circle : set α)
variable (inscribed : cyclic_quadrilateral A B C D circle)
variable (tangent1 : tangent Q E circle)
variable (tangent2 : tangent Q F circle)
variable (intersection1 : intersection_line_extension A D Q (line A D))
variable (intersection2 : intersection_line_extension B C Q (line B C))
variable (intersection3 : intersection_line_extension A B P (line A B))
variable (intersection4 : intersection_line_extension C D P (line C D))

-- Required to prove
theorem collinear_P_E_F
    (inscribed : cyclic_quadrilateral A B C D circle)
    (tangent1 : tangent Q E circle)
    (tangent2 : tangent Q F circle)
    (intersection1 : intersection_line_extension A D Q (line A D))
    (intersection2 : intersection_line_extension B C Q (line B C))
    (intersection3 : intersection_line_extension A B P (line A B))
    (intersection4 : intersection_line_extension C D P (line C D)) :
  collinear P E F :=
sorry

end collinear_P_E_F_l535_535089


namespace min_distance_between_curves_l535_535529

theorem min_distance_between_curves :
  let f₁ : ℝ → ℝ := λ x, exp (3 * x + 5)
  let f₂ : ℝ → ℝ := λ x, (log x - 5) / 3
  let ρ : ℝ → ℝ := λ x, Real.sqrt 2 * (abs (exp (3 * x + 5) - x))
  ρ ((- log 3 - 5) / 3) = Real.sqrt 2 * (2 + log 3 / 3) :=
by
  sorry

end min_distance_between_curves_l535_535529


namespace alberto_spent_more_l535_535808

-- Define the expenses of Alberto and Samara
def alberto_expenses : ℕ := 2457
def samara_oil_expense : ℕ := 25
def samara_tire_expense : ℕ := 467
def samara_detailing_expense : ℕ := 79
def samara_total_expenses : ℕ := samara_oil_expense + samara_tire_expense + samara_detailing_expense

-- State the theorem to prove the difference in expenses
theorem alberto_spent_more :
  alberto_expenses - samara_total_expenses = 1886 := by
  sorry

end alberto_spent_more_l535_535808


namespace graph_passes_through_point_l535_535860

theorem graph_passes_through_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : 
  f (1 : ℝ) = 4 :=
by
  -- Define the function f(x)
  let f (x : ℝ) := a^(x - 1) + 3
  -- Check if f(1) = 4 given the conditions
  calc f 1 = a^(1 - 1) + 3 : by rfl
       ... = a^0 + 3 : by simp
       ... = 1 + 3 : by simp [h₁]
       ... = 4 : by rfl

end graph_passes_through_point_l535_535860


namespace ratio_of_areas_l535_535011

noncomputable def circumferences_equal_arcs (C1 C2 : ℝ) (k1 k2 : ℕ) : Prop :=
  (k1 : ℝ) / 360 * C1 = (k2 : ℝ) / 360 * C2

theorem ratio_of_areas (C1 C2 : ℝ) (h : circumferences_equal_arcs C1 C2 60 30) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_l535_535011


namespace asymptote_equations_of_C2_l535_535207

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

def e1 : ℝ := sqrt (1 - (b^2 / a^2))
def e2 : ℝ := sqrt (1 + (b^2 / a^2))

def condition1 := a > b
def condition2 := (e1 * e2) = sqrt(3) / 2

theorem asymptote_equations_of_C2 :
  condition1 ∧ condition2 →
  ∀ x y : ℝ, (x + sqrt 2 * y = 0) ∨ (x - sqrt 2 * y = 0) :=
by
  intros h x y
  sorry

end asymptote_equations_of_C2_l535_535207


namespace mass_of_body_l535_535133

def delta (M : (ℝ × ℝ × ℝ)) : ℝ := M.2

def region_G (p : (ℝ × ℝ × ℝ)) : Prop :=
(p.1 ^ 2 = 2 * p.2) ∧ (1 - p.2 ≤ p.3) ∧ (p.3 ≤ 2 - 2 * p.2)

theorem mass_of_body :
  (∫ x in -sqrt(2) .. sqrt(2), ∫ y in 0 .. 1, ∫ z in 1 - y .. 2 - 2 * y, y) = 8 * sqrt(2) / 35 :=
by sorry

end mass_of_body_l535_535133


namespace find_angle_C_find_perimeter_l535_535272

-- Definitions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Opposite sides to angles A, B, C respectively
variables (acute : A < π/2 ∧ B < π/2 ∧ C < π/2) -- Acute triangle condition
variables (eq1 : sqrt 3 * a = 2 * c * sin A) -- Given condition
variables (c_val : c = sqrt 7) (ab_val: a * b = 6) -- Given values for Part 2

-- Proof statements
theorem find_angle_C (h : acute) (h1 : eq1) : C = π / 3 :=
sorry

theorem find_perimeter (h : acute) (h1 : eq1) (h2 : c_val) (h3 : ab_val) : a + b + c = 5 + sqrt 7 :=
sorry

end find_angle_C_find_perimeter_l535_535272


namespace percentage_with_diploma_l535_535625

-- Define the percentages as variables for clarity
def low_income_perc := 0.25
def lower_middle_income_perc := 0.35
def upper_middle_income_perc := 0.25
def high_income_perc := 0.15

def low_income_diploma := 0.05
def lower_middle_income_diploma := 0.35
def upper_middle_income_diploma := 0.60
def high_income_diploma := 0.80

theorem percentage_with_diploma :
  (low_income_perc * low_income_diploma +
   lower_middle_income_perc * lower_middle_income_diploma +
   upper_middle_income_perc * upper_middle_income_diploma +
   high_income_perc * high_income_diploma) = 0.405 :=
by sorry

end percentage_with_diploma_l535_535625


namespace construct_triangle_l535_535104

noncomputable def reconstruct_triangle (D E F : ℝ×ℝ) : ℝ×ℝ × ℝ×ℝ × ℝ×ℝ :=
sorry

theorem construct_triangle (D E F : ℝ×ℝ) (hD : Is_vertex D ABC) (hE : Is_vertex E ABC) (hF : Is_vertex F ABC) :
  ∃ A B C : ℝ×ℝ, reconstruct_triangle D E F = (A, B, C) :=
sorry

end construct_triangle_l535_535104


namespace calculate_average_age_l535_535973

variables (k : ℕ) (female_to_male_ratio : ℚ) (avg_young_female : ℚ) (avg_old_female : ℚ) (avg_young_male : ℚ) (avg_old_male : ℚ)

theorem calculate_average_age 
  (h_ratio : female_to_male_ratio = 7/8)
  (h_avg_yf : avg_young_female = 26)
  (h_avg_of : avg_old_female = 42)
  (h_avg_ym : avg_young_male = 28)
  (h_avg_om : avg_old_male = 46) : 
  (534/15 : ℚ) = 36 :=
by sorry

end calculate_average_age_l535_535973


namespace strip_width_l535_535066

theorem strip_width (w : ℝ) (h_floor : ℝ := 10) (b_floor : ℝ := 8) (area_rug : ℝ := 24) :
  (h_floor - 2 * w) * (b_floor - 2 * w) = area_rug → w = 2 := 
by 
  sorry

end strip_width_l535_535066


namespace tim_kittens_count_l535_535738

def initial_kittens : Nat := 6
def kittens_given_to_jessica : Nat := 3
def kittens_received_from_sara : Nat := 9

theorem tim_kittens_count : initial_kittens - kittens_given_to_jessica + kittens_received_from_sara = 12 :=
by
  sorry

end tim_kittens_count_l535_535738


namespace arithmetic_sequence_middle_term_l535_535279

theorem arithmetic_sequence_middle_term (a b c d e : ℕ) (seq : List ℕ)
  (h_seq : seq = [9, d, e, f, 63])
  (h_arith : ∀ (i : ℕ), i < seq.length - 1 → seq[i+1] - seq[i] = seq[1] - seq[0]) :
  e = 36 :=
by
  -- Arithmetic sequence: the difference between consecutive terms is constant
  -- Given sequence: [9, d, e, f, 63]
  -- Middle term e for arithmetic sequence of 5 terms is average of first and last terms
  calc
    e = (seq.head! + seq.getLast! ?) / 2 : by sorry
    e = (9 + 63) / 2 : by sorry
    e = 36 : by sorry

end arithmetic_sequence_middle_term_l535_535279


namespace find_f_2015_l535_535547

variables (f : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 6) = f x + f 3

theorem find_f_2015 (h1 : is_even_function f) (h2 : satisfies_condition f) (h3 : f 1 = 2) : f 2015 = 2 :=
by
  sorry

end find_f_2015_l535_535547


namespace greatest_distance_between_centers_of_circles_in_rectangle_l535_535744

theorem greatest_distance_between_centers_of_circles_in_rectangle
  (w h : ℝ) (d : ℝ) (r : ℝ) (c1 : 2 * r = d)
  (c2 : 0 < r) (c3 : r + r = d)
  (c4 : w > d) (c5 : h > d)
  (c6 : w = 17) (c7 : h = 15) (c8 : d = 7) :
  let x_dist := w - d,
      y_dist := h - d in
  (x_dist^2 + y_dist^2) = 164 :=
sorry

end greatest_distance_between_centers_of_circles_in_rectangle_l535_535744


namespace MeasureAngleBDF_l535_535636

noncomputable def measure_angle_BDF (A B C D E F : Type*) 
  (angle_ADE angle_DEF angle_EDF : ℝ) : ℝ :=
  if angle_ADE = 50 ∧ angle_DEF = 30 ∧ angle_EDF = 40 
  then 40 else 0

theorem MeasureAngleBDF
  (A B C D E F : Type*)
  (hADE : ∠ ADE = 50)
  (hDEF : ∠ DEF = 30)
  (hEDF : ∠ EDF = 40) :
  ∠ BDF = measure_angle_BDF A B C D E F 50 30 40 :=
by {
  rw measure_angle_BDF,
  split_ifs,
  repeat {exact rfl},
  sorry
}

end MeasureAngleBDF_l535_535636


namespace tangent_line_at_pi_over_4_l535_535531

noncomputable def tangent_eq (x y : ℝ) : Prop :=
  y = 2 * x * Real.tan x

noncomputable def tangent_line_eq (x y : ℝ) : Prop :=
  (2 + Real.pi) * x - y - (Real.pi^2 / 4) = 0

theorem tangent_line_at_pi_over_4 :
  tangent_eq (Real.pi / 4) (Real.pi / 2) →
  tangent_line_eq (Real.pi / 4) (Real.pi / 2) :=
by
  sorry

end tangent_line_at_pi_over_4_l535_535531


namespace imaginary_part_i_mul_neg1_add_2i_l535_535367

def imaginary_part (z : ℂ) : ℂ :=
  complex.im z

theorem imaginary_part_i_mul_neg1_add_2i :
  imaginary_part (i * (-1 + 2 * i)) = -1 := by
  sorry

end imaginary_part_i_mul_neg1_add_2i_l535_535367


namespace problem1_problem2_l535_535768

-- Proof of the first problem statement
theorem problem1 : (8^(2/3) + (16/81)^(-3/4) - ((sqrt 2) - 1)^0) = 51/8 :=
  sorry

-- Proof of the second problem statement
theorem problem2 : (9^(Real.log 2 / Real.log 9) + 1/3 * Real.logBase 6 8 - 2 * Real.logBase 6⁻¹ (sqrt 3)) = 3 :=
  sorry

end problem1_problem2_l535_535768


namespace unit_digit_k_is_3_5_or_7_l535_535607

noncomputable def proof_unit_digit_k : Prop :=
  ∃ k : ℤ, ∃ a : ℝ,
    k > 1 ∧ (a^2 - k * a + 1 = 0) ∧ 
    (∀ n : ℕ, n > 10 → ((a^2)^n + (a^(-2))^n) % 10 = 7) ∧
    ((k % 10 = 3) ∨ (k % 10 = 5) ∨ (k % 10 = 7))

theorem unit_digit_k_is_3_5_or_7 (h : proof_unit_digit_k) : 
  ∃ k : ℤ, (k % 10 = 3) ∨ (k % 10 = 5) ∨ (k % 10 = 7) :=
sorry

end unit_digit_k_is_3_5_or_7_l535_535607


namespace measure_of_angle_y_l535_535628

noncomputable theory

open Classical

-- Define the given angles and lines conditions
def parallel (m n : ℝ) : Prop := sorry  -- definition for parallel lines
def angle_P_AQ := 50
def angle_P_BQ := 40
def angle_Q_PB := 90

-- State the theorem to prove
theorem measure_of_angle_y 
  (m n : ℝ) 
  (P Q : ℝ) 
  (H1 : parallel m n)
  (H2 : angle_P_AQ = 50)
  (H3 : angle_P_BQ = 40)
  (H4 : angle_Q_PB = 90):
  ∃y, y = 140 := 
sorry

end measure_of_angle_y_l535_535628


namespace quadrilateral_area_l535_535413

def vertex1 : ℝ × ℝ := (2, 1)
def vertex2 : ℝ × ℝ := (4, 3)
def vertex3 : ℝ × ℝ := (7, 1)
def vertex4 : ℝ × ℝ := (4, 6)

noncomputable def shoelace_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) -
       (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1)) / 2

theorem quadrilateral_area :
  shoelace_area vertex1 vertex2 vertex3 vertex4 = 7.5 :=
by
  sorry

end quadrilateral_area_l535_535413


namespace coordinates_of_B_l535_535979

theorem coordinates_of_B (x1 y1 : ℤ) (d : ℤ) 
    (h_parallel : ∀ x2 y2, AB_parallel_x x1 y1 x2 y2) 
    (h_distance : AB_distance x1 y1 x2 y2 = d) :
    (y1 = -3) → (x1 = 1) → d = 2 → 
    (x2 = 3 ∧ y2 = -3) ∨ (x2 = -1 ∧ y2 = -3) := 
by
  sorry

-- Auxiliary definitions
def AB_parallel_x (x1 y1 x2 y2 : ℤ) : Prop := y1 = y2

def AB_distance (x1 y1 x2 y2 : ℤ) : ℤ := abs (x2 - x1)

end coordinates_of_B_l535_535979


namespace smallest_among_l535_535085

theorem smallest_among {a b c d : ℝ} (h1 : a = Real.pi) (h2 : b = -2) (h3 : c = 0) (h4 : d = -1) : 
  ∃ (x : ℝ), x = b ∧ x < a ∧ x < c ∧ x < d := 
by {
  sorry
}

end smallest_among_l535_535085


namespace max_quarters_l535_535334

/-- Prove that given the conditions for the number of nickels, dimes, and quarters,
    the maximum number of quarters can be 20. --/
theorem max_quarters {a b c : ℕ} (h1 : a + b + c = 120) (h2 : 5 * a + 10 * b + 25 * c = 1000) :
  c ≤ 20 :=
sorry

end max_quarters_l535_535334


namespace part_I_part_II_l535_535565

variables (a b : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1)

-- Part I
theorem part_I (h : ‖a - 2 • b‖ = 2) : ‖a - b‖ = sqrt(3) / 2 := 
  sorry

-- Part II
theorem part_II (h_angle : real.angle a b = real.pi / 3) (m n : ℝ^3) 
  (hm : m = a + b) (hn : n = a - 3 • b) : 
  real.cos (real.angle m n) = -sqrt(21) / 7 := 
  sorry

end part_I_part_II_l535_535565


namespace intersection_P_Q_equals_P_l535_535496

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := { y | ∃ x ∈ Set.univ, y = Real.cos x }

theorem intersection_P_Q_equals_P : P ∩ Q = P := by
  sorry

end intersection_P_Q_equals_P_l535_535496


namespace differentiable_function_inequality_l535_535210

variable {f : ℝ → ℝ}

theorem differentiable_function_inequality
  (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x, f x > f' x)
  (a : ℝ) (h_a : a > 0) : 
  f a < exp a * f 0 := 
by 
  sorry

end differentiable_function_inequality_l535_535210


namespace laura_change_l535_535998

-- Define the cost of a pair of pants and a shirt.
def cost_of_pants := 54
def cost_of_shirts := 33

-- Define the number of pants and shirts Laura bought.
def num_pants := 2
def num_shirts := 4

-- Define the amount Laura gave to the cashier.
def amount_given := 250

-- Calculate the total cost.
def total_cost := num_pants * cost_of_pants + num_shirts * cost_of_shirts

-- Define the expected change.
def expected_change := 10

-- The main theorem stating the problem and its solution.
theorem laura_change :
  amount_given - total_cost = expected_change :=
by
  sorry

end laura_change_l535_535998


namespace mean_median_arithmetic_sequence_l535_535070

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := 
  a₁ + (n - 1) * d

def is_geometric_sequence (a₁ a₃ a₇ : ℝ) : Prop :=
  a₃ / a₁ = a₇ / a₃

theorem mean_median_arithmetic_sequence :
  ∃ (a₁ d : ℝ),
    d ≠ 0 ∧
    arithmetic_sequence a₁ d 3 = 8 ∧
    is_geometric_sequence (arithmetic_sequence a₁ d 1) 
                          (arithmetic_sequence a₁ d 3) 
                          (arithmetic_sequence a₁ d 7) ∧
    let mean := (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 10) / 2 in
    let median := (arithmetic_sequence a₁ d 5 + arithmetic_sequence a₁ d 6) / 2 in
    mean = 13 ∧ median = 13 :=
by {
  sorry
}

end mean_median_arithmetic_sequence_l535_535070


namespace count_valid_numbers_l535_535598

def valid_tens_digits : List ℕ := [1, 7, 9]
def valid_ones_digits : List ℕ := [0, 1, 7, 9]

def is_valid_number (n : ℕ) : Prop :=
  n < 100 ∧
  n / 10 ≠ n % 10 ∧
  (n / 10 ∈ valid_tens_digits) ∧
  (n % 10 ∈ valid_ones_digits)

theorem count_valid_numbers : (Finset.filter is_valid_number (Finset.range 100)).card = 9 := 
by
  sorry

end count_valid_numbers_l535_535598


namespace weekly_earnings_l535_535476

-- Define constants for the problem
def regular_rate : ℝ := 10
def simple_survey_count : ℝ := 30
def moderate_survey_count : ℝ := 20
def complex_survey_count : ℝ := 10
def non_cellphone_survey_count : ℝ := 40

-- Define rates for cellphone surveys
def simple_rate : ℝ := regular_rate * 1.3
def moderate_rate : ℝ := regular_rate * 1.5
def complex_rate : ℝ := regular_rate * 1.75

-- Define bonus categories
def tier_50_74_bonus : ℝ := 100
def tier_75_99_bonus : ℝ := 150
def tier_100_plus_bonus : ℝ := 250

def simple_bonus : ℝ := 50
def moderate_bonus : ℝ := 75
def complex_bonus : ℝ := 125

-- Calculate total earnings from surveys
def non_cellphone_earnings : ℝ := regular_rate * non_cellphone_survey_count
def simple_earnings : ℝ := simple_rate * simple_survey_count
def moderate_earnings : ℝ := moderate_rate * moderate_survey_count
def complex_earnings : ℝ := complex_rate * complex_survey_count

def cellphone_earnings : ℝ := simple_earnings + moderate_earnings + complex_earnings
def total_survey_earnings : ℝ := non_cellphone_earnings + cellphone_earnings

-- Calculate tiered bonus
def tiered_bonus : ℝ :=
  if simple_survey_count + moderate_survey_count + complex_survey_count + non_cellphone_survey_count >= 100 then tier_100_plus_bonus
  else if simple_survey_count + moderate_survey_count + complex_survey_count + non_cellphone_survey_count >= 75 then tier_75_99_bonus
  else if simple_survey_count + moderate_survey_count + complex_survey_count + non_cellphone_survey_count >= 50 then tier_50_74_bonus
  else 0

-- Calculate milestone bonuses
def milestone_bonus : ℝ :=
  (if simple_survey_count >= 25 then simple_bonus else 0) +
  (if moderate_survey_count >= 15 then moderate_bonus else 0) +
  (if complex_survey_count >= 5 then complex_bonus else 0)

-- Calculate total earnings
def total_earnings : ℝ := total_survey_earnings + tiered_bonus + milestone_bonus

-- Prove the worker's earnings for the week is Rs. 1765
theorem weekly_earnings : total_earnings = 1765 := by
  sorry

end weekly_earnings_l535_535476


namespace even_and_increasing_func_l535_535083

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≤ f y

def fA (x : ℝ) : ℝ := (1 / 2) ^ x
def fB (x : ℝ) : ℝ := -x ^ 2
def fC (x : ℝ) : ℝ := Real.log x / Real.log 2
def fD (x : ℝ) : ℝ := abs x + 1

theorem even_and_increasing_func :
  (is_even fD ∧ is_monotonically_increasing fD {x : ℝ | 0 < x}) ∧
  ¬ (is_even fA ∧ is_monotonically_increasing fA {x : ℝ | 0 < x}) ∧
  ¬ (is_even fB ∧ is_monotonically_increasing fB {x : ℝ | 0 < x}) ∧
  ¬ (is_even fC ∧ is_monotonically_increasing fC {x : ℝ | 0 < x}) :=
by sorry

end even_and_increasing_func_l535_535083


namespace person_speed_is_correct_l535_535063

def distance_meters : ℝ := 300
def time_minutes : ℝ := 4
def distance_kilometers : ℝ := distance_meters / 1000
def time_hours : ℝ := time_minutes / 60
def speed_kmph : ℝ := distance_kilometers / time_hours

theorem person_speed_is_correct : speed_kmph = 4.5 := by
  sorry

end person_speed_is_correct_l535_535063


namespace sum_of_place_values_of_threes_in_63130_l535_535032

def place_value_sum (n : Nat) : Nat :=
  let digits := [6, 3, 1, 3, 0]  -- represents the numeral 63130
  let place_values := [10000, 1000, 100, 10, 1]
  let three_positions := [1, 3]  -- the indices of the digit 3 in the numeral 63130
  three_positions.map (λ i => digits[i] * place_values[i]).sum

theorem sum_of_place_values_of_threes_in_63130 :
  place_value_sum 63130 = 3030 :=
by
  sorry

end sum_of_place_values_of_threes_in_63130_l535_535032


namespace quadratic_root_is_zero_then_m_neg_one_l535_535556

theorem quadratic_root_is_zero_then_m_neg_one (m : ℝ) (h_eq : (m-1) * 0^2 + 2 * 0 + m^2 - 1 = 0) : m = -1 := by
  sorry

end quadratic_root_is_zero_then_m_neg_one_l535_535556


namespace triangle_right_hypotenuse_l535_535222

theorem triangle_right_hypotenuse (c : ℝ) (a : ℝ) (h₀ : c = 4) (h₁ : 0 < a) (h₂ : a^2 + b^2 = c^2) :
  a ≤ 2 * Real.sqrt 2 :=
sorry

end triangle_right_hypotenuse_l535_535222


namespace stratified_sampling_male_athletes_l535_535481

theorem stratified_sampling_male_athletes : 
  ∀ (total_males total_females total_to_sample : ℕ), 
    total_males = 20 → 
    total_females = 10 → 
    total_to_sample = 6 → 
    20 * (total_to_sample / (total_males + total_females)) = 4 :=
by
  intros total_males total_females total_to_sample h_males h_females h_sample
  rw [h_males, h_females, h_sample]
  sorry

end stratified_sampling_male_athletes_l535_535481


namespace correct_expression_l535_535430

theorem correct_expression :
  (∀ x : ℝ, (sqrt 9 ≠ ±3) ∧ (√16 ≠ -4) ∧ (√((-2)^2) ≠ -2)) →
  -cube_root (-8) = 2 :=
by
  sorry

end correct_expression_l535_535430


namespace is_linear_eq_l535_535026

theorem is_linear_eq (x y : ℝ) : (2 * x + 3 = 7) :=
by
  -- Conditions
  have optionA : x + y = 3 := sorry
  have optionB : x * y + 2 = 5 := sorry
  have optionD : x^2 - 3 * x = 4 := sorry
  -- Given equation (Option C)
  exact 2 * x + 3 = 7

end is_linear_eq_l535_535026


namespace relationship_between_abc_l535_535870

section
variables (a b c : ℝ)
noncomputable def a_def := real.log 4 / real.log 3
noncomputable def b_def := (1 / 3) ^ (1 / 3)
noncomputable def c_def := (1 / 3) ^ (1 / 4)

theorem relationship_between_abc : b_def < c_def ∧ c_def < a_def := by
  sorry
end

end relationship_between_abc_l535_535870


namespace correct_number_for_question_mark_l535_535385

def first_row := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 200]
def second_row_no_quest := [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
def question_mark (x : ℕ) := first_row.sum = second_row_no_quest.sum + x

theorem correct_number_for_question_mark : question_mark 155 := 
by sorry -- proof to be completed

end correct_number_for_question_mark_l535_535385


namespace find_fourth_digit_l535_535468

theorem find_fourth_digit (a b c d : ℕ) (h : 0 ≤ a ∧ a < 8 ∧ 0 ≤ b ∧ b < 8 ∧ 0 ≤ c ∧ c < 8 ∧ 0 ≤ d ∧ d < 8)
  (h_eq : 511 * a + 54 * b - 92 * c - 999 * d = 0) : d = 6 :=
by
  sorry

end find_fourth_digit_l535_535468


namespace amount_lent_to_B_l535_535463

variable (P_B : ℝ)
variable (P_C : ℝ := 3000)
variable (r : ℝ := 0.12)
variable (t_B : ℝ := 2)
variable (t_C : ℝ := 4)
variable (total_interest : ℝ := 2640)

theorem amount_lent_to_B : P_B = 5000 :=
  let I_B := P_B * r * t_B
  let I_C := P_C * r * t_C
  have h : I_B + I_C = total_interest, by sorry
  have I_C_val : I_C = 1440, by sorry
  have I_B_val : I_B = total_interest - I_C_val, by sorry
  have P_B_val : P_B = I_B_val / (r * t_B), by sorry
  P_B_val

end amount_lent_to_B_l535_535463


namespace distance_point_to_line_l535_535362

noncomputable def point := (5, -3)
noncomputable def line := (1, 0, 2)  -- Representing the line x + 2 = 0 in Ax + By + C = 0 form

theorem distance_point_to_line (p : ℝ × ℝ) (line : ℝ × ℝ × ℝ) :
  let A := line.1,
      B := line.2,
      C := line.3,
      x1 := p.1,
      y1 := p.2 in
  real.dist (A * x1 + B * y1 + C) / real.sqrt (A^2 + B^2) = 7 :=
by
  sorry

end distance_point_to_line_l535_535362


namespace nonagon_side_length_l535_535359

theorem nonagon_side_length (C : ℕ) (n : ℕ) (h1 : C = 171) (h2 : n = 9) : C / n = 19 := by
  rw [h1, h2]
  norm_num

end nonagon_side_length_l535_535359


namespace count_not_squares_or_cubes_l535_535942

theorem count_not_squares_or_cubes (n : ℕ) : 
  let total := 200 in
  let perfect_squares := 14 in
  let perfect_cubes := 5 in
  let perfect_sixth_powers := 2 in
  let squares_or_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers in
  let count_not_squares_or_cubes := total - squares_or_cubes in
  n = count_not_squares_or_cubes :=
by
  let total := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let perfect_sixth_powers := 2
  let squares_or_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let count_not_squares_or_cubes := total - squares_or_cubes
  show _ from sorry

end count_not_squares_or_cubes_l535_535942


namespace first_group_men_l535_535611

theorem first_group_men (M : ℕ) (h : M * 15 = 25 * 24) : M = 40 := sorry

end first_group_men_l535_535611


namespace microbrewery_output_increase_l535_535685

theorem microbrewery_output_increase (O H : ℝ) (hO : O > 0) (hH : H > 0) :
  let new_output := 1.2 * O
  let new_hours := 0.7 * H
  let initial_output_per_hour := O / H
  let new_output_per_hour := new_output / new_hours
  (new_output_per_hour / initial_output_per_hour - 1) * 100 ≈ 71.43 :=
by
  sorry

end microbrewery_output_increase_l535_535685


namespace carpet_cost_proof_l535_535442

noncomputable def total_carpet_cost (floor_length : ℕ) (floor_width : ℕ) (carpet_length : ℕ) (carpet_width : ℕ) (carpet_cost : ℕ) : ℕ :=
  let area_floor := floor_length * floor_width
  let area_carpet_square := carpet_length * carpet_width
  let number_of_carpet_squares := area_floor / area_carpet_square
  number_of_carpet_squares * carpet_cost

theorem carpet_cost_proof (h1 : floor_length = 6) (h2 : floor_width = 10) (h3 : carpet_length = 2) (h4 : carpet_width = 2) (h5 : carpet_cost = 15) :
  total_carpet_cost floor_length floor_width carpet_length carpet_width carpet_cost = 225 :=
by
  simp [total_carpet_cost, h1, h2, h3, h4, h5]
  apply sorry

end carpet_cost_proof_l535_535442


namespace rate_per_meter_l535_535129

theorem rate_per_meter (d : ℝ) (total_cost : ℝ) (rate_per_meter : ℝ) (h_d : d = 30)
    (h_total_cost : total_cost = 188.49555921538757) :
    rate_per_meter = 2 :=
by
  sorry

end rate_per_meter_l535_535129


namespace f_is_increasing_l535_535319

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + 3 * x

theorem f_is_increasing : ∀ (x : ℝ), (deriv f x) > 0 :=
by
  intro x
  calc
    deriv f x = 2 * Real.exp (2 * x) + 3 := by sorry
    _ > 0 := by sorry

end f_is_increasing_l535_535319


namespace meeting_point_2015_l535_535182

/-- 
A motorist starts at point A, and a cyclist starts at point B. They travel towards each other and 
meet for the first time at point C. After meeting, they turn around and travel back to their starting 
points and continue this pattern of meeting, turning around, and traveling back to their starting points. 
Prove that their 2015th meeting point is at point C.
-/
theorem meeting_point_2015 
  (A B C D : Type) 
  (x y t : ℕ)
  (odd_meeting : ∀ n : ℕ, (2 * n + 1) % 2 = 1) : 
  ∃ n, (n = 2015) → odd_meeting n = 1 → (n % 2 = 1 → (C = "C")) := 
sorry

end meeting_point_2015_l535_535182


namespace max_min_values_of_f_l535_535184

noncomputable def f (x : ℝ) : ℝ :=
  4^x - 2^(x+1) - 3

theorem max_min_values_of_f :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → (∀ y, y = f x → y ≤ 5) ∧ (∃ y, y = f 2 ∧ y = 5) ∧ (∀ y, y = f x → y ≥ -4) ∧ (∃ y, y = f 0 ∧ y = -4) :=
by
  sorry

end max_min_values_of_f_l535_535184


namespace calculate_expression_l535_535486

theorem calculate_expression : (8^5 / 8^2) * 3^6 = 373248 := by
  sorry

end calculate_expression_l535_535486


namespace min_value_of_cos_C_l535_535614

noncomputable def minCosC (a b c : ℝ) (h : a^2 + 2 * b^2 = 3 * c^2) : ℝ :=
  if a > 0 ∧ b > 0 ∧ c > 0 ∧ (a < b + c ∧ b < a + c ∧ c < a + b) then
    min (real.cos (real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) (real.cos (real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))))
  else 0

theorem min_value_of_cos_C (a b c : ℝ) (h : a^2 + 2 * b^2 = 3 * c^2) :
  minCosC a b c h = sqrt 2 / 3 := sorry

end min_value_of_cos_C_l535_535614


namespace apple_cost_l535_535763

theorem apple_cost
  (l : ℝ) (q : ℝ)
  (cost_33 : ℝ) (cost_36 : ℝ)
  (h1 : 33 * l + 3 * q = cost_33)
  (h2 : 36 * l + q = cost_36)
  (cost_33_eq : cost_33 = 11.67)
  (cost_36_eq : cost_36 = 12.48) :
  10 * l = 3.62 := by
  have l_def : l = 0.362 := sorry
  show 10 * l = 3.62 from by
    rw [l_def]
    norm_num

end apple_cost_l535_535763


namespace find_roots_l535_535039

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_roots 
  (h_symm : ∀ x : ℝ, f (2 + x) = f (2 - x))
  (h_three_roots : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = 0 ∧ f b = 0 ∧ f c = 0)
  (h_zero_root : f 0 = 0) :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ f a = 0 ∧ f b = 0 :=
sorry

end find_roots_l535_535039


namespace max_correct_answers_l535_535615

-- Define the variables for the problem
variables (a b c : ℕ)

-- Define the conditions of the problem
def conditions (a b c : ℕ) : Prop :=
  a + b + c = 60 ∧
  5 * a - c = 140 ∧
  a + c ≤ 60

-- Define the theorem to prove
theorem max_correct_answers : ∃ a b c, conditions a b c ∧ a = 33 :=
by
  exists 33
  exists 2
  exists 25
  simp [conditions]
  have h1 : 33 + 2 + 25 = 60 := by norm_num
  have h2 : 5 * 33 - 25 = 140 := by norm_num
  have h3 : 33 + 25 ≤ 60 := by norm_num
  exact ⟨⟨h1, h2, h3⟩, rfl⟩

end max_correct_answers_l535_535615


namespace friends_walked_total_distance_l535_535675

theorem friends_walked_total_distance 
  (lionel_miles : ℕ) (esther_yards : ℕ) (niklaus_feet : ℕ)
  (mile_to_feet : ℕ) (yard_to_feet : ℕ) :
  lionel_miles = 4 →
  esther_yards = 975 →
  niklaus_feet = 1287 →
  mile_to_feet = 5280 →
  yard_to_feet = 3 →
  lionel_miles * mile_to_feet + esther_yards * yard_to_feet + niklaus_feet = 25332 :=
by {
  intros,
  sorry
}

end friends_walked_total_distance_l535_535675


namespace probability_two_white_balls_l535_535049

/-- Given a box containing 7 white balls and 8 black balls, if two balls are drawn at random without replacement, prove that the probability of drawing two white balls is 1/5. -/
theorem probability_two_white_balls :
  let total_balls := 15,
      ways_to_choose_two_white := Nat.choose 7 2,
      ways_to_choose_two_any := Nat.choose total_balls 2 in
  (ways_to_choose_two_white : ℚ) / ways_to_choose_two_any = 1 / 5 :=
by
  let total_balls := 15
  let ways_to_choose_two_white := Nat.choose 7 2
  let ways_to_choose_two_any := Nat.choose total_balls 2
  change (ways_to_choose_two_white : ℚ) / ways_to_choose_two_any = 1 / 5
  sorry

end probability_two_white_balls_l535_535049


namespace sum_zero_of_distinct_and_ratio_l535_535320

noncomputable def distinct (a b c d : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

theorem sum_zero_of_distinct_and_ratio (x y u v : ℝ) 
  (h_distinct : distinct x y u v)
  (h_ratio : (x + u) / (x + v) = (y + v) / (y + u)) : 
  x + y + u + v = 0 := 
sorry

end sum_zero_of_distinct_and_ratio_l535_535320


namespace determine_number_l535_535131

def is_divisible_by_9 (n : ℕ) : Prop :=
  (n.digits 10).sum % 9 = 0

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

def ten_power (n p : ℕ) : ℕ :=
  n * 10 ^ p

theorem determine_number (a b : ℕ) (h₁ : b = 0 ∨ b = 5)
  (h₂ : is_divisible_by_9 (7 + 2 + a + 3 + b))
  (h₃ : is_divisible_by_5 (7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b)) :
  (7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b = 72630 ∨ 
   7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b = 72135) :=
by sorry

end determine_number_l535_535131


namespace max_value_inequality_l535_535889

theorem max_value_inequality
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (h3 : x1 * x2 + y1 * y2 = ⅟2) :
  (|x1 + y1 - 1| / Real.sqrt 2) + (|x2 + y2 - 1| / Real.sqrt 2) ≤ 1 :=
by {
  sorry
}

end max_value_inequality_l535_535889


namespace hexagon_side_count_l535_535836

noncomputable def convex_hexagon_sides (a b perimeter : ℕ) : ℕ := 
  if a ≠ b then 6 - (perimeter - (6 * b)) else 0

theorem hexagon_side_count (G H I J K L : ℕ)
  (a b : ℕ)
  (p : ℕ)
  (dist_a : a = 7)
  (dist_b : b = 8)
  (perimeter : p = 46)
  (cond : GHIJKL = [a, b, X, Y, Z, W] ∧ ∀ x ∈ [X, Y, Z, W], x = a ∨ x = b)
  : convex_hexagon_sides a b p = 4 :=
by 
  sorry

end hexagon_side_count_l535_535836


namespace solve_heat_equation_l535_535698

noncomputable def heat_equation_solution (u : ℝ → ℝ → ℝ) := 
  ∀ x t, (∀ x, u x 0 = real.exp (-x)) ∧ (∀ x t, (∂u/∂t)(x, t) = (∂^2u/∂x^2)(x, t))

theorem solve_heat_equation : 
  heat_equation_solution (λ x t, real.exp (t - x)) :=
by
  intros x t
  apply And.intro
  { intro x
    sorry  }
  { intros x t
    sorry  }

end solve_heat_equation_l535_535698


namespace compute_f_2011_l535_535653

noncomputable def A : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2}

noncomputable def f (x : ℚ) (hx : x ∈ A) : ℝ := sorry

theorem compute_f_2011 (h_f : ∀ (x : ℚ) (hx : x ∈ A), 
  f x hx + f (2 - 1 / x) (by {
    have h2 : x ≠ 0 := hx.1,
    have h1 : x ≠ 1 := hx.2.1,
    have h0 : x ≠ 2 := hx.2.2,
    split; norm_cast;
    linarith } ) = Real.log (abs (2 * x)) ) : 
  f 2011 (by {
    split;
    norm_cast;
    linarith [show (2012 : ℝ) > 0, by norm_num] }) = Real.log (2011 / 2) := sorry

end compute_f_2011_l535_535653


namespace measure_angle_C_perimeter_triangle_l535_535268

-- Definitions
variable (a b c : ℝ)
variable (A B C : ℝ)  -- angles in radians

-- Acute triangle with specific relation
variable (h1 : 0 < A ∧ A < π/2)
variable (h2 : 0 < B ∧ B < π/2)
variable (h3 : 0 < C ∧ C < π/2)
variable (h4 : a = c*sin A*2/sqrt 3)  -- Given condition

-- Part 1: Prove the measure of angle C is π/3
theorem measure_angle_C (h4 : a = 2 * c * sin A / sqrt 3): C = π/3 :=
sorry

-- Additional conditions for Part 2
variable (h5 : c = sqrt 7)
variable (h6 : a * b = 6)

-- Part 2: Prove the perimeter of the triangle
theorem perimeter_triangle (h4 : a = 2 * c * sin A / sqrt 3)
                           (h5 : c = sqrt 7)
                           (h6 : a * b = 6) :
  a + b + c = 5 + sqrt 7 :=
sorry

end measure_angle_C_perimeter_triangle_l535_535268


namespace part_a_solution_l535_535040

def sequence_a (n : ℕ) : ℕ → ℝ
| 0       := 0
| 100     := 1
| (k + 1) := if k = 99 then (1 : ℝ)
              else (sequence_a k + sequence_a (k + 2)) / 2

theorem part_a_solution :
  ∀ k ∈ finset.range 101, sequence_a k = k / 100 :=
sorry

end part_a_solution_l535_535040


namespace parabola_expression_point_on_parabola_l535_535223

-- The first condition translates directly: the parabola passes through points (1, 3) and (-1, -1)
def condition_1 (a b : ℝ) : Prop := (3 = a * 1^2 + b * 1) ∧ (-1 = a * (-1)^2 + b * (-1))

-- Given this condition, prove the expression of the parabola is y = x^2 + 2x
theorem parabola_expression (a b : ℝ) (h : condition_1 a b) : 
  ∃ p : ℝ → ℝ, p = (λ x, x^2 + 2 * x) :=
by
  sorry

-- Given the expression of the parabola, check if point B(2, 6) lies on it
theorem point_on_parabola : ¬ (6 = 2^2 + 2 * 2) :=
by
  sorry

end parabola_expression_point_on_parabola_l535_535223


namespace at_least_one_two_prob_l535_535746

-- Definitions and conditions corresponding to the problem
def total_outcomes (n : ℕ) : ℕ := n * n
def no_twos_outcomes (n : ℕ) : ℕ := (n - 1) * (n - 1)

-- The probability calculation
def probability_at_least_one_two (n : ℕ) : ℚ := 
  let tot_outs := total_outcomes n
  let no_twos := no_twos_outcomes n
  (tot_outs - no_twos : ℚ) / tot_outs

-- Our main theorem to be proved
theorem at_least_one_two_prob : 
  probability_at_least_one_two 6 = 11 / 36 := 
by
  sorry

end at_least_one_two_prob_l535_535746


namespace count_negative_numbers_l535_535259

-- Define the evaluations
def eval_expr1 := - (2^2)
def eval_expr2 := (-2) ^ 2
def eval_expr3 := - (-2)
def eval_expr4 := - (abs (-2))

-- Define the negativity checks
def is_negative (x : ℤ) : Prop := x < 0

-- Prove the number of negative results
theorem count_negative_numbers :
  (∑ b in [eval_expr1, eval_expr2, eval_expr3, eval_expr4].map is_negative, if b then 1 else 0) = 2 :=
by sorry

end count_negative_numbers_l535_535259


namespace trader_goal_l535_535475

theorem trader_goal 
  (profit : ℕ)
  (half_profit : ℕ)
  (donation : ℕ)
  (total_funds : ℕ)
  (made_above_goal : ℕ)
  (goal : ℕ)
  (h1 : profit = 960)
  (h2 : half_profit = profit / 2)
  (h3 : donation = 310)
  (h4 : total_funds = half_profit + donation)
  (h5 : made_above_goal = 180)
  (h6 : goal = total_funds - made_above_goal) :
  goal = 610 :=
by 
  sorry

end trader_goal_l535_535475


namespace robot_first_row_probability_l535_535683

def robot_moves (position : ℕ × ℕ) (direction : bool) : ℕ × ℕ :=
  match position with
  | (r, c) => if direction then (r, (c + 1) % 2) else (r, (c + 1) % 2)

def probability_robot_first_row_after_4_moves : ℚ :=
  15 / 16

theorem robot_first_row_probability :
  let initial_position := (2, 2)
  let row_position_after_moves : ℕ × ℕ :=
    robot_moves
      (robot_moves
        (robot_moves
          (robot_moves initial_position direction) 
          direction) 
        direction) 
      direction
  row_position_after_moves.1 = 1 ∨ 
  row_position_after_moves.1 = 2 ∧
  row_position_after_moves.2 = 2 →
  row_position_after_moves.1 = 1 →
  probability_robot_first_row_after_4_moves = 15 / 16 :=
sorry

end robot_first_row_probability_l535_535683


namespace initial_position_of_third_l535_535391

-- Define the conditions as functions or properties
def initial_student_count : Nat := 300

def remove_multiples_of_3 (remaining_students : Nat) : Nat :=
  remaining_students - remaining_students / 3

-- A recursive function to model the process of elimination
def count_down (students : Nat) (steps : Nat) : Nat :=
  if students <= 3 then students
  else count_down (remove_multiples_of_3 students) (steps + 1)

-- Statement to prove the initial position of the third person among the final 3 students
theorem initial_position_of_third (initial_count : Nat) (final_position : Nat) :
  initial_count = 300 → final_position = 212 → 
  count_down initial_count 0 = 3 ∧ true := 
begin
  intros h1 h2,
  have : count_down initial_count 0 = 3 := sorry, -- The main proof argument
  exact ⟨this, trivial⟩
end

end initial_position_of_third_l535_535391


namespace average_of_consecutive_sequences_l535_535535

theorem average_of_consecutive_sequences (a b : ℕ) (h : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) : 
    ((b + (b+1) + (b+2) + (b+3) + (b+4)) / 5) = a + 4 :=
by
  sorry

end average_of_consecutive_sequences_l535_535535


namespace sin_70_given_sin_10_l535_535187

theorem sin_70_given_sin_10 (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by 
  sorry

end sin_70_given_sin_10_l535_535187


namespace triangle_circumradius_perimeter_area_inequality_l535_535690

theorem triangle_circumradius_perimeter_area_inequality
  (R P S : ℝ)
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (R = (a * b * c) / (4 * S)) ∧ (P = a + b + c) ∧ (S = sqrt ((a + (b + c)) * ((a + b) - c) * ((a + c) - b) * ((b + c) - a)) / 4)) :
  R * P ≥ 4 * S :=
sorry

end triangle_circumradius_perimeter_area_inequality_l535_535690


namespace allocation_schemes_count_l535_535516

-- Define the problem parameters
def volunteers := 6
def groups := 4
def min_group_size := 1
def max_group_size := 2

-- Define the correct answer
def correct_answer := 1080

-- State the theorem
theorem allocation_schemes_count :
  ∃ (v g : ℕ) (min_gs max_gs : ℕ) (correct : ℕ),
    v = volunteers ∧
    g = groups ∧
    min_gs = min_group_size ∧
    max_gs = max_group_size ∧
    correct = correct_answer ∧
    ∃ f : fin v → fin g, (nat.factorial g * 
      (nat.choose 6 2 * nat.choose 4 2 * nat.choose 2 1 * nat.choose 1 1) 
      / (nat.factorial 2 * nat.factorial 2) = correct) :=
begin
  -- The group allocation proof would go here
  sorry
end

end allocation_schemes_count_l535_535516


namespace cds_unique_to_either_l535_535810

-- Declare the variables for the given problem
variables (total_alice_shared : ℕ) (total_alice : ℕ) (unique_bob : ℕ)

-- The given conditions in the problem
def condition_alice : Prop := total_alice_shared + unique_bob + (total_alice - total_alice_shared) = total_alice

-- The theorem to prove: number of CDs in either Alice's or Bob's collection but not both is 19
theorem cds_unique_to_either (h1 : total_alice = 23) 
                             (h2 : total_alice_shared = 12) 
                             (h3 : unique_bob = 8) : 
                             (total_alice - total_alice_shared) + unique_bob = 19 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end cds_unique_to_either_l535_535810
