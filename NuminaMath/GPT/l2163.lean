import Mathlib

namespace age_sum_proof_l2163_216331

theorem age_sum_proof : 
  ∀ (Matt Fem Jake : ℕ), 
    Matt = 4 * Fem →
    Fem = 11 →
    Jake = Matt + 5 →
    (Matt + 2) + (Fem + 2) + (Jake + 2) = 110 :=
by
  intros Matt Fem Jake h1 h2 h3
  sorry

end age_sum_proof_l2163_216331


namespace cos_identity_arithmetic_sequence_in_triangle_l2163_216365

theorem cos_identity_arithmetic_sequence_in_triangle
  {A B C : ℝ} {a b c : ℝ}
  (h1 : 2 * b = a + c)
  (h2 : a / Real.sin A = b / Real.sin B)
  (h3 : b / Real.sin B = c / Real.sin C)
  (h4 : A + B + C = Real.pi)
  : 5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 4 := 
  sorry

end cos_identity_arithmetic_sequence_in_triangle_l2163_216365


namespace smaller_angle_measure_l2163_216352

theorem smaller_angle_measure (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 := by
  sorry

end smaller_angle_measure_l2163_216352


namespace find_h_s_pairs_l2163_216337

def num_regions (h s : ℕ) : ℕ :=
  1 + h * (s + 1) + s * (s + 1) / 2

theorem find_h_s_pairs (h s : ℕ) :
  h > 0 ∧ s > 0 ∧
  num_regions h s = 1992 ↔ 
  (h, s) = (995, 1) ∨ (h, s) = (176, 10) ∨ (h, s) = (80, 21) :=
by
  sorry

end find_h_s_pairs_l2163_216337


namespace evaluate_polynomial_at_6_eq_1337_l2163_216323

theorem evaluate_polynomial_at_6_eq_1337 :
  (3 * 6^2 + 15 * 6 + 7) + (4 * 6^3 + 8 * 6^2 - 5 * 6 + 10) = 1337 := by
  sorry

end evaluate_polynomial_at_6_eq_1337_l2163_216323


namespace thermos_count_l2163_216388

theorem thermos_count
  (total_gallons : ℝ)
  (pints_per_gallon : ℝ)
  (thermoses_drunk_by_genevieve : ℕ)
  (pints_drunk_by_genevieve : ℝ)
  (total_pints : ℝ) :
  total_gallons * pints_per_gallon = total_pints ∧
  pints_drunk_by_genevieve / thermoses_drunk_by_genevieve = 2 →
  total_pints / 2 = 18 :=
by
  intros h
  have := h.2
  sorry

end thermos_count_l2163_216388


namespace max_digit_d_l2163_216328

theorem max_digit_d (d f : ℕ) (h₁ : d ≤ 9) (h₂ : f ≤ 9) (h₃ : (18 + d + f) % 3 = 0) (h₄ : (12 - (d + f)) % 11 = 0) : d = 1 :=
sorry

end max_digit_d_l2163_216328


namespace rectangle_area_l2163_216363

theorem rectangle_area (l w : ℝ) (h1 : 2 * l + 2 * w = 14) (h2 : l^2 + w^2 = 25) : l * w = 12 :=
by
  sorry

end rectangle_area_l2163_216363


namespace range_of_fraction_l2163_216385

variable {x y : ℝ}

-- Condition given in the problem
def equation (x y : ℝ) : Prop := x + 2 * y - 6 = 0

-- The range condition for x
def x_range (x : ℝ) : Prop := 0 < x ∧ x < 3

-- The corresponding theorem statement
theorem range_of_fraction (h_eq : equation x y) (h_x_range : x_range x) :
  ∃ a b : ℝ, (a < 1 ∧ 10 < b) ∧ (a, b) = (1, 10) ∧
  ∀ k : ℝ, k = (x + 2) / (y - 1) → 1 < k ∧ k < 10 :=
sorry

end range_of_fraction_l2163_216385


namespace max_c_value_l2163_216345

variable {a b c : ℝ}

theorem max_c_value (h1 : 2 * (a + b) = a * b) (h2 : a + b + c = a * b * c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  c ≤ 8 / 15 :=
sorry

end max_c_value_l2163_216345


namespace multiply_by_3_l2163_216330

variable (x : ℕ)  -- Declare x as a natural number

-- Define the conditions
def condition : Prop := x + 14 = 56

-- The goal to prove
theorem multiply_by_3 (h : condition x) : 3 * x = 126 := sorry

end multiply_by_3_l2163_216330


namespace eric_boxes_l2163_216394

def numberOfBoxes (totalPencils : Nat) (pencilsPerBox : Nat) : Nat :=
  totalPencils / pencilsPerBox

theorem eric_boxes :
  numberOfBoxes 27 9 = 3 := by
  sorry

end eric_boxes_l2163_216394


namespace tan_sum_identity_l2163_216300

theorem tan_sum_identity (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) : Real.tan x = 1 / 3 := 
by 
  sorry

end tan_sum_identity_l2163_216300


namespace sum_of_ages_l2163_216344

theorem sum_of_ages (y : ℕ) 
  (h_diff : 38 - y = 2) : y + 38 = 74 := 
by {
  sorry
}

end sum_of_ages_l2163_216344


namespace find_a_l2163_216393

theorem find_a (a : ℚ) :
  let p1 := (3, 4)
  let p2 := (-4, 1)
  let direction_vector := (a, -2)
  let vector_between_points := (p2.1 - p1.1, p2.2 - p1.2)
  ∃ k : ℚ, direction_vector = (k * vector_between_points.1, k * vector_between_points.2) →
  a = -14 / 3 := by
    sorry

end find_a_l2163_216393


namespace articles_in_selling_price_l2163_216326

theorem articles_in_selling_price (C : ℝ) (N : ℕ) 
  (h1 : 50 * C = N * (1.25 * C)) 
  (h2 : 0.25 * C = 25 / 100 * C) :
  N = 40 :=
by
  sorry

end articles_in_selling_price_l2163_216326


namespace solution_exists_l2163_216306

theorem solution_exists (x : ℝ) :
  (|2 * x - 3| ≤ 3 ∧ (1 / x) < 1 ∧ x ≠ 0) ↔ (1 < x ∧ x ≤ 3) :=
by
  sorry

end solution_exists_l2163_216306


namespace ratio_problem_l2163_216398

theorem ratio_problem 
  (x y z w : ℚ) 
  (h1 : x / y = 12) 
  (h2 : z / y = 4) 
  (h3 : z / w = 3 / 4) : 
  w / x = 4 / 9 := 
  sorry

end ratio_problem_l2163_216398


namespace sum_of_solutions_eq_neg_six_l2163_216351

theorem sum_of_solutions_eq_neg_six (x r s : ℝ) :
  (81 : ℝ) - 18 * x - 3 * x^2 = 0 →
  (r + s = -6) :=
by
  sorry

end sum_of_solutions_eq_neg_six_l2163_216351


namespace cube_split_odd_numbers_l2163_216359

theorem cube_split_odd_numbers (m : ℕ) (h1 : 1 < m) (h2 : ∃ k, (31 = 2 * k + 1 ∧ (m - 1) * m / 2 = k)) : m = 6 := 
by
  sorry

end cube_split_odd_numbers_l2163_216359


namespace division_quotient_l2163_216349

-- Define conditions
def dividend : ℕ := 686
def divisor : ℕ := 36
def remainder : ℕ := 2

-- Define the quotient
def quotient : ℕ := dividend - remainder

theorem division_quotient :
  quotient = divisor * 19 :=
sorry

end division_quotient_l2163_216349


namespace copy_pages_l2163_216379

theorem copy_pages (cost_per_5_pages : ℝ) (total_dollars : ℝ) : 
  (cost_per_5_pages = 10) → (total_dollars = 15) → (15 * 100 / 10 * 5 = 750) :=
by
  intros
  sorry

end copy_pages_l2163_216379


namespace num_solutions_3x_plus_2y_eq_806_l2163_216384

theorem num_solutions_3x_plus_2y_eq_806 :
  (∃ y : ℕ, ∃ x : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 806) ∧
  ((∃ t : ℤ, x = 268 - 2 * t ∧ y = 1 + 3 * t) ∧ (∃ t : ℤ, 0 ≤ t ∧ t ≤ 133)) :=
sorry

end num_solutions_3x_plus_2y_eq_806_l2163_216384


namespace n_cubed_plus_20n_div_48_l2163_216321

theorem n_cubed_plus_20n_div_48 (n : ℕ) (h_even : n % 2 = 0) : (n^3 + 20 * n) % 48 = 0 :=
sorry

end n_cubed_plus_20n_div_48_l2163_216321


namespace max_min_values_l2163_216371

namespace ProofPrimary

-- Define the polynomial function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 36 * x + 1

-- State the interval of interest
def interval : Set ℝ := Set.Icc 1 11

-- Main theorem asserting the minimum and maximum values
theorem max_min_values : 
  (∀ x ∈ interval, f x ≥ -43 ∧ f x ≤ 2630) ∧
  (∃ x ∈ interval, f x = -43) ∧
  (∃ x ∈ interval, f x = 2630) :=
by
  sorry

end ProofPrimary

end max_min_values_l2163_216371


namespace value_of_business_l2163_216327

variable (V : ℝ)
variable (h1 : (2 / 3) * V = S)
variable (h2 : (3 / 4) * S = 75000)

theorem value_of_business (h1 : (2 / 3) * V = S) (h2 : (3 / 4) * S = 75000) : V = 150000 :=
sorry

end value_of_business_l2163_216327


namespace f_g_of_3_l2163_216396

def f (x : ℤ) : ℤ := 2 * x + 3
def g (x : ℤ) : ℤ := x^3 - 6

theorem f_g_of_3 : f (g 3) = 45 := by
  sorry

end f_g_of_3_l2163_216396


namespace minimal_polynomial_correct_l2163_216334

noncomputable def minimal_polynomial : Polynomial ℚ :=
  (Polynomial.X^2 - 4 * Polynomial.X + 1) * (Polynomial.X^2 - 6 * Polynomial.X + 2)

theorem minimal_polynomial_correct :
  Polynomial.X^4 - 10 * Polynomial.X^3 + 29 * Polynomial.X^2 - 26 * Polynomial.X + 2 = minimal_polynomial :=
  sorry

end minimal_polynomial_correct_l2163_216334


namespace arithmetic_sequence_75th_term_l2163_216332

theorem arithmetic_sequence_75th_term (a1 d : ℤ) (n : ℤ) (h1 : a1 = 3) (h2 : d = 5) (h3 : n = 75) :
  a1 + (n - 1) * d = 373 :=
by
  rw [h1, h2, h3]
  -- Here, we arrive at the explicitly stated elements and evaluate:
  -- 3 + (75 - 1) * 5 = 373
  sorry

end arithmetic_sequence_75th_term_l2163_216332


namespace series_sum_eq_l2163_216315

noncomputable def sum_series : ℝ :=
∑' n : ℕ, if h : n > 0 then (4 * n + 3) / ((4 * n)^2 * (4 * n + 4)^2) else 0

theorem series_sum_eq :
  sum_series = 1 / 256 := by
  sorry

end series_sum_eq_l2163_216315


namespace integral_evaluation_l2163_216304

noncomputable def integral_value : Real :=
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - (x - 1)^2) - x)

theorem integral_evaluation :
  integral_value = (Real.pi / 4) - 1 / 2 :=
by
  sorry

end integral_evaluation_l2163_216304


namespace ilya_arithmetic_l2163_216346

theorem ilya_arithmetic (v t : ℝ) (h : v + t = v * t ∧ v + t = v / t) : False :=
by
  sorry

end ilya_arithmetic_l2163_216346


namespace max_pieces_with_three_cuts_l2163_216360

def cake := Type

noncomputable def max_identical_pieces (cuts : ℕ) (max_cuts : ℕ) : ℕ :=
  if cuts = 3 ∧ max_cuts = 3 then 8 else sorry

theorem max_pieces_with_three_cuts : ∀ (c : cake), max_identical_pieces 3 3 = 8 :=
by
  intro c
  sorry

end max_pieces_with_three_cuts_l2163_216360


namespace algebraic_expression_defined_iff_l2163_216395

theorem algebraic_expression_defined_iff (x : ℝ) : (∃ y, y = 3 / (x - 2)) ↔ x ≠ 2 := by
  sorry

end algebraic_expression_defined_iff_l2163_216395


namespace simplify_expression_l2163_216354

variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 9) - (x + 6) * (3 * x - 2) = 7 * x - 24 :=
by
  sorry

end simplify_expression_l2163_216354


namespace expand_subtract_equals_result_l2163_216391

-- Definitions of the given expressions
def expand_and_subtract (x : ℝ) : ℝ :=
  (x + 3) * (2 * x - 5) - (2 * x + 1)

-- Expected result
def expected_result (x : ℝ) : ℝ :=
  2 * x ^ 2 - x - 16

-- The theorem stating the equivalence of the expanded and subtracted expression with the expected result
theorem expand_subtract_equals_result (x : ℝ) : expand_and_subtract x = expected_result x :=
  sorry

end expand_subtract_equals_result_l2163_216391


namespace minimum_letters_for_grid_coloring_l2163_216374

theorem minimum_letters_for_grid_coloring : 
  ∀ (grid_paper : Type) 
  (is_node : grid_paper → Prop) 
  (marked : grid_paper → Prop)
  (mark_with_letter : grid_paper → ℕ) 
  (connected : grid_paper → grid_paper → Prop), 
  (∀ n₁ n₂ : grid_paper, is_node n₁ → is_node n₂ → mark_with_letter n₁ = mark_with_letter n₂ → 
  (n₁ ≠ n₂ → ∃ n₃ : grid_paper, is_node n₃ ∧ connected n₁ n₃ ∧ connected n₃ n₂ ∧ mark_with_letter n₃ ≠ mark_with_letter n₁)) → 
  ∃ (k : ℕ), k = 2 :=
by
  sorry

end minimum_letters_for_grid_coloring_l2163_216374


namespace eccentricity_of_ellipse_l2163_216355

theorem eccentricity_of_ellipse (a b c e : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = Real.sqrt (a^2 - b^2)) 
  (h4 : b = Real.sqrt 3 * c) : e = 1/2 :=
by
  sorry

end eccentricity_of_ellipse_l2163_216355


namespace sum_base7_l2163_216308

def base7_to_base10 (n : ℕ) : ℕ := 
  -- Function to convert base 7 to base 10 (implementation not shown)
  sorry

def base10_to_base7 (n : ℕ) : ℕ :=
  -- Function to convert base 10 to base 7 (implementation not shown)
  sorry

theorem sum_base7 (a b : ℕ) (ha : a = base7_to_base10 12) (hb : b = base7_to_base10 245) :
  base10_to_base7 (a + b) = 260 :=
sorry

end sum_base7_l2163_216308


namespace a_plus_b_eq_l2163_216325

-- Define the sets A and B
def A := { x : ℝ | -1 < x ∧ x < 3 }
def B := { x : ℝ | -3 < x ∧ x < 2 }

-- Define the intersection set A ∩ B
def A_inter_B := { x : ℝ | -1 < x ∧ x < 2 }

-- Define a condition
noncomputable def is_solution_set (a b : ℝ) : Prop :=
  ∀ x : ℝ, (-1 < x ∧ x < 2) ↔ (x^2 + a * x + b < 0)

-- The proof statement
theorem a_plus_b_eq : ∃ a b : ℝ, is_solution_set a b ∧ a + b = -3 := by
  sorry

end a_plus_b_eq_l2163_216325


namespace lucy_l2163_216392

theorem lucy's_age 
  (L V: ℕ)
  (h1: L - 5 = 3 * (V - 5))
  (h2: L + 10 = 2 * (V + 10)) :
  L = 50 :=
by
  sorry

end lucy_l2163_216392


namespace rhombus_area_l2163_216366

/-
  We want to prove that the area of a rhombus with given diagonals' lengths is 
  equal to the computed value according to the formula Area = (d1 * d2) / 2.
-/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) : 
  (d1 * d2) / 2 = 60 :=
by
  rw [h1, h2]
  sorry

end rhombus_area_l2163_216366


namespace annie_jacob_ratio_l2163_216380

theorem annie_jacob_ratio :
  ∃ (a j : ℕ), ∃ (m : ℕ), (m = 2 * a) ∧ (j = 90) ∧ (m = 60) ∧ (a / j = 1 / 3) :=
by
  sorry

end annie_jacob_ratio_l2163_216380


namespace basketball_total_points_l2163_216364

variable (Jon_points Jack_points Tom_points : ℕ)

def Jon_score := 3
def Jack_score := Jon_score + 5
def Tom_score := (Jon_score + Jack_score) - 4

theorem basketball_total_points :
  Jon_score + Jack_score + Tom_score = 18 := by
  sorry

end basketball_total_points_l2163_216364


namespace num_pairs_in_arithmetic_progression_l2163_216338

theorem num_pairs_in_arithmetic_progression : 
  ∃ n : ℕ, n = 2 ∧ ∀ a b : ℝ, (a = (15 + b) / 2 ∧ (a + a * b = 2 * b)) ↔ 
  (a = (9 + 3 * Real.sqrt 7) / 2 ∧ b = -6 + 3 * Real.sqrt 7)
  ∨ (a = (9 - 3 * Real.sqrt 7) / 2 ∧ b = -6 - 3 * Real.sqrt 7) :=    
  sorry

end num_pairs_in_arithmetic_progression_l2163_216338


namespace games_played_by_third_player_l2163_216370

theorem games_played_by_third_player
    (games_first : ℕ)
    (games_second : ℕ)
    (games_first_eq : games_first = 10)
    (games_second_eq : games_second = 21) :
    ∃ (games_third : ℕ), games_third = 11 := by
  sorry

end games_played_by_third_player_l2163_216370


namespace factor_problem_l2163_216353

theorem factor_problem (C D : ℤ) (h1 : 16 * x^2 - 88 * x + 63 = (C * x - 21) * (D * x - 3)) (h2 : C * D + C = 21) : C = 7 ∧ D = 2 :=
by 
  sorry

end factor_problem_l2163_216353


namespace find_line_equation_through_point_intersecting_hyperbola_l2163_216378

theorem find_line_equation_through_point_intersecting_hyperbola 
  (x y : ℝ) 
  (hx : x = -2 / 3)
  (hy : (x : ℝ) = 0) : 
  ∃ k : ℝ, (∀ x y : ℝ, y = k * x - 1 → ((x^2 / 2) - (y^2 / 5) = 1)) ∧ k = 1 := 
sorry

end find_line_equation_through_point_intersecting_hyperbola_l2163_216378


namespace max_value_fraction_l2163_216310

theorem max_value_fraction : ∀ (x y : ℝ), (-5 ≤ x ∧ x ≤ -1) → (1 ≤ y ∧ y ≤ 3) → (1 + y / x ≤ -2) :=
  by
    intros x y hx hy
    sorry

end max_value_fraction_l2163_216310


namespace average_is_3_l2163_216311

theorem average_is_3 (A B C : ℝ) (h1 : 1501 * C - 3003 * A = 6006)
                              (h2 : 1501 * B + 4504 * A = 7507)
                              (h3 : A + B = 1) :
  (A + B + C) / 3 = 3 :=
by sorry

end average_is_3_l2163_216311


namespace compute_expression_l2163_216367
-- Start with importing math library utilities for linear algebra and dot product

-- Define vector 'a' and 'b' in Lean
def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

-- Define dot product operation 
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- Define the expression and the theorem
theorem compute_expression : dot_product ((2 * a.1 + b.1, 2 * a.2 + b.2)) a = 1 :=
by
  -- Insert the proof steps here
  sorry

end compute_expression_l2163_216367


namespace abs_diff_m_n_l2163_216312

variable (m n : ℝ)

theorem abs_diff_m_n (h1 : m * n = 6) (h2 : m + n = 7) (h3 : m^2 - n^2 = 13) : |m - n| = 13 / 7 :=
by
  sorry

end abs_diff_m_n_l2163_216312


namespace part1_part2_l2163_216361

-- Part 1: Proving the inequality
theorem part1 (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := by
  sorry

-- Part 2: Maximizing 2a + b
theorem part2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_constraint : a^2 + b^2 = 5) : 
  2 * a + b ≤ 5 := by
  sorry

end part1_part2_l2163_216361


namespace sum_of_two_numbers_l2163_216381

variables {x y : ℝ}

theorem sum_of_two_numbers (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l2163_216381


namespace mode_is_37_median_is_36_l2163_216350

namespace ProofProblem

def data_set : List ℕ := [34, 35, 36, 34, 36, 37, 37, 36, 37, 37]

def mode (l : List ℕ) : ℕ := sorry -- Implementing a mode function

def median (l : List ℕ) : ℕ := sorry -- Implementing a median function

theorem mode_is_37 : mode data_set = 37 := 
  by 
    sorry -- Proof of mode

theorem median_is_36 : median data_set = 36 := 
  by
    sorry -- Proof of median

end ProofProblem

end mode_is_37_median_is_36_l2163_216350


namespace journey_distance_l2163_216357

theorem journey_distance
  (total_time : ℝ)
  (speed1 speed2 : ℝ)
  (journey_time : total_time = 10)
  (speed1_val : speed1 = 21)
  (speed2_val : speed2 = 24) :
  ∃ D : ℝ, (D / 2 / speed1 + D / 2 / speed2 = total_time) ∧ D = 224 :=
by
  sorry

end journey_distance_l2163_216357


namespace distance_between_points_l2163_216399

theorem distance_between_points :
  let p1 := (-4, 17)
  let p2 := (12, -1)
  let distance := Real.sqrt ((12 - (-4))^2 + (-1 - 17)^2)
  distance = 2 * Real.sqrt 145 := sorry

end distance_between_points_l2163_216399


namespace part1_part2_l2163_216317

variables {a_n b_n : ℕ → ℤ} {k m : ℕ}

-- Part 1: Arithmetic Sequence
axiom a2_eq_3 : a_n 2 = 3
axiom S5_eq_25 : (5 * (2 * (a_n 1 + 2 * (a_n 1 + 1)) / 2)) = 25

-- Part 2: Geometric Sequence
axiom b1_eq_1 : b_n 1 = 1
axiom q_eq_3 : ∀ n, b_n n = 3^(n-1)

noncomputable def arithmetic_seq (n : ℕ) : ℤ :=
  2 * n - 1

theorem part1 : (a_n 2 + a_n 4) / 2 = 5 :=
  sorry

theorem part2 (k : ℕ) (hk : 0 < k) : ∃ m, b_n k = arithmetic_seq m ∧ m = (3^(k-1) + 1) / 2 :=
  sorry

end part1_part2_l2163_216317


namespace part_I_part_II_l2163_216319

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)

theorem part_I (x : ℝ) : (f x > 5) ↔ (x < -3 ∨ x > 2) :=
  sorry

theorem part_II (a : ℝ) : (∀ x, f x < a ↔ false) ↔ (a ≤ 3) :=
  sorry

end part_I_part_II_l2163_216319


namespace anne_age_ratio_l2163_216389

-- Define the given conditions and prove the final ratio
theorem anne_age_ratio (A M : ℕ) (h1 : A = 4 * (A - 4 * M) + M) 
(h2 : A - M = 3 * (A - 4 * M)) : (A : ℚ) / (M : ℚ) = 5.5 := 
sorry

end anne_age_ratio_l2163_216389


namespace percentage_markup_l2163_216356

variable (W R : ℝ) -- W is the wholesale cost, R is the normal retail price

-- The condition that, at 60% discount, the sale price nets a 35% profit on the wholesale cost
variable (h : 0.4 * R = 1.35 * W)

-- The goal statement to prove
theorem percentage_markup (h : 0.4 * R = 1.35 * W) : ((R - W) / W) * 100 = 237.5 :=
by
  sorry

end percentage_markup_l2163_216356


namespace find_k_value_l2163_216375

theorem find_k_value 
  (A B C k : ℤ)
  (hA : A = -3)
  (hB : B = -5)
  (hC : C = 6)
  (hSum : A + B + C + k = -A - B - C - k) : 
  k = 2 :=
sorry

end find_k_value_l2163_216375


namespace interval_intersection_l2163_216302

theorem interval_intersection (x : ℝ) :
  (4 * x > 2 ∧ 4 * x < 3) ∧ (5 * x > 2 ∧ 5 * x < 3) ↔ (x > 1/2 ∧ x < 3/5) :=
by
  sorry

end interval_intersection_l2163_216302


namespace sophie_clothes_expense_l2163_216373

theorem sophie_clothes_expense :
  let initial_fund := 260
  let shirt_cost := 18.50
  let trousers_cost := 63
  let num_shirts := 2
  let num_remaining_clothes := 4
  let total_spent := num_shirts * shirt_cost + trousers_cost
  let remaining_amount := initial_fund - total_spent
  let individual_item_cost := remaining_amount / num_remaining_clothes
  individual_item_cost = 40 := 
by 
  sorry

end sophie_clothes_expense_l2163_216373


namespace not_coincidence_l2163_216347

theorem not_coincidence (G : Type) [Fintype G] [DecidableEq G]
    (friend_relation : G → G → Prop)
    (h_friend : ∀ (a b : G), friend_relation a b → friend_relation b a)
    (initial_condition : ∀ (subset : Finset G), subset.card = 4 → 
         ∃ x ∈ subset, ∀ y ∈ subset, x ≠ y → friend_relation x y) :
    ∀ (subset : Finset G), subset.card = 4 → 
        ∃ x ∈ subset, ∀ y ∈ Finset.univ, x ≠ y → friend_relation x y :=
by
  intros subset h_card
  -- The proof would be constructed here
  sorry

end not_coincidence_l2163_216347


namespace Shelby_fog_time_l2163_216383

variable (x y : ℕ)

-- Conditions
def speed_sun := 7/12
def speed_rain := 5/12
def speed_fog := 1/4
def total_time := 60
def total_distance := 20

theorem Shelby_fog_time :
  ((speed_sun * (total_time - x - y)) + (speed_rain * x) + (speed_fog * y) = total_distance) → y = 45 :=
by
  sorry

end Shelby_fog_time_l2163_216383


namespace triangle_BC_length_l2163_216390

noncomputable def length_of_BC (ABC : Triangle) (incircle_radius : ℝ) (altitude_A_to_BC : ℝ) 
    (BD_squared_plus_CD_squared : ℝ) : ℝ :=
  if incircle_radius = 3 ∧ altitude_A_to_BC = 15 ∧ BD_squared_plus_CD_squared = 33 then
    3 * Real.sqrt 7
  else
    0 -- This value is arbitrary, as the conditions above are specific

theorem triangle_BC_length {ABC : Triangle}
    (incircle_radius : ℝ) (altitude_A_to_BC : ℝ) (BD_squared_plus_CD_squared : ℝ) :
    incircle_radius = 3 →
    altitude_A_to_BC = 15 →
    BD_squared_plus_CD_squared = 33 →
    length_of_BC ABC incircle_radius altitude_A_to_BC BD_squared_plus_CD_squared = 3 * Real.sqrt 7 :=
by intros; sorry

end triangle_BC_length_l2163_216390


namespace find_number_l2163_216341

theorem find_number (x : ℝ) (h : (((18 + x) / 3 + 10) / 5 = 4)) : x = 12 :=
by
  sorry

end find_number_l2163_216341


namespace cosh_le_exp_sqr_l2163_216376

open Real

theorem cosh_le_exp_sqr {x k : ℝ} : (∀ x : ℝ, cosh x ≤ exp (k * x^2)) ↔ k ≥ 1/2 :=
sorry

end cosh_le_exp_sqr_l2163_216376


namespace evaluate_expression_l2163_216340

theorem evaluate_expression : 12^2 + 2 * 12 * 5 + 5^2 = 289 := by
  sorry

end evaluate_expression_l2163_216340


namespace first_day_price_l2163_216348

theorem first_day_price (x n: ℝ) :
  n * x = (n + 100) * (x - 1) ∧ 
  n * x = (n - 200) * (x + 2) → 
  x = 4 :=
by
  sorry

end first_day_price_l2163_216348


namespace fraction_of_remaining_birds_left_l2163_216368

theorem fraction_of_remaining_birds_left :
  ∀ (total_birds initial_fraction next_fraction x : ℚ), 
    total_birds = 60 ∧ 
    initial_fraction = 1 / 3 ∧ 
    next_fraction = 2 / 5 ∧ 
    8 = (total_birds * (1 - initial_fraction)) * (1 - next_fraction) * (1 - x) →
    x = 2 / 3 :=
by
  intros total_birds initial_fraction next_fraction x h
  obtain ⟨hb, hi, hn, he⟩ := h
  sorry

end fraction_of_remaining_birds_left_l2163_216368


namespace inequality_solution_set_l2163_216307

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x + 4)^2

theorem inequality_solution_set :
  {x : ℝ | (x^2 + 1) / (x + 4)^2 ≥ 0} = {x : ℝ | x ≠ -4} :=
by
  sorry

end inequality_solution_set_l2163_216307


namespace product_of_A_and_B_l2163_216322

theorem product_of_A_and_B (A B : ℕ) (h1 : 3 / 9 = 6 / A) (h2 : B / 63 = 6 / A) : A * B = 378 :=
  sorry

end product_of_A_and_B_l2163_216322


namespace sequence_initial_term_l2163_216301

theorem sequence_initial_term (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n + 1) = a n + n)
  (h2 : a 61 = 2010) : a 1 = 180 :=
by
  sorry

end sequence_initial_term_l2163_216301


namespace employed_males_percent_l2163_216339

def percent_employed_population : ℝ := 96
def percent_females_among_employed : ℝ := 75

theorem employed_males_percent :
  percent_employed_population * (1 - percent_females_among_employed / 100) = 24 := by
    sorry

end employed_males_percent_l2163_216339


namespace girls_left_to_play_kho_kho_l2163_216316

theorem girls_left_to_play_kho_kho (B G x : ℕ) 
  (h_eq : B = G)
  (h_twice : B = 2 * (G - x))
  (h_total : B + G = 32) :
  x = 8 :=
by sorry

end girls_left_to_play_kho_kho_l2163_216316


namespace sum_of_consecutive_even_numbers_l2163_216343

theorem sum_of_consecutive_even_numbers (n : ℕ) (h : (n + 2)^2 - n^2 = 84) :
  n + (n + 2) = 42 :=
sorry

end sum_of_consecutive_even_numbers_l2163_216343


namespace fraction_simplification_l2163_216313

theorem fraction_simplification (x : ℝ) (h: x ≠ 1) : (5 * x / (x - 1) - 5 / (x - 1)) = 5 := 
sorry

end fraction_simplification_l2163_216313


namespace radius_of_circle_with_center_on_line_and_passing_through_points_l2163_216369

theorem radius_of_circle_with_center_on_line_and_passing_through_points : 
  (∃ a b : ℝ, 2 * a + b = 0 ∧ 
              (a - 1) ^ 2 + (b - 3) ^ 2 = r ^ 2 ∧ 
              (a - 4) ^ 2 + (b - 2) ^ 2 = r ^ 2 
              → r = 5) := 
by 
  sorry

end radius_of_circle_with_center_on_line_and_passing_through_points_l2163_216369


namespace correct_inequality_l2163_216358

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_increasing : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0

theorem correct_inequality : f (-2) < f 1 ∧ f 1 < f 3 :=
by 
  sorry

end correct_inequality_l2163_216358


namespace solve_equation_floor_l2163_216305

theorem solve_equation_floor (x : ℚ) :
  (⌊(5 + 6 * x) / 8⌋ : ℚ) = (15 * x - 7) / 5 ↔ x = 7 / 15 ∨ x = 4 / 5 :=
by
  sorry

end solve_equation_floor_l2163_216305


namespace Jed_cards_after_4_weeks_l2163_216303

theorem Jed_cards_after_4_weeks :
  (∀ n: ℕ, (if n % 2 = 0 then 20 + 4*n - 2*n else 20 + 4*n - 2*(n-1)) = 40) :=
by {
  sorry
}

end Jed_cards_after_4_weeks_l2163_216303


namespace find_value_of_expression_l2163_216382

variable {x y : ℝ}
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
variable (h : x + y = x * y + 1)

theorem find_value_of_expression (h : x + y = x * y + 1) : 
  (1 / x) + (1 / y) = 1 + (1 / (x * y)) :=
  sorry

end find_value_of_expression_l2163_216382


namespace ab_ac_bc_nonpositive_l2163_216324

theorem ab_ac_bc_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ∃ y : ℝ, y = ab + ac + bc ∧ y ≤ 0 :=
by
  sorry

end ab_ac_bc_nonpositive_l2163_216324


namespace album_count_l2163_216314

def albums_total (A B K M C : ℕ) : Prop :=
  A = 30 ∧ B = A - 15 ∧ K = 6 * B ∧ M = 5 * K ∧ C = 3 * M ∧ (A + B + K + M + C) = 1935

theorem album_count (A B K M C : ℕ) : albums_total A B K M C :=
by
  sorry

end album_count_l2163_216314


namespace find_integer_pairs_l2163_216333

def satisfies_conditions (m n : ℤ) : Prop :=
  m^2 = n^5 + n^4 + 1 ∧ ((m - 7 * n) ∣ (m - 4 * n))

theorem find_integer_pairs :
  ∀ (m n : ℤ), satisfies_conditions m n → (m, n) = (-1, 0) ∨ (m, n) = (1, 0) := by
  sorry

end find_integer_pairs_l2163_216333


namespace a_beats_b_by_7_seconds_l2163_216329

/-
  Given:
  1. A's time to finish the race is 28 seconds (tA = 28).
  2. The race distance is 280 meters (d = 280).
  3. A beats B by 56 meters (dA - dB = 56).
  
  Prove:
  A beats B by 7 seconds (tB - tA = 7).
-/

theorem a_beats_b_by_7_seconds 
  (tA : ℕ) (d : ℕ) (speedA : ℕ) (dB : ℕ) (tB : ℕ) 
  (h1 : tA = 28) 
  (h2 : d = 280) 
  (h3 : d - dB = 56) 
  (h4 : speedA = d / tA) 
  (h5 : dB = speedA * tA) 
  (h6 : tB = d / speedA) :
  tB - tA = 7 := 
sorry

end a_beats_b_by_7_seconds_l2163_216329


namespace removed_term_is_a11_l2163_216372

noncomputable def sequence_a (n : ℕ) (a1 d : ℤ) := a1 + (n - 1) * d

def sequence_sum (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem removed_term_is_a11 :
  ∃ d : ℤ, ∀ a1 d : ℤ, 
            a1 = -5 ∧ 
            sequence_sum 11 a1 d = 55 ∧ 
            (sequence_sum 11 a1 d - sequence_a 11 a1 d) / 10 = 4 
          → sequence_a 11 a1 d = removed_term :=
sorry

end removed_term_is_a11_l2163_216372


namespace clock_angle_at_3_45_l2163_216336

theorem clock_angle_at_3_45 :
  let minute_angle_rate := 6.0 -- degrees per minute
  let hour_angle_rate := 0.5  -- degrees per minute
  let initial_angle := 90.0   -- degrees at 3:00
  let minutes_passed := 45.0  -- minutes since 3:00
  let angle_difference_rate := minute_angle_rate - hour_angle_rate
  let angle_change := angle_difference_rate * minutes_passed
  let final_angle := initial_angle - angle_change
  let smaller_angle := if final_angle < 0 then 360.0 + final_angle else final_angle
  smaller_angle = 157.5 :=
by
  sorry

end clock_angle_at_3_45_l2163_216336


namespace hexagon_area_l2163_216386

theorem hexagon_area (s : ℝ) (hex_area : ℝ) (p q : ℤ) :
  s = 3 ∧ hex_area = (3 * Real.sqrt 3 / 2) * s^2 ∧ hex_area = Real.sqrt p + Real.sqrt q → p + q = 545 :=
by
  sorry

end hexagon_area_l2163_216386


namespace fraction_to_decimal_l2163_216318

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 := 
sorry

end fraction_to_decimal_l2163_216318


namespace alice_bob_meet_l2163_216397

theorem alice_bob_meet :
  ∃ k : ℕ, (4 * k - 4 * (k / 5) ≡ 8 * k [MOD 15]) ∧ (k = 5) :=
by
  sorry

end alice_bob_meet_l2163_216397


namespace largest_y_coordinate_l2163_216387

theorem largest_y_coordinate (x y : ℝ) (h : (x^2 / 49) + ((y - 3)^2 / 25) = 0) : y = 3 :=
sorry

end largest_y_coordinate_l2163_216387


namespace geometric_progression_condition_l2163_216342

theorem geometric_progression_condition (a b c : ℝ) (h_b_neg : b < 0) : 
  (b^2 = a * c) ↔ (∃ (r : ℝ), a = r * b ∧ b = r * c) :=
sorry

end geometric_progression_condition_l2163_216342


namespace power_mul_eq_l2163_216320

variable (a : ℝ)

theorem power_mul_eq :
  (-a)^2 * a^4 = a^6 :=
by sorry

end power_mul_eq_l2163_216320


namespace sum_of_transformed_numbers_l2163_216362

theorem sum_of_transformed_numbers (a b S : ℕ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l2163_216362


namespace original_price_l2163_216377

variable (a : ℝ)

-- Given the price after a 20% discount is a yuan per unit,
-- Prove that the original price per unit was (5/4) * a yuan.
theorem original_price (h : a > 0) : (a / (4 / 5)) = (5 / 4) * a :=
by sorry

end original_price_l2163_216377


namespace evaluate_expression_l2163_216335

theorem evaluate_expression :
  (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))) = (3 / 4) :=
sorry

end evaluate_expression_l2163_216335


namespace tropical_fish_count_l2163_216309

theorem tropical_fish_count (total_fish : ℕ) (koi_count : ℕ) (total_fish_eq : total_fish = 52) (koi_count_eq : koi_count = 37) : 
    (total_fish - koi_count) = 15 := by
    sorry

end tropical_fish_count_l2163_216309
