import Mathlib

namespace bakery_profit_l1946_194632

noncomputable def revenue_per_piece : ℝ := 4
noncomputable def pieces_per_pie : ℕ := 3
noncomputable def pies_per_hour : ℕ := 12
noncomputable def cost_per_pie : ℝ := 0.5

theorem bakery_profit (pieces_per_pie_pos : 0 < pieces_per_pie) 
                      (pies_per_hour_pos : 0 < pies_per_hour) 
                      (cost_per_pie_pos : 0 < cost_per_pie) :
  pies_per_hour * (pieces_per_pie * revenue_per_piece) - (pies_per_hour * cost_per_pie) = 138 := 
sorry

end bakery_profit_l1946_194632


namespace find_p_l1946_194605

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2 * p * x
def quadrilateral_is_rectangle (A B C D : ℝ × ℝ) : Prop := 
  A.1 = C.1 ∧ B.1 = D.1 ∧ A.2 = D.2 ∧ B.2 = C.2

theorem find_p (A B C D : ℝ × ℝ) (p : ℝ) (h1 : ∃ x y, circle_eq x y ∧ parabola_eq p x y) 
  (h2 : ∃ x y, circle_eq x y ∧ x = 0) 
  (h3 : quadrilateral_is_rectangle A B C D) 
  (h4 : 0 < p) : 
  p = 2 := 
sorry

end find_p_l1946_194605


namespace hypotenuse_length_l1946_194652

-- Let a and b be the lengths of the non-hypotenuse sides of a right triangle.
-- We are given that a = 6 and b = 8, and we need to prove that the hypotenuse c is 10.
theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c ^ 2 = a ^ 2 + b ^ 2) : c = 10 :=
by
  -- The proof goes here.
  sorry

end hypotenuse_length_l1946_194652


namespace solve_system_l1946_194607

theorem solve_system (x y : ℝ) (h1 : 4 * x - y = 2) (h2 : 3 * x - 2 * y = -1) : x - y = -1 := 
by
  sorry

end solve_system_l1946_194607


namespace find_number_l1946_194640

theorem find_number (x : ℤ) (h : 33 + 3 * x = 48) : x = 5 :=
by
  sorry

end find_number_l1946_194640


namespace complement_intersection_l1946_194614

open Set

theorem complement_intersection
  (U : Set ℝ) (A B : Set ℝ) 
  (hU : U = univ) 
  (hA : A = { x : ℝ | x ≤ -2 }) 
  (hB : B = { x : ℝ | x < 1 }) :
  (U \ A) ∩ B = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_l1946_194614


namespace sin_identity_proof_l1946_194672

theorem sin_identity_proof (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) :
  Real.sin (5 * π / 6 - x) + Real.sin (π / 3 - x) ^ 2 = 19 / 16 :=
by
  sorry

end sin_identity_proof_l1946_194672


namespace Jazmin_strips_width_l1946_194693

theorem Jazmin_strips_width (w1 w2 g : ℕ) (h1 : w1 = 44) (h2 : w2 = 33) (hg : g = Nat.gcd w1 w2) : g = 11 := by
  -- Markdown above outlines:
  -- w1, w2 are widths of the construction paper
  -- h1: w1 = 44
  -- h2: w2 = 33
  -- hg: g = gcd(w1, w2)
  -- Prove g == 11
  sorry

end Jazmin_strips_width_l1946_194693


namespace combined_average_age_l1946_194637

theorem combined_average_age 
    (avgA : ℕ → ℕ → ℕ) -- defines the average function
    (avgA_cond : avgA 6 240 = 40) 
    (avgB : ℕ → ℕ → ℕ)
    (avgB_cond : avgB 4 100 = 25) 
    (combined_total_age : ℕ := 340) 
    (total_people : ℕ := 10) : avgA (total_people) (combined_total_age) = 34 := 
by
  sorry

end combined_average_age_l1946_194637


namespace quadratic_roots_solve_equation_l1946_194656

theorem quadratic_roots (a b c : ℝ) (x1 x2 : ℝ) (h : a ≠ 0)
  (root_eq : x1 = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
            ∧ x2 = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a))
  (h_eq : a*x^2 + b*x + c = 0) :
  ∀ x, a*x^2 + b*x + c = 0 → x = x1 ∨ x = x2 :=
by
  sorry -- Proof not given

theorem solve_equation (x : ℝ) :
  7*x*(5*x + 2) = 6*(5*x + 2) ↔ x = -2 / 5 ∨ x = 6 / 7 :=
by
  sorry -- Proof not given

end quadratic_roots_solve_equation_l1946_194656


namespace field_size_l1946_194679

theorem field_size
  (cost_per_foot : ℝ)
  (total_money : ℝ)
  (cannot_fence : ℝ)
  (cost_per_foot_eq : cost_per_foot = 30)
  (total_money_eq : total_money = 120000)
  (cannot_fence_eq : cannot_fence > 1000) :
  ∃ (side_length : ℝ), side_length * side_length = 1000000 := 
by
  sorry

end field_size_l1946_194679


namespace solve_floor_sum_eq_125_l1946_194628

def floorSum (x : ℕ) : ℕ :=
  (x - 1) * x * (4 * x + 1) / 6

theorem solve_floor_sum_eq_125 (x : ℕ) (h_pos : 0 < x) : floorSum x = 125 → x = 6 := by
  sorry

end solve_floor_sum_eq_125_l1946_194628


namespace find_M_l1946_194687

theorem find_M 
  (M : ℕ)
  (h : 997 + 999 + 1001 + 1003 + 1005 = 5100 - M) :
  M = 95 :=
by
  sorry

end find_M_l1946_194687


namespace part1_part2_l1946_194677

section
  variable {x a : ℝ}

  def f (x a : ℝ) := |x - a| + 3 * x

  theorem part1 (h : a = 1) : 
    (∀ x, f x a ≥ 3 * x + 2 ↔ (x ≥ 3 ∨ x ≤ -1)) :=
    sorry

  theorem part2 : 
    (∀ x, (f x a) ≤ 0 ↔ (x ≤ -1)) → a = 2 :=
    sorry
end

end part1_part2_l1946_194677


namespace arithmetic_expression_eval_l1946_194624

theorem arithmetic_expression_eval : 
  (1000 * 0.09999) / 10 * 999 = 998001 := 
by 
  sorry

end arithmetic_expression_eval_l1946_194624


namespace pond_diameter_l1946_194690

theorem pond_diameter 
  (h k r : ℝ)
  (H1 : (4 - h) ^ 2 + (11 - k) ^ 2 = r ^ 2)
  (H2 : (12 - h) ^ 2 + (9 - k) ^ 2 = r ^ 2)
  (H3 : (2 - h) ^ 2 + (7 - k) ^ 2 = (r - 1) ^ 2) :
  2 * r = 9.2 :=
sorry

end pond_diameter_l1946_194690


namespace simplify_and_evaluate_l1946_194671

theorem simplify_and_evaluate (x : Real) (h : x = Real.sqrt 2 - 1) :
  ( (1 / (x - 1) - 1 / (x + 1)) / (2 / (x - 1) ^ 2) ) = 1 - Real.sqrt 2 :=
by
  subst h
  sorry

end simplify_and_evaluate_l1946_194671


namespace jenna_filter_change_15th_is_March_l1946_194654

def month_of_nth_change (startMonth interval n : ℕ) : ℕ :=
  ((interval * (n - 1)) % 12 + startMonth) % 12

theorem jenna_filter_change_15th_is_March :
  month_of_nth_change 1 7 15 = 3 := 
  sorry

end jenna_filter_change_15th_is_March_l1946_194654


namespace find_f_2017_l1946_194647

noncomputable def f (x : ℤ) (a α b β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem find_f_2017
(x : ℤ)
(a α b β : ℝ)
(h : f 4 a α b β = 3) :
f 2017 a α b β = -3 := 
sorry

end find_f_2017_l1946_194647


namespace min_value_of_n_l1946_194606

theorem min_value_of_n : 
  ∃ (n : ℕ), (∃ r : ℕ, 4 * n - 7 * r = 0) ∧ n = 7 := 
sorry

end min_value_of_n_l1946_194606


namespace cutoff_score_admission_l1946_194691

theorem cutoff_score_admission (x : ℝ) 
  (h1 : (2 / 5) * (x + 15) + (3 / 5) * (x - 20) = 90) : x = 96 :=
sorry

end cutoff_score_admission_l1946_194691


namespace find_a_l1946_194658

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (1 - x) - Real.log (1 + x) + a

theorem find_a 
  (M : ℝ) (N : ℝ) (a : ℝ)
  (h1 : M = f a (-1/2))
  (h2 : N = f a (1/2))
  (h3 : M + N = 1) :
  a = 1 / 2 := 
sorry

end find_a_l1946_194658


namespace fish_population_estimate_l1946_194633

theorem fish_population_estimate :
  (∀ (x : ℕ),
    ∃ (m n k : ℕ), 
      m = 30 ∧
      k = 2 ∧
      n = 30 ∧
      ((k : ℚ) / n = m / x) → x = 450) :=
by
  sorry

end fish_population_estimate_l1946_194633


namespace age_ratio_l1946_194681

-- Definitions as per the conditions
variable (j e x : ℕ)

-- Conditions from the problem
def condition1 : Prop := j - 4 = 2 * (e - 4)
def condition2 : Prop := j - 10 = 3 * (e - 10)

-- The statement we need to prove
theorem age_ratio (j e x : ℕ) (h1 : condition1 j e)
(h2 : condition2 j e) :
(j + x) * 2 = (e + x) * 3 ↔ x = 8 :=
sorry

end age_ratio_l1946_194681


namespace solve_quadratics_and_sum_l1946_194667

theorem solve_quadratics_and_sum (d e f : ℤ) 
  (h1 : ∃ d e : ℤ, d + e = 19 ∧ d * e = 88) 
  (h2 : ∃ e f : ℤ, e + f = 23 ∧ e * f = 120) : 
  d + e + f = 31 := by
  sorry

end solve_quadratics_and_sum_l1946_194667


namespace num_decompositions_144_l1946_194692

theorem num_decompositions_144 : ∃ D, D = 45 ∧ 
  (∀ (factors : List ℕ), 
    (∀ x, x ∈ factors → x > 1) ∧ factors.prod = 144 → 
    factors.permutations.length = D) :=
sorry

end num_decompositions_144_l1946_194692


namespace total_weight_on_scale_l1946_194600

def weight_blue_ball : ℝ := 6
def weight_brown_ball : ℝ := 3.12

theorem total_weight_on_scale :
  weight_blue_ball + weight_brown_ball = 9.12 :=
by sorry

end total_weight_on_scale_l1946_194600


namespace tan_alpha_plus_pi_over_4_l1946_194697

noncomputable def sin_cos_identity (α : ℝ) : Prop :=
  (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 3

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : sin_cos_identity α) :
  Real.tan (α + Real.pi / 4) = -3 :=
  by
  sorry

end tan_alpha_plus_pi_over_4_l1946_194697


namespace vasya_numbers_l1946_194695

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l1946_194695


namespace starting_number_is_10_l1946_194665

axiom between_nums_divisible_by_10 (n : ℕ) : 
  (∃ start : ℕ, start ≤ n ∧ n ≤ 76 ∧ 
  ∀ m, start ≤ m ∧ m ≤ n → m % 10 = 0 ∧ 
  (¬ (76 % 10 = 0) → start = 10) ∧ 
  ((76 - (76 % 10)) / 10 = 6) )

theorem starting_number_is_10 
  (start : ℕ) 
  (h1 : ∃ n, (start ≤ n ∧ n ≤ 76 ∧ 
             ∀ m, start ≤ m ∧ m ≤ n → m % 10 = 0 ∧ 
             (n - start) / 10 = 6)):
  start = 10 :=
sorry

end starting_number_is_10_l1946_194665


namespace negation_of_existence_l1946_194609

theorem negation_of_existence (p : Prop) (h : ∃ (c : ℝ), c > 0 ∧ (∃ (x : ℝ), x^2 - x + c = 0)) : 
  ¬ (∃ (c : ℝ), c > 0 ∧ (∃ (x : ℝ), x^2 - x + c = 0)) ↔ 
  ∀ (c : ℝ), c > 0 → ¬ (∃ (x : ℝ), x^2 - x + c = 0) :=
by 
  sorry

end negation_of_existence_l1946_194609


namespace second_largest_of_five_consecutive_is_19_l1946_194617

theorem second_largest_of_five_consecutive_is_19 (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 90): 
  n + 3 = 19 :=
by sorry

end second_largest_of_five_consecutive_is_19_l1946_194617


namespace geometric_sum_of_ratios_l1946_194698

theorem geometric_sum_of_ratios (k p r : ℝ) (a2 a3 b2 b3 : ℝ) 
  (ha2 : a2 = k * p) (ha3 : a3 = k * p^2) 
  (hb2 : b2 = k * r) (hb3 : b3 = k * r^2) 
  (h : a3 - b3 = 5 * (a2 - b2)) :
  p + r = 5 :=
by {
  sorry
}

end geometric_sum_of_ratios_l1946_194698


namespace fractions_product_l1946_194608

theorem fractions_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by 
  sorry

end fractions_product_l1946_194608


namespace problem1_problem2_l1946_194657

open Real

/-- Problem 1: Simplify trigonometric expression. -/
theorem problem1 : 
  (sqrt (1 - 2 * sin (10 * pi / 180) * cos (10 * pi / 180)) /
  (sin (170 * pi / 180) - sqrt (1 - sin (170 * pi / 180)^2))) = -1 :=
sorry

/-- Problem 2: Given tan(θ) = 2, find the value.
  Required to prove: 2 + sin(θ) * cos(θ) - cos(θ)^2 equals 11/5 -/
theorem problem2 (θ : ℝ) (h : tan θ = 2) :
  2 + sin θ * cos θ - cos θ^2 = 11 / 5 :=
sorry

end problem1_problem2_l1946_194657


namespace lisa_children_l1946_194641

theorem lisa_children (C : ℕ) 
  (h1 : 5 * 52 = 260)
  (h2 : (2 * C + 3 + 2) * 260 = 3380) : 
  C = 4 := 
by
  sorry

end lisa_children_l1946_194641


namespace value_2_std_devs_less_than_mean_l1946_194683

-- Define the arithmetic mean
def mean : ℝ := 15.5

-- Define the standard deviation
def standard_deviation : ℝ := 1.5

-- Define the value that is 2 standard deviations less than the mean
def value_2_std_less_than_mean : ℝ := mean - 2 * standard_deviation

-- The theorem we want to prove
theorem value_2_std_devs_less_than_mean : value_2_std_less_than_mean = 12.5 := by
  sorry

end value_2_std_devs_less_than_mean_l1946_194683


namespace back_seat_tickets_sold_l1946_194623

variable (M B : ℕ)

theorem back_seat_tickets_sold:
  M + B = 20000 ∧ 55 * M + 45 * B = 955000 → B = 14500 :=
by
  sorry

end back_seat_tickets_sold_l1946_194623


namespace parabola_reflection_translation_l1946_194625

open Real

noncomputable def f (a b c x : ℝ) : ℝ := a * (x - 4)^2 + b * (x - 4) + c
noncomputable def g (a b c x : ℝ) : ℝ := -a * (x + 4)^2 - b * (x + 4) - c
noncomputable def fg_x (a b c x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_reflection_translation (a b c x : ℝ) (ha : a ≠ 0) :
  fg_x a b c x = -16 * a * x :=
by
  sorry

end parabola_reflection_translation_l1946_194625


namespace ratio_of_women_to_men_l1946_194612

theorem ratio_of_women_to_men (M W : ℕ) 
  (h1 : M + W = 72) 
  (h2 : M - 16 = W + 8) : 
  W / M = 1 / 2 :=
sorry

end ratio_of_women_to_men_l1946_194612


namespace maria_chairs_l1946_194622

variable (C : ℕ) -- Number of chairs Maria bought
variable (tables : ℕ := 2) -- Number of tables Maria bought is 2
variable (time_per_furniture : ℕ := 8) -- Time spent on each piece of furniture in minutes
variable (total_time : ℕ := 32) -- Total time spent assembling furniture

theorem maria_chairs :
  (time_per_furniture * C + time_per_furniture * tables = total_time) → C = 2 :=
by
  intro h
  sorry

end maria_chairs_l1946_194622


namespace symmetric_point_yoz_l1946_194648

theorem symmetric_point_yoz (x y z : ℝ) (hx : x = 2) (hy : y = 3) (hz : z = 4) :
  (-x, y, z) = (-2, 3, 4) :=
by
  -- The proof is skipped
  sorry

end symmetric_point_yoz_l1946_194648


namespace symmetric_circle_with_respect_to_origin_l1946_194676

theorem symmetric_circle_with_respect_to_origin :
  ∀ x y : ℝ, (x + 2) ^ 2 + (y - 1) ^ 2 = 1 → (x - 2) ^ 2 + (y + 1) ^ 2 = 1 :=
by
  intros x y h
  -- Symmetric transformation and verification will be implemented here
  sorry

end symmetric_circle_with_respect_to_origin_l1946_194676


namespace union_sets_example_l1946_194674

theorem union_sets_example : ({0, 1} ∪ {2} : Set ℕ) = {0, 1, 2} := by 
  sorry

end union_sets_example_l1946_194674


namespace arithmetic_progression_exists_l1946_194619

theorem arithmetic_progression_exists (a_1 a_2 a_3 a_4 : ℕ) (d : ℕ) :
  a_2 = a_1 + d →
  a_3 = a_1 + 2 * d →
  a_4 = a_1 + 3 * d →
  a_1 * a_2 * a_3 = 6 →
  a_1 * a_2 * a_3 * a_4 = 24 →
  a_1 = 1 ∧ a_2 = 2 ∧ a_3 = 3 ∧ a_4 = 4 :=
by
  sorry

end arithmetic_progression_exists_l1946_194619


namespace billboards_and_road_length_l1946_194669

theorem billboards_and_road_length :
  ∃ (x y : ℕ), 5 * (x + 21 - 1) = y ∧ (55 * (x - 1)) / 10 = y ∧ x = 200 ∧ y = 1100 :=
sorry

end billboards_and_road_length_l1946_194669


namespace imaginary_part_of_z_l1946_194649

-- Define the complex number z
def z : Complex := Complex.mk 3 (-4)

-- State the proof goal
theorem imaginary_part_of_z : z.im = -4 :=
by
  sorry

end imaginary_part_of_z_l1946_194649


namespace max_pages_l1946_194611

/-- Prove that the maximum number of pages the book has is 208 -/
theorem max_pages (pages: ℕ) (h1: pages ≥ 16 * 12 + 1) (h2: pages ≤ 13 * 16) 
(h3: pages ≥ 20 * 10 + 1) (h4: pages ≤ 11 * 20) : 
  pages ≤ 208 :=
by
  -- proof to be filled in
  sorry

end max_pages_l1946_194611


namespace vector_sum_magnitude_eq_2_or_5_l1946_194638

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := 3
def equal_angles (θ : ℝ) := θ = 120 ∨ θ = 0

theorem vector_sum_magnitude_eq_2_or_5
  (a_mag : ℝ := a)
  (b_mag : ℝ := b)
  (c_mag : ℝ := c)
  (θ : ℝ)
  (Hθ : equal_angles θ) :
  (|a_mag| = 1) ∧ (|b_mag| = 1) ∧ (|c_mag| = 3) →
  (|a_mag + b_mag + c_mag| = 2 ∨ |a_mag + b_mag + c_mag| = 5) :=
by
  sorry

end vector_sum_magnitude_eq_2_or_5_l1946_194638


namespace sequence_formula_l1946_194643

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | 2 => 6
  | 3 => 10
  | _ => sorry  -- The pattern is more general

theorem sequence_formula (n : ℕ) : a n = (n * (n + 1)) / 2 := 
  sorry

end sequence_formula_l1946_194643


namespace general_formulas_max_b_seq_l1946_194678

noncomputable def a_seq (n : ℕ) : ℕ := 4 * n - 2
noncomputable def b_seq (n : ℕ) : ℕ := 4 * n - 2 - 2^(n - 1)

-- The general formulas to be proved
theorem general_formulas :
  (∀ n : ℕ, a_seq n = 4 * n - 2) ∧ 
  (∀ n : ℕ, b_seq n = 4 * n - 2 - 2^(n - 1)) :=
by
  sorry

-- The maximum value condition to be proved
theorem max_b_seq :
  ((∀ n : ℕ, b_seq n ≤ b_seq 3) ∨ (∀ n : ℕ, b_seq n ≤ b_seq 4)) :=
by
  sorry

end general_formulas_max_b_seq_l1946_194678


namespace mod_remainder_l1946_194661

theorem mod_remainder (a b c : ℕ) : 
  (7 * 10 ^ 20 + 1 ^ 20) % 11 = 8 := by
  -- Lean proof will be written here
  sorry

end mod_remainder_l1946_194661


namespace task_force_combinations_l1946_194613

theorem task_force_combinations :
  (Nat.choose 10 4) * (Nat.choose 7 3) = 7350 :=
by
  sorry

end task_force_combinations_l1946_194613


namespace yang_hui_problem_l1946_194659

theorem yang_hui_problem (x : ℝ) :
  x * (x + 12) = 864 :=
sorry

end yang_hui_problem_l1946_194659


namespace inequality_proof_l1946_194627

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_proof_l1946_194627


namespace percentage_increase_in_average_visibility_l1946_194682

theorem percentage_increase_in_average_visibility :
  let avg_visibility_without_telescope := (100 + 110) / 2
  let avg_visibility_with_telescope := (150 + 165) / 2
  let increase_in_avg_visibility := avg_visibility_with_telescope - avg_visibility_without_telescope
  let percentage_increase := (increase_in_avg_visibility / avg_visibility_without_telescope) * 100
  percentage_increase = 50 := by
  -- calculations are omitted; proof goes here
  sorry

end percentage_increase_in_average_visibility_l1946_194682


namespace country_X_tax_l1946_194686

theorem country_X_tax (I T x : ℝ) (hI : I = 51999.99) (hT : T = 8000) (h : T = 0.14 * x + 0.20 * (I - x)) : 
  x = 39999.97 := sorry

end country_X_tax_l1946_194686


namespace max_slope_tangent_eqn_l1946_194644

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem max_slope_tangent_eqn (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) :
    (∃ m b, m = Real.sqrt 2 ∧ b = -Real.sqrt 2 * (Real.pi / 4) ∧ 
    (∀ y, y = m * x + b)) :=
sorry

end max_slope_tangent_eqn_l1946_194644


namespace equation_one_solution_equation_two_solution_l1946_194651

theorem equation_one_solution (x : ℝ) : ((x + 3) ^ 2 - 9 = 0) ↔ (x = 0 ∨ x = -6) := by
  sorry

theorem equation_two_solution (x : ℝ) : (x ^ 2 - 4 * x + 1 = 0) ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := by
  sorry

end equation_one_solution_equation_two_solution_l1946_194651


namespace temperature_problem_l1946_194635

theorem temperature_problem
  (M L N : ℝ)
  (h1 : M = L + N)
  (h2 : M - 9 = M - 9)
  (h3 : L + 5 = L + 5)
  (h4 : abs (M - 9 - (L + 5)) = 1) :
  (N = 15 ∨ N = 13) → (N = 15 ∧ N = 13 → 15 * 13 = 195) :=
by
  sorry

end temperature_problem_l1946_194635


namespace mary_stickers_left_l1946_194662

def initial_stickers : ℕ := 50
def stickers_per_friend : ℕ := 4
def number_of_friends : ℕ := 5
def total_students_including_mary : ℕ := 17
def stickers_per_other_student : ℕ := 2

theorem mary_stickers_left :
  let friends_stickers := stickers_per_friend * number_of_friends
  let other_students := total_students_including_mary - 1 - number_of_friends
  let other_students_stickers := stickers_per_other_student * other_students
  let total_given_away := friends_stickers + other_students_stickers
  initial_stickers - total_given_away = 8 :=
by
  sorry

end mary_stickers_left_l1946_194662


namespace smallest_number_among_bases_l1946_194618

noncomputable def convert_base_9 (n : ℕ) : ℕ :=
match n with
| 85 => 8 * 9 + 5
| _ => 0

noncomputable def convert_base_4 (n : ℕ) : ℕ :=
match n with
| 1000 => 1 * 4^3
| _ => 0

noncomputable def convert_base_2 (n : ℕ) : ℕ :=
match n with
| 111111 => 1 * 2^6 - 1
| _ => 0

theorem smallest_number_among_bases:
  min (min (convert_base_9 85) (convert_base_4 1000)) (convert_base_2 111111) = convert_base_2 111111 :=
by {
  sorry
}

end smallest_number_among_bases_l1946_194618


namespace molecular_weight_proof_l1946_194626

def atomic_weight_Al : Float := 26.98
def atomic_weight_O : Float := 16.00
def atomic_weight_H : Float := 1.01

def molecular_weight_AlOH3 : Float :=
  (1 * atomic_weight_Al) + (3 * atomic_weight_O) + (3 * atomic_weight_H)

def moles : Float := 7.0

def molecular_weight_7_moles_AlOH3 : Float :=
  moles * molecular_weight_AlOH3

theorem molecular_weight_proof : molecular_weight_7_moles_AlOH3 = 546.07 :=
by
  /- Here we calculate the molecular weight of Al(OH)3 and multiply it by 7.
     molecular_weight_AlOH3 = (1 * 26.98) + (3 * 16.00) + (3 * 1.01) = 78.01
     molecular_weight_7_moles_AlOH3 = 7 * 78.01 = 546.07 -/
  sorry

end molecular_weight_proof_l1946_194626


namespace pond_field_ratio_l1946_194684

theorem pond_field_ratio (L W : ℕ) (pond_side : ℕ) (hL : L = 24) (hLW : L = 2 * W) (hPond : pond_side = 6) :
  pond_side * pond_side / (L * W) = 1 / 8 :=
by
  sorry

end pond_field_ratio_l1946_194684


namespace albert_mary_age_ratio_l1946_194620

theorem albert_mary_age_ratio
  (A M B : ℕ)
  (h1 : A = 4 * B)
  (h2 : M = A - 14)
  (h3 : B = 7)
  :
  A / M = 2 := 
by sorry

end albert_mary_age_ratio_l1946_194620


namespace starting_player_ensures_non_trivial_solution_l1946_194602

theorem starting_player_ensures_non_trivial_solution :
  ∀ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℚ), 
    ∃ (x y z : ℚ), 
    ((a1 * x + b1 * y + c1 * z = 0) ∧ 
     (a2 * x + b2 * y + c2 * z = 0) ∧ 
     (a3 * x + b3 * y + c3 * z = 0)) 
    ∧ ((a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2) = 0) ∧ 
         (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by
  intros a1 b1 c1 a2 b2 c2 a3 b3 c3
  sorry

end starting_player_ensures_non_trivial_solution_l1946_194602


namespace parallelogram_sides_eq_l1946_194660

theorem parallelogram_sides_eq (x y : ℚ) :
  (5 * x - 2 = 10 * x - 4) → 
  (3 * y + 7 = 6 * y + 13) → 
  x + y = -1.6 := by
  sorry

end parallelogram_sides_eq_l1946_194660


namespace geometric_series_sum_l1946_194604

theorem geometric_series_sum :
  ∑' n : ℕ, (2 : ℝ) * (1 / 4) ^ n = 8 / 3 := by
  sorry

end geometric_series_sum_l1946_194604


namespace eval_expression_l1946_194699

theorem eval_expression : (2^5 - 5^2) = 7 :=
by {
  -- Proof steps will be here
  sorry
}

end eval_expression_l1946_194699


namespace solve_for_N_l1946_194696

theorem solve_for_N : ∃ N : ℕ, 32^4 * 4^5 = 2^N ∧ N = 30 := by
  sorry

end solve_for_N_l1946_194696


namespace expression_evaluation_l1946_194636

-- Definitions of the expressions
def expr (x y : ℤ) : ℤ :=
  ((x - 2 * y) ^ 2 + (3 * x - y) * (3 * x + y) - 3 * y ^ 2) / (-2 * x)

-- Proof that the expression evaluates to -11 when x = 1 and y = -3
theorem expression_evaluation : expr 1 (-3) = -11 :=
by
  -- Declarations
  let x := 1
  let y := -3
  -- The core calculation
  show expr x y = -11
  sorry

end expression_evaluation_l1946_194636


namespace perimeter_of_playground_l1946_194664

theorem perimeter_of_playground 
  (x y : ℝ) 
  (h1 : x^2 + y^2 = 900) 
  (h2 : x * y = 216) : 
  2 * (x + y) = 72 := 
by 
  sorry

end perimeter_of_playground_l1946_194664


namespace seth_pounds_lost_l1946_194666

-- Definitions
def pounds_lost_by_Seth (S : ℝ) : Prop := 
  let total_loss := S + 3 * S + (S + 1.5)
  total_loss = 89

theorem seth_pounds_lost (S : ℝ) : pounds_lost_by_Seth S → S = 17.5 := by
  sorry

end seth_pounds_lost_l1946_194666


namespace john_total_expense_l1946_194610

-- Define variables
variables (M D : ℝ)

-- Define the conditions
axiom cond1 : M = 20 * D
axiom cond2 : M = 24 * (D - 3)

-- State the theorem to prove
theorem john_total_expense : M = 360 :=
by
  -- Add the proof steps here
  sorry

end john_total_expense_l1946_194610


namespace smallest_four_digits_valid_remainder_l1946_194650

def isFourDigit (x : ℕ) : Prop := 1000 ≤ x ∧ x ≤ 9999 

def validRemainder (x : ℕ) : Prop := 
  ∀ k ∈ [2, 3, 4, 5, 6], x % k = 1

theorem smallest_four_digits_valid_remainder :
  ∃ x1 x2 x3 x4 : ℕ,
    isFourDigit x1 ∧ validRemainder x1 ∧
    isFourDigit x2 ∧ validRemainder x2 ∧
    isFourDigit x3 ∧ validRemainder x3 ∧
    isFourDigit x4 ∧ validRemainder x4 ∧
    x1 = 1021 ∧ x2 = 1081 ∧ x3 = 1141 ∧ x4 = 1201 := 
sorry

end smallest_four_digits_valid_remainder_l1946_194650


namespace problem_solution_l1946_194673

theorem problem_solution (x : ℕ) (h : x = 3) : x + x * x^(x^2) = 59052 :=
by
  rw [h]
  -- The condition is now x = 3
  let t := 3 + 3 * 3^(3^2)
  have : t = 59052 := sorry
  exact this

end problem_solution_l1946_194673


namespace problems_completed_l1946_194629

theorem problems_completed (p t : ℕ) (h1 : p ≥ 15) (h2 : p * t = (2 * p - 10) * (t - 1)) : p * t = 60 := sorry

end problems_completed_l1946_194629


namespace sheena_completes_in_37_weeks_l1946_194621

-- Definitions based on the conditions
def hours_per_dress : List Nat := [15, 18, 20, 22, 24, 26, 28]
def hours_cycle : List Nat := [5, 3, 6, 4]
def finalize_hours : Nat := 10

-- The total hours needed to sew all dresses
def total_dress_hours : Nat := hours_per_dress.sum

-- The total hours needed including finalizing hours
def total_hours : Nat := total_dress_hours + finalize_hours

-- Total hours sewed in each 4-week cycle
def hours_per_cycle : Nat := hours_cycle.sum

-- Total number of weeks it will take to complete all dresses
def weeks_needed : Nat := 4 * ((total_hours + hours_per_cycle - 1) / hours_per_cycle)
def additional_weeks : Nat := if total_hours % hours_per_cycle == 0 then 0 else 1

theorem sheena_completes_in_37_weeks : weeks_needed + additional_weeks = 37 := by
  sorry

end sheena_completes_in_37_weeks_l1946_194621


namespace rainfall_wednesday_correct_l1946_194631

def monday_rainfall : ℝ := 0.9
def tuesday_rainfall : ℝ := monday_rainfall - 0.7
def wednesday_rainfall : ℝ := 2 * (monday_rainfall + tuesday_rainfall)

theorem rainfall_wednesday_correct : wednesday_rainfall = 2.2 := by
sorry

end rainfall_wednesday_correct_l1946_194631


namespace undefined_expression_iff_l1946_194634

theorem undefined_expression_iff (x : ℝ) :
  (x^2 - 24 * x + 144 = 0) ↔ (x = 12) := 
sorry

end undefined_expression_iff_l1946_194634


namespace product_of_solutions_l1946_194685

theorem product_of_solutions (x : ℚ) (h : abs (12 / x + 3) = 2) :
  x = -12 ∨ x = -12 / 5 → x₁ * x₂ = 144 / 5 := by
  sorry

end product_of_solutions_l1946_194685


namespace degrees_to_radians_18_l1946_194688

theorem degrees_to_radians_18 (degrees : ℝ) (h : degrees = 18) : 
  (degrees * (Real.pi / 180) = Real.pi / 10) :=
by
  sorry

end degrees_to_radians_18_l1946_194688


namespace parallel_segments_k_value_l1946_194668

open Real

theorem parallel_segments_k_value :
  let A' := (-6, 0)
  let B' := (0, -6)
  let X' := (0, 12)
  ∃ k : ℝ,
  let Y' := (18, k)
  let m_ab := (B'.2 - A'.2) / (B'.1 - A'.1)
  let m_xy := (Y'.2 - X'.2) / (Y'.1 - X'.1)
  m_ab = m_xy → k = -6 :=
by
  sorry

end parallel_segments_k_value_l1946_194668


namespace remainder_when_2x_div_8_is_1_l1946_194642

theorem remainder_when_2x_div_8_is_1 (x y : ℤ) 
  (h1 : x = 11 * y + 4)
  (h2 : ∃ r : ℤ, 2 * x = 8 * (3 * y) + r)
  (h3 : 13 * y - x = 3) : ∃ r : ℤ, r = 1 :=
by
  sorry

end remainder_when_2x_div_8_is_1_l1946_194642


namespace complete_square_transform_l1946_194663

theorem complete_square_transform (x : ℝ) (h : x^2 + 8*x + 7 = 0) : (x + 4)^2 = 9 :=
by sorry

end complete_square_transform_l1946_194663


namespace petri_dish_count_l1946_194694

theorem petri_dish_count (total_germs : ℝ) (germs_per_dish : ℝ) (h1 : total_germs = 0.036 * 10^5) (h2 : germs_per_dish = 199.99999999999997) :
  total_germs / germs_per_dish = 18 :=
by
  sorry

end petri_dish_count_l1946_194694


namespace even_function_f_l1946_194645

noncomputable def f (x : ℝ) : ℝ := if 0 < x ∧ x < 10 then Real.log x else 0

theorem even_function_f (x : ℝ) (h : f (-x) = f x) (h1 : ∀ x, 0 < x ∧ x < 10 → f x = Real.log x) :
  f (-Real.exp 1) + f (Real.exp 2) = 3 := by
  sorry

end even_function_f_l1946_194645


namespace dogwood_trees_current_l1946_194670

variable (X : ℕ)
variable (trees_today : ℕ := 41)
variable (trees_tomorrow : ℕ := 20)
variable (total_trees_after : ℕ := 100)

theorem dogwood_trees_current (h : X + trees_today + trees_tomorrow = total_trees_after) : X = 39 :=
by
  sorry

end dogwood_trees_current_l1946_194670


namespace inverse_variation_l1946_194630

theorem inverse_variation (w : ℝ) (h1 : ∃ (c : ℝ), ∀ (x : ℝ), x^4 * w^(1/4) = c)
  (h2 : (3 : ℝ)^4 * (16 : ℝ)^(1/4) = (6 : ℝ)^4 * w^(1/4)) : 
  w = 1 / 4096 :=
by
  sorry

end inverse_variation_l1946_194630


namespace f_is_even_f_range_l1946_194615

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (|x| + 2) / (1 - |x|)

-- Prove that f(x) is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

-- Prove the range of f(x) is (-∞, -1) ∪ [2, +∞)
theorem f_range : ∀ y : ℝ, ∃ x : ℝ, y = f x ↔ y ≥ 2 ∨ y < -1 := by
  sorry

end f_is_even_f_range_l1946_194615


namespace arithmetic_progression_power_of_two_l1946_194680

theorem arithmetic_progression_power_of_two 
  (a d : ℤ) (n : ℕ) (k : ℕ) 
  (Sn : ℤ)
  (h_sum : Sn = 2^k)
  (h_ap : Sn = n * (2 * a + (n - 1) * d) / 2)  :
  ∃ m : ℕ, n = 2^m := 
sorry

end arithmetic_progression_power_of_two_l1946_194680


namespace probability_longer_piece_at_least_x_squared_l1946_194601

noncomputable def probability_longer_piece (x : ℝ) : ℝ :=
  if x = 0 then 1 else (2 / (x^2 + 1))

theorem probability_longer_piece_at_least_x_squared (x : ℝ) :
  probability_longer_piece x = (2 / (x^2 + 1)) :=
sorry

end probability_longer_piece_at_least_x_squared_l1946_194601


namespace floor_value_correct_l1946_194639

def calc_floor_value : ℤ :=
  let a := (15 : ℚ) / 8
  let b := a^2
  let c := (225 : ℚ) / 64
  let d := 4
  let e := (19 : ℚ) / 5
  let f := d + e
  ⌊f⌋

theorem floor_value_correct : calc_floor_value = 7 := by
  sorry

end floor_value_correct_l1946_194639


namespace triangle_min_perimeter_l1946_194689

-- Definitions of points A, B, and C and the conditions specified in the problem.
def pointA : ℝ × ℝ := (3, 2)
def pointB (t : ℝ) : ℝ × ℝ := (t, t)
def pointC (c : ℝ) : ℝ × ℝ := (c, 0)

-- Main theorem which states that the minimum perimeter of triangle ABC is sqrt(26).
theorem triangle_min_perimeter : 
  ∃ (B C : ℝ × ℝ), B = pointB (B.1) ∧ C = pointC (C.1) ∧ 
  ∀ (B' C' : ℝ × ℝ), B' = pointB (B'.1) ∧ C' = pointC (C'.1) →
  (dist pointA B + dist B C + dist C pointA ≥ dist (2, 3) (3, -2)) :=
by 
  sorry

end triangle_min_perimeter_l1946_194689


namespace average_age_in_club_l1946_194616

theorem average_age_in_club (women men children : ℕ) 
    (avg_age_women avg_age_men avg_age_children : ℤ)
    (hw : women = 12) (hm : men = 18) (hc : children = 20)
    (haw : avg_age_women = 32) (ham : avg_age_men = 36) (hac : avg_age_children = 10) :
    (12 * 32 + 18 * 36 + 20 * 10) / (12 + 18 + 20) = 24 := by
  sorry

end average_age_in_club_l1946_194616


namespace number_of_schools_l1946_194653

theorem number_of_schools (total_students d : ℕ) (S : ℕ) (ellen frank : ℕ) (d_median : total_students = 2 * d - 1)
    (d_highest : ellen < d) (ellen_position : ellen = 29) (frank_position : frank = 50) (team_size : ∀ S, total_students = 3 * S) : 
    S = 19 := 
by 
  sorry

end number_of_schools_l1946_194653


namespace frank_reads_pages_per_day_l1946_194603

-- Define the conditions and problem statement
def total_pages : ℕ := 450
def total_chapters : ℕ := 41
def total_days : ℕ := 30

-- The derived value we need to prove
def pages_per_day : ℕ := total_pages / total_days

-- The theorem to prove
theorem frank_reads_pages_per_day : pages_per_day = 15 :=
  by
  -- Proof goes here
  sorry

end frank_reads_pages_per_day_l1946_194603


namespace evaluate_expression_l1946_194646

theorem evaluate_expression : 
  3 * (-4) - ((5 * (-5)) * (-2)) + 6 = -56 := 
by 
  sorry

end evaluate_expression_l1946_194646


namespace shopkeeper_percentage_gain_l1946_194655

theorem shopkeeper_percentage_gain (false_weight true_weight : ℝ) 
    (h_false_weight : false_weight = 930)
    (h_true_weight : true_weight = 1000) : 
    (true_weight - false_weight) / false_weight * 100 = 7.53 := 
by
  rw [h_false_weight, h_true_weight]
  sorry

end shopkeeper_percentage_gain_l1946_194655


namespace number_of_sheep_l1946_194675

theorem number_of_sheep (legs animals : ℕ) (h1 : legs = 60) (h2 : animals = 20)
  (chickens sheep : ℕ) (hc : chickens + sheep = animals) (hl : 2 * chickens + 4 * sheep = legs) :
  sheep = 10 :=
sorry

end number_of_sheep_l1946_194675
