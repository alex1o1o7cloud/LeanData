import Mathlib

namespace NUMINAMATH_GPT_union_of_sets_l56_5618

def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {3, 4, 5}

theorem union_of_sets : M ∪ N = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_GPT_union_of_sets_l56_5618


namespace NUMINAMATH_GPT_range_a_real_numbers_l56_5668

theorem range_a_real_numbers (x a : ℝ) : 
  (∀ x : ℝ, (x - a) * (1 - (x + a)) < 1) → (a ∈ Set.univ) :=
by
  sorry

end NUMINAMATH_GPT_range_a_real_numbers_l56_5668


namespace NUMINAMATH_GPT_discarded_second_number_l56_5612

-- Define the conditions
def avg_original_50 : ℝ := 38
def total_sum_50_numbers : ℝ := 50 * avg_original_50
def discarded_first : ℝ := 45
def avg_remaining_48 : ℝ := 37.5
def total_sum_remaining_48 : ℝ := 48 * avg_remaining_48
def sum_discarded := total_sum_50_numbers - total_sum_remaining_48

-- Define the proof statement
theorem discarded_second_number (x : ℝ) (h : discarded_first + x = sum_discarded) : x = 55 :=
by
  sorry

end NUMINAMATH_GPT_discarded_second_number_l56_5612


namespace NUMINAMATH_GPT_problem_statement_l56_5660

theorem problem_statement (x : ℝ) (h : x - 1/x = 5) : x^4 - (1 / x)^4 = 527 :=
sorry

end NUMINAMATH_GPT_problem_statement_l56_5660


namespace NUMINAMATH_GPT_inequality_solution_set_l56_5685

theorem inequality_solution_set (x : ℝ) : (x^2 ≥ 4) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l56_5685


namespace NUMINAMATH_GPT_find_n_l56_5666

theorem find_n (n : ℕ) (h : Nat.lcm n (n - 30) = n + 1320) : n = 165 := 
sorry

end NUMINAMATH_GPT_find_n_l56_5666


namespace NUMINAMATH_GPT_B_profit_l56_5680

-- Definitions based on conditions
def investment_ratio (B_invest A_invest : ℕ) : Prop := A_invest = 3 * B_invest
def period_ratio (B_period A_period : ℕ) : Prop := A_period = 2 * B_period
def total_profit (total : ℕ) : Prop := total = 28000
def B_share (total : ℕ) := total / 7

-- Theorem statement based on the proof problem
theorem B_profit (B_invest A_invest B_period A_period total : ℕ)
  (h1 : investment_ratio B_invest A_invest)
  (h2 : period_ratio B_period A_period)
  (h3 : total_profit total) :
  B_share total = 4000 :=
by
  sorry

end NUMINAMATH_GPT_B_profit_l56_5680


namespace NUMINAMATH_GPT_abs_ab_eq_2128_l56_5686

theorem abs_ab_eq_2128 (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ r s : ℤ, r ≠ s ∧ ∃ r' : ℤ, r' = r ∧ 
          (x^3 + a * x^2 + b * x + 16 * a = (x - r)^2 * (x - s) ∧ r * r * s = -16 * a)) :
  |a * b| = 2128 :=
sorry

end NUMINAMATH_GPT_abs_ab_eq_2128_l56_5686


namespace NUMINAMATH_GPT_gcd_b2_add_11b_add_28_b_add_6_eq_2_l56_5627

theorem gcd_b2_add_11b_add_28_b_add_6_eq_2 {b : ℤ} (h : ∃ k : ℤ, b = 1836 * k) : 
  Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
by
  sorry

end NUMINAMATH_GPT_gcd_b2_add_11b_add_28_b_add_6_eq_2_l56_5627


namespace NUMINAMATH_GPT_minimum_value_of_f_l56_5605

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3 * x + 9

-- State the theorem about the minimum value of the function
theorem minimum_value_of_f : ∃ x : ℝ, f x = 7 ∧ ∀ y : ℝ, f y ≥ 7 := sorry

end NUMINAMATH_GPT_minimum_value_of_f_l56_5605


namespace NUMINAMATH_GPT_interval_of_x_l56_5622

theorem interval_of_x (x : ℝ) :
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) := by
  sorry

end NUMINAMATH_GPT_interval_of_x_l56_5622


namespace NUMINAMATH_GPT_model_to_statue_ratio_l56_5613

theorem model_to_statue_ratio (h_statue : ℝ) (h_model : ℝ) (h_statue_eq : h_statue = 60) (h_model_eq : h_model = 4) :
  (h_statue / h_model) = 15 := by
  sorry

end NUMINAMATH_GPT_model_to_statue_ratio_l56_5613


namespace NUMINAMATH_GPT_compare_neg_thirds_and_halves_l56_5692

theorem compare_neg_thirds_and_halves : (-1 : ℚ) / 3 > (-1 : ℚ) / 2 :=
by
  sorry

end NUMINAMATH_GPT_compare_neg_thirds_and_halves_l56_5692


namespace NUMINAMATH_GPT_harriet_ran_48_miles_l56_5636

def total_distance : ℕ := 195
def katarina_distance : ℕ := 51
def equal_distance (n : ℕ) : Prop := (total_distance - katarina_distance) = 3 * n
def harriet_distance : ℕ := 48

theorem harriet_ran_48_miles
  (total_eq : total_distance = 195)
  (kat_eq : katarina_distance = 51)
  (equal_dist_eq : equal_distance harriet_distance) :
  harriet_distance = 48 :=
by
  sorry

end NUMINAMATH_GPT_harriet_ran_48_miles_l56_5636


namespace NUMINAMATH_GPT_johns_total_due_l56_5699

noncomputable def total_amount_due (initial_amount : ℝ) (first_charge_rate : ℝ) 
  (second_charge_rate : ℝ) (third_charge_rate : ℝ) : ℝ := 
  let after_first_charge := initial_amount * first_charge_rate
  let after_second_charge := after_first_charge * second_charge_rate
  let after_third_charge := after_second_charge * third_charge_rate
  after_third_charge

theorem johns_total_due : total_amount_due 500 1.02 1.03 1.025 = 538.43 := 
  by
    -- The proof would go here.
    sorry

end NUMINAMATH_GPT_johns_total_due_l56_5699


namespace NUMINAMATH_GPT_find_product_l56_5662

theorem find_product (a b c d : ℚ) 
  (h₁ : 2 * a + 4 * b + 6 * c + 8 * d = 48)
  (h₂ : 4 * (d + c) = b)
  (h₃ : 4 * b + 2 * c = a)
  (h₄ : c + 1 = d) :
  a * b * c * d = -319603200 / 10503489 := sorry

end NUMINAMATH_GPT_find_product_l56_5662


namespace NUMINAMATH_GPT_hypotenuse_length_l56_5642

-- Definition of the right triangle with the given leg lengths
structure RightTriangle :=
  (BC AC AB : ℕ)
  (right : BC^2 + AC^2 = AB^2)

-- The theorem we need to prove
theorem hypotenuse_length (T : RightTriangle) (h1 : T.BC = 5) (h2 : T.AC = 12) :
  T.AB = 13 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l56_5642


namespace NUMINAMATH_GPT_reporters_not_covering_politics_l56_5694

def total_reporters : ℝ := 8000
def politics_local : ℝ := 0.12 + 0.08 + 0.08 + 0.07 + 0.06 + 0.05 + 0.04 + 0.03 + 0.02 + 0.01
def politics_non_local : ℝ := 0.15
def politics_total : ℝ := politics_local + politics_non_local

theorem reporters_not_covering_politics :
  1 - politics_total = 0.29 :=
by
  -- Required definition and intermediate proof steps.
  sorry

end NUMINAMATH_GPT_reporters_not_covering_politics_l56_5694


namespace NUMINAMATH_GPT_circumscribed_sphere_eqn_l56_5633

-- Define vertices of the tetrahedron
variables {A_1 A_2 A_3 A_4 : Point}

-- Define barycentric coordinates
variables {x_1 x_2 x_3 x_4 : ℝ}

-- Define edge lengths
variables {a_12 a_13 a_14 a_23 a_24 a_34: ℝ}

-- Define the equation of the circumscribed sphere in barycentric coordinates
theorem circumscribed_sphere_eqn (h1 : A_1 ≠ A_2) (h2 : A_1 ≠ A_3) (h3 : A_1 ≠ A_4)
                                 (h4 : A_2 ≠ A_3) (h5 : A_2 ≠ A_4) (h6 : A_3 ≠ A_4) :
    (x_1 * x_2 * a_12^2 + x_1 * x_3 * a_13^2 + x_1 * x_4 * a_14^2 +
     x_2 * x_3 * a_23^2 + x_2 * x_4 * a_24^2 + x_3 * x_4 * a_34^2) = 0 :=
 sorry

end NUMINAMATH_GPT_circumscribed_sphere_eqn_l56_5633


namespace NUMINAMATH_GPT_largest_sphere_radius_on_torus_l56_5619

theorem largest_sphere_radius_on_torus :
  ∀ r : ℝ, 16 + (r - 1)^2 = (r + 2)^2 → r = 13 / 6 :=
by
  intro r
  intro h
  sorry

end NUMINAMATH_GPT_largest_sphere_radius_on_torus_l56_5619


namespace NUMINAMATH_GPT_construct_inaccessible_angle_bisector_l56_5656

-- Definitions for problem context
structure Point :=
  (x y : ℝ)

structure Line :=
  (p1 p2 : Point)

structure Angle := 
  (vertex : Point)
  (ray1 ray2 : Line)

-- Predicate to determine if a line bisects an angle
def IsAngleBisector (L : Line) (A : Angle) : Prop := sorry

-- The inaccessible vertex angle we are considering
-- Let's assume the vertex is defined but we cannot access it physically in constructions
noncomputable def inaccessible_angle : Angle := sorry

-- Statement to prove: Construct a line that bisects the inaccessible angle
theorem construct_inaccessible_angle_bisector :
  ∃ L : Line, IsAngleBisector L inaccessible_angle :=
sorry

end NUMINAMATH_GPT_construct_inaccessible_angle_bisector_l56_5656


namespace NUMINAMATH_GPT_find_k_l56_5634

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^2 + k * x - 7

-- Define the given condition f(5) - g(5) = 20
def condition (k : ℝ) : Prop := f 5 - g 5 k = 20

-- The theorem to prove that k = 16.4
theorem find_k : ∃ k : ℝ, condition k ∧ k = 16.4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l56_5634


namespace NUMINAMATH_GPT_problem_solution_l56_5667

theorem problem_solution (y : Fin 8 → ℝ)
  (h1 : y 0 + 4 * y 1 + 9 * y 2 + 16 * y 3 + 25 * y 4 + 36 * y 5 + 49 * y 6 + 64 * y 7 = 2)
  (h2 : 4 * y 0 + 9 * y 1 + 16 * y 2 + 25 * y 3 + 36 * y 4 + 49 * y 5 + 64 * y 6 + 81 * y 7 = 15)
  (h3 : 9 * y 0 + 16 * y 1 + 25 * y 2 + 36 * y 3 + 49 * y 4 + 64 * y 5 + 81 * y 6 + 100 * y 7 = 156)
  (h4 : 16 * y 0 + 25 * y 1 + 36 * y 2 + 49 * y 3 + 64 * y 4 + 81 * y 5 + 100 * y 6 + 121 * y 7 = 1305) :
  25 * y 0 + 36 * y 1 + 49 * y 2 + 64 * y 3 + 81 * y 4 + 100 * y 5 + 121 * y 6 + 144 * y 7 = 4360 :=
sorry

end NUMINAMATH_GPT_problem_solution_l56_5667


namespace NUMINAMATH_GPT_no_real_solution_l56_5632

theorem no_real_solution (x : ℝ) : x + 64 / (x + 3) ≠ -13 :=
by {
  -- Proof is not required, so we mark it as sorry.
  sorry
}

end NUMINAMATH_GPT_no_real_solution_l56_5632


namespace NUMINAMATH_GPT_expression_change_l56_5609

theorem expression_change (a b c : ℝ) : 
  a - (2 * b - 3 * c) = a + (-2 * b + 3 * c) := 
by sorry

end NUMINAMATH_GPT_expression_change_l56_5609


namespace NUMINAMATH_GPT_inequality_proof_l56_5648

theorem inequality_proof (x y z : ℝ) : 
    x^4 + y^4 + z^2 + 1 ≥ 2 * x * (x * y^2 - x + z + 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l56_5648


namespace NUMINAMATH_GPT_arithmetic_sequence_x_values_l56_5683

theorem arithmetic_sequence_x_values {x : ℝ} (h_nonzero : x ≠ 0) (h_arith_seq : ∃ (k : ℤ), x - k = 1/2 ∧ x + 1 - (k + 1) = (k + 1) - 1/2) (h_lt_four : x < 4) :
  x = 0.5 ∨ x = 1.5 ∨ x = 2.5 ∨ x = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_x_values_l56_5683


namespace NUMINAMATH_GPT_money_left_in_wallet_l56_5670

def olivia_initial_money : ℕ := 54
def olivia_spent_money : ℕ := 25

theorem money_left_in_wallet : olivia_initial_money - olivia_spent_money = 29 :=
by
  sorry

end NUMINAMATH_GPT_money_left_in_wallet_l56_5670


namespace NUMINAMATH_GPT_quadratic_roots_in_range_l56_5665

theorem quadratic_roots_in_range (a : ℝ) (α β : ℝ)
  (h_eq : ∀ x : ℝ, x^2 + (a^2 + 1) * x + a - 2 = 0)
  (h_root1 : α > 1)
  (h_root2 : β < -1)
  (h_viete_sum : α + β = -(a^2 + 1))
  (h_viete_prod : α * β = a - 2) :
  0 < a ∧ a < 2 :=
  sorry

end NUMINAMATH_GPT_quadratic_roots_in_range_l56_5665


namespace NUMINAMATH_GPT_exists_unique_adjacent_sums_in_circle_l56_5682

theorem exists_unique_adjacent_sums_in_circle :
  ∃ (f : Fin 10 → Fin 11),
    (∀ (i j : Fin 10), i ≠ j → (f i + f (i + 1)) % 11 ≠ (f j + f (j + 1)) % 11) :=
sorry

end NUMINAMATH_GPT_exists_unique_adjacent_sums_in_circle_l56_5682


namespace NUMINAMATH_GPT_vertex_angle_is_130_8_l56_5625

-- Define the given conditions
variables {a b h : ℝ}

def is_isosceles_triangle (a b h : ℝ) : Prop :=
  a^2 = b * 3 * h ∧ b = 2 * h

-- Define the obtuse condition on the vertex angle
def vertex_angle_obtuse (a b h : ℝ) : Prop :=
  ∃ θ : ℝ, 120 < θ ∧ θ < 180 ∧ θ = (130.8 : ℝ)

-- The formal proof statement using Lean 4
theorem vertex_angle_is_130_8 (a b h : ℝ) 
  (h1 : is_isosceles_triangle a b h)
  (h2 : vertex_angle_obtuse a b h) : 
  ∃ (φ : ℝ), φ = 130.8 :=
sorry

end NUMINAMATH_GPT_vertex_angle_is_130_8_l56_5625


namespace NUMINAMATH_GPT_lizzy_wealth_after_loan_l56_5661

theorem lizzy_wealth_after_loan 
  (initial_wealth : ℕ)
  (loan : ℕ)
  (interest_rate : ℕ)
  (h1 : initial_wealth = 30)
  (h2 : loan = 15)
  (h3 : interest_rate = 20)
  : (initial_wealth - loan) + (loan + loan * interest_rate / 100) = 33 :=
by
  sorry

end NUMINAMATH_GPT_lizzy_wealth_after_loan_l56_5661


namespace NUMINAMATH_GPT_factor_diff_of_squares_l56_5620

-- Define the expression t^2 - 49 and show it is factored as (t - 7)(t + 7)
theorem factor_diff_of_squares (t : ℝ) : t^2 - 49 = (t - 7) * (t + 7) := by
  sorry

end NUMINAMATH_GPT_factor_diff_of_squares_l56_5620


namespace NUMINAMATH_GPT_mode_of_dataset_l56_5676

def dataset : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def frequency (n : ℕ) (l : List ℕ) : ℕ :=
  l.count n

theorem mode_of_dataset :
  (∀ n ≠ 3, frequency n dataset ≤ 3) ∧ frequency 3 dataset = 3 :=
by
  sorry

end NUMINAMATH_GPT_mode_of_dataset_l56_5676


namespace NUMINAMATH_GPT_number_of_bookshelves_l56_5637

-- Definitions based on the conditions
def books_per_shelf : ℕ := 2
def total_books : ℕ := 38

-- Statement to prove
theorem number_of_bookshelves (books_per_shelf total_books : ℕ) : total_books / books_per_shelf = 19 :=
by sorry

end NUMINAMATH_GPT_number_of_bookshelves_l56_5637


namespace NUMINAMATH_GPT_least_positive_integer_condition_l56_5663

theorem least_positive_integer_condition :
  ∃ n > 1, (∀ k ∈ [3, 4, 5, 6, 7, 8, 9, 10], n % k = 1) → n = 25201 := by
  sorry

end NUMINAMATH_GPT_least_positive_integer_condition_l56_5663


namespace NUMINAMATH_GPT_Jessica_cut_40_roses_l56_5664

-- Define the problem's conditions as variables
variables (initialVaseRoses : ℕ) (finalVaseRoses : ℕ) (rosesGivenToSarah : ℕ)

-- Define the number of roses Jessica cut from her garden
def rosesCutFromGarden (initialVaseRoses finalVaseRoses rosesGivenToSarah : ℕ) : ℕ :=
  (finalVaseRoses - initialVaseRoses) + rosesGivenToSarah

-- Problem statement: Prove Jessica cut 40 roses from her garden
theorem Jessica_cut_40_roses (initialVaseRoses finalVaseRoses rosesGivenToSarah : ℕ) :
  initialVaseRoses = 7 →
  finalVaseRoses = 37 →
  rosesGivenToSarah = 10 →
  rosesCutFromGarden initialVaseRoses finalVaseRoses rosesGivenToSarah = 40 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_Jessica_cut_40_roses_l56_5664


namespace NUMINAMATH_GPT_first_offset_length_l56_5654

theorem first_offset_length (diagonal : ℝ) (offset2 : ℝ) (area : ℝ) (h_diagonal : diagonal = 50) (h_offset2 : offset2 = 8) (h_area : area = 450) :
  ∃ offset1 : ℝ, offset1 = 10 :=
by
  sorry

end NUMINAMATH_GPT_first_offset_length_l56_5654


namespace NUMINAMATH_GPT_pi_is_irrational_l56_5638

theorem pi_is_irrational :
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ π = p / q) :=
by
  sorry

end NUMINAMATH_GPT_pi_is_irrational_l56_5638


namespace NUMINAMATH_GPT_find_a_n_l56_5675

variable (a : ℕ → ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a1_eq : a 1 = 1
axiom rec_relation : ∀ n, a n * (n * a n - a (n + 1)) = (n + 1) * (a (n + 1)) ^ 2

theorem find_a_n : ∀ n, a n = 1 / n := by
  sorry

end NUMINAMATH_GPT_find_a_n_l56_5675


namespace NUMINAMATH_GPT_system_of_linear_eq_with_two_variables_l56_5621

-- Definitions of individual equations
def eqA (x : ℝ) : Prop := 3 * x - 2 = 5
def eqB (x : ℝ) : Prop := 6 * x^2 - 2 = 0
def eqC (x y : ℝ) : Prop := 1 / x + y = 3
def eqD (x y : ℝ) : Prop := 5 * x + y = 2

-- The main theorem to prove that D is a system of linear equations with two variables
theorem system_of_linear_eq_with_two_variables :
    (∃ x y : ℝ, eqD x y) ∧ (¬∃ x : ℝ, eqA x) ∧ (¬∃ x : ℝ, eqB x) ∧ (¬∃ x y : ℝ, eqC x y) :=
by
  sorry

end NUMINAMATH_GPT_system_of_linear_eq_with_two_variables_l56_5621


namespace NUMINAMATH_GPT_regular_polygon_sides_l56_5690

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end NUMINAMATH_GPT_regular_polygon_sides_l56_5690


namespace NUMINAMATH_GPT_goods_train_speed_l56_5606

theorem goods_train_speed (Vm : ℝ) (T : ℝ) (L : ℝ) (Vg : ℝ) :
  Vm = 50 → T = 9 → L = 280 →
  Vg = ((L / T) - (Vm * 1000 / 3600)) * 3600 / 1000 →
  Vg = 62 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_goods_train_speed_l56_5606


namespace NUMINAMATH_GPT_part1_case1_part1_case2_part1_case3_part2_l56_5674

def f (m x : ℝ) : ℝ := (m+1)*x^2 - (m-1)*x + (m-1)

theorem part1_case1 (m x : ℝ) (h : m = -1) : 
  f m x ≥ (m+1)*x → x ≥ 1 := sorry

theorem part1_case2 (m x : ℝ) (h : m > -1) :
  f m x ≥ (m+1)*x →
  (x ≤ (m-1)/(m+1) ∨ x ≥ 1) := sorry

theorem part1_case3 (m x : ℝ) (h : m < -1) : 
  f m x ≥ (m+1)*x →
  (1 ≤ x ∧ x ≤ (m-1)/(m+1)) := sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) →
  m ≥ 1 := sorry

end NUMINAMATH_GPT_part1_case1_part1_case2_part1_case3_part2_l56_5674


namespace NUMINAMATH_GPT_acid_percentage_in_original_mixture_l56_5679

theorem acid_percentage_in_original_mixture 
  {a w : ℕ} 
  (h1 : a / (a + w + 1) = 1 / 5) 
  (h2 : (a + 1) / (a + w + 2) = 1 / 3) : 
  a / (a + w) = 1 / 4 :=
sorry

end NUMINAMATH_GPT_acid_percentage_in_original_mixture_l56_5679


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l56_5691

theorem quadratic_inequality_solution (x : ℝ) : 
  ((x - 1) * x ≥ 2) ↔ (x ≤ -1 ∨ x ≥ 2) := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l56_5691


namespace NUMINAMATH_GPT_arithmetic_sequence_l56_5624

theorem arithmetic_sequence {a b : ℤ} :
  (-1 < a ∧ a < b ∧ b < 8) ∧
  (8 - (-1) = 9) ∧
  (a + b = 7) →
  (a = 2 ∧ b = 5) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_l56_5624


namespace NUMINAMATH_GPT_explicit_formula_l56_5626

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem explicit_formula (x1 x2 : ℝ) (h1 : x1 ∈ Set.Icc (-1 : ℝ) 1) (h2 : x2 ∈ Set.Icc (-1 : ℝ) 1) :
  f x = x^3 - 3 * x ∧ |f x1 - f x2| ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_explicit_formula_l56_5626


namespace NUMINAMATH_GPT_average_minutes_per_day_l56_5651

theorem average_minutes_per_day
  (f : ℕ) -- Number of fifth graders
  (third_grade_minutes : ℕ := 10)
  (fourth_grade_minutes : ℕ := 18)
  (fifth_grade_minutes : ℕ := 12)
  (third_grade_students : ℕ := 3 * f)
  (fourth_grade_students : ℕ := (3 / 2) * f) -- Assumed to work with integer or rational numbers
  (fifth_grade_students : ℕ := f)
  (total_minutes_third_grade : ℕ := third_grade_minutes * third_grade_students)
  (total_minutes_fourth_grade : ℕ := fourth_grade_minutes * fourth_grade_students)
  (total_minutes_fifth_grade : ℕ := fifth_grade_minutes * fifth_grade_students)
  (total_minutes : ℕ := total_minutes_third_grade + total_minutes_fourth_grade + total_minutes_fifth_grade)
  (total_students : ℕ := third_grade_students + fourth_grade_students + fifth_grade_students) :
  (total_minutes / total_students : ℝ) = 12.55 :=
by
  sorry

end NUMINAMATH_GPT_average_minutes_per_day_l56_5651


namespace NUMINAMATH_GPT_max_m_value_l56_5652

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 1 / b = 1 / 4) : ∃ m : ℝ, (∀ a b : ℝ,  a > 0 ∧ b > 0 ∧ (2 / a + 1 / b = 1 / 4) → 2 * a + b ≥ 4 * m) ∧ m = 7 / 4 :=
sorry

end NUMINAMATH_GPT_max_m_value_l56_5652


namespace NUMINAMATH_GPT_compare_fractions_l56_5615

theorem compare_fractions {x : ℝ} (h : 3 < x ∧ x < 4) : 
  (2 / 3) > ((5 - x) / 3) :=
by sorry

end NUMINAMATH_GPT_compare_fractions_l56_5615


namespace NUMINAMATH_GPT_min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l56_5640

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end NUMINAMATH_GPT_min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l56_5640


namespace NUMINAMATH_GPT_base_r_5555_square_palindrome_l56_5671

theorem base_r_5555_square_palindrome (r : ℕ) (a b c d : ℕ) 
  (h1 : r % 2 = 0) 
  (h2 : r >= 18) 
  (h3 : d - c = 2)
  (h4 : ∀ x, (x = 5 * r^3 + 5 * r^2 + 5 * r + 5) → 
    (x^2 = a * r^7 + b * r^6 + c * r^5 + d * r^4 + d * r^3 + c * r^2 + b * r + a)) : 
  r = 24 := 
sorry

end NUMINAMATH_GPT_base_r_5555_square_palindrome_l56_5671


namespace NUMINAMATH_GPT_weight_of_5_diamonds_l56_5673

-- Define the weight of one diamond and one jade
variables (D J : ℝ)

-- Conditions:
-- 1. Total weight of 4 diamonds and 2 jades
def condition1 : Prop := 4 * D + 2 * J = 140
-- 2. A jade is 10 g heavier than a diamond
def condition2 : Prop := J = D + 10

-- Total weight of 5 diamonds
def total_weight_of_5_diamonds : ℝ := 5 * D

-- Theorem: Prove that the total weight of 5 diamonds is 100 g
theorem weight_of_5_diamonds (h1 : condition1 D J) (h2 : condition2 D J) : total_weight_of_5_diamonds D = 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_weight_of_5_diamonds_l56_5673


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l56_5655

theorem perpendicular_line_through_point (x y : ℝ) : (x, y) = (0, -3) ∧ (∀ x y : ℝ, 2 * x + 3 * y - 6 = 0) → 3 * x - 2 * y - 6 = 0 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l56_5655


namespace NUMINAMATH_GPT_shaded_area_fraction_l56_5641

/-- The fraction of the larger square's area that is inside the shaded rectangle 
    formed by the points (2,2), (3,2), (3,5), and (2,5) on a 6 by 6 grid 
    is 1/12. -/
theorem shaded_area_fraction : 
  let grid_size := 6
  let rectangle_points := [(2, 2), (3, 2), (3, 5), (2, 5)]
  let rectangle_length := 1
  let rectangle_height := 3
  let rectangle_area := rectangle_length * rectangle_height
  let square_area := grid_size^2
  rectangle_area / square_area = 1 / 12 := 
by 
  sorry

end NUMINAMATH_GPT_shaded_area_fraction_l56_5641


namespace NUMINAMATH_GPT_log_400_cannot_be_computed_l56_5603

theorem log_400_cannot_be_computed :
  let log_8 : ℝ := 0.9031
  let log_9 : ℝ := 0.9542
  let log_7 : ℝ := 0.8451
  (∀ (log_2 log_3 log_5 : ℝ), log_2 = 1 / 3 * log_8 → log_3 = 1 / 2 * log_9 → log_5 = 1 → 
    (∀ (log_val : ℝ), 
      (log_val = log_21 → log_21 = log_3 + log_7 → log_val = (1 / 2) * log_9 + log_7)
      ∧ (log_val = log_9_over_8 → log_9_over_8 = log_9 - log_8)
      ∧ (log_val = log_126 → log_126 = log_2 + log_7 + log_9 → log_val = (1 / 3) * log_8 + log_7 + log_9)
      ∧ (log_val = log_0_875 → log_0_875 = log_7 - log_8)
      ∧ (log_val = log_400 → log_400 = log_8 + 1 + log_5) 
      → False))
:= 
sorry

end NUMINAMATH_GPT_log_400_cannot_be_computed_l56_5603


namespace NUMINAMATH_GPT_max_value_n_for_positive_an_l56_5658

-- Define the arithmetic sequence
noncomputable def arithmetic_seq (a d : ℤ) (n : ℤ) := a + (n - 1) * d

-- Define the sum of first n terms of an arithmetic sequence
noncomputable def sum_arith_seq (a d n : ℤ) := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
axiom S15_pos (a d : ℤ) : sum_arith_seq a d 15 > 0
axiom S16_neg (a d : ℤ) : sum_arith_seq a d 16 < 0

-- Proof problem
theorem max_value_n_for_positive_an (a d : ℤ) :
  ∃ n : ℤ, n = 8 ∧ ∀ m : ℤ, (1 ≤ m ∧ m ≤ 8) → arithmetic_seq a d m > 0 :=
sorry

end NUMINAMATH_GPT_max_value_n_for_positive_an_l56_5658


namespace NUMINAMATH_GPT_power_rule_for_fractions_calculate_fraction_l56_5698

theorem power_rule_for_fractions (a b : ℚ) (n : ℕ) : (a / b)^n = (a^n) / (b^n) := 
by sorry

theorem calculate_fraction (a b n : ℕ) (h : a = 3 ∧ b = 5 ∧ n = 3) : (a / b)^n = 27 / 125 :=
by
  obtain ⟨ha, hb, hn⟩ := h
  simp [ha, hb, hn, power_rule_for_fractions (3 : ℚ) (5 : ℚ) 3]

end NUMINAMATH_GPT_power_rule_for_fractions_calculate_fraction_l56_5698


namespace NUMINAMATH_GPT_length_AB_l56_5678

theorem length_AB 
  (P : ℝ × ℝ) 
  (hP : 3 * P.1 + 4 * P.2 + 8 = 0)
  (C : ℝ × ℝ := (1, 1))
  (A B : ℝ × ℝ)
  (hA : (A.1 - 1)^2 + (A.2 - 1)^2 = 1 ∧ (3 * A.1 + 4 * A.2 + 8 ≠ 0))
  (hB : (B.1 - 1)^2 + (B.2 - 1)^2 = 1 ∧ (3 * B.1 + 4 * B.2 + 8 ≠ 0)) :
  dist A B = 4 * Real.sqrt 2 / 3 := sorry

end NUMINAMATH_GPT_length_AB_l56_5678


namespace NUMINAMATH_GPT_sum_of_squares_l56_5669

theorem sum_of_squares (x y : ℝ) (h₁ : x + y = 16) (h₂ : x * y = 28) : x^2 + y^2 = 200 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l56_5669


namespace NUMINAMATH_GPT_friends_count_l56_5646

def bananas_total : ℝ := 63
def bananas_per_friend : ℝ := 21.0

theorem friends_count : bananas_total / bananas_per_friend = 3 := sorry

end NUMINAMATH_GPT_friends_count_l56_5646


namespace NUMINAMATH_GPT_range_of_x_l56_5645

theorem range_of_x (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -2) :
  x > 1/3 ∨ x < -1/2 :=
sorry

end NUMINAMATH_GPT_range_of_x_l56_5645


namespace NUMINAMATH_GPT_range_of_a_l56_5608

-- Define the inequality condition
def inequality (a x : ℝ) : Prop := (a-2)*x^2 + 2*(a-2)*x < 4

-- The main theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, inequality a x) ↔ (-2 : ℝ) < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l56_5608


namespace NUMINAMATH_GPT_solution_set_inequality_l56_5657

variable (f : ℝ → ℝ)

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, HasDerivAt f (f' x) x

def condition_x_f_prime (f f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x^2 * f' x > 2 * x * f (-x)

-- Main theorem to prove the solution set of inequality
theorem solution_set_inequality (f' : ℝ → ℝ) :
  is_odd_function f →
  derivative f f' →
  condition_x_f_prime f f' →
  ∀ x : ℝ, x^2 * f x < (3 * x - 1)^2 * f (1 - 3 * x) → x < (1 / 4) := 
  by
    intros h_odd h_deriv h_cond x h_ineq
    sorry

end NUMINAMATH_GPT_solution_set_inequality_l56_5657


namespace NUMINAMATH_GPT_ninth_grade_students_eq_l56_5623

-- Let's define the conditions
def total_students : ℕ := 50
def seventh_grade_students (x : ℕ) : ℕ := 2 * x - 1
def eighth_grade_students (x : ℕ) : ℕ := x

-- Define the expression for ninth grade students based on the conditions
def ninth_grade_students (x : ℕ) : ℕ :=
  total_students - (seventh_grade_students x + eighth_grade_students x)

-- The theorem statement to prove
theorem ninth_grade_students_eq (x : ℕ) : ninth_grade_students x = 51 - 3 * x :=
by
  sorry

end NUMINAMATH_GPT_ninth_grade_students_eq_l56_5623


namespace NUMINAMATH_GPT_profit_percentage_l56_5616

theorem profit_percentage (C S : ℝ) (h : 17 * C = 16 * S) : (S - C) / C * 100 = 6.25 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_l56_5616


namespace NUMINAMATH_GPT_tan_double_angle_cos_beta_l56_5631

theorem tan_double_angle (α β : ℝ) (h1 : Real.sin α = 4 * Real.sqrt 3 / 7) 
  (h2 : Real.cos (β - α) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < Real.pi / 2) : 
  Real.tan (2 * α) = -(8 * Real.sqrt 3) / 47 :=
  sorry

theorem cos_beta (α β : ℝ) (h1 : Real.sin α = 4 * Real.sqrt 3 / 7) 
  (h2 : Real.cos (β - α) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < Real.pi / 2) : 
  Real.cos β = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_tan_double_angle_cos_beta_l56_5631


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_solution_l56_5693

theorem arithmetic_geometric_sequence_solution 
  (a1 a2 b1 b2 b3 : ℝ) 
  (h1 : -2 * 2 + a2 = a1)
  (h2 : a1 * 2 - 8 = a2)
  (h3 : b2 ^ 2 = -2 * -8)
  (h4 : b2 = -4) :
  (a2 - a1) / b2 = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_solution_l56_5693


namespace NUMINAMATH_GPT_most_likely_number_of_red_balls_l56_5614

-- Define the conditions
def total_balls : ℕ := 20
def red_ball_frequency : ℝ := 0.8

-- Define the statement we want to prove
theorem most_likely_number_of_red_balls : red_ball_frequency * total_balls = 16 :=
by sorry

end NUMINAMATH_GPT_most_likely_number_of_red_balls_l56_5614


namespace NUMINAMATH_GPT_largest_in_eight_consecutive_integers_l56_5653

theorem largest_in_eight_consecutive_integers (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) = 4304) :
  n + 7 = 544 :=
by
  sorry

end NUMINAMATH_GPT_largest_in_eight_consecutive_integers_l56_5653


namespace NUMINAMATH_GPT_race_results_l56_5659

-- Competitor times in seconds
def time_A : ℕ := 40
def time_B : ℕ := 50
def time_C : ℕ := 55

-- Time difference calculations
def time_diff_AB := time_B - time_A
def time_diff_AC := time_C - time_A
def time_diff_BC := time_C - time_B

theorem race_results :
  time_diff_AB = 10 ∧ time_diff_AC = 15 ∧ time_diff_BC = 5 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_race_results_l56_5659


namespace NUMINAMATH_GPT_max_diagonal_intersections_l56_5695

theorem max_diagonal_intersections (n : ℕ) (h : n ≥ 4) : 
    ∃ k, k = n * (n - 1) * (n - 2) * (n - 3) / 24 :=
by
    sorry

end NUMINAMATH_GPT_max_diagonal_intersections_l56_5695


namespace NUMINAMATH_GPT_scientific_notation_of_819000_l56_5647

theorem scientific_notation_of_819000 : (819000 : ℝ) = 8.19 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_819000_l56_5647


namespace NUMINAMATH_GPT_sector_angle_l56_5601

theorem sector_angle (r l : ℝ) (h₁ : 2 * r + l = 4) (h₂ : 1/2 * l * r = 1) : l / r = 2 :=
by
  sorry

end NUMINAMATH_GPT_sector_angle_l56_5601


namespace NUMINAMATH_GPT_arithmetic_geom_seq_S5_l56_5677

theorem arithmetic_geom_seq_S5 (a_n : ℕ → ℚ) (S_n : ℕ → ℚ)
  (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * (1/2))
  (h_sum : ∀ n, S_n n = n * a_n 1 + (n * (n - 1) / 2) * (1/2))
  (h_geom_seq : (a_n 2) * (a_n 14) = (a_n 6) ^ 2) :
  S_n 5 = 25 / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geom_seq_S5_l56_5677


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l56_5650

-- Definition of the first line l1
def line1 (x y : ℝ) (c1 : ℝ) : Prop := 3 * x + 4 * y + c1 = 0

-- Definition of the second line l2
def line2 (x y : ℝ) (c2 : ℝ) : Prop := 6 * x + 8 * y + c2 = 0

-- The problem statement in Lean:
theorem distance_between_parallel_lines (c1 c2 : ℝ) :
  ∃ d : ℝ, d = |2 * c1 - c2| / 10 :=
sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l56_5650


namespace NUMINAMATH_GPT_range_of_a_l56_5635

variable (a : ℝ)
variable (f : ℝ → ℝ)

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def fWhenNegative (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x = 9 * x + a^2 / x + 7

def fNonNegativeCondition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x ≥ a + 1

-- Theorem to prove
theorem range_of_a (odd_f : isOddFunction f) (f_neg : fWhenNegative f a) 
  (nonneg_cond : fNonNegativeCondition f a) : 
  a ≤ -8 / 7 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l56_5635


namespace NUMINAMATH_GPT_intersection_M_N_eq_M_inter_N_l56_5611

def M : Set ℝ := { x | x^2 - 4 > 0 }
def N : Set ℝ := { x | x < 0 }
def M_inter_N : Set ℝ := { x | x < -2 }

theorem intersection_M_N_eq_M_inter_N : M ∩ N = M_inter_N := 
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_eq_M_inter_N_l56_5611


namespace NUMINAMATH_GPT_solve_sine_equation_l56_5687

theorem solve_sine_equation (x : ℝ) (k : ℤ) (h : |Real.sin x| ≠ 1) :
  (8.477 * ((∑' n, Real.sin x ^ n) / (∑' n, ((-1 : ℝ) * Real.sin x) ^ n)) = 4 / (1 + Real.tan x ^ 2)) 
  ↔ (x = (-1)^k * (Real.pi / 6) + k * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_solve_sine_equation_l56_5687


namespace NUMINAMATH_GPT_negation_proof_converse_proof_l56_5600

-- Define the proposition
def prop_last_digit_zero_or_five (n : ℤ) : Prop := (n % 10 = 0) ∨ (n % 10 = 5)
def divisible_by_five (n : ℤ) : Prop := ∃ k : ℤ, n = 5 * k

-- Negation of the proposition
def negation_prop : Prop :=
  ∃ n : ℤ, prop_last_digit_zero_or_five n ∧ ¬ divisible_by_five n

-- Converse of the proposition
def converse_prop : Prop :=
  ∀ n : ℤ, ¬ prop_last_digit_zero_or_five n → ¬ divisible_by_five n

theorem negation_proof : negation_prop :=
  sorry  -- to be proved

theorem converse_proof : converse_prop :=
  sorry  -- to be proved

end NUMINAMATH_GPT_negation_proof_converse_proof_l56_5600


namespace NUMINAMATH_GPT_plus_minus_pairs_l56_5688

theorem plus_minus_pairs (a b p q : ℕ) (h_plus_pairs : p = a) (h_minus_pairs : q = b) : 
  a - b = p - q := 
by 
  sorry

end NUMINAMATH_GPT_plus_minus_pairs_l56_5688


namespace NUMINAMATH_GPT_fewer_hours_worked_l56_5604

noncomputable def total_earnings_summer := 6000
noncomputable def total_weeks_summer := 10
noncomputable def hours_per_week_summer := 50
noncomputable def total_earnings_school_year := 8000
noncomputable def total_weeks_school_year := 40

noncomputable def hourly_wage := total_earnings_summer / (hours_per_week_summer * total_weeks_summer)
noncomputable def total_hours_school_year := total_earnings_school_year / hourly_wage
noncomputable def hours_per_week_school_year := total_hours_school_year / total_weeks_school_year
noncomputable def fewer_hours_per_week := hours_per_week_summer - hours_per_week_school_year

theorem fewer_hours_worked :
  fewer_hours_per_week = hours_per_week_summer - (total_earnings_school_year / hourly_wage / total_weeks_school_year) := by
  sorry

end NUMINAMATH_GPT_fewer_hours_worked_l56_5604


namespace NUMINAMATH_GPT_multiple_of_1897_l56_5629

theorem multiple_of_1897 (n : ℕ) : ∃ k : ℤ, 2903^n - 803^n - 464^n + 261^n = k * 1897 := by
  sorry

end NUMINAMATH_GPT_multiple_of_1897_l56_5629


namespace NUMINAMATH_GPT_columbia_distinct_arrangements_l56_5643

theorem columbia_distinct_arrangements : 
  let total_letters := 8
  let repeat_I := 2
  let repeat_U := 2
  Nat.factorial total_letters / (Nat.factorial repeat_I * Nat.factorial repeat_U) = 90720 := by
  sorry

end NUMINAMATH_GPT_columbia_distinct_arrangements_l56_5643


namespace NUMINAMATH_GPT_stopped_babysitting_16_years_ago_l56_5639

-- Definitions of given conditions
def started_babysitting_age (Jane_age_start : ℕ) := Jane_age_start = 16
def age_half_constraint (Jane_age child_age : ℕ) := child_age ≤ Jane_age / 2
def current_age (Jane_age_now : ℕ) := Jane_age_now = 32
def oldest_babysat_age_now (child_age_now : ℕ) := child_age_now = 24

-- The proposition to be proved
theorem stopped_babysitting_16_years_ago 
  (Jane_age_start Jane_age_now child_age_now : ℕ)
  (h1 : started_babysitting_age Jane_age_start)
  (h2 : ∀ (Jane_age child_age : ℕ), age_half_constraint Jane_age child_age → Jane_age > Jane_age_start → child_age_now = 24 → Jane_age = 24)
  (h3 : current_age Jane_age_now)
  (h4 : oldest_babysat_age_now child_age_now) :
  Jane_age_now - Jane_age_start = 16 :=
by sorry

end NUMINAMATH_GPT_stopped_babysitting_16_years_ago_l56_5639


namespace NUMINAMATH_GPT_find_distance_between_foci_l56_5672

noncomputable def distance_between_foci (pts : List (ℝ × ℝ)) : ℝ :=
  let c := (1, -1)  -- center of the ellipse
  let x1 := (1, 3)
  let x2 := (1, -5)
  let y := (7, -5)
  let b := 4       -- semi-minor axis length
  let a := 2 * Real.sqrt 13  -- semi-major axis length
  let foci_distance := 2 * Real.sqrt (a^2 - b^2)
  foci_distance

theorem find_distance_between_foci :
  distance_between_foci [(1, 3), (7, -5), (1, -5)] = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_between_foci_l56_5672


namespace NUMINAMATH_GPT_alcohol_percentage_in_original_solution_l56_5610

theorem alcohol_percentage_in_original_solution
  (P : ℚ)
  (alcohol_in_new_mixture : ℚ)
  (original_solution_volume : ℚ)
  (added_water_volume : ℚ)
  (new_mixture_volume : ℚ)
  (percentage_in_new_mixture : ℚ) :
  original_solution_volume = 11 →
  added_water_volume = 3 →
  new_mixture_volume = original_solution_volume + added_water_volume →
  percentage_in_new_mixture = 33 →
  alcohol_in_new_mixture = (percentage_in_new_mixture / 100) * new_mixture_volume →
  (P / 100) * original_solution_volume = alcohol_in_new_mixture →
  P = 42 :=
by
  sorry

end NUMINAMATH_GPT_alcohol_percentage_in_original_solution_l56_5610


namespace NUMINAMATH_GPT_find_f3_l56_5644

theorem find_f3 (a b : ℝ) (f : ℝ → ℝ)
  (h1 : f 1 = 3)
  (h2 : f 2 = 6)
  (h3 : ∀ x, f x = a * x^2 + b * x + 1) :
  f 3 = 10 :=
sorry

end NUMINAMATH_GPT_find_f3_l56_5644


namespace NUMINAMATH_GPT_minimum_perimeter_l56_5628

/-
Given:
1. (a: ℤ), (b: ℤ), (c: ℤ)
2. (a ≠ b)
3. 2 * a + 10 * c = 2 * b + 8 * c
4. 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2
5. 10 * c / 8 * c = 5 / 4

Prove:
The minimum perimeter is 1180 
-/

theorem minimum_perimeter (a b c : ℤ) 
(h1 : a ≠ b)
(h2 : 2 * a + 10 * c = 2 * b + 8 * c)
(h3 : 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2)
(h4 : 10 * c / 8 * c = 5 / 4) :
2 * a + 10 * c = 1180 ∨ 2 * b + 8 * c = 1180 :=
sorry

end NUMINAMATH_GPT_minimum_perimeter_l56_5628


namespace NUMINAMATH_GPT_student_ages_inconsistent_l56_5684

theorem student_ages_inconsistent :
  let total_students := 24
  let avg_age_total := 18
  let group1_students := 6
  let avg_age_group1 := 16
  let group2_students := 10
  let avg_age_group2 := 20
  let group3_students := 7
  let avg_age_group3 := 22
  let total_age_all_students := total_students * avg_age_total
  let total_age_group1 := group1_students * avg_age_group1
  let total_age_group2 := group2_students * avg_age_group2
  let total_age_group3 := group3_students * avg_age_group3
  total_age_all_students < total_age_group1 + total_age_group2 + total_age_group3 :=
by {
  let total_students := 24
  let avg_age_total := 18
  let group1_students := 6
  let avg_age_group1 := 16
  let group2_students := 10
  let avg_age_group2 := 20
  let group3_students := 7
  let avg_age_group3 := 22
  let total_age_all_students := total_students * avg_age_total
  let total_age_group1 := group1_students * avg_age_group1
  let total_age_group2 := group2_students * avg_age_group2
  let total_age_group3 := group3_students * avg_age_group3
  have h₁ : total_age_all_students = 24 * 18 := rfl
  have h₂ : total_age_group1 = 6 * 16 := rfl
  have h₃ : total_age_group2 = 10 * 20 := rfl
  have h₄ : total_age_group3 = 7 * 22 := rfl
  have h₅ : 432 = 24 * 18 := by norm_num
  have h₆ : 96 = 6 * 16 := by norm_num
  have h₇ : 200 = 10 * 20 := by norm_num
  have h₈ : 154 = 7 * 22 := by norm_num
  have h₉ : 432 < 96 + 200 + 154 := by norm_num
  exact h₉
}

end NUMINAMATH_GPT_student_ages_inconsistent_l56_5684


namespace NUMINAMATH_GPT_quadratic_minimum_value_l56_5602

theorem quadratic_minimum_value (p q : ℝ) (h_min_value : ∀ x : ℝ, 2 * x^2 + p * x + q ≥ 10) :
  q = 10 + p^2 / 8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_minimum_value_l56_5602


namespace NUMINAMATH_GPT_trip_duration_is_6_hours_l56_5681

def distance_1 := 55 * 4
def total_distance (A : ℕ) := distance_1 + 70 * A
def total_time (A : ℕ) := 4 + A
def average_speed (A : ℕ) := total_distance A / total_time A

theorem trip_duration_is_6_hours (A : ℕ) (h : 60 = average_speed A) : total_time A = 6 :=
by
  sorry

end NUMINAMATH_GPT_trip_duration_is_6_hours_l56_5681


namespace NUMINAMATH_GPT_age_impossibility_l56_5696

/-
Problem statement:
Ann is 5 years older than Kristine.
Their current ages sum up to 24.
Prove that it's impossible for both their ages to be whole numbers.
-/

theorem age_impossibility 
  (K A : ℕ) -- Kristine's and Ann's ages are natural numbers
  (h1 : A = K + 5) -- Ann is 5 years older than Kristine
  (h2 : K + A = 24) -- their combined age is 24
  : false := sorry

end NUMINAMATH_GPT_age_impossibility_l56_5696


namespace NUMINAMATH_GPT_find_fraction_l56_5649

theorem find_fraction (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 20 * b) / (b + 20 * a) = 3) : a / b = 0.33 :=
sorry

end NUMINAMATH_GPT_find_fraction_l56_5649


namespace NUMINAMATH_GPT_perfect_square_sum_l56_5630

-- Define the numbers based on the given conditions
def A (n : ℕ) : ℕ := 4 * (10^(2 * n) - 1) / 9
def B (n : ℕ) : ℕ := 2 * (10^(n + 1) - 1) / 9
def C (n : ℕ) : ℕ := 8 * (10^n - 1) / 9

-- Define the main theorem to be proved
theorem perfect_square_sum (n : ℕ) : 
  ∃ k, A n + B n + C n + 7 = k * k :=
sorry

end NUMINAMATH_GPT_perfect_square_sum_l56_5630


namespace NUMINAMATH_GPT_value_of_k_l56_5697

theorem value_of_k (k : ℝ) :
  ∃ (k : ℝ), k ≠ 1 ∧ (k-1) * (0 : ℝ)^2 + 6 * (0 : ℝ) + k^2 - 1 = 0 ∧ k = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l56_5697


namespace NUMINAMATH_GPT_part1_part2_l56_5689

noncomputable def a : ℝ := 2 + Real.sqrt 3
noncomputable def b : ℝ := 2 - Real.sqrt 3

theorem part1 : a * b = 1 := 
by 
  unfold a b
  sorry

theorem part2 : a^2 + b^2 - a * b = 13 :=
by 
  unfold a b
  sorry

end NUMINAMATH_GPT_part1_part2_l56_5689


namespace NUMINAMATH_GPT_exists_geometric_weak_arithmetic_l56_5617

theorem exists_geometric_weak_arithmetic (m : ℕ) (hm : 3 ≤ m) :
  ∃ (k : ℕ) (a : ℕ → ℕ), 
    (∀ i, 1 ≤ i → i ≤ m → a i = k^(m - i)*(k + 1)^(i - 1)) ∧
    ((∀ i, 1 ≤ i → i < m → a i < a (i + 1)) ∧ 
    ∃ (x : ℕ → ℕ) (d : ℕ), 
      (x 0 ≤ a 1 ∧ 
      ∀ i, 1 ≤ i → i < m → (x i ≤ a (i + 1) ∧ a (i + 1) < x (i + 1)) ∧ 
      ∀ i, 0 ≤ i → i < m - 1 → x (i + 1) - x i = d)) :=
by
  sorry

end NUMINAMATH_GPT_exists_geometric_weak_arithmetic_l56_5617


namespace NUMINAMATH_GPT_fraction_of_red_knights_magical_l56_5607

variable {knights : ℕ}
variable {red_knights : ℕ}
variable {blue_knights : ℕ}
variable {magical_knights : ℕ}
variable {magical_red_knights : ℕ}
variable {magical_blue_knights : ℕ}

axiom total_knights : knights > 0
axiom red_knights_fraction : red_knights = (3 * knights) / 8
axiom blue_knights_fraction : blue_knights = (5 * knights) / 8
axiom magical_knights_fraction : magical_knights = knights / 4
axiom magical_fraction_relation : 3 * magical_blue_knights = magical_red_knights

theorem fraction_of_red_knights_magical :
  (magical_red_knights : ℚ) / red_knights = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_red_knights_magical_l56_5607
