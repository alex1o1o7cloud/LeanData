import Mathlib

namespace NUMINAMATH_GPT_dilution_problem_l527_52781

/-- Samantha needs to add 7.2 ounces of water to achieve a 25% alcohol concentration
given that she starts with 12 ounces of solution containing 40% alcohol. -/
theorem dilution_problem (x : ℝ) : (12 + x) * 0.25 = 4.8 ↔ x = 7.2 :=
by sorry

end NUMINAMATH_GPT_dilution_problem_l527_52781


namespace NUMINAMATH_GPT_negation_correct_l527_52755

-- Define the original statement as a predicate
def original_statement (x : ℝ) : Prop := x > 1 → x^2 ≤ x

-- Define the negation of the original statement as a predicate
def negated_statement : Prop := ∃ x : ℝ, x > 1 ∧ x^2 > x

-- Define the theorem that the negation of the original statement implies the negated statement
theorem negation_correct :
  ¬ (∀ x : ℝ, original_statement x) ↔ negated_statement := by
  sorry

end NUMINAMATH_GPT_negation_correct_l527_52755


namespace NUMINAMATH_GPT_scooterValue_after_4_years_with_maintenance_l527_52712

noncomputable def scooterDepreciation (initial_value : ℝ) (years : ℕ) : ℝ :=
  initial_value * ((3 : ℝ) / 4) ^ years

theorem scooterValue_after_4_years_with_maintenance (M : ℝ) :
  scooterDepreciation 40000 4 - 4 * M = 12656.25 - 4 * M :=
by
  sorry

end NUMINAMATH_GPT_scooterValue_after_4_years_with_maintenance_l527_52712


namespace NUMINAMATH_GPT_num_candidates_l527_52745

theorem num_candidates (n : ℕ) (h : n * (n - 1) = 30) : n = 6 :=
sorry

end NUMINAMATH_GPT_num_candidates_l527_52745


namespace NUMINAMATH_GPT_num_real_roots_of_eq_l527_52791

theorem num_real_roots_of_eq (x : ℝ) (h : x * |x| - 3 * |x| - 4 = 0) : 
  ∃! x : ℝ, x * |x| - 3 * |x| - 4 = 0 :=
sorry

end NUMINAMATH_GPT_num_real_roots_of_eq_l527_52791


namespace NUMINAMATH_GPT_Harry_Terry_difference_l527_52707

theorem Harry_Terry_difference : 
(12 - (4 * 3)) - (12 - 4 * 3) = -24 := 
by
  sorry

end NUMINAMATH_GPT_Harry_Terry_difference_l527_52707


namespace NUMINAMATH_GPT_problem_statement_l527_52779

variable (a : ℝ)

theorem problem_statement (h : 5 = a + a⁻¹) : a^4 + (a⁻¹)^4 = 527 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l527_52779


namespace NUMINAMATH_GPT_josh_total_money_left_l527_52722

-- Definitions of the conditions
def profit_per_bracelet : ℝ := 1.5 - 1
def total_bracelets : ℕ := 12
def cost_of_cookies : ℝ := 3

-- The proof problem: 
theorem josh_total_money_left : total_bracelets * profit_per_bracelet - cost_of_cookies = 3 :=
by
  sorry

end NUMINAMATH_GPT_josh_total_money_left_l527_52722


namespace NUMINAMATH_GPT_difference_of_numbers_l527_52764

/-- Given two natural numbers a and 10a whose sum is 23,320,
prove that the difference between them is 19,080. -/
theorem difference_of_numbers (a : ℕ) (h : a + 10 * a = 23320) : 10 * a - a = 19080 := by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l527_52764


namespace NUMINAMATH_GPT_diagonals_from_vertex_l527_52706

theorem diagonals_from_vertex (n : ℕ) (h : (n-2) * 180 + 360 = 1800) : (n - 3) = 7 :=
sorry

end NUMINAMATH_GPT_diagonals_from_vertex_l527_52706


namespace NUMINAMATH_GPT_rectangle_diagonal_l527_52775

theorem rectangle_diagonal (k : ℕ) (h1 : 2 * (5 * k + 4 * k) = 72) : 
  (Real.sqrt ((5 * k) ^ 2 + (4 * k) ^ 2)) = Real.sqrt 656 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_l527_52775


namespace NUMINAMATH_GPT_cos_C_correct_l527_52782

noncomputable def cos_C (B : ℝ) (AD BD : ℝ) : ℝ :=
  let sinB := Real.sin B
  let angleBAC := (2 : ℝ) * Real.arcsin ((Real.sqrt 3 / 3) * (sinB / 2)) -- derived from bisector property.
  let cosA := (2 : ℝ) * Real.cos angleBAC / 2 - 1
  let sinA := 2 * Real.sin angleBAC / 2 * Real.cos angleBAC / 2
  let cos2thirds := -1 / 2
  let sin2thirds := Real.sqrt 3 / 2
  cos2thirds * cosA + sin2thirds * sinA

theorem cos_C_correct : 
  ∀ (π : ℝ), 
  ∀ (A B C : ℝ),
  B = π / 3 →
  ∀ (AD : ℝ), AD = 3 →
  ∀ (BD : ℝ), BD = 2 →
  cos_C B AD BD = (2 * Real.sqrt 6 - 1) / 6 :=
by
  intros π A B C hB angleBisectorI hAD hBD
  sorry

end NUMINAMATH_GPT_cos_C_correct_l527_52782


namespace NUMINAMATH_GPT_tram_speed_l527_52741

theorem tram_speed
  (L v : ℝ)
  (h1 : L = 2 * v)
  (h2 : 96 + L = 10 * v) :
  v = 12 := 
by sorry

end NUMINAMATH_GPT_tram_speed_l527_52741


namespace NUMINAMATH_GPT_Chris_age_l527_52728

variable (a b c : ℕ)

theorem Chris_age : a + b + c = 36 ∧ b = 2*c + 9 ∧ b = a → c = 4 :=
by
  sorry

end NUMINAMATH_GPT_Chris_age_l527_52728


namespace NUMINAMATH_GPT_minimum_A_l527_52794

noncomputable def minA : ℝ := (1 + Real.sqrt 2) / 2

theorem minimum_A (x y z w : ℝ) (A : ℝ) 
    (h : xy + 2 * yz + zw ≤ A * (x^2 + y^2 + z^2 + w^2)) :
    A ≥ minA := 
sorry

end NUMINAMATH_GPT_minimum_A_l527_52794


namespace NUMINAMATH_GPT_problem_statement_l527_52793

theorem problem_statement (p q : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) (r s : ℕ)
  (consecutive_primes : Nat.Prime r ∧ Nat.Prime s ∧ (r + 1 = s ∨ s + 1 = r))
  (roots_condition : r + s = p ∧ r * s = 2 * q) :
  (r * s = 2 * q) ∧ (Nat.Prime (p^2 - 2 * q)) ∧ (Nat.Prime (p + 2 * q)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l527_52793


namespace NUMINAMATH_GPT_only_triple_l527_52785

theorem only_triple (a b c : ℕ) (h1 : (a * b + 1) % c = 0)
                                (h2 : (a * c + 1) % b = 0)
                                (h3 : (b * c + 1) % a = 0) :
    (a = 1 ∧ b = 1 ∧ c = 1) :=
by
  sorry

end NUMINAMATH_GPT_only_triple_l527_52785


namespace NUMINAMATH_GPT_integer_solution_x_l527_52726

theorem integer_solution_x (x : ℤ) (h₁ : x + 8 > 10) (h₂ : -3 * x < -9) : x ≥ 4 ↔ x > 3 := by
  sorry

end NUMINAMATH_GPT_integer_solution_x_l527_52726


namespace NUMINAMATH_GPT_no_such_fractions_l527_52738

open Nat

theorem no_such_fractions : ¬ ∃ (x y : ℕ), (x.gcd y = 1) ∧ (x > 0) ∧ (y > 0) ∧ ((x + 1) * 5 * y = ((y + 1) * 6 * x)) :=
by
  sorry

end NUMINAMATH_GPT_no_such_fractions_l527_52738


namespace NUMINAMATH_GPT_juice_cost_l527_52777

-- Given conditions
def sandwich_cost : ℝ := 0.30
def total_money : ℝ := 2.50
def num_friends : ℕ := 4

-- Cost calculation
def total_sandwich_cost : ℝ := num_friends * sandwich_cost
def remaining_money : ℝ := total_money - total_sandwich_cost

-- The theorem to prove
theorem juice_cost : (remaining_money / num_friends) = 0.325 := by
  sorry

end NUMINAMATH_GPT_juice_cost_l527_52777


namespace NUMINAMATH_GPT_same_parity_iff_exists_c_d_l527_52792

theorem same_parity_iff_exists_c_d (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (a % 2 = b % 2) ↔ ∃ (c d : ℕ), 0 < c ∧ 0 < d ∧ a^2 + b^2 + c^2 + 1 = d^2 := 
by 
  sorry

end NUMINAMATH_GPT_same_parity_iff_exists_c_d_l527_52792


namespace NUMINAMATH_GPT_train_length_is_150_l527_52751

-- Let length_of_train be the length of the train in meters
def length_of_train (speed_kmh : ℕ) (time_s : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600) * time_s

theorem train_length_is_150 (speed_kmh time_s : ℕ) (h_speed : speed_kmh = 180) (h_time : time_s = 3) :
  length_of_train speed_kmh time_s = 150 := by
  sorry

end NUMINAMATH_GPT_train_length_is_150_l527_52751


namespace NUMINAMATH_GPT_problem_l527_52776

def seq (a : ℕ → ℝ) := a 0 = 1 / 2 ∧ ∀ n > 0, a n = a (n - 1) + (1 / n^2) * (a (n - 1))^2

theorem problem (a : ℕ → ℝ) (n : ℕ) (h_seq : seq a) (h_n_pos : n > 0) :
  (1 / a (n - 1) - 1 / a n < 1 / n^2) ∧
  (∀ n > 0, a n < n) ∧
  (∀ n > 0, 1 / a n < 5 / 6 + 1 / (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_problem_l527_52776


namespace NUMINAMATH_GPT_soccer_team_students_l527_52735

theorem soccer_team_students :
  ∀ (n p b m : ℕ),
    n = 25 →
    p = 10 →
    b = 6 →
    n - (p - b) = m →
    m = 21 :=
by
  intros n p b m h_n h_p h_b h_trivial
  sorry

end NUMINAMATH_GPT_soccer_team_students_l527_52735


namespace NUMINAMATH_GPT_average_speed_of_trip_l527_52733

theorem average_speed_of_trip 
  (total_distance : ℝ)
  (first_leg_distance : ℝ)
  (first_leg_speed : ℝ)
  (second_leg_distance : ℝ)
  (second_leg_speed : ℝ)
  (h_dist : total_distance = 50)
  (h_first_leg : first_leg_distance = 25)
  (h_second_leg : second_leg_distance = 25)
  (h_first_speed : first_leg_speed = 60)
  (h_second_speed : second_leg_speed = 30) :
  (total_distance / 
   ((first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed)) = 40) :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_trip_l527_52733


namespace NUMINAMATH_GPT_area_of_PQRSUV_proof_l527_52732

noncomputable def PQRSW_area (PQ QR RS SW : ℝ) : ℝ :=
  (1 / 2) * PQ * QR + (1 / 2) * (RS + SW) * 5

noncomputable def WUV_area (WU UV : ℝ) : ℝ :=
  WU * UV

theorem area_of_PQRSUV_proof 
  (PQ QR RS SW WU UV : ℝ)
  (hPQ : PQ = 8) (hQR : QR = 5) (hRS : RS = 7) (hSW : SW = 10)
  (hWU : WU = 6) (hUV : UV = 7) :
  PQRSW_area PQ QR RS SW + WUV_area WU UV = 147 :=
by
  simp only [PQRSW_area, WUV_area, hPQ, hQR, hRS, hSW, hWU, hUV]
  norm_num
  sorry

end NUMINAMATH_GPT_area_of_PQRSUV_proof_l527_52732


namespace NUMINAMATH_GPT_expand_polynomial_l527_52756

variable (x : ℝ)

theorem expand_polynomial :
  2 * (5 * x^2 - 3 * x + 4 - x^3) = -2 * x^3 + 10 * x^2 - 6 * x + 8 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l527_52756


namespace NUMINAMATH_GPT_fraction_evaluation_l527_52724

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem fraction_evaluation :
  (sqrt 2 * (sqrt 3 - sqrt 7)) / (2 * sqrt (3 + sqrt 5)) =
  (30 - 10 * sqrt 5 - 6 * sqrt 21 + 2 * sqrt 105) / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l527_52724


namespace NUMINAMATH_GPT_capacity_of_each_bag_is_approximately_63_l527_52790

noncomputable def capacity_of_bag (total_sand : ℤ) (num_bags : ℤ) : ℤ :=
  Int.ceil (total_sand / num_bags)

theorem capacity_of_each_bag_is_approximately_63 :
  capacity_of_bag 757 12 = 63 :=
by
  sorry

end NUMINAMATH_GPT_capacity_of_each_bag_is_approximately_63_l527_52790


namespace NUMINAMATH_GPT_pizza_area_increase_l527_52747

theorem pizza_area_increase (A1 A2 r1 r2 : ℝ) (r1_eq : r1 = 7) (r2_eq : r2 = 5) (A1_eq : A1 = Real.pi * r1^2) (A2_eq : A2 = Real.pi * r2^2) :
  ((A1 - A2) / A2) * 100 = 96 := by
  sorry

end NUMINAMATH_GPT_pizza_area_increase_l527_52747


namespace NUMINAMATH_GPT_range_of_m_l527_52770

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x

theorem range_of_m (m : ℝ) : f m > 1 → m < 0 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l527_52770


namespace NUMINAMATH_GPT_odd_square_minus_one_div_by_eight_l527_52725

theorem odd_square_minus_one_div_by_eight (n : ℤ) : ∃ k : ℤ, (2 * n + 1) ^ 2 - 1 = 8 * k :=
by
  sorry

end NUMINAMATH_GPT_odd_square_minus_one_div_by_eight_l527_52725


namespace NUMINAMATH_GPT_present_age_of_B_l527_52796

theorem present_age_of_B :
  ∃ (A B : ℕ), (A + 20 = 2 * (B - 20)) ∧ (A = B + 10) ∧ (B = 70) :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_B_l527_52796


namespace NUMINAMATH_GPT_find_k_l527_52773

theorem find_k (k : ℝ) : 
  (k - 10) / (-8) = (5 - k) / (-8) → k = 7.5 :=
by
  intro h
  let slope1 := (k - 10) / (-8)
  let slope2 := (5 - k) / (-8)
  have h_eq : slope1 = slope2 := h
  sorry

end NUMINAMATH_GPT_find_k_l527_52773


namespace NUMINAMATH_GPT_min_value_exprB_four_min_value_exprC_four_l527_52788

noncomputable def exprB (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def exprC (x : ℝ) : ℝ := 1 / (Real.sin x)^2 + 1 / (Real.cos x)^2

theorem min_value_exprB_four : ∃ x : ℝ, exprB x = 4 := sorry

theorem min_value_exprC_four : ∃ x : ℝ, exprC x = 4 := sorry

end NUMINAMATH_GPT_min_value_exprB_four_min_value_exprC_four_l527_52788


namespace NUMINAMATH_GPT_centroid_tetrahedron_l527_52787

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D M : V)

def is_centroid (M A B C D : V) : Prop :=
  M = (1/4:ℝ) • (A + B + C + D)

theorem centroid_tetrahedron (h : is_centroid M A B C D) :
  (M - A) + (M - B) + (M - C) + (M - D) = (0 : V) :=
by {
  sorry
}

end NUMINAMATH_GPT_centroid_tetrahedron_l527_52787


namespace NUMINAMATH_GPT_no_real_solutions_l527_52723

theorem no_real_solutions (x : ℝ) : ¬ (3 * x^2 + 5 = |4 * x + 2| - 3) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l527_52723


namespace NUMINAMATH_GPT_cost_price_is_700_l527_52780

noncomputable def cost_price_was_700 : Prop :=
  ∃ (CP : ℝ),
    (∀ (SP1 SP2 : ℝ),
      SP1 = CP * 0.84 ∧
        SP2 = CP * 1.04 ∧
        SP2 = SP1 + 140) ∧
    CP = 700

theorem cost_price_is_700 : cost_price_was_700 :=
  sorry

end NUMINAMATH_GPT_cost_price_is_700_l527_52780


namespace NUMINAMATH_GPT_xy_value_l527_52748

theorem xy_value (x y : ℝ) (h₁ : x + y = 2) (h₂ : x^2 * y^3 + y^2 * x^3 = 32) :
  x * y = 2^(5/3) :=
by
  sorry

end NUMINAMATH_GPT_xy_value_l527_52748


namespace NUMINAMATH_GPT_focus_parabola_y_eq_neg4x2_plus_4x_minus_1_l527_52701

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  let p := b^2 / (4 * a) - c / (4 * a)
  (p, 1 / (4 * a))

theorem focus_parabola_y_eq_neg4x2_plus_4x_minus_1 :
  focus_of_parabola (-4) 4 (-1) = (1 / 2, -1 / 8) :=
sorry

end NUMINAMATH_GPT_focus_parabola_y_eq_neg4x2_plus_4x_minus_1_l527_52701


namespace NUMINAMATH_GPT_seashells_broken_l527_52739

theorem seashells_broken (total_seashells : ℕ) (unbroken_seashells : ℕ) (broken_seashells : ℕ) : 
  total_seashells = 6 → unbroken_seashells = 2 → broken_seashells = total_seashells - unbroken_seashells → broken_seashells = 4 :=
by
  intros ht hu hb
  rw [ht, hu] at hb
  exact hb

end NUMINAMATH_GPT_seashells_broken_l527_52739


namespace NUMINAMATH_GPT_minimum_shirts_for_saving_money_l527_52730

-- Define the costs for Acme and Gamma
def acme_cost (x : ℕ) : ℕ := 60 + 10 * x
def gamma_cost (x : ℕ) : ℕ := 15 * x

-- Prove that the minimum number of shirts x for which a customer saves money by using Acme is 13
theorem minimum_shirts_for_saving_money : ∃ (x : ℕ), 60 + 10 * x < 15 * x ∧ x = 13 := by
  sorry

end NUMINAMATH_GPT_minimum_shirts_for_saving_money_l527_52730


namespace NUMINAMATH_GPT_totalNumberOfPupils_l527_52716

-- Definitions of the conditions
def numberOfGirls : Nat := 232
def numberOfBoys : Nat := 253

-- Statement of the problem
theorem totalNumberOfPupils : numberOfGirls + numberOfBoys = 485 := by
  sorry

end NUMINAMATH_GPT_totalNumberOfPupils_l527_52716


namespace NUMINAMATH_GPT_intersection_lines_l527_52789

theorem intersection_lines (c d : ℝ) (h1 : 6 = 2 * 4 + c) (h2 : 6 = 5 * 4 + d) : c + d = -16 := 
by
  sorry

end NUMINAMATH_GPT_intersection_lines_l527_52789


namespace NUMINAMATH_GPT_smallest_n_between_76_and_100_l527_52767

theorem smallest_n_between_76_and_100 :
  ∃ (n : ℕ), (n > 1) ∧ (n % 3 = 2) ∧ (n % 7 = 2) ∧ (n % 5 = 1) ∧ (76 < n) ∧ (n < 100) :=
sorry

end NUMINAMATH_GPT_smallest_n_between_76_and_100_l527_52767


namespace NUMINAMATH_GPT_find_f_neg_a_l527_52709

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.exp x + Real.exp (-x)) + 2

theorem find_f_neg_a (a : ℝ) (h : f a = 4) : f (-a) = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_a_l527_52709


namespace NUMINAMATH_GPT_nathan_tokens_l527_52703

theorem nathan_tokens
  (hockey_games : Nat := 5)
  (hockey_cost : Nat := 4)
  (basketball_games : Nat := 7)
  (basketball_cost : Nat := 5)
  (skee_ball_games : Nat := 3)
  (skee_ball_cost : Nat := 3)
  : hockey_games * hockey_cost + basketball_games * basketball_cost + skee_ball_games * skee_ball_cost = 64 := 
by
  sorry

end NUMINAMATH_GPT_nathan_tokens_l527_52703


namespace NUMINAMATH_GPT_triangle_area_qin_jiushao_l527_52778

theorem triangle_area_qin_jiushao (a b c : ℝ) (h1: a = 2) (h2: b = 3) (h3: c = Real.sqrt 13) :
  Real.sqrt ((1 / 4) * (a^2 * b^2 - (1 / 4) * (a^2 + b^2 - c^2)^2)) = 3 :=
by
  -- Hypotheses
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_triangle_area_qin_jiushao_l527_52778


namespace NUMINAMATH_GPT_range_of_m_l527_52740

def f (m x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
def g (m x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ 0 < m ∧ m < 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l527_52740


namespace NUMINAMATH_GPT_three_zeros_implies_a_lt_neg3_l527_52763

noncomputable def f (a x : ℝ) := x^3 + a * x + 2

theorem three_zeros_implies_a_lt_neg3 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  a < -3 :=
by
  sorry

end NUMINAMATH_GPT_three_zeros_implies_a_lt_neg3_l527_52763


namespace NUMINAMATH_GPT_math_problem_l527_52759

theorem math_problem (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^2 * b - 2 * a * b + a * b^2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l527_52759


namespace NUMINAMATH_GPT_greatest_diff_l527_52798

theorem greatest_diff (x y : ℤ) (hx1 : 6 < x) (hx2 : x < 10) (hy1 : 10 < y) (hy2 : y < 17) : y - x = 7 :=
sorry

end NUMINAMATH_GPT_greatest_diff_l527_52798


namespace NUMINAMATH_GPT_find_m_if_root_zero_l527_52736

theorem find_m_if_root_zero (m : ℝ) :
  (∀ x : ℝ, (m - 1) * x^2 + x + (m^2 - 1) = 0) → m = -1 :=
by
  intro h
  -- Term after this point not necessary, hence a placeholder
  sorry

end NUMINAMATH_GPT_find_m_if_root_zero_l527_52736


namespace NUMINAMATH_GPT_find_number_mul_l527_52766

theorem find_number_mul (n : ℕ) (h : n * 9999 = 724777430) : n = 72483 :=
by
  sorry

end NUMINAMATH_GPT_find_number_mul_l527_52766


namespace NUMINAMATH_GPT_Jamal_crayon_cost_l527_52713

/-- Jamal bought 4 half dozen colored crayons at $2 per crayon. 
    He got a 10% discount on the total cost, and an additional 5% discount on the remaining amount. 
    After paying in US Dollars (USD), we want to know how much he spent in Euros (EUR) and British Pounds (GBP) 
    given that 1 USD is equal to 0.85 EUR and 1 USD is equal to 0.75 GBP. 
    This statement proves that the total cost was 34.884 EUR and 30.78 GBP. -/
theorem Jamal_crayon_cost :
  let number_of_crayons := 4 * 6
  let initial_cost := number_of_crayons * 2
  let first_discount := 0.10 * initial_cost
  let cost_after_first_discount := initial_cost - first_discount
  let second_discount := 0.05 * cost_after_first_discount
  let final_cost_usd := cost_after_first_discount - second_discount
  let final_cost_eur := final_cost_usd * 0.85
  let final_cost_gbp := final_cost_usd * 0.75
  final_cost_eur = 34.884 ∧ final_cost_gbp = 30.78 := 
by
  sorry

end NUMINAMATH_GPT_Jamal_crayon_cost_l527_52713


namespace NUMINAMATH_GPT_coloring_satisfies_conditions_l527_52746

-- Define lattice points as points with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the color function
def color (p : LatticePoint) : ℕ :=
  if (p.x % 2 = 0) ∧ (p.y % 2 = 1) then 0 -- Black
  else if (p.x % 2 = 1) ∧ (p.y % 2 = 0) then 1 -- White
  else 2 -- Red

-- Define condition (1)
def infinite_lines_with_color (c : ℕ) : Prop :=
  ∀ k : ℤ, ∃ p : LatticePoint, color p = c ∧ p.x = k

-- Define condition (2)
def parallelogram_exists (A B C : LatticePoint) (wc rc bc : ℕ) : Prop :=
  (color A = wc) ∧ (color B = rc) ∧ (color C = bc) →
  ∃ D : LatticePoint, color D = rc ∧ D.x = C.x + (A.x - B.x) ∧ D.y = C.y + (A.y - B.y)

-- Main theorem
theorem coloring_satisfies_conditions :
  (∀ c : ℕ, ∃ p : LatticePoint, infinite_lines_with_color c) ∧
  (∀ A B C : LatticePoint, ∃ wc rc bc : ℕ, parallelogram_exists A B C wc rc bc) :=
sorry

end NUMINAMATH_GPT_coloring_satisfies_conditions_l527_52746


namespace NUMINAMATH_GPT_rectangle_area_inscribed_circle_l527_52743

theorem rectangle_area_inscribed_circle 
  (radius : ℝ) (width len : ℝ) 
  (h_radius : radius = 5) 
  (h_width : width = 2 * radius) 
  (h_len_ratio : len = 3 * width) 
  : width * len = 300 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_inscribed_circle_l527_52743


namespace NUMINAMATH_GPT_total_money_l527_52772

theorem total_money (John Alice Bob : ℝ) (hJohn : John = 5 / 8) (hAlice : Alice = 7 / 20) (hBob : Bob = 1 / 4) :
  John + Alice + Bob = 1.225 := 
by 
  sorry

end NUMINAMATH_GPT_total_money_l527_52772


namespace NUMINAMATH_GPT_forty_percent_of_number_is_240_l527_52774

-- Define the conditions as assumptions in Lean
variable (N : ℝ)
variable (h1 : (1/4) * (1/3) * (2/5) * N = 20)

-- Prove that 40% of the number N is 240
theorem forty_percent_of_number_is_240 (h1: (1/4) * (1/3) * (2/5) * N = 20) : 0.40 * N = 240 :=
  sorry

end NUMINAMATH_GPT_forty_percent_of_number_is_240_l527_52774


namespace NUMINAMATH_GPT_nth_term_series_l527_52786

def a_n (n : ℕ) : ℝ :=
  if n % 2 = 1 then -4 else 7

theorem nth_term_series (n : ℕ) : a_n n = 1.5 + 5.5 * (-1) ^ n :=
by
  sorry

end NUMINAMATH_GPT_nth_term_series_l527_52786


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l527_52797

theorem quadratic_distinct_real_roots (k : ℝ) :
  (k > -2 ∧ k ≠ 0) ↔ ( ∃ (a b c : ℝ), a = k ∧ b = -4 ∧ c = -2 ∧ (b^2 - 4 * a * c) > 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l527_52797


namespace NUMINAMATH_GPT_bacteria_population_at_2_15_l527_52708

noncomputable def bacteria_at_time (initial_pop : ℕ) (start_time end_time : ℕ) (interval : ℕ) : ℕ :=
  initial_pop * 2 ^ ((end_time - start_time) / interval)

theorem bacteria_population_at_2_15 :
  let initial_pop := 50
  let start_time := 0  -- 2:00 p.m.
  let end_time := 15   -- 2:15 p.m.
  let interval := 4
  bacteria_at_time initial_pop start_time end_time interval = 400 := sorry

end NUMINAMATH_GPT_bacteria_population_at_2_15_l527_52708


namespace NUMINAMATH_GPT_multiplication_of_fractions_l527_52749

theorem multiplication_of_fractions :
  (77 / 4) * (5 / 2) = 48 + 1 / 8 := 
sorry

end NUMINAMATH_GPT_multiplication_of_fractions_l527_52749


namespace NUMINAMATH_GPT_will_net_calorie_intake_is_600_l527_52765

-- Given conditions translated into Lean definitions and assumptions
def breakfast_calories : ℕ := 900
def jogging_time_minutes : ℕ := 30
def calories_burned_per_minute : ℕ := 10

-- Proof statement in Lean
theorem will_net_calorie_intake_is_600 :
  breakfast_calories - (jogging_time_minutes * calories_burned_per_minute) = 600 :=
by
  sorry

end NUMINAMATH_GPT_will_net_calorie_intake_is_600_l527_52765


namespace NUMINAMATH_GPT_nina_total_spending_l527_52737

-- Defining the quantities and prices of each category of items
def num_toys : Nat := 3
def price_per_toy : Nat := 10

def num_basketball_cards : Nat := 2
def price_per_card : Nat := 5

def num_shirts : Nat := 5
def price_per_shirt : Nat := 6

-- Calculating the total cost for each category
def cost_toys : Nat := num_toys * price_per_toy
def cost_cards : Nat := num_basketball_cards * price_per_card
def cost_shirts : Nat := num_shirts * price_per_shirt

-- Calculating the total amount spent
def total_cost : Nat := cost_toys + cost_cards + cost_shirts

-- The final theorem statement to verify the answer
theorem nina_total_spending : total_cost = 70 :=
by
  sorry

end NUMINAMATH_GPT_nina_total_spending_l527_52737


namespace NUMINAMATH_GPT_arithmetic_sequence_first_term_l527_52760

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_terms_int : ∀ n, ∃ k : ℤ, a n = k) 
  (ha20 : a 20 = 205) : a 1 = 91 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_first_term_l527_52760


namespace NUMINAMATH_GPT_ellipse_condition_l527_52719

theorem ellipse_condition (m : ℝ) :
  (1 < m ∧ m < 3) → ((m > 1 ∧ m < 3 ∧ m ≠ 2) ∨ (m = 2)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_condition_l527_52719


namespace NUMINAMATH_GPT_area_increase_correct_l527_52753

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end NUMINAMATH_GPT_area_increase_correct_l527_52753


namespace NUMINAMATH_GPT_jerry_birthday_games_l527_52752

def jerry_original_games : ℕ := 7
def jerry_total_games_after_birthday : ℕ := 9
def games_jerry_got_for_birthday (original total : ℕ) : ℕ := total - original

theorem jerry_birthday_games :
  games_jerry_got_for_birthday jerry_original_games jerry_total_games_after_birthday = 2 := by
  sorry

end NUMINAMATH_GPT_jerry_birthday_games_l527_52752


namespace NUMINAMATH_GPT_subtraction_of_decimals_l527_52717

theorem subtraction_of_decimals :
  888.8888 - 444.4444 = 444.4444 := 
sorry

end NUMINAMATH_GPT_subtraction_of_decimals_l527_52717


namespace NUMINAMATH_GPT_quotient_of_f_div_g_l527_52714

-- Define the polynomial f(x) = x^5 + 5
def f (x : ℝ) : ℝ := x ^ 5 + 5

-- Define the divisor polynomial g(x) = x - 1
def g (x : ℝ) : ℝ := x - 1

-- Define the expected quotient polynomial q(x) = x^4 + x^3 + x^2 + x + 1
def q (x : ℝ) : ℝ := x ^ 4 + x ^ 3 + x ^ 2 + x + 1

-- State and prove the main theorem
theorem quotient_of_f_div_g (x : ℝ) :
  ∃ r : ℝ, f x = g x * (q x) + r :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_f_div_g_l527_52714


namespace NUMINAMATH_GPT_completing_the_square_transformation_l527_52731

theorem completing_the_square_transformation (x : ℝ) : 
  (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro h
  -- Transformation steps will be shown in the proof
  sorry

end NUMINAMATH_GPT_completing_the_square_transformation_l527_52731


namespace NUMINAMATH_GPT_find_z_l527_52727

noncomputable def solve_for_z (i : ℂ) (z : ℂ) :=
  (2 - i) * z = i ^ 2021

theorem find_z (i z : ℂ) (h1 : solve_for_z i z) : 
  z = -1/5 + 2/5 * i := 
by 
  sorry

end NUMINAMATH_GPT_find_z_l527_52727


namespace NUMINAMATH_GPT_persimmons_count_l527_52700

variables {P T : ℕ}

-- Conditions from the problem
axiom total_eq : P + T = 129
axiom diff_eq : P = T - 43

-- Theorem to prove that there are 43 persimmons
theorem persimmons_count : P = 43 :=
by
  -- Putting the proof placeholder
  sorry

end NUMINAMATH_GPT_persimmons_count_l527_52700


namespace NUMINAMATH_GPT_sum_of_money_l527_52718

theorem sum_of_money (x : ℝ)
  (hC : 0.50 * x = 64)
  (hB : ∀ x, B_shares = 0.75 * x)
  (hD : ∀ x, D_shares = 0.25 * x) :
  let total_sum := x + 0.75 * x + 0.50 * x + 0.25 * x
  total_sum = 320 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_money_l527_52718


namespace NUMINAMATH_GPT_function_monotonicity_l527_52768

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  (a^x) / (b^x + c^x) + (b^x) / (a^x + c^x) + (c^x) / (a^x + b^x)

theorem function_monotonicity (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f a b c x ≤ f a b c y) ∧
  (∀ x y : ℝ, y ≤ x → x < 0 → f a b c x ≤ f a b c y) :=
by
  sorry

end NUMINAMATH_GPT_function_monotonicity_l527_52768


namespace NUMINAMATH_GPT_sequence_sum_l527_52721

theorem sequence_sum:
  ∀ (y : ℕ → ℕ), 
  (y 1 = 100) → 
  (∀ k ≥ 2, y k = y (k - 1) ^ 2 + 2 * y (k - 1) + 1) →
  ( ∑' n, 1 / (y n + 1) = 1 / 101 ) :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l527_52721


namespace NUMINAMATH_GPT_min_gb_for_plan_y_to_be_cheaper_l527_52704

theorem min_gb_for_plan_y_to_be_cheaper (g : ℕ) : 20 * g > 3000 + 10 * g → g ≥ 301 := by
  sorry

end NUMINAMATH_GPT_min_gb_for_plan_y_to_be_cheaper_l527_52704


namespace NUMINAMATH_GPT_B_investment_is_72000_l527_52784

noncomputable def A_investment : ℝ := 27000
noncomputable def C_investment : ℝ := 81000
noncomputable def C_profit : ℝ := 36000
noncomputable def total_profit : ℝ := 80000

noncomputable def B_investment : ℝ :=
  let total_investment := (C_investment * total_profit) / C_profit
  total_investment - A_investment - C_investment

theorem B_investment_is_72000 :
  B_investment = 72000 :=
by
  sorry

end NUMINAMATH_GPT_B_investment_is_72000_l527_52784


namespace NUMINAMATH_GPT_pages_per_donut_l527_52757

def pages_written (total_pages : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ) : ℕ :=
  let donuts := total_calories / calories_per_donut
  total_pages / donuts

theorem pages_per_donut (total_pages : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ): 
  total_pages = 12 → calories_per_donut = 150 → total_calories = 900 → pages_written total_pages calories_per_donut total_calories = 2 := by
  intros
  sorry

end NUMINAMATH_GPT_pages_per_donut_l527_52757


namespace NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l527_52761

noncomputable def f (a x : ℝ) : ℝ := 2 * a * Real.log x - x^2 + a

theorem monotonicity_of_f (a : ℝ) :
  (∀ x > 0, (a ≤ 0 → f a x ≤ f a (x - 1)) ∧ 
           (a > 0 → ((x < Real.sqrt a → f a x ≤ f a (x + 1)) ∨ 
                     (x > Real.sqrt a → f a x ≥ f a (x - 1))))) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) → (0 ≤ a ∧ a ≤ 1) := sorry

end NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l527_52761


namespace NUMINAMATH_GPT_inverse_100_mod_101_l527_52758

theorem inverse_100_mod_101 : (100 * 100) % 101 = 1 :=
by
  -- Proof can be provided here.
  sorry

end NUMINAMATH_GPT_inverse_100_mod_101_l527_52758


namespace NUMINAMATH_GPT_triangle_perimeter_l527_52720

theorem triangle_perimeter (r A : ℝ) (h_r : r = 2.5) (h_A : A = 50) : 
  ∃ p : ℝ, p = 40 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l527_52720


namespace NUMINAMATH_GPT_same_exponent_for_all_bases_l527_52715

theorem same_exponent_for_all_bases {a : Type} [LinearOrderedField a] {C : a} (ha : ∀ (a : a), a ≠ 0 → a^0 = C) : C = 1 :=
by
  sorry

end NUMINAMATH_GPT_same_exponent_for_all_bases_l527_52715


namespace NUMINAMATH_GPT_last_four_digits_of_5_pow_2016_l527_52705

theorem last_four_digits_of_5_pow_2016 :
  (5^2016) % 10000 = 625 :=
by
  -- Establish periodicity of last four digits in powers of 5
  sorry

end NUMINAMATH_GPT_last_four_digits_of_5_pow_2016_l527_52705


namespace NUMINAMATH_GPT_contrapositive_l527_52711

theorem contrapositive (p q : Prop) (h : p → q) : ¬q → ¬p :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_l527_52711


namespace NUMINAMATH_GPT_expected_value_of_difference_is_4_point_5_l527_52783

noncomputable def expected_value_difference : ℚ :=
  (2 * 6 / 56 + 3 * 10 / 56 + 4 * 12 / 56 + 5 * 12 / 56 + 6 * 10 / 56 + 7 * 6 / 56)

theorem expected_value_of_difference_is_4_point_5 :
  expected_value_difference = 4.5 := sorry

end NUMINAMATH_GPT_expected_value_of_difference_is_4_point_5_l527_52783


namespace NUMINAMATH_GPT_five_pm_is_seventeen_hours_ten_pm_is_twenty_two_hours_time_difference_is_forty_minutes_l527_52750

-- Define what "5 PM" and "10 PM" mean in hours
def five_pm: ℕ := 17
def ten_pm: ℕ := 22

-- Define function for converting from PM to 24-hour time
def pm_to_hours (n: ℕ): ℕ := n + 12

-- Define the times in minutes for comparison
def time_16_40: ℕ := 16 * 60 + 40
def time_17_20: ℕ := 17 * 60 + 20

-- Define the differences in minutes
def minutes_passed (start end_: ℕ): ℕ := end_ - start

-- Prove the equivalences
theorem five_pm_is_seventeen_hours: pm_to_hours 5 = five_pm := by 
  unfold pm_to_hours
  unfold five_pm
  rfl

theorem ten_pm_is_twenty_two_hours: pm_to_hours 10 = ten_pm := by 
  unfold pm_to_hours
  unfold ten_pm
  rfl

theorem time_difference_is_forty_minutes: minutes_passed time_16_40 time_17_20 = 40 := by 
  unfold time_16_40
  unfold time_17_20
  unfold minutes_passed
  rfl

#check five_pm_is_seventeen_hours
#check ten_pm_is_twenty_two_hours
#check time_difference_is_forty_minutes

end NUMINAMATH_GPT_five_pm_is_seventeen_hours_ten_pm_is_twenty_two_hours_time_difference_is_forty_minutes_l527_52750


namespace NUMINAMATH_GPT_sum_first_ten_terms_arithmetic_sequence_l527_52710

theorem sum_first_ten_terms_arithmetic_sequence (a₁ d : ℤ) (h₁ : a₁ = -3) (h₂ : d = 4) : 
  let a₁₀ := a₁ + (9 * d)
  let S := ((a₁ + a₁₀) / 2) * 10
  S = 150 :=
by
  subst h₁
  subst h₂
  let a₁₀ := -3 + (9 * 4)
  let S := ((-3 + a₁₀) / 2) * 10
  sorry

end NUMINAMATH_GPT_sum_first_ten_terms_arithmetic_sequence_l527_52710


namespace NUMINAMATH_GPT_ab_finish_job_in_15_days_l527_52769

theorem ab_finish_job_in_15_days (A B C : ℝ) (h1 : A + B + C = 1/12) (h2 : C = 1/60) : 1 / (A + B) = 15 := 
by
  sorry

end NUMINAMATH_GPT_ab_finish_job_in_15_days_l527_52769


namespace NUMINAMATH_GPT_find_y_l527_52734

theorem find_y (y : ℝ) (h : 2 * y / 3 = 30) : y = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l527_52734


namespace NUMINAMATH_GPT_f_2_equals_12_l527_52754

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^3 + x^2 else - (2 * (-x)^3 + (-x)^2)

theorem f_2_equals_12 : f 2 = 12 := by
  sorry

end NUMINAMATH_GPT_f_2_equals_12_l527_52754


namespace NUMINAMATH_GPT_base4_sum_correct_l527_52795

/-- Define the base-4 numbers as natural numbers. -/
def a := 3 * 4^2 + 1 * 4^1 + 2 * 4^0
def b := 3 * 4^1 + 1 * 4^0
def c := 3 * 4^0

/-- Define their sum in base 10. -/
def sum_base_10 := a + b + c

/-- Define the target sum in base 4 as a natural number. -/
def target := 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0

/-- Prove that the sum of the base-4 numbers equals the target sum in base 4. -/
theorem base4_sum_correct : sum_base_10 = target := by
  sorry

end NUMINAMATH_GPT_base4_sum_correct_l527_52795


namespace NUMINAMATH_GPT_triangle_side_AC_l527_52744

theorem triangle_side_AC 
  (AB BC : ℝ)
  (angle_C : ℝ)
  (h1 : AB = Real.sqrt 13)
  (h2 : BC = 3)
  (h3 : angle_C = Real.pi / 3) :
  ∃ AC : ℝ, AC = 4 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_side_AC_l527_52744


namespace NUMINAMATH_GPT_fish_population_estimation_l527_52771

-- Definitions based on conditions
def fish_tagged_day1 : ℕ := 80
def fish_caught_day2 : ℕ := 100
def fish_tagged_day2 : ℕ := 20
def fish_caught_day3 : ℕ := 120
def fish_tagged_day3 : ℕ := 36

-- The average percentage of tagged fish caught on the second and third days
def avg_tag_percentage : ℚ := (20 / 100 + 36 / 120) / 2

-- Statement of the proof problem
theorem fish_population_estimation :
  (avg_tag_percentage * P = fish_tagged_day1) → 
  P = 320 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fish_population_estimation_l527_52771


namespace NUMINAMATH_GPT_find_sum_of_squares_l527_52742

theorem find_sum_of_squares (x y : ℝ) (h1: x * y = 16) (h2: x^2 + y^2 = 34) : (x + y) ^ 2 = 66 :=
by sorry

end NUMINAMATH_GPT_find_sum_of_squares_l527_52742


namespace NUMINAMATH_GPT_expansion_correct_l527_52762

-- Define the polynomials
def poly1 (z : ℤ) : ℤ := 3 * z^2 + 4 * z - 5
def poly2 (z : ℤ) : ℤ := 4 * z^4 - 3 * z^2 + 2

-- Define the expected expanded polynomial
def expanded_poly (z : ℤ) : ℤ := 12 * z^6 + 16 * z^5 - 29 * z^4 - 12 * z^3 + 21 * z^2 + 8 * z - 10

-- The theorem that proves the equivalence of the expanded form
theorem expansion_correct (z : ℤ) : (poly1 z) * (poly2 z) = expanded_poly z := by
  sorry

end NUMINAMATH_GPT_expansion_correct_l527_52762


namespace NUMINAMATH_GPT_value_of_B_l527_52702

theorem value_of_B (x y : ℕ) (h1 : x > y) (h2 : y > 1) (h3 : x * y = x + y + 22) :
  (x / y) = 12 :=
sorry

end NUMINAMATH_GPT_value_of_B_l527_52702


namespace NUMINAMATH_GPT_bridge_length_l527_52729

noncomputable def length_of_bridge (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_of_train_ms := (speed_of_train_kmh * 1000) / 3600
  let total_distance := speed_of_train_ms * time_seconds
  total_distance - length_of_train

theorem bridge_length (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_seconds : ℕ) (h1 : length_of_train = 170) (h2 : speed_of_train_kmh = 45) (h3 : time_seconds = 30) :
  length_of_bridge length_of_train speed_of_train_kmh time_seconds = 205 :=
by 
  rw [h1, h2, h3]
  unfold length_of_bridge
  simp
  sorry

end NUMINAMATH_GPT_bridge_length_l527_52729


namespace NUMINAMATH_GPT_Emily_age_is_23_l527_52799

variable (UncleBob Daniel Emily Zoe : ℕ)

-- Conditions
axiom h1 : UncleBob = 54
axiom h2 : Daniel = UncleBob / 2
axiom h3 : Emily = Daniel - 4
axiom h4 : Emily = 2 * Zoe / 3

-- Question: Prove that Emily's age is 23
theorem Emily_age_is_23 : Emily = 23 :=
by
  sorry

end NUMINAMATH_GPT_Emily_age_is_23_l527_52799
