import Mathlib

namespace NUMINAMATH_GPT_probability_three_black_balls_probability_white_ball_l2264_226434

-- Definitions representing conditions
def total_ratio (A B C : ℕ) := A / B = 5 / 4 ∧ B / C = 4 / 6

-- Proportions of black balls in each box
def proportion_black_A (black_A total_A : ℕ) := black_A = 40 * total_A / 100
def proportion_black_B (black_B total_B : ℕ) := black_B = 25 * total_B / 100
def proportion_black_C (black_C total_C : ℕ) := black_C = 50 * total_C / 100

-- Problem 1: Probability of selecting a black ball from each box
theorem probability_three_black_balls
  (A B C : ℕ)
  (total_A total_B total_C : ℕ)
  (black_A black_B black_C : ℕ)
  (h1 : total_ratio A B C)
  (h2 : proportion_black_A black_A total_A)
  (h3 : proportion_black_B black_B total_B)
  (h4 : proportion_black_C black_C total_C) :
  (black_A / total_A) * (black_B / total_B) * (black_C / total_C) = 1 / 20 :=
  sorry

-- Problem 2: Probability of selecting a white ball from the mixed total
theorem probability_white_ball
  (A B C : ℕ)
  (total_A total_B total_C : ℕ)
  (black_A black_B black_C : ℕ)
  (white_A white_B white_C : ℕ)
  (h1 : total_ratio A B C)
  (h2 : proportion_black_A black_A total_A)
  (h3 : proportion_black_B black_B total_B)
  (h4 : proportion_black_C black_C total_C)
  (h5 : white_A = total_A - black_A)
  (h6 : white_B = total_B - black_B)
  (h7 : white_C = total_C - black_C) :
  (white_A + white_B + white_C) / (total_A + total_B + total_C) = 3 / 5 :=
  sorry

end NUMINAMATH_GPT_probability_three_black_balls_probability_white_ball_l2264_226434


namespace NUMINAMATH_GPT_find_range_of_a_l2264_226413

def setA (x : ℝ) : Prop := 1 < x ∧ x < 2
def setB (x : ℝ) : Prop := 3 / 2 < x ∧ x < 4
def setUnion (x : ℝ) : Prop := 1 < x ∧ x < 4
def setP (a x : ℝ) : Prop := a < x ∧ x < a + 2

theorem find_range_of_a (a : ℝ) :
  (∀ x, setP a x → setUnion x) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_a_l2264_226413


namespace NUMINAMATH_GPT_ellipse_nec_but_not_suff_l2264_226408

-- Definitions and conditions
def isEllipse (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c

/-- Given that the sum of the distances from a moving point P in the plane to two fixed points is constant,
the condition is necessary but not sufficient for the trajectory of the moving point P being an ellipse. -/
theorem ellipse_nec_but_not_suff (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (c : ℝ) :
  (∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c) →
  (c > dist F1 F2 → ¬ isEllipse P F1 F2) ∧ (isEllipse P F1 F2 → ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_nec_but_not_suff_l2264_226408


namespace NUMINAMATH_GPT_inequality_proof_l2264_226492

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b * c = 1) : 
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2264_226492


namespace NUMINAMATH_GPT_product_bc_l2264_226482

theorem product_bc {b c : ℤ} (h1 : ∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) :
    b * c = 110 :=
sorry

end NUMINAMATH_GPT_product_bc_l2264_226482


namespace NUMINAMATH_GPT_Piglet_ate_one_l2264_226490

theorem Piglet_ate_one (V S K P : ℕ) (h1 : V + S + K + P = 70)
  (h2 : S + K = 45) (h3 : V > S) (h4 : V > K) (h5 : V > P) 
  (h6 : V ≥ 1) (h7 : S ≥ 1) (h8 : K ≥ 1) (h9 : P ≥ 1) : P = 1 :=
sorry

end NUMINAMATH_GPT_Piglet_ate_one_l2264_226490


namespace NUMINAMATH_GPT_opposite_of_neg2_is_2_l2264_226475

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_neg2_is_2 : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg2_is_2_l2264_226475


namespace NUMINAMATH_GPT_club_membership_l2264_226473

theorem club_membership (n : ℕ) : 
  n ≡ 6 [MOD 10] → n ≡ 6 [MOD 11] → 200 ≤ n ∧ n ≤ 300 → n = 226 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_club_membership_l2264_226473


namespace NUMINAMATH_GPT_time_until_heavy_lifting_l2264_226420

-- Define the conditions given
def pain_subside_days : ℕ := 3
def healing_multiplier : ℕ := 5
def additional_wait_days : ℕ := 3
def weeks_before_lifting : ℕ := 3
def days_in_week : ℕ := 7

-- Define the proof statement
theorem time_until_heavy_lifting : 
    let full_healing_days := pain_subside_days * healing_multiplier
    let total_days_before_exercising := full_healing_days + additional_wait_days
    let lifting_wait_days := weeks_before_lifting * days_in_week
    total_days_before_exercising + lifting_wait_days = 39 := 
by
  sorry

end NUMINAMATH_GPT_time_until_heavy_lifting_l2264_226420


namespace NUMINAMATH_GPT_misread_weight_l2264_226466

-- Definitions based on given conditions in part (a)
def initial_avg_weight : ℝ := 58.4
def num_boys : ℕ := 20
def correct_weight : ℝ := 61
def correct_avg_weight : ℝ := 58.65

-- The Lean theorem statement that needs to be proved
theorem misread_weight :
  let incorrect_total_weight := initial_avg_weight * num_boys
  let correct_total_weight := correct_avg_weight * num_boys
  let weight_diff := correct_total_weight - incorrect_total_weight
  correct_weight - weight_diff = 56 := sorry

end NUMINAMATH_GPT_misread_weight_l2264_226466


namespace NUMINAMATH_GPT_mrs_hilt_chapters_read_l2264_226499

def number_of_books : ℝ := 4.0
def chapters_per_book : ℝ := 4.25
def total_chapters_read : ℝ := number_of_books * chapters_per_book

theorem mrs_hilt_chapters_read : total_chapters_read = 17 :=
by
  unfold total_chapters_read
  norm_num
  sorry

end NUMINAMATH_GPT_mrs_hilt_chapters_read_l2264_226499


namespace NUMINAMATH_GPT_no_such_b_exists_l2264_226417

theorem no_such_b_exists (b : ℝ) (hb : 0 < b) :
  ¬(∃ k : ℝ, 0 < k ∧ ∀ n : ℕ, 0 < n → (n - k ≤ (⌊b * n⌋ : ℤ) ∧ (⌊b * n⌋ : ℤ) < n)) :=
by
  sorry

end NUMINAMATH_GPT_no_such_b_exists_l2264_226417


namespace NUMINAMATH_GPT_girls_more_than_boys_l2264_226437

theorem girls_more_than_boys (boys girls : ℕ) (ratio_boys ratio_girls : ℕ) 
  (h1 : ratio_boys = 5)
  (h2 : ratio_girls = 13)
  (h3 : boys = 50)
  (h4 : girls = (boys / ratio_boys) * ratio_girls) : 
  girls - boys = 80 :=
by
  sorry

end NUMINAMATH_GPT_girls_more_than_boys_l2264_226437


namespace NUMINAMATH_GPT_quadratic_function_even_l2264_226498

theorem quadratic_function_even (a b : ℝ) (h1 : ∀ x : ℝ, x^2 + (a-1)*x + a + b = x^2 - (a-1)*x + a + b) (h2 : 4 + (a-1)*2 + a + b = 0) : a + b = -4 := 
sorry

end NUMINAMATH_GPT_quadratic_function_even_l2264_226498


namespace NUMINAMATH_GPT_inverse_proportion_neg_k_l2264_226491

theorem inverse_proportion_neg_k (x1 x2 y1 y2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 > y2) :
  ∃ k : ℝ, k < 0 ∧ (∀ x, (x = x1 → y1 = k / x) ∧ (x = x2 → y2 = k / x)) := by
  use -1
  sorry

end NUMINAMATH_GPT_inverse_proportion_neg_k_l2264_226491


namespace NUMINAMATH_GPT_length_BE_l2264_226411

-- Define points and distances
variables (A B C D E : Type)
variable {AB : ℝ}
variable {BC : ℝ}
variable {CD : ℝ}
variable {DA : ℝ}

-- Given conditions
axiom AB_length : AB = 5
axiom BC_length : BC = 7
axiom CD_length : CD = 8
axiom DA_length : DA = 6

-- Bugs travelling in opposite directions from point A meet at E
axiom bugs_meet_at_E : True

-- Proving the length BE
theorem length_BE : BE = 6 :=
by
  -- Currently, this is a statement. The proof is not included.
  sorry

end NUMINAMATH_GPT_length_BE_l2264_226411


namespace NUMINAMATH_GPT_remainder_x1001_mod_poly_l2264_226427

noncomputable def remainder_poly_div (n k : ℕ) (f g : Polynomial ℚ) : Polynomial ℚ :=
  Polynomial.modByMonic f g

theorem remainder_x1001_mod_poly :
  remainder_poly_div 1001 3 (Polynomial.X ^ 1001) (Polynomial.X ^ 3 - Polynomial.X ^ 2 - Polynomial.X + 1) = Polynomial.X ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_x1001_mod_poly_l2264_226427


namespace NUMINAMATH_GPT_smallest_number_is_a_l2264_226483

def smallest_number_among_options : ℤ :=
  let a: ℤ := -3
  let b: ℤ := 0
  let c: ℤ := -(-1)
  let d: ℤ := (-1)^2
  min a (min b (min c d))

theorem smallest_number_is_a : smallest_number_among_options = -3 :=
  by
    sorry

end NUMINAMATH_GPT_smallest_number_is_a_l2264_226483


namespace NUMINAMATH_GPT_find_k_l2264_226480

theorem find_k (k : ℝ) (A B : ℝ → ℝ)
  (hA : ∀ x, A x = 2 * x^2 + k * x - 6 * x)
  (hB : ∀ x, B x = -x^2 + k * x - 1)
  (hIndependent : ∀ x, ∃ C : ℝ, A x + 2 * B x = C) :
  k = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_k_l2264_226480


namespace NUMINAMATH_GPT_number_of_valid_triples_l2264_226410

theorem number_of_valid_triples :
  ∃ (count : ℕ), count = 3 ∧
  ∀ (x y z : ℕ), 0 < x → 0 < y → 0 < z →
  Nat.lcm x y = 120 → Nat.lcm y z = 1000 → Nat.lcm x z = 480 →
  (∃ (u v w : ℕ), u = x ∧ v = y ∧ w = z ∧ count = 3) :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_triples_l2264_226410


namespace NUMINAMATH_GPT_car_speed_ratio_l2264_226431

-- Assuming the bridge length as L, pedestrian's speed as v_p, and car's speed as v_c.
variables (L v_p v_c : ℝ)

-- Mathematically equivalent proof problem statement in Lean 4.
theorem car_speed_ratio (h1 : 2/5 * L = 2/5 * L)
                       (h2 : (L - 2/5 * L) / v_p = L / v_c) :
    v_c = 5 * v_p := 
  sorry

end NUMINAMATH_GPT_car_speed_ratio_l2264_226431


namespace NUMINAMATH_GPT_complex_square_eq_l2264_226496

open Complex

theorem complex_square_eq {a b : ℝ} (h : (a + b * Complex.I)^2 = Complex.mk 3 4) : a^2 + b^2 = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_complex_square_eq_l2264_226496


namespace NUMINAMATH_GPT_ratio_of_sums_l2264_226467

/-- Define the relevant arithmetic sequences and sums -/

-- Sequence 1: 3, 6, 9, ..., 45
def seq1 : ℕ → ℕ
| n => 3 * n + 3

-- Sequence 2: 4, 8, 12, ..., 64
def seq2 : ℕ → ℕ
| n => 4 * n + 4

-- Sum function for arithmetic sequences
def sum_arith_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n-1) * d) / 2

noncomputable def sum_seq1 : ℕ := sum_arith_seq 3 3 15 -- 3 + 6 + ... + 45
noncomputable def sum_seq2 : ℕ := sum_arith_seq 4 4 16 -- 4 + 8 + ... + 64

-- Prove that the ratio of sums is 45/68
theorem ratio_of_sums : (sum_seq1 : ℚ) / sum_seq2 = 45 / 68 :=
  sorry

end NUMINAMATH_GPT_ratio_of_sums_l2264_226467


namespace NUMINAMATH_GPT_boat_shipments_divisor_l2264_226442

/-- 
Given:
1. There exists an integer B representing the number of boxes that can be divided into S equal shipments by boat.
2. B can be divided into 24 equal shipments by truck.
3. The smallest number of boxes B is 120.
Prove that S, the number of equal shipments by boat, is 60.
--/
theorem boat_shipments_divisor (B S : ℕ) (h1 : B % S = 0) (h2 : B % 24 = 0) (h3 : B = 120) : S = 60 := 
sorry

end NUMINAMATH_GPT_boat_shipments_divisor_l2264_226442


namespace NUMINAMATH_GPT_singh_gain_l2264_226497

def initial_amounts (B A S : ℕ) : Prop :=
  B = 70 ∧ A = 70 ∧ S = 70

def ratio_Ashtikar_Singh (A S : ℕ) : Prop :=
  2 * A = S

def ratio_Singh_Bhatia (S B : ℕ) : Prop :=
  4 * B = S

def total_conservation (A S B : ℕ) : Prop :=
  A + S + B = 210

theorem singh_gain : ∀ B A S fA fB fS : ℕ,
  initial_amounts B A S →
  ratio_Ashtikar_Singh fA fS →
  ratio_Singh_Bhatia fS fB →
  total_conservation fA fS fB →
  fS - S = 50 :=
by
  intros B A S fA fB fS
  intros i rA rS tC
  sorry

end NUMINAMATH_GPT_singh_gain_l2264_226497


namespace NUMINAMATH_GPT_age_difference_l2264_226401

theorem age_difference (A B C : ℕ) (h1 : B = 10) (h2 : B = 2 * C) (h3 : A + B + C = 27) : A - B = 2 :=
 by
  sorry

end NUMINAMATH_GPT_age_difference_l2264_226401


namespace NUMINAMATH_GPT_correct_option_l2264_226478

-- Define the given conditions
def a : ℕ := 7^5
def b : ℕ := 5^7

-- State the theorem to be proven
theorem correct_option : a^7 * b^5 = 35^35 := by
  -- insert the proof here
  sorry

end NUMINAMATH_GPT_correct_option_l2264_226478


namespace NUMINAMATH_GPT_triangle_right_angle_l2264_226421

theorem triangle_right_angle {A B C : ℝ} 
  (h1 : A + B + C = 180)
  (h2 : A = B)
  (h3 : A = (1/2) * C) :
  C = 90 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_right_angle_l2264_226421


namespace NUMINAMATH_GPT_christine_makes_two_cakes_l2264_226406

theorem christine_makes_two_cakes (tbsp_per_egg_white : ℕ) 
  (egg_whites_per_cake : ℕ) 
  (total_tbsp_aquafaba : ℕ)
  (h1 : tbsp_per_egg_white = 2) 
  (h2 : egg_whites_per_cake = 8) 
  (h3 : total_tbsp_aquafaba = 32) : 
  total_tbsp_aquafaba / tbsp_per_egg_white / egg_whites_per_cake = 2 := by 
  sorry

end NUMINAMATH_GPT_christine_makes_two_cakes_l2264_226406


namespace NUMINAMATH_GPT_complement_U_A_correct_l2264_226487

-- Step 1: Define the universal set U
def U (x : ℝ) := x > 0

-- Step 2: Define the set A
def A (x : ℝ) := 0 < x ∧ x < 1

-- Step 3: Define the complement of A in U
def complement_U_A (x : ℝ) := U x ∧ ¬ A x

-- Step 4: Define the expected complement
def expected_complement (x : ℝ) := x ≥ 1

-- Step 5: The proof problem statement
theorem complement_U_A_correct (x : ℝ) : complement_U_A x = expected_complement x := by
  sorry

end NUMINAMATH_GPT_complement_U_A_correct_l2264_226487


namespace NUMINAMATH_GPT_seeds_total_l2264_226444

-- Define the conditions as given in the problem.
def Bom_seeds : ℕ := 300
def Gwi_seeds : ℕ := Bom_seeds + 40
def Yeon_seeds : ℕ := 3 * Gwi_seeds

-- Lean statement to prove the total number of seeds.
theorem seeds_total : Bom_seeds + Gwi_seeds + Yeon_seeds = 1660 := 
by
  -- Assuming all given definitions and conditions are true,
  -- we aim to prove the final theorem statement.
  sorry

end NUMINAMATH_GPT_seeds_total_l2264_226444


namespace NUMINAMATH_GPT_curve_in_second_quadrant_range_l2264_226438

theorem curve_in_second_quadrant_range (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0)) → a > 2 :=
by
  sorry

end NUMINAMATH_GPT_curve_in_second_quadrant_range_l2264_226438


namespace NUMINAMATH_GPT_salary_percentage_change_l2264_226404

theorem salary_percentage_change (S : ℝ) (x : ℝ) :
  (S * (1 - (x / 100)) * (1 + (x / 100)) = S * 0.84) ↔ (x = 40) :=
by
  sorry

end NUMINAMATH_GPT_salary_percentage_change_l2264_226404


namespace NUMINAMATH_GPT_dollar_op_5_neg2_l2264_226402

def dollar_op (x y : Int) : Int := x * (2 * y - 1) + 2 * x * y

theorem dollar_op_5_neg2 :
  dollar_op 5 (-2) = -45 := by
  sorry

end NUMINAMATH_GPT_dollar_op_5_neg2_l2264_226402


namespace NUMINAMATH_GPT_zeros_at_end_of_product1_value_of_product2_l2264_226464

-- Definitions and conditions
def product1 := 360 * 5
def product2 := 250 * 4

-- Statements of the proof problems
theorem zeros_at_end_of_product1 : Nat.digits 10 product1 = [0, 0, 8, 1] := by
  sorry

theorem value_of_product2 : product2 = 1000 := by
  sorry

end NUMINAMATH_GPT_zeros_at_end_of_product1_value_of_product2_l2264_226464


namespace NUMINAMATH_GPT_white_squares_95th_figure_l2264_226418

theorem white_squares_95th_figure : ∀ (T : ℕ → ℕ),
  T 1 = 8 →
  (∀ n ≥ 1, T (n + 1) = T n + 5) →
  T 95 = 478 :=
by
  intros T hT1 hTrec
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_white_squares_95th_figure_l2264_226418


namespace NUMINAMATH_GPT_smallest_c_geometric_arithmetic_progression_l2264_226409

theorem smallest_c_geometric_arithmetic_progression (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : 0 < c) 
(h4 : b ^ 2 = a * c) (h5 : a + b = 2 * c) : c = 1 :=
sorry

end NUMINAMATH_GPT_smallest_c_geometric_arithmetic_progression_l2264_226409


namespace NUMINAMATH_GPT_tan_120_deg_l2264_226456

theorem tan_120_deg : Real.tan (120 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_120_deg_l2264_226456


namespace NUMINAMATH_GPT_factorize_16x2_minus_1_l2264_226414

theorem factorize_16x2_minus_1 (x : ℝ) : 16 * x^2 - 1 = (4 * x + 1) * (4 * x - 1) := by
  sorry

end NUMINAMATH_GPT_factorize_16x2_minus_1_l2264_226414


namespace NUMINAMATH_GPT_parallel_vectors_eq_l2264_226465

theorem parallel_vectors_eq (m : ℤ) (h : (m, 4) = (3 * k, -2 * k)) : m = -6 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_eq_l2264_226465


namespace NUMINAMATH_GPT_polynomial_evaluation_l2264_226441

theorem polynomial_evaluation (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2 * a^2 + 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l2264_226441


namespace NUMINAMATH_GPT_minimal_reciprocal_sum_l2264_226452

theorem minimal_reciprocal_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) :
    (4 / m) + (1 / n) = (30 / (m * n)) → m = 10 ∧ n = 5 :=
sorry

end NUMINAMATH_GPT_minimal_reciprocal_sum_l2264_226452


namespace NUMINAMATH_GPT_solve_equation_l2264_226445

theorem solve_equation :
  ∃ a b x : ℤ, 
  ((a * x^2 + b * x + 14)^2 + (b * x^2 + a * x + 8)^2 = 0) 
  ↔ (a = -6 ∧ b = -5 ∧ x = -2) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_equation_l2264_226445


namespace NUMINAMATH_GPT_left_side_value_l2264_226458

-- Define the relevant variables and conditions
variable (L R B : ℕ)

-- Assuming conditions
def sum_of_sides (L R B : ℕ) : Prop := L + R + B = 50
def right_side_relation (L R : ℕ) : Prop := R = L + 2
def base_value (B : ℕ) : Prop := B = 24

-- Main theorem statement
theorem left_side_value (L R B : ℕ) (h1 : sum_of_sides L R B) (h2 : right_side_relation L R) (h3 : base_value B) : L = 12 :=
sorry

end NUMINAMATH_GPT_left_side_value_l2264_226458


namespace NUMINAMATH_GPT_range_of_a_l2264_226426

def p (a : ℝ) : Prop := a > -1
def q (a : ℝ) : Prop := ∀ m : ℝ, -2 ≤ m ∧ m ≤ 4 → a^2 - a ≥ 4 - m

theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ (-1 < a ∧ a < 3) ∨ a ≤ -2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2264_226426


namespace NUMINAMATH_GPT_depak_bank_account_l2264_226455

theorem depak_bank_account :
  ∃ (n : ℕ), (x + 1 = 6 * n) ∧ n = 1 → x = 5 := 
sorry

end NUMINAMATH_GPT_depak_bank_account_l2264_226455


namespace NUMINAMATH_GPT_ice_cream_depth_l2264_226447

theorem ice_cream_depth 
  (r_sphere : ℝ) 
  (r_cylinder : ℝ) 
  (h_cylinder : ℝ) 
  (V_sphere : ℝ) 
  (V_cylinder : ℝ) 
  (constant_density : V_sphere = V_cylinder)
  (r_sphere_eq : r_sphere = 2) 
  (r_cylinder_eq : r_cylinder = 8) 
  (V_sphere_def : V_sphere = (4 / 3) * Real.pi * r_sphere^3) 
  (V_cylinder_def : V_cylinder = Real.pi * r_cylinder^2 * h_cylinder) 
  : h_cylinder = 1 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_ice_cream_depth_l2264_226447


namespace NUMINAMATH_GPT_units_digit_17_pow_39_l2264_226459

theorem units_digit_17_pow_39 : 
  ∃ d : ℕ, d < 10 ∧ (17^39 % 10 = d) ∧ d = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_17_pow_39_l2264_226459


namespace NUMINAMATH_GPT_find_n_l2264_226453

theorem find_n (n : ℕ) (h : 1 < n) :
  (∀ a b : ℕ, Nat.gcd a b = 1 → (a % n = b % n ↔ (a * b) % n = 1)) →
  (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12 ∨ n = 24) :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2264_226453


namespace NUMINAMATH_GPT_cube_volume_split_l2264_226440

theorem cube_volume_split (x y z : ℝ) (h : x > 0) :
  ∃ y z : ℝ, y > 0 ∧ z > 0 ∧ y^3 + z^3 = x^3 :=
sorry

end NUMINAMATH_GPT_cube_volume_split_l2264_226440


namespace NUMINAMATH_GPT_least_element_of_special_set_l2264_226433

theorem least_element_of_special_set :
  ∃ T : Finset ℕ, T ⊆ Finset.range 16 ∧ T.card = 7 ∧
    (∀ {x y : ℕ}, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) ∧ 
    (∀ {z : ℕ}, z ∈ T → ∀ {x y : ℕ}, x ≠ y → x ∈ T → y ∈ T → z ≠ x + y) ∧
    ∀ (x : ℕ), x ∈ T → x ≥ 4 :=
sorry

end NUMINAMATH_GPT_least_element_of_special_set_l2264_226433


namespace NUMINAMATH_GPT_group_capacity_l2264_226432

theorem group_capacity (total_students : ℕ) (selected_students : ℕ) (removed_students : ℕ) :
  total_students = 5008 → selected_students = 200 → removed_students = 8 →
  (total_students - removed_students) / selected_students = 25 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_group_capacity_l2264_226432


namespace NUMINAMATH_GPT_Trevor_tip_l2264_226485

variable (Uber Lyft Taxi : ℕ)
variable (TotalCost : ℕ)

theorem Trevor_tip 
  (h1 : Uber = Lyft + 3) 
  (h2 : Lyft = Taxi + 4) 
  (h3 : Uber = 22) 
  (h4 : TotalCost = 18)
  (h5 : Taxi = 15) :
  (TotalCost - Taxi) * 100 / Taxi = 20 := by
  sorry

end NUMINAMATH_GPT_Trevor_tip_l2264_226485


namespace NUMINAMATH_GPT_total_cost_of_projectors_and_computers_l2264_226448

theorem total_cost_of_projectors_and_computers :
  let n_p := 8
  let c_p := 7500
  let n_c := 32
  let c_c := 3600
  (n_p * c_p + n_c * c_c) = 175200 := by
  let n_p := 8
  let c_p := 7500
  let n_c := 32
  let c_c := 3600
  sorry 

end NUMINAMATH_GPT_total_cost_of_projectors_and_computers_l2264_226448


namespace NUMINAMATH_GPT_comic_books_collection_l2264_226428

theorem comic_books_collection (initial_ky: ℕ) (rate_ky: ℕ) (initial_la: ℕ) (rate_la: ℕ) (months: ℕ) :
  initial_ky = 50 → rate_ky = 1 → initial_la = 20 → rate_la = 7 → months = 33 →
  initial_la + rate_la * months = 3 * (initial_ky + rate_ky * months) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end NUMINAMATH_GPT_comic_books_collection_l2264_226428


namespace NUMINAMATH_GPT_rachel_earnings_one_hour_l2264_226451

-- Define Rachel's hourly wage
def rachelWage : ℝ := 12.00

-- Define the number of people Rachel serves in one hour
def peopleServed : ℕ := 20

-- Define the tip amount per person
def tipPerPerson : ℝ := 1.25

-- Calculate the total tips received
def totalTips : ℝ := (peopleServed : ℝ) * tipPerPerson

-- Calculate the total amount Rachel makes in one hour
def totalEarnings : ℝ := rachelWage + totalTips

-- The theorem to state Rachel's total earnings in one hour
theorem rachel_earnings_one_hour : totalEarnings = 37.00 := 
by
  sorry

end NUMINAMATH_GPT_rachel_earnings_one_hour_l2264_226451


namespace NUMINAMATH_GPT_missing_fraction_of_coins_l2264_226429

-- Defining the initial conditions
def total_coins (x : ℕ) := x
def lost_coins (x : ℕ) := (1 / 2) * x
def found_coins (x : ℕ) := (3 / 8) * x

-- Theorem statement
theorem missing_fraction_of_coins (x : ℕ) : 
  (total_coins x - lost_coins x + found_coins x) = (7 / 8) * x :=
by
  sorry  -- proof is omitted as per the instructions

end NUMINAMATH_GPT_missing_fraction_of_coins_l2264_226429


namespace NUMINAMATH_GPT_probability_sum_odd_l2264_226425

theorem probability_sum_odd (x y : ℕ) 
  (hx : x > 0) (hy : y > 0) 
  (h_even : ∃ z : ℕ, z % 2 = 0 ∧ z > 0) 
  (h_odd : ∃ z : ℕ, z % 2 = 1 ∧ z > 0) : 
  (∃ p : ℝ, 0 < p ∧ p < 1 ∧ p = 0.5) :=
sorry

end NUMINAMATH_GPT_probability_sum_odd_l2264_226425


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2264_226415

theorem quadratic_inequality_solution : 
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3 / 2} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2264_226415


namespace NUMINAMATH_GPT_max_product_l2264_226419

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end NUMINAMATH_GPT_max_product_l2264_226419


namespace NUMINAMATH_GPT_number_of_dozen_eggs_to_mall_l2264_226472

-- Define the conditions as assumptions
def number_of_dozen_eggs_collected (x : Nat) : Prop :=
  x = 2 * 8

def number_of_dozen_eggs_to_market (x : Nat) : Prop :=
  x = 3

def number_of_dozen_eggs_for_pie (x : Nat) : Prop :=
  x = 4

def number_of_dozen_eggs_to_charity (x : Nat) : Prop :=
  x = 4

-- The theorem stating the answer to the problem
theorem number_of_dozen_eggs_to_mall 
  (h1 : ∃ x, number_of_dozen_eggs_collected x)
  (h2 : ∃ x, number_of_dozen_eggs_to_market x)
  (h3 : ∃ x, number_of_dozen_eggs_for_pie x)
  (h4 : ∃ x, number_of_dozen_eggs_to_charity x)
  : ∃ z, z = 5 := 
sorry

end NUMINAMATH_GPT_number_of_dozen_eggs_to_mall_l2264_226472


namespace NUMINAMATH_GPT_fraction_power_equals_l2264_226476

theorem fraction_power_equals :
  (5 / 7) ^ 7 = (78125 : ℚ) / 823543 := 
by
  sorry

end NUMINAMATH_GPT_fraction_power_equals_l2264_226476


namespace NUMINAMATH_GPT_range_of_k_l2264_226436

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) → 0 < k ∧ k < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l2264_226436


namespace NUMINAMATH_GPT_largest_integer_is_59_l2264_226479

theorem largest_integer_is_59 
  {w x y z : ℤ} 
  (h₁ : (w + x + y) / 3 = 32)
  (h₂ : (w + x + z) / 3 = 39)
  (h₃ : (w + y + z) / 3 = 40)
  (h₄ : (x + y + z) / 3 = 44) :
  max (max w x) (max y z) = 59 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_integer_is_59_l2264_226479


namespace NUMINAMATH_GPT_flower_bed_dimensions_l2264_226474

variable (l w : ℕ)

theorem flower_bed_dimensions :
  (l + 3) * (w + 2) = l * w + 64 →
  (l + 2) * (w + 3) = l * w + 68 →
  l = 14 ∧ w = 10 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_flower_bed_dimensions_l2264_226474


namespace NUMINAMATH_GPT_nova_monthly_donation_l2264_226494

def total_annual_donation : ℕ := 20484
def months_in_year : ℕ := 12
def monthly_donation : ℕ := total_annual_donation / months_in_year

theorem nova_monthly_donation :
  monthly_donation = 1707 :=
by
  unfold monthly_donation
  sorry

end NUMINAMATH_GPT_nova_monthly_donation_l2264_226494


namespace NUMINAMATH_GPT_roots_of_cubic_equation_l2264_226457

theorem roots_of_cubic_equation 
  (k m : ℝ) 
  (h : ∀r1 r2 r3: ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 + r2 + r3 = 7 ∧ r1 * r2 * r3 = m ∧ (r1 * r2 + r2 * r3 + r1 * r3) = k) : 
  k + m = 22 := sorry

end NUMINAMATH_GPT_roots_of_cubic_equation_l2264_226457


namespace NUMINAMATH_GPT_no_solution_exists_l2264_226450

theorem no_solution_exists : 
  ¬ ∃ (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0), 
    45 * x = (35 / 100) * 900 ∧
    y^2 + x = 100 ∧
    z = x^3 * y - (2 * x + 1) / (y + 4) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l2264_226450


namespace NUMINAMATH_GPT_box_dimensions_sum_l2264_226489

theorem box_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 30) 
  (h2 : A * C = 50)
  (h3 : B * C = 90) : 
  A + B + C = (58 * Real.sqrt 15) / 3 :=
sorry

end NUMINAMATH_GPT_box_dimensions_sum_l2264_226489


namespace NUMINAMATH_GPT_square_perimeter_l2264_226469

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by
  sorry

end NUMINAMATH_GPT_square_perimeter_l2264_226469


namespace NUMINAMATH_GPT_staircase_tile_cover_possible_l2264_226430
-- Import the necessary Lean Lean libraries

-- We use natural numbers here
open Nat

-- Declare the problem as a theorem in Lean
theorem staircase_tile_cover_possible (m n : ℕ) (h_m : 6 ≤ m) (h_n : 6 ≤ n) :
  (∃ a b, m = 12 * a ∧ n = b ∧ a ≥ 1 ∧ b ≥ 6) ∨ 
  (∃ c d, m = 3 * c ∧ n = 4 * d ∧ c ≥ 2 ∧ d ≥ 3) :=
sorry

end NUMINAMATH_GPT_staircase_tile_cover_possible_l2264_226430


namespace NUMINAMATH_GPT_students_taking_all_three_l2264_226468

-- Definitions and Conditions
def total_students : ℕ := 25
def coding_students : ℕ := 12
def chess_students : ℕ := 15
def photography_students : ℕ := 10
def at_least_two_classes : ℕ := 10

-- Request to prove: Number of students taking all three classes
theorem students_taking_all_three (x y w z : ℕ) :
  (x + y + z + w = 10) →
  (coding_students - (10 - y) + chess_students - (10 - w) + (10 - x) = 21) →
  z = 4 :=
by
  intros
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_students_taking_all_three_l2264_226468


namespace NUMINAMATH_GPT_monotonic_intervals_of_f_f_gt_x_ln_x_plus_1_l2264_226423

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / x

theorem monotonic_intervals_of_f :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂) ∧ (∀ x₁ x₂ : ℝ, x₁ > x₂ → f x₁ ≥ f x₂) :=
sorry

theorem f_gt_x_ln_x_plus_1 (x : ℝ) (hx : x > 0) : f x > x * Real.log (x + 1) :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_of_f_f_gt_x_ln_x_plus_1_l2264_226423


namespace NUMINAMATH_GPT_evaluate_expression_l2264_226486

-- Defining the conditions for the cosine and sine values
def cos_0 : Real := 1
def sin_3pi_2 : Real := -1

-- Proving the given expression equals -1
theorem evaluate_expression : 3 * cos_0 + 4 * sin_3pi_2 = -1 :=
by 
  -- Given the definitions, this will simplify as expected.
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2264_226486


namespace NUMINAMATH_GPT_intersection_of_A_and_B_when_a_is_2_range_of_a_such_that_B_subset_A_l2264_226412

-- Definitions for the sets A and B
def setA (a : ℝ) : Set ℝ := { x | (x - 2) * (x - (3 * a + 1)) < 0 }
def setB (a : ℝ) : Set ℝ := { x | (x - 2 * a) / (x - (a ^ 2 + 1)) < 0 }

-- Theorem for question (1): Intersection of A and B when a = 2
theorem intersection_of_A_and_B_when_a_is_2 :
  setA 2 ∩ setB 2 = { x | 4 < x ∧ x < 5 } :=
sorry

-- Theorem for question (2): Range of a such that B ⊆ A
theorem range_of_a_such_that_B_subset_A :
  { a : ℝ | setB a ⊆ setA a } = { x | 1 < x ∧ x ≤ 3 } ∪ { -1 } :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_when_a_is_2_range_of_a_such_that_B_subset_A_l2264_226412


namespace NUMINAMATH_GPT_distance_between_towns_in_kilometers_l2264_226454

theorem distance_between_towns_in_kilometers :
  (20 * 5) * 1.60934 = 160.934 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_towns_in_kilometers_l2264_226454


namespace NUMINAMATH_GPT_greatest_possible_integer_l2264_226422

theorem greatest_possible_integer (n k l : ℕ) (h1 : n < 150) (h2 : n = 11 * k - 1) (h3 : n = 9 * l + 2) : n = 65 :=
by sorry

end NUMINAMATH_GPT_greatest_possible_integer_l2264_226422


namespace NUMINAMATH_GPT_area_of_triangle_PQR_l2264_226495

structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 2 }
def Q : Point := { x := 7, y := 2 }
def R : Point := { x := 5, y := 9 }

noncomputable def triangleArea (A B C : Point) : ℝ :=
  (1 / 2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem area_of_triangle_PQR : triangleArea P Q R = 17.5 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_PQR_l2264_226495


namespace NUMINAMATH_GPT_range_of_a_l2264_226407

def P (x : ℝ) : Prop := x^2 - 4 * x - 5 < 0
def Q (x : ℝ) (a : ℝ) : Prop := x < a

theorem range_of_a (a : ℝ) : (∀ x, P x → Q x a) → (∀ x, Q x a → P x) → a ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2264_226407


namespace NUMINAMATH_GPT_simplify_expression_l2264_226416

variable (y : ℝ)

theorem simplify_expression : (5 * y + 6 * y + 7 * y + 2) = (18 * y + 2) := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2264_226416


namespace NUMINAMATH_GPT_sum_of_remainders_l2264_226446

theorem sum_of_remainders (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 5) 
  (h3 : c % 30 = 20) : 
  (a + b + c) % 30 = 10 := 
by sorry

end NUMINAMATH_GPT_sum_of_remainders_l2264_226446


namespace NUMINAMATH_GPT_total_distance_hiked_east_l2264_226461

-- Define Annika's constant rate of hiking
def constant_rate : ℝ := 10 -- minutes per kilometer

-- Define already hiked distance
def distance_hiked : ℝ := 2.75 -- kilometers

-- Define total available time to return
def total_time : ℝ := 45 -- minutes

-- Prove that the total distance hiked east is 4.5 kilometers
theorem total_distance_hiked_east : distance_hiked + (total_time - distance_hiked * constant_rate) / constant_rate = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_hiked_east_l2264_226461


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_for_x2_ne_y2_l2264_226488

theorem necessary_and_sufficient_condition_for_x2_ne_y2 (x y : ℤ) :
  (x ^ 2 ≠ y ^ 2) ↔ (x ≠ y ∧ x ≠ -y) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_for_x2_ne_y2_l2264_226488


namespace NUMINAMATH_GPT_dan_job_time_l2264_226462

theorem dan_job_time
  (Annie_time : ℝ) (Dan_work_time : ℝ) (Annie_work_remain : ℝ) (total_work : ℝ)
  (Annie_time_cond : Annie_time = 9)
  (Dan_work_time_cond : Dan_work_time = 8)
  (Annie_work_remain_cond : Annie_work_remain = 3.0000000000000004)
  (total_work_cond : total_work = 1) :
  ∃ (Dan_time : ℝ), Dan_time = 12 := by
  sorry

end NUMINAMATH_GPT_dan_job_time_l2264_226462


namespace NUMINAMATH_GPT_find_f_inv_64_l2264_226403

noncomputable def f : ℝ → ℝ :=
  sorry  -- We don't know the exact form of f.

axiom f_property_1 : f 5 = 2

axiom f_property_2 : ∀ x : ℝ, f (2 * x) = 2 * f x

def f_inv (y : ℝ) : ℝ :=
  sorry  -- We define the inverse function in terms of y.

theorem find_f_inv_64 : f_inv 64 = 160 :=
by {
  -- Main statement to be proved.
  sorry
}

end NUMINAMATH_GPT_find_f_inv_64_l2264_226403


namespace NUMINAMATH_GPT_words_per_page_l2264_226470

theorem words_per_page (p : ℕ) (hp : p ≤ 120) (h : 150 * p ≡ 210 [MOD 221]) : p = 98 := by
  sorry

end NUMINAMATH_GPT_words_per_page_l2264_226470


namespace NUMINAMATH_GPT_min_value_x_add_one_div_y_l2264_226481

theorem min_value_x_add_one_div_y (x y : ℝ) (h1 : x > 1) (h2 : x - y = 1) : 
x + 1 / y ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_x_add_one_div_y_l2264_226481


namespace NUMINAMATH_GPT_passengers_final_count_l2264_226439

structure BusStop :=
  (initial_passengers : ℕ)
  (first_stop_increase : ℕ)
  (other_stops_decrease : ℕ)
  (other_stops_increase : ℕ)

def passengers_at_last_stop (b : BusStop) : ℕ :=
  b.initial_passengers + b.first_stop_increase - b.other_stops_decrease + b.other_stops_increase

theorem passengers_final_count :
  passengers_at_last_stop ⟨50, 16, 22, 5⟩ = 49 := by
  rfl

end NUMINAMATH_GPT_passengers_final_count_l2264_226439


namespace NUMINAMATH_GPT_cos_sin_exp_l2264_226493

theorem cos_sin_exp (n : ℕ) (t : ℝ) (h : n ≤ 1000) :
  (Complex.exp (t * Complex.I)) ^ n = Complex.exp (n * t * Complex.I) :=
by
  sorry

end NUMINAMATH_GPT_cos_sin_exp_l2264_226493


namespace NUMINAMATH_GPT_bird_families_left_l2264_226460

theorem bird_families_left (B_initial B_flew_away : ℕ) (h_initial : B_initial = 41) (h_flew_away : B_flew_away = 27) :
  B_initial - B_flew_away = 14 :=
by
  sorry

end NUMINAMATH_GPT_bird_families_left_l2264_226460


namespace NUMINAMATH_GPT_speed_ratio_l2264_226405

-- Definition of speeds
def B_speed : ℚ := 1 / 12
def combined_speed : ℚ := 1 / 4

-- The theorem statement to be proven
theorem speed_ratio (A_speed B_speed combined_speed : ℚ) (h1 : B_speed = 1 / 12) (h2 : combined_speed = 1 / 4) (h3 : A_speed + B_speed = combined_speed) :
  A_speed / B_speed = 2 :=
by
  sorry

end NUMINAMATH_GPT_speed_ratio_l2264_226405


namespace NUMINAMATH_GPT_exterior_angle_DEF_l2264_226435

theorem exterior_angle_DEF :
  let heptagon_angle := (180 * (7 - 2)) / 7
  let octagon_angle := (180 * (8 - 2)) / 8
  let total_degrees := 360
  total_degrees - (heptagon_angle + octagon_angle) = 96.43 :=
by
  sorry

end NUMINAMATH_GPT_exterior_angle_DEF_l2264_226435


namespace NUMINAMATH_GPT_sarah_cupcakes_l2264_226471

theorem sarah_cupcakes (c k d : ℕ) (h1 : c + k = 6) (h2 : 90 * c + 40 * k = 100 * d) : c = 4 ∨ c = 6 :=
by {
  sorry -- Proof is omitted as requested.
}

end NUMINAMATH_GPT_sarah_cupcakes_l2264_226471


namespace NUMINAMATH_GPT_number_of_testing_methods_l2264_226463

-- Definitions based on conditions
def num_genuine_items : ℕ := 6
def num_defective_items : ℕ := 4
def total_tests : ℕ := 5

-- Theorem stating the number of testing methods
theorem number_of_testing_methods 
    (h1 : total_tests = 5) 
    (h2 : num_genuine_items = 6) 
    (h3 : num_defective_items = 4) :
    ∃ n : ℕ, n = 576 := 
sorry

end NUMINAMATH_GPT_number_of_testing_methods_l2264_226463


namespace NUMINAMATH_GPT_a_work_days_alone_l2264_226449

-- Definitions based on conditions
def work_days_a   (a: ℝ)    : Prop := ∃ (x:ℝ), a = x
def work_days_b   (b: ℝ)    : Prop := b = 36
def alternate_work (a b W x: ℝ) : Prop := 9 * (W / 36 + W / x) = W ∧ x > 0

-- The main theorem to prove
theorem a_work_days_alone (x W: ℝ) (b: ℝ) (h_work_days_b: work_days_b b)
                          (h_alternate_work: alternate_work a b W x) : 
                          work_days_a a → a = 12 :=
by sorry

end NUMINAMATH_GPT_a_work_days_alone_l2264_226449


namespace NUMINAMATH_GPT_manager_final_price_l2264_226400

noncomputable def wholesale_cost : ℝ := 200
noncomputable def retail_price : ℝ := wholesale_cost + 0.2 * wholesale_cost
noncomputable def manager_discount : ℝ := 0.1 * retail_price
noncomputable def price_after_manager_discount : ℝ := retail_price - manager_discount
noncomputable def weekend_sale_discount : ℝ := 0.1 * price_after_manager_discount
noncomputable def price_after_weekend_sale : ℝ := price_after_manager_discount - weekend_sale_discount
noncomputable def sales_tax : ℝ := 0.08 * price_after_weekend_sale
noncomputable def total_price : ℝ := price_after_weekend_sale + sales_tax

theorem manager_final_price : total_price = 209.95 := by
  sorry

end NUMINAMATH_GPT_manager_final_price_l2264_226400


namespace NUMINAMATH_GPT_range_of_a_l2264_226477

noncomputable def f (x a : ℝ) : ℝ := x * abs (x - a)

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), 3 ≤ x1 ∧ 3 ≤ x2 ∧ x1 ≠ x2 → (x1 - x2) * (f x1 a - f x2 a) > 0) → a ≤ 3 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l2264_226477


namespace NUMINAMATH_GPT_number_exceeds_percent_l2264_226424

theorem number_exceeds_percent (x : ℝ) (h : x = 0.12 * x + 52.8) : x = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_exceeds_percent_l2264_226424


namespace NUMINAMATH_GPT_ratio_population_X_to_Z_l2264_226443

-- Given definitions
def population_of_Z : ℕ := sorry
def population_of_Y : ℕ := 2 * population_of_Z
def population_of_X : ℕ := 5 * population_of_Y

-- Theorem to prove
theorem ratio_population_X_to_Z : population_of_X / population_of_Z = 10 :=
by
  sorry

end NUMINAMATH_GPT_ratio_population_X_to_Z_l2264_226443


namespace NUMINAMATH_GPT_laboratory_spent_on_flasks_l2264_226484

theorem laboratory_spent_on_flasks:
  ∀ (F : ℝ), (∃ cost_test_tubes : ℝ, cost_test_tubes = (2 / 3) * F) →
  (∃ cost_safety_gear : ℝ, cost_safety_gear = (1 / 3) * F) →
  2 * F = 300 → F = 150 :=
by
  intros F h1 h2 h3
  sorry

end NUMINAMATH_GPT_laboratory_spent_on_flasks_l2264_226484
