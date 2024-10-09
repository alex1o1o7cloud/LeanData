import Mathlib

namespace unique_solution_l644_64473

def unique_ordered_pair : Prop :=
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ 
               (∃ x : ℝ, x = (m : ℝ)^(1/3) - (n : ℝ)^(1/3) ∧ x^6 + 4 * x^3 - 36 * x^2 + 4 = 0) ∧
               m = 2 ∧ n = 4

theorem unique_solution : unique_ordered_pair := sorry

end unique_solution_l644_64473


namespace division_simplification_l644_64434

theorem division_simplification : 180 / (12 + 13 * 3) = 60 / 17 := by
  sorry

end division_simplification_l644_64434


namespace sail_pressure_l644_64445

theorem sail_pressure (k : ℝ) :
  (forall (V A : ℝ), P = k * A * (V : ℝ)^2) 
  → (P = 1.25) → (V = 20) → (A = 1)
  → (A = 4) → (V = 40)
  → (P = 20) :=
by
  sorry

end sail_pressure_l644_64445


namespace hanna_has_money_l644_64439

variable (total_roses money_spent : ℕ)
variable (rose_price : ℕ := 2)

def hanna_gives_roses (total_roses : ℕ) : Bool :=
  (1 / 3 * total_roses + 1 / 2 * total_roses) = 125

theorem hanna_has_money (H : hanna_gives_roses total_roses) : money_spent = 300 := sorry

end hanna_has_money_l644_64439


namespace pencil_and_eraser_cost_l644_64474

theorem pencil_and_eraser_cost (p e : ℕ) :
  2 * p + e = 40 →
  p > e →
  e ≥ 3 →
  p + e = 22 :=
by
  sorry

end pencil_and_eraser_cost_l644_64474


namespace marbles_count_l644_64415

theorem marbles_count (M : ℕ)
  (h_blue : (M / 2) = n_blue)
  (h_red : (M / 4) = n_red)
  (h_green : 27 = n_green)
  (h_yellow : 14 = n_yellow)
  (h_total : (n_blue + n_red + n_green + n_yellow) = M) :
  M = 164 :=
by
  sorry

end marbles_count_l644_64415


namespace dilation_translation_correct_l644_64457

def transformation_matrix (d: ℝ) (tx: ℝ) (ty: ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![d, 0, tx],
    ![0, d, ty],
    ![0, 0, 1]
  ]

theorem dilation_translation_correct :
  transformation_matrix 4 2 3 =
  ![
    ![4, 0, 2],
    ![0, 4, 3],
    ![0, 0, 1]
  ] :=
by
  sorry

end dilation_translation_correct_l644_64457


namespace factor_ax2_minus_ay2_l644_64431

variable (a x y : ℝ)

theorem factor_ax2_minus_ay2 : a * x^2 - a * y^2 = a * (x + y) * (x - y) := 
sorry

end factor_ax2_minus_ay2_l644_64431


namespace solve_for_y_l644_64404

theorem solve_for_y (y : ℕ) (h : 9 / y^2 = 3 * y / 81) : y = 9 :=
sorry

end solve_for_y_l644_64404


namespace arithmetic_sequence_sum_l644_64444

theorem arithmetic_sequence_sum :
  ∃ a : ℕ → ℤ, 
    a 3 = 7 ∧ a 4 = 11 ∧ a 5 = 15 ∧ 
    (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 54) := 
by {
  sorry
}

end arithmetic_sequence_sum_l644_64444


namespace digit_B_divisible_by_9_l644_64488

theorem digit_B_divisible_by_9 (B : ℕ) (k : ℤ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 2 * B + 6 = 9 * k) : B = 6 := 
by
  sorry

end digit_B_divisible_by_9_l644_64488


namespace find_k_l644_64441

theorem find_k (x k : ℝ) (h : ((x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) ∧ k ≠ 0) : k = 6 :=
by
  sorry

end find_k_l644_64441


namespace value_that_number_exceeds_l644_64467

theorem value_that_number_exceeds (V : ℤ) (h : 69 = V + 3 * (86 - 69)) : V = 18 :=
by
  sorry

end value_that_number_exceeds_l644_64467


namespace paper_thickness_after_folds_l644_64425

def folded_thickness (initial_thickness : ℝ) (folds : ℕ) : ℝ :=
  initial_thickness * 2^folds

theorem paper_thickness_after_folds :
  folded_thickness 0.1 4 = 1.6 :=
by
  sorry

end paper_thickness_after_folds_l644_64425


namespace product_of_fractions_l644_64414

theorem product_of_fractions :
  (1 / 2) * (2 / 3) * (3 / 4) * (3 / 2) = 3 / 8 := by
  sorry

end product_of_fractions_l644_64414


namespace probability_sum_equals_6_l644_64490

theorem probability_sum_equals_6 : 
  let possible_outcomes := 36
  let favorable_outcomes := 5
  (favorable_outcomes / possible_outcomes : ℚ) = 5 / 36 := 
by 
  sorry

end probability_sum_equals_6_l644_64490


namespace coin_collection_problem_l644_64417

variable (n d q : ℚ)

theorem coin_collection_problem 
  (h1 : n + d + q = 30)
  (h2 : 5 * n + 10 * d + 20 * q = 340)
  (h3 : d = 2 * n) :
  q - n = 2 / 7 := by
  sorry

end coin_collection_problem_l644_64417


namespace cubic_sum_identity_l644_64409

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 10) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 100 :=
by sorry

end cubic_sum_identity_l644_64409


namespace initial_necklaces_15_l644_64412

variable (N E : ℕ)
variable (initial_necklaces : ℕ) (initial_earrings : ℕ) (store_necklaces : ℕ) (store_earrings : ℕ) (mother_earrings : ℕ) (total_jewelry : ℕ)

axiom necklaces_eq_initial : N = initial_necklaces
axiom earrings_eq_15 : E = initial_earrings
axiom initial_earrings_15 : initial_earrings = 15
axiom store_necklaces_eq_initial : store_necklaces = initial_necklaces
axiom store_earrings_eq_23_initial : store_earrings = 2 * initial_earrings / 3
axiom mother_earrings_eq_115_store : mother_earrings = 1 * store_earrings / 5
axiom total_jewelry_is_57 : total_jewelry = 57
axiom jewelry_pieces_eq : 2 * initial_necklaces + initial_earrings + store_earrings + mother_earrings = total_jewelry

theorem initial_necklaces_15 : initial_necklaces = 15 := by
  sorry

end initial_necklaces_15_l644_64412


namespace divisible_by_101_l644_64413

theorem divisible_by_101 (n : ℕ) : (101 ∣ (10^n - 1)) ↔ (∃ k : ℕ, n = 4 * k) :=
by
  sorry

end divisible_by_101_l644_64413


namespace evaluate_expression_l644_64440

theorem evaluate_expression : 
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := 
by
  sorry

end evaluate_expression_l644_64440


namespace candies_for_50_rubles_l644_64485

theorem candies_for_50_rubles : 
  ∀ (x : ℕ), (45 * x = 45) → (50 / x = 50) := 
by
  intros x h
  sorry

end candies_for_50_rubles_l644_64485


namespace base_n_not_divisible_by_11_l644_64460

theorem base_n_not_divisible_by_11 :
  ∀ n, 2 ≤ n ∧ n ≤ 100 → (6 + 2*n + 5*n^2 + 4*n^3 + 2*n^4 + 4*n^5) % 11 ≠ 0 := by
  sorry

end base_n_not_divisible_by_11_l644_64460


namespace sqrt_square_of_neg_four_l644_64418

theorem sqrt_square_of_neg_four : Real.sqrt ((-4:Real)^2) = 4 := by
  sorry

end sqrt_square_of_neg_four_l644_64418


namespace probability_at_least_two_same_post_l644_64430

theorem probability_at_least_two_same_post : 
  let volunteers := 3
  let posts := 4
  let total_assignments := posts ^ volunteers
  let different_post_assignments := Nat.factorial posts / (Nat.factorial (posts - volunteers))
  let probability_all_different := different_post_assignments / total_assignments
  let probability_two_same := 1 - probability_all_different
  (1 - (Nat.factorial posts / (total_assignments * Nat.factorial (posts - volunteers)))) = 5 / 8 :=
by
  sorry

end probability_at_least_two_same_post_l644_64430


namespace mother_sold_rings_correct_l644_64424

noncomputable def motherSellsRings (initial_bought_rings mother_bought_rings remaining_rings final_stock : ℤ) : ℤ :=
  let initial_stock := initial_bought_rings / 2
  let total_stock := initial_bought_rings + initial_stock
  let sold_by_eliza := (3 * total_stock) / 4
  let remaining_after_eliza := total_stock - sold_by_eliza
  let new_total_stock := remaining_after_eliza + mother_bought_rings
  new_total_stock - final_stock

theorem mother_sold_rings_correct :
  motherSellsRings 200 300 225 300 = 150 :=
by
  sorry

end mother_sold_rings_correct_l644_64424


namespace find_num_pennies_l644_64420

def total_value (nickels : ℕ) (dimes : ℕ) (pennies : ℕ) : ℕ :=
  5 * nickels + 10 * dimes + pennies

def num_pennies (nickels_value: ℕ) (dimes_value: ℕ) (total: ℕ): ℕ :=
  total - (nickels_value + dimes_value)

theorem find_num_pennies : 
  ∀ (total : ℕ) (num_nickels : ℕ) (num_dimes: ℕ),
  total = 59 → num_nickels = 4 → num_dimes = 3 → num_pennies (5 * num_nickels) (10 * num_dimes) total = 9 :=
by
  intros
  sorry

end find_num_pennies_l644_64420


namespace range_of_m_l644_64446

theorem range_of_m (x m : ℝ) : (|x - 3| ≤ 2) → ((x - m + 1) * (x - m - 1) ≤ 0) → 
  (¬(|x - 3| ≤ 2) → ¬((x - m + 1) * (x - m - 1) ≤ 0)) → 2 ≤ m ∧ m ≤ 4 :=
by
  intro h1 h2 h3
  sorry

end range_of_m_l644_64446


namespace arithmetic_seq_max_S_l644_64491

theorem arithmetic_seq_max_S {S : ℕ → ℝ} (h1 : S 2023 > 0) (h2 : S 2024 < 0) : S 1012 > S 1013 :=
sorry

end arithmetic_seq_max_S_l644_64491


namespace combined_resistance_parallel_l644_64458

theorem combined_resistance_parallel (R1 R2 : ℝ) (r : ℝ) 
  (hR1 : R1 = 8) (hR2 : R2 = 9) (h_parallel : (1 / r) = (1 / R1) + (1 / R2)) : 
  r = 72 / 17 :=
by
  sorry

end combined_resistance_parallel_l644_64458


namespace number_is_correct_l644_64421

theorem number_is_correct (x : ℝ) (h : (30 / 100) * x = (25 / 100) * 40) : x = 100 / 3 :=
by
  -- Proof would go here.
  sorry

end number_is_correct_l644_64421


namespace array_element_count_l644_64433

theorem array_element_count (A : Finset ℕ) 
  (h1 : ∀ n ∈ A, n ≠ 1 → (∃ a ∈ [2, 3, 5], a ∣ n)) 
  (h2 : ∀ n ∈ A, (2 * n ∈ A ∨ 3 * n ∈ A ∨ 5 * n ∈ A) ↔ (n ∈ A ∧ 2 * n ∈ A ∧ 3 * n ∈ A ∧ 5 * n ∈ A)) 
  (card_A_range : 300 ≤ A.card ∧ A.card ≤ 400) : 
  A.card = 364 := 
sorry

end array_element_count_l644_64433


namespace more_ones_than_twos_in_digital_roots_l644_64492

/-- Define the digital root (i.e., repeated sum of digits until a single digit). -/
def digitalRoot (n : Nat) : Nat :=
  if n == 0 then 0 else 1 + (n - 1) % 9

/-- Statement of the problem: For numbers 1 to 1,000,000, the count of digital root 1 is higher than the count of digital root 2. -/
theorem more_ones_than_twos_in_digital_roots :
  (Finset.filter (fun n => digitalRoot n = 1) (Finset.range 1000000)).card >
  (Finset.filter (fun n => digitalRoot n = 2) (Finset.range 1000000)).card :=
by
  sorry

end more_ones_than_twos_in_digital_roots_l644_64492


namespace find_certain_value_l644_64448

noncomputable def certain_value 
  (total_area : ℝ) (smaller_part : ℝ) (difference_fraction : ℝ) : ℝ :=
  (total_area - 2 * smaller_part) / difference_fraction

theorem find_certain_value (total_area : ℝ) (smaller_part : ℝ) (X : ℝ) : 
  total_area = 700 → 
  smaller_part = 315 → 
  (total_area - 2 * smaller_part) / (1/5) = X → 
  X = 350 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end find_certain_value_l644_64448


namespace kyunghwan_spent_the_most_l644_64401

-- Define initial pocket money for everyone
def initial_money : ℕ := 20000

-- Define remaining money
def remaining_S : ℕ := initial_money / 4
def remaining_K : ℕ := initial_money / 8
def remaining_D : ℕ := initial_money / 5

-- Calculate spent money
def spent_S : ℕ := initial_money - remaining_S
def spent_K : ℕ := initial_money - remaining_K
def spent_D : ℕ := initial_money - remaining_D

theorem kyunghwan_spent_the_most 
  (h1 : remaining_S = initial_money / 4)
  (h2 : remaining_K = initial_money / 8)
  (h3 : remaining_D = initial_money / 5) :
  spent_K > spent_S ∧ spent_K > spent_D :=
by
  -- Proof skipped
  sorry

end kyunghwan_spent_the_most_l644_64401


namespace x_plus_q_eq_five_l644_64442

theorem x_plus_q_eq_five (x q : ℝ) (h1 : abs (x - 5) = q) (h2 : x < 5) : x + q = 5 :=
by
  sorry

end x_plus_q_eq_five_l644_64442


namespace find_13th_result_l644_64402

theorem find_13th_result (avg25 : ℕ) (avg12_first : ℕ) (avg12_last : ℕ)
  (h_avg25 : avg25 = 18) (h_avg12_first : avg12_first = 10) (h_avg12_last : avg12_last = 20) :
  ∃ r13 : ℕ, r13 = 90 := by
  sorry

end find_13th_result_l644_64402


namespace geom_prog_roots_a_eq_22_l644_64483

theorem geom_prog_roots_a_eq_22 (x1 x2 x3 a : ℝ) :
  (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) → 
  (∃ b q, (x1 = b ∧ x2 = b * q ∧ x3 = b * q^2) ∧ (x1 + x2 + x3 = 11) ∧ (x1 * x2 * x3 = 8) ∧ (x1*x2 + x2*x3 + x3*x1 = a)) → 
  a = 22 :=
sorry

end geom_prog_roots_a_eq_22_l644_64483


namespace sum_first_9_terms_l644_64419

-- Definitions of the arithmetic sequence and sum.
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Conditions
def a_n (n : ℕ) : ℤ := sorry -- we assume this function gives the n-th term of the arithmetic sequence
def S_n (n : ℕ) : ℤ := sorry -- sum of first n terms
axiom a_5_eq_2 : a_n 5 = 2
axiom arithmetic_sequence_proof : arithmetic_sequence a_n
axiom sum_first_n_proof : sum_first_n a_n S_n

-- Statement to prove
theorem sum_first_9_terms : S_n 9 = 18 :=
by
  sorry

end sum_first_9_terms_l644_64419


namespace series_convergence_p_geq_2_l644_64416

noncomputable def ai_series_converges (a : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, ∑' i, a i ^ 2 = l

noncomputable def bi_series_converges (b : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, ∑' i, b i ^ 2 = l

theorem series_convergence_p_geq_2 
  (a b : ℕ → ℝ) 
  (h₁ : ai_series_converges a)
  (h₂ : bi_series_converges b) 
  (p : ℝ) (hp : p ≥ 2) : 
  ∃ l : ℝ, ∑' i, |a i - b i| ^ p = l := 
sorry

end series_convergence_p_geq_2_l644_64416


namespace smallest_n_l644_64452

theorem smallest_n(vc: ℕ) (n: ℕ) : 
    (vc = 25) ∧ ∃ y o i : ℕ, ((25 * n = 10 * y) ∨ (25 * n = 18 * o) ∨ (25 * n = 20 * i)) → 
    n = 16 := by
    -- We state that given conditions should imply n = 16.
    sorry

end smallest_n_l644_64452


namespace period_change_l644_64472

theorem period_change {f : ℝ → ℝ} (T : ℝ) (hT : 0 < T) (h_period : ∀ x, f (x + T) = f x) (α : ℝ) (hα : 0 < α) :
  ∀ x, f (α * (x + T / α)) = f (α * x) :=
by
  sorry

end period_change_l644_64472


namespace num_senior_in_sample_l644_64494

-- Definitions based on conditions
def total_students : ℕ := 2000
def senior_students : ℕ := 700
def sample_size : ℕ := 400

-- Theorem statement for the number of senior students in the sample
theorem num_senior_in_sample : 
  (senior_students * sample_size) / total_students = 140 :=
by 
  sorry

end num_senior_in_sample_l644_64494


namespace evaluate_expression_l644_64403

theorem evaluate_expression : 3^5 * 6^5 * 3^6 * 6^6 = 18^11 :=
by sorry

end evaluate_expression_l644_64403


namespace roots_of_quadratic_serve_as_eccentricities_l644_64456

theorem roots_of_quadratic_serve_as_eccentricities :
  ∀ (x1 x2 : ℝ), x1 * x2 = 1 ∧ x1 + x2 = 79 → (x1 > 1 ∧ x2 < 1) → 
  (x1 > 1 ∧ x2 < 1) ∧ x1 > 1 ∧ x2 < 1 :=
by
  sorry

end roots_of_quadratic_serve_as_eccentricities_l644_64456


namespace A_intersect_B_eq_l644_64468

def A (x : ℝ) : Prop := x > 0
def B (x : ℝ) : Prop := x ≤ 1
def A_cap_B (x : ℝ) : Prop := x ∈ {y | A y} ∧ x ∈ {y | B y}

theorem A_intersect_B_eq (x : ℝ) : (A_cap_B x) ↔ (x ∈ Set.Ioc 0 1) :=
by
  sorry

end A_intersect_B_eq_l644_64468


namespace sum_abcd_l644_64436

theorem sum_abcd (a b c d : ℤ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 8) : 
  a + b + c + d = -6 :=
sorry

end sum_abcd_l644_64436


namespace sum_of_solutions_l644_64475

theorem sum_of_solutions (a b : ℝ) (h1 : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 := 
by {
  -- missing proof part
  sorry
}

end sum_of_solutions_l644_64475


namespace perp_bisector_of_AB_l644_64478

noncomputable def perpendicular_bisector_eq : Prop :=
  ∀ (x y : ℝ), (x - y + 1 = 0) ∧ (x^2 + y^2 = 1) → (x + y = 0)

-- The proof is omitted
theorem perp_bisector_of_AB : perpendicular_bisector_eq :=
sorry

end perp_bisector_of_AB_l644_64478


namespace sum_of_six_angles_l644_64437

theorem sum_of_six_angles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 + angle3 + angle5 = 180)
  (h2 : angle2 + angle4 + angle6 = 180) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 360 :=
by
  sorry

end sum_of_six_angles_l644_64437


namespace analytical_expression_satisfies_conditions_l644_64423

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := 1 + Real.exp x

theorem analytical_expression_satisfies_conditions :
  is_increasing f ∧ (∀ x : ℝ, f x > 1) :=
by
  sorry

end analytical_expression_satisfies_conditions_l644_64423


namespace roses_after_trading_equals_36_l644_64453

-- Definitions of the given conditions
def initial_roses_given : ℕ := 24
def roses_after_trade (n : ℕ) : ℕ := n
def remaining_roses_after_first_wilt (roses : ℕ) : ℕ := roses / 2
def remaining_roses_after_second_wilt (roses : ℕ) : ℕ := roses / 2
def roses_remaining_second_day : ℕ := 9

-- The statement we want to prove
theorem roses_after_trading_equals_36 (n : ℕ) (h : roses_remaining_second_day = 9) :
  ( ∃ x, roses_after_trade x = n ∧ remaining_roses_after_first_wilt (remaining_roses_after_first_wilt x) = roses_remaining_second_day ) →
  n = 36 :=
by
  sorry

end roses_after_trading_equals_36_l644_64453


namespace union_A_B_comp_U_A_inter_B_range_of_a_l644_64432

namespace ProofProblem

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 < x ∧ x < 6 }
def C (a : ℝ) : Set ℝ := { x | x > a }
def U : Set ℝ := Set.univ

theorem union_A_B : A ∪ B = { x | 1 < x ∧ x ≤ 8 } := by
  sorry

theorem comp_U_A_inter_B : (U \ A) ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by
  sorry

end ProofProblem

end union_A_B_comp_U_A_inter_B_range_of_a_l644_64432


namespace player_B_wins_l644_64427

variable {R : Type*} [Ring R]

noncomputable def polynomial_game (n : ℕ) (f : Polynomial R) : Prop :=
  (f.degree = 2 * n) ∧ (∃ (a b : R) (x y : R), f.eval x = 0 ∨ f.eval y = 0)

theorem player_B_wins (n : ℕ) (f : Polynomial ℝ)
  (h1 : n ≥ 2)
  (h2 : f.degree = 2 * n) :
  polynomial_game n f :=
by
  sorry

end player_B_wins_l644_64427


namespace value_of_expression_l644_64450

theorem value_of_expression (x : ℝ) (h : x ^ 2 - 3 * x + 1 = 0) : 
  x ≠ 0 → (x ^ 2) / (x ^ 4 + x ^ 2 + 1) = 1 / 8 :=
by 
  intros h1 
  sorry

end value_of_expression_l644_64450


namespace total_pupils_count_l644_64447

theorem total_pupils_count (girls boys : ℕ) (h1 : girls = 692) (h2 : girls = boys + 458) : girls + boys = 926 :=
by 
  sorry

end total_pupils_count_l644_64447


namespace conclusion1_conclusion2_conclusion3_l644_64454

-- Define the Δ operation
def delta (m n : ℚ) : ℚ := (m + n) / (1 + m * n)

-- 1. Proof that (-2^2) Δ 4 = 0
theorem conclusion1 : delta (-4) 4 = 0 := sorry

-- 2. Proof that (1/3) Δ (1/4) = 3 Δ 4
theorem conclusion2 : delta (1/3) (1/4) = delta 3 4 := sorry

-- 3. Proof that (-m) Δ n = m Δ (-n)
theorem conclusion3 (m n : ℚ) : delta (-m) n = delta m (-n) := sorry

end conclusion1_conclusion2_conclusion3_l644_64454


namespace part1_real_values_part2_imaginary_values_l644_64435

namespace ComplexNumberProblem

-- Definitions of conditions for part 1
def imaginaryZero (x : ℝ) : Prop :=
  x^2 + 3*x + 2 = 0

def realPositive (x : ℝ) : Prop :=
  x^2 - 2*x - 2 > 0

-- Definition of question for part 1
def realValues (x : ℝ) : Prop :=
  x = -1 ∨ x = -2

-- Proof problem for part 1
theorem part1_real_values (x : ℝ) (h1 : imaginaryZero x) (h2 : realPositive x) : realValues x :=
by
  have h : realValues x := sorry
  exact h

-- Definitions of conditions for part 2
def realPartOne (x : ℝ) : Prop :=
  x^2 - 2*x - 2 = 1

def imaginaryNonZero (x : ℝ) : Prop :=
  x^2 + 3*x + 2 ≠ 0

-- Definition of question for part 2
def imaginaryValues (x : ℝ) : Prop :=
  x = 3

-- Proof problem for part 2
theorem part2_imaginary_values (x : ℝ) (h1 : realPartOne x) (h2 : imaginaryNonZero x) : imaginaryValues x :=
by
  have h : imaginaryValues x := sorry
  exact h

end ComplexNumberProblem

end part1_real_values_part2_imaginary_values_l644_64435


namespace fraction_spent_is_one_third_l644_64461

-- Define the initial conditions and money variables
def initial_money := 32
def cost_bread := 3
def cost_candy := 2
def remaining_money_after_all := 18

-- Define the calculation for the money left after buying bread and candy bar
def money_left_after_bread_candy := initial_money - cost_bread - cost_candy

-- Define the calculation for the money spent on turkey
def money_spent_on_turkey := money_left_after_bread_candy - remaining_money_after_all

-- The fraction of the remaining money spent on the Turkey
noncomputable def fraction_spent_on_turkey := (money_spent_on_turkey : ℚ) / money_left_after_bread_candy

-- State the theorem that verifies the fraction spent on turkey is 1/3
theorem fraction_spent_is_one_third : fraction_spent_on_turkey = 1 / 3 := by
  sorry

end fraction_spent_is_one_third_l644_64461


namespace gauss_algorithm_sum_l644_64498

def f (x : Nat) (m : Nat) : Rat := x / (3 * m + 6054)

theorem gauss_algorithm_sum (m : Nat) :
  (Finset.sum (Finset.range (m + 2017 + 1)) (λ x => f x m)) = (m + 2017) / 6 := by
sorry

end gauss_algorithm_sum_l644_64498


namespace quadrilateral_angles_l644_64499

theorem quadrilateral_angles 
  (A B C D : Type) 
  (a d b c : Float)
  (hAD : a = d ∧ d = c) 
  (hBDC_twice_BDA : ∃ x : Float, b = 2 * x) 
  (hBDA_CAD_ratio : ∃ x : Float, d = 2/3 * x) :
  (∃ α β γ δ : Float, 
    α = 75 ∧ 
    β = 135 ∧ 
    γ = 60 ∧ 
    δ = 90) := 
sorry

end quadrilateral_angles_l644_64499


namespace cos_neg_30_eq_sqrt_3_div_2_l644_64426

theorem cos_neg_30_eq_sqrt_3_div_2 : 
  Real.cos (-30 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_neg_30_eq_sqrt_3_div_2_l644_64426


namespace other_divisor_l644_64465

theorem other_divisor (x : ℕ) (h1 : 261 % 37 = 2) (h2 : 261 % x = 2) (h3 : 259 = 261 - 2) :
  ∃ x : ℕ, 259 % 37 = 0 ∧ 259 % x = 0 ∧ x = 7 :=
by
  sorry

end other_divisor_l644_64465


namespace power_function_general_form_l644_64486

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_general_form (α : ℝ) :
  ∃ y : ℝ, ∃ α : ℝ, f 3 α = y ∧ ∀ x : ℝ, f x α = x ^ α :=
by
  sorry

end power_function_general_form_l644_64486


namespace find_x1_l644_64469

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3) (h2 : x3 ≤ x2) (h3 : x2 ≤ x1) (h4 : x1 ≤ 1) 
    (h5 : (1 - x1)^3 + (x1 - x2)^3 + (x2 - x3)^3 + x3^3 = 1 / 8) : x1 = 3 / 4 := 
by 
  sorry

end find_x1_l644_64469


namespace range_of_a_l644_64466

noncomputable def p (x : ℝ) : Prop := (1 / (x - 3)) ≥ 1

noncomputable def q (x a : ℝ) : Prop := abs (x - a) < 1

theorem range_of_a (a : ℝ) : (∀ x, p x → q x a) ∧ (∃ x, ¬ (p x) ∧ (q x a)) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l644_64466


namespace arrange_abc_l644_64422

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom cos_a_eq_a : Real.cos a = a
axiom sin_cos_b_eq_b : Real.sin (Real.cos b) = b
axiom cos_sin_c_eq_c : Real.cos (Real.sin c) = c

theorem arrange_abc : b < a ∧ a < c := 
by
  sorry

end arrange_abc_l644_64422


namespace monomial_same_type_l644_64406

-- Define a structure for monomials
structure Monomial where
  coeff : ℕ
  vars : List String

-- Monomials definitions based on the given conditions
def m1 := Monomial.mk 3 ["a"]
def m2 := Monomial.mk 2 ["b"]
def m3 := Monomial.mk 1 ["a", "b"]
def m4 := Monomial.mk 3 ["a", "c"]
def target := Monomial.mk 2 ["a", "b"]

-- Define a predicate to check if two monomials are of the same type
def sameType (m n : Monomial) : Prop :=
  m.vars = n.vars

theorem monomial_same_type :
  sameType m3 target := sorry

end monomial_same_type_l644_64406


namespace sufficient_but_not_necessary_l644_64471

-- Define the quadratic function
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + 2*x + m

-- The problem statement to prove that "m < 1" is a sufficient condition
-- but not a necessary condition for the function f(x) to have a root.
theorem sufficient_but_not_necessary (m : ℝ) :
  (m < 1 → ∃ x : ℝ, f x m = 0) ∧ ¬(¬(m < 1) → ∃ x : ℝ, f x m = 0) :=
sorry

end sufficient_but_not_necessary_l644_64471


namespace inequality_always_holds_l644_64464

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 :=
sorry

end inequality_always_holds_l644_64464


namespace alice_acorns_purchase_l644_64481

variable (bob_payment : ℕ) (alice_payment_rate : ℕ) (price_per_acorn : ℕ)

-- Given conditions
def bob_paid : Prop := bob_payment = 6000
def alice_paid : Prop := alice_payment_rate = 9
def acorn_price : Prop := price_per_acorn = 15

-- Proof statement
theorem alice_acorns_purchase
  (h1 : bob_paid bob_payment)
  (h2 : alice_paid alice_payment_rate)
  (h3 : acorn_price price_per_acorn) :
  ∃ n : ℕ, n = (alice_payment_rate * bob_payment) / price_per_acorn ∧ n = 3600 := 
by
  sorry

end alice_acorns_purchase_l644_64481


namespace units_digit_2_pow_2015_minus_1_l644_64407

theorem units_digit_2_pow_2015_minus_1 : (2^2015 - 1) % 10 = 7 := by
  sorry

end units_digit_2_pow_2015_minus_1_l644_64407


namespace constant_term_is_21_l644_64429

def poly1 (x : ℕ) := x^3 + x^2 + 3
def poly2 (x : ℕ) := 2*x^4 + x^2 + 7
def expanded_poly (x : ℕ) := poly1 x * poly2 x

theorem constant_term_is_21 : expanded_poly 0 = 21 := by
  sorry

end constant_term_is_21_l644_64429


namespace residue_of_neg_1237_mod_37_l644_64443

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by 
  sorry

end residue_of_neg_1237_mod_37_l644_64443


namespace find_constant_x_geom_prog_l644_64480

theorem find_constant_x_geom_prog (x : ℝ) :
  (30 + x) ^ 2 = (10 + x) * (90 + x) → x = 0 :=
by
  -- Proof omitted
  sorry

end find_constant_x_geom_prog_l644_64480


namespace parametric_circle_section_l644_64489

theorem parametric_circle_section (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  ∃ (x y : ℝ), (x = 4 - Real.cos θ ∧ y = 1 - Real.sin θ) ∧ (4 - x)^2 + (1 - y)^2 = 1 :=
sorry

end parametric_circle_section_l644_64489


namespace tanya_work_days_l644_64459

theorem tanya_work_days (days_sakshi : ℕ) (efficiency_increase : ℚ) (work_rate_sakshi : ℚ) (work_rate_tanya : ℚ) (days_tanya : ℚ) :
  days_sakshi = 15 ->
  efficiency_increase = 1.25 ->
  work_rate_sakshi = 1 / days_sakshi ->
  work_rate_tanya = work_rate_sakshi * efficiency_increase ->
  days_tanya = 1 / work_rate_tanya ->
  days_tanya = 12 :=
by
  intros h_sakshi h_efficiency h_work_rate_sakshi h_work_rate_tanya h_days_tanya
  sorry

end tanya_work_days_l644_64459


namespace min_value_fraction_sum_l644_64484

theorem min_value_fraction_sum (p q r a b : ℝ) (hpq : 0 < p) (hq : p < q) (hr : q < r)
  (h_sum : p + q + r = a) (h_prod_sum : p * q + q * r + r * p = b) (h_prod : p * q * r = 48) :
  ∃ (min_val : ℝ), min_val = (1 / p) + (2 / q) + (3 / r) ∧ min_val = 3 / 2 :=
sorry

end min_value_fraction_sum_l644_64484


namespace kamari_toys_eq_65_l644_64455

-- Define the number of toys Kamari has
def number_of_toys_kamari_has : ℕ := sorry

-- Define the number of toys Anais has in terms of K
def number_of_toys_anais_has (K : ℕ) : ℕ := K + 30

-- Define the total number of toys
def total_number_of_toys (K A : ℕ) := K + A

-- Prove that the number of toys Kamari has is 65
theorem kamari_toys_eq_65 : ∃ K : ℕ, (number_of_toys_anais_has K) = K + 30 ∧ total_number_of_toys K (number_of_toys_anais_has K) = 160 ∧ K = 65 :=
by
  sorry

end kamari_toys_eq_65_l644_64455


namespace total_prime_dates_in_non_leap_year_l644_64408

def prime_dates_in_non_leap_year (days_in_months : List (Nat × Nat)) : Nat :=
  let prime_numbers := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  days_in_months.foldl 
    (λ acc (month, days) => 
      acc + (prime_numbers.filter (λ day => day ≤ days)).length) 
    0

def month_days : List (Nat × Nat) :=
  [(2, 28), (3, 31), (5, 31), (7, 31), (11,30)]

theorem total_prime_dates_in_non_leap_year : prime_dates_in_non_leap_year month_days = 52 :=
  sorry

end total_prime_dates_in_non_leap_year_l644_64408


namespace algebraic_expression_value_l644_64497

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m = 1) : 
  (m - 1)^2 + (m + 1) * (m - 1) + 2022 = 2024 :=
by
  sorry

end algebraic_expression_value_l644_64497


namespace painting_price_difference_l644_64495

theorem painting_price_difference :
  let previous_painting := 9000
  let recent_painting := 44000
  let five_times_more := 5 * previous_painting + previous_painting
  five_times_more - recent_painting = 10000 :=
by
  intros
  sorry

end painting_price_difference_l644_64495


namespace simplify_expression_l644_64496

theorem simplify_expression :
  ((3 + 5 + 6 + 2) / 3) + ((2 * 3 + 4 * 2 + 5) / 4) = 121 / 12 :=
by
  sorry

end simplify_expression_l644_64496


namespace xyz_sum_sqrt14_l644_64476

theorem xyz_sum_sqrt14 (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 1) (h2 : x + 2 * y + 3 * z = Real.sqrt 14) :
  x + y + z = (3 * Real.sqrt 14) / 7 :=
sorry

end xyz_sum_sqrt14_l644_64476


namespace sum_a2_a4_a6_l644_64438

theorem sum_a2_a4_a6 : ∀ {a : ℕ → ℕ}, (∀ i, a (i+1) = (1 / 2 : ℝ) * a i) → a 2 = 32 → a 2 + a 4 + a 6 = 42 :=
by
  intros a ha h2
  sorry

end sum_a2_a4_a6_l644_64438


namespace largest_integer_satisfying_inequality_l644_64463

theorem largest_integer_satisfying_inequality :
  ∃ x : ℤ, (6 * x - 5 < 3 * x + 4) ∧ (∀ y : ℤ, (6 * y - 5 < 3 * y + 4) → y ≤ x) ∧ x = 2 :=
by
  sorry

end largest_integer_satisfying_inequality_l644_64463


namespace rational_solutions_are_integers_l644_64410

-- Given two integers a and b, and two equations with rational solutions
variables (a b : ℤ)

-- The first equation is y - 2x = a
def eq1 (y x : ℚ) : Prop := y - 2 * x = a

-- The second equation is y^2 - xy + x^2 = b
def eq2 (y x : ℚ) : Prop := y^2 - x * y + x^2 = b

-- We want to prove that if y and x are rational solutions, they must be integers
theorem rational_solutions_are_integers (y x : ℚ) (h1 : eq1 a y x) (h2 : eq2 b y x) : 
    ∃ (y_int x_int : ℤ), y = y_int ∧ x = x_int :=
sorry

end rational_solutions_are_integers_l644_64410


namespace no_solution_for_squares_l644_64493

theorem no_solution_for_squares (x y : ℤ) (hx : x > 0) (hy : y > 0) :
  ¬ ∃ k m : ℤ, x^2 + y + 2 = k^2 ∧ y^2 + 4 * x = m^2 :=
sorry

end no_solution_for_squares_l644_64493


namespace find_the_triplet_l644_64428

theorem find_the_triplet (x y z : ℕ) (h : x + y + z = x * y * z) : (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
by
  sorry

end find_the_triplet_l644_64428


namespace quadratic_positive_range_l644_64487

theorem quadratic_positive_range (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 3 → ax^2 - 2 * a * x + 3 > 0) ↔ ((-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3)) := 
by {
  sorry
}

end quadratic_positive_range_l644_64487


namespace area_of_trapezium_is_105_l644_64405

-- Define points in the coordinate plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨14, 3⟩
def C : Point := ⟨18, 10⟩
def D : Point := ⟨0, 10⟩

noncomputable def length (p1 p2 : Point) : ℝ := abs (p2.x - p1.x)
noncomputable def height (p1 p2 : Point) : ℝ := abs (p2.y - p1.y)

-- Calculate lengths of parallel sides AB and CD, and height
noncomputable def AB := length A B
noncomputable def CD := length C D
noncomputable def heightAC := height A C

-- Define the area of trapezium
noncomputable def area_trapezium (AB CD height : ℝ) : ℝ := (1/2) * (AB + CD) * height

-- The proof problem statement
theorem area_of_trapezium_is_105 :
  area_trapezium AB CD heightAC = 105 := by
  sorry

end area_of_trapezium_is_105_l644_64405


namespace smallest_x_y_sum_299_l644_64400

theorem smallest_x_y_sum_299 : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x < y ∧ (100 + (x / y : ℚ) = 2 * (100 * x / y : ℚ)) ∧ (x + y = 299) :=
by
  sorry

end smallest_x_y_sum_299_l644_64400


namespace similarity_coefficient_interval_l644_64462

-- Definitions
def similarTriangles (x y z p : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ x = k * y ∧ y = k * z ∧ z = k * p

-- Theorem statement
theorem similarity_coefficient_interval (x y z p k : ℝ) (h_sim : similarTriangles x y z p) :
  0 ≤ k ∧ k ≤ 2 :=
sorry

end similarity_coefficient_interval_l644_64462


namespace no_upper_bound_l644_64477

-- Given Conditions
variables {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {M : ℝ}

-- Condition: widths and lengths of plates are 1 and a1, a2, a3, ..., respectively
axiom width_1 : ∀ n, (S n > 0)

-- Condition: a1 ≠ 1
axiom a1_neq_1 : a 1 ≠ 1

-- Condition: plates are similar but not congruent starting from the second
axiom similar_not_congruent : ∀ n > 1, (a (n+1) > a n)

-- Condition: S_n denotes the length covered after placing n plates
axiom Sn_length : ∀ n, S (n+1) = S n + a (n+1)

-- Condition: a_{n+1} = 1 / S_n
axiom an_reciprocal : ∀ n, a (n+1) = 1 / S n

-- The final goal: no such real number exists that S_n does not exceed
theorem no_upper_bound : ∀ M : ℝ, ∃ n : ℕ, S n > M := 
sorry

end no_upper_bound_l644_64477


namespace books_arrangement_l644_64482

-- All conditions provided in Lean as necessary definitions
def num_arrangements (math_books english_books science_books : ℕ) : ℕ :=
  if math_books = 4 ∧ english_books = 6 ∧ science_books = 2 then
    let arrangements_groups := 2 * 3  -- Number of valid group placements
    let arrangements_math := Nat.factorial math_books
    let arrangements_english := Nat.factorial english_books
    let arrangements_science := Nat.factorial science_books
    arrangements_groups * arrangements_math * arrangements_english * arrangements_science
  else
    0

theorem books_arrangement : num_arrangements 4 6 2 = 207360 :=
by
  sorry

end books_arrangement_l644_64482


namespace functional_equation_solution_l644_64470

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y, f (x ^ 2) - f (y ^ 2) + 2 * x + 1 = f (x + y) * f (x - y)) :
  (∀ x, f x = x + 1) ∨ (∀ x, f x = -x - 1) :=
by
  sorry

end functional_equation_solution_l644_64470


namespace solve_inequality_group_l644_64411

theorem solve_inequality_group (x : ℝ) (h1 : -9 < 2 * x - 1) (h2 : 2 * x - 1 ≤ 6) :
  -4 < x ∧ x ≤ 3.5 := 
sorry

end solve_inequality_group_l644_64411


namespace painting_time_l644_64451

theorem painting_time (t₁₂ : ℕ) (h : t₁₂ = 6) (r : ℝ) (hr : r = t₁₂ / 12) (n : ℕ) (hn : n = 20) : 
  t₁₂ + n * r = 16 := by
  sorry

end painting_time_l644_64451


namespace triangle_area_example_l644_64479

def Point := (ℝ × ℝ)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_example :
  triangle_area (-2, 3) (7, -1) (4, 6) = 25.5 :=
by
  -- Proof will be here
  sorry

end triangle_area_example_l644_64479


namespace power_product_rule_l644_64449

theorem power_product_rule (a : ℤ) : (-a^2)^3 = -a^6 := 
by 
  sorry

end power_product_rule_l644_64449
