import Mathlib

namespace percentage_change_difference_l1325_132505

-- Define initial and final percentages
def initial_yes : ℝ := 0.4
def initial_no : ℝ := 0.6
def final_yes : ℝ := 0.6
def final_no : ℝ := 0.4

-- Definition for the percentage of students who changed their opinion
def y_min : ℝ := 0.2 -- 20%
def y_max : ℝ := 0.6 -- 60%

-- Calculate the difference
def difference_y : ℝ := y_max - y_min

theorem percentage_change_difference :
  difference_y = 0.4 := by
  sorry

end percentage_change_difference_l1325_132505


namespace ratio_x_y_l1325_132507

noncomputable def side_length_x (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    a = 5 ∧ b = 12 ∧ c = 13 ∧ 
    (12 - x) / x = 5 / 12 ∧
    12 * x = 5 * x + 60 ∧
    7 * x = 60

noncomputable def side_length_y (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    a = 5 ∧ b = 12 ∧ c = 13 ∧
    y = 60 / 17

theorem ratio_x_y (x y : ℝ) (hx : side_length_x x) (hy : side_length_y y) : x / y = 17 / 7 :=
by
  sorry

end ratio_x_y_l1325_132507


namespace quadratic_roster_method_l1325_132531

theorem quadratic_roster_method :
  {x : ℝ | x^2 - 3 * x + 2 = 0} = {1, 2} :=
by
  sorry

end quadratic_roster_method_l1325_132531


namespace new_average_weight_l1325_132518

theorem new_average_weight (original_players : ℕ) (new_players : ℕ) 
  (average_weight_original : ℝ) (weight_new_player1 : ℝ) (weight_new_player2 : ℝ) : 
  original_players = 7 → 
  new_players = 2 →
  average_weight_original = 76 → 
  weight_new_player1 = 110 → 
  weight_new_player2 = 60 → 
  (original_players * average_weight_original + weight_new_player1 + weight_new_player2) / (original_players + new_players) = 78 :=
by 
  intros h1 h2 h3 h4 h5;
  sorry

end new_average_weight_l1325_132518


namespace train_crosses_signal_post_time_l1325_132591

theorem train_crosses_signal_post_time 
  (length_train : ℕ) 
  (length_bridge : ℕ) 
  (time_bridge_minutes : ℕ) 
  (time_signal_post_seconds : ℕ) 
  (h_length_train : length_train = 600) 
  (h_length_bridge : length_bridge = 1800) 
  (h_time_bridge_minutes : time_bridge_minutes = 2) 
  (h_time_signal_post : time_signal_post_seconds = 30) : 
  (length_train / ((length_train + length_bridge) / (time_bridge_minutes * 60))) = time_signal_post_seconds :=
by
  sorry

end train_crosses_signal_post_time_l1325_132591


namespace sequence_solution_l1325_132502

-- Define the sequence x_n
def x (n : ℕ) : ℚ := n / (n + 2016)

-- Given condition: x_2016 = x_m * x_n
theorem sequence_solution (m n : ℕ) (h : x 2016 = x m * x n) : 
  m = 4032 ∧ n = 6048 := 
  by sorry

end sequence_solution_l1325_132502


namespace polyhedron_with_12_edges_l1325_132586

def prism_edges (n : Nat) : Nat :=
  3 * n

def pyramid_edges (n : Nat) : Nat :=
  2 * n

def Quadrangular_prism : Nat := prism_edges 4
def Quadrangular_pyramid : Nat := pyramid_edges 4
def Pentagonal_pyramid : Nat := pyramid_edges 5
def Pentagonal_prism : Nat := prism_edges 5

theorem polyhedron_with_12_edges :
  (Quadrangular_prism = 12) ∧
  (Quadrangular_pyramid ≠ 12) ∧
  (Pentagonal_pyramid ≠ 12) ∧
  (Pentagonal_prism ≠ 12) := by
  sorry

end polyhedron_with_12_edges_l1325_132586


namespace part1_part2_l1325_132540

-- Part (1)
theorem part1 : -6 * -2 + -5 * 16 = -68 := by
  sorry

-- Part (2)
theorem part2 : -1^4 + (1 / 4) * (2 * -6 - (-4)^2) = -8 := by
  sorry

end part1_part2_l1325_132540


namespace score_after_7_hours_l1325_132598

theorem score_after_7_hours (score : ℕ) (time : ℕ) : 
  (score / time = 90 / 5) → time = 7 → score = 126 :=
by
  sorry

end score_after_7_hours_l1325_132598


namespace problem_proof_l1325_132504

theorem problem_proof (n : ℕ) 
  (h : ∃ k, 2 * k = n) :
  4 ∣ n :=
sorry

end problem_proof_l1325_132504


namespace roots_of_equation_l1325_132550

theorem roots_of_equation :
  ∀ x : ℝ, x * (x - 1) + 3 * (x - 1) = 0 ↔ x = -3 ∨ x = 1 :=
by {
  sorry
}

end roots_of_equation_l1325_132550


namespace part1_part2_l1325_132527

def p (x : ℝ) : Prop := x^2 - 10*x + 16 ≤ 0
def q (x m : ℝ) : Prop := m > 0 ∧ x^2 - 4*m*x + 3*m^2 ≤ 0

theorem part1 (x : ℝ) : 
  (∃ (m : ℝ), m = 1 ∧ (p x ∨ q x m)) → 1 ≤ x ∧ x ≤ 8 :=
by
  intros
  sorry

theorem part2 (m : ℝ) :
  (∀ x, q x m → p x) ∧ ∃ x, ¬ q x m ∧ p x → 2 ≤ m ∧ m ≤ 8/3 :=
by
  intros
  sorry

end part1_part2_l1325_132527


namespace johns_sister_age_l1325_132566

variable (j d s : ℝ)

theorem johns_sister_age 
  (h1 : j = d - 15)
  (h2 : j + d = 100)
  (h3 : s = j - 5) :
  s = 37.5 := 
sorry

end johns_sister_age_l1325_132566


namespace calculate_cells_after_12_days_l1325_132594

theorem calculate_cells_after_12_days :
  let initial_cells := 5
  let division_factor := 3
  let days := 12
  let period := 3
  let n := days / period
  initial_cells * division_factor ^ (n - 1) = 135 := by
  sorry

end calculate_cells_after_12_days_l1325_132594


namespace find_k_parallel_vectors_l1325_132500

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem find_k_parallel_vectors (k : ℝ) :
  let a := (1, k)
  let b := (-2, 6)
  vector_parallel a b → k = -3 :=
by
  sorry

end find_k_parallel_vectors_l1325_132500


namespace integer_parts_are_divisible_by_17_l1325_132562

-- Define that a is the greatest positive root of the given polynomial
def is_greatest_positive_root (a : ℝ) : Prop :=
  (∀ x : ℝ, x^3 - 3 * x^2 + 1 = 0 → x ≤ a) ∧ a > 0 ∧ (a^3 - 3 * a^2 + 1 = 0)

-- Define the main theorem to prove
theorem integer_parts_are_divisible_by_17 (a : ℝ)
  (h_root : is_greatest_positive_root a) :
  (⌊a ^ 1788⌋ % 17 = 0) ∧ (⌊a ^ 1988⌋ % 17 = 0) := 
sorry

end integer_parts_are_divisible_by_17_l1325_132562


namespace problem_p_s_difference_l1325_132584

def P : ℤ := 12 - (3 * 4)
def S : ℤ := (12 - 3) * 4

theorem problem_p_s_difference : P - S = -36 := by
  sorry

end problem_p_s_difference_l1325_132584


namespace simple_interest_problem_l1325_132526

theorem simple_interest_problem (P : ℝ) (R : ℝ) (T : ℝ) : T = 10 → 
  ((P * R * T) / 100 = (4 / 5) * P) → R = 8 :=
by
  intros hT hsi
  sorry

end simple_interest_problem_l1325_132526


namespace Kaleb_second_half_points_l1325_132571

theorem Kaleb_second_half_points (first_half_points total_points : ℕ) (h1 : first_half_points = 43) (h2 : total_points = 66) : total_points - first_half_points = 23 := by
  sorry

end Kaleb_second_half_points_l1325_132571


namespace find_angle_phi_l1325_132546

-- Definitions for the conditions given in the problem
def folded_paper_angle (φ : ℝ) : Prop := 0 < φ ∧ φ < 90

def angle_XOY := 144

-- The main statement to be proven
theorem find_angle_phi (φ : ℝ) (h1 : folded_paper_angle φ) : φ = 81 :=
sorry

end find_angle_phi_l1325_132546


namespace solve_for_x_l1325_132503

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end solve_for_x_l1325_132503


namespace mowing_work_rate_l1325_132557

variables (A B C : ℚ)

theorem mowing_work_rate :
  A + B = 1/28 → A + B + C = 1/21 → C = 1/84 :=
by
  intros h1 h2
  sorry

end mowing_work_rate_l1325_132557


namespace positive_integer_pairs_l1325_132574

theorem positive_integer_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  a^b = b^(a^2) ↔ (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 16) ∨ (a = 3 ∧ b = 27) :=
by sorry

end positive_integer_pairs_l1325_132574


namespace last_four_digits_of_5_pow_2017_l1325_132577

theorem last_four_digits_of_5_pow_2017 : (5 ^ 2017) % 10000 = 3125 :=
by sorry

end last_four_digits_of_5_pow_2017_l1325_132577


namespace primes_with_no_sum_of_two_cubes_l1325_132534

theorem primes_with_no_sum_of_two_cubes (p : ℕ) [Fact (Nat.Prime p)] :
  (∃ n : ℤ, ∀ x y : ℤ, x^3 + y^3 ≠ n % p) ↔ p = 7 :=
sorry

end primes_with_no_sum_of_two_cubes_l1325_132534


namespace min_xyz_product_l1325_132509

open Real

theorem min_xyz_product
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_sum : x + y + z = 1)
  (h_no_more_than_twice : x ≤ 2 * y ∧ y ≤ 2 * x ∧ y ≤ 2 * z ∧ z ≤ 2 * y) :
  ∃ p : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x + y + z = 1 → x ≤ 2 * y ∧ y ≤ 2 * x ∧ y ≤ 2 * z ∧ z ≤ 2 * y → x * y * z ≥ p) ∧ p = 1 / 32 :=
by
  sorry

end min_xyz_product_l1325_132509


namespace remainder_b_div_6_l1325_132523

theorem remainder_b_div_6 (a b : ℕ) (r_a r_b : ℕ) 
  (h1 : a ≡ r_a [MOD 6]) 
  (h2 : b ≡ r_b [MOD 6]) 
  (h3 : a > b) 
  (h4 : (a - b) % 6 = 5) 
  : b % 6 = 0 := 
sorry

end remainder_b_div_6_l1325_132523


namespace prove_y_identity_l1325_132559

theorem prove_y_identity (y : ℤ) (h1 : y^2 = 2209) : (y + 2) * (y - 2) = 2205 :=
by
  sorry

end prove_y_identity_l1325_132559


namespace find_missing_number_l1325_132569

theorem find_missing_number (n : ℤ) (h : 1234562 - n * 3 * 2 = 1234490) : 
  n = 12 :=
by
  sorry

end find_missing_number_l1325_132569


namespace stacy_height_now_l1325_132568

-- Definitions based on the given conditions
def S_initial : ℕ := 50
def J_initial : ℕ := 45
def J_growth : ℕ := 1
def S_growth : ℕ := J_growth + 6

-- Prove statement about Stacy's current height
theorem stacy_height_now : S_initial + S_growth = 57 := by
  sorry

end stacy_height_now_l1325_132568


namespace geom_prog_common_ratio_unique_l1325_132545

theorem geom_prog_common_ratio_unique (b q : ℝ) (hb : b > 0) (hq : q > 1) :
  (∃ b : ℝ, (q = (1 + Real.sqrt 5) / 2) ∧ 
    (0 < b ∧ b * q ≠ b ∧ b * q^2 ≠ b ∧ b * q^3 ≠ b) ∧ 
    ((2 * b * q = b + b * q^2) ∨ (2 * b * q = b + b * q^3) ∨ (2 * b * q^2 = b + b * q^3))) := 
sorry

end geom_prog_common_ratio_unique_l1325_132545


namespace union_sets_intersection_complement_l1325_132548

open Set

noncomputable def U := (univ : Set ℝ)
def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | x < 5 }

theorem union_sets : A ∪ B = univ := by
  sorry

theorem intersection_complement : (U \ A) ∩ B = { x : ℝ | x < 2 } := by
  sorry

end union_sets_intersection_complement_l1325_132548


namespace scientific_notation_460_billion_l1325_132551

theorem scientific_notation_460_billion : 460000000000 = 4.6 * 10^11 := 
sorry

end scientific_notation_460_billion_l1325_132551


namespace inequality_always_true_l1325_132592

theorem inequality_always_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| := 
sorry

end inequality_always_true_l1325_132592


namespace infinite_series_correct_l1325_132581

noncomputable def infinite_series_sum : ℚ := 
  ∑' n : ℕ, (n+1)^2 * (1/999)^n

theorem infinite_series_correct : infinite_series_sum = 997005 / 996004 :=
  sorry

end infinite_series_correct_l1325_132581


namespace sqrt_450_eq_15_sqrt_2_l1325_132538

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l1325_132538


namespace stateA_issues_more_than_stateB_l1325_132565

-- Definitions based on conditions
def stateA_format : ℕ := 26^5 * 10^1
def stateB_format : ℕ := 26^3 * 10^3

-- Proof problem statement
theorem stateA_issues_more_than_stateB : stateA_format - stateB_format = 10123776 := by
  sorry

end stateA_issues_more_than_stateB_l1325_132565


namespace orthographic_projection_area_l1325_132525

theorem orthographic_projection_area (s : ℝ) (h : s = 1) : 
  let S := (Real.sqrt 3) / 4 
  let factor := (Real.sqrt 2) / 2
  let S' := (factor ^ 2) * S
  S' = (Real.sqrt 6) / 16 :=
by
  let S := (Real.sqrt 3) / 4
  let factor := (Real.sqrt 2) / 2
  let S' := (factor ^ 2) * S
  sorry

end orthographic_projection_area_l1325_132525


namespace unique_outfits_count_l1325_132547

theorem unique_outfits_count (s : Fin 5) (p : Fin 6) (restricted_pairings : (Fin 1 × Fin 2) → Prop) 
  (r : restricted_pairings (0, 0) ∧ restricted_pairings (0, 1)) : ∃ n, n = 28 ∧ 
  ∃ (outfits : Fin 5 → Fin 6 → Prop), 
    (∀ s p, outfits s p) ∧ 
    (∀ p, ¬outfits 0 p ↔ p = 0 ∨ p = 1) := by
  sorry

end unique_outfits_count_l1325_132547


namespace quadractic_integer_roots_l1325_132590

theorem quadractic_integer_roots (n : ℕ) (h : n > 0) :
  (∃ x y : ℤ, x^2 - 4 * x + n = 0 ∧ y^2 - 4 * y + n = 0) ↔ (n = 3 ∨ n = 4) :=
by
  sorry

end quadractic_integer_roots_l1325_132590


namespace difference_of_squares_divisible_by_18_l1325_132579

-- Definitions of odd integers.
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- The main theorem stating the equivalence.
theorem difference_of_squares_divisible_by_18 (a b : ℤ) 
  (ha : is_odd a) (hb : is_odd b) : 
  ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) % 18 = 0 := 
by
  sorry

end difference_of_squares_divisible_by_18_l1325_132579


namespace total_games_played_l1325_132529

theorem total_games_played (n : ℕ) (h : n = 7) : (n.choose 2) = 21 := by
  sorry

end total_games_played_l1325_132529


namespace coeffs_sum_of_binomial_expansion_l1325_132552

theorem coeffs_sum_of_binomial_expansion :
  (3 * x - 2) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_0 = 64 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = -63 :=
by
  sorry

end coeffs_sum_of_binomial_expansion_l1325_132552


namespace sequence_general_term_l1325_132597

def recurrence_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n / (1 + a n)

theorem sequence_general_term :
  ∀ a : ℕ → ℚ, recurrence_sequence a → ∀ n : ℕ, n ≥ 1 → a n = 2 / (2 * n - 1) :=
by
  intro a h n hn
  sorry

end sequence_general_term_l1325_132597


namespace part_I_solution_set_part_II_range_a_l1325_132573

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part_I_solution_set :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} :=
by
  sorry

theorem part_II_range_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - a) ↔ (-1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end part_I_solution_set_part_II_range_a_l1325_132573


namespace shoes_produced_min_pairs_for_profit_l1325_132563

-- given conditions
def production_cost (n : ℕ) : ℕ := 4000 + 50 * n

-- Question (1)
theorem shoes_produced (C : ℕ) (h : C = 36000) : ∃ n : ℕ, production_cost n = C :=
by sorry

-- given conditions for part (2)
def selling_price (price_per_pair : ℕ) (n : ℕ) : ℕ := price_per_pair * n
def profit (price_per_pair : ℕ) (n : ℕ) : ℕ := selling_price price_per_pair n - production_cost n

-- Question (2)
theorem min_pairs_for_profit (price_per_pair profit_goal : ℕ) (h : price_per_pair = 90) (h1 : profit_goal = 8500) :
  ∃ n : ℕ, profit price_per_pair n ≥ profit_goal :=
by sorry

end shoes_produced_min_pairs_for_profit_l1325_132563


namespace trig_relation_l1325_132501

theorem trig_relation : (Real.pi/4 < 1) ∧ (1 < Real.pi/2) → Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := 
by 
  intro h
  sorry

end trig_relation_l1325_132501


namespace average_of_remaining_four_l1325_132580

theorem average_of_remaining_four (avg10 : ℕ → ℕ) (avg6 : ℕ → ℕ) 
  (h_avg10 : avg10 10 = 80) 
  (h_avg6 : avg6 6 = 58) : 
  (avg10 10 - avg6 6 * 6) / 4 = 113 :=
sorry

end average_of_remaining_four_l1325_132580


namespace domain_of_sqrt_and_fraction_l1325_132593

def domain_of_function (x : ℝ) : Prop :=
  2 * x - 3 ≥ 0 ∧ x ≠ 3

theorem domain_of_sqrt_and_fraction :
  {x : ℝ | domain_of_function x} = {x : ℝ | x ≥ 3 / 2} \ {3} :=
by sorry

end domain_of_sqrt_and_fraction_l1325_132593


namespace kayla_apples_l1325_132549

variable (x y : ℕ)
variable (h1 : x + (10 + 4 * x) = 340)
variable (h2 : y = 10 + 4 * x)

theorem kayla_apples : y = 274 :=
by
  sorry

end kayla_apples_l1325_132549


namespace Inez_initial_money_l1325_132533

theorem Inez_initial_money (X : ℝ) (h : X - (X / 2 + 50) = 25) : X = 150 :=
by
  sorry

end Inez_initial_money_l1325_132533


namespace trigonometric_proof_l1325_132519

noncomputable def cos30 : ℝ := Real.sqrt 3 / 2
noncomputable def tan60 : ℝ := Real.sqrt 3
noncomputable def sin45 : ℝ := Real.sqrt 2 / 2
noncomputable def cos45 : ℝ := Real.sqrt 2 / 2

theorem trigonometric_proof :
  2 * cos30 - tan60 + sin45 * cos45 = 1 / 2 :=
by
  sorry

end trigonometric_proof_l1325_132519


namespace book_cost_price_l1325_132560

theorem book_cost_price (SP : ℝ) (P : ℝ) (C : ℝ) (hSP: SP = 260) (hP: P = 0.20) : C = 216.67 :=
by 
  sorry

end book_cost_price_l1325_132560


namespace circle_standard_equation_l1325_132535

noncomputable def circle_through_ellipse_vertices : Prop :=
  ∃ (a : ℝ) (r : ℝ), a < 0 ∧
    (∀ (x y : ℝ),   -- vertices of the ellipse
      ((x = 4 ∧ y = 0) ∨ (x = 0 ∧ (y = 2 ∨ y = -2)))
      → (x + a)^2 + y^2 = r^2) ∧
    ( a = -3/2 ∧ r = 5/2 ∧ 
      ∀ (x y : ℝ), (x + 3/2)^2 + y^2 = (5/2)^2
    )

theorem circle_standard_equation :
  circle_through_ellipse_vertices :=
sorry

end circle_standard_equation_l1325_132535


namespace f_2007_l1325_132585

def A : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 1}

noncomputable def f : A → ℝ := sorry

theorem f_2007 :
  (∀ x : ℚ, x ∈ A → f ⟨x, sorry⟩ + f ⟨1 - (1/x), sorry⟩ = Real.log (|x|)) →
  f ⟨2007, sorry⟩ = Real.log (|2007|) :=
sorry

end f_2007_l1325_132585


namespace minimum_value_f_on_interval_l1325_132514

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^3 / (Real.sin x) + (Real.sin x)^3 / (Real.cos x)

theorem minimum_value_f_on_interval : ∃ x ∈ Set.Ioo 0 (Real.pi / 2), f x = 1 ∧ ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≥ 1 :=
by sorry

end minimum_value_f_on_interval_l1325_132514


namespace correct_statements_eq_l1325_132539

-- Definitions used in the Lean 4 statement should only directly appear in the conditions
variable {a b c : ℝ} 

-- Use the condition directly
theorem correct_statements_eq (h : a / c = b / c) (hc : c ≠ 0) : a = b := 
by
  -- This is where the proof would go
  sorry

end correct_statements_eq_l1325_132539


namespace boat_travel_distance_downstream_l1325_132530

def boat_speed : ℝ := 22 -- Speed of boat in still water in km/hr
def stream_speed : ℝ := 5 -- Speed of the stream in km/hr
def time_downstream : ℝ := 7 -- Time taken to travel downstream in hours
def effective_speed_downstream : ℝ := boat_speed + stream_speed -- Effective speed downstream

theorem boat_travel_distance_downstream : effective_speed_downstream * time_downstream = 189 := by
  -- Since effective_speed_downstream = 27 (22 + 5)
  -- Distance = Speed * Time
  -- Hence, Distance = 27 km/hr * 7 hours = 189 km
  sorry

end boat_travel_distance_downstream_l1325_132530


namespace sum_of_digits_18_l1325_132575

def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem sum_of_digits_18 (A B C D : ℕ) 
(h1 : A + D = 10)
(h2 : B + C + 1 = 10 + D)
(h3 : C + B + 1 = 10 + B)
(h4 : D + A + 1 = 11)
(h_distinct : distinct_digits A B C D) :
  A + B + C + D = 18 :=
sorry

end sum_of_digits_18_l1325_132575


namespace salary_reduction_l1325_132599

theorem salary_reduction (S : ℝ) (x : ℝ) 
  (H1 : S > 0) 
  (H2 : 1.25 * S * (1 - 0.01 * x) = 1.0625 * S) : 
  x = 15 := 
  sorry

end salary_reduction_l1325_132599


namespace age_comparison_l1325_132555

variable (P A F X : ℕ)

theorem age_comparison :
  P = 50 →
  P = 5 / 4 * A →
  P = 5 / 6 * F →
  X = 50 - A →
  X = 10 :=
by { sorry }

end age_comparison_l1325_132555


namespace no_nat_solutions_l1325_132564

theorem no_nat_solutions (x y : ℕ) : (2 * x + y) * (2 * y + x) ≠ 2017 ^ 2017 := by sorry

end no_nat_solutions_l1325_132564


namespace minimum_value_y_l1325_132510

theorem minimum_value_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (∀ x : ℝ, x = (1 / a + 4 / b) → x ≥ 9 / 2) :=
sorry

end minimum_value_y_l1325_132510


namespace pizza_slices_with_both_toppings_l1325_132556

theorem pizza_slices_with_both_toppings :
  let T := 16
  let c := 10
  let p := 12
  let n := c + p - T
  n = 6 :=
by
  let T := 16
  let c := 10
  let p := 12
  let n := c + p - T
  show n = 6
  sorry

end pizza_slices_with_both_toppings_l1325_132556


namespace smallest_positive_int_linear_combination_l1325_132582

theorem smallest_positive_int_linear_combination (m n : ℤ) :
  ∃ k : ℤ, 4509 * m + 27981 * n = k ∧ k > 0 ∧ k ≤ 4509 * m + 27981 * n → k = 3 :=
by
  sorry

end smallest_positive_int_linear_combination_l1325_132582


namespace triangle_angle_bisector_YE_l1325_132520

noncomputable def triangle_segs_YE : ℝ := (36 : ℝ) / 7

theorem triangle_angle_bisector_YE
  (XYZ: Type)
  (XY XZ YZ YE EZ: ℝ)
  (YZ_length : YZ = 12)
  (side_ratios : XY / XZ = 3 / 4 ∧ XY / YZ  = 3 / 5 ∧ XZ / YZ = 4 / 5)
  (angle_bisector : YE / EZ = XY / XZ)
  (seg_sum : YE + EZ = YZ) :
  YE = (36 : ℝ) / 7 :=
by sorry

end triangle_angle_bisector_YE_l1325_132520


namespace at_least_one_not_greater_than_neg_two_l1325_132553

open Real

theorem at_least_one_not_greater_than_neg_two
  {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + (1 / b) ≤ -2 ∨ b + (1 / c) ≤ -2 ∨ c + (1 / a) ≤ -2 :=
sorry

end at_least_one_not_greater_than_neg_two_l1325_132553


namespace negative_870_in_third_quadrant_l1325_132508

noncomputable def angle_in_third_quadrant (theta : ℝ) : Prop :=
  180 < theta ∧ theta < 270

theorem negative_870_in_third_quadrant:
  angle_in_third_quadrant 210 :=
by
  sorry

end negative_870_in_third_quadrant_l1325_132508


namespace g_is_even_l1325_132522

noncomputable def g (x : ℝ) : ℝ := Real.log (Real.cos x + Real.sqrt (1 + Real.sin x ^ 2))

theorem g_is_even : ∀ x : ℝ, g (-x) = g (x) :=
by
  intro x
  sorry

end g_is_even_l1325_132522


namespace decreasing_function_on_real_l1325_132536

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x + y) = f x + f y
axiom f_negative (x : ℝ) : x > 0 → f x < 0
axiom f_not_identically_zero : ∃ x, f x ≠ 0

theorem decreasing_function_on_real :
  ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
sorry

end decreasing_function_on_real_l1325_132536


namespace sequence_initial_term_l1325_132561

theorem sequence_initial_term (a : ℕ) :
  let a_1 := a
  let a_2 := 2
  let a_3 := a_1 + a_2
  let a_4 := a_1 + a_2 + a_3
  let a_5 := a_1 + a_2 + a_3 + a_4
  let a_6 := a_1 + a_2 + a_3 + a_4 + a_5
  a_6 = 56 → a = 5 :=
by
  intros h
  sorry

end sequence_initial_term_l1325_132561


namespace segment_length_l1325_132511

theorem segment_length (A B C : ℝ) (hAB : abs (A - B) = 3) (hBC : abs (B - C) = 5) :
  abs (A - C) = 2 ∨ abs (A - C) = 8 := by
  sorry

end segment_length_l1325_132511


namespace lily_disproves_tom_claim_l1325_132572

-- Define the cards and the claim
inductive Card
| A : Card
| R : Card
| Circle : Card
| Square : Card
| Triangle : Card

def has_consonant (c : Card) : Prop :=
  match c with
  | Card.R => true
  | _ => false

def has_triangle (c : Card) : Card → Prop :=
  fun c' =>
    match c with
    | Card.R => c' = Card.Triangle
    | _ => true

def tom_claim (c : Card) (c' : Card) : Prop :=
  has_consonant c → has_triangle c c'

-- Proof problem statement:
theorem lily_disproves_tom_claim (c : Card) (c' : Card) : c = Card.R → ¬ has_triangle c c' → ¬ tom_claim c c' :=
by
  intros
  sorry

end lily_disproves_tom_claim_l1325_132572


namespace at_least_one_negative_l1325_132587

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a^2 + 1 / b = b^2 + 1 / a) : a < 0 ∨ b < 0 :=
by
  sorry

end at_least_one_negative_l1325_132587


namespace intersection_of_sets_l1325_132506

def A := { x : ℝ | x^2 - 2 * x - 8 < 0 }
def B := { x : ℝ | x >= 0 }
def intersection := { x : ℝ | 0 <= x ∧ x < 4 }

theorem intersection_of_sets : (A ∩ B) = intersection := 
sorry

end intersection_of_sets_l1325_132506


namespace triangle_area_13_14_15_l1325_132537

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let sin_C := Real.sqrt (1 - cos_C^2)
  (1/2) * a * b * sin_C

theorem triangle_area_13_14_15 : area_of_triangle 13 14 15 = 84 :=
by sorry

end triangle_area_13_14_15_l1325_132537


namespace select_and_swap_ways_l1325_132512

theorem select_and_swap_ways :
  let n := 8
  let k := 3
  Nat.choose n k * 2 = 112 := 
by
  let n := 8
  let k := 3
  sorry

end select_and_swap_ways_l1325_132512


namespace water_level_function_l1325_132532

def water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : ℝ :=
  0.3 * x + 6

theorem water_level_function :
  ∀ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5), water_level x h = 6 + 0.3 * x :=
by
  intros
  unfold water_level
  sorry -- Proof skipped

end water_level_function_l1325_132532


namespace ratio_of_perimeters_l1325_132521

theorem ratio_of_perimeters (s₁ s₂ : ℝ) (h : (s₁^2 / s₂^2) = (16 / 49)) : (4 * s₁) / (4 * s₂) = 4 / 7 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l1325_132521


namespace smallest_n_for_divisibility_condition_l1325_132554

theorem smallest_n_for_divisibility_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) →
    (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (xyz ∣ (x + y + z)^n)) ∧
    n = 13 :=
by
  use 13
  sorry

end smallest_n_for_divisibility_condition_l1325_132554


namespace jane_savings_l1325_132524

noncomputable def cost_promotion_A (price: ℝ) : ℝ :=
  price + (price / 2)

noncomputable def cost_promotion_B (price: ℝ) : ℝ :=
  price + (price - (price * 0.25))

theorem jane_savings (price : ℝ) (h_price_pos : 0 < price) : 
  cost_promotion_B price - cost_promotion_A price = 12.5 :=
by
  let price := 50
  unfold cost_promotion_A
  unfold cost_promotion_B
  norm_num
  sorry

end jane_savings_l1325_132524


namespace tile_5x7_rectangle_with_L_trominos_l1325_132570

theorem tile_5x7_rectangle_with_L_trominos :
  ∀ k : ℕ, ¬ (∃ (tile : ℕ → ℕ → ℕ), (∀ i j, tile (i+1) (j+1) = tile (i+3) (j+3)) ∧
    ∀ i j, (i < 5 ∧ j < 7) → (tile i j = k)) :=
by sorry

end tile_5x7_rectangle_with_L_trominos_l1325_132570


namespace quadratic_has_equal_roots_l1325_132515

-- Proposition: If the quadratic equation 3x^2 + 6x + m = 0 has two equal real roots, then m = 3.

theorem quadratic_has_equal_roots (m : ℝ) : 3 * 6 - 12 * m = 0 → m = 3 :=
by
  intro h
  sorry

end quadratic_has_equal_roots_l1325_132515


namespace sam_balloons_l1325_132558

theorem sam_balloons (f d t S : ℝ) (h₁ : f = 10.0) (h₂ : d = 16.0) (h₃ : t = 40.0) (h₄ : f + S - d = t) : S = 46.0 := 
by 
  -- Replace "sorry" with a valid proof to solve this problem
  sorry

end sam_balloons_l1325_132558


namespace proof_math_problem_l1325_132528

-- Define the conditions
structure Conditions where
  person1_start_noon : ℕ -- Person 1 starts from Appleminster at 12:00 PM
  person2_start_2pm : ℕ -- Person 2 starts from Boniham at 2:00 PM
  meet_time : ℕ -- They meet at 4:55 PM
  finish_time_simultaneously : Bool -- They finish their journey simultaneously

-- Define the problem
def math_problem (c : Conditions) : Prop :=
  let arrival_time := 7 * 60 -- 7:00 PM in minutes
  c.person1_start_noon = 0 ∧ -- Noon as 0 minutes (12:00 PM)
  c.person2_start_2pm = 120 ∧ -- 2:00 PM as 120 minutes
  c.meet_time = 295 ∧ -- 4:55 PM as 295 minutes
  c.finish_time_simultaneously = true → arrival_time = 420 -- 7:00 PM in minutes

-- Prove the problem statement, skipping actual proof
theorem proof_math_problem (c : Conditions) : math_problem c :=
  by sorry

end proof_math_problem_l1325_132528


namespace min_value_of_expr_min_value_at_specific_points_l1325_132542

noncomputable def min_value_expr (p q r : ℝ) : ℝ := 8 * p^4 + 18 * q^4 + 50 * r^4 + 1 / (8 * p * q * r)

theorem min_value_of_expr : ∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → min_value_expr p q r ≥ 6 :=
by
  intro p q r hp hq hr
  sorry

theorem min_value_at_specific_points : min_value_expr (1 / (8 : ℝ)^(1 / 4)) (1 / (18 : ℝ)^(1 / 4)) (1 / (50 : ℝ)^(1 / 4)) = 6 :=
by
  sorry

end min_value_of_expr_min_value_at_specific_points_l1325_132542


namespace q1_q2_q3_l1325_132589

-- (1) Given |a| = 3, |b| = 1, and a < b, prove a + b = -2 or -4.
theorem q1 (a b : ℚ) (h1 : |a| = 3) (h2 : |b| = 1) (h3 : a < b) : a + b = -2 ∨ a + b = -4 := sorry

-- (2) Given rational numbers a and b such that ab ≠ 0, prove the value of (a/|a|) + (b/|b|) is 2, -2, or 0.
theorem q2 (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) : (a / |a|) + (b / |b|) = 2 ∨ (a / |a|) + (b / |b|) = -2 ∨ (a / |a|) + (b / |b|) = 0 := sorry

-- (3) Given rational numbers a, b, c such that a + b + c = 0 and abc < 0, prove the value of (b+c)/|a| + (a+c)/|b| + (a+b)/|c| is -1.
theorem q3 (a b c : ℚ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : (b + c) / |a| + (a + c) / |b| + (a + b) / |c| = -1 := sorry

end q1_q2_q3_l1325_132589


namespace trigonometric_inequality_l1325_132513

-- Define the necessary mathematical objects and structures:
noncomputable def sin (x : ℝ) : ℝ := sorry -- Assume sine function as given

-- The theorem statement
theorem trigonometric_inequality {x y z A B C : ℝ} 
  (hA : A + B + C = π) -- A, B, C are angles of a triangle
  :
  ((x + y + z) / 2) ^ 2 ≥ x * y * (sin A) ^ 2 + y * z * (sin B) ^ 2 + z * x * (sin C) ^ 2 :=
sorry

end trigonometric_inequality_l1325_132513


namespace solution_set_of_inequality_l1325_132517

theorem solution_set_of_inequality (x : ℝ) : {x | x * (x - 1) > 0} = { x | x < 0 } ∪ { x | x > 1 } :=
sorry

end solution_set_of_inequality_l1325_132517


namespace stickers_earned_correct_l1325_132516

-- Define the initial and final number of stickers.
def initial_stickers : ℕ := 39
def final_stickers : ℕ := 61

-- Define how many stickers Pat earned during the week
def stickers_earned : ℕ := final_stickers - initial_stickers

-- State the main theorem
theorem stickers_earned_correct : stickers_earned = 22 :=
by
  show final_stickers - initial_stickers = 22
  sorry

end stickers_earned_correct_l1325_132516


namespace evaluate_dollar_op_l1325_132588

def dollar_op (x y : ℤ) := x * (y + 2) + 2 * x * y

theorem evaluate_dollar_op : dollar_op 4 (-1) = -4 :=
by
  -- Proof steps here
  sorry

end evaluate_dollar_op_l1325_132588


namespace algebra_expression_value_l1325_132578

theorem algebra_expression_value (x y : ℤ) (h : x - 2 * y + 2 = 5) : 2 * x - 4 * y - 1 = 5 :=
by
  sorry

end algebra_expression_value_l1325_132578


namespace complement_union_A_B_in_U_l1325_132595

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l1325_132595


namespace ages_of_boys_l1325_132544

theorem ages_of_boys (a b c : ℕ) (h1 : a + b + c = 29) (h2 : a = b) (h3 : c = 11) : a = 9 :=
by
  sorry

end ages_of_boys_l1325_132544


namespace set_union_proof_l1325_132576

theorem set_union_proof (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {1, 2^a})
  (hB : B = {a, b}) 
  (h_inter : A ∩ B = {1/4}) :
  A ∪ B = {-2, 1, 1/4} := 
by 
  sorry

end set_union_proof_l1325_132576


namespace reflected_light_eq_l1325_132541

theorem reflected_light_eq
  (incident_light : ∀ x y : ℝ, 2 * x - y + 6 = 0)
  (reflection_line : ∀ x y : ℝ, y = x) :
  ∃ x y : ℝ, x + 2 * y + 18 = 0 :=
sorry

end reflected_light_eq_l1325_132541


namespace solve_for_y_l1325_132583

-- The given condition as a hypothesis
variables {x y : ℝ}

-- The theorem statement
theorem solve_for_y (h : 3 * x - y + 5 = 0) : y = 3 * x + 5 :=
sorry

end solve_for_y_l1325_132583


namespace unoccupied_seats_in_business_class_l1325_132567

def airplane_seating (fc bc ec : ℕ) (pfc pbc pec : ℕ) : Nat :=
  let num_ec := ec / 2
  let num_bc := num_ec - pfc
  bc - num_bc

theorem unoccupied_seats_in_business_class :
  airplane_seating 10 30 50 3 (50/2) (50/2) = 8 := by
    sorry

end unoccupied_seats_in_business_class_l1325_132567


namespace toby_deleted_nine_bad_shots_l1325_132596

theorem toby_deleted_nine_bad_shots 
  (x : ℕ)
  (h1 : 63 > x)
  (h2 : (63 - x) + 15 - 3 = 84)
  : x = 9 :=
by
  sorry

end toby_deleted_nine_bad_shots_l1325_132596


namespace find_principal_l1325_132543

theorem find_principal (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) 
  (h_r : r = 0.20) 
  (h_t : t = 2) 
  (h_diff : CI - SI = 144) 
  (h_CI : CI = P * (1 + r)^t - P) 
  (h_SI : SI = P * r * t) : 
  P = 3600 :=
by
  sorry

end find_principal_l1325_132543
