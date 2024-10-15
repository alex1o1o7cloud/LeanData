import Mathlib

namespace NUMINAMATH_GPT_math_proof_problem_l1702_170283

theorem math_proof_problem : (10^8 / (2 * 10^5) - 50) = 450 := 
  by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l1702_170283


namespace NUMINAMATH_GPT_popsicle_sum_l1702_170265

-- Gino has 63 popsicle sticks
def gino_popsicle_sticks : Nat := 63

-- I have 50 popsicle sticks
def my_popsicle_sticks : Nat := 50

-- The sum of our popsicle sticks
def total_popsicle_sticks : Nat := gino_popsicle_sticks + my_popsicle_sticks

-- Prove that the total is 113
theorem popsicle_sum : total_popsicle_sticks = 113 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_popsicle_sum_l1702_170265


namespace NUMINAMATH_GPT_find_new_ratio_l1702_170259

def initial_ratio (H C : ℕ) : Prop := H = 6 * C

def transaction (H C : ℕ) : Prop :=
  H - 15 = (C + 15) + 70

def new_ratio (H C : ℕ) : Prop := H - 15 = 3 * (C + 15)

theorem find_new_ratio (H C : ℕ) (h1 : initial_ratio H C) (h2 : transaction H C) : 
  new_ratio H C :=
sorry

end NUMINAMATH_GPT_find_new_ratio_l1702_170259


namespace NUMINAMATH_GPT_vector_subtraction_l1702_170252

def vector_a : ℝ × ℝ := (3, 5)
def vector_b : ℝ × ℝ := (-2, 1)

theorem vector_subtraction :
  vector_a - 2 • vector_b = (7, 3) :=
sorry

end NUMINAMATH_GPT_vector_subtraction_l1702_170252


namespace NUMINAMATH_GPT_value_of_a_l1702_170286

def hyperbolaFociSharedEllipse : Prop :=
  ∃ a > 0, 
    (∃ c h k : ℝ, c = 3 ∧ (h, k) = (3, 0) ∨ (h, k) = (-3, 0)) ∧ 
    ∃ x y : ℝ, ((x^2) / 4) - ((y^2) / 5) = 1 ∧ ((x^2) / (a^2)) + ((y^2) / 16) = 1

theorem value_of_a : ∃ a > 0, hyperbolaFociSharedEllipse ∧ a = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1702_170286


namespace NUMINAMATH_GPT_find_r_in_arithmetic_sequence_l1702_170257

-- Define an arithmetic sequence
def is_arithmetic_sequence (a b c d e f : ℤ) : Prop :=
  (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d) ∧ (e - d = f - e)

-- Define the given problem
theorem find_r_in_arithmetic_sequence :
  ∃ r : ℤ, ∀ p q s : ℤ, is_arithmetic_sequence 23 p q r s 59 → r = 41 :=
by
  sorry

end NUMINAMATH_GPT_find_r_in_arithmetic_sequence_l1702_170257


namespace NUMINAMATH_GPT_arithmetic_mean_solution_l1702_170288

/-- Given the arithmetic mean of six expressions is 30, prove the values of x and y are as follows. -/
theorem arithmetic_mean_solution (x y : ℝ) (h : ((2 * x - y) + 20 + (3 * x + y) + 16 + (x + 5) + (y + 8)) / 6 = 30) (hy : y = 10) : 
  x = 18.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_mean_solution_l1702_170288


namespace NUMINAMATH_GPT_number_of_slices_per_package_l1702_170254

-- Define the problem's conditions
def packages_of_bread := 2
def slices_per_package_of_ham := 8
def packages_of_ham := 2
def leftover_slices_of_bread := 8
def total_ham_slices := packages_of_ham * slices_per_package_of_ham
def total_ham_required_bread := total_ham_slices * 2
def total_initial_bread_slices (B : ℕ) := packages_of_bread * B
def total_bread_used (B : ℕ) := total_ham_required_bread
def slices_leftover (B : ℕ) := total_initial_bread_slices B - total_bread_used B

-- Specify the goal
theorem number_of_slices_per_package (B : ℕ) (h : total_initial_bread_slices B = total_bread_used B + leftover_slices_of_bread) : B = 20 :=
by
  -- Use the provided conditions along with the hypothesis
  -- of the initial bread slices equation equating to used and leftover slices
  sorry

end NUMINAMATH_GPT_number_of_slices_per_package_l1702_170254


namespace NUMINAMATH_GPT_pascal_30th_31st_numbers_l1702_170201

-- Definitions based on conditions
def pascal_triangle_row_34 (k : ℕ) : ℕ := Nat.choose 34 k

-- Problem statement in Lean 4: proving the equations
theorem pascal_30th_31st_numbers :
  pascal_triangle_row_34 29 = 278256 ∧
  pascal_triangle_row_34 30 = 46376 :=
by
  sorry

end NUMINAMATH_GPT_pascal_30th_31st_numbers_l1702_170201


namespace NUMINAMATH_GPT_smallest_k_satisfying_condition_l1702_170258

def is_smallest_prime_greater_than (n : ℕ) (p : ℕ) : Prop :=
  Nat.Prime p ∧ n < p ∧ ∀ q, Nat.Prime q ∧ q > n → q >= p

def is_divisible_by (m k : ℕ) : Prop := k % m = 0

theorem smallest_k_satisfying_condition :
  ∃ k, is_smallest_prime_greater_than 19 23 ∧ is_divisible_by 3 k ∧ 64 ^ k > 4 ^ (19 * 23) ∧ (∀ k' < k, is_divisible_by 3 k' → 64 ^ k' ≤ 4 ^ (19 * 23)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_satisfying_condition_l1702_170258


namespace NUMINAMATH_GPT_pyr_sphere_ineq_l1702_170289

open Real

theorem pyr_sphere_ineq (h a : ℝ) (R r : ℝ) 
  (h_pos : h > 0) (a_pos : a > 0) 
  (pyr_in_sphere : ∀ h a : ℝ, R = (2*a^2 + h^2) / (2*h))
  (pyr_circ_sphere : ∀ h a : ℝ, r = (a * h) / (sqrt (h^2 + a^2) + a)) :
  R ≥ (sqrt 2 + 1) * r := 
sorry

end NUMINAMATH_GPT_pyr_sphere_ineq_l1702_170289


namespace NUMINAMATH_GPT_max_sum_a_b_c_l1702_170290

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

theorem max_sum_a_b_c (a b c : ℝ) (h : ∀ x : ℝ, f a b c x ≥ -1) : a + b + c ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_sum_a_b_c_l1702_170290


namespace NUMINAMATH_GPT_solve_inequality_l1702_170245

theorem solve_inequality (x : ℝ) : 
  (x - 2) / (x + 1) ≤ 2 ↔ x ∈ Set.Iic (-4) ∪ Set.Ioi (-1) := 
sorry

end NUMINAMATH_GPT_solve_inequality_l1702_170245


namespace NUMINAMATH_GPT_value_of_y_l1702_170253

theorem value_of_y (x y z : ℕ) (h_positive_x : 0 < x) (h_positive_y : 0 < y) (h_positive_z : 0 < z)
    (h_sum : x + y + z = 37) (h_eq : 4 * x = 6 * z) : y = 32 :=
sorry

end NUMINAMATH_GPT_value_of_y_l1702_170253


namespace NUMINAMATH_GPT_eval_expression_l1702_170213

theorem eval_expression : 5 * 7 + 9 * 4 - 36 / 3 = 59 :=
by sorry

end NUMINAMATH_GPT_eval_expression_l1702_170213


namespace NUMINAMATH_GPT_cistern_fill_time_l1702_170275

theorem cistern_fill_time (F E : ℝ) (hF : F = 1/3) (hE : E = 1/6) : (1 / (F - E)) = 6 :=
by sorry

end NUMINAMATH_GPT_cistern_fill_time_l1702_170275


namespace NUMINAMATH_GPT_smaller_base_length_trapezoid_l1702_170281

variable (p q a b : ℝ)
variable (h : p < q)
variable (angle_ratio : ∃ α, ((2 * α) : ℝ) = α + (α : ℝ))

theorem smaller_base_length_trapezoid :
  b = (p^2 + a * p - q^2) / p :=
sorry

end NUMINAMATH_GPT_smaller_base_length_trapezoid_l1702_170281


namespace NUMINAMATH_GPT_exists_disjoint_nonempty_subsets_with_equal_sum_l1702_170287

theorem exists_disjoint_nonempty_subsets_with_equal_sum :
  ∀ (A : Finset ℕ), (A.card = 11) → (∀ a ∈ A, 1 ≤ a ∧ a ≤ 100) →
  ∃ (B C : Finset ℕ), B ≠ ∅ ∧ C ≠ ∅ ∧ B ∩ C = ∅ ∧ (B ∪ C ⊆ A) ∧ (B.sum id = C.sum id) :=
by
  sorry

end NUMINAMATH_GPT_exists_disjoint_nonempty_subsets_with_equal_sum_l1702_170287


namespace NUMINAMATH_GPT_carmen_more_miles_l1702_170215

-- Definitions for the conditions
def carmen_distance : ℕ := 90
def daniel_distance : ℕ := 75

-- The theorem statement
theorem carmen_more_miles : carmen_distance - daniel_distance = 15 :=
by
  sorry

end NUMINAMATH_GPT_carmen_more_miles_l1702_170215


namespace NUMINAMATH_GPT_circle_center_sum_l1702_170262

theorem circle_center_sum (x y : ℝ) :
  (x^2 + y^2 = 10*x - 12*y + 40) →
  x + y = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_center_sum_l1702_170262


namespace NUMINAMATH_GPT_free_cytosine_molecules_req_l1702_170227

-- Definition of conditions
def DNA_base_pairs := 500
def AT_percentage := 34 / 100
def CG_percentage := 1 - AT_percentage

-- The total number of bases
def total_bases := 2 * DNA_base_pairs

-- The number of C or G bases
def CG_bases := total_bases * CG_percentage

-- Finally, the total number of free cytosine deoxyribonucleotide molecules 
def free_cytosine_molecules := 2 * CG_bases

-- Problem statement: Prove that the number of free cytosine deoxyribonucleotide molecules required is 1320
theorem free_cytosine_molecules_req : free_cytosine_molecules = 1320 :=
by
  -- conditions are defined, the proof is omitted
  sorry

end NUMINAMATH_GPT_free_cytosine_molecules_req_l1702_170227


namespace NUMINAMATH_GPT_f_2012_eq_3_l1702_170233

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem f_2012_eq_3 
  (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) 
  (h : f a b α β 2011 = 5) : 
  f a b α β 2012 = 3 :=
by
  sorry

end NUMINAMATH_GPT_f_2012_eq_3_l1702_170233


namespace NUMINAMATH_GPT_rocco_piles_of_quarters_proof_l1702_170237

-- Define the value of a pile of different types of coins
def pile_value (coin_value : ℕ) (num_coins_in_pile : ℕ) : ℕ :=
  coin_value * num_coins_in_pile

-- Define the number of piles for different coins
def num_piles_of_dimes : ℕ := 6
def num_piles_of_nickels : ℕ := 9
def num_piles_of_pennies : ℕ := 5
def num_coins_in_pile : ℕ := 10

-- Define the total value of each type of coin
def value_of_a_dime : ℕ := 10  -- in cents
def value_of_a_nickel : ℕ := 5  -- in cents
def value_of_a_penny : ℕ := 1  -- in cents
def value_of_a_quarter : ℕ := 25  -- in cents

-- Define the total money Rocco has in cents
def total_money : ℕ := 2100  -- since $21 = 2100 cents

-- Calculate the value of all piles of each type of coin
def total_dimes_value : ℕ := num_piles_of_dimes * (pile_value value_of_a_dime num_coins_in_pile)
def total_nickels_value : ℕ := num_piles_of_nickels * (pile_value value_of_a_nickel num_coins_in_pile)
def total_pennies_value : ℕ := num_piles_of_pennies * (pile_value value_of_a_penny num_coins_in_pile)

-- Calculate the value of the quarters
def value_of_quarters : ℕ := total_money - (total_dimes_value + total_nickels_value + total_pennies_value)
def num_piles_of_quarters : ℕ := value_of_quarters / 250 -- since each pile of quarters is worth 250 cents

-- Theorem to prove
theorem rocco_piles_of_quarters_proof : num_piles_of_quarters = 4 := by
  sorry

end NUMINAMATH_GPT_rocco_piles_of_quarters_proof_l1702_170237


namespace NUMINAMATH_GPT_odd_power_divisible_by_sum_l1702_170291

theorem odd_power_divisible_by_sum (x y : ℝ) (k : ℕ) (h : k > 0) :
  (x^((2*k - 1)) + y^((2*k - 1))) ∣ (x^(2*k + 1) + y^(2*k + 1)) :=
sorry

end NUMINAMATH_GPT_odd_power_divisible_by_sum_l1702_170291


namespace NUMINAMATH_GPT_tree_height_l1702_170240

theorem tree_height (future_height : ℕ) (growth_per_year : ℕ) (years : ℕ) (inches_per_foot : ℕ) :
  future_height = 1104 →
  growth_per_year = 5 →
  years = 8 →
  inches_per_foot = 12 →
  (future_height / inches_per_foot - growth_per_year * years) = 52 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_tree_height_l1702_170240


namespace NUMINAMATH_GPT_find_b_and_c_find_b_with_c_range_of_b_l1702_170210

-- Part (Ⅰ)
theorem find_b_and_c (b c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 + 2 * b * x + c)
  (h_zeros : f (-1) = 0 ∧ f 1 = 0) : b = 0 ∧ c = -1 := sorry

-- Part (Ⅱ)
theorem find_b_with_c (b : ℝ) (f : ℝ → ℝ)
  (x1 x2 : ℝ) 
  (h_def : ∀ x, f x = x^2 + 2 * b * x + (b^2 + 2 * b + 3))
  (h_eq : (x1 + 1) * (x2 + 1) = 8) 
  (h_roots : f x1 = 0 ∧ f x2 = 0) : b = -2 := sorry

-- Part (Ⅲ)
theorem range_of_b (b : ℝ) (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_f_def : ∀ x, f x = x^2 + 2 * b * x + (-1 - 2 * b))
  (h_f_1 : f 1 = 0)
  (h_g_def : ∀ x, g x = f x + x + b)
  (h_intervals : ∀ x, 
    ((-3 < x) ∧ (x < -2) → g x > 0) ∧
    ((-2 < x) ∧ (x < 0) → g x < 0) ∧
    ((0 < x) ∧ (x < 1) → g x < 0) ∧
    ((1 < x) → g x > 0)) : (1/5) < b ∧ b < (5/7) := sorry

end NUMINAMATH_GPT_find_b_and_c_find_b_with_c_range_of_b_l1702_170210


namespace NUMINAMATH_GPT_train_speed_l1702_170272

theorem train_speed (length_of_train time_to_cross : ℝ) (h_length : length_of_train = 800) (h_time : time_to_cross = 12) : (length_of_train / time_to_cross) = 66.67 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1702_170272


namespace NUMINAMATH_GPT_max_marks_tests_l1702_170248

theorem max_marks_tests :
  ∃ (T1 T2 T3 T4 : ℝ),
    0.30 * T1 = 80 + 40 ∧
    0.40 * T2 = 105 + 35 ∧
    0.50 * T3 = 150 + 50 ∧
    0.60 * T4 = 180 + 60 ∧
    T1 = 400 ∧
    T2 = 350 ∧
    T3 = 400 ∧
    T4 = 400 :=
by
    sorry

end NUMINAMATH_GPT_max_marks_tests_l1702_170248


namespace NUMINAMATH_GPT_two_digit_number_exists_l1702_170274

theorem two_digit_number_exists (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) :
  (9 * x + 8) * (80 - 9 * x) = 1855 → (9 * x + 8 = 35 ∨ 9 * x + 8 = 53) := by
  sorry

end NUMINAMATH_GPT_two_digit_number_exists_l1702_170274


namespace NUMINAMATH_GPT_sum_of_first_3_geometric_terms_eq_7_l1702_170282

theorem sum_of_first_3_geometric_terms_eq_7 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_ratio_gt_1 : r > 1)
  (h_eq : (a 0 + a 2 = 5) ∧ (a 0 * a 2 = 4)) 
  : (a 0 + a 1 + a 2) = 7 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_first_3_geometric_terms_eq_7_l1702_170282


namespace NUMINAMATH_GPT_inequality_for_positive_real_numbers_l1702_170206

theorem inequality_for_positive_real_numbers (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^2 + 2*y^2 + 3*z^2 > x*y + 3*y*z + z*x := 
by 
  sorry

end NUMINAMATH_GPT_inequality_for_positive_real_numbers_l1702_170206


namespace NUMINAMATH_GPT_remaining_pens_l1702_170239

theorem remaining_pens (blue_initial black_initial red_initial green_initial purple_initial : ℕ)
                        (blue_removed black_removed red_removed green_removed purple_removed : ℕ) :
  blue_initial = 15 → black_initial = 27 → red_initial = 12 → green_initial = 10 → purple_initial = 8 →
  blue_removed = 8 → black_removed = 9 → red_removed = 3 → green_removed = 5 → purple_removed = 6 →
  blue_initial - blue_removed + black_initial - black_removed + red_initial - red_removed +
  green_initial - green_removed + purple_initial - purple_removed = 41 :=
by
  intros
  sorry

end NUMINAMATH_GPT_remaining_pens_l1702_170239


namespace NUMINAMATH_GPT_parabola_x_intercepts_count_l1702_170280

theorem parabola_x_intercepts_count :
  ∃! x, ∃ y, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_parabola_x_intercepts_count_l1702_170280


namespace NUMINAMATH_GPT_domain_of_f_l1702_170231

noncomputable def f (x : ℝ) := (Real.sqrt (x + 3)) / x

theorem domain_of_f :
  { x : ℝ | x ≥ -3 ∧ x ≠ 0 } = { x : ℝ | ∃ y, f y ≠ 0 } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1702_170231


namespace NUMINAMATH_GPT_original_game_start_player_wins_modified_game_start_player_wins_l1702_170268

def divisor_game_condition (num : ℕ) := ∀ d : ℕ, d ∣ num → ∀ x : ℕ, x ∣ d → x = d ∨ x = 1
def modified_divisor_game_condition (num d_prev : ℕ) := ∀ d : ℕ, d ∣ num → d ≠ d_prev → ∃ k l : ℕ, d = k * l ∧ k ≠ 1 ∧ l ≠ 1 ∧ k ≤ l

/-- Prove that if the starting player plays wisely, they will always win the original game. -/
theorem original_game_start_player_wins : ∀ d : ℕ, divisor_game_condition 1000 → d = 100 → (∃ p : ℕ, p != 1000) := 
sorry

/-- What happens if the game is modified such that a divisor cannot be mentioned if it has fewer divisors than any previously mentioned number? -/
theorem modified_game_start_player_wins : ∀ d_prev : ℕ, modified_divisor_game_condition 1000 d_prev → d_prev = 100 → (∃ p : ℕ, p != 1000) := 
sorry

end NUMINAMATH_GPT_original_game_start_player_wins_modified_game_start_player_wins_l1702_170268


namespace NUMINAMATH_GPT_scientific_notation_of_15510000_l1702_170298

/--
Express 15,510,000 in scientific notation.

Theorem: 
Given that the scientific notation for large numbers is of the form \(a \times 10^n\) where \(1 \leq |a| < 10\),
prove that expressing 15,510,000 in scientific notation results in 1.551 × 10^7.
-/
theorem scientific_notation_of_15510000 : ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 15510000 = a * 10 ^ n ∧ a = 1.551 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_15510000_l1702_170298


namespace NUMINAMATH_GPT_gravel_weight_40_pounds_l1702_170221

def weight_of_gravel_in_mixture (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) : ℝ :=
total_weight - (sand_fraction * total_weight + water_fraction * total_weight)

theorem gravel_weight_40_pounds
  (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) 
  (h1 : total_weight = 40) (h2 : sand_fraction = 1 / 4) (h3 : water_fraction = 2 / 5) :
  weight_of_gravel_in_mixture total_weight sand_fraction water_fraction = 14 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_gravel_weight_40_pounds_l1702_170221


namespace NUMINAMATH_GPT_star_3_5_l1702_170209

def star (a b : ℕ) : ℕ := a^2 + 3 * a * b + b^2

theorem star_3_5 : star 3 5 = 79 := 
by
  sorry

end NUMINAMATH_GPT_star_3_5_l1702_170209


namespace NUMINAMATH_GPT_number_of_technicians_l1702_170203

-- Define the problem statements
variables (T R : ℕ)

-- Conditions based on the problem description
def condition1 : Prop := T + R = 42
def condition2 : Prop := 3 * T + R = 56

-- The main goal to prove
theorem number_of_technicians (h1 : condition1 T R) (h2 : condition2 T R) : T = 7 :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_GPT_number_of_technicians_l1702_170203


namespace NUMINAMATH_GPT_Vasya_fraction_impossible_l1702_170249

theorem Vasya_fraction_impossible
  (a b n : ℕ) (h_ab : a < b) (h_na : n < a) (h_nb : n < b)
  (h1 : (a + n) / (b + n) > 3 * a / (2 * b))
  (h2 : (a - n) / (b - n) > a / (2 * b)) : false :=
by
  sorry

end NUMINAMATH_GPT_Vasya_fraction_impossible_l1702_170249


namespace NUMINAMATH_GPT_hannahs_vegetarian_restaurant_l1702_170212

theorem hannahs_vegetarian_restaurant :
  let total_weight_of_peppers := 0.6666666666666666
  let weight_of_green_peppers := 0.3333333333333333
  total_weight_of_peppers - weight_of_green_peppers = 0.3333333333333333 :=
by
  sorry

end NUMINAMATH_GPT_hannahs_vegetarian_restaurant_l1702_170212


namespace NUMINAMATH_GPT_simplify_fraction_eq_one_over_thirty_nine_l1702_170270

theorem simplify_fraction_eq_one_over_thirty_nine :
  let a1 := (1 / 3)^1
  let a2 := (1 / 3)^2
  let a3 := (1 / 3)^3
  (1 / (1 / a1 + 1 / a2 + 1 / a3)) = 1 / 39 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_eq_one_over_thirty_nine_l1702_170270


namespace NUMINAMATH_GPT_carl_sold_each_watermelon_for_3_l1702_170235

def profit : ℕ := 105
def final_watermelons : ℕ := 18
def starting_watermelons : ℕ := 53
def sold_watermelons : ℕ := starting_watermelons - final_watermelons
def price_per_watermelon : ℕ := profit / sold_watermelons

theorem carl_sold_each_watermelon_for_3 :
  price_per_watermelon = 3 :=
by
  sorry

end NUMINAMATH_GPT_carl_sold_each_watermelon_for_3_l1702_170235


namespace NUMINAMATH_GPT_integer_root_of_quadratic_eq_l1702_170226

theorem integer_root_of_quadratic_eq (m : ℤ) (hm : ∃ x : ℤ, m * x^2 + 2 * (m - 5) * x + (m - 4) = 0) : m = -4 ∨ m = 4 ∨ m = -16 :=
sorry

end NUMINAMATH_GPT_integer_root_of_quadratic_eq_l1702_170226


namespace NUMINAMATH_GPT_union_complement_l1702_170218

def universalSet : Set ℤ := { x | x^2 < 9 }

def A : Set ℤ := {1, 2}

def B : Set ℤ := {-2, -1, 2}

def complement_I_B : Set ℤ := universalSet \ B

theorem union_complement :
  A ∪ complement_I_B = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_union_complement_l1702_170218


namespace NUMINAMATH_GPT_tablecloth_width_l1702_170219

theorem tablecloth_width (length_tablecloth : ℕ) (napkins_count : ℕ) (napkin_length : ℕ) (napkin_width : ℕ) (total_material : ℕ) (width_tablecloth : ℕ) :
  length_tablecloth = 102 →
  napkins_count = 8 →
  napkin_length = 6 →
  napkin_width = 7 →
  total_material = 5844 →
  total_material = length_tablecloth * width_tablecloth + napkins_count * (napkin_length * napkin_width) →
  width_tablecloth = 54 :=
by
  intros h1 h2 h3 h4 h5 h_eq
  sorry

end NUMINAMATH_GPT_tablecloth_width_l1702_170219


namespace NUMINAMATH_GPT_solve_r_l1702_170241

def E (a : ℝ) (b : ℝ) (c : ℕ) : ℝ := a * b^c

theorem solve_r : ∃ (r : ℝ), E r r 5 = 1024 ∧ r = 2^(5/3) :=
by
  sorry

end NUMINAMATH_GPT_solve_r_l1702_170241


namespace NUMINAMATH_GPT_alternating_sequence_probability_l1702_170220

theorem alternating_sequence_probability : 
  let total_balls := 10 -- Total number of balls
  let white_balls := 5 -- Number of white balls
  let black_balls := 5 -- Number of black balls
  let successful_sequences := 2 -- Number of successful alternating sequences (BWBWBWBWBW and WBWBWBWBWB)
  let total_arrangements := Nat.choose total_balls white_balls -- Binomial coefficient for total arrangements
  (successful_sequences : ℚ) / total_arrangements = 1 / 126 :=
by
  sorry

end NUMINAMATH_GPT_alternating_sequence_probability_l1702_170220


namespace NUMINAMATH_GPT_trapezoid_area_l1702_170246

theorem trapezoid_area (AD BC AC : ℝ) (BD : ℝ) 
  (hAD : AD = 24) 
  (hBC : BC = 8) 
  (hAC : AC = 13) 
  (hBD : BD = 5 * Real.sqrt 17) : 
  (1 / 2 * (AD + BC) * Real.sqrt (AC^2 - (BC + (AD - BC) / 2)^2)) = 80 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1702_170246


namespace NUMINAMATH_GPT_price_reduction_is_not_10_yuan_l1702_170230

theorem price_reduction_is_not_10_yuan (current_price original_price : ℝ)
  (CurrentPrice : current_price = 45)
  (Reduction : current_price = 0.9 * original_price)
  (TenPercentReduction : 0.1 * original_price = 10) :
  (original_price - current_price) ≠ 10 := by
  sorry

end NUMINAMATH_GPT_price_reduction_is_not_10_yuan_l1702_170230


namespace NUMINAMATH_GPT_find_a_value_l1702_170247

theorem find_a_value (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53^2017 + a) % 13 = 0) : a = 12 :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_find_a_value_l1702_170247


namespace NUMINAMATH_GPT_calculate_expression_l1702_170229

theorem calculate_expression :
  5 * Real.sqrt 3 + (Real.sqrt 4 + 2 * Real.sqrt 3) = 7 * Real.sqrt 3 + 2 :=
by sorry

end NUMINAMATH_GPT_calculate_expression_l1702_170229


namespace NUMINAMATH_GPT_prime_between_30_40_with_remainder_l1702_170207

theorem prime_between_30_40_with_remainder :
  ∃ n : ℕ, Prime n ∧ 30 < n ∧ n < 40 ∧ n % 9 = 4 ∧ n = 31 :=
by
  sorry

end NUMINAMATH_GPT_prime_between_30_40_with_remainder_l1702_170207


namespace NUMINAMATH_GPT_last_day_of_third_quarter_l1702_170222

def is_common_year (year: Nat) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0) 

def days_in_month (year: Nat) (month: Nat) : Nat :=
  if month = 2 then 28
  else if month = 4 ∨ month = 6 ∨ month = 9 ∨ month = 11 then 30
  else 31

def last_day_of_month (year: Nat) (month: Nat) : Nat :=
  days_in_month year month

theorem last_day_of_third_quarter (year: Nat) (h : is_common_year year) : last_day_of_month year 9 = 30 :=
by
  sorry

end NUMINAMATH_GPT_last_day_of_third_quarter_l1702_170222


namespace NUMINAMATH_GPT_sum_divisible_by_ten_l1702_170269

    -- Given conditions
    def is_natural_number (n : ℕ) : Prop := true

    -- Sum S as defined in the conditions
    def S (n : ℕ) : ℕ := n ^ 2 + (n + 1) ^ 2 + (n + 2) ^ 2 + (n + 3) ^ 2

    -- The equivalent math proof problem in Lean 4 statement
    theorem sum_divisible_by_ten (n : ℕ) : S n % 10 = 0 ↔ n % 5 = 1 := by
      sorry
    
end NUMINAMATH_GPT_sum_divisible_by_ten_l1702_170269


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1702_170264

variable (a : ℕ → ℝ)
variable (d : ℝ)

noncomputable def arithmetic_sequence := ∀ n : ℕ, a n = a 0 + n * d

theorem arithmetic_sequence_sum (h₁ : a 1 + a 2 = 3) (h₂ : a 3 + a 4 = 5) :
  a 7 + a 8 = 9 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1702_170264


namespace NUMINAMATH_GPT_shifted_parabola_eq_l1702_170205

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := -(x^2)

-- Define the transformation for shifting left 2 units
def shift_left (x : ℝ) : ℝ := x + 2

-- Define the transformation for shifting down 3 units
def shift_down (y : ℝ) : ℝ := y - 3

-- Define the new parabola equation after shifting
def new_parabola (x : ℝ) : ℝ := shift_down (original_parabola (shift_left x))

-- The theorem to be proven
theorem shifted_parabola_eq : new_parabola x = -(x + 2)^2 - 3 := by
  sorry

end NUMINAMATH_GPT_shifted_parabola_eq_l1702_170205


namespace NUMINAMATH_GPT_compare_powers_l1702_170225

theorem compare_powers :
  100^100 > 50^50 * 150^50 := sorry

end NUMINAMATH_GPT_compare_powers_l1702_170225


namespace NUMINAMATH_GPT_symmedian_length_l1702_170216

theorem symmedian_length (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ AS : ℝ, AS = (b * c^2 / (b^2 + c^2)) * (Real.sqrt (2 * b^2 + 2 * c^2 - a^2)) :=
sorry

end NUMINAMATH_GPT_symmedian_length_l1702_170216


namespace NUMINAMATH_GPT_parabola_points_l1702_170211

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_points :
  ∃ (a c m n : ℝ),
  a = 2 ∧ c = -2 ∧
  parabola a 1 c 2 = m ∧
  parabola a 1 c n = -2 ∧
  m = 8 ∧
  n = -1 / 2 :=
by
  use 2, -2, 8, -1/2
  simp [parabola]
  sorry

end NUMINAMATH_GPT_parabola_points_l1702_170211


namespace NUMINAMATH_GPT_trigonometric_identity_l1702_170256

theorem trigonometric_identity
  (θ : ℝ)
  (h1 : θ > -π/2)
  (h2 : θ < 0)
  (h3 : Real.tan θ = -2) :
  (Real.sin θ)^2 / (Real.cos (2 * θ) + 2) = 4 / 7 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1702_170256


namespace NUMINAMATH_GPT_f_1982_l1702_170234

-- Define the function f and the essential properties and conditions
def f : ℕ → ℕ := sorry

axiom f_nonneg (n : ℕ) : f n ≥ 0
axiom f_add_property (m n : ℕ) : f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1
axiom f_2 : f 2 = 0
axiom f_3_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

-- Statement of the theorem we want to prove
theorem f_1982 : f 1982 = 660 := 
  by sorry

end NUMINAMATH_GPT_f_1982_l1702_170234


namespace NUMINAMATH_GPT_number_of_negative_x_l1702_170214

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end NUMINAMATH_GPT_number_of_negative_x_l1702_170214


namespace NUMINAMATH_GPT_sum_f_values_l1702_170200

noncomputable def f (x : ℤ) : ℤ := (x - 1)^3 + 1

theorem sum_f_values :
  (f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7) = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_f_values_l1702_170200


namespace NUMINAMATH_GPT_general_formula_l1702_170223

def sequence_a (n : ℕ) : ℕ :=
by sorry

def partial_sum (n : ℕ) : ℕ :=
by sorry

axiom base_case : partial_sum 1 = 5

axiom recurrence_relation (n : ℕ) (h : 2 ≤ n) : partial_sum (n - 1) = sequence_a n

theorem general_formula (n : ℕ) : partial_sum n = 5 * 2^(n-1) :=
by
-- Proof will be provided here
sorry

end NUMINAMATH_GPT_general_formula_l1702_170223


namespace NUMINAMATH_GPT_total_courses_attended_l1702_170263

-- Define the number of courses attended by Max
def maxCourses : ℕ := 40

-- Define the number of courses attended by Sid (four times as many as Max)
def sidCourses : ℕ := 4 * maxCourses

-- Define the total number of courses attended by both Max and Sid
def totalCourses : ℕ := maxCourses + sidCourses

-- The proof statement
theorem total_courses_attended : totalCourses = 200 := by
  sorry

end NUMINAMATH_GPT_total_courses_attended_l1702_170263


namespace NUMINAMATH_GPT_original_numbers_correct_l1702_170242

noncomputable def restore_original_numbers : List ℕ :=
  let T : ℕ := 5
  let EL : ℕ := 12
  let EK : ℕ := 19
  let LA : ℕ := 26
  let SS : ℕ := 33
  [T, EL, EK, LA, SS]

theorem original_numbers_correct :
  restore_original_numbers = [5, 12, 19, 26, 33] :=
by
  sorry

end NUMINAMATH_GPT_original_numbers_correct_l1702_170242


namespace NUMINAMATH_GPT_rowing_distance_correct_l1702_170224

variable (D : ℝ) -- distance to the place
variable (speed_in_still_water : ℝ := 10) -- rowing speed in still water
variable (current_speed : ℝ := 2) -- speed of the current
variable (total_time : ℝ := 30) -- total time for round trip
variable (effective_speed_with_current : ℝ := speed_in_still_water + current_speed) -- effective speed with current
variable (effective_speed_against_current : ℝ := speed_in_still_water - current_speed) -- effective speed against current

theorem rowing_distance_correct : 
  D / effective_speed_with_current + D / effective_speed_against_current = total_time → 
  D = 144 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_rowing_distance_correct_l1702_170224


namespace NUMINAMATH_GPT_breadth_of_boat_l1702_170204

theorem breadth_of_boat
  (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (ρ : ℝ) (B : ℝ)
  (hL : L = 3)
  (hh : h = 0.01)
  (hm : m = 60)
  (hg : g = 9.81)
  (hρ : ρ = 1000) :
  B = 2 := by
  sorry

end NUMINAMATH_GPT_breadth_of_boat_l1702_170204


namespace NUMINAMATH_GPT_Dawn_sold_glasses_l1702_170279

variable (x : ℕ)

def Bea_price_per_glass : ℝ := 0.25
def Dawn_price_per_glass : ℝ := 0.28
def Bea_glasses_sold : ℕ := 10
def Bea_extra_earnings : ℝ := 0.26
def Bea_total_earnings : ℝ := Bea_glasses_sold * Bea_price_per_glass
def Dawn_total_earnings (x : ℕ) : ℝ := x * Dawn_price_per_glass

theorem Dawn_sold_glasses :
  Bea_total_earnings - Bea_extra_earnings = Dawn_total_earnings x → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_Dawn_sold_glasses_l1702_170279


namespace NUMINAMATH_GPT_blocks_for_fort_l1702_170296

theorem blocks_for_fort :
  let length := 15 
  let width := 12 
  let height := 6
  let thickness := 1
  let V_original := length * width * height
  let interior_length := length - 2 * thickness
  let interior_width := width - 2 * thickness
  let interior_height := height - thickness
  let V_interior := interior_length * interior_width * interior_height
  let V_blocks := V_original - V_interior
  V_blocks = 430 :=
by
  sorry

end NUMINAMATH_GPT_blocks_for_fort_l1702_170296


namespace NUMINAMATH_GPT_initial_average_mark_l1702_170297

theorem initial_average_mark (A : ℝ) (n : ℕ) (excluded_avg remaining_avg : ℝ) :
  n = 25 →
  excluded_avg = 40 →
  remaining_avg = 90 →
  (A * n = (n - 5) * remaining_avg + 5 * excluded_avg) →
  A = 80 :=
by
  intros hn_hexcluded_avg hremaining_avg htotal_correct
  sorry

end NUMINAMATH_GPT_initial_average_mark_l1702_170297


namespace NUMINAMATH_GPT_ordered_triple_l1702_170267

theorem ordered_triple (a b c : ℝ) (h1 : 4 < a) (h2 : 4 < b) (h3 : 4 < c) 
  (h_eq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) 
  : (a, b, c) = (12, 10, 8) :=
  sorry

end NUMINAMATH_GPT_ordered_triple_l1702_170267


namespace NUMINAMATH_GPT_lollipop_cases_l1702_170261

theorem lollipop_cases (total_cases : ℕ) (chocolate_cases : ℕ) (lollipop_cases : ℕ) 
  (h1 : total_cases = 80) (h2 : chocolate_cases = 25) : lollipop_cases = 55 :=
by
  sorry

end NUMINAMATH_GPT_lollipop_cases_l1702_170261


namespace NUMINAMATH_GPT_time_to_cover_escalator_l1702_170217

def escalator_speed : ℝ := 12
def escalator_length : ℝ := 160
def person_speed : ℝ := 8

theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed)) = 8 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_escalator_l1702_170217


namespace NUMINAMATH_GPT_sampling_interval_is_100_l1702_170276

-- Define the total number of numbers (N), the number of samples to be taken (k), and the condition for systematic sampling.
def N : ℕ := 2005
def k : ℕ := 20

-- Define the concept of systematic sampling interval
def sampling_interval (N k : ℕ) : ℕ := N / k

-- The proof that the sampling interval is 100 when 2005 numbers are sampled as per the systematic sampling method.
theorem sampling_interval_is_100 (N k : ℕ) 
  (hN : N = 2005) 
  (hk : k = 20) 
  (h1 : N % k ≠ 0) : 
  sampling_interval (N - (N % k)) k = 100 :=
by
  -- Initialization
  sorry

end NUMINAMATH_GPT_sampling_interval_is_100_l1702_170276


namespace NUMINAMATH_GPT_dessert_distribution_l1702_170260

theorem dessert_distribution 
  (mini_cupcakes : ℕ) 
  (donut_holes : ℕ) 
  (total_desserts : ℕ) 
  (students : ℕ) 
  (h1 : mini_cupcakes = 14)
  (h2 : donut_holes = 12) 
  (h3 : students = 13)
  (h4 : total_desserts = mini_cupcakes + donut_holes)
  : total_desserts / students = 2 :=
by sorry

end NUMINAMATH_GPT_dessert_distribution_l1702_170260


namespace NUMINAMATH_GPT_geometric_number_difference_l1702_170284

theorem geometric_number_difference : 
  ∀ (a b c : ℕ), 8 = a → b ≠ c → (∃ k : ℕ, 8 ≠ k ∧ b = k ∧ c = k * k / 8) → (10^2 * a + 10 * b + c = 842) ∧ (10^2 * a + 10 * b + c = 842) → (10^2 * a + 10 * b + c) - (10^2 * a + 10 * b + c) = 0 :=
by
  intro a b c
  intro ha hb
  intro hk
  intro hseq
  sorry

end NUMINAMATH_GPT_geometric_number_difference_l1702_170284


namespace NUMINAMATH_GPT_sheep_count_l1702_170285

theorem sheep_count (cows sheep shepherds : ℕ) 
  (h_cows : cows = 12) 
  (h_ears : 2 * cows < sheep) 
  (h_legs : sheep < 4 * cows) 
  (h_shepherds : sheep = 12 * shepherds) :
  sheep = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_sheep_count_l1702_170285


namespace NUMINAMATH_GPT_kelly_needs_more_apples_l1702_170273

theorem kelly_needs_more_apples (initial_apples : ℕ) (total_apples : ℕ) (needed_apples : ℕ) :
  initial_apples = 128 → total_apples = 250 → needed_apples = total_apples - initial_apples → needed_apples = 122 :=
by
  intros h_initial h_total h_needed
  rw [h_initial, h_total] at h_needed
  exact h_needed

end NUMINAMATH_GPT_kelly_needs_more_apples_l1702_170273


namespace NUMINAMATH_GPT_projection_of_a_onto_b_is_three_l1702_170294

def vec_a : ℝ × ℝ := (3, 4)
def vec_b : ℝ × ℝ := (1, 0)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def projection (u v : ℝ × ℝ) : ℝ := dot_product u v / magnitude v

theorem projection_of_a_onto_b_is_three : projection vec_a vec_b = 3 := by
  sorry

end NUMINAMATH_GPT_projection_of_a_onto_b_is_three_l1702_170294


namespace NUMINAMATH_GPT_magnitude_2a_sub_b_l1702_170232

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-1, 2)

theorem magnitude_2a_sub_b : (‖(2 * a.1 - b.1, 2 * a.2 - b.2)‖ = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_magnitude_2a_sub_b_l1702_170232


namespace NUMINAMATH_GPT_simplify_eval_expression_l1702_170243

theorem simplify_eval_expression (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ((a^2 - 1) / (a^2 - 2 * a + 1) - 1 / (1 - a)) / (1 / (a^2 - a)) = 1 :=
  sorry

end NUMINAMATH_GPT_simplify_eval_expression_l1702_170243


namespace NUMINAMATH_GPT_afternoon_registration_l1702_170293

variable (m a t morning_absent : ℕ)

theorem afternoon_registration (m a t morning_absent afternoon : ℕ) (h1 : m = 25) (h2 : a = 4) (h3 : t = 42) (h4 : morning_absent = 3) : 
  afternoon = t - (m - morning_absent + morning_absent + a) :=
by sorry

end NUMINAMATH_GPT_afternoon_registration_l1702_170293


namespace NUMINAMATH_GPT_mrs_lovely_class_l1702_170255

-- Define the number of students in Mrs. Lovely's class
def number_of_students (g b : ℕ) : ℕ := g + b

theorem mrs_lovely_class (g b : ℕ): 
  (b = g + 3) →
  (500 - 10 = g * g + b * b) →
  number_of_students g b = 23 :=
by
  sorry

end NUMINAMATH_GPT_mrs_lovely_class_l1702_170255


namespace NUMINAMATH_GPT_number_of_chocolates_bought_l1702_170277

theorem number_of_chocolates_bought (C S : ℝ) 
  (h1 : ∃ n : ℕ, n * C = 21 * S) 
  (h2 : (S - C) / C * 100 = 66.67) : 
  ∃ n : ℕ, n = 35 := 
by
  sorry

end NUMINAMATH_GPT_number_of_chocolates_bought_l1702_170277


namespace NUMINAMATH_GPT_tank_capacity_l1702_170299

theorem tank_capacity (T : ℝ) (h1 : 0.6 * T = 0.7 * T - 45) : T = 450 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l1702_170299


namespace NUMINAMATH_GPT_part1_part2_part3_l1702_170202

theorem part1 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0 ↔ x < -3 ∨ x > -2) : k = -2/5 :=
sorry

theorem part2 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) : k < -Real.sqrt 6 / 6 :=
sorry

theorem part3 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, ¬ (k * x^2 - 2 * x + 6 * k < 0)) : k ≥ Real.sqrt 6 / 6 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1702_170202


namespace NUMINAMATH_GPT_max_f_theta_l1702_170266

noncomputable def determinant (a b c d : ℝ) : ℝ := a*d - b*c

noncomputable def f (θ : ℝ) : ℝ :=
  determinant (Real.sin θ) (Real.cos θ) (-1) 1

theorem max_f_theta :
  ∀ θ : ℝ, 0 < θ ∧ θ < (Real.pi / 3) →
  f θ ≤ (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_f_theta_l1702_170266


namespace NUMINAMATH_GPT_find_multiple_of_t_l1702_170251

theorem find_multiple_of_t (k t x y : ℝ) (h1 : x = 1 - k * t) (h2 : y = 2 * t - 2) :
  t = 0.5 → x = y → k = 4 :=
by
  intros ht hxy
  sorry

end NUMINAMATH_GPT_find_multiple_of_t_l1702_170251


namespace NUMINAMATH_GPT_second_term_of_geometric_series_l1702_170238

noncomputable def geometric_series_second_term (a r : ℝ) (S : ℝ) : ℝ :=
a * r

theorem second_term_of_geometric_series 
  (a r S : ℝ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 10) 
  (h3 : S = a / (1 - r)) 
  : geometric_series_second_term a r S = 1.875 :=
by
  sorry

end NUMINAMATH_GPT_second_term_of_geometric_series_l1702_170238


namespace NUMINAMATH_GPT_farmer_pomelos_dozen_l1702_170271

theorem farmer_pomelos_dozen (pomelos_last_week : ℕ) (boxes_last_week : ℕ) (boxes_this_week : ℕ) :
  pomelos_last_week = 240 → boxes_last_week = 10 → boxes_this_week = 20 →
  (pomelos_last_week / boxes_last_week) * boxes_this_week / 12 = 40 := 
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_farmer_pomelos_dozen_l1702_170271


namespace NUMINAMATH_GPT_valerie_needs_72_stamps_l1702_170295

noncomputable def total_stamps_needed : ℕ :=
  let thank_you_cards := 5
  let stamps_per_thank_you := 2
  let water_bill_stamps := 3
  let electric_bill_stamps := 2
  let internet_bill_stamps := 5
  let rebates_more_than_bills := 3
  let rebate_stamps := 2
  let job_applications_factor := 2
  let job_application_stamps := 1

  let total_thank_you_stamps := thank_you_cards * stamps_per_thank_you
  let total_bill_stamps := water_bill_stamps + electric_bill_stamps + internet_bill_stamps
  let total_rebates := total_bill_stamps + rebates_more_than_bills
  let total_rebate_stamps := total_rebates * rebate_stamps
  let total_job_applications := total_rebates * job_applications_factor
  let total_job_application_stamps := total_job_applications * job_application_stamps

  total_thank_you_stamps + total_bill_stamps + total_rebate_stamps + total_job_application_stamps

theorem valerie_needs_72_stamps : total_stamps_needed = 72 :=
  by
    sorry

end NUMINAMATH_GPT_valerie_needs_72_stamps_l1702_170295


namespace NUMINAMATH_GPT_coeff_sum_eq_twenty_l1702_170208

theorem coeff_sum_eq_twenty 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h : ((2 * x - 3) ^ 5) = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) :
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ = 20 :=
by
  sorry

end NUMINAMATH_GPT_coeff_sum_eq_twenty_l1702_170208


namespace NUMINAMATH_GPT_scientific_notation_of_308000000_l1702_170250

theorem scientific_notation_of_308000000 :
  ∃ (a : ℝ) (n : ℤ), (a = 3.08) ∧ (n = 8) ∧ (308000000 = a * 10 ^ n) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_308000000_l1702_170250


namespace NUMINAMATH_GPT_cat_toy_cost_l1702_170236

-- Define the conditions
def cost_of_cage : ℝ := 11.73
def total_cost_of_purchases : ℝ := 21.95

-- Define the proof statement
theorem cat_toy_cost : (total_cost_of_purchases - cost_of_cage) = 10.22 := by
  sorry

end NUMINAMATH_GPT_cat_toy_cost_l1702_170236


namespace NUMINAMATH_GPT_problem_arithmetic_sequence_l1702_170292

-- Definitions based on given conditions
def a1 : ℕ := 2
def d := (13 - 2 * a1) / 3

-- Definition of the nth term in the arithmetic sequence
def a (n : ℕ) : ℕ := a1 + (n - 1) * d

-- The required proof problem statement
theorem problem_arithmetic_sequence : a 4 + a 5 + a 6 = 42 := 
by
  -- placeholders for the actual proof
  sorry

end NUMINAMATH_GPT_problem_arithmetic_sequence_l1702_170292


namespace NUMINAMATH_GPT_div_by_37_l1702_170278

theorem div_by_37 : (333^555 + 555^333) % 37 = 0 :=
by sorry

end NUMINAMATH_GPT_div_by_37_l1702_170278


namespace NUMINAMATH_GPT_sequence_sum_100_eq_200_l1702_170228

theorem sequence_sum_100_eq_200
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = 2)
  (h4 : ∀ n : ℕ, a n * a (n + 1) * a (n + 2) ≠ 1)
  (h5 : ∀ n : ℕ, a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3)) :
  (Finset.range 100).sum (a ∘ Nat.succ) = 200 := by
  sorry

end NUMINAMATH_GPT_sequence_sum_100_eq_200_l1702_170228


namespace NUMINAMATH_GPT_desired_on_time_departure_rate_l1702_170244

theorem desired_on_time_departure_rate :
  let first_late := 1
  let on_time_flights_next := 3
  let additional_on_time_flights := 2
  let total_on_time_flights := on_time_flights_next + additional_on_time_flights
  let total_flights := first_late + on_time_flights_next + additional_on_time_flights
  let on_time_departure_rate := (total_on_time_flights : ℚ) / (total_flights : ℚ) * 100
  on_time_departure_rate > 83.33 :=
by
  sorry

end NUMINAMATH_GPT_desired_on_time_departure_rate_l1702_170244
