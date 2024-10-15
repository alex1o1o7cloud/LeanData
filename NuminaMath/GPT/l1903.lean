import Mathlib

namespace NUMINAMATH_GPT_net_investment_change_l1903_190362

variable (I : ℝ)

def first_year_increase (I : ℝ) : ℝ := I * 1.75
def second_year_decrease (W : ℝ) : ℝ := W * 0.70

theorem net_investment_change : 
  let I' := first_year_increase 100 
  let I'' := second_year_decrease I' 
  I'' - 100 = 22.50 :=
by
  sorry

end NUMINAMATH_GPT_net_investment_change_l1903_190362


namespace NUMINAMATH_GPT_problem_statement_l1903_190317

-- Define the universal set U, and the sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4}

-- Define the complement of B in U
def C_U_B : Set ℕ := { x | x ∈ U ∧ x ∉ B }

-- State the theorem
theorem problem_statement : (A ∩ C_U_B) = {1, 2} :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1903_190317


namespace NUMINAMATH_GPT_johnny_ran_4_times_l1903_190390

-- Block length is 200 meters
def block_length : ℕ := 200

-- Distance run by Johnny is Johnny's running times times the block length
def johnny_distance (J : ℕ) : ℕ := J * block_length

-- Distance run by Mickey is half of Johnny's running times times the block length
def mickey_distance (J : ℕ) : ℕ := (J / 2) * block_length

-- Average distance run by Johnny and Mickey is 600 meters
def average_distance_condition (J : ℕ) : Prop :=
  ((johnny_distance J + mickey_distance J) / 2) = 600

-- We are to prove that Johnny ran 4 times based on the condition
theorem johnny_ran_4_times (J : ℕ) (h : average_distance_condition J) : J = 4 :=
sorry

end NUMINAMATH_GPT_johnny_ran_4_times_l1903_190390


namespace NUMINAMATH_GPT_perfect_square_digits_l1903_190398

theorem perfect_square_digits (x y : ℕ) (h_ne_zero : x ≠ 0) (h_perfect_square : ∀ n: ℕ, n ≥ 1 → ∃ k: ℕ, (10^(n + 2) * x + 10^(n + 1) * 6 + 10 * y + 4) = k^2) :
  (x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0) :=
sorry

end NUMINAMATH_GPT_perfect_square_digits_l1903_190398


namespace NUMINAMATH_GPT_cost_price_percentage_l1903_190345

/-- The cost price (CP) as a percentage of the marked price (MP) given 
that the discount is 18% and the gain percent is 28.125%. -/
theorem cost_price_percentage (MP CP : ℝ) (h1 : CP / MP = 0.64) : 
  (CP / MP) * 100 = 64 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_percentage_l1903_190345


namespace NUMINAMATH_GPT_infinite_sum_fraction_equals_quarter_l1903_190379

theorem infinite_sum_fraction_equals_quarter :
  (∑' n : ℕ, (3 ^ n) / (1 + 3 ^ n + 3 ^ (n + 1) + 3 ^ (2 * n + 1))) = 1 / 4 :=
by
  -- With the given conditions, we need to prove the above statement
  -- The conditions have been used to express the problem in Lean
  sorry

end NUMINAMATH_GPT_infinite_sum_fraction_equals_quarter_l1903_190379


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l1903_190373

-- Problem 1
theorem problem_1 : 4.7 + (-2.5) - (-5.3) - 7.5 = 0 := by
  sorry

-- Problem 2
theorem problem_2 : 18 + 48 / (-2)^2 - (-4)^2 * 5 = -50 := by
  sorry

-- Problem 3
theorem problem_3 : -1^4 + (-2)^2 / 4 * (5 - (-3)^2) = -5 := by
  sorry

-- Problem 4
theorem problem_4 : (-19 + 15 / 16) * 8 = -159 + 1 / 2 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l1903_190373


namespace NUMINAMATH_GPT_number_picked_by_person_announcing_average_5_l1903_190389

-- Definition of given propositions and assumptions
def numbers_picked (b : Fin 6 → ℕ) (average : Fin 6 → ℕ) :=
  (b 4 = 15) ∧
  (average 4 = 8) ∧
  (average 1 = 5) ∧
  (b 2 + b 4 = 16) ∧
  (b 0 + b 2 = 10) ∧
  (b 4 + b 0 = 12)

-- Prove that given the conditions, the number picked by the person announcing an average of 5 is 7
theorem number_picked_by_person_announcing_average_5 (b : Fin 6 → ℕ) (average : Fin 6 → ℕ)
  (h : numbers_picked b average) : b 2 = 7 :=
  sorry

end NUMINAMATH_GPT_number_picked_by_person_announcing_average_5_l1903_190389


namespace NUMINAMATH_GPT_polynomial_expansion_correct_l1903_190386

def polynomial1 (x : ℝ) := 3 * x^2 - 4 * x + 3
def polynomial2 (x : ℝ) := -2 * x^2 + 3 * x - 4

theorem polynomial_expansion_correct {x : ℝ} :
  (polynomial1 x) * (polynomial2 x) = -6 * x^4 + 17 * x^3 - 30 * x^2 + 25 * x - 12 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_correct_l1903_190386


namespace NUMINAMATH_GPT_sqrt_difference_l1903_190365

theorem sqrt_difference (a b : ℝ) (ha : a = 7 + 4 * Real.sqrt 3) (hb : b = 7 - 4 * Real.sqrt 3) :
  Real.sqrt a - Real.sqrt b = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_sqrt_difference_l1903_190365


namespace NUMINAMATH_GPT_smallest_positive_period_1_smallest_positive_period_2_l1903_190355

-- To prove the smallest positive period T for f(x) = |sin x| + |cos x| is π/2
theorem smallest_positive_period_1 : ∃ T > 0, T = Real.pi / 2 ∧ ∀ x : ℝ, (abs (Real.sin (x + T)) + abs (Real.cos (x + T)) = abs (Real.sin x) + abs (Real.cos x))  := sorry

-- To prove the smallest positive period T for f(x) = tan (2x/3) is 3π/2
theorem smallest_positive_period_2 : ∃ T > 0, T = 3 * Real.pi / 2 ∧ ∀ x : ℝ, (Real.tan ((2 * x) / 3 + T) = Real.tan ((2 * x) / 3)) := sorry

end NUMINAMATH_GPT_smallest_positive_period_1_smallest_positive_period_2_l1903_190355


namespace NUMINAMATH_GPT_square_length_QP_l1903_190316

theorem square_length_QP (r1 r2 dist : ℝ) (h_r1 : r1 = 10) (h_r2 : r2 = 7) (h_dist : dist = 15)
  (x : ℝ) (h_equal_chords: QP = PR) :
  x ^ 2 = 65 :=
sorry

end NUMINAMATH_GPT_square_length_QP_l1903_190316


namespace NUMINAMATH_GPT_missing_bricks_is_26_l1903_190382

-- Define the number of bricks per row and the number of rows
def bricks_per_row : Nat := 10
def number_of_rows : Nat := 6

-- Calculate the total number of bricks for a fully completed wall
def total_bricks_full_wall : Nat := bricks_per_row * number_of_rows

-- Assume the number of bricks currently present
def bricks_currently_present : Nat := total_bricks_full_wall - 26

-- Define a function that calculates the number of missing bricks
def number_of_missing_bricks (total_bricks : Nat) (bricks_present : Nat) : Nat :=
  total_bricks - bricks_present

-- Prove that the number of missing bricks is 26
theorem missing_bricks_is_26 : 
  number_of_missing_bricks total_bricks_full_wall bricks_currently_present = 26 :=
by
  sorry

end NUMINAMATH_GPT_missing_bricks_is_26_l1903_190382


namespace NUMINAMATH_GPT_new_average_mark_l1903_190334

theorem new_average_mark (average_mark : ℕ) (average_excluded : ℕ) (total_students : ℕ) (excluded_students: ℕ)
    (h1 : average_mark = 90)
    (h2 : average_excluded = 45)
    (h3 : total_students = 20)
    (h4 : excluded_students = 2) :
  ((total_students * average_mark - excluded_students * average_excluded) / (total_students - excluded_students)) = 95 := by
  sorry

end NUMINAMATH_GPT_new_average_mark_l1903_190334


namespace NUMINAMATH_GPT_remaining_mushroom_pieces_l1903_190380

theorem remaining_mushroom_pieces 
  (mushrooms : ℕ) 
  (pieces_per_mushroom : ℕ) 
  (pieces_used_by_kenny : ℕ) 
  (pieces_used_by_karla : ℕ) 
  (mushrooms_cut : mushrooms = 22) 
  (pieces_per_mushroom_def : pieces_per_mushroom = 4) 
  (kenny_pieces_def : pieces_used_by_kenny = 38) 
  (karla_pieces_def : pieces_used_by_karla = 42) : 
  (mushrooms * pieces_per_mushroom - (pieces_used_by_kenny + pieces_used_by_karla)) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_mushroom_pieces_l1903_190380


namespace NUMINAMATH_GPT_isosceles_right_triangle_hypotenuse_l1903_190335

theorem isosceles_right_triangle_hypotenuse (a : ℝ) (h : ℝ) (hyp : a = 30 ∧ h^2 = a^2 + a^2) : h = 30 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_isosceles_right_triangle_hypotenuse_l1903_190335


namespace NUMINAMATH_GPT_max_rabbits_with_traits_l1903_190375

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end NUMINAMATH_GPT_max_rabbits_with_traits_l1903_190375


namespace NUMINAMATH_GPT_largest_n_l1903_190358

-- Define the condition that n, x, y, z are positive integers
def conditions (n x y z : ℕ) := (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < n) 

-- Formulate the main theorem
theorem largest_n (x y z : ℕ) : 
  conditions 8 x y z →
  8^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 10 :=
by 
  sorry

end NUMINAMATH_GPT_largest_n_l1903_190358


namespace NUMINAMATH_GPT_min_value_of_quadratic_l1903_190353

theorem min_value_of_quadratic (x : ℝ) : ∃ y, y = x^2 + 14*x + 20 ∧ ∀ z, z = x^2 + 14*x + 20 → z ≥ -29 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l1903_190353


namespace NUMINAMATH_GPT_solution_concentration_l1903_190318

theorem solution_concentration (y z : ℝ) :
  let x_vol := 300
  let y_vol := 2 * z
  let z_vol := z
  let total_vol := x_vol + y_vol + z_vol
  let alcohol_x := 0.10 * x_vol
  let alcohol_y := 0.30 * y_vol
  let alcohol_z := 0.40 * z_vol
  let total_alcohol := alcohol_x + alcohol_y + alcohol_z
  total_vol = 600 ∧ y_vol = 2 * z_vol ∧ y_vol + z_vol = 300 → 
  total_alcohol / total_vol = 21.67 / 100 :=
by
  sorry

end NUMINAMATH_GPT_solution_concentration_l1903_190318


namespace NUMINAMATH_GPT_no_integer_solution_l1903_190326

open Polynomial

theorem no_integer_solution (P : Polynomial ℤ) (a b c d : ℤ)
  (h₁ : P.eval a = 2016) (h₂ : P.eval b = 2016) (h₃ : P.eval c = 2016) 
  (h₄ : P.eval d = 2016) (dist : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ¬ ∃ x : ℤ, P.eval x = 2019 :=
sorry

end NUMINAMATH_GPT_no_integer_solution_l1903_190326


namespace NUMINAMATH_GPT_fewer_columns_after_rearrangement_l1903_190356

theorem fewer_columns_after_rearrangement : 
  ∀ (T R R' C C' fewer_columns : ℕ),
    T = 30 → 
    R = 5 → 
    R' = R + 4 →
    C * R = T →
    C' * R' = T →
    fewer_columns = C - C' →
    fewer_columns = 3 :=
by
  intros T R R' C C' fewer_columns hT hR hR' hCR hC'R' hfewer_columns
  -- sorry to skip the proof part
  sorry

end NUMINAMATH_GPT_fewer_columns_after_rearrangement_l1903_190356


namespace NUMINAMATH_GPT_katie_ds_games_l1903_190359

theorem katie_ds_games (new_friends_games old_friends_games total_friends_games katie_games : ℕ) 
  (h1 : new_friends_games = 88)
  (h2 : old_friends_games = 53)
  (h3 : total_friends_games = 141)
  (h4 : total_friends_games = new_friends_games + old_friends_games + katie_games) :
  katie_games = 0 :=
by
  sorry

end NUMINAMATH_GPT_katie_ds_games_l1903_190359


namespace NUMINAMATH_GPT_complement_intersection_l1903_190363

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = {1, 2, 3})
variable (hB : B = {2, 3, 4})

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1903_190363


namespace NUMINAMATH_GPT_range_of_a_l1903_190327

open Real 

noncomputable def trigonometric_inequality (θ a : ℝ) : Prop :=
  sin (2 * θ) - (2 * sqrt 2 + sqrt 2 * a) * sin (θ + π / 4) - 2 * sqrt 2 / cos (θ - π / 4) > -3 - 2 * a

theorem range_of_a (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → trigonometric_inequality θ a) ↔ (a > 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1903_190327


namespace NUMINAMATH_GPT_option_D_correct_l1903_190360

variables (Line : Type) (Plane : Type)
variables (parallel : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop)
variables (perpendicular_planes : Plane → Plane → Prop)

theorem option_D_correct (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → perpendicular_planes α β :=
sorry

end NUMINAMATH_GPT_option_D_correct_l1903_190360


namespace NUMINAMATH_GPT_fraction_remaining_distance_l1903_190344

theorem fraction_remaining_distance
  (total_distance : ℕ)
  (first_stop_fraction : ℚ)
  (remaining_distance_after_second_stop : ℕ)
  (fraction_between_stops : ℚ) :
  total_distance = 280 →
  first_stop_fraction = 1/2 →
  remaining_distance_after_second_stop = 105 →
  (fraction_between_stops * (total_distance - (first_stop_fraction * total_distance)) + remaining_distance_after_second_stop = (total_distance - (first_stop_fraction * total_distance))) →
  fraction_between_stops = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_remaining_distance_l1903_190344


namespace NUMINAMATH_GPT_solve_inequality_l1903_190399

open Set

theorem solve_inequality (x : ℝ) :
  { x | (x^2 - 9) / (x^2 - 16) > 0 } = (Iio (-4)) ∪ (Ioi 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1903_190399


namespace NUMINAMATH_GPT_solve_quadratic_l1903_190384

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, 
  (-6) * x1^2 + 11 * x1 - 3 = 0 ∧ (-6) * x2^2 + 11 * x2 - 3 = 0 ∧ x1 = 1.5 ∧ x2 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1903_190384


namespace NUMINAMATH_GPT_proof_equiv_l1903_190369

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 6 * x + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem proof_equiv (x : ℝ) : f (g x) - g (f x) = 6 * x ^ 2 - 12 * x + 9 := by
  sorry

end NUMINAMATH_GPT_proof_equiv_l1903_190369


namespace NUMINAMATH_GPT_problem_I_problem_II_l1903_190308

-- Declaration of function f(x)
def f (x a b : ℝ) := |x + a| - |x - b|

-- Proof 1: When a = 1, b = 1, solve the inequality f(x) > 1
theorem problem_I (x : ℝ) : (f x 1 1) > 1 ↔ x > 1/2 := by
  sorry

-- Proof 2: If the maximum value of the function f(x) is 2, prove that (1/a) + (1/b) ≥ 2
theorem problem_II (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max_f : ∀ x, f x a b ≤ 2) : 1 / a + 1 / b ≥ 2 := by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1903_190308


namespace NUMINAMATH_GPT_desiredCircleEquation_l1903_190321

-- Definition of the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 + x - 6*y + 3 = 0

-- Definition of the given line
def givenLine (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- The required proof problem statement
theorem desiredCircleEquation :
  (∀ P Q : ℝ × ℝ, givenCircle P.1 P.2 ∧ givenLine P.1 P.2 → givenCircle Q.1 Q.2 ∧ givenLine Q.1 Q.2 →
  (P ≠ Q) → 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0)) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_desiredCircleEquation_l1903_190321


namespace NUMINAMATH_GPT_problem_statement_l1903_190352

theorem problem_statement (A B : ℤ) (h1 : A * B = 15) (h2 : -7 * B - 8 * A = -94) : AB + A = 20 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1903_190352


namespace NUMINAMATH_GPT_set_intersection_l1903_190325

open Set

def U : Set ℤ := univ
def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {-1, 1}
def C_U_B : Set ℤ := U \ B

theorem set_intersection :
  A ∩ C_U_B = {2} := 
by
  sorry

end NUMINAMATH_GPT_set_intersection_l1903_190325


namespace NUMINAMATH_GPT_basketball_lineups_l1903_190310

noncomputable def num_starting_lineups (total_players : ℕ) (fixed_players : ℕ) (chosen_players : ℕ) : ℕ :=
  Nat.choose (total_players - fixed_players) (chosen_players - fixed_players)

theorem basketball_lineups :
  num_starting_lineups 15 2 6 = 715 := by
  sorry

end NUMINAMATH_GPT_basketball_lineups_l1903_190310


namespace NUMINAMATH_GPT_prove_true_statement_l1903_190301

-- Definitions based on conditions in the problem
def A_statement := ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0

-- Equivalent proof problem in Lean 4
theorem prove_true_statement : A_statement :=
by
  sorry

end NUMINAMATH_GPT_prove_true_statement_l1903_190301


namespace NUMINAMATH_GPT_distance_apart_after_two_hours_l1903_190328

theorem distance_apart_after_two_hours :
  (Jay_walk_rate : ℝ) = 1 / 20 →
  (Paul_jog_rate : ℝ) = 3 / 40 →
  (time_duration : ℝ) = 2 * 60 →
  (distance_apart : ℝ) = 15 :=
by
  sorry

end NUMINAMATH_GPT_distance_apart_after_two_hours_l1903_190328


namespace NUMINAMATH_GPT_negation_of_proposition_l1903_190392

theorem negation_of_proposition :
  (∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (¬ ∃ x : ℝ, x^2 + 1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1903_190392


namespace NUMINAMATH_GPT_complement_set_l1903_190300

open Set

theorem complement_set (U M : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hM : M = {1, 2, 4}) :
  compl M ∩ U = {3, 5, 6} := 
by
  rw [compl, hU, hM]
  sorry

end NUMINAMATH_GPT_complement_set_l1903_190300


namespace NUMINAMATH_GPT_sqrt_15_minus_1_range_l1903_190391

theorem sqrt_15_minus_1_range (h : 9 < 15 ∧ 15 < 16) : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := 
  sorry

end NUMINAMATH_GPT_sqrt_15_minus_1_range_l1903_190391


namespace NUMINAMATH_GPT_factor_expression_l1903_190393

theorem factor_expression (y : ℝ) : 
  (16 * y ^ 6 + 36 * y ^ 4 - 9) - (4 * y ^ 6 - 6 * y ^ 4 - 9) = 6 * y ^ 4 * (2 * y ^ 2 + 7) := 
by sorry

end NUMINAMATH_GPT_factor_expression_l1903_190393


namespace NUMINAMATH_GPT_sum_infinite_geometric_series_l1903_190364

theorem sum_infinite_geometric_series :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 3
  (a / (1 - r) = (3 : ℚ) / 8) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 3
  sorry

end NUMINAMATH_GPT_sum_infinite_geometric_series_l1903_190364


namespace NUMINAMATH_GPT_roots_eq_two_iff_a_gt_neg1_l1903_190339

theorem roots_eq_two_iff_a_gt_neg1 (a : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*|x₁ + 1| = a ∧ x₂^2 + 2*x₂ + 2*|x₂ + 1| = a) ↔ a > -1 :=
by sorry

end NUMINAMATH_GPT_roots_eq_two_iff_a_gt_neg1_l1903_190339


namespace NUMINAMATH_GPT_additional_discount_percentage_l1903_190366

theorem additional_discount_percentage
  (MSRP : ℝ)
  (p : ℝ)
  (d : ℝ)
  (sale_price : ℝ)
  (H1 : MSRP = 45.0)
  (H2 : p = 0.30)
  (H3 : d = MSRP - (p * MSRP))
  (H4 : d = 31.50)
  (H5 : sale_price = 25.20) :
  sale_price = d - (0.20 * d) :=
by
  sorry

end NUMINAMATH_GPT_additional_discount_percentage_l1903_190366


namespace NUMINAMATH_GPT_a_2_correct_l1903_190341

noncomputable def a_2_value (a a1 a2 a3 : ℝ) : Prop :=
∀ x : ℝ, x^3 = a + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3

theorem a_2_correct (a a1 a2 a3 : ℝ) (h : a_2_value a a1 a2 a3) : a2 = 6 :=
sorry

end NUMINAMATH_GPT_a_2_correct_l1903_190341


namespace NUMINAMATH_GPT_minimum_n_divisible_20_l1903_190347

theorem minimum_n_divisible_20 :
  ∃ (n : ℕ), (∀ (l : List ℕ), l.length = n → 
    ∃ (a b c d : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ d ∈ l ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b - c - d) % 20 = 0) ∧ 
  (∀ m, m < n → ¬(∀ (l : List ℕ), l.length = m → 
    ∃ (a b c d : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ d ∈ l ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b - c - d) % 20 = 0)) := 
⟨9, 
  by sorry, 
  by sorry⟩

end NUMINAMATH_GPT_minimum_n_divisible_20_l1903_190347


namespace NUMINAMATH_GPT_find_y_l1903_190395

theorem find_y 
  (y : ℝ) 
  (h1 : (y^2 - 11 * y + 24) / (y - 3) + (2 * y^2 + 7 * y - 18) / (2 * y - 3) = -10)
  (h2 : y ≠ 3)
  (h3 : y ≠ 3 / 2) : 
  y = -4 := 
sorry

end NUMINAMATH_GPT_find_y_l1903_190395


namespace NUMINAMATH_GPT_min_cards_for_certain_event_l1903_190303

-- Let's define the deck configuration
structure DeckConfig where
  spades : ℕ
  clubs : ℕ
  hearts : ℕ
  total : ℕ

-- Define the given condition of the deck
def givenDeck : DeckConfig := { spades := 5, clubs := 4, hearts := 6, total := 15 }

-- Predicate to check if m cards drawn guarantees all three suits are present
def is_certain_event (m : ℕ) (deck : DeckConfig) : Prop :=
  m >= deck.spades + deck.hearts + 1

-- The main theorem to prove the minimum number of cards m
theorem min_cards_for_certain_event : ∀ m, is_certain_event m givenDeck ↔ m = 12 :=
by
  sorry

end NUMINAMATH_GPT_min_cards_for_certain_event_l1903_190303


namespace NUMINAMATH_GPT_courtyard_width_l1903_190350

theorem courtyard_width 
  (length_of_courtyard : ℝ) 
  (num_paving_stones : ℕ) 
  (length_of_stone width_of_stone : ℝ) 
  (total_area_stone : ℝ) 
  (W : ℝ) : 
  length_of_courtyard = 40 →
  num_paving_stones = 132 →
  length_of_stone = 2.5 →
  width_of_stone = 2 →
  total_area_stone = 660 →
  40 * W = 660 →
  W = 16.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_courtyard_width_l1903_190350


namespace NUMINAMATH_GPT_root_equation_val_l1903_190315

theorem root_equation_val (a : ℝ) (h : a^2 - 2 * a - 5 = 0) : 2 * a^2 - 4 * a = 10 :=
by 
  sorry

end NUMINAMATH_GPT_root_equation_val_l1903_190315


namespace NUMINAMATH_GPT_problem_proof_l1903_190377

variable (A B C a b c : ℝ)
variable (ABC_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (sides_opposite : a = (b * sin A / sin B) ∧ b = (a * sin B / sin A))
variable (cos_eq : b + b * cos A = a * cos B)

theorem problem_proof :
  (A = 2 * B ∧ (π / 6 < B ∧ B < π / 4) ∧ a^2 = b^2 + b * c) :=
  sorry

end NUMINAMATH_GPT_problem_proof_l1903_190377


namespace NUMINAMATH_GPT_solve_fractional_eq_l1903_190357

theorem solve_fractional_eq {x : ℚ} : (3 / (x - 1)) = (1 / x) ↔ x = -1/2 :=
by sorry

end NUMINAMATH_GPT_solve_fractional_eq_l1903_190357


namespace NUMINAMATH_GPT_smallest_sector_angle_l1903_190370

-- Definitions and conditions identified in step a.

def a1 (d : ℕ) : ℕ := (48 - 14 * d) / 2

-- Proof statement
theorem smallest_sector_angle : ∀ d : ℕ, d ≥ 0 → d ≤ 3 → 15 * (a1 d + (a1 d + 14 * d)) = 720 → (a1 d = 3) :=
by
  sorry

end NUMINAMATH_GPT_smallest_sector_angle_l1903_190370


namespace NUMINAMATH_GPT_cubic_product_of_roots_l1903_190378

theorem cubic_product_of_roots (k : ℝ) :
  (∃ a b c : ℝ, a + b + c = 2 ∧ ab + bc + ca = 1 ∧ abc = -k ∧ -k = (max (max a b) c - min (min a b) c)^2) ↔ k = -2 :=
by
  sorry

end NUMINAMATH_GPT_cubic_product_of_roots_l1903_190378


namespace NUMINAMATH_GPT_find_a_perpendicular_lines_l1903_190338

variable (a : ℝ)

theorem find_a_perpendicular_lines :
  (∃ a : ℝ, ∀ x y : ℝ, (a * x - y + 2 * a = 0) ∧ ((2 * a - 1) * x + a * y + a = 0) → a = 0 ∨ a = 1) := 
sorry

end NUMINAMATH_GPT_find_a_perpendicular_lines_l1903_190338


namespace NUMINAMATH_GPT_number_of_boys_l1903_190329

theorem number_of_boys (B G : ℕ) 
    (h1 : B + G = 345) 
    (h2 : G = B + 69) : B = 138 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l1903_190329


namespace NUMINAMATH_GPT_smartphone_customers_l1903_190342

theorem smartphone_customers (k : ℝ) (p1 p2 c1 c2 : ℝ)
  (h₁ : p1 * c1 = k)
  (h₂ : 20 = p1)
  (h₃ : 200 = c1)
  (h₄ : 400 = c2) :
  p2 * c2 = k  → p2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_smartphone_customers_l1903_190342


namespace NUMINAMATH_GPT_problem_D_l1903_190388

variable (f : ℕ → ℝ)

-- Function condition: If f(k) ≥ k^2, then f(k+1) ≥ (k+1)^2
axiom f_property (k : ℕ) (hk : f k ≥ k^2) : f (k + 1) ≥ (k + 1)^2

theorem problem_D (hf4 : f 4 ≥ 25) : ∀ k ≥ 4, f k ≥ k^2 :=
by
  sorry

end NUMINAMATH_GPT_problem_D_l1903_190388


namespace NUMINAMATH_GPT_debby_vacation_pictures_l1903_190319

theorem debby_vacation_pictures :
  let zoo_initial := 150
  let aquarium_initial := 210
  let museum_initial := 90
  let amusement_park_initial := 120
  let zoo_deleted := (25 * zoo_initial) / 100  -- 25% of zoo pictures deleted
  let aquarium_deleted := (15 * aquarium_initial) / 100  -- 15% of aquarium pictures deleted
  let museum_added := 30  -- 30 additional pictures at the museum
  let amusement_park_deleted := 20  -- 20 pictures deleted at the amusement park
  let zoo_kept := zoo_initial - zoo_deleted
  let aquarium_kept := aquarium_initial - aquarium_deleted
  let museum_kept := museum_initial + museum_added
  let amusement_park_kept := amusement_park_initial - amusement_park_deleted
  let total_pictures := zoo_kept + aquarium_kept + museum_kept + amusement_park_kept
  total_pictures = 512 :=
by
  sorry

end NUMINAMATH_GPT_debby_vacation_pictures_l1903_190319


namespace NUMINAMATH_GPT_min_num_cuboids_l1903_190307

/-
Definitions based on the conditions:
- Dimensions of the cuboid are given as 3 cm, 4 cm, and 5 cm.
- We need to find the Least Common Multiple (LCM) of these dimensions.
- Calculate the volume of the smallest cube.
- Calculate the volume of the given cuboid.
- Find the number of such cuboids needed to form the cube.
-/
def cuboid_length : ℤ := 3
def cuboid_width : ℤ := 4
def cuboid_height : ℤ := 5

noncomputable def lcm_3_4_5 : ℤ := Int.lcm (Int.lcm cuboid_length cuboid_width) cuboid_height

noncomputable def cube_side_length : ℤ := lcm_3_4_5
noncomputable def cube_volume : ℤ := cube_side_length * cube_side_length * cube_side_length
noncomputable def cuboid_volume : ℤ := cuboid_length * cuboid_width * cuboid_height

noncomputable def num_cuboids : ℤ := cube_volume / cuboid_volume

theorem min_num_cuboids :
  num_cuboids = 3600 := by
  sorry

end NUMINAMATH_GPT_min_num_cuboids_l1903_190307


namespace NUMINAMATH_GPT_color_5x5_grid_excluding_two_corners_l1903_190374

-- Define the total number of ways to color a 5x5 grid with each row and column having exactly one colored cell
def total_ways : Nat := 120

-- Define the number of ways to color a 5x5 grid excluding one specific corner cell such that each row and each column has exactly one colored cell
def ways_excluding_one_corner : Nat := 96

-- Prove the number of ways to color the grid excluding two specific corner cells is 78
theorem color_5x5_grid_excluding_two_corners : total_ways - (ways_excluding_one_corner + ways_excluding_one_corner - 6) = 78 := by
  -- We state our given conditions directly as definitions
  -- Now we state our theorem explicitly and use the correct answer we derived
  sorry

end NUMINAMATH_GPT_color_5x5_grid_excluding_two_corners_l1903_190374


namespace NUMINAMATH_GPT_negation_exists_lt_zero_l1903_190322

variable {f : ℝ → ℝ}

theorem negation_exists_lt_zero :
  ¬ (∃ x : ℝ, f x < 0) → ∀ x : ℝ, 0 ≤ f x := by
  sorry

end NUMINAMATH_GPT_negation_exists_lt_zero_l1903_190322


namespace NUMINAMATH_GPT_probability_problem_l1903_190367

def ang_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]
def ben_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]
def jasmin_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]

def boxes : Fin 6 := sorry  -- represents 6 empty boxes
def white_restriction (box : Fin 6) : Prop := box ≠ 0  -- white block can't be in the first box

def probability_at_least_one_box_three_same_color : ℚ := 1 / 72  -- The given probability

theorem probability_problem (p q : ℕ) 
  (hpq_coprime : Nat.gcd p q = 1) 
  (hprob_eq : probability_at_least_one_box_three_same_color = p / q) :
  p + q = 73 :=
sorry

end NUMINAMATH_GPT_probability_problem_l1903_190367


namespace NUMINAMATH_GPT_solution_set_inequality_l1903_190331

noncomputable def f : ℝ → ℝ := sorry
noncomputable def derivative_f : ℝ → ℝ := sorry -- f' is the derivative of f

-- Conditions
axiom f_domain {x : ℝ} (h1 : 0 < x) : f x ≠ 0
axiom derivative_condition {x : ℝ} (h1 : 0 < x) : f x + x * derivative_f x > 0
axiom initial_value : f 1 = 2

-- Proof that the solution set of the inequality f(x) < 2/x is (0, 1)
theorem solution_set_inequality : ∀ x : ℝ, 0 < x ∧ x < 1 → f x < 2 / x := sorry

end NUMINAMATH_GPT_solution_set_inequality_l1903_190331


namespace NUMINAMATH_GPT_units_digit_of_square_l1903_190376

theorem units_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_of_square_l1903_190376


namespace NUMINAMATH_GPT_hypotenuse_length_l1903_190397

-- Define the properties of the right-angled triangle
variables (α β γ : ℝ) (a b c : ℝ)
-- Right-angled triangle condition
axiom right_angled_triangle : α = 30 ∧ β = 60 ∧ γ = 90 → c = 2 * a

-- Given side opposite 30° angle is 6 cm
axiom side_opposite_30_is_6cm : a = 6

-- Proof that hypotenuse is 12 cm
theorem hypotenuse_length : c = 12 :=
by 
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1903_190397


namespace NUMINAMATH_GPT_find_cost_price_l1903_190383

-- Definitions based on conditions
def cost_price (C : ℝ) : Prop := 0.05 * C = 10

-- The theorem stating the problem to be proven
theorem find_cost_price (C : ℝ) (h : cost_price C) : C = 200 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l1903_190383


namespace NUMINAMATH_GPT_lilly_fish_l1903_190368

-- Define the conditions
def total_fish : ℕ := 18
def rosy_fish : ℕ := 8

-- Statement: Prove that Lilly has 10 fish
theorem lilly_fish (h1 : total_fish = 18) (h2 : rosy_fish = 8) :
  total_fish - rosy_fish = 10 :=
by sorry

end NUMINAMATH_GPT_lilly_fish_l1903_190368


namespace NUMINAMATH_GPT_grains_in_gray_parts_l1903_190340

theorem grains_in_gray_parts (total1 total2 shared : ℕ) (h1 : total1 = 87) (h2 : total2 = 110) (h_shared : shared = 68) :
  (total1 - shared) + (total2 - shared) = 61 :=
by sorry

end NUMINAMATH_GPT_grains_in_gray_parts_l1903_190340


namespace NUMINAMATH_GPT_trig_identity_l1903_190312

theorem trig_identity :
  2 * Real.sin (Real.pi / 6) - Real.cos (Real.pi / 4)^2 + Real.cos (Real.pi / 3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1903_190312


namespace NUMINAMATH_GPT_count_congruent_to_4_mod_7_l1903_190337

theorem count_congruent_to_4_mod_7 : 
  ∃ (n : ℕ), 
  n = 71 ∧ 
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 70 → ∃ m : ℕ, m = 4 + 7 * k ∧ m < 500 := 
by
  sorry

end NUMINAMATH_GPT_count_congruent_to_4_mod_7_l1903_190337


namespace NUMINAMATH_GPT_woman_work_rate_l1903_190305

theorem woman_work_rate :
  let M := 1/6
  let B := 1/9
  let combined_rate := 1/3
  ∃ W : ℚ, M + B + W = combined_rate ∧ 1 / W = 18 := 
by
  sorry

end NUMINAMATH_GPT_woman_work_rate_l1903_190305


namespace NUMINAMATH_GPT_find_parallel_line_through_point_l1903_190302

-- Definition of a point in Cartesian coordinates
structure Point :=
(x : ℝ)
(y : ℝ)

-- Definition of a line in slope-intercept form
def line (a b c : ℝ) : Prop := ∀ p : Point, a * p.x + b * p.y + c = 0

-- Conditions provided in the problem
def P : Point := ⟨-1, 3⟩
def line1 : Prop := line 1 (-2) 3
def parallel_line (c : ℝ) : Prop := line 1 (-2) c

-- Theorem to prove
theorem find_parallel_line_through_point : parallel_line 7 :=
sorry

end NUMINAMATH_GPT_find_parallel_line_through_point_l1903_190302


namespace NUMINAMATH_GPT_scientific_notation_of_570_million_l1903_190304

theorem scientific_notation_of_570_million :
  570000000 = 5.7 * 10^8 := sorry

end NUMINAMATH_GPT_scientific_notation_of_570_million_l1903_190304


namespace NUMINAMATH_GPT_sum_of_first_and_third_l1903_190372

theorem sum_of_first_and_third :
  ∀ (A B C : ℕ),
  A + B + C = 330 →
  A = 2 * B →
  C = A / 3 →
  B = 90 →
  A + C = 240 :=
by
  intros A B C h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sum_of_first_and_third_l1903_190372


namespace NUMINAMATH_GPT_find_initial_books_l1903_190349

/-- The number of books the class initially obtained from the library --/
def initial_books : ℕ := sorry

/-- The number of books added later --/
def books_added_later : ℕ := 23

/-- The total number of books the class has --/
def total_books : ℕ := 77

theorem find_initial_books : initial_books + books_added_later = total_books → initial_books = 54 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_initial_books_l1903_190349


namespace NUMINAMATH_GPT_max_area_of_sector_l1903_190313

theorem max_area_of_sector (α R C : Real) (hC : C > 0) (h : C = 2 * R + α * R) : 
  ∃ S_max : Real, S_max = (C^2) / 16 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_sector_l1903_190313


namespace NUMINAMATH_GPT_positive_integer_solutions_x_plus_2y_eq_5_l1903_190311

theorem positive_integer_solutions_x_plus_2y_eq_5 :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (x + 2 * y = 5) ∧ ((x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 1)) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_x_plus_2y_eq_5_l1903_190311


namespace NUMINAMATH_GPT_mia_bought_more_pencils_l1903_190309

theorem mia_bought_more_pencils (p : ℝ) (n1 n2 : ℕ) 
  (price_pos : p > 0.01)
  (liam_spent : 2.10 = p * n1)
  (mia_spent : 2.82 = p * n2) :
  (n2 - n1) = 12 := 
by
  sorry

end NUMINAMATH_GPT_mia_bought_more_pencils_l1903_190309


namespace NUMINAMATH_GPT_adjacent_side_length_l1903_190333

-- Given the conditions
variables (a b : ℝ)
-- Area of the rectangular flower bed
def area := 6 * a * b - 2 * b
-- One side of the rectangular flower bed
def side1 := 2 * b

-- Prove the length of the adjacent side
theorem adjacent_side_length : 
  (6 * a * b - 2 * b) / (2 * b) = 3 * a - 1 :=
by sorry

end NUMINAMATH_GPT_adjacent_side_length_l1903_190333


namespace NUMINAMATH_GPT_M_inter_P_eq_l1903_190346

-- Define the sets M and P
def M : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ 4 * x + y = 6 }
def P : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ 3 * x + 2 * y = 7 }

-- Prove that the intersection of M and P is {(1, 2)}
theorem M_inter_P_eq : M ∩ P = { (1, 2) } := 
by 
sorry

end NUMINAMATH_GPT_M_inter_P_eq_l1903_190346


namespace NUMINAMATH_GPT_sphere_volume_l1903_190323

theorem sphere_volume (S : ℝ) (r : ℝ) (V : ℝ) (h₁ : S = 256 * Real.pi) (h₂ : S = 4 * Real.pi * r^2) : V = 2048 / 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_volume_l1903_190323


namespace NUMINAMATH_GPT_xyz_line_segments_total_length_l1903_190371

noncomputable def total_length_XYZ : ℝ :=
  let length_X := 2 * Real.sqrt 2
  let length_Y := 2 + 2 * Real.sqrt 2
  let length_Z := 2 + Real.sqrt 2
  length_X + length_Y + length_Z

theorem xyz_line_segments_total_length : total_length_XYZ = 4 + 5 * Real.sqrt 2 := 
  sorry

end NUMINAMATH_GPT_xyz_line_segments_total_length_l1903_190371


namespace NUMINAMATH_GPT_assistant_professors_charts_l1903_190324

theorem assistant_professors_charts (A B C : ℕ) (h1 : 2 * A + B = 10) (h2 : A + B * C = 11) (h3 : A + B = 7) : C = 2 :=
by
  sorry

end NUMINAMATH_GPT_assistant_professors_charts_l1903_190324


namespace NUMINAMATH_GPT_no_finite_spells_guarantee_second_wizard_win_exists_infinite_spells_guarantee_second_wizard_win_l1903_190387

variables {a b : ℝ} (spells : list (ℝ × ℝ)) (infinite_spells : ℕ → ℝ × ℝ)

-- Condition: 0 < a < b
def valid_spell (spell : ℝ × ℝ) : Prop := 0 < spell.1 ∧ spell.1 < spell.2

-- Question a: Finite set of spells, prove that no spell set exists such that the second wizard can guarantee a win.
theorem no_finite_spells_guarantee_second_wizard_win :
  (∀ spell ∈ spells, valid_spell spell) →
  ¬(∃ (strategy : ℕ → ℝ × ℝ), ∀ n, valid_spell (strategy n) ∧ ∃ k, n < k ∧ valid_spell (strategy k)) :=
sorry

-- Question b: Infinite set of spells, prove that there exists a spell set such that the second wizard can guarantee a win.
theorem exists_infinite_spells_guarantee_second_wizard_win :
  (∀ n, valid_spell (infinite_spells n)) →
  ∃ (strategy : ℕ → ℝ × ℝ), ∀ n, ∃ k, n < k ∧ valid_spell (strategy k) :=
sorry

end NUMINAMATH_GPT_no_finite_spells_guarantee_second_wizard_win_exists_infinite_spells_guarantee_second_wizard_win_l1903_190387


namespace NUMINAMATH_GPT_tim_total_spent_l1903_190351

variable (lunch_cost : ℝ)
variable (tip_percentage : ℝ)
variable (total_spent : ℝ)

theorem tim_total_spent (h_lunch_cost : lunch_cost = 60.80)
                        (h_tip_percentage : tip_percentage = 0.20)
                        (h_total_spent : total_spent = lunch_cost + (tip_percentage * lunch_cost)) :
                        total_spent = 72.96 :=
sorry

end NUMINAMATH_GPT_tim_total_spent_l1903_190351


namespace NUMINAMATH_GPT_usual_time_to_reach_school_l1903_190361

variable (R T : ℝ)
variable (h : T * R = (T - 4) * (7/6 * R))

theorem usual_time_to_reach_school (h : T * R = (T - 4) * (7/6 * R)) : T = 28 := by
  sorry

end NUMINAMATH_GPT_usual_time_to_reach_school_l1903_190361


namespace NUMINAMATH_GPT_pages_for_thirty_dollars_l1903_190332

-- Problem Statement Definitions
def costPerCopy := 4 -- cents
def pagesPerCopy := 2 -- pages
def totalCents := 3000 -- cents
def totalPages := 1500 -- pages

-- Theorem: Calculating the number of pages for a given cost.
theorem pages_for_thirty_dollars (c_per_copy : ℕ) (p_per_copy : ℕ) (t_cents : ℕ) (t_pages : ℕ) : 
  c_per_copy = 4 → p_per_copy = 2 → t_cents = 3000 → t_pages = 1500 := by
  intros h_cpc h_ppc h_tc
  sorry

end NUMINAMATH_GPT_pages_for_thirty_dollars_l1903_190332


namespace NUMINAMATH_GPT_points_needed_for_office_l1903_190330

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25

def jerry_interruptions : ℕ := 2
def jerry_insults : ℕ := 4
def jerry_throwings : ℕ := 2

def jerry_total_points (interrupt_points insult_points throw_points : ℕ) 
                       (interruptions insults throwings : ℕ) : ℕ :=
  (interrupt_points * interruptions) +
  (insult_points * insults) +
  (throw_points * throwings)

theorem points_needed_for_office : 
  jerry_total_points points_for_interrupting points_for_insulting points_for_throwing 
                     (jerry_interruptions) 
                     (jerry_insults) 
                     (jerry_throwings) = 100 := 
  sorry

end NUMINAMATH_GPT_points_needed_for_office_l1903_190330


namespace NUMINAMATH_GPT_alina_sent_fewer_messages_l1903_190343

-- Definitions based on conditions
def messages_lucia_day1 : Nat := 120
def messages_lucia_day2 : Nat := 1 / 3 * messages_lucia_day1
def messages_lucia_day3 : Nat := messages_lucia_day1
def messages_total : Nat := 680

-- Def statement for Alina's messages on the first day, which we need to find as 100
def messages_alina_day1 : Nat := 100

-- Condition checks
def condition_alina_day2 : Prop := 2 * messages_alina_day1 = 2 * 100
def condition_alina_day3 : Prop := messages_alina_day1 = 100
def condition_total_messages : Prop := 
  messages_alina_day1 + messages_lucia_day1 +
  2 * messages_alina_day1 + messages_lucia_day2 +
  messages_alina_day1 + messages_lucia_day1 = messages_total

-- Theorem statement
theorem alina_sent_fewer_messages :
  messages_lucia_day1 - messages_alina_day1 = 20 :=
by
  -- Ensure the conditions hold
  have h1 : messages_alina_day1 = 100 := by sorry
  have h2 : condition_alina_day2 := by sorry
  have h3 : condition_alina_day3 := by sorry
  have h4 : condition_total_messages := by sorry
  -- Prove the theorem
  sorry

end NUMINAMATH_GPT_alina_sent_fewer_messages_l1903_190343


namespace NUMINAMATH_GPT_total_boys_in_camp_l1903_190336

theorem total_boys_in_camp (T : ℝ) (h : 0.70 * (0.20 * T) = 28) : T = 200 := 
by
  sorry

end NUMINAMATH_GPT_total_boys_in_camp_l1903_190336


namespace NUMINAMATH_GPT_Jorge_age_in_2005_l1903_190396

theorem Jorge_age_in_2005
  (age_Simon_2010 : ℕ)
  (age_difference : ℕ)
  (age_of_Simon_2010 : age_Simon_2010 = 45)
  (age_difference_Simon_Jorge : age_difference = 24)
  (age_Simon_2005 : ℕ := age_Simon_2010 - 5)
  (age_Jorge_2005 : ℕ := age_Simon_2005 - age_difference) :
  age_Jorge_2005 = 16 := by
  sorry

end NUMINAMATH_GPT_Jorge_age_in_2005_l1903_190396


namespace NUMINAMATH_GPT_calculate_f_5_5_l1903_190306

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic_condition (x : ℝ) (h₂ : 2 ≤ x ∧ x ≤ 3) : f (x + 2) = -1 / f x
axiom defined_segment (x : ℝ) (h₂ : 2 ≤ x ∧ x ≤ 3) : f x = x

theorem calculate_f_5_5 : f 5.5 = 2.5 := sorry

end NUMINAMATH_GPT_calculate_f_5_5_l1903_190306


namespace NUMINAMATH_GPT_initial_sugar_weight_l1903_190354

-- Definitions corresponding to the conditions
def num_packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Statement of the proof problem
theorem initial_sugar_weight : 
  (num_packs * weight_per_pack + leftover_sugar = 3020) :=
by
  sorry

end NUMINAMATH_GPT_initial_sugar_weight_l1903_190354


namespace NUMINAMATH_GPT_problem_solution_l1903_190381

theorem problem_solution (a b c : ℝ) (h : (a / (36 - a)) + (b / (45 - b)) + (c / (54 - c)) = 8) :
    (4 / (36 - a)) + (5 / (45 - b)) + (6 / (54 - c)) = 11 / 9 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1903_190381


namespace NUMINAMATH_GPT_parallel_lines_coplanar_l1903_190348

axiom Plane : Type
axiom Point : Type
axiom Line : Type

axiom A : Point
axiom B : Point
axiom C : Point
axiom D : Point

axiom α : Plane
axiom β : Plane

axiom in_plane (p : Point) (π : Plane) : Prop
axiom parallel_plane (π1 π2 : Plane) : Prop
axiom parallel_line (l1 l2 : Line) : Prop
axiom line_through (P Q : Point) : Line
axiom coplanar (P Q R S : Point) : Prop

-- Conditions
axiom A_in_α : in_plane A α
axiom C_in_α : in_plane C α
axiom B_in_β : in_plane B β
axiom D_in_β : in_plane D β
axiom α_parallel_β : parallel_plane α β

-- Statement
theorem parallel_lines_coplanar :
  parallel_line (line_through A C) (line_through B D) ↔ coplanar A B C D :=
sorry

end NUMINAMATH_GPT_parallel_lines_coplanar_l1903_190348


namespace NUMINAMATH_GPT_ryan_learning_schedule_l1903_190314

theorem ryan_learning_schedule
  (E1 E2 E3 S1 S2 S3 : ℕ)
  (hE1 : E1 = 7) (hE2 : E2 = 6) (hE3 : E3 = 8)
  (hS1 : S1 = 4) (hS2 : S2 = 5) (hS3 : S3 = 3):
  (E1 + E2 + E3) - (S1 + S2 + S3) = 9 :=
by
  sorry

end NUMINAMATH_GPT_ryan_learning_schedule_l1903_190314


namespace NUMINAMATH_GPT_average_infected_per_round_is_nine_l1903_190394

theorem average_infected_per_round_is_nine (x : ℝ) :
  1 + x + x * (1 + x) = 100 → x = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_infected_per_round_is_nine_l1903_190394


namespace NUMINAMATH_GPT_volume_third_bottle_is_250_milliliters_l1903_190320

-- Define the volumes of the bottles in milliliters
def volume_first_bottle : ℕ := 2 * 1000                        -- 2000 milliliters
def volume_second_bottle : ℕ := 750                            -- 750 milliliters
def total_volume : ℕ := 3 * 1000                               -- 3000 milliliters
def volume_third_bottle : ℕ := total_volume - (volume_first_bottle + volume_second_bottle)

-- The theorem stating the volume of the third bottle
theorem volume_third_bottle_is_250_milliliters :
  volume_third_bottle = 250 :=
by
  sorry

end NUMINAMATH_GPT_volume_third_bottle_is_250_milliliters_l1903_190320


namespace NUMINAMATH_GPT_find_f_neg5_l1903_190385

-- Define the function f and the constants a, b, and c
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 5

-- State the main theorem we want to prove
theorem find_f_neg5 (a b c : ℝ) (h : f 5 a b c = 9) : f (-5) a b c = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg5_l1903_190385
