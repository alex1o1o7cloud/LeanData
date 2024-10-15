import Mathlib

namespace NUMINAMATH_GPT_polynomial_roots_l1243_124391

theorem polynomial_roots :
  ∀ x, (3 * x^4 + 16 * x^3 - 36 * x^2 + 8 * x = 0) ↔ 
       (x = 0 ∨ x = 1 / 3 ∨ x = -3 + 2 * Real.sqrt 17 ∨ x = -3 - 2 * Real.sqrt 17) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l1243_124391


namespace NUMINAMATH_GPT_abs_sqrt2_sub_2_l1243_124338

theorem abs_sqrt2_sub_2 (h : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2) : |Real.sqrt 2 - 2| = 2 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_sqrt2_sub_2_l1243_124338


namespace NUMINAMATH_GPT_negate_universal_proposition_l1243_124328

theorem negate_universal_proposition : 
  (¬ (∀ x : ℝ, x^2 - 2 * x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negate_universal_proposition_l1243_124328


namespace NUMINAMATH_GPT_sum_of_x_and_y_l1243_124335

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 :=
sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l1243_124335


namespace NUMINAMATH_GPT_clarinet_players_count_l1243_124310

-- Given weights and counts
def weight_trumpet : ℕ := 5
def weight_clarinet : ℕ := 5
def weight_trombone : ℕ := 10
def weight_tuba : ℕ := 20
def weight_drum : ℕ := 15
def count_trumpets : ℕ := 6
def count_trombones : ℕ := 8
def count_tubas : ℕ := 3
def count_drummers : ℕ := 2
def total_weight : ℕ := 245

-- Calculated known weight
def known_weight : ℕ :=
  (count_trumpets * weight_trumpet) +
  (count_trombones * weight_trombone) +
  (count_tubas * weight_tuba) +
  (count_drummers * weight_drum)

-- Weight carried by clarinets
def weight_clarinets : ℕ := total_weight - known_weight

-- Number of clarinet players
def number_of_clarinet_players : ℕ := weight_clarinets / weight_clarinet

theorem clarinet_players_count :
  number_of_clarinet_players = 9 := by
  unfold number_of_clarinet_players
  unfold weight_clarinets
  unfold known_weight
  calc
    (245 - (
      (6 * 5) + 
      (8 * 10) + 
      (3 * 20) + 
      (2 * 15))) / 5 = 9 := by norm_num

end NUMINAMATH_GPT_clarinet_players_count_l1243_124310


namespace NUMINAMATH_GPT_forty_percent_of_number_l1243_124348

theorem forty_percent_of_number (N : ℝ) 
  (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 17) : 
  0.40 * N = 204 :=
sorry

end NUMINAMATH_GPT_forty_percent_of_number_l1243_124348


namespace NUMINAMATH_GPT_reciprocal_neg_3_div_4_l1243_124332

theorem reciprocal_neg_3_div_4 : (- (3 / 4 : ℚ))⁻¹ = -(4 / 3 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_neg_3_div_4_l1243_124332


namespace NUMINAMATH_GPT_total_bananas_bought_l1243_124343

-- Define the conditions
def went_to_store_times : ℕ := 2
def bananas_per_trip : ℕ := 10

-- State the theorem/question and provide the answer
theorem total_bananas_bought : (went_to_store_times * bananas_per_trip) = 20 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_total_bananas_bought_l1243_124343


namespace NUMINAMATH_GPT_inequality_proof_l1243_124313

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) : 
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1243_124313


namespace NUMINAMATH_GPT_exists_square_in_interval_l1243_124322

def x_k (k : ℕ) : ℕ := k * (k + 1) / 2

noncomputable def sum_x (n : ℕ) : ℕ := (List.range n).map x_k |>.sum

theorem exists_square_in_interval (n : ℕ) (hn : n ≥ 10) :
  ∃ m, (sum_x n - x_k n ≤ m^2 ∧ m^2 ≤ sum_x n) :=
by sorry

end NUMINAMATH_GPT_exists_square_in_interval_l1243_124322


namespace NUMINAMATH_GPT_hendricks_payment_l1243_124373

variable (Hendricks Gerald : ℝ)
variable (less_percent : ℝ) (amount_paid : ℝ)

theorem hendricks_payment (h g : ℝ) (h_less_g : h = g * (1 - less_percent)) (g_val : g = amount_paid) (less_percent_val : less_percent = 0.2) (amount_paid_val: amount_paid = 250) :
h = 200 :=
by
  sorry

end NUMINAMATH_GPT_hendricks_payment_l1243_124373


namespace NUMINAMATH_GPT_part_a_part_b_l1243_124334

-- Define the functions K_m and K_4
def K (m : ℕ) (x y z : ℝ) : ℝ :=
  x * (x - y)^m * (x - z)^m + y * (y - x)^m * (y - z)^m + z * (z - x)^m * (z - y)^m

-- Define M
def M (x y z : ℝ) : ℝ :=
  (x - y)^2 * (y - z)^2 * (z - x)^2

-- The proof goals:
-- 1. Prove K_m >= 0 for odd positive integer m
theorem part_a (m : ℕ) (hm : m % 2 = 1) (x y z : ℝ) : 
  0 ≤ K m x y z := 
sorry

-- 2. Prove K_7 + M^2 * K_1 >= M * K_4
theorem part_b (x y z : ℝ) : 
  K 7 x y z + (M x y z)^2 * K 1 x y z ≥ M x y z * K 4 x y z := 
sorry

end NUMINAMATH_GPT_part_a_part_b_l1243_124334


namespace NUMINAMATH_GPT_sqrt_sqr_l1243_124309

theorem sqrt_sqr (x : ℝ) (hx : 0 ≤ x) : (Real.sqrt x) ^ 2 = x := 
by sorry

example : (Real.sqrt 3) ^ 2 = 3 := 
by apply sqrt_sqr; linarith

end NUMINAMATH_GPT_sqrt_sqr_l1243_124309


namespace NUMINAMATH_GPT_employees_salaries_l1243_124372

theorem employees_salaries (M N P : ℝ)
  (hM : M = 1.20 * N)
  (hN_median : N = N) -- Indicates N is the median
  (hP : P = 0.65 * M)
  (h_total : N + M + P = 3200) :
  M = 1288.58 ∧ N = 1073.82 ∧ P = 837.38 :=
by
  sorry

end NUMINAMATH_GPT_employees_salaries_l1243_124372


namespace NUMINAMATH_GPT_two_trucks_carry_2_tons_l1243_124319

theorem two_trucks_carry_2_tons :
  ∀ (truck_capacity : ℕ), truck_capacity = 999 →
  (truck_capacity * 2) / 1000 = 2 :=
by
  intros truck_capacity h_capacity
  rw [h_capacity]
  exact sorry

end NUMINAMATH_GPT_two_trucks_carry_2_tons_l1243_124319


namespace NUMINAMATH_GPT_find_XY_square_l1243_124363

noncomputable def triangleABC := Type

variables (A B C T X Y : triangleABC)
variables (ω : Type) (BT CT BC TX TY XY : ℝ)

axiom acute_scalene_triangle (ABC : triangleABC) : Prop
axiom circumcircle (ABC: triangleABC) (ω: Type) : Prop
axiom tangents_intersect (ω: Type) (B C T: triangleABC) (BT CT : ℝ) : Prop
axiom projections (T: triangleABC) (X: triangleABC) (AB: triangleABC) (Y: triangleABC) (AC: triangleABC) : Prop

axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom TX_TY_XY_relation : TX^2 + TY^2 + XY^2 = 1450

theorem find_XY_square : XY^2 = 841 :=
by { sorry }

end NUMINAMATH_GPT_find_XY_square_l1243_124363


namespace NUMINAMATH_GPT_expression_simplify_l1243_124306

theorem expression_simplify
  (a b : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : b ≠ 0)
  (h₃ : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ - (b / 3)⁻¹) = - (1 / (a * b)) :=
by
  sorry

end NUMINAMATH_GPT_expression_simplify_l1243_124306


namespace NUMINAMATH_GPT_maddie_spent_in_all_l1243_124370

-- Define the given conditions
def white_packs : ℕ := 2
def blue_packs : ℕ := 4
def t_shirts_per_white_pack : ℕ := 5
def t_shirts_per_blue_pack : ℕ := 3
def cost_per_t_shirt : ℕ := 3

-- Define the question as a theorem to be proved
theorem maddie_spent_in_all :
  (white_packs * t_shirts_per_white_pack + blue_packs * t_shirts_per_blue_pack) * cost_per_t_shirt = 66 :=
by 
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_maddie_spent_in_all_l1243_124370


namespace NUMINAMATH_GPT_jasmine_average_pace_l1243_124333

-- Define the conditions given in the problem
def totalDistance : ℝ := 45
def totalTime : ℝ := 9

-- Define the assertion that needs to be proved
theorem jasmine_average_pace : totalDistance / totalTime = 5 :=
by sorry

end NUMINAMATH_GPT_jasmine_average_pace_l1243_124333


namespace NUMINAMATH_GPT_not_possible_coloring_possible_coloring_l1243_124351

-- Problem (a): For n = 2001 and k = 4001, prove that such coloring is not possible.
theorem not_possible_coloring (n : ℕ) (k : ℕ) (h_n : n = 2001) (h_k : k = 4001) :
  ¬ ∃ (color : ℕ × ℕ → ℕ), (∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) → 1 ≤ color (i, j) ∧ color (i, j) ≤ k)
  ∧ (∀ i, 1 ≤ i ∧ i ≤ n → ∀ j1 j2, (1 ≤ j1 ∧ j1 ≤ n) ∧ (1 ≤ j2 ∧ j2 ≤ n) → j1 ≠ j2 → color (i, j1) ≠ color (i, j2))
  ∧ (∀ j, 1 ≤ j ∧ j ≤ n → ∀ i1 i2, (1 ≤ i1 ∧ i1 ≤ n) ∧ (1 ≤ i2 ∧ i2 ≤ n) → i1 ≠ i2 → color (i1, j) ≠ color (i2, j)) := 
sorry

-- Problem (b): For n = 2^m - 1 and k = 2^(m+1) - 1, prove that such coloring is possible.
theorem possible_coloring (m : ℕ) (n k : ℕ) (h_n : n = 2^m - 1) (h_k : k = 2^(m+1) - 1) :
  ∃ (color : ℕ × ℕ → ℕ), (∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) → 1 ≤ color (i, j) ∧ color (i, j) ≤ k)
  ∧ (∀ i, 1 ≤ i ∧ i ≤ n → ∀ j1 j2, (1 ≤ j1 ∧ j1 ≤ n) ∧ (1 ≤ j2 ∧ j2 ≤ n) → j1 ≠ j2 → color (i, j1) ≠ color (i, j2))
  ∧ (∀ j, 1 ≤ j ∧ j ≤ n → ∀ i1 i2, (1 ≤ i1 ∧ i1 ≤ n) ∧ (1 ≤ i2 ∧ i2 ≤ n) → i1 ≠ i2 → color (i1, j) ≠ color (i2, j)) := 
sorry

end NUMINAMATH_GPT_not_possible_coloring_possible_coloring_l1243_124351


namespace NUMINAMATH_GPT_right_triangle_with_a_as_hypotenuse_l1243_124329

theorem right_triangle_with_a_as_hypotenuse
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a = (b^2 + c^2 - a^2) / (2 * b * c))
  (h2 : b = (a^2 + c^2 - b^2) / (2 * a * c))
  (h3 : c = (a^2 + b^2 - c^2) / (2 * a * b))
  (h4 : a * ((b^2 + c^2 - a^2) / (2 * b * c)) + b * ((a^2 + c^2 - b^2) / (2 * a * c)) = c * ((a^2 + b^2 - c^2) / (2 * a * b))) :
  a^2 = b^2 + c^2 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_with_a_as_hypotenuse_l1243_124329


namespace NUMINAMATH_GPT_count_three_digit_numbers_divisible_by_seventeen_l1243_124377

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_divisible_by_seventeen (n : ℕ) : Prop := n % 17 = 0

theorem count_three_digit_numbers_divisible_by_seventeen : 
  ∃ (count : ℕ), count = 53 ∧ 
    (∀ (n : ℕ), is_three_digit_number n → is_divisible_by_seventeen n → response) := 
sorry

end NUMINAMATH_GPT_count_three_digit_numbers_divisible_by_seventeen_l1243_124377


namespace NUMINAMATH_GPT_total_trees_correct_l1243_124339

def apricot_trees : ℕ := 58
def peach_trees : ℕ := 3 * apricot_trees
def total_trees : ℕ := apricot_trees + peach_trees

theorem total_trees_correct : total_trees = 232 :=
by
  sorry

end NUMINAMATH_GPT_total_trees_correct_l1243_124339


namespace NUMINAMATH_GPT_measure_of_B_l1243_124369

theorem measure_of_B (A B C : ℝ) (h1 : B = A + 20) (h2 : C = 50) (h3 : A + B + C = 180) : B = 75 := by
  sorry

end NUMINAMATH_GPT_measure_of_B_l1243_124369


namespace NUMINAMATH_GPT_surveyor_problem_l1243_124395

theorem surveyor_problem
  (GF : ℝ) (G4 : ℝ)
  (hGF : GF = 70)
  (hG4 : G4 = 60) :
  (1/2) * GF * G4 = 2100 := 
  by
  sorry

end NUMINAMATH_GPT_surveyor_problem_l1243_124395


namespace NUMINAMATH_GPT_total_surface_area_correct_l1243_124376

def surface_area_calculation (height_e height_f height_g : ℚ) : ℚ :=
  let top_bottom_area := 4
  let side_area := (height_e + height_f + height_g) * 2
  let front_back_area := 4
  top_bottom_area + side_area + front_back_area

theorem total_surface_area_correct :
  surface_area_calculation (5 / 8) (1 / 4) (9 / 8) = 12 := 
by
  sorry

end NUMINAMATH_GPT_total_surface_area_correct_l1243_124376


namespace NUMINAMATH_GPT_range_of_a_l1243_124383

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 else -x^2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ x ∧ x ≤ a + 2 → f (x + a) ≥ 2 * f x) → a ≥ Real.sqrt 2 :=
by
  -- provided condition
  intros h
  sorry

end NUMINAMATH_GPT_range_of_a_l1243_124383


namespace NUMINAMATH_GPT_min_abs_sum_l1243_124326

theorem min_abs_sum (x : ℝ) : ∃ x : ℝ, (∀ y, abs (y + 3) + abs (y - 2) ≥ abs (x + 3) + abs (x - 2)) ∧ (abs (x + 3) + abs (x - 2) = 5) := sorry

end NUMINAMATH_GPT_min_abs_sum_l1243_124326


namespace NUMINAMATH_GPT_fraction_of_married_men_l1243_124380

/-- At a social gathering, there are only single women and married men with their wives.
     The probability that a randomly selected woman is single is 3/7.
     The fraction of the people in the gathering that are married men is 4/11. -/
theorem fraction_of_married_men (women : ℕ) (single_women : ℕ) (married_men : ℕ) (total_people : ℕ) 
  (h_women_total : women = 7)
  (h_single_women_probability : single_women = women * 3 / 7)
  (h_married_women : women - single_women = married_men)
  (h_total_people : total_people = women + married_men) :
  married_men / total_people = 4 / 11 := 
by sorry

end NUMINAMATH_GPT_fraction_of_married_men_l1243_124380


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l1243_124384

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l1243_124384


namespace NUMINAMATH_GPT_exists_even_in_sequence_l1243_124387

theorem exists_even_in_sequence 
  (a : ℕ → ℕ)
  (h₀ : ∀ n : ℕ, a (n+1) = a n + (a n % 10)) :
  ∃ n : ℕ, a n % 2 = 0 :=
sorry

end NUMINAMATH_GPT_exists_even_in_sequence_l1243_124387


namespace NUMINAMATH_GPT_multiple_of_area_l1243_124346

-- Define the given conditions
def perimeter (s : ℝ) : ℝ := 4 * s
def area (s : ℝ) : ℝ := s * s

theorem multiple_of_area (m s a p : ℝ) 
  (h1 : p = perimeter s)
  (h2 : a = area s)
  (h3 : m * a = 10 * p + 45)
  (h4 : p = 36) : m = 5 :=
by 
  sorry

end NUMINAMATH_GPT_multiple_of_area_l1243_124346


namespace NUMINAMATH_GPT_book_length_l1243_124365

variable (length width perimeter : ℕ)

theorem book_length
  (h1 : perimeter = 100)
  (h2 : width = 20)
  (h3 : perimeter = 2 * (length + width)) :
  length = 30 :=
by sorry

end NUMINAMATH_GPT_book_length_l1243_124365


namespace NUMINAMATH_GPT_cost_of_dried_fruit_l1243_124307

variable (x : ℝ)

theorem cost_of_dried_fruit 
  (h1 : 3 * 12 + 2.5 * x = 56) : 
  x = 8 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_dried_fruit_l1243_124307


namespace NUMINAMATH_GPT_prism_diagonal_correct_l1243_124331

open Real

noncomputable def prism_diagonal_1 := 2 * sqrt 6
noncomputable def prism_diagonal_2 := sqrt 66

theorem prism_diagonal_correct (length width : ℝ) (h1 : length = 8) (h2 : width = 4) :
  (prism_diagonal_1 = 2 * sqrt 6 ∧ prism_diagonal_2 = sqrt 66) :=
by
  sorry

end NUMINAMATH_GPT_prism_diagonal_correct_l1243_124331


namespace NUMINAMATH_GPT_alice_has_ball_after_three_turns_l1243_124367

def probability_Alice_has_ball (turns: ℕ) : ℚ :=
  match turns with
  | 0 => 1 -- Alice starts with the ball
  | _ => sorry -- We would typically calculate this by recursion or another approach.

theorem alice_has_ball_after_three_turns :
  probability_Alice_has_ball 3 = 11 / 27 :=
by
  sorry

end NUMINAMATH_GPT_alice_has_ball_after_three_turns_l1243_124367


namespace NUMINAMATH_GPT_minimum_framing_needed_l1243_124327

-- Definitions given the conditions
def original_width := 5
def original_height := 7
def enlargement_factor := 4
def border_width := 3
def inches_per_foot := 12

-- Conditions translated to definitions
def enlarged_width := original_width * enlargement_factor
def enlarged_height := original_height * enlargement_factor
def bordered_width := enlarged_width + 2 * border_width
def bordered_height := enlarged_height + 2 * border_width
def perimeter := 2 * (bordered_width + bordered_height)
def perimeter_in_feet := perimeter / inches_per_foot

-- Prove that the minimum number of linear feet of framing required is 10 feet
theorem minimum_framing_needed : perimeter_in_feet = 10 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_framing_needed_l1243_124327


namespace NUMINAMATH_GPT_length_of_path_along_arrows_l1243_124390

theorem length_of_path_along_arrows (s : List ℝ) (h : s.sum = 73) :
  (3 * s.sum = 219) :=
by
  sorry

end NUMINAMATH_GPT_length_of_path_along_arrows_l1243_124390


namespace NUMINAMATH_GPT_ratio_of_numbers_l1243_124356

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : 2 * ((a + b) / 2) = Real.sqrt (10 * a * b)) : abs (a / b - 8) < 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l1243_124356


namespace NUMINAMATH_GPT_roots_cubic_eq_l1243_124371

theorem roots_cubic_eq (r s p q : ℝ) (h1 : r + s = p) (h2 : r * s = q) :
    r^3 + s^3 = p^3 - 3 * q * p :=
by
    -- Placeholder for proof
    sorry

end NUMINAMATH_GPT_roots_cubic_eq_l1243_124371


namespace NUMINAMATH_GPT_parabola_equation_l1243_124394

theorem parabola_equation (P : ℝ × ℝ) (hp : P = (4, -2)) : 
  ∃ m : ℝ, (∀ x y : ℝ, (y^2 = m * x) → (x, y) = P) ∧ (m = 1) :=
by
  have m_val : 1 = 1 := rfl
  sorry

end NUMINAMATH_GPT_parabola_equation_l1243_124394


namespace NUMINAMATH_GPT_arccos_sin_three_pi_over_two_eq_pi_l1243_124385

theorem arccos_sin_three_pi_over_two_eq_pi : 
  Real.arccos (Real.sin (3 * Real.pi / 2)) = Real.pi :=
by
  sorry

end NUMINAMATH_GPT_arccos_sin_three_pi_over_two_eq_pi_l1243_124385


namespace NUMINAMATH_GPT_cos_neg_three_pi_over_two_eq_zero_l1243_124349

noncomputable def cos_neg_three_pi_over_two : ℝ :=
  Real.cos (-3 * Real.pi / 2)

theorem cos_neg_three_pi_over_two_eq_zero :
  cos_neg_three_pi_over_two = 0 :=
by
  -- Using trigonometric identities and periodicity of cosine function
  sorry

end NUMINAMATH_GPT_cos_neg_three_pi_over_two_eq_zero_l1243_124349


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_of_sqrt3_l1243_124314

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_of_sqrt3_l1243_124314


namespace NUMINAMATH_GPT_Grant_spending_is_200_l1243_124393

def Juanita_daily_spending (day: String) : Float :=
  if day = "Sunday" then 2.0 else 0.5

def Juanita_weekly_spending : Float :=
  6 * Juanita_daily_spending "weekday" + Juanita_daily_spending "Sunday"

def Juanita_yearly_spending : Float :=
  52 * Juanita_weekly_spending

def Grant_yearly_spending := Juanita_yearly_spending - 60

theorem Grant_spending_is_200 : Grant_yearly_spending = 200 := by
  sorry

end NUMINAMATH_GPT_Grant_spending_is_200_l1243_124393


namespace NUMINAMATH_GPT_balls_in_boxes_l1243_124382

theorem balls_in_boxes :
  let n := 7
  let k := 3
  (Nat.choose (n + k - 1) (k - 1)) = 36 :=
by
  let n := 7
  let k := 3
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l1243_124382


namespace NUMINAMATH_GPT_problem_statement_l1243_124359

variables {x y z w p q : Prop}

theorem problem_statement (h1 : x = y → z ≠ w) (h2 : z = w → p ≠ q) : x ≠ y → p ≠ q :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1243_124359


namespace NUMINAMATH_GPT_simplify_expr_l1243_124337

theorem simplify_expr (x y : ℝ) : 
  (3 * x - 2 * y - 4) * (x + y + 5) - (x + 2 * y + 5) * (3 * x - y - 1) = -4 * x * y - 3 * x - 7 * y - 15 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expr_l1243_124337


namespace NUMINAMATH_GPT_abs_eq_two_implies_l1243_124321

theorem abs_eq_two_implies (x : ℝ) (h : |x - 3| = 2) : x = 5 ∨ x = 1 := 
sorry

end NUMINAMATH_GPT_abs_eq_two_implies_l1243_124321


namespace NUMINAMATH_GPT_binary_division_example_l1243_124304

theorem binary_division_example : 
  let a := 0b10101  -- binary representation of 21
  let b := 0b11     -- binary representation of 3
  let quotient := 0b111  -- binary representation of 7
  a / b = quotient := 
by sorry

end NUMINAMATH_GPT_binary_division_example_l1243_124304


namespace NUMINAMATH_GPT_num_vec_a_exists_l1243_124374

-- Define the vectors and the conditions
def vec_a (x y : ℝ) : (ℝ × ℝ) := (x, y)
def vec_b (x y : ℝ) : (ℝ × ℝ) := (x^2, y^2)
def vec_c : (ℝ × ℝ) := (1, 1)

-- Define the dot product
def dot_prod (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the conditions
def cond_1 (x y : ℝ) : Prop := (x + y = 1)
def cond_2 (x y : ℝ) : Prop := (x^2 / 4 + (1 - x)^2 / 9 = 1)

-- The proof problem statement
theorem num_vec_a_exists : ∃! (x y : ℝ), cond_1 x y ∧ cond_2 x y := by
  sorry

end NUMINAMATH_GPT_num_vec_a_exists_l1243_124374


namespace NUMINAMATH_GPT_dan_age_l1243_124312

theorem dan_age (D : ℕ) (h : D + 20 = 7 * (D - 4)) : D = 8 :=
by
  sorry

end NUMINAMATH_GPT_dan_age_l1243_124312


namespace NUMINAMATH_GPT_fish_worth_bags_of_rice_l1243_124358

variable (f l a r : ℝ)

theorem fish_worth_bags_of_rice
    (h1 : 5 * f = 3 * l)
    (h2 : l = 6 * a)
    (h3 : 2 * a = r) :
    1 / f = 9 / (5 * r) :=
by
  sorry

end NUMINAMATH_GPT_fish_worth_bags_of_rice_l1243_124358


namespace NUMINAMATH_GPT_proportion_in_triangle_l1243_124330

-- Definitions of the variables and conditions
variables {P Q R E : Point}
variables {p q r m n : ℝ}

-- Conditions
def angle_bisector_theorem (h : p = 2 * q) (h1 : m = q + q) (h2 : n = 2 * q) : Prop :=
  ∀ (p q r m n : ℝ), 
  (m / r) = (n / q) ∧ 
  (m + n = p) ∧
  (p = 2 * q)

-- The theorem to be proved
theorem proportion_in_triangle (h : p = 2 * q) (h1 : m / r = n / q) (h2 : m + n = p) : 
  (n / q = 2 * q / (r + q)) :=
by
  sorry

end NUMINAMATH_GPT_proportion_in_triangle_l1243_124330


namespace NUMINAMATH_GPT_cos_4_3pi_add_alpha_l1243_124344

theorem cos_4_3pi_add_alpha (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
    Real.cos (4 * Real.pi / 3 + α) = -1 / 3 := 
by sorry

end NUMINAMATH_GPT_cos_4_3pi_add_alpha_l1243_124344


namespace NUMINAMATH_GPT_min_sum_of_dimensions_l1243_124320

theorem min_sum_of_dimensions (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_vol : a * b * c = 3003) : 
  a + b + c ≥ 57 := sorry

end NUMINAMATH_GPT_min_sum_of_dimensions_l1243_124320


namespace NUMINAMATH_GPT_car_win_probability_l1243_124353

-- Definitions from conditions
def total_cars : ℕ := 12
def p_X : ℚ := 1 / 6
def p_Y : ℚ := 1 / 10
def p_Z : ℚ := 1 / 8

-- Proof statement: The probability that one of the cars X, Y, or Z will win is 47/120
theorem car_win_probability : p_X + p_Y + p_Z = 47 / 120 := by
  sorry

end NUMINAMATH_GPT_car_win_probability_l1243_124353


namespace NUMINAMATH_GPT_destiny_cookies_divisible_l1243_124361

theorem destiny_cookies_divisible (C : ℕ) (h : C % 6 = 0) : ∃ k : ℕ, C = 6 * k :=
by {
  sorry
}

end NUMINAMATH_GPT_destiny_cookies_divisible_l1243_124361


namespace NUMINAMATH_GPT_polygon_sides_l1243_124360

theorem polygon_sides (n : ℕ) (h : 44 = n * (n - 3) / 2) : n = 11 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l1243_124360


namespace NUMINAMATH_GPT_painting_time_equation_l1243_124392

theorem painting_time_equation (t : ℝ) :
  (1/6 + 1/8) * (t - 2) = 1 :=
sorry

end NUMINAMATH_GPT_painting_time_equation_l1243_124392


namespace NUMINAMATH_GPT_calculate_x_l1243_124378

theorem calculate_x (a b x : ℕ) (h1 : b = 9) (h2 : b - a = 5) (h3 : a * b = 2 * (a + b) + x) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_calculate_x_l1243_124378


namespace NUMINAMATH_GPT_mira_weekly_distance_l1243_124379

noncomputable def total_distance_jogging : ℝ :=
  let monday_distance := 4 * 2
  let thursday_distance := 5 * 1.5
  monday_distance + thursday_distance

noncomputable def total_distance_swimming : ℝ :=
  2 * 1

noncomputable def total_distance_cycling : ℝ :=
  12 * 1

noncomputable def total_distance : ℝ :=
  total_distance_jogging + total_distance_swimming + total_distance_cycling

theorem mira_weekly_distance : total_distance = 29.5 := by
  unfold total_distance
  unfold total_distance_jogging
  unfold total_distance_swimming
  unfold total_distance_cycling
  sorry

end NUMINAMATH_GPT_mira_weekly_distance_l1243_124379


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1243_124350

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D F : V)

-- Problem 1:
theorem simplify_expression1 : 
  (D - C) + (C - B) + (B - A) = D - A := 
sorry

-- Problem 2:
theorem simplify_expression2 : 
  (B - A) + (F - D) + (D - C) + (C - B) + (A - F) = 0 := 
sorry

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1243_124350


namespace NUMINAMATH_GPT_possible_roots_l1243_124316

theorem possible_roots (a b p q : ℤ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : a ≠ b)
  (h4 : p = -(a + b))
  (h5 : q = ab)
  (h6 : (a + p) % (q - 2 * b) = 0) :
  a = 1 ∨ a = 3 :=
  sorry

end NUMINAMATH_GPT_possible_roots_l1243_124316


namespace NUMINAMATH_GPT_exp_mul_l1243_124398

variable {a : ℝ}

-- Define a theorem stating the problem: proof that a^2 * a^3 = a^5
theorem exp_mul (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_GPT_exp_mul_l1243_124398


namespace NUMINAMATH_GPT_infinite_solutions_x2_y3_z5_l1243_124305

theorem infinite_solutions_x2_y3_z5 :
  ∃ (t : ℕ), ∃ (x y z : ℕ), x = 2^(15*t + 12) ∧ y = 2^(10*t + 8) ∧ z = 2^(6*t + 5) ∧ (x^2 + y^3 = z^5) :=
sorry

end NUMINAMATH_GPT_infinite_solutions_x2_y3_z5_l1243_124305


namespace NUMINAMATH_GPT_sum_of_roots_l1243_124399

theorem sum_of_roots (r s t : ℝ) (hroots : 3 * (r^3 + s^3 + t^3) + 9 * (r^2 + s^2 + t^2) - 36 * (r + s + t) + 12 = 0) :
  r + s + t = -3 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1243_124399


namespace NUMINAMATH_GPT_solid_with_square_views_is_cube_l1243_124308

-- Define the conditions and the solid type
def is_square_face (view : Type) : Prop := 
  -- Definition to characterize a square view. This is general,
  -- as the detailed characterization of a 'square' in Lean would depend
  -- on more advanced geometry modules, assuming a simple predicate here.
  sorry

structure Solid := (front_view : Type) (top_view : Type) (left_view : Type)

-- Conditions indicating that all views are squares
def all_views_square (S : Solid) : Prop :=
  is_square_face S.front_view ∧ is_square_face S.top_view ∧ is_square_face S.left_view

-- The theorem we are aiming to prove
theorem solid_with_square_views_is_cube (S : Solid) (h : all_views_square S) : S = {front_view := ℝ, top_view := ℝ, left_view := ℝ} := sorry

end NUMINAMATH_GPT_solid_with_square_views_is_cube_l1243_124308


namespace NUMINAMATH_GPT_original_rectangle_area_at_least_90_l1243_124368

variable (a b c x y z : ℝ)
variable (hx1 : a * x = 1)
variable (hx2 : c * x = 3)
variable (hy : b * y = 10)
variable (hz : a * z = 9)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hx : 0 < x) (hy' : 0 < y) (hz' : 0 < z)

theorem original_rectangle_area_at_least_90 : ∀ {a b c x y z : ℝ},
  (a * x = 1) →
  (c * x = 3) →
  (b * y = 10) →
  (a * z = 9) →
  (0 < a) →
  (0 < b) →
  (0 < c) →
  (0 < x) →
  (0 < y) →
  (0 < z) →
  (a + b + c) * (x + y + z) ≥ 90 :=
sorry

end NUMINAMATH_GPT_original_rectangle_area_at_least_90_l1243_124368


namespace NUMINAMATH_GPT_toothpicks_15_l1243_124347

noncomputable def toothpicks : ℕ → ℕ
| 0       => 0  -- since the stage count n >= 1, stage 0 is not required, default 0.
| 1       => 5
| (n + 1) => 2 * toothpicks n + 2

theorem toothpicks_15 : toothpicks 15 = 32766 := by
  sorry

end NUMINAMATH_GPT_toothpicks_15_l1243_124347


namespace NUMINAMATH_GPT_num_undefined_values_l1243_124345

theorem num_undefined_values :
  ∃! x : Finset ℝ, (∀ y ∈ x, (y + 5 = 0) ∨ (y - 1 = 0) ∨ (y - 4 = 0)) ∧ (x.card = 3) := sorry

end NUMINAMATH_GPT_num_undefined_values_l1243_124345


namespace NUMINAMATH_GPT_shuai_fen_ratio_l1243_124362

theorem shuai_fen_ratio 
  (C : ℕ) (B_and_D : ℕ) (a : ℕ) (x : ℚ) 
  (hC : C = 36) (hB_and_D : B_and_D = 75) :
  (x = 0.25) ∧ (a = 175) := 
by {
  -- This is where the proof steps would go
  sorry
}

end NUMINAMATH_GPT_shuai_fen_ratio_l1243_124362


namespace NUMINAMATH_GPT_principal_amount_unique_l1243_124389

theorem principal_amount_unique (SI R T : ℝ) (P : ℝ) : 
  SI = 4016.25 → R = 14 → T = 5 → SI = (P * R * T) / 100 → P = 5737.5 :=
by
  intro h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃] at h₄
  sorry

end NUMINAMATH_GPT_principal_amount_unique_l1243_124389


namespace NUMINAMATH_GPT_travel_time_to_Virgo_island_l1243_124396

theorem travel_time_to_Virgo_island (boat_time : ℝ) (plane_time : ℝ) (total_time : ℝ) 
  (h1 : boat_time ≤ 2) (h2 : plane_time = 4 * boat_time) (h3 : total_time = plane_time + boat_time) : 
  total_time = 10 :=
by
  sorry

end NUMINAMATH_GPT_travel_time_to_Virgo_island_l1243_124396


namespace NUMINAMATH_GPT_sara_total_score_l1243_124354

-- Definitions based on the conditions
def correct_points (correct_answers : Nat) : Int := correct_answers * 2
def incorrect_points (incorrect_answers : Nat) : Int := incorrect_answers * (-1)
def unanswered_points (unanswered_questions : Nat) : Int := unanswered_questions * 0

def total_score (correct_answers incorrect_answers unanswered_questions : Nat) : Int :=
  correct_points correct_answers + incorrect_points incorrect_answers + unanswered_points unanswered_questions

-- The main theorem stating the problem requirement
theorem sara_total_score :
  total_score 18 10 2 = 26 :=
by
  sorry

end NUMINAMATH_GPT_sara_total_score_l1243_124354


namespace NUMINAMATH_GPT_tan_square_B_eq_tan_A_tan_C_range_l1243_124323

theorem tan_square_B_eq_tan_A_tan_C_range (A B C : ℝ) (h_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π) 
  (h_tan : Real.tan B * Real.tan B = Real.tan A * Real.tan C) : (π / 3) ≤ B ∧ B < (π / 2) :=
by
  sorry

end NUMINAMATH_GPT_tan_square_B_eq_tan_A_tan_C_range_l1243_124323


namespace NUMINAMATH_GPT_fraction_to_decimal_and_add_l1243_124311

theorem fraction_to_decimal_and_add (a b : ℚ) (h : a = 7 / 16) : (a + b) = 2.4375 ↔ b = 2 :=
by
   sorry

end NUMINAMATH_GPT_fraction_to_decimal_and_add_l1243_124311


namespace NUMINAMATH_GPT_find_quotient_l1243_124300

theorem find_quotient
  (larger smaller : ℕ)
  (h1 : larger - smaller = 1200)
  (h2 : larger = 1495)
  (rem : ℕ := 4)
  (h3 : larger % smaller = rem) :
  larger / smaller = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_quotient_l1243_124300


namespace NUMINAMATH_GPT_dasha_flags_proof_l1243_124341

variable (Tata_flags_right Yasha_flags_right Vera_flags_right Maxim_flags_right : ℕ)
variable (Total_flags : ℕ)

theorem dasha_flags_proof 
  (hTata: Tata_flags_right = 14)
  (hYasha: Yasha_flags_right = 32)
  (hVera: Vera_flags_right = 20)
  (hMaxim: Maxim_flags_right = 8)
  (hTotal: Total_flags = 37) :
  ∃ (Dasha_flags : ℕ), Dasha_flags = 8 :=
by
  sorry

end NUMINAMATH_GPT_dasha_flags_proof_l1243_124341


namespace NUMINAMATH_GPT_a2_a8_sum_l1243_124325

variable {a : ℕ → ℝ}  -- Define the arithmetic sequence a

-- Conditions:
axiom arithmetic_sequence (n : ℕ) : a (n + 1) - a n = a 1 - a 0
axiom a1_a9_sum : a 1 + a 9 = 8

-- Theorem stating the question and the answer
theorem a2_a8_sum : a 2 + a 8 = 8 :=
by
  sorry

end NUMINAMATH_GPT_a2_a8_sum_l1243_124325


namespace NUMINAMATH_GPT_prove_ordered_triple_l1243_124318

theorem prove_ordered_triple (x y z : ℝ) (h1 : x > 2) (h2 : y > 2) (h3 : z > 2)
  (h4 : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) : 
  (x, y, z) = (13, 11, 6) :=
sorry

end NUMINAMATH_GPT_prove_ordered_triple_l1243_124318


namespace NUMINAMATH_GPT_tan_difference_l1243_124397

theorem tan_difference (α β : ℝ) (hα : Real.tan α = 5) (hβ : Real.tan β = 3) : 
    Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_tan_difference_l1243_124397


namespace NUMINAMATH_GPT_bulgarian_inequality_l1243_124340

theorem bulgarian_inequality (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) :
    (a^4 / (a^3 + a^2 * b + a * b^2 + b^3) + 
     b^4 / (b^3 + b^2 * c + b * c^2 + c^3) + 
     c^4 / (c^3 + c^2 * d + c * d^2 + d^3) + 
     d^4 / (d^3 + d^2 * a + d * a^2 + a^3)) 
    ≥ (a + b + c + d) / 4 :=
sorry

end NUMINAMATH_GPT_bulgarian_inequality_l1243_124340


namespace NUMINAMATH_GPT_tangent_line_at_x0_minimum_value_on_interval_minimum_value_on_interval_high_minimum_value_on_interval_mid_l1243_124386

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.exp x

theorem tangent_line_at_x0 (a : ℝ) (h : a = 2) : 
    (∃ m b : ℝ, (∀ x : ℝ, f x a = m * x + b) ∧ m = -1 ∧ b = -2) :=
by 
    sorry

theorem minimum_value_on_interval (a : ℝ) :
    (1 ≤ a) → (a ≤ 2) → f 1 a = (1 - a) * Real.exp 1 :=
by 
    sorry

theorem minimum_value_on_interval_high (a : ℝ) :
    (a ≥ 3) → f 2 a = (2 - a) * Real.exp 2 :=
by 
    sorry

theorem minimum_value_on_interval_mid (a : ℝ) :
    (2 < a) → (a < 3) → f (a - 1) a = -(Real.exp (a - 1)) :=
by 
    sorry

end NUMINAMATH_GPT_tangent_line_at_x0_minimum_value_on_interval_minimum_value_on_interval_high_minimum_value_on_interval_mid_l1243_124386


namespace NUMINAMATH_GPT_variance_of_dataSet_l1243_124336

-- Define the given data set
def dataSet : List ℤ := [-2, -1, 0, 1, 2]

-- Define the function to calculate mean
def mean (data : List ℤ) : ℚ :=
  (data.sum : ℚ) / data.length

-- Define the function to calculate variance
def variance (data : List ℤ) : ℚ :=
  let μ := mean data
  (data.map (λ x => (x - μ) ^ 2)).sum / data.length

-- State the theorem: The variance of the given data set is 2
theorem variance_of_dataSet : variance dataSet = 2 := by
  sorry

end NUMINAMATH_GPT_variance_of_dataSet_l1243_124336


namespace NUMINAMATH_GPT_kolya_pays_90_rubles_l1243_124381

theorem kolya_pays_90_rubles {x y : ℝ} 
  (h1 : x + 3 * y = 78) 
  (h2 : x + 8 * y = 108) :
  x + 5 * y = 90 :=
by sorry

end NUMINAMATH_GPT_kolya_pays_90_rubles_l1243_124381


namespace NUMINAMATH_GPT_prob_defective_first_draw_prob_defective_both_draws_prob_defective_second_given_first_l1243_124388

-- Definitions
def total_products := 20
def defective_products := 5

-- Probability of drawing a defective product on the first draw
theorem prob_defective_first_draw : (defective_products / total_products : ℚ) = 1 / 4 :=
sorry

-- Probability of drawing defective products on both the first and the second draws
theorem prob_defective_both_draws : (defective_products / total_products * (defective_products - 1) / (total_products - 1) : ℚ) = 1 / 19 :=
sorry

-- Probability of drawing a defective product on the second draw given that the first was defective
theorem prob_defective_second_given_first : ((defective_products - 1) / (total_products - 1) / (defective_products / total_products) : ℚ) = 4 / 19 :=
sorry

end NUMINAMATH_GPT_prob_defective_first_draw_prob_defective_both_draws_prob_defective_second_given_first_l1243_124388


namespace NUMINAMATH_GPT_password_guess_probability_l1243_124302

def probability_correct_digit_within_two_attempts : Prop :=
  let total_digits := 10
  let prob_first_attempt := 1 / total_digits
  let prob_second_attempt := (9 / total_digits) * (1 / (total_digits - 1))
  (prob_first_attempt + prob_second_attempt) = 1 / 5

theorem password_guess_probability :
  probability_correct_digit_within_two_attempts :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_password_guess_probability_l1243_124302


namespace NUMINAMATH_GPT_largest_prime_mersenne_below_500_l1243_124357

def is_mersenne (m : ℕ) (n : ℕ) := m = 2^n - 1
def is_power_of_2 (n : ℕ) := ∃ (k : ℕ), n = 2^k

theorem largest_prime_mersenne_below_500 : ∀ (m : ℕ), 
  m < 500 →
  (∃ n, is_power_of_2 n ∧ is_mersenne m n ∧ Nat.Prime m) →
  m ≤ 3 := 
by
  sorry

end NUMINAMATH_GPT_largest_prime_mersenne_below_500_l1243_124357


namespace NUMINAMATH_GPT_math_quiz_scores_stability_l1243_124324

theorem math_quiz_scores_stability :
  let avgA := (90 + 82 + 88 + 96 + 94) / 5
  let avgB := (94 + 86 + 88 + 90 + 92) / 5
  let varA := ((90 - avgA) ^ 2 + (82 - avgA) ^ 2 + (88 - avgA) ^ 2 + (96 - avgA) ^ 2 + (94 - avgA) ^ 2) / 5
  let varB := ((94 - avgB) ^ 2 + (86 - avgB) ^ 2 + (88 - avgB) ^ 2 + (90 - avgB) ^ 2 + (92 - avgB) ^ 2) / 5
  avgA = avgB ∧ varB < varA :=
by
  sorry

end NUMINAMATH_GPT_math_quiz_scores_stability_l1243_124324


namespace NUMINAMATH_GPT_regular_polygon_sides_eq_seven_l1243_124301

theorem regular_polygon_sides_eq_seven (n : ℕ) (h1 : D = n * (n-3) / 2) (h2 : D = 2 * n) : n = 7 := 
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_eq_seven_l1243_124301


namespace NUMINAMATH_GPT_gcd_ab_eq_one_l1243_124375

def a : ℕ := 97^10 + 1
def b : ℕ := 97^10 + 97^3 + 1

theorem gcd_ab_eq_one : Nat.gcd a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_ab_eq_one_l1243_124375


namespace NUMINAMATH_GPT_matt_current_age_is_65_l1243_124315

variable (matt_age james_age : ℕ)

def james_current_age := 30
def james_age_in_5_years := james_current_age + 5
def matt_age_in_5_years := 2 * james_age_in_5_years
def matt_current_age := matt_age_in_5_years - 5

theorem matt_current_age_is_65 : matt_current_age = 65 := 
by
  -- sorry is here to skip the proof.
  sorry

end NUMINAMATH_GPT_matt_current_age_is_65_l1243_124315


namespace NUMINAMATH_GPT_coin_toss_probability_l1243_124342

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end NUMINAMATH_GPT_coin_toss_probability_l1243_124342


namespace NUMINAMATH_GPT_fx_leq_one_l1243_124352

noncomputable def f (x : ℝ) : ℝ := (x + 1) / Real.exp x

theorem fx_leq_one : ∀ x : ℝ, f x ≤ 1 := by
  sorry

end NUMINAMATH_GPT_fx_leq_one_l1243_124352


namespace NUMINAMATH_GPT_find_k_max_product_l1243_124364

theorem find_k_max_product : 
  (∃ k : ℝ, (3 : ℝ) * (x ^ 2) - 4 * x + k = 0 ∧ 16 - 12 * k ≥ 0 ∧ (∀ x1 x2 : ℝ, x1 * x2 = k / 3 → x1 + x2 = 4 / 3 → x1 * x2 ≤ (2 / 3) ^ 2)) →
  k = 4 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_k_max_product_l1243_124364


namespace NUMINAMATH_GPT_major_axis_length_l1243_124366

theorem major_axis_length (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 = 8 :=
by
  sorry

end NUMINAMATH_GPT_major_axis_length_l1243_124366


namespace NUMINAMATH_GPT_multiplication_problem_l1243_124303

-- Define the problem in Lean 4.
theorem multiplication_problem (a b : ℕ) (ha : a < 10) (hb : b < 10) 
  (h : (30 + a) * (10 * b + 4) = 126) : a + b = 7 :=
sorry

end NUMINAMATH_GPT_multiplication_problem_l1243_124303


namespace NUMINAMATH_GPT_circle_points_l1243_124355

noncomputable def proof_problem (x1 y1 x2 y2: ℝ) : Prop :=
  (x1^2 + y1^2 = 4) ∧ (x2^2 + y2^2 = 4) ∧ ((x1 - x2)^2 + (y1 - y2)^2 = 12) →
    (x1 * x2 + y1 * y2 = -2)

theorem circle_points (x1 y1 x2 y2 : ℝ) : proof_problem x1 y1 x2 y2 := 
by
  sorry

end NUMINAMATH_GPT_circle_points_l1243_124355


namespace NUMINAMATH_GPT_right_triangle_count_l1243_124317

theorem right_triangle_count :
  ∃! (a b : ℕ), (a^2 + b^2 = (b + 3)^2) ∧ (b < 50) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_count_l1243_124317
