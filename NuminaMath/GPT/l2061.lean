import Mathlib

namespace NUMINAMATH_GPT_largest_fraction_l2061_206149

theorem largest_fraction (A B C D E : ℚ)
    (hA: A = 5 / 11)
    (hB: B = 7 / 16)
    (hC: C = 23 / 50)
    (hD: D = 99 / 200)
    (hE: E = 202 / 403) : 
    E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  sorry

end NUMINAMATH_GPT_largest_fraction_l2061_206149


namespace NUMINAMATH_GPT_x1_value_l2061_206182

theorem x1_value (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h2 : (1 - x1)^2 + 2 * (x1 - x2)^2 + 2 * (x2 - x3)^2 + x3^2 = 1 / 2) : 
  x1 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_x1_value_l2061_206182


namespace NUMINAMATH_GPT_Peter_bought_4_notebooks_l2061_206156

theorem Peter_bought_4_notebooks :
  (let green_notebooks := 2
   let black_notebook := 1
   let pink_notebook := 1
   green_notebooks + black_notebook + pink_notebook = 4) :=
by sorry

end NUMINAMATH_GPT_Peter_bought_4_notebooks_l2061_206156


namespace NUMINAMATH_GPT_mom_t_shirts_total_l2061_206115

-- Definitions based on the conditions provided in the problem
def packages : ℕ := 71
def t_shirts_per_package : ℕ := 6

-- The statement to prove that the total number of white t-shirts is 426
theorem mom_t_shirts_total : packages * t_shirts_per_package = 426 := by sorry

end NUMINAMATH_GPT_mom_t_shirts_total_l2061_206115


namespace NUMINAMATH_GPT_min_x_4y_is_minimum_l2061_206172

noncomputable def min_value_x_4y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1 / x) + (1 / (2 * y)) = 2) : ℝ :=
  x + 4 * y

theorem min_x_4y_is_minimum : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (1 / x + 1 / (2 * y) = 2) ∧ (x + 4 * y = (3 / 2) + Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_min_x_4y_is_minimum_l2061_206172


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l2061_206111

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l2061_206111


namespace NUMINAMATH_GPT_max_score_per_student_l2061_206173

theorem max_score_per_student (score_tests : ℕ → ℕ) (avg_score_tests_lt_8 : ℕ) (combined_score_two_tests : ℕ) : (∀ i, 1 ≤ i ∧ i ≤ 8 → score_tests i ≤ 100) ∧ avg_score_tests_lt_8 = 70 ∧ combined_score_two_tests = 290 →
  ∃ max_score : ℕ, max_score = 145 := 
by
  sorry

end NUMINAMATH_GPT_max_score_per_student_l2061_206173


namespace NUMINAMATH_GPT_number_of_tests_initially_l2061_206167

-- Given conditions
variables (n S : ℕ)
variables (h1 : S / n = 70)
variables (h2 : S = 70 * n)
variables (h3 : (S - 55) / (n - 1) = 75)

-- Prove the number of tests initially, n, is 4.
theorem number_of_tests_initially (n : ℕ) (S : ℕ)
  (h1 : S / n = 70) (h2 : S = 70 * n) (h3 : (S - 55) / (n - 1) = 75) :
  n = 4 :=
sorry

end NUMINAMATH_GPT_number_of_tests_initially_l2061_206167


namespace NUMINAMATH_GPT_marcia_minutes_worked_l2061_206101

/--
If Marcia worked for 5 hours on her science project,
then she worked for 300 minutes.
-/
theorem marcia_minutes_worked (hours : ℕ) (h : hours = 5) : (hours * 60) = 300 := by
  sorry

end NUMINAMATH_GPT_marcia_minutes_worked_l2061_206101


namespace NUMINAMATH_GPT_max_candy_one_student_l2061_206155

theorem max_candy_one_student (n : ℕ) (mu : ℕ) (at_least_two : ℕ → Prop) :
  n = 35 → mu = 6 →
  (∀ x, at_least_two x → x ≥ 2) →
  ∃ max_candy : ℕ, (∀ x, at_least_two x → x ≤ max_candy) ∧ max_candy = 142 :=
by
sorry

end NUMINAMATH_GPT_max_candy_one_student_l2061_206155


namespace NUMINAMATH_GPT_geometric_sequence_problem_l2061_206108

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

def condition (a : ℕ → ℝ) : Prop :=
a 4 + a 8 = -3

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : condition a) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l2061_206108


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l2061_206129

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l2061_206129


namespace NUMINAMATH_GPT_max_triangle_area_l2061_206185

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4 * y

theorem max_triangle_area
  (x1 y1 x2 y2 : ℝ)
  (hA : parabola x1 y1)
  (hB : parabola x2 y2)
  (h_sum_y : y1 + y2 = 2)
  (h_neq : y1 ≠ y2) :
  ∃ area : ℝ, area = 121 / 12 :=
sorry

end NUMINAMATH_GPT_max_triangle_area_l2061_206185


namespace NUMINAMATH_GPT_algebraic_expression_value_l2061_206174

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m = 1) : 
  (m - 1)^2 + (m + 1) * (m - 1) + 2022 = 2024 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2061_206174


namespace NUMINAMATH_GPT_cost_of_each_ruler_l2061_206163
-- Import the necessary library

-- Define the conditions and statement
theorem cost_of_each_ruler (students : ℕ) (rulers_each : ℕ) (cost_per_ruler : ℕ) (total_cost : ℕ) 
  (cond1 : students = 42)
  (cond2 : students / 2 < 42 / 2)
  (cond3 : cost_per_ruler > rulers_each)
  (cond4 : students * rulers_each * cost_per_ruler = 2310) : 
  cost_per_ruler = 11 :=
sorry

end NUMINAMATH_GPT_cost_of_each_ruler_l2061_206163


namespace NUMINAMATH_GPT_monroe_collection_legs_l2061_206195

theorem monroe_collection_legs : 
  let ants := 12 
  let spiders := 8 
  let beetles := 15 
  let centipedes := 5 
  let legs_ants := 6 
  let legs_spiders := 8 
  let legs_beetles := 6 
  let legs_centipedes := 100
  (ants * legs_ants + spiders * legs_spiders + beetles * legs_beetles + centipedes * legs_centipedes = 726) := 
by 
  sorry

end NUMINAMATH_GPT_monroe_collection_legs_l2061_206195


namespace NUMINAMATH_GPT_sum_of_50th_row_l2061_206120

-- Define triangular numbers
def T (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of numbers in the nth row
def f (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1 -- T_1 is 1 for the base case
  else 2 * f (n - 1) + n * (n + 1)

-- Prove the sum of the 50th row
theorem sum_of_50th_row : f 50 = 2^50 - 2550 := 
  sorry

end NUMINAMATH_GPT_sum_of_50th_row_l2061_206120


namespace NUMINAMATH_GPT_smallest_lcm_of_4_digit_integers_with_gcd_5_l2061_206192

-- Definition of the given integers k and l
def positive_4_digit_integers (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- The main theorem we want to prove
theorem smallest_lcm_of_4_digit_integers_with_gcd_5 :
  ∃ (k l : ℕ), positive_4_digit_integers k ∧ positive_4_digit_integers l ∧ gcd k l = 5 ∧ lcm k l = 201000 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_lcm_of_4_digit_integers_with_gcd_5_l2061_206192


namespace NUMINAMATH_GPT_time_to_plough_together_l2061_206165

def work_rate_r := 1 / 15
def work_rate_s := 1 / 20
def combined_work_rate := work_rate_r + work_rate_s
def total_field := 1
def T := total_field / combined_work_rate

theorem time_to_plough_together : T = 60 / 7 :=
by
  -- Here you would provide the proof steps if it were required
  -- Since the proof steps are not needed, we indicate the end with sorry
  sorry

end NUMINAMATH_GPT_time_to_plough_together_l2061_206165


namespace NUMINAMATH_GPT_arithmetic_seq_max_S_l2061_206181

theorem arithmetic_seq_max_S {S : ℕ → ℝ} (h1 : S 2023 > 0) (h2 : S 2024 < 0) : S 1012 > S 1013 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_max_S_l2061_206181


namespace NUMINAMATH_GPT_no_solution_for_squares_l2061_206177

theorem no_solution_for_squares (x y : ℤ) (hx : x > 0) (hy : y > 0) :
  ¬ ∃ k m : ℤ, x^2 + y + 2 = k^2 ∧ y^2 + 4 * x = m^2 :=
sorry

end NUMINAMATH_GPT_no_solution_for_squares_l2061_206177


namespace NUMINAMATH_GPT_smallest_value_l2061_206138

noncomputable def smallest_possible_value (a b : ℝ) : ℝ := 2 * a + b

theorem smallest_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 ≥ 3 * b) (h4 : b^2 ≥ (8 / 9) * a) :
  smallest_possible_value a b = 5.602 :=
sorry

end NUMINAMATH_GPT_smallest_value_l2061_206138


namespace NUMINAMATH_GPT_common_ratio_l2061_206148

-- Problem Statement Definitions
variable (a1 q : ℝ)

-- Given Conditions
def a3 := a1 * q^2
def S3 := a1 * (1 + q + q^2)

-- Proof Statement
theorem common_ratio (h1 : a3 = 3/2) (h2 : S3 = 9/2) : q = 1 ∨ q = -1/2 := by
  sorry

end NUMINAMATH_GPT_common_ratio_l2061_206148


namespace NUMINAMATH_GPT_cooking_ways_l2061_206183

noncomputable def comb (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem cooking_ways : comb 5 2 = 10 :=
  by
  sorry

end NUMINAMATH_GPT_cooking_ways_l2061_206183


namespace NUMINAMATH_GPT_problem_1_problem_2_l2061_206191

theorem problem_1 (a b c d : ℝ) (h : d > 0) (h_sum : a + b + c + d = 3) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 3 / 4 := 
sorry

theorem problem_2 (a b c d : ℝ) (h : d > 0) (h_sum : a + b + c + d = 3) :
  (a / (b + 2 * c + 3 * d) + b / (c + 2 * d + 3 * a) + c / (d + 2 * a + 3 * b) + d / (a + 2 * b + 3 * c)) ≥ 2 / 3 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2061_206191


namespace NUMINAMATH_GPT_solve_for_n_l2061_206102

theorem solve_for_n (n : ℕ) (h1 : n > 2)
  (h2 : 6 * (n - 2) ^ 2 = 12 * 12 * (n - 2)) :
  n = 26 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_n_l2061_206102


namespace NUMINAMATH_GPT_cost_to_fix_car_l2061_206132

variable {S A : ℝ}

theorem cost_to_fix_car (h1 : A = 3 * S + 50) (h2 : S + A = 450) : A = 350 := 
by
  sorry

end NUMINAMATH_GPT_cost_to_fix_car_l2061_206132


namespace NUMINAMATH_GPT_parametric_circle_section_l2061_206198

theorem parametric_circle_section (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  ∃ (x y : ℝ), (x = 4 - Real.cos θ ∧ y = 1 - Real.sin θ) ∧ (4 - x)^2 + (1 - y)^2 = 1 :=
sorry

end NUMINAMATH_GPT_parametric_circle_section_l2061_206198


namespace NUMINAMATH_GPT_digit_B_divisible_by_9_l2061_206197

theorem digit_B_divisible_by_9 (B : ℕ) (k : ℤ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 2 * B + 6 = 9 * k) : B = 6 := 
by
  sorry

end NUMINAMATH_GPT_digit_B_divisible_by_9_l2061_206197


namespace NUMINAMATH_GPT_avg_difference_in_circumferences_l2061_206171

-- Define the conditions
def inner_circle_diameter : ℝ := 30
def min_track_width : ℝ := 10
def max_track_width : ℝ := 15

-- Define the average difference in the circumferences of the two circles
theorem avg_difference_in_circumferences :
  let avg_width := (min_track_width + max_track_width) / 2
  let outer_circle_diameter := inner_circle_diameter + 2 * avg_width
  let inner_circle_circumference := Real.pi * inner_circle_diameter
  let outer_circle_circumference := Real.pi * outer_circle_diameter
  outer_circle_circumference - inner_circle_circumference = 25 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_avg_difference_in_circumferences_l2061_206171


namespace NUMINAMATH_GPT_problem_am_gm_inequality_l2061_206143

theorem problem_am_gm_inequality
  (a b c : ℝ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_sq : a^2 + b^2 + c^2 = 3) : 
  (1 / (1 + a * b)) + (1 / (1 + b * c)) + (1 / (1 + c * a)) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_am_gm_inequality_l2061_206143


namespace NUMINAMATH_GPT_painting_price_difference_l2061_206187

theorem painting_price_difference :
  let previous_painting := 9000
  let recent_painting := 44000
  let five_times_more := 5 * previous_painting + previous_painting
  five_times_more - recent_painting = 10000 :=
by
  intros
  sorry

end NUMINAMATH_GPT_painting_price_difference_l2061_206187


namespace NUMINAMATH_GPT_perp_bisector_of_AB_l2061_206193

noncomputable def perpendicular_bisector_eq : Prop :=
  ∀ (x y : ℝ), (x - y + 1 = 0) ∧ (x^2 + y^2 = 1) → (x + y = 0)

-- The proof is omitted
theorem perp_bisector_of_AB : perpendicular_bisector_eq :=
sorry

end NUMINAMATH_GPT_perp_bisector_of_AB_l2061_206193


namespace NUMINAMATH_GPT_ron_l2061_206103

-- Definitions for the given problem conditions
def cost_of_chocolate_bar : ℝ := 1.5
def s'mores_per_chocolate_bar : ℕ := 3
def number_of_scouts : ℕ := 15
def s'mores_per_scout : ℕ := 2

-- Proof that Ron will spend $15.00 on chocolate bars
theorem ron's_chocolate_bar_cost :
  (number_of_scouts * s'mores_per_scout / s'mores_per_chocolate_bar) * cost_of_chocolate_bar = 15 :=
by
  sorry

end NUMINAMATH_GPT_ron_l2061_206103


namespace NUMINAMATH_GPT_distance_internal_tangent_l2061_206157

noncomputable def radius_O := 5
noncomputable def distance_external := 9

theorem distance_internal_tangent (radius_O radius_dist_external : ℝ) 
  (h1 : radius_O = 5) (h2: radius_dist_external = 9) : 
  ∃ r : ℝ, r = 4 ∧ abs (r - radius_O) = 1 := by
  sorry

end NUMINAMATH_GPT_distance_internal_tangent_l2061_206157


namespace NUMINAMATH_GPT_parameter_conditions_l2061_206131

theorem parameter_conditions (p x y : ℝ) :
  (x - p)^2 = 16 * (y - 3 + p) →
  y^2 + ((x - 3) / (|x| - 3))^2 = 1 →
  |x| ≠ 3 →
  p > 3 ∧ 
  ((p ≤ 4 ∨ p ≥ 12) ∧ (p < 19 ∨ 19 < p)) :=
sorry

end NUMINAMATH_GPT_parameter_conditions_l2061_206131


namespace NUMINAMATH_GPT_train_crossing_time_l2061_206160

-- Define the conditions
def length_of_train : ℕ := 200  -- in meters
def speed_of_train_kmph : ℕ := 90  -- in km per hour
def length_of_tunnel : ℕ := 2500  -- in meters

-- Conversion of speed from kmph to m/s
def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600

-- Define the total distance to be covered (train length + tunnel length)
def total_distance : ℕ := length_of_train + length_of_tunnel

-- Define the expected time to cross the tunnel (in seconds)
def expected_time : ℕ := 108

-- The theorem statement to prove
theorem train_crossing_time : (total_distance / speed_of_train_mps) = expected_time := 
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2061_206160


namespace NUMINAMATH_GPT_smallest_n_inequality_l2061_206140

theorem smallest_n_inequality :
  ∃ n : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ m : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_n_inequality_l2061_206140


namespace NUMINAMATH_GPT_mutually_exclusive_not_opposite_l2061_206137

namespace event_theory

-- Definition to represent the student group
structure Group where
  boys : ℕ
  girls : ℕ

def student_group : Group := {boys := 3, girls := 2}

-- Definition of events
inductive Event
| AtLeastOneBoyAndOneGirl
| ExactlyOneBoyExactlyTwoBoys
| AtLeastOneBoyAllGirls
| AtMostOneBoyAllGirls

open Event

-- Conditions provided in the problem
def condition (grp : Group) : Prop :=
  grp.boys = 3 ∧ grp.girls = 2

-- The main statement to prove in Lean
theorem mutually_exclusive_not_opposite :
  condition student_group →
  ∃ e₁ e₂ : Event, e₁ = ExactlyOneBoyExactlyTwoBoys ∧ e₂ = ExactlyOneBoyExactlyTwoBoys ∧ (
    (e₁ ≠ e₂) ∧ (¬ (e₁ = e₂ ∧ e₁ = ExactlyOneBoyExactlyTwoBoys))
  ) :=
by
  sorry

end event_theory

end NUMINAMATH_GPT_mutually_exclusive_not_opposite_l2061_206137


namespace NUMINAMATH_GPT_geometric_sum_4_terms_l2061_206104

theorem geometric_sum_4_terms 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 2 = 9) 
  (h2 : a 5 = 243) 
  (hq : ∀ n, a (n + 1) = a n * q) 
  : a 1 * (1 - q^4) / (1 - q) = 120 := 
sorry

end NUMINAMATH_GPT_geometric_sum_4_terms_l2061_206104


namespace NUMINAMATH_GPT_total_carrots_l2061_206121

/-- 
  If Pleasant Goat and Beautiful Goat each receive 6 carrots, and the other goats each receive 3 carrots, there will be 6 carrots left over.
  If Pleasant Goat and Beautiful Goat each receive 7 carrots, and the other goats each receive 5 carrots, there will be a shortage of 14 carrots.
  Prove the total number of carrots (n) is 45. 
--/
theorem total_carrots (X n : ℕ) 
  (h1 : n = 3 * X + 18) 
  (h2 : n = 5 * X) : 
  n = 45 := 
by
  sorry

end NUMINAMATH_GPT_total_carrots_l2061_206121


namespace NUMINAMATH_GPT_emily_subtracts_99_from_50_squared_l2061_206164

theorem emily_subtracts_99_from_50_squared :
  (50 - 1) ^ 2 = 50 ^ 2 - 99 := by
  sorry

end NUMINAMATH_GPT_emily_subtracts_99_from_50_squared_l2061_206164


namespace NUMINAMATH_GPT_geometric_sequence_term_l2061_206199

noncomputable def b_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 1 => Real.sin x ^ 2
  | 2 => Real.sin x * Real.cos x
  | 3 => Real.cos x ^ 2 / Real.sin x
  | n + 4 => (Real.cos x / Real.sin x) ^ n * Real.cos x ^ 3 / Real.sin x ^ 2
  | _ => 0 -- Placeholder to cover all case

theorem geometric_sequence_term (x : ℝ) :
  ∃ n, b_n n x = Real.cos x + Real.sin x ∧ n = 7 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_term_l2061_206199


namespace NUMINAMATH_GPT_area_error_percent_l2061_206112

theorem area_error_percent (L W : ℝ) (L_pos : 0 < L) (W_pos : 0 < W) :
  let A := L * W
  let A_measured := (1.05 * L) * (0.96 * W)
  let error_percent := ((A_measured - A) / A) * 100
  error_percent = 0.8 :=
by
  let A := L * W
  let A_measured := (1.05 * L) * (0.96 * W)
  let error := A_measured - A
  let error_percent := (error / A) * 100
  sorry

end NUMINAMATH_GPT_area_error_percent_l2061_206112


namespace NUMINAMATH_GPT_find_constant_x_geom_prog_l2061_206178

theorem find_constant_x_geom_prog (x : ℝ) :
  (30 + x) ^ 2 = (10 + x) * (90 + x) → x = 0 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_constant_x_geom_prog_l2061_206178


namespace NUMINAMATH_GPT_union_of_A_and_B_l2061_206109

open Set

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | -1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > -1} :=
by sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2061_206109


namespace NUMINAMATH_GPT_mary_walking_speed_l2061_206189

-- Definitions based on the conditions:
def distance_sharon (t : ℝ) : ℝ := 6 * t
def distance_mary (x t : ℝ) : ℝ := x * t
def total_distance (x t : ℝ) : ℝ := distance_sharon t + distance_mary x t

-- Lean statement to prove that the speed x is 4 given the conditions
theorem mary_walking_speed (x : ℝ) (t : ℝ) (h1 : t = 0.3) (h2 : total_distance x t = 3) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_mary_walking_speed_l2061_206189


namespace NUMINAMATH_GPT_area_inside_C_but_outside_A_and_B_l2061_206175

def radius_A := 1
def radius_B := 1
def radius_C := 2
def tangency_AB := true
def tangency_AC_non_midpoint := true

theorem area_inside_C_but_outside_A_and_B :
  let areaC := π * (radius_C ^ 2)
  let areaA := π * (radius_A ^ 2)
  let areaB := π * (radius_B ^ 2)
  let overlapping_area := 2 * (π * (radius_A ^ 2) / 2) -- approximation
  areaC - overlapping_area = 3 * π - 2 :=
by
  sorry

end NUMINAMATH_GPT_area_inside_C_but_outside_A_and_B_l2061_206175


namespace NUMINAMATH_GPT_audrey_older_than_heracles_l2061_206117

variable (A H : ℕ)
variable (hH : H = 10)
variable (hFutureAge : A + 3 = 2 * H)

theorem audrey_older_than_heracles : A - H = 7 :=
by
  have h1 : H = 10 := by assumption
  have h2 : A + 3 = 2 * H := by assumption
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_audrey_older_than_heracles_l2061_206117


namespace NUMINAMATH_GPT_units_digit_33_exp_l2061_206144

def units_digit_of_power_cyclic (base exponent : ℕ) (cycle : List ℕ) : ℕ :=
  cycle.get! (exponent % cycle.length)

theorem units_digit_33_exp (n : ℕ) (h1 : 33 = 1 + 4 * 8) (h2 : 44 = 4 * 11) :
  units_digit_of_power_cyclic 33 (33 * 44 ^ 44) [3, 9, 7, 1] = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_33_exp_l2061_206144


namespace NUMINAMATH_GPT_angles_cosine_condition_l2061_206126

theorem angles_cosine_condition {A B : ℝ} (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  (A > B) ↔ (Real.cos A < Real.cos B) :=
by
sorry

end NUMINAMATH_GPT_angles_cosine_condition_l2061_206126


namespace NUMINAMATH_GPT_vector_addition_example_l2061_206168

theorem vector_addition_example :
  let a := (1, 2)
  let b := (-2, 1)
  a.1 + 2 * b.1 = -3 ∧ a.2 + 2 * b.2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_example_l2061_206168


namespace NUMINAMATH_GPT_factorize_expression_l2061_206113

theorem factorize_expression (a b : ℝ) :
  ab^(3 : ℕ) - 4 * ab = ab * (b + 2) * (b - 2) :=
by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_factorize_expression_l2061_206113


namespace NUMINAMATH_GPT_second_number_is_22_l2061_206125

theorem second_number_is_22 
    (A B : ℤ)
    (h1 : A - B = 88) 
    (h2 : A = 110) :
    B = 22 :=
by
  sorry

end NUMINAMATH_GPT_second_number_is_22_l2061_206125


namespace NUMINAMATH_GPT_solve_system_eq_pos_reals_l2061_206147

theorem solve_system_eq_pos_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + y^2 + x * y = 7)
  (h2 : x^2 + z^2 + x * z = 13)
  (h3 : y^2 + z^2 + y * z = 19) :
  x = 1 ∧ y = 2 ∧ z = 3 :=
sorry

end NUMINAMATH_GPT_solve_system_eq_pos_reals_l2061_206147


namespace NUMINAMATH_GPT_alice_acorns_purchase_l2061_206179

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

end NUMINAMATH_GPT_alice_acorns_purchase_l2061_206179


namespace NUMINAMATH_GPT_area_of_triangle_formed_by_lines_l2061_206150

theorem area_of_triangle_formed_by_lines (x y : ℝ) (h1 : y = x) (h2 : x = -5) :
  let base := 5
  let height := 5
  let area := (1 / 2 : ℝ) * base * height
  area = 12.5 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_formed_by_lines_l2061_206150


namespace NUMINAMATH_GPT_coordinates_of_B_l2061_206114

-- Define the initial coordinates of point A
def A : ℝ × ℝ := (1, -2)

-- Define the transformation to get point B from A
def B : ℝ × ℝ := (A.1 - 2, A.2 + 3)

theorem coordinates_of_B : B = (-1, 1) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_B_l2061_206114


namespace NUMINAMATH_GPT_sum_q_p_values_is_neg42_l2061_206166

def p (x : Int) : Int := 2 * Int.natAbs x - 1

def q (x : Int) : Int := -(Int.natAbs x) - 1

def values : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

def q_p_sum : Int :=
  let q_p_values := values.map (λ x => q (p x))
  q_p_values.sum

theorem sum_q_p_values_is_neg42 : q_p_sum = -42 :=
  by
    sorry

end NUMINAMATH_GPT_sum_q_p_values_is_neg42_l2061_206166


namespace NUMINAMATH_GPT_sum_of_star_angles_l2061_206170

theorem sum_of_star_angles :
  let n := 12
  let angle_per_arc := 360 / n
  let arcs_per_tip := 3
  let internal_angle_per_tip := 360 - arcs_per_tip * angle_per_arc
  let sum_of_angles := n * (360 - internal_angle_per_tip)
  sum_of_angles = 1080 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_star_angles_l2061_206170


namespace NUMINAMATH_GPT_arithmetic_sequence_diff_l2061_206141

-- Define the arithmetic sequence properties
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (a : ℕ → ℤ)
variable (h1 : is_arithmetic_sequence a 2)

-- Prove that a_5 - a_2 = 6
theorem arithmetic_sequence_diff : a 5 - a 2 = 6 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_diff_l2061_206141


namespace NUMINAMATH_GPT_monotonic_intervals_value_of_a_inequality_a_minus_one_l2061_206162

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem monotonic_intervals (a : ℝ) :
  (∀ x, 0 < x → 0 ≤ a → 0 < (a * x + 1) / x) ∧
  (∀ x, 0 < x → a < 0 → (0 < x ∧ x < -1/a → 0 < (a * x + 1) / x) ∧
    (-1/a < x → 0 > (a * x + 1) / x)) :=
sorry

theorem value_of_a (a : ℝ) (h_a : a < 0) (h_max : (∀ x, x ∈ Set.Icc 0 e → f a x ≤ -2) ∧ (∃ x, x ∈ Set.Icc 0 e ∧ f a x = -2)) :
  a = -Real.exp 1 := 
sorry

theorem inequality_a_minus_one (a : ℝ) (h_a : a = -1) :
  (∀ x, 0 < x → x * |f a x| > Real.log x + 1/2 * x) :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_value_of_a_inequality_a_minus_one_l2061_206162


namespace NUMINAMATH_GPT_power_comparison_l2061_206152

theorem power_comparison (A B : ℝ) (h1 : A = 1997 ^ (1998 ^ 1999)) (h2 : B = 1999 ^ (1998 ^ 1997)) (h3 : 1997 < 1999) :
  A > B :=
by
  sorry

end NUMINAMATH_GPT_power_comparison_l2061_206152


namespace NUMINAMATH_GPT_num_senior_in_sample_l2061_206186

-- Definitions based on conditions
def total_students : ℕ := 2000
def senior_students : ℕ := 700
def sample_size : ℕ := 400

-- Theorem statement for the number of senior students in the sample
theorem num_senior_in_sample : 
  (senior_students * sample_size) / total_students = 140 :=
by 
  sorry

end NUMINAMATH_GPT_num_senior_in_sample_l2061_206186


namespace NUMINAMATH_GPT_find_time_ball_hits_ground_l2061_206169

theorem find_time_ball_hits_ground :
  ∃ t : ℝ, (-16 * t^2 + 40 * t + 30 = 0) ∧ (t = (5 + 5 * Real.sqrt 22) / 4) := 
by
  sorry

end NUMINAMATH_GPT_find_time_ball_hits_ground_l2061_206169


namespace NUMINAMATH_GPT_total_students_correct_l2061_206105

def students_in_general_hall : ℕ := 30
def students_in_biology_hall : ℕ := 2 * students_in_general_hall
def combined_students_general_biology : ℕ := students_in_general_hall + students_in_biology_hall
def students_in_math_hall : ℕ := (3 * combined_students_general_biology) / 5
def total_students_in_all_halls : ℕ := students_in_general_hall + students_in_biology_hall + students_in_math_hall

theorem total_students_correct : total_students_in_all_halls = 144 := by
  -- Proof omitted, it should be
  sorry

end NUMINAMATH_GPT_total_students_correct_l2061_206105


namespace NUMINAMATH_GPT_sum_not_prime_l2061_206135

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) := 
sorry

end NUMINAMATH_GPT_sum_not_prime_l2061_206135


namespace NUMINAMATH_GPT_simplify_expression_l2061_206161

theorem simplify_expression : (3 + 3 + 5) / 2 - 1 / 2 = 5 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2061_206161


namespace NUMINAMATH_GPT_distance_between_centers_l2061_206116

noncomputable def distance_centers_inc_exc (PQ PR QR: ℝ) (hPQ: PQ = 17) (hPR: PR = 15) (hQR: QR = 8) : ℝ :=
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  let r := area / s
  let r' := area / (s - QR)
  let PU := s - PQ
  let PV := s
  let PI := Real.sqrt ((PU)^2 + (r)^2)
  let PE := Real.sqrt ((PV)^2 + (r')^2)
  PE - PI

theorem distance_between_centers (PQ PR QR : ℝ) (hPQ: PQ = 17) (hPR: PR = 15) (hQR: QR = 8) :
  distance_centers_inc_exc PQ PR QR hPQ hPR hQR = 5 * Real.sqrt 17 - 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_distance_between_centers_l2061_206116


namespace NUMINAMATH_GPT_grid_spiral_infinite_divisible_by_68_grid_spiral_unique_center_sums_l2061_206139

theorem grid_spiral_infinite_divisible_by_68 (n : ℕ) :
  ∃ (k : ℕ), ∃ (m : ℕ), ∃ (t : ℕ), 
  let A := t + 0;
  let B := t + 4;
  let C := t + 12;
  let D := t + 8;
  (k = n * 68 ∧ (n ≥ 1)) ∧ 
  (m = A + B + C + D) ∧ (m % 68 = 0) := by
  sorry

theorem grid_spiral_unique_center_sums (n : ℕ) :
  ∀ (i j : ℕ), 
  let Si := n * 68 + i;
  let Sj := n * 68 + j;
  ¬ (Si = Sj) := by
  sorry

end NUMINAMATH_GPT_grid_spiral_infinite_divisible_by_68_grid_spiral_unique_center_sums_l2061_206139


namespace NUMINAMATH_GPT_magnitude_BC_range_l2061_206159

theorem magnitude_BC_range (AB AC : EuclideanSpace ℝ (Fin 2)) 
  (h₁ : ‖AB‖ = 18) (h₂ : ‖AC‖ = 5) : 
  13 ≤ ‖AC - AB‖ ∧ ‖AC - AB‖ ≤ 23 := 
  sorry

end NUMINAMATH_GPT_magnitude_BC_range_l2061_206159


namespace NUMINAMATH_GPT_farmer_total_land_l2061_206145

noncomputable def total_land_owned_by_farmer (cleared_land_with_tomato : ℝ) (cleared_percentage : ℝ) (grape_percentage : ℝ) (potato_percentage : ℝ) : ℝ :=
  let cleared_land := cleared_percentage
  let total_clearance_with_tomato := cleared_land_with_tomato
  let unused_cleared_percentage := 1 - grape_percentage - potato_percentage
  let total_cleared_land := total_clearance_with_tomato / unused_cleared_percentage
  total_cleared_land / cleared_land

theorem farmer_total_land (cleared_land_with_tomato : ℝ) (cleared_percentage : ℝ) (grape_percentage : ℝ) (potato_percentage : ℝ) :
  (cleared_land_with_tomato = 450) →
  (cleared_percentage = 0.90) →
  (grape_percentage = 0.10) →
  (potato_percentage = 0.80) →
  total_land_owned_by_farmer cleared_land_with_tomato 90 10 80 = 1666.6667 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_farmer_total_land_l2061_206145


namespace NUMINAMATH_GPT_sum_of_roots_l2061_206127

theorem sum_of_roots : 
  (∃ x1 x2 : ℚ, (3 * x1 + 4) * (2 * x1 - 12) = 0 ∧ (3 * x2 + 4) * (2 * x2 - 12) = 0 ∧ x1 ≠ x2 ∧ x1 + x2 = 14 / 3) :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l2061_206127


namespace NUMINAMATH_GPT_inequality_2_inequality_4_l2061_206188

variables (a b : ℝ)
variables (h₁ : 0 < a) (h₂ : 0 < b)

theorem inequality_2 (h₁ : 0 < a) (h₂ : 0 < b) : a > |a - b| - b :=
by
  sorry

theorem inequality_4 (h₁ : 0 < a) (h₂ : 0 < b) : ab + 2 / ab > 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_2_inequality_4_l2061_206188


namespace NUMINAMATH_GPT_exists_integers_a_b_l2061_206107

theorem exists_integers_a_b : 
  ∃ (a b : ℤ), 2003 < a + b * (Real.sqrt 2) ∧ a + b * (Real.sqrt 2) < 2003.01 :=
by
  sorry

end NUMINAMATH_GPT_exists_integers_a_b_l2061_206107


namespace NUMINAMATH_GPT_outlet_pipe_emptying_time_l2061_206136

noncomputable def fill_rate_pipe1 : ℝ := 1 / 18
noncomputable def fill_rate_pipe2 : ℝ := 1 / 30
noncomputable def empty_rate_outlet_pipe (x : ℝ) : ℝ := 1 / x
noncomputable def combined_rate (x : ℝ) : ℝ := fill_rate_pipe1 + fill_rate_pipe2 - empty_rate_outlet_pipe x
noncomputable def total_fill_time : ℝ := 0.06666666666666665

theorem outlet_pipe_emptying_time : ∃ x : ℝ, combined_rate x = 1 / total_fill_time ∧ x = 45 :=
by
  sorry

end NUMINAMATH_GPT_outlet_pipe_emptying_time_l2061_206136


namespace NUMINAMATH_GPT_calc_radical_power_l2061_206123

theorem calc_radical_power : (Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 16))) ^ 12) = 4096 := sorry

end NUMINAMATH_GPT_calc_radical_power_l2061_206123


namespace NUMINAMATH_GPT_fraction_to_decimal_l2061_206100

theorem fraction_to_decimal (numerator : ℚ) (denominator : ℚ) (h : numerator = 5 ∧ denominator = 40) : 
  (numerator / denominator) = 0.125 :=
sorry

end NUMINAMATH_GPT_fraction_to_decimal_l2061_206100


namespace NUMINAMATH_GPT_objective_function_range_l2061_206146

noncomputable def feasible_region (A B C : ℝ × ℝ) := 
  let (x, y) := A
  let (x1, y1) := B 
  let (x2, y2) := C 
  {p : ℝ × ℝ | True} -- The exact feasible region description is not specified

theorem objective_function_range
  (A B C: ℝ × ℝ)
  (a b : ℝ)
  (x y : ℝ)
  (hA : A = (x, y))
  (hB : B = (1, 1))
  (hC : C = (5, 2))
  (h1 : a + b = 3)
  (h2 : 5 * a + 2 * b = 12) :
  let z := a * x + b * y
  3 ≤ z ∧ z ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_objective_function_range_l2061_206146


namespace NUMINAMATH_GPT_table_coverage_percentage_l2061_206133

def A := 204  -- Total area of the runners
def T := 175  -- Area of the table
def A2 := 24  -- Area covered by exactly two layers of runner
def A3 := 20  -- Area covered by exactly three layers of runner

theorem table_coverage_percentage : 
  (A - 2 * A2 - 3 * A3 + A2 + A3) / T * 100 = 80 := 
by
  sorry

end NUMINAMATH_GPT_table_coverage_percentage_l2061_206133


namespace NUMINAMATH_GPT_more_ones_than_twos_in_digital_roots_l2061_206176

/-- Define the digital root (i.e., repeated sum of digits until a single digit). -/
def digitalRoot (n : Nat) : Nat :=
  if n == 0 then 0 else 1 + (n - 1) % 9

/-- Statement of the problem: For numbers 1 to 1,000,000, the count of digital root 1 is higher than the count of digital root 2. -/
theorem more_ones_than_twos_in_digital_roots :
  (Finset.filter (fun n => digitalRoot n = 1) (Finset.range 1000000)).card >
  (Finset.filter (fun n => digitalRoot n = 2) (Finset.range 1000000)).card :=
by
  sorry

end NUMINAMATH_GPT_more_ones_than_twos_in_digital_roots_l2061_206176


namespace NUMINAMATH_GPT_discarded_number_l2061_206153

theorem discarded_number (S x : ℕ) (h1 : S / 50 = 50) (h2 : (S - x - 55) / 48 = 50) : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_discarded_number_l2061_206153


namespace NUMINAMATH_GPT_find_b_l2061_206190

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (317212435 * 101 - b) % 25 = 0 ∧ b = 13 := by
  sorry

end NUMINAMATH_GPT_find_b_l2061_206190


namespace NUMINAMATH_GPT_expected_number_of_digits_is_1_55_l2061_206194

def probability_one_digit : ℚ := 9 / 20
def probability_two_digits : ℚ := 1 / 2
def probability_twenty : ℚ := 1 / 20
def expected_digits : ℚ := (1 * probability_one_digit) + (2 * probability_two_digits) + (2 * probability_twenty)

theorem expected_number_of_digits_is_1_55 :
  expected_digits = 1.55 :=
sorry

end NUMINAMATH_GPT_expected_number_of_digits_is_1_55_l2061_206194


namespace NUMINAMATH_GPT_Seokjin_paper_count_l2061_206106

theorem Seokjin_paper_count (Jimin_paper : ℕ) (h1 : Jimin_paper = 41) (h2 : ∀ x : ℕ, Seokjin_paper = Jimin_paper - 1) : Seokjin_paper = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_Seokjin_paper_count_l2061_206106


namespace NUMINAMATH_GPT_gauss_algorithm_sum_l2061_206184

def f (x : Nat) (m : Nat) : Rat := x / (3 * m + 6054)

theorem gauss_algorithm_sum (m : Nat) :
  (Finset.sum (Finset.range (m + 2017 + 1)) (λ x => f x m)) = (m + 2017) / 6 := by
sorry

end NUMINAMATH_GPT_gauss_algorithm_sum_l2061_206184


namespace NUMINAMATH_GPT_area_of_frame_l2061_206142

def width : ℚ := 81 / 4
def depth : ℚ := 148 / 9
def area (w d : ℚ) : ℚ := w * d

theorem area_of_frame : area width depth = 333 := by
  sorry

end NUMINAMATH_GPT_area_of_frame_l2061_206142


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_4_l2061_206110

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^5 - 8 * x^4 + 15 * x^3 + 20 * x^2 - 5 * x - 20

-- State the problem as a theorem
theorem remainder_when_divided_by_x_minus_4 : 
    (f 4 = 216) := 
by 
    -- Calculation goes here
    sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_4_l2061_206110


namespace NUMINAMATH_GPT_pipe_B_fills_6_times_faster_l2061_206128

theorem pipe_B_fills_6_times_faster :
  let R_A := 1 / 32
  let combined_rate := 7 / 32
  let R_B := combined_rate - R_A
  (R_B / R_A = 6) :=
by
  let R_A := 1 / 32
  let combined_rate := 7 / 32
  let R_B := combined_rate - R_A
  sorry

end NUMINAMATH_GPT_pipe_B_fills_6_times_faster_l2061_206128


namespace NUMINAMATH_GPT_find_integer_value_of_a_l2061_206130

-- Define the conditions for the equation and roots
def equation_has_two_distinct_negative_integer_roots (a : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, x1 ≠ x2 ∧ x1 < 0 ∧ x2 < 0 ∧ (a^2 - 1) * x1^2 - 2 * (5 * a + 1) * x1 + 24 = 0 ∧ (a^2 - 1) * x2^2 - 2 * (5 * a + 1) * x2 + 24 = 0 ∧
  x1 = 6 / (a - 1) ∧ x2 = 4 / (a + 1)

-- Prove that the only integer value of a that satisfies these conditions is -2
theorem find_integer_value_of_a : 
  ∃ (a : ℤ), equation_has_two_distinct_negative_integer_roots a ∧ a = -2 := 
sorry

end NUMINAMATH_GPT_find_integer_value_of_a_l2061_206130


namespace NUMINAMATH_GPT_horizontal_asymptote_value_l2061_206122

theorem horizontal_asymptote_value :
  (∃ y : ℝ, ∀ x : ℝ, (y = (18 * x^5 + 6 * x^3 + 3 * x^2 + 5 * x + 4) / (6 * x^5 + 4 * x^3 + 5 * x^2 + 2 * x + 1)) → y = 3) :=
by
  sorry

end NUMINAMATH_GPT_horizontal_asymptote_value_l2061_206122


namespace NUMINAMATH_GPT_max_legs_lengths_l2061_206124

theorem max_legs_lengths (a x y : ℝ) (h₁ : x^2 + y^2 = a^2) (h₂ : 3 * x + 4 * y ≤ 5 * a) :
  3 * x + 4 * y = 5 * a → x = (3 * a / 5) ∧ y = (4 * a / 5) :=
by
  sorry

end NUMINAMATH_GPT_max_legs_lengths_l2061_206124


namespace NUMINAMATH_GPT_triangle_area_inscribed_in_circle_l2061_206196

theorem triangle_area_inscribed_in_circle (R : ℝ) 
    (h_pos : R > 0) 
    (h_ratio : ∃ (x : ℝ)(hx : x > 0), 2*x + 5*x + 17*x = 2*π) :
  (∃ (area : ℝ), area = (R^2 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_inscribed_in_circle_l2061_206196


namespace NUMINAMATH_GPT_polynomial_degree_le_one_l2061_206180

theorem polynomial_degree_le_one {P : ℝ → ℝ} (h : ∀ x : ℝ, 2 * P x = P (x + 3) + P (x - 3)) :
  ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x + b :=
sorry

end NUMINAMATH_GPT_polynomial_degree_le_one_l2061_206180


namespace NUMINAMATH_GPT_evaluate_expression_l2061_206118

theorem evaluate_expression : 
  ( (2^12)^2 - (2^10)^2 ) / ( (2^11)^2 - (2^9)^2 ) = 4 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2061_206118


namespace NUMINAMATH_GPT_complex_division_l2061_206158

theorem complex_division (i : ℂ) (h : i^2 = -1) : (2 + i) / (1 - 2 * i) = i := 
by
  sorry

end NUMINAMATH_GPT_complex_division_l2061_206158


namespace NUMINAMATH_GPT_max_marks_l2061_206119

-- Define the conditions
def passing_marks (M : ℕ) : ℕ := 40 * M / 100

def Ravish_got_marks : ℕ := 40
def marks_failed_by : ℕ := 40

-- Lean statement to prove
theorem max_marks (M : ℕ) (h : passing_marks M = Ravish_got_marks + marks_failed_by) : M = 200 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l2061_206119


namespace NUMINAMATH_GPT_geometric_sequence_product_l2061_206154

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_roots : (∃ a₁ a₁₉ : ℝ, (a₁ + a₁₉ = 10) ∧ (a₁ * a₁₉ = 16) ∧ a 1 = a₁ ∧ a 19 = a₁₉)) :
  a 8 * a 12 = 16 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l2061_206154


namespace NUMINAMATH_GPT_abs_neg_ten_l2061_206151

theorem abs_neg_ten : abs (-10) = 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_abs_neg_ten_l2061_206151


namespace NUMINAMATH_GPT_value_of_m_l2061_206134

theorem value_of_m (x m : ℝ) (h : x ≠ 3) (H : (x / (x - 3) = 2 - m / (3 - x))) : m = 3 :=
sorry

end NUMINAMATH_GPT_value_of_m_l2061_206134
