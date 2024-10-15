import Mathlib

namespace NUMINAMATH_GPT_original_number_is_144_l2192_219253

theorem original_number_is_144 :
  ∃ (A B C : ℕ), A ≠ 0 ∧
  (100 * A + 11 * B = 144) ∧
  (A * B^2 = 10 * A + C) ∧
  (A * C = C) ∧
  A = 1 ∧ B = 4 ∧ C = 6 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_144_l2192_219253


namespace NUMINAMATH_GPT_gcd_polynomial_multiple_l2192_219259

theorem gcd_polynomial_multiple (b : ℕ) (hb : 620 ∣ b) : gcd (4 * b^3 + 2 * b^2 + 5 * b + 93) b = 93 := by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_multiple_l2192_219259


namespace NUMINAMATH_GPT_b_remainder_l2192_219237

theorem b_remainder (n : ℕ) (hn : n > 0) : ∃ b : ℕ, b % 11 = 5 :=
by
  sorry

end NUMINAMATH_GPT_b_remainder_l2192_219237


namespace NUMINAMATH_GPT_new_girl_weight_l2192_219240

theorem new_girl_weight (W : ℝ) (h : (W + 24) / 8 = W / 8 + 3) :
  (W + 24) - (W - 70) = 94 :=
by
  sorry

end NUMINAMATH_GPT_new_girl_weight_l2192_219240


namespace NUMINAMATH_GPT_find_value_of_expression_l2192_219202

variable {a : ℕ → ℤ}

-- Define arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (h1 : a 1 + 3 * a 8 + a 15 = 120)
variable (h2 : is_arithmetic_sequence a)

-- Theorem to be proved
theorem find_value_of_expression : 2 * a 6 - a 4 = 24 :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l2192_219202


namespace NUMINAMATH_GPT_min_m_plus_n_l2192_219290

theorem min_m_plus_n (m n : ℕ) (h₁ : m > n) (h₂ : 4^m + 4^n % 100 = 0) : m + n = 7 :=
sorry

end NUMINAMATH_GPT_min_m_plus_n_l2192_219290


namespace NUMINAMATH_GPT_sequence_term_expression_l2192_219266

theorem sequence_term_expression (a : ℕ → ℝ) (S : ℕ → ℝ) (C : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, S n + n * a n = C)
  (h3 : ∀ n ≥ 2, (n + 1) * a n = (n - 1) * a (n - 1)) :
  ∀ n, a n = 2 / (n * (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_expression_l2192_219266


namespace NUMINAMATH_GPT_ratio_of_dancers_l2192_219277

theorem ratio_of_dancers (total_kids total_dancers slow_dance non_slow_dance : ℕ)
  (h1 : total_kids = 140)
  (h2 : slow_dance = 25)
  (h3 : non_slow_dance = 10)
  (h4 : total_dancers = slow_dance + non_slow_dance) :
  (total_dancers : ℚ) / total_kids = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_dancers_l2192_219277


namespace NUMINAMATH_GPT_bretschneider_l2192_219201

noncomputable def bretschneider_theorem 
  (a b c d m n : ℝ) 
  (A C : ℝ) : Prop :=
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C)

theorem bretschneider (a b c d m n A C : ℝ) :
  bretschneider_theorem a b c d m n A C :=
sorry

end NUMINAMATH_GPT_bretschneider_l2192_219201


namespace NUMINAMATH_GPT_number_of_students_run_red_light_l2192_219239

theorem number_of_students_run_red_light :
  let total_students := 300
  let yes_responses := 90
  let odd_id_students := 75
  let coin_probability := 1/2
  -- Calculate using the conditions:
  total_students / 2 - odd_id_students / 2 * coin_probability + total_students / 2 * coin_probability = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_run_red_light_l2192_219239


namespace NUMINAMATH_GPT_profit_percentage_is_33_point_33_l2192_219212

variable (C S : ℝ)

-- Initial condition based on the problem statement
axiom cost_eq_sell : 20 * C = 15 * S

-- Statement to prove
theorem profit_percentage_is_33_point_33 (h : 20 * C = 15 * S) : (S - C) / C * 100 = 33.33 := 
sorry

end NUMINAMATH_GPT_profit_percentage_is_33_point_33_l2192_219212


namespace NUMINAMATH_GPT_city_population_l2192_219273

theorem city_population (P: ℝ) (h: 0.85 * P = 85000) : P = 100000 := 
by
  sorry

end NUMINAMATH_GPT_city_population_l2192_219273


namespace NUMINAMATH_GPT_determine_h_l2192_219292

theorem determine_h (x : ℝ) : 
  ∃ h : ℝ → ℝ, (4*x^4 + 11*x^3 + h x = 10*x^3 - x^2 + 4*x - 7) ↔ (h x = -4*x^4 - x^3 - x^2 + 4*x - 7) :=
by
  sorry

end NUMINAMATH_GPT_determine_h_l2192_219292


namespace NUMINAMATH_GPT_problem_statement_l2192_219248

def horse_lap_times : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Least common multiple of a set of numbers
noncomputable def LCM_set (s : List ℕ) : ℕ :=
s.foldl LCM 1

-- Calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem problem_statement :
  let T := LCM_set [2, 3, 5, 7, 11, 13]
  sum_of_digits T = 6 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2192_219248


namespace NUMINAMATH_GPT_acute_angle_slope_neg_product_l2192_219263

   theorem acute_angle_slope_neg_product (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) (acute_inclination : ∃ (k : ℝ), k > 0 ∧ y = -a/b): (a * b < 0) :=
   by
     sorry
   
end NUMINAMATH_GPT_acute_angle_slope_neg_product_l2192_219263


namespace NUMINAMATH_GPT_uncovered_side_length_l2192_219218

theorem uncovered_side_length
  (A : ℝ) (F : ℝ)
  (h1 : A = 600)
  (h2 : F = 130) :
  ∃ L : ℝ, L = 120 :=
by {
  sorry
}

end NUMINAMATH_GPT_uncovered_side_length_l2192_219218


namespace NUMINAMATH_GPT_tommy_initial_balloons_l2192_219205

theorem tommy_initial_balloons :
  ∃ x : ℝ, x + 78.5 = 132.25 ∧ x = 53.75 := by
  sorry

end NUMINAMATH_GPT_tommy_initial_balloons_l2192_219205


namespace NUMINAMATH_GPT_train_speed_l2192_219208

noncomputable def speed_in_kmh (distance : ℕ) (time : ℕ) : ℚ :=
  (distance : ℚ) / (time : ℚ) * 3600 / 1000

theorem train_speed
  (distance : ℕ) (time : ℕ)
  (h_dist : distance = 150)
  (h_time : time = 9) :
  speed_in_kmh distance time = 60 :=
by
  rw [h_dist, h_time]
  sorry

end NUMINAMATH_GPT_train_speed_l2192_219208


namespace NUMINAMATH_GPT_volume_of_given_solid_l2192_219261

noncomputable def volume_of_solid (s : ℝ) (h : ℝ) : ℝ :=
  (h / 3) * (s^2 + (s * (3 / 2))^2 + (s * (3 / 2)) * s)

theorem volume_of_given_solid : volume_of_solid 8 10 = 3040 / 3 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_given_solid_l2192_219261


namespace NUMINAMATH_GPT_total_albums_l2192_219200

-- Defining the initial conditions
def albumsAdele : ℕ := 30
def albumsBridget : ℕ := albumsAdele - 15
def albumsKatrina : ℕ := 6 * albumsBridget
def albumsMiriam : ℕ := 7 * albumsKatrina
def albumsCarlos : ℕ := 3 * albumsMiriam
def albumsDiane : ℕ := 2 * albumsKatrina

-- Proving the total number of albums
theorem total_albums :
  albumsAdele + albumsBridget + albumsKatrina + albumsMiriam + albumsCarlos + albumsDiane = 2835 :=
by
  sorry

end NUMINAMATH_GPT_total_albums_l2192_219200


namespace NUMINAMATH_GPT_son_age_l2192_219296

theorem son_age {M S : ℕ} 
  (h1 : M = S + 18) 
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 16 := 
by
  sorry

end NUMINAMATH_GPT_son_age_l2192_219296


namespace NUMINAMATH_GPT_find_b_l2192_219235

noncomputable def geom_seq_term (a b c : ℝ) : Prop :=
∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r

theorem find_b (b : ℝ) (h_geom : geom_seq_term 160 b (108 / 64)) (h_pos : b > 0) :
  b = 15 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l2192_219235


namespace NUMINAMATH_GPT_op_proof_l2192_219204

-- Definition of the operation \(\oplus\)
def op (x y : ℝ) : ℝ := x^2 + y

-- Theorem statement for the given proof problem
theorem op_proof (h : ℝ) : op h (op h h) = 2 * h^2 + h :=
by 
  sorry

end NUMINAMATH_GPT_op_proof_l2192_219204


namespace NUMINAMATH_GPT_sharks_at_newport_l2192_219255

theorem sharks_at_newport :
  ∃ (x : ℕ), (∃ (y : ℕ), y = 4 * x ∧ x + y = 110) ∧ x = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_sharks_at_newport_l2192_219255


namespace NUMINAMATH_GPT_johns_climb_height_correct_l2192_219275

noncomputable def johns_total_height : ℝ :=
  let stair1_height := 4 * 15
  let stair2_height := 5 * 12.5
  let total_stair_height := stair1_height + stair2_height
  let rope1_height := (2 / 3) * stair1_height
  let rope2_height := (3 / 5) * stair2_height
  let total_rope_height := rope1_height + rope2_height
  let rope1_height_m := rope1_height / 3.281
  let rope2_height_m := rope2_height / 3.281
  let total_rope_height_m := rope1_height_m + rope2_height_m
  let ladder_height := 1.5 * total_rope_height_m * 3.281
  let rock_wall_height := (2 / 3) * ladder_height
  let total_pre_tree := total_stair_height + total_rope_height + ladder_height + rock_wall_height
  let tree_height := (3 / 4) * total_pre_tree - 10
  total_stair_height + total_rope_height + ladder_height + rock_wall_height + tree_height

theorem johns_climb_height_correct : johns_total_height = 679.115 := by
  sorry

end NUMINAMATH_GPT_johns_climb_height_correct_l2192_219275


namespace NUMINAMATH_GPT_A_superset_C_l2192_219262

-- Definitions of the sets as given in the problem statement
def U : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {-1, 3}
def C : Set ℝ := {x | -1 < x ∧ x < 3}

-- Statement to be proved: A ⊇ C
theorem A_superset_C : A ⊇ C :=
by sorry

end NUMINAMATH_GPT_A_superset_C_l2192_219262


namespace NUMINAMATH_GPT_circle_radius_l2192_219286

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end NUMINAMATH_GPT_circle_radius_l2192_219286


namespace NUMINAMATH_GPT_smallest_possible_gcd_l2192_219268

noncomputable def smallestGCD (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 9) : ℕ :=
  Nat.gcd (12 * a) (18 * b)

theorem smallest_possible_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 9) : 
  smallestGCD a b h1 h2 h3 = 54 :=
sorry

end NUMINAMATH_GPT_smallest_possible_gcd_l2192_219268


namespace NUMINAMATH_GPT_not_divisible_by_81_l2192_219288

theorem not_divisible_by_81 (n : ℤ) : ¬ (81 ∣ n^3 - 9 * n + 27) :=
sorry

end NUMINAMATH_GPT_not_divisible_by_81_l2192_219288


namespace NUMINAMATH_GPT_find_n_l2192_219245

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 8): n = 24 := by
  sorry

end NUMINAMATH_GPT_find_n_l2192_219245


namespace NUMINAMATH_GPT_natural_solutions_3x_4y_eq_12_l2192_219234

theorem natural_solutions_3x_4y_eq_12 :
  ∃ x y : ℕ, (3 * x + 4 * y = 12) ∧ ((x = 4 ∧ y = 0) ∨ (x = 0 ∧ y = 3)) := 
sorry

end NUMINAMATH_GPT_natural_solutions_3x_4y_eq_12_l2192_219234


namespace NUMINAMATH_GPT_cerulean_survey_l2192_219241

theorem cerulean_survey :
  let total_people := 120
  let kind_of_blue := 80
  let kind_and_green := 35
  let neither := 20
  total_people = kind_of_blue + (total_people - kind_of_blue - neither)
  → (kind_and_green + (total_people - kind_of_blue - kind_and_green - neither) + neither) = total_people
  → 55 = (kind_and_green + (total_people - kind_of_blue - kind_and_green - neither)) :=
by
  sorry

end NUMINAMATH_GPT_cerulean_survey_l2192_219241


namespace NUMINAMATH_GPT_alpha_more_economical_l2192_219276

theorem alpha_more_economical (n : ℕ) : n ≥ 12 → 80 + 12 * n < 10 + 18 * n := 
by
  sorry

end NUMINAMATH_GPT_alpha_more_economical_l2192_219276


namespace NUMINAMATH_GPT_student_average_always_greater_l2192_219271

theorem student_average_always_greater (x y z : ℝ) (h1 : x < z) (h2 : z < y) :
  (B = (x + z + 2 * y) / 4) > (A = (x + y + z) / 3) := by
  sorry

end NUMINAMATH_GPT_student_average_always_greater_l2192_219271


namespace NUMINAMATH_GPT_number_of_hens_l2192_219215

-- Let H be the number of hens and C be the number of cows
def hens_and_cows (H C : Nat) : Prop :=
  H + C = 50 ∧ 2 * H + 4 * C = 144

theorem number_of_hens : ∃ H C : Nat, hens_and_cows H C ∧ H = 28 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_number_of_hens_l2192_219215


namespace NUMINAMATH_GPT_sequence_is_arithmetic_l2192_219256

-- Define a_n as a sequence in terms of n, where the formula is given.
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Theorem stating that the sequence is arithmetic with a common difference of 2.
theorem sequence_is_arithmetic : ∀ (n : ℕ), n > 0 → (a_n n) - (a_n (n - 1)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_is_arithmetic_l2192_219256


namespace NUMINAMATH_GPT_five_letter_words_with_at_least_one_vowel_l2192_219278

theorem five_letter_words_with_at_least_one_vowel :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F']
  let vowels := ['A', 'E', 'F']
  (6 ^ 5) - (3 ^ 5) = 7533 := by 
  sorry

end NUMINAMATH_GPT_five_letter_words_with_at_least_one_vowel_l2192_219278


namespace NUMINAMATH_GPT_total_books_correct_l2192_219209

-- Definitions based on the conditions
def num_books_bottom_shelf (T : ℕ) := T / 3
def num_books_middle_shelf (T : ℕ) := T / 4
def num_books_top_shelf : ℕ := 30
def total_books (T : ℕ) := num_books_bottom_shelf T + num_books_middle_shelf T + num_books_top_shelf

theorem total_books_correct : total_books 72 = 72 :=
by
  sorry

end NUMINAMATH_GPT_total_books_correct_l2192_219209


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2192_219281

theorem simplify_and_evaluate (x : ℝ) (h : x = 2 + Real.sqrt 2) :
  (1 - 3 / (x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = Real.sqrt 2 / 2 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2192_219281


namespace NUMINAMATH_GPT_crow_eating_time_l2192_219264

/-- 
We are given that a crow eats a fifth of the total number of nuts in 6 hours.
We are to prove that it will take the crow 7.5 hours to finish a quarter of the nuts.
-/
theorem crow_eating_time (h : (1/5:ℚ) * t = 6) : (1/4) * t = 7.5 := 
by 
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_crow_eating_time_l2192_219264


namespace NUMINAMATH_GPT_points_per_touchdown_l2192_219274

theorem points_per_touchdown (number_of_touchdowns : ℕ) (total_points : ℕ) (h1 : number_of_touchdowns = 3) (h2 : total_points = 21) : (total_points / number_of_touchdowns) = 7 :=
by
  sorry

end NUMINAMATH_GPT_points_per_touchdown_l2192_219274


namespace NUMINAMATH_GPT_math_problem_l2192_219217

theorem math_problem
  (a b c d : ℚ)
  (h₁ : a = 1 / 3)
  (h₂ : b = 1 / 6)
  (h₃ : c = 1 / 9)
  (h₄ : d = 1 / 18) :
  9 * (a + b + c + d)⁻¹ = 27 / 2 := 
sorry

end NUMINAMATH_GPT_math_problem_l2192_219217


namespace NUMINAMATH_GPT_equation_solution_l2192_219270

theorem equation_solution (x : ℤ) (h : x + 1 = 2) : x = 1 :=
sorry

end NUMINAMATH_GPT_equation_solution_l2192_219270


namespace NUMINAMATH_GPT_sum_of_coefficients_eq_l2192_219211

theorem sum_of_coefficients_eq :
  ∃ n : ℕ, (∀ a b : ℕ, (3 * a + 5 * b)^n = 2^15) → n = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_eq_l2192_219211


namespace NUMINAMATH_GPT_parabola_directrix_l2192_219242

theorem parabola_directrix (a : ℝ) (h1 : ∀ x : ℝ, - (1 / (4 * a)) = 2):
  a = -(1 / 8) :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l2192_219242


namespace NUMINAMATH_GPT_value_of_expression_l2192_219227

theorem value_of_expression (x : ℤ) (h : x ^ 2 = 2209) : (x + 2) * (x - 2) = 2205 := 
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_value_of_expression_l2192_219227


namespace NUMINAMATH_GPT_balloons_remaining_l2192_219221
-- Importing the necessary libraries

-- Defining the conditions
def originalBalloons : Nat := 709
def givenBalloons : Nat := 221

-- Stating the theorem
theorem balloons_remaining : originalBalloons - givenBalloons = 488 := by
  sorry

end NUMINAMATH_GPT_balloons_remaining_l2192_219221


namespace NUMINAMATH_GPT_find_x_l2192_219293

theorem find_x 
  (x : ℝ) 
  (angle_PQS angle_QSR angle_SRQ : ℝ) 
  (h1 : angle_PQS = 2 * x)
  (h2 : angle_QSR = 50)
  (h3 : angle_SRQ = x) :
  x = 50 :=
sorry

end NUMINAMATH_GPT_find_x_l2192_219293


namespace NUMINAMATH_GPT_sixDigitIntegersCount_l2192_219236

-- Define the digits to use.
def digits : List ℕ := [1, 2, 2, 5, 9, 9]

-- Define the factorial function as it might not be pre-defined in Mathlib.
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Calculate the number of unique permutations accounting for repeated digits.
def numberOfUniquePermutations : ℕ :=
  factorial 6 / (factorial 2 * factorial 2)

-- State the theorem proving that we can form exactly 180 unique six-digit integers.
theorem sixDigitIntegersCount : numberOfUniquePermutations = 180 :=
  sorry

end NUMINAMATH_GPT_sixDigitIntegersCount_l2192_219236


namespace NUMINAMATH_GPT_negative_integers_abs_le_4_l2192_219222

theorem negative_integers_abs_le_4 :
  ∀ x : ℤ, x < 0 ∧ |x| ≤ 4 ↔ (x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4) :=
by
  sorry

end NUMINAMATH_GPT_negative_integers_abs_le_4_l2192_219222


namespace NUMINAMATH_GPT_find_denominator_l2192_219258

noncomputable def original_denominator (d : ℝ) : Prop :=
  (7 / (d + 3)) = 2 / 3

theorem find_denominator : ∃ d : ℝ, original_denominator d ∧ d = 7.5 :=
by
  use 7.5
  unfold original_denominator
  sorry

end NUMINAMATH_GPT_find_denominator_l2192_219258


namespace NUMINAMATH_GPT_midpoint_chord_hyperbola_l2192_219238

-- Definitions to use in our statement
variables (a b x y : ℝ)
def ellipse : Prop := (x^2)/(a^2) + (y^2)/(b^2) = 1
def line_ellipse : Prop := x / (a^2) + y / (b^2) = 0
def hyperbola : Prop := (x^2)/(a^2) - (y^2)/(b^2) = 1
def line_hyperbola : Prop := x / (a^2) - y / (b^2) = 0

-- The theorem to prove
theorem midpoint_chord_hyperbola (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (x y : ℝ) 
    (h_ellipse : ellipse a b x y)
    (h_line_ellipse : line_ellipse a b x y)
    (h_hyperbola : hyperbola a b x y) :
    line_hyperbola a b x y :=
sorry

end NUMINAMATH_GPT_midpoint_chord_hyperbola_l2192_219238


namespace NUMINAMATH_GPT_hardware_contract_probability_l2192_219280

noncomputable def P_S' : ℚ := 3 / 5
noncomputable def P_at_least_one : ℚ := 5 / 6
noncomputable def P_H_and_S : ℚ := 0.31666666666666654 -- 19 / 60 in fraction form
noncomputable def P_S : ℚ := 1 - P_S'

theorem hardware_contract_probability :
  (P_at_least_one = P_H + P_S - P_H_and_S) →
  P_H = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_hardware_contract_probability_l2192_219280


namespace NUMINAMATH_GPT_area_of_rhombus_l2192_219284

theorem area_of_rhombus (x y : ℝ) (d1 d2 : ℝ) (hx : x^2 + y^2 = 130) (hy : d1 = 2 * x) (hz : d2 = 2 * y) (h_diff : abs (d1 - d2) = 4) : 
  4 * 0.5 * x * y = 126 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l2192_219284


namespace NUMINAMATH_GPT_find_diameters_l2192_219272

theorem find_diameters (x y z : ℕ) (hx : x ≠ y) (hy : y ≠ z) (hz : x ≠ z) :
  x + y + z = 26 ∧ x^2 + y^2 + z^2 = 338 :=
  sorry

end NUMINAMATH_GPT_find_diameters_l2192_219272


namespace NUMINAMATH_GPT_pool_balls_pyramid_arrangement_l2192_219224

/-- In how many distinguishable ways can 10 distinct pool balls be arranged in a pyramid
    (6 on the bottom, 3 in the middle, 1 on the top), assuming that all rotations of the pyramid are indistinguishable? -/
def pyramid_pool_balls_distinguishable_arrangements : Nat :=
  let total_arrangements := Nat.factorial 10
  let indistinguishable_rotations := 9
  total_arrangements / indistinguishable_rotations

theorem pool_balls_pyramid_arrangement :
  pyramid_pool_balls_distinguishable_arrangements = 403200 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_pool_balls_pyramid_arrangement_l2192_219224


namespace NUMINAMATH_GPT_john_buys_1000_balloons_l2192_219257

-- Define conditions
def balloon_volume : ℕ := 10
def tank_volume : ℕ := 500
def num_tanks : ℕ := 20

-- Define the total volume of gas
def total_gas_volume : ℕ := num_tanks * tank_volume

-- Define the number of balloons
def num_balloons : ℕ := total_gas_volume / balloon_volume

-- Prove that the number of balloons is 1,000
theorem john_buys_1000_balloons : num_balloons = 1000 := by
  sorry

end NUMINAMATH_GPT_john_buys_1000_balloons_l2192_219257


namespace NUMINAMATH_GPT_remainder_five_n_minus_eleven_l2192_219287

theorem remainder_five_n_minus_eleven (n : ℤ) (h : n % 7 = 3) : (5 * n - 11) % 7 = 4 := 
    sorry

end NUMINAMATH_GPT_remainder_five_n_minus_eleven_l2192_219287


namespace NUMINAMATH_GPT_find_number_l2192_219226

theorem find_number (x q : ℕ) (h1 : x = 7 * q) (h2 : q + x + 7 = 175) : x = 147 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l2192_219226


namespace NUMINAMATH_GPT_inequality_proof_l2192_219297

theorem inequality_proof (a b : ℝ) (h_a : a > 0) (h_b : 3 + b = a) : 
  3 / b + 1 / a >= 3 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2192_219297


namespace NUMINAMATH_GPT_solution_set_l2192_219279

theorem solution_set (x : ℝ) : (3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9) ↔ (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) :=
sorry

end NUMINAMATH_GPT_solution_set_l2192_219279


namespace NUMINAMATH_GPT_terminal_side_of_minus_330_in_first_quadrant_l2192_219203

def angle_quadrant (angle : ℤ) : ℕ :=
  let reduced_angle := ((angle % 360) + 360) % 360
  if reduced_angle < 90 then 1
  else if reduced_angle < 180 then 2
  else if reduced_angle < 270 then 3
  else 4

theorem terminal_side_of_minus_330_in_first_quadrant :
  angle_quadrant (-330) = 1 :=
by
  -- We need a proof to justify the theorem, so we leave it with 'sorry' as instructed.
  sorry

end NUMINAMATH_GPT_terminal_side_of_minus_330_in_first_quadrant_l2192_219203


namespace NUMINAMATH_GPT_kath_total_cost_l2192_219214

def admission_cost : ℝ := 8
def discount_percentage_pre6pm : ℝ := 0.25
def discount_percentage_student : ℝ := 0.10
def time_of_movie : ℝ := 4
def num_people : ℕ := 6
def num_students : ℕ := 2

theorem kath_total_cost :
  let discounted_price := admission_cost * (1 - discount_percentage_pre6pm)
  let student_price := discounted_price * (1 - discount_percentage_student)
  let num_non_students := num_people - num_students - 1 -- remaining people (total - 2 students - Kath)
  let kath_and_siblings_cost := 3 * discounted_price
  let student_friends_cost := num_students * student_price
  let non_student_friend_cost := num_non_students * discounted_price
  let total_cost := kath_and_siblings_cost + student_friends_cost + non_student_friend_cost
  total_cost = 34.80 := by
  let discounted_price := admission_cost * (1 - discount_percentage_pre6pm)
  let student_price := discounted_price * (1 - discount_percentage_student)
  let num_non_students := num_people - num_students - 1
  let kath_and_siblings_cost := 3 * discounted_price
  let student_friends_cost := num_students * student_price
  let non_student_friend_cost := num_non_students * discounted_price
  let total_cost := kath_and_siblings_cost + student_friends_cost + non_student_friend_cost
  sorry

end NUMINAMATH_GPT_kath_total_cost_l2192_219214


namespace NUMINAMATH_GPT_min_value_fraction_sum_l2192_219265

theorem min_value_fraction_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (∃ x : ℝ, x = (1 / a + 4 / b) ∧ x = 9 / 4) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_sum_l2192_219265


namespace NUMINAMATH_GPT_ones_digit_largest_power_of_three_divides_factorial_3_pow_3_l2192_219294

theorem ones_digit_largest_power_of_three_divides_factorial_3_pow_3 :
  (3 ^ 13) % 10 = 3 := by
  sorry

end NUMINAMATH_GPT_ones_digit_largest_power_of_three_divides_factorial_3_pow_3_l2192_219294


namespace NUMINAMATH_GPT_john_pays_total_l2192_219289

-- Definitions based on conditions
def total_cans : ℕ := 30
def price_per_can : ℝ := 0.60

-- Main statement to be proven
theorem john_pays_total : (total_cans / 2) * price_per_can = 9 := 
by
  sorry

end NUMINAMATH_GPT_john_pays_total_l2192_219289


namespace NUMINAMATH_GPT_victory_points_value_l2192_219254

theorem victory_points_value (V : ℕ) (H : ∀ (v d t : ℕ), 
    v + d + t = 20 ∧ v * V + d ≥ 40 ∧ v ≥ 6 ∧ (t = 20 - 5)) : 
    V = 3 := 
sorry

end NUMINAMATH_GPT_victory_points_value_l2192_219254


namespace NUMINAMATH_GPT_students_exceed_pets_by_70_l2192_219269

theorem students_exceed_pets_by_70 :
  let n_classrooms := 5
  let students_per_classroom := 22
  let rabbits_per_classroom := 3
  let hamsters_per_classroom := 5
  let total_students := students_per_classroom * n_classrooms
  let total_rabbits := rabbits_per_classroom * n_classrooms
  let total_hamsters := hamsters_per_classroom * n_classrooms
  let total_pets := total_rabbits + total_hamsters
  total_students - total_pets = 70 :=
  by
    sorry

end NUMINAMATH_GPT_students_exceed_pets_by_70_l2192_219269


namespace NUMINAMATH_GPT_route_a_faster_by_8_minutes_l2192_219250

theorem route_a_faster_by_8_minutes :
  let route_a_distance := 8 -- miles
  let route_a_speed := 40 -- miles per hour
  let route_b_distance := 9 -- miles
  let route_b_speed := 45 -- miles per hour
  let route_b_stop := 8 -- minutes
  let time_route_a := route_a_distance / route_a_speed * 60 -- time in minutes
  let time_route_b := (route_b_distance / route_b_speed) * 60 + route_b_stop -- time in minutes
  time_route_b - time_route_a = 8 :=
by
  sorry

end NUMINAMATH_GPT_route_a_faster_by_8_minutes_l2192_219250


namespace NUMINAMATH_GPT_Tina_profit_correct_l2192_219225

theorem Tina_profit_correct :
  ∀ (price_per_book cost_per_book books_per_customer total_customers : ℕ),
  price_per_book = 20 →
  cost_per_book = 5 →
  books_per_customer = 2 →
  total_customers = 4 →
  (price_per_book * (books_per_customer * total_customers) - 
   cost_per_book * (books_per_customer * total_customers) = 120) :=
by
  intros price_per_book cost_per_book books_per_customer total_customers
  sorry

end NUMINAMATH_GPT_Tina_profit_correct_l2192_219225


namespace NUMINAMATH_GPT_population_ratio_l2192_219260

variables (Px Py Pz : ℕ)

theorem population_ratio (h1 : Py = 2 * Pz) (h2 : Px = 8 * Py) : Px / Pz = 16 :=
by
  sorry

end NUMINAMATH_GPT_population_ratio_l2192_219260


namespace NUMINAMATH_GPT_x_when_y_is_125_l2192_219210

noncomputable def C : ℝ := (2^2) * (5^2)

theorem x_when_y_is_125 
  (x y : ℝ) 
  (h_pos : x > 0 ∧ y > 0) 
  (h_inv : x^2 * y^2 = C) 
  (h_initial : y = 5) 
  (h_x_initial : x = 2) 
  (h_y : y = 125) : 
  x = 2 / 25 :=
by
  sorry

end NUMINAMATH_GPT_x_when_y_is_125_l2192_219210


namespace NUMINAMATH_GPT_ratio_B_to_A_l2192_219244

theorem ratio_B_to_A (A B C : ℝ) 
  (hA : A = 1 / 21) 
  (hC : C = 2 * B) 
  (h_sum : A + B + C = 1 / 3) : 
  B / A = 2 := 
by 
  /- Proof goes here, but it's omitted as per instructions -/
  sorry

end NUMINAMATH_GPT_ratio_B_to_A_l2192_219244


namespace NUMINAMATH_GPT_expected_score_shooting_competition_l2192_219247

theorem expected_score_shooting_competition (hit_rate : ℝ)
  (miss_both_score : ℝ) (hit_one_score : ℝ) (hit_both_score : ℝ)
  (prob_0 : ℝ) (prob_10 : ℝ) (prob_15 : ℝ) :
  hit_rate = 4 / 5 →
  miss_both_score = 0 →
  hit_one_score = 10 →
  hit_both_score = 15 →
  prob_0 = (1 - 4 / 5) * (1 - 4 / 5) →
  prob_10 = 2 * (4 / 5) * (1 - 4 / 5) →
  prob_15 = (4 / 5) * (4 / 5) →
  (0 * prob_0 + 10 * prob_10 + 15 * prob_15) = 12.8 :=
by
  intros h_hit_rate h_miss_both_score h_hit_one_score h_hit_both_score
         h_prob_0 h_prob_10 h_prob_15
  sorry

end NUMINAMATH_GPT_expected_score_shooting_competition_l2192_219247


namespace NUMINAMATH_GPT_max_gcd_of_15m_plus_4_and_14m_plus_3_l2192_219206

theorem max_gcd_of_15m_plus_4_and_14m_plus_3 (m : ℕ) (hm : 0 < m) :
  ∃ k : ℕ, k = gcd (15 * m + 4) (14 * m + 3) ∧ k = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_gcd_of_15m_plus_4_and_14m_plus_3_l2192_219206


namespace NUMINAMATH_GPT_find_parallel_lines_a_l2192_219251

/--
Given two lines \(l_1\): \(x + 2y - 3 = 0\) and \(l_2\): \(2x - ay + 3 = 0\),
prove that if the lines are parallel, then \(a = -4\).
-/
theorem find_parallel_lines_a (a : ℝ) :
  (∀ (x y : ℝ), x + 2*y - 3 = 0) 
  → (∀ (x y : ℝ), 2*x - a*y + 3 = 0)
  → (-1 / 2 = 2 / -a) 
  → a = -4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_parallel_lines_a_l2192_219251


namespace NUMINAMATH_GPT_sourball_candies_division_l2192_219232

theorem sourball_candies_division (N J L : ℕ) (total_candies : ℕ) (remaining_candies : ℕ) :
  N = 12 →
  J = N / 2 →
  L = J - 3 →
  total_candies = 30 →
  remaining_candies = total_candies - (N + J + L) →
  (remaining_candies / 3) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_sourball_candies_division_l2192_219232


namespace NUMINAMATH_GPT_cos_double_angle_l2192_219207

theorem cos_double_angle {α : ℝ} (h1 : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = Real.sqrt 5 / 3 := 
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l2192_219207


namespace NUMINAMATH_GPT_rhombus_diagonals_l2192_219283

theorem rhombus_diagonals (p d1 d2 : ℝ) (h1 : p = 100) (h2 : abs (d1 - d2) = 34) :
  ∃ d1 d2 : ℝ, d1 = 14 ∧ d2 = 48 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_rhombus_diagonals_l2192_219283


namespace NUMINAMATH_GPT_mice_population_l2192_219295

theorem mice_population :
  ∃ (mice_initial : ℕ) (pups_per_mouse : ℕ) (survival_rate_first_gen : ℕ → ℕ) 
    (survival_rate_second_gen : ℕ → ℕ) (num_dead_first_gen : ℕ) (pups_eaten_per_adult : ℕ)
    (total_mice : ℕ),
    mice_initial = 8 ∧ pups_per_mouse = 7 ∧
    (∀ n, survival_rate_first_gen n = (n * 80) / 100) ∧
    (∀ n, survival_rate_second_gen n = (n * 60) / 100) ∧
    num_dead_first_gen = 2 ∧ pups_eaten_per_adult = 3 ∧
    total_mice = mice_initial + (survival_rate_first_gen (mice_initial * pups_per_mouse)) - num_dead_first_gen + (survival_rate_second_gen ((mice_initial + (survival_rate_first_gen (mice_initial * pups_per_mouse))) * pups_per_mouse)) - ((mice_initial - num_dead_first_gen) * pups_eaten_per_adult) :=
  sorry

end NUMINAMATH_GPT_mice_population_l2192_219295


namespace NUMINAMATH_GPT_find_x_l2192_219282

-- Given condition that x is 11 percent greater than 90
def eleven_percent_greater (x : ℝ) : Prop := x = 90 + (11 / 100) * 90

-- Theorem statement
theorem find_x (x : ℝ) (h: eleven_percent_greater x) : x = 99.9 :=
  sorry

end NUMINAMATH_GPT_find_x_l2192_219282


namespace NUMINAMATH_GPT_abs_val_inequality_solution_l2192_219219

theorem abs_val_inequality_solution (x : ℝ) : |x - 2| + |x + 3| ≥ 4 ↔ x ≤ - (5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_abs_val_inequality_solution_l2192_219219


namespace NUMINAMATH_GPT_remainder_when_divided_by_5_l2192_219230

-- Definitions of the conditions
def condition1 (N : ℤ) : Prop := ∃ R1 : ℤ, N = 5 * 2 + R1
def condition2 (N : ℤ) : Prop := ∃ Q2 : ℤ, N = 4 * Q2 + 2

-- Statement to prove
theorem remainder_when_divided_by_5 (N : ℤ) (R1 : ℤ) (Q2 : ℤ) :
  (N = 5 * 2 + R1) ∧ (N = 4 * Q2 + 2) → (R1 = 4) :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_5_l2192_219230


namespace NUMINAMATH_GPT_no_base_6_digit_divisible_by_7_l2192_219216

theorem no_base_6_digit_divisible_by_7 :
  ∀ (d : ℕ), d < 6 → ¬ (7 ∣ (652 + 42 * d)) :=
by
  intros d hd
  sorry

end NUMINAMATH_GPT_no_base_6_digit_divisible_by_7_l2192_219216


namespace NUMINAMATH_GPT_common_divisors_count_l2192_219285

-- Define the numbers
def num1 := 9240
def num2 := 13860

-- Define the gcd of the numbers
def gcdNum := Nat.gcd num1 num2

-- Prove the number of divisors of the gcd is 48
theorem common_divisors_count : (Nat.divisors gcdNum).card = 48 :=
by
  -- Normally we would provide a detailed proof here
  sorry

end NUMINAMATH_GPT_common_divisors_count_l2192_219285


namespace NUMINAMATH_GPT_journey_speed_first_half_l2192_219223

theorem journey_speed_first_half (total_distance : ℕ) (total_time : ℕ) (second_half_distance : ℕ) (second_half_speed : ℕ)
  (distance_first_half_eq_half_total : second_half_distance = total_distance / 2)
  (time_for_journey_eq : total_time = 20)
  (journey_distance_eq : total_distance = 240)
  (second_half_speed_eq : second_half_speed = 15) :
  let v := second_half_distance / (total_time - (second_half_distance / second_half_speed))
  v = 10 := 
by
  sorry

end NUMINAMATH_GPT_journey_speed_first_half_l2192_219223


namespace NUMINAMATH_GPT_calc_expression_l2192_219252

theorem calc_expression : 3 * 3^4 - 9^32 / 9^30 = 162 := by
  -- We would provide the proof here, but skipping with sorry
  sorry

end NUMINAMATH_GPT_calc_expression_l2192_219252


namespace NUMINAMATH_GPT_total_production_l2192_219298

theorem total_production (S : ℝ) 
  (h1 : 4 * S = 4400) : 
  4400 + S = 5500 := 
by
  sorry

end NUMINAMATH_GPT_total_production_l2192_219298


namespace NUMINAMATH_GPT_C_is_14_years_younger_than_A_l2192_219213

variable (A B C D : ℕ)

-- Conditions
axiom cond1 : A + B = (B + C) + 14
axiom cond2 : B + D = (C + A) + 10
axiom cond3 : D = C + 6

-- To prove
theorem C_is_14_years_younger_than_A : A - C = 14 :=
by
  sorry

end NUMINAMATH_GPT_C_is_14_years_younger_than_A_l2192_219213


namespace NUMINAMATH_GPT_waiting_period_l2192_219243

-- Variable declarations
variables (P : ℕ) (H : ℕ) (W : ℕ) (A : ℕ) (T : ℕ)
-- Condition declarations
variables (hp : P = 3) (hh : H = 5 * P) (ha : A = 3 * 7) (ht : T = 39)
-- Total time equation
variables (h_total : P + H + W + A = T)

-- Statement to prove
theorem waiting_period (hp : P = 3) (hh : H = 5 * P) (ha : A = 3 * 7) (ht : T = 39) (h_total : P + H + W + A = T) : 
  W = 3 :=
sorry

end NUMINAMATH_GPT_waiting_period_l2192_219243


namespace NUMINAMATH_GPT_solve_equations_l2192_219299

theorem solve_equations :
  (∃ x1 x2 : ℝ, (x1 = 1 ∧ x2 = 3) ∧ (x1^2 - 4 * x1 + 3 = 0) ∧ (x2^2 - 4 * x2 + 3 = 0)) ∧
  (∃ y1 y2 : ℝ, (y1 = 9 ∧ y2 = 11 / 7) ∧ (4 * (2 * y1 - 5)^2 = (3 * y1 - 1)^2) ∧ (4 * (2 * y2 - 5)^2 = (3 * y2 - 1)^2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_equations_l2192_219299


namespace NUMINAMATH_GPT_cookies_per_bag_l2192_219267

-- Definitions based on given conditions
def total_cookies : ℕ := 75
def number_of_bags : ℕ := 25

-- The statement of the problem
theorem cookies_per_bag : total_cookies / number_of_bags = 3 := by
  sorry

end NUMINAMATH_GPT_cookies_per_bag_l2192_219267


namespace NUMINAMATH_GPT_determine_m_l2192_219231

def f (x : ℝ) := 5 * x^2 + 3 * x + 7
def g (x : ℝ) (m : ℝ) := 2 * x^2 - m * x + 1

theorem determine_m (m : ℝ) : f 5 - g 5 m = 55 → m = -7 :=
by
  unfold f
  unfold g
  sorry

end NUMINAMATH_GPT_determine_m_l2192_219231


namespace NUMINAMATH_GPT_value_of_work_clothes_l2192_219220

theorem value_of_work_clothes (x y : ℝ) (h1 : x + 70 = 30 * y) (h2 : x + 20 = 20 * y) : x = 80 :=
by
  sorry

end NUMINAMATH_GPT_value_of_work_clothes_l2192_219220


namespace NUMINAMATH_GPT_factorization1_factorization2_factorization3_l2192_219246

-- Problem 1
theorem factorization1 (a x : ℝ) : a * x^2 - 4 * a = a * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorization2 (m x y : ℝ) : m * x^2 + 2 * m * x * y + m * y^2 = m * (x + y)^2 :=
sorry

-- Problem 3
theorem factorization3 (a b : ℝ) : (1 / 2) * a^2 - a * b + (1 / 2) * b^2 = (1 / 2) * (a - b)^2 :=
sorry

end NUMINAMATH_GPT_factorization1_factorization2_factorization3_l2192_219246


namespace NUMINAMATH_GPT_simplify_fraction_subtraction_l2192_219249

theorem simplify_fraction_subtraction : (1 / 210) - (17 / 35) = -101 / 210 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_subtraction_l2192_219249


namespace NUMINAMATH_GPT_jerry_age_l2192_219233

variable (M J : ℕ) -- Declare Mickey's and Jerry's ages as natural numbers

-- Define the conditions as hypotheses
def condition1 := M = 2 * J - 6
def condition2 := M = 18

-- Theorem statement where we need to prove J = 12 given the conditions
theorem jerry_age
  (h1 : condition1 M J)
  (h2 : condition2 M) :
  J = 12 :=
sorry

end NUMINAMATH_GPT_jerry_age_l2192_219233


namespace NUMINAMATH_GPT_number_of_teachers_l2192_219291

theorem number_of_teachers
    (number_of_students : ℕ)
    (classes_per_student : ℕ)
    (classes_per_teacher : ℕ)
    (students_per_class : ℕ)
    (total_teachers : ℕ)
    (h1 : number_of_students = 2400)
    (h2 : classes_per_student = 5)
    (h3 : classes_per_teacher = 4)
    (h4 : students_per_class = 30)
    (h5 : total_teachers * classes_per_teacher * students_per_class = number_of_students * classes_per_student) :
    total_teachers = 100 :=
by
  sorry

end NUMINAMATH_GPT_number_of_teachers_l2192_219291


namespace NUMINAMATH_GPT_charity_distribution_l2192_219229

theorem charity_distribution
    (amount_raised : ℝ)
    (donation_percentage : ℝ)
    (num_organizations : ℕ)
    (h_amount_raised : amount_raised = 2500)
    (h_donation_percentage : donation_percentage = 0.80)
    (h_num_organizations : num_organizations = 8) :
    (amount_raised * donation_percentage) / num_organizations = 250 := by
  sorry

end NUMINAMATH_GPT_charity_distribution_l2192_219229


namespace NUMINAMATH_GPT_add_to_frac_eq_l2192_219228

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_add_to_frac_eq_l2192_219228
