import Mathlib

namespace NUMINAMATH_GPT_intersecting_sets_a_eq_1_l1621_162107

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := { x | a * x^2 - 1 = 0 }
def N : Set ℝ := { -1/2, 1/2, 1 }

-- Define the intersection condition
def sets_intersect (M N : Set ℝ) : Prop :=
  ∃ x, x ∈ M ∧ x ∈ N

-- Statement of the problem
theorem intersecting_sets_a_eq_1 (a : ℝ) (h_intersect : sets_intersect (M a) N) : a = 1 :=
  sorry

end NUMINAMATH_GPT_intersecting_sets_a_eq_1_l1621_162107


namespace NUMINAMATH_GPT_find_f_neg3_l1621_162141

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x^2 - 2 * x else -(x^2 - 2 * -x)

theorem find_f_neg3 (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_pos : ∀ x : ℝ, 0 < x → f x = x^2 - 2 * x) : f (-3) = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg3_l1621_162141


namespace NUMINAMATH_GPT_find_other_asymptote_l1621_162191

-- Define the conditions
def one_asymptote (x : ℝ) : ℝ := 3 * x
def foci_x_coordinate : ℝ := 5

-- Define the expected answer
def other_asymptote (x : ℝ) : ℝ := -3 * x + 30

-- Theorem statement to prove the equation of the other asymptote
theorem find_other_asymptote :
  (∀ x, y = one_asymptote x) →
  (∀ _x, _x = foci_x_coordinate) →
  (∀ x, y = other_asymptote x) :=
by
  intros h_one_asymptote h_foci_x
  sorry

end NUMINAMATH_GPT_find_other_asymptote_l1621_162191


namespace NUMINAMATH_GPT_find_principal_l1621_162133

-- Define the conditions
def interest_rate : ℝ := 0.05
def time_period : ℕ := 10
def interest_less_than_principal : ℝ := 3100

-- Define the principal
def principal : ℝ := 6200

-- The theorem statement
theorem find_principal :
  ∃ P : ℝ, P - interest_less_than_principal = P * interest_rate * time_period ∧ P = principal :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l1621_162133


namespace NUMINAMATH_GPT_value_of_f1_l1621_162152

variable (f : ℝ → ℝ)
open Function

theorem value_of_f1
  (h : ∀ x y : ℝ, f (f (x - y)) = f x * f y - f x + f y - 2 * x * y + 2 * x - 2 * y) :
  f 1 = -1 :=
sorry

end NUMINAMATH_GPT_value_of_f1_l1621_162152


namespace NUMINAMATH_GPT_tan_sum_pi_eighths_l1621_162171

theorem tan_sum_pi_eighths : (Real.tan (Real.pi / 8) + Real.tan (3 * Real.pi / 8) = 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_pi_eighths_l1621_162171


namespace NUMINAMATH_GPT_solid_color_marble_percentage_l1621_162113

theorem solid_color_marble_percentage (solid striped dotted swirl red blue green yellow purple : ℝ)
  (h_solid: solid = 0.7) (h_striped: striped = 0.1) (h_dotted: dotted = 0.1) (h_swirl: swirl = 0.1)
  (h_red: red = 0.25) (h_blue: blue = 0.25) (h_green: green = 0.2) (h_yellow: yellow = 0.15) (h_purple: purple = 0.15) :
  solid * (red + blue + green) * 100 = 49 :=
by
  sorry

end NUMINAMATH_GPT_solid_color_marble_percentage_l1621_162113


namespace NUMINAMATH_GPT_find_number_l1621_162139

variable {x : ℝ}

theorem find_number (h : (30 / 100) * x = (40 / 100) * 40) : x = 160 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1621_162139


namespace NUMINAMATH_GPT_interpretation_of_neg_two_pow_six_l1621_162140

theorem interpretation_of_neg_two_pow_six :
  - (2^6) = -(6 * 2) :=
by
  sorry

end NUMINAMATH_GPT_interpretation_of_neg_two_pow_six_l1621_162140


namespace NUMINAMATH_GPT_find_larger_number_l1621_162187

theorem find_larger_number 
  (x y : ℚ) 
  (h1 : 4 * y = 9 * x) 
  (h2 : y - x = 12) : 
  y = 108 / 5 := 
sorry

end NUMINAMATH_GPT_find_larger_number_l1621_162187


namespace NUMINAMATH_GPT_number_of_teams_l1621_162165

theorem number_of_teams (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : x = 8 :=
sorry

end NUMINAMATH_GPT_number_of_teams_l1621_162165


namespace NUMINAMATH_GPT_student_score_in_first_subject_l1621_162145

theorem student_score_in_first_subject 
  (x : ℝ)  -- Percentage in the first subject
  (w : ℝ)  -- Constant weight (as all subjects have same weight)
  (S2_score : ℝ)  -- Score in the second subject
  (S3_score : ℝ)  -- Score in the third subject
  (target_avg : ℝ) -- Target average score
  (hS2 : S2_score = 70)  -- Second subject score is 70%
  (hS3 : S3_score = 80)  -- Third subject score is 80%
  (havg : (x + S2_score + S3_score) / 3 = target_avg) :  -- The desired average is equal to the target average
  target_avg = 70 → x = 60 :=   -- Target average score is 70%
by
  sorry

end NUMINAMATH_GPT_student_score_in_first_subject_l1621_162145


namespace NUMINAMATH_GPT_exists_nat_solution_for_A_415_l1621_162143

theorem exists_nat_solution_for_A_415 : ∃ (m n : ℕ), 3 * m^2 * n = n^3 + 415 := by
  sorry

end NUMINAMATH_GPT_exists_nat_solution_for_A_415_l1621_162143


namespace NUMINAMATH_GPT_n_plus_one_sum_of_three_squares_l1621_162175

theorem n_plus_one_sum_of_three_squares (n x : ℤ) (h1 : n > 1) (h2 : 3 * n + 1 = x^2) :
  ∃ a b c : ℤ, n + 1 = a^2 + b^2 + c^2 :=
by
  sorry

end NUMINAMATH_GPT_n_plus_one_sum_of_three_squares_l1621_162175


namespace NUMINAMATH_GPT_percentage_decrease_in_savings_l1621_162126

theorem percentage_decrease_in_savings (I : ℝ) (F : ℝ) (IncPercent : ℝ) (decPercent : ℝ)
  (h1 : I = 125) (h2 : IncPercent = 0.25) (h3 : F = 125) :
  let P := (I * (1 + IncPercent))
  ∃ decPercent, decPercent = ((P - F) / P) * 100 ∧ decPercent = 20 := 
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_in_savings_l1621_162126


namespace NUMINAMATH_GPT_exponent_of_term_on_right_side_l1621_162157

theorem exponent_of_term_on_right_side
  (s m : ℕ) 
  (h1 : (2^16) * (25^s) = 5 * (10^m))
  (h2 : m = 16) : m = 16 := 
by
  sorry

end NUMINAMATH_GPT_exponent_of_term_on_right_side_l1621_162157


namespace NUMINAMATH_GPT_ceil_minus_val_eq_one_minus_frac_l1621_162185

variable (x : ℝ)

theorem ceil_minus_val_eq_one_minus_frac (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ f : ℝ, 0 ≤ f ∧ f < 1 ∧ ⌈x⌉ - x = 1 - f := 
sorry

end NUMINAMATH_GPT_ceil_minus_val_eq_one_minus_frac_l1621_162185


namespace NUMINAMATH_GPT_minimum_point_translation_l1621_162134

theorem minimum_point_translation (x y : ℝ) : 
  (∀ (x : ℝ), y = 2 * |x| - 4) →
  x = 0 →
  y = -4 →
  (∀ (x y : ℝ), x_new = x + 3 ∧ y_new = y + 4) →
  (x_new, y_new) = (3, 0) :=
sorry

end NUMINAMATH_GPT_minimum_point_translation_l1621_162134


namespace NUMINAMATH_GPT_smallest_k_for_sequence_l1621_162190

theorem smallest_k_for_sequence (a : ℕ → ℕ) (k : ℕ) (h₁ : a 1 = 1) (h₂ : a 2018 = 2020)
  (h₃ : ∀ n, n ≥ 2 → a (n+1) = k * (a n) / (a (n-1))) : k = 2020 :=
sorry

end NUMINAMATH_GPT_smallest_k_for_sequence_l1621_162190


namespace NUMINAMATH_GPT_triangle_arithmetic_angles_l1621_162132

/-- The angles in a triangle are in arithmetic progression and the side lengths are 6, 7, and y.
    The sum of the possible values of y equals a + sqrt b + sqrt c,
    where a, b, and c are positive integers. Prove that a + b + c = 68. -/
theorem triangle_arithmetic_angles (y : ℝ) (a b c : ℕ) (h1 : a = 3) (h2 : b = 22) (h3 : c = 43) :
    (∃ y1 y2 : ℝ, y1 = 3 + Real.sqrt 22 ∧ y2 = Real.sqrt 43 ∧ (y = y1 ∨ y = y2))
    → a + b + c = 68 :=
by
  sorry

end NUMINAMATH_GPT_triangle_arithmetic_angles_l1621_162132


namespace NUMINAMATH_GPT_right_triangle_area_l1621_162117

theorem right_triangle_area (base hypotenuse : ℕ) (h_base : base = 8) (h_hypotenuse : hypotenuse = 10) :
  ∃ height : ℕ, height^2 = hypotenuse^2 - base^2 ∧ (base * height) / 2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1621_162117


namespace NUMINAMATH_GPT_total_birds_correct_l1621_162158

-- Define the conditions
def number_of_trees : ℕ := 7
def blackbirds_per_tree : ℕ := 3
def magpies : ℕ := 13

-- Define the total number of blackbirds using the conditions
def total_blackbirds : ℕ := number_of_trees * blackbirds_per_tree

-- Define the total number of birds using the total number of blackbirds and the number of magpies
def total_birds : ℕ := total_blackbirds + magpies

-- The theorem statement that should be proven
theorem total_birds_correct : total_birds = 34 := 
sorry

end NUMINAMATH_GPT_total_birds_correct_l1621_162158


namespace NUMINAMATH_GPT_carlos_marbles_l1621_162111

theorem carlos_marbles:
  ∃ M, M > 1 ∧ 
       M % 5 = 1 ∧ 
       M % 7 = 1 ∧ 
       M % 11 = 1 ∧ 
       M % 4 = 2 ∧ 
       M = 386 := by
  sorry

end NUMINAMATH_GPT_carlos_marbles_l1621_162111


namespace NUMINAMATH_GPT_potato_bag_weight_l1621_162109

theorem potato_bag_weight :
  ∃ w : ℝ, w = 16 / (w / 4) ∧ w = 16 := 
by
  sorry

end NUMINAMATH_GPT_potato_bag_weight_l1621_162109


namespace NUMINAMATH_GPT_find_f2_plus_g2_l1621_162164

variable (f g : ℝ → ℝ)

def even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x
def odd_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem find_f2_plus_g2 (hf : even_function f) (hg : odd_function g) (h : ∀ x, f x - g x = x^3 - 2 * x^2) :
  f 2 + g 2 = -16 :=
sorry

end NUMINAMATH_GPT_find_f2_plus_g2_l1621_162164


namespace NUMINAMATH_GPT_smaller_number_is_four_l1621_162181

theorem smaller_number_is_four (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 10) : y = 4 :=
by
  sorry

end NUMINAMATH_GPT_smaller_number_is_four_l1621_162181


namespace NUMINAMATH_GPT_solve_x_l1621_162174

variable (x : ℝ)

def vector_a := (2, 1)
def vector_b := (1, x)

def vectors_parallel : Prop :=
  let a_plus_b := (2 + 1, 1 + x)
  let a_minus_b := (2 - 1, 1 - x)
  a_plus_b.1 * a_minus_b.2 = a_plus_b.2 * a_minus_b.1

theorem solve_x (hx : vectors_parallel x) : x = 1/2 := by
  sorry

end NUMINAMATH_GPT_solve_x_l1621_162174


namespace NUMINAMATH_GPT_percent_covered_by_larger_triangles_l1621_162122

-- Define the number of small triangles in one large hexagon
def total_small_triangles := 16

-- Define the number of small triangles that are part of the larger triangles within one hexagon
def small_triangles_in_larger_triangles := 9

-- Calculate the fraction of the area of the hexagon covered by larger triangles
def fraction_covered_by_larger_triangles := 
  small_triangles_in_larger_triangles / total_small_triangles

-- Define the expected result as a fraction of the total area
def expected_fraction := 56 / 100

-- The proof problem in Lean 4 statement:
theorem percent_covered_by_larger_triangles
  (h1 : fraction_covered_by_larger_triangles = 9 / 16) :
  fraction_covered_by_larger_triangles = expected_fraction :=
  by
    sorry

end NUMINAMATH_GPT_percent_covered_by_larger_triangles_l1621_162122


namespace NUMINAMATH_GPT_find_vector_at_6_l1621_162151

structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vec_add (v1 v2 : Vector3D) : Vector3D :=
  { x := v1.x + v2.x, y := v1.y + v2.y, z := v1.z + v2.z }

def vec_scale (c : ℝ) (v : Vector3D) : Vector3D :=
  { x := c * v.x, y := c * v.y, z := c * v.z }

noncomputable def vector_at_t (a d : Vector3D) (t : ℝ) : Vector3D :=
  vec_add a (vec_scale t d)

theorem find_vector_at_6 :
  let a := { x := 2, y := -1, z := 3 }
  let d := { x := 1, y := 2, z := -1 }
  vector_at_t a d 6 = { x := 8, y := 11, z := -3 } :=
by
  sorry

end NUMINAMATH_GPT_find_vector_at_6_l1621_162151


namespace NUMINAMATH_GPT_original_distance_cycled_l1621_162147

theorem original_distance_cycled
  (x t d : ℝ)
  (h1 : d = x * t)
  (h2 : d = (x + 1/4) * (3/4 * t))
  (h3 : d = (x - 1/4) * (t + 3)) :
  d = 4.5 := 
sorry

end NUMINAMATH_GPT_original_distance_cycled_l1621_162147


namespace NUMINAMATH_GPT_jacket_initial_reduction_l1621_162129

theorem jacket_initial_reduction (x : ℝ) :
  (1 - x / 100) * 1.53846 = 1 → x = 35 :=
by
  sorry

end NUMINAMATH_GPT_jacket_initial_reduction_l1621_162129


namespace NUMINAMATH_GPT_coprime_unique_residues_non_coprime_same_residue_l1621_162124

-- Part (a)

theorem coprime_unique_residues (m k : ℕ) (h : m.gcd k = 1) : 
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ), 
    ∀ (i : Fin m) (j : Fin k), 
      ∀ (i' : Fin m) (j' : Fin k), 
        (i, j) ≠ (i', j') → (a i * b j) % (m * k) ≠ (a i' * b j') % (m * k) := 
sorry

-- Part (b)

theorem non_coprime_same_residue (m k : ℕ) (h : m.gcd k > 1) : 
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ), 
    ∃ (i : Fin m) (j : Fin k) (i' : Fin m) (j' : Fin k), 
      (i, j) ≠ (i', j') ∧ (a i * b j) % (m * k) = (a i' * b j') % (m * k) := 
sorry

end NUMINAMATH_GPT_coprime_unique_residues_non_coprime_same_residue_l1621_162124


namespace NUMINAMATH_GPT_susan_homework_time_l1621_162184

theorem susan_homework_time :
  ∀ (start finish practice : ℕ),
  start = 119 ->
  practice = 240 ->
  finish = practice - 25 ->
  (start < finish) ->
  (finish - start) = 96 :=
by
  intros start finish practice h_start h_practice h_finish h_lt
  sorry

end NUMINAMATH_GPT_susan_homework_time_l1621_162184


namespace NUMINAMATH_GPT_tom_reads_pages_l1621_162195

-- Definition of conditions
def initial_speed : ℕ := 12   -- pages per hour
def speed_factor : ℕ := 3
def time_period : ℕ := 2     -- hours

-- Calculated speeds
def increased_speed (initial_speed speed_factor : ℕ) : ℕ := initial_speed * speed_factor
def total_pages (increased_speed time_period : ℕ) : ℕ := increased_speed * time_period

-- Theorem statement
theorem tom_reads_pages :
  total_pages (increased_speed initial_speed speed_factor) time_period = 72 :=
by
  -- Omitting proof as only theorem statement is required
  sorry

end NUMINAMATH_GPT_tom_reads_pages_l1621_162195


namespace NUMINAMATH_GPT_average_value_of_T_l1621_162176

def average_T (boys girls : ℕ) (starts_with_boy : Bool) (ends_with_girl : Bool) : ℕ :=
  if boys = 9 ∧ girls = 15 ∧ starts_with_boy ∧ ends_with_girl then 12 else 0

theorem average_value_of_T :
  average_T 9 15 true true = 12 :=
sorry

end NUMINAMATH_GPT_average_value_of_T_l1621_162176


namespace NUMINAMATH_GPT_find_p_l1621_162142

theorem find_p (p : ℕ) : 18^3 = (16^2 / 4) * 2^(8 * p) → p = 0 := 
by 
  sorry

end NUMINAMATH_GPT_find_p_l1621_162142


namespace NUMINAMATH_GPT_sequence_k_value_l1621_162168

theorem sequence_k_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ m n : ℕ, a (m + n) = a m * a n)
  (hk1 : ∀ k : ℕ, a (k + 1) = 1024) :
  ∃ k : ℕ, k = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_k_value_l1621_162168


namespace NUMINAMATH_GPT_children_to_add_l1621_162104

def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def desired_children := 30

theorem children_to_add : (desired_children - children) = 10 := by
  sorry

end NUMINAMATH_GPT_children_to_add_l1621_162104


namespace NUMINAMATH_GPT_min_distance_convex_lens_l1621_162100

theorem min_distance_convex_lens (t k f : ℝ) (hf : f > 0) (ht : t ≥ f)
    (h_lens: 1 / t + 1 / k = 1 / f) :
  t = 2 * f → t + k = 4 * f :=
by
  sorry

end NUMINAMATH_GPT_min_distance_convex_lens_l1621_162100


namespace NUMINAMATH_GPT_find_original_number_l1621_162183

theorem find_original_number (k : ℤ) (h : 25 * k = N + 4) : ∃ N, N = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l1621_162183


namespace NUMINAMATH_GPT_one_cow_one_bag_in_39_days_l1621_162106

-- Definitions
def cows : ℕ := 52
def husks : ℕ := 104
def days : ℕ := 78

-- Problem: Given that 52 cows eat 104 bags of husk in 78 days,
-- Prove that one cow will eat one bag of husk in 39 days.
theorem one_cow_one_bag_in_39_days (cows_cons : cows = 52) (husks_cons : husks = 104) (days_cons : days = 78) :
  ∃ d : ℕ, d = 39 :=
by
  -- Placeholder for the proof.
  sorry

end NUMINAMATH_GPT_one_cow_one_bag_in_39_days_l1621_162106


namespace NUMINAMATH_GPT_friends_bought_boxes_l1621_162166

def rainbow_colors : ℕ := 7
def total_pencils : ℕ := 56
def pencils_per_box : ℕ := rainbow_colors

theorem friends_bought_boxes (emily_box : ℕ := 1) :
  (total_pencils / pencils_per_box) - emily_box = 7 := by
  sorry

end NUMINAMATH_GPT_friends_bought_boxes_l1621_162166


namespace NUMINAMATH_GPT_max_distinct_terms_degree_6_l1621_162116

-- Step 1: Define the variables and conditions
def polynomial_max_num_terms (deg : ℕ) (vars : ℕ) : ℕ :=
  Nat.choose (deg + vars - 1) (vars - 1)

-- Step 2: State the specific problem
theorem max_distinct_terms_degree_6 :
  polynomial_max_num_terms 6 5 = 210 :=
by
  sorry

end NUMINAMATH_GPT_max_distinct_terms_degree_6_l1621_162116


namespace NUMINAMATH_GPT_age_ratio_l1621_162101

variable (Cindy Jan Marcia Greg: ℕ)

theorem age_ratio 
  (h1 : Cindy = 5)
  (h2 : Jan = Cindy + 2)
  (h3: Greg = 16)
  (h4 : Greg = Marcia + 2)
  (h5 : ∃ k : ℕ, Marcia = k * Jan) 
  : Marcia / Jan = 2 := 
    sorry

end NUMINAMATH_GPT_age_ratio_l1621_162101


namespace NUMINAMATH_GPT_find_n_l1621_162180

theorem find_n
    (h : Real.arctan (1 / 2) + Real.arctan (1 / 3) + Real.arctan (1 / 7) + Real.arctan (1 / n) = Real.pi / 2) :
    n = 46 :=
sorry

end NUMINAMATH_GPT_find_n_l1621_162180


namespace NUMINAMATH_GPT_average_of_multiples_l1621_162159

theorem average_of_multiples (n : ℕ) (hn : n > 0) :
  (60.5 : ℚ) = ((n / 2) * (11 + 11 * n)) / n → n = 10 :=
by
  sorry

end NUMINAMATH_GPT_average_of_multiples_l1621_162159


namespace NUMINAMATH_GPT_repeated_root_cubic_l1621_162179

theorem repeated_root_cubic (p : ℝ) :
  (∃ x : ℝ, (3 * x^3 - (p + 1) * x^2 + 4 * x - 12 = 0) ∧ (9 * x^2 - 2 * (p + 1) * x + 4 = 0)) →
  (p = 5 ∨ p = -7) :=
by
  sorry

end NUMINAMATH_GPT_repeated_root_cubic_l1621_162179


namespace NUMINAMATH_GPT_cycle_price_reduction_l1621_162199

theorem cycle_price_reduction (original_price : ℝ) :
  let price_after_first_reduction := original_price * 0.75
  let price_after_second_reduction := price_after_first_reduction * 0.60
  (original_price - price_after_second_reduction) / original_price = 0.55 :=
by
  sorry

end NUMINAMATH_GPT_cycle_price_reduction_l1621_162199


namespace NUMINAMATH_GPT_probability_at_5_5_equals_1_over_243_l1621_162186

-- Define the base probability function P
def P : ℕ → ℕ → ℚ
| 0, 0       => 1
| x+1, 0     => 0
| 0, y+1     => 0
| x+1, y+1   => (1/3 : ℚ) * P x (y+1) + (1/3 : ℚ) * P (x+1) y + (1/3 : ℚ) * P x y

-- Theorem statement that needs to be proved
theorem probability_at_5_5_equals_1_over_243 : P 5 5 = 1 / 243 :=
sorry

end NUMINAMATH_GPT_probability_at_5_5_equals_1_over_243_l1621_162186


namespace NUMINAMATH_GPT_Niko_total_profit_l1621_162162

-- Definitions based on conditions
def cost_per_pair : ℕ := 2
def total_pairs : ℕ := 9
def profit_margin_4_pairs : ℚ := 0.25
def profit_per_other_pair : ℚ := 0.2
def pairs_with_margin : ℕ := 4
def pairs_with_fixed_profit : ℕ := 5

-- Calculations based on definitions
def total_cost : ℚ := total_pairs * cost_per_pair
def profit_on_margin_pairs : ℚ := pairs_with_margin * (profit_margin_4_pairs * cost_per_pair)
def profit_on_fixed_profit_pairs : ℚ := pairs_with_fixed_profit * profit_per_other_pair
def total_profit : ℚ := profit_on_margin_pairs + profit_on_fixed_profit_pairs

-- Statement to prove
theorem Niko_total_profit : total_profit = 3 := by
  sorry

end NUMINAMATH_GPT_Niko_total_profit_l1621_162162


namespace NUMINAMATH_GPT_identify_quadratic_equation_l1621_162121

def is_quadratic (eq : String) : Prop :=
  eq = "a * x^2 + b * x + c = 0"  /-
  This definition is a placeholder for checking if a 
  given equation is in the quadratic form. In practice,
  more advanced techniques like parsing and formally
  verifying the quadratic form would be used. -/

theorem identify_quadratic_equation :
  (is_quadratic "2 * x^2 - x - 3 = 0") :=
by
  sorry

end NUMINAMATH_GPT_identify_quadratic_equation_l1621_162121


namespace NUMINAMATH_GPT_range_of_x_l1621_162110

theorem range_of_x (a : ℝ) (x : ℝ) (h_a : 1 ≤ a ∧ a ≤ 3) (h : a * x^2 + (a - 2) * x - 2 > 0) :
  x < -1 ∨ x > 2 / 3 :=
sorry

end NUMINAMATH_GPT_range_of_x_l1621_162110


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1621_162156

theorem quadratic_inequality_solution (a b c : ℝ) (h_solution_set : ∀ x, ax^2 + bx + c < 0 ↔ x < -1 ∨ x > 3) :
  (a < 0) ∧
  (a + b + c > 0) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ -1/3 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1621_162156


namespace NUMINAMATH_GPT_cone_altitude_ratio_l1621_162172

variable (r h : ℝ)
variable (radius_condition : r > 0)
variable (volume_condition : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3)

theorem cone_altitude_ratio {r h : ℝ}
  (radius_condition : r > 0) 
  (volume_condition : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_cone_altitude_ratio_l1621_162172


namespace NUMINAMATH_GPT_triangle_angles_l1621_162161

theorem triangle_angles
  (A B C M : Type)
  (ortho_divides_height_A : ∀ (H_AA1 : ℝ), ∃ (H_AM : ℝ), H_AA1 = H_AM * 3 ∧ H_AM = 2 * H_AA1 / 3)
  (ortho_divides_height_B : ∀ (H_BB1 : ℝ), ∃ (H_BM : ℝ), H_BB1 = H_BM * 5 / 2 ∧ H_BM = 3 * H_BB1 / 5) :
  ∃ α β γ : ℝ, α = 60 + 40 / 60 ∧ β = 64 + 36 / 60 ∧ γ = 54 + 44 / 60 :=
by { 
  sorry 
}

end NUMINAMATH_GPT_triangle_angles_l1621_162161


namespace NUMINAMATH_GPT_question1_question2_question3_l1621_162194

-- Define probabilities of renting and returning bicycles at different stations
def P (X Y : Char) : ℝ :=
  if X = 'A' ∧ Y = 'A' then 0.3 else
  if X = 'A' ∧ Y = 'B' then 0.2 else
  if X = 'A' ∧ Y = 'C' then 0.5 else
  if X = 'B' ∧ Y = 'A' then 0.7 else
  if X = 'B' ∧ Y = 'B' then 0.1 else
  if X = 'B' ∧ Y = 'C' then 0.2 else
  if X = 'C' ∧ Y = 'A' then 0.4 else
  if X = 'C' ∧ Y = 'B' then 0.5 else
  if X = 'C' ∧ Y = 'C' then 0.1 else 0

-- Question 1: Prove P(CC) = 0.1
theorem question1 : P 'C' 'C' = 0.1 := by
  sorry

-- Question 2: Prove P(AC) * P(CB) = 0.25
theorem question2 : P 'A' 'C' * P 'C' 'B' = 0.25 := by
  sorry

-- Question 3: Prove the probability P = 0.43
theorem question3 : P 'A' 'A' * P 'A' 'A' + P 'A' 'B' * P 'B' 'A' + P 'A' 'C' * P 'C' 'A' = 0.43 := by
  sorry

end NUMINAMATH_GPT_question1_question2_question3_l1621_162194


namespace NUMINAMATH_GPT_number_of_outfits_l1621_162136

theorem number_of_outfits (shirts pants : ℕ) (h_shirts : shirts = 5) (h_pants : pants = 3) 
    : shirts * pants = 15 := by
  sorry

end NUMINAMATH_GPT_number_of_outfits_l1621_162136


namespace NUMINAMATH_GPT_gasoline_storage_l1621_162167

noncomputable def total_distance : ℕ := 280 * 2

noncomputable def miles_per_segment : ℕ := 40

noncomputable def gasoline_consumption : ℕ := 8

noncomputable def total_segments : ℕ := total_distance / miles_per_segment

noncomputable def total_gasoline : ℕ := total_segments * gasoline_consumption

noncomputable def number_of_refills : ℕ := 14

theorem gasoline_storage (storage_capacity : ℕ) (h : number_of_refills * storage_capacity = total_gasoline) :
  storage_capacity = 8 :=
by
  sorry

end NUMINAMATH_GPT_gasoline_storage_l1621_162167


namespace NUMINAMATH_GPT_class_size_is_10_l1621_162108

theorem class_size_is_10 
  (num_92 : ℕ) (num_80 : ℕ) (last_score : ℕ) (target_avg : ℕ) (total_score : ℕ) 
  (h_num_92 : num_92 = 5) (h_num_80 : num_80 = 4) (h_last_score : last_score = 70) 
  (h_target_avg : target_avg = 85) (h_total_score : total_score = 85 * (num_92 + num_80 + 1)) 
  : (num_92 * 92 + num_80 * 80 + last_score = total_score) → 
    (num_92 + num_80 + 1 = 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_class_size_is_10_l1621_162108


namespace NUMINAMATH_GPT_functional_equation_solution_l1621_162160

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, 
  (∀ x y : ℝ, 
      y * f (2 * x) - x * f (2 * y) = 8 * x * y * (x^2 - y^2)
  ) → (∃ c : ℝ, ∀ x : ℝ, f x = x^3 + c * x) :=
by { sorry }

end NUMINAMATH_GPT_functional_equation_solution_l1621_162160


namespace NUMINAMATH_GPT_zero_clever_numbers_l1621_162112

def isZeroClever (n : Nat) : Prop :=
  ∃ a b c : Nat, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
  n = 1000 * a + 10 * b + c ∧
  n = 9 * (100 * a + 10 * b + c)

theorem zero_clever_numbers :
  ∀ n : Nat, isZeroClever n → n = 2025 ∨ n = 4050 ∨ n = 6075 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_zero_clever_numbers_l1621_162112


namespace NUMINAMATH_GPT_multiplication_correct_l1621_162169

theorem multiplication_correct :
  375680169467 * 4565579427629 = 1715110767607750737263 :=
  by sorry

end NUMINAMATH_GPT_multiplication_correct_l1621_162169


namespace NUMINAMATH_GPT_xiao_ming_fails_the_test_probability_l1621_162135

def probability_scoring_above_80 : ℝ := 0.69
def probability_scoring_between_70_and_79 : ℝ := 0.15
def probability_scoring_between_60_and_69 : ℝ := 0.09

theorem xiao_ming_fails_the_test_probability :
  1 - (probability_scoring_above_80 + probability_scoring_between_70_and_79 + probability_scoring_between_60_and_69) = 0.07 :=
by
  sorry

end NUMINAMATH_GPT_xiao_ming_fails_the_test_probability_l1621_162135


namespace NUMINAMATH_GPT_gcd_three_digit_palindromes_l1621_162128

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_gcd_three_digit_palindromes_l1621_162128


namespace NUMINAMATH_GPT_modular_inverse_sum_correct_l1621_162130

theorem modular_inverse_sum_correct :
  (3 * 8 + 9 * 13) % 56 = 29 :=
by
  sorry

end NUMINAMATH_GPT_modular_inverse_sum_correct_l1621_162130


namespace NUMINAMATH_GPT_calc_two_pow_a_mul_two_pow_b_l1621_162197

theorem calc_two_pow_a_mul_two_pow_b {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : (2^a)^b = 2^2) :
  2^a * 2^b = 8 :=
sorry

end NUMINAMATH_GPT_calc_two_pow_a_mul_two_pow_b_l1621_162197


namespace NUMINAMATH_GPT_angle_cosine_third_quadrant_l1621_162137

theorem angle_cosine_third_quadrant (B : ℝ) (h1 : π < B ∧ B < 3 * π / 2) (h2 : Real.sin B = 4 / 5) :
  Real.cos B = -3 / 5 :=
sorry

end NUMINAMATH_GPT_angle_cosine_third_quadrant_l1621_162137


namespace NUMINAMATH_GPT_intersection_hyperbola_circle_l1621_162149

theorem intersection_hyperbola_circle :
  {p : ℝ × ℝ | p.1^2 - 9 * p.2^2 = 36 ∧ p.1^2 + p.2^2 = 36} = {(6, 0), (-6, 0)} :=
by sorry

end NUMINAMATH_GPT_intersection_hyperbola_circle_l1621_162149


namespace NUMINAMATH_GPT_solution_set_l1621_162153

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- conditions
axiom differentiable_on_f : ∀ x < 0, DifferentiableAt ℝ f x
axiom derivative_f_x : ∀ x < 0, deriv f x = f' x

axiom condition_3fx_xf'x : ∀ x < 0, 3 * f x + x * f' x > 0

-- goal
theorem solution_set :
  ∀ x, (-2020 < x ∧ x < -2017) ↔ ((x + 2017)^3 * f (x + 2017) + 27 * f (-3) > 0) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1621_162153


namespace NUMINAMATH_GPT_rectangle_side_ratio_l1621_162120

theorem rectangle_side_ratio (s x y : ℝ) 
  (h1 : 8 * (x * y) = (9 - 1) * s^2) 
  (h2 : s + 4 * y = 3 * s) 
  (h3 : 2 * x + y = 3 * s) : 
  x / y = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_side_ratio_l1621_162120


namespace NUMINAMATH_GPT_sally_eggs_l1621_162178

def dozen := 12
def total_eggs := 48

theorem sally_eggs : total_eggs / dozen = 4 := by
  -- Normally a proof would follow here, but we will use sorry to skip it
  sorry

end NUMINAMATH_GPT_sally_eggs_l1621_162178


namespace NUMINAMATH_GPT_juan_original_number_l1621_162103

theorem juan_original_number (x : ℝ) (h : (3 * (x + 3) - 4) / 2 = 10) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_juan_original_number_l1621_162103


namespace NUMINAMATH_GPT_students_before_intersection_equal_l1621_162119

-- Define the conditions
def students_after_stop : Nat := 58
def percentage : Real := 0.40
def percentage_students_entered : Real := 12

-- Define the target number of students before stopping
def students_before_stop (total_after : Nat) (entered : Nat) : Nat :=
  total_after - entered

-- State the proof problem
theorem students_before_intersection_equal :
  ∃ (x : Nat), 
  percentage * (x : Real) = percentage_students_entered ∧ 
  students_before_stop students_after_stop x = 28 :=
by
  sorry

end NUMINAMATH_GPT_students_before_intersection_equal_l1621_162119


namespace NUMINAMATH_GPT_find_n_l1621_162118

theorem find_n (n : ℕ) (h : 2^n = 2 * 16^2 * 4^3) : n = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1621_162118


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t_3_l1621_162196

variable (t : ℝ)
def s (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_t_3 : 
  ∃ v, v = -1 + 2 * 3 ∧ v = 5 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t_3_l1621_162196


namespace NUMINAMATH_GPT_function_property_l1621_162192

noncomputable def f (x : ℝ) : ℝ := sorry
variable (a x1 x2 : ℝ)

-- Conditions
axiom f_defined_on_R : ∀ x : ℝ, f x ≠ 0
axiom f_increasing_on_left_of_a : ∀ x y : ℝ, x < y → y < a → f x < f y
axiom f_even_shifted_by_a : ∀ x : ℝ, f (x + a) = f (-(x + a))
axiom ordering : x1 < a ∧ a < x2
axiom distance_comp : |x1 - a| < |x2 - a|

-- Proof Goal
theorem function_property : f (2 * a - x1) > f (2 * a - x2) :=
by
  sorry

end NUMINAMATH_GPT_function_property_l1621_162192


namespace NUMINAMATH_GPT_slipper_cost_l1621_162114

def original_price : ℝ := 50.00
def discount_rate : ℝ := 0.10
def embroidery_rate_per_shoe : ℝ := 5.50
def number_of_shoes : ℕ := 2
def shipping_cost : ℝ := 10.00

theorem slipper_cost :
  (original_price - original_price * discount_rate) + 
  (embroidery_rate_per_shoe * number_of_shoes) + 
  shipping_cost = 66.00 :=
by sorry

end NUMINAMATH_GPT_slipper_cost_l1621_162114


namespace NUMINAMATH_GPT_proof_volume_l1621_162131

noncomputable def volume_set (a b c h r : ℝ) : ℝ := 
  let v_box := a * b * c
  let v_extensions := 2 * (a * b * h) + 2 * (a * c * h) + 2 * (b * c * h)
  let v_cylinder := Real.pi * r^2 * h
  let v_spheres := 8 * (1/6) * (Real.pi * r^3)
  v_box + v_extensions + v_cylinder + v_spheres

theorem proof_volume : 
  let a := 2; let b := 3; let c := 6
  let r := 2; let h := 3
  volume_set a b c h r = (540 + 48 * Real.pi) / 3 ∧ (540 + 48 + 3) = 591 :=
by 
  sorry

end NUMINAMATH_GPT_proof_volume_l1621_162131


namespace NUMINAMATH_GPT_time_for_first_three_workers_l1621_162155

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end NUMINAMATH_GPT_time_for_first_three_workers_l1621_162155


namespace NUMINAMATH_GPT_frisbee_total_distance_correct_l1621_162170

-- Define the conditions
def bess_distance_per_throw : ℕ := 20 * 2 -- 20 meters out and 20 meters back = 40 meters
def bess_number_of_throws : ℕ := 4
def holly_distance_per_throw : ℕ := 8
def holly_number_of_throws : ℕ := 5

-- Calculate total distances
def bess_total_distance : ℕ := bess_distance_per_throw * bess_number_of_throws
def holly_total_distance : ℕ := holly_distance_per_throw * holly_number_of_throws
def total_distance : ℕ := bess_total_distance + holly_total_distance

-- The proof statement
theorem frisbee_total_distance_correct :
  total_distance = 200 :=
by
  -- proof goes here (we use sorry to skip the proof)
  sorry

end NUMINAMATH_GPT_frisbee_total_distance_correct_l1621_162170


namespace NUMINAMATH_GPT_rhombus_diagonals_sum_squares_l1621_162115

-- Definition of the rhombus side length condition
def is_rhombus_side_length (side_length : ℝ) : Prop :=
  side_length = 2

-- Lean 4 statement for the proof problem
theorem rhombus_diagonals_sum_squares (side_length : ℝ) (d1 d2 : ℝ) 
  (h : is_rhombus_side_length side_length) :
  side_length = 2 → (d1^2 + d2^2 = 16) :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonals_sum_squares_l1621_162115


namespace NUMINAMATH_GPT_solution_set_I_range_of_a_l1621_162148

-- Define the function f(x) = |x + a| - |x + 1|
def f (x a : ℝ) : ℝ := abs (x + a) - abs (x + 1)

-- Part (I)
theorem solution_set_I (a : ℝ) : 
  (f a a > 1) ↔ (a < -2/3 ∨ a > 2) := by
  sorry

-- Part (II)
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≤ 2 * a) ↔ (a ≥ 1/3) := by
  sorry

end NUMINAMATH_GPT_solution_set_I_range_of_a_l1621_162148


namespace NUMINAMATH_GPT_smallest_nat_divisible_by_225_l1621_162144

def has_digits_0_or_1 (n : ℕ) : Prop := 
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 1

def divisible_by_225 (n : ℕ) : Prop := 225 ∣ n

theorem smallest_nat_divisible_by_225 :
  ∃ (n : ℕ), has_digits_0_or_1 n ∧ divisible_by_225 n 
    ∧ ∀ (m : ℕ), has_digits_0_or_1 m ∧ divisible_by_225 m → n ≤ m 
    ∧ n = 11111111100 := 
  sorry

end NUMINAMATH_GPT_smallest_nat_divisible_by_225_l1621_162144


namespace NUMINAMATH_GPT_one_div_lt_one_div_of_gt_l1621_162193

theorem one_div_lt_one_div_of_gt {a b : ℝ} (hab : a > b) (hb0 : b > 0) : (1 / a) < (1 / b) :=
sorry

end NUMINAMATH_GPT_one_div_lt_one_div_of_gt_l1621_162193


namespace NUMINAMATH_GPT_find_k_l1621_162127

theorem find_k (k : ℝ) : -x^2 - (k + 10) * x - 8 = -(x - 2) * (x - 4) → k = -16 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_l1621_162127


namespace NUMINAMATH_GPT_problem_a_problem_b_l1621_162177

-- Problem a conditions and statement
def digit1a : Nat := 1
def digit2a : Nat := 4
def digit3a : Nat := 2
def digit4a : Nat := 8
def digit5a : Nat := 5

theorem problem_a : (digit1a * 100000 + digit2a * 10000 + digit3a * 1000 + digit4a * 100 + digit5a * 10 + 7) * 5 = 
                    7 * (digit1a * 100000 + digit2a * 10000 + digit3a * 1000 + digit4a * 100 + digit5a * 10 + 285) := by
  sorry

-- Problem b conditions and statement
def digit1b : Nat := 4
def digit2b : Nat := 2
def digit3b : Nat := 8
def digit4b : Nat := 5
def digit5b : Nat := 7

theorem problem_b : (1 * 100000 + digit1b * 10000 + digit2b * 1000 + digit3b * 100 + digit4b * 10 + digit5b) * 3 = 
                    (digit1b * 100000 + digit2b * 10000 + digit3b * 1000 + digit4b * 100 + digit5b * 10 + 1) := by
  sorry

end NUMINAMATH_GPT_problem_a_problem_b_l1621_162177


namespace NUMINAMATH_GPT_egg_price_l1621_162102

theorem egg_price (num_eggs capital_remaining : ℕ) (total_cost price_per_egg : ℝ)
  (h1 : num_eggs = 30)
  (h2 : capital_remaining = 5)
  (h3 : total_cost = 5)
  (h4 : num_eggs - capital_remaining = 25)
  (h5 : 25 * price_per_egg = total_cost) :
  price_per_egg = 0.20 := sorry

end NUMINAMATH_GPT_egg_price_l1621_162102


namespace NUMINAMATH_GPT_dylans_mom_hotdogs_l1621_162198

theorem dylans_mom_hotdogs (hotdogs_total : ℕ) (helens_mom_hotdogs : ℕ) (dylans_mom_hotdogs : ℕ) 
  (h1 : hotdogs_total = 480) (h2 : helens_mom_hotdogs = 101) (h3 : hotdogs_total = helens_mom_hotdogs + dylans_mom_hotdogs) :
dylans_mom_hotdogs = 379 :=
by
  sorry

end NUMINAMATH_GPT_dylans_mom_hotdogs_l1621_162198


namespace NUMINAMATH_GPT_hog_cat_problem_l1621_162188

theorem hog_cat_problem (hogs cats : ℕ)
  (hogs_eq : hogs = 75)
  (hogs_cats_relation : hogs = 3 * cats)
  : 5 < (6 / 10) * cats - 5 := 
by
  sorry

end NUMINAMATH_GPT_hog_cat_problem_l1621_162188


namespace NUMINAMATH_GPT_find_second_number_l1621_162182

theorem find_second_number (x y z : ℚ) (h_sum : x + y + z = 120)
  (h_ratio1 : x = (3 / 4) * y) (h_ratio2 : z = (7 / 4) * y) :
  y = 240 / 7 :=
by {
  -- Definitions provided from conditions
  sorry  -- Proof omitted
}

end NUMINAMATH_GPT_find_second_number_l1621_162182


namespace NUMINAMATH_GPT_area_of_square_field_l1621_162138

theorem area_of_square_field (s : ℕ) (A : ℕ) (cost_per_meter : ℕ) 
  (total_cost : ℕ) (gate_width : ℕ) (num_gates : ℕ) 
  (h1 : cost_per_meter = 1)
  (h2 : total_cost = 666)
  (h3 : gate_width = 1)
  (h4 : num_gates = 2)
  (h5 : (4 * s - num_gates * gate_width) * cost_per_meter = total_cost) :
  A = s * s → A = 27889 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_field_l1621_162138


namespace NUMINAMATH_GPT_two_abc_square_l1621_162189

variable {R : Type*} [Ring R] [Fintype R]

-- Given condition: For any a, b ∈ R, ∃ c ∈ R such that a^2 + b^2 = c^2.
axiom ring_property (a b : R) : ∃ c : R, a^2 + b^2 = c^2

-- We need to prove: For any a, b, c ∈ R, ∃ d ∈ R such that 2abc = d^2.
theorem two_abc_square (a b c : R) : ∃ d : R, 2 * (a * b * c) = d^2 :=
by
  sorry

end NUMINAMATH_GPT_two_abc_square_l1621_162189


namespace NUMINAMATH_GPT_collinear_points_min_value_l1621_162173

open Real

/-- Let \(\overrightarrow{e_{1}}\) and \(\overrightarrow{e_{2}}\) be two non-collinear vectors in a plane,
    \(\overrightarrow{AB} = (a-1) \overrightarrow{e_{1}} + \overrightarrow{e_{2}}\),
    \(\overrightarrow{AC} = b \overrightarrow{e_{1}} - 2 \overrightarrow{e_{2}}\),
    with \(a > 0\) and \(b > 0\). 
    If points \(A\), \(B\), and \(C\) are collinear, then the minimum value of \(\frac{1}{a} + \frac{2}{b}\) is \(4\). -/
theorem collinear_points_min_value 
  (e1 e2 : ℝ) 
  (H_non_collinear : (e1 ≠ 0 ∨ e2 ≠ 0))
  (a b : ℝ) 
  (H_a_pos : a > 0) 
  (H_b_pos : b > 0)
  (H_collinear : ∃ x : ℝ, (a - 1) * e1 + e2 = x * (b * e1 - 2 * e2)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + (1/2) * b = 1 ∧ (∀ a b : ℝ, (1/a) + (2/b) ≥ 4) :=
sorry

end NUMINAMATH_GPT_collinear_points_min_value_l1621_162173


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1621_162125

noncomputable def a₂ (a₁ q : ℝ) : ℝ := a₁ * q
noncomputable def a₃ (a₁ q : ℝ) : ℝ := a₁ * q^2
noncomputable def a₄ (a₁ q : ℝ) : ℝ := a₁ * q^3
noncomputable def S₆ (a₁ q : ℝ) : ℝ := (a₁ * (1 - q^6)) / (1 - q)

theorem geometric_sequence_problem
  (a₁ q : ℝ)
  (h1 : a₁ * a₂ a₁ q * a₃ a₁ q = 27)
  (h2 : a₂ a₁ q + a₄ a₁ q = 30)
  : ((a₁ = 1 ∧ q = 3) ∨ (a₁ = -1 ∧ q = -3))
    ∧ (if a₁ = 1 ∧ q = 3 then S₆ a₁ q = 364 else true)
    ∧ (if a₁ = -1 ∧ q = -3 then S₆ a₁ q = -182 else true) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1621_162125


namespace NUMINAMATH_GPT_InequalityProof_l1621_162146

theorem InequalityProof (m n : ℝ) (h : m > n) : m / 4 > n / 4 :=
by sorry

end NUMINAMATH_GPT_InequalityProof_l1621_162146


namespace NUMINAMATH_GPT_walnut_trees_initial_count_l1621_162150

theorem walnut_trees_initial_count (x : ℕ) (h : x + 6 = 10) : x = 4 := 
by
  sorry

end NUMINAMATH_GPT_walnut_trees_initial_count_l1621_162150


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_sin_2alpha_fraction_l1621_162163

-- Question 1 (Proving tan(alpha + pi/4) = -3 given tan(alpha) = 2)
theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
sorry

-- Question 2 (Proving the given fraction equals 1 given tan(alpha) = 2)
theorem sin_2alpha_fraction (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (2 * α) / 
   (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1)) = 1 :=
sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_sin_2alpha_fraction_l1621_162163


namespace NUMINAMATH_GPT_proportion_solution_l1621_162105

theorem proportion_solution (x : ℝ) (h : x / 6 = 4 / 0.39999999999999997) : x = 60 := sorry

end NUMINAMATH_GPT_proportion_solution_l1621_162105


namespace NUMINAMATH_GPT_find_b_for_square_binomial_l1621_162123

theorem find_b_for_square_binomial 
  (b : ℝ)
  (u t : ℝ)
  (h₁ : u^2 = 4)
  (h₂ : 2 * t * u = 8)
  (h₃ : b = t^2) : b = 4 := 
  sorry

end NUMINAMATH_GPT_find_b_for_square_binomial_l1621_162123


namespace NUMINAMATH_GPT_part1_part2_l1621_162154

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |2 * x - a| + |x - 1|) :
  (∀ x, f x + |x - 1| ≥ 2) → (a ≤ 0 ∨ a ≥ 4) :=
by sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (h_a : a < 2) (h_f : ∀ x, f x = |2 * x - a| + |x - 1|) :
  (∀ x, f x ≥ a - 1) → (a = 4 / 3) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1621_162154
