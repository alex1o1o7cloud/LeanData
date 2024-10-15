import Mathlib

namespace NUMINAMATH_GPT_rounding_example_l2291_229104

theorem rounding_example (x : ℝ) (h : x = 8899.50241201) : round x = 8900 :=
by
  sorry

end NUMINAMATH_GPT_rounding_example_l2291_229104


namespace NUMINAMATH_GPT_random_point_between_R_S_l2291_229102

theorem random_point_between_R_S {P Q R S : ℝ} (PQ PR RS : ℝ) (h1 : PQ = 4 * PR) (h2 : PQ = 8 * RS) :
  let PS := PR + RS
  let probability := RS / PQ
  probability = 5 / 8 :=
by
  let PS := PR + RS
  let probability := RS / PQ
  sorry

end NUMINAMATH_GPT_random_point_between_R_S_l2291_229102


namespace NUMINAMATH_GPT_temperature_on_tuesday_l2291_229180

variable (T W Th F : ℝ)

theorem temperature_on_tuesday :
  (T + W + Th = 156) ∧ (W + Th + 53 = 162) → T = 47 :=
by
  sorry

end NUMINAMATH_GPT_temperature_on_tuesday_l2291_229180


namespace NUMINAMATH_GPT_polyhedron_faces_l2291_229156

theorem polyhedron_faces (V E F T P t p : ℕ)
  (hF : F = 20)
  (hFaces : t + p = 20)
  (hTriangles : t = 2 * p)
  (hVertex : T = 2 ∧ P = 2)
  (hEdges : E = (3 * t + 5 * p) / 2)
  (hEuler : V - E + F = 2) :
  100 * P + 10 * T + V = 238 :=
by
  sorry

end NUMINAMATH_GPT_polyhedron_faces_l2291_229156


namespace NUMINAMATH_GPT_union_sets_l2291_229161

open Set

/-- Given sets A and B defined as follows:
    A = {x | -1 ≤ x ∧ x ≤ 2}
    B = {x | x ≤ 4}
    Prove that A ∪ B = {x | x ≤ 4}
--/
theorem union_sets  :
    let A := {x | -1 ≤ x ∧ x ≤ 2}
    let B := {x | x ≤ 4}
    A ∪ B = {x | x ≤ 4} :=
by
    intros A B
    have : A = {x | -1 ≤ x ∧ x ≤ 2} := rfl
    have : B = {x | x ≤ 4} := rfl
    sorry

end NUMINAMATH_GPT_union_sets_l2291_229161


namespace NUMINAMATH_GPT_explorers_crossing_time_l2291_229195

/-- Define constants and conditions --/
def num_explorers : ℕ := 60
def boat_capacity : ℕ := 6
def crossing_time : ℕ := 3
def round_trip_crossings : ℕ := 2
def total_trips := 1 + (num_explorers - boat_capacity - 1) / (boat_capacity - 1) + 1

theorem explorers_crossing_time :
  total_trips * crossing_time * round_trip_crossings / 2 + crossing_time = 69 :=
by sorry

end NUMINAMATH_GPT_explorers_crossing_time_l2291_229195


namespace NUMINAMATH_GPT_LCM_20_45_75_is_900_l2291_229152

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end NUMINAMATH_GPT_LCM_20_45_75_is_900_l2291_229152


namespace NUMINAMATH_GPT_find_a_l2291_229144

noncomputable def tangent_condition (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧ (1 : ℝ) = (1 / (x₀ + a))

theorem find_a : ∃ a : ℝ, tangent_condition a ∧ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2291_229144


namespace NUMINAMATH_GPT_total_number_of_athletes_l2291_229174

theorem total_number_of_athletes (M F x : ℕ) (r1 r2 r3 : ℕ×ℕ) (H1 : r1 = (19, 12)) (H2 : r2 = (20, 13)) (H3 : r3 = (30, 19))
  (initial_males : M = 380 * x) (initial_females : F = 240 * x)
  (males_after_gym : M' = 390 * x) (females_after_gym : F' = 247 * x)
  (conditions : (M' - M) - (F' - F) = 30) : M' + F' = 6370 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_athletes_l2291_229174


namespace NUMINAMATH_GPT_non_gray_squares_count_l2291_229176

-- Define the dimensions of the grid strip
def width : ℕ := 5
def length : ℕ := 250

-- Define the repeating pattern dimensions and color distribution
def pattern_columns : ℕ := 4
def pattern_non_gray_squares : ℕ := 13
def pattern_total_squares : ℕ := width * pattern_columns

-- Define the number of complete patterns in the grid strip
def complete_patterns : ℕ := length / pattern_columns

-- Define the number of additional columns and additional non-gray squares
def additional_columns : ℕ := length % pattern_columns
def additional_non_gray_squares : ℕ := 6

-- Calculate the total non-gray squares
def total_non_gray_squares : ℕ := complete_patterns * pattern_non_gray_squares + additional_non_gray_squares

theorem non_gray_squares_count : total_non_gray_squares = 812 := by
  sorry

end NUMINAMATH_GPT_non_gray_squares_count_l2291_229176


namespace NUMINAMATH_GPT_decreasing_functions_l2291_229199

noncomputable def f1 (x : ℝ) : ℝ := -x^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def f4 (x : ℝ) : ℝ := 3 ^ x

theorem decreasing_functions :
  (∀ x y : ℝ, 0 < x → x < y → f1 y < f1 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f2 y > f2 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f3 y > f3 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f4 y > f4 x) :=
by {
  sorry
}

end NUMINAMATH_GPT_decreasing_functions_l2291_229199


namespace NUMINAMATH_GPT_max_a_value_l2291_229175

noncomputable def f (x k a : ℝ) : ℝ := x^2 - (k^2 - 5 * a * k + 3) * x + 7

theorem max_a_value : ∀ (k a : ℝ), (0 <= k) → (k <= 2) →
  (∀ (x1 : ℝ), (k <= x1) → (x1 <= k + a) →
  ∀ (x2 : ℝ), (k + 2 * a <= x2) → (x2 <= k + 4 * a) →
  f x1 k a >= f x2 k a) → 
  a <= (2 * Real.sqrt 6 - 4) / 5 := 
sorry

end NUMINAMATH_GPT_max_a_value_l2291_229175


namespace NUMINAMATH_GPT_age_of_teacher_l2291_229120

theorem age_of_teacher (avg_age_students : ℕ) (num_students : ℕ) (inc_avg_with_teacher : ℕ) (num_people_with_teacher : ℕ) :
  avg_age_students = 21 →
  num_students = 20 →
  inc_avg_with_teacher = 22 →
  num_people_with_teacher = 21 →
  let total_age_students := num_students * avg_age_students
  let total_age_with_teacher := num_people_with_teacher * inc_avg_with_teacher
  total_age_with_teacher - total_age_students = 42 :=
by
  intros
  sorry

end NUMINAMATH_GPT_age_of_teacher_l2291_229120


namespace NUMINAMATH_GPT_distance_correct_l2291_229150

-- Define geometry entities and properties
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Sphere where
  center : Point
  radius : ℝ

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define conditions
def sphere_center : Point := { x := 0, y := 0, z := 0 }
def sphere : Sphere := { center := sphere_center, radius := 5 }
def triangle : Triangle := { a := 13, b := 13, c := 10 }

-- Define the distance calculation
noncomputable def distance_from_sphere_center_to_plane (O : Point) (T : Triangle) : ℝ :=
  let h := 12  -- height calculation based on given triangle sides
  let A := 60  -- area of the triangle
  let s := 18  -- semiperimeter
  let r := 10 / 3  -- inradius calculation
  let x := 5 * (Real.sqrt 5) / 3  -- final distance calculation
  x

-- Prove the obtained distance matches expected value
theorem distance_correct :
  distance_from_sphere_center_to_plane sphere_center triangle = 5 * (Real.sqrt 5) / 3 :=
by
  sorry

end NUMINAMATH_GPT_distance_correct_l2291_229150


namespace NUMINAMATH_GPT_sine_tangent_coincide_3_decimal_places_l2291_229137

open Real

noncomputable def deg_to_rad (d : ℝ) : ℝ := d * (π / 180)

theorem sine_tangent_coincide_3_decimal_places :
  ∀ θ : ℝ,
    0 ≤ θ ∧ θ ≤ deg_to_rad (4 + 20 / 60) →
    |sin θ - tan θ| < 0.0005 :=
by
  intros θ hθ
  sorry

end NUMINAMATH_GPT_sine_tangent_coincide_3_decimal_places_l2291_229137


namespace NUMINAMATH_GPT_evaluate_expression_c_eq_4_l2291_229149

theorem evaluate_expression_c_eq_4 :
  (4^4 - 4 * (4-1)^(4-1))^(4-1) = 3241792 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_c_eq_4_l2291_229149


namespace NUMINAMATH_GPT_circle_equation_unique_circle_equation_l2291_229183

-- Definitions based on conditions
def radius (r : ℝ) : Prop := r = 1
def center_in_first_quadrant (a b : ℝ) : Prop := a > 0 ∧ b > 0
def tangent_to_line (a b : ℝ) : Prop := (|4 * a - 3 * b| / Real.sqrt (4^2 + (-3)^2)) = 1
def tangent_to_x_axis (b : ℝ) : Prop := b = 1

-- Main theorem statement
theorem circle_equation_unique 
  {a b : ℝ} 
  (h_rad : radius 1) 
  (h_center : center_in_first_quadrant a b) 
  (h_tan_line : tangent_to_line a b) 
  (h_tan_x : tangent_to_x_axis b) :
  (a = 2 ∧ b = 1) :=
sorry

-- Final circle equation
theorem circle_equation : 
  (∀ a b : ℝ, ((a = 2) ∧ (b = 1)) → (x - a)^2 + (y - b)^2 = 1) :=
sorry

end NUMINAMATH_GPT_circle_equation_unique_circle_equation_l2291_229183


namespace NUMINAMATH_GPT_child_to_grandmother_ratio_l2291_229178

variable (G D C : ℝ)

axiom condition1 : G + D + C = 150
axiom condition2 : D + C = 60
axiom condition3 : D = 42

theorem child_to_grandmother_ratio : (C / G) = (1 / 5) :=
by
  sorry

end NUMINAMATH_GPT_child_to_grandmother_ratio_l2291_229178


namespace NUMINAMATH_GPT_max_gold_coins_l2291_229108

-- Define the conditions as predicates
def divides_with_remainder (n : ℕ) (d r : ℕ) : Prop := n % d = r
def less_than (n k : ℕ) : Prop := n < k

-- Main statement incorporating the conditions and the conclusion
theorem max_gold_coins (n : ℕ) :
  divides_with_remainder n 15 3 ∧ less_than n 120 → n ≤ 105 :=
by
  sorry

end NUMINAMATH_GPT_max_gold_coins_l2291_229108


namespace NUMINAMATH_GPT_area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3_l2291_229143

noncomputable def area_enclosed_by_sine_and_line : ℝ :=
  (∫ x in (Real.pi / 6)..(5 * Real.pi / 6), (Real.sin x - 1 / 2))

theorem area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3 :
  area_enclosed_by_sine_and_line = Real.sqrt 3 - Real.pi / 3 := by
  sorry

end NUMINAMATH_GPT_area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3_l2291_229143


namespace NUMINAMATH_GPT_reflection_eqn_l2291_229173

theorem reflection_eqn 
  (x y : ℝ)
  (h : y = 2 * x + 3) : 
  -y = 2 * x + 3 :=
sorry

end NUMINAMATH_GPT_reflection_eqn_l2291_229173


namespace NUMINAMATH_GPT_fraction_is_percent_of_y_l2291_229163

theorem fraction_is_percent_of_y (y : ℝ) (hy : y > 0) : 
  (2 * y / 5 + 3 * y / 10) / y = 0.7 :=
sorry

end NUMINAMATH_GPT_fraction_is_percent_of_y_l2291_229163


namespace NUMINAMATH_GPT_average_percentage_l2291_229168

theorem average_percentage (num_students1 num_students2 : Nat) (avg1 avg2 avg : Nat) :
  num_students1 = 15 ->
  avg1 = 73 ->
  num_students2 = 10 ->
  avg2 = 88 ->
  (num_students1 * avg1 + num_students2 * avg2) / (num_students1 + num_students2) = avg ->
  avg = 79 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_average_percentage_l2291_229168


namespace NUMINAMATH_GPT_oliver_bags_fraction_l2291_229132

theorem oliver_bags_fraction
  (weight_james_bag : ℝ)
  (combined_weight_oliver_bags : ℝ)
  (h1 : weight_james_bag = 18)
  (h2 : combined_weight_oliver_bags = 6)
  (f : ℝ) :
  2 * f * weight_james_bag = combined_weight_oliver_bags → f = 1 / 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_oliver_bags_fraction_l2291_229132


namespace NUMINAMATH_GPT_tech_gadgets_components_total_l2291_229115

theorem tech_gadgets_components_total (a₁ r n : ℕ) (h₁ : a₁ = 8) (h₂ : r = 3) (h₃ : n = 4) :
  a₁ * (r^n - 1) / (r - 1) = 320 := by
  sorry

end NUMINAMATH_GPT_tech_gadgets_components_total_l2291_229115


namespace NUMINAMATH_GPT_totalNumberOfPeople_l2291_229123

def numGirls := 542
def numBoys := 387
def numTeachers := 45
def numStaff := 27

theorem totalNumberOfPeople : numGirls + numBoys + numTeachers + numStaff = 1001 := by
  sorry

end NUMINAMATH_GPT_totalNumberOfPeople_l2291_229123


namespace NUMINAMATH_GPT_average_of_roots_l2291_229190

theorem average_of_roots (p q : ℝ) (h : ∀ r : ℝ, r^2 * (3 * p) + r * (-6 * p) + q = 0 → ∃ a b : ℝ, r = a ∨ r = b) : 
  ∀ (r1 r2 : ℝ), (3 * p) * r1^2 + (-6 * p) * r1 + q = 0 ∧ (3 * p) * r2^2 + (-6 * p) * r2 + q = 0 → 
  (r1 + r2) / 2 = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_of_roots_l2291_229190


namespace NUMINAMATH_GPT_factorization_correct_l2291_229116

theorem factorization_correct (a b : ℝ) : 
  a^2 + 2 * b - b^2 - 1 = (a - b + 1) * (a + b - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l2291_229116


namespace NUMINAMATH_GPT_ratio_difference_l2291_229128

theorem ratio_difference (x : ℕ) (h : (2 * x + 4) * 7 = (3 * x + 4) * 5) : 3 * x - 2 * x = 8 := 
by sorry

end NUMINAMATH_GPT_ratio_difference_l2291_229128


namespace NUMINAMATH_GPT_vector_perpendicular_solve_x_l2291_229169

theorem vector_perpendicular_solve_x
  (x : ℝ)
  (a : ℝ × ℝ := (4, 8))
  (b : ℝ × ℝ := (x, 4))
  (h : 4 * x + 8 * 4 = 0) :
  x = -8 :=
sorry

end NUMINAMATH_GPT_vector_perpendicular_solve_x_l2291_229169


namespace NUMINAMATH_GPT_math_problem_l2291_229151

theorem math_problem :
  (-1 : ℤ) ^ 49 + 2 ^ (4 ^ 3 + 3 ^ 2 - 7 ^ 2) = 16777215 := by
  sorry

end NUMINAMATH_GPT_math_problem_l2291_229151


namespace NUMINAMATH_GPT_number_of_students_suggested_mashed_potatoes_l2291_229188

theorem number_of_students_suggested_mashed_potatoes 
    (students_suggested_bacon : ℕ := 374) 
    (students_suggested_tomatoes : ℕ := 128) 
    (total_students_participated : ℕ := 826) : 
    (total_students_participated - (students_suggested_bacon + students_suggested_tomatoes)) = 324 :=
by sorry

end NUMINAMATH_GPT_number_of_students_suggested_mashed_potatoes_l2291_229188


namespace NUMINAMATH_GPT_shelby_scooter_drive_l2291_229110

/-- 
Let y be the time (in minutes) Shelby drove when it was not raining.
Speed when not raining is 25 miles per hour, which is 5/12 mile per minute.
Speed when raining is 15 miles per hour, which is 1/4 mile per minute.
Total distance covered is 18 miles.
Total time taken is 36 minutes.
Prove that Shelby drove for 6 minutes when it was not raining.
-/
theorem shelby_scooter_drive
  (y : ℝ)
  (h_not_raining_speed : ∀ t (h : t = (25/60 : ℝ)), t = (5/12 : ℝ))
  (h_raining_speed : ∀ t (h : t = (15/60 : ℝ)), t = (1/4 : ℝ))
  (h_total_distance : ∀ d (h : d = ((5/12 : ℝ) * y + (1/4 : ℝ) * (36 - y))), d = 18)
  (h_total_time : ∀ t (h : t = 36), t = 36) :
  y = 6 :=
sorry

end NUMINAMATH_GPT_shelby_scooter_drive_l2291_229110


namespace NUMINAMATH_GPT_probability_all_truth_l2291_229106

noncomputable def probability_A : ℝ := 0.55
noncomputable def probability_B : ℝ := 0.60
noncomputable def probability_C : ℝ := 0.45
noncomputable def probability_D : ℝ := 0.70

theorem probability_all_truth : 
  (probability_A * probability_B * probability_C * probability_D = 0.10395) := 
by 
  sorry

end NUMINAMATH_GPT_probability_all_truth_l2291_229106


namespace NUMINAMATH_GPT_decrease_A_share_l2291_229107

theorem decrease_A_share :
  ∃ (a b x : ℝ),
    a + b + 495 = 1010 ∧
    (a - x) / 3 = 96 ∧
    (b - 10) / 2 = 96 ∧
    x = 25 :=
by
  sorry

end NUMINAMATH_GPT_decrease_A_share_l2291_229107


namespace NUMINAMATH_GPT_impossible_to_get_100_pieces_l2291_229136

/-- We start with 1 piece of paper. Each time a piece of paper is torn into 3 parts,
it increases the total number of pieces by 2.
Therefore, the number of pieces remains odd through any sequence of tears.
Prove that it is impossible to obtain exactly 100 pieces. -/
theorem impossible_to_get_100_pieces : 
  ∀ n, n = 1 ∨ (∃ k, n = 1 + 2 * k) → n ≠ 100 :=
by
  sorry

end NUMINAMATH_GPT_impossible_to_get_100_pieces_l2291_229136


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2291_229147

open Set

variable (A : Set ℕ) (B : Set ℕ)

theorem intersection_of_A_and_B (hA : A = {0, 1, 2}) (hB : B = {0, 2, 4}) :
  A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2291_229147


namespace NUMINAMATH_GPT_principal_amount_simple_interest_l2291_229197

theorem principal_amount_simple_interest 
    (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ)
    (hR : R = 4)
    (hT : T = 5)
    (hSI : SI = P - 2080)
    (hInterestFormula : SI = (P * R * T) / 100) :
    P = 2600 := 
by
  sorry

end NUMINAMATH_GPT_principal_amount_simple_interest_l2291_229197


namespace NUMINAMATH_GPT_probability_of_winning_pair_l2291_229129

/--
A deck consists of five red cards and five green cards, with each color having cards labeled from A to E. 
Two cards are drawn from this deck.
A winning pair is defined as two cards of the same color or two cards of the same letter. 
Prove that the probability of drawing a winning pair is 5/9.
-/
theorem probability_of_winning_pair :
  let total_cards := 10
  let total_ways := Nat.choose total_cards 2
  let same_letter_ways := 5
  let same_color_red_ways := Nat.choose 5 2
  let same_color_green_ways := Nat.choose 5 2
  let same_color_ways := same_color_red_ways + same_color_green_ways
  let favorable_outcomes := same_letter_ways + same_color_ways
  favorable_outcomes / total_ways = 5 / 9 := by
  sorry

end NUMINAMATH_GPT_probability_of_winning_pair_l2291_229129


namespace NUMINAMATH_GPT_certain_number_condition_l2291_229133

theorem certain_number_condition (x y z : ℤ) (N : ℤ)
  (hx : Even x) (hy : Odd y) (hz : Odd z)
  (hxy : x < y) (hyz : y < z)
  (h1 : y - x > N)
  (h2 : z - x = 7) :
  N < 3 := by
  sorry

end NUMINAMATH_GPT_certain_number_condition_l2291_229133


namespace NUMINAMATH_GPT_Katie_old_games_l2291_229198

theorem Katie_old_games (O : ℕ) (hk1 : Katie_new_games = 57) (hf1 : Friends_new_games = 34) (hk2 : Katie_total_games = Friends_total_games + 62) : 
  O = 39 :=
by
  sorry

variables (Katie_new_games Friends_new_games Katie_total_games Friends_total_games : ℕ)

end NUMINAMATH_GPT_Katie_old_games_l2291_229198


namespace NUMINAMATH_GPT_sum_of_abs_coeffs_in_binomial_expansion_l2291_229154

theorem sum_of_abs_coeffs_in_binomial_expansion :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ), 
  (3 * x - 1) ^ 7 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5 + a₆ * x ^ 6 + a₇ * x ^ 7
  → |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4 ^ 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_abs_coeffs_in_binomial_expansion_l2291_229154


namespace NUMINAMATH_GPT_mean_of_five_numbers_is_correct_l2291_229146

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_mean_of_five_numbers_is_correct_l2291_229146


namespace NUMINAMATH_GPT_magnification_factor_is_correct_l2291_229140

theorem magnification_factor_is_correct
    (diameter_magnified_image : ℝ)
    (actual_diameter_tissue : ℝ)
    (diameter_magnified_image_eq : diameter_magnified_image = 2)
    (actual_diameter_tissue_eq : actual_diameter_tissue = 0.002) :
  diameter_magnified_image / actual_diameter_tissue = 1000 := by
  -- Theorem and goal statement
  sorry

end NUMINAMATH_GPT_magnification_factor_is_correct_l2291_229140


namespace NUMINAMATH_GPT_maryann_time_spent_calling_clients_l2291_229184

theorem maryann_time_spent_calling_clients (a c : ℕ) 
  (h1 : a + c = 560) 
  (h2 : a = 7 * c) : c = 70 := 
by 
  sorry

end NUMINAMATH_GPT_maryann_time_spent_calling_clients_l2291_229184


namespace NUMINAMATH_GPT_select_female_athletes_l2291_229114

theorem select_female_athletes (males females sample_size total_size : ℕ)
    (h1 : males = 56) (h2 : females = 42) (h3 : sample_size = 28)
    (h4 : total_size = males + females) : 
    (females * sample_size / total_size = 12) := 
by
  sorry

end NUMINAMATH_GPT_select_female_athletes_l2291_229114


namespace NUMINAMATH_GPT_cole_cost_l2291_229167

def length_of_sides := 15
def length_of_back := 30
def cost_per_foot_side := 4
def cost_per_foot_back := 5
def cole_installation_fee := 50

def neighbor_behind_contribution := (length_of_back * cost_per_foot_back) / 2
def neighbor_left_contribution := (length_of_sides * cost_per_foot_side) / 3

def total_cost := 
  2 * length_of_sides * cost_per_foot_side + 
  length_of_back * cost_per_foot_back

def cole_contribution := 
  total_cost - neighbor_behind_contribution - neighbor_left_contribution + cole_installation_fee

theorem cole_cost (h : cole_contribution = 225) : cole_contribution = 225 := by
  sorry

end NUMINAMATH_GPT_cole_cost_l2291_229167


namespace NUMINAMATH_GPT_correct_statement_l2291_229124

section
variables {a b c d : Real}

-- Define the conditions as hypotheses/functions

-- Statement A: If a > b, then 1/a < 1/b
def statement_A (a b : Real) : Prop := a > b → 1 / a < 1 / b

-- Statement B: If a > b, then a^2 > b^2
def statement_B (a b : Real) : Prop := a > b → a^2 > b^2

-- Statement C: If a > b and c > d, then ac > bd
def statement_C (a b c d : Real) : Prop := a > b ∧ c > d → a * c > b * d

-- Statement D: If a^3 > b^3, then a > b
def statement_D (a b : Real) : Prop := a^3 > b^3 → a > b

-- The Lean statement to prove which statement is correct
theorem correct_statement : ¬ statement_A a b ∧ ¬ statement_B a b ∧ ¬ statement_C a b c d ∧ statement_D a b :=
by {
  sorry
}

end

end NUMINAMATH_GPT_correct_statement_l2291_229124


namespace NUMINAMATH_GPT_cos_B_plus_C_value_of_c_l2291_229192

variable {A B C a b c : ℝ}

-- Given conditions
axiom a_eq_2b : a = 2 * b
axiom sine_arithmetic_sequence : 2 * Real.sin C = Real.sin A + Real.sin B

-- First proof
theorem cos_B_plus_C (h : a = 2 * b) (h_seq : 2 * Real.sin C = Real.sin A + Real.sin B) :
  Real.cos (B + C) = 1 / 4 := 
sorry

-- Given additional condition for the area
axiom area_eq : (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 15) / 3

-- Second proof
theorem value_of_c (h : a = 2 * b) (h_seq : 2 * Real.sin C = Real.sin A + Real.sin B) (h_area : (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 15) / 3) :
  c = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_cos_B_plus_C_value_of_c_l2291_229192


namespace NUMINAMATH_GPT_sours_total_l2291_229141

variable (c l o T : ℕ)

axiom cherry_sours : c = 32
axiom ratio_cherry_lemon : 4 * l = 5 * c
axiom orange_sours_ratio : o = 25 * T / 100
axiom total_sours : T = c + l + o

theorem sours_total :
  T = 96 :=
by
  sorry

end NUMINAMATH_GPT_sours_total_l2291_229141


namespace NUMINAMATH_GPT_satisfactory_grades_fraction_l2291_229165

def total_satisfactory_students (gA gB gC gD gE : Nat) : Nat :=
  gA + gB + gC + gD + gE

def total_students (gA gB gC gD gE gF : Nat) : Nat :=
  total_satisfactory_students gA gB gC gD gE + gF

def satisfactory_fraction (gA gB gC gD gE gF : Nat) : Rat :=
  total_satisfactory_students gA gB gC gD gE / total_students gA gB gC gD gE gF

theorem satisfactory_grades_fraction :
  satisfactory_fraction 3 5 4 2 1 4 = (15 : Rat) / 19 :=
by
  sorry

end NUMINAMATH_GPT_satisfactory_grades_fraction_l2291_229165


namespace NUMINAMATH_GPT_wilsons_theorem_l2291_229118

theorem wilsons_theorem (p : ℕ) (hp : Nat.Prime p) : (Nat.factorial (p - 1)) % p = p - 1 :=
by
  sorry

end NUMINAMATH_GPT_wilsons_theorem_l2291_229118


namespace NUMINAMATH_GPT_tracy_initial_candies_l2291_229138

theorem tracy_initial_candies (y : ℕ) 
  (condition1 : y - y / 4 = y * 3 / 4)
  (condition2 : y * 3 / 4 - (y * 3 / 4) / 3 = y / 2)
  (condition3 : y / 2 - 24 = y / 2 - 12 - 12)
  (condition4 : y / 2 - 24 - 4 = 2) : 
  y = 60 :=
by sorry

end NUMINAMATH_GPT_tracy_initial_candies_l2291_229138


namespace NUMINAMATH_GPT_product_of_six_consecutive_nat_not_equal_776965920_l2291_229164

theorem product_of_six_consecutive_nat_not_equal_776965920 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ≠ 776965920) :=
by
  sorry

end NUMINAMATH_GPT_product_of_six_consecutive_nat_not_equal_776965920_l2291_229164


namespace NUMINAMATH_GPT_right_triangle_angles_l2291_229126

theorem right_triangle_angles (a b S : ℝ) (hS : S = 1 / 2 * a * b) (h : (a + b) ^ 2 = 8 * S) :
  ∃ θ₁ θ₂ θ₃ : ℝ, θ₁ = 45 ∧ θ₂ = 45 ∧ θ₃ = 90 :=
by {
  sorry
}

end NUMINAMATH_GPT_right_triangle_angles_l2291_229126


namespace NUMINAMATH_GPT_prove_midpoint_trajectory_eq_l2291_229193

noncomputable def midpoint_trajectory_eq {x y : ℝ} (h : ∃ (x_P y_P : ℝ), (x_P^2 - y_P^2 = 1) ∧ (x = x_P / 2) ∧ (y = y_P / 2)) : Prop :=
  4*x^2 - 4*y^2 = 1

theorem prove_midpoint_trajectory_eq (x y : ℝ) (h : ∃ (x_P y_P : ℝ), (x_P^2 - y_P^2 = 1) ∧ (x = x_P / 2) ∧ (y = y_P / 2)) :
  midpoint_trajectory_eq h :=
sorry

end NUMINAMATH_GPT_prove_midpoint_trajectory_eq_l2291_229193


namespace NUMINAMATH_GPT_value_of_y_at_x_3_l2291_229112

theorem value_of_y_at_x_3 (a b c : ℝ) (h : a * (-3 : ℝ)^5 + b * (-3)^3 + c * (-3) - 5 = 7) :
  a * (3 : ℝ)^5 + b * 3^3 + c * 3 - 5 = -17 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_at_x_3_l2291_229112


namespace NUMINAMATH_GPT_white_truck_percentage_is_17_l2291_229134

-- Define the conditions
def total_trucks : ℕ := 50
def total_cars : ℕ := 40
def total_vehicles : ℕ := total_trucks + total_cars

def red_trucks : ℕ := total_trucks / 2
def black_trucks : ℕ := (total_trucks * 20) / 100
def white_trucks : ℕ := total_trucks - red_trucks - black_trucks

def percentage_white_trucks : ℕ := (white_trucks * 100) / total_vehicles

theorem white_truck_percentage_is_17 :
  percentage_white_trucks = 17 :=
  by sorry

end NUMINAMATH_GPT_white_truck_percentage_is_17_l2291_229134


namespace NUMINAMATH_GPT_min_max_values_on_interval_l2291_229105

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) + (x + 1)*(Real.sin x) + 1

theorem min_max_values_on_interval :
  (∀ x ∈ Set.Icc 0 (2*Real.pi), f x ≥ -(3*Real.pi/2) ∧ f x ≤ (Real.pi/2 + 2)) ∧
  ( ∃ a ∈ Set.Icc 0 (2*Real.pi), f a = -(3*Real.pi/2) ) ∧
  ( ∃ b ∈ Set.Icc 0 (2*Real.pi), f b = (Real.pi/2 + 2) ) :=
by
  sorry

end NUMINAMATH_GPT_min_max_values_on_interval_l2291_229105


namespace NUMINAMATH_GPT_log_simplify_l2291_229191

open Real

theorem log_simplify : 
  (1 / (log 12 / log 3 + 1)) + 
  (1 / (log 8 / log 2 + 1)) + 
  (1 / (log 30 / log 5 + 1)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_log_simplify_l2291_229191


namespace NUMINAMATH_GPT_fraction_walk_home_l2291_229142

theorem fraction_walk_home : 
  (1 - ((1 / 2) + (1 / 4) + (1 / 10) + (1 / 8))) = (1 / 40) :=
by 
  sorry

end NUMINAMATH_GPT_fraction_walk_home_l2291_229142


namespace NUMINAMATH_GPT_value_at_4_value_of_x_when_y_is_0_l2291_229125

-- Problem statement
def f (x : ℝ) : ℝ := 2 * x - 3

-- Proof statement 1: When x = 4, y = 5
theorem value_at_4 : f 4 = 5 := sorry

-- Proof statement 2: When y = 0, x = 3/2
theorem value_of_x_when_y_is_0 : (∃ x : ℝ, f x = 0) → (∃ x : ℝ, x = 3 / 2) := sorry

end NUMINAMATH_GPT_value_at_4_value_of_x_when_y_is_0_l2291_229125


namespace NUMINAMATH_GPT_probability_both_truth_l2291_229181

noncomputable def probability_A_truth : ℝ := 0.75
noncomputable def probability_B_truth : ℝ := 0.60

theorem probability_both_truth : 
  (probability_A_truth * probability_B_truth) = 0.45 :=
by sorry

end NUMINAMATH_GPT_probability_both_truth_l2291_229181


namespace NUMINAMATH_GPT_larger_number_ratio_l2291_229103

theorem larger_number_ratio (x : ℕ) (a b : ℕ) (h1 : a = 3 * x) (h2 : b = 8 * x) 
(h3 : (a - 24) * 9 = (b - 24) * 4) : b = 192 :=
sorry

end NUMINAMATH_GPT_larger_number_ratio_l2291_229103


namespace NUMINAMATH_GPT_opponent_score_value_l2291_229155

-- Define the given conditions
def total_points : ℕ := 720
def games_played : ℕ := 24
def average_score := total_points / games_played
def championship_score := average_score / 2 - 2
def opponent_score := championship_score + 2

-- Lean theorem statement to prove
theorem opponent_score_value : opponent_score = 15 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_opponent_score_value_l2291_229155


namespace NUMINAMATH_GPT_real_roots_quadratic_range_l2291_229158

theorem real_roots_quadratic_range (k : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x - k = 0) ↔ k ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_real_roots_quadratic_range_l2291_229158


namespace NUMINAMATH_GPT_uncle_kahn_total_cost_l2291_229111

noncomputable def base_price : ℝ := 10
noncomputable def child_discount : ℝ := 0.3
noncomputable def senior_discount : ℝ := 0.1
noncomputable def handling_fee : ℝ := 5
noncomputable def discounted_senior_ticket_price : ℝ := 14
noncomputable def num_child_tickets : ℝ := 2
noncomputable def num_senior_tickets : ℝ := 2

theorem uncle_kahn_total_cost :
  let child_ticket_cost := (1 - child_discount) * base_price + handling_fee
  let senior_ticket_cost := discounted_senior_ticket_price
  num_child_tickets * child_ticket_cost + num_senior_tickets * senior_ticket_cost = 52 :=
by
  sorry

end NUMINAMATH_GPT_uncle_kahn_total_cost_l2291_229111


namespace NUMINAMATH_GPT_solve_for_y_l2291_229179

-- Define the conditions as Lean functions and statements
def is_positive (y : ℕ) : Prop := y > 0
def multiply_sixteen (y : ℕ) : Prop := 16 * y = 256

-- The theorem that states the value of y
theorem solve_for_y (y : ℕ) (h1 : is_positive y) (h2 : multiply_sixteen y) : y = 16 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l2291_229179


namespace NUMINAMATH_GPT_number_of_players_knight_moves_friend_not_winner_l2291_229196

-- Problem (a)
theorem number_of_players (sum_scores : ℕ) (h : sum_scores = 210) : 
  ∃ x : ℕ, x * (x - 1) = 210 :=
sorry

-- Problem (b)
theorem knight_moves (initial_positions : ℕ) (wrong_guess : ℕ) (correct_answer : ℕ) : 
  initial_positions = 1 ∧ wrong_guess = 64 ∧ correct_answer = 33 → 
  ∃ squares : ℕ, squares = 33 :=
sorry

-- Problem (c)
theorem friend_not_winner (total_scores : ℕ) (num_players : ℕ) (friend_score : ℕ) (avg_score : ℕ) : 
  total_scores = 210 ∧ num_players = 15 ∧ friend_score = 12 ∧ avg_score = 14 → 
  ∃ higher_score : ℕ, higher_score > friend_score :=
sorry

end NUMINAMATH_GPT_number_of_players_knight_moves_friend_not_winner_l2291_229196


namespace NUMINAMATH_GPT_ellipse_reflection_symmetry_l2291_229159

theorem ellipse_reflection_symmetry :
  (∀ x y, (x = -y ∧ y = -x) →
  (∀ a b : ℝ, 
    (a - 3)^2 / 9 + (b - 2)^2 / 4 = 1 ↔
    (b - 3)^2 / 4 + (a - 2)^2 / 9 = 1)
  )
  →
  (∀ x y, 
    ((x + 2)^2 / 9 + (y + 3)^2 / 4 = 1) = 
    (∃ a b : ℝ, 
      (a - 3)^2 / 9 + (b - 2)^2 / 4 = 1 ∧ 
      (a = -y ∧ b = -x))
  ) :=
by
  intros
  sorry

end NUMINAMATH_GPT_ellipse_reflection_symmetry_l2291_229159


namespace NUMINAMATH_GPT_friend_selling_price_correct_l2291_229166

-- Definition of the original cost price
def original_cost_price : ℕ := 50000

-- Definition of the loss percentage
def loss_percentage : ℕ := 10

-- Definition of the gain percentage
def gain_percentage : ℕ := 20

-- Definition of the man's selling price after loss
def man_selling_price : ℕ := original_cost_price - (original_cost_price * loss_percentage / 100)

-- Definition of the friend's selling price after gain
def friend_selling_price : ℕ := man_selling_price + (man_selling_price * gain_percentage / 100)

theorem friend_selling_price_correct : friend_selling_price = 54000 := by
  sorry

end NUMINAMATH_GPT_friend_selling_price_correct_l2291_229166


namespace NUMINAMATH_GPT_find_x_l2291_229177

def a : ℝ × ℝ := (-2, 0)
def b : ℝ × ℝ := (2, 1)
def c (x : ℝ) : ℝ × ℝ := (x, -1)
def scalar_multiply (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def collinear (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_x :
  ∃ x : ℝ, collinear (vector_add (scalar_multiply 3 a) b) (c x) ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2291_229177


namespace NUMINAMATH_GPT_no_positive_integer_solutions_m2_m3_positive_integer_solutions_m4_l2291_229127

theorem no_positive_integer_solutions_m2_m3 (x y z t : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  (∃ m, m = 2 ∨ m = 3 → (x / y + y / z + z / t + t / x = m) → false) :=
sorry

theorem positive_integer_solutions_m4 (x y z t : ℕ) :
  x / y + y / z + z / t + t / x = 4 ↔ ∃ k : ℕ, k > 0 ∧ (x = k ∧ y = k ∧ z = k ∧ t = k) :=
sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_m2_m3_positive_integer_solutions_m4_l2291_229127


namespace NUMINAMATH_GPT_percentage_of_girl_scouts_with_slips_l2291_229119

-- Define the proposition that captures the problem
theorem percentage_of_girl_scouts_with_slips 
    (total_scouts : ℕ)
    (scouts_with_slips : ℕ := total_scouts * 60 / 100)
    (boy_scouts : ℕ := total_scouts * 45 / 100)
    (boy_scouts_with_slips : ℕ := boy_scouts * 50 / 100)
    (girl_scouts : ℕ := total_scouts - boy_scouts)
    (girl_scouts_with_slips : ℕ := scouts_with_slips - boy_scouts_with_slips) :
  (girl_scouts_with_slips * 100 / girl_scouts) = 68 :=
by 
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_percentage_of_girl_scouts_with_slips_l2291_229119


namespace NUMINAMATH_GPT_percent_formula_l2291_229117

theorem percent_formula (x y p : ℝ) (h : x = (p / 100) * y) : p = 100 * x / y :=
by
    sorry

end NUMINAMATH_GPT_percent_formula_l2291_229117


namespace NUMINAMATH_GPT_no_valid_n_for_three_digit_conditions_l2291_229139

theorem no_valid_n_for_three_digit_conditions :
  ∃ (n : ℕ) (h₁ : 100 ≤ n / 4 ∧ n / 4 ≤ 999) (h₂ : 100 ≤ 4 * n ∧ 4 * n ≤ 999), false :=
by sorry

end NUMINAMATH_GPT_no_valid_n_for_three_digit_conditions_l2291_229139


namespace NUMINAMATH_GPT_julie_upstream_distance_l2291_229100

noncomputable def speed_of_stream : ℝ := 0.5
noncomputable def distance_downstream : ℝ := 72
noncomputable def time_spent : ℝ := 4
noncomputable def speed_of_julie_in_still_water : ℝ := 17.5
noncomputable def distance_upstream : ℝ := 68

theorem julie_upstream_distance :
  (distance_upstream / (speed_of_julie_in_still_water - speed_of_stream) = time_spent) ∧
  (distance_downstream / (speed_of_julie_in_still_water + speed_of_stream) = time_spent) →
  distance_upstream = 68 :=
by 
  sorry

end NUMINAMATH_GPT_julie_upstream_distance_l2291_229100


namespace NUMINAMATH_GPT_cube_mod7_not_divisible_7_l2291_229171

theorem cube_mod7_not_divisible_7 (a : ℤ) (h : ¬ (7 ∣ a)) :
  (a^3 % 7 = 1) ∨ (a^3 % 7 = -1) :=
sorry

end NUMINAMATH_GPT_cube_mod7_not_divisible_7_l2291_229171


namespace NUMINAMATH_GPT_liquid_x_percentage_l2291_229153

theorem liquid_x_percentage (a_weight b_weight : ℝ) (a_percentage b_percentage : ℝ)
  (result_weight : ℝ) (x_weight_result : ℝ) (x_percentage_result : ℝ) :
  a_weight = 500 → b_weight = 700 → a_percentage = 0.8 / 100 →
  b_percentage = 1.8 / 100 → result_weight = a_weight + b_weight →
  x_weight_result = a_weight * a_percentage + b_weight * b_percentage →
  x_percentage_result = (x_weight_result / result_weight) * 100 →
  x_percentage_result = 1.3833 :=
by sorry

end NUMINAMATH_GPT_liquid_x_percentage_l2291_229153


namespace NUMINAMATH_GPT_export_volume_scientific_notation_l2291_229135

theorem export_volume_scientific_notation :
  (234.1 * 10^6) = (2.341 * 10^8) := 
sorry

end NUMINAMATH_GPT_export_volume_scientific_notation_l2291_229135


namespace NUMINAMATH_GPT_length_AB_indeterminate_l2291_229101

theorem length_AB_indeterminate
  (A B C : Type)
  (AC : ℝ) (BC : ℝ)
  (AC_eq_1 : AC = 1)
  (BC_eq_3 : BC = 3) :
  (2 < AB ∧ AB < 4) ∨ (AB = 2 ∨ AB = 4) → false :=
by sorry

end NUMINAMATH_GPT_length_AB_indeterminate_l2291_229101


namespace NUMINAMATH_GPT_min_value_am_hm_l2291_229109

theorem min_value_am_hm (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_am_hm_l2291_229109


namespace NUMINAMATH_GPT_total_floor_area_covered_l2291_229187

-- Definitions for the given problem
def combined_area : ℕ := 204
def overlap_two_layers : ℕ := 24
def overlap_three_layers : ℕ := 20
def total_floor_area : ℕ := 140

-- Theorem to prove the total floor area covered by the rugs
theorem total_floor_area_covered :
  combined_area - overlap_two_layers - 2 * overlap_three_layers = total_floor_area := by
  sorry

end NUMINAMATH_GPT_total_floor_area_covered_l2291_229187


namespace NUMINAMATH_GPT_jessie_problem_l2291_229189

def round_to_nearest_five (n : ℤ) : ℤ :=
  if n % 5 = 0 then n
  else if n % 5 < 3 then n - (n % 5)
  else n - (n % 5) + 5

theorem jessie_problem :
  round_to_nearest_five ((82 + 56) - 15) = 125 :=
by
  sorry

end NUMINAMATH_GPT_jessie_problem_l2291_229189


namespace NUMINAMATH_GPT_pancake_problem_l2291_229170

theorem pancake_problem :
  let mom_rate := (100 : ℚ) / 30
  let anya_rate := (100 : ℚ) / 40
  let andrey_rate := (100 : ℚ) / 60
  let combined_baking_rate := mom_rate + anya_rate
  let net_rate := combined_baking_rate - andrey_rate
  let target_pancakes := 100
  let time := target_pancakes / net_rate
  time = 24 := by
sorry

end NUMINAMATH_GPT_pancake_problem_l2291_229170


namespace NUMINAMATH_GPT_sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth_l2291_229145

noncomputable def sin_pi_div_two_plus_2alpha (α : ℝ) : ℝ :=
  Real.sin ((Real.pi / 2) + 2 * α)

def cos_alpha (α : ℝ) := Real.cos α = - (Real.sqrt 2) / 3

theorem sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth (α : ℝ) (h : cos_alpha α) :
  sin_pi_div_two_plus_2alpha α = -5 / 9 :=
sorry

end NUMINAMATH_GPT_sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth_l2291_229145


namespace NUMINAMATH_GPT_polygon_interior_angles_eq_360_l2291_229131

theorem polygon_interior_angles_eq_360 (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 :=
sorry

end NUMINAMATH_GPT_polygon_interior_angles_eq_360_l2291_229131


namespace NUMINAMATH_GPT_three_digit_number_452_l2291_229148

theorem three_digit_number_452 (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9) (h5 : 1 ≤ c) (h6 : c ≤ 9) 
  (h7 : 100 * a + 10 * b + c % (a + b + c) = 1)
  (h8 : 100 * c + 10 * b + a % (a + b + c) = 1)
  (h9 : a ≠ b) (h10 : b ≠ c) (h11 : a ≠ c)
  (h12 : a > c) :
  100 * a + 10 * b + c = 452 :=
sorry

end NUMINAMATH_GPT_three_digit_number_452_l2291_229148


namespace NUMINAMATH_GPT_average_mpg_highway_l2291_229185

variable (mpg_city : ℝ) (H mpg : ℝ) (gallons : ℝ) (max_distance : ℝ)

noncomputable def SUV_fuel_efficiency : Prop :=
  mpg_city  = 7.6 ∧
  gallons = 20 ∧
  max_distance = 244 ∧
  H * gallons = max_distance

theorem average_mpg_highway (h1 : mpg_city = 7.6) (h2 : gallons = 20) (h3 : max_distance = 244) :
  SUV_fuel_efficiency mpg_city H gallons max_distance → H = 12.2 :=
by
  intros h
  cases h
  sorry

end NUMINAMATH_GPT_average_mpg_highway_l2291_229185


namespace NUMINAMATH_GPT_integer_solution_x_l2291_229182

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end NUMINAMATH_GPT_integer_solution_x_l2291_229182


namespace NUMINAMATH_GPT_polar_to_rectangular_coords_l2291_229186

theorem polar_to_rectangular_coords (r θ : ℝ) (x y : ℝ) 
  (hr : r = 5) (hθ : θ = 5 * Real.pi / 4)
  (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  x = - (5 * Real.sqrt 2) / 2 ∧ y = - (5 * Real.sqrt 2) / 2 := 
by
  rw [hr, hθ] at hx hy
  simp [Real.cos, Real.sin] at hx hy
  rw [hx, hy]
  constructor
  . sorry
  . sorry

end NUMINAMATH_GPT_polar_to_rectangular_coords_l2291_229186


namespace NUMINAMATH_GPT_magical_stack_example_l2291_229113

-- Definitions based on the conditions
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def belongs_to_pile_A (card : ℕ) (n : ℕ) : Prop :=
  card <= n

def belongs_to_pile_B (card : ℕ) (n : ℕ) : Prop :=
  n < card

def magical_stack (cards : ℕ) (n : ℕ) : Prop :=
  ∀ (card : ℕ), (belongs_to_pile_A card n ∨ belongs_to_pile_B card n) → 
  (card + n) % (2 * n) = 1

-- The theorem to prove
theorem magical_stack_example :
  ∃ (n : ℕ), magical_stack 482 n ∧ (2 * n = 482) :=
by
  sorry

end NUMINAMATH_GPT_magical_stack_example_l2291_229113


namespace NUMINAMATH_GPT_nine_consecutive_arithmetic_mean_divisible_1111_l2291_229157

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end NUMINAMATH_GPT_nine_consecutive_arithmetic_mean_divisible_1111_l2291_229157


namespace NUMINAMATH_GPT_modified_expression_range_l2291_229160

open Int

theorem modified_expression_range (m : ℤ) :
  ∃ n_min n_max : ℤ, 1 < 4 * n_max + 7 ∧ 4 * n_min + 7 < 60 ∧ (n_max - n_min + 1 = 15) →
  ∃ k_min k_max : ℤ, 1 < m * k_max + 7 ∧ m * k_min + 7 < 60 ∧ (k_max - k_min + 1 ≥ 15) := 
sorry

end NUMINAMATH_GPT_modified_expression_range_l2291_229160


namespace NUMINAMATH_GPT_union_of_sets_l2291_229194

theorem union_of_sets (P Q : Set ℝ) 
  (hP : P = {x | 2 ≤ x ∧ x ≤ 3}) 
  (hQ : Q = {x | x^2 ≤ 4}) : 
  P ∪ Q = {x | -2 ≤ x ∧ x ≤ 3} := 
sorry

end NUMINAMATH_GPT_union_of_sets_l2291_229194


namespace NUMINAMATH_GPT_evaluate_expression_l2291_229162

noncomputable def a : ℕ := 2
noncomputable def b : ℕ := 1

theorem evaluate_expression : (1 / 2)^(b - a + 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2291_229162


namespace NUMINAMATH_GPT_mrs_sheridan_initial_cats_l2291_229121

theorem mrs_sheridan_initial_cats (bought_cats total_cats : ℝ) (h_bought : bought_cats = 43.0) (h_total : total_cats = 54) : total_cats - bought_cats = 11 :=
by
  rw [h_bought, h_total]
  norm_num

end NUMINAMATH_GPT_mrs_sheridan_initial_cats_l2291_229121


namespace NUMINAMATH_GPT_find_YJ_l2291_229172

structure Triangle :=
  (XY XZ YZ : ℝ)
  (XY_pos : XY > 0)
  (XZ_pos : XZ > 0)
  (YZ_pos : YZ > 0)

noncomputable def incenter_length (T : Triangle) : ℝ := 
  let XY := T.XY
  let XZ := T.XZ
  let YZ := T.YZ
  -- calculation using the provided constraints goes here
  3 * Real.sqrt 13 -- this should be computed based on the constraints, but is directly given as the answer

theorem find_YJ
  (T : Triangle)
  (XY_eq : T.XY = 17)
  (XZ_eq : T.XZ = 19)
  (YZ_eq : T.YZ = 20) :
  incenter_length T = 3 * Real.sqrt 13 :=
by 
  sorry

end NUMINAMATH_GPT_find_YJ_l2291_229172


namespace NUMINAMATH_GPT_intersection_S_T_l2291_229130

def S : Set ℝ := { x | 2 * x + 1 > 0 }
def T : Set ℝ := { x | 3 * x - 5 < 0 }

theorem intersection_S_T :
  S ∩ T = { x | -1/2 < x ∧ x < 5/3 } := by
  sorry

end NUMINAMATH_GPT_intersection_S_T_l2291_229130


namespace NUMINAMATH_GPT_race_course_length_to_finish_at_same_time_l2291_229122

variable (v : ℝ) -- speed of B
variable (d : ℝ) -- length of the race course

-- A's speed is 4 times B's speed and A gives B a 75-meter head start.
theorem race_course_length_to_finish_at_same_time (h1 : v > 0) (h2 : d > 75) : 
  (1 : ℝ) / 4 * (d / v) = ((d - 75) / v) ↔ d = 100 := 
sorry

end NUMINAMATH_GPT_race_course_length_to_finish_at_same_time_l2291_229122
