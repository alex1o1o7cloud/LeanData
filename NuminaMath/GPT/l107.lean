import Mathlib

namespace NUMINAMATH_GPT_certain_number_l107_10728

theorem certain_number (p q : ℝ) (h1 : 3 / p = 6) (h2 : p - q = 0.3) : 3 / q = 15 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_l107_10728


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l107_10760

variable (a : ℝ)

theorem necessary_and_sufficient_condition :
  (a^2 + 4 * a - 5 > 0) ↔ (|a + 2| > 3) := sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l107_10760


namespace NUMINAMATH_GPT_real_cube_inequality_l107_10775

theorem real_cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_real_cube_inequality_l107_10775


namespace NUMINAMATH_GPT_exists_perfect_square_sum_l107_10700

theorem exists_perfect_square_sum (n : ℕ) (h : n > 2) : ∃ m : ℕ, ∃ k : ℕ, n^2 + m^2 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_exists_perfect_square_sum_l107_10700


namespace NUMINAMATH_GPT_double_increase_divide_l107_10798

theorem double_increase_divide (x : ℤ) (h : (2 * x + 7) / 5 = 17) : x = 39 := by
  sorry

end NUMINAMATH_GPT_double_increase_divide_l107_10798


namespace NUMINAMATH_GPT_least_number_subtracted_l107_10720

theorem least_number_subtracted (a b c : ℕ) (h1 : a = 7) (h2 : b = 9) (h3 : c = 11) :
  ∃ x, 0 ≤ x ∧ x < 1398 ∧ (1398 - x) % a = 5 ∧ (1398 - x) % b = 5 ∧ (1398 - x) % c = 5 ∧ x = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_least_number_subtracted_l107_10720


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_sequence_l107_10788

-- Problem 1
theorem arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) (Sₙ : ℝ) 
  (h₁ : a₁ = 3 / 2) (h₂ : d = -1 / 2) (h₃ : Sₙ = -15) :
  n = 12 ∧ (a₁ + (n - 1) * d) = -4 := 
sorry

-- Problem 2
theorem geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) (aₙ Sₙ : ℝ) 
  (h₁ : q = 2) (h₂ : aₙ = 96) (h₃ : Sₙ = 189) :
  a₁ = 3 ∧ n = 6 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_sequence_l107_10788


namespace NUMINAMATH_GPT_determine_q_l107_10704

-- Define the polynomial p(x) and its square
def p (x : ℝ) : ℝ := x^2 + x + 1
def p_squared (x : ℝ) : ℝ := (x^2 + x + 1)^2

-- Define the identity condition
def identity_condition (x : ℝ) (q : ℝ → ℝ) : Prop := 
  p_squared x - 2 * p x * q x + (q x)^2 - 4 * p x + 3 * q x + 3 = 0

-- Ellaboration on the required solution
def correct_q (q : ℝ → ℝ) : Prop :=
  (∀ x, q x = x^2 + 2 * x) ∨ (∀ x, q x = x^2 - 1)

-- The theorem statement
theorem determine_q :
  ∀ q : ℝ → ℝ, (∀ x : ℝ, identity_condition x q) → correct_q q :=
by
  intros
  sorry

end NUMINAMATH_GPT_determine_q_l107_10704


namespace NUMINAMATH_GPT_number_of_black_and_white_films_l107_10766

theorem number_of_black_and_white_films (B x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h_fraction : (6 * y : ℚ) / ((y / (x : ℚ))/100 * (B : ℚ) + 6 * y) = 20 / 21) :
  B = 30 * x :=
sorry

end NUMINAMATH_GPT_number_of_black_and_white_films_l107_10766


namespace NUMINAMATH_GPT_natasha_dimes_l107_10762

theorem natasha_dimes (n : ℕ) :
  100 < n ∧ n < 200 ∧
  n % 3 = 2 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 7 = 2 ↔ n = 182 := by
sorry

end NUMINAMATH_GPT_natasha_dimes_l107_10762


namespace NUMINAMATH_GPT_alicia_total_payment_l107_10759

def daily_rent_cost : ℕ := 30
def miles_cost_per_mile : ℝ := 0.25
def rental_days : ℕ := 5
def driven_miles : ℕ := 500

def total_cost (daily_rent_cost : ℕ) (rental_days : ℕ)
               (miles_cost_per_mile : ℝ) (driven_miles : ℕ) : ℝ :=
  (daily_rent_cost * rental_days) + (miles_cost_per_mile * driven_miles)

theorem alicia_total_payment :
  total_cost daily_rent_cost rental_days miles_cost_per_mile driven_miles = 275 := by
  sorry

end NUMINAMATH_GPT_alicia_total_payment_l107_10759


namespace NUMINAMATH_GPT_find_p_l107_10770

theorem find_p (m n p : ℝ) 
  (h1 : m = 3 * n + 5) 
  (h2 : m + 2 = 3 * (n + p) + 5) : p = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l107_10770


namespace NUMINAMATH_GPT_probability_at_least_one_l107_10783

variable (p_A p_B : ℚ) (hA : p_A = 1 / 4) (hB : p_B = 2 / 5)

theorem probability_at_least_one (h : p_A * (1 - p_B) + (1 - p_A) * p_B + p_A * p_B = 11 / 20) : 
  (1 - (1 - p_A) * (1 - p_B) = 11 / 20) :=
by
  rw [hA, hB,←h]
  sorry

end NUMINAMATH_GPT_probability_at_least_one_l107_10783


namespace NUMINAMATH_GPT_investment_value_after_five_years_l107_10711

theorem investment_value_after_five_years :
  let initial_investment := 10000
  let year1 := initial_investment * (1 - 0.05) * (1 + 0.02)
  let year2 := year1 * (1 + 0.10) * (1 + 0.02)
  let year3 := year2 * (1 + 0.04) * (1 + 0.02)
  let year4 := year3 * (1 - 0.03) * (1 + 0.02)
  let year5 := year4 * (1 + 0.08) * (1 + 0.02)
  year5 = 12570.99 :=
  sorry

end NUMINAMATH_GPT_investment_value_after_five_years_l107_10711


namespace NUMINAMATH_GPT_real_roots_of_polynomial_l107_10752

theorem real_roots_of_polynomial :
  (∀ x : ℝ, (x^10 + 36 * x^6 + 13 * x^2 = 13 * x^8 + x^4 + 36) ↔ 
    (x = 1 ∨ x = -1 ∨ x = 3 ∨ x = -3 ∨ x = 2 ∨ x = -2)) :=
by 
  sorry

end NUMINAMATH_GPT_real_roots_of_polynomial_l107_10752


namespace NUMINAMATH_GPT_exists_representation_of_77_using_fewer_sevens_l107_10769

-- Definition of the problem
def represent_77 (expr : String) : Prop :=
  ∀ n : ℕ, expr = "77" ∨ 
             expr = "(77 - 7) + 7" ∨ 
             expr = "(10 * 7) + 7" ∨ 
             expr = "(70 + 7)" ∨ 
             expr = "(7 * 11)" ∨ 
             expr = "7 + 7 * 7 + (7 / 7)"

-- The proof statement
theorem exists_representation_of_77_using_fewer_sevens : ∃ expr : String, represent_77 expr ∧ String.length expr < 3 := 
sorry

end NUMINAMATH_GPT_exists_representation_of_77_using_fewer_sevens_l107_10769


namespace NUMINAMATH_GPT_unique_surjective_f_l107_10768

-- Define the problem conditions
variable (f : ℕ → ℕ)

-- Define that f is surjective
axiom surjective_f : Function.Surjective f

-- Define condition that for every m, n and prime p
axiom condition_f : ∀ m n : ℕ, ∀ p : ℕ, Nat.Prime p → (p ∣ f (m + n) ↔ p ∣ f m + f n)

-- The theorem we need to prove: the only surjective function f satisfying the condition is the identity function
theorem unique_surjective_f : ∀ x : ℕ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_unique_surjective_f_l107_10768


namespace NUMINAMATH_GPT_isosceles_triangle_congruent_side_length_l107_10793

theorem isosceles_triangle_congruent_side_length (BC : ℝ) (BM : ℝ) :
  BC = 4 * Real.sqrt 2 → BM = 5 → ∃ (AB : ℝ), AB = Real.sqrt 34 :=
by
  -- sorry is used here to indicate proof is not provided, but the statement is expected to build successfully.
  sorry

end NUMINAMATH_GPT_isosceles_triangle_congruent_side_length_l107_10793


namespace NUMINAMATH_GPT_rachel_homework_l107_10736

theorem rachel_homework : 5 + 2 = 7 := by
  sorry

end NUMINAMATH_GPT_rachel_homework_l107_10736


namespace NUMINAMATH_GPT_range_of_x_l107_10721

theorem range_of_x (x : ℝ) (h : Real.log (x - 1) < 1) : 1 < x ∧ x < Real.exp 1 + 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l107_10721


namespace NUMINAMATH_GPT_subcommittee_has_teacher_l107_10758

def total_combinations (n k : ℕ) : ℕ := Nat.choose n k

def teacher_subcommittee_count : ℕ := total_combinations 12 5 - total_combinations 7 5

theorem subcommittee_has_teacher : teacher_subcommittee_count = 771 := 
by
  sorry

end NUMINAMATH_GPT_subcommittee_has_teacher_l107_10758


namespace NUMINAMATH_GPT_tan_of_perpendicular_vectors_l107_10771

theorem tan_of_perpendicular_vectors (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (ha : ℝ × ℝ := (Real.cos θ, 2)) (hb : ℝ × ℝ := (-1, Real.sin θ))
  (h_perpendicular : ha.1 * hb.1 + ha.2 * hb.2 = 0) :
  Real.tan θ = 1 / 2 := 
sorry

end NUMINAMATH_GPT_tan_of_perpendicular_vectors_l107_10771


namespace NUMINAMATH_GPT_expression_evaluation_l107_10795

def e1 : ℤ := 72 + (120 / 15) + (15 * 12) - 250 - (480 / 8)

theorem expression_evaluation : e1 = -50 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l107_10795


namespace NUMINAMATH_GPT_crude_oil_mixture_l107_10784

theorem crude_oil_mixture (x y : ℝ) 
  (h1 : x + y = 50)
  (h2 : 0.25 * x + 0.75 * y = 0.55 * 50) : 
  y = 30 :=
by
  sorry

end NUMINAMATH_GPT_crude_oil_mixture_l107_10784


namespace NUMINAMATH_GPT_number_of_floors_l107_10713

-- Definitions
def height_regular_floor : ℝ := 3
def height_last_floor : ℝ := 3.5
def total_height : ℝ := 61

-- Theorem statement
theorem number_of_floors (n : ℕ) : 
  (n ≥ 2) →
  (2 * height_last_floor + (n - 2) * height_regular_floor = total_height) →
  n = 20 :=
sorry

end NUMINAMATH_GPT_number_of_floors_l107_10713


namespace NUMINAMATH_GPT_cos_C_value_l107_10725

namespace Triangle

theorem cos_C_value (A B C : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (sin_A : Real.sin A = 2/3)
  (cos_B : Real.cos B = 1/2) :
  Real.cos C = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := 
sorry

end Triangle

end NUMINAMATH_GPT_cos_C_value_l107_10725


namespace NUMINAMATH_GPT_common_face_sum_is_9_l107_10729

noncomputable def common_sum (vertices : Fin 9 → ℕ) : ℕ :=
  let total_sum := (Finset.sum (Finset.univ : Finset (Fin 9)) vertices)
  let additional_sum := 9
  let total_with_addition := total_sum + additional_sum
  total_with_addition / 6

theorem common_face_sum_is_9 :
  ∀ (vertices : Fin 9 → ℕ), (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 9) →
  Finset.sum (Finset.univ : Finset (Fin 9)) vertices = 45 →
  common_sum vertices = 9 := 
by
  intros vertices h1 h_sum
  unfold common_sum
  sorry

end NUMINAMATH_GPT_common_face_sum_is_9_l107_10729


namespace NUMINAMATH_GPT_exists_sum_or_diff_divisible_by_1000_l107_10709

theorem exists_sum_or_diff_divisible_by_1000 (nums : Fin 502 → Nat) :
  ∃ a b : Nat, (∃ i j : Fin 502, nums i = a ∧ nums j = b ∧ i ≠ j) ∧
  (a - b) % 1000 = 0 ∨ (a + b) % 1000 = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_sum_or_diff_divisible_by_1000_l107_10709


namespace NUMINAMATH_GPT_ramola_rank_from_first_l107_10755

-- Conditions definitions
def total_students : ℕ := 26
def ramola_rank_from_last : ℕ := 13

-- Theorem statement
theorem ramola_rank_from_first : total_students - (ramola_rank_from_last - 1) = 14 := 
by 
-- We use 'by' to begin the proof block
sorry 
-- We use 'sorry' to indicate the proof is omitted

end NUMINAMATH_GPT_ramola_rank_from_first_l107_10755


namespace NUMINAMATH_GPT_calculate_expression_l107_10745

theorem calculate_expression : 15 * 30 + 45 * 15 + 90 = 1215 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l107_10745


namespace NUMINAMATH_GPT_chess_club_boys_count_l107_10753

theorem chess_club_boys_count (B G : ℕ) 
  (h1 : B + G = 30)
  (h2 : (2/3 : ℝ) * G + B = 18) : 
  B = 6 :=
by
  sorry

end NUMINAMATH_GPT_chess_club_boys_count_l107_10753


namespace NUMINAMATH_GPT_trig_identity_l107_10780

theorem trig_identity {α : ℝ} (h : Real.tan α = 2) : 
  (Real.sin (π + α) - Real.cos (π - α)) / 
  (Real.sin (π / 2 + α) - Real.cos (3 * π / 2 - α)) 
  = -1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l107_10780


namespace NUMINAMATH_GPT_tan_expression_l107_10722

theorem tan_expression (a : ℝ) (h₀ : 45 = 2 * a) (h₁ : Real.tan 45 = 1) 
  (h₂ : Real.tan (2 * a) = 2 * Real.tan a / (1 - Real.tan a * Real.tan a)) :
  Real.tan a / (1 - Real.tan a * Real.tan a) = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_tan_expression_l107_10722


namespace NUMINAMATH_GPT_function_periodicity_even_l107_10724

theorem function_periodicity_even (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_period : ∀ x : ℝ, x ≥ 0 → f (x + 2) = -f x)
  (h_def : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - 1) :
  f (-2017) + f 2018 = 1 :=
sorry

end NUMINAMATH_GPT_function_periodicity_even_l107_10724


namespace NUMINAMATH_GPT_slope_of_parallel_line_l107_10742

-- Given condition: the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - 4 * y = 9

-- Goal: the slope of any line parallel to the given line is 1/2
theorem slope_of_parallel_line (x y : ℝ) (m : ℝ) :
  (∀ x y, line_equation x y) → m = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l107_10742


namespace NUMINAMATH_GPT_temperature_difference_l107_10781

theorem temperature_difference (H L : ℝ) (hH : H = 8) (hL : L = -2) :
  H - L = 10 :=
by
  rw [hH, hL]
  norm_num

end NUMINAMATH_GPT_temperature_difference_l107_10781


namespace NUMINAMATH_GPT_total_robots_correct_l107_10761

def number_of_shapes : ℕ := 3
def number_of_colors : ℕ := 4
def total_types_of_robots : ℕ := number_of_shapes * number_of_colors

theorem total_robots_correct : total_types_of_robots = 12 := by
  sorry

end NUMINAMATH_GPT_total_robots_correct_l107_10761


namespace NUMINAMATH_GPT_stamps_on_last_page_l107_10701

theorem stamps_on_last_page
  (B : ℕ) (P_b : ℕ) (S_p : ℕ) (S_p_star : ℕ) 
  (B_comp : ℕ) (P_last : ℕ) 
  (stamps_total : ℕ := B * P_b * S_p) 
  (pages_total : ℕ := stamps_total / S_p_star)
  (pages_comp : ℕ := B_comp * P_b)
  (pages_filled : ℕ := pages_total - pages_comp) :
  stamps_total - (pages_total - 1) * S_p_star = 8 :=
by
  -- Proof steps would follow here.
  sorry

end NUMINAMATH_GPT_stamps_on_last_page_l107_10701


namespace NUMINAMATH_GPT_question_1_solution_question_2_solution_l107_10706

def f (m x : ℝ) := m*x^2 - (m^2 + 1)*x + m

theorem question_1_solution (x : ℝ) :
  (f 2 x ≤ 0) ↔ (1 / 2 ≤ x ∧ x ≤ 2) :=
sorry

theorem question_2_solution (x m : ℝ) :
  (m > 0) → 
  ((0 < m ∧ m < 1 → f m x > 0 ↔ x < m ∨ x > 1 / m) ∧
  (m = 1 → f m x > 0 ↔ x ≠ 1) ∧
  (m > 1 → f m x > 0 ↔ x < 1 / m ∨ x > m)) :=
sorry

end NUMINAMATH_GPT_question_1_solution_question_2_solution_l107_10706


namespace NUMINAMATH_GPT_sum_of_cubes_of_ages_l107_10708

noncomputable def dick_age : ℕ := 2
noncomputable def tom_age : ℕ := 5
noncomputable def harry_age : ℕ := 6

theorem sum_of_cubes_of_ages :
  4 * dick_age + 2 * tom_age = 3 * harry_age ∧ 
  3 * harry_age^2 = 2 * dick_age^2 + 4 * tom_age^2 ∧ 
  Nat.gcd (Nat.gcd dick_age tom_age) harry_age = 1 → 
  dick_age^3 + tom_age^3 + harry_age^3 = 349 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_ages_l107_10708


namespace NUMINAMATH_GPT_factorize_expression_l107_10714

-- Define the variables m and n
variables (m n : ℝ)

-- The statement to prove
theorem factorize_expression : -8 * m^2 + 2 * m * n = -2 * m * (4 * m - n) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l107_10714


namespace NUMINAMATH_GPT_expression_meaningful_range_l107_10743

theorem expression_meaningful_range (a : ℝ) : (∃ x, x = (a + 3) ^ (1/2) / (a - 1)) ↔ (a ≥ -3 ∧ a ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_expression_meaningful_range_l107_10743


namespace NUMINAMATH_GPT_non_congruent_rectangles_unique_l107_10717

theorem non_congruent_rectangles_unique (P : ℕ) (w : ℕ) (h : ℕ) :
  P = 72 ∧ w = 14 ∧ 2 * (w + h) = P → 
  (∃ h, w = 14 ∧ 2 * (w + h) = 72 ∧ 
  ∀ w' h', w' = w → 2 * (w' + h') = 72 → (h' = h)) :=
by
  sorry

end NUMINAMATH_GPT_non_congruent_rectangles_unique_l107_10717


namespace NUMINAMATH_GPT_right_triangle_area_semi_perimeter_inequality_l107_10741

theorem right_triangle_area_semi_perimeter_inequality 
  (x y : ℝ) (h : x > 0 ∧ y > 0) 
  (p : ℝ := (x + y + Real.sqrt (x^2 + y^2)) / 2)
  (S : ℝ := x * y / 2) 
  (hypotenuse : ℝ := Real.sqrt (x^2 + y^2)) 
  (right_triangle : hypotenuse ^ 2 = x ^ 2 + y ^ 2) : 
  S <= p^2 / 5.5 := 
sorry

end NUMINAMATH_GPT_right_triangle_area_semi_perimeter_inequality_l107_10741


namespace NUMINAMATH_GPT_false_proposition_l107_10782

theorem false_proposition :
  ¬ (∀ x : ℕ, (x > 0) → (x - 2)^2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_false_proposition_l107_10782


namespace NUMINAMATH_GPT_inequality_transform_l107_10705

theorem inequality_transform {a b c d e : ℝ} (hab : a > b) (hb0 : b > 0) 
  (hcd : c < d) (hd0 : d < 0) (he : e < 0) : 
  e / (a - c)^2 > e / (b - d)^2 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_transform_l107_10705


namespace NUMINAMATH_GPT_ratio_of_division_of_chord_l107_10757

theorem ratio_of_division_of_chord (R AP PB O: ℝ) (radius_given: R = 11) (chord_length: AP + PB = 18) (point_distance: O = 7) : 
  (AP / PB = 2 ∨ PB / AP = 2) :=
by 
  -- Proof goes here, to be filled in later
  sorry

end NUMINAMATH_GPT_ratio_of_division_of_chord_l107_10757


namespace NUMINAMATH_GPT_find_number_l107_10779

theorem find_number (x : ℝ) (h : x = 0.16 * x + 21) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l107_10779


namespace NUMINAMATH_GPT_hockey_league_teams_l107_10751

theorem hockey_league_teams (n : ℕ) (h : (n * (n - 1) * 10) / 2 = 1710) : n = 19 :=
by {
  sorry
}

end NUMINAMATH_GPT_hockey_league_teams_l107_10751


namespace NUMINAMATH_GPT_relay_team_permutations_l107_10774

-- Definitions of conditions
def runners := ["Tony", "Leah", "Nina"]
def fixed_positions := ["Maria runs the third lap", "Jordan runs the fifth lap"]

-- Proof statement
theorem relay_team_permutations : 
  ∃ permutations, permutations = 6 := by
sorry

end NUMINAMATH_GPT_relay_team_permutations_l107_10774


namespace NUMINAMATH_GPT_find_sales_discount_l107_10794

noncomputable def salesDiscountPercentage (P N : ℝ) (D : ℝ): Prop :=
  let originalGrossIncome := P * N
  let newPrice := P * (1 - D / 100)
  let newNumberOfItems := N * 1.20
  let newGrossIncome := newPrice * newNumberOfItems
  newGrossIncome = originalGrossIncome * 1.08

theorem find_sales_discount (P N : ℝ) (hP : P > 0) (hN : N > 0) (h: ∃ D, salesDiscountPercentage P N D) :
  ∃ D, D = 10 :=
sorry

end NUMINAMATH_GPT_find_sales_discount_l107_10794


namespace NUMINAMATH_GPT_abc_sum_leq_three_l107_10702

open Real

theorem abc_sum_leq_three {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 + a * b * c = 4) :
  a + b + c ≤ 3 :=
sorry

end NUMINAMATH_GPT_abc_sum_leq_three_l107_10702


namespace NUMINAMATH_GPT_two_y_minus_three_x_l107_10785

variable (x y : ℝ)

noncomputable def x_val : ℝ := 1.2 * 98
noncomputable def y_val : ℝ := 0.9 * (x_val + 35)

theorem two_y_minus_three_x : 2 * y_val - 3 * x_val = -78.12 := by
  sorry

end NUMINAMATH_GPT_two_y_minus_three_x_l107_10785


namespace NUMINAMATH_GPT_min_value_expression_l107_10739

theorem min_value_expression : 
  ∀ (x y : ℝ), (3 * x * x + 4 * x * y + 4 * y * y - 12 * x - 8 * y ≥ -28) ∧ 
  (3 * ((8:ℝ)/3) * ((8:ℝ)/3) + 4 * ((8:ℝ)/3) * -1 + 4 * -1 * -1 - 12 * ((8:ℝ)/3) - 8 * -1 = -28) := 
by sorry

end NUMINAMATH_GPT_min_value_expression_l107_10739


namespace NUMINAMATH_GPT_train_length_l107_10723

theorem train_length (L V : ℝ) 
  (h1 : V = L / 10) 
  (h2 : V = (L + 870) / 39) 
  : L = 300 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l107_10723


namespace NUMINAMATH_GPT_square_of_binomial_l107_10790

-- Define a condition that the given term is the square of a binomial.
theorem square_of_binomial (a b: ℝ) : (a + b) * (a + b) = (a + b) ^ 2 :=
by {
  -- The proof is omitted.
  sorry
}

end NUMINAMATH_GPT_square_of_binomial_l107_10790


namespace NUMINAMATH_GPT_picnic_adults_children_difference_l107_10703

theorem picnic_adults_children_difference :
  ∃ (M W A C : ℕ),
    (M = 65) ∧
    (M = W + 20) ∧
    (A = M + W) ∧
    (C = 200 - A) ∧
    ((A - C) = 20) :=
by
  sorry

end NUMINAMATH_GPT_picnic_adults_children_difference_l107_10703


namespace NUMINAMATH_GPT_least_integer_greater_than_sqrt_500_l107_10734

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end NUMINAMATH_GPT_least_integer_greater_than_sqrt_500_l107_10734


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l107_10715

variable (x : ℝ)

def p := x > 2
def q := x^2 > 4

theorem sufficient_but_not_necessary_condition : (p x) → (q x) ∧ ¬((q x) → (p x)) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l107_10715


namespace NUMINAMATH_GPT_tetrahedron_face_area_inequality_l107_10712

theorem tetrahedron_face_area_inequality
  (T_ABC T_ABD T_ACD T_BCD : ℝ)
  (h : T_ABC ≥ 0 ∧ T_ABD ≥ 0 ∧ T_ACD ≥ 0 ∧ T_BCD ≥ 0) :
  T_ABC < T_ABD + T_ACD + T_BCD :=
sorry

end NUMINAMATH_GPT_tetrahedron_face_area_inequality_l107_10712


namespace NUMINAMATH_GPT_infinite_series_sum_l107_10727

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l107_10727


namespace NUMINAMATH_GPT_solution_set_equivalence_l107_10710

theorem solution_set_equivalence (a : ℝ) : 
    (-1 < a ∧ a < 1) ∧ (3 * a^2 - 2 * a - 5 < 0) → 
    (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) :=
by
    sorry

end NUMINAMATH_GPT_solution_set_equivalence_l107_10710


namespace NUMINAMATH_GPT_salary_reduction_l107_10763

noncomputable def percentageIncrease : ℝ := 16.27906976744186 / 100

theorem salary_reduction (S R : ℝ) (P : ℝ) (h1 : R = S * (1 - P / 100)) (h2 : S = R * (1 + percentageIncrease)) : P = 14 :=
by
  sorry

end NUMINAMATH_GPT_salary_reduction_l107_10763


namespace NUMINAMATH_GPT_correct_option_l107_10756

-- Define the options as propositions
def OptionA (a : ℕ) := a ^ 3 * a ^ 5 = a ^ 15
def OptionB (a : ℕ) := a ^ 8 / a ^ 2 = a ^ 4
def OptionC (a : ℕ) := a ^ 2 + a ^ 3 = a ^ 5
def OptionD (a : ℕ) := 3 * a - a = 2 * a

-- Prove that Option D is the only correct statement
theorem correct_option (a : ℕ) : OptionD a ∧ ¬OptionA a ∧ ¬OptionB a ∧ ¬OptionC a :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l107_10756


namespace NUMINAMATH_GPT_coordinate_plane_line_l107_10718

theorem coordinate_plane_line (m n p : ℝ) (h1 : m = n / 5 - 2 / 5) (h2 : m + p = (n + 15) / 5 - 2 / 5) : p = 3 := by
  sorry

end NUMINAMATH_GPT_coordinate_plane_line_l107_10718


namespace NUMINAMATH_GPT_cost_price_per_meter_l107_10787

theorem cost_price_per_meter (number_of_meters : ℕ) (selling_price : ℝ) (profit_per_meter : ℝ) (total_cost_price : ℝ) (cost_per_meter : ℝ) :
  number_of_meters = 85 →
  selling_price = 8925 →
  profit_per_meter = 15 →
  total_cost_price = selling_price - (profit_per_meter * number_of_meters) →
  cost_per_meter = total_cost_price / number_of_meters →
  cost_per_meter = 90 :=
by
  intros h1 h2 h3 h4 h5 
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l107_10787


namespace NUMINAMATH_GPT_train_speed_is_40_kmh_l107_10792

noncomputable def speed_of_train (train_length_m : ℝ) 
                                   (man_speed_kmh : ℝ) 
                                   (pass_time_s : ℝ) : ℝ :=
  let train_length_km := train_length_m / 1000
  let pass_time_h := pass_time_s / 3600
  let relative_speed_kmh := train_length_km / pass_time_h
  relative_speed_kmh - man_speed_kmh
  
theorem train_speed_is_40_kmh :
  speed_of_train 110 4 9 = 40 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_is_40_kmh_l107_10792


namespace NUMINAMATH_GPT_polynomial_satisfies_conditions_l107_10738

noncomputable def f (x y z : ℝ) : ℝ :=
  (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

theorem polynomial_satisfies_conditions :
  (f x (z^2) y + f x (y^2) z = 0) ∧ (f (z^3) y x + f (x^3) y z = 0) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_satisfies_conditions_l107_10738


namespace NUMINAMATH_GPT_team_testing_equation_l107_10746

variable (x : ℝ)

theorem team_testing_equation (h : x > 15) : (600 / x = 500 / (x - 15) * 0.9) :=
sorry

end NUMINAMATH_GPT_team_testing_equation_l107_10746


namespace NUMINAMATH_GPT_alcohol_percentage_calculation_l107_10731

-- Define the conditions as hypothesis
variables (original_solution_volume : ℝ) (original_alcohol_percent : ℝ)
          (added_alcohol_volume : ℝ) (added_water_volume : ℝ)

-- Assume the given values in the problem
variables (h1 : original_solution_volume = 40) (h2 : original_alcohol_percent = 5)
          (h3 : added_alcohol_volume = 2.5) (h4 : added_water_volume = 7.5)

-- Define the proof goal
theorem alcohol_percentage_calculation :
  let original_alcohol_volume := original_solution_volume * (original_alcohol_percent / 100)
  let total_alcohol_volume := original_alcohol_volume + added_alcohol_volume
  let total_solution_volume := original_solution_volume + added_alcohol_volume + added_water_volume
  let new_alcohol_percent := (total_alcohol_volume / total_solution_volume) * 100
  new_alcohol_percent = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_alcohol_percentage_calculation_l107_10731


namespace NUMINAMATH_GPT_fraction_checked_by_worker_y_l107_10778

variable (P : ℝ) -- Total number of products
variable (f_X f_Y : ℝ) -- Fraction of products checked by worker X and Y
variable (dx : ℝ) -- Defective rate for worker X
variable (dy : ℝ) -- Defective rate for worker Y
variable (dt : ℝ) -- Total defective rate

-- Conditions
axiom f_sum : f_X + f_Y = 1
axiom dx_val : dx = 0.005
axiom dy_val : dy = 0.008
axiom dt_val : dt = 0.0065

-- Proof
theorem fraction_checked_by_worker_y : f_Y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_checked_by_worker_y_l107_10778


namespace NUMINAMATH_GPT_solve_for_y_l107_10791

theorem solve_for_y (y : ℝ) : y^2 - 6 * y + 5 = 0 ↔ y = 1 ∨ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l107_10791


namespace NUMINAMATH_GPT_area_S4_is_3_125_l107_10776

theorem area_S4_is_3_125 (S_1 : Type) (area_S1 : ℝ) 
  (hS1 : area_S1 = 25)
  (bisect_and_construct : ∀ (S : Type) (area : ℝ),
    ∃ S' : Type, ∃ area' : ℝ, area' = area / 2) :
  ∃ S_4 : Type, ∃ area_S4 : ℝ, area_S4 = 3.125 :=
by
  sorry

end NUMINAMATH_GPT_area_S4_is_3_125_l107_10776


namespace NUMINAMATH_GPT_smallest_lcm_l107_10789

theorem smallest_lcm (k l : ℕ) (hk : k ≥ 1000) (hl : l ≥ 1000) (huk : k < 10000) (hul : l < 10000) (hk_pos : 0 < k) (hl_pos : 0 < l) (h_gcd: Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
by
  sorry

end NUMINAMATH_GPT_smallest_lcm_l107_10789


namespace NUMINAMATH_GPT_g_of_5_l107_10716

variable {g : ℝ → ℝ}
variable (h1 : ∀ x y : ℝ, 2 * x * g y = 3 * y * g x)
variable (h2 : g 10 = 15)

theorem g_of_5 : g 5 = 45 / 4 :=
  sorry

end NUMINAMATH_GPT_g_of_5_l107_10716


namespace NUMINAMATH_GPT_integer_not_in_range_of_f_l107_10749

noncomputable def f (x : ℝ) : ℤ :=
  if x > -1 then ⌈1 / (x + 1)⌉ else ⌊1 / (x + 1)⌋

theorem integer_not_in_range_of_f :
  ¬ ∃ x : ℝ, x ≠ -1 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_integer_not_in_range_of_f_l107_10749


namespace NUMINAMATH_GPT_area_to_be_painted_correct_l107_10765

-- Define the dimensions and areas involved
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window_height : ℕ := 3
def window_length : ℕ := 5
def painting_height : ℕ := 2
def painting_length : ℕ := 2

-- Calculate the areas
def wall_area : ℕ := wall_height * wall_length
def window_area : ℕ := window_height * window_length
def painting_area : ℕ := painting_height * painting_length
def area_not_painted : ℕ := window_area + painting_area
def area_to_be_painted : ℕ := wall_area - area_not_painted

-- Theorem: The area to be painted is 131 square feet
theorem area_to_be_painted_correct : area_to_be_painted = 131 := by
  sorry

end NUMINAMATH_GPT_area_to_be_painted_correct_l107_10765


namespace NUMINAMATH_GPT_distinct_patterns_4x4_3_shaded_l107_10747

def num_distinct_patterns (n : ℕ) (shading : ℕ) : ℕ :=
  if n = 4 ∧ shading = 3 then 15
  else 0 -- Placeholder for other cases, not relevant for our problem

theorem distinct_patterns_4x4_3_shaded :
  num_distinct_patterns 4 3 = 15 :=
by {
  -- The proof would go here
  sorry
}

end NUMINAMATH_GPT_distinct_patterns_4x4_3_shaded_l107_10747


namespace NUMINAMATH_GPT_projectile_reaches_35m_first_at_10_over_7_l107_10719

theorem projectile_reaches_35m_first_at_10_over_7 :
  ∃ (t : ℝ), (y : ℝ) = -4.9 * t^2 + 30 * t ∧ y = 35 ∧ t = 10 / 7 :=
by
  sorry

end NUMINAMATH_GPT_projectile_reaches_35m_first_at_10_over_7_l107_10719


namespace NUMINAMATH_GPT_toby_photos_l107_10764

variable (p0 d c e x : ℕ)
def photos_remaining : ℕ := p0 - d + c + x - e

theorem toby_photos (h1 : p0 = 63) (h2 : d = 7) (h3 : c = 15) (h4 : e = 3) : photos_remaining p0 d c e x = 68 + x :=
by
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_toby_photos_l107_10764


namespace NUMINAMATH_GPT_sin_minus_cos_eq_l107_10744

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end NUMINAMATH_GPT_sin_minus_cos_eq_l107_10744


namespace NUMINAMATH_GPT_union_of_M_N_l107_10750

def M : Set ℝ := { x | x^2 + 2*x = 0 }

def N : Set ℝ := { x | x^2 - 2*x = 0 }

theorem union_of_M_N : M ∪ N = {0, -2, 2} := sorry

end NUMINAMATH_GPT_union_of_M_N_l107_10750


namespace NUMINAMATH_GPT_inequality_preservation_l107_10735

theorem inequality_preservation (a b x : ℝ) (h : a > b) : a * 2^x > b * 2^x :=
sorry

end NUMINAMATH_GPT_inequality_preservation_l107_10735


namespace NUMINAMATH_GPT_number_of_boundaries_l107_10799

def total_runs : ℕ := 120
def sixes : ℕ := 4
def runs_per_six : ℕ := 6
def percentage_runs_by_running : ℚ := 0.60
def runs_per_boundary : ℕ := 4

theorem number_of_boundaries :
  let runs_by_running := (percentage_runs_by_running * total_runs : ℚ)
  let runs_by_sixes := (sixes * runs_per_six)
  let runs_by_boundaries := (total_runs - runs_by_running - runs_by_sixes : ℚ)
  (runs_by_boundaries / runs_per_boundary) = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_boundaries_l107_10799


namespace NUMINAMATH_GPT_ratio_of_rectangles_l107_10732

theorem ratio_of_rectangles (p q : ℝ) (h1 : q ≠ 0) 
    (h2 : q^2 = 1/4 * (2 * p * q  - q^2)) : p / q = 5 / 2 := 
sorry

end NUMINAMATH_GPT_ratio_of_rectangles_l107_10732


namespace NUMINAMATH_GPT_inequality_proof_l107_10777

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
    (x^4 / (y * (1 - y^2))) + (y^4 / (z * (1 - z^2))) + (z^4 / (x * (1 - x^2))) ≥ 1 / 8 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l107_10777


namespace NUMINAMATH_GPT_alpha_sin_beta_lt_beta_sin_alpha_l107_10730

variable {α β : ℝ}

theorem alpha_sin_beta_lt_beta_sin_alpha (h1 : 0 < α) (h2 : α < β) (h3 : β < Real.pi / 2) : 
  α * Real.sin β < β * Real.sin α := 
by
  sorry

end NUMINAMATH_GPT_alpha_sin_beta_lt_beta_sin_alpha_l107_10730


namespace NUMINAMATH_GPT_maximum_elevation_l107_10767

-- Define the elevation function
def elevation (t : ℝ) : ℝ := 200 * t - 17 * t^2 - 3 * t^3

-- State that the maximum elevation is 368.1 feet
theorem maximum_elevation :
  ∃ t : ℝ, t > 0 ∧ (∀ t' : ℝ, t' ≠ t → elevation t ≤ elevation t') ∧ elevation t = 368.1 :=
by
  sorry

end NUMINAMATH_GPT_maximum_elevation_l107_10767


namespace NUMINAMATH_GPT_find_intersection_A_B_find_range_t_l107_10733

-- Define sets A, B, C
def A : Set ℝ := {y | ∃ x, (1 ≤ x ∧ x ≤ 2) ∧ y = 2^x}
def B : Set ℝ := {x | 0 < Real.log x ∧ Real.log x < 1}
def C (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2 * t}

-- Theorem 1: Finding A ∩ B
theorem find_intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < Real.exp 1} := 
by
  sorry

-- Theorem 2: If A ∩ C = C, find the range of values for t
theorem find_range_t (t : ℝ) (h : A ∩ C t = C t) : t ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_find_intersection_A_B_find_range_t_l107_10733


namespace NUMINAMATH_GPT_quadratic_condition_not_necessary_and_sufficient_l107_10726

theorem quadratic_condition_not_necessary_and_sufficient (a b c : ℝ) :
  ¬((∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (b^2 - 4 * a * c < 0)) :=
sorry

end NUMINAMATH_GPT_quadratic_condition_not_necessary_and_sufficient_l107_10726


namespace NUMINAMATH_GPT_A_investment_l107_10796

variable (x : ℕ)
variable (A_share : ℕ := 3780)
variable (Total_profit : ℕ := 12600)
variable (B_invest : ℕ := 4200)
variable (C_invest : ℕ := 10500)

theorem A_investment :
  (A_share : ℝ) / (Total_profit : ℝ) = (x : ℝ) / (x + B_invest + C_invest) →
  x = 6300 :=
by
  sorry

end NUMINAMATH_GPT_A_investment_l107_10796


namespace NUMINAMATH_GPT_scout_troop_profit_l107_10748

theorem scout_troop_profit :
  let bars_bought := 1200
  let cost_per_bar := 1 / 3
  let bars_per_dollar := 3
  let total_cost := bars_bought * cost_per_bar
  let selling_price_per_bar := 3 / 5
  let bars_per_three_dollars := 5
  let total_revenue := bars_bought * selling_price_per_bar
  let profit := total_revenue - total_cost
  profit = 320 := by
  let bars_bought := 1200
  let cost_per_bar := 1 / 3
  let total_cost := bars_bought * cost_per_bar
  let selling_price_per_bar := 3 / 5
  let total_revenue := bars_bought * selling_price_per_bar
  let profit := total_revenue - total_cost
  sorry

end NUMINAMATH_GPT_scout_troop_profit_l107_10748


namespace NUMINAMATH_GPT_part_a_part_b_l107_10797

-- Part (a): Prove that for N = a^2 + 2, the equation has positive integral solutions for infinitely many a.
theorem part_a (N : ℕ) (a : ℕ) (x y z t : ℕ) (hx : x = a * (a^2 + 2)) (hy : y = a) (hz : z = 1) (ht : t = 1) :
  (∃ (N : ℕ), ∀ (a : ℕ), ∃ (x y z t : ℕ),
    x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0) :=
sorry

-- Part (b): Prove that for N = 4^k(8m + 7), the equation has no positive integral solutions.
theorem part_b (N : ℕ) (k m : ℕ) (x y z t : ℕ) (hN : N = 4^k * (8 * m + 7)) :
  ¬ (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l107_10797


namespace NUMINAMATH_GPT_percentage_died_by_bombardment_l107_10786

def initial_population : ℕ := 4675
def remaining_population : ℕ := 3553
def left_percentage : ℕ := 20

theorem percentage_died_by_bombardment (x : ℕ) (h : initial_population * (100 - x) / 100 * 8 / 10 = remaining_population) : 
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_died_by_bombardment_l107_10786


namespace NUMINAMATH_GPT_average_community_age_l107_10737

variable (num_women num_men : Nat)
variable (avg_age_women avg_age_men : Nat)

def ratio_women_men := num_women = 7 * num_men / 8
def average_age_women := avg_age_women = 30
def average_age_men := avg_age_men = 35

theorem average_community_age (k : Nat) 
  (h_ratio : ratio_women_men (7 * k) (8 * k)) 
  (h_avg_women : average_age_women 30)
  (h_avg_men : average_age_men 35) : 
  (30 * (7 * k) + 35 * (8 * k)) / (15 * k) = 32 + (2 / 3) := 
sorry

end NUMINAMATH_GPT_average_community_age_l107_10737


namespace NUMINAMATH_GPT_peaches_eaten_correct_l107_10772

-- Given conditions
def total_peaches : ℕ := 18
def initial_ripe_peaches : ℕ := 4
def peaches_ripen_per_day : ℕ := 2
def days_passed : ℕ := 5
def ripe_unripe_difference : ℕ := 7

-- Definitions derived from conditions
def ripe_peaches_after_days := initial_ripe_peaches + peaches_ripen_per_day * days_passed
def unripe_peaches_initial := total_peaches - initial_ripe_peaches
def unripe_peaches_after_days := unripe_peaches_initial - peaches_ripen_per_day * days_passed
def actual_ripe_peaches_needed := unripe_peaches_after_days + ripe_unripe_difference
def peaches_eaten := ripe_peaches_after_days - actual_ripe_peaches_needed

-- Prove that the number of peaches eaten is equal to 3
theorem peaches_eaten_correct : peaches_eaten = 3 := by
  sorry

end NUMINAMATH_GPT_peaches_eaten_correct_l107_10772


namespace NUMINAMATH_GPT_second_discount_percentage_l107_10773

-- Define the original price as P
variables {P : ℝ} (hP : P > 0)

-- Define the price increase by 34%
def price_after_increase (P : ℝ) := 1.34 * P

-- Define the first discount of 10%
def price_after_first_discount (P : ℝ) := 0.90 * (price_after_increase P)

-- Define the second discount percentage as D (in decimal form)
variables {D : ℝ}

-- Define the price after the second discount
def price_after_second_discount (P D : ℝ) := (1 - D) * (price_after_first_discount P)

-- Define the overall percentage gain of 2.51%
def final_price (P : ℝ) := 1.0251 * P

-- The main theorem to prove
theorem second_discount_percentage (hP : P > 0) (hD : 0 ≤ D ∧ D ≤ 1) :
  price_after_second_discount P D = final_price P ↔ D = 0.1495 :=
by
  sorry

end NUMINAMATH_GPT_second_discount_percentage_l107_10773


namespace NUMINAMATH_GPT_Paco_cookies_left_l107_10707

/-
Problem: Paco had 36 cookies. He gave 14 cookies to his friend and ate 10 cookies. How many cookies did Paco have left?
Solution: Paco has 12 cookies left.

To formally state this in Lean:
-/

def initial_cookies := 36
def cookies_given_away := 14
def cookies_eaten := 10

theorem Paco_cookies_left : initial_cookies - (cookies_given_away + cookies_eaten) = 12 :=
by
  sorry

/-
This theorem states that Paco has 12 cookies left given initial conditions.
-/

end NUMINAMATH_GPT_Paco_cookies_left_l107_10707


namespace NUMINAMATH_GPT_krishan_money_l107_10740

theorem krishan_money 
  (x y : ℝ)
  (hx1 : 7 * x * 1.185 = 699.8)
  (hx2 : 10 * x * 0.8 = 800)
  (hy : 17 * x = 8 * y) : 
  16 * y = 3400 := 
by
  -- It's acceptable to leave the proof incomplete due to the focus being on the statement.
  sorry

end NUMINAMATH_GPT_krishan_money_l107_10740


namespace NUMINAMATH_GPT_number_of_sets_without_perfect_squares_l107_10754

/-- Define the set T_i of all integers n such that 200i ≤ n < 200(i + 1). -/
def T (i : ℕ) : Set ℕ := {n | 200 * i ≤ n ∧ n < 200 * (i + 1)}

/-- The total number of sets T_i from T_0 to T_{499}. -/
def total_sets : ℕ := 500

/-- The number of sets from T_0 to T_{499} that contain at least one perfect square. -/
def sets_with_perfect_squares : ℕ := 317

/-- The number of sets from T_0 to T_{499} that do not contain any perfect squares. -/
def sets_without_perfect_squares : ℕ := total_sets - sets_with_perfect_squares

/-- Proof that the number of sets T_0, T_1, T_2, ..., T_{499} that do not contain a perfect square is 183. -/
theorem number_of_sets_without_perfect_squares : sets_without_perfect_squares = 183 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sets_without_perfect_squares_l107_10754
