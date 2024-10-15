import Mathlib

namespace NUMINAMATH_GPT_fraction_of_ripe_oranges_eaten_l1495_149529

theorem fraction_of_ripe_oranges_eaten :
  ∀ (total_oranges ripe_oranges unripe_oranges uneaten_oranges eaten_unripe_oranges eaten_ripe_oranges : ℕ),
    total_oranges = 96 →
    ripe_oranges = total_oranges / 2 →
    unripe_oranges = total_oranges / 2 →
    eaten_unripe_oranges = unripe_oranges / 8 →
    uneaten_oranges = 78 →
    eaten_ripe_oranges = (total_oranges - uneaten_oranges) - eaten_unripe_oranges →
    (eaten_ripe_oranges : ℚ) / ripe_oranges = 1 / 4 :=
by
  intros total_oranges ripe_oranges unripe_oranges uneaten_oranges eaten_unripe_oranges eaten_ripe_oranges
  intros h_total h_ripe h_unripe h_eaten_unripe h_uneaten h_eaten_ripe
  sorry

end NUMINAMATH_GPT_fraction_of_ripe_oranges_eaten_l1495_149529


namespace NUMINAMATH_GPT_arithmetic_mean_15_23_37_45_l1495_149544

def arithmetic_mean (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem arithmetic_mean_15_23_37_45 :
  arithmetic_mean 15 23 37 45 = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_mean_15_23_37_45_l1495_149544


namespace NUMINAMATH_GPT_median_length_of_right_triangle_l1495_149536

theorem median_length_of_right_triangle (DE EF : ℝ) (hDE : DE = 5) (hEF : EF = 12) :
  let DF := Real.sqrt (DE^2 + EF^2)
  let N := (EF / 2)
  let DN := DF / 2
  DN = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_median_length_of_right_triangle_l1495_149536


namespace NUMINAMATH_GPT_prob_students_both_days_l1495_149556

def num_scenarios (students : ℕ) (choices : ℕ) : ℕ :=
  choices ^ students

def scenarios_sat_sun (total_scenarios : ℕ) (both_days_empty : ℕ) : ℕ :=
  total_scenarios - both_days_empty

theorem prob_students_both_days :
  let students := 3
  let choices := 2
  let total_scenarios := num_scenarios students choices
  let both_days_empty := 2 -- When all choose Saturday or all choose Sunday
  let scenarios_both := scenarios_sat_sun total_scenarios both_days_empty
  let probability := scenarios_both / total_scenarios
  probability = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_prob_students_both_days_l1495_149556


namespace NUMINAMATH_GPT_average_speed_is_five_l1495_149531

-- Define the speeds for each segment
def swimming_speed : ℝ := 2 -- km/h
def biking_speed : ℝ := 15 -- km/h
def running_speed : ℝ := 9 -- km/h
def kayaking_speed : ℝ := 6 -- km/h

-- Define the problem to prove the average speed
theorem average_speed_is_five :
  let segments := [swimming_speed, biking_speed, running_speed, kayaking_speed]
  let harmonic_mean (speeds : List ℝ) : ℝ :=
    let n := speeds.length
    n / (speeds.foldl (fun acc s => acc + 1 / s) 0)
  harmonic_mean segments = 5 := by
  sorry

end NUMINAMATH_GPT_average_speed_is_five_l1495_149531


namespace NUMINAMATH_GPT_peter_total_pizza_eaten_l1495_149535

def slices_total : Nat := 16
def peter_slices_eaten_alone : ℚ := 2 / 16
def shared_slice_total : ℚ := 1 / (3 * 16)

theorem peter_total_pizza_eaten : peter_slices_eaten_alone + shared_slice_total = 7 / 48 := by
  sorry

end NUMINAMATH_GPT_peter_total_pizza_eaten_l1495_149535


namespace NUMINAMATH_GPT_value_of_F_l1495_149522

   variables (B G P Q F : ℕ)

   -- Define the main hypothesis stating that the total lengths of the books are equal.
   def fill_shelf := 
     (∃ d a : ℕ, d = B * a + 2 * G * a ∧ d = P * a + 2 * Q * a ∧ d = F * a)

   -- Prove that F equals B + 2G and P + 2Q under the hypothesis.
   theorem value_of_F (h : fill_shelf B G P Q F) : F = B + 2 * G ∧ F = P + 2 * Q :=
   sorry
   
end NUMINAMATH_GPT_value_of_F_l1495_149522


namespace NUMINAMATH_GPT_correct_divisor_l1495_149519

-- Definitions of variables and conditions
variables (X D : ℕ)

-- Stating the theorem
theorem correct_divisor (h1 : X = 49 * 12) (h2 : X = 28 * D) : D = 21 :=
by
  sorry

end NUMINAMATH_GPT_correct_divisor_l1495_149519


namespace NUMINAMATH_GPT_gathering_gift_exchange_l1495_149574

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end NUMINAMATH_GPT_gathering_gift_exchange_l1495_149574


namespace NUMINAMATH_GPT_drawings_per_neighbor_l1495_149546

theorem drawings_per_neighbor (n_neighbors animals : ℕ) (h1 : n_neighbors = 6) (h2 : animals = 54) : animals / n_neighbors = 9 :=
by
  sorry

end NUMINAMATH_GPT_drawings_per_neighbor_l1495_149546


namespace NUMINAMATH_GPT_distinct_values_for_D_l1495_149555

-- Define distinct digits
def distinct_digits (a b c d e : ℕ) :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10

-- Declare the problem statement
theorem distinct_values_for_D : 
  ∃ D_values : Finset ℕ, 
    (∀ (A B C D E : ℕ), 
      distinct_digits A B C D E → 
      E + C = D ∧
      B + C = E ∧
      B + D = E) →
    D_values.card = 7 := 
by 
  sorry

end NUMINAMATH_GPT_distinct_values_for_D_l1495_149555


namespace NUMINAMATH_GPT_arith_seq_sum_signs_l1495_149569

variable {α : Type*} [LinearOrderedField α]
variable {a : ℕ → α} {S : ℕ → α} {d : α}

noncomputable def is_arith_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  n * (a 1 + a n) / 2

-- Given conditions
variable (a_8_neg : a 8 < 0)
variable (a_9_pos : a 9 > 0)
variable (a_9_greater_abs_a_8 : a 9 > abs (a 8))

-- The theorem to prove
theorem arith_seq_sum_signs (h : is_arith_seq a) :
  (∀ n, n ≤ 15 → sum_first_n_terms a n < 0) ∧ (∀ n, n ≥ 16 → sum_first_n_terms a n > 0) :=
sorry

end NUMINAMATH_GPT_arith_seq_sum_signs_l1495_149569


namespace NUMINAMATH_GPT_math_students_but_not_science_l1495_149584

theorem math_students_but_not_science (total_students : ℕ) (students_math : ℕ) (students_science : ℕ)
  (students_both : ℕ) (students_math_three_times : ℕ) :
  total_students = 30 ∧ students_both = 2 ∧ students_math = 3 * students_science ∧ 
  students_math = students_both + (22 - 2) → (students_math - students_both = 20) :=
by
  sorry

end NUMINAMATH_GPT_math_students_but_not_science_l1495_149584


namespace NUMINAMATH_GPT_num_integer_pairs_l1495_149594

theorem num_integer_pairs (m n : ℤ) :
  0 < m ∧ m < n ∧ n < 53 ∧ 53^2 + m^2 = 52^2 + n^2 →
  ∃ k, k = 3 := 
sorry

end NUMINAMATH_GPT_num_integer_pairs_l1495_149594


namespace NUMINAMATH_GPT_general_formula_a_S_n_no_arithmetic_sequence_in_b_l1495_149561

def sequence_a (a : ℕ → ℚ) :=
  (a 1 = 1 / 4) ∧ (∀ n : ℕ, n > 0 → 3 * a (n + 1) - 2 * a n = 1)

def sequence_b (b : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n : ℕ, n > 0 → b n = a (n + 1) - a n

theorem general_formula_a_S_n (a : ℕ → ℚ) (S : ℕ → ℚ) :
  sequence_a a →
  (∀ n : ℕ, n > 0 → a n = 1 - (3 / 4) * (2 / 3)^(n - 1)) →
  (∀ n : ℕ, n > 0 → S n = (2 / 3)^(n - 2) + n - 9 / 4) →
  True := sorry

theorem no_arithmetic_sequence_in_b (b : ℕ → ℚ) (a : ℕ → ℚ) :
  sequence_b b a →
  (∀ n : ℕ, n > 0 → b n = (1 / 4) * (2 / 3)^(n - 1)) →
  (∀ r s t : ℕ, r < s ∧ s < t → ¬ (b s - b r = b t - b s)) :=
  sorry

end NUMINAMATH_GPT_general_formula_a_S_n_no_arithmetic_sequence_in_b_l1495_149561


namespace NUMINAMATH_GPT_percentage_runs_by_running_l1495_149557

theorem percentage_runs_by_running 
  (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) 
  (runs_per_boundary : ℕ) (runs_per_six : ℕ)
  (H_total_runs : total_runs = 120)
  (H_boundaries : boundaries = 3)
  (H_sixes : sixes = 8)
  (H_runs_per_boundary : runs_per_boundary = 4)
  (H_runs_per_six : runs_per_six = 6) :
  ((total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs : ℚ) * 100 = 50 := 
by
  sorry

end NUMINAMATH_GPT_percentage_runs_by_running_l1495_149557


namespace NUMINAMATH_GPT_final_amount_simple_interest_l1495_149521

theorem final_amount_simple_interest (P R T : ℕ) (hP : P = 12500) (hR : R = 6) (hT : T = 4) : 
  P + (P * R * T) / 100 = 13250 :=
by
  rw [hP, hR, hT]
  norm_num
  sorry

end NUMINAMATH_GPT_final_amount_simple_interest_l1495_149521


namespace NUMINAMATH_GPT_perimeter_of_new_figure_is_correct_l1495_149586

-- Define the given conditions
def original_horizontal_segments := 16
def original_vertical_segments := 10
def original_side_length := 1
def new_side_length := 2

-- Define total lengths calculations
def total_horizontal_length (new_side_length original_horizontal_segments : ℕ) : ℕ :=
  original_horizontal_segments * new_side_length

def total_vertical_length (new_side_length original_vertical_segments : ℕ) : ℕ :=
  original_vertical_segments * new_side_length

-- Formulate the main theorem
theorem perimeter_of_new_figure_is_correct :
  total_horizontal_length new_side_length original_horizontal_segments + 
  total_vertical_length new_side_length original_vertical_segments = 52 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_new_figure_is_correct_l1495_149586


namespace NUMINAMATH_GPT_smallest_integer_k_l1495_149583

theorem smallest_integer_k (k : ℤ) : k > 2 ∧ k % 19 = 2 ∧ k % 7 = 2 ∧ k % 4 = 2 ↔ k = 534 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_k_l1495_149583


namespace NUMINAMATH_GPT_note_relationship_l1495_149593

theorem note_relationship
  (x y z : ℕ) 
  (h1 : x + 5 * y + 10 * z = 480)
  (h2 : x + y + z = 90)
  (h3 : y = 2 * x)
  (h4 : z = 3 * x) : 
  x = 15 ∧ y = 30 ∧ z = 45 :=
by 
  sorry

end NUMINAMATH_GPT_note_relationship_l1495_149593


namespace NUMINAMATH_GPT_smallest_m_plus_n_l1495_149506

theorem smallest_m_plus_n (m n : ℕ) (h1 : m > n) (h2 : n ≥ 1) 
(h3 : 1000 ∣ 1978^m - 1978^n) : m + n = 106 :=
sorry

end NUMINAMATH_GPT_smallest_m_plus_n_l1495_149506


namespace NUMINAMATH_GPT_min_direction_changes_l1495_149598

theorem min_direction_changes (n : ℕ) : 
  ∀ (path : Finset (ℕ × ℕ)), 
    (path.card = (n + 1) * (n + 2) / 2) → 
    (∀ (v : ℕ × ℕ), v ∈ path) →
    ∃ changes, (changes ≥ n) :=
by sorry

end NUMINAMATH_GPT_min_direction_changes_l1495_149598


namespace NUMINAMATH_GPT_horse_food_per_day_l1495_149592

-- Given conditions
def sheep_count : ℕ := 48
def horse_food_total : ℕ := 12880
def sheep_horse_ratio : ℚ := 6 / 7

-- Definition of the number of horses based on the ratio
def horse_count : ℕ := (sheep_count * 7) / 6

-- Statement to prove: each horse needs 230 ounces of food per day
theorem horse_food_per_day : horse_food_total / horse_count = 230 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_horse_food_per_day_l1495_149592


namespace NUMINAMATH_GPT_tile_count_difference_l1495_149579

theorem tile_count_difference (W : ℕ) (B : ℕ) (B' : ℕ) (added_black_tiles : ℕ)
  (hW : W = 16) (hB : B = 9) (h_add : added_black_tiles = 8) (hB' : B' = B + added_black_tiles) :
  B' - W = 1 :=
by
  sorry

end NUMINAMATH_GPT_tile_count_difference_l1495_149579


namespace NUMINAMATH_GPT_marbles_left_l1495_149595

def initial_marbles : ℝ := 9.0
def given_marbles : ℝ := 3.0

theorem marbles_left : initial_marbles - given_marbles = 6.0 := 
by
  sorry

end NUMINAMATH_GPT_marbles_left_l1495_149595


namespace NUMINAMATH_GPT_sum_is_correct_l1495_149528

noncomputable def calculate_sum : ℚ :=
  (4 / 3) + (13 / 9) + (40 / 27) + (121 / 81) - (8 / 3)

theorem sum_is_correct : calculate_sum = 171 / 81 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_is_correct_l1495_149528


namespace NUMINAMATH_GPT_original_wage_l1495_149525

theorem original_wage (W : ℝ) (h : 1.5 * W = 42) : W = 28 :=
by
  sorry

end NUMINAMATH_GPT_original_wage_l1495_149525


namespace NUMINAMATH_GPT_seconds_in_part_of_day_l1495_149587

theorem seconds_in_part_of_day : (1 / 4) * (1 / 6) * (1 / 8) * 24 * 60 * 60 = 450 := by
  sorry

end NUMINAMATH_GPT_seconds_in_part_of_day_l1495_149587


namespace NUMINAMATH_GPT_profit_in_december_l1495_149567

variable (a : ℝ)

theorem profit_in_december (h_a: a > 0):
  (1 - 0.06) * (1 + 0.10) * a = (1 - 0.06) * (1 + 0.10) * a :=
by
  sorry

end NUMINAMATH_GPT_profit_in_december_l1495_149567


namespace NUMINAMATH_GPT_min_max_expression_l1495_149551

theorem min_max_expression (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 19) (h2 : b^2 + b * c + c^2 = 19) :
  ∃ (min_val max_val : ℝ), 
    min_val = 0 ∧ max_val = 57 ∧ 
    (∀ x, x = c^2 + c * a + a^2 → min_val ≤ x ∧ x ≤ max_val) :=
by sorry

end NUMINAMATH_GPT_min_max_expression_l1495_149551


namespace NUMINAMATH_GPT_contradiction_proof_l1495_149511

theorem contradiction_proof :
  ∀ (a b c d : ℝ),
    a + b = 1 →
    c + d = 1 →
    ac + bd > 1 →
    (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) :=
by
  sorry

end NUMINAMATH_GPT_contradiction_proof_l1495_149511


namespace NUMINAMATH_GPT_quadratic_max_m_l1495_149523

theorem quadratic_max_m (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (m * x^2 - 2 * m * x + 2) ≤ 4) ∧ 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ (m * x^2 - 2 * m * x + 2) = 4) ∧ 
  m ≠ 0 → 
  (m = 2 / 3 ∨ m = -2) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_max_m_l1495_149523


namespace NUMINAMATH_GPT_distinct_students_count_l1495_149530

open Set

theorem distinct_students_count 
  (germain_students : ℕ := 15) 
  (newton_students : ℕ := 12) 
  (young_students : ℕ := 9)
  (overlap_students : ℕ := 3) :
  (germain_students + newton_students + young_students - overlap_students) = 33 := 
by
  sorry

end NUMINAMATH_GPT_distinct_students_count_l1495_149530


namespace NUMINAMATH_GPT_solve_system_of_equations_l1495_149596

theorem solve_system_of_equations :
  ∃ x y : ℝ, (3 * x - 5 * y = -1.5) ∧ (7 * x + 2 * y = 4.7) ∧ x = 0.5 ∧ y = 0.6 :=
by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_solve_system_of_equations_l1495_149596


namespace NUMINAMATH_GPT_inappropriate_expression_is_D_l1495_149542

-- Definitions of each expression as constants
def expr_A : String := "Recently, I have had the honor to read your masterpiece, and I felt enlightened."
def expr_B : String := "Your visit has brought glory to my humble abode."
def expr_C : String := "It's the first time you honor my place with a visit, and I apologize for any lack of hospitality."
def expr_D : String := "My mother has been slightly unwell recently, I hope you won't bother her."

-- Definition of the problem context
def is_inappropriate (expr : String) : Prop := 
  expr = expr_D

-- The theorem statement
theorem inappropriate_expression_is_D : is_inappropriate expr_D := 
by
  sorry

end NUMINAMATH_GPT_inappropriate_expression_is_D_l1495_149542


namespace NUMINAMATH_GPT_solve_equation_l1495_149599

-- Definitions based on the conditions
def equation (x : ℝ) : Prop :=
  1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 5) + 1 / (x^2 - 17*x - 10) = 0

-- Theorem stating that the solutions of the given equation are the expected values
theorem solve_equation :
  {x : ℝ | equation x} = {-2 + 2 * Real.sqrt 14, -2 - 2 * Real.sqrt 14, (7 + Real.sqrt 89) / 2, (7 - Real.sqrt 89) / 2} :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1495_149599


namespace NUMINAMATH_GPT_count_triples_satisfying_conditions_l1495_149503

theorem count_triples_satisfying_conditions :
  (∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ ab + bc = 72 ∧ ac + bc = 35) → 
  ∃! t : (ℕ × ℕ × ℕ), 0 < t.1 ∧ 0 < t.2.1 ∧ 0 < t.2.2 ∧ 
                     t.1 * t.2.1 + t.2.1 * t.2.2 = 72 ∧ 
                     t.1 * t.2.2 + t.2.1 * t.2.2 = 35 :=
by sorry

end NUMINAMATH_GPT_count_triples_satisfying_conditions_l1495_149503


namespace NUMINAMATH_GPT_m_value_for_perfect_square_l1495_149541

theorem m_value_for_perfect_square (m : ℤ) (x y : ℤ) :
  (∃ k : ℤ, 4 * x^2 - m * x * y + 9 * y^2 = k^2) → m = 12 ∨ m = -12 :=
by
  sorry

end NUMINAMATH_GPT_m_value_for_perfect_square_l1495_149541


namespace NUMINAMATH_GPT_total_students_shook_hands_l1495_149502

theorem total_students_shook_hands (S3 S2 S1 : ℕ) (h1 : S3 = 200) (h2 : S2 = S3 + 40) (h3 : S1 = 2 * S2) : 
  S1 + S2 + S3 = 920 :=
by
  sorry

end NUMINAMATH_GPT_total_students_shook_hands_l1495_149502


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_l1495_149543

theorem arithmetic_sequence_a6 {a : ℕ → ℤ}
  (h1 : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h2 : a 2 + a 8 = 16)
  (h3 : a 4 = 6) :
  a 6 = 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_l1495_149543


namespace NUMINAMATH_GPT_distance_to_school_l1495_149559

variable (T D : ℕ)

/-- Given the conditions, prove the distance from the child's home to the school is 630 meters --/
theorem distance_to_school :
  (5 * (T + 6) = D) →
  (7 * (T - 30) = D) →
  D = 630 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_distance_to_school_l1495_149559


namespace NUMINAMATH_GPT_famous_quote_author_l1495_149563

-- conditions
def statement_date := "July 20, 1969"
def mission := "Apollo 11"
def astronauts := ["Neil Armstrong", "Buzz Aldrin", "Michael Collins"]
def first_to_moon := "Neil Armstrong"

-- goal
theorem famous_quote_author : (statement_date = "July 20, 1969") ∧ (mission = "Apollo 11") ∧ (astronauts = ["Neil Armstrong", "Buzz Aldrin", "Michael Collins"]) ∧ (first_to_moon = "Neil Armstrong") → "Neil Armstrong" = "Neil Armstrong" :=
by 
  intros _; 
  exact rfl

end NUMINAMATH_GPT_famous_quote_author_l1495_149563


namespace NUMINAMATH_GPT_derivative_f_l1495_149547

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_f (x : ℝ) : deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by
  sorry

end NUMINAMATH_GPT_derivative_f_l1495_149547


namespace NUMINAMATH_GPT_king_chessboard_strategy_king_chessboard_strategy_odd_l1495_149576

theorem king_chessboard_strategy (m n : ℕ) : 
  (m * n) % 2 = 0 → (∀ p, p < m * n → ∃ p', p' < m * n ∧ p' ≠ p) := 
sorry

theorem king_chessboard_strategy_odd (m n : ℕ) : 
  (m * n) % 2 = 1 → (∀ p, p < m * n → ∃ p', p' < m * n ∧ p' ≠ p) :=
sorry

end NUMINAMATH_GPT_king_chessboard_strategy_king_chessboard_strategy_odd_l1495_149576


namespace NUMINAMATH_GPT_arithmetic_seq_common_difference_l1495_149571

theorem arithmetic_seq_common_difference (a1 d : ℝ) (h1 : a1 + 2 * d = 10) (h2 : 4 * a1 + 6 * d = 36) : d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_common_difference_l1495_149571


namespace NUMINAMATH_GPT_arrangements_count_l1495_149554

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of positions
def num_positions : ℕ := 3

-- Define a type for the students
inductive Student
| A | B | C | D | E

-- Define the positions
inductive Position
| athletics | swimming | ball_games

-- Constraint: student A cannot be the swimming volunteer
def cannot_be_swimming_volunteer (s : Student) (p : Position) : Prop :=
  (s = Student.A → p ≠ Position.swimming)

-- Define the function to count the arrangements given the constraints
noncomputable def count_arrangements : ℕ :=
  (num_students.choose num_positions) - 1 -- Placeholder for the actual count based on given conditions

-- The theorem statement
theorem arrangements_count : count_arrangements = 16 :=
by
  sorry

end NUMINAMATH_GPT_arrangements_count_l1495_149554


namespace NUMINAMATH_GPT_feathers_before_crossing_road_l1495_149572

theorem feathers_before_crossing_road : 
  ∀ (F : ℕ), 
  (F - (2 * 23) = 5217) → 
  F = 5263 :=
by
  intros F h
  sorry

end NUMINAMATH_GPT_feathers_before_crossing_road_l1495_149572


namespace NUMINAMATH_GPT_minimum_d_value_l1495_149509

theorem minimum_d_value :
  let d := (1 + 2 * Real.sqrt 10) / 3
  let distance := Real.sqrt ((2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2)
  distance = 4 * d :=
by
  let d := (1 + 2 * Real.sqrt 10) / 3
  let distance := Real.sqrt ((2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2)
  sorry

end NUMINAMATH_GPT_minimum_d_value_l1495_149509


namespace NUMINAMATH_GPT_cos_double_angle_l1495_149573

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 1 / 2) : Real.cos (2 * α) = 3 / 5 :=
by sorry

end NUMINAMATH_GPT_cos_double_angle_l1495_149573


namespace NUMINAMATH_GPT_medicine_types_count_l1495_149534

theorem medicine_types_count (n : ℕ) (hn : n = 5) : (Nat.choose n 2 = 10) :=
by
  sorry

end NUMINAMATH_GPT_medicine_types_count_l1495_149534


namespace NUMINAMATH_GPT_expression_evaluation_l1495_149552

theorem expression_evaluation : 1 + 3 + 5 + 7 - (2 + 4 + 6) + 3^2 + 5^2 = 38 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1495_149552


namespace NUMINAMATH_GPT_curve_not_parabola_l1495_149550

theorem curve_not_parabola (k : ℝ) : ¬(∃ a b c : ℝ, a ≠ 0 ∧ x^2 + ky^2 = a*x^2 + b*y + c) :=
sorry

end NUMINAMATH_GPT_curve_not_parabola_l1495_149550


namespace NUMINAMATH_GPT_total_fencing_needed_l1495_149508

def width1 : ℕ := 4
def length1 : ℕ := 2 * width1 - 1

def length2 : ℕ := length1 + 3
def width2 : ℕ := width1 - 2

def width3 : ℕ := (width1 + width2) / 2
def length3 : ℚ := (length1 + length2) / 2

def perimeter (w l : ℚ) : ℚ := 2 * (w + l)

def P1 : ℚ := perimeter width1 length1
def P2 : ℚ := perimeter width2 length2
def P3 : ℚ := perimeter width3 length3

def total_fence : ℚ := P1 + P2 + P3

theorem total_fencing_needed : total_fence = 69 := 
  sorry

end NUMINAMATH_GPT_total_fencing_needed_l1495_149508


namespace NUMINAMATH_GPT_total_pictures_480_l1495_149504

noncomputable def total_pictures (pictures_per_album : ℕ) (num_albums : ℕ) : ℕ :=
  pictures_per_album * num_albums

theorem total_pictures_480 : total_pictures 20 24 = 480 :=
  by
    sorry

end NUMINAMATH_GPT_total_pictures_480_l1495_149504


namespace NUMINAMATH_GPT_votes_lost_by_l1495_149524

theorem votes_lost_by (total_votes : ℕ) (candidate_percentage : ℕ) : total_votes = 20000 → candidate_percentage = 10 → 
  (total_votes * candidate_percentage / 100 - total_votes * (100 - candidate_percentage) / 100 = 16000) :=
by
  intros h_total_votes h_candidate_percentage
  have vote_candidate := total_votes * candidate_percentage / 100
  have vote_rival := total_votes * (100 - candidate_percentage) / 100
  have votes_diff := vote_rival - vote_candidate
  rw [h_total_votes, h_candidate_percentage] at *
  sorry

end NUMINAMATH_GPT_votes_lost_by_l1495_149524


namespace NUMINAMATH_GPT_sum_of_roots_eq_neg3_l1495_149548

theorem sum_of_roots_eq_neg3
  (a b c : ℝ)
  (h_eq : 2 * x^2 + 6 * x - 1 = 0)
  (h_a : a = 2)
  (h_b : b = 6) :
  (x1 x2 : ℝ) → x1 + x2 = -b / a :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_eq_neg3_l1495_149548


namespace NUMINAMATH_GPT_christopher_more_money_l1495_149526

-- Define the conditions provided in the problem

def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64
def quarter_value : ℝ := 0.25

-- Define the question as a theorem

theorem christopher_more_money : (christopher_quarters - karen_quarters) * quarter_value = 8 :=
by sorry

end NUMINAMATH_GPT_christopher_more_money_l1495_149526


namespace NUMINAMATH_GPT_exponentiation_rule_l1495_149501

variable {a b : ℝ}

theorem exponentiation_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end NUMINAMATH_GPT_exponentiation_rule_l1495_149501


namespace NUMINAMATH_GPT_circle_parabola_intersections_l1495_149578

theorem circle_parabola_intersections : 
  ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, (p.1 ^ 2 + p.2 ^ 2 = 16) ∧ (p.2 = p.1 ^ 2 - 4)) ∧
  points.card = 3 := 
sorry

end NUMINAMATH_GPT_circle_parabola_intersections_l1495_149578


namespace NUMINAMATH_GPT_quadratic_vertex_form_l1495_149560

theorem quadratic_vertex_form (a h k x: ℝ) (h_a : a = 3) (hx : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_vertex_form_l1495_149560


namespace NUMINAMATH_GPT_find_gamma_k_l1495_149577

noncomputable def alpha (n d : ℕ) : ℕ := 1 + (n - 1) * d
noncomputable def beta (n r : ℕ) : ℕ := r^(n - 1)
noncomputable def gamma (n d r : ℕ) : ℕ := alpha n d + beta n r

theorem find_gamma_k (k d r : ℕ) (hk1 : gamma (k-1) d r = 200) (hk2 : gamma (k+1) d r = 2000) :
    gamma k d r = 387 :=
sorry

end NUMINAMATH_GPT_find_gamma_k_l1495_149577


namespace NUMINAMATH_GPT_determine_x_l1495_149518

theorem determine_x (x : ℝ) (hx : 0 < x) (h : x * ⌊x⌋ = 72) : x = 9 :=
sorry

end NUMINAMATH_GPT_determine_x_l1495_149518


namespace NUMINAMATH_GPT_max_single_player_salary_is_426000_l1495_149516

noncomputable def max_single_player_salary (total_salary_cap : ℤ) (min_salary : ℤ) (num_players : ℤ) : ℤ :=
  total_salary_cap - (num_players - 1) * min_salary

theorem max_single_player_salary_is_426000 :
  ∃ y, max_single_player_salary 800000 17000 23 = y ∧ y = 426000 :=
by
  sorry

end NUMINAMATH_GPT_max_single_player_salary_is_426000_l1495_149516


namespace NUMINAMATH_GPT_incorrect_expressions_l1495_149510

-- Definitions for the conditions
def F : ℝ := sorry   -- F represents a repeating decimal
def X : ℝ := sorry   -- X represents the t digits of F that are non-repeating
def Y : ℝ := sorry   -- Y represents the u digits of F that repeat
def t : ℕ := sorry   -- t is the number of non-repeating digits
def u : ℕ := sorry   -- u is the number of repeating digits

-- Statement that expressions (C) and (D) are incorrect
theorem incorrect_expressions : 
  ¬ (10^(t + 2 * u) * F = X + Y / 10 ^ u) ∧ ¬ (10^t * (10^u - 1) * F = Y * (X - 1)) :=
sorry

end NUMINAMATH_GPT_incorrect_expressions_l1495_149510


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1495_149568

-- Define the universal set U
def U : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 1 / x ≥ 1}

-- Define the complement of A within U
def complement_U_A : Set ℝ := {x | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_of_A_in_U : complement_U_A = {x | -1 < x ∧ x ≤ 0} :=
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1495_149568


namespace NUMINAMATH_GPT_perimeter_rectangles_l1495_149500

theorem perimeter_rectangles (a b : ℕ) (p_rect1 p_rect2 : ℕ) (p_photo : ℕ) (h1 : 2 * (a + b) = p_photo) (h2 : a + b = 10) (h3 : p_rect1 = 40) (h4 : p_rect2 = 44) : 
p_rect1 ≠ p_rect2 -> (p_rect1 = 40 ∧ p_rect2 = 44) := 
by 
  sorry

end NUMINAMATH_GPT_perimeter_rectangles_l1495_149500


namespace NUMINAMATH_GPT_insect_population_calculations_l1495_149527

theorem insect_population_calculations :
  (let ants_1 := 100
   let ants_2 := ants_1 - 20 * ants_1 / 100
   let ants_3 := ants_2 - 25 * ants_2 / 100
   let bees_1 := 150
   let bees_2 := bees_1 - 30 * bees_1 / 100
   let termites_1 := 200
   let termites_2 := termites_1 - 10 * termites_1 / 100
   ants_3 = 60 ∧ bees_2 = 105 ∧ termites_2 = 180) :=
by
  sorry

end NUMINAMATH_GPT_insect_population_calculations_l1495_149527


namespace NUMINAMATH_GPT_fifteen_power_ab_l1495_149590

theorem fifteen_power_ab (a b : ℤ) (R S : ℝ) 
  (hR : R = 3^a) 
  (hS : S = 5^b) : 
  15^(a * b) = R^b * S^a :=
by sorry

end NUMINAMATH_GPT_fifteen_power_ab_l1495_149590


namespace NUMINAMATH_GPT_solve_Mary_height_l1495_149589

theorem solve_Mary_height :
  ∃ (m s : ℝ), 
  s = 150 ∧ 
  s * 1.2 = 180 ∧ 
  m = s + (180 - s) / 2 ∧ 
  m = 165 :=
by
  sorry

end NUMINAMATH_GPT_solve_Mary_height_l1495_149589


namespace NUMINAMATH_GPT_boys_in_other_communities_l1495_149537

def percentage_of_other_communities (p_M p_H p_S : ℕ) : ℕ :=
  100 - (p_M + p_H + p_S)

def number_of_boys_other_communities (total_boys : ℕ) (percentage_other : ℕ) : ℕ :=
  (percentage_other * total_boys) / 100

theorem boys_in_other_communities (N p_M p_H p_S : ℕ) (hN : N = 650) (hpM : p_M = 44) (hpH : p_H = 28) (hpS : p_S = 10) :
  number_of_boys_other_communities N (percentage_of_other_communities p_M p_H p_S) = 117 :=
by
  -- Steps to prove the theorem would go here
  sorry

end NUMINAMATH_GPT_boys_in_other_communities_l1495_149537


namespace NUMINAMATH_GPT_sub_of_neg_l1495_149515

theorem sub_of_neg : -3 - 2 = -5 :=
by 
  sorry

end NUMINAMATH_GPT_sub_of_neg_l1495_149515


namespace NUMINAMATH_GPT_rate_per_square_meter_is_3_l1495_149507

def floor_painting_rate 
  (length : ℝ) 
  (total_cost : ℝ)
  (length_more_than_breadth_by_percentage : ℝ)
  (expected_rate : ℝ) : Prop :=
  ∃ (breadth : ℝ) (rate : ℝ),
    length = (1 + length_more_than_breadth_by_percentage / 100) * breadth ∧
    total_cost = length * breadth * rate ∧
    rate = expected_rate

-- Given conditions
theorem rate_per_square_meter_is_3 :
  floor_painting_rate 15.491933384829668 240 200 3 :=
by
  sorry

end NUMINAMATH_GPT_rate_per_square_meter_is_3_l1495_149507


namespace NUMINAMATH_GPT_books_per_continent_l1495_149591

-- Definition of the given conditions
def total_books := 488
def continents_visited := 4

-- The theorem we need to prove
theorem books_per_continent : total_books / continents_visited = 122 :=
sorry

end NUMINAMATH_GPT_books_per_continent_l1495_149591


namespace NUMINAMATH_GPT_find_common_difference_l1495_149517

-- Define the arithmetic series sum formula
def arithmetic_series_sum (a₁ : ℕ) (d : ℚ) (n : ℕ) :=
  (n / 2) * (2 * a₁ + (n - 1) * d)

-- Define the first day's production, total days, and total fabric
def first_day := 5
def total_days := 30
def total_fabric := 390

-- The proof statement
theorem find_common_difference : 
  ∃ d : ℚ, arithmetic_series_sum first_day d total_days = total_fabric ∧ d = 16 / 29 :=
by
  sorry

end NUMINAMATH_GPT_find_common_difference_l1495_149517


namespace NUMINAMATH_GPT_difference_in_profit_l1495_149581

def records := 300
def price_sammy := 4
def price_bryan_two_thirds := 6
def price_bryan_one_third := 1
def price_christine_thirty := 10
def price_christine_remaining := 3

def profit_sammy := records * price_sammy
def profit_bryan := ((records * 2 / 3) * price_bryan_two_thirds) + ((records * 1 / 3) * price_bryan_one_third)
def profit_christine := (30 * price_christine_thirty) + ((records - 30) * price_christine_remaining)

theorem difference_in_profit : 
  max profit_sammy (max profit_bryan profit_christine) - min profit_sammy (min profit_bryan profit_christine) = 190 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_profit_l1495_149581


namespace NUMINAMATH_GPT_area_of_ABCD_l1495_149545

noncomputable def quadrilateral_area (AB BC AD DC : ℝ) : ℝ :=
  let area_ABC := 1 / 2 * AB * BC
  let area_ADC := 1 / 2 * AD * DC
  area_ABC + area_ADC

theorem area_of_ABCD {AB BC AD DC AC : ℝ}
  (h1 : AC = 5)
  (h2 : AB * AB + BC * BC = 25)
  (h3 : AD * AD + DC * DC = 25)
  (h4 : AB ≠ AD)
  (h5 : BC ≠ DC) :
  quadrilateral_area AB BC AD DC = 12 :=
sorry

end NUMINAMATH_GPT_area_of_ABCD_l1495_149545


namespace NUMINAMATH_GPT_fixed_point_l1495_149538

theorem fixed_point (m : ℝ) : (2 * m - 1) * 2 - (m + 3) * 3 - (m - 11) = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_fixed_point_l1495_149538


namespace NUMINAMATH_GPT_jacoby_needs_l1495_149585

-- Given conditions
def total_goal : ℤ := 5000
def job_earnings_per_hour : ℤ := 20
def total_job_hours : ℤ := 10
def cookie_price_each : ℤ := 4
def total_cookies_sold : ℤ := 24
def lottery_ticket_cost : ℤ := 10
def lottery_winning : ℤ := 500
def gift_from_sister_one : ℤ := 500
def gift_from_sister_two : ℤ := 500

-- Total money Jacoby has so far
def current_total_money : ℤ := 
  job_earnings_per_hour * total_job_hours +
  cookie_price_each * total_cookies_sold +
  lottery_winning +
  gift_from_sister_one + gift_from_sister_two -
  lottery_ticket_cost

-- The amount Jacoby needs to reach his goal
def amount_needed : ℤ := total_goal - current_total_money

-- The main statement to be proved
theorem jacoby_needs : amount_needed = 3214 := by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_jacoby_needs_l1495_149585


namespace NUMINAMATH_GPT_divide_400_l1495_149540

theorem divide_400 (a b c d : ℕ) (h1 : a + b + c + d = 400) 
  (h2 : a + 1 = b - 2) (h3 : a + 1 = 3 * c) (h4 : a + 1 = d / 4) 
  : a = 62 ∧ b = 65 ∧ c = 21 ∧ d = 252 :=
sorry

end NUMINAMATH_GPT_divide_400_l1495_149540


namespace NUMINAMATH_GPT_sum_odd_even_50_l1495_149512

def sum_first_n_odd (n : ℕ) : ℕ := n * n

def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

theorem sum_odd_even_50 : 
  sum_first_n_odd 50 + sum_first_n_even 50 = 5050 := by
  sorry

end NUMINAMATH_GPT_sum_odd_even_50_l1495_149512


namespace NUMINAMATH_GPT_subcommittee_count_l1495_149514

theorem subcommittee_count :
  (Nat.choose 10 4) * (Nat.choose 8 3) = 11760 :=
by
  sorry

end NUMINAMATH_GPT_subcommittee_count_l1495_149514


namespace NUMINAMATH_GPT_minimum_type_A_tickets_value_of_m_l1495_149539

theorem minimum_type_A_tickets (x : ℕ) (h1 : x + (500 - x) = 500) (h2 : x ≥ 3 * (500 - x)) : x = 375 := by
  sorry

theorem value_of_m (m : ℕ) (h : 500 * (1 + (m + 10) / 100) * (m + 20) = 56000) : m = 50 := by
  sorry

end NUMINAMATH_GPT_minimum_type_A_tickets_value_of_m_l1495_149539


namespace NUMINAMATH_GPT_quilt_square_side_length_l1495_149597

theorem quilt_square_side_length (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ s : ℝ, (length * width = s * s) ∧ s = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_quilt_square_side_length_l1495_149597


namespace NUMINAMATH_GPT_trig_identity_l1495_149566

theorem trig_identity (θ : ℝ) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 :=
  sorry

end NUMINAMATH_GPT_trig_identity_l1495_149566


namespace NUMINAMATH_GPT_f_g_g_f_l1495_149513

noncomputable def f (x: ℝ) := 1 - 2 * x
noncomputable def g (x: ℝ) := x^2 + 3

theorem f_g (x : ℝ) : f (g x) = -2 * x^2 - 5 :=
by
  sorry

theorem g_f (x : ℝ) : g (f x) = 4 * x^2 - 4 * x + 4 :=
by
  sorry

end NUMINAMATH_GPT_f_g_g_f_l1495_149513


namespace NUMINAMATH_GPT_third_rectangle_area_l1495_149575

-- Definitions for dimensions of the first two rectangles
def rect1_length := 3
def rect1_width := 8

def rect2_length := 2
def rect2_width := 5

-- Total area of the first two rectangles
def total_area := (rect1_length * rect1_width) + (rect2_length * rect2_width)

-- Declaration of the theorem to be proven
theorem third_rectangle_area :
  ∃ a b : ℝ, a * b = 4 ∧ total_area + a * b = total_area + 4 :=
by
  sorry

end NUMINAMATH_GPT_third_rectangle_area_l1495_149575


namespace NUMINAMATH_GPT_total_people_present_l1495_149562

def number_of_parents : ℕ := 105
def number_of_pupils : ℕ := 698
def total_people : ℕ := number_of_parents + number_of_pupils

theorem total_people_present : total_people = 803 :=
by
  sorry

end NUMINAMATH_GPT_total_people_present_l1495_149562


namespace NUMINAMATH_GPT_total_cases_l1495_149588

def NY : ℕ := 2000
def CA : ℕ := NY / 2
def TX : ℕ := CA - 400

theorem total_cases : NY + CA + TX = 3600 :=
by
  -- use sorry placeholder to indicate the solution is omitted
  sorry

end NUMINAMATH_GPT_total_cases_l1495_149588


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1495_149532

theorem isosceles_triangle_perimeter 
    (a b : ℕ) (h_iso : a = 3 ∨ a = 5) (h_other : b = 3 ∨ b = 5) 
    (h_distinct : a ≠ b) : 
    ∃ p : ℕ, p = (3 + 3 + 5) ∨ p = (5 + 5 + 3) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1495_149532


namespace NUMINAMATH_GPT_ratio_of_houses_second_to_first_day_l1495_149558

theorem ratio_of_houses_second_to_first_day 
    (houses_day1 : ℕ)
    (houses_day2 : ℕ)
    (sales_per_house : ℕ)
    (sold_pct_day2 : ℝ) 
    (total_sales_day1 : ℕ)
    (total_sales_day2 : ℝ) :
    houses_day1 = 20 →
    sales_per_house = 2 →
    sold_pct_day2 = 0.8 →
    total_sales_day1 = houses_day1 * sales_per_house →
    total_sales_day2 = sold_pct_day2 * houses_day2 * sales_per_house →
    total_sales_day1 = total_sales_day2 →
    (houses_day2 : ℝ) / houses_day1 = 5 / 4 :=
by
    intro h1 h2 h3 h4 h5 h6
    sorry

end NUMINAMATH_GPT_ratio_of_houses_second_to_first_day_l1495_149558


namespace NUMINAMATH_GPT_probability_A_fires_proof_l1495_149533

noncomputable def probability_A_fires (prob_A : ℝ) (prob_B : ℝ) : ℝ :=
  1 / 6 + (5 / 6) * (5 / 6) * prob_A

theorem probability_A_fires_proof :
  ∀ prob_A prob_B : ℝ,
  prob_A = (1 / 6 + (5 / 6) * (5 / 6) * prob_A) →
  prob_A = 6 / 11 :=
by
  sorry

end NUMINAMATH_GPT_probability_A_fires_proof_l1495_149533


namespace NUMINAMATH_GPT_area_decreases_by_28_l1495_149564

def decrease_in_area (s h : ℤ) (h_eq : h = s + 3) : ℤ :=
  let new_area := (s - 4) * (s + 7)
  let original_area := s * h
  new_area - original_area

theorem area_decreases_by_28 (s h : ℤ) (h_eq : h = s + 3) : decrease_in_area s h h_eq = -28 :=
sorry

end NUMINAMATH_GPT_area_decreases_by_28_l1495_149564


namespace NUMINAMATH_GPT_polar_to_cartesian_correct_l1495_149553

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_cartesian_correct : polar_to_cartesian 2 (5 * Real.pi / 6) = (-Real.sqrt 3, 1) :=
by
  sorry -- We are not required to provide the proof here

end NUMINAMATH_GPT_polar_to_cartesian_correct_l1495_149553


namespace NUMINAMATH_GPT_domain_of_function_l1495_149520

theorem domain_of_function :
  {x : ℝ | 3 - x > 0 ∧ x^2 - 1 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1495_149520


namespace NUMINAMATH_GPT_minimum_prime_factorization_sum_l1495_149580

theorem minimum_prime_factorization_sum (x y a b c d : ℕ) (hx : x > 0) (hy : y > 0)
  (h : 5 * x^7 = 13 * y^17) (h_pf: x = a ^ c * b ^ d) :
  a + b + c + d = 33 :=
sorry

end NUMINAMATH_GPT_minimum_prime_factorization_sum_l1495_149580


namespace NUMINAMATH_GPT_sequence_23rd_term_is_45_l1495_149570

def sequence_game (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * n - 1 else 2 * n + 1

theorem sequence_23rd_term_is_45 :
  sequence_game 23 = 45 :=
by
  -- Proving the 23rd term in the sequence as given by the game rules
  sorry

end NUMINAMATH_GPT_sequence_23rd_term_is_45_l1495_149570


namespace NUMINAMATH_GPT_bridge_length_l1495_149565

theorem bridge_length
  (train_length : ℝ) (train_speed : ℝ) (time_taken : ℝ)
  (h_train_length : train_length = 280)
  (h_train_speed : train_speed = 18)
  (h_time_taken : time_taken = 20) : ∃ L : ℝ, L = 80 :=
by
  let distance_covered := train_speed * time_taken
  have h_distance_covered : distance_covered = 360 := by sorry
  let bridge_length := distance_covered - train_length
  have h_bridge_length : bridge_length = 80 := by sorry
  existsi bridge_length
  exact h_bridge_length

end NUMINAMATH_GPT_bridge_length_l1495_149565


namespace NUMINAMATH_GPT_hens_count_l1495_149505

theorem hens_count (H C : ℕ) (heads_eq : H + C = 44) (feet_eq : 2 * H + 4 * C = 140) : H = 18 := by
  sorry

end NUMINAMATH_GPT_hens_count_l1495_149505


namespace NUMINAMATH_GPT_shopkeeper_loss_percentage_l1495_149549

theorem shopkeeper_loss_percentage
    (CP : ℝ) (profit_rate loss_percent : ℝ) 
    (SP : ℝ := CP * (1 + profit_rate)) 
    (value_after_theft : ℝ := SP * (1 - loss_percent)) 
    (goods_loss : ℝ := 100 * (1 - (value_after_theft / CP))) :
    goods_loss = 51.6 :=
by
    sorry

end NUMINAMATH_GPT_shopkeeper_loss_percentage_l1495_149549


namespace NUMINAMATH_GPT_sheila_earning_per_hour_l1495_149582

theorem sheila_earning_per_hour :
  (252 / ((8 * 3) + (6 * 2)) = 7) := 
by
  -- Prove that sheila earns $7 per hour
  
  sorry

end NUMINAMATH_GPT_sheila_earning_per_hour_l1495_149582
