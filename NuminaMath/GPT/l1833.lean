import Mathlib

namespace NUMINAMATH_GPT_line_through_vertex_has_two_a_values_l1833_183364

-- Definitions for the line and parabola as conditions
def line_eq (a x : ℝ) : ℝ := 2 * x + a
def parabola_eq (a x : ℝ) : ℝ := x^2 + 2 * a^2

-- The proof problem
theorem line_through_vertex_has_two_a_values :
  (∃ a1 a2 : ℝ, (a1 ≠ a2) ∧ (line_eq a1 0 = parabola_eq a1 0) ∧ (line_eq a2 0 = parabola_eq a2 0)) ∧
  (∀ a : ℝ, line_eq a 0 = parabola_eq a 0 → (a = 0 ∨ a = 1/2)) :=
sorry

end NUMINAMATH_GPT_line_through_vertex_has_two_a_values_l1833_183364


namespace NUMINAMATH_GPT_complement_union_eq_zero_or_negative_l1833_183362

def U : Set ℝ := Set.univ

def P : Set ℝ := { x | x > 1 }

def Q : Set ℝ := { x | x * (x - 2) < 0 }

theorem complement_union_eq_zero_or_negative :
  (U \ (P ∪ Q)) = { x | x ≤ 0 } := by
  sorry

end NUMINAMATH_GPT_complement_union_eq_zero_or_negative_l1833_183362


namespace NUMINAMATH_GPT_Madelyn_daily_pizza_expense_l1833_183353

theorem Madelyn_daily_pizza_expense (total_expense : ℕ) (days_in_may : ℕ) 
  (h1 : total_expense = 465) (h2 : days_in_may = 31) : 
  total_expense / days_in_may = 15 := 
by
  sorry

end NUMINAMATH_GPT_Madelyn_daily_pizza_expense_l1833_183353


namespace NUMINAMATH_GPT_jennifer_book_fraction_l1833_183321

theorem jennifer_book_fraction :
  (120 - (1/5 * 120 + 1/6 * 120 + 16)) / 120 = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_jennifer_book_fraction_l1833_183321


namespace NUMINAMATH_GPT_total_rainfall_over_3_days_l1833_183382

def rainfall_sunday : ℕ := 4
def rainfall_monday : ℕ := rainfall_sunday + 3
def rainfall_tuesday : ℕ := 2 * rainfall_monday

theorem total_rainfall_over_3_days : rainfall_sunday + rainfall_monday + rainfall_tuesday = 25 := by
  sorry

end NUMINAMATH_GPT_total_rainfall_over_3_days_l1833_183382


namespace NUMINAMATH_GPT_max_abc_value_l1833_183371

theorem max_abc_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + b * c = 518) (h2 : a * b - a * c = 360) : 
  a * b * c ≤ 1008 :=
sorry

end NUMINAMATH_GPT_max_abc_value_l1833_183371


namespace NUMINAMATH_GPT_value_of_abs_m_minus_n_l1833_183357

theorem value_of_abs_m_minus_n  (m n : ℝ) (h_eq : ∀ x, (x^2 - 2 * x + m) * (x^2 - 2 * x + n) = 0)
  (h_arith_seq : ∀ x₁ x₂ x₃ x₄ : ℝ, x₁ + x₂ = 2 ∧ x₃ + x₄ = 2 ∧ x₁ = 1 / 4 ∧ x₂ = 3 / 4 ∧ x₃ = 5 / 4 ∧ x₄ = 7 / 4) :
  |m - n| = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_abs_m_minus_n_l1833_183357


namespace NUMINAMATH_GPT_special_even_diff_regular_l1833_183361

def first_n_even_sum (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

def special_even_sum (n : ℕ) : ℕ :=
  let sum_cubes := (n * (n + 1) / 2) ^ 2
  let sum_squares := n * (n + 1) * (2 * n + 1) / 6
  2 * (sum_cubes + sum_squares)

theorem special_even_diff_regular : 
  let n := 100
  special_even_sum n - first_n_even_sum n = 51403900 :=
by
  sorry

end NUMINAMATH_GPT_special_even_diff_regular_l1833_183361


namespace NUMINAMATH_GPT_unit_digit_product_l1833_183329

-- Definition of unit digit function
def unit_digit (n : Nat) : Nat := n % 10

-- Conditions about unit digits of given powers
lemma unit_digit_3_pow_68 : unit_digit (3 ^ 68) = 1 := by sorry
lemma unit_digit_6_pow_59 : unit_digit (6 ^ 59) = 6 := by sorry
lemma unit_digit_7_pow_71 : unit_digit (7 ^ 71) = 3 := by sorry

-- Main statement
theorem unit_digit_product : unit_digit (3 ^ 68 * 6 ^ 59 * 7 ^ 71) = 8 := by
  have h3 := unit_digit_3_pow_68
  have h6 := unit_digit_6_pow_59
  have h7 := unit_digit_7_pow_71
  sorry

end NUMINAMATH_GPT_unit_digit_product_l1833_183329


namespace NUMINAMATH_GPT_ratio_of_c_and_d_l1833_183348

theorem ratio_of_c_and_d
  (x y c d : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c)
  (h2 : 9 * y - 12 * x = d) :
  c / d = -2 / 3 := 
  sorry

end NUMINAMATH_GPT_ratio_of_c_and_d_l1833_183348


namespace NUMINAMATH_GPT_eq_has_one_integral_root_l1833_183370

theorem eq_has_one_integral_root :
  ∀ x : ℝ, (x - (9 / (x - 5)) = 4 - (9 / (x-5))) → x = 4 := by
  intros x h
  sorry

end NUMINAMATH_GPT_eq_has_one_integral_root_l1833_183370


namespace NUMINAMATH_GPT_jack_initial_checked_plates_l1833_183386

-- Define Jack's initial and resultant plate counts
variable (C : Nat)
variable (initial_flower_plates : Nat := 4)
variable (broken_flower_plates : Nat := 1)
variable (polka_dotted_plates := 2 * C)
variable (total_plates : Nat := 27)

-- Statement of the problem
theorem jack_initial_checked_plates (h_eq : 3 + C + 2 * C = total_plates) : C = 8 :=
by
  sorry

end NUMINAMATH_GPT_jack_initial_checked_plates_l1833_183386


namespace NUMINAMATH_GPT_time_for_B_to_complete_work_l1833_183305

theorem time_for_B_to_complete_work 
  (A B C : ℝ)
  (h1 : A = 1 / 4) 
  (h2 : B + C = 1 / 3) 
  (h3 : A + C = 1 / 2) :
  1 / B = 12 :=
by
  -- Proof is omitted, as per instruction.
  sorry

end NUMINAMATH_GPT_time_for_B_to_complete_work_l1833_183305


namespace NUMINAMATH_GPT_range_of_m_l1833_183313

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem range_of_m {m : ℝ} :
  (∀ x : ℝ, (f x m = 0) → (∃ y z : ℝ, y ≠ z ∧ f y m = 0 ∧ f z m = 0)) ∧
  (∀ x : ℝ, f (1 - x) m ≥ -1)
  → (0 ≤ m ∧ m < 1) := 
sorry

end NUMINAMATH_GPT_range_of_m_l1833_183313


namespace NUMINAMATH_GPT_terminal_side_in_first_quadrant_l1833_183327

noncomputable def theta := -5

def in_first_quadrant (θ : ℝ) : Prop :=
  by sorry

theorem terminal_side_in_first_quadrant : in_first_quadrant theta := 
  by sorry

end NUMINAMATH_GPT_terminal_side_in_first_quadrant_l1833_183327


namespace NUMINAMATH_GPT_growth_comparison_l1833_183380

theorem growth_comparison (x : ℝ) (h : ℝ) (hx : x > 0) : 
  (0 < x ∧ x < 1 / 2 → (x + h) - x > ((x + h)^2 - x^2)) ∧
  (x > 1 / 2 → ((x + h)^2 - x^2) > (x + h) - x) :=
by
  sorry

end NUMINAMATH_GPT_growth_comparison_l1833_183380


namespace NUMINAMATH_GPT_Sam_weight_l1833_183387

theorem Sam_weight :
  ∃ (sam_weight : ℕ), (∀ (tyler_weight : ℕ), (∀ (peter_weight : ℕ), peter_weight = 65 → tyler_weight = 2 * peter_weight → tyler_weight = sam_weight + 25 → sam_weight = 105)) :=
by {
    sorry
}

end NUMINAMATH_GPT_Sam_weight_l1833_183387


namespace NUMINAMATH_GPT_find_phi_l1833_183369

theorem find_phi :
  ∀ φ : ℝ, 0 < φ ∧ φ < 90 → 
    (∃θ : ℝ, θ = 144 ∧ θ = 2 * φ ∧ (144 - θ) = 72) → φ = 81 :=
by
  intros φ h1 h2
  sorry

end NUMINAMATH_GPT_find_phi_l1833_183369


namespace NUMINAMATH_GPT_percent_of_rectangle_area_inside_square_l1833_183392

theorem percent_of_rectangle_area_inside_square
  (s : ℝ)  -- Let the side length of the square be \( s \).
  (width : ℝ) (length: ℝ)
  (h1 : width = 3 * s)  -- The width of the rectangle is \( 3s \).
  (h2 : length = 2 * width) :  -- The length of the rectangle is \( 2 * width \).
  (s^2 / (length * width)) * 100 = 5.56 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_rectangle_area_inside_square_l1833_183392


namespace NUMINAMATH_GPT_hot_dog_cost_l1833_183344

variable {Real : Type} [LinearOrderedField Real]

-- Define the cost of a hamburger and a hot dog
variables (h d : Real)

-- Arthur's buying conditions
def condition1 := 3 * h + 4 * d = 10
def condition2 := 2 * h + 3 * d = 7

-- Problem statement: Proving that the cost of a hot dog is 1 dollar
theorem hot_dog_cost
    (h d : Real)
    (hc1 : condition1 h d)
    (hc2 : condition2 h d) : 
    d = 1 :=
sorry

end NUMINAMATH_GPT_hot_dog_cost_l1833_183344


namespace NUMINAMATH_GPT_remainder_x_squared_div_25_l1833_183309

theorem remainder_x_squared_div_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 0 [ZMOD 25] :=
sorry

end NUMINAMATH_GPT_remainder_x_squared_div_25_l1833_183309


namespace NUMINAMATH_GPT_original_deck_size_l1833_183349

-- Define the conditions
def boys_kept_away (remaining_cards kept_away_cards : ℕ) : Prop :=
  remaining_cards + kept_away_cards = 52

-- Define the problem
theorem original_deck_size (remaining_cards : ℕ) (kept_away_cards := 2) :
  boys_kept_away remaining_cards kept_away_cards → remaining_cards + kept_away_cards = 52 :=
by
  intro h
  exact h

end NUMINAMATH_GPT_original_deck_size_l1833_183349


namespace NUMINAMATH_GPT_cube_side_length_l1833_183300

-- Given definitions and conditions
variables (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

-- Statement of the theorem
theorem cube_side_length (x : ℝ) : 
  ( ∃ (y z : ℝ), 
      y + x + z = c ∧ 
      x + z = c * a / b ∧
      y = c * x / b ∧
      z = c * x / a 
  ) → x = a * b * c / (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_GPT_cube_side_length_l1833_183300


namespace NUMINAMATH_GPT_solution_correct_l1833_183372

-- Conditions of the problem
variable (f : ℝ → ℝ)
variable (h_f_domain : ∀ (x : ℝ), 0 < x → 0 < f x)
variable (h_f_eq : ∀ (x y : ℝ), 0 < x → 0 < y → f x * f (y * f x) = f (x + y))

-- Correct answer to be proven
theorem solution_correct :
  ∃ b : ℝ, 0 ≤ b ∧ ∀ t : ℝ, 0 < t → f t = 1 / (1 + b * t) :=
sorry

end NUMINAMATH_GPT_solution_correct_l1833_183372


namespace NUMINAMATH_GPT_determine_weights_of_balls_l1833_183376

theorem determine_weights_of_balls (A B C D E m1 m2 m3 m4 m5 m6 m7 m8 m9 : ℝ)
  (h1 : m1 = A)
  (h2 : m2 = B)
  (h3 : m3 = C)
  (h4 : m4 = A + D)
  (h5 : m5 = A + E)
  (h6 : m6 = B + D)
  (h7 : m7 = B + E)
  (h8 : m8 = C + D)
  (h9 : m9 = C + E) :
  ∃ (A' B' C' D' E' : ℝ), 
    ((A' = A ∨ B' = B ∨ C' = C ∨ D' = D ∨ E' = E) ∧
     (A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ A' ≠ E' ∧
      B' ≠ C' ∧ B' ≠ D' ∧ B' ≠ E' ∧
      C' ≠ D' ∧ C' ≠ E' ∧
      D' ≠ E')) :=
sorry

end NUMINAMATH_GPT_determine_weights_of_balls_l1833_183376


namespace NUMINAMATH_GPT_max_xy_l1833_183314

open Real

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eqn : x + 4 * y = 4) :
  ∃ y : ℝ, (x = 4 - 4 * y) → y = 1 / 2 → x * y = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_xy_l1833_183314


namespace NUMINAMATH_GPT_find_a_range_l1833_183335

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else (a + 1) / x

theorem find_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → 
  - (7 / 2) ≤ a ∧ a ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_range_l1833_183335


namespace NUMINAMATH_GPT_indeterminate_original_value_percentage_l1833_183310

-- Lets define the problem as a structure with the given conditions
structure StockData where
  yield_percent : ℚ
  market_value : ℚ

-- We need to prove this condition
theorem indeterminate_original_value_percentage (d : StockData) :
  d.yield_percent = 8 ∧ d.market_value = 125 → false :=
by
  sorry

end NUMINAMATH_GPT_indeterminate_original_value_percentage_l1833_183310


namespace NUMINAMATH_GPT_original_grape_jelly_beans_l1833_183346

namespace JellyBeans

-- Definition of the problem conditions
variables (g c : ℕ)
axiom h1 : g = 3 * c
axiom h2 : g - 15 = 5 * (c - 5)

-- Proof goal statement
theorem original_grape_jelly_beans : g = 15 :=
by
  sorry

end JellyBeans

end NUMINAMATH_GPT_original_grape_jelly_beans_l1833_183346


namespace NUMINAMATH_GPT_problem_sol_l1833_183373

open Complex

theorem problem_sol (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : (a + i) / i = 1 + b * i) : a + b = 0 :=
sorry

end NUMINAMATH_GPT_problem_sol_l1833_183373


namespace NUMINAMATH_GPT_solve_coin_problem_l1833_183363

def coin_problem : Prop :=
  ∃ (x y z : ℕ), 
  1 * x + 2 * y + 5 * z = 71 ∧ 
  x = y ∧ 
  x + y + z = 31 ∧ 
  x = 12 ∧ 
  y = 12 ∧ 
  z = 7

theorem solve_coin_problem : coin_problem :=
  sorry

end NUMINAMATH_GPT_solve_coin_problem_l1833_183363


namespace NUMINAMATH_GPT_evaluate_expression_l1833_183366

-- Defining the primary condition
def condition (x : ℝ) : Prop := x > 3

-- Definition of the expression we need to evaluate
def expression (x : ℝ) : ℝ := abs (1 - abs (x - 3))

-- Stating the theorem
theorem evaluate_expression (x : ℝ) (h : condition x) : expression x = abs (4 - x) := 
by 
  -- Since the problem only asks for the statement, the proof is left as sorry.
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1833_183366


namespace NUMINAMATH_GPT_three_digit_numbers_not_multiple_of_3_5_7_l1833_183347

theorem three_digit_numbers_not_multiple_of_3_5_7 : 
  let total_three_digit_numbers := 900
  let multiples_of_3 := (999 - 100) / 3 + 1
  let multiples_of_5 := (995 - 100) / 5 + 1
  let multiples_of_7 := (994 - 105) / 7 + 1
  let multiples_of_15 := (990 - 105) / 15 + 1
  let multiples_of_21 := (987 - 105) / 21 + 1
  let multiples_of_35 := (980 - 105) / 35 + 1
  let multiples_of_105 := (945 - 105) / 105 + 1
  let total_multiples := multiples_of_3 + multiples_of_5 + multiples_of_7 - multiples_of_15 - multiples_of_21 - multiples_of_35 + multiples_of_105
  let non_multiples_total := total_three_digit_numbers - total_multiples
  non_multiples_total = 412 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_not_multiple_of_3_5_7_l1833_183347


namespace NUMINAMATH_GPT_students_with_uncool_parents_l1833_183319

def total_students : ℕ := 40
def cool_dads_count : ℕ := 18
def cool_moms_count : ℕ := 20
def both_cool_count : ℕ := 10

theorem students_with_uncool_parents :
  total_students - (cool_dads_count + cool_moms_count - both_cool_count) = 12 :=
by sorry

end NUMINAMATH_GPT_students_with_uncool_parents_l1833_183319


namespace NUMINAMATH_GPT_pond_length_l1833_183330

theorem pond_length (
    W L P : ℝ) 
    (h1 : L = 2 * W) 
    (h2 : L = 32) 
    (h3 : (L * W) / 8 = P^2) : 
  P = 8 := 
by 
  sorry

end NUMINAMATH_GPT_pond_length_l1833_183330


namespace NUMINAMATH_GPT_ancient_chinese_problem_l1833_183350

theorem ancient_chinese_problem (x y : ℤ) 
  (h1 : y = 8 * x - 3) 
  (h2 : y = 7 * x + 4) : 
  (y = 8 * x - 3) ∧ (y = 7 * x + 4) :=
by
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_ancient_chinese_problem_l1833_183350


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l1833_183393

noncomputable def solutions_equation1 : Set ℝ := { x | x^2 - 2 * x - 8 = 0 }
noncomputable def solutions_equation2 : Set ℝ := { x | x^2 - 2 * x - 5 = 0 }

theorem solve_equation1 :
  solutions_equation1 = {4, -2} := 
by
  sorry

theorem solve_equation2 :
  solutions_equation2 = {1 + Real.sqrt 6, 1 - Real.sqrt 6} :=
by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l1833_183393


namespace NUMINAMATH_GPT_range_of_a_l1833_183338

theorem range_of_a (a : ℝ) (h : ∀ t : ℝ, 0 < t → t ≤ 2 → (t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2)) : 
  (2 / 13) ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1833_183338


namespace NUMINAMATH_GPT_min_value_dot_product_l1833_183303

-- Side length of the square
def side_length: ℝ := 1

-- Definition of points in vector space
variables {A B C D O M N P: Type}

-- Definitions assuming standard Euclidean geometry
variables (O P : ℝ) (a b c : ℝ)

-- Points M and N on the edges AD and BC respectively, line MN passes through O
-- Point P satisfies 2 * vector OP = l * vector OA + (1-l) * vector OB
theorem min_value_dot_product (l : ℝ) (O P M N : ℝ) :
  (2 * (O + P)) = l * (O - a) + (1 - l) * (b + c) ∧
  ((O - P) * (O + P) - ((l^2 - l + 1/2) / 4) = -7/16) :=
by
  sorry

end NUMINAMATH_GPT_min_value_dot_product_l1833_183303


namespace NUMINAMATH_GPT_jimin_shared_fruits_total_l1833_183365

-- Define the quantities given in the conditions
def persimmons : ℕ := 2
def apples : ℕ := 7

-- State the theorem to be proved
theorem jimin_shared_fruits_total : persimmons + apples = 9 := by
  sorry

end NUMINAMATH_GPT_jimin_shared_fruits_total_l1833_183365


namespace NUMINAMATH_GPT_livestock_allocation_l1833_183359

theorem livestock_allocation :
  ∃ (x y z : ℕ), x + y + z = 100 ∧ 20 * x + 6 * y + z = 200 ∧ x = 5 ∧ y = 1 ∧ z = 94 :=
by
  sorry

end NUMINAMATH_GPT_livestock_allocation_l1833_183359


namespace NUMINAMATH_GPT_number_of_carbons_l1833_183397

-- Definitions of given conditions
def molecular_weight (total_c total_h total_o c_weight h_weight o_weight : ℕ) := 
    total_c * c_weight + total_h * h_weight + total_o * o_weight

-- Given values
def num_hydrogen_atoms : ℕ := 8
def num_oxygen_atoms : ℕ := 2
def molecular_wt : ℕ := 88
def atomic_weight_c : ℕ := 12
def atomic_weight_h : ℕ := 1
def atomic_weight_o : ℕ := 16

-- The theorem to be proved
theorem number_of_carbons (num_carbons : ℕ) 
    (H_hydrogen : num_hydrogen_atoms = 8)
    (H_oxygen : num_oxygen_atoms = 2)
    (H_molecular_weight : molecular_wt = 88)
    (H_atomic_weight_c : atomic_weight_c = 12)
    (H_atomic_weight_h : atomic_weight_h = 1)
    (H_atomic_weight_o : atomic_weight_o = 16) :
    molecular_weight num_carbons num_hydrogen_atoms num_oxygen_atoms atomic_weight_c atomic_weight_h atomic_weight_o = molecular_wt → 
    num_carbons = 4 :=
by
  intros h
  sorry 

end NUMINAMATH_GPT_number_of_carbons_l1833_183397


namespace NUMINAMATH_GPT_range_of_values_l1833_183339

theorem range_of_values (x : ℝ) : (x^2 - 5 * x + 6 < 0) ↔ (2 < x ∧ x < 3) :=
sorry

end NUMINAMATH_GPT_range_of_values_l1833_183339


namespace NUMINAMATH_GPT_determinant_matrix_zero_l1833_183384

theorem determinant_matrix_zero (θ φ : ℝ) : 
  Matrix.det ![
    ![0, Real.cos θ, -Real.sin θ],
    ![-Real.cos θ, 0, Real.cos φ],
    ![Real.sin θ, -Real.cos φ, 0]
  ] = 0 := by sorry

end NUMINAMATH_GPT_determinant_matrix_zero_l1833_183384


namespace NUMINAMATH_GPT_polynomial_difference_square_l1833_183375

theorem polynomial_difference_square (a : Fin 11 → ℝ) (x : ℝ) (sqrt2 : ℝ)
  (h_eq : (sqrt2 - x)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + 
          a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10) : 
  ((a 0 + a 2 + a 4 + a 6 + a 8 + a 10)^2 - (a 1 + a 3 + a 5 + a 7 + a 9)^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_difference_square_l1833_183375


namespace NUMINAMATH_GPT_probability_of_specific_cards_l1833_183328

noncomputable def probability_top_heart_second_spade_third_king 
  (deck_size : ℕ) (ranks_per_suit : ℕ) (suits : ℕ) (hearts : ℕ) (spades : ℕ) (kings : ℕ) : ℚ :=
  (hearts * spades * kings) / (deck_size * (deck_size - 1) * (deck_size - 2))

theorem probability_of_specific_cards :
  probability_top_heart_second_spade_third_king 104 26 4 26 26 8 = 169 / 34102 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_of_specific_cards_l1833_183328


namespace NUMINAMATH_GPT_polynomial_coefficients_even_or_odd_l1833_183388

-- Define the problem conditions as Lean definitions
variables {P Q : Polynomial ℤ}

-- Theorem: Given the conditions, prove the required statement
theorem polynomial_coefficients_even_or_odd
  (hP : ∀ n : ℕ, P.coeff n % 2 = 0)
  (hQ : ∀ n : ℕ, Q.coeff n % 2 = 0)
  (hProd : ¬ ∀ n : ℕ, (P * Q).coeff n % 4 = 0) :
  (∀ n : ℕ, P.coeff n % 2 = 0 ∧ ∃ k : ℕ, Q.coeff k % 2 ≠ 0) ∨
  (∀ n : ℕ, Q.coeff n % 2 = 0 ∧ ∃ k: ℕ, P.coeff k % 2 ≠ 0) :=
sorry

end NUMINAMATH_GPT_polynomial_coefficients_even_or_odd_l1833_183388


namespace NUMINAMATH_GPT_solution_inequality_l1833_183318

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

axiom odd_function (x : ℝ) : f (-x) = -f (x)
axiom increasing_function (x y : ℝ) : x < y → f x < f y

theorem solution_inequality (x : ℝ) : f (2 * x + 1) + f (x - 2) > 0 ↔ x > 1 / 3 := sorry

end NUMINAMATH_GPT_solution_inequality_l1833_183318


namespace NUMINAMATH_GPT_minimum_value_of_f_on_interval_l1833_183331

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem minimum_value_of_f_on_interval (a : ℝ) (h : ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 20) :
  a = -2 → ∃ min_val, min_val = -7 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_on_interval_l1833_183331


namespace NUMINAMATH_GPT_minimum_blue_chips_l1833_183385

theorem minimum_blue_chips (w r b : ℕ) : 
  (b ≥ w / 3) ∧ (b ≤ r / 4) ∧ (w + b ≥ 75) → b ≥ 19 :=
by sorry

end NUMINAMATH_GPT_minimum_blue_chips_l1833_183385


namespace NUMINAMATH_GPT_perfect_square_2n_plus_65_l1833_183322

theorem perfect_square_2n_plus_65 (n : ℕ) (h : n > 0) : 
  (∃ m : ℕ, m * m = 2^n + 65) → n = 4 ∨ n = 10 :=
by 
  sorry

end NUMINAMATH_GPT_perfect_square_2n_plus_65_l1833_183322


namespace NUMINAMATH_GPT_total_feet_l1833_183343

theorem total_feet (H C F : ℕ) (h1 : H + C = 48) (h2 : H = 28) :
  F = 2 * H + 4 * C → F = 136 :=
by
  -- substitute H = 28 and perform the calculations
  sorry

end NUMINAMATH_GPT_total_feet_l1833_183343


namespace NUMINAMATH_GPT_find_angle4_l1833_183306

theorem find_angle4 (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180) 
  (h2 : angle3 = angle4) 
  (h3 : angle3 + angle4 = 70) :
  angle4 = 35 := 
by 
  sorry

end NUMINAMATH_GPT_find_angle4_l1833_183306


namespace NUMINAMATH_GPT_rubles_greater_than_seven_l1833_183351

theorem rubles_greater_than_seven (x : ℕ) (h : x > 7) : ∃ a b : ℕ, x = 3 * a + 5 * b :=
sorry

end NUMINAMATH_GPT_rubles_greater_than_seven_l1833_183351


namespace NUMINAMATH_GPT_find_A_l1833_183352

theorem find_A :
  ∃ A B C : ℝ, 
  (1 : ℝ) / (x^3 - 7 * x^2 + 11 * x + 15) = 
  A / (x - 5) + B / (x + 3) + C / ((x + 3)^2) → 
  A = 1 / 64 := 
by 
  sorry

end NUMINAMATH_GPT_find_A_l1833_183352


namespace NUMINAMATH_GPT_alexis_pants_l1833_183379

theorem alexis_pants (P D : ℕ) (A_p : ℕ)
  (h1 : P + D = 13)
  (h2 : 3 * D = 18)
  (h3 : A_p = 3 * P) : A_p = 21 :=
  sorry

end NUMINAMATH_GPT_alexis_pants_l1833_183379


namespace NUMINAMATH_GPT_quadratic_solution_factoring_solution_l1833_183368

-- Define the first problem: Solve 2x^2 - 6x - 5 = 0
theorem quadratic_solution (x : ℝ) : 2 * x^2 - 6 * x - 5 = 0 ↔ x = (3 + Real.sqrt 19) / 2 ∨ x = (3 - Real.sqrt 19) / 2 :=
by
  sorry

-- Define the second problem: Solve 3x(4-x) = 2(x-4)
theorem factoring_solution (x : ℝ) : 3 * x * (4 - x) = 2 * (x - 4) ↔ x = 4 ∨ x = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_factoring_solution_l1833_183368


namespace NUMINAMATH_GPT_range_of_m_l1833_183340

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x < 3) ↔ (x / 3 < 1 - (x - 3) / 6 ∧ x < m)) → m ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1833_183340


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1833_183355

section problem

variable (m : ℝ)

-- Proposition p: The equation x^2 - 4mx + 1 = 0 has real solutions
def p : Prop := (16 * m^2 - 4) ≥ 0

-- Proposition q: There exists some x₀ ∈ ℝ such that mx₀^2 - 2x₀ - 1 > 0
def q : Prop := ∃ (x₀ : ℝ), (m * x₀^2 - 2 * x₀ - 1) > 0

-- Solution to (1): If p is true, the range of values for m
theorem problem1 (hp : p m) : m ≥ 1/2 ∨ m ≤ -1/2 := sorry

-- Solution to (2): If q is true, the range of values for m
theorem problem2 (hq : q m) : m > -1 := sorry

-- Solution to (3): If both p and q are false but either p or q is true,
-- find the range of values for m
theorem problem3 (hnp : ¬p m) (hnq : ¬q m) (hpq : p m ∨ q m) : -1 < m ∧ m < 1/2 := sorry

end problem

end NUMINAMATH_GPT_problem1_problem2_problem3_l1833_183355


namespace NUMINAMATH_GPT_domain_of_f_l1833_183390

-- The domain of the function is the set of all x such that the function is defined.
theorem domain_of_f:
  {x : ℝ | x > 3 ∧ x ≠ 4} = (Set.Ioo 3 4 ∪ Set.Ioi 4) := 
sorry

end NUMINAMATH_GPT_domain_of_f_l1833_183390


namespace NUMINAMATH_GPT_absolute_value_equation_solution_l1833_183315

-- mathematical problem representation in Lean
theorem absolute_value_equation_solution (y : ℝ) (h : |y + 2| = |y - 3|) : y = 1 / 2 :=
sorry

end NUMINAMATH_GPT_absolute_value_equation_solution_l1833_183315


namespace NUMINAMATH_GPT_quadratic_rational_root_contradiction_l1833_183396

def int_coefficients (a b c : ℤ) : Prop := true  -- Placeholder for the condition that coefficients are integers

def is_rational_root (a b c p q : ℤ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ p.gcd q = 1 ∧ a * p^2 + b * p * q + c * q^2 = 0  -- p/q is a rational root in simplest form

def ear_even (b c : ℤ) : Prop :=
  b % 2 = 0 ∨ c % 2 = 0

def assume_odd (a b c : ℤ) : Prop :=
  a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0

theorem quadratic_rational_root_contradiction (a b c p q : ℤ)
  (h1 : int_coefficients a b c)
  (h2 : a ≠ 0)
  (h3 : is_rational_root a b c p q)
  (h4 : ear_even b c) :
  assume_odd a b c :=
sorry

end NUMINAMATH_GPT_quadratic_rational_root_contradiction_l1833_183396


namespace NUMINAMATH_GPT_polygon_sides_sum_720_l1833_183391

theorem polygon_sides_sum_720 (n : ℕ) (h1 : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_sum_720_l1833_183391


namespace NUMINAMATH_GPT_fencing_cost_l1833_183341

theorem fencing_cost (w : ℝ) (h : ℝ) (p : ℝ) (cost_per_meter : ℝ) 
  (hw : h = w + 10) (perimeter : p = 220) (cost_rate : cost_per_meter = 6.5) : 
  ((p * cost_per_meter) = 1430) := by 
  sorry

end NUMINAMATH_GPT_fencing_cost_l1833_183341


namespace NUMINAMATH_GPT_third_smallest_four_digit_in_pascal_l1833_183389

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end NUMINAMATH_GPT_third_smallest_four_digit_in_pascal_l1833_183389


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_min_sum_l1833_183399

theorem arithmetic_geometric_sequence_min_sum :
  ∃ (A B C D : ℕ), 
    (C - B = B - A) ∧ 
    (C * 4 = B * 7) ∧ 
    (D * 4 = C * 7) ∧ 
    (16 ∣ B) ∧ 
    (A + B + C + D = 97) :=
by sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_min_sum_l1833_183399


namespace NUMINAMATH_GPT_probZ_eq_1_4_l1833_183354

noncomputable def probX : ℚ := 1/4
noncomputable def probY : ℚ := 1/3
noncomputable def probW : ℚ := 1/6

theorem probZ_eq_1_4 :
  let probZ : ℚ := 1 - (probX + probY + probW)
  probZ = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_probZ_eq_1_4_l1833_183354


namespace NUMINAMATH_GPT_solve_for_n_l1833_183312

theorem solve_for_n (n : ℕ) (h : 3^n * 9^n = 81^(n - 12)) : n = 48 :=
sorry

end NUMINAMATH_GPT_solve_for_n_l1833_183312


namespace NUMINAMATH_GPT_algebra_books_needed_l1833_183383

theorem algebra_books_needed (A' H' S' M' E' : ℕ) (x y : ℝ) (z : ℝ)
  (h1 : y > x)
  (h2 : A' ≠ H' ∧ A' ≠ S' ∧ A' ≠ M' ∧ A' ≠ E' ∧ H' ≠ S' ∧ H' ≠ M' ∧ H' ≠ E' ∧ S' ≠ M' ∧ S' ≠ E' ∧ M' ≠ E')
  (h3 : A' * x + H' * y = z)
  (h4 : S' * x + M' * y = z)
  (h5 : E' * x = 2 * z) :
  E' = (2 * A' * M' - 2 * S' * H') / (M' - H') :=
by
  sorry

end NUMINAMATH_GPT_algebra_books_needed_l1833_183383


namespace NUMINAMATH_GPT_half_vector_AB_l1833_183316

-- Define vectors MA and MB
def MA : ℝ × ℝ := (-2, 4)
def MB : ℝ × ℝ := (2, 6)

-- Define the proof statement 
theorem half_vector_AB : (1 / 2 : ℝ) • (MB - MA) = (2, 1) :=
by sorry

end NUMINAMATH_GPT_half_vector_AB_l1833_183316


namespace NUMINAMATH_GPT_apples_count_l1833_183301

variable (A : ℕ)

axiom h1 : 134 = 80 + 54
axiom h2 : A + 98 = 134

theorem apples_count : A = 36 :=
by
  sorry

end NUMINAMATH_GPT_apples_count_l1833_183301


namespace NUMINAMATH_GPT_bus_speed_including_stoppages_l1833_183345

-- Definitions based on conditions
def speed_excluding_stoppages : ℝ := 50 -- kmph
def stoppage_time_per_hour : ℝ := 18 -- minutes

-- Lean statement of the problem
theorem bus_speed_including_stoppages :
  (speed_excluding_stoppages * (1 - stoppage_time_per_hour / 60)) = 35 := by
  sorry

end NUMINAMATH_GPT_bus_speed_including_stoppages_l1833_183345


namespace NUMINAMATH_GPT_smallest_perimeter_is_23_l1833_183332

def is_odd_prime (n : ℕ) : Prop := Nat.Prime n ∧ n % 2 = 1

def are_consecutive_odd_primes (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧ b = a + 2 ∧ c = b + 2

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_perimeter_is_23 : 
  ∃ (a b c : ℕ), are_consecutive_odd_primes a b c ∧ satisfies_triangle_inequality a b c ∧ is_prime (a + b + c) ∧ (a + b + c) = 23 :=
by
  sorry

end NUMINAMATH_GPT_smallest_perimeter_is_23_l1833_183332


namespace NUMINAMATH_GPT_parking_cost_per_hour_l1833_183324

theorem parking_cost_per_hour (avg_cost : ℝ) (total_initial_cost : ℝ) (hours_excessive : ℝ) (total_hours : ℝ) (cost_first_2_hours : ℝ)
  (h1 : cost_first_2_hours = 9.00) 
  (h2 : avg_cost = 2.361111111111111)
  (h3 : total_hours = 9) 
  (h4 : hours_excessive = 7):
  (total_initial_cost / total_hours = avg_cost) -> 
  (total_initial_cost = cost_first_2_hours + hours_excessive * x) -> 
  x = 1.75 := 
by
  intros h5 h6
  sorry

end NUMINAMATH_GPT_parking_cost_per_hour_l1833_183324


namespace NUMINAMATH_GPT_find_locus_of_p_l1833_183377

noncomputable def locus_of_point_p (a b : ℝ) : Set (ℝ × ℝ) :=
{p | (p.snd = 0 ∧ -a < p.fst ∧ p.fst < a) ∨ (p.fst = a^2 / Real.sqrt (a^2 + b^2))}

theorem find_locus_of_p (a b : ℝ) (P : ℝ × ℝ) :
  (∃ (x0 y0: ℝ),
      P = (x0, y0) ∧
      ( ∃ (x1 y1 x2 y2 : ℝ),
        (x0 ≠ 0 ∨ y0 ≠ 0) ∧
        (x1 ≠ x2 ∨ y1 ≠ y2) ∧
        (y0 = 0 ∨ (b^2 * x0 = -a^2 * (x0 - Real.sqrt (a^2 + b^2)))) ∧
        ((y0 = 0 ∧ -a < x0 ∧ x0 < a) ∨ x0 = a^2 / Real.sqrt (a^2 + b^2)))) ↔ 
  P ∈ locus_of_point_p a b :=
sorry

end NUMINAMATH_GPT_find_locus_of_p_l1833_183377


namespace NUMINAMATH_GPT_part1_part2_l1833_183334

theorem part1 (x y : ℝ) (h1 : (1, 0) = (x, y)) (h2 : (0, 2) = (x, y)): 
    ∃ k b : ℝ, k = -2 ∧ b = 2 ∧ y = k * x + b := 
by 
  sorry

theorem part2 (m n : ℝ) (h : n = -2 * m + 2) (hm : -2 < m ∧ m ≤ 3):
    -4 ≤ n ∧ n < 6 := 
by 
  sorry

end NUMINAMATH_GPT_part1_part2_l1833_183334


namespace NUMINAMATH_GPT_find_x_given_scores_l1833_183337

theorem find_x_given_scores : 
  ∃ x : ℝ, (9.1 + 9.3 + x + 9.2 + 9.4) / 5 = 9.3 ∧ x = 9.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_given_scores_l1833_183337


namespace NUMINAMATH_GPT_find_x_l1833_183317

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition for perpendicular vectors
def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem find_x (x : ℝ) (h : perpendicular a (b x)) : x = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1833_183317


namespace NUMINAMATH_GPT_pages_for_ten_dollars_l1833_183360

theorem pages_for_ten_dollars (p c pages_per_cent : ℕ) (dollars cents : ℕ) (h1 : p = 5) (h2 : c = 10) (h3 : pages_per_cent = p / c) (h4 : dollars = 10) (h5 : cents = 100 * dollars) :
  (cents * pages_per_cent) = 500 :=
by
  sorry

end NUMINAMATH_GPT_pages_for_ten_dollars_l1833_183360


namespace NUMINAMATH_GPT_yellow_balls_count_l1833_183394

-- Definition of problem conditions
def initial_red_balls : ℕ := 16
def initial_blue_balls : ℕ := 2 * initial_red_balls
def red_balls_lost : ℕ := 6
def green_balls_given_away : ℕ := 7  -- This is not used in the calculations
def yellow_balls_bought : ℕ := 3 * red_balls_lost
def final_total_balls : ℕ := 74

-- Defining the total balls after all transactions
def remaining_red_balls : ℕ := initial_red_balls - red_balls_lost
def total_accounted_balls : ℕ := remaining_red_balls + initial_blue_balls + yellow_balls_bought

-- Lean statement to prove
theorem yellow_balls_count : yellow_balls_bought = 18 :=
by
  sorry

end NUMINAMATH_GPT_yellow_balls_count_l1833_183394


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1833_183381

theorem value_of_a_plus_b (a b : ℕ) (h1 : Real.sqrt 44 = 2 * Real.sqrt a) (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : a + b = 17 := 
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1833_183381


namespace NUMINAMATH_GPT_problem_statement_l1833_183358

noncomputable def f (x : ℝ) : ℝ := x / Real.cos x

theorem problem_statement (x1 x2 x3 : ℝ) (h1 : abs x1 < Real.pi / 2)
                         (h2 : abs x2 < Real.pi / 2) (h3 : abs x3 < Real.pi / 2)
                         (c1 : f x1 + f x2 ≥ 0) (c2 : f x2 + f x3 ≥ 0) (c3 : f x3 + f x1 ≥ 0) :
  f (x1 + x2 + x3) ≥ 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1833_183358


namespace NUMINAMATH_GPT_num_articles_produced_l1833_183302

-- Conditions
def production_rate (x : ℕ) : ℕ := 2 * x^3 / (x * x * 2 * x)
def articles_produced (y : ℕ) : ℕ := y * 2 * y * y * production_rate y

-- Proof: Given the production rate, prove the number of articles produced.
theorem num_articles_produced (y : ℕ) : articles_produced y = 2 * y^3 := by sorry

end NUMINAMATH_GPT_num_articles_produced_l1833_183302


namespace NUMINAMATH_GPT_impossible_to_achieve_12_percent_return_l1833_183326

-- Define the stock parameters and their individual returns
def stock_A_price : ℝ := 52
def stock_A_dividend_rate : ℝ := 0.09
def stock_A_transaction_fee_rate : ℝ := 0.02

def stock_B_price : ℝ := 80
def stock_B_dividend_rate : ℝ := 0.07
def stock_B_transaction_fee_rate : ℝ := 0.015

def stock_C_price : ℝ := 40
def stock_C_dividend_rate : ℝ := 0.10
def stock_C_transaction_fee_rate : ℝ := 0.01

def tax_rate : ℝ := 0.10
def desired_return : ℝ := 0.12

theorem impossible_to_achieve_12_percent_return :
  false :=
sorry

end NUMINAMATH_GPT_impossible_to_achieve_12_percent_return_l1833_183326


namespace NUMINAMATH_GPT_find_m_root_zero_l1833_183395

theorem find_m_root_zero (m : ℝ) : (m - 1) * 0 ^ 2 + 0 + m ^ 2 - 1 = 0 → m = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_m_root_zero_l1833_183395


namespace NUMINAMATH_GPT_intersection_complement_l1833_183304

-- Definitions based on the conditions in the problem
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

-- Definition of complement of set M in the universe U
def complement_U (M : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ M}

-- The proof statement
theorem intersection_complement :
  N ∩ (complement_U M) = {3, 5} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1833_183304


namespace NUMINAMATH_GPT_ratio_of_speeds_l1833_183398

theorem ratio_of_speeds (v1 v2 : ℝ) (h1 : v1 > v2) (h2 : 8 = (v1 + v2) * 2) (h3 : 8 = (v1 - v2) * 4) : v1 / v2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l1833_183398


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1833_183308

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, (m - 1) * (-2) - 3 + 2 * m + 1 = 0 :=
by
  intros m
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1833_183308


namespace NUMINAMATH_GPT_fold_hexagon_possible_l1833_183323

theorem fold_hexagon_possible (a b : ℝ) :
  (∃ x : ℝ, (a - x)^2 + (b - x)^2 = x^2) ↔ (1 / 2 < b / a ∧ b / a < 2) :=
by
  sorry

end NUMINAMATH_GPT_fold_hexagon_possible_l1833_183323


namespace NUMINAMATH_GPT_number_of_diagonals_dodecagon_sum_of_interior_angles_dodecagon_l1833_183311

-- Definitions for the problem
def n : Nat := 12

-- Statement 1: Number of diagonals in a dodecagon
theorem number_of_diagonals_dodecagon (n : Nat) (h : n = 12) : (n * (n - 3)) / 2 = 54 := by
  sorry

-- Statement 2: Sum of interior angles in a dodecagon
theorem sum_of_interior_angles_dodecagon (n : Nat) (h : n = 12) : 180 * (n - 2) = 1800 := by
  sorry

end NUMINAMATH_GPT_number_of_diagonals_dodecagon_sum_of_interior_angles_dodecagon_l1833_183311


namespace NUMINAMATH_GPT_pirate_coins_l1833_183333

def coins_remain (k : ℕ) (x : ℕ) : ℕ :=
  if k = 0 then x else coins_remain (k - 1) x * (15 - k) / 15

theorem pirate_coins (x : ℕ) :
  (∀ k < 15, (k + 1) * coins_remain k x % 15 = 0) → 
  coins_remain 14 x = 8442 :=
sorry

end NUMINAMATH_GPT_pirate_coins_l1833_183333


namespace NUMINAMATH_GPT_emily_curtains_purchase_l1833_183374

theorem emily_curtains_purchase 
    (c : ℕ) 
    (curtain_cost : ℕ := 30)
    (print_count : ℕ := 9)
    (print_cost_per_unit : ℕ := 15)
    (installation_cost : ℕ := 50)
    (total_cost : ℕ := 245) :
    (curtain_cost * c + print_count * print_cost_per_unit + installation_cost = total_cost) → c = 2 :=
by
  sorry

end NUMINAMATH_GPT_emily_curtains_purchase_l1833_183374


namespace NUMINAMATH_GPT_solve_quadratic_l1833_183356

theorem solve_quadratic : ∀ x : ℝ, x ^ 2 - 6 * x + 8 = 0 ↔ x = 2 ∨ x = 4 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1833_183356


namespace NUMINAMATH_GPT_polynomial_multiplication_identity_l1833_183336

-- Statement of the problem
theorem polynomial_multiplication_identity (x : ℝ) : 
  (25 * x^3) * (12 * x^2) * (1 / (5 * x)^3) = (12 / 5) * x^2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_multiplication_identity_l1833_183336


namespace NUMINAMATH_GPT_total_pieces_correct_l1833_183307

-- Definitions based on conditions
def rods_in_row (n : ℕ) : ℕ := 3 * n
def connectors_in_row (n : ℕ) : ℕ := n

-- Sum of natural numbers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Total rods in ten rows
def total_rods : ℕ := 3 * sum_first_n 10

-- Total connectors in eleven rows
def total_connectors : ℕ := sum_first_n 11

-- Total pieces
def total_pieces : ℕ := total_rods + total_connectors

-- Theorem to prove
theorem total_pieces_correct : total_pieces = 231 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_correct_l1833_183307


namespace NUMINAMATH_GPT_find_range_of_a_l1833_183320

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * a * x + 4 > 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 4 - 2 * a > 0 ∧ 4 - 2 * a < 1

noncomputable def problem_statement (a : ℝ) : Prop :=
  let p := proposition_p a
  let q := proposition_q a
  (p ∨ q) ∧ ¬(p ∧ q)

theorem find_range_of_a (a : ℝ) :
  problem_statement a → -2 < a ∧ a ≤ 3/2 :=
sorry

end NUMINAMATH_GPT_find_range_of_a_l1833_183320


namespace NUMINAMATH_GPT_domain_transform_l1833_183342

variable (f : ℝ → ℝ)

theorem domain_transform (h : ∀ x, -1 ≤ x ∧ x ≤ 4 → ∃ y, f y = x) :
  ∀ x, 0 ≤ x ∧ x ≤ 5 / 2 → ∃ y, f y = 2 * x - 1 :=
sorry

end NUMINAMATH_GPT_domain_transform_l1833_183342


namespace NUMINAMATH_GPT_find_k_l1833_183325

variable (m n k : ℝ)

-- Conditions from the problem
def quadratic_roots : Prop := (m + n = -2) ∧ (m * n = k) ∧ (1/m + 1/n = 6)

-- Theorem statement
theorem find_k (h : quadratic_roots m n k) : k = -1/3 :=
sorry

end NUMINAMATH_GPT_find_k_l1833_183325


namespace NUMINAMATH_GPT_sally_total_spent_l1833_183367

-- Define the prices paid by Sally for peaches after the coupon and for cherries
def P_peaches : ℝ := 12.32
def C_cherries : ℝ := 11.54

-- State the problem to prove that the total amount Sally spent is 23.86
theorem sally_total_spent : P_peaches + C_cherries = 23.86 := by
  sorry

end NUMINAMATH_GPT_sally_total_spent_l1833_183367


namespace NUMINAMATH_GPT_boat_stream_speed_l1833_183378

/-- A boat can travel with a speed of 22 km/hr in still water. 
If the speed of the stream is unknown, the boat takes 7 hours 
to go 189 km downstream. What is the speed of the stream?
-/
theorem boat_stream_speed (v : ℝ) : (22 + v) * 7 = 189 → v = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_boat_stream_speed_l1833_183378
