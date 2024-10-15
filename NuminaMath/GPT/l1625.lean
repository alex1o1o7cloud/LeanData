import Mathlib

namespace NUMINAMATH_GPT_teacher_age_l1625_162583

theorem teacher_age {student_count : ℕ} (avg_age_students : ℕ) (avg_age_with_teacher : ℕ)
    (h1 : student_count = 25) (h2 : avg_age_students = 26) (h3 : avg_age_with_teacher = 27) :
    ∃ (teacher_age : ℕ), teacher_age = 52 :=
by
  sorry

end NUMINAMATH_GPT_teacher_age_l1625_162583


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l1625_162594

open Real

theorem solve_equation1 (x : ℝ) : (x - 2)^2 = 9 → (x = 5 ∨ x = -1) :=
by
  intro h
  sorry -- Proof would go here

theorem solve_equation2 (x : ℝ) : (2 * x^2 - 3 * x - 1 = 0) → (x = (3 + sqrt 17) / 4 ∨ x = (3 - sqrt 17) / 4) :=
by
  intro h
  sorry -- Proof would go here

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l1625_162594


namespace NUMINAMATH_GPT_find_g_g2_l1625_162505

def g (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 1

theorem find_g_g2 : g (g 2) = 2630 := by
  sorry

end NUMINAMATH_GPT_find_g_g2_l1625_162505


namespace NUMINAMATH_GPT_TimSpentTotal_l1625_162531

variable (LunchCost : ℝ) (TipPercentage : ℝ)

def TotalAmountSpent (LunchCost : ℝ) (TipPercentage : ℝ) : ℝ := 
  LunchCost + (LunchCost * TipPercentage)

theorem TimSpentTotal (h1 : LunchCost = 50.50) (h2 : TipPercentage = 0.20) :
  TotalAmountSpent LunchCost TipPercentage = 60.60 := by
  sorry

end NUMINAMATH_GPT_TimSpentTotal_l1625_162531


namespace NUMINAMATH_GPT_cost_of_watermelon_and_grapes_l1625_162568

variable (x y z f : ℕ)

theorem cost_of_watermelon_and_grapes (h1 : x + y + z + f = 45) 
                                    (h2 : f = 3 * x) 
                                    (h3 : z = x + y) :
    y + z = 9 := by
  sorry

end NUMINAMATH_GPT_cost_of_watermelon_and_grapes_l1625_162568


namespace NUMINAMATH_GPT_prove_axisymmetric_char4_l1625_162501

-- Predicates representing whether a character is an axisymmetric figure
def is_axisymmetric (ch : Char) : Prop := sorry

-- Definitions for the conditions given in the problem
def char1 := '月'
def char2 := '右'
def char3 := '同'
def char4 := '干'

-- Statement that needs to be proven
theorem prove_axisymmetric_char4 (h1 : ¬ is_axisymmetric char1) 
                                  (h2 : ¬ is_axisymmetric char2) 
                                  (h3 : ¬ is_axisymmetric char3) : 
                                  is_axisymmetric char4 :=
sorry

end NUMINAMATH_GPT_prove_axisymmetric_char4_l1625_162501


namespace NUMINAMATH_GPT_age_problem_l1625_162553

theorem age_problem
    (D X : ℕ) 
    (h1 : D = 4 * X) 
    (h2 : D = X + 30) : D = 40 ∧ X = 10 := by
  sorry

end NUMINAMATH_GPT_age_problem_l1625_162553


namespace NUMINAMATH_GPT_line_through_circle_center_l1625_162526

theorem line_through_circle_center {m : ℝ} :
  (∃ (x y : ℝ), x - 2*y + m = 0 ∧ x^2 + y^2 + 2*x - 4*y = 0) → m = 5 :=
by
  sorry

end NUMINAMATH_GPT_line_through_circle_center_l1625_162526


namespace NUMINAMATH_GPT_cos_alpha_value_l1625_162595

variable (α : ℝ)
variable (x y r : ℝ)

-- Conditions
def point_condition : Prop := (x = 1 ∧ y = -Real.sqrt 3 ∧ r = 2 ∧ r = Real.sqrt (x^2 + y^2))

-- Question/Proof Statement
theorem cos_alpha_value (h : point_condition x y r) : Real.cos α = 1 / 2 :=
sorry

end NUMINAMATH_GPT_cos_alpha_value_l1625_162595


namespace NUMINAMATH_GPT_parabola_translation_l1625_162544

theorem parabola_translation :
  ∀ (x y : ℝ), (y = 2 * (x - 3) ^ 2) ↔ ∃ t : ℝ, t = x - 3 ∧ y = 2 * t ^ 2 :=
by sorry

end NUMINAMATH_GPT_parabola_translation_l1625_162544


namespace NUMINAMATH_GPT_convert_base4_to_base10_l1625_162585

-- Define a function to convert a base 4 number to base 10
def base4_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0

-- Assert the proof problem
theorem convert_base4_to_base10 : base4_to_base10 3201 = 225 :=
by
  -- The proof script goes here; for now, we use 'sorry' as a placeholder
  sorry

end NUMINAMATH_GPT_convert_base4_to_base10_l1625_162585


namespace NUMINAMATH_GPT_find_P_l1625_162586

theorem find_P (P : ℕ) (h : 4 * (P + 4 + 8 + 20) = 252) : P = 31 :=
by
  -- Assume this proof is nontrivial and required steps
  sorry

end NUMINAMATH_GPT_find_P_l1625_162586


namespace NUMINAMATH_GPT_correct_expression_must_hold_l1625_162524

variable {f : ℝ → ℝ}

-- Conditions
axiom increasing_function : ∀ x y : ℝ, x < y → f x < f y
axiom positive_function : ∀ x : ℝ, f x > 0

-- Problem Statement
theorem correct_expression_must_hold : 3 * f (-2) > 2 * f (-3) := by
  sorry

end NUMINAMATH_GPT_correct_expression_must_hold_l1625_162524


namespace NUMINAMATH_GPT_product_is_correct_l1625_162556

theorem product_is_correct :
  50 * 29.96 * 2.996 * 500 = 2244004 :=
by
  sorry

end NUMINAMATH_GPT_product_is_correct_l1625_162556


namespace NUMINAMATH_GPT_x_coordinate_incenter_eq_l1625_162558

theorem x_coordinate_incenter_eq {x y : ℝ} :
  (y = 0 → x + y = 3 → x = 0) → 
  (y = x → y = -x + 3 → x = 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_x_coordinate_incenter_eq_l1625_162558


namespace NUMINAMATH_GPT_markup_is_correct_l1625_162593

noncomputable def profit (S : ℝ) : ℝ := 0.12 * S
noncomputable def expenses (S : ℝ) : ℝ := 0.10 * S
noncomputable def cost (S : ℝ) : ℝ := S - (profit S + expenses S)
noncomputable def markup (S : ℝ) : ℝ :=
  ((S - cost S) / (cost S)) * 100

theorem markup_is_correct:
  markup 10 = 28.21 :=
by
  sorry

end NUMINAMATH_GPT_markup_is_correct_l1625_162593


namespace NUMINAMATH_GPT_find_x_l1625_162569

theorem find_x (n x q p : ℕ) (h1 : n = q * x + 2) (h2 : 2 * n = p * x + 4) : x = 6 :=
sorry

end NUMINAMATH_GPT_find_x_l1625_162569


namespace NUMINAMATH_GPT_sphere_radius_equal_l1625_162538

theorem sphere_radius_equal (r : ℝ) 
  (hvol : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end NUMINAMATH_GPT_sphere_radius_equal_l1625_162538


namespace NUMINAMATH_GPT_find_m_for_parallel_lines_l1625_162522

open Real

theorem find_m_for_parallel_lines :
  ∀ (m : ℝ),
    (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 → 6 * x + m * y + 14 = 0 → 3 * m = 4 * 6) →
    m = 8 :=
by
  intro m h
  have H : 3 * m = 4 * 6 := h 0 0 sorry sorry
  linarith

end NUMINAMATH_GPT_find_m_for_parallel_lines_l1625_162522


namespace NUMINAMATH_GPT_expenditure_proof_l1625_162591

namespace OreoCookieProblem

variables (O C : ℕ) (CO CC : ℕ → ℕ) (total_items cost_difference : ℤ)

def oreo_count_eq : Prop := O = (4 * (65 : ℤ) / 13)
def cookie_count_eq : Prop := C = (9 * (65 : ℤ) / 13)
def oreo_cost (o : ℕ) : ℕ := o * 2
def cookie_cost (c : ℕ) : ℕ := c * 3
def total_item_condition : Prop := O + C = 65
def ratio_condition : Prop := 9 * O = 4 * C
def cost_difference_condition (o_cost c_cost : ℕ) : Prop := cost_difference = (c_cost - o_cost)

theorem expenditure_proof :
  (O + C = 65) →
  (9 * O = 4 * C) →
  (O = 20) →
  (C = 45) →
  cost_difference = (45 * 3 - 20 * 2) →
  cost_difference = 95 :=
by sorry

end OreoCookieProblem

end NUMINAMATH_GPT_expenditure_proof_l1625_162591


namespace NUMINAMATH_GPT_parabola_standard_equation_l1625_162536

theorem parabola_standard_equation :
  ∃ m : ℝ, (∀ x y : ℝ, (x^2 = 2 * m * y ↔ (0, -6) ∈ ({p | 3 * p.1 - 4 * p.2 - 24 = 0}))) → 
  (x^2 = -24 * y) := 
by {
  sorry
}

end NUMINAMATH_GPT_parabola_standard_equation_l1625_162536


namespace NUMINAMATH_GPT_algebraic_expression_result_l1625_162517

theorem algebraic_expression_result (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 - 12 = -11 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_result_l1625_162517


namespace NUMINAMATH_GPT_rectangle_length_twice_breadth_l1625_162572

theorem rectangle_length_twice_breadth
  (b : ℝ) 
  (l : ℝ)
  (h1 : l = 2 * b)
  (h2 : (l - 5) * (b + 4) = l * b + 75) :
  l = 190 / 3 :=
sorry

end NUMINAMATH_GPT_rectangle_length_twice_breadth_l1625_162572


namespace NUMINAMATH_GPT_geometric_sequence_implies_condition_counterexample_condition_does_not_imply_geometric_sequence_geometric_sequence_sufficient_not_necessary_l1625_162510

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n ≠ 0 ∧ (a (n + 1) = a n * (a (n + 1) / a n))

theorem geometric_sequence_implies_condition (a : ℕ → ℝ) :
  is_geometric_sequence a → ∀ n, (a n)^2 = a (n - 1) * a (n + 1) := sorry

theorem counterexample_condition_does_not_imply_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, (a n)^2 = a (n - 1) * a (n + 1)) → ¬ is_geometric_sequence a := sorry

theorem geometric_sequence_sufficient_not_necessary (a : ℕ → ℝ) :
  (is_geometric_sequence a → ∀ n, (a n)^2 = a (n - 1) * a (n + 1)) ∧
  ((∀ n, (a n)^2 = a (n - 1) * a (n + 1)) → ¬ is_geometric_sequence a) := by
  exact ⟨geometric_sequence_implies_condition a, counterexample_condition_does_not_imply_geometric_sequence a⟩

end NUMINAMATH_GPT_geometric_sequence_implies_condition_counterexample_condition_does_not_imply_geometric_sequence_geometric_sequence_sufficient_not_necessary_l1625_162510


namespace NUMINAMATH_GPT_total_cookies_eaten_l1625_162557

theorem total_cookies_eaten :
  let charlie := 15
  let father := 10
  let mother := 5
  let grandmother := 12 / 2
  let dog := 3 * 0.75
  charlie + father + mother + grandmother + dog = 38.25 :=
by
  sorry

end NUMINAMATH_GPT_total_cookies_eaten_l1625_162557


namespace NUMINAMATH_GPT_line_parabola_one_point_l1625_162515

theorem line_parabola_one_point (k : ℝ) :
  (∃ x y : ℝ, y = k * x + 2 ∧ y^2 = 8 * x) 
  → (k = 0 ∨ k = 1) := 
by 
  sorry

end NUMINAMATH_GPT_line_parabola_one_point_l1625_162515


namespace NUMINAMATH_GPT_base_unit_digit_l1625_162597

def unit_digit (n : ℕ) : ℕ := n % 10

theorem base_unit_digit (x : ℕ) :
  unit_digit ((x^41) * (41^14) * (14^87) * (87^76)) = 4 →
  unit_digit x = 1 :=
by
  sorry

end NUMINAMATH_GPT_base_unit_digit_l1625_162597


namespace NUMINAMATH_GPT_range_of_a_l1625_162516

noncomputable def f (a x : ℝ) : ℝ := Real.sin x + 0.5 * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici 0, f a x ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1625_162516


namespace NUMINAMATH_GPT_non_planar_characterization_l1625_162503

-- Definitions:
structure Graph where
  V : ℕ
  E : ℕ
  F : ℕ

def is_planar (G : Graph) : Prop :=
  G.V - G.E + G.F = 2

def edge_inequality (G : Graph) : Prop :=
  G.E ≤ 3 * G.V - 6

def has_subgraph_K5_or_K33 (G : Graph) : Prop := sorry -- Placeholder for the complex subgraph check

-- Theorem statement:
theorem non_planar_characterization (G : Graph) (hV : G.V ≥ 3) :
  ¬ is_planar G ↔ ¬ edge_inequality G ∨ has_subgraph_K5_or_K33 G := sorry

end NUMINAMATH_GPT_non_planar_characterization_l1625_162503


namespace NUMINAMATH_GPT_rational_roots_of_quadratic_l1625_162579

theorem rational_roots_of_quadratic (r : ℚ) :
  (∃ a b : ℤ, a ≠ b ∧ (r * a^2 + (r + 1) * a + r = 1 ∧ r * b^2 + (r + 1) * b + r = 1)) ↔ (r = 1 ∨ r = -1 / 7) :=
by
  sorry

end NUMINAMATH_GPT_rational_roots_of_quadratic_l1625_162579


namespace NUMINAMATH_GPT_drums_filled_per_day_l1625_162519

-- Definition of given conditions
def pickers : ℕ := 266
def total_drums : ℕ := 90
def total_days : ℕ := 5

-- Statement to prove
theorem drums_filled_per_day : (total_drums / total_days) = 18 := by
  sorry

end NUMINAMATH_GPT_drums_filled_per_day_l1625_162519


namespace NUMINAMATH_GPT_timothy_movies_count_l1625_162504

variable (T : ℕ)

def timothy_movies_previous_year (T : ℕ) :=
  let timothy_2010 := T + 7
  let theresa_2010 := 2 * (T + 7)
  let theresa_previous := T / 2
  T + timothy_2010 + theresa_2010 + theresa_previous = 129

theorem timothy_movies_count (T : ℕ) (h : timothy_movies_previous_year T) : T = 24 := 
by 
  sorry

end NUMINAMATH_GPT_timothy_movies_count_l1625_162504


namespace NUMINAMATH_GPT_inequality_subtraction_l1625_162509

theorem inequality_subtraction (a b : ℝ) (h : a < b) : a - 5 < b - 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_subtraction_l1625_162509


namespace NUMINAMATH_GPT_factorization_correct_l1625_162581

theorem factorization_correct (x : ℝ) : 2 * x ^ 2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1625_162581


namespace NUMINAMATH_GPT_mark_brought_in_4_times_more_cans_l1625_162520

theorem mark_brought_in_4_times_more_cans (M J R : ℕ) (h1 : M = 100) 
  (h2 : J = 2 * R + 5) (h3 : M + J + R = 135) : M / J = 4 :=
by sorry

end NUMINAMATH_GPT_mark_brought_in_4_times_more_cans_l1625_162520


namespace NUMINAMATH_GPT_positive_difference_between_two_numbers_l1625_162570

theorem positive_difference_between_two_numbers :
  ∃ (x y : ℚ), x + y = 40 ∧ 3 * y - 4 * x = 10 ∧ abs (y - x) = 60 / 7 :=
sorry

end NUMINAMATH_GPT_positive_difference_between_two_numbers_l1625_162570


namespace NUMINAMATH_GPT_h_at_3_eq_3_l1625_162587

-- Define the function h(x) based on the given condition
noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * 
    (x^32 + 1) * (x^64 + 1) * (x^128 + 1) * (x^256 + 1) * (x^512 + 1) - 1) / 
  (x^(2^10 - 1) - 1)

-- State the required theorem
theorem h_at_3_eq_3 : h 3 = 3 := by
  sorry

end NUMINAMATH_GPT_h_at_3_eq_3_l1625_162587


namespace NUMINAMATH_GPT_dryer_sheets_per_load_l1625_162528

theorem dryer_sheets_per_load (loads_per_week : ℕ) (cost_of_box : ℝ) (sheets_per_box : ℕ)
  (annual_savings : ℝ) (weeks_in_year : ℕ) (x : ℕ)
  (h1 : loads_per_week = 4)
  (h2 : cost_of_box = 5.50)
  (h3 : sheets_per_box = 104)
  (h4 : annual_savings = 11)
  (h5 : weeks_in_year = 52)
  (h6 : annual_savings = 2 * cost_of_box)
  (h7 : sheets_per_box * 2 = weeks_in_year * (loads_per_week * x)):
  x = 1 :=
by
  sorry

end NUMINAMATH_GPT_dryer_sheets_per_load_l1625_162528


namespace NUMINAMATH_GPT_choose_3_from_12_l1625_162512

theorem choose_3_from_12 : (Nat.choose 12 3) = 220 := by
  sorry

end NUMINAMATH_GPT_choose_3_from_12_l1625_162512


namespace NUMINAMATH_GPT_compute_value_l1625_162545

theorem compute_value : 12 - 4 * (5 - 10)^3 = 512 :=
by
  sorry

end NUMINAMATH_GPT_compute_value_l1625_162545


namespace NUMINAMATH_GPT_positive_slope_asymptote_l1625_162552

def hyperbola (x y : ℝ) :=
  Real.sqrt ((x - 1) ^ 2 + (y + 2) ^ 2) - Real.sqrt ((x - 6) ^ 2 + (y + 2) ^ 2) = 4

theorem positive_slope_asymptote :
  ∃ (m : ℝ), m = 0.75 ∧ (∃ x y, hyperbola x y) :=
sorry

end NUMINAMATH_GPT_positive_slope_asymptote_l1625_162552


namespace NUMINAMATH_GPT_turquoise_beads_count_l1625_162578

-- Define the conditions
def num_beads_total : ℕ := 40
def num_amethyst : ℕ := 7
def num_amber : ℕ := 2 * num_amethyst

-- Define the main theorem to prove
theorem turquoise_beads_count :
  num_beads_total - (num_amethyst + num_amber) = 19 :=
by
  sorry

end NUMINAMATH_GPT_turquoise_beads_count_l1625_162578


namespace NUMINAMATH_GPT_leak_empties_cistern_in_12_hours_l1625_162532

theorem leak_empties_cistern_in_12_hours 
  (R : ℝ) (L : ℝ)
  (h1 : R = 1 / 4) 
  (h2 : R - L = 1 / 6) : 
  1 / L = 12 := 
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_leak_empties_cistern_in_12_hours_l1625_162532


namespace NUMINAMATH_GPT_outfits_count_l1625_162502

-- Definitions of various clothing counts
def numRedShirts : ℕ := 7
def numGreenShirts : ℕ := 3
def numPants : ℕ := 8
def numBlueShoes : ℕ := 5
def numRedShoes : ℕ := 5
def numGreenHats : ℕ := 10
def numRedHats : ℕ := 6

-- Statement of the theorem based on the problem description
theorem outfits_count :
  (numRedShirts * numPants * numBlueShoes * numGreenHats) + 
  (numGreenShirts * numPants * (numBlueShoes + numRedShoes) * numRedHats) = 4240 := 
by
  -- No proof required, only the statement is needed
  sorry

end NUMINAMATH_GPT_outfits_count_l1625_162502


namespace NUMINAMATH_GPT_second_and_fourth_rows_identical_l1625_162555

def count_occurrences (lst : List ℕ) (a : ℕ) (i : ℕ) : ℕ :=
  (lst.take (i + 1)).count a

def fill_next_row (current_row : List ℕ) : List ℕ :=
  current_row.enum.map (λ ⟨i, a⟩ => count_occurrences current_row a i)

theorem second_and_fourth_rows_identical (first_row : List ℕ) :
  let second_row := fill_next_row first_row 
  let third_row := fill_next_row second_row 
  let fourth_row := fill_next_row third_row 
  second_row = fourth_row :=
by
  sorry

end NUMINAMATH_GPT_second_and_fourth_rows_identical_l1625_162555


namespace NUMINAMATH_GPT_gary_money_left_l1625_162580

theorem gary_money_left (initial_amount spent_amount remaining_amount : ℕ)
  (h1 : initial_amount = 73)
  (h2 : spent_amount = 55)
  (h3 : remaining_amount = 18) : initial_amount - spent_amount = remaining_amount := 
by 
  sorry

end NUMINAMATH_GPT_gary_money_left_l1625_162580


namespace NUMINAMATH_GPT_range_of_a_l1625_162529

open Real

theorem range_of_a {a : ℝ} :
  (∃ x : ℝ, sqrt (3 * x + 6) + sqrt (14 - x) > a) → a < 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1625_162529


namespace NUMINAMATH_GPT_standard_eq_minimal_circle_l1625_162548

-- Definitions
variables {x y : ℝ}
variables (h₀ : 0 < x) (h₁ : 0 < y)
variables (h₂ : 3 / (2 + x) + 3 / (2 + y) = 1)

-- Theorem statement
theorem standard_eq_minimal_circle : (x - 4)^2 + (y - 4)^2 = 16^2 :=
sorry

end NUMINAMATH_GPT_standard_eq_minimal_circle_l1625_162548


namespace NUMINAMATH_GPT_tricia_age_is_5_l1625_162561

theorem tricia_age_is_5 :
  (∀ Amilia Yorick Eugene Khloe Rupert Vincent : ℕ,
    Tricia = 5 ∧
    (3 * Tricia = Amilia) ∧
    (4 * Amilia = Yorick) ∧
    (2 * Eugene = Yorick) ∧
    (Eugene / 3 = Khloe) ∧
    (Khloe + 10 = Rupert) ∧
    (Vincent = 22)) → 
  Tricia = 5 :=
by
  sorry

end NUMINAMATH_GPT_tricia_age_is_5_l1625_162561


namespace NUMINAMATH_GPT_count_oddly_powerful_integers_l1625_162577

def is_oddly_powerful (m : ℕ) : Prop :=
  ∃ (c d : ℕ), d > 1 ∧ d % 2 = 1 ∧ c^d = m

theorem count_oddly_powerful_integers :
  ∃ (S : Finset ℕ), 
  (∀ m, m ∈ S ↔ (m < 1500 ∧ is_oddly_powerful m)) ∧ S.card = 13 :=
by
  sorry

end NUMINAMATH_GPT_count_oddly_powerful_integers_l1625_162577


namespace NUMINAMATH_GPT_increase_80_by_50_percent_l1625_162539

theorem increase_80_by_50_percent :
  let initial_number : ℕ := 80
  let increase_percentage : ℝ := 0.5
  initial_number + (initial_number * increase_percentage) = 120 :=
by
  sorry

end NUMINAMATH_GPT_increase_80_by_50_percent_l1625_162539


namespace NUMINAMATH_GPT_ellipse_parabola_intersection_l1625_162596

theorem ellipse_parabola_intersection (c : ℝ) : 
  (∀ x y : ℝ, (x^2 + (y^2 / 4) = c^2 ∧ y = x^2 - 2 * c) → false) ↔ c > 1 := by
  sorry

end NUMINAMATH_GPT_ellipse_parabola_intersection_l1625_162596


namespace NUMINAMATH_GPT_magical_stack_130_cards_l1625_162525

theorem magical_stack_130_cards (n : ℕ) (h1 : 2 * n > 0) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ 2 * (n - k + 1) = 131 ∨ 
                                   (n + 1) ≤ k ∧ k ≤ 2 * n ∧ 2 * k - 1 = 131) : 2 * n = 130 :=
by
  sorry

end NUMINAMATH_GPT_magical_stack_130_cards_l1625_162525


namespace NUMINAMATH_GPT_actual_average_speed_l1625_162527

variable {t : ℝ} (h₁ : t > 0) -- ensure that time is positive
variable {v : ℝ} 

theorem actual_average_speed (h₂ : v > 0)
  (h3 : v * t = (v + 12) * (3 / 4 * t)) : v = 36 :=
by
  sorry

end NUMINAMATH_GPT_actual_average_speed_l1625_162527


namespace NUMINAMATH_GPT_number_of_real_roots_l1625_162533

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then 2010 * x + Real.log x / Real.log 2010
  else if x < 0 then - (2010 * (-x) + Real.log (-x) / Real.log 2010)
  else 0

theorem number_of_real_roots : 
  (∃ x1 x2 x3 : ℝ, 
    f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) ∧ 
    ∀ x y z : ℝ, 
    (f x = 0 ∧ f y = 0 ∧ f z = 0 → 
    (x = y ∨ x = z ∨ y = z)) 
  :=
by
  sorry

end NUMINAMATH_GPT_number_of_real_roots_l1625_162533


namespace NUMINAMATH_GPT_complex_numbers_not_comparable_l1625_162549

-- Definitions based on conditions
def is_real (z : ℂ) : Prop := ∃ r : ℝ, z = r
def is_not_entirely_real (z : ℂ) : Prop := ¬ is_real z

-- Proof problem statement
theorem complex_numbers_not_comparable (z1 z2 : ℂ) (h1 : is_not_entirely_real z1) (h2 : is_not_entirely_real z2) : 
  ¬ (z1.re = z2.re ∧ z1.im = z2.im) :=
sorry

end NUMINAMATH_GPT_complex_numbers_not_comparable_l1625_162549


namespace NUMINAMATH_GPT_walter_zoo_time_l1625_162582

def seals_time : ℕ := 13
def penguins_time : ℕ := 8 * seals_time
def elephants_time : ℕ := 13
def total_time_spent_at_zoo : ℕ := seals_time + penguins_time + elephants_time

theorem walter_zoo_time : total_time_spent_at_zoo = 130 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_walter_zoo_time_l1625_162582


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l1625_162550

theorem geometric_sequence_seventh_term (r : ℕ) (r_pos : 0 < r) 
  (h1 : 3 * r^4 = 243) : 
  3 * r^6 = 2187 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l1625_162550


namespace NUMINAMATH_GPT_div_by_prime_power_l1625_162599

theorem div_by_prime_power (p α x : ℕ) (hp : Nat.Prime p) (hpg : p > 2) (hα : α > 0) (t : ℤ) :
  (∃ k : ℤ, x^2 - 1 = k * p^α) ↔ (∃ t : ℤ, x = t * p^α + 1 ∨ x = t * p^α - 1) :=
sorry

end NUMINAMATH_GPT_div_by_prime_power_l1625_162599


namespace NUMINAMATH_GPT_equal_sums_of_squares_l1625_162506

-- Define the coordinates of a rectangle in a 3D space.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define vertices of the rectangle.
def A : Point3D := ⟨0, 0, 0⟩
def B (a : ℝ) : Point3D := ⟨a, 0, 0⟩
def C (a b : ℝ) : Point3D := ⟨a, b, 0⟩
def D (b : ℝ) : Point3D := ⟨0, b, 0⟩

-- Distance squared between two points in 3D space.
def distance_squared (M N : Point3D) : ℝ :=
  (M.x - N.x)^2 + (M.y - N.y)^2 + (M.z - N.z)^2

-- Prove that the sums of the squares of the distances between an arbitrary point M and opposite vertices of the rectangle are equal.
theorem equal_sums_of_squares (a b : ℝ) (M : Point3D) :
  distance_squared M A + distance_squared M (C a b) = distance_squared M (B a) + distance_squared M (D b) :=
by
  sorry

end NUMINAMATH_GPT_equal_sums_of_squares_l1625_162506


namespace NUMINAMATH_GPT_find_YZ_l1625_162547

noncomputable def triangle_YZ (angle_Y : ℝ) (XY : ℝ) (XZ : ℝ) : ℝ :=
  if angle_Y = 45 ∧ XY = 100 ∧ XZ = 50 * Real.sqrt 2 then
    50 * Real.sqrt 6
  else
    0

theorem find_YZ :
  triangle_YZ 45 100 (50 * Real.sqrt 2) = 50 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_find_YZ_l1625_162547


namespace NUMINAMATH_GPT_union_of_A_and_B_l1625_162540

namespace SetProof

def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x > 0}
def expectedUnion : Set ℝ := {x | -2 ≤ x}

theorem union_of_A_and_B : (A ∪ B) = expectedUnion := by
  sorry

end SetProof

end NUMINAMATH_GPT_union_of_A_and_B_l1625_162540


namespace NUMINAMATH_GPT_average_marks_l1625_162588

variable (P C M : ℕ)

theorem average_marks :
  P = 140 →
  (P + M) / 2 = 90 →
  (P + C) / 2 = 70 →
  (P + C + M) / 3 = 60 :=
by
  intros hP hM hC
  sorry

end NUMINAMATH_GPT_average_marks_l1625_162588


namespace NUMINAMATH_GPT_solve_digits_l1625_162537

variables (h t u : ℕ)

theorem solve_digits :
  (u = h + 6) →
  (u + h = 16) →
  (∀ (x y z : ℕ), 100 * h + 10 * t + u + 100 * u + 10 * t + h = 100 * x + 10 * y + z ∧ y = 9 ∧ z = 6) →
  (h = 5 ∧ t = 4 ∧ u = 11) :=
sorry

end NUMINAMATH_GPT_solve_digits_l1625_162537


namespace NUMINAMATH_GPT_anne_wandered_hours_l1625_162541

noncomputable def speed : ℝ := 2 -- miles per hour
noncomputable def distance : ℝ := 6 -- miles

theorem anne_wandered_hours (t : ℝ) (h : distance = speed * t) : t = 3 := by
  sorry

end NUMINAMATH_GPT_anne_wandered_hours_l1625_162541


namespace NUMINAMATH_GPT_quarters_in_school_year_l1625_162575

variable (students : ℕ) (artworks_per_student_per_quarter : ℕ) (total_artworks : ℕ) (school_years : ℕ)

def number_of_quarters (students : ℕ) (artworks_per_student_per_quarter : ℕ) (total_artworks : ℕ) (school_years : ℕ) : ℕ :=
  (total_artworks / (students * artworks_per_student_per_quarter * school_years))

theorem quarters_in_school_year :
  number_of_quarters 15 2 240 2 = 4 :=
by sorry

end NUMINAMATH_GPT_quarters_in_school_year_l1625_162575


namespace NUMINAMATH_GPT_mass_percentage_H3BO3_l1625_162565

theorem mass_percentage_H3BO3 :
  ∃ (element : String) (mass_percent : ℝ), 
    element ∈ ["H", "B", "O"] ∧ 
    mass_percent = 4.84 ∧ 
    mass_percent = 4.84 :=
sorry

end NUMINAMATH_GPT_mass_percentage_H3BO3_l1625_162565


namespace NUMINAMATH_GPT_accommodation_ways_l1625_162576

-- Definition of the problem
def triple_room_count : ℕ := 1
def double_room_count : ℕ := 2
def adults_count : ℕ := 3
def children_count : ℕ := 2
def total_ways : ℕ := 60

-- Main statement to be proved
theorem accommodation_ways :
  (triple_room_count = 1) →
  (double_room_count = 2) →
  (adults_count = 3) →
  (children_count = 2) →
  -- Children must be accompanied by adults, and not all rooms need to be occupied.
  -- We are to prove that the number of valid ways to assign the rooms is 60
  total_ways = 60 :=
by sorry

end NUMINAMATH_GPT_accommodation_ways_l1625_162576


namespace NUMINAMATH_GPT_smallest_product_not_factor_of_48_exists_l1625_162559

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end NUMINAMATH_GPT_smallest_product_not_factor_of_48_exists_l1625_162559


namespace NUMINAMATH_GPT_regular_nonagon_diagonals_correct_l1625_162543

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end NUMINAMATH_GPT_regular_nonagon_diagonals_correct_l1625_162543


namespace NUMINAMATH_GPT_rectangle_area_increase_l1625_162523

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let A_original := L * W
  let A_new := (2 * L) * (2 * W)
  (A_new - A_original) / A_original * 100 = 300 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_increase_l1625_162523


namespace NUMINAMATH_GPT_time_after_hours_l1625_162592

-- Definitions based on conditions
def current_time : ℕ := 3
def hours_later : ℕ := 2517
def clock_cycle : ℕ := 12

-- Statement to prove
theorem time_after_hours :
  (current_time + hours_later) % clock_cycle = 12 := 
sorry

end NUMINAMATH_GPT_time_after_hours_l1625_162592


namespace NUMINAMATH_GPT_ms_lee_class_difference_l1625_162514

noncomputable def boys_and_girls_difference (ratio_b : ℕ) (ratio_g : ℕ) (total_students : ℕ) : ℕ :=
  let x := total_students / (ratio_b + ratio_g)
  let boys := ratio_b * x
  let girls := ratio_g * x
  girls - boys

theorem ms_lee_class_difference :
  boys_and_girls_difference 3 4 42 = 6 :=
by
  sorry

end NUMINAMATH_GPT_ms_lee_class_difference_l1625_162514


namespace NUMINAMATH_GPT_peanuts_difference_is_correct_l1625_162566

-- Define the number of peanuts Jose has
def Jose_peanuts : ℕ := 85

-- Define the number of peanuts Kenya has
def Kenya_peanuts : ℕ := 133

-- Define the difference in the number of peanuts between Kenya and Jose
def peanuts_difference : ℕ := Kenya_peanuts - Jose_peanuts

-- Prove that the number of peanuts Kenya has minus the number of peanuts Jose has is equal to 48
theorem peanuts_difference_is_correct : peanuts_difference = 48 := by
  sorry

end NUMINAMATH_GPT_peanuts_difference_is_correct_l1625_162566


namespace NUMINAMATH_GPT_exists_four_digit_number_divisible_by_101_l1625_162554

theorem exists_four_digit_number_divisible_by_101 :
  ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
    b ≠ c ∧ b ≠ d ∧
    c ≠ d ∧
    (1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a) % 101 = 0 := 
by
  -- To be proven
  sorry

end NUMINAMATH_GPT_exists_four_digit_number_divisible_by_101_l1625_162554


namespace NUMINAMATH_GPT_average_weight_14_children_l1625_162560

theorem average_weight_14_children 
  (average_weight_boys : ℕ → ℤ → ℤ)
  (average_weight_girls : ℕ → ℤ → ℤ)
  (total_children : ℕ)
  (total_weight : ℤ)
  (total_average_weight : ℤ)
  (boys_count : ℕ)
  (girls_count : ℕ)
  (boys_average : ℤ)
  (girls_average : ℤ) :
  boys_count = 8 →
  girls_count = 6 →
  boys_average = 160 →
  girls_average = 130 →
  total_children = boys_count + girls_count →
  total_weight = average_weight_boys boys_count boys_average + average_weight_girls girls_count girls_average →
  average_weight_boys boys_count boys_average = boys_count * boys_average →
  average_weight_girls girls_count girls_average = girls_count * girls_average →
  total_average_weight = total_weight / total_children →
  total_average_weight = 147 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_14_children_l1625_162560


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1625_162508

/-- Define a square SEAN with side length 2. -/
def square_side_length : ℝ := 2

/-- Define a quarter-circle of radius 1. -/
def quarter_circle_radius : ℝ := 1

/-- Hypothesis: The radius of the largest circle that can be inscribed in the remaining figure. -/
theorem inscribed_circle_radius :
  let S : ℝ := square_side_length
  let R : ℝ := quarter_circle_radius
  ∃ (r : ℝ), (r = 5 - 3 * Real.sqrt 2) := 
sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1625_162508


namespace NUMINAMATH_GPT_rotated_angle_new_measure_l1625_162598

theorem rotated_angle_new_measure (initial_angle : ℝ) (rotation : ℝ) (final_angle : ℝ) :
  initial_angle = 60 ∧ rotation = 300 → final_angle = 120 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_rotated_angle_new_measure_l1625_162598


namespace NUMINAMATH_GPT_michelle_drives_294_miles_l1625_162534

theorem michelle_drives_294_miles
  (total_distance : ℕ)
  (michelle_drives : ℕ)
  (katie_drives : ℕ)
  (tracy_drives : ℕ)
  (h1 : total_distance = 1000)
  (h2 : michelle_drives = 3 * katie_drives)
  (h3 : tracy_drives = 2 * michelle_drives + 20)
  (h4 : katie_drives + michelle_drives + tracy_drives = total_distance) :
  michelle_drives = 294 := by
  sorry

end NUMINAMATH_GPT_michelle_drives_294_miles_l1625_162534


namespace NUMINAMATH_GPT_divisible_by_6_l1625_162562

theorem divisible_by_6 {n : ℕ} (h2 : 2 ∣ n) (h3 : 3 ∣ n) : 6 ∣ n :=
sorry

end NUMINAMATH_GPT_divisible_by_6_l1625_162562


namespace NUMINAMATH_GPT_root_sum_reciprocal_l1625_162573

theorem root_sum_reciprocal (p q r s : ℂ)
  (h1 : (∀ x : ℂ, x^4 - 6*x^3 + 11*x^2 - 6*x + 3 = 0 → x = p ∨ x = q ∨ x = r ∨ x = s))
  (h2 : p*q*r*s = 3) 
  (h3 : p*q + p*r + p*s + q*r + q*s + r*s = 11) :
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s)) = 11/3 :=
by
  sorry

end NUMINAMATH_GPT_root_sum_reciprocal_l1625_162573


namespace NUMINAMATH_GPT_albums_in_either_but_not_both_l1625_162551

-- Defining the conditions
def shared_albums : ℕ := 9
def total_albums_andrew : ℕ := 17
def unique_albums_john : ℕ := 6

-- Stating the theorem to prove
theorem albums_in_either_but_not_both :
  (total_albums_andrew - shared_albums) + unique_albums_john = 14 :=
sorry

end NUMINAMATH_GPT_albums_in_either_but_not_both_l1625_162551


namespace NUMINAMATH_GPT_greatest_root_of_g_l1625_162513

noncomputable def g (x : ℝ) : ℝ := 16 * x^4 - 20 * x^2 + 5

theorem greatest_root_of_g :
  ∃ r : ℝ, r = Real.sqrt 5 / 2 ∧ (forall x, g x ≤ g r) :=
sorry

end NUMINAMATH_GPT_greatest_root_of_g_l1625_162513


namespace NUMINAMATH_GPT_middle_number_is_40_l1625_162521

theorem middle_number_is_40 (A B C : ℕ) (h1 : C = 56) (h2 : C - A = 32) (h3 : B / C = 5 / 7) : B = 40 :=
  sorry

end NUMINAMATH_GPT_middle_number_is_40_l1625_162521


namespace NUMINAMATH_GPT_train_length_l1625_162563

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def jogger_head_start_m : ℝ := 240
noncomputable def train_passing_time_s : ℝ := 35.99712023038157

noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * (1000 / 3600)
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
noncomputable def distance_covered_by_train : ℝ := relative_speed_mps * train_passing_time_s

theorem train_length :
  distance_covered_by_train - jogger_head_start_m = 119.9712023038157 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1625_162563


namespace NUMINAMATH_GPT_smallest_number_to_add_quotient_of_resulting_number_l1625_162574

theorem smallest_number_to_add (k : ℕ) : 456 ∣ (897326 + k) → k = 242 := 
sorry

theorem quotient_of_resulting_number : (897326 + 242) / 456 = 1968 := 
sorry

end NUMINAMATH_GPT_smallest_number_to_add_quotient_of_resulting_number_l1625_162574


namespace NUMINAMATH_GPT_kevin_sold_13_crates_of_grapes_l1625_162571

-- Define the conditions
def total_crates : ℕ := 50
def crates_of_mangoes : ℕ := 20
def crates_of_passion_fruits : ℕ := 17

-- Define the question and expected answer
def crates_of_grapes : ℕ := total_crates - (crates_of_mangoes + crates_of_passion_fruits)

-- Prove that the crates of grapes equals to 13
theorem kevin_sold_13_crates_of_grapes :
  crates_of_grapes = 13 :=
by
  -- The proof steps are omitted as per instructions
  sorry

end NUMINAMATH_GPT_kevin_sold_13_crates_of_grapes_l1625_162571


namespace NUMINAMATH_GPT_pizza_share_l1625_162500

theorem pizza_share :
  forall (friends : ℕ) (leftover_pizza : ℚ), friends = 4 -> leftover_pizza = 5/6 -> (leftover_pizza / friends) = (5 / 24) :=
by
  intros friends leftover_pizza h_friends h_leftover_pizza
  sorry

end NUMINAMATH_GPT_pizza_share_l1625_162500


namespace NUMINAMATH_GPT_evaluate_expression_l1625_162546

noncomputable def x : ℚ := 4 / 8
noncomputable def y : ℚ := 5 / 6

theorem evaluate_expression : (8 * x + 6 * y) / (72 * x * y) = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1625_162546


namespace NUMINAMATH_GPT_determine_r_l1625_162535

theorem determine_r (S : ℕ → ℤ) (r : ℤ) (n : ℕ) (h1 : 2 ≤ n) (h2 : ∀ k, S k = 2^k + r) : 
  r = -1 :=
sorry

end NUMINAMATH_GPT_determine_r_l1625_162535


namespace NUMINAMATH_GPT_shaded_region_perimeter_l1625_162567

theorem shaded_region_perimeter :
  let side_length := 1
  let diagonal_length := Real.sqrt 2 * side_length
  let arc_TRU_length := (1 / 4) * (2 * Real.pi * diagonal_length)
  let arc_VPW_length := (1 / 4) * (2 * Real.pi * side_length)
  let arc_UV_length := (1 / 4) * (2 * Real.pi * (Real.sqrt 2 - side_length))
  let arc_WT_length := (1 / 4) * (2 * Real.pi * (Real.sqrt 2 - side_length))
  (arc_TRU_length + arc_VPW_length + arc_UV_length + arc_WT_length) = (2 * Real.sqrt 2 - 1) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_perimeter_l1625_162567


namespace NUMINAMATH_GPT_race_result_l1625_162564

-- Defining competitors
inductive Sprinter
| A
| B
| C

open Sprinter

-- Conditions as definitions
def position_changes : Sprinter → Nat
| A => sorry
| B => 5
| C => 6

def finishes_before (s1 s2 : Sprinter) : Prop := sorry

-- Stating the problem as a theorem
theorem race_result :
  position_changes C = 6 →
  position_changes B = 5 →
  finishes_before B A →
  (finishes_before B A ∧ finishes_before A C ∧ finishes_before B C) :=
by
  intros hC hB hBA
  sorry

end NUMINAMATH_GPT_race_result_l1625_162564


namespace NUMINAMATH_GPT_max_b_no_lattice_point_l1625_162542

theorem max_b_no_lattice_point (m : ℚ) (x : ℤ) (b : ℚ) :
  (y = mx + 3) → (0 < x ∧ x ≤ 50) → (2/5 < m ∧ m < b) → 
  ∀ (x : ℕ), y ≠ m * x + 3 →
  b = 11/51 :=
sorry

end NUMINAMATH_GPT_max_b_no_lattice_point_l1625_162542


namespace NUMINAMATH_GPT_inequality_range_l1625_162530

theorem inequality_range (a : ℝ) : 
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 2 ≤ y ∧ y ≤ 3 → x * y ≤ a * x^2 + 2 * y^2) ↔ a ≥ -1 := by 
  sorry

end NUMINAMATH_GPT_inequality_range_l1625_162530


namespace NUMINAMATH_GPT_sum_of_integers_with_product_neg13_l1625_162584

theorem sum_of_integers_with_product_neg13 (a b c : ℤ) (h : a * b * c = -13) : 
  a + b + c = 13 ∨ a + b + c = -11 := 
sorry

end NUMINAMATH_GPT_sum_of_integers_with_product_neg13_l1625_162584


namespace NUMINAMATH_GPT_airline_passenger_capacity_l1625_162590

def seats_per_row : Nat := 7
def rows_per_airplane : Nat := 20
def airplanes_owned : Nat := 5
def flights_per_day_per_airplane : Nat := 2

def seats_per_airplane : Nat := rows_per_airplane * seats_per_row
def total_seats : Nat := airplanes_owned * seats_per_airplane
def total_flights_per_day : Nat := airplanes_owned * flights_per_day_per_airplane
def total_passengers_per_day : Nat := total_flights_per_day * total_seats

theorem airline_passenger_capacity :
  total_passengers_per_day = 7000 := sorry

end NUMINAMATH_GPT_airline_passenger_capacity_l1625_162590


namespace NUMINAMATH_GPT_total_age_proof_l1625_162518

noncomputable def total_age : ℕ :=
  let susan := 15
  let arthur := susan + 2
  let bob := 11
  let tom := bob - 3
  let emily := susan / 2
  let david := (arthur + tom + emily) / 3
  susan + arthur + tom + bob + emily + david

theorem total_age_proof : total_age = 70 := by
  unfold total_age
  sorry

end NUMINAMATH_GPT_total_age_proof_l1625_162518


namespace NUMINAMATH_GPT_neg_p_true_l1625_162507

theorem neg_p_true :
  ∀ (x : ℝ), -2 < x ∧ x < 2 → |x - 1| + |x + 2| < 6 :=
by
  sorry

end NUMINAMATH_GPT_neg_p_true_l1625_162507


namespace NUMINAMATH_GPT_solve_for_b_l1625_162511

theorem solve_for_b (b x : ℚ)
  (h₁ : 3 * x + 5 = 1)
  (h₂ : b * x + 6 = 0) :
  b = 9 / 2 :=
sorry   -- The proof is omitted as per instruction.

end NUMINAMATH_GPT_solve_for_b_l1625_162511


namespace NUMINAMATH_GPT_factor_expression_l1625_162589

theorem factor_expression (x : ℝ) : 
  x^2 * (x + 3) + 3 * (x + 3) = (x^2 + 3) * (x + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1625_162589
