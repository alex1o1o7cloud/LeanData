import Mathlib

namespace NUMINAMATH_GPT_convert_spherical_to_rectangular_l1793_179385

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 15 (3 * Real.pi / 4) (Real.pi / 2) = 
    (-15 * Real.sqrt 2 / 2, 15 * Real.sqrt 2 / 2, 0) :=
by 
  sorry

end NUMINAMATH_GPT_convert_spherical_to_rectangular_l1793_179385


namespace NUMINAMATH_GPT_problem_statement_l1793_179305

noncomputable def expr : ℝ :=
  (1 - Real.sqrt 5)^0 + abs (-Real.sqrt 2) - 2 * Real.cos (Real.pi / 4) + (1 / 4 : ℝ)⁻¹

theorem problem_statement : expr = 5 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1793_179305


namespace NUMINAMATH_GPT_product_of_roots_of_cubic_l1793_179328

theorem product_of_roots_of_cubic :
  let a := 2
  let d := 18
  let product_of_roots := -(d / a)
  product_of_roots = -9 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_of_cubic_l1793_179328


namespace NUMINAMATH_GPT_coplanar_vertices_sum_even_l1793_179325

theorem coplanar_vertices_sum_even (a b c d e f g h : ℤ) :
  (∃ (a b c d : ℤ), true ∧ (a + b + c + d) % 2 = 0) :=
sorry

end NUMINAMATH_GPT_coplanar_vertices_sum_even_l1793_179325


namespace NUMINAMATH_GPT_find_X_value_l1793_179333

theorem find_X_value (X : ℝ) : 
  (1.5 * ((3.6 * 0.48 * 2.5) / (0.12 * X * 0.5)) = 1200.0000000000002) → 
  X = 0.225 :=
by
  sorry

end NUMINAMATH_GPT_find_X_value_l1793_179333


namespace NUMINAMATH_GPT_room_total_space_l1793_179391

-- Definitions based on the conditions
def bookshelf_space : ℕ := 80
def reserved_space : ℕ := 160
def number_of_shelves : ℕ := 3

-- The theorem statement
theorem room_total_space : 
  (number_of_shelves * bookshelf_space) + reserved_space = 400 := 
by
  sorry

end NUMINAMATH_GPT_room_total_space_l1793_179391


namespace NUMINAMATH_GPT_no_such_pairs_l1793_179378

theorem no_such_pairs :
  ¬ ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 - 4 * c < 0) ∧ (c^2 - 4 * b < 0) := sorry

end NUMINAMATH_GPT_no_such_pairs_l1793_179378


namespace NUMINAMATH_GPT_quadratic_completing_the_square_q_l1793_179336

theorem quadratic_completing_the_square_q (x p q : ℝ) (h : 4 * x^2 + 8 * x - 468 = 0) :
  (∃ p, (x + p)^2 = q) → q = 116 := sorry

end NUMINAMATH_GPT_quadratic_completing_the_square_q_l1793_179336


namespace NUMINAMATH_GPT_elaineExpenseChanges_l1793_179399

noncomputable def elaineIncomeLastYear : ℝ := 20000 + 5000
noncomputable def elaineExpensesLastYearRent := 0.10 * elaineIncomeLastYear
noncomputable def elaineExpensesLastYearGroceries := 0.20 * elaineIncomeLastYear
noncomputable def elaineExpensesLastYearHealthcare := 0.15 * elaineIncomeLastYear
noncomputable def elaineTotalExpensesLastYear := elaineExpensesLastYearRent + elaineExpensesLastYearGroceries + elaineExpensesLastYearHealthcare
noncomputable def elaineSavingsLastYear := elaineIncomeLastYear - elaineTotalExpensesLastYear

noncomputable def elaineIncomeThisYear : ℝ := 23000 + 10000
noncomputable def elaineExpensesThisYearRent := 0.30 * elaineIncomeThisYear
noncomputable def elaineExpensesThisYearGroceries := 0.25 * elaineIncomeThisYear
noncomputable def elaineExpensesThisYearHealthcare := (0.15 * elaineIncomeThisYear) * 1.10
noncomputable def elaineTotalExpensesThisYear := elaineExpensesThisYearRent + elaineExpensesThisYearGroceries + elaineExpensesThisYearHealthcare
noncomputable def elaineSavingsThisYear := elaineIncomeThisYear - elaineTotalExpensesThisYear

theorem elaineExpenseChanges :
  ( ((elaineExpensesThisYearRent - elaineExpensesLastYearRent) / elaineExpensesLastYearRent) * 100 = 296)
  ∧ ( ((elaineExpensesThisYearGroceries - elaineExpensesLastYearGroceries) / elaineExpensesLastYearGroceries) * 100 = 65)
  ∧ ( ((elaineExpensesThisYearHealthcare - elaineExpensesLastYearHealthcare) / elaineExpensesLastYearHealthcare) * 100 = 45.2)
  ∧ ( (elaineSavingsLastYear / elaineIncomeLastYear) * 100 = 55)
  ∧ ( (elaineSavingsThisYear / elaineIncomeThisYear) * 100 = 28.5)
  ∧ ( (elaineTotalExpensesLastYear / elaineIncomeLastYear) = 0.45 )
  ∧ ( (elaineTotalExpensesThisYear / elaineIncomeThisYear) = 0.715 )
  ∧ ( (elaineSavingsLastYear - elaineSavingsThisYear) = 4345 ∧ ( (55 - ((elaineSavingsThisYear / elaineIncomeThisYear) * 100)) = 26.5 ))
:= by sorry

end NUMINAMATH_GPT_elaineExpenseChanges_l1793_179399


namespace NUMINAMATH_GPT_river_width_l1793_179394

theorem river_width (w : ℕ) (speed_const : ℕ) 
(meeting1_from_nearest_shore : ℕ) (meeting2_from_other_shore : ℕ)
(h1 : speed_const = 1) 
(h2 : meeting1_from_nearest_shore = 720) 
(h3 : meeting2_from_other_shore = 400)
(h4 : 3 * w = 3 * meeting1_from_nearest_shore)
(h5 : 2160 = 2 * w - meeting2_from_other_shore) :
w = 1280 :=
by
  {
      sorry
  }

end NUMINAMATH_GPT_river_width_l1793_179394


namespace NUMINAMATH_GPT_triangle_side_ratio_triangle_area_l1793_179374

-- Definition of Problem 1
theorem triangle_side_ratio {A B C a b c : ℝ} 
  (h1 : 4 * Real.sin A = 3 * Real.sin B)
  (h2 : 2 * a * Real.cos C + 2 * c * Real.cos A = a + c)
  (h3 : a / b = Real.sin A / Real.sin B)
  (h4 : b / c = Real.sin B / Real.sin C)
  : c / b = 5 / 4 :=
sorry

-- Definition of Problem 2
theorem triangle_area {A B C a b c : ℝ} 
  (h1 : C = 2 * Real.pi / 3)
  (h2 : c - a = 8)
  (h3 : 2 * a * Real.cos C + 2 * c * Real.cos A = a + c)
  (h4 : a + c = 2 * b)
  : (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_triangle_side_ratio_triangle_area_l1793_179374


namespace NUMINAMATH_GPT_total_students_correct_l1793_179339

noncomputable def num_roman_numerals : ℕ := 7
noncomputable def sketches_per_numeral : ℕ := 5
noncomputable def total_students : ℕ := 35

theorem total_students_correct : num_roman_numerals * sketches_per_numeral = total_students := by
  sorry

end NUMINAMATH_GPT_total_students_correct_l1793_179339


namespace NUMINAMATH_GPT_discounted_price_l1793_179368

variable (marked_price : ℝ) (discount_rate : ℝ)
variable (marked_price_def : marked_price = 150)
variable (discount_rate_def : discount_rate = 20)

theorem discounted_price (hmp : marked_price = 150) (hdr : discount_rate = 20) : 
  marked_price - (discount_rate / 100) * marked_price = 120 := by
  rw [hmp, hdr]
  sorry

end NUMINAMATH_GPT_discounted_price_l1793_179368


namespace NUMINAMATH_GPT_comparison_of_abc_l1793_179375

noncomputable def a : ℝ := (4 - Real.log 4) / Real.exp 2
noncomputable def b : ℝ := Real.log 2 / 2
noncomputable def c : ℝ := 1 / Real.exp 1

theorem comparison_of_abc : b < a ∧ a < c :=
by
  sorry

end NUMINAMATH_GPT_comparison_of_abc_l1793_179375


namespace NUMINAMATH_GPT_acme_profit_calculation_l1793_179371

theorem acme_profit_calculation :
  let initial_outlay := 12450
  let cost_per_set := 20.75
  let selling_price := 50
  let number_of_sets := 950
  let total_revenue := number_of_sets * selling_price
  let total_manufacturing_costs := initial_outlay + cost_per_set * number_of_sets
  let profit := total_revenue - total_manufacturing_costs 
  profit = 15337.50 := 
by
  sorry

end NUMINAMATH_GPT_acme_profit_calculation_l1793_179371


namespace NUMINAMATH_GPT_feb1_is_wednesday_l1793_179316

-- Define the days of the week as a data type
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define a function that models the backward count for days of the week from a given day
def days_backward (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days % 7 with
  | 0 => start
  | 1 => match start with
         | Sunday => Saturday
         | Monday => Sunday
         | Tuesday => Monday
         | Wednesday => Tuesday
         | Thursday => Wednesday
         | Friday => Thursday
         | Saturday => Friday
  | 2 => match start with
         | Sunday => Friday
         | Monday => Saturday
         | Tuesday => Sunday
         | Wednesday => Monday
         | Thursday => Tuesday
         | Friday => Wednesday
         | Saturday => Thursday
  | 3 => match start with
         | Sunday => Thursday
         | Monday => Friday
         | Tuesday => Saturday
         | Wednesday => Sunday
         | Thursday => Monday
         | Friday => Tuesday
         | Saturday => Wednesday
  | 4 => match start with
         | Sunday => Wednesday
         | Monday => Thursday
         | Tuesday => Friday
         | Wednesday => Saturday
         | Thursday => Sunday
         | Friday => Monday
         | Saturday => Tuesday
  | 5 => match start with
         | Sunday => Tuesday
         | Monday => Wednesday
         | Tuesday => Thursday
         | Wednesday => Friday
         | Thursday => Saturday
         | Friday => Sunday
         | Saturday => Monday
  | 6 => match start with
         | Sunday => Monday
         | Monday => Tuesday
         | Tuesday => Wednesday
         | Wednesday => Thursday
         | Thursday => Friday
         | Friday => Saturday
         | Saturday => Sunday
  | _ => start  -- This case is unreachable because days % 7 is always between 0 and 6

-- Proof statement: given February 28 is a Tuesday, prove that February 1 is a Wednesday
theorem feb1_is_wednesday (h : days_backward Tuesday 27 = Wednesday) : True :=
by
  sorry

end NUMINAMATH_GPT_feb1_is_wednesday_l1793_179316


namespace NUMINAMATH_GPT_Maria_needs_more_l1793_179313

def num_mechanics : Nat := 20
def num_thermodynamics : Nat := 50
def num_optics : Nat := 30
def total_questions : Nat := num_mechanics + num_thermodynamics + num_optics

def correct_mechanics : Nat := (80 * num_mechanics) / 100
def correct_thermodynamics : Nat := (50 * num_thermodynamics) / 100
def correct_optics : Nat := (70 * num_optics) / 100
def correct_total : Nat := correct_mechanics + correct_thermodynamics + correct_optics

def correct_for_passing : Nat := (65 * total_questions) / 100
def additional_needed : Nat := correct_for_passing - correct_total

theorem Maria_needs_more:
  additional_needed = 3 := by
  sorry

end NUMINAMATH_GPT_Maria_needs_more_l1793_179313


namespace NUMINAMATH_GPT_total_cost_of_items_l1793_179379

theorem total_cost_of_items (m n : ℕ) : (8 * m + 5 * n) = 8 * m + 5 * n := 
by sorry

end NUMINAMATH_GPT_total_cost_of_items_l1793_179379


namespace NUMINAMATH_GPT_teairra_shirts_l1793_179311

theorem teairra_shirts (S : ℕ) (pants_total : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) (neither_plaid_nor_purple : ℕ)
  (pants_total_eq : pants_total = 24)
  (plaid_shirts_eq : plaid_shirts = 3)
  (purple_pants_eq : purple_pants = 5)
  (neither_plaid_nor_purple_eq : neither_plaid_nor_purple = 21) :
  (S - plaid_shirts + (pants_total - purple_pants) = neither_plaid_nor_purple) → S = 5 :=
by
  sorry

end NUMINAMATH_GPT_teairra_shirts_l1793_179311


namespace NUMINAMATH_GPT_find_b_d_l1793_179390

theorem find_b_d (b d : ℕ) (h1 : b + d = 41) (h2 : b < d) : 
  (∃! x, b * x * x + 24 * x + d = 0) → (b = 9 ∧ d = 32) :=
by 
  sorry

end NUMINAMATH_GPT_find_b_d_l1793_179390


namespace NUMINAMATH_GPT_initial_men_in_fort_l1793_179359

theorem initial_men_in_fort (M : ℕ) 
  (h1 : ∀ N : ℕ, M * 35 = (N - 25) * 42) 
  (h2 : 10 + 42 = 52) : M = 150 :=
sorry

end NUMINAMATH_GPT_initial_men_in_fort_l1793_179359


namespace NUMINAMATH_GPT_find_n_l1793_179303

theorem find_n (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : ∃ k : ℤ, 721 = n + 360 * k): n = 1 :=
sorry

end NUMINAMATH_GPT_find_n_l1793_179303


namespace NUMINAMATH_GPT_total_books_l1793_179348

theorem total_books (d k g : ℕ) 
  (h1 : d = 6) 
  (h2 : k = d / 2) 
  (h3 : g = 5 * (d + k)) : 
  d + k + g = 54 :=
by
  sorry

end NUMINAMATH_GPT_total_books_l1793_179348


namespace NUMINAMATH_GPT_count_three_digit_multiples_of_35_l1793_179312

theorem count_three_digit_multiples_of_35 : 
  ∃ n : ℕ, n = 26 ∧ ∀ x : ℕ, (100 ≤ x ∧ x < 1000) → (x % 35 = 0 → x = 35 * (3 + ((x / 35) - 3))) := 
sorry

end NUMINAMATH_GPT_count_three_digit_multiples_of_35_l1793_179312


namespace NUMINAMATH_GPT_consecutive_sum_l1793_179383

theorem consecutive_sum (m k : ℕ) (h : (k + 1) * (2 * m + k) = 2000) :
  (m = 1000 ∧ k = 0) ∨ 
  (m = 198 ∧ k = 4) ∨ 
  (m = 28 ∧ k = 24) ∨ 
  (m = 55 ∧ k = 15) :=
by sorry

end NUMINAMATH_GPT_consecutive_sum_l1793_179383


namespace NUMINAMATH_GPT_trig_expression_tangent_l1793_179344

theorem trig_expression_tangent (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / 
  (Real.cos (3 * π / 2 + α) + Real.cos (π - α)) = 1 :=
sorry

end NUMINAMATH_GPT_trig_expression_tangent_l1793_179344


namespace NUMINAMATH_GPT_wire_cut_ratio_l1793_179320

-- Define lengths a and b
variable (a b : ℝ)

-- Define perimeter equal condition
axiom perimeter_eq : 4 * (a / 4) = 6 * (b / 6)

-- The statement to prove
theorem wire_cut_ratio (h : 4 * (a / 4) = 6 * (b / 6)) : a / b = 1 :=
by
  sorry

end NUMINAMATH_GPT_wire_cut_ratio_l1793_179320


namespace NUMINAMATH_GPT_sofia_initial_floor_l1793_179307

theorem sofia_initial_floor (x : ℤ) (h1 : x + 7 - 6 + 5 = 20) : x = 14 := 
sorry

end NUMINAMATH_GPT_sofia_initial_floor_l1793_179307


namespace NUMINAMATH_GPT_mean_of_combined_sets_l1793_179376

theorem mean_of_combined_sets
  (S₁ : Finset ℝ) (S₂ : Finset ℝ)
  (h₁ : S₁.card = 7) (h₂ : S₂.card = 8)
  (mean_S₁ : (S₁.sum id) / S₁.card = 15)
  (mean_S₂ : (S₂.sum id) / S₂.card = 26)
  : (S₁.sum id + S₂.sum id) / (S₁.card + S₂.card) = 20.8667 := 
by
  sorry

end NUMINAMATH_GPT_mean_of_combined_sets_l1793_179376


namespace NUMINAMATH_GPT_find_x_values_l1793_179317

theorem find_x_values (x : ℝ) :
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + 2 * x)) ↔ - (12 / 7 : ℝ) ≤ x ∧ x < - (3 / 4 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_find_x_values_l1793_179317


namespace NUMINAMATH_GPT_annie_total_spent_l1793_179370

-- Define cost of a single television
def cost_per_tv : ℕ := 50
-- Define number of televisions bought
def number_of_tvs : ℕ := 5
-- Define cost of a single figurine
def cost_per_figurine : ℕ := 1
-- Define number of figurines bought
def number_of_figurines : ℕ := 10

-- Define total cost calculation
noncomputable def total_cost : ℕ :=
  number_of_tvs * cost_per_tv + number_of_figurines * cost_per_figurine

theorem annie_total_spent : total_cost = 260 := by
  sorry

end NUMINAMATH_GPT_annie_total_spent_l1793_179370


namespace NUMINAMATH_GPT_percentage_decrease_l1793_179389

theorem percentage_decrease (purchase_price selling_price decrease gross_profit : ℝ)
  (h_purchase : purchase_price = 81)
  (h_markup : selling_price = purchase_price + 0.25 * selling_price)
  (h_gross_profit : gross_profit = 5.40)
  (h_decrease : decrease = 108 - 102.60) :
  (decrease / 108) * 100 = 5 :=
by sorry

end NUMINAMATH_GPT_percentage_decrease_l1793_179389


namespace NUMINAMATH_GPT_ratio_13_2_l1793_179382

def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def total_trees_that_fell : ℕ := 5
def current_total_trees : ℕ := 88

def number_narra_trees_that_fell (N : ℕ) : Prop := N + (N + 1) = total_trees_that_fell
def total_trees_before_typhoon : ℕ := initial_mahogany_trees + initial_narra_trees

def ratio_of_planted_trees_to_narra_fallen (planted : ℕ) (N : ℕ) : Prop := 
  88 - (total_trees_before_typhoon - total_trees_that_fell) = planted ∧ 
  planted / N = 13 / 2

theorem ratio_13_2 : ∃ (planted N : ℕ), 
  number_narra_trees_that_fell N ∧ 
  ratio_of_planted_trees_to_narra_fallen planted N :=
sorry

end NUMINAMATH_GPT_ratio_13_2_l1793_179382


namespace NUMINAMATH_GPT_find_cubic_sum_l1793_179384

theorem find_cubic_sum
  {a b : ℝ}
  (h1 : a^5 - a^4 * b - a^4 + a - b - 1 = 0)
  (h2 : 2 * a - 3 * b = 1) :
  a^3 + b^3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_cubic_sum_l1793_179384


namespace NUMINAMATH_GPT_factor_of_polynomial_l1793_179340

def polynomial (x : ℝ) : ℝ := x^4 - 4*x^2 + 16
def q1 (x : ℝ) : ℝ := x^2 + 4
def q2 (x : ℝ) : ℝ := x - 2
def q3 (x : ℝ) : ℝ := x^2 - 4*x + 4
def q4 (x : ℝ) : ℝ := x^2 + 4*x + 4

theorem factor_of_polynomial : (∃ (f g : ℝ → ℝ), polynomial x = f x * g x) ∧ (q4 = f ∨ q4 = g) := by sorry

end NUMINAMATH_GPT_factor_of_polynomial_l1793_179340


namespace NUMINAMATH_GPT_fractions_product_l1793_179310

theorem fractions_product :
  (4 / 2) * (8 / 4) * (9 / 3) * (18 / 6) * (16 / 8) * (24 / 12) * (30 / 15) * (36 / 18) = 576 := by
  sorry

end NUMINAMATH_GPT_fractions_product_l1793_179310


namespace NUMINAMATH_GPT_constant_term_expansion_l1793_179360

theorem constant_term_expansion (n : ℕ) (hn : n = 9) :
  y^3 * (x + 1 / (x^2 * y))^n = 84 :=
by sorry

end NUMINAMATH_GPT_constant_term_expansion_l1793_179360


namespace NUMINAMATH_GPT_maximum_positive_factors_l1793_179365

theorem maximum_positive_factors (b n : ℕ) (hb : 0 < b ∧ b ≤ 20) (hn : 0 < n ∧ n ≤ 15) :
  ∃ k, (k = b^n) ∧ (∀ m, m = b^n → m.factors.count ≤ 61) :=
sorry

end NUMINAMATH_GPT_maximum_positive_factors_l1793_179365


namespace NUMINAMATH_GPT_count_four_digit_numbers_l1793_179301

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_count_four_digit_numbers_l1793_179301


namespace NUMINAMATH_GPT_fred_money_last_week_l1793_179349

theorem fred_money_last_week (F_current F_earned F_last_week : ℕ) 
  (h_current : F_current = 86)
  (h_earned : F_earned = 63)
  (h_last_week : F_last_week = 23) :
  F_current - F_earned = F_last_week := 
by
  sorry

end NUMINAMATH_GPT_fred_money_last_week_l1793_179349


namespace NUMINAMATH_GPT_chess_players_swim_not_every_swimmer_plays_tennis_tennis_players_play_chess_l1793_179354

variables (Bodyguards : Type)
variables (U S T : Bodyguards → Prop)

-- Conditions
axiom cond1 : ∀ x, (T x → (S x → U x))
axiom cond2 : ∀ x, (S x → (U x ∨ T x))
axiom cond3 : ∀ x, (¬ U x ∧ T x → S x)

-- To prove
theorem chess_players_swim : ∀ x, (S x → U x) := by
  sorry

theorem not_every_swimmer_plays_tennis : ¬ ∀ x, (U x → T x) := by
  sorry

theorem tennis_players_play_chess : ∀ x, (T x → S x) := by
  sorry

end NUMINAMATH_GPT_chess_players_swim_not_every_swimmer_plays_tennis_tennis_players_play_chess_l1793_179354


namespace NUMINAMATH_GPT_probability_of_both_selected_l1793_179397

variable (P_ram : ℚ) (P_ravi : ℚ) (P_both : ℚ)

def selection_probability (P_ram : ℚ) (P_ravi : ℚ) : ℚ :=
  P_ram * P_ravi

theorem probability_of_both_selected (h1 : P_ram = 3/7) (h2 : P_ravi = 1/5) :
  selection_probability P_ram P_ravi = P_both :=
by
  sorry

end NUMINAMATH_GPT_probability_of_both_selected_l1793_179397


namespace NUMINAMATH_GPT_race_positions_l1793_179358

theorem race_positions :
  ∀ (M J T R H D : ℕ),
    (M = J + 3) →
    (J = T + 1) →
    (T = R + 3) →
    (H = R + 5) →
    (D = H + 4) →
    (M = 9) →
    H = 7 :=
by sorry

end NUMINAMATH_GPT_race_positions_l1793_179358


namespace NUMINAMATH_GPT_sum_of_fractions_eq_decimal_l1793_179377

theorem sum_of_fractions_eq_decimal :
  (3 / 100) + (5 / 1000) + (7 / 10000) = 0.0357 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_eq_decimal_l1793_179377


namespace NUMINAMATH_GPT_lemonade_quart_calculation_l1793_179387

-- Define the conditions
def water_parts := 5
def lemon_juice_parts := 3
def total_parts := water_parts + lemon_juice_parts

def gallons := 2
def quarts_per_gallon := 4
def total_quarts := gallons * quarts_per_gallon
def quarts_per_part := total_quarts / total_parts

-- Proof problem
theorem lemonade_quart_calculation :
  let water_quarts := water_parts * quarts_per_part
  let lemon_juice_quarts := lemon_juice_parts * quarts_per_part
  water_quarts = 5 ∧ lemon_juice_quarts = 3 :=
by
  let water_quarts := water_parts * quarts_per_part
  let lemon_juice_quarts := lemon_juice_parts * quarts_per_part
  have h_w : water_quarts = 5 := sorry
  have h_l : lemon_juice_quarts = 3 := sorry
  exact ⟨h_w, h_l⟩

end NUMINAMATH_GPT_lemonade_quart_calculation_l1793_179387


namespace NUMINAMATH_GPT_subtract_one_from_solution_l1793_179364

theorem subtract_one_from_solution (x : ℝ) (h : 15 * x = 45) : (x - 1) = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_subtract_one_from_solution_l1793_179364


namespace NUMINAMATH_GPT_tanC_over_tanA_plus_tanC_over_tanB_l1793_179322

theorem tanC_over_tanA_plus_tanC_over_tanB {a b c : ℝ} (A B C : ℝ) (h : a / b + b / a = 6 * Real.cos C) (acute_triangle : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) :
  (Real.tan C / Real.tan A) + (Real.tan C / Real.tan B) = 4 :=
sorry -- Proof not required

end NUMINAMATH_GPT_tanC_over_tanA_plus_tanC_over_tanB_l1793_179322


namespace NUMINAMATH_GPT_calculate_correctly_l1793_179366

theorem calculate_correctly (x : ℕ) (h : 2 * x = 22) : 20 * x + 3 = 223 :=
by
  sorry

end NUMINAMATH_GPT_calculate_correctly_l1793_179366


namespace NUMINAMATH_GPT_find_slope_l1793_179318

theorem find_slope (m : ℝ) : 
    (∀ x : ℝ, (2, 13) = (x, 5 * x + 3)) → 
    (∀ x : ℝ, (2, 13) = (x, m * x + 1)) → 
    m = 6 :=
by 
  intros hP hQ
  have h_inter_p := hP 2
  have h_inter_q := hQ 2
  simp at h_inter_p h_inter_q
  have : 13 = 5 * 2 + 3 := h_inter_p
  have : 13 = m * 2 + 1 := h_inter_q
  linarith

end NUMINAMATH_GPT_find_slope_l1793_179318


namespace NUMINAMATH_GPT_fourth_square_area_l1793_179315

theorem fourth_square_area (AB BC CD AC x : ℝ) 
  (h_AB : AB^2 = 49) 
  (h_BC : BC^2 = 25) 
  (h_CD : CD^2 = 64) 
  (h_AC1 : AC^2 = AB^2 + BC^2) 
  (h_AC2 : AC^2 = CD^2 + x^2) :
  x^2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_fourth_square_area_l1793_179315


namespace NUMINAMATH_GPT_goods_train_speed_is_52_l1793_179331

def man_train_speed : ℕ := 60 -- speed of the man's train in km/h
def goods_train_length : ℕ := 280 -- length of the goods train in meters
def time_to_pass : ℕ := 9 -- time for the goods train to pass the man in seconds
def relative_speed_kmph : ℕ := (goods_train_length * 3600) / (time_to_pass * 1000) -- relative speed in km/h, calculated as (0.28 km / (9/3600) h)
def goods_train_speed : ℕ := relative_speed_kmph - man_train_speed -- speed of the goods train in km/h

theorem goods_train_speed_is_52 : goods_train_speed = 52 := by
  sorry

end NUMINAMATH_GPT_goods_train_speed_is_52_l1793_179331


namespace NUMINAMATH_GPT_half_sum_of_squares_of_even_or_odd_l1793_179323

theorem half_sum_of_squares_of_even_or_odd (n1 n2 : ℤ) (a b : ℤ) :
  (n1 % 2 = 0 ∧ n2 % 2 = 0 ∧ n1 = 2*a ∧ n2 = 2*b ∨
   n1 % 2 = 1 ∧ n2 % 2 = 1 ∧ n1 = 2*a + 1 ∧ n2 = 2*b + 1) →
  ∃ x y : ℤ, (n1^2 + n2^2) / 2 = x^2 + y^2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_half_sum_of_squares_of_even_or_odd_l1793_179323


namespace NUMINAMATH_GPT_sum_of_x_and_y_l1793_179395

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 := 
sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l1793_179395


namespace NUMINAMATH_GPT_solve_cubic_eq_a_solve_cubic_eq_b_solve_cubic_eq_c_l1793_179350

-- For the first polynomial equation
theorem solve_cubic_eq_a (x : ℝ) : x^3 - 3 * x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

-- For the second polynomial equation
theorem solve_cubic_eq_b (x : ℝ) : x^3 - 19 * x - 30 = 0 ↔ x = 5 ∨ x = -2 ∨ x = -3 :=
by sorry

-- For the third polynomial equation
theorem solve_cubic_eq_c (x : ℝ) : x^3 + 4 * x^2 + 6 * x + 4 = 0 ↔ x = -2 :=
by sorry

end NUMINAMATH_GPT_solve_cubic_eq_a_solve_cubic_eq_b_solve_cubic_eq_c_l1793_179350


namespace NUMINAMATH_GPT_eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256_l1793_179304

theorem eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256 :
  11^2 + 2 * 11 * 5 + 5^2 = 256 := by
  sorry

end NUMINAMATH_GPT_eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256_l1793_179304


namespace NUMINAMATH_GPT_max_n_for_neg_sum_correct_l1793_179398

noncomputable def max_n_for_neg_sum (S : ℕ → ℤ) : ℕ :=
  if h₁ : S 19 > 0 then
    if h₂ : S 20 < 0 then
      11
    else 0  -- default value
  else 0  -- default value

theorem max_n_for_neg_sum_correct (S : ℕ → ℤ) (h₁ : S 19 > 0) (h₂ : S 20 < 0) : max_n_for_neg_sum S = 11 :=
by
  sorry

end NUMINAMATH_GPT_max_n_for_neg_sum_correct_l1793_179398


namespace NUMINAMATH_GPT_fatima_total_donation_l1793_179309

theorem fatima_total_donation :
  let cloth1 := 100
  let cloth1_piece1 := 0.40 * cloth1
  let cloth1_piece2 := 0.30 * cloth1
  let cloth1_piece3 := 0.30 * cloth1
  let donation1 := cloth1_piece2 + cloth1_piece3

  let cloth2 := 65
  let cloth2_piece1 := 0.55 * cloth2
  let cloth2_piece2 := 0.45 * cloth2
  let donation2 := cloth2_piece2

  let cloth3 := 48
  let cloth3_piece1 := 0.60 * cloth3
  let cloth3_piece2 := 0.40 * cloth3
  let donation3 := cloth3_piece2

  donation1 + donation2 + donation3 = 108.45 :=
by
  sorry

end NUMINAMATH_GPT_fatima_total_donation_l1793_179309


namespace NUMINAMATH_GPT_solve_fractional_equation_1_solve_fractional_equation_2_l1793_179380

-- Proof Problem 1
theorem solve_fractional_equation_1 (x : ℝ) (h : 6 * x - 2 ≠ 0) :
  (3 / 2 - 1 / (3 * x - 1) = 5 / (6 * x - 2)) ↔ (x = 10 / 9) :=
sorry

-- Proof Problem 2
theorem solve_fractional_equation_2 (x : ℝ) (h1 : 3 * x - 6 ≠ 0) :
  (5 * x - 4) / (x - 2) = (4 * x + 10) / (3 * x - 6) - 1 → false :=
sorry

end NUMINAMATH_GPT_solve_fractional_equation_1_solve_fractional_equation_2_l1793_179380


namespace NUMINAMATH_GPT_trip_to_Atlanta_equals_Boston_l1793_179361

def distance_to_Boston : ℕ := 840
def daily_distance : ℕ := 40
def num_days (distance : ℕ) (daily : ℕ) : ℕ := distance / daily
def distance_to_Atlanta (days : ℕ) (daily : ℕ) : ℕ := days * daily

theorem trip_to_Atlanta_equals_Boston :
  distance_to_Atlanta (num_days distance_to_Boston daily_distance) daily_distance = distance_to_Boston :=
by
  -- Here we would insert the proof.
  sorry

end NUMINAMATH_GPT_trip_to_Atlanta_equals_Boston_l1793_179361


namespace NUMINAMATH_GPT_problem1_l1793_179392

theorem problem1 : abs (-3) + (-1: ℤ)^2021 * (Real.pi - 3.14)^0 - (- (1/2: ℝ))⁻¹ = 4 := 
  sorry

end NUMINAMATH_GPT_problem1_l1793_179392


namespace NUMINAMATH_GPT_red_candies_difference_l1793_179342

def jar1_ratio_red : ℕ := 7
def jar1_ratio_yellow : ℕ := 3
def jar2_ratio_red : ℕ := 5
def jar2_ratio_yellow : ℕ := 4
def total_yellow : ℕ := 108

theorem red_candies_difference :
  ∀ (x y : ℚ), jar1_ratio_yellow * x + jar2_ratio_yellow * y = total_yellow ∧ jar1_ratio_red + jar1_ratio_yellow = jar2_ratio_red + jar2_ratio_yellow → 10 * x = 9 * y → 7 * x - 5 * y = 21 := 
by sorry

end NUMINAMATH_GPT_red_candies_difference_l1793_179342


namespace NUMINAMATH_GPT_algebraic_inequality_solution_l1793_179327

theorem algebraic_inequality_solution (x : ℝ) : (1 + 2 * x ≤ 8 + 3 * x) → (x ≥ -7) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_inequality_solution_l1793_179327


namespace NUMINAMATH_GPT_find_angle_B_max_value_a_squared_plus_c_squared_l1793_179343

variable {A B C : ℝ} -- Angles A, B, C in radians
variable {a b c : ℝ} -- Sides opposite to these angles

-- Problem 1
theorem find_angle_B (h : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos B) : B = Real.pi / 3 :=
by
  sorry -- Proof is not needed

-- Problem 2
theorem max_value_a_squared_plus_c_squared (h : b = Real.sqrt 3)
  (h' : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos B) : (a^2 + c^2) ≤ 6 :=
by
  sorry -- Proof is not needed

end NUMINAMATH_GPT_find_angle_B_max_value_a_squared_plus_c_squared_l1793_179343


namespace NUMINAMATH_GPT_total_investment_with_interest_l1793_179302

theorem total_investment_with_interest
  (total_investment : ℝ)
  (amount_at_3_percent : ℝ)
  (interest_rate_3 : ℝ)
  (interest_rate_5 : ℝ)
  (remaining_amount : ℝ)
  (interest_3 : ℝ)
  (interest_5 : ℝ) :
  total_investment = 1000 →
  amount_at_3_percent = 199.99999999999983 →
  interest_rate_3 = 0.03 →
  interest_rate_5 = 0.05 →
  remaining_amount = total_investment - amount_at_3_percent →
  interest_3 = amount_at_3_percent * interest_rate_3 →
  interest_5 = remaining_amount * interest_rate_5 →
  total_investment + interest_3 + remaining_amount + interest_5 = 1046 :=
by
  intros H1 H2 H3 H4 H5 H6 H7
  sorry

end NUMINAMATH_GPT_total_investment_with_interest_l1793_179302


namespace NUMINAMATH_GPT_range_of_f_minus_2_l1793_179308

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_minus_2 (a b : ℝ) (h1 : 1 ≤ f (-1) a b) (h2 : f (-1) a b ≤ 2) (h3 : 2 ≤ f 1 a b) (h4 : f 1 a b ≤ 4) :
  6 ≤ f (-2) a b ∧ f (-2) a b ≤ 10 :=
sorry

end NUMINAMATH_GPT_range_of_f_minus_2_l1793_179308


namespace NUMINAMATH_GPT_solution_set_inequality_l1793_179352

theorem solution_set_inequality (x : ℝ) : 
  (2 < 1 / (x - 1) ∧ 1 / (x - 1) < 3) ↔ (4 / 3 < x ∧ x < 3 / 2) := 
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1793_179352


namespace NUMINAMATH_GPT_find_original_volume_l1793_179355

theorem find_original_volume
  (V : ℝ)
  (h1 : V - (3 / 4) * V = (1 / 4) * V)
  (h2 : (1 / 4) * V - (3 / 4) * ((1 / 4) * V) = (1 / 16) * V)
  (h3 : (1 / 16) * V = 0.2) :
  V = 3.2 :=
by 
  -- Proof skipped, as the assistant is instructed to provide only the statement 
  sorry

end NUMINAMATH_GPT_find_original_volume_l1793_179355


namespace NUMINAMATH_GPT_gas_pipe_probability_l1793_179332

-- Define the conditions as Lean hypotheses
theorem gas_pipe_probability (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hxy : x + y ≤ 100) (h25x : 25 ≤ x) (h25y : 25 ≤ y)
    (h100xy : 75 ≥ x + y) :
  ∃ (p : ℝ), p = 1/16 :=
by
  sorry

end NUMINAMATH_GPT_gas_pipe_probability_l1793_179332


namespace NUMINAMATH_GPT_min_value_fractions_l1793_179338

theorem min_value_fractions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  2 ≤ (5 * z) / (3 * x + y) + (5 * x) / (y + 3 * z) + (2 * y) / (x + z) :=
by sorry

end NUMINAMATH_GPT_min_value_fractions_l1793_179338


namespace NUMINAMATH_GPT_octagon_ratio_l1793_179351

theorem octagon_ratio (total_area : ℝ) (area_below_PQ : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) (XQ QY : ℝ) :
  total_area = 10 ∧
  area_below_PQ = 5 ∧
  triangle_base = 5 ∧
  triangle_height = 8 / 5 ∧
  area_below_PQ = 1 + (1 / 2) * triangle_base * triangle_height ∧
  XQ + QY = triangle_base ∧
  (1 / 2) * (XQ + QY) * triangle_height = 5
  → (XQ / QY) = 2 / 3 := 
sorry

end NUMINAMATH_GPT_octagon_ratio_l1793_179351


namespace NUMINAMATH_GPT_triangle_perimeter_l1793_179300

theorem triangle_perimeter
  (a : ℝ) (a_gt_5 : a > 5)
  (ellipse : ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / 25 = 1)
  (dist_foci : 8 = 2 * 4) :
  4 * Real.sqrt (41) = 4 * Real.sqrt (41) := by
sorry

end NUMINAMATH_GPT_triangle_perimeter_l1793_179300


namespace NUMINAMATH_GPT_cubic_sum_l1793_179369

theorem cubic_sum (p q r : ℝ) (h1 : p + q + r = 4) (h2 : p * q + q * r + r * p = 7) (h3 : p * q * r = -10) :
  p ^ 3 + q ^ 3 + r ^ 3 = 154 := 
by sorry

end NUMINAMATH_GPT_cubic_sum_l1793_179369


namespace NUMINAMATH_GPT_combined_stripes_is_22_l1793_179335

-- Definition of stripes per shoe for each person based on the conditions
def stripes_per_shoe_Olga : ℕ := 3
def stripes_per_shoe_Rick : ℕ := stripes_per_shoe_Olga - 1
def stripes_per_shoe_Hortense : ℕ := stripes_per_shoe_Olga * 2

-- The total combined number of stripes on all shoes for Olga, Rick, and Hortense
def total_stripes : ℕ := 2 * (stripes_per_shoe_Olga + stripes_per_shoe_Rick + stripes_per_shoe_Hortense)

-- The statement to prove that the total number of stripes on all their shoes is 22
theorem combined_stripes_is_22 : total_stripes = 22 :=
by
  sorry

end NUMINAMATH_GPT_combined_stripes_is_22_l1793_179335


namespace NUMINAMATH_GPT_mass_percentage_B_in_H3BO3_l1793_179341

noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_B : ℝ := 10.81
noncomputable def atomic_mass_O : ℝ := 16.00
noncomputable def molar_mass_H3BO3 : ℝ := 3 * atomic_mass_H + atomic_mass_B + 3 * atomic_mass_O

theorem mass_percentage_B_in_H3BO3 : (atomic_mass_B / molar_mass_H3BO3) * 100 = 17.48 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_B_in_H3BO3_l1793_179341


namespace NUMINAMATH_GPT_inequality_proof_l1793_179381

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c)) ≥ (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l1793_179381


namespace NUMINAMATH_GPT_race_distance_l1793_179337

theorem race_distance (a b c : ℝ) (d : ℝ) 
  (h1 : d / a = (d - 15) / b)
  (h2 : d / b = (d - 30) / c)
  (h3 : d / a = (d - 40) / c) : 
  d = 90 :=
by sorry

end NUMINAMATH_GPT_race_distance_l1793_179337


namespace NUMINAMATH_GPT_parakeets_in_each_cage_l1793_179363

variable (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ)

-- Given conditions
def total_parrots (num_cages parrots_per_cage : ℕ) : ℕ := num_cages * parrots_per_cage
def total_parakeets (total_birds total_parrots : ℕ) : ℕ := total_birds - total_parrots
def parakeets_per_cage (total_parakeets num_cages : ℕ) : ℕ := total_parakeets / num_cages

-- Theorem: Number of parakeets in each cage is 7
theorem parakeets_in_each_cage (h1 : num_cages = 8) (h2 : parrots_per_cage = 2) (h3 : total_birds = 72) : 
  parakeets_per_cage (total_parakeets total_birds (total_parrots num_cages parrots_per_cage)) num_cages = 7 :=
by
  sorry

end NUMINAMATH_GPT_parakeets_in_each_cage_l1793_179363


namespace NUMINAMATH_GPT_original_number_l1793_179324

theorem original_number (n : ℚ) (h : (3 * (n + 3) - 2) / 3 = 10) : n = 23 / 3 := 
sorry

end NUMINAMATH_GPT_original_number_l1793_179324


namespace NUMINAMATH_GPT_fraction_changes_l1793_179353

theorem fraction_changes (x y : ℝ) (h : 0 < x ∧ 0 < y) :
  (x + y) / (x * y) = 2 * ((2 * x + 2 * y) / (2 * x * 2 * y)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_changes_l1793_179353


namespace NUMINAMATH_GPT_abs_neg_three_l1793_179314

theorem abs_neg_three : abs (-3) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_abs_neg_three_l1793_179314


namespace NUMINAMATH_GPT_wire_not_used_is_20_l1793_179306

def initial_wire_length : ℕ := 50
def number_of_parts : ℕ := 5
def parts_used : ℕ := 3

def length_of_each_part (total_length : ℕ) (parts : ℕ) : ℕ := total_length / parts
def length_used (length_each_part : ℕ) (used_parts : ℕ) : ℕ := length_each_part * used_parts
def wire_not_used (total_length : ℕ) (used_length : ℕ) : ℕ := total_length - used_length

theorem wire_not_used_is_20 : 
  wire_not_used initial_wire_length 
    (length_used 
      (length_of_each_part initial_wire_length number_of_parts) 
    parts_used) = 20 := by
  sorry

end NUMINAMATH_GPT_wire_not_used_is_20_l1793_179306


namespace NUMINAMATH_GPT_devin_initial_height_l1793_179367

theorem devin_initial_height (h : ℝ) (p : ℝ) (p' : ℝ) :
  (p = 10 / 100) →
  (p' = (h - 66) / 100) →
  (h + 3 = 68) →
  (p + p' * (h + 3 - 66) = 30 / 100) →
  h = 68 :=
by
  intros hp hp' hg pt
  sorry

end NUMINAMATH_GPT_devin_initial_height_l1793_179367


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1793_179373

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1793_179373


namespace NUMINAMATH_GPT_triangle_side_c_l1793_179329

noncomputable def angle_B_eq_2A (A B : ℝ) := B = 2 * A
noncomputable def side_a_eq_1 (a : ℝ) := a = 1
noncomputable def side_b_eq_sqrt3 (b : ℝ) := b = Real.sqrt 3

noncomputable def find_side_c (A B C a b c : ℝ) :=
  angle_B_eq_2A A B ∧
  side_a_eq_1 a ∧
  side_b_eq_sqrt3 b →
  c = 2

theorem triangle_side_c (A B C a b c : ℝ) : find_side_c A B C a b c :=
by sorry

end NUMINAMATH_GPT_triangle_side_c_l1793_179329


namespace NUMINAMATH_GPT_solve_for_b_l1793_179388

theorem solve_for_b (b : ℝ) (h : b + b / 4 = 10 / 4) : b = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_b_l1793_179388


namespace NUMINAMATH_GPT_grace_walks_distance_l1793_179356

theorem grace_walks_distance
  (south_blocks west_blocks : ℕ)
  (block_length_in_miles : ℚ)
  (h_south_blocks : south_blocks = 4)
  (h_west_blocks : west_blocks = 8)
  (h_block_length : block_length_in_miles = 1 / 4)
  : ((south_blocks + west_blocks) * block_length_in_miles = 3) :=
by 
  sorry

end NUMINAMATH_GPT_grace_walks_distance_l1793_179356


namespace NUMINAMATH_GPT_selling_price_per_machine_l1793_179326

theorem selling_price_per_machine (parts_cost patent_cost : ℕ) (num_machines : ℕ) 
  (hc1 : parts_cost = 3600) (hc2 : patent_cost = 4500) (hc3 : num_machines = 45) :
  (parts_cost + patent_cost) / num_machines = 180 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_per_machine_l1793_179326


namespace NUMINAMATH_GPT_total_time_to_4864_and_back_l1793_179345

variable (speed_boat : ℝ) (speed_stream : ℝ) (distance : ℝ)
variable (Sboat : speed_boat = 14) (Sstream : speed_stream = 1.2) (Dist : distance = 4864)

theorem total_time_to_4864_and_back :
  let speed_downstream := speed_boat + speed_stream
  let speed_upstream := speed_boat - speed_stream
  let time_downstream := distance / speed_downstream
  let time_upstream := distance / speed_upstream
  let total_time := time_downstream + time_upstream
  total_time = 700 :=
by
  sorry

end NUMINAMATH_GPT_total_time_to_4864_and_back_l1793_179345


namespace NUMINAMATH_GPT_probability_event_comparison_l1793_179346

theorem probability_event_comparison (m n : ℕ) :
  let P_A := (2 * m * n) / (m + n)^2
  let P_B := (m^2 + n^2) / (m + n)^2
  P_A ≤ P_B ∧ (P_A = P_B ↔ m = n) :=
by
  sorry

end NUMINAMATH_GPT_probability_event_comparison_l1793_179346


namespace NUMINAMATH_GPT_cos_two_thirds_pi_l1793_179362

theorem cos_two_thirds_pi : Real.cos (2 / 3 * Real.pi) = -1 / 2 :=
by sorry

end NUMINAMATH_GPT_cos_two_thirds_pi_l1793_179362


namespace NUMINAMATH_GPT_find_certain_number_l1793_179330

-- Define the conditions
variable (m : ℕ)
variable (h_lcm : Nat.lcm 24 m = 48)
variable (h_gcd : Nat.gcd 24 m = 8)

-- State the theorem to prove
theorem find_certain_number (h_lcm : Nat.lcm 24 m = 48) (h_gcd : Nat.gcd 24 m = 8) : m = 16 :=
sorry

end NUMINAMATH_GPT_find_certain_number_l1793_179330


namespace NUMINAMATH_GPT_fraction_multiplication_subtraction_l1793_179347

theorem fraction_multiplication_subtraction :
  (3 + 1 / 117) * (4 + 1 / 119) - (2 - 1 / 117) * (6 - 1 / 119) - (5 / 119) = 10 / 117 :=
by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_subtraction_l1793_179347


namespace NUMINAMATH_GPT_tangent_line_y_intercept_l1793_179386

noncomputable def y_intercept_tangent_line (R1_center R2_center : ℝ × ℝ)
  (R1_radius R2_radius : ℝ) : ℝ :=
if R1_center = (3,0) ∧ R2_center = (8,0) ∧ R1_radius = 3 ∧ R2_radius = 2
then 15 * Real.sqrt 26 / 26
else 0

theorem tangent_line_y_intercept : 
  y_intercept_tangent_line (3,0) (8,0) 3 2 = 15 * Real.sqrt 26 / 26 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_tangent_line_y_intercept_l1793_179386


namespace NUMINAMATH_GPT_geometric_series_3000_terms_sum_l1793_179321

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_series_3000_terms_sum
    (a r : ℝ)
    (h_r : r ≠ 1)
    (sum_1000 : geometric_sum a r 1000 = 500)
    (sum_2000 : geometric_sum a r 2000 = 950) :
  geometric_sum a r 3000 = 1355 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_series_3000_terms_sum_l1793_179321


namespace NUMINAMATH_GPT_two_perfect_squares_not_two_perfect_cubes_l1793_179334

-- Define the initial conditions as Lean assertions
def isSumOfTwoPerfectSquares (n : ℕ) := ∃ a b : ℕ, n = a^2 + b^2

def isSumOfTwoPerfectCubes (n : ℕ) := ∃ a b : ℕ, n = a^3 + b^3

-- Lean 4 statement to show 2005^2005 is a sum of two perfect squares
theorem two_perfect_squares :
  isSumOfTwoPerfectSquares (2005^2005) :=
sorry

-- Lean 4 statement to show 2005^2005 is not a sum of two perfect cubes
theorem not_two_perfect_cubes :
  ¬ isSumOfTwoPerfectCubes (2005^2005) :=
sorry

end NUMINAMATH_GPT_two_perfect_squares_not_two_perfect_cubes_l1793_179334


namespace NUMINAMATH_GPT_sufficient_condition_not_necessary_condition_l1793_179393

variables (p q : Prop)
def φ := ¬p ∧ ¬q
def ψ := ¬p

theorem sufficient_condition : φ p q → ψ p := 
sorry

theorem not_necessary_condition : ψ p → ¬ (φ p q) :=
sorry

end NUMINAMATH_GPT_sufficient_condition_not_necessary_condition_l1793_179393


namespace NUMINAMATH_GPT_cylinder_height_relationship_l1793_179357

variables (π r₁ r₂ h₁ h₂ : ℝ)

theorem cylinder_height_relationship
  (h_volume_eq : π * r₁^2 * h₁ = π * r₂^2 * h₂)
  (h_radius_rel : r₂ = 1.2 * r₁) :
  h₁ = 1.44 * h₂ :=
by {
  sorry -- proof not required as per instructions
}

end NUMINAMATH_GPT_cylinder_height_relationship_l1793_179357


namespace NUMINAMATH_GPT_julia_played_with_34_kids_l1793_179396

-- Define the number of kids Julia played with on each day
def kidsMonday : Nat := 17
def kidsTuesday : Nat := 15
def kidsWednesday : Nat := 2

-- Define the total number of kids Julia played with
def totalKids : Nat := kidsMonday + kidsTuesday + kidsWednesday

-- Prove given conditions
theorem julia_played_with_34_kids :
  totalKids = 34 :=
by
  sorry

end NUMINAMATH_GPT_julia_played_with_34_kids_l1793_179396


namespace NUMINAMATH_GPT_anyas_hair_loss_l1793_179319

theorem anyas_hair_loss (H : ℝ) 
  (washes_hair_loss : H > 0) 
  (brushes_hair_loss : H / 2 > 0) 
  (grows_back : ∃ h : ℝ, h = 49 ∧ H + H / 2 + 1 = h) :
  H = 32 :=
by
  sorry

end NUMINAMATH_GPT_anyas_hair_loss_l1793_179319


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l1793_179372

theorem sufficient_and_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l1793_179372
