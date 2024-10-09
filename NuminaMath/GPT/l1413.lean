import Mathlib

namespace sum_of_first_five_multiples_of_15_l1413_141368

theorem sum_of_first_five_multiples_of_15 : (15 + 30 + 45 + 60 + 75) = 225 :=
by sorry

end sum_of_first_five_multiples_of_15_l1413_141368


namespace problem_l1413_141355

theorem problem (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  x + (x ^ 3 / y ^ 2) + (y ^ 3 / x ^ 2) + y = 440 := by
  sorry

end problem_l1413_141355


namespace smallest_sum_of_two_perfect_squares_l1413_141349

theorem smallest_sum_of_two_perfect_squares (x y : ℕ) (h : x^2 - y^2 = 143) :
  x + y = 13 ∧ x - y = 11 → x^2 + y^2 = 145 :=
by
  -- Add this placeholder "sorry" to skip the proof, as required.
  sorry

end smallest_sum_of_two_perfect_squares_l1413_141349


namespace solve_equation_l1413_141318

theorem solve_equation (x y : ℝ) : 3 * x^2 - 12 * y^2 = 0 ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  sorry

end solve_equation_l1413_141318


namespace cost_of_TOP_book_l1413_141384

theorem cost_of_TOP_book (T : ℝ) (h1 : T = 8)
  (abc_cost : ℝ := 23)
  (top_books_sold : ℝ := 13)
  (abc_books_sold : ℝ := 4)
  (earnings_difference : ℝ := 12)
  (h2 : top_books_sold * T - abc_books_sold * abc_cost = earnings_difference) :
  T = 8 := 
by
  sorry

end cost_of_TOP_book_l1413_141384


namespace volume_of_regular_triangular_pyramid_l1413_141357

noncomputable def pyramid_volume (R φ : ℝ) : ℝ :=
  (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)

theorem volume_of_regular_triangular_pyramid (R φ : ℝ) 
  (cond1 : R > 0)
  (cond2: 0 < φ ∧ φ < π) :
  ∃ V, V = pyramid_volume R φ := by
    use (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)
    sorry

end volume_of_regular_triangular_pyramid_l1413_141357


namespace areas_of_triangles_l1413_141347

-- Define the condition that the gcd of a, b, and c is 1
def gcd_one (a b c : ℤ) : Prop := Int.gcd (Int.gcd a b) c = 1

-- Define the set of possible areas for triangles in E
def f_E : Set ℝ :=
  { area | ∃ (a b c : ℤ), gcd_one a b c ∧ area = (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) }

theorem areas_of_triangles : 
  f_E = { area | ∃ (a b c : ℤ), gcd_one a b c ∧ area = (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) } :=
by {
  sorry
}

end areas_of_triangles_l1413_141347


namespace PlatformC_location_l1413_141381

noncomputable def PlatformA : ℝ := 9
noncomputable def PlatformB : ℝ := 1 / 3
noncomputable def PlatformC : ℝ := 7
noncomputable def AB := PlatformA - PlatformB
noncomputable def AC := PlatformA - PlatformC

theorem PlatformC_location :
  AB = (13 / 3) * AC → PlatformC = 7 :=
by
  intro h
  simp [AB, AC, PlatformA, PlatformB, PlatformC] at h
  sorry

end PlatformC_location_l1413_141381


namespace find_f_10_l1413_141324

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) (hx : x ≠ 0) : f x = f (1 / x) * Real.log x + 10

theorem find_f_10 : f 10 = 10 :=
by
  sorry

end find_f_10_l1413_141324


namespace bob_second_week_hours_l1413_141325

theorem bob_second_week_hours (total_earnings : ℕ) (total_hours_first_week : ℕ) (regular_hours_pay : ℕ) 
  (overtime_hours_pay : ℕ) (regular_hours_max : ℕ) (total_hours_overtime_first_week : ℕ) 
  (earnings_first_week : ℕ) (earnings_second_week : ℕ) : 
  total_earnings = 472 →
  total_hours_first_week = 44 →
  regular_hours_pay = 5 →
  overtime_hours_pay = 6 →
  regular_hours_max = 40 →
  total_hours_overtime_first_week = total_hours_first_week - regular_hours_max →
  earnings_first_week = regular_hours_max * regular_hours_pay + 
                          total_hours_overtime_first_week * overtime_hours_pay →
  earnings_second_week = total_earnings - earnings_first_week → 
  ∃ h, earnings_second_week = h * regular_hours_pay ∨ 
  earnings_second_week = (regular_hours_max * regular_hours_pay + (h - regular_hours_max) * overtime_hours_pay) ∧ 
  h = 48 :=
by 
  intros 
  sorry 

end bob_second_week_hours_l1413_141325


namespace series_solution_l1413_141362

theorem series_solution (r : ℝ) (h : (r^3 - r^2 + (1 / 4) * r - 1 = 0) ∧ r > 0) :
  (∑' (n : ℕ), (n + 1) * r^(3 * (n + 1))) = 16 * r :=
by
  sorry

end series_solution_l1413_141362


namespace exists_a_b_l1413_141386

theorem exists_a_b (r : Fin 5 → ℝ) : ∃ (i j : Fin 5), i ≠ j ∧ 0 ≤ (r i - r j) / (1 + r i * r j) ∧ (r i - r j) / (1 + r i * r j) ≤ 1 :=
by
  sorry

end exists_a_b_l1413_141386


namespace point_of_tangency_l1413_141360

theorem point_of_tangency : 
    ∃ (m n : ℝ), 
    (∀ x : ℝ, x ≠ 0 → n = 1 / m ∧ (-1 / m^2) = (n - 2) / (m - 0)) ∧ 
    m = 1 ∧ n = 1 :=
by
  sorry

end point_of_tangency_l1413_141360


namespace polygon_sides_l1413_141316

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by 
  sorry

end polygon_sides_l1413_141316


namespace find_k_l1413_141309

theorem find_k (k : ℝ) (α β : ℝ) 
  (h1 : α + β = -k) 
  (h2 : α * β = 12) 
  (h3 : α + 7 + β + 7 = k) : 
  k = -7 :=
sorry

end find_k_l1413_141309


namespace ratio_students_l1413_141365

theorem ratio_students
  (finley_students : ℕ)
  (johnson_students : ℕ)
  (h_finley : finley_students = 24)
  (h_johnson : johnson_students = 22)
  : (johnson_students : ℚ) / ((finley_students / 2 : ℕ) : ℚ) = 11 / 6 :=
by
  sorry

end ratio_students_l1413_141365


namespace sec_120_eq_neg_2_l1413_141337

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_120_eq_neg_2 : sec 120 = -2 := by
  sorry

end sec_120_eq_neg_2_l1413_141337


namespace sum_of_two_numbers_is_147_l1413_141320

theorem sum_of_two_numbers_is_147 (A B : ℝ) (h1 : A + B = 147) (h2 : A = 0.375 * B + 4) :
  A + B = 147 :=
by
  sorry

end sum_of_two_numbers_is_147_l1413_141320


namespace simplest_square_root_l1413_141319

noncomputable def sqrt8 : ℝ := Real.sqrt 8
noncomputable def inv_sqrt2 : ℝ := 1 / Real.sqrt 2
noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt_inv2 : ℝ := Real.sqrt (1 / 2)

theorem simplest_square_root : sqrt2 = Real.sqrt 2 := 
  sorry

end simplest_square_root_l1413_141319


namespace how_many_months_to_buy_tv_l1413_141323

-- Definitions based on given conditions
def monthly_income : ℕ := 30000
def food_expenses : ℕ := 15000
def utilities_expenses : ℕ := 5000
def other_expenses : ℕ := 2500

def total_expenses := food_expenses + utilities_expenses + other_expenses
def current_savings : ℕ := 10000
def tv_cost : ℕ := 25000
def monthly_savings := monthly_income - total_expenses

-- Theorem statement based on the problem
theorem how_many_months_to_buy_tv 
    (H_income : monthly_income = 30000)
    (H_food : food_expenses = 15000)
    (H_utilities : utilities_expenses = 5000)
    (H_other : other_expenses = 2500)
    (H_savings : current_savings = 10000)
    (H_tv_cost : tv_cost = 25000)
    : (tv_cost - current_savings) / monthly_savings = 2 :=
by
  sorry

end how_many_months_to_buy_tv_l1413_141323


namespace calories_per_person_l1413_141359

-- Definitions based on the conditions from a)
def oranges : ℕ := 5
def pieces_per_orange : ℕ := 8
def people : ℕ := 4
def calories_per_orange : ℝ := 80

-- Theorem based on the equivalent proof problem
theorem calories_per_person : 
    ((oranges * pieces_per_orange) / people) / pieces_per_orange * calories_per_orange = 100 := 
by
  sorry

end calories_per_person_l1413_141359


namespace sixty_percent_of_40_greater_than_four_fifths_of_25_l1413_141344

theorem sixty_percent_of_40_greater_than_four_fifths_of_25 :
  (60 / 100 : ℝ) * 40 - (4 / 5 : ℝ) * 25 = 4 := by
  sorry

end sixty_percent_of_40_greater_than_four_fifths_of_25_l1413_141344


namespace total_bread_amt_l1413_141358

-- Define the conditions
variables (bread_dinner bread_lunch bread_breakfast total_bread : ℕ)
axiom bread_dinner_amt : bread_dinner = 240
axiom dinner_lunch_ratio : bread_dinner = 8 * bread_lunch
axiom dinner_breakfast_ratio : bread_dinner = 6 * bread_breakfast

-- The proof statement
theorem total_bread_amt : total_bread = bread_dinner + bread_lunch + bread_breakfast → total_bread = 310 :=
by
  -- Use the axioms and the given conditions to derive the statement
  sorry

end total_bread_amt_l1413_141358


namespace parallel_lines_find_m_l1413_141332

theorem parallel_lines_find_m :
  (∀ (m : ℝ), ∀ (x y : ℝ), (2 * x + (m + 1) * y + 4 = 0) ∧ (m * x + 3 * y - 2 = 0) → (m = -3 ∨ m = 2)) := 
sorry

end parallel_lines_find_m_l1413_141332


namespace slope_of_line_l1413_141301

theorem slope_of_line : ∀ (x y : ℝ), (6 * x + 10 * y = 30) → (y = -((3 / 5) * x) + 3) :=
by
  -- Proof needs to be filled out
  sorry

end slope_of_line_l1413_141301


namespace find_discount_percentage_l1413_141350

noncomputable def discount_percentage (P B S : ℝ) (H1 : B = P * (1 - D / 100)) (H2 : S = B * 1.5) (H3 : S - P = P * 0.19999999999999996) : ℝ :=
D

theorem find_discount_percentage (P B S : ℝ) (H1 : B = P * (1 - (60 / 100))) (H2 : S = B * 1.5) (H3 : S - P = P * 0.19999999999999996) : 
  discount_percentage P B S H1 H2 H3 = 60 := sorry

end find_discount_percentage_l1413_141350


namespace skylar_current_age_l1413_141374

noncomputable def skylar_age_now (donation_start_age : ℕ) (annual_donation total_donation : ℕ) : ℕ :=
  donation_start_age + total_donation / annual_donation

theorem skylar_current_age : skylar_age_now 13 5000 105000 = 34 := by
  -- Proof follows from the conditions
  sorry

end skylar_current_age_l1413_141374


namespace quadrilateral_area_l1413_141304

theorem quadrilateral_area 
  (p : ℝ) (hp : p > 0)
  (P : ℝ × ℝ) (hP : P = (1, 1 / 4))
  (focus : ℝ × ℝ) (hfocus : focus = (0, 1))
  (directrix : ℝ → Prop) (hdirectrix : ∀ y, directrix y ↔ y = 1)
  (F : ℝ × ℝ) (hF : F = (0, 1))
  (M : ℝ × ℝ) (hM : M = (0, 1))
  (Q : ℝ × ℝ) 
  (PQ : ℝ)
  (area : ℝ) 
  (harea : area = 13 / 8) :
  ∃ (PQMF : ℝ), PQMF = 13 / 8 :=
sorry

end quadrilateral_area_l1413_141304


namespace probability_win_l1413_141303

theorem probability_win (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose) = 3 / 8 :=
by
  rw [h]
  norm_num

end probability_win_l1413_141303


namespace parabola_chords_reciprocal_sum_l1413_141308

theorem parabola_chords_reciprocal_sum (x y : ℝ) (AB CD : ℝ) (p : ℝ) :
  (y = (4 : ℝ) * x) ∧ (AB ≠ 0) ∧ (CD ≠ 0) ∧
  (p = (2 : ℝ)) ∧
  (|AB| = (2 * p / (Real.sin (Real.pi / 4))^2)) ∧ 
  (|CD| = (2 * p / (Real.cos (Real.pi / 4))^2)) →
  (1 / |AB| + 1 / |CD| = 1 / 4) :=
by
  sorry

end parabola_chords_reciprocal_sum_l1413_141308


namespace radius_of_shorter_cone_l1413_141334

theorem radius_of_shorter_cone {h : ℝ} (h_ne_zero : h ≠ 0) :
  ∀ r : ℝ, ∀ V_taller V_shorter : ℝ,
   (V_taller = (1/3) * π * (5 ^ 2) * (4 * h)) →
   (V_shorter = (1/3) * π * (r ^ 2) * h) →
   V_taller = V_shorter →
   r = 10 :=
by
  intros
  sorry

end radius_of_shorter_cone_l1413_141334


namespace ratio_values_l1413_141394

theorem ratio_values (x y z : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) 
  (h₀ : (x + y) / z = (y + z) / x) (h₀' : (y + z) / x = (z + x) / y) :
  ∃ a : ℝ, a = -1 ∨ a = 8 :=
sorry

end ratio_values_l1413_141394


namespace problem_statement_l1413_141327

-- Define rational number representations for points A, B, and C
def a : ℚ := (-4)^2 - 8

-- Define that B and C are opposites
def are_opposites (b c : ℚ) : Prop := b = -c

-- Define the distance condition
def distance_is_three (a c : ℚ) : Prop := |c - a| = 3

-- Main theorem statement
theorem problem_statement :
  (∃ b c : ℚ, are_opposites b c ∧ distance_is_three a c ∧ -a^2 + b - c = -74) ∨
  (∃ b c : ℚ, are_opposites b c ∧ distance_is_three a c ∧ -a^2 + b - c = -86) :=
sorry

end problem_statement_l1413_141327


namespace sin_inequality_solution_set_l1413_141387

theorem sin_inequality_solution_set : 
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x < - Real.sqrt 3 / 2} =
  {x : ℝ | (4 * Real.pi / 3) < x ∧ x < (5 * Real.pi / 3)} := by
  sorry

end sin_inequality_solution_set_l1413_141387


namespace log_cut_piece_weight_l1413_141338

-- Defining the conditions

def log_length : ℕ := 20
def half_log_length : ℕ := log_length / 2
def weight_per_foot : ℕ := 150

-- The main theorem stating the problem
theorem log_cut_piece_weight : (half_log_length * weight_per_foot) = 1500 := 
by 
  sorry

end log_cut_piece_weight_l1413_141338


namespace min_value_of_a2_b2_l1413_141330

noncomputable def f (x a b : ℝ) := Real.exp x + a * x + b

theorem min_value_of_a2_b2 {a b : ℝ} (h : ∃ t ∈ Set.Icc (1 : ℝ) (3 : ℝ), f t a b = 0) :
  a^2 + b^2 ≥ (Real.exp 1)^2 / 2 :=
by
  sorry

end min_value_of_a2_b2_l1413_141330


namespace tangent_sufficient_but_not_necessary_condition_l1413_141353

noncomputable def tangent_condition (m : ℝ) : Prop :=
  let line := fun (x y : ℝ) => x + y - m = 0
  let circle := fun (x y : ℝ) => (x - 1) ^ 2 + (y - 1) ^ 2 = 2
  ∃ (x y: ℝ), line x y ∧ circle x y -- A line and circle are tangent if they intersect exactly at one point

theorem tangent_sufficient_but_not_necessary_condition (m : ℝ) :
  (tangent_condition m) ↔ (m = 0 ∨ m = 4) := by
  sorry

end tangent_sufficient_but_not_necessary_condition_l1413_141353


namespace Thabo_books_problem_l1413_141367

theorem Thabo_books_problem 
  (P F : ℕ)
  (H1 : 180 = F + P + 30)
  (H2 : F = 2 * P)
  (H3 : P > 30) :
  P - 30 = 20 := 
sorry

end Thabo_books_problem_l1413_141367


namespace solution_to_inequality_system_l1413_141311

theorem solution_to_inequality_system (x : ℝ) :
  (x + 3 ≥ 2 ∧ (3 * x - 1) / 2 < 4) ↔ -1 ≤ x ∧ x < 3 :=
by
  sorry

end solution_to_inequality_system_l1413_141311


namespace like_terms_exponents_l1413_141328

theorem like_terms_exponents (m n : ℕ) (x y : ℝ) (h : 2 * x^(2*m) * y^6 = -3 * x^8 * y^(2*n)) : m = 4 ∧ n = 3 :=
by 
  sorry

end like_terms_exponents_l1413_141328


namespace vikas_rank_among_boys_l1413_141376

def vikas_rank_overall := 9
def tanvi_rank_overall := 17
def girls_between := 2
def vikas_rank_top_boys := 4
def vikas_rank_bottom_overall := 18

theorem vikas_rank_among_boys (vikas_rank_overall tanvi_rank_overall girls_between vikas_rank_top_boys vikas_rank_bottom_overall : ℕ) :
  vikas_rank_top_boys = 4 := by
  sorry

end vikas_rank_among_boys_l1413_141376


namespace boys_assigned_l1413_141392

theorem boys_assigned (B G : ℕ) (h1 : B + G = 18) (h2 : B = G - 2) : B = 8 :=
sorry

end boys_assigned_l1413_141392


namespace set_inclusion_l1413_141346

-- Definitions based on given conditions
def setA (x : ℝ) : Prop := 0 < x ∧ x < 2
def setB (x : ℝ) : Prop := x > 0

-- Statement of the proof problem
theorem set_inclusion : ∀ x, setA x → setB x :=
by
  intros x h
  sorry

end set_inclusion_l1413_141346


namespace non_increasing_condition_l1413_141396

variable {a b : ℝ} (f : ℝ → ℝ)

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem non_increasing_condition (h₀ : ∀ x y, a ≤ x → x < y → y ≤ b → ¬ (f x > f y)) :
  ¬ increasing_on_interval f a b :=
by
  intro h1
  have : ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y := h1
  exact sorry

end non_increasing_condition_l1413_141396


namespace initial_butterfat_percentage_l1413_141389

theorem initial_butterfat_percentage (P : ℝ) :
  let initial_butterfat := (P / 100) * 1000
  let removed_butterfat := (23 / 100) * 50
  let remaining_volume := 1000 - 50
  let desired_butterfat := (3 / 100) * remaining_volume
  initial_butterfat - removed_butterfat = desired_butterfat
→ P = 4 :=
by
  intros
  let initial_butterfat := (P / 100) * 1000
  let removed_butterfat := (23 / 100) * 50
  let remaining_volume := 1000 - 50
  let desired_butterfat := (3 / 100) * remaining_volume
  sorry

end initial_butterfat_percentage_l1413_141389


namespace intersection_property_l1413_141377

def universal_set : Set ℝ := Set.univ

def M : Set ℝ := {-1, 1, 2, 4}

def N : Set ℝ := {x : ℝ | x > 2}

theorem intersection_property : (M ∩ N) = {4} := by
  sorry

end intersection_property_l1413_141377


namespace factor_expression_l1413_141385

variable (y : ℝ)

theorem factor_expression : 
  6*y*(y + 2) + 15*(y + 2) + 12 = 3*(2*y + 5)*(y + 2) :=
sorry

end factor_expression_l1413_141385


namespace at_least_one_not_less_than_100_l1413_141326

-- Defining the original propositions
def p : Prop := ∀ (A_score : ℕ), A_score ≥ 100
def q : Prop := ∀ (B_score : ℕ), B_score < 100

-- Assertion to be proved in Lean
theorem at_least_one_not_less_than_100 (h1 : p) (h2 : q) : p ∨ ¬q := 
sorry

end at_least_one_not_less_than_100_l1413_141326


namespace arithmetic_sequence_formula_l1413_141335

theorem arithmetic_sequence_formula (x : ℤ) (a : ℕ → ℤ) 
  (h1 : a 1 = x - 1) (h2 : a 2 = x + 1) (h3 : a 3 = 2 * x + 3) :
  ∃ c d : ℤ, (∀ n : ℕ, a n = c + d * (n - 1)) ∧ ∀ n : ℕ, a n = 2 * n - 3 :=
by {
  sorry
}

end arithmetic_sequence_formula_l1413_141335


namespace number_of_children_riding_tricycles_l1413_141310

-- Definitions
def bicycles_wheels := 2
def tricycles_wheels := 3

def adults := 6
def total_wheels := 57

-- Problem statement
theorem number_of_children_riding_tricycles (c : ℕ) (H : 12 + 3 * c = total_wheels) : c = 15 :=
by
  sorry

end number_of_children_riding_tricycles_l1413_141310


namespace transformed_graph_area_l1413_141370

theorem transformed_graph_area (g : ℝ → ℝ) (a b : ℝ)
  (h_area_g : ∫ x in a..b, g x = 15) :
  ∫ x in a..b, 2 * g (x + 3) = 30 := 
sorry

end transformed_graph_area_l1413_141370


namespace minimum_xy_l1413_141329

noncomputable def f (x y : ℝ) := 2 * x + y + 6

theorem minimum_xy (x y : ℝ) (h : 0 < x ∧ 0 < y) (h1 : f x y = x * y) : x * y = 18 :=
by
  sorry

end minimum_xy_l1413_141329


namespace total_oranges_is_correct_l1413_141379

/-- Define the number of boxes and the number of oranges per box -/
def boxes : ℕ := 7
def oranges_per_box : ℕ := 6

/-- Prove that the total number of oranges is 42 -/
theorem total_oranges_is_correct : boxes * oranges_per_box = 42 := 
by 
  sorry

end total_oranges_is_correct_l1413_141379


namespace giraffes_difference_l1413_141305

theorem giraffes_difference :
  ∃ n : ℕ, (300 = 3 * n) ∧ (300 - n = 200) :=
by
  sorry

end giraffes_difference_l1413_141305


namespace arithmetic_sequence_product_l1413_141393

noncomputable def b (n : ℕ) : ℤ := sorry -- define the arithmetic sequence

theorem arithmetic_sequence_product (d : ℤ) 
  (h_seq : ∀ n, b (n + 1) = b n + d)
  (h_inc : ∀ m n, m < n → b m < b n)
  (h_prod : b 4 * b 5 = 30) :
  b 3 * b 6 = -1652 ∨ b 3 * b 6 = -308 ∨ b 3 * b 6 = -68 ∨ b 3 * b 6 = 28 := 
sorry

end arithmetic_sequence_product_l1413_141393


namespace simplify_expression_l1413_141321

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ( ((a ^ (4 / 3 / 5)) ^ (3 / 2)) / ((a ^ (4 / 1 / 5)) ^ 3) ) /
  ( ((a * (a ^ (2 / 3) * b ^ (1 / 3))) ^ (1 / 2)) ^ 4) * 
  (a ^ (1 / 4) * b ^ (1 / 8)) ^ 6 = 1 / ((a ^ (2 / 12)) * (b ^ (1 / 12))) :=
by
  sorry

end simplify_expression_l1413_141321


namespace area_ratio_S_T_l1413_141352

open Set

def T : Set (ℝ × ℝ × ℝ) := {p | let (x, y, z) := p; x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 1}

def supports (p q : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  let (a, b, c) := q
  (x ≥ a ∧ y ≥ b) ∨ (x ≥ a ∧ z ≥ c) ∨ (y ≥ b ∧ z ≥ c)

def S : Set (ℝ × ℝ × ℝ) := {p ∈ T | supports p (1/4, 1/4, 1/2)}

theorem area_ratio_S_T : ∃ k : ℝ, k = 3 / 4 ∧
  ∃ (area_T area_S : ℝ), area_T ≠ 0 ∧ (area_S / area_T = k) := sorry

end area_ratio_S_T_l1413_141352


namespace xiaoliang_prob_correct_l1413_141380

def initial_box_setup : List (Nat × Nat) := [(1, 2), (2, 2), (3, 2), (4, 2)]

def xiaoming_draw : List Nat := [1, 1, 3]

def remaining_balls_after_xiaoming : List (Nat × Nat) := [(1, 0), (2, 2), (3, 1), (4, 2)]

def remaining_ball_count (balls : List (Nat × Nat)) : Nat :=
  balls.foldl (λ acc ⟨_, count⟩ => acc + count) 0

theorem xiaoliang_prob_correct :
  (1 : ℚ) / (remaining_ball_count remaining_balls_after_xiaoming) = 1 / 5 :=
by
  sorry

end xiaoliang_prob_correct_l1413_141380


namespace common_difference_arithmetic_sequence_l1413_141348

noncomputable def first_term : ℕ := 5
noncomputable def last_term : ℕ := 50
noncomputable def sum_terms : ℕ := 275

theorem common_difference_arithmetic_sequence :
  ∃ d n, (last_term = first_term + (n - 1) * d) ∧ (sum_terms = n * (first_term + last_term) / 2) ∧ d = 5 :=
  sorry

end common_difference_arithmetic_sequence_l1413_141348


namespace case1_BL_case2_BL_l1413_141391

variable (AD BD BL AL : ℝ)

theorem case1_BL
  (h₁ : AD = 6)
  (h₂ : BD = 12 * Real.sqrt 3)
  (h₃ : AB = 6 * Real.sqrt 13)
  (hADBL : AD / BD = AL / BL)
  (h4 : BL = 2 * AL)
  : BL = 16 * Real.sqrt 3 - 12 := by
  sorry

theorem case2_BL
  (h₁ : AD = 6)
  (h₂ : BD = 12 * Real.sqrt 6)
  (h₃ : AB = 30)
  (hADBL : AD / BD = AL / BL)
  (h4 : BL = 4 * AL)
  : BL = (16 * Real.sqrt 6 - 6) / 5 := by
  sorry

end case1_BL_case2_BL_l1413_141391


namespace solve_complex_addition_l1413_141366

def complex_addition_problem : Prop :=
  let B := Complex.mk 3 (-2)
  let Q := Complex.mk (-5) 1
  let R := Complex.mk 1 (-2)
  let T := Complex.mk 4 3
  B - Q + R + T = Complex.mk 13 (-2)

theorem solve_complex_addition : complex_addition_problem := by
  sorry

end solve_complex_addition_l1413_141366


namespace sum_of_numbers_with_lcm_and_ratio_l1413_141312

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ) (h_lcm : Nat.lcm a b = 60) (h_ratio : a = 2 * b / 3) : a + b = 50 := 
by
  sorry

end sum_of_numbers_with_lcm_and_ratio_l1413_141312


namespace area_per_car_l1413_141397

/-- Given the length and width of the parking lot, 
and the percentage of usable area, 
and the number of cars that can be parked,
prove that the area per car is as expected. -/
theorem area_per_car 
  (length width : ℝ) 
  (usable_percentage : ℝ) 
  (number_of_cars : ℕ) 
  (h_length : length = 400) 
  (h_width : width = 500) 
  (h_usable_percentage : usable_percentage = 0.80) 
  (h_number_of_cars : number_of_cars = 16000) :
  (length * width * usable_percentage) / number_of_cars = 10 :=
by
  sorry

end area_per_car_l1413_141397


namespace calculate_mirror_area_l1413_141361

def outer_frame_width : ℝ := 65
def outer_frame_height : ℝ := 85
def frame_width : ℝ := 15

def mirror_width : ℝ := outer_frame_width - 2 * frame_width
def mirror_height : ℝ := outer_frame_height - 2 * frame_width
def mirror_area : ℝ := mirror_width * mirror_height

theorem calculate_mirror_area : mirror_area = 1925 := by
  sorry

end calculate_mirror_area_l1413_141361


namespace marked_elements_duplicate_l1413_141307

open Nat

def table : Matrix (Fin 4) (Fin 10) ℕ := ![
  ![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
  ![9, 0, 1, 2, 3, 4, 5, 6, 7, 8], 
  ![8, 9, 0, 1, 2, 3, 4, 5, 6, 7], 
  ![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
]

theorem marked_elements_duplicate 
  (marked : Fin 4 → Fin 10) 
  (h_marked_unique_row : ∀ i1 i2, i1 ≠ i2 → marked i1 ≠ marked i2)
  (h_marked_unique_col : ∀ j, ∃ i, marked i = j) :
  ∃ i1 i2, i1 ≠ i2 ∧ table i1 (marked i1) = table i2 (marked i2) := sorry

end marked_elements_duplicate_l1413_141307


namespace pool_capacity_percentage_l1413_141372

theorem pool_capacity_percentage
  (rate : ℕ := 60) -- cubic feet per minute
  (time : ℕ := 800) -- minutes
  (width : ℕ := 60) -- feet
  (length : ℕ := 100) -- feet
  (depth : ℕ := 10) -- feet
  : (rate * time * 100) / (width * length * depth) = 8 := by
{
  sorry
}

end pool_capacity_percentage_l1413_141372


namespace rebecca_marbles_l1413_141341

theorem rebecca_marbles (M : ℕ) (h1 : 20 = M + 14) : M = 6 :=
by
  sorry

end rebecca_marbles_l1413_141341


namespace measure_A_l1413_141300

noncomputable def angle_A (C B A : ℝ) : Prop :=
  C = 3 / 2 * B ∧ B = 30 ∧ A = 180 - B - C

theorem measure_A (A B C : ℝ) (h : angle_A C B A) : A = 105 :=
by
  -- Extract conditions from h
  obtain ⟨h1, h2, h3⟩ := h
  
  -- Use the conditions to prove the thesis
  simp [h1, h2, h3]
  sorry

end measure_A_l1413_141300


namespace maximize_side_area_of_cylinder_l1413_141313

noncomputable def radius_of_cylinder (x : ℝ) : ℝ :=
  (6 - x) / 3

noncomputable def side_area_of_cylinder (x : ℝ) : ℝ :=
  2 * Real.pi * (radius_of_cylinder x) * x

theorem maximize_side_area_of_cylinder :
  ∃ x : ℝ, (0 < x ∧ x < 6) ∧ (∀ y : ℝ, (0 < y ∧ y < 6) → (side_area_of_cylinder y ≤ side_area_of_cylinder x)) ∧ x = 3 :=
by
  sorry

end maximize_side_area_of_cylinder_l1413_141313


namespace rowing_upstream_speed_l1413_141339

-- Define the speed of the man in still water
def V_m : ℝ := 45

-- Define the speed of the man rowing downstream
def V_downstream : ℝ := 65

-- Define the speed of the stream
def V_s : ℝ := V_downstream - V_m

-- Define the speed of the man rowing upstream
def V_upstream : ℝ := V_m - V_s

-- Prove that the speed of the man rowing upstream is 25 kmph
theorem rowing_upstream_speed :
  V_upstream = 25 := by
  sorry

end rowing_upstream_speed_l1413_141339


namespace smallest_b_l1413_141322

theorem smallest_b (a b : ℕ) (pos_a : 0 < a) (pos_b : 0 < b)
    (h1 : a - b = 4)
    (h2 : gcd ((a^3 + b^3) / (a + b)) (a * b) = 4) : b = 2 :=
sorry

end smallest_b_l1413_141322


namespace speed_of_first_part_l1413_141383

theorem speed_of_first_part (v : ℝ) (h1 : v > 0)
  (h_total_distance : 50 = 25 + 25)
  (h_average_speed : 44 = 50 / ((25 / v) + (25 / 33))) :
  v = 66 :=
by sorry

end speed_of_first_part_l1413_141383


namespace sequence_general_term_l1413_141315

theorem sequence_general_term {a : ℕ → ℝ} (S : ℕ → ℝ) (n : ℕ) 
  (hS : ∀ n, S n = 4 * a n - 3) :
  a n = (4/3)^(n-1) :=
sorry

end sequence_general_term_l1413_141315


namespace discount_percentage_l1413_141364

variable {P P_b P_s : ℝ}
variable {D : ℝ}

theorem discount_percentage (P_s_eq_bought : P_s = 1.60 * P_b)
  (P_s_eq_original : P_s = 1.52 * P)
  (P_b_eq_discount : P_b = P * (1 - D)) :
  D = 0.05 := by
sorry

end discount_percentage_l1413_141364


namespace triangle_area_l1413_141399

theorem triangle_area
  (a b : ℝ)
  (C : ℝ)
  (h₁ : a = 2)
  (h₂ : b = 3)
  (h₃ : C = π / 3) :
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_l1413_141399


namespace exists_non_prime_form_l1413_141314

theorem exists_non_prime_form (n : ℕ) : ∃ n : ℕ, ¬Nat.Prime (n^2 + n + 41) :=
sorry

end exists_non_prime_form_l1413_141314


namespace no_real_solutions_for_equation_l1413_141375

theorem no_real_solutions_for_equation:
  ∀ x : ℝ, (3 * x / (x^2 + 2 * x + 4) + 4 * x / (x^2 - 4 * x + 5) = 1) →
  (¬(∃ x : ℝ, 3 * x / (x^2 + 2 * x + 4) + 4 * x / (x^2 - 4 * x + 5) = 1)) :=
by
  sorry

end no_real_solutions_for_equation_l1413_141375


namespace low_degree_polys_condition_l1413_141378

theorem low_degree_polys_condition :
  ∃ (f : Polynomial ℤ), ∃ (g : Polynomial ℤ), ∃ (h : Polynomial ℤ),
    (f = Polynomial.X ^ 3 + Polynomial.X ^ 2 + Polynomial.X + 1 ∨
          f = Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + 2 * Polynomial.X + 2 ∨
          f = 2 * Polynomial.X ^ 3 + Polynomial.X ^ 2 + 2 * Polynomial.X + 1 ∨
          f = 2 * Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + Polynomial.X + 2) ∧
          f ^ 4 + 2 * f + 2 = (Polynomial.X ^ 4 + 2 * Polynomial.X ^ 2 + 2) * g + 3 * h := 
sorry

end low_degree_polys_condition_l1413_141378


namespace perimeter_of_equilateral_triangle_l1413_141302

theorem perimeter_of_equilateral_triangle (a : ℕ) (h1 : a = 12) (h2 : ∀ sides, sides = 3) : 
  3 * a = 36 := 
by
  sorry

end perimeter_of_equilateral_triangle_l1413_141302


namespace product_xyz_l1413_141306

noncomputable def x : ℚ := 97 / 12
noncomputable def n : ℚ := 8 * x
noncomputable def y : ℚ := n + 7
noncomputable def z : ℚ := n - 11

theorem product_xyz 
  (h1: x + y + z = 190)
  (h2: n = 8 * x)
  (h3: n = y - 7)
  (h4: n = z + 11) : 
  x * y * z = (97 * 215 * 161) / 108 := 
by 
  sorry

end product_xyz_l1413_141306


namespace solve_ab_eq_l1413_141388

theorem solve_ab_eq (a b : ℕ) (h : a^b + a + b = b^a) : a = 5 ∧ b = 2 :=
sorry

end solve_ab_eq_l1413_141388


namespace other_continents_passengers_l1413_141342

def passengers_from_other_continents (T N_A E A As : ℕ) : ℕ := T - (N_A + E + A + As)

theorem other_continents_passengers :
  passengers_from_other_continents 108 (108 / 12) (108 / 4) (108 / 9) (108 / 6) = 42 :=
by
  -- Proof goes here
  sorry

end other_continents_passengers_l1413_141342


namespace unique_solution_condition_l1413_141317

theorem unique_solution_condition {a b : ℝ} : (∃ x : ℝ, 4 * x - 7 + a = b * x + 4) ↔ b ≠ 4 :=
by
  sorry

end unique_solution_condition_l1413_141317


namespace smallest_positive_period_of_f_is_pi_f_at_pi_over_2_not_sqrt_3_over_2_max_value_of_f_on_interval_l1413_141331

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem smallest_positive_period_of_f_is_pi : 
  (∀ x, f (x + Real.pi) = f x) ∧ (∀ ε > 0, ε < Real.pi → ∃ x, f (x + ε) ≠ f x) :=
by
  sorry

theorem f_at_pi_over_2_not_sqrt_3_over_2 : f (Real.pi / 2) ≠ Real.sqrt 3 / 2 :=
by
  sorry

theorem max_value_of_f_on_interval : 
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 6 → f x ≤ 1 :=
by
  sorry

end smallest_positive_period_of_f_is_pi_f_at_pi_over_2_not_sqrt_3_over_2_max_value_of_f_on_interval_l1413_141331


namespace total_budget_l1413_141336

-- Define the conditions for the problem
def fiscal_months : ℕ := 12
def total_spent_at_six_months : ℕ := 6580
def over_budget_at_six_months : ℕ := 280

-- Calculate the total budget for the project
theorem total_budget (budget : ℕ) 
  (h : 6 * (total_spent_at_six_months - over_budget_at_six_months) * 2 = budget) 
  : budget = 12600 := 
  by
    -- Proof will be here
    sorry

end total_budget_l1413_141336


namespace student_avg_greater_actual_avg_l1413_141371

theorem student_avg_greater_actual_avg
  (x y z : ℝ)
  (hxy : x < y)
  (hyz : y < z) :
  (x + y + 2 * z) / 4 > (x + y + z) / 3 := by
  sorry

end student_avg_greater_actual_avg_l1413_141371


namespace fill_box_with_L_blocks_l1413_141363

theorem fill_box_with_L_blocks (m n k : ℕ) 
  (hm : m > 1) (hn : n > 1) (hk : k > 1) (hk_div3 : k % 3 = 0) : 
  ∃ (fill : ℕ → ℕ → ℕ → Prop), fill m n k → True := 
by
  sorry

end fill_box_with_L_blocks_l1413_141363


namespace firm_partners_l1413_141333

theorem firm_partners
  (P A : ℕ)
  (h1 : P / A = 2 / 63)
  (h2 : P / (A + 35) = 1 / 34) :
  P = 14 :=
by
  sorry

end firm_partners_l1413_141333


namespace domain_shift_l1413_141351

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the domain of f
def domain_f : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

-- State the problem in Lean
theorem domain_shift (hf : ∀ x, f x ∈ domain_f) : 
    { x | 1 ≤ x ∧ x ≤ 2 } = { x | ∃ y, y ∈ domain_f ∧ x = y + 1 } :=
by
  sorry

end domain_shift_l1413_141351


namespace length_of_largest_square_l1413_141382

-- Define the conditions of the problem
def side_length_of_shaded_square : ℕ := 10
def side_length_of_largest_square : ℕ := 24

-- The statement to prove
theorem length_of_largest_square (x : ℕ) (h1 : x = side_length_of_shaded_square) : 
  4 * x = side_length_of_largest_square :=
  by
  -- Insert the proof here
  sorry

end length_of_largest_square_l1413_141382


namespace sum_infinite_series_l1413_141398

theorem sum_infinite_series :
  (∑' n : ℕ, 1 / (n + 1) / (n + 4)) = 1 / 3 :=
sorry

end sum_infinite_series_l1413_141398


namespace parallel_vectors_m_eq_neg3_l1413_141390

theorem parallel_vectors_m_eq_neg3
  (m : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (m + 1, -3))
  (h2 : b = (2, 3))
  (h3 : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  m = -3 := 
sorry

end parallel_vectors_m_eq_neg3_l1413_141390


namespace fliers_sent_afternoon_fraction_l1413_141345

-- Definitions of given conditions
def total_fliers : ℕ := 2000
def fliers_morning_fraction : ℚ := 1 / 10
def remaining_fliers_next_day : ℕ := 1350

-- Helper definitions based on conditions
def fliers_sent_morning := total_fliers * fliers_morning_fraction
def fliers_after_morning := total_fliers - fliers_sent_morning
def fliers_sent_afternoon := fliers_after_morning - remaining_fliers_next_day

-- Theorem stating the required proof
theorem fliers_sent_afternoon_fraction :
  fliers_sent_afternoon / fliers_after_morning = 1 / 4 :=
sorry

end fliers_sent_afternoon_fraction_l1413_141345


namespace sum_of_odd_coefficients_l1413_141369

theorem sum_of_odd_coefficients (a : ℝ) (h : (a + 1) * 16 = 32) : a = 3 :=
by
  sorry

end sum_of_odd_coefficients_l1413_141369


namespace ordered_pair_count_l1413_141354

theorem ordered_pair_count :
  (∃ (bc : ℕ × ℕ), bc.1 > 0 ∧ bc.2 > 0 ∧ bc.1 ^ 4 - 4 * bc.2 ≤ 0 ∧ bc.2 ^ 4 - 4 * bc.1 ≤ 0) ∧
  ∀ (bc1 bc2 : ℕ × ℕ),
    bc1 ≠ bc2 →
    bc1.1 > 0 ∧ bc1.2 > 0 ∧ bc1.1 ^ 4 - 4 * bc1.2 ≤ 0 ∧ bc1.2 ^ 4 - 4 * bc1.1 ≤ 0 →
    bc2.1 > 0 ∧ bc2.2 > 0 ∧ bc2.1 ^ 4 - 4 * bc2.2 ≤ 0 ∧ bc2.2 ^ 4 - 4 * bc2.1 ≤ 0 →
    false
:=
sorry

end ordered_pair_count_l1413_141354


namespace more_pairs_B_than_A_l1413_141395

theorem more_pairs_B_than_A :
    let pairs_per_box := 20
    let boxes_A := 8
    let pairs_A := boxes_A * pairs_per_box
    let pairs_B := 5 * pairs_A
    let more_pairs := pairs_B - pairs_A
    more_pairs = 640
:= by
    sorry

end more_pairs_B_than_A_l1413_141395


namespace outdoor_section_length_l1413_141356

theorem outdoor_section_length (W : ℝ) (A : ℝ) (hW : W = 4) (hA : A = 24) : ∃ L : ℝ, A = W * L ∧ L = 6 := 
by
  use 6
  sorry

end outdoor_section_length_l1413_141356


namespace find_c_l1413_141340

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 8)) : c = 17 / 3 := 
by
  -- Add the necessary assumptions and let Lean verify these assumptions.
  have b_eq : 3 * b = 8 := sorry
  have b_val : b = 8 / 3 := sorry
  have h_coeff : c = b + 3 := sorry
  exact h_coeff.trans (by rw [b_val]; norm_num)

end find_c_l1413_141340


namespace number_of_multiples_in_range_l1413_141373

-- Definitions based on given conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def in_range (x lower upper : ℕ) : Prop := lower ≤ x ∧ x ≤ upper

def lcm_18_24_30 := ((2^3) * (3^2) * 5) -- LCM of 18, 24, and 30

-- Main theorem statement
theorem number_of_multiples_in_range : 
  (∃ a b c : ℕ, in_range a 2000 3000 ∧ is_multiple_of a lcm_18_24_30 ∧ 
                in_range b 2000 3000 ∧ is_multiple_of b lcm_18_24_30 ∧ 
                in_range c 2000 3000 ∧ is_multiple_of c lcm_18_24_30 ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                ∀ z, in_range z 2000 3000 ∧ is_multiple_of z lcm_18_24_30 → z = a ∨ z = b ∨ z = c) := sorry

end number_of_multiples_in_range_l1413_141373


namespace moles_of_HCl_combined_eq_one_l1413_141343

-- Defining the chemical species involved in the reaction
def NaHCO3 : Type := Nat
def HCl : Type := Nat
def NaCl : Type := Nat
def H2O : Type := Nat
def CO2 : Type := Nat

-- Defining the balanced chemical equation as a condition
def reaction (n_NaHCO3 n_HCl n_NaCl n_H2O n_CO2 : Nat) : Prop :=
  n_NaHCO3 + n_HCl = n_NaCl + n_H2O + n_CO2

-- Given conditions
def one_mole_of_NaHCO3 : Nat := 1
def one_mole_of_NaCl_produced : Nat := 1

-- Proof problem
theorem moles_of_HCl_combined_eq_one :
  ∃ (n_HCl : Nat), reaction one_mole_of_NaHCO3 n_HCl one_mole_of_NaCl_produced 1 1 ∧ n_HCl = 1 := 
by
  sorry

end moles_of_HCl_combined_eq_one_l1413_141343
