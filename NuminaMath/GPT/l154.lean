import Mathlib

namespace NUMINAMATH_GPT_terminal_side_in_quadrant_l154_15483

theorem terminal_side_in_quadrant (α : ℝ) (h : α = -5) : 
  ∃ (q : ℕ), q = 4 ∧ 270 ≤ (α + 360) % 360 ∧ (α + 360) % 360 < 360 := by 
  sorry

end NUMINAMATH_GPT_terminal_side_in_quadrant_l154_15483


namespace NUMINAMATH_GPT_real_complex_number_l154_15498

theorem real_complex_number (x : ℝ) (hx1 : x^2 - 3 * x - 3 > 0) (hx2 : x - 3 = 1) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_real_complex_number_l154_15498


namespace NUMINAMATH_GPT_probability_yellow_chalk_is_three_fifths_l154_15452

open Nat

theorem probability_yellow_chalk_is_three_fifths
  (yellow_chalks : ℕ) (red_chalks : ℕ) (total_chalks : ℕ)
  (h_yellow : yellow_chalks = 3) (h_red : red_chalks = 2) (h_total : total_chalks = yellow_chalks + red_chalks) :
  (yellow_chalks : ℚ) / (total_chalks : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_yellow_chalk_is_three_fifths_l154_15452


namespace NUMINAMATH_GPT_factorize_expression_l154_15441

theorem factorize_expression (x : ℝ) :
  9 * x^2 - 6 * x + 1 = (3 * x - 1)^2 := 
by sorry

end NUMINAMATH_GPT_factorize_expression_l154_15441


namespace NUMINAMATH_GPT_Jake_watched_hours_on_Friday_l154_15408

theorem Jake_watched_hours_on_Friday :
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  total_show_hours - total_hours_before_Friday = 19 :=
by
  let Monday_hours := 12
  let Tuesday_hours := 4
  let Wednesday_hours := 6
  let Thursday_hours := (Monday_hours + Tuesday_hours + Wednesday_hours) / 2
  let total_hours_before_Friday := Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours
  let total_show_hours := 52
  sorry

end NUMINAMATH_GPT_Jake_watched_hours_on_Friday_l154_15408


namespace NUMINAMATH_GPT_find_distance_between_B_and_C_l154_15453

def problem_statement : Prop :=
  ∃ (x y : ℝ),
  (y / 75 + x / 145 = 4.8) ∧ 
  ((x + y) / 100 = 2 + y / 70) ∧ 
  x = 290

theorem find_distance_between_B_and_C : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_find_distance_between_B_and_C_l154_15453


namespace NUMINAMATH_GPT_weierstrass_limit_l154_15477

theorem weierstrass_limit (a_n : ℕ → ℝ) (M : ℝ) :
  (∀ n m, n ≤ m → a_n n ≤ a_n m) → 
  (∀ n, a_n n ≤ M ) → 
  ∃ c, ∀ ε > 0, ∃ N, ∀ n ≥ N, |a_n n - c| < ε :=
by
  sorry

end NUMINAMATH_GPT_weierstrass_limit_l154_15477


namespace NUMINAMATH_GPT_problem_statement_l154_15472

variable (F : ℕ → Prop)

theorem problem_statement (h1 : ∀ k : ℕ, F k → F (k + 1)) (h2 : ¬F 7) : ¬F 6 ∧ ¬F 5 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l154_15472


namespace NUMINAMATH_GPT_range_of_a_l154_15461

variable {f : ℝ → ℝ} {a : ℝ}
open Real

-- Conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def f_positive_at_2 (f : ℝ → ℝ) : Prop := f 2 > 1
def f_value_at_2014 (f : ℝ → ℝ) (a : ℝ) : Prop := f 2014 = (a + 3) / (a - 3)

-- Proof Problem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : odd_function f)
  (h2 : periodic_function f 7)
  (h3 : f_positive_at_2 f)
  (h4 : f_value_at_2014 f a) :
  0 < a ∧ a < 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l154_15461


namespace NUMINAMATH_GPT_distribute_balls_into_boxes_l154_15496

theorem distribute_balls_into_boxes : (Nat.choose (5 + 4 - 1) (4 - 1)) = 56 := by
  sorry

end NUMINAMATH_GPT_distribute_balls_into_boxes_l154_15496


namespace NUMINAMATH_GPT_red_shirts_count_l154_15427

theorem red_shirts_count :
  ∀ (total blue_fraction green_fraction : ℕ),
    total = 60 →
    blue_fraction = total / 3 →
    green_fraction = total / 4 →
    (total - (blue_fraction + green_fraction)) = 25 :=
by
  intros total blue_fraction green_fraction h_total h_blue h_green
  rw [h_total, h_blue, h_green]
  norm_num
  sorry

end NUMINAMATH_GPT_red_shirts_count_l154_15427


namespace NUMINAMATH_GPT_general_formula_sum_of_first_10_terms_l154_15432

variable (a : ℕ → ℝ) (d : ℝ) (S_10 : ℝ)
variable (h1 : a 5 = 11) (h2 : a 8 = 5)

theorem general_formula (n : ℕ) : a n = -2 * n + 21 :=
sorry

theorem sum_of_first_10_terms : S_10 = 100 :=
sorry

end NUMINAMATH_GPT_general_formula_sum_of_first_10_terms_l154_15432


namespace NUMINAMATH_GPT_trajectory_equation_of_P_l154_15489

variable {x y : ℝ}
variable (A B P : ℝ × ℝ)

def in_line_through (a b : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  let k := (p.2 - a.2) / (p.1 - a.1)
  (b.2 - a.2) / (b.1 - a.1) = k

theorem trajectory_equation_of_P
  (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : in_line_through A B P)
  (slope_product : (P.2 / (P.1 + 1)) * (P.2 / (P.1 - 1)) = -1) :
  P.1 ^ 2 + P.2 ^ 2 = 1 ∧ P.1 ≠ 1 ∧ P.1 ≠ -1 := 
sorry

end NUMINAMATH_GPT_trajectory_equation_of_P_l154_15489


namespace NUMINAMATH_GPT_average_s_t_l154_15409

theorem average_s_t (s t : ℝ) 
  (h : (1 + 3 + 7 + s + t) / 5 = 12) : 
  (s + t) / 2 = 24.5 :=
by
  sorry

end NUMINAMATH_GPT_average_s_t_l154_15409


namespace NUMINAMATH_GPT_remaining_integers_count_l154_15484

def set_of_integers_from_1_to_100 : Finset ℕ := (Finset.range 100).map ⟨Nat.succ, Nat.succ_injective⟩

def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ := s.filter (λ x => x % n = 0)

def T : Finset ℕ := set_of_integers_from_1_to_100
def M2 : Finset ℕ := multiples_of 2 T
def M3 : Finset ℕ := multiples_of 3 T
def M5 : Finset ℕ := multiples_of 5 T

def remaining_set : Finset ℕ := T \ (M2 ∪ M3 ∪ M5)

theorem remaining_integers_count : remaining_set.card = 26 := by
  sorry

end NUMINAMATH_GPT_remaining_integers_count_l154_15484


namespace NUMINAMATH_GPT_all_increased_quadratics_have_integer_roots_l154_15433

def original_quadratic (p q : ℤ) : Prop :=
  ∃ α β : ℤ, α + β = -p ∧ α * β = q

def increased_quadratic (p q n : ℤ) : Prop :=
  ∃ α β : ℤ, α + β = -(p + n) ∧ α * β = (q + n)

theorem all_increased_quadratics_have_integer_roots (p q : ℤ) :
  original_quadratic p q →
  (∀ n, 0 ≤ n ∧ n ≤ 9 → increased_quadratic p q n) :=
sorry

end NUMINAMATH_GPT_all_increased_quadratics_have_integer_roots_l154_15433


namespace NUMINAMATH_GPT_additional_money_earned_l154_15490

-- Define the conditions as variables
def price_duck : ℕ := 10
def price_chicken : ℕ := 8
def num_chickens_sold : ℕ := 5
def num_ducks_sold : ℕ := 2
def half (x : ℕ) : ℕ := x / 2
def double (x : ℕ) : ℕ := 2 * x

-- Define the calculations based on the conditions
def earnings_chickens : ℕ := num_chickens_sold * price_chicken 
def earnings_ducks : ℕ := num_ducks_sold * price_duck 
def total_earnings : ℕ := earnings_chickens + earnings_ducks 
def cost_wheelbarrow : ℕ := half total_earnings
def selling_price_wheelbarrow : ℕ := double cost_wheelbarrow
def additional_earnings : ℕ := selling_price_wheelbarrow - cost_wheelbarrow

-- The theorem to prove the correct additional earnings
theorem additional_money_earned : additional_earnings = 30 := by
  sorry

end NUMINAMATH_GPT_additional_money_earned_l154_15490


namespace NUMINAMATH_GPT_find_f_2012_l154_15474

noncomputable def f : ℤ → ℤ := sorry

axiom even_function : ∀ x : ℤ, f (-x) = f x
axiom f_1 : f 1 = 1
axiom f_2011_ne_1 : f 2011 ≠ 1
axiom max_property : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)

theorem find_f_2012 : f 2012 = 1 := sorry

end NUMINAMATH_GPT_find_f_2012_l154_15474


namespace NUMINAMATH_GPT_max_area_central_angle_l154_15451

theorem max_area_central_angle (r l : ℝ) (S α : ℝ) (h1 : 2 * r + l = 4)
  (h2 : S = (1 / 2) * l * r) : (∀ x y : ℝ, (1 / 2) * x * y ≤ (1 / 4) * ((x + y) / 2) ^ 2) → α = l / r → α = 2 :=
by
  sorry

end NUMINAMATH_GPT_max_area_central_angle_l154_15451


namespace NUMINAMATH_GPT_total_worth_of_travelers_checks_l154_15442

variable (x y : ℕ)

theorem total_worth_of_travelers_checks
  (h1 : x + y = 30)
  (h2 : 50 * (x - 15) + 100 * y = 1050) :
  50 * x + 100 * y = 1800 :=
sorry

end NUMINAMATH_GPT_total_worth_of_travelers_checks_l154_15442


namespace NUMINAMATH_GPT_probability_of_C_l154_15487

def region_prob_A := (1 : ℚ) / 4
def region_prob_B := (1 : ℚ) / 3
def region_prob_D := (1 : ℚ) / 6

theorem probability_of_C :
  (region_prob_A + region_prob_B + region_prob_D + (1 : ℚ) / 4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_C_l154_15487


namespace NUMINAMATH_GPT_max_profit_at_boundary_l154_15499

noncomputable def profit (x : ℝ) : ℝ :=
  -50 * (x - 55) ^ 2 + 11250

def within_bounds (x : ℝ) : Prop :=
  40 ≤ x ∧ x ≤ 52

theorem max_profit_at_boundary :
  within_bounds 52 ∧ 
  (∀ x : ℝ, within_bounds x → profit x ≤ profit 52) :=
by
  sorry

end NUMINAMATH_GPT_max_profit_at_boundary_l154_15499


namespace NUMINAMATH_GPT_range_of_a_l154_15424

noncomputable def M : Set ℝ := {2, 0, -1}
noncomputable def N (a : ℝ) : Set ℝ := {x | abs (x - a) < 1}

theorem range_of_a (a : ℝ) : (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 3) ↔ M ∩ N a = {x} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l154_15424


namespace NUMINAMATH_GPT_find_angle_B_l154_15450

-- Definitions and conditions
variables (α β γ δ : ℝ) -- representing angles ∠A, ∠B, ∠C, and ∠D

-- Given Condition: it's a parallelogram and sum of angles A and C
def quadrilateral_parallelogram (A B C D : ℝ) : Prop :=
  A + C = 200 ∧ A = C ∧ A + B = 180

-- Theorem: Degree of angle B is 80°
theorem find_angle_B (A B C D : ℝ) (h : quadrilateral_parallelogram A B C D) : B = 80 := 
  by sorry

end NUMINAMATH_GPT_find_angle_B_l154_15450


namespace NUMINAMATH_GPT_inequality_350_l154_15400

theorem inequality_350 (a b c d : ℝ) : 
  (a - b) * (b - c) * (c - d) * (d - a) + (a - c)^2 * (b - d)^2 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_350_l154_15400


namespace NUMINAMATH_GPT_smallest_value_of_a_for_polynomial_l154_15402

theorem smallest_value_of_a_for_polynomial (r1 r2 r3 : ℕ) (h_prod : r1 * r2 * r3 = 30030) :
  (r1 + r2 + r3 = 54) ∧ (r1 * r2 * r3 = 30030) → 
  (∀ a, a = r1 + r2 + r3 → a ≥ 54) :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_a_for_polynomial_l154_15402


namespace NUMINAMATH_GPT_sum_abc_l154_15492

variable {a b c : ℝ}
variables (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0)
variables (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (c + a))

theorem sum_abc (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (c + a))
   (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) :
   a + b + c = 1128 / 35 := 
sorry

end NUMINAMATH_GPT_sum_abc_l154_15492


namespace NUMINAMATH_GPT_jake_comic_books_l154_15439

variables (J : ℕ)

def brother_comic_books := J + 15
def total_comic_books := J + brother_comic_books

theorem jake_comic_books : total_comic_books = 87 → J = 36 :=
by
  sorry

end NUMINAMATH_GPT_jake_comic_books_l154_15439


namespace NUMINAMATH_GPT_father_children_age_l154_15468

theorem father_children_age (F C n : Nat) (h1 : F = C) (h2 : F = 75) (h3 : C + 5 * n = 2 * (F + n)) : 
  n = 25 :=
by
  sorry

end NUMINAMATH_GPT_father_children_age_l154_15468


namespace NUMINAMATH_GPT_kristy_baked_cookies_l154_15475

theorem kristy_baked_cookies (C : ℕ) :
  (C - 3) - 8 - 12 - 16 - 6 - 14 = 10 ↔ C = 69 := by
  sorry

end NUMINAMATH_GPT_kristy_baked_cookies_l154_15475


namespace NUMINAMATH_GPT_solve_for_k_l154_15438

theorem solve_for_k (k x : ℝ) (h₁ : 4 * k - 3 * x = 2) (h₂ : x = -1) : 
  k = -1 / 4 := 
by sorry

end NUMINAMATH_GPT_solve_for_k_l154_15438


namespace NUMINAMATH_GPT_cheaper_to_buy_more_cheaper_2_values_l154_15446

def cost_function (n : ℕ) : ℕ :=
  if (1 ≤ n ∧ n ≤ 30) then 15 * n - 20
  else if (31 ≤ n ∧ n ≤ 55) then 14 * n
  else if (56 ≤ n) then 13 * n + 10
  else 0  -- Assuming 0 for n < 1 as it shouldn't happen in this context

theorem cheaper_to_buy_more_cheaper_2_values : 
  ∃ n1 n2 : ℕ, n1 < n2 ∧ cost_function (n1 + 1) < cost_function n1 ∧ cost_function (n2 + 1) < cost_function n2 ∧
  ∀ n : ℕ, (cost_function (n + 1) < cost_function n → n = n1 ∨ n = n2) := 
sorry

end NUMINAMATH_GPT_cheaper_to_buy_more_cheaper_2_values_l154_15446


namespace NUMINAMATH_GPT_smallest_sum_of_xy_l154_15488

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_xy_l154_15488


namespace NUMINAMATH_GPT_price_of_coffee_increased_by_300_percent_l154_15418

theorem price_of_coffee_increased_by_300_percent
  (P : ℝ) -- cost per pound of milk powder and coffee in June
  (h1 : 0.20 * P = 0.20) -- price of a pound of milk powder in July
  (h2 : 1.5 * 0.20 = 0.30) -- cost of 1.5 lbs of milk powder in July
  (h3 : 6.30 - 0.30 = 6.00) -- cost of 1.5 lbs of coffee in July
  (h4 : 6.00 / 1.5 = 4.00) -- price per pound of coffee in July
  : ((4.00 - 1.00) / 1.00) * 100 = 300 := 
by 
  sorry

end NUMINAMATH_GPT_price_of_coffee_increased_by_300_percent_l154_15418


namespace NUMINAMATH_GPT_find_x_l154_15456

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_x (x : ℝ) : 
  (sqrt x / sqrt 0.81 + sqrt 1.44 / sqrt 0.49 = 3.0751133491652576) → 
  x = 1.5 :=
by { sorry }

end NUMINAMATH_GPT_find_x_l154_15456


namespace NUMINAMATH_GPT_dinner_cost_l154_15459

variable (total_cost : ℝ)
variable (tax_rate : ℝ)
variable (tip_rate : ℝ)
variable (pre_tax_cost : ℝ)
variable (tip : ℝ)
variable (tax : ℝ)
variable (final_cost : ℝ)

axiom h1 : total_cost = 27.50
axiom h2 : tax_rate = 0.10
axiom h3 : tip_rate = 0.15
axiom h4 : tax = tax_rate * pre_tax_cost
axiom h5 : tip = tip_rate * pre_tax_cost
axiom h6 : final_cost = pre_tax_cost + tax + tip

theorem dinner_cost : pre_tax_cost = 22 := by sorry

end NUMINAMATH_GPT_dinner_cost_l154_15459


namespace NUMINAMATH_GPT_squares_difference_l154_15494

theorem squares_difference (a b : ℝ) (h1 : a + b = 5) (h2 : a - b = 3) : a^2 - b^2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_squares_difference_l154_15494


namespace NUMINAMATH_GPT_mul_mod_remainder_l154_15481

theorem mul_mod_remainder (a b m : ℕ)
  (h₁ : a ≡ 8 [MOD 9])
  (h₂ : b ≡ 1 [MOD 9]) :
  (a * b) % 9 = 8 := 
  sorry

def main : IO Unit :=
  IO.println "The theorem statement has been defined."

end NUMINAMATH_GPT_mul_mod_remainder_l154_15481


namespace NUMINAMATH_GPT_isosceles_triangle_exists_l154_15401

-- Definitions for a triangle vertex and side lengths
structure Triangle :=
  (A B C : ℝ × ℝ) -- Vertices A, B, C
  (AB AC BC : ℝ)  -- Sides AB, AC, BC

-- Definition for all sides being less than 1 unit
def sides_less_than_one (T : Triangle) : Prop :=
  T.AB < 1 ∧ T.AC < 1 ∧ T.BC < 1

-- Definition for isosceles triangle containing the original one
def exists_isosceles_containing (T : Triangle) : Prop :=
  ∃ (T' : Triangle), 
    (T'.AB = T'.AC ∨ T'.AB = T'.BC ∨ T'.AC = T'.BC) ∧
    T'.A = T.A ∧ -- T'.A vertex is same as T.A
    (T'.AB < 1 ∧ T'.AC < 1 ∧ T'.BC < 1) ∧
    (∃ (B1 : ℝ × ℝ), -- There exists point B1 such that new triangle T' incorporates B1
      T'.B = B1 ∧
      T'.C = T.C) -- T' also has vertex C of original triangle

-- Complete theorem statement
theorem isosceles_triangle_exists (T : Triangle) (hT : sides_less_than_one T) : exists_isosceles_containing T :=
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_exists_l154_15401


namespace NUMINAMATH_GPT_students_who_saw_l154_15495

variable (B G : ℕ)

theorem students_who_saw (h : B + G = 33) : (2 * G / 3) + (2 * B / 3) = 22 :=
by
  sorry

end NUMINAMATH_GPT_students_who_saw_l154_15495


namespace NUMINAMATH_GPT_statue_original_cost_l154_15437

noncomputable def original_cost (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

theorem statue_original_cost :
  original_cost 660 0.20 = 550 := 
by
  sorry

end NUMINAMATH_GPT_statue_original_cost_l154_15437


namespace NUMINAMATH_GPT_value_added_to_number_l154_15426

theorem value_added_to_number (x : ℤ) : 
  (150 - 109 = 109 + x) → (x = -68) :=
by
  sorry

end NUMINAMATH_GPT_value_added_to_number_l154_15426


namespace NUMINAMATH_GPT_find_constant_e_l154_15443

theorem find_constant_e {x y e : ℝ} : (x / (2 * y) = 3 / e) → ((7 * x + 4 * y) / (x - 2 * y) = 25) → (e = 2) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_constant_e_l154_15443


namespace NUMINAMATH_GPT_area_of_triangle_ACD_l154_15448

theorem area_of_triangle_ACD (p : ℝ) (y1 y2 x1 x2 : ℝ)
  (h1 : y1^2 = 2 * p * x1)
  (h2 : y2^2 = 2 * p * x2)
  (h3 : y1 + y2 = 4 * p)
  (h4 : y2 - y1 = p)
  (h5 : 2 * y1 + 2 * y2 = 8 * p^2 / (x2 - x1))
  (h6 : x2 - x1 = 2 * p)
  (h7 : 8 * p^2 = (y1 + y2) * 2 * p) :
  1 / 2 * (y1 * (x1 - (x2 + x1) / 2) + y2 * (x2 - (x2 + x1) / 2)) = 15 / 2 * p^2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ACD_l154_15448


namespace NUMINAMATH_GPT_process_cannot_continue_indefinitely_l154_15440

theorem process_cannot_continue_indefinitely (n : ℕ) (hn : 2018 ∣ n) :
  ¬(∀ m, ∃ k, (10*m + k) % 11 = 0 ∧ (10*m + k) / 11 ∣ n) :=
sorry

end NUMINAMATH_GPT_process_cannot_continue_indefinitely_l154_15440


namespace NUMINAMATH_GPT_luke_bike_vs_bus_slowness_l154_15434

theorem luke_bike_vs_bus_slowness
  (luke_bus_time : ℕ)
  (paula_ratio : ℚ)
  (total_travel_time : ℕ)
  (paula_total_bus_time : ℕ)
  (luke_total_travel_time_lhs : ℕ)
  (luke_total_travel_time_rhs : ℕ)
  (bike_time : ℕ)
  (ratio : ℚ) :
  luke_bus_time = 70 ∧
  paula_ratio = 3 / 5 ∧
  total_travel_time = 504 ∧
  paula_total_bus_time = 2 * (paula_ratio * luke_bus_time) ∧
  luke_total_travel_time_lhs = luke_bus_time + bike_time ∧
  luke_total_travel_time_rhs + paula_total_bus_time = total_travel_time ∧
  bike_time = ratio * luke_bus_time ∧
  ratio = bike_time / luke_bus_time →
  ratio = 5 :=
sorry

end NUMINAMATH_GPT_luke_bike_vs_bus_slowness_l154_15434


namespace NUMINAMATH_GPT_count_even_factors_is_correct_l154_15466

def prime_factors_444_533_72 := (2^8 * 5^3 * 7^2)

def range_a := {a : ℕ | 0 ≤ a ∧ a ≤ 8}
def range_b := {b : ℕ | 0 ≤ b ∧ b ≤ 3}
def range_c := {c : ℕ | 0 ≤ c ∧ c ≤ 2}

def even_factors_count : ℕ :=
  (8 - 1 + 1) * (3 - 0 + 1) * (2 - 0 + 1)

theorem count_even_factors_is_correct :
  even_factors_count = 96 := by
  sorry

end NUMINAMATH_GPT_count_even_factors_is_correct_l154_15466


namespace NUMINAMATH_GPT_parabola_focus_l154_15415

theorem parabola_focus :
  ∀ (x y : ℝ), x^2 = 4 * y → (0, 1) = (0, (2 / 2)) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_parabola_focus_l154_15415


namespace NUMINAMATH_GPT_squirrel_acorns_l154_15497

theorem squirrel_acorns (S A : ℤ) 
  (h1 : A = 4 * S + 3) 
  (h2 : A = 5 * S - 6) : 
  A = 39 :=
by sorry

end NUMINAMATH_GPT_squirrel_acorns_l154_15497


namespace NUMINAMATH_GPT_triangle_inequality_necessary_conditions_triangle_inequality_sufficient_conditions_l154_15449

/-- Points \(P, Q, R, S\) are distinct, collinear, and ordered on a line with line segment lengths \( a, b, c \)
    such that \(a = PQ\), \(b = PR\), \(c = PS\). After rotating \(PQ\) and \(RS\) to make \( P \) and \( S \) coincide
    and form a triangle with a positive area, we must show:
    \(I. a < \frac{c}{3}\) must be satisfied in accordance to the triangle inequality revelations -/
theorem triangle_inequality_necessary_conditions (a b c : ℝ)
  (h_abc1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : b > c - b ∧ c > a ∧ c > b - a) :
  a < c / 3 :=
sorry

theorem triangle_inequality_sufficient_conditions (a b c : ℝ)
  (h_abc2 : b ≥ c / 3 ∧ a < c ∧ 2 * b ≤ c) :
  ¬ b < c / 3 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_necessary_conditions_triangle_inequality_sufficient_conditions_l154_15449


namespace NUMINAMATH_GPT_weeks_project_lasts_l154_15406

-- Definition of the conditions
def meal_cost : ℤ := 4
def people : ℤ := 4
def days_per_week : ℤ := 5
def total_spent : ℤ := 1280
def weekly_cost : ℤ := meal_cost * people * days_per_week

-- Problem statement: prove that the number of weeks the project will last equals 16 weeks.
theorem weeks_project_lasts : total_spent / weekly_cost = 16 := by 
  sorry

end NUMINAMATH_GPT_weeks_project_lasts_l154_15406


namespace NUMINAMATH_GPT_baguettes_leftover_l154_15412

-- Definitions based on conditions
def batches_per_day := 3
def baguettes_per_batch := 48
def sold_after_first_batch := 37
def sold_after_second_batch := 52
def sold_after_third_batch := 49

-- Prove the question equals the answer
theorem baguettes_leftover : 
  (batches_per_day * baguettes_per_batch - (sold_after_first_batch + sold_after_second_batch + sold_after_third_batch)) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_baguettes_leftover_l154_15412


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l154_15445

theorem system1_solution (x y : ℚ) :
  x + y = 4 ∧ 5 * (x - y) - 2 * (x + y) = -1 →
  x = 27 / 10 ∧ y = 13 / 10 := by
  sorry

theorem system2_solution (x y : ℚ) :
  (2 * (x - y) / 3) - ((x + y) / 4) = -1 / 12 ∧ 3 * (x + y) - 2 * (2 * x - y) = 3 →
  x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l154_15445


namespace NUMINAMATH_GPT_total_number_of_toy_cars_l154_15467

-- Definitions based on conditions
def numCarsBox1 : ℕ := 21
def numCarsBox2 : ℕ := 31
def numCarsBox3 : ℕ := 19

-- The proof statement
theorem total_number_of_toy_cars : numCarsBox1 + numCarsBox2 + numCarsBox3 = 71 := by
  sorry

end NUMINAMATH_GPT_total_number_of_toy_cars_l154_15467


namespace NUMINAMATH_GPT_age_of_child_l154_15425

theorem age_of_child 
  (avg_age_3_years_ago : ℕ)
  (family_size_3_years_ago : ℕ)
  (current_family_size : ℕ)
  (current_avg_age : ℕ)
  (h1 : avg_age_3_years_ago = 17)
  (h2 : family_size_3_years_ago = 5)
  (h3 : current_family_size = 6)
  (h4 : current_avg_age = 17)
  : ∃ age_of_baby : ℕ, age_of_baby = 2 := 
by
  sorry

end NUMINAMATH_GPT_age_of_child_l154_15425


namespace NUMINAMATH_GPT_degree_reduction_l154_15423

theorem degree_reduction (x : ℝ) (h1 : x^2 = x + 1) (h2 : 0 < x) : x^4 - 2 * x^3 + 3 * x = 1 + Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_degree_reduction_l154_15423


namespace NUMINAMATH_GPT_complex_number_in_first_quadrant_l154_15421

open Complex

theorem complex_number_in_first_quadrant 
    (z : ℂ) 
    (h : z = (3 + I) / (1 - 3 * I) + 2) : 
    0 < z.re ∧ 0 < z.im :=
by
  sorry

end NUMINAMATH_GPT_complex_number_in_first_quadrant_l154_15421


namespace NUMINAMATH_GPT_race_distance_l154_15476

def race_distance_problem (V_A V_B T : ℝ) : Prop :=
  V_A * T = 218.75 ∧
  V_B * T = 193.75 ∧
  V_B * (T + 10) = 218.75 ∧
  T = 77.5

theorem race_distance (D : ℝ) (V_A V_B T : ℝ) 
  (h1 : V_A * T = D) 
  (h2 : V_B * T = D - 25) 
  (h3 : V_B * (T + 10) = D) 
  (h4 : V_A * T = 218.75) 
  (h5 : T = 77.5) 
  : D = 218.75 := 
by 
  sorry

end NUMINAMATH_GPT_race_distance_l154_15476


namespace NUMINAMATH_GPT_least_number_to_divisible_sum_l154_15464

-- Define the conditions and variables
def initial_number : ℕ := 1100
def divisor : ℕ := 23
def least_number_to_add : ℕ := 4

-- Statement to prove
theorem least_number_to_divisible_sum :
  ∃ least_n, least_n + initial_number % divisor = divisor ∧ least_n = least_number_to_add :=
  by
    sorry

end NUMINAMATH_GPT_least_number_to_divisible_sum_l154_15464


namespace NUMINAMATH_GPT_sum_powers_of_i_l154_15479

variable (n : ℕ) (i : ℂ) (h_multiple_of_6 : n % 6 = 0) (h_i : i^2 = -1)

theorem sum_powers_of_i (h_n6 : n = 6) :
    1 + 2*i + 3*i^2 + 4*i^3 + 5*i^4 + 6*i^5 + 7*i^6 = 6*i - 7 := by
  sorry

end NUMINAMATH_GPT_sum_powers_of_i_l154_15479


namespace NUMINAMATH_GPT_ratio_first_part_l154_15405

theorem ratio_first_part (x : ℕ) (h1 : x / 3 = 2) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_first_part_l154_15405


namespace NUMINAMATH_GPT_find_f_2006_l154_15416

variable (f g : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x
def g_def (f : ℝ → ℝ) (g : ℝ → ℝ) := ∀ x : ℝ, g x = f (x - 1)
def f_at_2 (f : ℝ → ℝ) := f 2 = 2

-- The theorem to prove
theorem find_f_2006 (f g : ℝ → ℝ) 
  (even_f : is_even f) 
  (odd_g : is_odd g) 
  (g_eq_f_shift : g_def f g) 
  (f_eq_2 : f_at_2 f) : 
  f 2006 = 2 := 
sorry

end NUMINAMATH_GPT_find_f_2006_l154_15416


namespace NUMINAMATH_GPT_number_of_non_officers_l154_15414

theorem number_of_non_officers 
  (avg_salary_employees: ℝ) (avg_salary_officers: ℝ) (avg_salary_nonofficers: ℝ) 
  (num_officers: ℕ) (num_nonofficers: ℕ):
  avg_salary_employees = 120 ∧ avg_salary_officers = 440 ∧ avg_salary_nonofficers = 110 ∧
  num_officers = 15 ∧ 
  (15 * 440 + num_nonofficers * 110 = (15 + num_nonofficers) * 120)  → 
  num_nonofficers = 480 := 
by 
sorry

end NUMINAMATH_GPT_number_of_non_officers_l154_15414


namespace NUMINAMATH_GPT_range_a_l154_15469

noncomputable def f (x : ℝ) : ℝ := -(1 / 3) * x^3 + x

theorem range_a (a : ℝ) (h1 : a < 1) (h2 : 1 < 10 - a^2) (h3 : f a ≤ f 1) :
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_a_l154_15469


namespace NUMINAMATH_GPT_gary_initial_money_l154_15480

/-- The initial amount of money Gary had, given that he spent $55 and has $18 left. -/
theorem gary_initial_money (amount_spent : ℤ) (amount_left : ℤ) (initial_amount : ℤ) 
  (h1 : amount_spent = 55) 
  (h2 : amount_left = 18) 
  : initial_amount = amount_spent + amount_left :=
by
  sorry

end NUMINAMATH_GPT_gary_initial_money_l154_15480


namespace NUMINAMATH_GPT_average_pastries_per_day_l154_15465

def monday_sales : ℕ := 2
def increment_weekday : ℕ := 2
def increment_weekend : ℕ := 3

def tuesday_sales : ℕ := monday_sales + increment_weekday
def wednesday_sales : ℕ := tuesday_sales + increment_weekday
def thursday_sales : ℕ := wednesday_sales + increment_weekday
def friday_sales : ℕ := thursday_sales + increment_weekday
def saturday_sales : ℕ := friday_sales + increment_weekend
def sunday_sales : ℕ := saturday_sales + increment_weekend

def total_sales_week : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales
def average_sales_per_day : ℚ := total_sales_week / 7

theorem average_pastries_per_day : average_sales_per_day = 59 / 7 := by
  sorry

end NUMINAMATH_GPT_average_pastries_per_day_l154_15465


namespace NUMINAMATH_GPT_correct_option_l154_15428

-- Conditions
def option_A (a : ℝ) : Prop := a^3 + a^3 = a^6
def option_B (a : ℝ) : Prop := (a^3)^2 = a^9
def option_C (a : ℝ) : Prop := a^6 / a^3 = a^2
def option_D (a b : ℝ) : Prop := (a * b)^2 = a^2 * b^2

-- Proof Problem Statement
theorem correct_option (a b : ℝ) : option_D a b ↔ ¬option_A a ∧ ¬option_B a ∧ ¬option_C a :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l154_15428


namespace NUMINAMATH_GPT_distinct_natural_numbers_l154_15429

theorem distinct_natural_numbers (n : ℕ) (h : n = 100) : 
  ∃ (nums : Fin n → ℕ), 
    (∀ i j, i ≠ j → nums i ≠ nums j) ∧
    (∀ (a b c d e : Fin n), 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
     c ≠ d ∧ c ≠ e ∧ 
     d ≠ e →
      (nums a) * (nums b) * (nums c) * (nums d) * (nums e) % ((nums a) + (nums b) + (nums c) + (nums d) + (nums e)) = 0) :=
by
  sorry

end NUMINAMATH_GPT_distinct_natural_numbers_l154_15429


namespace NUMINAMATH_GPT_find_number_l154_15460

theorem find_number (x : ℝ) (h : 0.62 * x - 50 = 43) : x = 150 :=
sorry

end NUMINAMATH_GPT_find_number_l154_15460


namespace NUMINAMATH_GPT_solve_ordered_pair_l154_15404

theorem solve_ordered_pair (x y : ℝ) (h1 : x + y = (5 - x) + (5 - y)) (h2 : x - y = (x - 1) + (y - 1)) : (x, y) = (4, 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_ordered_pair_l154_15404


namespace NUMINAMATH_GPT_evaluate_polynomial_l154_15430

theorem evaluate_polynomial (x : ℝ) : x * (x * (x * (x - 3) - 5) + 9) + 2 = x^4 - 3 * x^3 - 5 * x^2 + 9 * x + 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_l154_15430


namespace NUMINAMATH_GPT_smallest_n_divisible_by_23_l154_15493

theorem smallest_n_divisible_by_23 :
  ∃ n : ℕ, (n^3 + 12 * n^2 + 15 * n + 180) % 23 = 0 ∧
            ∀ m : ℕ, (m^3 + 12 * m^2 + 15 * m + 180) % 23 = 0 → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_n_divisible_by_23_l154_15493


namespace NUMINAMATH_GPT_probability_at_least_one_defective_probability_at_most_one_defective_l154_15431

noncomputable def machine_a_defect_rate : ℝ := 0.05
noncomputable def machine_b_defect_rate : ℝ := 0.1

/-- 
Prove the probability that there is at least one defective part among the two parts
given the defect rates of machine A and machine B
--/
theorem probability_at_least_one_defective (pA pB : ℝ) (hA : pA = machine_a_defect_rate) (hB : pB = machine_b_defect_rate) : 
  (1 - (1 - pA) * (1 - pB)) = 0.145 :=
  sorry

/-- 
Prove the probability that there is at most one defective part among the two parts
given the defect rates of machine A and machine B
--/
theorem probability_at_most_one_defective (pA pB : ℝ) (hA : pA = machine_a_defect_rate) (hB : pB = machine_b_defect_rate) : 
  (1 - pA * pB) = 0.995 :=
  sorry

end NUMINAMATH_GPT_probability_at_least_one_defective_probability_at_most_one_defective_l154_15431


namespace NUMINAMATH_GPT_reflection_xy_plane_reflection_across_point_l154_15471

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def reflect_across_xy_plane (p : Point3D) : Point3D :=
  {x := p.x, y := p.y, z := -p.z}

def reflect_across_point (a p : Point3D) : Point3D :=
  {x := 2 * a.x - p.x, y := 2 * a.y - p.y, z := 2 * a.z - p.z}

theorem reflection_xy_plane :
  reflect_across_xy_plane {x := -2, y := 1, z := 4} = {x := -2, y := 1, z := -4} :=
by sorry

theorem reflection_across_point :
  reflect_across_point {x := 1, y := 0, z := 2} {x := -2, y := 1, z := 4} = {x := -5, y := -1, z := 0} :=
by sorry

end NUMINAMATH_GPT_reflection_xy_plane_reflection_across_point_l154_15471


namespace NUMINAMATH_GPT_find_a_l154_15435

-- Define the sets A and B and the condition that A union B is a subset of A intersect B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

theorem find_a (a : ℝ) :
  A ∪ B a ⊆ A ∩ B a → a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l154_15435


namespace NUMINAMATH_GPT_units_digit_of_n_l154_15491

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 11^4) (h2 : m % 10 = 9) : n % 10 = 9 := 
sorry

end NUMINAMATH_GPT_units_digit_of_n_l154_15491


namespace NUMINAMATH_GPT_count_positive_multiples_of_7_ending_in_5_below_1500_l154_15455

theorem count_positive_multiples_of_7_ending_in_5_below_1500 : 
  ∃ n : ℕ, n = 21 ∧ (∀ k : ℕ, (k < 1500) → ((k % 7 = 0) ∧ (k % 10 = 5) → (∃ m : ℕ, k = 35 + 70 * m) ∧ (0 ≤ m) ∧ (m < 21))) :=
sorry

end NUMINAMATH_GPT_count_positive_multiples_of_7_ending_in_5_below_1500_l154_15455


namespace NUMINAMATH_GPT_multiply_3_6_and_0_3_l154_15407

theorem multiply_3_6_and_0_3 : 3.6 * 0.3 = 1.08 :=
by
  sorry

end NUMINAMATH_GPT_multiply_3_6_and_0_3_l154_15407


namespace NUMINAMATH_GPT_max_square_side_length_l154_15458

-- Given: distances between consecutive lines in L and P
def distances_L : List ℕ := [2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6, 2]
def distances_P : List ℕ := [3, 1, 2, 6, 3, 1, 2, 6, 3, 1, 2, 6, 3, 1]

-- Theorem: Maximum possible side length of a square with sides on lines L and P
theorem max_square_side_length : ∀ (L P : List ℕ), L = distances_L → P = distances_P → ∃ s, s = 40 :=
by
  intros L P hL hP
  sorry

end NUMINAMATH_GPT_max_square_side_length_l154_15458


namespace NUMINAMATH_GPT_students_in_second_class_l154_15457

variable (x : ℕ)

theorem students_in_second_class :
  (∃ x, 30 * 40 + 70 * x = (30 + x) * 58.75) → x = 50 :=
by
  sorry

end NUMINAMATH_GPT_students_in_second_class_l154_15457


namespace NUMINAMATH_GPT_alcohol_percentage_in_new_solution_l154_15413

theorem alcohol_percentage_in_new_solution :
  let original_volume := 40 -- liters
  let original_percentage_alcohol := 0.05
  let added_alcohol := 5.5 -- liters
  let added_water := 4.5 -- liters
  let original_alcohol := original_percentage_alcohol * original_volume
  let new_alcohol := original_alcohol + added_alcohol
  let new_volume := original_volume + added_alcohol + added_water
  (new_alcohol / new_volume) * 100 = 15 := by
  sorry

end NUMINAMATH_GPT_alcohol_percentage_in_new_solution_l154_15413


namespace NUMINAMATH_GPT_prove_ratio_l154_15411

variable (a b c d : ℚ)

-- Conditions
def cond1 : a / b = 5 := sorry
def cond2 : b / c = 1 / 4 := sorry
def cond3 : c / d = 7 := sorry

-- Theorem to prove the final result
theorem prove_ratio (h1 : a / b = 5) (h2 : b / c = 1 / 4) (h3 : c / d = 7) :
  d / a = 4 / 35 :=
by
  sorry

end NUMINAMATH_GPT_prove_ratio_l154_15411


namespace NUMINAMATH_GPT_find_Q_plus_R_l154_15486

-- P, Q, R must be digits in base 8 (distinct and non-zero)
def is_valid_digit (d : Nat) : Prop :=
  d > 0 ∧ d < 8

def digits_distinct (P Q R : Nat) : Prop :=
  P ≠ Q ∧ Q ≠ R ∧ R ≠ P

-- Define the base 8 number from its digits
def base8_number (P Q R : Nat) : Nat :=
  8^2 * P + 8 * Q + R

-- Define the given condition
def condition (P Q R : Nat) : Prop :=
  is_valid_digit P ∧ is_valid_digit Q ∧ is_valid_digit R ∧ digits_distinct P Q R ∧ 
  (base8_number P Q R + base8_number Q R P + base8_number R P Q = 8^3 * P + 8^2 * P + 8 * P + 8)

-- The result: Q + R in base 8 is 10_8 which is 8 + 2 (in decimal is 10)
theorem find_Q_plus_R (P Q R : Nat) (h : condition P Q R) : Q + R = 8 + 2 :=
sorry

end NUMINAMATH_GPT_find_Q_plus_R_l154_15486


namespace NUMINAMATH_GPT_larger_volume_of_rotated_rectangle_l154_15478

-- Definitions based on the conditions
def length : ℝ := 4
def width : ℝ := 3

-- Problem statement: Proving the volume of the larger geometric solid
theorem larger_volume_of_rotated_rectangle :
  max (Real.pi * (width ^ 2) * length) (Real.pi * (length ^ 2) * width) = 48 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_larger_volume_of_rotated_rectangle_l154_15478


namespace NUMINAMATH_GPT_mass_percentage_K_l154_15444

theorem mass_percentage_K (compound : Type) (m : ℝ) (mass_percentage : ℝ) (h : mass_percentage = 23.81) : mass_percentage = 23.81 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_K_l154_15444


namespace NUMINAMATH_GPT_problem1_l154_15485

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := 
sorry

end NUMINAMATH_GPT_problem1_l154_15485


namespace NUMINAMATH_GPT_number_chosen_div_8_sub_100_eq_6_l154_15417

variable (n : ℤ)

theorem number_chosen_div_8_sub_100_eq_6 (h : (n / 8) - 100 = 6) : n = 848 := 
by
  sorry

end NUMINAMATH_GPT_number_chosen_div_8_sub_100_eq_6_l154_15417


namespace NUMINAMATH_GPT_sequence_a_n_l154_15422

theorem sequence_a_n (a : ℕ → ℚ)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 1) * (a n + 1) = a n) :
  a 6 = 1 / 6 :=
  sorry

end NUMINAMATH_GPT_sequence_a_n_l154_15422


namespace NUMINAMATH_GPT_machine_does_not_require_repair_l154_15447

-- Define the conditions.

def max_deviation := 37

def nominal_portion_max_deviation_percentage := 0.10

def deviation_within_limit (M : ℝ) : Prop :=
  37 ≤ 0.10 * M

def unreadable_measurements_deviation (deviation : ℝ) : Prop :=
  deviation < 37

-- Define the theorem we want to prove

theorem machine_does_not_require_repair (M : ℝ)
  (h1 : deviation_within_limit M)
  (h2 : ∀ deviation, unreadable_measurements_deviation deviation) :
  true := 
sorry

end NUMINAMATH_GPT_machine_does_not_require_repair_l154_15447


namespace NUMINAMATH_GPT_malcolm_social_media_followers_l154_15420

theorem malcolm_social_media_followers :
  let instagram_initial := 240
  let facebook_initial := 500
  let twitter_initial := (instagram_initial + facebook_initial) / 2
  let tiktok_initial := 3 * twitter_initial
  let youtube_initial := tiktok_initial + 510
  let pinterest_initial := 120
  let snapchat_initial := pinterest_initial / 2

  let instagram_after := instagram_initial + (15 * instagram_initial / 100)
  let facebook_after := facebook_initial + (20 * facebook_initial / 100)
  let twitter_after := twitter_initial - 12
  let tiktok_after := tiktok_initial + (10 * tiktok_initial / 100)
  let youtube_after := youtube_initial + (8 * youtube_initial / 100)
  let pinterest_after := pinterest_initial + 20
  let snapchat_after := snapchat_initial - (5 * snapchat_initial / 100)

  instagram_after + facebook_after + twitter_after + tiktok_after + youtube_after + pinterest_after + snapchat_after = 4402 := sorry

end NUMINAMATH_GPT_malcolm_social_media_followers_l154_15420


namespace NUMINAMATH_GPT_total_distance_fourth_time_l154_15419

/-- 
A super ball is dropped from a height of 100 feet and rebounds half the distance it falls each time.
We need to prove that the total distance the ball travels when it hits the ground
the fourth time is 275 feet.
-/
noncomputable def total_distance : ℝ :=
  let first_descent := 100
  let second_descent := first_descent / 2
  let third_descent := second_descent / 2
  let fourth_descent := third_descent / 2
  let first_ascent := second_descent
  let second_ascent := third_descent
  let third_ascent := fourth_descent
  first_descent + second_descent + third_descent + fourth_descent +
  first_ascent + second_ascent + third_ascent

theorem total_distance_fourth_time : total_distance = 275 := 
  by
  sorry

end NUMINAMATH_GPT_total_distance_fourth_time_l154_15419


namespace NUMINAMATH_GPT_vector_b_value_l154_15403

theorem vector_b_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, -2)
  2 • a + b = (3, 2) → b = (1, -2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_vector_b_value_l154_15403


namespace NUMINAMATH_GPT_proposition_induction_l154_15482

theorem proposition_induction {P : ℕ → Prop} (h : ∀ n, P n → P (n + 1)) (hn : ¬ P 7) : ¬ P 6 :=
by
  sorry

end NUMINAMATH_GPT_proposition_induction_l154_15482


namespace NUMINAMATH_GPT_tan_5105_eq_tan_85_l154_15473

noncomputable def tan_deg (d : ℝ) := Real.tan (d * Real.pi / 180)

theorem tan_5105_eq_tan_85 :
  tan_deg 5105 = tan_deg 85 := by
  have eq_265 : tan_deg 5105 = tan_deg 265 := by sorry
  have eq_neg : tan_deg 265 = tan_deg 85 := by sorry
  exact Eq.trans eq_265 eq_neg

end NUMINAMATH_GPT_tan_5105_eq_tan_85_l154_15473


namespace NUMINAMATH_GPT_ab_value_l154_15470

variable (a b : ℝ)

theorem ab_value (h1 : a^5 * b^8 = 12) (h2 : a^8 * b^13 = 18) : a * b = 128 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_ab_value_l154_15470


namespace NUMINAMATH_GPT_intersection_eq_l154_15462

def setM (x : ℝ) : Prop := x > -1
def setN (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem intersection_eq : {x : ℝ | setM x} ∩ {x | setN x} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l154_15462


namespace NUMINAMATH_GPT_unvisited_planet_exists_l154_15410

theorem unvisited_planet_exists (n : ℕ) (h : 1 ≤ n)
  (planets : Fin (2 * n + 1) → ℝ) 
  (distinct_distances : ∀ i j : Fin (2 * n + 1), i ≠ j → planets i ≠ planets j) 
  (expeditions : Fin (2 * n + 1) → Fin (2 * n + 1))
  (closest : ∀ i : Fin (2 * n + 1), expeditions i = i ↔ False) :
  ∃ p : Fin (2 * n + 1), ∀ q : Fin (2 * n + 1), expeditions q ≠ p := sorry

end NUMINAMATH_GPT_unvisited_planet_exists_l154_15410


namespace NUMINAMATH_GPT_percent_savings_12_roll_package_l154_15454

def percent_savings_per_roll (package_cost : ℕ) (individual_cost : ℕ) (num_rolls : ℕ) : ℚ :=
  let individual_total := num_rolls * individual_cost
  let package_total := package_cost
  let per_roll_package_cost := package_total / num_rolls
  let savings_per_roll := individual_cost - per_roll_package_cost
  (savings_per_roll / individual_cost) * 100

theorem percent_savings_12_roll_package :
  percent_savings_per_roll 9 1 12 = 25 := 
sorry

end NUMINAMATH_GPT_percent_savings_12_roll_package_l154_15454


namespace NUMINAMATH_GPT_radian_measure_15_degrees_l154_15436

theorem radian_measure_15_degrees : (15 * (Real.pi / 180)) = (Real.pi / 12) :=
by
  sorry

end NUMINAMATH_GPT_radian_measure_15_degrees_l154_15436


namespace NUMINAMATH_GPT_remove_one_to_get_average_of_75_l154_15463

theorem remove_one_to_get_average_of_75 : 
  ∃ l : List ℕ, l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] ∧ 
  (∃ m : ℕ, List.erase l m = ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] : List ℕ) ∧ 
  (12 : ℕ) = List.length (List.erase l m) ∧
  7.5 = ((List.sum (List.erase l m) : ℚ) / 12)) :=
sorry

end NUMINAMATH_GPT_remove_one_to_get_average_of_75_l154_15463
