import Mathlib

namespace NUMINAMATH_GPT_root_of_function_l619_61998

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

theorem root_of_function (f : ℝ → ℝ) (x₀ : ℝ) (h₀ : odd_function f) (h₁ : f (x₀) = Real.exp (x₀)) :
  (f (-x₀) * Real.exp (-x₀) + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_root_of_function_l619_61998


namespace NUMINAMATH_GPT_unique_four_letter_sequence_l619_61950

def alphabet_value (c : Char) : ℕ :=
  if 'A' <= c ∧ c <= 'Z' then (c.toNat - 'A'.toNat + 1) else 0

def sequence_product (s : String) : ℕ :=
  s.foldl (λ acc c => acc * alphabet_value c) 1

theorem unique_four_letter_sequence (s : String) :
  sequence_product "WXYZ" = sequence_product s → s = "WXYZ" :=
by
  sorry

end NUMINAMATH_GPT_unique_four_letter_sequence_l619_61950


namespace NUMINAMATH_GPT_calc_value_of_fraction_l619_61968

theorem calc_value_of_fraction :
  (10^9 / (2 * 5^2 * 10^3)) = 20000 := by
  sorry

end NUMINAMATH_GPT_calc_value_of_fraction_l619_61968


namespace NUMINAMATH_GPT_find_a_7_l619_61987

-- Define the arithmetic sequence conditions
variable {a : ℕ → ℤ} -- The sequence a_n
variable (a_4_eq : a 4 = 4)
variable (a_3_a_8_eq : a 3 + a 8 = 5)

-- Prove that a_7 = 1
theorem find_a_7 : a 7 = 1 := by
  sorry

end NUMINAMATH_GPT_find_a_7_l619_61987


namespace NUMINAMATH_GPT_circle_tangent_l619_61934

theorem circle_tangent (t : ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 4 → (x - t)^2 + y^2 = 1 → |t| = 3) :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_l619_61934


namespace NUMINAMATH_GPT_Tom_marble_choices_l619_61963

theorem Tom_marble_choices :
  let total_marbles := 18
  let special_colors := 4
  let choose_one_from_special := (Nat.choose special_colors 1)
  let remaining_marbles := total_marbles - special_colors
  let choose_remaining := (Nat.choose remaining_marbles 5)
  choose_one_from_special * choose_remaining = 8008
:= sorry

end NUMINAMATH_GPT_Tom_marble_choices_l619_61963


namespace NUMINAMATH_GPT_polynomial_solution_l619_61969

open Polynomial

noncomputable def p (x : ℝ) : ℝ := -4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2

theorem polynomial_solution (x : ℝ) :
  4 * x^5 + 3 * x^3 + 2 * x^2 + (-4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2) = 6 * x^3 - 5 * x^2 + 4 * x - 2 :=
by
  -- Verification of the equality
  sorry

end NUMINAMATH_GPT_polynomial_solution_l619_61969


namespace NUMINAMATH_GPT_min_value_expression_l619_61957

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l619_61957


namespace NUMINAMATH_GPT_power_of_two_grows_faster_l619_61918

theorem power_of_two_grows_faster (n : ℕ) (h : n ≥ 5) : 2^n > n^2 :=
sorry

end NUMINAMATH_GPT_power_of_two_grows_faster_l619_61918


namespace NUMINAMATH_GPT_number_exceeds_its_fraction_by_35_l619_61980

theorem number_exceeds_its_fraction_by_35 (x : ℝ) (h : x = (3 / 8) * x + 35) : x = 56 :=
by
  sorry

end NUMINAMATH_GPT_number_exceeds_its_fraction_by_35_l619_61980


namespace NUMINAMATH_GPT_cos_75_deg_l619_61910

theorem cos_75_deg : Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_GPT_cos_75_deg_l619_61910


namespace NUMINAMATH_GPT_range_of_y_l619_61990

theorem range_of_y :
  ∀ (y x : ℝ), x = 4 - y → (-2 ≤ x ∧ x ≤ -1) → (5 ≤ y ∧ y ≤ 6) :=
by
  intros y x h1 h2
  sorry

end NUMINAMATH_GPT_range_of_y_l619_61990


namespace NUMINAMATH_GPT_range_of_f_l619_61942

noncomputable def f (x : ℝ) : ℝ := 2^x
def valid_range (S : Set ℝ) : Prop := ∃ x ∈ Set.Icc (0 : ℝ) (3 : ℝ), f x ∈ S

theorem range_of_f : valid_range (Set.Icc (1 : ℝ) (8 : ℝ)) :=
sorry

end NUMINAMATH_GPT_range_of_f_l619_61942


namespace NUMINAMATH_GPT_fibonacci_series_sum_l619_61902

noncomputable def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n + 1) + fib n

theorem fibonacci_series_sum :
  (∑' n, (fib n : ℝ) / 7^n) = (49 : ℝ) / 287 := 
by
  sorry

end NUMINAMATH_GPT_fibonacci_series_sum_l619_61902


namespace NUMINAMATH_GPT_range_of_a_l619_61952

  variable {A : Set ℝ} {B : Set ℝ}
  variable {a : ℝ}

  def A_def (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ 2 * a - 4 }
  def B_def : Set ℝ := { x | -1 < x ∧ x < 6 }

  theorem range_of_a (h : A_def a ∩ B_def = A_def a) : a < 5 :=
  sorry
  
end NUMINAMATH_GPT_range_of_a_l619_61952


namespace NUMINAMATH_GPT_mark_money_left_l619_61912

theorem mark_money_left (initial_money : ℕ) (cost_book1 cost_book2 cost_book3 : ℕ) (n_book1 n_book2 n_book3 : ℕ) 
  (total_cost : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 85)
  (h2 : cost_book1 = 7)
  (h3 : n_book1 = 3)
  (h4 : cost_book2 = 5)
  (h5 : n_book2 = 4)
  (h6 : cost_book3 = 9)
  (h7 : n_book3 = 2)
  (h8 : total_cost = 21 + 20 + 18)
  (h9 : money_left = initial_money - total_cost):
  money_left = 26 := by
  sorry

end NUMINAMATH_GPT_mark_money_left_l619_61912


namespace NUMINAMATH_GPT_minimum_value_of_x_plus_y_l619_61925

theorem minimum_value_of_x_plus_y
  (x y : ℝ)
  (h1 : x > y)
  (h2 : y > 0)
  (h3 : 1 / (x - y) + 8 / (x + 2 * y) = 1) :
  x + y = 25 / 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_x_plus_y_l619_61925


namespace NUMINAMATH_GPT_find_C_l619_61977

theorem find_C 
  (m n : ℝ)
  (C : ℝ)
  (h1 : m = 6 * n + C)
  (h2 : m + 2 = 6 * (n + 0.3333333333333333) + C) 
  : C = 0 := by
  sorry

end NUMINAMATH_GPT_find_C_l619_61977


namespace NUMINAMATH_GPT_hour_minute_hand_coincide_at_l619_61945

noncomputable def coinciding_time : ℚ :=
  90 / (6 - 0.5)

theorem hour_minute_hand_coincide_at : coinciding_time = 16 + 4 / 11 := 
  sorry

end NUMINAMATH_GPT_hour_minute_hand_coincide_at_l619_61945


namespace NUMINAMATH_GPT_monthly_growth_rate_l619_61903

-- Definitions and conditions
def initial_height : ℝ := 20
def final_height : ℝ := 80
def months_in_year : ℕ := 12

-- Theorem stating the monthly growth rate
theorem monthly_growth_rate :
  (final_height - initial_height) / (months_in_year : ℝ) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_monthly_growth_rate_l619_61903


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l619_61994

theorem neither_sufficient_nor_necessary 
  (a b c : ℝ) : 
  ¬ ((∀ x : ℝ, b^2 - 4 * a * c < 0 → a * x^2 + b * x + c > 0) ∧ 
     (∀ x : ℝ, a * x^2 + b * x + c > 0 → b^2 - 4 * a * c < 0)) := 
by
  sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l619_61994


namespace NUMINAMATH_GPT_question_1_question_2_question_3_l619_61911

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 2 * b

-- Question 1
theorem question_1 (a b : ℝ) (h : a = b) (ha : a > 0) :
  ∀ x : ℝ, (f a b x < 0) ↔ (-2 < x ∧ x < 1) :=
sorry

-- Question 2
theorem question_2 (b : ℝ) :
  (∀ x : ℝ, x < 2 → (f 1 b x ≥ 1)) → (b ≤ 2 * Real.sqrt 3 - 4) :=
sorry

-- Question 3
theorem question_3 (a b : ℝ) (h1 : |f a b (-1)| ≤ 1) (h2 : |f a b 1| ≤ 3) :
  (5 / 3 ≤ |a| + |b + 2| ∧ |a| + |b + 2| ≤ 9) :=
sorry

end NUMINAMATH_GPT_question_1_question_2_question_3_l619_61911


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l619_61914

theorem solve_equation_1 (y: ℝ) : y^2 - 6 * y + 1 = 0 ↔ (y = 3 + 2 * Real.sqrt 2 ∨ y = 3 - 2 * Real.sqrt 2) :=
sorry

theorem solve_equation_2 (x: ℝ) : 2 * (x - 4)^2 = x^2 - 16 ↔ (x = 4 ∨ x = 12) :=
sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l619_61914


namespace NUMINAMATH_GPT_range_of_a_part1_range_of_a_part2_l619_61953

def set_A (x : ℝ) : Prop := -1 < x ∧ x < 6

def set_B (x : ℝ) (a : ℝ) : Prop := (x ≥ 1 + a) ∨ (x ≤ 1 - a)

def condition_1 (a : ℝ) : Prop :=
  (∀ x, set_A x → ¬ set_B x a) → (a ≥ 5)

def condition_2 (a : ℝ) : Prop :=
  (∀ x, (x ≥ 6 ∨ x ≤ -1) → set_B x a) ∧ (∃ x, set_B x a ∧ ¬ (x ≥ 6 ∨ x ≤ -1)) → (0 < a ∧ a ≤ 2)

theorem range_of_a_part1 (a : ℝ) : condition_1 a :=
  sorry

theorem range_of_a_part2 (a : ℝ) : condition_2 a :=
  sorry

end NUMINAMATH_GPT_range_of_a_part1_range_of_a_part2_l619_61953


namespace NUMINAMATH_GPT_freddy_spent_10_dollars_l619_61988

theorem freddy_spent_10_dollars 
  (talk_time_dad : ℕ) (talk_time_brother : ℕ) 
  (local_cost_per_minute : ℕ) (international_cost_per_minute : ℕ)
  (conversion_cents_to_dollar : ℕ)
  (h1 : talk_time_dad = 45)
  (h2 : talk_time_brother = 31)
  (h3 : local_cost_per_minute = 5)
  (h4 : international_cost_per_minute = 25)
  (h5 : conversion_cents_to_dollar = 100):
  (local_cost_per_minute * talk_time_dad + international_cost_per_minute * talk_time_brother) / conversion_cents_to_dollar = 10 :=
by
  sorry

end NUMINAMATH_GPT_freddy_spent_10_dollars_l619_61988


namespace NUMINAMATH_GPT_relationship_between_m_and_n_l619_61904

variable (a b m n : ℝ)

axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : m = Real.sqrt a - Real.sqrt b
axiom h4 : n = Real.sqrt (a - b)

theorem relationship_between_m_and_n : m < n :=
by
  -- Lean requires 'sorry' to be used as a placeholder for the proof
  sorry

end NUMINAMATH_GPT_relationship_between_m_and_n_l619_61904


namespace NUMINAMATH_GPT_solve_diamond_l619_61920

theorem solve_diamond : ∃ (D : ℕ), D < 10 ∧ (D * 9 + 5 = D * 10 + 2) ∧ D = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_diamond_l619_61920


namespace NUMINAMATH_GPT_eval_expr_l619_61981

theorem eval_expr : (2.1 * (49.7 + 0.3)) + 15 = 120 :=
  by
  sorry

end NUMINAMATH_GPT_eval_expr_l619_61981


namespace NUMINAMATH_GPT_least_sum_of_bases_l619_61906

theorem least_sum_of_bases :
  ∃ (c d : ℕ), (5 * c + 7 = 7 * d + 5) ∧ (c > 0) ∧ (d > 0) ∧ (c + d = 14) :=
by
  sorry

end NUMINAMATH_GPT_least_sum_of_bases_l619_61906


namespace NUMINAMATH_GPT_find_y_from_expression_l619_61956

theorem find_y_from_expression :
  ∀ y : ℕ, 2^10 + 2^10 + 2^10 + 2^10 = 4^y → y = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_y_from_expression_l619_61956


namespace NUMINAMATH_GPT_find_y_ratio_l619_61936

variable {R : Type} [LinearOrderedField R]
variables (x y : R → R) (x1 x2 y1 y2 : R)

-- Condition: x is inversely proportional to y, so xy is constant.
def inversely_proportional (x y : R → R) : Prop := ∀ (a b : R), x a * y a = x b * y b

-- Condition: ∀ nonzero x values, we have these specific ratios
variable (h_inv_prop : inversely_proportional x y)
variable (h_ratio_x : x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 / x2 = 4 / 5)
variable (h_nonzero_y : y1 ≠ 0 ∧ y2 ≠ 0)

-- Claim to prove
theorem find_y_ratio : (y1 / y2) = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_y_ratio_l619_61936


namespace NUMINAMATH_GPT_isosceles_in_27_gon_l619_61985

def vertices := {x : ℕ // x < 27}

def is_isosceles_triangle (a b c : vertices) : Prop :=
  (a.val + c.val) / 2 % 27 = b.val

def is_isosceles_trapezoid (a b c d : vertices) : Prop :=
  (a.val + d.val) / 2 % 27 = (b.val + c.val) / 2 % 27

def seven_points_form_isosceles (s : Finset vertices) : Prop :=
  ∃ (a b c : vertices) (h1 : a ∈ s) (h2 : b ∈ s) (h3 : c ∈ s), is_isosceles_triangle a b c

def seven_points_form_isosceles_trapezoid (s : Finset vertices) : Prop :=
  ∃ (a b c d : vertices) (h1 : a ∈ s) (h2 : b ∈ s) (h3 : c ∈ s) (h4 : d ∈ s), is_isosceles_trapezoid a b c d

theorem isosceles_in_27_gon :
  ∀ (s : Finset vertices), s.card = 7 → 
  (seven_points_form_isosceles s) ∨ (seven_points_form_isosceles_trapezoid s) :=
by sorry

end NUMINAMATH_GPT_isosceles_in_27_gon_l619_61985


namespace NUMINAMATH_GPT_difference_in_cents_l619_61930

-- Given definitions and conditions
def number_of_coins : ℕ := 3030
def min_nickels : ℕ := 3
def ratio_pennies_to_nickels : ℕ := 10

-- Problem statement: Prove that the difference in cents between the maximum and minimum monetary amounts is 1088
theorem difference_in_cents (p n : ℕ) (h1 : p + n = number_of_coins)
  (h2 : p ≥ ratio_pennies_to_nickels * n) (h3 : n ≥ min_nickels) :
  4 * 275 = 1100 ∧ (3030 + 1100) - (3030 + 4 * 3) = 1088 :=
by {
  sorry
}

end NUMINAMATH_GPT_difference_in_cents_l619_61930


namespace NUMINAMATH_GPT_initial_ratio_of_partners_to_associates_l619_61941

theorem initial_ratio_of_partners_to_associates
  (P : ℕ) (A : ℕ)
  (hP : P = 18)
  (h_ratio_after_hiring : ∀ A, 45 + A = 18 * 34) :
  (P : ℤ) / (A : ℤ) = 2 / 63 := 
sorry

end NUMINAMATH_GPT_initial_ratio_of_partners_to_associates_l619_61941


namespace NUMINAMATH_GPT_carson_gold_stars_yesterday_l619_61984

def goldStarsEarnedYesterday (total: ℕ) (earnedToday: ℕ) : ℕ :=
  total - earnedToday

theorem carson_gold_stars_yesterday :
  goldStarsEarnedYesterday 15 9 = 6 :=
by 
  sorry

end NUMINAMATH_GPT_carson_gold_stars_yesterday_l619_61984


namespace NUMINAMATH_GPT_taco_variants_count_l619_61955

theorem taco_variants_count :
  let toppings := 8
  let meat_variants := 3
  let shell_variants := 2
  2 ^ toppings * meat_variants * shell_variants = 1536 := by
sorry

end NUMINAMATH_GPT_taco_variants_count_l619_61955


namespace NUMINAMATH_GPT_fraction_addition_l619_61937

variable (d : ℝ)

theorem fraction_addition (d : ℝ) : (5 + 2 * d) / 8 + 3 = (29 + 2 * d) / 8 := 
sorry

end NUMINAMATH_GPT_fraction_addition_l619_61937


namespace NUMINAMATH_GPT_cost_price_percentage_of_marked_price_l619_61962

theorem cost_price_percentage_of_marked_price
  (MP : ℝ) -- Marked Price
  (CP : ℝ) -- Cost Price
  (discount_percent : ℝ) (gain_percent : ℝ)
  (H1 : CP = (x / 100) * MP) -- Cost Price is x percent of Marked Price
  (H2 : discount_percent = 13) -- Discount percentage
  (H3 : gain_percent = 55.35714285714286) -- Gain percentage
  : x = 56 :=
sorry

end NUMINAMATH_GPT_cost_price_percentage_of_marked_price_l619_61962


namespace NUMINAMATH_GPT_problem_x_value_l619_61915

theorem problem_x_value (x : ℝ) (h : (max 3 (max 6 (max 9 x)) * min 3 (min 6 (min 9 x)) = 3 + 6 + 9 + x)) : 
    x = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_x_value_l619_61915


namespace NUMINAMATH_GPT_no_solution_implies_a_eq_one_l619_61958

theorem no_solution_implies_a_eq_one (a : ℝ) : 
  ¬(∃ x y : ℝ, a * x + y = 1 ∧ x + y = 2) → a = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_solution_implies_a_eq_one_l619_61958


namespace NUMINAMATH_GPT_prob_not_less_than_30_l619_61965

-- Define the conditions
def prob_less_than_30 : ℝ := 0.3
def prob_between_30_and_40 : ℝ := 0.5

-- State the theorem
theorem prob_not_less_than_30 (h1 : prob_less_than_30 = 0.3) : 1 - prob_less_than_30 = 0.7 :=
by
  sorry

end NUMINAMATH_GPT_prob_not_less_than_30_l619_61965


namespace NUMINAMATH_GPT_product_of_roots_of_quartic_polynomial_l619_61966

theorem product_of_roots_of_quartic_polynomial :
  (∀ x : ℝ, (3 * x^4 - 8 * x^3 + x^2 - 10 * x - 24 = 0) → x = p ∨ x = q ∨ x = r ∨ x = s) →
  (p * q * r * s = -8) :=
by
  intros
  -- proof goes here
  sorry

end NUMINAMATH_GPT_product_of_roots_of_quartic_polynomial_l619_61966


namespace NUMINAMATH_GPT_binomial_coefficient_10_3_l619_61933

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end NUMINAMATH_GPT_binomial_coefficient_10_3_l619_61933


namespace NUMINAMATH_GPT_valid_colorings_l619_61986

-- Define the coloring function and the condition
variable (f : ℕ → ℕ) -- f assigns a color (0, 1, or 2) to each natural number
variable (a b c : ℕ)
-- Colors are represented by 0, 1, or 2
variable (colors : Fin 3)

-- Define the condition to be checked
def valid_coloring : Prop :=
  ∀ a b c, 2000 * (a + b) = c → (f a = f b ∧ f b = f c) ∨ (f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f a)

-- Now define the two possible valid ways of coloring
def all_same_color : Prop :=
  ∃ color, ∀ n, f n = color

def every_third_different : Prop :=
  (∀ k : ℕ, f (3 * k) = 0 ∧ f (3 * k + 1) = 1 ∧ f (3 * k + 2) = 2)

-- Prove that these are the only two valid ways
theorem valid_colorings :
  valid_coloring f →
  all_same_color f ∨ every_third_different f :=
sorry

end NUMINAMATH_GPT_valid_colorings_l619_61986


namespace NUMINAMATH_GPT_equilateral_triangle_intersection_impossible_l619_61946

noncomputable def trihedral_angle (α β γ : ℝ) : Prop :=
  α + β + γ = 180 ∧ β = 90 ∧ γ = 90 ∧ α > 0

theorem equilateral_triangle_intersection_impossible :
  ¬ ∀ (α : ℝ), ∀ (β γ : ℝ), trihedral_angle α β γ → 
    ∃ (plane : ℝ → ℝ → ℝ), 
      ∀ (x y z : ℝ), plane x y = z → x = y ∧ y = z ∧ z = x ∧ 
                      x + y + z = 60 :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_intersection_impossible_l619_61946


namespace NUMINAMATH_GPT_xiaoli_time_l619_61901

variable {t : ℕ} -- Assuming t is a natural number (time in seconds)

theorem xiaoli_time (record_time : ℕ) (t_non_break : t ≥ record_time) (h : record_time = 14) : t ≥ 14 :=
by
  rw [h] at t_non_break
  exact t_non_break

end NUMINAMATH_GPT_xiaoli_time_l619_61901


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l619_61924

theorem trajectory_of_midpoint 
  (x y : ℝ)
  (P : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : (M.fst - 4)^2 + M.snd^2 = 16)
  (hP : P = (x, y))
  (h_mid : M = (2 * P.1 + 4, 2 * P.2 - 8)) :
  x^2 + (y - 4)^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l619_61924


namespace NUMINAMATH_GPT_negation_of_universal_prop_l619_61916

variable (a : ℝ)

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, 0 < x → Real.log x = a) ↔ (∃ x : ℝ, 0 < x ∧ Real.log x ≠ a) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_prop_l619_61916


namespace NUMINAMATH_GPT_is_not_age_of_child_l619_61931

-- Initial conditions
def mrs_smith_child_ages : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Given number
def n : Nat := 1124

-- Mrs. Smith's age 
noncomputable def mrs_smith_age : Nat := 46

-- Divisibility check
def is_divisible (n k : Nat) : Bool := n % k = 0

-- Prove the statement
theorem is_not_age_of_child (child_age : Nat) : 
  child_age ∈ mrs_smith_child_ages ∧ ¬ is_divisible n child_age → child_age = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_is_not_age_of_child_l619_61931


namespace NUMINAMATH_GPT_matrix_exponentiation_l619_61943

theorem matrix_exponentiation (a n : ℕ) (M : Matrix (Fin 3) (Fin 3) ℕ) (N : Matrix (Fin 3) (Fin 3) ℕ) :
  (M^n = N) →
  M = ![
    ![1, 3, a],
    ![0, 1, 5],
    ![0, 0, 1]
  ] →
  N = ![
    ![1, 27, 3060],
    ![0, 1, 45],
    ![0, 0, 1]
  ] →
  a + n = 289 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_matrix_exponentiation_l619_61943


namespace NUMINAMATH_GPT_graph_passes_through_fixed_point_l619_61976

-- Define the linear function given in the conditions
def linearFunction (k x y : ℝ) : ℝ :=
  (2 * k - 1) * x - (k + 3) * y - (k - 11)

-- Define the fixed point (2, 3)
def fixedPoint : ℝ × ℝ :=
  (2, 3)

-- State the theorem that the graph of the linear function always passes through the fixed point 
theorem graph_passes_through_fixed_point :
  ∀ k : ℝ, linearFunction k fixedPoint.1 fixedPoint.2 = 0 :=
by sorry  -- proof skipped

end NUMINAMATH_GPT_graph_passes_through_fixed_point_l619_61976


namespace NUMINAMATH_GPT_sum_of_integers_remainders_l619_61932

theorem sum_of_integers_remainders (a b c : ℕ) :
  (a % 15 = 11) →
  (b % 15 = 13) →
  (c % 15 = 14) →
  ((a + b + c) % 15 = 8) ∧ ((a + b + c) % 10 = 8) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_remainders_l619_61932


namespace NUMINAMATH_GPT_probability_multiple_of_45_l619_61961

def multiples_of_3 := [3, 6, 9]
def primes_less_than_20 := [2, 3, 5, 7, 11, 13, 17, 19]

def favorable_outcomes := (9, 5)
def total_outcomes := (multiples_of_3.length * primes_less_than_20.length)

theorem probability_multiple_of_45 : (multiples_of_3.length = 3 ∧ primes_less_than_20.length = 8) → 
  ∃ w : ℚ, w = 1 / 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_multiple_of_45_l619_61961


namespace NUMINAMATH_GPT_polynomial_roots_problem_l619_61921

theorem polynomial_roots_problem (γ δ : ℝ) (h₁ : γ^2 - 3*γ + 2 = 0) (h₂ : δ^2 - 3*δ + 2 = 0) :
  8*γ^3 - 6*δ^2 = 48 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_problem_l619_61921


namespace NUMINAMATH_GPT_max_value_m_l619_61948

noncomputable def max_m : ℝ := 10

theorem max_value_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = x + 2 * y) : x * y ≥ max_m - 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_m_l619_61948


namespace NUMINAMATH_GPT_greatest_common_divisor_is_40_l619_61907

def distance_to_boston : ℕ := 840
def distance_to_atlanta : ℕ := 440

theorem greatest_common_divisor_is_40 :
  Nat.gcd distance_to_boston distance_to_atlanta = 40 :=
by
  -- The theorem statement as described is correct
  -- Proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_greatest_common_divisor_is_40_l619_61907


namespace NUMINAMATH_GPT_min_variance_l619_61983

/--
Given a sample x, 1, y, 5 with an average of 2,
prove that the minimum value of the variance of this sample is 3.
-/
theorem min_variance (x y : ℝ) 
  (h_avg : (x + 1 + y + 5) / 4 = 2) :
  3 ≤ (1 / 4) * ((x - 2) ^ 2 + (y - 2) ^ 2 + (1 - 2) ^ 2 + (5 - 2) ^ 2) :=
sorry

end NUMINAMATH_GPT_min_variance_l619_61983


namespace NUMINAMATH_GPT_range_of_function_l619_61923

noncomputable def range_of_y : Set ℝ :=
  {y | ∃ x : ℝ, y = |x + 5| - |x - 3|}

theorem range_of_function : range_of_y = Set.Icc (-2) 12 :=
by
  sorry

end NUMINAMATH_GPT_range_of_function_l619_61923


namespace NUMINAMATH_GPT_solution_set_inequality_l619_61995

noncomputable def f : ℝ → ℝ := sorry

variable {f : ℝ → ℝ}
variable (hf_diff : Differentiable ℝ f)
variable (hf_ineq : ∀ x, f x > deriv f x)
variable (hf_zero : f 0 = 2)

theorem solution_set_inequality : {x : ℝ | f x < 2 * Real.exp x} = {x | 0 < x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l619_61995


namespace NUMINAMATH_GPT_melody_initial_food_l619_61919

-- Conditions
variable (dogs : ℕ) (food_per_meal : ℚ) (meals_per_day : ℕ) (days_in_week : ℕ) (food_left : ℚ)
variable (initial_food : ℚ)

-- Values given in the problem statement
axiom h_dogs : dogs = 3
axiom h_food_per_meal : food_per_meal = 1/2
axiom h_meals_per_day : meals_per_day = 2
axiom h_days_in_week : days_in_week = 7
axiom h_food_left : food_left = 9

-- Theorem to prove
theorem melody_initial_food : initial_food = 30 :=
  sorry

end NUMINAMATH_GPT_melody_initial_food_l619_61919


namespace NUMINAMATH_GPT_smallest_five_digit_perfect_square_and_cube_l619_61989

theorem smallest_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 := 
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_perfect_square_and_cube_l619_61989


namespace NUMINAMATH_GPT_domain_of_function_l619_61929

theorem domain_of_function :
  ∀ x, (2 * x - 1 ≥ 0) ∧ (x^2 ≠ 1) → (x ≥ 1/2 ∧ x < 1) ∨ (x > 1) := 
sorry

end NUMINAMATH_GPT_domain_of_function_l619_61929


namespace NUMINAMATH_GPT_prove_b_plus_m_equals_391_l619_61973

def matrix_A (b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ := ![
  ![1, 3, b],
  ![0, 1, 5],
  ![0, 0, 1]
]

def matrix_power_A (m b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ := 
  (matrix_A b)^(m : ℕ)

def target_matrix : Matrix (Fin 3) (Fin 3) ℕ := ![
  ![1, 21, 3003],
  ![0, 1, 45],
  ![0, 0, 1]
]

theorem prove_b_plus_m_equals_391 (b m : ℕ) (h1 : matrix_power_A m b = target_matrix) : b + m = 391 := by
  sorry

end NUMINAMATH_GPT_prove_b_plus_m_equals_391_l619_61973


namespace NUMINAMATH_GPT_mass_percentage_C_in_C6H8Ox_undetermined_l619_61996

-- Define the molar masses of Carbon, Hydrogen, and Oxygen
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.008
def molar_mass_O : ℝ := 16.00

-- Define the molecular formula
def molar_mass_C6H8O6 : ℝ := (6 * molar_mass_C) + (8 * molar_mass_H) + (6 * molar_mass_O)

-- Given the mass percentage of Carbon in C6H8O6
def mass_percentage_C_in_C6H8O6 : ℝ := 40.91

-- Problem Definition
theorem mass_percentage_C_in_C6H8Ox_undetermined (x : ℕ) : 
  x ≠ 6 → ¬ (∃ p : ℝ, p = (6 * molar_mass_C) / ((6 * molar_mass_C) + (8 * molar_mass_H) + x * molar_mass_O) * 100) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_mass_percentage_C_in_C6H8Ox_undetermined_l619_61996


namespace NUMINAMATH_GPT_solve_quadratic_l619_61974

theorem solve_quadratic : ∃ x : ℚ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 ∧ x = 5/3 := 
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l619_61974


namespace NUMINAMATH_GPT_college_application_ways_correct_l619_61927

def college_application_ways : ℕ :=
  -- Scenario 1: Student does not apply to either of the two conflicting colleges
  (Nat.choose 4 3) +
  -- Scenario 2: Student applies to one of the two conflicting colleges
  ((Nat.choose 2 1) * (Nat.choose 4 2))

theorem college_application_ways_correct : college_application_ways = 16 := by
  -- We can skip the proof
  sorry

end NUMINAMATH_GPT_college_application_ways_correct_l619_61927


namespace NUMINAMATH_GPT_number_of_pupils_l619_61997

-- Define the conditions.
variables (n : ℕ) -- Number of pupils in the class.

-- Axioms based on the problem statement.
axiom marks_difference : 67 - 45 = 22
axiom avg_increase : (1 / 2 : ℝ) * n = 22 

-- The theorem we need to prove.
theorem number_of_pupils : n = 44 := by
  -- Proof will go here.
  sorry

end NUMINAMATH_GPT_number_of_pupils_l619_61997


namespace NUMINAMATH_GPT_soda_difference_l619_61992

def regular_soda : ℕ := 67
def diet_soda : ℕ := 9

theorem soda_difference : regular_soda - diet_soda = 58 := 
  by
  sorry

end NUMINAMATH_GPT_soda_difference_l619_61992


namespace NUMINAMATH_GPT_distance_between_trees_l619_61972

def yard_length : ℕ := 350
def num_trees : ℕ := 26
def num_intervals : ℕ := num_trees - 1

theorem distance_between_trees :
  yard_length / num_intervals = 14 := 
sorry

end NUMINAMATH_GPT_distance_between_trees_l619_61972


namespace NUMINAMATH_GPT_marcia_banana_count_l619_61975

variable (B : ℕ)

-- Conditions
def appleCost := 2
def bananaCost := 1
def orangeCost := 3
def numApples := 12
def numOranges := 4
def avgCost := 2

-- Prove that given the conditions, B equals 4
theorem marcia_banana_count : 
  (24 + 12 + B) / (16 + B) = avgCost → B = 4 :=
by sorry

end NUMINAMATH_GPT_marcia_banana_count_l619_61975


namespace NUMINAMATH_GPT_trig_expression_value_l619_61913

open Real

theorem trig_expression_value (θ : ℝ)
  (h1 : cos (π - θ) > 0)
  (h2 : cos (π / 2 + θ) * (1 - 2 * cos (θ / 2) ^ 2) < 0) :
  (sin θ / |sin θ|) + (|cos θ| / cos θ) + (tan θ / |tan θ|) = -1 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l619_61913


namespace NUMINAMATH_GPT_product_xyz_equals_1080_l619_61908

noncomputable def xyz_product (x y z : ℝ) : ℝ :=
  if (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x * (y + z) = 198) ∧ (y * (z + x) = 216) ∧ (z * (x + y) = 234)
  then x * y * z
  else 0 

theorem product_xyz_equals_1080 {x y z : ℝ} :
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x * (y + z) = 198) ∧ (y * (z + x) = 216) ∧ (z * (x + y) = 234) →
  xyz_product x y z = 1080 :=
by
  intros h
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_product_xyz_equals_1080_l619_61908


namespace NUMINAMATH_GPT_log_product_eq_one_l619_61926

noncomputable def log_base (b a : ℝ) := Real.log a / Real.log b

theorem log_product_eq_one :
  log_base 2 3 * log_base 9 4 = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_log_product_eq_one_l619_61926


namespace NUMINAMATH_GPT_doughnuts_in_shop_l619_61991

def ratio_of_doughnuts_to_muffins : Nat := 5

def number_of_muffins_in_shop : Nat := 10

def number_of_doughnuts (D M : Nat) : Prop :=
  D = ratio_of_doughnuts_to_muffins * M

theorem doughnuts_in_shop :
  number_of_doughnuts D number_of_muffins_in_shop → D = 50 :=
by
  sorry

end NUMINAMATH_GPT_doughnuts_in_shop_l619_61991


namespace NUMINAMATH_GPT_determine_d_minus_b_l619_61938

theorem determine_d_minus_b 
  (a b c d : ℕ) 
  (h1 : a^5 = b^4)
  (h2 : c^3 = d^2)
  (h3 : c - a = 19) 
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  : d - b = 757 := 
  sorry

end NUMINAMATH_GPT_determine_d_minus_b_l619_61938


namespace NUMINAMATH_GPT_sarah_more_than_cecily_l619_61971

theorem sarah_more_than_cecily (t : ℕ) (ht : t = 144) :
  let s := (1 / 3 : ℚ) * t
  let a := (3 / 8 : ℚ) * t
  let c := t - (s + a)
  s - c = 6 := by
  sorry

end NUMINAMATH_GPT_sarah_more_than_cecily_l619_61971


namespace NUMINAMATH_GPT_largest_sum_is_8_over_15_l619_61917

theorem largest_sum_is_8_over_15 :
  max ((1 / 3) + (1 / 6)) (max ((1 / 3) + (1 / 7)) (max ((1 / 3) + (1 / 5)) (max ((1 / 3) + (1 / 9)) ((1 / 3) + (1 / 8))))) = 8 / 15 :=
sorry

end NUMINAMATH_GPT_largest_sum_is_8_over_15_l619_61917


namespace NUMINAMATH_GPT_simplify_expression_l619_61928

theorem simplify_expression : 
  -2^2003 + (-2)^2004 + 2^2005 - 2^2006 = -3 * 2^2003 := 
  by
    sorry

end NUMINAMATH_GPT_simplify_expression_l619_61928


namespace NUMINAMATH_GPT_combination_sum_l619_61967

theorem combination_sum :
  (Nat.choose 3 2) + (Nat.choose 4 2) + (Nat.choose 5 2) + (Nat.choose 6 2) = 34 :=
by
  sorry

end NUMINAMATH_GPT_combination_sum_l619_61967


namespace NUMINAMATH_GPT_not_p_is_sufficient_but_not_necessary_for_q_l619_61951

-- Definitions for the conditions
def p (x : ℝ) : Prop := x^2 > 4
def q (x : ℝ) : Prop := x ≤ 2

-- Definition of ¬p based on the solution derived
def not_p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- The theorem statement
theorem not_p_is_sufficient_but_not_necessary_for_q :
  ∀ x : ℝ, (not_p x → q x) ∧ ¬(q x → not_p x) := sorry

end NUMINAMATH_GPT_not_p_is_sufficient_but_not_necessary_for_q_l619_61951


namespace NUMINAMATH_GPT_count_negative_x_with_sqrt_pos_int_l619_61900

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end NUMINAMATH_GPT_count_negative_x_with_sqrt_pos_int_l619_61900


namespace NUMINAMATH_GPT_fourth_friend_age_is_8_l619_61954

-- Define the given data
variables (a1 a2 a3 a4 : ℕ)
variables (h_avg : (a1 + a2 + a3 + a4) / 4 = 9)
variables (h1 : a1 = 7) (h2 : a2 = 9) (h3 : a3 = 12)

-- Formalize the theorem to prove that the fourth friend's age is 8
theorem fourth_friend_age_is_8 : a4 = 8 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_fourth_friend_age_is_8_l619_61954


namespace NUMINAMATH_GPT_first_term_of_sequence_l619_61944

theorem first_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + 1) : a 1 = 2 := by
  sorry

end NUMINAMATH_GPT_first_term_of_sequence_l619_61944


namespace NUMINAMATH_GPT_belfried_payroll_l619_61982

noncomputable def tax_paid (payroll : ℝ) : ℝ :=
  if payroll < 200000 then 0 else 0.002 * (payroll - 200000)

theorem belfried_payroll (payroll : ℝ) (h : tax_paid payroll = 400) : payroll = 400000 :=
by
  sorry

end NUMINAMATH_GPT_belfried_payroll_l619_61982


namespace NUMINAMATH_GPT_donation_total_is_correct_l619_61960

-- Definitions and conditions
def Megan_inheritance : ℤ := 1000000
def Dan_inheritance : ℤ := 10000
def donation_percentage : ℚ := 0.1
def Megan_donation := Megan_inheritance * donation_percentage
def Dan_donation := Dan_inheritance * donation_percentage
def total_donation := Megan_donation + Dan_donation

-- Theorem statement
theorem donation_total_is_correct : total_donation = 101000 := by
  sorry

end NUMINAMATH_GPT_donation_total_is_correct_l619_61960


namespace NUMINAMATH_GPT_fish_filets_total_l619_61979

/- Define the number of fish caught by each family member -/
def ben_fish : ℕ := 4
def judy_fish : ℕ := 1
def billy_fish : ℕ := 3
def jim_fish : ℕ := 2
def susie_fish : ℕ := 5

/- Define the number of fish thrown back -/
def fish_thrown_back : ℕ := 3

/- Define the number of filets per fish -/
def filets_per_fish : ℕ := 2

/- Calculate the number of fish filets -/
theorem fish_filets_total : ℕ :=
  let total_fish_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let fish_kept := total_fish_caught - fish_thrown_back
  fish_kept * filets_per_fish

example : fish_filets_total = 24 :=
by {
  /- This 'sorry' placeholder indicates that a proof should be here -/
  sorry
}

end NUMINAMATH_GPT_fish_filets_total_l619_61979


namespace NUMINAMATH_GPT_factorize_expression_triangle_is_isosceles_l619_61970

-- Define the first problem: Factorize the expression.
theorem factorize_expression (a b : ℝ) : a^2 - 4 * a - b^2 + 4 = (a + b - 2) * (a - b - 2) := 
by
  sorry

-- Define the second problem: Determine the shape of the triangle.
theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - ab - ac + bc = 0) : a = b ∨ a = c :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_triangle_is_isosceles_l619_61970


namespace NUMINAMATH_GPT_minimize_distance_l619_61964

-- Definitions of points and lines in the Euclidean plane
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Line is defined by a point and a direction vector
structure Line : Type :=
(point : Point)
(direction : Point)

-- Distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Given conditions
variables (a b : Line) -- lines a and b
variables (A1 A2 : Point) -- positions of point A on line a
variables (B1 B2 : Point) -- positions of point B on line b

-- Hypotheses about uniform motion along the lines
def moves_uniformly (A1 A2 : Point) (a : Line) (B1 B2 : Point) (b : Line) : Prop :=
  ∀ t : ℝ, ∃ (At Bt : Point), 
  At.x = A1.x + t * (A2.x - A1.x) ∧ At.y = A1.y + t * (A2.y - A1.y) ∧
  Bt.x = B1.x + t * (B2.x - B1.x) ∧ Bt.y = B1.y + t * (B2.y - B1.y) ∧
  ∀ s : ℝ, At.x + s * (a.direction.x) = Bt.x + s * (b.direction.x) ∧
           At.y + s * (a.direction.y) = Bt.y + s * (b.direction.y)

-- Problem statement: Prove the existence of points such that AB is minimized
theorem minimize_distance (a b : Line) (A1 A2 B1 B2 : Point) (h : moves_uniformly A1 A2 a B1 B2 b) : 
  ∃ (A B : Point), distance A B = Real.sqrt ((A2.x - B2.x) ^ 2 + (A2.y - B2.y) ^ 2) ∧ distance A B ≤ distance A1 B1 ∧ distance A B ≤ distance A2 B2 :=
sorry

end NUMINAMATH_GPT_minimize_distance_l619_61964


namespace NUMINAMATH_GPT_sum_of_cuberoots_gt_two_l619_61909

theorem sum_of_cuberoots_gt_two {x₁ x₂ : ℝ} (h₁: x₁^3 = 6 / 5) (h₂: x₂^3 = 5 / 6) : x₁ + x₂ > 2 :=
sorry

end NUMINAMATH_GPT_sum_of_cuberoots_gt_two_l619_61909


namespace NUMINAMATH_GPT_carolyn_removal_sum_correct_l619_61939

-- Define the initial conditions
def n : Nat := 10
def initialList : List Nat := List.range (n + 1)  -- equals [0, 1, 2, ..., 10]

-- Given that Carolyn removes specific numbers based on the game rules
def carolynRemovals : List Nat := [6, 10, 8]

-- Sum of numbers removed by Carolyn
def carolynRemovalSum : Nat := carolynRemovals.sum

-- Theorem stating the sum of numbers removed by Carolyn
theorem carolyn_removal_sum_correct : carolynRemovalSum = 24 := by
  sorry

end NUMINAMATH_GPT_carolyn_removal_sum_correct_l619_61939


namespace NUMINAMATH_GPT_nth_term_206_l619_61922

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 10 ∧ a 1 = -10 ∧ ∀ n, a (n + 2) = -a n

theorem nth_term_206 (a : ℕ → ℝ) (h : geometric_sequence a) : a 205 = -10 :=
by
  -- Utilizing the sequence property to determine the 206th term
  sorry

end NUMINAMATH_GPT_nth_term_206_l619_61922


namespace NUMINAMATH_GPT_store_earnings_l619_61999

theorem store_earnings (num_pencils : ℕ) (num_erasers : ℕ) (price_eraser : ℝ) 
  (multiplier : ℝ) (price_pencil : ℝ) (total_earnings : ℝ) :
  num_pencils = 20 →
  price_eraser = 1 →
  num_erasers = num_pencils * 2 →
  price_pencil = (price_eraser * num_erasers) * multiplier →
  multiplier = 2 →
  total_earnings = num_pencils * price_pencil + num_erasers * price_eraser →
  total_earnings = 120 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_store_earnings_l619_61999


namespace NUMINAMATH_GPT_cost_of_french_bread_is_correct_l619_61947

noncomputable def cost_of_sandwiches := 2 * 7.75
noncomputable def cost_of_salami := 4.00
noncomputable def cost_of_brie := 3 * cost_of_salami
noncomputable def cost_of_olives := 10.00 * (1/4)
noncomputable def cost_of_feta := 8.00 * (1/2)
noncomputable def total_cost_of_items := cost_of_sandwiches + cost_of_salami + cost_of_brie + cost_of_olives + cost_of_feta
noncomputable def total_spent := 40.00
noncomputable def cost_of_french_bread := total_spent - total_cost_of_items

theorem cost_of_french_bread_is_correct :
  cost_of_french_bread = 2.00 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_french_bread_is_correct_l619_61947


namespace NUMINAMATH_GPT_fraction_relation_l619_61959

theorem fraction_relation (n d : ℕ) (h1 : (n + 1 : ℚ) / (d + 1) = 3 / 5) (h2 : (n : ℚ) / d = 5 / 9) :
  ∃ k : ℚ, d = k * 2 * n ∧ k = 9 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_relation_l619_61959


namespace NUMINAMATH_GPT_tan_A_tan_B_eq_one_third_l619_61949

theorem tan_A_tan_B_eq_one_third (A B C : ℕ) (hC : C = 120) (hSum : Real.tan A + Real.tan B = 2 * Real.sqrt 3 / 3) : 
  Real.tan A * Real.tan B = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_A_tan_B_eq_one_third_l619_61949


namespace NUMINAMATH_GPT_relationship_between_line_and_circle_l619_61935

variables {a b r : ℝ} (M : ℝ × ℝ) (l m : ℝ → ℝ)

def point_inside_circle_not_on_axes 
    (M : ℝ × ℝ) (r : ℝ) : Prop := 
    (M.fst^2 + M.snd^2 < r^2) ∧ (M.fst ≠ 0) ∧ (M.snd ≠ 0)

def line_eq (a b r : ℝ) (x y : ℝ) : Prop := 
    a * x + b * y = r^2

def chord_midpoint (M : ℝ × ℝ) (m : ℝ → ℝ) : Prop := 
    ∃ x1 y1 x2 y2, 
    (M.fst = (x1 + x2) / 2 ∧ M.snd = (y1 + y2) / 2) ∧ 
    (m x1 = y1 ∧ m x2 = y2)

def circle_external (O : ℝ → ℝ) (l : ℝ → ℝ) : Prop := 
    ∀ x y, O x = y → l x ≠ y

theorem relationship_between_line_and_circle
    (M_inside : point_inside_circle_not_on_axes M r)
    (M_chord : chord_midpoint M m)
    (line_eq_l : line_eq a b r M.fst M.snd) :
    (m (M.fst) = - (a / b) * M.snd) ∧ 
    (∀ x, l x ≠ m x) :=
sorry

end NUMINAMATH_GPT_relationship_between_line_and_circle_l619_61935


namespace NUMINAMATH_GPT_remainder_of_expression_l619_61993

theorem remainder_of_expression (k : ℤ) (hk : 0 < k) :
  (4 * k * (2 + 4 + 4 * k) + 3) % 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_expression_l619_61993


namespace NUMINAMATH_GPT_max_sundays_in_51_days_l619_61905

theorem max_sundays_in_51_days (days_in_week: ℕ) (total_days: ℕ) 
  (start_on_first: Bool) (first_day_sunday: Prop) 
  (is_sunday: ℕ → Bool) :
  days_in_week = 7 ∧ total_days = 51 ∧ start_on_first = tt ∧ first_day_sunday → 
  (∃ n, ∀ i < total_days, is_sunday i → n ≤ 8) ∧ 
  (∀ j, j ≤ total_days → is_sunday j → j ≤ 8) := by
  sorry

end NUMINAMATH_GPT_max_sundays_in_51_days_l619_61905


namespace NUMINAMATH_GPT_min_value_n_minus_m_l619_61978

noncomputable def f (x : ℝ) : ℝ :=
  if 1 < x then Real.log x else (1 / 2) * x + (1 / 2)

theorem min_value_n_minus_m (m n : ℝ) (hmn : m < n) (hf_eq : f m = f n) : n - m = 3 - 2 * Real.log 2 :=
  sorry

end NUMINAMATH_GPT_min_value_n_minus_m_l619_61978


namespace NUMINAMATH_GPT_probability_rain_all_three_days_l619_61940

-- Define the probabilities as constant values
def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.5
def prob_rain_sunday : ℝ := 0.3
def prob_rain_sunday_given_fri_sat : ℝ := 0.6

-- Define the probability of raining all three days considering the conditional probabilities
def prob_rain_all_three_days : ℝ :=
  prob_rain_friday * prob_rain_saturday * prob_rain_sunday_given_fri_sat

-- Prove that the probability of rain on all three days is 12%
theorem probability_rain_all_three_days : prob_rain_all_three_days = 0.12 :=
by
  sorry

end NUMINAMATH_GPT_probability_rain_all_three_days_l619_61940
