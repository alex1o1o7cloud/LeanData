import Mathlib

namespace NUMINAMATH_GPT_total_revenue_l1374_137485

theorem total_revenue (C A : ℕ) (P_C P_A total_tickets adult_tickets revenue : ℕ)
  (hCC : C = 6) -- Children's ticket price
  (hAC : A = 9) -- Adult's ticket price
  (hTT : total_tickets = 225) -- Total tickets sold
  (hAT : adult_tickets = 175) -- Adult tickets sold
  (hTR : revenue = 1875) -- Total revenue
  : revenue = adult_tickets * A + (total_tickets - adult_tickets) * C := sorry

end NUMINAMATH_GPT_total_revenue_l1374_137485


namespace NUMINAMATH_GPT_bead_game_solution_l1374_137478

-- Define the main theorem, stating the solution is valid for r = (b + 1) / b
theorem bead_game_solution {r : ℚ} (h : r > 1) (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 1010) :
  r = (b + 1) / b ∧ (∀ k : ℕ, k ≤ 2021 → True) := by
  sorry

end NUMINAMATH_GPT_bead_game_solution_l1374_137478


namespace NUMINAMATH_GPT_alternating_intersections_l1374_137445

theorem alternating_intersections (n : ℕ)
  (roads : Fin n → ℝ → ℝ) -- Roads are functions from reals to reals
  (h_straight : ∀ (i : Fin n), ∃ (a b : ℝ), ∀ x, roads i x = a * x + b) 
  (h_intersect : ∀ (i j : Fin n), i ≠ j → ∃ x, roads i x = roads j x)
  (h_two_roads : ∀ (x y : ℝ), ∃! (i j : Fin n), i ≠ j ∧ roads i x = roads j y) :
  ∃ (design : ∀ (i : Fin n), ℝ → Prop), 
  -- ensuring alternation, road 'i' alternates crossings with other roads 
  (∀ (i : Fin n) (x y : ℝ), 
    roads i x = roads i y → (design i x ↔ ¬design i y)) := sorry

end NUMINAMATH_GPT_alternating_intersections_l1374_137445


namespace NUMINAMATH_GPT_y_at_40_l1374_137486

def y_at_x (x : ℤ) : ℤ :=
  3 * x + 4

theorem y_at_40 : y_at_x 40 = 124 :=
by {
  sorry
}

end NUMINAMATH_GPT_y_at_40_l1374_137486


namespace NUMINAMATH_GPT_equal_animals_per_aquarium_l1374_137411

theorem equal_animals_per_aquarium (aquariums animals : ℕ) (h1 : aquariums = 26) (h2 : animals = 52) (h3 : ∀ a, a = animals / aquariums) : a = 2 := 
by
  sorry

end NUMINAMATH_GPT_equal_animals_per_aquarium_l1374_137411


namespace NUMINAMATH_GPT_contingency_fund_amount_l1374_137435

theorem contingency_fund_amount :
  ∀ (donation : ℝ),
  (1/3 * donation + 1/2 * donation + 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2  * donation)))) →
  (donation = 240) → (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = 30) :=
by
    intro donation h1 h2
    sorry

end NUMINAMATH_GPT_contingency_fund_amount_l1374_137435


namespace NUMINAMATH_GPT_find_f91_plus_fm91_l1374_137434

def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 3

theorem find_f91_plus_fm91 (a b c : ℝ) (h : f 91 a b c = 1) : f 91 a b c + f (-91) a b c = 2 := by
  sorry

end NUMINAMATH_GPT_find_f91_plus_fm91_l1374_137434


namespace NUMINAMATH_GPT_minimum_cuts_for_48_pieces_l1374_137404

theorem minimum_cuts_for_48_pieces 
  (rearrange_without_folding : Prop)
  (can_cut_multiple_layers_simultaneously : Prop)
  (straight_line_cut : Prop)
  (cut_doubles_pieces : ∀ n, ∃ m, m = 2 * n) :
  ∃ n, (2^n ≥ 48 ∧ ∀ m, (m < n → 2^m < 48)) ∧ n = 6 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_cuts_for_48_pieces_l1374_137404


namespace NUMINAMATH_GPT_mary_score_unique_l1374_137429

theorem mary_score_unique (c w : ℕ) (s : ℕ) (h_score_formula : s = 35 + 4 * c - w)
  (h_limit : c + w ≤ 35) (h_greater_90 : s > 90) :
  (∀ s' > 90, s' ≠ s → ¬ ∃ c' w', s' = 35 + 4 * c' - w' ∧ c' + w' ≤ 35) → s = 91 :=
by
  sorry

end NUMINAMATH_GPT_mary_score_unique_l1374_137429


namespace NUMINAMATH_GPT_geometric_solution_l1374_137446

theorem geometric_solution (x y : ℝ) (h : x^2 + 2 * y^2 - 10 * x + 12 * y + 43 = 0) : x = 5 ∧ y = -3 := 
  by sorry

end NUMINAMATH_GPT_geometric_solution_l1374_137446


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1374_137463

-- Part (a)
theorem part_a : 
  ∃ n : ℕ, n = 2023066 ∧ (∃ x y z : ℕ, x + y + z = 2013 ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

-- Part (b)
theorem part_b : 
  ∃ n : ℕ, n = 1006 ∧ (∃ x y z : ℕ, x + y + z = 2013 ∧ x = y ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

-- Part (c)
theorem part_c : 
  ∃ (x y z : ℕ), (x + y + z = 2013 ∧ (x * y * z = 671 * 671 * 671)) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1374_137463


namespace NUMINAMATH_GPT_paintings_on_Sep27_l1374_137414

-- Definitions for the problem conditions
def total_days := 6
def paintings_per_2_days := (6 : ℕ)
def paintings_per_3_days := (8 : ℕ)
def paintings_P22_to_P26 := 30

-- Function to calculate paintings over a given period
def paintings_in_days (days : ℕ) (frequency : ℕ) : ℕ := days / frequency

-- Function to calculate total paintings from the given artists
def total_paintings (d : ℕ) (p2 : ℕ) (p3 : ℕ) : ℕ :=
  p2 * paintings_in_days d 2 + p3 * paintings_in_days d 3

-- Calculate total paintings in 6 days
def total_paintings_in_6_days := total_paintings total_days paintings_per_2_days paintings_per_3_days

-- Proof problem: Show the number of paintings on the last day (September 27)
theorem paintings_on_Sep27 : total_paintings_in_6_days - paintings_P22_to_P26 = 4 :=
by
  sorry

end NUMINAMATH_GPT_paintings_on_Sep27_l1374_137414


namespace NUMINAMATH_GPT_algebra_expression_evaluation_l1374_137480

theorem algebra_expression_evaluation (a : ℝ) (h : a^2 + 2 * a - 1 = 5) : -2 * a^2 - 4 * a + 5 = -7 :=
by
  sorry

end NUMINAMATH_GPT_algebra_expression_evaluation_l1374_137480


namespace NUMINAMATH_GPT_power_function_passes_through_point_l1374_137424

theorem power_function_passes_through_point (a : ℝ) : (2 ^ a = Real.sqrt 2) → (a = 1 / 2) :=
  by
  intro h
  sorry

end NUMINAMATH_GPT_power_function_passes_through_point_l1374_137424


namespace NUMINAMATH_GPT_find_x1_l1374_137443

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3) : 
  x1 = 4 / 5 := 
  sorry

end NUMINAMATH_GPT_find_x1_l1374_137443


namespace NUMINAMATH_GPT_find_d_l1374_137441

theorem find_d (a b c d : ℤ) (h_poly : ∃ s1 s2 s3 s4 : ℤ, s1 > 0 ∧ s2 > 0 ∧ s3 > 0 ∧ s4 > 0 ∧ 
  ( ∀ x, (Polynomial.eval x (Polynomial.C d + Polynomial.X * Polynomial.C c + Polynomial.X^2 * Polynomial.C b + Polynomial.X^3 * Polynomial.C a + Polynomial.X^4)) =
    (x + s1) * (x + s2) * (x + s3) * (x + s4) ) ) 
  (h_sum : a + b + c + d = 2013) : d = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1374_137441


namespace NUMINAMATH_GPT_arithmetic_sequence_k_value_l1374_137472

theorem arithmetic_sequence_k_value (a : ℕ → ℤ) (S: ℕ → ℤ)
    (h1 : ∀ n, S (n + 1) = S n + a (n + 1))
    (h2 : S 11 = S 4)
    (h3 : a 1 = 1)
    (h4 : ∃ k, a k + a 4 = 0) :
    ∃ k, k = 12 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_k_value_l1374_137472


namespace NUMINAMATH_GPT_relationship_between_y_l1374_137426

theorem relationship_between_y (y1 y2 y3 : ℝ)
  (hA : y1 = 3 / -5)
  (hB : y2 = 3 / -3)
  (hC : y3 = 3 / 2) : y2 < y1 ∧ y1 < y3 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_y_l1374_137426


namespace NUMINAMATH_GPT_commodity_x_increase_rate_l1374_137489

variable (x_increase : ℕ) -- annual increase in cents of commodity X
variable (y_increase : ℕ := 20) -- annual increase in cents of commodity Y
variable (x_2001_price : ℤ := 420) -- price of commodity X in cents in 2001
variable (y_2001_price : ℤ := 440) -- price of commodity Y in cents in 2001
variable (year_difference : ℕ := 2010 - 2001) -- difference in years between 2010 and 2001
variable (x_y_diff_2010 : ℕ := 70) -- cents by which X is more expensive than Y in 2010

theorem commodity_x_increase_rate :
  x_increase * year_difference = (x_2001_price + x_increase * year_difference) - (y_2001_price + y_increase * year_difference) + x_y_diff_2010 := by
  sorry

end NUMINAMATH_GPT_commodity_x_increase_rate_l1374_137489


namespace NUMINAMATH_GPT_eqidistant_point_on_x_axis_l1374_137495

theorem eqidistant_point_on_x_axis (x : ℝ) : 
    (dist (x, 0) (-3, 0) = dist (x, 0) (2, 5)) → 
    x = 2 := by
  sorry

end NUMINAMATH_GPT_eqidistant_point_on_x_axis_l1374_137495


namespace NUMINAMATH_GPT_value_of_m_l1374_137418

-- Define the function given m
def f (x m : ℝ) : ℝ := x^2 - 2 * (abs x) + 2 - m

-- State the theorem to be proved
theorem value_of_m (m : ℝ) :
  (∃ x1 x2 x3 : ℝ, f x1 m = 0 ∧ f x2 m = 0 ∧ f x3 m = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1) →
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1374_137418


namespace NUMINAMATH_GPT_reciprocal_eq_self_l1374_137410

theorem reciprocal_eq_self {x : ℝ} (h : x ≠ 0) : (1 / x = x) → (x = 1 ∨ x = -1) :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_reciprocal_eq_self_l1374_137410


namespace NUMINAMATH_GPT_option_C_correct_l1374_137461

theorem option_C_correct {a : ℝ} : a^2 * a^3 = a^5 := by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_option_C_correct_l1374_137461


namespace NUMINAMATH_GPT_percentage_of_useful_items_l1374_137448

theorem percentage_of_useful_items
  (junk_percentage : ℚ)
  (useful_items junk_items total_items : ℕ)
  (h1 : junk_percentage = 0.70)
  (h2 : useful_items = 8)
  (h3 : junk_items = 28)
  (h4 : junk_percentage * total_items = junk_items) :
  (useful_items : ℚ) / (total_items : ℚ) * 100 = 20 :=
sorry

end NUMINAMATH_GPT_percentage_of_useful_items_l1374_137448


namespace NUMINAMATH_GPT_two_cos_45_eq_sqrt_two_l1374_137439

theorem two_cos_45_eq_sqrt_two
  (h1 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2) :
  2 * Real.cos (Real.pi / 4) = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_two_cos_45_eq_sqrt_two_l1374_137439


namespace NUMINAMATH_GPT_inequality_solution_range_l1374_137460

variable (a : ℝ)

def f (x : ℝ) := 2 * x^2 - 8 * x - 4

theorem inequality_solution_range :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ f x - a > 0) ↔ a < -4 := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_range_l1374_137460


namespace NUMINAMATH_GPT_solve_fractional_equation_l1374_137492

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -5) : 
    (2 * x / (x - 1)) - 1 = 4 / (1 - x) → x = -5 := 
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1374_137492


namespace NUMINAMATH_GPT_Dave_has_more_money_than_Derek_l1374_137422

def Derek_initial := 40
def Derek_expense1 := 14
def Derek_expense2 := 11
def Derek_expense3 := 5
def Derek_remaining := Derek_initial - Derek_expense1 - Derek_expense2 - Derek_expense3

def Dave_initial := 50
def Dave_expense := 7
def Dave_remaining := Dave_initial - Dave_expense

def money_difference := Dave_remaining - Derek_remaining

theorem Dave_has_more_money_than_Derek : money_difference = 33 := by sorry

end NUMINAMATH_GPT_Dave_has_more_money_than_Derek_l1374_137422


namespace NUMINAMATH_GPT_two_p_plus_q_l1374_137428

theorem two_p_plus_q (p q : ℚ) (h : p / q = 6 / 7) : 2 * p + q = 19 / 7 * q :=
by {
  sorry
}

end NUMINAMATH_GPT_two_p_plus_q_l1374_137428


namespace NUMINAMATH_GPT_domain_of_f_l1374_137475

noncomputable def f (x k : ℝ) := (3 * x ^ 2 + 4 * x - 7) / (-7 * x ^ 2 + 4 * x + k)

theorem domain_of_f {x k : ℝ} (h : k < -4/7): ∀ x, -7 * x ^ 2 + 4 * x + k ≠ 0 :=
by 
  intro x
  sorry

end NUMINAMATH_GPT_domain_of_f_l1374_137475


namespace NUMINAMATH_GPT_min_value_fraction_expression_l1374_137462

theorem min_value_fraction_expression {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 := 
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_expression_l1374_137462


namespace NUMINAMATH_GPT_minimum_a_plus_3b_l1374_137484

-- Define the conditions
variables (a b : ℝ)
axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_eq : a + 3 * b = 1 / a + 3 / b

-- State the theorem
theorem minimum_a_plus_3b (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a + 3 * b = 1 / a + 3 / b) : 
  a + 3 * b ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_a_plus_3b_l1374_137484


namespace NUMINAMATH_GPT_expected_allergies_correct_expected_both_correct_l1374_137408

noncomputable def p_allergies : ℚ := 2 / 7
noncomputable def sample_size : ℕ := 350
noncomputable def expected_allergies : ℚ := (2 / 7) * 350

noncomputable def p_left_handed : ℚ := 3 / 10
noncomputable def expected_both : ℚ := (3 / 10) * (2 / 7) * 350

theorem expected_allergies_correct : expected_allergies = 100 := by
  sorry

theorem expected_both_correct : expected_both = 30 := by
  sorry

end NUMINAMATH_GPT_expected_allergies_correct_expected_both_correct_l1374_137408


namespace NUMINAMATH_GPT_find_a_l1374_137488

theorem find_a (a : ℝ) (x : ℝ) (h₀ : a > 0) (h₁ : x > 0)
  (h₂ : a * Real.sqrt x = Real.log (Real.sqrt x))
  (h₃ : (a / (2 * Real.sqrt x)) = (1 / (2 * x))) : a = Real.exp (-1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1374_137488


namespace NUMINAMATH_GPT_base7_to_base10_l1374_137483

theorem base7_to_base10 : 
  let digit0 := 2
  let digit1 := 3
  let digit2 := 4
  let digit3 := 5
  let base := 7
  digit0 * base^0 + digit1 * base^1 + digit2 * base^2 + digit3 * base^3 = 1934 :=
by
  let digit0 := 2
  let digit1 := 3
  let digit2 := 4
  let digit3 := 5
  let base := 7
  sorry

end NUMINAMATH_GPT_base7_to_base10_l1374_137483


namespace NUMINAMATH_GPT_same_type_l1374_137454

variable (X Y : Prop) 

-- Definition of witnesses A and B based on their statements
def witness_A (A : Prop) := A ↔ (X → Y)
def witness_B (B : Prop) := B ↔ (¬X ∨ Y)

-- Proposition stating that A and B must be of the same type
theorem same_type (A B : Prop) (HA : witness_A X Y A) (HB : witness_B X Y B) : 
  (A = B) := 
sorry

end NUMINAMATH_GPT_same_type_l1374_137454


namespace NUMINAMATH_GPT_jenna_reading_pages_l1374_137455

theorem jenna_reading_pages :
  ∀ (total_pages goal_pages flight_pages busy_days total_days reading_days : ℕ),
    total_days = 30 →
    busy_days = 4 →
    flight_pages = 100 →
    goal_pages = 600 →
    reading_days = total_days - busy_days - 1 →
    (goal_pages - flight_pages) / reading_days = 20 :=
by
  intros total_pages goal_pages flight_pages busy_days total_days reading_days
  sorry

end NUMINAMATH_GPT_jenna_reading_pages_l1374_137455


namespace NUMINAMATH_GPT_flower_beds_fraction_l1374_137494

open Real

noncomputable def parkArea (a b h : ℝ) := (a + b) / 2 * h
noncomputable def triangleArea (a : ℝ) := (1 / 2) * a ^ 2

theorem flower_beds_fraction 
  (a b h : ℝ) 
  (h_a: a = 15) 
  (h_b: b = 30) 
  (h_h: h = (b - a) / 2) :
  (2 * triangleArea h) / parkArea a b h = 1 / 4 := by 
  sorry

end NUMINAMATH_GPT_flower_beds_fraction_l1374_137494


namespace NUMINAMATH_GPT_equation_has_at_least_two_distinct_roots_l1374_137477

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end NUMINAMATH_GPT_equation_has_at_least_two_distinct_roots_l1374_137477


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_equals_area_l1374_137444

/-- Given a right triangle where the hypotenuse is equal to the area, 
    show that the scaling factor x satisfies the equation. -/
theorem right_triangle_hypotenuse_equals_area 
  (m n x : ℝ) (h_hyp: (m^2 + n^2) * x = mn * (m^2 - n^2) * x^2) :
  x = (m^2 + n^2) / (mn * (m^2 - n^2)) := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_equals_area_l1374_137444


namespace NUMINAMATH_GPT_terminating_decimal_expansion_of_7_over_72_l1374_137419

theorem terminating_decimal_expansion_of_7_over_72 : (7 / 72) = 0.175 := 
sorry

end NUMINAMATH_GPT_terminating_decimal_expansion_of_7_over_72_l1374_137419


namespace NUMINAMATH_GPT_divisor_of_1053_added_with_5_is_2_l1374_137451

theorem divisor_of_1053_added_with_5_is_2 :
  ∃ d : ℕ, d > 1 ∧ ∀ (x : ℝ), x = 5.000000000000043 → (1053 + x) % d = 0 → d = 2 :=
by
  sorry

end NUMINAMATH_GPT_divisor_of_1053_added_with_5_is_2_l1374_137451


namespace NUMINAMATH_GPT_maximum_monthly_profit_l1374_137437

-- Let's set up our conditions

def selling_price := 25
def monthly_profit := 120
def cost_price := 20
def selling_price_threshold := 32
def relationship (x n : ℝ) := -10 * x + n

-- Define the value of n
def value_of_n : ℝ := 370

-- Profit function
def profit_function (x n : ℝ) : ℝ := (x - cost_price) * (relationship x n)

-- Define the condition for maximum profit where the selling price should be higher than 32
def max_profit_condition (n : ℝ) (x : ℝ) := x > selling_price_threshold

-- Define what the maximum profit should be
def max_profit := 160

-- The main theorem to be proven
theorem maximum_monthly_profit :
  (relationship selling_price value_of_n = monthly_profit) →
  max_profit_condition value_of_n 32 →
  profit_function 32 value_of_n = max_profit :=
by sorry

end NUMINAMATH_GPT_maximum_monthly_profit_l1374_137437


namespace NUMINAMATH_GPT_abs_eq_neg_self_iff_l1374_137420

theorem abs_eq_neg_self_iff (a : ℝ) : |a| = -a ↔ a ≤ 0 :=
by
  -- skipping proof with sorry
  sorry

end NUMINAMATH_GPT_abs_eq_neg_self_iff_l1374_137420


namespace NUMINAMATH_GPT_weight_of_rod_l1374_137433

theorem weight_of_rod (length1 length2 weight1 weight2 weight_per_meter : ℝ)
  (h1 : length1 = 6) (h2 : weight1 = 22.8) (h3 : length2 = 11.25)
  (h4 : weight_per_meter = weight1 / length1) :
  weight2 = weight_per_meter * length2 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_weight_of_rod_l1374_137433


namespace NUMINAMATH_GPT_range_of_x_l1374_137452

noncomputable def y (x : ℝ) : ℝ := (x + 2) / (x - 1)

theorem range_of_x : ∀ x : ℝ, (y x ≠ 0) → x ≠ 1 := by
  intro x h
  sorry

end NUMINAMATH_GPT_range_of_x_l1374_137452


namespace NUMINAMATH_GPT_vector_t_solution_l1374_137474

theorem vector_t_solution (t : ℝ) :
  ∃ t, (∃ (AB AC BC : ℝ × ℝ), 
         AB = (t, 1) ∧ AC = (2, 2) ∧ BC = (2 - t, 1) ∧ 
         (AC.1 - AB.1) * AC.1 + (AC.2 - AB.2) * AC.2 = 0 ) → 
         t = 3 :=
by {
  sorry -- proof content omitted as per instructions
}

end NUMINAMATH_GPT_vector_t_solution_l1374_137474


namespace NUMINAMATH_GPT_sqrt_of_26244_div_by_100_l1374_137476

theorem sqrt_of_26244_div_by_100 (h : Real.sqrt 262.44 = 16.2) : Real.sqrt 2.6244 = 1.62 :=
sorry

end NUMINAMATH_GPT_sqrt_of_26244_div_by_100_l1374_137476


namespace NUMINAMATH_GPT_unique_value_of_n_l1374_137440

theorem unique_value_of_n
  (n t : ℕ) (h1 : t ≠ 0)
  (h2 : 15 * t + (n - 20) * t / 3 = (n * t) / 2) :
  n = 50 :=
by sorry

end NUMINAMATH_GPT_unique_value_of_n_l1374_137440


namespace NUMINAMATH_GPT_δ_can_be_arbitrarily_small_l1374_137498

-- Define δ(r) as the distance from the circle to the nearest point with integer coordinates.
def δ (r : ℝ) : ℝ := sorry -- exact definition would depend on the implementation details

-- The main theorem to be proven.
theorem δ_can_be_arbitrarily_small (ε : ℝ) (hε : ε > 0) : ∃ r : ℝ, r > 0 ∧ δ r < ε :=
sorry

end NUMINAMATH_GPT_δ_can_be_arbitrarily_small_l1374_137498


namespace NUMINAMATH_GPT_students_meet_time_l1374_137427

theorem students_meet_time :
  ∀ (distance rate1 rate2 : ℝ),
    distance = 350 ∧ rate1 = 1.6 ∧ rate2 = 1.9 →
    distance / (rate1 + rate2) = 100 := by
  sorry

end NUMINAMATH_GPT_students_meet_time_l1374_137427


namespace NUMINAMATH_GPT_consecutive_even_integers_sum_l1374_137413

theorem consecutive_even_integers_sum :
  ∀ (y : Int), (y = 2 * (y + 2)) → y + (y + 2) = -6 :=
by
  intro y
  intro h
  sorry

end NUMINAMATH_GPT_consecutive_even_integers_sum_l1374_137413


namespace NUMINAMATH_GPT_problem_statement_l1374_137479

def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + 6*x + a^2 - 1

theorem problem_statement (a : ℝ) :
  f (Real.sqrt 2) a < f 4 a ∧ f 4 a < f 3 a :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1374_137479


namespace NUMINAMATH_GPT_find_monotonic_function_l1374_137487

-- Define Jensen's functional equation property
def jensens_eq (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 → f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y

-- Define monotonicity property
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The main theorem stating the equivalence
theorem find_monotonic_function (f : ℝ → ℝ) (h₁ : jensens_eq f) (h₂ : monotonic f) : 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := 
sorry

end NUMINAMATH_GPT_find_monotonic_function_l1374_137487


namespace NUMINAMATH_GPT_greatest_x_inequality_l1374_137499

theorem greatest_x_inequality :
  ∃ x, -x^2 + 11 * x - 28 = 0 ∧ (∀ y, -y^2 + 11 * y - 28 ≥ 0 → y ≤ x) ∧ x = 7 :=
sorry

end NUMINAMATH_GPT_greatest_x_inequality_l1374_137499


namespace NUMINAMATH_GPT_number_of_10_digit_numbers_divisible_by_66667_l1374_137442

def ten_digit_numbers_composed_of_3_4_5_6_divisible_by_66667 : ℕ := 33

theorem number_of_10_digit_numbers_divisible_by_66667 :
  ∃ n : ℕ, n = ten_digit_numbers_composed_of_3_4_5_6_divisible_by_66667 :=
by
  sorry

end NUMINAMATH_GPT_number_of_10_digit_numbers_divisible_by_66667_l1374_137442


namespace NUMINAMATH_GPT_multiply_polynomials_l1374_137409

theorem multiply_polynomials (x : ℝ) :
  (x^4 + 8 * x^2 + 64) * (x^2 - 8) = x^4 + 16 * x^2 :=
by
  sorry

end NUMINAMATH_GPT_multiply_polynomials_l1374_137409


namespace NUMINAMATH_GPT_initial_hours_per_day_l1374_137423

/-- 
Given:
1. 18 men working a certain number of hours per day dig 30 meters deep.
2. To dig to a depth of 50 meters, working 6 hours per day, 22 extra men should be put to work (total of 40 men).

Prove:
The initial 18 men were working \(\frac{200}{9}\) hours per day.
-/
theorem initial_hours_per_day 
  (h : ℚ)
  (work_done_18_men : 18 * h * 30 = 40 * 6 * 50) :
  h = 200 / 9 :=
by
  sorry

end NUMINAMATH_GPT_initial_hours_per_day_l1374_137423


namespace NUMINAMATH_GPT_solve_equation_l1374_137402

theorem solve_equation : ∀ x : ℝ, (2 / 3 * x - 2 = 4) → x = 9 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l1374_137402


namespace NUMINAMATH_GPT_S_11_is_22_l1374_137465

-- Definitions and conditions
variable (a_1 d : ℤ) -- first term and common difference of the arithmetic sequence
noncomputable def S (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- The given condition
variable (h : S a_1 d 8 - S a_1 d 3 = 10)

-- The proof goal
theorem S_11_is_22 : S a_1 d 11 = 22 :=
by
  sorry

end NUMINAMATH_GPT_S_11_is_22_l1374_137465


namespace NUMINAMATH_GPT_correct_operation_l1374_137457

-- Define the conditions as hypotheses
variable (a : ℝ)

-- A: \(a^2 \cdot a = a^3\)
def condition_A : Prop := a^2 * a = a^3

-- B: \((a^3)^3 = a^6\)
def condition_B : Prop := (a^3)^3 = a^6

-- C: \(a^3 + a^3 = a^5\)
def condition_C : Prop := a^3 + a^3 = a^5

-- D: \(a^6 \div a^2 = a^3\)
def condition_D : Prop := a^6 / a^2 = a^3

-- Proof that only condition A is correct:
theorem correct_operation : condition_A a ∧ ¬condition_B a ∧ ¬condition_C a ∧ ¬condition_D a :=
by
  sorry  -- Actual proofs would go here

end NUMINAMATH_GPT_correct_operation_l1374_137457


namespace NUMINAMATH_GPT_truck_travel_distance_l1374_137401

noncomputable def truck_distance (gallons: ℕ) : ℕ :=
  let efficiency_10_gallons := 300 / 10 -- miles per gallon
  let efficiency_initial := efficiency_10_gallons
  let efficiency_decreased := efficiency_initial * 9 / 10 -- 10% decrease
  if gallons <= 12 then
    gallons * efficiency_initial
  else
    12 * efficiency_initial + (gallons - 12) * efficiency_decreased

theorem truck_travel_distance (gallons: ℕ) :
  gallons = 15 → truck_distance gallons = 441 :=
by
  intros h
  rw [h]
  -- skipping proof
  sorry

end NUMINAMATH_GPT_truck_travel_distance_l1374_137401


namespace NUMINAMATH_GPT_sequence_general_term_l1374_137438

theorem sequence_general_term {a : ℕ → ℕ} (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 2) :
  ∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1 :=
by
  -- skip the proof with sorry
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1374_137438


namespace NUMINAMATH_GPT_max_product_of_sum_2000_l1374_137493

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end NUMINAMATH_GPT_max_product_of_sum_2000_l1374_137493


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l1374_137412

noncomputable def a_n (n : ℕ) : ℤ := 5 - 4 * n

theorem common_difference_arithmetic_sequence :
  ∀ n ≥ 1, a_n n - a_n (n - 1) = -4 :=
by
  intros n hn
  unfold a_n
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l1374_137412


namespace NUMINAMATH_GPT_find_acute_angle_l1374_137473

noncomputable def vector_a (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 2)
noncomputable def vector_b (α : ℝ) : ℝ × ℝ := (3, 4 * Real.sin α)
def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem find_acute_angle (α : ℝ) (h : are_parallel (vector_a α) (vector_b α)) (h_acute : 0 < α ∧ α < π / 2) : 
  α = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_acute_angle_l1374_137473


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1374_137417

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n m : ℕ, a (n+1) = a n + d

theorem arithmetic_sequence_problem
  (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 2 + a 3 = 32)
  (h2 : a 11 + a 12 + a 13 = 118) :
  a 4 + a 10 = 50 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1374_137417


namespace NUMINAMATH_GPT_find_a9_l1374_137453

variable (a : ℕ → ℤ)
variable (h1 : a 2 = -3)
variable (h2 : a 3 = -5)
variable (d : ℤ := a 3 - a 2)

theorem find_a9 : a 9 = -17 :=
by
  sorry

end NUMINAMATH_GPT_find_a9_l1374_137453


namespace NUMINAMATH_GPT_gcd_689_1021_l1374_137421

theorem gcd_689_1021 : Nat.gcd 689 1021 = 1 :=
by sorry

end NUMINAMATH_GPT_gcd_689_1021_l1374_137421


namespace NUMINAMATH_GPT_contradiction_proof_l1374_137415

theorem contradiction_proof (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end NUMINAMATH_GPT_contradiction_proof_l1374_137415


namespace NUMINAMATH_GPT_find_a_b_l1374_137458

-- Conditions defining the solution sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 3 }
def B : Set ℝ := { x | -3 < x ∧ x < 2 }

-- The solution set of the inequality x^2 + ax + b < 0 is the intersection A∩B
def C : Set ℝ := A ∩ B

-- Proving that there exist values of a and b such that the solution set C corresponds to the inequality x^2 + ax + b < 0
theorem find_a_b : ∃ a b : ℝ, (∀ x : ℝ, C x ↔ x^2 + a*x + b < 0) ∧ a + b = -3 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_b_l1374_137458


namespace NUMINAMATH_GPT_banana_difference_l1374_137491

theorem banana_difference (d : ℕ) :
  (8 + (8 + d) + (8 + 2 * d) + (8 + 3 * d) + (8 + 4 * d) = 100) →
  d = 6 :=
by
  sorry

end NUMINAMATH_GPT_banana_difference_l1374_137491


namespace NUMINAMATH_GPT_largest_multiple_of_7_negation_gt_neg150_l1374_137496

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_7_negation_gt_neg150_l1374_137496


namespace NUMINAMATH_GPT_f_g_x_eq_l1374_137466

noncomputable def f (x : ℝ) : ℝ := (x * (x + 1)) / 3
noncomputable def g (x : ℝ) : ℝ := x + 3

theorem f_g_x_eq (x : ℝ) : f (g x) = (x^2 + 7*x + 12) / 3 := by
  sorry

end NUMINAMATH_GPT_f_g_x_eq_l1374_137466


namespace NUMINAMATH_GPT_sum_of_coefficients_l1374_137407

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def polynomial (a : ℝ) (x : ℝ) : ℝ :=
  (2 + a * x) * (1 + x)^5

def x2_coefficient_condition (a : ℝ) : Prop :=
  2 * binomial_coefficient 5 2 + a * binomial_coefficient 5 1 = 15

theorem sum_of_coefficients (a : ℝ) (h : x2_coefficient_condition a) : 
  polynomial a 1 = 64 := 
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1374_137407


namespace NUMINAMATH_GPT_total_pieces_of_gum_l1374_137416

def packages := 43
def pieces_per_package := 23
def extra_pieces := 8

theorem total_pieces_of_gum :
  (packages * pieces_per_package) + extra_pieces = 997 := sorry

end NUMINAMATH_GPT_total_pieces_of_gum_l1374_137416


namespace NUMINAMATH_GPT_find_p_and_q_l1374_137449

theorem find_p_and_q :
  (∀ p q: ℝ, (∃ x : ℝ, x^2 + p * x + q = 0 ∧ q * x^2 + p * x + 1 = 0) ∧ (-2) ^ 2 + p * (-2) + q = 0 ∧ p ≠ 0 ∧ q ≠ 0 → 
    (p, q) = (1, -2) ∨ (p, q) = (3, 2) ∨ (p, q) = (5/2, 1)) :=
sorry

end NUMINAMATH_GPT_find_p_and_q_l1374_137449


namespace NUMINAMATH_GPT_line_through_point_equal_intercepts_l1374_137436

theorem line_through_point_equal_intercepts (x y a b : ℝ) :
  ∀ (x y : ℝ), 
    (x - 1) = a → 
    (y - 2) = b →
    (a = -1 ∨ a = 2) → 
    ((x + y - 3 = 0) ∨ (2 * x - y = 0)) := by
  sorry

end NUMINAMATH_GPT_line_through_point_equal_intercepts_l1374_137436


namespace NUMINAMATH_GPT_divisor_is_twelve_l1374_137490

theorem divisor_is_twelve (d : ℕ) (h : 64 = 5 * d + 4) : d = 12 := 
sorry

end NUMINAMATH_GPT_divisor_is_twelve_l1374_137490


namespace NUMINAMATH_GPT_susan_fraction_apples_given_out_l1374_137400

theorem susan_fraction_apples_given_out (frank_apples : ℕ) (frank_sold_fraction : ℚ) 
  (total_remaining_apples : ℕ) (susan_multiple : ℕ) 
  (H1 : frank_apples = 36) 
  (H2 : susan_multiple = 3) 
  (H3 : frank_sold_fraction = 1 / 3) 
  (H4 : total_remaining_apples = 78) :
  let susan_apples := susan_multiple * frank_apples
  let frank_sold_apples := frank_sold_fraction * frank_apples
  let frank_remaining_apples := frank_apples - frank_sold_apples
  let total_before_susan_gave_out := susan_apples + frank_remaining_apples
  let susan_gave_out := total_before_susan_gave_out - total_remaining_apples
  let susan_gave_fraction := susan_gave_out / susan_apples
  susan_gave_fraction = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_susan_fraction_apples_given_out_l1374_137400


namespace NUMINAMATH_GPT_find_length_DY_l1374_137431

noncomputable def length_DY : Real :=
    let AE := 2
    let AY := 4 * AE
    let DY  := Real.sqrt (66 + Real.sqrt 5)
    DY

theorem find_length_DY : length_DY = Real.sqrt (66 + Real.sqrt 5) := 
  by
    sorry

end NUMINAMATH_GPT_find_length_DY_l1374_137431


namespace NUMINAMATH_GPT_ratio_of_cubes_l1374_137467

/-- A cubical block of metal weighs 7 pounds. Another cube of the same metal, with sides of a certain ratio longer, weighs 56 pounds. Prove that the ratio of the side length of the second cube to the first cube is 2:1. --/
theorem ratio_of_cubes (s r : ℝ) (weight1 weight2 : ℝ)
  (h1 : weight1 = 7) (h2 : weight2 = 56)
  (h_vol1 : weight1 = s^3)
  (h_vol2 : weight2 = (r * s)^3) :
  r = 2 := 
sorry

end NUMINAMATH_GPT_ratio_of_cubes_l1374_137467


namespace NUMINAMATH_GPT_circles_tangent_l1374_137468

theorem circles_tangent
  (rA rB rC rD rF : ℝ) (rE : ℚ) (m n : ℕ)
  (m_n_rel_prime : Int.gcd m n = 1)
  (rA_pos : 0 < rA) (rB_pos : 0 < rB)
  (rC_pos : 0 < rC) (rD_pos : 0 < rD)
  (rF_pos : 0 < rF)
  (inscribed_triangle_in_A : True)  -- Triangle T is inscribed in circle A
  (B_tangent_A : True)  -- Circle B is internally tangent to circle A
  (C_tangent_A : True)  -- Circle C is internally tangent to circle A
  (D_tangent_A : True)  -- Circle D is internally tangent to circle A
  (B_externally_tangent_E : True)  -- Circle B is externally tangent to circle E
  (C_externally_tangent_E : True)  -- Circle C is externally tangent to circle E
  (D_externally_tangent_E : True)  -- Circle D is externally tangent to circle E
  (F_tangent_A : True)  -- Circle F is internally tangent to circle A at midpoint of side opposite to B's tangency
  (F_externally_tangent_E : True)  -- Circle F is externally tangent to circle E
  (rA_eq : rA = 12) (rB_eq : rB = 5)
  (rC_eq : rC = 3) (rD_eq : rD = 2)
  (rF_eq : rF = 1)
  (rE_eq : rE = m / n)
  : m + n = 23 :=
by
  sorry

end NUMINAMATH_GPT_circles_tangent_l1374_137468


namespace NUMINAMATH_GPT_max_possible_b_l1374_137447

theorem max_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
by sorry

end NUMINAMATH_GPT_max_possible_b_l1374_137447


namespace NUMINAMATH_GPT_man_speed_proof_l1374_137459

noncomputable def train_length : ℝ := 150 
noncomputable def crossing_time : ℝ := 6 
noncomputable def train_speed_kmph : ℝ := 84.99280057595394 
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

noncomputable def relative_speed_mps : ℝ := train_length / crossing_time
noncomputable def man_speed_mps : ℝ := relative_speed_mps - train_speed_mps
noncomputable def man_speed_kmph : ℝ := man_speed_mps * (3600 / 1000)

theorem man_speed_proof : man_speed_kmph = 5.007198224048459 := by 
  sorry

end NUMINAMATH_GPT_man_speed_proof_l1374_137459


namespace NUMINAMATH_GPT_polygon_sides_l1374_137481

-- Definitions of the conditions
def is_regular_polygon (n : ℕ) (int_angle ext_angle : ℝ) : Prop :=
  int_angle = 5 * ext_angle ∧ (int_angle + ext_angle = 180)

-- Main theorem statement
theorem polygon_sides (n : ℕ) (int_angle ext_angle : ℝ) :
  is_regular_polygon n int_angle ext_angle →
  (ext_angle = 360 / n) →
  n = 12 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l1374_137481


namespace NUMINAMATH_GPT_exam_maximum_marks_l1374_137450

theorem exam_maximum_marks :
  (∃ M S E : ℕ, 
    (90 + 20 = 40 * M / 100) ∧ 
    (110 + 35 = 35 * S / 100) ∧ 
    (80 + 10 = 30 * E / 100) ∧ 
    M = 275 ∧ 
    S = 414 ∧ 
    E = 300) :=
by
  sorry

end NUMINAMATH_GPT_exam_maximum_marks_l1374_137450


namespace NUMINAMATH_GPT_find_sin_θ_l1374_137405

open Real

noncomputable def θ_in_range_and_sin_2θ (θ : ℝ) : Prop :=
  (θ ∈ Set.Icc (π / 4) (π / 2)) ∧ (sin (2 * θ) = 3 * sqrt 7 / 8)

theorem find_sin_θ (θ : ℝ) (h : θ_in_range_and_sin_2θ θ) : sin θ = 3 / 4 :=
  sorry

end NUMINAMATH_GPT_find_sin_θ_l1374_137405


namespace NUMINAMATH_GPT_food_expenditure_increase_l1374_137406

-- Conditions
def linear_relationship (x : ℝ) : ℝ := 0.254 * x + 0.321

-- Proof statement
theorem food_expenditure_increase (x : ℝ) : linear_relationship (x + 1) - linear_relationship x = 0.254 :=
by
  sorry

end NUMINAMATH_GPT_food_expenditure_increase_l1374_137406


namespace NUMINAMATH_GPT_f_g_eq_g_f_iff_n_zero_l1374_137482

def f (x n : ℝ) : ℝ := x + n
def g (x q : ℝ) : ℝ := x^2 + q

theorem f_g_eq_g_f_iff_n_zero (x n q : ℝ) : (f (g x q) n = g (f x n) q) ↔ n = 0 := by 
  sorry

end NUMINAMATH_GPT_f_g_eq_g_f_iff_n_zero_l1374_137482


namespace NUMINAMATH_GPT_cups_of_flour_put_in_l1374_137456

-- Conditions
def recipeSugar : ℕ := 3
def recipeFlour : ℕ := 10
def neededMoreFlourThanSugar : ℕ := 5

-- Question: How many cups of flour did she put in?
-- Answer: 5 cups of flour
theorem cups_of_flour_put_in : (recipeSugar + neededMoreFlourThanSugar = recipeFlour) → recipeFlour - neededMoreFlourThanSugar = 5 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_cups_of_flour_put_in_l1374_137456


namespace NUMINAMATH_GPT_equation_1_solution_equation_2_solution_l1374_137470

theorem equation_1_solution (x : ℝ) : (x-1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4 := 
by 
  sorry

theorem equation_2_solution (x : ℝ) : 3 * x * (x - 2) = x -2 ↔ x = 2 ∨ x = 1/3 := 
by 
  sorry

end NUMINAMATH_GPT_equation_1_solution_equation_2_solution_l1374_137470


namespace NUMINAMATH_GPT_total_spending_l1374_137430

-- Conditions
def pop_spending : ℕ := 15
def crackle_spending : ℕ := 3 * pop_spending
def snap_spending : ℕ := 2 * crackle_spending

-- Theorem stating the total spending
theorem total_spending : snap_spending + crackle_spending + pop_spending = 150 :=
by
  sorry

end NUMINAMATH_GPT_total_spending_l1374_137430


namespace NUMINAMATH_GPT_sum_of_numbers_l1374_137471

theorem sum_of_numbers (x y : ℕ) (h1 : x * y = 9375) (h2 : y / x = 15) : x + y = 400 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1374_137471


namespace NUMINAMATH_GPT_even_quadruple_composition_l1374_137425

variable {α : Type*} [AddGroup α]

-- Definition of an odd function
def is_odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

theorem even_quadruple_composition {f : α → α} 
  (hf_odd : is_odd_function f) : 
  ∀ x, f (f (f (f x))) = f (f (f (f (-x)))) :=
by
  sorry

end NUMINAMATH_GPT_even_quadruple_composition_l1374_137425


namespace NUMINAMATH_GPT_employee_pays_correct_amount_l1374_137432

theorem employee_pays_correct_amount
    (wholesale_cost : ℝ)
    (retail_markup : ℝ)
    (employee_discount : ℝ)
    (weekend_discount : ℝ)
    (sales_tax : ℝ)
    (final_price : ℝ) :
    wholesale_cost = 200 →
    retail_markup = 0.20 →
    employee_discount = 0.05 →
    weekend_discount = 0.10 →
    sales_tax = 0.08 →
    final_price = 221.62 :=
by
  intros h0 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_employee_pays_correct_amount_l1374_137432


namespace NUMINAMATH_GPT_graph_transformation_point_l1374_137469

theorem graph_transformation_point {f : ℝ → ℝ} (h : f 1 = 0) : f (0 + 1) + 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_graph_transformation_point_l1374_137469


namespace NUMINAMATH_GPT_no_nat_solution_l1374_137464

theorem no_nat_solution (x y z : ℕ) : ¬ (x^3 + 2 * y^3 = 4 * z^3) :=
sorry

end NUMINAMATH_GPT_no_nat_solution_l1374_137464


namespace NUMINAMATH_GPT_solve_quadratic_l1374_137497

theorem solve_quadratic (x : ℝ) (h : x^2 - 2 * x - 3 = 0) : x = 3 ∨ x = -1 := 
sorry

end NUMINAMATH_GPT_solve_quadratic_l1374_137497


namespace NUMINAMATH_GPT_polygon_diagonals_l1374_137403

theorem polygon_diagonals (n : ℕ) (k_0 k_1 k_2 : ℕ)
  (h1 : 2 * k_2 + k_1 = n)
  (h2 : k_2 + k_1 + k_0 = n - 2) :
  k_2 ≥ 2 :=
sorry

end NUMINAMATH_GPT_polygon_diagonals_l1374_137403
