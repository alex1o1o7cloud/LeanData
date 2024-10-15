import Mathlib

namespace NUMINAMATH_GPT_son_l1122_112251

theorem son's_age (S F : ℕ) (h₁ : F = 7 * (S - 8)) (h₂ : F / 4 = 14) : S = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_son_l1122_112251


namespace NUMINAMATH_GPT_linear_function_points_relation_l1122_112241

theorem linear_function_points_relation (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : y1 = 5 * x1 - 3) 
  (h2 : y2 = 5 * x2 - 3) 
  (h3 : x1 < x2) : 
  y1 < y2 :=
sorry

end NUMINAMATH_GPT_linear_function_points_relation_l1122_112241


namespace NUMINAMATH_GPT_adding_2_to_odd_integer_can_be_prime_l1122_112211

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0
def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem adding_2_to_odd_integer_can_be_prime :
  ∃ n : ℤ, is_odd n ∧ is_prime (n + 2) :=
by
  sorry

end NUMINAMATH_GPT_adding_2_to_odd_integer_can_be_prime_l1122_112211


namespace NUMINAMATH_GPT_brain_can_always_open_door_l1122_112250

noncomputable def can_open_door (a b c n m k : ℕ) : Prop :=
∃ x y z : ℕ, a^n = x^3 ∧ b^m = y^3 ∧ c^k = z^3

theorem brain_can_always_open_door :
  ∀ (a b c n m k : ℕ), 
  ∃ x y z : ℕ, a^n = x^3 ∧ b^m = y^3 ∧ c^k = z^3 :=
by sorry

end NUMINAMATH_GPT_brain_can_always_open_door_l1122_112250


namespace NUMINAMATH_GPT_bandit_showdown_l1122_112226

theorem bandit_showdown :
  ∃ b : ℕ, b ≥ 8 ∧ b < 50 ∧
         ∀ i j : ℕ, i ≠ j → (i < 50 ∧ j < 50) →
         ∃ k : ℕ, k < 50 ∧
         ∀ b : ℕ, b < 50 → 
         ∃ l m : ℕ, l ≠ m ∧ l < 50 ∧ m < 50 ∧ l ≠ b ∧ m ≠ b :=
sorry

end NUMINAMATH_GPT_bandit_showdown_l1122_112226


namespace NUMINAMATH_GPT_maximal_number_of_coins_l1122_112200

noncomputable def largest_number_of_coins (n k : ℕ) : Prop :=
n < 100 ∧ n = 12 * k + 3

theorem maximal_number_of_coins (n k : ℕ) : largest_number_of_coins n k → n = 99 :=
by
  sorry

end NUMINAMATH_GPT_maximal_number_of_coins_l1122_112200


namespace NUMINAMATH_GPT_smallest_b_factors_l1122_112243

theorem smallest_b_factors (b p q : ℤ) (hb : b = p + q) (hpq : p * q = 2052) : b = 132 :=
sorry

end NUMINAMATH_GPT_smallest_b_factors_l1122_112243


namespace NUMINAMATH_GPT_color_films_count_l1122_112245

variables (x y C : ℕ)
variables (h1 : 0.9615384615384615 = (C : ℝ) / ((2 * (y : ℝ) / 5) + (C : ℝ)))

theorem color_films_count (x y : ℕ) (C : ℕ) (h1 : 0.9615384615384615 = (C : ℝ) / ((2 * (y : ℝ) / 5) + (C : ℝ))) :
  C = 10 * y :=
sorry

end NUMINAMATH_GPT_color_films_count_l1122_112245


namespace NUMINAMATH_GPT_trees_left_after_typhoon_l1122_112249

variable (initial_trees : ℕ)
variable (died_trees : ℕ)
variable (remaining_trees : ℕ)

theorem trees_left_after_typhoon :
  initial_trees = 20 →
  died_trees = 16 →
  remaining_trees = initial_trees - died_trees →
  remaining_trees = 4 :=
by
  intros h_initial h_died h_remaining
  rw [h_initial, h_died] at h_remaining
  exact h_remaining

end NUMINAMATH_GPT_trees_left_after_typhoon_l1122_112249


namespace NUMINAMATH_GPT_probability_of_roots_l1122_112278

theorem probability_of_roots (k : ℝ) (h1 : 8 ≤ k) (h2 : k ≤ 13) :
  let a := k^2 - 2 * k - 35
  let b := 3 * k - 9
  let c := 2
  let discriminant := b^2 - 4 * a * c
  discriminant ≥ 0 → 
  (∃ x1 x2 : ℝ, 
    a * x1^2 + b * x1 + c = 0 ∧ 
    a * x2^2 + b * x2 + c = 0 ∧
    x1 ≤ 2 * x2) ↔ 
  ∃ p : ℝ, p = 0.6 := 
sorry

end NUMINAMATH_GPT_probability_of_roots_l1122_112278


namespace NUMINAMATH_GPT_total_time_l1122_112267

/-- Define the different time periods in years --/
def getting_in_shape : ℕ := 2
def learning_to_climb : ℕ := 2 * getting_in_shape
def months_climbing : ℕ := 7 * 5
def climbing : ℚ := months_climbing / 12
def break_after_climbing : ℚ := 13 / 12
def diving : ℕ := 2

/-- Prove that the total time taken to achieve all goals is 12 years --/
theorem total_time : getting_in_shape + learning_to_climb + climbing + break_after_climbing + diving = 12 := by
  sorry

end NUMINAMATH_GPT_total_time_l1122_112267


namespace NUMINAMATH_GPT_parabola_standard_equations_l1122_112268

noncomputable def parabola_focus_condition (x y : ℝ) : Prop := 
  x + 2 * y + 3 = 0

theorem parabola_standard_equations (x y : ℝ) 
  (h : parabola_focus_condition x y) :
  (y ^ 2 = -12 * x) ∨ (x ^ 2 = -6 * y) :=
by
  sorry

end NUMINAMATH_GPT_parabola_standard_equations_l1122_112268


namespace NUMINAMATH_GPT_book_vs_necklace_price_difference_l1122_112291

-- Problem-specific definitions and conditions
def necklace_price : ℕ := 34
def limit_price : ℕ := 70
def overspent : ℕ := 3
def total_spent : ℕ := limit_price + overspent
def book_price : ℕ := total_spent - necklace_price

-- Lean statement to prove the correct answer
theorem book_vs_necklace_price_difference :
  book_price - necklace_price = 5 := by
  sorry

end NUMINAMATH_GPT_book_vs_necklace_price_difference_l1122_112291


namespace NUMINAMATH_GPT_line_passes_through_point_has_correct_equation_l1122_112295

theorem line_passes_through_point_has_correct_equation :
  (∃ (L : ℝ × ℝ → Prop), (L (-2, 5)) ∧ (∃ m : ℝ, m = -3 / 4 ∧ ∀ (x y : ℝ), L (x, y) ↔ y - 5 = -3 / 4 * (x + 2))) →
  ∀ x y : ℝ, (3 * x + 4 * y - 14 = 0) ↔ (y - 5 = -3 / 4 * (x + 2)) :=
by
  intro h_L
  sorry

end NUMINAMATH_GPT_line_passes_through_point_has_correct_equation_l1122_112295


namespace NUMINAMATH_GPT_correct_avg_weight_l1122_112277

theorem correct_avg_weight (initial_avg_weight : ℚ) (num_boys : ℕ) (misread_weight : ℚ) (correct_weight : ℚ) :
  initial_avg_weight = 58.4 → num_boys = 20 → misread_weight = 56 → correct_weight = 60 →
  (initial_avg_weight * num_boys + (correct_weight - misread_weight)) / num_boys = 58.6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Plugging in the values makes the calculation straightforward, resulting in: 
  -- (58.4 * 20 + (60 - 56)) / 20 = 58.6 
  -- thus this verification step is:
  sorry

end NUMINAMATH_GPT_correct_avg_weight_l1122_112277


namespace NUMINAMATH_GPT_inequality_solution_sets_l1122_112223

theorem inequality_solution_sets (a : ℝ)
  (h1 : ∀ x : ℝ, (1/2) < x ∧ x < 2 ↔ ax^2 + 5*x - 2 > 0) :
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) ↔ ax^2 - 5*x + a^2 - 1 > 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_solution_sets_l1122_112223


namespace NUMINAMATH_GPT_factor_expression_l1122_112202

-- Define the expression to be factored
def expr (b : ℝ) := 348 * b^2 + 87 * b + 261

-- Define the supposedly factored form of the expression
def factored_expr (b : ℝ) := 87 * (4 * b^2 + b + 3)

-- The theorem stating that the original expression is equal to its factored form
theorem factor_expression (b : ℝ) : expr b = factored_expr b := 
by
  unfold expr factored_expr
  sorry

end NUMINAMATH_GPT_factor_expression_l1122_112202


namespace NUMINAMATH_GPT_billed_minutes_l1122_112265

noncomputable def John_bill (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) : ℝ :=
  (total_bill - monthly_fee) / cost_per_minute

theorem billed_minutes : ∀ (monthly_fee cost_per_minute total_bill : ℝ), 
  monthly_fee = 5 → 
  cost_per_minute = 0.25 → 
  total_bill = 12.02 → 
  John_bill monthly_fee cost_per_minute total_bill = 28 :=
by
  intros monthly_fee cost_per_minute total_bill hf hm hb
  rw [hf, hm, hb, John_bill]
  norm_num
  sorry

end NUMINAMATH_GPT_billed_minutes_l1122_112265


namespace NUMINAMATH_GPT_divisor_exists_l1122_112281

theorem divisor_exists (n : ℕ) : (∃ k, 10 ≤ k ∧ k ≤ 50 ∧ n ∣ k) →
                                (∃ k, 10 ≤ k ∧ k ≤ 50 ∧ n ∣ k) ∧
                                (n = 3) :=
by
  sorry

end NUMINAMATH_GPT_divisor_exists_l1122_112281


namespace NUMINAMATH_GPT_hide_and_seek_friends_l1122_112229

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end NUMINAMATH_GPT_hide_and_seek_friends_l1122_112229


namespace NUMINAMATH_GPT_g_at_pi_over_4_l1122_112235

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) / 2 * Real.sin (2 * x) + (Real.sqrt 6) / 2 * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 4)

theorem g_at_pi_over_4 : g (Real.pi / 4) = (Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_GPT_g_at_pi_over_4_l1122_112235


namespace NUMINAMATH_GPT_bottle_caps_total_l1122_112221

theorem bottle_caps_total (groups : ℕ) (bottle_caps_per_group : ℕ) (h1 : groups = 7) (h2 : bottle_caps_per_group = 5) : (groups * bottle_caps_per_group = 35) :=
by
  sorry

end NUMINAMATH_GPT_bottle_caps_total_l1122_112221


namespace NUMINAMATH_GPT_cost_of_weed_eater_string_l1122_112247

-- Definitions
def num_blades := 4
def cost_per_blade := 8
def total_spent := 39
def total_cost_of_blades := num_blades * cost_per_blade
def cost_of_string := total_spent - total_cost_of_blades

-- The theorem statement
theorem cost_of_weed_eater_string : cost_of_string = 7 :=
by {
  -- The proof would go here
  sorry
}

end NUMINAMATH_GPT_cost_of_weed_eater_string_l1122_112247


namespace NUMINAMATH_GPT_find_other_number_l1122_112260

def smallest_multiple_of_711 (n : ℕ) : ℕ := Nat.lcm n 711

theorem find_other_number (n : ℕ) : smallest_multiple_of_711 n = 3555 → n = 5 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l1122_112260


namespace NUMINAMATH_GPT_points_on_line_l1122_112248

theorem points_on_line (x y : ℝ) (h : x + y = 0) : y = -x :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_l1122_112248


namespace NUMINAMATH_GPT_unique_solution_l1122_112279

theorem unique_solution (m n : ℤ) (h : 231 * m^2 = 130 * n^2) : m = 0 ∧ n = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_solution_l1122_112279


namespace NUMINAMATH_GPT_find_a_f_greater_than_1_l1122_112298

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) := x^2 * Real.exp x - a * Real.log x

-- Condition: Slope at x = 1 is 3e - 1
theorem find_a (a : ℝ) (h : deriv (fun x => f x a) 1 = 3 * Real.exp 1 - 1) : a = 1 := sorry

-- Given a = 1
theorem f_greater_than_1 (x : ℝ) (hx : x > 0) : f x 1 > 1 := sorry

end NUMINAMATH_GPT_find_a_f_greater_than_1_l1122_112298


namespace NUMINAMATH_GPT_circle_through_two_points_on_y_axis_l1122_112284

theorem circle_through_two_points_on_y_axis :
  ∃ (b : ℝ), (∀ (x y : ℝ), (x + 1)^2 + (y - 4)^2 = (x - 3)^2 + (y - 2)^2 → b = 1) ∧ 
  (∀ (x y : ℝ), (x - 0)^2 + (y - b)^2 = 10) := 
sorry

end NUMINAMATH_GPT_circle_through_two_points_on_y_axis_l1122_112284


namespace NUMINAMATH_GPT_rectangle_perimeter_l1122_112227

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def satisfies_relations (b1 b2 b3 b4 b5 b6 b7 : ℕ) : Prop :=
  b1 + b2 = b3 ∧
  b1 + b3 = b4 ∧
  b3 + b4 = b5 ∧
  b4 + b5 = b6 ∧
  b2 + b5 = b7

def non_overlapping_squares (b1 b2 b3 b4 b5 b6 b7 : ℕ) : Prop :=
  -- Placeholder for expressing that the squares are non-overlapping.
  true -- This is assumed as given in the problem.

theorem rectangle_perimeter (b1 b2 b3 b4 b5 b6 b7 : ℕ)
  (h1 : b1 = 1) (h2 : b2 = 2)
  (h_relations : satisfies_relations b1 b2 b3 b4 b5 b6 b7)
  (h_non_overlapping : non_overlapping_squares b1 b2 b3 b4 b5 b6 b7)
  (h_rel_prime : relatively_prime b6 b7) :
  2 * (b6 + b7) = 46 := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1122_112227


namespace NUMINAMATH_GPT_product_divisible_by_10_l1122_112288

noncomputable def probability_divisible_by_10 (n : ℕ) (h : n > 1) : ℝ :=
  1 - (8^n + 5^n - 4^n) / 9^n

theorem product_divisible_by_10 (n : ℕ) (h : n > 1) :
  probability_divisible_by_10 n h = 1 - (8^n + 5^n - 4^n)/(9^n) :=
by
  sorry

end NUMINAMATH_GPT_product_divisible_by_10_l1122_112288


namespace NUMINAMATH_GPT_compute_expression_l1122_112252

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1122_112252


namespace NUMINAMATH_GPT_robyn_packs_l1122_112269

-- Define the problem conditions
def total_packs : ℕ := 76
def lucy_packs : ℕ := 29

-- Define the goal to be proven
theorem robyn_packs : total_packs - lucy_packs = 47 := 
by
  sorry

end NUMINAMATH_GPT_robyn_packs_l1122_112269


namespace NUMINAMATH_GPT_mean_calculation_incorrect_l1122_112219

theorem mean_calculation_incorrect (a b c : ℝ) (h : a < b) (h1 : b < c) :
  let x := (a + b) / 2
  let y := (x + c) / 2
  y < (a + b + c) / 3 :=
by 
  let x := (a + b) / 2
  let y := (x + c) / 2
  sorry

end NUMINAMATH_GPT_mean_calculation_incorrect_l1122_112219


namespace NUMINAMATH_GPT_larger_solution_quadratic_l1122_112280

theorem larger_solution_quadratic :
  ∃ x : ℝ, x^2 - 13 * x + 30 = 0 ∧ (∀ y : ℝ, y^2 - 13 * y + 30 = 0 → y ≤ x) ∧ x = 10 := 
by
  sorry

end NUMINAMATH_GPT_larger_solution_quadratic_l1122_112280


namespace NUMINAMATH_GPT_fraction_of_ABCD_is_shaded_l1122_112220

noncomputable def squareIsDividedIntoTriangles : Type := sorry
noncomputable def areTrianglesIdentical (s : squareIsDividedIntoTriangles) : Prop := sorry
noncomputable def isFractionShadedCorrect : Prop := 
  ∃ (s : squareIsDividedIntoTriangles), 
  areTrianglesIdentical s ∧ 
  (7 / 16 : ℚ) = 7 / 16

theorem fraction_of_ABCD_is_shaded (s : squareIsDividedIntoTriangles) :
  areTrianglesIdentical s → (7 / 16 : ℚ) = 7 / 16 :=
sorry

end NUMINAMATH_GPT_fraction_of_ABCD_is_shaded_l1122_112220


namespace NUMINAMATH_GPT_exponential_quotient_l1122_112275

variable {x a b : ℝ}

theorem exponential_quotient (h1 : x^a = 3) (h2 : x^b = 5) : x^(a-b) = 3 / 5 :=
sorry

end NUMINAMATH_GPT_exponential_quotient_l1122_112275


namespace NUMINAMATH_GPT_polynomial_expansion_l1122_112232

variable (x : ℝ)

theorem polynomial_expansion : 
  (-2*x - 1) * (3*x - 2) = -6*x^2 + x + 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l1122_112232


namespace NUMINAMATH_GPT_number_of_saturday_sales_l1122_112204

def caricatures_sold_on_saturday (total_earnings weekend_earnings price_per_drawing sunday_sales : ℕ) : ℕ :=
  (total_earnings - (sunday_sales * price_per_drawing)) / price_per_drawing

theorem number_of_saturday_sales : caricatures_sold_on_saturday 800 800 20 16 = 24 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_saturday_sales_l1122_112204


namespace NUMINAMATH_GPT_trig_proof_1_trig_proof_2_l1122_112292

variables {α : ℝ}

-- Given condition
def tan_alpha (a : ℝ) := Real.tan a = -3

-- Proof problem statement
theorem trig_proof_1 (h : tan_alpha α) :
  (3 * Real.sin α - 3 * Real.cos α) / (6 * Real.cos α + Real.sin α) = -4 := sorry

theorem trig_proof_2 (h : tan_alpha α) :
  1 / (Real.sin α * Real.cos α + 1 + Real.cos (2 * α)) = -10 := sorry

end NUMINAMATH_GPT_trig_proof_1_trig_proof_2_l1122_112292


namespace NUMINAMATH_GPT_jenny_best_neighborhood_earnings_l1122_112261

theorem jenny_best_neighborhood_earnings :
  let cost_per_box := 2
  let neighborhood_a_homes := 10
  let neighborhood_a_boxes_per_home := 2
  let neighborhood_b_homes := 5
  let neighborhood_b_boxes_per_home := 5
  let earnings_a := neighborhood_a_homes * neighborhood_a_boxes_per_home * cost_per_box
  let earnings_b := neighborhood_b_homes * neighborhood_b_boxes_per_home * cost_per_box
  max earnings_a earnings_b = 50
:= by
  sorry

end NUMINAMATH_GPT_jenny_best_neighborhood_earnings_l1122_112261


namespace NUMINAMATH_GPT_cost_of_producing_one_component_l1122_112282

-- Define the conditions as constants
def shipping_cost_per_unit : ℕ := 5
def fixed_monthly_cost : ℕ := 16500
def components_per_month : ℕ := 150
def selling_price_per_component : ℕ := 195

-- Define the cost of producing one component as a variable
variable (C : ℕ)

/-- Prove that C must be less than or equal to 80 given the conditions -/
theorem cost_of_producing_one_component : 
  150 * C + 150 * shipping_cost_per_unit + fixed_monthly_cost ≤ 150 * selling_price_per_component → C ≤ 80 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_producing_one_component_l1122_112282


namespace NUMINAMATH_GPT_sum_of_ages_is_22_l1122_112262

noncomputable def Ashley_Age := 8
def Mary_Age (M : ℕ) := 7 * Ashley_Age = 4 * M

theorem sum_of_ages_is_22 (M : ℕ) (h : Mary_Age M):
  Ashley_Age + M = 22 :=
by
  -- skipping proof details
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_22_l1122_112262


namespace NUMINAMATH_GPT_find_dividend_l1122_112270

theorem find_dividend (k : ℕ) (quotient : ℕ) (dividend : ℕ) (h1 : k = 14) (h2 : quotient = 4) (h3 : dividend = quotient * k) : dividend = 56 :=
by
  sorry

end NUMINAMATH_GPT_find_dividend_l1122_112270


namespace NUMINAMATH_GPT_compare_a_b_c_l1122_112205

noncomputable
def a : ℝ := Real.exp 0.1 - 1

def b : ℝ := 0.1

noncomputable
def c : ℝ := Real.log 1.1

theorem compare_a_b_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_compare_a_b_c_l1122_112205


namespace NUMINAMATH_GPT_exchange_rate_l1122_112266

def jackPounds : ℕ := 42
def jackEuros : ℕ := 11
def jackYen : ℕ := 3000
def poundsPerYen : ℕ := 100
def totalYen : ℕ := 9400

theorem exchange_rate :
  ∃ (x : ℕ), 100 * jackPounds + 100 * jackEuros * x + jackYen = totalYen ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_exchange_rate_l1122_112266


namespace NUMINAMATH_GPT_carl_wins_in_4950_configurations_l1122_112246

noncomputable def num_distinct_configurations_at_Carl_win : ℕ :=
  sorry
  
theorem carl_wins_in_4950_configurations :
  num_distinct_configurations_at_Carl_win = 4950 :=
sorry

end NUMINAMATH_GPT_carl_wins_in_4950_configurations_l1122_112246


namespace NUMINAMATH_GPT_moles_of_Cl2_combined_l1122_112217

theorem moles_of_Cl2_combined (nCH4 : ℕ) (nCl2 : ℕ) (nHCl : ℕ) 
  (h1 : nCH4 = 3) 
  (h2 : nHCl = nCl2) 
  (h3 : nHCl ≤ nCH4) : 
  nCl2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_Cl2_combined_l1122_112217


namespace NUMINAMATH_GPT_find_base_and_digit_sum_l1122_112209

theorem find_base_and_digit_sum (n d : ℕ) (h1 : 4 * n^2 + 5 * n + d = 392) (h2 : 4 * n^2 + 5 * n + 7 = 740 + 7 * d) : n + d = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_base_and_digit_sum_l1122_112209


namespace NUMINAMATH_GPT_root_product_identity_l1122_112256

theorem root_product_identity (a b c : ℝ) (h1 : a * b * c = -8) (h2 : a * b + b * c + c * a = 20) (h3 : a + b + c = 15) :
    (1 + a) * (1 + b) * (1 + c) = 28 :=
by
  sorry

end NUMINAMATH_GPT_root_product_identity_l1122_112256


namespace NUMINAMATH_GPT_number_of_blocks_l1122_112253

theorem number_of_blocks (children_per_block : ℕ) (total_children : ℕ) (h1: children_per_block = 6) (h2: total_children = 54) : (total_children / children_per_block) = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_blocks_l1122_112253


namespace NUMINAMATH_GPT_cost_of_water_l1122_112294

theorem cost_of_water (total_cost sandwiches_cost : ℕ) (num_sandwiches sandwich_price water_price : ℕ) 
  (h1 : total_cost = 11) 
  (h2 : sandwiches_cost = num_sandwiches * sandwich_price) 
  (h3 : num_sandwiches = 3) 
  (h4 : sandwich_price = 3) 
  (h5 : total_cost = sandwiches_cost + water_price) : 
  water_price = 2 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_water_l1122_112294


namespace NUMINAMATH_GPT_sum_ad_eq_two_l1122_112259

theorem sum_ad_eq_two (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 7) 
  (h3 : c + d = 5) : 
  a + d = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_ad_eq_two_l1122_112259


namespace NUMINAMATH_GPT_root_expression_value_l1122_112271

variables (a b : ℝ)
noncomputable def quadratic_eq (a b : ℝ) : Prop := (a + b = 1 ∧ a * b = -1)

theorem root_expression_value (h : quadratic_eq a b) : 3 * a ^ 2 + 4 * b + (2 / a ^ 2) = 11 := sorry

end NUMINAMATH_GPT_root_expression_value_l1122_112271


namespace NUMINAMATH_GPT_part1_part2_l1122_112244

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then -3 * x + (1/2)^x - 1 else sorry -- Placeholder: function definition incomplete for x ≤ 0

def odd (f : ℝ → ℝ) :=
∀ x, f (-x) = - f x

def monotonic_decreasing (f : ℝ → ℝ) :=
∀ x y, x < y → f x > f y

axiom f_conditions :
  monotonic_decreasing f ∧
  odd f ∧
  (∀ x, x > 0 → f x = -3 * x + (1/2)^x - 1)

theorem part1 : f (-1) = 3.5 :=
by
  sorry

theorem part2 (t : ℝ) (k : ℝ) :
  (∀ t, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) ↔ k < -1/3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1122_112244


namespace NUMINAMATH_GPT_range_of_quadratic_expression_l1122_112283

theorem range_of_quadratic_expression :
  (∃ x : ℝ, y = 2 * x^2 - 4 * x + 12) ↔ (y ≥ 10) :=
by
  sorry

end NUMINAMATH_GPT_range_of_quadratic_expression_l1122_112283


namespace NUMINAMATH_GPT_segment_length_reflection_l1122_112299

theorem segment_length_reflection (F : ℝ × ℝ) (F' : ℝ × ℝ)
  (hF : F = (-4, -2)) (hF' : F' = (4, -2)) :
  dist F F' = 8 :=
by
  sorry

end NUMINAMATH_GPT_segment_length_reflection_l1122_112299


namespace NUMINAMATH_GPT_arithmetic_expression_eval_l1122_112240

theorem arithmetic_expression_eval : 3 + (12 / 3 - 1) ^ 2 = 12 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_eval_l1122_112240


namespace NUMINAMATH_GPT_rope_subdivision_length_l1122_112238

theorem rope_subdivision_length 
  (initial_length : ℕ) 
  (num_parts : ℕ) 
  (num_subdivided_parts : ℕ) 
  (final_subdivision_factor : ℕ) 
  (initial_length_eq : initial_length = 200) 
  (num_parts_eq : num_parts = 4) 
  (num_subdivided_parts_eq : num_subdivided_parts = num_parts / 2) 
  (final_subdivision_factor_eq : final_subdivision_factor = 2) :
  initial_length / num_parts / final_subdivision_factor = 25 := 
by 
  sorry

end NUMINAMATH_GPT_rope_subdivision_length_l1122_112238


namespace NUMINAMATH_GPT_find_D_l1122_112207

theorem find_D (A B C D : ℕ) (h₁ : A + A = 6) (h₂ : B - A = 4) (h₃ : C + B = 9) (h₄ : D - C = 7) : D = 9 :=
sorry

end NUMINAMATH_GPT_find_D_l1122_112207


namespace NUMINAMATH_GPT_total_wheels_correct_l1122_112276

-- Define the initial state of the garage
def initial_bicycles := 20
def initial_cars := 10
def initial_motorcycles := 5
def initial_tricycles := 3
def initial_quads := 2

-- Define the changes in the next hour
def bicycles_leaving := 7
def cars_arriving := 4
def motorcycles_arriving := 3
def motorcycles_leaving := 2

-- Define the damaged vehicles
def damaged_bicycles := 5  -- each missing 1 wheel
def damaged_cars := 2      -- each missing 1 wheel
def damaged_motorcycle := 1 -- missing 2 wheels

-- Define the number of wheels per type of vehicle
def bicycle_wheels := 2
def car_wheels := 4
def motorcycle_wheels := 2
def tricycle_wheels := 3
def quad_wheels := 4

-- Calculate the state of vehicles at the end of the hour
def final_bicycles := initial_bicycles - bicycles_leaving
def final_cars := initial_cars + cars_arriving
def final_motorcycles := initial_motorcycles + motorcycles_arriving - motorcycles_leaving

-- Calculate the total wheels in the garage at the end of the hour
def total_wheels : Nat := 
  (final_bicycles - damaged_bicycles) * bicycle_wheels + damaged_bicycles +
  (final_cars - damaged_cars) * car_wheels + damaged_cars * 3 +
  (final_motorcycles - damaged_motorcycle) * motorcycle_wheels +
  initial_tricycles * tricycle_wheels +
  initial_quads * quad_wheels

-- The goal is to prove that the total number of wheels in the garage is 102 at the end of the hour
theorem total_wheels_correct : total_wheels = 102 := 
  by
    sorry

end NUMINAMATH_GPT_total_wheels_correct_l1122_112276


namespace NUMINAMATH_GPT_calculate_expression_l1122_112216

theorem calculate_expression : 56.8 * 35.7 + 56.8 * 28.5 + 64.2 * 43.2 = 6420 := 
by sorry

end NUMINAMATH_GPT_calculate_expression_l1122_112216


namespace NUMINAMATH_GPT_no_contradiction_to_thermodynamics_l1122_112257

variables (T_handle T_environment : ℝ) (cold_water : Prop)
noncomputable def increased_grip_increases_heat_transfer (A1 A2 : ℝ) (k : ℝ) (dT dx : ℝ) : Prop :=
  A2 > A1 ∧ k * (A2 - A1) * (dT / dx) > 0

theorem no_contradiction_to_thermodynamics (T_handle T_environment : ℝ) (cold_water : Prop) :
  T_handle > T_environment ∧ cold_water →
  ∃ A1 A2 k dT dx, T_handle > T_environment ∧ k > 0 ∧ dT > 0 ∧ dx > 0 → increased_grip_increases_heat_transfer A1 A2 k dT dx :=
sorry

end NUMINAMATH_GPT_no_contradiction_to_thermodynamics_l1122_112257


namespace NUMINAMATH_GPT_quadratic_roots_eqn_l1122_112236

theorem quadratic_roots_eqn (b c : ℝ) (x1 x2 : ℝ) (h1 : x1 = -2) (h2 : x2 = 3) (h3 : b = -(x1 + x2)) (h4 : c = x1 * x2) : 
    (x^2 + b * x + c = 0) ↔ (x^2 - x - 6 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_eqn_l1122_112236


namespace NUMINAMATH_GPT_distance_is_660_km_l1122_112212

def distance_between_cities (x y : ℝ) : ℝ :=
  3.3 * (x + y)

def train_A_dep_earlier (x y : ℝ) : Prop :=
  3.4 * (x + y) = 3.3 * (x + y) + 14

def train_B_dep_earlier (x y : ℝ) : Prop :=
  3.6 * (x + y) = 3.3 * (x + y) + 9

theorem distance_is_660_km (x y : ℝ) (hx : train_A_dep_earlier x y) (hy : train_B_dep_earlier x y) :
    distance_between_cities x y = 660 :=
sorry

end NUMINAMATH_GPT_distance_is_660_km_l1122_112212


namespace NUMINAMATH_GPT_exists_another_nice_triple_l1122_112208

noncomputable def is_nice_triple (a b c : ℕ) : Prop :=
  (a ≤ b ∧ b ≤ c ∧ (b - a) = (c - b)) ∧
  (Nat.gcd b a = 1 ∧ Nat.gcd b c = 1) ∧ 
  (∃ k, a * b * c = k^2)

theorem exists_another_nice_triple (a b c : ℕ) 
  (h : is_nice_triple a b c) : ∃ a' b' c', 
  (is_nice_triple a' b' c') ∧ 
  (a' = a ∨ a' = b ∨ a' = c ∨ 
   b' = a ∨ b' = b ∨ b' = c ∨ 
   c' = a ∨ c' = b ∨ c' = c) :=
by sorry

end NUMINAMATH_GPT_exists_another_nice_triple_l1122_112208


namespace NUMINAMATH_GPT_candy_problem_l1122_112287

theorem candy_problem (n : ℕ) (h : n ∈ [2, 5, 9, 11, 14]) : ¬(23 - n) % 3 ≠ 0 → n = 9 := by
  sorry

end NUMINAMATH_GPT_candy_problem_l1122_112287


namespace NUMINAMATH_GPT_camels_horses_oxen_elephants_l1122_112222

theorem camels_horses_oxen_elephants :
  ∀ (C H O E : ℝ),
  10 * C = 24 * H →
  H = 4 * O →
  6 * O = 4 * E →
  10 * E = 170000 →
  C = 4184.615384615385 →
  (4 * O) / H = 1 :=
by
  intros C H O E h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_camels_horses_oxen_elephants_l1122_112222


namespace NUMINAMATH_GPT_greatest_N_consecutive_sum_50_l1122_112297

theorem greatest_N_consecutive_sum_50 :
  ∃ N a : ℤ, (N > 0) ∧ (N * (2 * a + N - 1) = 100) ∧ (N = 100) :=
by
  sorry

end NUMINAMATH_GPT_greatest_N_consecutive_sum_50_l1122_112297


namespace NUMINAMATH_GPT_garden_length_l1122_112273

theorem garden_length (P B : ℕ) (h₁ : P = 600) (h₂ : B = 95) : (∃ L : ℕ, 2 * (L + B) = P ∧ L = 205) :=
by
  sorry

end NUMINAMATH_GPT_garden_length_l1122_112273


namespace NUMINAMATH_GPT_matches_for_ladder_l1122_112210

theorem matches_for_ladder (n : ℕ) (h : n = 25) : 
  (6 + 6 * (n - 1) = 150) :=
by
  sorry

end NUMINAMATH_GPT_matches_for_ladder_l1122_112210


namespace NUMINAMATH_GPT_weight_of_second_square_l1122_112233

noncomputable def weight_of_square (side_length : ℝ) (density : ℝ) : ℝ :=
  side_length^2 * density

theorem weight_of_second_square :
  let s1 := 4
  let m1 := 20
  let s2 := 7
  let density := m1 / (s1 ^ 2)
  ∃ (m2 : ℝ), m2 = 61.25 :=
by
  have s1 := 4
  have m1 := 20
  have s2 := 7
  let density := m1 / (s1 ^ 2)
  have m2 := weight_of_square s2 density
  use m2
  sorry

end NUMINAMATH_GPT_weight_of_second_square_l1122_112233


namespace NUMINAMATH_GPT_bacteria_growth_l1122_112255

-- Define the original and current number of bacteria
def original_bacteria := 600
def current_bacteria := 8917

-- Define the increase in bacteria count
def additional_bacteria := 8317

-- Prove the statement
theorem bacteria_growth : current_bacteria - original_bacteria = additional_bacteria :=
by {
    -- Lean will require the proof here, so we use sorry for now 
    sorry
}

end NUMINAMATH_GPT_bacteria_growth_l1122_112255


namespace NUMINAMATH_GPT_average_shifted_samples_l1122_112274

variables (x1 x2 x3 x4 : ℝ)

theorem average_shifted_samples (h : (x1 + x2 + x3 + x4) / 4 = 2) :
  ((x1 + 3) + (x2 + 3) + (x3 + 3) + (x4 + 3)) / 4 = 5 :=
by
  sorry

end NUMINAMATH_GPT_average_shifted_samples_l1122_112274


namespace NUMINAMATH_GPT_range_of_a_l1122_112296

open Real

theorem range_of_a (a : ℝ) :
  (0 < a ∧ a < 6) ∨ (a ≥ 5 ∨ a ≤ 1) ∧ ¬((0 < a ∧ a < 6) ∧ (a ≥ 5 ∨ a ≤ 1)) ↔ 
  (a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5)) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1122_112296


namespace NUMINAMATH_GPT_largest_n_exists_ints_l1122_112231

theorem largest_n_exists_ints (n : ℤ) :
  (∃ x y z : ℤ, n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 12) →
  n ≤ 10 :=
sorry

end NUMINAMATH_GPT_largest_n_exists_ints_l1122_112231


namespace NUMINAMATH_GPT_total_cost_of_fruit_l1122_112213

theorem total_cost_of_fruit (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 58) 
  (h2 : 3 * x + 2 * y = 72) : 
  3 * x + 3 * y = 78 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_of_fruit_l1122_112213


namespace NUMINAMATH_GPT_smallest_c_inequality_l1122_112206

theorem smallest_c_inequality (x : ℕ → ℝ) (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 = 10) :
  ∃ c : ℝ, (∀ x : ℕ → ℝ, x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 = 10 →
    |x 0| + |x 1| + |x 2| + |x 3| + |x 4| + |x 5| + |x 6| + |x 7| + |x 8| ≥ c * |x 4|) ∧ c = 9 := 
by
  sorry

end NUMINAMATH_GPT_smallest_c_inequality_l1122_112206


namespace NUMINAMATH_GPT_total_heads_l1122_112290

def total_legs : ℕ := 45
def num_cats : ℕ := 7
def legs_per_cat : ℕ := 4
def captain_legs : ℕ := 1
def legs_humans := total_legs - (num_cats * legs_per_cat) - captain_legs
def num_humans := legs_humans / 2

theorem total_heads : (num_cats + (num_humans + 1)) = 15 := by
  sorry

end NUMINAMATH_GPT_total_heads_l1122_112290


namespace NUMINAMATH_GPT_matrices_commute_l1122_112230

variable {n : Nat}
variable (A B X : Matrix (Fin n) (Fin n) ℝ)

theorem matrices_commute (h : A * X * B + A + B = 0) : A * X * B = B * X * A :=
by
  sorry

end NUMINAMATH_GPT_matrices_commute_l1122_112230


namespace NUMINAMATH_GPT_ordered_triple_exists_l1122_112293

theorem ordered_triple_exists (a b c : ℝ) (h₁ : 4 < a) (h₂ : 4 < b) (h₃ : 4 < c)
  (h₄ : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  (a, b, c) = (12, 10, 8) :=
sorry

end NUMINAMATH_GPT_ordered_triple_exists_l1122_112293


namespace NUMINAMATH_GPT_x_divisible_by_5_l1122_112254

theorem x_divisible_by_5
  (x y : ℕ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_gt_1 : 1 < x)
  (h_eq : 2 * x^2 - 1 = y^15) : x % 5 = 0 :=
sorry

end NUMINAMATH_GPT_x_divisible_by_5_l1122_112254


namespace NUMINAMATH_GPT_domain_of_composite_function_l1122_112234

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → -1 ≤ x + 1) →
  (∀ x, -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 3 → -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3 → 0 ≤ x ∧ x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_composite_function_l1122_112234


namespace NUMINAMATH_GPT_motorcycle_travel_distance_l1122_112286

noncomputable def motorcycle_distance : ℝ :=
  let t : ℝ := 1 / 2  -- time in hours (30 minutes)
  let v_bus : ℝ := 90  -- speed of the bus in km/h
  let v_motorcycle : ℝ := (2 / 3) * v_bus  -- speed of the motorcycle in km/h
  v_motorcycle * t  -- calculates the distance traveled by the motorcycle in km

theorem motorcycle_travel_distance :
  motorcycle_distance = 30 := by
  sorry

end NUMINAMATH_GPT_motorcycle_travel_distance_l1122_112286


namespace NUMINAMATH_GPT_geometric_seq_an_minus_2_l1122_112228

-- Definitions of conditions based on given problem
def seq_a : ℕ → ℝ := sorry -- The sequence {a_n}
def sum_s : ℕ → ℝ := sorry -- The sum of the first n terms {s_n}

axiom cond1 (n : ℕ) (hn : n > 0) : seq_a (n + 1) ≠ seq_a n
axiom cond2 (n : ℕ) (hn : n > 0) : sum_s n + seq_a n = 2 * n

-- Theorem statement
theorem geometric_seq_an_minus_2 (n : ℕ) (hn : n > 0) : 
  ∃ r : ℝ, ∀ k : ℕ, seq_a (k + 1) - 2 = r * (seq_a k - 2) := 
sorry

end NUMINAMATH_GPT_geometric_seq_an_minus_2_l1122_112228


namespace NUMINAMATH_GPT_english_score_is_96_l1122_112239

variable (Science_score : ℕ) (Social_studies_score : ℕ) (English_score : ℕ)

/-- Jimin's social studies score is 6 points higher than his science score -/
def social_studies_score_condition := Social_studies_score = Science_score + 6

/-- The science score is 87 -/
def science_score_condition := Science_score = 87

/-- The average score for science, social studies, and English is 92 -/
def average_score_condition := (Science_score + Social_studies_score + English_score) / 3 = 92

theorem english_score_is_96
  (h1 : social_studies_score_condition Science_score Social_studies_score)
  (h2 : science_score_condition Science_score)
  (h3 : average_score_condition Science_score Social_studies_score English_score) :
  English_score = 96 :=
  by
    sorry

end NUMINAMATH_GPT_english_score_is_96_l1122_112239


namespace NUMINAMATH_GPT_fraction_of_single_female_students_l1122_112218

variables (total_students : ℕ) (male_students married_students married_male_students female_students single_female_students : ℕ)

-- Given conditions
def condition1 : male_students = (7 * total_students) / 10 := sorry
def condition2 : married_students = (3 * total_students) / 10 := sorry
def condition3 : married_male_students = male_students / 7 := sorry

-- Derived conditions
def condition4 : female_students = total_students - male_students := sorry
def condition5 : married_female_students = married_students - married_male_students := sorry
def condition6 : single_female_students = female_students - married_female_students := sorry

-- The proof goal
theorem fraction_of_single_female_students 
  (h1 : male_students = (7 * total_students) / 10)
  (h2 : married_students = (3 * total_students) / 10)
  (h3 : married_male_students = male_students / 7)
  (h4 : female_students = total_students - male_students)
  (h5 : married_female_students = married_students - married_male_students)
  (h6 : single_female_students = female_students - married_female_students) :
  (single_female_students : ℚ) / (female_students : ℚ) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_fraction_of_single_female_students_l1122_112218


namespace NUMINAMATH_GPT_all_ones_l1122_112263

theorem all_ones (k : ℕ) (h₁ : k ≥ 2) (n : ℕ → ℕ) (h₂ : ∀ i, 1 ≤ i → i < k → n (i + 1) ∣ (2 ^ n i - 1))
(h₃ : n 1 ∣ (2 ^ n k - 1)) : (∀ i, 1 ≤ i → i ≤ k → n i = 1) :=
by
  sorry

end NUMINAMATH_GPT_all_ones_l1122_112263


namespace NUMINAMATH_GPT_simplify_fraction_result_l1122_112264

theorem simplify_fraction_result :
  (144: ℝ) / 1296 * 72 = 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_result_l1122_112264


namespace NUMINAMATH_GPT_car_average_speed_l1122_112201

theorem car_average_speed 
  (d1 d2 d3 d5 d6 d7 d8 : ℝ) 
  (t_total : ℝ) 
  (avg_speed : ℝ)
  (h1 : d1 = 90)
  (h2 : d2 = 50)
  (h3 : d3 = 70)
  (h5 : d5 = 80)
  (h6 : d6 = 60)
  (h7 : d7 = -40)
  (h8 : d8 = -55)
  (h_t_total : t_total = 8)
  (h_avg_speed : avg_speed = (d1 + d2 + d3 + d5 + d6 + d7 + d8) / t_total) :
  avg_speed = 31.875 := 
by sorry

end NUMINAMATH_GPT_car_average_speed_l1122_112201


namespace NUMINAMATH_GPT_villages_population_equal_l1122_112272

def population_x (initial_population rate_decrease : Int) (n : Int) := initial_population - rate_decrease * n
def population_y (initial_population rate_increase : Int) (n : Int) := initial_population + rate_increase * n

theorem villages_population_equal
    (initial_population_x : Int) (rate_decrease_x : Int)
    (initial_population_y : Int) (rate_increase_y : Int)
    (h₁ : initial_population_x = 76000) (h₂ : rate_decrease_x = 1200)
    (h₃ : initial_population_y = 42000) (h₄ : rate_increase_y = 800) :
    ∃ n : Int, population_x initial_population_x rate_decrease_x n = population_y initial_population_y rate_increase_y n ∧ n = 17 :=
by
    sorry

end NUMINAMATH_GPT_villages_population_equal_l1122_112272


namespace NUMINAMATH_GPT_meadow_total_revenue_correct_l1122_112258

-- Define the given quantities and conditions as Lean definitions
def total_diapers : ℕ := 192000
def price_per_diaper : ℝ := 4.0
def bundle_discount : ℝ := 0.05
def purchase_discount : ℝ := 0.05
def tax_rate : ℝ := 0.10

-- Define a function that calculates the revenue from selling all the diapers
def calculate_revenue (total_diapers : ℕ) (price_per_diaper : ℝ) (bundle_discount : ℝ) 
    (purchase_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let gross_revenue := total_diapers * price_per_diaper
  let bundle_discounted_revenue := gross_revenue * (1 - bundle_discount)
  let purchase_discounted_revenue := bundle_discounted_revenue * (1 - purchase_discount)
  let taxed_revenue := purchase_discounted_revenue * (1 + tax_rate)
  taxed_revenue

-- The main theorem to prove that the calculated revenue matches the expected value
theorem meadow_total_revenue_correct : 
  calculate_revenue total_diapers price_per_diaper bundle_discount purchase_discount tax_rate = 762432 := 
by
  sorry

end NUMINAMATH_GPT_meadow_total_revenue_correct_l1122_112258


namespace NUMINAMATH_GPT_principal_made_mistake_l1122_112215

-- Definitions based on given conditions
def students_per_class (x : ℤ) : Prop := x > 0
def total_students (x : ℤ) : ℤ := 2 * x
def non_failing_grades (y : ℤ) : ℤ := y
def failing_grades (y : ℤ) : ℤ := y + 11
def total_grades (x y : ℤ) : Prop := total_students x = non_failing_grades y + failing_grades y

-- Proposition stating the principal made a mistake
theorem principal_made_mistake (x y : ℤ) (hx : students_per_class x) : ¬ total_grades x y :=
by
  -- Assume the proof for the hypothesis is required here
  sorry

end NUMINAMATH_GPT_principal_made_mistake_l1122_112215


namespace NUMINAMATH_GPT_find_z_plus_one_over_y_l1122_112225

theorem find_z_plus_one_over_y 
  (x y z : ℝ) 
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : 0 < z)
  (h4 : x * y * z = 1)
  (h5 : x + 1/z = 4)
  (h6 : y + 1/x = 20) :
  z + 1/y = 26 / 79 :=
by
  sorry

end NUMINAMATH_GPT_find_z_plus_one_over_y_l1122_112225


namespace NUMINAMATH_GPT_marble_draw_l1122_112203

/-- A container holds 30 red marbles, 25 green marbles, 23 yellow marbles,
15 blue marbles, 10 white marbles, and 7 black marbles. Prove that the
minimum number of marbles that must be drawn from the container without
replacement to ensure that at least 10 marbles of a single color are drawn
is 53. -/
theorem marble_draw (R G Y B W Bl : ℕ) (hR : R = 30) (hG : G = 25)
                               (hY : Y = 23) (hB : B = 15) (hW : W = 10)
                               (hBl : Bl = 7) : 
  ∃ (n : ℕ), n = 53 ∧ (∀ (x : ℕ), x ≠ n → 
  (x ≤ R → x ≤ G → x ≤ Y → x ≤ B → x ≤ W → x ≤ Bl → x < 10)) := 
by
  sorry

end NUMINAMATH_GPT_marble_draw_l1122_112203


namespace NUMINAMATH_GPT_time_to_eat_quarter_l1122_112285

noncomputable def total_nuts : ℕ := sorry

def rate_first_crow (N : ℕ) := N / 40
def rate_second_crow (N : ℕ) := N / 36

theorem time_to_eat_quarter (N : ℕ) (T : ℝ) :
  (rate_first_crow N + rate_second_crow N) * T = (1 / 4 : ℝ) * N → 
  T = (90 / 19 : ℝ) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_time_to_eat_quarter_l1122_112285


namespace NUMINAMATH_GPT_tiling_tetromino_divisibility_l1122_112214

theorem tiling_tetromino_divisibility (n : ℕ) : 
  (∃ (t : ℕ), n = 4 * t) ↔ (∃ (k : ℕ), n * n = 4 * k) :=
by
  sorry

end NUMINAMATH_GPT_tiling_tetromino_divisibility_l1122_112214


namespace NUMINAMATH_GPT_winningTicketProbability_l1122_112224

-- Given conditions
def sharpBallProbability : ℚ := 1 / 30
def prizeBallsProbability : ℚ := 1 / (Nat.descFactorial 50 6)

-- The target probability that we are supposed to prove
def targetWinningProbability : ℚ := 1 / 476721000

-- Main theorem stating the required probability calculation
theorem winningTicketProbability :
  sharpBallProbability * prizeBallsProbability = targetWinningProbability :=
  sorry

end NUMINAMATH_GPT_winningTicketProbability_l1122_112224


namespace NUMINAMATH_GPT_Winnie_lollipops_remain_l1122_112289

theorem Winnie_lollipops_remain :
  let cherry_lollipops := 45
  let wintergreen_lollipops := 116
  let grape_lollipops := 4
  let shrimp_cocktail_lollipops := 229
  let total_lollipops := cherry_lollipops + wintergreen_lollipops + grape_lollipops + shrimp_cocktail_lollipops
  let friends := 11
  total_lollipops % friends = 9 :=
by
  sorry

end NUMINAMATH_GPT_Winnie_lollipops_remain_l1122_112289


namespace NUMINAMATH_GPT_nate_total_run_l1122_112242

def field_length := 168
def initial_run := 4 * field_length
def additional_run := 500
def total_run := initial_run + additional_run

theorem nate_total_run : total_run = 1172 := by
  sorry

end NUMINAMATH_GPT_nate_total_run_l1122_112242


namespace NUMINAMATH_GPT_ant_population_percentage_l1122_112237

theorem ant_population_percentage (R : ℝ) 
  (h1 : 0.45 * R = 46.75) 
  (h2 : R * 0.55 = 46.75) : 
  R = 0.85 := 
by 
  sorry

end NUMINAMATH_GPT_ant_population_percentage_l1122_112237
