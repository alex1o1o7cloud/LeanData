import Mathlib

namespace NUMINAMATH_GPT_zero_point_interval_l2281_228199

noncomputable def f (x : ℝ) : ℝ := Real.pi * x + Real.log x / Real.log 2

theorem zero_point_interval : 
  f (1/4) < 0 ∧ f (1/2) > 0 → ∃ x : ℝ, 1/4 ≤ x ∧ x ≤ 1/2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_point_interval_l2281_228199


namespace NUMINAMATH_GPT_min_orders_to_minimize_spent_l2281_228162

-- Definitions for the given conditions
def original_price (n p : ℕ) : ℕ := n * p
def discounted_price (T : ℕ) : ℕ := (3 * T) / 5  -- Equivalent to 0.6 * T, using integer math

-- Define the conditions
theorem min_orders_to_minimize_spent 
  (n p : ℕ)
  (h1 : n = 42)
  (h2 : p = 48)
  : ∃ m : ℕ, m = 3 :=
by 
  sorry

end NUMINAMATH_GPT_min_orders_to_minimize_spent_l2281_228162


namespace NUMINAMATH_GPT_smallest_consecutive_odd_sum_l2281_228181

theorem smallest_consecutive_odd_sum (a b c d e : ℤ)
    (h1 : b = a + 2)
    (h2 : c = a + 4)
    (h3 : d = a + 6)
    (h4 : e = a + 8)
    (h5 : a + b + c + d + e = 375) : a = 71 :=
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_smallest_consecutive_odd_sum_l2281_228181


namespace NUMINAMATH_GPT_functional_equation_solution_l2281_228179

noncomputable def f (x : ℝ) (c : ℝ) : ℝ :=
  (c * x - c^2) / (1 + c)

def g (x : ℝ) (c : ℝ) : ℝ :=
  c * x - c^2

theorem functional_equation_solution (f g : ℝ → ℝ) (c : ℝ) (h : c ≠ -1) :
  (∀ x y : ℝ, f (x + g y) = x * f y - y * f x + g x) ∧
  (∀ x, f x = (c * x - c^2) / (1 + c)) ∧
  (∀ x, g x = c * x - c^2) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l2281_228179


namespace NUMINAMATH_GPT_number_of_shelves_l2281_228174

theorem number_of_shelves (a d S : ℕ) (h1 : a = 3) (h2 : d = 3) (h3 : S = 225) : 
  ∃ n : ℕ, (S = n * (2 * a + (n - 1) * d) / 2) ∧ (n = 15) := 
by {
  sorry
}

end NUMINAMATH_GPT_number_of_shelves_l2281_228174


namespace NUMINAMATH_GPT_purity_of_alloy_l2281_228102

theorem purity_of_alloy (w1 w2 : ℝ) (p1 p2 : ℝ) (h_w1 : w1 = 180) (h_p1 : p1 = 920) (h_w2 : w2 = 100) (h_p2 : p2 = 752) : 
  let a := w1 * (p1 / 1000) + w2 * (p2 / 1000)
  let b := w1 + w2
  let p_result := (a / b) * 1000
  p_result = 860 :=
by
  sorry

end NUMINAMATH_GPT_purity_of_alloy_l2281_228102


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l2281_228140

theorem smallest_positive_integer_n (n : ℕ) (cube : Finset (Fin 8)) :
    (∀ (coloring : Finset (Fin 8)), 
      coloring.card = n → 
      ∃ (v : Fin 8), 
        (∀ (adj : Finset (Fin 8)), adj.card = 3 → adj ⊆ cube → v ∈ adj → adj ⊆ coloring)) 
    ↔ n = 5 := 
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l2281_228140


namespace NUMINAMATH_GPT_ratio_of_sides_l2281_228117

open Real

variable (s y x : ℝ)

-- Assuming the rectangles and squares conditions
def condition1 := 4 * (x * y) + s * s = 9 * (s * s)
def condition2 := x = 2 * s
def condition3 := y = s

-- Stating the theorem
theorem ratio_of_sides (h1 : condition1 s y x) (h2 : condition2 s x) (h3 : condition3 s y) :
  x / y = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_sides_l2281_228117


namespace NUMINAMATH_GPT_albert_needs_more_money_l2281_228173

def cost_of_paintbrush : ℝ := 1.50
def cost_of_paints : ℝ := 4.35
def cost_of_easel : ℝ := 12.65
def amount_already_has : ℝ := 6.50

theorem albert_needs_more_money : 
  (cost_of_paintbrush + cost_of_paints + cost_of_easel) - amount_already_has = 12.00 := 
by
  sorry

end NUMINAMATH_GPT_albert_needs_more_money_l2281_228173


namespace NUMINAMATH_GPT_min_ab_eq_4_l2281_228155

theorem min_ab_eq_4 (a b : ℝ) (h : 4 / a + 1 / b = Real.sqrt (a * b)) : a * b ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_ab_eq_4_l2281_228155


namespace NUMINAMATH_GPT_x_intercept_of_line_is_six_l2281_228135

theorem x_intercept_of_line_is_six : ∃ x : ℝ, (∃ y : ℝ, y = 0) ∧ (2*x - 4*y = 12) ∧ x = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_x_intercept_of_line_is_six_l2281_228135


namespace NUMINAMATH_GPT_marla_errand_total_time_l2281_228132

theorem marla_errand_total_time :
  let d1 := 20 -- Driving to son's school
  let b := 30  -- Taking a bus to the grocery store
  let s := 15  -- Shopping at the grocery store
  let w := 10  -- Walking to the gas station
  let g := 5   -- Filling up gas
  let r := 25  -- Riding a bicycle to the school
  let p := 70  -- Attending parent-teacher night
  let c := 30  -- Catching up with a friend at a coffee shop
  let sub := 40-- Taking the subway home
  let d2 := 20 -- Driving home
  d1 + b + s + w + g + r + p + c + sub + d2 = 265 := by
  sorry

end NUMINAMATH_GPT_marla_errand_total_time_l2281_228132


namespace NUMINAMATH_GPT_desired_line_equation_exists_l2281_228128

theorem desired_line_equation_exists :
  ∃ (a b c : ℝ), (a * 0 + b * 1 + c = 0) ∧
  (x - 3 * y + 10 = 0) ∧
  (2 * x + y - 8 = 0) ∧
  (a * x + b * y + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_desired_line_equation_exists_l2281_228128


namespace NUMINAMATH_GPT_num_distinct_log_values_l2281_228121

-- Defining the set of numbers
def number_set : Set ℕ := {1, 2, 3, 4, 6, 9}

-- Define a function to count distinct logarithmic values
noncomputable def distinct_log_values (s : Set ℕ) : ℕ := 
  -- skipped, assume the implementation is done correctly
  sorry 

theorem num_distinct_log_values : distinct_log_values number_set = 17 :=
by
  sorry

end NUMINAMATH_GPT_num_distinct_log_values_l2281_228121


namespace NUMINAMATH_GPT_distinct_real_roots_range_l2281_228131

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem distinct_real_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 2 * x + a = 0) ∧ (y^2 - 2 * y + a = 0))
  ↔ a < 1 := 
by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_range_l2281_228131


namespace NUMINAMATH_GPT_fifth_friend_paid_40_l2281_228124

-- Defining the conditions given in the problem
variables {a b c d e : ℝ}
variables (h1 : a = (1/3) * (b + c + d + e))
variables (h2 : b = (1/4) * (a + c + d + e))
variables (h3 : c = (1/5) * (a + b + d + e))
variables (h4 : d = (1/6) * (a + b + c + e))
variables (h5 : a + b + c + d + e = 120)

-- Proving that the amount paid by the fifth friend is $40
theorem fifth_friend_paid_40 : e = 40 :=
by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_fifth_friend_paid_40_l2281_228124


namespace NUMINAMATH_GPT_negation_proposition_l2281_228195

theorem negation_proposition :
  (∀ x : ℝ, |x - 2| + |x - 4| > 3) = ¬(∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) :=
  by sorry

end NUMINAMATH_GPT_negation_proposition_l2281_228195


namespace NUMINAMATH_GPT_sin_cos_product_l2281_228100

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin x * Real.cos x = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_sin_cos_product_l2281_228100


namespace NUMINAMATH_GPT_find_original_six_digit_number_l2281_228143

theorem find_original_six_digit_number (N x y : ℕ) (h1 : N = 10 * x + y) (h2 : N - x = 654321) (h3 : 0 ≤ y ∧ y ≤ 9) :
  N = 727023 :=
sorry

end NUMINAMATH_GPT_find_original_six_digit_number_l2281_228143


namespace NUMINAMATH_GPT_oil_depth_solution_l2281_228120

theorem oil_depth_solution
  (length diameter surface_area : ℝ)
  (h : ℝ)
  (h_length : length = 12)
  (h_diameter : diameter = 4)
  (h_surface_area : surface_area = 24)
  (r : ℝ := diameter / 2)
  (c : ℝ := surface_area / length) :
  (h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_oil_depth_solution_l2281_228120


namespace NUMINAMATH_GPT_point_to_real_l2281_228145

-- Condition: Real numbers correspond one-to-one with points on the number line.
def real_numbers_correspond (x : ℝ) : Prop :=
  ∃ (p : ℝ), p = x

-- Condition: Any real number can be represented by a point on the number line.
def represent_real_by_point (x : ℝ) : Prop :=
  real_numbers_correspond x

-- Condition: Conversely, any point on the number line represents a real number.
def point_represents_real (p : ℝ) : Prop :=
  ∃ (x : ℝ), x = p

-- Condition: The number represented by any point on the number line is either a rational number or an irrational number.
def rational_or_irrational (p : ℝ) : Prop :=
  (∃ q : ℚ, (q : ℝ) = p) ∨ (¬∃ q : ℚ, (q : ℝ) = p)

theorem point_to_real (p : ℝ) : represent_real_by_point p ∧ point_represents_real p ∧ rational_or_irrational p → real_numbers_correspond p :=
by sorry

end NUMINAMATH_GPT_point_to_real_l2281_228145


namespace NUMINAMATH_GPT_part_a_part_b_l2281_228137

/-- Two equally skilled chess players with p = 0.5, q = 0.5. -/
def p : ℝ := 0.5
def q : ℝ := 0.5

-- Definition for binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Binomial distribution
def P (n k : ℕ) : ℝ := (binomial_coeff n k) * (p^k) * (q^(n-k))

/-- Prove that the probability of winning one out of two games is greater than the probability of winning two out of four games -/
theorem part_a : (P 2 1) > (P 4 2) := sorry

/-- Prove that the probability of winning at least two out of four games is greater than the probability of winning at least three out of five games -/
theorem part_b : (P 4 2 + P 4 3 + P 4 4) > (P 5 3 + P 5 4 + P 5 5) := sorry

end NUMINAMATH_GPT_part_a_part_b_l2281_228137


namespace NUMINAMATH_GPT_geometric_sequence_y_value_l2281_228114

theorem geometric_sequence_y_value (q : ℝ) 
  (h1 : 2 * q^4 = 18) 
  : 2 * (q^2) = 6 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_y_value_l2281_228114


namespace NUMINAMATH_GPT_gwen_remaining_money_l2281_228196

theorem gwen_remaining_money:
  ∀ (Gwen_received Gwen_spent Gwen_remaining: ℕ),
    Gwen_received = 5 →
    Gwen_spent = 3 →
    Gwen_remaining = Gwen_received - Gwen_spent →
    Gwen_remaining = 2 :=
by
  intros Gwen_received Gwen_spent Gwen_remaining h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end NUMINAMATH_GPT_gwen_remaining_money_l2281_228196


namespace NUMINAMATH_GPT_remainder_of_c_plus_d_l2281_228167

theorem remainder_of_c_plus_d (c d : ℕ) (k l : ℕ) 
  (hc : c = 120 * k + 114) 
  (hd : d = 180 * l + 174) : 
  (c + d) % 60 = 48 := 
by sorry

end NUMINAMATH_GPT_remainder_of_c_plus_d_l2281_228167


namespace NUMINAMATH_GPT_multiply_polynomials_l2281_228191

theorem multiply_polynomials (x : ℂ) : 
  (x^6 + 27 * x^3 + 729) * (x^3 - 27) = x^12 + 27 * x^9 - 19683 * x^3 - 531441 :=
by
  sorry

end NUMINAMATH_GPT_multiply_polynomials_l2281_228191


namespace NUMINAMATH_GPT_power_mod_remainder_l2281_228153

theorem power_mod_remainder 
  (h1 : 7^2 % 17 = 15)
  (h2 : 15 % 17 = -2 % 17)
  (h3 : 2^4 % 17 = -1 % 17)
  (h4 : 1011 % 2 = 1) :
  7^2023 % 17 = 12 := 
  sorry

end NUMINAMATH_GPT_power_mod_remainder_l2281_228153


namespace NUMINAMATH_GPT_backyard_area_l2281_228129

theorem backyard_area {length width : ℝ} 
  (h1 : 30 * length = 1500) 
  (h2 : 12 * (2 * (length + width)) = 1500) : 
  length * width = 625 :=
by
  sorry

end NUMINAMATH_GPT_backyard_area_l2281_228129


namespace NUMINAMATH_GPT_total_earnings_l2281_228149

-- Definitions based on conditions
def bead_necklaces : ℕ := 7
def gem_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 9

-- The main theorem to prove
theorem total_earnings : (bead_necklaces + gem_necklaces) * cost_per_necklace = 90 :=
by
  sorry

end NUMINAMATH_GPT_total_earnings_l2281_228149


namespace NUMINAMATH_GPT_tan_diff_eqn_l2281_228112

theorem tan_diff_eqn (α : ℝ) (h : Real.tan α = 2) : Real.tan (α - 3 * Real.pi / 4) = -3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_diff_eqn_l2281_228112


namespace NUMINAMATH_GPT_abs_sum_example_l2281_228186

theorem abs_sum_example : |(-8 : ℤ)| + |(-4 : ℤ)| = 12 := by
  sorry

end NUMINAMATH_GPT_abs_sum_example_l2281_228186


namespace NUMINAMATH_GPT_original_cost_111_l2281_228139

theorem original_cost_111 (P : ℝ) (h1 : 0.76 * P * 0.90 = 760) : P = 111 :=
by sorry

end NUMINAMATH_GPT_original_cost_111_l2281_228139


namespace NUMINAMATH_GPT_a7_of_arithmetic_seq_l2281_228107

-- Defining the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) = a n + d

theorem a7_of_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : arithmetic_seq a d) 
  (h_a4 : a 4 = 5) 
  (h_a5_a6 : a 5 + a 6 = 11) : 
  a 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_a7_of_arithmetic_seq_l2281_228107


namespace NUMINAMATH_GPT_paint_usage_correct_l2281_228130

-- Define the parameters representing paint usage and number of paintings
def largeCanvasPaint : Nat := 3
def smallCanvasPaint : Nat := 2
def largePaintings : Nat := 3
def smallPaintings : Nat := 4

-- Define the total paint used
def totalPaintUsed : Nat := largeCanvasPaint * largePaintings + smallCanvasPaint * smallPaintings

-- Prove that total paint used is 17 ounces
theorem paint_usage_correct : totalPaintUsed = 17 :=
  by
    sorry

end NUMINAMATH_GPT_paint_usage_correct_l2281_228130


namespace NUMINAMATH_GPT_other_solution_l2281_228168

theorem other_solution (x : ℚ) (h : 30*x^2 + 13 = 47*x - 2) (hx : x = 3/5) : x = 5/6 ∨ x = 3/5 := by
  sorry

end NUMINAMATH_GPT_other_solution_l2281_228168


namespace NUMINAMATH_GPT_calculate_number_of_boys_l2281_228166

theorem calculate_number_of_boys (old_average new_average misread correct_weight : ℝ) (number_of_boys : ℕ)
  (h1 : old_average = 58.4)
  (h2 : misread = 56)
  (h3 : correct_weight = 61)
  (h4 : new_average = 58.65)
  (h5 : (number_of_boys : ℝ) * old_average + (correct_weight - misread) = (number_of_boys : ℝ) * new_average) :
  number_of_boys = 20 :=
by
  sorry

end NUMINAMATH_GPT_calculate_number_of_boys_l2281_228166


namespace NUMINAMATH_GPT_marketing_percentage_l2281_228197

-- Define the conditions
variable (monthly_budget : ℝ)
variable (rent : ℝ := monthly_budget / 5)
variable (remaining_after_rent : ℝ := monthly_budget - rent)
variable (food_beverages : ℝ := remaining_after_rent / 4)
variable (remaining_after_food_beverages : ℝ := remaining_after_rent - food_beverages)
variable (employee_salaries : ℝ := remaining_after_food_beverages / 3)
variable (remaining_after_employee_salaries : ℝ := remaining_after_food_beverages - employee_salaries)
variable (utilities : ℝ := remaining_after_employee_salaries / 7)
variable (remaining_after_utilities : ℝ := remaining_after_employee_salaries - utilities)
variable (marketing : ℝ := 0.15 * remaining_after_utilities)

-- Define the theorem we want to prove
theorem marketing_percentage : marketing / monthly_budget * 100 = 5.14 := by
  sorry

end NUMINAMATH_GPT_marketing_percentage_l2281_228197


namespace NUMINAMATH_GPT_divisibility_l2281_228170

theorem divisibility (k n : ℕ) (hk : k > 0) (hn : n > 0) :
  (n^5 + 1) ∣ (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4 * k - 1) := 
sorry

end NUMINAMATH_GPT_divisibility_l2281_228170


namespace NUMINAMATH_GPT_inequality_proof_l2281_228160

theorem inequality_proof
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * b + b * c + c * a = 1) :
  (a + b) / Real.sqrt (a * b * (1 - a * b)) + 
  (b + c) / Real.sqrt (b * c * (1 - b * c)) + 
  (c + a) / Real.sqrt (c * a * (1 - c * a)) ≤ Real.sqrt 2 / (a * b * c) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2281_228160


namespace NUMINAMATH_GPT_num_distinct_sums_of_three_distinct_elements_l2281_228141

noncomputable def arith_seq_sum_of_three_distinct : Nat :=
  let a (i : Nat) : Nat := 3 * i + 1
  let lower_bound := 21
  let upper_bound := 129
  (upper_bound - lower_bound) / 3 + 1

theorem num_distinct_sums_of_three_distinct_elements : arith_seq_sum_of_three_distinct = 37 := by
  -- We are skipping the proof by using sorry
  sorry

end NUMINAMATH_GPT_num_distinct_sums_of_three_distinct_elements_l2281_228141


namespace NUMINAMATH_GPT_jason_egg_consumption_in_two_weeks_l2281_228164

def breakfast_pattern : List Nat := 
  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] -- Two weeks pattern alternating 3-egg and (2+1)-egg meals

noncomputable def count_eggs (pattern : List Nat) : Nat :=
  pattern.foldl (· + ·) 0

theorem jason_egg_consumption_in_two_weeks : 
  count_eggs breakfast_pattern = 42 :=
sorry

end NUMINAMATH_GPT_jason_egg_consumption_in_two_weeks_l2281_228164


namespace NUMINAMATH_GPT_sally_bread_consumption_l2281_228161

theorem sally_bread_consumption :
  (2 * 2) + (1 * 2) = 6 :=
by
  sorry

end NUMINAMATH_GPT_sally_bread_consumption_l2281_228161


namespace NUMINAMATH_GPT_find_g2_l2281_228189

open Function

variable (g : ℝ → ℝ)

axiom g_condition : ∀ x : ℝ, g x + 2 * g (1 - x) = 5 * x ^ 2

theorem find_g2 : g 2 = -10 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_g2_l2281_228189


namespace NUMINAMATH_GPT_match_foci_of_parabola_and_hyperbola_l2281_228116

noncomputable def focus_of_parabola (a : ℝ) : ℝ :=
a / 4

noncomputable def foci_of_hyperbola : Set ℝ :=
{2, -2}

theorem match_foci_of_parabola_and_hyperbola (a : ℝ) :
  focus_of_parabola a ∈ foci_of_hyperbola ↔ a = 8 ∨ a = -8 :=
by
  -- This is the placeholder for the proof.
  sorry

end NUMINAMATH_GPT_match_foci_of_parabola_and_hyperbola_l2281_228116


namespace NUMINAMATH_GPT_total_profit_l2281_228142

noncomputable def profit_x (P : ℕ) : ℕ := 3 * P
noncomputable def profit_y (P : ℕ) : ℕ := 2 * P

theorem total_profit
  (P_x P_y : ℕ)
  (h_ratio : P_x = 3 * (P_y / 2))
  (h_diff : P_x - P_y = 100) :
  P_x + P_y = 500 :=
by
  sorry

end NUMINAMATH_GPT_total_profit_l2281_228142


namespace NUMINAMATH_GPT_share_difference_3600_l2281_228147

theorem share_difference_3600 (x : ℕ) (p q r : ℕ) (h1 : p = 3 * x) (h2 : q = 7 * x) (h3 : r = 12 * x) (h4 : r - q = 4500) : q - p = 3600 := by
  sorry

end NUMINAMATH_GPT_share_difference_3600_l2281_228147


namespace NUMINAMATH_GPT_line_equation_intersects_ellipse_l2281_228175

theorem line_equation_intersects_ellipse :
  ∃ l : ℝ → ℝ → Prop,
    (∀ x y : ℝ, l x y ↔ 5 * x + 4 * y - 9 = 0) ∧
    (∃ M N : ℝ × ℝ,
      (M.1^2 / 20 + M.2^2 / 16 = 1) ∧
      (N.1^2 / 20 + N.2^2 / 16 = 1) ∧
      ((M.1 + N.1) / 2 = 1) ∧
      ((M.2 + N.2) / 2 = 1)) :=
sorry

end NUMINAMATH_GPT_line_equation_intersects_ellipse_l2281_228175


namespace NUMINAMATH_GPT_product_of_solutions_of_x_squared_eq_49_l2281_228113

theorem product_of_solutions_of_x_squared_eq_49 : 
  (∀ x, x^2 = 49 → x = 7 ∨ x = -7) → (7 * (-7) = -49) :=
by
  intros
  sorry

end NUMINAMATH_GPT_product_of_solutions_of_x_squared_eq_49_l2281_228113


namespace NUMINAMATH_GPT_birth_age_of_mother_l2281_228185

def harrys_age : ℕ := 50

def fathers_age (h : ℕ) : ℕ := h + 24

def mothers_age (f h : ℕ) : ℕ := f - h / 25

theorem birth_age_of_mother (h f m : ℕ) (H1 : h = harrys_age)
  (H2 : f = fathers_age h) (H3 : m = mothers_age f h) :
  m - h = 22 := sorry

end NUMINAMATH_GPT_birth_age_of_mother_l2281_228185


namespace NUMINAMATH_GPT_graph_depicts_one_line_l2281_228152

theorem graph_depicts_one_line {x y : ℝ} :
  (x - 1) ^ 2 * (x + y - 2) = (y - 1) ^ 2 * (x + y - 2) →
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b :=
by
  intros h
  sorry

end NUMINAMATH_GPT_graph_depicts_one_line_l2281_228152


namespace NUMINAMATH_GPT_purchased_only_A_l2281_228111

-- Definitions for the conditions
def total_B (x : ℕ) := x + 500
def total_A (y : ℕ) := 2 * y

-- Question formulated in Lean 4
theorem purchased_only_A : 
  ∃ C : ℕ, (∀ x y : ℕ, 2 * x = 500 → y = total_B x → 2 * y = total_A y → C = total_A y - 500) ∧ C = 1000 :=
  sorry

end NUMINAMATH_GPT_purchased_only_A_l2281_228111


namespace NUMINAMATH_GPT_passes_to_left_l2281_228115

theorem passes_to_left (total_passes right_passes center_passes left_passes : ℕ)
  (h_total : total_passes = 50)
  (h_right : right_passes = 2 * left_passes)
  (h_center : center_passes = left_passes + 2)
  (h_sum : left_passes + right_passes + center_passes = total_passes) :
  left_passes = 12 := 
by
  sorry

end NUMINAMATH_GPT_passes_to_left_l2281_228115


namespace NUMINAMATH_GPT_matrix_power_eigenvector_l2281_228108

section MatrixEigen
variable (B : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ)

theorem matrix_power_eigenvector (h : B.mulVec ![3, -1] = ![-12, 4]) :
  (B ^ 5).mulVec ![3, -1] = ![-3072, 1024] := 
  sorry
end MatrixEigen

end NUMINAMATH_GPT_matrix_power_eigenvector_l2281_228108


namespace NUMINAMATH_GPT_probability_one_first_class_product_l2281_228169

-- Define the probabilities for the interns processing first-class products
def P_first_intern_first_class : ℚ := 2 / 3
def P_second_intern_first_class : ℚ := 3 / 4

-- Define the events 
def P_A1 : ℚ := P_first_intern_first_class * (1 - P_second_intern_first_class)
def P_A2 : ℚ := (1 - P_first_intern_first_class) * P_second_intern_first_class

-- Probability of exactly one of the two parts being first-class product
def P_one_first_class_product : ℚ := P_A1 + P_A2

-- Theorem to be proven: the probability is 5/12
theorem probability_one_first_class_product : 
    P_one_first_class_product = 5 / 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_one_first_class_product_l2281_228169


namespace NUMINAMATH_GPT_correct_integer_with_7_divisors_l2281_228136

theorem correct_integer_with_7_divisors (n : ℕ) (p : ℕ) (h_prime : Prime p) 
  (h_3_divisors : ∃ (d : ℕ), d = 3 ∧ n = p^2) : n = 4 :=
by
-- Proof omitted
sorry

end NUMINAMATH_GPT_correct_integer_with_7_divisors_l2281_228136


namespace NUMINAMATH_GPT_john_marbles_selection_l2281_228192

theorem john_marbles_selection :
  let total_marbles := 15
  let special_colors := 4
  let total_chosen := 5
  let chosen_special_colors := 2
  let remaining_colors := total_marbles - special_colors
  let chosen_normal_colors := total_chosen - chosen_special_colors
  (Nat.choose 4 2) * (Nat.choose 11 3) = 990 :=
by
  sorry

end NUMINAMATH_GPT_john_marbles_selection_l2281_228192


namespace NUMINAMATH_GPT_total_spend_on_four_games_l2281_228122

noncomputable def calculate_total_spend (batman_price : ℝ) (superman_price : ℝ)
                                        (batman_discount : ℝ) (superman_discount : ℝ)
                                        (tax_rate : ℝ) (game1_price : ℝ) (game2_price : ℝ) : ℝ :=
  let batman_discounted_price := batman_price - batman_discount * batman_price
  let superman_discounted_price := superman_price - superman_discount * superman_price
  let batman_price_after_tax := batman_discounted_price + tax_rate * batman_discounted_price
  let superman_price_after_tax := superman_discounted_price + tax_rate * superman_discounted_price
  batman_price_after_tax + superman_price_after_tax + game1_price + game2_price

theorem total_spend_on_four_games :
  calculate_total_spend 13.60 5.06 0.10 0.05 0.08 7.25 12.50 = 38.16 :=
by sorry

end NUMINAMATH_GPT_total_spend_on_four_games_l2281_228122


namespace NUMINAMATH_GPT_probability_both_visible_l2281_228126

noncomputable def emma_lap_time : ℕ := 100
noncomputable def ethan_lap_time : ℕ := 75
noncomputable def start_time : ℕ := 0
noncomputable def photo_start_minute : ℕ := 12 * 60 -- converted to seconds
noncomputable def photo_end_minute : ℕ := 13 * 60 -- converted to seconds
noncomputable def photo_visible_angle : ℚ := 1 / 3

theorem probability_both_visible :
  ∀ start_time photo_start_minute photo_end_minute emma_lap_time ethan_lap_time photo_visible_angle,
  start_time = 0 →
  photo_start_minute = 12 * 60 →
  photo_end_minute = 13 * 60 →
  emma_lap_time = 100 →
  ethan_lap_time = 75 →
  photo_visible_angle = 1 / 3 →
  (∃ t, photo_start_minute ≤ t ∧ t < photo_end_minute ∧
        (t % emma_lap_time ≤ (photo_visible_angle * emma_lap_time) / 2 ∨
         t % emma_lap_time ≥ emma_lap_time - (photo_visible_angle * emma_lap_time) / 2) ∧
        (t % ethan_lap_time ≤ (photo_visible_angle * ethan_lap_time) / 2 ∨
         t % ethan_lap_time ≥ ethan_lap_time - (photo_visible_angle * ethan_lap_time) / 2)) ↔
  true :=
sorry

end NUMINAMATH_GPT_probability_both_visible_l2281_228126


namespace NUMINAMATH_GPT_trigonometric_identity_l2281_228177

theorem trigonometric_identity 
  (x : ℝ)
  (h : Real.cos (π / 6 - x) = - Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2281_228177


namespace NUMINAMATH_GPT_problem_I_problem_II_l2281_228158

variable (x a m : ℝ)

theorem problem_I (h: ¬ (∃ r : ℝ, r^2 - 2*a*r + 2*a^2 - a - 6 = 0)) : 
  a < -2 ∨ a > 3 := by
  sorry

theorem problem_II (p : ∃ r : ℝ, r^2 - 2*a*r + 2*a^2 - a - 6 = 0) (q : m-1 ≤ a ∧ a ≤ m+3) :
  ∀ a : ℝ, -2 ≤ a ∧ a ≤ 3 → m ∈ [-1, 0] := by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l2281_228158


namespace NUMINAMATH_GPT_intersecting_lines_l2281_228165

-- Definitions for the conditions
def line1 (x y a : ℝ) : Prop := x = (1/3) * y + a
def line2 (x y b : ℝ) : Prop := y = (1/3) * x + b

-- The theorem we need to prove
theorem intersecting_lines (a b : ℝ) (h1 : ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ line1 x y a) 
                           (h2 : ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ line2 x y b) : 
  a + b = 10 / 3 :=
sorry

end NUMINAMATH_GPT_intersecting_lines_l2281_228165


namespace NUMINAMATH_GPT_tomato_land_correct_l2281_228198

-- Define the conditions
def total_land : ℝ := 4999.999999999999
def cleared_fraction : ℝ := 0.9
def grapes_fraction : ℝ := 0.1
def potato_fraction : ℝ := 0.8

-- Define the calculated values based on conditions
def cleared_land : ℝ := cleared_fraction * total_land
def grapes_land : ℝ := grapes_fraction * cleared_land
def potato_land : ℝ := potato_fraction * cleared_land
def tomato_land : ℝ := cleared_land - (grapes_land + potato_land)

-- Prove the question using conditions, which should end up being 450 acres.
theorem tomato_land_correct : tomato_land = 450 :=
by sorry

end NUMINAMATH_GPT_tomato_land_correct_l2281_228198


namespace NUMINAMATH_GPT_y_is_triangular_l2281_228176

theorem y_is_triangular (k : ℕ) (hk : k > 0) : 
  ∃ n : ℕ, y = (n * (n + 1)) / 2 :=
by
  let y := (9^k - 1) / 8
  sorry

end NUMINAMATH_GPT_y_is_triangular_l2281_228176


namespace NUMINAMATH_GPT_system_solution_l2281_228171

theorem system_solution (x y z : ℝ) 
  (h1 : 2 * x - 3 * y + z = 8) 
  (h2 : 4 * x - 6 * y + 2 * z = 16) 
  (h3 : x + y - z = 1) : 
  x = 11 / 3 ∧ y = 1 ∧ z = 11 / 3 :=
by
  sorry

end NUMINAMATH_GPT_system_solution_l2281_228171


namespace NUMINAMATH_GPT_find_k_l2281_228187
-- Import the necessary library

-- Given conditions as definitions
def circle_eq (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + 8 * x + y^2 + 2 * y + k = 0

def radius_sq : ℝ := 25  -- since radius = 5, radius squared is 25

-- The statement to prove
theorem find_k (x y k : ℝ) : circle_eq x y k → radius_sq = 25 → k = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2281_228187


namespace NUMINAMATH_GPT_five_dice_not_all_same_number_l2281_228194
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end NUMINAMATH_GPT_five_dice_not_all_same_number_l2281_228194


namespace NUMINAMATH_GPT_find_triples_l2281_228184

theorem find_triples (x y z : ℕ) :
  (x + 1)^(y + 1) + 1 = (x + 2)^(z + 1) ↔ (x = 1 ∧ y = 2 ∧ z = 1) :=
sorry

end NUMINAMATH_GPT_find_triples_l2281_228184


namespace NUMINAMATH_GPT_cost_of_first_book_l2281_228138

-- Define the initial amount of money Shelby had.
def initial_amount : ℕ := 20

-- Define the cost of the second book.
def cost_of_second_book : ℕ := 4

-- Define the cost of one poster.
def cost_of_poster : ℕ := 4

-- Define the number of posters bought.
def num_posters : ℕ := 2

-- Define the total cost that Shelby had to spend on posters.
def total_cost_of_posters : ℕ := num_posters * cost_of_poster

-- Define the total amount spent on books and posters.
def total_spent (X : ℕ) : ℕ := X + cost_of_second_book + total_cost_of_posters

-- Prove that the cost of the first book is 8 dollars.
theorem cost_of_first_book (X : ℕ) (h : total_spent X = initial_amount) : X = 8 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_first_book_l2281_228138


namespace NUMINAMATH_GPT_scientific_notation_248000_l2281_228151

theorem scientific_notation_248000 : (248000 : Float) = 2.48 * 10^5 := 
sorry

end NUMINAMATH_GPT_scientific_notation_248000_l2281_228151


namespace NUMINAMATH_GPT_geometric_sequence_problem_l2281_228178

variable {a : ℕ → ℝ}
variable (r a1 : ℝ)
variable (h_pos : ∀ n, a n > 0)
variable (h_geom : ∀ n, a (n + 1) = a 1 * r ^ n)
variable (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 2025)

theorem geometric_sequence_problem :
  a 3 + a 5 = 45 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l2281_228178


namespace NUMINAMATH_GPT_pascal_row_contains_prime_47_l2281_228163

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end NUMINAMATH_GPT_pascal_row_contains_prime_47_l2281_228163


namespace NUMINAMATH_GPT_coin_pile_problem_l2281_228150

theorem coin_pile_problem (x y z : ℕ) (h1 : 2 * (x - y) = 16) (h2 : 2 * y - z = 16) (h3 : 2 * z - x + y = 16) :
  x = 22 ∧ y = 14 ∧ z = 12 :=
by
  sorry

end NUMINAMATH_GPT_coin_pile_problem_l2281_228150


namespace NUMINAMATH_GPT_number_of_juniors_twice_seniors_l2281_228118

variable (j s : ℕ)

theorem number_of_juniors_twice_seniors
  (h1 : (3 / 7 : ℝ) * j = (6 / 7 : ℝ) * s) : j = 2 * s := 
sorry

end NUMINAMATH_GPT_number_of_juniors_twice_seniors_l2281_228118


namespace NUMINAMATH_GPT_kids_from_lawrence_county_go_to_camp_l2281_228148

theorem kids_from_lawrence_county_go_to_camp : 
  (1201565 - 590796 = 610769) := 
by
  sorry

end NUMINAMATH_GPT_kids_from_lawrence_county_go_to_camp_l2281_228148


namespace NUMINAMATH_GPT_find_unit_prices_l2281_228183

theorem find_unit_prices (price_A price_B : ℕ) 
  (h1 : price_A = price_B + 5) 
  (h2 : 1000 / price_A = 750 / price_B) : 
  price_A = 20 ∧ price_B = 15 := 
by 
  sorry

end NUMINAMATH_GPT_find_unit_prices_l2281_228183


namespace NUMINAMATH_GPT_days_worked_prove_l2281_228101

/-- Work rate of A is 1/15 work per day -/
def work_rate_A : ℚ := 1/15

/-- Work rate of B is 1/20 work per day -/
def work_rate_B : ℚ := 1/20

/-- Combined work rate of A and B -/
def combined_work_rate : ℚ := work_rate_A + work_rate_B

/-- Fraction of work left after some days -/
def fraction_work_left : ℚ := 8/15

/-- Fraction of work completed after some days -/
def fraction_work_completed : ℚ := 1 - fraction_work_left

/-- Number of days A and B worked together -/
def days_worked_together : ℚ := fraction_work_completed / combined_work_rate

theorem days_worked_prove : 
    days_worked_together = 4 := 
by 
    sorry

end NUMINAMATH_GPT_days_worked_prove_l2281_228101


namespace NUMINAMATH_GPT_largest_value_of_c_l2281_228133

noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem largest_value_of_c :
  ∃ (c : ℝ), (∀ (x d : ℝ), |x - 1| ≤ d ∧ 0 < d ∧ 0 < c → |g x - 1| ≤ c) ∧ (∀ (c' : ℝ), (∀ (x d : ℝ), |x - 1| ≤ d ∧ 0 < d ∧ 0 < c' → |g x - 1| ≤ c') → c' ≤ c) :=
sorry

end NUMINAMATH_GPT_largest_value_of_c_l2281_228133


namespace NUMINAMATH_GPT_quotient_sum_40_5_l2281_228104

theorem quotient_sum_40_5 : (40 + 5) / 5 = 9 := by
  sorry

end NUMINAMATH_GPT_quotient_sum_40_5_l2281_228104


namespace NUMINAMATH_GPT_lower_limit_brother_l2281_228105

variable (W B : Real)

-- Arun's opinion
def aruns_opinion := 66 < W ∧ W < 72

-- Brother's opinion
def brothers_opinion := B < W ∧ W < 70

-- Mother's opinion
def mothers_opinion := W ≤ 69

-- Given the average probable weight of Arun which is 68 kg
def average_weight := (69 + (max 66 B)) / 2 = 68

theorem lower_limit_brother (h₁ : aruns_opinion W) (h₂ : brothers_opinion W B) (h₃ : mothers_opinion W) (h₄ : average_weight B) :
  B = 67 := sorry

end NUMINAMATH_GPT_lower_limit_brother_l2281_228105


namespace NUMINAMATH_GPT_polynomial_division_result_l2281_228154

-- Define the given polynomials
def f (x : ℝ) : ℝ := 4 * x ^ 4 + 12 * x ^ 3 - 9 * x ^ 2 + 2 * x + 3
def d (x : ℝ) : ℝ := x ^ 2 + 2 * x - 3

-- Define the computed quotient and remainder
def q (x : ℝ) : ℝ := 4 * x ^ 2 + 4
def r (x : ℝ) : ℝ := -12 * x + 42

theorem polynomial_division_result :
  (∀ x : ℝ, f x = q x * d x + r x) ∧ (q 1 + r (-1) = 62) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_division_result_l2281_228154


namespace NUMINAMATH_GPT_count_of_changing_quantities_l2281_228156

-- Definitions of the problem conditions
def length_AC_unchanged : Prop := ∀ P A B C D : ℝ, true
def perimeter_square_unchanged : Prop := ∀ P A B C D : ℝ, true
def area_square_unchanged : Prop := ∀ P A B C D : ℝ, true
def area_quadrilateral_changed : Prop := ∀ P A B C D M N : ℝ, true

-- The main theorem to prove
theorem count_of_changing_quantities :
  length_AC_unchanged ∧
  perimeter_square_unchanged ∧
  area_square_unchanged ∧
  area_quadrilateral_changed →
  (1 = 1) :=
by
  sorry

end NUMINAMATH_GPT_count_of_changing_quantities_l2281_228156


namespace NUMINAMATH_GPT_quadratic_other_root_l2281_228159

theorem quadratic_other_root (m x2 : ℝ) (h₁ : 1^2 - 4*1 + m = 0) (h₂ : x2^2 - 4*x2 + m = 0) : x2 = 3 :=
sorry

end NUMINAMATH_GPT_quadratic_other_root_l2281_228159


namespace NUMINAMATH_GPT_find_m_of_ellipse_l2281_228172

theorem find_m_of_ellipse (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (10 - m) + y^2 / (m - 2) = 1) ∧ (m - 2 > 10 - m) ∧ ((4)^2 = (m - 2) - (10 - m))) → m = 14 :=
by sorry

end NUMINAMATH_GPT_find_m_of_ellipse_l2281_228172


namespace NUMINAMATH_GPT_largest_possible_value_for_a_l2281_228180

theorem largest_possible_value_for_a (a b c d : ℕ) 
  (h1: a < 3 * b) 
  (h2: b < 2 * c + 1) 
  (h3: c < 5 * d - 2)
  (h4: d ≤ 50) 
  (h5: d % 5 = 0) : 
  a ≤ 1481 :=
sorry

end NUMINAMATH_GPT_largest_possible_value_for_a_l2281_228180


namespace NUMINAMATH_GPT_traveler_journey_possible_l2281_228103

structure Archipelago (Island : Type) :=
  (n : ℕ)
  (fare : Island → Island → ℝ)
  (unique_ferry : ∀ i j : Island, i ≠ j → fare i j ≠ fare j i)
  (distinct_fares : ∀ i j k l: Island, i ≠ j ∧ k ≠ l → fare i j ≠ fare k l)
  (connected : ∀ i j : Island, i ≠ j → fare i j = fare j i)

theorem traveler_journey_possible {Island : Type} (arch : Archipelago Island) :
  ∃ (t : Island) (seq : List (Island × Island)), -- there exists a starting island and a sequence of journeys
    seq.length = arch.n - 1 ∧                   -- length of the sequence is n-1
    (∀ i j, (i, j) ∈ seq → j ≠ i ∧ arch.fare i j < arch.fare j i) := -- fare decreases with each journey
sorry

end NUMINAMATH_GPT_traveler_journey_possible_l2281_228103


namespace NUMINAMATH_GPT_total_kids_played_l2281_228127

def kids_played_week (monday tuesday wednesday thursday: ℕ): ℕ :=
  let friday := thursday + (thursday * 20 / 100)
  let saturday := friday - (friday * 30 / 100)
  let sunday := 2 * monday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem total_kids_played : 
  kids_played_week 15 18 25 30 = 180 :=
by
  sorry

end NUMINAMATH_GPT_total_kids_played_l2281_228127


namespace NUMINAMATH_GPT_find_b_over_a_l2281_228106

variables {a b c : ℝ}
variables {b₃ b₇ b₁₁ : ℝ}

-- Conditions
def roots_of_quadratic (a b c b₃ b₁₁ : ℝ) : Prop :=
  ∃ p q, p + q = -b / a ∧ p * q = c / a ∧ (p = b₃ ∨ p = b₁₁) ∧ (q = b₃ ∨ q = b₁₁)

def middle_term_value (b₇ : ℝ) : Prop :=
  b₇ = 3

-- The statement to be proved
theorem find_b_over_a
  (h1 : roots_of_quadratic a b c b₃ b₁₁)
  (h2 : middle_term_value b₇)
  (h3 : b₃ + b₁₁ = 2 * b₇) :
  b / a = -6 :=
sorry

end NUMINAMATH_GPT_find_b_over_a_l2281_228106


namespace NUMINAMATH_GPT_initial_payment_mr_dubois_l2281_228125

-- Definition of the given conditions
def total_cost_of_car : ℝ := 13380
def monthly_payment : ℝ := 420
def number_of_months : ℝ := 19

-- Calculate the total amount paid in monthly installments
def total_amount_paid_in_installments : ℝ := monthly_payment * number_of_months

-- Statement of the theorem we want to prove
theorem initial_payment_mr_dubois :
  total_cost_of_car - total_amount_paid_in_installments = 5400 :=
by
  sorry

end NUMINAMATH_GPT_initial_payment_mr_dubois_l2281_228125


namespace NUMINAMATH_GPT_max_distance_point_circle_l2281_228190

open Real

noncomputable def distance (P C : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)

theorem max_distance_point_circle :
  let C : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (3, 3)
  let r : ℝ := 2
  let max_distance : ℝ := distance P C + r
  ∃ M : ℝ × ℝ, distance P M = max_distance ∧ (M.1 - 1)^2 + (M.2 - 2)^2 = r^2 :=
by
  sorry

end NUMINAMATH_GPT_max_distance_point_circle_l2281_228190


namespace NUMINAMATH_GPT_product_of_cubes_eq_l2281_228109

theorem product_of_cubes_eq :
  ( (3^3 - 1) / (3^3 + 1) ) * 
  ( (4^3 - 1) / (4^3 + 1) ) * 
  ( (5^3 - 1) / (5^3 + 1) ) * 
  ( (6^3 - 1) / (6^3 + 1) ) * 
  ( (7^3 - 1) / (7^3 + 1) ) * 
  ( (8^3 - 1) / (8^3 + 1) ) 
  = 73 / 256 :=
by
  sorry

end NUMINAMATH_GPT_product_of_cubes_eq_l2281_228109


namespace NUMINAMATH_GPT_problem_solution_l2281_228110

-- Definition of the geometric sequence and the arithmetic condition
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = q * a n
def arithmetic_condition (a : ℕ → ℕ) := 2 * (a 3 + 1) = a 2 + a 4

-- Definitions used in the proof
def a_n (n : ℕ) : ℕ := 2^(n-1)
def b_n (n : ℕ) := a_n n + n
def S_5 := b_n 1 + b_n 2 + b_n 3 + b_n 4 + b_n 5

-- Proof statement
theorem problem_solution : 
  (∃ a : ℕ → ℕ, geometric_sequence a 2 ∧ arithmetic_condition a ∧ a 1 = 1 ∧ (∀ n, a n = 2^(n-1))) ∧
  S_5 = 46 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2281_228110


namespace NUMINAMATH_GPT_max_log_sum_l2281_228119

open Real

theorem max_log_sum (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 4 * y = 40) :
  log x + log y ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_log_sum_l2281_228119


namespace NUMINAMATH_GPT_find_a_b_l2281_228134

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem find_a_b (a b : ℝ) (x : ℝ) (h : 5 * (log a x) ^ 2 + 2 * (log b x) ^ 2 = (10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)) + (Real.log x) ^ 2) :
  b = a ^ (2 / (5 + Real.sqrt 17)) ∨ b = a ^ (2 / (5 - Real.sqrt 17)) :=
sorry

end NUMINAMATH_GPT_find_a_b_l2281_228134


namespace NUMINAMATH_GPT_number_of_valid_pairs_l2281_228193

theorem number_of_valid_pairs : ∃ p : Finset (ℕ × ℕ), 
  (∀ (a b : ℕ), (a, b) ∈ p ↔ a ≤ 10 ∧ b ≤ 10 ∧ 3 * b < a ∧ a < 4 * b) ∧ p.card = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_pairs_l2281_228193


namespace NUMINAMATH_GPT_infinite_alternating_parity_l2281_228123

theorem infinite_alternating_parity (m : ℕ) : ∃ᶠ n in at_top, 
  ∀ i < m, ((5^n / 10^i) % 2) ≠ (((5^n / 10^(i+1)) % 10) % 2) :=
sorry

end NUMINAMATH_GPT_infinite_alternating_parity_l2281_228123


namespace NUMINAMATH_GPT_range_of_a_l2281_228144

variable (a x : ℝ)

theorem range_of_a (h : x - 5 = -3 * a) (hx_neg : x < 0) : a > 5 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l2281_228144


namespace NUMINAMATH_GPT_probability_of_six_and_queen_l2281_228188

variable {deck : Finset (ℕ × String)}
variable (sixes : Finset (ℕ × String))
variable (queens : Finset (ℕ × String))

def standard_deck : Finset (ℕ × String) := sorry

-- Condition: the deck contains 52 cards (13 hearts, 13 clubs, 13 spades, 13 diamonds)
-- and it has 4 sixes and 4 Queens.
axiom h_deck_size : standard_deck.card = 52
axiom h_sixes : ∀ c ∈ standard_deck, c.1 = 6 → c ∈ sixes
axiom h_queens : ∀ c ∈ standard_deck, c.1 = 12 → c ∈ queens

-- Define the probability function for dealing cards
noncomputable def prob_first_six_and_second_queen : ℚ :=
  (4 / 52) * (4 / 51)

theorem probability_of_six_and_queen :
  prob_first_six_and_second_queen = 4 / 663 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_six_and_queen_l2281_228188


namespace NUMINAMATH_GPT_arrangements_APPLE_is_60_l2281_228182

-- Definition of the problem statement based on the given conditions
def distinct_arrangements_APPLE : Nat :=
  let n := 5
  let n_A := 1
  let n_P := 2
  let n_L := 1
  let n_E := 1
  (n.factorial / (n_A.factorial * n_P.factorial * n_L.factorial * n_E.factorial))

-- The proof statement (without the proof itself, which is "sorry")
theorem arrangements_APPLE_is_60 : distinct_arrangements_APPLE = 60 := by
  sorry

end NUMINAMATH_GPT_arrangements_APPLE_is_60_l2281_228182


namespace NUMINAMATH_GPT_log_expression_value_l2281_228157

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_expression_value :
  log_base 10 3 + 3 * log_base 10 2 + 2 * log_base 10 5 + 4 * log_base 10 3 + log_base 10 9 = 5.34 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_value_l2281_228157


namespace NUMINAMATH_GPT_profit_percentage_correct_l2281_228146

noncomputable def cost_price : ℝ := 47.50
noncomputable def selling_price : ℝ := 70
noncomputable def list_price : ℝ := selling_price / 0.95
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem profit_percentage_correct :
  abs (profit_percentage - 47.37) < 0.01 := sorry

end NUMINAMATH_GPT_profit_percentage_correct_l2281_228146
