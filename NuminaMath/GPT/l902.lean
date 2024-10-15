import Mathlib

namespace NUMINAMATH_GPT_Lance_daily_earnings_l902_90289

theorem Lance_daily_earnings :
  ∀ (hours_per_week : ℕ) (workdays_per_week : ℕ) (hourly_rate : ℕ) (total_earnings : ℕ) (daily_earnings : ℕ),
  hours_per_week = 35 →
  workdays_per_week = 5 →
  hourly_rate = 9 →
  total_earnings = hours_per_week * hourly_rate →
  daily_earnings = total_earnings / workdays_per_week →
  daily_earnings = 63 := 
by
  intros hours_per_week workdays_per_week hourly_rate total_earnings daily_earnings
  intros H1 H2 H3 H4 H5
  sorry

end NUMINAMATH_GPT_Lance_daily_earnings_l902_90289


namespace NUMINAMATH_GPT_exponent_calculation_l902_90223

theorem exponent_calculation : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end NUMINAMATH_GPT_exponent_calculation_l902_90223


namespace NUMINAMATH_GPT_find_value_l902_90203

variable (number : ℝ) (V : ℝ)

theorem find_value
  (h1 : number = 8)
  (h2 : 0.75 * number + V = 8) : V = 2 := by
  sorry

end NUMINAMATH_GPT_find_value_l902_90203


namespace NUMINAMATH_GPT_greatest_x_is_53_l902_90217

-- Define the polynomial expression
def polynomial (x : ℤ) : ℤ := x^2 + 2 * x + 13

-- Define the condition for the expression to be an integer
def isIntegerWhenDivided (x : ℤ) : Prop := (polynomial x) % (x - 5) = 0

-- Define the theorem to prove the greatest integer value of x
theorem greatest_x_is_53 : ∃ x : ℤ, isIntegerWhenDivided x ∧ (∀ y : ℤ, isIntegerWhenDivided y → y ≤ x) ∧ x = 53 :=
by
  sorry

end NUMINAMATH_GPT_greatest_x_is_53_l902_90217


namespace NUMINAMATH_GPT_sum_of_segments_AK_KB_eq_AB_l902_90241

-- Given conditions: length of segment AB is 9 cm
def length_AB : ℝ := 9

-- For any point K on segment AB, prove that AK + KB = AB
theorem sum_of_segments_AK_KB_eq_AB (K : ℝ) (h : 0 ≤ K ∧ K ≤ length_AB) : 
  K + (length_AB - K) = length_AB := by
  sorry

end NUMINAMATH_GPT_sum_of_segments_AK_KB_eq_AB_l902_90241


namespace NUMINAMATH_GPT_find_third_number_l902_90216

theorem find_third_number (x : ℕ) (h : 3 * 16 + 3 * 17 + 3 * x + 11 = 170) : x = 20 := by
  sorry

end NUMINAMATH_GPT_find_third_number_l902_90216


namespace NUMINAMATH_GPT_triangle_inequality_l902_90240

theorem triangle_inequality 
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  2 * (a + b + c) * (a * b + b * c + c * a) ≤ (a + b + c) * (a^2 + b^2 + c^2) + 9 * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l902_90240


namespace NUMINAMATH_GPT_sum_of_numbers_l902_90201

theorem sum_of_numbers :
  1357 + 7531 + 3175 + 5713 = 17776 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l902_90201


namespace NUMINAMATH_GPT_find_pairs_l902_90211

theorem find_pairs (a b : ℕ) (q r : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : 0 ≤ r) (h5 : r < a + b)
  (h6 : q^2 + r = 1977) :
  (a, b) = (50, 37) ∨ (a, b) = (50, 7) ∨ (a, b) = (37, 50) ∨ (a, b) = (7, 50) :=
  sorry

end NUMINAMATH_GPT_find_pairs_l902_90211


namespace NUMINAMATH_GPT_fraction_subtraction_l902_90250

theorem fraction_subtraction (a b : ℝ) (h1 : 2 * b = 1 + a * b) (h2 : a ≠ 1) (h3 : b ≠ 1) :
  (a + 1) / (a - 1) - (b + 1) / (b - 1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l902_90250


namespace NUMINAMATH_GPT_dig_site_date_l902_90273

theorem dig_site_date (S1 S2 S3 S4 : ℕ) (S2_bc : S2 = 852) 
  (h1 : S1 = S2 - 352) 
  (h2 : S3 = S1 + 3700) 
  (h3 : S4 = 2 * S3) : 
  S4 = 6400 :=
by sorry

end NUMINAMATH_GPT_dig_site_date_l902_90273


namespace NUMINAMATH_GPT_second_planner_cheaper_l902_90233

theorem second_planner_cheaper (x : ℕ) :
  (∀ x, 250 + 15 * x < 150 + 18 * x → x ≥ 34) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_second_planner_cheaper_l902_90233


namespace NUMINAMATH_GPT_range_of_a_min_value_ab_range_of_y_l902_90246
-- Import the necessary Lean library 

-- Problem 1
theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x - 3| ≥ a^2 + a) → (-2 ≤ a ∧ a ≤ 1) := 
sorry

-- Problem 2
theorem min_value_ab (a b : ℝ) (h₁ : a + b = 1) : 
  (∀ x, |x - 1| + |x - 3| ≥ a^2 + a) → 
  (min ((1 : ℝ) / (4 * |b|) + |b| / a) = 3 / 4 ∧ (a = 2)) :=
sorry

-- Problem 3
theorem range_of_y (a : ℝ) (y : ℝ) (h₁ : a ∈ Set.Ici (2 : ℝ)) : 
  y = (2 * a) / (a^2 + 1) → 0 < y ∧ y ≤ (4 / 5) :=
sorry

end NUMINAMATH_GPT_range_of_a_min_value_ab_range_of_y_l902_90246


namespace NUMINAMATH_GPT_arith_seq_S13_value_l902_90291

variable {α : Type*} [LinearOrderedField α]

-- Definitions related to an arithmetic sequence
structure ArithSeq (α : Type*) :=
  (a : ℕ → α) -- the sequence itself
  (sum_first_n_terms : ℕ → α) -- sum of the first n terms

def is_arith_seq (seq : ArithSeq α) :=
  ∀ (n : ℕ), seq.a (n + 1) - seq.a n = seq.a 2 - seq.a 1

-- Our conditions
noncomputable def a5 (seq : ArithSeq α) := seq.a 5
noncomputable def a7 (seq : ArithSeq α) := seq.a 7
noncomputable def a9 (seq : ArithSeq α) := seq.a 9
noncomputable def S13 (seq : ArithSeq α) := seq.sum_first_n_terms 13

-- Problem statement
theorem arith_seq_S13_value (seq : ArithSeq α) 
  (h_arith_seq : is_arith_seq seq)
  (h_condition : 2 * (a5 seq) + 3 * (a7 seq) + 2 * (a9 seq) = 14) : 
  S13 seq = 26 := 
  sorry

end NUMINAMATH_GPT_arith_seq_S13_value_l902_90291


namespace NUMINAMATH_GPT_laps_run_l902_90236

theorem laps_run (x : ℕ) (total_distance required_distance lap_length extra_laps : ℕ) (h1 : total_distance = 2400) (h2 : lap_length = 150) (h3 : extra_laps = 4) (h4 : total_distance = lap_length * (x + extra_laps)) : x = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_laps_run_l902_90236


namespace NUMINAMATH_GPT_find_x_l902_90222

theorem find_x :
  ∃ x : ℝ, x = (1/x) * (-x) - 3*x + 4 ∧ x = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l902_90222


namespace NUMINAMATH_GPT_sum_50th_set_l902_90242

-- Definition of the sequence repeating pattern
def repeating_sequence : List (List Nat) :=
  [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]]

-- Definition to get the nth set in the repeating sequence
def nth_set (n : Nat) : List Nat :=
  repeating_sequence.get! ((n - 1) % 4)

-- Definition to sum the elements of a list
def sum_list (l : List Nat) : Nat :=
  l.sum

-- Proposition to prove that the sum of the 50th set is 4
theorem sum_50th_set : sum_list (nth_set 50) = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_50th_set_l902_90242


namespace NUMINAMATH_GPT_least_number_to_add_l902_90262

theorem least_number_to_add (k : ℕ) (h : 1019 % 25 = 19) : (1019 + k) % 25 = 0 ↔ k = 6 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l902_90262


namespace NUMINAMATH_GPT_jack_sees_color_change_l902_90225

noncomputable def traffic_light_cycle := 95    -- Total duration of the traffic light cycle
noncomputable def change_window := 15          -- Duration window where color change occurs
def observation_interval := 5                  -- Length of Jack's observation interval

/-- Probability that Jack sees the color change during his observation. -/
def probability_of_observing_change (cycle: ℕ) (window: ℕ) : ℚ :=
  window / cycle

theorem jack_sees_color_change :
  probability_of_observing_change traffic_light_cycle change_window = 3 / 19 :=
by
  -- We only need the statement for verification
  sorry

end NUMINAMATH_GPT_jack_sees_color_change_l902_90225


namespace NUMINAMATH_GPT_floral_shop_bouquets_total_l902_90278

theorem floral_shop_bouquets_total (sold_monday_rose : ℕ) (sold_monday_lily : ℕ) (sold_monday_orchid : ℕ)
  (price_monday_rose : ℕ) (price_monday_lily : ℕ) (price_monday_orchid : ℕ)
  (sold_tuesday_rose : ℕ) (sold_tuesday_lily : ℕ) (sold_tuesday_orchid : ℕ)
  (price_tuesday_rose : ℕ) (price_tuesday_lily : ℕ) (price_tuesday_orchid : ℕ)
  (sold_wednesday_rose : ℕ) (sold_wednesday_lily : ℕ) (sold_wednesday_orchid : ℕ)
  (price_wednesday_rose : ℕ) (price_wednesday_lily : ℕ) (price_wednesday_orchid : ℕ)
  (H1 : sold_monday_rose = 12) (H2 : sold_monday_lily = 8) (H3 : sold_monday_orchid = 6)
  (H4 : price_monday_rose = 10) (H5 : price_monday_lily = 15) (H6 : price_monday_orchid = 20)
  (H7 : sold_tuesday_rose = 3 * sold_monday_rose) (H8 : sold_tuesday_lily = 2 * sold_monday_lily)
  (H9 : sold_tuesday_orchid = sold_monday_orchid / 2) (H10 : price_tuesday_rose = 12)
  (H11 : price_tuesday_lily = 18) (H12 : price_tuesday_orchid = 22)
  (H13 : sold_wednesday_rose = sold_tuesday_rose / 3) (H14 : sold_wednesday_lily = sold_tuesday_lily / 4)
  (H15 : sold_wednesday_orchid = 2 * sold_tuesday_orchid / 3) (H16 : price_wednesday_rose = 8)
  (H17 : price_wednesday_lily = 12) (H18 : price_wednesday_orchid = 16) :
  (sold_monday_rose + sold_tuesday_rose + sold_wednesday_rose = 60) ∧
  (sold_monday_lily + sold_tuesday_lily + sold_wednesday_lily = 28) ∧
  (sold_monday_orchid + sold_tuesday_orchid + sold_wednesday_orchid = 11) ∧
  ((sold_monday_rose * price_monday_rose + sold_tuesday_rose * price_tuesday_rose + sold_wednesday_rose * price_wednesday_rose) = 648) ∧
  ((sold_monday_lily * price_monday_lily + sold_tuesday_lily * price_tuesday_lily + sold_wednesday_lily * price_wednesday_lily) = 456) ∧
  ((sold_monday_orchid * price_monday_orchid + sold_tuesday_orchid * price_tuesday_orchid + sold_wednesday_orchid * price_wednesday_orchid) = 218) ∧
  ((sold_monday_rose + sold_tuesday_rose + sold_wednesday_rose + sold_monday_lily + sold_tuesday_lily + sold_wednesday_lily + sold_monday_orchid + sold_tuesday_orchid + sold_wednesday_orchid) = 99) ∧
  ((sold_monday_rose * price_monday_rose + sold_tuesday_rose * price_tuesday_rose + sold_wednesday_rose * price_wednesday_rose + sold_monday_lily * price_monday_lily + sold_tuesday_lily * price_tuesday_lily + sold_wednesday_lily * price_wednesday_lily + sold_monday_orchid * price_monday_orchid + sold_tuesday_orchid * price_tuesday_orchid + sold_wednesday_orchid * price_wednesday_orchid) = 1322) :=
  by sorry

end NUMINAMATH_GPT_floral_shop_bouquets_total_l902_90278


namespace NUMINAMATH_GPT_roots_opposite_sign_eq_magnitude_l902_90210

theorem roots_opposite_sign_eq_magnitude (c d e n : ℝ) (h : ((n+2) * (x^2 + c*x + d)) = (n-2) * (2*x - e)) :
  n = (-4 - 2 * c) / (c - 2) :=
by
  sorry

end NUMINAMATH_GPT_roots_opposite_sign_eq_magnitude_l902_90210


namespace NUMINAMATH_GPT_squirrels_cannot_divide_equally_l902_90256

theorem squirrels_cannot_divide_equally
    (n : ℕ) : ¬ (∃ k, 2022 + n * (n + 1) = 5 * k) :=
by
sorry

end NUMINAMATH_GPT_squirrels_cannot_divide_equally_l902_90256


namespace NUMINAMATH_GPT_hyperbola_equation_l902_90265

theorem hyperbola_equation 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_asymptote : (b / a) = (Real.sqrt 3 / 2))
  (c : ℝ) (hc : c = Real.sqrt 7)
  (foci_directrix_condition : a^2 + b^2 = c^2) :
  (∀ x y : ℝ, (x^2 / 4 - y^2 / 3 = 1)) :=
by
  -- We do not provide the proof as per instructions
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l902_90265


namespace NUMINAMATH_GPT_incorrect_statement_A_l902_90227

theorem incorrect_statement_A (x_1 x_2 y_1 y_2 : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 - 2*x - 4*y - 4 = 0) ∧
  x_1 = 1 - Real.sqrt 5 ∧
  x_2 = 1 + Real.sqrt 5 ∧
  y_1 = 2 - 2 * Real.sqrt 2 ∧
  y_2 = 2 + 2 * Real.sqrt 2 →
  x_1 + x_2 ≠ -2 := by
  intro h
  sorry

end NUMINAMATH_GPT_incorrect_statement_A_l902_90227


namespace NUMINAMATH_GPT_expand_product_l902_90214

theorem expand_product (x : ℝ) (hx : x ≠ 0) : 
  (3 / 7) * (7 / x^3 - 14 * x^4) = 3 / x^3 - 6 * x^4 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l902_90214


namespace NUMINAMATH_GPT_geom_seq_min_val_l902_90293

-- Definition of geometric sequence with common ratio q
def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Main theorem
theorem geom_seq_min_val (a : ℕ → ℝ) (q : ℝ) 
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_geom : geom_seq a q)
  (h_cond : 2 * a 3 + a 2 - 2 * a 1 - a 0 = 8) :
  2 * a 4 + a 3 = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_geom_seq_min_val_l902_90293


namespace NUMINAMATH_GPT_gain_percent_correct_l902_90257

noncomputable def cycleCP : ℝ := 900
noncomputable def cycleSP : ℝ := 1180
noncomputable def gainPercent : ℝ := (cycleSP - cycleCP) / cycleCP * 100

theorem gain_percent_correct :
  gainPercent = 31.11 := by
  sorry

end NUMINAMATH_GPT_gain_percent_correct_l902_90257


namespace NUMINAMATH_GPT_find_price_of_pants_l902_90204

theorem find_price_of_pants
  (price_jacket : ℕ)
  (num_jackets : ℕ)
  (price_shorts : ℕ)
  (num_shorts : ℕ)
  (num_pants : ℕ)
  (total_cost : ℕ)
  (h1 : price_jacket = 10)
  (h2 : num_jackets = 3)
  (h3 : price_shorts = 6)
  (h4 : num_shorts = 2)
  (h5 : num_pants = 4)
  (h6 : total_cost = 90)
  : (total_cost - (num_jackets * price_jacket + num_shorts * price_shorts)) / num_pants = 12 :=
by sorry

end NUMINAMATH_GPT_find_price_of_pants_l902_90204


namespace NUMINAMATH_GPT_problem_solution_l902_90268

variable (x : ℝ)

-- Given condition
def condition1 : Prop := (7 / 8) * x = 28

-- The main statement to prove
theorem problem_solution (h : condition1 x) : (x + 16) * (5 / 16) = 15 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l902_90268


namespace NUMINAMATH_GPT_inequality_solution_l902_90215

theorem inequality_solution (x : ℝ) : x^3 - 9 * x^2 + 27 * x > 0 → (x > 0 ∧ x < 3) ∨ (x > 6) := sorry

end NUMINAMATH_GPT_inequality_solution_l902_90215


namespace NUMINAMATH_GPT_lesser_number_l902_90244

theorem lesser_number (x y : ℕ) (h1: x + y = 60) (h2: x - y = 10) : y = 25 :=
sorry

end NUMINAMATH_GPT_lesser_number_l902_90244


namespace NUMINAMATH_GPT_solution_set_of_inequality_l902_90259

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 4 * x - 3 > 0 } = { x : ℝ | 1 < x ∧ x < 3 } := sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l902_90259


namespace NUMINAMATH_GPT_find_f_2_l902_90209

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

-- The statement to prove: if f is monotonically increasing and satisfies the functional equation
-- for all x, then f(2) = e^2 + 1.
theorem find_f_2
  (h_mono : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2)
  (h_eq : ∀ x : ℝ, f (f x - exp x) = exp 1 + 1) :
  f 2 = exp 2 + 1 := sorry

end NUMINAMATH_GPT_find_f_2_l902_90209


namespace NUMINAMATH_GPT_lisa_marbles_l902_90274

def ConnieMarbles : ℕ := 323
def JuanMarbles (ConnieMarbles : ℕ) : ℕ := ConnieMarbles + 175
def MarkMarbles (JuanMarbles : ℕ) : ℕ := 3 * JuanMarbles
def LisaMarbles (MarkMarbles : ℕ) : ℕ := MarkMarbles / 2 - 200

theorem lisa_marbles :
  LisaMarbles (MarkMarbles (JuanMarbles ConnieMarbles)) = 547 := by
  sorry

end NUMINAMATH_GPT_lisa_marbles_l902_90274


namespace NUMINAMATH_GPT_find_a2_plus_b2_l902_90213

theorem find_a2_plus_b2 (a b : ℝ) :
  (∀ x, |a * Real.sin x + b * Real.cos x - 1| + |b * Real.sin x - a * Real.cos x| ≤ 11)
  → a^2 + b^2 = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_a2_plus_b2_l902_90213


namespace NUMINAMATH_GPT_seq_fifth_term_l902_90295

def seq (a : ℕ → ℤ) : Prop :=
  (a 1 = 3) ∧ (a 2 = 6) ∧ (∀ n : ℕ, a (n + 2) = a (n + 1) - a n)

theorem seq_fifth_term (a : ℕ → ℤ) (h : seq a) : a 5 = -6 :=
by
  sorry

end NUMINAMATH_GPT_seq_fifth_term_l902_90295


namespace NUMINAMATH_GPT_hyperbola_sufficient_not_necessary_condition_l902_90281

-- Define the equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 16 = 1

-- Define the asymptotic line equations of the hyperbola
def asymptotes_eq (x y : ℝ) : Prop :=
  y = 2 * x ∨ y = -2 * x

-- Prove that the equation of the hyperbola is a sufficient but not necessary condition for the asymptotic lines
theorem hyperbola_sufficient_not_necessary_condition :
  (∀ x y : ℝ, hyperbola_eq x y → asymptotes_eq x y) ∧ ¬ (∀ x y : ℝ, asymptotes_eq x y → hyperbola_eq x y) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_sufficient_not_necessary_condition_l902_90281


namespace NUMINAMATH_GPT_percentage_of_invalid_votes_l902_90270

theorem percentage_of_invalid_votes:
  ∃ (A B V I VV : ℕ), 
    V = 5720 ∧
    B = 1859 ∧
    A = B + 15 / 100 * V ∧
    VV = A + B ∧
    V = VV + I ∧
    (I: ℚ) / V * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_invalid_votes_l902_90270


namespace NUMINAMATH_GPT_smallest_three_digit_solution_l902_90266

theorem smallest_three_digit_solution :
  ∃ n : ℕ, 70 * n ≡ 210 [MOD 350] ∧ 100 ≤ n ∧ n = 103 :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_solution_l902_90266


namespace NUMINAMATH_GPT_range_of_a_minimize_S_l902_90280

open Real

-- Problem 1: Prove the range of a 
theorem range_of_a (a : ℝ) : (∃ x ≠ 0, x^3 - 3*x^2 + (2 - a)*x = 0) ↔ a > -1 / 4 := sorry

-- Problem 2: Prove the minimizing value of a for the area function S(a)
noncomputable def S (a : ℝ) : ℝ := 
  let α := sorry -- α is the root depending on a (to be determined from the context)
  let β := sorry -- β is the root depending on a (to be determined from the context)
  (1/4 * α^4 - α^3 + (1/2) * (2-a) * α^2) + (1/4 * β^4 - β^3 + (1/2) * (2-a) * β^2)

theorem minimize_S (a : ℝ) : a = 38 - 27 * sqrt 2 → S a = S (38 - 27 * sqrt 2) := sorry

end NUMINAMATH_GPT_range_of_a_minimize_S_l902_90280


namespace NUMINAMATH_GPT_cooling_time_condition_l902_90279

theorem cooling_time_condition :
  ∀ (θ0 θ1 θ1' θ0' : ℝ) (t : ℝ), 
    θ0 = 20 → θ1 = 100 → θ1' = 60 → θ0' = 20 →
    let θ := θ0 + (θ1 - θ0) * Real.exp (-t / 4)
    let θ' := θ0' + (θ1' - θ0') * Real.exp (-t / 4)
    (θ - θ' ≤ 10) → (t ≥ 5.52) :=
sorry

end NUMINAMATH_GPT_cooling_time_condition_l902_90279


namespace NUMINAMATH_GPT_john_billed_for_28_minutes_l902_90200

variable (monthlyFee : ℝ) (costPerMinute : ℝ) (totalBill : ℝ)
variable (minutesBilled : ℝ)

def is_billed_correctly (monthlyFee totalBill costPerMinute minutesBilled : ℝ) : Prop :=
  totalBill - monthlyFee = minutesBilled * costPerMinute ∧ minutesBilled = 28

theorem john_billed_for_28_minutes : 
  is_billed_correctly 5 12.02 0.25 28 := 
by
  sorry

end NUMINAMATH_GPT_john_billed_for_28_minutes_l902_90200


namespace NUMINAMATH_GPT_find_4_oplus_2_l902_90284

def operation (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem find_4_oplus_2 : operation 4 2 = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_4_oplus_2_l902_90284


namespace NUMINAMATH_GPT_find_y_l902_90230

theorem find_y (y : ℕ) (h1 : 27 = 3^3) (h2 : 3^9 = 27^y) : y = 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_l902_90230


namespace NUMINAMATH_GPT_max_ratio_MO_MF_on_parabola_l902_90207

theorem max_ratio_MO_MF_on_parabola (F M : ℝ × ℝ) : 
  let O := (0, 0)
  let focus := (1 / 2, 0)
  ∀ (M : ℝ × ℝ), (M.snd ^ 2 = 2 * M.fst) →
  F = focus →
  (∃ m > 0, M.fst = m ∧ M.snd ^ 2 = 2 * m) →
  (∃ t, t = m - (1 / 4)) →
  ∃ value, value = (2 * Real.sqrt 3) / 3 ∧
  ∃ rat, rat = dist M O / dist M F ∧
  rat = value := 
by
  admit

end NUMINAMATH_GPT_max_ratio_MO_MF_on_parabola_l902_90207


namespace NUMINAMATH_GPT_total_time_is_correct_l902_90251

-- Defining the number of items
def chairs : ℕ := 7
def tables : ℕ := 3
def bookshelves : ℕ := 2
def lamps : ℕ := 4

-- Defining the time spent on each type of furniture
def time_per_chair : ℕ := 4
def time_per_table : ℕ := 8
def time_per_bookshelf : ℕ := 12
def time_per_lamp : ℕ := 2

-- Defining the total time calculation
def total_time : ℕ :=
  (chairs * time_per_chair) + 
  (tables * time_per_table) +
  (bookshelves * time_per_bookshelf) +
  (lamps * time_per_lamp)

-- Theorem stating the total time
theorem total_time_is_correct : total_time = 84 :=
by
  -- Skipping the proof details
  sorry

end NUMINAMATH_GPT_total_time_is_correct_l902_90251


namespace NUMINAMATH_GPT_simplify_expression_l902_90249

theorem simplify_expression : 
  (1 / ((1 / ((1 / 2)^1)) + (1 / ((1 / 2)^3)) + (1 / ((1 / 2)^4)))) = (1 / 26) := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l902_90249


namespace NUMINAMATH_GPT_find_a5_l902_90219

theorem find_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n^2 + 1) 
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) 
  (h3 : S 1 = 2) :
  a 5 = 9 :=
sorry

end NUMINAMATH_GPT_find_a5_l902_90219


namespace NUMINAMATH_GPT_arithmetic_sequence_20th_term_l902_90237

theorem arithmetic_sequence_20th_term :
  let a := 2
  let d := 5
  let n := 20
  let a_n := a + (n - 1) * d
  a_n = 97 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_20th_term_l902_90237


namespace NUMINAMATH_GPT_minimum_value_of_F_l902_90245

noncomputable def F (m n : ℝ) : ℝ := (m - n)^2 + (m^2 - n + 1)^2

theorem minimum_value_of_F : 
  (∀ m n : ℝ, F m n ≥ 9 / 32) ∧ (∃ m n : ℝ, F m n = 9 / 32) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_F_l902_90245


namespace NUMINAMATH_GPT_triangle_side_length_l902_90298

theorem triangle_side_length 
  (X Z : ℝ) (x z y : ℝ)
  (h1 : x = 36)
  (h2 : z = 72)
  (h3 : Z = 4 * X) :
  y = 72 := by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l902_90298


namespace NUMINAMATH_GPT_number_of_zeros_l902_90264

noncomputable def f (x : Real) : Real :=
if x > 0 then -1 + Real.log x
else 3 * x + 4

theorem number_of_zeros : (∃ a b : Real, f a = 0 ∧ f b = 0 ∧ a ≠ b) := 
sorry

end NUMINAMATH_GPT_number_of_zeros_l902_90264


namespace NUMINAMATH_GPT_asymptote_slope_l902_90220

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 81 = 1

-- Lean statement to prove slope of asymptotes
theorem asymptote_slope :
  (∀ x y : ℝ, hyperbola x y → (y/x) = 3/4 ∨ (y/x) = -(3/4)) :=
by
  sorry

end NUMINAMATH_GPT_asymptote_slope_l902_90220


namespace NUMINAMATH_GPT_supply_without_leak_last_for_20_days_l902_90232

variable (C V : ℝ)

-- Condition 1: if there is a 10-liter leak per day, the supply lasts for 15 days
axiom h1 : C = 15 * (V + 10)

-- Condition 2: if there is a 20-liter leak per day, the supply lasts for 12 days
axiom h2 : C = 12 * (V + 20)

-- The problem to prove: without any leak, the tank can supply water to the village for 20 days
theorem supply_without_leak_last_for_20_days (C V : ℝ) (h1 : C = 15 * (V + 10)) (h2 : C = 12 * (V + 20)) : C / V = 20 := 
by 
  sorry

end NUMINAMATH_GPT_supply_without_leak_last_for_20_days_l902_90232


namespace NUMINAMATH_GPT_max_students_for_distribution_l902_90247

theorem max_students_for_distribution : 
  ∃ (n : Nat), (∀ k, k ∣ 1048 ∧ k ∣ 828 → k ≤ n) ∧ 
               (n ∣ 1048 ∧ n ∣ 828) ∧ 
               n = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_students_for_distribution_l902_90247


namespace NUMINAMATH_GPT_profit_is_5000_l902_90205

namespace HorseshoeProfit

-- Defining constants and conditions
def initialOutlay : ℝ := 10000
def costPerSet : ℝ := 20
def sellingPricePerSet : ℝ := 50
def numberOfSets : ℝ := 500

-- Calculating the profit
def profit : ℝ :=
  let revenue := numberOfSets * sellingPricePerSet
  let manufacturingCosts := initialOutlay + (costPerSet * numberOfSets)
  revenue - manufacturingCosts

-- The main theorem: the profit is $5,000
theorem profit_is_5000 : profit = 5000 := by
  sorry

end HorseshoeProfit

end NUMINAMATH_GPT_profit_is_5000_l902_90205


namespace NUMINAMATH_GPT_sum_of_five_consecutive_odd_integers_l902_90228

theorem sum_of_five_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 8) = 156) :
  n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 390 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_five_consecutive_odd_integers_l902_90228


namespace NUMINAMATH_GPT_smallest_a_for_nonprime_l902_90290

theorem smallest_a_for_nonprime (a : ℕ) : (∀ x : ℤ, ∃ d : ℤ, d ∣ (x^4 + a^4) ∧ d ≠ 1 ∧ d ≠ (x^4 + a^4)) ↔ a = 3 := by
  sorry

end NUMINAMATH_GPT_smallest_a_for_nonprime_l902_90290


namespace NUMINAMATH_GPT_exists_non_regular_triangle_with_similar_medians_as_sides_l902_90294

theorem exists_non_regular_triangle_with_similar_medians_as_sides 
  (a b c : ℝ) 
  (s_a s_b s_c : ℝ)
  (h1 : 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2)
  (h2 : 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2)
  (h3 : 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2)
  (similarity_cond : (2*c^2 + 2*b^2 - a^2) / c^2 = (2*c^2 + 2*a^2 - b^2) / b^2 ∧ (2*c^2 + 2*a^2 - b^2) / b^2 = (2*a^2 + 2*b^2 - c^2) / a^2) :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (∃ (s_a s_b s_c : ℝ), 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2 ∧ 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2 ∧ 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2) ∧
  ((2*c^2 + 2*b^2 - a^2) / c^2 = (2*c^2 + 2*a^2 - b^2) / b^2 ∧ (2*c^2 + 2*a^2 - b^2) / b^2 = (2*a^2 + 2*b^2 - c^2) / a^2) :=
sorry

end NUMINAMATH_GPT_exists_non_regular_triangle_with_similar_medians_as_sides_l902_90294


namespace NUMINAMATH_GPT_deficit_percentage_l902_90272

variable (A B : ℝ) -- Actual lengths of the sides of the rectangle
variable (x : ℝ) -- Percentage in deficit
variable (measuredA := A * 1.06) -- One side measured 6% in excess
variable (errorPercent := 0.7) -- Error percent in area
variable (measuredB := B * (1 - x / 100)) -- Other side measured x% in deficit
variable (actualArea := A * B) -- Actual area of the rectangle
variable (calculatedArea := (A * 1.06) * (B * (1 - x / 100))) -- Calculated area with measurement errors
variable (correctArea := actualArea * (1 + errorPercent / 100)) -- Correct area considering the error

theorem deficit_percentage : 
  calculatedArea = correctArea → 
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_deficit_percentage_l902_90272


namespace NUMINAMATH_GPT_select_3_males_2_females_select_at_least_1_captain_select_at_least_1_female_select_both_captain_and_female_l902_90221

variable (n m : ℕ) -- n for males, m for females
variable (mc fc : ℕ) -- mc for male captain, fc for female captain

def num_ways_3_males_2_females : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 4 2)

def num_ways_at_least_1_captain : ℕ :=
  (2 * (Nat.choose 8 4)) + (Nat.choose 8 3)

def num_ways_at_least_1_female : ℕ :=
  (Nat.choose 10 5) - (Nat.choose 6 5)

def num_ways_both_captain_and_female : ℕ :=
  (Nat.choose 10 5) - (Nat.choose 8 5) - (Nat.choose 5 4)

theorem select_3_males_2_females : num_ways_3_males_2_females = 120 := by
  sorry
  
theorem select_at_least_1_captain : num_ways_at_least_1_captain = 196 := by
  sorry
  
theorem select_at_least_1_female : num_ways_at_least_1_female = 246 := by
  sorry
  
theorem select_both_captain_and_female : num_ways_both_captain_and_female = 191 := by
  sorry

end NUMINAMATH_GPT_select_3_males_2_females_select_at_least_1_captain_select_at_least_1_female_select_both_captain_and_female_l902_90221


namespace NUMINAMATH_GPT_sum_last_two_digits_is_correct_l902_90231

def fibs : List Nat := [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

def factorial_last_two_digits (n : Nat) : Nat :=
  (Nat.factorial n) % 100

def modified_fib_factorial_series : List Nat :=
  fibs.map (λ k => (factorial_last_two_digits k + 2) % 100)

def sum_last_two_digits : Nat :=
  (modified_fib_factorial_series.sum) % 100

theorem sum_last_two_digits_is_correct :
  sum_last_two_digits = 14 :=
sorry

end NUMINAMATH_GPT_sum_last_two_digits_is_correct_l902_90231


namespace NUMINAMATH_GPT_min_cost_per_ounce_l902_90288

theorem min_cost_per_ounce 
  (cost_40 : ℝ := 200) (cost_90 : ℝ := 400)
  (percentage_40 : ℝ := 0.4) (percentage_90 : ℝ := 0.9)
  (desired_percentage : ℝ := 0.5) :
  (∀ (x y : ℝ), 0.4 * x + 0.9 * y = 0.5 * (x + y) → 200 * x + 400 * y / (x + y) = 240) :=
sorry

end NUMINAMATH_GPT_min_cost_per_ounce_l902_90288


namespace NUMINAMATH_GPT_find_matrix_N_l902_90239

theorem find_matrix_N (N : Matrix (Fin 4) (Fin 4) ℤ)
  (hi : N.mulVec ![1, 0, 0, 0] = ![3, 4, -9, 1])
  (hj : N.mulVec ![0, 1, 0, 0] = ![-1, 6, -3, 2])
  (hk : N.mulVec ![0, 0, 1, 0] = ![8, -2, 5, 0])
  (hl : N.mulVec ![0, 0, 0, 1] = ![1, 0, 7, -1]) :
  N = ![![3, -1, 8, 1],
         ![4, 6, -2, 0],
         ![-9, -3, 5, 7],
         ![1, 2, 0, -1]] := by
  sorry

end NUMINAMATH_GPT_find_matrix_N_l902_90239


namespace NUMINAMATH_GPT_circle_center_l902_90285

theorem circle_center (x y : ℝ) :
  x^2 + y^2 - 4 * x - 2 * y - 5 = 0 → (x - 2)^2 + (y - 1)^2 = 10 :=
by sorry

end NUMINAMATH_GPT_circle_center_l902_90285


namespace NUMINAMATH_GPT_number_of_months_to_fully_pay_off_car_l902_90267

def total_price : ℕ := 13380
def initial_payment : ℕ := 5400
def monthly_payment : ℕ := 420

theorem number_of_months_to_fully_pay_off_car :
  (total_price - initial_payment) / monthly_payment = 19 :=
by
  sorry

end NUMINAMATH_GPT_number_of_months_to_fully_pay_off_car_l902_90267


namespace NUMINAMATH_GPT_probability_letter_in_MATHEMATICS_l902_90292

theorem probability_letter_in_MATHEMATICS :
  let alphabet := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  let mathematics := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S']
  (mathematics.length : ℚ) / (alphabet.length : ℚ) = 4 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_letter_in_MATHEMATICS_l902_90292


namespace NUMINAMATH_GPT_selection_methods_count_l902_90299

noncomputable def num_selection_methods (total_students chosen_students : ℕ) (A B : ℕ) : ℕ :=
  let with_A_and_B := Nat.choose (total_students - 2) (chosen_students - 2)
  let with_one_A_or_B := Nat.choose (total_students - 2) (chosen_students - 1) * Nat.choose 2 1
  with_A_and_B + with_one_A_or_B

theorem selection_methods_count :
  num_selection_methods 10 4 1 2 = 140 :=
by
  -- We can add detailed proof here, for now we provide a placeholder
  sorry

end NUMINAMATH_GPT_selection_methods_count_l902_90299


namespace NUMINAMATH_GPT_max_m_for_inequality_min_4a2_9b2_c2_l902_90261

theorem max_m_for_inequality (m : ℝ) : (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 := 
sorry

theorem min_4a2_9b2_c2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (4 * a^2 + 9 * b^2 + c^2) = 36 / 49 ∧ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49 :=
sorry

end NUMINAMATH_GPT_max_m_for_inequality_min_4a2_9b2_c2_l902_90261


namespace NUMINAMATH_GPT_min_distance_between_graphs_l902_90235

noncomputable def minimum_distance (a : ℝ) (h : 1 < a) : ℝ :=
  if h1 : a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / (Real.log a)

theorem min_distance_between_graphs (a : ℝ) (h1 : 1 < a) :
  minimum_distance a h1 = 
  if a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / (Real.log a) :=
by
  intros
  sorry

end NUMINAMATH_GPT_min_distance_between_graphs_l902_90235


namespace NUMINAMATH_GPT_area_of_fourth_rectangle_l902_90282

theorem area_of_fourth_rectangle (A B C D E F G H I J K L : Type) 
  (x y z w : ℕ) (a1 : x * y = 20) (a2 : x * w = 12) (a3 : z * w = 16) : 
  y * w = 16 :=
by sorry

end NUMINAMATH_GPT_area_of_fourth_rectangle_l902_90282


namespace NUMINAMATH_GPT_hypotenuse_min_length_l902_90255

theorem hypotenuse_min_length
  (a b l : ℝ)
  (h_area : (1/2) * a * b = 8)
  (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = l)
  (h_min_l : l = 8 + 4 * Real.sqrt 2) :
  Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_min_length_l902_90255


namespace NUMINAMATH_GPT_prime_sum_product_l902_90206

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 91) : p * q = 178 := 
by
  sorry

end NUMINAMATH_GPT_prime_sum_product_l902_90206


namespace NUMINAMATH_GPT_solve_for_x_l902_90238

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l902_90238


namespace NUMINAMATH_GPT_zoo_animal_difference_l902_90212

variable (giraffes non_giraffes : ℕ)

theorem zoo_animal_difference (h1 : giraffes = 300) (h2 : giraffes = 3 * non_giraffes) : giraffes - non_giraffes = 200 :=
by 
  sorry

end NUMINAMATH_GPT_zoo_animal_difference_l902_90212


namespace NUMINAMATH_GPT_max_y_value_l902_90276

theorem max_y_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = (x - y) / (x + 3 * y)) : y ≤ 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_y_value_l902_90276


namespace NUMINAMATH_GPT_factor_polynomial_l902_90243

def A (x : ℝ) : ℝ := x^2 + 5 * x + 3
def B (x : ℝ) : ℝ := x^2 + 9 * x + 20
def C (x : ℝ) : ℝ := x^2 + 7 * x - 8

theorem factor_polynomial (x : ℝ) :
  (A x) * (B x) + (C x) = (x^2 + 7 * x + 8) * (x^2 + 7 * x + 14) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l902_90243


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l902_90263

theorem simplify_and_evaluate_expression (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 2) : 
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) :=
by
  sorry

example : (∃ x : ℝ, x ≠ 1 ∧ x ≠ 2 ∧ (x = 3) ∧ ((x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = 5)) :=
  ⟨3, by norm_num, by norm_num, rfl, by norm_num⟩

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l902_90263


namespace NUMINAMATH_GPT_divisibility_equiv_l902_90229

-- Definition of the functions a(n) and b(n)
def a (n : ℕ) := n^5 + 5^n
def b (n : ℕ) := n^5 * 5^n + 1

-- Define a positive integer
variables (n : ℕ) (hn : n > 0)

-- The theorem stating the equivalence
theorem divisibility_equiv : (a n) % 11 = 0 ↔ (b n) % 11 = 0 :=
sorry
 
end NUMINAMATH_GPT_divisibility_equiv_l902_90229


namespace NUMINAMATH_GPT_bullet_speed_difference_l902_90286

def bullet_speed_in_same_direction (v_h v_b : ℝ) : ℝ :=
  v_b + v_h

def bullet_speed_in_opposite_direction (v_h v_b : ℝ) : ℝ :=
  v_b - v_h

theorem bullet_speed_difference (v_h v_b : ℝ) (h_h : v_h = 20) (h_b : v_b = 400) :
  bullet_speed_in_same_direction v_h v_b - bullet_speed_in_opposite_direction v_h v_b = 40 :=
by
  rw [h_h, h_b]
  sorry

end NUMINAMATH_GPT_bullet_speed_difference_l902_90286


namespace NUMINAMATH_GPT_polynomial_root_problem_l902_90234

theorem polynomial_root_problem (a b c d : ℤ) (r1 r2 r3 r4 : ℕ)
  (h_roots : ∀ x, x^4 + a * x^3 + b * x^2 + c * x + d = (x + r1) * (x + r2) * (x + r3) * (x + r4))
  (h_sum : a + b + c + d = 2009) :
  d = 528 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_root_problem_l902_90234


namespace NUMINAMATH_GPT_investment_return_l902_90260

theorem investment_return (y_r : ℝ) :
  (500 + 1500) * 0.085 = 500 * 0.07 + 1500 * y_r → y_r = 0.09 :=
by
  sorry

end NUMINAMATH_GPT_investment_return_l902_90260


namespace NUMINAMATH_GPT_product_ABCD_is_9_l902_90258

noncomputable def A : ℝ := Real.sqrt 2018 + Real.sqrt 2019 + 1
noncomputable def B : ℝ := -Real.sqrt 2018 - Real.sqrt 2019 - 1
noncomputable def C : ℝ := Real.sqrt 2018 - Real.sqrt 2019 + 1
noncomputable def D : ℝ := Real.sqrt 2019 - Real.sqrt 2018 + 1

theorem product_ABCD_is_9 : A * B * C * D = 9 :=
by sorry

end NUMINAMATH_GPT_product_ABCD_is_9_l902_90258


namespace NUMINAMATH_GPT_factorization_correct_l902_90254

theorem factorization_correct (x : ℝ) : 2 * x^2 - 6 * x - 8 = 2 * (x - 4) * (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l902_90254


namespace NUMINAMATH_GPT_fraction_multiplication_l902_90296

noncomputable def a : ℚ := 5 / 8
noncomputable def b : ℚ := 7 / 12
noncomputable def c : ℚ := 3 / 7
noncomputable def n : ℚ := 1350

theorem fraction_multiplication : a * b * c * n = 210.9375 := by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_l902_90296


namespace NUMINAMATH_GPT_fraction_increase_by_3_l902_90287

theorem fraction_increase_by_3 (x y : ℝ) (h₁ : x' = 3 * x) (h₂ : y' = 3 * y) : 
  (x' * y') / (x' - y') = 3 * (x * y) / (x - y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_increase_by_3_l902_90287


namespace NUMINAMATH_GPT_calculate_flat_tax_l902_90248

open Real

def price_per_sq_ft (property: String) : Real :=
  if property = "Condo" then 98
  else if property = "BarnHouse" then 84
  else if property = "DetachedHouse" then 102
  else if property = "Townhouse" then 96
  else if property = "Garage" then 60
  else if property = "PoolArea" then 50
  else 0

def area_in_sq_ft (property: String) : Real :=
  if property = "Condo" then 2400
  else if property = "BarnHouse" then 1200
  else if property = "DetachedHouse" then 3500
  else if property = "Townhouse" then 2750
  else if property = "Garage" then 480
  else if property = "PoolArea" then 600
  else 0

def total_value : Real :=
  (price_per_sq_ft "Condo" * area_in_sq_ft "Condo") +
  (price_per_sq_ft "BarnHouse" * area_in_sq_ft "BarnHouse") +
  (price_per_sq_ft "DetachedHouse" * area_in_sq_ft "DetachedHouse") +
  (price_per_sq_ft "Townhouse" * area_in_sq_ft "Townhouse") +
  (price_per_sq_ft "Garage" * area_in_sq_ft "Garage") +
  (price_per_sq_ft "PoolArea" * area_in_sq_ft "PoolArea")

def tax_rate : Real := 0.0125

theorem calculate_flat_tax : total_value * tax_rate = 12697.50 := by
  sorry

end NUMINAMATH_GPT_calculate_flat_tax_l902_90248


namespace NUMINAMATH_GPT_semicircle_radius_l902_90218

theorem semicircle_radius (P L W : ℝ) (π : Real) (r : ℝ) 
  (hP : P = 144) (hL : L = 48) (hW : W = 24) (hD : ∃ d, d = 2 * r ∧ d = L) :
  r = 48 / (π + 2) := 
by
  sorry

end NUMINAMATH_GPT_semicircle_radius_l902_90218


namespace NUMINAMATH_GPT_payment_denotation_is_correct_l902_90269

-- Define the initial condition of receiving money
def received_amount : ℤ := 120

-- Define the payment amount
def payment_amount : ℤ := 85

-- The expected payoff
def expected_payment_denotation : ℤ := -85

-- Theorem stating that the payment should be denoted as -85 yuan
theorem payment_denotation_is_correct : (payment_amount = -expected_payment_denotation) :=
by
  sorry

end NUMINAMATH_GPT_payment_denotation_is_correct_l902_90269


namespace NUMINAMATH_GPT_non_equivalent_paintings_wheel_l902_90277

theorem non_equivalent_paintings_wheel :
  let num_sections := 7
  let num_colors := 2
  let total_paintings := num_colors ^ num_sections
  let single_color_cases := 2
  let non_single_color_paintings := total_paintings - single_color_cases
  let equivalent_rotation_count := num_sections
  (non_single_color_paintings / equivalent_rotation_count) + single_color_cases = 20 :=
by
  let num_sections := 7
  let num_colors := 2
  let total_paintings := num_colors ^ num_sections
  let single_color_cases := 2
  let non_single_color_paintings := total_paintings - single_color_cases
  let equivalent_rotation_count := num_sections
  have h1 := (non_single_color_paintings / equivalent_rotation_count) + single_color_cases
  sorry

end NUMINAMATH_GPT_non_equivalent_paintings_wheel_l902_90277


namespace NUMINAMATH_GPT_vector_dot_product_zero_implies_orthogonal_l902_90275

theorem vector_dot_product_zero_implies_orthogonal
  (a b : ℝ → ℝ)
  (h0 : ∀ (x y : ℝ), a x * b y = 0) :
  ¬(a = 0 ∨ b = 0) := 
sorry

end NUMINAMATH_GPT_vector_dot_product_zero_implies_orthogonal_l902_90275


namespace NUMINAMATH_GPT_first_month_sale_l902_90271

def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029
def sale6 : ℕ := 4937
def average_sale : ℕ := 5600

theorem first_month_sale :
  let total_sales := average_sale * 6
  let known_sales := sale2 + sale3 + sale4 + sale5 + sale6
  let sale1 := total_sales - known_sales
  sale1 = 5266 :=
by
  sorry

end NUMINAMATH_GPT_first_month_sale_l902_90271


namespace NUMINAMATH_GPT_find_sums_of_integers_l902_90297

theorem find_sums_of_integers (x y : ℤ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_prod_sum : x * y + x + y = 125) (h_rel_prime : Int.gcd x y = 1) (h_lt_x : x < 30) (h_lt_y : y < 30) : 
  (x + y = 25) ∨ (x + y = 23) ∨ (x + y = 21) := 
by 
  sorry

end NUMINAMATH_GPT_find_sums_of_integers_l902_90297


namespace NUMINAMATH_GPT_pizza_order_cost_l902_90252

def base_cost_per_pizza : ℕ := 10
def cost_per_topping : ℕ := 1
def topping_count_pepperoni : ℕ := 1
def topping_count_sausage : ℕ := 1
def topping_count_black_olive_and_mushroom : ℕ := 2
def tip : ℕ := 5

theorem pizza_order_cost :
  3 * base_cost_per_pizza + (topping_count_pepperoni * cost_per_topping) + (topping_count_sausage * cost_per_topping) + (topping_count_black_olive_and_mushroom * cost_per_topping) + tip = 39 := by
  sorry

end NUMINAMATH_GPT_pizza_order_cost_l902_90252


namespace NUMINAMATH_GPT_fraction_sum_reciprocal_ge_two_l902_90283

theorem fraction_sum_reciprocal_ge_two (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  (a / b) + (b / a) ≥ 2 :=
sorry

end NUMINAMATH_GPT_fraction_sum_reciprocal_ge_two_l902_90283


namespace NUMINAMATH_GPT_smallest_sum_of_4_numbers_l902_90253

noncomputable def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

noncomputable def not_relatively_prime (a b : ℕ) : Prop :=
  ¬ relatively_prime a b

noncomputable def problem_statement : Prop :=
  ∃ (V1 V2 V3 V4 : ℕ), 
  relatively_prime V1 V3 ∧ 
  relatively_prime V2 V4 ∧ 
  not_relatively_prime V1 V2 ∧ 
  not_relatively_prime V1 V4 ∧ 
  not_relatively_prime V2 V3 ∧ 
  not_relatively_prime V3 V4 ∧ 
  V1 + V2 + V3 + V4 = 60

theorem smallest_sum_of_4_numbers : problem_statement := sorry

end NUMINAMATH_GPT_smallest_sum_of_4_numbers_l902_90253


namespace NUMINAMATH_GPT_hours_per_day_l902_90224

-- Define the parameters
def A1 := 57
def D1 := 12
def H2 := 6
def A2 := 30
def D2 := 19

-- Define the target Equation
theorem hours_per_day :
  A1 * D1 * H = A2 * D2 * H2 → H = 5 :=
by
  sorry

end NUMINAMATH_GPT_hours_per_day_l902_90224


namespace NUMINAMATH_GPT_iron_heating_time_l902_90208

-- Define the conditions as constants
def ironHeatingRate : ℝ := 9 -- degrees Celsius per 20 seconds
def ironCoolingRate : ℝ := 15 -- degrees Celsius per 30 seconds
def coolingTime : ℝ := 180 -- seconds

-- Define the theorem to prove the heating back time
theorem iron_heating_time :
  (coolingTime / 30) * ironCoolingRate = 90 →
  (90 / ironHeatingRate) * 20 = 200 :=
by
  sorry

end NUMINAMATH_GPT_iron_heating_time_l902_90208


namespace NUMINAMATH_GPT_range_of_m_l902_90226

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, abs (x - 3) + abs (x - m) < 5) : -2 < m ∧ m < 8 :=
  sorry

end NUMINAMATH_GPT_range_of_m_l902_90226


namespace NUMINAMATH_GPT_sum_of_possible_values_l902_90202

noncomputable def solution : ℕ :=
  sorry

theorem sum_of_possible_values (x : ℝ) (h : |x - 5| - 4 = 0) : solution = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_l902_90202
