import Mathlib

namespace heartsuit_calc_l43_43872

-- Define the operation x ♡ y = 4x + 6y
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_calc : heartsuit 5 3 = 38 := by
  -- Proof omitted
  sorry

end heartsuit_calc_l43_43872


namespace neces_not_suff_cond_l43_43439

theorem neces_not_suff_cond (a : ℝ) (h : a ≠ 0) : (1 / a < 1) → (a > 1) :=
sorry

end neces_not_suff_cond_l43_43439


namespace wine_price_increase_l43_43868

-- Definitions translating the conditions
def wine_cost_today : ℝ := 20.0
def bottles_count : ℕ := 5
def tariff_rate : ℝ := 0.25

-- Statement to prove
theorem wine_price_increase (wine_cost_today : ℝ) (bottles_count : ℕ) (tariff_rate : ℝ) : 
  bottles_count * wine_cost_today * tariff_rate = 25.0 := 
by
  -- Proof is omitted
  sorry

end wine_price_increase_l43_43868


namespace line_equation_l43_43345

theorem line_equation (x y : ℝ) (c : ℝ)
  (h1 : 2 * x - y + 3 = 0)
  (h2 : 4 * x + 3 * y + 1 = 0)
  (h3 : 3 * x + 2 * y + c = 0) :
  c = 1 := sorry

end line_equation_l43_43345


namespace union_of_sets_l43_43113

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Prove that A ∪ B = {x | -1 < x ∧ x ≤ 2}
theorem union_of_sets (x : ℝ) : x ∈ (A ∪ B) ↔ x ∈ {x | -1 < x ∧ x ≤ 2} :=
by
  sorry

end union_of_sets_l43_43113


namespace boxes_with_neither_l43_43457

def total_boxes : ℕ := 15
def boxes_with_crayons : ℕ := 9
def boxes_with_markers : ℕ := 6
def boxes_with_both : ℕ := 4

theorem boxes_with_neither : total_boxes - (boxes_with_crayons + boxes_with_markers - boxes_with_both) = 4 := by
  sorry

end boxes_with_neither_l43_43457


namespace remaining_amount_is_12_l43_43046

-- Define initial amount and amount spent
def initial_amount : ℕ := 90
def amount_spent : ℕ := 78

-- Define the remaining amount after spending
def remaining_amount : ℕ := initial_amount - amount_spent

-- Theorem asserting the remaining amount is 12
theorem remaining_amount_is_12 : remaining_amount = 12 :=
by
  -- Proof omitted
  sorry

end remaining_amount_is_12_l43_43046


namespace interest_rate_per_annum_l43_43191

variable (P : ℝ := 1200) (T : ℝ := 1) (diff : ℝ := 2.999999999999936) (r : ℝ)
noncomputable def SI (P : ℝ) (r : ℝ) (T : ℝ) : ℝ := P * r * T
noncomputable def CI (P : ℝ) (r : ℝ) (T : ℝ) : ℝ := P * ((1 + r / 2) ^ (2 * T) - 1)

theorem interest_rate_per_annum :
  CI P r T - SI P r T = diff → r = 0.1 :=
by
  -- Proof to be provided
  sorry

end interest_rate_per_annum_l43_43191


namespace sum_inequality_l43_43482

noncomputable def f (x : ℝ) : ℝ :=
  (3 * x^2 - x) / (1 + x^2)

theorem sum_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 1) : 
  f x + f y + f z ≥ 0 :=
by
  sorry

end sum_inequality_l43_43482


namespace calendar_sum_l43_43452

theorem calendar_sum (n : ℕ) : 
    n + (n + 7) + (n + 14) = 3 * n + 21 :=
by sorry

end calendar_sum_l43_43452


namespace inequality_has_no_solutions_l43_43282

theorem inequality_has_no_solutions (x : ℝ) : ¬ (3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  sorry
}

end inequality_has_no_solutions_l43_43282


namespace value_of_f5_f_neg5_l43_43336

-- Define the function f
def f (x a b : ℝ) : ℝ := x^5 - a * x^3 + b * x + 2

-- Given conditions
variable (a b : ℝ)
axiom h1 : f (-5) a b = 3

-- The proposition to prove
theorem value_of_f5_f_neg5 : f 5 a b + f (-5) a b = 4 :=
by
  -- Include the result of the proof
  sorry

end value_of_f5_f_neg5_l43_43336


namespace difference_of_two_smallest_integers_divisors_l43_43427

theorem difference_of_two_smallest_integers_divisors (n m : ℕ) (h₁ : n > 1) (h₂ : m > 1) 
(h₃ : n % 2 = 1) (h₄ : n % 3 = 1) (h₅ : n % 4 = 1) (h₆ : n % 5 = 1) 
(h₇ : n % 6 = 1) (h₈ : n % 7 = 1) (h₉ : n % 8 = 1) (h₁₀ : n % 9 = 1) 
(h₁₁ : n % 10 = 1) (h₃' : m % 2 = 1) (h₄' : m % 3 = 1) (h₅' : m % 4 = 1) 
(h₆' : m % 5 = 1) (h₇' : m % 6 = 1) (h₈' : m % 7 = 1) (h₉' : m % 8 = 1) 
(h₁₀' : m % 9 = 1) (h₁₁' : m % 10 = 1): m - n = 2520 :=
sorry

end difference_of_two_smallest_integers_divisors_l43_43427


namespace rice_mixture_price_l43_43092

-- Defining the costs per kg for each type of rice
def rice_cost1 : ℝ := 16
def rice_cost2 : ℝ := 24

-- Defining the given ratio
def mixing_ratio : ℝ := 3

-- Main theorem stating the problem
theorem rice_mixture_price
  (x : ℝ)  -- The common measure of quantity in the ratio
  (h1 : 3 * x * rice_cost1 + x * rice_cost2 = 72 * x)
  (h2 : 3 * x + x = 4 * x) :
  (3 * x * rice_cost1 + x * rice_cost2) / (3 * x + x) = 18 :=
by
  sorry

end rice_mixture_price_l43_43092


namespace average_population_is_1000_l43_43259

-- Define the populations of the villages.
def populations : List ℕ := [803, 900, 1100, 1023, 945, 980, 1249]

-- Define the number of villages.
def num_villages : ℕ := 7

-- Define the total population.
def total_population (pops : List ℕ) : ℕ :=
  pops.foldl (λ acc x => acc + x) 0

-- Define the average population computation.
def average_population (pops : List ℕ) (n : ℕ) : ℕ :=
  total_population pops / n

-- Prove that the average population of the 7 villages is 1000.
theorem average_population_is_1000 :
  average_population populations num_villages = 1000 := by
  -- Proof omitted.
  sorry

end average_population_is_1000_l43_43259


namespace fraction_subtraction_simplest_form_l43_43749

theorem fraction_subtraction_simplest_form :
  (8 / 24 - 5 / 40 = 5 / 24) :=
by
  sorry

end fraction_subtraction_simplest_form_l43_43749


namespace smallest_integer_remainder_l43_43817

theorem smallest_integer_remainder :
  ∃ n : ℕ, n > 1 ∧
           (n % 3 = 2) ∧
           (n % 4 = 2) ∧
           (n % 5 = 2) ∧
           (n % 7 = 2) ∧
           n = 422 :=
by
  sorry

end smallest_integer_remainder_l43_43817


namespace rectangle_area_l43_43821

-- Definitions of the conditions
variables (Length Width Area : ℕ)
variable (h1 : Length = 4 * Width)
variable (h2 : Length = 20)

-- Statement to prove
theorem rectangle_area : Area = Length * Width → Area = 100 :=
by
  sorry

end rectangle_area_l43_43821


namespace collinear_vector_l43_43669

theorem collinear_vector (c R : ℝ) (A B : ℝ × ℝ) (hA: A.1 ^ 2 + A.2 ^ 2 = R ^ 2) (hB: B.1 ^ 2 + B.2 ^ 2 = R ^ 2) 
                         (h_line_A: 2 * A.1 + A.2 = c) (h_line_B: 2 * B.1 + B.2 = c) :
                         ∃ k : ℝ, (4, 2) = (k * (A.1 + B.1), k * (A.2 + B.2)) :=
sorry

end collinear_vector_l43_43669


namespace inequality_example_l43_43317

open Real

theorem inequality_example 
    (x y z : ℝ) 
    (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1):
    (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := 
by 
  sorry

end inequality_example_l43_43317


namespace sequence_formula_l43_43711

theorem sequence_formula (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a n - a (n + 1) + 2 = 0) :
  ∀ n : ℕ, a n = 2 * n - 1 := 
sorry

end sequence_formula_l43_43711


namespace last_number_is_four_l43_43059

theorem last_number_is_four (a b c d e last_number : ℕ) (h_counts : a = 6 ∧ b = 12 ∧ c = 1 ∧ d = 12 ∧ e = 7)
    (h_mean : (a + b + c + d + e + last_number) / 6 = 7) : last_number = 4 := 
sorry

end last_number_is_four_l43_43059


namespace production_rate_l43_43400

theorem production_rate (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (H : x * x * 2 * x = 2 * x^3) :
  y * y * 3 * y = 3 * y^3 := by
  sorry

end production_rate_l43_43400


namespace cross_number_puzzle_digit_star_l43_43136

theorem cross_number_puzzle_digit_star :
  ∃ N₁ N₂ N₃ N₄ : ℕ,
    N₁ % 1000 / 100 = 4 ∧ N₁ % 10 = 1 ∧ ∃ n : ℕ, N₁ = n ^ 2 ∧
    N₃ % 1000 / 100 = 6 ∧ ∃ m : ℕ, N₃ = m ^ 4 ∧
    ∃ p : ℕ, N₂ = 2 * p ^ 5 ∧ 100 ≤ N₂ ∧ N₂ < 1000 ∧
    N₄ % 10 = 5 ∧ ∃ q : ℕ, N₄ = q ^ 3 ∧ 100 ≤ N₄ ∧ N₄ < 1000 ∧
    (N₁ % 10 = 4) :=
by
  sorry

end cross_number_puzzle_digit_star_l43_43136


namespace area_enclosed_by_region_l43_43049

theorem area_enclosed_by_region :
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 6*y - 3 = 0) → 
  (∃ r : ℝ, r = 4 ∧ area = (π * r^2)) :=
by
  -- Starting proof setup
  sorry

end area_enclosed_by_region_l43_43049


namespace smallest_number_of_pencils_l43_43261

theorem smallest_number_of_pencils 
  (p : ℕ) 
  (h1 : p % 6 = 5)
  (h2 : p % 7 = 3)
  (h3 : p % 8 = 7) :
  p = 35 := 
sorry

end smallest_number_of_pencils_l43_43261


namespace salt_mixture_problem_l43_43836

theorem salt_mixture_problem :
  ∃ (m : ℝ), 0.20 = (150 + 0.05 * m) / (600 + m) :=
by
  sorry

end salt_mixture_problem_l43_43836


namespace ball_distribution_l43_43982

theorem ball_distribution (n : ℕ) (P_white P_red P_yellow : ℚ) (num_white num_red num_yellow : ℕ) 
  (total_balls : n = 6)
  (prob_white : P_white = 1/2)
  (prob_red : P_red = 1/3)
  (prob_yellow : P_yellow = 1/6) :
  num_white = 3 ∧ num_red = 2 ∧ num_yellow = 1 := 
sorry

end ball_distribution_l43_43982


namespace connie_blue_markers_l43_43265

theorem connie_blue_markers :
  ∀ (total_markers red_markers blue_markers : ℕ),
    total_markers = 105 →
    red_markers = 41 →
    blue_markers = total_markers - red_markers →
    blue_markers = 64 :=
by
  intros total_markers red_markers blue_markers htotal hred hblue
  rw [htotal, hred] at hblue
  exact hblue

end connie_blue_markers_l43_43265


namespace f_is_odd_l43_43719

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f (x : ℝ) : ℝ := x * |x|

theorem f_is_odd : is_odd_function f :=
by sorry

end f_is_odd_l43_43719


namespace base_number_pow_19_mod_10_l43_43258

theorem base_number_pow_19_mod_10 (x : ℕ) (h : x ^ 19 % 10 = 7) : x % 10 = 3 :=
sorry

end base_number_pow_19_mod_10_l43_43258


namespace outlet_pipe_rate_l43_43230

theorem outlet_pipe_rate (V_ft : ℝ) (cf : ℝ) (V_in : ℝ) (r_in : ℝ) (r_out1 : ℝ) (t : ℝ) (r_out2 : ℝ) :
    V_ft = 30 ∧ cf = 1728 ∧
    V_in = V_ft * cf ∧
    r_in = 5 ∧ r_out1 = 9 ∧ t = 4320 ∧
    V_in = (r_out1 + r_out2 - r_in) * t →
    r_out2 = 8 := by
  intros h
  sorry

end outlet_pipe_rate_l43_43230


namespace ratio_of_numbers_l43_43875

theorem ratio_of_numbers (A B : ℕ) (hA : A = 45) (hLCM : Nat.lcm A B = 180) : A / Nat.lcm A B = 45 / 4 :=
by
  sorry

end ratio_of_numbers_l43_43875


namespace new_remainder_when_scaled_l43_43124

theorem new_remainder_when_scaled (a b c : ℕ) (h : a = b * c + 7) : (10 * a) % (10 * b) = 70 := by
  sorry

end new_remainder_when_scaled_l43_43124


namespace repeating_decimal_sum_l43_43730

noncomputable def repeating_decimal_0_3 : ℚ := 1 / 3
noncomputable def repeating_decimal_0_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_0_2 : ℚ := 2 / 9

theorem repeating_decimal_sum :
  repeating_decimal_0_3 + repeating_decimal_0_6 - repeating_decimal_0_2 = 7 / 9 :=
by
  sorry

end repeating_decimal_sum_l43_43730


namespace exists_six_digit_number_l43_43039

theorem exists_six_digit_number : ∃ (n : ℕ), 100000 ≤ n ∧ n < 1000000 ∧ (∃ (x y : ℕ), n = 1000 * x + y ∧ 0 ≤ x ∧ x < 1000 ∧ 0 ≤ y ∧ y < 1000 ∧ 6 * n = 1000 * y + x) :=
by
  sorry

end exists_six_digit_number_l43_43039


namespace sum_fractions_geq_six_l43_43813

variable (x y z : ℝ)
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

theorem sum_fractions_geq_six : 
  (x / y + y / z + z / x + x / z + z / y + y / x) ≥ 6 := 
by
  sorry

end sum_fractions_geq_six_l43_43813


namespace quadratic_inequality_solution_l43_43764

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 6 * x + 5 > 0) ↔ (x < 1 ∨ x > 5) := sorry

end quadratic_inequality_solution_l43_43764


namespace max_value_f_1_max_value_f_2_max_value_f_3_l43_43193
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - m * x

theorem max_value_f_1 (m : ℝ) (h : m ≤ 1 / Real.exp 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ 1 - m * Real.exp 1 :=
sorry

theorem max_value_f_2 (m : ℝ) (h1 : 1 / Real.exp 1 < m) (h2 : m < 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ -Real.log m - 1 :=
sorry

theorem max_value_f_3 (m : ℝ) (h : m ≥ 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ -m :=
sorry

end max_value_f_1_max_value_f_2_max_value_f_3_l43_43193


namespace no_politics_reporters_l43_43930

theorem no_politics_reporters (X Y Both XDontY YDontX International PercentageTotal : ℝ) 
  (hX : X = 0.35)
  (hY : Y = 0.25)
  (hBoth : Both = 0.20)
  (hXDontY : XDontY = 0.30)
  (hInternational : International = 0.15)
  (hPercentageTotal : PercentageTotal = 1.0) :
  PercentageTotal - ((X + Y - Both) - XDontY + International) = 0.75 :=
by sorry

end no_politics_reporters_l43_43930


namespace work_completed_in_30_days_l43_43549

theorem work_completed_in_30_days (ravi_days : ℕ) (prakash_days : ℕ)
  (h1 : ravi_days = 50) (h2 : prakash_days = 75) : 
  let ravi_rate := (1 / 50 : ℚ)
  let prakash_rate := (1 / 75 : ℚ)
  let combined_rate := ravi_rate + prakash_rate
  let days_to_complete := 1 / combined_rate
  days_to_complete = 30 := by
  sorry

end work_completed_in_30_days_l43_43549


namespace total_cost_is_135_25_l43_43933

-- defining costs and quantities
def cost_A : ℕ := 9
def num_A : ℕ := 4
def cost_B := cost_A + 5
def num_B : ℕ := 2
def cost_clay_pot := cost_A + 20
def cost_bag_soil := cost_A - 2
def cost_fertilizer := cost_A + (cost_A / 2)
def cost_gardening_tools := cost_clay_pot - (cost_clay_pot / 4)

-- total cost calculation
def total_cost : ℚ :=
  (num_A * cost_A) + 
  (num_B * cost_B) + 
  cost_clay_pot + 
  cost_bag_soil + 
  cost_fertilizer + 
  cost_gardening_tools

theorem total_cost_is_135_25 : total_cost = 135.25 := by
  sorry

end total_cost_is_135_25_l43_43933


namespace exponential_comparisons_l43_43704

open Real

noncomputable def a : ℝ := 5 ^ (log 3.4 / log 2)
noncomputable def b : ℝ := 5 ^ (log 3.6 / (log 4))
noncomputable def c : ℝ := 5 ^ (log (10 / 3))

theorem exponential_comparisons :
  a > c ∧ c > b := by
  sorry

end exponential_comparisons_l43_43704


namespace average_number_of_stickers_per_album_is_correct_l43_43587

def average_stickers_per_album (albums : List ℕ) (n : ℕ) : ℚ := (albums.sum : ℚ) / n

theorem average_number_of_stickers_per_album_is_correct :
  average_stickers_per_album [5, 7, 9, 14, 19, 12, 26, 18, 11, 15] 10 = 13.6 := 
by
  sorry

end average_number_of_stickers_per_album_is_correct_l43_43587


namespace wine_problem_l43_43409

theorem wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + (1 / 3) * y = 33) : x + y = 19 ∧ 3 * x + (1 / 3) * y = 33 :=
by
  sorry

end wine_problem_l43_43409


namespace number_of_small_spheres_l43_43625

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem number_of_small_spheres
  (d_large : ℝ) (d_small : ℝ)
  (h1 : d_large = 6) (h2 : d_small = 2) :
  let V_large := volume_of_sphere (d_large / 2)
  let V_small := volume_of_sphere (d_small / 2)
  V_large / V_small = 27 := 
by
  sorry

end number_of_small_spheres_l43_43625


namespace grazing_b_l43_43200

theorem grazing_b (A_oxen_months B_oxen_months C_oxen_months total_months total_rent C_rent B_oxen : ℕ) 
  (hA : A_oxen_months = 10 * 7)
  (hB : B_oxen_months = B_oxen * 5)
  (hC : C_oxen_months = 15 * 3)
  (htotal : total_months = A_oxen_months + B_oxen_months + C_oxen_months)
  (hrent : total_rent = 175)
  (hC_rent : C_rent = 45)
  (hC_share : C_oxen_months / total_months = C_rent / total_rent) :
  B_oxen = 12 :=
by
  sorry

end grazing_b_l43_43200


namespace probability_T_H_E_equal_L_A_V_A_l43_43641

noncomputable def probability_condition : ℚ :=
  -- Number of total sample space (3^6)
  (3 ^ 6 : ℚ)

noncomputable def favorable_events_0 : ℚ :=
  -- Number of favorable outcomes where 𝑻 ⋅ 𝑯 ⋅ 𝑬 is 0 and 𝑳 ⋅ 𝑨 ⋅ 𝑽 ⋅ 𝑨 is 0
  26 * 19

noncomputable def favorable_events_1 : ℚ :=
  -- Number of favorable outcomes where 𝑻 ⋅ 𝑯 ⋅ 𝑬 is 1 and 𝑳 ⋅ 𝑨 ⋅ 𝑽 ⋅ 𝑨 is 1
  1

noncomputable def total_favorable_events : ℚ :=
  favorable_events_0 + favorable_events_1

theorem probability_T_H_E_equal_L_A_V_A :
  (total_favorable_events / probability_condition) = 55 / 81 :=
sorry

end probability_T_H_E_equal_L_A_V_A_l43_43641


namespace value_of_expression_l43_43250

theorem value_of_expression (a b : ℝ) (h1 : ∃ x : ℝ, x^2 + 3 * x - 5 = 0)
  (h2 : ∃ y : ℝ, y^2 + 3 * y - 5 = 0)
  (h3 : a ≠ b)
  (h4 : ∀ r : ℝ, r^2 + 3 * r - 5 = 0 → r = a ∨ r = b) : a^2 + 3 * a * b + a - 2 * b = -4 :=
by
  sorry

end value_of_expression_l43_43250


namespace sum_from_neg_50_to_75_l43_43903

def sum_of_integers (a b : ℤ) : ℤ :=
  (b * (b + 1)) / 2 - (a * (a - 1)) / 2

theorem sum_from_neg_50_to_75 : sum_of_integers (-50) 75 = 1575 := by
  sorry

end sum_from_neg_50_to_75_l43_43903


namespace spring_mass_relationship_l43_43550

theorem spring_mass_relationship (x y : ℕ) (h1 : y = 18 + 2 * x) : 
  y = 32 → x = 7 :=
by
  sorry

end spring_mass_relationship_l43_43550


namespace perpendicular_line_through_point_l43_43410

def point : ℝ × ℝ := (1, 0)

def given_line (x y : ℝ) : Prop := x - y + 2 = 0

def is_perpendicular_to (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l1 x y → l2 (y - x) (-x - y + 2)

def target_line (x y : ℝ) : Prop := x + y - 1 = 0

theorem perpendicular_line_through_point (l1 : ℝ → ℝ → Prop) (p : ℝ × ℝ) :
  given_line = l1 ∧ p = point →
  (∃ l2 : ℝ → ℝ → Prop, is_perpendicular_to l1 l2 ∧ l2 p.1 p.2) →
  target_line p.1 p.2 :=
by
  intro hp hl2
  sorry

end perpendicular_line_through_point_l43_43410


namespace ratio_of_products_l43_43299

theorem ratio_of_products (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  ((a - c) * (b - d)) / ((a - b) * (c - d)) = -4 / 3 :=
by 
  sorry

end ratio_of_products_l43_43299


namespace average_income_l43_43678

theorem average_income (income1 income2 income3 income4 income5 : ℝ)
    (h1 : income1 = 600) (h2 : income2 = 250) (h3 : income3 = 450) (h4 : income4 = 400) (h5 : income5 = 800) :
    (income1 + income2 + income3 + income4 + income5) / 5 = 500 := by
    sorry

end average_income_l43_43678


namespace quadratic_roots_l43_43238

theorem quadratic_roots {a : ℝ} :
  (4 < a ∧ a < 6) ∨ (a > 12) → 
  (∃ x1 x2 : ℝ, x1 = a + Real.sqrt (18 * (a - 4)) ∧ x2 = a - Real.sqrt (18 * (a - 4)) ∧ x1 > 0 ∧ x2 > 0) :=
by sorry

end quadratic_roots_l43_43238


namespace product_of_possible_values_of_x_l43_43579

noncomputable def product_of_roots (a b c : ℤ) : ℤ :=
  c / a

theorem product_of_possible_values_of_x :
  ∃ x : ℝ, (x + 3) * (x - 4) = 18 ∧ product_of_roots 1 (-1) (-30) = -30 := 
by
  sorry

end product_of_possible_values_of_x_l43_43579


namespace James_total_water_capacity_l43_43981

theorem James_total_water_capacity : 
  let cask_capacity := 20 -- capacity of a cask in gallons
  let barrel_capacity := 2 * cask_capacity + 3 -- capacity of a barrel in gallons
  let total_capacity := 4 * barrel_capacity + cask_capacity -- total water storage capacity
  total_capacity = 192 := by
    let cask_capacity := 20
    let barrel_capacity := 2 * cask_capacity + 3
    let total_capacity := 4 * barrel_capacity + cask_capacity
    have h : total_capacity = 192 := by sorry
    exact h

end James_total_water_capacity_l43_43981


namespace tetrahedron_altitudes_l43_43289

theorem tetrahedron_altitudes (r h₁ h₂ h₃ h₄ : ℝ)
  (h₁_def : h₁ = 3 * r)
  (h₂_def : h₂ = 4 * r)
  (h₃_def : h₃ = 4 * r)
  (altitude_sum : 1/h₁ + 1/h₂ + 1/h₃ + 1/h₄ = 1/r) : 
  h₄ = 6 * r :=
by
  rw [h₁_def, h₂_def, h₃_def] at altitude_sum
  sorry

end tetrahedron_altitudes_l43_43289


namespace profit_calculation_l43_43359

def Initial_Value : ℕ := 100
def Multiplier : ℕ := 3
def New_Value : ℕ := Initial_Value * Multiplier
def Profit : ℕ := New_Value - Initial_Value

theorem profit_calculation : Profit = 200 := by
  sorry

end profit_calculation_l43_43359


namespace EG_perpendicular_to_AC_l43_43910

noncomputable def rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 < B.1 ∧ A.2 = B.2 ∧ B.1 < C.1 ∧ B.2 < C.2 ∧ C.1 = D.1 ∧ C.2 > D.2 ∧ D.1 > A.1 ∧ D.2 = A.2

theorem EG_perpendicular_to_AC
  {A B C D E F G: ℝ × ℝ}
  (h1: rectangle A B C D)
  (h2: E = (B.1, C.2) ∨ E = (C.1, B.2)) -- Assuming E lies on BC or BA
  (h3: F = (B.1, A.2) ∨ F = (A.1, B.2)) -- Assuming F lies on BA or BC
  (h4: G = (C.1, D.2) ∨ G = (D.1, C.2)) -- Assuming G lies on CD
  (h5: (F.1, G.2) = (A.1, C.2)) -- Line through F parallel to AC meets CD at G
: ∃ (H : ℝ × ℝ → ℝ × ℝ → ℝ), H E G = 0 := sorry

end EG_perpendicular_to_AC_l43_43910


namespace problem_1_problem_2_l43_43219

theorem problem_1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : ∀ x, |x + a| + |x - b| + c ≥ 4) : 
  a + b + c = 4 :=
sorry

theorem problem_2 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 4) : 
  (1/4) * a^2 + (1/9) * b^2 + c^2 = 8 / 7 :=
sorry

end problem_1_problem_2_l43_43219


namespace sum_of_coefficients_factors_l43_43487

theorem sum_of_coefficients_factors :
  ∃ (a b c d e : ℤ), 
    (343 * (x : ℤ)^3 + 125 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
    (a + b + c + d + e = 51) :=
sorry

end sum_of_coefficients_factors_l43_43487


namespace find_point_A_l43_43094

-- Define the point -3, 4
def pointP : ℝ × ℝ := (-3, 4)

-- Define the point 0, 2
def pointB : ℝ × ℝ := (0, 2)

-- Define the coordinates of point A
def pointA (x : ℝ) : ℝ × ℝ := (x, 0)

-- The hypothesis using the condition derived from the problem
def ray_reflection_condition (x : ℝ) : Prop :=
  4 / (x + 3) = -2 / x

-- The main theorem we need to prove that the coordinates of point A are (-1, 0)
theorem find_point_A :
  ∃ x : ℝ, ray_reflection_condition x ∧ pointA x = (-1, 0) :=
sorry

end find_point_A_l43_43094


namespace tan_alpha_eq_two_imp_inv_sin_double_angle_l43_43912

theorem tan_alpha_eq_two_imp_inv_sin_double_angle (α : ℝ) (h : Real.tan α = 2) : 
  (1 / Real.sin (2 * α)) = 5 / 4 :=
by
  sorry

end tan_alpha_eq_two_imp_inv_sin_double_angle_l43_43912


namespace communication_system_connections_l43_43264

theorem communication_system_connections (n : ℕ) (h : ∀ k < 2001, ∃ l < 2001, l ≠ k ∧ k ≠ l) :
  (∀ k < 2001, ∃ l < 2001, k ≠ l) → (n % 2 = 0 ∧ n ≤ 2000) ∨ n = 0 :=
sorry

end communication_system_connections_l43_43264


namespace proof_problem_l43_43430

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define the set M
def M : Set Nat := {2, 4}

-- Define the set N
def N : Set Nat := {0, 4}

-- Define the union of sets M and N
def M_union_N : Set Nat := M ∪ N

-- Define the complement of M ∪ N in U
def complement_U (s : Set Nat) : Set Nat := U \ s

-- State the theorem
theorem proof_problem : complement_U M_union_N = {1, 3} := by
  sorry

end proof_problem_l43_43430


namespace henrikh_commute_distance_l43_43346

theorem henrikh_commute_distance (x : ℕ)
    (h1 : ∀ y : ℕ, y = x → y = x)
    (h2 : 1 * x = x)
    (h3 : 20 * x = (x : ℕ))
    (h4 : x = (x / 3) + 8) :
    x = 12 := sorry

end henrikh_commute_distance_l43_43346


namespace product_uvw_l43_43841

theorem product_uvw (a x y c : ℝ) (u v w : ℤ) :
  (a^u * x - a^v) * (a^w * y - a^3) = a^5 * c^5 → 
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1) → 
  u * v * w = 6 :=
by
  intros h1 h2
  -- Proof will go here
  sorry

end product_uvw_l43_43841


namespace inequality_condition_l43_43352

-- Define the inequality (x - 2) * (x + 2) > 0
def inequality_holds (x : ℝ) : Prop := (x - 2) * (x + 2) > 0

-- The sufficient and necessary condition for the inequality to hold is x > 2 or x < -2
theorem inequality_condition (x : ℝ) : inequality_holds x ↔ (x > 2 ∨ x < -2) :=
  sorry

end inequality_condition_l43_43352


namespace div_mul_fraction_eq_neg_81_over_4_l43_43327

theorem div_mul_fraction_eq_neg_81_over_4 : 
  -4 / (4 / 9) * (9 / 4) = - (81 / 4) := 
by
  sorry

end div_mul_fraction_eq_neg_81_over_4_l43_43327


namespace rectangle_area_l43_43986

-- Define the length and width of the rectangle based on given ratio
def length (k: ℝ) := 5 * k
def width (k: ℝ) := 2 * k

-- The perimeter condition
def perimeter (k: ℝ) := 2 * (length k) + 2 * (width k) = 280

-- The diagonal condition
def diagonal_condition (k: ℝ) := (width k) * Real.sqrt 2 = (length k) / 2

-- The area of the rectangle
def area (k: ℝ) := (length k) * (width k)

-- The main theorem to be proven
theorem rectangle_area : ∃ k: ℝ, perimeter k ∧ diagonal_condition k ∧ area k = 4000 :=
by
  sorry

end rectangle_area_l43_43986


namespace find_minimum_fuse_length_l43_43023

def safeZone : ℝ := 70
def fuseBurningSpeed : ℝ := 0.112
def personSpeed : ℝ := 7
def minimumFuseLength : ℝ := 1.1

theorem find_minimum_fuse_length (x : ℝ) (h1 : x ≥ 0):
  (safeZone / personSpeed) * fuseBurningSpeed ≤ x :=
by
  sorry

end find_minimum_fuse_length_l43_43023


namespace original_polygon_sides_l43_43731

theorem original_polygon_sides {n : ℕ} 
  (h : (n - 2) * 180 = 1620) : n = 10 ∨ n = 11 ∨ n = 12 :=
sorry

end original_polygon_sides_l43_43731


namespace price_of_orange_is_60_l43_43554

-- Given: 
-- 1. The price of each apple is 40 cents.
-- 2. Mary selects 10 pieces of fruit in total.
-- 3. The average price of these 10 pieces is 56 cents.
-- 4. Mary must put back 6 oranges so that the remaining average price is 50 cents.
-- Prove: The price of each orange is 60 cents.

theorem price_of_orange_is_60 (a o : ℕ) (x : ℕ) 
  (h1 : a + o = 10)
  (h2 : 40 * a + x * o = 560)
  (h3 : 40 * a + x * (o - 6) = 200) : 
  x = 60 :=
by
  have eq1 : 40 * a + x * o = 560 := h2
  have eq2 : 40 * a + x * (o - 6) = 200 := h3
  sorry

end price_of_orange_is_60_l43_43554


namespace quadratic_inequality_solution_set_l43_43438

theorem quadratic_inequality_solution_set (x : ℝ) : 
  (x^2 - 2 * x < 0) ↔ (0 < x ∧ x < 2) := 
sorry

end quadratic_inequality_solution_set_l43_43438


namespace equivalent_spherical_coords_l43_43541

theorem equivalent_spherical_coords (ρ θ φ : ℝ) (hρ : ρ = 4) (hθ : θ = 3 * π / 8) (hφ : φ = 9 * π / 5) :
  ∃ (ρ' θ' φ' : ℝ), ρ' = 4 ∧ θ' = 11 * π / 8 ∧ φ' = π / 5 ∧ 
  (ρ' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π ∧ 0 ≤ φ' ∧ φ' ≤ π) :=
by
  sorry

end equivalent_spherical_coords_l43_43541


namespace relationship_sides_l43_43865

-- Definitions for the given condition
variables (a b c : ℝ)

-- Statement of the theorem to prove
theorem relationship_sides (h : a^2 - 16 * b^2 - c^2 + 6 * a * b + 10 * b * c = 0) : a + c = 2 * b :=
sorry

end relationship_sides_l43_43865


namespace third_stack_shorter_by_five_l43_43844

theorem third_stack_shorter_by_five
    (first_stack second_stack third_stack fourth_stack : ℕ)
    (h1 : first_stack = 5)
    (h2 : second_stack = first_stack + 2)
    (h3 : fourth_stack = third_stack + 5)
    (h4 : first_stack + second_stack + third_stack + fourth_stack = 21) :
    second_stack - third_stack = 5 :=
by
  sorry

end third_stack_shorter_by_five_l43_43844


namespace product_div_by_six_l43_43816

theorem product_div_by_six (A B C : ℤ) (h1 : A^2 + B^2 = C^2) 
  (h2 : ∀ n : ℤ, ¬ ∃ k : ℤ, n^2 = 4 * k + 2) 
  (h3 : ∀ n : ℤ, ¬ ∃ k : ℤ, n^2 = 3 * k + 2) : 
  6 ∣ (A * B) :=
sorry

end product_div_by_six_l43_43816


namespace sin_cos_product_l43_43357

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l43_43357


namespace part1_part2_part3_l43_43417

-- Part 1
theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem part2 (a m n : ℤ) (h1 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) (h2 : 0 < a) (h3 : 0 < m) (h4 : 0 < n) : 
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem part3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 :=
sorry

end part1_part2_part3_l43_43417


namespace volume_ratio_sphere_cylinder_inscribed_l43_43718

noncomputable def ratio_of_volumes (d : ℝ) : ℝ :=
  let Vs := (4 / 3) * Real.pi * (d / 2)^3
  let Vc := Real.pi * (d / 2)^2 * d
  Vs / Vc

theorem volume_ratio_sphere_cylinder_inscribed (d : ℝ) (h : d > 0) : 
  ratio_of_volumes d = 2 / 3 := 
by
  sorry

end volume_ratio_sphere_cylinder_inscribed_l43_43718


namespace num_solutions_20_l43_43240

def num_solutions (n : ℕ) : ℕ :=
  4 * n

theorem num_solutions_20 : num_solutions 20 = 80 := by
  sorry

end num_solutions_20_l43_43240


namespace equal_lengths_l43_43122

noncomputable def F (x y z : ℝ) := (x+y+z) * (x+y-z) * (y+z-x) * (x+z-y)

variables {a b c d e f : ℝ}

axiom acute_angled_triangle (x y z : ℝ) : Prop

axiom altitudes_sum_greater (x y z : ℝ) : Prop

axiom cond1 : acute_angled_triangle a b c
axiom cond2 : acute_angled_triangle b d f
axiom cond3 : acute_angled_triangle a e f
axiom cond4 : acute_angled_triangle e c d

axiom cond5 : altitudes_sum_greater a b c
axiom cond6 : altitudes_sum_greater b d f
axiom cond7 : altitudes_sum_greater a e f
axiom cond8 : altitudes_sum_greater e c d

axiom cond9 : F a b c = F b d f
axiom cond10 : F a e f = F e c d

theorem equal_lengths : a = d ∧ b = e ∧ c = f := by
  sorry -- Proof not required.

end equal_lengths_l43_43122


namespace no_valid_bases_l43_43459

theorem no_valid_bases
  (x y : ℕ)
  (h1 : 4 * x + 9 = 4 * y + 1)
  (h2 : 4 * x^2 + 7 * x + 7 = 3 * y^2 + 2 * y + 9)
  (hx : x > 1)
  (hy : y > 1)
  : false :=
by
  sorry

end no_valid_bases_l43_43459


namespace total_spots_l43_43822

variable (P : ℕ)
variable (Bill_spots : ℕ := 2 * P - 1)

-- Given conditions
variable (h1 : Bill_spots = 39)

-- Theorem we need to prove
theorem total_spots (P : ℕ) (Bill_spots : ℕ := 2 * P - 1) (h1 : Bill_spots = 39) : 
  Bill_spots + P = 59 := 
by
  sorry

end total_spots_l43_43822


namespace somu_current_age_l43_43433

variable (S F : ℕ)

theorem somu_current_age
  (h1 : S = F / 3)
  (h2 : S - 10 = (F - 10) / 5) :
  S = 20 := by
  sorry

end somu_current_age_l43_43433


namespace xyz_value_l43_43736

variable {x y z : ℝ}

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 18) 
                  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 6) : 
                  x * y * z = 4 := 
by
  sorry

end xyz_value_l43_43736


namespace cubic_roots_result_l43_43800

theorem cubic_roots_result (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : a * 64 + b * 16 + c * 4 + d = 0) (h₃ : a * (-27) + b * 9 + c * (-3) + d = 0) :
  (b + c) / a = -13 :=
by
  sorry

end cubic_roots_result_l43_43800


namespace rectangle_area_l43_43158

theorem rectangle_area :
  ∃ (a b : ℕ), a ≠ b ∧ Even a ∧ (a * b = 3 * (2 * a + 2 * b)) ∧ (a * b = 162) :=
by
  sorry

end rectangle_area_l43_43158


namespace find_number_A_l43_43202

theorem find_number_A (A B : ℝ) (h₁ : A + B = 14.85) (h₂ : B = 10 * A) : A = 1.35 :=
sorry

end find_number_A_l43_43202


namespace moles_of_NaHSO4_l43_43083

def react_eq (naoh h2so4 nahso4 h2o : ℕ) : Prop :=
  naoh + h2so4 = nahso4 + h2o

theorem moles_of_NaHSO4
  (naoh h2so4 : ℕ)
  (h : 2 = naoh ∧ 2 = h2so4)
  (react : react_eq naoh h2so4 2 2):
  2 = 2 :=
by
  sorry

end moles_of_NaHSO4_l43_43083


namespace math_problem_proof_l43_43373

def eight_to_zero : ℝ := 1
def log_base_10_of_100 : ℝ := 2

theorem math_problem_proof : eight_to_zero - log_base_10_of_100 = -1 :=
by sorry

end math_problem_proof_l43_43373


namespace total_handshakes_l43_43279

-- Define the conditions
def number_of_players_per_team : Nat := 11
def number_of_referees : Nat := 3
def total_number_of_players : Nat := number_of_players_per_team * 2

-- Prove the total number of handshakes
theorem total_handshakes : 
  (number_of_players_per_team * number_of_players_per_team) + (total_number_of_players * number_of_referees) = 187 := 
by {
  sorry
}

end total_handshakes_l43_43279


namespace sum_of_possible_values_l43_43325

theorem sum_of_possible_values (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := 
sorry

end sum_of_possible_values_l43_43325


namespace perimeter_is_32_l43_43615

-- Define the side lengths of the triangle
def a : ℕ := 13
def b : ℕ := 9
def c : ℕ := 10

-- Definition of the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Theorem stating the perimeter is 32
theorem perimeter_is_32 : perimeter a b c = 32 :=
by
  sorry

end perimeter_is_32_l43_43615


namespace james_trip_time_l43_43303

def speed : ℝ := 60
def distance : ℝ := 360
def stop_time : ℝ := 1

theorem james_trip_time:
  (distance / speed) + stop_time = 7 := 
by
  sorry

end james_trip_time_l43_43303


namespace candy_from_sister_l43_43735

variable (f : ℕ) (e : ℕ) (t : ℕ)

theorem candy_from_sister (h₁ : f = 47) (h₂ : e = 25) (h₃ : t = 62) :
  ∃ x : ℕ, x = t - (f - e) ∧ x = 40 :=
by sorry

end candy_from_sister_l43_43735


namespace intersection_eq_l43_43780

variable {x : ℝ}

def set_A := {x : ℝ | x^2 - 4 * x < 0}
def set_B := {x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5}
def set_intersection := {x : ℝ | 1 / 3 ≤ x ∧ x < 4}

theorem intersection_eq : (set_A ∩ set_B) = set_intersection := by
  sorry

end intersection_eq_l43_43780


namespace dog_food_weighs_more_l43_43675

def weight_in_ounces (weight_in_pounds: ℕ) := weight_in_pounds * 16
def total_food_weight (cat_food_bags dog_food_bags: ℕ) (cat_food_pounds dog_food_pounds: ℕ) :=
  (cat_food_bags * weight_in_ounces cat_food_pounds) + (dog_food_bags * weight_in_ounces dog_food_pounds)

theorem dog_food_weighs_more
  (cat_food_bags: ℕ) (cat_food_pounds: ℕ) (dog_food_bags: ℕ) (total_weight_ounces: ℕ) (ounces_in_pound: ℕ)
  (H1: cat_food_bags * weight_in_ounces cat_food_pounds = 96)
  (H2: total_food_weight cat_food_bags dog_food_bags cat_food_pounds dog_food_pounds = total_weight_ounces)
  (H3: ounces_in_pound = 16) :
  dog_food_pounds - cat_food_pounds = 2 := 
by sorry

end dog_food_weighs_more_l43_43675


namespace ratio_of_wilted_roses_to_total_l43_43682

-- Defining the conditions
def initial_roses := 24
def traded_roses := 12
def total_roses := initial_roses + traded_roses
def remaining_roses_after_second_night := 9
def roses_before_second_night := remaining_roses_after_second_night * 2
def wilted_roses_after_first_night := total_roses - roses_before_second_night
def ratio_wilted_to_total := wilted_roses_after_first_night / total_roses

-- Proving the ratio of wilted flowers to the total number of flowers after the first night is 1:2
theorem ratio_of_wilted_roses_to_total :
  ratio_wilted_to_total = (1/2) := by
  sorry

end ratio_of_wilted_roses_to_total_l43_43682


namespace num_ways_to_make_change_l43_43213

-- Define the standard U.S. coins
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Define the total amount
def total_amount : ℕ := 50

-- Condition to exclude two quarters
def valid_combination (num_pennies num_nickels num_dimes num_quarters : ℕ) : Prop :=
  (num_quarters != 2) ∧ (num_pennies + 5 * num_nickels + 10 * num_dimes + 25 * num_quarters = total_amount)

-- Prove that there are 39 ways to make change for 50 cents
theorem num_ways_to_make_change : 
  ∃ count : ℕ, count = 39 ∧ (∀ 
    (num_pennies num_nickels num_dimes num_quarters : ℕ),
    valid_combination num_pennies num_nickels num_dimes num_quarters → 
    (num_pennies, num_nickels, num_dimes, num_quarters) = count) :=
sorry

end num_ways_to_make_change_l43_43213


namespace solve_arcsin_eq_l43_43185

noncomputable def arcsin (x : ℝ) : ℝ := Real.arcsin x
noncomputable def pi : ℝ := Real.pi

theorem solve_arcsin_eq :
  ∃ x : ℝ, arcsin x + arcsin (3 * x) = pi / 4 ∧ x = 1 / Real.sqrt 19 :=
sorry

end solve_arcsin_eq_l43_43185


namespace amount_spent_on_marbles_l43_43867

-- Definitions of conditions
def cost_of_football : ℝ := 5.71
def total_spent_on_toys : ℝ := 12.30

-- Theorem statement
theorem amount_spent_on_marbles : (total_spent_on_toys - cost_of_football) = 6.59 :=
by
  sorry

end amount_spent_on_marbles_l43_43867


namespace mutually_exclusive_complementary_event_l43_43211

-- Definitions of events
def hitting_target_at_least_once (shots: ℕ) : Prop := shots > 0
def not_hitting_target_at_all (shots: ℕ) : Prop := shots = 0

-- The statement to prove
theorem mutually_exclusive_complementary_event : 
  ∀ (shots: ℕ), (not_hitting_target_at_all shots ↔ ¬ hitting_target_at_least_once shots) :=
by 
  sorry

end mutually_exclusive_complementary_event_l43_43211


namespace minimum_value_l43_43484

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

theorem minimum_value : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x ≥ f (-1) :=
by
  sorry

end minimum_value_l43_43484


namespace trapezoid_height_l43_43093

-- Definitions of the problem conditions
def is_isosceles_trapezoid (a b : ℝ) : Prop :=
  ∃ (AB CD BM CN h : ℝ), a = 24 ∧ b = 10 ∧ AB = 25 ∧ CD = 25 ∧ BM = h ∧ CN = h ∧
  BM ^ 2 + ((24 - 10) / 2) ^ 2 = AB ^ 2

-- The theorem to prove
theorem trapezoid_height (a b : ℝ) (h : ℝ) 
  (H : is_isosceles_trapezoid a b) : h = 24 :=
sorry

end trapezoid_height_l43_43093


namespace max_value_2x_plus_y_l43_43107

def max_poly_value : ℝ :=
  sorry

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2 * y ≤ 3) (h2 : 0 ≤ x) (h3 : 0 ≤ y) : 
  2 * x + y ≤ 6 :=
sorry

example (x y : ℝ) (h1 : x + 2 * y ≤ 3) (h2 : 0 ≤ x) (h3 : 0 ≤ y) : 2 * x + y = 6 
  ↔ x = 3 ∧ y = 0 :=
by exact sorry

end max_value_2x_plus_y_l43_43107


namespace carol_remaining_distance_l43_43605

def fuel_efficiency : ℕ := 25 -- miles per gallon
def gas_tank_capacity : ℕ := 18 -- gallons
def distance_to_home : ℕ := 350 -- miles

def total_distance_on_full_tank : ℕ := fuel_efficiency * gas_tank_capacity
def distance_after_home : ℕ := total_distance_on_full_tank - distance_to_home

theorem carol_remaining_distance :
  distance_after_home = 100 :=
sorry

end carol_remaining_distance_l43_43605


namespace min_value_expression_l43_43335

theorem min_value_expression (x y : ℝ) : (x^2 + y^2 - 6 * x + 4 * y + 18) ≥ 5 :=
sorry

end min_value_expression_l43_43335


namespace equation_of_circle_unique_l43_43856

noncomputable def equation_of_circle := 
  ∃ (d e f : ℝ), 
    (4 + 4 + 2*d + 2*e + f = 0) ∧ 
    (25 + 9 + 5*d + 3*e + f = 0) ∧ 
    (9 + 1 + 3*d - e + f = 0) ∧ 
    (∀ (x y : ℝ), x^2 + y^2 + d*x + e*y + f = 0 → (x = 2 ∧ y = 2) ∨ (x = 5 ∧ y = 3) ∨ (x = 3 ∧ y = -1))

theorem equation_of_circle_unique :
  equation_of_circle := sorry

end equation_of_circle_unique_l43_43856


namespace exponents_of_equation_l43_43612

theorem exponents_of_equation :
  ∃ (x y : ℕ), 2 * (3 ^ 8) ^ 2 * (2 ^ 3) ^ 2 * 3 = 2 ^ x * 3 ^ y ∧ x = 7 ∧ y = 17 :=
by
  use 7
  use 17
  sorry

end exponents_of_equation_l43_43612


namespace salt_mixture_l43_43883

theorem salt_mixture (x y : ℝ) (p c z : ℝ) (hx : x = 50) (hp : p = 0.60) (hc : c = 0.40) (hy_eq : y = 50) :
  (50 * z) + (50 * 0.60) = 0.40 * (50 + 50) → (50 * z) + (50 * p) = c * (x + y) → y = 50 :=
by sorry

end salt_mixture_l43_43883


namespace total_distance_l43_43254

def morning_distance : ℕ := 2
def evening_multiplier : ℕ := 5

theorem total_distance : morning_distance + (evening_multiplier * morning_distance) = 12 :=
by
  sorry

end total_distance_l43_43254


namespace total_hours_watched_l43_43603

theorem total_hours_watched (Monday Tuesday Wednesday Thursday Friday : ℕ) (hMonday : Monday = 12) (hTuesday : Tuesday = 4) (hWednesday : Wednesday = 6) (hThursday : Thursday = (Monday + Tuesday + Wednesday) / 2) (hFriday : Friday = 19) :
  Monday + Tuesday + Wednesday + Thursday + Friday = 52 := by
  sorry

end total_hours_watched_l43_43603


namespace smallest_number_greater_than_500000_has_56_positive_factors_l43_43060

/-- Let n be the smallest number greater than 500,000 
    that is the product of the first four terms of both
    an arithmetic sequence and a geometric sequence.
    Prove that n has 56 positive factors. -/
theorem smallest_number_greater_than_500000_has_56_positive_factors :
  ∃ n : ℕ,
    (500000 < n) ∧
    (∀ a d b r, a > 0 → d > 0 → b > 0 → r > 0 →
      n = (a * (a + d) * (a + 2 * d) * (a + 3 * d)) ∧
          n = (b * (b * r) * (b * r^2) * (b * r^3))) ∧
    (n.factors.length = 56) :=
by sorry

end smallest_number_greater_than_500000_has_56_positive_factors_l43_43060


namespace factorize_expression_l43_43531

variable (x y : ℝ)

theorem factorize_expression : (x - y) ^ 2 + 2 * y * (x - y) = (x - y) * (x + y) := by
  sorry

end factorize_expression_l43_43531


namespace smaller_angle_between_hands_at_3_40_pm_larger_angle_between_hands_at_3_40_pm_l43_43854

def degree_movement_per_minute_of_minute_hand : ℝ := 6
def degree_movement_per_hour_of_hour_hand : ℝ := 30
def degree_movement_per_minute_of_hour_hand : ℝ := 0.5

def minute_position_at_3_40_pm : ℝ := 40 * degree_movement_per_minute_of_minute_hand
def hour_position_at_3_40_pm : ℝ := 3 * degree_movement_per_hour_of_hour_hand + 40 * degree_movement_per_minute_of_hour_hand

def clockwise_angle_between_hands_at_3_40_pm : ℝ := minute_position_at_3_40_pm - hour_position_at_3_40_pm
def counterclockwise_angle_between_hands_at_3_40_pm : ℝ := 360 - clockwise_angle_between_hands_at_3_40_pm

theorem smaller_angle_between_hands_at_3_40_pm : clockwise_angle_between_hands_at_3_40_pm = 130.0 := 
by
  sorry

theorem larger_angle_between_hands_at_3_40_pm : counterclockwise_angle_between_hands_at_3_40_pm = 230.0 := 
by
  sorry

end smaller_angle_between_hands_at_3_40_pm_larger_angle_between_hands_at_3_40_pm_l43_43854


namespace rectangle_area_l43_43665

-- Definitions
variables {height length : ℝ} (h : height = length / 2)
variables {area perimeter : ℝ} (a : area = perimeter)

-- Problem statement
theorem rectangle_area : ∃ h : ℝ, ∃ l : ℝ, ∃ area : ℝ, 
  (l = 2 * h) ∧ (area = l * h) ∧ (area = 2 * (l + h)) ∧ (area = 18) :=
sorry

end rectangle_area_l43_43665


namespace solve_system_of_equations_l43_43795

theorem solve_system_of_equations :
  ∃ (x y : ℤ), 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 ∧ x = 4 ∧ y = -1 :=
by
  sorry

end solve_system_of_equations_l43_43795


namespace point_outside_circle_l43_43389

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), (a * x + b * y = 1) ∧ (x^2 + y^2 = 1)) : a^2 + b^2 > 1 :=
by
  sorry

end point_outside_circle_l43_43389


namespace price_per_foot_l43_43728

theorem price_per_foot (area : ℝ) (cost : ℝ) (side_length : ℝ) (perimeter : ℝ) 
  (h1 : area = 289) (h2 : cost = 3740) 
  (h3 : side_length^2 = area) (h4 : perimeter = 4 * side_length) : 
  (cost / perimeter = 55) :=
by
  sorry

end price_per_foot_l43_43728


namespace remainder_division_39_l43_43154

theorem remainder_division_39 (N : ℕ) (k m R1 : ℕ) (hN1 : N = 39 * k + R1) (hN2 : N % 13 = 5) (hR1_lt_39 : R1 < 39) :
  R1 = 5 :=
by sorry

end remainder_division_39_l43_43154


namespace Kyle_is_25_l43_43993

-- Definitions based on the conditions
def Tyson_age : Nat := 20
def Frederick_age : Nat := 2 * Tyson_age
def Julian_age : Nat := Frederick_age - 20
def Kyle_age : Nat := Julian_age + 5

-- The theorem to prove
theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l43_43993


namespace percentage_of_third_number_l43_43423

theorem percentage_of_third_number (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : B = 0.18 * C) 
  (h3 : A = 0.3333333333333333 * B) : 
  A / C = 0.06 := 
by
  sorry

end percentage_of_third_number_l43_43423


namespace possible_value_of_a_l43_43015

variable {a b x : ℝ}

theorem possible_value_of_a (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) :
  a = 3 * x :=
sorry

end possible_value_of_a_l43_43015


namespace max_value_of_squares_l43_43139

theorem max_value_of_squares (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 8) :
  a^2 + b^2 + c^2 + d^2 ≤ 4 :=
sorry

end max_value_of_squares_l43_43139


namespace equalize_vertex_values_impossible_l43_43493

theorem equalize_vertex_values_impossible 
  (n : ℕ) (h₁ : 2 ≤ n) 
  (vertex_values : Fin n → ℤ) 
  (h₂ : ∃! i : Fin n, vertex_values i = 1 ∧ ∀ j ≠ i, vertex_values j = 0) 
  (k : ℕ) (hk : k ∣ n) :
  ¬ (∃ c : ℤ, ∀ v : Fin n, vertex_values v = c) := 
sorry

end equalize_vertex_values_impossible_l43_43493


namespace find_pairs_l43_43810

theorem find_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m^2 + n^2) ∣ (3 * m * n + 3 * m) ↔ (m, n) = (1, 1) ∨ (m, n) = (4, 2) ∨ (m, n) = (4, 10) :=
sorry

end find_pairs_l43_43810


namespace smallest_x_mod_equation_l43_43519

theorem smallest_x_mod_equation : ∃ x : ℕ, 42 * x + 10 ≡ 5 [MOD 15] ∧ ∀ y : ℕ, 42 * y + 10 ≡ 5 [MOD 15] → x ≤ y :=
by
sorry

end smallest_x_mod_equation_l43_43519


namespace sam_quarters_l43_43183

theorem sam_quarters (pennies : ℕ) (total : ℝ) (value_penny : ℝ) (value_quarter : ℝ) (quarters : ℕ) :
  pennies = 9 →
  total = 1.84 →
  value_penny = 0.01 →
  value_quarter = 0.25 →
  quarters = (total - pennies * value_penny) / value_quarter →
  quarters = 7 :=
by
  intros
  sorry

end sam_quarters_l43_43183


namespace july_husband_current_age_l43_43304

-- Define the initial ages and the relationship between Hannah and July's age
def hannah_initial_age : ℕ := 6
def hannah_july_age_relation (hannah_age july_age : ℕ) : Prop := hannah_age = 2 * july_age

-- Define the time that has passed and the age difference between July and her husband
def time_passed : ℕ := 20
def july_husband_age_relation (july_age husband_age : ℕ) : Prop := husband_age = july_age + 2

-- Lean statement to prove July's husband's current age
theorem july_husband_current_age : ∃ (july_age husband_age : ℕ),
  hannah_july_age_relation hannah_initial_age july_age ∧
  july_husband_age_relation (july_age + time_passed) husband_age ∧
  husband_age = 25 :=
by
  sorry

end july_husband_current_age_l43_43304


namespace gummy_bear_production_time_l43_43797

theorem gummy_bear_production_time 
  (gummy_bears_per_minute : ℕ)
  (gummy_bears_per_packet : ℕ)
  (total_packets : ℕ)
  (h1 : gummy_bears_per_minute = 300)
  (h2 : gummy_bears_per_packet = 50)
  (h3 : total_packets = 240) :
  (total_packets / (gummy_bears_per_minute / gummy_bears_per_packet) = 40) :=
sorry

end gummy_bear_production_time_l43_43797


namespace intersection_complement_l43_43721

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def complement_U (A : Set ℝ) : Set ℝ := {x | ¬ (A x)}

theorem intersection_complement (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  B ∩ (complement_U A) = {x | 2 < x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l43_43721


namespace solve_abs_linear_eq_l43_43027

theorem solve_abs_linear_eq (x : ℝ) : (|x - 1| + x - 1 = 0) ↔ (x ≤ 1) :=
sorry

end solve_abs_linear_eq_l43_43027


namespace simplify_and_evaluate_l43_43112

noncomputable def x := Real.tan (Real.pi / 4) + Real.cos (Real.pi / 6)

theorem simplify_and_evaluate :
  ((x / (x ^ 2 - 1)) * ((x - 1) / x - 2)) = - (2 * Real.sqrt 3) / 3 := 
sorry

end simplify_and_evaluate_l43_43112


namespace product_prs_l43_43337

open Real

theorem product_prs (p r s : ℕ) 
  (h1 : 4 ^ p + 64 = 272) 
  (h2 : 3 ^ r = 81)
  (h3 : 6 ^ s = 478) : 
  p * r * s = 64 :=
by
  sorry

end product_prs_l43_43337


namespace total_tiles_l43_43241

theorem total_tiles (n : ℕ) (h : 3 * n - 2 = 55) : n^2 = 361 :=
by
  sorry

end total_tiles_l43_43241


namespace tan_double_angle_l43_43628

theorem tan_double_angle (α β : ℝ) (h1 : Real.tan (α + β) = 7) (h2 : Real.tan (α - β) = 1) : 
  Real.tan (2 * α) = -4/3 :=
by
  sorry

end tan_double_angle_l43_43628


namespace compute_div_square_of_negatives_l43_43388

theorem compute_div_square_of_negatives : (-128)^2 / (-64)^2 = 4 := by
  sorry

end compute_div_square_of_negatives_l43_43388


namespace second_solution_volume_l43_43408

theorem second_solution_volume
  (V : ℝ)
  (h1 : 0.20 * 6 + 0.60 * V = 0.36 * (6 + V)) : 
  V = 4 :=
sorry

end second_solution_volume_l43_43408


namespace fraction_doubled_unchanged_l43_43281

theorem fraction_doubled_unchanged (x y : ℝ) (h : x ≠ y) : 
  (2 * x) / (2 * x - 2 * y) = x / (x - y) :=
by
  sorry

end fraction_doubled_unchanged_l43_43281


namespace locus_of_P_coordinates_of_P_l43_43526

-- Define the points A and B
def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (2, -1)

-- Define the line l : 4x + 3y - 2 = 0
def l (x y: ℝ) := 4 * x + 3 * y - 2 = 0

-- Problem (1): Equation of the locus of point P such that |PA| = |PB|
theorem locus_of_P (P : ℝ × ℝ) :
  (∃ P, dist P A = dist P B) ↔ (∀ x y : ℝ, P = (x, y) → x - y - 5 = 0) :=
sorry

-- Problem (2): Coordinates of P such that |PA| = |PB| and the distance from P to line l is 2
theorem coordinates_of_P (a b : ℝ):
  (dist (a, b) A = dist (a, b) B ∧ abs (4 * a + 3 * b - 2) / 5 = 2) ↔
  ((a = 1 ∧ b = -4) ∨ (a = 27 / 7 ∧ b = -8 / 7)) :=
sorry

end locus_of_P_coordinates_of_P_l43_43526


namespace fixed_monthly_fee_l43_43232

variable (x y : Real)

theorem fixed_monthly_fee :
  (x + y = 15.30) →
  (x + 1.5 * y = 20.55) →
  (x = 4.80) :=
by
  intros h1 h2
  sorry

end fixed_monthly_fee_l43_43232


namespace cell_cycle_correct_statement_l43_43197

theorem cell_cycle_correct_statement :
  ∃ (correct_statement : String), correct_statement = "In the cell cycle, chromatin DNA is easier to replicate than chromosome DNA" :=
by
  let A := "The separation of alleles occurs during the interphase of the cell cycle"
  let B := "In the cell cycle of plant cells, spindle fibers appear during the interphase"
  let C := "In the cell cycle, chromatin DNA is easier to replicate than chromosome DNA"
  let D := "In the cell cycle of liver cells, chromosomes exist for a longer time than chromatin"
  existsi C
  sorry

end cell_cycle_correct_statement_l43_43197


namespace paula_bracelets_count_l43_43149

-- Defining the given conditions
def cost_bracelet := 4
def cost_keychain := 5
def cost_coloring_book := 3
def total_spent := 20

-- Defining the cost for Paula's items
def cost_paula (B : ℕ) := B * cost_bracelet + cost_keychain

-- Defining the cost for Olive's items
def cost_olive := cost_coloring_book + cost_bracelet

-- Defining the main problem
theorem paula_bracelets_count (B : ℕ) (h : cost_paula B + cost_olive = total_spent) : B = 2 := by
  sorry

end paula_bracelets_count_l43_43149


namespace subcommittees_with_at_least_one_teacher_l43_43534

theorem subcommittees_with_at_least_one_teacher :
  let n := 12
  let t := 5
  let k := 4
  let total_subcommittees := Nat.choose n k
  let non_teacher_subcommittees := Nat.choose (n - t) k
  total_subcommittees - non_teacher_subcommittees = 460 :=
by
  -- Definitions and conditions based on the problem statement
  let n := 12
  let t := 5
  let k := 4
  let total_subcommittees := Nat.choose n k
  let non_teacher_subcommittees := Nat.choose (n - t) k
  sorry -- Proof goes here

end subcommittees_with_at_least_one_teacher_l43_43534


namespace survey_is_sample_of_population_l43_43172

-- Definitions based on the conditions in a)
def population_size := 50000
def sample_size := 2000
def is_comprehensive_survey := false
def is_sampling_survey := true
def is_population_student (n : ℕ) : Prop := n ≤ population_size
def is_individual_unit (n : ℕ) : Prop := n ≤ sample_size

-- Theorem that encapsulates the proof problem
theorem survey_is_sample_of_population : is_sampling_survey ∧ ∃ n, is_individual_unit n :=
by
  sorry

end survey_is_sample_of_population_l43_43172


namespace greatest_possible_length_l43_43548

-- Define the lengths of the ropes
def rope_lengths : List ℕ := [72, 48, 120, 96]

-- Define the gcd function to find the greatest common divisor of a list of numbers
def list_gcd (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- Define the target problem statement
theorem greatest_possible_length 
  (h : list_gcd rope_lengths = 24) : 
  ∀ length ∈ rope_lengths, length % 24 = 0 :=
by
  intros length h_length
  sorry

end greatest_possible_length_l43_43548


namespace square_side_length_l43_43808

theorem square_side_length (s : ℝ) (h : s^2 = 1 / 9) : s = 1 / 3 :=
sorry

end square_side_length_l43_43808


namespace integer_modulo_solution_l43_43313

theorem integer_modulo_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 137 ∧ 12345 ≡ n [ZMOD 137] ∧ n = 15 :=
sorry

end integer_modulo_solution_l43_43313


namespace find_y_l43_43885

theorem find_y (x y : ℕ) (h1 : x % y = 9) (h2 : x / y = 86 ∧ ((x % y : ℚ) / y = 0.12)) : y = 75 :=
by
  sorry

end find_y_l43_43885


namespace tie_to_shirt_ratio_l43_43683

-- Definitions for the conditions
def pants_cost : ℝ := 20
def shirt_cost : ℝ := 2 * pants_cost
def socks_cost : ℝ := 3
def r : ℝ := sorry -- This will be proved
def tie_cost : ℝ := r * shirt_cost
def uniform_cost : ℝ := pants_cost + shirt_cost + tie_cost + socks_cost

-- The total cost for five uniforms
def total_cost : ℝ := 5 * uniform_cost

-- The given total cost
def given_total_cost : ℝ := 355

-- The theorem to be proved
theorem tie_to_shirt_ratio :
  total_cost = given_total_cost → r = 1 / 5 := 
sorry

end tie_to_shirt_ratio_l43_43683


namespace projection_of_a_in_direction_of_b_l43_43896

noncomputable def vector_projection_in_direction (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / magnitude_b

theorem projection_of_a_in_direction_of_b :
  vector_projection_in_direction (3, 2) (-2, 1) = -4 * Real.sqrt 5 / 5 := 
by
  sorry

end projection_of_a_in_direction_of_b_l43_43896


namespace how_many_trucks_l43_43562

-- Define the conditions given in the problem
def people_to_lift_car : ℕ := 5
def people_to_lift_truck : ℕ := 2 * people_to_lift_car

-- Set up the problem conditions
def total_people_needed (cars : ℕ) (trucks : ℕ) : ℕ :=
  cars * people_to_lift_car + trucks * people_to_lift_truck

-- Now state the precise theorem we need to prove
theorem how_many_trucks (cars trucks total_people : ℕ) 
  (h1 : cars = 6)
  (h2 : trucks = 3)
  (h3 : total_people = total_people_needed cars trucks) :
  trucks = 3 :=
by
  sorry

end how_many_trucks_l43_43562


namespace classroom_gpa_l43_43305

theorem classroom_gpa (x : ℝ) (h1 : (1 / 3) * x + (2 / 3) * 18 = 17) : x = 15 := 
by 
    sorry

end classroom_gpa_l43_43305


namespace speed_of_man_cycling_l43_43085

theorem speed_of_man_cycling (L B : ℝ) (h1 : L / B = 1 / 3) (h2 : B = 3 * L)
  (h3 : L * B = 30000) (h4 : ∀ t : ℝ, t = 4 / 60): 
  ( (2 * L + 2 * B) / (4 / 60) ) = 12000 :=
by
  -- Assume given conditions
  sorry

end speed_of_man_cycling_l43_43085


namespace cannot_determine_students_answered_both_correctly_l43_43253

-- Definitions based on the given conditions
def students_enrolled : ℕ := 25
def students_answered_q1_correctly : ℕ := 22
def students_not_taken_test : ℕ := 3
def some_students_answered_q2_correctly : Prop := -- definition stating that there's an undefined number of students that answered question 2 correctly
  ∃ n : ℕ, (n ≤ students_enrolled) ∧ n > 0

-- Statement for the proof problem
theorem cannot_determine_students_answered_both_correctly :
  ∃ n, (n ≤ students_answered_q1_correctly) ∧ n > 0 → false :=
by sorry

end cannot_determine_students_answered_both_correctly_l43_43253


namespace ball_radius_l43_43394

theorem ball_radius (x r : ℝ) (h1 : x^2 + 256 = r^2) (h2 : r = x + 16) : r = 16 :=
by
  sorry

end ball_radius_l43_43394


namespace simplify_expression_l43_43672

theorem simplify_expression : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
    ((Real.sqrt 3 - 2 * Real.sqrt 5 - 1) * (-16 - 2 * Real.sqrt 3)) / 244 := 
  sorry

end simplify_expression_l43_43672


namespace proportionality_intersect_calculation_l43_43471

variables {x1 x2 y1 y2 : ℝ}

/-- Proof that (x1 - 2 * x2) * (3 * y1 + 4 * y2) = -15,
    given specific conditions on x1, x2, y1, and y2. -/
theorem proportionality_intersect_calculation
  (h1 : y1 = 5 / x1) 
  (h2 : y2 = 5 / x2)
  (h3 : x1 * y1 = 5)
  (h4 : x2 * y2 = 5)
  (h5 : x1 = -x2)
  (h6 : y1 = -y2) :
  (x1 - 2 * x2) * (3 * y1 + 4 * y2) = -15 := 
sorry

end proportionality_intersect_calculation_l43_43471


namespace part_a_part_b_l43_43751

def triangle := Type
def point := Type

structure TriangleInCircle (ABC : triangle) where
  A : point
  B : point
  C : point
  A1 : point
  B1 : point
  C1 : point
  M : point
  r : Real
  R : Real

theorem part_a (ABC : triangle) (t : TriangleInCircle ABC) :
  ∃ MA MC MB_1, (MA * MC) / MB_1 = 2 * t.r := sorry
  
theorem part_b (ABC : triangle) (t : TriangleInCircle ABC) :
  ∃ MA_1 MC_1 MB, ( (MA_1 * MC_1) / MB) = t.R := sorry

end part_a_part_b_l43_43751


namespace min_value_le_one_l43_43367

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (a : ℝ) : ℝ := a - a * Real.log a

theorem min_value_le_one (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, f x a ≥ g a) ∧ g a ≤ 1 := sorry

end min_value_le_one_l43_43367


namespace correct_answer_B_l43_43366

def point_slope_form (k : ℝ) (x y : ℝ) : Prop := y + 1 = k * (x - 2)

def proposition_2 (k : ℝ) (x y : ℝ) : Prop :=
  ∃ k : ℝ, @point_slope_form k x y

def proposition_3 (k : ℝ) : Prop := point_slope_form k 2 (-1)

def proposition_4 (k : ℝ) : Prop := k ≠ 0

theorem correct_answer_B : 
  (∃ k : ℝ, @point_slope_form k 2 (-1)) ∧ 
  (∀ k : ℝ, @point_slope_form k 2 (-1)) ∧
  (∀ k : ℝ, k ≠ 0) → true := 
by
  intro h
  sorry

end correct_answer_B_l43_43366


namespace total_output_correct_l43_43919

variable (a : ℝ)

-- Define a function that captures the total output from this year to the fifth year
def totalOutput (a : ℝ) : ℝ :=
  1.1 * a + (1.1 ^ 2) * a + (1.1 ^ 3) * a + (1.1 ^ 4) * a + (1.1 ^ 5) * a

theorem total_output_correct (a : ℝ) : 
  totalOutput a = 11 * (1.1 ^ 5 - 1) * a := by
  sorry

end total_output_correct_l43_43919


namespace total_turtles_taken_l43_43497

theorem total_turtles_taken (number_of_green_turtles number_of_hawksbill_turtles total_number_of_turtles : ℕ)
  (h1 : number_of_green_turtles = 800)
  (h2 : number_of_hawksbill_turtles = 2 * number_of_green_turtles)
  (h3 : total_number_of_turtles = number_of_green_turtles + number_of_hawksbill_turtles) :
  total_number_of_turtles = 2400 :=
by
  sorry

end total_turtles_taken_l43_43497


namespace sum_of_distinct_products_of_6_23H_508_3G4_l43_43189

theorem sum_of_distinct_products_of_6_23H_508_3G4 (G H : ℕ) : 
  (G < 10) → (H < 10) →
  (623 * 1000 + H * 100 + 508 * 10 + 3 * 10 + G * 1 + 4) % 72 = 0 →
  (if G = 0 then 0 + if G = 4 then 4 else 0 else 0) = 4 :=
by
  intros
  sorry

end sum_of_distinct_products_of_6_23H_508_3G4_l43_43189


namespace find_wrongly_noted_mark_l43_43465

-- Definitions of given conditions
def average_marks := 100
def number_of_students := 25
def reported_correct_mark := 10
def correct_average_marks := 98
def wrongly_noted_mark : ℕ := sorry

-- Computing the sum with the wrong mark
def incorrect_sum := number_of_students * average_marks

-- Sum corrected by replacing wrong mark with correct mark
def sum_with_correct_replacement (wrongly_noted_mark : ℕ) := 
  incorrect_sum - wrongly_noted_mark + reported_correct_mark

-- Correct total sum for correct average
def correct_sum := number_of_students * correct_average_marks

-- The statement to be proven
theorem find_wrongly_noted_mark : wrongly_noted_mark = 60 :=
by sorry

end find_wrongly_noted_mark_l43_43465


namespace fourth_vertex_l43_43144

-- Define the given vertices
def vertex1 := (2, 1)
def vertex2 := (4, 1)
def vertex3 := (2, 5)

-- Define what it means to be a rectangle in this context
def is_vertical_segment (p1 p2 : ℕ × ℕ) : Prop :=
  p1.1 = p2.1

def is_horizontal_segment (p1 p2 : ℕ × ℕ) : Prop :=
  p1.2 = p2.2

def is_rectangle (v1 v2 v3 v4: (ℕ × ℕ)) : Prop :=
  is_vertical_segment v1 v3 ∧
  is_horizontal_segment v1 v2 ∧
  is_vertical_segment v2 v4 ∧
  is_horizontal_segment v3 v4 ∧
  is_vertical_segment v1 v4 ∧ -- additional condition to ensure opposite sides are equal
  is_horizontal_segment v2 v3

-- Prove the coordinates of the fourth vertex of the rectangle
theorem fourth_vertex (v4 : ℕ × ℕ) : 
  is_rectangle vertex1 vertex2 vertex3 v4 → v4 = (4, 5) := 
by
  intro h_rect
  sorry

end fourth_vertex_l43_43144


namespace max_sum_of_squares_l43_43617

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17) 
  (h2 : ab + c + d = 85) 
  (h3 : ad + bc = 196) 
  (h4 : cd = 120) : 
  ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 918 :=
by {
  sorry
}

end max_sum_of_squares_l43_43617


namespace segment_length_l43_43418

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem segment_length (x : ℝ) 
  (h : |x - cbrt 27| = 5) : (abs ((cbrt 27 + 5) - (cbrt 27 - 5)) = 10) :=
by
  sorry

end segment_length_l43_43418


namespace candidate_a_votes_l43_43445

theorem candidate_a_votes (x : ℕ) (h : 2 * x + x = 21) : 2 * x = 14 :=
by sorry

end candidate_a_votes_l43_43445


namespace find_x_in_terms_of_z_l43_43850

variable (z : ℝ)
variable (x y : ℝ)

theorem find_x_in_terms_of_z (h1 : 0.35 * (400 + y) = 0.20 * x) 
                             (h2 : x = 2 * z^2) 
                             (h3 : y = 3 * z - 5) : 
  x = 2 * z^2 :=
by
  exact h2

end find_x_in_terms_of_z_l43_43850


namespace ratio_of_x_to_y_l43_43105

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x^2 - 2 * y^2) / (x^2 + 4 * y^2) = 5 / 7) : 
  x / y = Real.sqrt (17 / 8) :=
by
  sorry

end ratio_of_x_to_y_l43_43105


namespace toms_crab_buckets_l43_43341

def crabs_per_bucket := 12
def price_per_crab := 5
def weekly_earnings := 3360

theorem toms_crab_buckets : (weekly_earnings / (crabs_per_bucket * price_per_crab)) = 56 := by
  sorry

end toms_crab_buckets_l43_43341


namespace diane_harvest_increase_l43_43741

-- Define the conditions
def last_year_harvest : ℕ := 2479
def this_year_harvest : ℕ := 8564

-- Definition of the increase in honey harvest
def increase_in_harvest : ℕ := this_year_harvest - last_year_harvest

-- The theorem statement we need to prove
theorem diane_harvest_increase : increase_in_harvest = 6085 := 
by
  -- skip the proof for now
  sorry

end diane_harvest_increase_l43_43741


namespace cyclist_total_distance_l43_43927

-- Definitions for velocities and times
def v1 : ℝ := 2  -- velocity in the first minute (m/s)
def v2 : ℝ := 4  -- velocity in the second minute (m/s)
def t : ℝ := 60  -- time interval in seconds (1 minute)

-- Total distance covered in two minutes
def total_distance : ℝ := v1 * t + v2 * t

-- The proof statement
theorem cyclist_total_distance : total_distance = 360 := by
  sorry

end cyclist_total_distance_l43_43927


namespace smallest_x_y_sum_l43_43192

theorem smallest_x_y_sum (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y)
                        (h4 : (1 / (x : ℝ)) + (1 / (y : ℝ)) = (1 / 20)) :
    x + y = 81 :=
sorry

end smallest_x_y_sum_l43_43192


namespace trig_expression_value_l43_43552

theorem trig_expression_value (α : Real) (h : Real.tan (3 * Real.pi + α) = 3) :
  (Real.sin (α - 3 * Real.pi) + Real.cos (Real.pi - α) + Real.sin (Real.pi / 2 - α) - 2 * Real.cos (Real.pi / 2 + α)) /
  (-Real.sin (-α) + Real.cos (Real.pi + α)) = 3 :=
by
  sorry

end trig_expression_value_l43_43552


namespace sequence_term_n_l43_43215

theorem sequence_term_n (a : ℕ → ℕ) (a1 d : ℕ) (n : ℕ) (h1 : a 1 = a1) (h2 : d = 2)
  (h3 : a n = 19) (h_seq : ∀ n, a n = a1 + (n - 1) * d) : n = 10 :=
by
  sorry

end sequence_term_n_l43_43215


namespace geometric_sequence_problem_l43_43757

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 6) 
  (h2 : a 3 + a 5 + a 7 = 78) :
  a 5 = 18 :=
sorry

end geometric_sequence_problem_l43_43757


namespace combine_like_terms_l43_43479

theorem combine_like_terms : ∀ (x y : ℝ), -2 * x * y^2 + 2 * x * y^2 = 0 :=
by
  intros
  sorry

end combine_like_terms_l43_43479


namespace Chandler_more_rolls_needed_l43_43842

theorem Chandler_more_rolls_needed :
  let total_goal := 12
  let sold_to_grandmother := 3
  let sold_to_uncle := 4
  let sold_to_neighbor := 3
  let total_sold := sold_to_grandmother + sold_to_uncle + sold_to_neighbor
  total_goal - total_sold = 2 :=
by
  sorry

end Chandler_more_rolls_needed_l43_43842


namespace cube_root_opposite_zero_l43_43677

theorem cube_root_opposite_zero (x : ℝ) (h : x^(1/3) = -x) : x = 0 :=
sorry

end cube_root_opposite_zero_l43_43677


namespace calories_consumed_Jean_l43_43914

def donuts_per_page (pages : ℕ) : ℕ := pages / 2

def calories_per_donut : ℕ := 150

def total_calories (pages : ℕ) : ℕ :=
  let donuts := donuts_per_page pages
  donuts * calories_per_donut

theorem calories_consumed_Jean (h1 : ∀ pages, donuts_per_page pages = pages / 2)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories 12 = 900) :
  total_calories 12 = 900 := by
  sorry

end calories_consumed_Jean_l43_43914


namespace ratio_of_third_week_growth_l43_43203

-- Define the given conditions
def week1_growth : ℕ := 2  -- growth in week 1
def week2_growth : ℕ := 2 * week1_growth  -- growth in week 2
def total_height : ℕ := 22  -- total height after three weeks

/- 
  Statement: Prove that the growth in the third week divided by 
  the growth in the second week is 4, i.e., the ratio 4:1.
-/
theorem ratio_of_third_week_growth :
  ∃ x : ℕ, 4 * x = (total_height - week1_growth - week2_growth) ∧ x = 4 :=
by
  use 4
  sorry

end ratio_of_third_week_growth_l43_43203


namespace remainder_of_poly_division_l43_43245

theorem remainder_of_poly_division :
  ∀ (x : ℝ), (x^2023 + x + 1) % (x^6 - x^4 + x^2 - 1) = x^7 + x + 1 :=
by
  sorry

end remainder_of_poly_division_l43_43245


namespace trigonometric_expression_evaluation_l43_43330

theorem trigonometric_expression_evaluation :
  1 / Real.sin (70 * Real.pi / 180) - Real.sqrt 3 / Real.cos (70 * Real.pi / 180) = -4 :=
by
  sorry

end trigonometric_expression_evaluation_l43_43330


namespace minimum_filtration_process_l43_43529

noncomputable def filtration_process (n : ℕ) : Prop :=
  (0.8 : ℝ) ^ n < 0.05

theorem minimum_filtration_process : ∃ n : ℕ, filtration_process n ∧ n ≥ 14 := 
  sorry

end minimum_filtration_process_l43_43529


namespace men_with_ac_at_least_12_l43_43980

-- Define the variables and conditions
variable (total_men : ℕ) (married_men : ℕ) (tv_men : ℕ) (radio_men : ℕ) (men_with_all_four : ℕ)

-- Assume the given conditions
axiom h1 : total_men = 100
axiom h2 : married_men = 82
axiom h3 : tv_men = 75
axiom h4 : radio_men = 85
axiom h5 : men_with_all_four = 12

-- Define the number of men with AC
variable (ac_men : ℕ)

-- State the proposition that the number of men with AC is at least 12
theorem men_with_ac_at_least_12 : ac_men ≥ 12 := sorry

end men_with_ac_at_least_12_l43_43980


namespace bug_paths_l43_43838

-- Define the problem conditions
structure PathSetup (A B : Type) :=
  (red_arrows : ℕ) -- number of red arrows from point A
  (red_to_blue : ℕ) -- number of blue arrows reachable from each red arrow
  (blue_to_green : ℕ) -- number of green arrows reachable from each blue arrow
  (green_to_orange : ℕ) -- number of orange arrows reachable from each green arrow
  (start_arrows : ℕ) -- starting number of arrows from point A to red arrows
  (orange_arrows : ℕ) -- number of orange arrows equivalent to green arrows

-- Define the conditions for our specific problem setup
def problem_setup : PathSetup Point Point :=
  {
    red_arrows := 3,
    red_to_blue := 2,
    blue_to_green := 2,
    green_to_orange := 1,
    start_arrows := 3,
    orange_arrows := 6 * 2 * 2 -- derived from blue_to_green and red_to_blue steps
  }

-- Prove the number of unique paths from A to B
theorem bug_paths (setup : PathSetup Point Point) : 
  setup.start_arrows * setup.red_to_blue * setup.blue_to_green * setup.green_to_orange * setup.orange_arrows = 1440 :=
by
  -- Calculations are performed; exact values must hold
  sorry

end bug_paths_l43_43838


namespace equation_of_chord_l43_43592

-- Define the ellipse equation and point P
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144
def P : ℝ × ℝ := (3, 2)
def is_midpoint (A B P : ℝ × ℝ) : Prop := A.1 + B.1 = 2 * P.1 ∧ A.2 + B.2 = 2 * P.2
def on_chord (A B : ℝ × ℝ) (x y : ℝ) : Prop := (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1)

-- Lean Statement
theorem equation_of_chord :
  ∀ A B : ℝ × ℝ,
    ellipse_eq A.1 A.2 →
    ellipse_eq B.1 B.2 →
    is_midpoint A B P →
    ∀ x y : ℝ,
      on_chord A B x y →
      2 * x + 3 * y = 12 :=
by
  sorry

end equation_of_chord_l43_43592


namespace cost_of_six_burritos_and_seven_sandwiches_l43_43722

variable (b s : ℝ)
variable (h1 : 4 * b + 2 * s = 5.00)
variable (h2 : 3 * b + 5 * s = 6.50)

theorem cost_of_six_burritos_and_seven_sandwiches : 6 * b + 7 * s = 11.50 :=
  sorry

end cost_of_six_burritos_and_seven_sandwiches_l43_43722


namespace slope_of_parallel_line_l43_43302

theorem slope_of_parallel_line (x y : ℝ) :
  (∃ (b : ℝ), 3 * x - 6 * y = 12) → ∀ (m₁ x₁ y₁ x₂ y₂ : ℝ), (y₁ = (1/2) * x₁ + b) ∧ (y₂ = (1/2) * x₂ + b) → (x₁ ≠ x₂) → m₁ = 1/2 :=
by 
  sorry

end slope_of_parallel_line_l43_43302


namespace length_GH_l43_43631

def length_AB : ℕ := 11
def length_FE : ℕ := 13
def length_CD : ℕ := 5

theorem length_GH : length_AB + length_CD + length_FE = 29 :=
by
  refine rfl -- This will unroll the constants and perform arithmetic

end length_GH_l43_43631


namespace chocolate_bar_cost_l43_43626

theorem chocolate_bar_cost (total_bars : ℕ) (sold_bars : ℕ) (total_money : ℕ) (cost : ℕ) 
  (h1 : total_bars = 13)
  (h2 : sold_bars = total_bars - 4)
  (h3 : total_money = 18)
  (h4 : total_money = sold_bars * cost) :
  cost = 2 :=
by sorry

end chocolate_bar_cost_l43_43626


namespace sum_of_consecutive_integers_l43_43830

theorem sum_of_consecutive_integers (a : ℤ) (h₁ : a = 18) (h₂ : a + 1 = 19) (h₃ : a + 2 = 20) : a + (a + 1) + (a + 2) = 57 :=
by
  -- Add a sorry to focus on creating the statement successfully
  sorry

end sum_of_consecutive_integers_l43_43830


namespace player_A_winning_probability_l43_43322

theorem player_A_winning_probability :
  let P_draw := 1 / 2
  let P_B_wins := 1 / 3
  let P_total := 1
  P_total - P_draw - P_B_wins = 1 / 6 :=
by
  let P_draw := 1 / 2
  let P_B_wins := 1 / 3
  let P_total := 1
  sorry

end player_A_winning_probability_l43_43322


namespace louisa_second_day_miles_l43_43276

theorem louisa_second_day_miles (T1 T2 : ℕ) (speed miles_first_day miles_second_day : ℕ)
  (h1 : speed = 25) 
  (h2 : miles_first_day = 100)
  (h3 : T1 = miles_first_day / speed) 
  (h4 : T2 = T1 + 3) 
  (h5 : miles_second_day = speed * T2) :
  miles_second_day = 175 := 
by
  -- We can add the necessary calculations here, but for now, sorry is used to skip the proof.
  sorry

end louisa_second_day_miles_l43_43276


namespace polygon_sides_l43_43540

theorem polygon_sides (side_length perimeter : ℕ) (h1 : side_length = 4) (h2 : perimeter = 24) : 
  perimeter / side_length = 6 :=
by 
  sorry

end polygon_sides_l43_43540


namespace solution_is_unique_l43_43137

noncomputable def solution (f : ℝ → ℝ) (α : ℝ) :=
  ∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

theorem solution_is_unique (f : ℝ → ℝ) (α : ℝ)
  (h : solution f α) :
  f = id ∧ α = -1 :=
sorry

end solution_is_unique_l43_43137


namespace k_less_than_zero_l43_43618

variable (k : ℝ)

def function_decreases (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂

theorem k_less_than_zero (h : function_decreases (λ x => k * x - 5)) : k < 0 :=
sorry

end k_less_than_zero_l43_43618


namespace positive_number_equals_seven_l43_43577

theorem positive_number_equals_seven (x : ℝ) (h_pos : x > 0) (h_eq : x - 4 = 21 / x) : x = 7 :=
sorry

end positive_number_equals_seven_l43_43577


namespace frequency_count_third_group_l43_43290

theorem frequency_count_third_group 
  (x n : ℕ)
  (h1 : n = 420 - x)
  (h2 : x / (n:ℚ) = 0.20) :
  x = 70 :=
by sorry

end frequency_count_third_group_l43_43290


namespace magic_square_sum_l43_43877

theorem magic_square_sum (v w x y z : ℤ)
    (h1 : 25 + z + 23 = 25 + x + w)
    (h2 : 18 + x + y = 25 + x + w)
    (h3 : v + 22 + w = 25 + x + w)
    (h4 : 25 + 18 + v = 25 + x + w)
    (h5 : z + x + 22 = 25 + x + w)
    (h6 : 23 + y + w = 25 + x + w)
    (h7 : 25 + x + w = 25 + x + w)
    (h8 : v + x + 23 = 25 + x + w) 
:
    y + z = 45 :=
by
  sorry

end magic_square_sum_l43_43877


namespace no_int_representation_l43_43802

theorem no_int_representation (A B : ℤ) : (99999 + 111111 * Real.sqrt 3) ≠ (A + B * Real.sqrt 3)^2 :=
by
  sorry

end no_int_representation_l43_43802


namespace gcd_of_powers_l43_43613

theorem gcd_of_powers (a b : ℕ) (h1 : a = 2^300 - 1) (h2 : b = 2^315 - 1) :
  gcd a b = 32767 :=
by
  sorry

end gcd_of_powers_l43_43613


namespace range_of_m_l43_43894

open Set Real

noncomputable def A := {x : ℝ | x^2 - 2 * x - 3 < 0}
noncomputable def B (m : ℝ) := {x : ℝ | -1 < x ∧ x < m}

theorem range_of_m (m : ℝ) : 
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → 3 < m :=
by sorry

end range_of_m_l43_43894


namespace necessary_but_not_sufficient_condition_l43_43807

noncomputable def p (x : ℝ) : Prop := (1 - x^2 < 0 ∧ |x| - 2 > 0) ∨ (1 - x^2 > 0 ∧ |x| - 2 < 0)
noncomputable def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) :=
sorry

end necessary_but_not_sufficient_condition_l43_43807


namespace problem_l43_43086

open Real

noncomputable def f (x : ℝ) : ℝ := x + 1

theorem problem (f : ℝ → ℝ)
  (h : ∀ x, 2 * f x - f (-x) = 3 * x + 1) :
  f 1 = 2 :=
by
  sorry

end problem_l43_43086


namespace find_x_l43_43099

theorem find_x (x : ℕ) (hx1 : 1 ≤ x) (hx2 : x ≤ 100) (hx3 : (31 + 58 + 98 + 3 * x) / 6 = 2 * x) : x = 21 :=
by
  sorry

end find_x_l43_43099


namespace max_value_is_one_eighth_l43_43974

noncomputable def find_max_value (a b c : ℝ) : ℝ :=
  a^2 * b^2 * c^2 * (a + b + c) / ((a + b)^3 * (b + c)^3)

theorem max_value_is_one_eighth (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  find_max_value a b c ≤ 1 / 8 :=
by
  sorry

end max_value_is_one_eighth_l43_43974


namespace large_font_pages_l43_43284

theorem large_font_pages (L S : ℕ) (h1 : L + S = 21) (h2 : 3 * L = 2 * S) : L = 8 :=
by {
  sorry -- Proof can be filled in Lean; this ensures the statement aligns with problem conditions.
}

end large_font_pages_l43_43284


namespace largest_result_l43_43475

theorem largest_result :
  let A := (1 / 17 - 1 / 19) / 20
  let B := (1 / 15 - 1 / 21) / 60
  let C := (1 / 13 - 1 / 23) / 100
  let D := (1 / 11 - 1 / 25) / 140
  D > A ∧ D > B ∧ D > C := by
  sorry

end largest_result_l43_43475


namespace contradiction_proof_example_l43_43323

theorem contradiction_proof_example (a b : ℝ) (h: a ≤ b → False) : a > b :=
by sorry

end contradiction_proof_example_l43_43323


namespace cubic_root_sum_l43_43456

-- Assume we have three roots a, b, and c of the polynomial x^3 - 3x - 2 = 0
variables {a b c : ℝ}

-- Using Vieta's formulas for the polynomial x^3 - 3x - 2 = 0
axiom Vieta1 : a + b + c = 0
axiom Vieta2 : a * b + a * c + b * c = -3
axiom Vieta3 : a * b * c = -2

-- The proof that the given expression evaluates to 9
theorem cubic_root_sum:
  a^2 * (b - c)^2 + b^2 * (c - a)^2 + c^2 * (a - b)^2 = 9 :=
by
  sorry

end cubic_root_sum_l43_43456


namespace solve_inequality_l43_43255

theorem solve_inequality (x : ℝ) : 2 * x ^ 2 - 7 * x - 30 < 0 ↔ - (5 / 2) < x ∧ x < 6 := 
sorry

end solve_inequality_l43_43255


namespace root_quadratic_eq_l43_43143

theorem root_quadratic_eq (n m : ℝ) (h : n ≠ 0) (root_condition : n^2 + m * n + 3 * n = 0) : m + n = -3 :=
  sorry

end root_quadratic_eq_l43_43143


namespace sum_of_prism_features_l43_43416

theorem sum_of_prism_features : (12 + 8 + 6 = 26) := by
  sorry

end sum_of_prism_features_l43_43416


namespace min_tablets_to_extract_l43_43383

noncomputable def min_tablets_needed : ℕ :=
  let A := 10
  let B := 14
  let C := 18
  let D := 20
  let required_A := 3
  let required_B := 4
  let required_C := 3
  let required_D := 2
  let worst_case := B + C + D
  worst_case + required_A -- 14 + 18 + 20 + 3 = 55

theorem min_tablets_to_extract : min_tablets_needed = 55 :=
by {
  let A := 10
  let B := 14
  let C := 18
  let D := 20
  let required_A := 3
  let required_B := 4
  let required_C := 3
  let required_D := 2
  let worst_case := B + C + D
  have h : worst_case + required_A = 55 := by decide
  exact h
}

end min_tablets_to_extract_l43_43383


namespace product_of_three_numbers_l43_43570

theorem product_of_three_numbers :
  ∃ (x y z : ℚ), 
    (x + y + z = 30) ∧ 
    (x = 3 * (y + z)) ∧ 
    (y = 8 * z) ∧ 
    (x * y * z = 125) := 
by
  sorry

end product_of_three_numbers_l43_43570


namespace inequality_holds_l43_43089

noncomputable def positive_real_numbers := { x : ℝ // 0 < x }

theorem inequality_holds (a b c : positive_real_numbers) (h : (a.val * b.val + b.val * c.val + c.val * a.val) = 1) :
    (a.val / b.val + b.val / c.val + c.val / a.val) ≥ (a.val^2 + b.val^2 + c.val^2 + 2) :=
by
  sorry

end inequality_holds_l43_43089


namespace coffee_break_l43_43004

theorem coffee_break (n k : ℕ) (h1 : n = 14) (h2 : 0 < n - 2 * k) (h3 : n - 2 * k < n) :
  n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12 :=
by
  sorry

end coffee_break_l43_43004


namespace remainder_eq_52_l43_43916

noncomputable def polynomial : Polynomial ℤ := Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C (-4) * Polynomial.X ^ 2 + Polynomial.C 7

theorem remainder_eq_52 : Polynomial.eval (-3) polynomial = 52 :=
by
    sorry

end remainder_eq_52_l43_43916


namespace maximum_xy_l43_43522

theorem maximum_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) : xy ≤ 2 :=
sorry

end maximum_xy_l43_43522


namespace largest_minus_smallest_eq_13_l43_43051

theorem largest_minus_smallest_eq_13 :
  let a := (-1 : ℤ) ^ 3
  let b := (-1 : ℤ) ^ 2
  let c := -(2 : ℤ) ^ 2
  let d := (-3 : ℤ) ^ 2
  max (max a (max b c)) d - min (min a (min b c)) d = 13 := by
  sorry

end largest_minus_smallest_eq_13_l43_43051


namespace fraction_to_decimal_l43_43999

theorem fraction_to_decimal : (3 : ℝ) / 50 = 0.06 := by
  sorry

end fraction_to_decimal_l43_43999


namespace questionnaires_drawn_from_unit_D_l43_43782

theorem questionnaires_drawn_from_unit_D 
  (total_sample: ℕ) 
  (sample_from_B: ℕ) 
  (d: ℕ) 
  (h_total_sample: total_sample = 150) 
  (h_sample_from_B: sample_from_B = 30) 
  (h_arithmetic_sequence: (30 - d) + 30 + (30 + d) + (30 + 2 * d) = total_sample) 
  : 30 + 2 * d = 60 :=
by 
  sorry

end questionnaires_drawn_from_unit_D_l43_43782


namespace triangle_inequality_l43_43975

-- Define the conditions as Lean hypotheses
variables {a b c : ℝ}

-- Lean statement for the problem
theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 :=
sorry

end triangle_inequality_l43_43975


namespace right_triangle_cos_B_l43_43360

theorem right_triangle_cos_B (A B C : ℝ) (hC : C = 90) (hSinA : Real.sin A = 2 / 3) :
  Real.cos B = 2 / 3 :=
sorry

end right_triangle_cos_B_l43_43360


namespace find_sum_l43_43647

variables (a b c d : ℕ)

axiom h1 : 6 * a + 2 * b = 3848
axiom h2 : 6 * c + 3 * d = 4410
axiom h3 : a + 3 * b + 2 * d = 3080

theorem find_sum : a + b + c + d = 1986 :=
by
  sorry

end find_sum_l43_43647


namespace max_value_frac_sqrt_eq_sqrt_35_l43_43153

theorem max_value_frac_sqrt_eq_sqrt_35 :
  ∀ x y : ℝ, 
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 35 
  ∧ (∃ x y : ℝ, x = 2 / 5 ∧ y = 6 / 5 ∧ (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) = Real.sqrt 35) :=
by {
  sorry
}

end max_value_frac_sqrt_eq_sqrt_35_l43_43153


namespace probability_of_selecting_green_ball_l43_43294

def container_I :  ℕ × ℕ := (5, 5) -- (red balls, green balls)
def container_II : ℕ × ℕ := (3, 3) -- (red balls, green balls)
def container_III : ℕ × ℕ := (4, 2) -- (red balls, green balls)
def container_IV : ℕ × ℕ := (6, 6) -- (red balls, green balls)

def total_containers : ℕ := 4

def probability_of_green_ball (red_green : ℕ × ℕ) : ℚ :=
  let (red, green) := red_green
  green / (red + green)

noncomputable def combined_probability_of_green_ball : ℚ :=
  (1 / total_containers) *
  (probability_of_green_ball container_I +
   probability_of_green_ball container_II +
   probability_of_green_ball container_III +
   probability_of_green_ball container_IV)

theorem probability_of_selecting_green_ball : 
  combined_probability_of_green_ball = 11 / 24 :=
sorry

end probability_of_selecting_green_ball_l43_43294


namespace sum_of_digits_in_7_pow_1500_l43_43691

-- Define the problem and conditions
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem sum_of_digits_in_7_pow_1500 :
  sum_of_digits (7^1500) = 2 :=
by
  sorry

end sum_of_digits_in_7_pow_1500_l43_43691


namespace isosceles_triangle_base_function_l43_43697

theorem isosceles_triangle_base_function (x : ℝ) (hx : 5 < x ∧ x < 10) :
  ∃ y : ℝ, y = 20 - 2 * x := 
by
  sorry

end isosceles_triangle_base_function_l43_43697


namespace max_value_expression_l43_43511

theorem max_value_expression (a b : ℝ) (ha: 0 < a) (hb: 0 < b) :
  ∃ M, M = 2 * Real.sqrt 87 ∧
       (∀ a b: ℝ, 0 < a → 0 < b →
       (|4 * a - 10 * b| + |2 * (a - b * Real.sqrt 3) - 5 * (a * Real.sqrt 3 + b)|) / Real.sqrt (a ^ 2 + b ^ 2) ≤ M) :=
sorry

end max_value_expression_l43_43511


namespace Cindy_coins_l43_43412

theorem Cindy_coins (n : ℕ) (h1 : ∃ X Y : ℕ, n = X * Y ∧ Y > 1 ∧ Y < n) (h2 : ∀ Y, Y > 1 ∧ Y < n → ¬Y ∣ n → False) : n = 65536 :=
by
  sorry

end Cindy_coins_l43_43412


namespace periodic_sum_constant_l43_43961

noncomputable def is_periodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
a ≠ 0 ∧ ∀ x : ℝ, f (a + x) = f x

theorem periodic_sum_constant (f g : ℝ → ℝ) (a b : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hfa : is_periodic f a) (hgb : is_periodic g b)
  (harational : ∃ m n : ℤ, (a : ℝ) = m / n) (hbirrational : ¬ ∃ m n : ℤ, (b : ℝ) = m / n) :
  (∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, (f + g) (c + x) = (f + g) x) →
  (∀ x : ℝ, f x = f 0) ∨ (∀ x : ℝ, g x = g 0) :=
sorry

end periodic_sum_constant_l43_43961


namespace area_of_rectangle_l43_43450

-- Definitions of the conditions
def length (w : ℝ) : ℝ := 4 * w
def perimeter_eq_200 (w l : ℝ) : Prop := 2 * l + 2 * w = 200

-- Main theorem statement
theorem area_of_rectangle (w l : ℝ) (h1 : length w = l) (h2 : perimeter_eq_200 w l) : l * w = 1600 :=
by
  -- Skip the proof
  sorry

end area_of_rectangle_l43_43450


namespace problem_statement_l43_43876

def f (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n - 3

theorem problem_statement : f (f (f 3)) = 31 :=
by
  sorry

end problem_statement_l43_43876


namespace log2_monotone_l43_43687

theorem log2_monotone (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a > b) ↔ (Real.log a / Real.log 2 > Real.log b / Real.log 2) :=
sorry

end log2_monotone_l43_43687


namespace find_a_and_b_maximize_profit_l43_43547

variable (a b x : ℝ)

-- The given conditions
def condition1 : Prop := 2 * a + b = 120
def condition2 : Prop := 4 * a + 3 * b = 270
def constraint : Prop := 75 ≤ 300 - x

-- The questions translated into a proof problem
theorem find_a_and_b :
  condition1 a b ∧ condition2 a b → a = 45 ∧ b = 30 :=
by
  intros h
  sorry

theorem maximize_profit (a : ℝ) (b : ℝ) (x : ℝ) :
  condition1 a b → condition2 a b → constraint x →
  x = 75 → (300 - x) = 225 → 
  (10 * x + 20 * (300 - x) = 5250) :=
by
  intros h1 h2 hc hx hx1
  sorry

end find_a_and_b_maximize_profit_l43_43547


namespace number_of_articles_sold_at_cost_price_l43_43538

-- Let C be the cost price of one article.
-- Let S be the selling price of one article.
-- Let X be the number of articles sold at cost price.

variables (C S : ℝ) (X : ℕ)

-- Condition 1: The cost price of X articles is equal to the selling price of 32 articles.
axiom condition1 : (X : ℝ) * C = 32 * S

-- Condition 2: The profit is 25%, so the selling price S is 1.25 times the cost price C.
axiom condition2 : S = 1.25 * C

-- The theorem we need to prove
theorem number_of_articles_sold_at_cost_price : X = 40 :=
by
  -- Proof here
  sorry

end number_of_articles_sold_at_cost_price_l43_43538


namespace max_servings_l43_43966

-- Definitions based on the conditions
def servings_recipe := 3
def bananas_per_serving := 2 / servings_recipe
def strawberries_per_serving := 1 / servings_recipe
def yogurt_per_serving := 2 / servings_recipe

def emily_bananas := 4
def emily_strawberries := 3
def emily_yogurt := 6

-- Prove that Emily can make at most 6 servings while keeping the proportions the same
theorem max_servings :
  min (emily_bananas / bananas_per_serving) 
      (min (emily_strawberries / strawberries_per_serving) 
           (emily_yogurt / yogurt_per_serving)) = 6 := sorry

end max_servings_l43_43966


namespace polynomials_symmetric_l43_43196

noncomputable def P : ℕ → (ℝ → ℝ → ℝ → ℝ)
  | 0       => λ x y z => 1
  | (m + 1) => λ x y z => (x + z) * (y + z) * (P m x y (z + 1)) - z^2 * (P m x y z)

theorem polynomials_symmetric (m : ℕ) (x y z : ℝ) : 
  P m x y z = P m y x z ∧ P m x y z = P m x z y := 
sorry

end polynomials_symmetric_l43_43196


namespace data_point_frequency_l43_43703

theorem data_point_frequency 
  (data : Type) 
  (categories : data → Prop) 
  (group_counts : data → ℕ) :
  ∀ d, categories d → group_counts d = frequency := sorry

end data_point_frequency_l43_43703


namespace sphere_surface_area_from_volume_l43_43692

theorem sphere_surface_area_from_volume 
  (V : ℝ) (h : V = 72 * Real.pi) :
  ∃ (A : ℝ), A = 36 * Real.pi * 2^(2/3) :=
by
  sorry

end sphere_surface_area_from_volume_l43_43692


namespace quadratic_inequality_solution_set_l43_43458

theorem quadratic_inequality_solution_set (a b c : ℝ) (h₁ : a < 0) (h₂ : b^2 - 4 * a * c < 0) :
  ∀ x : ℝ, a * x^2 + b * x + c < 0 :=
sorry

end quadratic_inequality_solution_set_l43_43458


namespace inscribed_circle_radius_is_correct_l43_43624

noncomputable def radius_of_inscribed_circle (base height : ℝ) : ℝ := sorry

theorem inscribed_circle_radius_is_correct :
  radius_of_inscribed_circle 20 24 = 120 / 13 := sorry

end inscribed_circle_radius_is_correct_l43_43624


namespace probability_non_adjacent_sum_l43_43597

-- Definitions and conditions from the problem
def total_trees := 13
def maple_trees := 4
def oak_trees := 3
def birch_trees := 6

-- Total possible arrangements of 13 trees
def total_arrangements := Nat.choose total_trees birch_trees

-- Number of ways to arrange birch trees with no two adjacent
def favorable_arrangements := Nat.choose (maple_trees + oak_trees + 1) birch_trees

-- Probability calculation
def probability_non_adjacent := (favorable_arrangements : ℚ) / (total_arrangements : ℚ)

-- This value should be simplified to form m/n in lowest terms
def fraction_part_m := 7
def fraction_part_n := 429

-- Verify m + n
def sum_m_n := fraction_part_m + fraction_part_n

-- Check that sum_m_n is equal to 436
theorem probability_non_adjacent_sum :
  sum_m_n = 436 := by {
    -- Placeholder proof
    sorry
}

end probability_non_adjacent_sum_l43_43597


namespace bunchkin_total_distance_l43_43845

theorem bunchkin_total_distance
  (a b c d e : ℕ)
  (ha : a = 17)
  (hb : b = 43)
  (hc : c = 56)
  (hd : d = 66)
  (he : e = 76) :
  (a + b + c + d + e) / 2 = 129 :=
by
  sorry

end bunchkin_total_distance_l43_43845


namespace gina_tom_goals_l43_43324

theorem gina_tom_goals :
  let g_day1 := 2
  let t_day1 := g_day1 + 3
  let t_day2 := 6
  let g_day2 := t_day2 - 2
  let g_total := g_day1 + g_day2
  let t_total := t_day1 + t_day2
  g_total + t_total = 17 := by
  sorry

end gina_tom_goals_l43_43324


namespace sin_alpha_plus_3pi_div_2_l43_43627

theorem sin_alpha_plus_3pi_div_2 (α : ℝ) (h : Real.cos α = 1 / 3) : 
  Real.sin (α + 3 * Real.pi / 2) = -1 / 3 :=
by
  sorry

end sin_alpha_plus_3pi_div_2_l43_43627


namespace gcd_323_391_l43_43594

theorem gcd_323_391 : Nat.gcd 323 391 = 17 := 
by sorry

end gcd_323_391_l43_43594


namespace find_multiple_of_brothers_l43_43987

theorem find_multiple_of_brothers : 
  ∃ x : ℕ, (x * 4) - 2 = 6 :=
by
  -- Provide the correct Lean statement for the problem
  sorry

end find_multiple_of_brothers_l43_43987


namespace find_fourth_month_sale_l43_43754

theorem find_fourth_month_sale (s1 s2 s3 s4 s5 : ℕ) (avg_sale nL5 : ℕ)
  (h1 : s1 = 5420)
  (h2 : s2 = 5660)
  (h3 : s3 = 6200)
  (h5 : s5 = 6500)
  (havg : avg_sale = 6300)
  (hnL5 : nL5 = 5)
  (h_average : avg_sale * nL5 = s1 + s2 + s3 + s4 + s5) :
  s4 = 7720 := sorry

end find_fourth_month_sale_l43_43754


namespace total_bricks_required_l43_43220

def courtyard_length : ℕ := 24 * 100  -- convert meters to cm
def courtyard_width : ℕ := 14 * 100  -- convert meters to cm
def brick_length : ℕ := 25
def brick_width : ℕ := 15

-- Calculate the area of the courtyard in square centimeters
def courtyard_area : ℕ := courtyard_length * courtyard_width

-- Calculate the area of one brick in square centimeters
def brick_area : ℕ := brick_length * brick_width

theorem total_bricks_required :  courtyard_area / brick_area = 8960 := by
  -- This part will have the proof, for now, we use sorry to skip it
  sorry

end total_bricks_required_l43_43220


namespace f_even_l43_43765

variable (g : ℝ → ℝ)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

def f (x : ℝ) := |g (x^2)|

theorem f_even (h_g_odd : is_odd g) : ∀ x : ℝ, f g x = f g (-x) :=
by
  intro x
  -- Proof can be added here
  sorry

end f_even_l43_43765


namespace sum_of_constants_eq_17_l43_43642

theorem sum_of_constants_eq_17
  (x y : ℝ)
  (a b c d : ℕ)
  (ha : a = 6)
  (hb : b = 2)
  (hc : c = 3)
  (hd : d = 3)
  (h1 : x + y = 4)
  (h2 : 3 * x * y = 4)
  (h3 : x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d) :
  a + b + c + d = 17 :=
sorry

end sum_of_constants_eq_17_l43_43642


namespace acres_for_corn_l43_43759

theorem acres_for_corn (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ)
  (total_ratio : beans_ratio + wheat_ratio + corn_ratio = 11)
  (land_parts : total_land / 11 = 94)
  : (corn_ratio = 4) → (total_land = 1034) → 4 * 94 = 376 :=
by
  intros
  sorry

end acres_for_corn_l43_43759


namespace equation_of_parallel_line_l43_43650

theorem equation_of_parallel_line (c : ℕ) :
  (∃ c, x + 2 * y + c = 0) ∧ (1 + 2 * 1 + c = 0) -> x + 2 * y - 3 = 0 :=
by 
  sorry

end equation_of_parallel_line_l43_43650


namespace perfect_squares_in_range_100_400_l43_43955

theorem perfect_squares_in_range_100_400 : ∃ n : ℕ, (∀ m, 100 ≤ m^2 → m^2 ≤ 400 → m^2 = (m - 10 + 1)^2) ∧ n = 9 := 
by
  sorry

end perfect_squares_in_range_100_400_l43_43955


namespace length_of_square_side_is_correct_l43_43019

noncomputable def length_of_square_side : ℚ :=
  let PQ : ℚ := 7
  let QR : ℚ := 24
  let hypotenuse := (PQ^2 + QR^2).sqrt
  (25 * 175) / (24 * 32)

theorem length_of_square_side_is_correct :
  length_of_square_side = 4375 / 768 := 
by 
  sorry

end length_of_square_side_is_correct_l43_43019


namespace power_function_solution_l43_43781

theorem power_function_solution (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = x ^ α) (h2 : f 4 = 2) : f 3 = Real.sqrt 3 :=
sorry

end power_function_solution_l43_43781


namespace problem_l43_43485

variable (a b c d : ℕ)

theorem problem (h1 : a + b = 12) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 6 :=
sorry

end problem_l43_43485


namespace last_child_loses_l43_43100

-- Definitions corresponding to conditions
def num_children := 11
def child_sequence := List.range' 1 num_children
def valid_two_digit_numbers := 90
def invalid_digit_sum_6 := 6
def invalid_digit_sum_9 := 9
def valid_numbers := valid_two_digit_numbers - invalid_digit_sum_6 - invalid_digit_sum_9
def complete_cycles := valid_numbers / num_children
def remaining_numbers := valid_numbers % num_children

-- Statement to be proven
theorem last_child_loses (h1 : num_children = 11)
                         (h2 : valid_two_digit_numbers = 90)
                         (h3 : invalid_digit_sum_6 = 6)
                         (h4 : invalid_digit_sum_9 = 9)
                         (h5 : valid_numbers = valid_two_digit_numbers - invalid_digit_sum_6 - invalid_digit_sum_9)
                         (h6 : remaining_numbers = valid_numbers % num_children) :
  (remaining_numbers = 9) ∧ (num_children - remaining_numbers = 2) :=
by
  sorry

end last_child_loses_l43_43100


namespace fisherman_daily_earnings_l43_43902

def red_snapper_quantity : Nat := 8
def tuna_quantity : Nat := 14
def red_snapper_cost : Nat := 3
def tuna_cost : Nat := 2

theorem fisherman_daily_earnings
  (rs_qty : Nat := red_snapper_quantity)
  (t_qty : Nat := tuna_quantity)
  (rs_cost : Nat := red_snapper_cost)
  (t_cost : Nat := tuna_cost) :
  rs_qty * rs_cost + t_qty * t_cost = 52 := 
by {
  sorry
}

end fisherman_daily_earnings_l43_43902


namespace base9_minus_base6_to_decimal_l43_43169

theorem base9_minus_base6_to_decimal :
  let b9 := 3 * 9^2 + 2 * 9^1 + 1 * 9^0
  let b6 := 2 * 6^2 + 5 * 6^1 + 4 * 6^0
  b9 - b6 = 156 := by
sorry

end base9_minus_base6_to_decimal_l43_43169


namespace find_A2_A7_l43_43760

theorem find_A2_A7 (A : ℕ → ℝ) (hA1A11 : A 11 - A 1 = 56)
  (hAiAi2 : ∀ i, 1 ≤ i ∧ i ≤ 9 → A (i+2) - A i ≤ 12)
  (hAjAj3 : ∀ j, 1 ≤ j ∧ j ≤ 8 → A (j+3) - A j ≥ 17) : 
  A 7 - A 2 = 29 :=
by
  sorry

end find_A2_A7_l43_43760


namespace number_of_tons_is_3_l43_43996

noncomputable def calculate_tons_of_mulch {total_cost price_per_pound pounds_per_ton : ℝ} 
  (h_total_cost : total_cost = 15000) 
  (h_price_per_pound : price_per_pound = 2.5) 
  (h_pounds_per_ton : pounds_per_ton = 2000) : ℝ := 
  total_cost / price_per_pound / pounds_per_ton

theorem number_of_tons_is_3 
  (total_cost price_per_pound pounds_per_ton : ℝ) 
  (h_total_cost : total_cost = 15000) 
  (h_price_per_pound : price_per_pound = 2.5) 
  (h_pounds_per_ton : pounds_per_ton = 2000) : 
  calculate_tons_of_mulch h_total_cost h_price_per_pound h_pounds_per_ton = 3 := 
by
  sorry

end number_of_tons_is_3_l43_43996


namespace sum_of_coefficients_l43_43774

def P (x : ℝ) : ℝ :=
  -3 * (x^8 - x^5 + 2*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)

theorem sum_of_coefficients : P 1 = 48 := by
  sorry

end sum_of_coefficients_l43_43774


namespace candy_necklaces_l43_43567

theorem candy_necklaces (friends : ℕ) (candies_per_necklace : ℕ) (candies_per_block : ℕ)(blocks_needed : ℕ):
  friends = 8 →
  candies_per_necklace = 10 →
  candies_per_block = 30 →
  80 / 30 > 2.67 →
  blocks_needed = 3 :=
by
  intros
  sorry

end candy_necklaces_l43_43567


namespace count_integers_divis_by_8_l43_43056

theorem count_integers_divis_by_8 : 
  ∃ k : ℕ, k = 49 ∧ ∀ n : ℕ, 2 ≤ n ∧ n ≤ 80 → (∃ m : ℤ, (n-1) * n * (n+1) = 8 * m) ↔ (∃ m : ℕ, m ≤ k) :=
by 
  sorry

end count_integers_divis_by_8_l43_43056


namespace cost_of_paving_floor_l43_43041

-- Definitions of the constants
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_meter : ℝ := 400

-- Definitions of the calculated area and cost
def area : ℝ := length * width
def cost : ℝ := area * rate_per_sq_meter

-- Statement to prove
theorem cost_of_paving_floor : cost = 8250 := by
  sorry

end cost_of_paving_floor_l43_43041


namespace average_monthly_income_P_and_R_l43_43995

theorem average_monthly_income_P_and_R 
  (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250)
  (h3 : P = 4000) :
  (P + R) / 2 = 5200 :=
sorry

end average_monthly_income_P_and_R_l43_43995


namespace game_positions_l43_43651

def spots := ["top-left", "top-right", "bottom-right", "bottom-left"]
def segments := ["top-left", "top-middle-left", "top-middle-right", "top-right", "right-top", "right-middle-top", "right-middle-bottom", "right-bottom", "bottom-right", "bottom-middle-right", "bottom-middle-left", "bottom-left", "left-top", "left-middle-top", "left-middle-bottom", "left-bottom"]

def cat_position_after_moves (n : Nat) : String :=
  spots.get! (n % 4)

def mouse_position_after_moves (n : Nat) : String :=
  segments.get! ((12 - (n % 12)) % 12)

theorem game_positions :
  cat_position_after_moves 359 = "bottom-right" ∧ 
  mouse_position_after_moves 359 = "left-middle-bottom" :=
by
  sorry

end game_positions_l43_43651


namespace number_of_players_l43_43287

-- Definitions based on conditions
def socks_price : ℕ := 6
def tshirt_price : ℕ := socks_price + 7
def total_cost_per_player : ℕ := 2 * (socks_price + tshirt_price)
def total_expenditure : ℕ := 4092

-- Lean theorem statement
theorem number_of_players : total_expenditure / total_cost_per_player = 108 := 
by
  sorry

end number_of_players_l43_43287


namespace length_four_implies_value_twenty_four_l43_43291

-- Definition of prime factors of an integer
def prime_factors (n : ℕ) : List ℕ := sorry

-- Definition of the length of an integer
def length_of_integer (n : ℕ) : ℕ :=
  List.length (prime_factors n)

-- Statement of the problem
theorem length_four_implies_value_twenty_four (k : ℕ) (h1 : k > 1) (h2 : length_of_integer k = 4) : k = 24 :=
by
  sorry

end length_four_implies_value_twenty_four_l43_43291


namespace starters_choice_l43_43508

/-- There are 18 players including a set of quadruplets: Bob, Bill, Ben, and Bert. -/
def total_players : ℕ := 18

/-- The set of quadruplets: Bob, Bill, Ben, and Bert. -/
def quadruplets : Finset (String) := {"Bob", "Bill", "Ben", "Bert"}

/-- We need to choose 7 starters, exactly 3 of which are from the set of quadruplets. -/
def ways_to_choose_starters : ℕ :=
  let quadruplet_combinations := Nat.choose 4 3
  let remaining_spots := 4
  let remaining_players := total_players - 4
  quadruplet_combinations * Nat.choose remaining_players remaining_spots

theorem starters_choice (h1 : total_players = 18)
                        (h2 : quadruplets.card = 4) :
  ways_to_choose_starters = 4004 :=
by 
  -- conditional setups here
  sorry

end starters_choice_l43_43508


namespace remainder_3211_div_103_l43_43270

theorem remainder_3211_div_103 :
  3211 % 103 = 18 :=
by
  sorry

end remainder_3211_div_103_l43_43270


namespace base_8_add_sub_l43_43503

-- Definitions of the numbers in base 8
def n1 : ℕ := 4 * 8^2 + 5 * 8^1 + 1 * 8^0
def n2 : ℕ := 1 * 8^2 + 6 * 8^1 + 2 * 8^0
def n3 : ℕ := 1 * 8^2 + 2 * 8^1 + 3 * 8^0

-- Convert the result to base 8
def to_base_8 (n : ℕ) : ℕ :=
  let d2 := n / 64
  let rem1 := n % 64
  let d1 := rem1 / 8
  let d0 := rem1 % 8
  d2 * 100 + d1 * 10 + d0

-- Proof statement
theorem base_8_add_sub :
  to_base_8 ((n1 + n2) - n3) = to_base_8 (5 * 8^2 + 1 * 8^1 + 0 * 8^0) :=
by
  sorry

end base_8_add_sub_l43_43503


namespace ways_to_distribute_balls_l43_43831

theorem ways_to_distribute_balls :
  let balls : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}
  let boxes : Finset ℕ := {0, 1, 2, 3}
  let choose_distinct (n k : ℕ) : ℕ := Nat.choose n k
  let distribution_patterns : List (ℕ × ℕ × ℕ × ℕ) := 
    [(6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0), 
     (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)]
  let ways_to_pattern (pattern : ℕ × ℕ × ℕ × ℕ) : ℕ :=
    match pattern with
    | (6,0,0,0) => 1
    | (5,1,0,0) => choose_distinct 6 5
    | (4,2,0,0) => choose_distinct 6 4 * choose_distinct 2 2
    | (4,1,1,0) => choose_distinct 6 4
    | (3,3,0,0) => choose_distinct 6 3 * choose_distinct 3 3 / 2
    | (3,2,1,0) => choose_distinct 6 3 * choose_distinct 3 2 * choose_distinct 1 1
    | (3,1,1,1) => choose_distinct 6 3
    | (2,2,2,0) => choose_distinct 6 2 * choose_distinct 4 2 * choose_distinct 2 2 / 6
    | (2,2,1,1) => choose_distinct 6 2 * choose_distinct 4 2 / 2
    | _ => 0
  let total_ways : ℕ := distribution_patterns.foldl (λ acc x => acc + ways_to_pattern x) 0
  total_ways = 182 := by
  sorry

end ways_to_distribute_balls_l43_43831


namespace fraction_equivalence_l43_43419

variable {m n p q : ℚ}

theorem fraction_equivalence
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5) :
  m / q = 1 :=
by {
  sorry
}

end fraction_equivalence_l43_43419


namespace problem_statement_l43_43849

variable (a b : ℝ)

-- Conditions
variable (h1 : a > 0) (h2 : b > 0) (h3 : ∃ x, x = (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a)))

-- The Lean theorem statement for the problem
theorem problem_statement : 
  ∀ x, (x = (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))) →
  (2 * a * Real.sqrt (1 + x^2)) / (x + Real.sqrt (1 + x^2)) = a + b := 
sorry


end problem_statement_l43_43849


namespace geometric_sequence_solution_l43_43533

-- Define the geometric sequence a_n with a common ratio q and first term a_1
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * q^n

-- Given conditions in the problem
variables {a : ℕ → ℝ} {q a1 : ℝ}

-- Common ratio is greater than 1
axiom ratio_gt_one : q > 1

-- Given conditions a_3a_7 = 72 and a_2 + a_8 = 27
axiom condition1 : a 3 * a 7 = 72
axiom condition2 : a 2 + a 8 = 27

-- Defining the property that we are looking to prove a_12 = 96
theorem geometric_sequence_solution :
  geometric_sequence a a1 q →
  a 12 = 96 :=
by
  -- This part of the proof would be filled in
  -- Show the conditions and relations leading to the solution a_12 = 96
  sorry

end geometric_sequence_solution_l43_43533


namespace necessary_but_not_sufficient_l43_43114

theorem necessary_but_not_sufficient (x y : ℕ) : x + y = 3 → (x = 1 ∧ y = 2) ↔ (¬ (x = 0 ∧ y = 3)) := by
  sorry

end necessary_but_not_sufficient_l43_43114


namespace find_f2_g2_l43_43128

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x
def equation (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x - g x = x^3 + 2^(-x)

theorem find_f2_g2 (f g : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : odd_function g)
  (h3 : equation f g) :
  f 2 + g 2 = -2 :=
sorry

end find_f2_g2_l43_43128


namespace num_7_digit_integers_correct_l43_43662

-- Define the number of choices for each digit
def first_digit_choices : ℕ := 9
def other_digit_choices : ℕ := 10

-- Define the number of 7-digit positive integers
def num_7_digit_integers : ℕ := first_digit_choices * other_digit_choices^6

-- State the theorem to prove
theorem num_7_digit_integers_correct : num_7_digit_integers = 9000000 :=
by
  sorry

end num_7_digit_integers_correct_l43_43662


namespace percent_defective_units_shipped_l43_43011

theorem percent_defective_units_shipped (h1 : 8 / 100 * 4 / 100 = 32 / 10000) :
  (32 / 10000) * 100 = 0.32 := 
sorry

end percent_defective_units_shipped_l43_43011


namespace correct_transformation_l43_43069

structure Point :=
  (x : ℝ)
  (y : ℝ)

def rotate180 (p : Point) : Point :=
  Point.mk (-p.x) (-p.y)

def is_rotation_180 (p p' : Point) : Prop :=
  rotate180 p = p'

theorem correct_transformation (C D : Point) (C' D' : Point) 
  (hC : C = Point.mk 3 (-2)) 
  (hC' : C' = Point.mk (-3) 2)
  (hD : D = Point.mk 2 (-5)) 
  (hD' : D' = Point.mk (-2) 5) :
  is_rotation_180 C C' ∧ is_rotation_180 D D' :=
by
  sorry

end correct_transformation_l43_43069


namespace leak_drain_time_l43_43956

theorem leak_drain_time (P L : ℝ) (hP : P = 1/2) (h_combined : P - L = 3/7) : 1 / L = 14 :=
by
  -- Definitions of the conditions
  -- The rate of the pump filling the tank
  have hP : P = 1 / 2 := hP
  -- The combined rate of the pump (filling) and leak (draining)
  have h_combined : P - L = 3 / 7 := h_combined
  -- From these definitions, continue the proof
  sorry

end leak_drain_time_l43_43956


namespace arrange_numbers_l43_43512

namespace MathProofs

theorem arrange_numbers (a b : ℚ) (h1 : a > 0) (h2 : b < 0) (h3 : a + b < 0) :
  b < -a ∧ -a < a ∧ a < -b :=
by
  -- Proof to be completed
  sorry

end MathProofs

end arrange_numbers_l43_43512


namespace min_fuse_length_l43_43737

theorem min_fuse_length 
  (safe_distance : ℝ := 70) 
  (personnel_speed : ℝ := 7) 
  (fuse_burning_speed : ℝ := 10.3) : 
  ∃ (x : ℝ), x ≥ 103 := 
by
  sorry

end min_fuse_length_l43_43737


namespace solution_set_of_inequality_l43_43600

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 6 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l43_43600


namespace find_t1_t2_l43_43799

-- Define the vectors a and b
def a (t : ℝ) : ℝ × ℝ := (2, t)
def b : ℝ × ℝ := (1, 2)

-- Define the conditions for t1 and t2
def t1_condition (t1 : ℝ) : Prop := (2 / 1) = (t1 / 2)
def t2_condition (t2 : ℝ) : Prop := (2 * 1 + t2 * 2 = 0)

-- The statement to prove
theorem find_t1_t2 (t1 t2 : ℝ) (h1 : t1_condition t1) (h2 : t2_condition t2) : (t1 = 4) ∧ (t2 = -1) :=
by
  sorry

end find_t1_t2_l43_43799


namespace negate_exists_l43_43214

theorem negate_exists : 
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end negate_exists_l43_43214


namespace combined_distance_is_12_l43_43588

-- Define the distances the two ladies walked
def distance_second_lady : ℝ := 4
def distance_first_lady := 2 * distance_second_lady

-- Define the combined total distance
def combined_distance := distance_first_lady + distance_second_lady

-- Statement of the problem as a proof goal in Lean
theorem combined_distance_is_12 : combined_distance = 12 :=
by
  -- Definitions required for the proof
  let second := distance_second_lady
  let first := distance_first_lady
  let total := combined_distance
  
  -- Insert the necessary calculations and proof steps here
  -- Conclude with the desired result
  sorry

end combined_distance_is_12_l43_43588


namespace Jill_llamas_count_l43_43743

theorem Jill_llamas_count :
  let initial_pregnant_with_one_calf := 9
  let initial_pregnant_with_twins := 5
  let total_calves_born := (initial_pregnant_with_one_calf * 1) + (initial_pregnant_with_twins * 2)
  let calves_after_trade := total_calves_born - 8
  let initial_pregnant_lamas := initial_pregnant_with_one_calf + initial_pregnant_with_twins
  let total_lamas_after_birth := initial_pregnant_lamas + total_calves_born
  let lamas_after_trade := total_lamas_after_birth - 8 + 2
  let lamas_sold := lamas_after_trade / 3
  let final_lamas := lamas_after_trade - lamas_sold
  final_lamas = 18 :=
by
  sorry

end Jill_llamas_count_l43_43743


namespace true_proposition_l43_43399

def p : Prop := ∃ x₀ : ℝ, x₀^2 < x₀
def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem true_proposition : p ∧ q :=
by 
  sorry

end true_proposition_l43_43399


namespace find_x_given_y_l43_43517

noncomputable def constantRatio : Prop :=
  ∃ k : ℚ, ∀ x y : ℚ, (5 * x - 6) / (2 * y + 10) = k

theorem find_x_given_y :
  (constantRatio ∧ (3, 2) ∈ {(x, y) | (5 * x - 6) / (2 * y + 10) = 9 / 14}) →
  ∃ x : ℚ, ((5 * x - 6) / 20 = 9 / 14 ∧ x = 53 / 14) :=
by
  sorry

end find_x_given_y_l43_43517


namespace lawn_length_l43_43846

-- Defining the main conditions
def area : ℕ := 20
def width : ℕ := 5

-- The proof statement (goal)
theorem lawn_length : (area / width) = 4 := by
  sorry

end lawn_length_l43_43846


namespace find_cd_minus_dd_base_d_l43_43316

namespace MathProof

variables (d C D : ℤ)

def digit_sum (C D : ℤ) (d : ℤ) : ℤ := d * C + D
def digit_sum_same (C : ℤ) (d : ℤ) : ℤ := d * C + C

theorem find_cd_minus_dd_base_d (h_d : d > 8) (h_eq : digit_sum C D d + digit_sum_same C d = d^2 + 8 * d + 4) :
  C - D = 1 :=
by
  sorry

end MathProof

end find_cd_minus_dd_base_d_l43_43316


namespace percentage_paid_l43_43837

/-- 
Given the marked price is 80% of the suggested retail price,
and Alice paid 60% of the marked price,
prove that the percentage of the suggested retail price Alice paid is 48%.
-/
theorem percentage_paid (P : ℝ) (MP : ℝ) (price_paid : ℝ)
  (h1 : MP = 0.80 * P)
  (h2 : price_paid = 0.60 * MP) :
  (price_paid / P) * 100 = 48 := 
sorry

end percentage_paid_l43_43837


namespace carl_olivia_cookie_difference_l43_43349

-- Defining the various conditions
def Carl_cookies : ℕ := 7
def Olivia_cookies : ℕ := 2

-- Stating the theorem we need to prove
theorem carl_olivia_cookie_difference : Carl_cookies - Olivia_cookies = 5 :=
by sorry

end carl_olivia_cookie_difference_l43_43349


namespace total_hours_charged_l43_43236

theorem total_hours_charged (K P M : ℕ) 
  (h₁ : P = 2 * K)
  (h₂ : P = (1 / 3 : ℚ) * (K + 80))
  (h₃ : M = K + 80) : K + P + M = 144 :=
by {
    sorry
}

end total_hours_charged_l43_43236


namespace problem_part1_problem_part2_l43_43943

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (b * x / Real.log x) - (a * x)
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ :=
  (b * (Real.log x - 1) / (Real.log x)^2) - a

theorem problem_part1 (a b : ℝ) :
  (f' (Real.exp 2) a b = -(3/4)) ∧ (f (Real.exp 2) a b = -(1/2) * (Real.exp 2)) →
  a = 1 ∧ b = 1 :=
sorry

theorem problem_part2 (a : ℝ) :
  (∃ x1 x2, x1 ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ x2 ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ f x1 a 1 ≤ f' x2 a 1 + a) →
  a ≥ (1/2 - 1/(4 * Real.exp 2)) :=
sorry

end problem_part1_problem_part2_l43_43943


namespace sector_area_l43_43132

-- Define the given parameters
def central_angle : ℝ := 2
def radius : ℝ := 3

-- Define the statement about the area of the sector
theorem sector_area (α r : ℝ) (hα : α = 2) (hr : r = 3) :
  let l := α * r
  let A := 0.5 * l * r
  A = 9 :=
by
  -- The proof is not required
  sorry

end sector_area_l43_43132


namespace complex_identity_l43_43285

open Complex

noncomputable def z := 1 + 2 * I
noncomputable def z_inv := (1 - 2 * I) / 5
noncomputable def z_conj := 1 - 2 * I

theorem complex_identity : 
  (z + z_inv) * z_conj = (22 / 5 : ℂ) - (4 / 5) * I := 
by
  sorry

end complex_identity_l43_43285


namespace part_I_part_II_l43_43476

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
noncomputable def g (x : ℝ) := |2 * x - 1|

theorem part_I (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end part_I_part_II_l43_43476


namespace problem_solution_l43_43495

theorem problem_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := 
by
  sorry

end problem_solution_l43_43495


namespace marbles_difference_l43_43402

theorem marbles_difference {red_marbles blue_marbles : ℕ} 
  (h₁ : red_marbles = 288) (bags_red : ℕ) (h₂ : bags_red = 12) 
  (h₃ : blue_marbles = 243) (bags_blue : ℕ) (h₄ : bags_blue = 9) :
  (blue_marbles / bags_blue) - (red_marbles / bags_red) = 3 :=
by
  sorry

end marbles_difference_l43_43402


namespace first_expression_second_expression_l43_43024

-- Define the variables
variables {a x y : ℝ}

-- Statement for the first expression
theorem first_expression (a : ℝ) : (2 * a^2)^3 + (-3 * a^3)^2 = 17 * a^6 := sorry

-- Statement for the second expression
theorem second_expression (x y : ℝ) : (x + 3 * y) * (x - y) = x^2 + 2 * x * y - 3 * y^2 := sorry

end first_expression_second_expression_l43_43024


namespace sixth_graders_count_l43_43082

theorem sixth_graders_count (total_students seventh_graders_percentage sixth_graders_percentage : ℝ)
                            (seventh_graders_count : ℕ)
                            (h1 : seventh_graders_percentage = 0.32)
                            (h2 : seventh_graders_count = 64)
                            (h3 : sixth_graders_percentage = 0.38)
                            (h4 : seventh_graders_count = seventh_graders_percentage * total_students) :
                            sixth_graders_percentage * total_students = 76 := by
  sorry

end sixth_graders_count_l43_43082


namespace trapezoid_perimeter_l43_43792

-- Define the problem conditions
variables (A B C D : Point) (BC AD : Line) (AB CD : Segment)

-- Conditions
def is_parallel (L1 L2 : Line) : Prop := sorry
def is_right_angle (A B C : Point) : Prop := sorry
def is_angle_150 (A B C : Point) : Prop := sorry

noncomputable def length (s : Segment) : ℝ := sorry

def trapezoid_conditions (A B C D : Point) (BC AD : Line) (AB CD : Segment) : Prop :=
  is_parallel BC AD ∧ is_angle_150 A B C ∧ is_right_angle C D B ∧
  length AB = 4 ∧ length BC = 3 - Real.sqrt 3

-- Perimeter calculation
noncomputable def perimeter (A B C D : Point) (BC AD : Line) (AB CD : Segment) : ℝ :=
  length AB + length BC + length CD + length AD

-- Lean statement for the math proof problem
theorem trapezoid_perimeter (A B C D : Point) (BC AD : Line) (AB CD : Segment) :
  trapezoid_conditions A B C D BC AD AB CD → perimeter A B C D BC AD AB CD = 12 :=
sorry

end trapezoid_perimeter_l43_43792


namespace simplify_and_evaluate_l43_43123

noncomputable def expr (x : ℝ) : ℝ :=
  ((x^2 + x - 2) / (x - 2) - x - 2) / ((x^2 + 4 * x + 4) / x)

theorem simplify_and_evaluate : expr 1 = -1 / 3 :=
by
  sorry

end simplify_and_evaluate_l43_43123


namespace pamela_spilled_sugar_l43_43422

theorem pamela_spilled_sugar 
  (original_amount : ℝ)
  (amount_left : ℝ)
  (h1 : original_amount = 9.8)
  (h2 : amount_left = 4.6)
  : original_amount - amount_left = 5.2 :=
by 
  sorry

end pamela_spilled_sugar_l43_43422


namespace total_votes_l43_43583

theorem total_votes (V : ℕ) 
  (h1 : V * 45 / 100 + V * 25 / 100 + V * 15 / 100 + 180 + 50 = V) : 
  V = 1533 := 
by
  sorry

end total_votes_l43_43583


namespace border_area_correct_l43_43658

-- Define the dimensions of the photograph
def photograph_height : ℕ := 12
def photograph_width : ℕ := 15

-- Define the width of the border
def border_width : ℕ := 3

-- Define the area of the photograph
def area_photograph : ℕ := photograph_height * photograph_width

-- Define the total dimensions including the frame
def total_height : ℕ := photograph_height + 2 * border_width
def total_width : ℕ := photograph_width + 2 * border_width

-- Define the area of the framed area
def area_framed : ℕ := total_height * total_width

-- Define the area of the border
def area_border : ℕ := area_framed - area_photograph

theorem border_area_correct : area_border = 198 := by
  sorry

end border_area_correct_l43_43658


namespace polynomial_remainder_division_l43_43504

theorem polynomial_remainder_division (x : ℝ) : 
  (x^4 + 1) % (x^2 - 4 * x + 6) = 16 * x - 59 := 
sorry

end polynomial_remainder_division_l43_43504


namespace exists_g_l43_43969

variable {R : Type} [Field R]

-- Define the function f with the given condition
def f (x y : R) : R := sorry

-- The main theorem to prove the existence of g
theorem exists_g (f_condition: ∀ x y z : R, f x y + f y z + f z x = 0) : ∃ g : R → R, ∀ x y : R, f x y = g x - g y := 
by 
  sorry

end exists_g_l43_43969


namespace total_students_is_17_l43_43501

def total_students_in_class (students_liking_both_baseball_football : ℕ)
                             (students_only_baseball : ℕ)
                             (students_only_football : ℕ)
                             (students_liking_basketball_as_well : ℕ)
                             (students_liking_basketball_and_football_only : ℕ)
                             (students_liking_all_three : ℕ)
                             (students_liking_none : ℕ) : ℕ :=
  students_liking_both_baseball_football -
  students_liking_all_three +
  students_only_baseball +
  students_only_football +
  students_liking_basketball_and_football_only +
  students_liking_all_three +
  students_liking_none +
  (students_liking_basketball_as_well -
   (students_liking_all_three +
    students_liking_basketball_and_football_only))

theorem total_students_is_17 :
    total_students_in_class 7 3 4 2 1 2 5 = 17 :=
by sorry

end total_students_is_17_l43_43501


namespace range_f_l43_43090

noncomputable def f (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_f : (Set.range f) = (Set.univ \ { -27 }) :=
by
  sorry

end range_f_l43_43090


namespace quadrilaterals_property_A_false_l43_43179

theorem quadrilaterals_property_A_false (Q A : Type → Prop) 
  (h : ¬ ∃ x, Q x ∧ A x) : ¬ ∀ x, Q x → A x :=
by
  sorry

end quadrilaterals_property_A_false_l43_43179


namespace Brenda_weight_correct_l43_43784

-- Conditions
def MelWeight : ℕ := 70
def BrendaWeight : ℕ := 3 * MelWeight + 10

-- Proof problem
theorem Brenda_weight_correct : BrendaWeight = 220 := by
  sorry

end Brenda_weight_correct_l43_43784


namespace investment_duration_l43_43988

theorem investment_duration 
  (P : ℝ) (A : ℝ) (r : ℝ) (t : ℝ)
  (h1 : P = 939.60)
  (h2 : A = 1120)
  (h3 : r = 8) :
  t = 2.4 :=
by
  sorry

end investment_duration_l43_43988


namespace third_quadrant_angles_l43_43355

theorem third_quadrant_angles :
  {α : ℝ | ∃ k : ℤ, π + 2 * k * π < α ∧ α < 3 * π / 2 + 2 * k * π} =
  {α | π < α ∧ α < 3 * π / 2} :=
sorry

end third_quadrant_angles_l43_43355


namespace solve_x_l43_43639

theorem solve_x (x : ℝ) (h : (x + 1) ^ 2 = 9) : x = 2 ∨ x = -4 :=
sorry

end solve_x_l43_43639


namespace remainder_div_by_7_l43_43778

theorem remainder_div_by_7 (n : ℤ) (k m : ℤ) (r : ℤ) (h₀ : n = 7 * k + r) (h₁ : 3 * n = 7 * m + 3) (hrange : 0 ≤ r ∧ r < 7) : r = 1 :=
by
  sorry

end remainder_div_by_7_l43_43778


namespace solve_inequality_l43_43862

theorem solve_inequality (x : ℝ) : 
  let quad := (x - 2)^2 + 9
  let numerator := x - 3
  quad > 0 ∧ numerator ≥ 0 ↔ x ≥ 3 :=
by
    sorry

end solve_inequality_l43_43862


namespace gift_certificate_value_is_correct_l43_43507

-- Define the conditions
def total_race_time_minutes : ℕ := 12
def one_lap_meters : ℕ := 100
def total_laps : ℕ := 24
def earning_rate_per_minute : ℕ := 7

-- The total distance run in meters
def total_distance_meters : ℕ := total_laps * one_lap_meters

-- The total earnings in dollars
def total_earnings_dollars : ℕ := earning_rate_per_minute * total_race_time_minutes

-- The worth of the gift certificate per 100 meters (to be proven as 3.50 dollars)
def gift_certificate_value : ℚ := total_earnings_dollars / (total_distance_meters / one_lap_meters)

-- Prove that the gift certificate value is $3.50
theorem gift_certificate_value_is_correct : 
    gift_certificate_value = 3.5 := by
  sorry

end gift_certificate_value_is_correct_l43_43507


namespace anna_reading_time_l43_43946

theorem anna_reading_time 
  (C : ℕ)
  (T_per_chapter : ℕ)
  (hC : C = 31) 
  (hT : T_per_chapter = 20) :
  (C - (C / 3)) * T_per_chapter / 60 = 7 := 
by 
  -- proof steps will go here
  sorry

end anna_reading_time_l43_43946


namespace find_angle_C_l43_43162

theorem find_angle_C 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 10 * a * Real.cos B = 3 * b * Real.cos A) 
  (h2 : Real.cos A = (5 * Real.sqrt 26) / 26) 
  (h3 : A + B + C = π) : 
  C = (3 * π) / 4 :=
sorry

end find_angle_C_l43_43162


namespace remaining_shoes_to_sell_l43_43351

def shoes_goal : Nat := 80
def shoes_sold_last_week : Nat := 27
def shoes_sold_this_week : Nat := 12

theorem remaining_shoes_to_sell : shoes_goal - (shoes_sold_last_week + shoes_sold_this_week) = 41 :=
by
  sorry

end remaining_shoes_to_sell_l43_43351


namespace total_toys_l43_43494

theorem total_toys (A M T : ℕ) (h1 : A = 3 * M + M) (h2 : T = A + 2) (h3 : M = 6) : A + M + T = 56 :=
by
  sorry

end total_toys_l43_43494


namespace solve_inequality_l43_43878

theorem solve_inequality (x : ℝ) : 
  1 / (x^2 + 2) > 4 / x + 21 / 10 ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) := 
sorry

end solve_inequality_l43_43878


namespace sum_boundary_values_of_range_l43_43131

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + 3 * x^2 + 6 * x)

theorem sum_boundary_values_of_range : 
  let c := 0
  let d := 1
  c + d = 1 :=
by
  sorry

end sum_boundary_values_of_range_l43_43131


namespace integer_k_values_l43_43839

noncomputable def is_integer_solution (k x : ℤ) : Prop :=
  ((k - 2013) * x = 2015 - 2014 * x)

theorem integer_k_values (k : ℤ) (h : ∃ x : ℤ, is_integer_solution k x) :
  ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_k_values_l43_43839


namespace necessary_but_not_sufficient_l43_43042

theorem necessary_but_not_sufficient (x y : ℝ) :
  (x = 0) → (x^2 + y^2 = 0) ↔ (x = 0 ∧ y = 0) :=
by sorry

end necessary_but_not_sufficient_l43_43042


namespace smallest_x_value_l43_43387

theorem smallest_x_value : ∃ x : ℤ, ∃ y : ℤ, (xy + 7 * x + 6 * y = -8) ∧ x = -40 :=
by
  sorry

end smallest_x_value_l43_43387


namespace find_rate_of_interest_l43_43568

-- Definitions based on conditions
def Principal : ℝ := 7200
def SimpleInterest : ℝ := 3150
def Time : ℝ := 2.5
def RatePerAnnum (R : ℝ) : Prop := SimpleInterest = (Principal * R * Time) / 100

-- Theorem statement
theorem find_rate_of_interest (R : ℝ) (h : RatePerAnnum R) : R = 17.5 :=
by { sorry }

end find_rate_of_interest_l43_43568


namespace original_prices_correct_l43_43244

-- Define the problem conditions
def Shirt_A_discount1 := 0.10
def Shirt_A_discount2 := 0.20
def Shirt_A_final_price := 420

def Shirt_B_discount1 := 0.15
def Shirt_B_discount2 := 0.25
def Shirt_B_final_price := 405

def Shirt_C_discount1 := 0.05
def Shirt_C_discount2 := 0.15
def Shirt_C_final_price := 680

def sales_tax := 0.05

-- Define the original prices for each shirt.
def original_price_A := 420 / (0.9 * 0.8)
def original_price_B := 405 / (0.85 * 0.75)
def original_price_C := 680 / (0.95 * 0.85)

-- Prove the original prices of the shirts
theorem original_prices_correct:
  original_price_A = 583.33 ∧ 
  original_price_B = 635 ∧ 
  original_price_C = 842.24 := 
by
  sorry

end original_prices_correct_l43_43244


namespace white_line_longer_l43_43527

theorem white_line_longer :
  let white_line := 7.67
  let blue_line := 3.33
  white_line - blue_line = 4.34 := by
  sorry

end white_line_longer_l43_43527


namespace ratio_of_work_done_by_women_to_men_l43_43606

theorem ratio_of_work_done_by_women_to_men 
  (total_work_men : ℕ := 15 * 21 * 8)
  (total_work_women : ℕ := 21 * 36 * 5) :
  (total_work_women : ℚ) / (total_work_men : ℚ) = 2 / 3 :=
by
  -- Proof goes here
  sorry

end ratio_of_work_done_by_women_to_men_l43_43606


namespace least_number_to_subtract_l43_43823

theorem least_number_to_subtract (n : ℕ) (h : n = 652543) : 
  ∃ x : ℕ, x = 7 ∧ (n - x) % 12 = 0 :=
by
  sorry

end least_number_to_subtract_l43_43823


namespace price_increase_percentage_l43_43152

variables
  (coffees_daily_before : ℕ := 4)
  (price_per_coffee_before : ℝ := 2)
  (coffees_daily_after : ℕ := 2)
  (price_increase_savings : ℝ := 2)
  (spending_before := coffees_daily_before * price_per_coffee_before)
  (spending_after := spending_before - price_increase_savings)
  (price_per_coffee_after := spending_after / coffees_daily_after)

theorem price_increase_percentage :
  ((price_per_coffee_after - price_per_coffee_before) / price_per_coffee_before) * 100 = 50 :=
by
  sorry

end price_increase_percentage_l43_43152


namespace ned_washed_shirts_l43_43623

theorem ned_washed_shirts (short_sleeve long_sleeve not_washed: ℕ) (h1: short_sleeve = 9) (h2: long_sleeve = 21) (h3: not_washed = 1) : 
    (short_sleeve + long_sleeve - not_washed = 29) :=
by
  sorry

end ned_washed_shirts_l43_43623


namespace base_11_arithmetic_l43_43297

-- Define the base and the numbers in base 11
def base := 11

def a := 6 * base^2 + 7 * base + 4  -- 674 in base 11
def b := 2 * base^2 + 7 * base + 9  -- 279 in base 11
def c := 1 * base^2 + 4 * base + 3  -- 143 in base 11
def result := 5 * base^2 + 5 * base + 9  -- 559 in base 11

theorem base_11_arithmetic :
  (a - b + c) = result :=
sorry

end base_11_arithmetic_l43_43297


namespace value_of_k_l43_43523

theorem value_of_k (k : ℝ) : 
  (∃ x y : ℝ, x = 1/3 ∧ y = -8 ∧ -3/4 - 3 * k * x = 7 * y) → k = 55.25 :=
by
  intro h
  sorry

end value_of_k_l43_43523


namespace convert_246_octal_to_decimal_l43_43275

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l43_43275


namespace smallest_integer_satisfying_mod_conditions_l43_43575

theorem smallest_integer_satisfying_mod_conditions :
  ∃ n : ℕ, n > 0 ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 4) ∧ 
  (n % 7 = 6) ∧ 
  (n % 11 = 10) ∧ 
  n = 1154 := 
sorry

end smallest_integer_satisfying_mod_conditions_l43_43575


namespace friends_count_l43_43148

-- Define that Laura has 28 blocks
def blocks := 28

-- Define that each friend gets 7 blocks
def blocks_per_friend := 7

-- The proof statement we want to prove
theorem friends_count : blocks / blocks_per_friend = 4 := by
  sorry

end friends_count_l43_43148


namespace genevieve_initial_amount_l43_43425

def cost_per_kg : ℕ := 8
def kg_bought : ℕ := 250
def short_amount : ℕ := 400
def total_cost : ℕ := kg_bought * cost_per_kg
def initial_amount := total_cost - short_amount

theorem genevieve_initial_amount : initial_amount = 1600 := by
  unfold initial_amount total_cost cost_per_kg kg_bought short_amount
  sorry

end genevieve_initial_amount_l43_43425


namespace find_cosine_l43_43643
open Real

noncomputable def alpha (α : ℝ) : Prop := 0 < α ∧ α < π / 2 ∧ sin α = 3 / 5

theorem find_cosine (α : ℝ) (h : alpha α) :
  cos (π - α / 2) = - (3 * sqrt 10) / 10 :=
by sorry

end find_cosine_l43_43643


namespace mat_length_is_correct_l43_43361

noncomputable def mat_length (r : ℝ) (w : ℝ) : ℝ :=
  let θ := 2 * Real.pi / 5
  let side := 2 * r * Real.sin (θ / 2)
  let D := r * Real.cos (Real.pi / 5)
  let x := ((Real.sqrt (r^2 - ((w / 2) ^ 2))) - D + (w / 2))
  x

theorem mat_length_is_correct :
  mat_length 5 1 = 1.4 :=
by
  sorry

end mat_length_is_correct_l43_43361


namespace circle_area_l43_43901

theorem circle_area (x y : ℝ) :
  (3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
  (π * ((1 / 2) * (1 / 2)) = (π / 4)) := 
by
  intro h
  sorry

end circle_area_l43_43901


namespace value_of_one_house_l43_43938

theorem value_of_one_house
  (num_brothers : ℕ) (num_houses : ℕ) (payment_each : ℕ) 
  (total_money_paid : ℕ) (num_older : ℕ) (num_younger : ℕ)
  (share_per_younger : ℕ) (total_inheritance : ℕ) (value_of_house : ℕ) :
  num_brothers = 5 →
  num_houses = 3 →
  num_older = 3 →
  num_younger = 2 →
  payment_each = 800 →
  total_money_paid = num_older * payment_each →
  share_per_younger = total_money_paid / num_younger →
  total_inheritance = num_brothers * share_per_younger →
  value_of_house = total_inheritance / num_houses →
  value_of_house = 2000 :=
by {
  -- Provided conditions and statements without proofs
  sorry
}

end value_of_one_house_l43_43938


namespace joao_chocolates_l43_43175

theorem joao_chocolates (n : ℕ) (hn1 : 30 < n) (hn2 : n < 100) (h1 : n % 7 = 1) (h2 : n % 10 = 2) : n = 92 :=
sorry

end joao_chocolates_l43_43175


namespace glenda_speed_is_8_l43_43050

noncomputable def GlendaSpeed : ℝ :=
  let AnnSpeed := 6
  let Hours := 3
  let Distance := 42
  let AnnDistance := AnnSpeed * Hours
  let GlendaDistance := Distance - AnnDistance
  GlendaDistance / Hours

theorem glenda_speed_is_8 : GlendaSpeed = 8 := by
  sorry

end glenda_speed_is_8_l43_43050


namespace total_pikes_l43_43344

theorem total_pikes (x : ℝ) (h : x = 4 + (1/2) * x) : x = 8 :=
sorry

end total_pikes_l43_43344


namespace solution_cos_eq_l43_43103

open Real

theorem solution_cos_eq (x : ℝ) :
  (cos x)^2 + (cos (2 * x))^2 + (cos (3 * x))^2 = 1 ↔
  (∃ k : ℤ, x = k * π / 2 + π / 4) ∨ (∃ k : ℤ, x = k * π / 3 + π / 6) :=
by sorry

end solution_cos_eq_l43_43103


namespace find_b_and_area_l43_43653

open Real

variables (a c : ℝ) (A b S : ℝ)

theorem find_b_and_area 
  (h1 : a = sqrt 7) 
  (h2 : c = 3) 
  (h3 : A = π / 3) :
  (b = 1 ∨ b = 2) ∧ (S = 3 * sqrt 3 / 4 ∨ S = 3 * sqrt 3 / 2) := 
by sorry

end find_b_and_area_l43_43653


namespace trigonometric_identity_l43_43035

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end trigonometric_identity_l43_43035


namespace johns_total_spent_l43_43002

def total_spent (num_tshirts: Nat) (price_per_tshirt: Nat) (price_pants: Nat): Nat :=
  (num_tshirts * price_per_tshirt) + price_pants

theorem johns_total_spent : total_spent 3 20 50 = 110 := by
  sorry

end johns_total_spent_l43_43002


namespace max_volume_at_6_l43_43616

noncomputable def volume (x : ℝ) : ℝ :=
  x * (36 - 2 * x)^2

theorem max_volume_at_6 :
  ∃ x : ℝ, (0 < x) ∧ (x < 18) ∧ 
  (∀ y : ℝ, (0 < y) ∧ (y < 18) → volume y ≤ volume 6) :=
by
  sorry

end max_volume_at_6_l43_43616


namespace hourly_wage_calculation_l43_43560

variable (H : ℝ)
variable (hours_per_week : ℝ := 40)
variable (wage_per_widget : ℝ := 0.16)
variable (widgets_per_week : ℝ := 500)
variable (total_earnings : ℝ := 580)

theorem hourly_wage_calculation :
  (hours_per_week * H + widgets_per_week * wage_per_widget = total_earnings) →
  H = 12.5 :=
by
  intro h_equation
  -- Proof steps would go here
  sorry

end hourly_wage_calculation_l43_43560


namespace solve_equation_l43_43283

-- Define the given equation
def equation (x : ℝ) : Prop := (x^3 - 3 * x^2) / (x^2 - 4 * x + 4) + x = -3

-- State the theorem indicating the solutions to the equation
theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  equation x ↔ x = -2 ∨ x = 3 / 2 :=
sorry

end solve_equation_l43_43283


namespace total_weight_correct_l43_43428

variable (c1 c2 w2 c : Float)

def total_weight (c1 c2 w2 c : Float) (W x : Float) :=
  (c1 * x + c2 * w2) / (x + w2) = c ∧ W = x + w2

theorem total_weight_correct :
  total_weight 9 8 12 8.40 20 8 :=
by sorry

end total_weight_correct_l43_43428


namespace find_number_l43_43729

-- Definitions and conditions for the problem
def N_div_7 (N R_1 : ℕ) : ℕ := (N / 7) * 7 + R_1
def N_div_11 (N R_2 : ℕ) : ℕ := (N / 11) * 11 + R_2
def N_div_13 (N R_3 : ℕ) : ℕ := (N / 13) * 13 + R_3

theorem find_number 
  (N a b c R_1 R_2 R_3 : ℕ) 
  (hN7 : N = 7 * a + R_1)
  (hN11 : N = 11 * b + R_2)
  (hN13 : N = 13 * c + R_3)
  (hQ : a + b + c = 21)
  (hR : R_1 + R_2 + R_3 = 21)
  (hR1_lt : R_1 < 7)
  (hR2_lt : R_2 < 11)
  (hR3_lt : R_3 < 13) : 
  N = 74 :=
sorry

end find_number_l43_43729


namespace triangle_properties_l43_43578

open Real

noncomputable def is_isosceles_triangle (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧ (b = c)

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def area (a b c : ℝ) (A : ℝ) : ℝ :=
  1/2 * b * c * sin A

theorem triangle_properties 
  (A B C a b c : ℝ) 
  (h1 : sin B * sin C = 1/4) 
  (h2 : tan B * tan C = 1/3) 
  (h3 : a = 4 * sqrt 3) 
  (h4 : A + B + C = π) 
  (isosceles : is_isosceles_triangle A B C a b c) :
  is_isosceles_triangle A B C a b c ∧ 
  perimeter a b c = 8 + 4 * sqrt 3 ∧ 
  area a b c A = 4 * sqrt 3 :=
sorry

end triangle_properties_l43_43578


namespace triangular_number_difference_l43_43045

-- Definition of the nth triangular number
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Theorem stating the problem
theorem triangular_number_difference :
  triangular_number 2010 - triangular_number 2008 = 4019 :=
by
  sorry

end triangular_number_difference_l43_43045


namespace range_of_a_l43_43947

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a ∈ Set.Iic 0 ∪ Set.Ici 3) := 
sorry

end range_of_a_l43_43947


namespace find_g3_value_l43_43720

def g (n : ℕ) : ℕ :=
  if n < 5 then 2 * n ^ 2 + 3 else 4 * n + 1

theorem find_g3_value : g (g (g 3)) = 341 := by
  sorry

end find_g3_value_l43_43720


namespace find_last_four_digits_of_N_l43_43794

def P (n : Nat) : Nat :=
  match n with
  | 0     => 1 -- usually not needed but for completeness
  | 1     => 2
  | _     => 2 + (n - 1) * n

theorem find_last_four_digits_of_N : (P 2011) % 10000 = 2112 := by
  -- we define P(2011) as per the general formula derived and then verify the modulo operation
  sorry

end find_last_four_digits_of_N_l43_43794


namespace power_identity_l43_43309

theorem power_identity {a n m k : ℝ} (h1: a^n = 2) (h2: a^m = 3) (h3: a^k = 4) :
  a^(2 * n + m - 2 * k) = 3 / 4 :=
by
  sorry

end power_identity_l43_43309


namespace impossible_all_matches_outside_own_country_l43_43080

theorem impossible_all_matches_outside_own_country (n : ℕ) (h_teams : n = 16) : 
  ¬ ∀ (T : Fin n → Fin n → Prop), (∀ i j, i ≠ j → T i j) ∧ 
  (∀ i, ∀ j, i ≠ j → T i j → T j i) ∧ 
  (∀ i, T i i = false) → 
  ∀ i, ∃ j, T i j ∧ i ≠ j :=
by
  intro H
  sorry

end impossible_all_matches_outside_own_country_l43_43080


namespace multiplication_of_monomials_l43_43319

-- Define the constants and assumptions
def a : ℝ := -2
def b : ℝ := 4
def e1 : ℤ := 4
def e2 : ℤ := 5
def result : ℝ := -8
def result_exp : ℤ := 9

-- State the theorem to be proven
theorem multiplication_of_monomials :
  (a * 10^e1) * (b * 10^e2) = result * 10^result_exp := 
by
  sorry

end multiplication_of_monomials_l43_43319


namespace people_behind_yuna_l43_43684

theorem people_behind_yuna (total_people : ℕ) (people_in_front : ℕ) (yuna : ℕ)
  (h1 : total_people = 7) (h2 : people_in_front = 2) (h3 : yuna = 1) :
  total_people - people_in_front - yuna = 4 :=
by
  sorry

end people_behind_yuna_l43_43684


namespace part_a_part_b_part_c_part_d_l43_43248

-- Part a
theorem part_a (x : ℝ) : 
  (5 / x - x / 3 = 1 / 6) ↔ x = 6 := 
by
  sorry

-- Part b
theorem part_b (a : ℝ) : 
  ¬ ∃ a, (1 / 2 + a / 4 = a / 4) := 
by
  sorry

-- Part c
theorem part_c (y : ℝ) : 
  (9 / y - y / 21 = 17 / 21) ↔ y = 7 := 
by
  sorry

-- Part d
theorem part_d (z : ℝ) : 
  (z / 8 - 1 / z = 3 / 8) ↔ z = 4 := 
by
  sorry

end part_a_part_b_part_c_part_d_l43_43248


namespace number_of_integers_satisfying_condition_l43_43951

def satisfies_condition (n : ℤ) : Prop :=
  1 + Int.floor (101 * n / 102) = Int.ceil (98 * n / 99)

noncomputable def number_of_solutions : ℤ :=
  10198

theorem number_of_integers_satisfying_condition :
  (∃ n : ℤ, satisfies_condition n) ↔ number_of_solutions = 10198 :=
sorry

end number_of_integers_satisfying_condition_l43_43951


namespace alice_has_ball_after_two_turns_l43_43102

noncomputable def probability_alice_has_ball_twice_turns : ℚ :=
  let P_AB_A : ℚ := 1/2 * 1/3
  let P_ABC_A : ℚ := 1/2 * 1/3 * 1/2
  let P_AA : ℚ := 1/2 * 1/2
  P_AB_A + P_ABC_A + P_AA

theorem alice_has_ball_after_two_turns :
  probability_alice_has_ball_twice_turns = 1/2 := 
by
  sorry

end alice_has_ball_after_two_turns_l43_43102


namespace find_a_l43_43233

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem find_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3 / 4 :=
sorry

end find_a_l43_43233


namespace find_original_number_l43_43130

/-- Given that one less than the reciprocal of a number is 5/2, the original number must be -2/3. -/
theorem find_original_number (y : ℚ) (h : 1 - 1 / y = 5 / 2) : y = -2 / 3 :=
sorry

end find_original_number_l43_43130


namespace sequence_value_2016_l43_43142

theorem sequence_value_2016 (a : ℕ → ℕ) (h₁ : a 1 = 0) (h₂ : ∀ n, a (n + 1) = a n + 2 * n) : a 2016 = 2016 * 2015 :=
by 
  sorry

end sequence_value_2016_l43_43142


namespace ratio_of_awards_l43_43129

theorem ratio_of_awards 
  (Scott_awards : ℕ) (Scott_awards_eq : Scott_awards = 4)
  (Jessie_awards : ℕ) (Jessie_awards_eq : Jessie_awards = 3 * Scott_awards)
  (rival_awards : ℕ) (rival_awards_eq : rival_awards = 24) :
  rival_awards / Jessie_awards = 2 :=
by sorry

end ratio_of_awards_l43_43129


namespace product_of_distinct_nonzero_real_satisfying_eq_l43_43708

theorem product_of_distinct_nonzero_real_satisfying_eq (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
    (h : x + 3/x = y + 3/y) : x * y = 3 :=
by sorry

end product_of_distinct_nonzero_real_satisfying_eq_l43_43708


namespace jori_water_left_l43_43809

theorem jori_water_left (initial used : ℚ) (h1 : initial = 3) (h2 : used = 4 / 3) :
  initial - used = 5 / 3 :=
by
  sorry

end jori_water_left_l43_43809


namespace solve_problem_l43_43273

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) :
  f (x^3 + y^3) = (x + y) * ((f x)^2 - (f x) * (f y) + (f (f y))^2)

theorem solve_problem (x : ℝ) : f (1996 * x) = 1996 * f x :=
sorry

end solve_problem_l43_43273


namespace mary_initial_baseball_cards_l43_43252

theorem mary_initial_baseball_cards (X : ℕ) :
  (X - 8 + 26 + 40 = 84) → (X = 26) :=
by
  sorry

end mary_initial_baseball_cards_l43_43252


namespace average_marks_110_l43_43022

def marks_problem (P C M B E : ℕ) : Prop :=
  (C = P + 90) ∧
  (M = P + 140) ∧
  (P + C + M + B + E = P + 350) ∧
  (B = E) ∧
  (P ≥ 40) ∧
  (C ≥ 40) ∧
  (M ≥ 40) ∧
  (B ≥ 40) ∧
  (E ≥ 40)

theorem average_marks_110 (P C M B E : ℕ) (h : marks_problem P C M B E) : 
    (B + C + M) / 3 = 110 := 
by
  sorry

end average_marks_110_l43_43022


namespace complex_arithmetic_l43_43135

def Q : ℂ := 7 + 3 * Complex.I
def E : ℂ := 2 * Complex.I
def D : ℂ := 7 - 3 * Complex.I
def F : ℂ := 1 + Complex.I

theorem complex_arithmetic : (Q * E * D) + F = 1 + 117 * Complex.I := by
  sorry

end complex_arithmetic_l43_43135


namespace steven_more_peaches_than_apples_l43_43674

def steven_peaches : Nat := 17
def steven_apples : Nat := 16

theorem steven_more_peaches_than_apples : steven_peaches - steven_apples = 1 := by
  sorry

end steven_more_peaches_than_apples_l43_43674


namespace sum_of_remainders_mod_8_l43_43746

theorem sum_of_remainders_mod_8 
  (x y z w : ℕ)
  (hx : x % 8 = 3)
  (hy : y % 8 = 5)
  (hz : z % 8 = 7)
  (hw : w % 8 = 1) :
  (x + y + z + w) % 8 = 0 :=
by
  sorry

end sum_of_remainders_mod_8_l43_43746


namespace sum_of_other_endpoint_coordinates_l43_43386

theorem sum_of_other_endpoint_coordinates
  (x₁ y₁ x₂ y₂ : ℝ)
  (hx : (x₁ + x₂) / 2 = 5)
  (hy : (y₁ + y₂) / 2 = -8)
  (endpt1 : x₁ = 7)
  (endpt2 : y₁ = -2) :
  x₂ + y₂ = -11 :=
sorry

end sum_of_other_endpoint_coordinates_l43_43386


namespace range_of_a_l43_43906

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x : ℝ, 4 ≤ x → (if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a) ≥ 4) :
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l43_43906


namespace train_speed_excluding_stoppages_l43_43178

-- Define the speed of the train excluding stoppages and including stoppages
variables (S : ℕ) -- S is the speed of the train excluding stoppages
variables (including_stoppages_speed : ℕ := 40) -- The speed including stoppages is 40 kmph

-- The train stops for 20 minutes per hour. This means it runs for (60 - 20) minutes per hour.
def running_time_per_hour := 40

-- Converting 40 minutes to hours
def running_fraction_of_hour : ℚ := 40 / 60

-- Formulate the main theorem:
theorem train_speed_excluding_stoppages
    (H1 : including_stoppages_speed = 40)
    (H2 : running_fraction_of_hour = 2 / 3) :
    S = 60 :=
by
    sorry

end train_speed_excluding_stoppages_l43_43178


namespace product_implication_l43_43378

theorem product_implication (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab : a * b > 1) : a > 1 ∨ b > 1 :=
sorry

end product_implication_l43_43378


namespace abs_difference_lt_2t_l43_43034

/-- Given conditions of absolute values with respect to t -/
theorem abs_difference_lt_2t (x y s t : ℝ) (h₁ : |x - s| < t) (h₂ : |y - s| < t) :
  |x - y| < 2 * t :=
sorry

end abs_difference_lt_2t_l43_43034


namespace right_triangle_ratio_l43_43379

theorem right_triangle_ratio (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : a^2 + b^2 = c^2) (r s : ℝ) (h3 : r = a^2 / c) (h4 : s = b^2 / c) : 
  r / s = 9 / 16 := by
 sorry

end right_triangle_ratio_l43_43379


namespace non_defective_probability_l43_43622

theorem non_defective_probability :
  let p_B := 0.03
  let p_C := 0.01
  let p_def := p_B + p_C
  let p_non_def := 1 - p_def
  p_non_def = 0.96 :=
by
  let p_B := 0.03
  let p_C := 0.01
  let p_def := p_B + p_C
  let p_non_def := 1 - p_def
  sorry

end non_defective_probability_l43_43622


namespace point_in_second_quadrant_l43_43825

-- Definitions for the coordinates of the points
def A : ℤ × ℤ := (3, 2)
def B : ℤ × ℤ := (-3, -2)
def C : ℤ × ℤ := (3, -2)
def D : ℤ × ℤ := (-3, 2)

-- Definition for the second quadrant condition
def isSecondQuadrant (p : ℤ × ℤ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- The theorem we need to prove
theorem point_in_second_quadrant : isSecondQuadrant D :=
by
  sorry

end point_in_second_quadrant_l43_43825


namespace find_possible_values_l43_43146

theorem find_possible_values (a b c k : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) :
  (k * a^2 * b^2 + k * a^2 * c^2 + k * b^2 * c^2) / 
  ((a^2 - b * c) * (b^2 - a * c) + 
   (a^2 - b * c) * (c^2 - a * b) + 
   (b^2 - a * c) * (c^2 - a * b)) 
  = k / 3 :=
by 
  sorry

end find_possible_values_l43_43146


namespace evaluation_of_expression_l43_43828

theorem evaluation_of_expression
  (a b x y m : ℤ)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = -1) :
  2023 * (a + b) + 3 * (|m|) - 2 * (x * y) = 1 :=
by
  -- skipping the proof
  sorry

end evaluation_of_expression_l43_43828


namespace reporters_cover_local_politics_l43_43714

structure Reporters :=
(total : ℕ)
(politics : ℕ)
(local_politics : ℕ)

def percentages (reporters : Reporters) : Prop :=
  reporters.politics = (40 * reporters.total) / 100 ∧
  reporters.local_politics = (75 * reporters.politics) / 100

theorem reporters_cover_local_politics (reporters : Reporters) (h : percentages reporters) :
  (reporters.local_politics * 100) / reporters.total = 30 :=
by
  -- Proof steps would be added here
  sorry

end reporters_cover_local_politics_l43_43714


namespace pencils_in_total_l43_43585

theorem pencils_in_total
  (rows : ℕ) (pencils_per_row : ℕ) (total_pencils : ℕ)
  (h1 : rows = 14)
  (h2 : pencils_per_row = 11)
  (h3 : total_pencils = rows * pencils_per_row) :
  total_pencils = 154 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end pencils_in_total_l43_43585


namespace inequality_solution_l43_43789

theorem inequality_solution (x : ℝ) : (x^3 - 12*x^2 + 36*x > 0) ↔ (0 < x ∧ x < 6) ∨ (x > 6) := by
  sorry

end inequality_solution_l43_43789


namespace charles_picked_50_pears_l43_43421

variable (P B S : ℕ)

theorem charles_picked_50_pears 
  (cond1 : S = B + 10)
  (cond2 : B = 3 * P)
  (cond3 : S = 160) : 
  P = 50 := by
  sorry

end charles_picked_50_pears_l43_43421


namespace total_cost_of_antibiotics_l43_43280

-- Definitions based on the conditions
def cost_A_per_dose : ℝ := 3
def cost_B_per_dose : ℝ := 4.50
def doses_per_day_A : ℕ := 2
def days_A : ℕ := 3
def doses_per_day_B : ℕ := 1
def days_B : ℕ := 4

-- Total cost calculations
def total_cost_A : ℝ := days_A * doses_per_day_A * cost_A_per_dose
def total_cost_B : ℝ := days_B * doses_per_day_B * cost_B_per_dose

-- Final proof statement
theorem total_cost_of_antibiotics : total_cost_A + total_cost_B = 36 :=
by
  -- The proof goes here
  sorry

end total_cost_of_antibiotics_l43_43280


namespace jelly_bean_probability_l43_43539

variable (P_red P_orange P_yellow P_green : ℝ)

theorem jelly_bean_probability :
  P_red = 0.15 ∧ P_orange = 0.35 ∧ (P_red + P_orange + P_yellow + P_green = 1) →
  (P_yellow + P_green = 0.5) :=
by
  intro h
  obtain ⟨h_red, h_orange, h_total⟩ := h
  sorry

end jelly_bean_probability_l43_43539


namespace least_multiple_x_correct_l43_43804

noncomputable def least_multiple_x : ℕ :=
  let x := 20
  let y := 8
  let z := 5
  5 * y

theorem least_multiple_x_correct (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 33) (h5 : 5 * y = 8 * z) : least_multiple_x = 40 :=
by
  sorry

end least_multiple_x_correct_l43_43804


namespace find_functions_l43_43368

theorem find_functions (M N : ℝ × ℝ)
  (hM : M.fst = -4) (hM_quad2 : 0 < M.snd)
  (hN : N = (-6, 0))
  (h_area : 1 / 2 * 6 * M.snd = 15) :
  (∃ k, ∀ x, (M = (-4, 5) → N = (-6, 0) → x * k = -5 / 4 * x)) ∧ 
  (∃ a b, ∀ x, (M = (-4, 5) → N = (-6, 0) → x * a + b = 5 / 2 * x + 15)) := 
sorry

end find_functions_l43_43368


namespace faye_rows_l43_43125

theorem faye_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h_total_pencils : total_pencils = 720)
  (h_pencils_per_row : pencils_per_row = 24) : 
  total_pencils / pencils_per_row = 30 := by 
  sorry

end faye_rows_l43_43125


namespace max_average_growth_rate_l43_43480

theorem max_average_growth_rate 
  (P1 P2 : ℝ) (M : ℝ)
  (h1 : P1 + P2 = M) : 
  (1 + (M / 2))^2 ≥ (1 + P1) * (1 + P2) := 
by
  -- AM-GM Inequality application and other mathematical steps go here.
  sorry

end max_average_growth_rate_l43_43480


namespace range_of_t_sum_of_squares_l43_43614

-- Define the conditions and the problem statement in Lean

variables (a b c t x : ℝ)
variables (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variables (ineq1 : |x + 1| - |x - 2| ≥ |t - 1| + t)
variables (sum_pos : 2 * a + b + c = 2)

theorem range_of_t :
  (∃ x, |x + 1| - |x - 2| ≥ |t - 1| + t) → t ≤ 2 :=
sorry

theorem sum_of_squares :
  2 * a + b + c = 2 → 0 < a → 0 < b → 0 < c → a^2 + b^2 + c^2 ≥ 2 / 3 :=
sorry

end range_of_t_sum_of_squares_l43_43614


namespace seq_satisfies_recurrence_sq_seq_satisfies_recurrence_cube_l43_43338

-- Define the sequences
def a_sq (n : ℕ) : ℕ := n ^ 2
def a_cube (n : ℕ) : ℕ := n ^ 3

-- First proof problem statement
theorem seq_satisfies_recurrence_sq :
  (a_sq 0 = 0) ∧ (a_sq 1 = 1) ∧ (a_sq 2 = 4) ∧ (a_sq 3 = 9) ∧ (a_sq 4 = 16) →
  (∀ n : ℕ, n ≥ 3 → a_sq n = 3 * a_sq (n - 1) - 3 * a_sq (n - 2) + a_sq (n - 3)) :=
by
  sorry

-- Second proof problem statement
theorem seq_satisfies_recurrence_cube :
  (a_cube 0 = 0) ∧ (a_cube 1 = 1) ∧ (a_cube 2 = 8) ∧ (a_cube 3 = 27) ∧ (a_cube 4 = 64) →
  (∀ n : ℕ, n ≥ 4 → a_cube n = 4 * a_cube (n - 1) - 6 * a_cube (n - 2) + 4 * a_cube (n - 3) - a_cube (n - 4)) :=
by
  sorry

end seq_satisfies_recurrence_sq_seq_satisfies_recurrence_cube_l43_43338


namespace hyperbola_range_of_k_l43_43295

theorem hyperbola_range_of_k (k : ℝ) :
  (∃ x y : ℝ, (x^2)/(k + 4) + (y^2)/(k - 1) = 1) → -4 < k ∧ k < 1 :=
by 
  sorry

end hyperbola_range_of_k_l43_43295


namespace probability_of_multiple_of_42_is_zero_l43_43890

-- Given conditions
def factors_200 : Set ℕ := {1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200}
def multiple_of_42 (n : ℕ) : Prop := n % 42 = 0

-- Problem statement: the probability of selecting a multiple of 42 from the factors of 200 is 0.
theorem probability_of_multiple_of_42_is_zero : 
  ∀ (n : ℕ), n ∈ factors_200 → ¬ multiple_of_42 n := 
by
  sorry

end probability_of_multiple_of_42_is_zero_l43_43890


namespace extreme_value_and_tangent_line_l43_43655

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2 - 3 * x

theorem extreme_value_and_tangent_line (a b : ℝ) (h1 : f a b 1 = 0) (h2 : f a b (-1) = 0) :
  (f 1 0 (-1) = 2) ∧ (f 1 0 1 = -2) ∧ (∀ x : ℝ, x = -2 → (9 * x - (x^3 - 3 * x) + 16 = 0)) :=
by
  sorry

end extreme_value_and_tangent_line_l43_43655


namespace line_ellipse_common_points_l43_43081

def point (P : Type*) := P → ℝ × ℝ

theorem line_ellipse_common_points
  (m n : ℝ)
  (no_common_points_with_circle : ∀ (x y : ℝ), mx + ny - 3 = 0 → x^2 + y^2 ≠ 3) :
  ∀ (Px Py : ℝ), (Px = m ∧ Py = n) →
  (∃ (x1 y1 x2 y2 : ℝ), ((x1^2 / 7) + (y1^2 / 3) = 1 ∧ (x2^2 / 7) + (y2^2 / 3) = 1) ∧ (x1, y1) ≠ (x2, y2)) :=
by
  sorry

end line_ellipse_common_points_l43_43081


namespace johns_number_is_thirteen_l43_43913

theorem johns_number_is_thirteen (x : ℕ) (h1 : 10 ≤ x) (h2 : x < 100) (h3 : ∃ a b : ℕ, 10 * a + b = 4 * x + 17 ∧ 92 ≤ 10 * b + a ∧ 10 * b + a ≤ 96) : x = 13 :=
sorry

end johns_number_is_thirteen_l43_43913


namespace MishaTotalMoney_l43_43354

-- Define Misha's initial amount of money
def initialMoney : ℕ := 34

-- Define the amount of money Misha earns
def earnedMoney : ℕ := 13

-- Define the total amount of money Misha will have
def totalMoney : ℕ := initialMoney + earnedMoney

-- Statement to prove
theorem MishaTotalMoney : totalMoney = 47 := by
  sorry

end MishaTotalMoney_l43_43354


namespace heaviest_and_lightest_in_13_weighings_l43_43566

/-- Given ten coins of different weights and a balance scale.
    Prove that it is possible to identify the heaviest and the lightest coin
    within 13 weighings. -/
theorem heaviest_and_lightest_in_13_weighings
  (coins : Fin 10 → ℝ)
  (h_different: ∀ i j : Fin 10, i ≠ j → coins i ≠ coins j)
  : ∃ (heaviest lightest : Fin 10),
      (heaviest ≠ lightest) ∧
      (∀ i : Fin 10, coins i ≤ coins heaviest) ∧
      (∀ i : Fin 10, coins lightest ≤ coins i) :=
sorry

end heaviest_and_lightest_in_13_weighings_l43_43566


namespace am_gm_iq_l43_43801

theorem am_gm_iq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (a + 1/a) * (b + 1/b) ≥ 25/4 := sorry

end am_gm_iq_l43_43801


namespace num_congruent_mod_7_count_mod_7_eq_22_l43_43944

theorem num_congruent_mod_7 (n : ℕ) :
  (1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 1) → ∃ k, 0 ≤ k ∧ k ≤ 21 ∧ n = 7 * k + 1 :=
sorry

theorem count_mod_7_eq_22 : 
  (∃ n_set : Finset ℕ, 
    (∀ n ∈ n_set, 1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 1) ∧ 
    Finset.card n_set = 22) :=
sorry

end num_congruent_mod_7_count_mod_7_eq_22_l43_43944


namespace m_ge_1_l43_43463

open Set

theorem m_ge_1 (m : ℝ) :
  (∀ x, x ∈ {x | x ≤ 1} ∩ {x | ¬ (x ≤ m)} → False) → m ≥ 1 :=
by
  intro h
  sorry

end m_ge_1_l43_43463


namespace chords_intersecting_theorem_l43_43958

noncomputable def intersecting_chords_theorem (P A B C D : ℝ) (h_circle : P ≠ A) (h_ab : A ≠ B) (h_cd : C ≠ D) : ℝ :=
  sorry

theorem chords_intersecting_theorem (P A B C D : ℝ) (h_circle : P ≠ A) (h_ab : A ≠ B) (h_cd : C ≠ D) :
  (P - A) * (P - B) = (P - C) * (P - D) :=
by sorry

end chords_intersecting_theorem_l43_43958


namespace remainder_9_pow_2023_div_50_l43_43483

theorem remainder_9_pow_2023_div_50 : (9 ^ 2023) % 50 = 41 := by
  sorry

end remainder_9_pow_2023_div_50_l43_43483


namespace fraction_tips_l43_43758

theorem fraction_tips {S : ℝ} (H1 : S > 0) (H2 : tips = (7 / 3 : ℝ) * S) (H3 : bonuses = (2 / 5 : ℝ) * S) :
  (tips / (S + tips + bonuses)) = (5 / 8 : ℝ) :=
by
  sorry

end fraction_tips_l43_43758


namespace more_supermarkets_in_us_l43_43115

-- Definitions based on conditions
def total_supermarkets : ℕ := 84
def us_supermarkets : ℕ := 47
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

-- Prove that the number of more FGH supermarkets in the US than in Canada is 10
theorem more_supermarkets_in_us : us_supermarkets - canada_supermarkets = 10 :=
by
  -- adding 'sorry' as the proof
  sorry

end more_supermarkets_in_us_l43_43115


namespace ursula_initial_money_l43_43119

def cost_per_hot_dog : ℝ := 1.50
def number_of_hot_dogs : ℕ := 5
def cost_per_salad : ℝ := 2.50
def number_of_salads : ℕ := 3
def change_received : ℝ := 5.00

def total_cost_of_hot_dogs : ℝ := number_of_hot_dogs * cost_per_hot_dog
def total_cost_of_salads : ℝ := number_of_salads * cost_per_salad
def total_cost : ℝ := total_cost_of_hot_dogs + total_cost_of_salads
def amount_paid : ℝ := total_cost + change_received

theorem ursula_initial_money : amount_paid = 20.00 := by
  /- Proof here, which is not required for the task -/
  sorry

end ursula_initial_money_l43_43119


namespace remainder_is_162_l43_43010

def polynomial (x : ℝ) : ℝ := 2 * x^4 - x^3 + 4 * x^2 - 5 * x + 6

theorem remainder_is_162 : polynomial 3 = 162 :=
by 
  sorry

end remainder_is_162_l43_43010


namespace Monica_books_read_l43_43499

theorem Monica_books_read : 
  let books_last_year := 16 
  let books_this_year := 2 * books_last_year
  let books_next_year := 2 * books_this_year + 5
  books_next_year = 69 :=
by
  let books_last_year := 16
  let books_this_year := 2 * books_last_year
  let books_next_year := 2 * books_this_year + 5
  sorry

end Monica_books_read_l43_43499


namespace expected_score_is_6_l43_43608

-- Define the probabilities of making a shot
def p : ℝ := 0.5

-- Define the scores for each scenario
def score_first_shot : ℝ := 8
def score_second_shot : ℝ := 6
def score_third_shot : ℝ := 4
def score_no_shot : ℝ := 0

-- Compute the expected value
def expected_score : ℝ :=
  p * score_first_shot +
  (1 - p) * p * score_second_shot +
  (1 - p) * (1 - p) * p * score_third_shot +
  (1 - p) * (1 - p) * (1 - p) * score_no_shot

theorem expected_score_is_6 : expected_score = 6 := by
  sorry

end expected_score_is_6_l43_43608


namespace baker_price_l43_43071

theorem baker_price
  (P : ℝ)
  (h1 : 8 * P = 320)
  (h2 : 10 * (0.80 * P) = 320)
  : P = 40 := sorry

end baker_price_l43_43071


namespace quadratic_solutions_l43_43380

theorem quadratic_solutions (x : ℝ) :
  (4 * x^2 - 6 * x = 0) ↔ (x = 0) ∨ (x = 3 / 2) :=
sorry

end quadratic_solutions_l43_43380


namespace b_in_terms_of_a_l43_43263

noncomputable def a (k : ℝ) : ℝ := 3 + 3^k
noncomputable def b (k : ℝ) : ℝ := 3 + 3^(-k)

theorem b_in_terms_of_a (k : ℝ) :
  b k = (3 * (a k) - 8) / ((a k) - 3) := 
sorry

end b_in_terms_of_a_l43_43263


namespace valid_votes_per_candidate_l43_43226

theorem valid_votes_per_candidate (total_votes : ℕ) (invalid_percentage valid_percentage_A valid_percentage_B : ℚ) 
                                  (A_votes B_votes C_votes valid_votes : ℕ) :
  total_votes = 1250000 →
  invalid_percentage = 20 →
  valid_percentage_A = 45 →
  valid_percentage_B = 35 →
  valid_votes = total_votes * (1 - invalid_percentage / 100) →
  A_votes = valid_votes * (valid_percentage_A / 100) →
  B_votes = valid_votes * (valid_percentage_B / 100) →
  C_votes = valid_votes - A_votes - B_votes →
  valid_votes = 1000000 ∧ A_votes = 450000 ∧ B_votes = 350000 ∧ C_votes = 200000 :=
by {
  sorry
}

end valid_votes_per_candidate_l43_43226


namespace derivative_f_l43_43871

noncomputable def f (x : ℝ) : ℝ := 1 + Real.cos x

theorem derivative_f (x : ℝ) : deriv f x = -Real.sin x := 
by 
  sorry

end derivative_f_l43_43871


namespace probability_accurate_forecast_l43_43266

theorem probability_accurate_forecast (p q : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : 0 ≤ q ∧ q ≤ 1) : 
  p * (1 - q) = p * (1 - q) :=
by {
  sorry
}

end probability_accurate_forecast_l43_43266


namespace original_number_is_80_l43_43805

-- Define the existence of the numbers A and B
variable (A B : ℕ)

-- Define the conditions from the problem
def conditions :=
  A = 35 ∧ A / 7 = B / 9

-- Define the statement to prove
theorem original_number_is_80 (h : conditions A B) : A + B = 80 :=
by
  -- Proof is omitted
  sorry

end original_number_is_80_l43_43805


namespace intersection_range_l43_43709

noncomputable def function_f (x: ℝ) : ℝ := abs (x^2 - 4 * x + 3)

theorem intersection_range (b : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ function_f x1 = b ∧ function_f x2 = b ∧ function_f x3 = b) ↔ (0 < b ∧ b ≤ 1) := 
sorry

end intersection_range_l43_43709


namespace defective_probability_l43_43414

theorem defective_probability {total_switches checked_switches defective_checked : ℕ}
  (h1 : total_switches = 2000)
  (h2 : checked_switches = 100)
  (h3 : defective_checked = 10) :
  (defective_checked : ℚ) / checked_switches = 1 / 10 :=
sorry

end defective_probability_l43_43414


namespace min_value_d_l43_43382

theorem min_value_d (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (unique_solution : ∃! x y : ℤ, 2 * x + y = 2007 ∧ y = (abs (x - a) + abs (x - b) + abs (x - c) + abs (x - d))) :
  d = 504 :=
sorry

end min_value_d_l43_43382


namespace percentage_of_mortality_l43_43645

theorem percentage_of_mortality
  (P : ℝ) -- The population size could be represented as a real number
  (affected_fraction : ℝ) (dead_fraction : ℝ)
  (h1 : affected_fraction = 0.15) -- 15% of the population is affected
  (h2 : dead_fraction = 0.08) -- 8% of the affected population died
: (affected_fraction * dead_fraction) * 100 = 1.2 :=
by
  sorry

end percentage_of_mortality_l43_43645


namespace cosine_inequality_l43_43036

theorem cosine_inequality (a b c : ℝ) : ∃ x : ℝ, 
    a * Real.cos x + b * Real.cos (3 * x) + c * Real.cos (9 * x) ≥ (|a| + |b| + |c|) / 2 :=
sorry

end cosine_inequality_l43_43036


namespace arithmetic_sequence_30th_term_value_l43_43917

def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

-- Given conditions
def a1 : ℤ := 3
def a2 : ℤ := 15
def a3 : ℤ := 27

-- Calculate the common difference d
def d : ℤ := a2 - a1

-- Define the 30th term
def a30 := arithmetic_sequence a1 d 30

theorem arithmetic_sequence_30th_term_value :
  a30 = 351 := by
  sorry

end arithmetic_sequence_30th_term_value_l43_43917


namespace num_dogs_l43_43097

-- Define the conditions
def total_animals := 11
def ducks := 6
def total_legs := 32
def legs_per_duck := 2
def legs_per_dog := 4

-- Calculate intermediate values based on conditions
def duck_legs := ducks * legs_per_duck
def remaining_legs := total_legs - duck_legs

-- The proof statement
theorem num_dogs : ∃ D : ℕ, D = remaining_legs / legs_per_dog ∧ D + ducks = total_animals :=
by
  sorry

end num_dogs_l43_43097


namespace system_of_equations_solution_l43_43717

theorem system_of_equations_solution 
  (x y z : ℤ) 
  (h1 : x^2 - y - z = 8) 
  (h2 : 4 * x + y^2 + 3 * z = -11) 
  (h3 : 2 * x - 3 * y + z^2 = -11) : 
  x = -3 ∧ y = 2 ∧ z = -1 :=
sorry

end system_of_equations_solution_l43_43717


namespace log_579_between_consec_ints_l43_43077

theorem log_579_between_consec_ints (a b : ℤ) (h₁ : 2 < Real.log 579 / Real.log 10) (h₂ : Real.log 579 / Real.log 10 < 3) : a + b = 5 :=
sorry

end log_579_between_consec_ints_l43_43077


namespace price_reduction_after_markup_l43_43054

theorem price_reduction_after_markup (p : ℝ) (x : ℝ) (h₁ : 0 < p) (h₂ : 0 ≤ x ∧ x < 1) :
  (1.25 : ℝ) * (1 - x) = 1 → x = 0.20 := by
  sorry

end price_reduction_after_markup_l43_43054


namespace comparison_of_prices_l43_43032

theorem comparison_of_prices:
  ∀ (x y : ℝ), (6 * x + 3 * y > 24) → (4 * x + 5 * y < 22) → (2 * x > 3 * y) :=
by
  intros x y h1 h2
  sorry

end comparison_of_prices_l43_43032


namespace boxes_of_orange_crayons_l43_43157

theorem boxes_of_orange_crayons
  (n_orange_boxes : ℕ)
  (orange_crayons_per_box : ℕ := 8)
  (blue_boxes : ℕ := 7) (blue_crayons_per_box : ℕ := 5)
  (red_boxes : ℕ := 1) (red_crayons_per_box : ℕ := 11)
  (total_crayons : ℕ := 94)
  (h_total_crayons : (n_orange_boxes * orange_crayons_per_box) + (blue_boxes * blue_crayons_per_box) + (red_boxes * red_crayons_per_box) = total_crayons):
  n_orange_boxes = 6 := 
by sorry

end boxes_of_orange_crayons_l43_43157


namespace part1_part2_part3_l43_43084

-- Problem Definitions
def air_conditioner_cost (A B : ℕ → ℕ) :=
  A 3 + B 2 = 39000 ∧ 4 * A 1 - 5 * B 1 = 6000

def possible_schemes (A B : ℕ → ℕ) :=
  ∀ a b, a ≥ b / 2 ∧ 9000 * a + 6000 * b ≤ 217000 ∧ a + b = 30

def minimize_cost (A B : ℕ → ℕ) :=
  ∃ a, (a = 10 ∧ 9000 * a + 6000 * (30 - a) = 210000) ∧
  ∀ b, b ≥ 10 → b ≤ 12 → 9000 * b + 6000 * (30 - b) ≥ 210000

-- Theorem Statements
theorem part1 (A B : ℕ → ℕ) : air_conditioner_cost A B → A 1 = 9000 ∧ B 1 = 6000 :=
by sorry

theorem part2 (A B : ℕ → ℕ) : air_conditioner_cost A B →
  possible_schemes A B :=
by sorry

theorem part3 (A B : ℕ → ℕ) : air_conditioner_cost A B ∧ possible_schemes A B →
  minimize_cost A B :=
by sorry

end part1_part2_part3_l43_43084


namespace maximum_value_of_a_squared_b_l43_43239

theorem maximum_value_of_a_squared_b {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a * (a + b) = 27) : 
  a^2 * b ≤ 54 :=
sorry

end maximum_value_of_a_squared_b_l43_43239


namespace karen_packs_cookies_l43_43076

-- Conditions stated as definitions
def school_days := 5
def peanut_butter_days := 2
def ham_sandwich_days := school_days - peanut_butter_days
def cake_days := 1
def probability_ham_and_cake := 0.12

-- Lean theorem statement
theorem karen_packs_cookies : 
  (school_days - cake_days - peanut_butter_days) = 2 :=
by
  sorry

end karen_packs_cookies_l43_43076


namespace ax_by_n_sum_l43_43453

theorem ax_by_n_sum {a b x y : ℝ} 
  (h1 : a * x + b * y = 2)
  (h2 : a * x^2 + b * y^2 = 5)
  (h3 : a * x^3 + b * y^3 = 15)
  (h4 : a * x^4 + b * y^4 = 35) :
  a * x^5 + b * y^5 = 10 :=
sorry

end ax_by_n_sum_l43_43453


namespace tangent_function_intersection_l43_43932

theorem tangent_function_intersection (ω : ℝ) (hω : ω > 0) (h_period : (π / ω) = 3 * π) :
  let f (x : ℝ) := Real.tan (ω * x + π / 3)
  f π = -Real.sqrt 3 :=
by
  sorry

end tangent_function_intersection_l43_43932


namespace pages_called_this_week_l43_43525

-- Definitions as per conditions
def pages_called_last_week := 10.2
def total_pages_called := 18.8

-- Theorem to prove the solution
theorem pages_called_this_week :
  total_pages_called - pages_called_last_week = 8.6 :=
by
  sorry

end pages_called_this_week_l43_43525


namespace no_solution_for_inequality_l43_43474

theorem no_solution_for_inequality (x : ℝ) (h : |x| > 2) : ¬ (5 * x^2 + 6 * x + 8 < 0) := 
by
  sorry

end no_solution_for_inequality_l43_43474


namespace value_of_M_l43_43545

theorem value_of_M (M : ℝ) (h : 0.25 * M = 0.35 * 1200) : M = 1680 := 
sorry

end value_of_M_l43_43545


namespace remainder_of_division_l43_43188

theorem remainder_of_division : 
  ∀ (L x : ℕ), (L = 1430) → 
               (L - x = 1311) → 
               (L = 11 * x + (L % x)) → 
               (L % x = 121) :=
by
  intros L x L_value diff quotient
  sorry

end remainder_of_division_l43_43188


namespace part_a_gray_black_area_difference_l43_43908

theorem part_a_gray_black_area_difference :
    ∀ (a b : ℕ), 
        a = 4 → 
        b = 3 →
        a^2 - b^2 = 7 :=
by
  intros a b h_a h_b
  sorry

end part_a_gray_black_area_difference_l43_43908


namespace convert_to_polar_l43_43964

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (r, θ)

theorem convert_to_polar (x y : ℝ) (hx : x = 8) (hy : y = 3 * Real.sqrt 3) :
  polar_coordinates x y = (Real.sqrt 91, Real.arctan (3 * Real.sqrt 3 / 8)) :=
by
  rw [hx, hy]
  simp [polar_coordinates]
  -- place to handle conversions and simplifications if necessary
  sorry

end convert_to_polar_l43_43964


namespace smallest_prime_dividing_sum_l43_43888

theorem smallest_prime_dividing_sum :
  ∃ p : ℕ, Prime p ∧ p ∣ (7^14 + 11^15) ∧ ∀ q : ℕ, Prime q ∧ q ∣ (7^14 + 11^15) → p ≤ q := by
  sorry

end smallest_prime_dividing_sum_l43_43888


namespace bob_second_third_lap_time_l43_43660

theorem bob_second_third_lap_time :
  ∀ (lap_length : ℕ) (first_lap_time : ℕ) (average_speed : ℕ),
  lap_length = 400 →
  first_lap_time = 70 →
  average_speed = 5 →
  ∃ (second_third_lap_time : ℕ), second_third_lap_time = 85 :=
by
  intros lap_length first_lap_time average_speed lap_length_eq first_lap_time_eq average_speed_eq
  sorry

end bob_second_third_lap_time_l43_43660


namespace son_age_l43_43666

variable (F S : ℕ)
variable (h₁ : F = 3 * S)
variable (h₂ : F - 8 = 4 * (S - 8))

theorem son_age : S = 24 := 
by 
  sorry

end son_age_l43_43666


namespace range_of_g_l43_43392

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : Set.Icc (-1.1071) 1.1071 = Set.image g (Set.Icc (-1:ℝ) 1) := by
  sorry

end range_of_g_l43_43392


namespace relatively_prime_27x_plus_4_18x_plus_3_l43_43555

theorem relatively_prime_27x_plus_4_18x_plus_3 (x : ℕ) :
  Nat.gcd (27 * x + 4) (18 * x + 3) = 1 :=
sorry

end relatively_prime_27x_plus_4_18x_plus_3_l43_43555


namespace count_valid_rods_l43_43486

def isValidRodLength (d : ℕ) : Prop :=
  5 ≤ d ∧ d < 27

def countValidRodLengths (lower upper : ℕ) : ℕ :=
  upper - lower + 1

theorem count_valid_rods :
  let valid_rods_count := countValidRodLengths 5 26
  valid_rods_count = 22 :=
by
  sorry

end count_valid_rods_l43_43486


namespace journey_distance_l43_43401

theorem journey_distance :
  ∃ D T : ℝ,
    D = 100 * T ∧
    D = 80 * (T + 1/3) ∧
    D = 400 / 3 :=
by
  sorry

end journey_distance_l43_43401


namespace total_sequins_correct_l43_43755

def blue_rows : ℕ := 6
def blue_columns : ℕ := 8
def purple_rows : ℕ := 5
def purple_columns : ℕ := 12
def green_rows : ℕ := 9
def green_columns : ℕ := 6

def total_sequins : ℕ :=
  (blue_rows * blue_columns) + (purple_rows * purple_columns) + (green_rows * green_columns)

theorem total_sequins_correct : total_sequins = 162 := by
  sorry

end total_sequins_correct_l43_43755


namespace moles_of_nacl_formed_l43_43171

noncomputable def reaction (nh4cl: ℕ) (naoh: ℕ) : ℕ :=
  if nh4cl = naoh then nh4cl else min nh4cl naoh

theorem moles_of_nacl_formed (nh4cl: ℕ) (naoh: ℕ) (h_nh4cl: nh4cl = 2) (h_naoh: naoh = 2) :
  reaction nh4cl naoh = 2 :=
by
  rw [h_nh4cl, h_naoh]
  sorry

end moles_of_nacl_formed_l43_43171


namespace parabola_vertex_intercept_l43_43362

variable (a b c p : ℝ)

theorem parabola_vertex_intercept (h_vertex : ∀ x : ℝ, (a * (x - p) ^ 2 + p) = a * x^2 + b * x + c)
                                  (h_intercept : a * p^2 + p = 2 * p)
                                  (hp : p ≠ 0) : b = -2 :=
sorry

end parabola_vertex_intercept_l43_43362


namespace negation_of_universal_prop_l43_43440

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^3 + 3^x > 0) ↔ (∃ x : ℝ, x^3 + 3^x ≤ 0) :=
by sorry

end negation_of_universal_prop_l43_43440


namespace maximum_sum_of_squares_l43_43922

theorem maximum_sum_of_squares (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 5) :
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ 20 :=
sorry

end maximum_sum_of_squares_l43_43922


namespace find_x_squared_plus_y_squared_l43_43580

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + x + y = 75) : x^2 + y^2 = 3205 / 121 :=
by
  sorry

end find_x_squared_plus_y_squared_l43_43580


namespace no_solution_to_system_l43_43447

open Real

theorem no_solution_to_system (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^(1/3) - y^(1/3) - z^(1/3) = 64) ∧ (x^(1/4) - y^(1/4) - z^(1/4) = 32) ∧ (x^(1/6) - y^(1/6) - z^(1/6) = 8) → False := by
  sorry

end no_solution_to_system_l43_43447


namespace average_cost_per_individual_before_gratuity_l43_43785

theorem average_cost_per_individual_before_gratuity
  (total_bill : ℝ)
  (num_people : ℕ)
  (gratuity_percentage : ℝ)
  (bill_including_gratuity : total_bill = 840)
  (group_size : num_people = 7)
  (gratuity : gratuity_percentage = 0.20) :
  (total_bill / (1 + gratuity_percentage)) / num_people = 100 :=
by
  sorry

end average_cost_per_individual_before_gratuity_l43_43785


namespace max_fruit_to_teacher_l43_43954

theorem max_fruit_to_teacher (A G : ℕ) : (A % 7 ≤ 6) ∧ (G % 7 ≤ 6) :=
by
  sorry

end max_fruit_to_teacher_l43_43954


namespace upper_limit_of_range_l43_43667

theorem upper_limit_of_range (N : ℕ) :
  (∀ n : ℕ, (20 + n * 10 ≤ N) = (n < 198)) → N = 1990 :=
by
  sorry

end upper_limit_of_range_l43_43667


namespace find_integer_l43_43826

theorem find_integer (n : ℕ) (h1 : 0 < n) (h2 : 200 % n = 2) (h3 : 398 % n = 2) : n = 6 :=
sorry

end find_integer_l43_43826


namespace devin_basketball_chances_l43_43834

theorem devin_basketball_chances 
  (initial_chances : ℝ := 0.1) 
  (base_height : ℕ := 66) 
  (chance_increase_per_inch : ℝ := 0.1)
  (initial_height : ℕ := 65) 
  (growth : ℕ := 3) :
  initial_chances + (growth + initial_height - base_height) * chance_increase_per_inch = 0.3 := 
by 
  sorry

end devin_basketball_chances_l43_43834


namespace smallest_perimeter_scalene_triangle_l43_43790

theorem smallest_perimeter_scalene_triangle (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) :
  a + b + c = 9 := 
sorry

end smallest_perimeter_scalene_triangle_l43_43790


namespace reach_any_composite_from_4_l43_43490

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

def can_reach (A : ℕ) : Prop :=
  ∀ n : ℕ, is_composite n → ∃ seq : ℕ → ℕ, seq 0 = A ∧ seq (n + 1) - seq n ∣ seq n ∧ seq (n + 1) ≠ seq n ∧ seq (n + 1) ≠ 1 ∧ seq (n + 1) = n

theorem reach_any_composite_from_4 : can_reach 4 :=
  sorry

end reach_any_composite_from_4_l43_43490


namespace age_ratio_l43_43312

variable (p q : ℕ)

-- Conditions
def condition1 := p - 6 = (q - 6) / 2
def condition2 := p + q = 21

-- Theorem stating the desired ratio
theorem age_ratio (h1 : condition1 p q) (h2 : condition2 p q) : p / Nat.gcd p q = 3 ∧ q / Nat.gcd p q = 4 :=
by
  sorry

end age_ratio_l43_43312


namespace find_x_for_opposite_expressions_l43_43963

theorem find_x_for_opposite_expressions :
  ∃ x : ℝ, (x + 1) + (3 * x - 5) = 0 ↔ x = 1 :=
by
  sorry

end find_x_for_opposite_expressions_l43_43963


namespace small_seat_capacity_l43_43530

-- Definitions for the conditions
def smallSeats : Nat := 2
def largeSeats : Nat := 23
def capacityLargeSeat : Nat := 54
def totalPeopleSmallSeats : Nat := 28

-- Theorem statement
theorem small_seat_capacity : totalPeopleSmallSeats / smallSeats = 14 := by
  sorry

end small_seat_capacity_l43_43530


namespace third_part_of_156_division_proof_l43_43510

theorem third_part_of_156_division_proof :
  ∃ (x : ℚ), (2 * x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 156) ∧ ((1 / 4) * x = 13 + 15 / 23) :=
by
  sorry

end third_part_of_156_division_proof_l43_43510


namespace roberta_listen_days_l43_43798

-- Define the initial number of records
def initial_records : ℕ := 8

-- Define the number of records received as gifts
def gift_records : ℕ := 12

-- Define the number of records bought
def bought_records : ℕ := 30

-- Define the number of days to listen to 1 record
def days_per_record : ℕ := 2

-- Define the total number of records
def total_records : ℕ := initial_records + gift_records + bought_records

-- Define the total number of days required to listen to all records
def total_days : ℕ := total_records * days_per_record

-- Theorem to prove the total days needed to listen to all records is 100
theorem roberta_listen_days : total_days = 100 := by
  sorry

end roberta_listen_days_l43_43798


namespace gumball_cost_l43_43776

theorem gumball_cost (n : ℕ) (T : ℕ) (h₁ : n = 4) (h₂ : T = 32) : T / n = 8 := by
  sorry

end gumball_cost_l43_43776


namespace parallel_lines_l43_43306

theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (3 * a - 1) * x - 4 * a * y - 1 = 0 → False) → 
  (a = 0 ∨ a = -1/3) :=
sorry

end parallel_lines_l43_43306


namespace width_of_room_l43_43111

theorem width_of_room (length room_area cost paving_rate : ℝ) 
  (H_length : length = 5.5) 
  (H_cost : cost = 17600)
  (H_paving_rate : paving_rate = 800)
  (H_area : room_area = cost / paving_rate) :
  room_area = length * 4 :=
by
  -- sorry to skip proof
  sorry

end width_of_room_l43_43111


namespace sequence_k_value_l43_43384

theorem sequence_k_value {k : ℕ} (h : 9 < (2 * k - 8) ∧ (2 * k - 8) < 12) 
  (Sn : ℕ → ℤ) (hSn : ∀ n, Sn n = n^2 - 7*n) 
  : k = 9 :=
by
  sorry

end sequence_k_value_l43_43384


namespace number_of_BMWs_sold_l43_43593

theorem number_of_BMWs_sold (total_cars : ℕ) (Audi_percent Toyota_percent Acura_percent Ford_percent : ℝ)
  (h_total_cars : total_cars = 250) 
  (h_percentages : Audi_percent = 0.10 ∧ Toyota_percent = 0.20 ∧ Acura_percent = 0.15 ∧ Ford_percent = 0.25) :
  ∃ (BMWs_sold : ℕ), BMWs_sold = 75 := 
by
  sorry

end number_of_BMWs_sold_l43_43593


namespace distinct_real_roots_of_quadratic_l43_43558

theorem distinct_real_roots_of_quadratic (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x : ℝ, x^2 - 4*x + 2*m = 0 ↔ x = x₁ ∨ x = x₂)) ↔ m < 2 := by
sorry

end distinct_real_roots_of_quadratic_l43_43558


namespace rewrite_subtraction_rewrite_division_l43_43432

theorem rewrite_subtraction : -8 - 5 = -8 + (-5) :=
by sorry

theorem rewrite_division : (1/2) / (-2) = (1/2) * (-1/2) :=
by sorry

end rewrite_subtraction_rewrite_division_l43_43432


namespace beans_in_jar_l43_43861

theorem beans_in_jar (B : ℕ) 
  (h1 : B / 4 = number_of_red_beans)
  (h2 : number_of_red_beans = B / 4)
  (h3 : number_of_white_beans = (B * 3 / 4) / 3)
  (h4 : number_of_white_beans = B / 4)
  (h5 : number_of_remaining_beans_after_white = B / 2)
  (h6 : 143 = B / 4):
  B = 572 :=
by
  sorry

end beans_in_jar_l43_43861


namespace at_least_six_heads_in_10_flips_is_129_over_1024_l43_43773

def fair_coin_flip (n : ℕ) (prob_heads prob_tails : ℚ) : Prop :=
  (prob_heads = 1/2 ∧ prob_tails = 1/2)

noncomputable def at_least_six_consecutive_heads_probability (n : ℕ) : ℚ :=
  if n = 10 then 129 / 1024 else 0  -- this is specific to 10 flips and should be defined based on actual calculation for different n
  
theorem at_least_six_heads_in_10_flips_is_129_over_1024 :
  fair_coin_flip 10 (1/2) (1/2) →
  at_least_six_consecutive_heads_probability 10 = 129 / 1024 :=
by
  intros
  sorry

end at_least_six_heads_in_10_flips_is_129_over_1024_l43_43773


namespace union_A_B_equals_x_lt_3_l43_43648

theorem union_A_B_equals_x_lt_3 :
  let A := { x : ℝ | 3 - x > 0 ∧ x + 2 > 0 }
  let B := { x : ℝ | 3 > 2*x - 1 }
  A ∪ B = { x : ℝ | x < 3 } :=
by
  sorry

end union_A_B_equals_x_lt_3_l43_43648


namespace problem_statement_l43_43681

def g (x : ℝ) : ℝ :=
  x^2 - 5 * x

theorem problem_statement (x : ℝ) :
  (g (g x) = g x) ↔ (x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1) :=
by
  sorry

end problem_statement_l43_43681


namespace find_monthly_salary_l43_43028

variable (S : ℝ)

theorem find_monthly_salary
  (h1 : 0.20 * S - 0.20 * (0.20 * S) = 220) :
  S = 1375 :=
by
  -- Proof goes here
  sorry

end find_monthly_salary_l43_43028


namespace neg_sin_leq_one_l43_43832

theorem neg_sin_leq_one (p : Prop) :
  (∀ x : ℝ, Real.sin x ≤ 1) → (¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end neg_sin_leq_one_l43_43832


namespace negation_of_universal_statement_l43_43584

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, x^2 ≤ 1) ↔ ∃ x : ℝ, x^2 > 1 :=
by
  sorry

end negation_of_universal_statement_l43_43584


namespace english_teachers_count_l43_43702

theorem english_teachers_count (E : ℕ) 
    (h_prob : 6 / ((E + 6) * (E + 5) / 2) = 1 / 12) : 
    E = 3 :=
by
  sorry

end english_teachers_count_l43_43702


namespace radio_price_and_total_items_l43_43205

theorem radio_price_and_total_items :
  ∃ (n : ℕ) (p : ℝ),
    (∀ (i : ℕ), (1 ≤ i ∧ i ≤ n) → (i = 1 ∨ ∃ (j : ℕ), i = j + 1 ∧ p = 1 + (j * 0.50))) ∧
    (n - 49 = 85) ∧
    (p = 43) ∧
    (n = 134) :=
by {
  sorry
}

end radio_price_and_total_items_l43_43205


namespace div_poly_iff_l43_43864

-- Definitions from conditions
def P (x : ℂ) (n : ℕ) := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℂ) := x^4 + x^3 + x^2 + x + 1

-- The main theorem stating the problem
theorem div_poly_iff (n : ℕ) : 
  ∀ x : ℂ, (P x n) ∣ (Q x) ↔ n % 5 ≠ 0 :=
by sorry

end div_poly_iff_l43_43864


namespace total_sheets_l43_43472

-- Define the conditions
def sheets_in_bundle : ℕ := 10
def bundles : ℕ := 3
def additional_sheets : ℕ := 8

-- Theorem to prove the total number of sheets Jungkook has
theorem total_sheets : bundles * sheets_in_bundle + additional_sheets = 38 := by
  sorry

end total_sheets_l43_43472


namespace minimum_words_to_learn_l43_43448

-- Definition of the problem
def total_words : ℕ := 600
def required_percentage : ℕ := 90

-- Lean statement of the problem
theorem minimum_words_to_learn : ∃ x : ℕ, (x / total_words : ℚ) = required_percentage / 100 ∧ x = 540 :=
sorry

end minimum_words_to_learn_l43_43448


namespace value_divided_by_l43_43833

theorem value_divided_by {x : ℝ} : (5 / x) * 12 = 10 → x = 6 :=
by
  sorry

end value_divided_by_l43_43833


namespace price_is_219_l43_43454

noncomputable def discount_coupon1 (price : ℝ) : ℝ :=
  if price > 50 then 0.1 * price else 0

noncomputable def discount_coupon2 (price : ℝ) : ℝ :=
  if price > 100 then 20 else 0

noncomputable def discount_coupon3 (price : ℝ) : ℝ :=
  if price > 100 then 0.18 * (price - 100) else 0

noncomputable def more_savings_coupon1 (price : ℝ) : Prop :=
  discount_coupon1 price > discount_coupon2 price ∧ discount_coupon1 price > discount_coupon3 price

theorem price_is_219 (price : ℝ) :
  more_savings_coupon1 price → price = 219 :=
by
  sorry

end price_is_219_l43_43454


namespace abs_quotient_eq_sqrt_7_div_2_l43_43907

theorem abs_quotient_eq_sqrt_7_div_2 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 5 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 2) :=
by
  sorry

end abs_quotient_eq_sqrt_7_div_2_l43_43907


namespace connor_cats_l43_43333

theorem connor_cats (j : ℕ) (a : ℕ) (m : ℕ) (c : ℕ) (co : ℕ) (x : ℕ) 
  (h1 : a = j / 3)
  (h2 : m = 2 * a)
  (h3 : c = a / 2)
  (h4 : c = co + 5)
  (h5 : j = 90)
  (h6 : x = j + a + m + c + co) : 
  co = 10 := 
by
  sorry

end connor_cats_l43_43333


namespace positive_solution_x_l43_43673

theorem positive_solution_x (x y z : ℝ) (h1 : x * y = 10 - 3 * x - 2 * y) 
(h2 : y * z = 10 - 5 * y - 3 * z) 
(h3 : x * z = 40 - 5 * x - 2 * z) 
(h_pos : x > 0) : 
  x = 8 :=
sorry

end positive_solution_x_l43_43673


namespace concyclic_iff_ratio_real_l43_43332

noncomputable def concyclic_condition (z1 z2 z3 z4 : ℂ) : Prop :=
  (∃ c : ℂ, c ≠ 0 ∧ ∀ (w : ℂ), (w - z1) * (w - z3) / ((w - z2) * (w - z4)) = c)

noncomputable def ratio_real (z1 z2 z3 z4 : ℂ) : Prop :=
  ∃ r : ℝ, (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3)) = r

theorem concyclic_iff_ratio_real (z1 z2 z3 z4 : ℂ) :
  concyclic_condition z1 z2 z3 z4 ↔ ratio_real z1 z2 z3 z4 :=
sorry

end concyclic_iff_ratio_real_l43_43332


namespace difference_max_min_y_l43_43654

theorem difference_max_min_y {total_students : ℕ} (initial_yes_pct initial_no_pct final_yes_pct final_no_pct : ℝ)
  (initial_conditions : initial_yes_pct = 0.4 ∧ initial_no_pct = 0.6)
  (final_conditions : final_yes_pct = 0.8 ∧ final_no_pct = 0.2) :
  ∃ (min_change max_change : ℝ), max_change - min_change = 0.2 := by
  sorry

end difference_max_min_y_l43_43654


namespace variable_swap_l43_43509

theorem variable_swap (x y t : Nat) (h1 : x = 5) (h2 : y = 6) (h3 : t = x) (h4 : x = y) (h5 : y = t) : 
  x = 6 ∧ y = 5 := 
by
  sorry

end variable_swap_l43_43509


namespace ratio_new_average_to_original_l43_43160

theorem ratio_new_average_to_original (scores : List ℝ) (h_len : scores.length = 50) :
  let A := (scores.sum / scores.length : ℝ)
  let new_sum := scores.sum + 2 * A
  let new_avg := new_sum / (scores.length + 2)
  new_avg / A = 1 := 
by
  sorry

end ratio_new_average_to_original_l43_43160


namespace concentric_circles_circumference_difference_and_area_l43_43293

theorem concentric_circles_circumference_difference_and_area {r_inner r_outer : ℝ} (h1 : r_inner = 25) (h2 : r_outer = r_inner + 15) :
  2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 30 * Real.pi ∧ Real.pi * r_outer^2 - Real.pi * r_inner^2 = 975 * Real.pi :=
by
  sorry

end concentric_circles_circumference_difference_and_area_l43_43293


namespace undefined_value_l43_43267

theorem undefined_value (x : ℝ) : (x^2 - 16 * x + 64 = 0) → (x = 8) := by
  sorry

end undefined_value_l43_43267


namespace quadratic_complete_square_l43_43701

theorem quadratic_complete_square :
  ∃ a b c : ℝ, (∀ x : ℝ, 4 * x^2 - 40 * x + 100 = a * (x + b)^2 + c) ∧ a + b + c = -1 :=
sorry

end quadratic_complete_square_l43_43701


namespace green_flower_percentage_l43_43706

theorem green_flower_percentage (yellow purple green total : ℕ)
  (hy : yellow = 10)
  (hp : purple = 18)
  (ht : total = 35)
  (hgreen : green = total - (yellow + purple)) :
  ((green * 100) / (yellow + purple)) = 25 := 
by {
  sorry
}

end green_flower_percentage_l43_43706


namespace largest_digit_B_divisible_by_4_l43_43390

theorem largest_digit_B_divisible_by_4 :
  ∃ (B : ℕ), B ≤ 9 ∧ ∀ B', (B' ≤ 9 ∧ (4 * 10^5 + B' * 10^4 + 5 * 10^3 + 7 * 10^2 + 8 * 10 + 4) % 4 = 0) → B' ≤ B :=
by
  sorry

end largest_digit_B_divisible_by_4_l43_43390


namespace blackRhinoCount_correct_l43_43582

noncomputable def numberOfBlackRhinos : ℕ :=
  let whiteRhinoCount := 7
  let whiteRhinoWeight := 5100
  let blackRhinoWeightInTons := 1
  let totalWeight := 51700
  let oneTonInPounds := 2000
  let totalWhiteRhinoWeight := whiteRhinoCount * whiteRhinoWeight
  let totalBlackRhinoWeight := totalWeight - totalWhiteRhinoWeight
  totalBlackRhinoWeight / (blackRhinoWeightInTons * oneTonInPounds)

theorem blackRhinoCount_correct : numberOfBlackRhinos = 8 := by
  sorry

end blackRhinoCount_correct_l43_43582


namespace password_probability_l43_43893

def isNonNegativeSingleDigit (n : ℕ) : Prop := n ≤ 9

def isOddSingleDigit (n : ℕ) : Prop := isNonNegativeSingleDigit n ∧ n % 2 = 1

def isPositiveSingleDigit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def isVowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

-- Probability that an odd single-digit number followed by a vowel and a positive single-digit number
def prob_odd_vowel_positive_digits : ℚ :=
  let prob_first := 5 / 10 -- Probability of odd single-digit number
  let prob_vowel := 5 / 26 -- Probability of vowel
  let prob_last := 9 / 10 -- Probability of positive single-digit number
  prob_first * prob_vowel * prob_last

theorem password_probability :
  prob_odd_vowel_positive_digits = 9 / 104 :=
by
  sorry

end password_probability_l43_43893


namespace zoe_correct_percentage_l43_43712

variable (t : ℝ) -- total number of problems

-- Conditions
variable (chloe_solved_fraction : ℝ := 0.60)
variable (zoe_solved_fraction : ℝ := 0.40)
variable (chloe_correct_percentage_alone : ℝ := 0.75)
variable (chloe_correct_percentage_total : ℝ := 0.85)
variable (zoe_correct_percentage_alone : ℝ := 0.95)

theorem zoe_correct_percentage (h1 : chloe_solved_fraction = 0.60)
                               (h2 : zoe_solved_fraction = 0.40)
                               (h3 : chloe_correct_percentage_alone = 0.75)
                               (h4 : chloe_correct_percentage_total = 0.85)
                               (h5 : zoe_correct_percentage_alone = 0.95) :
  (zoe_correct_percentage_alone * zoe_solved_fraction * 100 + (chloe_correct_percentage_total - chloe_correct_percentage_alone * chloe_solved_fraction) * 100 = 78) :=
sorry

end zoe_correct_percentage_l43_43712


namespace parabola_directrix_l43_43994

-- Defining the given condition
def given_parabola_equation (x y : ℝ) : Prop := y = 2 * x^2

-- Defining the expected directrix equation for the parabola
def directrix_equation (y : ℝ) : Prop := y = -1 / 8

-- The theorem we aim to prove
theorem parabola_directrix :
  (∀ x y : ℝ, given_parabola_equation x y) → (directrix_equation (-1 / 8)) :=
by
  -- Using 'sorry' here since the proof is not required
  sorry

end parabola_directrix_l43_43994


namespace two_digit_number_formed_l43_43971

theorem two_digit_number_formed (A B C D E F : ℕ) 
  (A_C_D_const : A + C + D = constant)
  (A_B_const : A + B = constant)
  (B_D_F_const : B + D + F = constant)
  (E_F_const : E + F = constant)
  (E_B_C_const : E + B + C = constant)
  (B_eq_C_D : B = C + D)
  (B_D_eq_E : B + D = E)
  (E_C_eq_A : E + C = A) 
  (hA : A = 6) 
  (hB : B = 3)
  : 10 * A + B = 63 :=
by sorry

end two_digit_number_formed_l43_43971


namespace solve_equation_l43_43727

theorem solve_equation (x : ℝ) (h : (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2) : 
  x = -2/3 :=
sorry

end solve_equation_l43_43727


namespace find_a1_l43_43370

-- Defining the conditions
variables (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_monotone : ∀ n, a n ≥ a (n + 1)) -- Monotonically decreasing

-- Specific values from the problem
axiom h_a3 : a 3 = 1
axiom h_a2_a4 : a 2 + a 4 = 5 / 2
axiom h_geom_seq : ∀ n, a (n + 1) = a n * q  -- Geometric sequence property

-- The goal is to prove that a 1 = 4
theorem find_a1 : a 1 = 4 :=
by
  -- Insert proof here
  sorry

end find_a1_l43_43370


namespace problem_inequality_problem_equality_condition_l43_43976

theorem problem_inequality (a b c : ℕ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c :=
sorry

theorem problem_equality_condition (a b c : ℕ) :
  (a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ a + 1 = b ∧ b + 1 = c :=
sorry

end problem_inequality_problem_equality_condition_l43_43976


namespace ellipse_semimajor_axis_value_l43_43756

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l43_43756


namespace base_8_addition_l43_43426

-- Definitions
def five_base_8 : ℕ := 5
def thirteen_base_8 : ℕ := 1 * 8 + 3 -- equivalent of (13)_8 in base 10

-- Theorem statement
theorem base_8_addition :
  (five_base_8 + thirteen_base_8) = 2 * 8 + 0 :=
sorry

end base_8_addition_l43_43426


namespace robert_cash_spent_as_percentage_l43_43194

theorem robert_cash_spent_as_percentage 
  (raw_material_cost : ℤ) (machinery_cost : ℤ) (total_amount : ℤ) 
  (h_raw : raw_material_cost = 100) 
  (h_machinery : machinery_cost = 125) 
  (h_total : total_amount = 250) :
  ((total_amount - (raw_material_cost + machinery_cost)) * 100 / total_amount) = 10 := 
by 
  -- Proof will be filled here
  sorry

end robert_cash_spent_as_percentage_l43_43194


namespace correct_calculation_result_l43_43318

theorem correct_calculation_result (x : ℤ) (h : x + 44 - 39 = 63) : x + 39 - 44 = 53 := by
  sorry

end correct_calculation_result_l43_43318


namespace degree_of_vertex_angle_of_isosceles_triangle_l43_43505

theorem degree_of_vertex_angle_of_isosceles_triangle (exterior_angle : ℝ) (h_exterior_angle : exterior_angle = 40) : 
∃ vertex_angle : ℝ, vertex_angle = 140 :=
by 
  sorry

end degree_of_vertex_angle_of_isosceles_triangle_l43_43505


namespace Mike_books_l43_43429

theorem Mike_books
  (initial_books : ℝ)
  (books_sold : ℝ)
  (books_gifts : ℝ) 
  (books_bought : ℝ)
  (h_initial : initial_books = 51.5)
  (h_sold : books_sold = 45.75)
  (h_gifts : books_gifts = 12.25)
  (h_bought : books_bought = 3.5):
  initial_books - books_sold + books_gifts + books_bought = 21.5 := 
sorry

end Mike_books_l43_43429


namespace age_of_oldest_child_l43_43176

def average_age_of_children (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem age_of_oldest_child :
  ∀ (a b c d : ℕ), a = 6 → b = 9 → c = 12 → average_age_of_children a b c d = 9 → d = 9 :=
by
  intros a b c d h_a h_b h_c h_avg
  sorry

end age_of_oldest_child_l43_43176


namespace evaluate_expression_l43_43156

theorem evaluate_expression : 4 * 12 + 5 * 11 + 6^2 + 7 * 9 = 202 :=
by sorry

end evaluate_expression_l43_43156


namespace negation_of_existence_l43_43256

theorem negation_of_existence :
  ¬ (∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
sorry

end negation_of_existence_l43_43256


namespace find_total_amount_before_brokerage_l43_43770

noncomputable def total_amount_before_brokerage (realized_amount : ℝ) (brokerage_rate : ℝ) : ℝ :=
  realized_amount / (1 - brokerage_rate / 100)

theorem find_total_amount_before_brokerage :
  total_amount_before_brokerage 107.25 (1 / 4) = 107.25 * 400 / 399 := by
sorry

end find_total_amount_before_brokerage_l43_43770


namespace first_month_sale_l43_43965

theorem first_month_sale 
(sale_2 sale_3 sale_4 sale_5 sale_6 : ℕ)
(avg_sale : ℕ) 
(h_avg: avg_sale = 6500)
(h_sale2: sale_2 = 6927)
(h_sale3: sale_3 = 6855)
(h_sale4: sale_4 = 7230)
(h_sale5: sale_5 = 6562)
(h_sale6: sale_6 = 4791)
: sale_1 = 6635 := by
  sorry

end first_month_sale_l43_43965


namespace logician1_max_gain_l43_43104

noncomputable def maxCoinsDistribution (logician1 logician2 logician3 : ℕ) := (logician1, logician2, logician3)

theorem logician1_max_gain 
  (total_coins : ℕ) 
  (coins1 coins2 coins3 : ℕ) 
  (H : total_coins = 10)
  (H1 : ¬ (coins1 = 9 ∧ coins2 = 0 ∧ coins3 = 1) → coins1 = 2):
  maxCoinsDistribution coins1 coins2 coins3 = (9, 0, 1) :=
by
  sorry

end logician1_max_gain_l43_43104


namespace log_comparison_l43_43110

theorem log_comparison (a b : ℝ) (h1 : 0 < a) (h2 : a < e) (h3 : 0 < b) (h4 : b < e) (h5 : a < b) :
  a * Real.log b > b * Real.log a := sorry

end log_comparison_l43_43110


namespace cat_moves_on_circular_arc_l43_43843

theorem cat_moves_on_circular_arc (L : ℝ) (x y : ℝ)
  (h : x^2 + y^2 = L^2) :
  (x / 2)^2 + (y / 2)^2 = (L / 2)^2 :=
  by sorry

end cat_moves_on_circular_arc_l43_43843


namespace solve_for_x_l43_43663

theorem solve_for_x (x y : ℝ) 
  (h1 : 3 * x - 2 * y = 8) 
  (h2 : 2 * x + 3 * y = 11) :
  x = 46 / 13 :=
by
  sorry

end solve_for_x_l43_43663


namespace find_a_value_l43_43260

theorem find_a_value (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : (∃ l : ℝ, ∃ f : ℝ → ℝ, f x = a^x ∧ deriv f 0 = -1)) :
  a = 1 / Real.exp 1 := by
  sorry

end find_a_value_l43_43260


namespace igors_number_l43_43008

-- Define the initial lineup of players
def initialLineup : List ℕ := [9, 7, 11, 10, 6, 8, 5, 4, 1]

-- Define the condition for a player running to the locker room
def runsToLockRoom (n : ℕ) (left : Option ℕ) (right : Option ℕ) : Prop :=
  match left, right with
  | some l, some r => n < l ∨ n < r
  | some l, none   => n < l
  | none, some r   => n < r
  | none, none     => False

-- Define the process of players running to the locker room iteratively
def runProcess : List ℕ → List ℕ := 
  sorry   -- Implementation of the run process is skipped

-- Define the remaining players after repeated commands until 3 players are left
def remainingPlayers (lineup : List ℕ) : List ℕ :=
  sorry  -- Implementation to find the remaining players is skipped

-- Statement of the theorem
theorem igors_number (afterIgorRanOff : List ℕ := remainingPlayers initialLineup)
  (finalLineup : List ℕ := [9, 11, 10]) :
  ∃ n, n ∈ initialLineup ∧ ¬(n ∈ finalLineup) ∧ afterIgorRanOff.length = 3 → n = 5 :=
  sorry

end igors_number_l43_43008


namespace total_commute_time_l43_43724

theorem total_commute_time 
  (first_bus : ℕ) (delay1 : ℕ) (wait1 : ℕ) 
  (second_bus : ℕ) (delay2 : ℕ) (wait2 : ℕ) 
  (third_bus : ℕ) (delay3 : ℕ) 
  (arrival_time : ℕ) :
  first_bus = 40 →
  delay1 = 10 →
  wait1 = 10 →
  second_bus = 50 →
  delay2 = 5 →
  wait2 = 15 →
  third_bus = 95 →
  delay3 = 15 →
  arrival_time = 540 →
  first_bus + delay1 + wait1 + second_bus + delay2 + wait2 + third_bus + delay3 = 240 :=
by
  intros
  sorry

end total_commute_time_l43_43724


namespace percentage_seeds_from_dandelions_l43_43899

def Carla_sunflowers := 6
def Carla_dandelions := 8
def seeds_per_sunflower := 9
def seeds_per_dandelion := 12

theorem percentage_seeds_from_dandelions :
  96 / 150 * 100 = 64 := by
  sorry

end percentage_seeds_from_dandelions_l43_43899


namespace full_time_worked_year_l43_43204

-- Define the conditions as constants
def total_employees : ℕ := 130
def full_time : ℕ := 80
def worked_year : ℕ := 100
def neither : ℕ := 20

-- Define the question as a theorem stating the correct answer
theorem full_time_worked_year : full_time + worked_year - total_employees + neither = 70 :=
by
  sorry

end full_time_worked_year_l43_43204


namespace batsman_new_average_l43_43515

def batsman_average_after_16_innings (A : ℕ) (new_avg : ℕ) (runs_16th : ℕ) : Prop :=
  15 * A + runs_16th = 16 * new_avg

theorem batsman_new_average (A : ℕ) (runs_16th : ℕ) (h1 : batsman_average_after_16_innings A (A + 3) runs_16th) : A + 3 = 19 :=
by
  sorry

end batsman_new_average_l43_43515


namespace simplify_and_evaluate_expression_l43_43012

theorem simplify_and_evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = -1) :
  (5 * x ^ 2 - 2 * (3 * y ^ 2 + 6 * x) + (2 * y ^ 2 - 5 * x ^ 2)) = 8 :=
by
  sorry

end simplify_and_evaluate_expression_l43_43012


namespace simplify_fraction_l43_43043

theorem simplify_fraction (a : ℕ) (h : a = 5) : (15 * a^4) / (75 * a^3) = 1 := 
by
  sorry

end simplify_fraction_l43_43043


namespace sqrt_diff_nat_l43_43742

open Nat

theorem sqrt_diff_nat (a b : ℕ) (h : 2015 * a^2 + a = 2016 * b^2 + b) : ∃ k : ℕ, a - b = k^2 := 
by
  sorry

end sqrt_diff_nat_l43_43742


namespace planks_ratio_l43_43992

theorem planks_ratio (P S : ℕ) (H : S + 100 + 20 + 30 = 200) (T : P = 200) (R : S = 200 / 2) : 
(S : ℚ) / P = 1 / 2 :=
by
  sorry

end planks_ratio_l43_43992


namespace zero_points_ordering_l43_43939

noncomputable def f (x : ℝ) : ℝ := x + 2^x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) : ℝ := x^3 + x - 2

theorem zero_points_ordering :
  ∃ x1 x2 x3 : ℝ,
    f x1 = 0 ∧ x1 < 0 ∧ 
    g x2 = 0 ∧ 0 < x2 ∧ x2 < 1 ∧
    h x3 = 0 ∧ 1 < x3 ∧ x3 < 2 ∧
    x1 < x2 ∧ x2 < x3 := sorry

end zero_points_ordering_l43_43939


namespace parallel_line_slope_l43_43277

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : 
  ∃ m : ℝ, m = 1 / 2 ∧ (∀ x1 y1 : ℝ, 3 * x1 - 6 * y1 = 12 → 
    ∃ k : ℝ, y1 = m * x1 + k) :=
by
  sorry

end parallel_line_slope_l43_43277


namespace range_of_a_l43_43109

theorem range_of_a (a : ℝ) (x : ℝ) :
  ((a < x ∧ x < a + 2) → x > 3) ∧ ¬(∀ x, (x > 3) → (a < x ∧ x < a + 2)) → a ≥ 3 :=
by
  sorry

end range_of_a_l43_43109


namespace integer_solutions_to_equation_l43_43609

theorem integer_solutions_to_equation :
  { p : ℤ × ℤ | (p.1 ^ 2 * p.2 + 1 = p.1 ^ 2 + 2 * p.1 * p.2 + 2 * p.1 + p.2) } =
  { (-1, -1), (0, 1), (1, -1), (2, -7), (3, 7) } :=
by
  sorry

end integer_solutions_to_equation_l43_43609


namespace jude_age_today_l43_43272
-- Import the necessary libraries

-- Define the conditions as hypotheses and then state the required proof
theorem jude_age_today (heath_age_today : ℕ) (heath_age_in_5_years : ℕ) (jude_age_in_5_years : ℕ) 
  (H1 : heath_age_today = 16)
  (H2 : heath_age_in_5_years = heath_age_today + 5)
  (H3 : heath_age_in_5_years = 3 * jude_age_in_5_years) :
  jude_age_in_5_years - 5 = 2 :=
by
  -- Given conditions imply Jude's age today is 2. Proof is omitted.
  sorry

end jude_age_today_l43_43272


namespace volunteer_org_percentage_change_l43_43923

theorem volunteer_org_percentage_change :
  ∀ (X : ℝ), X > 0 → 
  let fall_increase := 1.09 * X
  let spring_decrease := 0.81 * fall_increase
  (X - spring_decrease) / X * 100 = 11.71 :=
by
  intro X hX
  let fall_increase := 1.09 * X
  let spring_decrease := 0.81 * fall_increase
  show (_ - _) / _ * _ = _
  sorry

end volunteer_org_percentage_change_l43_43923


namespace min_value_x_y_l43_43723

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 8 / y = 1) : x + y ≥ 18 := 
sorry

end min_value_x_y_l43_43723


namespace sum_f_1_to_10_l43_43469

-- Define the function f with the properties given.

def f (x : ℝ) : ℝ := sorry

-- Specify the conditions of the problem
local notation "R" => ℝ

axiom odd_function : ∀ (x : R), f (-x) = -f (x)
axiom periodicity : ∀ (x : R), f (x + 3) = f (x)
axiom f_neg1 : f (-1) = 1

-- State the theorem to be proved
theorem sum_f_1_to_10 : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry
end sum_f_1_to_10_l43_43469


namespace subject_difference_l43_43973

-- Define the problem in terms of conditions and question
theorem subject_difference (C R M : ℕ) (hC : C = 10) (hR : R = C + 4) (hM : M + R + C = 41) : M - R = 3 :=
by
  -- Lean expects a proof here, we skip it with sorry
  sorry

end subject_difference_l43_43973


namespace arcsin_one_eq_pi_div_two_l43_43375

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = (Real.pi / 2) :=
by
  sorry

end arcsin_one_eq_pi_div_two_l43_43375


namespace break_even_point_l43_43411

/-- Conditions of the problem -/
def fixed_costs : ℝ := 10410
def variable_cost_per_unit : ℝ := 2.65
def selling_price_per_unit : ℝ := 20

/-- The mathematically equivalent proof problem / statement -/
theorem break_even_point :
  fixed_costs / (selling_price_per_unit - variable_cost_per_unit) = 600 := 
by
  -- Proof to be filled in
  sorry

end break_even_point_l43_43411


namespace compare_sqrt_differences_l43_43787

theorem compare_sqrt_differences :
  let a := (Real.sqrt 7) - (Real.sqrt 6)
  let b := (Real.sqrt 3) - (Real.sqrt 2)
  a < b :=
by
  sorry -- Proof goes here

end compare_sqrt_differences_l43_43787


namespace f_even_l43_43072

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  sorry

end f_even_l43_43072


namespace asymptotes_N_are_correct_l43_43744

-- Given the conditions of the hyperbola M
def hyperbola_M (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / m - y^2 / 6 = 1

-- Eccentricity condition
def eccentricity (m : ℝ) (e : ℝ) : Prop :=
  e = 2 ∧ (m > 0)

-- Given hyperbola N
def hyperbola_N (x y : ℝ) (m : ℝ) : Prop :=
  x^2 - y^2 / m = 1

-- The theorem to be proved
theorem asymptotes_N_are_correct (m : ℝ) (x y : ℝ) :
  hyperbola_M x y 2 → eccentricity 2 2 → hyperbola_N x y m →
  (y = x * Real.sqrt 2 ∨ y = -x * Real.sqrt 2) :=
by
  sorry

end asymptotes_N_are_correct_l43_43744


namespace quadratic_function_value_at_2_l43_43274

theorem quadratic_function_value_at_2 
  (a b c : ℝ) (h_a : a ≠ 0) 
  (h1 : 7 = a * (-3)^2 + b * (-3) + c)
  (h2 : 7 = a * (5)^2 + b * 5 + c)
  (h3 : -8 = c) :
  a * 2^2 + b * 2 + c = -8 := by 
  sorry

end quadratic_function_value_at_2_l43_43274


namespace find_c_l43_43945

theorem find_c (x c : ℝ) (h : ((5 * x + 38 + c) / 5) = (x + 4) + 5) : c = 7 :=
by
  sorry

end find_c_l43_43945


namespace cello_viola_pairs_l43_43926

theorem cello_viola_pairs (cellos violas : Nat) (p_same_tree : ℚ) (P : Nat)
  (h_cellos : cellos = 800)
  (h_violas : violas = 600)
  (h_p_same_tree : p_same_tree = 0.00020833333333333335)
  (h_equation : P * ((1 : ℚ) / cellos * (1 : ℚ) / violas) = p_same_tree) :
  P = 100 := 
by
  sorry

end cello_viola_pairs_l43_43926


namespace number_of_tickets_l43_43195

-- Define the given conditions
def initial_premium := 50 -- dollars per month
def premium_increase_accident (initial_premium : ℕ) := initial_premium / 10 -- 10% increase
def premium_increase_ticket := 5 -- dollars per month per ticket
def num_accidents := 1
def new_premium := 70 -- dollars per month

-- Define the target question
theorem number_of_tickets (tickets : ℕ) :
  initial_premium + premium_increase_accident initial_premium * num_accidents + premium_increase_ticket * tickets = new_premium → 
  tickets = 3 :=
by
   sorry

end number_of_tickets_l43_43195


namespace negation_of_proposition_l43_43607

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, 0 ≤ x ∧ (x^2 - 2*x - 3 = 0)) ↔ (∀ x : ℝ, 0 ≤ x → (x^2 - 2*x - 3 ≠ 0)) := 
by 
  sorry

end negation_of_proposition_l43_43607


namespace find_p_l43_43630

-- Define the coordinates as given in the problem
def Q : ℝ × ℝ := (0, 15)
def A : ℝ × ℝ := (3, 15)
def B : ℝ × ℝ := (15, 0)
def O : ℝ × ℝ := (0, 0)
def C (p : ℝ) : ℝ × ℝ := (0, p)

-- Defining the function to calculate area of triangle given three points
def area_of_triangle (P1 P2 P3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (P1.fst * (P2.snd - P3.snd) + P2.fst * (P3.snd - P1.snd) + P3.fst * (P1.snd - P2.snd))

-- The statement we need to prove
theorem find_p :
  ∃ p : ℝ, area_of_triangle A B (C p) = 42 ∧ p = 11.75 :=
by
  sorry

end find_p_l43_43630


namespace connor_sleep_duration_l43_43514

variables {Connor_sleep Luke_sleep Puppy_sleep : ℕ}

def sleeps_two_hours_longer (Luke_sleep Connor_sleep : ℕ) : Prop :=
  Luke_sleep = Connor_sleep + 2

def sleeps_twice_as_long (Puppy_sleep Luke_sleep : ℕ) : Prop :=
  Puppy_sleep = 2 * Luke_sleep

def sleeps_sixteen_hours (Puppy_sleep : ℕ) : Prop :=
  Puppy_sleep = 16

theorem connor_sleep_duration 
  (h1 : sleeps_two_hours_longer Luke_sleep Connor_sleep)
  (h2 : sleeps_twice_as_long Puppy_sleep Luke_sleep)
  (h3 : sleeps_sixteen_hours Puppy_sleep) :
  Connor_sleep = 6 :=
by {
  sorry
}

end connor_sleep_duration_l43_43514


namespace Vikas_submitted_6_questions_l43_43796

theorem Vikas_submitted_6_questions (R V A : ℕ) (h1 : 7 * V = 3 * R) (h2 : 2 * V = 3 * A) (h3 : R + V + A = 24) : V = 6 :=
by
  sorry

end Vikas_submitted_6_questions_l43_43796


namespace systematic_sampling_l43_43164

theorem systematic_sampling (E P: ℕ) (a b: ℕ) (g: ℕ) 
  (hE: E = 840)
  (hP: P = 42)
  (ha: a = 61)
  (hb: b = 140)
  (hg: g = E / P)
  (hEpos: 0 < E)
  (hPpos: 0 < P)
  (hgpos: 0 < g):
  (b - a + 1) / g = 4 := 
by
  sorry

end systematic_sampling_l43_43164


namespace marcella_shoes_l43_43621

theorem marcella_shoes :
  ∀ (original_pairs lost_shoes : ℕ), original_pairs = 27 → lost_shoes = 9 → 
  ∃ (remaining_pairs : ℕ), remaining_pairs = 18 ∧ remaining_pairs ≤ original_pairs - lost_shoes / 2 :=
by
  intros original_pairs lost_shoes h1 h2
  use 18
  constructor
  . exact rfl
  . sorry

end marcella_shoes_l43_43621


namespace sufficient_but_not_necessary_condition_not_neccessary_condition_l43_43860

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  ((x + 3)^2 + (y - 4)^2 = 0) → ((x + 3) * (y - 4) = 0) :=
by { sorry }

theorem not_neccessary_condition (x y : ℝ) :
  ((x + 3) * (y - 4) = 0) ↔ ((x + 3)^2 + (y - 4)^2 = 0) :=
by { sorry }

end sufficient_but_not_necessary_condition_not_neccessary_condition_l43_43860


namespace lucas_purchase_l43_43186

-- Define the variables and assumptions.
variables (a b c : ℕ)
variables (h1 : a + b + c = 50) (h2 : 50 * a + 400 * b + 500 * c = 10000)

-- Goal: Prove that the number of 50-cent items (a) is 30.
theorem lucas_purchase : a = 30 :=
by sorry

end lucas_purchase_l43_43186


namespace travel_time_second_bus_l43_43715

def distance_AB : ℝ := 100 -- kilometers
def passengers_first : ℕ := 20
def speed_first : ℝ := 60 -- kilometers per hour
def breakdown_time : ℝ := 0.5 -- hours
def passengers_second_initial : ℕ := 22
def speed_second_initial : ℝ := 50 -- kilometers per hour
def additional_passengers_speed_decrease : ℝ := 1 -- speed decrease for every additional 2 passengers
def passenger_factor : ℝ := 2
def additional_passengers : ℕ := 20
def total_time_second_bus : ℝ := 2.35 -- hours

theorem travel_time_second_bus :
  let distance_first_half := (breakdown_time * speed_first)
  let remaining_distance := distance_AB - distance_first_half
  let time_to_reach_breakdown := distance_first_half / speed_second_initial
  let new_speed_second_bus := speed_second_initial - (additional_passengers / passenger_factor) * additional_passengers_speed_decrease
  let time_from_breakdown_to_B := remaining_distance / new_speed_second_bus
  total_time_second_bus = time_to_reach_breakdown + time_from_breakdown_to_B := 
sorry

end travel_time_second_bus_l43_43715


namespace simplify_expression_l43_43356

theorem simplify_expression (x y m : ℤ) 
  (h1 : (x-5)^2 = -|m-1|)
  (h2 : y + 1 = 5) :
  (2 * x^2 - 3 * x * y - 4 * y^2) - m * (3 * x^2 - x * y + 9 * y^2) = -273 :=
sorry

end simplify_expression_l43_43356


namespace solve_rational_equation_l43_43572

theorem solve_rational_equation : 
  ∀ x : ℝ, x ≠ 1 -> (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) → 
  (x = 6 ∨ x = -2) :=
by
  intro x
  intro h
  intro h_eq
  sorry

end solve_rational_equation_l43_43572


namespace factor_expression_l43_43788

variable (x y : ℝ)

theorem factor_expression :
(3*x^3 + 28*(x^2)*y + 4*x) - (-4*x^3 + 5*(x^2)*y - 4*x) = x*(x + 8)*(7*x + 1) := sorry

end factor_expression_l43_43788


namespace simplify_sqrt1_simplify_sqrt2_find_a_l43_43048

-- Part 1
theorem simplify_sqrt1 : ∃ m n : ℝ, m^2 + n^2 = 6 ∧ m * n = Real.sqrt 5 ∧ Real.sqrt (6 + 2 * Real.sqrt 5) = m + n :=
by sorry

-- Part 2
theorem simplify_sqrt2 : ∃ m n : ℝ, m^2 + n^2 = 5 ∧ m * n = -Real.sqrt 6 ∧ Real.sqrt (5 - 2 * Real.sqrt 6) = abs (m - n) :=
by sorry

-- Part 3
theorem find_a (a : ℝ) : (Real.sqrt (a^2 + 4 * Real.sqrt 5) = 2 + Real.sqrt 5) → (a = 3 ∨ a = -3) :=
by sorry

end simplify_sqrt1_simplify_sqrt2_find_a_l43_43048


namespace shells_put_back_l43_43009

def shells_picked_up : ℝ := 324.0
def shells_left : ℝ := 32.0

theorem shells_put_back : shells_picked_up - shells_left = 292 := by
  sorry

end shells_put_back_l43_43009


namespace impossible_to_have_only_stacks_of_three_l43_43929

theorem impossible_to_have_only_stacks_of_three (n J : ℕ) (h_initial_n : n = 1) (h_initial_J : J = 1001) :
  (∀ n J, (n + J = 1002) → (∀ k : ℕ, 3 * k ≤ J → k + 3 * k ≠ 1002)) 
  :=
sorry

end impossible_to_have_only_stacks_of_three_l43_43929


namespace exp_ineq_solution_set_l43_43073

theorem exp_ineq_solution_set (e : ℝ) (h : e = Real.exp 1) :
  {x : ℝ | e^(2*x - 1) < 1} = {x : ℝ | x < 1 / 2} :=
sorry

end exp_ineq_solution_set_l43_43073


namespace solution_set_inequality_f_solution_range_a_l43_43689

-- Define the function f 
def f (x : ℝ) := |x + 1| + |x - 3|

-- Statement for question 1
theorem solution_set_inequality_f (x : ℝ) : f x < 6 ↔ -2 < x ∧ x < 4 :=
sorry

-- Statement for question 2
theorem solution_range_a (a : ℝ) (h : ∃ x : ℝ, f x = |a - 2|) : a ≥ 6 ∨ a ≤ -2 :=
sorry

end solution_set_inequality_f_solution_range_a_l43_43689


namespace system_of_equations_solution_l43_43210

theorem system_of_equations_solution (x y z : ℝ) 
  (h1 : x + 2 * y = 4)
  (h2 : 2 * x + 5 * y - 2 * z = 11)
  (h3 : 3 * x - 5 * y + 2 * z = -1) : 
  x = 2 ∧ y = 1 ∧ z = -1 :=
by {
  sorry
}

end system_of_equations_solution_l43_43210


namespace g_two_gt_one_third_g_n_gt_one_third_l43_43222

def seq_a (n : ℕ) : ℕ := 3 * n - 2
noncomputable def f (n : ℕ) : ℝ := (Finset.range n).sum (λ i => 1 / (seq_a (i + 1) : ℝ))
noncomputable def g (n : ℕ) : ℝ := f (n^2) - f (n - 1)

theorem g_two_gt_one_third : g 2 > 1 / 3 :=
sorry

theorem g_n_gt_one_third (n : ℕ) (h : n ≥ 3) : g n > 1 / 3 :=
sorry

end g_two_gt_one_third_g_n_gt_one_third_l43_43222


namespace salt_solution_l43_43461

variable (x : ℝ) (v_water : ℝ) (c_initial : ℝ) (c_final : ℝ)

theorem salt_solution (h1 : v_water = 1) (h2 : c_initial = 0.60) (h3 : c_final = 0.20)
  (h4 : (v_water + x) * c_final = x * c_initial) :
  x = 0.5 :=
by {
  sorry
}

end salt_solution_l43_43461


namespace locus_equation_of_points_at_distance_2_from_line_l43_43745

theorem locus_equation_of_points_at_distance_2_from_line :
  {P : ℝ × ℝ | abs ((3 / 5) * P.1 - (4 / 5) * P.2 - (1 / 5)) = 2} =
    {P : ℝ × ℝ | 3 * P.1 - 4 * P.2 - 11 = 0} ∪ {P : ℝ × ℝ | 3 * P.1 - 4 * P.2 + 9 = 0} :=
by
  -- Proof goes here
  sorry

end locus_equation_of_points_at_distance_2_from_line_l43_43745


namespace eggs_per_hen_l43_43859

theorem eggs_per_hen (total_eggs : Float) (num_hens : Float) (h1 : total_eggs = 303.0) (h2 : num_hens = 28.0) : 
  total_eggs / num_hens = 10.821428571428571 :=
by 
  sorry

end eggs_per_hen_l43_43859


namespace weight_ratios_l43_43029

theorem weight_ratios {x y z k : ℝ} (h1 : x + y = k * z) (h2 : y + z = k * x) (h3 : z + x = k * y) : x = y ∧ y = z :=
by 
  -- Proof to be filled in later
  sorry

end weight_ratios_l43_43029


namespace gcd_150_450_l43_43814

theorem gcd_150_450 : Nat.gcd 150 450 = 150 := by
  sorry

end gcd_150_450_l43_43814


namespace pipe_cistern_l43_43716

theorem pipe_cistern (rate: ℚ) (duration: ℚ) (portion: ℚ) : 
  rate = (2/3) / 10 → duration = 8 → portion = 8/15 →
  portion = duration * rate := 
by 
  intros h1 h2 h3
  sorry

end pipe_cistern_l43_43716


namespace candy_cost_l43_43905

theorem candy_cost
  (C : ℝ) -- cost per pound of the first candy
  (w1 : ℝ := 30) -- weight of the first candy
  (c2 : ℝ := 5) -- cost per pound of the second candy
  (w2 : ℝ := 60) -- weight of the second candy
  (w_mix : ℝ := 90) -- total weight of the mixture
  (c_mix : ℝ := 6) -- desired cost per pound of the mixture
  (h1 : w1 * C + w2 * c2 = w_mix * c_mix) -- cost equation for the mixture
  : C = 8 :=
by
  sorry

end candy_cost_l43_43905


namespace domain_not_neg1_increasing_on_neg1_infty_min_max_on_3_5_l43_43652

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

namespace f_props

theorem domain_not_neg1 : ∀ x : ℝ, x ≠ -1 ↔ x ∈ {y | y ≠ -1} :=
by simp [f]

theorem increasing_on_neg1_infty : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → -1 < x2 → f x1 < f x2 :=
sorry

theorem min_max_on_3_5 : (∀ y : ℝ, y = f 3 → y = 5 / 4) ∧ (∀ y : ℝ, y = f 5 → y = 3 / 2) :=
sorry

end f_props

end domain_not_neg1_increasing_on_neg1_infty_min_max_on_3_5_l43_43652


namespace total_expenditure_eq_fourteen_l43_43601

variable (cost_barrette cost_comb : ℕ)
variable (kristine_barrettes kristine_combs crystal_barrettes crystal_combs : ℕ)

theorem total_expenditure_eq_fourteen 
  (h_cost_barrette : cost_barrette = 3)
  (h_cost_comb : cost_comb = 1)
  (h_kristine_barrettes : kristine_barrettes = 1)
  (h_kristine_combs : kristine_combs = 1)
  (h_crystal_barrettes : crystal_barrettes = 3)
  (h_crystal_combs : crystal_combs = 1) :
  (kristine_barrettes * cost_barrette + kristine_combs * cost_comb) +
  (crystal_barrettes * cost_barrette + crystal_combs * cost_comb) = 14 := 
by 
  sorry

end total_expenditure_eq_fourteen_l43_43601


namespace instantaneous_velocity_at_t2_l43_43857

def displacement (t : ℝ) : ℝ := 2 * (1 - t) ^ 2

theorem instantaneous_velocity_at_t2 :
  (deriv displacement 2) = 4 :=
by
  sorry

end instantaneous_velocity_at_t2_l43_43857


namespace doubled_team_completes_half_in_three_days_l43_43918

theorem doubled_team_completes_half_in_three_days
  (R : ℝ) -- Combined work rate of the original team
  (h : R * 12 = W) -- Original team completes the work W in 12 days
  (W : ℝ) : -- Total work to be done
  (2 * R) * 3 = W/2 := -- Doubled team completes half the work in 3 days
by 
  sorry

end doubled_team_completes_half_in_three_days_l43_43918


namespace tank_filled_in_96_minutes_l43_43040

-- conditions
def pipeA_fill_time : ℝ := 6
def pipeB_empty_time : ℝ := 24
def time_with_both_pipes_open : ℝ := 96

-- rate computations and final proof
noncomputable def pipeA_fill_rate : ℝ := 1 / pipeA_fill_time
noncomputable def pipeB_empty_rate : ℝ := 1 / pipeB_empty_time
noncomputable def net_fill_rate : ℝ := pipeA_fill_rate - pipeB_empty_rate
noncomputable def tank_filled_in_time_with_both : ℝ := time_with_both_pipes_open * net_fill_rate

theorem tank_filled_in_96_minutes (HA : pipeA_fill_time = 6) (HB : pipeB_empty_time = 24)
  (HT : time_with_both_pipes_open = 96) : tank_filled_in_time_with_both = 1 :=
by
  sorry

end tank_filled_in_96_minutes_l43_43040


namespace no_solution_exists_l43_43180

theorem no_solution_exists :
  ∀ a b : ℕ, a - b = 5 ∨ b - a = 5 → a * b = 132 → false :=
by
  sorry

end no_solution_exists_l43_43180


namespace correct_total_cost_l43_43873

noncomputable def total_cost_after_discount : ℝ :=
  let sandwich_cost := 4
  let soda_cost := 3
  let sandwich_count := 7
  let soda_count := 5
  let total_items := sandwich_count + soda_count
  let total_cost := sandwich_count * sandwich_cost + soda_count * soda_cost
  let discount := if total_items ≥ 10 then 0.1 * total_cost else 0
  total_cost - discount

theorem correct_total_cost :
  total_cost_after_discount = 38.7 :=
by
  -- The proof would go here
  sorry

end correct_total_cost_l43_43873


namespace weighted_avg_surfers_per_day_l43_43812

theorem weighted_avg_surfers_per_day 
  (total_surfers : ℕ) 
  (ratio1_day1 ratio1_day2 ratio2_day3 ratio2_day4 : ℕ) 
  (h_total_surfers : total_surfers = 12000)
  (h_ratio_first_two_days : ratio1_day1 = 5 ∧ ratio1_day2 = 7)
  (h_ratio_last_two_days : ratio2_day3 = 3 ∧ ratio2_day4 = 2) 
  : (total_surfers / (ratio1_day1 + ratio1_day2 + ratio2_day3 + ratio2_day4)) * 
    ((ratio1_day1 + ratio1_day2 + ratio2_day3 + ratio2_day4) / 4) = 3000 :=
by
  sorry

end weighted_avg_surfers_per_day_l43_43812


namespace unique_solution_otimes_l43_43363

def otimes (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem unique_solution_otimes : 
  (∃! y : ℝ, otimes 2 y = 20) := 
by
  sorry

end unique_solution_otimes_l43_43363


namespace correct_statement_l43_43441

theorem correct_statement : -3 > -5 := 
by {
  sorry
}

end correct_statement_l43_43441


namespace coloring_satisfies_conditions_l43_43031

/-- Define what it means for a point to be a lattice point -/
def is_lattice_point (x y : ℤ) : Prop := true

/-- Define the coloring function based on coordinates -/
def color (x y : ℤ) : Prop :=
  (x % 2 = 1 ∧ y % 2 = 1) ∨   -- white
  (x % 2 = 1 ∧ y % 2 = 0) ∨   -- black
  (x % 2 = 0)                 -- red (both (even even) and (even odd) are included)

/-- Proving the method of coloring lattice points satisfies the given conditions -/
theorem coloring_satisfies_conditions :
  (∀ x y : ℤ, is_lattice_point x y → 
    color x y ∧ 
    ∃ (A B C : ℤ × ℤ), 
      (is_lattice_point A.fst A.snd ∧ 
       is_lattice_point B.fst B.snd ∧ 
       is_lattice_point C.fst C.snd ∧ 
       color A.fst A.snd ∧ 
       color B.fst B.snd ∧ 
       color C.fst C.snd ∧
       ∃ D : ℤ × ℤ, 
         (is_lattice_point D.fst D.snd ∧ 
          color D.fst D.snd ∧ 
          D.fst = A.fst + C.fst - B.fst ∧ 
          D.snd = A.snd + C.snd - B.snd))) :=
sorry

end coloring_satisfies_conditions_l43_43031


namespace polynomial_not_factorable_l43_43506

theorem polynomial_not_factorable :
  ¬ ∃ (A B : Polynomial ℤ), A.degree < 5 ∧ B.degree < 5 ∧ A * B = (Polynomial.C 1 * Polynomial.X ^ 5 - Polynomial.C 3 * Polynomial.X ^ 4 + Polynomial.C 6 * Polynomial.X ^ 3 - Polynomial.C 3 * Polynomial.X ^ 2 + Polynomial.C 9 * Polynomial.X - Polynomial.C 6) :=
by
  sorry

end polynomial_not_factorable_l43_43506


namespace op_example_l43_43866

def myOp (c d : Int) : Int :=
  c * (d + 1) + c * d

theorem op_example : myOp 5 (-2) = -15 := 
  by
    sorry

end op_example_l43_43866


namespace bill_before_tax_l43_43553

theorem bill_before_tax (T E : ℝ) (h1 : E = 2) (h2 : 3 * T + 5 * E = 12.70) : 2 * T + 3 * E = 7.80 :=
by
  sorry

end bill_before_tax_l43_43553


namespace find_a_l43_43017

variable (a : ℝ) (h_pos : a > 0) (h_integral : ∫ x in 0..a, (2 * x - 2) = 3)

theorem find_a : a = 3 :=
by sorry

end find_a_l43_43017


namespace domain_of_f_l43_43108

open Real

noncomputable def f (x : ℝ) : ℝ := log (log x)

theorem domain_of_f : { x : ℝ | 1 < x } = { x : ℝ | ∃ y > 1, x = y } :=
by
  sorry

end domain_of_f_l43_43108


namespace movie_tickets_ratio_l43_43310

theorem movie_tickets_ratio (R H : ℕ) (hR : R = 25) (hH : H = 93) : 
  (H / R : ℚ) = 93 / 25 :=
by
  sorry

end movie_tickets_ratio_l43_43310


namespace nonnegative_integer_pairs_solution_l43_43936

open Int

theorem nonnegative_integer_pairs_solution (x y : ℕ) : 
  3 * x ^ 2 + 2 * 9 ^ y = x * (4 ^ (y + 1) - 1) ↔ (x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) :=
by 
  sorry

end nonnegative_integer_pairs_solution_l43_43936


namespace no_snow_probability_l43_43695

noncomputable def probability_of_no_snow (p_snow : ℚ) : ℚ :=
  1 - p_snow

theorem no_snow_probability : probability_of_no_snow (2/5) = 3/5 :=
  sorry

end no_snow_probability_l43_43695


namespace even_function_expression_l43_43637

theorem even_function_expression (f : ℝ → ℝ)
  (h₀ : ∀ x, x ≥ 0 → f x = x^2 - 3 * x + 4)
  (h_even : ∀ x, f x = f (-x)) :
  ∀ x, f x = if x < 0 then x^2 + 3 * x + 4 else x^2 - 3 * x + 4 :=
by {
  sorry
}

end even_function_expression_l43_43637


namespace sum_of_reciprocals_eq_six_l43_43962

theorem sum_of_reciprocals_eq_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + y = 6 * x * y) (h2 : y = 2 * x) :
  (1 / x) + (1 / y) = 6 := by
  sorry

end sum_of_reciprocals_eq_six_l43_43962


namespace option_C_qualified_l43_43288

-- Define the acceptable range
def lower_bound : ℝ := 25 - 0.2
def upper_bound : ℝ := 25 + 0.2

-- Define the option to be checked
def option_C : ℝ := 25.1

-- The theorem stating that option C is within the acceptable range
theorem option_C_qualified : lower_bound ≤ option_C ∧ option_C ≤ upper_bound := 
by 
  sorry

end option_C_qualified_l43_43288


namespace cylinder_area_ratio_l43_43748

noncomputable def ratio_of_areas (r h : ℝ) (h_cond : 2 * r / h = h / (2 * Real.pi * r)) : ℝ :=
  let lateral_area := 2 * Real.pi * r * h
  let total_area := lateral_area + 2 * Real.pi * r * r
  lateral_area / total_area

theorem cylinder_area_ratio {r h : ℝ} (h_cond : 2 * r / h = h / (2 * Real.pi * r)) :
  ratio_of_areas r h h_cond = 2 * Real.sqrt Real.pi / (2 * Real.sqrt Real.pi + 1) := 
sorry

end cylinder_area_ratio_l43_43748


namespace find_x_values_l43_43891

theorem find_x_values (x : ℝ) :
  x^3 - 9 * x^2 + 27 * x > 0 ↔ (0 < x ∧ x < 3) ∨ (6 < x) :=
by
  sorry

end find_x_values_l43_43891


namespace production_days_l43_43967

theorem production_days (n : ℕ) (P : ℕ)
  (h1 : P = 40 * n)
  (h2 : (P + 90) / (n + 1) = 45) :
  n = 9 :=
by
  sorry

end production_days_l43_43967


namespace proposition_3_proposition_4_l43_43705

variables {Plane : Type} {Line : Type} 
variables {α β : Plane} {a b : Line}

-- Assuming necessary properties of parallel planes and lines being subsets of planes
axiom plane_parallel (α β : Plane) : Prop
axiom line_in_plane (l : Line) (p : Plane) : Prop
axiom line_parallel (l m : Line) : Prop
axiom lines_skew (l m : Line) : Prop
axiom lines_coplanar (l m : Line) : Prop
axiom lines_do_not_intersect (l m : Line) : Prop

-- Assume the given conditions
variables (h1 : plane_parallel α β) 
variables (h2 : line_in_plane a α)
variables (h3 : line_in_plane b β)

-- State the equivalent proof problem as propositions to be proved in Lean
theorem proposition_3 (h1 : plane_parallel α β) 
                     (h2 : line_in_plane a α) 
                     (h3 : line_in_plane b β) : 
                     lines_do_not_intersect a b :=
sorry

theorem proposition_4 (h1 : plane_parallel α β) 
                     (h2 : line_in_plane a α) 
                     (h3 : line_in_plane b β) : 
                     lines_coplanar a b ∨ lines_skew a b :=
sorry

end proposition_3_proposition_4_l43_43705


namespace find_m_eccentricity_l43_43611

theorem find_m_eccentricity :
  (∃ m : ℝ, (m > 0) ∧ (∃ c : ℝ, (c = 4 - m ∧ c = (1 / 2) * 2) ∨ (c = m - 4 ∧ c = (1 / 2) * 2)) ∧
  (m = 3 ∨ m = 16 / 3)) :=
sorry

end find_m_eccentricity_l43_43611


namespace B_share_is_102_l43_43170

variables (A B C : ℝ)
variables (total : ℝ)
variables (rA_B : ℝ) (rB_C : ℝ)

-- Conditions
def conditions : Prop :=
  (total = 578) ∧
  (rA_B = 2 / 3) ∧
  (rB_C = 1 / 4) ∧
  (A = rA_B * B) ∧
  (B = rB_C * C) ∧
  (A + B + C = total)

-- Theorem to prove B's share
theorem B_share_is_102 (h : conditions A B C total rA_B rB_C) : B = 102 :=
by sorry

end B_share_is_102_l43_43170


namespace solve_problem_l43_43590

-- Define the constants c and d
variables (c d : ℝ)

-- Define the conditions of the problem
def condition1 : Prop := 
  (∀ x : ℝ, (x + c) * (x + d) * (x + 15) = 0 ↔ x = -c ∨ x = -d ∨ x = -15) ∧
  -4 ≠ -c ∧ -4 ≠ -d ∧ -4 ≠ -15

def condition2 : Prop := 
  (∀ x : ℝ, (x + 3 * c) * (x + 4) * (x + 9) = 0 ↔ x = -4) ∧
  d ≠ -4 ∧ d ≠ -15

-- We need to prove this final result under the given conditions
theorem solve_problem (h1 : condition1 c d) (h2 : condition2 c d) : 100 * c + d = -291 := 
  sorry

end solve_problem_l43_43590


namespace ellipse_foci_y_axis_range_l43_43574

theorem ellipse_foci_y_axis_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 / (|m| - 1) + y^2 / (2 - m) = 1) ↔ (m < -1 ∨ (1 < m ∧ m < 3 / 2)) :=
sorry

end ellipse_foci_y_axis_range_l43_43574


namespace max_intersections_circle_pentagon_l43_43407

theorem max_intersections_circle_pentagon : 
  ∃ (circle : Set Point) (pentagon : List (Set Point)),
    (∀ (side : Set Point), side ∈ pentagon → ∃ p1 p2 : Point, p1 ∈ circle ∧ p2 ∈ circle ∧ p1 ≠ p2) ∧
    pentagon.length = 5 →
    (∃ n : ℕ, n = 10) :=
by
  sorry

end max_intersections_circle_pentagon_l43_43407


namespace sin_identity_cos_identity_l43_43026

-- Define the condition that alpha + beta + gamma = 180 degrees.
def angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi

-- Prove that sin 4α + sin 4β + sin 4γ = -4 sin 2α sin 2β sin 2γ.
theorem sin_identity (α β γ : ℝ) (h : angles_sum_to_180 α β γ) :
  Real.sin (4 * α) + Real.sin (4 * β) + Real.sin (4 * γ) = -4 * Real.sin (2 * α) * Real.sin (2 * β) * Real.sin (2 * γ) := by
  sorry

-- Prove that cos 4α + cos 4β + cos 4γ = 4 cos 2α cos 2β cos 2γ - 1.
theorem cos_identity (α β γ : ℝ) (h : angles_sum_to_180 α β γ) :
  Real.cos (4 * α) + Real.cos (4 * β) + Real.cos (4 * γ) = 4 * Real.cos (2 * α) * Real.cos (2 * β) * Real.cos (2 * γ) - 1 := by
  sorry

end sin_identity_cos_identity_l43_43026


namespace smallest_n_mod_l43_43003

theorem smallest_n_mod : ∃ n : ℕ, 5 * n ≡ 2024 [MOD 26] ∧ n > 0 ∧ ∀ m : ℕ, (5 * m ≡ 2024 [MOD 26] ∧ m > 0) → n ≤ m :=
  sorry

end smallest_n_mod_l43_43003


namespace sum_mod_11_l43_43772

theorem sum_mod_11 (h1 : 8735 % 11 = 1) (h2 : 8736 % 11 = 2) (h3 : 8737 % 11 = 3) (h4 : 8738 % 11 = 4) :
  (8735 + 8736 + 8737 + 8738) % 11 = 10 :=
by
  sorry

end sum_mod_11_l43_43772


namespace largest_sum_of_base8_digits_l43_43106

theorem largest_sum_of_base8_digits (a b c y : ℕ) (h1 : a < 8) (h2 : b < 8) (h3 : c < 8) (h4 : 0 < y ∧ y ≤ 16) (h5 : (a * 64 + b * 8 + c) * y = 512) :
  a + b + c ≤ 5 :=
sorry

end largest_sum_of_base8_digits_l43_43106


namespace polynomial_coeffs_l43_43058

theorem polynomial_coeffs (a b c d e f : ℤ) :
  (((2 : ℤ) * x - 1) ^ 5 = a * x ^ 5 + b * x ^ 4 + c * x ^ 3 + d * x ^ 2 + e * x + f) →
  (a + b + c + d + e + f = 1) ∧ 
  (b + c + d + e = -30) ∧
  (a + c + e = 122) :=
by
  intro h
  sorry  -- Proof omitted

end polynomial_coeffs_l43_43058


namespace proof_problem_l43_43551

theorem proof_problem (x : ℝ) (h : x < 1) : -2 * x + 2 > 0 :=
by
  sorry

end proof_problem_l43_43551


namespace sum_of_possible_values_of_x_l43_43377

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l43_43377


namespace lower_tap_used_earlier_l43_43693

-- Define the conditions given in the problem
def capacity : ℕ := 36
def midway_capacity : ℕ := capacity / 2
def lower_tap_rate : ℕ := 4  -- minutes per litre
def upper_tap_rate : ℕ := 6  -- minutes per litre

def lower_tap_draw (minutes : ℕ) : ℕ := minutes / lower_tap_rate  -- litres drawn by lower tap
def beer_left_after_draw (initial_amount litres_drawn : ℕ) : ℕ := initial_amount - litres_drawn

-- Define the assistant's drawing condition
def assistant_draw_min : ℕ := 16
def assistant_draw_litres : ℕ := lower_tap_draw assistant_draw_min

-- Define proof statement
theorem lower_tap_used_earlier :
  let initial_amount := capacity
  let litres_when_midway := midway_capacity
  let litres_beer_left := beer_left_after_draw initial_amount assistant_draw_litres
  let additional_litres := litres_beer_left - litres_when_midway
  let time_earlier := additional_litres * upper_tap_rate
  time_earlier = 84 := 
by
  sorry

end lower_tap_used_earlier_l43_43693


namespace factorize_polynomial_l43_43766
   
   -- Define the polynomial
   def polynomial (x : ℝ) : ℝ :=
     x^3 + 3 * x^2 - 4
   
   -- Define the factorized form
   def factorized_form (x : ℝ) : ℝ :=
     (x - 1) * (x + 2)^2
   
   -- The theorem statement
   theorem factorize_polynomial (x : ℝ) : polynomial x = factorized_form x := 
   by
     sorry
   
end factorize_polynomial_l43_43766


namespace find_m_and_y_range_l43_43397

open Set

noncomputable def y (m x : ℝ) := (6 + 2 * m) * x^2 - 5 * x^((abs (m + 2))) + 3 

theorem find_m_and_y_range :
  (∃ m : ℝ, (∀ x : ℝ, y m x = (6 + 2*m) * x^2 - 5*x^((abs (m+2))) + 3) ∧ (∀ x : ℝ, y m x = -5 * x + 3 → m = -3)) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 → y (-3) x ∈ Icc (-22 : ℝ) (8 : ℝ)) :=
by
  sorry

end find_m_and_y_range_l43_43397


namespace Haley_initial_trees_l43_43434

theorem Haley_initial_trees (T : ℕ) (h1 : T - 4 ≥ 0) (h2 : (T - 4) + 5 = 10): T = 9 :=
by
  -- proof goes here
  sorry

end Haley_initial_trees_l43_43434


namespace CorrectChoice_l43_43679

open Classical

-- Define the integer n
variable (n : ℤ)

-- Define proposition p: 2n - 1 is always odd
def p : Prop := ∃ k : ℤ, 2 * k + 1 = 2 * n - 1

-- Define proposition q: 2n + 1 is always even
def q : Prop := ∃ k : ℤ, 2 * k = 2 * n + 1

-- The theorem we want to prove
theorem CorrectChoice : (p n ∨ q n) :=
by
  sorry

end CorrectChoice_l43_43679


namespace smallest_x_l43_43661

noncomputable def f (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x ≤ 4 then x^2 - 4 * x + 5 else sorry

theorem smallest_x (x : ℝ) (h₁ : ∀ x > 0, f (4 * x) = 4 * f x)
  (h₂ : ∀ x, (1 ≤ x ∧ x ≤ 4) → f x = x^2 - 4 * x + 5) :
  ∃ x₀, x₀ > 0 ∧ f x₀ = 1024 ∧ (∀ y, y > 0 ∧ f y = 1024 → y ≥ x₀) :=
sorry

end smallest_x_l43_43661


namespace polygon_sides_arithmetic_progression_l43_43371

theorem polygon_sides_arithmetic_progression 
  (n : ℕ) 
  (h1 : ∀ n, ∃ a_1, ∃ a_n, ∀ i, a_n = 172 ∧ (a_i = a_1 + (i - 1) * 4) ∧ (i ≤ n))
  (h2 : ∀ S, S = 180 * (n - 2)) 
  (h3 : ∀ S, S = n * ((172 - 4 * (n - 1) + 172) / 2)) 
  : n = 12 := 
by 
  sorry

end polygon_sides_arithmetic_progression_l43_43371


namespace sum_of_four_consecutive_integers_divisible_by_two_l43_43959

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n-1) + n + (n+1) + (n+2)) :=
by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l43_43959


namespace tree_sidewalk_space_l43_43308

theorem tree_sidewalk_space
  (num_trees : ℕ)
  (distance_between_trees : ℝ)
  (total_road_length : ℝ)
  (total_gaps : ℝ)
  (space_each_tree : ℝ)
  (H1 : num_trees = 11)
  (H2 : distance_between_trees = 14)
  (H3 : total_road_length = 151)
  (H4 : total_gaps = (num_trees - 1) * distance_between_trees)
  (H5 : space_each_tree = (total_road_length - total_gaps) / num_trees)
  : space_each_tree = 1 := 
by
  sorry

end tree_sidewalk_space_l43_43308


namespace n_value_condition_l43_43037

theorem n_value_condition (n : ℤ) : 
  (3 * (n ^ 2 + n) + 7) % 5 = 0 ↔ n % 5 = 2 := sorry

end n_value_condition_l43_43037


namespace sqrt3_minus1_plus_inv3_pow_minus2_l43_43898

theorem sqrt3_minus1_plus_inv3_pow_minus2 :
  (Real.sqrt 3 - 1) + (1 / (1/3) ^ 2) = Real.sqrt 3 + 8 :=
by
  sorry

end sqrt3_minus1_plus_inv3_pow_minus2_l43_43898


namespace range_of_a_for_decreasing_f_l43_43167

theorem range_of_a_for_decreasing_f :
  (∀ x : ℝ, (-3) * x^2 + 2 * a * x - 1 ≤ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by
  -- The proof goes here
  sorry

end range_of_a_for_decreasing_f_l43_43167


namespace roots_quadratic_eq_sum_prod_l43_43535

theorem roots_quadratic_eq_sum_prod (r s p q : ℝ) (hr : r + s = p) (hq : r * s = q) : r^2 + s^2 = p^2 - 2 * q :=
by
  sorry

end roots_quadratic_eq_sum_prod_l43_43535


namespace num_false_statements_is_three_l43_43038

-- Definitions of the statements on the card
def s1 : Prop := ∀ (false_statements : ℕ), false_statements = 1
def s2 : Prop := ∀ (false_statements_card1 false_statements_card2 : ℕ), false_statements_card1 + false_statements_card2 = 2
def s3 : Prop := ∀ (false_statements : ℕ), false_statements = 3
def s4 : Prop := ∀ (false_statements_card1 false_statements_card2 : ℕ), false_statements_card1 = false_statements_card2

-- Main proof problem: The number of false statements on this card is 3
theorem num_false_statements_is_three 
  (h_s1 : ¬ s1)
  (h_s2 : ¬ s2)
  (h_s3 : s3)
  (h_s4 : ¬ s4) :
  ∃ (n : ℕ), n = 3 :=
by
  sorry

end num_false_statements_is_three_l43_43038


namespace andy_time_correct_l43_43342

-- Define the conditions
def time_dawn_wash_dishes : ℕ := 20
def time_andy_put_laundry : ℕ := 2 * time_dawn_wash_dishes + 6

-- The theorem to prove
theorem andy_time_correct : time_andy_put_laundry = 46 :=
by
  -- Proof goes here
  sorry

end andy_time_correct_l43_43342


namespace car_speed_l43_43413

theorem car_speed (distance: ℚ) (hours minutes: ℚ) (h_distance: distance = 360) (h_hours: hours = 4) (h_minutes: minutes = 30) : 
  (distance / (hours + (minutes / 60))) = 80 := by
  sorry

end car_speed_l43_43413


namespace f_20_equals_97_l43_43237

noncomputable def f_rec (f : ℕ → ℝ) (n : ℕ) := (2 * f n + n) / 2

theorem f_20_equals_97 (f : ℕ → ℝ) (h₁ : f 1 = 2)
    (h₂ : ∀ n : ℕ, f (n + 1) = f_rec f n) : 
    f 20 = 97 :=
sorry

end f_20_equals_97_l43_43237


namespace find_values_of_a_l43_43087

-- Definitions for sets A and B
def A : Set ℝ := {x | x^2 - x - 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 6 = 0}

-- The theorem we want to prove
theorem find_values_of_a (a : ℝ) : (A ∪ B a = A) ↔ (a = -6 ∨ a = 0 ∨ a = 3) :=
by
  sorry

end find_values_of_a_l43_43087


namespace average_speed_of_bike_l43_43561

theorem average_speed_of_bike (distance : ℕ) (time : ℕ) (h1 : distance = 21) (h2 : time = 7) : distance / time = 3 := by
  sorry

end average_speed_of_bike_l43_43561


namespace part1_part2_l43_43268

-- Definitions from conditions
def U := ℝ
def A := {x : ℝ | -x^2 + 12*x - 20 > 0}
def B (a : ℝ) := {x : ℝ | 5 - a < x ∧ x < a}

-- (1) If "x ∈ A" is a necessary condition for "x ∈ B", find the range of a
theorem part1 (a : ℝ) : (∀ x : ℝ, x ∈ B a → x ∈ A) → a ≤ 3 :=
by sorry

-- (2) If A ∩ B ≠ ∅, find the range of a
theorem part2 (a : ℝ) : (∃ x : ℝ, x ∈ A ∧ x ∈ B a) → a > 5 / 2 :=
by sorry

end part1_part2_l43_43268


namespace four_digit_number_perfect_square_l43_43775

theorem four_digit_number_perfect_square (abcd : ℕ) (h1 : abcd ≥ 1000 ∧ abcd < 10000) (h2 : ∃ k : ℕ, k^2 = 4000000 + abcd) :
  abcd = 4001 ∨ abcd = 8004 :=
sorry

end four_digit_number_perfect_square_l43_43775


namespace jerry_trips_l43_43225

-- Define the conditions
def trays_per_trip : Nat := 8
def trays_table1 : Nat := 9
def trays_table2 : Nat := 7

-- Define the proof problem
theorem jerry_trips :
  trays_table1 + trays_table2 = 16 →
  (16 / trays_per_trip) = 2 :=
by
  sorry

end jerry_trips_l43_43225


namespace bacteria_reach_target_l43_43249

def bacteria_growth (initial : ℕ) (target : ℕ) (doubling_time : ℕ) (delay : ℕ) : ℕ :=
  let doubling_count := Nat.log2 (target / initial)
  doubling_count * doubling_time + delay

theorem bacteria_reach_target : 
  bacteria_growth 800 25600 5 3 = 28 := by
  sorry

end bacteria_reach_target_l43_43249


namespace focus_of_parabola_l43_43768

theorem focus_of_parabola : (∃ p : ℝ × ℝ, p = (-1, 35/12)) :=
by
  sorry

end focus_of_parabola_l43_43768


namespace find_k_l43_43620

theorem find_k : 
  ∃ x y k : ℝ, y = 7 * x - 2 ∧ y = -3 * x + 14 ∧ y = 4 * x + k ∧ k = 2.8 :=
by
  sorry

end find_k_l43_43620


namespace julie_initial_savings_l43_43088

-- Definition of the simple interest condition
def simple_interest_condition (P : ℝ) : Prop :=
  575 = P * 0.04 * 5

-- Definition of the compound interest condition
def compound_interest_condition (P : ℝ) : Prop :=
  635 = P * ((1 + 0.05) ^ 5 - 1)

-- The final proof problem
theorem julie_initial_savings (P : ℝ) :
  simple_interest_condition P →
  compound_interest_condition P →
  2 * P = 5750 :=
by sorry

end julie_initial_savings_l43_43088


namespace borrowed_amount_correct_l43_43005

noncomputable def principal_amount (I: ℚ) (r1 r2 r3 r4 t1 t2 t3 t4: ℚ): ℚ :=
  I / (r1 * t1 + r2 * t2 + r3 * t3 + r4 * t4)

def interest_rate_1 := (6.5 / 100 : ℚ)
def interest_rate_2 := (9.5 / 100 : ℚ)
def interest_rate_3 := (11 / 100 : ℚ)
def interest_rate_4 := (14.5 / 100 : ℚ)

def time_period_1 := (2.5 : ℚ)
def time_period_2 := (3.75 : ℚ)
def time_period_3 := (1.5 : ℚ)
def time_period_4 := (4.25 : ℚ)

def total_interest := (14500 : ℚ)

def expected_principal := (11153.846153846154 : ℚ)

theorem borrowed_amount_correct :
  principal_amount total_interest interest_rate_1 interest_rate_2 interest_rate_3 interest_rate_4 time_period_1 time_period_2 time_period_3 time_period_4 = expected_principal :=
by
  sorry

end borrowed_amount_correct_l43_43005


namespace product_of_two_numbers_l43_43292

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y ≠ 0) 
  (h2 : (x + y) / (x - y) = 7)
  (h3 : xy = 24 * (x - y)) : xy = 48 := 
sorry

end product_of_two_numbers_l43_43292


namespace sue_nuts_count_l43_43328

theorem sue_nuts_count (B H S : ℕ) 
  (h1 : B = 6 * H) 
  (h2 : H = 2 * S) 
  (h3 : B + H = 672) : S = 48 := 
by
  sorry

end sue_nuts_count_l43_43328


namespace largest_value_l43_43696

noncomputable def a : ℕ := 2 ^ 6
noncomputable def b : ℕ := 3 ^ 5
noncomputable def c : ℕ := 4 ^ 4
noncomputable def d : ℕ := 5 ^ 3
noncomputable def e : ℕ := 6 ^ 2

theorem largest_value : c > a ∧ c > b ∧ c > d ∧ c > e := by
  sorry

end largest_value_l43_43696


namespace prove_zero_function_l43_43941

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ x y : ℝ, f (x ^ 333 + y) = f (x ^ 2018 + 2 * y) + f (x ^ 42)

theorem prove_zero_function : ∀ x : ℝ, f x = 0 :=
by
  sorry

end prove_zero_function_l43_43941


namespace probability_three_specific_cards_l43_43391

noncomputable def deck_size : ℕ := 52
noncomputable def num_suits : ℕ := 4
noncomputable def cards_per_suit : ℕ := 13
noncomputable def p_king_spades : ℚ := 1 / deck_size
noncomputable def p_10_hearts : ℚ := 1 / (deck_size - 1)
noncomputable def p_queen : ℚ := 4 / (deck_size - 2)

theorem probability_three_specific_cards :
  (p_king_spades * p_10_hearts * p_queen) = 1 / 33150 := 
sorry

end probability_three_specific_cards_l43_43391


namespace ice_cream_cones_sixth_day_l43_43315

theorem ice_cream_cones_sixth_day (cones_day1 cones_day2 cones_day3 cones_day4 cones_day5 cones_day7 : ℝ)
  (mean : ℝ) (h1 : cones_day1 = 100) (h2 : cones_day2 = 92) 
  (h3 : cones_day3 = 109) (h4 : cones_day4 = 96) 
  (h5 : cones_day5 = 103) (h7 : cones_day7 = 105) 
  (h_mean : mean = 100.1) : 
  ∃ cones_day6 : ℝ, cones_day6 = 95.7 :=
by 
  sorry

end ice_cream_cones_sixth_day_l43_43315


namespace num_ways_for_volunteers_l43_43713

theorem num_ways_for_volunteers:
  let pavilions := 4
  let volunteers := 5
  let ways_to_choose_A := 4
  let ways_to_choose_B_after_A := 3
  let total_distributions := 
    let case_1 := 2
    let case_2 := (2^3) - 2
    case_1 + case_2
  ways_to_choose_A * ways_to_choose_B_after_A * total_distributions = 72 := 
by
  sorry

end num_ways_for_volunteers_l43_43713


namespace frac_sum_equals_seven_eights_l43_43145

theorem frac_sum_equals_seven_eights (p q r u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 := 
  sorry

end frac_sum_equals_seven_eights_l43_43145


namespace total_bottles_in_box_l43_43957

def dozens (n : ℕ) := 12 * n

def water_bottles : ℕ := dozens 2

def apple_bottles : ℕ := water_bottles + 6

def total_bottles : ℕ := water_bottles + apple_bottles

theorem total_bottles_in_box : total_bottles = 54 := 
by
  sorry

end total_bottles_in_box_l43_43957


namespace jellybeans_left_in_jar_l43_43271

def original_jellybeans : ℕ := 250
def class_size : ℕ := 24
def sick_children : ℕ := 2
def sick_jellybeans_each : ℕ := 7
def first_group_size : ℕ := 12
def first_group_jellybeans_each : ℕ := 5
def second_group_size : ℕ := 10
def second_group_jellybeans_each : ℕ := 4

theorem jellybeans_left_in_jar : 
  original_jellybeans - ((first_group_size * first_group_jellybeans_each) + 
  (second_group_size * second_group_jellybeans_each)) = 150 := by
  sorry

end jellybeans_left_in_jar_l43_43271


namespace expression_value_l43_43403

theorem expression_value (a b : ℚ) (h₁ : a = -1/2) (h₂ : b = 3/2) : -a - 2 * b^2 + 3 * a * b = -25/4 :=
by
  sorry

end expression_value_l43_43403


namespace no_real_roots_equationD_l43_43998

def discriminant (a b c : ℕ) : ℤ := b^2 - 4 * a * c

def equationA := (1, -2, -4)
def equationB := (1, -4, 4)
def equationC := (1, -2, -5)
def equationD := (1, 3, 5)

theorem no_real_roots_equationD :
  discriminant (1 : ℕ) 3 5 < 0 :=
by
  show discriminant 1 3 5 < 0
  sorry

end no_real_roots_equationD_l43_43998


namespace Tom_completes_wall_l43_43161

theorem Tom_completes_wall :
  let avery_rate_per_hour := (1:ℝ)/3
  let tom_rate_per_hour := (1:ℝ)/2
  let combined_rate_per_hour := avery_rate_per_hour + tom_rate_per_hour
  let portion_completed_together := combined_rate_per_hour * 1 
  let remaining_wall := 1 - portion_completed_together
  let time_for_tom := remaining_wall / tom_rate_per_hour
  time_for_tom = (1:ℝ)/3 := 
by 
  sorry

end Tom_completes_wall_l43_43161


namespace display_glasses_count_l43_43231

noncomputable def tall_cupboards := 2
noncomputable def wide_cupboards := 2
noncomputable def narrow_cupboards := 2
noncomputable def shelves_per_narrow_cupboard := 3
noncomputable def glasses_tall_cupboard := 30
noncomputable def glasses_wide_cupboard := 2 * glasses_tall_cupboard
noncomputable def glasses_narrow_cupboard := 45
noncomputable def broken_shelf_glasses := glasses_narrow_cupboard / shelves_per_narrow_cupboard

theorem display_glasses_count :
  (tall_cupboards * glasses_tall_cupboard) +
  (wide_cupboards * glasses_wide_cupboard) +
  (1 * (broken_shelf_glasses * (shelves_per_narrow_cupboard - 1)) + glasses_narrow_cupboard) =
  255 :=
by sorry

end display_glasses_count_l43_43231


namespace inequality_solution_set_range_of_m_l43_43446

noncomputable def f (x : ℝ) : ℝ := |x - 1|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m

theorem inequality_solution_set :
  {x : ℝ | f x + x^2 - 1 > 0} = {x : ℝ | x > 1 ∨ x < 0} :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x < g x m) → m > 4 :=
sorry

end inequality_solution_set_range_of_m_l43_43446


namespace smallest_n_with_digits_315_l43_43138

-- Defining the conditions
def relatively_prime (m n : ℕ) := Nat.gcd m n = 1
def valid_fraction (m n : ℕ) := (m < n) ∧ relatively_prime m n

-- Predicate for the sequence 3, 1, 5 in the decimal representation of m/n
def contains_digits_315 (m n : ℕ) : Prop :=
  ∃ k d : ℕ, 10^k * m % n = 315 * 10^(d - 3) ∧ d ≥ 3

-- The main theorem: smallest n for which the conditions are satisfied
theorem smallest_n_with_digits_315 :
  ∃ n : ℕ, valid_fraction m n ∧ contains_digits_315 m n ∧ n = 159 :=
sorry

end smallest_n_with_digits_315_l43_43138


namespace total_bowling_balls_l43_43931

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l43_43931


namespace white_red_balls_l43_43892

theorem white_red_balls (w r : ℕ) 
  (h1 : 3 * w = 5 * r)
  (h2 : w + 15 + r = 50) : 
  r = 12 :=
by
  sorry

end white_red_balls_l43_43892


namespace minimum_value_l43_43791

-- Define geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * ((a 2 / a 1) ^ n)

-- Define the condition for positive geometric sequence
def positive_geometric_sequence (a : ℕ → ℝ) :=
  is_geometric_sequence a ∧ ∀ n : ℕ, a n > 0

-- Condition given in the problem
def condition (a : ℕ → ℝ) :=
  2 * a 4 + a 3 = 2 * a 2 + a 1 + 8

-- Define the problem statement to be proved
theorem minimum_value (a : ℕ → ℝ) (h1 : positive_geometric_sequence a) (h2 : condition a) :
  2 * a 6 + a 5 = 32 :=
sorry

end minimum_value_l43_43791


namespace win_sector_area_l43_43657

theorem win_sector_area (r : ℝ) (p_win : ℝ) (area_total : ℝ) 
  (h1 : r = 8)
  (h2 : p_win = 3 / 8)
  (h3 : area_total = π * r^2) :
  ∃ area_win, area_win = 24 * π ∧ area_win = p_win * area_total :=
by
  sorry

end win_sector_area_l43_43657


namespace sufficient_but_not_necessary_condition_l43_43121

theorem sufficient_but_not_necessary_condition 
  (x : ℝ) (h : x > 0) : (∃ y : ℝ, (y < -3 ∨ y > -1) ∧ y > 0) := by
  sorry

end sufficient_but_not_necessary_condition_l43_43121


namespace unique_triple_sum_l43_43021

theorem unique_triple_sum :
  ∃ (a b c : ℕ), 
    (10 ≤ a ∧ a < 100) ∧ 
    (10 ≤ b ∧ b < 100) ∧ 
    (10 ≤ c ∧ c < 100) ∧ 
    (a^3 + 3 * b^3 + 9 * c^3 = 9 * a * b * c + 1) ∧ 
    (a + b + c = 9) := 
sorry

end unique_triple_sum_l43_43021


namespace luke_trays_l43_43492

theorem luke_trays 
  (carries_per_trip : ℕ)
  (trips : ℕ)
  (second_table_trays : ℕ)
  (total_trays : carries_per_trip * trips = 36)
  (second_table_value : second_table_trays = 16) : 
  carries_per_trip * trips - second_table_trays = 20 :=
by sorry

end luke_trays_l43_43492


namespace least_number_subtracted_divisible_by_six_l43_43874

theorem least_number_subtracted_divisible_by_six :
  ∃ d : ℕ, d = 6 ∧ (427398 - 6) % d = 0 := by
sorry

end least_number_subtracted_divisible_by_six_l43_43874


namespace candy_problem_l43_43586

theorem candy_problem
  (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999)
  (h3 : n + 7 ≡ 0 [MOD 9])
  (h4 : n - 9 ≡ 0 [MOD 6]) :
  n = 101 :=
sorry

end candy_problem_l43_43586


namespace total_signs_at_intersections_l43_43314

-- Definitions based on the given conditions
def first_intersection_signs : ℕ := 40
def second_intersection_signs : ℕ := first_intersection_signs + first_intersection_signs / 4
def third_intersection_signs : ℕ := 2 * second_intersection_signs
def fourth_intersection_signs : ℕ := third_intersection_signs - 20

-- Prove the total number of signs at the four intersections is 270
theorem total_signs_at_intersections :
  first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs = 270 := by
  sorry

end total_signs_at_intersections_l43_43314


namespace negation_of_p_l43_43014

open Real

def p : Prop := ∃ x : ℝ, sin x < (1 / 2) * x

theorem negation_of_p : ¬p ↔ ∀ x : ℝ, sin x ≥ (1 / 2) * x := 
by
  sorry

end negation_of_p_l43_43014


namespace appropriate_sampling_method_l43_43649

/--
Given there are 40 products in total, consisting of 10 first-class products,
25 second-class products, and 5 defective products, if we need to select
8 products for quality analysis, then the appropriate sampling method is
the stratified sampling method.
-/
theorem appropriate_sampling_method
  (total_products : ℕ)
  (first_class_products : ℕ)
  (second_class_products : ℕ)
  (defective_products : ℕ)
  (selected_products : ℕ)
  (stratified_sampling : ℕ → ℕ → ℕ → ℕ → Prop) :
  total_products = 40 →
  first_class_products = 10 →
  second_class_products = 25 →
  defective_products = 5 →
  selected_products = 8 →
  stratified_sampling total_products first_class_products second_class_products defective_products →
  stratified_sampling total_products first_class_products second_class_products defective_products :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end appropriate_sampling_method_l43_43649


namespace profit_is_55_l43_43734

-- Define the given conditions:
def cost_of_chocolates (bars: ℕ) (price_per_bar: ℕ) : ℕ :=
  bars * price_per_bar

def cost_of_packaging (bars: ℕ) (cost_per_bar: ℕ) : ℕ :=
  bars * cost_per_bar

def total_sales : ℕ :=
  90

def total_cost (cost_of_chocolates cost_of_packaging: ℕ) : ℕ :=
  cost_of_chocolates + cost_of_packaging

def profit (total_sales total_cost: ℕ) : ℕ :=
  total_sales - total_cost

-- Given values:
def bars: ℕ := 5
def price_per_bar: ℕ := 5
def cost_per_packaging_bar: ℕ := 2

-- Define the profit calculation theorem:
theorem profit_is_55 : 
  profit total_sales (total_cost (cost_of_chocolates bars price_per_bar) (cost_of_packaging bars cost_per_packaging_bar)) = 55 :=
by {
  -- The proof will be inserted here
  sorry
}

end profit_is_55_l43_43734


namespace ben_points_l43_43656

theorem ben_points (B : ℕ) 
  (h1 : 42 = B + 21) : B = 21 := 
by 
-- Proof can be filled in here
sorry

end ben_points_l43_43656


namespace math_problem_l43_43690

noncomputable def compute_value (c d : ℝ) : ℝ := 100 * c + d

-- Problem statement as a theorem
theorem math_problem
  (c d : ℝ)
  (H1 : ∀ x : ℝ, (x + c) * (x + d) * (x + 10) = 0 → x = -c ∨ x = -d ∨ x = -10)
  (H2 : ∀ x : ℝ, (x + 3 * c) * (x + 5) * (x + 8) = 0 → (x = -4 ∧ ∀ y : ℝ, y ≠ -4 → (y + d) * (y + 10) ≠ 0))
  (H3 : c ≠ 4 / 3 → 3 * c = d ∨ 3 * c = 10) :
  compute_value c d = 141.33 :=
by sorry

end math_problem_l43_43690


namespace arithmetic_mean_of_set_l43_43068

theorem arithmetic_mean_of_set {x : ℝ} (mean_eq_12 : (8 + 16 + 20 + x + 12) / 5 = 12) : x = 4 :=
by
  sorry

end arithmetic_mean_of_set_l43_43068


namespace smallest_value_of_f4_l43_43710

def f (x : ℝ) : ℝ := (x + 3) ^ 2 - 2

theorem smallest_value_of_f4 : ∀ x : ℝ, f (f (f (f x))) ≥ 23 :=
by 
  sorry -- Proof goes here.

end smallest_value_of_f4_l43_43710


namespace correct_number_of_six_letter_words_l43_43595

def number_of_six_letter_words (alphabet_size : ℕ) : ℕ :=
  alphabet_size ^ 4

theorem correct_number_of_six_letter_words :
  number_of_six_letter_words 26 = 456976 :=
by
  -- We write 'sorry' to omit the detailed proof.
  sorry

end correct_number_of_six_letter_words_l43_43595


namespace find_k_collinear_l43_43576

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, -1)
def c : ℝ × ℝ := (1, 2)

theorem find_k_collinear : ∃ k : ℝ, (1 - 2 * k, 3 - k) = (-k, k) * c ∧ k = -1/3 :=
by
  sorry

end find_k_collinear_l43_43576


namespace problem_solution_l43_43311

theorem problem_solution (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 500 / 9 :=
by
  sorry

end problem_solution_l43_43311


namespace emily_lemon_juice_fraction_l43_43212

/-- 
Emily places 6 ounces of tea into a twelve-ounce cup and 6 ounces of honey into a second cup
of the same size. Then she adds 3 ounces of lemon juice to the second cup. Next, she pours half
the tea from the first cup into the second, mixes thoroughly, and then pours one-third of the
mixture in the second cup back into the first. 
Prove that the fraction of the mixture in the first cup that is lemon juice is 1/7.
--/
theorem emily_lemon_juice_fraction :
  let cup1_tea := 6
  let cup2_honey := 6
  let cup2_lemon_juice := 3
  let cup1_tea_transferred := cup1_tea / 2
  let cup1 := cup1_tea - cup1_tea_transferred
  let cup2 := cup2_honey + cup2_lemon_juice + cup1_tea_transferred
  let mix_ratio (x y : ℕ) := (x : ℚ) / (x + y)
  let cup1_after_transfer := cup1 + (cup2 / 3)
  let cup2_tea := cup1_tea_transferred
  let cup2_honey := cup2_honey
  let cup2_lemon_juice := cup2_lemon_juice
  let cup1_lemon_transferred := 1
  cup1_tea + (cup2 / 3) = 3 + (cup2_tea * (1 / 3)) + 1 + (cup2_honey * (1 / 3)) + cup2_lemon_juice / 3 →
  cup1 / (cup1 + cup2_honey) = 1/7 :=
sorry

end emily_lemon_juice_fraction_l43_43212


namespace inequality_for_large_n_l43_43320

theorem inequality_for_large_n (n : ℕ) (hn : n > 1) : 
  (1 / Real.exp 1 - 1 / (n * Real.exp 1)) < (1 - 1 / n) ^ n ∧ (1 - 1 / n) ^ n < (1 / Real.exp 1 - 1 / (2 * n * Real.exp 1)) :=
sorry

end inequality_for_large_n_l43_43320


namespace calculate_division_of_powers_l43_43634

theorem calculate_division_of_powers (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end calculate_division_of_powers_l43_43634


namespace points_total_l43_43738

/--
In a game, Samanta has 8 more points than Mark,
and Mark has 50% more points than Eric. Eric has 6 points.
How many points do Samanta, Mark, and Eric have in total?
-/
theorem points_total (Samanta Mark Eric : ℕ)
  (h1 : Samanta = Mark + 8)
  (h2 : Mark = Eric + Eric / 2)
  (h3 : Eric = 6) :
  Samanta + Mark + Eric = 32 := by
  sorry

end points_total_l43_43738


namespace patternD_cannot_form_pyramid_l43_43928

-- Define the patterns
inductive Pattern
| A
| B
| C
| D

-- Define the condition for folding into a pyramid with a square base
def canFormPyramidWithSquareBase (p : Pattern) : Prop :=
  p = Pattern.A ∨ p = Pattern.B ∨ p = Pattern.C

-- Goal: Prove that Pattern D cannot be folded into a pyramid with a square base
theorem patternD_cannot_form_pyramid : ¬ canFormPyramidWithSquareBase Pattern.D :=
by
  -- Need to provide the proof here
  sorry

end patternD_cannot_form_pyramid_l43_43928


namespace fuelA_amount_l43_43385

def tankCapacity : ℝ := 200
def ethanolInFuelA : ℝ := 0.12
def ethanolInFuelB : ℝ := 0.16
def totalEthanol : ℝ := 30
def limitedFuelA : ℝ := 100
def limitedFuelB : ℝ := 150

theorem fuelA_amount : ∃ (x : ℝ), 
  (x ≤ limitedFuelA ∧ x ≥ 0) ∧ 
  ((tankCapacity - x) ≤ limitedFuelB ∧ (tankCapacity - x) ≥ 0) ∧ 
  (ethanolInFuelA * x + ethanolInFuelB * (tankCapacity - x)) = totalEthanol ∧ 
  x = 50 := 
by
  sorry

end fuelA_amount_l43_43385


namespace power_addition_l43_43398

theorem power_addition :
  (-2 : ℤ) ^ 2009 + (-2 : ℤ) ^ 2010 = 2 ^ 2009 :=
by
  sorry

end power_addition_l43_43398


namespace school_days_per_week_l43_43415

-- Definitions based on the conditions given
def paper_per_class_per_day : ℕ := 200
def total_paper_per_week : ℕ := 9000
def number_of_classes : ℕ := 9

-- The theorem stating the main claim to prove
theorem school_days_per_week :
  total_paper_per_week / (paper_per_class_per_day * number_of_classes) = 5 :=
  by
  sorry

end school_days_per_week_l43_43415


namespace find_v2002_l43_43886

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 6
  | 4 => 2
  | 5 => 1
  | 6 => 7
  | 7 => 4
  | _ => 0

def seq_v : ℕ → ℕ
| 0       => 5
| (n + 1) => g (seq_v n)

theorem find_v2002 : seq_v 2002 = 5 :=
  sorry

end find_v2002_l43_43886


namespace yu_chan_walked_distance_l43_43680

def step_length : ℝ := 0.75
def walking_time : ℝ := 13
def steps_per_minute : ℝ := 70

theorem yu_chan_walked_distance : step_length * steps_per_minute * walking_time = 682.5 :=
by
  sorry

end yu_chan_walked_distance_l43_43680


namespace vertical_lines_count_l43_43806

theorem vertical_lines_count (n : ℕ) 
  (h_intersections : (18 * n * (n - 1)) = 756) : 
  n = 7 :=
by 
  sorry

end vertical_lines_count_l43_43806


namespace sqrt_450_simplified_l43_43591

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l43_43591


namespace geometric_sequence_problem_l43_43481

-- Definitions
def is_geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n, a (n + 1) = q * a n

-- Problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ)
    (h_geom : is_geom_seq a q)
    (h1 : a 3 * a 7 = 8)
    (h2 : a 4 + a 6 = 6) :
    a 2 + a 8 = 9 :=
sorry

end geometric_sequence_problem_l43_43481


namespace part_I_part_II_l43_43863

open Real

noncomputable def alpha₁ : Real := sorry -- Placeholder for the angle α in part I
noncomputable def alpha₂ : Real := sorry -- Placeholder for the angle α in part II

-- Given a point P(-4, 3) and a point on the terminal side of angle α₁ such that tan(α₁) = -3/4
theorem part_I :
  tan α₁ = - (3 / 4) → 
  (cos (π / 2 + α₁) * sin (-π - α₁)) / (cos (11 * π / 2 - α₁) * sin (9 * π / 2 + α₁)) = - (3 / 4) :=
by 
  intro h
  sorry

-- Given vector a = (3,1) and b = (sin α, cos α) where a is parallel to b such that tan(α₂) = 3
theorem part_II :
  tan α₂ = 3 → 
  (4 * sin α₂ - 2 * cos α₂) / (5 * cos α₂ + 3 * sin α₂) = 5 / 7 :=
by 
  intro h
  sorry

end part_I_part_II_l43_43863


namespace smallest_side_for_table_rotation_l43_43884

theorem smallest_side_for_table_rotation (S : ℕ) : (S ≥ Int.ofNat (Nat.sqrt (8^2 + 12^2) + 1)) → S = 15 := 
by
  sorry

end smallest_side_for_table_rotation_l43_43884


namespace original_square_area_is_correct_l43_43771

noncomputable def original_square_side_length (s : ℝ) :=
  let original_area := s^2
  let new_width := 0.8 * s
  let new_length := 5 * s
  let new_area := new_width * new_length
  let increased_area := new_area - original_area
  increased_area = 15.18

theorem original_square_area_is_correct (s : ℝ) (h : original_square_side_length s) : s^2 = 5.06 := by
  sorry

end original_square_area_is_correct_l43_43771


namespace problem1_problem2_l43_43300

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- Problem 1: (0 < m < 1/e) implies g(x) = f(x) - m has two zeros
theorem problem1 (m : ℝ) (h1 : 0 < m) (h2 : m < 1 / Real.exp 1) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = m ∧ f x2 = m :=
sorry

-- Problem 2: (2/e^2 ≤ a < 1/e) implies f^2(x) - af(x) > 0 has only one integer solution
theorem problem2 (a : ℝ) (h1 : 2 / (Real.exp 2) ≤ a) (h2 : a < 1 / Real.exp 1) :
  ∃! x : ℤ, ∀ y : ℤ, (f y)^2 - a * (f y) > 0 → y = x :=
sorry

end problem1_problem2_l43_43300


namespace volume_of_prism_l43_43753

   theorem volume_of_prism (a b c : ℝ)
     (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) :
     a * b * c = 24 * Real.sqrt 3 :=
   sorry
   
end volume_of_prism_l43_43753


namespace range_of_a_for_inequality_l43_43879

noncomputable def has_solution_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ (x^2 + a*x - 2 < 0)

theorem range_of_a_for_inequality : ∀ a : ℝ, has_solution_in_interval a ↔ a < 1 :=
by sorry

end range_of_a_for_inequality_l43_43879


namespace f_neg1_gt_f_1_l43_43598

-- Definition of the function f and its properties.
variable {f : ℝ → ℝ}
variable (df : Differentiable ℝ f)
variable (eq_f : ∀ x : ℝ, f x = x^2 + 2 * x * f' 2)

-- The problem statement to prove f(-1) > f(1).
theorem f_neg1_gt_f_1 (h_deriv : ∀ x : ℝ, deriv f x = 2 * x - 8):
  f (-1) > f 1 :=
by
  sorry

end f_neg1_gt_f_1_l43_43598


namespace base_rate_of_second_company_l43_43733

-- Define the conditions
def United_base_rate : ℝ := 8.00
def United_rate_per_minute : ℝ := 0.25
def Other_rate_per_minute : ℝ := 0.20
def minutes : ℕ := 80

-- Define the total bill equations
def United_total_bill (minutes : ℕ) : ℝ := United_base_rate + United_rate_per_minute * minutes
def Other_total_bill (minutes : ℕ) (B : ℝ) : ℝ := B + Other_rate_per_minute * minutes

-- Define the claim to prove
theorem base_rate_of_second_company : ∃ B : ℝ, Other_total_bill minutes B = United_total_bill minutes ∧ B = 12.00 := by
  sorry

end base_rate_of_second_company_l43_43733


namespace truthful_dwarfs_count_l43_43381

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l43_43381


namespace distinguishable_squares_count_is_70_l43_43347

def count_distinguishable_squares : ℕ :=
  let total_colorings : ℕ := 2^9
  let rotation_90_270_fixed : ℕ := 2^3
  let rotation_180_fixed : ℕ := 2^5
  let average_fixed_colorings : ℕ :=
    (total_colorings + rotation_90_270_fixed + rotation_90_270_fixed + rotation_180_fixed) / 4
  let distinguishable_squares : ℕ := average_fixed_colorings / 2
  distinguishable_squares

theorem distinguishable_squares_count_is_70 :
  count_distinguishable_squares = 70 := by
  sorry

end distinguishable_squares_count_is_70_l43_43347


namespace tangent_product_equals_2_pow_23_l43_43563

noncomputable def tangent_product : ℝ :=
  (1 + Real.tan (1 * Real.pi / 180)) *
  (1 + Real.tan (2 * Real.pi / 180)) *
  (1 + Real.tan (3 * Real.pi / 180)) *
  (1 + Real.tan (4 * Real.pi / 180)) *
  (1 + Real.tan (5 * Real.pi / 180)) *
  (1 + Real.tan (6 * Real.pi / 180)) *
  (1 + Real.tan (7 * Real.pi / 180)) *
  (1 + Real.tan (8 * Real.pi / 180)) *
  (1 + Real.tan (9 * Real.pi / 180)) *
  (1 + Real.tan (10 * Real.pi / 180)) *
  (1 + Real.tan (11 * Real.pi / 180)) *
  (1 + Real.tan (12 * Real.pi / 180)) *
  (1 + Real.tan (13 * Real.pi / 180)) *
  (1 + Real.tan (14 * Real.pi / 180)) *
  (1 + Real.tan (15 * Real.pi / 180)) *
  (1 + Real.tan (16 * Real.pi / 180)) *
  (1 + Real.tan (17 * Real.pi / 180)) *
  (1 + Real.tan (18 * Real.pi / 180)) *
  (1 + Real.tan (19 * Real.pi / 180)) *
  (1 + Real.tan (20 * Real.pi / 180)) *
  (1 + Real.tan (21 * Real.pi / 180)) *
  (1 + Real.tan (22 * Real.pi / 180)) *
  (1 + Real.tan (23 * Real.pi / 180)) *
  (1 + Real.tan (24 * Real.pi / 180)) *
  (1 + Real.tan (25 * Real.pi / 180)) *
  (1 + Real.tan (26 * Real.pi / 180)) *
  (1 + Real.tan (27 * Real.pi / 180)) *
  (1 + Real.tan (28 * Real.pi / 180)) *
  (1 + Real.tan (29 * Real.pi / 180)) *
  (1 + Real.tan (30 * Real.pi / 180)) *
  (1 + Real.tan (31 * Real.pi / 180)) *
  (1 + Real.tan (32 * Real.pi / 180)) *
  (1 + Real.tan (33 * Real.pi / 180)) *
  (1 + Real.tan (34 * Real.pi / 180)) *
  (1 + Real.tan (35 * Real.pi / 180)) *
  (1 + Real.tan (36 * Real.pi / 180)) *
  (1 + Real.tan (37 * Real.pi / 180)) *
  (1 + Real.tan (38 * Real.pi / 180)) *
  (1 + Real.tan (39 * Real.pi / 180)) *
  (1 + Real.tan (40 * Real.pi / 180)) *
  (1 + Real.tan (41 * Real.pi / 180)) *
  (1 + Real.tan (42 * Real.pi / 180)) *
  (1 + Real.tan (43 * Real.pi / 180)) *
  (1 + Real.tan (44 * Real.pi / 180)) *
  (1 + Real.tan (45 * Real.pi / 180))

theorem tangent_product_equals_2_pow_23 : tangent_product = 2 ^ 23 :=
  sorry

end tangent_product_equals_2_pow_23_l43_43563


namespace choir_average_age_l43_43064

theorem choir_average_age 
  (avg_f : ℝ) (n_f : ℕ)
  (avg_m : ℝ) (n_m : ℕ)
  (h_f : avg_f = 28) 
  (h_nf : n_f = 12) 
  (h_m : avg_m = 40) 
  (h_nm : n_m = 18) 
  : (n_f * avg_f + n_m * avg_m) / (n_f + n_m) = 35.2 := 
by 
  sorry

end choir_average_age_l43_43064


namespace moving_circle_passes_through_fixed_point_l43_43921

-- Define the parabola x^2 = 12y
def parabola (x y : ℝ) : Prop := x^2 = 12 * y

-- Define the directrix line y = -3
def directrix (y : ℝ) : Prop := y = -3

-- The fixed point we need to show the circle always passes through
def fixed_point : ℝ × ℝ := (0, 3)

-- Define the condition that the moving circle is centered on the parabola and tangent to the directrix
def circle_centered_on_parabola_and_tangent_to_directrix (x y : ℝ) (r : ℝ) : Prop :=
  parabola x y ∧ r = abs (y + 3)

-- Main theorem statement
theorem moving_circle_passes_through_fixed_point :
  (∀ (x y r : ℝ), circle_centered_on_parabola_and_tangent_to_directrix x y r → 
    (∃ (px py : ℝ), (px, py) = fixed_point ∧ (px - x)^2 + (py - y)^2 = r^2)) :=
sorry

end moving_circle_passes_through_fixed_point_l43_43921


namespace sin_585_eq_neg_sqrt2_div_2_l43_43227

theorem sin_585_eq_neg_sqrt2_div_2 : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_585_eq_neg_sqrt2_div_2_l43_43227


namespace beneficial_for_kati_l43_43968

variables (n : ℕ) (x y : ℝ)

theorem beneficial_for_kati (hn : n > 0) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / (n + 2) > (x + y / 2) / (n + 1) :=
sorry

end beneficial_for_kati_l43_43968


namespace sum_of_roots_l43_43502

theorem sum_of_roots (a b c : ℝ) (h : 6 * a ^ 3 - 7 * a ^ 2 + 2 * a = 0 ∧ 
                                   6 * b ^ 3 - 7 * b ^ 2 + 2 * b = 0 ∧ 
                                   6 * c ^ 3 - 7 * c ^ 2 + 2 * c = 0 ∧ 
                                   a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
    a + b + c = 7 / 6 :=
sorry

end sum_of_roots_l43_43502


namespace find_five_value_l43_43952

def f (x : ℝ) : ℝ := x^2 - x

theorem find_five_value : f 5 = 20 := by
  sorry

end find_five_value_l43_43952


namespace range_of_m_l43_43911

def p (m : ℝ) : Prop :=
  let Δ := m^2 - 4
  Δ > 0 ∧ -m < 0

def q (m : ℝ) : Prop :=
  let Δ := 16*(m-2)^2 - 16
  Δ < 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ ((1 < m ∧ m ≤ 2) ∨ 3 ≤ m) :=
by {
  sorry
}

end range_of_m_l43_43911


namespace product_of_possible_b_values_l43_43460

theorem product_of_possible_b_values (b : ℝ) :
  (∀ (y1 y2 x1 x2 : ℝ), y1 = -1 ∧ y2 = 3 ∧ x1 = 2 ∧ (x2 = b) ∧ (y2 - y1 = 4) → 
   (b = 2 + 4 ∨ b = 2 - 4)) → 
  (b = 6 ∨ b = -2) → (b = 6) ∧ (b = -2) → 6 * -2 = -12 :=
sorry

end product_of_possible_b_values_l43_43460


namespace probability_first_spade_last_ace_l43_43047

-- Define the problem parameters
def standard_deck : ℕ := 52
def spades_count : ℕ := 13
def aces_count : ℕ := 4
def ace_of_spades : ℕ := 1

-- Probability of drawing a spade but not an ace as the first card
def prob_spade_not_ace_first : ℚ := 12 / 52

-- Probability of drawing any of the four aces among the two remaining cards
def prob_ace_among_two_remaining : ℚ := 4 / 50

-- Probability of drawing the ace of spades as the first card
def prob_ace_of_spades_first : ℚ := 1 / 52

-- Probability of drawing one of three remaining aces among two remaining cards
def prob_three_aces_among_two_remaining : ℚ := 3 / 50

-- Combined probability according to the cases
def final_probability : ℚ := (prob_spade_not_ace_first * prob_ace_among_two_remaining) + (prob_ace_of_spades_first * prob_three_aces_among_two_remaining)

-- The theorem stating that the computed probability matches the expected result
theorem probability_first_spade_last_ace : final_probability = 51 / 2600 := 
  by
    -- inserting proof steps here would solve the theorem
    sorry

end probability_first_spade_last_ace_l43_43047


namespace total_students_registered_l43_43500

theorem total_students_registered 
  (students_yesterday : ℕ) (absent_today : ℕ) 
  (attended_today : ℕ)
  (h1 : students_yesterday = 70)
  (h2 : absent_today = 30)
  (h3 : attended_today = (2 * students_yesterday) - (10 * (2 * students_yesterday) / 100)) :
  students_yesterday + absent_today = 156 := 
by
  sorry

end total_students_registered_l43_43500


namespace part1_part2_l43_43685

noncomputable def A (x : ℝ) (k : ℝ) := -2 * x ^ 2 - (k - 1) * x + 1
noncomputable def B (x : ℝ) := -2 * (x ^ 2 - x + 2)

-- Part 1: If A is a quadratic binomial, then the value of k is 1
theorem part1 (x : ℝ) (k : ℝ) (h : ∀ x, A x k ≠ 0) : k = 1 :=
sorry

-- Part 2: When k = -1, C + 2A = B, then C = 2x^2 - 2x - 6
theorem part2 (x : ℝ) (C : ℝ → ℝ) (h1 : k = -1) (h2 : ∀ x, C x + 2 * A x k = B x) : (C x = 2 * x ^ 2 - 2 * x - 6) :=
sorry

end part1_part2_l43_43685


namespace abs_lt_one_suff_but_not_necc_l43_43358

theorem abs_lt_one_suff_but_not_necc (x : ℝ) : (|x| < 1 → x^2 + x - 2 < 0) ∧ ¬(x^2 + x - 2 < 0 → |x| < 1) :=
by
  sorry

end abs_lt_one_suff_but_not_necc_l43_43358


namespace value_of_x2_plus_9y2_l43_43640

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end value_of_x2_plus_9y2_l43_43640


namespace general_term_formula_l43_43557

def seq (n : ℕ) : ℤ :=
match n with
| 0       => 1
| 1       => -3
| 2       => 5
| 3       => -7
| 4       => 9
| (n + 1) => (-1)^(n+1) * (2*n + 1) -- extends indefinitely for general natural number

theorem general_term_formula (n : ℕ) : 
  seq n = (-1)^(n+1) * (2*n-1) :=
sorry

end general_term_formula_l43_43557


namespace ratio_of_distances_l43_43166

-- Define the speeds and times for ferries P and Q
def speed_P : ℝ := 8
def time_P : ℝ := 3
def speed_Q : ℝ := speed_P + 1
def time_Q : ℝ := time_P + 5

-- Define the distances covered by ferries P and Q
def distance_P : ℝ := speed_P * time_P
def distance_Q : ℝ := speed_Q * time_Q

-- The statement to prove: the ratio of the distances
theorem ratio_of_distances : distance_Q / distance_P = 3 :=
sorry

end ratio_of_distances_l43_43166


namespace jill_total_watch_time_l43_43688

theorem jill_total_watch_time :
  ∀ (length_first_show length_second_show total_watch_time : ℕ),
    length_first_show = 30 →
    length_second_show = 4 * length_first_show →
    total_watch_time = length_first_show + length_second_show →
    total_watch_time = 150 :=
by
  sorry

end jill_total_watch_time_l43_43688


namespace equal_poly_terms_l43_43365

theorem equal_poly_terms (p q : ℝ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) : 
  (7 * p^6 * q = 21 * p^5 * q^2) -> p = 3 / 4 :=
by
  sorry

end equal_poly_terms_l43_43365


namespace diana_hourly_wage_l43_43524

theorem diana_hourly_wage :
  (∃ (hours_monday : ℕ) (hours_tuesday : ℕ) (hours_wednesday : ℕ) (hours_thursday : ℕ) (hours_friday : ℕ) (weekly_earnings : ℝ),
    hours_monday = 10 ∧
    hours_tuesday = 15 ∧
    hours_wednesday = 10 ∧
    hours_thursday = 15 ∧
    hours_friday = 10 ∧
    weekly_earnings = 1800 ∧
    (weekly_earnings / (hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday) = 30)) :=
sorry

end diana_hourly_wage_l43_43524


namespace each_person_tip_l43_43489

-- Definitions based on the conditions
def julie_cost : ℝ := 10
def letitia_cost : ℝ := 20
def anton_cost : ℝ := 30
def tip_rate : ℝ := 0.2

-- Theorem statement
theorem each_person_tip (total_cost := julie_cost + letitia_cost + anton_cost)
 (total_tip := total_cost * tip_rate) :
 (total_tip / 3) = 4 := by
  sorry

end each_person_tip_l43_43489


namespace gcd_7488_12467_eq_39_l43_43207

noncomputable def gcd_7488_12467 : ℕ := Nat.gcd 7488 12467

theorem gcd_7488_12467_eq_39 : gcd_7488_12467 = 39 :=
sorry

end gcd_7488_12467_eq_39_l43_43207


namespace factorization_correct_l43_43803

theorem factorization_correct (x : ℝ) : 
  (x^2 + 5 * x + 2) * (x^2 + 5 * x + 3) - 12 = (x + 2) * (x + 3) * (x^2 + 5 * x - 1) :=
by
  sorry

end factorization_correct_l43_43803


namespace find_prime_pair_l43_43098
open Int

theorem find_prime_pair :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ ∃ (p : ℕ), Prime p ∧ p = a * b^2 / (a + b) ∧ (a, b) = (6, 2) := by
  sorry

end find_prime_pair_l43_43098


namespace total_people_attended_l43_43870

theorem total_people_attended (A C : ℕ) (ticket_price_adult ticket_price_child : ℕ) (total_receipts : ℕ) 
  (number_of_children : ℕ) (h_ticket_prices : ticket_price_adult = 60 ∧ ticket_price_child = 25)
  (h_total_receipts : total_receipts = 140 * 100) (h_children : C = 80) 
  (h_equation : ticket_price_adult * A + ticket_price_child * C = total_receipts) : 
  A + C = 280 :=
by
  sorry

end total_people_attended_l43_43870


namespace slope_of_intersection_points_l43_43990

theorem slope_of_intersection_points :
  ∀ s : ℝ, ∃ k b : ℝ, (∀ (x y : ℝ), (2 * x - 3 * y = 4 * s + 6) ∧ (2 * x + y = 3 * s + 1) → y = k * x + b) ∧ k = -2/13 := 
by
  intros s
  -- Proof to be provided here
  sorry

end slope_of_intersection_points_l43_43990


namespace fewest_coach_handshakes_l43_43659

theorem fewest_coach_handshakes (n k : ℕ) (h1 : (n * (n - 1)) / 2 + k = 281) : k = 5 :=
sorry

end fewest_coach_handshakes_l43_43659


namespace ms_hatcher_total_students_l43_43055

theorem ms_hatcher_total_students :
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  third_graders + fourth_graders + fifth_graders = 70 :=
by 
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  show third_graders + fourth_graders + fifth_graders = 70
  sorry

end ms_hatcher_total_students_l43_43055


namespace sufficient_but_not_necessary_l43_43769

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > b + 1) → (a > b) ∧ ¬(a > b → a > b + 1) :=
by
  sorry

end sufficient_but_not_necessary_l43_43769


namespace trader_gain_percentage_l43_43096

structure PenType :=
  (pens_sold : ℕ)
  (cost_per_pen : ℕ)

def total_cost (pen : PenType) : ℕ :=
  pen.pens_sold * pen.cost_per_pen

def gain (pen : PenType) (multiplier : ℕ) : ℕ :=
  multiplier * pen.cost_per_pen

def weighted_average_gain_percentage (penA penB penC : PenType) (gainA gainB gainC : ℕ) : ℚ :=
  (((gainA + gainB + gainC):ℚ) / ((total_cost penA + total_cost penB + total_cost penC):ℚ)) * 100

theorem trader_gain_percentage :
  ∀ (penA penB penC : PenType)
  (gainA gainB gainC : ℕ),
  penA.pens_sold = 60 →
  penA.cost_per_pen = 2 →
  penB.pens_sold = 40 →
  penB.cost_per_pen = 3 →
  penC.pens_sold = 50 →
  penC.cost_per_pen = 4 →
  gainA = 20 * penA.cost_per_pen →
  gainB = 15 * penB.cost_per_pen →
  gainC = 10 * penC.cost_per_pen →
  weighted_average_gain_percentage penA penB penC gainA gainB gainC = 28.41 := 
by
  intros
  sorry

end trader_gain_percentage_l43_43096


namespace find_tan_of_cos_in_4th_quadrant_l43_43343

-- Given conditions
variable (α : ℝ) (h1 : Real.cos α = 3/5) (h2 : α > 3*Real.pi/2 ∧ α < 2*Real.pi)

-- Lean statement to prove the question
theorem find_tan_of_cos_in_4th_quadrant : Real.tan α = - (4 / 3) := 
by
  sorry

end find_tan_of_cos_in_4th_quadrant_l43_43343


namespace right_triangle_legs_l43_43451

theorem right_triangle_legs (a b : ℝ) (r R : ℝ) (hypotenuse : ℝ) (h_ab : a + b = 14) (h_c : hypotenuse = 10)
  (h_leg: a * b = a + b + 10) (h_Pythag : a^2 + b^2 = hypotenuse^2) 
  (h_inradius : r = 2) (h_circumradius : R = 5) : (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) :=
by
  sorry

end right_triangle_legs_l43_43451


namespace manager_hourly_wage_l43_43443

open Real

theorem manager_hourly_wage (M D C : ℝ) 
  (hD : D = M / 2)
  (hC : C = 1.20 * D)
  (hC_manager : C = M - 3.40) :
  M = 8.50 :=
by
  sorry

end manager_hourly_wage_l43_43443


namespace smallest_integer_representation_l43_43020

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end smallest_integer_representation_l43_43020


namespace find_three_numbers_l43_43298

theorem find_three_numbers :
  ∃ (a₁ a₄ a₂₅ : ℕ), a₁ + a₄ + a₂₅ = 114 ∧
    ( ∃ r ≠ 1, a₄ = a₁ * r ∧ a₂₅ = a₄ * r * r ) ∧
    ( ∃ d, a₄ = a₁ + 3 * d ∧ a₂₅ = a₁ + 24 * d ) ∧
    a₁ = 2 ∧ a₄ = 14 ∧ a₂₅ = 98 :=
by
  sorry

end find_three_numbers_l43_43298


namespace salty_cookies_initial_at_least_34_l43_43571

variable {S : ℕ}  -- S will represent the initial number of salty cookies

-- Conditions from the problem
def sweet_cookies_initial := 8
def sweet_cookies_ate := 20
def salty_cookies_ate := 34
def more_salty_than_sweet := 14

theorem salty_cookies_initial_at_least_34 :
  8 = sweet_cookies_initial ∧
  20 = sweet_cookies_ate ∧
  34 = salty_cookies_ate ∧
  salty_cookies_ate = sweet_cookies_ate + more_salty_than_sweet
  → S ≥ 34 :=
by sorry

end salty_cookies_initial_at_least_34_l43_43571


namespace dependence_of_Q_l43_43783

theorem dependence_of_Q (a d k : ℕ) :
    ∃ (Q : ℕ), Q = (2 * k * (2 * a + 4 * k * d - d)) 
                - (k * (2 * a + (2 * k - 1) * d)) 
                - (k / 2 * (2 * a + (k - 1) * d)) 
                → Q = k * a + 13 * k^2 * d := 
sorry

end dependence_of_Q_l43_43783


namespace pattern_equation_l43_43198

theorem pattern_equation (n : ℕ) : n^2 + n = n * (n + 1) := 
  sorry

end pattern_equation_l43_43198


namespace yulgi_allowance_l43_43163

theorem yulgi_allowance (Y G : ℕ) (h₁ : Y + G = 6000) (h₂ : (Y + G) - (Y - G) = 4800) (h₃ : Y > G) : Y = 3600 :=
sorry

end yulgi_allowance_l43_43163


namespace value_of_y_l43_43396

theorem value_of_y (x y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x) : y = 9 / 2 :=
sorry

end value_of_y_l43_43396


namespace option_D_is_negative_l43_43126

theorem option_D_is_negative :
  let A := abs (-4)
  let B := -(-4)
  let C := (-4) ^ 2
  let D := -(4 ^ 2)
  D < 0 := by
{
  -- Place sorry here since we are not required to provide the proof
  sorry
}

end option_D_is_negative_l43_43126


namespace remainder_when_7645_divided_by_9_l43_43840

/--
  Prove that the remainder when 7645 is divided by 9 is 4,
  given that a number is congruent to the sum of its digits modulo 9.
-/
theorem remainder_when_7645_divided_by_9 :
  7645 % 9 = 4 :=
by
  -- Main proof should go here
  sorry

end remainder_when_7645_divided_by_9_l43_43840


namespace number_of_possible_values_for_b_l43_43694

theorem number_of_possible_values_for_b : 
  ∃ (n : ℕ), n = 10 ∧ ∀ (b : ℕ), (2 ≤ b) ∧ (b^2 ≤ 256) ∧ (256 < b^3) ↔ (7 ≤ b ∧ b ≤ 16) :=
by {
  sorry
}

end number_of_possible_values_for_b_l43_43694


namespace solve_for_x_l43_43925

theorem solve_for_x (x y : ℕ) (h1 : x / y = 10 / 4) (h2 : y = 18) : x = 45 :=
sorry

end solve_for_x_l43_43925


namespace dot_product_AB_BC_l43_43824

variable (AB AC : ℝ × ℝ)

def BC (AB AC : ℝ × ℝ) : ℝ × ℝ := (AC.1 - AB.1, AC.2 - AB.2)

def dot_product (u v : ℝ × ℝ) : ℝ := (u.1 * v.1) + (u.2 * v.2)

theorem dot_product_AB_BC :
  ∀ (AB AC : ℝ × ℝ), AB = (2, 3) → AC = (3, 4) →
  dot_product AB (BC AB AC) = 5 :=
by
  intros
  unfold BC
  unfold dot_product
  sorry

end dot_product_AB_BC_l43_43824


namespace factorize_expression_l43_43229

theorem factorize_expression (a : ℚ) : 2 * a^2 - 4 * a = 2 * a * (a - 2) := by
  sorry

end factorize_expression_l43_43229


namespace inequality_solution_l43_43301

theorem inequality_solution (x : ℝ) : 
  (0 < (x + 2) / ((x - 3)^3)) ↔ (x < -2 ∨ x > 3)  :=
by
  sorry

end inequality_solution_l43_43301


namespace shorter_piece_length_correct_l43_43247

noncomputable def shorter_piece_length (total_length : ℝ) (ratio : ℝ) : ℝ := 
  total_length * ratio / (ratio + 1)

theorem shorter_piece_length_correct :
  shorter_piece_length 57.134 (3.25678 / 7.81945) = 16.790 :=
by
  sorry

end shorter_piece_length_correct_l43_43247


namespace total_students_calculation_l43_43061

variable (x : ℕ)
variable (girls_jelly_beans boys_jelly_beans total_jelly_beans : ℕ)
variable (total_students : ℕ)
variable (remaining_jelly_beans : ℕ)

-- Defining the number of boys as per the problem's conditions
def boys (x : ℕ) : ℕ := 2 * x + 3

-- Defining the jelly beans given to girls
def jelly_beans_given_to_girls (x girls_jelly_beans : ℕ) : Prop :=
  girls_jelly_beans = 2 * x * x

-- Defining the jelly beans given to boys
def jelly_beans_given_to_boys (x boys_jelly_beans : ℕ) : Prop :=
  boys_jelly_beans = 3 * (2 * x + 3) * (2 * x + 3)

-- Defining the total jelly beans given out
def total_jelly_beans_given_out (girls_jelly_beans boys_jelly_beans total_jelly_beans : ℕ) : Prop :=
  total_jelly_beans = girls_jelly_beans + boys_jelly_beans

-- Defining the total number of students
def total_students_in_class (x total_students : ℕ) : Prop :=
  total_students = x + boys x

-- Proving that the total number of students is 18 under given conditions
theorem total_students_calculation (h1 : jelly_beans_given_to_girls x girls_jelly_beans)
                                   (h2 : jelly_beans_given_to_boys x boys_jelly_beans)
                                   (h3 : total_jelly_beans_given_out girls_jelly_beans boys_jelly_beans total_jelly_beans)
                                   (h4 : total_jelly_beans - remaining_jelly_beans = 642)
                                   (h5 : remaining_jelly_beans = 3) :
                                   total_students = 18 :=
by
  sorry

end total_students_calculation_l43_43061


namespace parabola_vertex_l43_43033

theorem parabola_vertex :
  ∃ (x y : ℤ), ((∀ x : ℝ, 2 * x^2 - 4 * x - 7 = y) ∧ x = 1 ∧ y = -9) := 
sorry

end parabola_vertex_l43_43033


namespace inequality_solution_l43_43972

theorem inequality_solution (x : ℝ) : (x ≠ -2) ↔ (0 ≤ x^2 / (x + 2)^2) := by
  sorry

end inequality_solution_l43_43972


namespace parametric_plane_equiv_l43_43393

/-- Define the parametric form of the plane -/
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + s - t, 2 - s, 3 - 2*s + 2*t)

/-- Define the equation of the plane in standard form -/
def plane_equation (x y z : ℝ) : Prop :=
  2 * x + z - 5 = 0

/-- The theorem stating that the parametric form corresponds to the given plane equation -/
theorem parametric_plane_equiv :
  ∃ x y z s t,
    (x, y, z) = parametric_plane s t ∧ plane_equation x y z :=
by
  sorry

end parametric_plane_equiv_l43_43393


namespace equidistant_point_l43_43997

/-- 
  Find the point in the xz-plane that is equidistant from the points (1, 0, 0), 
  (0, -2, 3), and (4, 2, -2). The point in question is \left( \frac{41}{7}, 0, -\frac{19}{14} \right).
-/
theorem equidistant_point :
  ∃ (x z : ℚ), 
    (x - 1)^2 + z^2 = x^2 + 4 + (z - 3)^2 ∧
    (x - 1)^2 + z^2 = (x - 4)^2 + 4 + (z + 2)^2 ∧
    x = 41 / 7 ∧ z = -19 / 14 :=
by
  sorry

end equidistant_point_l43_43997


namespace length_of_goods_train_l43_43364

-- Define the given data
def speed_kmph := 72
def platform_length_m := 250
def crossing_time_s := 36

-- Convert speed from kmph to m/s
def speed_mps := speed_kmph * (5 / 18)

-- Define the total distance covered while crossing the platform
def distance_covered_m := speed_mps * crossing_time_s

-- Define the length of the train
def train_length_m := distance_covered_m - platform_length_m

-- The theorem to be proven
theorem length_of_goods_train : train_length_m = 470 := by
  sorry

end length_of_goods_train_l43_43364


namespace number_of_schools_in_pythagoras_city_l43_43521

theorem number_of_schools_in_pythagoras_city (n : ℕ) (h1 : true) 
    (h2 : true) (h3 : ∃ m, m = (3 * n + 1) / 2)
    (h4 : true) (h5 : true) : n = 24 :=
by 
  have h6 : 69 < 3 * n := sorry
  have h7 : 3 * n < 79 := sorry
  sorry

end number_of_schools_in_pythagoras_city_l43_43521


namespace jamies_class_girls_count_l43_43467

theorem jamies_class_girls_count 
  (g b : ℕ)
  (h_ratio : 4 * g = 3 * b)
  (h_total : g + b = 35) 
  : g = 15 := 
by 
  sorry 

end jamies_class_girls_count_l43_43467


namespace mark_has_24_dollars_l43_43880

theorem mark_has_24_dollars
  (small_bag_cost : ℕ := 4)
  (small_bag_balloons : ℕ := 50)
  (medium_bag_cost : ℕ := 6)
  (medium_bag_balloons : ℕ := 75)
  (large_bag_cost : ℕ := 12)
  (large_bag_balloons : ℕ := 200)
  (total_balloons : ℕ := 400) :
  total_balloons / large_bag_balloons = 2 ∧ 2 * large_bag_cost = 24 := by
  sorry

end mark_has_24_dollars_l43_43880


namespace average_of_numbers_l43_43777

theorem average_of_numbers (x : ℝ) (h : (5 + -1 + -2 + x) / 4 = 1) : x = 2 :=
by
  sorry

end average_of_numbers_l43_43777


namespace cards_per_page_l43_43602

noncomputable def total_cards (new_cards old_cards : ℕ) : ℕ := new_cards + old_cards

theorem cards_per_page
  (new_cards old_cards : ℕ)
  (total_pages : ℕ)
  (h_new_cards : new_cards = 3)
  (h_old_cards : old_cards = 13)
  (h_total_pages : total_pages = 2) :
  total_cards new_cards old_cards / total_pages = 8 :=
by
  rw [h_new_cards, h_old_cards, h_total_pages]
  rfl

end cards_per_page_l43_43602


namespace inequality_example_l43_43707

theorem inequality_example (a b c : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (sum_eq_one : a + b + c = 1) :
  (a + 1 / a) * (b + 1 / b) * (c + 1 / c) ≥ 1000 / 27 := 
by 
  sorry

end inequality_example_l43_43707


namespace system_of_equations_solution_l43_43811

theorem system_of_equations_solution (x y z : ℝ) 
  (h1 : x + y = -1) 
  (h2 : x + z = 0) 
  (h3 : y + z = 1) : 
  x = -1 ∧ y = 0 ∧ z = 1 :=
by
  sorry

end system_of_equations_solution_l43_43811


namespace range_of_x_l43_43564

open Set

noncomputable def M (x : ℝ) : Set ℝ := {x^2, 1}

theorem range_of_x (x : ℝ) (hx : M x) : x ≠ 1 ∧ x ≠ -1 :=
by
  sorry

end range_of_x_l43_43564


namespace complement_union_A_B_l43_43786

-- Define the sets U, A, and B as per the conditions
def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- Specify the statement to prove the complement of A ∪ B with respect to U
theorem complement_union_A_B : (U \ (A ∪ B)) = {2, 4} :=
by
  sorry

end complement_union_A_B_l43_43786


namespace expand_expression_l43_43431

theorem expand_expression (x y : ℝ) : 
  (2 * x + 3) * (5 * y + 7) = 10 * x * y + 14 * x + 15 * y + 21 := 
by sorry

end expand_expression_l43_43431


namespace KHSO4_formed_l43_43858

-- Define the reaction condition and result using moles
def KOH_moles : ℕ := 2
def H2SO4_moles : ℕ := 2

-- The balanced chemical reaction in terms of moles
-- 1 mole of KOH reacts with 1 mole of H2SO4 to produce 
-- 1 mole of KHSO4
def react (koh : ℕ) (h2so4 : ℕ) : ℕ := 
  -- stoichiometry 1:1 ratio of KOH and H2SO4 to KHSO4
  if koh ≤ h2so4 then koh else h2so4

-- The proof statement that verifies the expected number of moles of KHSO4
theorem KHSO4_formed (koh : ℕ) (h2so4 : ℕ) (hrs : react koh h2so4 = koh) : 
  koh = KOH_moles → h2so4 = H2SO4_moles → react koh h2so4 = 2 := 
by
  intros 
  sorry

end KHSO4_formed_l43_43858


namespace xy_equals_nine_l43_43376

theorem xy_equals_nine (x y : ℝ) (h : (|x + 3| > 0 ∧ (y - 2)^2 = 0) ∨ (|x + 3| = 0 ∧ (y - 2)^2 > 0)) : x^y = 9 :=
sorry

end xy_equals_nine_l43_43376


namespace radius_increase_l43_43159

-- Definitions and conditions
def initial_circumference : ℝ := 24
def final_circumference : ℝ := 30
def circumference_radius_relation (C : ℝ) (r : ℝ) : Prop := C = 2 * Real.pi * r

-- Required proof statement
theorem radius_increase (r1 r2 Δr : ℝ)
  (h1 : circumference_radius_relation initial_circumference r1)
  (h2 : circumference_radius_relation final_circumference r2)
  (h3 : Δr = r2 - r1) :
  Δr = 3 / Real.pi :=
by
  sorry

end radius_increase_l43_43159


namespace isosceles_triangle_equal_sides_length_l43_43855

noncomputable def equal_side_length_isosceles_triangle (base median : ℝ) (vertex_angle_deg : ℝ) : ℝ :=
  if base = 36 ∧ median = 15 ∧ vertex_angle_deg = 60 then 3 * Real.sqrt 191 else 0

theorem isosceles_triangle_equal_sides_length:
  equal_side_length_isosceles_triangle 36 15 60 = 3 * Real.sqrt 191 :=
by
  sorry

end isosceles_triangle_equal_sides_length_l43_43855


namespace f_equality_2019_l43_43698

theorem f_equality_2019 (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), f (m + n) ≥ f m + f (f n) - 1) : 
  f 2019 = 2019 :=
sorry

end f_equality_2019_l43_43698


namespace tournament_teams_matches_l43_43133

theorem tournament_teams_matches (teams : Fin 10 → ℕ) 
  (h : ∀ i, teams i ≤ 9) : 
  ∃ i j : Fin 10, i ≠ j ∧ teams i = teams j := 
by 
  sorry

end tournament_teams_matches_l43_43133


namespace product_of_primes_is_582_l43_43018

-- Define the relevant primes based on the conditions.
def smallest_one_digit_prime_1 := 2
def smallest_one_digit_prime_2 := 3
def largest_two_digit_prime := 97

-- Define the product of these primes as stated in the problem.
def product_of_primes := smallest_one_digit_prime_1 * smallest_one_digit_prime_2 * largest_two_digit_prime

-- Prove that this product equals to 582.
theorem product_of_primes_is_582 : product_of_primes = 582 :=
by {
  sorry
}

end product_of_primes_is_582_l43_43018


namespace f_odd_and_increasing_l43_43223

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end f_odd_and_increasing_l43_43223


namespace complement_of_A_in_U_l43_43235

open Set

variable (U : Set ℕ) (A : Set ℕ)

theorem complement_of_A_in_U (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 6}) :
  (U \ A) = {1, 3, 5} := by 
  sorry

end complement_of_A_in_U_l43_43235


namespace sum_max_min_on_interval_l43_43670

-- Defining the function f
def f (x : ℝ) : ℝ := x + 2

-- The proof statement
theorem sum_max_min_on_interval : 
  let M := max (f 0) (f 4)
  let N := min (f 0) (f 4)
  M + N = 8 := by
  -- Placeholder for proof
  sorry

end sum_max_min_on_interval_l43_43670


namespace quadrilateral_inscribed_circumscribed_l43_43141

theorem quadrilateral_inscribed_circumscribed 
  (r R d : ℝ) --Given variables with their types
  (K O : Type) (radius_K : K → ℝ) (radius_O : O → ℝ) (dist : (K × O) → ℝ)  -- Defining circles properties
  (K_inside_O : ∀ p : K × O, radius_K p.fst < radius_O p.snd) 
  (dist_centers : ∀ p : K × O, dist p = d) -- Distance between the centers
  : 
  (1 / (R + d)^2) + (1 / (R - d)^2) = (1 / r^2) := 
by 
  sorry

end quadrilateral_inscribed_circumscribed_l43_43141


namespace cheese_wedge_volume_l43_43473

theorem cheese_wedge_volume (r h : ℝ) (n : ℕ) (V : ℝ) (π : ℝ) 
: r = 8 → h = 10 → n = 3 → V = π * r^2 * h → V / n = (640 * π) / 3  :=
by
  intros r_eq h_eq n_eq V_eq
  rw [r_eq, h_eq] at V_eq
  rw [V_eq]
  sorry

end cheese_wedge_volume_l43_43473


namespace function_behaviour_l43_43217

theorem function_behaviour (a : ℝ) (h : a ≠ 0) :
  ¬ ((a * (-2)^2 + 2 * a * (-2) + 1 > a * (-1)^2 + 2 * a * (-1) + 1) ∧
     (a * (-1)^2 + 2 * a * (-1) + 1 > a * 0^2 + 2 * a * 0 + 1)) :=
by
  sorry

end function_behaviour_l43_43217


namespace necklaces_made_l43_43761

theorem necklaces_made (total_beads : ℕ) (beads_per_necklace : ℕ) (h1 : total_beads = 18) (h2 : beads_per_necklace = 3) : total_beads / beads_per_necklace = 6 := 
by {
  sorry
}

end necklaces_made_l43_43761


namespace pipe_length_l43_43209

theorem pipe_length (L x : ℝ) 
  (h1 : 20 = L - x)
  (h2 : 140 = L + 7 * x) : 
  L = 35 := by
  sorry

end pipe_length_l43_43209


namespace expected_heads_l43_43353

def coin_flips : Nat := 64

def prob_heads (tosses : ℕ) : ℚ :=
  1 / 2^(tosses + 1)

def total_prob_heads : ℚ :=
  prob_heads 0 + prob_heads 1 + prob_heads 2 + prob_heads 3

theorem expected_heads : (coin_flips : ℚ) * total_prob_heads = 60 := by
  sorry

end expected_heads_l43_43353


namespace meetings_percent_40_l43_43725

def percent_of_workday_in_meetings (workday_hours : ℕ) (first_meeting_min : ℕ) (second_meeting_min : ℕ) (third_meeting_min : ℕ) : ℕ :=
  (first_meeting_min + second_meeting_min + third_meeting_min) * 100 / (workday_hours * 60)

theorem meetings_percent_40 (workday_hours : ℕ) (first_meeting_min : ℕ) (second_meeting_min : ℕ) (third_meeting_min : ℕ)
  (h_workday : workday_hours = 10) 
  (h_first_meeting : first_meeting_min = 40) 
  (h_second_meeting : second_meeting_min = 2 * first_meeting_min) 
  (h_third_meeting : third_meeting_min = first_meeting_min + second_meeting_min) : 
  percent_of_workday_in_meetings workday_hours first_meeting_min second_meeting_min third_meeting_min = 40 :=
by
  sorry

end meetings_percent_40_l43_43725


namespace part1_part2_l43_43436

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

theorem part1 (x : ℝ) : f x ≥ 1 ↔ (x ≤ -5/2 ∨ x ≥ 3/2) :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end part1_part2_l43_43436


namespace football_hits_ground_l43_43835

theorem football_hits_ground :
  ∃ t : ℚ, -16 * t^2 + 18 * t + 60 = 0 ∧ 0 < t ∧ t = 41 / 16 :=
by
  sorry

end football_hits_ground_l43_43835


namespace total_boxes_packed_l43_43030

-- Definitions of the conditions
def initial_boxes : ℕ := 400
def food_value_per_box : ℕ := 80
def supplies_value_per_box : ℕ := 165
def donor_multiplier : ℕ := 4

-- Total value of one box
def total_value_per_box : ℕ := food_value_per_box + supplies_value_per_box

-- Initial spending
def initial_spending : ℕ := initial_boxes * total_value_per_box

-- Donation amount
def donation_amount : ℕ := donor_multiplier * initial_spending

-- Number of additional boxes packed with the donation
def additional_boxes : ℕ := donation_amount / total_value_per_box

-- Total number of boxes packed
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Statement to be proven
theorem total_boxes_packed : total_boxes = 2000 := by
  -- Proof for this theorem goes here...
  -- The proof is omitted in this statement as requested.
  sorry

end total_boxes_packed_l43_43030


namespace inv_func_eval_l43_43074

theorem inv_func_eval (a : ℝ) (h : 8^(1/3) = a) : (fun y => (Real.log y / Real.log 8)) (a + 2) = 2/3 :=
by
  sorry

end inv_func_eval_l43_43074


namespace true_statement_given_conditions_l43_43334

theorem true_statement_given_conditions (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a < b) :
  |1| / |a| > |1| / |b| := 
by
  sorry

end true_statement_given_conditions_l43_43334


namespace math_ineq_problem_l43_43948

variable (a b c : ℝ)

theorem math_ineq_problem
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : a + b + c ≤ 1)
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 :=
by
  sorry

end math_ineq_problem_l43_43948


namespace john_score_l43_43904

theorem john_score (s1 s2 s3 s4 s5 s6 : ℕ) (h1 : s1 = 85) (h2 : s2 = 88) (h3 : s3 = 90) (h4 : s4 = 92) (h5 : s5 = 83) (h6 : s6 = 102) :
  (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 90 :=
by
  sorry

end john_score_l43_43904


namespace age_of_son_l43_43339

theorem age_of_son (D S : ℕ) (h₁ : S = D / 4) (h₂ : D - S = 27) (h₃ : D = 36) : S = 9 :=
by
  sorry

end age_of_son_l43_43339


namespace ratio_of_friends_l43_43646

theorem ratio_of_friends (friends_in_classes friends_in_clubs : ℕ) (thread_per_keychain total_thread : ℕ) 
  (h1 : thread_per_keychain = 12) (h2 : friends_in_classes = 6) (h3 : total_thread = 108)
  (keychains_total : total_thread / thread_per_keychain = 9) 
  (keychains_clubs : (total_thread / thread_per_keychain) - friends_in_classes = friends_in_clubs) :
  friends_in_clubs / friends_in_classes = 1 / 2 :=
by
  sorry

end ratio_of_friends_l43_43646


namespace weight_units_correct_l43_43664

-- Definitions of weights
def weight_peanut_kernel := 1 -- gram
def weight_truck_capacity := 8 -- ton
def weight_xiao_ming := 30 -- kilogram
def weight_basketball := 580 -- gram

-- Proof that the weights have correct units
theorem weight_units_correct :
  (weight_peanut_kernel = 1 ∧ weight_truck_capacity = 8 ∧ weight_xiao_ming = 30 ∧ weight_basketball = 580) :=
by {
  sorry
}

end weight_units_correct_l43_43664


namespace mass_percentage_O_in_Al2_CO3_3_correct_l43_43243

noncomputable def mass_percentage_O_in_Al2_CO3_3 : ℚ := 
  let mass_O := 9 * 16.00
  let molar_mass_Al2_CO3_3 := (2 * 26.98) + (3 * 12.01) + (9 * 16.00)
  (mass_O / molar_mass_Al2_CO3_3) * 100

theorem mass_percentage_O_in_Al2_CO3_3_correct :
  mass_percentage_O_in_Al2_CO3_3 = 61.54 :=
by
  unfold mass_percentage_O_in_Al2_CO3_3
  sorry

end mass_percentage_O_in_Al2_CO3_3_correct_l43_43243


namespace combined_percent_increase_proof_l43_43779

variable (initial_stock_A_price : ℝ := 25)
variable (initial_stock_B_price : ℝ := 45)
variable (initial_stock_C_price : ℝ := 60)
variable (final_stock_A_price : ℝ := 28)
variable (final_stock_B_price : ℝ := 50)
variable (final_stock_C_price : ℝ := 75)

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

noncomputable def combined_percent_increase (initial_a initial_b initial_c final_a final_b final_c : ℝ) : ℝ :=
  (percent_increase initial_a final_a + percent_increase initial_b final_b + percent_increase initial_c final_c) / 3

theorem combined_percent_increase_proof :
  combined_percent_increase initial_stock_A_price initial_stock_B_price initial_stock_C_price
                            final_stock_A_price final_stock_B_price final_stock_C_price = 16.04 := by
  sorry

end combined_percent_increase_proof_l43_43779


namespace part_a_part_b_l43_43095

-- Define the function with the given conditions
variable {f : ℝ → ℝ}
variable (h_nonneg : ∀ x, 0 ≤ x → 0 ≤ f x)
variable (h_f1 : f 1 = 1)
variable (h_subadditivity : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

-- Part (a): Prove that f(x) ≤ 2x for all x ∈ [0, 1]
theorem part_a : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x :=
by
  sorry -- Proof required.

-- Part (b): Prove that it is not true that f(x) ≤ 1.9x for all x ∈ [0,1]
theorem part_b : ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ 1.9 * x < f x :=
by
  sorry -- Proof required.

end part_a_part_b_l43_43095


namespace part1_f1_part1_f2_part1_f3_part2_f_gt_1_for_n_1_2_part2_f_lt_1_for_n_ge_3_l43_43406

noncomputable def a (n : ℕ) : ℚ := 1 / (n : ℚ)

noncomputable def S (n : ℕ) : ℚ := (Finset.range (n+1)).sum (λ k => a (k + 1))

noncomputable def f (n : ℕ) : ℚ :=
  if n = 1 then S 2
  else S (2 * n) - S (n - 1)

theorem part1_f1 : f 1 = 3 / 2 := by sorry

theorem part1_f2 : f 2 = 13 / 12 := by sorry

theorem part1_f3 : f 3 = 19 / 20 := by sorry

theorem part2_f_gt_1_for_n_1_2 (n : ℕ) (h₁ : n = 1 ∨ n = 2) : f n > 1 := by sorry

theorem part2_f_lt_1_for_n_ge_3 (n : ℕ) (h₁ : n ≥ 3) : f n < 1 := by sorry

end part1_f1_part1_f2_part1_f3_part2_f_gt_1_for_n_1_2_part2_f_lt_1_for_n_ge_3_l43_43406


namespace net_change_in_collection_is_94_l43_43686

-- Definitions for the given conditions
def thrown_away_caps : Nat := 6
def initially_found_caps : Nat := 50
def additionally_found_caps : Nat := 44 + thrown_away_caps

-- Definition of the total found bottle caps
def total_found_caps : Nat := initially_found_caps + additionally_found_caps

-- Net change in Bottle Cap collection
def net_change_in_collection : Nat := total_found_caps - thrown_away_caps

-- Proof statement
theorem net_change_in_collection_is_94 : net_change_in_collection = 94 :=
by
  -- skipped proof
  sorry

end net_change_in_collection_is_94_l43_43686


namespace geometric_sequence_a4_l43_43827

theorem geometric_sequence_a4 (a : ℕ → ℝ) (q : ℝ) 
    (h_geom : ∀ n, a (n + 1) = a n * q)
    (h_a2 : a 2 = 1)
    (h_q : q = 2) : 
    a 4 = 4 :=
by
  -- Skip the proof as instructed
  sorry

end geometric_sequence_a4_l43_43827


namespace piravena_trip_total_cost_l43_43942

-- Define the distances
def d_A_to_B : ℕ := 4000
def d_B_to_C : ℕ := 3000

-- Define the costs per kilometer
def bus_cost_per_km : ℝ := 0.15
def airplane_cost_per_km : ℝ := 0.12
def airplane_booking_fee : ℝ := 120

-- Define the individual costs and the total cost
def cost_A_to_B : ℝ := d_A_to_B * airplane_cost_per_km + airplane_booking_fee
def cost_B_to_C : ℝ := d_B_to_C * bus_cost_per_km
def total_cost : ℝ := cost_A_to_B + cost_B_to_C

-- Define the theorem we want to prove
theorem piravena_trip_total_cost :
  total_cost = 1050 := sorry

end piravena_trip_total_cost_l43_43942


namespace lily_pad_growth_rate_l43_43329

theorem lily_pad_growth_rate 
  (day_37_covers_full : ℕ → ℝ)
  (day_36_covers_half : ℕ → ℝ)
  (exponential_growth : day_37_covers_full = 2 * day_36_covers_half) :
  (2 - 1) / 1 * 100 = 100 :=
by sorry

end lily_pad_growth_rate_l43_43329


namespace cs_competition_hits_l43_43057

theorem cs_competition_hits :
  (∃ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 ∧ x + y + z ≥ 1)
  ∧ (∀ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 ∧ x + y + z ≥ 1 → (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1)) :=
by
  sorry

end cs_competition_hits_l43_43057


namespace intersection_eq_l43_43262

def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def intersection : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_eq : M ∩ N = intersection := by
  sorry

end intersection_eq_l43_43262


namespace cities_drawn_from_group_b_l43_43909

def group_b_cities : ℕ := 8
def selection_probability : ℝ := 0.25

theorem cities_drawn_from_group_b : 
  group_b_cities * selection_probability = 2 :=
by
  sorry

end cities_drawn_from_group_b_l43_43909


namespace solve_for_s_l43_43251

theorem solve_for_s :
  let numerator := Real.sqrt (7^2 + 24^2)
  let denominator := Real.sqrt (64 + 36)
  let s := numerator / denominator
  s = 5 / 2 :=
by
  sorry

end solve_for_s_l43_43251


namespace real_solutions_count_l43_43218

-- Define the system of equations
def sys_eqs (x y z w : ℝ) :=
  (x = z + w + z * w * x) ∧
  (z = x + y + x * y * z) ∧
  (y = w + x + w * x * y) ∧
  (w = y + z + y * z * w)

-- The statement of the proof problem
theorem real_solutions_count : ∃ S : Finset (ℝ × ℝ × ℝ × ℝ), (∀ t : ℝ × ℝ × ℝ × ℝ, t ∈ S ↔ sys_eqs t.1 t.2.1 t.2.2.1 t.2.2.2) ∧ S.card = 5 :=
by {
  sorry
}

end real_solutions_count_l43_43218


namespace vector_addition_l43_43537

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (5, -7, 8)

-- State the theorem to prove 2a + b = (7, -3, 2)
theorem vector_addition : (2 • a + b) = (7, -3, 2) := by
  sorry

end vector_addition_l43_43537


namespace value_of_a4_l43_43118

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem value_of_a4 {a : ℕ → ℝ} {S : ℕ → ℝ} (h1 : arithmetic_sequence a)
  (h2 : sum_of_arithmetic_sequence S a) (h3 : S 7 = 28) :
  a 4 = 4 := 
  sorry

end value_of_a4_l43_43118


namespace vector_x_value_l43_43340

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def b : ℝ × ℝ := (1, 2)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_x_value (x : ℝ) : (perpendicular (a x) b) → x = -2 / 3 := by
  intro h
  sorry

end vector_x_value_l43_43340


namespace age_of_B_present_l43_43242

theorem age_of_B_present (A B C : ℕ) (h1 : A + B + C = 90)
  (h2 : (A - 10) * 2 = (B - 10))
  (h3 : (B - 10) * 3 = (C - 10) * 2) :
  B = 30 := 
sorry

end age_of_B_present_l43_43242


namespace maximum_value_parabola_l43_43520

theorem maximum_value_parabola (x : ℝ) : 
  ∃ y : ℝ, y = -3 * x^2 + 6 ∧ ∀ z : ℝ, (∃ a : ℝ, z = -3 * a^2 + 6) → z ≤ 6 :=
by
  sorry

end maximum_value_parabola_l43_43520


namespace number_of_students_l43_43013

def candiesPerStudent : ℕ := 2
def totalCandies : ℕ := 18
def expectedStudents : ℕ := 9

theorem number_of_students :
  totalCandies / candiesPerStudent = expectedStudents :=
sorry

end number_of_students_l43_43013


namespace even_function_m_eq_neg_one_l43_43321

theorem even_function_m_eq_neg_one (m : ℝ) :
  (∀ x : ℝ, (m - 1)*x^2 - (m^2 - 1)*x + (m + 2) = (m - 1)*(-x)^2 - (m^2 - 1)*(-x) + (m + 2)) →
  m = -1 :=
  sorry

end even_function_m_eq_neg_one_l43_43321


namespace total_number_of_students_l43_43869

theorem total_number_of_students 
  (b g : ℕ) 
  (ratio_condition : 5 * g = 8 * b) 
  (girls_count : g = 160) : 
  b + g = 260 := by
  sorry

end total_number_of_students_l43_43869


namespace simple_interest_sum_l43_43700

theorem simple_interest_sum (P_SI : ℕ) :
  let P_CI := 5000
  let r_CI := 12
  let t_CI := 2
  let r_SI := 10
  let t_SI := 5
  let CI := (P_CI * (1 + r_CI / 100)^t_CI - P_CI)
  let SI := CI / 2
  (P_SI * r_SI * t_SI / 100 = SI) -> 
  P_SI = 1272 := by {
  sorry
}

end simple_interest_sum_l43_43700


namespace empty_set_iff_k_single_element_set_iff_k_l43_43815

noncomputable def quadratic_set (k : ℝ) : Set ℝ := {x | k * x^2 - 3 * x + 2 = 0}

theorem empty_set_iff_k (k : ℝ) : 
  quadratic_set k = ∅ ↔ k > 9/8 := by
  sorry

theorem single_element_set_iff_k (k : ℝ) : 
  (∃ x : ℝ, quadratic_set k = {x}) ↔ (k = 0 ∧ quadratic_set k = {2 / 3}) ∨ (k = 9 / 8 ∧ quadratic_set k = {4 / 3}) := by
  sorry

end empty_set_iff_k_single_element_set_iff_k_l43_43815


namespace smaller_angle_at_9_am_l43_43546

-- Define the angular positions of the minute and hour hands
def minute_hand_angle (minute : Nat) : ℕ := 0  -- At the 12 position
def hour_hand_angle (hour : Nat) : ℕ := hour * 30  -- 30 degrees per hour

-- Define the function to get the smaller angle between two angles on the clock from 0 to 360 degrees
def smaller_angle (angle1 angle2 : ℕ) : ℕ :=
  let angle_diff := Int.natAbs (angle1 - angle2)
  min angle_diff (360 - angle_diff)

-- The theorem to prove
theorem smaller_angle_at_9_am : smaller_angle (minute_hand_angle 0) (hour_hand_angle 9) = 90 := sorry

end smaller_angle_at_9_am_l43_43546


namespace percent_absent_math_dept_l43_43150

theorem percent_absent_math_dept (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (male_absent_fraction : ℚ) (female_absent_fraction : ℚ)
  (h1 : total_students = 160) 
  (h2 : male_students = 90) 
  (h3 : female_students = 70) 
  (h4 : male_absent_fraction = 1 / 5) 
  (h5 : female_absent_fraction = 2 / 7) :
  ((male_absent_fraction * male_students + female_absent_fraction * female_students) / total_students) * 100 = 23.75 :=
by
  sorry

end percent_absent_math_dept_l43_43150


namespace smallest_n_l43_43234

def in_interval (x y z : ℝ) (n : ℕ) : Prop :=
  2 ≤ x ∧ x ≤ n ∧ 2 ≤ y ∧ y ≤ n ∧ 2 ≤ z ∧ z ≤ n

def no_two_within_one_unit (x y z : ℝ) : Prop :=
  abs (x - y) ≥ 1 ∧ abs (y - z) ≥ 1 ∧ abs (z - x) ≥ 1

def more_than_two_units_apart (x y z : ℝ) (n : ℕ) : Prop :=
  x > 2 ∧ x < n - 2 ∧ y > 2 ∧ y < n - 2 ∧ z > 2 ∧ z < n - 2

def probability_condition (n : ℕ) : Prop :=
  (n-4)^3 / (n-2)^3 > 1/3

theorem smallest_n (n : ℕ) : 11 = n → (∃ x y z : ℝ, in_interval x y z n ∧ no_two_within_one_unit x y z ∧ more_than_two_units_apart x y z n ∧ probability_condition n) :=
by
  sorry

end smallest_n_l43_43234


namespace ab_equals_six_l43_43065

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l43_43065


namespace tickets_used_l43_43246

def total_rides (ferris_wheel_rides bumper_car_rides : ℕ) : ℕ :=
  ferris_wheel_rides + bumper_car_rides

def tickets_per_ride : ℕ := 3

def total_tickets (total_rides tickets_per_ride : ℕ) : ℕ :=
  total_rides * tickets_per_ride

theorem tickets_used :
  total_tickets (total_rides 7 3) tickets_per_ride = 30 := by
  sorry

end tickets_used_l43_43246


namespace temperature_difference_l43_43395

variable (high_temp : ℝ) (low_temp : ℝ)

theorem temperature_difference (h1 : high_temp = 15) (h2 : low_temp = 7) : high_temp - low_temp = 8 :=
by {
  sorry
}

end temperature_difference_l43_43395


namespace andrew_total_days_l43_43668

noncomputable def hours_per_day : ℝ := 2.5
noncomputable def total_hours : ℝ := 7.5

theorem andrew_total_days : total_hours / hours_per_day = 3 := 
by 
  sorry

end andrew_total_days_l43_43668


namespace average_distance_is_600_l43_43829

-- Definitions based on the conditions
def one_lap_distance : ℕ := 200
def johnny_lap_count : ℕ := 4
def mickey_lap_count : ℕ := johnny_lap_count / 2

-- Total distances run by Johnny and Mickey
def johnny_distance : ℕ := johnny_lap_count * one_lap_distance
def mickey_distance : ℕ := mickey_lap_count * one_lap_distance

-- Sum of distances
def total_distance : ℕ := johnny_distance + mickey_distance

-- Number of people
def number_of_people : ℕ := 2

-- Average distance calculation
def average_distance : ℕ := total_distance / number_of_people

-- The theorem to prove: the average distance run by Johnny and Mickey
theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end average_distance_is_600_l43_43829


namespace successive_increases_eq_single_l43_43747

variable (P : ℝ)

def increase_by (initial : ℝ) (pct : ℝ) : ℝ := initial * (1 + pct)
def discount_by (initial : ℝ) (pct : ℝ) : ℝ := initial * (1 - pct)

theorem successive_increases_eq_single (P : ℝ) :
  increase_by (increase_by (discount_by (increase_by P 0.30) 0.10) 0.15) 0.20 = increase_by P 0.6146 :=
  sorry

end successive_increases_eq_single_l43_43747


namespace convex_cyclic_quadrilaterals_perimeter_40_l43_43201

theorem convex_cyclic_quadrilaterals_perimeter_40 :
  ∃ (n : ℕ), n = 750 ∧ ∀ (a b c d : ℕ), a + b + c + d = 40 → a ≥ b → b ≥ c → c ≥ d →
  (a < b + c + d) ∧ (b < a + c + d) ∧ (c < a + b + d) ∧ (d < a + b + c) :=
sorry

end convex_cyclic_quadrilaterals_perimeter_40_l43_43201


namespace amount_paid_l43_43449

def cost_cat_toy : ℝ := 8.77
def cost_cage : ℝ := 10.97
def change_received : ℝ := 0.26

theorem amount_paid : (cost_cat_toy + cost_cage + change_received) = 20.00 := by
  sorry

end amount_paid_l43_43449


namespace point_P_x_coordinate_l43_43740

variable {P : Type} [LinearOrderedField P]

-- Definitions from the conditions
def line_equation (x : P) : P := 0.8 * x
def y_coordinate_P : P := 6
def x_coordinate_P : P := 7.5

-- Theorems to prove that the x-coordinate of P is 7.5.
theorem point_P_x_coordinate (x : P) :
  line_equation x = y_coordinate_P → x = x_coordinate_P :=
by
  intro h
  sorry

end point_P_x_coordinate_l43_43740


namespace sum_of_coordinates_D_l43_43882

theorem sum_of_coordinates_D (M C D : ℝ × ℝ)
  (h1 : M = (5, 5))
  (h2 : C = (10, 10))
  (h3 : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 + D.2 = 0 := 
sorry

end sum_of_coordinates_D_l43_43882


namespace inequality_proof_l43_43177

theorem inequality_proof (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) : 
  1 + a^2 + b^2 > 3 * a * b := 
sorry

end inequality_proof_l43_43177


namespace average_people_moving_l43_43206

theorem average_people_moving (days : ℕ) (total_people : ℕ) 
    (h_days : days = 5) (h_total_people : total_people = 3500) : 
    (total_people / days) = 700 :=
by
  sorry

end average_people_moving_l43_43206


namespace cube_sum_identity_l43_43147

theorem cube_sum_identity (p q r : ℝ)
  (h₁ : p + q + r = 4)
  (h₂ : pq + qr + rp = 6)
  (h₃ : pqr = -8) :
  p^3 + q^3 + r^3 = 64 := 
by
  sorry

end cube_sum_identity_l43_43147


namespace students_per_bench_l43_43052

-- Definitions based on conditions
def num_male_students : ℕ := 29
def num_female_students : ℕ := 4 * num_male_students
def num_benches : ℕ := 29
def total_students : ℕ := num_male_students + num_female_students

-- Theorem to prove
theorem students_per_bench : total_students / num_benches = 5 := by
  sorry

end students_per_bench_l43_43052


namespace bill_weight_training_l43_43851

theorem bill_weight_training (jugs : ℕ) (gallons_per_jug : ℝ) (percent_filled : ℝ) (density : ℝ) 
  (h_jugs : jugs = 2)
  (h_gallons_per_jug : gallons_per_jug = 2)
  (h_percent_filled : percent_filled = 0.70)
  (h_density : density = 5) :
  jugs * gallons_per_jug * percent_filled * density = 14 := 
by
  subst h_jugs
  subst h_gallons_per_jug
  subst h_percent_filled
  subst h_density
  norm_num
  done

end bill_weight_training_l43_43851


namespace geom_series_sum_correct_l43_43644

noncomputable def geometric_series_sum (b1 r : ℚ) (n : ℕ) : ℚ :=
b1 * (1 - r ^ n) / (1 - r)

theorem geom_series_sum_correct :
  geometric_series_sum (3/4) (3/4) 15 = 3177905751 / 1073741824 := by
sorry

end geom_series_sum_correct_l43_43644


namespace problem_solution_l43_43599

-- Define the sets and the conditions given in the problem
def setA : Set ℝ := 
  {y | ∃ (x : ℝ), (x ∈ Set.Icc (3 / 4) 2) ∧ (y = x^2 - (3 / 2) * x + 1)}

def setB (m : ℝ) : Set ℝ := 
  {x | x + m^2 ≥ 1}

-- The proof statement contains two parts
theorem problem_solution (m : ℝ) :
  -- Part (I) - Prove the set A
  setA = Set.Icc (7 / 16) 2
  ∧
  -- Part (II) - Prove the range for m
  (∀ x, x ∈ setA → x ∈ setB m) → (m ≥ 3 / 4 ∨ m ≤ -3 / 4) :=
by
  sorry

end problem_solution_l43_43599


namespace profit_amount_calc_l43_43604

-- Define the conditions as hypotheses
variables (SP : ℝ) (profit_percent : ℝ) (cost_price profit_amount : ℝ)

-- Given conditions
axiom selling_price : SP = 900
axiom profit_percentage : profit_percent = 50
axiom profit_formula : profit_amount = 0.5 * cost_price
axiom selling_price_formula : SP = cost_price + profit_amount

-- The theorem to be proven
theorem profit_amount_calc : profit_amount = 300 :=
by
  sorry

end profit_amount_calc_l43_43604


namespace find_m_value_l43_43101

theorem find_m_value (m : ℤ) : (x^2 + m * x - 35 = (x - 7) * (x + 5)) → m = -2 :=
by
  sorry

end find_m_value_l43_43101


namespace sub_decimal_proof_l43_43979

theorem sub_decimal_proof : 2.5 - 0.32 = 2.18 :=
  by sorry

end sub_decimal_proof_l43_43979


namespace evaluate_fraction_sum_squared_l43_43016

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6

theorem evaluate_fraction_sum_squared :
  ( (1 / a + 1 / b + 1 / c + 1 / d)^2 = (11 + 2 * Real.sqrt 30) / 9 ) := 
by
  sorry

end evaluate_fraction_sum_squared_l43_43016


namespace cost_of_parts_l43_43155

theorem cost_of_parts (C : ℝ) 
  (h1 : ∀ n ∈ List.range 60, (1.4 * C * n) = (1.4 * C * 60))
  (h2 : 5000 + 3000 = 8000)
  (h3 : 60 * C * 1.4 - (60 * C + 8000) = 11200) : 
  C = 800 := by
  sorry

end cost_of_parts_l43_43155


namespace A_salary_less_than_B_by_20_percent_l43_43900

theorem A_salary_less_than_B_by_20_percent (A B : ℝ) (h1 : B = 1.25 * A) : 
  (B - A) / B * 100 = 20 :=
by
  sorry

end A_salary_less_than_B_by_20_percent_l43_43900


namespace min_value_of_f_l43_43989

noncomputable def f (a b x : ℝ) : ℝ :=
  (a / (Real.sin x) ^ 2) + b * (Real.sin x) ^ 2

theorem min_value_of_f (a b : ℝ) (h1 : a = 2) (h2 : b = 1) (h3 : a > b) (h4 : b > 0) :
  ∃ x, f a b x = 3 := 
sorry

end min_value_of_f_l43_43989


namespace percentage_assigned_exam_l43_43636

-- Define the conditions of the problem
def total_students : ℕ := 100
def average_assigned : ℝ := 0.55
def average_makeup : ℝ := 0.95
def average_total : ℝ := 0.67

-- Define the proof problem statement
theorem percentage_assigned_exam :
  ∃ (x : ℝ), (x / total_students) * average_assigned + ((total_students - x) / total_students) * average_makeup = average_total ∧ x = 70 :=
by
  sorry

end percentage_assigned_exam_l43_43636


namespace age_of_b_l43_43559

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 72) : b = 28 :=
by
  sorry

end age_of_b_l43_43559


namespace find_f_neg_3_l43_43881

theorem find_f_neg_3
    (a : ℝ)
    (f : ℝ → ℝ)
    (h : ∀ x, f x = a^2 * x^3 + a * Real.sin x + abs x + 1)
    (h_f3 : f 3 = 5) :
    f (-3) = 3 :=
by
    sorry

end find_f_neg_3_l43_43881


namespace sum_ab_system_1_l43_43478

theorem sum_ab_system_1 {a b : ℝ} 
  (h1 : a^3 - a^2 + a - 5 = 0) 
  (h2 : b^3 - 2*b^2 + 2*b + 4 = 0) : 
  a + b = 1 := 
by 
  sorry

end sum_ab_system_1_l43_43478


namespace square_three_times_side_length_l43_43635

theorem square_three_times_side_length (a : ℝ) : 
  ∃ s, s = a * Real.sqrt 3 ∧ s ^ 2 = 3 * a ^ 2 := 
by 
  sorry

end square_three_times_side_length_l43_43635


namespace intersection_complement_l43_43897

def M (x : ℝ) : Prop := x^2 - 2 * x < 0
def N (x : ℝ) : Prop := x < 1

theorem intersection_complement (x : ℝ) :
  (M x ∧ ¬N x) ↔ (1 ≤ x ∧ x < 2) := 
sorry

end intersection_complement_l43_43897


namespace value_of_f_at_3_l43_43190

noncomputable def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

theorem value_of_f_at_3 : f 3 = 155 := by
  sorry

end value_of_f_at_3_l43_43190


namespace dealership_vans_expected_l43_43937

theorem dealership_vans_expected (trucks vans : ℕ) (h_ratio : 3 * vans = 5 * trucks) (h_trucks : trucks = 45) : vans = 75 :=
by
  sorry

end dealership_vans_expected_l43_43937


namespace gcd_lcm_sum_l43_43435

-- Define the numbers and their prime factorizations
def a := 120
def b := 4620
def a_prime_factors := (2, 3) -- 2^3
def b_prime_factors := (2, 2) -- 2^2

-- Define gcd and lcm based on the problem statement
def gcd_ab := 60
def lcm_ab := 4620

-- The statement to be proved
theorem gcd_lcm_sum : gcd a b + lcm a b = 4680 :=
by sorry

end gcd_lcm_sum_l43_43435


namespace find_c_interval_l43_43676

theorem find_c_interval (c : ℚ) : 
  (c / 4 ≤ 3 + c ∧ 3 + c < -3 * (1 + c)) ↔ (-4 ≤ c ∧ c < -3 / 2) := 
by 
  sorry

end find_c_interval_l43_43676


namespace total_students_surveyed_l43_43184

-- Define the constants for liked and disliked students.
def liked_students : ℕ := 235
def disliked_students : ℕ := 165

-- The theorem to prove the total number of students surveyed.
theorem total_students_surveyed : liked_students + disliked_students = 400 :=
by
  -- The proof will go here.
  sorry

end total_students_surveyed_l43_43184


namespace distance_between_trees_l43_43940

theorem distance_between_trees (L : ℝ) (n : ℕ) (hL : L = 375) (hn : n = 26) : 
  (L / (n - 1) = 15) :=
by
  sorry

end distance_between_trees_l43_43940


namespace Zhu_Zaiyu_problem_l43_43470

theorem Zhu_Zaiyu_problem
  (f : ℕ → ℝ) 
  (q : ℝ)
  (h_geom_seq : ∀ n, f (n+1) = q * f n)
  (h_octave : f 13 = 2 * f 1) :
  (f 7) / (f 3) = 2^(1/3) :=
by
  sorry

end Zhu_Zaiyu_problem_l43_43470


namespace first_number_is_seven_l43_43348

variable (x y : ℝ)

theorem first_number_is_seven (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) : x = 7 :=
sorry

end first_number_is_seven_l43_43348


namespace find_smaller_number_l43_43544

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 84) (h2 : y = 3 * x) : x = 21 := 
by
  sorry

end find_smaller_number_l43_43544


namespace find_angle_y_l43_43216

-- Definitions of the angles in the triangle
def angle_ACD : ℝ := 90
def angle_DEB : ℝ := 58

-- Theorem proving the value of angle DCE (denoted as y)
theorem find_angle_y (angle_sum_property : angle_ACD + y + angle_DEB = 180) : y = 32 :=
by sorry

end find_angle_y_l43_43216


namespace percentage_of_3rd_graders_l43_43513

theorem percentage_of_3rd_graders (students_jackson students_madison : ℕ)
  (percent_3rd_grade_jackson percent_3rd_grade_madison : ℝ) :
  students_jackson = 200 → percent_3rd_grade_jackson = 25 →
  students_madison = 300 → percent_3rd_grade_madison = 35 →
  ((percent_3rd_grade_jackson / 100 * students_jackson +
    percent_3rd_grade_madison / 100 * students_madison) /
   (students_jackson + students_madison) * 100) = 31 :=
by 
  intros hjackson_percent hmpercent 
    hpercent_jack_percent hpercent_mad_percent
  -- Proof Placeholder
  sorry

end percentage_of_3rd_graders_l43_43513


namespace original_plan_was_to_produce_125_sets_per_day_l43_43078

-- We state our conditions
def plans_to_complete_in_days : ℕ := 30
def produces_sets_per_day : ℕ := 150
def finishes_days_ahead_of_schedule : ℕ := 5

-- Calculations based on conditions
def actual_days_used : ℕ := plans_to_complete_in_days - finishes_days_ahead_of_schedule
def total_production : ℕ := produces_sets_per_day * actual_days_used
def original_planned_production_per_day : ℕ := total_production / plans_to_complete_in_days

-- Claim we want to prove
theorem original_plan_was_to_produce_125_sets_per_day :
  original_planned_production_per_day = 125 :=
by
  sorry

end original_plan_was_to_produce_125_sets_per_day_l43_43078


namespace square_of_real_not_always_positive_l43_43739

theorem square_of_real_not_always_positive (a : ℝ) : ¬(a^2 > 0) := 
sorry

end square_of_real_not_always_positive_l43_43739


namespace martin_ratio_of_fruits_eaten_l43_43442

theorem martin_ratio_of_fruits_eaten
    (initial_fruits : ℕ)
    (current_oranges : ℕ)
    (current_oranges_twice_limes : current_oranges = 2 * (current_oranges / 2))
    (initial_fruits_count : initial_fruits = 150)
    (current_oranges_count : current_oranges = 50) :
    (initial_fruits - (current_oranges + (current_oranges / 2))) / initial_fruits = 1 / 2 := 
by
    sorry

end martin_ratio_of_fruits_eaten_l43_43442


namespace equation_elliptic_and_canonical_form_l43_43269

-- Defining the necessary conditions and setup
def a11 := 1
def a12 := 1
def a22 := 2

def is_elliptic (a11 a12 a22 : ℝ) : Prop :=
  a12^2 - a11 * a22 < 0

def canonical_form (u_xx u_xy u_yy u_x u_y u x y : ℝ) : Prop :=
  let ξ := y - x
  let η := x
  let u_ξξ := u_xx -- Assuming u_xx represents u_ξξ after change of vars
  let u_ξη := u_xy
  let u_ηη := u_yy
  let u_ξ := u_x -- Assuming u_x represents u_ξ after change of vars
  let u_η := u_y
  u_ξξ + u_ηη = -2 * u_η + u + η + (ξ + η)^2

theorem equation_elliptic_and_canonical_form (u_xx u_xy u_yy u_x u_y u x y : ℝ) :
  is_elliptic a11 a12 a22 ∧
  canonical_form u_xx u_xy u_yy u_x u_y u x y :=
by
  sorry -- Proof to be completed

end equation_elliptic_and_canonical_form_l43_43269


namespace correct_calculation_result_l43_43350

theorem correct_calculation_result :
  ∀ (A B D : ℝ),
  C = 6 →
  E = 5 →
  (A * 10 + B) * 6 + D * E = 39.6 ∨ (A * 10 + B) * 6 * D * E = 36.9 →
  (A * 10 + B) * 6 + D * E = 26.1 :=
by
  intros A B D C_eq E_eq errors
  sorry

end correct_calculation_result_l43_43350


namespace pencil_price_in_units_l43_43895

noncomputable def price_of_pencil_in_units (base_price additional_price unit_size : ℕ) : ℝ :=
  (base_price + additional_price) / unit_size

theorem pencil_price_in_units :
  price_of_pencil_in_units 5000 200 10000 = 0.52 := 
  by 
  sorry

end pencil_price_in_units_l43_43895


namespace speed_of_stream_l43_43374

theorem speed_of_stream (v_s : ℝ) (D : ℝ) (h1 : D / (78 - v_s) = 2 * (D / (78 + v_s))) : v_s = 26 :=
by
  sorry

end speed_of_stream_l43_43374


namespace distinct_values_of_b_l43_43819

theorem distinct_values_of_b : ∃ b_list : List ℝ, b_list.length = 8 ∧ ∀ b ∈ b_list, ∃ p q : ℤ, p + q = b ∧ p * q = 8 * b :=
by
  sorry

end distinct_values_of_b_l43_43819


namespace vinny_final_weight_l43_43444

theorem vinny_final_weight :
  let initial_weight := 300
  let first_month_loss := 20
  let second_month_loss := first_month_loss / 2
  let third_month_loss := second_month_loss / 2
  let fourth_month_loss := third_month_loss / 2
  let fifth_month_loss := 12
  let total_loss := first_month_loss + second_month_loss + third_month_loss + fourth_month_loss + fifth_month_loss
  let final_weight := initial_weight - total_loss
  final_weight = 250.5 :=
by
  sorry

end vinny_final_weight_l43_43444


namespace simplify_A_minus_B_value_of_A_minus_B_given_condition_l43_43462

variable (a b : ℝ)

def A := (a + b) ^ 2 - 3 * b ^ 2
def B := 2 * (a + b) * (a - b) - 3 * a * b

theorem simplify_A_minus_B :
  A a b - B a b = -a ^ 2 + 5 * a * b :=
by sorry

theorem value_of_A_minus_B_given_condition :
  (a - 3) ^ 2 + |b - 4| = 0 → A a b - B a b = 51 :=
by sorry

end simplify_A_minus_B_value_of_A_minus_B_given_condition_l43_43462


namespace find_number_l43_43924

theorem find_number (x : ℝ) : 0.40 * x = 0.80 * 5 + 2 → x = 15 :=
by
  intros h
  sorry

end find_number_l43_43924


namespace robe_initial_savings_l43_43117

noncomputable def initial_savings (repair_fee corner_light_cost brake_disk_cost tires_cost remaining_savings : ℕ) : ℕ :=
  remaining_savings + repair_fee + corner_light_cost + 2 * brake_disk_cost + tires_cost

theorem robe_initial_savings :
  let R := 10
  let corner_light := 2 * R
  let brake_disk := 3 * corner_light
  let tires := corner_light + 2 * brake_disk
  let remaining := 480
  initial_savings R corner_light brake_disk tires remaining = 770 :=
by
  sorry

end robe_initial_savings_l43_43117


namespace silvia_escalator_time_l43_43120

noncomputable def total_time_standing (v s : ℝ) : ℝ := 
  let d := 80 * v
  d / s

theorem silvia_escalator_time (v s t : ℝ) (h1 : 80 * v = 28 * (v + s)) (h2 : t = total_time_standing v s) : 
  t = 43 := by
  sorry

end silvia_escalator_time_l43_43120


namespace bah_to_yah_conversion_l43_43405

theorem bah_to_yah_conversion :
  (10 : ℝ) * (1500 * (3/5) * (10/16)) / 16 = 562.5 := by
sorry

end bah_to_yah_conversion_l43_43405


namespace bushes_for_60_zucchinis_l43_43468

/-- 
Given:
1. Each blueberry bush yields twelve containers of blueberries.
2. Four containers of blueberries can be traded for three pumpkins.
3. Six pumpkins can be traded for five zucchinis.

Prove that eight bushes are needed to harvest 60 zucchinis.
-/
theorem bushes_for_60_zucchinis (bush_to_containers : ℕ) (containers_to_pumpkins : ℕ) (pumpkins_to_zucchinis : ℕ) :
  (bush_to_containers = 12) → (containers_to_pumpkins = 4) → (pumpkins_to_zucchinis = 6) →
  ∃ bushes_needed, bushes_needed = 8 ∧ (60 * pumpkins_to_zucchinis / 5 * containers_to_pumpkins / 3 / bush_to_containers) = bushes_needed :=
by
  intros h1 h2 h3
  sorry

end bushes_for_60_zucchinis_l43_43468


namespace estimated_watched_students_l43_43496

-- Definitions for the problem conditions
def total_students : ℕ := 3600
def surveyed_students : ℕ := 200
def watched_students : ℕ := 160

-- Problem statement (proof not included yet)
theorem estimated_watched_students :
  total_students * (watched_students / surveyed_students : ℝ) = 2880 := by
  -- skipping proof step
  sorry

end estimated_watched_students_l43_43496


namespace color_column_l43_43006

theorem color_column (n : ℕ) (color : ℕ) (board : ℕ → ℕ → ℕ) 
  (h_colors : ∀ i j, 1 ≤ board i j ∧ board i j ≤ n^2)
  (h_block : ∀ i j, (∀ k l : ℕ, k < n → l < n → ∃ c, ∀ a b : ℕ, k + a * n < n → l + b * n < n → board (i + k + a * n) (j + l + b * n) = c))
  (h_row : ∃ r, ∀ k, k < n → ∃ c, 1 ≤ c ∧ c ≤ n ∧ board r k = c) :
  ∃ c, (∀ j, 1 ≤ board c j ∧ board c j ≤ n) :=
sorry

end color_column_l43_43006


namespace recurring_decimal_sum_l43_43762

-- Definitions based on the conditions identified
def recurringDecimal (n : ℕ) : ℚ := n / 9
def r8 := recurringDecimal 8
def r2 := recurringDecimal 2
def r6 := recurringDecimal 6
def r6_simplified : ℚ := 2 / 3

-- The theorem to prove
theorem recurring_decimal_sum : r8 + r2 - r6_simplified = 4 / 9 :=
by
  -- Proof steps will go here (but are omitted because of the problem requirements)
  sorry

end recurring_decimal_sum_l43_43762


namespace part1_part2_l43_43532

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k * k = n

def calculate_P (x y : ℤ) : ℤ := 
  (x - y) / 9

def y_from_x (x : ℤ) : ℤ :=
  let first_three := x / 10
  let last_digit := x % 10
  last_digit * 1000 + first_three

def calculate_s (a b : ℕ) : ℤ :=
  1100 + 20 * a + b

def calculate_t (a b : ℕ) : ℤ :=
  b * 1000 + a * 100 + 23

theorem part1 : calculate_P 5324 (y_from_x 5324) = 88 := by
  sorry

theorem part2 :
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 9 ∧
  let s := calculate_s a b
  let t := calculate_t a b
  let P_s := calculate_P s (y_from_x s)
  let P_t := calculate_P t (y_from_x t)
  let difference := P_t - P_s - a - b
  is_perfect_square difference ∧ P_t = -161 := by
  sorry

end part1_part2_l43_43532


namespace maya_total_pages_l43_43491

def books_first_week : ℕ := 5
def pages_per_book_first_week : ℕ := 300
def books_second_week := books_first_week * 2
def pages_per_book_second_week : ℕ := 350
def books_third_week := books_first_week * 3
def pages_per_book_third_week : ℕ := 400

def total_pages_first_week : ℕ := books_first_week * pages_per_book_first_week
def total_pages_second_week : ℕ := books_second_week * pages_per_book_second_week
def total_pages_third_week : ℕ := books_third_week * pages_per_book_third_week

def total_pages_maya_read : ℕ := total_pages_first_week + total_pages_second_week + total_pages_third_week

theorem maya_total_pages : total_pages_maya_read = 11000 := by
  sorry

end maya_total_pages_l43_43491


namespace find_a_l43_43991

noncomputable def pure_imaginary_simplification (a : ℝ) (i : ℂ) (hi : i * i = -1) : Prop :=
  let denom := (3 : ℂ) - (4 : ℂ) * i
  let numer := (15 : ℂ)
  let complex_num := a + numer / denom
  let simplified_real := a + (9 : ℝ) / (5 : ℝ)
  simplified_real = 0

theorem find_a (i : ℂ) (hi : i * i = -1) : pure_imaginary_simplification (- 9 / 5 : ℝ) i hi :=
by
  sorry

end find_a_l43_43991


namespace value_of_expression_l43_43000

theorem value_of_expression (x1 x2 : ℝ) 
  (h1 : x1 ^ 2 - 3 * x1 - 4 = 0) 
  (h2 : x2 ^ 2 - 3 * x2 - 4 = 0)
  (h3 : x1 + x2 = 3) 
  (h4 : x1 * x2 = -4) : 
  x1 ^ 2 - 4 * x1 - x2 + 2 * x1 * x2 = -7 := by
  sorry

end value_of_expression_l43_43000


namespace inequality_proof_l43_43140

theorem inequality_proof
  (a b c d e f : ℝ)
  (h : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
sorry

end inequality_proof_l43_43140


namespace kitten_current_length_l43_43632

theorem kitten_current_length (initial_length : ℕ) (double_after_2_weeks : ℕ → ℕ) (double_after_4_months : ℕ → ℕ)
  (h1 : initial_length = 4)
  (h2 : double_after_2_weeks initial_length = 2 * initial_length)
  (h3 : double_after_4_months (double_after_2_weeks initial_length) = 2 * (double_after_2_weeks initial_length)) :
  double_after_4_months (double_after_2_weeks initial_length) = 16 := 
by
  sorry

end kitten_current_length_l43_43632


namespace village_population_l43_43404

theorem village_population (initial_population: ℕ) (died_percent left_percent: ℕ) (remaining_population current_population: ℕ)
    (h1: initial_population = 6324)
    (h2: died_percent = 10)
    (h3: left_percent = 20)
    (h4: remaining_population = initial_population - (initial_population * died_percent / 100))
    (h5: current_population = remaining_population - (remaining_population * left_percent / 100)):
  current_population = 4554 :=
  by
    sorry

end village_population_l43_43404


namespace problem_1_problem_2_l43_43852

noncomputable def is_positive_real (x : ℝ) : Prop := x > 0

theorem problem_1 (a b : ℝ) (ha : is_positive_real a) (hb : is_positive_real b)
  (h : 1 / a + 1 / b = 2 * Real.sqrt 2) : 
  a^2 + b^2 ≥ 1 := by
  sorry

theorem problem_2 (a b : ℝ) (ha : is_positive_real a) (hb : is_positive_real b)
  (h : 1 / a + 1 / b = 2 * Real.sqrt 2) (h_extra : (a - b)^2 ≥ 4 * (a * b)^3) : 
  a * b = 1 := by
  sorry

end problem_1_problem_2_l43_43852


namespace base9_4318_is_base10_3176_l43_43007

def base9_to_base10 (n : Nat) : Nat :=
  let d₀ := (n % 10) * 9^0
  let d₁ := ((n / 10) % 10) * 9^1
  let d₂ := ((n / 100) % 10) * 9^2
  let d₃ := ((n / 1000) % 10) * 9^3
  d₀ + d₁ + d₂ + d₃

theorem base9_4318_is_base10_3176 :
  base9_to_base10 4318 = 3176 :=
by
  sorry

end base9_4318_is_base10_3176_l43_43007


namespace find_a_5_l43_43165

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem find_a_5 {a : ℕ → ℤ} {S : ℕ → ℤ}
  (h_seq : arithmetic_sequence a)
  (h_S6 : S 6 = 3)
  (h_a4 : a 4 = 2)
  (h_sum_first_n : sum_first_n a S) :
  a 5 = 5 := 
sorry

end find_a_5_l43_43165


namespace original_people_in_room_l43_43543

theorem original_people_in_room (x : ℕ) (h1 : 18 = (2 * x / 3) - (x / 6)) : x = 36 :=
by sorry

end original_people_in_room_l43_43543


namespace C_younger_than_A_l43_43950

variables (A B C : ℕ)

-- Original Condition
axiom age_condition : A + B = B + C + 17

-- Lean Statement to Prove
theorem C_younger_than_A (A B C : ℕ) (h : A + B = B + C + 17) : C + 17 = A :=
by {
  -- Proof would go here but is omitted.
  sorry
}

end C_younger_than_A_l43_43950


namespace ratio_of_y_to_x_l43_43853

theorem ratio_of_y_to_x (c x y : ℝ) (hx : x = 0.90 * c) (hy : y = 1.20 * c) :
  y / x = 4 / 3 := 
sorry

end ratio_of_y_to_x_l43_43853


namespace min_k_spherical_cap_cylinder_l43_43565

/-- Given a spherical cap and a cylinder sharing a common inscribed sphere with volumes V1 and V2 respectively,
we show that the minimum value of k such that V1 = k * V2 is 4/3. -/
theorem min_k_spherical_cap_cylinder (R : ℝ) (V1 V2 : ℝ) (h1 : V1 = (4/3) * π * R^3) 
(h2 : V2 = 2 * π * R^3) : 
∃ k : ℝ, V1 = k * V2 ∧ k = 4/3 := 
by 
  use (4/3)
  constructor
  . sorry
  . sorry

end min_k_spherical_cap_cylinder_l43_43565


namespace total_families_l43_43983

theorem total_families (F_2dogs F_1dog F_2cats total_animals total_families : ℕ) 
  (h1: F_2dogs = 15)
  (h2: F_1dog = 20)
  (h3: total_animals = 80)
  (h4: 2 * F_2dogs + F_1dog + 2 * F_2cats = total_animals) :
  total_families = F_2dogs + F_1dog + F_2cats := 
by 
  sorry

end total_families_l43_43983


namespace percentage_less_than_a_plus_d_l43_43224

def symmetric_distribution (a d : ℝ) (p : ℝ) : Prop :=
  p = (68 / 100 : ℝ) ∧ 
  (p / 2) = (34 / 100 : ℝ)

theorem percentage_less_than_a_plus_d (a d : ℝ) 
  (symmetry : symmetric_distribution a d (68 / 100)) : 
  (0.5 + (34 / 100) : ℝ) = (84 / 100 : ℝ) :=
by
  sorry

end percentage_less_than_a_plus_d_l43_43224


namespace gcd_a_b_eq_one_l43_43516

def a : ℕ := 47^5 + 1
def b : ℕ := 47^5 + 47^3 + 1

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 := by
  sorry

end gcd_a_b_eq_one_l43_43516


namespace has_exactly_one_zero_interval_l43_43985

noncomputable def f (a x : ℝ) : ℝ := x^2 - a*x + 1

theorem has_exactly_one_zero_interval (a : ℝ) (h : a > 3) : ∃! x, 0 < x ∧ x < 2 ∧ f a x = 0 :=
sorry

end has_exactly_one_zero_interval_l43_43985


namespace least_number_divisible_increased_by_seven_l43_43589

theorem least_number_divisible_increased_by_seven : 
  ∃ n : ℕ, (∀ k ∈ [24, 32, 36, 54], (n + 7) % k = 0) ∧ n = 857 := 
by
  sorry

end least_number_divisible_increased_by_seven_l43_43589


namespace truncated_cone_sphere_radius_l43_43518

theorem truncated_cone_sphere_radius :
  ∀ (r1 r2 h : ℝ), 
  r1 = 24 → 
  r2 = 6 → 
  h = 20 → 
  ∃ r, 
  r = 17 * Real.sqrt 2 / 2 := by
  intros r1 r2 h hr1 hr2 hh
  sorry

end truncated_cone_sphere_radius_l43_43518


namespace shampoo_duration_l43_43977

-- Conditions
def rose_shampoo : ℚ := 1/3
def jasmine_shampoo : ℚ := 1/4
def daily_usage : ℚ := 1/12

-- Question
theorem shampoo_duration : (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end shampoo_duration_l43_43977


namespace man_l43_43726

noncomputable def man's_rate_in_still_water (downstream upstream : ℝ) : ℝ :=
  (downstream + upstream) / 2

theorem man's_rate_correct :
  let downstream := 6
  let upstream := 3
  man's_rate_in_still_water downstream upstream = 4.5 :=
by
  sorry

end man_l43_43726


namespace Pyarelal_loss_l43_43960

variables (capital_of_pyarelal capital_of_ashok : ℝ) (total_loss : ℝ)

def is_ninth (a b : ℝ) : Prop := a = b / 9

def applied_loss (loss : ℝ) (ratio : ℝ) : ℝ := ratio * loss

theorem Pyarelal_loss (h1: is_ninth capital_of_ashok capital_of_pyarelal) 
                        (h2: total_loss = 1600) : 
                        applied_loss total_loss (9/10) = 1440 :=
by 
  unfold is_ninth at h1
  sorry

end Pyarelal_loss_l43_43960


namespace polygon_properties_l43_43286

-- Assume n is the number of sides of the polygon
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_of_exterior_angles : ℝ := 360

-- Given the condition
def given_condition (n : ℕ) : Prop := sum_of_interior_angles n = 5 * sum_of_exterior_angles

theorem polygon_properties (n : ℕ) (h1 : given_condition n) :
  n = 12 ∧ (n * (n - 3)) / 2 = 54 :=
by
  sorry

end polygon_properties_l43_43286


namespace gift_box_spinning_tops_l43_43970

theorem gift_box_spinning_tops
  (red_box_cost : ℕ) (red_box_tops : ℕ)
  (yellow_box_cost : ℕ) (yellow_box_tops : ℕ)
  (total_spent : ℕ) (total_boxes : ℕ)
  (h_red_box_cost : red_box_cost = 5)
  (h_red_box_tops : red_box_tops = 3)
  (h_yellow_box_cost : yellow_box_cost = 9)
  (h_yellow_box_tops : yellow_box_tops = 5)
  (h_total_spent : total_spent = 600)
  (h_total_boxes : total_boxes = 72) :
  ∃ (red_boxes : ℕ) (yellow_boxes : ℕ), (red_boxes + yellow_boxes = total_boxes) ∧
  (red_box_cost * red_boxes + yellow_box_cost * yellow_boxes = total_spent) ∧
  (red_box_tops * red_boxes + yellow_box_tops * yellow_boxes = 336) :=
by
  sorry

end gift_box_spinning_tops_l43_43970


namespace median_a_sq_correct_sum_of_medians_sq_l43_43168

noncomputable def median_a_sq (a b c : ℝ) := (2 * b^2 + 2 * c^2 - a^2) / 4
noncomputable def median_b_sq (a b c : ℝ) := (2 * a^2 + 2 * c^2 - b^2) / 4
noncomputable def median_c_sq (a b c : ℝ) := (2 * a^2 + 2 * b^2 - c^2) / 4

theorem median_a_sq_correct (a b c : ℝ) : 
  median_a_sq a b c = (2 * b^2 + 2 * c^2 - a^2) / 4 :=
sorry

theorem sum_of_medians_sq (a b c : ℝ) :
  median_a_sq a b c + median_b_sq a b c + median_c_sq a b c = 
  3 * (a^2 + b^2 + c^2) / 4 :=
sorry

end median_a_sq_correct_sum_of_medians_sq_l43_43168


namespace number_equals_fifty_l43_43847

def thirty_percent_less_than_ninety : ℝ := 0.7 * 90

theorem number_equals_fifty (x : ℝ) (h : (5 / 4) * x = thirty_percent_less_than_ninety) : x = 50 :=
by
  sorry

end number_equals_fifty_l43_43847


namespace shortest_remaining_side_l43_43488

theorem shortest_remaining_side (a b : ℝ) (h1 : a = 7) (h2 : b = 24) (right_triangle : ∃ c, c^2 = a^2 + b^2) : a = 7 :=
by
  sorry

end shortest_remaining_side_l43_43488


namespace k_gt_4_l43_43455

theorem k_gt_4 {x y k : ℝ} (h1 : 2 * x + y = 2 * k - 1) (h2 : x + 2 * y = -4) (h3 : x + y > 1) : k > 4 :=
by
  -- This 'sorry' serves as a placeholder for the actual proof steps
  sorry

end k_gt_4_l43_43455


namespace parabola_no_intersection_inequality_l43_43498

-- Definitions for the problem
theorem parabola_no_intersection_inequality
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, (a * x^2 + b * x + c ≠ x) ∧ (a * x^2 + b * x + c ≠ -x)) :
  |b^2 - 4 * a * c| > 1 := 
sorry

end parabola_no_intersection_inequality_l43_43498


namespace permutation_by_transpositions_l43_43763

-- Formalizing the conditions in Lean
section permutations
  variable {n : ℕ}

  -- Define permutations
  def is_permutation (σ : Fin n → Fin n) : Prop :=
    ∃ σ_inv : Fin n → Fin n, 
      (∀ i, σ (σ_inv i) = i) ∧ 
      (∀ i, σ_inv (σ i) = i)

  -- Define transposition
  def transposition (σ : Fin n → Fin n) (i j : Fin n) : Fin n → Fin n :=
    fun x => if x = i then j else if x = j then i else σ x

  -- Main theorem stating that any permutation can be obtained through a series of transpositions
  theorem permutation_by_transpositions (σ : Fin n → Fin n) (h : is_permutation σ) :
    ∃ τ : ℕ → (Fin n → Fin n),
      (∀ i, is_permutation (τ i)) ∧
      (∀ m, ∃ k, τ m = transposition (τ (m - 1)) (⟨ k, sorry ⟩) (σ (⟨ k, sorry⟩))) ∧
      (∃ m, τ m = σ) :=
  sorry
end permutations

end permutation_by_transpositions_l43_43763


namespace jasmine_money_left_l43_43542

theorem jasmine_money_left 
  (initial_amount : ℝ)
  (apple_cost : ℝ) (num_apples : ℕ)
  (orange_cost : ℝ) (num_oranges : ℕ)
  (pear_cost : ℝ) (num_pears : ℕ)
  (h_initial : initial_amount = 100.00)
  (h_apple_cost : apple_cost = 1.50)
  (h_num_apples : num_apples = 5)
  (h_orange_cost : orange_cost = 2.00)
  (h_num_oranges : num_oranges = 10)
  (h_pear_cost : pear_cost = 2.25)
  (h_num_pears : num_pears = 4) : 
  initial_amount - (num_apples * apple_cost + num_oranges * orange_cost + num_pears * pear_cost) = 63.50 := 
by 
  sorry

end jasmine_money_left_l43_43542


namespace min_value_z_l43_43134

theorem min_value_z : ∀ (x y : ℝ), ∃ z, z = 3 * x^2 + y^2 + 12 * x - 6 * y + 40 ∧ z = 19 :=
by
  intro x y
  use 3 * x^2 + y^2 + 12 * x - 6 * y + 40 -- Define z
  sorry -- Proof is skipped for now

end min_value_z_l43_43134


namespace solve_for_y_l43_43369

-- Define the conditions and the goal to prove in Lean 4
theorem solve_for_y
  (x y : ℤ) 
  (h1 : x + y = 250) 
  (h2 : x - y = 200) : 
  y = 25 :=
by
  sorry

end solve_for_y_l43_43369


namespace curve_passes_through_fixed_point_l43_43331

theorem curve_passes_through_fixed_point (m n : ℝ) :
  (2:ℝ)^2 + (-2:ℝ)^2 - 2 * m * (2:ℝ) - 2 * n * (-2:ℝ) + 4 * (m - n - 2) = 0 :=
by sorry

end curve_passes_through_fixed_point_l43_43331


namespace math_problem_l43_43067

theorem math_problem : 
  (Real.sqrt 4) * (4 ^ (1 / 2: ℝ)) + (16 / 4) * 2 - (8 ^ (1 / 2: ℝ)) = 12 - 2 * Real.sqrt 2 :=
by
  sorry

end math_problem_l43_43067


namespace int_div_condition_l43_43752

theorem int_div_condition (n : ℕ) (hn₁ : ∃ m : ℤ, 2^n - 2 = m * n) :
  ∃ k : ℤ, 2^(2^n - 1) - 2 = k * (2^n - 1) :=
by sorry

end int_div_condition_l43_43752


namespace max_regions_divided_l43_43257

theorem max_regions_divided (n m : ℕ) (h_n : n = 10) (h_m : m = 4) (h_m_le_n : m ≤ n) : 
  ∃ r : ℕ, r = 50 :=
by
  have non_parallel_lines := n - m
  have regions_non_parallel := (non_parallel_lines * (non_parallel_lines + 1)) / 2 + 1
  have regions_parallel := m * non_parallel_lines + m
  have total_regions := regions_non_parallel + regions_parallel
  use total_regions
  sorry

end max_regions_divided_l43_43257


namespace sum_le_two_of_cubics_sum_to_two_l43_43174

theorem sum_le_two_of_cubics_sum_to_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) : a + b ≤ 2 := 
sorry

end sum_le_two_of_cubics_sum_to_two_l43_43174


namespace fraction_of_milk_in_second_cup_l43_43920

noncomputable def ratio_mixture (V: ℝ) (x: ℝ) :=
  ((2 / 5 * V + (1 - x) * V) / (3 / 5 * V + x * V))

theorem fraction_of_milk_in_second_cup
  (V: ℝ) 
  (hV: V > 0)
  (hx: ratio_mixture V x = 3 / 7) :
  x = 4 / 5 :=
by
  sorry

end fraction_of_milk_in_second_cup_l43_43920


namespace smallest_divisor_l43_43953

noncomputable def even_four_digit_number (m : ℕ) : Prop :=
  1000 ≤ m ∧ m < 10000 ∧ m % 2 = 0

def divisor_ordered (m : ℕ) (d : ℕ) : Prop :=
  d ∣ m

theorem smallest_divisor (m : ℕ) (h1 : even_four_digit_number m) (h2 : divisor_ordered m 437) :
  ∃ d,  d > 437 ∧ divisor_ordered m d ∧ (∀ e, e > 437 → divisor_ordered m e → d ≤ e) ∧ d = 874 :=
sorry

end smallest_divisor_l43_43953


namespace solution_set_of_inequality_l43_43228

theorem solution_set_of_inequality 
  {f : ℝ → ℝ}
  (hf : ∀ x y : ℝ, x < y → f x > f y)
  (hA : f 0 = -2)
  (hB : f (-3) = 2) :
  {x : ℝ | |f (x - 2)| > 2 } = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end solution_set_of_inequality_l43_43228


namespace trig_expression_value_l43_43199

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 :=
sorry

end trig_expression_value_l43_43199


namespace students_at_1544_l43_43181

noncomputable def students_in_lab : Nat := 44

theorem students_at_1544 :
  let initial_students := 20
  let enter_interval := 3
  let enter_students := 4
  let leave_interval := 10
  let leave_students := 8

  ∃ students : Nat,
    students = initial_students
    + (34 / enter_interval) * enter_students
    - (34 / leave_interval) * leave_students
    ∧ students = students_in_lab :=
by
  let initial_students := 20
  let enter_interval := 3
  let enter_students := 4
  let leave_interval := 10
  let leave_students := 8
  use 20 + (34 / 3) * 4 - (34 / 10) * 8
  sorry

end students_at_1544_l43_43181


namespace sum_of_decimals_as_fraction_l43_43984

/-- Define the problem inputs as constants -/
def d1 : ℚ := 2 / 10
def d2 : ℚ := 4 / 100
def d3 : ℚ := 6 / 1000
def d4 : ℚ := 8 / 10000
def d5 : ℚ := 1 / 100000

/-- The main theorem statement -/
theorem sum_of_decimals_as_fraction : 
  d1 + d2 + d3 + d4 + d5 = 24681 / 100000 := 
by 
  sorry

end sum_of_decimals_as_fraction_l43_43984


namespace find_a_l43_43151

theorem find_a :
  ∃ a : ℝ, 
    (∀ x : ℝ, f x = 3 * x + a * x^3) ∧ 
    (f 1 = a + 3) ∧ 
    (∃ k : ℝ, k = 6 ∧ k = deriv f 1 ∧ ((∀ x : ℝ, deriv f x = 3 + 3 * a * x^2))) → 
    a = 1 :=
by sorry

end find_a_l43_43151


namespace tree_heights_l43_43296

theorem tree_heights (T S : ℕ) (h1 : T - S = 20) (h2 : T - 10 = 3 * (S - 10)) : T = 40 := 
by
  sorry

end tree_heights_l43_43296


namespace problem_l43_43573

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem problem (h₁ : f a b c 0 = f a b c 4) (h₂ : f a b c 4 > f a b c 1) : a > 0 ∧ 4 * a + b = 0 :=
by 
  sorry

end problem_l43_43573


namespace avg_of_numbers_l43_43075

theorem avg_of_numbers (a b c d : ℕ) (avg : ℕ) (h₁ : a = 6) (h₂ : b = 16) (h₃ : c = 8) (h₄ : d = 22) (h₅ : avg = 13) :
  (a + b + c + d) / 4 = avg := by
  -- Proof here
  sorry

end avg_of_numbers_l43_43075


namespace ab_plus_cd_eq_neg_346_over_9_l43_43556

theorem ab_plus_cd_eq_neg_346_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := 
sorry

end ab_plus_cd_eq_neg_346_over_9_l43_43556


namespace sin_from_tan_l43_43182

theorem sin_from_tan (A : ℝ) (h : Real.tan A = Real.sqrt 2 / 3) : 
  Real.sin A = Real.sqrt 22 / 11 := 
by 
  sorry

end sin_from_tan_l43_43182


namespace cost_price_USD_l43_43581

-- Assume the conditions in Lean as given:
variable {C_USD : ℝ}

def condition1 (C_USD : ℝ) : Prop := 0.9 * C_USD + 200 = 1.04 * C_USD

theorem cost_price_USD (h : condition1 C_USD) : C_USD = 200 / 0.14 :=
by
  sorry

end cost_price_USD_l43_43581


namespace total_price_correct_l43_43062

-- Define the initial price, reduction, and the number of boxes
def initial_price : ℝ := 104
def price_reduction : ℝ := 24
def number_of_boxes : ℕ := 20

-- Define the new price as initial price minus the reduction
def new_price := initial_price - price_reduction

-- Define the total price as the new price times the number of boxes
def total_price := (number_of_boxes : ℝ) * new_price

-- The goal is to prove the total price equals 1600
theorem total_price_correct : total_price = 1600 := by
  sorry

end total_price_correct_l43_43062


namespace find_x_minus_y_l43_43699

variables (x y z : ℝ)

theorem find_x_minus_y (h1 : x - (y + z) = 19) (h2 : x - y - z = 7): x - y = 13 :=
by {
  sorry
}

end find_x_minus_y_l43_43699


namespace medal_award_count_l43_43424

theorem medal_award_count :
  let total_runners := 10
  let canadian_runners := 4
  let medals := 3
  let non_canadian_runners := total_runners - canadian_runners
  let no_canadians_get_medals := Nat.choose non_canadian_runners medals * Nat.factorial medals
  let one_canadian_gets_medal := canadian_runners * medals * (Nat.choose non_canadian_runners (medals - 1) * Nat.factorial (medals - 1))
  no_canadians_get_medals + one_canadian_gets_medal = 480 :=
by
  let total_runners := 10
  let canadian_runners := 4
  let medals := 3
  let non_canadian_runners := total_runners - canadian_runners
  let no_canadians_get_medals := Nat.choose non_canadian_runners medals * Nat.factorial medals
  let one_canadian_gets_medal := canadian_runners * medals * (Nat.choose non_canadian_runners (medals - 1) * Nat.factorial (medals - 1))
  show no_canadians_get_medals + one_canadian_gets_medal = 480
  -- here should be the steps skipped
  sorry

end medal_award_count_l43_43424


namespace triangle_area_with_median_l43_43820

theorem triangle_area_with_median (a b m : ℝ) (area : ℝ) 
  (h_a : a = 6) (h_b : b = 8) (h_m : m = 5) : 
  area = 24 :=
sorry

end triangle_area_with_median_l43_43820


namespace highland_park_science_fair_l43_43934

noncomputable def juniors_and_seniors_participants (j s : ℕ) : ℕ :=
  (3 * j) / 4 + s / 2

theorem highland_park_science_fair 
  (j s : ℕ)
  (h1 : (3 * j) / 4 = s / 2)
  (h2 : j + s = 240) :
  juniors_and_seniors_participants j s = 144 := by
  sorry

end highland_park_science_fair_l43_43934


namespace no_groups_of_six_l43_43091

theorem no_groups_of_six (x y z : ℕ) 
  (h1 : (2 * x + 6 * y + 10 * z) / (x + y + z) = 5)
  (h2 : (2 * x + 30 * y + 90 * z) / (2 * x + 6 * y + 10 * z) = 7) : 
  y = 0 := 
sorry

end no_groups_of_six_l43_43091


namespace evaluate_expr_right_to_left_l43_43278

variable (a b c d : ℝ)

theorem evaluate_expr_right_to_left :
  (a - b * c + d) = a - b * (c + d) :=
sorry

end evaluate_expr_right_to_left_l43_43278


namespace function_relationship_minimize_total_cost_l43_43477

noncomputable def y (a x : ℕ) : ℕ :=
6400 * x + 50 * a + 100 * a^2 / (x - 1)

theorem function_relationship (a : ℕ) (hx : 2 ≤ x) : 
  y a x = 6400 * x + 50 * a + 100 * a^2 / (x - 1) :=
by sorry

theorem minimize_total_cost (a : ℕ) (hx : 2 ≤ x) (ha : a = 56) : 
  y a x ≥ 1650 * a + 6400 ∧ (x = 8) :=
by sorry

end function_relationship_minimize_total_cost_l43_43477


namespace minimum_tickets_needed_l43_43569

noncomputable def min_tickets {α : Type*} (winning_permutation : Fin 50 → α) (tickets : List (Fin 50 → α)) : ℕ :=
  List.length tickets

theorem minimum_tickets_needed
  (winning_permutation : Fin 50 → ℕ)
  (tickets : List (Fin 50 → ℕ))
  (h_tickets_valid : ∀ t ∈ tickets, Function.Surjective t)
  (h_at_least_one_match : ∀ winning_permutation : Fin 50 → ℕ,
      ∃ t ∈ tickets, ∃ i : Fin 50, t i = winning_permutation i) : 
  min_tickets winning_permutation tickets ≥ 26 :=
sorry

end minimum_tickets_needed_l43_43569


namespace binomial_np_sum_l43_43978

-- Definitions of variance and expectation for a binomial distribution
def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)
def binomial_expectation (n : ℕ) (p : ℚ) : ℚ := n * p

-- Statement of the problem
theorem binomial_np_sum (n : ℕ) (p : ℚ) (h_var : binomial_variance n p = 4) (h_exp : binomial_expectation n p = 12) :
    n + p = 56 / 3 := by
  sorry

end binomial_np_sum_l43_43978


namespace original_price_l43_43420

theorem original_price (P : ℝ) (h : 0.75 * (0.75 * P) = 17) : P = 30.22 :=
by
  sorry

end original_price_l43_43420


namespace problem_statement_l43_43750

theorem problem_statement (n : ℕ) (hn : n > 0) : (122 ^ n - 102 ^ n - 21 ^ n) % 2020 = 2019 :=
by
  sorry

end problem_statement_l43_43750


namespace sum_of_proper_divisors_less_than_100_of_780_l43_43187

def is_divisor (n d : ℕ) : Bool :=
  d ∣ n

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d ∣ n ∧ d < n)

def proper_divisors_less_than (n bound : ℕ) : List ℕ :=
  (proper_divisors n).filter (λ d => d < bound)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => acc + x) 0

theorem sum_of_proper_divisors_less_than_100_of_780 :
  sum_list (proper_divisors_less_than 780 100) = 428 :=
by
  sorry

end sum_of_proper_divisors_less_than_100_of_780_l43_43187


namespace brendan_weekly_capacity_l43_43066

/-- Brendan can cut 8 yards of grass per day on flat terrain under normal weather conditions. Bought a lawnmower that improved his cutting speed by 50 percent on flat terrain. On uneven terrain, his speed is reduced by 35 percent. Rain reduces his cutting capacity by 20 percent. Extreme heat reduces his cutting capacity by 10 percent. The conditions for each day of the week are given and we want to prove that the total yards Brendan can cut in a week is 65.46 yards.
  Monday: Flat terrain, normal weather
  Tuesday: Flat terrain, rain
  Wednesday: Uneven terrain, normal weather
  Thursday: Flat terrain, extreme heat
  Friday: Uneven terrain, rain
  Saturday: Flat terrain, normal weather
  Sunday: Uneven terrain, extreme heat
-/
def brendan_cutting_capacity : ℝ :=
  let base_capacity := 8.0
  let flat_terrain_boost := 1.5
  let uneven_terrain_penalty := 0.65
  let rain_penalty := 0.8
  let extreme_heat_penalty := 0.9
  let monday_capacity := base_capacity * flat_terrain_boost
  let tuesday_capacity := monday_capacity * rain_penalty
  let wednesday_capacity := monday_capacity * uneven_terrain_penalty
  let thursday_capacity := monday_capacity * extreme_heat_penalty
  let friday_capacity := wednesday_capacity * rain_penalty
  let saturday_capacity := monday_capacity
  let sunday_capacity := wednesday_capacity * extreme_heat_penalty
  monday_capacity + tuesday_capacity + wednesday_capacity + thursday_capacity + friday_capacity + saturday_capacity + sunday_capacity

theorem brendan_weekly_capacity : brendan_cutting_capacity = 65.46 := 
by 
  sorry

end brendan_weekly_capacity_l43_43066


namespace next_four_customers_cases_l43_43127

theorem next_four_customers_cases (total_people : ℕ) (first_eight_cases : ℕ) (last_eight_cases : ℕ) (total_cases : ℕ) :
    total_people = 20 →
    first_eight_cases = 24 →
    last_eight_cases = 8 →
    total_cases = 40 →
    (total_cases - (first_eight_cases + last_eight_cases)) / 4 = 2 :=
by
  intro h1 h2 h3 h4
  -- Fill in the proof steps using h1, h2, h3, and h4
  sorry

end next_four_customers_cases_l43_43127


namespace distance_between_A_and_B_l43_43466

noncomputable def time_from_A_to_B (D : ℝ) : ℝ := D / 200

noncomputable def time_from_B_to_A (D : ℝ) : ℝ := time_from_A_to_B D + 3

def condition (D : ℝ) : Prop := 
  D = 100 * (time_from_B_to_A D)

theorem distance_between_A_and_B :
  ∃ D : ℝ, condition D ∧ D = 600 :=
by
  sorry

end distance_between_A_and_B_l43_43466


namespace smallest_prime_10_less_than_perfect_square_l43_43025

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_10_less_than_perfect_square :
  ∃ (a : ℕ), is_prime a ∧ (∃ (n : ℕ), a = n^2 - 10) ∧ (∀ (b : ℕ), is_prime b ∧ (∃ (m : ℕ), b = m^2 - 10) → a ≤ b) ∧ a = 71 := 
by
  sorry

end smallest_prime_10_less_than_perfect_square_l43_43025


namespace smallest_period_find_a_l43_43610

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + a

theorem smallest_period (a : ℝ) : 
  ∃ T > 0, ∀ x, f x a = f (x + T) a ∧ (∀ T' > 0, (∀ x, f x a = f (x + T') a) → T ≤ T') :=
by
  sorry

theorem find_a :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) ∧ (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = 4) ∧ a = 1 :=
by
  sorry

end smallest_period_find_a_l43_43610


namespace find_k_l43_43767

theorem find_k (k x : ℝ) (h1 : x + k - 4 = 0) (h2 : x = 2) : k = 2 :=
by
  sorry

end find_k_l43_43767


namespace correct_option_is_C_l43_43732

-- Definitions based on the problem conditions
def option_A : Prop := (-3 + (-3)) = 0
def option_B : Prop := (-3 - abs (-3)) = 0
def option_C (a b : ℝ) : Prop := (3 * a^2 * b - 4 * b * a^2) = - a^2 * b
def option_D (x : ℝ) : Prop := (-(5 * x - 2)) = -5 * x - 2

-- The theorem to be proved that option C is the correct calculation
theorem correct_option_is_C (a b : ℝ) : option_C a b :=
sorry

end correct_option_is_C_l43_43732


namespace min_f_value_l43_43372

noncomputable def f (a b : ℝ) := 
  Real.sqrt (2 * a^2 - 8 * a + 10) + 
  Real.sqrt (b^2 - 6 * b + 10) + 
  Real.sqrt (2 * a^2 - 2 * a * b + b^2)

theorem min_f_value : ∃ a b : ℝ, f a b = 2 * Real.sqrt 5 :=
sorry

end min_f_value_l43_43372


namespace max_marked_cells_no_shared_vertices_l43_43326

theorem max_marked_cells_no_shared_vertices (N : ℕ) (cube_side : ℕ) (total_cells : ℕ) (total_vertices : ℕ) :
  cube_side = 3 →
  total_cells = cube_side ^ 3 →
  total_vertices = 8 + 12 * 2 + 6 * 4 →
  ∀ (max_cells : ℕ), (4 * max_cells ≤ total_vertices) → (max_cells ≤ 14) :=
by
  sorry

end max_marked_cells_no_shared_vertices_l43_43326


namespace Carl_chops_more_onions_than_Brittney_l43_43793

theorem Carl_chops_more_onions_than_Brittney :
  let Brittney_rate := 15 / 5
  let Carl_rate := 20 / 5
  let Brittney_onions := Brittney_rate * 30
  let Carl_onions := Carl_rate * 30
  Carl_onions = Brittney_onions + 30 :=
by
  sorry

end Carl_chops_more_onions_than_Brittney_l43_43793


namespace M_infinite_l43_43536

open Nat

-- Define the set M
def M : Set ℕ := {k | ∃ n : ℕ, 3 ^ n % n = k % n}

-- Statement of the problem
theorem M_infinite : Set.Infinite M :=
sorry

end M_infinite_l43_43536


namespace part1_part2_l43_43671

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Statement for part 1
theorem part1 (a b : ℝ) (h1 : f a b (-1) = 0) (h2 : f a b 3 = 0) (h3 : a ≠ 0) :
  a = -1 ∧ b = 4 :=
sorry

-- Statement for part 2
theorem part2 (a b : ℝ) (h1 : f a b 1 = 2) (h2 : a + b = 1) (h3 : a > 0) (h4 : b > 0) :
  (∀ x > 0, 1 / a + 4 / b ≥ 9) :=
sorry

end part1_part2_l43_43671


namespace intersection_of_sets_l43_43464

-- Define sets A and B
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- State the theorem
theorem intersection_of_sets : A ∩ B = {1, 2} := by
  sorry

end intersection_of_sets_l43_43464


namespace inequality_holds_for_all_real_l43_43044

theorem inequality_holds_for_all_real (x : ℝ) : x^2 + 6 * x + 8 ≥ -(x + 4) * (x + 6) :=
  sorry

end inequality_holds_for_all_real_l43_43044


namespace num_prime_divisors_50_factorial_eq_15_l43_43629

theorem num_prime_divisors_50_factorial_eq_15 :
  (Finset.filter Nat.Prime (Finset.range 51)).card = 15 :=
by
  sorry

end num_prime_divisors_50_factorial_eq_15_l43_43629


namespace megan_files_in_folder_l43_43633

theorem megan_files_in_folder :
  let initial_files := 93.0
  let added_files := 21.0
  let total_files := initial_files + added_files
  let total_folders := 14.25
  (total_files / total_folders) = 8.0 :=
by
  let initial_files := 93.0
  let added_files := 21.0
  let total_files := initial_files + added_files
  let total_folders := 14.25
  have h1 : total_files = initial_files + added_files := rfl
  have h2 : total_files = 114.0 := by sorry -- 93.0 + 21.0 = 114.0
  have h3 : total_files / total_folders = 8.0 := by sorry -- 114.0 / 14.25 = 8.0
  exact h3

end megan_files_in_folder_l43_43633


namespace factorization_correct_l43_43818

theorem factorization_correct (x : ℝ) : 
  98 * x^7 - 266 * x^13 = 14 * x^7 * (7 - 19 * x^6) :=
by
  sorry

end factorization_correct_l43_43818


namespace toys_per_box_l43_43619

theorem toys_per_box (number_of_boxes total_toys : ℕ) (h₁ : number_of_boxes = 4) (h₂ : total_toys = 32) :
  total_toys / number_of_boxes = 8 :=
by
  sorry

end toys_per_box_l43_43619


namespace Jerry_wants_to_raise_average_l43_43528

theorem Jerry_wants_to_raise_average 
  (first_three_tests_avg : ℕ) (fourth_test_score : ℕ) (desired_increase : ℕ) 
  (h1 : first_three_tests_avg = 90) (h2 : fourth_test_score = 98) 
  : desired_increase = 2 := 
by
  sorry

end Jerry_wants_to_raise_average_l43_43528


namespace tenth_pirate_receives_exactly_1296_coins_l43_43001

noncomputable def pirate_coins (n : ℕ) : ℕ :=
  if n = 0 then 0
  else Nat.factorial 9 / 11^9 * 11^(10 - n)

theorem tenth_pirate_receives_exactly_1296_coins :
  pirate_coins 10 = 1296 :=
sorry

end tenth_pirate_receives_exactly_1296_coins_l43_43001


namespace number_of_adults_l43_43437

theorem number_of_adults (total_apples : ℕ) (children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) (h : total_apples = 450) (h1 : children = 33) (h2 : apples_per_child = 10) (h3 : apples_per_adult = 3) :
  total_apples - (children * apples_per_child) = 120 →
  (total_apples - (children * apples_per_child)) / apples_per_adult = 40 :=
by
  intros
  sorry

end number_of_adults_l43_43437


namespace problem_part1_problem_part2_l43_43596

open Real

noncomputable def f (x : ℝ) := (sqrt 3) * sin x * cos x - (1 / 2) * cos (2 * x)

theorem problem_part1 : 
  (∀ x : ℝ, -1 ≤ f x) ∧ 
  (∃ T : ℝ, (T > 0) ∧ ∀ x : ℝ, f (x + T) = f x ∧ T = π) := 
sorry

theorem problem_part2 (C A B c : ℝ) :
  (f C = 1) → 
  (B = π / 6) → 
  (c = 2 * sqrt 3) → 
  ∃ b : ℝ, ∃ area : ℝ, b = 2 ∧ area = (1 / 2) * b * c * sin A ∧ area = 2 * sqrt 3 := 
sorry

end problem_part1_problem_part2_l43_43596


namespace correctly_calculated_value_l43_43063

theorem correctly_calculated_value (n : ℕ) (h : 5 * n = 30) : n / 6 = 1 :=
sorry

end correctly_calculated_value_l43_43063


namespace width_of_channel_at_bottom_l43_43070

theorem width_of_channel_at_bottom
    (top_width : ℝ)
    (area : ℝ)
    (depth : ℝ)
    (b : ℝ)
    (H1 : top_width = 12)
    (H2 : area = 630)
    (H3 : depth = 70)
    (H4 : area = 0.5 * (top_width + b) * depth) :
    b = 6 := 
sorry

end width_of_channel_at_bottom_l43_43070


namespace find_f_1000_l43_43915

theorem find_f_1000 (f : ℕ → ℕ) 
    (h1 : ∀ n : ℕ, 0 < n → f (f n) = 2 * n) 
    (h2 : ∀ n : ℕ, 0 < n → f (3 * n + 1) = 3 * n + 2) : 
    f 1000 = 1008 :=
by
  sorry

end find_f_1000_l43_43915


namespace solve_sum_of_squares_l43_43848

theorem solve_sum_of_squares
  (k l m n a b c : ℕ)
  (h_cond1 : k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ l ≠ m ∧ l ≠ n ∧ m ≠ n)
  (h_cond2 : a * k^2 - b * k + c = 0)
  (h_cond3 : a * l^2 - b * l + c = 0)
  (h_cond4 : c * m^2 - 16 * b * m + 256 * a = 0)
  (h_cond5 : c * n^2 - 16 * b * n + 256 * a = 0) :
  k^2 + l^2 + m^2 + n^2 = 325 :=
by
  sorry

end solve_sum_of_squares_l43_43848


namespace transformed_polynomial_roots_l43_43079

theorem transformed_polynomial_roots (a b c d : ℝ) 
  (h1 : a + b + c + d = 0)
  (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h3 : a * b * c * d ≠ 0)
  (h4 : Polynomial.eval a (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h5 : Polynomial.eval b (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h6 : Polynomial.eval c (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0)
  (h7 : Polynomial.eval d (Polynomial.X ^ 4 - 2 * Polynomial.X - 6) = 0):
  Polynomial.eval (-2 / d^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / c^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / b^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 ∧
    Polynomial.eval (-2 / a^2) (2 * Polynomial.X ^ 4 - 2 * Polynomial.X + 48) = 0 :=
sorry

end transformed_polynomial_roots_l43_43079


namespace problem_l43_43887

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (x - Real.pi / 2)

theorem problem 
: (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ c, c = (Real.pi / 2) ∧ f c = 0) → (T = Real.pi ∧ c = (Real.pi / 2)) :=
sorry

end problem_l43_43887


namespace problem_divisible_by_factors_l43_43116

theorem problem_divisible_by_factors (n : ℕ) (x : ℝ) : 
  ∃ k : ℝ, (x + 1)^(2 * n) - x^(2 * n) - 2 * x - 1 = k * x * (x + 1) * (2 * x + 1) :=
by
  sorry

end problem_divisible_by_factors_l43_43116


namespace exists_N_for_sqrt_expressions_l43_43208

theorem exists_N_for_sqrt_expressions 
  (p q n : ℕ) (hp : 0 < p) (hq : 0 < q) (hn : 0 < n) (h_q_le_p2 : q ≤ p^2) :
  ∃ N : ℕ, 
    (N > 0) ∧ 
    ((p - Real.sqrt (p^2 - q))^n = N - Real.sqrt (N^2 - q^n)) ∧ 
    ((p + Real.sqrt (p^2 - q))^n = N + Real.sqrt (N^2 - q^n)) :=
sorry

end exists_N_for_sqrt_expressions_l43_43208


namespace max_horizontal_segment_length_l43_43889

theorem max_horizontal_segment_length (y : ℝ → ℝ) (h : ∀ x, y x = x^3 - x) :
  ∃ a, (∀ x₁, y x₁ = y (x₁ + a)) ∧ a = 2 :=
by
  sorry

end max_horizontal_segment_length_l43_43889


namespace inequality_solution_1_inequality_solution_2_l43_43949

-- Definition for part 1
theorem inequality_solution_1 (x : ℝ) : x^2 + 3*x - 4 > 0 ↔ x > 1 ∨ x < -4 :=
sorry

-- Definition for part 2
theorem inequality_solution_2 (x : ℝ) : (1 - x) / (x - 5) ≥ 1 ↔ 3 ≤ x ∧ x < 5 :=
sorry

end inequality_solution_1_inequality_solution_2_l43_43949


namespace problem_statement_l43_43173

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

variable (f g : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_neg : ∀ x : ℝ, x < 0 → f x = x^3 - 1
axiom f_pos : ∀ x : ℝ, x > 0 → f x = g x

theorem problem_statement : f (-1) + g 2 = 7 :=
by
  sorry

end problem_statement_l43_43173


namespace cuboid_edge_lengths_l43_43935

theorem cuboid_edge_lengths (a b c : ℕ) (S V : ℕ) :
  (S = 2 * (a * b + b * c + c * a)) ∧ (V = a * b * c) ∧ (V = S) ∧ 
  (∃ d : ℕ, d = Int.sqrt (a^2 + b^2 + c^2)) →
  (∃ a b c : ℕ, a = 4 ∧ b = 8 ∧ c = 8) :=
by
  sorry

end cuboid_edge_lengths_l43_43935


namespace volume_ratio_john_emma_l43_43638

theorem volume_ratio_john_emma (r_J h_J r_E h_E : ℝ) (diam_J diam_E : ℝ)
  (h_diam_J : diam_J = 8) (h_r_J : r_J = diam_J / 2) (h_h_J : h_J = 15)
  (h_diam_E : diam_E = 10) (h_r_E : r_E = diam_E / 2) (h_h_E : h_E = 12) :
  (π * r_J^2 * h_J) / (π * r_E^2 * h_E) = 4 / 5 := by
  sorry

end volume_ratio_john_emma_l43_43638


namespace seating_arrangement_l43_43307

theorem seating_arrangement (x y z : ℕ) (h1 : z = x + y) (h2 : x*10 + y*9 = 67) : x = 4 :=
by
  sorry

end seating_arrangement_l43_43307


namespace smallest_number_of_digits_to_append_l43_43053

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end smallest_number_of_digits_to_append_l43_43053


namespace E1_E2_complementary_l43_43221

-- Define the universal set for a fair die with six faces
def universalSet : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define each event as a set based on the problem conditions
def E1 : Set ℕ := {1, 3, 5}
def E2 : Set ℕ := {2, 4, 6}
def E3 : Set ℕ := {4, 5, 6}
def E4 : Set ℕ := {1, 2}

-- Define complementary events
def areComplementary (A B : Set ℕ) : Prop :=
  (A ∪ B = universalSet) ∧ (A ∩ B = ∅)

-- State the theorem that events E1 and E2 are complementary
theorem E1_E2_complementary : areComplementary E1 E2 :=
sorry

end E1_E2_complementary_l43_43221
