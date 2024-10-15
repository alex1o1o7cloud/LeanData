import Mathlib

namespace NUMINAMATH_GPT_z_rate_per_rupee_of_x_l1452_145281

-- Given conditions as definitions in Lean 4
def x_share := 1 -- x gets Rs. 1 for this proof
def y_rate_per_rupee_of_x := 0.45
def y_share := 27
def total_amount := 105

-- The statement to prove
theorem z_rate_per_rupee_of_x :
  (105 - (1 * 60) - 27) / 60 = 0.30 :=
by
  sorry

end NUMINAMATH_GPT_z_rate_per_rupee_of_x_l1452_145281


namespace NUMINAMATH_GPT_total_passengers_landed_l1452_145247

theorem total_passengers_landed 
  (passengers_on_time : ℕ) 
  (passengers_late : ℕ) 
  (passengers_connecting : ℕ) 
  (passengers_changed_plans : ℕ)
  (H1 : passengers_on_time = 14507)
  (H2 : passengers_late = 213)
  (H3 : passengers_connecting = 320)
  (H4 : passengers_changed_plans = 95) : 
  passengers_on_time + passengers_late + passengers_connecting = 15040 :=
by 
  sorry

end NUMINAMATH_GPT_total_passengers_landed_l1452_145247


namespace NUMINAMATH_GPT_scale_length_discrepancy_l1452_145276

theorem scale_length_discrepancy
  (scale_length_feet : ℝ)
  (parts : ℕ)
  (part_length_inches : ℝ)
  (ft_to_inch : ℝ := 12)
  (total_length_inches : ℝ := parts * part_length_inches)
  (scale_length_inches : ℝ := scale_length_feet * ft_to_inch) :
  scale_length_feet = 7 → 
  parts = 4 → 
  part_length_inches = 24 →
  total_length_inches - scale_length_inches = 12 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_scale_length_discrepancy_l1452_145276


namespace NUMINAMATH_GPT_cyclic_quadrilateral_AC_plus_BD_l1452_145220

theorem cyclic_quadrilateral_AC_plus_BD (AB BC CD DA : ℝ) (AC BD : ℝ) (h1 : AB = 5) (h2 : BC = 10) (h3 : CD = 11) (h4 : DA = 14)
  (h5 : AC = Real.sqrt 221) (h6 : BD = 195 / Real.sqrt 221) :
  AC + BD = 416 / Real.sqrt (13 * 17) ∧ (AC = Real.sqrt 221 ∧ BD = 195 / Real.sqrt 221) →
  (AC + BD = 416 / Real.sqrt (13 * 17)) ∧ (AC + BD = 446) :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_AC_plus_BD_l1452_145220


namespace NUMINAMATH_GPT_running_problem_l1452_145298

variables (x y : ℝ)

theorem running_problem :
  (5 * x = 5 * y + 10) ∧ (4 * x = 4 * y + 2 * y) :=
by
  sorry

end NUMINAMATH_GPT_running_problem_l1452_145298


namespace NUMINAMATH_GPT_sqrt_expression_l1452_145273

theorem sqrt_expression (y : ℝ) (hy : y < 0) : 
  Real.sqrt (y / (1 - ((y - 2) / y))) = -y / Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_sqrt_expression_l1452_145273


namespace NUMINAMATH_GPT_range_m_l1452_145232

theorem range_m (m : ℝ) :
  (∀ x : ℝ, (1 / 3 < x ∧ x < 1 / 2) ↔ abs (x - m) < 1) →
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_m_l1452_145232


namespace NUMINAMATH_GPT_cistern_filling_time_with_leak_l1452_145270

theorem cistern_filling_time_with_leak (T : ℝ) (h1 : 1 / T - 1 / 4 = 1 / (T + 2)) : T = 4 :=
by
  sorry

end NUMINAMATH_GPT_cistern_filling_time_with_leak_l1452_145270


namespace NUMINAMATH_GPT_exist_end_2015_l1452_145230

def in_sequence (n : Nat) : Nat :=
  90 * n + 75

theorem exist_end_2015 :
  ∃ n : Nat, in_sequence n % 10000 = 2015 :=
by
  sorry

end NUMINAMATH_GPT_exist_end_2015_l1452_145230


namespace NUMINAMATH_GPT_ethanol_percentage_in_fuel_A_l1452_145223

variable {capacity_A fuel_A : ℝ}
variable (ethanol_A ethanol_B total_ethanol : ℝ)
variable (E : ℝ)

def fuelTank (capacity_A fuel_A ethanol_A ethanol_B total_ethanol : ℝ) (E : ℝ) : Prop := 
  (ethanol_A / fuel_A = E) ∧
  (capacity_A - fuel_A = 200 - 99.99999999999999) ∧
  (ethanol_B = 0.16 * (200 - 99.99999999999999)) ∧
  (total_ethanol = ethanol_A + ethanol_B) ∧
  (total_ethanol = 28)

theorem ethanol_percentage_in_fuel_A : 
  ∃ E, fuelTank 99.99999999999999 99.99999999999999 ethanol_A ethanol_B 28 E ∧ E = 0.12 := 
sorry

end NUMINAMATH_GPT_ethanol_percentage_in_fuel_A_l1452_145223


namespace NUMINAMATH_GPT_area_of_triangle_l1452_145218

theorem area_of_triangle (base : ℝ) (height : ℝ) (h_base : base = 3.6) (h_height : height = 2.5 * base) : 
  (base * height) / 2 = 16.2 :=
by {
  sorry
}

end NUMINAMATH_GPT_area_of_triangle_l1452_145218


namespace NUMINAMATH_GPT_solve_for_x_l1452_145288

theorem solve_for_x :
  ∃ x : ℝ, 40 + (5 * x) / (180 / 3) = 41 ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1452_145288


namespace NUMINAMATH_GPT_parrots_left_l1452_145297

theorem parrots_left 
  (c : Nat)   -- The initial number of crows
  (x : Nat)   -- The number of parrots and crows that flew away
  (h1 : 7 + c = 13)          -- Initial total number of birds
  (h2 : c - x = 1)           -- Number of crows left
  : 7 - x = 2 :=             -- Number of parrots left
by
  sorry

end NUMINAMATH_GPT_parrots_left_l1452_145297


namespace NUMINAMATH_GPT_find_m_value_l1452_145254

def symmetric_inverse (g : ℝ → ℝ) (h : ℝ → ℝ) :=
  ∀ x, g (h x) = x ∧ h (g x) = x

def symmetric_y_axis (f : ℝ → ℝ) (g : ℝ → ℝ) :=
  ∀ x, f x = g (-x)

theorem find_m_value :
  (∀ g, symmetric_inverse g (Real.exp) → (∀ f, symmetric_y_axis f g → (∀ m, f m = -1 → m = - (1 / Real.exp 1)))) := by
  sorry

end NUMINAMATH_GPT_find_m_value_l1452_145254


namespace NUMINAMATH_GPT_fraction_of_percent_l1452_145208

theorem fraction_of_percent (h : (1 / 8 * (1 / 100)) * 800 = 1) : true :=
by
  trivial

end NUMINAMATH_GPT_fraction_of_percent_l1452_145208


namespace NUMINAMATH_GPT_total_distance_l1452_145279

theorem total_distance (x y : ℝ) (h1 : x * y = 18) :
  let D2 := (y - 1) * (x + 1)
  let D3 := 15
  let D_total := 18 + D2 + D3
  D_total = y * x + y - x + 32 :=
by
  let D2 := (y - 1) * (x + 1)
  let D3 := 15
  let D_total := 18 + D2 + D3
  sorry

end NUMINAMATH_GPT_total_distance_l1452_145279


namespace NUMINAMATH_GPT_map_length_l1452_145251

theorem map_length 
  (width : ℝ) (area : ℝ) 
  (h_width : width = 10) (h_area : area = 20) : 
  ∃ length : ℝ, area = width * length ∧ length = 2 :=
by 
  sorry

end NUMINAMATH_GPT_map_length_l1452_145251


namespace NUMINAMATH_GPT_ab_operation_l1452_145213

theorem ab_operation (a b : ℤ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h1 : a + b = 10) (h2 : a * b = 24) : 
  (1 / a + 1 / b) = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_ab_operation_l1452_145213


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1452_145205

theorem solution_set_of_inequality (x : ℝ) : -2 * x - 1 < 3 ↔ x > -2 := 
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1452_145205


namespace NUMINAMATH_GPT_part1_and_part2_l1452_145296

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end NUMINAMATH_GPT_part1_and_part2_l1452_145296


namespace NUMINAMATH_GPT_coby_travel_time_l1452_145258

theorem coby_travel_time :
  let d1 := 640
  let d2 := 400
  let d3 := 250
  let d4 := 380
  let s1 := 80
  let s2 := 65
  let s3 := 75
  let s4 := 50
  let time1 := d1 / s1
  let time2 := d2 / s2
  let time3 := d3 / s3
  let time4 := d4 / s4
  let total_time := time1 + time2 + time3 + time4
  total_time = 25.08 :=
by
  sorry

end NUMINAMATH_GPT_coby_travel_time_l1452_145258


namespace NUMINAMATH_GPT_value_of_a_b_l1452_145253

theorem value_of_a_b (a b : ℕ) (ha : 2 * 100 + a * 10 + 3 + 326 = 5 * 100 + b * 10 + 9) (hb : (5 + b + 9) % 9 = 0): 
  a + b = 6 := 
sorry

end NUMINAMATH_GPT_value_of_a_b_l1452_145253


namespace NUMINAMATH_GPT_ammonium_chloride_reaction_l1452_145263

/-- 
  Given the reaction NH4Cl + H2O → NH4OH + HCl, 
  if 1 mole of NH4Cl reacts with 1 mole of H2O to produce 1 mole of NH4OH, 
  then 1 mole of HCl is formed.
-/
theorem ammonium_chloride_reaction :
  (∀ (NH4Cl H2O NH4OH HCl : ℕ), NH4Cl = 1 ∧ H2O = 1 ∧ NH4OH = 1 → HCl = 1) :=
by
  sorry

end NUMINAMATH_GPT_ammonium_chloride_reaction_l1452_145263


namespace NUMINAMATH_GPT_intersection_of_sets_l1452_145284

def setA := { x : ℝ | x / (x - 1) < 0 }
def setB := { x : ℝ | 0 < x ∧ x < 3 }
def setIntersect := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_of_sets :
  ∀ x : ℝ, x ∈ setA ∧ x ∈ setB ↔ x ∈ setIntersect := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1452_145284


namespace NUMINAMATH_GPT_find_k_l1452_145268

theorem find_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (h3 : k ≠ 1) (h4 : 1/a + 2/b = 1) : k = 18 := by
  sorry

end NUMINAMATH_GPT_find_k_l1452_145268


namespace NUMINAMATH_GPT_closing_price_l1452_145294

theorem closing_price (opening_price : ℝ) (percent_increase : ℝ) (closing_price : ℝ) 
  (h₀ : opening_price = 6) (h₁ : percent_increase = 0.3333) : closing_price = 8 :=
by
  sorry

end NUMINAMATH_GPT_closing_price_l1452_145294


namespace NUMINAMATH_GPT_x_cubed_plus_y_cubed_l1452_145231

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end NUMINAMATH_GPT_x_cubed_plus_y_cubed_l1452_145231


namespace NUMINAMATH_GPT_greatest_k_dividing_n_l1452_145249

theorem greatest_k_dividing_n (n : ℕ) 
  (h1 : Nat.totient n = 72) 
  (h2 : Nat.totient (3 * n) = 96) : ∃ k : ℕ, 3^k ∣ n ∧ ∀ j : ℕ, 3^j ∣ n → j ≤ 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_greatest_k_dividing_n_l1452_145249


namespace NUMINAMATH_GPT_Maggie_earnings_l1452_145245

theorem Maggie_earnings
    (price_per_subscription : ℕ)
    (subscriptions_parents : ℕ)
    (subscriptions_grandfather : ℕ)
    (subscriptions_nextdoor : ℕ)
    (subscriptions_another : ℕ)
    (total_subscriptions : ℕ)
    (total_earnings : ℕ) :
    subscriptions_parents = 4 →
    subscriptions_grandfather = 1 →
    subscriptions_nextdoor = 2 →
    subscriptions_another = 2 * subscriptions_nextdoor →
    total_subscriptions = subscriptions_parents + subscriptions_grandfather + subscriptions_nextdoor + subscriptions_another →
    price_per_subscription = 5 →
    total_earnings = price_per_subscription * total_subscriptions →
    total_earnings = 55 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Maggie_earnings_l1452_145245


namespace NUMINAMATH_GPT_boxes_with_neither_l1452_145282

def total_boxes : ℕ := 15
def boxes_with_stickers : ℕ := 9
def boxes_with_stamps : ℕ := 5
def boxes_with_both : ℕ := 3

theorem boxes_with_neither
  (total_boxes : ℕ)
  (boxes_with_stickers : ℕ)
  (boxes_with_stamps : ℕ)
  (boxes_with_both : ℕ) :
  total_boxes - ((boxes_with_stickers + boxes_with_stamps) - boxes_with_both) = 4 :=
by
  sorry

end NUMINAMATH_GPT_boxes_with_neither_l1452_145282


namespace NUMINAMATH_GPT_planted_area_ratio_l1452_145262

noncomputable def ratio_of_planted_area_to_total_area : ℚ := 145 / 147

theorem planted_area_ratio (h : ∃ (S : ℚ), 
  (∃ (x y : ℚ), x * x + y * y ≤ S * S) ∧
  (∃ (a b : ℚ), 3 * a + 4 * b = 12 ∧ (3 * x + 4 * y - 12) / 5 = 2)) :
  ratio_of_planted_area_to_total_area = 145 / 147 :=
sorry

end NUMINAMATH_GPT_planted_area_ratio_l1452_145262


namespace NUMINAMATH_GPT_largest_sphere_radius_in_prism_l1452_145292

noncomputable def largestInscribedSphereRadius (m : ℝ) : ℝ :=
  (Real.sqrt 6 - Real.sqrt 2) / 4 * m

theorem largest_sphere_radius_in_prism (m : ℝ) (h : 0 < m) :
  ∃ r, r = largestInscribedSphereRadius m ∧ r < m/2 :=
sorry

end NUMINAMATH_GPT_largest_sphere_radius_in_prism_l1452_145292


namespace NUMINAMATH_GPT_problem_statement_l1452_145259

open Classical

variable (a_n : ℕ → ℝ) (a1 d : ℝ)

-- Condition: Arithmetic sequence with first term a1 and common difference d
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ (n : ℕ), a_n (n + 1) = a1 + n * d 

-- Condition: Geometric relationship between a1, a3, and a9
def geometric_relation (a1 a3 a9 : ℝ) : Prop :=
  a3 / a1 = a9 / a3

-- Given conditions for the arithmetic sequence and geometric relation
axiom arith : arithmetic_sequence a_n a1 d
axiom geom : geometric_relation a1 (a1 + 2 * d) (a1 + 8 * d)

theorem problem_statement : d ≠ 0 → (∃ (a1 d : ℝ), d ≠ 0 ∧ arithmetic_sequence a_n a1 d ∧ geometric_relation a1 (a1 + 2 * d) (a1 + 8 * d)) → (a1 + 2 * d) / a1 = 3 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1452_145259


namespace NUMINAMATH_GPT_benny_apples_l1452_145277

theorem benny_apples (benny dan : ℕ) (total : ℕ) (H1 : dan = 9) (H2 : total = 11) (H3 : benny + dan = total) : benny = 2 :=
by
  sorry

end NUMINAMATH_GPT_benny_apples_l1452_145277


namespace NUMINAMATH_GPT_magic_shop_change_l1452_145237

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end NUMINAMATH_GPT_magic_shop_change_l1452_145237


namespace NUMINAMATH_GPT_Ruby_math_homework_l1452_145250

theorem Ruby_math_homework : 
  ∃ M : ℕ, ∃ R : ℕ, R = 2 ∧ 5 * M + 9 * R = 48 ∧ M = 6 := by
  sorry

end NUMINAMATH_GPT_Ruby_math_homework_l1452_145250


namespace NUMINAMATH_GPT_mary_flour_indeterminate_l1452_145291

theorem mary_flour_indeterminate 
  (sugar : ℕ) (flour : ℕ) (salt : ℕ) (needed_sugar_more : ℕ) 
  (h_sugar : sugar = 11) (h_flour : flour = 6)
  (h_salt : salt = 9) (h_condition : needed_sugar_more = 2) :
  ∃ (current_flour : ℕ), current_flour ≠ current_flour :=
by
  sorry

end NUMINAMATH_GPT_mary_flour_indeterminate_l1452_145291


namespace NUMINAMATH_GPT_function_identity_l1452_145209

theorem function_identity (f : ℕ → ℕ) 
  (h_pos : f 1 > 0) 
  (h_property : ∀ m n : ℕ, f (m^2 + n^2) = f m^2 + f n^2) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end NUMINAMATH_GPT_function_identity_l1452_145209


namespace NUMINAMATH_GPT_find_three_xsq_ysq_l1452_145202

theorem find_three_xsq_ysq (x y : ℤ) (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 3*x^2*y^2 = 588 :=
sorry

end NUMINAMATH_GPT_find_three_xsq_ysq_l1452_145202


namespace NUMINAMATH_GPT_k_value_range_l1452_145214

-- Definitions
def f (x : ℝ) (k : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- The theorem we are interested in
theorem k_value_range (k : ℝ) (h : ∀ x₁ x₂ : ℝ, (x₁ > 5 → x₂ > 5 → f x₁ k ≤ f x₂ k) ∨ (x₁ > 5 → x₂ > 5 → f x₁ k ≥ f x₂ k)) :
  k ≥ 40 :=
sorry

end NUMINAMATH_GPT_k_value_range_l1452_145214


namespace NUMINAMATH_GPT_estimate_less_Exact_l1452_145240

variables (a b c d : ℕ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

def round_up (x : ℕ) : ℕ := x + 1
def round_down (x : ℕ) : ℕ := x - 1

theorem estimate_less_Exact
  (h₁ : round_down a = a - 1)
  (h₂ : round_down b = b - 1)
  (h₃ : round_down c = c - 1)
  (h₄ : round_up d = d + 1) :
  (round_down a + round_down b) / round_down c - round_up d < 
  (a + b) / c - d :=
sorry

end NUMINAMATH_GPT_estimate_less_Exact_l1452_145240


namespace NUMINAMATH_GPT_calculation_result_l1452_145244

theorem calculation_result : 7 * (9 + 2 / 5) + 3 = 68.8 :=
by
  sorry

end NUMINAMATH_GPT_calculation_result_l1452_145244


namespace NUMINAMATH_GPT_smallest_trees_in_three_types_l1452_145266

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end NUMINAMATH_GPT_smallest_trees_in_three_types_l1452_145266


namespace NUMINAMATH_GPT_capital_of_z_l1452_145242

theorem capital_of_z (x y z : ℕ) (annual_profit z_share : ℕ) (months_x months_y months_z : ℕ) 
    (rx ry : ℕ) (r : ℚ) :
  x = 20000 →
  y = 25000 →
  z_share = 14000 →
  annual_profit = 50000 →
  rx = 240000 →
  ry = 300000 →
  months_x = 12 →
  months_y = 12 →
  months_z = 7 →
  r = 7 / 25 →
  z * months_z * r = z_share / (rx + ry + z * months_z) →
  z = 30000 := 
by intros; sorry

end NUMINAMATH_GPT_capital_of_z_l1452_145242


namespace NUMINAMATH_GPT_probabilityOfWearingSunglassesGivenCap_l1452_145280

-- Define the conditions as Lean constants
def peopleWearingSunglasses : ℕ := 80
def peopleWearingCaps : ℕ := 60
def probabilityOfWearingCapGivenSunglasses : ℚ := 3 / 8
def peopleWearingBoth : ℕ := (3 / 8) * 80

-- Prove the desired probability
theorem probabilityOfWearingSunglassesGivenCap : (peopleWearingBoth / peopleWearingCaps = 1 / 2) :=
by
  -- sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_probabilityOfWearingSunglassesGivenCap_l1452_145280


namespace NUMINAMATH_GPT_no_same_distribution_of_silver_as_gold_l1452_145233

theorem no_same_distribution_of_silver_as_gold (n m : ℕ) 
  (hn : n ≡ 5 [MOD 10]) 
  (hm : m = 2 * n) 
  : ∀ (f : Fin 10 → ℕ), (∀ i j : Fin 10, i ≠ j → ¬ (f i - f j ≡ 0 [MOD 10])) 
  → ∀ (g : Fin 10 → ℕ), ¬ (∀ i j : Fin 10, i ≠ j → ¬ (g i - g j ≡ 0 [MOD 10])) :=
sorry

end NUMINAMATH_GPT_no_same_distribution_of_silver_as_gold_l1452_145233


namespace NUMINAMATH_GPT_volume_of_cuboctahedron_l1452_145295

def points (i j : ℕ) (A : ℕ → ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x0, y0, z0) := A 0
  let (xi, yi, zi) := A i
  let (xj, yj, zj) := A j
  (xi - xj, yi - yj, zi - zj)

def is_cuboctahedron (points_set : Set (ℝ × ℝ × ℝ)) : Prop :=
  -- Insert specific conditions that define a cuboctahedron
  sorry

theorem volume_of_cuboctahedron : 
  let A := fun 
    | 0 => (0, 0, 0)
    | 1 => (1, 0, 0)
    | 2 => (0, 1, 0)
    | 3 => (0, 0, 1)
    | _ => (0, 0, 0)
  let P_ij := 
    {p | ∃ i j : ℕ, i ≠ j ∧ p = points i j A}
  ∃ v : ℝ, is_cuboctahedron P_ij ∧ v = 10 / 3 :=
sorry

end NUMINAMATH_GPT_volume_of_cuboctahedron_l1452_145295


namespace NUMINAMATH_GPT_multiple_of_totient_l1452_145234

theorem multiple_of_totient (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (a : ℕ), ∀ (i : ℕ), 0 ≤ i ∧ i ≤ n → m ∣ Nat.totient (a + i) :=
by
sorry

end NUMINAMATH_GPT_multiple_of_totient_l1452_145234


namespace NUMINAMATH_GPT_circle_intersection_zero_l1452_145271

theorem circle_intersection_zero :
  (∀ θ : ℝ, ∀ r1 : ℝ, r1 = 3 * Real.cos θ → ∀ r2 : ℝ, r2 = 6 * Real.sin (2 * θ) → False) :=
by 
  sorry

end NUMINAMATH_GPT_circle_intersection_zero_l1452_145271


namespace NUMINAMATH_GPT_angle_invariant_under_magnification_l1452_145211

theorem angle_invariant_under_magnification :
  ∀ (angle magnification : ℝ), angle = 10 → magnification = 5 → angle = 10 := by
  intros angle magnification h_angle h_magnification
  exact h_angle

end NUMINAMATH_GPT_angle_invariant_under_magnification_l1452_145211


namespace NUMINAMATH_GPT_prime_divisor_condition_l1452_145275

theorem prime_divisor_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hdiv : q ∣ 2^p - 1) : p ∣ q - 1 :=
  sorry

end NUMINAMATH_GPT_prime_divisor_condition_l1452_145275


namespace NUMINAMATH_GPT_negation_example_l1452_145255

theorem negation_example :
  (¬ (∀ x: ℝ, x > 0 → x^2 + x + 1 > 0)) ↔ (∃ x: ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l1452_145255


namespace NUMINAMATH_GPT_simplify_expression_l1452_145203

variable (m : ℝ)

theorem simplify_expression (h₁ : m ≠ 2) (h₂ : m ≠ 3) :
  (m - (4 * m - 9) / (m - 2)) / ((m ^ 2 - 9) / (m - 2)) = (m - 3) / (m + 3) := 
sorry

end NUMINAMATH_GPT_simplify_expression_l1452_145203


namespace NUMINAMATH_GPT_solution_set_inequality_l1452_145206

theorem solution_set_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x, f x + (deriv^[2] f) x < 1) (h_f0 : f 0 = 2018) :
  ∀ x, x > 0 → f x < 2017 * Real.exp (-x) + 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1452_145206


namespace NUMINAMATH_GPT_find_angle_C_find_side_a_l1452_145278

namespace TriangleProof

-- Declare the conditions and the proof promises
variables {A B C : ℝ} {a b c S : ℝ}

-- First part: Prove angle C
theorem find_angle_C (h1 : c^2 = a^2 + b^2 - a * b) : C = 60 :=
sorry

-- Second part: Prove the value of a
theorem find_side_a (h2 : b = 2) (h3 : S = (3 * Real.sqrt 3) / 2) : a = 3 :=
sorry

end TriangleProof

end NUMINAMATH_GPT_find_angle_C_find_side_a_l1452_145278


namespace NUMINAMATH_GPT_sum_of_values_of_N_l1452_145200

theorem sum_of_values_of_N (N : ℂ) : (N * (N - 8) = 12) → (∃ x y : ℂ, N = x ∨ N = y ∧ x + y = 8) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_values_of_N_l1452_145200


namespace NUMINAMATH_GPT_pig_farm_fence_l1452_145204

theorem pig_farm_fence (fenced_side : ℝ) (area : ℝ) 
  (h1 : fenced_side * 2 * fenced_side = area) 
  (h2 : area = 1250) :
  4 * fenced_side = 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_pig_farm_fence_l1452_145204


namespace NUMINAMATH_GPT_arithmetic_sequence_S30_l1452_145201

theorem arithmetic_sequence_S30
  (S : ℕ → ℕ)
  (h_arith_seq: ∀ m : ℕ, 2 * (S (2 * m) - S m) = S m + S (3 * m) - S (2 * m))
  (h_S10: S 10 = 4)
  (h_S20: S 20 = 20) :
  S 30 = 48 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_S30_l1452_145201


namespace NUMINAMATH_GPT_find_distinct_prime_triples_l1452_145236

noncomputable def areDistinctPrimes (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r

def satisfiesConditions (p q r : ℕ) : Prop :=
  p ∣ (q + r) ∧ q ∣ (r + 2 * p) ∧ r ∣ (p + 3 * q)

theorem find_distinct_prime_triples :
  { (p, q, r) : ℕ × ℕ × ℕ | areDistinctPrimes p q r ∧ satisfiesConditions p q r } =
  { (5, 3, 2), (2, 11, 7), (2, 3, 11) } :=
by
  sorry

end NUMINAMATH_GPT_find_distinct_prime_triples_l1452_145236


namespace NUMINAMATH_GPT_ratio_b4_b3_a2_a1_l1452_145217

variables {x y d d' : ℝ}
variables {a1 a2 a3 b1 b2 b3 b4 : ℝ}
-- Conditions
variables (h1 : x ≠ y)
variables (h2 : a1 = x + d)
variables (h3 : a2 = x + 2 * d)
variables (h4 : a3 = x + 3 * d)
variables (h5 : y = x + 4 * d)
variables (h6 : b2 = x + d')
variables (h7 : b3 = x + 2 * d')
variables (h8 : y = x + 3 * d')
variables (h9 : b4 = x + 4 * d')

theorem ratio_b4_b3_a2_a1 :
  (b4 - b3) / (a2 - a1) = 8 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_b4_b3_a2_a1_l1452_145217


namespace NUMINAMATH_GPT_find_number_l1452_145221

theorem find_number (x : ℝ) : 
  10 * ((2 * (x * x + 2) + 3) / 5) = 50 → x = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1452_145221


namespace NUMINAMATH_GPT_at_least_one_fraction_lt_two_l1452_145216

theorem at_least_one_fraction_lt_two 
  (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : 2 < x + y) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_fraction_lt_two_l1452_145216


namespace NUMINAMATH_GPT_tickets_sold_l1452_145290

def advanced_purchase_tickets := ℕ
def door_purchase_tickets := ℕ

variable (A D : ℕ)

theorem tickets_sold :
  (A + D = 140) →
  (8 * A + 14 * D = 1720) →
  A = 40 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_tickets_sold_l1452_145290


namespace NUMINAMATH_GPT_value_of_a_l1452_145248

theorem value_of_a (a b c : ℕ) (h1 : a + b = 12) (h2 : b + c = 16) (h3 : c = 7) : a = 3 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l1452_145248


namespace NUMINAMATH_GPT_number_of_boys_is_12500_l1452_145226

-- Define the number of boys and girls in the school
def numberOfBoys (B : ℕ) : ℕ := B
def numberOfGirls : ℕ := 5000

-- Define the total attendance
def totalAttendance (B : ℕ) : ℕ := B + numberOfGirls

-- Define the condition for the percentage increase from boys to total attendance
def percentageIncreaseCondition (B : ℕ) : Prop :=
  totalAttendance B = B + Int.ofNat numberOfGirls

-- Statement to prove
theorem number_of_boys_is_12500 (B : ℕ) (h : totalAttendance B = B + numberOfGirls) : B = 12500 :=
sorry

end NUMINAMATH_GPT_number_of_boys_is_12500_l1452_145226


namespace NUMINAMATH_GPT_housewife_spending_l1452_145228

theorem housewife_spending (P R A : ℝ) (h1 : R = 34.2) (h2 : R = 0.8 * P) (h3 : A / R - A / P = 4) :
  A = 683.45 :=
by
  sorry

end NUMINAMATH_GPT_housewife_spending_l1452_145228


namespace NUMINAMATH_GPT_odd_factor_form_l1452_145287

theorem odd_factor_form (n : ℕ) (x y : ℕ) (h_n : n > 0) (h_gcd : Nat.gcd x y = 1) :
  ∀ p, p ∣ (x ^ (2 ^ n) + y ^ (2 ^ n)) ∧ Odd p → ∃ k > 0, p = 2^(n+1) * k + 1 := 
by
  sorry

end NUMINAMATH_GPT_odd_factor_form_l1452_145287


namespace NUMINAMATH_GPT_acute_angle_at_9_35_is_77_5_degrees_l1452_145225

def degrees_in_acute_angle_formed_by_hands_of_clock_9_35 : ℝ := 77.5

theorem acute_angle_at_9_35_is_77_5_degrees 
  (hour_angle : ℝ := 270 + (35/60 * 30))
  (minute_angle : ℝ := 35/60 * 360) : 
  |hour_angle - minute_angle| < 180 → |hour_angle - minute_angle| = degrees_in_acute_angle_formed_by_hands_of_clock_9_35 := 
by 
  sorry

end NUMINAMATH_GPT_acute_angle_at_9_35_is_77_5_degrees_l1452_145225


namespace NUMINAMATH_GPT_final_price_is_correct_l1452_145299

/-- 
  The original price of a suit is $200.
-/
def original_price : ℝ := 200

/-- 
  The price increased by 25%, therefore the increase is 25% of the original price.
-/
def increase : ℝ := 0.25 * original_price

/-- 
  The new price after the price increase.
-/
def increased_price : ℝ := original_price + increase

/-- 
  After the increase, a 25% off coupon is applied.
-/
def discount : ℝ := 0.25 * increased_price

/-- 
  The final price consumers pay for the suit.
-/
def final_price : ℝ := increased_price - discount

/-- 
  Prove that the consumers paid $187.50 for the suit.
-/
theorem final_price_is_correct : final_price = 187.50 :=
by sorry

end NUMINAMATH_GPT_final_price_is_correct_l1452_145299


namespace NUMINAMATH_GPT_number_of_unit_distance_pairs_lt_bound_l1452_145207

/-- Given n distinct points in the plane, the number of pairs of points with a unit distance between them is less than n / 4 + (1 / sqrt 2) * n^(3 / 2). -/
theorem number_of_unit_distance_pairs_lt_bound (n : ℕ) (hn : 0 < n) :
  ∃ E : ℕ, E < n / 4 + (1 / Real.sqrt 2) * n^(3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_number_of_unit_distance_pairs_lt_bound_l1452_145207


namespace NUMINAMATH_GPT_fraction_multiplication_l1452_145238

theorem fraction_multiplication :
  ((3 : ℚ) / 4) ^ 3 * ((2 : ℚ) / 5) ^ 3 = (27 : ℚ) / 1000 := sorry

end NUMINAMATH_GPT_fraction_multiplication_l1452_145238


namespace NUMINAMATH_GPT_problem_solution_l1452_145256

noncomputable def f (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^2 - p * x + q

theorem problem_solution
  (a b p q : ℝ)
  (h1 : a ≠ b)
  (h2 : p > 0)
  (h3 : q > 0)
  (h4 : f a p q = 0)
  (h5 : f b p q = 0)
  (h6 : ∃ k : ℝ, (a = -2 + k ∧ b = -2 - k) ∨ (a = -2 - k ∧ b = -2 + k))
  (h7 : ∃ l : ℝ, (a = -2 * l ∧ b = 4 * l) ∨ (a = 4 * l ∧ b = -2 * l))
  : p + q = 9 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1452_145256


namespace NUMINAMATH_GPT_cost_of_calf_l1452_145257

theorem cost_of_calf (C : ℝ) (total_cost : ℝ) (cow_to_calf_ratio : ℝ) :
  total_cost = 990 ∧ cow_to_calf_ratio = 8 ∧ total_cost = C + 8 * C → C = 110 := by
  sorry

end NUMINAMATH_GPT_cost_of_calf_l1452_145257


namespace NUMINAMATH_GPT_fraction_uninterested_students_interested_l1452_145224

theorem fraction_uninterested_students_interested 
  (students : Nat)
  (interest_ratio : ℚ)
  (express_interest_ratio_if_interested : ℚ)
  (express_disinterest_ratio_if_not_interested : ℚ) 
  (h1 : students > 0)
  (h2 : interest_ratio = 0.70)
  (h3 : express_interest_ratio_if_interested = 0.75)
  (h4 : express_disinterest_ratio_if_not_interested = 0.85) :
  let interested_students := students * interest_ratio
  let not_interested_students := students * (1 - interest_ratio)
  let express_interest_and_interested := interested_students * express_interest_ratio_if_interested
  let not_express_interest_and_interested := interested_students * (1 - express_interest_ratio_if_interested)
  let express_disinterest_and_not_interested := not_interested_students * express_disinterest_ratio_if_not_interested
  let express_interest_and_not_interested := not_interested_students * (1 - express_disinterest_ratio_if_not_interested)
  let not_express_interest_total := not_express_interest_and_interested + express_disinterest_and_not_interested
  let fraction := not_express_interest_and_interested / not_express_interest_total
  fraction = 0.407 := 
by
  sorry

end NUMINAMATH_GPT_fraction_uninterested_students_interested_l1452_145224


namespace NUMINAMATH_GPT_sqrt_of_4_l1452_145285

theorem sqrt_of_4 :
  ∃ x : ℝ, x^2 = 4 ∧ (x = 2 ∨ x = -2) :=
sorry

end NUMINAMATH_GPT_sqrt_of_4_l1452_145285


namespace NUMINAMATH_GPT_more_white_animals_than_cats_l1452_145241

theorem more_white_animals_than_cats (C W WC : ℕ) 
  (h1 : WC = C / 3) 
  (h2 : WC = W / 6) : W = 2 * C :=
by {
  sorry
}

end NUMINAMATH_GPT_more_white_animals_than_cats_l1452_145241


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1452_145219

theorem simplify_and_evaluate (a : ℤ) (h : a = 0) : 
  ((a / (a - 1) : ℚ) + ((a + 1) / (a^2 - 1) : ℚ)) = (-1 : ℚ) := by
  have ha_ne1 : a ≠ 1 := by norm_num [h]
  have ha_ne_neg1 : a ≠ -1 := by norm_num [h]
  have h1 : (a^2 - 1) ≠ 0 := by
    rw [sub_ne_zero]
    norm_num [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1452_145219


namespace NUMINAMATH_GPT_additional_people_needed_l1452_145286

-- Define the initial number of people and time they take to mow the lawn 
def initial_people : ℕ := 8
def initial_time : ℕ := 3

-- Define total person-hours required to mow the lawn
def total_person_hours : ℕ := initial_people * initial_time

-- Define the time in which we want to find out how many people can mow the lawn
def desired_time : ℕ := 2

-- Define the number of people needed in desired_time to mow the lawn
def required_people : ℕ := total_person_hours / desired_time

-- Define the additional people required to mow the lawn in desired_time
def additional_people : ℕ := required_people - initial_people

-- Statement to be proved
theorem additional_people_needed : additional_people = 4 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_additional_people_needed_l1452_145286


namespace NUMINAMATH_GPT_second_daily_rate_l1452_145229

noncomputable def daily_rate_sunshine : ℝ := 17.99
noncomputable def mileage_cost_sunshine : ℝ := 0.18
noncomputable def mileage_cost_second : ℝ := 0.16
noncomputable def distance : ℝ := 48.0

theorem second_daily_rate (daily_rate_second : ℝ) : 
  daily_rate_sunshine + (mileage_cost_sunshine * distance) = 
  daily_rate_second + (mileage_cost_second * distance) → 
  daily_rate_second = 18.95 :=
by 
  sorry

end NUMINAMATH_GPT_second_daily_rate_l1452_145229


namespace NUMINAMATH_GPT_find_m_l1452_145227

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x + α / x + Real.log x

theorem find_m (α : ℝ) (m : ℝ) (l e : ℝ) (hα_range : α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 2))
(h1 : f 1 α < m) (he : f (Real.exp 1) α < m) :
m > 1 + 2 * Real.exp 2 := by
  sorry

end NUMINAMATH_GPT_find_m_l1452_145227


namespace NUMINAMATH_GPT_polygon_sides_count_l1452_145210

theorem polygon_sides_count :
    ∀ (n1 n2 n3 n4 n5 n6 : ℕ),
    n1 = 3 ∧ n2 = 4 ∧ n3 = 5 ∧ n4 = 6 ∧ n5 = 7 ∧ n6 = 8 →
    (n1 - 2) + (n2 - 2) + (n3 - 2) + (n4 - 2) + (n5 - 2) + (n6 - 1) + 3 = 24 :=
by
  intros n1 n2 n3 n4 n5 n6 h
  sorry

end NUMINAMATH_GPT_polygon_sides_count_l1452_145210


namespace NUMINAMATH_GPT_triangle_inequality_for_n6_l1452_145222

variables {a b c : ℝ} {n : ℕ}
open Real

-- Define the main statement as a theorem
theorem triangle_inequality_for_n6 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end NUMINAMATH_GPT_triangle_inequality_for_n6_l1452_145222


namespace NUMINAMATH_GPT_f_f_2_l1452_145265

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 2 then 2 * Real.exp (x - 1) else Real.log (2^x - 1) / Real.log 3

theorem f_f_2 : f (f 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_f_f_2_l1452_145265


namespace NUMINAMATH_GPT_average_speed_l1452_145293

-- Define the problem conditions and provide the proof statement
theorem average_speed (D : ℝ) (hD0 : D > 0) : 
  let speed_1 := 80
  let speed_2 := 24
  let speed_3 := 60
  let time_1 := (D / 3) / speed_1
  let time_2 := (D / 3) / speed_2
  let time_3 := (D / 3) / speed_3
  let total_time := time_1 + time_2 + time_3
  let average_speed := D / total_time
  average_speed = 720 / 17 := 
by
  sorry

end NUMINAMATH_GPT_average_speed_l1452_145293


namespace NUMINAMATH_GPT_sin_cos_product_l1452_145267

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end NUMINAMATH_GPT_sin_cos_product_l1452_145267


namespace NUMINAMATH_GPT_find_x_values_l1452_145260

theorem find_x_values (x1 x2 x3 x4 : ℝ)
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) :
  x1 = 4 / 5 ∧ x2 = 3 / 5 ∧ x3 = 2 / 5 ∧ x4 = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_values_l1452_145260


namespace NUMINAMATH_GPT_tiffany_optimal_area_l1452_145289

def optimal_area (A : ℕ) : Prop :=
  ∃ l w : ℕ, l + w = 160 ∧ l ≥ 85 ∧ w ≥ 45 ∧ A = l * w

theorem tiffany_optimal_area : optimal_area 6375 :=
  sorry

end NUMINAMATH_GPT_tiffany_optimal_area_l1452_145289


namespace NUMINAMATH_GPT_find_x_l1452_145274

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l1452_145274


namespace NUMINAMATH_GPT_amusing_permutations_formula_l1452_145246

-- Definition of amusing permutations and their count
def amusing_permutations_count (n : ℕ) : ℕ :=
  2^(n-1)

-- Theorem statement: The number of amusing permutations of the set {1, 2, ..., n} is 2^(n-1)
theorem amusing_permutations_formula (n : ℕ) : 
  -- The number of amusing permutations should be equal to 2^(n-1)
  amusing_permutations_count n = 2^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_amusing_permutations_formula_l1452_145246


namespace NUMINAMATH_GPT_geometric_quadratic_root_l1452_145261

theorem geometric_quadratic_root (a b c : ℝ) (h1 : a > 0) (h2 : b = a * (1 / 4)) (h3 : c = a * (1 / 16)) (h4 : a * a * (1 / 4)^2 = 4 * a * a * (1 / 16)) : 
    -b / (2 * a) = -1 / 8 :=
by 
    sorry

end NUMINAMATH_GPT_geometric_quadratic_root_l1452_145261


namespace NUMINAMATH_GPT_pascal_triangle_10_to_30_l1452_145215

-- Definitions
def pascal_row_numbers (n : ℕ) : ℕ := n + 1

def total_numbers_up_to (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

-- Proof Statement
theorem pascal_triangle_10_to_30 :
  total_numbers_up_to 29 - total_numbers_up_to 9 = 400 := by
  sorry

end NUMINAMATH_GPT_pascal_triangle_10_to_30_l1452_145215


namespace NUMINAMATH_GPT_compare_exponents_l1452_145264

noncomputable def exp_of_log (a : ℝ) (b : ℝ) : ℝ :=
  Real.exp ((1 / b) * Real.log a)

theorem compare_exponents :
  let a := exp_of_log 4 4
  let b := exp_of_log 5 5
  let c := exp_of_log 16 16
  let d := exp_of_log 25 25
  a = max a (max b (max c d)) ∧
  b = max (min a (max b (max c d))) (max (min b (max c d)) (max (min c d) (min d (min a b))))
  :=
  by
    sorry

end NUMINAMATH_GPT_compare_exponents_l1452_145264


namespace NUMINAMATH_GPT_ratio_of_volumes_l1452_145235

-- Define the edge lengths
def edge_length_cube1 : ℝ := 9
def edge_length_cube2 : ℝ := 24

-- Theorem stating the ratio of the volumes
theorem ratio_of_volumes :
  (edge_length_cube1 / edge_length_cube2) ^ 3 = 27 / 512 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_l1452_145235


namespace NUMINAMATH_GPT_smallest_b_l1452_145269

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x - 2 * a

theorem smallest_b (a : ℝ) (b : ℝ) (x : ℝ) : (1 < a ∧ a < 4) → (0 < x) → (f a b x > 0) → b ≥ 11 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_smallest_b_l1452_145269


namespace NUMINAMATH_GPT_trajectory_eq_range_of_k_l1452_145243

-- definitions based on the conditions:
def fixed_circle (x y : ℝ) := (x + 1)^2 + y^2 = 16
def moving_circle_passing_through_B (M : ℝ × ℝ) (B : ℝ × ℝ) := 
    B = (1, 0) ∧ M.1^2 / 4 + M.2^2 / 3 = 1 -- the ellipse trajectory equation

-- question 1: prove the equation of the ellipse
theorem trajectory_eq :
    ∀ M : ℝ × ℝ, (∃ B : ℝ × ℝ, moving_circle_passing_through_B M B)
    → (M.1^2 / 4 + M.2^2 / 3 = 1) :=
sorry

-- question 2: find the range of k which satisfies given area condition
theorem range_of_k (k : ℝ) :
    (∃ M : ℝ × ℝ, ∃ B : ℝ × ℝ, moving_circle_passing_through_B M B) → 
    (0 < k) → (¬ (k = 0)) →
    ((∃ m : ℝ, (4 * k^2 + 3 - m^2 > 0) ∧ 
    (1 / 2) * (|k| * m^2 / (4 * k^2 + 3)^2) = 1 / 14) → (3 / 4 < k ∧ k < 1) 
    ∨ (-1 < k ∧ k < -3 / 4)) :=
sorry

end NUMINAMATH_GPT_trajectory_eq_range_of_k_l1452_145243


namespace NUMINAMATH_GPT_parallel_lines_m_value_l1452_145239

theorem parallel_lines_m_value (x y m : ℝ) (h₁ : 2 * x + m * y - 7 = 0) (h₂ : m * x + 8 * y - 14 = 0) (parallel : (2 / m = m / 8)) : m = -4 := 
sorry

end NUMINAMATH_GPT_parallel_lines_m_value_l1452_145239


namespace NUMINAMATH_GPT_next_term_geometric_sequence_l1452_145272

theorem next_term_geometric_sequence (x : ℝ) (r : ℝ) (a₀ a₃ next_term : ℝ)
    (h1 : a₀ = 2)
    (h2 : r = 3 * x)
    (h3 : a₃ = 54 * x^3)
    (h4 : next_term = a₃ * r) :
    next_term = 162 * x^4 := by
  sorry

end NUMINAMATH_GPT_next_term_geometric_sequence_l1452_145272


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_neg_s_sufficient_not_necessary_for_neg_q_l1452_145252

-- Define conditions
def p (x : ℝ) : Prop := -x^2 + 2 * x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def s (x : ℝ) : Prop := -x^2 + 8 * x + 20 ≥ 0

variable {x m : ℝ}

-- Question 1
theorem p_sufficient_not_necessary_for_q (hp : ∀ x, p x → q x m) : m ≥ 3 :=
sorry

-- Defining negation of s and q
def neg_s (x : ℝ) : Prop := ¬s x
def neg_q (x m : ℝ) : Prop := ¬q x m

-- Question 2
theorem neg_s_sufficient_not_necessary_for_neg_q (hp : ∀ x, neg_s x → neg_q x m) : false :=
sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_neg_s_sufficient_not_necessary_for_neg_q_l1452_145252


namespace NUMINAMATH_GPT_center_of_hyperbola_l1452_145283

-- Define the given equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  ((3 * y - 6)^2 / 8^2) - ((4 * x - 5)^2 / 3^2) = 1

-- Prove that the center of the hyperbola is (5 / 4, 2)
theorem center_of_hyperbola :
  (∃ h k : ℝ, h = 5 / 4 ∧ k = 2 ∧ ∀ x y : ℝ, hyperbola_eq x y ↔ ((y - k)^2 / (8 / 3)^2 - (x - h)^2 / (3 / 4)^2 = 1)) :=
sorry

end NUMINAMATH_GPT_center_of_hyperbola_l1452_145283


namespace NUMINAMATH_GPT_find_unknown_number_l1452_145212

theorem find_unknown_number (x : ℝ) (h : (2 / 3) * x + 6 = 10) : x = 6 :=
  sorry

end NUMINAMATH_GPT_find_unknown_number_l1452_145212
