import Mathlib

namespace minimum_daily_expense_l393_39322

-- Defining the context
variables (x y : ℕ)
def total_capacity (x y : ℕ) : ℕ := 24 * x + 30 * y
def cost (x y : ℕ) : ℕ := 320 * x + 504 * y

theorem minimum_daily_expense :
  (total_capacity x y ≥ 180) →
  (x ≤ 8) →
  (y ≤ 4) →
  cost x y = 2560 := sorry

end minimum_daily_expense_l393_39322


namespace ellipse_equation_l393_39371

theorem ellipse_equation (e : ℝ) (P : ℝ × ℝ) (d_max : ℝ) (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
    (h3 : e = Real.sqrt 3 / 2) (h4 : P = (0, 3 / 2)) (h5 : ∀ P1 : ℝ × ℝ, (P1.1 ^ 2 / a ^ 2 + P1.2 ^ 2 / b ^ 2 = 1) → 
    ∃ P2 : ℝ × ℝ, dist P P2 = d_max ∧ (P2.1 ^ 2 / a ^ 2 + P2.2 ^ 2 / b ^ 2 = 1)) :
  (a = 2 ∧ b = 1) → (∀ x y : ℝ, (x ^ 2 / 4) + y ^ 2 ≤ 1) := by
  sorry

end ellipse_equation_l393_39371


namespace combined_ratio_is_1_l393_39378

-- Conditions
variables (V1 V2 M1 W1 M2 W2 : ℝ)
variables (x : ℝ)
variables (ratio_volumes ratio_milk_water_v1 ratio_milk_water_v2 : ℝ)

-- Given conditions as hypotheses
-- Condition: V1 / V2 = 3 / 5
-- Hypothesis 1: The volume ratio of the first and second vessels
def volume_ratio : Prop :=
  V1 / V2 = 3 / 5

-- Condition: M1 / W1 = 1 / 2 in first vessel
-- Hypothesis 2: The milk to water ratio in the first vessel
def milk_water_ratio_v1 : Prop :=
  M1 / W1 = 1 / 2

-- Condition: M2 / W2 = 3 / 2 in the second vessel
-- Hypothesis 3: The milk to water ratio in the second vessel
def milk_water_ratio_v2 : Prop :=
  M2 / W2 = 3 / 2

-- Definition: Total volumes of milk and water in the larger vessel
def total_milk_water_ratio : Prop :=
  (M1 + M2) / (W1 + W2) = 1 / 1

-- Main theorem: Given the ratios, the ratio of milk to water in the larger vessel is 1:1
theorem combined_ratio_is_1 :
  (volume_ratio V1 V2) →
  (milk_water_ratio_v1 M1 W1) →
  (milk_water_ratio_v2 M2 W2) →
  total_milk_water_ratio M1 W1 M2 W2 :=
by
  -- Proof omitted
  sorry

end combined_ratio_is_1_l393_39378


namespace meal_combinations_l393_39392

def menu_items : ℕ := 12
def special_dish_chosen : Prop := true

theorem meal_combinations : (special_dish_chosen → (menu_items - 1) * (menu_items - 1) = 121) :=
by
  sorry

end meal_combinations_l393_39392


namespace calculate_expression_l393_39370

theorem calculate_expression :
  ((16^10 / 16^8) ^ 3 * 8 ^ 3) / 2 ^ 9 = 16777216 := by
  sorry

end calculate_expression_l393_39370


namespace maximum_value_of_f_l393_39308

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 - 3 * x else -2 * x + 1

theorem maximum_value_of_f : ∃ (m : ℝ), (∀ x : ℝ, f x ≤ m) ∧ (m = 2) := by
  sorry

end maximum_value_of_f_l393_39308


namespace golden_ratio_problem_l393_39303

theorem golden_ratio_problem
  (m n : ℝ) (sin cos : ℝ → ℝ)
  (h1 : m = 2 * sin (Real.pi / 10))
  (h2 : m ^ 2 + n = 4)
  (sin63 : sin (7 * Real.pi / 18) ≠ 0) :
  (m + Real.sqrt n) / (sin (7 * Real.pi / 18)) = 2 * Real.sqrt 2 := by
  sorry

end golden_ratio_problem_l393_39303


namespace khalil_total_payment_l393_39352

def cost_dog := 60
def cost_cat := 40
def cost_parrot := 70
def cost_rabbit := 50

def num_dogs := 25
def num_cats := 45
def num_parrots := 15
def num_rabbits := 10

def total_cost := num_dogs * cost_dog + num_cats * cost_cat + num_parrots * cost_parrot + num_rabbits * cost_rabbit

theorem khalil_total_payment : total_cost = 4850 := by
  sorry

end khalil_total_payment_l393_39352


namespace wealth_ratio_l393_39396

theorem wealth_ratio 
  (P W : ℝ)
  (hP_pos : 0 < P)
  (hW_pos : 0 < W)
  (pop_A : ℝ := 0.30 * P)
  (wealth_A : ℝ := 0.40 * W)
  (pop_B : ℝ := 0.20 * P)
  (wealth_B : ℝ := 0.25 * W)
  (avg_wealth_A : ℝ := wealth_A / pop_A)
  (avg_wealth_B : ℝ := wealth_B / pop_B) :
  avg_wealth_A / avg_wealth_B = 16 / 15 :=
by
  sorry

end wealth_ratio_l393_39396


namespace initial_carrots_l393_39324

theorem initial_carrots (n : ℕ) 
    (h1: 3640 = 180 * (n - 4) + 760) 
    (h2: 180 * (n - 4) < 3640) 
    (h3: 4 * 190 = 760) : 
    n = 20 :=
by
  sorry

end initial_carrots_l393_39324


namespace quadratic_completing_square_l393_39394

theorem quadratic_completing_square:
  ∃ (b c : ℝ), (∀ x : ℝ, x^2 + 900 * x + 1800 = (x + b)^2 + c) ∧ (c / b = -446.22222) :=
by
  -- We'll skip the proof steps here
  sorry

end quadratic_completing_square_l393_39394


namespace vector_at_t5_l393_39376

theorem vector_at_t5 :
  ∃ (a : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ),
    a + (1 : ℝ) • d = (2, -1, 3) ∧
    a + (4 : ℝ) • d = (8, -5, 11) ∧
    a + (5 : ℝ) • d = (10, -19/3, 41/3) := 
sorry

end vector_at_t5_l393_39376


namespace fraction_b_plus_c_over_a_l393_39329

variable (a b c d : ℝ)

theorem fraction_b_plus_c_over_a :
  (a ≠ 0) →
  (a * 4^3 + b * 4^2 + c * 4 + d = 0) →
  (a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) →
  (b + c) / a = -13 :=
by
  intros h₁ h₂ h₃ 
  sorry

end fraction_b_plus_c_over_a_l393_39329


namespace solution_set_of_inequality_l393_39362

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2
  else if x > 0 then x - 2
  else 0

theorem solution_set_of_inequality :
  {x : ℝ | 2 * f x - 1 < 0} = {x | x < -3 / 2 ∨ (0 ≤ x ∧ x < 5 / 2)} :=
by
  sorry

end solution_set_of_inequality_l393_39362


namespace find_m_l393_39328

-- Definition of the function as a direct proportion function with respect to x
def isDirectProportion (m : ℝ) : Prop :=
  m^2 - 8 = 1

-- Definition of the graph passing through the second and fourth quadrants
def passesThroughQuadrants (m : ℝ) : Prop :=
  m - 2 < 0

-- The theorem combining the conditions and proving the correct value of m
theorem find_m (m : ℝ) 
  (h1 : isDirectProportion m)
  (h2 : passesThroughQuadrants m) : 
  m = -3 :=
  sorry

end find_m_l393_39328


namespace find_sum_l393_39386

theorem find_sum (P : ℕ) (h_total : P * (4/100 + 6/100 + 8/100) = 2700) : P = 15000 :=
by
  sorry

end find_sum_l393_39386


namespace dogwood_trees_proof_l393_39345

def dogwood_trees_left (a b c : Float) : Float :=
  a + b - c

theorem dogwood_trees_proof : dogwood_trees_left 5.0 4.0 7.0 = 2.0 :=
by
  -- The proof itself is left out intentionally as per the instructions
  sorry

end dogwood_trees_proof_l393_39345


namespace poles_needed_l393_39374

theorem poles_needed (L W : ℕ) (dist : ℕ)
  (hL : L = 90) (hW : W = 40) (hdist : dist = 5) :
  (2 * (L + W)) / dist = 52 :=
by 
  sorry

end poles_needed_l393_39374


namespace net_rate_of_pay_l393_39302

theorem net_rate_of_pay :
  ∀ (duration_travel : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (earnings_rate : ℝ) (gas_cost : ℝ),
  duration_travel = 3 → speed = 50 → fuel_efficiency = 30 → earnings_rate = 0.75 → gas_cost = 2.50 →
  (earnings_rate * speed * duration_travel - (speed * duration_travel / fuel_efficiency) * gas_cost) / duration_travel = 33.33 :=
by
  intros duration_travel speed fuel_efficiency earnings_rate gas_cost
  intros h1 h2 h3 h4 h5
  sorry

end net_rate_of_pay_l393_39302


namespace necessary_and_sufficient_condition_perpendicular_lines_l393_39313

def are_perpendicular (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x + y = 0) → (x - a * y = 0) → x = 0 ∧ y = 0

theorem necessary_and_sufficient_condition_perpendicular_lines :
  ∀ (a : ℝ), are_perpendicular a → a = 1 :=
sorry

end necessary_and_sufficient_condition_perpendicular_lines_l393_39313


namespace sum_of_ages_of_cousins_l393_39349

noncomputable def is_valid_age_group (a b c d : ℕ) : Prop :=
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧
  (1 ≤ a) ∧ (a ≤ 9) ∧ (1 ≤ b) ∧ (b ≤ 9) ∧ (1 ≤ c) ∧ (c ≤ 9) ∧ (1 ≤ d) ∧ (d ≤ 9)

theorem sum_of_ages_of_cousins :
  ∃ (a b c d : ℕ), is_valid_age_group a b c d ∧ (a * b = 40) ∧ (c * d = 36) ∧ (a + b + c + d = 26) := 
sorry

end sum_of_ages_of_cousins_l393_39349


namespace chemist_salt_solution_l393_39351

theorem chemist_salt_solution (x : ℝ) 
  (hx : 0.60 * x = 0.20 * (1 + x)) : x = 0.5 :=
sorry

end chemist_salt_solution_l393_39351


namespace sum_of_abcd_l393_39388

theorem sum_of_abcd (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : ∀ x, x^2 - 8*a*x - 9*b = 0 → x = c ∨ x = d)
  (h2 : ∀ x, x^2 - 8*c*x - 9*d = 0 → x = a ∨ x = b) :
  a + b + c + d = 648 := sorry

end sum_of_abcd_l393_39388


namespace Jeremy_payment_total_l393_39338

theorem Jeremy_payment_total :
  let room_rate := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let window_rate := (5 : ℚ) / 2
  let windows_cleaned := (11 : ℚ) / 4
  let payment_rooms := room_rate * rooms_cleaned
  let payment_windows := window_rate * windows_cleaned
  let total_payment := payment_rooms + payment_windows
  total_payment = (553 : ℚ) / 40 :=
by {
  -- Definitions
  let room_rate := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let window_rate := (5 : ℚ) / 2
  let windows_cleaned := (11 : ℚ) / 4
  let payment_rooms := room_rate * rooms_cleaned
  let payment_windows := window_rate * windows_cleaned
  let total_payment := payment_rooms + payment_windows
  
  -- Main goal
  sorry
}

end Jeremy_payment_total_l393_39338


namespace inequality_proof_l393_39307

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b+c)^2) + (b^2 + 9) / (2*b^2 + (c+a)^2) + (c^2 + 9) / (2*c^2 + (a+b)^2) ≤ 5 :=
by
  sorry

end inequality_proof_l393_39307


namespace completion_time_l393_39377

variables {P E : ℝ}
theorem completion_time (h1 : (20 : ℝ) * P * E / 2 = D * (2.5 * P * E)) : D = 4 :=
by
  -- Given h1 as the condition
  sorry

end completion_time_l393_39377


namespace solution_set_range_ineq_l393_39321

theorem solution_set_range_ineq (m : ℝ) :
  ∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0 ↔ (-5: ℝ)⁻¹ < m ∧ m ≤ 3 :=
by
  sorry

end solution_set_range_ineq_l393_39321


namespace quadratic_root_inequality_l393_39336

theorem quadratic_root_inequality (a : ℝ) :
  2015 < a ∧ a < 2017 ↔ 
  ∃ x₁ x₂ : ℝ, (2 * x₁^2 - 2016 * (x₁ - 2016 + a) - 1 = a^2) ∧ 
               (2 * x₂^2 - 2016 * (x₂ - 2016 + a) - 1 = a^2) ∧
               x₁ < a ∧ a < x₂ :=
sorry

end quadratic_root_inequality_l393_39336


namespace number_of_valid_pairs_l393_39346

theorem number_of_valid_pairs :
  (∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2044 ∧ 5^n < 2^m ∧ 2^m < 2^(m + 1) ∧ 2^(m + 1) < 5^(n + 1)) ↔
  ((∃ (x y : ℕ), 2^2100 < 5^900 ∧ 5^900 < 2^2101)) → 
  (∃ (count : ℕ), count = 900) :=
by sorry

end number_of_valid_pairs_l393_39346


namespace points_five_from_origin_l393_39360

theorem points_five_from_origin (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end points_five_from_origin_l393_39360


namespace min_m_l393_39334

theorem min_m (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
by
  sorry

end min_m_l393_39334


namespace find_m_value_l393_39330

theorem find_m_value
  (y_squared_4x : ∀ x y : ℝ, y^2 = 4 * x)
  (Focus_F : ℝ × ℝ)
  (M N : ℝ × ℝ)
  (E : ℝ)
  (P Q : ℝ × ℝ)
  (k1 k2 : ℝ)
  (MN_slope : k1 = (N.snd - M.snd) / (N.fst - M.fst))
  (PQ_slope : k2 = (Q.snd - P.snd) / (Q.fst - P.fst))
  (slope_condition : k1 = 3 * k2) :
  E = 3 := 
sorry

end find_m_value_l393_39330


namespace remainder_8_pow_1996_mod_5_l393_39389

theorem remainder_8_pow_1996_mod_5 :
  (8: ℕ) ≡ 3 [MOD 5] →
  3^4 ≡ 1 [MOD 5] →
  8^1996 ≡ 1 [MOD 5] :=
by
  sorry

end remainder_8_pow_1996_mod_5_l393_39389


namespace divide_into_two_groups_l393_39327

theorem divide_into_two_groups (n : ℕ) (A : Fin n → Type) 
  (acquaintances : (Fin n) → (Finset (Fin n)))
  (c : (Fin n) → ℕ) (d : (Fin n) → ℕ) :
  (∀ i : Fin n, c i = (acquaintances i).card) →
  ∃ G1 G2 : Finset (Fin n), G1 ∩ G2 = ∅ ∧ G1 ∪ G2 = Finset.univ ∧
  (∀ i : Fin n, d i = (acquaintances i ∩ (if i ∈ G1 then G2 else G1)).card ∧ d i ≥ (c i) / 2) :=
by 
  sorry

end divide_into_two_groups_l393_39327


namespace ArletteAge_l393_39337

/-- Define the ages of Omi, Kimiko, and Arlette -/
def OmiAge (K : ℕ) : ℕ := 2 * K
def KimikoAge : ℕ := 28   /- K = 28 -/
def averageAge (O K A : ℕ) : Prop := (O + K + A) / 3 = 35

/-- Prove Arlette's age given the conditions -/
theorem ArletteAge (A : ℕ) (h1 : A + OmiAge KimikoAge + KimikoAge = 3 * 35) : A = 21 := by
  /- Hypothesis h1 unpacks the third condition into equality involving O, K, and A -/
  sorry

end ArletteAge_l393_39337


namespace fish_price_eq_shrimp_price_l393_39368

-- Conditions
variable (x : ℝ) -- regular price for a full pound of fish
variable (h1 : 0.6 * (x / 4) = 1.50) -- quarter-pound fish price after 60% discount
variable (shrimp_price : ℝ) -- price per pound of shrimp
variable (h2 : shrimp_price = 10) -- given shrimp price

-- Proof Statement
theorem fish_price_eq_shrimp_price (h1 : 0.6 * (x / 4) = 1.50) (h2 : shrimp_price = 10) :
  x = 10 ∧ x = shrimp_price :=
by
  sorry

end fish_price_eq_shrimp_price_l393_39368


namespace meals_per_day_l393_39318

-- Definitions based on given conditions
def number_of_people : Nat := 6
def total_plates_used : Nat := 144
def number_of_days : Nat := 4
def plates_per_meal : Nat := 2

-- Theorem to prove
theorem meals_per_day : (total_plates_used / number_of_days) / plates_per_meal / number_of_people = 3 :=
by
  sorry

end meals_per_day_l393_39318


namespace angle_between_v1_v2_l393_39325

-- Define vectors
def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (4, 6)

-- Define the dot product function
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Define the magnitude function
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the cosine of the angle between two vectors
noncomputable def cos_theta (a b : ℝ × ℝ) : ℝ := (dot_product a b) / (magnitude a * magnitude b)

-- Define the angle in degrees between two vectors
noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := Real.arccos (cos_theta a b) * (180 / Real.pi)

-- The statement to prove
theorem angle_between_v1_v2 : angle_between_vectors v1 v2 = Real.arccos (-6 * Real.sqrt 13 / 65) * (180 / Real.pi) :=
sorry

end angle_between_v1_v2_l393_39325


namespace part1_part2_l393_39350

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2
noncomputable def g (x a : ℝ) : ℝ := x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x

theorem part1 : ∃ xₘ : ℝ, (∀ x > 0, f x ≤ f xₘ) ∧ f xₘ = -1 :=
by sorry

theorem part2 (a : ℝ) : (∀ x > 0, f x + g x a ≥ 0) ↔ a ≤ 1 :=
by sorry

end part1_part2_l393_39350


namespace fish_remaining_l393_39395

def initial_fish : ℝ := 47.0
def given_away_fish : ℝ := 22.5

theorem fish_remaining : initial_fish - given_away_fish = 24.5 :=
by
  sorry

end fish_remaining_l393_39395


namespace quiz_answer_keys_count_l393_39367

noncomputable def count_answer_keys : ℕ :=
  (Nat.choose 10 5) * (Nat.factorial 6)

theorem quiz_answer_keys_count :
  count_answer_keys = 181440 := 
by
  -- Proof is skipped, using sorry
  sorry

end quiz_answer_keys_count_l393_39367


namespace perimeter_of_triangle_is_36_l393_39353

variable (inradius : ℝ)
variable (area : ℝ)
variable (P : ℝ)

theorem perimeter_of_triangle_is_36 (h1 : inradius = 2.5) (h2 : area = 45) : 
  P / 2 * inradius = area → P = 36 :=
sorry

end perimeter_of_triangle_is_36_l393_39353


namespace smaller_angle_between_east_and_northwest_l393_39348

theorem smaller_angle_between_east_and_northwest
  (rays : ℕ)
  (each_angle : ℕ)
  (direction : ℕ → ℝ)
  (h1 : rays = 10)
  (h2 : each_angle = 36)
  (h3 : direction 0 = 0) -- ray at due North
  (h4 : direction 3 = 90) -- ray at due East
  (h5 : direction 5 = 135) -- ray at due Northwest
  : direction 5 - direction 3 = each_angle :=
by
  -- to be proved
  sorry

end smaller_angle_between_east_and_northwest_l393_39348


namespace smallest_number_divisible_by_conditions_l393_39310

theorem smallest_number_divisible_by_conditions (N : ℕ) (X : ℕ) (H1 : (N - 12) % 8 = 0) (H2 : (N - 12) % 12 = 0)
(H3 : (N - 12) % X = 0) (H4 : (N - 12) % 24 = 0) (H5 : (N - 12) / 24 = 276) : N = 6636 :=
by
  sorry

end smallest_number_divisible_by_conditions_l393_39310


namespace g_seven_l393_39331

def g (x : ℚ) : ℚ := (2 * x + 3) / (4 * x - 5)

theorem g_seven : g 7 = 17 / 23 := by
  sorry

end g_seven_l393_39331


namespace factors_of_12_factors_of_18_l393_39341

def is_factor (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

theorem factors_of_12 : 
  {k : ℕ | is_factor 12 k} = {1, 12, 2, 6, 3, 4} :=
by
  sorry

theorem factors_of_18 : 
  {k : ℕ | is_factor 18 k} = {1, 18, 2, 9, 3, 6} :=
by
  sorry

end factors_of_12_factors_of_18_l393_39341


namespace product_roots_l393_39391

noncomputable def root1 (x1 : ℝ) : Prop := x1 * Real.log x1 = 2006
noncomputable def root2 (x2 : ℝ) : Prop := x2 * Real.exp x2 = 2006

theorem product_roots (x1 x2 : ℝ) (h1 : root1 x1) (h2 : root2 x2) : x1 * x2 = 2006 := sorry

end product_roots_l393_39391


namespace candidates_appeared_in_each_state_equals_7900_l393_39393

theorem candidates_appeared_in_each_state_equals_7900 (x : ℝ) (h : 0.07 * x = 0.06 * x + 79) : x = 7900 :=
sorry

end candidates_appeared_in_each_state_equals_7900_l393_39393


namespace together_complete_days_l393_39364

-- Define the work rates of x and y
def work_rate_x := (1 : ℚ) / 30
def work_rate_y := (1 : ℚ) / 45

-- Define the combined work rate when x and y work together
def combined_work_rate := work_rate_x + work_rate_y

-- Define the number of days to complete the work together
def days_to_complete_work := 1 / combined_work_rate

-- The theorem we want to prove
theorem together_complete_days : days_to_complete_work = 18 := by
  sorry

end together_complete_days_l393_39364


namespace g_at_5_l393_39369

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 22 * x ^ 3 + 47 * x ^ 2 - 44 * x + 24

theorem g_at_5 : g 5 = 104 := by
  sorry

end g_at_5_l393_39369


namespace divisor_between_40_and_50_l393_39332

theorem divisor_between_40_and_50 (n : ℕ) (h1 : 40 ≤ n) (h2 : n ≤ 50) (h3 : n ∣ (2^36 - 1)) : n = 49 :=
sorry

end divisor_between_40_and_50_l393_39332


namespace n_power_of_two_if_2_pow_n_plus_one_odd_prime_l393_39314

-- Definition: a positive integer n is a power of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

-- Theorem: if 2^n +1 is an odd prime, then n must be a power of 2
theorem n_power_of_two_if_2_pow_n_plus_one_odd_prime (n : ℕ) (hp : Prime (2^n + 1)) (hn : Odd (2^n + 1)) : is_power_of_two n :=
by
  sorry

end n_power_of_two_if_2_pow_n_plus_one_odd_prime_l393_39314


namespace solve_for_y_l393_39340

theorem solve_for_y (y : ℝ) (h : (1 / 4) - (1 / 6) = 2 / y) : y = 24 :=
sorry

end solve_for_y_l393_39340


namespace squared_expression_is_matching_string_l393_39347

theorem squared_expression_is_matching_string (n : ℕ) (h : n > 0) :
  let a := (10^n - 1) / 9
  let term1 := 4 * a * (9 * a + 2)
  let term2 := 10 * a + 1
  let term3 := 6 * a
  let exp := term1 + term2 - term3
  Nat.sqrt exp = 6 * a + 1 := by
  sorry

end squared_expression_is_matching_string_l393_39347


namespace triangle_perimeter_l393_39356

-- Definitions for the conditions
def side_length1 : ℕ := 3
def side_length2 : ℕ := 6
def equation (x : ℤ) := x^2 - 6 * x + 8 = 0

-- Perimeter calculation given the sides form a triangle
theorem triangle_perimeter (x : ℤ) (h₁ : equation x) (h₂ : 3 + 6 > x) (h₃ : 3 + x > 6) (h₄ : 6 + x > 3) :
  3 + 6 + x = 13 :=
by sorry

end triangle_perimeter_l393_39356


namespace transmission_prob_correct_transmission_scheme_comparison_l393_39372

noncomputable def transmission_prob_single (α β : ℝ) : ℝ :=
  (1 - α) * (1 - β)^2

noncomputable def transmission_prob_triple_sequence (β : ℝ) : ℝ :=
  β * (1 - β)^2

noncomputable def transmission_prob_triple_decoding_one (β : ℝ) : ℝ :=
  β * (1 - β)^2 + (1 - β)^3

noncomputable def transmission_prob_triple_decoding_zero (α : ℝ) : ℝ :=
  3 * α * (1 - α)^2 + (1 - α)^3

noncomputable def transmission_prob_single_decoding_zero (α : ℝ) : ℝ :=
  1 - α

theorem transmission_prob_correct (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  transmission_prob_single α β = (1 - α) * (1 - β)^2 ∧
  transmission_prob_triple_sequence β = β * (1 - β)^2 ∧
  transmission_prob_triple_decoding_one β = β * (1 - β)^2 + (1 - β)^3 :=
sorry

theorem transmission_scheme_comparison (α : ℝ) (hα : 0 < α ∧ α < 0.5) :
  transmission_prob_triple_decoding_zero α > transmission_prob_single_decoding_zero α :=
sorry

end transmission_prob_correct_transmission_scheme_comparison_l393_39372


namespace width_to_length_ratio_l393_39363

variable (w : ℕ)

def length := 10
def perimeter := 36

theorem width_to_length_ratio
  (h_perimeter : 2 * w + 2 * length = perimeter) :
  w / length = 4 / 5 :=
by
  -- Skipping proof steps, putting sorry
  sorry

end width_to_length_ratio_l393_39363


namespace sum_max_min_values_l393_39358

noncomputable def y (x : ℝ) : ℝ := 2 * x^2 + 32 / x

theorem sum_max_min_values :
  y 1 = 34 ∧ y 2 = 24 ∧ y 4 = 40 → ((y 4 + y 2) = 64) :=
by
  sorry

end sum_max_min_values_l393_39358


namespace janet_acres_l393_39384

-- Defining the variables and conditions
variable (horse_production : ℕ) (acre_requirement : ℕ) (spread_rate : ℕ) (num_horses : ℕ) (days : ℕ)

-- Assigning the given values to the variables
def horseFertilizer := 5
def acreFertilizer := 400
def janetSpreadRate := 4
def janetHorses := 80
def fertilizingDays := 25

-- Main theorem stating the question and proving the answer
theorem janet_acres : 
  ∀ (horse_production : ℕ) (acre_requirement : ℕ) (spread_rate : ℕ) (num_horses : ℕ) (days : ℕ),
  horse_production = 5 → 
  acre_requirement = 400 →
  spread_rate = 4 →
  num_horses = 80 →
  days = 25 →
  (spread_rate * days = 100) := 
by
  intros
  -- Proof would be inserted here
  sorry

end janet_acres_l393_39384


namespace largest_n_for_divisibility_l393_39357

theorem largest_n_for_divisibility : 
  ∃ n : ℕ, (n + 12 ∣ n^3 + 150) ∧ (∀ m : ℕ, (m + 12 ∣ m^3 + 150) → m ≤ 246) :=
sorry

end largest_n_for_divisibility_l393_39357


namespace factors_of_m_multiples_of_200_l393_39382

theorem factors_of_m_multiples_of_200 (m : ℕ) (h : m = 2^12 * 3^10 * 5^9) : 
  (∃ k, 200 * k ≤ m ∧ ∃ a b c, k = 2^a * 3^b * 5^c ∧ 3 ≤ a ∧ a ≤ 12 ∧ 2 ≤ c ∧ c ≤ 9 ∧ 0 ≤ b ∧ b ≤ 10) := 
by sorry

end factors_of_m_multiples_of_200_l393_39382


namespace ramesh_share_correct_l393_39320

-- Define basic conditions
def suresh_investment := 24000
def ramesh_investment := 40000
def total_profit := 19000

-- Define Ramesh's share calculation
def ramesh_share : ℤ :=
  let ratio_ramesh := ramesh_investment / (suresh_investment + ramesh_investment)
  ratio_ramesh * total_profit

-- Proof statement
theorem ramesh_share_correct : ramesh_share = 11875 := by
  sorry

end ramesh_share_correct_l393_39320


namespace min_value_k_l393_39304

variables (x : ℕ → ℚ) (k n c : ℚ)

theorem min_value_k
  (k_gt_one : k > 1) -- condition that k > 1
  (n_gt_2018 : n > 2018) -- condition that n > 2018
  (n_odd : n % 2 = 1) -- condition that n is odd
  (non_zero_rational : ∀ i : ℕ, x i ≠ 0) -- non-zero rational numbers x₁, x₂, ..., xₙ
  (not_all_equal : ∃ i j : ℕ, x i ≠ x j) -- they are not all equal
  (relations : ∀ i : ℕ, x i + k / x (i + 1) = c) -- given relations
  : k = 4 :=
sorry

end min_value_k_l393_39304


namespace y_minus_x_is_7_l393_39387

theorem y_minus_x_is_7 (x y : ℕ) (hx : x ≠ y) (h1 : 3 + y = 10) (h2 : 0 + x + 1 = 1) (h3 : 3 + 7 = 10) :
  y - x = 7 :=
by
  sorry

end y_minus_x_is_7_l393_39387


namespace polygon_interior_angle_l393_39317

theorem polygon_interior_angle (n : ℕ) (hn : 3 * (180 - 180 * (n - 2) / n) + 180 = 180 * (n - 2) / n + 180) : n = 9 :=
by {
  sorry
}

end polygon_interior_angle_l393_39317


namespace combined_tax_rate_correct_l393_39326

noncomputable def combined_tax_rate (income_john income_ingrid tax_rate_john tax_rate_ingrid : ℝ) : ℝ :=
  let tax_john := tax_rate_john * income_john
  let tax_ingrid := tax_rate_ingrid * income_ingrid
  let total_tax := tax_john + tax_ingrid
  let combined_income := income_john + income_ingrid
  total_tax / combined_income * 100

theorem combined_tax_rate_correct :
  combined_tax_rate 56000 74000 0.30 0.40 = 35.69 := by
  sorry

end combined_tax_rate_correct_l393_39326


namespace valid_outfit_combinations_l393_39306

theorem valid_outfit_combinations :
  let shirts := 8
  let pants := 5
  let hats := 6
  let shared_colors := 5
  let extra_colors := 2
  let total_combinations := shirts * pants * hats
  let invalid_shared_combinations := shared_colors * pants
  let invalid_extra_combinations := extra_colors * pants
  let invalid_combinations := invalid_shared_combinations + invalid_extra_combinations
  total_combinations - invalid_combinations = 205 :=
by
  let shirts := 8
  let pants := 5
  let hats := 6
  let shared_colors := 5
  let extra_colors := 2
  let total_combinations := shirts * pants * hats
  let invalid_shared_combinations := shared_colors * pants
  let invalid_extra_combinations := extra_colors * pants
  let invalid_combinations := invalid_shared_combinations + invalid_extra_combinations
  have h : total_combinations - invalid_combinations = 205 := sorry
  exact h

end valid_outfit_combinations_l393_39306


namespace area_of_square_l393_39305

theorem area_of_square (ABCD MN : ℝ) (h1 : 4 * (ABCD / 4) = ABCD) (h2 : MN = 3) : ABCD = 64 :=
by
  sorry

end area_of_square_l393_39305


namespace local_minimum_of_reflected_function_l393_39319

noncomputable def f : ℝ → ℝ := sorry

theorem local_minimum_of_reflected_function (f : ℝ → ℝ) (x_0 : ℝ) (h1 : x_0 ≠ 0) (h2 : ∃ ε > 0, ∀ x, abs (x - x_0) < ε → f x ≤ f x_0) :
  ∃ δ > 0, ∀ x, abs (x - (-x_0)) < δ → -f (-x) ≥ -f (-x_0) :=
sorry

end local_minimum_of_reflected_function_l393_39319


namespace zoe_pictures_l393_39399

theorem zoe_pictures (P : ℕ) (h1 : P + 16 = 44) : P = 28 :=
by sorry

end zoe_pictures_l393_39399


namespace find_number_l393_39323

theorem find_number (k r n : ℤ) (hk : k = 38) (hr : r = 7) (h : n = 23 * k + r) : n = 881 := 
  by
  sorry

end find_number_l393_39323


namespace total_fruits_l393_39373

theorem total_fruits (Mike_fruits Matt_fruits Mark_fruits : ℕ)
  (Mike_receives : Mike_fruits = 3)
  (Matt_receives : Matt_fruits = 2 * Mike_fruits)
  (Mark_receives : Mark_fruits = Mike_fruits + Matt_fruits) :
  Mike_fruits + Matt_fruits + Mark_fruits = 18 := by
  sorry

end total_fruits_l393_39373


namespace inequality_abc_l393_39390

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2 * b + 3 * c) ^ 2 / (a ^ 2 + 2 * b ^ 2 + 3 * c ^ 2) ≤ 6 :=
sorry

end inequality_abc_l393_39390


namespace minimize_f_l393_39333

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l393_39333


namespace meaningful_expression_range_l393_39383

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by 
  sorry

end meaningful_expression_range_l393_39383


namespace exists_arithmetic_progression_with_11_numbers_exists_arithmetic_progression_with_10000_numbers_not_exists_infinite_arithmetic_progression_l393_39315

open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_arithmetic_progression_with_11_numbers :
  ∃ a d : ℕ, ∀ i j : ℕ, i < 11 → j < 11 → i < j → a + i * d < a + j * d ∧ 
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d) := by
  sorry

theorem exists_arithmetic_progression_with_10000_numbers :
  ∃ a d : ℕ, ∀ i j : ℕ, i < 10000 → j < 10000 → i < j → a + i * d < a + j * d ∧
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d) := by
  sorry

theorem not_exists_infinite_arithmetic_progression :
  ¬ (∃ a d : ℕ, ∀ i j : ℕ, i < j → a + i * d < a + j * d ∧
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d)) := by
  sorry

end exists_arithmetic_progression_with_11_numbers_exists_arithmetic_progression_with_10000_numbers_not_exists_infinite_arithmetic_progression_l393_39315


namespace problem_1_problem_2_l393_39359

-- Definition of sets A and B as in the problem's conditions
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | x > 2 ∨ x < -2}
def C (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

-- Prove that A ∩ B is as described
theorem problem_1 : A ∩ B = {x | 2 < x ∧ x ≤ 5} := by
  sorry

-- Prove that a ≥ 6 given the conditions in the problem
theorem problem_2 (a : ℝ) : (A ⊆ C a) → a ≥ 6 := by
  sorry

end problem_1_problem_2_l393_39359


namespace total_boxes_correct_l393_39398

noncomputable def friday_boxes : ℕ := 40

noncomputable def saturday_boxes : ℕ := 2 * friday_boxes - 10

noncomputable def sunday_boxes : ℕ := saturday_boxes / 2

noncomputable def monday_boxes : ℕ := 
  let extra_boxes := (25 * sunday_boxes + 99) / 100 -- (25/100) * sunday_boxes rounded to nearest integer
  sunday_boxes + extra_boxes

noncomputable def total_boxes : ℕ := 
  friday_boxes + saturday_boxes + sunday_boxes + monday_boxes

theorem total_boxes_correct : total_boxes = 189 := by
  sorry

end total_boxes_correct_l393_39398


namespace coordinates_B_l393_39366

theorem coordinates_B (A B : ℝ × ℝ) (distance : ℝ) (A_coords : A = (-1, 3)) 
  (AB_parallel_x : A.snd = B.snd) (AB_distance : abs (A.fst - B.fst) = distance) :
  (B = (-6, 3) ∨ B = (4, 3)) :=
by
  sorry

end coordinates_B_l393_39366


namespace area_of_quadrilateral_NLMK_l393_39380

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_quadrilateral_NLMK 
  (AB BC AC AK CN CL : ℝ)
  (h_AB : AB = 13)
  (h_BC : BC = 20)
  (h_AC : AC = 21)
  (h_AK : AK = 4)
  (h_CN : CN = 1)
  (h_CL : CL = 20 / 21) : 
  triangle_area AB BC AC - 
  (1 * CL / (BC * AC) * triangle_area AB BC AC) - 
  (9 * (BC - CN) / (AB * BC) * triangle_area AB BC AC) -
  (16 * 41 / (169 * 21) * triangle_area AB BC AC) = 
  493737 / 11830 := 
sorry

end area_of_quadrilateral_NLMK_l393_39380


namespace find_integer_k_l393_39375

theorem find_integer_k {k : ℤ} :
  (∀ x : ℝ, (k^2 + 1) * x^2 - (4 - k) * x + 1 = 0 →
    (∃ m n : ℝ, m ≠ n ∧ m * n = 1 / (k^2 + 1) ∧ m + n = (4 - k) / (k^2 + 1) ∧
      ((1 < m ∧ n < 1) ∨ (1 < n ∧ m < 1)))) →
  k = -1 ∨ k = 0 :=
by
  sorry

end find_integer_k_l393_39375


namespace no_sum_of_three_squares_l393_39343

theorem no_sum_of_three_squares (a k : ℕ) : 
  ¬ ∃ x y z : ℤ, 4^a * (8*k + 7) = x^2 + y^2 + z^2 :=
by
  sorry

end no_sum_of_three_squares_l393_39343


namespace quadratic_solution_l393_39397

theorem quadratic_solution (x : ℝ) : x^2 - 5 * x - 6 = 0 ↔ (x = 6 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l393_39397


namespace imaginary_part_z_l393_39354

open Complex

theorem imaginary_part_z : (im ((i - 1) / (i + 1))) = 1 :=
by
  -- The proof goes here, but it can be marked with sorry for now
  sorry

end imaginary_part_z_l393_39354


namespace nancy_kept_chips_correct_l393_39301

/-- Define the initial conditions -/
def total_chips : ℕ := 22
def chips_to_brother : ℕ := 7
def chips_to_sister : ℕ := 5

/-- Define the number of chips Nancy kept -/
def chips_kept : ℕ := total_chips - (chips_to_brother + chips_to_sister)

theorem nancy_kept_chips_correct : chips_kept = 10 := by
  /- This is a placeholder. The proof would go here. -/
  sorry

end nancy_kept_chips_correct_l393_39301


namespace translate_quadratic_l393_39342

-- Define the original quadratic function
def original_quadratic (x : ℝ) : ℝ := (x - 2)^2 - 4

-- Define the translation of the graph one unit to the left and two units up
def translated_quadratic (x : ℝ) : ℝ := (x - 1)^2 - 2

-- Statement to be proved
theorem translate_quadratic :
  ∀ x : ℝ, translated_quadratic x = original_quadratic (x-1) + 2 :=
by
  intro x
  unfold translated_quadratic original_quadratic
  sorry

end translate_quadratic_l393_39342


namespace boxes_count_l393_39311

theorem boxes_count (notebooks_per_box : ℕ) (total_notebooks : ℕ) (h1 : notebooks_per_box = 9) (h2 : total_notebooks = 27) : (total_notebooks / notebooks_per_box) = 3 :=
by
  sorry

end boxes_count_l393_39311


namespace arithmetic_sequence_a3_l393_39335

theorem arithmetic_sequence_a3 (a1 d : ℤ) (h : a1 + (a1 + d) + (a1 + 2 * d) + (a1 + 3 * d) + (a1 + 4 * d) = 20) : 
  a1 + 2 * d = 4 := by
  sorry

end arithmetic_sequence_a3_l393_39335


namespace positive_integer_iff_positive_real_l393_39365

theorem positive_integer_iff_positive_real (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℕ, n > 0 ∧ abs ((x - 2 * abs x) * abs x) / x = n) ↔ x > 0 :=
by
  sorry

end positive_integer_iff_positive_real_l393_39365


namespace quadratic_rewrite_l393_39300

theorem quadratic_rewrite  (a b c x : ℤ) (h : 25 * x^2 + 30 * x - 35 = 0) (hp : 25 * x^2 + 30 * x + 9 = (5 * x + 3) ^ 2)
(hc : c = 44) : a = 5 → b = 3 → a + b + c = 52 := 
by
  intro ha hb
  sorry

end quadratic_rewrite_l393_39300


namespace men_apples_l393_39385

theorem men_apples (M W : ℕ) (h1 : M = W - 20) (h2 : 2 * M + 3 * W = 210) : M = 30 :=
by
  -- skipping the proof
  sorry

end men_apples_l393_39385


namespace two_digit_sum_of_original_and_reverse_l393_39312

theorem two_digit_sum_of_original_and_reverse
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9) -- a is a digit
  (h2 : 0 ≤ b ∧ b ≤ 9) -- b is a digit
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end two_digit_sum_of_original_and_reverse_l393_39312


namespace smallest_n_Sn_gt_2023_l393_39316

open Nat

theorem smallest_n_Sn_gt_2023 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 4) →
  (∀ n : ℕ, n > 0 → a n + a (n + 1) = 4 * n + 2) →
  (∀ m : ℕ, S m = if m % 2 = 0 then m ^ 2 + m else m ^ 2 + m + 2) →
  ∃ n : ℕ, S n > 2023 ∧ ∀ k : ℕ, k < n → S k ≤ 2023 :=
sorry

end smallest_n_Sn_gt_2023_l393_39316


namespace range_of_m_l393_39339

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt (1 + x) + Real.sqrt (1 - x)) * (2 * Real.sqrt (1 - x^2) - 1)

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x = m) ↔ -Real.sqrt 2 ≤ m ∧ m ≤ Real.sqrt 2 :=
sorry

end range_of_m_l393_39339


namespace possible_lost_rectangle_area_l393_39379

theorem possible_lost_rectangle_area (areas : Fin 10 → ℕ) (total_area : ℕ) (h_total : total_area = 65) :
  (∃ (i : Fin 10), (64 = total_area - areas i) ∨ (49 = total_area - areas i)) ↔
  (∃ (i : Fin 10), (areas i = 1) ∨ (areas i = 16)) :=
by
  sorry

end possible_lost_rectangle_area_l393_39379


namespace problem1_problem2_l393_39381

theorem problem1 : (Real.sqrt 24 - Real.sqrt 18) - Real.sqrt 6 = Real.sqrt 6 - 3 * Real.sqrt 2 := by
  sorry

theorem problem2 : 2 * Real.sqrt 12 * Real.sqrt (1 / 8) + 5 * Real.sqrt 2 = Real.sqrt 6 + 5 * Real.sqrt 2 := by
  sorry

end problem1_problem2_l393_39381


namespace find_a10_l393_39355

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Given conditions
variables (a : ℕ → ℤ) (h_arith_seq : arithmetic_sequence a)
variables (h_a2 : a 2 = 2) (h_a6 : a 6 = 10)

-- Goal to prove
theorem find_a10 : a 10 = 18 :=
by
  sorry

end find_a10_l393_39355


namespace problem_statement_l393_39361

noncomputable def polynomial_expansion (x : ℚ) : ℚ := (1 - 2 * x) ^ 8

theorem problem_statement :
  (8 * (1 - 2 * 1) ^ 7 * (-2)) = (a_1 : ℚ) + 2 * (a_2 : ℚ) + 3 * (a_3 : ℚ) + 4 * (a_4 : ℚ) +
  5 * (a_5 : ℚ) + 6 * (a_6 : ℚ) + 7 * (a_7 : ℚ) + 8 * (a_8 : ℚ) := by 
  sorry

end problem_statement_l393_39361


namespace area_of_region_l393_39309

noncomputable def area : ℝ :=
  ∫ x in Set.Icc (-2 : ℝ) 0, (2 - (x + 1)^2 / 4) +
  ∫ x in Set.Icc (0 : ℝ) 2, (2 - x - (x + 1)^2 / 4)

theorem area_of_region : area = 5 / 3 := 
sorry

end area_of_region_l393_39309


namespace probability_of_same_color_l393_39344

noncomputable def prob_same_color (P_A P_B : ℚ) : ℚ :=
  P_A + P_B

theorem probability_of_same_color :
  let P_A := (1 : ℚ) / 7
  let P_B := (12 : ℚ) / 35
  prob_same_color P_A P_B = 17 / 35 := 
by 
  -- Definition of P_A and P_B
  let P_A := (1 : ℚ) / 7
  let P_B := (12 : ℚ) / 35
  -- Use the definition of prob_same_color
  let result := prob_same_color P_A P_B
  -- Now we are supposed to prove that result = 17 / 35
  have : result = (5 : ℚ) / 35 + (12 : ℚ) / 35 := by
    -- Simplifying the fractions individually can be done at this intermediate step
    sorry
  sorry

end probability_of_same_color_l393_39344
