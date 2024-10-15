import Mathlib

namespace NUMINAMATH_GPT_Priya_driving_speed_l531_53178

/-- Priya's driving speed calculation -/
theorem Priya_driving_speed
  (time_XZ : ℝ) (rate_back : ℝ) (time_ZY : ℝ)
  (midway_condition : time_XZ = 5)
  (speed_back_condition : rate_back = 60)
  (time_back_condition : time_ZY = 2.0833333333333335) :
  ∃ speed_XZ : ℝ, speed_XZ = 50 :=
by
  have distance_ZY : ℝ := rate_back * time_ZY
  have distance_XZ : ℝ := 2 * distance_ZY
  have speed_XZ : ℝ := distance_XZ / time_XZ
  existsi speed_XZ
  sorry

end NUMINAMATH_GPT_Priya_driving_speed_l531_53178


namespace NUMINAMATH_GPT_find_dividend_l531_53187

noncomputable def dividend (divisor quotient remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem find_dividend :
  ∀ (divisor quotient remainder : ℕ), 
  divisor = 16 → 
  quotient = 8 → 
  remainder = 4 → 
  dividend divisor quotient remainder = 132 :=
by
  intros divisor quotient remainder hdiv hquo hrem
  sorry

end NUMINAMATH_GPT_find_dividend_l531_53187


namespace NUMINAMATH_GPT_gas_cost_per_gallon_is_4_l531_53133

noncomputable def cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (total_miles / miles_per_gallon)

theorem gas_cost_per_gallon_is_4 :
  cost_per_gallon 32 432 54 = 4 := by
  sorry

end NUMINAMATH_GPT_gas_cost_per_gallon_is_4_l531_53133


namespace NUMINAMATH_GPT_nuts_needed_for_cookies_l531_53151

-- Given conditions
def total_cookies : Nat := 120
def fraction_nuts : Rat := 1 / 3
def fraction_chocolate : Rat := 0.25
def nuts_per_cookie : Nat := 3

-- Translated conditions as helpful functions
def cookies_with_nuts : Nat := Nat.floor (fraction_nuts * total_cookies)
def cookies_with_chocolate : Nat := Nat.floor (fraction_chocolate * total_cookies)
def cookies_with_both : Nat := total_cookies - cookies_with_nuts - cookies_with_chocolate
def total_cookies_with_nuts : Nat := cookies_with_nuts + cookies_with_both
def total_nuts_needed : Nat := total_cookies_with_nuts * nuts_per_cookie

-- Proof problem: proving that total nuts needed is 270
theorem nuts_needed_for_cookies : total_nuts_needed = 270 :=
by
  sorry

end NUMINAMATH_GPT_nuts_needed_for_cookies_l531_53151


namespace NUMINAMATH_GPT_find_x_y_l531_53120

theorem find_x_y (x y : ℝ) (h1 : x + Real.cos y = 2023) (h2 : x + 2023 * Real.sin y = 2022) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2022 :=
sorry

end NUMINAMATH_GPT_find_x_y_l531_53120


namespace NUMINAMATH_GPT_complement_union_l531_53163

open Set Real

noncomputable def S : Set ℝ := {x | x > -2}
noncomputable def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

theorem complement_union (x : ℝ): x ∈ (univ \ S) ∪ T ↔ x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l531_53163


namespace NUMINAMATH_GPT_inequality_solution_l531_53185

theorem inequality_solution (x : ℝ) : (4 + 2 * x > -6) → (x > -5) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l531_53185


namespace NUMINAMATH_GPT_division_of_decimals_l531_53136

theorem division_of_decimals : (0.05 / 0.002) = 25 :=
by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_division_of_decimals_l531_53136


namespace NUMINAMATH_GPT_product_of_distinct_solutions_l531_53168

theorem product_of_distinct_solutions (x y : ℝ) (h₁ : x ≠ y) (h₂ : x ≠ 0) (h₃ : y ≠ 0) (h₄ : x - 2 / x = y - 2 / y) :
  x * y = -2 :=
sorry

end NUMINAMATH_GPT_product_of_distinct_solutions_l531_53168


namespace NUMINAMATH_GPT_calculate_expression_l531_53150

theorem calculate_expression :
  -2^3 * (-3)^2 / (9 / 8) - abs (1 / 2 - 3 / 2) = -65 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l531_53150


namespace NUMINAMATH_GPT_find_a_b_find_m_l531_53189

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * x^2

theorem find_a_b (a b : ℝ) (h₁ : f 1 a b = 4)
  (h₂ : 3 * a + 2 * b = 9) : a = 1 ∧ b = 3 :=
by
  sorry

theorem find_m (m : ℝ) (h : ∀ x, (m ≤ x ∧ x ≤ m + 1) → (3 * x^2 + 6 * x > 0)) :
  m ≥ 0 ∨ m ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_find_m_l531_53189


namespace NUMINAMATH_GPT_product_of_consecutive_integers_is_square_l531_53196

theorem product_of_consecutive_integers_is_square (x : ℤ) : 
  x * (x + 1) * (x + 2) * (x + 3) + 1 = (x^2 + 3 * x + 1) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_is_square_l531_53196


namespace NUMINAMATH_GPT_range_of_4a_minus_2b_l531_53140

theorem range_of_4a_minus_2b (a b : ℝ) (h1 : 0 ≤ a - b) (h2 : a - b ≤ 1) (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) :
  2 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 7 := 
sorry

end NUMINAMATH_GPT_range_of_4a_minus_2b_l531_53140


namespace NUMINAMATH_GPT_grandma_vasya_cheapest_option_l531_53156

/-- Constants and definitions for the cost calculations --/
def train_ticket_cost : ℕ := 200
def collected_berries_kg : ℕ := 5
def market_berries_cost_per_kg : ℕ := 150
def sugar_cost_per_kg : ℕ := 54
def jam_made_per_kg_combination : ℕ := 15 / 10  -- representing 1.5 kg (as ratio 15/10)
def ready_made_jam_cost_per_kg : ℕ := 220

/-- Compute the cost per kg of jam for different methods --/
def cost_per_kg_jam_option1 : ℕ := (train_ticket_cost / collected_berries_kg + sugar_cost_per_kg)
def cost_per_kg_jam_option2 : ℕ := market_berries_cost_per_kg + sugar_cost_per_kg
def cost_per_kg_jam_option3 : ℕ := ready_made_jam_cost_per_kg

/-- Numbers converted to per 1.5 kg --/
def cost_for_1_5_kg (cost_per_kg: ℕ) : ℕ := cost_per_kg * (15 / 10)

/-- Theorem stating option 1 is the cheapest --/
theorem grandma_vasya_cheapest_option :
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option2 ∧
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option3 :=
by sorry

end NUMINAMATH_GPT_grandma_vasya_cheapest_option_l531_53156


namespace NUMINAMATH_GPT_z_amount_per_rupee_l531_53113

theorem z_amount_per_rupee (x y z : ℝ) 
  (h1 : ∀ rupees_x, y = 0.45 * rupees_x)
  (h2 : y = 36)
  (h3 : x + y + z = 156)
  (h4 : ∀ rupees_x, x = rupees_x) :
  ∃ a : ℝ, z = a * x ∧ a = 0.5 := 
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_z_amount_per_rupee_l531_53113


namespace NUMINAMATH_GPT_number_of_friends_l531_53108

-- Define the conditions
def kendra_packs : ℕ := 7
def tony_packs : ℕ := 5
def pens_per_kendra_pack : ℕ := 4
def pens_per_tony_pack : ℕ := 6
def pens_kendra_keep : ℕ := 3
def pens_tony_keep : ℕ := 3

-- Define the theorem to be proved
theorem number_of_friends 
  (packs_k : ℕ := kendra_packs)
  (packs_t : ℕ := tony_packs)
  (pens_per_pack_k : ℕ := pens_per_kendra_pack)
  (pens_per_pack_t : ℕ := pens_per_tony_pack)
  (kept_k : ℕ := pens_kendra_keep)
  (kept_t : ℕ := pens_tony_keep) :
  packs_k * pens_per_pack_k + packs_t * pens_per_pack_t - (kept_k + kept_t) = 52 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l531_53108


namespace NUMINAMATH_GPT_total_area_covered_is_60_l531_53102

-- Declare the dimensions of the strips
def length_strip : ℕ := 12
def width_strip : ℕ := 2
def num_strips : ℕ := 3

-- Define the total area covered without overlaps
def total_area_no_overlap := num_strips * (length_strip * width_strip)

-- Define the area of overlap for each pair of strips
def overlap_area_per_pair := width_strip * width_strip

-- Define the total overlap area given 3 pairs
def total_overlap_area := 3 * overlap_area_per_pair

-- Define the actual total covered area
def total_covered_area := total_area_no_overlap - total_overlap_area

-- Prove that the total covered area is 60 square units
theorem total_area_covered_is_60 : total_covered_area = 60 := by 
  sorry

end NUMINAMATH_GPT_total_area_covered_is_60_l531_53102


namespace NUMINAMATH_GPT_equivalent_expression_l531_53153

theorem equivalent_expression : 8^8 * 4^4 / 2^28 = 16 := by
  -- Here, we're stating the equivalency directly
  sorry

end NUMINAMATH_GPT_equivalent_expression_l531_53153


namespace NUMINAMATH_GPT_find_y_l531_53171

theorem find_y (x y : ℝ) (hA : {2, Real.log x} = {a | a = 2 ∨ a = Real.log x})
                (hB : {x, y} = {a | a = x ∨ a = y})
                (hInt : {a | a = 2 ∨ a = Real.log x} ∩ {a | a = x ∨ a = y} = {0}) :
  y = 0 :=
  sorry

end NUMINAMATH_GPT_find_y_l531_53171


namespace NUMINAMATH_GPT_subset_condition_l531_53109

theorem subset_condition (a : ℝ) :
  (∀ x : ℝ, |2 * x - 1| < 1 → x^2 - 2 * a * x + a^2 - 1 > 0) →
  (a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_subset_condition_l531_53109


namespace NUMINAMATH_GPT_expand_expression_l531_53148

variable (y : ℤ)

theorem expand_expression : 12 * (3 * y - 4) = 36 * y - 48 := 
by
  sorry

end NUMINAMATH_GPT_expand_expression_l531_53148


namespace NUMINAMATH_GPT_sin_double_angle_value_l531_53145

open Real

theorem sin_double_angle_value (x : ℝ) (h : sin (x + π / 4) = - 5 / 13) : sin (2 * x) = - 119 / 169 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_value_l531_53145


namespace NUMINAMATH_GPT_prove_M_squared_l531_53174

noncomputable def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -1, 2], ![ (5/2:ℝ), x]]

def eigenvalue_condition (x : ℝ) : Prop :=
  let A := M x
  ∃ v : ℝ, (A - (-2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)).det = 0

theorem prove_M_squared (x : ℝ) (h : eigenvalue_condition x) :
  (M x * M x) = ![![ 6, -9], ![ - (45/4:ℝ), 69/4]] :=
sorry

end NUMINAMATH_GPT_prove_M_squared_l531_53174


namespace NUMINAMATH_GPT_range_of_a_l531_53198

-- Define the conditions
def line1 (a x y : ℝ) : Prop := a * x + y - 4 = 0
def line2 (x y : ℝ) : Prop := x - y - 2 = 0
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- The main theorem to state
theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, line1 a x y ∧ line2 x y ∧ first_quadrant x y) ↔ -1 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l531_53198


namespace NUMINAMATH_GPT_find_k_l531_53193

theorem find_k (k : ℝ) :
    (1 - 7) * (k - 3) = (3 - k) * (7 - 1) → k = 6.5 :=
by
sorry

end NUMINAMATH_GPT_find_k_l531_53193


namespace NUMINAMATH_GPT_dave_winfield_home_runs_l531_53180

theorem dave_winfield_home_runs (W : ℕ) (h : 755 = 2 * W - 175) : W = 465 :=
by
  sorry

end NUMINAMATH_GPT_dave_winfield_home_runs_l531_53180


namespace NUMINAMATH_GPT_line_parabola_intersection_one_point_l531_53161

theorem line_parabola_intersection_one_point (k : ℝ) :
  (∃ y : ℝ, (-3 * y^2 - 4 * y + 7 = k) ∧ ∀ y1 y2 : ℝ, ( 3 * y1^2 + 4 * y1 + (k - 7) = 0 → 3 * y2^2 + 4 * y2 + (k - 7) = 0 → y1 = y2)) ↔ (k = 25 / 3) :=
by
  sorry

end NUMINAMATH_GPT_line_parabola_intersection_one_point_l531_53161


namespace NUMINAMATH_GPT_angle_P_measure_l531_53143

theorem angle_P_measure (P Q : ℝ) (h1 : P + Q = 180) (h2 : P = 5 * Q) : P = 150 := by
  sorry

end NUMINAMATH_GPT_angle_P_measure_l531_53143


namespace NUMINAMATH_GPT_min_distance_to_line_l531_53199

theorem min_distance_to_line (m n : ℝ) (h : 4 * m + 3 * n = 10)
  : m^2 + n^2 ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_distance_to_line_l531_53199


namespace NUMINAMATH_GPT_mean_value_theorem_for_integrals_l531_53179

variable {a b : ℝ} (f : ℝ → ℝ)

theorem mean_value_theorem_for_integrals (h_cont : ContinuousOn f (Set.Icc a b)) :
  ∃ ξ ∈ Set.Icc a b, ∫ x in a..b, f x = f ξ * (b - a) :=
sorry

end NUMINAMATH_GPT_mean_value_theorem_for_integrals_l531_53179


namespace NUMINAMATH_GPT_range_of_g_l531_53107

noncomputable def g (x : ℝ) : ℝ := (Real.cos x)^4 + (Real.sin x)^2

theorem range_of_g : Set.Icc (3 / 4) 1 = Set.range g :=
by
  sorry

end NUMINAMATH_GPT_range_of_g_l531_53107


namespace NUMINAMATH_GPT_number_of_friends_l531_53106

-- Definitions based on the given problem conditions
def total_candy := 420
def candy_per_friend := 12

-- Proof statement in Lean 4
theorem number_of_friends : total_candy / candy_per_friend = 35 := by
  sorry

end NUMINAMATH_GPT_number_of_friends_l531_53106


namespace NUMINAMATH_GPT_largest_modulus_z_l531_53173

open Complex

noncomputable def z_largest_value (a b c z : ℂ) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem largest_modulus_z (a b c z : ℂ) (r : ℝ) (hr_pos : 0 < r)
  (hmod_a : Complex.abs a = r) (hmod_b : Complex.abs b = r) (hmod_c : Complex.abs c = r)
  (heqn : a * z ^ 2 + b * z + c = 0) :
  Complex.abs z ≤ z_largest_value a b c z :=
sorry

end NUMINAMATH_GPT_largest_modulus_z_l531_53173


namespace NUMINAMATH_GPT_hyperbola_and_line_properties_l531_53155

open Real

def hyperbola (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def asymptote1 (x y : ℝ) : Prop := y = sqrt 3 * x
def asymptote2 (x y : ℝ) : Prop := y = -sqrt 3 * x
def line (x y t : ℝ) : Prop := y = x + t

theorem hyperbola_and_line_properties :
  ∃ a b t : ℝ,
  a > 0 ∧ b > 0 ∧ a = 1 ∧ b^2 = 3 ∧
  (∀ x y, hyperbola x y a b ↔ (x^2 - y^2 / 3 = 1)) ∧
  (∀ x y, asymptote1 x y ↔ y = sqrt 3 * x) ∧
  (∀ x y, asymptote2 x y ↔ y = -sqrt 3 * x) ∧
  (∀ x y, (line x y t ↔ (y = x + sqrt 3) ∨ (y = x - sqrt 3))) := sorry

end NUMINAMATH_GPT_hyperbola_and_line_properties_l531_53155


namespace NUMINAMATH_GPT_chicken_nuggets_order_l531_53197

theorem chicken_nuggets_order (cost_per_box : ℕ) (nuggets_per_box : ℕ) (total_amount_paid : ℕ) 
  (h1 : cost_per_box = 4) (h2 : nuggets_per_box = 20) (h3 : total_amount_paid = 20) : 
  total_amount_paid / cost_per_box * nuggets_per_box = 100 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_chicken_nuggets_order_l531_53197


namespace NUMINAMATH_GPT_gcd_of_1230_and_990_l531_53184

theorem gcd_of_1230_and_990 : Nat.gcd 1230 990 = 30 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_1230_and_990_l531_53184


namespace NUMINAMATH_GPT_range_of_a_l531_53137

-- Definitions of conditions
def is_odd_function {A : Type} [AddGroup A] (f : A → A) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing {A : Type} [LinearOrderedAddCommGroup A] (f : A → A) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Main statement
theorem range_of_a 
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_monotone_dec : is_monotonically_decreasing f)
  (h_domain : ∀ x, -7 < x ∧ x < 7 → -7 < f x ∧ f x < 7)
  (h_cond : ∀ a, f (1 - a) + f (2 * a - 5) < 0): 
  ∀ a, 4 < a → a < 6 :=
sorry

end NUMINAMATH_GPT_range_of_a_l531_53137


namespace NUMINAMATH_GPT_avg_velocity_2_to_2_1_l531_53183

def motion_eq (t : ℝ) : ℝ := 3 + t^2

theorem avg_velocity_2_to_2_1 : 
  (motion_eq 2.1 - motion_eq 2) / (2.1 - 2) = 4.1 :=
by
  sorry

end NUMINAMATH_GPT_avg_velocity_2_to_2_1_l531_53183


namespace NUMINAMATH_GPT_two_colonies_reach_limit_same_time_l531_53158

theorem two_colonies_reach_limit_same_time
  (doubles_in_size : ∀ (n : ℕ), n = n * 2)
  (reaches_limit_in_25_days : ∃ N : ℕ, ∀ t : ℕ, t = 25 → N = N * 2^t) :
  ∀ t : ℕ, t = 25 := sorry

end NUMINAMATH_GPT_two_colonies_reach_limit_same_time_l531_53158


namespace NUMINAMATH_GPT_square_difference_division_l531_53118

theorem square_difference_division (a b : ℕ) (h₁ : a = 121) (h₂ : b = 112) :
  (a^2 - b^2) / 9 = 233 :=
by
  sorry

end NUMINAMATH_GPT_square_difference_division_l531_53118


namespace NUMINAMATH_GPT_simplify_and_evaluate_l531_53177

theorem simplify_and_evaluate (x y : ℤ) (h1 : x = -1) (h2 : y = -2) :
  ((x + y) ^ 2 - (3 * x - y) * (3 * x + y) - 2 * y ^ 2) / (-2 * x) = -2 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l531_53177


namespace NUMINAMATH_GPT_friendly_snakes_not_blue_l531_53101

variable (Snakes : Type)
variable (sally_snakes : Finset Snakes)
variable (blue : Snakes → Prop)
variable (friendly : Snakes → Prop)
variable (can_swim : Snakes → Prop)
variable (can_climb : Snakes → Prop)

variable [DecidablePred blue] [DecidablePred friendly] [DecidablePred can_swim] [DecidablePred can_climb]

-- The number of snakes in Sally's collection
axiom h_snakes_count : sally_snakes.card = 20
-- There are 7 blue snakes
axiom h_blue : (sally_snakes.filter blue).card = 7
-- There are 10 friendly snakes
axiom h_friendly : (sally_snakes.filter friendly).card = 10
-- All friendly snakes can swim
axiom h1 : ∀ s ∈ sally_snakes, friendly s → can_swim s
-- No blue snakes can climb
axiom h2 : ∀ s ∈ sally_snakes, blue s → ¬ can_climb s
-- Snakes that can't climb also can't swim
axiom h3 : ∀ s ∈ sally_snakes, ¬ can_climb s → ¬ can_swim s

theorem friendly_snakes_not_blue :
  ∀ s ∈ sally_snakes, friendly s → ¬ blue s :=
by
  sorry

end NUMINAMATH_GPT_friendly_snakes_not_blue_l531_53101


namespace NUMINAMATH_GPT_sin_lt_alpha_lt_tan_l531_53162

open Real

theorem sin_lt_alpha_lt_tan {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 2) : sin α < α ∧ α < tan α := by
  sorry

end NUMINAMATH_GPT_sin_lt_alpha_lt_tan_l531_53162


namespace NUMINAMATH_GPT_g_of_neg3_l531_53110

def g (x : ℝ) : ℝ := x^2 + 2 * x

theorem g_of_neg3 : g (-3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_g_of_neg3_l531_53110


namespace NUMINAMATH_GPT_fraction_is_determined_l531_53194

theorem fraction_is_determined (y x : ℕ) (h1 : y * 3 = x - 1) (h2 : (y + 4) * 2 = x) : 
  y = 7 ∧ x = 22 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_determined_l531_53194


namespace NUMINAMATH_GPT_has_local_maximum_l531_53134

noncomputable def func (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - 4 * x + 4

theorem has_local_maximum :
  ∃ x, x = -2 ∧ func x = 28 / 3 :=
by
  sorry

end NUMINAMATH_GPT_has_local_maximum_l531_53134


namespace NUMINAMATH_GPT_average_salary_rest_of_workers_l531_53167

theorem average_salary_rest_of_workers
  (avg_salary_all : ℝ)
  (num_all_workers : ℕ)
  (avg_salary_techs : ℝ)
  (num_techs : ℕ)
  (avg_salary_rest : ℝ)
  (num_rest : ℕ) :
  avg_salary_all = 8000 →
  num_all_workers = 21 →
  avg_salary_techs = 12000 →
  num_techs = 7 →
  num_rest = num_all_workers - num_techs →
  avg_salary_rest = (avg_salary_all * num_all_workers - avg_salary_techs * num_techs) / num_rest →
  avg_salary_rest = 6000 :=
by
  intros h_avg_all h_num_all h_avg_techs h_num_techs h_num_rest h_avg_rest
  sorry

end NUMINAMATH_GPT_average_salary_rest_of_workers_l531_53167


namespace NUMINAMATH_GPT_three_digit_permuted_mean_l531_53126

theorem three_digit_permuted_mean (N : ℕ) :
  (∃ x y z : ℕ, N = 100 * x + 10 * y + z ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
    (N = 111 ∨ N = 222 ∨ N = 333 ∨ N = 444 ∨ N = 555 ∨ N = 666 ∨ N = 777 ∨ N = 888 ∨ N = 999 ∨
     N = 407 ∨ N = 518 ∨ N = 629 ∨ N = 370 ∨ N = 481 ∨ N = 592)) ↔
    (∃ x y z : ℕ, N = 100 * x + 10 * y + z ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧ 7 * x = 3 * y + 4 * z) := by
sorry

end NUMINAMATH_GPT_three_digit_permuted_mean_l531_53126


namespace NUMINAMATH_GPT_volume_of_solid_bounded_by_planes_l531_53129

theorem volume_of_solid_bounded_by_planes (a : ℝ) : 
  ∃ v, v = (a ^ 3) / 6 :=
by 
  sorry

end NUMINAMATH_GPT_volume_of_solid_bounded_by_planes_l531_53129


namespace NUMINAMATH_GPT_no_integers_with_cube_sum_l531_53182

theorem no_integers_with_cube_sum (a b : ℤ) (h1 : a^3 + b^3 = 4099) (h2 : Prime 4099) : false :=
sorry

end NUMINAMATH_GPT_no_integers_with_cube_sum_l531_53182


namespace NUMINAMATH_GPT_area_of_path_is_675_l531_53159

def rectangular_field_length : ℝ := 75
def rectangular_field_width : ℝ := 55
def path_width : ℝ := 2.5

def area_of_path : ℝ :=
  let new_length := rectangular_field_length + 2 * path_width
  let new_width := rectangular_field_width + 2 * path_width
  let area_with_path := new_length * new_width
  let area_of_grass_field := rectangular_field_length * rectangular_field_width
  area_with_path - area_of_grass_field

theorem area_of_path_is_675 : area_of_path = 675 := by
  sorry

end NUMINAMATH_GPT_area_of_path_is_675_l531_53159


namespace NUMINAMATH_GPT_train_length_l531_53141

theorem train_length (L : ℝ) (h1 : L + 110 / 15 = (L + 250) / 20) : L = 310 := 
sorry

end NUMINAMATH_GPT_train_length_l531_53141


namespace NUMINAMATH_GPT_functional_equation_zero_l531_53122

open Function

theorem functional_equation_zero (f : ℕ+ → ℝ) 
  (h : ∀ (m n : ℕ+), n ≥ m → f (n + m) + f (n - m) = f (3 * n)) :
  ∀ n : ℕ+, f n = 0 := sorry

end NUMINAMATH_GPT_functional_equation_zero_l531_53122


namespace NUMINAMATH_GPT_age_difference_l531_53111

variable (S R : ℝ)

theorem age_difference (h1 : S = 38.5) (h2 : S / R = 11 / 9) : S - R = 7 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l531_53111


namespace NUMINAMATH_GPT_proof_equiv_l531_53142

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = 2 ^ Real.sqrt (3 + 2 * x - x ^ 2) }
noncomputable def N : Set ℝ := { x | ∃ y : ℝ, y = Real.log (x - 2) }
def I : Set ℝ := Set.univ
def complement_N : Set ℝ := I \ N

theorem proof_equiv : M ∩ complement_N = { y | 1 ≤ y ∧ y ≤ 2 } :=
sorry

end NUMINAMATH_GPT_proof_equiv_l531_53142


namespace NUMINAMATH_GPT_base_length_of_isosceles_triangle_l531_53195

theorem base_length_of_isosceles_triangle (a b : ℕ) (h1 : a = 8) (h2 : b + 2 * a = 26) : b = 10 :=
by
  have h3 : 2 * 8 = 16 := by norm_num
  rw [h1] at h2
  rw [h3] at h2
  linarith

end NUMINAMATH_GPT_base_length_of_isosceles_triangle_l531_53195


namespace NUMINAMATH_GPT_minimum_abs_phi_l531_53128

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem minimum_abs_phi 
  (ω φ b : ℝ)
  (hω : ω > 0)
  (hb : 0 < b ∧ b < 2)
  (h_intersections : f ω φ (π / 6) = b ∧ f ω φ (5 * π / 6) = b ∧ f ω φ (7 * π / 6) = b)
  (h_minimum : f ω φ (3 * π / 2) = -2) : 
  |φ| = π / 2 :=
sorry

end NUMINAMATH_GPT_minimum_abs_phi_l531_53128


namespace NUMINAMATH_GPT_corey_lowest_score_l531_53166

theorem corey_lowest_score
  (e1 e2 e3 e4 : ℕ)
  (h1 : e1 = 84)
  (h2 : e2 = 67)
  (max_score : ∀ (e : ℕ), e ≤ 100)
  (avg_at_least_75 : (e1 + e2 + e3 + e4) / 4 ≥ 75) :
  e3 ≥ 49 ∨ e4 ≥ 49 :=
by
  sorry

end NUMINAMATH_GPT_corey_lowest_score_l531_53166


namespace NUMINAMATH_GPT_find_number_l531_53175

theorem find_number (x : ℤ) (h : 3 * (x + 8) = 36) : x = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l531_53175


namespace NUMINAMATH_GPT_a_finishes_work_in_four_days_l531_53127

theorem a_finishes_work_in_four_days (x : ℝ) 
  (B_work_rate : ℝ) 
  (work_done_together : ℝ) 
  (work_done_by_B_alone : ℝ) : 
  B_work_rate = 1 / 16 → 
  work_done_together = 2 * (1 / x + 1 / 16) → 
  work_done_by_B_alone = 6 * (1 / 16) → 
  work_done_together + work_done_by_B_alone = 1 → 
  x = 4 :=
by
  intros hB hTogether hBAlone hTotal
  sorry

end NUMINAMATH_GPT_a_finishes_work_in_four_days_l531_53127


namespace NUMINAMATH_GPT_polynomial_expansion_a5_l531_53112

theorem polynomial_expansion_a5 :
  (x - 1) ^ 8 = (1 : ℤ) + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 →
  a₅ = -56 :=
by
  intro h
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_polynomial_expansion_a5_l531_53112


namespace NUMINAMATH_GPT_gauss_family_mean_age_l531_53157

theorem gauss_family_mean_age :
  let ages := [8, 8, 8, 8, 16, 17]
  let num_children := 6
  let sum_ages := 65
  (sum_ages : ℚ) / (num_children : ℚ) = 65 / 6 :=
by
  sorry

end NUMINAMATH_GPT_gauss_family_mean_age_l531_53157


namespace NUMINAMATH_GPT_leak_time_l531_53190

theorem leak_time (A L : ℝ) (PipeA_filling_rate : A = 1 / 6) (Combined_rate : A - L = 1 / 10) : 
  1 / L = 15 :=
by
  sorry

end NUMINAMATH_GPT_leak_time_l531_53190


namespace NUMINAMATH_GPT_smallest_x_value_l531_53170

theorem smallest_x_value (x : ℤ) (h : 3 * x^2 - 4 < 20) : x = -2 :=
sorry

end NUMINAMATH_GPT_smallest_x_value_l531_53170


namespace NUMINAMATH_GPT_range_of_m_l531_53188

-- Definitions based on conditions
def inequality_holds (m : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (m * Real.exp x / x ≥ 6 - 4 * x)

-- The statement to be proved
theorem range_of_m (m : ℝ) : inequality_holds m → m ≥ 2 * Real.exp (-(1 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l531_53188


namespace NUMINAMATH_GPT_parkway_girls_not_playing_soccer_l531_53172

theorem parkway_girls_not_playing_soccer (total_students boys soccer_students : ℕ) 
    (percent_boys_playing_soccer : ℕ) 
    (h1 : total_students = 420)
    (h2 : boys = 312)
    (h3 : soccer_students = 250)
    (h4 : percent_boys_playing_soccer = 86) :
   (total_students - boys - (soccer_students - soccer_students * percent_boys_playing_soccer / 100)) = 73 :=
by sorry

end NUMINAMATH_GPT_parkway_girls_not_playing_soccer_l531_53172


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l531_53121

def A := Set.Ioo 1 3
def B := Set.Ioo 2 4

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 2 3 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l531_53121


namespace NUMINAMATH_GPT_smallest_n_containing_375_consecutively_l531_53152

theorem smallest_n_containing_375_consecutively :
  ∃ (m n : ℕ), m < n ∧ Nat.gcd m n = 1 ∧ (n = 8) ∧ (∀ (d : ℕ), d < 1000 →
  ∃ (k : ℕ), k * d % n = m ∧ (d / 100) % 10 = 3 ∧ (d / 10) % 10 = 7 ∧ d % 10 = 5) :=
sorry

end NUMINAMATH_GPT_smallest_n_containing_375_consecutively_l531_53152


namespace NUMINAMATH_GPT_reward_function_conditions_l531_53131

theorem reward_function_conditions :
  (∀ x : ℝ, (10 ≤ x ∧ x ≤ 1000) → (y = x / 150 + 2 → y ≤ 90 ∧ y ≤ x / 5) → False) ∧
  (∃ a : ℕ, (∀ x : ℝ, (10 ≤ x ∧ x ≤ 1000) → (y = (10 * x - 3 * a) / (x + 2) → y ≤ 9 ∧ y ≤ x / 5)) ∧ (a = 328)) :=
by
  sorry

end NUMINAMATH_GPT_reward_function_conditions_l531_53131


namespace NUMINAMATH_GPT_total_amount_paid_is_correct_l531_53119

-- Define constants based on conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The proof problem statement
theorem total_amount_paid_is_correct :
  total_cost = 360 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_is_correct_l531_53119


namespace NUMINAMATH_GPT_roots_expression_value_l531_53123

theorem roots_expression_value {a b : ℝ} 
  (h₁ : a^2 + a - 3 = 0) 
  (h₂ : b^2 + b - 3 = 0) 
  (ha_ne_hb : a ≠ b) : 
  a * b - 2023 * a - 2023 * b = 2020 :=
by 
  sorry

end NUMINAMATH_GPT_roots_expression_value_l531_53123


namespace NUMINAMATH_GPT_square_perimeter_l531_53169

-- We define a structure for a square with an area as a condition.
structure Square (s : ℝ) :=
(area_eq : s ^ 2 = 400)

-- The theorem states that given the area of the square is 400 square meters,
-- the perimeter of the square is 80 meters.
theorem square_perimeter (s : ℝ) (sq : Square s) : 4 * s = 80 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_square_perimeter_l531_53169


namespace NUMINAMATH_GPT_max_min_values_l531_53164

def f (x a : ℝ) : ℝ := -x^2 + 2*x + a

theorem max_min_values (a : ℝ) (h : a ≠ 0) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 1 + a) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f x a = 1 + a) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 3 → -3 + a ≤ f x a) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f x a = -3 + a) := 
sorry

end NUMINAMATH_GPT_max_min_values_l531_53164


namespace NUMINAMATH_GPT_max_variance_l531_53181

theorem max_variance (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) : 
  ∃ q, p * (1 - p) ≤ q ∧ q = 1 / 4 :=
by
  existsi (1 / 4)
  sorry

end NUMINAMATH_GPT_max_variance_l531_53181


namespace NUMINAMATH_GPT_remainder_of_exponentiation_l531_53147

theorem remainder_of_exponentiation (n : ℕ) : (3 ^ (2 * n) + 8) % 8 = 1 := 
by sorry

end NUMINAMATH_GPT_remainder_of_exponentiation_l531_53147


namespace NUMINAMATH_GPT_find_sum_of_a_and_c_l531_53191

variable (a b c d : ℝ)

theorem find_sum_of_a_and_c (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) :
  a + c = 8 := by sorry

end NUMINAMATH_GPT_find_sum_of_a_and_c_l531_53191


namespace NUMINAMATH_GPT_project_completion_days_l531_53138

theorem project_completion_days (A B C : ℝ) (h1 : 1/A + 1/B = 1/2) (h2 : 1/B + 1/C = 1/4) (h3 : 1/C + 1/A = 1/2.4) : A = 3 :=
by
sorry

end NUMINAMATH_GPT_project_completion_days_l531_53138


namespace NUMINAMATH_GPT_buratino_loss_l531_53130

def buratino_dollars_lost (x y : ℕ) : ℕ := 5 * y - 3 * x

theorem buratino_loss :
  ∃ (x y : ℕ), x + y = 50 ∧ 3 * y - 2 * x = 0 ∧ buratino_dollars_lost x y = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_buratino_loss_l531_53130


namespace NUMINAMATH_GPT_fibonacci_units_digit_l531_53139

def fibonacci (n : ℕ) : ℕ :=
match n with
| 0     => 4
| 1     => 3
| (n+2) => fibonacci (n+1) + fibonacci n

def units_digit (n : ℕ) : ℕ :=
n % 10

theorem fibonacci_units_digit : units_digit (fibonacci (fibonacci 10)) = 3 := by
  sorry

end NUMINAMATH_GPT_fibonacci_units_digit_l531_53139


namespace NUMINAMATH_GPT_vertical_asymptotes_A_plus_B_plus_C_l531_53192

noncomputable def A : ℤ := -6
noncomputable def B : ℤ := 5
noncomputable def C : ℤ := 12

theorem vertical_asymptotes_A_plus_B_plus_C :
  (x + 1) * (x - 3) * (x - 4) = x^3 + A*x^2 + B*x + C ∧ A + B + C = 11 := by
  sorry

end NUMINAMATH_GPT_vertical_asymptotes_A_plus_B_plus_C_l531_53192


namespace NUMINAMATH_GPT_cube_volume_ratio_l531_53154

theorem cube_volume_ratio
  (a : ℕ) (b : ℕ)
  (h₁ : a = 5)
  (h₂ : b = 24)
  : (a^3 : ℚ) / (b^3 : ℚ) = 125 / 13824 := by
  sorry

end NUMINAMATH_GPT_cube_volume_ratio_l531_53154


namespace NUMINAMATH_GPT_find_m_l531_53117

noncomputable def polynomial (x : ℝ) (m : ℝ) := 4 * x^2 - 3 * x + 5 - 2 * m * x^2 - x + 1

theorem find_m (m : ℝ) : 
  ∀ x : ℝ, (4 * x^2 - 3 * x + 5 - 2 * m * x^2 - x + 1 = (4 - 2 * m) * x^2 - 4 * x + 6)
  → (4 - 2 * m = 0) → (m = 2) :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_find_m_l531_53117


namespace NUMINAMATH_GPT_probability_correct_l531_53160

/-- 
The set of characters in "HMMT2005".
-/
def characters : List Char := ['H', 'M', 'M', 'T', '2', '0', '0', '5']

/--
The number of ways to choose 4 positions out of 8.
-/
def choose_4_from_8 : ℕ := Nat.choose 8 4

/-- 
The factorial of an integer n.
-/
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

/-- 
The number of ways to arrange "HMMT".
-/
def arrangements_hmmt : ℕ := choose_4_from_8 * (factorial 4 / factorial 2)

/-- 
The number of ways to arrange "2005".
-/
def arrangements_2005 : ℕ := choose_4_from_8 * (factorial 4 / factorial 2)

/-- 
The number of arrangements where both "HMMT" and "2005" appear.
-/
def arrangements_both : ℕ := choose_4_from_8

/-- 
The total number of possible arrangements of "HMMT2005".
-/
def total_arrangements : ℕ := factorial 8 / (factorial 2 * factorial 2)

/-- 
The number of desirable arrangements using inclusion-exclusion.
-/
def desirable_arrangements : ℕ := arrangements_hmmt + arrangements_2005 - arrangements_both

/-- 
The probability of being able to read either "HMMT" or "2005" 
in a random arrangement of "HMMT2005".
-/
def probability : ℚ := (desirable_arrangements : ℚ) / (total_arrangements : ℚ)

/-- 
Prove that the computed probability is equal to 23/144.
-/
theorem probability_correct : probability = 23 / 144 := sorry

end NUMINAMATH_GPT_probability_correct_l531_53160


namespace NUMINAMATH_GPT_sum_of_possible_values_of_a_l531_53132

theorem sum_of_possible_values_of_a :
  ∀ (a b c d : ℝ), a > b → b > c → c > d → a + b + c + d = 50 → 
  (a - b = 4 ∧ b - d = 7 ∧ a - c = 5 ∧ c - d = 6 ∧ b - c = 2 ∨
   a - b = 5 ∧ b - d = 6 ∧ a - c = 4 ∧ c - d = 7 ∧ b - c = 2) →
  (a = 17.75 ∨ a = 18.25) →
  a + 18.25 + 17.75 - a = 36 :=
by sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_a_l531_53132


namespace NUMINAMATH_GPT_jo_thinking_number_l531_53114

theorem jo_thinking_number 
  (n : ℕ) 
  (h1 : n < 100) 
  (h2 : n % 8 = 7) 
  (h3 : n % 7 = 4) 
  : n = 95 :=
sorry

end NUMINAMATH_GPT_jo_thinking_number_l531_53114


namespace NUMINAMATH_GPT_sufficient_condition_for_proposition_l531_53186

theorem sufficient_condition_for_proposition (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 5) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_proposition_l531_53186


namespace NUMINAMATH_GPT_smallest_Norwegian_l531_53176

def is_Norwegian (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a * b * c ∧ a + b + c = 2022

theorem smallest_Norwegian :
  ∀ n : ℕ, is_Norwegian n → 1344 ≤ n := by
  sorry

end NUMINAMATH_GPT_smallest_Norwegian_l531_53176


namespace NUMINAMATH_GPT_james_new_fuel_cost_l531_53135

def original_cost : ℕ := 200
def price_increase_rate : ℕ := 20
def extra_tank_factor : ℕ := 2

theorem james_new_fuel_cost :
  let new_price := original_cost + (price_increase_rate * original_cost / 100)
  let total_cost := extra_tank_factor * new_price
  total_cost = 480 :=
by
  sorry

end NUMINAMATH_GPT_james_new_fuel_cost_l531_53135


namespace NUMINAMATH_GPT_find_h_l531_53125

theorem find_h: 
  ∃ h k, (∀ x, 2 * x ^ 2 + 6 * x + 11 = 2 * (x - h) ^ 2 + k) ∧ h = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_h_l531_53125


namespace NUMINAMATH_GPT_drawings_with_colored_pencils_l531_53100

-- Definitions based on conditions
def total_drawings : Nat := 25
def blending_markers_drawings : Nat := 7
def charcoal_drawings : Nat := 4
def colored_pencils_drawings : Nat := total_drawings - (blending_markers_drawings + charcoal_drawings)

-- Theorem to be proven
theorem drawings_with_colored_pencils : colored_pencils_drawings = 14 :=
by
  sorry

end NUMINAMATH_GPT_drawings_with_colored_pencils_l531_53100


namespace NUMINAMATH_GPT_solve_for_y_l531_53149

theorem solve_for_y (y : ℝ) (h : (2 / y) + (3 / y) / (6 / y) = 1.5) : y = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l531_53149


namespace NUMINAMATH_GPT_crate_weight_l531_53105

variable (C : ℝ)
variable (carton_weight : ℝ := 3)
variable (total_weight : ℝ := 96)
variable (num_crates : ℝ := 12)
variable (num_cartons : ℝ := 16)

theorem crate_weight :
  (num_crates * C + num_cartons * carton_weight = total_weight) → (C = 4) :=
by
  sorry

end NUMINAMATH_GPT_crate_weight_l531_53105


namespace NUMINAMATH_GPT_find_k_l531_53124

theorem find_k (k r s : ℝ) 
  (h1 : r + s = -k) 
  (h2 : r * s = 12) 
  (h3 : (r + 7) + (s + 7) = k) : 
  k = 7 := by 
  sorry

end NUMINAMATH_GPT_find_k_l531_53124


namespace NUMINAMATH_GPT_minimum_value_expression_l531_53116

theorem minimum_value_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x ^ 2 + 4 * x * y + 2 * y ^ 2 - 6 * x + 8 * y + 9 ≥ -10 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l531_53116


namespace NUMINAMATH_GPT_second_discount_percentage_l531_53144

-- Define the initial conditions.
def listed_price : ℝ := 200
def first_discount_rate : ℝ := 0.20
def final_sale_price : ℝ := 144

-- Calculate the price after the first discount.
def first_discount_amount := first_discount_rate * listed_price
def price_after_first_discount := listed_price - first_discount_amount

-- Define the second discount amount.
def second_discount_amount := price_after_first_discount - final_sale_price

-- Define the theorem to prove the second discount rate.
theorem second_discount_percentage : 
  (second_discount_amount / price_after_first_discount) * 100 = 10 :=
by 
  sorry -- Proof placeholder

end NUMINAMATH_GPT_second_discount_percentage_l531_53144


namespace NUMINAMATH_GPT_central_angle_of_cone_l531_53115

theorem central_angle_of_cone (A : ℝ) (l : ℝ) (r : ℝ) (θ : ℝ)
  (hA : A = (1 / 2) * 2 * Real.pi * r)
  (hl : l = 1)
  (ha : A = (3 / 8) * Real.pi) :
  θ = (3 / 4) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_cone_l531_53115


namespace NUMINAMATH_GPT_function_f_not_all_less_than_half_l531_53146

theorem function_f_not_all_less_than_half (p q : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = x^2 + p*x + q) :
  ¬ (|f 1| < 1 / 2 ∧ |f 2| < 1 / 2 ∧ |f 3| < 1 / 2) :=
sorry

end NUMINAMATH_GPT_function_f_not_all_less_than_half_l531_53146


namespace NUMINAMATH_GPT_part_I_part_II_l531_53165

def S (n : ℕ) : ℕ := 2 ^ n - 1

def a (n : ℕ) : ℕ := 2 ^ (n - 1)

def T (n : ℕ) : ℕ := (n - 1) * 2 ^ n + 1

theorem part_I (S : ℕ → ℕ) (hS : ∀ n, S n = 2^n - 1) :
  ∀ n, ∃ a : ℕ → ℕ, a n = 2^(n-1) :=
by
  sorry

theorem part_II (a : ℕ → ℕ) (ha : ∀ n, a n = 2^(n-1)) :
  ∀ n, ∃ T : ℕ → ℕ, T n = (n - 1) * 2 ^ n + 1 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l531_53165


namespace NUMINAMATH_GPT_parabola_x_intercepts_l531_53103

theorem parabola_x_intercepts :
  ∃! y : ℝ, -3 * y^2 + 2 * y + 4 = y := 
by
  sorry

end NUMINAMATH_GPT_parabola_x_intercepts_l531_53103


namespace NUMINAMATH_GPT_box_upper_surface_area_l531_53104

theorem box_upper_surface_area (L W H : ℕ) 
    (h1 : L * W = 120) 
    (h2 : L * H = 72) 
    (h3 : L * W * H = 720) : 
    L * W = 120 := 
by 
  sorry

end NUMINAMATH_GPT_box_upper_surface_area_l531_53104
