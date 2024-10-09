import Mathlib

namespace sufficient_but_not_necessary_condition_l1792_179233

variable (x : ℝ)

def p : Prop := (x - 1) / (x + 2) ≥ 0
def q : Prop := (x - 1) * (x + 2) ≥ 0

theorem sufficient_but_not_necessary_condition : (p x) → (q x) ∧ ¬((q x) → (p x)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1792_179233


namespace avg_speed_in_mph_l1792_179281

/-- 
Given conditions:
1. The man travels 10,000 feet due north.
2. He travels 6,000 feet due east in 1/4 less time than he took heading north, traveling at 3 miles per minute.
3. He returns to his starting point by traveling south at 1 mile per minute.
4. He travels back west at the same speed as he went east.
We aim to prove that the average speed for the entire trip is 22.71 miles per hour.
-/
theorem avg_speed_in_mph :
  let distance_north_feet := 10000
  let distance_east_feet := 6000
  let speed_east_miles_per_minute := 3
  let speed_south_miles_per_minute := 1
  let feet_per_mile := 5280
  let distance_north_mil := (distance_north_feet / feet_per_mile : ℝ)
  let distance_east_mil := (distance_east_feet / feet_per_mile : ℝ)
  let time_north_min := distance_north_mil / (1 / 3)
  let time_east_min := time_north_min * 0.75
  let time_south_min := distance_north_mil / speed_south_miles_per_minute
  let time_west_min := time_east_min
  let total_time_hr := (time_north_min + time_east_min + time_south_min + time_west_min) / 60
  let total_distance_miles := 2 * (distance_north_mil + distance_east_mil)
  let avg_speed_mph := total_distance_miles / total_time_hr
  avg_speed_mph = 22.71 := by
sorry

end avg_speed_in_mph_l1792_179281


namespace midpoint_of_segment_l1792_179220

theorem midpoint_of_segment (A B : (ℤ × ℤ)) (hA : A = (12, 3)) (hB : B = (-8, -5)) :
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = -1 :=
by
  sorry

end midpoint_of_segment_l1792_179220


namespace product_of_roots_is_four_thirds_l1792_179232

theorem product_of_roots_is_four_thirds :
  (∀ p q r s : ℚ, (∃ a b c: ℚ, (3 * a^3 - 9 * a^2 + 5 * a - 4 = 0 ∧
                                   3 * b^3 - 9 * b^2 + 5 * b - 4 = 0 ∧
                                   3 * c^3 - 9 * c^2 + 5 * c - 4 = 0)) → 
  - s / p = (4 : ℚ) / 3) := sorry

end product_of_roots_is_four_thirds_l1792_179232


namespace max_xy_l1792_179239

open Real

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 6 * x + 8 * y = 72) (h4 : x = 2 * y) : 
  x * y = 25.92 := 
sorry

end max_xy_l1792_179239


namespace sin_2B_value_l1792_179288

-- Define the triangle's internal angles and the tangent of angles
variables (A B C : ℝ) 

-- Given conditions from the problem
def tan_sequence (tanA tanB tanC : ℝ) : Prop :=
  tanA = (1/2) * tanB ∧
  tanC = (3/2) * tanB ∧
  2 * tanB = tanC + tanB + (tanC - tanA)

-- The statement to be proven
theorem sin_2B_value (h : tan_sequence (Real.tan A) (Real.tan B) (Real.tan C)) :
  Real.sin (2 * B) = 4 / 5 :=
sorry

end sin_2B_value_l1792_179288


namespace paint_needed_to_buy_l1792_179203

def total_paint := 333
def existing_paint := 157

theorem paint_needed_to_buy : total_paint - existing_paint = 176 := by
  sorry

end paint_needed_to_buy_l1792_179203


namespace negation_of_all_exp_monotonic_l1792_179211

theorem negation_of_all_exp_monotonic :
  ¬ (∀ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f x < f y) → (∃ g : ℝ → ℝ, ∃ x y : ℝ, x < y ∧ g x ≥ g y)) :=
sorry

end negation_of_all_exp_monotonic_l1792_179211


namespace carrots_total_l1792_179219

def carrots_grown_by_sally := 6
def carrots_grown_by_fred := 4
def total_carrots := carrots_grown_by_sally + carrots_grown_by_fred

theorem carrots_total : total_carrots = 10 := 
by 
  sorry  -- proof to be filled in

end carrots_total_l1792_179219


namespace problem_KMO_16_l1792_179224

theorem problem_KMO_16
  (m : ℕ) (h_pos : m > 0) :
  (2^(m+1) + 1) ∣ (3^(2^m) + 1) ↔ Nat.Prime (2^(m+1) + 1) :=
by
  sorry

end problem_KMO_16_l1792_179224


namespace gcd_153_119_l1792_179217

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_l1792_179217


namespace volume_of_prism_l1792_179279

-- Define the conditions
variables {a b c : ℝ}
-- Areas of the faces
def ab := 50
def ac := 72
def bc := 45

-- Theorem stating the volume of the prism
theorem volume_of_prism : a * b * c = 180 * Real.sqrt 5 :=
by
  sorry

end volume_of_prism_l1792_179279


namespace baker_weekend_hours_l1792_179296

noncomputable def loaves_per_hour : ℕ := 5
noncomputable def ovens : ℕ := 4
noncomputable def weekday_hours : ℕ := 5
noncomputable def total_loaves : ℕ := 1740
noncomputable def weeks : ℕ := 3
noncomputable def weekday_days : ℕ := 5
noncomputable def weekend_days : ℕ := 2

theorem baker_weekend_hours :
  ((total_loaves - (weeks * weekday_days * weekday_hours * (loaves_per_hour * ovens))) / (weeks * (loaves_per_hour * ovens))) / weekend_days = 4 := by
  sorry

end baker_weekend_hours_l1792_179296


namespace condition_for_all_real_solutions_l1792_179252

theorem condition_for_all_real_solutions (c : ℝ) :
  (∀ x : ℝ, x^2 + x + c > 0) ↔ c > 1 / 4 :=
sorry

end condition_for_all_real_solutions_l1792_179252


namespace initial_men_count_l1792_179261

theorem initial_men_count (M : ℕ) (P : ℝ) 
  (h1 : P = M * 12) 
  (h2 : P = (M + 300) * 9.662337662337663) :
  M = 1240 :=
sorry

end initial_men_count_l1792_179261


namespace sales_tax_difference_l1792_179298

theorem sales_tax_difference
  (price : ℝ)
  (rate1 rate2 : ℝ)
  (h_rate1 : rate1 = 0.075)
  (h_rate2 : rate2 = 0.07)
  (h_price : price = 30) :
  (price * rate1 - price * rate2 = 0.15) :=
by
  sorry

end sales_tax_difference_l1792_179298


namespace percentage_reduction_l1792_179206

theorem percentage_reduction (S P : ℝ) (h : S - (P / 100) * S = S / 2) : P = 50 :=
by
  sorry

end percentage_reduction_l1792_179206


namespace b_2023_value_l1792_179229

noncomputable def seq (b : ℕ → ℝ) : Prop := 
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem b_2023_value (b : ℕ → ℝ) (h1 : seq b) (h2 : b 1 = 2 + Real.sqrt 5) (h3 : b 1984 = 12 + Real.sqrt 5) : 
  b 2023 = -4/3 + 10 * Real.sqrt 5 / 3 :=
sorry

end b_2023_value_l1792_179229


namespace isosceles_triangle_perimeter_l1792_179268

-- Definitions of the conditions
def is_isosceles (a b : ℕ) : Prop :=
  a = b

def has_side_lengths (a b : ℕ) (c : ℕ) : Prop :=
  true

-- The statement to be proved
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h₁ : is_isosceles a b) (h₂ : has_side_lengths a b c) :
  (a + b + c = 16 ∨ a + b + c = 17) :=
sorry

end isosceles_triangle_perimeter_l1792_179268


namespace power_calculation_l1792_179205

theorem power_calculation :
  ((8^5 / 8^3) * 4^6) = 262144 := by
  sorry

end power_calculation_l1792_179205


namespace sum_positive_implies_at_least_one_positive_l1792_179259

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l1792_179259


namespace cost_price_of_watch_l1792_179251

theorem cost_price_of_watch (C : ℝ) (h1 : ∃ C, 0.91 * C + 220 = 1.04 * C) : C = 1692.31 :=
sorry  -- proof to be provided

end cost_price_of_watch_l1792_179251


namespace interest_rate_l1792_179273

theorem interest_rate (P T R : ℝ) (SI CI : ℝ) (difference : ℝ)
  (hP : P = 1700)
  (hT : T = 1)
  (hdiff : difference = 4.25)
  (hSI : SI = P * R * T / 100)
  (hCI : CI = P * ((1 + R / 200)^2 - 1))
  (hDiff : CI - SI = difference) : 
  R = 10 := sorry

end interest_rate_l1792_179273


namespace petya_vasya_equal_again_l1792_179249

theorem petya_vasya_equal_again (n : ℤ) (hn : n ≠ 0) :
  ∃ (k m : ℕ), (∃ P V : ℤ, P = n + 10 * k ∧ V = n - 10 * k ∧ 2014 * P * V = n) :=
sorry

end petya_vasya_equal_again_l1792_179249


namespace angle_BDE_60_l1792_179225

noncomputable def is_isosceles_triangle (A B C : Type) (angle_BAC : ℝ) : Prop :=
angle_BAC = 20

noncomputable def equal_sides (BC BD BE : ℝ) : Prop :=
BC = BD ∧ BD = BE

theorem angle_BDE_60 (A B C D E : Type) (BC BD BE : ℝ) 
  (h1 : is_isosceles_triangle A B C 20) 
  (h2 : equal_sides BC BD BE) : 
  ∃ (angle_BDE : ℝ), angle_BDE = 60 :=
by
  sorry

end angle_BDE_60_l1792_179225


namespace sufficient_condition_of_square_inequality_l1792_179204

variables (a b : ℝ)

theorem sufficient_condition_of_square_inequality (ha : a > 0) (hb : b > 0) (h : a > b) : a^2 > b^2 :=
by {
  sorry
}

end sufficient_condition_of_square_inequality_l1792_179204


namespace simplify_expression_correct_l1792_179213

def simplify_expression (i : ℂ) (h : i ^ 2 = -1) : ℂ :=
  3 * (4 - 2 * i) + 2 * i * (3 - i)

theorem simplify_expression_correct (i : ℂ) (h : i ^ 2 = -1) : simplify_expression i h = 14 := 
by
  sorry

end simplify_expression_correct_l1792_179213


namespace original_avg_is_40_l1792_179242

noncomputable def original_average (A : ℝ) := (15 : ℝ) * A

noncomputable def new_sum (A : ℝ) := (15 : ℝ) * A + 15 * (15 : ℝ)

theorem original_avg_is_40 (A : ℝ) (h : new_sum A / 15 = 55) :
  A = 40 :=
by sorry

end original_avg_is_40_l1792_179242


namespace prove_m_range_l1792_179267

theorem prove_m_range (m : ℝ) :
  (∀ x : ℝ, (2 * x + 5) / 3 - 1 ≤ 2 - x → 3 * (x - 1) + 5 > 5 * x + 2 * (m + x)) → m < -3 / 5 := by
  sorry

end prove_m_range_l1792_179267


namespace sumata_family_miles_driven_l1792_179230

def total_miles_driven (days : ℝ) (miles_per_day : ℝ) : ℝ :=
  days * miles_per_day

theorem sumata_family_miles_driven :
  total_miles_driven 5 50 = 250 :=
by
  sorry

end sumata_family_miles_driven_l1792_179230


namespace complex_exp_sum_l1792_179258

def w : ℂ := sorry  -- We define w as a complex number, satisfying the given condition.

theorem complex_exp_sum (h : w^2 - w + 1 = 0) : 
  w^97 + w^98 + w^99 + w^100 + w^101 + w^102 = -2 + 2 * w :=
by
  sorry

end complex_exp_sum_l1792_179258


namespace find_k_l1792_179240

noncomputable def geometric_series_sum (k : ℝ) (h : k > 1) : ℝ :=
  ∑' n, ((7 * n - 2) / k ^ n)

theorem find_k (k : ℝ) (h : k > 1)
  (series_sum : geometric_series_sum k h = 18 / 5) :
  k = 3.42 :=
by
  sorry

end find_k_l1792_179240


namespace simplify_expression_l1792_179201

theorem simplify_expression (x : ℝ) : 
  (x^2 + 2 * x + 3) / 4 + (3 * x - 5) / 6 = (3 * x^2 + 12 * x - 1) / 12 := 
by
  sorry

end simplify_expression_l1792_179201


namespace range_and_period_range_of_m_l1792_179234

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos (x + Real.pi / 3) * (Real.sin (x + Real.pi / 3) - Real.sqrt 3 * Real.cos (x + Real.pi / 3))

theorem range_and_period (x : ℝ) :
  (Set.range f = Set.Icc (-2 - Real.sqrt 3) (2 - Real.sqrt 3)) ∧ (∀ x, f (x + Real.pi) = f x) := sorry

theorem range_of_m (x m : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi / 6) (h2 : m * (f x + Real.sqrt 3) + 2 = 0) :
  m ∈ Set.Icc (- 2 * Real.sqrt 3 / 3) (-1) := sorry

end range_and_period_range_of_m_l1792_179234


namespace recurring_decimal_to_fraction_l1792_179210

theorem recurring_decimal_to_fraction (a b : ℕ) (ha : a = 356) (hb : b = 999) (hab_gcd : Nat.gcd a b = 1)
  (x : ℚ) (hx : x = 356 / 999) 
  (hx_recurring : x = {num := 356, den := 999}): a + b = 1355 :=
by
  sorry  -- Proof is not required as per the instructions

end recurring_decimal_to_fraction_l1792_179210


namespace arithmetic_sequence_sum_first_five_terms_l1792_179293

theorem arithmetic_sequence_sum_first_five_terms:
  ∀ (a : ℕ → ℤ), a 2 = 1 → a 4 = 7 → (a 1 + a 5 = a 2 + a 4) → (5 * (a 1 + a 5) / 2 = 20) :=
by
  intros a h1 h2 h3
  sorry

end arithmetic_sequence_sum_first_five_terms_l1792_179293


namespace min_value_eq_six_l1792_179228

theorem min_value_eq_six
    (α β : ℝ)
    (k : ℝ)
    (h1 : α^2 + 2 * (k + 3) * α + (k^2 + 3) = 0)
    (h2 : β^2 + 2 * (k + 3) * β + (k^2 + 3) = 0)
    (h3 : (2 * (k + 3))^2 - 4 * (k^2 + 3) ≥ 0) :
    ( (α - 1)^2 + (β - 1)^2 = 6 ) := 
sorry

end min_value_eq_six_l1792_179228


namespace main_theorem_l1792_179280

open Nat

-- Define the conditions
def conditions (p q n : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Odd p ∧ Odd q ∧ n > 1 ∧
  (q^(n+2) % p^n = 3^(n+2) % p^n) ∧ (p^(n+2) % q^n = 3^(n+2) % q^n)

-- Define the conclusion
def conclusion (p q n : ℕ) : Prop :=
  (p = 3 ∧ q = 3)

-- Define the main problem
theorem main_theorem : ∀ p q n : ℕ, conditions p q n → conclusion p q n :=
  by
    intros p q n h
    sorry

end main_theorem_l1792_179280


namespace added_number_is_five_l1792_179282

def original_number := 19
def final_resultant := 129
def doubling_expression (x : ℕ) (y : ℕ) := 3 * (2 * x + y)

theorem added_number_is_five:
  ∃ y, doubling_expression original_number y = final_resultant ↔ y = 5 :=
sorry

end added_number_is_five_l1792_179282


namespace frank_total_cans_l1792_179218

def total_cans_picked_up (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ) : ℕ :=
  let total_bags := bags_saturday + bags_sunday
  total_bags * cans_per_bag

theorem frank_total_cans : total_cans_picked_up 5 3 5 = 40 := by
  sorry

end frank_total_cans_l1792_179218


namespace whiskers_ratio_l1792_179266

/-- Four cats live in the old grey house at the end of the road. Their names are Puffy, Scruffy, Buffy, and Juniper.
Puffy has three times more whiskers than Juniper, but a certain ratio as many as Scruffy. Buffy has the same number of whiskers
as the average number of whiskers on the three other cats. Prove that the ratio of Puffy's whiskers to Scruffy's whiskers is 1:2
given Juniper has 12 whiskers and Buffy has 40 whiskers. -/
theorem whiskers_ratio (J B P S : ℕ) (hJ : J = 12) (hB : B = 40) (hP : P = 3 * J) (hAvg : B = (P + S + J) / 3) :
  P / gcd P S = 1 ∧ S / gcd P S = 2 := by
  sorry

end whiskers_ratio_l1792_179266


namespace ab_greater_than_a_plus_b_l1792_179253

theorem ab_greater_than_a_plus_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a - b = a / b) : ab > a + b :=
sorry

end ab_greater_than_a_plus_b_l1792_179253


namespace prob_2022_2023_l1792_179235

theorem prob_2022_2023 (n : ℤ) (h : (n - 2022)^2 + (2023 - n)^2 = 1) : (n - 2022) * (2023 - n) = 0 :=
sorry

end prob_2022_2023_l1792_179235


namespace average_sales_per_month_after_discount_is_93_l1792_179243

theorem average_sales_per_month_after_discount_is_93 :
  let salesJanuary := 120
  let salesFebruary := 80
  let salesMarch := 70
  let salesApril := 150
  let salesMayBeforeDiscount := 50
  let discountRate := 0.10
  let discountedSalesMay := salesMayBeforeDiscount - (discountRate * salesMayBeforeDiscount)
  let totalSales := salesJanuary + salesFebruary + salesMarch + salesApril + discountedSalesMay
  let numberOfMonths := 5
  let averageSales := totalSales / numberOfMonths
  averageSales = 93 :=
by {
  -- The actual proof code would go here, but we will skip the proof steps as instructed.
  sorry
}

end average_sales_per_month_after_discount_is_93_l1792_179243


namespace complement_union_l1792_179291

def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B : Set ℝ := { x | x ≥ 1 }
def C (s : Set ℝ) : Set ℝ := { x | ¬ s x }

theorem complement_union :
  C (A ∪ B) = { x | x ≤ -1 } :=
by {
  sorry
}

end complement_union_l1792_179291


namespace area_of_midpoint_quadrilateral_l1792_179231

theorem area_of_midpoint_quadrilateral (length width : ℝ) (h_length : length = 15) (h_width : width = 8) :
  let A := (0, width / 2)
  let B := (length / 2, 0)
  let C := (length, width / 2)
  let D := (length / 2, width)
  let mid_quad_area := (length / 2) * (width / 2)
  mid_quad_area = 30 :=
by
  simp [h_length, h_width]
  sorry

end area_of_midpoint_quadrilateral_l1792_179231


namespace no_adjacent_standing_prob_l1792_179270

def coin_flip_probability : ℚ :=
  let a2 := 3
  let a3 := 4
  let a4 := a3 + a2
  let a5 := a4 + a3
  let a6 := a5 + a4
  let a7 := a6 + a5
  let a8 := a7 + a6
  let a9 := a8 + a7
  let a10 := a9 + a8
  let favorable_outcomes := a10
  favorable_outcomes / (2 ^ 10)

theorem no_adjacent_standing_prob :
  coin_flip_probability = (123 / 1024 : ℚ) :=
by sorry

end no_adjacent_standing_prob_l1792_179270


namespace functional_equation_to_linear_l1792_179209

-- Define that f satisfies the Cauchy functional equation
variable (f : ℕ → ℝ)
axiom cauchy_eq (x y : ℕ) : f (x + y) = f x + f y

-- The theorem we want to prove
theorem functional_equation_to_linear (h : ∀ n k : ℕ, f (n * k) = n * f k) : ∃ a : ℝ, ∀ n : ℕ, f n = a * n :=
by
  sorry

end functional_equation_to_linear_l1792_179209


namespace black_circles_count_l1792_179216

theorem black_circles_count (a1 d n : ℕ) (h1 : a1 = 2) (h2 : d = 1) (h3 : n = 16) :
  (n * (a1 + (n - 1) * d) / 2) + n ≤ 160 :=
by
  rw [h1, h2, h3]
  -- Here we will carry out the arithmetic to prove the statement
  sorry

end black_circles_count_l1792_179216


namespace range_of_m_l1792_179236

variable (m : ℝ) -- variable m in the real numbers

-- Definition of proposition p
def p : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

-- Definition of proposition q
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- The theorem statement with the given conditions
theorem range_of_m (h : p m ∧ q m) : -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l1792_179236


namespace g_difference_l1792_179272

def g (n : ℕ) : ℚ :=
  (1 / 4 : ℚ) * n^2 * (n + 1) * (n + 3) + 1

theorem g_difference (m : ℕ) : 
  g m - g (m - 1) = (3 / 4 : ℚ) * m^2 * (m + 5 / 3) :=
by
  sorry

end g_difference_l1792_179272


namespace vector_solution_l1792_179275

theorem vector_solution
  (x y : ℝ)
  (h1 : (2*x - y = 0))
  (h2 : (x^2 + y^2 = 20)) :
  (x = 2 ∧ y = 4) ∨ (x = -2 ∧ y = -4) := 
by
  sorry

end vector_solution_l1792_179275


namespace joyce_new_property_is_10_times_larger_l1792_179214

theorem joyce_new_property_is_10_times_larger :
  let previous_property := 2
  let suitable_acres := 19
  let pond := 1
  let new_property := suitable_acres + pond
  new_property / previous_property = 10 := by {
    let previous_property := 2
    let suitable_acres := 19
    let pond := 1
    let new_property := suitable_acres + pond
    sorry
  }

end joyce_new_property_is_10_times_larger_l1792_179214


namespace math_problem_l1792_179260

noncomputable def find_min_value (a m n : ℝ) (h_a_pos : a > 0) (h_bn : n = 2 * m + 1 / 2)
  (h_b : -a^2 / 2 + 3 * Real.log a = -1 / 2) : ℝ :=
  (3 * Real.sqrt 5 / 5) ^ 2

theorem math_problem (a m n : ℝ) (h_a_pos : a > 0) (h_bn : n = 2 * m + 1 / 2) :
  ∃ b : ℝ, b = -a^2 / 2 + 3 * Real.log a →
  (a - m) ^ 2 + (b - n) ^ 2 = 9 / 5 :=
by
  sorry

end math_problem_l1792_179260


namespace pq_identity_l1792_179290

theorem pq_identity (p q : ℝ) (h1 : p * q = 20) (h2 : p + q = 10) : p^2 + q^2 = 60 :=
sorry

end pq_identity_l1792_179290


namespace seashells_total_correct_l1792_179277

-- Define the initial counts for Henry, John, and Adam.
def initial_seashells_Henry : ℕ := 11
def initial_seashells_John : ℕ := 24
def initial_seashells_Adam : ℕ := 17

-- Define the total initial seashells collected by all.
def total_initial_seashells : ℕ := 83

-- Calculate Leo's initial seashells.
def initial_seashells_Leo : ℕ := total_initial_seashells - (initial_seashells_Henry + initial_seashells_John + initial_seashells_Adam)

-- Define the changes occurred when they returned home.
def extra_seashells_Henry : ℕ := 3
def given_away_seashells_John : ℕ := 5
def percentage_given_away_Leo : ℕ := 40
def extra_seashells_Leo : ℕ := 5

-- Define the final number of seashells each person has.
def final_seashells_Henry : ℕ := initial_seashells_Henry + extra_seashells_Henry
def final_seashells_John : ℕ := initial_seashells_John - given_away_seashells_John
def given_away_seashells_Leo : ℕ := (initial_seashells_Leo * percentage_given_away_Leo) / 100
def final_seashells_Leo : ℕ := initial_seashells_Leo - given_away_seashells_Leo + extra_seashells_Leo
def final_seashells_Adam : ℕ := initial_seashells_Adam

-- Define the total number of seashells they have now.
def total_final_seashells : ℕ := final_seashells_Henry + final_seashells_John + final_seashells_Leo + final_seashells_Adam

-- Proposition that asserts the total number of seashells is 74.
theorem seashells_total_correct :
  total_final_seashells = 74 :=
sorry

end seashells_total_correct_l1792_179277


namespace carla_paints_120_square_feet_l1792_179238

def totalWork : ℕ := 360
def ratioAlex : ℕ := 3
def ratioBen : ℕ := 5
def ratioCarla : ℕ := 4
def ratioTotal : ℕ := ratioAlex + ratioBen + ratioCarla
def workPerPart : ℕ := totalWork / ratioTotal
def carlasWork : ℕ := ratioCarla * workPerPart

theorem carla_paints_120_square_feet : carlasWork = 120 := by
  sorry

end carla_paints_120_square_feet_l1792_179238


namespace andrew_bought_mangoes_l1792_179237

theorem andrew_bought_mangoes (m : ℕ) 
    (grapes_cost : 6 * 74 = 444) 
    (mangoes_cost : m * 59 = total_mangoes_cost) 
    (total_cost_eq_975 : 444 + total_mangoes_cost = 975) 
    (total_cost := 444 + total_mangoes_cost) 
    (total_mangoes_cost := 59 * m) 
    : m = 9 := 
sorry

end andrew_bought_mangoes_l1792_179237


namespace cyclic_quadrilateral_angle_D_l1792_179287

theorem cyclic_quadrilateral_angle_D (A B C D : ℝ) (h1 : A + C = 180) (h2 : B + D = 180) (h3 : 3 * A = 4 * B) (h4 : 3 * A = 6 * C) : D = 100 :=
by
  sorry

end cyclic_quadrilateral_angle_D_l1792_179287


namespace books_at_end_of_month_l1792_179294

-- Definitions based on provided conditions
def initial_books : ℕ := 75
def loaned_books (x : ℕ) : ℕ := 40  -- Rounded from 39.99999999999999
def returned_books (x : ℕ) : ℕ := (loaned_books x * 70) / 100
def not_returned_books (x : ℕ) : ℕ := loaned_books x - returned_books x

-- The statement to be proved
theorem books_at_end_of_month (x : ℕ) : initial_books - not_returned_books x = 63 :=
by
  -- This will be filled in with the actual proof steps later
  sorry

end books_at_end_of_month_l1792_179294


namespace children_attended_play_l1792_179255

variables (A C : ℕ)

theorem children_attended_play
  (h1 : A + C = 610)
  (h2 : 2 * A + C = 960) : 
  C = 260 := 
by 
  -- Proof goes here
  sorry

end children_attended_play_l1792_179255


namespace prob_of_yellow_second_l1792_179269

-- Defining the probabilities based on the given conditions
def prob_white_from_X : ℚ := 5 / 8
def prob_black_from_X : ℚ := 3 / 8
def prob_yellow_from_Y : ℚ := 8 / 10
def prob_yellow_from_Z : ℚ := 3 / 7

-- Combining probabilities
def combined_prob_white_Y : ℚ := prob_white_from_X * prob_yellow_from_Y
def combined_prob_black_Z : ℚ := prob_black_from_X * prob_yellow_from_Z

-- Total probability of drawing a yellow marble in the second draw
def total_prob_yellow_second : ℚ := combined_prob_white_Y + combined_prob_black_Z

-- Proof statement
theorem prob_of_yellow_second :
  total_prob_yellow_second = 37 / 56 := 
sorry

end prob_of_yellow_second_l1792_179269


namespace find_c_values_l1792_179278

noncomputable def line_intercept_product (c : ℝ) : Prop :=
  let x_intercept := -c / 8
  let y_intercept := -c / 5
  x_intercept * y_intercept = 24

theorem find_c_values :
  ∃ c : ℝ, (line_intercept_product c) ∧ (c = 8 * Real.sqrt 15 ∨ c = -8 * Real.sqrt 15) :=
by
  sorry

end find_c_values_l1792_179278


namespace min_guests_l1792_179207

/-- Problem statement:
Given:
1. The total food consumed by all guests is 319 pounds.
2. Each guest consumes no more than 1.5 pounds of meat, 0.3 pounds of vegetables, and 0.2 pounds of dessert.
3. Each guest has equal proportions of meat, vegetables, and dessert.

Prove:
The minimum number of guests such that the total food consumed is less than or equal to 319 pounds is 160.
-/
theorem min_guests (total_food : ℝ) (meat_per_guest : ℝ) (veg_per_guest : ℝ) (dessert_per_guest : ℝ) (G : ℕ) :
  total_food = 319 ∧ meat_per_guest ≤ 1.5 ∧ veg_per_guest ≤ 0.3 ∧ dessert_per_guest ≤ 0.2 ∧
  (meat_per_guest + veg_per_guest + dessert_per_guest = 2.0) →
  G = 160 :=
by
  intros h
  sorry

end min_guests_l1792_179207


namespace frustum_volume_correct_l1792_179292

noncomputable def volume_of_frustum 
(base_edge_original_pyramid : ℝ) (height_original_pyramid : ℝ) 
(base_edge_smaller_pyramid : ℝ) (height_smaller_pyramid : ℝ) : ℝ :=
  let base_area_original := base_edge_original_pyramid ^ 2
  let volume_original := 1 / 3 * base_area_original * height_original_pyramid
  let similarity_ratio := base_edge_smaller_pyramid / base_edge_original_pyramid
  let volume_smaller := volume_original * (similarity_ratio ^ 3)
  volume_original - volume_smaller

theorem frustum_volume_correct 
(base_edge_original_pyramid : ℝ) (height_original_pyramid : ℝ) 
(base_edge_smaller_pyramid : ℝ) (height_smaller_pyramid : ℝ) 
(h_orig_base_edge : base_edge_original_pyramid = 16) 
(h_orig_height : height_original_pyramid = 10) 
(h_smaller_base_edge : base_edge_smaller_pyramid = 8) 
(h_smaller_height : height_smaller_pyramid = 5) : 
  volume_of_frustum base_edge_original_pyramid height_original_pyramid base_edge_smaller_pyramid height_smaller_pyramid = 746.66 :=
by 
  sorry

end frustum_volume_correct_l1792_179292


namespace value_of_p_l1792_179284

variable (m n p : ℝ)

-- The conditions from the problem
def first_point_on_line := m = (n / 6) - (2 / 5)
def second_point_on_line := m + p = ((n + 18) / 6) - (2 / 5)

-- The theorem to prove
theorem value_of_p (h1 : first_point_on_line m n) (h2 : second_point_on_line m n p) : p = 3 :=
  sorry

end value_of_p_l1792_179284


namespace sum_of_integers_l1792_179246

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : 
  x + y = 32 :=
by
  sorry

end sum_of_integers_l1792_179246


namespace cows_in_group_l1792_179286

theorem cows_in_group (D C : ℕ) 
  (h : 2 * D + 4 * C = 2 * (D + C) + 36) : 
  C = 18 :=
by
  sorry

end cows_in_group_l1792_179286


namespace third_number_is_forty_four_l1792_179285

theorem third_number_is_forty_four (a b c d e : ℕ) (h1 : a = e + 1) (h2 : b = e) 
  (h3 : c = e - 1) (h4 : d = e - 2) (h5 : e = e - 3) 
  (h6 : (a + b + c) / 3 = 45) (h7 : (c + d + e) / 3 = 43) : 
  c = 44 := 
sorry

end third_number_is_forty_four_l1792_179285


namespace abs_neg_sqrt_six_l1792_179264

noncomputable def abs_val (x : ℝ) : ℝ :=
  if x < 0 then -x else x

theorem abs_neg_sqrt_six : abs_val (- Real.sqrt 6) = Real.sqrt 6 := by
  -- Proof goes here
  sorry

end abs_neg_sqrt_six_l1792_179264


namespace final_weight_is_correct_l1792_179202

-- Define the initial weight of marble
def initial_weight := 300.0

-- Define the percentage reductions each week
def first_week_reduction := 0.3 * initial_weight
def second_week_reduction := 0.3 * (initial_weight - first_week_reduction)
def third_week_reduction := 0.15 * (initial_weight - first_week_reduction - second_week_reduction)

-- Calculate the final weight of the statue
def final_weight := initial_weight - first_week_reduction - second_week_reduction - third_week_reduction

-- The statement to prove
theorem final_weight_is_correct : final_weight = 124.95 := by
  -- Here would be the proof, which we are omitting
  sorry

end final_weight_is_correct_l1792_179202


namespace quadratic_roots_is_correct_l1792_179227

theorem quadratic_roots_is_correct (a b : ℝ) 
    (h1 : a + b = 16) 
    (h2 : a * b = 225) :
    (∀ x, x^2 - 16 * x + 225 = 0 ↔ x = a ∨ x = b) := sorry

end quadratic_roots_is_correct_l1792_179227


namespace total_tickets_l1792_179223

theorem total_tickets (R K : ℕ) (hR : R = 12) (h_income : 2 * R + (9 / 2) * K = 60) : R + K = 20 :=
sorry

end total_tickets_l1792_179223


namespace divide_group_among_boats_l1792_179283
noncomputable def number_of_ways_divide_group 
  (boatA_capacity : ℕ) 
  (boatB_capacity : ℕ) 
  (boatC_capacity : ℕ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (constraint : ∀ {boat : ℕ}, boat > 1 → num_children ≥ 1 → num_adults ≥ 1) : ℕ := 
    sorry

theorem divide_group_among_boats 
  (boatA_capacity : ℕ := 3) 
  (boatB_capacity : ℕ := 2) 
  (boatC_capacity : ℕ := 1) 
  (num_adults : ℕ := 2) 
  (num_children : ℕ := 2) 
  (constraint : ∀ {boat : ℕ}, boat > 1 → num_children ≥ 1 → num_adults ≥ 1) : 
  number_of_ways_divide_group boatA_capacity boatB_capacity boatC_capacity num_adults num_children constraint = 8 := 
sorry

end divide_group_among_boats_l1792_179283


namespace original_number_is_13_l1792_179265

theorem original_number_is_13 (x : ℝ) (h : 3 * (2 * x + 7) = 99) : x = 13 :=
sorry

end original_number_is_13_l1792_179265


namespace race_distance_l1792_179221

variable (distance : ℝ)

theorem race_distance :
  (0.25 * distance = 50) → (distance = 200) :=
by
  intro h
  sorry

end race_distance_l1792_179221


namespace smallest_square_contains_five_disks_l1792_179295

noncomputable def smallest_side_length := 2 + 2 * Real.sqrt 2

theorem smallest_square_contains_five_disks :
  ∃ (a : ℝ), a = smallest_side_length ∧ (∃ (d : ℕ → ℝ × ℝ), 
    (∀ i, 0 ≤ i ∧ i < 5 → (d i).fst ^ 2 + (d i).snd ^ 2 < (a / 2 - 1) ^ 2) ∧ 
    (∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 ∧ i ≠ j → 
      (d i).fst ^ 2 + (d i).snd ^ 2 + (d j).fst ^ 2 + (d j).snd ^ 2 ≥ 4)) :=
sorry

end smallest_square_contains_five_disks_l1792_179295


namespace least_number_of_cookies_l1792_179241

theorem least_number_of_cookies (c : ℕ) :
  (c % 6 = 5) ∧ (c % 8 = 7) ∧ (c % 9 = 6) → c = 23 :=
by
  sorry

end least_number_of_cookies_l1792_179241


namespace division_rounded_nearest_hundredth_l1792_179222

theorem division_rounded_nearest_hundredth :
  Float.round (285 * 387 / (981^2) * 100) / 100 = 0.11 :=
by
  sorry

end division_rounded_nearest_hundredth_l1792_179222


namespace geometric_sequence_l1792_179274

theorem geometric_sequence (a : ℝ) (h1 : a > 0)
  (h2 : ∃ r : ℝ, 210 * r = a ∧ a * r = 63 / 40) :
  a = 18.1875 :=
by
  sorry

end geometric_sequence_l1792_179274


namespace total_area_is_8_units_l1792_179245

-- Let s be the side length of the original square and x be the leg length of each isosceles right triangle
variables (s x : ℕ)

-- The side length of the smaller square is 8 units
axiom smaller_square_length : s - 2 * x = 8

-- The area of one isosceles right triangle
def area_triangle : ℕ := x * x / 2

-- There are four triangles
def total_area_triangles : ℕ := 4 * area_triangle x

-- The aim is to prove that the total area of the removed triangles is 8 square units
theorem total_area_is_8_units : total_area_triangles x = 8 :=
sorry

end total_area_is_8_units_l1792_179245


namespace bisectors_form_inscribed_quadrilateral_l1792_179248

noncomputable def angle_sum_opposite_bisectors {α β γ δ : ℝ} (a_bisector b_bisector c_bisector d_bisector : ℝ)
  (cond : α + β + γ + δ = 360) : Prop :=
  (a_bisector + b_bisector + c_bisector + d_bisector) = 180

theorem bisectors_form_inscribed_quadrilateral
  {α β γ δ : ℝ} (convex_quad : α + β + γ + δ = 360) :
  ∃ a_bisector b_bisector c_bisector d_bisector : ℝ,
  angle_sum_opposite_bisectors a_bisector b_bisector c_bisector d_bisector convex_quad := 
sorry

end bisectors_form_inscribed_quadrilateral_l1792_179248


namespace arithmetic_prog_triangle_l1792_179212

theorem arithmetic_prog_triangle (a b c : ℝ) (h : a < b ∧ b < c ∧ 2 * b = a + c)
    (hα : ∀ t, t = a ↔ t = min a (min b c))
    (hγ : ∀ t, t = c ↔ t = max a (max b c)) :
    3 * (Real.tan (α / 2)) * (Real.tan (γ / 2)) = 1 := sorry

end arithmetic_prog_triangle_l1792_179212


namespace overall_percentage_favoring_new_tool_l1792_179200

theorem overall_percentage_favoring_new_tool (teachers students : ℕ) 
  (favor_teachers favor_students : ℚ) 
  (surveyed_teachers surveyed_students : ℕ) : 
  surveyed_teachers = 200 → 
  surveyed_students = 800 → 
  favor_teachers = 0.4 → 
  favor_students = 0.75 → 
  ( ( (favor_teachers * surveyed_teachers) + (favor_students * surveyed_students) ) / (surveyed_teachers + surveyed_students) ) * 100 = 68 := 
by 
  sorry

end overall_percentage_favoring_new_tool_l1792_179200


namespace find_x3_l1792_179215

noncomputable def x3 : ℝ :=
  Real.log ((2 / 3) + (1 / 3) * Real.exp 2)

theorem find_x3 
  (x1 x2 : ℝ)
  (h1 : x1 = 0)
  (h2 : x2 = 2)
  (A : ℝ × ℝ := (x1, Real.exp x1))
  (B : ℝ × ℝ := (x2, Real.exp x2))
  (C : ℝ × ℝ := ((2 * A.1 + B.1) / 3, (2 * A.2 + B.2) / 3))
  (yC : ℝ := (2 / 3) * A.2 + (1 / 3) * B.2)
  (E : ℝ × ℝ := (x3, yC)) :
  E.1 = Real.log ((2 / 3) + (1 / 3) * Real.exp x2) := sorry

end find_x3_l1792_179215


namespace simplify_sqrt_expr_l1792_179299

-- We need to prove that simplifying √(5 - 2√6) is equal to √3 - √2.
theorem simplify_sqrt_expr : 
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 :=
by 
  sorry

end simplify_sqrt_expr_l1792_179299


namespace first_to_receive_10_pieces_l1792_179250

-- Definitions and conditions
def children := [1, 2, 3, 4, 5, 6, 7, 8]
def distribution_cycle := [1, 3, 6, 8, 3, 5, 8, 2, 5, 7, 2, 4, 7, 1, 4, 6]

def count_occurrences (n : ℕ) (lst : List ℕ) : ℕ :=
  lst.count n

-- Theorem
theorem first_to_receive_10_pieces : ∃ k, k = 3 ∧ count_occurrences k distribution_cycle = 2 :=
by
  sorry

end first_to_receive_10_pieces_l1792_179250


namespace posts_needed_l1792_179208

-- Define the main properties
def length_of_side_W_stone_wall := 80
def short_side := 50
def intervals (metres: ℕ) := metres / 10 + 1 

-- Define the conditions
def posts_along_w_stone_wall := intervals length_of_side_W_stone_wall
def posts_along_short_sides := 2 * (intervals short_side - 1)

-- Calculate total posts
def total_posts := posts_along_w_stone_wall + posts_along_short_sides

-- Define the theorem
theorem posts_needed : total_posts = 19 := 
by
  sorry

end posts_needed_l1792_179208


namespace fraction_of_students_speak_foreign_language_l1792_179254

noncomputable def students_speak_foreign_language_fraction (M F : ℕ) (h1 : M = F) (m_frac : ℚ) (f_frac : ℚ) : ℚ :=
  ((3 / 5) * M + (2 / 3) * F) / (M + F)

theorem fraction_of_students_speak_foreign_language (M F : ℕ) (h1 : M = F) :
  students_speak_foreign_language_fraction M F h1 (3 / 5) (2 / 3) = 19 / 30 :=
by 
  sorry

end fraction_of_students_speak_foreign_language_l1792_179254


namespace maximum_k_value_l1792_179297

noncomputable def max_value_k (a : ℝ) (b : ℝ) (k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ a^2 + b^2 ≥ k ∧ k = 1 / 2

theorem maximum_k_value (a b : ℝ) :
  (a > 0 ∧ b > 0 ∧ a + b = 1) → a^2 + b^2 ≥ 1 / 2 :=
by
  intro h
  obtain ⟨ha, hb, hab⟩ := h
  sorry

end maximum_k_value_l1792_179297


namespace mary_total_nickels_l1792_179244

-- Definitions for the conditions
def initial_nickels := 7
def dad_nickels := 5
def mom_nickels := 3 * dad_nickels
def chore_nickels := 2

-- The proof problem statement
theorem mary_total_nickels : 
  initial_nickels + dad_nickels + mom_nickels + chore_nickels = 29 := 
by
  sorry

end mary_total_nickels_l1792_179244


namespace Maurice_current_age_l1792_179289

variable (Ron_now Maurice_now : ℕ)

theorem Maurice_current_age
  (h1 : Ron_now = 43)
  (h2 : ∀ t Ron_future Maurice_future : ℕ, Ron_future = 4 * Maurice_future → Ron_future = Ron_now + 5 → Maurice_future = Maurice_now + 5) :
  Maurice_now = 7 := 
sorry

end Maurice_current_age_l1792_179289


namespace number_of_polynomials_satisfying_P_neg1_eq_neg12_l1792_179247

noncomputable def count_polynomials_satisfying_condition : ℕ := 
  sorry

theorem number_of_polynomials_satisfying_P_neg1_eq_neg12 :
  count_polynomials_satisfying_condition = 455 := 
  sorry

end number_of_polynomials_satisfying_P_neg1_eq_neg12_l1792_179247


namespace math_expression_evaluation_l1792_179271

theorem math_expression_evaluation :
  |1 - Real.sqrt 3| + 3 * Real.tan (Real.pi / 6) - (1/2)⁻¹ + (3 - Real.pi)^0 = 3.732 + Real.sqrt 3 := by
  sorry

end math_expression_evaluation_l1792_179271


namespace total_area_is_71_l1792_179226

noncomputable def area_of_combined_regions 
  (PQ QR RS TU : ℕ) 
  (PQRSTU_is_rectangle : true) 
  (right_angles : true): ℕ :=
  let Area_PQRSTU := PQ * QR
  let VU := TU - PQ
  let WT := TU - RS
  let Area_triangle_PVU := (1 / 2) * VU * PQ
  let Area_triangle_RWT := (1 / 2) * WT * RS
  Area_PQRSTU + Area_triangle_PVU + Area_triangle_RWT

theorem total_area_is_71
  (PQ QR RS TU : ℕ) 
  (h1 : PQ = 8)
  (h2 : QR = 6)
  (h3 : RS = 5)
  (h4 : TU = 10)
  (PQRSTU_is_rectangle : true)
  (right_angles : true) :
  area_of_combined_regions PQ QR RS TU PQRSTU_is_rectangle right_angles = 71 :=
by
  -- The proof is omitted as per the instructions
  sorry

end total_area_is_71_l1792_179226


namespace promotional_savings_l1792_179257

noncomputable def y (x : ℝ) : ℝ :=
if x ≤ 500 then x
else if x ≤ 1000 then 500 + 0.8 * (x - 500)
else 500 + 400 + 0.5 * (x - 1000)

theorem promotional_savings (payment : ℝ) (hx : y 2400 = 1600) : 2400 - payment = 800 :=
by sorry

end promotional_savings_l1792_179257


namespace find_second_number_l1792_179263

theorem find_second_number (x y z : ℚ) (h₁ : x + y + z = 150) (h₂ : x = (3 / 4) * y) (h₃ : z = (7 / 5) * y) : 
  y = 1000 / 21 :=
by sorry

end find_second_number_l1792_179263


namespace find_weight_b_l1792_179262

theorem find_weight_b (A B C : ℕ) 
  (h1 : A + B + C = 90)
  (h2 : A + B = 50)
  (h3 : B + C = 56) : 
  B = 16 :=
sorry

end find_weight_b_l1792_179262


namespace solve_for_y_l1792_179276

theorem solve_for_y : ∃ y : ℕ, 8^4 = 2^y ∧ y = 12 := by
  sorry

end solve_for_y_l1792_179276


namespace range_independent_variable_l1792_179256

def domain_of_function (x : ℝ) : Prop :=
  x ≥ -1 ∧ x ≠ 0

theorem range_independent_variable (x : ℝ) :
  domain_of_function x ↔ x ≥ -1 ∧ x ≠ 0 :=
by
  sorry

end range_independent_variable_l1792_179256
