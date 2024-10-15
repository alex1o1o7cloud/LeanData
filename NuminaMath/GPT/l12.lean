import Mathlib

namespace NUMINAMATH_GPT_solution_exists_l12_1218

theorem solution_exists (x : ℝ) : (x - 1)^2 = 4 → (x = 3 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_solution_exists_l12_1218


namespace NUMINAMATH_GPT_find_x1_l12_1228

theorem find_x1 (x1 x2 x3 x4 : ℝ) (h1 : 0 ≤ x4) (h2 : x4 ≤ x3) (h3 : x3 ≤ x2) (h4 : x2 ≤ x1) (h5 : x1 ≤ 1)
  (h6 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) : x1 = 4 / 5 := 
  sorry

end NUMINAMATH_GPT_find_x1_l12_1228


namespace NUMINAMATH_GPT_tyler_brother_age_difference_l12_1202

-- Definitions of Tyler's age and the sum of their ages:
def tyler_age : ℕ := 7
def sum_of_ages (brother_age : ℕ) : Prop := tyler_age + brother_age = 11

-- Proof problem: Prove that Tyler's brother's age minus Tyler's age equals 4 years.
theorem tyler_brother_age_difference (B : ℕ) (h : sum_of_ages B) : B - tyler_age = 4 :=
by
  sorry

end NUMINAMATH_GPT_tyler_brother_age_difference_l12_1202


namespace NUMINAMATH_GPT_binomial_expansion_a5_l12_1233

theorem binomial_expansion_a5 (x : ℝ) 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) 
  (h : (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x) ^ 2 + a_3 * (1 + x) ^ 3 + a_4 * (1 + x) ^ 4 + a_5 * (1 + x) ^ 5 + a_6 * (1 + x) ^ 6 + a_7 * (1 + x) ^ 7 + a_8 * (1 + x) ^ 8) : 
  a_5 = -448 := 
sorry

end NUMINAMATH_GPT_binomial_expansion_a5_l12_1233


namespace NUMINAMATH_GPT_parallel_vectors_sum_coords_l12_1215

theorem parallel_vectors_sum_coords
  (x y : ℝ)
  (a b : ℝ × ℝ × ℝ)
  (h_a : a = (2, x, 3))
  (h_b : b = (-4, 2, y))
  (h_parallel : ∃ k : ℝ, a = k • b) :
  x + y = -7 :=
sorry

end NUMINAMATH_GPT_parallel_vectors_sum_coords_l12_1215


namespace NUMINAMATH_GPT_tan_150_deg_l12_1222

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_150_deg_l12_1222


namespace NUMINAMATH_GPT_sufficient_conditions_for_x_squared_lt_one_l12_1254

variable (x : ℝ)

theorem sufficient_conditions_for_x_squared_lt_one :
  (∀ x, (0 < x ∧ x < 1) → (x^2 < 1)) ∧
  (∀ x, (-1 < x ∧ x < 0) → (x^2 < 1)) ∧
  (∀ x, (-1 < x ∧ x < 1) → (x^2 < 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_conditions_for_x_squared_lt_one_l12_1254


namespace NUMINAMATH_GPT_fraction_zero_value_l12_1263

theorem fraction_zero_value (x : ℝ) (h : (3 - x) ≠ 0) : (x+2)/(3-x) = 0 ↔ x = -2 := by
  sorry

end NUMINAMATH_GPT_fraction_zero_value_l12_1263


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l12_1248

theorem sufficient_but_not_necessary (a : ℝ) :
  0 < a ∧ a < 1 → (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) ∧ ¬ (∀ a, (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) → 0 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l12_1248


namespace NUMINAMATH_GPT_largest_inscribed_rectangle_area_l12_1206

theorem largest_inscribed_rectangle_area : 
  ∀ (width length : ℝ) (a b : ℝ), 
  width = 8 → length = 12 → 
  (a = (8 / Real.sqrt 3) ∧ b = 2 * a) → 
  (area : ℝ) = (12 * (8 - a)) → 
  area = (96 - 32 * Real.sqrt 3) :=
by
  intros width length a b hw hl htr harea
  sorry

end NUMINAMATH_GPT_largest_inscribed_rectangle_area_l12_1206


namespace NUMINAMATH_GPT_intersect_three_points_l12_1226

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * x

theorem intersect_three_points (a : ℝ) :
  (∃ (t1 t2 t3 : ℝ), t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    f t1 = g t1 a ∧ f t2 = g t2 a ∧ f t3 = g t3 a) ↔ 
  a ∈ Set.Ioo (2 / (7 * Real.pi)) (2 / (3 * Real.pi)) ∨ a = -2 / (5 * Real.pi) :=
sorry

end NUMINAMATH_GPT_intersect_three_points_l12_1226


namespace NUMINAMATH_GPT_systematic_sampling_l12_1264

theorem systematic_sampling (N n : ℕ) (hN : N = 1650) (hn : n = 35) :
  let E := 5 
  let segments := 35 
  let individuals_per_segment := 47 
  1650 % 35 = E ∧ 
  (1650 - E) / 35 = individuals_per_segment :=
by 
  sorry

end NUMINAMATH_GPT_systematic_sampling_l12_1264


namespace NUMINAMATH_GPT_remainder_of_division_l12_1238

theorem remainder_of_division :
  ∃ R : ℕ, 176 = (19 * 9) + R ∧ R = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_division_l12_1238


namespace NUMINAMATH_GPT_simplify_expression_l12_1250

theorem simplify_expression :
  (-2) ^ 2006 + (-1) ^ 3007 + 1 ^ 3010 - (-2) ^ 2007 = -2 ^ 2006 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l12_1250


namespace NUMINAMATH_GPT_triangle_sides_arithmetic_progression_l12_1225

theorem triangle_sides_arithmetic_progression (a d : ℤ) (h : 3 * a = 15) (h1 : a > 0) (h2 : d ≥ 0) :
  (a - d = 5 ∨ a - d = 4 ∨ a - d = 3) ∧ 
  (a = 5) ∧ 
  (a + d = 5 ∨ a + d = 6 ∨ a + d = 7) := 
  sorry

end NUMINAMATH_GPT_triangle_sides_arithmetic_progression_l12_1225


namespace NUMINAMATH_GPT_line_through_center_parallel_to_given_line_l12_1220

def point_in_line (p : ℝ × ℝ) (a b c : ℝ) : Prop :=
  a * p.1 + b * p.2 + c = 0

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  -a / b

theorem line_through_center_parallel_to_given_line :
  ∃ a b c : ℝ, a = 2 ∧ b = -1 ∧ c = -4 ∧
    point_in_line (2, 0) a b c ∧
    slope_of_line a b c = slope_of_line 2 (-1) 1 :=
by
  sorry

end NUMINAMATH_GPT_line_through_center_parallel_to_given_line_l12_1220


namespace NUMINAMATH_GPT_ratio_square_areas_l12_1286

theorem ratio_square_areas (r : ℝ) (h1 : r > 0) :
  let s1 := 2 * r / Real.sqrt 5
  let area1 := (s1) ^ 2
  let h := r * Real.sqrt 3
  let s2 := r
  let area2 := (s2) ^ 2
  area1 / area2 = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_square_areas_l12_1286


namespace NUMINAMATH_GPT_A_iff_B_l12_1245

-- Define Proposition A: ab > b^2
def PropA (a b : ℝ) : Prop := a * b > b ^ 2

-- Define Proposition B: 1/b < 1/a < 0
def PropB (a b : ℝ) : Prop := 1 / b < 1 / a ∧ 1 / a < 0

theorem A_iff_B (a b : ℝ) : (PropA a b) ↔ (PropB a b) := sorry

end NUMINAMATH_GPT_A_iff_B_l12_1245


namespace NUMINAMATH_GPT_chocolate_distribution_l12_1259

theorem chocolate_distribution (n : ℕ) 
  (h1 : 12 * 2 ≤ n * 2 ∨ n * 2 ≤ 12 * 2) 
  (h2 : ∃ d : ℚ, (12 / n) = d ∧ d * n = 12) : 
  n = 15 :=
by 
  sorry

end NUMINAMATH_GPT_chocolate_distribution_l12_1259


namespace NUMINAMATH_GPT_tim_morning_running_hours_l12_1217

theorem tim_morning_running_hours 
  (runs_per_week : ℕ) 
  (total_hours_per_week : ℕ) 
  (runs_per_day : ℕ → ℕ) 
  (hrs_per_day_morning_evening_equal : ∀ (d : ℕ), runs_per_day d = runs_per_week * total_hours_per_week / runs_per_week) 
  (hrs_per_day : ℕ) 
  (hrs_per_morning : ℕ) 
  (hrs_per_evening : ℕ) 
  : hrs_per_morning = 1 :=
by 
  -- Given conditions
  have hrs_per_day := total_hours_per_week / runs_per_week
  have hrs_per_morning_evening := hrs_per_day / 2
  -- Conclusion
  sorry

end NUMINAMATH_GPT_tim_morning_running_hours_l12_1217


namespace NUMINAMATH_GPT_product_of_primes_impossible_l12_1292

theorem product_of_primes_impossible (q : ℕ) (hq1 : Nat.Prime q) (hq2 : q % 2 = 1) :
  ¬ ∀ i ∈ Finset.range (q-1), ∃ p1 p2 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ (i^2 + i + q = p1 * p2) :=
sorry

end NUMINAMATH_GPT_product_of_primes_impossible_l12_1292


namespace NUMINAMATH_GPT_sheela_monthly_income_l12_1249

theorem sheela_monthly_income (d : ℝ) (p : ℝ) (income : ℝ) (h1 : d = 4500) (h2 : p = 0.28) (h3 : d = p * income) : 
  income = 16071.43 :=
by
  sorry

end NUMINAMATH_GPT_sheela_monthly_income_l12_1249


namespace NUMINAMATH_GPT_helmet_price_for_given_profit_helmet_price_for_max_profit_l12_1271

section helmet_sales

-- Define the conditions
variable (original_price : ℝ := 80) (initial_sales : ℝ := 200) (cost_price : ℝ := 50) 
variable (price_reduction_unit : ℝ := 1) (additional_sales_per_reduction : ℝ := 10)
variable (minimum_price_reduction : ℝ := 10)

-- Profits
def profit (x : ℝ) : ℝ :=
  (original_price - x - cost_price) * (initial_sales + additional_sales_per_reduction * x)

-- Prove the selling price when profit is 5250 yuan
theorem helmet_price_for_given_profit (GDP : profit 15 = 5250) : (original_price - 15) = 65 :=
by
  sorry

-- Prove the price for maximum profit
theorem helmet_price_for_max_profit : 
  ∃ x, x = 10 ∧ (original_price - x = 70) ∧ (profit x = 6000) :=
by 
  sorry

end helmet_sales

end NUMINAMATH_GPT_helmet_price_for_given_profit_helmet_price_for_max_profit_l12_1271


namespace NUMINAMATH_GPT_compute_9_times_one_seventh_pow_4_l12_1269

theorem compute_9_times_one_seventh_pow_4 :
  9 * (1 / 7) ^ 4 = 9 / 2401 :=
by
  -- The actual proof would go here.
  sorry

end NUMINAMATH_GPT_compute_9_times_one_seventh_pow_4_l12_1269


namespace NUMINAMATH_GPT_number_multiplied_by_3_l12_1296

theorem number_multiplied_by_3 (k : ℕ) : 
  2^13 - 2^(13-2) = 3 * k → k = 2048 :=
by
  sorry

end NUMINAMATH_GPT_number_multiplied_by_3_l12_1296


namespace NUMINAMATH_GPT_intersection_S_T_l12_1246

def S := {x : ℝ | abs x < 5}
def T := {x : ℝ | (x + 7) * (x - 3) < 0}

theorem intersection_S_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_S_T_l12_1246


namespace NUMINAMATH_GPT_distance_walked_east_l12_1268

-- Definitions for distances
def s1 : ℕ := 25   -- distance walked south
def s2 : ℕ := 20   -- distance walked east
def s3 : ℕ := 25   -- distance walked north
def final_distance : ℕ := 35   -- final distance from the starting point

-- Proof problem: Prove that the distance walked east in the final step is as expected
theorem distance_walked_east (d : Real) :
  d = Real.sqrt (final_distance ^ 2 - s2 ^ 2) :=
sorry

end NUMINAMATH_GPT_distance_walked_east_l12_1268


namespace NUMINAMATH_GPT_combination_sum_l12_1239

theorem combination_sum : Nat.choose 10 3 + Nat.choose 10 4 = 330 := 
by
  sorry

end NUMINAMATH_GPT_combination_sum_l12_1239


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_solve_equation_3_l12_1230

theorem solve_equation_1 : ∀ x : ℝ, (4 * (x + 3) = 25) ↔ (x = 13 / 4) :=
by
  sorry

theorem solve_equation_2 : ∀ x : ℝ, (5 * x^2 - 3 * x = x + 1) ↔ (x = -1 / 5 ∨ x = 1) :=
by
  sorry

theorem solve_equation_3 : ∀ x : ℝ, (2 * (x - 2)^2 - (x - 2) = 0) ↔ (x = 2 ∨ x = 5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_solve_equation_3_l12_1230


namespace NUMINAMATH_GPT_number_of_players_l12_1275

theorem number_of_players (S : ℕ) (h1 : S = 22) (h2 : ∀ (n : ℕ), S = n * 2) : ∃ n, n = 11 :=
by
  sorry

end NUMINAMATH_GPT_number_of_players_l12_1275


namespace NUMINAMATH_GPT_zack_traveled_to_18_countries_l12_1284

-- Defining the conditions
variables (countries_traveled_by_george countries_traveled_by_joseph 
           countries_traveled_by_patrick countries_traveled_by_zack : ℕ)

-- Set the conditions as per the problem statement
axiom george_traveled : countries_traveled_by_george = 6
axiom joseph_traveled : countries_traveled_by_joseph = countries_traveled_by_george / 2
axiom patrick_traveled : countries_traveled_by_patrick = 3 * countries_traveled_by_joseph
axiom zack_traveled : countries_traveled_by_zack = 2 * countries_traveled_by_patrick

-- The theorem to prove Zack traveled to 18 countries
theorem zack_traveled_to_18_countries : countries_traveled_by_zack = 18 :=
by
  -- Adding the proof here is unnecessary as per the instructions
  sorry

end NUMINAMATH_GPT_zack_traveled_to_18_countries_l12_1284


namespace NUMINAMATH_GPT_minimum_prime_product_l12_1221

noncomputable def is_prime : ℕ → Prop := sorry -- Assume the definition of prime

theorem minimum_prime_product (m n p : ℕ) 
  (hm : is_prime m) 
  (hn : is_prime n) 
  (hp : is_prime p) 
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_sum : m + n = p) : 
  m * n * p = 30 :=
sorry

end NUMINAMATH_GPT_minimum_prime_product_l12_1221


namespace NUMINAMATH_GPT_yogurt_price_l12_1256

theorem yogurt_price (x y : ℝ) (h1 : 4 * x + 4 * y = 14) (h2 : 2 * x + 8 * y = 13) : x = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_yogurt_price_l12_1256


namespace NUMINAMATH_GPT_hyperbola_asymptote_m_l12_1260

def isAsymptote (x y : ℝ) (m : ℝ) : Prop :=
  y = m * x ∨ y = -m * x

theorem hyperbola_asymptote_m (m : ℝ) : 
  (∀ x y, (x^2 / 25 - y^2 / 16 = 1 → isAsymptote x y m)) ↔ m = 4 / 5 := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_m_l12_1260


namespace NUMINAMATH_GPT_cube_volume_fourth_power_l12_1236

theorem cube_volume_fourth_power (s : ℝ) (h : 6 * s^2 = 864) : s^4 = 20736 :=
sorry

end NUMINAMATH_GPT_cube_volume_fourth_power_l12_1236


namespace NUMINAMATH_GPT_range_of_x_l12_1285

variable {p : ℝ} {x : ℝ}

theorem range_of_x (h : 0 ≤ p ∧ p ≤ 4) : x^2 + p * x > 4 * x + p - 3 ↔ (x ≤ -1 ∨ x ≥ 3) :=
sorry

end NUMINAMATH_GPT_range_of_x_l12_1285


namespace NUMINAMATH_GPT_sweater_markup_percentage_l12_1201

-- The wholesale cost W and retail price R
variables (W R : ℝ)

-- The given condition
variable (h : 0.30 * R = 1.40 * W)

-- The theorem to prove
theorem sweater_markup_percentage (h : 0.30 * R = 1.40 * W) : (R - W) / W * 100 = 366.67 :=
by
  -- The solution steps would be placed here, if we were proving.
  sorry

end NUMINAMATH_GPT_sweater_markup_percentage_l12_1201


namespace NUMINAMATH_GPT_solve_linear_function_l12_1214

theorem solve_linear_function :
  (∀ (x y : ℤ), (x = -3 ∧ y = -4) ∨ (x = -2 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ 
                      (x = 0 ∧ y = 2) ∨ (x = 1 ∧ y = 4) ∨ (x = 2 ∧ y = 6) →
   ∃ (a b : ℤ), y = a * x + b ∧ a * 1 + b = 4) :=
sorry

end NUMINAMATH_GPT_solve_linear_function_l12_1214


namespace NUMINAMATH_GPT_simplify_expression_l12_1241

variable (a b : ℤ) -- Define variables a and b

theorem simplify_expression : 
  (35 * a + 70 * b + 15) + (15 * a + 54 * b + 5) - (20 * a + 85 * b + 10) =
  30 * a + 39 * b + 10 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l12_1241


namespace NUMINAMATH_GPT_martha_children_l12_1255

noncomputable def num_children (total_cakes : ℕ) (cakes_per_child : ℕ) : ℕ :=
  total_cakes / cakes_per_child

theorem martha_children : num_children 18 6 = 3 := by
  sorry

end NUMINAMATH_GPT_martha_children_l12_1255


namespace NUMINAMATH_GPT_number_of_rolls_l12_1276

theorem number_of_rolls (p : ℚ) (h : p = 1 / 9) : (2 : ℕ) = 2 :=
by 
  have h1 : 2 = 2 := rfl
  exact h1

end NUMINAMATH_GPT_number_of_rolls_l12_1276


namespace NUMINAMATH_GPT_simplify_expression_l12_1251

theorem simplify_expression : 
  (3.875 * (1 / 5) + (38 + 3 / 4) * 0.09 - 0.155 / 0.4) / 
  (2 + 1 / 6 + (((4.32 - 1.68 - (1 + 8 / 25)) * (5 / 11) - 2 / 7) / (1 + 9 / 35)) + (1 + 11 / 24))
  = 1 := sorry

end NUMINAMATH_GPT_simplify_expression_l12_1251


namespace NUMINAMATH_GPT_slab_length_l12_1205

noncomputable def area_of_one_slab (total_area: ℝ) (num_slabs: ℕ) : ℝ :=
  total_area / num_slabs

noncomputable def length_of_one_slab (slab_area : ℝ) : ℝ :=
  Real.sqrt slab_area

theorem slab_length (total_area : ℝ) (num_slabs : ℕ)
  (h_total_area : total_area = 98)
  (h_num_slabs : num_slabs = 50) :
  length_of_one_slab (area_of_one_slab total_area num_slabs) = 1.4 :=
by
  sorry

end NUMINAMATH_GPT_slab_length_l12_1205


namespace NUMINAMATH_GPT_least_number_subtracted_l12_1243

theorem least_number_subtracted (n : ℕ) (h : n = 2361) : 
  ∃ k, (n - k) % 23 = 0 ∧ k = 15 := 
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l12_1243


namespace NUMINAMATH_GPT_rectangle_shaded_area_fraction_l12_1208

-- Defining necessary parameters and conditions
variables {R : Type} [LinearOrderedField R]

noncomputable def shaded_fraction (length width : R) : R :=
  let P : R × R := (0, width / 2)
  let Q : R × R := (length / 2, width)
  let rect_area := length * width
  let tri_area := (1 / 2) * (length / 2) * (width / 2)
  let shaded_area := rect_area - tri_area
  shaded_area / rect_area

-- The theorem stating our desired proof goal
theorem rectangle_shaded_area_fraction (length width : R) (h_length : 0 < length) (h_width : 0 < width) :
  shaded_fraction length width = 7 / 8 := by
  sorry

end NUMINAMATH_GPT_rectangle_shaded_area_fraction_l12_1208


namespace NUMINAMATH_GPT_distance_and_area_of_triangle_l12_1294

theorem distance_and_area_of_triangle :
  let p1 := (0, 6)
  let p2 := (8, 0)
  let origin := (0, 0)
  let distance := Real.sqrt ((8 - 0)^2 + (0 - 6)^2)
  let area := (1 / 2 : ℝ) * 8 * 6
  distance = 10 ∧ area = 24 :=
by
  let p1 := (0, 6)
  let p2 := (8, 0)
  let origin := (0, 0)
  let distance := Real.sqrt ((8 - 0)^2 + (0 - 6)^2)
  let area := (1 / 2 : ℝ) * 8 * 6
  have h_dist : distance = 10 := sorry
  have h_area : area = 24 := sorry
  exact ⟨h_dist, h_area⟩

end NUMINAMATH_GPT_distance_and_area_of_triangle_l12_1294


namespace NUMINAMATH_GPT_subcommittee_count_l12_1219

theorem subcommittee_count :
  let total_members := 12
  let total_teachers := 5
  let subcommittee_size := 5
  let total_subcommittees := Nat.choose total_members subcommittee_size
  let non_teacher_subcommittees_with_0_teachers := Nat.choose (total_members - total_teachers) subcommittee_size
  let non_teacher_subcommittees_with_1_teacher :=
    Nat.choose total_teachers 1 * Nat.choose (total_members - total_teachers) (subcommittee_size - 1)
  (total_subcommittees
   - (non_teacher_subcommittees_with_0_teachers + non_teacher_subcommittees_with_1_teacher)) = 596 := 
by
  sorry

end NUMINAMATH_GPT_subcommittee_count_l12_1219


namespace NUMINAMATH_GPT_number_under_35_sampled_l12_1244

-- Define the conditions
def total_employees : ℕ := 500
def employees_under_35 : ℕ := 125
def employees_35_to_49 : ℕ := 280
def employees_over_50 : ℕ := 95
def sample_size : ℕ := 100

-- Define the theorem stating the desired result
theorem number_under_35_sampled : (employees_under_35 * sample_size / total_employees) = 25 :=
by
  sorry

end NUMINAMATH_GPT_number_under_35_sampled_l12_1244


namespace NUMINAMATH_GPT_inverse_square_variation_l12_1200

variable (x y : ℝ)

theorem inverse_square_variation (h1 : x = 1) (h2 : y = 3) (h3 : y = 2) : x = 2.25 :=
by
  sorry

end NUMINAMATH_GPT_inverse_square_variation_l12_1200


namespace NUMINAMATH_GPT_find_c_l12_1209

-- Define c and the floor function
def c : ℝ := 13.1

theorem find_c (h : c + ⌊c⌋ = 25.6) : c = 13.1 :=
sorry

end NUMINAMATH_GPT_find_c_l12_1209


namespace NUMINAMATH_GPT_composite_prime_fraction_l12_1262

theorem composite_prime_fraction :
  let P1 : ℕ := 4 * 6 * 8 * 9 * 10 * 12 * 14 * 15
  let P2 : ℕ := 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26
  let first_prime : ℕ := 2
  let second_prime : ℕ := 3
  (P1 + first_prime) / (P2 + second_prime) =
    (4 * 6 * 8 * 9 * 10 * 12 * 14 * 15 + 2) / (16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 + 3) := by
  sorry

end NUMINAMATH_GPT_composite_prime_fraction_l12_1262


namespace NUMINAMATH_GPT_inequality_hold_l12_1289

theorem inequality_hold {a b : ℝ} (h : a < b) : -3 * a > -3 * b :=
sorry

end NUMINAMATH_GPT_inequality_hold_l12_1289


namespace NUMINAMATH_GPT_eval_expression_l12_1291

theorem eval_expression (a b : ℤ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := 
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l12_1291


namespace NUMINAMATH_GPT_full_price_ticket_revenue_correct_l12_1247

-- Define the constants and assumptions
variables (f t : ℕ) (p : ℝ)

-- Total number of tickets sold
def total_tickets := (f + t = 180)

-- Total revenue from ticket sales
def total_revenue := (f * p + t * (p / 3) = 2600)

-- Full price ticket revenue
def full_price_revenue := (f * p = 975)

-- The theorem combines the above conditions to prove the correct revenue from full-price tickets
theorem full_price_ticket_revenue_correct :
  total_tickets f t →
  total_revenue f t p →
  full_price_revenue f p :=
by
  sorry

end NUMINAMATH_GPT_full_price_ticket_revenue_correct_l12_1247


namespace NUMINAMATH_GPT_incorrect_value_l12_1237

theorem incorrect_value:
  ∀ (n : ℕ) (initial_mean corrected_mean : ℚ) (correct_value incorrect_value : ℚ),
  n = 50 →
  initial_mean = 36 →
  corrected_mean = 36.5 →
  correct_value = 48 →
  incorrect_value = correct_value - (corrected_mean * n - initial_mean * n) →
  incorrect_value = 23 :=
by
  intros n initial_mean corrected_mean correct_value incorrect_value
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_incorrect_value_l12_1237


namespace NUMINAMATH_GPT_hearty_buys_red_packages_l12_1282

-- Define the conditions
def packages_of_blue := 3
def beads_per_package := 40
def total_beads := 320

-- Calculate the number of blue beads
def blue_beads := packages_of_blue * beads_per_package

-- Calculate the number of red beads
def red_beads := total_beads - blue_beads

-- Prove that the number of red packages is 5
theorem hearty_buys_red_packages : (red_beads / beads_per_package) = 5 := by
  sorry

end NUMINAMATH_GPT_hearty_buys_red_packages_l12_1282


namespace NUMINAMATH_GPT_compute_r_l12_1211

variables {j p t m n x y r : ℝ}

theorem compute_r
    (h1 : j = 0.75 * p)
    (h2 : j = 0.80 * t)
    (h3 : t = p - r * p / 100)
    (h4 : m = 1.10 * p)
    (h5 : n = 0.70 * m)
    (h6 : j + p + t = m * n)
    (h7 : x = 1.15 * j)
    (h8 : y = 0.80 * n)
    (h9 : x * y = (j + p + t) ^ 2) : r = 6.25 := by
  sorry

end NUMINAMATH_GPT_compute_r_l12_1211


namespace NUMINAMATH_GPT_price_of_soda_l12_1261

theorem price_of_soda (regular_price_per_can : ℝ) (case_discount : ℝ) (bulk_discount : ℝ) (num_cases : ℕ) (num_cans : ℕ) :
  regular_price_per_can = 0.15 →
  case_discount = 0.12 →
  bulk_discount = 0.05 →
  num_cases = 3 →
  num_cans = 75 →
  (num_cans * ((regular_price_per_can * (1 - case_discount)) * (1 - bulk_discount))) = 9.405 :=
by
  intros h1 h2 h3 h4 h5
  -- normal price per can
  have hp1 : ℝ := regular_price_per_can
  -- price after case discount
  have hp2 : ℝ := hp1 * (1 - case_discount)
  -- price after bulk discount
  have hp3 : ℝ := hp2 * (1 - bulk_discount)
  -- total price
  have total_price : ℝ := num_cans * hp3
  -- goal
  sorry -- skip the proof, as only the statement is needed.

end NUMINAMATH_GPT_price_of_soda_l12_1261


namespace NUMINAMATH_GPT_percentage_is_60_l12_1204

-- Definitions based on the conditions
def fraction_value (x : ℕ) : ℕ := x / 3
def percentage_less_value (x p : ℕ) : ℕ := x - (p * x) / 100

-- Lean statement based on the mathematically equivalent proof problem
theorem percentage_is_60 : ∀ (x p : ℕ), x = 180 → fraction_value x = 60 → percentage_less_value 60 p = 24 → p = 60 :=
by
  intros x p H1 H2 H3
  -- Proof is not required, so we use sorry
  sorry

end NUMINAMATH_GPT_percentage_is_60_l12_1204


namespace NUMINAMATH_GPT_teacher_total_score_l12_1290

variable (written_score : ℕ)
variable (interview_score : ℕ)
variable (weight_written : ℝ)
variable (weight_interview : ℝ)

theorem teacher_total_score :
  (written_score = 80) → (interview_score = 60) → (weight_written = 0.6) → (weight_interview = 0.4) →
  (written_score * weight_written + interview_score * weight_interview = 72) :=
by
  sorry

end NUMINAMATH_GPT_teacher_total_score_l12_1290


namespace NUMINAMATH_GPT_carla_glasses_lemonade_l12_1203

theorem carla_glasses_lemonade (time_total : ℕ) (rate : ℕ) (glasses : ℕ) 
  (h1 : time_total = 3 * 60 + 40) 
  (h2 : rate = 20) 
  (h3 : glasses = time_total / rate) : 
  glasses = 11 := 
by 
  -- We'll fill in the proof here in a real scenario
  sorry

end NUMINAMATH_GPT_carla_glasses_lemonade_l12_1203


namespace NUMINAMATH_GPT_moving_circle_trajectory_l12_1258

theorem moving_circle_trajectory (x y : ℝ) 
  (fixed_circle : x^2 + y^2 = 4): 
  (x^2 + y^2 = 9) ∨ (x^2 + y^2 = 1) :=
sorry

end NUMINAMATH_GPT_moving_circle_trajectory_l12_1258


namespace NUMINAMATH_GPT_min_shaded_triangles_l12_1227

-- Definitions (conditions) directly from the problem
def Triangle (n : ℕ) := { x : ℕ // x ≤ n }
def side_length := 8
def smaller_side_length := 1

-- Goal (question == correct answer)
theorem min_shaded_triangles : ∃ (shaded : ℕ), shaded = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_shaded_triangles_l12_1227


namespace NUMINAMATH_GPT_problem_statement_l12_1279

noncomputable def proposition_p (x : ℝ) : Prop := ∃ x0 : ℝ, x0 - 2 > 0
noncomputable def proposition_q (x : ℝ) : Prop := ∀ x : ℝ, (2:ℝ)^x > x^2

theorem problem_statement : ∃ (p q : Prop), (∃ x0 : ℝ, x0 - 2 > 0) ∧ (¬ (∀ x : ℝ, (2:ℝ)^x > x^2)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l12_1279


namespace NUMINAMATH_GPT_sin_405_eq_sqrt2_div_2_l12_1293

theorem sin_405_eq_sqrt2_div_2 :
  Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_405_eq_sqrt2_div_2_l12_1293


namespace NUMINAMATH_GPT_compute_expression_l12_1253

theorem compute_expression : 45 * 1313 - 10 * 1313 = 45955 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l12_1253


namespace NUMINAMATH_GPT_average_class_is_45_6_l12_1213

noncomputable def average_class_score (total_students : ℕ) (top_scorers : ℕ) (top_score : ℕ) 
  (zero_scorers : ℕ) (remaining_students_avg : ℕ) : ℚ :=
  let total_top_score := top_scorers * top_score
  let total_zero_score := zero_scorers * 0
  let remaining_students := total_students - top_scorers - zero_scorers
  let total_remaining_score := remaining_students * remaining_students_avg
  let total_score := total_top_score + total_zero_score + total_remaining_score
  total_score / total_students

theorem average_class_is_45_6 : average_class_score 25 3 95 3 45 = 45.6 := 
by
  -- sorry is used here to skip the proof. Lean will expect a proof here.
  sorry

end NUMINAMATH_GPT_average_class_is_45_6_l12_1213


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l12_1288

-- Definition for the expression (2x - 3y)²
def expr1 (x y : ℝ) : ℝ := (2 * x - 3 * y) ^ 2

-- Theorem to prove that (2x - 3y)² = 4x² - 12xy + 9y²
theorem simplify_expr1 (x y : ℝ) : expr1 x y = 4 * (x ^ 2) - 12 * x * y + 9 * (y ^ 2) := 
sorry

-- Definition for the expression (x + y) * (x + y) * (x² + y²)
def expr2 (x y : ℝ) : ℝ := (x + y) * (x + y) * (x ^ 2 + y ^ 2)

-- Theorem to prove that (x + y) * (x + y) * (x² + y²) = x⁴ + 2x²y² + y⁴ + 2x³y + 2xy³
theorem simplify_expr2 (x y : ℝ) : expr2 x y = x ^ 4 + 2 * (x ^ 2) * (y ^ 2) + y ^ 4 + 2 * (x ^ 3) * y + 2 * x * (y ^ 3) := 
sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l12_1288


namespace NUMINAMATH_GPT_lesser_fraction_l12_1297

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 13 / 14) (h2 : x * y = 1 / 8) : min x y = 163 / 625 :=
by sorry

end NUMINAMATH_GPT_lesser_fraction_l12_1297


namespace NUMINAMATH_GPT_op_15_5_eq_33_l12_1240

def op (x y : ℕ) : ℕ :=
  2 * x + x / y

theorem op_15_5_eq_33 : op 15 5 = 33 := by
  sorry

end NUMINAMATH_GPT_op_15_5_eq_33_l12_1240


namespace NUMINAMATH_GPT_solve_for_a_l12_1212

theorem solve_for_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 16 - 6 * a + a ^ 2) : 
  a = -5 + Real.sqrt 41 ∨ a = -5 - Real.sqrt 41 := by
  sorry

end NUMINAMATH_GPT_solve_for_a_l12_1212


namespace NUMINAMATH_GPT_tensor_example_l12_1272
-- Import the necessary library

-- Define the binary operation ⊗
def tensor (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the main theorem
theorem tensor_example : tensor (tensor 8 6) 2 = 9 / 5 := by
  sorry

end NUMINAMATH_GPT_tensor_example_l12_1272


namespace NUMINAMATH_GPT_point_below_line_l12_1257

theorem point_below_line (a : ℝ) (h : 2 * a - 3 > 3) : a > 3 :=
sorry

end NUMINAMATH_GPT_point_below_line_l12_1257


namespace NUMINAMATH_GPT_minimum_g_a_l12_1265

noncomputable def f (x a : ℝ) : ℝ := x ^ 2 + 2 * a * x + 3

noncomputable def g (a : ℝ) : ℝ := 3 * a ^ 2 + 2 * a

theorem minimum_g_a : ∀ a : ℝ, a ≤ -1 → g a = 3 * a ^ 2 + 2 * a → g a ≥ 1 := by
  sorry

end NUMINAMATH_GPT_minimum_g_a_l12_1265


namespace NUMINAMATH_GPT_smallest_nine_consecutive_sum_l12_1235

theorem smallest_nine_consecutive_sum (n : ℕ) (h : (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) + (n+8) = 2007)) : n = 219 :=
sorry

end NUMINAMATH_GPT_smallest_nine_consecutive_sum_l12_1235


namespace NUMINAMATH_GPT_curve_focus_x_axis_l12_1281

theorem curve_focus_x_axis : 
    (x^2 - y^2 = 1)
    ∨ (∃ a b : ℝ, a ≠ 0 ∧ a ≠ b ∧ 
        (a*x^2 + b*y^2 = 1 → False)
    )
    ∨ (∃ a b : ℝ, a ≠ 0 ∧ a ≠ b ∧ 
        (b*y^2 - a*x^2 = 1 → False)
    )
    ∨ (∃ c : ℝ, c ≠ 0 ∧ 
        (y = c*x^2 → False)
    ) :=
sorry

end NUMINAMATH_GPT_curve_focus_x_axis_l12_1281


namespace NUMINAMATH_GPT_odd_three_mn_l12_1298

theorem odd_three_mn (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) : (3 * m * n) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_odd_three_mn_l12_1298


namespace NUMINAMATH_GPT_solution_exists_l12_1231

def valid_grid (grid : List (List Nat)) : Prop :=
  grid = [[2, 3, 6], [6, 3, 2]] ∨
  grid = [[2, 4, 8], [8, 4, 2]]

theorem solution_exists :
  ∃ (grid : List (List Nat)), valid_grid grid := by
  sorry

end NUMINAMATH_GPT_solution_exists_l12_1231


namespace NUMINAMATH_GPT_fourth_vertex_of_parallelogram_l12_1273

structure Point where
  x : ℤ
  y : ℤ

def midPoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def isMidpoint (M P Q : Point) : Prop :=
  M = midPoint P Q

theorem fourth_vertex_of_parallelogram (A B C D : Point)
  (hA : A = {x := -2, y := 1})
  (hB : B = {x := -1, y := 3})
  (hC : C = {x := 3, y := 4})
  (h1 : isMidpoint (midPoint A C) B D ∨
        isMidpoint (midPoint A B) C D ∨
        isMidpoint (midPoint B C) A D) :
  D = {x := 2, y := 2} ∨ D = {x := -6, y := 0} ∨ D = {x := 4, y := 6} := by
  sorry

end NUMINAMATH_GPT_fourth_vertex_of_parallelogram_l12_1273


namespace NUMINAMATH_GPT_last_four_digits_of_m_smallest_l12_1277

theorem last_four_digits_of_m_smallest (m : ℕ) (h1 : m > 0)
  (h2 : m % 6 = 0) (h3 : m % 8 = 0)
  (h4 : ∀ d, d ∈ (m.digits 10) → d = 2 ∨ d = 7)
  (h5 : 2 ∈ (m.digits 10)) (h6 : 7 ∈ (m.digits 10)) :
  (m % 10000) = 2722 :=
sorry

end NUMINAMATH_GPT_last_four_digits_of_m_smallest_l12_1277


namespace NUMINAMATH_GPT_triangle_ABC_AC_l12_1266

-- Defining the relevant points and lengths in the triangle
variables {A B C D : Type} 
variables (AB CD : ℝ)
variables (AD BC AC : ℝ)

-- Given constants
axiom hAB : AB = 3
axiom hCD : CD = Real.sqrt 3
axiom hAD_BC : AD = BC

-- The final theorem statement that needs to be proved
theorem triangle_ABC_AC :
  (AD = BC) ∧ (CD = Real.sqrt 3) ∧ (AB = 3) → AC = Real.sqrt 7 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_triangle_ABC_AC_l12_1266


namespace NUMINAMATH_GPT_matrix_power_eq_l12_1216

def MatrixC : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-8, -10]]

def MatrixA : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![201, 200], ![-400, -449]]

theorem matrix_power_eq :
  MatrixC ^ 50 = MatrixA := 
  sorry

end NUMINAMATH_GPT_matrix_power_eq_l12_1216


namespace NUMINAMATH_GPT_first_player_wins_l12_1207

noncomputable def game_win_guarantee : Prop :=
  ∃ (first_can_guarantee_win : Bool),
    first_can_guarantee_win = true

theorem first_player_wins :
  ∀ (nuts : ℕ) (players : (ℕ × ℕ)) (move : ℕ → ℕ) (end_condition : ℕ → Prop),
    nuts = 10 →
    players = (1, 2) →
    (∀ n, 0 < n ∧ n ≤ nuts → move n = n - 1) →
    (end_condition 3 = true) →
    (∀ x y z, x + y + z = 3 ↔ end_condition (x + y + z)) → 
    game_win_guarantee :=
by
  intros nuts players move end_condition H1 H2 H3 H4 H5
  sorry

end NUMINAMATH_GPT_first_player_wins_l12_1207


namespace NUMINAMATH_GPT_simplify_and_evaluate_l12_1229

theorem simplify_and_evaluate (x : ℝ) (h : x = 3 / 2) : 
  (2 + x) * (2 - x) + (x - 1) * (x + 5) = 5 := 
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l12_1229


namespace NUMINAMATH_GPT_greatest_divisible_by_13_l12_1280

theorem greatest_divisible_by_13 (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9)
  (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) : (10000 * A + 1000 * B + 100 * C + 10 * B + A = 96769) 
  ↔ (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 13 = 0 :=
sorry

end NUMINAMATH_GPT_greatest_divisible_by_13_l12_1280


namespace NUMINAMATH_GPT_lcm_140_225_is_6300_l12_1252

def lcm_140_225 : ℕ := Nat.lcm 140 225

theorem lcm_140_225_is_6300 : lcm_140_225 = 6300 :=
by
  sorry

end NUMINAMATH_GPT_lcm_140_225_is_6300_l12_1252


namespace NUMINAMATH_GPT_calculation_result_l12_1224

theorem calculation_result : (1955 - 1875)^2 / 64 = 100 := by
  sorry

end NUMINAMATH_GPT_calculation_result_l12_1224


namespace NUMINAMATH_GPT_inequality_proof_l12_1287

variable (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_ab_bc_ca : a * b + b * c + c * a = 1)

theorem inequality_proof :
  (3 / Real.sqrt (a^2 + 1)) + (4 / Real.sqrt (b^2 + 1)) + (12 / Real.sqrt (c^2 + 1)) < (39 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l12_1287


namespace NUMINAMATH_GPT_total_earnings_correct_l12_1278

-- Define the weekly earnings and the duration of the harvest.
def weekly_earnings : ℕ := 16
def harvest_duration : ℕ := 76

-- Theorems to state the problem requiring a proof.
theorem total_earnings_correct : (weekly_earnings * harvest_duration = 1216) := 
by
  sorry -- Proof is not required.

end NUMINAMATH_GPT_total_earnings_correct_l12_1278


namespace NUMINAMATH_GPT_complementary_angle_l12_1223

-- Define the complementary angle condition
def complement (angle : ℚ) := 90 - angle

theorem complementary_angle : complement 30.467 = 59.533 :=
by
  -- Adding sorry to signify the missing proof to ensure Lean builds successfully
  sorry

end NUMINAMATH_GPT_complementary_angle_l12_1223


namespace NUMINAMATH_GPT_polynomial_divisibility_l12_1232

theorem polynomial_divisibility (C D : ℝ) (h : ∀ (ω : ℂ), ω^2 + ω + 1 = 0 → (ω^106 + C * ω + D = 0)) : C + D = -1 :=
by
  -- Add proof here
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l12_1232


namespace NUMINAMATH_GPT_problem_solution_exists_l12_1242

theorem problem_solution_exists (a b n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)]
  (h : a > 0 ∧ b > 0 ∧ n > 0 ∧ a ^ 2013 + b ^ 2013 = p ^ n) :
  ∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ n = 2013 * k + 1 ∧ p = 2 := by
  sorry

end NUMINAMATH_GPT_problem_solution_exists_l12_1242


namespace NUMINAMATH_GPT_find_weight_per_square_inch_l12_1234

-- Define the TV dimensions and other given data
def bill_tv_width : ℕ := 48
def bill_tv_height : ℕ := 100
def bob_tv_width : ℕ := 70
def bob_tv_height : ℕ := 60
def weight_difference_pounds : ℕ := 150
def ounces_per_pound : ℕ := 16

-- Compute areas
def bill_tv_area := bill_tv_width * bill_tv_height
def bob_tv_area := bob_tv_width * bob_tv_height

-- Assume weight per square inch
def weight_per_square_inch : ℕ := 4

-- Total weight computation given in ounces
def bill_tv_weight := bill_tv_area * weight_per_square_inch
def bob_tv_weight := bob_tv_area * weight_per_square_inch
def weight_difference_ounces := weight_difference_pounds * ounces_per_pound

-- The theorem to prove
theorem find_weight_per_square_inch : 
  bill_tv_weight - bob_tv_weight = weight_difference_ounces → weight_per_square_inch = 4 :=
by
  intros
  /- Proof by computation -/
  sorry

end NUMINAMATH_GPT_find_weight_per_square_inch_l12_1234


namespace NUMINAMATH_GPT_probability_A_not_lose_l12_1267

-- Define the probabilities
def P_A_wins : ℝ := 0.30
def P_draw : ℝ := 0.25
def P_A_not_lose : ℝ := 0.55

-- Statement to prove
theorem probability_A_not_lose : P_A_wins + P_draw = P_A_not_lose :=
by 
  sorry

end NUMINAMATH_GPT_probability_A_not_lose_l12_1267


namespace NUMINAMATH_GPT_factorize_polynomial_l12_1210

theorem factorize_polynomial :
  ∀ (x : ℝ), x^4 + 2021 * x^2 + 2020 * x + 2021 = (x^2 + x + 1) * (x^2 - x + 2021) :=
by
  intros x
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l12_1210


namespace NUMINAMATH_GPT_new_bag_marbles_l12_1283

open Nat

theorem new_bag_marbles 
  (start_marbles : ℕ)
  (lost_marbles : ℕ)
  (given_marbles : ℕ)
  (received_back_marbles : ℕ)
  (end_marbles : ℕ)
  (h_start : start_marbles = 40)
  (h_lost : lost_marbles = 3)
  (h_given : given_marbles = 5)
  (h_received_back : received_back_marbles = 2 * given_marbles)
  (h_end : end_marbles = 54) :
  (end_marbles = (start_marbles - lost_marbles - given_marbles + received_back_marbles + new_bag) ∧ new_bag = 12) :=
by
  sorry

end NUMINAMATH_GPT_new_bag_marbles_l12_1283


namespace NUMINAMATH_GPT_interval_of_defined_expression_l12_1299

theorem interval_of_defined_expression (x : ℝ) :
  (x > 2 ∧ x < 5) ↔ (x - 2 > 0 ∧ 5 - x > 0) :=
by
  sorry

end NUMINAMATH_GPT_interval_of_defined_expression_l12_1299


namespace NUMINAMATH_GPT_marty_combinations_l12_1295

theorem marty_combinations : 
  ∃ n : ℕ, n = 5 * 4 ∧ n = 20 :=
by
  sorry

end NUMINAMATH_GPT_marty_combinations_l12_1295


namespace NUMINAMATH_GPT_second_train_speed_l12_1270

noncomputable def speed_of_second_train (length1 length2 speed1 clearance_time : ℝ) : ℝ :=
  let total_distance := (length1 + length2) / 1000 -- convert meters to kilometers
  let time_in_hours := clearance_time / 3600 -- convert seconds to hours
  let relative_speed := total_distance / time_in_hours
  relative_speed - speed1

theorem second_train_speed : 
  speed_of_second_train 60 280 42 16.998640108791296 = 30.05 := 
by
  sorry

end NUMINAMATH_GPT_second_train_speed_l12_1270


namespace NUMINAMATH_GPT_add_in_base14_l12_1274

-- Define symbols A, B, C, D in base 10 as they are used in the base 14 representation
def base14_A : ℕ := 10
def base14_B : ℕ := 11
def base14_C : ℕ := 12
def base14_D : ℕ := 13

-- Define the numbers given in base 14
def num1_base14 : ℕ := 9 * 14^2 + base14_C * 14 + 7
def num2_base14 : ℕ := 4 * 14^2 + base14_B * 14 + 3

-- Define the expected result in base 14
def result_base14 : ℕ := 1 * 14^2 + 0 * 14 + base14_A

-- The theorem statement that needs to be proven
theorem add_in_base14 : num1_base14 + num2_base14 = result_base14 := by
  sorry

end NUMINAMATH_GPT_add_in_base14_l12_1274
