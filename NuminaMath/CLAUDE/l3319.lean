import Mathlib

namespace NUMINAMATH_CALUDE_abs_neg_2023_l3319_331925

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l3319_331925


namespace NUMINAMATH_CALUDE_total_pools_calculation_l3319_331927

/-- The number of Pat's Pool Supply stores -/
def pool_supply_stores : ℕ := 4

/-- The number of Pat's Ark & Athletic Wear stores -/
def ark_athletic_stores : ℕ := 6

/-- The ratio of pools between Pat's Pool Supply and Pat's Ark & Athletic Wear stores -/
def pool_ratio : ℕ := 5

/-- The initial number of pools at one Pat's Ark & Athletic Wear store -/
def initial_pools : ℕ := 200

/-- The number of pools sold at one Pat's Ark & Athletic Wear store -/
def pools_sold : ℕ := 8

/-- The number of pools returned to one Pat's Ark & Athletic Wear store -/
def pools_returned : ℕ := 3

/-- The total number of swimming pools across all stores -/
def total_pools : ℕ := 5070

theorem total_pools_calculation :
  let current_pools := initial_pools - pools_sold + pools_returned
  let supply_store_pools := pool_ratio * current_pools
  total_pools = ark_athletic_stores * current_pools + pool_supply_stores * supply_store_pools := by
  sorry

end NUMINAMATH_CALUDE_total_pools_calculation_l3319_331927


namespace NUMINAMATH_CALUDE_floor_length_percentage_l3319_331979

theorem floor_length_percentage (length width area : ℝ) : 
  length = 23 ∧ 
  area = 529 / 3 ∧ 
  area = length * width → 
  length = width * 3 :=
by sorry

end NUMINAMATH_CALUDE_floor_length_percentage_l3319_331979


namespace NUMINAMATH_CALUDE_fourth_power_nested_root_l3319_331987

theorem fourth_power_nested_root : 
  (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^4 = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_nested_root_l3319_331987


namespace NUMINAMATH_CALUDE_f_continuous_iff_a_eq_5_l3319_331926

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then x^2 + 2 else 2*x + a

-- State the theorem
theorem f_continuous_iff_a_eq_5 (a : ℝ) :
  Continuous (f a) ↔ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_continuous_iff_a_eq_5_l3319_331926


namespace NUMINAMATH_CALUDE_percentage_caught_sampling_candy_l3319_331953

/-- The percentage of customers caught sampling candy -/
def percentage_caught (total_sample_percent : ℝ) (not_caught_ratio : ℝ) : ℝ :=
  total_sample_percent - (not_caught_ratio * total_sample_percent)

/-- Theorem stating the percentage of customers caught sampling candy -/
theorem percentage_caught_sampling_candy :
  let total_sample_percent : ℝ := 23.913043478260867
  let not_caught_ratio : ℝ := 0.08
  percentage_caught total_sample_percent not_caught_ratio = 22 := by
  sorry


end NUMINAMATH_CALUDE_percentage_caught_sampling_candy_l3319_331953


namespace NUMINAMATH_CALUDE_complex_calculation_l3319_331910

theorem complex_calculation : (2 - I) / (1 - I) - I = 3/2 - 1/2 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l3319_331910


namespace NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l3319_331963

/-- Proves that the first interest rate is 10% given the problem conditions -/
theorem first_interest_rate_is_ten_percent
  (total_amount : ℕ)
  (first_part : ℕ)
  (second_part : ℕ)
  (second_rate : ℚ)
  (total_profit : ℕ)
  (h1 : total_amount = 70000)
  (h2 : first_part = 60000)
  (h3 : second_part = 10000)
  (h4 : total_amount = first_part + second_part)
  (h5 : second_rate = 20 / 100)
  (h6 : total_profit = 8000)
  (h7 : total_profit = first_part * (first_rate / 100) + second_part * (second_rate / 100)) :
  first_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l3319_331963


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l3319_331957

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l3319_331957


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l3319_331940

theorem ice_cream_combinations :
  (5 : ℕ) * (Nat.choose 7 3) = 175 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l3319_331940


namespace NUMINAMATH_CALUDE_power_function_through_point_l3319_331981

theorem power_function_through_point (a : ℝ) : 
  (2 : ℝ) ^ a = (1 / 2 : ℝ) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3319_331981


namespace NUMINAMATH_CALUDE_f_increasing_l3319_331922

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x - Real.sin x else x^3 + 1

theorem f_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_increasing_l3319_331922


namespace NUMINAMATH_CALUDE_keystone_arch_angle_l3319_331993

/-- Represents a keystone arch composed of congruent isosceles trapezoids -/
structure KeystoneArch where
  num_trapezoids : ℕ
  trapezoids_congruent : Bool
  trapezoids_isosceles : Bool
  bottom_sides_horizontal : Bool

/-- Calculates the smaller interior angle of a trapezoid in a keystone arch -/
def smaller_interior_angle (arch : KeystoneArch) : ℝ :=
  if arch.num_trapezoids = 8 ∧ 
     arch.trapezoids_congruent ∧ 
     arch.trapezoids_isosceles ∧ 
     arch.bottom_sides_horizontal
  then 78.75
  else 0

/-- Theorem stating that the smaller interior angle of each trapezoid in the specified keystone arch is 78.75° -/
theorem keystone_arch_angle (arch : KeystoneArch) :
  arch.num_trapezoids = 8 ∧ 
  arch.trapezoids_congruent ∧ 
  arch.trapezoids_isosceles ∧ 
  arch.bottom_sides_horizontal →
  smaller_interior_angle arch = 78.75 := by
  sorry

end NUMINAMATH_CALUDE_keystone_arch_angle_l3319_331993


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3319_331999

def P (n : Nat) : Nat :=
  (n / 10) * (n % 10)

def S (n : Nat) : Nat :=
  (n / 10) + (n % 10)

theorem two_digit_number_property : ∃! N : Nat, 
  10 ≤ N ∧ N < 100 ∧ N = P N + 2 * S N :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3319_331999


namespace NUMINAMATH_CALUDE_expand_expression_l3319_331913

theorem expand_expression (x : ℝ) : (7*x + 11 - 3) * 4*x = 28*x^2 + 32*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3319_331913


namespace NUMINAMATH_CALUDE_rubber_band_difference_l3319_331975

theorem rubber_band_difference (justine bailey ylona : ℕ) : 
  ylona = 24 →
  justine = ylona - 2 →
  bailey + 4 = 8 →
  justine > bailey →
  justine - bailey = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rubber_band_difference_l3319_331975


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l3319_331977

/-- The equation of a reflected ray given an incident ray and a reflecting line -/
theorem reflected_ray_equation 
  (incident_ray : ℝ → ℝ → Prop) 
  (reflecting_line : ℝ → ℝ → Prop) 
  (reflected_ray : ℝ → ℝ → Prop) : 
  (∀ x y, incident_ray x y ↔ x - 2*y + 3 = 0) →
  (∀ x y, reflecting_line x y ↔ y = x) →
  (∀ x y, reflected_ray x y ↔ 2*x - y - 3 = 0) := by
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l3319_331977


namespace NUMINAMATH_CALUDE_fixed_charge_is_28_l3319_331936

-- Define the variables
def fixed_charge : ℝ := sorry
def january_call_charge : ℝ := sorry
def february_call_charge : ℝ := sorry

-- Define the conditions
axiom january_bill : fixed_charge + january_call_charge = 52
axiom february_bill : fixed_charge + february_call_charge = 76
axiom february_double_january : february_call_charge = 2 * january_call_charge

-- Theorem to prove
theorem fixed_charge_is_28 : fixed_charge = 28 := by sorry

end NUMINAMATH_CALUDE_fixed_charge_is_28_l3319_331936


namespace NUMINAMATH_CALUDE_zhang_fei_probabilities_l3319_331978

/-- The set of events Zhang Fei can participate in -/
inductive Event : Type
  | LongJump : Event
  | Meters100 : Event
  | Meters200 : Event
  | Meters400 : Event

/-- The probability of selecting an event -/
def selectProbability (e : Event) : ℚ :=
  1 / 4

/-- The probability of selecting two specific events when choosing at most two events -/
def selectTwoEventsProbability (e1 e2 : Event) : ℚ :=
  2 / 12

theorem zhang_fei_probabilities :
  (selectProbability Event.LongJump = 1 / 4) ∧
  (selectTwoEventsProbability Event.LongJump Event.Meters100 = 1 / 6) := by
  sorry

end NUMINAMATH_CALUDE_zhang_fei_probabilities_l3319_331978


namespace NUMINAMATH_CALUDE_xiao_ming_age_problem_l3319_331924

/-- Proves that Xiao Ming was 7 years old when his father's age was 5 times Xiao Ming's age -/
theorem xiao_ming_age_problem (current_age : ℕ) (father_current_age : ℕ) 
  (h1 : current_age = 12) (h2 : father_current_age = 40) : 
  ∃ (past_age : ℕ), past_age = 7 ∧ father_current_age - (current_age - past_age) = 5 * past_age :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_age_problem_l3319_331924


namespace NUMINAMATH_CALUDE_equation_solution_expression_result_l3319_331968

-- Problem 1
theorem equation_solution :
  ∃ y : ℝ, 4 * (y - 1) = 1 - 3 * (y - 3) ∧ y = 2 := by sorry

-- Problem 2
theorem expression_result :
  (-2)^3 / 4 + 6 * |1/3 - 1| - 1/2 * 14 = -5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_expression_result_l3319_331968


namespace NUMINAMATH_CALUDE_trajectory_of_moving_point_l3319_331950

/-- The trajectory of a point M(x, y) that is twice as far from A(-4, 0) as it is from B(2, 0) -/
theorem trajectory_of_moving_point (x y : ℝ) : 
  (((x + 4)^2 + y^2).sqrt = 2 * ((x - 2)^2 + y^2).sqrt) ↔ 
  (x^2 + y^2 - 8*x = 0) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_point_l3319_331950


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3319_331996

-- System 1
theorem system_one_solution (x y : ℚ) : 
  2 * x - y = 5 ∧ x - 1 = (2 * y - 1) / 2 → x = 9/2 ∧ y = 4 := by sorry

-- System 2
theorem system_two_solution (x y : ℚ) :
  3 * x + 2 * y = 1 ∧ 2 * x - 3 * y = 5 → x = 1 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3319_331996


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l3319_331967

theorem p_or_q_is_true : 
  let p : Prop := 2 + 3 = 5
  let q : Prop := 5 < 4
  p ∨ q := by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l3319_331967


namespace NUMINAMATH_CALUDE_power_value_from_condition_l3319_331942

theorem power_value_from_condition (x y : ℝ) : 
  |x - 3| + (y + 3)^2 = 0 → y^x = -27 := by sorry

end NUMINAMATH_CALUDE_power_value_from_condition_l3319_331942


namespace NUMINAMATH_CALUDE_expression_simplification_l3319_331966

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x + a)^2 / ((a - b) * (a - c)) + (x + b)^2 / ((b - a) * (b - c)) + (x + c)^2 / ((c - a) * (c - b)) =
  a * x + b * x + c * x - a - b - c :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3319_331966


namespace NUMINAMATH_CALUDE_max_p_plus_q_l3319_331928

theorem max_p_plus_q (p q : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → 2*p*x^2 + q*x - p + 1 ≥ 0) → 
  p + q ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_p_plus_q_l3319_331928


namespace NUMINAMATH_CALUDE_white_balls_count_l3319_331960

theorem white_balls_count (a : ℕ) : 
  (a : ℝ) / (a + 3) = 4/5 → a = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3319_331960


namespace NUMINAMATH_CALUDE_reading_time_calculation_l3319_331980

def total_homework_time : ℕ := 120
def math_time : ℕ := 25
def spelling_time : ℕ := 30
def history_time : ℕ := 20
def science_time : ℕ := 15

theorem reading_time_calculation :
  total_homework_time - (math_time + spelling_time + history_time + science_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l3319_331980


namespace NUMINAMATH_CALUDE_three_year_officer_pays_51_l3319_331939

/-- The price of duty shoes for an officer who has served at least three years -/
def price_for_three_year_officer : ℝ :=
  let full_price : ℝ := 85
  let first_year_discount : ℝ := 0.20
  let three_year_discount : ℝ := 0.25
  let discounted_price : ℝ := full_price * (1 - first_year_discount)
  discounted_price * (1 - three_year_discount)

/-- Theorem stating that an officer who has served at least three years pays $51 for duty shoes -/
theorem three_year_officer_pays_51 :
  price_for_three_year_officer = 51 := by
  sorry

end NUMINAMATH_CALUDE_three_year_officer_pays_51_l3319_331939


namespace NUMINAMATH_CALUDE_min_colors_correct_l3319_331958

/-- The number of distribution centers to be represented -/
def num_centers : ℕ := 12

/-- Calculates the number of unique representations possible with n colors -/
def num_representations (n : ℕ) : ℕ := n + n.choose 2

/-- Checks if a given number of colors is sufficient to represent all centers -/
def is_sufficient (n : ℕ) : Prop := num_representations n ≥ num_centers

/-- The minimum number of colors needed -/
def min_colors : ℕ := 5

/-- Theorem stating that min_colors is the minimum number of colors needed -/
theorem min_colors_correct :
  is_sufficient min_colors ∧ ∀ k < min_colors, ¬is_sufficient k :=
sorry

end NUMINAMATH_CALUDE_min_colors_correct_l3319_331958


namespace NUMINAMATH_CALUDE_intersection_range_l3319_331969

theorem intersection_range (k : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + 2 ∧ 
    y₂ = k * x₂ + 2 ∧ 
    x₁ = Real.sqrt (y₁^2 + 6) ∧ 
    x₂ = Real.sqrt (y₂^2 + 6)) →
  -Real.sqrt 15 / 3 < k ∧ k < -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_range_l3319_331969


namespace NUMINAMATH_CALUDE_remainder_problem_l3319_331918

theorem remainder_problem (a b : ℤ) 
  (ha : a % 98 = 92) 
  (hb : b % 147 = 135) : 
  (3 * a + b) % 49 = 19 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3319_331918


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l3319_331932

def T : Finset Nat := Finset.range 15

def disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (S : Finset Nat) (h : S = T) : 
  disjoint_subsets S % 500 = 186 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l3319_331932


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l3319_331904

theorem sqrt_3_irrational :
  ∀ (a b c : ℚ), (a = 1/2 ∧ b = 1/5 ∧ c = -5) →
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = p / q := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l3319_331904


namespace NUMINAMATH_CALUDE_sugar_price_proof_l3319_331906

/-- Proves that given the initial price of sugar as 6 Rs/kg, a new price of 7.50 Rs/kg, 
    and a reduction in consumption of 19.999999999999996%, the initial price of sugar is 6 Rs/kg. -/
theorem sugar_price_proof (initial_price : ℝ) (new_price : ℝ) (consumption_reduction : ℝ) : 
  initial_price = 6 ∧ new_price = 7.5 ∧ consumption_reduction = 19.999999999999996 → initial_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_sugar_price_proof_l3319_331906


namespace NUMINAMATH_CALUDE_cake_muffin_probability_l3319_331971

theorem cake_muffin_probability (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ)
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_both : both = 17) :
  (total - (cake + muffin - both)) / total = 27 / 100 := by
sorry

end NUMINAMATH_CALUDE_cake_muffin_probability_l3319_331971


namespace NUMINAMATH_CALUDE_williams_books_l3319_331900

theorem williams_books (w : ℕ) : 
  (3 * w + 8 + 4 = w + 2 * 8) → w = 2 := by
  sorry

end NUMINAMATH_CALUDE_williams_books_l3319_331900


namespace NUMINAMATH_CALUDE_bathroom_flooring_area_l3319_331911

/-- The total area of hardwood flooring installed in Nancy's bathroom -/
def total_area (central_length central_width hallway_length hallway_width : ℝ) : ℝ :=
  central_length * central_width + hallway_length * hallway_width

/-- Proof that the total area of hardwood flooring installed in Nancy's bathroom is 124 square feet -/
theorem bathroom_flooring_area :
  total_area 10 10 6 4 = 124 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_flooring_area_l3319_331911


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_30_l3319_331903

def complement (α : ℝ) : ℝ := 90 - α

def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_30 :
  supplement (complement 30) = 120 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_30_l3319_331903


namespace NUMINAMATH_CALUDE_inequality_proof_l3319_331902

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4*a/(b+c)) * (1 + 4*b/(c+a)) * (1 + 4*c/(a+b)) > 25 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3319_331902


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3319_331982

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.I * z = z + a * Complex.I) 
  (h2 : Complex.abs z = Real.sqrt 5) 
  (h3 : a > 0) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3319_331982


namespace NUMINAMATH_CALUDE_pens_in_pack_l3319_331921

/-- The number of pens in each pack -/
def pens_per_pack : ℕ := sorry

/-- The number of packs Kendra has -/
def kendra_packs : ℕ := 4

/-- The number of packs Tony has -/
def tony_packs : ℕ := 2

/-- The number of pens Kendra and Tony keep for themselves -/
def pens_kept : ℕ := 4

/-- The number of friends they give pens to -/
def friends : ℕ := 14

theorem pens_in_pack : 
  (kendra_packs + tony_packs) * pens_per_pack - pens_kept - friends = 0 ∧ 
  pens_per_pack = 3 := by sorry

end NUMINAMATH_CALUDE_pens_in_pack_l3319_331921


namespace NUMINAMATH_CALUDE_calculation_proofs_l3319_331988

theorem calculation_proofs (x y a b : ℝ) :
  (((1/2) * x * y)^2 * (6 * x^2 * y) = (3/2) * x^4 * y^3) ∧
  ((2*a + b)^2 = 4*a^2 + 4*a*b + b^2) := by sorry

end NUMINAMATH_CALUDE_calculation_proofs_l3319_331988


namespace NUMINAMATH_CALUDE_fiftieth_number_is_fourteen_l3319_331951

/-- Defines the cumulative sum of elements up to the nth row -/
def cumulativeSum (n : ℕ) : ℕ := 
  (List.range n).foldl (fun acc i => acc + 2 * (i + 1)) 0

/-- Defines the value of each element in the nth row -/
def rowValue (n : ℕ) : ℕ := 2 * n

theorem fiftieth_number_is_fourteen : 
  ∃ (n : ℕ), cumulativeSum n < 50 ∧ 50 ≤ cumulativeSum (n + 1) ∧ rowValue (n + 1) = 14 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_number_is_fourteen_l3319_331951


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3319_331984

theorem quadratic_inequality_solution_range (d : ℝ) : 
  (d > 0 ∧ ∃ x : ℝ, x^2 - 8*x + d < 0) ↔ 0 < d ∧ d < 16 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3319_331984


namespace NUMINAMATH_CALUDE_article_cost_l3319_331916

/-- 
Given an article that can be sold at two different prices, prove that the cost of the article
is 140 if the higher price yields a 5% greater gain than the lower price.
-/
theorem article_cost (selling_price_high selling_price_low : ℕ) 
  (h1 : selling_price_high = 350)
  (h2 : selling_price_low = 340)
  (h3 : selling_price_high - selling_price_low = 10) :
  ∃ (cost gain : ℕ),
    selling_price_low = cost + gain ∧
    selling_price_high = cost + gain + (gain * 5 / 100) ∧
    cost = 140 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l3319_331916


namespace NUMINAMATH_CALUDE_flow_rates_theorem_l3319_331941

/-- Represents an irrigation channel in the system -/
inductive Channel
| AB | BC | CD | DE | BG | GD | GF | FE

/-- Represents a node in the irrigation system -/
inductive Node
| A | B | C | D | E | F | G | H

/-- The flow rate in a channel -/
def flow_rate (c : Channel) : ℝ := sorry

/-- The total input flow rate -/
def q₀ : ℝ := sorry

/-- The irrigation system is symmetric -/
axiom symmetric_system : ∀ c₁ c₂ : Channel, flow_rate c₁ = flow_rate c₂

/-- The sum of flow rates remains constant along any path -/
axiom constant_flow_sum : ∀ path : List Channel, 
  (∀ c ∈ path, c ∈ [Channel.AB, Channel.BC, Channel.CD, Channel.DE, Channel.BG, Channel.GD, Channel.GF, Channel.FE]) →
  (List.sum (path.map flow_rate) = q₀)

/-- Theorem stating the flow rates in channels DE, BC, and GF -/
theorem flow_rates_theorem :
  flow_rate Channel.DE = (4/7) * q₀ ∧
  flow_rate Channel.BC = (2/7) * q₀ ∧
  flow_rate Channel.GF = (3/7) * q₀ := by
  sorry

end NUMINAMATH_CALUDE_flow_rates_theorem_l3319_331941


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l3319_331923

/-- A quadratic function f(x) = a + bx - x^2 satisfying certain conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a + b * x - x^2

/-- The theorem stating the range of m for the given conditions -/
theorem quadratic_function_m_range (a b m : ℝ) :
  (∀ x, f a b (1 + x) = f a b (1 - x)) →
  (∀ x ≤ 4, Monotone (fun x => f a b (x + m))) →
  m ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l3319_331923


namespace NUMINAMATH_CALUDE_laura_change_l3319_331935

/-- The change Laura received after purchasing pants and shirts -/
theorem laura_change (pants_cost shirt_cost : ℕ) (pants_quantity shirt_quantity : ℕ) (amount_given : ℕ) : 
  pants_cost = 54 → 
  pants_quantity = 2 → 
  shirt_cost = 33 → 
  shirt_quantity = 4 → 
  amount_given = 250 → 
  amount_given - (pants_cost * pants_quantity + shirt_cost * shirt_quantity) = 10 := by
  sorry

end NUMINAMATH_CALUDE_laura_change_l3319_331935


namespace NUMINAMATH_CALUDE_teacup_box_ratio_l3319_331992

theorem teacup_box_ratio : 
  let total_boxes : ℕ := 26
  let pan_boxes : ℕ := 6
  let cups_per_box : ℕ := 5 * 4
  let broken_cups_per_box : ℕ := 2
  let remaining_cups : ℕ := 180
  
  let non_pan_boxes : ℕ := total_boxes - pan_boxes
  let teacup_boxes : ℕ := remaining_cups / (cups_per_box - broken_cups_per_box)
  let decoration_boxes : ℕ := non_pan_boxes - teacup_boxes
  
  (decoration_boxes : ℚ) / total_boxes = 5 / 13 :=
by sorry

end NUMINAMATH_CALUDE_teacup_box_ratio_l3319_331992


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3319_331947

/-- Proves that the average salary of all workers is 8000 Rs given the specified conditions -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technician_count : ℕ)
  (technician_avg_salary : ℕ)
  (non_technician_avg_salary : ℕ)
  (h1 : total_workers = 14)
  (h2 : technician_count = 7)
  (h3 : technician_avg_salary = 10000)
  (h4 : non_technician_avg_salary = 6000) :
  (technician_count * technician_avg_salary + (total_workers - technician_count) * non_technician_avg_salary) / total_workers = 8000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3319_331947


namespace NUMINAMATH_CALUDE_third_day_sales_formula_l3319_331976

/-- Represents the sales of sportswear over three days -/
structure SportswearSales where
  /-- Sales on the first day -/
  first_day : ℕ
  /-- Parameter m used in calculations -/
  m : ℕ

/-- Calculates the sales on the second day -/
def second_day_sales (s : SportswearSales) : ℤ :=
  3 * s.first_day - 3 * s.m

/-- Calculates the sales on the third day -/
def third_day_sales (s : SportswearSales) : ℤ :=
  second_day_sales s + s.m

/-- Theorem stating that the third day sales equal 3a - 2m -/
theorem third_day_sales_formula (s : SportswearSales) :
  third_day_sales s = 3 * s.first_day - 2 * s.m :=
by
  sorry

end NUMINAMATH_CALUDE_third_day_sales_formula_l3319_331976


namespace NUMINAMATH_CALUDE_max_value_x_plus_1000y_l3319_331905

theorem max_value_x_plus_1000y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x + 2018 / y = 1000) (eq2 : 9 / x + y = 1) :
  ∃ (x' y' : ℝ), x' + 2018 / y' = 1000 ∧ 9 / x' + y' = 1 ∧
  ∀ (a b : ℝ), a + 2018 / b = 1000 → 9 / a + b = 1 → x' + 1000 * y' ≥ a + 1000 * b ∧
  x' + 1000 * y' = 1991 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_1000y_l3319_331905


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_divisible_by_five_l3319_331964

theorem count_four_digit_numbers_divisible_by_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_divisible_by_five_l3319_331964


namespace NUMINAMATH_CALUDE_apple_price_correct_l3319_331938

/-- The price of one apple in dollars -/
def apple_price : ℚ := 49/30

/-- The price of one orange in dollars -/
def orange_price : ℚ := 3/4

/-- The number of apples that equal the price of 2 watermelons or 3 pineapples -/
def apple_equiv : ℕ := 6

/-- The number of watermelons that equal the price of 6 apples or 3 pineapples -/
def watermelon_equiv : ℕ := 2

/-- The number of pineapples that equal the price of 6 apples or 2 watermelons -/
def pineapple_equiv : ℕ := 3

/-- The number of oranges bought -/
def oranges_bought : ℕ := 24

/-- The number of apples bought -/
def apples_bought : ℕ := 18

/-- The number of watermelons bought -/
def watermelons_bought : ℕ := 12

/-- The number of pineapples bought -/
def pineapples_bought : ℕ := 18

/-- The total bill in dollars -/
def total_bill : ℚ := 165

theorem apple_price_correct :
  apple_price * apple_equiv = apple_price * watermelon_equiv * 3 ∧
  apple_price * 2 * pineapple_equiv = apple_price * watermelon_equiv * 3 ∧
  orange_price * oranges_bought + apple_price * apples_bought + 
  (apple_price * 3) * watermelons_bought + (apple_price * 2) * pineapples_bought = total_bill :=
by sorry

end NUMINAMATH_CALUDE_apple_price_correct_l3319_331938


namespace NUMINAMATH_CALUDE_problem_solution_l3319_331934

def A (a : ℝ) : Set ℝ := {x | a - 2 < x ∧ x < a + 2}

def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a = 0}

theorem problem_solution :
  (∀ x, x ∈ (A 0 ∪ B 0) ↔ -2 < x ∧ x ≤ 2) ∧
  (∀ a, (Aᶜ a ∩ B a).Nonempty ↔ a ≤ 0 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3319_331934


namespace NUMINAMATH_CALUDE_smallest_n_for_g_greater_than_15_l3319_331929

def g (n : ℕ+) : ℕ := 
  (Nat.digits 10 ((10^n.val) / (7^n.val))).sum

theorem smallest_n_for_g_greater_than_15 : 
  ∀ k : ℕ+, k < 12 → g k ≤ 15 ∧ g 12 > 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_g_greater_than_15_l3319_331929


namespace NUMINAMATH_CALUDE_rational_with_smallest_abs_value_l3319_331974

theorem rational_with_smallest_abs_value : ∃ q : ℚ, |q| < |1| := by
  -- The proof would go here, but we're only writing the statement
  sorry

end NUMINAMATH_CALUDE_rational_with_smallest_abs_value_l3319_331974


namespace NUMINAMATH_CALUDE_negative_three_squared_l3319_331933

theorem negative_three_squared : (-3 : ℤ)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_l3319_331933


namespace NUMINAMATH_CALUDE_ratio_equality_l3319_331965

theorem ratio_equality (x y z : ℝ) 
  (h1 : x * y * z ≠ 0) 
  (h2 : 2 * x * y = 3 * y * z) 
  (h3 : 3 * y * z = 5 * x * z) : 
  (x + 3 * y - 3 * z) / (x + 3 * y - 6 * z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3319_331965


namespace NUMINAMATH_CALUDE_sum_product_difference_l3319_331907

theorem sum_product_difference (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 126) : 
  |x - y| = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_product_difference_l3319_331907


namespace NUMINAMATH_CALUDE_compare_with_negative_three_l3319_331990

theorem compare_with_negative_three : 
  let numbers : List ℝ := [-3, 2, 0, -4]
  numbers.filter (λ x => x < -3) = [-4] := by sorry

end NUMINAMATH_CALUDE_compare_with_negative_three_l3319_331990


namespace NUMINAMATH_CALUDE_planes_parallel_implies_lines_parallel_l3319_331943

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_implies_lines_parallel
  (α β γ : Plane) (m n : Line)
  (h1 : α ≠ β ∧ α ≠ γ ∧ β ≠ γ)
  (h2 : intersect α γ = m)
  (h3 : intersect β γ = n)
  (h4 : parallel_planes α β) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_implies_lines_parallel_l3319_331943


namespace NUMINAMATH_CALUDE_frequency_of_eighth_group_l3319_331915

theorem frequency_of_eighth_group 
  (num_rectangles : ℕ) 
  (sample_size : ℕ) 
  (area_last_rectangle : ℝ) 
  (sum_area_other_rectangles : ℝ) :
  num_rectangles = 8 →
  sample_size = 200 →
  area_last_rectangle = (1/4 : ℝ) * sum_area_other_rectangles →
  (area_last_rectangle / (area_last_rectangle + sum_area_other_rectangles)) * sample_size = 40 :=
by sorry

end NUMINAMATH_CALUDE_frequency_of_eighth_group_l3319_331915


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_2_over_sqrt_2_l3319_331955

theorem sqrt_18_minus_sqrt_2_over_sqrt_2 : (Real.sqrt 18 - Real.sqrt 2) / Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_2_over_sqrt_2_l3319_331955


namespace NUMINAMATH_CALUDE_rational_sqrt_equation_zero_l3319_331914

theorem rational_sqrt_equation_zero (a b c : ℚ) 
  (h : a + b * Real.sqrt 32 + c * Real.sqrt 34 = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_equation_zero_l3319_331914


namespace NUMINAMATH_CALUDE_product_equals_eight_l3319_331983

theorem product_equals_eight : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eight_l3319_331983


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_plus_1_min_value_f_no_positive_a_b_l3319_331985

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem 1: Solution set of f(x) ≤ x + 1
theorem solution_set_f_leq_x_plus_1 :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 2/3 ≤ x ∧ x ≤ 4} :=
sorry

-- Theorem 2: Minimum value of f(x)
theorem min_value_f :
  ∃ k : ℝ, k = 1 ∧ ∀ x : ℝ, f x ≥ k :=
sorry

-- Theorem 3: Non-existence of positive a, b satisfying conditions
theorem no_positive_a_b :
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + b = 1 ∧ 1/a + 2/b = 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_plus_1_min_value_f_no_positive_a_b_l3319_331985


namespace NUMINAMATH_CALUDE_probability_of_valid_triangle_l3319_331945

-- Define a regular 15-gon
def regular_15gon : Set (ℝ × ℝ) := sorry

-- Define a function to get all segments in the 15-gon
def all_segments (polygon : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

-- Define a function to check if three segments form a triangle with positive area
def forms_triangle (s1 s2 s3 : Set (ℝ × ℝ)) : Prop := sorry

-- Define the total number of ways to choose 3 segments
def total_combinations : ℕ := Nat.choose 105 3

-- Define the number of valid triangles
def valid_triangles : ℕ := sorry

-- Theorem statement
theorem probability_of_valid_triangle :
  (valid_triangles : ℚ) / total_combinations = 713 / 780 := by sorry

end NUMINAMATH_CALUDE_probability_of_valid_triangle_l3319_331945


namespace NUMINAMATH_CALUDE_three_cubes_of_27_equals_3_to_10_l3319_331930

theorem three_cubes_of_27_equals_3_to_10 : ∃ x : ℕ, 27^3 + 27^3 + 27^3 = 3^x ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_three_cubes_of_27_equals_3_to_10_l3319_331930


namespace NUMINAMATH_CALUDE_triangle_vector_dot_product_l3319_331946

/-- Given a triangle ABC with vectors AB and AC, prove that the dot product of AB and BC equals 5 -/
theorem triangle_vector_dot_product (A B C : ℝ × ℝ) : 
  let AB : ℝ × ℝ := (2, 3)
  let AC : ℝ × ℝ := (3, 4)
  let BC : ℝ × ℝ := AC - AB
  (AB.1 * BC.1 + AB.2 * BC.2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_dot_product_l3319_331946


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3319_331997

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  l1 : ℝ → ℝ → ℝ := λ x y => a * x + (a + 1) * y + 1
  l2 : ℝ → ℝ → ℝ := λ x y => x + a * y + 2

/-- Perpendicularity condition for two lines -/
def isPerpendicular (lines : TwoLines) : Prop :=
  lines.a * 1 + (lines.a + 1) * lines.a = 0

/-- Theorem stating that a = -2 is a sufficient but not necessary condition for perpendicularity -/
theorem sufficient_not_necessary (lines : TwoLines) :
  (lines.a = -2 → isPerpendicular lines) ∧
  ¬(isPerpendicular lines → lines.a = -2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3319_331997


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l3319_331956

theorem max_sum_given_constraints (a b : ℝ) 
  (h1 : a^2 + b^2 = 130) 
  (h2 : a * b = 45) : 
  a + b ≤ 2 * Real.sqrt 55 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l3319_331956


namespace NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l3319_331917

theorem perpendicular_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, ax + y - 1 = 0 ↔ x + ay - 1 = 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l3319_331917


namespace NUMINAMATH_CALUDE_certain_number_problem_l3319_331998

theorem certain_number_problem (x N : ℝ) :
  625^(-x) + N^(-2*x) + 5^(-4*x) = 11 →
  x = 0.25 →
  N = 25/2809 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3319_331998


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3319_331901

/-- Given a quadratic function y = x² + 2x - 2, prove that if the maximum value of y is 1
    when a ≤ x ≤ 1/2, then a = -3. -/
theorem quadratic_max_value (y : ℝ → ℝ) (a : ℝ) :
  (∀ x, y x = x^2 + 2*x - 2) →
  (∀ x, a ≤ x → x ≤ 1/2 → y x ≤ 1) →
  (∃ x, a ≤ x ∧ x ≤ 1/2 ∧ y x = 1) →
  a = -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3319_331901


namespace NUMINAMATH_CALUDE_lisa_savings_l3319_331937

theorem lisa_savings (x : ℚ) : 
  (x + 3/5 * x + 2 * (3/5 * x) = 3760 - 400) → x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_lisa_savings_l3319_331937


namespace NUMINAMATH_CALUDE_complex_integer_sum_of_squares_l3319_331948

theorem complex_integer_sum_of_squares (x y : ℤ) :
  (∃ (a b c d : ℤ), x + y * I = (a + b * I)^2 + (c + d * I)^2) ↔ Even y := by
  sorry

end NUMINAMATH_CALUDE_complex_integer_sum_of_squares_l3319_331948


namespace NUMINAMATH_CALUDE_function_relationship_l3319_331962

/-- Given functions f and g, and constants A, B, and C, prove the relationship between A, B, and C. -/
theorem function_relationship (A B C : ℝ) (hB : B ≠ 0) :
  let f := fun x => A * x^2 - 2 * B^2 * x + 3
  let g := fun x => B * x + 1
  f (g 1) = C →
  A = (C + 2 * B^3 + 2 * B^2 - 3) / (B^2 + 2 * B + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_relationship_l3319_331962


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3319_331931

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3319_331931


namespace NUMINAMATH_CALUDE_candidate_votes_l3319_331908

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_percent = 80 / 100 →
  ∃ (valid_votes : ℕ) (candidate_votes : ℕ),
    valid_votes = (1 - invalid_percent) * total_votes ∧
    candidate_votes = candidate_percent * valid_votes ∧
    candidate_votes = 380800 := by
  sorry

end NUMINAMATH_CALUDE_candidate_votes_l3319_331908


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l3319_331995

def U : Set Nat := {1, 2, 3}
def A : Set Nat := {1, 2}

theorem complement_of_A_wrt_U : (U \ A) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l3319_331995


namespace NUMINAMATH_CALUDE_hundredth_term_value_l3319_331989

/-- A geometric sequence with first term 5 and second term -15 -/
def geometric_sequence (n : ℕ) : ℚ :=
  5 * (-3)^(n - 1)

/-- The 100th term of the geometric sequence -/
def a_100 : ℚ := geometric_sequence 100

theorem hundredth_term_value : a_100 = -5 * 3^99 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_value_l3319_331989


namespace NUMINAMATH_CALUDE_camera_price_difference_l3319_331952

-- Define the list price
def list_price : ℚ := 49.95

-- Define the price at Budget Buys
def budget_buys_price : ℚ := list_price - 10

-- Define the price at Value Mart
def value_mart_price : ℚ := list_price * (1 - 0.2)

-- Theorem to prove
theorem camera_price_difference :
  (max budget_buys_price value_mart_price - min budget_buys_price value_mart_price) * 100 = 1 := by
  sorry


end NUMINAMATH_CALUDE_camera_price_difference_l3319_331952


namespace NUMINAMATH_CALUDE_even_monotone_inequality_l3319_331959

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- State the theorem
theorem even_monotone_inequality (h1 : is_even f) (h2 : monotone_increasing_on_positive f) :
  f (-1) < f 2 ∧ f 2 < f (-3) :=
sorry

end NUMINAMATH_CALUDE_even_monotone_inequality_l3319_331959


namespace NUMINAMATH_CALUDE_third_term_is_six_l3319_331920

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_second_fourth : a 2 + a 4 = 10
  fourth_minus_third : a 4 = a 3 + 2

/-- The third term of the arithmetic sequence is 6 -/
theorem third_term_is_six (seq : ArithmeticSequence) : seq.a 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_six_l3319_331920


namespace NUMINAMATH_CALUDE_minuend_subtrahend_difference_problem_l3319_331970

theorem minuend_subtrahend_difference_problem :
  ∃ (a b c : ℤ),
    (a + b + c = 1024) ∧
    (c = b - 88) ∧
    (a = b + c) ∧
    (a = 712) ∧
    (b = 400) ∧
    (c = 312) := by
  sorry

end NUMINAMATH_CALUDE_minuend_subtrahend_difference_problem_l3319_331970


namespace NUMINAMATH_CALUDE_coin_problem_l3319_331909

theorem coin_problem :
  ∀ (x y : ℕ),
    x + y = 15 →
    2 * x + 5 * y = 51 →
    x = y + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3319_331909


namespace NUMINAMATH_CALUDE_davids_biology_marks_l3319_331954

theorem davids_biology_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (chemistry : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 61) 
  (h2 : mathematics = 65) 
  (h3 : physics = 82) 
  (h4 : chemistry = 67) 
  (h5 : average = 72) 
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) : 
  biology = 85 := by
sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l3319_331954


namespace NUMINAMATH_CALUDE_jenny_distance_relationship_l3319_331944

/-- Given Jenny's running and walking speeds and times, prove the relationship between distances -/
theorem jenny_distance_relationship 
  (x : ℝ) -- Jenny's running speed in miles per hour
  (y : ℝ) -- Jenny's walking speed in miles per hour
  (r : ℝ) -- Time spent running in minutes
  (w : ℝ) -- Time spent walking in minutes
  (d : ℝ) -- Difference in distance (running - walking) in miles
  (hx : x > 0) -- Assumption: running speed is positive
  (hy : y > 0) -- Assumption: walking speed is positive
  (hr : r ≥ 0) -- Assumption: time spent running is non-negative
  (hw : w ≥ 0) -- Assumption: time spent walking is non-negative
  : x * r - y * w = 60 * d :=
by sorry

end NUMINAMATH_CALUDE_jenny_distance_relationship_l3319_331944


namespace NUMINAMATH_CALUDE_largest_consecutive_even_integer_l3319_331991

theorem largest_consecutive_even_integer : ∃ n : ℕ,
  (n - 8) + (n - 6) + (n - 4) + (n - 2) + n = 2 * (25 * 26 / 2) ∧
  n % 2 = 0 ∧
  n = 134 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_even_integer_l3319_331991


namespace NUMINAMATH_CALUDE_tan_570_degrees_l3319_331961

theorem tan_570_degrees : Real.tan (570 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_570_degrees_l3319_331961


namespace NUMINAMATH_CALUDE_tangency_point_satisfies_equations_tangency_point_is_unique_l3319_331912

/-- The point of tangency for two parabolas -/
def point_of_tangency : ℝ × ℝ := (-7, -25)

/-- First parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 17*x + 40

/-- Second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 51*y + 650

/-- Theorem stating that the point_of_tangency satisfies both parabola equations -/
theorem tangency_point_satisfies_equations :
  parabola1 point_of_tangency.1 point_of_tangency.2 ∧
  parabola2 point_of_tangency.1 point_of_tangency.2 :=
by sorry

/-- Theorem stating that the point_of_tangency is the unique point satisfying both equations -/
theorem tangency_point_is_unique :
  ∀ (x y : ℝ), parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency :=
by sorry

end NUMINAMATH_CALUDE_tangency_point_satisfies_equations_tangency_point_is_unique_l3319_331912


namespace NUMINAMATH_CALUDE_floor_product_inequality_l3319_331949

theorem floor_product_inequality (m n : ℕ+) :
  ⌊Real.sqrt 2 * m⌋ * ⌊Real.sqrt 7 * n⌋ < ⌊Real.sqrt 14 * (m * n)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_product_inequality_l3319_331949


namespace NUMINAMATH_CALUDE_pentagon_area_l3319_331973

/-- The area of a pentagon with specific dimensions -/
theorem pentagon_area : 
  ∀ (right_triangle_base right_triangle_height trapezoid_base1 trapezoid_base2 trapezoid_height : ℝ),
  right_triangle_base = 28 →
  right_triangle_height = 30 →
  trapezoid_base1 = 25 →
  trapezoid_base2 = 18 →
  trapezoid_height = 39 →
  (1/2 * right_triangle_base * right_triangle_height) + 
  (1/2 * (trapezoid_base1 + trapezoid_base2) * trapezoid_height) = 1257 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_area_l3319_331973


namespace NUMINAMATH_CALUDE_min_value_expression_l3319_331994

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 1) (hc : c > 1) :
  (((a^2 + 1) / (2*a*b) - 1) * c + (Real.sqrt 2 / (c - 1))) ≥ 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3319_331994


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3319_331986

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) / (2 * x + 1) ≤ 0 ↔ x ∈ Set.Ioo (-1/2 : ℝ) 1 ∪ {1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3319_331986


namespace NUMINAMATH_CALUDE_triangle_area_l3319_331919

theorem triangle_area (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let angle_C := Real.arccos ((AB^2 + BC^2 - AC^2) / (2 * AB * BC))
  AB = 2 * Real.sqrt 3 ∧ BC = 2 ∧ angle_C = π / 3 →
  (1 / 2) * AB * BC * Real.sin angle_C = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3319_331919


namespace NUMINAMATH_CALUDE_outfits_with_restrictions_l3319_331972

/-- The number of unique outfits that can be made with shirts and pants, with restrictions -/
def uniqueOutfits (shirts : ℕ) (pants : ℕ) (restrictedShirts : ℕ) (restrictedPants : ℕ) : ℕ :=
  shirts * pants - restrictedShirts * restrictedPants

/-- Theorem stating the number of unique outfits under given conditions -/
theorem outfits_with_restrictions :
  uniqueOutfits 5 6 1 2 = 28 := by
  sorry

#eval uniqueOutfits 5 6 1 2

end NUMINAMATH_CALUDE_outfits_with_restrictions_l3319_331972
