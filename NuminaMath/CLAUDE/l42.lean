import Mathlib

namespace no_family_of_lines_exist_l42_4261

theorem no_family_of_lines_exist :
  ¬ ∃ (k : ℕ → ℝ), 
    (∀ n, k (n + 1) = (1 - 1 / k n) - (1 - k n)) ∧ 
    (∀ n, k n * k (n + 1) ≥ 0) ∧ 
    (∀ n, k n ≠ 0) := by
  sorry

end no_family_of_lines_exist_l42_4261


namespace unique_tuple_l42_4250

def satisfies_condition (a : Fin 9 → ℕ+) : Prop :=
  ∀ i j k l, i < j → j < k → k ≤ 9 → l ≠ i → l ≠ j → l ≠ k → l ≤ 9 →
    a i + a j + a k + a l = 100

theorem unique_tuple : ∃! a : Fin 9 → ℕ+, satisfies_condition a := by
  sorry

end unique_tuple_l42_4250


namespace inverse_equals_one_implies_a_equals_one_l42_4290

theorem inverse_equals_one_implies_a_equals_one (a : ℝ) (h : a ≠ 0) :
  a⁻¹ = (-1)^0 → a = 1 := by
  sorry

end inverse_equals_one_implies_a_equals_one_l42_4290


namespace right_triangle_arithmetic_progression_and_inradius_l42_4220

theorem right_triangle_arithmetic_progression_and_inradius (d : ℝ) (h : d > 0) :
  ∃ (a b c : ℝ),
    a^2 + b^2 = c^2 ∧  -- Pythagorean theorem
    a = 3*d ∧ 
    b = 4*d ∧ 
    c = 5*d ∧ 
    (a + b - c) / 2 = d  -- Inradius formula
  := by sorry

end right_triangle_arithmetic_progression_and_inradius_l42_4220


namespace root_interval_sum_l42_4279

def f (x : ℝ) := x^3 - x + 1

theorem root_interval_sum (a b : ℤ) : 
  (∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) →
  b - a = 1 →
  a + b = -3 := by
sorry

end root_interval_sum_l42_4279


namespace line_intersection_xz_plane_l42_4208

/-- The line passing through two points intersects the xz-plane at a specific point -/
theorem line_intersection_xz_plane (p₁ p₂ : ℝ × ℝ × ℝ) (intersection : ℝ × ℝ × ℝ) : 
  p₁ = (2, 3, 2) → 
  p₂ = (6, -1, 7) → 
  intersection.2 = 0 → 
  ∃ t : ℝ, intersection = (2 + 4*t, 3 - 4*t, 2 + 5*t) ∧ 
        intersection = (5, 0, 23/4) := by
  sorry

#check line_intersection_xz_plane

end line_intersection_xz_plane_l42_4208


namespace students_wanting_fruit_l42_4214

theorem students_wanting_fruit (red_apples green_apples extra_apples : ℕ) :
  red_apples = 43 →
  green_apples = 32 →
  extra_apples = 73 →
  (red_apples + green_apples + extra_apples) - (red_apples + green_apples) = extra_apples :=
by sorry

end students_wanting_fruit_l42_4214


namespace irrational_identification_l42_4235

theorem irrational_identification :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (5 : ℚ)^(1/3) = a / b) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (9 : ℚ)^(1/2) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (-8/3 : ℚ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (60.25 : ℚ) = a / b) :=
by sorry

end irrational_identification_l42_4235


namespace forty_five_million_scientific_notation_l42_4217

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem forty_five_million_scientific_notation :
  toScientificNotation 45000000 = ScientificNotation.mk 4.5 7 sorry := by sorry

end forty_five_million_scientific_notation_l42_4217


namespace max_value_with_remainder_l42_4219

theorem max_value_with_remainder (A B : ℕ) (h1 : A ≠ B) (h2 : A = 17 * 25 + B) : 
  (∀ C : ℕ, C < 17 → B ≤ C) → A = 441 :=
by sorry

end max_value_with_remainder_l42_4219


namespace newton_basketball_league_members_l42_4280

theorem newton_basketball_league_members :
  let headband_cost : ℕ := 3
  let jersey_cost : ℕ := headband_cost + 7
  let items_per_member : ℕ := 2  -- 2 headbands and 2 jerseys
  let total_cost : ℕ := 2700
  (total_cost = (headband_cost * items_per_member + jersey_cost * items_per_member) * 103) :=
by sorry

end newton_basketball_league_members_l42_4280


namespace project_completion_time_l42_4202

/-- Given a project requiring 1500 hours and a daily work schedule of 15 hours,
    prove that the number of days needed to complete the project is 100. -/
theorem project_completion_time (project_hours : ℕ) (daily_hours : ℕ) :
  project_hours = 1500 →
  daily_hours = 15 →
  project_hours / daily_hours = 100 := by
  sorry

end project_completion_time_l42_4202


namespace fruit_purchase_problem_l42_4205

/-- Fruit purchase problem -/
theorem fruit_purchase_problem (x y : ℝ) :
  let apple_weight : ℝ := 2
  let orange_weight : ℝ := 5 * apple_weight
  let total_weight : ℝ := apple_weight + orange_weight
  let total_cost : ℝ := x * apple_weight + y * orange_weight
  (orange_weight = 10 ∧ total_cost = 2*x + 10*y) ∧ total_weight = 12 := by
  sorry

end fruit_purchase_problem_l42_4205


namespace determinant_equality_l42_4287

theorem determinant_equality (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 5 →
  Matrix.det ![![a - c, b - d], ![c, d]] = 5 := by
  sorry

end determinant_equality_l42_4287


namespace hexagon_smallest_angle_l42_4256

-- Define a hexagon with angles in arithmetic progression
def hexagon_angles (x : ℝ) : List ℝ := [x, x + 10, x + 20, x + 30, x + 40, x + 50]

-- Theorem statement
theorem hexagon_smallest_angle :
  ∃ (x : ℝ), 
    (List.sum (hexagon_angles x) = 720) ∧ 
    (∀ (angle : ℝ), angle ∈ hexagon_angles x → angle ≥ x) ∧
    x = 95 := by
  sorry

end hexagon_smallest_angle_l42_4256


namespace unique_valid_config_l42_4286

/-- Represents a fence configuration --/
structure FenceConfig where
  max_length : Nat
  num_max : Nat
  num_minus_one : Nat
  num_minus_two : Nat
  num_minus_three : Nat

/-- Checks if a fence configuration is valid --/
def is_valid_config (config : FenceConfig) : Prop :=
  config.num_max + config.num_minus_one + config.num_minus_two + config.num_minus_three = 16 ∧
  config.num_max * config.max_length +
  config.num_minus_one * (config.max_length - 1) +
  config.num_minus_two * (config.max_length - 2) +
  config.num_minus_three * (config.max_length - 3) = 297 ∧
  config.num_max = 8

/-- The unique valid fence configuration --/
def unique_config : FenceConfig :=
  { max_length := 20
  , num_max := 8
  , num_minus_one := 0
  , num_minus_two := 7
  , num_minus_three := 1
  }

/-- Theorem stating that the unique_config is the only valid configuration --/
theorem unique_valid_config :
  is_valid_config unique_config ∧
  (∀ config : FenceConfig, is_valid_config config → config = unique_config) := by
  sorry


end unique_valid_config_l42_4286


namespace correct_operation_l42_4226

theorem correct_operation (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = -x * y^2 := by
  sorry

end correct_operation_l42_4226


namespace fraction_calculation_l42_4251

theorem fraction_calculation : (5 / 3 : ℚ) ^ 3 * (2 / 5 : ℚ) = 50 / 27 := by
  sorry

end fraction_calculation_l42_4251


namespace number_line_is_line_l42_4296

/-- A number line represents the set of real numbers. -/
def NumberLine : Type := ℝ

/-- A line is an infinite one-dimensional figure extending in both directions. -/
def Line : Type := ℝ

/-- A number line is equivalent to a line. -/
theorem number_line_is_line : NumberLine ≃ Line := by sorry

end number_line_is_line_l42_4296


namespace snack_cost_theorem_l42_4240

/-- The total cost of snacks bought by Robert and Teddy -/
def total_cost (pizza_price : ℕ) (pizza_quantity : ℕ) (drink_price : ℕ) (robert_drink_quantity : ℕ) (hamburger_price : ℕ) (hamburger_quantity : ℕ) (teddy_drink_quantity : ℕ) : ℕ :=
  pizza_price * pizza_quantity + 
  drink_price * robert_drink_quantity + 
  hamburger_price * hamburger_quantity + 
  drink_price * teddy_drink_quantity

theorem snack_cost_theorem : 
  total_cost 10 5 2 10 3 6 10 = 108 := by
  sorry

end snack_cost_theorem_l42_4240


namespace necessary_not_sufficient_l42_4246

-- Define a real-valued function
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Define what it means for a function to have extreme values
def has_extreme_value (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

-- Define what it means for a function to have real roots
def has_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = 0

-- Statement: f'(x) = 0 having real roots is necessary but not sufficient for f(x) having extreme values
theorem necessary_not_sufficient :
  (has_extreme_value f → has_real_roots f') ∧
  ¬(has_real_roots f' → has_extreme_value f) :=
sorry

end necessary_not_sufficient_l42_4246


namespace real_part_of_complex_fraction_l42_4295

theorem real_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) : 
  Complex.re ((1 - 2*i) / (2 + i^5)) = 0 := by sorry

end real_part_of_complex_fraction_l42_4295


namespace S_inter_T_eq_T_l42_4245

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem S_inter_T_eq_T : S ∩ T = T := by sorry

end S_inter_T_eq_T_l42_4245


namespace sofia_survey_l42_4291

theorem sofia_survey (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 185) 
  (h2 : bacon = 125) : 
  mashed_potatoes + bacon = 310 := by
sorry

end sofia_survey_l42_4291


namespace money_distribution_l42_4232

theorem money_distribution (raquel sam nataly tom : ℚ) : 
  raquel = 40 →
  nataly = 3 * raquel →
  nataly = (5/3) * sam →
  tom = (1/4) * nataly →
  tom + raquel + nataly + sam = 262 := by
sorry

end money_distribution_l42_4232


namespace cube_root_fifth_power_sixth_l42_4238

theorem cube_root_fifth_power_sixth : (((5 ^ (1/2)) ^ 4) ^ (1/3)) ^ 6 = 625 := by
  sorry

end cube_root_fifth_power_sixth_l42_4238


namespace expression_equals_24_l42_4274

def arithmetic_expression (a b c d : ℕ) : Prop :=
  ∃ (e : ℕ → ℕ → ℕ → ℕ → ℕ), e a b c d = 24

theorem expression_equals_24 : arithmetic_expression 8 8 8 10 := by
  sorry

end expression_equals_24_l42_4274


namespace unique_friendship_configs_l42_4227

/-- Represents a friendship configuration in a group of 8 people --/
structure FriendshipConfig :=
  (num_friends : Nat)
  (valid : num_friends = 0 ∨ num_friends = 1 ∨ num_friends = 6)

/-- Counts the number of unique friendship configurations --/
def count_unique_configs : Nat :=
  sorry

/-- Theorem stating that the number of unique friendship configurations is 37 --/
theorem unique_friendship_configs :
  count_unique_configs = 37 :=
sorry

end unique_friendship_configs_l42_4227


namespace geometric_sequence_converse_l42_4254

/-- The converse of a proposition "If P, then Q" is "If Q, then P" -/
def converse_of (P Q : Prop) : Prop :=
  Q → P

/-- Three real numbers form a geometric sequence if the middle term 
    is the geometric mean of the other two -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

/-- The proposition "If a, b, c form a geometric sequence, then b^2 = ac" 
    and its converse -/
theorem geometric_sequence_converse :
  converse_of (is_geometric_sequence a b c) (b^2 = a * c) =
  (b^2 = a * c → is_geometric_sequence a b c) :=
sorry

end geometric_sequence_converse_l42_4254


namespace percentage_increase_relation_l42_4262

theorem percentage_increase_relation (A B k x : ℝ) : 
  A > 0 → B > 0 → k > 1 → A = k * B → A = B * (1 + x / 100) → k = 1 + x / 100 := by
  sorry

end percentage_increase_relation_l42_4262


namespace sum_of_digits_power_product_l42_4233

def power_product : ℕ := 2^2010 * 5^2008 * 7

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_power_product : sum_of_digits power_product = 10 := by sorry

end sum_of_digits_power_product_l42_4233


namespace square_difference_l42_4255

theorem square_difference (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end square_difference_l42_4255


namespace tire_cost_theorem_l42_4225

/-- Calculates the total cost of tires with given prices, discounts, and taxes -/
def totalTireCost (allTerrainPrice : ℝ) (allTerrainDiscount : ℝ) (allTerrainTax : ℝ)
                  (sparePrice : ℝ) (spareDiscount : ℝ) (spareTax : ℝ) : ℝ :=
  let allTerrainDiscountedPrice := allTerrainPrice * (1 - allTerrainDiscount)
  let allTerrainFinalPrice := allTerrainDiscountedPrice * (1 + allTerrainTax)
  let allTerrainTotal := 4 * allTerrainFinalPrice

  let spareDiscountedPrice := sparePrice * (1 - spareDiscount)
  let spareFinalPrice := spareDiscountedPrice * (1 + spareTax)

  allTerrainTotal + spareFinalPrice

/-- The total cost of tires is $291.20 -/
theorem tire_cost_theorem :
  totalTireCost 60 0.15 0.08 75 0.10 0.05 = 291.20 := by
  sorry

end tire_cost_theorem_l42_4225


namespace quarter_circle_arcs_sum_limit_l42_4224

/-- The sum of the lengths of quarter-circle arcs approaches πR/2 as n approaches infinity -/
theorem quarter_circle_arcs_sum_limit (R : ℝ) (h : R > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * R / (2 * n)) - π * R / 2| < ε := by
  sorry

end quarter_circle_arcs_sum_limit_l42_4224


namespace no_square_subdivision_l42_4258

theorem no_square_subdivision : ¬ ∃ (s : ℝ) (n : ℕ), 
  s > 0 ∧ n > 0 ∧ 
  ∃ (a : ℝ), a > 0 ∧ 
  s * s = n * (1/2 * a * a * Real.sqrt 3) ∧
  s = a * Real.sqrt 3 ∨ s = 2 * a ∨ s = 3 * a :=
sorry

end no_square_subdivision_l42_4258


namespace perpendicular_lines_sum_l42_4215

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def point_on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

theorem perpendicular_lines_sum (a b c : ℝ) : 
  perpendicular (-a/4) (2/5) →
  point_on_line 1 c a 4 (-2) →
  point_on_line 1 c 2 (-5) b →
  a + b + c = -4 := by
  sorry

end perpendicular_lines_sum_l42_4215


namespace train_length_l42_4213

/-- Calculates the length of a train given its speed and time to pass a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 5 → speed_kmh * (1000 / 3600) * time_s = 100 := by
  sorry

#check train_length

end train_length_l42_4213


namespace triangle_city_population_l42_4244

theorem triangle_city_population : ∃ (x y z : ℕ+), 
  x^2 + 50 = y^2 + 1 ∧ 
  y^2 + 351 = z^2 ∧ 
  x^2 = 576 := by
sorry

end triangle_city_population_l42_4244


namespace bernoulli_inequality_l42_4282

theorem bernoulli_inequality (x : ℝ) (n : ℕ) 
  (h1 : x > -1) (h2 : x ≠ 0) (h3 : n > 1) : 
  (1 + x)^n > 1 + n * x := by
  sorry

end bernoulli_inequality_l42_4282


namespace sequence_properties_l42_4263

/-- Given a sequence and its partial sum satisfying certain conditions, 
    prove that it's geometric and find the range of t when the sum converges to 1 -/
theorem sequence_properties (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) (t : ℝ) 
    (h1 : ∀ n : ℕ+, S n = 1 + t * a n) 
    (h2 : t ≠ 1) (h3 : t ≠ 0) :
  (∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n) ∧ 
  (∀ ε > 0, ∃ N : ℕ+, ∀ n ≥ N, |S n - 1| < ε) → 
  (t < 1/2 ∧ t ≠ 0) := by
  sorry

end sequence_properties_l42_4263


namespace min_output_no_loss_l42_4288

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the sales revenue function
def sales_revenue (x : ℝ) : ℝ := 25 * x

-- Define the domain constraint
def in_domain (x : ℝ) : Prop := 0 < x ∧ x < 240

-- Theorem statement
theorem min_output_no_loss :
  ∃ (x_min : ℝ), x_min = 150 ∧
  in_domain x_min ∧
  (∀ x : ℝ, in_domain x → sales_revenue x ≥ total_cost x → x ≥ x_min) :=
sorry

end min_output_no_loss_l42_4288


namespace quadratic_form_minimum_l42_4259

theorem quadratic_form_minimum : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -3 ∧
  ∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * x₀ * y₀ + 5 * y₀^2 - 8 * x₀ - 10 * y₀ = -3 :=
by sorry

end quadratic_form_minimum_l42_4259


namespace jake_peaches_l42_4269

theorem jake_peaches (steven_peaches jill_peaches jake_peaches : ℕ) : 
  jake_peaches + 7 = steven_peaches → 
  steven_peaches = jill_peaches + 14 → 
  steven_peaches = 15 → 
  jake_peaches = 8 := by
sorry

end jake_peaches_l42_4269


namespace simplify_expression_1_simplify_expression_2_l42_4292

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  2 * x^2 * y - 5 * x * y - 4 * x * y^2 + x * y + 4 * x^2 * y - 7 * x * y^2 =
  6 * x^2 * y - 4 * x * y - 11 * x * y^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) :
  (5 * a^2 + 2 * a - 1) - 4 * (2 * a^2 - 3 * a) =
  -3 * a^2 + 14 * a - 1 := by sorry

end simplify_expression_1_simplify_expression_2_l42_4292


namespace pizza_lovers_count_l42_4241

theorem pizza_lovers_count (total pupils_like_burgers pupils_like_both : ℕ) 
  (h1 : total = 200)
  (h2 : pupils_like_burgers = 115)
  (h3 : pupils_like_both = 40)
  : ∃ pupils_like_pizza : ℕ, 
    pupils_like_pizza + pupils_like_burgers - pupils_like_both = total ∧ 
    pupils_like_pizza = 125 :=
by sorry

end pizza_lovers_count_l42_4241


namespace dodecagon_diagonals_l42_4268

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by sorry

end dodecagon_diagonals_l42_4268


namespace literature_study_time_l42_4297

def science_time : ℕ := 60
def math_time : ℕ := 80
def total_time_hours : ℕ := 3

theorem literature_study_time :
  total_time_hours * 60 - science_time - math_time = 40 := by
  sorry

end literature_study_time_l42_4297


namespace discount_percentage_l42_4206

/-- The percentage discount for buying 3 pairs of shorts at once, given the regular price and savings -/
theorem discount_percentage
  (regular_price : ℚ)  -- Regular price of one pair of shorts
  (total_savings : ℚ)  -- Total savings when buying 3 pairs at once
  (h1 : regular_price = 10)  -- Each pair costs $10 normally
  (h2 : total_savings = 3)   -- Saving $3 by buying 3 pairs at once
  : (total_savings / (3 * regular_price)) * 100 = 10 := by
  sorry

end discount_percentage_l42_4206


namespace one_third_percent_of_180_l42_4272

theorem one_third_percent_of_180 : (1 / 3 : ℚ) / 100 * 180 = 0.6 := by
  sorry

end one_third_percent_of_180_l42_4272


namespace worker_delay_l42_4216

/-- Proves that reducing speed to 5/6 of normal results in a 12-minute delay -/
theorem worker_delay (usual_time : ℝ) (speed_ratio : ℝ) 
  (h1 : usual_time = 60)
  (h2 : speed_ratio = 5 / 6) : 
  (usual_time / speed_ratio) - usual_time = 12 := by
  sorry

#check worker_delay

end worker_delay_l42_4216


namespace decimal_23_equals_binary_10111_l42_4231

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_23_equals_binary_10111 :
  to_binary 23 = [true, true, true, false, true] ∧
  from_binary [true, true, true, false, true] = 23 := by
  sorry

end decimal_23_equals_binary_10111_l42_4231


namespace game_cost_l42_4237

theorem game_cost (initial_money : ℕ) (num_toys : ℕ) (toy_price : ℕ) (game_cost : ℕ) : 
  initial_money = 57 → 
  num_toys = 5 → 
  toy_price = 6 → 
  initial_money = game_cost + (num_toys * toy_price) → 
  game_cost = 27 := by
sorry

end game_cost_l42_4237


namespace binomial_square_constant_l42_4298

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 9*x^2 + 30*x + c = (a*x + b)^2) → c = 25 := by
  sorry

end binomial_square_constant_l42_4298


namespace gregs_ppo_reward_l42_4276

theorem gregs_ppo_reward (
  ppo_percentage : Real)
  (coinrun_max_reward : Real)
  (procgen_max_reward : Real)
  (h1 : ppo_percentage = 0.9)
  (h2 : coinrun_max_reward = procgen_max_reward / 2)
  (h3 : procgen_max_reward = 240)
  : ppo_percentage * coinrun_max_reward = 108 := by
  sorry

end gregs_ppo_reward_l42_4276


namespace profit_difference_l42_4230

def original_profit_percentage : ℝ := 0.1
def new_purchase_discount : ℝ := 0.1
def new_profit_percentage : ℝ := 0.3
def original_selling_price : ℝ := 1099.999999999999

theorem profit_difference :
  let original_purchase_price := original_selling_price / (1 + original_profit_percentage)
  let new_purchase_price := original_purchase_price * (1 - new_purchase_discount)
  let new_selling_price := new_purchase_price * (1 + new_profit_percentage)
  new_selling_price - original_selling_price = 70 := by sorry

end profit_difference_l42_4230


namespace quadratic_equivalence_l42_4253

theorem quadratic_equivalence : ∀ x : ℝ, x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by sorry

end quadratic_equivalence_l42_4253


namespace flour_calculation_l42_4242

/-- Given a recipe for cookies, calculate the amount of each type of flour needed when doubling the recipe and using two types of flour. -/
theorem flour_calculation (original_cookies : ℕ) (original_flour : ℚ) (new_cookies : ℕ) :
  original_cookies > 0 →
  original_flour > 0 →
  new_cookies = 2 * original_cookies →
  ∃ (flour_each : ℚ),
    flour_each = original_flour ∧
    flour_each * 2 = new_cookies / original_cookies * original_flour :=
by sorry

end flour_calculation_l42_4242


namespace debbys_flour_amount_l42_4200

/-- Proves that Debby's total flour amount is correct given her initial amount and purchase. -/
theorem debbys_flour_amount (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 12 → bought = 4 → total = initial + bought → total = 16 := by
  sorry

end debbys_flour_amount_l42_4200


namespace current_speed_l42_4281

/-- Calculates the speed of the current given the rowing speed in still water and the time taken to cover a distance downstream. -/
theorem current_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  rowing_speed = 30 →
  distance = 100 →
  time = 9.99920006399488 →
  (distance / time) * 3.6 - rowing_speed = 6 := by
  sorry

#eval (100 / 9.99920006399488) * 3.6 - 30

end current_speed_l42_4281


namespace budget_allocation_l42_4294

/-- Given a family's budget allocation, calculate the fraction spent on eating out -/
theorem budget_allocation (budget_groceries : ℝ) (budget_total_food : ℝ) 
  (h1 : budget_groceries = 0.6) 
  (h2 : budget_total_food = 0.8) :
  budget_total_food - budget_groceries = 0.2 := by
  sorry

end budget_allocation_l42_4294


namespace train_speed_train_speed_is_50_l42_4218

/-- The speed of a train given travel time and alternative speed scenario -/
theorem train_speed (travel_time : ℝ) (alt_time : ℝ) (alt_speed : ℝ) : ℝ :=
  let distance := alt_speed * alt_time
  distance / travel_time

/-- Proof that the train's speed is 50 mph given the specified conditions -/
theorem train_speed_is_50 :
  train_speed 4 2 100 = 50 := by
  sorry

end train_speed_train_speed_is_50_l42_4218


namespace max_students_in_dance_l42_4252

theorem max_students_in_dance (x : ℕ) : 
  x < 100 ∧ 
  x % 8 = 5 ∧ 
  x % 5 = 3 →
  x ≤ 93 ∧ 
  ∃ y : ℕ, y = 93 ∧ 
    y < 100 ∧ 
    y % 8 = 5 ∧ 
    y % 5 = 3 :=
by sorry

end max_students_in_dance_l42_4252


namespace current_velocity_l42_4285

/-- Velocity of current given rowing speed and round trip time -/
theorem current_velocity (rowing_speed : ℝ) (distance : ℝ) (total_time : ℝ) :
  rowing_speed = 5 →
  distance = 2.4 →
  total_time = 1 →
  ∃ v : ℝ,
    v > 0 ∧
    (distance / (rowing_speed - v) + distance / (rowing_speed + v) = total_time) ∧
    v = 1 := by
  sorry

end current_velocity_l42_4285


namespace max_perimeter_after_cut_l42_4228

theorem max_perimeter_after_cut (original_length original_width cut_length cut_width : ℝ) 
  (h1 : original_length = 20)
  (h2 : original_width = 16)
  (h3 : cut_length = 8)
  (h4 : cut_width = 4)
  (h5 : cut_length ≤ original_length ∧ cut_width ≤ original_width) :
  ∃ (remaining_perimeter : ℝ), 
    remaining_perimeter ≤ 2 * (original_length + original_width) + 2 * min cut_length cut_width ∧
    remaining_perimeter = 88 := by
  sorry

end max_perimeter_after_cut_l42_4228


namespace min_z_in_triangle_ABC_l42_4293

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (1, 0)

-- Define the function z
def z (p : ℝ × ℝ) : ℝ := p.1 - p.2

-- Define the set of points inside or on the boundary of triangle ABC
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    p = (a * A.1 + b * B.1 + c * C.1, a * A.2 + b * B.2 + c * C.2)}

-- Theorem statement
theorem min_z_in_triangle_ABC :
  ∃ (p : ℝ × ℝ), p ∈ triangle_ABC ∧ ∀ (q : ℝ × ℝ), q ∈ triangle_ABC → z p ≤ z q ∧ z p = -3 :=
sorry

end min_z_in_triangle_ABC_l42_4293


namespace wire_cutting_l42_4249

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) :
  total_length = 49 →
  ratio = 2 / 5 →
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 14 := by
sorry

end wire_cutting_l42_4249


namespace cube_root_1728_l42_4299

theorem cube_root_1728 (a b : ℕ+) (h1 : (1728 : ℝ)^(1/3) = a * b^(1/3)) 
  (h2 : ∀ c d : ℕ+, (1728 : ℝ)^(1/3) = c * d^(1/3) → b ≤ d) : 
  a + b = 13 := by sorry

end cube_root_1728_l42_4299


namespace pencil_distribution_result_l42_4278

/-- Represents the pencil distribution problem --/
structure PencilDistribution where
  gloria_initial : ℕ
  lisa_initial : ℕ
  tim_initial : ℕ

/-- Calculates the final pencil counts after Lisa's distribution --/
def final_counts (pd : PencilDistribution) : ℕ × ℕ × ℕ :=
  let lisa_half := pd.lisa_initial / 2
  (pd.gloria_initial + lisa_half, 0, pd.tim_initial + lisa_half)

/-- Theorem stating the final pencil counts after distribution --/
theorem pencil_distribution_result (pd : PencilDistribution)
  (h1 : pd.gloria_initial = 2500)
  (h2 : pd.lisa_initial = 75800)
  (h3 : pd.tim_initial = 1950) :
  final_counts pd = (40400, 0, 39850) := by
  sorry

end pencil_distribution_result_l42_4278


namespace point_inside_circle_implies_a_range_l42_4284

/-- The circle with equation (x-a)^2 + (y+a)^2 = 4 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 + a)^2 = 4}

/-- A point is inside the circle if its distance from the center is less than the radius -/
def IsInside (p : ℝ × ℝ) (a : ℝ) : Prop :=
  (p.1 - a)^2 + (p.2 + a)^2 < 4

/-- The theorem stating that if P(1,1) is inside the circle, then -1 < a < 1 -/
theorem point_inside_circle_implies_a_range :
  ∀ a : ℝ, IsInside (1, 1) a → -1 < a ∧ a < 1 :=
by sorry

end point_inside_circle_implies_a_range_l42_4284


namespace sin_330_degrees_l42_4257

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l42_4257


namespace workshop_schedule_l42_4222

theorem workshop_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 10))) = 360 := by
  sorry

end workshop_schedule_l42_4222


namespace hyperbola_standard_form_l42_4265

theorem hyperbola_standard_form (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : 2 * a = 8) (h4 : (a^2 + b^2) / a^2 = (5/4)^2) :
  a = 4 ∧ b = 3 := by
  sorry

end hyperbola_standard_form_l42_4265


namespace transformation_result_l42_4267

/-- Reflect a point about the line y = x -/
def reflect_about_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Rotate a point 180° counterclockwise around (2,3) -/
def rotate_180_around_2_3 (p : ℝ × ℝ) : ℝ × ℝ :=
  (4 - p.1, 6 - p.2)

/-- The final position after transformations -/
def final_position : ℝ × ℝ := (-2, -1)

theorem transformation_result (m n : ℝ) : 
  rotate_180_around_2_3 (reflect_about_y_eq_x (m, n)) = final_position → n - m = -1 := by
  sorry

#check transformation_result

end transformation_result_l42_4267


namespace parabola_y_intercepts_l42_4260

theorem parabola_y_intercepts (y : ℝ) : ¬ ∃ y, 3 * y^2 - 4 * y + 8 = 0 := by
  sorry

end parabola_y_intercepts_l42_4260


namespace contrapositive_equivalence_l42_4201

theorem contrapositive_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0) :=
by sorry

end contrapositive_equivalence_l42_4201


namespace sum_of_positive_reals_l42_4239

theorem sum_of_positive_reals (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_sq_xy : x^2 + y^2 = 2500)
  (sum_sq_zw : z^2 + w^2 = 2500)
  (prod_xz : x * z = 1200)
  (prod_yw : y * w = 1200) :
  x + y + z + w = 140 := by
  sorry

end sum_of_positive_reals_l42_4239


namespace jones_clothing_count_l42_4212

def pants_count : ℕ := 40
def shirts_per_pants : ℕ := 6
def ties_per_pants : ℕ := 5
def socks_per_shirt : ℕ := 3

def total_clothing : ℕ := 
  pants_count + 
  (pants_count * shirts_per_pants) + 
  (pants_count * ties_per_pants) + 
  (pants_count * shirts_per_pants * socks_per_shirt)

theorem jones_clothing_count : total_clothing = 1200 := by
  sorry

end jones_clothing_count_l42_4212


namespace immigrant_count_l42_4204

/-- The number of people born in the country last year -/
def births : ℕ := 90171

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := 106491

/-- The number of immigrants to the country last year -/
def immigrants : ℕ := total_new_people - births

theorem immigrant_count : immigrants = 16320 := by
  sorry

end immigrant_count_l42_4204


namespace vidyas_age_multiple_l42_4283

theorem vidyas_age_multiple (vidya_age mother_age : ℕ) (h1 : vidya_age = 13) (h2 : mother_age = 44) :
  ∃ m : ℕ, m * vidya_age + 5 = mother_age ∧ m = 3 := by
  sorry

end vidyas_age_multiple_l42_4283


namespace batsman_score_l42_4207

theorem batsman_score (T : ℝ) : 
  (5 * 4 + 5 * 6 : ℝ) + (2/3) * T = T → T = 150 := by sorry

end batsman_score_l42_4207


namespace line_through_1_0_perpendicular_to_polar_axis_l42_4275

-- Define the polar coordinate system
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

-- Define a line in polar coordinates
structure PolarLine where
  equation : PolarPoint → Prop

-- Define the polar axis
def polarAxis : PolarLine :=
  { equation := fun p => p.θ = 0 }

-- Define perpendicularity in polar coordinates
def perpendicular (l1 l2 : PolarLine) : Prop :=
  sorry

-- Define the point (1, 0) in polar coordinates
def point_1_0 : PolarPoint :=
  { ρ := 1, θ := 0 }

-- The theorem to be proved
theorem line_through_1_0_perpendicular_to_polar_axis :
  ∃ (l : PolarLine),
    l.equation = fun p => p.ρ * Real.cos p.θ = 1 ∧
    l.equation point_1_0 ∧
    perpendicular l polarAxis :=
  sorry

end line_through_1_0_perpendicular_to_polar_axis_l42_4275


namespace inverse_proportion_k_value_l42_4266

/-- Given an inverse proportion function y = k/x passing through (2, -6), prove k = -12 -/
theorem inverse_proportion_k_value : ∀ k : ℝ, 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = k / x) ∧ f 2 = -6) → 
  k = -12 := by
  sorry

end inverse_proportion_k_value_l42_4266


namespace arithmetic_sequence_condition_l42_4210

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_condition
  (a : ℕ → ℝ) (m p q : ℕ) (h : arithmetic_sequence a) :
  (∀ m p q : ℕ, p + q = 2 * m → a p + a q = 2 * a m) ∧
  (∃ m p q : ℕ, a p + a q = 2 * a m ∧ p + q ≠ 2 * m) :=
sorry

end arithmetic_sequence_condition_l42_4210


namespace income_p_is_3000_l42_4289

/-- The monthly income of three people given their pairwise averages -/
def monthly_income (avg_pq avg_qr avg_pr : ℚ) : ℚ × ℚ × ℚ :=
  let p := 2 * (avg_pq + avg_pr - avg_qr)
  let q := 2 * (avg_pq + avg_qr - avg_pr)
  let r := 2 * (avg_qr + avg_pr - avg_pq)
  (p, q, r)

theorem income_p_is_3000 (avg_pq avg_qr avg_pr : ℚ) :
  avg_pq = 2050 → avg_qr = 5250 → avg_pr = 6200 →
  (monthly_income avg_pq avg_qr avg_pr).1 = 3000 := by
  sorry

end income_p_is_3000_l42_4289


namespace intersection_and_trajectory_l42_4248

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

-- Define the line l passing through the origin
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the intersection of line l and circle C₁
def intersection (k x y : ℝ) : Prop := C₁ x y ∧ line_l k x y

-- Define the range of k for intersection
def k_range (k : ℝ) : Prop := -2 * Real.sqrt 5 / 5 ≤ k ∧ k ≤ 2 * Real.sqrt 5 / 5

-- Define the trajectory of midpoint M
def trajectory_M (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = 9/4 ∧ 5/3 < x ∧ x ≤ 3

-- Theorem statement
theorem intersection_and_trajectory :
  (∀ k, k_range k ↔ ∃ x y, intersection k x y) ∧
  (∀ k x₁ y₁ x₂ y₂,
    intersection k x₁ y₁ ∧ intersection k x₂ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) →
    trajectory_M ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)) :=
sorry

end intersection_and_trajectory_l42_4248


namespace intersection_equality_l42_4211

theorem intersection_equality (m : ℝ) : 
  let A : Set ℝ := {0, 1, 2}
  let B : Set ℝ := {1, m}
  A ∩ B = B → m = 0 ∨ m = 2 := by
sorry

end intersection_equality_l42_4211


namespace geometric_sequence_sum_l42_4264

/-- Given a geometric sequence {a_n} with a_3 * a_7 = 8 and a_4 + a_6 = 6, prove that a_2 + a_8 = 9 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_prod : a 3 * a 7 = 8) (h_sum : a 4 + a 6 = 6) : 
  a 2 + a 8 = 9 := by
sorry

end geometric_sequence_sum_l42_4264


namespace point_coordinates_l42_4236

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance from a point to x-axis and y-axis
def distToXAxis (p : Point2D) : ℝ := |p.y|
def distToYAxis (p : Point2D) : ℝ := |p.x|

-- Define the set of possible coordinates
def possibleCoordinates : Set Point2D :=
  {⟨2, 1⟩, ⟨2, -1⟩, ⟨-2, 1⟩, ⟨-2, -1⟩}

-- Theorem statement
theorem point_coordinates (M : Point2D) :
  distToXAxis M = 1 ∧ distToYAxis M = 2 → M ∈ possibleCoordinates := by
  sorry

end point_coordinates_l42_4236


namespace no_adjacent_x_probability_l42_4203

-- Define the number of X tiles and O tiles
def num_x : ℕ := 4
def num_o : ℕ := 3

-- Define the total number of tiles
def total_tiles : ℕ := num_x + num_o

-- Function to calculate the number of ways to arrange tiles
def arrange_tiles (n k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate the number of valid arrangements (no adjacent X tiles)
def valid_arrangements : ℕ := 1

-- Theorem statement
theorem no_adjacent_x_probability :
  (valid_arrangements : ℚ) / (arrange_tiles total_tiles num_x) = 1 / 35 := by
  sorry

end no_adjacent_x_probability_l42_4203


namespace sqrt_x_minus_2_real_l42_4223

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_x_minus_2_real_l42_4223


namespace green_triangle_cost_l42_4221

/-- Calculates the cost of greening a right-angled triangle -/
theorem green_triangle_cost 
  (a b c : ℝ) 
  (cost_per_sqm : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_a : a = 8) 
  (h_b : b = 15) 
  (h_c : c = 17) 
  (h_cost : cost_per_sqm = 50) : 
  (1/2 * a * b) * cost_per_sqm = 3000 := by
sorry

end green_triangle_cost_l42_4221


namespace sum_of_possible_x_values_l42_4209

theorem sum_of_possible_x_values (x : ℝ) (h : |x - 12| = 100) : 
  ∃ (x₁ x₂ : ℝ), |x₁ - 12| = 100 ∧ |x₂ - 12| = 100 ∧ x₁ + x₂ = 24 := by
sorry

end sum_of_possible_x_values_l42_4209


namespace polygon_sides_when_interior_twice_exterior_l42_4229

theorem polygon_sides_when_interior_twice_exterior :
  ∀ n : ℕ,
  (n ≥ 3) →
  ((n - 2) * 180 = 2 * 360) →
  n = 6 :=
by
  sorry

end polygon_sides_when_interior_twice_exterior_l42_4229


namespace no_common_terms_except_one_l42_4247

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem no_common_terms_except_one (m n : ℕ) : x m = y n → m = 0 ∧ n = 0 := by
  sorry

end no_common_terms_except_one_l42_4247


namespace greatest_integer_third_side_l42_4234

theorem greatest_integer_third_side (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), c = 17 ∧ 
  (∀ (x : ℕ), x > c → ¬(x < a + b ∧ x > |a - b| ∧ a < b + x ∧ b < a + x)) ∧
  (c < a + b ∧ c > |a - b| ∧ a < b + c ∧ b < a + c) :=
sorry

end greatest_integer_third_side_l42_4234


namespace fraction_equation_solution_l42_4270

theorem fraction_equation_solution (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  (2 / x) - (1 / y) = (3 / z) → z = (2 * y - x) / 3 := by
  sorry

end fraction_equation_solution_l42_4270


namespace max_profit_pork_zongzi_l42_4271

/-- Represents the wholesale and retail prices of zongzi -/
structure ZongziPrices where
  porkWholesale : ℝ
  redBeanWholesale : ℝ
  porkRetail : ℝ

/-- Represents the daily sales and profit of pork zongzi -/
structure PorkZongziSales where
  price : ℝ
  quantity : ℝ
  profit : ℝ

/-- The conditions given in the problem -/
def zongziConditions (z : ZongziPrices) : Prop :=
  z.porkWholesale = z.redBeanWholesale + 10 ∧
  z.porkWholesale + 2 * z.redBeanWholesale = 100

/-- The relationship between price and quantity sold for pork zongzi -/
def porkZongziDemand (basePrice baseQuantity : ℝ) (z : ZongziPrices) (s : PorkZongziSales) : Prop :=
  s.quantity = baseQuantity - 2 * (s.price - basePrice)

/-- The profit function for pork zongzi -/
def porkZongziProfit (z : ZongziPrices) (s : PorkZongziSales) : Prop :=
  s.profit = (s.price - z.porkWholesale) * s.quantity

/-- The main theorem stating the maximum profit -/
theorem max_profit_pork_zongzi (z : ZongziPrices) (s : PorkZongziSales) :
  zongziConditions z →
  porkZongziDemand 50 100 z s →
  porkZongziProfit z s →
  ∃ maxProfit : ℝ, maxProfit = 1800 ∧ ∀ s', porkZongziProfit z s' → s'.profit ≤ maxProfit :=
sorry

end max_profit_pork_zongzi_l42_4271


namespace restaurant_menu_fraction_l42_4277

theorem restaurant_menu_fraction (total_dishes : ℕ) 
  (vegan_dishes : ℕ) 
  (vegan_with_gluten : ℕ) 
  (low_sugar_gluten_free_vegan : ℕ) :
  vegan_dishes = total_dishes / 4 →
  vegan_dishes = 6 →
  vegan_with_gluten = 4 →
  low_sugar_gluten_free_vegan = 1 →
  low_sugar_gluten_free_vegan = total_dishes / 24 :=
by sorry

end restaurant_menu_fraction_l42_4277


namespace probability_ratio_l42_4273

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The probability of drawing 5 slips with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

/-- The probability of drawing 2 slips with one number and 3 slips with a different number -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 3 : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

theorem probability_ratio :
  q / p = 450 := by sorry

end probability_ratio_l42_4273


namespace survivor_quitters_probability_l42_4243

/-- The probability that all three quitters are from the same tribe in a Survivor-like scenario -/
theorem survivor_quitters_probability (n : ℕ) (k : ℕ) (q : ℕ) : 
  n = 20 → -- Total number of contestants
  k = 10 → -- Number of contestants in each tribe
  q = 3 →  -- Number of quitters
  (n = 2 * k) → -- Two equally sized tribes
  (Fintype.card {s : Finset (Fin n) // s.card = q ∧ (∀ i ∈ s, i < k) } +
   Fintype.card {s : Finset (Fin n) // s.card = q ∧ (∀ i ∈ s, k ≤ i) }) /
  Fintype.card {s : Finset (Fin n) // s.card = q} = 20 / 95 :=
by sorry


end survivor_quitters_probability_l42_4243
