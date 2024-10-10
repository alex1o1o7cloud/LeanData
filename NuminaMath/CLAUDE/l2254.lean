import Mathlib

namespace sum_of_powers_equals_negative_one_l2254_225423

theorem sum_of_powers_equals_negative_one :
  -1^2010 + (-1)^2011 + 1^2012 - 1^2013 + (-1)^2014 = -1 := by
  sorry

end sum_of_powers_equals_negative_one_l2254_225423


namespace quadratic_function_range_l2254_225434

theorem quadratic_function_range (m : ℝ) : 
  (∀ x : ℝ, -1 < x → x < 0 → x^2 - 4*m*x + 3 > 1) → m > -3/4 := by
  sorry

end quadratic_function_range_l2254_225434


namespace geometric_series_common_ratio_l2254_225495

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 18/49
  let a₃ : ℚ := 162/343
  let r : ℚ := a₂ / a₁
  r = 63/98 := by sorry

end geometric_series_common_ratio_l2254_225495


namespace equation_solution_l2254_225435

theorem equation_solution :
  ∃ x : ℝ, (5 * x - 3 * x = 360 + 6 * (x + 4)) ∧ (x = -96) :=
by
  sorry

end equation_solution_l2254_225435


namespace james_car_sale_l2254_225483

/-- The percentage at which James sold his car -/
def sell_percentage : ℝ → Prop := λ P =>
  let old_car_value : ℝ := 20000
  let new_car_sticker : ℝ := 30000
  let new_car_discount : ℝ := 0.9
  let out_of_pocket : ℝ := 11000
  new_car_sticker * new_car_discount - old_car_value * (P / 100) = out_of_pocket

theorem james_car_sale : 
  sell_percentage 80 := by sorry

end james_car_sale_l2254_225483


namespace no_coin_solution_l2254_225479

theorem no_coin_solution : ¬∃ (x y z : ℕ), 
  x + y + z = 50 ∧ 10 * x + 34 * y + 62 * z = 910 := by
  sorry

end no_coin_solution_l2254_225479


namespace largest_valid_factor_of_130000_l2254_225425

/-- A function that checks if a natural number contains the digit 0 or 5 --/
def containsZeroOrFive (n : ℕ) : Prop := sorry

/-- The largest factor of 130000 that does not contain the digit 0 or 5 --/
def largestValidFactor : ℕ := 26

theorem largest_valid_factor_of_130000 :
  (largestValidFactor ∣ 130000) ∧ 
  ¬containsZeroOrFive largestValidFactor ∧
  ∀ k : ℕ, k > largestValidFactor → (k ∣ 130000) → containsZeroOrFive k := by sorry

end largest_valid_factor_of_130000_l2254_225425


namespace total_population_l2254_225450

/-- The population of New England -/
def new_england_pop : ℕ := 2100000

/-- The population of New York -/
def new_york_pop : ℕ := (2 * new_england_pop) / 3

/-- The population of Pennsylvania -/
def pennsylvania_pop : ℕ := (3 * new_england_pop) / 2

/-- The combined population of Maryland and New Jersey -/
def md_nj_pop : ℕ := new_england_pop + new_england_pop / 5

/-- Theorem stating the total population of all five states -/
theorem total_population : 
  new_york_pop + new_england_pop + pennsylvania_pop + md_nj_pop = 9170000 := by
  sorry

end total_population_l2254_225450


namespace min_distance_to_line_l2254_225496

theorem min_distance_to_line (x y : ℝ) (h : 6 * x + 8 * y - 1 = 0) :
  ∃ (min_val : ℝ), min_val = 7 / 10 ∧
  ∀ (x' y' : ℝ), 6 * x' + 8 * y' - 1 = 0 →
    Real.sqrt (x'^2 + y'^2 - 2*y' + 1) ≥ min_val :=
by sorry

end min_distance_to_line_l2254_225496


namespace min_beans_betty_buys_l2254_225416

/-- The minimum number of pounds of beans Betty could buy given the conditions on rice and beans -/
theorem min_beans_betty_buys (r b : ℝ) 
  (h1 : r ≥ 4 + 2 * b) 
  (h2 : r ≤ 3 * b) : 
  b ≥ 4 := by
sorry

end min_beans_betty_buys_l2254_225416


namespace negation_of_both_even_l2254_225440

theorem negation_of_both_even (a b : ℤ) :
  ¬(Even a ∧ Even b) ↔ ¬(Even a ∧ Even b) :=
by sorry

end negation_of_both_even_l2254_225440


namespace all_expressions_distinct_exactly_five_distinct_expressions_l2254_225475

/-- Represents the different ways to parenthesize 3^(3^(3^3)) -/
inductive ExpressionType
  | Type1  -- 3^(3^(3^3))
  | Type2  -- 3^((3^3)^3)
  | Type3  -- ((3^3)^3)^3
  | Type4  -- (3^(3^3))^3
  | Type5  -- (3^3)^(3^3)

/-- Evaluates the expression based on its type -/
noncomputable def evaluate (e : ExpressionType) : ℕ :=
  match e with
  | ExpressionType.Type1 => 3^(3^(3^3))
  | ExpressionType.Type2 => 3^((3^3)^3)
  | ExpressionType.Type3 => ((3^3)^3)^3
  | ExpressionType.Type4 => (3^(3^3))^3
  | ExpressionType.Type5 => (3^3)^(3^3)

/-- Theorem stating that all expression types result in distinct values -/
theorem all_expressions_distinct :
  ∀ (e1 e2 : ExpressionType), e1 ≠ e2 → evaluate e1 ≠ evaluate e2 := by
  sorry

/-- Theorem stating that there are exactly 5 distinct ways to parenthesize the expression -/
theorem exactly_five_distinct_expressions :
  ∃! (s : Finset ExpressionType), (∀ e, e ∈ s) ∧ s.card = 5 := by
  sorry

end all_expressions_distinct_exactly_five_distinct_expressions_l2254_225475


namespace log_expression_equals_one_l2254_225414

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- Define the common logarithm (base 10) function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_one :
  2 * lg (Real.sqrt 2) + log2 5 * lg 2 = 1 := by
  sorry

end log_expression_equals_one_l2254_225414


namespace defective_product_selection_l2254_225433

def total_products : ℕ := 10
def qualified_products : ℕ := 8
def defective_products : ℕ := 2
def products_to_select : ℕ := 3

theorem defective_product_selection :
  (Nat.choose total_products products_to_select - 
   Nat.choose qualified_products products_to_select) = 64 := by
  sorry

end defective_product_selection_l2254_225433


namespace seventh_equation_properties_l2254_225443

/-- Defines the last number on the left side of the nth equation -/
def last_left (n : ℕ) : ℕ := 3 * n - 2

/-- Defines the result on the right side of the nth equation -/
def right_result (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- Defines the sum of consecutive integers from n to (3n-2) -/
def left_sum (n : ℕ) : ℕ := 
  (n + last_left n) * (last_left n - n + 1) / 2

theorem seventh_equation_properties :
  last_left 7 = 19 ∧ right_result 7 = 169 ∧ left_sum 7 = right_result 7 := by
  sorry

end seventh_equation_properties_l2254_225443


namespace zeroth_power_of_nonzero_rational_l2254_225427

theorem zeroth_power_of_nonzero_rational (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end zeroth_power_of_nonzero_rational_l2254_225427


namespace cube_diagonal_l2254_225457

theorem cube_diagonal (s : ℝ) (h : s > 0) (eq : s^3 + 36*s = 12*s^2) : 
  Real.sqrt (3 * s^2) = 6 * Real.sqrt 3 := by
sorry

end cube_diagonal_l2254_225457


namespace valid_arrangements_count_l2254_225471

/-- The number of available colors for glass panes -/
def num_colors : ℕ := 10

/-- The number of panes in the window frame -/
def num_panes : ℕ := 4

/-- A function that calculates the number of valid arrangements -/
def valid_arrangements (colors : ℕ) (panes : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of valid arrangements is 3430 -/
theorem valid_arrangements_count :
  valid_arrangements num_colors num_panes = 3430 := by
  sorry

end valid_arrangements_count_l2254_225471


namespace coeff_x3_is_30_l2254_225445

/-- The coefficient of x^3 in the expansion of (2x-1)(1/x + x)^6 -/
def coeff_x3 : ℤ := 30

/-- The expression (2x-1)(1/x + x)^6 -/
def expression (x : ℚ) : ℚ := (2*x - 1) * (1/x + x)^6

theorem coeff_x3_is_30 : coeff_x3 = 30 := by sorry

end coeff_x3_is_30_l2254_225445


namespace count_negative_rationals_l2254_225421

def rational_set : Finset ℚ := {-1/2, 5, 0, -(-3), -2, -|-25|}

theorem count_negative_rationals : 
  (rational_set.filter (λ x => x < 0)).card = 3 := by sorry

end count_negative_rationals_l2254_225421


namespace two_colonies_growth_time_l2254_225419

/-- Represents the number of days it takes for a colony to reach the habitat's limit -/
def daysToLimit : ℕ := 21

/-- Represents the daily growth factor of a bacteria colony -/
def growthFactor : ℕ := 2

/-- Represents the number of initial colonies -/
def initialColonies : ℕ := 2

theorem two_colonies_growth_time (daysToLimit : ℕ) (growthFactor : ℕ) (initialColonies : ℕ) :
  daysToLimit = 21 ∧ growthFactor = 2 ∧ initialColonies = 2 →
  daysToLimit = 21 :=
by sorry

end two_colonies_growth_time_l2254_225419


namespace tank_fill_time_is_30_l2254_225499

/-- Represents the time it takes to fill a tank given two pipes with different fill/empty rates and a specific operating scenario. -/
def tank_fill_time (fill_rate_A : ℚ) (empty_rate_B : ℚ) (both_open_time : ℚ) : ℚ :=
  let net_fill_rate := fill_rate_A - empty_rate_B
  let filled_portion := net_fill_rate * both_open_time
  let remaining_portion := 1 - filled_portion
  both_open_time + remaining_portion / fill_rate_A

/-- Theorem stating that under the given conditions, the tank will be filled in 30 minutes. -/
theorem tank_fill_time_is_30 :
  tank_fill_time (1/16) (1/24) 21 = 30 :=
by sorry

end tank_fill_time_is_30_l2254_225499


namespace stream_speed_l2254_225476

theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 120)
  (h2 : upstream_distance = 60)
  (h3 : time = 2) :
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = downstream_distance / time ∧
    boat_speed - stream_speed = upstream_distance / time ∧
    stream_speed = 15 := by
sorry

end stream_speed_l2254_225476


namespace number_relations_l2254_225468

theorem number_relations :
  (∃ x : ℤ, x = -2 - 4 ∧ x = -6) ∧
  (∃ y : ℤ, y = -5 + 3 ∧ y = -2) := by
  sorry

end number_relations_l2254_225468


namespace biggest_number_is_five_l2254_225452

def yoongi_number : ℕ := 4
def jungkook_number : ℕ := 6 - 3
def yuna_number : ℕ := 5

theorem biggest_number_is_five :
  max yoongi_number (max jungkook_number yuna_number) = yuna_number :=
by sorry

end biggest_number_is_five_l2254_225452


namespace complement_of_M_in_U_l2254_225413

def U : Finset ℤ := {0, -1, -2, -3, -4}
def M : Finset ℤ := {0, -1, -2}

theorem complement_of_M_in_U : U \ M = {-3, -4} := by
  sorry

end complement_of_M_in_U_l2254_225413


namespace perpendicular_tangents_ratio_l2254_225487

/-- The slope of the tangent line to y = x³ at x = 1 -/
def tangent_slope : ℝ := 3

/-- The line equation ax - by - 2 = 0 -/
def line_equation (a b : ℝ) (x y : ℝ) : Prop :=
  a * x - b * y - 2 = 0

/-- The curve equation y = x³ -/
def curve_equation (x y : ℝ) : Prop :=
  y = x^3

theorem perpendicular_tangents_ratio (a b : ℝ) :
  line_equation a b 1 1 ∧ 
  curve_equation 1 1 ∧
  (a / b) * tangent_slope = -1 →
  a / b = -1/3 :=
sorry

end perpendicular_tangents_ratio_l2254_225487


namespace triangle_angle_sum_l2254_225436

theorem triangle_angle_sum (A B C : ℝ) : 
  (0 < A) → (0 < B) → (0 < C) →
  (A + B = 90) → (A + B + C = 180) →
  C = 90 := by
sorry

end triangle_angle_sum_l2254_225436


namespace eventually_all_zero_l2254_225405

/-- Represents a quadruple of integers -/
structure Quadruple where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Generates the next quadruple in the sequence -/
def nextQuadruple (q : Quadruple) : Quadruple := {
  a := |q.a - q.b|
  b := |q.b - q.c|
  c := |q.c - q.d|
  d := |q.d - q.a|
}

/-- Checks if all elements in a quadruple are zero -/
def isAllZero (q : Quadruple) : Prop :=
  q.a = 0 ∧ q.b = 0 ∧ q.c = 0 ∧ q.d = 0

/-- Theorem: The sequence will eventually reach all zeros -/
theorem eventually_all_zero (q₀ : Quadruple) : 
  ∃ n : ℕ, isAllZero ((nextQuadruple^[n]) q₀) :=
sorry


end eventually_all_zero_l2254_225405


namespace hyperbola_eccentricity_l2254_225494

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → e = Real.sqrt 13 / 3) :=
by sorry

end hyperbola_eccentricity_l2254_225494


namespace brother_sister_age_diff_l2254_225456

/-- The age difference between Mandy's brother and sister -/
def age_difference (mandy_age brother_age_factor sister_mandy_diff : ℕ) : ℕ :=
  brother_age_factor * mandy_age - (mandy_age + sister_mandy_diff)

/-- Theorem stating the age difference between Mandy's brother and sister -/
theorem brother_sister_age_diff :
  ∀ (mandy_age brother_age_factor sister_mandy_diff : ℕ),
    mandy_age = 3 →
    brother_age_factor = 4 →
    sister_mandy_diff = 4 →
    age_difference mandy_age brother_age_factor sister_mandy_diff = 5 :=
by
  sorry


end brother_sister_age_diff_l2254_225456


namespace least_subtraction_for_divisibility_l2254_225497

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 26 ∧ 
  (99 ∣ (12702 - x)) ∧ 
  (∀ (y : ℕ), y < x → ¬(99 ∣ (12702 - y))) := by
  sorry

end least_subtraction_for_divisibility_l2254_225497


namespace geometric_series_problem_l2254_225492

theorem geometric_series_problem (x y : ℝ) (h : y ≠ 1.375) :
  (∑' n, x / y^n) = 10 →
  (∑' n, x / (x - 2*y)^n) = 5/4 := by
sorry

end geometric_series_problem_l2254_225492


namespace triangle_properties_l2254_225442

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle given by the dot product formula -/
def area_dot_product (t : Triangle) : ℝ := (Real.sqrt 3 / 2) * (t.b * t.c * Real.cos t.A)

/-- The area of the triangle given by the sine formula -/
def area_sine (t : Triangle) : ℝ := (1 / 2) * t.b * t.c * Real.sin t.A

theorem triangle_properties (t : Triangle) 
  (h1 : area_dot_product t = area_sine t) 
  (h2 : t.b + t.c = 5) 
  (h3 : t.a = Real.sqrt 7) : 
  t.A = π / 3 ∧ area_sine t = (3 * Real.sqrt 3) / 2 := by
  sorry

end

end triangle_properties_l2254_225442


namespace cos_four_arccos_two_fifths_l2254_225446

theorem cos_four_arccos_two_fifths :
  Real.cos (4 * Real.arccos (2/5)) = -47/625 := by
  sorry

end cos_four_arccos_two_fifths_l2254_225446


namespace max_apple_recipients_l2254_225432

theorem max_apple_recipients : ∃ n : ℕ, n = 13 ∧ 
  (∀ k : ℕ, k > n → k * (k + 1) > 200) ∧
  (n * (n + 1) ≤ 200) := by
  sorry

end max_apple_recipients_l2254_225432


namespace regular_polygon_sides_l2254_225477

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) : 
  (180 * (n - 2) : ℝ) = 156 * n ↔ n = 15 := by sorry

end regular_polygon_sides_l2254_225477


namespace students_playing_both_sports_l2254_225406

theorem students_playing_both_sports (total : ℕ) (hockey : ℕ) (basketball : ℕ) (neither : ℕ) :
  total = 50 →
  hockey = 30 →
  basketball = 35 →
  neither = 10 →
  hockey + basketball - (total - neither) = 25 :=
by sorry

end students_playing_both_sports_l2254_225406


namespace abc_problem_l2254_225451

def base_6_value (a b : ℕ) : ℕ := a * 6 + b

theorem abc_problem (A B C : ℕ) : 
  (0 < A) → (A ≤ 5) →
  (0 < B) → (B ≤ 5) →
  (0 < C) → (C ≤ 5) →
  (A ≠ B) → (B ≠ C) → (A ≠ C) →
  (base_6_value A B + A = base_6_value B A) →
  (base_6_value A B + B = base_6_value C 1) →
  (A = 5 ∧ B = 5 ∧ C = 1) := by
sorry

end abc_problem_l2254_225451


namespace sqrt_four_squared_l2254_225460

theorem sqrt_four_squared : (Real.sqrt 4)^2 = 4 := by
  sorry

end sqrt_four_squared_l2254_225460


namespace bathtub_volume_l2254_225480

/-- Represents the problem of calculating the volume of a bathtub filled with jello --/
def BathtubProblem (jello_per_pound : ℚ) (gallons_per_cubic_foot : ℚ) (pounds_per_gallon : ℚ) 
                   (cost_per_tablespoon : ℚ) (total_spent : ℚ) : Prop :=
  let tablespoons := total_spent / cost_per_tablespoon
  let pounds_of_water := tablespoons / jello_per_pound
  let gallons_of_water := pounds_of_water / pounds_per_gallon
  let cubic_feet := gallons_of_water / gallons_per_cubic_foot
  cubic_feet = 6

/-- The main theorem stating that given the problem conditions, the bathtub holds 6 cubic feet of water --/
theorem bathtub_volume : 
  BathtubProblem (3/2) (15/2) 8 (1/2) 270 := by
  sorry

#check bathtub_volume

end bathtub_volume_l2254_225480


namespace earth_land_area_scientific_notation_l2254_225438

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a given number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

theorem earth_land_area_scientific_notation :
  let earthLandArea : ℝ := 149000000
  let scientificNotation := toScientificNotation earthLandArea 3
  scientificNotation.coefficient = 1.49 ∧ scientificNotation.exponent = 8 := by
  sorry

end earth_land_area_scientific_notation_l2254_225438


namespace simplify_fraction_l2254_225462

theorem simplify_fraction (x y : ℚ) (hx : x = 2) (hy : y = 3) :
  (18 * x^3 * y^2) / (9 * x^2 * y^4) = 4 / 9 := by
  sorry

end simplify_fraction_l2254_225462


namespace equation_solution_l2254_225429

theorem equation_solution (a : ℝ) : 
  ((4 - 2) / 2 + a = 4) → a = 3 := by
  sorry

end equation_solution_l2254_225429


namespace factorization_equality_l2254_225437

theorem factorization_equality (a b : ℝ) : (a - b)^2 + 6*(b - a) + 9 = (a - b - 3)^2 := by
  sorry

end factorization_equality_l2254_225437


namespace apple_lovers_joined_correct_number_joined_l2254_225465

theorem apple_lovers_joined (total_apples : ℕ) (initial_per_person : ℕ) (decrease : ℕ) : ℕ :=
  let initial_group_size := total_apples / initial_per_person
  let final_per_person := initial_per_person - decrease
  let final_group_size := total_apples / final_per_person
  final_group_size - initial_group_size

theorem correct_number_joined :
  apple_lovers_joined 1430 22 9 = 45 :=
by sorry

end apple_lovers_joined_correct_number_joined_l2254_225465


namespace orchid_bushes_after_planting_l2254_225449

/-- The number of orchid bushes in a park after planting new ones -/
def total_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The total number of orchid bushes after planting is the sum of initial and planted bushes -/
theorem orchid_bushes_after_planting (initial : ℕ) (planted : ℕ) :
  total_bushes initial planted = initial + planted := by
  sorry

/-- Example with given values -/
example : total_bushes 2 4 = 6 := by
  sorry

end orchid_bushes_after_planting_l2254_225449


namespace cow_heart_ratio_l2254_225441

/-- The number of hearts on a standard deck of 52 playing cards -/
def hearts_on_deck : ℕ := 13

/-- The cost of each cow in dollars -/
def cost_per_cow : ℕ := 200

/-- The total cost of all cows in dollars -/
def total_cost : ℕ := 83200

/-- The number of cows in Devonshire -/
def num_cows : ℕ := total_cost / cost_per_cow

theorem cow_heart_ratio :
  num_cows / hearts_on_deck = 32 :=
sorry

end cow_heart_ratio_l2254_225441


namespace preimage_of_3_1_l2254_225411

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

theorem preimage_of_3_1 : f⁻¹' {(3, 1)} = {(2, 1)} := by
  sorry

end preimage_of_3_1_l2254_225411


namespace calculate_individual_tip_l2254_225424

/-- Calculates the individual tip amount for a group dining out -/
theorem calculate_individual_tip (julie_order : ℚ) (letitia_order : ℚ) (anton_order : ℚ) 
  (tip_rate : ℚ) (h1 : julie_order = 10) (h2 : letitia_order = 20) (h3 : anton_order = 30) 
  (h4 : tip_rate = 0.2) : 
  (julie_order + letitia_order + anton_order) * tip_rate / 3 = 4 := by
  sorry

end calculate_individual_tip_l2254_225424


namespace train_speed_l2254_225404

/-- Proves that a train of length 400 meters crossing a pole in 12 seconds has a speed of 120 km/hr -/
theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) : 
  length = 400 ∧ time = 12 → speed = (length / 1000) / (time / 3600) → speed = 120 := by
  sorry

end train_speed_l2254_225404


namespace arithmetic_sequence_sum_l2254_225491

/-- The sum of an arithmetic sequence with first term a₁ = k^2 - k + 1 and
    common difference d = 1, for the first 2k terms. -/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let a₁ := k^2 - k + 1
  let d := 1
  let n := 2 * k
  let S := n * (2 * a₁ + (n - 1) * d) / 2
  S = 2 * k^3 + k := by sorry

end arithmetic_sequence_sum_l2254_225491


namespace inequality_pattern_l2254_225428

theorem inequality_pattern (x : ℝ) (a : ℝ) 
  (h_x : x > 0)
  (h1 : x + 1/x ≥ 2)
  (h2 : x + 4/x^2 ≥ 3)
  (h3 : x + 27/x^3 ≥ 4)
  (h4 : x + a/x^4 ≥ 5) :
  a = 4^4 := by
  sorry

end inequality_pattern_l2254_225428


namespace regular_polygon_sides_l2254_225407

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → interior_angle = 144 → (n - 2) * 180 = n * interior_angle → n = 10 := by
  sorry

end regular_polygon_sides_l2254_225407


namespace partner_b_investment_l2254_225493

/-- Represents the investment and profit share of a partner in a partnership. -/
structure Partner where
  investment : ℝ
  profitShare : ℝ

/-- Represents a partnership with three partners. -/
structure Partnership where
  a : Partner
  b : Partner
  c : Partner

/-- Theorem stating that given the conditions of the problem, partner b's investment is $21000. -/
theorem partner_b_investment (p : Partnership)
  (h1 : p.a.investment = 15000)
  (h2 : p.c.investment = 27000)
  (h3 : p.b.profitShare = 1540)
  (h4 : p.a.profitShare = 1100)
  : p.b.investment = 21000 := by
  sorry

#check partner_b_investment

end partner_b_investment_l2254_225493


namespace rose_difference_is_34_l2254_225498

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := 58

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The difference in the number of red roses between Mrs. Santiago and Mrs. Garrett -/
def rose_difference : ℕ := santiago_roses - garrett_roses

theorem rose_difference_is_34 : rose_difference = 34 := by
  sorry

end rose_difference_is_34_l2254_225498


namespace circle_configuration_diameter_l2254_225444

/-- Given a configuration of circles as described, prove the diameter length --/
theorem circle_configuration_diameter : 
  ∀ (r s : ℝ) (shaded_area circle_c_area : ℝ),
  r > 0 → s > 0 →
  shaded_area = 39 * Real.pi →
  circle_c_area = 9 * Real.pi →
  shaded_area = (Real.pi / 2) * ((r + s)^2 - r^2 - s^2) - circle_c_area →
  2 * (r + s) = 32 := by
  sorry

end circle_configuration_diameter_l2254_225444


namespace simplify_and_rationalize_l2254_225448

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 14) =
  3 * Real.sqrt 420 / 42 := by
  sorry

end simplify_and_rationalize_l2254_225448


namespace ladies_walking_distance_l2254_225469

/-- The total distance walked by a group of ladies over a period of days. -/
def total_distance (
  group_size : ℕ
  ) (
  group_distance : ℝ
  ) (
  jamie_extra : ℝ
  ) (
  sue_extra : ℝ
  ) (
  days : ℕ
  ) : ℝ :=
  group_size * group_distance * days + jamie_extra * days + sue_extra * days

/-- Proof that the total distance walked by the ladies is 36 miles. -/
theorem ladies_walking_distance :
  let group_size : ℕ := 5
  let group_distance : ℝ := 3
  let jamie_extra : ℝ := 2
  let sue_extra : ℝ := jamie_extra / 2
  let days : ℕ := 6
  total_distance group_size group_distance jamie_extra sue_extra days = 36 := by
  sorry


end ladies_walking_distance_l2254_225469


namespace like_terms_proof_l2254_225410

/-- Given that -3x^(m-1)y^3 and 4xy^(m+n) are like terms, prove that m = 2 and n = 1 -/
theorem like_terms_proof (m n : ℤ) : 
  (∀ x y : ℝ, ∃ k : ℝ, -3 * x^(m-1) * y^3 = k * 4 * x * y^(m+n)) → 
  m = 2 ∧ n = 1 := by
sorry

end like_terms_proof_l2254_225410


namespace equation_equivalence_l2254_225417

-- Define the original equation
def original_equation (x y : ℝ) : Prop := 2 * x - 3 * y - 4 = 0

-- Define the intercept form
def intercept_form (x y : ℝ) : Prop := x / 2 + y / (-4/3) = 1

-- Theorem stating the equivalence of the two forms
theorem equation_equivalence :
  ∀ x y : ℝ, original_equation x y ↔ intercept_form x y :=
by sorry

end equation_equivalence_l2254_225417


namespace power_mod_eleven_l2254_225461

theorem power_mod_eleven : 3^225 ≡ 1 [MOD 11] := by sorry

end power_mod_eleven_l2254_225461


namespace vector_parallel_cosine_value_l2254_225401

theorem vector_parallel_cosine_value (θ : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : (9/10, 3) = (Real.cos (θ + π/6), 2)) : 
  Real.cos θ = (4 + 3 * Real.sqrt 3) / 10 := by
  sorry

end vector_parallel_cosine_value_l2254_225401


namespace geometric_sequence_sum_l2254_225430

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 :=
by
  sorry

end geometric_sequence_sum_l2254_225430


namespace constant_value_proof_l2254_225485

/-- The coefficient of x in the expansion of (x - a/x)(1 - √x)^6 -/
def coefficient_of_x (a : ℝ) : ℝ := 1 - 15 * a

/-- The theorem stating that a = -2 when the coefficient of x is 31 -/
theorem constant_value_proof (a : ℝ) : coefficient_of_x a = 31 → a = -2 := by
  sorry

end constant_value_proof_l2254_225485


namespace line_equation_through_point_with_slope_one_l2254_225409

/-- The equation of a line passing through (-1, -1) with slope 1 is y = x -/
theorem line_equation_through_point_with_slope_one :
  ∀ (x y : ℝ), (y + 1 = 1 * (x + 1)) ↔ (y = x) :=
by sorry

end line_equation_through_point_with_slope_one_l2254_225409


namespace max_brownie_pieces_l2254_225473

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The pan dimensions -/
def pan : Rectangle := { length := 24, width := 20 }

/-- The brownie piece dimensions -/
def piece : Rectangle := { length := 4, width := 3 }

/-- Theorem: The maximum number of brownie pieces that can be cut from the pan is 40 -/
theorem max_brownie_pieces : (area pan) / (area piece) = 40 := by
  sorry

end max_brownie_pieces_l2254_225473


namespace square_of_binomial_constant_l2254_225490

theorem square_of_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 164*x + c = (x + a)^2) → c = 6724 := by
  sorry

end square_of_binomial_constant_l2254_225490


namespace min_questions_to_determine_l2254_225459

def questions_to_determine (x : ℕ) : ℕ :=
  if x ≥ 10 ∧ x ≤ 19 then
    if x ≤ 14 then
      if x ≤ 12 then
        if x = 11 then 3 else 3
      else
        if x = 13 then 3 else 3
    else
      if x ≤ 17 then
        if x ≤ 16 then
          if x = 15 then 4 else 4
        else 3
      else
        if x = 18 then 3 else 3
  else 0

theorem min_questions_to_determine :
  ∀ x : ℕ, x ≥ 10 ∧ x ≤ 19 → questions_to_determine x ≤ 3 ∧
  (∀ y : ℕ, y ≥ 10 ∧ y ≤ 19 ∧ y ≠ x → ∃ q : ℕ, q < questions_to_determine x ∧
    (∀ z : ℕ, z ≥ 10 ∧ z ≤ 19 → questions_to_determine z < q → z ≠ x ∧ z ≠ y)) :=
sorry

end min_questions_to_determine_l2254_225459


namespace flower_bed_circumference_l2254_225472

/-- Given a square garden with a circular flower bed, prove the circumference of the flower bed -/
theorem flower_bed_circumference 
  (a p t : ℝ) 
  (h1 : a > 0) 
  (h2 : p > 0) 
  (h3 : t > 0) 
  (h4 : a = 2 * p + 14.25) 
  (h5 : ∃ s : ℝ, s > 0 ∧ a = s^2 ∧ p = 4 * s) 
  (h6 : ∃ r : ℝ, r > 0 ∧ r = s / 4 ∧ t = a + π * r^2) : 
  ∃ C : ℝ, C = 4.75 * π := by sorry

end flower_bed_circumference_l2254_225472


namespace investment_value_l2254_225426

/-- Proves that the value of the larger investment is $1500 given the specified conditions. -/
theorem investment_value (x : ℝ) : 
  (0.07 * 500 + 0.27 * x = 0.22 * (500 + x)) → x = 1500 := by
  sorry

end investment_value_l2254_225426


namespace angle_of_inclination_range_l2254_225420

theorem angle_of_inclination_range (θ : Real) (x y : Real) :
  x - y * Real.sin θ + 1 = 0 →
  ∃ α, α ∈ Set.Icc (π/4) (3*π/4) ∧
       (α = π/2 ∨ Real.tan α = 1 / Real.sin θ) :=
by sorry

end angle_of_inclination_range_l2254_225420


namespace remove_horizontal_eliminates_triangles_fifteen_is_minimum_removal_l2254_225431

/-- Represents a triangular figure constructed with toothpicks -/
structure TriangularFigure where
  total_toothpicks : ℕ
  horizontal_toothpicks : ℕ
  has_upward_triangles : Bool
  has_downward_triangles : Bool

/-- Represents the number of toothpicks that need to be removed -/
def toothpicks_to_remove (figure : TriangularFigure) : ℕ := figure.horizontal_toothpicks

/-- Theorem stating that removing horizontal toothpicks eliminates all triangles -/
theorem remove_horizontal_eliminates_triangles (figure : TriangularFigure) 
  (h1 : figure.total_toothpicks = 45)
  (h2 : figure.horizontal_toothpicks = 15)
  (h3 : figure.has_upward_triangles = true)
  (h4 : figure.has_downward_triangles = true) :
  toothpicks_to_remove figure = 15 ∧ 
  (∀ n : ℕ, n < 15 → ∃ triangle_remains : Bool, triangle_remains = true) := by
  sorry

/-- Theorem stating that 15 is the minimum number of toothpicks to remove -/
theorem fifteen_is_minimum_removal (figure : TriangularFigure)
  (h1 : figure.total_toothpicks = 45)
  (h2 : figure.horizontal_toothpicks = 15)
  (h3 : figure.has_upward_triangles = true)
  (h4 : figure.has_downward_triangles = true) :
  ∀ n : ℕ, n < 15 → ∃ triangle_remains : Bool, triangle_remains = true := by
  sorry

end remove_horizontal_eliminates_triangles_fifteen_is_minimum_removal_l2254_225431


namespace complex_magnitude_l2254_225464

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end complex_magnitude_l2254_225464


namespace line_passes_through_fixed_point_l2254_225489

/-- The line equation passes through a fixed point for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end line_passes_through_fixed_point_l2254_225489


namespace consecutive_four_plus_one_is_square_l2254_225474

theorem consecutive_four_plus_one_is_square (n : ℕ) (h : n ≥ 1) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end consecutive_four_plus_one_is_square_l2254_225474


namespace boat_stream_speed_ratio_l2254_225447

/-- Proves that the ratio of boat speed to stream speed is 3:1 given the time relation -/
theorem boat_stream_speed_ratio 
  (D : ℝ) -- Distance rowed
  (B : ℝ) -- Speed of the boat in still water
  (S : ℝ) -- Speed of the stream
  (h_positive : D > 0 ∧ B > 0 ∧ S > 0) -- Positive distances and speeds
  (h_time_ratio : D / (B - S) = 2 * (D / (B + S))) -- Time against stream is twice time with stream
  : B / S = 3 := by
  sorry

end boat_stream_speed_ratio_l2254_225447


namespace glued_cubes_surface_area_l2254_225484

/-- Represents a 3D shape formed by two glued cubes -/
structure GluedCubes where
  large_cube_side : ℝ
  small_cube_side : ℝ
  glued : Bool

/-- Calculate the surface area of the GluedCubes shape -/
def surface_area (shape : GluedCubes) : ℝ :=
  let large_cube_area := 6 * shape.large_cube_side ^ 2
  let small_cube_area := 5 * shape.small_cube_side ^ 2
  large_cube_area + small_cube_area

/-- The theorem stating that the surface area of the specific GluedCubes shape is 74 -/
theorem glued_cubes_surface_area :
  let shape := GluedCubes.mk 3 1 true
  surface_area shape = 74 := by
  sorry

end glued_cubes_surface_area_l2254_225484


namespace unique_solution_system_l2254_225400

theorem unique_solution_system (x y z : ℂ) :
  x + y + z = 3 ∧
  x^2 + y^2 + z^2 = 3 ∧
  x^3 + y^3 + z^3 = 3 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end unique_solution_system_l2254_225400


namespace surface_area_after_corner_removal_l2254_225470

/-- The surface area of a cube after removing smaller cubes from its corners --/
theorem surface_area_after_corner_removal (edge_length original_cube_edge : ℝ) 
  (h1 : original_cube_edge = 4)
  (h2 : edge_length = 2) :
  6 * original_cube_edge^2 = 
  6 * original_cube_edge^2 - 8 * (3 * edge_length^2 - 3 * edge_length^2) :=
by sorry

end surface_area_after_corner_removal_l2254_225470


namespace largest_multiple_of_15_under_500_l2254_225478

theorem largest_multiple_of_15_under_500 :
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by sorry

end largest_multiple_of_15_under_500_l2254_225478


namespace curtis_farm_egg_laying_hens_l2254_225412

/-- The number of egg-laying hens on Mr. Curtis's farm -/
def egg_laying_hens (total_chickens roosters non_laying_hens : ℕ) : ℕ :=
  total_chickens - roosters - non_laying_hens

/-- Theorem stating the number of egg-laying hens on Mr. Curtis's farm -/
theorem curtis_farm_egg_laying_hens :
  egg_laying_hens 325 28 20 = 277 := by
  sorry

end curtis_farm_egg_laying_hens_l2254_225412


namespace initial_men_is_four_l2254_225488

/-- The number of men initially checking exam papers -/
def initial_men : ℕ := 4

/-- The number of days for the initial group to check papers -/
def initial_days : ℕ := 8

/-- The number of hours per day for the initial group -/
def initial_hours_per_day : ℕ := 5

/-- The number of men in the second group -/
def second_men : ℕ := 2

/-- The number of days for the second group to check papers -/
def second_days : ℕ := 20

/-- The number of hours per day for the second group -/
def second_hours_per_day : ℕ := 8

/-- Theorem stating that the initial number of men is 4 -/
theorem initial_men_is_four :
  initial_men * initial_days * initial_hours_per_day = 
  (second_men * second_days * second_hours_per_day) / 2 :=
by sorry

end initial_men_is_four_l2254_225488


namespace sara_quarters_l2254_225418

/-- The number of quarters Sara has after receiving more from her dad -/
def total_quarters (initial : ℕ) (received : ℕ) : ℕ := initial + received

/-- Theorem stating that Sara now has 70 quarters -/
theorem sara_quarters : total_quarters 21 49 = 70 := by
  sorry

end sara_quarters_l2254_225418


namespace sum_of_solutions_squared_diff_sum_of_solutions_eq_eight_l2254_225402

theorem sum_of_solutions_squared_diff (a c : ℝ) :
  (∀ x : ℝ, (x - a)^2 = c) → 
  (∃ x₁ x₂ : ℝ, (x₁ - a)^2 = c ∧ (x₂ - a)^2 = c ∧ x₁ + x₂ = 2 * a) :=
by sorry

-- The specific problem
theorem sum_of_solutions_eq_eight :
  (∃ x₁ x₂ : ℝ, (x₁ - 4)^2 = 49 ∧ (x₂ - 4)^2 = 49 ∧ x₁ + x₂ = 8) :=
by sorry

end sum_of_solutions_squared_diff_sum_of_solutions_eq_eight_l2254_225402


namespace symmetric_point_coordinates_l2254_225453

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry with respect to the origin
def symmetricToOrigin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Theorem statement
theorem symmetric_point_coordinates :
  let A : Point3D := { x := 2, y := 1, z := 0 }
  let B : Point3D := symmetricToOrigin A
  B.x = -2 ∧ B.y = -1 ∧ B.z = 0 := by sorry

end symmetric_point_coordinates_l2254_225453


namespace max_sum_constrained_l2254_225467

theorem max_sum_constrained (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_constraint : x^2 + y^2 + z^2 + x + 2*y + 3*z = 13/4) :
  x + y + z ≤ 3/2 :=
by sorry

end max_sum_constrained_l2254_225467


namespace triangle_abc_problem_l2254_225422

theorem triangle_abc_problem (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  2 * Real.sin A ^ 2 + 3 * Real.cos (B + C) = 0 →
  S = 5 * Real.sqrt 3 →
  a = Real.sqrt 21 →
  A = π / 3 ∧ b + c = 9 := by
  sorry

end triangle_abc_problem_l2254_225422


namespace polynomial_remainder_l2254_225458

theorem polynomial_remainder (q : ℝ → ℝ) :
  (∃ r : ℝ → ℝ, ∀ x, q x = (x - 2) * r x + 3) →
  (∃ s : ℝ → ℝ, ∀ x, q x = (x + 3) * s x - 9) →
  ∃ t : ℝ → ℝ, ∀ x, q x = (x - 2) * (x + 3) * t x + (12/5 * x - 9/5) :=
by
  sorry

end polynomial_remainder_l2254_225458


namespace dinner_time_calculation_l2254_225463

/-- Calculates the time spent eating dinner during a train ride given the total duration and time spent on other activities. -/
theorem dinner_time_calculation (total_duration reading_time movie_time nap_time : ℕ) 
  (h1 : total_duration = 9)
  (h2 : reading_time = 2)
  (h3 : movie_time = 3)
  (h4 : nap_time = 3) :
  total_duration - (reading_time + movie_time + nap_time) = 1 := by
  sorry

#check dinner_time_calculation

end dinner_time_calculation_l2254_225463


namespace amoeba_fill_time_l2254_225454

def amoeba_population (initial : ℕ) (time : ℕ) : ℕ :=
  initial * 2^time

theorem amoeba_fill_time :
  ∀ (tube_capacity : ℕ),
  tube_capacity > 0 →
  (∃ (t : ℕ), amoeba_population 1 t = tube_capacity) →
  (∃ (s : ℕ), amoeba_population 2 s = tube_capacity ∧ s + 1 = t) :=
by sorry

end amoeba_fill_time_l2254_225454


namespace prob_sum_24_four_dice_is_correct_l2254_225439

/-- The probability of rolling a sum of 24 with four fair, six-sided dice -/
def prob_sum_24_four_dice : ℚ := 1 / 1296

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The sum we're looking for -/
def target_sum : ℕ := 24

/-- Theorem: The probability of rolling a sum of 24 with four fair, six-sided dice is 1/1296 -/
theorem prob_sum_24_four_dice_is_correct : 
  prob_sum_24_four_dice = (1 : ℚ) / sides_per_die ^ num_dice ∧
  target_sum = sides_per_die * num_dice :=
sorry

end prob_sum_24_four_dice_is_correct_l2254_225439


namespace gdp_scientific_notation_l2254_225408

theorem gdp_scientific_notation :
  let gdp_billion : ℝ := 32.07
  let billion : ℝ := 10^9
  let gdp : ℝ := gdp_billion * billion
  ∃ (a : ℝ) (n : ℤ), gdp = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.207 ∧ n = 10 :=
by sorry

end gdp_scientific_notation_l2254_225408


namespace tom_hockey_games_attendance_l2254_225486

/-- The number of hockey games Tom attended over six years -/
def total_games_attended (year1 year2 year3 year4 year5 year6 : ℕ) : ℕ :=
  year1 + year2 + year3 + year4 + year5 + year6

/-- Theorem stating that Tom attended 41 hockey games over six years -/
theorem tom_hockey_games_attendance :
  total_games_attended 4 9 5 10 6 7 = 41 := by
  sorry

end tom_hockey_games_attendance_l2254_225486


namespace sum_of_numerator_and_denominator_l2254_225403

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.4̅5̅ -/
def decimal : ℚ := RepeatingDecimal 4 5

/-- The fraction representation of 0.4̅5̅ in lowest terms -/
def fraction : ℚ := 5 / 11

theorem sum_of_numerator_and_denominator : 
  decimal = fraction ∧ fraction.num + fraction.den = 16 := by sorry

end sum_of_numerator_and_denominator_l2254_225403


namespace unique_factor_solution_l2254_225415

theorem unique_factor_solution (A B C D : ℕ+) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * B = 72 →
  C * D = 72 →
  A + B = C - D →
  A = 4 := by
sorry

end unique_factor_solution_l2254_225415


namespace paper_cut_end_time_l2254_225466

def minutes_per_cut : ℕ := 3
def rest_minutes : ℕ := 1
def start_time : ℕ := 9 * 60 + 40  -- 9:40 in minutes since midnight
def num_cuts : ℕ := 10

def total_time : ℕ := (num_cuts - 1) * (minutes_per_cut + rest_minutes) + minutes_per_cut

def end_time : ℕ := start_time + total_time

theorem paper_cut_end_time :
  (end_time / 60, end_time % 60) = (10, 19) := by
  sorry

end paper_cut_end_time_l2254_225466


namespace jury_duty_ratio_l2254_225482

theorem jury_duty_ratio (jury_selection : ℕ) (trial_duration : ℕ) (jury_deliberation : ℕ) (total_days : ℕ) :
  jury_selection = 2 →
  jury_deliberation = 6 →
  total_days = 19 →
  total_days = jury_selection + trial_duration + jury_deliberation →
  (trial_duration : ℚ) / (jury_selection : ℚ) = 11 / 2 :=
by
  sorry

end jury_duty_ratio_l2254_225482


namespace log_inequality_l2254_225481

theorem log_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) : 
  Real.log (Real.sqrt (x₁ * x₂)) = (Real.log x₁ + Real.log x₂) / 2 ∧
  Real.log (Real.sqrt (x₁ * x₂)) < Real.log ((x₁ + x₂) / 2) := by
  sorry

end log_inequality_l2254_225481


namespace fraction_sum_simplest_form_fraction_simplest_form_l2254_225455

theorem fraction_sum_simplest_form : (7 : ℚ) / 12 + (8 : ℚ) / 15 = (67 : ℚ) / 60 := by
  sorry

theorem fraction_simplest_form : (67 : ℚ) / 60 = (67 : ℚ) / 60 := by
  sorry

end fraction_sum_simplest_form_fraction_simplest_form_l2254_225455
