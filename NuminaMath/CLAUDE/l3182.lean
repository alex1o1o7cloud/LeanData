import Mathlib

namespace line_slope_l3182_318299

/-- Given a line with equation x/4 + y/3 = 2, its slope is -3/4 -/
theorem line_slope (x y : ℝ) : (x / 4 + y / 3 = 2) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) := by
  sorry

end line_slope_l3182_318299


namespace friend_meeting_probability_l3182_318297

/-- The probability that two friends meet given specific conditions -/
theorem friend_meeting_probability : 
  ∀ (wait_time : ℝ) (window : ℝ),
  wait_time > 0 → 
  window > wait_time →
  (∃ (prob : ℝ), 
    prob = (window^2 - 2 * (window - wait_time)^2 / 2) / window^2 ∧ 
    prob = 8/9) := by
  sorry

end friend_meeting_probability_l3182_318297


namespace jens_birds_l3182_318289

theorem jens_birds (ducks chickens : ℕ) : 
  ducks > 4 * chickens →
  ducks = 150 →
  ducks + chickens = 185 →
  ducks - 4 * chickens = 10 := by
sorry

end jens_birds_l3182_318289


namespace backyard_area_l3182_318253

/-- Proves that the area of a rectangular backyard with given conditions is 400 square meters -/
theorem backyard_area (length walk_length perimeter : ℝ) 
  (h1 : length * 30 = 1200)
  (h2 : perimeter * 12 = 1200)
  (h3 : perimeter = 2 * length + 2 * (perimeter / 2 - length)) : 
  length * (perimeter / 2 - length) = 400 := by
  sorry

end backyard_area_l3182_318253


namespace seconds_in_12_5_minutes_l3182_318223

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes to convert -/
def minutes_to_convert : ℚ := 12.5

/-- Theorem: The number of seconds in 12.5 minutes is 750 -/
theorem seconds_in_12_5_minutes :
  (minutes_to_convert * seconds_per_minute : ℚ) = 750 := by sorry

end seconds_in_12_5_minutes_l3182_318223


namespace smallest_positive_equivalent_angle_proof_l3182_318226

/-- The smallest positive angle (in degrees) with the same terminal side as -2002° -/
def smallest_positive_equivalent_angle : ℝ := 158

theorem smallest_positive_equivalent_angle_proof :
  ∃ (k : ℤ), smallest_positive_equivalent_angle = -2002 + 360 * k ∧
  0 < smallest_positive_equivalent_angle ∧
  smallest_positive_equivalent_angle < 360 :=
by
  sorry

end smallest_positive_equivalent_angle_proof_l3182_318226


namespace factorization_problem1_l3182_318294

theorem factorization_problem1 (a b : ℝ) : -3 * a^2 + 6 * a * b - 3 * b^2 = -3 * (a - b)^2 := by
  sorry

end factorization_problem1_l3182_318294


namespace ninth_term_of_arithmetic_sequence_l3182_318269

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem ninth_term_of_arithmetic_sequence 
  (a₁ a₁₇ : ℚ) 
  (h₁ : a₁ = 3/4) 
  (h₁₇ : a₁₇ = 6/7) 
  (h_seq : ∃ d, ∀ n, arithmetic_sequence a₁ d n = a₁ + (n - 1) * d) :
  ∃ d, arithmetic_sequence a₁ d 9 = 45/56 :=
sorry

end ninth_term_of_arithmetic_sequence_l3182_318269


namespace line_through_circle_center_l3182_318238

/-- The center of a circle given by the equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop := 3 * x + y + a = 0

/-- The theorem stating that if the line 3x + y + a = 0 passes through the center of the circle x^2 + y^2 + 2x - 4y = 0, then a = 1 -/
theorem line_through_circle_center (a : ℝ) : 
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end line_through_circle_center_l3182_318238


namespace bill_toilet_paper_usage_l3182_318213

/-- Calculates the number of toilet paper squares used per bathroom visit -/
def toilet_paper_usage (bathroom_visits_per_day : ℕ) (total_rolls : ℕ) (squares_per_roll : ℕ) (total_days : ℕ) : ℕ :=
  (total_rolls * squares_per_roll) / (total_days * bathroom_visits_per_day)

/-- Proves that Bill uses 5 squares of toilet paper per bathroom visit -/
theorem bill_toilet_paper_usage :
  toilet_paper_usage 3 1000 300 20000 = 5 := by
  sorry

end bill_toilet_paper_usage_l3182_318213


namespace addition_subtraction_problem_l3182_318291

theorem addition_subtraction_problem : (0.45 + 52.7) - 0.25 = 52.9 := by
  sorry

end addition_subtraction_problem_l3182_318291


namespace apple_calculation_correct_l3182_318222

/-- Represents the weight difference from the standard weight and its frequency --/
structure WeightDifference :=
  (difference : ℝ)
  (frequency : ℕ)

/-- Calculates the total weight and profit from a batch of apples --/
def apple_calculation (total_boxes : ℕ) (price_per_box : ℝ) (selling_price_per_kg : ℝ) 
  (weight_differences : List WeightDifference) : ℝ × ℝ :=
  sorry

/-- The main theorem stating the correctness of the calculation --/
theorem apple_calculation_correct : 
  let weight_differences := [
    ⟨-0.2, 5⟩, ⟨-0.1, 8⟩, ⟨0, 2⟩, ⟨0.1, 6⟩, ⟨0.2, 8⟩, ⟨0.5, 1⟩
  ]
  let (total_weight, profit) := apple_calculation 400 60 10 weight_differences
  total_weight = 300.9 ∧ profit = 16120 :=
by sorry

end apple_calculation_correct_l3182_318222


namespace total_volume_of_cubes_combined_cube_volume_l3182_318258

theorem total_volume_of_cubes : ℕ → ℕ → ℕ → ℕ → ℕ
  | carl_count, carl_side, kate_count, kate_side =>
    (carl_count * carl_side^3) + (kate_count * kate_side^3)

theorem combined_cube_volume : total_volume_of_cubes 8 2 3 3 = 145 := by
  sorry

end total_volume_of_cubes_combined_cube_volume_l3182_318258


namespace reciprocal_of_2023_l3182_318205

theorem reciprocal_of_2023 : (2023⁻¹ : ℚ) = 1 / 2023 := by
  sorry

end reciprocal_of_2023_l3182_318205


namespace simplify_fraction_1_l3182_318276

theorem simplify_fraction_1 (a b : ℝ) (h : a ≠ b) :
  (a^4 - b^4) / (a^2 - b^2) = a^2 + b^2 := by
sorry

end simplify_fraction_1_l3182_318276


namespace only_f₂_passes_through_origin_l3182_318207

-- Define the functions
def f₁ (x : ℝ) := x + 1
def f₂ (x : ℝ) := x^2
def f₃ (x : ℝ) := (x - 4)^2
noncomputable def f₄ (x : ℝ) := 1/x

-- Theorem statement
theorem only_f₂_passes_through_origin :
  (f₁ 0 ≠ 0) ∧ 
  (f₂ 0 = 0) ∧ 
  (f₃ 0 ≠ 0) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f₄ x| > 1/ε) :=
by sorry

end only_f₂_passes_through_origin_l3182_318207


namespace expression_simplification_l3182_318286

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 - 1) :
  (x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2*x + 1)) = Real.sqrt 5 / 5 := by
  sorry

end expression_simplification_l3182_318286


namespace three_numbers_problem_l3182_318281

theorem three_numbers_problem (x y z : ℤ) : 
  x - y = 12 ∧ 
  (x + y) / 4 = 7 ∧ 
  z = 2 * y ∧ 
  x + z = 24 → 
  x = 20 ∧ y = 8 ∧ z = 16 := by
sorry

end three_numbers_problem_l3182_318281


namespace water_speed_calculation_l3182_318280

/-- Proves that the speed of the water is 8 km/h given the swimming conditions -/
theorem water_speed_calculation (swimming_speed : ℝ) (distance : ℝ) (time : ℝ) :
  swimming_speed = 16 →
  distance = 12 →
  time = 1.5 →
  swimming_speed - (distance / time) = 8 :=
by
  sorry

end water_speed_calculation_l3182_318280


namespace pump_water_in_35_minutes_l3182_318236

theorem pump_water_in_35_minutes : 
  let pump_rate : ℚ := 300  -- gallons per hour
  let time : ℚ := 35 / 60   -- 35 minutes converted to hours
  pump_rate * time = 175
  := by sorry

end pump_water_in_35_minutes_l3182_318236


namespace tangent_line_to_logarithmic_curve_l3182_318227

theorem tangent_line_to_logarithmic_curve (a : ℝ) : 
  (∃ x₀ y₀ : ℝ, 
    y₀ = x₀ + 1 ∧ 
    y₀ = Real.log (x₀ + a) ∧ 
    (Real.exp y₀)⁻¹ = 1) → 
  a = 2 := by
sorry

end tangent_line_to_logarithmic_curve_l3182_318227


namespace log_equation_equals_zero_l3182_318218

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_equals_zero :
  (log10 2)^2 + log10 2 * log10 50 - log10 4 = 0 := by sorry

end log_equation_equals_zero_l3182_318218


namespace equilateral_triangle_division_l3182_318202

/-- Given an equilateral triangle with sides divided into three equal parts and an inner equilateral
    triangle formed by connecting corresponding division points, if the inscribed circle in the inner
    triangle has radius 6 cm, then the side length of the inner triangle is 12√3 cm and the side
    length of the outer triangle is 36 cm. -/
theorem equilateral_triangle_division (r : ℝ) (inner_side outer_side : ℝ) :
  r = 6 →
  inner_side = 12 * Real.sqrt 3 →
  outer_side = 36 :=
by sorry

end equilateral_triangle_division_l3182_318202


namespace coffee_per_donut_l3182_318252

/-- Proves that the number of ounces of coffee needed per donut is 2, given the specified conditions. -/
theorem coffee_per_donut (ounces_per_pot : ℕ) (cost_per_pot : ℕ) (dozen_donuts : ℕ) (total_coffee_cost : ℕ) :
  ounces_per_pot = 12 →
  cost_per_pot = 3 →
  dozen_donuts = 3 →
  total_coffee_cost = 18 →
  (total_coffee_cost / cost_per_pot * ounces_per_pot) / (dozen_donuts * 12) = 2 :=
by sorry

end coffee_per_donut_l3182_318252


namespace dessert_shop_theorem_l3182_318219

/-- Represents the dessert shop problem -/
structure DessertShop where
  x : ℕ  -- portions of dessert A
  y : ℕ  -- portions of dessert B
  a : ℕ  -- profit per portion of dessert A in yuan

/-- Conditions of the dessert shop problem -/
def DessertShopConditions (shop : DessertShop) : Prop :=
  shop.a > 0 ∧
  30 * shop.x + 10 * shop.y = 2000 ∧
  15 * shop.x + 20 * shop.y ≤ 3100

/-- Theorem stating the main results of the dessert shop problem -/
theorem dessert_shop_theorem (shop : DessertShop) 
  (h : DessertShopConditions shop) : 
  (shop.y = 200 - 3 * shop.x) ∧ 
  (shop.a = 3 → 3 * shop.x + 2 * shop.y ≥ 220 → 15 * shop.x + 20 * shop.y ≥ 1300) ∧
  (3 * shop.x + 2 * shop.y = 450 → shop.a = 8) := by
  sorry

end dessert_shop_theorem_l3182_318219


namespace smallest_positive_integer_congruence_l3182_318245

theorem smallest_positive_integer_congruence :
  ∃ (n : ℕ), n > 0 ∧ (77 * n) % 385 = 308 % 385 ∧
  ∀ (m : ℕ), m > 0 → (77 * m) % 385 = 308 % 385 → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_positive_integer_congruence_l3182_318245


namespace min_sum_on_unit_circle_l3182_318261

theorem min_sum_on_unit_circle (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = 1) :
  ∃ (m : ℝ), m = Real.sqrt 2 ∧ ∀ (a b : ℝ), 0 < a → 0 < b → a^2 + b^2 = 1 → m ≤ a + b :=
sorry

end min_sum_on_unit_circle_l3182_318261


namespace power_function_through_point_l3182_318201

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 4 = 2 → f 16 = 4 := by
  sorry

end power_function_through_point_l3182_318201


namespace unique_number_divisible_by_24_with_cube_root_between_7_9_and_8_l3182_318237

theorem unique_number_divisible_by_24_with_cube_root_between_7_9_and_8 :
  ∃! (n : ℕ), n > 0 ∧ 24 ∣ n ∧ (7.9 : ℝ) < n^(1/3) ∧ n^(1/3) < 8 :=
by sorry

end unique_number_divisible_by_24_with_cube_root_between_7_9_and_8_l3182_318237


namespace expected_score_is_seven_sixths_l3182_318214

/-- Represents the score obtained from a single die roll -/
inductive Score
| one
| two
| three

/-- The probability of getting each score -/
def prob (s : Score) : ℚ :=
  match s with
  | Score.one => 1/2
  | Score.two => 1/3
  | Score.three => 1/6

/-- The point value associated with each score -/
def value (s : Score) : ℕ :=
  match s with
  | Score.one => 1
  | Score.two => 2
  | Score.three => 3

/-- The expected score for a single roll of the die -/
def expected_score : ℚ :=
  (prob Score.one * value Score.one) +
  (prob Score.two * value Score.two) +
  (prob Score.three * value Score.three)

theorem expected_score_is_seven_sixths :
  expected_score = 7/6 := by
  sorry

end expected_score_is_seven_sixths_l3182_318214


namespace product_sum_theorem_l3182_318287

theorem product_sum_theorem (p q r s t : ℤ) :
  (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -48 →
  p + q + r + s + t = 22 := by
  sorry

end product_sum_theorem_l3182_318287


namespace gmat_scores_l3182_318240

theorem gmat_scores (u v w : ℝ) 
  (h_order : u > v ∧ v > w)
  (h_avg : u - w = (u + v + w) / 3)
  (h_diff : u - v = 2 * (v - w)) :
  v / u = 4 / 7 := by
sorry

end gmat_scores_l3182_318240


namespace fraction_equality_l3182_318221

theorem fraction_equality (a b : ℝ) (h : a + b ≠ 0) : (-a - b) / (a + b) = -1 := by
  sorry

end fraction_equality_l3182_318221


namespace min_value_theorem_l3182_318246

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1 = 2*m + n) :
  (1/m + 2/n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 1 = 2*m₀ + n₀ ∧ 1/m₀ + 2/n₀ = 8 := by
  sorry

end min_value_theorem_l3182_318246


namespace set_of_a_values_l3182_318217

theorem set_of_a_values (a : ℝ) : 
  (2 ∉ {x : ℝ | x - a < 0}) ↔ (a ∈ {a : ℝ | a ≤ 2}) := by
  sorry

end set_of_a_values_l3182_318217


namespace sculpture_surface_area_l3182_318271

/-- Represents a cube sculpture with three layers -/
structure CubeSculpture where
  top_layer : Nat
  middle_layer : Nat
  bottom_layer : Nat
  cube_edge_length : Real

/-- Calculates the exposed surface area of a cube sculpture -/
def exposed_surface_area (sculpture : CubeSculpture) : Real :=
  let top_area := sculpture.top_layer * (5 * sculpture.cube_edge_length ^ 2)
  let middle_area := 4 * sculpture.middle_layer * sculpture.cube_edge_length ^ 2
  let bottom_area := sculpture.bottom_layer * sculpture.cube_edge_length ^ 2
  top_area + middle_area + bottom_area

/-- The main theorem stating that the exposed surface area of the specific sculpture is 35 square meters -/
theorem sculpture_surface_area :
  let sculpture : CubeSculpture := {
    top_layer := 1,
    middle_layer := 6,
    bottom_layer := 12,
    cube_edge_length := 1
  }
  exposed_surface_area sculpture = 35 := by
  sorry

end sculpture_surface_area_l3182_318271


namespace workers_days_per_week_l3182_318209

/-- The number of toys produced per week -/
def weekly_production : ℕ := 5500

/-- The number of toys produced per day -/
def daily_production : ℕ := 1100

/-- The number of days worked per week -/
def days_worked : ℕ := weekly_production / daily_production

theorem workers_days_per_week :
  days_worked = 5 :=
sorry

end workers_days_per_week_l3182_318209


namespace sum_of_squares_l3182_318231

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 30) : x^2 + y^2 = 840 := by
  sorry

end sum_of_squares_l3182_318231


namespace factorization_theorem_l3182_318272

theorem factorization_theorem (a : ℝ) : (a^2 + 4)^2 - 16*a^2 = (a + 2)^2 * (a - 2)^2 := by
  sorry

end factorization_theorem_l3182_318272


namespace chase_cardinals_l3182_318275

/-- The number of birds Gabrielle saw -/
def gabrielle_birds : ℕ := 5 + 4 + 3

/-- The number of robins and blue jays Chase saw -/
def chase_known_birds : ℕ := 2 + 3

/-- The ratio of birds Gabrielle saw compared to Chase -/
def ratio : ℚ := 1.2

theorem chase_cardinals :
  ∃ (chase_total : ℕ) (chase_cardinals : ℕ),
    chase_total = (gabrielle_birds : ℚ) / ratio ∧
    chase_total = chase_known_birds + chase_cardinals ∧
    chase_cardinals = 5 := by
  sorry

end chase_cardinals_l3182_318275


namespace z_squared_minus_one_equals_two_plus_four_i_l3182_318263

def z : ℂ := 2 + Complex.I

theorem z_squared_minus_one_equals_two_plus_four_i :
  z^2 - 1 = 2 + 4*Complex.I := by
  sorry

end z_squared_minus_one_equals_two_plus_four_i_l3182_318263


namespace smallest_number_formed_by_2_and_4_l3182_318255

def is_formed_by_2_and_4 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2)

theorem smallest_number_formed_by_2_and_4 :
  ∀ n : ℕ, is_formed_by_2_and_4 n → n ≥ 24 :=
by
  sorry

end smallest_number_formed_by_2_and_4_l3182_318255


namespace script_writing_problem_l3182_318262

/-- Represents the number of lines for each character in the script -/
structure ScriptLines where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the script writing problem -/
def script_conditions (s : ScriptLines) : Prop :=
  s.first = s.second + 8 ∧
  s.third = 2 ∧
  s.second = 3 * s.third + 6 ∧
  s.first = 20

/-- The theorem stating the solution to the script writing problem -/
theorem script_writing_problem (s : ScriptLines) 
  (h : script_conditions s) : ∃ m : ℕ, s.second = m * s.third + 6 ∧ m = 3 := by
  sorry

end script_writing_problem_l3182_318262


namespace mrs_hilt_friday_miles_l3182_318249

/-- Mrs. Hilt's running schedule for a week -/
structure RunningSchedule where
  monday : ℕ
  wednesday : ℕ
  friday : ℕ
  total : ℕ

/-- Theorem: Given Mrs. Hilt's running schedule, prove she ran 7 miles on Friday -/
theorem mrs_hilt_friday_miles (schedule : RunningSchedule) 
  (h1 : schedule.monday = 3)
  (h2 : schedule.wednesday = 2)
  (h3 : schedule.total = 12)
  (h4 : schedule.total = schedule.monday + schedule.wednesday + schedule.friday) :
  schedule.friday = 7 := by
  sorry

end mrs_hilt_friday_miles_l3182_318249


namespace oliver_baseball_cards_l3182_318267

theorem oliver_baseball_cards (cards_per_page new_cards old_cards : ℕ) 
  (h1 : cards_per_page = 3)
  (h2 : new_cards = 2)
  (h3 : old_cards = 10) :
  (new_cards + old_cards) / cards_per_page = 4 := by
  sorry

end oliver_baseball_cards_l3182_318267


namespace prism_18_edges_has_8_faces_l3182_318273

/-- A prism is a polyhedron with two congruent parallel faces (bases) and lateral faces that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given the number of edges. -/
def num_faces (p : Prism) : ℕ :=
  2 + (p.edges / 3)  -- 2 bases + lateral faces

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_has_8_faces (p : Prism) (h : p.edges = 18) : num_faces p = 8 := by
  sorry


end prism_18_edges_has_8_faces_l3182_318273


namespace jenny_pokemon_cards_l3182_318229

theorem jenny_pokemon_cards (J : ℕ) : 
  J + (J + 2) + 3 * (J + 2) = 38 → J = 6 := by
  sorry

end jenny_pokemon_cards_l3182_318229


namespace combined_tax_rate_l3182_318298

/-- Calculate the combined tax rate for three individuals given their tax rates and income ratios -/
theorem combined_tax_rate
  (mork_rate : ℝ)
  (mindy_rate : ℝ)
  (orson_rate : ℝ)
  (mindy_income_ratio : ℝ)
  (orson_income_ratio : ℝ)
  (h1 : mork_rate = 0.45)
  (h2 : mindy_rate = 0.15)
  (h3 : orson_rate = 0.25)
  (h4 : mindy_income_ratio = 4)
  (h5 : orson_income_ratio = 2) :
  let total_tax := mork_rate + mindy_rate * mindy_income_ratio + orson_rate * orson_income_ratio
  let total_income := 1 + mindy_income_ratio + orson_income_ratio
  (total_tax / total_income) * 100 = 22.14 := by
  sorry

end combined_tax_rate_l3182_318298


namespace x_squared_mod_20_l3182_318264

theorem x_squared_mod_20 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 20]) (h2 : 2 * x ≡ 8 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] := by
  sorry

end x_squared_mod_20_l3182_318264


namespace triangle_area_l3182_318248

theorem triangle_area (base height : ℝ) (h1 : base = 25) (h2 : height = 60) :
  (base * height) / 2 = 750 := by
  sorry

end triangle_area_l3182_318248


namespace two_digit_number_puzzle_l3182_318279

theorem two_digit_number_puzzle : ∃ (n : ℕ) (x y : ℕ),
  0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
  n = 10 * x + y ∧
  x^2 + y^2 = 10 * x + y + 11 ∧
  2 * x * y = 10 * x + y - 5 :=
by sorry

end two_digit_number_puzzle_l3182_318279


namespace partition_scores_with_equal_average_l3182_318228

theorem partition_scores_with_equal_average 
  (N : ℕ) 
  (scores : List ℤ) 
  (h_length : scores.length = 3 * N)
  (h_range : ∀ s ∈ scores, 60 ≤ s ∧ s ≤ 100)
  (h_freq : ∀ s ∈ scores, (scores.filter (· = s)).length ≥ 2)
  (h_avg : scores.sum / (3 * N) = 824 / 10) :
  ∃ (class1 class2 class3 : List ℤ),
    class1.length = N ∧ 
    class2.length = N ∧ 
    class3.length = N ∧
    scores = class1 ++ class2 ++ class3 ∧
    class1.sum / N = 824 / 10 ∧
    class2.sum / N = 824 / 10 ∧
    class3.sum / N = 824 / 10 :=
by sorry

end partition_scores_with_equal_average_l3182_318228


namespace earth_pile_fraction_l3182_318203

theorem earth_pile_fraction (P : ℚ) (P_pos : P > 0) : 
  P * (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) * (1 - 1/6) * (1 - 1/7) = P * (1/7) := by
  sorry

end earth_pile_fraction_l3182_318203


namespace correct_quotient_l3182_318257

theorem correct_quotient (D : ℕ) : 
  D % 21 = 0 →  -- The remainder is 0 when divided by 21
  D / 12 = 35 →  -- Dividing by 12 yields a quotient of 35
  D / 21 = 20  -- The correct quotient when dividing by 21 is 20
:= by sorry

end correct_quotient_l3182_318257


namespace boat_speed_in_still_water_l3182_318285

/-- Given a boat that travels 13 km/hr along a stream and 5 km/hr against the same stream,
    its speed in still water is 9 km/hr. -/
theorem boat_speed_in_still_water
  (speed_along_stream : ℝ)
  (speed_against_stream : ℝ)
  (h_along : speed_along_stream = 13)
  (h_against : speed_against_stream = 5) :
  (speed_along_stream + speed_against_stream) / 2 = 9 :=
by sorry

end boat_speed_in_still_water_l3182_318285


namespace boat_speed_in_still_water_l3182_318243

/-- Given a boat that travels 13 km along a stream and 5 km against the same stream
    in one hour each, its speed in still water is 9 km/hr. -/
theorem boat_speed_in_still_water
  (along_stream : ℝ) (against_stream : ℝ)
  (h_along : along_stream = 13)
  (h_against : against_stream = 5)
  (h_time : along_stream = (boat_speed + stream_speed) * 1 ∧
            against_stream = (boat_speed - stream_speed) * 1)
  : boat_speed = 9 :=
by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l3182_318243


namespace solve_equation_l3182_318288

-- Define the function F
def F (a b c d : ℕ) : ℕ := a^b + c * d

-- State the theorem
theorem solve_equation : ∃ x : ℕ, F 2 x 4 11 = 300 ∧ x = 8 := by
  sorry

end solve_equation_l3182_318288


namespace even_increasing_function_inequality_l3182_318270

/-- An even function that is monotonically increasing on the non-negative reals -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)

theorem even_increasing_function_inequality 
  (f : ℝ → ℝ) 
  (h_even_increasing : EvenIncreasingFunction f) 
  (h_f_1 : f 1 = 0) :
  {x : ℝ | f (x - 2) ≥ 0} = {x : ℝ | x ≥ 3 ∨ x ≤ 1} := by
  sorry

end even_increasing_function_inequality_l3182_318270


namespace derivative_f_at_zero_l3182_318296

-- Define the function f
def f (x : ℝ) : ℝ := (2*x + 1)^3

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 6 := by sorry

end derivative_f_at_zero_l3182_318296


namespace sqrt_sum_difference_l3182_318277

theorem sqrt_sum_difference : Real.sqrt 50 + Real.sqrt 32 - Real.sqrt 2 = 8 * Real.sqrt 2 := by
  sorry

end sqrt_sum_difference_l3182_318277


namespace square_sum_equals_sixteen_l3182_318284

theorem square_sum_equals_sixteen (x : ℝ) : 
  (x - 1)^2 + 2*(x - 1)*(5 - x) + (5 - x)^2 = 16 := by
  sorry

end square_sum_equals_sixteen_l3182_318284


namespace sector_central_angle_l3182_318210

theorem sector_central_angle (l : ℝ) (S : ℝ) (h1 : l = 6) (h2 : S = 18) : ∃ (r : ℝ) (α : ℝ), 
  S = (1/2) * l * r ∧ l = r * α ∧ α = 1 :=
sorry

end sector_central_angle_l3182_318210


namespace star_three_neg_two_thirds_l3182_318242

-- Define the ☆ operation
def star (a b : ℚ) : ℚ := a^2 + a*b - 5

-- Theorem statement
theorem star_three_neg_two_thirds : star 3 (-2/3) = 2 := by
  sorry

end star_three_neg_two_thirds_l3182_318242


namespace youngest_child_age_l3182_318292

/-- Given 5 children born at intervals of 2 years each, 
    if the sum of their ages is 55 years, 
    then the age of the youngest child is 7 years. -/
theorem youngest_child_age 
  (n : ℕ) 
  (h1 : n = 5) 
  (interval : ℕ) 
  (h2 : interval = 2) 
  (total_age : ℕ) 
  (h3 : total_age = 55) 
  (youngest_age : ℕ) 
  (h4 : youngest_age * n + (n * (n - 1) / 2) * interval = total_age) : 
  youngest_age = 7 := by
sorry

end youngest_child_age_l3182_318292


namespace joan_seashells_l3182_318259

theorem joan_seashells (jessica_shells : ℕ) (total_shells : ℕ) (h1 : jessica_shells = 8) (h2 : total_shells = 14) :
  total_shells - jessica_shells = 6 :=
sorry

end joan_seashells_l3182_318259


namespace average_monthly_sales_l3182_318247

def monthly_sales : List ℝ := [150, 120, 80, 100, 90, 130]

theorem average_monthly_sales :
  (List.sum monthly_sales) / (List.length monthly_sales) = 111.67 := by
  sorry

end average_monthly_sales_l3182_318247


namespace leg_equals_sum_of_radii_l3182_318251

/-- An isosceles right triangle with its inscribed and circumscribed circles -/
structure IsoscelesRightTriangle where
  /-- The length of each leg of the triangle -/
  a : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The leg length is positive -/
  a_pos : 0 < a
  /-- The inscribed circle radius is half the leg length -/
  r_def : r = a / 2
  /-- The circumscribed circle radius is (a√2)/2 -/
  R_def : R = (a * Real.sqrt 2) / 2

/-- 
The length of the legs of an isosceles right triangle is equal to 
the sum of the radii of its inscribed and circumscribed circles 
-/
theorem leg_equals_sum_of_radii (t : IsoscelesRightTriangle) : t.a = t.r + t.R := by
  sorry

end leg_equals_sum_of_radii_l3182_318251


namespace max_value_of_x_l3182_318204

/-- Given a > 0 and b > 0, x is defined as the minimum of {1, a, b / (a² + b²)}.
    This theorem states that the maximum value of x is √2 / 2. -/
theorem max_value_of_x (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := min 1 (min a (b / (a^2 + b^2)))
  ∃ (max_x : ℝ), max_x = Real.sqrt 2 / 2 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 →
    min 1 (min a' (b' / (a'^2 + b'^2))) ≤ max_x :=
by sorry

end max_value_of_x_l3182_318204


namespace line_passes_through_fixed_point_projection_trajectory_l3182_318225

/-- The line equation as a function of x, y, and m -/
def line_equation (x y m : ℝ) : Prop := 2 * x + (1 + m) * y + 2 * m = 0

/-- The fixed point through which all lines pass -/
def fixed_point : ℝ × ℝ := (1, -2)

/-- Point P coordinates -/
def point_p : ℝ × ℝ := (-1, 0)

/-- Trajectory equation of point M -/
def trajectory_equation (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 2

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation (fixed_point.1) (fixed_point.2) m := by sorry

theorem projection_trajectory :
  ∀ x y : ℝ, (∃ m : ℝ, line_equation x y m ∧ 
    (x - point_p.1)^2 + (y - point_p.2)^2 = 
    ((x - point_p.1) * (fixed_point.1 - point_p.1) + (y - point_p.2) * (fixed_point.2 - point_p.2))^2 / 
    ((fixed_point.1 - point_p.1)^2 + (fixed_point.2 - point_p.2)^2)) →
  trajectory_equation x y := by sorry

end line_passes_through_fixed_point_projection_trajectory_l3182_318225


namespace min_perimeter_triangle_l3182_318211

theorem min_perimeter_triangle (a b c : ℕ) : 
  a = 52 → b = 76 → c > 0 → 
  (a + b > c) → (a + c > b) → (b + c > a) →
  (∀ x : ℕ, x > 0 → (a + b > x) → (a + x > b) → (b + x > a) → c ≤ x) →
  a + b + c = 153 :=
sorry

end min_perimeter_triangle_l3182_318211


namespace mean_median_difference_l3182_318268

/-- Represents the frequency distribution of days missed --/
structure FrequencyDistribution :=
  (zero_days : Nat)
  (one_day : Nat)
  (two_days : Nat)
  (three_days : Nat)
  (four_days : Nat)
  (five_days : Nat)

/-- Calculates the median of the dataset --/
def median (fd : FrequencyDistribution) : Rat :=
  2

/-- Calculates the mean of the dataset --/
def mean (fd : FrequencyDistribution) : Rat :=
  (0 * fd.zero_days + 1 * fd.one_day + 2 * fd.two_days + 
   3 * fd.three_days + 4 * fd.four_days + 5 * fd.five_days) / 20

/-- The main theorem to prove --/
theorem mean_median_difference 
  (fd : FrequencyDistribution) 
  (h1 : fd.zero_days = 4)
  (h2 : fd.one_day = 2)
  (h3 : fd.two_days = 5)
  (h4 : fd.three_days = 3)
  (h5 : fd.four_days = 2)
  (h6 : fd.five_days = 4)
  (h7 : fd.zero_days + fd.one_day + fd.two_days + fd.three_days + fd.four_days + fd.five_days = 20) :
  mean fd - median fd = 9 / 20 := by
  sorry

end mean_median_difference_l3182_318268


namespace common_value_theorem_l3182_318260

theorem common_value_theorem (a b : ℝ) 
  (h1 : a * (a - 4) = b * (b - 4))
  (h2 : a ≠ b)
  (h3 : a + b = 4) :
  a * (a - 4) = -3 := by
sorry

end common_value_theorem_l3182_318260


namespace symmetric_points_sum_power_l3182_318295

theorem symmetric_points_sum_power (m n : ℤ) : 
  (m = -6 ∧ n = 5) → (m + n)^2012 = 1 := by sorry

end symmetric_points_sum_power_l3182_318295


namespace sixty_eighth_digit_of_largest_n_l3182_318208

def largest_n : ℕ := (10^100 - 1) / 14

def digit_at_position (n : ℕ) (pos : ℕ) : ℕ :=
  (n / 10^(pos - 1)) % 10

theorem sixty_eighth_digit_of_largest_n :
  digit_at_position largest_n 68 = 1 := by
  sorry

end sixty_eighth_digit_of_largest_n_l3182_318208


namespace inverse_variation_problem_l3182_318206

theorem inverse_variation_problem (z w : ℝ) (k : ℝ) (h1 : z * Real.sqrt w = k) 
  (h2 : 6 * Real.sqrt 3 = k) (h3 : z = 3/2) : w = 48 := by
  sorry

end inverse_variation_problem_l3182_318206


namespace not_all_tv_owners_have_gellert_pass_l3182_318254

-- Define the universe of discourse
variable (Person : Type)

-- Define predicates
variable (isTelevisionOwner : Person → Prop)
variable (isPainter : Person → Prop)
variable (hasGellertPass : Person → Prop)

-- State the theorem
theorem not_all_tv_owners_have_gellert_pass
  (h1 : ∃ x, isTelevisionOwner x ∧ ¬isPainter x)
  (h2 : ∀ x, hasGellertPass x ∧ ¬isPainter x → ¬isTelevisionOwner x) :
  ∃ x, isTelevisionOwner x ∧ ¬hasGellertPass x :=
by sorry

end not_all_tv_owners_have_gellert_pass_l3182_318254


namespace rectangle_division_distinctness_l3182_318283

theorem rectangle_division_distinctness (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  a * c = b * d →
  a + c = b + d →
  ¬(a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
by sorry

end rectangle_division_distinctness_l3182_318283


namespace dog_year_conversion_l3182_318290

/-- Represents the conversion of dog years to human years -/
structure DogYearConversion where
  first_year : ℕ
  second_year : ℕ
  later_years : ℕ

/-- Calculates the total human years for a given dog age -/
def human_years (c : DogYearConversion) (dog_age : ℕ) : ℕ :=
  if dog_age = 0 then 0
  else if dog_age = 1 then c.first_year
  else if dog_age = 2 then c.first_year + c.second_year
  else c.first_year + c.second_year + (dog_age - 2) * c.later_years

/-- The main theorem to prove -/
theorem dog_year_conversion (c : DogYearConversion) :
  c.first_year = 15 → c.second_year = 9 → human_years c 10 = 64 → c.later_years = 5 := by
  sorry

end dog_year_conversion_l3182_318290


namespace part_one_part_two_l3182_318256

-- Part 1
theorem part_one (x : ℝ) (h1 : x^2 - 4*x + 3 < 0) (h2 : |x - 3| < 1) :
  2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h_pos : a > 0) 
  (h_subset : {x : ℝ | x^2 - 4*a*x + 3*a^2 ≥ 0} ⊂ {x : ℝ | |x - 3| ≥ 1}) :
  4/3 ≤ a ∧ a ≤ 2 := by sorry

end part_one_part_two_l3182_318256


namespace project_work_difference_l3182_318265

/-- Represents the work times of four people on a project -/
structure ProjectWork where
  time1 : ℕ
  time2 : ℕ
  time3 : ℕ
  time4 : ℕ

/-- The total work time of the project -/
def totalTime (pw : ProjectWork) : ℕ :=
  pw.time1 + pw.time2 + pw.time3 + pw.time4

/-- The work times are in the ratio 1:2:3:4 -/
def validRatio (pw : ProjectWork) : Prop :=
  2 * pw.time1 = pw.time2 ∧
  3 * pw.time1 = pw.time3 ∧
  4 * pw.time1 = pw.time4

theorem project_work_difference (pw : ProjectWork) 
  (h1 : totalTime pw = 240)
  (h2 : validRatio pw) :
  pw.time4 - pw.time1 = 72 := by
  sorry

end project_work_difference_l3182_318265


namespace cos_pi_third_minus_alpha_l3182_318250

theorem cos_pi_third_minus_alpha (α : Real) 
  (h : Real.sin (π / 6 + α) = 2 / 3) : 
  Real.cos (π / 3 - α) = 2 / 3 := by
  sorry

end cos_pi_third_minus_alpha_l3182_318250


namespace distance_between_hyperbola_and_ellipse_l3182_318293

theorem distance_between_hyperbola_and_ellipse 
  (x y z w : ℝ) 
  (h1 : x * y = 4) 
  (h2 : z^2 + 4 * w^2 = 4) : 
  (x - z)^2 + (y - w)^2 ≥ 1.6 := by
sorry

end distance_between_hyperbola_and_ellipse_l3182_318293


namespace correlated_relationships_l3182_318220

-- Define the set of all relationships
inductive Relationship
| A  -- A person's height and weight
| B  -- The distance traveled by a vehicle moving at a constant speed and the time of travel
| C  -- A person's height and eyesight
| D  -- The volume of a cube and its edge length

-- Define a function to check if a relationship is correlated
def is_correlated (r : Relationship) : Prop :=
  match r with
  | Relationship.A => true  -- Height and weight are correlated
  | Relationship.B => true  -- Distance and time at constant speed are correlated (functional)
  | Relationship.C => false -- Height and eyesight are not correlated
  | Relationship.D => true  -- Volume and edge length of a cube are correlated (functional)

-- Theorem stating that the set of correlated relationships is {A, B, D}
theorem correlated_relationships :
  {r : Relationship | is_correlated r} = {Relationship.A, Relationship.B, Relationship.D} :=
by sorry

end correlated_relationships_l3182_318220


namespace gcd_of_sequence_is_three_l3182_318234

def a (n : ℕ) : ℕ := (2*n - 1) * (2*n + 1) * (2*n + 3)

theorem gcd_of_sequence_is_three :
  ∃ d : ℕ, d > 0 ∧ 
  (∀ k : ℕ, k ≥ 1 → k ≤ 2008 → d ∣ a k) ∧
  (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ≥ 1 → k ≤ 2008 → m ∣ a k) → m ≤ d) ∧
  d = 3 := by
sorry

end gcd_of_sequence_is_three_l3182_318234


namespace savings_increase_percentage_l3182_318200

theorem savings_increase_percentage (I : ℝ) (I_pos : I > 0) : 
  let regular_expense_ratio : ℝ := 0.75
  let additional_expense_ratio : ℝ := 0.10
  let income_increase_ratio : ℝ := 0.20
  let regular_expense_increase_ratio : ℝ := 0.10
  let additional_expense_increase_ratio : ℝ := 0.25
  
  let initial_savings := I * (1 - regular_expense_ratio - additional_expense_ratio)
  let new_income := I * (1 + income_increase_ratio)
  let new_regular_expense := I * regular_expense_ratio * (1 + regular_expense_increase_ratio)
  let new_additional_expense := I * additional_expense_ratio * (1 + additional_expense_increase_ratio)
  let new_savings := new_income - new_regular_expense - new_additional_expense
  
  (new_savings - initial_savings) / initial_savings = 2/3 :=
by
  sorry

end savings_increase_percentage_l3182_318200


namespace race_total_time_l3182_318230

theorem race_total_time (total_runners : Nat) (fast_runners : Nat) (fast_time : Nat) (extra_time : Nat) :
  total_runners = 8 →
  fast_runners = 5 →
  fast_time = 8 →
  extra_time = 2 →
  (fast_runners * fast_time) + ((total_runners - fast_runners) * (fast_time + extra_time)) = 70 := by
  sorry

end race_total_time_l3182_318230


namespace volunteer_schedule_l3182_318233

theorem volunteer_schedule (sasha leo uma kim : ℕ) 
  (h1 : sasha = 5) 
  (h2 : leo = 8) 
  (h3 : uma = 9) 
  (h4 : kim = 10) : 
  Nat.lcm (Nat.lcm (Nat.lcm sasha leo) uma) kim = 360 := by
  sorry

end volunteer_schedule_l3182_318233


namespace survey_B_most_suitable_for_census_l3182_318215

-- Define the characteristics of a survey
structure Survey where
  population : Set String
  method : String
  is_destructive : Bool
  is_manageable : Bool

-- Define the conditions for a census
def is_census_suitable (s : Survey) : Prop :=
  s.is_manageable ∧ ¬s.is_destructive ∧ s.method = "complete enumeration"

-- Define the surveys
def survey_A : Survey := {
  population := {"televisions"},
  method := "sampling",
  is_destructive := true,
  is_manageable := false
}

def survey_B : Survey := {
  population := {"ninth grade students in a certain middle school class"},
  method := "complete enumeration",
  is_destructive := false,
  is_manageable := true
}

def survey_C : Survey := {
  population := {"middle school students in Chongqing"},
  method := "sampling",
  is_destructive := false,
  is_manageable := false
}

def survey_D : Survey := {
  population := {"middle school students in Chongqing"},
  method := "sampling",
  is_destructive := false,
  is_manageable := false
}

-- Theorem stating that survey B is the most suitable for a census
theorem survey_B_most_suitable_for_census :
  is_census_suitable survey_B ∧
  ¬is_census_suitable survey_A ∧
  ¬is_census_suitable survey_C ∧
  ¬is_census_suitable survey_D :=
sorry

end survey_B_most_suitable_for_census_l3182_318215


namespace largest_valid_coloring_l3182_318239

/-- A coloring of an n × n grid with two colors. -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a rectangle in the grid has all four corners the same color. -/
def hasMonochromaticRectangle (c : Coloring n) : Prop :=
  ∃ (i j k l : Fin n), i < k ∧ j < l ∧
    c i j = c i l ∧ c i l = c k j ∧ c k j = c k l

/-- The largest n for which a valid coloring exists. -/
def largestValidN : ℕ := 4

theorem largest_valid_coloring :
  (∃ (c : Coloring largestValidN), ¬hasMonochromaticRectangle c) ∧
  (∀ (m : ℕ), m > largestValidN →
    ∀ (c : Coloring m), hasMonochromaticRectangle c) :=
sorry

end largest_valid_coloring_l3182_318239


namespace profit_percentage_previous_year_l3182_318282

/-- Given the conditions of a company's financial performance over two years,
    prove that the profit percentage in the previous year was 10%. -/
theorem profit_percentage_previous_year
  (R : ℝ) -- Revenues in the previous year
  (P : ℝ) -- Profits in the previous year
  (h1 : R > 0) -- Assume positive revenue
  (h2 : P > 0) -- Assume positive profit
  (h3 : 0.8 * R * 0.12 = 0.96 * P) -- Condition from the problem
  : P / R = 0.1 := by
  sorry

end profit_percentage_previous_year_l3182_318282


namespace calculation_proofs_l3182_318278

theorem calculation_proofs :
  (1) -2^2 * (1/4) + 4 / (4/9) + (-1)^2023 = 7 ∧
  (2) -1^4 + |2 - (-3)^2| + (1/2) / (-3/2) = 17/3 := by
  sorry

end calculation_proofs_l3182_318278


namespace tetrahedron_volume_l3182_318224

/-- The volume of a regular tetrahedron with given base side length and lateral face angle -/
theorem tetrahedron_volume 
  (base_side : ℝ) 
  (lateral_angle : ℝ) 
  (h_base : base_side = Real.sqrt 3) 
  (h_angle : lateral_angle = π / 3) : 
  (1 / 3 : ℝ) * base_side ^ 2 * (base_side / 2) / Real.tan lateral_angle = 1 / 2 := by
  sorry

end tetrahedron_volume_l3182_318224


namespace complex_equation_modulus_l3182_318241

theorem complex_equation_modulus : ∀ (x y : ℝ),
  (Complex.I * (x + 2 * Complex.I) = y - Complex.I) →
  Complex.abs (x - y * Complex.I) = Real.sqrt 5 := by sorry

end complex_equation_modulus_l3182_318241


namespace marble_calculation_l3182_318216

/-- Calculate the final number of marbles and prove its square root is 7 --/
theorem marble_calculation (initial : ℕ) (triple : ℕ → ℕ) (add : ℕ → ℕ → ℕ) 
  (lose_percent : ℕ → ℚ → ℕ) (find : ℕ → ℕ → ℕ) : 
  initial = 16 →
  (∀ x, triple x = 3 * x) →
  (∀ x y, add x y = x + y) →
  (∀ x p, lose_percent x p = x - ⌊(p * x : ℚ)⌋) →
  (∀ x y, find x y = x + y) →
  ∃ (final : ℕ), final = find (lose_percent (add (triple initial) 10) (1/4)) 5 ∧ 
  (final : ℝ).sqrt = 7 := by
sorry

end marble_calculation_l3182_318216


namespace replaced_student_weight_l3182_318266

theorem replaced_student_weight
  (n : ℕ)
  (new_weight : ℝ)
  (avg_decrease : ℝ)
  (h1 : n = 6)
  (h2 : new_weight = 62)
  (h3 : avg_decrease = 3)
  : ∃ (old_weight : ℝ), old_weight = 80 :=
by
  sorry

end replaced_student_weight_l3182_318266


namespace quadratic_equation_roots_l3182_318274

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - k * x + 6 = 0 ∧ x = 2) → 
  (k = 11 ∧ ∃ y : ℝ, 4 * y^2 - k * y + 6 = 0 ∧ y = 3/4) :=
by sorry

end quadratic_equation_roots_l3182_318274


namespace intersection_A_B_l3182_318244

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end intersection_A_B_l3182_318244


namespace cone_height_l3182_318232

/-- For a cone with slant height 2 and lateral area 4 times the area of its base, 
    the height of the cone is π/2. -/
theorem cone_height (r : ℝ) (h : ℝ) : 
  r > 0 → h > 0 → 
  r^2 + h^2 = 4 → -- slant height is 2
  2 * π * r = 4 * π * r^2 → -- lateral area is 4 times base area
  h = π / 2 := by
sorry

end cone_height_l3182_318232


namespace marble_capacity_l3182_318212

/-- 
Given:
- A small bottle with volume 20 ml can hold 40 marbles
- A larger bottle has volume 60 ml
Prove that the larger bottle can hold 120 marbles
-/
theorem marble_capacity (small_volume small_capacity large_volume : ℕ) 
  (h1 : small_volume = 20)
  (h2 : small_capacity = 40)
  (h3 : large_volume = 60) :
  (large_volume * small_capacity) / small_volume = 120 := by
  sorry

end marble_capacity_l3182_318212


namespace quadratic_expression_minimum_l3182_318235

theorem quadratic_expression_minimum :
  ∀ x y : ℝ, x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ -22 ∧
  ∃ x y : ℝ, x^2 + 4*x*y + 5*y^2 - 8*x - 6*y = -22 :=
by sorry

end quadratic_expression_minimum_l3182_318235
