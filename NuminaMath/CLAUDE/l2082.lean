import Mathlib

namespace typist_margin_l2082_208208

theorem typist_margin (sheet_width sheet_length side_margin : ℝ)
  (percentage_used : ℝ) (h1 : sheet_width = 20)
  (h2 : sheet_length = 30) (h3 : side_margin = 2)
  (h4 : percentage_used = 0.64) :
  let total_area := sheet_width * sheet_length
  let typing_width := sheet_width - 2 * side_margin
  let top_bottom_margin := (total_area * percentage_used / typing_width - sheet_length) / (-2)
  top_bottom_margin = 3 := by
  sorry

end typist_margin_l2082_208208


namespace sum_zero_fraction_l2082_208239

theorem sum_zero_fraction (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (h_sum : x + y + z = 0) : 
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = -1/2 := by
  sorry

end sum_zero_fraction_l2082_208239


namespace circle_diameter_from_area_l2082_208255

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = π → A = π * r^2 → d = 2 * r → d = 2 := by
  sorry

end circle_diameter_from_area_l2082_208255


namespace pauls_reading_rate_l2082_208251

theorem pauls_reading_rate (total_books : ℕ) (total_weeks : ℕ) 
  (h1 : total_books = 20) (h2 : total_weeks = 5) : 
  total_books / total_weeks = 4 := by
sorry

end pauls_reading_rate_l2082_208251


namespace train_bridge_crossing_time_l2082_208243

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 255.03) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30.0024 :=
by sorry

end train_bridge_crossing_time_l2082_208243


namespace tan_sum_difference_pi_fourth_l2082_208223

theorem tan_sum_difference_pi_fourth (a : ℝ) : 
  Real.tan (a + π/4) - Real.tan (a - π/4) = 2 * Real.tan (2*a) := by
  sorry

end tan_sum_difference_pi_fourth_l2082_208223


namespace inequality_problem_l2082_208299

theorem inequality_problem (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : m > 0)
  (h4 : ∀ a b c, a > b → b > c → (1 / (a - b) + m / (b - c) ≥ 9 / (a - c))) :
  m ≥ 4 := by
  sorry

end inequality_problem_l2082_208299


namespace specific_theater_seats_l2082_208236

/-- A theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculate the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- The theorem stating the total number of seats in the specific theater -/
theorem specific_theater_seats :
  let t : Theater := { first_row_seats := 14, seat_increase := 3, last_row_seats := 50 }
  total_seats t = 416 := by
  sorry


end specific_theater_seats_l2082_208236


namespace sum_of_coefficients_l2082_208261

theorem sum_of_coefficients (a b c d : ℤ) :
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 11*x + 6) →
  a + b + c + d = 9 := by
sorry

end sum_of_coefficients_l2082_208261


namespace divides_or_divides_l2082_208227

theorem divides_or_divides (m n : ℤ) (h : m.lcm n + m.gcd n = m + n) :
  n ∣ m ∨ m ∣ n := by sorry

end divides_or_divides_l2082_208227


namespace winter_sales_is_seven_million_l2082_208266

/-- The number of pizzas sold in millions for each season --/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The percentage of pizzas sold in fall --/
def fall_percentage : ℝ := 0.20

/-- The given seasonal sales data --/
def given_sales : SeasonalSales where
  spring := 6
  summer := 7
  fall := fall_percentage * (6 + 7 + fall_percentage * (6 + 7 + 5 + 7) + 7)
  winter := 7

/-- Theorem stating that the winter sales is 7 million pizzas --/
theorem winter_sales_is_seven_million (s : SeasonalSales) :
  s.spring = 6 →
  s.summer = 7 →
  s.fall = fall_percentage * (s.spring + s.summer + s.fall + s.winter) →
  s.winter = 7 := by
  sorry

#eval given_sales.winter

end winter_sales_is_seven_million_l2082_208266


namespace prob_at_least_one_head_l2082_208242

/-- The probability of getting at least one head when tossing five coins,
    each with a 3/4 chance of heads, is 1023/1024. -/
theorem prob_at_least_one_head :
  let n : ℕ := 5  -- number of coins
  let p : ℚ := 3/4  -- probability of heads for each coin
  1 - (1 - p)^n = 1023/1024 :=
by sorry

end prob_at_least_one_head_l2082_208242


namespace factor_expression_l2082_208252

theorem factor_expression (y : ℝ) : 64 - 16 * y^3 = 16 * (2 - y) * (4 + 2*y + y^2) := by
  sorry

end factor_expression_l2082_208252


namespace continued_fraction_value_l2082_208281

theorem continued_fraction_value : 
  ∃ x : ℝ, x = 3 + 4 / (2 + 4 / x) ∧ x = 4 := by sorry

end continued_fraction_value_l2082_208281


namespace sine_function_parameters_l2082_208203

theorem sine_function_parameters
  (A ω m : ℝ)
  (h_A_pos : A > 0)
  (h_ω_pos : ω > 0)
  (h_max : ∀ x, A * Real.sin (ω * x + π / 6) + m ≤ 3)
  (h_min : ∀ x, A * Real.sin (ω * x + π / 6) + m ≥ -5)
  (h_max_achieved : ∃ x, A * Real.sin (ω * x + π / 6) + m = 3)
  (h_min_achieved : ∃ x, A * Real.sin (ω * x + π / 6) + m = -5)
  (h_symmetry : ω * (π / 2) = π) :
  A = 4 ∧ ω = 2 ∧ m = -1 := by
  sorry

end sine_function_parameters_l2082_208203


namespace sequence_sum_l2082_208269

theorem sequence_sum : 
  let seq1 := [2, 13, 24, 35]
  let seq2 := [8, 18, 28, 38, 48]
  let seq3 := [4, 7]
  (seq1.sum + seq2.sum + seq3.sum) = 225 := by
sorry

end sequence_sum_l2082_208269


namespace valid_seating_count_l2082_208258

-- Define the set of people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person

-- Define a seating arrangement as a function from position to person
def SeatingArrangement := Fin 5 → Person

-- Define the condition that two people cannot sit next to each other
def NotAdjacent (arr : SeatingArrangement) (p1 p2 : Person) : Prop :=
  ∀ i : Fin 4, arr i ≠ p1 ∨ arr (i + 1) ≠ p2

-- Define a valid seating arrangement
def ValidSeating (arr : SeatingArrangement) : Prop :=
  NotAdjacent arr Person.Alice Person.Bob ∧
  NotAdjacent arr Person.Alice Person.Carla ∧
  NotAdjacent arr Person.Derek Person.Eric ∧
  NotAdjacent arr Person.Derek Person.Carla

-- The theorem to prove
theorem valid_seating_count :
  ∃ (arrangements : Finset SeatingArrangement),
    (∀ arr ∈ arrangements, ValidSeating arr) ∧
    (∀ arr, ValidSeating arr → arr ∈ arrangements) ∧
    arrangements.card = 6 := by
  sorry

end valid_seating_count_l2082_208258


namespace faster_speed_calculation_l2082_208232

/-- Proves that a faster speed allowing 20 km more distance in the same time as 50 km at 10 km/hr is 14 km/hr -/
theorem faster_speed_calculation (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ) :
  actual_distance = 50 →
  actual_speed = 10 →
  additional_distance = 20 →
  ∃ (faster_speed : ℝ),
    (actual_distance / actual_speed = (actual_distance + additional_distance) / faster_speed) ∧
    faster_speed = 14 :=
by sorry

end faster_speed_calculation_l2082_208232


namespace infinite_series_sum_l2082_208265

/-- The sum of the infinite series ∑_{n=1}^∞ (3^n / (1 + 3^n + 3^{n+1} + 3^{2n+1})) is equal to 1/4 -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1)) = 1/4 := by
  sorry


end infinite_series_sum_l2082_208265


namespace cone_volume_from_sector_l2082_208275

/-- Given a cone whose lateral surface area is a sector with central angle 120° and area 3π,
    prove that the volume of the cone is (2√2π)/3 -/
theorem cone_volume_from_sector (θ : Real) (A : Real) (V : Real) : 
  θ = 2 * π / 3 →  -- 120° in radians
  A = 3 * π →
  V = (2 * Real.sqrt 2 * π) / 3 →
  ∃ (r l h : Real),
    r > 0 ∧ l > 0 ∧ h > 0 ∧
    A = (1/2) * l^2 * θ ∧  -- Area of sector
    r = l * θ / (2 * π) ∧  -- Relation between radius and arc length
    h^2 = l^2 - r^2 ∧     -- Pythagorean theorem
    V = (1/3) * π * r^2 * h  -- Volume of cone
    := by sorry

end cone_volume_from_sector_l2082_208275


namespace symmetrical_point_y_axis_l2082_208289

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

theorem symmetrical_point_y_axis :
  let A : Point := ⟨-1, 2⟩
  reflect_y_axis A = ⟨1, 2⟩ := by
  sorry

end symmetrical_point_y_axis_l2082_208289


namespace union_equals_A_l2082_208246

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equals_A (m : ℝ) : 
  (A m ∪ B m = A m) → (m = 0 ∨ m = 3) :=
by
  sorry

end union_equals_A_l2082_208246


namespace lines_not_parallel_l2082_208218

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem lines_not_parallel : 
  let m : ℝ := -1
  let l1 : Line := { a := 1, b := m, c := 6 }
  let l2 : Line := { a := m - 2, b := 3, c := 2 * m }
  ¬ parallel l1 l2 := by
  sorry

end lines_not_parallel_l2082_208218


namespace max_graduates_few_calls_l2082_208237

/-- The number of graduates -/
def total_graduates : ℕ := 100

/-- The number of universities -/
def num_universities : ℕ := 5

/-- The number of graduates each university attempts to contact -/
def contacts_per_university : ℕ := 50

/-- The total number of contact attempts made by all universities -/
def total_contacts : ℕ := num_universities * contacts_per_university

/-- The maximum number of graduates who received at most 2 calls -/
def max_graduates_with_few_calls : ℕ := 83

theorem max_graduates_few_calls :
  ∀ n : ℕ,
  n ≤ total_graduates →
  2 * n + 5 * (total_graduates - n) ≥ total_contacts →
  n ≤ max_graduates_with_few_calls :=
by sorry

end max_graduates_few_calls_l2082_208237


namespace digits_making_864n_divisible_by_4_l2082_208217

theorem digits_making_864n_divisible_by_4 : 
  ∃! (s : Finset Nat), 
    (∀ n ∈ s, n < 10) ∧ 
    (∀ n, n ∈ s ↔ (864 * 10 + n) % 4 = 0) ∧
    s.card = 5 := by
  sorry

end digits_making_864n_divisible_by_4_l2082_208217


namespace center_cell_value_l2082_208222

theorem center_cell_value (a b c d e f g h i : ℝ) : 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0) ∧ (g > 0) ∧ (h > 0) ∧ (i > 0) →
  (a * b * c = 1) ∧ (d * e * f = 1) ∧ (g * h * i = 1) ∧
  (a * d * g = 1) ∧ (b * e * h = 1) ∧ (c * f * i = 1) →
  (a * b * d * e = 2) ∧ (b * c * e * f = 2) ∧ (d * e * g * h = 2) ∧ (e * f * h * i = 2) →
  e = 1 :=
by sorry

end center_cell_value_l2082_208222


namespace rational_equal_to_reciprocal_l2082_208221

theorem rational_equal_to_reciprocal (x : ℚ) : x = 1 ∨ x = -1 ↔ x = 1 / x := by sorry

end rational_equal_to_reciprocal_l2082_208221


namespace rectangle_area_ratio_l2082_208220

theorem rectangle_area_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (2 * (a + c) = 2 * (2 * (b + c))) → (a = 2 * b) →
  ((a * c) = 2 * (b * c)) := by
sorry

end rectangle_area_ratio_l2082_208220


namespace original_group_size_l2082_208260

-- Define the work completion rate for a group
def work_rate (num_men : ℕ) (days : ℕ) : ℚ := 1 / (num_men * days)

-- Define the theorem
theorem original_group_size :
  ∃ (x : ℕ),
    -- Condition 1: Original group completes work in 20 days
    work_rate x 20 =
    -- Condition 2 & 3: Remaining group (x - 10) completes work in 40 days
    work_rate (x - 10) 40 ∧
    -- Answer: The original group size is 20
    x = 20 := by
  sorry

end original_group_size_l2082_208260


namespace eva_orange_count_l2082_208238

/-- Calculates the number of oranges Eva needs to buy given her dietary requirements --/
def calculate_oranges (total_days : ℕ) (orange_frequency : ℕ) : ℕ :=
  total_days / orange_frequency

/-- Theorem stating that Eva needs to buy 10 oranges given her dietary requirements --/
theorem eva_orange_count : calculate_oranges 30 3 = 10 := by
  sorry

end eva_orange_count_l2082_208238


namespace smallest_non_factor_product_l2082_208280

def is_factor (a b : ℕ) : Prop := b % a = 0

def are_non_consecutive (a b : ℕ) : Prop := b > a + 1

theorem smallest_non_factor_product (x y : ℕ) : 
  x ≠ y →
  x > 0 →
  y > 0 →
  is_factor x 48 →
  is_factor y 48 →
  are_non_consecutive x y →
  ¬(is_factor (x * y) 48) →
  ∀ (a b : ℕ), a ≠ b → a > 0 → b > 0 → 
    is_factor a 48 → is_factor b 48 → 
    are_non_consecutive a b → ¬(is_factor (a * b) 48) →
    x * y ≤ a * b →
  x * y = 18 :=
sorry

end smallest_non_factor_product_l2082_208280


namespace age_ratio_change_l2082_208216

theorem age_ratio_change (man_age son_age : ℕ) (h1 : man_age = 36) (h2 : son_age = 12) 
  (h3 : man_age = 3 * son_age) : 
  ∃ y : ℕ, man_age + y = 2 * (son_age + y) ∧ y = 12 :=
by sorry

end age_ratio_change_l2082_208216


namespace customers_who_tipped_l2082_208205

/-- The number of customers who left a tip at 'The Greasy Spoon' restaurant -/
theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) :
  initial_customers = 39 →
  additional_customers = 12 →
  non_tipping_customers = 49 →
  initial_customers + additional_customers - non_tipping_customers = 2 :=
by sorry

end customers_who_tipped_l2082_208205


namespace sin_arccos_12_13_l2082_208294

theorem sin_arccos_12_13 : Real.sin (Real.arccos (12/13)) = 5/13 := by
  sorry

end sin_arccos_12_13_l2082_208294


namespace cost_per_meal_is_8_l2082_208224

-- Define the number of adults
def num_adults : ℕ := 2

-- Define the number of children
def num_children : ℕ := 5

-- Define the total bill amount
def total_bill : ℚ := 56

-- Define the total number of people
def total_people : ℕ := num_adults + num_children

-- Theorem to prove
theorem cost_per_meal_is_8 : 
  total_bill / total_people = 8 := by sorry

end cost_per_meal_is_8_l2082_208224


namespace perfect_square_factorization_l2082_208200

theorem perfect_square_factorization (a : ℝ) : a^2 - 2*a + 1 = (a - 1)^2 := by
  sorry

end perfect_square_factorization_l2082_208200


namespace inequality_proof_l2082_208271

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ (1/2) * (a + b + c) := by
  sorry

end inequality_proof_l2082_208271


namespace hyperbola_eccentricity_l2082_208254

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (F₁ F₂ P : ℝ × ℝ),
    -- F₁ and F₂ are the foci of the hyperbola
    (F₁.1 < 0 ∧ F₁.2 = 0) ∧ 
    (F₂.1 > 0 ∧ F₂.2 = 0) ∧ 
    -- P is on the hyperbola in the first quadrant
    (P.1 > 0 ∧ P.2 > 0) ∧
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
    -- P is on the circle with center O and radius OF₁
    (P.1^2 + P.2^2 = F₁.1^2) ∧
    -- The area of triangle PF₁F₂ is a²
    (abs (P.1 * F₂.1 - P.2 * F₂.2) / 2 = a^2) →
    -- The eccentricity is √2
    ((F₂.1 - F₁.1) / 2) / a = Real.sqrt 2 := by
  sorry

end hyperbola_eccentricity_l2082_208254


namespace roes_speed_l2082_208264

/-- Proves that Roe's speed is 40 miles per hour given the conditions of the problem -/
theorem roes_speed (teena_speed : ℝ) (initial_distance : ℝ) (time : ℝ) (final_distance : ℝ)
  (h1 : teena_speed = 55)
  (h2 : initial_distance = 7.5)
  (h3 : time = 1.5)
  (h4 : final_distance = 15)
  (h5 : teena_speed * time - initial_distance - final_distance = time * roe_speed) :
  roe_speed = 40 :=
by sorry

end roes_speed_l2082_208264


namespace equation_represents_line_l2082_208277

/-- The equation (2x + 3y - 1)(-1) = 0 represents a single straight line in the Cartesian plane. -/
theorem equation_represents_line : ∃ (a b c : ℝ) (h : (a, b) ≠ (0, 0)),
  ∀ (x y : ℝ), (2*x + 3*y - 1)*(-1) = 0 ↔ a*x + b*y + c = 0 :=
sorry

end equation_represents_line_l2082_208277


namespace paint_cost_per_kg_l2082_208247

/-- The cost of paint per kg for a cube painting problem -/
theorem paint_cost_per_kg (coverage : ℝ) (total_cost : ℝ) (side_length : ℝ) : 
  coverage = 20 →
  total_cost = 10800 →
  side_length = 30 →
  (total_cost / (6 * side_length^2 / coverage)) = 40 := by
  sorry

end paint_cost_per_kg_l2082_208247


namespace gain_percent_calculation_l2082_208295

theorem gain_percent_calculation (C S : ℝ) (h : C > 0) :
  50 * C = 20 * S → (S - C) / C * 100 = 150 :=
by
  sorry

end gain_percent_calculation_l2082_208295


namespace fraction_addition_theorem_l2082_208249

theorem fraction_addition_theorem : (5 * 8) / 10 + 3 = 7 := by
  sorry

end fraction_addition_theorem_l2082_208249


namespace treadmill_price_correct_l2082_208229

/-- The price of the treadmill at Toby's garage sale. -/
def treadmill_price : ℝ := 133.33

/-- The total sum of money Toby made at the garage sale. -/
def total_money : ℝ := 600

/-- Theorem stating that the treadmill price is correct given the conditions of the garage sale. -/
theorem treadmill_price_correct : 
  treadmill_price + 0.5 * treadmill_price + 3 * treadmill_price = total_money :=
by sorry

end treadmill_price_correct_l2082_208229


namespace power_of_three_mod_ten_l2082_208241

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by
  sorry

end power_of_three_mod_ten_l2082_208241


namespace arthur_walked_four_point_five_miles_l2082_208230

/-- The distance Arthur walked in miles -/
def arthur_distance (blocks_west blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 4.5 miles -/
theorem arthur_walked_four_point_five_miles :
  arthur_distance 8 10 (1/4) = 4.5 := by
  sorry

end arthur_walked_four_point_five_miles_l2082_208230


namespace max_value_theorem_l2082_208284

theorem max_value_theorem (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1)
  (hax : a^x = 2) (hby : b^y = 2) (hab : a + Real.sqrt b = 4) :
  ∃ (M : ℝ), M = 4 ∧ ∀ (z : ℝ), (2/x + 1/y) ≤ z → z ≤ M :=
sorry

end max_value_theorem_l2082_208284


namespace front_wheel_revolutions_l2082_208211

/-- Given a front wheel with perimeter 30 and a back wheel with perimeter 20,
    if the back wheel revolves 360 times, then the front wheel revolves 240 times. -/
theorem front_wheel_revolutions
  (front_perimeter : ℕ) (back_perimeter : ℕ) (back_revolutions : ℕ)
  (h1 : front_perimeter = 30)
  (h2 : back_perimeter = 20)
  (h3 : back_revolutions = 360) :
  (back_perimeter * back_revolutions) / front_perimeter = 240 := by
  sorry

end front_wheel_revolutions_l2082_208211


namespace modulus_of_complex_fraction_l2082_208202

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_complex_fraction_l2082_208202


namespace remainder_of_9876543210_mod_101_l2082_208234

theorem remainder_of_9876543210_mod_101 : 9876543210 % 101 = 4 := by
  sorry

end remainder_of_9876543210_mod_101_l2082_208234


namespace charlie_age_when_jenny_thrice_bobby_l2082_208259

/-- 
Given:
- Jenny is older than Charlie by 12 years
- Charlie is older than Bobby by 7 years

Prove that Charlie is 18 years old when Jenny's age is three times Bobby's age.
-/
theorem charlie_age_when_jenny_thrice_bobby (jenny charlie bobby : ℕ) 
  (h1 : jenny = charlie + 12)
  (h2 : charlie = bobby + 7) :
  (jenny = 3 * bobby) → charlie = 18 := by
  sorry

end charlie_age_when_jenny_thrice_bobby_l2082_208259


namespace derivative_tan_cot_l2082_208204

open Real

theorem derivative_tan_cot (x : ℝ) (k : ℤ) : 
  (∀ k, x ≠ (2 * k + 1) * π / 2 → deriv tan x = 1 / (cos x)^2) ∧
  (∀ k, x ≠ k * π → deriv cot x = -(1 / (sin x)^2)) :=
by
  sorry

end derivative_tan_cot_l2082_208204


namespace middle_number_is_nine_l2082_208287

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem middle_number_is_nine (a b c : ℤ) : 
  is_odd a ∧ is_odd b ∧ is_odd c ∧  -- a, b, c are odd numbers
  b = a + 2 ∧ c = b + 2 ∧            -- a, b, c are consecutive
  a + b + c = a + 20                 -- sum is 20 more than first number
  → b = 9 := by sorry

end middle_number_is_nine_l2082_208287


namespace seashell_collection_l2082_208286

/-- Calculates the remaining number of seashells after Leo gives away a quarter of his collection -/
def remaining_seashells (henry_shells : ℕ) (paul_shells : ℕ) (total_shells : ℕ) : ℕ :=
  let leo_shells := total_shells - henry_shells - paul_shells
  let leo_gave_away := leo_shells / 4
  total_shells - leo_gave_away

theorem seashell_collection (henry_shells paul_shells total_shells : ℕ) 
  (h1 : henry_shells = 11)
  (h2 : paul_shells = 24)
  (h3 : total_shells = 59) :
  remaining_seashells henry_shells paul_shells total_shells = 53 := by
  sorry

end seashell_collection_l2082_208286


namespace roots_less_than_one_l2082_208267

theorem roots_less_than_one (a b : ℝ) 
  (h1 : |a| + |b| < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x : ℝ, x^2 + a*x + b = 0 → |x| < 1 := by
  sorry

end roots_less_than_one_l2082_208267


namespace max_gcd_13n_plus_4_8n_plus_3_l2082_208293

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ k : ℕ+, Nat.gcd (13 * k + 4) (8 * k + 3) = 3) ∧
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 3) := by
  sorry

end max_gcd_13n_plus_4_8n_plus_3_l2082_208293


namespace sum_of_roots_is_fifteen_l2082_208279

/-- A function g: ℝ → ℝ that satisfies g(3+x) = g(3-x) for all real x -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The theorem stating that if g is symmetric about 3 and has exactly 5 distinct real roots,
    then the sum of these roots is 15 -/
theorem sum_of_roots_is_fifteen
  (g : ℝ → ℝ)
  (h_symmetric : SymmetricAboutThree g)
  (h_five_roots : ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, g x = 0) :
  ∃ (s : Finset ℝ), s.card = 5 ∧ (∀ x ∈ s, g x = 0) ∧ (s.sum id = 15) :=
sorry

end sum_of_roots_is_fifteen_l2082_208279


namespace rectangle_breadth_l2082_208285

theorem rectangle_breadth (area : ℝ) (length_ratio : ℝ) :
  area = 460 →
  length_ratio = 1.15 →
  ∃ (breadth : ℝ), 
    area = length_ratio * breadth * breadth ∧
    breadth = 20 := by
  sorry

end rectangle_breadth_l2082_208285


namespace binomial_coefficient_10_3_l2082_208257

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l2082_208257


namespace circumscribed_sphere_surface_area_l2082_208296

/-- The surface area of the circumscribed sphere of a rectangular solid with face diagonals √3, √5, and 2 -/
theorem circumscribed_sphere_surface_area (a b c : ℝ) : 
  a^2 + b^2 = 3 → b^2 + c^2 = 5 → c^2 + a^2 = 4 → 
  4 * Real.pi * ((a^2 + b^2 + c^2) / 4) = 6 * Real.pi := by
  sorry

#check circumscribed_sphere_surface_area

end circumscribed_sphere_surface_area_l2082_208296


namespace systematic_sampling_interval_l2082_208272

def population : ℕ := 1203
def sample_size : ℕ := 40

theorem systematic_sampling_interval :
  ∃ (k : ℕ) (eliminated : ℕ),
    eliminated ≤ sample_size ∧
    (population - eliminated) % sample_size = 0 ∧
    k = (population - eliminated) / sample_size ∧
    k = 30 := by
  sorry

end systematic_sampling_interval_l2082_208272


namespace square_of_difference_l2082_208288

theorem square_of_difference (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2*x := by
  sorry

end square_of_difference_l2082_208288


namespace remainder_8_pow_305_mod_9_l2082_208235

theorem remainder_8_pow_305_mod_9 : 8^305 % 9 = 8 := by sorry

end remainder_8_pow_305_mod_9_l2082_208235


namespace f_composition_negative_two_l2082_208245

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2 else 3^(x + 1)

theorem f_composition_negative_two : f (f (-2)) = 3 := by
  sorry

end f_composition_negative_two_l2082_208245


namespace min_value_function_l2082_208213

theorem min_value_function (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 - x*y + y^2) + Real.sqrt (x^2 - 9*x + 27) + Real.sqrt (y^2 - 15*y + 75) ≥ 7 * Real.sqrt 3 := by
  sorry

end min_value_function_l2082_208213


namespace quadratic_expression_value_l2082_208282

/-- Represents a quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Evaluates the quadratic function at a given x -/
def evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The theorem to be proved -/
theorem quadratic_expression_value (f : QuadraticFunction)
  (h1 : evaluate f (-2) = -2.5)
  (h2 : evaluate f (-1) = -5)
  (h3 : evaluate f 0 = -2.5)
  (h4 : evaluate f 1 = 5)
  (h5 : evaluate f 2 = 17.5) :
  16 * f.a - 4 * f.b + f.c = 17.5 := by
  sorry

end quadratic_expression_value_l2082_208282


namespace circle_radius_from_area_circumference_ratio_l2082_208283

/-- Given a circle with area X and circumference Y, if X/Y = 10, then the radius is 20 -/
theorem circle_radius_from_area_circumference_ratio (X Y : ℝ) (h1 : X > 0) (h2 : Y > 0) (h3 : X / Y = 10) :
  ∃ r : ℝ, r > 0 ∧ X = π * r^2 ∧ Y = 2 * π * r ∧ r = 20 := by
  sorry

end circle_radius_from_area_circumference_ratio_l2082_208283


namespace square_side_length_l2082_208274

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * Real.sqrt 2 = diagonal ∧ side = 2 := by
  sorry

end square_side_length_l2082_208274


namespace incorrect_inference_l2082_208263

-- Define the types for our geometric objects
variable (Point : Type)
variable (Line : Type)
variable (Plane : Type)

-- Define the relationships between geometric objects
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)
variable (line_on_plane : Line → Plane → Prop)

-- State the theorem
theorem incorrect_inference
  (l : Line) (α : Plane) (A : Point) :
  ¬(∀ (l : Line) (α : Plane) (A : Point),
    (¬ line_on_plane l α ∧ on_line A l) → ¬ on_plane A α) :=
by sorry

end incorrect_inference_l2082_208263


namespace fraction_simplification_l2082_208270

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by sorry

end fraction_simplification_l2082_208270


namespace pencil_distribution_ways_l2082_208250

/-- The number of ways to distribute n identical objects among k distinct groups,
    where each group receives at least one object. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- The number of ways to distribute 8 pencils among 4 friends,
    where each friend receives at least one pencil. -/
def pencil_distribution : ℕ :=
  distribute_with_minimum 8 4

theorem pencil_distribution_ways : pencil_distribution = 35 := by
  sorry

end pencil_distribution_ways_l2082_208250


namespace tadpoles_let_go_75_percent_l2082_208233

/-- The percentage of tadpoles let go, given the total caught and number kept -/
def tadpoles_let_go_percentage (total : ℕ) (kept : ℕ) : ℚ :=
  (total - kept : ℚ) / total * 100

/-- Theorem stating that the percentage of tadpoles let go is 75% -/
theorem tadpoles_let_go_75_percent (total : ℕ) (kept : ℕ) 
  (h1 : total = 180) (h2 : kept = 45) : 
  tadpoles_let_go_percentage total kept = 75 := by
  sorry

#eval tadpoles_let_go_percentage 180 45

end tadpoles_let_go_75_percent_l2082_208233


namespace pen_cost_l2082_208248

theorem pen_cost (pen_price pencil_price : ℚ) 
  (h1 : 6 * pen_price + 2 * pencil_price = 348/100)
  (h2 : 3 * pen_price + 4 * pencil_price = 234/100) :
  pen_price = 51/100 := by
sorry

end pen_cost_l2082_208248


namespace two_digit_number_interchange_property_l2082_208231

theorem two_digit_number_interchange_property (a b k : ℕ) (h1 : 10 * a + b = 2 * k * (a + b)) :
  10 * b + a - 3 * (a + b) = (9 - 4 * k) * (a + b) := by sorry

end two_digit_number_interchange_property_l2082_208231


namespace basketball_tryouts_l2082_208276

theorem basketball_tryouts (girls : ℕ) (boys : ℕ) (called_back : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : called_back = 26) :
  girls + boys - called_back = 17 := by
sorry

end basketball_tryouts_l2082_208276


namespace vector_sum_magnitude_l2082_208210

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  b = (Real.sqrt 3, 1) →
  a • b = 0 →
  ‖2 • a + b‖ = 2 * Real.sqrt 2 := by
  sorry

end vector_sum_magnitude_l2082_208210


namespace sons_age_l2082_208209

theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 22 →
  (man_age + 2) = 2 * (son_age + 2) →
  son_age = 20 :=
by
  sorry

end sons_age_l2082_208209


namespace greater_number_problem_l2082_208207

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * (x - y) = 12) (h3 : x > y) : x = 22 := by
  sorry

end greater_number_problem_l2082_208207


namespace binary_1101100_eq_108_l2082_208226

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1101100₂ -/
def binary_1101100 : List Bool := [false, false, true, true, false, true, true]

/-- Theorem stating that 1101100₂ is equal to 108 in decimal -/
theorem binary_1101100_eq_108 : binary_to_decimal binary_1101100 = 108 := by
  sorry

end binary_1101100_eq_108_l2082_208226


namespace franks_fruits_l2082_208262

/-- The total number of fruits left after Frank's dog eats some -/
def fruits_left (apples_on_tree apples_on_ground oranges_on_tree oranges_on_ground apples_eaten oranges_eaten : ℕ) : ℕ :=
  (apples_on_tree + apples_on_ground - apples_eaten) + (oranges_on_tree + oranges_on_ground - oranges_eaten)

/-- Theorem stating the total number of fruits left in Frank's scenario -/
theorem franks_fruits :
  fruits_left 5 8 7 10 3 2 = 25 := by
  sorry

end franks_fruits_l2082_208262


namespace positive_intervals_l2082_208268

-- Define the expression
def f (x : ℝ) : ℝ := (x + 2) * (x - 2)

-- State the theorem
theorem positive_intervals (x : ℝ) : f x > 0 ↔ x < -2 ∨ x > 2 := by
  sorry

end positive_intervals_l2082_208268


namespace initial_deposit_proof_l2082_208225

/-- Calculates the final amount after simple interest --/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the initial deposit was 6200, given the problem conditions --/
theorem initial_deposit_proof (rate : ℝ) : 
  (∃ (principal : ℝ), 
    simpleInterest principal rate 5 = 7200 ∧ 
    simpleInterest principal (rate + 0.03) 5 = 8130) → 
  (∃ (principal : ℝ), 
    simpleInterest principal rate 5 = 7200 ∧ 
    simpleInterest principal (rate + 0.03) 5 = 8130 ∧ 
    principal = 6200) := by
  sorry

#check initial_deposit_proof

end initial_deposit_proof_l2082_208225


namespace ines_peaches_bought_l2082_208298

def peaches_bought (initial_amount : ℕ) (remaining_amount : ℕ) (price_per_pound : ℕ) : ℕ :=
  (initial_amount - remaining_amount) / price_per_pound

theorem ines_peaches_bought :
  peaches_bought 20 14 2 = 3 := by
  sorry

end ines_peaches_bought_l2082_208298


namespace grape_rate_calculation_l2082_208291

def grape_purchase : ℕ := 8
def mango_purchase : ℕ := 10
def mango_rate : ℕ := 55
def total_paid : ℕ := 1110

theorem grape_rate_calculation :
  ∃ (grape_rate : ℕ),
    grape_rate * grape_purchase + mango_rate * mango_purchase = total_paid ∧
    grape_rate = 70 := by
  sorry

end grape_rate_calculation_l2082_208291


namespace jean_jail_time_l2082_208278

/-- Calculates the total jail time for Jean based on his charges --/
def total_jail_time (arson_counts : ℕ) (burglary_charges : ℕ) (arson_sentence : ℕ) (burglary_sentence : ℕ) : ℕ :=
  let petty_larceny_charges := 6 * burglary_charges
  let petty_larceny_sentence := burglary_sentence / 3
  arson_counts * arson_sentence + 
  burglary_charges * burglary_sentence + 
  petty_larceny_charges * petty_larceny_sentence

/-- Theorem stating that Jean's total jail time is 216 months --/
theorem jean_jail_time :
  total_jail_time 3 2 36 18 = 216 := by
  sorry

#eval total_jail_time 3 2 36 18

end jean_jail_time_l2082_208278


namespace total_attendance_percentage_l2082_208219

/-- Represents the departments in the company -/
inductive Department
  | IT
  | HR
  | Marketing

/-- Represents the genders of employees -/
inductive Gender
  | Male
  | Female

/-- Attendance rate for each department and gender -/
def attendance_rate (d : Department) (g : Gender) : ℝ :=
  match d, g with
  | Department.IT, Gender.Male => 0.25
  | Department.IT, Gender.Female => 0.60
  | Department.HR, Gender.Male => 0.30
  | Department.HR, Gender.Female => 0.50
  | Department.Marketing, Gender.Male => 0.10
  | Department.Marketing, Gender.Female => 0.45

/-- Employee composition for each department and gender -/
def employee_composition (d : Department) (g : Gender) : ℝ :=
  match d, g with
  | Department.IT, Gender.Male => 0.40
  | Department.IT, Gender.Female => 0.25
  | Department.HR, Gender.Male => 0.30
  | Department.HR, Gender.Female => 0.20
  | Department.Marketing, Gender.Male => 0.30
  | Department.Marketing, Gender.Female => 0.55

/-- Calculate the total attendance percentage -/
def total_attendance : ℝ :=
  (attendance_rate Department.IT Gender.Male * employee_composition Department.IT Gender.Male) +
  (attendance_rate Department.IT Gender.Female * employee_composition Department.IT Gender.Female) +
  (attendance_rate Department.HR Gender.Male * employee_composition Department.HR Gender.Male) +
  (attendance_rate Department.HR Gender.Female * employee_composition Department.HR Gender.Female) +
  (attendance_rate Department.Marketing Gender.Male * employee_composition Department.Marketing Gender.Male) +
  (attendance_rate Department.Marketing Gender.Female * employee_composition Department.Marketing Gender.Female)

/-- Theorem: The total attendance percentage is 71.75% -/
theorem total_attendance_percentage : total_attendance = 0.7175 := by
  sorry

end total_attendance_percentage_l2082_208219


namespace candy_expenditure_l2082_208253

theorem candy_expenditure (total_spent : ℚ) :
  total_spent = 75 →
  (1 / 2 : ℚ) + (1 / 3 : ℚ) + (1 / 10 : ℚ) + (candy_fraction : ℚ) = 1 →
  candy_fraction * total_spent = 5 :=
by sorry

end candy_expenditure_l2082_208253


namespace sufficient_condition_for_inequality_l2082_208292

theorem sufficient_condition_for_inequality (a : ℝ) (h : a ≥ 5) :
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0 := by sorry

end sufficient_condition_for_inequality_l2082_208292


namespace sticker_collection_probability_l2082_208244

theorem sticker_collection_probability : 
  let total_stickers : ℕ := 18
  let selected_stickers : ℕ := 10
  let uncollected_stickers : ℕ := 6
  let collected_stickers : ℕ := 12
  (Nat.choose uncollected_stickers uncollected_stickers * Nat.choose collected_stickers (selected_stickers - uncollected_stickers)) / 
  Nat.choose total_stickers selected_stickers = 5 / 442 := by
sorry

end sticker_collection_probability_l2082_208244


namespace perfect_square_trinomial_m_value_l2082_208201

/-- A polynomial is a perfect square trinomial if it can be expressed as (ax + b)^2 -/
def is_perfect_square_trinomial (a b m : ℝ) : Prop :=
  ∀ x, x^2 + m*x + 4 = (a*x + b)^2

/-- If x^2 + mx + 4 is a perfect square trinomial, then m = 4 or m = -4 -/
theorem perfect_square_trinomial_m_value (m : ℝ) :
  (∃ a b : ℝ, is_perfect_square_trinomial a b m) → m = 4 ∨ m = -4 := by
  sorry

end perfect_square_trinomial_m_value_l2082_208201


namespace election_vote_count_l2082_208206

/-- Given that the ratio of votes for candidate A to candidate B is 2:1,
    and candidate A received 14 votes, prove that the total number of
    votes for both candidates is 21. -/
theorem election_vote_count (votes_A : ℕ) (votes_B : ℕ) : 
  votes_A = 14 → 
  votes_A = 2 * votes_B → 
  votes_A + votes_B = 21 :=
by
  sorry

end election_vote_count_l2082_208206


namespace six_hundred_million_scientific_notation_l2082_208240

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem six_hundred_million_scientific_notation :
  toScientificNotation 600000000 = ScientificNotation.mk 6 7 (by norm_num) :=
sorry

end six_hundred_million_scientific_notation_l2082_208240


namespace floor_negative_seven_fourths_l2082_208212

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end floor_negative_seven_fourths_l2082_208212


namespace book_e_chapters_l2082_208290

/-- Represents the number of chapters in each book --/
structure BookChapters where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The problem statement --/
def book_chapters_problem (total : ℕ) (books : BookChapters) : Prop :=
  books.a = 17 ∧
  books.b = books.a + 5 ∧
  books.c = books.b - 7 ∧
  books.d = 2 * books.c ∧
  total = 97 ∧
  total = books.a + books.b + books.c + books.d + books.e

/-- The theorem to prove --/
theorem book_e_chapters (total : ℕ) (books : BookChapters) 
  (h : book_chapters_problem total books) : books.e = 13 := by
  sorry


end book_e_chapters_l2082_208290


namespace average_salary_feb_to_may_l2082_208215

def average_salary_jan_to_apr : ℝ := 8000
def salary_may : ℝ := 6500
def salary_jan : ℝ := 5700

theorem average_salary_feb_to_may :
  let total_jan_to_apr := average_salary_jan_to_apr * 4
  let total_feb_to_apr := total_jan_to_apr - salary_jan
  let total_feb_to_may := total_feb_to_apr + salary_may
  (total_feb_to_may / 4 : ℝ) = 8200 := by
  sorry

end average_salary_feb_to_may_l2082_208215


namespace bookman_purchase_theorem_l2082_208214

theorem bookman_purchase_theorem (hardback_price : ℕ) (paperback_price : ℕ) 
  (hardback_count : ℕ) (total_sold : ℕ) (remaining_value : ℕ) :
  hardback_price = 20 →
  paperback_price = 10 →
  hardback_count = 10 →
  total_sold = 14 →
  remaining_value = 360 →
  ∃ (total_copies : ℕ),
    total_copies = hardback_count + (remaining_value / paperback_price) + (total_sold - hardback_count) ∧
    total_copies = 50 :=
by sorry

end bookman_purchase_theorem_l2082_208214


namespace exactly_one_multiple_of_five_l2082_208297

theorem exactly_one_multiple_of_five (a b : ℤ) (h : 24 * a^2 + 1 = b^2) :
  (a % 5 = 0 ∧ b % 5 ≠ 0) ∨ (a % 5 ≠ 0 ∧ b % 5 = 0) :=
by sorry

end exactly_one_multiple_of_five_l2082_208297


namespace valid_12_letter_words_mod_1000_l2082_208273

/-- Represents a letter in Zuminglish -/
inductive ZumLetter
| M
| O
| P

/-- Represents a Zuminglish word -/
def ZumWord := List ZumLetter

/-- Checks if a letter is a vowel -/
def isVowel (l : ZumLetter) : Bool :=
  match l with
  | ZumLetter.O => true
  | _ => false

/-- Checks if a Zuminglish word is valid -/
def isValidWord (w : ZumWord) : Bool :=
  sorry

/-- Counts the number of valid n-letter Zuminglish words -/
def countValidWords (n : Nat) : Nat :=
  sorry

/-- The main theorem: number of valid 12-letter Zuminglish words modulo 1000 -/
theorem valid_12_letter_words_mod_1000 :
  countValidWords 12 % 1000 = 416 := by
  sorry

end valid_12_letter_words_mod_1000_l2082_208273


namespace problem_one_problem_two_l2082_208256

-- Problem 1
theorem problem_one : Real.sqrt 3 - Real.sqrt 3 * (1 - Real.sqrt 3) = 3 := by sorry

-- Problem 2
theorem problem_two : (Real.sqrt 3 - 2)^2 + Real.sqrt 12 + 6 * Real.sqrt (1/3) = 7 := by sorry

end problem_one_problem_two_l2082_208256


namespace five_level_pieces_l2082_208228

/-- Calculates the number of pieces in a square-based pyramid -/
def pyramid_pieces (levels : ℕ) : ℕ :=
  let rods := levels * (levels + 1) * 2
  let connectors := levels * 4
  rods + connectors

/-- Properties of a two-level square-based pyramid -/
axiom two_level_total : pyramid_pieces 2 = 20
axiom two_level_rods : 2 * (2 + 1) * 2 = 12
axiom two_level_connectors : 2 * 4 = 8

/-- Theorem: A five-level square-based pyramid requires 80 pieces -/
theorem five_level_pieces : pyramid_pieces 5 = 80 := by
  sorry

end five_level_pieces_l2082_208228
