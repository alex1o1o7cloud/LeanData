import Mathlib

namespace equation_solution_l262_26240

theorem equation_solution :
  ∃ x : ℝ, (7 - 2*x = -3) ∧ (x = 5) := by
  sorry

end equation_solution_l262_26240


namespace circle_radius_given_area_circumference_ratio_l262_26284

/-- Given a circle with area A and circumference C, if A/C = 15, then the radius is 30 -/
theorem circle_radius_given_area_circumference_ratio (A C : ℝ) (h : A / C = 15) :
  ∃ (r : ℝ), A = π * r^2 ∧ C = 2 * π * r ∧ r = 30 := by
  sorry

end circle_radius_given_area_circumference_ratio_l262_26284


namespace harry_hike_water_remaining_l262_26202

/-- Calculates the remaining water in Harry's canteen after a hike -/
def remaining_water (initial_water : ℝ) (hike_distance : ℝ) (hike_duration : ℝ) 
  (leak_rate : ℝ) (last_mile_consumption : ℝ) (first_miles_consumption_rate : ℝ) : ℝ :=
  initial_water - 
  (leak_rate * hike_duration) - 
  (first_miles_consumption_rate * (hike_distance - 1)) - 
  last_mile_consumption

/-- Theorem stating that the remaining water in Harry's canteen is 2 cups -/
theorem harry_hike_water_remaining :
  remaining_water 11 7 3 1 3 0.5 = 2 := by
  sorry

end harry_hike_water_remaining_l262_26202


namespace cube_volumes_theorem_l262_26252

/-- The edge length of the first cube in centimeters -/
def x : ℝ := 18

/-- The volume of a cube with edge length l -/
def cube_volume (l : ℝ) : ℝ := l^3

/-- The edge length of the second cube in centimeters -/
def second_edge : ℝ := x - 4

/-- The edge length of the third cube in centimeters -/
def third_edge : ℝ := second_edge - 2

/-- The volume of water remaining in the first cube after filling the second -/
def remaining_first : ℝ := cube_volume x - cube_volume second_edge

/-- The volume of water remaining in the second cube after filling the third -/
def remaining_second : ℝ := cube_volume second_edge - cube_volume third_edge

theorem cube_volumes_theorem : 
  remaining_first = 3 * remaining_second + 40 ∧ 
  cube_volume x = 5832 ∧ 
  cube_volume second_edge = 2744 ∧ 
  cube_volume third_edge = 1728 := by
  sorry

end cube_volumes_theorem_l262_26252


namespace simplify_fraction_l262_26208

theorem simplify_fraction : 9 * (12 / 7) * (-35 / 36) = -15 := by
  sorry

end simplify_fraction_l262_26208


namespace halls_per_floor_wing2_is_9_l262_26237

/-- A hotel with two wings -/
structure Hotel where
  total_rooms : ℕ
  wing1_floors : ℕ
  wing1_halls_per_floor : ℕ
  wing1_rooms_per_hall : ℕ
  wing2_floors : ℕ
  wing2_rooms_per_hall : ℕ

/-- The number of halls on each floor of the second wing -/
def halls_per_floor_wing2 (h : Hotel) : ℕ :=
  (h.total_rooms - h.wing1_floors * h.wing1_halls_per_floor * h.wing1_rooms_per_hall) /
  (h.wing2_floors * h.wing2_rooms_per_hall)

/-- Theorem stating that the number of halls on each floor of the second wing is 9 -/
theorem halls_per_floor_wing2_is_9 (h : Hotel)
  (h_total : h.total_rooms = 4248)
  (h_wing1_floors : h.wing1_floors = 9)
  (h_wing1_halls : h.wing1_halls_per_floor = 6)
  (h_wing1_rooms : h.wing1_rooms_per_hall = 32)
  (h_wing2_floors : h.wing2_floors = 7)
  (h_wing2_rooms : h.wing2_rooms_per_hall = 40) :
  halls_per_floor_wing2 h = 9 := by
  sorry

#eval halls_per_floor_wing2 {
  total_rooms := 4248,
  wing1_floors := 9,
  wing1_halls_per_floor := 6,
  wing1_rooms_per_hall := 32,
  wing2_floors := 7,
  wing2_rooms_per_hall := 40
}

end halls_per_floor_wing2_is_9_l262_26237


namespace splitting_number_345_l262_26286

/-- The first splitting number for a given base number -/
def first_split (n : ℕ) : ℕ := n * (n - 1) + 1

/-- The property that 345 is one of the splitting numbers of m³ -/
def is_splitting_number (m : ℕ) : Prop :=
  m > 1 ∧ ∃ k, k ≥ 0 ∧ k < m ∧ first_split m + 2 * k = 345

theorem splitting_number_345 (m : ℕ) :
  is_splitting_number m → m = 19 := by
  sorry

end splitting_number_345_l262_26286


namespace ceiling_minus_y_l262_26238

theorem ceiling_minus_y (x : ℝ) : 
  let y := 2 * x
  let f := y - ⌊y⌋
  (⌈y⌉ - ⌊y⌋ = 1) → (0 < f ∧ f < 1) → (⌈y⌉ - y = 1 - f) :=
by sorry

end ceiling_minus_y_l262_26238


namespace extreme_values_and_inequality_l262_26200

def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 + b*x + c

theorem extreme_values_and_inequality 
  (a b c : ℝ) 
  (h1 : ∃ y, (deriv (f a b c)) (-1) = y ∧ y = 0)
  (h2 : ∃ y, (deriv (f a b c)) 3 = y ∧ y = 0)
  (h3 : ∀ x ∈ Set.Icc (-2) 6, f a b c x < c^2 + 4*c) :
  a = 3 ∧ b = -9 ∧ (c > 6 ∨ c < -9) := by sorry

end extreme_values_and_inequality_l262_26200


namespace truck_driver_pay_l262_26251

/-- Calculates the pay for a round trip given the pay rate per mile and one-way distance -/
def round_trip_pay (pay_rate : ℚ) (one_way_distance : ℕ) : ℚ :=
  2 * pay_rate * one_way_distance

/-- Proves that given a pay rate of $0.40 per mile and a one-way trip distance of 400 miles,
    the total pay for a round trip is $320 -/
theorem truck_driver_pay : round_trip_pay (40/100) 400 = 320 := by
  sorry

end truck_driver_pay_l262_26251


namespace probability_point_on_subsegment_l262_26250

/-- The probability of a randomly chosen point on a segment also lying on its subsegment -/
theorem probability_point_on_subsegment 
  (L ℓ : ℝ) 
  (hL : L = 40) 
  (hℓ : ℓ = 15) 
  (h_pos_L : L > 0) 
  (h_pos_ℓ : ℓ > 0) 
  (h_subsegment : ℓ ≤ L) :
  ℓ / L = 3 / 8 :=
sorry

end probability_point_on_subsegment_l262_26250


namespace conic_section_equation_l262_26255

/-- A conic section that satisfies specific conditions -/
structure ConicSection where
  -- The conic section passes through these two points
  point_a : (ℝ × ℝ)
  point_b : (ℝ × ℝ)
  -- The conic section shares a common asymptote with this hyperbola
  asymptote_hyperbola : (ℝ → ℝ → Prop)
  -- The conic section is a hyperbola with this focal length
  focal_length : ℝ

/-- The standard equation of a hyperbola -/
def standard_hyperbola_equation (a b : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating the standard equation of the conic section -/
theorem conic_section_equation (c : ConicSection)
  (h1 : c.point_a = (2, -Real.sqrt 2 / 2))
  (h2 : c.point_b = (-Real.sqrt 2, -Real.sqrt 3 / 2))
  (h3 : c.asymptote_hyperbola = standard_hyperbola_equation 5 3)
  (h4 : c.focal_length = 8) :
  (standard_hyperbola_equation 10 6 = c.asymptote_hyperbola) ∨
  (standard_hyperbola_equation 6 10 = c.asymptote_hyperbola) :=
sorry

end conic_section_equation_l262_26255


namespace washing_machine_cost_l262_26256

/-- The cost of a washing machine and dryer, with a discount applied --/
theorem washing_machine_cost 
  (washing_machine_cost : ℝ) 
  (dryer_cost : ℝ) 
  (discount_rate : ℝ) 
  (total_after_discount : ℝ) :
  washing_machine_cost = 100 ∧ 
  dryer_cost = washing_machine_cost - 30 ∧
  discount_rate = 0.1 ∧
  total_after_discount = 153 ∧
  (1 - discount_rate) * (washing_machine_cost + dryer_cost) = total_after_discount →
  washing_machine_cost = 100 := by
sorry

end washing_machine_cost_l262_26256


namespace rectangle_area_l262_26290

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 6), G(y, 6), and H(y, 0),
    if the area of the rectangle is 42 square units and y > 0, then y = 7. -/
theorem rectangle_area (y : ℝ) : y > 0 → (6 * y = 42) → y = 7 := by
  sorry

end rectangle_area_l262_26290


namespace max_value_on_interval_max_value_is_11_l262_26266

def f (x : ℝ) : ℝ := x^4 - 8*x^2 + 2

theorem max_value_on_interval (a b : ℝ) (h : a ≤ b) :
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c :=
sorry

theorem max_value_is_11 :
  ∃ c ∈ Set.Icc (-1) 3, f c = 11 ∧ ∀ x ∈ Set.Icc (-1) 3, f x ≤ f c :=
sorry

end max_value_on_interval_max_value_is_11_l262_26266


namespace inequality_proof_l262_26291

theorem inequality_proof (a b x : ℝ) (ha : a > 0) (hb : b > 0) :
  a * b ≤ (a * Real.sin x ^ 2 + b * Real.cos x ^ 2) * (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) ∧
  (a * Real.sin x ^ 2 + b * Real.cos x ^ 2) * (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) ≤ (a + b) ^ 2 / 4 := by
  sorry

end inequality_proof_l262_26291


namespace single_point_condition_l262_26267

/-- The equation of the curve -/
def curve_equation (x y c : ℝ) : Prop :=
  3 * x^2 + y^2 + 6 * x - 12 * y + c = 0

/-- The curve is a single point -/
def is_single_point (c : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, curve_equation p.1 p.2 c

/-- The value of c for which the curve is a single point -/
theorem single_point_condition :
  ∃! c : ℝ, is_single_point c ∧ c = 39 :=
sorry

end single_point_condition_l262_26267


namespace square_floor_tiles_l262_26242

/-- Represents a square floor covered with tiles -/
structure TiledFloor where
  side_length : ℕ
  is_even : Even side_length
  diagonal_tiles : ℕ
  h_diagonal : diagonal_tiles = 2 * side_length

/-- The total number of tiles on a square floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.side_length ^ 2

theorem square_floor_tiles (floor : TiledFloor) 
  (h_diagonal_count : floor.diagonal_tiles = 88) : 
  total_tiles floor = 1936 := by
  sorry

end square_floor_tiles_l262_26242


namespace writer_productivity_l262_26296

/-- Given a writer's manuscript details, calculate their writing productivity. -/
theorem writer_productivity (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ) :
  total_words = 60000 →
  total_hours = 120 →
  break_hours = 20 →
  (total_words : ℝ) / (total_hours - break_hours : ℝ) = 600 := by
  sorry

end writer_productivity_l262_26296


namespace sun_xing_zhe_product_sum_l262_26283

theorem sun_xing_zhe_product_sum : ∃ (S X Z : ℕ), 
  (S < 10 ∧ X < 10 ∧ Z < 10) ∧ 
  (100 * S + 10 * X + Z) * (100 * Z + 10 * X + S) = 78445 ∧
  (100 * S + 10 * X + Z) + (100 * Z + 10 * X + S) = 1372 := by
  sorry

end sun_xing_zhe_product_sum_l262_26283


namespace remainder_problem_l262_26262

theorem remainder_problem : (((1234567 % 135) * 5) % 27) = 1 := by
  sorry

end remainder_problem_l262_26262


namespace ratio_of_segments_l262_26299

/-- Given four points P, Q, R, and S on a line in that order, 
    with PQ = 3, QR = 7, and PS = 17, the ratio of PR to QS is 10/7. -/
theorem ratio_of_segments (P Q R S : ℝ) : 
  Q < R ∧ R < S ∧ Q - P = 3 ∧ R - Q = 7 ∧ S - P = 17 → 
  (R - P) / (S - Q) = 10 / 7 := by
sorry

end ratio_of_segments_l262_26299


namespace gcd_120_75_l262_26278

theorem gcd_120_75 : Nat.gcd 120 75 = 15 := by
  sorry

end gcd_120_75_l262_26278


namespace sum_of_a_and_b_is_one_l262_26245

theorem sum_of_a_and_b_is_one (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : a / (1 - i) = 1 - b * i) : a + b = 1 := by
  sorry

end sum_of_a_and_b_is_one_l262_26245


namespace completing_square_quadratic_l262_26260

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 8*x + 6 = 0 ↔ (x - 4)^2 = 10 := by
  sorry

end completing_square_quadratic_l262_26260


namespace power_of_fraction_five_sevenths_sixth_l262_26279

theorem power_of_fraction_five_sevenths_sixth : (5 : ℚ) / 7 ^ 6 = 15625 / 117649 := by
  sorry

end power_of_fraction_five_sevenths_sixth_l262_26279


namespace simplify_polynomial_l262_26293

theorem simplify_polynomial (s : ℝ) : (2*s^2 - 5*s + 3) - (s^2 + 4*s - 6) = s^2 - 9*s + 9 := by
  sorry

end simplify_polynomial_l262_26293


namespace inequality_proof_l262_26235

theorem inequality_proof (a b : ℝ) (h : a + b > 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b :=
by sorry

end inequality_proof_l262_26235


namespace smallest_circle_covering_region_line_intersecting_circle_l262_26295

-- Define the planar region
def planar_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + 2*y - 4 ≤ 0

-- Define the circle (C)
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 5

-- Define the line (l)
def line_l (x y : ℝ) : Prop :=
  y = x - 1 + Real.sqrt 5 ∨ y = x - 1 - Real.sqrt 5

-- Theorem for the smallest circle covering the region
theorem smallest_circle_covering_region :
  (∀ x y, planar_region x y → circle_C x y) ∧
  (∀ x' y', (∀ x y, planar_region x y → (x - x')^2 + (y - y')^2 ≤ r'^2) →
    r'^2 ≥ 5) :=
sorry

-- Theorem for the line intersecting the circle
theorem line_intersecting_circle :
  ∃ A B : ℝ × ℝ,
    A ≠ B ∧
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    ((A.1 - 2) * (B.1 - 2) + (A.2 - 1) * (B.2 - 1) = 0) :=
sorry

end smallest_circle_covering_region_line_intersecting_circle_l262_26295


namespace exists_valid_8_stamp_combination_min_8_stamps_for_valid_combination_min_stamps_is_8_l262_26276

/-- Represents the number of stamps of each denomination -/
structure StampCombination :=
  (s06 : ℕ)  -- number of 0.6 yuan stamps
  (s08 : ℕ)  -- number of 0.8 yuan stamps
  (s11 : ℕ)  -- number of 1.1 yuan stamps

/-- The total postage value of a stamp combination -/
def postageValue (sc : StampCombination) : ℚ :=
  0.6 * sc.s06 + 0.8 * sc.s08 + 1.1 * sc.s11

/-- The total number of stamps in a combination -/
def totalStamps (sc : StampCombination) : ℕ :=
  sc.s06 + sc.s08 + sc.s11

/-- A stamp combination is valid if it exactly equals the required postage -/
def isValidCombination (sc : StampCombination) : Prop :=
  postageValue sc = 7.5

/-- There exists a valid stamp combination using 8 stamps -/
theorem exists_valid_8_stamp_combination :
  ∃ (sc : StampCombination), isValidCombination sc ∧ totalStamps sc = 8 :=
sorry

/-- Any valid stamp combination uses at least 8 stamps -/
theorem min_8_stamps_for_valid_combination :
  ∀ (sc : StampCombination), isValidCombination sc → totalStamps sc ≥ 8 :=
sorry

/-- The minimum number of stamps required for a valid combination is 8 -/
theorem min_stamps_is_8 :
  (∃ (sc : StampCombination), isValidCombination sc ∧ totalStamps sc = 8) ∧
  (∀ (sc : StampCombination), isValidCombination sc → totalStamps sc ≥ 8) :=
sorry

end exists_valid_8_stamp_combination_min_8_stamps_for_valid_combination_min_stamps_is_8_l262_26276


namespace yellow_surface_fraction_l262_26229

/-- Represents a large cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  yellow_cubes : ℕ
  blue_cubes : ℕ

/-- Calculates the minimum possible yellow surface area for a given large cube configuration -/
def min_yellow_surface_area (cube : LargeCube) : ℚ :=
  sorry

/-- Calculates the total surface area of the large cube -/
def total_surface_area (cube : LargeCube) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem yellow_surface_fraction (cube : LargeCube) 
  (h1 : cube.edge_length = 4)
  (h2 : cube.total_small_cubes = 64)
  (h3 : cube.yellow_cubes = 14)
  (h4 : cube.blue_cubes = 50)
  (h5 : cube.yellow_cubes + cube.blue_cubes = cube.total_small_cubes) :
  (min_yellow_surface_area cube) / (total_surface_area cube) = 7 / 48 :=
sorry

end yellow_surface_fraction_l262_26229


namespace ellipse_properties_l262_26288

-- Define the ellipse C
def ellipse_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (n : ℝ) : ℝ × ℝ := (2, n)

-- Define the perpendicular condition
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Define the isosceles condition
def isosceles (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Theorem statement
theorem ellipse_properties :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x y, ellipse_C a x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  ∃ m n, perpendicular (A m) (B n) F₁ ∧
         isosceles (A m) (B n) F₁ ∧
         abs ((A m).1 - (B n).1) * abs ((A m).2 - F₁.2) / 2 = 6 * Real.sqrt 10 :=
sorry

end ellipse_properties_l262_26288


namespace divisibility_rule_3_decimal_true_divisibility_rule_3_duodecimal_false_l262_26248

/-- Represents a positional numeral system with a given base. -/
structure NumeralSystem (base : ℕ) where
  (digits : List ℕ)
  (valid_digits : ∀ d ∈ digits, d < base)

/-- The value of a number in a given numeral system. -/
def value (base : ℕ) (num : NumeralSystem base) : ℕ :=
  (num.digits.reverse.enum.map (λ (i, d) => d * base ^ i)).sum

/-- The sum of digits of a number in a given numeral system. -/
def digit_sum (base : ℕ) (num : NumeralSystem base) : ℕ :=
  num.digits.sum

/-- Divisibility rule for 3 in a given numeral system. -/
def divisibility_rule_3 (base : ℕ) : Prop :=
  ∀ (num : NumeralSystem base), 
    (value base num) % 3 = 0 ↔ (digit_sum base num) % 3 = 0

theorem divisibility_rule_3_decimal_true : 
  divisibility_rule_3 10 := by sorry

theorem divisibility_rule_3_duodecimal_false : 
  ¬(divisibility_rule_3 12) := by sorry

end divisibility_rule_3_decimal_true_divisibility_rule_3_duodecimal_false_l262_26248


namespace lisas_marbles_problem_l262_26234

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisas_marbles_problem :
  min_additional_marbles 12 40 = 38 := by
  sorry

#eval min_additional_marbles 12 40

end lisas_marbles_problem_l262_26234


namespace sum_of_possible_y_values_l262_26258

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- Two angles of the triangle
  angle1 : ℝ
  angle2 : ℝ
  -- The triangle is isosceles
  isIsosceles : angle1 = angle2 ∨ angle1 = 180 - angle1 - angle2 ∨ angle2 = 180 - angle1 - angle2
  -- The sum of angles in a triangle is 180°
  sumOfAngles : angle1 + angle2 + (180 - angle1 - angle2) = 180

-- Theorem statement
theorem sum_of_possible_y_values (t : IsoscelesTriangle) (h1 : t.angle1 = 40 ∨ t.angle2 = 40) :
  ∃ y1 y2 : ℝ, (y1 = t.angle1 ∨ y1 = t.angle2) ∧ 
             (y2 = t.angle1 ∨ y2 = t.angle2) ∧
             y1 ≠ y2 ∧
             y1 + y2 = 140 := by
  sorry

end sum_of_possible_y_values_l262_26258


namespace marc_total_spend_l262_26218

/-- The total amount spent by Marc on his purchase of model cars, paint bottles, and paintbrushes. -/
def total_spent (num_cars num_paint num_brushes : ℕ) (price_car price_paint price_brush : ℚ) : ℚ :=
  num_cars * price_car + num_paint * price_paint + num_brushes * price_brush

/-- Theorem stating that Marc's total spend is $160 given his purchases. -/
theorem marc_total_spend :
  total_spent 5 5 5 20 10 2 = 160 := by
  sorry

end marc_total_spend_l262_26218


namespace rectangle_length_eq_five_l262_26254

/-- The length of a rectangle with width 20 cm and perimeter equal to that of a regular pentagon with side length 10 cm is 5 cm. -/
theorem rectangle_length_eq_five (width : ℝ) (pentagon_side : ℝ) (length : ℝ) : 
  width = 20 →
  pentagon_side = 10 →
  2 * (length + width) = 5 * pentagon_side →
  length = 5 := by
  sorry

end rectangle_length_eq_five_l262_26254


namespace math_textbooks_in_same_box_l262_26285

def total_textbooks : ℕ := 13
def math_textbooks : ℕ := 4
def box1_capacity : ℕ := 4
def box2_capacity : ℕ := 4
def box3_capacity : ℕ := 5

def probability_all_math_in_one_box : ℚ := 1 / 4120

theorem math_textbooks_in_same_box :
  let total_arrangements := (total_textbooks.choose box1_capacity) *
                            ((total_textbooks - box1_capacity).choose box2_capacity) *
                            ((total_textbooks - box1_capacity - box2_capacity).choose box3_capacity)
  let favorable_outcomes := (total_textbooks - math_textbooks).choose 1 *
                            ((total_textbooks - math_textbooks - 1).choose box1_capacity) *
                            ((total_textbooks - math_textbooks - 1 - box1_capacity).choose box2_capacity)
  (favorable_outcomes : ℚ) / total_arrangements = probability_all_math_in_one_box :=
sorry

end math_textbooks_in_same_box_l262_26285


namespace nine_rings_puzzle_5_l262_26213

def nine_rings_puzzle (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Define for 0 to satisfy recursion
  | 1 => 1
  | n + 1 =>
    if n % 2 = 0 then
      2 * nine_rings_puzzle n + 2
    else
      2 * nine_rings_puzzle n - 1

theorem nine_rings_puzzle_5 :
  nine_rings_puzzle 5 = 16 :=
by
  sorry

end nine_rings_puzzle_5_l262_26213


namespace division_with_maximum_remainder_l262_26275

theorem division_with_maximum_remainder :
  ∃ (star : ℕ) (triangle : ℕ),
    star / 6 = 102 ∧
    star % 6 = triangle ∧
    triangle ≤ 5 ∧
    (∀ (s t : ℕ), s / 6 = 102 ∧ s % 6 = t → t ≤ triangle) ∧
    triangle = 5 ∧
    star = 617 := by
  sorry

end division_with_maximum_remainder_l262_26275


namespace inverse_proportion_problem_l262_26246

/-- Given that p is inversely proportional to q+2 and p = 1 when q = 4,
    prove that p = 2 when q = 1. -/
theorem inverse_proportion_problem (p q : ℝ) (h : ∃ k : ℝ, ∀ q, p = k / (q + 2)) 
  (h1 : p = 1 → q = 4) : p = 2 → q = 1 := by
  sorry

end inverse_proportion_problem_l262_26246


namespace solution_set_inequality_l262_26212

theorem solution_set_inequality (x : ℝ) : 
  (Set.Ioo (-2 : ℝ) 0).Nonempty ∧ 
  (∀ y ∈ Set.Ioo (-2 : ℝ) 0, |1 + y + y^2/2| < 1) ∧
  (∀ z : ℝ, z ∉ Set.Ioo (-2 : ℝ) 0 → |1 + z + z^2/2| ≥ 1) :=
by sorry

end solution_set_inequality_l262_26212


namespace hockey_team_starters_l262_26273

/-- The number of ways to choose starters from a hockey team with quadruplets -/
def chooseStarters (totalPlayers : ℕ) (quadruplets : ℕ) (starters : ℕ) (maxQuadruplets : ℕ) : ℕ :=
  (Nat.choose (totalPlayers - quadruplets) starters) +
  (quadruplets * Nat.choose (totalPlayers - quadruplets) (starters - 1)) +
  (Nat.choose quadruplets 2 * Nat.choose (totalPlayers - quadruplets) (starters - 2))

/-- The theorem stating the correct number of ways to choose starters -/
theorem hockey_team_starters :
  chooseStarters 18 4 7 2 = 27456 := by
  sorry

end hockey_team_starters_l262_26273


namespace boys_in_class_l262_26236

/-- Proves that in a class of 20 students, if exactly one-third of the boys sit with a girl
    and exactly one-half of the girls sit with a boy, then there are 12 boys in the class. -/
theorem boys_in_class (total_students : ℕ) (boys : ℕ) (girls : ℕ) :
  total_students = 20 →
  boys + girls = total_students →
  (boys / 3 : ℚ) = (girls / 2 : ℚ) →
  boys = 12 := by
  sorry

end boys_in_class_l262_26236


namespace g_value_at_9_l262_26211

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧  -- g is a cubic polynomial
  (g 0 = -1) ∧  -- g(0) = -1
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2)  -- roots of g are squares of roots of f

-- State the theorem
theorem g_value_at_9 (g : ℝ → ℝ) (hg : g_properties g) : g 9 = 899 := by
  sorry

end g_value_at_9_l262_26211


namespace johnson_finley_class_difference_l262_26231

theorem johnson_finley_class_difference (finley_class : ℕ) (johnson_class : ℕ) : 
  finley_class = 24 →
  johnson_class = 22 →
  johnson_class > finley_class / 2 →
  johnson_class - finley_class / 2 = 10 := by
sorry

end johnson_finley_class_difference_l262_26231


namespace letters_identity_l262_26210

-- Define the Letter type
inductive Letter
| A
| B

-- Define a function to represent whether a letter tells the truth
def tellsTruth (l : Letter) : Bool :=
  match l with
  | Letter.A => true
  | Letter.B => false

-- Define the statements made by each letter
def statement1 (l1 l2 l3 : Letter) : Prop :=
  (l1 = l2 ∧ l1 ≠ l3) ∨ (l1 = l3 ∧ l1 ≠ l2)

def statement2 (l1 l2 l3 : Letter) : Prop :=
  (l1 = Letter.A ∧ l2 = Letter.A ∧ l3 = Letter.B) ∨
  (l1 = Letter.A ∧ l2 = Letter.B ∧ l3 = Letter.B) ∨
  (l1 = Letter.B ∧ l2 = Letter.A ∧ l3 = Letter.B) ∨
  (l1 = Letter.B ∧ l2 = Letter.B ∧ l3 = Letter.B)

def statement3 (l1 l2 l3 : Letter) : Prop :=
  (l1 = Letter.B ∧ l2 ≠ Letter.B ∧ l3 ≠ Letter.B) ∨
  (l1 ≠ Letter.B ∧ l2 = Letter.B ∧ l3 ≠ Letter.B) ∨
  (l1 ≠ Letter.B ∧ l2 ≠ Letter.B ∧ l3 = Letter.B)

-- Define the main theorem
theorem letters_identity :
  ∃! (l1 l2 l3 : Letter),
    (tellsTruth l1 = statement1 l1 l2 l3) ∧
    (tellsTruth l2 = statement2 l1 l2 l3) ∧
    (tellsTruth l3 = statement3 l1 l2 l3) ∧
    l1 = Letter.B ∧ l2 = Letter.A ∧ l3 = Letter.A :=
by sorry

end letters_identity_l262_26210


namespace mean_transformation_l262_26230

theorem mean_transformation (x₁ x₂ x₃ x₄ : ℝ) 
  (h : (x₁ + x₂ + x₃ + x₄) / 4 = 5) : 
  ((x₁ + 1) + (x₂ + 2) + (x₃ + x₄ + 4) + (5 + 5)) / 4 = 8 := by
  sorry

end mean_transformation_l262_26230


namespace equation_has_real_roots_l262_26272

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
sorry

end equation_has_real_roots_l262_26272


namespace bowling_team_average_weight_l262_26207

theorem bowling_team_average_weight 
  (initial_players : ℕ) 
  (initial_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (h1 : initial_players = 7) 
  (h2 : initial_average = 94) 
  (h3 : new_player1_weight = 110) 
  (h4 : new_player2_weight = 60) :
  let total_weight := initial_players * initial_average + new_player1_weight + new_player2_weight
  let new_players := initial_players + 2
  (total_weight / new_players : ℝ) = 92 := by
sorry

end bowling_team_average_weight_l262_26207


namespace initial_books_l262_26227

theorem initial_books (initial : ℕ) (sold : ℕ) (bought : ℕ) (final : ℕ) : 
  sold = 11 → bought = 23 → final = 45 → initial - sold + bought = final → initial = 33 := by
  sorry

end initial_books_l262_26227


namespace sarahs_test_score_l262_26244

theorem sarahs_test_score 
  (hunter_score : ℕ) 
  (john_score : ℕ) 
  (grant_score : ℕ) 
  (sarah_score : ℕ) 
  (hunter_score_val : hunter_score = 45)
  (john_score_def : john_score = 2 * hunter_score)
  (grant_score_def : grant_score = john_score + 10)
  (sarah_score_def : sarah_score = grant_score - 5) :
  sarah_score = 95 := by
sorry

end sarahs_test_score_l262_26244


namespace triangle_perpendicular_theorem_l262_26232

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the foot of a perpendicular
def perpFoot (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Main theorem
theorem triangle_perpendicular_theorem (ABC : Triangle) :
  let A := ABC.A
  let B := ABC.B
  let C := ABC.C
  let D := perpFoot A B C
  length A B = 12 →
  length A C = 20 →
  (length B D) / (length C D) = 3 / 4 →
  length A D = 36 * Real.sqrt 14 / 7 := by
  sorry

end triangle_perpendicular_theorem_l262_26232


namespace cos_inequality_range_l262_26220

theorem cos_inequality_range (x : Real) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.cos x ≤ 1 / 2 ↔ x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3)) := by
  sorry

end cos_inequality_range_l262_26220


namespace smallest_n_for_inequality_l262_26243

theorem smallest_n_for_inequality : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → 3^(3^(m+1)) ≥ 1007 → n ≤ m) ∧
  3^(3^(n+1)) ≥ 1007 := by
  sorry

end smallest_n_for_inequality_l262_26243


namespace least_repeating_digits_of_seven_thirteenths_l262_26215

theorem least_repeating_digits_of_seven_thirteenths : 
  (∀ n : ℕ, 0 < n → n < 6 → (10^n : ℤ) % 13 ≠ 1) ∧ (10^6 : ℤ) % 13 = 1 := by
  sorry

end least_repeating_digits_of_seven_thirteenths_l262_26215


namespace intersection_M_P_l262_26205

def M : Set ℝ := {x | x^2 = x}
def P : Set ℝ := {x | |x - 1| = 1}

theorem intersection_M_P : M ∩ P = {0} := by
  sorry

end intersection_M_P_l262_26205


namespace fraction_value_l262_26271

theorem fraction_value (a b c d : ℚ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 4 * d) :
  a * c / (b * d) = 16 := by
  sorry

end fraction_value_l262_26271


namespace parabola_coordinate_shift_l262_26298

/-- Given a parabola y = 3x² in a Cartesian coordinate system, 
    if the coordinate system is shifted 3 units right and 3 units up,
    then the equation of the parabola in the new coordinate system is y = 3(x+3)² - 3 -/
theorem parabola_coordinate_shift (x y : ℝ) : 
  (y = 3 * x^2) → 
  (∃ x' y', x' = x - 3 ∧ y' = y - 3 ∧ y' = 3 * (x' + 3)^2 - 3) :=
by sorry

end parabola_coordinate_shift_l262_26298


namespace problem_statement_l262_26259

theorem problem_statement (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*z/(y+z) + x*y/(z+x) = -10)
  (h2 : y*z/(x+y) + z*x/(y+z) + x*y/(z+x) = 15) :
  y/(x+y) + z/(y+z) + x/(z+x) = 14 := by
  sorry

end problem_statement_l262_26259


namespace simplify_expression_l262_26226

theorem simplify_expression : 
  5 * Real.sqrt 3 + (Real.sqrt 4 + 2 * Real.sqrt 3) = 7 * Real.sqrt 3 + 2 :=
by
  sorry

end simplify_expression_l262_26226


namespace pages_to_read_tonight_l262_26206

theorem pages_to_read_tonight (total_pages : ℕ) (first_night : ℕ) : 
  total_pages = 100 → 
  first_night = 15 → 
  (total_pages - (first_night + 2 * first_night + (2 * first_night + 5))) = 20 := by
  sorry

end pages_to_read_tonight_l262_26206


namespace natural_number_equations_l262_26265

theorem natural_number_equations :
  (∃! (x : ℕ), 2^(x-5) = 2) ∧
  (∃! (x : ℕ), 2^x = 512) ∧
  (∃! (x : ℕ), x^5 = 243) ∧
  (∃! (x : ℕ), x^4 = 625) :=
by
  sorry

end natural_number_equations_l262_26265


namespace tangent_problem_l262_26209

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β + Real.pi/4) = 1/4) :
  Real.tan (α - Real.pi/4) = 3/22 := by
  sorry

end tangent_problem_l262_26209


namespace complex_equation_solution_l262_26270

theorem complex_equation_solution (m A B : ℝ) :
  (((2 : ℂ) - m * I) / ((1 : ℂ) + 2 * I) = A + B * I) →
  A + B = 0 →
  m = -2/3 := by
sorry

end complex_equation_solution_l262_26270


namespace product_square_of_sum_and_diff_l262_26233

theorem product_square_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 23) 
  (diff_eq : x - y = 7) : 
  (x * y)^2 = 14400 := by
sorry

end product_square_of_sum_and_diff_l262_26233


namespace log_inequality_l262_26282

theorem log_inequality (c a b : ℝ) (h1 : 0 < c) (h2 : c < 1) (h3 : b > 1) (h4 : a > b) :
  Real.log c / Real.log a > Real.log c / Real.log b := by
  sorry

end log_inequality_l262_26282


namespace amalia_reading_time_l262_26223

/-- Represents the time in minutes it takes Amalia to read a given number of pages -/
def reading_time (pages : ℕ) : ℚ :=
  (pages : ℚ) * 2 / 4

/-- Theorem stating that it takes Amalia 9 minutes to read 18 pages -/
theorem amalia_reading_time :
  reading_time 18 = 9 := by
  sorry

end amalia_reading_time_l262_26223


namespace compound_molar_mass_l262_26253

/-- Given a compound where 5 moles weigh 1170 grams, prove its molar mass is 234 grams/mole. -/
theorem compound_molar_mass (mass : ℝ) (moles : ℝ) (h1 : mass = 1170) (h2 : moles = 5) :
  mass / moles = 234 := by
sorry

end compound_molar_mass_l262_26253


namespace floor_ceil_product_l262_26247

theorem floor_ceil_product : ⌊(0.998 : ℝ)⌋ * ⌈(1.999 : ℝ)⌉ = 0 := by sorry

end floor_ceil_product_l262_26247


namespace calculate_expression_l262_26257

theorem calculate_expression : (2 - 5 * (-1/2)^2) / (-1/4) = -3 := by
  sorry

end calculate_expression_l262_26257


namespace expand_and_simplify_l262_26294

theorem expand_and_simplify (x : ℝ) : (2 * x - 3) * (4 * x + 5) = 8 * x^2 - 2 * x - 15 := by
  sorry

end expand_and_simplify_l262_26294


namespace sqrt_50_between_consecutive_integers_l262_26263

theorem sqrt_50_between_consecutive_integers :
  ∃ (n : ℕ), (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 := by
  sorry

end sqrt_50_between_consecutive_integers_l262_26263


namespace product_mod_seventeen_l262_26264

theorem product_mod_seventeen : (2021 * 2023 * 2025 * 2027 * 2029) % 17 = 13 := by
  sorry

end product_mod_seventeen_l262_26264


namespace asterisk_replacement_l262_26225

/-- The expression after substituting 2x for * and expanding -/
def expanded_expression (x : ℝ) : ℝ := x^6 + x^4 + 4*x^2 + 4

/-- The number of terms in the expanded expression -/
def num_terms (x : ℝ) : ℕ := 4

theorem asterisk_replacement :
  ∀ x : ℝ, (x^3 - 2)^2 + (x^2 + 2*x)^2 = expanded_expression x ∧
           num_terms x = 4 :=
by sorry

end asterisk_replacement_l262_26225


namespace symmetric_line_equation_l262_26274

/-- Given a line with equation x + y + 1 = 0 and a point of symmetry (1, 2),
    the symmetric line has the equation x + y - 7 = 0 -/
theorem symmetric_line_equation :
  let original_line := {(x, y) : ℝ × ℝ | x + y + 1 = 0}
  let symmetry_point := (1, 2)
  let symmetric_line := {(x, y) : ℝ × ℝ | x + y - 7 = 0}
  ∀ (p : ℝ × ℝ), p ∈ symmetric_line ↔
    (2 * symmetry_point.1 - p.1, 2 * symmetry_point.2 - p.2) ∈ original_line :=
by sorry

end symmetric_line_equation_l262_26274


namespace even_periodic_function_range_l262_26203

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem even_periodic_function_range (f : ℝ → ℝ) (a : ℝ) :
  IsEven f →
  HasPeriod f 3 →
  f 1 < 1 →
  f 5 = (2*a - 3) / (a + 1) →
  -1 < a ∧ a < 4 := by
  sorry

end even_periodic_function_range_l262_26203


namespace algebraic_expression_correct_l262_26277

/-- The algebraic expression for the number that is 2 less than three times the cube of a and b -/
def algebraic_expression (a b : ℝ) : ℝ := 3 * (a^3 + b^3) - 2

/-- Theorem stating that the algebraic expression is correct -/
theorem algebraic_expression_correct (a b : ℝ) :
  algebraic_expression a b = 3 * (a^3 + b^3) - 2 := by sorry

end algebraic_expression_correct_l262_26277


namespace parabola_equation_proof_l262_26217

/-- A parabola is defined by three points: A(4,0), C(0,-4), and B(-1,0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of the parabola is y = ax^2 + bx + c -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The parabola passes through point A(4,0) -/
def passes_through_A (p : Parabola) : Prop :=
  parabola_equation p 4 = 0

/-- The parabola passes through point C(0,-4) -/
def passes_through_C (p : Parabola) : Prop :=
  parabola_equation p 0 = -4

/-- The parabola passes through point B(-1,0) -/
def passes_through_B (p : Parabola) : Prop :=
  parabola_equation p (-1) = 0

/-- The theorem states that the parabola passing through A, C, and B
    has the equation y = x^2 - 3x - 4 -/
theorem parabola_equation_proof :
  ∃ p : Parabola,
    passes_through_A p ∧
    passes_through_C p ∧
    passes_through_B p ∧
    p.a = 1 ∧ p.b = -3 ∧ p.c = -4 :=
  sorry

end parabola_equation_proof_l262_26217


namespace triangle_area_with_arithmetic_sides_l262_26222

/-- Given a triangle ABC with one angle of 120° and sides in arithmetic progression with common difference 2, its area is 15√3/4 -/
theorem triangle_area_with_arithmetic_sides : ∀ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c →
  ∃ (θ : ℝ), θ = 2 * π / 3 →
  ∃ (d : ℝ), d = 2 →
  b = a + d ∧ c = b + d →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos θ →
  (1/2) * a * b * Real.sin θ = 15 * Real.sqrt 3 / 4 := by
  sorry


end triangle_area_with_arithmetic_sides_l262_26222


namespace gails_wallet_l262_26224

/-- Represents the contents of Gail's wallet -/
structure Wallet where
  total : ℕ
  five_dollar_bills : ℕ
  twenty_dollar_bills : ℕ
  ten_dollar_bills : ℕ

/-- Calculates the total amount in the wallet based on the bill counts -/
def wallet_total (w : Wallet) : ℕ :=
  5 * w.five_dollar_bills + 20 * w.twenty_dollar_bills + 10 * w.ten_dollar_bills

/-- Theorem stating that given the conditions, Gail has 2 ten-dollar bills -/
theorem gails_wallet :
  ∃ (w : Wallet),
    w.total = 100 ∧
    w.five_dollar_bills = 4 ∧
    w.twenty_dollar_bills = 3 ∧
    wallet_total w = w.total ∧
    w.ten_dollar_bills = 2 := by
  sorry


end gails_wallet_l262_26224


namespace variable_prime_count_l262_26269

/-- The number of primes between n^2 + 1 and n^2 + n is not constant for n > 1 -/
theorem variable_prime_count (n : ℕ) (h : n > 1) :
  ∃ m : ℕ, m > n ∧ 
  (Finset.filter (Nat.Prime) (Finset.range (n^2 + n - (n^2 + 2) + 1))).card ≠
  (Finset.filter (Nat.Prime) (Finset.range (m^2 + m - (m^2 + 2) + 1))).card :=
by sorry

end variable_prime_count_l262_26269


namespace shaded_probability_is_one_third_l262_26280

/-- Represents a triangle in the diagram -/
structure Triangle where
  shaded : Bool

/-- Represents the diagram with triangles -/
structure Diagram where
  triangles : List Triangle
  shaded_count : Nat
  h_more_than_five : triangles.length > 5
  h_shaded_count : shaded_count = (triangles.filter (·.shaded)).length

/-- The probability of selecting a shaded triangle -/
def shaded_probability (d : Diagram) : ℚ :=
  d.shaded_count / d.triangles.length

/-- Theorem stating the probability of selecting a shaded triangle is 1/3 -/
theorem shaded_probability_is_one_third (d : Diagram) :
  d.shaded_count = 3 ∧ d.triangles.length = 9 →
  shaded_probability d = 1/3 := by
  sorry

end shaded_probability_is_one_third_l262_26280


namespace inequality_proof_l262_26228

theorem inequality_proof (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt ((w^2 + x^2 + y^2 + z^2) / 4) ≥ ((wxy + wxz + wyz + xyz) / 4)^(1/3) :=
by
  sorry

where
  wxy := w * x * y
  wxz := w * x * z
  wyz := w * y * z
  xyz := x * y * z

end inequality_proof_l262_26228


namespace cone_sphere_volume_l262_26214

/-- Given a cone with lateral surface forming a semicircle of radius 2√3 when unrolled,
    and with vertex and base circle on the surface of a sphere O,
    prove that the volume of sphere O is 32π/3 -/
theorem cone_sphere_volume (l : ℝ) (r : ℝ) (h : ℝ) (R : ℝ) :
  l = 2 * Real.sqrt 3 →
  r = l / 2 →
  h^2 = l^2 - r^2 →
  2 * R = l^2 / h →
  (4 / 3) * Real.pi * R^3 = (32 / 3) * Real.pi :=
sorry

end cone_sphere_volume_l262_26214


namespace probability_one_red_two_blue_l262_26221

/-- The probability of selecting one red marble and two blue marbles from a bag -/
theorem probability_one_red_two_blue (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ) 
  (h1 : total_marbles = red_marbles + blue_marbles)
  (h2 : red_marbles = 10)
  (h3 : blue_marbles = 6) : 
  (red_marbles * blue_marbles * (blue_marbles - 1) + 
   blue_marbles * red_marbles * (blue_marbles - 1) + 
   blue_marbles * (blue_marbles - 1) * red_marbles) / 
  (total_marbles * (total_marbles - 1) * (total_marbles - 2)) = 15 / 56 := by
  sorry

end probability_one_red_two_blue_l262_26221


namespace smallest_consecutive_sum_l262_26201

theorem smallest_consecutive_sum (n : ℕ) (a : ℕ) (h1 : n > 1) 
  (h2 : n * a + n * (n - 1) / 2 = 2016) : 
  ∃ (m : ℕ), m ≥ 1 ∧ m * a + m * (m - 1) / 2 = 2016 ∧ m > 1 → a ≥ 1 :=
by sorry

end smallest_consecutive_sum_l262_26201


namespace cooking_time_for_remaining_potatoes_l262_26216

/-- Given a chef cooking potatoes with the following conditions:
  - The total number of potatoes to cook is 16
  - The number of potatoes already cooked is 7
  - Each potato takes 5 minutes to cook
  Prove that the time required to cook the remaining potatoes is 45 minutes. -/
theorem cooking_time_for_remaining_potatoes :
  ∀ (total_potatoes cooked_potatoes cooking_time_per_potato : ℕ),
    total_potatoes = 16 →
    cooked_potatoes = 7 →
    cooking_time_per_potato = 5 →
    (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 45 :=
by sorry

end cooking_time_for_remaining_potatoes_l262_26216


namespace shell_difference_l262_26241

theorem shell_difference (perfect_total : ℕ) (broken_total : ℕ)
  (broken_spiral_percent : ℚ) (broken_clam_percent : ℚ)
  (perfect_spiral_percent : ℚ) (perfect_clam_percent : ℚ)
  (h1 : perfect_total = 30)
  (h2 : broken_total = 80)
  (h3 : broken_spiral_percent = 35 / 100)
  (h4 : broken_clam_percent = 40 / 100)
  (h5 : perfect_spiral_percent = 25 / 100)
  (h6 : perfect_clam_percent = 50 / 100) :
  ⌊broken_total * broken_spiral_percent⌋ - ⌊perfect_total * perfect_spiral_percent⌋ = 21 :=
by sorry

end shell_difference_l262_26241


namespace m_geq_two_l262_26297

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Given condition: f'(x) < x for all x ∈ ℝ
axiom f'_less_than_x : ∀ x, f' x < x

-- Define m as a real number
variable (m : ℝ)

-- Given inequality involving f
axiom f_inequality : f (4 - m) - f m ≥ 8 - 4 * m

-- Theorem to prove
theorem m_geq_two : m ≥ 2 := by sorry

end m_geq_two_l262_26297


namespace cubic_roots_sum_l262_26289

theorem cubic_roots_sum (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (∀ w : ℝ, w^3 - 9*w^2 + a*w - b = 0 ↔ (w = x ∨ w = y ∨ w = z))) →
  a + b = 38 := by
sorry

end cubic_roots_sum_l262_26289


namespace triangle_side_and_area_l262_26219

theorem triangle_side_and_area 
  (a b c : ℝ) 
  (A : ℝ) 
  (h1 : a = Real.sqrt 7)
  (h2 : c = 3)
  (h3 : A = π / 3) :
  (b = 1 ∨ b = 2) ∧
  ((b = 1 → (1/2 * b * c * Real.sin A = (3 * Real.sqrt 3) / 4)) ∧
   (b = 2 → (1/2 * b * c * Real.sin A = (3 * Real.sqrt 3) / 2))) :=
by sorry

end triangle_side_and_area_l262_26219


namespace factorial_340_trailing_zeros_l262_26268

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 340! ends with 83 zeros -/
theorem factorial_340_trailing_zeros :
  trailingZeros 340 = 83 := by
  sorry

end factorial_340_trailing_zeros_l262_26268


namespace parametric_to_ordinary_equation_l262_26261

theorem parametric_to_ordinary_equation :
  ∀ (θ : ℝ) (x y : ℝ),
    x = Real.cos θ ^ 2 →
    y = 2 * Real.sin θ ^ 2 →
    2 * x + y - 2 = 0 ∧ x ∈ Set.Icc 0 1 := by
  sorry

end parametric_to_ordinary_equation_l262_26261


namespace evaluate_expression_l262_26239

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x = 789 := by
  sorry

end evaluate_expression_l262_26239


namespace equation_one_solutions_equation_two_solutions_l262_26204

-- Equation 1
theorem equation_one_solutions (x : ℝ) : 
  x * (x - 2) = x - 2 ↔ x = 1 ∨ x = 2 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) : 
  x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by sorry

end equation_one_solutions_equation_two_solutions_l262_26204


namespace intersection_of_three_lines_l262_26292

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y : ℝ) :
  y = 4 * x + 5 ∧ 
  y = -3 * x + 10 ∧ 
  y = 2 * x + k →
  k = 45 / 7 := by
  sorry

end intersection_of_three_lines_l262_26292


namespace tangent_line_problem_l262_26281

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2
def g (b : ℝ) (x : ℝ) : ℝ := Real.sin x + b * x

def is_tangent_at (l : ℝ → ℝ) (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  l x₀ = f x₀ ∧ (deriv l) x₀ = (deriv f) x₀

theorem tangent_line_problem (a b : ℝ) (l : ℝ → ℝ) :
  is_tangent_at l (f a) 0 →
  is_tangent_at l (g b) (Real.pi / 2) →
  (a = 1 ∧ b = 1) ∧
  (∀ x, l x = x + 1) ∧
  (∀ x, Real.exp x + x^2 - x - Real.sin x > 0) :=
sorry

end

end tangent_line_problem_l262_26281


namespace batsman_average_l262_26287

/-- Calculates the overall average runs per match for a batsman -/
def overall_average (matches1 : ℕ) (avg1 : ℚ) (matches2 : ℕ) (avg2 : ℚ) : ℚ :=
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2)

/-- The batsman's overall average is approximately 21.43 -/
theorem batsman_average : 
  let matches1 := 15
  let avg1 := 30
  let matches2 := 20
  let avg2 := 15
  abs (overall_average matches1 avg1 matches2 avg2 - 21.43) < 0.01 := by
  sorry

end batsman_average_l262_26287


namespace largest_lcm_with_15_l262_26249

theorem largest_lcm_with_15 :
  let lcm_list := [lcm 15 3, lcm 15 5, lcm 15 6, lcm 15 9, lcm 15 10, lcm 15 12]
  List.maximum lcm_list = some 60 := by
  sorry

end largest_lcm_with_15_l262_26249
