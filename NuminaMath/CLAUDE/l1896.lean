import Mathlib

namespace NUMINAMATH_CALUDE_remainder_proof_l1896_189642

theorem remainder_proof : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l1896_189642


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_third_l1896_189667

theorem arctan_sum_equals_pi_third (n : ℕ+) : 
  Real.arctan (1/7) + Real.arctan (1/8) + Real.arctan (1/9) + Real.arctan (1/n) = π/3 → n = 84 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_third_l1896_189667


namespace NUMINAMATH_CALUDE_brick_height_is_7_point_5_cm_l1896_189646

/-- Proves that the height of a brick is 7.5 cm given the dimensions of the wall,
    the number of bricks, and the length and width of a single brick. -/
theorem brick_height_is_7_point_5_cm
  (brick_length : ℝ)
  (brick_width : ℝ)
  (wall_length : ℝ)
  (wall_width : ℝ)
  (wall_height : ℝ)
  (num_bricks : ℕ)
  (h_brick_length : brick_length = 20)
  (h_brick_width : brick_width = 10)
  (h_wall_length : wall_length = 2600)
  (h_wall_width : wall_width = 200)
  (h_wall_height : wall_height = 75)
  (h_num_bricks : num_bricks = 26000) :
  ∃ (brick_height : ℝ), brick_height = 7.5 ∧
    num_bricks * brick_length * brick_width * brick_height =
    wall_length * wall_width * wall_height :=
by sorry

end NUMINAMATH_CALUDE_brick_height_is_7_point_5_cm_l1896_189646


namespace NUMINAMATH_CALUDE_smallest_truck_shipments_l1896_189625

theorem smallest_truck_shipments (B : ℕ) : 
  B ≥ 120 → 
  B % 5 = 0 → 
  ∃ (T : ℕ), T ≠ 5 ∧ T > 1 ∧ B % T = 0 ∧ 
  ∀ (S : ℕ), S ≠ 5 → S > 1 → B % S = 0 → T ≤ S :=
by sorry

end NUMINAMATH_CALUDE_smallest_truck_shipments_l1896_189625


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1896_189674

def num_arrangements (n_pushkin n_tarle : ℕ) : ℕ :=
  3 * (Nat.factorial 2) * (Nat.factorial 4)

theorem book_arrangement_count :
  num_arrangements 2 4 = 144 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1896_189674


namespace NUMINAMATH_CALUDE_wage_increase_hours_reduction_l1896_189601

/-- Proves that when an employee's hourly wage increases by 10% and they want to maintain
    the same total weekly income, the percent reduction in hours worked is (1 - 1/1.10) * 100% -/
theorem wage_increase_hours_reduction (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let new_wage := 1.1 * w
  let new_hours := h * w / new_wage
  let percent_reduction := (h - new_hours) / h * 100
  percent_reduction = (1 - 1 / 1.1) * 100 := by
  sorry


end NUMINAMATH_CALUDE_wage_increase_hours_reduction_l1896_189601


namespace NUMINAMATH_CALUDE_ray_AB_not_equal_ray_BA_l1896_189643

-- Define a point type
def Point := ℝ × ℝ

-- Define a ray type
structure Ray where
  start : Point
  direction : Point

-- Define an equality relation for rays
def ray_eq (r1 r2 : Ray) : Prop :=
  r1.start = r2.start ∧ r1.direction = r2.direction

-- Theorem statement
theorem ray_AB_not_equal_ray_BA (A B : Point) (h : A ≠ B) :
  ¬(ray_eq (Ray.mk A B) (Ray.mk B A)) := by
  sorry

end NUMINAMATH_CALUDE_ray_AB_not_equal_ray_BA_l1896_189643


namespace NUMINAMATH_CALUDE_factor_expression_l1896_189614

theorem factor_expression (t : ℝ) : 4 * t^2 - 144 + 8 = 4 * (t^2 - 34) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1896_189614


namespace NUMINAMATH_CALUDE_equal_cost_guests_correct_l1896_189658

/-- The number of guests for which the costs of renting Caesar's and Venus Hall are equal -/
def equal_cost_guests : ℕ :=
  let caesar_rental : ℕ := 800
  let caesar_per_meal : ℕ := 30
  let venus_rental : ℕ := 500
  let venus_per_meal : ℕ := 35
  60

theorem equal_cost_guests_correct :
  let caesar_rental : ℕ := 800
  let caesar_per_meal : ℕ := 30
  let venus_rental : ℕ := 500
  let venus_per_meal : ℕ := 35
  caesar_rental + caesar_per_meal * equal_cost_guests = venus_rental + venus_per_meal * equal_cost_guests :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_guests_correct_l1896_189658


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_16820_l1896_189635

/-- Calculates the cost of whitewashing a room with given dimensions and openings. -/
def whitewashing_cost (room_length room_width room_height : ℝ)
                      (door1_length door1_width : ℝ)
                      (door2_length door2_width : ℝ)
                      (window1_length window1_width : ℝ)
                      (window2_length window2_width : ℝ)
                      (window3_length window3_width : ℝ)
                      (window4_length window4_width : ℝ)
                      (window5_length window5_width : ℝ)
                      (cost_per_sqft : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let openings_area := door1_length * door1_width + door2_length * door2_width +
                       window1_length * window1_width + window2_length * window2_width +
                       window3_length * window3_width + window4_length * window4_width +
                       window5_length * window5_width
  let whitewash_area := wall_area - openings_area
  whitewash_area * cost_per_sqft

/-- The cost of whitewashing the room with given dimensions and openings is Rs. 16820. -/
theorem whitewashing_cost_is_16820 :
  whitewashing_cost 40 20 15 7 4 5 3 5 4 4 3 3 3 4 2.5 6 4 10 = 16820 := by
  sorry


end NUMINAMATH_CALUDE_whitewashing_cost_is_16820_l1896_189635


namespace NUMINAMATH_CALUDE_annika_age_l1896_189615

theorem annika_age (hans_age : ℕ) (annika_age : ℕ) : 
  hans_age = 8 →
  annika_age + 4 = 3 * (hans_age + 4) →
  annika_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_annika_age_l1896_189615


namespace NUMINAMATH_CALUDE_rectangle_length_equal_square_side_l1896_189630

/-- The length of a rectangle with width 4 cm and area equal to a square with side length 4 cm -/
theorem rectangle_length_equal_square_side : ∀ (length : ℝ), 
  (4 : ℝ) * length = (4 : ℝ) * (4 : ℝ) → length = (4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_equal_square_side_l1896_189630


namespace NUMINAMATH_CALUDE_nine_bulb_configurations_l1896_189634

def f : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | n + 4 => f (n + 3) + f (n + 2) + f (n + 1) + f n

def circularConfigurations (n : ℕ) : ℕ :=
  f n - 3 * f 3 - 2 * f 2 - f 1

theorem nine_bulb_configurations :
  circularConfigurations 9 = 367 := by sorry

end NUMINAMATH_CALUDE_nine_bulb_configurations_l1896_189634


namespace NUMINAMATH_CALUDE_boat_race_spacing_l1896_189653

theorem boat_race_spacing (river_width : ℝ) (num_boats : ℕ) (boat_width : ℝ)
  (hw : river_width = 42)
  (hn : num_boats = 8)
  (hb : boat_width = 3) :
  (river_width - num_boats * boat_width) / (num_boats + 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_boat_race_spacing_l1896_189653


namespace NUMINAMATH_CALUDE_square_root_eleven_minus_one_squared_plus_two_times_plus_one_equals_eleven_l1896_189687

theorem square_root_eleven_minus_one_squared_plus_two_times_plus_one_equals_eleven :
  ∀ a : ℝ, a = Real.sqrt 11 - 1 → a^2 + 2*a + 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_root_eleven_minus_one_squared_plus_two_times_plus_one_equals_eleven_l1896_189687


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1896_189632

/-- The eccentricity of an ellipse with equation x²/m² + y²/9 = 1 (m > 0) and one focus at (4, 0) is 4/5 -/
theorem ellipse_eccentricity (m : ℝ) (h1 : m > 0) : 
  let ellipse := { (x, y) : ℝ × ℝ | x^2 / m^2 + y^2 / 9 = 1 }
  let focus : ℝ × ℝ := (4, 0)
  focus ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 ∧ p ∈ ellipse } →
  (∃ (a b c : ℝ), a > b ∧ b > 0 ∧ c > 0 ∧ a^2 = m^2 ∧ b^2 = 9 ∧ c^2 = a^2 - b^2 ∧ c / a = 4 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1896_189632


namespace NUMINAMATH_CALUDE_original_price_calculation_l1896_189616

theorem original_price_calculation (total_sale : ℝ) (profit_rate : ℝ) (loss_rate : ℝ)
  (h1 : total_sale = 660)
  (h2 : profit_rate = 0.1)
  (h3 : loss_rate = 0.1) :
  ∃ (original_price : ℝ),
    original_price = total_sale / (1 + profit_rate) + total_sale / (1 - loss_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1896_189616


namespace NUMINAMATH_CALUDE_calculation_result_l1896_189633

theorem calculation_result : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l1896_189633


namespace NUMINAMATH_CALUDE_eighty_percent_of_forty_l1896_189676

theorem eighty_percent_of_forty (x : ℚ) : x * 20 + 16 = 32 → x = 4/5 := by sorry

end NUMINAMATH_CALUDE_eighty_percent_of_forty_l1896_189676


namespace NUMINAMATH_CALUDE_tatiana_age_l1896_189611

/-- Calculates the total full years given an age in years, months, weeks, days, and hours -/
def calculate_full_years (years months weeks days hours : ℕ) : ℕ :=
  let months_to_years := months / 12
  let weeks_to_years := weeks / 52
  let days_to_years := days / 365
  let hours_to_years := hours / (24 * 365)
  years + months_to_years + weeks_to_years + days_to_years + hours_to_years

/-- Theorem stating that the age of 72 years, 72 months, 72 weeks, 72 days, and 72 hours is equivalent to 79 full years -/
theorem tatiana_age : calculate_full_years 72 72 72 72 72 = 79 := by
  sorry

end NUMINAMATH_CALUDE_tatiana_age_l1896_189611


namespace NUMINAMATH_CALUDE_division_properties_7529_l1896_189689

theorem division_properties_7529 : 
  (7529 % 9 = 5) ∧ ¬(11 ∣ 7529) := by
  sorry

end NUMINAMATH_CALUDE_division_properties_7529_l1896_189689


namespace NUMINAMATH_CALUDE_cubic_not_decreasing_param_range_l1896_189621

/-- Given a cubic function that is not strictly decreasing, prove the range of its parameter. -/
theorem cubic_not_decreasing_param_range (b : ℝ) : 
  (∃ x y : ℝ, x < y ∧ (-x^3 + b*x^2 - (2*b + 3)*x + 2 - b) ≤ (-y^3 + b*y^2 - (2*b + 3)*y + 2 - b)) →
  (b < -1 ∨ b > 3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_not_decreasing_param_range_l1896_189621


namespace NUMINAMATH_CALUDE_soap_decrease_l1896_189660

theorem soap_decrease (x : ℝ) (h : x > 0) : x * (0.8 ^ 2) ≤ (2/3) * x :=
sorry

end NUMINAMATH_CALUDE_soap_decrease_l1896_189660


namespace NUMINAMATH_CALUDE_min_value_expression_l1896_189603

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) ≥ 24 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 6 ∧ (9 / a₀ + 16 / b₀ + 25 / c₀) = 24 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1896_189603


namespace NUMINAMATH_CALUDE_prime_between_30_and_40_with_remainder_7_mod_12_l1896_189651

theorem prime_between_30_and_40_with_remainder_7_mod_12 (n : ℕ) : 
  Prime n → 
  30 < n → 
  n < 40 → 
  n % 12 = 7 → 
  n = 31 := by
sorry

end NUMINAMATH_CALUDE_prime_between_30_and_40_with_remainder_7_mod_12_l1896_189651


namespace NUMINAMATH_CALUDE_base13_addition_proof_l1896_189648

/-- Represents a digit in base 13 -/
inductive Base13Digit
  | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Represents a number in base 13 -/
def Base13Number := List Base13Digit

/-- Addition of two Base13Numbers -/
def add_base13 : Base13Number → Base13Number → Base13Number
  | _, _ => sorry  -- Implementation details omitted

/-- Conversion of a natural number to Base13Number -/
def nat_to_base13 : Nat → Base13Number
  | _ => sorry  -- Implementation details omitted

theorem base13_addition_proof :
  add_base13 (nat_to_base13 528) (nat_to_base13 274) =
  [Base13Digit.D7, Base13Digit.A, Base13Digit.C] :=
by sorry

end NUMINAMATH_CALUDE_base13_addition_proof_l1896_189648


namespace NUMINAMATH_CALUDE_g_range_l1896_189697

noncomputable def g (x : ℝ) : ℝ := (Real.arcsin x)^3 - (Real.arccos x)^3

theorem g_range :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 →
  (Real.arccos x + Real.arcsin x = π / 2) →
  ∃ y ∈ Set.Icc (-((7 * π^3) / 16)) ((π^3) / 16), g x = y :=
by sorry

end NUMINAMATH_CALUDE_g_range_l1896_189697


namespace NUMINAMATH_CALUDE_factor_theorem_application_l1896_189619

theorem factor_theorem_application (c : ℚ) : 
  (∀ x : ℚ, (x + 5) ∣ (2*c*x^3 + 14*x^2 - 6*c*x + 25)) → c = 75/44 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l1896_189619


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1896_189604

theorem polynomial_expansion :
  ∀ z : ℂ, (3 * z^3 + 2 * z^2 - 4 * z + 1) * (2 * z^4 - 3 * z^2 + z - 5) =
  6 * z^7 + 4 * z^6 - 4 * z^5 - 9 * z^3 + 7 * z^2 + z - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1896_189604


namespace NUMINAMATH_CALUDE_base_8_243_equals_163_l1896_189691

def base_8_to_10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

theorem base_8_243_equals_163 :
  base_8_to_10 2 4 3 = 163 := by
  sorry

end NUMINAMATH_CALUDE_base_8_243_equals_163_l1896_189691


namespace NUMINAMATH_CALUDE_ingrids_tax_rate_l1896_189677

/-- Calculates the tax rate of the second person given the tax rate of the first person,
    both incomes, and their combined tax rate. -/
def calculate_second_tax_rate (first_tax_rate first_income second_income combined_tax_rate : ℚ) : ℚ :=
  let combined_income := first_income + second_income
  let total_tax := combined_tax_rate * combined_income
  let first_tax := first_tax_rate * first_income
  let second_tax := total_tax - first_tax
  second_tax / second_income

/-- Proves that given the specified conditions, Ingrid's tax rate is 40.00% -/
theorem ingrids_tax_rate :
  let john_tax_rate : ℚ := 30 / 100
  let john_income : ℚ := 56000
  let ingrid_income : ℚ := 74000
  let combined_tax_rate : ℚ := 3569 / 10000
  calculate_second_tax_rate john_tax_rate john_income ingrid_income combined_tax_rate = 40 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ingrids_tax_rate_l1896_189677


namespace NUMINAMATH_CALUDE_min_value_theorem_l1896_189637

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 3

-- Define a point on the line
def point_on_line (a b : ℝ) : Prop := line_l a b

-- Theorem statement
theorem min_value_theorem (a b : ℝ) (h : point_on_line a b) :
  3^a + 3^b ≥ 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1896_189637


namespace NUMINAMATH_CALUDE_necklace_ratio_l1896_189655

/-- The number of necklaces Haley, Jason, and Josh have. -/
structure Necklaces where
  haley : ℕ
  jason : ℕ
  josh : ℕ

/-- The conditions given in the problem. -/
def problem_conditions (n : Necklaces) : Prop :=
  n.haley = n.jason + 5 ∧
  n.haley = 25 ∧
  n.haley = n.josh + 15

/-- The theorem stating that under the given conditions, 
    the ratio of Josh's necklaces to Jason's necklaces is 1:2. -/
theorem necklace_ratio (n : Necklaces) 
  (h : problem_conditions n) : n.josh * 2 = n.jason := by
  sorry


end NUMINAMATH_CALUDE_necklace_ratio_l1896_189655


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1896_189659

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1, focal length 10, and point P(2, 1) on its asymptote, prove that the equation of C is x²/20 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 + b^2 ∧ 2*c = 10) →
  (∃ x y : ℝ, x = 2 ∧ y = 1 ∧ y = (b/a) * x) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/20 - y^2/5 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1896_189659


namespace NUMINAMATH_CALUDE_motorist_gas_plan_l1896_189657

/-- The number of gallons a motorist initially planned to buy given certain conditions -/
theorem motorist_gas_plan (actual_price expected_price_difference affordable_gallons : ℚ) :
  actual_price = 150 ∧ 
  expected_price_difference = 30 ∧ 
  affordable_gallons = 10 →
  (actual_price * affordable_gallons) / (actual_price - expected_price_difference) = 25/2 :=
by sorry

end NUMINAMATH_CALUDE_motorist_gas_plan_l1896_189657


namespace NUMINAMATH_CALUDE_runners_meet_time_l1896_189650

def carla_lap_time : ℕ := 5
def jose_lap_time : ℕ := 8
def mary_lap_time : ℕ := 10

theorem runners_meet_time :
  Nat.lcm (Nat.lcm carla_lap_time jose_lap_time) mary_lap_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_time_l1896_189650


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l1896_189692

/-- The number of distinct arrangements of books on a shelf. -/
def distinct_arrangements (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem: The number of distinct arrangements of 7 books with 3 identical copies is 840. -/
theorem book_arrangement_proof :
  distinct_arrangements 7 3 = 840 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l1896_189692


namespace NUMINAMATH_CALUDE_mrs_heine_biscuits_l1896_189623

/-- Calculates the total number of biscuits needed for Mrs. Heine's pets -/
def total_biscuits (num_dogs : ℕ) (num_cats : ℕ) (num_birds : ℕ) 
                   (biscuits_per_dog : ℕ) (biscuits_per_cat : ℕ) (biscuits_per_bird : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog + num_cats * biscuits_per_cat + num_birds * biscuits_per_bird

/-- Theorem stating that Mrs. Heine needs to buy 11 biscuits in total -/
theorem mrs_heine_biscuits : 
  total_biscuits 2 1 3 3 2 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mrs_heine_biscuits_l1896_189623


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1896_189622

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1896_189622


namespace NUMINAMATH_CALUDE_complex_fraction_equals_221_l1896_189682

theorem complex_fraction_equals_221 : 
  (((12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324)) : ℚ) /
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324)) = 221 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_221_l1896_189682


namespace NUMINAMATH_CALUDE_joyce_land_theorem_l1896_189656

/-- Calculates the suitable land for growing vegetables given the previous property size,
    the factor by which the new property is larger, and the size of the pond. -/
def suitable_land (previous_property : ℝ) (size_factor : ℝ) (pond_size : ℝ) : ℝ :=
  previous_property * size_factor - pond_size

/-- Theorem stating that given a previous property of 2 acres, a new property 8 times larger,
    and a 3-acre pond, the land suitable for growing vegetables is 13 acres. -/
theorem joyce_land_theorem :
  suitable_land 2 8 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_joyce_land_theorem_l1896_189656


namespace NUMINAMATH_CALUDE_derivative_at_negative_one_l1896_189663

/-- Given a function f(x) = x², prove that the derivative of f at x = -1 is -2. -/
theorem derivative_at_negative_one (f : ℝ → ℝ) (h : ∀ x, f x = x^2) :
  deriv f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_one_l1896_189663


namespace NUMINAMATH_CALUDE_n_pointed_star_angle_sum_l1896_189626

/-- Represents an n-pointed star created from a convex n-gon. -/
structure NPointedStar where
  n : ℕ
  n_ge_7 : n ≥ 7

/-- The sum of interior angles at the n intersection points of an n-pointed star. -/
def interior_angle_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem stating that the sum of interior angles at the n intersection points
    of an n-pointed star is 180°(n-2). -/
theorem n_pointed_star_angle_sum (star : NPointedStar) :
  interior_angle_sum star = 180 * (star.n - 2) := by
  sorry

end NUMINAMATH_CALUDE_n_pointed_star_angle_sum_l1896_189626


namespace NUMINAMATH_CALUDE_fold_line_length_squared_l1896_189629

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  let d_AB := ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2)^(1/2)
  let d_BC := ((t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2)^(1/2)
  let d_CA := ((t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2)^(1/2)
  d_AB = d_BC ∧ d_BC = d_CA

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)^(1/2)

/-- Theorem: The square of the length of the fold line in the given triangle problem -/
theorem fold_line_length_squared (t : Triangle) (P : Point) :
  isEquilateral t →
  distance t.A t.B = 15 →
  distance t.B P = 11 →
  P.x = t.B.x + 11 * (t.C.x - t.B.x) / 15 →
  P.y = t.B.y + 11 * (t.C.y - t.B.y) / 15 →
  ∃ Q : Point,
    Q.x = t.A.x + (P.x - t.A.x) / 2 ∧
    Q.y = t.A.y + (P.y - t.A.y) / 2 ∧
    (distance Q P)^2 = 1043281 / 31109 :=
sorry

end NUMINAMATH_CALUDE_fold_line_length_squared_l1896_189629


namespace NUMINAMATH_CALUDE_sequence_monotonicity_l1896_189690

theorem sequence_monotonicity (k : ℝ) : 
  (∀ n : ℕ, (n + 1)^2 + k*(n + 1) + 2 > n^2 + k*n + 2) → k > -3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_monotonicity_l1896_189690


namespace NUMINAMATH_CALUDE_min_sum_squares_l1896_189620

theorem min_sum_squares (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 1) :
  ∃ (m : ℝ), ∀ (x y : ℝ), 0 < x → 0 < y → 2 * x + y = 1 → x^2 + y^2 ≥ m ∧ (∃ (u v : ℝ), 0 < u ∧ 0 < v ∧ 2 * u + v = 1 ∧ u^2 + v^2 = m) ∧ m = 1/5 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1896_189620


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1896_189684

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and focal distance c,
    prove that if the point symmetric to the focus with respect to y = (b/c)x lies on the ellipse,
    then the eccentricity is √2/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  let e := c / a
  let ellipse := fun (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let focus := (c, 0)
  let symmetry_line := fun (x : ℝ) ↦ (b / c) * x
  let Q := (
    let m := (c^3 - c*b^2) / a^2
    let n := 2*b*c^2 / a^2
    (m, n)
  )
  (ellipse Q.1 Q.2) → e = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1896_189684


namespace NUMINAMATH_CALUDE_chord_length_is_two_l1896_189686

/-- The chord length intercepted by y = 1 - x on x² + y² + 2y - 2 = 0 is 2 -/
theorem chord_length_is_two (x y : ℝ) : 
  (x^2 + y^2 + 2*y - 2 = 0) → 
  (y = 1 - x) → 
  ∃ (a b : ℝ), (a^2 + b^2 = 1) ∧ 
               ((x - a)^2 + (y - b)^2 = 2^2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_is_two_l1896_189686


namespace NUMINAMATH_CALUDE_jack_marbles_remaining_l1896_189600

/-- 
Given that Jack starts with a certain number of marbles and shares some with Rebecca,
this theorem proves how many marbles Jack ends up with.
-/
theorem jack_marbles_remaining (initial : ℕ) (shared : ℕ) (remaining : ℕ) 
  (h1 : initial = 62)
  (h2 : shared = 33)
  (h3 : remaining = initial - shared) : 
  remaining = 29 := by
  sorry

end NUMINAMATH_CALUDE_jack_marbles_remaining_l1896_189600


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1896_189641

theorem product_of_three_numbers (p q r m : ℝ) 
  (sum_eq : p + q + r = 180)
  (p_eq : 8 * p = m)
  (q_eq : q - 10 = m)
  (r_eq : r + 10 = m)
  (p_smallest : p < q ∧ p < r) :
  p * q * r = 90000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1896_189641


namespace NUMINAMATH_CALUDE_billys_remaining_crayons_l1896_189631

/-- Given Billy's initial number of crayons and the number eaten by a hippopotamus,
    this theorem proves that the remaining number of crayons is the difference between
    the initial number and the number eaten. -/
theorem billys_remaining_crayons (initial : ℕ) (eaten : ℕ) :
  initial ≥ eaten → initial - eaten = initial - eaten :=
by
  sorry

end NUMINAMATH_CALUDE_billys_remaining_crayons_l1896_189631


namespace NUMINAMATH_CALUDE_delores_initial_money_l1896_189678

/-- Calculates the final price of an item after applying discount and sales tax -/
def finalPrice (originalPrice discount salesTax : ℚ) : ℚ :=
  (originalPrice * (1 - discount)) * (1 + salesTax)

/-- Represents the problem of calculating Delores' initial amount of money -/
theorem delores_initial_money (computerPrice printerPrice headphonesPrice : ℚ)
  (computerDiscount computerTax printerTax headphonesTax leftoverMoney : ℚ) :
  computerPrice = 400 →
  printerPrice = 40 →
  headphonesPrice = 60 →
  computerDiscount = 0.1 →
  computerTax = 0.08 →
  printerTax = 0.05 →
  headphonesTax = 0.06 →
  leftoverMoney = 10 →
  ∃ initialMoney : ℚ,
    initialMoney = 
      finalPrice computerPrice computerDiscount computerTax +
      finalPrice printerPrice 0 printerTax +
      finalPrice headphonesPrice 0 headphonesTax +
      leftoverMoney ∧
    initialMoney = 504.4 := by
  sorry

end NUMINAMATH_CALUDE_delores_initial_money_l1896_189678


namespace NUMINAMATH_CALUDE_hotel_towels_l1896_189675

/-- A hotel with a fixed number of rooms, people per room, and towels per person. -/
structure Hotel where
  rooms : ℕ
  peoplePerRoom : ℕ
  towelsPerPerson : ℕ

/-- Calculate the total number of towels handed out in a full hotel. -/
def totalTowels (h : Hotel) : ℕ :=
  h.rooms * h.peoplePerRoom * h.towelsPerPerson

/-- Theorem stating that a specific hotel configuration hands out 60 towels. -/
theorem hotel_towels :
  ∃ (h : Hotel), h.rooms = 10 ∧ h.peoplePerRoom = 3 ∧ h.towelsPerPerson = 2 ∧ totalTowels h = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_hotel_towels_l1896_189675


namespace NUMINAMATH_CALUDE_ratio_expression_l1896_189666

theorem ratio_expression (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_l1896_189666


namespace NUMINAMATH_CALUDE_sine_cosine_sum_l1896_189610

theorem sine_cosine_sum (α : Real) : 
  (∃ (x y : Real), x = 3 ∧ y = -4 ∧ x = 5 * Real.cos α ∧ y = 5 * Real.sin α) →
  Real.sin α + 2 * Real.cos α = 2/5 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_l1896_189610


namespace NUMINAMATH_CALUDE_garden_radius_increase_l1896_189664

theorem garden_radius_increase (initial_circumference final_circumference : ℝ) 
  (h1 : initial_circumference = 40)
  (h2 : final_circumference = 50) :
  (final_circumference / (2 * Real.pi)) - (initial_circumference / (2 * Real.pi)) = 5 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_garden_radius_increase_l1896_189664


namespace NUMINAMATH_CALUDE_mark_kate_difference_l1896_189606

/-- Represents the hours charged by each person on the project -/
structure ProjectHours where
  kate : ℝ
  pat : ℝ
  ravi : ℝ
  sarah : ℝ
  mark : ℝ

/-- Defines the conditions of the project hours problem -/
def validProjectHours (h : ProjectHours) : Prop :=
  h.pat = 2 * h.kate ∧
  h.ravi = 1.5 * h.kate ∧
  h.sarah = 4 * h.ravi ∧
  h.sarah = 2/3 * h.mark ∧
  h.kate + h.pat + h.ravi + h.sarah + h.mark = 310

/-- Theorem stating the difference between Mark's and Kate's hours -/
theorem mark_kate_difference (h : ProjectHours) (hvalid : validProjectHours h) :
  ∃ ε > 0, |h.mark - h.kate - 127.2| < ε :=
sorry

end NUMINAMATH_CALUDE_mark_kate_difference_l1896_189606


namespace NUMINAMATH_CALUDE_path_width_l1896_189683

theorem path_width (R r : ℝ) (h1 : R > r) (h2 : 2 * π * R - 2 * π * r = 15 * π) : R - r = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_path_width_l1896_189683


namespace NUMINAMATH_CALUDE_expression_proof_l1896_189607

theorem expression_proof (a b E : ℝ) 
  (h1 : a / b = 4 / 3) 
  (h2 : E / (3 * a - 2 * b) = 3) : 
  E = 6 * b := by
sorry

end NUMINAMATH_CALUDE_expression_proof_l1896_189607


namespace NUMINAMATH_CALUDE_largest_integer_less_than_150_over_11_l1896_189645

theorem largest_integer_less_than_150_over_11 : 
  ∃ (x : ℤ), (∀ (y : ℤ), 11 * y < 150 → y ≤ x) ∧ (11 * x < 150) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_150_over_11_l1896_189645


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1896_189661

theorem decimal_to_fraction : (2.25 : ℚ) = 9 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1896_189661


namespace NUMINAMATH_CALUDE_problem_statement_problem_statement_2_l1896_189612

/-- The quadratic function used in the problem -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 8

/-- Proposition p: x^2 - 2x - 8 ≤ 0 -/
def p (x : ℝ) : Prop := f x ≤ 0

/-- Proposition q: 2 - m ≤ x ≤ 2 + m -/
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

theorem problem_statement (m : ℝ) (h : m > 0) :
  (∀ x, p x → q m x) ∧ (∃ x, q m x ∧ ¬p x) → m ≥ 4 :=
sorry

theorem problem_statement_2 (x : ℝ) :
  let m := 5
  (p x ∨ q m x) ∧ ¬(p x ∧ q m x) →
  (-3 ≤ x ∧ x < -2) ∨ (4 < x ∧ x ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_problem_statement_2_l1896_189612


namespace NUMINAMATH_CALUDE_extremum_at_three_l1896_189649

noncomputable def f (x : ℝ) : ℝ := (x - 2) / Real.exp x

theorem extremum_at_three :
  ∀ x₀ : ℝ, (∀ x : ℝ, f x ≤ f x₀) ∨ (∀ x : ℝ, f x ≥ f x₀) → x₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_extremum_at_three_l1896_189649


namespace NUMINAMATH_CALUDE_total_ants_l1896_189647

theorem total_ants (red_ants : ℕ) (black_ants : ℕ) 
  (h1 : red_ants = 413) (h2 : black_ants = 487) : 
  red_ants + black_ants = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_ants_l1896_189647


namespace NUMINAMATH_CALUDE_total_shells_eq_195_l1896_189688

/-- The number of shells David has -/
def david_shells : ℕ := 15

/-- The number of shells Mia has -/
def mia_shells : ℕ := 4 * david_shells

/-- The number of shells Ava has -/
def ava_shells : ℕ := mia_shells + 20

/-- The number of shells Alice has -/
def alice_shells : ℕ := ava_shells / 2

/-- The total number of shells -/
def total_shells : ℕ := david_shells + mia_shells + ava_shells + alice_shells

theorem total_shells_eq_195 : total_shells = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_eq_195_l1896_189688


namespace NUMINAMATH_CALUDE_max_colors_is_six_l1896_189654

/-- A cube is a structure with edges and a coloring function. -/
structure Cube where
  edges : Finset (Fin 12)
  coloring : Fin 12 → Nat

/-- Two edges are adjacent if they share a common vertex. -/
def adjacent (e1 e2 : Fin 12) : Prop := sorry

/-- A valid coloring satisfies the problem conditions. -/
def valid_coloring (c : Cube) : Prop :=
  ∀ (color1 color2 : Nat), color1 ≠ color2 →
    ∃ (e1 e2 : Fin 12), adjacent e1 e2 ∧ c.coloring e1 = color1 ∧ c.coloring e2 = color2

/-- The maximum number of colors that can be used. -/
def max_colors (c : Cube) : Nat :=
  Finset.card (Finset.image c.coloring c.edges)

/-- The main theorem: The maximum number of colors is 6. -/
theorem max_colors_is_six (c : Cube) (h : valid_coloring c) : max_colors c = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_colors_is_six_l1896_189654


namespace NUMINAMATH_CALUDE_line_properties_l1896_189644

/-- Represents a line in the form ax + 3y + 1 = 0 -/
structure Line where
  a : ℝ

/-- Checks if the intercepts of the line on the coordinate axes are equal -/
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.a = 3

/-- Checks if the line l is parallel to the line x + (a-2)y + a = 0 -/
def is_parallel_to_given_line (l : Line) : Prop :=
  l.a * (l.a - 2) - 3 = 0 ∧ l.a^2 - 1 ≠ 0

theorem line_properties (l : Line) :
  (has_equal_intercepts l ↔ l.a = 3) ∧
  (is_parallel_to_given_line l ↔ l.a = 3) := by sorry

end NUMINAMATH_CALUDE_line_properties_l1896_189644


namespace NUMINAMATH_CALUDE_stating_reduce_to_zero_iff_even_odds_l1896_189695

/-- 
Given a natural number n, this function returns true if it's possible to reduce
all numbers in the sequence 1 to n to zero by repeatedly replacing any two numbers
with their difference, and false otherwise.
-/
def canReduceToZero (n : ℕ) : Prop :=
  Even ((n + 1) / 2)

/-- 
Theorem stating that for a sequence of integers from 1 to n, it's possible to reduce
all numbers to zero using the given operation if and only if the number of odd integers
in the sequence is even.
-/
theorem reduce_to_zero_iff_even_odds (n : ℕ) :
  canReduceToZero n ↔ Even ((n + 1) / 2) := by sorry

end NUMINAMATH_CALUDE_stating_reduce_to_zero_iff_even_odds_l1896_189695


namespace NUMINAMATH_CALUDE_x_range_l1896_189613

theorem x_range (x : ℝ) (h1 : (1 : ℝ) / x < 3) (h2 : (1 : ℝ) / x > -4) : 
  x > 1/3 ∨ x < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1896_189613


namespace NUMINAMATH_CALUDE_product_72_difference_sum_l1896_189673

theorem product_72_difference_sum (P Q R S : ℕ+) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P * Q = 72 →
  R * S = 72 →
  (P : ℤ) - (Q : ℤ) = (R : ℤ) + (S : ℤ) →
  P = 18 :=
by sorry

end NUMINAMATH_CALUDE_product_72_difference_sum_l1896_189673


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l1896_189627

theorem salary_increase_percentage (x : ℝ) : 
  (((100 + x) / 100) * 0.8 = 1.04) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l1896_189627


namespace NUMINAMATH_CALUDE_larger_cuboid_height_l1896_189628

/-- The height of a larger cuboid given its dimensions and the number and dimensions of smaller cuboids it contains. -/
theorem larger_cuboid_height (length width : ℝ) (num_small_cuboids : ℕ) 
  (small_length small_width small_height : ℝ) : 
  length = 12 →
  width = 14 →
  num_small_cuboids = 56 →
  small_length = 5 →
  small_width = 3 →
  small_height = 2 →
  (length * width * (num_small_cuboids * small_length * small_width * small_height) / (length * width)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_larger_cuboid_height_l1896_189628


namespace NUMINAMATH_CALUDE_square_numbers_between_20_and_120_divisible_by_3_l1896_189699

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem square_numbers_between_20_and_120_divisible_by_3 :
  {x : ℕ | is_square x ∧ x % 3 = 0 ∧ 20 < x ∧ x < 120} = {36, 81} := by
  sorry

end NUMINAMATH_CALUDE_square_numbers_between_20_and_120_divisible_by_3_l1896_189699


namespace NUMINAMATH_CALUDE_soccer_team_bottles_l1896_189665

theorem soccer_team_bottles (total_bottles : ℕ) (football_players : ℕ) (football_bottles_per_player : ℕ)
  (lacrosse_extra_bottles : ℕ) (rugby_bottles : ℕ) :
  total_bottles = 254 →
  football_players = 11 →
  football_bottles_per_player = 6 →
  lacrosse_extra_bottles = 12 →
  rugby_bottles = 49 →
  total_bottles - (football_players * football_bottles_per_player + 
    (football_players * football_bottles_per_player + lacrosse_extra_bottles) + 
    rugby_bottles) = 61 := by
  sorry

#check soccer_team_bottles

end NUMINAMATH_CALUDE_soccer_team_bottles_l1896_189665


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l1896_189636

theorem arithmetic_expression_equals_24 : 
  (2 + 4 / 10) * 10 = 24 := by sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l1896_189636


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_sixteen_l1896_189668

theorem sum_of_fractions_equals_sixteen :
  let fractions : List ℚ := [2/10, 4/10, 6/10, 8/10, 10/10, 15/10, 20/10, 25/10, 30/10, 40/10]
  fractions.sum = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_sixteen_l1896_189668


namespace NUMINAMATH_CALUDE_get_ready_time_l1896_189639

/-- The time it takes for Jack and his two toddlers to get ready -/
def total_time (jack_socks jack_shoes jack_jacket toddler_socks toddler_shoes toddler_shoelaces : ℕ) : ℕ :=
  let jack_time := jack_socks + jack_shoes + jack_jacket
  let toddler_time := toddler_socks + toddler_shoes + 2 * toddler_shoelaces
  jack_time + 2 * toddler_time

theorem get_ready_time :
  total_time 2 4 3 2 5 1 = 27 :=
by sorry

end NUMINAMATH_CALUDE_get_ready_time_l1896_189639


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1896_189698

/-- The minimum value of k*sin^4(x) + cos^4(x) for k ≥ 0 is 0 -/
theorem min_value_trig_expression (k : ℝ) (hk : k ≥ 0) :
  ∃ m : ℝ, m = 0 ∧ ∀ x : ℝ, k * Real.sin x ^ 4 + Real.cos x ^ 4 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1896_189698


namespace NUMINAMATH_CALUDE_smallest_x_factorization_l1896_189662

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def is_perfect_fifth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^5

theorem smallest_x_factorization :
  let x := 2^15 * 3^20 * 5^24
  ∀ y : ℕ, y > 0 →
    (is_perfect_square (2*y) ∧ 
     is_perfect_cube (3*y) ∧ 
     is_perfect_fifth_power (5*y)) →
    y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_factorization_l1896_189662


namespace NUMINAMATH_CALUDE_average_marks_chem_math_l1896_189680

/-- Given that the total marks in physics, chemistry, and mathematics is 150 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 75. -/
theorem average_marks_chem_math (P C M : ℝ) 
  (h : P + C + M = P + 150) : (C + M) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chem_math_l1896_189680


namespace NUMINAMATH_CALUDE_exists_special_function_l1896_189672

theorem exists_special_function :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f (n + 1)) = f (f n) + 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l1896_189672


namespace NUMINAMATH_CALUDE_x_twelfth_power_l1896_189608

theorem x_twelfth_power (x : ℝ) (h : x + 1/x = 2 * Real.sqrt 2) : x^12 = 46656 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_l1896_189608


namespace NUMINAMATH_CALUDE_bill_amount_calculation_l1896_189679

/-- Given a true discount and a banker's discount, calculate the amount of the bill. -/
def billAmount (trueDiscount : ℚ) (bankersDiscount : ℚ) : ℚ :=
  trueDiscount + trueDiscount

/-- Theorem: Given a true discount of 360 and a banker's discount of 428.21, the amount of the bill is 720. -/
theorem bill_amount_calculation :
  let trueDiscount : ℚ := 360
  let bankersDiscount : ℚ := 428.21
  billAmount trueDiscount bankersDiscount = 720 := by
  sorry

#eval billAmount 360 428.21

end NUMINAMATH_CALUDE_bill_amount_calculation_l1896_189679


namespace NUMINAMATH_CALUDE_project_cost_sharing_l1896_189671

/-- Given initial payments P and Q, and an additional cost R, 
    calculate the amount Javier must pay to Cora for equal cost sharing. -/
theorem project_cost_sharing 
  (P Q R : ℝ) 
  (h1 : R = 3 * Q - 2 * P) 
  (h2 : P < Q) : 
  (2 * Q - P) / 2 = (P + Q + R) / 2 - Q := by sorry

end NUMINAMATH_CALUDE_project_cost_sharing_l1896_189671


namespace NUMINAMATH_CALUDE_only_statement5_true_l1896_189693

-- Define the statements as functions
def statement1 (a b : ℝ) : Prop := b * (a + b) = b * a + b * b
def statement2 (x y : ℝ) : Prop := Real.log (x + y) = Real.log x + Real.log y
def statement3 (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2
def statement4 (a b : ℝ) : Prop := b^(a + b) = b^a + b^b
def statement5 (x y : ℝ) : Prop := x^2 / y^2 = (x / y)^2

-- Theorem stating that only statement5 is true for all real numbers
theorem only_statement5_true :
  (∀ x y : ℝ, statement5 x y) ∧
  (∃ a b : ℝ, ¬statement1 a b) ∧
  (∃ x y : ℝ, ¬statement2 x y) ∧
  (∃ x y : ℝ, ¬statement3 x y) ∧
  (∃ a b : ℝ, ¬statement4 a b) :=
sorry

end NUMINAMATH_CALUDE_only_statement5_true_l1896_189693


namespace NUMINAMATH_CALUDE_julia_played_with_two_kids_on_monday_l1896_189618

/-- The number of kids Julia played with on Monday and Tuesday -/
def total_kids : ℕ := 16

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 14

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := total_kids - tuesday_kids

theorem julia_played_with_two_kids_on_monday :
  monday_kids = 2 := by sorry

end NUMINAMATH_CALUDE_julia_played_with_two_kids_on_monday_l1896_189618


namespace NUMINAMATH_CALUDE_gum_pack_size_l1896_189670

theorem gum_pack_size (y : ℝ) : 
  (25 - 2 * y) / 40 = 25 / (40 + 4 * y) → y = 2.5 := by
sorry

end NUMINAMATH_CALUDE_gum_pack_size_l1896_189670


namespace NUMINAMATH_CALUDE_closest_point_is_correct_l1896_189605

/-- The point on the line y = -4x - 8 that is closest to (3, 6) -/
def closest_point : ℚ × ℚ := (-53/17, 76/17)

/-- The line y = -4x - 8 -/
def mouse_trajectory (x : ℚ) : ℚ := -4 * x - 8

theorem closest_point_is_correct :
  let (a, b) := closest_point
  -- The point is on the line
  (mouse_trajectory a = b) ∧
  -- It's the closest point to (3, 6)
  (∀ x y, mouse_trajectory x = y →
    (x - 3)^2 + (y - 6)^2 ≥ (a - 3)^2 + (b - 6)^2) ∧
  -- The sum of its coordinates is 23/17
  (a + b = 23/17) := by sorry


end NUMINAMATH_CALUDE_closest_point_is_correct_l1896_189605


namespace NUMINAMATH_CALUDE_indefinite_integral_ln_4x2_plus_1_l1896_189638

open Real

theorem indefinite_integral_ln_4x2_plus_1 (x : ℝ) :
  (deriv fun x => x * log (4 * x^2 + 1) - 8 * x + 4 * arctan (2 * x)) x = log (4 * x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_ln_4x2_plus_1_l1896_189638


namespace NUMINAMATH_CALUDE_length_of_trisected_segment_l1896_189640

/-- Given a line segment AD with points B, C, and M, where:
  * B and C trisect AD
  * M is one-third the way from A to D
  * The length of MC is 5
  Prove that the length of AD is 15. -/
theorem length_of_trisected_segment (A B C D M : ℝ) : 
  (B - A = C - B) ∧ (C - B = D - C) →  -- B and C trisect AD
  (M - A = (1/3) * (D - A)) →          -- M is one-third the way from A to D
  (C - M = 5) →                        -- MC = 5
  (D - A = 15) :=                      -- AD = 15
by sorry

end NUMINAMATH_CALUDE_length_of_trisected_segment_l1896_189640


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_l1896_189681

def small_bottle : ℕ := 25
def medium_bottle : ℕ := 75
def large_bottle : ℕ := 600

theorem min_bottles_to_fill :
  (large_bottle / medium_bottle : ℕ) = 8 ∧
  large_bottle % medium_bottle = 0 ∧
  (∀ n : ℕ, n < 8 → n * medium_bottle < large_bottle) :=
by sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_l1896_189681


namespace NUMINAMATH_CALUDE_water_remaining_l1896_189669

/-- Given 3 gallons of water and using 5/4 gallons, prove that the remaining amount is 7/4 gallons. -/
theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l1896_189669


namespace NUMINAMATH_CALUDE_boys_camp_total_l1896_189624

theorem boys_camp_total (total : ℕ) : 
  (total * 20 / 100 : ℕ) > 0 →  -- Ensure there are boys from school A
  (((total * 20 / 100) * 70 / 100 : ℕ) = 35) →  -- 35 boys from school A not studying science
  total = 250 := by
sorry

end NUMINAMATH_CALUDE_boys_camp_total_l1896_189624


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1896_189602

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 5 ≤ x + 1) ∧ ((x - 1) / 2 > x - 4) ↔ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1896_189602


namespace NUMINAMATH_CALUDE_special_line_properties_l1896_189609

/-- A line passing through (1,3) with intercepts that are negatives of each other -/
def special_line (x y : ℝ) : Prop :=
  y = x + 2

theorem special_line_properties :
  (∃ a : ℝ, a ≠ 0 ∧ special_line a (-a)) ∧ 
  special_line 1 3 :=
by sorry

end NUMINAMATH_CALUDE_special_line_properties_l1896_189609


namespace NUMINAMATH_CALUDE_longest_tape_measure_l1896_189685

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 600) (hb : b = 500) (hc : c = 1200) : 
  Nat.gcd a (Nat.gcd b c) = 100 := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_measure_l1896_189685


namespace NUMINAMATH_CALUDE_eighth_term_is_84_l1896_189617

/-- The n-th term of the sequence -/
def S (n : ℕ) : ℚ := (3 * n * (n - 1)) / 2

/-- Theorem: The 8th term of the sequence is 84 -/
theorem eighth_term_is_84 : S 8 = 84 := by sorry

end NUMINAMATH_CALUDE_eighth_term_is_84_l1896_189617


namespace NUMINAMATH_CALUDE_sine_is_periodic_l1896_189652

-- Define the properties
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the sine function
def sin : ℝ → ℝ := sorry

-- Theorem statement
theorem sine_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric sin →
  IsPeriodic sin := by sorry

end NUMINAMATH_CALUDE_sine_is_periodic_l1896_189652


namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l1896_189694

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 4 + 8 = 11) → initial_people = 7 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l1896_189694


namespace NUMINAMATH_CALUDE_sum_of_21st_set_l1896_189696

/-- The sum of elements in the n-th set of a sequence where:
    1. Each set contains consecutive integers
    2. Each set contains one more element than the previous set
    3. The first element of each set is one greater than the last element of the previous set
-/
def S (n : ℕ) : ℚ :=
  n * (n^2 - n + 2) / 2

theorem sum_of_21st_set :
  S 21 = 4641 :=
sorry

end NUMINAMATH_CALUDE_sum_of_21st_set_l1896_189696
