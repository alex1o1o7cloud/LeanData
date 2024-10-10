import Mathlib

namespace f_value_at_pi_24_max_monotone_interval_exists_max_monotone_interval_l1812_181250

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2

theorem f_value_at_pi_24 : f (Real.pi / 24) = Real.sqrt 2 + 1 := by sorry

theorem max_monotone_interval : 
  ∀ m : ℝ, (∀ x y : ℝ, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) → m ≤ Real.pi / 6 := by sorry

theorem exists_max_monotone_interval : 
  ∃ m : ℝ, m = Real.pi / 6 ∧ 
    (∀ x y : ℝ, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) ∧
    (∀ m' : ℝ, m' > Real.pi / 6 → ¬(∀ x y : ℝ, -m' ≤ x ∧ x < y ∧ y ≤ m' → f x < f y)) := by sorry

end f_value_at_pi_24_max_monotone_interval_exists_max_monotone_interval_l1812_181250


namespace stock_market_value_l1812_181245

/-- Given a stock with a 10% dividend rate and an 8% yield, its market value is $125. -/
theorem stock_market_value (face_value : ℝ) (dividend_rate : ℝ) (yield : ℝ) : 
  dividend_rate = 0.1 → yield = 0.08 → (dividend_rate * face_value) / yield = 125 := by
  sorry

end stock_market_value_l1812_181245


namespace fraction_sum_positive_l1812_181241

theorem fraction_sum_positive (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (1 / (a - b) + 1 / (b - c) + 1 / (c - a)) > 0 := by
  sorry

end fraction_sum_positive_l1812_181241


namespace circle_intersection_properties_l1812_181256

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the intersection points
def intersectionPoints (A B : ℝ × ℝ) : Prop :=
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧ A ≠ B

-- Theorem statement
theorem circle_intersection_properties
  (A B : ℝ × ℝ) (h : intersectionPoints A B) :
  -- 1. Equation of the line containing chord AB
  (∀ x y : ℝ, (x - y - 3 = 0) ↔ (∃ t : ℝ, x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2))) ∧
  -- 2. Length of the common chord AB
  Real.sqrt 2 = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
  -- 3. Equation of the perpendicular bisector of AB
  (∀ x y : ℝ, (x + y = 0) ↔ ((x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4)) :=
by sorry

end circle_intersection_properties_l1812_181256


namespace back_seat_capacity_is_nine_l1812_181215

/-- Represents the seating capacity of a bus -/
structure BusSeats where
  leftSeats : Nat
  rightSeats : Nat
  peoplePerSeat : Nat
  totalCapacity : Nat

/-- Calculates the number of people who can sit at the back seat of the bus -/
def backSeatCapacity (bus : BusSeats) : Nat :=
  bus.totalCapacity - (bus.leftSeats + bus.rightSeats) * bus.peoplePerSeat

/-- Theorem stating the back seat capacity of the given bus configuration -/
theorem back_seat_capacity_is_nine :
  let bus : BusSeats := {
    leftSeats := 15,
    rightSeats := 12,
    peoplePerSeat := 3,
    totalCapacity := 90
  }
  backSeatCapacity bus = 9 := by sorry

end back_seat_capacity_is_nine_l1812_181215


namespace probability_continuous_stripe_l1812_181239

/-- Represents a single cube with stripes on its faces -/
structure StripedCube where
  faces : Fin 6 → Bool

/-- Represents a tower of three cubes -/
structure CubeTower where
  top : StripedCube
  middle : StripedCube
  bottom : StripedCube

/-- Checks if there's a continuous stripe through a vertical face pair -/
def has_continuous_stripe (face_pair : Fin 4) (tower : CubeTower) : Bool :=
  sorry

/-- Counts the number of cube towers with a continuous vertical stripe -/
def count_continuous_stripes (towers : List CubeTower) : Nat :=
  sorry

/-- The total number of possible cube tower configurations -/
def total_configurations : Nat := 2^18

/-- The number of cube tower configurations with a continuous vertical stripe -/
def favorable_configurations : Nat := 64

theorem probability_continuous_stripe :
  (favorable_configurations : ℚ) / total_configurations = 1 / 4096 :=
sorry

end probability_continuous_stripe_l1812_181239


namespace max_value_product_sum_l1812_181234

theorem max_value_product_sum (X Y Z : ℕ) (sum_constraint : X + Y + Z = 15) :
  (∀ a b c : ℕ, a + b + c = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ a * b * c + a * b + b * c + c * a) →
  X * Y * Z + X * Y + Y * Z + Z * X = 200 :=
by sorry

end max_value_product_sum_l1812_181234


namespace sqrt_nested_equality_l1812_181237

theorem sqrt_nested_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt x)) = (x^7)^(1/8) := by
  sorry

end sqrt_nested_equality_l1812_181237


namespace some_magical_creatures_are_mystical_beings_l1812_181246

-- Define our sets
variable (U : Type) -- Universe set
variable (D : Set U) -- Set of dragons
variable (M : Set U) -- Set of magical creatures
variable (B : Set U) -- Set of mystical beings

-- Define our premises
variable (h1 : D ⊆ M) -- All dragons are magical creatures
variable (h2 : ∃ x, x ∈ B ∩ D) -- Some mystical beings are dragons

-- Theorem to prove
theorem some_magical_creatures_are_mystical_beings : 
  ∃ x, x ∈ M ∩ B := by sorry

end some_magical_creatures_are_mystical_beings_l1812_181246


namespace reciprocal_complement_sum_square_l1812_181273

theorem reciprocal_complement_sum_square (p q r : ℝ) (h : p * q * r = 1) :
  (1 / (1 - p))^2 + (1 / (1 - q))^2 + (1 / (1 - r))^2 ≥ 1 := by
  sorry

end reciprocal_complement_sum_square_l1812_181273


namespace base8_4523_equals_2387_l1812_181212

def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base8_4523_equals_2387 :
  base8_to_base10 [3, 2, 5, 4] = 2387 := by
  sorry

end base8_4523_equals_2387_l1812_181212


namespace expression_value_l1812_181206

-- Define the expression
def f (x : ℝ) : ℝ := 3 * x^2 + 5

-- Theorem statement
theorem expression_value : f (-1) = 8 := by sorry

end expression_value_l1812_181206


namespace coordinates_sum_of_point_b_l1812_181243

/-- Given two points A and B, where A is at the origin and B is on the line y=5,
    and the slope of segment AB is 3/4, prove that the sum of the x- and y-coordinates of B is 35/3 -/
theorem coordinates_sum_of_point_b (A B : ℝ × ℝ) : 
  A = (0, 0) →
  B.2 = 5 →
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 →
  B.1 + B.2 = 35 / 3 := by
  sorry

end coordinates_sum_of_point_b_l1812_181243


namespace negation_of_universal_quadratic_inequality_l1812_181232

theorem negation_of_universal_quadratic_inequality :
  ¬(∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) ↔ ∃ x : ℝ, x^2 - 2*x + 1 < 0 :=
by sorry

end negation_of_universal_quadratic_inequality_l1812_181232


namespace rain_probability_l1812_181217

/-- Probability of rain on at least one of two days -/
theorem rain_probability (p1 p2 p2_given_r1 : ℝ) 
  (h1 : p1 = 0.3) 
  (h2 : p2 = 0.4) 
  (h3 : p2_given_r1 = 0.7) 
  (h4 : 0 ≤ p1 ∧ p1 ≤ 1)
  (h5 : 0 ≤ p2 ∧ p2 ≤ 1)
  (h6 : 0 ≤ p2_given_r1 ∧ p2_given_r1 ≤ 1) : 
  1 - ((1 - p1) * (1 - p2) + p1 * (1 - p2_given_r1)) = 0.49 := by
sorry

end rain_probability_l1812_181217


namespace expression_evaluation_l1812_181251

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5 - 1/6) = 37 := by
  sorry

end expression_evaluation_l1812_181251


namespace circle_radius_l1812_181242

theorem circle_radius (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 40) :
  ∃ (r : ℝ), r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 80 := by
  sorry

end circle_radius_l1812_181242


namespace age_ratio_problem_l1812_181261

theorem age_ratio_problem (a b : ℕ) 
  (h1 : a = 2 * b)  -- Present age ratio 6:3 simplifies to 2:1
  (h2 : a - 4 = b + 4)  -- A's age 4 years ago equals B's age 4 years hence
  : (a + 4) / (b - 4) = 5 := by
sorry

end age_ratio_problem_l1812_181261


namespace weight_of_new_person_l1812_181258

theorem weight_of_new_person (initial_count : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  initial_count = 15 →
  avg_increase = 2.3 →
  old_weight = 80 →
  let total_increase := initial_count * avg_increase
  let new_weight := old_weight + total_increase
  new_weight = 114.5 := by
  sorry

end weight_of_new_person_l1812_181258


namespace iron_nickel_percentage_l1812_181269

/-- Represents the exchange of quarters for nickels, including special iron nickels --/
def nickel_exchange (num_quarters : ℕ) (total_value : ℚ) (iron_nickel_value : ℚ) : Prop :=
  ∃ (num_iron_nickels : ℕ),
    let num_nickels : ℕ := num_quarters * 5
    let regular_nickel_value : ℚ := 1/20
    (num_iron_nickels : ℚ) * iron_nickel_value + 
    ((num_nickels - num_iron_nickels) : ℚ) * regular_nickel_value = total_value ∧
    (num_iron_nickels : ℚ) / (num_nickels : ℚ) = 1/5

theorem iron_nickel_percentage 
  (h : nickel_exchange 20 64 3) : 
  ∃ (num_iron_nickels : ℕ), 
    (num_iron_nickels : ℚ) / 100 = 1/5 :=
by
  sorry

#check iron_nickel_percentage

end iron_nickel_percentage_l1812_181269


namespace x_power_2023_l1812_181203

theorem x_power_2023 (x : ℝ) (h : (x - 1) * (x^4 + x^3 + x^2 + x + 1) = -2) : x^2023 = -1 := by
  sorry

end x_power_2023_l1812_181203


namespace range_of_m_for_real_solutions_l1812_181240

theorem range_of_m_for_real_solutions (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y + Real.sin y ^ 2 + m - 4 = 0) →
  0 ≤ m ∧ m ≤ 8 :=
by sorry

end range_of_m_for_real_solutions_l1812_181240


namespace digit2015_is_8_l1812_181288

/-- The function that generates the list of digits of positive even numbers -/
def evenNumberDigits : ℕ → List ℕ := sorry

/-- The 2015th digit in the list of digits of positive even numbers -/
def digit2015 : ℕ := (evenNumberDigits 0).nthLe 2014 sorry

/-- Theorem stating that the 2015th digit is 8 -/
theorem digit2015_is_8 : digit2015 = 8 := by sorry

end digit2015_is_8_l1812_181288


namespace joannas_family_money_ratio_l1812_181286

/-- Prove that given the conditions of Joanna's family's money, the ratio of her brother's money to Joanna's money is 3:1 -/
theorem joannas_family_money_ratio :
  ∀ (brother_multiple : ℚ),
  (8 : ℚ) + 8 * brother_multiple + 4 = 36 →
  brother_multiple = 3 :=
by sorry

end joannas_family_money_ratio_l1812_181286


namespace quadrupled_container_volume_l1812_181264

/-- A container with an initial volume and a scale factor for its dimensions. -/
structure Container :=
  (initial_volume : ℝ)
  (scale_factor : ℝ)

/-- The new volume of a container after scaling its dimensions. -/
def new_volume (c : Container) : ℝ :=
  c.initial_volume * c.scale_factor^3

/-- Theorem stating that a container with 5 gallons initial volume and dimensions quadrupled results in 320 gallons. -/
theorem quadrupled_container_volume :
  let c := Container.mk 5 4
  new_volume c = 320 := by
  sorry

end quadrupled_container_volume_l1812_181264


namespace jerry_age_l1812_181226

/-- Given that Mickey's age is 16 and Mickey's age is 6 years less than 200% of Jerry's age,
    prove that Jerry's age is 11. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 16) 
  (h2 : mickey_age = 2 * jerry_age - 6) : 
  jerry_age = 11 := by
  sorry

end jerry_age_l1812_181226


namespace one_thirds_in_nine_halves_l1812_181299

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 27 / 2 := by
  sorry

end one_thirds_in_nine_halves_l1812_181299


namespace point_not_in_region_iff_m_in_interval_l1812_181221

/-- The function representing the left side of the inequality -/
def f (m x y : ℝ) : ℝ := x - (m^2 - 2*m + 4)*y - 6

/-- The theorem stating the equivalence between the point (-1, -1) not being in the region
    and m being in the interval [-1, 3] -/
theorem point_not_in_region_iff_m_in_interval :
  ∀ m : ℝ, f m (-1) (-1) ≤ 0 ↔ -1 ≤ m ∧ m ≤ 3 := by sorry

end point_not_in_region_iff_m_in_interval_l1812_181221


namespace highest_probability_high_speed_rail_l1812_181210

theorem highest_probability_high_speed_rail (beidou tianyan high_speed_rail : ℕ) :
  beidou = 3 →
  tianyan = 2 →
  high_speed_rail = 5 →
  let total := beidou + tianyan + high_speed_rail
  (high_speed_rail : ℚ) / total > (beidou : ℚ) / total ∧
  (high_speed_rail : ℚ) / total > (tianyan : ℚ) / total :=
by sorry

end highest_probability_high_speed_rail_l1812_181210


namespace integer_root_of_cubic_l1812_181277

theorem integer_root_of_cubic (a b c : ℚ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + b*x + c
  (f (3 - Real.sqrt 5) = 0) →
  (∃ r : ℤ, f r = 0) →
  (∃ r : ℤ, f r = 0 ∧ r = -6) :=
by sorry

end integer_root_of_cubic_l1812_181277


namespace fourth_root_16_times_sixth_root_64_l1812_181283

theorem fourth_root_16_times_sixth_root_64 : (16 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/6) = 4 := by
  sorry

end fourth_root_16_times_sixth_root_64_l1812_181283


namespace polygon_sides_l1812_181296

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 + 360 = 900 → n = 5 := by
sorry

end polygon_sides_l1812_181296


namespace min_sum_intercepts_l1812_181293

/-- A line passing through (1, 1) with positive intercepts -/
structure LineThroughOneOne where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : 1 / a + 1 / b = 1

/-- The sum of intercepts of a line -/
def sumOfIntercepts (l : LineThroughOneOne) : ℝ := l.a + l.b

/-- The equation x + y - 2 = 0 minimizes the sum of intercepts -/
theorem min_sum_intercepts :
  ∀ l : LineThroughOneOne, sumOfIntercepts l ≥ 4 ∧
  (sumOfIntercepts l = 4 ↔ l.a = 2 ∧ l.b = 2) :=
sorry

end min_sum_intercepts_l1812_181293


namespace f_values_l1812_181276

def f (x : ℝ) : ℝ := x^2 + x + 1

theorem f_values : f 2 = 7 ∧ f (f 1) = 13 := by sorry

end f_values_l1812_181276


namespace circle_center_l1812_181278

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 - 8*y - 16 = 0

-- Define the center of a circle
def is_center (h k : ℝ) (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ r, ∀ x y, eq x y ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_center :
  is_center 3 4 circle_equation :=
sorry

end circle_center_l1812_181278


namespace crayon_difference_is_1040_l1812_181272

/-- The number of crayons Willy and Lucy have combined, minus the number of crayons Max has -/
def crayon_difference (willy_crayons lucy_crayons max_crayons : ℕ) : ℕ :=
  (willy_crayons + lucy_crayons) - max_crayons

/-- Theorem stating that the difference in crayons is 1040 -/
theorem crayon_difference_is_1040 :
  crayon_difference 1400 290 650 = 1040 := by
  sorry

end crayon_difference_is_1040_l1812_181272


namespace original_number_proof_l1812_181209

theorem original_number_proof (x : ℚ) : 1 + (1 / x) = 9 / 5 → x = 5 / 4 := by
  sorry

end original_number_proof_l1812_181209


namespace smallest_b_for_five_not_in_range_l1812_181255

theorem smallest_b_for_five_not_in_range :
  ∃ (b : ℤ), (∀ x : ℝ, x^2 + b*x + 10 ≠ 5) ∧
             (∀ c : ℤ, c < b → ∃ x : ℝ, x^2 + c*x + 10 = 5) ∧
             b = -4 :=
by sorry

end smallest_b_for_five_not_in_range_l1812_181255


namespace marys_remaining_cards_l1812_181224

theorem marys_remaining_cards (initial_cards promised_cards bought_cards : ℝ) :
  initial_cards + bought_cards - promised_cards =
  initial_cards + bought_cards - promised_cards :=
by sorry

end marys_remaining_cards_l1812_181224


namespace kendall_driving_distance_l1812_181260

/-- The distance Kendall drove with her mother -/
def distance_with_mother : ℝ := 0.67 - 0.5

/-- The total distance Kendall drove -/
def total_distance : ℝ := 0.67

/-- The distance Kendall drove with her father -/
def distance_with_father : ℝ := 0.5

theorem kendall_driving_distance :
  distance_with_mother = 0.17 :=
by sorry

end kendall_driving_distance_l1812_181260


namespace min_value_expression_l1812_181205

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^3 / (y - 2)) + (y^3 / (x - 2)) ≥ 64 ∧
  ∃ x y, x > 2 ∧ y > 2 ∧ (x^3 / (y - 2)) + (y^3 / (x - 2)) = 64 := by
  sorry

end min_value_expression_l1812_181205


namespace vector_properties_l1812_181238

/-- Prove properties of vectors a, b, and c in a plane -/
theorem vector_properties (a b c : ℝ × ℝ) (θ : ℝ) : 
  a = (1, -2) →
  ‖c‖ = 2 * Real.sqrt 5 →
  ∃ (k : ℝ), c = k • a →
  ‖b‖ = 1 →
  (a + b) • (a - 2 • b) = 0 →
  (c = (-2, 4) ∨ c = (2, -4)) ∧ 
  Real.cos θ = (3 * Real.sqrt 5) / 5 :=
by sorry


end vector_properties_l1812_181238


namespace simplified_fraction_sum_l1812_181233

theorem simplified_fraction_sum (a b : ℕ) (h : a = 75 ∧ b = 100) :
  ∃ (c d : ℕ), (c.gcd d = 1) ∧ (a * d = b * c) ∧ (c + d = 7) := by
  sorry

end simplified_fraction_sum_l1812_181233


namespace pool_swimmers_l1812_181295

theorem pool_swimmers (total : ℕ) (first_day : ℕ) (extra_second_day : ℕ) 
  (h1 : total = 246)
  (h2 : first_day = 79)
  (h3 : extra_second_day = 47) :
  ∃ (third_day : ℕ), 
    third_day = 60 ∧ 
    first_day + (third_day + extra_second_day) + third_day = total :=
by
  sorry

end pool_swimmers_l1812_181295


namespace alcohol_mixture_proof_l1812_181213

/-- Proves that 3 liters of 33% alcohol solution mixed with 1 liter of water results in 24.75% alcohol concentration -/
theorem alcohol_mixture_proof (x : ℝ) :
  (x > 0) →
  (0.33 * x = 0.2475 * (x + 1)) →
  x = 3 := by
  sorry

#check alcohol_mixture_proof

end alcohol_mixture_proof_l1812_181213


namespace total_cost_for_two_rides_l1812_181249

def base_fare : ℚ := 2
def per_mile_charge : ℚ := (3 : ℚ) / 10
def first_ride_distance : ℕ := 8
def second_ride_distance : ℕ := 5

def ride_cost (distance : ℕ) : ℚ :=
  base_fare + per_mile_charge * distance

theorem total_cost_for_two_rides :
  ride_cost first_ride_distance + ride_cost second_ride_distance = (79 : ℚ) / 10 := by
  sorry

end total_cost_for_two_rides_l1812_181249


namespace three_incorrect_statements_l1812_181211

theorem three_incorrect_statements (a b c : ℕ+) 
  (h1 : Nat.Coprime a.val b.val) 
  (h2 : Nat.Coprime b.val c.val) : 
  ∃ (a b c : ℕ+), 
    (¬(¬(b.val ∣ (a.val + c.val)^2))) ∧ 
    (¬(¬(b.val ∣ a.val^2 + c.val^2))) ∧ 
    (¬(¬(c.val ∣ (a.val + b.val)^2))) :=
sorry

end three_incorrect_statements_l1812_181211


namespace simplify_and_evaluate_l1812_181297

theorem simplify_and_evaluate : 
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  5 * (3 * a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) = 2/3 := by
  sorry

end simplify_and_evaluate_l1812_181297


namespace max_area_region_S_l1812_181236

/-- A circle in a plane -/
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

/-- The line to which the circles are tangent -/
def TangentLine : Set (ℝ × ℝ) := sorry

/-- The point at which the circles are tangent to the line -/
def TangentPoint : ℝ × ℝ := sorry

/-- The set of four circles with radii 1, 3, 5, and 7 -/
def FourCircles : Set Circle := sorry

/-- The region S composed of all points that lie within one of the four circles -/
def RegionS (circles : Set Circle) : Set (ℝ × ℝ) := sorry

/-- The area of a set in the plane -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the maximum area of region S is 65π -/
theorem max_area_region_S :
  ∃ (c : Set Circle), c = FourCircles ∧
    (∀ circle ∈ c, ∃ p ∈ TangentLine, p = TangentPoint ∧
      (circle.center.1 - p.1)^2 + (circle.center.2 - p.2)^2 = circle.radius^2) ∧
    (∀ arrangement : Set Circle, arrangement = FourCircles →
      area (RegionS arrangement) ≤ area (RegionS c)) ∧
    area (RegionS c) = 65 * Real.pi :=
sorry

end max_area_region_S_l1812_181236


namespace rectangular_box_dimension_sum_square_l1812_181216

/-- Given a rectangular box with dimensions a, b, c, where a = b + c + 10,
    prove that the square of the sum of dimensions is equal to 4(b+c)^2 + 40(b+c) + 100 -/
theorem rectangular_box_dimension_sum_square (b c : ℝ) :
  let a : ℝ := b + c + 10
  let D : ℝ := a + b + c
  D^2 = 4*(b+c)^2 + 40*(b+c) + 100 := by
  sorry

end rectangular_box_dimension_sum_square_l1812_181216


namespace value_of_b_l1812_181291

theorem value_of_b (a b c : ℝ) 
  (h1 : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
  (h2 : 6 * b * 11 = 1) : 
  b = 15 := by
sorry

end value_of_b_l1812_181291


namespace square_root_divided_by_18_equals_4_l1812_181214

theorem square_root_divided_by_18_equals_4 (x : ℝ) : 
  (Real.sqrt x) / 18 = 4 → x = 5184 := by sorry

end square_root_divided_by_18_equals_4_l1812_181214


namespace cost_per_tire_to_produce_l1812_181285

/-- Proves that the cost per tire to produce is $8 given the specified conditions --/
theorem cost_per_tire_to_produce
  (fixed_cost : ℝ)
  (selling_price : ℝ)
  (batch_size : ℝ)
  (profit_per_tire : ℝ)
  (h1 : fixed_cost = 22500)
  (h2 : selling_price = 20)
  (h3 : batch_size = 15000)
  (h4 : profit_per_tire = 10.5) :
  ∃ (cost_per_tire : ℝ),
    cost_per_tire = 8 ∧
    batch_size * (selling_price - cost_per_tire) - fixed_cost = batch_size * profit_per_tire :=
by sorry

end cost_per_tire_to_produce_l1812_181285


namespace shoe_savings_l1812_181268

theorem shoe_savings (max_budget : ℝ) (original_price : ℝ) (discount_percent : ℝ) 
  (h1 : max_budget = 130)
  (h2 : original_price = 120)
  (h3 : discount_percent = 30) : 
  max_budget - (original_price * (1 - discount_percent / 100)) = 46 := by
  sorry

end shoe_savings_l1812_181268


namespace problem_solution_l1812_181248

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 + Real.log x₀ = 0

-- Theorem statement
theorem problem_solution : ¬p ∧ q := by sorry

end problem_solution_l1812_181248


namespace a_beats_b_by_seven_seconds_l1812_181204

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ
  time : ℝ

/-- The race scenario -/
def race_scenario (a b : Runner) : Prop :=
  a.distance = 280 ∧
  b.distance = 224 ∧
  a.time = 28 ∧
  a.speed = a.distance / a.time ∧
  b.speed = b.distance / a.time

/-- Theorem stating that A beats B by 7 seconds -/
theorem a_beats_b_by_seven_seconds (a b : Runner) (h : race_scenario a b) :
  b.distance / b.speed - a.time = 7 := by
  sorry


end a_beats_b_by_seven_seconds_l1812_181204


namespace euler_formula_squared_l1812_181225

theorem euler_formula_squared (x : ℝ) : (Complex.cos x + Complex.I * Complex.sin x)^2 = Complex.cos (2*x) + Complex.I * Complex.sin (2*x) :=
by
  sorry

-- Euler's formula as an axiom
axiom euler_formula (x : ℝ) : Complex.exp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x

end euler_formula_squared_l1812_181225


namespace arithmetic_geometric_mean_inequality_condition_l1812_181259

theorem arithmetic_geometric_mean_inequality_condition (a b : ℝ) : 
  (a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b)) ∧ 
  ∃ a b : ℝ, ¬(a > 0 ∧ b > 0) ∧ (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end arithmetic_geometric_mean_inequality_condition_l1812_181259


namespace customers_who_tipped_l1812_181290

/-- The number of customers who left a tip at 'The Gourmet Kitchen' restaurant --/
theorem customers_who_tipped (total_customers early_morning_customers priority_customers regular_evening_customers : ℕ)
  (h_total : total_customers = 215)
  (h_early : early_morning_customers = 20)
  (h_priority : priority_customers = 60)
  (h_regular : regular_evening_customers = 22)
  (h_early_no_tip : ⌊early_morning_customers * (30 : ℚ) / 100⌋ = 6)
  (h_priority_no_tip : ⌊priority_customers * (60 : ℚ) / 100⌋ = 36)
  (h_regular_no_tip : ⌊regular_evening_customers * (50 : ℚ) / 100⌋ = 11)
  (h_remaining : total_customers - early_morning_customers - priority_customers - regular_evening_customers = 113)
  (h_remaining_no_tip : ⌊113 * (25 : ℚ) / 100⌋ = 28) :
  total_customers - (6 + 36 + 11 + 28) = 134 := by
  sorry

#check customers_who_tipped

end customers_who_tipped_l1812_181290


namespace pie_chart_most_appropriate_for_percentages_l1812_181281

/-- Represents different types of charts --/
inductive ChartType
| PieChart
| LineChart
| BarChart

/-- Represents characteristics of data --/
structure DataCharacteristics where
  is_percentage : Bool
  total_is_100_percent : Bool
  part_whole_relationship_important : Bool

/-- Determines the most appropriate chart type based on data characteristics --/
def most_appropriate_chart (data : DataCharacteristics) : ChartType :=
  if data.is_percentage ∧ data.total_is_100_percent ∧ data.part_whole_relationship_important then
    ChartType.PieChart
  else
    ChartType.BarChart

/-- Theorem stating that a pie chart is most appropriate for percentage data summing to 100% 
    where the part-whole relationship is important --/
theorem pie_chart_most_appropriate_for_percentages 
  (data : DataCharacteristics) 
  (h1 : data.is_percentage = true) 
  (h2 : data.total_is_100_percent = true)
  (h3 : data.part_whole_relationship_important = true) : 
  most_appropriate_chart data = ChartType.PieChart :=
by
  sorry

#check pie_chart_most_appropriate_for_percentages

end pie_chart_most_appropriate_for_percentages_l1812_181281


namespace cube_root_of_negative_eight_l1812_181244

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end cube_root_of_negative_eight_l1812_181244


namespace room_area_square_inches_l1812_181252

-- Define the number of inches in a foot
def inches_per_foot : ℕ := 12

-- Define the length of the room in feet
def room_length_feet : ℕ := 10

-- Theorem: The area of the room in square inches is 14400
theorem room_area_square_inches :
  (room_length_feet * inches_per_foot) ^ 2 = 14400 := by
  sorry

end room_area_square_inches_l1812_181252


namespace rationalize_denominator_l1812_181230

theorem rationalize_denominator : 
  (7 / Real.sqrt 98) = Real.sqrt 2 / 2 := by sorry

end rationalize_denominator_l1812_181230


namespace unique_value_of_2n_plus_m_l1812_181298

theorem unique_value_of_2n_plus_m (n m : ℤ) 
  (h1 : 3*n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3*m - 2*n < 46) :
  2*n + m = 36 := by
  sorry

end unique_value_of_2n_plus_m_l1812_181298


namespace number_of_subsets_complement_union_l1812_181231

universe u

def U : Finset ℕ := {1, 3, 5, 7, 9}
def A : Finset ℕ := {1, 5, 9}
def B : Finset ℕ := {3, 5, 9}

theorem number_of_subsets_complement_union : Finset.card (Finset.powerset (U \ (A ∪ B))) = 2 := by
  sorry

end number_of_subsets_complement_union_l1812_181231


namespace hyperbola_eccentricity_l1812_181287

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let line := fun (x : ℝ) => (3/2) * x
  let intersection_projection_is_focus := 
    ∃ (x y : ℝ), hyperbola x y ∧ y = line x ∧ 
    (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ x = c)
  2

#check hyperbola_eccentricity

end hyperbola_eccentricity_l1812_181287


namespace power_inequality_l1812_181202

theorem power_inequality (a b : ℝ) (ha : a > 0) (ha1 : a ≠ 1) (hab : a^b > 1) : a * b > b := by
  sorry

end power_inequality_l1812_181202


namespace angle_measure_from_vector_sum_l1812_181218

/-- Given a triangle ABC with vectors m and n defined in terms of angle A, 
    prove that if the magnitude of their sum is √3, then A = π/3. -/
theorem angle_measure_from_vector_sum (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ A < π →
  m.1 = Real.cos (3 * A / 2) ∧ m.2 = Real.sin (3 * A / 2) →
  n.1 = Real.cos (A / 2) ∧ n.2 = Real.sin (A / 2) →
  Real.sqrt ((m.1 + n.1)^2 + (m.2 + n.2)^2) = Real.sqrt 3 →
  A = π / 3 := by
  sorry

end angle_measure_from_vector_sum_l1812_181218


namespace original_number_proof_l1812_181247

theorem original_number_proof :
  ∃! x : ℕ, (x + 2) % 17 = 0 ∧ x < 17 :=
by
  -- The proof goes here
  sorry

end original_number_proof_l1812_181247


namespace fraction_subtraction_l1812_181253

theorem fraction_subtraction : (18 : ℚ) / 42 - 2 / 9 = 13 / 63 := by
  sorry

end fraction_subtraction_l1812_181253


namespace cookie_distribution_l1812_181207

theorem cookie_distribution (total_cookies : ℚ) (blue_green_fraction : ℚ) (green_ratio : ℚ) :
  blue_green_fraction = 2/3 ∧ 
  green_ratio = 5/9 → 
  ∃ blue_fraction : ℚ, blue_fraction = 8/27 ∧ 
    blue_fraction + (blue_green_fraction - blue_fraction) + (1 - blue_green_fraction) = 1 ∧
    (blue_green_fraction - blue_fraction) / blue_green_fraction = green_ratio :=
by sorry

end cookie_distribution_l1812_181207


namespace notebooks_given_to_tom_l1812_181208

def bernard_notebooks (red blue white remaining : ℕ) : Prop :=
  red + blue + white - remaining = 46

theorem notebooks_given_to_tom :
  bernard_notebooks 15 17 19 5 := by
  sorry

end notebooks_given_to_tom_l1812_181208


namespace max_angle_on_circle_l1812_181227

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the angle between three points
def angle (p1 p2 p3 : Point) : ℝ := sorry

-- Define a function to check if a point is on a circle
def isOnCircle (c : Circle) (p : Point) : Prop := sorry

-- Theorem statement
theorem max_angle_on_circle (c : Circle) (A : Point) :
  ∃ M : Point, isOnCircle c M ∧
  (∀ N : Point, isOnCircle c N → angle c.center M A ≥ angle c.center N A) ∧
  angle c.center A M = π / 2 := by
  sorry

end max_angle_on_circle_l1812_181227


namespace cos_pi_plus_2alpha_l1812_181257

theorem cos_pi_plus_2alpha (α : Real) 
  (h : Real.sin (Real.pi / 2 - α) = 1 / 3) : 
  Real.cos (Real.pi + 2 * α) = 7 / 9 := by
  sorry

end cos_pi_plus_2alpha_l1812_181257


namespace parabola_zeros_difference_l1812_181270

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_zeros_difference (p : Parabola) :
  p.y_coord 3 = -9 →   -- vertex condition
  p.y_coord 6 = 27 →   -- point condition
  ∃ (x1 x2 : ℝ), 
    p.y_coord x1 = 0 ∧ 
    p.y_coord x2 = 0 ∧ 
    |x1 - x2| = 3 :=
by sorry

end parabola_zeros_difference_l1812_181270


namespace difference_nonnegative_equivalence_l1812_181267

theorem difference_nonnegative_equivalence (x : ℝ) :
  (x - 8 ≥ 0) ↔ (∃ (y : ℝ), y ≥ 0 ∧ x - 8 = y) :=
by sorry

end difference_nonnegative_equivalence_l1812_181267


namespace three_times_m_minus_n_squared_l1812_181229

/-- Expresses "3 times m minus n squared" in algebraic notation -/
theorem three_times_m_minus_n_squared (m n : ℝ) : 
  (3 * m - n^2 : ℝ) = (3*m - n)^2 := by sorry

end three_times_m_minus_n_squared_l1812_181229


namespace sqrt_seven_minus_fraction_inequality_l1812_181201

theorem sqrt_seven_minus_fraction_inequality (m n : ℕ) (h : Real.sqrt 7 - (m : ℝ) / n > 0) :
  Real.sqrt 7 - (m : ℝ) / n > 1 / (m * n) := by
  sorry

end sqrt_seven_minus_fraction_inequality_l1812_181201


namespace quadratic_range_l1812_181220

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

-- Theorem statement
theorem quadratic_range :
  Set.range f = {y : ℝ | y ≥ 4} :=
sorry

end quadratic_range_l1812_181220


namespace arcsin_one_half_l1812_181294

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end arcsin_one_half_l1812_181294


namespace evaluate_expression_l1812_181292

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) : y * (y - 2 * x + 1) = 4 := by
  sorry

end evaluate_expression_l1812_181292


namespace auction_bid_ratio_l1812_181262

/-- Auction bidding problem --/
theorem auction_bid_ratio :
  let start_price : ℕ := 300
  let harry_first_bid : ℕ := start_price + 200
  let second_bid : ℕ := 2 * harry_first_bid
  let harry_final_bid : ℕ := 4000
  let third_bid : ℕ := harry_final_bid - 1500
  (third_bid : ℚ) / harry_first_bid = 5 := by sorry

end auction_bid_ratio_l1812_181262


namespace sample_mean_inequality_l1812_181271

theorem sample_mean_inequality (n m : ℕ) (x_bar y_bar z_bar α : ℝ) :
  x_bar ≠ y_bar →
  0 < α →
  α < 1 / 2 →
  z_bar = α * x_bar + (1 - α) * y_bar →
  z_bar = (n * x_bar + m * y_bar) / (n + m) →
  n < m :=
by sorry

end sample_mean_inequality_l1812_181271


namespace future_age_relationship_l1812_181235

/-- Represents the current ages and future relationship between Rehana, Jacob, and Phoebe -/
theorem future_age_relationship (x : ℕ) : 
  let rehana_current_age : ℕ := 25
  let jacob_current_age : ℕ := 3
  let phoebe_current_age : ℕ := jacob_current_age * 5 / 3
  x = 5 ↔ 
    rehana_current_age + x = 3 * (phoebe_current_age + x) ∧
    x ≥ 0 := by
  sorry

end future_age_relationship_l1812_181235


namespace simplify_expression_1_simplify_expression_2_l1812_181200

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : (x + 2) * (x - 1) - 3 * x * (x + 3) = -2 * x^2 - 8 * x - 2 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : (a + 3) * (a^2 + 9) * (a - 3) = a^4 - 81 := by
  sorry

end simplify_expression_1_simplify_expression_2_l1812_181200


namespace trajectory_of_Q_l1812_181263

/-- The trajectory of point Q given the conditions of the problem -/
theorem trajectory_of_Q (A P Q : ℝ × ℝ) : 
  A = (4, 0) →
  (P.1^2 + P.2^2 = 4) →
  (Q.1 - A.1, Q.2 - A.2) = (2*(P.1 - Q.1), 2*(P.2 - Q.2)) →
  (Q.1 - 4/3)^2 + Q.2^2 = 16/9 :=
by sorry

end trajectory_of_Q_l1812_181263


namespace ball_placement_count_is_144_l1812_181280

/-- The number of ways to place four different balls into four numbered boxes with exactly one box remaining empty -/
def ballPlacementCount : ℕ := 144

/-- The number of different balls -/
def numBalls : ℕ := 4

/-- The number of boxes -/
def numBoxes : ℕ := 4

/-- Theorem stating that the number of ways to place four different balls into four numbered boxes with exactly one box remaining empty is 144 -/
theorem ball_placement_count_is_144 : ballPlacementCount = 144 := by sorry

end ball_placement_count_is_144_l1812_181280


namespace merry_sunday_boxes_l1812_181265

/-- Represents the number of apples in each box -/
def apples_per_box : ℕ := 10

/-- Represents the number of boxes Merry had on Saturday -/
def saturday_boxes : ℕ := 50

/-- Represents the total number of apples sold on Saturday and Sunday -/
def total_apples_sold : ℕ := 720

/-- Represents the number of boxes left after selling -/
def boxes_left : ℕ := 3

/-- Represents the number of boxes Merry had on Sunday -/
def sunday_boxes : ℕ := 25

theorem merry_sunday_boxes :
  sunday_boxes = 25 :=
by sorry

end merry_sunday_boxes_l1812_181265


namespace unique_prime_in_set_l1812_181228

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def six_digit_number (A : ℕ) : ℕ := 303100 + A

theorem unique_prime_in_set : 
  ∃! A : ℕ, A < 10 ∧ is_prime (six_digit_number A) ∧ six_digit_number A = 303103 :=
sorry

end unique_prime_in_set_l1812_181228


namespace parallel_lines_k_value_l1812_181254

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

/-- The first line: (k-1)x + y + 2 = 0 -/
def line1 (k x y : ℝ) : Prop :=
  (k - 1) * x + y + 2 = 0

/-- The second line: 8x + (k+1)y + k - 1 = 0 -/
def line2 (k x y : ℝ) : Prop :=
  8 * x + (k + 1) * y + k - 1 = 0

theorem parallel_lines_k_value (k : ℝ) :
  (∀ x y : ℝ, are_parallel (k - 1) 1 8 (k + 1)) →
  k = 3 :=
sorry

end parallel_lines_k_value_l1812_181254


namespace sphere_to_hemisphere_volume_ratio_l1812_181275

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio (p : ℝ) (p_pos : p > 0) :
  (4 / 3 * Real.pi * p ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p) ^ 3) = 2 / 27 := by
  sorry

end sphere_to_hemisphere_volume_ratio_l1812_181275


namespace circumscribing_sphere_surface_area_l1812_181222

/-- A right triangular rectangular pyramid with side length a -/
structure RightTriangularRectangularPyramid where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A sphere containing the vertices of a right triangular rectangular pyramid -/
structure CircumscribingSphere (p : RightTriangularRectangularPyramid) where
  radius : ℝ
  radius_pos : radius > 0

/-- The theorem stating the surface area of the circumscribing sphere -/
theorem circumscribing_sphere_surface_area
  (p : RightTriangularRectangularPyramid)
  (s : CircumscribingSphere p) :
  4 * Real.pi * s.radius ^ 2 = 3 * Real.pi * p.side_length ^ 2 := by
  sorry

end circumscribing_sphere_surface_area_l1812_181222


namespace flag_design_count_l1812_181266

/-- Represents the number of available colors for the flag stripes -/
def num_colors : ℕ := 3

/-- Represents the number of stripes in the flag -/
def num_stripes : ℕ := 3

/-- Calculates the number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem: The number of unique three-stripe flags that can be created
    using three colors, where adjacent stripes may be the same color, is 27 -/
theorem flag_design_count :
  num_flag_designs = 27 := by sorry

end flag_design_count_l1812_181266


namespace monotonic_increasing_interval_l1812_181223

open Real

noncomputable def f (ω x : ℝ) : ℝ := Real.sqrt 3 * sin (ω * x) + cos (ω * x)

theorem monotonic_increasing_interval
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (α β : ℝ)
  (h_f_α : f ω α = 2)
  (h_f_β : f ω β = 0)
  (h_min_diff : |α - β| = π / 2) :
  ∃ k : ℤ, StrictMonoOn f (Set.Icc (2 * k * π - 2 * π / 3) (2 * k * π + π / 3)) :=
by sorry

end monotonic_increasing_interval_l1812_181223


namespace triangle_sine_inequality_l1812_181279

theorem triangle_sine_inequality (α β γ : Real) (h : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) :
  (Real.sin α + Real.sin β + Real.sin γ)^2 > 9 * Real.sin α * Real.sin β * Real.sin γ := by
  sorry

end triangle_sine_inequality_l1812_181279


namespace min_value_reciprocal_sum_l1812_181282

/-- Given a data set (2, 4, 6, 8) with median m and variance n, 
    and the equation ma + nb = 1 where a > 0 and b > 0,
    prove that the minimum value of 1/a + 1/b is 20. -/
theorem min_value_reciprocal_sum (m n a b : ℝ) : 
  m = 5 → 
  n = 5 → 
  m * a + n * b = 1 → 
  a > 0 → 
  b > 0 → 
  (1 / a + 1 / b) ≥ 20 := by
  sorry

end min_value_reciprocal_sum_l1812_181282


namespace min_value_cos_squared_plus_sin_l1812_181289

theorem min_value_cos_squared_plus_sin (f : ℝ → ℝ) :
  (∀ x, -π/4 ≤ x ∧ x ≤ π/4 → f x = Real.cos x ^ 2 + Real.sin x) →
  ∃ x₀, -π/4 ≤ x₀ ∧ x₀ ≤ π/4 ∧ f x₀ = (1 - Real.sqrt 2) / 2 ∧
  ∀ x, -π/4 ≤ x ∧ x ≤ π/4 → f x₀ ≤ f x :=
by sorry

end min_value_cos_squared_plus_sin_l1812_181289


namespace arithmetic_sequence_properties_l1812_181274

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) : ℕ+ → ℤ := fun n => a₁ + d * (n - 1)

theorem arithmetic_sequence_properties
  (a : ℕ+ → ℤ)
  (h₁ : a 1 = -60)
  (h₂ : a 17 = -12) :
  let d := (a 17 - a 1) / 16
  ∃ (S T : ℕ+ → ℤ),
    (∀ n : ℕ+, a n = arithmetic_sequence (-60) d n) ∧
    (∀ n : ℕ+, n < 22 → a n ≤ 0) ∧
    (a 22 > 0) ∧
    (∀ n : ℕ+, S n = n * (a 1 + a n) / 2) ∧
    (S 20 = S 21) ∧
    (S 20 = -630) ∧
    (∀ n : ℕ+, n ≤ 21 → T n = n * (123 - 3 * n) / 2) ∧
    (∀ n : ℕ+, n ≥ 22 → T n = (3 * n^2 - 123 * n + 2520) / 2) :=
by sorry

end arithmetic_sequence_properties_l1812_181274


namespace linear_function_point_range_l1812_181284

theorem linear_function_point_range (x y : ℝ) : 
  y = 4 - 3 * x → y > -5 → x < 3 := by sorry

end linear_function_point_range_l1812_181284


namespace range_of_a_l1812_181219

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 2

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) → 
  a ≥ 5 ∨ a ≤ 1 := by sorry

end range_of_a_l1812_181219
