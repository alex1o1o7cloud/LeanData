import Mathlib

namespace NUMINAMATH_CALUDE_initial_distance_between_cars_l2273_227361

theorem initial_distance_between_cars (speed_A speed_B time_to_overtake distance_ahead : ℝ) 
  (h1 : speed_A = 58)
  (h2 : speed_B = 50)
  (h3 : time_to_overtake = 4.75)
  (h4 : distance_ahead = 8) : 
  (speed_A - speed_B) * time_to_overtake = 30 + distance_ahead := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_between_cars_l2273_227361


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l2273_227359

theorem geometric_sequence_second_term 
  (a₁ : ℝ) 
  (a₃ : ℝ) 
  (b : ℝ) 
  (h₁ : a₁ = 120) 
  (h₂ : a₃ = 64 / 30) 
  (h₃ : b > 0) 
  (h₄ : ∃ r : ℝ, a₁ * r = b ∧ b * r = a₃) : 
  b = 16 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_second_term_l2273_227359


namespace NUMINAMATH_CALUDE_magazine_boxes_l2273_227307

theorem magazine_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_magazine_boxes_l2273_227307


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2273_227325

/-- Given two functions f and g, prove that |k| ≤ 2 is a sufficient but not necessary condition
for f(x) ≥ g(x) to hold for all x ∈ ℝ. -/
theorem sufficient_not_necessary_condition (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + 3 ≥ k*x - 1) ↔ -6 ≤ k ∧ k ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2273_227325


namespace NUMINAMATH_CALUDE_polar_bear_fish_consumption_l2273_227321

/-- The amount of trout eaten daily by the polar bear in buckets -/
def trout_amount : ℝ := 0.2

/-- The amount of salmon eaten daily by the polar bear in buckets -/
def salmon_amount : ℝ := 0.4

/-- The total amount of fish eaten daily by the polar bear in buckets -/
def total_fish : ℝ := trout_amount + salmon_amount

theorem polar_bear_fish_consumption :
  total_fish = 0.6 := by sorry

end NUMINAMATH_CALUDE_polar_bear_fish_consumption_l2273_227321


namespace NUMINAMATH_CALUDE_sector_radius_l2273_227392

theorem sector_radius (α : Real) (S : Real) (r : Real) : 
  α = 3/4 * Real.pi → 
  S = 3/2 * Real.pi → 
  S = 1/2 * r^2 * α → 
  r = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l2273_227392


namespace NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l2273_227380

/-- The percentage of chromium in the new alloy formed by melting two alloys -/
theorem chromium_percentage_in_new_alloy
  (chromium_percent1 : ℝ) (chromium_percent2 : ℝ)
  (weight1 : ℝ) (weight2 : ℝ)
  (h1 : chromium_percent1 = 12)
  (h2 : chromium_percent2 = 8)
  (h3 : weight1 = 15)
  (h4 : weight2 = 35) :
  let total_chromium := (chromium_percent1 / 100) * weight1 + (chromium_percent2 / 100) * weight2
  let total_weight := weight1 + weight2
  (total_chromium / total_weight) * 100 = 9.2 := by
sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l2273_227380


namespace NUMINAMATH_CALUDE_product_even_even_is_even_product_odd_odd_is_odd_product_even_odd_is_even_product_odd_even_is_even_l2273_227342

-- Define even and odd integers
def IsEven (n : Int) : Prop := ∃ k : Int, n = 2 * k
def IsOdd (n : Int) : Prop := ∃ k : Int, n = 2 * k + 1

-- Theorem statements
theorem product_even_even_is_even (a b : Int) (ha : IsEven a) (hb : IsEven b) :
  IsEven (a * b) := by sorry

theorem product_odd_odd_is_odd (a b : Int) (ha : IsOdd a) (hb : IsOdd b) :
  IsOdd (a * b) := by sorry

theorem product_even_odd_is_even (a b : Int) (ha : IsEven a) (hb : IsOdd b) :
  IsEven (a * b) := by sorry

theorem product_odd_even_is_even (a b : Int) (ha : IsOdd a) (hb : IsEven b) :
  IsEven (a * b) := by sorry

end NUMINAMATH_CALUDE_product_even_even_is_even_product_odd_odd_is_odd_product_even_odd_is_even_product_odd_even_is_even_l2273_227342


namespace NUMINAMATH_CALUDE_special_set_average_l2273_227398

/-- A finite set of positive integers satisfying specific conditions -/
def SpecialSet (T : Finset ℕ) : Prop :=
  ∃ (m : ℕ), m > 1 ∧ 
  (∃ (b₁ bₘ : ℕ), b₁ ∈ T ∧ bₘ ∈ T ∧
    (∀ x ∈ T, b₁ ≤ x ∧ x ≤ bₘ) ∧
    bₘ = b₁ + 50 ∧
    (T.sum id - bₘ) / (m - 1) = 45 ∧
    (T.sum id - b₁ - bₘ) / (m - 2) = 50 ∧
    (T.sum id - b₁) / (m - 1) = 55)

/-- The average of all integers in a SpecialSet is 50 -/
theorem special_set_average (T : Finset ℕ) (h : SpecialSet T) :
  (T.sum id) / T.card = 50 := by
  sorry

end NUMINAMATH_CALUDE_special_set_average_l2273_227398


namespace NUMINAMATH_CALUDE_elevator_occupancy_l2273_227313

/-- Proves that the total number of people in the elevator is 7 after a new person enters --/
theorem elevator_occupancy (initial_people : ℕ) (initial_avg_weight : ℝ) (new_avg_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 160 →
  new_avg_weight = 151 →
  initial_people + 1 = 7 :=
by sorry

end NUMINAMATH_CALUDE_elevator_occupancy_l2273_227313


namespace NUMINAMATH_CALUDE_arithmetic_progression_equality_l2273_227389

theorem arithmetic_progression_equality (n : ℕ) 
  (a b : Fin n → ℕ+) 
  (h_n : n ≥ 2018)
  (h_distinct : ∀ i j : Fin n, i ≠ j → (a i ≠ a j ∧ b i ≠ b j))
  (h_bound : ∀ i : Fin n, a i ≤ 5*n ∧ b i ≤ 5*n)
  (h_arithmetic : ∃ d : ℚ, ∀ i j : Fin n, (a j : ℚ) / (b j : ℚ) - (a i : ℚ) / (b i : ℚ) = (j : ℚ) - (i : ℚ) * d) :
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equality_l2273_227389


namespace NUMINAMATH_CALUDE_probability_multiple_of_three_in_eight_rolls_l2273_227314

theorem probability_multiple_of_three_in_eight_rolls : 
  let p : ℚ := 1 - (2/3)^8
  p = 6305/6561 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_three_in_eight_rolls_l2273_227314


namespace NUMINAMATH_CALUDE_triangle_theorem_l2273_227324

noncomputable section

theorem triangle_theorem (a b c A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin A = 4 * b * Real.sin B →
  a * c = Real.sqrt 5 * (a^2 - b^2 - c^2) →
  Real.cos A = -Real.sqrt 5 / 5 ∧
  Real.sin (2 * B - A) = -2 * Real.sqrt 5 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l2273_227324


namespace NUMINAMATH_CALUDE_problem_solution_l2273_227347

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 4*x + 4/x + 1/x^2 = 35)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 23 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2273_227347


namespace NUMINAMATH_CALUDE_petals_per_rose_correct_petals_per_rose_l2273_227317

theorem petals_per_rose (petals_per_ounce : ℕ) (roses_per_bush : ℕ) (bushes_harvested : ℕ) 
  (bottles_produced : ℕ) (ounces_per_bottle : ℕ) : ℕ :=
  let total_ounces := bottles_produced * ounces_per_bottle
  let total_petals := total_ounces * petals_per_ounce
  let petals_per_bush := total_petals / bushes_harvested
  petals_per_bush / roses_per_bush

theorem correct_petals_per_rose :
  petals_per_rose 320 12 800 20 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_petals_per_rose_correct_petals_per_rose_l2273_227317


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l2273_227316

theorem sum_of_cubes_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -7) : a^3 + b^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l2273_227316


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l2273_227358

/-- 
Given a person swimming against a current, prove their swimming speed in still water.
-/
theorem swimming_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 2) 
  (h2 : distance = 6) 
  (h3 : time = 3) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 4 ∧ 
    distance = (still_water_speed - current_speed) * time := by
  sorry

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l2273_227358


namespace NUMINAMATH_CALUDE_max_xy_value_l2273_227337

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 18) :
  x * y ≤ 81 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 18 ∧ x * y = 81 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l2273_227337


namespace NUMINAMATH_CALUDE_alok_ice_cream_order_l2273_227378

/-- The number of ice-cream cups ordered by Alok -/
def ice_cream_cups (chapatis rice mixed_veg : ℕ) 
  (chapati_cost rice_cost mixed_veg_cost ice_cream_cost total_paid : ℕ) : ℕ :=
  (total_paid - (chapatis * chapati_cost + rice * rice_cost + mixed_veg * mixed_veg_cost)) / ice_cream_cost

/-- Theorem stating that Alok ordered 6 ice-cream cups -/
theorem alok_ice_cream_order : 
  ice_cream_cups 16 5 7 6 45 70 40 1051 = 6 := by
  sorry

end NUMINAMATH_CALUDE_alok_ice_cream_order_l2273_227378


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l2273_227376

/-- The volume of a rectangular parallelepiped with diagonal d, which forms angles of 60° and 45° with two of its edges, is equal to d³√2 / 8 -/
theorem parallelepiped_volume (d : ℝ) (h_d_pos : d > 0) : ∃ (V : ℝ),
  V = d^3 * Real.sqrt 2 / 8 ∧
  ∃ (a b h : ℝ),
    a > 0 ∧ b > 0 ∧ h > 0 ∧
    V = a * b * h ∧
    d^2 = a^2 + b^2 + h^2 ∧
    a / d = Real.cos (π / 4) ∧
    b / d = Real.cos (π / 3) :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l2273_227376


namespace NUMINAMATH_CALUDE_alternative_rate_calculation_l2273_227310

/-- Calculates simple interest -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time

theorem alternative_rate_calculation (principal : ℚ) (time : ℚ) (actual_rate : ℚ) 
  (interest_difference : ℚ) (alternative_rate : ℚ) : 
  principal = 2500 →
  time = 2 →
  actual_rate = 18 / 100 →
  interest_difference = 300 →
  simple_interest principal actual_rate time - simple_interest principal alternative_rate time = interest_difference →
  alternative_rate = 12 / 100 := by
sorry

end NUMINAMATH_CALUDE_alternative_rate_calculation_l2273_227310


namespace NUMINAMATH_CALUDE_parking_lot_spaces_l2273_227315

/-- The number of spaces a single caravan occupies -/
def spaces_per_caravan : ℕ := 2

/-- The number of caravans currently parked -/
def number_of_caravans : ℕ := 3

/-- The number of spaces left for other vehicles -/
def spaces_left : ℕ := 24

/-- The total number of spaces in the parking lot -/
def total_spaces : ℕ := spaces_per_caravan * number_of_caravans + spaces_left

theorem parking_lot_spaces : total_spaces = 30 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_spaces_l2273_227315


namespace NUMINAMATH_CALUDE_divisibility_by_17_l2273_227362

theorem divisibility_by_17 (x y : ℤ) : 
  (∃ k : ℤ, 2*x + 3*y = 17*k) → (∃ m : ℤ, 9*x + 5*y = 17*m) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_17_l2273_227362


namespace NUMINAMATH_CALUDE_profit_share_difference_is_370_l2273_227356

/-- Calculates the difference between Rose's and Tom's share in the profit --/
def profit_share_difference (john_investment : ℕ) (john_duration : ℕ) 
                            (rose_investment : ℕ) (rose_duration : ℕ) 
                            (tom_investment : ℕ) (tom_duration : ℕ) 
                            (total_profit : ℕ) : ℕ :=
  let john_investment_months := john_investment * john_duration
  let rose_investment_months := rose_investment * rose_duration
  let tom_investment_months := tom_investment * tom_duration
  let total_investment_months := john_investment_months + rose_investment_months + tom_investment_months
  let rose_share := (rose_investment_months * total_profit) / total_investment_months
  let tom_share := (tom_investment_months * total_profit) / total_investment_months
  rose_share - tom_share

/-- Theorem stating that the difference between Rose's and Tom's profit share is 370 --/
theorem profit_share_difference_is_370 :
  profit_share_difference 18000 12 12000 9 9000 8 4070 = 370 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_is_370_l2273_227356


namespace NUMINAMATH_CALUDE_marks_fish_count_l2273_227354

/-- Calculates the total number of young fish given the number of tanks, 
    pregnant fish per tank, and young per fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Proves that given 3 tanks, 4 pregnant fish per tank, and 20 young per fish, 
    the total number of young fish is equal to 240. -/
theorem marks_fish_count : total_young_fish 3 4 20 = 240 := by
  sorry

end NUMINAMATH_CALUDE_marks_fish_count_l2273_227354


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l2273_227305

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) :
  (1/a + 1/b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    Real.sqrt 3 = Real.sqrt (3^a₀ * 3^b₀) ∧ 1/a₀ + 1/b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l2273_227305


namespace NUMINAMATH_CALUDE_hexagon_diagonal_area_bound_l2273_227377

/-- A convex hexagon is a six-sided polygon where all interior angles are less than or equal to 180 degrees. -/
structure ConvexHexagon where
  -- We assume the existence of a convex hexagon without explicitly defining its properties
  -- as the specific geometric representation is not crucial for this theorem.

/-- The theorem states that for any convex hexagon, there exists a diagonal that cuts off a triangle
    with an area less than or equal to one-sixth of the total area of the hexagon. -/
theorem hexagon_diagonal_area_bound (h : ConvexHexagon) (S : ℝ) (h_area : S > 0) :
  ∃ (triangle_area : ℝ), triangle_area ≤ S / 6 ∧ triangle_area > 0 := by
  sorry


end NUMINAMATH_CALUDE_hexagon_diagonal_area_bound_l2273_227377


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l2273_227350

theorem complex_modulus_equation (a : ℝ) : 
  Complex.abs ((5 : ℂ) / (2 + Complex.I) + a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l2273_227350


namespace NUMINAMATH_CALUDE_part_one_part_two_l2273_227339

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x ≤ 10}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Part 1
theorem part_one : 
  M ∩ (Set.univ \ N 2) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, M ∪ N a = M → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2273_227339


namespace NUMINAMATH_CALUDE_specific_sandwich_calories_l2273_227365

/-- Represents a sandwich with bacon strips -/
structure BaconSandwich where
  bacon_strips : ℕ
  calories_per_strip : ℕ
  bacon_percentage : ℚ

/-- Calculates the total calories of a bacon sandwich -/
def total_calories (s : BaconSandwich) : ℚ :=
  (s.bacon_strips * s.calories_per_strip : ℚ) / s.bacon_percentage

/-- Theorem stating the total calories of the specific sandwich -/
theorem specific_sandwich_calories :
  let s : BaconSandwich := {
    bacon_strips := 2,
    calories_per_strip := 125,
    bacon_percentage := 1/5
  }
  total_calories s = 1250 := by sorry

end NUMINAMATH_CALUDE_specific_sandwich_calories_l2273_227365


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2273_227399

/-- The polar equation r = 1 / (sin θ + cos θ) represents a circle in Cartesian coordinates -/
theorem polar_to_cartesian_circle :
  ∃ (h k r : ℝ), ∀ (x y : ℝ),
    (∃ (θ : ℝ), x = (1 / (Real.sin θ + Real.cos θ)) * Real.cos θ ∧
                 y = (1 / (Real.sin θ + Real.cos θ)) * Real.sin θ) →
    (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2273_227399


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2273_227369

/-- Given an ellipse with equation x^2 + ky^2 = 2, foci on the x-axis, and focal distance √3, 
    its minor axis length is √5 -/
theorem ellipse_minor_axis_length (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*y^2 = 2) →  -- Equation of the ellipse
  (∃ c : ℝ, c^2 = 3 ∧ 
    ∀ x y : ℝ, x^2 + k*y^2 = 2 → 
      (x - c)^2 + y^2 = (x + c)^2 + y^2) →  -- Foci on x-axis with distance √3
  ∃ b : ℝ, b^2 = 5 ∧ 
    ∀ x y : ℝ, x^2 + k*y^2 = 2 → 
      y^2 ≤ b^2/4 :=  -- Minor axis length is √5
by sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2273_227369


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2273_227355

/-- Rationalize the denominator of (2 + √5) / (3 - √5) -/
theorem rationalize_denominator :
  ∃ (A B : ℚ) (C : ℕ), 
    (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧
    A = 11 / 4 ∧
    B = 5 / 4 ∧
    C = 5 ∧
    A * B * C = 275 / 16 := by
  sorry

#check rationalize_denominator

end NUMINAMATH_CALUDE_rationalize_denominator_l2273_227355


namespace NUMINAMATH_CALUDE_cubic_extrema_opposite_signs_l2273_227329

/-- A cubic function with coefficients p and q -/
def cubic_function (p q : ℝ) (x : ℝ) : ℝ := x^3 + p*x + q

/-- The derivative of the cubic function -/
def cubic_derivative (p : ℝ) (x : ℝ) : ℝ := 3*x^2 + p

/-- Condition for opposite signs of extremum points -/
def opposite_signs_condition (p q : ℝ) : Prop := 
  (q/2)^2 + (p/3)^3 < 0 ∧ p < 0

theorem cubic_extrema_opposite_signs 
  (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    cubic_derivative p x₁ = 0 ∧ 
    cubic_derivative p x₂ = 0 ∧ 
    cubic_function p q x₁ * cubic_function p q x₂ < 0) ↔ 
  opposite_signs_condition p q :=
sorry

end NUMINAMATH_CALUDE_cubic_extrema_opposite_signs_l2273_227329


namespace NUMINAMATH_CALUDE_exists_function_and_constant_l2273_227381

theorem exists_function_and_constant : 
  ∃ (f : ℝ → ℝ) (a : ℝ), 
    a ∈ Set.Icc 0 π ∧ 
    (∀ x : ℝ, (1 + Real.sqrt 2 * Real.sin x) * (1 + Real.sqrt 2 * Real.sin (x + π)) = Real.cos (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_exists_function_and_constant_l2273_227381


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l2273_227382

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 15) :
  Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l2273_227382


namespace NUMINAMATH_CALUDE_revenue_growth_equation_l2273_227394

/-- Represents the average monthly growth rate of revenue -/
def x : ℝ := sorry

/-- Represents the revenue in January in thousands of dollars -/
def january_revenue : ℝ := 36

/-- Represents the revenue in March in thousands of dollars -/
def march_revenue : ℝ := 48

/-- Theorem stating that the equation representing the revenue growth is 36(1+x)^2 = 48 -/
theorem revenue_growth_equation : 
  january_revenue * (1 + x)^2 = march_revenue := by sorry

end NUMINAMATH_CALUDE_revenue_growth_equation_l2273_227394


namespace NUMINAMATH_CALUDE_birds_total_distance_l2273_227338

/-- Calculates the total distance flown by six birds given their speeds and flight times -/
def total_distance_flown (eagle_speed falcon_speed pelican_speed hummingbird_speed hawk_speed swallow_speed : ℝ)
  (eagle_time falcon_time pelican_time hummingbird_time hawk_time swallow_time : ℝ) : ℝ :=
  eagle_speed * eagle_time +
  falcon_speed * falcon_time +
  pelican_speed * pelican_time +
  hummingbird_speed * hummingbird_time +
  hawk_speed * hawk_time +
  swallow_speed * swallow_time

/-- The total distance flown by all birds is 482.5 miles -/
theorem birds_total_distance :
  total_distance_flown 15 46 33 30 45 25 2.5 2.5 2.5 2.5 3 1.5 = 482.5 := by
  sorry

end NUMINAMATH_CALUDE_birds_total_distance_l2273_227338


namespace NUMINAMATH_CALUDE_inequality_proof_l2273_227303

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (|x^2 + y^2|) / (x + y) < (|x^2 - y^2|) / (x - y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2273_227303


namespace NUMINAMATH_CALUDE_inequalities_for_ordered_reals_l2273_227391

theorem inequalities_for_ordered_reals 
  (a b c d : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 0 > c) 
  (h4 : c > d) : 
  (a + c > b + d) ∧ 
  (a * d^2 > b * c^2) ∧ 
  ((1 : ℝ) / (b * c) < (1 : ℝ) / (a * d)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_ordered_reals_l2273_227391


namespace NUMINAMATH_CALUDE_obtuse_triangle_one_obtuse_angle_l2273_227327

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define an obtuse triangle
def ObtuseTriangle (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i > 90

-- Define an obtuse angle
def ObtuseAngle (angle : ℝ) : Prop := angle > 90

-- Theorem: An obtuse triangle has exactly one obtuse interior angle
theorem obtuse_triangle_one_obtuse_angle (t : Triangle) (h : ObtuseTriangle t) :
  ∃! i : Fin 3, ObtuseAngle (t.angles i) :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_one_obtuse_angle_l2273_227327


namespace NUMINAMATH_CALUDE_evaluate_expression_l2273_227388

theorem evaluate_expression : 
  Real.sqrt (9/4) - Real.sqrt (4/9) + (Real.sqrt (9/4) + Real.sqrt (4/9))^2 = 199/36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2273_227388


namespace NUMINAMATH_CALUDE_malcolm_route_ratio_l2273_227351

/-- Malcolm's route to school problem -/
theorem malcolm_route_ratio : 
  ∀ (r : ℝ), 
  (6 + 6*r + (1/3)*(6 + 6*r) + 18 = 42) → 
  r = 17/4 := by
sorry

end NUMINAMATH_CALUDE_malcolm_route_ratio_l2273_227351


namespace NUMINAMATH_CALUDE_tenth_term_value_l2273_227390

theorem tenth_term_value (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n : ℕ, S n = n * (2 * n + 1)) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  a 10 = 39 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_value_l2273_227390


namespace NUMINAMATH_CALUDE_school_population_l2273_227357

theorem school_population (total_girls : ℕ) (difference : ℕ) (total_boys : ℕ) : 
  total_girls = 697 → difference = 228 → total_girls - total_boys = difference → total_boys = 469 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l2273_227357


namespace NUMINAMATH_CALUDE_ice_cream_scoops_l2273_227319

/-- The number of scoops in a single cone of ice cream -/
def single_cone_scoops : ℕ := sorry

/-- The number of scoops in a banana split -/
def banana_split_scoops : ℕ := 3 * single_cone_scoops

/-- The number of scoops in a waffle bowl -/
def waffle_bowl_scoops : ℕ := banana_split_scoops + 1

/-- The number of scoops in a double cone -/
def double_cone_scoops : ℕ := 2 * single_cone_scoops

/-- The total number of scoops served -/
def total_scoops : ℕ := 10

theorem ice_cream_scoops : 
  single_cone_scoops + banana_split_scoops + waffle_bowl_scoops + double_cone_scoops = total_scoops ∧
  single_cone_scoops = 1 := by sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_l2273_227319


namespace NUMINAMATH_CALUDE_geometric_inequalities_l2273_227393

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a point inside a triangle
def InsideTriangle (t : Triangle) (D : ℝ × ℝ) : Prop := sorry

-- Define a point inside a convex quadrilateral
def InsideConvexQuadrilateral (q : Quadrilateral) (E : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex A of a triangle
def angle_A (t : Triangle) : ℝ := sorry

-- Define the ratio k
def ratio_k (q : Quadrilateral) (E : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem geometric_inequalities 
  (t : Triangle) 
  (D : ℝ × ℝ) 
  (q : Quadrilateral) 
  (E : ℝ × ℝ) 
  (h1 : InsideTriangle t D) 
  (h2 : InsideConvexQuadrilateral q E) : 
  (distance t.B t.C / min (distance t.A D) (min (distance t.B D) (distance t.C D)) ≥ 
    if angle_A t < π/2 then 2 * Real.sin (angle_A t) else 2) ∧
  (ratio_k q E ≥ 2 * Real.sin (70 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_inequalities_l2273_227393


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l2273_227311

theorem smallest_integer_satisfying_conditions :
  ∃ x : ℤ, (3 * |x| + 4 < 25) ∧ (x + 3 > 0) ∧
  (∀ y : ℤ, (3 * |y| + 4 < 25) ∧ (y + 3 > 0) → x ≤ y) ∧
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l2273_227311


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2273_227364

theorem imaginary_part_of_complex_fraction : Complex.im ((3 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2273_227364


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2273_227396

/-- Given a line y = mx + b passing through (0, 3) and reflecting (2, -4) to (-4, 8), prove that m + b = 3.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (3 = m * 0 + b) →  -- Line passes through (0, 3)
  (let midpoint_x := (2 + (-4)) / 2
   let midpoint_y := (-4 + 8) / 2
   (midpoint_y - 3) = m * (midpoint_x - 0)) →  -- Midpoint lies on the line
  (8 - (-4)) / (-4 - 2) = -1 / m →  -- Perpendicular slopes
  m + b = 3.5 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2273_227396


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2273_227375

theorem smallest_integer_with_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧ 
  x % 6 = 5 ∧
  (∀ y : ℕ, y > 0 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 6 = 5 → x ≤ y) ∧
  x = 59 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2273_227375


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_l2273_227367

def p (x : ℝ) : ℝ := 3*x^3 + 2*x^2 + 5*x + 4
def q (x : ℝ) : ℝ := 7*x^3 + 5*x^2 + 6*x + 7

theorem coefficient_x_cubed (x : ℝ) : 
  ∃ (a b c d : ℝ), p x * q x = 38*x^3 + a*x^4 + b*x^2 + c*x + d :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_l2273_227367


namespace NUMINAMATH_CALUDE_middle_number_bounds_l2273_227328

theorem middle_number_bounds (a b c : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : a + b + c = 10) (h4 : a - c = 3) : 7/3 < b ∧ b < 13/3 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_bounds_l2273_227328


namespace NUMINAMATH_CALUDE_molecular_weight_BaCl2_calculation_l2273_227366

/-- The molecular weight of 8 moles of BaCl2 -/
def molecular_weight_BaCl2 (atomic_weight_Ba : ℝ) (atomic_weight_Cl : ℝ) : ℝ :=
  8 * (atomic_weight_Ba + 2 * atomic_weight_Cl)

/-- Theorem stating the molecular weight of 8 moles of BaCl2 -/
theorem molecular_weight_BaCl2_calculation :
  molecular_weight_BaCl2 137.33 35.45 = 1665.84 := by
  sorry

#eval molecular_weight_BaCl2 137.33 35.45

end NUMINAMATH_CALUDE_molecular_weight_BaCl2_calculation_l2273_227366


namespace NUMINAMATH_CALUDE_largest_number_l2273_227334

/-- Represents a real number with a repeating decimal expansion -/
def RepeatingDecimal (whole : ℕ) (nonRepeating : List ℕ) (repeating : List ℕ) : ℚ :=
  sorry

/-- The number 8.12356 -/
def num1 : ℚ := 8.12356

/-- The number 8.123$\overline{5}$ -/
def num2 : ℚ := RepeatingDecimal 8 [1, 2, 3] [5]

/-- The number 8.12$\overline{356}$ -/
def num3 : ℚ := RepeatingDecimal 8 [1, 2] [3, 5, 6]

/-- The number 8.1$\overline{2356}$ -/
def num4 : ℚ := RepeatingDecimal 8 [1] [2, 3, 5, 6]

/-- The number 8.$\overline{12356}$ -/
def num5 : ℚ := RepeatingDecimal 8 [] [1, 2, 3, 5, 6]

theorem largest_number : 
  num2 > num1 ∧ num2 > num3 ∧ num2 > num4 ∧ num2 > num5 :=
sorry

end NUMINAMATH_CALUDE_largest_number_l2273_227334


namespace NUMINAMATH_CALUDE_cookie_distribution_l2273_227386

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) 
  (h1 : total_cookies = 35)
  (h2 : num_people = 5)
  (h3 : total_cookies = num_people * cookies_per_person) :
  cookies_per_person = 7 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l2273_227386


namespace NUMINAMATH_CALUDE_intersection_when_a_zero_subset_condition_l2273_227397

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- Theorem 1: When a = 0, A ∩ B = {x | 0 < x < 1}
theorem intersection_when_a_zero :
  A 0 ∩ B = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: A ⊆ B if and only if 1 ≤ a ≤ 2
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ 1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_zero_subset_condition_l2273_227397


namespace NUMINAMATH_CALUDE_factorization_equality_l2273_227302

theorem factorization_equality (a b : ℝ) : a * b^2 - 8 * a * b + 16 * a = a * (b - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2273_227302


namespace NUMINAMATH_CALUDE_green_blue_difference_after_two_borders_l2273_227352

/-- Calculates the number of tiles in a border of a hexagonal figure -/
def border_tiles (side_length : ℕ) : ℕ := 6 * side_length

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Adds a border of green tiles to a hexagonal figure -/
def add_border (fig : HexFigure) (border_size : ℕ) : HexFigure :=
  { blue_tiles := fig.blue_tiles,
    green_tiles := fig.green_tiles + border_tiles border_size }

theorem green_blue_difference_after_two_borders :
  let initial_figure : HexFigure := { blue_tiles := 14, green_tiles := 8 }
  let first_border := add_border initial_figure 3
  let second_border := add_border first_border 5
  second_border.green_tiles - second_border.blue_tiles = 42 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_after_two_borders_l2273_227352


namespace NUMINAMATH_CALUDE_expected_rainfall_theorem_l2273_227363

-- Define the daily weather probabilities and rainfall amounts
def sunny_prob : ℝ := 0.30
def light_rain_prob : ℝ := 0.20
def heavy_rain_prob : ℝ := 0.25
def cloudy_prob : ℝ := 0.25

def light_rain_amount : ℝ := 5
def heavy_rain_amount : ℝ := 7

def days : ℕ := 7

-- State the theorem
theorem expected_rainfall_theorem :
  let daily_expected_rainfall := 
    light_rain_prob * light_rain_amount + heavy_rain_prob * heavy_rain_amount
  (days : ℝ) * daily_expected_rainfall = 19.25 := by
  sorry


end NUMINAMATH_CALUDE_expected_rainfall_theorem_l2273_227363


namespace NUMINAMATH_CALUDE_intersection_A_B_l2273_227371

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {0, 1, 2, 3, 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2273_227371


namespace NUMINAMATH_CALUDE_lake_superior_depth_l2273_227345

/-- The depth of a lake given its water surface elevation above sea level and lowest point below sea level -/
def lake_depth (water_surface_elevation : ℝ) (lowest_point_below_sea : ℝ) : ℝ :=
  water_surface_elevation + lowest_point_below_sea

/-- Theorem: The depth of Lake Superior at its deepest point is 400 meters -/
theorem lake_superior_depth :
  lake_depth 180 220 = 400 := by
  sorry

end NUMINAMATH_CALUDE_lake_superior_depth_l2273_227345


namespace NUMINAMATH_CALUDE_line_intersects_midpoint_of_segment_l2273_227301

/-- The value of c for which the line 2x + y = c intersects the midpoint of the line segment from (1, 4) to (7, 10) -/
theorem line_intersects_midpoint_of_segment (c : ℝ) : 
  (∃ (x y : ℝ), 2*x + y = c ∧ 
   x = (1 + 7) / 2 ∧ 
   y = (4 + 10) / 2) → 
  c = 15 := by
sorry


end NUMINAMATH_CALUDE_line_intersects_midpoint_of_segment_l2273_227301


namespace NUMINAMATH_CALUDE_solve_equation_l2273_227372

/-- The original equation --/
def original_equation (x a : ℝ) : Prop :=
  (2*x - 1) / 5 + 1 = (x + a) / 2

/-- The equation with Bingbing's mistake --/
def incorrect_equation (x a : ℝ) : Prop :=
  2*(2*x - 1) + 1 = 5*(x + a)

/-- Theorem stating the correct value of a and the solution to the original equation --/
theorem solve_equation :
  ∃ (a : ℝ), 
    (incorrect_equation (-6) a) ∧ 
    (a = 1) ∧
    (original_equation 3 a) :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l2273_227372


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2273_227331

theorem inequality_equivalence (x : ℝ) (h : x ≠ 4) :
  (x^2 - 16) / (x - 4) ≤ 0 ↔ x ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2273_227331


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l2273_227335

theorem polynomial_expansion_equality (x : ℝ) :
  (3*x - 2) * (6*x^8 + 3*x^7 - 2*x^3 + x) = 18*x^9 - 3*x^8 - 6*x^7 - 6*x^4 - 4*x^3 + x :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l2273_227335


namespace NUMINAMATH_CALUDE_bargain_bin_books_theorem_l2273_227304

/-- Calculates the number of books in the bargain bin after two weeks of sales and additions. -/
def books_after_two_weeks (initial : ℕ) (sold_week1 sold_week2 added_week1 added_week2 : ℕ) : ℕ :=
  initial - sold_week1 + added_week1 - sold_week2 + added_week2

/-- Theorem stating that given the initial number of books and the changes during two weeks,
    the final number of books in the bargain bin is 391. -/
theorem bargain_bin_books_theorem :
  books_after_two_weeks 500 115 289 65 230 = 391 := by
  sorry

#eval books_after_two_weeks 500 115 289 65 230

end NUMINAMATH_CALUDE_bargain_bin_books_theorem_l2273_227304


namespace NUMINAMATH_CALUDE_sum_properties_l2273_227343

theorem sum_properties (x y : ℤ) (hx : ∃ m : ℤ, x = 5 * m) (hy : ∃ n : ℤ, y = 10 * n) :
  (∃ k : ℤ, x + y = 5 * k) ∧ x + y ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_properties_l2273_227343


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2273_227395

theorem sqrt_equation_solution (x : ℚ) :
  Real.sqrt (2 - 5 * x) = 8 → x = -62 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2273_227395


namespace NUMINAMATH_CALUDE_division_with_remainder_l2273_227348

theorem division_with_remainder : ∃ (q r : ℤ), 1234567 = 127 * q + r ∧ 0 ≤ r ∧ r < 127 ∧ r = 51 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l2273_227348


namespace NUMINAMATH_CALUDE_min_value_of_P_sum_l2273_227309

def P (τ : ℝ) : ℝ := (τ + 1)^3

theorem min_value_of_P_sum (x y : ℝ) (h : x + y = 0) :
  ∃ (m : ℝ), m = 2 ∧ ∀ (a b : ℝ), a + b = 0 → P a + P b ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_P_sum_l2273_227309


namespace NUMINAMATH_CALUDE_city_population_l2273_227330

theorem city_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 85 / 100 →
  partial_population = 85000 →
  (percentage * total_population : ℚ) = partial_population →
  total_population = 100000 := by
sorry

end NUMINAMATH_CALUDE_city_population_l2273_227330


namespace NUMINAMATH_CALUDE_airplane_travel_time_l2273_227387

/-- Proves that the time taken for an airplane to travel against the wind is 5 hours -/
theorem airplane_travel_time 
  (distance : ℝ) 
  (return_time : ℝ) 
  (still_air_speed : ℝ) 
  (h1 : distance = 3600) 
  (h2 : return_time = 4) 
  (h3 : still_air_speed = 810) : 
  (distance / (still_air_speed - (distance / return_time - still_air_speed))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_airplane_travel_time_l2273_227387


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l2273_227346

theorem quadratic_root_implies_coefficient (c : ℝ) : 
  ((-9 : ℝ)^2 + c*(-9) + 36 = 0) → c = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l2273_227346


namespace NUMINAMATH_CALUDE_triangle_point_distance_l2273_227383

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  AB = 13 ∧ AC = 13 ∧ BC = 10

-- Define the point P
def PointInside (P A B C : ℝ × ℝ) : Prop :=
  ∃ t u v : ℝ, t > 0 ∧ u > 0 ∧ v > 0 ∧ t + u + v = 1 ∧
  P = (t * A.1 + u * B.1 + v * C.1, t * A.2 + u * B.2 + v * C.2)

-- Define the distances PA and PB
def Distances (P A B : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 15 ∧
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 9

-- Define the angle equality
def AngleEquality (P A B C : ℝ × ℝ) : Prop :=
  let angle (X Y Z : ℝ × ℝ) := Real.arccos (
    ((X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2)) /
    (Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) * Real.sqrt ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2))
  )
  angle A P B = angle B P C ∧ angle B P C = angle C P A

-- Main theorem
theorem triangle_point_distance (A B C P : ℝ × ℝ) :
  Triangle A B C →
  PointInside P A B C →
  Distances P A B →
  AngleEquality P A B C →
  Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = (-9 + Real.sqrt 157) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_point_distance_l2273_227383


namespace NUMINAMATH_CALUDE_eightiethDigitIsOne_l2273_227332

/-- The sequence of digits formed by concatenating consecutive integers from 60 to 1 in descending order -/
def descendingSequence : List Nat := sorry

/-- The 80th digit in the descendingSequence -/
def eightiethDigit : Nat := sorry

/-- Theorem stating that the 80th digit in the sequence is 1 -/
theorem eightiethDigitIsOne : eightiethDigit = 1 := by sorry

end NUMINAMATH_CALUDE_eightiethDigitIsOne_l2273_227332


namespace NUMINAMATH_CALUDE_car_value_correct_l2273_227368

/-- The value of the car Lil Jon bought for DJ Snake's engagement -/
def car_value : ℕ := 30000

/-- The cost of the hotel stay per night -/
def hotel_cost_per_night : ℕ := 4000

/-- The number of nights stayed at the hotel -/
def nights_stayed : ℕ := 2

/-- The total value of all treats received -/
def total_value : ℕ := 158000

/-- Theorem stating that the car value is correct given the conditions -/
theorem car_value_correct :
  car_value = 30000 ∧
  hotel_cost_per_night = 4000 ∧
  nights_stayed = 2 ∧
  total_value = 158000 ∧
  (hotel_cost_per_night * nights_stayed + car_value + 4 * car_value = total_value) :=
by sorry

end NUMINAMATH_CALUDE_car_value_correct_l2273_227368


namespace NUMINAMATH_CALUDE_same_parity_of_extrema_l2273_227333

/-- A set with certain properties related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def smallest_element (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def largest_element (A : Set ℤ) : ℤ := sorry

/-- A function to determine if a number is even -/
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_of_extrema :
  is_even (smallest_element A_P) ↔ is_even (largest_element A_P) := by
  sorry

end NUMINAMATH_CALUDE_same_parity_of_extrema_l2273_227333


namespace NUMINAMATH_CALUDE_probability_product_multiple_of_four_l2273_227318

def dodecahedral_die : Finset ℕ := Finset.range 12
def eight_sided_die : Finset ℕ := Finset.range 8

def is_multiple_of_four (n : ℕ) : Bool := n % 4 = 0

theorem probability_product_multiple_of_four :
  let outcomes := dodecahedral_die.product eight_sided_die
  let favorable_outcomes := outcomes.filter (fun (x, y) => is_multiple_of_four (x * y))
  (favorable_outcomes.card : ℚ) / outcomes.card = 7 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_product_multiple_of_four_l2273_227318


namespace NUMINAMATH_CALUDE_unique_n_with_divisor_sum_property_l2273_227320

theorem unique_n_with_divisor_sum_property (n : ℕ+) 
  (h1 : ∃ d₁ d₂ d₃ d₄ : ℕ+, d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ 
        (∀ m : ℕ+, m ∣ n → m ≥ d₁) ∧
        (∀ m : ℕ+, m ∣ n → m = d₁ ∨ m ≥ d₂) ∧
        (∀ m : ℕ+, m ∣ n → m = d₁ ∨ m = d₂ ∨ m ≥ d₃) ∧
        (∀ m : ℕ+, m ∣ n → m = d₁ ∨ m = d₂ ∨ m = d₃ ∨ m ≥ d₄))
  (h2 : ∃ d₁ d₂ d₃ d₄ : ℕ+, d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ 
        d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧
        d₁^2 + d₂^2 + d₃^2 + d₄^2 = n) :
  n = 130 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_with_divisor_sum_property_l2273_227320


namespace NUMINAMATH_CALUDE_equal_water_amounts_l2273_227353

theorem equal_water_amounts (hot_fill_time cold_fill_time : ℝ) 
  (h_hot : hot_fill_time = 23)
  (h_cold : cold_fill_time = 19) :
  let delay := 2
  let hot_rate := 1 / hot_fill_time
  let cold_rate := 1 / cold_fill_time
  let total_time := hot_fill_time / 2 + delay
  hot_rate * total_time = cold_rate * (total_time - delay) :=
by sorry

end NUMINAMATH_CALUDE_equal_water_amounts_l2273_227353


namespace NUMINAMATH_CALUDE_simplified_expression_ratio_l2273_227300

theorem simplified_expression_ratio (k : ℤ) : 
  let simplified := (6 * k + 12) / 6
  let a : ℤ := 1
  let b : ℤ := 2
  simplified = a * k + b ∧ a / b = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_simplified_expression_ratio_l2273_227300


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2273_227323

theorem solve_linear_equation :
  ∃ x : ℝ, 3 * x - 7 = 2 * x + 5 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2273_227323


namespace NUMINAMATH_CALUDE_valid_representation_count_l2273_227370

/-- Represents a natural number in binary form -/
def BinaryRepresentation := List Bool

/-- Checks if a binary representation has three consecutive identical digits -/
def hasThreeConsecutiveIdenticalDigits (bin : BinaryRepresentation) : Bool :=
  sorry

/-- Counts the number of valid binary representations between 4 and 1023 -/
def countValidRepresentations : Nat :=
  sorry

/-- The main theorem stating the count of valid representations is 228 -/
theorem valid_representation_count :
  countValidRepresentations = 228 := by
  sorry

end NUMINAMATH_CALUDE_valid_representation_count_l2273_227370


namespace NUMINAMATH_CALUDE_dave_guitar_strings_l2273_227374

theorem dave_guitar_strings 
  (strings_per_night : ℕ) 
  (shows_per_week : ℕ) 
  (total_weeks : ℕ) 
  (h1 : strings_per_night = 2) 
  (h2 : shows_per_week = 6) 
  (h3 : total_weeks = 12) : 
  strings_per_night * shows_per_week * total_weeks = 144 := by
sorry

end NUMINAMATH_CALUDE_dave_guitar_strings_l2273_227374


namespace NUMINAMATH_CALUDE_translation_down_three_units_l2273_227312

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translateLine (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - dy }

theorem translation_down_three_units :
  let originalLine : Line := { slope := 1/2, intercept := 0 }
  let translatedLine : Line := translateLine originalLine 3
  translatedLine = { slope := 1/2, intercept := -3 } := by
  sorry

end NUMINAMATH_CALUDE_translation_down_three_units_l2273_227312


namespace NUMINAMATH_CALUDE_value_of_A_minus_2B_A_minus_2B_independent_of_y_l2273_227336

/-- Definition of A in terms of x and y -/
def A (x y : ℝ) : ℝ := 2 * x^2 + x * y + 3 * y

/-- Definition of B in terms of x and y -/
def B (x y : ℝ) : ℝ := x^2 - x * y

/-- Theorem stating the value of A - 2B under the given condition -/
theorem value_of_A_minus_2B (x y : ℝ) :
  (x + 2)^2 + |y - 3| = 0 → A x y - 2 * B x y = -9 := by sorry

/-- Theorem stating the condition for A - 2B to be independent of y -/
theorem A_minus_2B_independent_of_y (x : ℝ) :
  (∀ y : ℝ, ∃ k : ℝ, A x y - 2 * B x y = k) ↔ x = -1 := by sorry

end NUMINAMATH_CALUDE_value_of_A_minus_2B_A_minus_2B_independent_of_y_l2273_227336


namespace NUMINAMATH_CALUDE_place_face_difference_46_4_l2273_227322

/-- The place value of a digit in a two-digit number -/
def placeValue (n : ℕ) (d : ℕ) : ℕ :=
  if n ≥ 10 ∧ n < 100 ∧ d = n / 10 then d * 10 else 0

/-- The face value of a digit -/
def faceValue (d : ℕ) : ℕ := d

/-- The difference between place value and face value for a digit in a two-digit number -/
def placeFaceDifference (n : ℕ) (d : ℕ) : ℕ :=
  placeValue n d - faceValue d

theorem place_face_difference_46_4 : 
  placeFaceDifference 46 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_place_face_difference_46_4_l2273_227322


namespace NUMINAMATH_CALUDE_line_through_point_l2273_227360

/-- Given a line with equation y = 2x + b passing through the point (-4, 0), prove that b = 8 -/
theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b) → -- The line has equation y = 2x + b
  (0 = 2 * (-4) + b) →         -- The line passes through the point (-4, 0)
  b = 8 :=                     -- The value of b is 8
by sorry

end NUMINAMATH_CALUDE_line_through_point_l2273_227360


namespace NUMINAMATH_CALUDE_count_special_integers_l2273_227373

theorem count_special_integers (a : ℕ) : 
  (∃ (count : ℕ), count = (Finset.filter 
    (fun a => a > 0 ∧ a < 100 ∧ (a^3 + 23) % 24 = 0) 
    (Finset.range 100)).card ∧ count = 9) := by
  sorry

end NUMINAMATH_CALUDE_count_special_integers_l2273_227373


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2273_227384

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let original_area := L * B
  let new_length := L / 2
  let new_breadth := 3 * B
  let new_area := new_length * new_breadth
  ((new_area - original_area) / original_area) * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2273_227384


namespace NUMINAMATH_CALUDE_ratio_squares_equality_l2273_227385

theorem ratio_squares_equality : (1625^2 - 1612^2) / (1631^2 - 1606^2) = 13 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_squares_equality_l2273_227385


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l2273_227326

/-- Given that 26 cows eat 26 bags of husk in 26 days, prove that one cow will eat one bag of husk in 26 days -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 26 ∧ bags = 26 ∧ days = 26) :
  (1 : ℕ) * bags = (1 : ℕ) * cows * days := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l2273_227326


namespace NUMINAMATH_CALUDE_part_one_part_two_l2273_227379

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Define set B
def B : Set ℝ := {x : ℝ | |2*x - 1| ≤ 3}

-- Part I
theorem part_one : 
  {x : ℝ | f 5 x > 9} = {x : ℝ | x < -6 ∨ x > 3} := by sorry

-- Part II
-- Define set A
def A (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ |x - 4|}

theorem part_two :
  {a : ℝ | A a ∪ B = A a} = {a : ℝ | -1 ≤ a ∧ a ≤ 0} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2273_227379


namespace NUMINAMATH_CALUDE_helen_oranges_l2273_227340

/-- The number of oranges Helen started with -/
def initial_oranges : ℕ := sorry

/-- Helen gets 29 more oranges from Ann -/
def oranges_from_ann : ℕ := 29

/-- Helen ends up with 38 oranges -/
def final_oranges : ℕ := 38

/-- Theorem stating that the initial number of oranges plus the oranges from Ann equals the final number of oranges -/
theorem helen_oranges : initial_oranges + oranges_from_ann = final_oranges := by sorry

end NUMINAMATH_CALUDE_helen_oranges_l2273_227340


namespace NUMINAMATH_CALUDE_nine_points_interior_lattice_point_l2273_227344

/-- A lattice point in 3D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ
  z : ℤ

/-- The statement that there exists an interior lattice point -/
def exists_interior_lattice_point (points : Finset LatticePoint) : Prop :=
  ∃ p q : LatticePoint, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧
    ∃ r : LatticePoint, r.x = (p.x + q.x) / 2 ∧ 
                        r.y = (p.y + q.y) / 2 ∧ 
                        r.z = (p.z + q.z) / 2

/-- The main theorem -/
theorem nine_points_interior_lattice_point 
  (points : Finset LatticePoint) 
  (h : points.card = 9) : 
  exists_interior_lattice_point points := by
  sorry

#check nine_points_interior_lattice_point

end NUMINAMATH_CALUDE_nine_points_interior_lattice_point_l2273_227344


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2273_227341

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2273_227341


namespace NUMINAMATH_CALUDE_product_of_fractions_l2273_227308

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2273_227308


namespace NUMINAMATH_CALUDE_bryan_pushups_l2273_227306

/-- The number of push-ups Bryan did in total -/
def total_pushups (sets : ℕ) (pushups_per_set : ℕ) (reduced_pushups : ℕ) : ℕ :=
  (sets - 1) * pushups_per_set + (pushups_per_set - reduced_pushups)

/-- Theorem stating that Bryan did 100 push-ups in total -/
theorem bryan_pushups :
  total_pushups 9 12 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_bryan_pushups_l2273_227306


namespace NUMINAMATH_CALUDE_matrix_fourth_power_l2273_227349

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_fourth_power :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_fourth_power_l2273_227349
