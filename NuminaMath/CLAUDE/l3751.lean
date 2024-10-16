import Mathlib

namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3751_375185

theorem max_sum_of_factors (A B C : ℕ) : 
  A ≠ B → B ≠ C → A ≠ C → 
  A > 0 → B > 0 → C > 0 → 
  A * B * C = 2310 → 
  A + B + C ≤ 52 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3751_375185


namespace NUMINAMATH_CALUDE_horner_v4_value_l3751_375138

def horner_step (v : ℤ) (a : ℤ) (x : ℤ) : ℤ := v * x + a

def horner_method (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc coeff => horner_step acc coeff x) 0

theorem horner_v4_value :
  let coeffs := [3, 5, 6, 20, -8, 35, 12]
  let x := -2
  let v0 := 3
  let v1 := horner_step v0 5 x
  let v2 := horner_step v1 6 x
  let v3 := horner_step v2 20 x
  let v4 := horner_step v3 (-8) x
  v4 = -16 := by sorry

end NUMINAMATH_CALUDE_horner_v4_value_l3751_375138


namespace NUMINAMATH_CALUDE_tangent_slope_angle_is_45_degrees_l3751_375143

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- The point of interest
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_slope_angle_is_45_degrees :
  let slope := f' point.1
  let angle := Real.arctan slope
  angle = π / 4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_is_45_degrees_l3751_375143


namespace NUMINAMATH_CALUDE_total_trees_cut_is_1021_l3751_375192

/-- Calculates the total number of trees cut down by James and his helpers. -/
def total_trees_cut (james_rate : ℕ) (brother_rate : ℕ) (cousin_rate : ℕ) (professional_rate : ℕ) : ℕ :=
  let james_alone := 2 * james_rate
  let with_brothers := 3 * (james_rate + 2 * brother_rate)
  let with_cousin := 4 * (james_rate + 2 * brother_rate + cousin_rate)
  let all_together := 5 * (james_rate + 2 * brother_rate + cousin_rate + professional_rate)
  james_alone + with_brothers + with_cousin + all_together

/-- The theorem states that the total number of trees cut down is 1021. -/
theorem total_trees_cut_is_1021 :
  total_trees_cut 20 16 23 30 = 1021 := by
  sorry

#eval total_trees_cut 20 16 23 30

end NUMINAMATH_CALUDE_total_trees_cut_is_1021_l3751_375192


namespace NUMINAMATH_CALUDE_range_of_2sin_squared_l3751_375129

theorem range_of_2sin_squared (x : ℝ) : 0 ≤ 2 * (Real.sin x)^2 ∧ 2 * (Real.sin x)^2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2sin_squared_l3751_375129


namespace NUMINAMATH_CALUDE_average_sale_per_month_l3751_375166

def sales : List ℕ := [120, 80, 100, 140, 160]

theorem average_sale_per_month : 
  (List.sum sales) / (List.length sales) = 120 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_per_month_l3751_375166


namespace NUMINAMATH_CALUDE_cos_equality_proof_l3751_375147

theorem cos_equality_proof (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_proof_l3751_375147


namespace NUMINAMATH_CALUDE_james_van_capacity_l3751_375113

/-- Proves that the total capacity of James' vans is 57600 gallons --/
theorem james_van_capacity :
  let total_vans : ℕ := 6
  let large_van_capacity : ℕ := 8000
  let large_van_count : ℕ := 2
  let medium_van_capacity : ℕ := large_van_capacity * 7 / 10  -- 30% less than 8000
  let medium_van_count : ℕ := 1
  let small_van_count : ℕ := total_vans - large_van_count - medium_van_count
  let total_capacity : ℕ := 57600
  let remaining_capacity : ℕ := total_capacity - (large_van_capacity * large_van_count + medium_van_capacity * medium_van_count)
  let small_van_capacity : ℕ := remaining_capacity / small_van_count

  (large_van_capacity * large_van_count + 
   medium_van_capacity * medium_van_count + 
   small_van_capacity * small_van_count) = total_capacity := by
  sorry

end NUMINAMATH_CALUDE_james_van_capacity_l3751_375113


namespace NUMINAMATH_CALUDE_rectangle_area_l3751_375196

theorem rectangle_area (w : ℝ) (h : w > 0) : 
  w^2 + (3*w)^2 = 16^2 → w * (3*w) = 76.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3751_375196


namespace NUMINAMATH_CALUDE_distance_to_karasuk_is_210_l3751_375173

/-- The distance from Novosibirsk to Karasuk --/
def distance_to_karasuk : ℝ := 210

/-- The initial distance between the bus and the car --/
def initial_distance : ℝ := 70

/-- The distance the car travels after catching up with the bus --/
def car_distance_after_catchup : ℝ := 40

/-- The distance the bus travels after the car catches up --/
def bus_distance_after_catchup : ℝ := 20

/-- The speed of the bus --/
def bus_speed : ℝ := sorry

/-- The speed of the car --/
def car_speed : ℝ := sorry

/-- The time taken for the car to catch up with the bus --/
def catchup_time : ℝ := sorry

theorem distance_to_karasuk_is_210 :
  distance_to_karasuk = initial_distance + car_speed * catchup_time :=
by sorry

end NUMINAMATH_CALUDE_distance_to_karasuk_is_210_l3751_375173


namespace NUMINAMATH_CALUDE_milk_problem_solution_l3751_375128

/-- Calculates the final amount of milk in a storage tank given initial amount,
    pumping out rate and duration, and adding rate and duration. -/
def final_milk_amount (initial : ℝ) (pump_rate : ℝ) (pump_hours : ℝ) 
                      (add_rate : ℝ) (add_hours : ℝ) : ℝ :=
  initial - pump_rate * pump_hours + add_rate * add_hours

/-- Theorem stating that given the specific conditions from the problem,
    the final amount of milk in the storage tank is 28,980 gallons. -/
theorem milk_problem_solution :
  final_milk_amount 30000 2880 4 1500 7 = 28980 := by
  sorry

end NUMINAMATH_CALUDE_milk_problem_solution_l3751_375128


namespace NUMINAMATH_CALUDE_pats_calculation_l3751_375106

theorem pats_calculation (x : ℝ) : 
  (x / 4 - 18 = 12) → (400 < 4*x + 18 ∧ 4*x + 18 < 600) := by
  sorry

end NUMINAMATH_CALUDE_pats_calculation_l3751_375106


namespace NUMINAMATH_CALUDE_smallest_a_in_special_progression_l3751_375187

theorem smallest_a_in_special_progression (a b c : ℤ) 
  (h1 : a < c ∧ c < b)
  (h2 : 2 * c = a + b)  -- arithmetic progression condition
  (h3 : b * b = a * c)  -- geometric progression condition
  : a ≥ -4 ∧ ∃ (a₀ b₀ c₀ : ℤ), a₀ = -4 ∧ b₀ = 2 ∧ c₀ = -1 ∧ 
    a₀ < c₀ ∧ c₀ < b₀ ∧ 
    2 * c₀ = a₀ + b₀ ∧ 
    b₀ * b₀ = a₀ * c₀ :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_in_special_progression_l3751_375187


namespace NUMINAMATH_CALUDE_restaurant_production_in_june_l3751_375103

/-- Represents the daily production of a restaurant -/
structure DailyProduction where
  cheesePizzas : ℕ
  pepperoniPizzas : ℕ
  beefHotDogs : ℕ
  chickenHotDogs : ℕ

/-- Calculates the monthly production based on daily production and number of days -/
def monthlyProduction (daily : DailyProduction) (days : ℕ) : DailyProduction :=
  { cheesePizzas := daily.cheesePizzas * days
  , pepperoniPizzas := daily.pepperoniPizzas * days
  , beefHotDogs := daily.beefHotDogs * days
  , chickenHotDogs := daily.chickenHotDogs * days
  }

theorem restaurant_production_in_june 
  (daily : DailyProduction)
  (cheese_more_than_hotdogs : daily.cheesePizzas = daily.beefHotDogs + daily.chickenHotDogs + 40)
  (pepperoni_twice_cheese : daily.pepperoniPizzas = 2 * daily.cheesePizzas)
  (total_hotdogs : daily.beefHotDogs + daily.chickenHotDogs = 60)
  (beef_hotdogs : daily.beefHotDogs = 30)
  (chicken_hotdogs : daily.chickenHotDogs = 30)
  : monthlyProduction daily 30 = 
    { cheesePizzas := 3000
    , pepperoniPizzas := 6000
    , beefHotDogs := 900
    , chickenHotDogs := 900
    } := by
  sorry


end NUMINAMATH_CALUDE_restaurant_production_in_june_l3751_375103


namespace NUMINAMATH_CALUDE_sexagenary_cycle_3023_l3751_375130

/-- Represents a year in the sexagenary cycle -/
structure SexagenaryYear where
  heavenlyStem : Fin 10
  earthlyBranch : Fin 12

/-- The sexagenary cycle -/
def sexagenaryCycle : ℕ → SexagenaryYear := sorry

/-- Maps a natural number to its representation in the sexagenary cycle -/
def toSexagenaryYear (year : ℕ) : SexagenaryYear :=
  sexagenaryCycle (year % 60)

/-- Checks if a given SexagenaryYear corresponds to "Gui Mao" -/
def isGuiMao (year : SexagenaryYear) : Prop :=
  year.heavenlyStem = 9 ∧ year.earthlyBranch = 3

/-- Checks if a given SexagenaryYear corresponds to "Gui Wei" -/
def isGuiWei (year : SexagenaryYear) : Prop :=
  year.heavenlyStem = 9 ∧ year.earthlyBranch = 7

theorem sexagenary_cycle_3023 :
  isGuiMao (toSexagenaryYear 2023) →
  isGuiWei (toSexagenaryYear 3023) := by
  sorry

end NUMINAMATH_CALUDE_sexagenary_cycle_3023_l3751_375130


namespace NUMINAMATH_CALUDE_tank_b_one_third_full_time_l3751_375181

/-- Represents a rectangular tank with given dimensions -/
structure RectangularTank where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular tank -/
def RectangularTank.volume (tank : RectangularTank) : ℝ :=
  tank.length * tank.width * tank.height

/-- Represents the filling process of a tank -/
structure TankFilling where
  tank : RectangularTank
  fillRate : ℝ  -- in cm³/s

/-- Theorem: Tank B will be 1/3 full after 30 seconds -/
theorem tank_b_one_third_full_time (tank_b : TankFilling) 
    (h1 : tank_b.tank.length = 5)
    (h2 : tank_b.tank.width = 9)
    (h3 : tank_b.tank.height = 8)
    (h4 : tank_b.fillRate = 4) : 
    tank_b.fillRate * 30 = (1/3) * tank_b.tank.volume := by
  sorry

#check tank_b_one_third_full_time

end NUMINAMATH_CALUDE_tank_b_one_third_full_time_l3751_375181


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3751_375125

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3751_375125


namespace NUMINAMATH_CALUDE_fraction_inequality_counterexample_l3751_375119

theorem fraction_inequality_counterexample : 
  ∃ (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ), 
    a₁ > 0 ∧ a₂ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ c₁ > 0 ∧ c₂ > 0 ∧ d₁ > 0 ∧ d₂ > 0 ∧
    (a₁ : ℚ) / b₁ < a₂ / b₂ ∧
    (c₁ : ℚ) / d₁ < c₂ / d₂ ∧
    (a₁ + c₁ : ℚ) / (b₁ + d₁) ≥ (a₂ + c₂) / (b₂ + d₂) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_counterexample_l3751_375119


namespace NUMINAMATH_CALUDE_no_solution_3a_squared_equals_b_squared_plus_1_l3751_375172

theorem no_solution_3a_squared_equals_b_squared_plus_1 :
  ¬ ∃ (a b : ℕ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_3a_squared_equals_b_squared_plus_1_l3751_375172


namespace NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l3751_375124

/-- Given a 25% profit, prove that the ratio of cost price to selling price is 4 : 5 -/
theorem cost_to_selling_price_ratio (cost_price selling_price : ℝ) 
  (h_positive : cost_price > 0)
  (h_profit : selling_price = cost_price * (1 + 0.25)) :
  cost_price / selling_price = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l3751_375124


namespace NUMINAMATH_CALUDE_no_valid_schedule_l3751_375133

theorem no_valid_schedule : ¬∃ (a b : ℕ+), (29 ∣ a) ∧ (32 ∣ b) ∧ (a + b = 29 * 32) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_schedule_l3751_375133


namespace NUMINAMATH_CALUDE_no_unique_solution_l3751_375170

/-- The system of equations does not have a unique solution if and only if k = 3 -/
theorem no_unique_solution (k : ℝ) : 
  (∃ (x y : ℝ), 4 * (3 * x + 4 * y) = 48 ∧ k * x + 12 * y = 30) ∧ 
  ¬(∃! (x y : ℝ), 4 * (3 * x + 4 * y) = 48 ∧ k * x + 12 * y = 30) ↔ 
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_no_unique_solution_l3751_375170


namespace NUMINAMATH_CALUDE_angle_bisector_length_l3751_375176

/-- The length of an angle bisector in a triangle -/
theorem angle_bisector_length (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (htri : a < b + c ∧ b < a + c ∧ c < a + b) :
  let p := (a + b + c) / 2
  ∃ l_a : ℝ, l_a = (2 / (b + c)) * Real.sqrt (b * c * p * (p - a)) := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l3751_375176


namespace NUMINAMATH_CALUDE_orange_pricing_theorem_l3751_375118

/-- Pricing tiers for oranges -/
def price_4 : ℕ := 15
def price_7 : ℕ := 25
def price_10 : ℕ := 32

/-- Number of groups purchased -/
def num_groups : ℕ := 3

/-- Total number of oranges purchased -/
def total_oranges : ℕ := 4 * num_groups + 7 * num_groups + 10 * num_groups

/-- Calculate the total cost in cents -/
def total_cost : ℕ := price_4 * num_groups + price_7 * num_groups + price_10 * num_groups

/-- Calculate the average cost per orange in cents (as a rational number) -/
def avg_cost_per_orange : ℚ := total_cost / total_oranges

theorem orange_pricing_theorem :
  total_oranges = 21 ∧ 
  total_cost = 216 ∧ 
  avg_cost_per_orange = 1029 / 100 := by
  sorry

end NUMINAMATH_CALUDE_orange_pricing_theorem_l3751_375118


namespace NUMINAMATH_CALUDE_farm_animals_l3751_375117

/-- The number of animals in a farm with ducks and dogs -/
def total_animals (num_ducks : ℕ) (total_legs : ℕ) : ℕ :=
  num_ducks + (total_legs - 2 * num_ducks) / 4

/-- Theorem: Given the conditions, there are 11 animals in total -/
theorem farm_animals : total_animals 6 32 = 11 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l3751_375117


namespace NUMINAMATH_CALUDE_lunch_ratio_proof_l3751_375139

theorem lunch_ratio_proof (total_students : Nat) (cafeteria_students : Nat) (no_lunch_students : Nat) :
  total_students = 60 →
  cafeteria_students = 10 →
  no_lunch_students = 20 →
  ∃ k : Nat, total_students - cafeteria_students - no_lunch_students = k * cafeteria_students →
  (total_students - cafeteria_students - no_lunch_students) / cafeteria_students = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_lunch_ratio_proof_l3751_375139


namespace NUMINAMATH_CALUDE_quadratic_roots_preservation_l3751_375140

theorem quadratic_roots_preservation
  (a b : ℝ) (k : ℝ)
  (h_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*a*x₁ + b = 0 ∧ x₂^2 + 2*a*x₂ + b = 0)
  (h_k_pos : k > 0) :
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧
    (y₁^2 + 2*a*y₁ + b) + k*(y₁ + a)^2 = 0 ∧
    (y₂^2 + 2*a*y₂ + b) + k*(y₂ + a)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_preservation_l3751_375140


namespace NUMINAMATH_CALUDE_merchant_profit_l3751_375134

theorem merchant_profit (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
  markup_percentage = 0.40 →
  discount_percentage = 0.10 →
  let marked_price := cost_price * (1 + markup_percentage)
  let selling_price := marked_price * (1 - discount_percentage)
  let profit := selling_price - cost_price
  let profit_percentage := profit / cost_price
  profit_percentage = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_l3751_375134


namespace NUMINAMATH_CALUDE_chameleon_theorem_l3751_375199

/-- Represents the resting period before catching the m-th fly -/
def resting_period (m : ℕ) : ℕ :=
  sorry

/-- Represents the total time before catching the m-th fly -/
def total_time (m : ℕ) : ℕ :=
  sorry

/-- Represents the number of flies caught after t minutes -/
def flies_caught (t : ℕ) : ℕ :=
  sorry

/-- The chameleon's resting and catching behavior -/
axiom resting_rule_1 : resting_period 1 = 1
axiom resting_rule_2 : ∀ m : ℕ, resting_period (2 * m) = resting_period m
axiom resting_rule_3 : ∀ m : ℕ, resting_period (2 * m + 1) = resting_period m + 1
axiom catch_instantly : ∀ m : ℕ, total_time (m + 1) = total_time m + resting_period (m + 1) + 1

theorem chameleon_theorem :
  (∃ m : ℕ, m = 510 ∧ resting_period (m + 1) = 9 ∧ ∀ k < m, resting_period (k + 1) < 9) ∧
  (total_time 98 = 312) ∧
  (flies_caught 1999 = 462) :=
sorry

end NUMINAMATH_CALUDE_chameleon_theorem_l3751_375199


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3751_375194

theorem unique_three_digit_number : 
  ∃! (m g u : ℕ), 
    m ≠ g ∧ m ≠ u ∧ g ≠ u ∧
    m < 10 ∧ g < 10 ∧ u < 10 ∧
    m ≠ 0 ∧
    100 * m + 10 * g + u = (m + g + u) * (m + g + u - 2) ∧
    100 * m + 10 * g + u = 195 := by
  sorry


end NUMINAMATH_CALUDE_unique_three_digit_number_l3751_375194


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_curves_l3751_375156

/-- Given two curves in polar coordinates that intersect, 
    prove the equation of the perpendicular bisector of their intersection points. -/
theorem perpendicular_bisector_of_intersecting_curves 
  (C₁ : ℝ → ℝ → Prop) 
  (C₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ θ ρ, C₁ ρ θ ↔ ρ = 2 * Real.sin θ)
  (h₂ : ∀ θ ρ, C₂ ρ θ ↔ ρ = 2 * Real.cos θ)
  (A B : ℝ × ℝ)
  (hA : C₁ A.1 A.2 ∧ C₂ A.1 A.2)
  (hB : C₁ B.1 B.2 ∧ C₂ B.1 B.2)
  (hAB : A ≠ B) :
  ∃ (ρ θ : ℝ), ρ * Real.sin θ + ρ * Real.cos θ = 1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_curves_l3751_375156


namespace NUMINAMATH_CALUDE_inequality_and_quadratic_solution_l3751_375175

theorem inequality_and_quadratic_solution :
  ∃ (m : ℤ), 1 < m ∧ m < 4 ∧
  ∀ (x : ℝ), 1 < x ∧ x < 4 →
  (x^2 - 2*x - m = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_quadratic_solution_l3751_375175


namespace NUMINAMATH_CALUDE_arithmetic_progression_y_range_l3751_375144

theorem arithmetic_progression_y_range (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    Real.log r = Real.log 2 - Real.log (Real.sin x - 1/3) ∧ 
    Real.log (Real.sin x - 1/3) = Real.log 2 - Real.log (1 - y)) →
  (∃ y_min : ℝ, y_min = 7/9 ∧ y ≥ y_min) ∧ 
  (∀ y_max : ℝ, y < y_max) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_y_range_l3751_375144


namespace NUMINAMATH_CALUDE_x_plus_y_positive_l3751_375159

theorem x_plus_y_positive (x y : ℝ) (h1 : x * y < 0) (h2 : x > |y|) : x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_positive_l3751_375159


namespace NUMINAMATH_CALUDE_yogurt_combinations_l3751_375146

def num_flavors : ℕ := 5
def num_toppings : ℕ := 8

def combinations_with_no_topping : ℕ := 1
def combinations_with_one_topping (n : ℕ) : ℕ := n
def combinations_with_two_toppings (n : ℕ) : ℕ := n * (n - 1) / 2

def total_topping_combinations (n : ℕ) : ℕ :=
  combinations_with_no_topping + 
  combinations_with_one_topping n + 
  combinations_with_two_toppings n

theorem yogurt_combinations : 
  num_flavors * total_topping_combinations num_toppings = 185 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l3751_375146


namespace NUMINAMATH_CALUDE_lindas_cookies_l3751_375100

theorem lindas_cookies (classmates : Nat) (cookies_per_student : Nat) 
  (cookies_per_batch : Nat) (oatmeal_batches : Nat) (additional_batches : Nat) :
  classmates = 24 →
  cookies_per_student = 10 →
  cookies_per_batch = 48 →
  oatmeal_batches = 1 →
  additional_batches = 2 →
  ∃ (chocolate_chip_batches : Nat),
    chocolate_chip_batches * cookies_per_batch + 
    oatmeal_batches * cookies_per_batch + 
    additional_batches * cookies_per_batch = 
    classmates * cookies_per_student ∧
    chocolate_chip_batches = 2 :=
by sorry

end NUMINAMATH_CALUDE_lindas_cookies_l3751_375100


namespace NUMINAMATH_CALUDE_product_equals_243_l3751_375135

theorem product_equals_243 :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l3751_375135


namespace NUMINAMATH_CALUDE_equality_of_triples_l3751_375193

theorem equality_of_triples (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x + y + z = a + b + c)
  (h_prod : x * y * z = a * b * c)
  (h_order1 : a ≤ x ∧ x < y ∧ y < z ∧ z ≤ c)
  (h_order2 : a < b ∧ b < c) :
  a = x ∧ b = y ∧ c = z := by
  sorry

end NUMINAMATH_CALUDE_equality_of_triples_l3751_375193


namespace NUMINAMATH_CALUDE_haley_marble_distribution_l3751_375169

/-- The number of marbles given to each boy in Haley's class. -/
def marbles_per_boy (total_marbles : ℕ) (num_boys : ℕ) : ℕ :=
  total_marbles / num_boys

/-- Theorem stating that Haley gave 9 marbles to each boy. -/
theorem haley_marble_distribution :
  marbles_per_boy 99 11 = 9 :=
by
  -- The proof goes here
  sorry

#eval marbles_per_boy 99 11  -- This should output 9

end NUMINAMATH_CALUDE_haley_marble_distribution_l3751_375169


namespace NUMINAMATH_CALUDE_a_equals_four_l3751_375190

theorem a_equals_four (a : ℝ) (h : a * 2 * (2^3) = 2^6) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_four_l3751_375190


namespace NUMINAMATH_CALUDE_garden_dimensions_l3751_375150

/-- Represents a rectangular garden with given perimeter and length-width relationship --/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_width_relation : length = width + 3
  perimeter_formula : perimeter = 2 * (length + width)

/-- Theorem stating the dimensions of the garden given the conditions --/
theorem garden_dimensions (g : RectangularGarden) 
  (h : g.perimeter = 26) : g.width = 5 ∧ g.length = 8 := by
  sorry

#check garden_dimensions

end NUMINAMATH_CALUDE_garden_dimensions_l3751_375150


namespace NUMINAMATH_CALUDE_maria_budget_excess_l3751_375127

theorem maria_budget_excess : 
  let sweater_price : ℚ := 35
  let scarf_price : ℚ := 25
  let mittens_price : ℚ := 15
  let hat_price : ℚ := 12
  let family_members : ℕ := 15
  let discount_threshold : ℚ := 800
  let discount_rate : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.05
  let spending_limit : ℚ := 1500

  let set_price := 2 * sweater_price + scarf_price + mittens_price + hat_price
  let total_price := family_members * set_price
  let discounted_price := if total_price > discount_threshold 
                          then total_price * (1 - discount_rate) 
                          else total_price
  let final_price := discounted_price * (1 + sales_tax_rate)

  final_price - spending_limit = 229.35 := by sorry

end NUMINAMATH_CALUDE_maria_budget_excess_l3751_375127


namespace NUMINAMATH_CALUDE_initial_birds_l3751_375184

theorem initial_birds (initial_birds final_birds additional_birds : ℕ) 
  (h1 : additional_birds = 21)
  (h2 : final_birds = 35)
  (h3 : final_birds = initial_birds + additional_birds) : 
  initial_birds = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_l3751_375184


namespace NUMINAMATH_CALUDE_gcd_3869_6497_l3751_375162

theorem gcd_3869_6497 : Nat.gcd 3869 6497 = 73 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3869_6497_l3751_375162


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l3751_375178

theorem fraction_sum_inequality (a b : ℝ) (h : a * b ≠ 0) :
  (a * b > 0 → b / a + a / b ≥ 2) ∧
  (a * b < 0 → |b / a + a / b| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l3751_375178


namespace NUMINAMATH_CALUDE_quadratic_function_difference_l3751_375186

/-- A quadratic function with the property g(x+1) - g(x) = 2x + 3 for all real x -/
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (x + 1) - g x = 2 * x + 3

theorem quadratic_function_difference (g : ℝ → ℝ) (h : g_property g) : 
  g 2 - g 6 = -40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_difference_l3751_375186


namespace NUMINAMATH_CALUDE_least_integer_with_specific_divisibility_l3751_375114

theorem least_integer_with_specific_divisibility : ∃ n : ℕ+,
  (∀ k : ℕ, k ≤ 28 → k ∣ n) ∧
  (31 ∣ n) ∧
  ¬(29 ∣ n) ∧
  ¬(30 ∣ n) ∧
  (∀ m : ℕ+, m < n →
    ¬((∀ k : ℕ, k ≤ 28 → k ∣ m) ∧
      (31 ∣ m) ∧
      ¬(29 ∣ m) ∧
      ¬(30 ∣ m))) ∧
  n = 477638700 := by
sorry

end NUMINAMATH_CALUDE_least_integer_with_specific_divisibility_l3751_375114


namespace NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l3751_375126

theorem quadratic_trinomial_minimum (a b : ℝ) (h1 : a > b)
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + b = 0) :
  ∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ 
    (∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ m) ∧
    (∃ x : ℝ, (a^2 + b^2) / (a - b) = m) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l3751_375126


namespace NUMINAMATH_CALUDE_blueberry_pies_count_l3751_375168

/-- Proves that the number of blueberry pies is 10, given 30 total pies equally divided among three types -/
theorem blueberry_pies_count (total_pies : ℕ) (num_types : ℕ) (h1 : total_pies = 30) (h2 : num_types = 3) :
  total_pies / num_types = 10 := by
  sorry

#check blueberry_pies_count

end NUMINAMATH_CALUDE_blueberry_pies_count_l3751_375168


namespace NUMINAMATH_CALUDE_three_number_sum_l3751_375107

theorem three_number_sum (a b c : ℝ) 
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : (a + b + c) / 3 = a + 7)
  (h4 : (a + b + c) / 3 = c - 20)
  (h5 : b = 8) : 
  a + b + c = 63 := by sorry

end NUMINAMATH_CALUDE_three_number_sum_l3751_375107


namespace NUMINAMATH_CALUDE_line_through_points_with_45_degree_inclination_l3751_375161

/-- Given a line passing through points P(-2, m) and Q(m, 4) with an inclination angle of 45°, prove that m = 1. -/
theorem line_through_points_with_45_degree_inclination (m : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    ((-2, m) ∈ line) ∧ 
    ((m, 4) ∈ line) ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → (y - m) = (x + 2))) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_with_45_degree_inclination_l3751_375161


namespace NUMINAMATH_CALUDE_principal_cup_problem_l3751_375131

/-- The probability of team A answering correctly -/
def P_A : ℚ := 3/4

/-- The probability of both teams A and C answering incorrectly -/
def P_AC_incorrect : ℚ := 1/12

/-- The probability of both teams B and C answering correctly -/
def P_BC_correct : ℚ := 1/4

/-- The probability of team B answering correctly -/
def P_B : ℚ := 3/8

/-- The probability of team C answering correctly -/
def P_C : ℚ := 2/3

/-- The probability of exactly two teams answering correctly -/
def P_two_correct : ℚ := 15/32

theorem principal_cup_problem (P_A P_AC_incorrect P_BC_correct P_B P_C P_two_correct : ℚ) :
  P_A = 3/4 →
  P_AC_incorrect = 1/12 →
  P_BC_correct = 1/4 →
  P_B = 3/8 ∧
  P_C = 2/3 ∧
  P_two_correct = 15/32 :=
by
  sorry

end NUMINAMATH_CALUDE_principal_cup_problem_l3751_375131


namespace NUMINAMATH_CALUDE_new_rectangle_area_comparison_l3751_375152

theorem new_rectangle_area_comparison (a b : ℝ) (h : 0 < a ∧ a < b) :
  let new_base := 2 * a * b
  let new_height := (a * Real.sqrt (a^2 + b^2)) / 2
  let new_area := new_base * new_height
  let circle_area := Real.pi * b^2
  new_area = a^2 * b * Real.sqrt (a^2 + b^2) ∧ 
  ∃ (a b : ℝ), new_area ≠ circle_area :=
by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_comparison_l3751_375152


namespace NUMINAMATH_CALUDE_hamburgers_served_l3751_375104

/-- Given a restaurant that made a certain number of hamburgers and had some left over,
    calculate the number of hamburgers served. -/
theorem hamburgers_served (total : ℕ) (leftover : ℕ) (h1 : total = 9) (h2 : leftover = 6) :
  total - leftover = 3 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_served_l3751_375104


namespace NUMINAMATH_CALUDE_f_of_f_one_eq_two_l3751_375116

def f (x : ℝ) : ℝ := 4 * x^2 - 6 * x + 2

theorem f_of_f_one_eq_two : f (f 1) = 2 := by sorry

end NUMINAMATH_CALUDE_f_of_f_one_eq_two_l3751_375116


namespace NUMINAMATH_CALUDE_multiple_properties_l3751_375165

theorem multiple_properties (a b c : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k)
  (hb : ∃ k : ℤ, b = 12 * k)
  (hc : ∃ k : ℤ, c = 9 * k) :
  (∃ k : ℤ, b = 3 * k) ∧
  (∃ k : ℤ, a - b = 3 * k) ∧
  (∃ k : ℤ, a - c = 3 * k) ∧
  (∃ k : ℤ, c - b = 3 * k) := by
  sorry

end NUMINAMATH_CALUDE_multiple_properties_l3751_375165


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l3751_375191

theorem algebraic_expression_simplification (a : ℝ) (h : a = Real.sqrt 2) :
  a / (a^2 - 2*a + 1) / (1 + 1 / (a - 1)) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l3751_375191


namespace NUMINAMATH_CALUDE_notebook_cost_l3751_375108

theorem notebook_cost (notebook_cost pen_cost : ℚ) 
  (total_cost : notebook_cost + pen_cost = 5/2)
  (price_difference : notebook_cost = pen_cost + 2) :
  notebook_cost = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l3751_375108


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l3751_375198

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  eight_hour_days : Nat
  six_hour_days : Nat
  weekly_earnings : Nat

/-- Calculate Sheila's hourly rate --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (8 * schedule.eight_hour_days + 6 * schedule.six_hour_days)

/-- Theorem: Sheila's hourly rate is $6 --/
theorem sheila_hourly_rate :
  let schedule : WorkSchedule := {
    eight_hour_days := 3,
    six_hour_days := 2,
    weekly_earnings := 216
  }
  hourly_rate schedule = 6 := by sorry

end NUMINAMATH_CALUDE_sheila_hourly_rate_l3751_375198


namespace NUMINAMATH_CALUDE_cranberry_juice_cost_l3751_375136

/-- The total cost of a can of cranberry juice -/
theorem cranberry_juice_cost (ounces : ℕ) (cost_per_ounce : ℕ) : 
  ounces = 12 → cost_per_ounce = 7 → ounces * cost_per_ounce = 84 := by
  sorry

end NUMINAMATH_CALUDE_cranberry_juice_cost_l3751_375136


namespace NUMINAMATH_CALUDE_charlies_bus_ride_l3751_375160

theorem charlies_bus_ride (oscars_ride : ℝ) (difference : ℝ) :
  oscars_ride = 0.75 →
  oscars_ride = difference + charlies_ride →
  difference = 0.5 →
  charlies_ride = 0.25 :=
by
  sorry

end NUMINAMATH_CALUDE_charlies_bus_ride_l3751_375160


namespace NUMINAMATH_CALUDE_permutation_cover_iff_m_gt_half_n_l3751_375112

/-- A permutation of the set {1, ..., n} -/
def Permutation (n : ℕ) := { f : Fin n → Fin n // Function.Bijective f }

/-- Two permutations have common points if they agree on at least one element -/
def have_common_points {n : ℕ} (f g : Permutation n) : Prop :=
  ∃ k : Fin n, f.val k = g.val k

/-- The main theorem: m permutations cover all permutations iff m > n/2 -/
theorem permutation_cover_iff_m_gt_half_n (n m : ℕ) :
  (∃ (fs : Fin m → Permutation n), ∀ f : Permutation n, ∃ i : Fin m, have_common_points f (fs i)) ↔
  m > n / 2 := by sorry

end NUMINAMATH_CALUDE_permutation_cover_iff_m_gt_half_n_l3751_375112


namespace NUMINAMATH_CALUDE_pencil_distribution_l3751_375137

/-- Given a total number of pencils and students, calculate the number of pencils per student -/
def pencils_per_student (total_pencils : ℕ) (total_students : ℕ) : ℕ :=
  total_pencils / total_students

theorem pencil_distribution (total_pencils : ℕ) (total_students : ℕ) 
  (h1 : total_pencils = 195)
  (h2 : total_students = 65) :
  pencils_per_student total_pencils total_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3751_375137


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3751_375101

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -3; 0, 3, -1; -1, 3, 2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -1, 0; 2, 1, -2; 3, 0, 1]
def c : ℤ := 2

theorem matrix_multiplication_result :
  c • (A * B) = !![(-14:ℤ), -4, -6; 6, 6, -14; 22, 8, -8] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3751_375101


namespace NUMINAMATH_CALUDE_johns_allowance_l3751_375180

theorem johns_allowance (A : ℚ) : 
  (A > 0) →
  ((4 / 15 : ℚ) * A = 92 / 100) →
  A = 345 / 100 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_l3751_375180


namespace NUMINAMATH_CALUDE_earthworm_catches_centipede_l3751_375183

/-- The time (in minutes) it takes for an earthworm to catch up with a centipede given their speeds and initial distance -/
def catch_up_time (centipede_speed earthworm_speed initial_distance : ℚ) : ℚ :=
  initial_distance / (earthworm_speed - centipede_speed)

/-- Theorem stating that under the given conditions, the earthworm catches up with the centipede in 24 minutes -/
theorem earthworm_catches_centipede :
  let centipede_speed : ℚ := 5 / 3  -- 5 meters in 3 minutes
  let earthworm_speed : ℚ := 5 / 2  -- 5 meters in 2 minutes
  let initial_distance : ℚ := 20    -- 20 meters ahead
  catch_up_time centipede_speed earthworm_speed initial_distance = 24 := by
  sorry

#eval catch_up_time (5/3) (5/2) 20

end NUMINAMATH_CALUDE_earthworm_catches_centipede_l3751_375183


namespace NUMINAMATH_CALUDE_store_price_reduction_l3751_375189

theorem store_price_reduction (original_price : ℝ) (h1 : original_price > 0) :
  let first_reduction := 0.09
  let final_price_ratio := 0.819
  let price_after_first := original_price * (1 - first_reduction)
  let second_reduction := 1 - (final_price_ratio / (1 - first_reduction))
  second_reduction = 0.181 := by sorry

end NUMINAMATH_CALUDE_store_price_reduction_l3751_375189


namespace NUMINAMATH_CALUDE_toy_cost_calculation_l3751_375145

def initial_money : ℕ := 57
def game_cost : ℕ := 27
def num_toys : ℕ := 5

theorem toy_cost_calculation (remaining_money : ℕ) (toy_cost : ℕ) : 
  remaining_money = initial_money - game_cost →
  remaining_money = num_toys * toy_cost →
  toy_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_calculation_l3751_375145


namespace NUMINAMATH_CALUDE_existence_of_polynomials_l3751_375141

-- Define the function f
def f (x y z : ℝ) : ℝ := x^2 + y^2 + z^2 + x*y*z

-- Define the theorem
theorem existence_of_polynomials :
  ∃ (a b c : ℝ → ℝ → ℝ → ℝ),
    (∀ x y z, f (a x y z) (b x y z) (c x y z) = f x y z) ∧
    (∃ x y z, (a x y z, b x y z, c x y z) ≠ (x, y, z) ∧
              (a x y z, b x y z, c x y z) ≠ (x, y, -z) ∧
              (a x y z, b x y z, c x y z) ≠ (x, -y, z) ∧
              (a x y z, b x y z, c x y z) ≠ (-x, y, z) ∧
              (a x y z, b x y z, c x y z) ≠ (x, -y, -z) ∧
              (a x y z, b x y z, c x y z) ≠ (-x, y, -z) ∧
              (a x y z, b x y z, c x y z) ≠ (-x, -y, z)) :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_polynomials_l3751_375141


namespace NUMINAMATH_CALUDE_circle_equation_l3751_375110

-- Define the point P
def P : ℝ × ℝ := (-2, 1)

-- Define the line y = x + 1
def line1 (x y : ℝ) : Prop := y = x + 1

-- Define the line 3x + 4y - 11 = 0
def line2 (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0

-- Define the circle C
def circle_C (c : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x - c.1)^2 + (y - c.2)^2 = 18

-- Define the symmetry condition
def symmetric_point (p1 p2 : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), line1 x y ∧ 
  (p1.1 + p2.1) / 2 = x ∧ 
  (p1.2 + p2.2) / 2 = y

-- Define the intersection condition
def intersects (c : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), 
  circle_C c A.1 A.2 ∧ 
  circle_C c B.1 B.2 ∧
  line2 A.1 A.2 ∧ 
  line2 B.1 B.2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36

-- Theorem statement
theorem circle_equation (c : ℝ × ℝ) :
  symmetric_point P c ∧ intersects c →
  ∀ (x y : ℝ), circle_C c x y ↔ x^2 + (y+1)^2 = 18 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3751_375110


namespace NUMINAMATH_CALUDE_pastry_sale_revenue_l3751_375158

/-- Calculates the total money made from selling discounted pastries. -/
theorem pastry_sale_revenue (cupcake_price cookie_price : ℚ)
  (cupcakes_sold cookies_sold : ℕ) : 
  cupcake_price = 3 ∧ cookie_price = 2 ∧ cupcakes_sold = 16 ∧ cookies_sold = 8 →
  (cupcake_price / 2 * cupcakes_sold + cookie_price / 2 * cookies_sold : ℚ) = 32 := by
  sorry

#check pastry_sale_revenue

end NUMINAMATH_CALUDE_pastry_sale_revenue_l3751_375158


namespace NUMINAMATH_CALUDE_well_cared_fish_lifespan_l3751_375123

/-- The average lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a dog relative to a hamster -/
def dog_lifespan_factor : ℝ := 4

/-- The additional lifespan of a well-cared fish compared to a dog, in years -/
def fish_extra_lifespan : ℝ := 2

/-- The number of months in a year -/
def months_per_year : ℝ := 12

/-- Theorem: A well-cared fish can live 144 months -/
theorem well_cared_fish_lifespan :
  hamster_lifespan * dog_lifespan_factor * months_per_year + fish_extra_lifespan * months_per_year = 144 :=
by sorry

end NUMINAMATH_CALUDE_well_cared_fish_lifespan_l3751_375123


namespace NUMINAMATH_CALUDE_balls_after_1729_steps_l3751_375132

/-- Represents the state of boxes in Lisa's ball-placing game -/
def BoxState := List Nat

/-- Converts a natural number to its septenary (base-7) representation -/
def toSeptenary (n : Nat) : List Nat :=
  sorry

/-- Calculates the sum of a list of natural numbers -/
def sum (l : List Nat) : Nat :=
  sorry

/-- Simulates Lisa's ball-placing process for a given number of steps -/
def simulateSteps (steps : Nat) : BoxState :=
  sorry

/-- Counts the total number of balls in a given box state -/
def countBalls (state : BoxState) : Nat :=
  sorry

/-- Theorem stating that the number of balls after 1729 steps
    is equal to the sum of digits in the septenary representation of 1729 -/
theorem balls_after_1729_steps :
  countBalls (simulateSteps 1729) = sum (toSeptenary 1729) :=
sorry

end NUMINAMATH_CALUDE_balls_after_1729_steps_l3751_375132


namespace NUMINAMATH_CALUDE_mixed_div_frac_example_l3751_375142

-- Define the division operation for mixed numbers and fractions
def mixedDivFrac (whole : ℤ) (num : ℕ) (den : ℕ) (frac_num : ℕ) (frac_den : ℕ) : ℚ :=
  (whole : ℚ) + (num : ℚ) / (den : ℚ) / ((frac_num : ℚ) / (frac_den : ℚ))

-- State the theorem
theorem mixed_div_frac_example : mixedDivFrac 2 1 4 3 5 = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_mixed_div_frac_example_l3751_375142


namespace NUMINAMATH_CALUDE_kendra_suvs_count_l3751_375111

/-- The number of SUVs Kendra saw in the afternoon -/
def afternoon_suvs : ℕ := 10

/-- The number of SUVs Kendra saw in the evening -/
def evening_suvs : ℕ := 5

/-- The total number of SUVs Kendra saw during her road trip -/
def total_suvs : ℕ := afternoon_suvs + evening_suvs

theorem kendra_suvs_count : total_suvs = 15 := by
  sorry

end NUMINAMATH_CALUDE_kendra_suvs_count_l3751_375111


namespace NUMINAMATH_CALUDE_savings_theorem_l3751_375179

def savings_problem (monday_savings : ℕ) (tuesday_savings : ℕ) (wednesday_savings : ℕ) : ℕ :=
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  total_savings / 2

theorem savings_theorem (monday_savings tuesday_savings wednesday_savings : ℕ) :
  monday_savings = 15 →
  tuesday_savings = 28 →
  wednesday_savings = 13 →
  savings_problem monday_savings tuesday_savings wednesday_savings = 28 := by
  sorry

end NUMINAMATH_CALUDE_savings_theorem_l3751_375179


namespace NUMINAMATH_CALUDE_sector_radius_gt_two_l3751_375174

theorem sector_radius_gt_two (R : ℝ) (l : ℝ) (h : R > 0) (h_l : l > 0) :
  (1/2 * l * R = 2 * R + l) → R > 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_gt_two_l3751_375174


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_84_l3751_375151

/-- Represents a quadrilateral ABCD -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- The length of a side of a quadrilateral -/
def side_length (q : Quadrilateral) (side : Fin 4) : ℝ := sorry

/-- The measure of an angle in a quadrilateral -/
def angle_measure (q : Quadrilateral) (vertex : Fin 4) : ℝ := sorry

/-- Whether a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_area_is_84 (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_ab : side_length q 0 = 5)
  (h_bc : side_length q 1 = 12)
  (h_cd : side_length q 2 = 13)
  (h_ad : side_length q 3 = 15)
  (h_angle_abc : angle_measure q 1 = 90) :
  area q = 84 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_84_l3751_375151


namespace NUMINAMATH_CALUDE_remainder_after_addition_l3751_375153

theorem remainder_after_addition (m : ℤ) (h : m % 5 = 2) : (m + 2535) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_addition_l3751_375153


namespace NUMINAMATH_CALUDE_complex_roots_distance_l3751_375122

/-- Given three complex numbers z₁, z₂, z₃ with |zⱼ| ≤ 1 for j = 1, 2, 3, 
    and w₁, w₂ being the roots of the equation 
    (z - z₁)(z - z₂) + (z - z₂)(z - z₃) + (z - z₃)(z - z₁) = 0,
    then for j = 1, 2, 3, min{|zⱼ - w₁|, |zⱼ - w₂|} ≤ 1. -/
theorem complex_roots_distance (z₁ z₂ z₃ w₁ w₂ : ℂ) 
  (h₁ : Complex.abs z₁ ≤ 1)
  (h₂ : Complex.abs z₂ ≤ 1)
  (h₃ : Complex.abs z₃ ≤ 1)
  (hw : (w₁ - z₁) * (w₁ - z₂) + (w₁ - z₂) * (w₁ - z₃) + (w₁ - z₃) * (w₁ - z₁) = 0 ∧
        (w₂ - z₁) * (w₂ - z₂) + (w₂ - z₂) * (w₂ - z₃) + (w₂ - z₃) * (w₂ - z₁) = 0) :
  (min (Complex.abs (z₁ - w₁)) (Complex.abs (z₁ - w₂)) ≤ 1) ∧
  (min (Complex.abs (z₂ - w₁)) (Complex.abs (z₂ - w₂)) ≤ 1) ∧
  (min (Complex.abs (z₃ - w₁)) (Complex.abs (z₃ - w₂)) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_distance_l3751_375122


namespace NUMINAMATH_CALUDE_binary_linear_equation_problem_l3751_375149

theorem binary_linear_equation_problem (m n : ℤ) : 
  (3 * m - 2 * n = -2) → 
  (3 * (m + 405) - 2 * (n - 405) = 2023) := by
  sorry

end NUMINAMATH_CALUDE_binary_linear_equation_problem_l3751_375149


namespace NUMINAMATH_CALUDE_square_area_proof_l3751_375163

/-- Given a rectangle, circle, and square with specific relationships, 
    prove that the area of the square is 3600 square units. -/
theorem square_area_proof (rectangle_length rectangle_breadth circle_radius square_side : ℝ) 
  (h1 : rectangle_length = (2 / 5) * circle_radius)
  (h2 : circle_radius = square_side)
  (h3 : rectangle_length * rectangle_breadth = 240)
  (h4 : rectangle_breadth = 10) : 
  square_side ^ 2 = 3600 := by
  sorry

#check square_area_proof

end NUMINAMATH_CALUDE_square_area_proof_l3751_375163


namespace NUMINAMATH_CALUDE_arithmetic_expressions_evaluation_l3751_375105

theorem arithmetic_expressions_evaluation :
  (2 * (-1)^3 - (-2)^2 / 4 + 10 = 7) ∧
  (abs (-3) - (-6 + 4) / (-1/2)^3 + (-1)^2013 = -14) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_evaluation_l3751_375105


namespace NUMINAMATH_CALUDE_total_purchase_ways_l3751_375148

/-- The number of oreo flavors --/
def oreo_flavors : ℕ := 5

/-- The number of milk flavors --/
def milk_flavors : ℕ := 3

/-- The total number of product types --/
def total_products : ℕ := oreo_flavors + milk_flavors

/-- The number of products they must purchase collectively --/
def purchase_count : ℕ := 3

/-- Represents the ways Alpha can choose items without repetition --/
def alpha_choices (k : ℕ) : ℕ := Nat.choose total_products k

/-- Represents the ways Beta can choose oreos with possible repetition --/
def beta_choices (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k = 1 then oreo_flavors
  else if k = 2 then Nat.choose oreo_flavors 2 + oreo_flavors
  else Nat.choose oreo_flavors 3 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors

/-- The total number of ways Alpha and Beta can purchase 3 products collectively --/
def total_ways : ℕ := 
  alpha_choices 3 +
  alpha_choices 2 * beta_choices 1 +
  alpha_choices 1 * beta_choices 2 +
  beta_choices 3

theorem total_purchase_ways : total_ways = 351 := by sorry

end NUMINAMATH_CALUDE_total_purchase_ways_l3751_375148


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3751_375182

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 = 3*x + 4) → (x = Real.sqrt (3*x + 4)) ∧
  ¬(∀ x : ℝ, (x = Real.sqrt (3*x + 4)) → (x^2 = 3*x + 4)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3751_375182


namespace NUMINAMATH_CALUDE_phi_value_l3751_375197

theorem phi_value : ∃ (Φ : ℕ), 504 / Φ = 40 + 3 * Φ ∧ 0 ≤ Φ ∧ Φ ≤ 9 ∧ Φ = 8 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l3751_375197


namespace NUMINAMATH_CALUDE_number_145_column_l3751_375155

/-- Represents the columns in the arrangement --/
inductive Column
| A | B | C | D | E | F

/-- The function that determines the column for a given position in the sequence --/
def column_for_position (n : ℕ) : Column :=
  match n % 11 with
  | 1 => Column.A
  | 2 => Column.B
  | 3 => Column.C
  | 4 => Column.D
  | 5 => Column.E
  | 6 => Column.F
  | 7 => Column.E
  | 8 => Column.D
  | 9 => Column.C
  | 10 => Column.B
  | 0 => Column.A
  | _ => Column.A  -- This case should never occur, but Lean requires it for completeness

theorem number_145_column :
  column_for_position 143 = Column.A :=
sorry

end NUMINAMATH_CALUDE_number_145_column_l3751_375155


namespace NUMINAMATH_CALUDE_hyperbola_condition_ellipse_x_major_condition_l3751_375102

-- Define the curve C
def C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - t) + y^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) := ∀ (x y : ℝ), (x, y) ∈ C t → (4 - t) * (t - 1) < 0

-- Define what it means for C to be an ellipse with major axis on x-axis
def is_ellipse_x_major (t : ℝ) := ∀ (x y : ℝ), (x, y) ∈ C t → (4 - t) > (t - 1) ∧ (t - 1) > 0

-- Theorem statements
theorem hyperbola_condition (t : ℝ) :
  is_hyperbola t → t > 4 ∨ t < 1 := by sorry

theorem ellipse_x_major_condition (t : ℝ) :
  is_ellipse_x_major t → 1 < t ∧ t < 5/2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_ellipse_x_major_condition_l3751_375102


namespace NUMINAMATH_CALUDE_batsman_average_l3751_375109

/-- Calculates the new average of a batsman after the 17th inning -/
def newAverage (prevAverage : ℚ) (inningScore : ℕ) (numInnings : ℕ) : ℚ :=
  (prevAverage * (numInnings - 1) + inningScore) / numInnings

/-- Proves that the batsman's new average is 39 runs -/
theorem batsman_average : 
  ∀ (prevAverage : ℚ),
  newAverage prevAverage 87 17 = prevAverage + 3 →
  newAverage prevAverage 87 17 = 39 := by
    sorry

end NUMINAMATH_CALUDE_batsman_average_l3751_375109


namespace NUMINAMATH_CALUDE_polynomial_real_root_l3751_375120

/-- The polynomial in question -/
def P (a x : ℝ) : ℝ := x^4 + a*x^3 - 2*x^2 + a*x + 2

/-- The theorem stating the condition for the polynomial to have at least one real root -/
theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, P a x = 0) ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l3751_375120


namespace NUMINAMATH_CALUDE_candy_comparison_l3751_375164

/-- Represents a person with their candy bags -/
structure Person where
  name : String
  bags : List Nat

/-- Calculates the total candy for a person -/
def totalCandy (p : Person) : Nat :=
  p.bags.sum

theorem candy_comparison (sandra roger emily : Person)
  (h_sandra : sandra.bags = [6, 6])
  (h_roger : roger.bags = [11, 3])
  (h_emily : emily.bags = [4, 7, 5]) :
  totalCandy emily > totalCandy roger ∧
  totalCandy roger > totalCandy sandra ∧
  totalCandy sandra = 12 := by
  sorry

#eval totalCandy { name := "Sandra", bags := [6, 6] }
#eval totalCandy { name := "Roger", bags := [11, 3] }
#eval totalCandy { name := "Emily", bags := [4, 7, 5] }

end NUMINAMATH_CALUDE_candy_comparison_l3751_375164


namespace NUMINAMATH_CALUDE_x_minus_y_values_l3751_375195

theorem x_minus_y_values (x y : ℤ) (hx : x = -3) (hy : |y| = 2) : 
  x - y = -5 ∨ x - y = -1 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l3751_375195


namespace NUMINAMATH_CALUDE_sin_product_identity_l3751_375177

theorem sin_product_identity :
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (72 * π / 180) * Real.sin (84 * π / 180) =
  (1 / 8) * (1 + Real.cos (24 * π / 180)) := by
sorry

end NUMINAMATH_CALUDE_sin_product_identity_l3751_375177


namespace NUMINAMATH_CALUDE_paint_usage_l3751_375171

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ) 
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1/4)
  (h3 : second_week_fraction = 1/6) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 135 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_l3751_375171


namespace NUMINAMATH_CALUDE_product_local_abs_value_4_in_564823_l3751_375115

/-- The local value of a digit in a number -/
def local_value (n : ℕ) (d : ℕ) (p : ℕ) : ℕ := d * (10 ^ p)

/-- The absolute value of a natural number -/
def abs_nat (n : ℕ) : ℕ := n

theorem product_local_abs_value_4_in_564823 :
  let n : ℕ := 564823
  let d : ℕ := 4
  let p : ℕ := 4  -- position of 4 in 564823 (0-indexed from right)
  (local_value n d p) * (abs_nat d) = 160000 := by sorry

end NUMINAMATH_CALUDE_product_local_abs_value_4_in_564823_l3751_375115


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3751_375167

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  ∀ z w : ℝ, z > 0 ∧ w > 0 ∧ 2/z + 1/w = 1 → x + 2*y ≤ z + 2*w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2/a + 1/b = 1 ∧ a + 2*b = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3751_375167


namespace NUMINAMATH_CALUDE_harvest_duration_l3751_375154

def total_earnings : ℕ := 1216
def weekly_earnings : ℕ := 16

theorem harvest_duration :
  total_earnings / weekly_earnings = 76 :=
sorry

end NUMINAMATH_CALUDE_harvest_duration_l3751_375154


namespace NUMINAMATH_CALUDE_twitter_to_insta_fb_ratio_l3751_375188

/-- Represents the number of followers on each social media platform -/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Conditions for Malcolm's social media followers -/
def malcolm_followers (f : Followers) : Prop :=
  f.instagram = 240 ∧
  f.facebook = 500 ∧
  f.tiktok = 3 * f.twitter ∧
  f.youtube = f.tiktok + 510 ∧
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube = 3840

/-- Theorem stating the ratio of Twitter followers to Instagram and Facebook followers -/
theorem twitter_to_insta_fb_ratio (f : Followers) 
  (h : malcolm_followers f) : 
  f.twitter * 2 = f.instagram + f.facebook := by
  sorry

end NUMINAMATH_CALUDE_twitter_to_insta_fb_ratio_l3751_375188


namespace NUMINAMATH_CALUDE_total_soaking_time_with_ink_l3751_375157

/-- Represents the soaking time for different types of stains -/
def SoakingTime : Type := Nat → Nat

/-- Calculates the total soaking time for a piece of clothing -/
def totalSoakingTime (stainCounts : List Nat) (soakingTimes : List Nat) : Nat :=
  List.sum (List.zipWith (· * ·) stainCounts soakingTimes)

/-- Calculates the additional time needed for ink stains -/
def additionalInkTime (inkStainCount : Nat) (extraTimePerInkStain : Nat) : Nat :=
  inkStainCount * extraTimePerInkStain

theorem total_soaking_time_with_ink (shirtStainCounts shirtSoakingTimes
                                     pantsStainCounts pantsSoakingTimes
                                     socksStainCounts socksSoakingTimes : List Nat)
                                    (inkStainCount extraTimePerInkStain : Nat) :
  totalSoakingTime shirtStainCounts shirtSoakingTimes +
  totalSoakingTime pantsStainCounts pantsSoakingTimes +
  totalSoakingTime socksStainCounts socksSoakingTimes +
  additionalInkTime inkStainCount extraTimePerInkStain = 54 :=
by
  sorry

#check total_soaking_time_with_ink

end NUMINAMATH_CALUDE_total_soaking_time_with_ink_l3751_375157


namespace NUMINAMATH_CALUDE_total_cutlery_after_addition_l3751_375121

/-- Represents the number of each type of cutlery in a drawer -/
structure Cutlery :=
  (forks : ℕ)
  (knives : ℕ)
  (spoons : ℕ)
  (teaspoons : ℕ)

/-- Calculates the total number of cutlery pieces -/
def totalCutlery (c : Cutlery) : ℕ :=
  c.forks + c.knives + c.spoons + c.teaspoons

/-- Represents the initial state of the cutlery drawer -/
def initialCutlery : Cutlery :=
  { forks := 6
  , knives := 6 + 9
  , spoons := 2 * (6 + 9)
  , teaspoons := 6 / 2 }

/-- Represents the final state of the cutlery drawer after adding 2 of each type -/
def finalCutlery : Cutlery :=
  { forks := initialCutlery.forks + 2
  , knives := initialCutlery.knives + 2
  , spoons := initialCutlery.spoons + 2
  , teaspoons := initialCutlery.teaspoons + 2 }

/-- Theorem: The total number of cutlery pieces after adding 2 of each type is 62 -/
theorem total_cutlery_after_addition : totalCutlery finalCutlery = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_cutlery_after_addition_l3751_375121
