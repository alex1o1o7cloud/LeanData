import Mathlib

namespace NUMINAMATH_CALUDE_cheapest_pie_eggs_butter_cost_l591_59190

/-- The cost of eggs and butter for the cheapest pie --/
def cost_eggs_butter : ℚ := sorry

/-- The cost of flour --/
def flour_cost : ℚ := 2

/-- The cost of sugar --/
def sugar_cost : ℚ := 1

/-- The cost of blueberries per 8 oz container --/
def blueberry_cost_per_container : ℚ := 2.25

/-- The cost of cherries for a four-pound bag --/
def cherry_cost_per_bag : ℚ := 14

/-- The amount of blueberries needed in pounds --/
def blueberry_amount : ℚ := 3

/-- The amount of cherries needed in pounds --/
def cherry_amount : ℚ := 4

/-- The total price to make the cheapest pie --/
def total_price : ℚ := 18

/-- Ounces in a pound --/
def oz_per_pound : ℚ := 16

/-- Ounces per container of blueberries --/
def oz_per_container : ℚ := 8

theorem cheapest_pie_eggs_butter_cost :
  let blueberry_containers := (blueberry_amount * oz_per_pound) / oz_per_container
  let blueberry_cost := blueberry_containers * blueberry_cost_per_container
  let cherry_cost := cherry_cost_per_bag
  let base_cost := flour_cost + sugar_cost
  let blueberry_pie_cost := blueberry_cost + base_cost
  let cherry_pie_cost := cherry_cost + base_cost
  let cheapest_pie_cost := min blueberry_pie_cost cherry_pie_cost
  cost_eggs_butter = total_price - cheapest_pie_cost := by sorry

end NUMINAMATH_CALUDE_cheapest_pie_eggs_butter_cost_l591_59190


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l591_59191

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence, if a₂ * a₈ = 16, then a₁ * a₉ = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 2 * a 8 = 16) : a 1 * a 9 = 16 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l591_59191


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l591_59163

/-- Represents the number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 18 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l591_59163


namespace NUMINAMATH_CALUDE_power_sum_inequality_l591_59121

theorem power_sum_inequality (a b c : ℝ) (n : ℕ) (p q r : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hpqr : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l591_59121


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l591_59100

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℤ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def to_rat (d : RepeatingDecimal) : ℚ :=
  d.whole + (d.repeating : ℚ) / (999 : ℚ)

/-- The repeating decimal 0.428428... -/
def d1 : RepeatingDecimal := ⟨0, 428⟩

/-- The repeating decimal 2.857857... -/
def d2 : RepeatingDecimal := ⟨2, 857⟩

theorem repeating_decimal_fraction :
  (to_rat d1) / (to_rat d2) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l591_59100


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l591_59147

/-- Represents the quantity of an ingredient on a given day -/
structure IngredientQuantity where
  day1 : Float
  day7 : Float

/-- Represents the price of an ingredient -/
structure IngredientPrice where
  value : Float
  unit : String

/-- Represents an ingredient with its quantities and price -/
structure Ingredient where
  name : String
  quantity : IngredientQuantity
  price : IngredientPrice
  unit : String

def ingredients : List Ingredient := [
  { name := "Baking powder",
    quantity := { day1 := 12, day7 := 6 },
    price := { value := 3, unit := "per pound" },
    unit := "lbs" },
  { name := "Flour",
    quantity := { day1 := 6, day7 := 3.5 },
    price := { value := 1.5, unit := "per pound" },
    unit := "kg" },
  { name := "Sugar",
    quantity := { day1 := 20, day7 := 15 },
    price := { value := 0.5, unit := "per pound" },
    unit := "lbs" },
  { name := "Chocolate chips",
    quantity := { day1 := 5000, day7 := 1500 },
    price := { value := 0.015, unit := "per gram" },
    unit := "g" }
]

def kgToPounds : Float := 2.20462
def gToPounds : Float := 0.00220462

def calculateTotalCost (ingredients : List Ingredient) : Float :=
  sorry

theorem total_cost_is_correct :
  calculateTotalCost ingredients = 81.27 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l591_59147


namespace NUMINAMATH_CALUDE_cyclist_distance_l591_59183

/-- The distance traveled by a cyclist given the conditions of the problem -/
theorem cyclist_distance (distance_AB : ℝ) (pedestrian_speed : ℝ) 
  (h1 : distance_AB = 5)
  (h2 : pedestrian_speed > 0) : 
  let cyclist_speed := 2 * pedestrian_speed
  let time := distance_AB / pedestrian_speed
  cyclist_speed * time = 10 := by sorry

end NUMINAMATH_CALUDE_cyclist_distance_l591_59183


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l591_59177

theorem polynomial_root_problem (a b : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 - 3 * Complex.I : ℂ) ^ 3 + a * (2 - 3 * Complex.I : ℂ) ^ 2 - 2 * (2 - 3 * Complex.I : ℂ) + b = 0 →
  a = -1/4 ∧ b = 195/4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l591_59177


namespace NUMINAMATH_CALUDE_fill_time_correct_l591_59103

/-- The time in seconds for eight faucets to fill a 30-gallon tub, given that four faucets can fill a 120-gallon tub in 8 minutes -/
def fill_time : ℝ := 60

/-- The volume of the large tub in gallons -/
def large_tub_volume : ℝ := 120

/-- The volume of the small tub in gallons -/
def small_tub_volume : ℝ := 30

/-- The time in minutes for four faucets to fill the large tub -/
def large_tub_fill_time : ℝ := 8

/-- The number of faucets used to fill the large tub -/
def large_tub_faucets : ℕ := 4

/-- The number of faucets used to fill the small tub -/
def small_tub_faucets : ℕ := 8

/-- Conversion factor from minutes to seconds -/
def minutes_to_seconds : ℝ := 60

theorem fill_time_correct : fill_time = 
  (small_tub_volume / large_tub_volume) * 
  (large_tub_faucets / small_tub_faucets) * 
  large_tub_fill_time * 
  minutes_to_seconds := by
  sorry

end NUMINAMATH_CALUDE_fill_time_correct_l591_59103


namespace NUMINAMATH_CALUDE_function_non_positive_on_interval_l591_59165

theorem function_non_positive_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc 0 1, a^2 * x - 2*a + 1 ≤ 0) ↔ a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_function_non_positive_on_interval_l591_59165


namespace NUMINAMATH_CALUDE_base7_subtraction_theorem_l591_59185

/-- Represents a number in base 7 --/
def Base7 : Type := ℕ

/-- Converts a base 7 number to its decimal representation --/
def to_decimal (n : Base7) : ℕ := sorry

/-- Converts a decimal number to its base 7 representation --/
def from_decimal (n : ℕ) : Base7 := sorry

/-- Subtracts two base 7 numbers --/
def base7_subtract (a b : Base7) : Base7 := sorry

theorem base7_subtraction_theorem :
  base7_subtract (from_decimal 4321) (from_decimal 1234) = from_decimal 3054 := by
  sorry

end NUMINAMATH_CALUDE_base7_subtraction_theorem_l591_59185


namespace NUMINAMATH_CALUDE_min_knight_liar_pairs_l591_59188

/-- Represents the type of people on the island -/
inductive Person
| Knight
| Liar

/-- Represents a friendship between two people -/
structure Friendship where
  person1 : Person
  person2 : Person

/-- The total number of people on the island -/
def total_people : Nat := 200

/-- The number of knights on the island -/
def num_knights : Nat := 100

/-- The number of liars on the island -/
def num_liars : Nat := 100

/-- The number of people who said "All my friends are knights" -/
def num_all_knight_claims : Nat := 100

/-- The number of people who said "All my friends are liars" -/
def num_all_liar_claims : Nat := 100

/-- Definition: Each person has at least one friend -/
axiom has_friend (p : Person) : ∃ (f : Friendship), f.person1 = p ∨ f.person2 = p

/-- Definition: Knights always tell the truth -/
axiom knight_truth (k : Person) (claim : Prop) : k = Person.Knight → (claim ↔ true)

/-- Definition: Liars always lie -/
axiom liar_lie (l : Person) (claim : Prop) : l = Person.Liar → (claim ↔ false)

/-- The main theorem to be proved -/
theorem min_knight_liar_pairs :
  ∃ (friendships : List Friendship),
    (∀ f ∈ friendships, (f.person1 = Person.Knight ∧ f.person2 = Person.Liar) ∨
                        (f.person1 = Person.Liar ∧ f.person2 = Person.Knight)) ∧
    friendships.length = 50 ∧
    (∀ friendships' : List Friendship,
      (∀ f' ∈ friendships', (f'.person1 = Person.Knight ∧ f'.person2 = Person.Liar) ∨
                            (f'.person1 = Person.Liar ∧ f'.person2 = Person.Knight)) →
      friendships'.length ≥ 50) := by
  sorry

end NUMINAMATH_CALUDE_min_knight_liar_pairs_l591_59188


namespace NUMINAMATH_CALUDE_triangle_side_expression_zero_l591_59176

theorem triangle_side_expression_zero (a b c : ℝ) 
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  |a - b - c| - |c - a + b| = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_expression_zero_l591_59176


namespace NUMINAMATH_CALUDE_average_problem_l591_59160

theorem average_problem (x : ℝ) : (1 + 3 + x) / 3 = 3 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l591_59160


namespace NUMINAMATH_CALUDE_school_students_count_l591_59193

theorem school_students_count (total : ℕ) (difference : ℕ) (boys : ℕ) : 
  total = 650 →
  difference = 106 →
  boys + (boys + difference) = total →
  boys = 272 := by
sorry

end NUMINAMATH_CALUDE_school_students_count_l591_59193


namespace NUMINAMATH_CALUDE_exponential_functional_equation_l591_59197

theorem exponential_functional_equation 
  (a : ℝ) (ha : a > 0 ∧ a ≠ 1) : 
  ∀ x y : ℝ, (fun x => a^x) x * (fun x => a^x) y = (fun x => a^x) (x + y) :=
by sorry

end NUMINAMATH_CALUDE_exponential_functional_equation_l591_59197


namespace NUMINAMATH_CALUDE_train_distance_l591_59127

/-- Calculates the distance traveled by a train given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a train traveling at 7 m/s for 6 seconds covers 42 meters -/
theorem train_distance : distance_traveled 7 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l591_59127


namespace NUMINAMATH_CALUDE_base_c_problem_l591_59170

/-- Representation of a number in base c -/
def baseC (n : ℕ) (c : ℕ) : ℕ → ℕ
| 0 => n % c
| i + 1 => baseC (n / c) c i

/-- Given that in base c, 33_c squared equals 1201_c, prove that c = 10 -/
theorem base_c_problem (c : ℕ) (h : c > 1) :
  (baseC 33 c 1 * c + baseC 33 c 0)^2 = 
  baseC 1201 c 3 * c^3 + baseC 1201 c 2 * c^2 + baseC 1201 c 1 * c + baseC 1201 c 0 →
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_base_c_problem_l591_59170


namespace NUMINAMATH_CALUDE_probability_of_two_primes_l591_59166

/-- A function that determines if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of integers from 1 to 30 inclusive -/
def integerSet : Finset ℕ := sorry

/-- The set of prime numbers from 1 to 30 inclusive -/
def primeSet : Finset ℕ := sorry

/-- The number of ways to choose 2 items from a set of size n -/
def choose (n : ℕ) (k : ℕ) : ℕ := sorry

theorem probability_of_two_primes :
  (choose (Finset.card primeSet) 2 : ℚ) / (choose (Finset.card integerSet) 2) = 10 / 87 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_two_primes_l591_59166


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l591_59143

/-- The repeating decimal 0.464646... expressed as a real number -/
def repeating_decimal : ℚ := 46 / 99

/-- The theorem stating that the repeating decimal 0.464646... is equal to 46/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 46 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l591_59143


namespace NUMINAMATH_CALUDE_equation_has_one_solution_l591_59106

theorem equation_has_one_solution :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_has_one_solution_l591_59106


namespace NUMINAMATH_CALUDE_rational_function_characterization_l591_59136

theorem rational_function_characterization (f : ℚ → ℚ) 
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 := by sorry

end NUMINAMATH_CALUDE_rational_function_characterization_l591_59136


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l591_59167

theorem no_solution_to_inequalities :
  ¬∃ (x y z t : ℝ),
    (abs x < abs (y - z + t)) ∧
    (abs y < abs (x - z + t)) ∧
    (abs z < abs (x - y + t)) ∧
    (abs t < abs (x - y + z)) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l591_59167


namespace NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l591_59118

theorem video_recorder_wholesale_cost :
  ∀ (wholesale_cost : ℝ),
    let retail_price := wholesale_cost * 1.2
    let employee_price := retail_price * 0.8
    employee_price = 192 →
    wholesale_cost = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l591_59118


namespace NUMINAMATH_CALUDE_three_coins_same_probability_l591_59156

def coin_flip := Bool

def total_outcomes (n : ℕ) : ℕ := 2^n

def favorable_outcomes (n : ℕ) : ℕ := 2 * 2^(n - 3)

theorem three_coins_same_probability (n : ℕ) (h : n = 6) :
  (favorable_outcomes n : ℚ) / (total_outcomes n : ℚ) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_three_coins_same_probability_l591_59156


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l591_59181

/-- A rectangle with perimeter 14 cm and area 12 square cm has a diagonal of length 5 cm -/
theorem rectangle_diagonal (l w : ℝ) : 
  (2 * l + 2 * w = 14) →  -- Perimeter condition
  (l * w = 12) →          -- Area condition
  Real.sqrt (l^2 + w^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l591_59181


namespace NUMINAMATH_CALUDE_semicircle_radius_l591_59133

theorem semicircle_radius (p : ℝ) (h : p = 108) : 
  ∃ r : ℝ, p = r * (Real.pi + 2) ∧ r = p / (Real.pi + 2) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l591_59133


namespace NUMINAMATH_CALUDE_meaningful_sqrt_range_l591_59110

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_range_l591_59110


namespace NUMINAMATH_CALUDE_winter_ball_attendance_l591_59128

theorem winter_ball_attendance 
  (total_students : ℕ) 
  (ball_attendees : ℕ) 
  (girls : ℕ) 
  (boys : ℕ) : 
  total_students = 1500 →
  ball_attendees = 900 →
  girls + boys = total_students →
  (3 * girls / 4 : ℚ) + (2 * boys / 3 : ℚ) = ball_attendees →
  3 * girls / 4 = 900 :=
by sorry

end NUMINAMATH_CALUDE_winter_ball_attendance_l591_59128


namespace NUMINAMATH_CALUDE_truck_driver_net_pay_rate_l591_59101

/-- Calculates the net rate of pay for a truck driver given specific conditions --/
theorem truck_driver_net_pay_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (diesel_cost : ℝ)
  (h_travel_time : travel_time = 3)
  (h_speed : speed = 50)
  (h_fuel_efficiency : fuel_efficiency = 25)
  (h_pay_rate : pay_rate = 0.60)
  (h_diesel_cost : diesel_cost = 2.50)
  : (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * diesel_cost) / travel_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_truck_driver_net_pay_rate_l591_59101


namespace NUMINAMATH_CALUDE_platform_length_l591_59122

/-- The length of a platform that a train crosses, given the train's length and crossing times. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) :
  train_length = 300 →
  platform_crossing_time = 39 →
  pole_crossing_time = 18 →
  ∃ platform_length : ℝ,
    platform_length = 350 ∧
    (train_length + platform_length) / platform_crossing_time = train_length / pole_crossing_time :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l591_59122


namespace NUMINAMATH_CALUDE_union_complement_equality_l591_59149

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equality : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l591_59149


namespace NUMINAMATH_CALUDE_school_merger_ratio_l591_59116

theorem school_merger_ratio (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (8 * a) / (7 * a) = 8 / 7 →
  (30 * b) / (31 * b) = 30 / 31 →
  (8 * a + 30 * b) / (7 * a + 31 * b) = 27 / 26 →
  (8 * a + 7 * a) / (30 * b + 31 * b) = 27 / 26 :=
by sorry

end NUMINAMATH_CALUDE_school_merger_ratio_l591_59116


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l591_59119

theorem hyperbola_focal_distance (P F₁ F₂ : ℝ × ℝ) :
  (∃ x y : ℝ, P = (x, y) ∧ x^2 / 64 - y^2 / 36 = 1) →  -- P is on the hyperbola
  (∃ c : ℝ, c > 0 ∧ F₁ = (-c, 0) ∧ F₂ = (c, 0)) →  -- F₁ and F₂ are foci
  ‖P - F₁‖ = 15 →  -- |PF₁| = 15
  ‖P - F₂‖ = 31 :=  -- |PF₂| = 31
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l591_59119


namespace NUMINAMATH_CALUDE_louisa_average_speed_l591_59129

/-- Proves that given the conditions of Louisa's travel, her average speed was 60 miles per hour -/
theorem louisa_average_speed :
  ∀ (v : ℝ), -- v represents the average speed in miles per hour
  v > 0 → -- speed is positive
  ∃ (t : ℝ), -- t represents the time for the 240-mile trip
  t > 0 → -- time is positive
  240 = v * t ∧ -- equation for the first day's travel
  420 = v * (t + 3) → -- equation for the second day's travel
  v = 60 := by
sorry

end NUMINAMATH_CALUDE_louisa_average_speed_l591_59129


namespace NUMINAMATH_CALUDE_three_two_zero_zero_properties_l591_59199

/-- Represents a number with its decimal representation -/
structure DecimalNumber where
  value : ℝ
  representation : String

/-- Counts the number of significant figures in a decimal representation -/
def countSignificantFigures (n : DecimalNumber) : ℕ :=
  sorry

/-- Determines the precision of a decimal representation -/
def getPrecision (n : DecimalNumber) : String :=
  sorry

/-- The main theorem about the number 3200 -/
theorem three_two_zero_zero_properties :
  let n : DecimalNumber := ⟨3200, "0.320"⟩
  countSignificantFigures n = 3 ∧ getPrecision n = "thousandth" := by
  sorry

end NUMINAMATH_CALUDE_three_two_zero_zero_properties_l591_59199


namespace NUMINAMATH_CALUDE_solve_linear_equation_l591_59169

theorem solve_linear_equation :
  ∀ x : ℚ, 3 * x + 8 = -4 * x - 16 → x = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l591_59169


namespace NUMINAMATH_CALUDE_vehicle_overtake_problem_l591_59150

/-- The initial distance between two vehicles, where one overtakes the other --/
def initial_distance (speed_x speed_y : ℝ) (time : ℝ) (final_distance : ℝ) : ℝ :=
  (speed_y - speed_x) * time - final_distance

theorem vehicle_overtake_problem :
  let speed_x : ℝ := 36
  let speed_y : ℝ := 45
  let time : ℝ := 5
  let final_distance : ℝ := 23
  initial_distance speed_x speed_y time final_distance = 22 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_overtake_problem_l591_59150


namespace NUMINAMATH_CALUDE_monotonic_increasing_iff_m_ge_four_thirds_l591_59137

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

-- State the theorem
theorem monotonic_increasing_iff_m_ge_four_thirds (m : ℝ) :
  (∀ x : ℝ, Monotone (f m)) ↔ m ≥ 4/3 := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_iff_m_ge_four_thirds_l591_59137


namespace NUMINAMATH_CALUDE_cupcake_dozens_correct_l591_59115

/-- The number of dozens of cupcakes Jose needs to make -/
def cupcake_dozens : ℕ := 3

/-- The number of tablespoons of lemon juice needed for one dozen cupcakes -/
def juice_per_dozen : ℕ := 12

/-- The number of tablespoons of lemon juice provided by one lemon -/
def juice_per_lemon : ℕ := 4

/-- The number of lemons Jose needs -/
def lemons_needed : ℕ := 9

/-- Theorem stating that the number of dozens of cupcakes Jose needs to make is correct -/
theorem cupcake_dozens_correct : 
  cupcake_dozens = (lemons_needed * juice_per_lemon) / juice_per_dozen :=
by sorry

end NUMINAMATH_CALUDE_cupcake_dozens_correct_l591_59115


namespace NUMINAMATH_CALUDE_a_equals_two_l591_59139

theorem a_equals_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x : ℝ, a^(2*x - 4) ≤ 2^(x^2 - 2*x)) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_l591_59139


namespace NUMINAMATH_CALUDE_y_value_l591_59168

theorem y_value (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 4 / y) : y = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l591_59168


namespace NUMINAMATH_CALUDE_point_on_curve_l591_59180

/-- Curve C is defined by the parametric equations x = 4t² and y = t -/
def curve_C (t : ℝ) : ℝ × ℝ := (4 * t^2, t)

/-- Point P has coordinates (m, 2) -/
def point_P (m : ℝ) : ℝ × ℝ := (m, 2)

/-- Theorem: If point P(m, 2) lies on curve C, then m = 16 -/
theorem point_on_curve (m : ℝ) : 
  (∃ t : ℝ, curve_C t = point_P m) → m = 16 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_l591_59180


namespace NUMINAMATH_CALUDE_new_bus_distance_l591_59144

theorem new_bus_distance (old_distance : ℝ) (percentage_increase : ℝ) (new_distance : ℝ) : 
  old_distance = 300 →
  percentage_increase = 0.30 →
  new_distance = old_distance * (1 + percentage_increase) →
  new_distance = 390 := by
sorry

end NUMINAMATH_CALUDE_new_bus_distance_l591_59144


namespace NUMINAMATH_CALUDE_box_dimensions_l591_59195

/-- A box with a square base and specific ribbon tying properties has dimensions 22 cm × 22 cm × 11 cm -/
theorem box_dimensions (s : ℝ) (b : ℝ) :
  s > 0 →
  6 * s + b = 156 →
  7 * s + b = 178 →
  s = 22 ∧ s / 2 = 11 :=
by sorry

end NUMINAMATH_CALUDE_box_dimensions_l591_59195


namespace NUMINAMATH_CALUDE_ellipse_focus_k_value_l591_59194

/-- Given an ellipse 3kx^2 + y^2 = 1 with focus F(2,0), prove that k = 1/15 -/
theorem ellipse_focus_k_value (k : ℝ) : 
  (∃ (x y : ℝ), 3 * k * x^2 + y^2 = 1) →  -- Ellipse equation
  (2 : ℝ)^2 = (1 / (3 * k)) - 1 →  -- Focus condition (c^2 = a^2 - b^2)
  k = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focus_k_value_l591_59194


namespace NUMINAMATH_CALUDE_largest_solution_logarithm_equation_l591_59135

theorem largest_solution_logarithm_equation (x : ℝ) : 
  (x > 0) → 
  (∀ y, y > 0 → (Real.log 2 / Real.log (2*y) + Real.log 2 / Real.log (4*y^2) = -1) → x ≥ y) →
  (Real.log 2 / Real.log (2*x) + Real.log 2 / Real.log (4*x^2) = -1) →
  1 / x^12 = 4096 := by
sorry

end NUMINAMATH_CALUDE_largest_solution_logarithm_equation_l591_59135


namespace NUMINAMATH_CALUDE_parabola_directrix_l591_59120

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix_equation (y : ℝ) : Prop :=
  y = -1/2

/-- Theorem: The directrix of the given parabola is y = -1/2 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l591_59120


namespace NUMINAMATH_CALUDE_max_value_theorem_l591_59125

theorem max_value_theorem (x y : ℝ) (hx : |x - 1| ≤ 1) (hy : |y - 2| ≤ 1) : 
  |x - 2*y + 1| ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l591_59125


namespace NUMINAMATH_CALUDE_festival_allowance_days_l591_59140

/-- Calculates the maximum number of full days for festival allowance --/
def maxAllowanceDays (staffCount : Nat) (dailyRate : Nat) (totalAmount : Nat) (pettyCashAmount : Nat) : Nat :=
  let totalAvailable := totalAmount + pettyCashAmount
  (totalAvailable - pettyCashAmount) / (staffCount * dailyRate)

theorem festival_allowance_days :
  maxAllowanceDays 20 100 65000 1000 = 32 := by
  sorry

end NUMINAMATH_CALUDE_festival_allowance_days_l591_59140


namespace NUMINAMATH_CALUDE_hockey_league_games_l591_59130

/-- The number of teams in the hockey league -/
def num_teams : ℕ := 15

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- Calculates the total number of games in the season -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_per_pair

/-- Theorem: The total number of games in the season is 1050 -/
theorem hockey_league_games : total_games = 1050 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l591_59130


namespace NUMINAMATH_CALUDE_husband_age_difference_l591_59186

-- Define the initial ages
def hannah_initial_age : ℕ := 6
def july_initial_age : ℕ := hannah_initial_age / 2

-- Define the time passed
def years_passed : ℕ := 20

-- Define July's current age
def july_current_age : ℕ := july_initial_age + years_passed

-- Define July's husband's age
def husband_age : ℕ := 25

-- Theorem to prove
theorem husband_age_difference : husband_age - july_current_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_husband_age_difference_l591_59186


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l591_59132

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 + 6*x + 2
  ∃ x1 x2 : ℝ, x1 = -3 + Real.sqrt 7 ∧ x2 = -3 - Real.sqrt 7 ∧ f x1 = 0 ∧ f x2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l591_59132


namespace NUMINAMATH_CALUDE_rowing_downstream_speed_l591_59155

/-- The speed of a man rowing downstream, given his upstream speed and still water speed -/
theorem rowing_downstream_speed (upstream_speed still_water_speed : ℝ) :
  upstream_speed = 27 →
  still_water_speed = 31 →
  still_water_speed + (still_water_speed - upstream_speed) = 35 := by
sorry

end NUMINAMATH_CALUDE_rowing_downstream_speed_l591_59155


namespace NUMINAMATH_CALUDE_fraction_of_employees_laid_off_l591_59182

/-- Proves that the fraction of employees laid off is 1/3 given the initial conditions -/
theorem fraction_of_employees_laid_off 
  (initial_employees : ℕ) 
  (salary_per_employee : ℕ) 
  (total_paid_after_layoff : ℕ) 
  (h1 : initial_employees = 450)
  (h2 : salary_per_employee = 2000)
  (h3 : total_paid_after_layoff = 600000) :
  (initial_employees * salary_per_employee - total_paid_after_layoff) / (initial_employees * salary_per_employee) = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_of_employees_laid_off_l591_59182


namespace NUMINAMATH_CALUDE_rectangular_box_area_volume_relation_l591_59164

/-- A rectangular box with dimensions x, y, and z -/
structure RectangularBox where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Properties of the rectangular box -/
def RectangularBox.properties (box : RectangularBox) : Prop :=
  let top_area := box.x * box.y
  let side_area := box.y * box.z
  let volume := box.x * box.y * box.z
  (side_area * volume ^ 2 = box.z ^ 3 * volume)

/-- Theorem: The product of the side area and square of volume equals z³V -/
theorem rectangular_box_area_volume_relation (box : RectangularBox) :
  box.properties :=
by
  sorry

#check rectangular_box_area_volume_relation

end NUMINAMATH_CALUDE_rectangular_box_area_volume_relation_l591_59164


namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l591_59174

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 6 + 3 = 6) → initial_people = 9 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l591_59174


namespace NUMINAMATH_CALUDE_four_numbers_with_avg_six_l591_59134

theorem four_numbers_with_avg_six (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 6 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    (w + x + y + z : ℚ) / 4 = 6 →
    max a (max b (max c d)) - min a (min b (min c d)) ≥ max w (max x (max y z)) - min w (min x (min y z)) →
  (((max a (max b (max c d)) + min a (min b (min c d))) - (a + b + c + d)) / 2 : ℚ) = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_with_avg_six_l591_59134


namespace NUMINAMATH_CALUDE_cherie_boxes_count_l591_59175

/-- The number of boxes Koby bought -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of sparklers in each of Cherie's boxes -/
def cherie_sparklers_per_box : ℕ := 8

/-- The number of whistlers in each of Cherie's boxes -/
def cherie_whistlers_per_box : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 33

/-- The number of boxes Cherie bought -/
def cherie_boxes : ℕ := 1

theorem cherie_boxes_count : 
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box) + 
  cherie_boxes * (cherie_sparklers_per_box + cherie_whistlers_per_box) = 
  total_fireworks :=
by sorry

end NUMINAMATH_CALUDE_cherie_boxes_count_l591_59175


namespace NUMINAMATH_CALUDE_principal_is_8000_l591_59198

/-- The principal amount that satisfies the given compound interest conditions -/
def find_principal : ℝ := by sorry

/-- The annual interest rate -/
def interest_rate : ℝ := by sorry

theorem principal_is_8000 :
  find_principal = 8000 ∧
  find_principal * (1 + interest_rate)^2 = 8820 ∧
  find_principal * (1 + interest_rate)^3 = 9261 := by sorry

end NUMINAMATH_CALUDE_principal_is_8000_l591_59198


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l591_59123

/-- Represents a square tile pattern -/
structure TilePattern where
  side : ℕ
  black_tiles : ℕ
  white_tiles : ℕ

/-- Creates an extended pattern by adding a border of black tiles -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 2,
    black_tiles := p.black_tiles + 4 * p.side + 4,
    white_tiles := p.white_tiles }

/-- The theorem to be proved -/
theorem extended_pattern_ratio (p : TilePattern)
  (h1 : p.side * p.side = p.black_tiles + p.white_tiles)
  (h2 : p.black_tiles = 12)
  (h3 : p.white_tiles = 24) :
  let extended := extend_pattern p
  (extended.black_tiles : ℚ) / extended.white_tiles = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l591_59123


namespace NUMINAMATH_CALUDE_work_completion_time_l591_59157

/-- 
Given a piece of work that can be completed by 9 laborers in 15 days, 
this theorem proves that it would take 9 days for 15 laborers to complete the same work.
-/
theorem work_completion_time 
  (total_laborers : ℕ) 
  (available_laborers : ℕ) 
  (actual_days : ℕ) 
  (h1 : total_laborers = 15)
  (h2 : available_laborers = total_laborers - 6)
  (h3 : actual_days = 15)
  : (available_laborers * actual_days) / total_laborers = 9 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l591_59157


namespace NUMINAMATH_CALUDE_function_inequality_existence_l591_59158

theorem function_inequality_existence (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, x + a * f y ≤ y + f (f x)) ↔ (a < 0 ∨ a = 1) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_existence_l591_59158


namespace NUMINAMATH_CALUDE_total_pay_for_two_employees_l591_59146

/-- Proves that the total amount paid to two employees X and Y is 770 units of currency,
    given that X is paid 120% of Y's pay and Y is paid 350 units per week. -/
theorem total_pay_for_two_employees (y_pay : ℝ) (x_pay : ℝ) : 
  y_pay = 350 → x_pay = 1.2 * y_pay → x_pay + y_pay = 770 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_for_two_employees_l591_59146


namespace NUMINAMATH_CALUDE_power_of_two_equality_l591_59112

theorem power_of_two_equality (y : ℤ) : (1 / 8 : ℚ) * (2 ^ 40 : ℚ) = (2 : ℚ) ^ y → y = 37 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l591_59112


namespace NUMINAMATH_CALUDE_sandra_feeding_days_l591_59152

/-- The number of days Sandra can feed the puppies with the given formula -/
def feeding_days (num_puppies : ℕ) (total_portions : ℕ) (feedings_per_day : ℕ) : ℕ :=
  total_portions / (num_puppies * feedings_per_day)

/-- Theorem stating that Sandra can feed the puppies for 5 days with the given formula -/
theorem sandra_feeding_days :
  feeding_days 7 105 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sandra_feeding_days_l591_59152


namespace NUMINAMATH_CALUDE_points_eight_units_from_negative_three_l591_59187

def distance (x y : ℝ) : ℝ := |x - y|

theorem points_eight_units_from_negative_three :
  ∀ x : ℝ, distance x (-3) = 8 ↔ x = -11 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_points_eight_units_from_negative_three_l591_59187


namespace NUMINAMATH_CALUDE_G_equals_3F_l591_59172

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((3 * x + x^3) / (1 + 3 * x^2))

theorem G_equals_3F (x : ℝ) : G x = 3 * F x :=
  sorry

end NUMINAMATH_CALUDE_G_equals_3F_l591_59172


namespace NUMINAMATH_CALUDE_janets_sandcastle_height_l591_59159

/-- Given the heights of two sandcastles, proves that the taller one is the sum of the shorter one's height and the difference between them. -/
theorem janets_sandcastle_height 
  (sisters_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : sisters_height = 2.3333333333333335)
  (h2 : height_difference = 1.3333333333333333) : 
  sisters_height + height_difference = 3.6666666666666665 := by
  sorry

#check janets_sandcastle_height

end NUMINAMATH_CALUDE_janets_sandcastle_height_l591_59159


namespace NUMINAMATH_CALUDE_rectangle_equation_l591_59173

/-- Given a rectangle with area 864 square steps and perimeter 120 steps,
    prove that the equation relating its length x to its area is x(60 - x) = 864 -/
theorem rectangle_equation (x : ℝ) 
  (area : ℝ) (perimeter : ℝ)
  (h_area : area = 864)
  (h_perimeter : perimeter = 120)
  (h_x : x > 0 ∧ x < 60) :
  x * (60 - x) = 864 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_equation_l591_59173


namespace NUMINAMATH_CALUDE_f_2015_5_l591_59124

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_2015_5 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 4)
  (h_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)) :
  f 2015.5 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_2015_5_l591_59124


namespace NUMINAMATH_CALUDE_product_remainder_by_ten_l591_59161

theorem product_remainder_by_ten : (1734 * 5389 * 80607) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_ten_l591_59161


namespace NUMINAMATH_CALUDE_second_digit_of_three_digit_number_l591_59141

/-- Given a three-digit number xyz, if 100x + 10y + z - (x + y + z) = 261, then y = 7 -/
theorem second_digit_of_three_digit_number (x y z : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ z ≥ 0 ∧ z ≤ 9 →
  100 * x + 10 * y + z - (x + y + z) = 261 →
  y = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_digit_of_three_digit_number_l591_59141


namespace NUMINAMATH_CALUDE_top_books_sold_l591_59113

/-- The number of "TOP" books sold last week -/
def top_books : ℕ := 13

/-- The price of a "TOP" book in dollars -/
def top_price : ℕ := 8

/-- The price of an "ABC" book in dollars -/
def abc_price : ℕ := 23

/-- The number of "ABC" books sold last week -/
def abc_books : ℕ := 4

/-- The difference in earnings between "TOP" and "ABC" books in dollars -/
def earnings_difference : ℕ := 12

theorem top_books_sold : 
  top_books * top_price - abc_books * abc_price = earnings_difference := by
  sorry

end NUMINAMATH_CALUDE_top_books_sold_l591_59113


namespace NUMINAMATH_CALUDE_total_seashells_is_142_l591_59142

/-- The number of seashells Joan found initially -/
def joans_initial_seashells : ℕ := 79

/-- The number of seashells Mike gave to Joan -/
def mikes_seashells : ℕ := 63

/-- The total number of seashells Joan has -/
def total_seashells : ℕ := joans_initial_seashells + mikes_seashells

/-- Theorem stating that the total number of seashells Joan has is 142 -/
theorem total_seashells_is_142 : total_seashells = 142 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_is_142_l591_59142


namespace NUMINAMATH_CALUDE_adam_shopping_cost_l591_59153

/-- The total cost of Adam's shopping given the number of sandwiches, cost per sandwich, and cost of water. -/
def total_cost (num_sandwiches : ℕ) (sandwich_price : ℕ) (water_price : ℕ) : ℕ :=
  num_sandwiches * sandwich_price + water_price

/-- Theorem stating that Adam's total shopping cost is $11. -/
theorem adam_shopping_cost :
  total_cost 3 3 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_adam_shopping_cost_l591_59153


namespace NUMINAMATH_CALUDE_min_black_edges_is_four_l591_59105

/-- Represents the coloring of a cube's edges -/
structure CubeColoring where
  edges : Fin 12 → Bool  -- True represents black, False represents red

/-- Checks if a face has an even number of black edges -/
def has_even_black_edges (c : CubeColoring) (face : Fin 6) : Bool :=
  sorry

/-- Checks if all faces have an even number of black edges -/
def all_faces_even_black (c : CubeColoring) : Prop :=
  ∀ face : Fin 6, has_even_black_edges c face

/-- Counts the number of black edges in a coloring -/
def count_black_edges (c : CubeColoring) : Nat :=
  sorry

/-- The main theorem: The minimum number of black edges required is 4 -/
theorem min_black_edges_is_four :
  (∃ c : CubeColoring, all_faces_even_black c ∧ count_black_edges c = 4) ∧
  (∀ c : CubeColoring, all_faces_even_black c → count_black_edges c ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_min_black_edges_is_four_l591_59105


namespace NUMINAMATH_CALUDE_fish_length_difference_l591_59107

theorem fish_length_difference : 
  let first_fish_length : Real := 0.3
  let second_fish_length : Real := 0.2
  first_fish_length - second_fish_length = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_fish_length_difference_l591_59107


namespace NUMINAMATH_CALUDE_sqrt_equation_unique_solution_l591_59117

theorem sqrt_equation_unique_solution :
  ∃! x : ℝ, Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 6 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_unique_solution_l591_59117


namespace NUMINAMATH_CALUDE_prob_same_type_three_pairs_l591_59151

/-- Represents a collection of paired items -/
structure PairedCollection :=
  (num_pairs : ℕ)
  (items_per_pair : ℕ)
  (total_items : ℕ)
  (h_total : total_items = num_pairs * items_per_pair)

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the probability of selecting two items of the same type -/
def prob_same_type (collection : PairedCollection) : ℚ :=
  (collection.num_pairs : ℚ) / (choose collection.total_items 2)

/-- The main theorem to be proved -/
theorem prob_same_type_three_pairs :
  let shoe_collection : PairedCollection :=
    { num_pairs := 3
    , items_per_pair := 2
    , total_items := 6
    , h_total := rfl }
  prob_same_type shoe_collection = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_type_three_pairs_l591_59151


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l591_59126

/-- The minimum distance between a point on the circle x² + y² = 4 
    and the line √3y + x + 4√3 = 0 is 2√3 - 2 -/
theorem min_distance_circle_line : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let line := {p : ℝ × ℝ | Real.sqrt 3 * p.2 + p.1 + 4 * Real.sqrt 3 = 0}
  ∃ (d : ℝ), d = 2 * Real.sqrt 3 - 2 ∧ 
    (∀ p ∈ circle, ∀ q ∈ line, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ p ∈ circle, ∃ q ∈ line, d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_circle_line_l591_59126


namespace NUMINAMATH_CALUDE_smallest_with_eight_divisors_l591_59179

/-- A function that returns the number of distinct positive divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- Proposition: 24 is the smallest positive integer with exactly eight distinct positive divisors -/
theorem smallest_with_eight_divisors :
  (∀ m : ℕ, m > 0 → m < 24 → numDivisors m ≠ 8) ∧ numDivisors 24 = 8 := by sorry

end NUMINAMATH_CALUDE_smallest_with_eight_divisors_l591_59179


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l591_59111

theorem binomial_expansion_properties :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  (∀ x : ℝ, (2 * x - Real.sqrt 3) ^ 10 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + 
                                         a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9 + a₁₀ * x^10) →
  (a₀ = 243 ∧ 
   (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀) * 
   (a₀ - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ - a₇ + a₈ - a₉ + a₁₀) = 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l591_59111


namespace NUMINAMATH_CALUDE_x_value_proof_l591_59148

theorem x_value_proof (x y : ℝ) 
  (eq1 : x^2 - 4*x + y = 0) 
  (eq2 : y = 4) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l591_59148


namespace NUMINAMATH_CALUDE_multiples_4_or_9_less_than_201_l591_59104

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (max divisor : ℕ) : ℕ :=
  (max - 1) / divisor

def count_either_not_both (max a b : ℕ) : ℕ :=
  count_multiples max a + count_multiples max b - 2 * count_multiples max (lcm a b)

theorem multiples_4_or_9_less_than_201 :
  count_either_not_both 201 4 9 = 62 := by sorry

end NUMINAMATH_CALUDE_multiples_4_or_9_less_than_201_l591_59104


namespace NUMINAMATH_CALUDE_thousand_pow_ten_zeros_l591_59189

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- 1000 is equal to 10^3 -/
axiom thousand_eq_ten_cubed : (1000 : ℕ) = 10^3

/-- The number of trailing zeros in 1000^10 is 30 -/
theorem thousand_pow_ten_zeros : trailingZeros (1000^10) = 30 := by sorry

end NUMINAMATH_CALUDE_thousand_pow_ten_zeros_l591_59189


namespace NUMINAMATH_CALUDE_pictures_per_album_l591_59184

theorem pictures_per_album 
  (phone_pics : ℕ) 
  (camera_pics : ℕ) 
  (num_albums : ℕ) 
  (h1 : phone_pics = 35) 
  (h2 : camera_pics = 5) 
  (h3 : num_albums = 5) : 
  (phone_pics + camera_pics) / num_albums = 8 := by
  sorry

end NUMINAMATH_CALUDE_pictures_per_album_l591_59184


namespace NUMINAMATH_CALUDE_geometric_series_sum_l591_59162

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 6) :
  ∑' n, 2*a / (a + b)^n = 12/7 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l591_59162


namespace NUMINAMATH_CALUDE_z_value_l591_59178

theorem z_value (x : ℝ) (z : ℝ) (h1 : 3 * x = 0.75 * z) (h2 : x = 20) : z = 80 := by
  sorry

end NUMINAMATH_CALUDE_z_value_l591_59178


namespace NUMINAMATH_CALUDE_weight_replacement_l591_59102

theorem weight_replacement (initial_total : ℝ) (replaced_weight : ℝ) :
  (∃ (average_increase : ℝ),
    average_increase = 1.5 ∧
    4 * (initial_total / 4 + average_increase) = initial_total - replaced_weight + 71) →
  replaced_weight = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l591_59102


namespace NUMINAMATH_CALUDE_greatest_c_for_no_real_solutions_l591_59171

theorem greatest_c_for_no_real_solutions : 
  (∃ c : ℤ, c = (Nat.floor (Real.sqrt 116)) ∧ 
   ∀ x : ℝ, x^2 + c*x + 29 ≠ 0 ∧
   ∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 29 = 0) ∧
  (Nat.floor (Real.sqrt 116) = 10) :=
by sorry

end NUMINAMATH_CALUDE_greatest_c_for_no_real_solutions_l591_59171


namespace NUMINAMATH_CALUDE_modulus_of_one_over_one_minus_i_l591_59114

theorem modulus_of_one_over_one_minus_i :
  let z : ℂ := 1 / (1 - I)
  ‖z‖ = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_one_over_one_minus_i_l591_59114


namespace NUMINAMATH_CALUDE_predicted_value_theorem_l591_59196

/-- A linear regression model with given slope and sample centroid -/
structure LinearRegressionModel where
  slope : ℝ
  centroid_x : ℝ
  centroid_y : ℝ

/-- Calculate the predicted value of the dependent variable -/
def predict (model : LinearRegressionModel) (x : ℝ) : ℝ :=
  let intercept := model.centroid_y - model.slope * model.centroid_x
  model.slope * x + intercept

theorem predicted_value_theorem (model : LinearRegressionModel) 
  (h1 : model.slope = 1.23)
  (h2 : model.centroid_x = 4)
  (h3 : model.centroid_y = 5)
  (x : ℝ)
  (h4 : x = 10) :
  predict model x = 12.38 := by
  sorry

end NUMINAMATH_CALUDE_predicted_value_theorem_l591_59196


namespace NUMINAMATH_CALUDE_john_reads_bible_in_four_weeks_l591_59131

/-- Represents John's Bible reading scenario -/
structure BibleReading where
  daily_hours : ℕ               -- Hours John reads per day
  pages_per_hour : ℕ            -- Pages John reads per hour
  bible_pages : ℕ               -- Total pages in the Bible
  days_per_week : ℕ := 7        -- Number of days in a week

/-- Calculates the number of weeks needed to read the entire Bible -/
def weeks_to_read (br : BibleReading) : ℚ :=
  br.bible_pages / (br.daily_hours * br.pages_per_hour * br.days_per_week)

/-- Theorem stating that it takes John 4 weeks to read the entire Bible -/
theorem john_reads_bible_in_four_weeks :
  let br : BibleReading := {
    daily_hours := 2,
    pages_per_hour := 50,
    bible_pages := 2800
  }
  weeks_to_read br = 4 := by sorry

end NUMINAMATH_CALUDE_john_reads_bible_in_four_weeks_l591_59131


namespace NUMINAMATH_CALUDE_inner_quadrilateral_area_ratio_l591_59108

-- Define the quadrilateral type
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the area function for quadrilaterals
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Define the function to get points on the sides of a quadrilateral
def getInnerPoints (q : Quadrilateral) (p : ℝ) : Quadrilateral := sorry

-- Main theorem
theorem inner_quadrilateral_area_ratio 
  (ABCD : Quadrilateral) (p : ℝ) (h : p < 0.5) :
  let A₁B₁C₁D₁ := getInnerPoints ABCD p
  area A₁B₁C₁D₁ / area ABCD = 1 - 2 * p := by sorry

end NUMINAMATH_CALUDE_inner_quadrilateral_area_ratio_l591_59108


namespace NUMINAMATH_CALUDE_no_natural_pairs_satisfying_divisibility_l591_59109

theorem no_natural_pairs_satisfying_divisibility : 
  ¬∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (b^a ∣ a^b - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_pairs_satisfying_divisibility_l591_59109


namespace NUMINAMATH_CALUDE_circle_center_l591_59145

/-- A circle passing through (0,0) and tangent to y = x^2 at (1,1) has center (-1, 2) -/
theorem circle_center (c : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x - c.1)^2 + (y - c.2)^2 = (c.1)^2 + (c.2)^2 → (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
  (∃ (r : ℝ), ∀ (x y : ℝ), y = x^2 → ((x - 1)^2 + (y - 1)^2 = r^2 → x = 1 ∧ y = 1)) →
  c = (-1, 2) := by
sorry


end NUMINAMATH_CALUDE_circle_center_l591_59145


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l591_59154

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l591_59154


namespace NUMINAMATH_CALUDE_solution_set_for_half_range_of_m_l591_59192

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| - |2*x - 2*m|

-- Part 1
theorem solution_set_for_half (x : ℝ) :
  (f x (1/2) ≥ 1/2) ↔ (1/3 ≤ x ∧ x < 1) :=
sorry

-- Part 2
theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧ m < 7/2) ↔
    (∀ x : ℝ, ∃ t : ℝ, f x m + |t - 3| < |t + 4|) :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_half_range_of_m_l591_59192


namespace NUMINAMATH_CALUDE_cookie_jar_solution_l591_59138

def cookie_jar_problem (initial_amount doris_spent martha_spent remaining : ℝ) : Prop :=
  doris_spent = 6 ∧
  martha_spent = doris_spent / 2 ∧
  remaining = 15 ∧
  initial_amount = doris_spent + martha_spent + remaining

theorem cookie_jar_solution :
  ∃ initial_amount doris_spent martha_spent remaining : ℝ,
    cookie_jar_problem initial_amount doris_spent martha_spent remaining ∧
    initial_amount = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_solution_l591_59138
