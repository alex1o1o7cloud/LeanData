import Mathlib

namespace NUMINAMATH_CALUDE_hundreds_digit_of_13_pow_2023_l2183_218349

theorem hundreds_digit_of_13_pow_2023 : 13^2023 % 1000 = 99 := by sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_13_pow_2023_l2183_218349


namespace NUMINAMATH_CALUDE_age_problem_l2183_218399

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →  -- A is two years older than B
  b = 2 * c →  -- B is twice as old as C
  a + b + c = 37 →  -- Total age is 37
  b = 14 :=  -- B's age is 14
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l2183_218399


namespace NUMINAMATH_CALUDE_accessory_production_equation_l2183_218390

theorem accessory_production_equation 
  (initial_production : ℕ) 
  (total_production : ℕ) 
  (x : ℝ) 
  (h1 : initial_production = 600000) 
  (h2 : total_production = 2180000) :
  (600 : ℝ) + 600 * (1 + x) + 600 * (1 + x)^2 = 2180 :=
by sorry

end NUMINAMATH_CALUDE_accessory_production_equation_l2183_218390


namespace NUMINAMATH_CALUDE_square_to_octagon_triangle_to_icosagon_l2183_218389

-- Define a square
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define a triangle
structure Triangle :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define an octagon
structure Octagon :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define a 20-sided polygon (icosagon)
structure Icosagon :=
  (side : ℝ)
  (side_positive : side > 0)

-- Function to cut a square into two parts
def cut_square (s : Square) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Function to form an octagon from two parts
def form_octagon (parts : (ℝ × ℝ) × (ℝ × ℝ)) : Octagon := sorry

-- Function to cut a triangle into two parts
def cut_triangle (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Function to form an icosagon from two parts
def form_icosagon (parts : (ℝ × ℝ) × (ℝ × ℝ)) : Icosagon := sorry

-- Theorem stating that a square can be cut into two parts to form an octagon
theorem square_to_octagon (s : Square) :
  ∃ (o : Octagon), form_octagon (cut_square s) = o := sorry

-- Theorem stating that a triangle can be cut into two parts to form an icosagon
theorem triangle_to_icosagon (t : Triangle) :
  ∃ (i : Icosagon), form_icosagon (cut_triangle t) = i := sorry

end NUMINAMATH_CALUDE_square_to_octagon_triangle_to_icosagon_l2183_218389


namespace NUMINAMATH_CALUDE_special_set_property_l2183_218363

/-- A set of points in ℝ³ that intersects every plane but has finite intersection with each plane -/
def SpecialSet : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | ∃ t : ℝ, x = t^5 ∧ y = t^3 ∧ z = t}

/-- Definition of a plane in ℝ³ -/
def Plane (a b c d : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | a * x + b * y + c * z + d = 0}

theorem special_set_property :
  ∃ S : Set (ℝ × ℝ × ℝ),
    (∀ a b c d : ℝ, (Plane a b c d ∩ S).Nonempty) ∧
    (∀ a b c d : ℝ, (Plane a b c d ∩ S).Finite) :=
by
  use SpecialSet
  sorry

end NUMINAMATH_CALUDE_special_set_property_l2183_218363


namespace NUMINAMATH_CALUDE_fraction_meaningful_l2183_218323

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1) / (x - 2)) ↔ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l2183_218323


namespace NUMINAMATH_CALUDE_problem_solution_l2183_218362

/-- S(n, k) denotes the number of coefficients in the expansion of (x+1)^n that are not divisible by k -/
def S (n k : ℕ) : ℕ := sorry

theorem problem_solution :
  (S 2012 3 = 324) ∧ (2012 ∣ S (2012^2011) 2011) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2183_218362


namespace NUMINAMATH_CALUDE_min_tablets_to_extract_l2183_218328

/-- Represents the number of tablets for each medicine type in the box -/
structure TabletCount where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the minimum number of tablets to extract to guarantee at least two of each type -/
def minTablets (count : TabletCount) : Nat :=
  (count.a - 1) + (count.b - 1) + 2

/-- Theorem stating the minimum number of tablets to extract for the given problem -/
theorem min_tablets_to_extract (box : TabletCount) 
  (ha : box.a = 25) (hb : box.b = 30) (hc : box.c = 20) : 
  minTablets box = 55 := by
  sorry

#eval minTablets { a := 25, b := 30, c := 20 }

end NUMINAMATH_CALUDE_min_tablets_to_extract_l2183_218328


namespace NUMINAMATH_CALUDE_andre_total_cost_l2183_218331

/-- Calculates the total cost of Andre's purchases including sales tax -/
def total_cost (treadmill_price : ℝ) (treadmill_discount : ℝ) 
                (plate_price : ℝ) (plate_discount : ℝ) (num_plates : ℕ)
                (sales_tax : ℝ) : ℝ :=
  let discounted_treadmill := treadmill_price * (1 - treadmill_discount)
  let discounted_plates := plate_price * num_plates * (1 - plate_discount)
  let subtotal := discounted_treadmill + discounted_plates
  subtotal * (1 + sales_tax)

/-- Theorem stating that Andre's total cost is $1120.29 -/
theorem andre_total_cost :
  total_cost 1350 0.30 60 0.15 2 0.07 = 1120.29 := by
  sorry

end NUMINAMATH_CALUDE_andre_total_cost_l2183_218331


namespace NUMINAMATH_CALUDE_new_average_weight_l2183_218383

/-- Given 29 students with an average weight of 28 kg, after admitting a new student weighing 1 kg,
    the new average weight of all 30 students is 27.1 kg. -/
theorem new_average_weight (initial_count : ℕ) (initial_avg : ℝ) (new_student_weight : ℝ) :
  initial_count = 29 →
  initial_avg = 28 →
  new_student_weight = 1 →
  let total_weight := initial_count * initial_avg + new_student_weight
  let new_count := initial_count + 1
  (total_weight / new_count : ℝ) = 27.1 :=
by sorry

end NUMINAMATH_CALUDE_new_average_weight_l2183_218383


namespace NUMINAMATH_CALUDE_no_primes_satisfying_equation_l2183_218379

theorem no_primes_satisfying_equation : 
  ¬ ∃ (a b c d : ℕ), 
    Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
    a < b ∧ b < c ∧ c < d ∧
    (1 : ℚ) / a + (1 : ℚ) / d = (1 : ℚ) / b + (1 : ℚ) / c :=
by sorry

end NUMINAMATH_CALUDE_no_primes_satisfying_equation_l2183_218379


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2183_218376

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (4 * a^3 + 502 * a + 1004 = 0) →
  (4 * b^3 + 502 * b + 1004 = 0) →
  (4 * c^3 + 502 * c + 1004 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 753 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2183_218376


namespace NUMINAMATH_CALUDE_power_of_fraction_to_decimal_l2183_218347

theorem power_of_fraction_to_decimal :
  (4 / 5 : ℚ) ^ 3 = 512 / 1000 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_to_decimal_l2183_218347


namespace NUMINAMATH_CALUDE_sequence_increasing_l2183_218300

/-- Given positive real numbers a, b, c, and a natural number n,
    prove that a_n < a_{n+1} where a_n = (a*n)/(b*n + c) -/
theorem sequence_increasing (a b c : ℝ) (n : ℕ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    let a_n := (a * n) / (b * n + c)
    let a_n_plus_1 := (a * (n + 1)) / (b * (n + 1) + c)
    a_n < a_n_plus_1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_l2183_218300


namespace NUMINAMATH_CALUDE_refrigerator_transport_cost_l2183_218368

/-- Calculate the transport cost for a refrigerator purchase --/
theorem refrigerator_transport_cost 
  (purchase_price : ℕ) 
  (discount_rate : ℚ) 
  (installation_cost : ℕ) 
  (profit_rate : ℚ) 
  (selling_price : ℕ) 
  (h1 : purchase_price = 14500)
  (h2 : discount_rate = 1/5)
  (h3 : installation_cost = 250)
  (h4 : profit_rate = 1/10)
  (h5 : selling_price = 20350) : 
  ∃ (transport_cost : ℕ), transport_cost = 3375 :=
by
  sorry

#check refrigerator_transport_cost

end NUMINAMATH_CALUDE_refrigerator_transport_cost_l2183_218368


namespace NUMINAMATH_CALUDE_sum_equality_l2183_218313

theorem sum_equality : 9548 + 7314 = 3362 + 13500 := by
  sorry

end NUMINAMATH_CALUDE_sum_equality_l2183_218313


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l2183_218367

-- Define the geometric mean
def geometric_mean (a c : ℝ) : Set ℝ :=
  {b : ℝ | a * c = b^2}

-- Theorem statement
theorem geometric_mean_of_4_and_9 :
  geometric_mean 4 9 = {6, -6} := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l2183_218367


namespace NUMINAMATH_CALUDE_min_distance_ant_spider_l2183_218318

/-- The minimum distance between a point on the unit circle and a corresponding point on the x-axis -/
theorem min_distance_ant_spider :
  let f : ℝ → ℝ := λ a => Real.sqrt ((a - (1 - 2*a))^2 + (Real.sqrt (1 - a^2))^2)
  ∃ a : ℝ, ∀ x : ℝ, f x ≥ f a ∧ f a = Real.sqrt 14 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_ant_spider_l2183_218318


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l2183_218397

theorem max_value_cos_sin (x : ℝ) : 
  let f := fun (x : ℝ) => 2 * Real.cos x + Real.sin x
  f x ≤ Real.sqrt 5 ∧ ∃ y, f y = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l2183_218397


namespace NUMINAMATH_CALUDE_unique_hour_conversion_l2183_218325

theorem unique_hour_conversion : 
  ∃! n : ℕ, 
    (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n = 234000 + x * 1000 + y * 100) ∧ 
    (n % 3600 = 0) ∧
    (∃ h : ℕ, n = h * 3600) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_hour_conversion_l2183_218325


namespace NUMINAMATH_CALUDE_min_stamps_for_37_cents_l2183_218311

/-- Represents the number of ways to make a certain amount with given coin denominations -/
def numWays (amount : ℕ) (coins : List ℕ) : ℕ := sorry

/-- Finds the minimum number of coins needed to make the amount -/
def minCoins (amount : ℕ) (coins : List ℕ) : ℕ := sorry

theorem min_stamps_for_37_cents :
  minCoins 37 [5, 7] = 7 := by sorry

end NUMINAMATH_CALUDE_min_stamps_for_37_cents_l2183_218311


namespace NUMINAMATH_CALUDE_division_of_decimals_l2183_218338

theorem division_of_decimals : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l2183_218338


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2183_218351

/-- The polynomial function f(x) = x^10 + 5x^9 + 28x^8 + 145x^7 - 1897x^6 -/
def f (x : ℝ) : ℝ := x^10 + 5*x^9 + 28*x^8 + 145*x^7 - 1897*x^6

/-- Theorem: The equation f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2183_218351


namespace NUMINAMATH_CALUDE_philip_farm_animals_l2183_218364

/-- The number of animals on Philip's farm -/
def total_animals (cows ducks pigs : ℕ) : ℕ := cows + ducks + pigs

/-- The number of cows on Philip's farm -/
def number_of_cows : ℕ := 20

/-- The number of ducks on Philip's farm -/
def number_of_ducks : ℕ := number_of_cows + (number_of_cows / 2)

/-- The number of pigs on Philip's farm -/
def number_of_pigs : ℕ := (number_of_cows + number_of_ducks) / 5

theorem philip_farm_animals :
  total_animals number_of_cows number_of_ducks number_of_pigs = 60 := by
  sorry

end NUMINAMATH_CALUDE_philip_farm_animals_l2183_218364


namespace NUMINAMATH_CALUDE_equality_of_areas_l2183_218384

theorem equality_of_areas (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  (∃ r : Real, r > 0 ∧ 
    (r^2 * θ / 2 = r^2 * Real.tan θ / 2 - r^2 * θ / 2)) ↔ 
  Real.tan θ = 2 * θ := by
sorry

end NUMINAMATH_CALUDE_equality_of_areas_l2183_218384


namespace NUMINAMATH_CALUDE_factorization_equality_l2183_218357

theorem factorization_equality (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^3 + b^3 + c^3 - 3*a*b*c) :=
by sorry

end NUMINAMATH_CALUDE_factorization_equality_l2183_218357


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_semi_axes_product_l2183_218330

/-- Given an ellipse and a hyperbola with specific foci, prove the product of their semi-axes -/
theorem ellipse_hyperbola_semi_axes_product (c d : ℝ) : 
  (∀ (x y : ℝ), x^2/c^2 + y^2/d^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2/c^2 - y^2/d^2 = 1 → (x = 8 ∧ y = 0) ∨ (x = -8 ∧ y = 0)) →
  |c * d| = Real.sqrt 868.5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_semi_axes_product_l2183_218330


namespace NUMINAMATH_CALUDE_four_is_square_root_of_sixteen_l2183_218316

-- Definition of square root
def is_square_root (x y : ℝ) : Prop := y * y = x

-- Theorem to prove
theorem four_is_square_root_of_sixteen : is_square_root 16 4 := by
  sorry

end NUMINAMATH_CALUDE_four_is_square_root_of_sixteen_l2183_218316


namespace NUMINAMATH_CALUDE_tournament_games_theorem_l2183_218317

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (num_teams : ℕ)
  (num_players_per_team : ℕ)

/-- Calculates the number of games needed to determine the champion -/
def games_to_champion (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- The theorem stating that a tournament with 128 teams requires 127 games to determine the champion -/
theorem tournament_games_theorem :
  ∀ (t : Tournament), t.num_teams = 128 → t.num_players_per_team = 4 → games_to_champion t = 127 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_theorem_l2183_218317


namespace NUMINAMATH_CALUDE_car_speed_theorem_l2183_218386

def car_speed_problem (first_hour_speed average_speed : ℝ) : Prop :=
  let total_time : ℝ := 2
  let second_hour_speed : ℝ := 2 * average_speed - first_hour_speed
  second_hour_speed = 50

theorem car_speed_theorem :
  car_speed_problem 90 70 := by sorry

end NUMINAMATH_CALUDE_car_speed_theorem_l2183_218386


namespace NUMINAMATH_CALUDE_power_sum_difference_l2183_218365

theorem power_sum_difference : 3^(1+2+3+4) - (3^1 + 3^2 + 3^3 + 3^4) - 3^5 = 58686 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l2183_218365


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2183_218340

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (L m : Line) (α : Plane) 
  (h1 : parallel m L) 
  (h2 : perpendicular m α) : 
  perpendicular L α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2183_218340


namespace NUMINAMATH_CALUDE_tire_repair_cost_l2183_218345

/-- Calculates the final cost of tire repairs -/
def final_cost (repair_cost : ℚ) (sales_tax : ℚ) (num_tires : ℕ) : ℚ :=
  (repair_cost + sales_tax) * num_tires

/-- Theorem: The final cost for repairing 4 tires is $30 -/
theorem tire_repair_cost : final_cost 7 0.5 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_tire_repair_cost_l2183_218345


namespace NUMINAMATH_CALUDE_simplify_expression_l2183_218385

theorem simplify_expression (x y : ℝ) (hx : x = 3) (hy : y = 4) : 
  (12 * x^2 * y^3) / (8 * x * y^2) = 18 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2183_218385


namespace NUMINAMATH_CALUDE_crow_count_proof_l2183_218398

/-- The number of crows in the first group -/
def first_group_count : ℕ := 3

/-- The number of worms eaten by the first group in one hour -/
def first_group_worms : ℕ := 30

/-- The number of crows in the second group -/
def second_group_count : ℕ := 5

/-- The number of worms eaten by the second group in two hours -/
def second_group_worms : ℕ := 100

/-- The number of hours the second group took to eat their worms -/
def second_group_hours : ℕ := 2

theorem crow_count_proof : first_group_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_crow_count_proof_l2183_218398


namespace NUMINAMATH_CALUDE_muffin_division_l2183_218344

theorem muffin_division (total_muffins : ℕ) (total_people : ℕ) (muffins_per_person : ℕ) : 
  total_muffins = 20 →
  total_people = 5 →
  total_muffins = total_people * muffins_per_person →
  muffins_per_person = 4 :=
by sorry

end NUMINAMATH_CALUDE_muffin_division_l2183_218344


namespace NUMINAMATH_CALUDE_road_trip_cost_l2183_218322

/-- Represents a city with its distance from the starting point and gas price -/
structure City where
  distance : ℝ
  gasPrice : ℝ

/-- Calculates the total cost of a road trip given the car's specifications and cities visited -/
def totalTripCost (fuelEfficiency : ℝ) (tankCapacity : ℝ) (cities : List City) : ℝ :=
  cities.foldl (fun acc city => acc + tankCapacity * city.gasPrice) 0

/-- Theorem: The total cost of the road trip is $192.00 -/
theorem road_trip_cost :
  let fuelEfficiency : ℝ := 30
  let tankCapacity : ℝ := 20
  let cities : List City := [
    { distance := 290, gasPrice := 3.10 },
    { distance := 450, gasPrice := 3.30 },
    { distance := 620, gasPrice := 3.20 }
  ]
  totalTripCost fuelEfficiency tankCapacity cities = 192 :=
by
  sorry

#eval totalTripCost 30 20 [
  { distance := 290, gasPrice := 3.10 },
  { distance := 450, gasPrice := 3.30 },
  { distance := 620, gasPrice := 3.20 }
]

end NUMINAMATH_CALUDE_road_trip_cost_l2183_218322


namespace NUMINAMATH_CALUDE_trip_distance_l2183_218392

/-- Proves that the total distance of a trip is 350 km given specific conditions -/
theorem trip_distance (first_distance : ℝ) (first_speed : ℝ) (second_speed : ℝ) (avg_speed : ℝ) :
  first_distance = 200 →
  first_speed = 20 →
  second_speed = 15 →
  avg_speed = 17.5 →
  ∃ (total_distance : ℝ),
    total_distance = first_distance + (avg_speed * (first_distance / first_speed + (total_distance - first_distance) / second_speed) - first_distance) ∧
    total_distance = 350 :=
by sorry

end NUMINAMATH_CALUDE_trip_distance_l2183_218392


namespace NUMINAMATH_CALUDE_total_lives_after_third_level_l2183_218324

/-- Game rules for calculating lives --/
def game_lives : ℕ → ℕ :=
  let initial_lives := 2
  let first_level_gain := 6 / 2
  let second_level_gain := 11 - 3
  let third_level_multiplier := 2
  fun level =>
    if level = 0 then
      initial_lives
    else if level = 1 then
      initial_lives + first_level_gain
    else if level = 2 then
      initial_lives + first_level_gain + second_level_gain
    else
      initial_lives + first_level_gain + second_level_gain +
      (first_level_gain + second_level_gain) * third_level_multiplier

/-- Theorem stating the total number of lives after completing the third level --/
theorem total_lives_after_third_level :
  game_lives 3 = 35 := by sorry

end NUMINAMATH_CALUDE_total_lives_after_third_level_l2183_218324


namespace NUMINAMATH_CALUDE_complex_equation_proof_l2183_218381

theorem complex_equation_proof (a : ℝ) : 
  ((2 * a) / (1 + Complex.I) + 1 + Complex.I).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l2183_218381


namespace NUMINAMATH_CALUDE_clothing_store_problem_l2183_218355

/-- The clothing store problem -/
theorem clothing_store_problem 
  (cost : ℝ) 
  (initial_price : ℝ) 
  (initial_volume : ℝ) 
  (price_increase : ℝ) 
  (volume_decrease : ℝ)
  (h1 : cost = 50)
  (h2 : initial_price = 60)
  (h3 : initial_volume = 800)
  (h4 : price_increase = 5)
  (h5 : volume_decrease = 100) :
  let sales_volume (x : ℝ) := initial_volume - (volume_decrease / price_increase) * (x - initial_price)
  let profit (x : ℝ) := (x - cost) * sales_volume x
  ∃ (max_price : ℝ) (max_profit : ℝ),
    -- 1. Sales volume at 70 yuan
    sales_volume 70 = 600 ∧
    -- 2. Profit at 70 yuan
    profit 70 = 12000 ∧
    -- 3. Profit function
    (∀ x, profit x = -20 * x^2 + 3000 * x - 100000) ∧
    -- 4. Maximum profit
    (∀ x, profit x ≤ max_profit) ∧ max_price = 75 ∧ max_profit = 12500 ∧
    -- 5. Selling prices for 12000 yuan profit
    profit 70 = 12000 ∧ profit 80 = 12000 ∧
    (∀ x, profit x = 12000 → (x = 70 ∨ x = 80)) := by
  sorry

end NUMINAMATH_CALUDE_clothing_store_problem_l2183_218355


namespace NUMINAMATH_CALUDE_new_arithmetic_mean_l2183_218360

/-- Given a set of 60 numbers with arithmetic mean 42, prove that removing 50 and 60
    and increasing each remaining number by 2 results in a new arithmetic mean of 43.55 -/
theorem new_arithmetic_mean (S : Finset ℝ) (sum_S : ℝ) : 
  S.card = 60 →
  sum_S = S.sum id →
  sum_S / 60 = 42 →
  50 ∈ S →
  60 ∈ S →
  let S' := S.erase 50 ⊔ S.erase 60
  let sum_S' := S'.sum (fun x => x + 2)
  sum_S' / 58 = 43.55 := by
sorry

end NUMINAMATH_CALUDE_new_arithmetic_mean_l2183_218360


namespace NUMINAMATH_CALUDE_overall_average_percentage_l2183_218320

theorem overall_average_percentage (students_A students_B students_C students_D : ℕ)
  (average_A average_B average_C average_D : ℚ) :
  students_A = 15 →
  students_B = 10 →
  students_C = 20 →
  students_D = 5 →
  average_A = 75 / 100 →
  average_B = 90 / 100 →
  average_C = 80 / 100 →
  average_D = 65 / 100 →
  let total_students := students_A + students_B + students_C + students_D
  let total_percentage := students_A * average_A + students_B * average_B +
                          students_C * average_C + students_D * average_D
  total_percentage / total_students = 79 / 100 := by
  sorry

end NUMINAMATH_CALUDE_overall_average_percentage_l2183_218320


namespace NUMINAMATH_CALUDE_alison_large_tubs_l2183_218387

/-- The number of large tubs Alison bought -/
def num_large_tubs : ℕ := 3

/-- The number of small tubs Alison bought -/
def num_small_tubs : ℕ := 6

/-- The cost of each large tub in dollars -/
def cost_large_tub : ℕ := 6

/-- The cost of each small tub in dollars -/
def cost_small_tub : ℕ := 5

/-- The total cost of all tubs in dollars -/
def total_cost : ℕ := 48

theorem alison_large_tubs : 
  num_large_tubs * cost_large_tub + num_small_tubs * cost_small_tub = total_cost := by
  sorry

end NUMINAMATH_CALUDE_alison_large_tubs_l2183_218387


namespace NUMINAMATH_CALUDE_overlapping_sectors_area_l2183_218354

/-- The area of the overlapping region of two 45° sectors in a circle with radius 15 -/
theorem overlapping_sectors_area (r : ℝ) (angle : ℝ) : 
  r = 15 → angle = 45 → 
  2 * (angle / 360 * π * r^2 - 1/2 * r^2 * Real.sin (angle * π / 180)) = 225/4 * (π - 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_overlapping_sectors_area_l2183_218354


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2183_218335

/-- Triangle ABC with given conditions -/
structure TriangleABC where
  -- Vertex A coordinates
  A : ℝ × ℝ
  -- Equation of line containing median CM on side AB
  median_CM_eq : ℝ → ℝ → ℝ
  -- Equation of line containing altitude BH on side AC
  altitude_BH_eq : ℝ → ℝ → ℝ
  -- Conditions
  h_A : A = (5, 1)
  h_median_CM : ∀ x y, median_CM_eq x y = 2*x - y - 5
  h_altitude_BH : ∀ x y, altitude_BH_eq x y = x - 2*y - 5

/-- Main theorem about Triangle ABC -/
theorem triangle_abc_properties (t : TriangleABC) :
  -- 1. Coordinates of vertex C
  ∃ C : ℝ × ℝ, C = (4, 3) ∧
  -- 2. Length of AC
  Real.sqrt ((C.1 - t.A.1)^2 + (C.2 - t.A.2)^2) = Real.sqrt 5 ∧
  -- 3. Equation of line BC
  ∃ BC_eq : ℝ → ℝ → ℝ, (∀ x y, BC_eq x y = 6*x - 5*y - 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2183_218335


namespace NUMINAMATH_CALUDE_inequality_solution_l2183_218380

def choose (n k : ℕ) : ℕ := Nat.choose n k

def permute (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem inequality_solution (x : ℕ) :
  x > 0 → (choose 5 x + permute x 3 < 30 ↔ x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2183_218380


namespace NUMINAMATH_CALUDE_joes_gym_people_l2183_218336

/-- The number of people in Joe's Gym during Bethany's shift --/
theorem joes_gym_people (W A : ℕ) : 
  W + A + 5 + 2 - 3 - 4 + 2 = 20 → W + A = 18 := by
  sorry

#check joes_gym_people

end NUMINAMATH_CALUDE_joes_gym_people_l2183_218336


namespace NUMINAMATH_CALUDE_prize_winning_probability_l2183_218329

def num_card_types : ℕ := 3
def num_bags : ℕ := 4

def winning_probability : ℚ :=
  1 - (num_card_types.choose 2 * 2^num_bags - num_card_types) / num_card_types^num_bags

theorem prize_winning_probability :
  winning_probability = 4/9 := by sorry

end NUMINAMATH_CALUDE_prize_winning_probability_l2183_218329


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2183_218369

/-- Represents a student number in the range [1, 1000] -/
def StudentNumber := Fin 1000

/-- The total number of students -/
def totalStudents : Nat := 1000

/-- The number of students to be selected in the sample -/
def sampleSize : Nat := 100

/-- The interval between selected students in systematic sampling -/
def samplingInterval : Nat := totalStudents / sampleSize

/-- Predicate to determine if a student number is selected in the systematic sample -/
def isSelected (n : StudentNumber) : Prop :=
  n.val % samplingInterval = 122 % samplingInterval

theorem systematic_sampling_theorem :
  isSelected ⟨121, by norm_num⟩ → isSelected ⟨926, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2183_218369


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_triangle_shape_l2183_218308

/-- Factorization of 2a^2 - 8a + 8 --/
theorem factorization_1 (a : ℝ) : 2*a^2 - 8*a + 8 = 2*(a-2)^2 := by sorry

/-- Factorization of x^2 - y^2 + 3x - 3y --/
theorem factorization_2 (x y : ℝ) : x^2 - y^2 + 3*x - 3*y = (x-y)*(x+y+3) := by sorry

/-- Shape of triangle ABC given a^2 - ab - ac + bc = 0 --/
theorem triangle_shape (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (eq : a^2 - a*b - a*c + b*c = 0) :
  (a = b ∨ a = c ∨ b = c) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_triangle_shape_l2183_218308


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_through_point_implies_a_l2183_218391

/-- A hyperbola with equation x²/a² - y²/4 = 1 where a > 0 -/
structure Hyperbola where
  a : ℝ
  a_pos : a > 0

/-- The asymptotes of the hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = (2/h.a) * x ∨ y = -(2/h.a) * x}

/-- Theorem stating that if one asymptote passes through (2, 1), then a = 4 -/
theorem hyperbola_asymptote_through_point_implies_a
  (h : Hyperbola)
  (asymptote_through_point : (2, 1) ∈ asymptotes h) :
  h.a = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_through_point_implies_a_l2183_218391


namespace NUMINAMATH_CALUDE_division_problem_solution_l2183_218339

/-- Represents the division problem with given conditions -/
structure DivisionProblem where
  D : ℕ  -- dividend
  d : ℕ  -- divisor
  q : ℕ  -- quotient
  r : ℕ  -- remainder
  P : ℕ  -- prime number
  h1 : D = d * q + r
  h2 : r = 6
  h3 : d = 5 * q
  h4 : d = 3 * r + 2
  h5 : ∃ k : ℕ, D = P * k
  h6 : ∃ n : ℕ, q = n * n
  h7 : Nat.Prime P

theorem division_problem_solution (prob : DivisionProblem) : prob.D = 86 ∧ ∃ k : ℕ, prob.D = prob.P * k := by
  sorry

#check division_problem_solution

end NUMINAMATH_CALUDE_division_problem_solution_l2183_218339


namespace NUMINAMATH_CALUDE_param_line_point_l2183_218348

/-- A parameterized line in 2D space -/
structure ParamLine where
  /-- The vector on the line at parameter t -/
  vector : ℝ → ℝ × ℝ

/-- Theorem: Given a parameterized line with known points, we can determine another point -/
theorem param_line_point (l : ParamLine)
  (h1 : l.vector 5 = (2, 1))
  (h2 : l.vector 6 = (5, -7)) :
  l.vector 1 = (-40, 113) := by
  sorry

end NUMINAMATH_CALUDE_param_line_point_l2183_218348


namespace NUMINAMATH_CALUDE_smaller_mold_radius_l2183_218374

/-- The radius of a smaller hemisphere-shaped mold when a large hemisphere-shaped bowl
    with radius 1 foot is evenly distributed into 64 congruent smaller molds. -/
theorem smaller_mold_radius : ℝ → ℝ → ℝ → Prop :=
  fun (large_radius : ℝ) (num_molds : ℝ) (small_radius : ℝ) =>
    large_radius = 1 ∧
    num_molds = 64 ∧
    (2/3 * Real.pi * large_radius^3) = (num_molds * (2/3 * Real.pi * small_radius^3)) →
    small_radius = 1/4

/-- Proof of the smaller_mold_radius theorem. -/
lemma prove_smaller_mold_radius : smaller_mold_radius 1 64 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_smaller_mold_radius_l2183_218374


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2183_218321

theorem quadratic_inequality_solution_set (c : ℝ) (h : c > 1) :
  {x : ℝ | x^2 - (c + 1/c)*x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2183_218321


namespace NUMINAMATH_CALUDE_square_area_decrease_l2183_218377

theorem square_area_decrease (s : ℝ) (h : s > 0) :
  let initial_area := s^2
  let new_side := s * 0.9
  let new_area := new_side * s
  (initial_area - new_area) / initial_area * 100 = 19 := by
  sorry

end NUMINAMATH_CALUDE_square_area_decrease_l2183_218377


namespace NUMINAMATH_CALUDE_samantha_bedtime_l2183_218356

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Calculates the bedtime given wake-up time and sleep duration -/
def calculateBedtime (wakeUpTime : Time) (sleepDuration : Nat) : Time :=
  let totalMinutes := wakeUpTime.hour * 60 + wakeUpTime.minute
  let bedtimeMinutes := (totalMinutes - sleepDuration * 60 + 24 * 60) % (24 * 60)
  { hour := bedtimeMinutes / 60, minute := bedtimeMinutes % 60 }

theorem samantha_bedtime :
  let wakeUpTime : Time := { hour := 11, minute := 0 }
  let sleepDuration : Nat := 6
  calculateBedtime wakeUpTime sleepDuration = { hour := 5, minute := 0 } := by
  sorry

end NUMINAMATH_CALUDE_samantha_bedtime_l2183_218356


namespace NUMINAMATH_CALUDE_johns_number_is_eleven_l2183_218333

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_switch (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem johns_number_is_eleven :
  ∃! x : ℕ, is_two_digit x ∧
    82 ≤ digit_switch (5 * x + 13) ∧
    digit_switch (5 * x + 13) ≤ 86 ∧
    x = 11 := by sorry

end NUMINAMATH_CALUDE_johns_number_is_eleven_l2183_218333


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2183_218304

/-- Calculate the profit percentage given the selling price and profit -/
theorem profit_percentage_calculation (selling_price profit : ℝ) :
  selling_price = 850 ∧ profit = 205 →
  abs ((profit / (selling_price - profit)) * 100 - 31.78) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2183_218304


namespace NUMINAMATH_CALUDE_arrangements_count_l2183_218332

/-- The number of distinct arrangements of 4 boys and 4 girls in a row,
    where girls cannot be at either end. -/
def num_arrangements : ℕ := 8640

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 4

/-- The total number of people -/
def total_people : ℕ := num_boys + num_girls

/-- Theorem stating that the number of distinct arrangements of 4 boys and 4 girls in a row,
    where girls cannot be at either end, is equal to 8640. -/
theorem arrangements_count :
  num_arrangements = (num_boys * (num_boys - 1)) * Nat.factorial (total_people - 2) :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l2183_218332


namespace NUMINAMATH_CALUDE_smallest_surface_area_of_glued_cubes_smallest_surface_area_proof_l2183_218358

/-- The smallest possible surface area of a polyhedron formed by gluing three cubes with volumes 1, 8, and 27 at their faces. -/
theorem smallest_surface_area_of_glued_cubes : ℝ :=
  let cube1 : ℝ := 1
  let cube2 : ℝ := 8
  let cube3 : ℝ := 27
  let surface_area : ℝ := 72
  surface_area

/-- Proof that the smallest possible surface area of a polyhedron formed by gluing three cubes with volumes 1, 8, and 27 at their faces is 72. -/
theorem smallest_surface_area_proof :
  smallest_surface_area_of_glued_cubes = 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_surface_area_of_glued_cubes_smallest_surface_area_proof_l2183_218358


namespace NUMINAMATH_CALUDE_cylinder_cut_area_l2183_218303

/-- The area of the newly exposed circular segment face when cutting a cylinder -/
theorem cylinder_cut_area (r h : ℝ) (h_r : r = 8) (h_h : h = 10) :
  let base_area := π * r^2
  let sector_area := (1/4) * base_area
  sector_area = 16 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_cut_area_l2183_218303


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l2183_218395

theorem more_girls_than_boys (total_students : ℕ) 
  (h_total : total_students = 42)
  (h_ratio : ∃ (x : ℕ), 3 * x + 4 * x = total_students) : 
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    4 * boys = 3 * girls ∧ 
    girls - boys = 6 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l2183_218395


namespace NUMINAMATH_CALUDE_area_sin_3x_l2183_218310

open Real MeasureTheory

/-- The area of a function f on [a, b] -/
noncomputable def area (f : ℝ → ℝ) (a b : ℝ) : ℝ := ∫ x in a..b, f x

/-- For any positive integer n, the area of sin(nx) on [0, π/n] is 2/n -/
axiom area_sin_nx (n : ℕ+) : area (fun x ↦ sin (n * x)) 0 (π / n) = 2 / n

/-- The area of sin(3x) on [0, π/3] is 2/3 -/
theorem area_sin_3x : area (fun x ↦ sin (3 * x)) 0 (π / 3) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_sin_3x_l2183_218310


namespace NUMINAMATH_CALUDE_average_salary_all_employees_l2183_218346

/-- Calculate the average salary of all employees in an office --/
theorem average_salary_all_employees 
  (avg_salary_officers : ℝ) 
  (avg_salary_non_officers : ℝ) 
  (num_officers : ℕ) 
  (num_non_officers : ℕ) 
  (h1 : avg_salary_officers = 440)
  (h2 : avg_salary_non_officers = 110)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 480) :
  (avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers) / (num_officers + num_non_officers) = 120 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_all_employees_l2183_218346


namespace NUMINAMATH_CALUDE_commute_time_difference_l2183_218307

theorem commute_time_difference (distance : ℝ) (speed_actual : ℝ) (speed_suggested : ℝ) :
  distance = 10 ∧ speed_actual = 30 ∧ speed_suggested = 25 →
  (distance / speed_suggested - distance / speed_actual) * 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_difference_l2183_218307


namespace NUMINAMATH_CALUDE_stream_speed_is_three_l2183_218342

/-- Represents the scenario of a rower traveling upstream and downstream -/
structure RiverJourney where
  distance : ℝ
  normalSpeedDiff : ℝ
  tripleSpeedDiff : ℝ

/-- Calculates the stream speed given a RiverJourney -/
def calculateStreamSpeed (journey : RiverJourney) : ℝ :=
  sorry

/-- Theorem stating that the stream speed is 3 for the given conditions -/
theorem stream_speed_is_three (journey : RiverJourney)
  (h1 : journey.distance = 21)
  (h2 : journey.normalSpeedDiff = 4)
  (h3 : journey.tripleSpeedDiff = 0.5) :
  calculateStreamSpeed journey = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_is_three_l2183_218342


namespace NUMINAMATH_CALUDE_dimitri_calories_l2183_218394

/-- Calculates the total calories consumed by Dimitri over two days -/
def calories_two_days (burgers_per_day : ℕ) (calories_per_burger : ℕ) : ℕ :=
  2 * burgers_per_day * calories_per_burger

/-- Proves that Dimitri consumes 120 calories over two days -/
theorem dimitri_calories : calories_two_days 3 20 = 120 := by
  sorry

end NUMINAMATH_CALUDE_dimitri_calories_l2183_218394


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l2183_218305

theorem fraction_zero_implies_x_equals_two (x : ℝ) : 
  (x^2 - 4)/(x + 2) = 0 → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l2183_218305


namespace NUMINAMATH_CALUDE_total_interest_earned_l2183_218371

def initial_investment : ℝ := 1500
def annual_interest_rate : ℝ := 0.12
def time_period : ℕ := 4

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem total_interest_earned :
  let final_amount := compound_interest initial_investment annual_interest_rate time_period
  final_amount - initial_investment = 862.2 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_earned_l2183_218371


namespace NUMINAMATH_CALUDE_tangent_length_specific_tangent_length_l2183_218312

/-- Given a circle with radius r, a point M at distance d from the center,
    and a line through M tangent to the circle at A, 
    the length of AM is sqrt(d^2 - r^2) -/
theorem tangent_length (r d : ℝ) (hr : r > 0) (hd : d > r) :
  let am := Real.sqrt (d^2 - r^2)
  am^2 = d^2 - r^2 := by sorry

/-- In a circle with radius 10, if a point M is 26 units away from the center
    and a line passing through M touches the circle at point A,
    then the length of AM is 24 units -/
theorem specific_tangent_length :
  let r : ℝ := 10
  let d : ℝ := 26
  let am := Real.sqrt (d^2 - r^2)
  am = 24 := by sorry

end NUMINAMATH_CALUDE_tangent_length_specific_tangent_length_l2183_218312


namespace NUMINAMATH_CALUDE_range_of_m_for_propositions_l2183_218378

theorem range_of_m_for_propositions (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ∨
  (∀ x : ℝ, 4*x^2 + 4*(m+2)*x + 1 ≠ 0) →
  m < -1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_for_propositions_l2183_218378


namespace NUMINAMATH_CALUDE_nonzero_term_count_correct_l2183_218375

/-- The number of nonzero terms in the expanded and simplified expression of (x+y+z+w)^2008 + (x-y-z-w)^2008 -/
def nonzeroTermCount : ℕ := 1010025

/-- The degree of the polynomial expression -/
def degree : ℕ := 2008

theorem nonzero_term_count_correct :
  nonzeroTermCount = (degree / 2 + 1)^2 :=
sorry

end NUMINAMATH_CALUDE_nonzero_term_count_correct_l2183_218375


namespace NUMINAMATH_CALUDE_enclosing_rectangle_exists_l2183_218301

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  is_convex : Convex ℝ vertices
  area : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  bottom_left : ℝ × ℝ
  top_right : ℝ × ℝ

/-- Checks if a polygon is enclosed within a rectangle -/
def enclosed (p : ConvexPolygon) (r : Rectangle) : Prop :=
  ∀ v ∈ p.vertices, 
    r.bottom_left.1 ≤ v.1 ∧ v.1 ≤ r.top_right.1 ∧
    r.bottom_left.2 ≤ v.2 ∧ v.2 ≤ r.top_right.2

/-- Calculates the area of a rectangle -/
def rectangle_area (r : Rectangle) : ℝ :=
  (r.top_right.1 - r.bottom_left.1) * (r.top_right.2 - r.bottom_left.2)

/-- The main theorem -/
theorem enclosing_rectangle_exists (p : ConvexPolygon) (h : p.area = 1) :
  ∃ r : Rectangle, enclosed p r ∧ rectangle_area r ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_enclosing_rectangle_exists_l2183_218301


namespace NUMINAMATH_CALUDE_inequality_proof_l2183_218361

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt (1 + 2*a) + Real.sqrt (1 + 2*b) ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2183_218361


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2183_218341

def tangent_line (f : ℝ → ℝ) (a : ℝ) (m : ℝ) (b : ℝ) :=
  ∀ x, f a + m * (x - a) = m * x + b

theorem tangent_line_sum (f : ℝ → ℝ) :
  tangent_line f 5 (-1) 8 → f 5 + (deriv f) 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2183_218341


namespace NUMINAMATH_CALUDE_separating_chord_length_l2183_218396

/-- Represents a hexagon inscribed in a circle -/
structure InscribedHexagon where
  -- The lengths of the sides
  side_lengths : Fin 6 → ℝ
  -- Condition that alternating sides have lengths 5 and 4
  alternating_sides : ∀ i : Fin 6, side_lengths i = if i % 2 = 0 then 5 else 4

/-- The chord that separates the hexagon into two trapezoids -/
def separating_chord (h : InscribedHexagon) : ℝ := sorry

/-- Theorem stating the length of the separating chord -/
theorem separating_chord_length (h : InscribedHexagon) :
  separating_chord h = 180 / 49 := by sorry

end NUMINAMATH_CALUDE_separating_chord_length_l2183_218396


namespace NUMINAMATH_CALUDE_jane_quiz_score_l2183_218306

/-- Represents the scoring system for a quiz -/
structure QuizScoring where
  correct : Int
  incorrect : Int
  unanswered : Int

/-- Represents a student's quiz results -/
structure QuizResults where
  total : Nat
  correct : Nat
  incorrect : Nat
  unanswered : Nat

/-- Calculates the final score based on quiz results and scoring system -/
def calculateScore (results : QuizResults) (scoring : QuizScoring) : Int :=
  results.correct * scoring.correct + 
  results.incorrect * scoring.incorrect + 
  results.unanswered * scoring.unanswered

/-- Theorem: Jane's final score in the quiz is 20 -/
theorem jane_quiz_score : 
  let scoring : QuizScoring := ⟨2, -1, 0⟩
  let results : QuizResults := ⟨30, 15, 10, 5⟩
  calculateScore results scoring = 20 := by
  sorry


end NUMINAMATH_CALUDE_jane_quiz_score_l2183_218306


namespace NUMINAMATH_CALUDE_binary_to_decimal_example_l2183_218315

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number -/
def binary_number : List Nat := [1, 1, 0, 1, 1, 1, 1, 0, 1]

/-- Theorem stating that the given binary number is equal to 379 in decimal -/
theorem binary_to_decimal_example : binary_to_decimal binary_number = 379 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_example_l2183_218315


namespace NUMINAMATH_CALUDE_fraction_equality_l2183_218302

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2183_218302


namespace NUMINAMATH_CALUDE_exists_max_in_finite_list_l2183_218393

theorem exists_max_in_finite_list : 
  ∀ (L : List ℝ), L.length = 1000 → ∃ (m : ℝ), ∀ (x : ℝ), x ∈ L → x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_exists_max_in_finite_list_l2183_218393


namespace NUMINAMATH_CALUDE_ride_to_total_ratio_l2183_218372

def total_money : ℚ := 30
def dessert_cost : ℚ := 5
def money_left : ℚ := 10

theorem ride_to_total_ratio : 
  (total_money - dessert_cost - money_left) / total_money = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ride_to_total_ratio_l2183_218372


namespace NUMINAMATH_CALUDE_smallest_common_factor_l2183_218370

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 43 → gcd (5 * m - 3) (11 * m + 4) = 1) ∧ 
  gcd (5 * 43 - 3) (11 * 43 + 4) > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l2183_218370


namespace NUMINAMATH_CALUDE_restaurant_tip_percentage_l2183_218327

/-- Calculates the tip percentage given the cost of an appetizer, number of entrees,
    cost per entree, and total amount spent at a restaurant. -/
theorem restaurant_tip_percentage
  (appetizer_cost : ℚ)
  (num_entrees : ℕ)
  (entree_cost : ℚ)
  (total_spent : ℚ)
  (h1 : appetizer_cost = 10)
  (h2 : num_entrees = 4)
  (h3 : entree_cost = 20)
  (h4 : total_spent = 108) :
  (total_spent - (appetizer_cost + num_entrees * entree_cost)) / (appetizer_cost + num_entrees * entree_cost) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_tip_percentage_l2183_218327


namespace NUMINAMATH_CALUDE_quadratic_properties_l2183_218309

/-- The quadratic function f(x) = x^2 - 4x - 5 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 5

theorem quadratic_properties :
  (∀ x, f x ≥ -9) ∧ 
  (f 5 = 0 ∧ f (-1) = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2183_218309


namespace NUMINAMATH_CALUDE_visit_all_points_prob_one_l2183_218382

/-- Represents a one-dimensional random walk --/
structure RandomWalk where
  p : ℝ  -- Probability of moving right or left
  r : ℝ  -- Probability of staying in place
  prob_sum : p + p + r = 1  -- Sum of probabilities equals 1

/-- The probability of eventually reaching any point from any starting position --/
def eventual_visit_prob (rw : RandomWalk) : ℝ → ℝ := sorry

/-- Theorem stating that if p > 0, the probability of visiting any point is 1 --/
theorem visit_all_points_prob_one (rw : RandomWalk) (h : rw.p > 0) :
  ∀ x, eventual_visit_prob rw x = 1 := by sorry

end NUMINAMATH_CALUDE_visit_all_points_prob_one_l2183_218382


namespace NUMINAMATH_CALUDE_jennas_tanning_schedule_l2183_218352

/-- Jenna's tanning schedule problem -/
theorem jennas_tanning_schedule 
  (total_time : ℕ) 
  (daily_time : ℕ) 
  (last_two_weeks_time : ℕ) 
  (h1 : total_time = 200)
  (h2 : daily_time = 30)
  (h3 : last_two_weeks_time = 80) :
  (total_time - last_two_weeks_time) / (2 * daily_time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_jennas_tanning_schedule_l2183_218352


namespace NUMINAMATH_CALUDE_marksmen_hit_probability_l2183_218314

theorem marksmen_hit_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.6) (h2 : p2 = 0.7) (h3 : p3 = 0.75) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 0.97 := by
  sorry

end NUMINAMATH_CALUDE_marksmen_hit_probability_l2183_218314


namespace NUMINAMATH_CALUDE_shoulder_width_conversion_l2183_218353

/-- Converts centimeters to millimeters -/
def cm_to_mm (cm : ℝ) : ℝ := cm * 10

theorem shoulder_width_conversion :
  let cm_per_m : ℝ := 100
  let mm_per_m : ℝ := 1000
  let shoulder_width_cm : ℝ := 45
  cm_to_mm shoulder_width_cm = 450 := by
  sorry

end NUMINAMATH_CALUDE_shoulder_width_conversion_l2183_218353


namespace NUMINAMATH_CALUDE_sqrt_72_equals_6_sqrt_2_l2183_218343

theorem sqrt_72_equals_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_72_equals_6_sqrt_2_l2183_218343


namespace NUMINAMATH_CALUDE_train_length_train_length_correct_l2183_218350

/-- Represents the scenario of two people walking alongside a moving train --/
structure TrainScenario where
  train_speed : ℝ
  walking_speed : ℝ
  person_a_distance : ℝ
  person_b_distance : ℝ
  (train_speed_positive : train_speed > 0)
  (walking_speed_positive : walking_speed > 0)
  (person_a_distance_positive : person_a_distance > 0)
  (person_b_distance_positive : person_b_distance > 0)
  (person_a_distance_eq : person_a_distance = 45)
  (person_b_distance_eq : person_b_distance = 30)

/-- The theorem stating that given the conditions, the train length is 180 meters --/
theorem train_length (scenario : TrainScenario) : ℝ :=
  180

/-- The main theorem proving that the train length is correct --/
theorem train_length_correct (scenario : TrainScenario) :
  train_length scenario = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_correct_l2183_218350


namespace NUMINAMATH_CALUDE_not_passed_implies_scored_less_than_90_percent_l2183_218373

-- Define the proposition for scoring at least 90% on the final exam
def scored_at_least_90_percent (student : Type) : Prop := sorry

-- Define the proposition for passing the course
def passed_course (student : Type) : Prop := sorry

-- State the given condition
axiom condition (student : Type) : passed_course student → scored_at_least_90_percent student

-- State the theorem to be proved
theorem not_passed_implies_scored_less_than_90_percent (student : Type) :
  ¬(passed_course student) → ¬(scored_at_least_90_percent student) := by sorry

end NUMINAMATH_CALUDE_not_passed_implies_scored_less_than_90_percent_l2183_218373


namespace NUMINAMATH_CALUDE_milk_water_solution_volume_l2183_218319

theorem milk_water_solution_volume 
  (initial_milk_percentage : ℝ) 
  (final_milk_percentage : ℝ) 
  (added_water : ℝ) 
  (initial_milk_percentage_value : initial_milk_percentage = 0.84)
  (final_milk_percentage_value : final_milk_percentage = 0.58)
  (added_water_value : added_water = 26.9) : 
  ∃ (initial_volume : ℝ), 
    initial_volume > 0 ∧ 
    initial_milk_percentage * initial_volume / (initial_volume + added_water) = final_milk_percentage ∧
    initial_volume = 60 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_solution_volume_l2183_218319


namespace NUMINAMATH_CALUDE_yield_contradiction_l2183_218326

theorem yield_contradiction (x y z : ℝ) : ¬(0.4 * z + 0.2 * x = 1 ∧
                                           0.1 * y - 0.1 * z = -0.5 ∧
                                           0.1 * x + 0.2 * y = 4) := by
  sorry

end NUMINAMATH_CALUDE_yield_contradiction_l2183_218326


namespace NUMINAMATH_CALUDE_hyperbola_equation_proof_l2183_218366

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The general form of the hyperbola is x²/a² - y²/b² = 1 -/
  a : ℝ
  b : ℝ
  /-- One focus of the hyperbola is at (2,0) -/
  focus_x : a = 2
  /-- The equations of the asymptotes are y = ±√3x -/
  asymptote_slope : b / a = Real.sqrt 3

/-- The equation of the hyperbola with given properties -/
def hyperbola_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 / 3 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Theorem stating that the hyperbola with given properties has the equation x² - y²/3 = 1 -/
theorem hyperbola_equation_proof (h : Hyperbola) : hyperbola_equation h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_proof_l2183_218366


namespace NUMINAMATH_CALUDE_constant_term_of_equation_l2183_218388

/-- The constant term of a quadratic equation ax^2 + bx + c = 0 is c -/
def constant_term (a b c : ℝ) : ℝ := c

theorem constant_term_of_equation :
  constant_term 3 1 5 = 5 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_equation_l2183_218388


namespace NUMINAMATH_CALUDE_scaling_transformation_result_l2183_218359

/-- The scaling transformation applied to a point (x, y) -/
def scaling (x y : ℝ) : ℝ × ℝ := (x, 3 * y)

/-- The original curve C: x^2 + 9y^2 = 9 -/
def original_curve (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

/-- The transformed curve -/
def transformed_curve (x' y' : ℝ) : Prop := x'^2 + y'^2 = 9

/-- Theorem stating that the scaling transformation of the original curve
    results in the transformed curve -/
theorem scaling_transformation_result :
  ∀ x y : ℝ, original_curve x y →
  let (x', y') := scaling x y
  transformed_curve x' y' := by
  sorry

end NUMINAMATH_CALUDE_scaling_transformation_result_l2183_218359


namespace NUMINAMATH_CALUDE_quadratic_sign_l2183_218334

/-- A quadratic function of the form f(x) = x^2 + x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + x + c

theorem quadratic_sign (c : ℝ) (p : ℝ) 
  (h1 : f c 0 > 0) 
  (h2 : f c p < 0) : 
  f c (p + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sign_l2183_218334


namespace NUMINAMATH_CALUDE_prism_height_l2183_218337

/-- Represents a triangular prism with a regular triangular base -/
structure TriangularPrism where
  baseSideLength : ℝ
  totalEdgeLength : ℝ
  height : ℝ

/-- Theorem: The height of a specific triangular prism -/
theorem prism_height (p : TriangularPrism) 
  (h1 : p.baseSideLength = 10)
  (h2 : p.totalEdgeLength = 84) :
  p.height = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_height_l2183_218337
