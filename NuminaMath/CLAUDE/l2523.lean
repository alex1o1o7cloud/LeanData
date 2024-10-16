import Mathlib

namespace NUMINAMATH_CALUDE_power_sum_zero_l2523_252349

theorem power_sum_zero : (-2 : ℤ) ^ (3^2) + 2 ^ (3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_zero_l2523_252349


namespace NUMINAMATH_CALUDE_train_passing_time_l2523_252303

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 280 →
  train_speed_kmh = 72 →
  passing_time = 14 →
  passing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l2523_252303


namespace NUMINAMATH_CALUDE_total_vehicles_is_2800_l2523_252355

/-- Calculates the total number of vehicles on a road with given conditions -/
def totalVehicles (lanes : ℕ) (trucksPerLane : ℕ) (busesPerLane : ℕ) : ℕ :=
  let totalTrucks := lanes * trucksPerLane
  let carsPerLane := 2 * totalTrucks
  let totalCars := lanes * carsPerLane
  let totalBuses := lanes * busesPerLane
  let motorcyclesPerLane := 3 * busesPerLane
  let totalMotorcycles := lanes * motorcyclesPerLane
  totalTrucks + totalCars + totalBuses + totalMotorcycles

/-- Theorem stating that under the given conditions, the total number of vehicles is 2800 -/
theorem total_vehicles_is_2800 : totalVehicles 4 60 40 = 2800 := by
  sorry

#eval totalVehicles 4 60 40

end NUMINAMATH_CALUDE_total_vehicles_is_2800_l2523_252355


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2523_252317

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (|x₁ + 8| = Real.sqrt 256) ∧
  (|x₂ + 8| = Real.sqrt 256) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 32 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2523_252317


namespace NUMINAMATH_CALUDE_system_solution_l2523_252395

theorem system_solution (p q r s t : ℝ) :
  p^2 + q^2 + r^2 = 6 ∧ p * q - s^2 - t^2 = 3 →
  ((p = Real.sqrt 3 ∧ q = Real.sqrt 3 ∧ r = 0 ∧ s = 0 ∧ t = 0) ∨
   (p = -Real.sqrt 3 ∧ q = -Real.sqrt 3 ∧ r = 0 ∧ s = 0 ∧ t = 0)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2523_252395


namespace NUMINAMATH_CALUDE_min_total_distance_l2523_252362

/-- The number of trees planted -/
def num_trees : ℕ := 20

/-- The distance between adjacent trees in meters -/
def tree_distance : ℕ := 10

/-- The function that calculates the total distance traveled for a given tree position -/
def total_distance (n : ℕ) : ℕ :=
  10 * n^2 - 210 * n + 2100

/-- The theorem stating that the minimum total distance is 2000 meters -/
theorem min_total_distance :
  ∃ (n : ℕ), n > 0 ∧ n ≤ num_trees ∧ total_distance n = 2000 ∧
  ∀ (m : ℕ), m > 0 → m ≤ num_trees → total_distance m ≥ 2000 :=
sorry

end NUMINAMATH_CALUDE_min_total_distance_l2523_252362


namespace NUMINAMATH_CALUDE_solve_equation_l2523_252399

/-- Given the equation fp - w = 20000, where f = 10 and w = 10 + 250i, prove that p = 2001 + 25i -/
theorem solve_equation (f w p : ℂ) : 
  f = 10 → w = 10 + 250 * I → f * p - w = 20000 → p = 2001 + 25 * I := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2523_252399


namespace NUMINAMATH_CALUDE_simple_random_sampling_probability_l2523_252339

theorem simple_random_sampling_probability 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 8)
  (h2 : sample_size = 4) :
  (sample_size : ℚ) / population_size = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_random_sampling_probability_l2523_252339


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2523_252375

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  (∀ x, equation x = 0) → sum_of_roots = 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2523_252375


namespace NUMINAMATH_CALUDE_product_one_sum_greater_than_reciprocals_l2523_252336

theorem product_one_sum_greater_than_reciprocals 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > 1/a + 1/b + 1/c) : 
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ 
  (a < 1 ∧ b > 1 ∧ c < 1) ∨ 
  (a < 1 ∧ b < 1 ∧ c > 1) :=
sorry

end NUMINAMATH_CALUDE_product_one_sum_greater_than_reciprocals_l2523_252336


namespace NUMINAMATH_CALUDE_weight_lowering_feel_l2523_252342

theorem weight_lowering_feel (num_plates : ℕ) (weight_per_plate : ℝ) (increase_percentage : ℝ) :
  num_plates = 10 →
  weight_per_plate = 30 →
  increase_percentage = 0.2 →
  (num_plates : ℝ) * weight_per_plate * (1 + increase_percentage) = 360 := by
  sorry

end NUMINAMATH_CALUDE_weight_lowering_feel_l2523_252342


namespace NUMINAMATH_CALUDE_dans_initial_money_l2523_252326

/-- Calculates the initial amount of money given the number of items bought,
    the cost per item, and the amount left after purchase. -/
def initialMoney (itemsBought : ℕ) (costPerItem : ℕ) (amountLeft : ℕ) : ℕ :=
  itemsBought * costPerItem + amountLeft

/-- Theorem stating that given the specific conditions of Dan's purchase,
    his initial amount of money was $298. -/
theorem dans_initial_money :
  initialMoney 99 3 1 = 298 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_money_l2523_252326


namespace NUMINAMATH_CALUDE_apple_cost_l2523_252322

theorem apple_cost (initial_apples : ℕ) (initial_oranges : ℕ) (orange_cost : ℚ) 
  (final_apples : ℕ) (final_oranges : ℕ) (total_earnings : ℚ) :
  initial_apples = 50 →
  initial_oranges = 40 →
  orange_cost = 1/2 →
  final_apples = 10 →
  final_oranges = 6 →
  total_earnings = 49 →
  ∃ (apple_cost : ℚ), apple_cost = 4/5 := by
  sorry

#check apple_cost

end NUMINAMATH_CALUDE_apple_cost_l2523_252322


namespace NUMINAMATH_CALUDE_toms_lifting_capacity_l2523_252378

/-- Calculates the total weight Tom can lift after training -/
def totalWeightAfterTraining (initialCapacity : ℝ) : ℝ :=
  let afterIntensiveTraining := initialCapacity * (1 + 1.5)
  let afterSpecialization := afterIntensiveTraining * (1 + 0.25)
  let afterNewGripTechnique := afterSpecialization * (1 + 0.1)
  2 * afterNewGripTechnique

/-- Theorem stating that Tom's final lifting capacity is 687.5 kg -/
theorem toms_lifting_capacity :
  totalWeightAfterTraining 100 = 687.5 := by
  sorry

#eval totalWeightAfterTraining 100

end NUMINAMATH_CALUDE_toms_lifting_capacity_l2523_252378


namespace NUMINAMATH_CALUDE_book_price_reduction_l2523_252386

theorem book_price_reduction (original_price : ℝ) : 
  original_price = 20 → 
  (original_price * (1 - 0.25) * (1 - 0.40) = 9) := by
  sorry

end NUMINAMATH_CALUDE_book_price_reduction_l2523_252386


namespace NUMINAMATH_CALUDE_seven_prime_pairs_l2523_252341

/-- A function that returns true if n is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that counts the number of pairs of distinct primes p and q such that p^2 * q^2 < n -/
def countPrimePairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that there are exactly 7 pairs of distinct primes p and q such that p^2 * q^2 < 1000 -/
theorem seven_prime_pairs :
  countPrimePairs 1000 = 7 := by sorry

end NUMINAMATH_CALUDE_seven_prime_pairs_l2523_252341


namespace NUMINAMATH_CALUDE_swim_time_proof_l2523_252315

-- Define the given constants
def downstream_distance : ℝ := 16
def upstream_distance : ℝ := 10
def still_water_speed : ℝ := 6.5

-- Define the theorem
theorem swim_time_proof :
  ∃ (t c : ℝ),
    t > 0 ∧
    c ≥ 0 ∧
    c < still_water_speed ∧
    downstream_distance / (still_water_speed + c) = t ∧
    upstream_distance / (still_water_speed - c) = t ∧
    t = 2 := by
  sorry

end NUMINAMATH_CALUDE_swim_time_proof_l2523_252315


namespace NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l2523_252330

theorem polar_to_cartesian_conversion :
  let r : ℝ := 2
  let θ : ℝ := π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (Real.sqrt 3, 1) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l2523_252330


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_five_fourths_l2523_252334

theorem trigonometric_expression_equals_five_fourths :
  Real.sqrt 2 * Real.cos (π / 4) - Real.sin (π / 3) ^ 2 + Real.tan (π / 4) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_five_fourths_l2523_252334


namespace NUMINAMATH_CALUDE_problem_solution_l2523_252381

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) := Real.log x + a
noncomputable def h (x : ℝ) := x * f x

theorem problem_solution :
  (∃ (x_min : ℝ), ∀ (x : ℝ), h x ≥ h x_min ∧ h x_min = -1 / Real.exp 1) ∧
  (∀ (a : ℝ), (∃! (p : ℝ), f p = g a p) →
    (∃ (p : ℝ), f p = g a p ∧
      (deriv f p : ℝ) = (deriv (g a) p : ℝ) ∧
      2 < a ∧ a < 5/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2523_252381


namespace NUMINAMATH_CALUDE_bill_denomination_l2523_252305

-- Define the problem parameters
def total_bill : ℕ := 285
def coin_value : ℕ := 5
def total_items : ℕ := 24
def num_bills : ℕ := 11
def num_coins : ℕ := 11

-- Theorem to prove
theorem bill_denomination :
  ∃ (x : ℕ), 
    x * num_bills + coin_value * num_coins = total_bill ∧
    num_bills + num_coins = total_items ∧
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_bill_denomination_l2523_252305


namespace NUMINAMATH_CALUDE_equidistant_points_on_line_in_quadrants_I_and_IV_l2523_252331

/-- The line equation 4x + 3y = 12 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y = 12

/-- A point (x, y) is equidistant from coordinate axes if |x| = |y| -/
def equidistant_from_axes (x y : ℝ) : Prop := abs x = abs y

/-- Quadrant I: x > 0 and y > 0 -/
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Quadrant IV: x > 0 and y < 0 -/
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The main theorem: points on the line 4x + 3y = 12 that are equidistant 
    from coordinate axes exist only in quadrants I and IV -/
theorem equidistant_points_on_line_in_quadrants_I_and_IV :
  ∀ x y : ℝ, line_equation x y → equidistant_from_axes x y →
  (in_quadrant_I x y ∨ in_quadrant_IV x y) ∧
  ¬(∃ x y : ℝ, line_equation x y ∧ equidistant_from_axes x y ∧ 
    ¬(in_quadrant_I x y ∨ in_quadrant_IV x y)) :=
sorry

end NUMINAMATH_CALUDE_equidistant_points_on_line_in_quadrants_I_and_IV_l2523_252331


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l2523_252307

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l m : Line) (α β : Plane)
  (h_diff_lines : l ≠ m)
  (h_diff_planes : α ≠ β)
  (h_parallel : parallel_line_plane l α)
  (h_perpendicular : perpendicular_line_plane l β) :
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l2523_252307


namespace NUMINAMATH_CALUDE_peanuts_remaining_l2523_252312

theorem peanuts_remaining (initial : ℕ) (eaten_by_bonita : ℕ) : 
  initial = 148 → 
  eaten_by_bonita = 29 → 
  82 = initial - (initial / 4) - eaten_by_bonita := by
  sorry

end NUMINAMATH_CALUDE_peanuts_remaining_l2523_252312


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l2523_252351

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the operations and relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Set Point)
variable (planes_parallel : Plane → Plane → Prop)

-- Define the given lines and planes
variable (m n l₁ l₂ : Line)
variable (α β : Plane)
variable (M : Point)

theorem parallel_planes_condition
  (h1 : subset m α)
  (h2 : subset n α)
  (h3 : subset l₁ β)
  (h4 : subset l₂ β)
  (h5 : intersect l₁ l₂ = {M})
  (h6 : parallel m l₁)
  (h7 : parallel n l₂) :
  planes_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l2523_252351


namespace NUMINAMATH_CALUDE_work_completion_time_l2523_252311

theorem work_completion_time (a b : ℕ) (h1 : a = 20) (h2 : (4 : ℝ) * ((1 : ℝ) / a + (1 : ℝ) / b) = (1 : ℝ) / 3) : b = 30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2523_252311


namespace NUMINAMATH_CALUDE_warehouse_inventory_l2523_252360

theorem warehouse_inventory (x y : ℝ) : 
  x + y = 92 ∧ 
  (2/5) * x + (1/4) * y = 26 → 
  x = 20 ∧ y = 72 := by
sorry

end NUMINAMATH_CALUDE_warehouse_inventory_l2523_252360


namespace NUMINAMATH_CALUDE_A_power_50_l2523_252353

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem A_power_50 : A^50 = !![(-301 : ℤ), -100; 800, 299] := by sorry

end NUMINAMATH_CALUDE_A_power_50_l2523_252353


namespace NUMINAMATH_CALUDE_subscription_difference_is_5000_l2523_252332

/-- Represents the subscription amounts and profit distribution in a business venture -/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  a_profit : ℕ
  a_extra : ℕ

/-- Calculates the difference between B's and C's subscriptions -/
def subscription_difference (bv : BusinessVenture) : ℕ :=
  let b_subscription := (bv.total_subscription * bv.a_profit * 2) / (bv.total_profit * 3) - bv.a_extra
  let c_subscription := bv.total_subscription - b_subscription - (b_subscription + bv.a_extra)
  b_subscription - c_subscription

/-- Theorem stating that the difference between B's and C's subscriptions is 5000 -/
theorem subscription_difference_is_5000 (bv : BusinessVenture) 
    (h1 : bv.total_subscription = 50000)
    (h2 : bv.total_profit = 35000)
    (h3 : bv.a_profit = 14700)
    (h4 : bv.a_extra = 4000) :
  subscription_difference bv = 5000 := by
  sorry

#eval subscription_difference ⟨50000, 35000, 14700, 4000⟩

end NUMINAMATH_CALUDE_subscription_difference_is_5000_l2523_252332


namespace NUMINAMATH_CALUDE_count_integers_satisfying_conditions_l2523_252309

theorem count_integers_satisfying_conditions :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ 9 ∣ n ∧ Nat.lcm (Nat.factorial 6) n = 9 * Nat.gcd (Nat.factorial 9) n) ∧
    S.card = 30 ∧
    (∀ n : ℕ, n > 0 → 9 ∣ n → Nat.lcm (Nat.factorial 6) n = 9 * Nat.gcd (Nat.factorial 9) n → n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_conditions_l2523_252309


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l2523_252313

theorem geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = 1/4) 
  (h2 : S = 80) 
  (h3 : S = a / (1 - r)) : 
  a = 60 := by sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l2523_252313


namespace NUMINAMATH_CALUDE_zookeeper_excess_fish_l2523_252338

/-- The number of penguins in the zoo -/
def total_penguins : ℕ := 48

/-- The ratio of Emperor to Adelie penguins -/
def emperor_ratio : ℕ := 3
def adelie_ratio : ℕ := 5

/-- The amount of fish needed for each type of penguin -/
def emperor_fish_need : ℚ := 3/2
def adelie_fish_need : ℕ := 2

/-- The percentage of additional fish the zookeeper has -/
def additional_fish_percentage : ℕ := 150

theorem zookeeper_excess_fish :
  let emperor_count : ℕ := (emperor_ratio * total_penguins) / (emperor_ratio + adelie_ratio)
  let adelie_count : ℕ := (adelie_ratio * total_penguins) / (emperor_ratio + adelie_ratio)
  let total_fish_needed : ℚ := emperor_count * emperor_fish_need + adelie_count * adelie_fish_need
  let zookeeper_fish : ℕ := total_penguins + (additional_fish_percentage * total_penguins) / 100
  (zookeeper_fish : ℚ) - total_fish_needed = 33 := by
  sorry

end NUMINAMATH_CALUDE_zookeeper_excess_fish_l2523_252338


namespace NUMINAMATH_CALUDE_max_value_of_sine_sum_l2523_252333

theorem max_value_of_sine_sum (x y z : Real) 
  (hx : x ∈ Set.Icc 0 Real.pi) 
  (hy : y ∈ Set.Icc 0 Real.pi) 
  (hz : z ∈ Set.Icc 0 Real.pi) : 
  Real.sin (x - y) + Real.sin (y - z) + Real.sin (z - x) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sine_sum_l2523_252333


namespace NUMINAMATH_CALUDE_grocery_shopping_problem_l2523_252385

theorem grocery_shopping_problem (initial_budget : ℚ) (bread_cost : ℚ) (candy_cost : ℚ) 
  (h1 : initial_budget = 32)
  (h2 : bread_cost = 3)
  (h3 : candy_cost = 2) : 
  let remaining_after_bread_candy := initial_budget - (bread_cost + candy_cost)
  let turkey_cost := (1 / 3) * remaining_after_bread_candy
  initial_budget - (bread_cost + candy_cost + turkey_cost) = 18 := by
sorry

end NUMINAMATH_CALUDE_grocery_shopping_problem_l2523_252385


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l2523_252376

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*y - 16 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 5

/-- Theorem stating that the radius of the circle is 5 -/
theorem circle_radius_is_five :
  ∀ x y : ℝ, circle_equation x y → ∃ center_x center_y : ℝ,
    (x - center_x)^2 + (y - center_y)^2 = circle_radius^2 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l2523_252376


namespace NUMINAMATH_CALUDE_volleyball_club_boys_count_l2523_252361

theorem volleyball_club_boys_count :
  ∀ (total_members boys girls present : ℕ),
  total_members = 30 →
  present = 18 →
  boys + girls = total_members →
  present = boys + girls / 3 →
  boys = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_club_boys_count_l2523_252361


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l2523_252365

/-- Represents the composition of a chemical solution -/
structure Solution where
  a : ℝ  -- Percentage of chemical a
  b : ℝ  -- Percentage of chemical b

/-- Represents a mixture of two solutions -/
structure Mixture where
  x : Solution  -- First solution
  y : Solution  -- Second solution
  x_ratio : ℝ   -- Ratio of solution x in the mixture

/-- The problem statement -/
theorem chemical_mixture_problem (x y : Solution) (m : Mixture) :
  x.a = 0.1 ∧                 -- Solution x is 10% chemical a
  x.b = 0.9 ∧                 -- Solution x is 90% chemical b
  y.b = 0.8 ∧                 -- Solution y is 80% chemical b
  m.x = x ∧                   -- Mixture contains solution x
  m.y = y ∧                   -- Mixture contains solution y
  m.x_ratio = 0.8 ∧           -- 80% of the mixture is solution x
  m.x_ratio * x.a + (1 - m.x_ratio) * y.a = 0.12  -- Mixture is 12% chemical a
  →
  y.a = 0.2                   -- Percentage of chemical a in solution y is 20%
  := by sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l2523_252365


namespace NUMINAMATH_CALUDE_soccer_substitutions_remainder_l2523_252329

/-- Number of players in a soccer team -/
def total_players : ℕ := 24

/-- Number of starting players -/
def starting_players : ℕ := 12

/-- Number of substitute players -/
def substitute_players : ℕ := 12

/-- Maximum number of substitutions allowed -/
def max_substitutions : ℕ := 4

/-- 
Calculate the number of ways to make substitutions in a soccer game
n: current number of substitutions made
-/
def substitution_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 12 * (13 - n) * substitution_ways (n - 1)

/-- 
The total number of ways to make substitutions is the sum of ways
to make 0, 1, 2, 3, and 4 substitutions
-/
def total_ways : ℕ := 
  (List.range 5).map substitution_ways |>.sum

/-- Main theorem to prove -/
theorem soccer_substitutions_remainder :
  total_ways % 1000 = 573 := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_remainder_l2523_252329


namespace NUMINAMATH_CALUDE_erasers_per_group_l2523_252392

theorem erasers_per_group (total_erasers : ℕ) (num_groups : ℕ) (erasers_per_group : ℕ) 
  (h1 : total_erasers = 270)
  (h2 : num_groups = 3)
  (h3 : erasers_per_group = total_erasers / num_groups) :
  erasers_per_group = 90 := by
  sorry

end NUMINAMATH_CALUDE_erasers_per_group_l2523_252392


namespace NUMINAMATH_CALUDE_article_cost_price_l2523_252325

theorem article_cost_price : ∃ (C : ℝ), 
  (C = 600) ∧ 
  (∃ (SP : ℝ), SP = 1.05 * C) ∧ 
  (∃ (SP_new C_new : ℝ), 
    C_new = 0.95 * C ∧ 
    SP_new = 1.05 * C - 3 ∧ 
    SP_new = 1.045 * C_new) :=
by sorry

end NUMINAMATH_CALUDE_article_cost_price_l2523_252325


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2523_252379

/-- For a parallelogram with given height and area, prove that its base length is as calculated. -/
theorem parallelogram_base_length (height area : ℝ) (h_height : height = 11) (h_area : area = 44) :
  area / height = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2523_252379


namespace NUMINAMATH_CALUDE_function_property_l2523_252335

/-- Given a function f(x) = ax^5 + bx^3 + 3 where f(2023) = 16, prove that f(-2023) = -10 -/
theorem function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^5 + b * x^3 + 3
  f 2023 = 16 → f (-2023) = -10 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2523_252335


namespace NUMINAMATH_CALUDE_convex_pentagon_exists_l2523_252377

-- Define the points
variable (A₁ A₂ A₃ A₄ A₅ A₆ A₇ A₈ A₉ : ℝ × ℝ)

-- Define the square
def is_square (A₁ A₂ A₃ A₄ : ℝ × ℝ) : Prop := sorry

-- Define the convex quadrilateral
def is_convex_quadrilateral (A₅ A₆ A₇ A₈ : ℝ × ℝ) : Prop := sorry

-- Define the point inside the quadrilateral
def point_inside_quadrilateral (A₉ A₅ A₆ A₇ A₈ : ℝ × ℝ) : Prop := sorry

-- Define the non-collinearity condition
def no_three_collinear (points : List (ℝ × ℝ)) : Prop := sorry

-- Define a convex pentagon
def is_convex_pentagon (points : List (ℝ × ℝ)) : Prop := sorry

theorem convex_pentagon_exists 
  (h1 : is_square A₁ A₂ A₃ A₄)
  (h2 : is_convex_quadrilateral A₅ A₆ A₇ A₈)
  (h3 : point_inside_quadrilateral A₉ A₅ A₆ A₇ A₈)
  (h4 : no_three_collinear [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉]) :
  ∃ (points : List (ℝ × ℝ)), points.length = 5 ∧ 
    (∀ p ∈ points, p ∈ [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉]) ∧ 
    is_convex_pentagon points :=
sorry

end NUMINAMATH_CALUDE_convex_pentagon_exists_l2523_252377


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l2523_252347

theorem average_marks_combined_classes (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) :
  students1 = 25 →
  students2 = 40 →
  avg1 = 50 →
  avg2 = 65 →
  let total_students := students1 + students2
  let total_marks := students1 * avg1 + students2 * avg2
  abs ((total_marks / total_students) - 59.23) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l2523_252347


namespace NUMINAMATH_CALUDE_unique_valid_number_l2523_252304

def is_valid_number (n : ℕ) : Prop :=
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  (100 * x + 10 * z + y = 64 * x + 8 * z + y) ∧ 
  (100 * y + 10 * x + z = 36 * y + 6 * x + z - 16) ∧
  (100 * z + 10 * y + x = 16 * z + 4 * y + x + 18)

theorem unique_valid_number : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ is_valid_number n :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2523_252304


namespace NUMINAMATH_CALUDE_student_average_age_l2523_252384

theorem student_average_age
  (n : ℕ) -- number of students
  (teacher_age : ℕ) -- age of the teacher
  (avg_increase : ℝ) -- increase in average when teacher is included
  (h1 : n = 25) -- there are 25 students
  (h2 : teacher_age = 52) -- teacher's age is 52
  (h3 : avg_increase = 1) -- average increases by 1 when teacher is included
  : (n : ℝ) * ((n + 1 : ℝ) * (x + avg_increase) - teacher_age) / n = 26 :=
by sorry

#check student_average_age

end NUMINAMATH_CALUDE_student_average_age_l2523_252384


namespace NUMINAMATH_CALUDE_certain_number_problem_l2523_252366

theorem certain_number_problem : ∃! x : ℝ, x / 9 + x + 9 = 69 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2523_252366


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2523_252368

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2523_252368


namespace NUMINAMATH_CALUDE_gillians_spending_proof_l2523_252373

/-- Calculates Gillian's spending at the farmer's market based on Sandi's initial amount -/
def gillians_spending (sandis_initial_amount : ℕ) : ℕ :=
  let sandis_spending := sandis_initial_amount / 2
  3 * sandis_spending + 150

theorem gillians_spending_proof (sandis_initial_amount : ℕ) 
  (h : sandis_initial_amount = 600) : 
  gillians_spending sandis_initial_amount = 1050 := by
  sorry

#eval gillians_spending 600

end NUMINAMATH_CALUDE_gillians_spending_proof_l2523_252373


namespace NUMINAMATH_CALUDE_system_solution_l2523_252387

theorem system_solution (x y z : ℝ) : 
  (x^2 + x - 1 = y ∧ y^2 + y - 1 = z ∧ z^2 + z - 1 = x) → 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2523_252387


namespace NUMINAMATH_CALUDE_candy_mixture_problem_l2523_252391

/-- Proves that the cost of candy B is 1.70 dollars per pound given the conditions of the candy mixture problem. -/
theorem candy_mixture_problem (total_weight : ℝ) (mixture_cost_per_pound : ℝ) (candy_A_cost : ℝ) (candy_A_weight : ℝ) :
  total_weight = 5 →
  mixture_cost_per_pound = 2 →
  candy_A_cost = 3.20 →
  candy_A_weight = 1 →
  ∃ (candy_B_cost : ℝ),
    candy_B_cost = 1.70 ∧
    total_weight * mixture_cost_per_pound = candy_A_weight * candy_A_cost + (total_weight - candy_A_weight) * candy_B_cost :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_problem_l2523_252391


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l2523_252343

theorem not_all_perfect_squares (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k^2 :=
by
  sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l2523_252343


namespace NUMINAMATH_CALUDE_part1_part2_l2523_252316

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + 3*x - 10 ≤ 0}

-- Define set B for part (1)
def B1 (m : ℝ) : Set ℝ := {x : ℝ | -2*m + 1 ≤ x ∧ x ≤ -m - 1}

-- Define set B for part (2)
def B2 (m : ℝ) : Set ℝ := {x : ℝ | -2*m + 1 ≤ x ∧ x ≤ -m - 1}

-- Theorem for part (1)
theorem part1 : ∀ m : ℝ, (A ∪ B1 m = A) → (2 < m ∧ m ≤ 3) := by sorry

-- Theorem for part (2)
theorem part2 : ∀ m : ℝ, (A ∪ B2 m = A) → m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2523_252316


namespace NUMINAMATH_CALUDE_circle_equation_problem1_circle_equation_problem2_l2523_252350

-- Problem 1
theorem circle_equation_problem1 (x y : ℝ) :
  (∃ (h : ℝ), x - 2*y - 2 = 0 ∧ 
    (x - 0)^2 + (y - 4)^2 = (x - 4)^2 + (y - 6)^2) →
  (x - 4)^2 + (y - 1)^2 = 25 :=
sorry

-- Problem 2
theorem circle_equation_problem2 (x y : ℝ) :
  (2*2 + 3*2 - 10 = 0 ∧
    ((x - 2)^2 + (y - 2)^2 = 13 ∧
     (y - 2)/(x - 2) * (-2/3) = -1)) →
  ((x - 4)^2 + (y - 5)^2 = 13 ∨ x^2 + (y + 1)^2 = 13) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_problem1_circle_equation_problem2_l2523_252350


namespace NUMINAMATH_CALUDE_inequality_proof_l2523_252371

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  Real.sqrt (b^2 - a*c) < Real.sqrt 3 * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2523_252371


namespace NUMINAMATH_CALUDE_second_point_x_coordinate_l2523_252345

/-- Given two points (m, n) and (m + 2, n + 1) on the line x = 2y + 3,
    the x-coordinate of the second point is m + 2. -/
theorem second_point_x_coordinate (m n : ℝ) : 
  (m = 2 * n + 3) → -- First point (m, n) lies on the line
  (m + 2 = 2 * (n + 1) + 3) → -- Second point (m + 2, n + 1) lies on the line
  (m + 2 = m + 2) -- The x-coordinate of the second point is m + 2
:= by sorry

end NUMINAMATH_CALUDE_second_point_x_coordinate_l2523_252345


namespace NUMINAMATH_CALUDE_min_seats_occupied_l2523_252327

theorem min_seats_occupied (total_seats : ℕ) (initial_occupied : ℕ) : 
  total_seats = 150 → initial_occupied = 2 → 
  (∃ (additional_seats : ℕ), 
    additional_seats = 49 ∧ 
    ∀ (x : ℕ), x < additional_seats → 
      ∃ (y : ℕ), y ≤ total_seats - initial_occupied - x ∧ 
      y ≥ 2 ∧ 
      ∀ (z : ℕ), z < y → (z = 1 ∨ z = y)) :=
by sorry

end NUMINAMATH_CALUDE_min_seats_occupied_l2523_252327


namespace NUMINAMATH_CALUDE_remainder_equality_l2523_252328

theorem remainder_equality (a b c : ℕ) :
  (2 * a + b) % 10 = (2 * b + c) % 10 ∧
  (2 * b + c) % 10 = (2 * c + a) % 10 →
  a % 10 = b % 10 ∧ b % 10 = c % 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l2523_252328


namespace NUMINAMATH_CALUDE_total_chimpanzees_l2523_252390

/- Define the number of chimps moving to the new cage -/
def chimps_new_cage : ℕ := 18

/- Define the number of chimps staying in the old cage -/
def chimps_old_cage : ℕ := 27

/- Theorem stating that the total number of chimpanzees is 45 -/
theorem total_chimpanzees : chimps_new_cage + chimps_old_cage = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_chimpanzees_l2523_252390


namespace NUMINAMATH_CALUDE_stratified_sampling_group_size_l2523_252314

theorem stratified_sampling_group_size 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (group_a_sample : ℕ) :
  total_population = 200 →
  sample_size = 40 →
  group_a_sample = 16 →
  (total_population - (total_population * group_a_sample / sample_size) : ℕ) = 120 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_size_l2523_252314


namespace NUMINAMATH_CALUDE_polygon_reassembly_l2523_252374

-- Define a polygon as a set of points in 2D space
def Polygon : Type := Set (ℝ × ℝ)

-- Define the area of a polygon
def area (P : Polygon) : ℝ := sorry

-- Define a function to represent cutting and reassembling a polygon
def can_reassemble (P Q : Polygon) : Prop := sorry

-- Define a rectangle with one side of length 1
def rectangle_with_unit_side (R : Polygon) : Prop := sorry

theorem polygon_reassembly (P Q : Polygon) (h : area P = area Q) :
  (∃ R : Polygon, can_reassemble P R ∧ rectangle_with_unit_side R) ∧
  can_reassemble P Q := by sorry

end NUMINAMATH_CALUDE_polygon_reassembly_l2523_252374


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2523_252364

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ - 1 = 0 ∧ k * x₂^2 - 2 * x₂ - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2523_252364


namespace NUMINAMATH_CALUDE_games_for_512_participants_l2523_252393

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  participants : ℕ
  is_power_of_two : ∃ n : ℕ, participants = 2^n

/-- Calculates the number of games required to determine a winner in a single-elimination tournament. -/
def games_required (tournament : SingleEliminationTournament) : ℕ :=
  tournament.participants - 1

/-- Theorem stating that a single-elimination tournament with 512 participants requires 511 games. -/
theorem games_for_512_participants :
  ∀ (tournament : SingleEliminationTournament),
  tournament.participants = 512 →
  games_required tournament = 511 := by
  sorry

#eval games_required ⟨512, ⟨9, rfl⟩⟩

end NUMINAMATH_CALUDE_games_for_512_participants_l2523_252393


namespace NUMINAMATH_CALUDE_wendy_trip_miles_l2523_252321

def three_day_trip (day1_miles day2_miles total_miles : ℕ) : Prop :=
  ∃ day3_miles : ℕ, day1_miles + day2_miles + day3_miles = total_miles

theorem wendy_trip_miles :
  three_day_trip 125 223 493 →
  ∃ day3_miles : ℕ, day3_miles = 145 :=
by sorry

end NUMINAMATH_CALUDE_wendy_trip_miles_l2523_252321


namespace NUMINAMATH_CALUDE_valid_sequence_count_l2523_252354

/-- Represents a sequence of coin tosses -/
def CoinSequence := List Bool

/-- Counts the number of occurrences of a specific subsequence in a coin sequence -/
def countSubsequence (seq : CoinSequence) (subseq : CoinSequence) : Nat :=
  sorry

/-- Checks if a coin sequence satisfies the given conditions -/
def isValidSequence (seq : CoinSequence) : Prop :=
  (countSubsequence seq [true, true] = 3) ∧
  (countSubsequence seq [true, false] = 4) ∧
  (countSubsequence seq [false, true] = 5) ∧
  (countSubsequence seq [false, false] = 6)

/-- The number of valid coin sequences -/
def validSequenceCount : Nat :=
  sorry

theorem valid_sequence_count :
  validSequenceCount = 16170 :=
sorry

end NUMINAMATH_CALUDE_valid_sequence_count_l2523_252354


namespace NUMINAMATH_CALUDE_cards_taken_away_l2523_252397

theorem cards_taken_away (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 67)
  (h2 : final_cards = 58) :
  initial_cards - final_cards = 9 := by
  sorry

end NUMINAMATH_CALUDE_cards_taken_away_l2523_252397


namespace NUMINAMATH_CALUDE_total_pencils_l2523_252383

/-- Given an initial number of pencils in a drawer and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l2523_252383


namespace NUMINAMATH_CALUDE_game_outcome_probability_l2523_252363

/-- Represents the probability of a specific outcome in a game with 8 rounds and 3 players. -/
def game_probability (p_alex p_mel p_chelsea : ℝ) : Prop :=
  p_alex = 1/2 ∧
  p_mel = 2 * p_chelsea ∧
  p_alex + p_mel + p_chelsea = 1 ∧
  0 ≤ p_alex ∧ p_alex ≤ 1 ∧
  0 ≤ p_mel ∧ p_mel ≤ 1 ∧
  0 ≤ p_chelsea ∧ p_chelsea ≤ 1

/-- The probability of a specific outcome in the game. -/
def outcome_probability (p_alex p_mel p_chelsea : ℝ) : ℝ :=
  (p_alex ^ 4) * (p_mel ^ 3) * p_chelsea

/-- The number of ways to arrange 4 wins for Alex, 3 for Mel, and 1 for Chelsea in 8 rounds. -/
def arrangements : ℕ := 
  Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 1)

/-- Theorem stating the probability of the specific game outcome. -/
theorem game_outcome_probability :
  ∀ p_alex p_mel p_chelsea : ℝ,
  game_probability p_alex p_mel p_chelsea →
  (arrangements : ℝ) * outcome_probability p_alex p_mel p_chelsea = 35/324 :=
by
  sorry


end NUMINAMATH_CALUDE_game_outcome_probability_l2523_252363


namespace NUMINAMATH_CALUDE_tims_trip_duration_l2523_252372

/-- Calculates the total duration of Tim's trip given the specified conditions -/
theorem tims_trip_duration :
  let total_driving_time : ℝ := 5
  let num_traffic_jams : ℕ := 3
  let first_jam_multiplier : ℝ := 1.5
  let second_jam_multiplier : ℝ := 2
  let third_jam_multiplier : ℝ := 3
  let num_pit_stops : ℕ := 2
  let pit_stop_duration : ℝ := 0.5
  let time_before_first_jam : ℝ := 1
  let time_between_first_and_second_jam : ℝ := 1.5

  let first_jam_duration : ℝ := first_jam_multiplier * time_before_first_jam
  let second_jam_duration : ℝ := second_jam_multiplier * time_between_first_and_second_jam
  let time_before_third_jam : ℝ := total_driving_time - time_before_first_jam - time_between_first_and_second_jam
  let third_jam_duration : ℝ := third_jam_multiplier * time_before_third_jam
  let total_pit_stop_time : ℝ := num_pit_stops * pit_stop_duration

  let total_duration : ℝ := total_driving_time + first_jam_duration + second_jam_duration + third_jam_duration + total_pit_stop_time

  total_duration = 18 := by sorry

end NUMINAMATH_CALUDE_tims_trip_duration_l2523_252372


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l2523_252344

def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def satisfies_condition (n : ℕ) : Prop :=
  is_perfect_square (sum_squares n * (sum_squares (3 * n) - sum_squares n))

theorem smallest_n_satisfying_condition :
  (∀ m : ℕ, 10 ≤ m ∧ m < 71 → ¬ satisfies_condition m) ∧
  satisfies_condition 71 := by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l2523_252344


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l2523_252396

theorem right_triangle_leg_length (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 1 →
  4 * (1/2 * triangle_leg * triangle_leg) = square_side * square_side →
  triangle_leg = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l2523_252396


namespace NUMINAMATH_CALUDE_covered_area_equals_transformed_square_l2523_252398

/-- A square in a 2D plane -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Transformation of a square by rotation and scaling -/
def transform_square (s : Square) (angle : ℝ) (scale : ℝ) : Square :=
  { center := s.center,
    side_length := s.side_length * scale }

/-- The set of all points covered by squares with one diagonal on the given square -/
def covered_area (s : Square) : Set (ℝ × ℝ) :=
  { p | ∃ (sq : Square), sq.center = s.center ∧ 
        (sq.side_length)^2 = 2 * (s.side_length)^2 ∧
        p ∈ { q | ∃ (x y : ℝ), 
              (x - sq.center.1)^2 + (y - sq.center.2)^2 ≤ (sq.side_length / 2)^2 } }

theorem covered_area_equals_transformed_square (s : Square) :
  covered_area s = { p | ∃ (x y : ℝ), 
                        (x - s.center.1)^2 + (y - s.center.2)^2 ≤ (s.side_length * Real.sqrt 2)^2 } := by
  sorry

end NUMINAMATH_CALUDE_covered_area_equals_transformed_square_l2523_252398


namespace NUMINAMATH_CALUDE_max_perpendicular_faces_theorem_l2523_252301

/-- The maximum number of lateral faces of an n-sided pyramid that can be perpendicular to the base -/
def max_perpendicular_faces (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

/-- Theorem stating the maximum number of lateral faces of an n-sided pyramid that can be perpendicular to the base -/
theorem max_perpendicular_faces_theorem (n : ℕ) (h : n > 0) :
  max_perpendicular_faces n = 
    if n % 2 = 0 
    then n / 2 
    else (n + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_perpendicular_faces_theorem_l2523_252301


namespace NUMINAMATH_CALUDE_deer_families_moved_out_l2523_252380

theorem deer_families_moved_out (total : ℕ) (stayed : ℕ) (moved_out : ℕ) : 
  total = 79 → stayed = 45 → moved_out = total - stayed → moved_out = 34 := by
  sorry

end NUMINAMATH_CALUDE_deer_families_moved_out_l2523_252380


namespace NUMINAMATH_CALUDE_average_goals_calculation_l2523_252358

theorem average_goals_calculation (layla_goals kristin_goals : ℕ) : 
  layla_goals = 104 →
  kristin_goals = layla_goals - 24 →
  (layla_goals + kristin_goals) / 2 = 92 := by
  sorry

end NUMINAMATH_CALUDE_average_goals_calculation_l2523_252358


namespace NUMINAMATH_CALUDE_negative_three_star_five_l2523_252346

-- Define the operation *
def star (a b : ℚ) : ℚ := (a - 2*b) / (2*a - b)

-- Theorem statement
theorem negative_three_star_five :
  star (-3) 5 = 13/11 := by sorry

end NUMINAMATH_CALUDE_negative_three_star_five_l2523_252346


namespace NUMINAMATH_CALUDE_reservoir_capacity_problem_l2523_252369

/-- Theorem about a reservoir's capacity and water levels -/
theorem reservoir_capacity_problem (current_amount : ℝ) 
  (h1 : current_amount = 6)
  (h2 : current_amount = 2 * normal_level)
  (h3 : current_amount = 0.6 * total_capacity) :
  total_capacity - normal_level = 7 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_problem_l2523_252369


namespace NUMINAMATH_CALUDE_digit_proportion_theorem_l2523_252357

theorem digit_proportion_theorem :
  ∀ n : ℕ,
  (n / 2 : ℚ) + (n / 5 : ℚ) + (n / 5 : ℚ) + (n / 10 : ℚ) = n →
  (n / 2 : ℕ) + (n / 5 : ℕ) + (n / 5 : ℕ) + (n / 10 : ℕ) = n →
  n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_digit_proportion_theorem_l2523_252357


namespace NUMINAMATH_CALUDE_square_of_negative_two_i_l2523_252367

theorem square_of_negative_two_i (i : ℂ) (hi : i^2 = -1) : (-2 * i)^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_i_l2523_252367


namespace NUMINAMATH_CALUDE_train_passing_time_l2523_252356

/-- Proves that a train of given length and speed takes approximately the calculated time to pass a pole -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (ε : ℝ) :
  train_length = 125 →
  train_speed_kmh = 60 →
  ε > 0 →
  ∃ (t : ℝ), t > 0 ∧ abs (t - 7.5) < ε ∧ t = train_length / (train_speed_kmh * 1000 / 3600) :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l2523_252356


namespace NUMINAMATH_CALUDE_larger_cuboid_length_l2523_252302

/-- Proves that the length of a larger cuboid is 18m, given the specified conditions --/
theorem larger_cuboid_length : 
  ∀ (small_length small_width small_height : ℝ)
    (large_width large_height : ℝ)
    (num_small_cuboids : ℕ),
  small_length = 5 →
  small_width = 6 →
  small_height = 3 →
  large_width = 15 →
  large_height = 2 →
  num_small_cuboids = 6 →
  ∃ (large_length : ℝ),
    large_length * large_width * large_height = 
    num_small_cuboids * (small_length * small_width * small_height) ∧
    large_length = 18 := by
sorry


end NUMINAMATH_CALUDE_larger_cuboid_length_l2523_252302


namespace NUMINAMATH_CALUDE_target_hit_probability_l2523_252300

theorem target_hit_probability (p1 p2 : ℝ) (h1 : p1 = 0.8) (h2 : p2 = 0.7) :
  1 - (1 - p1) * (1 - p2) = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l2523_252300


namespace NUMINAMATH_CALUDE_parallel_plane_through_point_l2523_252389

def plane_equation (x y z : ℝ) : ℝ := 3*x + 2*y - 4*z - 16

theorem parallel_plane_through_point :
  let given_plane (x y z : ℝ) := 3*x + 2*y - 4*z - 5
  (∀ (x y z : ℝ), plane_equation x y z = 0 ↔ given_plane x y z = k) ∧
  plane_equation 2 3 (-1) = 0 ∧
  (∃ (A B C D : ℤ), 
    (∀ (x y z : ℝ), plane_equation x y z = A*x + B*y + C*z + D) ∧
    A > 0 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) :=
by sorry

end NUMINAMATH_CALUDE_parallel_plane_through_point_l2523_252389


namespace NUMINAMATH_CALUDE_three_numbers_equation_l2523_252323

theorem three_numbers_equation (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : x^2 - y^2 = y*z) (eq2 : y^2 - z^2 = x*z) :
  x^2 - z^2 = x*y :=
by
  sorry

end NUMINAMATH_CALUDE_three_numbers_equation_l2523_252323


namespace NUMINAMATH_CALUDE_estimate_wheat_amount_l2523_252337

/-- Estimates the amount of wheat in a mixed batch of grain -/
theorem estimate_wheat_amount (total_mixed : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) : 
  total_mixed = 1524 →
  sample_size = 254 →
  wheat_in_sample = 28 →
  (total_mixed * wheat_in_sample) / sample_size = 168 :=
by sorry

end NUMINAMATH_CALUDE_estimate_wheat_amount_l2523_252337


namespace NUMINAMATH_CALUDE_triangle_area_l2523_252340

/-- Given a triangle ABC with the following properties:
  * sin(C/2) = √6/4
  * c = 2
  * sin B = 2 sin A
  Prove that the area of the triangle is √15/4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) 
  (h_sin_half_C : Real.sin (C / 2) = Real.sqrt 6 / 4)
  (h_c : c = 2)
  (h_sin_B : Real.sin B = 2 * Real.sin A) :
  (1 / 2) * a * b * Real.sin C = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2523_252340


namespace NUMINAMATH_CALUDE_distance_philadelphia_los_angeles_l2523_252310

/-- The distance between two points on a complex plane, where one point is at (1950, 1950) and the other is at (0, 0), is equal to 1950√2. -/
theorem distance_philadelphia_los_angeles : 
  let philadelphia : ℂ := 1950 + 1950 * Complex.I
  let los_angeles : ℂ := 0
  Complex.abs (philadelphia - los_angeles) = 1950 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_philadelphia_los_angeles_l2523_252310


namespace NUMINAMATH_CALUDE_cylinder_volume_l2523_252394

/-- The volume of a cylinder with height 300 cm and circular base area of 9 square cm is 2700 cubic centimeters. -/
theorem cylinder_volume (h : ℝ) (base_area : ℝ) (volume : ℝ) 
  (h_val : h = 300)
  (base_area_val : base_area = 9)
  (volume_def : volume = base_area * h) : 
  volume = 2700 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l2523_252394


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l2523_252319

theorem solution_implies_m_value (x m : ℝ) :
  x = 1 → 2 * x + m - 6 = 0 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l2523_252319


namespace NUMINAMATH_CALUDE_garden_perimeter_is_700_l2523_252352

/-- The perimeter of a rectangular garden with given length and breadth -/
def garden_perimeter (length : ℝ) (breadth : ℝ) : ℝ :=
  2 * (length + breadth)

theorem garden_perimeter_is_700 :
  garden_perimeter 250 100 = 700 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_is_700_l2523_252352


namespace NUMINAMATH_CALUDE_power_function_positive_l2523_252324

theorem power_function_positive (α : ℝ) (x : ℝ) (h : x > 0) : x^α > 0 := by
  sorry

end NUMINAMATH_CALUDE_power_function_positive_l2523_252324


namespace NUMINAMATH_CALUDE_range_of_k_l2523_252370

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := x^2 - x - 2 > 0

-- Define the property that p is sufficient but not necessary for q
def sufficient_not_necessary (k : ℝ) : Prop :=
  (∀ x, p x k → q x) ∧ (∃ x, q x ∧ ¬(p x k))

-- Theorem statement
theorem range_of_k :
  ∀ k, sufficient_not_necessary k ↔ k > 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l2523_252370


namespace NUMINAMATH_CALUDE_midpoint_property_l2523_252348

/-- Given two points A and B in a 2D plane, if C is their midpoint,
    then 3 times the x-coordinate of C minus 2 times the y-coordinate of C equals -3. -/
theorem midpoint_property (A B C : ℝ × ℝ) :
  A = (-6, 9) →
  B = (8, -3) →
  C.1 = (A.1 + B.1) / 2 →
  C.2 = (A.2 + B.2) / 2 →
  3 * C.1 - 2 * C.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_property_l2523_252348


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2523_252382

/-- 
Given a sum P put at simple interest for 7 years, if increasing the interest rate 
by 2% results in $140 more interest, then P = $1000.
-/
theorem simple_interest_problem (P : ℚ) (R : ℚ) : 
  (P * (R + 2) * 7 / 100 = P * R * 7 / 100 + 140) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2523_252382


namespace NUMINAMATH_CALUDE_max_marked_cells_theorem_marked_cells_property_l2523_252306

/-- Represents an equilateral triangle divided into n^2 cells -/
structure DividedTriangle where
  n : ℕ
  cells : ℕ := n^2

/-- Represents the maximum number of cells that can be marked -/
def max_marked_cells (t : DividedTriangle) : ℕ :=
  if t.n = 10 then 7
  else if t.n = 9 then 6
  else 0  -- undefined for other values of n

/-- Theorem stating the maximum number of marked cells for n = 10 and n = 9 -/
theorem max_marked_cells_theorem (t : DividedTriangle) :
  (t.n = 10 → max_marked_cells t = 7) ∧
  (t.n = 9 → max_marked_cells t = 6) := by
  sorry

/-- Represents a strip in the divided triangle -/
structure Strip where
  cells : Finset ℕ

/-- Function to check if two cells are in the same strip -/
def in_same_strip (c1 c2 : ℕ) (s : Strip) : Prop :=
  c1 ∈ s.cells ∧ c2 ∈ s.cells

/-- The main theorem to be proved -/
theorem marked_cells_property (t : DividedTriangle) (marked_cells : Finset ℕ) :
  (∀ (s : Strip), ∀ (c1 c2 : ℕ), c1 ∈ marked_cells → c2 ∈ marked_cells →
    in_same_strip c1 c2 s → c1 = c2) →
  (t.n = 10 → marked_cells.card ≤ 7) ∧
  (t.n = 9 → marked_cells.card ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_max_marked_cells_theorem_marked_cells_property_l2523_252306


namespace NUMINAMATH_CALUDE_probability_play_exactly_one_l2523_252320

def total_people : ℕ := 800
def play_at_least_one_ratio : ℚ := 1 / 5
def play_two_or_more : ℕ := 64

theorem probability_play_exactly_one (total_people : ℕ) (play_at_least_one_ratio : ℚ) (play_two_or_more : ℕ) :
  (play_at_least_one_ratio * total_people - play_two_or_more : ℚ) / total_people = 12 / 100 :=
by sorry

end NUMINAMATH_CALUDE_probability_play_exactly_one_l2523_252320


namespace NUMINAMATH_CALUDE_circle_equation_l2523_252359

theorem circle_equation (h : ℝ) :
  (∃ (x : ℝ), (x - 2)^2 + (-3)^2 = 5^2 ∧ h = x) →
  ((h - 6)^2 + y^2 = 25 ∨ (h + 2)^2 + y^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2523_252359


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l2523_252388

/-- Define the first element of the nth set -/
def first_element (n : ℕ) : ℕ := 1 + (n - 1) * n / 2

/-- Define the last element of the nth set -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- Define the sum of elements in the nth set -/
def sum_of_set (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem: The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : sum_of_set 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l2523_252388


namespace NUMINAMATH_CALUDE_particular_number_proof_l2523_252308

theorem particular_number_proof (x : ℚ) : x / 4 + 3 = 5 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_proof_l2523_252308


namespace NUMINAMATH_CALUDE_power_of_fraction_five_sevenths_fourth_l2523_252318

theorem power_of_fraction_five_sevenths_fourth : (5 / 7 : ℚ) ^ 4 = 625 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_five_sevenths_fourth_l2523_252318
