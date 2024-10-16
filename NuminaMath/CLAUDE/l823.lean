import Mathlib

namespace NUMINAMATH_CALUDE_jack_sugar_usage_l823_82378

/-- Represents the amount of sugar Jack has initially -/
def initial_sugar : ℕ := 65

/-- Represents the amount of sugar Jack buys after usage -/
def sugar_bought : ℕ := 50

/-- Represents the final amount of sugar Jack has -/
def final_sugar : ℕ := 97

/-- Represents the amount of sugar Jack uses -/
def sugar_used : ℕ := 18

theorem jack_sugar_usage :
  initial_sugar - sugar_used + sugar_bought = final_sugar :=
by sorry

end NUMINAMATH_CALUDE_jack_sugar_usage_l823_82378


namespace NUMINAMATH_CALUDE_onion_to_carrot_ratio_l823_82346

/-- Represents the number of vegetables Maria wants to cut -/
structure Vegetables where
  potatoes : ℕ
  carrots : ℕ
  onions : ℕ
  green_beans : ℕ

/-- The conditions of Maria's vegetable cutting plan -/
def cutting_plan (v : Vegetables) : Prop :=
  v.carrots = 6 * v.potatoes ∧
  v.onions = v.carrots ∧
  v.green_beans = v.onions / 3 ∧
  v.potatoes = 2 ∧
  v.green_beans = 8

theorem onion_to_carrot_ratio (v : Vegetables) 
  (h : cutting_plan v) : v.onions = v.carrots := by
  sorry

#check onion_to_carrot_ratio

end NUMINAMATH_CALUDE_onion_to_carrot_ratio_l823_82346


namespace NUMINAMATH_CALUDE_initial_worth_is_30_l823_82328

/-- Represents the value of a single gold coin -/
def coin_value : ℕ := 6

/-- Represents the number of coins Roman sold -/
def sold_coins : ℕ := 3

/-- Represents the number of coins Roman has left after the sale -/
def remaining_coins : ℕ := 2

/-- Represents the amount of money Roman has after the sale -/
def money_after_sale : ℕ := 12

/-- Theorem stating that the initial total worth of Roman's gold coins was $30 -/
theorem initial_worth_is_30 :
  (sold_coins + remaining_coins) * coin_value = 30 :=
sorry

end NUMINAMATH_CALUDE_initial_worth_is_30_l823_82328


namespace NUMINAMATH_CALUDE_tangent_line_curve_n_value_l823_82347

/-- Given a line and a curve that are tangent at a point, prove the value of n. -/
theorem tangent_line_curve_n_value :
  ∀ (k m n : ℝ),
  (∀ x, k * x + 1 = x^3 + m * x + n → x = 1 ∧ k * x + 1 = 3) →
  (∀ x, (3 * x^2 + m) * (x - 1) + (1^3 + m * 1 + n) = k * x + 1) →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_curve_n_value_l823_82347


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l823_82350

theorem line_tangent_to_parabola :
  ∃! d : ℝ, ∀ x y : ℝ,
    (y = 3 * x + d) ∧ (y^2 = 12 * x) →
    (∃! t : ℝ, y = 3 * t + d ∧ y^2 = 12 * t) →
    d = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l823_82350


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l823_82390

theorem least_number_with_remainder (n : ℕ) : n = 130 ↔ 
  (∀ m, m < n → ¬(m % 6 = 4 ∧ m % 7 = 4 ∧ m % 9 = 4 ∧ m % 18 = 4)) ∧
  n % 6 = 4 ∧ n % 7 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l823_82390


namespace NUMINAMATH_CALUDE_curve_and_line_properties_l823_82395

-- Define the unit circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the stretched curve C₂
def C₂ (x y : ℝ) : Prop := (x / Real.sqrt 3)^2 + (y / 2)^2 = 1

-- Define the line l
def l (x y : ℝ) : Prop := 2 * x - y - 6 = 0

-- Theorem statement
theorem curve_and_line_properties :
  -- 1. Parametric equations of C₂
  (∀ φ : ℝ, C₂ (Real.sqrt 3 * Real.cos φ) (2 * Real.sin φ)) ∧
  -- 2. Point P(-3/2, 1) on C₂ has maximum distance to l
  (C₂ (-3/2) 1 ∧
   ∀ x y : ℝ, C₂ x y →
     (x + 3/2)^2 + (y - 1)^2 ≤ (2 * Real.sqrt 5)^2) ∧
  -- 3. Maximum distance from C₂ to l is 2√5
  (∃ x y : ℝ, C₂ x y ∧
    |2*x - y - 6| / Real.sqrt 5 = 2 * Real.sqrt 5 ∧
    ∀ x' y' : ℝ, C₂ x' y' →
      |2*x' - y' - 6| / Real.sqrt 5 ≤ 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_curve_and_line_properties_l823_82395


namespace NUMINAMATH_CALUDE_lamps_with_burnt_bulbs_l823_82360

/-- Given a set of lamps with some burnt-out bulbs, proves the number of bulbs per lamp -/
theorem lamps_with_burnt_bulbs 
  (total_lamps : ℕ) 
  (burnt_fraction : ℚ) 
  (burnt_per_lamp : ℕ) 
  (working_bulbs : ℕ) : 
  total_lamps = 20 → 
  burnt_fraction = 1/4 → 
  burnt_per_lamp = 2 → 
  working_bulbs = 130 → 
  (total_lamps * (burnt_fraction * burnt_per_lamp + (1 - burnt_fraction) * working_bulbs / total_lamps)) / total_lamps = 7 := by
sorry

end NUMINAMATH_CALUDE_lamps_with_burnt_bulbs_l823_82360


namespace NUMINAMATH_CALUDE_example_is_quadratic_l823_82336

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - x + 1 = 0 is a quadratic equation in terms of x -/
theorem example_is_quadratic : is_quadratic_equation (λ x => x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_example_is_quadratic_l823_82336


namespace NUMINAMATH_CALUDE_bacteria_population_growth_l823_82314

/-- The final population of a bacteria culture after doubling for a given time -/
def finalPopulation (initialPopulation : ℕ) (doublingTime : ℕ) (totalTime : ℕ) : ℕ :=
  initialPopulation * 2^(totalTime / doublingTime)

/-- Theorem: The bacteria population doubles from 1000 to 1,024,000 in 20 minutes -/
theorem bacteria_population_growth :
  finalPopulation 1000 2 20 = 1024000 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_population_growth_l823_82314


namespace NUMINAMATH_CALUDE_fuel_spending_reduction_l823_82342

theorem fuel_spending_reduction 
  (old_efficiency : ℝ) 
  (old_fuel_cost : ℝ) 
  (efficiency_improvement : ℝ) 
  (fuel_cost_increase : ℝ) : 
  let new_efficiency : ℝ := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost : ℝ := old_fuel_cost * (1 + fuel_cost_increase)
  let old_trip_cost : ℝ := old_fuel_cost
  let new_trip_cost : ℝ := (1 / (1 + efficiency_improvement)) * new_fuel_cost
  let cost_reduction : ℝ := (old_trip_cost - new_trip_cost) / old_trip_cost
  efficiency_improvement = 0.75 ∧ 
  fuel_cost_increase = 0.30 → 
  cost_reduction = 25 / 28 := by
sorry

end NUMINAMATH_CALUDE_fuel_spending_reduction_l823_82342


namespace NUMINAMATH_CALUDE_min_value_theorem_l823_82353

def f (a x : ℝ) : ℝ := x^2 + (a+8)*x + a^2 + a - 12

theorem min_value_theorem (a : ℝ) (h1 : a < 0) 
  (h2 : f a (a^2 - 4) = f a (2*a - 8)) :
  (∀ n : ℕ+, (f a n - 4*a) / (n + 1) ≥ 37/4) ∧ 
  (∃ n : ℕ+, (f a n - 4*a) / (n + 1) = 37/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l823_82353


namespace NUMINAMATH_CALUDE_money_difference_specific_money_difference_l823_82341

/-- The difference between Dave's and Derek's remaining money after expenses -/
theorem money_difference (derek_initial : ℕ) (derek_lunch1 : ℕ) (derek_lunch_dad : ℕ) (derek_lunch2 : ℕ)
                         (dave_initial : ℕ) (dave_lunch_mom : ℕ) : ℕ :=
  let derek_spent := derek_lunch1 + derek_lunch_dad + derek_lunch2
  let derek_remaining := derek_initial - derek_spent
  let dave_remaining := dave_initial - dave_lunch_mom
  dave_remaining - derek_remaining

/-- Proof of the specific problem -/
theorem specific_money_difference :
  money_difference 40 14 11 5 50 7 = 33 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_specific_money_difference_l823_82341


namespace NUMINAMATH_CALUDE_investment_percentage_proof_l823_82394

/-- Proves that given the investment conditions, the percentage at which $3,500 was invested is 4% --/
theorem investment_percentage_proof (total_investment : ℝ) (investment1 : ℝ) (investment2 : ℝ) 
  (rate1 : ℝ) (rate3 : ℝ) (desired_income : ℝ) (x : ℝ) :
  total_investment = 10000 →
  investment1 = 4000 →
  investment2 = 3500 →
  rate1 = 0.05 →
  rate3 = 0.064 →
  desired_income = 500 →
  investment1 * rate1 + investment2 * (x / 100) + (total_investment - investment1 - investment2) * rate3 = desired_income →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_proof_l823_82394


namespace NUMINAMATH_CALUDE_angle_at_point_l823_82374

theorem angle_at_point (x : ℝ) : 
  (x + x + 160 = 360) → x = 100 := by sorry

end NUMINAMATH_CALUDE_angle_at_point_l823_82374


namespace NUMINAMATH_CALUDE_point_on_line_implies_fraction_value_l823_82372

/-- If (m,7) lies on the graph of y = 3x + 1, then m²/(m-1) = 4 -/
theorem point_on_line_implies_fraction_value (m : ℝ) : 
  7 = 3 * m + 1 → m^2 / (m - 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_implies_fraction_value_l823_82372


namespace NUMINAMATH_CALUDE_debby_bottles_per_day_l823_82368

/-- The number of bottles Debby bought -/
def total_bottles : ℕ := 8066

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := 74

/-- The number of bottles Debby drank per day -/
def bottles_per_day : ℕ := total_bottles / days_lasted

theorem debby_bottles_per_day :
  bottles_per_day = 109 := by sorry

end NUMINAMATH_CALUDE_debby_bottles_per_day_l823_82368


namespace NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l823_82365

/-- The number of ordered quadruplets (a, b, c, d) with given gcd and lcm -/
def count_quadruplets (gcd lcm : ℕ) : ℕ := sorry

/-- The theorem stating the smallest n satisfying the conditions -/
theorem smallest_n_for_quadruplets :
  ∃ n : ℕ, n > 0 ∧ 
  count_quadruplets 72 n = 72000 ∧
  (∀ m : ℕ, m > 0 → m < n → count_quadruplets 72 m ≠ 72000) ∧
  n = 36288 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l823_82365


namespace NUMINAMATH_CALUDE_intersection_when_a_2_b_subset_a_range_l823_82352

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Theorem 1: Intersection of A and B when a = 2
theorem intersection_when_a_2 :
  A 2 ∩ B 2 = {x | 4 < x ∧ x < 5} :=
sorry

-- Theorem 2: Range of a for which B is a subset of A
theorem b_subset_a_range :
  ∀ a : ℝ, B a ⊆ A a ↔ a = -1 ∨ (1 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_intersection_when_a_2_b_subset_a_range_l823_82352


namespace NUMINAMATH_CALUDE_appropriate_units_l823_82306

-- Define the mass units
inductive MassUnit
| Kilogram
| Gram
| Ton

-- Define a structure for an object with a weight and unit
structure WeightedObject where
  weight : ℝ
  unit : MassUnit

-- Define the objects
def basketOfEggs : WeightedObject := { weight := 5, unit := MassUnit.Kilogram }
def honeybee : WeightedObject := { weight := 5, unit := MassUnit.Gram }
def tank : WeightedObject := { weight := 6, unit := MassUnit.Ton }

-- Function to determine if a unit is appropriate for an object
def isAppropriateUnit (obj : WeightedObject) : Prop :=
  match obj with
  | { weight := w, unit := MassUnit.Kilogram } => w ≥ 1 ∧ w < 1000
  | { weight := w, unit := MassUnit.Gram } => w ≥ 0.1 ∧ w < 1000
  | { weight := w, unit := MassUnit.Ton } => w ≥ 1 ∧ w < 1000

-- Theorem stating that the given units are appropriate for each object
theorem appropriate_units :
  isAppropriateUnit basketOfEggs ∧
  isAppropriateUnit honeybee ∧
  isAppropriateUnit tank := by
  sorry

end NUMINAMATH_CALUDE_appropriate_units_l823_82306


namespace NUMINAMATH_CALUDE_spongebob_fries_sold_l823_82356

/-- Calculates the number of large fries sold given the total earnings, 
    number of burgers sold, price per burger, and price per large fries. -/
def large_fries_sold (total_earnings : ℚ) (num_burgers : ℕ) (price_burger : ℚ) (price_fries : ℚ) : ℚ :=
  (total_earnings - num_burgers * price_burger) / price_fries

/-- Proves that Spongebob sold 12 large fries given the conditions -/
theorem spongebob_fries_sold : 
  large_fries_sold 78 30 2 (3/2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_spongebob_fries_sold_l823_82356


namespace NUMINAMATH_CALUDE_gcd_8_factorial_12_factorial_l823_82317

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_8_factorial_12_factorial :
  Nat.gcd (factorial 8) (factorial 12) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_12_factorial_l823_82317


namespace NUMINAMATH_CALUDE_least_value_quadratic_l823_82391

theorem least_value_quadratic (y : ℝ) : 
  (2 * y^2 + 7 * y + 3 = 6) → 
  y ≥ (-7 - Real.sqrt 73) / 4 ∧ 
  ∃ (y_min : ℝ), 2 * y_min^2 + 7 * y_min + 3 = 6 ∧ y_min = (-7 - Real.sqrt 73) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l823_82391


namespace NUMINAMATH_CALUDE_helen_cookies_l823_82380

/-- The total number of chocolate chip cookies Helen baked -/
def total_cookies (yesterday today : ℕ) : ℕ := yesterday + today

/-- Theorem stating that Helen baked 1081 chocolate chip cookies in total -/
theorem helen_cookies : total_cookies 527 554 = 1081 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l823_82380


namespace NUMINAMATH_CALUDE_parallelogram_area_calculation_l823_82327

/-- The area of a parallelogram generated by two vectors -/
def parallelogram_area (a b : ℝ × ℝ) : ℝ := sorry

theorem parallelogram_area_calculation 
  (a b : ℝ × ℝ) 
  (h1 : parallelogram_area a b = 20)
  (u : ℝ × ℝ := (1/2 : ℝ) • a + (5/2 : ℝ) • b)
  (v : ℝ × ℝ := 3 • a - 2 • b) :
  parallelogram_area u v = 130 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_calculation_l823_82327


namespace NUMINAMATH_CALUDE_binomial_sum_l823_82324

theorem binomial_sum : (Nat.choose 10 3) + (Nat.choose 10 4) = 330 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l823_82324


namespace NUMINAMATH_CALUDE_james_pizza_slices_l823_82311

theorem james_pizza_slices :
  let total_slices : ℕ := 20
  let tom_slices : ℕ := 5
  let alice_slices : ℕ := 3
  let bob_slices : ℕ := 4
  let friends_slices : ℕ := tom_slices + alice_slices + bob_slices
  let remaining_slices : ℕ := total_slices - friends_slices
  let james_slices : ℕ := remaining_slices / 2
  james_slices = 4 := by sorry

end NUMINAMATH_CALUDE_james_pizza_slices_l823_82311


namespace NUMINAMATH_CALUDE_b_4_lt_b_7_l823_82326

def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1 + 1 / α 1
  | n + 1 => 1 + 1 / (α 1 + 1 / b n (fun k => α (k + 1)))

theorem b_4_lt_b_7 (α : ℕ → ℕ) (h : ∀ k, α k ≥ 1) : b 4 α < b 7 α := by
  sorry

end NUMINAMATH_CALUDE_b_4_lt_b_7_l823_82326


namespace NUMINAMATH_CALUDE_quadrilateral_weighted_centers_l823_82376

-- Define a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a function to calculate the ratio of distances
def distanceRatio (P Q R : Point) : ℝ :=
  sorry

-- Define the weighted center
def weightedCenter (P Q : Point) (m₁ m₂ : ℝ) : Point :=
  sorry

-- Main theorem
theorem quadrilateral_weighted_centers 
  (quad : Quadrilateral) (P Q R S : Point) :
  (∃ (m₁ m₂ m₃ m₄ : ℝ), 
    P = weightedCenter quad.A quad.B m₁ m₂ ∧
    Q = weightedCenter quad.B quad.C m₂ m₃ ∧
    R = weightedCenter quad.C quad.D m₃ m₄ ∧
    S = weightedCenter quad.D quad.A m₄ m₁) ↔
  distanceRatio quad.A P quad.B *
  distanceRatio quad.B Q quad.C *
  distanceRatio quad.C R quad.D *
  distanceRatio quad.D S quad.A = 1 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_weighted_centers_l823_82376


namespace NUMINAMATH_CALUDE_customers_left_l823_82354

/-- Given a waiter with an initial number of customers and a number of remaining customers,
    prove that the number of customers who left is the difference between the initial and remaining customers. -/
theorem customers_left (initial remaining : ℕ) (h1 : initial = 21) (h2 : remaining = 12) :
  initial - remaining = 9 := by
  sorry

end NUMINAMATH_CALUDE_customers_left_l823_82354


namespace NUMINAMATH_CALUDE_friendly_sequences_exist_l823_82359

/-- Definition of a friendly pair of sequences -/
def is_friendly_pair (a b : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0 ∧ b n > 0) ∧
  (∀ k : ℕ, ∃! (i j : ℕ), a i * b j = k)

/-- Theorem stating the existence of friendly sequences -/
theorem friendly_sequences_exist : ∃ (a b : ℕ → ℕ), is_friendly_pair a b :=
sorry

end NUMINAMATH_CALUDE_friendly_sequences_exist_l823_82359


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l823_82334

/-- Represents a parallelogram EFGH with given side lengths and diagonal -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  EH : ℝ

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := 2 * (p.EF + p.FG)

/-- Theorem: The perimeter of parallelogram EFGH is 140 units -/
theorem parallelogram_perimeter (p : Parallelogram) 
  (h1 : p.EF = 40)
  (h2 : p.FG = 30)
  (h3 : p.EH = 50) : 
  perimeter p = 140 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l823_82334


namespace NUMINAMATH_CALUDE_store_exit_ways_l823_82375

/-- The number of different oreo flavors --/
def oreo_flavors : ℕ := 8

/-- The number of different milk types --/
def milk_types : ℕ := 4

/-- The total number of items Charlie can choose from --/
def charlie_choices : ℕ := oreo_flavors + milk_types

/-- The total number of products they leave with --/
def total_products : ℕ := 5

/-- Function to calculate the number of ways Delta can choose n oreos --/
def delta_choices (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then oreo_flavors
  else if n = 2 then (Nat.choose oreo_flavors 2) + oreo_flavors
  else if n = 3 then (Nat.choose oreo_flavors 3) + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
  else if n = 4 then (Nat.choose oreo_flavors 4) + (Nat.choose oreo_flavors 2) * (Nat.choose (oreo_flavors - 2) 2) / 2 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
  else (Nat.choose oreo_flavors 5) + (Nat.choose oreo_flavors 2) * (Nat.choose (oreo_flavors - 2) 3) + oreo_flavors * (Nat.choose (oreo_flavors - 1) 2) + oreo_flavors

/-- The total number of ways Charlie and Delta could have left the store --/
def total_ways : ℕ :=
  (Nat.choose charlie_choices total_products) +
  (Nat.choose charlie_choices 4) * (delta_choices 1) +
  (Nat.choose charlie_choices 3) * (delta_choices 2) +
  (Nat.choose charlie_choices 2) * (delta_choices 3) +
  (Nat.choose charlie_choices 1) * (delta_choices 4) +
  (delta_choices 5)

theorem store_exit_ways : total_ways = 25512 := by
  sorry

end NUMINAMATH_CALUDE_store_exit_ways_l823_82375


namespace NUMINAMATH_CALUDE_gcd_of_squares_l823_82388

theorem gcd_of_squares : Nat.gcd (101^2 + 203^2 + 307^2) (100^2 + 202^2 + 308^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l823_82388


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l823_82337

/-- Given a geometric sequence with common ratio 2 and sum of first 4 terms equal to 1,
    prove that the sum of the first 8 terms is 17. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  (a 0 + a 1 + a 2 + a 3 = 1) →  -- sum of first 4 terms is 1
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 17) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l823_82337


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_l823_82345

theorem cube_volume_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 3) :
  let s := d / Real.sqrt 3
  s ^ 3 = 512 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_l823_82345


namespace NUMINAMATH_CALUDE_bug_pentagon_probability_l823_82358

/-- Probability of the bug being at the starting vertex after n moves -/
def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 0
  | 2 => 1/2
  | n+1 => 1/2 * (1 - Q n)

/-- The probability of returning to the starting vertex on the 12th move in a regular pentagon -/
theorem bug_pentagon_probability : Q 12 = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_bug_pentagon_probability_l823_82358


namespace NUMINAMATH_CALUDE_joan_oranges_l823_82302

theorem joan_oranges (total_oranges sara_oranges : ℕ) 
  (h1 : total_oranges = 47) 
  (h2 : sara_oranges = 10) : 
  total_oranges - sara_oranges = 37 := by
  sorry

end NUMINAMATH_CALUDE_joan_oranges_l823_82302


namespace NUMINAMATH_CALUDE_max_x_minus_y_l823_82325

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ z :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l823_82325


namespace NUMINAMATH_CALUDE_tenth_minus_ninth_square_tiles_l823_82384

-- Define the sequence of square side lengths
def squareSideLength (n : ℕ) : ℕ := n

-- Define the number of tiles in the nth square
def tilesInSquare (n : ℕ) : ℕ := (squareSideLength n) ^ 2

-- Theorem statement
theorem tenth_minus_ninth_square_tiles : 
  tilesInSquare 10 - tilesInSquare 9 = 19 := by sorry

end NUMINAMATH_CALUDE_tenth_minus_ninth_square_tiles_l823_82384


namespace NUMINAMATH_CALUDE_complex_argument_range_l823_82363

theorem complex_argument_range (z : ℂ) (h : Complex.abs (2 * z + z⁻¹) = 1) :
  ∃ k : ℤ, k ∈ ({0, 1} : Set ℤ) ∧
  k * π + π / 2 - Real.arccos (3 / 4) / 2 ≤ Complex.arg z ∧
  Complex.arg z ≤ k * π + π / 2 + Real.arccos (3 / 4) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_argument_range_l823_82363


namespace NUMINAMATH_CALUDE_fisherman_catch_l823_82308

theorem fisherman_catch (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) : 
  bass = 32 → 
  trout = bass / 4 → 
  blue_gill = 2 * bass → 
  bass + trout + blue_gill = 104 := by
  sorry

end NUMINAMATH_CALUDE_fisherman_catch_l823_82308


namespace NUMINAMATH_CALUDE_complex_power_sum_l823_82398

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^100 + 1/(z^100) = -2 * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l823_82398


namespace NUMINAMATH_CALUDE_smallest_square_partition_l823_82339

/-- A partition of a square into smaller squares -/
structure SquarePartition (n : ℕ) :=
  (num_40 : ℕ)  -- number of 40x40 squares
  (num_49 : ℕ)  -- number of 49x49 squares
  (valid : num_40 * 40 * 40 + num_49 * 49 * 49 = n * n)
  (both_present : num_40 > 0 ∧ num_49 > 0)

/-- The main theorem stating that 2000 is the smallest n that satisfies the conditions -/
theorem smallest_square_partition :
  (∃ (p : SquarePartition 2000), True) ∧
  (∀ n < 2000, ¬ ∃ (p : SquarePartition n), True) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_partition_l823_82339


namespace NUMINAMATH_CALUDE_rectangle_rotation_volume_l823_82389

/-- The volume of a cylinder formed by rotating a rectangle around its length -/
theorem rectangle_rotation_volume (length width : ℝ) (h_length : length = 6) (h_width : width = 3) :
  π * width^2 * length = 54 * π := by sorry

end NUMINAMATH_CALUDE_rectangle_rotation_volume_l823_82389


namespace NUMINAMATH_CALUDE_parcel_weight_theorem_l823_82357

theorem parcel_weight_theorem (x y z : ℕ) 
  (h1 : x + y = 132)
  (h2 : y + z = 135)
  (h3 : z + x = 140) :
  x + y + z = 204 := by
sorry

end NUMINAMATH_CALUDE_parcel_weight_theorem_l823_82357


namespace NUMINAMATH_CALUDE_inequality_proof_l823_82364

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l823_82364


namespace NUMINAMATH_CALUDE_geometric_series_sum_l823_82367

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let series_sum := (a * (1 - r^n)) / (1 - r)
  let a := 1 / 5
  let r := -1 / 5
  let n := 5
  series_sum = 521 / 3125 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l823_82367


namespace NUMINAMATH_CALUDE_total_students_shaking_hands_l823_82399

/-- The number of students from each school who participated in the debate --/
structure SchoolParticipation where
  school1 : ℕ
  school2 : ℕ
  school3 : ℕ

/-- The conditions of the debate participation --/
def debateConditions (p : SchoolParticipation) : Prop :=
  p.school1 = 2 * p.school2 ∧
  p.school2 = p.school3 + 40 ∧
  p.school3 = 200

/-- The theorem stating the total number of students who shook the mayor's hand --/
theorem total_students_shaking_hands (p : SchoolParticipation) 
  (h : debateConditions p) : p.school1 + p.school2 + p.school3 = 920 := by
  sorry

end NUMINAMATH_CALUDE_total_students_shaking_hands_l823_82399


namespace NUMINAMATH_CALUDE_f_properties_l823_82386

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + a

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * a * x + 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f_derivative a x + (2 * a - 1) * x

theorem f_properties (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) (f_derivative a x) x) ∧
  (∀ x > 0, HasDerivAt (g a) ((1 - x) / x) x) ∧
  (∃ x₀ > 0, IsLocalMax (g a) x₀ ∧ g a x₀ = 0) ∧
  (∀ x₀ > 0, ¬ IsLocalMin (g a) x₀) ∧
  (∀ x > 1, f a x < 0) ↔ a ≥ (1 / 2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l823_82386


namespace NUMINAMATH_CALUDE_power_of_two_sum_l823_82381

theorem power_of_two_sum : 2^3 * 2^4 + 2^5 = 160 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l823_82381


namespace NUMINAMATH_CALUDE_compare_powers_l823_82330

theorem compare_powers : 5^333 < 3^555 ∧ 3^555 < 4^444 := by
  sorry

end NUMINAMATH_CALUDE_compare_powers_l823_82330


namespace NUMINAMATH_CALUDE_new_room_ratio_l823_82312

/-- The ratio of a new room's size to the combined size of a bedroom and bathroom -/
theorem new_room_ratio (bedroom_size bathroom_size new_room_size : ℝ) 
  (h1 : bedroom_size = 309)
  (h2 : bathroom_size = 150)
  (h3 : new_room_size = 918) :
  new_room_size / (bedroom_size + bathroom_size) = 2 := by
  sorry

end NUMINAMATH_CALUDE_new_room_ratio_l823_82312


namespace NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l823_82396

-- Define the set M
def M : Set ℝ := {x : ℝ | |2*x - 1| < 1}

-- Theorem 1: Characterization of M
theorem characterization_of_M : M = {x : ℝ | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_for_M_elements (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  a * b + 1 > a + b := by sorry

end NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l823_82396


namespace NUMINAMATH_CALUDE_sum_digits_ratio_bound_l823_82301

/-- Sum of digits function -/
def S (n : ℕ+) : ℕ := sorry

/-- The theorem stating the upper bound and its achievability -/
theorem sum_digits_ratio_bound :
  (∀ n : ℕ+, (S n : ℚ) / (S (16 * n) : ℚ) ≤ 13) ∧
  (∃ n : ℕ+, (S n : ℚ) / (S (16 * n) : ℚ) = 13) :=
sorry

end NUMINAMATH_CALUDE_sum_digits_ratio_bound_l823_82301


namespace NUMINAMATH_CALUDE_tan_sum_specific_l823_82300

theorem tan_sum_specific (a b : Real) 
  (ha : Real.tan a = 1/2) (hb : Real.tan b = 1/3) : 
  Real.tan (a + b) = 1 := by sorry

end NUMINAMATH_CALUDE_tan_sum_specific_l823_82300


namespace NUMINAMATH_CALUDE_special_hexagon_area_l823_82310

/-- A hexagon with specific side lengths that can be divided into a rectangle and two triangles -/
structure SpecialHexagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  rectangle_width : ℝ
  rectangle_height : ℝ
  triangle_base : ℝ
  triangle_height : ℝ
  side1_eq : side1 = 20
  side2_eq : side2 = 15
  side3_eq : side3 = 22
  side4_eq : side4 = 27
  side5_eq : side5 = 18
  side6_eq : side6 = 15
  rectangle_width_eq : rectangle_width = 18
  rectangle_height_eq : rectangle_height = 22
  triangle_base_eq : triangle_base = 18
  triangle_height_eq : triangle_height = 15

/-- The area of the special hexagon is 666 square units -/
theorem special_hexagon_area (h : SpecialHexagon) : 
  h.rectangle_width * h.rectangle_height + 2 * (1/2 * h.triangle_base * h.triangle_height) = 666 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_area_l823_82310


namespace NUMINAMATH_CALUDE_alex_phone_bill_l823_82323

/-- Calculates the total cost of a cell phone plan based on usage --/
def calculate_phone_bill (base_cost : ℚ) (included_texts : ℕ) (text_cost : ℚ) 
                         (included_hours : ℕ) (minute_cost : ℚ)
                         (texts_sent : ℕ) (hours_talked : ℕ) : ℚ :=
  let extra_texts := max (texts_sent - included_texts) 0
  let extra_minutes := max ((hours_talked - included_hours) * 60) 0
  base_cost + (extra_texts : ℚ) * text_cost + (extra_minutes : ℚ) * minute_cost

theorem alex_phone_bill :
  calculate_phone_bill 25 20 0.1 20 (15 / 100) 150 25 = 83 := by
  sorry

end NUMINAMATH_CALUDE_alex_phone_bill_l823_82323


namespace NUMINAMATH_CALUDE_regular_polygon_2022_probability_l823_82348

/-- A regular polygon with 2022 sides -/
structure RegularPolygon2022 where
  area : ℝ
  sides : Nat
  is_regular : sides = 2022

/-- A point on the perimeter of a polygon -/
structure PerimeterPoint (P : RegularPolygon2022) where
  x : ℝ
  y : ℝ
  on_perimeter : True  -- This is a placeholder for the actual condition

/-- The distance between two points -/
def distance (A B : PerimeterPoint P) : ℝ := sorry

/-- The probability of an event -/
def probability (event : Prop) : ℝ := sorry

theorem regular_polygon_2022_probability 
  (P : RegularPolygon2022) 
  (h : P.area = 1) :
  probability (
    ∀ (A B : PerimeterPoint P), 
    distance A B ≥ Real.sqrt (2 / Real.pi)
  ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_2022_probability_l823_82348


namespace NUMINAMATH_CALUDE_remainder_theorem_l823_82307

theorem remainder_theorem (n m p : ℤ) 
  (hn : n % 18 = 10)
  (hm : m % 27 = 16)
  (hp : p % 6 = 4) :
  (2*n + 3*m - p) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l823_82307


namespace NUMINAMATH_CALUDE_tuesday_calls_l823_82338

/-- Represents the number of calls answered by Jean for each day of the work week -/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of calls answered in a week -/
def totalCalls (w : WeekCalls) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- Calculates the average number of calls per day -/
def averageCalls (w : WeekCalls) : ℚ :=
  totalCalls w / 5

theorem tuesday_calls (w : WeekCalls) 
  (h1 : w.monday = 35)
  (h2 : w.wednesday = 27)
  (h3 : w.thursday = 61)
  (h4 : w.friday = 31)
  (h5 : averageCalls w = 40) :
  w.tuesday = 46 := by
  sorry

#check tuesday_calls

end NUMINAMATH_CALUDE_tuesday_calls_l823_82338


namespace NUMINAMATH_CALUDE_pascal_triangle_45th_number_l823_82332

theorem pascal_triangle_45th_number (n : ℕ) : n = 46 →
  Nat.choose n 2 = 1035 :=
by sorry

end NUMINAMATH_CALUDE_pascal_triangle_45th_number_l823_82332


namespace NUMINAMATH_CALUDE_waiter_tables_l823_82309

theorem waiter_tables (total_customers : ℕ) (people_per_table : ℕ) (h1 : total_customers = 90) (h2 : people_per_table = 10) :
  total_customers / people_per_table = 9 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l823_82309


namespace NUMINAMATH_CALUDE_turtle_ratio_l823_82319

theorem turtle_ratio : 
  ∀ (trey kris kristen : ℕ),
  kristen = 12 →
  kris = kristen / 4 →
  trey = kristen + 9 →
  trey / kris = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_turtle_ratio_l823_82319


namespace NUMINAMATH_CALUDE_average_time_to_find_waldo_l823_82343

theorem average_time_to_find_waldo (num_books : ℕ) (puzzles_per_book : ℕ) (total_time : ℕ) :
  num_books = 15 →
  puzzles_per_book = 30 →
  total_time = 1350 →
  (total_time : ℚ) / (num_books * puzzles_per_book : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_time_to_find_waldo_l823_82343


namespace NUMINAMATH_CALUDE_haley_recycling_cans_l823_82349

theorem haley_recycling_cans (collected : ℕ) (in_bag : ℕ) 
  (h1 : collected = 9) (h2 : in_bag = 7) : 
  collected - in_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_haley_recycling_cans_l823_82349


namespace NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l823_82387

theorem chocolate_bars_in_large_box :
  let small_boxes : ℕ := 17
  let bars_per_small_box : ℕ := 26
  let total_bars : ℕ := small_boxes * bars_per_small_box
  total_bars = 442 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l823_82387


namespace NUMINAMATH_CALUDE_inverse_variation_example_l823_82318

-- Define the inverse variation relationship
def inverse_variation (p q : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ p * q = k

-- State the theorem
theorem inverse_variation_example :
  ∀ p q : ℝ,
  inverse_variation p q →
  (p = 1500 → q = 0.25) →
  (p = 3000 → q = 0.125) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_example_l823_82318


namespace NUMINAMATH_CALUDE_T_properties_l823_82370

-- Define the operation T
def T (m n x y : ℚ) : ℚ := (m*x + n*y) * (x + 2*y)

-- State the theorem
theorem T_properties (m n : ℚ) (hm : m ≠ 0) (hn : n ≠ 0) :
  T m n 1 (-1) = 0 ∧ T m n 0 2 = 8 →
  (m = 1 ∧ n = 1) ∧
  (∀ x y : ℚ, x^2 ≠ y^2 → T m n x y = T m n y x → m = 2*n) :=
by sorry

end NUMINAMATH_CALUDE_T_properties_l823_82370


namespace NUMINAMATH_CALUDE_fourth_child_receives_24_l823_82351

/-- Represents the distribution of sweets among a mother and her children -/
structure SweetDistribution where
  total : ℕ
  mother_fraction : ℚ
  num_children : ℕ
  eldest_youngest_ratio : ℕ
  second_third_diff : ℕ
  third_fourth_diff : ℕ
  youngest_second_ratio : ℚ

/-- Calculates the number of sweets the fourth child receives -/
def fourth_child_sweets (d : SweetDistribution) : ℕ :=
  sorry

/-- Theorem stating that given the problem conditions, the fourth child receives 24 sweets -/
theorem fourth_child_receives_24 (d : SweetDistribution) 
  (h1 : d.total = 120)
  (h2 : d.mother_fraction = 1/4)
  (h3 : d.num_children = 5)
  (h4 : d.eldest_youngest_ratio = 2)
  (h5 : d.second_third_diff = 6)
  (h6 : d.third_fourth_diff = 8)
  (h7 : d.youngest_second_ratio = 4/5) : 
  fourth_child_sweets d = 24 :=
sorry

end NUMINAMATH_CALUDE_fourth_child_receives_24_l823_82351


namespace NUMINAMATH_CALUDE_log_sum_equation_l823_82320

theorem log_sum_equation (x y z : ℝ) (hx : x = 625) (hy : y = 5) (hz : z = 1/25) :
  Real.log x / Real.log 5 + Real.log y / Real.log 5 - Real.log z / Real.log 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equation_l823_82320


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l823_82322

/-- The surface area of a rectangular parallelepiped with given dimensions -/
theorem cuboid_surface_area (w : ℝ) (h l : ℝ) : 
  w = 4 →
  l = w + 6 →
  h = l + 5 →
  2 * l * w + 2 * l * h + 2 * w * h = 500 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l823_82322


namespace NUMINAMATH_CALUDE_carl_weight_l823_82382

theorem carl_weight (billy brad carl : ℕ) 
  (h1 : billy = brad + 9)
  (h2 : brad = carl + 5)
  (h3 : billy = 159) : 
  carl = 145 := by sorry

end NUMINAMATH_CALUDE_carl_weight_l823_82382


namespace NUMINAMATH_CALUDE_both_runners_in_photo_probability_l823_82304

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℕ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the photography setup -/
structure Photo where
  trackFraction : ℚ
  timeRange : Set ℕ

/-- Calculates the probability of both runners being in the photo -/
def probabilityBothInPhoto (r1 r2 : Runner) (p : Photo) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem both_runners_in_photo_probability
  (rachel : Runner)
  (robert : Runner)
  (photo : Photo)
  (h1 : rachel.name = "Rachel" ∧ rachel.lapTime = 75 ∧ rachel.direction = true)
  (h2 : robert.name = "Robert" ∧ robert.lapTime = 95 ∧ robert.direction = false)
  (h3 : photo.trackFraction = 1/5)
  (h4 : photo.timeRange = {t | 900 ≤ t ∧ t < 960}) :
  probabilityBothInPhoto rachel robert photo = 1/4 :=
sorry

end NUMINAMATH_CALUDE_both_runners_in_photo_probability_l823_82304


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l823_82373

theorem arithmetic_mean_problem (a b c d : ℝ) :
  (a + b + c + d + 130) / 5 = 90 →
  (a + b + c + d) / 4 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l823_82373


namespace NUMINAMATH_CALUDE_pyramid_lego_count_l823_82333

/-- Calculates the number of legos for a square level -/
def square_level (side : ℕ) : ℕ := side * side

/-- Calculates the number of legos for a rectangular level -/
def rectangular_level (length width : ℕ) : ℕ := length * width

/-- Calculates the number of legos for a triangular level -/
def triangular_level (side : ℕ) : ℕ := side * (side + 1) / 2 - 3

/-- Calculates the total number of legos for the pyramid -/
def total_legos : ℕ :=
  square_level 10 + rectangular_level 8 6 + triangular_level 4 + 1

theorem pyramid_lego_count : total_legos = 156 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_lego_count_l823_82333


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2010_l823_82369

theorem smallest_n_divisible_by_2010 (a : ℕ → ℤ) 
  (h1 : ∃ k, a 1 = 2 * k + 1)
  (h2 : ∀ n : ℕ, n > 0 → n * (a (n + 1) - a n + 3) = a (n + 1) + a n + 3)
  (h3 : ∃ k, a 2009 = 2010 * k) :
  ∃ n : ℕ, n ≥ 2 ∧ (∃ k, a n = 2010 * k) ∧ (∀ m, 2 ≤ m ∧ m < n → ¬∃ k, a m = 2010 * k) ∧ n = 671 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2010_l823_82369


namespace NUMINAMATH_CALUDE_playground_girls_l823_82355

theorem playground_girls (total_children : ℕ) (boys : ℕ) (girls : ℕ) :
  total_children = 63 → boys = 35 → girls = total_children - boys → girls = 28 := by
  sorry

end NUMINAMATH_CALUDE_playground_girls_l823_82355


namespace NUMINAMATH_CALUDE_customer_satisfaction_probability_l823_82316

-- Define the probability of a customer being satisfied
def p : ℝ := sorry

-- Define the conditions
def dissatisfied_review_rate : ℝ := 0.80
def satisfied_review_rate : ℝ := 0.15
def angry_reviews : ℕ := 60
def positive_reviews : ℕ := 20

-- Theorem statement
theorem customer_satisfaction_probability :
  dissatisfied_review_rate * (1 - p) * (angry_reviews + positive_reviews) = angry_reviews ∧
  satisfied_review_rate * p * (angry_reviews + positive_reviews) = positive_reviews →
  p = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_customer_satisfaction_probability_l823_82316


namespace NUMINAMATH_CALUDE_fraction_evaluation_l823_82393

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l823_82393


namespace NUMINAMATH_CALUDE_can_form_triangle_l823_82344

/-- Triangle inequality theorem for a set of three line segments -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Proof that (5, 13, 12) can form a triangle -/
theorem can_form_triangle : satisfies_triangle_inequality 5 13 12 := by
  sorry

end NUMINAMATH_CALUDE_can_form_triangle_l823_82344


namespace NUMINAMATH_CALUDE_percentage_within_one_std_dev_l823_82379

-- Define a symmetric distribution
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percentage_less_than_mean_plus_std : ℝ

-- Theorem statement
theorem percentage_within_one_std_dev 
  (d : SymmetricDistribution) 
  (h1 : d.is_symmetric = true) 
  (h2 : d.percentage_less_than_mean_plus_std = 84) : 
  (d.percentage_less_than_mean_plus_std - (100 - d.percentage_less_than_mean_plus_std)) = 68 := by
  sorry

end NUMINAMATH_CALUDE_percentage_within_one_std_dev_l823_82379


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_greater_than_five_l823_82329

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 5) / Real.log (Real.sin 1)

theorem decreasing_f_implies_a_greater_than_five (a : ℝ) :
  (∀ x y, a < x ∧ x < y → f y < f x) →
  a > 5 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_greater_than_five_l823_82329


namespace NUMINAMATH_CALUDE_vector_addition_l823_82362

theorem vector_addition : 
  let v1 : Fin 3 → ℝ := ![4, -9, 2]
  let v2 : Fin 3 → ℝ := ![-3, 8, -5]
  v1 + v2 = ![1, -1, -3] := by
sorry

end NUMINAMATH_CALUDE_vector_addition_l823_82362


namespace NUMINAMATH_CALUDE_lily_pad_coverage_time_l823_82383

def days_to_half_coverage : ℕ := 57

theorem lily_pad_coverage_time :
  ∀ (total_coverage : ℝ) (daily_growth_factor : ℝ),
    total_coverage > 0 →
    daily_growth_factor = 2 →
    (daily_growth_factor ^ days_to_half_coverage : ℝ) * (1 / 2) = 1 →
    (daily_growth_factor ^ (days_to_half_coverage + 1) : ℝ) = total_coverage :=
by sorry

end NUMINAMATH_CALUDE_lily_pad_coverage_time_l823_82383


namespace NUMINAMATH_CALUDE_add_7455_seconds_to_8_15_00_l823_82303

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The starting time: 8:15:00 -/
def startTime : Time :=
  { hours := 8, minutes := 15, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 7455

/-- The expected final time: 10:19:15 -/
def expectedFinalTime : Time :=
  { hours := 10, minutes := 19, seconds := 15 }

theorem add_7455_seconds_to_8_15_00 :
  addSeconds startTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_7455_seconds_to_8_15_00_l823_82303


namespace NUMINAMATH_CALUDE_unique_three_digit_sum_l823_82315

/-- A three-digit integer with all digits different -/
def ThreeDigitDistinct : Type := { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) }

/-- The theorem stating that there exists a unique three-digit integer with all digits different that, when summed 9 times, equals 2331 -/
theorem unique_three_digit_sum :
  ∃! (n : ThreeDigitDistinct), 9 * n.val = 2331 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_sum_l823_82315


namespace NUMINAMATH_CALUDE_line_relations_l823_82371

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The first line l₁ -/
def l₁ (m : ℝ) : Line := { a := 1, b := m, c := 6 }

/-- The second line l₂ -/
def l₂ (m : ℝ) : Line := { a := m - 2, b := 3 * m, c := 18 }

/-- Two lines are parallel -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a ∧ l₁.a * l₂.c ≠ l₁.c * l₂.a

/-- Two lines are perpendicular -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

/-- Main theorem -/
theorem line_relations (m : ℝ) :
  (parallel (l₁ m) (l₂ m) ↔ m = 0) ∧
  (perpendicular (l₁ m) (l₂ m) ↔ m = -1 ∨ m = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_line_relations_l823_82371


namespace NUMINAMATH_CALUDE_cupcakes_frosted_l823_82397

-- Define the frosting rates and working time
def cagney_rate : ℚ := 1 / 25
def lacey_rate : ℚ := 1 / 35
def pat_rate : ℚ := 1 / 45
def working_time : ℕ := 10 * 60  -- 10 minutes in seconds

-- Theorem statement
theorem cupcakes_frosted : 
  ∃ (n : ℕ), n = 54 ∧ 
  (n : ℚ) ≤ (cagney_rate + lacey_rate + pat_rate) * working_time ∧
  (n + 1 : ℚ) > (cagney_rate + lacey_rate + pat_rate) * working_time :=
by sorry

end NUMINAMATH_CALUDE_cupcakes_frosted_l823_82397


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l823_82321

theorem simplify_and_evaluate (a : ℚ) (h : a = -1/3) :
  (a + 1) * (a - 1) - a * (a + 3) = 0 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l823_82321


namespace NUMINAMATH_CALUDE_haley_trees_l823_82392

theorem haley_trees (initial_trees : ℕ) (dead_trees : ℕ) (final_trees : ℕ) 
  (h1 : initial_trees = 9)
  (h2 : dead_trees = 4)
  (h3 : final_trees = 10) :
  final_trees - (initial_trees - dead_trees) = 5 := by
sorry

end NUMINAMATH_CALUDE_haley_trees_l823_82392


namespace NUMINAMATH_CALUDE_complex_modulus_identity_l823_82377

theorem complex_modulus_identity 
  (z₁ z₂ z₃ z₄ : ℂ) 
  (h₁ : Complex.abs z₁ = 1) 
  (h₂ : Complex.abs z₂ = 1) 
  (h₃ : Complex.abs z₃ = 1) 
  (h₄ : Complex.abs z₄ = 1) : 
  Complex.abs (z₁ - z₂) ^ 2 * Complex.abs (z₃ - z₄) ^ 2 + 
  Complex.abs (z₁ + z₄) ^ 2 * Complex.abs (z₃ - z₂) ^ 2 = 
  Complex.abs (z₁ * (z₂ - z₃) + z₃ * (z₂ - z₁) + z₄ * (z₁ - z₃)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_identity_l823_82377


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l823_82331

theorem ceiling_negative_three_point_seven : ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l823_82331


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l823_82366

/-- A geometric sequence with specific properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) ∧
  |a 1| = 1 ∧
  a 5 = -8 * a 2 ∧
  a 5 > a 2

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := (-2) ^ (n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  geometric_sequence a → (∀ n : ℕ, a n = general_term n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l823_82366


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_l823_82385

theorem triangle_arithmetic_sequence (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  -- a, b, c are sides opposite to angles A, B, C respectively
  (a = 2 * Real.sin A) ∧ (b = 2 * Real.sin B) ∧ (c = 2 * Real.sin C) →
  -- a*cos(C), b*cos(B), c*cos(A) form an arithmetic sequence
  (a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B) →
  -- Conclusions
  (B = π / 3) ∧
  (∀ x, x ∈ Set.Icc (-1/2) (1 + Real.sqrt 3) ↔ 
    ∃ A C, (0 < A) ∧ (A < 2*π/3) ∧ (C = 2*π/3 - A) ∧
    (x = 2 * Real.sin A * Real.sin A + Real.cos (A - C))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_l823_82385


namespace NUMINAMATH_CALUDE_prob_less_than_8_l823_82361

/-- The probability of an archer scoring less than 8 in a single shot -/
theorem prob_less_than_8 (p_10 p_9 p_8 : ℝ) 
  (h1 : p_10 = 0.24)
  (h2 : p_9 = 0.28)
  (h3 : p_8 = 0.19) : 
  1 - (p_10 + p_9 + p_8) = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_8_l823_82361


namespace NUMINAMATH_CALUDE_rectangle_center_line_slope_l823_82313

/-- The slope of a line passing through the origin and the center of a rectangle with given vertices is 1/5 -/
theorem rectangle_center_line_slope :
  let vertices : List (ℝ × ℝ) := [(1, 0), (9, 0), (1, 2), (9, 2)]
  let center_x : ℝ := (vertices.map Prod.fst).sum / vertices.length
  let center_y : ℝ := (vertices.map Prod.snd).sum / vertices.length
  let slope : ℝ := center_y / center_x
  slope = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_center_line_slope_l823_82313


namespace NUMINAMATH_CALUDE_chairs_distribution_l823_82335

theorem chairs_distribution (total_chairs : Nat) (h1 : total_chairs = 123) :
  ∃! (num_boys chairs_per_boy : Nat),
    num_boys * chairs_per_boy = total_chairs ∧
    num_boys > 0 ∧
    chairs_per_boy > 0 ∧
    num_boys = 41 ∧
    chairs_per_boy = 3 := by
  sorry

end NUMINAMATH_CALUDE_chairs_distribution_l823_82335


namespace NUMINAMATH_CALUDE_alien_legs_count_l823_82305

/-- Represents the number of limbs for an alien or martian -/
structure Limbs where
  arms : ℕ
  legs : ℕ

/-- Defines the properties of alien limbs -/
def alien_limbs (l : ℕ) : Limbs :=
  { arms := 3, legs := l }

/-- Defines the properties of martian limbs based on alien legs -/
def martian_limbs (l : ℕ) : Limbs :=
  { arms := 2 * 3, legs := l / 2 }

/-- Theorem stating that aliens have 8 legs -/
theorem alien_legs_count : 
  ∃ l : ℕ, 
    (alien_limbs l).legs = 8 ∧ 
    5 * ((alien_limbs l).arms + (alien_limbs l).legs) = 
    5 * ((martian_limbs l).arms + (martian_limbs l).legs) + 5 :=
by
  sorry


end NUMINAMATH_CALUDE_alien_legs_count_l823_82305


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l823_82340

/-- Given two vectors OA and OB in 2D space, if they are perpendicular,
    then the second component of OB is 3/2. -/
theorem perpendicular_vectors_m_value (OA OB : ℝ × ℝ) :
  OA = (-1, 2) → OB.1 = 3 → OA.1 * OB.1 + OA.2 * OB.2 = 0 → OB.2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l823_82340
