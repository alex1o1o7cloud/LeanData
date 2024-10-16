import Mathlib

namespace NUMINAMATH_CALUDE_problem_1_problem_2_l652_65278

-- Problem 1
theorem problem_1 : 
  Real.sqrt 12 + |(-4)| - (2003 - Real.pi)^0 - 2 * Real.cos (30 * π / 180) = Real.sqrt 3 + 3 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℤ) (h1 : 0 < a) (h2 : a < 4) (h3 : a ≠ 2) : 
  (a + 2 - 5 / (a - 2)) / ((3 - a) / (2 * a - 4)) = -2 * a - 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l652_65278


namespace NUMINAMATH_CALUDE_dubblefud_yellow_chips_l652_65241

theorem dubblefud_yellow_chips :
  ∀ (yellow blue green red : ℕ),
  -- Yellow chips are worth 2 points
  -- Blue chips are worth 4 points
  -- Green chips are worth 5 points
  -- Red chips are worth 7 points
  -- The product of the point values of the chips is 560000
  2^yellow * 4^blue * 5^green * 7^red = 560000 →
  -- The number of blue chips equals twice the number of green chips
  blue = 2 * green →
  -- The number of red chips is half the number of blue chips
  red = blue / 2 →
  -- The number of yellow chips is 2
  yellow = 2 := by
sorry

end NUMINAMATH_CALUDE_dubblefud_yellow_chips_l652_65241


namespace NUMINAMATH_CALUDE_travel_equation_correct_l652_65267

/-- Represents the scenario of two vehicles traveling to a museum -/
structure TravelScenario where
  distance : ℝ
  bus_speed : ℝ
  car_speed_ratio : ℝ
  time_difference : ℝ

/-- The equation representing the travel scenario -/
def travel_equation (s : TravelScenario) : Prop :=
  s.distance / s.bus_speed - s.distance / (s.car_speed_ratio * s.bus_speed) = s.time_difference

/-- Theorem stating that the given equation correctly represents the travel scenario -/
theorem travel_equation_correct (s : TravelScenario) 
  (h1 : s.distance = 20)
  (h2 : s.car_speed_ratio = 1.5)
  (h3 : s.time_difference = 1/6) :
  travel_equation s :=
sorry

end NUMINAMATH_CALUDE_travel_equation_correct_l652_65267


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l652_65235

def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l652_65235


namespace NUMINAMATH_CALUDE_total_subjects_l652_65237

theorem total_subjects (average_all : ℝ) (average_five : ℝ) (last_subject : ℝ) 
  (h1 : average_all = 76)
  (h2 : average_five = 74)
  (h3 : last_subject = 86) :
  ∃ n : ℕ, n = 6 ∧ 
    n * average_all = (n - 1) * average_five + last_subject :=
by
  sorry

end NUMINAMATH_CALUDE_total_subjects_l652_65237


namespace NUMINAMATH_CALUDE_qq_level_difference_l652_65287

/-- Represents the QQ level system -/
structure QQLevel where
  activedays : ℕ
  stars : ℕ
  moons : ℕ
  suns : ℕ

/-- Calculates the total number of stars for a given level -/
def totalStars (level : ℕ) : ℕ := level

/-- Calculates the number of active days required for a given level -/
def activeDaysForLevel (level : ℕ) : ℕ := level * (level + 4)

/-- Converts stars to an equivalent QQ level -/
def starsToLevel (stars : ℕ) : ℕ := stars

/-- Theorem: The difference in active days between 1 sun and 2 moons 1 star is 203 -/
theorem qq_level_difference : 
  let sunLevel := starsToLevel (4 * 4)
  let currentLevel := starsToLevel (2 * 4 + 1)
  activeDaysForLevel sunLevel - activeDaysForLevel currentLevel = 203 := by
  sorry


end NUMINAMATH_CALUDE_qq_level_difference_l652_65287


namespace NUMINAMATH_CALUDE_triangle_theorem_l652_65268

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  angle_sum : A + B + C = π
  sine_rule_ab : a / (Real.sin A) = b / (Real.sin B)
  sine_rule_bc : b / (Real.sin B) = c / (Real.sin C)

/-- Main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h : Real.sin t.A * Real.sin t.B + Real.sin t.B * Real.sin t.C + Real.cos (2 * t.B) = 1) :
  -- Part 1: a, b, c are in arithmetic progression
  t.a + t.c = 2 * t.b ∧ 
  -- Part 2: If C = 2π/3, then a/b = 3/5
  (t.C = 2 * π / 3 → t.a / t.b = 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l652_65268


namespace NUMINAMATH_CALUDE_ackermann_3_2_l652_65210

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_3_2 : A 3 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_ackermann_3_2_l652_65210


namespace NUMINAMATH_CALUDE_probability_ratio_is_twenty_l652_65276

def total_balls : ℕ := 25
def num_bins : ℕ := 6

def distribution_A : List ℕ := [4, 4, 4, 5, 5, 2]
def distribution_B : List ℕ := [5, 5, 5, 5, 5, 0]

def probability_ratio : ℚ :=
  (Nat.choose num_bins 3 * Nat.choose 3 2 * Nat.choose 1 1 *
   (Nat.factorial total_balls / (Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 2))) /
  (Nat.choose num_bins 5 * Nat.choose 1 1 *
   (Nat.factorial total_balls / (Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 5 * Nat.factorial 0)))

theorem probability_ratio_is_twenty :
  probability_ratio = 20 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_is_twenty_l652_65276


namespace NUMINAMATH_CALUDE_suv_fuel_efficiency_l652_65275

theorem suv_fuel_efficiency (highway_mpg city_mpg : ℝ) (max_distance gallons : ℝ) 
  (h1 : highway_mpg = 12.2)
  (h2 : city_mpg = 7.6)
  (h3 : max_distance = 268.4)
  (h4 : gallons = 22)
  (h5 : max_distance = highway_mpg * gallons) :
  city_mpg * gallons = 167.2 := by
  sorry

end NUMINAMATH_CALUDE_suv_fuel_efficiency_l652_65275


namespace NUMINAMATH_CALUDE_factoring_equation_l652_65232

theorem factoring_equation (m : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + m * x + 1 = (2 * x - 1)^2) → 
  ∃ f : ℝ → ℝ, ∀ x : ℝ, 4 * x^2 + m * x + 1 = f x * f x :=
by sorry

end NUMINAMATH_CALUDE_factoring_equation_l652_65232


namespace NUMINAMATH_CALUDE_largest_angle_of_hexagon_l652_65224

/-- Proves that in a convex hexagon with given interior angle measures, the largest angle is 4374/21 degrees -/
theorem largest_angle_of_hexagon (a : ℚ) : 
  (a + 2) + (2 * a - 3) + (3 * a + 1) + (4 * a) + (5 * a - 4) + (6 * a + 2) = 720 →
  max (a + 2) (max (2 * a - 3) (max (3 * a + 1) (max (4 * a) (max (5 * a - 4) (6 * a + 2))))) = 4374 / 21 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_of_hexagon_l652_65224


namespace NUMINAMATH_CALUDE_interest_rate_proof_l652_65247

/-- Represents the annual interest rate as a real number between 0 and 1 -/
def annual_interest_rate : ℝ := 0.05

/-- The initial principal amount in rupees -/
def principal : ℝ := 4800

/-- The final amount after 2 years in rupees -/
def final_amount : ℝ := 5292

/-- The number of years the money is invested -/
def time : ℕ := 2

/-- The number of times interest is compounded per year -/
def compounds_per_year : ℕ := 1

theorem interest_rate_proof :
  final_amount = principal * (1 + annual_interest_rate) ^ (compounds_per_year * time) :=
sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l652_65247


namespace NUMINAMATH_CALUDE_quadratic_roots_square_relation_l652_65212

theorem quadratic_roots_square_relation (q : ℝ) : 
  (∃ (a b : ℝ), a ≠ b ∧ a^2 = b ∧ a^2 - 12*a + q = 0 ∧ b^2 - 12*b + q = 0) →
  (q = 27 ∨ q = -64) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_square_relation_l652_65212


namespace NUMINAMATH_CALUDE_good_numbers_exist_l652_65242

def has_no_repeating_digits (n : ℕ) : Prop :=
  (n / 10) % 10 ≠ n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def increase_digits (n : ℕ) : ℕ :=
  ((n / 10) + 1) * 10 + ((n % 10) + 1)

theorem good_numbers_exist : ∃ n₁ n₂ : ℕ,
  n₁ ≠ n₂ ∧
  10 ≤ n₁ ∧ n₁ < 100 ∧
  10 ≤ n₂ ∧ n₂ < 100 ∧
  has_no_repeating_digits n₁ ∧
  has_no_repeating_digits n₂ ∧
  n₁ % sum_of_digits n₁ = 0 ∧
  n₂ % sum_of_digits n₂ = 0 ∧
  has_no_repeating_digits (increase_digits n₁) ∧
  has_no_repeating_digits (increase_digits n₂) ∧
  (increase_digits n₁) % sum_of_digits (increase_digits n₁) = 0 ∧
  (increase_digits n₂) % sum_of_digits (increase_digits n₂) = 0 :=
sorry

end NUMINAMATH_CALUDE_good_numbers_exist_l652_65242


namespace NUMINAMATH_CALUDE_room_equation_l652_65296

/-- 
Theorem: For a positive integer x representing the number of rooms, 
if accommodating 6 people per room leaves exactly one room vacant, 
and accommodating 5 people per room leaves 4 people unaccommodated, 
then the equation 6(x-1) = 5x + 4 holds true.
-/
theorem room_equation (x : ℕ+) 
  (h1 : 6 * (x - 1) = 6 * x - 6)  -- With 6 people per room, one room is vacant
  (h2 : 5 * x + 4 = 6 * x - 6)    -- With 5 people per room, 4 people are unaccommodated
  : 6 * (x - 1) = 5 * x + 4 := by
  sorry


end NUMINAMATH_CALUDE_room_equation_l652_65296


namespace NUMINAMATH_CALUDE_quadratic_minimum_l652_65226

/-- Given a quadratic function f(x) = x^2 + px + qx where p and q are positive constants,
    prove that the x-coordinate of its minimum value occurs at x = -(p+q)/2 -/
theorem quadratic_minimum (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let f : ℝ → ℝ := λ x => x^2 + p*x + q*x
  ∃ (x_min : ℝ), x_min = -(p + q) / 2 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l652_65226


namespace NUMINAMATH_CALUDE_hilary_pakora_orders_l652_65218

/-- Represents the cost of a meal at Delicious Delhi restaurant -/
structure MealCost where
  samosas : ℕ
  pakoras : ℕ
  lassi : ℕ
  tip_percent : ℚ
  total_with_tax : ℚ

/-- Calculates the number of pakora orders given the meal cost details -/
def calculate_pakora_orders (meal : MealCost) : ℚ :=
  let samosa_cost := 2 * meal.samosas
  let lassi_cost := 2 * meal.lassi
  let pakora_cost := 3 * meal.pakoras
  let subtotal := samosa_cost + lassi_cost + pakora_cost
  let total_with_tip := subtotal * (1 + meal.tip_percent)
  (meal.total_with_tax - total_with_tip) / 3

/-- Theorem stating that Hilary bought 4 orders of pakoras -/
theorem hilary_pakora_orders :
  let meal := MealCost.mk 3 4 1 (1/4) 25
  calculate_pakora_orders meal = 4 := by
  sorry

end NUMINAMATH_CALUDE_hilary_pakora_orders_l652_65218


namespace NUMINAMATH_CALUDE_check_amount_error_l652_65289

theorem check_amount_error (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 →  -- x and y are two-digit numbers
  y - x = 18 →                         -- difference is $17.82
  ∃ x y : ℕ, y = 2 * x                 -- y can be twice x
:= by sorry

end NUMINAMATH_CALUDE_check_amount_error_l652_65289


namespace NUMINAMATH_CALUDE_race_result_kilometer_race_result_l652_65231

/-- Represents a runner in the race -/
structure Runner where
  time : ℝ  -- Time taken to complete the race in seconds
  distance : ℝ  -- Distance covered in meters

/-- The race scenario -/
def race_scenario (race_distance : ℝ) (a b : Runner) : Prop :=
  a.distance = race_distance ∧
  b.distance = race_distance ∧
  a.time + 10 = b.time ∧
  a.time = 390

/-- The theorem to be proved -/
theorem race_result (race_distance : ℝ) (a b : Runner) 
  (h : race_scenario race_distance a b) : 
  a.distance - b.distance * (a.time / b.time) = 25 := by
  sorry

/-- Main theorem stating the race result -/
theorem kilometer_race_result :
  ∃ (a b : Runner), race_scenario 1000 a b ∧ 
  a.distance - b.distance * (a.time / b.time) = 25 := by
  sorry

end NUMINAMATH_CALUDE_race_result_kilometer_race_result_l652_65231


namespace NUMINAMATH_CALUDE_f_monotonically_decreasing_l652_65207

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 2*x - 2

-- Theorem statement
theorem f_monotonically_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 1 → f x₁ > f x₂ := by sorry

end NUMINAMATH_CALUDE_f_monotonically_decreasing_l652_65207


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l652_65204

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l652_65204


namespace NUMINAMATH_CALUDE_puppy_cost_problem_l652_65265

theorem puppy_cost_problem (total_cost : ℕ) (sale_price : ℕ) (distinct_cost1 : ℕ) (distinct_cost2 : ℕ) :
  total_cost = 2200 →
  sale_price = 180 →
  distinct_cost1 = 250 →
  distinct_cost2 = 300 →
  ∃ (remaining_price : ℕ),
    4 * sale_price + distinct_cost1 + distinct_cost2 + 2 * remaining_price = total_cost ∧
    remaining_price = 465 := by
  sorry

end NUMINAMATH_CALUDE_puppy_cost_problem_l652_65265


namespace NUMINAMATH_CALUDE_point_on_ellipse_l652_65258

/-- Given points A(-5,0) and B(5,0), and a point M(x,y) such that the product of the slopes of AM
and BM is -2, prove that M lies on an ellipse centered at the origin. -/
theorem point_on_ellipse (x y : ℝ) (h : x ≠ 5 ∧ x ≠ -5) :
  (y / (x + 5)) * (y / (x - 5)) = -2 →
  x^2 / 25 + y^2 / 50 = 1 :=
by sorry

end NUMINAMATH_CALUDE_point_on_ellipse_l652_65258


namespace NUMINAMATH_CALUDE_shot_put_surface_area_l652_65272

/-- The surface area of a sphere with diameter 9 inches is 81π square inches. -/
theorem shot_put_surface_area :
  ∀ (d : ℝ), d = 9 → 4 * Real.pi * (d / 2)^2 = 81 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shot_put_surface_area_l652_65272


namespace NUMINAMATH_CALUDE_quadratic_trinomial_pairs_l652_65288

/-- Represents a quadratic trinomial ax^2 + bx + c -/
structure QuadraticTrinomial (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Checks if two numbers are roots of a quadratic trinomial -/
def areRoots {α : Type*} [Ring α] (t : QuadraticTrinomial α) (r1 r2 : α) : Prop :=
  t.a * r1 * r1 + t.b * r1 + t.c = 0 ∧ t.a * r2 * r2 + t.b * r2 + t.c = 0

theorem quadratic_trinomial_pairs 
  {α : Type*} [Field α] [CharZero α]
  (t1 t2 : QuadraticTrinomial α)
  (h1 : areRoots t2 t1.b t1.c)
  (h2 : areRoots t1 t2.b t2.c) :
  (∃ (a : α), t1 = ⟨1, a, 0⟩ ∧ t2 = ⟨1, -a, 0⟩) ∨
  (t1 = ⟨1, 1, -2⟩ ∧ t2 = ⟨1, 1, -2⟩) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_trinomial_pairs_l652_65288


namespace NUMINAMATH_CALUDE_coconut_grove_yield_is_120_l652_65266

/-- Represents the yield of x trees in a coconut grove with specific conditions -/
def coconut_grove_yield (x : ℕ) (yield_x : ℕ) : Prop :=
  let yield_xplus2 : ℕ := 30 * (x + 2)
  let yield_xminus2 : ℕ := 180 * (x - 2)
  let total_trees : ℕ := (x + 2) + x + (x - 2)
  let total_yield : ℕ := yield_xplus2 + (x * yield_x) + yield_xminus2
  (total_yield = total_trees * 100) ∧ (x = 10)

theorem coconut_grove_yield_is_120 :
  coconut_grove_yield 10 120 := by sorry

end NUMINAMATH_CALUDE_coconut_grove_yield_is_120_l652_65266


namespace NUMINAMATH_CALUDE_part_one_part_two_l652_65279

/-- A quadratic equation ax^2 + bx + c = 0 is a double root equation if one root is twice the other -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ 0 ∧ y = 2*x ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0

/-- The first part of the theorem -/
theorem part_one : is_double_root_equation 1 (-3) 2 := by sorry

/-- The second part of the theorem -/
theorem part_two :
  ∀ (a b : ℝ), is_double_root_equation a b (-6) →
  (∃ (x : ℝ), x = 2 ∧ a*x^2 + b*x - 6 = 0) →
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l652_65279


namespace NUMINAMATH_CALUDE_candy_ratio_l652_65293

/-- Given:
  - There were 22 sweets on the table initially.
  - Jack took some portion of all the candies and 4 more candies.
  - Paul took the remaining 7 sweets.
Prove that the ratio of candies Jack took (excluding the 4 additional candies) 
to the total number of candies is 1/2. -/
theorem candy_ratio : 
  ∀ (jack_portion : ℕ),
  jack_portion + 4 + 7 = 22 →
  (jack_portion : ℚ) / 22 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_candy_ratio_l652_65293


namespace NUMINAMATH_CALUDE_mixture_weight_theorem_l652_65251

/-- Atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Phosphorus in g/mol -/
def P_weight : ℝ := 30.97

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Sodium in g/mol -/
def Na_weight : ℝ := 22.99

/-- Atomic weight of Sulfur in g/mol -/
def S_weight : ℝ := 32.07

/-- Molecular weight of Aluminum phosphate (AlPO4) in g/mol -/
def AlPO4_weight : ℝ := Al_weight + P_weight + 4 * O_weight

/-- Molecular weight of Sodium sulfate (Na2SO4) in g/mol -/
def Na2SO4_weight : ℝ := 2 * Na_weight + S_weight + 4 * O_weight

/-- Total weight of the mixture in grams -/
def total_mixture_weight : ℝ := 5 * AlPO4_weight + 3 * Na2SO4_weight

theorem mixture_weight_theorem :
  total_mixture_weight = 1035.90 := by sorry

end NUMINAMATH_CALUDE_mixture_weight_theorem_l652_65251


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l652_65273

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l652_65273


namespace NUMINAMATH_CALUDE_geometry_propositions_l652_65282

-- Define the concepts
def Plane : Type := sorry
def Line : Type := sorry
def perpendicular (a b : Plane) : Prop := sorry
def parallel (a b : Plane) : Prop := sorry
def passes_through (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line (p : Plane) : Line := sorry
def in_plane (l : Line) (p : Plane) : Prop := sorry
def intersection_line (p q : Plane) : Line := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- Define the propositions
def proposition_1 : Prop :=
  ∀ (p q : Plane) (l : Line),
    passes_through l q ∧ l = perpendicular_line p → perpendicular p q

def proposition_2 : Prop :=
  ∀ (p q : Plane) (l m : Line),
    in_plane l p ∧ in_plane m p ∧ parallel l q ∧ parallel m q → parallel p q

def proposition_3 : Prop :=
  ∀ (p q : Plane) (l : Line),
    perpendicular p q ∧ in_plane l p ∧ ¬perpendicular l (intersection_line p q) →
    ¬perpendicular l q

def proposition_4 : Prop :=
  ∀ (p : Plane) (l m : Line),
    parallel l p ∧ parallel m p → parallel_lines l m

-- State the theorem
theorem geometry_propositions :
  proposition_1 ∧ proposition_3 ∧ ¬proposition_2 ∧ ¬proposition_4 := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l652_65282


namespace NUMINAMATH_CALUDE_banana_weight_per_truck_l652_65259

theorem banana_weight_per_truck 
  (total_apples : ℝ) 
  (apples_per_truck : ℝ) 
  (total_bananas : ℝ) 
  (h1 : total_apples = 132.6) 
  (h2 : apples_per_truck = 13.26) 
  (h3 : total_bananas = 6.4) :
  let num_trucks : ℝ := total_apples / apples_per_truck
  total_bananas / num_trucks = 0.64 := by
sorry

end NUMINAMATH_CALUDE_banana_weight_per_truck_l652_65259


namespace NUMINAMATH_CALUDE_angle_side_relationship_l652_65229

-- Define a triangle with angles A, B, C and sides a, b, c
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  -- Triangle inequality
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  -- Angle sum in a triangle is π
  angle_sum : A + B + C = π
  -- Side lengths satisfy triangle inequality
  side_ineq_a : a < b + c
  side_ineq_b : b < a + c
  side_ineq_c : c < a + b

theorem angle_side_relationship (t : Triangle) : t.A > t.B ↔ t.a > t.b := by
  sorry

end NUMINAMATH_CALUDE_angle_side_relationship_l652_65229


namespace NUMINAMATH_CALUDE_negation_equivalence_l652_65264

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l652_65264


namespace NUMINAMATH_CALUDE_set_operations_and_range_l652_65291

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

theorem set_operations_and_range :
  (A ∩ B = {x | 2 < x ∧ x ≤ 3}) ∧
  (A ∪ (U \ B) = {x | x ≤ 3 ∨ x ≥ 4}) ∧
  (∀ a : ℝ, B ∩ C a = C a → 2 < a ∧ a < 3) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l652_65291


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l652_65215

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 2) :
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l652_65215


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l652_65227

-- Define the lines and point
def line1 (m : ℝ) (x y : ℝ) : Prop := 2 * x + m * y - 1 = 0
def line2 (n : ℝ) (x y : ℝ) : Prop := 3 * x - 2 * y + n = 0
def foot (p : ℝ) : ℝ × ℝ := (2, p)

-- State the theorem
theorem perpendicular_lines_sum (m n p : ℝ) : 
  (∀ x y, line1 m x y → line2 n x y → (x - 2) * (3 * x - 2 * y + n) + (y - p) * (2 * x + m * y - 1) = 0) →  -- perpendicularity condition
  line1 m 2 p →  -- foot satisfies line1
  line2 n 2 p →  -- foot satisfies line2
  m + n + p = -6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l652_65227


namespace NUMINAMATH_CALUDE_gold_bars_weighing_l652_65223

theorem gold_bars_weighing (C₁ C₂ C₃ C₄ C₅ C₆ C₇ C₈ C₉ C₁₀ C₁₁ C₁₂ C₁₃ : ℝ) 
  (h₁ : C₁ ≥ 0) (h₂ : C₂ ≥ 0) (h₃ : C₃ ≥ 0) (h₄ : C₄ ≥ 0) (h₅ : C₅ ≥ 0)
  (h₆ : C₆ ≥ 0) (h₇ : C₇ ≥ 0) (h₈ : C₈ ≥ 0) (h₉ : C₉ ≥ 0) (h₁₀ : C₁₀ ≥ 0)
  (h₁₁ : C₁₁ ≥ 0) (h₁₂ : C₁₂ ≥ 0) (h₁₃ : C₁₃ ≥ 0)
  (W₁ : ℝ) (hW₁ : W₁ = C₁ + C₂)
  (W₂ : ℝ) (hW₂ : W₂ = C₁ + C₃)
  (W₃ : ℝ) (hW₃ : W₃ = C₂ + C₃)
  (W₄ : ℝ) (hW₄ : W₄ = C₄ + C₅)
  (W₅ : ℝ) (hW₅ : W₅ = C₆ + C₇)
  (W₆ : ℝ) (hW₆ : W₆ = C₈ + C₉)
  (W₇ : ℝ) (hW₇ : W₇ = C₁₀ + C₁₁)
  (W₈ : ℝ) (hW₈ : W₈ = C₁₂ + C₁₃) :
  C₁ + C₂ + C₃ + C₄ + C₅ + C₆ + C₇ + C₈ + C₉ + C₁₀ + C₁₁ + C₁₂ + C₁₃ = 
  (W₁ + W₂ + W₃) / 2 + (W₄ + W₅ + W₆ + W₇ + W₈) :=
by sorry

end NUMINAMATH_CALUDE_gold_bars_weighing_l652_65223


namespace NUMINAMATH_CALUDE_three_part_division_l652_65299

theorem three_part_division (total : ℚ) (p1 p2 p3 : ℚ) (h1 : total = 78)
  (h2 : p1 + p2 + p3 = total) (h3 : p2 = (1/3) * p1) (h4 : p3 = (1/6) * p1) :
  p2 = 17 + (1/3) :=
by sorry

end NUMINAMATH_CALUDE_three_part_division_l652_65299


namespace NUMINAMATH_CALUDE_parallelogram_area_l652_65260

/-- The area of a parallelogram is the product of two adjacent sides and the sine of the angle between them. -/
theorem parallelogram_area (a b : ℝ) (γ : ℝ) (ha : 0 < a) (hb : 0 < b) (hγ : 0 < γ ∧ γ < π) :
  ∃ (S : ℝ), S = a * b * Real.sin γ ∧ S > 0 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l652_65260


namespace NUMINAMATH_CALUDE_rectangular_field_length_l652_65236

theorem rectangular_field_length (length width : ℝ) 
  (h1 : length * width = 144)
  (h2 : (length + 6) * width = 198) :
  length = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l652_65236


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l652_65290

theorem wrapping_paper_fraction (total_fraction : ℚ) (num_presents : ℕ) 
  (h1 : total_fraction = 3 / 10)
  (h2 : num_presents = 3) :
  total_fraction / num_presents = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l652_65290


namespace NUMINAMATH_CALUDE_divides_m_implies_divides_m_times_n_plus_one_l652_65244

theorem divides_m_implies_divides_m_times_n_plus_one (m n : ℤ) :
  n ∣ m * (n + 1) → n ∣ m := by
  sorry

end NUMINAMATH_CALUDE_divides_m_implies_divides_m_times_n_plus_one_l652_65244


namespace NUMINAMATH_CALUDE_salary_change_percentage_l652_65225

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 96 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l652_65225


namespace NUMINAMATH_CALUDE_berries_taken_l652_65213

theorem berries_taken (stacy_initial : ℕ) (steve_initial : ℕ) (difference : ℕ) : 
  stacy_initial = 32 →
  steve_initial = 21 →
  difference = 7 →
  ∃ (berries_taken : ℕ), 
    steve_initial + berries_taken = stacy_initial - difference ∧
    berries_taken = 4 :=
by sorry

end NUMINAMATH_CALUDE_berries_taken_l652_65213


namespace NUMINAMATH_CALUDE_leftHandedLikeMusicalCount_l652_65277

/-- Represents the Drama Club -/
structure DramaClub where
  total : Nat
  leftHanded : Nat
  likeMusical : Nat
  rightHandedDislike : Nat

/-- The number of left-handed people who like musical theater in the Drama Club -/
def leftHandedLikeMusical (club : DramaClub) : Nat :=
  club.leftHanded + club.likeMusical - (club.total - club.rightHandedDislike)

/-- Theorem stating the number of left-handed musical theater lovers in the specific Drama Club -/
theorem leftHandedLikeMusicalCount : leftHandedLikeMusical { 
  total := 25,
  leftHanded := 10,
  likeMusical := 18,
  rightHandedDislike := 3
} = 6 := by sorry

end NUMINAMATH_CALUDE_leftHandedLikeMusicalCount_l652_65277


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l652_65245

/-- Given a large square with side length 8y and a smaller central square with side length 3y,
    where the large square is divided into the smaller central square and four congruent rectangles,
    the perimeter of one of these rectangles is 16y. -/
theorem rectangle_perimeter (y : ℝ) : 
  let large_square_side : ℝ := 8 * y
  let small_square_side : ℝ := 3 * y
  let rectangle_width : ℝ := small_square_side
  let rectangle_height : ℝ := large_square_side - small_square_side
  let rectangle_perimeter : ℝ := 2 * (rectangle_width + rectangle_height)
  rectangle_perimeter = 16 * y :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l652_65245


namespace NUMINAMATH_CALUDE_height_growth_l652_65211

theorem height_growth (current_height : ℝ) (growth_rate : ℝ) (original_height : ℝ) : 
  current_height = 147 ∧ growth_rate = 0.05 → original_height = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_height_growth_l652_65211


namespace NUMINAMATH_CALUDE_segment_length_problem_l652_65248

/-- Given a line segment AD of length 56 units, divided into three segments AB, BC, and CD,
    where AB : BC = 1 : 2 and BC : CD = 6 : 5, the length of AB is 12 units. -/
theorem segment_length_problem (AB BC CD : ℝ) : 
  AB + BC + CD = 56 → 
  AB / BC = 1 / 2 → 
  BC / CD = 6 / 5 → 
  AB = 12 := by
sorry

end NUMINAMATH_CALUDE_segment_length_problem_l652_65248


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l652_65202

theorem arithmetic_calculation : 
  let a := 65 * ((13/3 + 7/2) / (11/5 - 5/3))
  ∃ (n : ℕ) (m : ℚ), 0 ≤ m ∧ m < 1 ∧ a = n + m ∧ n = 954 ∧ m = 33/48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l652_65202


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l652_65219

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

/-- Point A -/
def A : ℝ × ℝ := (2, 2)

/-- The line on which point A lies -/
def line (m n l : ℝ) (x y : ℝ) : Prop := m*x + n*y = l

theorem tangent_line_minimum_value (m n l : ℝ) (hm : m > 0) (hn : n > 0) :
  line m n l A.1 A.2 →
  f' A.1 = 4 →
  ∀ k₁ k₂ : ℝ, k₁ > 0 → k₂ > 0 → line k₁ k₂ l A.1 A.2 → 
  1/k₁ + 2/k₂ ≥ 6 + 4*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l652_65219


namespace NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l652_65240

theorem triangle_pentagon_side_ratio :
  ∀ (t p : ℝ),
  t > 0 ∧ p > 0 →
  3 * t = 30 →
  5 * p = 30 →
  t / p = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l652_65240


namespace NUMINAMATH_CALUDE_circle_area_difference_l652_65201

theorem circle_area_difference : 
  let r1 : ℝ := 15
  let d2 : ℝ := 14
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 176 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l652_65201


namespace NUMINAMATH_CALUDE_difference_of_squares_24_13_l652_65208

theorem difference_of_squares_24_13 : (24 + 13)^2 - (24 - 13)^2 = 407 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_24_13_l652_65208


namespace NUMINAMATH_CALUDE_original_light_wattage_l652_65216

/-- Given a new light with 25% higher wattage than the original light,
    proves that if the new light has 100 watts, then the original light had 80 watts. -/
theorem original_light_wattage (new_wattage : ℝ) (h1 : new_wattage = 100) :
  let original_wattage := new_wattage / 1.25
  original_wattage = 80 := by
sorry

end NUMINAMATH_CALUDE_original_light_wattage_l652_65216


namespace NUMINAMATH_CALUDE_nursing_home_flowers_l652_65230

/-- The number of flower sets bought by Mayor Harvey -/
def num_sets : ℕ := 3

/-- The number of flowers in each set -/
def flowers_per_set : ℕ := 90

/-- The total number of flowers bought for the nursing home -/
def total_flowers : ℕ := num_sets * flowers_per_set

theorem nursing_home_flowers : total_flowers = 270 := by
  sorry

end NUMINAMATH_CALUDE_nursing_home_flowers_l652_65230


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l652_65281

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l652_65281


namespace NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_for_skew_lines_l652_65214

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the intersection relation between two planes
variable (intersect : Plane → Plane → Prop)

-- Define the skew relation between two lines
variable (skew : Line → Line → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

theorem planes_intersect_necessary_not_sufficient_for_skew_lines
  (α β : Plane) (m n : Line)
  (h_distinct : α ≠ β)
  (h_perp_m : perp m α)
  (h_perp_n : perp n β) :
  (∀ α β m n, skew m n → intersect α β) ∧
  (∃ α β m n, intersect α β ∧ perp m α ∧ perp n β ∧ ¬skew m n) :=
sorry

end NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_for_skew_lines_l652_65214


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l652_65285

theorem binomial_expansion_example : 121 + 2 * 11 * 9 + 81 = 400 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l652_65285


namespace NUMINAMATH_CALUDE_solid_volume_l652_65284

/-- A solid with a square base and specific edge lengths -/
structure Solid where
  s : ℝ
  base_side_length : s > 0
  upper_edge_length : ℝ := 3 * s
  other_edge_length : ℝ := s

/-- The volume of the solid -/
def volume (solid : Solid) : ℝ := sorry

theorem solid_volume : 
  ∀ (solid : Solid), solid.s = 8 * Real.sqrt 2 → volume solid = 5760 := by
  sorry

end NUMINAMATH_CALUDE_solid_volume_l652_65284


namespace NUMINAMATH_CALUDE_calculation_proof_l652_65200

theorem calculation_proof : (8 * 5.4 - 0.6 * 10 / 1.2)^2 = 1459.24 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l652_65200


namespace NUMINAMATH_CALUDE_easter_eggs_per_basket_l652_65261

theorem easter_eggs_per_basket : ∃ (n : ℕ), n ≥ 5 ∧ n ∣ 30 ∧ n ∣ 42 ∧ ∀ (m : ℕ), m ≥ 5 ∧ m ∣ 30 ∧ m ∣ 42 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_easter_eggs_per_basket_l652_65261


namespace NUMINAMATH_CALUDE_unique_set_satisfying_condition_l652_65255

theorem unique_set_satisfying_condition :
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    (a * b * c % d = 1) ∧
    (a * b * d % c = 1) ∧
    (a * c * d % b = 1) ∧
    (b * c * d % a = 1) →
    ({a, b, c, d} : Set ℕ) = {1, 2, 3, 4} :=
by sorry

end NUMINAMATH_CALUDE_unique_set_satisfying_condition_l652_65255


namespace NUMINAMATH_CALUDE_sum_of_cubes_l652_65217

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 14) : 
  x^3 + y^3 = 176 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l652_65217


namespace NUMINAMATH_CALUDE_problem_solution_l652_65221

theorem problem_solution (x y : ℝ) 
  (h1 : x = 52) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y + 100 = 540000) : 
  y = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l652_65221


namespace NUMINAMATH_CALUDE_sandwich_filler_percentage_l652_65253

/-- Given a sandwich with a total weight of 180 grams and filler weight of 45 grams,
    prove that the percentage of the sandwich that is not filler is 75%. -/
theorem sandwich_filler_percentage (total_weight filler_weight : ℝ) 
    (h1 : total_weight = 180)
    (h2 : filler_weight = 45) :
    (total_weight - filler_weight) / total_weight = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_filler_percentage_l652_65253


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l652_65257

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l652_65257


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l652_65269

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_of_geometric_progression 
  (a : ℝ) (r : ℝ) (h1 : a > 0) (h2 : r > 0) :
  geometric_progression a r 1 = 4 ∧ 
  geometric_progression a r 2 = Real.sqrt 4 ∧ 
  geometric_progression a r 3 = 4^(1/4) →
  geometric_progression a r 4 = 4^(1/8) := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l652_65269


namespace NUMINAMATH_CALUDE_stripe_width_for_equal_areas_l652_65238

/-- Given a rectangle with dimensions 40 cm × 20 cm and two perpendicular stripes of equal width,
    prove that the width of the stripes for equal white and gray areas is 30 - 5√5 cm. -/
theorem stripe_width_for_equal_areas : ∃ (x : ℝ),
  x > 0 ∧ x < 20 ∧
  (40 * x + 20 * x - x^2 = (40 * 20) / 2) ∧
  x = 30 - 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_stripe_width_for_equal_areas_l652_65238


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l652_65274

theorem quadratic_completing_square (x : ℝ) : 
  (4 * x^2 - 24 * x - 96 = 0) → 
  ∃ q t : ℝ, ((x + q)^2 = t) ∧ (t = 33) := by
sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l652_65274


namespace NUMINAMATH_CALUDE_calculation_equality_algebraic_simplification_l652_65298

-- Part 1
theorem calculation_equality : (-(1/3))⁻¹ + (2015 - Real.sqrt 3)^0 - 4 * Real.sin (60 * π / 180) + |(- Real.sqrt 12)| = -2 := by sorry

-- Part 2
theorem algebraic_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 ≠ b^2) :
  ((1 / (a + b) - 1 / (a - b)) / (b / (a^2 - 2*a*b + b^2))) = -2*(a - b)/(a + b) := by sorry

end NUMINAMATH_CALUDE_calculation_equality_algebraic_simplification_l652_65298


namespace NUMINAMATH_CALUDE_four_common_tangents_l652_65234

-- Define the circle type
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

-- Define the function to count common tangents
def countCommonTangents (c1 c2 : Circle) : ℕ := sorry

-- Theorem statement
theorem four_common_tangents (c1 c2 : Circle) 
  (h1 : c1.radius = 3)
  (h2 : c2.radius = 5)
  (h3 : Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 10) :
  countCommonTangents c1 c2 = 4 := by sorry

end NUMINAMATH_CALUDE_four_common_tangents_l652_65234


namespace NUMINAMATH_CALUDE_retail_price_increase_l652_65270

theorem retail_price_increase (W R : ℝ) 
  (h : 0.80 * R = 1.44000000000000014 * W) : 
  (R - W) / W * 100 = 80.000000000000017 :=
by sorry

end NUMINAMATH_CALUDE_retail_price_increase_l652_65270


namespace NUMINAMATH_CALUDE_die_roll_probability_l652_65263

/-- The probability of rolling a different number on a six-sided die -/
def p_different : ℚ := 5 / 6

/-- The probability of rolling the same number on a six-sided die -/
def p_same : ℚ := 1 / 6

/-- The number of rolls before the final roll -/
def n : ℕ := 9

theorem die_roll_probability :
  p_different ^ n * p_same = (5^8 : ℚ) / (6^9 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l652_65263


namespace NUMINAMATH_CALUDE_m_range_theorem_l652_65254

-- Define the statements p and q
def p (x : ℝ) : Prop := -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

-- Define the set A (where p is true)
def A : Set ℝ := {x | p x}

-- Define the set B (where q is true)
def B (m : ℝ) : Set ℝ := {x | q x m}

-- State the theorem
theorem m_range_theorem :
  ∀ m : ℝ, 
    (∀ x : ℝ, x ∈ A → x ∈ B m) ∧  -- p implies q
    (∃ x : ℝ, x ∈ B m ∧ x ∉ A) ∧  -- q does not imply p
    m ≥ 40 ∧ m < 50               -- m is in [40, 50)
  ↔ m ∈ Set.Icc 40 50 := by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l652_65254


namespace NUMINAMATH_CALUDE_marble_problem_l652_65286

theorem marble_problem (total : ℕ) (white : ℕ) (remaining : ℕ) : 
  total = 50 → 
  white = 20 → 
  remaining = 40 → 
  ∃ (red blue removed : ℕ),
    red = blue ∧ 
    total = white + red + blue ∧
    removed = total - remaining ∧
    removed = 2 * (white - blue) :=
by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l652_65286


namespace NUMINAMATH_CALUDE_zero_in_M_l652_65295

def M : Set ℝ := {x | x ≤ 2}

theorem zero_in_M : (0 : ℝ) ∈ M := by
  sorry

end NUMINAMATH_CALUDE_zero_in_M_l652_65295


namespace NUMINAMATH_CALUDE_factors_of_M_l652_65262

/-- The number of natural-number factors of M, where M = 2^4 · 3^3 · 7^1 · 11^2 -/
def num_factors (M : ℕ) : ℕ :=
  if M = 2^4 * 3^3 * 7^1 * 11^2 then 120 else 0

/-- Theorem stating that the number of natural-number factors of M is 120 -/
theorem factors_of_M :
  ∀ M : ℕ, M = 2^4 * 3^3 * 7^1 * 11^2 → num_factors M = 120 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_M_l652_65262


namespace NUMINAMATH_CALUDE_no_m_for_all_x_range_for_m_in_interval_l652_65243

-- Part 1
theorem no_m_for_all_x : ∀ m : ℝ, ∃ x : ℝ, 2 * x - 1 ≤ m * (x^2 - 1) := by sorry

-- Part 2
def inequality_set (m : ℝ) : Set ℝ := {x | 2 * x - 1 > m * (x^2 - 1)}

theorem range_for_m_in_interval :
  ∀ m ∈ Set.Icc (-2 : ℝ) 2,
  inequality_set m = Set.Ioo (((-1 : ℝ) + Real.sqrt 7) / 2) ((1 + Real.sqrt 3) / 2) := by sorry

end NUMINAMATH_CALUDE_no_m_for_all_x_range_for_m_in_interval_l652_65243


namespace NUMINAMATH_CALUDE_right_triangle_sets_l652_65239

theorem right_triangle_sets : ∃! (a b c : ℝ), (a = 4 ∧ b = 6 ∧ c = 8) ∧
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) ∧
  ((3^2 + 4^2 = 5^2) ∧ (5^2 + 12^2 = 13^2) ∧ (2^2 + 3^2 = (Real.sqrt 13)^2)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l652_65239


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l652_65228

theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l652_65228


namespace NUMINAMATH_CALUDE_points_form_parabola_l652_65206

-- Define the set of points (x, y) parametrically
def S : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p.1 = Real.cos t ^ 2 ∧ p.2 = Real.sin (2 * t)}

-- Define a parabola in general form
def IsParabola (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧
    ∀ p ∈ S, a * p.1^2 + b * p.1 * p.2 + c * p.2^2 + d * p.1 + e * p.2 = 0

-- Theorem statement
theorem points_form_parabola : IsParabola S := by
  sorry

end NUMINAMATH_CALUDE_points_form_parabola_l652_65206


namespace NUMINAMATH_CALUDE_min_perimeter_of_rectangle_l652_65297

theorem min_perimeter_of_rectangle (l w : ℕ) : 
  l * w = 50 → 2 * (l + w) ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_of_rectangle_l652_65297


namespace NUMINAMATH_CALUDE_prob_two_red_without_replacement_prob_two_red_with_replacement_prob_at_least_one_red_with_replacement_l652_65209

def total_balls : ℕ := 5
def red_balls : ℕ := 3
def white_balls : ℕ := 2

/-- Probability of drawing exactly 2 red balls without replacement -/
theorem prob_two_red_without_replacement :
  (Nat.choose red_balls 2 : ℚ) / (Nat.choose total_balls 2) = 3 / 10 := by sorry

/-- Probability of drawing exactly 2 red balls with replacement -/
theorem prob_two_red_with_replacement :
  (red_balls : ℚ) / total_balls * (red_balls : ℚ) / total_balls = 9 / 25 := by sorry

/-- Probability of drawing at least 1 red ball with replacement -/
theorem prob_at_least_one_red_with_replacement :
  1 - ((white_balls : ℚ) / total_balls) ^ 2 = 21 / 25 := by sorry

end NUMINAMATH_CALUDE_prob_two_red_without_replacement_prob_two_red_with_replacement_prob_at_least_one_red_with_replacement_l652_65209


namespace NUMINAMATH_CALUDE_total_decrease_percentage_l652_65294

-- Define the percentage decreases
def first_year_decrease : ℝ := 0.4
def second_year_decrease : ℝ := 0.1

-- Define the theorem
theorem total_decrease_percentage :
  ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - first_year_decrease)
  let final_value := value_after_first_year * (1 - second_year_decrease)
  let total_decrease := (initial_value - final_value) / initial_value
  total_decrease = 0.46 := by
  sorry

end NUMINAMATH_CALUDE_total_decrease_percentage_l652_65294


namespace NUMINAMATH_CALUDE_contrapositive_truth_square_less_than_one_implies_absolute_less_than_one_contrapositive_of_square_less_than_one_is_true_l652_65203

theorem contrapositive_truth (P Q : Prop) :
  (P → Q) → (¬Q → ¬P) := by sorry

theorem square_less_than_one_implies_absolute_less_than_one :
  ∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1 := by sorry

theorem contrapositive_of_square_less_than_one_is_true :
  (∀ x : ℝ, ¬(-1 < x ∧ x < 1) → ¬(x^2 < 1)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_truth_square_less_than_one_implies_absolute_less_than_one_contrapositive_of_square_less_than_one_is_true_l652_65203


namespace NUMINAMATH_CALUDE_sum_negative_implies_at_most_one_positive_l652_65246

theorem sum_negative_implies_at_most_one_positive (a b : ℚ) :
  a + b < 0 → (0 < a ∧ 0 < b) → False := by sorry

end NUMINAMATH_CALUDE_sum_negative_implies_at_most_one_positive_l652_65246


namespace NUMINAMATH_CALUDE_union_M_complement_N_equals_U_l652_65222

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x < 1}

def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem union_M_complement_N_equals_U : M ∪ (U \ N) = U := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_equals_U_l652_65222


namespace NUMINAMATH_CALUDE_atlantic_charge_calculation_l652_65280

/-- Represents the additional charge per minute for Atlantic Call -/
def atlantic_charge_per_minute : ℚ := 1/5

/-- United Telephone's base rate -/
def united_base_rate : ℚ := 11

/-- United Telephone's charge per minute -/
def united_charge_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate -/
def atlantic_base_rate : ℚ := 12

/-- Number of minutes for which the bills are equal -/
def equal_bill_minutes : ℕ := 20

theorem atlantic_charge_calculation :
  united_base_rate + united_charge_per_minute * equal_bill_minutes =
  atlantic_base_rate + atlantic_charge_per_minute * equal_bill_minutes :=
sorry

end NUMINAMATH_CALUDE_atlantic_charge_calculation_l652_65280


namespace NUMINAMATH_CALUDE_product_not_divisible_by_prime_l652_65233

theorem product_not_divisible_by_prime (p a b : ℕ) : 
  Prime p → a > 0 → b > 0 → a < p → b < p → ¬(p ∣ (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_product_not_divisible_by_prime_l652_65233


namespace NUMINAMATH_CALUDE_parallelogram_bisector_intersection_inside_l652_65252

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : sorry

/-- An angle bisector of a parallelogram is a line that bisects one of its angles. -/
def angle_bisector (p : Parallelogram) (i : Fin 4) : Set (ℝ × ℝ) := sorry

/-- The pairwise intersections of angle bisectors of a parallelogram. -/
def bisector_intersections (p : Parallelogram) : Set (ℝ × ℝ) := sorry

/-- A point is inside a parallelogram if it's in the interior of the parallelogram. -/
def inside_parallelogram (p : Parallelogram) (point : ℝ × ℝ) : Prop := sorry

theorem parallelogram_bisector_intersection_inside 
  (p : Parallelogram) : 
  ∃ (point : ℝ × ℝ), point ∈ bisector_intersections p ∧ inside_parallelogram p point :=
sorry

end NUMINAMATH_CALUDE_parallelogram_bisector_intersection_inside_l652_65252


namespace NUMINAMATH_CALUDE_problem_solving_probability_l652_65249

theorem problem_solving_probability (p_a p_either : ℝ) (h1 : p_a = 0.7) (h2 : p_either = 0.94) :
  ∃ p_b : ℝ, p_b = 0.8 ∧ p_either = 1 - (1 - p_a) * (1 - p_b) := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l652_65249


namespace NUMINAMATH_CALUDE_johns_workday_end_l652_65205

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_def : minutes < 60

/-- Calculates the difference between two times in hours -/
def time_diff (t1 t2 : Time) : ℚ :=
  (t2.hours - t1.hours : ℚ) + (t2.minutes - t1.minutes : ℚ) / 60

/-- Adds hours and minutes to a given time -/
def add_time (t : Time) (h : ℕ) (m : ℕ) : Time :=
  let total_minutes := t.hours * 60 + t.minutes + h * 60 + m
  { hours := total_minutes / 60,
    minutes := total_minutes % 60,
    inv_def := by sorry }

theorem johns_workday_end (work_hours : ℕ) (lunch_break : Time) (start_time end_time : Time) :
  work_hours = 9 →
  lunch_break.hours = 1 ∧ lunch_break.minutes = 15 →
  start_time.hours = 6 ∧ start_time.minutes = 30 →
  time_diff start_time { hours := 11, minutes := 30, inv_def := by sorry } = 5 →
  add_time { hours := 11, minutes := 30, inv_def := by sorry } lunch_break.hours lunch_break.minutes = { hours := 12, minutes := 45, inv_def := by sorry } →
  add_time { hours := 12, minutes := 45, inv_def := by sorry } 4 0 = end_time →
  end_time.hours = 16 ∧ end_time.minutes = 45 :=
by sorry

end NUMINAMATH_CALUDE_johns_workday_end_l652_65205


namespace NUMINAMATH_CALUDE_real_solutions_condition_l652_65292

theorem real_solutions_condition (a : ℝ) :
  (∃ x : ℝ, |x| + x^2 = a) ↔ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_condition_l652_65292


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l652_65283

/-- An even function that is decreasing on the non-negative reals -/
def EvenDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

/-- The inequality condition from the problem -/
def InequalityCondition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, 0 < x → x ≤ Real.sqrt 2 → 
    f (-a * x + x^3 + 1) + f (a * x - x^3 - 1) ≥ 2 * f 1

theorem function_inequality_implies_a_range 
  (f : ℝ → ℝ) (a : ℝ) 
  (hf : EvenDecreasingFunction f) 
  (h_ineq : InequalityCondition f a) : 
  2 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l652_65283


namespace NUMINAMATH_CALUDE_women_married_long_service_fraction_l652_65256

theorem women_married_long_service_fraction 
  (total_employees : ℕ) 
  (women_percentage : ℚ)
  (married_percentage : ℚ)
  (single_men_fraction : ℚ)
  (married_long_service_women_percentage : ℚ)
  (h1 : women_percentage = 76 / 100)
  (h2 : married_percentage = 60 / 100)
  (h3 : single_men_fraction = 2 / 3)
  (h4 : married_long_service_women_percentage = 70 / 100)
  : ℚ :=
by
  sorry

#check women_married_long_service_fraction

end NUMINAMATH_CALUDE_women_married_long_service_fraction_l652_65256


namespace NUMINAMATH_CALUDE_quadratic_factorization_l652_65250

theorem quadratic_factorization (y : ℝ) : 3 * y^2 - 6 * y + 3 = 3 * (y - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l652_65250


namespace NUMINAMATH_CALUDE_order_of_abc_l652_65220

theorem order_of_abc : 
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l652_65220


namespace NUMINAMATH_CALUDE_gohul_independent_time_l652_65271

/-- Ram's work rate in job completion per day -/
def ram_rate : ℚ := 1 / 10

/-- Time taken when Ram and Gohul work together -/
def combined_time : ℚ := 5.999999999999999

/-- Gohul's independent work time -/
def gohul_time : ℚ := 15

/-- Combined work rate of Ram and Gohul -/
def combined_rate : ℚ := 1 / combined_time

theorem gohul_independent_time :
  ram_rate + (1 / gohul_time) = combined_rate :=
sorry

end NUMINAMATH_CALUDE_gohul_independent_time_l652_65271
