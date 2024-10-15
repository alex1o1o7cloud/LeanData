import Mathlib

namespace NUMINAMATH_CALUDE_sheep_flock_size_l3954_395474

theorem sheep_flock_size :
  ∀ (x y : ℕ),
  (x - 1 : ℚ) / y = 7 / 5 →
  x / (y - 1 : ℚ) = 5 / 3 →
  x + y = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_sheep_flock_size_l3954_395474


namespace NUMINAMATH_CALUDE_units_digit_sum_series_l3954_395423

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_series : ℕ := 
  (units_digit (factorial 1)) + 
  (units_digit ((factorial 2)^2)) + 
  (units_digit (factorial 3)) + 
  (units_digit ((factorial 4)^2)) + 
  (units_digit (factorial 5)) + 
  (units_digit ((factorial 6)^2)) + 
  (units_digit (factorial 7)) + 
  (units_digit ((factorial 8)^2)) + 
  (units_digit (factorial 9)) + 
  (units_digit ((factorial 10)^2))

theorem units_digit_sum_series : units_digit sum_series = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_series_l3954_395423


namespace NUMINAMATH_CALUDE_base6_addition_example_l3954_395451

/-- Addition in base 6 -/
def base6_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 6 to base 10 -/
def base6_to_base10 (n : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 6 -/
def base10_to_base6 (n : ℕ) : ℕ := sorry

theorem base6_addition_example : base6_add 152 35 = 213 := by sorry

end NUMINAMATH_CALUDE_base6_addition_example_l3954_395451


namespace NUMINAMATH_CALUDE_circle_radius_l3954_395457

theorem circle_radius (x y d : Real) (h : x + y + d = 164 * Real.pi) :
  ∃ (r : Real), r = 10 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ d = 2 * r := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3954_395457


namespace NUMINAMATH_CALUDE_expression_evaluation_l3954_395424

theorem expression_evaluation : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3954_395424


namespace NUMINAMATH_CALUDE_min_distance_sum_l3954_395435

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -4*x

-- Define the focus F (we don't know its exact coordinates, but we know it exists)
axiom F : ℝ × ℝ

-- Define point A
def A : ℝ × ℝ := (-2, 1)

-- Define a point P on the parabola
structure PointOnParabola where
  P : ℝ × ℝ
  on_parabola : parabola P.1 P.2

-- State the theorem
theorem min_distance_sum (p : PointOnParabola) :
  ∃ (min : ℝ), min = 3 ∧ ∀ (q : PointOnParabola), 
    Real.sqrt ((q.P.1 - F.1)^2 + (q.P.2 - F.2)^2) +
    Real.sqrt ((q.P.1 - A.1)^2 + (q.P.2 - A.2)^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_sum_l3954_395435


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l3954_395463

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Theorem: The point symmetric to (2, -3) with respect to the origin is (-2, 3) -/
theorem symmetric_point_theorem :
  let P : Point := { x := 2, y := -3 }
  let P' : Point := symmetricToOrigin P
  P'.x = -2 ∧ P'.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l3954_395463


namespace NUMINAMATH_CALUDE_kevin_ran_17_miles_l3954_395466

/-- Calculates the total distance Kevin ran given his running segments -/
def kevin_total_distance (speed1 speed2 speed3 : ℝ) (time1 time2 time3 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3

/-- Theorem stating that Kevin's total distance is 17 miles -/
theorem kevin_ran_17_miles :
  kevin_total_distance 10 20 8 0.5 0.5 0.25 = 17 := by
  sorry

#eval kevin_total_distance 10 20 8 0.5 0.5 0.25

end NUMINAMATH_CALUDE_kevin_ran_17_miles_l3954_395466


namespace NUMINAMATH_CALUDE_ab_bc_ratio_l3954_395468

/-- A rectangle divided into five congruent rectangles -/
structure DividedRectangle where
  -- The width of each congruent rectangle
  x : ℝ
  -- Assumption that x is positive
  x_pos : x > 0

/-- The length of side AB in the divided rectangle -/
def length_AB (r : DividedRectangle) : ℝ := 5 * r.x

/-- The length of side BC in the divided rectangle -/
def length_BC (r : DividedRectangle) : ℝ := 3 * r.x

/-- Theorem stating that the ratio of AB to BC is 5:3 -/
theorem ab_bc_ratio (r : DividedRectangle) :
  length_AB r / length_BC r = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_bc_ratio_l3954_395468


namespace NUMINAMATH_CALUDE_stratified_sampling_total_students_l3954_395447

theorem stratified_sampling_total_students 
  (total_sample : ℕ) 
  (grade_10_sample : ℕ) 
  (grade_11_sample : ℕ) 
  (grade_12_students : ℕ) 
  (h1 : total_sample = 100)
  (h2 : grade_10_sample = 24)
  (h3 : grade_11_sample = 26)
  (h4 : grade_12_students = 600)
  (h5 : grade_12_students * total_sample = 
        (total_sample - grade_10_sample - grade_11_sample) * total_students) : 
  total_students = 1200 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_total_students_l3954_395447


namespace NUMINAMATH_CALUDE_expression_evaluation_l3954_395497

theorem expression_evaluation : 2 + 3 * 4 - 1^2 + 6 / 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3954_395497


namespace NUMINAMATH_CALUDE_reflection_theorem_l3954_395484

/-- Original function -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- Reflection line -/
def reflection_line : ℝ := -2

/-- Resulting function after reflection -/
def g (x : ℝ) : ℝ := 2 * x + 9

/-- Theorem stating that g is the reflection of f across x = -2 -/
theorem reflection_theorem :
  ∀ x : ℝ, g (2 * reflection_line - x) = f x :=
sorry

end NUMINAMATH_CALUDE_reflection_theorem_l3954_395484


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3954_395406

theorem max_value_sqrt_sum (a b : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3954_395406


namespace NUMINAMATH_CALUDE_train_length_is_500_l3954_395464

/-- The length of a train that passes a pole in 50 seconds and a 500 m long platform in 100 seconds -/
def train_length : ℝ := by sorry

/-- The time it takes for the train to pass a pole -/
def pole_passing_time : ℝ := 50

/-- The time it takes for the train to pass a platform -/
def platform_passing_time : ℝ := 100

/-- The length of the platform -/
def platform_length : ℝ := 500

theorem train_length_is_500 :
  train_length = 500 :=
by
  have h1 : train_length / pole_passing_time = (train_length + platform_length) / platform_passing_time :=
    by sorry
  sorry

#check train_length_is_500

end NUMINAMATH_CALUDE_train_length_is_500_l3954_395464


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3954_395414

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3954_395414


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l3954_395427

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares : a^2 + b^2 + c^2 = 4) : 
  a^4 + b^4 + c^4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l3954_395427


namespace NUMINAMATH_CALUDE_alcohol_mixture_concentration_l3954_395434

/-- Proves that the new concentration of the mixture is 29% given the initial conditions --/
theorem alcohol_mixture_concentration
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 25)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percentage = 40)
  (h5 : total_liquid = 8)
  (h6 : final_vessel_capacity = 10) :
  (vessel1_capacity * vessel1_alcohol_percentage / 100 +
   vessel2_capacity * vessel2_alcohol_percentage / 100) /
  final_vessel_capacity * 100 = 29 := by
  sorry


end NUMINAMATH_CALUDE_alcohol_mixture_concentration_l3954_395434


namespace NUMINAMATH_CALUDE_conor_weekly_vegetables_l3954_395433

def eggplants : ℕ := 12
def carrots : ℕ := 9
def potatoes : ℕ := 8
def onions : ℕ := 15
def zucchinis : ℕ := 7
def work_days : ℕ := 6

def vegetables_per_day : ℕ := eggplants + carrots + potatoes + onions + zucchinis

theorem conor_weekly_vegetables :
  vegetables_per_day * work_days = 306 := by sorry

end NUMINAMATH_CALUDE_conor_weekly_vegetables_l3954_395433


namespace NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l3954_395459

/-- Given a rectangle with perimeter 60 and length 3 times its width, 
    and a triangle with height 36, if their areas are equal, 
    then the base of the triangle (which is also one side of the rectangle) is 9.375. -/
theorem rectangle_triangle_equal_area (w : ℝ) (x : ℝ) : 
  (2 * (w + 3*w) = 60) →  -- Rectangle perimeter is 60
  (w * (3*w) = (1/2) * 36 * x) →  -- Rectangle and triangle have equal area
  x = 9.375 := by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l3954_395459


namespace NUMINAMATH_CALUDE_find_genuine_coin_l3954_395403

/-- Represents a coin, which can be either genuine or counterfeit -/
inductive Coin
| genuine : Coin
| counterfeit : ℕ → Coin

/-- Represents the result of weighing two coins -/
inductive WeighingResult
| equal : WeighingResult
| unequal : WeighingResult

/-- Represents a collection of coins -/
def CoinSet := List Coin

/-- Represents a weighing action -/
def Weighing := Coin → Coin → WeighingResult

/-- Represents a strategy to find a genuine coin -/
def Strategy := CoinSet → Weighing → Option Coin

theorem find_genuine_coin 
  (coins : CoinSet) 
  (h_total : coins.length = 9)
  (h_counterfeit : (coins.filter (λ c => match c with 
    | Coin.counterfeit _ => true 
    | _ => false)).length = 4)
  (h_genuine_equal : ∀ c1 c2, c1 = Coin.genuine ∧ c2 = Coin.genuine → 
    (λ _ _ => WeighingResult.equal) c1 c2 = WeighingResult.equal)
  (h_counterfeit_differ : ∀ c1 c2, c1 ≠ c2 → 
    (c1 = Coin.genuine ∨ (∃ n, c1 = Coin.counterfeit n)) ∧ 
    (c2 = Coin.genuine ∨ (∃ m, c2 = Coin.counterfeit m)) → 
    (λ _ _ => WeighingResult.unequal) c1 c2 = WeighingResult.unequal)
  : ∃ (s : Strategy), ∀ w : Weighing, 
    (∃ c, s coins w = some c ∧ c = Coin.genuine) ∧ 
    (s coins w).isSome → (Nat.card {p : Coin × Coin | w p.1 p.2 ≠ WeighingResult.equal}) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_find_genuine_coin_l3954_395403


namespace NUMINAMATH_CALUDE_flour_for_cookies_l3954_395480

/-- Given a recipe where 24 cookies require 1.5 cups of flour,
    calculate the amount of flour needed for 72 cookies. -/
theorem flour_for_cookies (original_cookies : ℕ) (original_flour : ℚ) (new_cookies : ℕ) :
  original_cookies = 24 →
  original_flour = 3/2 →
  new_cookies = 72 →
  (original_flour / original_cookies) * new_cookies = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_flour_for_cookies_l3954_395480


namespace NUMINAMATH_CALUDE_non_similar_1200_pointed_stars_l3954_395417

/-- Definition of a regular n-pointed star (placeholder) -/
def RegularStar (n : ℕ) : Type := sorry

/-- Counts the number of non-similar regular n-pointed stars -/
def countNonSimilarStars (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

theorem non_similar_1200_pointed_stars :
  countNonSimilarStars 1200 = 160 :=
by sorry

end NUMINAMATH_CALUDE_non_similar_1200_pointed_stars_l3954_395417


namespace NUMINAMATH_CALUDE_units_digit_of_sum_factorial_and_square_10_l3954_395402

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def sum_factorial_and_square (n : ℕ) : ℕ := 
  (List.range n).foldl (λ acc i => acc + factorial (i + 1) + (i + 1)^2) 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_factorial_and_square_10 : 
  units_digit (sum_factorial_and_square 10) = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_factorial_and_square_10_l3954_395402


namespace NUMINAMATH_CALUDE_tan_two_implies_expression_eq_neg_two_l3954_395407

theorem tan_two_implies_expression_eq_neg_two (θ : Real) (h : Real.tan θ = 2) :
  (2 * Real.cos θ) / (Real.sin (π / 2 + θ) + Real.sin (π + θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_expression_eq_neg_two_l3954_395407


namespace NUMINAMATH_CALUDE_jellybean_mass_theorem_l3954_395450

/-- The price of jellybeans in cents per gram -/
def price_per_gram : ℚ := 750 / 250

/-- The mass of jellybeans in grams that can be bought for 180 cents -/
def mass_for_180_cents : ℚ := 180 / price_per_gram

theorem jellybean_mass_theorem :
  mass_for_180_cents = 60 := by sorry

end NUMINAMATH_CALUDE_jellybean_mass_theorem_l3954_395450


namespace NUMINAMATH_CALUDE_hot_dog_bun_packages_l3954_395475

theorem hot_dog_bun_packages : ∃ n : ℕ, n > 0 ∧ 12 * n % 9 = 0 ∧ ∀ m : ℕ, m > 0 → 12 * m % 9 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_bun_packages_l3954_395475


namespace NUMINAMATH_CALUDE_race_length_for_simultaneous_finish_l3954_395460

theorem race_length_for_simultaneous_finish 
  (speed_ratio : ℝ) 
  (head_start : ℝ) 
  (race_length : ℝ) : 
  speed_ratio = 4 →
  head_start = 63 →
  race_length / speed_ratio = (race_length - head_start) / 1 →
  race_length = 84 := by
sorry

end NUMINAMATH_CALUDE_race_length_for_simultaneous_finish_l3954_395460


namespace NUMINAMATH_CALUDE_root_implies_difference_l3954_395495

theorem root_implies_difference (a b : ℝ) :
  (∃ x, x^2 + 4*a^2*b^2*x = 4 ∧ x = (a^2 - b^2)^2) →
  (b^4 - a^4 = 2 ∨ b^4 - a^4 = -2) :=
by sorry

end NUMINAMATH_CALUDE_root_implies_difference_l3954_395495


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3954_395420

theorem inequality_system_solutions (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), x₁ > 4 ∧ x₁ ≤ a ∧ 
                      x₂ > 4 ∧ x₂ ≤ a ∧ 
                      x₃ > 4 ∧ x₃ ≤ a ∧ 
                      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                      (∀ (y : ℤ), y > 4 ∧ y ≤ a → y = x₁ ∨ y = x₂ ∨ y = x₃)) →
  7 ≤ a ∧ a < 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3954_395420


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3954_395456

theorem polynomial_division_remainder :
  ∃ (q r : Polynomial ℚ),
    3 * X^4 + 14 * X^3 - 50 * X^2 - 72 * X + 55 = (X^2 + 8 * X - 4) * q + r ∧
    r = 224 * X - 113 ∧
    r.degree < (X^2 + 8 * X - 4).degree :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3954_395456


namespace NUMINAMATH_CALUDE_single_intersection_l3954_395409

def f (a x : ℝ) : ℝ := (a - 1) * x^2 - 4 * x + 2 * a

theorem single_intersection (a : ℝ) : 
  (∃! x, f a x = 0) ↔ (a = -1 ∨ a = 2 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_single_intersection_l3954_395409


namespace NUMINAMATH_CALUDE_rainfall_third_week_l3954_395453

theorem rainfall_third_week (total : ℝ) (week1 : ℝ) (week2 : ℝ) (week3 : ℝ)
  (h_total : total = 45)
  (h_week2 : week2 = 1.5 * week1)
  (h_week3 : week3 = 2 * week2)
  (h_sum : week1 + week2 + week3 = total) :
  week3 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_third_week_l3954_395453


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l3954_395436

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  h_pop_size : population_size > 0
  h_sample_size : sample_size > 0
  h_sample_le_pop : sample_size ≤ population_size

/-- The probability of an individual being selected in systematic sampling -/
def selection_probability (s : SystematicSampling) : ℚ :=
  s.sample_size / s.population_size

/-- Theorem stating the probability of selection in the given scenario -/
theorem systematic_sampling_probability 
  (s : SystematicSampling) 
  (h_pop : s.population_size = 42) 
  (h_sample : s.sample_size = 10) : 
  selection_probability s = 5 / 21 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l3954_395436


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3954_395487

theorem rationalize_denominator : 
  (35 - Real.sqrt 35) / Real.sqrt 35 = Real.sqrt 35 - 1 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3954_395487


namespace NUMINAMATH_CALUDE_hula_hoop_ratio_l3954_395452

def nancy_time : ℕ := 10
def casey_time : ℕ := nancy_time - 3
def morgan_time : ℕ := 21

theorem hula_hoop_ratio : 
  ∃ (k : ℕ), k > 0 ∧ morgan_time = k * casey_time ∧ morgan_time / casey_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_hula_hoop_ratio_l3954_395452


namespace NUMINAMATH_CALUDE_jimmy_payment_l3954_395471

/-- Represents the cost of a pizza in dollars -/
def pizza_cost : ℕ := 12

/-- Represents the delivery charge in dollars for distances over 1 km -/
def delivery_charge : ℕ := 2

/-- Represents the distance threshold in meters for applying delivery charge -/
def distance_threshold : ℕ := 1000

/-- Represents the number of pizzas delivered to the park -/
def park_pizzas : ℕ := 3

/-- Represents the distance to the park in meters -/
def park_distance : ℕ := 100

/-- Represents the number of pizzas delivered to the building -/
def building_pizzas : ℕ := 2

/-- Represents the distance to the building in meters -/
def building_distance : ℕ := 2000

/-- Calculates the total amount Jimmy got paid for the pizzas -/
def total_amount : ℕ :=
  (park_pizzas + building_pizzas) * pizza_cost +
  (if building_distance > distance_threshold then building_pizzas * delivery_charge else 0)

theorem jimmy_payment : total_amount = 64 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_payment_l3954_395471


namespace NUMINAMATH_CALUDE_rose_price_calculation_l3954_395431

/-- Calculates the price per rose given the initial number of roses, 
    remaining roses, and total earnings -/
def price_per_rose (initial_roses : ℕ) (remaining_roses : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (initial_roses - remaining_roses)

theorem rose_price_calculation (initial_roses remaining_roses total_earnings : ℕ) 
  (h1 : initial_roses = 9)
  (h2 : remaining_roses = 4)
  (h3 : total_earnings = 35) :
  price_per_rose initial_roses remaining_roses total_earnings = 7 := by
  sorry

end NUMINAMATH_CALUDE_rose_price_calculation_l3954_395431


namespace NUMINAMATH_CALUDE_male_students_count_l3954_395477

theorem male_students_count (total : ℕ) (difference : ℕ) (male : ℕ) (female : ℕ) : 
  total = 1443 →
  difference = 141 →
  male = female + difference →
  total = male + female →
  male = 792 := by
sorry

end NUMINAMATH_CALUDE_male_students_count_l3954_395477


namespace NUMINAMATH_CALUDE_eighth_fibonacci_is_21_l3954_395488

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem eighth_fibonacci_is_21 : fibonacci 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eighth_fibonacci_is_21_l3954_395488


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3954_395489

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - 6 * x
  (f 0 = 0 ∧ f (3/2) = 0) ∧
  ∀ x : ℝ, f x = 0 → (x = 0 ∨ x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3954_395489


namespace NUMINAMATH_CALUDE_special_function_at_eight_l3954_395411

/-- A monotonic function on (0, +∞) satisfying certain conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧ 
  (∀ x, x > 0 → f x > -4/x) ∧
  (∀ x, x > 0 → f (f x + 4/x) = 3)

/-- The main theorem stating that f(8) = 7/2 for a SpecialFunction -/
theorem special_function_at_eight (f : ℝ → ℝ) (h : SpecialFunction f) : f 8 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_eight_l3954_395411


namespace NUMINAMATH_CALUDE_circle_area_in_square_l3954_395410

theorem circle_area_in_square (square_area : ℝ) (h : square_area = 400) :
  let square_side := Real.sqrt square_area
  let circle_radius := square_side / 2
  let circle_area := Real.pi * circle_radius ^ 2
  circle_area = 100 * Real.pi := by sorry

end NUMINAMATH_CALUDE_circle_area_in_square_l3954_395410


namespace NUMINAMATH_CALUDE_vector_problem_l3954_395412

/-- Given a vector a and a unit vector b not parallel to the x-axis such that a · b = √3, prove that b = (1/2, √3/2) -/
theorem vector_problem (a b : ℝ × ℝ) : 
  a = (Real.sqrt 3, 1) →
  ‖b‖ = 1 →
  b.1 ≠ b.2 →
  a.1 * b.1 + a.2 * b.2 = Real.sqrt 3 →
  b = (1/2, Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l3954_395412


namespace NUMINAMATH_CALUDE_sine_function_property_l3954_395408

/-- Given a function f(x) = sin(ωx) where ω > 0, if f(x - 1/2) = f(x + 1/2) for all real x,
    and f(-1/4) = a, then f(9/4) = -a -/
theorem sine_function_property (ω : ℝ) (a : ℝ) (h_ω : ω > 0) :
  (∀ x : ℝ, Real.sin (ω * (x - 1/2)) = Real.sin (ω * (x + 1/2))) →
  Real.sin (ω * (-1/4)) = a →
  Real.sin (ω * (9/4)) = -a :=
by sorry

end NUMINAMATH_CALUDE_sine_function_property_l3954_395408


namespace NUMINAMATH_CALUDE_number_of_students_l3954_395469

theorem number_of_students (initial_avg : ℝ) (wrong_mark : ℝ) (correct_mark : ℝ) (correct_avg : ℝ) :
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_mark = 10 →
  correct_avg = 95 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * correct_avg = n * initial_avg - (wrong_mark - correct_mark) ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_l3954_395469


namespace NUMINAMATH_CALUDE_connie_marbles_l3954_395441

/-- Calculates the number of marbles Connie has after giving some away -/
def marblesRemaining (initial : ℕ) (givenAway : ℕ) : ℕ :=
  initial - givenAway

/-- Proves that Connie has 3 marbles remaining after giving away 70 from her initial 73 -/
theorem connie_marbles : marblesRemaining 73 70 = 3 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l3954_395441


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3954_395467

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (l : Line) 
  (p : Point) 
  (given_line : Line) :
  l.a = 1 ∧ l.b = 3 ∧ l.c = -2 →
  p.x = -1 ∧ p.y = 1 →
  given_line.a = 1 ∧ given_line.b = 3 ∧ given_line.c = 4 →
  p.liesOn l ∧ l.isParallelTo given_line :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3954_395467


namespace NUMINAMATH_CALUDE_integer_solutions_of_manhattan_distance_equation_l3954_395485

def solution_set : Set (ℤ × ℤ) := {(2,2), (2,0), (3,1), (1,1)}

theorem integer_solutions_of_manhattan_distance_equation :
  {(x, y) : ℤ × ℤ | |x - 2| + |y - 1| = 1} = solution_set := by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_manhattan_distance_equation_l3954_395485


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3954_395415

theorem inequality_system_solution (x : ℝ) :
  (1 - x > 0) ∧ ((x + 2) / 3 - 1 ≤ x) → -1/2 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3954_395415


namespace NUMINAMATH_CALUDE_sally_has_hundred_l3954_395470

/-- Sally's current amount of money -/
def sally_money : ℕ := sorry

/-- The condition that if Sally had $20 less, she would have $80 -/
axiom sally_condition : sally_money - 20 = 80

/-- Theorem: Sally has $100 -/
theorem sally_has_hundred : sally_money = 100 := by sorry

end NUMINAMATH_CALUDE_sally_has_hundred_l3954_395470


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l3954_395421

/-- Given a cylinder with volume 72π cm³ and height twice its radius,
    prove that a cone with the same radius and height has a volume of 144π cm³. -/
theorem cone_volume_from_cylinder_volume (r h : ℝ) : 
  (π * r^2 * h = 72 * π) → 
  (h = 2 * r) → 
  ((1/3) * π * r^2 * h = 144 * π) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_volume_l3954_395421


namespace NUMINAMATH_CALUDE_circle_center_transformation_l3954_395455

/-- Reflect a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Translate a point vertically -/
def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

/-- The transformation described in the problem -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_up (reflect_y p) 12

theorem circle_center_transformation :
  transform (3, -4) = (-3, 8) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l3954_395455


namespace NUMINAMATH_CALUDE_C_power_50_l3954_395481

def C : Matrix (Fin 2) (Fin 2) ℤ := !![2, 1; -4, -1]

theorem C_power_50 : C^50 = !![4^49 + 1, 4^49; -4^50, -2 * 4^49 + 1] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l3954_395481


namespace NUMINAMATH_CALUDE_range_of_m_l3954_395448

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : ∀ x y, x > 0 → y > 0 → (2 * y / x + 8 * x / y ≥ m^2 + 2*m)) : 
  m ∈ Set.Icc (-4 : ℝ) 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3954_395448


namespace NUMINAMATH_CALUDE_sum_abs_bound_l3954_395437

theorem sum_abs_bound (x y z : ℝ) 
  (eq1 : x^2 + y^2 + z = 15)
  (eq2 : x + y + z^2 = 27)
  (eq3 : x*y + y*z + z*x = 7) :
  7 ≤ |x + y + z| ∧ |x + y + z| ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_abs_bound_l3954_395437


namespace NUMINAMATH_CALUDE_valid_arrangement_iff_odd_l3954_395465

/-- A permutation of numbers from 1 to n -/
def OuterRingPermutation (n : ℕ) := Fin n → Fin n

/-- Checks if a permutation satisfies the rotation property -/
def SatisfiesRotationProperty (n : ℕ) (p : OuterRingPermutation n) : Prop :=
  ∀ k : Fin n, ∃! j : Fin n, (p j - j : ℤ) ≡ k [ZMOD n]

/-- The main theorem: a valid arrangement exists if and only if n is odd -/
theorem valid_arrangement_iff_odd (n : ℕ) (h : n ≥ 3) :
  (∃ p : OuterRingPermutation n, SatisfiesRotationProperty n p) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_iff_odd_l3954_395465


namespace NUMINAMATH_CALUDE_bowling_team_score_l3954_395405

/-- Represents the scores of a bowling team with three members -/
structure BowlingTeam where
  first_bowler : ℕ
  second_bowler : ℕ
  third_bowler : ℕ

/-- Calculates the total score of a bowling team -/
def total_score (team : BowlingTeam) : ℕ :=
  team.first_bowler + team.second_bowler + team.third_bowler

/-- Theorem stating the total score of the bowling team under given conditions -/
theorem bowling_team_score :
  ∃ (team : BowlingTeam),
    team.third_bowler = 162 ∧
    team.second_bowler = 3 * team.third_bowler ∧
    team.first_bowler = team.second_bowler / 3 ∧
    total_score team = 810 := by
  sorry


end NUMINAMATH_CALUDE_bowling_team_score_l3954_395405


namespace NUMINAMATH_CALUDE_parabola_b_value_l3954_395439

/-- A parabola passing through two given points has a specific 'b' value -/
theorem parabola_b_value (b c : ℝ) : 
  ((-1)^2 + b*(-1) + c = -8) → 
  (2^2 + b*2 + c = 10) → 
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3954_395439


namespace NUMINAMATH_CALUDE_cafeteria_apples_l3954_395493

theorem cafeteria_apples (initial : ℕ) : 
  initial - 20 + 28 = 46 → initial = 38 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l3954_395493


namespace NUMINAMATH_CALUDE_temperature_difference_l3954_395462

/-- The temperature difference problem -/
theorem temperature_difference (T_NY T_M T_SD : ℝ) : 
  T_NY = 80 →
  T_M = T_NY + 10 →
  (T_NY + T_M + T_SD) / 3 = 95 →
  T_SD - T_M = 25 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3954_395462


namespace NUMINAMATH_CALUDE_john_earnings_calculation_l3954_395472

/-- Calculates John's weekly earnings after fees and taxes --/
def johnWeeklyEarnings : ℝ :=
  let streamingHours : ℕ := 4
  let mondayRate : ℝ := 10
  let wednesdayRate : ℝ := 12
  let fridayRate : ℝ := 15
  let saturdayRate : ℝ := 20
  let platformFeeRate : ℝ := 0.20
  let taxRate : ℝ := 0.25

  let grossEarnings : ℝ := streamingHours * (mondayRate + wednesdayRate + fridayRate + saturdayRate)
  let platformFee : ℝ := grossEarnings * platformFeeRate
  let netEarningsBeforeTax : ℝ := grossEarnings - platformFee
  let tax : ℝ := netEarningsBeforeTax * taxRate
  netEarningsBeforeTax - tax

theorem john_earnings_calculation :
  johnWeeklyEarnings = 136.80 := by sorry

end NUMINAMATH_CALUDE_john_earnings_calculation_l3954_395472


namespace NUMINAMATH_CALUDE_room_width_calculation_l3954_395401

theorem room_width_calculation (room_length : ℝ) (carpet_width : ℝ) (carpet_cost_per_sqm : ℝ) (total_cost : ℝ) :
  room_length = 13 →
  carpet_width = 0.75 →
  carpet_cost_per_sqm = 12 →
  total_cost = 1872 →
  room_length * (total_cost / (room_length * carpet_cost_per_sqm)) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l3954_395401


namespace NUMINAMATH_CALUDE_equilateral_triangle_roots_l3954_395419

/-- Given complex roots z₁ and z₂ of z² + az + b = 0 where a and b are complex,
    and z₂ = ω z₁ with ω = e^(2πi/3), prove that a²/b = 1 -/
theorem equilateral_triangle_roots (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 →
  z₂^2 + a*z₂ + b = 0 →
  z₂ = (Complex.exp (2 * Complex.I * Real.pi / 3)) * z₁ →
  a^2 / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_roots_l3954_395419


namespace NUMINAMATH_CALUDE_max_value_theorem_l3954_395429

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x^2 - x*y + y^2 = 15) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a^2 - a*b + b^2 = 15 → 
    2 * x^2 + x*y + y^2 ≥ 2 * a^2 + a*b + b^2) → 
  2 * x^2 + x*y + y^2 = (75 + 60 * Real.sqrt 2) / 7 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3954_395429


namespace NUMINAMATH_CALUDE_division_multiplication_result_l3954_395438

theorem division_multiplication_result : 
  let x : ℝ := 5.5
  let y : ℝ := (x / 6) * 12
  y = 11 := by sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l3954_395438


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3954_395432

/-- Given a line y = kx (k > 0) intersecting a circle (x-2)^2 + y^2 = 1 at two points A and B,
    where the distance AB = (2/5)√5, prove that k = 1/2 -/
theorem line_circle_intersection (k : ℝ) (h_k_pos : k > 0) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 - 2)^2 + (k * A.1)^2 = 1 ∧ 
    (B.1 - 2)^2 + (k * B.1)^2 = 1 ∧ 
    (A.1 - B.1)^2 + (k * A.1 - k * B.1)^2 = (2/5)^2 * 5) → 
  k = 1/2 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3954_395432


namespace NUMINAMATH_CALUDE_remaining_integers_l3954_395458

theorem remaining_integers (T : Finset ℕ) : 
  T = Finset.range 100 → 
  (T.filter (λ x => x % 4 ≠ 0 ∧ x % 5 ≠ 0)).card = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_integers_l3954_395458


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3954_395498

noncomputable def trapezoidArea (h α β : ℝ) : ℝ :=
  2 * h^2 * (Real.tan β + Real.tan α)

theorem isosceles_trapezoid_area
  (h α β : ℝ)
  (h_pos : h > 0)
  (α_pos : α > 0)
  (β_pos : β > 0)
  (α_lt_90 : α < π / 2)
  (β_lt_90 : β < π / 2)
  (h_eq : h = 2)
  (α_eq : α = 15 * π / 180)
  (β_eq : β = 75 * π / 180) :
  trapezoidArea h α β = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3954_395498


namespace NUMINAMATH_CALUDE_kolya_always_wins_l3954_395430

/-- Represents a player's move in the game -/
inductive Move
| ChangeA (delta : Int) : Move
| ChangeB (delta : Int) : Move

/-- Represents the state of the game -/
structure GameState where
  a : Int
  b : Int

/-- Defines a valid move for Petya -/
def validPetyaMove (m : Move) : Prop :=
  match m with
  | Move.ChangeA delta => delta = 1 ∨ delta = -1
  | Move.ChangeB delta => delta = 1 ∨ delta = -1

/-- Defines a valid move for Kolya -/
def validKolyaMove (m : Move) : Prop :=
  match m with
  | Move.ChangeA delta => delta = 1 ∨ delta = -1 ∨ delta = 3 ∨ delta = -3
  | Move.ChangeB delta => delta = 1 ∨ delta = -1 ∨ delta = 3 ∨ delta = -3

/-- Applies a move to the game state -/
def applyMove (state : GameState) (m : Move) : GameState :=
  match m with
  | Move.ChangeA delta => { state with a := state.a + delta }
  | Move.ChangeB delta => { state with b := state.b + delta }

/-- Checks if the polynomial has integer roots -/
def hasIntegerRoots (state : GameState) : Prop :=
  ∃ x y : Int, x^2 + state.a * x + state.b = 0 ∧ y^2 + state.a * y + state.b = 0 ∧ x ≠ y

/-- Theorem stating Kolya can always win -/
theorem kolya_always_wins :
  ∀ (initial : GameState),
  ∃ (kolyaMoves : List Move),
    (∀ m ∈ kolyaMoves, validKolyaMove m) ∧
    (∀ (petyaMoves : List Move),
      (petyaMoves.length = kolyaMoves.length) →
      (∀ m ∈ petyaMoves, validPetyaMove m) →
      ∃ (finalState : GameState),
        finalState = (kolyaMoves.zip petyaMoves).foldl
          (λ state (km, pm) => applyMove (applyMove state pm) km)
          initial ∧
        hasIntegerRoots finalState) :=
sorry

end NUMINAMATH_CALUDE_kolya_always_wins_l3954_395430


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3954_395491

theorem product_of_three_numbers (a b c m : ℝ) : 
  a + b + c = 180 ∧
  5 * a = m ∧
  b = m + 12 ∧
  c = m - 6 →
  a * b * c = 42184 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3954_395491


namespace NUMINAMATH_CALUDE_expression_simplification_l3954_395486

theorem expression_simplification (a : ℝ) (h : a^2 - a - 2 = 0) :
  (1 + 1/a) / ((a^2 - 1)/a) - (2*a - 2)/(a^2 - 2*a + 1) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3954_395486


namespace NUMINAMATH_CALUDE_unique_solution_l3954_395442

def product_of_digits (n : ℕ) : ℕ := sorry

theorem unique_solution : ∃! x : ℕ+, 
  (x : ℕ) > 0 ∧ product_of_digits x = x^2 - 10*x - 22 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3954_395442


namespace NUMINAMATH_CALUDE_student_height_survey_is_comprehensive_l3954_395496

/-- Represents a survey --/
structure Survey where
  population : ℕ
  measurementType : Type
  isFeasible : Bool

/-- Defines the conditions for a comprehensive survey --/
def isComprehensiveSurvey (s : Survey) : Prop :=
  s.population ≤ 100 ∧ s.isFeasible = true

/-- Represents the survey of students' heights in a class --/
def studentHeightSurvey : Survey :=
  { population := 45,
    measurementType := ℝ,
    isFeasible := true }

/-- Theorem stating that the student height survey is a comprehensive survey --/
theorem student_height_survey_is_comprehensive :
  isComprehensiveSurvey studentHeightSurvey :=
by
  sorry


end NUMINAMATH_CALUDE_student_height_survey_is_comprehensive_l3954_395496


namespace NUMINAMATH_CALUDE_pen_price_calculation_l3954_395422

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 450 →
  num_pens = 30 →
  num_pencils = 75 →
  pencil_price = 2 →
  (total_cost - (num_pencils : ℝ) * pencil_price) / (num_pens : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l3954_395422


namespace NUMINAMATH_CALUDE_susan_apples_l3954_395449

/-- The number of apples each person has -/
structure Apples where
  phillip : ℝ
  ben : ℝ
  tom : ℝ
  susan : ℝ

/-- The conditions of the problem -/
def apple_conditions (a : Apples) : Prop :=
  a.phillip = 38.25 ∧
  a.ben = a.phillip + 8.5 ∧
  a.tom = (3/8) * a.ben ∧
  a.susan = (1/2) * a.tom + 7

/-- The theorem stating that under the given conditions, Susan has 15.765625 apples -/
theorem susan_apples (a : Apples) (h : apple_conditions a) : a.susan = 15.765625 := by
  sorry

end NUMINAMATH_CALUDE_susan_apples_l3954_395449


namespace NUMINAMATH_CALUDE_square_ratio_sum_l3954_395454

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 75 / 128 →
  (∃ (side_ratio : ℝ), side_ratio = Real.sqrt (area_ratio) ∧ 
    side_ratio = a * Real.sqrt b / c) →
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_sum_l3954_395454


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_88_875_l3954_395413

/-- Represents the grid and shapes configuration --/
structure GridConfig where
  gridSize : ℕ
  squareSide : ℝ
  smallCircleDiameter : ℝ
  largeCircleDiameter : ℝ
  hexagonSide : ℝ
  smallCircleCount : ℕ

/-- Calculates the coefficients A, B, and C for the shaded area expression --/
def calculateCoefficients (config : GridConfig) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem stating that the sum of coefficients equals 88.875 for the given configuration --/
theorem sum_of_coefficients_equals_88_875 : 
  let config : GridConfig := {
    gridSize := 6,
    squareSide := 1.5,
    smallCircleDiameter := 1.5,
    largeCircleDiameter := 3,
    hexagonSide := 1.5,
    smallCircleCount := 4
  }
  let (A, B, C) := calculateCoefficients config
  A + B + C = 88.875 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_88_875_l3954_395413


namespace NUMINAMATH_CALUDE_sum_of_roots_l3954_395479

theorem sum_of_roots (x : ℝ) : 
  (∃ a b : ℝ, (2*x + 3)*(x - 4) + (2*x + 3)*(x - 6) = 0 ∧ 
   {y : ℝ | (2*y + 3)*(y - 4) + (2*y + 3)*(y - 6) = 0} = {a, b} ∧
   a + b = 7/2) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3954_395479


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l3954_395490

theorem fourth_root_simplification (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (2^7 * 3^3 : ℚ)^(1/4) = a * b^(1/4) → a + b = 218 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l3954_395490


namespace NUMINAMATH_CALUDE_car_rental_cost_per_mile_l3954_395499

/-- Represents a car rental plan with an initial fee and a per-mile cost. -/
structure RentalPlan where
  initialFee : ℝ
  costPerMile : ℝ

/-- The total cost of a rental plan for a given number of miles. -/
def totalCost (plan : RentalPlan) (miles : ℝ) : ℝ :=
  plan.initialFee + plan.costPerMile * miles

theorem car_rental_cost_per_mile :
  let plan1 : RentalPlan := { initialFee := 65, costPerMile := x }
  let plan2 : RentalPlan := { initialFee := 0, costPerMile := 0.60 }
  let miles : ℝ := 325
  totalCost plan1 miles = totalCost plan2 miles →
  x = 0.40 := by
sorry

end NUMINAMATH_CALUDE_car_rental_cost_per_mile_l3954_395499


namespace NUMINAMATH_CALUDE_easel_cost_l3954_395476

def paintbrush_cost : ℚ := 2.4
def paints_cost : ℚ := 9.2
def rose_has : ℚ := 7.1
def rose_needs : ℚ := 11

theorem easel_cost : 
  let total_cost := rose_has + rose_needs
  let other_items_cost := paintbrush_cost + paints_cost
  total_cost - other_items_cost = 6.5 := by sorry

end NUMINAMATH_CALUDE_easel_cost_l3954_395476


namespace NUMINAMATH_CALUDE_clock_angle_at_2_30_l3954_395425

/-- The number of degrees in a circle -/
def circle_degrees : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The angle moved by the hour hand in one hour -/
def hour_hand_degrees_per_hour : ℚ := circle_degrees / clock_hours

/-- The angle moved by the minute hand in one minute -/
def minute_hand_degrees_per_minute : ℚ := circle_degrees / minutes_per_hour

/-- The position of the hour hand at 2:30 -/
def hour_hand_position : ℚ := 2.5 * hour_hand_degrees_per_hour

/-- The position of the minute hand at 2:30 -/
def minute_hand_position : ℚ := 30 * minute_hand_degrees_per_minute

/-- The angle between the hour hand and minute hand at 2:30 -/
def angle_between_hands : ℚ := |minute_hand_position - hour_hand_position|

theorem clock_angle_at_2_30 :
  min angle_between_hands (circle_degrees - angle_between_hands) = 105 :=
sorry

end NUMINAMATH_CALUDE_clock_angle_at_2_30_l3954_395425


namespace NUMINAMATH_CALUDE_sequence_general_term_l3954_395494

theorem sequence_general_term (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) = a n / (1 + 3 * a n)) →
  a 1 = 2 →
  ∀ n : ℕ, n ≥ 1 → a n = 2 / (6 * n - 5) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3954_395494


namespace NUMINAMATH_CALUDE_exists_number_with_properties_l3954_395404

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_number_with_properties : ∃ n : ℕ, 
  2019 ∣ n ∧ 2019 ∣ sum_of_digits n := by sorry

end NUMINAMATH_CALUDE_exists_number_with_properties_l3954_395404


namespace NUMINAMATH_CALUDE_f_of_2_equals_6_l3954_395428

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 - 3

-- Theorem statement
theorem f_of_2_equals_6 : f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_6_l3954_395428


namespace NUMINAMATH_CALUDE_opposite_sides_range_l3954_395400

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Determines if two points are on opposite sides of a line -/
def oppositeSides (p1 p2 : Point2D) (a : ℝ) : Prop :=
  (3 * p1.x - 2 * p1.y + a) * (3 * p2.x - 2 * p2.y + a) < 0

/-- The theorem stating the range of 'a' for which the given points are on opposite sides of the line -/
theorem opposite_sides_range :
  ∀ a : ℝ, 
    oppositeSides (Point2D.mk 3 1) (Point2D.mk (-4) 6) a ↔ -7 < a ∧ a < 24 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l3954_395400


namespace NUMINAMATH_CALUDE_flight_distance_calculation_l3954_395440

/-- Calculates the total flight distance with headwinds and tailwinds -/
def total_flight_distance (spain_russia : ℝ) (spain_germany : ℝ) (germany_france : ℝ) (france_russia : ℝ) 
  (headwind_increase : ℝ) (tailwind_decrease : ℝ) : ℝ :=
  let france_russia_with_headwind := france_russia * (1 + headwind_increase)
  let russia_spain_via_germany := (spain_russia + spain_germany) * (1 - tailwind_decrease)
  france_russia_with_headwind + russia_spain_via_germany

/-- The total flight distance is approximately 14863.98 km -/
theorem flight_distance_calculation :
  let spain_russia : ℝ := 7019
  let spain_germany : ℝ := 1615
  let germany_france : ℝ := 956
  let france_russia : ℝ := 6180
  let headwind_increase : ℝ := 0.05
  let tailwind_decrease : ℝ := 0.03
  abs (total_flight_distance spain_russia spain_germany germany_france france_russia 
    headwind_increase tailwind_decrease - 14863.98) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_flight_distance_calculation_l3954_395440


namespace NUMINAMATH_CALUDE_hash_four_neg_three_l3954_395483

-- Define the # operation
def hash (x y : Int) : Int := x * (y + 2) + 2 * x * y

-- Theorem statement
theorem hash_four_neg_three : hash 4 (-3) = -28 := by
  sorry

end NUMINAMATH_CALUDE_hash_four_neg_three_l3954_395483


namespace NUMINAMATH_CALUDE_stratified_sampling_seniors_l3954_395478

theorem stratified_sampling_seniors (total_students : ℕ) (senior_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 4500)
  (h2 : senior_students = 1500)
  (h3 : sample_size = 300)
  (h4 : senior_students ≤ total_students) :
  (senior_students * sample_size) / total_students = 100 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_seniors_l3954_395478


namespace NUMINAMATH_CALUDE_student_number_problem_l3954_395473

theorem student_number_problem (x : ℝ) : (7 * x - 150 = 130) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3954_395473


namespace NUMINAMATH_CALUDE_shaded_quadrilateral_area_l3954_395443

theorem shaded_quadrilateral_area : 
  let small_square_side : ℝ := 3
  let medium_square_side : ℝ := 5
  let large_square_side : ℝ := 7
  let total_base : ℝ := small_square_side + medium_square_side + large_square_side
  let diagonal_slope : ℝ := large_square_side / total_base
  let small_triangle_height : ℝ := small_square_side * diagonal_slope
  let medium_triangle_height : ℝ := (small_square_side + medium_square_side) * diagonal_slope
  let trapezoid_area : ℝ := (medium_square_side * (small_triangle_height + medium_triangle_height)) / 2
  trapezoid_area = 12.825 := by
  sorry

end NUMINAMATH_CALUDE_shaded_quadrilateral_area_l3954_395443


namespace NUMINAMATH_CALUDE_complex_power_eq_l3954_395444

theorem complex_power_eq (z : ℂ) : 
  (2 * Complex.cos (20 * π / 180) + 2 * Complex.I * Complex.sin (20 * π / 180)) ^ 6 = 
  -32 + 32 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eq_l3954_395444


namespace NUMINAMATH_CALUDE_max_min_m_values_l3954_395426

/-- Given conditions p and q, find the maximum and minimum values of m -/
theorem max_min_m_values (m : ℝ) (h_m_pos : m > 0) : 
  (∀ x : ℝ, |x| ≤ m → -1 ≤ x ∧ x ≤ 4) ∧ 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → |x| ≤ m) → 
  m = 4 := by sorry

end NUMINAMATH_CALUDE_max_min_m_values_l3954_395426


namespace NUMINAMATH_CALUDE_solve_for_y_l3954_395445

theorem solve_for_y : ∃ y : ℝ, (3 * y) / 4 = 15 ∧ y = 20 := by sorry

end NUMINAMATH_CALUDE_solve_for_y_l3954_395445


namespace NUMINAMATH_CALUDE_min_distance_theorem_l3954_395482

theorem min_distance_theorem (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ - a)^2 + (Real.log (x₀^2) - 2*a)^2 ≤ 4/5) →
  a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_theorem_l3954_395482


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3954_395416

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def B (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem intersection_implies_a_value (a : ℝ) :
  A ∩ B a = {3} → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3954_395416


namespace NUMINAMATH_CALUDE_haircut_tip_percentage_l3954_395418

theorem haircut_tip_percentage (womens_haircut_cost : ℝ) (childrens_haircut_cost : ℝ) 
  (num_children : ℕ) (tip_amount : ℝ) :
  womens_haircut_cost = 48 →
  childrens_haircut_cost = 36 →
  num_children = 2 →
  tip_amount = 24 →
  (tip_amount / (womens_haircut_cost + num_children * childrens_haircut_cost)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_haircut_tip_percentage_l3954_395418


namespace NUMINAMATH_CALUDE_norbs_age_l3954_395461

def guesses : List Nat := [24, 28, 30, 32, 36, 38, 41, 44, 47, 49]

def is_prime (n : Nat) : Prop := Nat.Prime n

def at_least_half_too_low (age : Nat) : Prop :=
  (guesses.filter (· < age)).length ≥ guesses.length / 2

def two_off_by_one (age : Nat) : Prop :=
  (guesses.filter (λ g => g = age - 1 ∨ g = age + 1)).length = 2

theorem norbs_age :
  ∃! age : Nat,
    age ∈ guesses ∧
    is_prime age ∧
    at_least_half_too_low age ∧
    two_off_by_one age ∧
    age = 37 :=
sorry

end NUMINAMATH_CALUDE_norbs_age_l3954_395461


namespace NUMINAMATH_CALUDE_mollys_age_l3954_395446

/-- Molly's age calculation --/
theorem mollys_age (initial_candles additional_candles : ℕ) :
  initial_candles = 14 → additional_candles = 6 →
  initial_candles + additional_candles = 20 := by
  sorry

end NUMINAMATH_CALUDE_mollys_age_l3954_395446


namespace NUMINAMATH_CALUDE_total_distance_is_6300_l3954_395492

/-- The distance Bomin walked in kilometers -/
def bomin_km : ℝ := 2

/-- The additional distance Bomin walked in meters -/
def bomin_additional_m : ℝ := 600

/-- The distance Yunshik walked in meters -/
def yunshik_m : ℝ := 3700

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- The total distance walked by Bomin and Yunshik in meters -/
def total_distance : ℝ := (bomin_km * km_to_m + bomin_additional_m) + yunshik_m

theorem total_distance_is_6300 : total_distance = 6300 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_6300_l3954_395492
