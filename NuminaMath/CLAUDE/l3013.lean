import Mathlib

namespace NUMINAMATH_CALUDE_digit_405_is_zero_l3013_301386

/-- The decimal representation of 18/47 -/
def decimal_rep : ℚ := 18 / 47

/-- The length of the repeating sequence in the decimal representation of 18/47 -/
def period : ℕ := 93

/-- The position of the target digit within the repeating sequence -/
def target_position : ℕ := 405 % period

/-- The digit at the specified position in the repeating sequence -/
def digit_at_position (n : ℕ) : ℕ := sorry

theorem digit_405_is_zero :
  digit_at_position target_position = 0 :=
sorry

end NUMINAMATH_CALUDE_digit_405_is_zero_l3013_301386


namespace NUMINAMATH_CALUDE_yield_contradiction_l3013_301311

theorem yield_contradiction (x y z : ℝ) : ¬(0.4 * z + 0.2 * x = 1 ∧
                                           0.1 * y - 0.1 * z = -0.5 ∧
                                           0.1 * x + 0.2 * y = 4) := by
  sorry

end NUMINAMATH_CALUDE_yield_contradiction_l3013_301311


namespace NUMINAMATH_CALUDE_factor_expression_l3013_301375

theorem factor_expression (y : ℝ) : y * (y + 3) + 2 * (y + 3) = (y + 2) * (y + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3013_301375


namespace NUMINAMATH_CALUDE_school_survey_probability_l3013_301367

theorem school_survey_probability (total_students : ℕ) (selected_students : ℕ) 
  (eliminated_students : ℕ) (h1 : total_students = 883) (h2 : selected_students = 80) 
  (h3 : eliminated_students = 3) :
  (selected_students : ℚ) / total_students = 80 / 883 := by
  sorry

end NUMINAMATH_CALUDE_school_survey_probability_l3013_301367


namespace NUMINAMATH_CALUDE_road_trip_cost_l3013_301332

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

end NUMINAMATH_CALUDE_road_trip_cost_l3013_301332


namespace NUMINAMATH_CALUDE_circle_equation_l3013_301378

/-- Given a circle with center (2, -3) and radius 4, its equation is (x-2)^2 + (y+3)^2 = 16 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -3)
  let radius : ℝ := 4
  (x - center.1)^2 + (y - center.2)^2 = radius^2 := by sorry

end NUMINAMATH_CALUDE_circle_equation_l3013_301378


namespace NUMINAMATH_CALUDE_tims_pencils_count_l3013_301317

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 2

/-- The total number of pencils after Tim's action -/
def total_pencils : ℕ := 5

/-- The number of pencils Tim placed in the drawer -/
def tims_pencils : ℕ := total_pencils - initial_pencils

theorem tims_pencils_count : tims_pencils = 3 := by
  sorry

end NUMINAMATH_CALUDE_tims_pencils_count_l3013_301317


namespace NUMINAMATH_CALUDE_existence_of_circle_with_parallel_chord_l3013_301316

-- Define the given objects
variable (ℓ : Line) (P₁ P₂ : Point) (Γ : Circle) (O : Point)

-- Define the property of being the center of Γ
def is_center_of (O : Point) (Γ : Circle) : Prop := sorry

-- Define the property of a circle passing through two points
def passes_through (C : Circle) (P₁ P₂ : Point) : Prop := sorry

-- Define the property of a circle having a chord parallel to a line within another circle
def has_parallel_chord (C : Circle) (ℓ : Line) (Γ : Circle) : Prop := sorry

-- Theorem statement
theorem existence_of_circle_with_parallel_chord 
  (h_center : is_center_of O Γ) : 
  ∃ C : Circle, passes_through C P₁ P₂ ∧ has_parallel_chord C ℓ Γ := by sorry

end NUMINAMATH_CALUDE_existence_of_circle_with_parallel_chord_l3013_301316


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l3013_301324

/-- Given a quadratic expression 8k^2 - 12k + 20, prove that when rewritten in the form a(k + b)^2 + r, the value of r/b is -47.33 -/
theorem quadratic_rewrite_ratio : 
  ∃ (a b r : ℝ), 
    (∀ k, 8 * k^2 - 12 * k + 20 = a * (k + b)^2 + r) ∧ 
    (r / b = -47.33) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l3013_301324


namespace NUMINAMATH_CALUDE_factorization_equality_l3013_301394

theorem factorization_equality (a b : ℝ) : 3*a - 9*a*b = 3*a*(1 - 3*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3013_301394


namespace NUMINAMATH_CALUDE_larger_jar_initial_fill_fraction_l3013_301388

/-- Proves that under the given conditions, the larger jar was initially 1/3 full -/
theorem larger_jar_initial_fill_fraction 
  (small_capacity large_capacity : ℝ) 
  (water_amount : ℝ) 
  (h1 : small_capacity > 0)
  (h2 : large_capacity > 0)
  (h3 : water_amount > 0)
  (h4 : water_amount = 1/3 * small_capacity)
  (h5 : water_amount < large_capacity)
  (h6 : water_amount + water_amount = 2/3 * large_capacity) :
  water_amount = 1/3 * large_capacity := by
sorry

end NUMINAMATH_CALUDE_larger_jar_initial_fill_fraction_l3013_301388


namespace NUMINAMATH_CALUDE_sequence_transformation_l3013_301359

/-- Represents a sequence of letters 'A' and 'B' -/
def Sequence := List Char

/-- An operation that can be performed on a sequence -/
inductive Operation
| Insert (c : Char) (pos : Nat) (count : Nat)
| Remove (pos : Nat) (count : Nat)

/-- Applies an operation to a sequence -/
def applyOperation (s : Sequence) (op : Operation) : Sequence :=
  match op with
  | Operation.Insert c pos count => sorry
  | Operation.Remove pos count => sorry

/-- Checks if a sequence contains only 'A' and 'B' -/
def isValidSequence (s : Sequence) : Prop :=
  s.all (fun c => c = 'A' ∨ c = 'B')

/-- Theorem: Any two valid sequences of length 100 can be transformed
    into each other using at most 100 operations -/
theorem sequence_transformation
  (s1 s2 : Sequence)
  (h1 : s1.length = 100)
  (h2 : s2.length = 100)
  (v1 : isValidSequence s1)
  (v2 : isValidSequence s2) :
  ∃ (ops : List Operation),
    ops.length ≤ 100 ∧
    (ops.foldl applyOperation s1 = s2) :=
  sorry

end NUMINAMATH_CALUDE_sequence_transformation_l3013_301359


namespace NUMINAMATH_CALUDE_gym_income_calculation_l3013_301318

/-- Calculates the monthly income of a gym given its bi-monthly charge and number of members. -/
def gym_monthly_income (bi_monthly_charge : ℕ) (num_members : ℕ) : ℕ :=
  2 * bi_monthly_charge * num_members

/-- Proves that a gym charging $18 twice a month with 300 members makes $10,800 per month. -/
theorem gym_income_calculation :
  gym_monthly_income 18 300 = 10800 := by
  sorry

end NUMINAMATH_CALUDE_gym_income_calculation_l3013_301318


namespace NUMINAMATH_CALUDE_sine_of_angle_plus_three_half_pi_l3013_301353

theorem sine_of_angle_plus_three_half_pi (α : Real) :
  (∃ (x y : Real), x = -5 ∧ y = -12 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin (3 * Real.pi / 2 + α) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_angle_plus_three_half_pi_l3013_301353


namespace NUMINAMATH_CALUDE_johns_number_is_eleven_l3013_301334

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_switch (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem johns_number_is_eleven :
  ∃! x : ℕ, is_two_digit x ∧
    82 ≤ digit_switch (5 * x + 13) ∧
    digit_switch (5 * x + 13) ≤ 86 ∧
    x = 11 := by sorry

end NUMINAMATH_CALUDE_johns_number_is_eleven_l3013_301334


namespace NUMINAMATH_CALUDE_complete_square_result_l3013_301350

/-- Given a quadratic equation x^2 + 6x - 3 = 0, prove that when completing the square, 
    the resulting equation (x + a)^2 = b has b = 12 -/
theorem complete_square_result (x : ℝ) : 
  (∃ a b : ℝ, x^2 + 6*x - 3 = 0 ↔ (x + a)^2 = b) → 
  (∃ a : ℝ, (x + a)^2 = 12) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_result_l3013_301350


namespace NUMINAMATH_CALUDE_cylinder_max_volume_l3013_301347

/-- The maximum volume of a cylinder with total surface area 1 is achieved when 
    the radius and height are both equal to 1/√(6π) -/
theorem cylinder_max_volume (r h : ℝ) :
  r > 0 ∧ h > 0 ∧ 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 1 →
  Real.pi * r^2 * h ≤ Real.pi * (1 / Real.sqrt (6 * Real.pi))^2 * (1 / Real.sqrt (6 * Real.pi)) :=
sorry

end NUMINAMATH_CALUDE_cylinder_max_volume_l3013_301347


namespace NUMINAMATH_CALUDE_inequality_region_range_l3013_301300

-- Define the inequality function
def f (m x y : ℝ) : Prop := x - (m^2 - 2*m + 4)*y - 6 > 0

-- Define the theorem
theorem inequality_region_range :
  ∀ m : ℝ, (∀ x y : ℝ, f m x y → (x ≠ -1 ∨ y ≠ -1)) ↔ -1 < m ∧ m < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_region_range_l3013_301300


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l3013_301376

theorem largest_prime_divisor_of_sum_of_squares : 
  (∃ p : Nat, Nat.Prime p ∧ p ∣ (17^2 + 60^2) ∧ ∀ q : Nat, Nat.Prime q → q ∣ (17^2 + 60^2) → q ≤ p) ∧ 
  (37 : Nat).Prime ∧ 
  37 ∣ (17^2 + 60^2) ∧ 
  ∀ q : Nat, Nat.Prime q → q ∣ (17^2 + 60^2) → q ≤ 37 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l3013_301376


namespace NUMINAMATH_CALUDE_problem_statement_l3013_301321

-- Define the function f(x) = ax^2 + 1
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1

-- Define what it means for a function to pass through a point
def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

-- Define parallel relation for lines and planes
def parallel (α β : Set (ℝ × ℝ × ℝ)) : Prop := sorry

theorem problem_statement :
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ¬(passes_through (f a) (-1) 2)) ∧
  (∀ α β m : Set (ℝ × ℝ × ℝ), 
    parallel α β → (parallel m α ↔ parallel m β)) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3013_301321


namespace NUMINAMATH_CALUDE_water_price_problem_l3013_301372

/-- The residential water price problem -/
theorem water_price_problem (last_year_price : ℝ) 
  (h1 : last_year_price > 0)
  (h2 : 30 / (1.2 * last_year_price) - 15 / last_year_price = 5) : 
  1.2 * last_year_price = 6 := by
  sorry

#check water_price_problem

end NUMINAMATH_CALUDE_water_price_problem_l3013_301372


namespace NUMINAMATH_CALUDE_log_expression_equals_five_l3013_301307

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_five :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 + Real.exp (Real.log 3) = 5 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_five_l3013_301307


namespace NUMINAMATH_CALUDE_max_triangle_area_l3013_301383

def parabola (x : ℝ) : ℝ := x^2 - 6*x + 9

theorem max_triangle_area :
  let A : ℝ × ℝ := (0, 9)
  let B : ℝ × ℝ := (6, 9)
  ∀ p q : ℝ,
    1 ≤ p → p ≤ 6 →
    q = parabola p →
    let C : ℝ × ℝ := (p, q)
    let area := abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1)) / 2
    area ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3013_301383


namespace NUMINAMATH_CALUDE_science_book_pages_l3013_301330

/-- Given a history book, novel, and science book, prove that the science book has 600 pages. -/
theorem science_book_pages
  (history_book novel science_book : ℕ) -- Define the books as natural numbers
  (h1 : novel = history_book / 2) -- The novel has half as many pages as the history book
  (h2 : science_book = 4 * novel) -- The science book has 4 times the amount of pages as the novel
  (h3 : history_book = 300) -- The history book has 300 pages
  : science_book = 600 := by
  sorry

end NUMINAMATH_CALUDE_science_book_pages_l3013_301330


namespace NUMINAMATH_CALUDE_hurricane_damage_conversion_l3013_301366

/-- Calculates the equivalent amount in Canadian dollars given an amount in American dollars and the exchange rate. -/
def convert_to_canadian_dollars (american_dollars : ℚ) (exchange_rate : ℚ) : ℚ :=
  american_dollars * exchange_rate

/-- Theorem stating the correct conversion of hurricane damage from American to Canadian dollars. -/
theorem hurricane_damage_conversion :
  let damage_usd : ℚ := 45000000
  let exchange_rate : ℚ := 3/2
  convert_to_canadian_dollars damage_usd exchange_rate = 67500000 := by
  sorry

#check hurricane_damage_conversion

end NUMINAMATH_CALUDE_hurricane_damage_conversion_l3013_301366


namespace NUMINAMATH_CALUDE_bullet_speed_difference_wild_bill_scenario_l3013_301381

/-- The speed difference of a bullet fired from a moving horse -/
theorem bullet_speed_difference (v_horse : ℝ) (v_bullet : ℝ) :
  v_horse > 0 → v_bullet > v_horse →
  (v_bullet + v_horse) - (v_bullet - v_horse) = 2 * v_horse := by
  sorry

/-- Wild Bill's scenario -/
theorem wild_bill_scenario :
  let v_horse : ℝ := 20
  let v_bullet : ℝ := 400
  (v_bullet + v_horse) - (v_bullet - v_horse) = 40 := by
  sorry

end NUMINAMATH_CALUDE_bullet_speed_difference_wild_bill_scenario_l3013_301381


namespace NUMINAMATH_CALUDE_wilson_sledding_l3013_301358

/-- The number of times Wilson sleds down a tall hill -/
def tall_hill_slides : ℕ := 4

/-- The number of small hills -/
def small_hills : ℕ := 3

/-- The total number of times Wilson sled down all hills -/
def total_slides : ℕ := 14

/-- The number of tall hills Wilson sled down -/
def tall_hills : ℕ := 2

theorem wilson_sledding :
  tall_hills * tall_hill_slides + small_hills * (tall_hill_slides / 2) = total_slides :=
by sorry

end NUMINAMATH_CALUDE_wilson_sledding_l3013_301358


namespace NUMINAMATH_CALUDE_third_day_sales_l3013_301368

/-- Represents the sales of a clothing store over three days -/
structure ClothingSales where
  /-- Number of pieces sold on the first day -/
  first_day : ℕ
  /-- Number of pieces sold on the second day -/
  second_day : ℕ
  /-- Number of pieces sold on the third day -/
  third_day : ℕ

/-- Theorem stating the relationship between sales on different days -/
theorem third_day_sales (a : ℕ) (sales : ClothingSales) 
  (h1 : sales.first_day = a)
  (h2 : sales.second_day = sales.first_day + 4)
  (h3 : sales.third_day = 2 * sales.second_day - 7) :
  sales.third_day = 2 * a + 1 := by
  sorry


end NUMINAMATH_CALUDE_third_day_sales_l3013_301368


namespace NUMINAMATH_CALUDE_lego_problem_l3013_301338

theorem lego_problem (simon bruce kent : ℕ) : 
  simon = (bruce * 6) / 5 →  -- Simon has 20% more legos than Bruce
  bruce = kent + 20 →        -- Bruce has 20 more legos than Kent
  simon = 72 →               -- Simon has 72 legos
  kent = 40 :=               -- Kent has 40 legos
by sorry

end NUMINAMATH_CALUDE_lego_problem_l3013_301338


namespace NUMINAMATH_CALUDE_men_on_bus_l3013_301331

theorem men_on_bus (total : ℕ) (women : ℕ) (children : ℕ) 
  (h1 : total = 54)
  (h2 : women = 26)
  (h3 : children = 10) :
  total - women - children = 18 := by
sorry

end NUMINAMATH_CALUDE_men_on_bus_l3013_301331


namespace NUMINAMATH_CALUDE_max_playground_area_l3013_301382

/-- Represents a rectangular playground --/
structure Playground where
  length : ℝ
  width : ℝ

/-- The perimeter of the playground is 400 feet --/
def perimeterConstraint (p : Playground) : Prop :=
  2 * p.length + 2 * p.width = 400

/-- The length of the playground is at least 100 feet --/
def lengthConstraint (p : Playground) : Prop :=
  p.length ≥ 100

/-- The width of the playground is at least 50 feet --/
def widthConstraint (p : Playground) : Prop :=
  p.width ≥ 50

/-- The area of the playground --/
def area (p : Playground) : ℝ :=
  p.length * p.width

/-- The maximum area of the playground satisfying all constraints is 10000 square feet --/
theorem max_playground_area :
  ∃ (p : Playground),
    perimeterConstraint p ∧
    lengthConstraint p ∧
    widthConstraint p ∧
    area p = 10000 ∧
    ∀ (q : Playground),
      perimeterConstraint q →
      lengthConstraint q →
      widthConstraint q →
      area q ≤ area p :=
by
  sorry


end NUMINAMATH_CALUDE_max_playground_area_l3013_301382


namespace NUMINAMATH_CALUDE_square_area_decrease_l3013_301328

theorem square_area_decrease (s : ℝ) (h : s > 0) :
  let initial_area := s^2
  let new_side := s * 0.9
  let new_area := new_side * s
  (initial_area - new_area) / initial_area * 100 = 19 := by
  sorry

end NUMINAMATH_CALUDE_square_area_decrease_l3013_301328


namespace NUMINAMATH_CALUDE_circle_area_l3013_301327

/-- The area of the circle defined by 3x^2 + 3y^2 - 12x + 18y + 27 = 0 is 4π. -/
theorem circle_area (x y : ℝ) : 
  (3 * x^2 + 3 * y^2 - 12 * x + 18 * y + 27 = 0) → 
  (∃ (center_x center_y radius : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = radius^2 ∧ 
    π * radius^2 = 4 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l3013_301327


namespace NUMINAMATH_CALUDE_modulus_of_z_l3013_301346

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z as (1+i)^2
def z : ℂ := (1 + i)^2

-- Theorem stating that the modulus of z is 2
theorem modulus_of_z : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3013_301346


namespace NUMINAMATH_CALUDE_distance_between_lines_l3013_301325

/-- A circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The lengths of the three chords created by the parallel lines -/
  chord1 : ℝ
  chord2 : ℝ
  chord3 : ℝ
  /-- The chords are positive -/
  chord1_pos : chord1 > 0
  chord2_pos : chord2 > 0
  chord3_pos : chord3 > 0
  /-- The radius is positive -/
  r_pos : r > 0
  /-- The distance between lines is positive -/
  d_pos : d > 0
  /-- The chords satisfy Stewart's theorem -/
  stewart_theorem1 : (chord1 / 2) ^ 2 * chord1 + (d / 2) ^ 2 * chord1 = (chord1 / 2) * r ^ 2 + (chord1 / 2) * r ^ 2
  stewart_theorem2 : (chord3 / 2) ^ 2 * chord3 + ((3 * d) / 2) ^ 2 * chord3 = (chord3 / 2) * r ^ 2 + (chord3 / 2) * r ^ 2

/-- The main theorem stating that for the given chord lengths, the distance between lines is 6 -/
theorem distance_between_lines (c : CircleWithParallelLines) 
    (h1 : c.chord1 = 40) (h2 : c.chord2 = 40) (h3 : c.chord3 = 36) : c.d = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_lines_l3013_301325


namespace NUMINAMATH_CALUDE_max_loss_is_9_l3013_301339

/-- Represents the ratio of money for each person --/
structure MoneyRatio :=
  (cara : ℕ)
  (janet : ℕ)
  (jerry : ℕ)
  (linda : ℕ)

/-- Represents the price range for oranges --/
structure PriceRange :=
  (min : ℚ)
  (max : ℚ)

/-- Calculates the maximum loss for Cara and Janet --/
def calculate_max_loss (ratio : MoneyRatio) (total_money : ℚ) (price_range : PriceRange) (sell_percentage : ℚ) : ℚ :=
  sorry

theorem max_loss_is_9 (ratio : MoneyRatio) (total_money : ℚ) (price_range : PriceRange) (sell_percentage : ℚ) :
  ratio.cara = 4 ∧ ratio.janet = 5 ∧ ratio.jerry = 6 ∧ ratio.linda = 7 ∧
  total_money = 110 ∧
  price_range.min = 1/2 ∧ price_range.max = 3/2 ∧
  sell_percentage = 4/5 →
  calculate_max_loss ratio total_money price_range sell_percentage = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_loss_is_9_l3013_301339


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3013_301320

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 6 = 0 ∧ x = 2) → 
  (∃ x : ℝ, x^2 + k*x + 6 = 0 ∧ x = 3 ∧ k = -5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3013_301320


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l3013_301303

theorem sphere_volume_ratio (r R : ℝ) (h : r > 0) (H : R > 0) :
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 4/9 →
  ((4/3) * Real.pi * r^3) / ((4/3) * Real.pi * R^3) = 8/27 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l3013_301303


namespace NUMINAMATH_CALUDE_no_valid_grid_l3013_301323

/-- Represents a 4x4 grid with some initial values -/
structure Grid :=
  (a11 : ℝ) (a12 : ℝ) (a13 : ℝ) (a14 : ℝ)
  (a21 : ℝ) (a22 : ℝ) (a23 : ℝ) (a24 : ℝ)
  (a31 : ℝ) (a32 : ℝ) (a33 : ℝ) (a34 : ℝ)
  (a41 : ℝ) (a42 : ℝ) (a43 : ℝ) (a44 : ℝ)

/-- Checks if a sequence of 4 numbers forms an arithmetic progression -/
def isArithmeticSequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

/-- Defines the conditions for the grid based on the problem statement -/
def validGrid (g : Grid) : Prop :=
  g.a12 = 9 ∧ g.a21 = 1 ∧ g.a34 = 5 ∧ g.a43 = 8 ∧
  isArithmeticSequence g.a11 g.a12 g.a13 g.a14 ∧
  isArithmeticSequence g.a21 g.a22 g.a23 g.a24 ∧
  isArithmeticSequence g.a31 g.a32 g.a33 g.a34 ∧
  isArithmeticSequence g.a41 g.a42 g.a43 g.a44 ∧
  isArithmeticSequence g.a11 g.a21 g.a31 g.a41 ∧
  isArithmeticSequence g.a12 g.a22 g.a32 g.a42 ∧
  isArithmeticSequence g.a13 g.a23 g.a33 g.a43 ∧
  isArithmeticSequence g.a14 g.a24 g.a34 g.a44

/-- The main theorem stating that no valid grid exists -/
theorem no_valid_grid : ¬ ∃ (g : Grid), validGrid g := by
  sorry

end NUMINAMATH_CALUDE_no_valid_grid_l3013_301323


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3013_301301

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  (x + 8*y) / (x*y) ≥ 9 :=
sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  (x + 8*y) / (x*y) = 9 ↔ x = 4/3 ∧ y = 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3013_301301


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_equals_five_halves_l3013_301364

theorem sqrt_x_div_sqrt_y_equals_five_halves (x y : ℝ) 
  (h : ((2/3)^2 + (1/6)^2) / ((1/2)^2 + (1/7)^2) = 28*x/(25*y)) : 
  Real.sqrt x / Real.sqrt y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_equals_five_halves_l3013_301364


namespace NUMINAMATH_CALUDE_quadratic_equation_with_integer_roots_l3013_301392

theorem quadratic_equation_with_integer_roots (m : ℤ) 
  (h1 : ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
    a^2 + m*a - m + 1 = 0 ∧ b^2 + m*b - m + 1 = 0) : 
  m = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_integer_roots_l3013_301392


namespace NUMINAMATH_CALUDE_always_possible_to_reach_final_state_l3013_301377

/-- Represents the two types of operations that can be performed. -/
inductive Operation
  | RedToBlue
  | BlueToRed

/-- Represents the state of the slips for a single MOPper. -/
structure MOPperState where
  number : Nat
  redSlip : Nat
  blueSlip : Nat

/-- Represents the state of all MOPpers' slips. -/
def SystemState := List MOPperState

/-- Initializes the system state based on the given A and B values. -/
def initializeState (A B : Nat) : SystemState :=
  sorry

/-- Performs a single operation on the system state. -/
def performOperation (state : SystemState) (op : Operation) : SystemState :=
  sorry

/-- Checks if the system state is in the desired final configuration. -/
def isFinalState (state : SystemState) : Bool :=
  sorry

/-- The main theorem to be proved. -/
theorem always_possible_to_reach_final_state :
  ∀ (A B : Nat), A ≤ 2010 → B ≤ 2010 →
  ∃ (ops : List Operation),
    isFinalState (ops.foldl performOperation (initializeState A B)) = true :=
  sorry

end NUMINAMATH_CALUDE_always_possible_to_reach_final_state_l3013_301377


namespace NUMINAMATH_CALUDE_inverse_sum_mod_31_l3013_301349

theorem inverse_sum_mod_31 : ∃ (a b : ℤ), (5 * a) % 31 = 1 ∧ (5^2 * b) % 31 = 1 ∧ (a + b) % 31 = 26 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_31_l3013_301349


namespace NUMINAMATH_CALUDE_min_xy_point_l3013_301380

theorem min_xy_point (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 1/x + 1/(2*y) + 3/(2*x*y) = 1) :
  x * y ≥ 9/2 ∧ (x * y = 9/2 ↔ x = 3 ∧ y = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_min_xy_point_l3013_301380


namespace NUMINAMATH_CALUDE_f_zero_at_three_l3013_301384

/-- The polynomial function f(x) = 3x^4 - 2x^3 + x^2 - 4x + r -/
def f (x r : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + x^2 - 4 * x + r

/-- Theorem stating that f(3) = 0 if and only if r = -186 -/
theorem f_zero_at_three (r : ℝ) : f 3 r = 0 ↔ r = -186 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_at_three_l3013_301384


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l3013_301389

theorem sum_of_four_numbers : 1357 + 3571 + 5713 + 7135 = 17776 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l3013_301389


namespace NUMINAMATH_CALUDE_smallest_positive_solution_tan_equation_l3013_301373

theorem smallest_positive_solution_tan_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 ∧ Real.tan (3 * y) - Real.tan (2 * y) = 1 / Real.cos (2 * y) → x ≤ y) ∧
  Real.tan (3 * x) - Real.tan (2 * x) = 1 / Real.cos (2 * x) ∧
  x = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_tan_equation_l3013_301373


namespace NUMINAMATH_CALUDE_store_profit_calculation_l3013_301355

-- Define the types of sweaters
inductive SweaterType
| Turtleneck
| Crewneck
| Vneck

-- Define the initial cost, quantity, and markup percentages for each sweater type
def initial_cost (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 30
  | SweaterType.Crewneck => 25
  | SweaterType.Vneck => 20

def quantity (s : SweaterType) : ℕ :=
  match s with
  | SweaterType.Turtleneck => 100
  | SweaterType.Crewneck => 150
  | SweaterType.Vneck => 200

def initial_markup (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 0.2
  | SweaterType.Crewneck => 0.35
  | SweaterType.Vneck => 0.25

def new_year_markup (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 0.25
  | SweaterType.Crewneck => 0.15
  | SweaterType.Vneck => 0.2

def february_discount (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 0.09
  | SweaterType.Crewneck => 0.12
  | SweaterType.Vneck => 0.15

-- Calculate the final price for each sweater type
def final_price (s : SweaterType) : ℚ :=
  let base_price := initial_cost s * (1 + initial_markup s)
  let new_year_price := base_price + initial_cost s * new_year_markup s
  new_year_price * (1 - february_discount s)

-- Calculate the profit for each sweater type
def profit (s : SweaterType) : ℚ :=
  (final_price s - initial_cost s) * quantity s

-- Calculate the total profit
def total_profit : ℚ :=
  profit SweaterType.Turtleneck + profit SweaterType.Crewneck + profit SweaterType.Vneck

-- Theorem statement
theorem store_profit_calculation :
  total_profit = 3088.5 := by sorry

end NUMINAMATH_CALUDE_store_profit_calculation_l3013_301355


namespace NUMINAMATH_CALUDE_inequality_solution_l3013_301314

def choose (n k : ℕ) : ℕ := Nat.choose n k

def permute (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem inequality_solution (x : ℕ) :
  x > 0 → (choose 5 x + permute x 3 < 30 ↔ x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3013_301314


namespace NUMINAMATH_CALUDE_opposite_signs_for_positive_solution_l3013_301313

theorem opposite_signs_for_positive_solution (a b : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ x : ℝ, x > 0 ∧ a * x + b = 0) : a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_for_positive_solution_l3013_301313


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_18_l3013_301393

theorem largest_divisor_of_n_squared_divisible_by_18 (n : ℕ+) (h : 18 ∣ n^2) :
  6 = Nat.gcd 6 n ∧ ∀ m : ℕ, m ∣ n → m ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_18_l3013_301393


namespace NUMINAMATH_CALUDE_concavity_and_inflection_point_l3013_301365

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 4

-- Define the second derivative of f
def f'' (x : ℝ) : ℝ := 6*x - 12

-- Theorem stating the concavity and inflection point properties
theorem concavity_and_inflection_point :
  (∀ x < 2, f'' x < 0) ∧
  (∀ x > 2, f'' x > 0) ∧
  f'' 2 = 0 ∧
  f 2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_concavity_and_inflection_point_l3013_301365


namespace NUMINAMATH_CALUDE_final_dislikes_is_300_l3013_301399

/-- Represents the number of likes and dislikes on a YouTube video -/
structure VideoStats where
  likes : ℕ
  dislikes : ℕ

/-- Calculates the final number of dislikes after changes -/
def finalDislikes (original : VideoStats) : ℕ :=
  3 * original.dislikes

/-- Theorem: Given the conditions, the final number of dislikes is 300 -/
theorem final_dislikes_is_300 (original : VideoStats) 
    (h1 : original.likes = 3 * original.dislikes)
    (h2 : original.likes = 100 + 2 * original.dislikes) : 
  finalDislikes original = 300 := by
  sorry

#eval finalDislikes {likes := 300, dislikes := 100}

end NUMINAMATH_CALUDE_final_dislikes_is_300_l3013_301399


namespace NUMINAMATH_CALUDE_odd_function_log_value_l3013_301357

theorem odd_function_log_value (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = Real.log x / Real.log 2) →  -- f(x) = log₂(x) for x > 0
  f (-2) = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_log_value_l3013_301357


namespace NUMINAMATH_CALUDE_residue_mod_13_l3013_301397

theorem residue_mod_13 : (250 * 11 - 20 * 6 + 5^2) % 13 = 3 := by sorry

end NUMINAMATH_CALUDE_residue_mod_13_l3013_301397


namespace NUMINAMATH_CALUDE_vending_machine_probability_l3013_301398

/-- The number of toys in the vending machine -/
def num_toys : ℕ := 10

/-- The price of the cheapest toy in cents -/
def min_price : ℕ := 50

/-- The price increment between toys in cents -/
def price_increment : ℕ := 25

/-- The price of Sam's favorite toy in cents -/
def favorite_toy_price : ℕ := 225

/-- The number of quarters Sam has initially -/
def initial_quarters : ℕ := 12

/-- The value of Sam's bill in cents -/
def bill_value : ℕ := 2000

/-- The probability that Sam has to break his twenty-dollar bill -/
def probability_break_bill : ℚ := 8/9

theorem vending_machine_probability :
  let total_permutations := Nat.factorial num_toys
  let favorable_permutations := Nat.factorial (num_toys - 1) + Nat.factorial (num_toys - 2)
  probability_break_bill = 1 - (favorable_permutations : ℚ) / total_permutations :=
by sorry

end NUMINAMATH_CALUDE_vending_machine_probability_l3013_301398


namespace NUMINAMATH_CALUDE_units_digit_is_seven_l3013_301306

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10
  hundreds_gt_0 : hundreds > 0

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed value of a three-digit number -/
def ThreeDigitNumber.reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- Theorem stating the units digit of the result is 7 -/
theorem units_digit_is_seven (n : ThreeDigitNumber) 
    (h : n.hundreds = n.units + 3) : 
    (n.reversed_value - 2 * n.value) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_is_seven_l3013_301306


namespace NUMINAMATH_CALUDE_solution_set_range_of_a_l3013_301361

-- Part 1
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem solution_set (x : ℝ) : f x ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2 := by sorry

-- Part 2
def g (a x : ℝ) : ℝ := |x - 1| + |x - a|

theorem range_of_a (a : ℝ) : 
  (a > 1 ∧ ∀ x, g a x + |x - 1| ≥ 1) ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_range_of_a_l3013_301361


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3013_301369

theorem negation_of_existence_proposition :
  ¬(∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔
  (∀ a : ℝ, ∀ x : ℝ, a * x^2 + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3013_301369


namespace NUMINAMATH_CALUDE_divisibility_by_two_iff_last_digit_even_l3013_301354

theorem divisibility_by_two_iff_last_digit_even (a : ℕ) : 
  ∃ b c : ℕ, a = 10 * b + c ∧ c < 10 → (∃ k : ℕ, a = 2 * k ↔ ∃ m : ℕ, c = 2 * m) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_two_iff_last_digit_even_l3013_301354


namespace NUMINAMATH_CALUDE_smallest_surface_area_of_glued_cubes_smallest_surface_area_proof_l3013_301310

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

end NUMINAMATH_CALUDE_smallest_surface_area_of_glued_cubes_smallest_surface_area_proof_l3013_301310


namespace NUMINAMATH_CALUDE_cheburashka_krakozyabra_relation_num_cheburashkas_is_eleven_l3013_301309

/-- Represents the number of Cheburashkas in Katya's drawing -/
def num_cheburashkas : ℕ := 11

/-- Represents the total number of Krakozyabras in the final drawing -/
def total_krakozyabras : ℕ := 29

/-- Represents the number of rows in Katya's drawing -/
def num_rows : ℕ := 2

/-- Theorem stating the relationship between Cheburashkas and Krakozyabras -/
theorem cheburashka_krakozyabra_relation :
  num_cheburashkas = (total_krakozyabras + num_rows) / 2 := by
  sorry

/-- Theorem proving that the number of Cheburashkas is 11 -/
theorem num_cheburashkas_is_eleven :
  num_cheburashkas = 11 := by
  sorry

end NUMINAMATH_CALUDE_cheburashka_krakozyabra_relation_num_cheburashkas_is_eleven_l3013_301309


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_through_point_implies_a_l3013_301342

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

end NUMINAMATH_CALUDE_hyperbola_asymptote_through_point_implies_a_l3013_301342


namespace NUMINAMATH_CALUDE_product_of_squared_terms_l3013_301371

theorem product_of_squared_terms (x : ℝ) : 3 * x^2 * (2 * x^2) = 6 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squared_terms_l3013_301371


namespace NUMINAMATH_CALUDE_line_parameterization_l3013_301363

/-- Given a line y = 2x - 40 parameterized by (x,y) = (g(t), 20t - 14), 
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t x y : ℝ, y = 2*x - 40 ∧ x = g t ∧ y = 20*t - 14) → 
  (∀ t : ℝ, g t = 10*t + 13) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l3013_301363


namespace NUMINAMATH_CALUDE_complex_cube_root_of_unity_l3013_301352

theorem complex_cube_root_of_unity : (1/2 - Complex.I * (Real.sqrt 3)/2)^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_of_unity_l3013_301352


namespace NUMINAMATH_CALUDE_set_operations_l3013_301360

def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

theorem set_operations :
  (Set.compl (A ∩ B) = {x | x < 3 ∨ x ≥ 6}) ∧
  ((Set.compl B ∪ A) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3013_301360


namespace NUMINAMATH_CALUDE_triangle_circumradius_l3013_301356

/-- Given a triangle with sides a and b, area S, and the median to the third side
    less than half of that side, prove that the radius of the circumcircle is 8 / √15 -/
theorem triangle_circumradius (a b : ℝ) (S : ℝ) (h_a : a = 2) (h_b : b = 3)
  (h_S : S = (3 * Real.sqrt 15) / 4)
  (h_median : ∃ (m : ℝ), m < (a + b) / 4 ∧ m^2 = (2 * (a^2 + b^2) - ((a + b) / 2)^2) / 4) :
  ∃ (R : ℝ), R = 8 / Real.sqrt 15 ∧ R * 2 * S = a * b * (Real.sqrt ((a + b + (a + b)) * (-a + b + (a + b)) * (a - b + (a + b)) * (a + b - (a + b))) / (4 * (a + b))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l3013_301356


namespace NUMINAMATH_CALUDE_average_salary_combined_l3013_301329

theorem average_salary_combined (num_supervisors : ℕ) (num_laborers : ℕ) 
  (avg_salary_supervisors : ℚ) (avg_salary_laborers : ℚ) :
  num_supervisors = 6 →
  num_laborers = 42 →
  avg_salary_supervisors = 2450 →
  avg_salary_laborers = 950 →
  let total_salary := num_supervisors * avg_salary_supervisors + num_laborers * avg_salary_laborers
  let total_workers := num_supervisors + num_laborers
  (total_salary / total_workers : ℚ) = 1137.5 := by
sorry

end NUMINAMATH_CALUDE_average_salary_combined_l3013_301329


namespace NUMINAMATH_CALUDE_arrangements_count_l3013_301333

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

end NUMINAMATH_CALUDE_arrangements_count_l3013_301333


namespace NUMINAMATH_CALUDE_different_gender_choices_eq_450_l3013_301395

/-- The number of boys in the club -/
def num_boys : ℕ := 15

/-- The number of girls in the club -/
def num_girls : ℕ := 15

/-- The total number of members in the club -/
def total_members : ℕ := num_boys + num_girls

/-- The number of ways to choose a president and a vice-president of different genders -/
def different_gender_choices : ℕ := num_boys * num_girls * 2

theorem different_gender_choices_eq_450 : different_gender_choices = 450 := by
  sorry

end NUMINAMATH_CALUDE_different_gender_choices_eq_450_l3013_301395


namespace NUMINAMATH_CALUDE_balloon_count_l3013_301390

theorem balloon_count (friend_balloons : ℕ) (difference : ℕ) : 
  friend_balloons = 5 → difference = 2 → friend_balloons + difference = 7 :=
by sorry

end NUMINAMATH_CALUDE_balloon_count_l3013_301390


namespace NUMINAMATH_CALUDE_angle_a_measure_l3013_301337

/-- An isosceles right triangle with side lengths and angles -/
structure IsoscelesRightTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  ac : ℝ
  -- Angles in radians
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  -- Properties
  ab_eq_bc : ab = bc
  right_angle_b : angle_b = Real.pi / 2
  angle_sum : angle_a + angle_b + angle_c = Real.pi

/-- The measure of angle A in an isosceles right triangle is π/4 radians (45 degrees) -/
theorem angle_a_measure (t : IsoscelesRightTriangle) : t.angle_a = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_a_measure_l3013_301337


namespace NUMINAMATH_CALUDE_discount_comparison_l3013_301340

def initial_amount : ℝ := 20000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.15, 0.15, 0.05]
def option2_discounts : List ℝ := [0.30, 0.10, 0.02]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem discount_comparison :
  apply_successive_discounts initial_amount option1_discounts -
  apply_successive_discounts initial_amount option2_discounts = 1379.50 :=
sorry

end NUMINAMATH_CALUDE_discount_comparison_l3013_301340


namespace NUMINAMATH_CALUDE_trip_distance_l3013_301343

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

end NUMINAMATH_CALUDE_trip_distance_l3013_301343


namespace NUMINAMATH_CALUDE_polly_hungry_tweet_rate_l3013_301319

def happy_tweets_per_minute : ℕ := 18
def mirror_tweets_per_minute : ℕ := 45
def duration_per_state : ℕ := 20
def total_tweets : ℕ := 1340

def hungry_tweets_per_minute : ℕ := 4

theorem polly_hungry_tweet_rate :
  happy_tweets_per_minute * duration_per_state +
  hungry_tweets_per_minute * duration_per_state +
  mirror_tweets_per_minute * duration_per_state = total_tweets :=
by sorry

end NUMINAMATH_CALUDE_polly_hungry_tweet_rate_l3013_301319


namespace NUMINAMATH_CALUDE_simplify_and_solve_for_t_l3013_301341

theorem simplify_and_solve_for_t
  (m Q : ℝ)
  (j : ℝ)
  (h : j ≠ -2)
  (h_pos_m : m > 0)
  (h_pos_Q : Q > 0)
  (h_eq : Q = m / (2 + j) ^ t) :
  t = Real.log (m / Q) / Real.log (2 + j) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_solve_for_t_l3013_301341


namespace NUMINAMATH_CALUDE_lisa_expenses_l3013_301336

theorem lisa_expenses (B : ℝ) (book coffee : ℝ) : 
  book = 0.3 * (B - 2 * coffee) →
  coffee = 0.1 * (B - book) →
  book + coffee = (31 : ℝ) / 94 * B := by
sorry

end NUMINAMATH_CALUDE_lisa_expenses_l3013_301336


namespace NUMINAMATH_CALUDE_quadratic_sign_l3013_301335

/-- A quadratic function of the form f(x) = x^2 + x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + x + c

theorem quadratic_sign (c : ℝ) (p : ℝ) 
  (h1 : f c 0 > 0) 
  (h2 : f c p < 0) : 
  f c (p + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sign_l3013_301335


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3013_301308

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3013_301308


namespace NUMINAMATH_CALUDE_remainder_17_pow_2046_mod_23_l3013_301391

theorem remainder_17_pow_2046_mod_23 : 17^2046 % 23 = 22 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_pow_2046_mod_23_l3013_301391


namespace NUMINAMATH_CALUDE_simplify_fraction_l3013_301344

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3013_301344


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3013_301312

theorem min_sum_of_squares (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3013_301312


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l3013_301362

theorem smallest_number_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < 3668 → 
    ¬((y + 7) % 25 = 0 ∧ (y + 7) % 49 = 0 ∧ (y + 7) % 15 = 0 ∧ (y + 7) % 21 = 0)) ∧
  ((3668 + 7) % 25 = 0 ∧ (3668 + 7) % 49 = 0 ∧ (3668 + 7) % 15 = 0 ∧ (3668 + 7) % 21 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l3013_301362


namespace NUMINAMATH_CALUDE_no_convex_equal_sided_all_obtuse_polygon_l3013_301348

/-- A polygon is represented as a list of points in 2D space -/
def Polygon := List (Real × Real)

/-- A polygon is convex if for any three consecutive vertices, the turn is always in the same direction -/
def is_convex (p : Polygon) : Prop := sorry

/-- All sides of a polygon have equal length -/
def has_equal_sides (p : Polygon) : Prop := sorry

/-- Three points form an obtuse triangle if one of its angles is greater than 90 degrees -/
def is_obtuse_triangle (a b c : Real × Real) : Prop := sorry

/-- Any three vertices of the polygon form an obtuse triangle -/
def all_triangles_obtuse (p : Polygon) : Prop := sorry

theorem no_convex_equal_sided_all_obtuse_polygon :
  ¬∃ (p : Polygon), is_convex p ∧ has_equal_sides p ∧ all_triangles_obtuse p := by
  sorry

end NUMINAMATH_CALUDE_no_convex_equal_sided_all_obtuse_polygon_l3013_301348


namespace NUMINAMATH_CALUDE_great_grandchildren_count_l3013_301302

theorem great_grandchildren_count (age : ℕ) (grandchildren : ℕ) (n : ℕ) 
  (h1 : age = 91)
  (h2 : grandchildren = 11)
  (h3 : grandchildren * n * age = n * 1000 + n) :
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_great_grandchildren_count_l3013_301302


namespace NUMINAMATH_CALUDE_division_inequality_l3013_301351

theorem division_inequality : ¬(∃ q r, 4900 = 600 * q + r ∧ r < 600 ∧ 49 = 6 * q + r ∧ r < 6) := by
  sorry

end NUMINAMATH_CALUDE_division_inequality_l3013_301351


namespace NUMINAMATH_CALUDE_complex_equation_proof_l3013_301315

theorem complex_equation_proof (a : ℝ) : 
  ((2 * a) / (1 + Complex.I) + 1 + Complex.I).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l3013_301315


namespace NUMINAMATH_CALUDE_simplify_expression_l3013_301304

theorem simplify_expression (x y : ℚ) (hx : x = 5) (hy : y = 2) :
  (10 * x^2 * y^3) / (15 * x * y^2) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3013_301304


namespace NUMINAMATH_CALUDE_congruence_system_solution_l3013_301305

theorem congruence_system_solution (n : ℤ) :
  (n % 5 = 3 ∧ n % 7 = 4 ∧ n % 3 = 2) ↔ ∃ k : ℤ, n = 105 * k + 53 :=
sorry

end NUMINAMATH_CALUDE_congruence_system_solution_l3013_301305


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l3013_301385

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop := x^2 + 2*x - 8*y^2 = 0

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  ∀ x y, f x y ↔ (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

/-- Theorem stating that the given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation := by
  sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l3013_301385


namespace NUMINAMATH_CALUDE_square_area_l3013_301387

theorem square_area (s : ℝ) (h : (2/5 * s) * 10 = 140) : s^2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l3013_301387


namespace NUMINAMATH_CALUDE_min_value_expression_l3013_301345

theorem min_value_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) (heq : 2 * m + n = 4) :
  1 / m + 2 / n ≥ 2 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 4 ∧ 1 / m₀ + 2 / n₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3013_301345


namespace NUMINAMATH_CALUDE_quadratic_roots_inversely_proportional_l3013_301396

/-- 
Given a quadratic equation x^2 + px + q = 0 where q is constant and p is variable,
prove that the roots x₁ and x₂ are inversely proportional to each other.
-/
theorem quadratic_roots_inversely_proportional 
  (p q : ℝ) (x₁ x₂ : ℝ) (h_const : q ≠ 0) :
  (x₁^2 + p*x₁ + q = 0) → (x₂^2 + p*x₂ + q = 0) → 
  ∃ (k : ℝ), k ≠ 0 ∧ x₁ * x₂ = k :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_inversely_proportional_l3013_301396


namespace NUMINAMATH_CALUDE_proposition_implication_l3013_301374

theorem proposition_implication (P : ℕ → Prop) :
  (∀ k : ℕ, k > 0 → (P k → P (k + 1))) →
  (¬ P 5) →
  (¬ P 4) :=
sorry

end NUMINAMATH_CALUDE_proposition_implication_l3013_301374


namespace NUMINAMATH_CALUDE_log_inequalities_l3013_301379

-- Define the logarithm functions
noncomputable def log₃ (x : ℝ) := Real.log x / Real.log 3
noncomputable def log₁₃ (x : ℝ) := Real.log x / Real.log (1/3)

-- State the theorem
theorem log_inequalities :
  (∀ x y, x < y → log₃ x < log₃ y) →  -- log₃ is increasing
  (∀ x y, x < y → log₁₃ x > log₁₃ y) →  -- log₁₃ is decreasing
  (1/5)^0 = 1 →
  log₃ 4 > (1/5)^0 ∧ (1/5)^0 > log₁₃ 10 :=
by sorry

end NUMINAMATH_CALUDE_log_inequalities_l3013_301379


namespace NUMINAMATH_CALUDE_ac_length_is_18_l3013_301322

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- AB length is 12 -/
  ab_length : dist A B = 12
  /-- AD length is 8 -/
  ad_length : dist A D = 8
  /-- DC length is 18 -/
  dc_length : dist D C = 18
  /-- AD is perpendicular to AB -/
  ad_perp_ab : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0
  /-- ABCD is symmetric about AC -/
  symmetric_about_ac : ∃ (m : ℝ) (b : ℝ), 
    (C.2 - A.2) = m * (C.1 - A.1) ∧
    B.2 - A.2 = m * (B.1 - A.1) + b ∧
    D.2 - A.2 = -(m * (D.1 - A.1) + b)

/-- The length of AC in a SpecialQuadrilateral is 18 -/
theorem ac_length_is_18 (q : SpecialQuadrilateral) : dist q.A q.C = 18 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_is_18_l3013_301322


namespace NUMINAMATH_CALUDE_money_calculation_l3013_301326

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def totalMoney (n50 : ℕ) (n500 : ℕ) : ℕ :=
  50 * n50 + 500 * n500

/-- Theorem stating that given 90 notes in total, with 77 being 50 rupee notes,
    the total amount of money is 10350 rupees -/
theorem money_calculation :
  let total_notes : ℕ := 90
  let n50 : ℕ := 77
  let n500 : ℕ := total_notes - n50
  totalMoney n50 n500 = 10350 := by
sorry

end NUMINAMATH_CALUDE_money_calculation_l3013_301326


namespace NUMINAMATH_CALUDE_necessary_condition_for_existence_l3013_301370

theorem necessary_condition_for_existence (a : ℝ) :
  (∃ x ∈ Set.Icc 1 2, x^2 - a > 0) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_existence_l3013_301370
