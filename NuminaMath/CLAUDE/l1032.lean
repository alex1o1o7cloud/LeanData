import Mathlib

namespace NUMINAMATH_CALUDE_distinct_selections_is_fifteen_l1032_103274

/-- The number of vowels in "MATHCOUNTS" -/
def num_vowels : ℕ := 3

/-- The number of distinct consonants in "MATHCOUNTS" excluding T -/
def num_distinct_consonants : ℕ := 5

/-- The number of T's in "MATHCOUNTS" -/
def num_t : ℕ := 2

/-- The total number of consonants in "MATHCOUNTS" -/
def total_consonants : ℕ := num_distinct_consonants + num_t

/-- The number of vowels to be selected -/
def vowels_to_select : ℕ := 3

/-- The number of consonants to be selected -/
def consonants_to_select : ℕ := 2

/-- The function to calculate the number of distinct ways to select letters -/
def distinct_selections : ℕ :=
  Nat.choose num_vowels vowels_to_select * Nat.choose num_distinct_consonants consonants_to_select +
  Nat.choose num_vowels vowels_to_select * Nat.choose (num_distinct_consonants - 1) (consonants_to_select - 1) +
  Nat.choose num_vowels vowels_to_select * Nat.choose (num_distinct_consonants - 2) (consonants_to_select - 2)

theorem distinct_selections_is_fifteen :
  distinct_selections = 15 :=
sorry

end NUMINAMATH_CALUDE_distinct_selections_is_fifteen_l1032_103274


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1032_103277

/-- 
If 3x^2 - 7x + m = 0 is a quadratic equation with exactly one solution for x, 
then m = 49/12.
-/
theorem unique_solution_quadratic (m : ℚ) : 
  (∃! x : ℚ, 3 * x^2 - 7 * x + m = 0) → m = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1032_103277


namespace NUMINAMATH_CALUDE_parrot_count_l1032_103204

theorem parrot_count (total_birds : ℕ) (remaining_parrots : ℕ) (remaining_crow : ℕ) 
  (h1 : total_birds = 13)
  (h2 : remaining_parrots = 2)
  (h3 : remaining_crow = 1)
  (h4 : ∃ (x : ℕ), total_birds = remaining_parrots + remaining_crow + 2 * x) :
  ∃ (initial_parrots : ℕ), initial_parrots = 7 ∧ 
    ∃ (initial_crows : ℕ), initial_crows + initial_parrots = total_birds :=
by
  sorry

end NUMINAMATH_CALUDE_parrot_count_l1032_103204


namespace NUMINAMATH_CALUDE_rightward_translation_of_point_l1032_103245

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translation to the right by a given distance -/
def translateRight (p : Point2D) (distance : ℝ) : Point2D :=
  { x := p.x + distance, y := p.y }

theorem rightward_translation_of_point :
  let initial_point : Point2D := { x := 4, y := -3 }
  let translated_point := translateRight initial_point 1
  translated_point = { x := 5, y := -3 } := by sorry

end NUMINAMATH_CALUDE_rightward_translation_of_point_l1032_103245


namespace NUMINAMATH_CALUDE_x_intercept_after_rotation_l1032_103297

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotate a line 90 degrees counterclockwise about a given point -/
def rotate90 (l : Line) (p : Point) : Line := sorry

/-- Find the x-intercept of a line -/
def xIntercept (l : Line) : ℝ := sorry

theorem x_intercept_after_rotation :
  let l : Line := { a := 2, b := -3, c := 30 }
  let p : Point := { x := 15, y := 10 }
  let k' := rotate90 l p
  xIntercept k' = 65 / 3 := by sorry

end NUMINAMATH_CALUDE_x_intercept_after_rotation_l1032_103297


namespace NUMINAMATH_CALUDE_horse_gram_consumption_l1032_103260

/-- If 15 horses eat 15 bags of gram in 15 days, then 1 horse will eat 1 bag of gram in 15 days. -/
theorem horse_gram_consumption 
  (horses : ℕ) (bags : ℕ) (days : ℕ) 
  (h_horses : horses = 15)
  (h_bags : bags = 15)
  (h_days : days = 15)
  (h_consumption : horses * bags = horses * days) :
  1 * 1 = 1 * days :=
sorry

end NUMINAMATH_CALUDE_horse_gram_consumption_l1032_103260


namespace NUMINAMATH_CALUDE_johnnys_travel_time_l1032_103246

/-- Proves that given the specified conditions, Johnny's total travel time is 1.6 hours -/
theorem johnnys_travel_time 
  (distance_to_school : ℝ)
  (jogging_speed : ℝ)
  (bus_speed : ℝ)
  (h1 : distance_to_school = 6.461538461538462)
  (h2 : jogging_speed = 5)
  (h3 : bus_speed = 21) :
  distance_to_school / jogging_speed + distance_to_school / bus_speed = 1.6 :=
by sorry


end NUMINAMATH_CALUDE_johnnys_travel_time_l1032_103246


namespace NUMINAMATH_CALUDE_milk_mixture_theorem_l1032_103239

/-- Proves that adding 8 gallons of 10% butterfat milk to 8 gallons of 30% butterfat milk
    results in a mixture with 20% butterfat. -/
theorem milk_mixture_theorem :
  let initial_milk : ℝ := 8
  let initial_butterfat_percent : ℝ := 30
  let added_milk : ℝ := 8
  let added_butterfat_percent : ℝ := 10
  let final_butterfat_percent : ℝ := 20
  let total_milk : ℝ := initial_milk + added_milk
  let total_butterfat : ℝ := (initial_milk * initial_butterfat_percent + added_milk * added_butterfat_percent) / 100
  total_butterfat / total_milk * 100 = final_butterfat_percent :=
by sorry

end NUMINAMATH_CALUDE_milk_mixture_theorem_l1032_103239


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l1032_103224

theorem last_digit_of_sum (n : ℕ) : (2^1992 + 3^1992) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_sum_l1032_103224


namespace NUMINAMATH_CALUDE_positive_solutions_x_minus_y_nonnegative_l1032_103237

-- Define the system of linear equations
def system (x y m : ℝ) : Prop :=
  x + y = 3 * m ∧ 2 * x - 3 * y = m + 5

-- Part 1: Positive solutions
theorem positive_solutions (m : ℝ) :
  (∃ x y : ℝ, system x y m ∧ x > 0 ∧ y > 0) → m > 1 := by
  sorry

-- Part 2: x - y ≥ 0
theorem x_minus_y_nonnegative (m : ℝ) :
  (∃ x y : ℝ, system x y m ∧ x - y ≥ 0) → m ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_positive_solutions_x_minus_y_nonnegative_l1032_103237


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1032_103218

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (1/a + a/b) ≥ 1 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1032_103218


namespace NUMINAMATH_CALUDE_equation_solution_l1032_103254

theorem equation_solution : 
  ∃! x : ℝ, 2 * x - 1 = 3 * x + 2 ∧ x = -3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1032_103254


namespace NUMINAMATH_CALUDE_first_fun_friday_march31_l1032_103206

/-- Represents a date in a year -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

/-- Function to determine if a given date is a Friday -/
def isFriday (d : Date) (startDay : DayOfWeek) : Bool := sorry

/-- Function to count the number of Fridays in a month -/
def countFridays (month : Nat) (startDay : DayOfWeek) : Nat := sorry

/-- Function to determine if a date is a Fun Friday -/
def isFunFriday (d : Date) (startDay : DayOfWeek) : Bool := sorry

/-- Theorem stating that the first Fun Friday of the year is on March 31 -/
theorem first_fun_friday_march31 (startDay : DayOfWeek) :
  startDay = DayOfWeek.Wednesday →
  (∀ d : Date, d.month < 3 → ¬isFunFriday d startDay) →
  isFunFriday { month := 3, day := 31 } startDay :=
sorry

end NUMINAMATH_CALUDE_first_fun_friday_march31_l1032_103206


namespace NUMINAMATH_CALUDE_distance_from_origin_l1032_103268

/-- Given a point (x,y) in the first quadrant satisfying certain conditions,
    prove that its distance from the origin is √(233 + 12√7). -/
theorem distance_from_origin (x y : ℝ) (h1 : y = 14) (h2 : (x - 3)^2 + (y - 8)^2 = 64) (h3 : x > 3) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (233 + 12 * Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1032_103268


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l1032_103241

theorem probability_nine_heads_in_twelve_flips :
  let n : ℕ := 12  -- total number of coin flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 220/4096 :=
by sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l1032_103241


namespace NUMINAMATH_CALUDE_fathers_age_l1032_103212

theorem fathers_age (son_age father_age : ℕ) : 
  father_age = 3 * son_age →
  father_age + 15 = 2 * (son_age + 15) →
  father_age = 45 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l1032_103212


namespace NUMINAMATH_CALUDE_joe_age_l1032_103265

/-- Given that Joe has a daughter Jane, and their ages satisfy certain conditions,
    prove that Joe's age is 38. -/
theorem joe_age (joe_age jane_age : ℕ) 
  (sum_ages : joe_age + jane_age = 54)
  (diff_ages : joe_age - jane_age = 22) : 
  joe_age = 38 := by
sorry

end NUMINAMATH_CALUDE_joe_age_l1032_103265


namespace NUMINAMATH_CALUDE_fraction_simplifiable_l1032_103266

theorem fraction_simplifiable (e : ℤ) : 
  (∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ (16 * e - 10) * b = (10 * e - 3) * a) ↔ 
  (∃ (k : ℤ), e = 13 * k + 12) := by
sorry

end NUMINAMATH_CALUDE_fraction_simplifiable_l1032_103266


namespace NUMINAMATH_CALUDE_minimum_postage_l1032_103211

/-- Calculates the postage for a given weight in grams -/
def calculatePostage (weight : ℕ) : ℚ :=
  if weight ≤ 100 then
    (((weight - 1) / 20 + 1) * 8) / 10
  else
    4 + (((weight - 101) / 100 + 1) * 2)

/-- Calculates the total postage for two envelopes -/
def totalPostage (x : ℕ) : ℚ :=
  calculatePostage (12 * x + 4) + calculatePostage (12 * (11 - x) + 4)

theorem minimum_postage :
  ∃ x : ℕ, x ≤ 11 ∧ totalPostage x = 56/10 ∧ ∀ y : ℕ, y ≤ 11 → totalPostage y ≥ 56/10 :=
sorry

end NUMINAMATH_CALUDE_minimum_postage_l1032_103211


namespace NUMINAMATH_CALUDE_vector_operation_equals_two_l1032_103217

-- Define the vectors
def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (1, -1)

-- Define the dot product operation
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define vector scalar multiplication
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

-- Define vector subtraction
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

-- Theorem statement
theorem vector_operation_equals_two :
  dot_product (vector_sub (scalar_mult 2 a) b) b = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_equals_two_l1032_103217


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1032_103219

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) * (Real.sqrt 9 / Real.sqrt 10) = Real.sqrt 210 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1032_103219


namespace NUMINAMATH_CALUDE_initial_candies_l1032_103283

theorem initial_candies (eaten : ℕ) (left : ℕ) (h1 : eaten = 15) (h2 : left = 13) :
  eaten + left = 28 := by
  sorry

end NUMINAMATH_CALUDE_initial_candies_l1032_103283


namespace NUMINAMATH_CALUDE_remaining_toenail_capacity_l1032_103243

/- Jar capacity in terms of regular toenails -/
def jar_capacity : ℕ := 100

/- Size ratio of big toenails to regular toenails -/
def big_toenail_ratio : ℕ := 2

/- Number of big toenails already in the jar -/
def big_toenails_in_jar : ℕ := 20

/- Number of regular toenails already in the jar -/
def regular_toenails_in_jar : ℕ := 40

/- Theorem: The number of additional regular toenails that can fit in the jar is 20 -/
theorem remaining_toenail_capacity :
  jar_capacity - (big_toenails_in_jar * big_toenail_ratio + regular_toenails_in_jar) = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_toenail_capacity_l1032_103243


namespace NUMINAMATH_CALUDE_sum_of_squares_l1032_103287

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 20 → 
  a * b + b * c + a * c = 131 → 
  a^2 + b^2 + c^2 = 138 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1032_103287


namespace NUMINAMATH_CALUDE_parking_lot_buses_l1032_103249

/-- Given a parking lot with buses and cars, prove the number of buses -/
theorem parking_lot_buses (total_vehicles : ℕ) (total_wheels : ℕ) : 
  total_vehicles = 40 →
  total_wheels = 210 →
  ∃ (buses cars : ℕ),
    buses + cars = total_vehicles ∧
    6 * buses + 4 * cars = total_wheels ∧
    buses = 25 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_buses_l1032_103249


namespace NUMINAMATH_CALUDE_no_nonzero_solution_l1032_103238

theorem no_nonzero_solution (x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  (x^2 + x = y^2 - y ∧ 
   y^2 + y = z^2 - z ∧ 
   z^2 + z = x^2 - x) → 
  False :=
sorry

end NUMINAMATH_CALUDE_no_nonzero_solution_l1032_103238


namespace NUMINAMATH_CALUDE_function_values_imply_parameters_l1032_103225

theorem function_values_imply_parameters 
  (f : ℝ → ℝ) 
  (a θ : ℝ) 
  (h1 : ∀ x, f x = Real.sin (x + θ) + a * Real.cos (x + 2 * θ))
  (h2 : θ > -Real.pi / 2 ∧ θ < Real.pi / 2)
  (h3 : f (Real.pi / 2) = 0)
  (h4 : f Real.pi = 1) :
  a = -1 ∧ θ = -Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_function_values_imply_parameters_l1032_103225


namespace NUMINAMATH_CALUDE_g_negative_one_equals_three_l1032_103257

-- Define an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the theorem
theorem g_negative_one_equals_three
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_g_def : ∀ x, g x = f x + 2)
  (h_g_one : g 1 = 1) :
  g (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_one_equals_three_l1032_103257


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l1032_103205

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (onion_weight carrot_weight : ℕ),
    onion_weight + carrot_weight = (bag_weights.sum - t) ∧
    carrot_weight = 2 * onion_weight ∧
    ∃ (onion_bags carrot_bags : List ℕ),
      onion_bags ++ carrot_bags = bag_weights.filter (· ≠ t) ∧
      onion_bags.sum = onion_weight ∧
      carrot_bags.sum = carrot_weight

theorem turnip_bag_weights :
  ∀ t, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 := by sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l1032_103205


namespace NUMINAMATH_CALUDE_log_2_5_gt_log_2_3_l1032_103242

-- Define log_2 as a strictly increasing function
def log_2 : ℝ → ℝ := sorry

-- Axiom: log_2 is strictly increasing
axiom log_2_strictly_increasing : 
  ∀ x y : ℝ, x > y → log_2 x > log_2 y

-- Theorem to prove
theorem log_2_5_gt_log_2_3 : log_2 5 > log_2 3 := by
  sorry

end NUMINAMATH_CALUDE_log_2_5_gt_log_2_3_l1032_103242


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1032_103253

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x y : ℝ, x = 2 ∧ y = -2 ∧ ∀ a : ℝ, a > 0 → a ≠ 1 → a^(x - 2) - 3 = y :=
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1032_103253


namespace NUMINAMATH_CALUDE_original_group_size_l1032_103255

theorem original_group_size (n : ℕ) (W : ℝ) : 
  W = n * 35 ∧ 
  W + 40 = (n + 1) * 36 →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_original_group_size_l1032_103255


namespace NUMINAMATH_CALUDE_linear_function_k_value_l1032_103248

/-- Given a linear function y = kx + 6 passing through the point (2, -2), prove that k = -4 -/
theorem linear_function_k_value :
  ∀ k : ℝ, (∀ x y : ℝ, y = k * x + 6) → -2 = k * 2 + 6 → k = -4 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l1032_103248


namespace NUMINAMATH_CALUDE_treasure_gold_amount_l1032_103291

theorem treasure_gold_amount (total_mass : ℝ) (num_brothers : ℕ) 
  (eldest_gold : ℝ) (eldest_silver_fraction : ℝ) :
  total_mass = num_brothers * 100 →
  eldest_gold = 25 →
  eldest_silver_fraction = 1 / 8 →
  ∃ (total_gold total_silver : ℝ),
    total_gold + total_silver = total_mass ∧
    total_gold = 100 ∧
    eldest_gold + eldest_silver_fraction * total_silver = 100 :=
by sorry

end NUMINAMATH_CALUDE_treasure_gold_amount_l1032_103291


namespace NUMINAMATH_CALUDE_shower_water_usage_l1032_103256

/-- The total water usage of Roman and Remy's showers -/
theorem shower_water_usage (R : ℝ) 
  (h1 : 3 * R + 1 = 25) : R + (3 * R + 1) = 33 := by
  sorry

end NUMINAMATH_CALUDE_shower_water_usage_l1032_103256


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_property_three_digit_numbers_with_property_exists_infinite_sequence_l1032_103293

-- Define a function to check if a number has the desired property
def has_property (n : ℕ) (base : ℕ) : Prop :=
  n^2 % base = n

-- Theorem for two-digit numbers
theorem two_digit_numbers_with_property :
  ∃ (A B : ℕ), A ≠ B ∧ 10 ≤ A ∧ A < 100 ∧ 10 ≤ B ∧ B < 100 ∧
  has_property A 100 ∧ has_property B 100 ∧
  ∀ (C : ℕ), 10 ≤ C ∧ C < 100 ∧ has_property C 100 → (C = A ∨ C = B) :=
sorry

-- Theorem for three-digit numbers
theorem three_digit_numbers_with_property :
  ∃ (A B : ℕ), A ≠ B ∧ 100 ≤ A ∧ A < 1000 ∧ 100 ≤ B ∧ B < 1000 ∧
  has_property A 1000 ∧ has_property B 1000 ∧
  ∀ (C : ℕ), 100 ≤ C ∧ C < 1000 ∧ has_property C 1000 → (C = A ∨ C = B) :=
sorry

-- Define a function to represent a number from a sequence of digits
def number_from_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * 10 + a (n - 1 - i)) 0

-- Theorem for the existence of an infinite sequence
theorem exists_infinite_sequence :
  ∃ (a : ℕ → ℕ), ∀ (n : ℕ), has_property (number_from_sequence a n) (10^n) ∧
  ¬(a 0 = 1 ∧ ∀ (k : ℕ), k > 0 → a k = 0) :=
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_property_three_digit_numbers_with_property_exists_infinite_sequence_l1032_103293


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1032_103294

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem: If f(2) = 3 and x = 2 is the axis of symmetry for f(x) = ax^2 + bx + c,
    then the vertex of the parabola is at (2, 3) -/
theorem quadratic_vertex (a b c : ℝ) :
  let f := QuadraticFunction a b c
  f 2 = 3 → -- f(2) = 3
  (∀ x, f (4 - x) = f x) → -- x = 2 is the axis of symmetry
  Vertex.mk 2 3 = Vertex.mk (2 : ℝ) (f 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1032_103294


namespace NUMINAMATH_CALUDE_book_cost_price_l1032_103289

theorem book_cost_price (profit_10 profit_15 additional_profit : ℝ) 
  (h1 : profit_10 = 0.10)
  (h2 : profit_15 = 0.15)
  (h3 : additional_profit = 120) :
  ∃ cost_price : ℝ, 
    cost_price * (1 + profit_15) - cost_price * (1 + profit_10) = additional_profit ∧ 
    cost_price = 2400 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l1032_103289


namespace NUMINAMATH_CALUDE_g_150_zeros_l1032_103288

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 200| - |x + 200|

-- Define gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 1

-- Theorem statement
theorem g_150_zeros :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x : ℝ, x ∈ s ↔ g 150 x = 0 :=
sorry

end NUMINAMATH_CALUDE_g_150_zeros_l1032_103288


namespace NUMINAMATH_CALUDE_y1_greater_y2_l1032_103231

/-- Given two points A(m-1, y₁) and B(m, y₂) on the line y = -2x + 1, prove that y₁ > y₂ -/
theorem y1_greater_y2 (m : ℝ) (y₁ y₂ : ℝ) 
  (hA : y₁ = -2 * (m - 1) + 1) 
  (hB : y₂ = -2 * m + 1) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_y2_l1032_103231


namespace NUMINAMATH_CALUDE_unique_solution_2000_l1032_103234

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ x > 0, deriv y x = Real.log (y x / x)

-- Define the solution y(x) with initial condition
noncomputable def y : ℝ → ℝ :=
  sorry

-- Main theorem
theorem unique_solution_2000 :
  ∃! x : ℝ, x > 0 ∧ y x = 2000 :=
by
  sorry


end NUMINAMATH_CALUDE_unique_solution_2000_l1032_103234


namespace NUMINAMATH_CALUDE_yellow_shirts_count_l1032_103213

theorem yellow_shirts_count (total : ℕ) (blue green red : ℕ) (h1 : total = 36) (h2 : blue = 8) (h3 : green = 11) (h4 : red = 6) :
  total - (blue + green + red) = 11 := by
sorry

end NUMINAMATH_CALUDE_yellow_shirts_count_l1032_103213


namespace NUMINAMATH_CALUDE_modular_inverse_3_mod_197_l1032_103259

theorem modular_inverse_3_mod_197 :
  ∃ x : ℕ, x < 197 ∧ (3 * x) % 197 = 1 :=
by
  use 66
  sorry

end NUMINAMATH_CALUDE_modular_inverse_3_mod_197_l1032_103259


namespace NUMINAMATH_CALUDE_wall_bricks_count_l1032_103232

/-- Represents the time (in hours) it takes Ben to build the wall alone -/
def ben_time : ℝ := 12

/-- Represents the time (in hours) it takes Arya to build the wall alone -/
def arya_time : ℝ := 15

/-- Represents the reduction in combined output (in bricks per hour) due to chattiness -/
def chattiness_reduction : ℝ := 15

/-- Represents the time (in hours) it takes Ben and Arya to build the wall together -/
def combined_time : ℝ := 6

/-- Represents the number of bricks in the wall -/
def wall_bricks : ℝ := 900

theorem wall_bricks_count : 
  ben_time * arya_time * (1 / ben_time + 1 / arya_time - chattiness_reduction / wall_bricks) * combined_time = arya_time + ben_time := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l1032_103232


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1032_103290

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x^2 * Real.exp (abs x) * Real.sin (1 / x^2) else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1032_103290


namespace NUMINAMATH_CALUDE_joan_remaining_oranges_l1032_103227

/-- The number of oranges Joan picked -/
def joan_oranges : ℕ := 37

/-- The number of oranges Sara sold -/
def sara_sold : ℕ := 10

/-- The number of oranges Joan is left with -/
def joan_remaining : ℕ := joan_oranges - sara_sold

theorem joan_remaining_oranges : joan_remaining = 27 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_oranges_l1032_103227


namespace NUMINAMATH_CALUDE_smallest_number_l1032_103236

theorem smallest_number : 
  let numbers : List ℚ := [0, (-3)^2, |-9|, -1^4]
  (∀ x ∈ numbers, -1^4 ≤ x) ∧ (-1^4 ∈ numbers) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1032_103236


namespace NUMINAMATH_CALUDE_salary_percentage_difference_l1032_103272

theorem salary_percentage_difference (raja_salary : ℝ) (ram_salary : ℝ) :
  ram_salary = raja_salary * 1.25 →
  (raja_salary - ram_salary) / ram_salary = -0.2 := by
sorry

end NUMINAMATH_CALUDE_salary_percentage_difference_l1032_103272


namespace NUMINAMATH_CALUDE_smiths_class_a_students_l1032_103229

theorem smiths_class_a_students (johnson_total : ℕ) (johnson_a : ℕ) (smith_total : ℕ) :
  johnson_total = 20 →
  johnson_a = 12 →
  smith_total = 30 →
  (johnson_a : ℚ) / johnson_total = (smith_a : ℚ) / smith_total →
  smith_a = 18 :=
by
  sorry
where
  smith_a : ℕ := sorry

end NUMINAMATH_CALUDE_smiths_class_a_students_l1032_103229


namespace NUMINAMATH_CALUDE_toy_count_l1032_103247

/-- The position of the yellow toy from the left -/
def position_from_left : ℕ := 10

/-- The position of the yellow toy from the right -/
def position_from_right : ℕ := 7

/-- The total number of toys in the row -/
def total_toys : ℕ := position_from_left + position_from_right - 1

theorem toy_count : total_toys = 16 := by
  sorry

end NUMINAMATH_CALUDE_toy_count_l1032_103247


namespace NUMINAMATH_CALUDE_complex_square_in_fourth_quadrant_l1032_103267

theorem complex_square_in_fourth_quadrant :
  let z : ℂ := 2 - I
  (z^2).re > 0 ∧ (z^2).im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_square_in_fourth_quadrant_l1032_103267


namespace NUMINAMATH_CALUDE_product_of_roots_l1032_103215

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 18 → ∃ y : ℝ, (x + 3) * (x - 5) = 18 ∧ (y + 3) * (y - 5) = 18 ∧ x * y = -33 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1032_103215


namespace NUMINAMATH_CALUDE_fruit_merchant_problem_l1032_103226

/-- Fruit merchant problem -/
theorem fruit_merchant_problem 
  (total_cost : ℝ) 
  (quantity : ℝ) 
  (cost_difference : ℝ) 
  (large_selling_price : ℝ) 
  (small_selling_price : ℝ) 
  (loss_percentage : ℝ) 
  (earnings_percentage : ℝ) 
  (h1 : total_cost = 8000) 
  (h2 : quantity = 200) 
  (h3 : cost_difference = 20) 
  (h4 : large_selling_price = 40) 
  (h5 : small_selling_price = 16) 
  (h6 : loss_percentage = 0.2) 
  (h7 : earnings_percentage = 0.9) :
  ∃ (small_cost large_cost earnings min_large_price : ℝ),
    small_cost = 10 ∧ 
    large_cost = 30 ∧ 
    earnings = 3200 ∧ 
    min_large_price = 41.6 ∧
    quantity * small_cost + quantity * large_cost = total_cost ∧
    large_cost = small_cost + cost_difference ∧
    earnings = quantity * (large_selling_price - large_cost) + quantity * (small_selling_price - small_cost) ∧
    quantity * min_large_price + small_selling_price * quantity * (1 - loss_percentage) - total_cost ≥ earnings * earnings_percentage :=
by sorry

end NUMINAMATH_CALUDE_fruit_merchant_problem_l1032_103226


namespace NUMINAMATH_CALUDE_solve_system_with_partial_info_l1032_103298

/-- Given a system of linear equations and information about its solutions,
    this theorem proves the values of the coefficients. -/
theorem solve_system_with_partial_info :
  ∀ (a b c : ℚ),
  (∀ x y : ℚ, a*x + b*y = 2 ∧ c*x - 3*y = -2 → x = 1 ∧ y = -1) →
  (a*2 + b*(-6) = 2) →
  (a = 5/2 ∧ b = 1/2 ∧ c = -5) :=
by sorry

end NUMINAMATH_CALUDE_solve_system_with_partial_info_l1032_103298


namespace NUMINAMATH_CALUDE_coloring_theorem_l1032_103285

/-- Given finite sets A, B, C, D, this function calculates the number of ways
    to color three adjacent elements with the condition that adjacent elements
    must have different colors. -/
def colorThreeAdjacent (A B C : Finset α) : ℕ :=
  A.card * (B.card - 1) * (C.card - 1)

/-- Given finite sets A, B, C, D, this function calculates the number of ways
    to color four adjacent elements with the condition that adjacent elements
    must have different colors. -/
def colorFourAdjacent (A B C D : Finset α) : ℕ :=
  A.card * (B.card - 1) * (C.card - 1) * (D.card - 1)

theorem coloring_theorem (A B C D : Finset α) :
  (colorThreeAdjacent A B C = A.card * (B.card - 1) * (C.card - 1)) ∧
  (colorFourAdjacent A B C D = A.card * (B.card - 1) * (C.card - 1) * (D.card - 1)) := by
  sorry

end NUMINAMATH_CALUDE_coloring_theorem_l1032_103285


namespace NUMINAMATH_CALUDE_time_to_clean_wall_l1032_103258

/-- Represents the dimensions of the wall in large squares -/
structure WallDimensions where
  height : ℕ
  width : ℕ

/-- Represents the cleaning progress and rate -/
structure CleaningProgress where
  totalArea : ℕ
  cleanedArea : ℕ
  timeSpent : ℕ

/-- Calculates the time needed to clean the remaining area -/
def timeToCleanRemaining (wall : WallDimensions) (progress : CleaningProgress) : ℕ :=
  let remainingArea := wall.height * wall.width - progress.cleanedArea
  (remainingArea * progress.timeSpent) / progress.cleanedArea

/-- Theorem: Given the wall dimensions and cleaning progress, 
    the time to clean the remaining area is 161 minutes -/
theorem time_to_clean_wall 
  (wall : WallDimensions) 
  (progress : CleaningProgress) 
  (h1 : wall.height = 6) 
  (h2 : wall.width = 12) 
  (h3 : progress.totalArea = wall.height * wall.width)
  (h4 : progress.cleanedArea = 9)
  (h5 : progress.timeSpent = 23) :
  timeToCleanRemaining wall progress = 161 := by
  sorry

end NUMINAMATH_CALUDE_time_to_clean_wall_l1032_103258


namespace NUMINAMATH_CALUDE_playground_paint_ratio_l1032_103208

/-- Given a square playground with side length s and diagonal paint lines of width w,
    if one-third of the playground's area is covered in paint,
    then the ratio of s to w is 3/2. -/
theorem playground_paint_ratio (s w : ℝ) (h_positive : s > 0 ∧ w > 0) 
    (h_paint_area : w^2 + (s - w)^2 / 2 = s^2 / 3) : s / w = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_playground_paint_ratio_l1032_103208


namespace NUMINAMATH_CALUDE_sequence_shortening_l1032_103252

/-- A sequence of digits where each digit is independently chosen from {0, 9} -/
def DigitSequence := Fin 2015 → Fin 10

/-- The probability of a digit being 0 or 9 -/
def p : ℝ := 0.1

/-- The number of digits in the original sequence -/
def n : ℕ := 2015

/-- The number of digits that can potentially be removed -/
def k : ℕ := 2014

theorem sequence_shortening (seq : DigitSequence) :
  /- The probability of the sequence shortening by exactly one digit -/
  (Nat.choose k 1 : ℝ) * p^1 * (1 - p)^(k - 1) = 
    (2014 : ℝ) * 0.1 * 0.9^2013 ∧
  /- The expected length of the new sequence -/
  (n : ℝ) - (k : ℝ) * p = 1813.6 := by
  sorry


end NUMINAMATH_CALUDE_sequence_shortening_l1032_103252


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l1032_103201

noncomputable def P (α β : ℝ) (x : ℝ) : ℝ := α * x^4 + α * x^3 + α * x^2 + α * x + β

noncomputable def Q (α : ℝ) (x : ℝ) : ℝ := α * x^3 + α * x

theorem polynomial_equation_solution (α β : ℝ) (hα : α ≠ 0) :
  (∀ x : ℝ, P α β (x^2) + Q α x = P α β x + x^5 * Q α x) ∧
  (∀ P' Q' : ℝ → ℝ, (∀ x : ℝ, P' (x^2) + Q' x = P' x + x^5 * Q' x) →
    (∃ c : ℝ, P' = P (c * α) (c * β) ∧ Q' = Q (c * α))) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l1032_103201


namespace NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l1032_103200

-- Define factorial for natural numbers
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem ten_factorial_mod_thirteen : factorial 10 % 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l1032_103200


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_solve_inequality_l1032_103214

def f (a x : ℝ) : ℝ := a * x^2 + (1 + a) * x + a

theorem quadratic_inequality_solutions (a : ℝ) :
  (∃ x : ℝ, f a x ≥ 0) ↔ a ≥ -1/3 :=
sorry

theorem solve_inequality (a : ℝ) (h : a > 0) :
  {x : ℝ | f a x < a - 1} =
    if a < 1 then {x : ℝ | -1/a < x ∧ x < -1}
    else if a = 1 then ∅
    else {x : ℝ | -1 < x ∧ x < -1/a} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_solve_inequality_l1032_103214


namespace NUMINAMATH_CALUDE_subset_implies_a_leq_two_l1032_103203

theorem subset_implies_a_leq_two (a : ℝ) : 
  let A : Set ℝ := {x | x ≥ a}
  let B : Set ℝ := {x | |x - 3| < 1}
  B ⊆ A → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_leq_two_l1032_103203


namespace NUMINAMATH_CALUDE_no_real_root_in_unit_interval_l1032_103296

theorem no_real_root_in_unit_interval (a b c d : ℝ) :
  (min d (b + d) > max (abs c) (abs (a + c))) →
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → (a * x^3 + b * x^2 + c * x + d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_root_in_unit_interval_l1032_103296


namespace NUMINAMATH_CALUDE_jar_capacity_ratio_l1032_103251

theorem jar_capacity_ratio (capacity_x capacity_y : ℝ) : 
  capacity_x > 0 → 
  capacity_y > 0 → 
  (1/2 : ℝ) * capacity_x + (1/2 : ℝ) * capacity_y = (3/4 : ℝ) * capacity_x → 
  capacity_y / capacity_x = (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_jar_capacity_ratio_l1032_103251


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1032_103286

/-- Given a triangle with sides 10, 24, and 26 units, and a rectangle with width 8 units
    and area equal to the triangle's area, the perimeter of the rectangle is 46 units. -/
theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) (h4 : w = 8)
  (h5 : w * (a * b / 2 / w) = a * b / 2) : 2 * (w + (a * b / 2 / w)) = 46 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1032_103286


namespace NUMINAMATH_CALUDE_parabola_focus_and_intersection_l1032_103244

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line
def line (n : ℝ) (x y : ℝ) : Prop := x = Real.sqrt 3 * y + n

-- Define the point E
def point_E : ℝ × ℝ := (4, 4)

-- Define the point D
def point_D (n : ℝ) : ℝ × ℝ := (n, 0)

-- Theorem statement
theorem parabola_focus_and_intersection
  (p : ℝ)
  (n : ℝ)
  (h1 : parabola p (point_E.1) (point_E.2))
  (h2 : ∃ (A B : ℝ × ℝ), A ≠ B ∧ A ≠ point_E ∧ B ≠ point_E ∧
        parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
        line n A.1 A.2 ∧ line n B.1 B.2)
  (h3 : ∃ (A B : ℝ × ℝ), 
        parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
        line n A.1 A.2 ∧ line n B.1 B.2 ∧
        (A.1 - (point_D n).1)^2 + A.2^2 * ((B.1 - (point_D n).1)^2 + B.2^2) = 64) :
  (p = 2 ∧ n = 4) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_and_intersection_l1032_103244


namespace NUMINAMATH_CALUDE_total_cash_realized_eq_17364_82065_l1032_103233

/-- Calculates the total cash realized in INR from selling four stocks -/
def total_cash_realized (stock1_value stock1_brokerage stock2_value stock2_brokerage : ℚ)
                        (stock3_value stock3_brokerage stock4_value stock4_brokerage : ℚ)
                        (usd_to_inr : ℚ) : ℚ :=
  let stock1_realized := stock1_value * (1 - stock1_brokerage / 100)
  let stock2_realized := stock2_value * (1 - stock2_brokerage / 100)
  let stock3_realized := stock3_value * (1 - stock3_brokerage / 100) * usd_to_inr
  let stock4_realized := stock4_value * (1 - stock4_brokerage / 100) * usd_to_inr
  stock1_realized + stock2_realized + stock3_realized + stock4_realized

/-- Theorem stating that the total cash realized is equal to 17364.82065 INR -/
theorem total_cash_realized_eq_17364_82065 :
  total_cash_realized 120.50 (1/4) 210.75 0.5 80.90 0.3 150.55 0.65 74 = 17364.82065 := by
  sorry

end NUMINAMATH_CALUDE_total_cash_realized_eq_17364_82065_l1032_103233


namespace NUMINAMATH_CALUDE_price_of_added_toy_l1032_103261

/-- Given 5 toys with an average price of $10, adding one toy to make the new average $11 for 6 toys, prove the price of the added toy is $16. -/
theorem price_of_added_toy (num_toys : ℕ) (avg_price : ℚ) (new_num_toys : ℕ) (new_avg_price : ℚ) :
  num_toys = 5 →
  avg_price = 10 →
  new_num_toys = num_toys + 1 →
  new_avg_price = 11 →
  (new_num_toys : ℚ) * new_avg_price - (num_toys : ℚ) * avg_price = 16 := by
  sorry

end NUMINAMATH_CALUDE_price_of_added_toy_l1032_103261


namespace NUMINAMATH_CALUDE_eliza_basketball_scores_l1032_103273

def first_ten_games : List Nat := [9, 3, 5, 4, 8, 2, 5, 3, 7, 6]

def total_first_ten : Nat := first_ten_games.sum

theorem eliza_basketball_scores :
  ∃ (game11 game12 : Nat),
    game11 < 10 ∧
    game12 < 10 ∧
    (total_first_ten + game11) % 11 = 0 ∧
    (total_first_ten + game11 + game12) % 12 = 0 ∧
    game11 * game12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eliza_basketball_scores_l1032_103273


namespace NUMINAMATH_CALUDE_jane_ice_cream_pudding_cost_difference_l1032_103230

theorem jane_ice_cream_pudding_cost_difference :
  let ice_cream_cones : ℕ := 15
  let pudding_cups : ℕ := 5
  let ice_cream_cost_per_cone : ℕ := 5
  let pudding_cost_per_cup : ℕ := 2
  let total_ice_cream_cost := ice_cream_cones * ice_cream_cost_per_cone
  let total_pudding_cost := pudding_cups * pudding_cost_per_cup
  total_ice_cream_cost - total_pudding_cost = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_jane_ice_cream_pudding_cost_difference_l1032_103230


namespace NUMINAMATH_CALUDE_tangent_circle_min_radius_l1032_103240

noncomputable section

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P on curve C
def P (x₀ y₀ : ℝ) : Prop := C x₀ y₀ ∧ y₀ > 0

-- Define the line l tangent to C at P
def l (x₀ y₀ k : ℝ) (x y : ℝ) : Prop := y - y₀ = k * (x - x₀)

-- Define the circle M centered at (a, 0)
def M (a r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = r^2

-- Main theorem
theorem tangent_circle_min_radius (a x₀ y₀ k r : ℝ) :
  a > 2 →
  P x₀ y₀ →
  (∀ x y, C x y → l x₀ y₀ k x y → x = x₀ ∧ y = y₀) →
  (∃ x y, l x₀ y₀ k x y ∧ M a r x y) →
  (∀ r' : ℝ, (∃ x y, l x₀ y₀ k x y ∧ M a r' x y) → r ≤ r') →
  a - x₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_min_radius_l1032_103240


namespace NUMINAMATH_CALUDE_car_cost_l1032_103202

/-- Calculates the cost of the car given the costs of other gifts and total worth --/
theorem car_cost (ring_cost bracelet_cost total_worth : ℕ) 
  (h1 : ring_cost = 4000)
  (h2 : bracelet_cost = 2 * ring_cost)
  (h3 : total_worth = 14000) :
  total_worth - (ring_cost + bracelet_cost) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_l1032_103202


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l1032_103282

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number represents the total number of coins in a stack with 30 layers -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l1032_103282


namespace NUMINAMATH_CALUDE_area_and_inequality_l1032_103281

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x - a) - x + a

-- State the theorem
theorem area_and_inequality (a : ℝ) (h : a > 0) :
  (∃ A : ℝ, A = (8:ℝ)/3 ∧ A = ∫ x in (2*a/3)..(2*a), (f a x - a)) → a = 2 ∧
  (∀ x : ℝ, f a x > x ↔ x < 3*a/4) :=
sorry

end NUMINAMATH_CALUDE_area_and_inequality_l1032_103281


namespace NUMINAMATH_CALUDE_rectangle_strip_proof_l1032_103222

theorem rectangle_strip_proof (a b c : ℕ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b + a * c + a * (b - a) + a * a + a * (c - a) = 43 →
  (a = 1 ∧ b + c = 22) ∨ (a = 1 ∧ c + b = 22) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_strip_proof_l1032_103222


namespace NUMINAMATH_CALUDE_existence_of_constant_sequence_l1032_103278

/-- An irreducible polynomial with integer coefficients -/
def IrreducibleIntPoly := Polynomial ℤ

/-- The number of solutions to p(x) ≡ 0 mod q^n -/
def num_solutions (p : IrreducibleIntPoly) (q : ℕ) (n : ℕ) : ℕ := sorry

theorem existence_of_constant_sequence 
  (p : IrreducibleIntPoly) 
  (q : ℕ) 
  (h_q : Nat.Prime q) :
  ∃ M : ℕ, ∀ n ≥ M, num_solutions p q n = num_solutions p q M := by
  sorry

end NUMINAMATH_CALUDE_existence_of_constant_sequence_l1032_103278


namespace NUMINAMATH_CALUDE_equilateral_pyramid_volume_l1032_103216

/-- A pyramid with an equilateral triangle base -/
structure EquilateralPyramid where
  -- The side length of the base triangle
  base_side : ℝ
  -- The angle between two edges from the apex to the base
  apex_angle : ℝ

/-- The volume of an equilateral pyramid -/
noncomputable def volume (p : EquilateralPyramid) : ℝ :=
  (Real.sqrt 3 / 9) * (2 / 3 * Real.sqrt 3 + 1 / Real.tan (p.apex_angle / 2))

/-- Theorem: The volume of a specific equilateral pyramid -/
theorem equilateral_pyramid_volume :
    ∀ (p : EquilateralPyramid),
      p.base_side = 2 →
      volume p = (Real.sqrt 3 / 9) * (2 / 3 * Real.sqrt 3 + 1 / Real.tan (p.apex_angle / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_pyramid_volume_l1032_103216


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l1032_103223

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l1032_103223


namespace NUMINAMATH_CALUDE_gasoline_price_change_l1032_103220

/-- The price of gasoline after five months of changes -/
def final_price (initial_price : ℝ) : ℝ :=
  initial_price * 1.30 * 0.75 * 1.10 * 0.85 * 0.80

/-- Theorem stating the relationship between the initial and final price -/
theorem gasoline_price_change (initial_price : ℝ) :
  final_price initial_price = 102.60 → initial_price = 140.67 := by
  sorry

#eval final_price 140.67

end NUMINAMATH_CALUDE_gasoline_price_change_l1032_103220


namespace NUMINAMATH_CALUDE_charles_total_money_l1032_103228

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of pennies Charles found -/
def pennies_found : ℕ := 6

/-- The number of nickels Charles had at home -/
def nickels_at_home : ℕ := 3

/-- Theorem stating that the total value of Charles' coins is 21 cents -/
theorem charles_total_money : 
  pennies_found * penny_value + nickels_at_home * nickel_value = 21 := by
  sorry

end NUMINAMATH_CALUDE_charles_total_money_l1032_103228


namespace NUMINAMATH_CALUDE_starting_number_proof_l1032_103284

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem starting_number_proof (n : ℕ) : 
  (∃! m : ℕ, m > n ∧ m < 580 ∧ 
   (∃ l : List ℕ, l.length = 6 ∧ 
    (∀ x ∈ l, x > n ∧ x < 580 ∧ is_divisible_by x 45 ∧ is_divisible_by x 6) ∧
    (∀ y : ℕ, y > n ∧ y < 580 ∧ is_divisible_by y 45 ∧ is_divisible_by y 6 → y ∈ l))) →
  (∀ k : ℕ, k > n → 
    ¬(∃ l : List ℕ, l.length = 6 ∧ 
      (∀ x ∈ l, x > k ∧ x < 580 ∧ is_divisible_by x 45 ∧ is_divisible_by x 6) ∧
      (∀ y : ℕ, y > k ∧ y < 580 ∧ is_divisible_by y 45 ∧ is_divisible_by y 6 → y ∈ l))) →
  n = 450 := by
sorry

end NUMINAMATH_CALUDE_starting_number_proof_l1032_103284


namespace NUMINAMATH_CALUDE_percentage_of_500_l1032_103292

/-- Prove that 25% of Rs. 500 is equal to Rs. 125 -/
theorem percentage_of_500 : (500 : ℝ) * 0.25 = 125 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_500_l1032_103292


namespace NUMINAMATH_CALUDE_seven_digit_palindrome_count_l1032_103270

/-- A seven-digit palindrome is a number of the form abcdcba where a, b, c, d are digits and a ≠ 0 -/
def SevenDigitPalindrome : Type := ℕ

/-- The count of valid digits for the first position of a seven-digit palindrome -/
def FirstDigitCount : ℕ := 9

/-- The count of valid digits for each of the second, third, and fourth positions of a seven-digit palindrome -/
def OtherDigitCount : ℕ := 10

/-- The total number of seven-digit palindromes -/
def TotalSevenDigitPalindromes : ℕ := FirstDigitCount * OtherDigitCount * OtherDigitCount * OtherDigitCount

theorem seven_digit_palindrome_count : TotalSevenDigitPalindromes = 9000 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_palindrome_count_l1032_103270


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l1032_103299

theorem cow_chicken_problem (c h : ℕ) : 
  (4 * c + 2 * h = 2 * (c + h) + 18) → c = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l1032_103299


namespace NUMINAMATH_CALUDE_evaluate_expression_l1032_103250

theorem evaluate_expression (a b : ℝ) (h1 : a = 5) (h2 : b = 6) :
  3 / (2 * a + b) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1032_103250


namespace NUMINAMATH_CALUDE_min_value_of_f_l1032_103209

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- State the theorem
theorem min_value_of_f :
  ∃ (y : ℝ), (∀ (x : ℝ), x ≥ 0 → f x ≥ y) ∧ (∃ (x : ℝ), x ≥ 0 ∧ f x = y) ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1032_103209


namespace NUMINAMATH_CALUDE_triangle_point_distance_height_inequality_l1032_103235

/-- Given a triangle and a point M inside it, this theorem states that the sum of the α-th powers
    of the ratios of distances from M to the sides to the corresponding heights of the triangle
    is always greater than or equal to 1/3ᵅ⁻¹, for α ≥ 1. -/
theorem triangle_point_distance_height_inequality
  (α : ℝ) (h_α : α ≥ 1)
  (k₁ k₂ k₃ h₁ h₂ h₃ : ℝ)
  (h_positive : k₁ > 0 ∧ k₂ > 0 ∧ k₃ > 0 ∧ h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0)
  (h_sum : k₁/h₁ + k₂/h₂ + k₃/h₃ = 1) :
  (k₁/h₁)^α + (k₂/h₂)^α + (k₃/h₃)^α ≥ 1/(3^(α-1)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_point_distance_height_inequality_l1032_103235


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1032_103295

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the properties of the triangle
def isOnCircumference (P Q : ℝ × ℝ) : Prop := sorry

def angleEqual (P Q R : ℝ × ℝ) : Prop := sorry

def distance (A B : ℝ × ℝ) : ℝ := sorry

def perimeter (t : Triangle) : ℝ :=
  distance t.P t.Q + distance t.Q t.R + distance t.R t.P

-- State the theorem
theorem triangle_perimeter (t : Triangle) :
  isOnCircumference t.P t.Q →
  angleEqual t.P t.Q t.R →
  distance t.Q t.R = 8 →
  distance t.P t.R = 10 →
  perimeter t = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1032_103295


namespace NUMINAMATH_CALUDE_root_is_factor_l1032_103276

theorem root_is_factor (P : ℝ → ℝ) (a : ℝ) :
  (P a = 0) → ∃ Q : ℝ → ℝ, ∀ x, P x = (x - a) * Q x := by
  sorry

end NUMINAMATH_CALUDE_root_is_factor_l1032_103276


namespace NUMINAMATH_CALUDE_distribute_items_eq_36_l1032_103263

/-- The number of ways to distribute 4 distinct items into 3 non-empty groups -/
def distribute_items : ℕ :=
  (Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 1 1) * Nat.factorial 3 / Nat.factorial 2

/-- Theorem stating that the number of ways to distribute 4 distinct items
    into 3 non-empty groups is 36 -/
theorem distribute_items_eq_36 : distribute_items = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_items_eq_36_l1032_103263


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1032_103264

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem tenth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 5/6) (h₃ : a₃ = 7/6) :
  arithmetic_sequence a₁ (a₂ - a₁) 10 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1032_103264


namespace NUMINAMATH_CALUDE_system_solution_l1032_103221

theorem system_solution (x y : ℝ) (eq1 : x + 5 * y = 6) (eq2 : 3 * x - y = 2) : 
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1032_103221


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1032_103271

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + m = 0 ∧ x₂^2 - 3*x₂ + m = 0) → 
  m = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1032_103271


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1032_103207

theorem coefficient_x_squared_in_expansion : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  (∀ x : ℝ, (x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) → 
  a₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1032_103207


namespace NUMINAMATH_CALUDE_max_contribution_scenario_l1032_103269

/-- Represents the maximum possible contribution by a single person given the total contribution and number of people. -/
def max_contribution (total : ℝ) (num_people : ℕ) (min_contribution : ℝ) : ℝ :=
  total - (min_contribution * (num_people - 1 : ℝ))

/-- Theorem stating the maximum possible contribution in the given scenario. -/
theorem max_contribution_scenario :
  max_contribution 20 10 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_contribution_scenario_l1032_103269


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1032_103280

/-- Represents a geometric sequence with first term a and common ratio q -/
def GeometricSequence (a : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a * q ^ (n - 1)

/-- The common ratio of a geometric sequence satisfying given conditions is 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℝ) (q : ℝ) (h_pos : q > 0) :
  let seq := GeometricSequence a q
  (seq 3 - 3 * seq 2 = 2) ∧ 
  (5 * seq 4 = (12 * seq 3 + 2 * seq 5) / 2) →
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1032_103280


namespace NUMINAMATH_CALUDE_max_quadrilateral_intersections_l1032_103279

/-- A quadrilateral is a polygon with 4 sides -/
def Quadrilateral : Type := Unit

/-- The number of sides in a quadrilateral -/
def num_sides (q : Quadrilateral) : ℕ := 4

/-- The maximum number of intersection points between two quadrilaterals -/
def max_intersection_points (q1 q2 : Quadrilateral) : ℕ :=
  num_sides q1 * num_sides q2

theorem max_quadrilateral_intersections :
  ∀ (q1 q2 : Quadrilateral), max_intersection_points q1 q2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_quadrilateral_intersections_l1032_103279


namespace NUMINAMATH_CALUDE_smallest_m_no_real_roots_l1032_103262

theorem smallest_m_no_real_roots : 
  ∀ m : ℤ, (∀ x : ℝ, 3*x*(m*x-5) - 2*x^2 + 7 ≠ 0) → m ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_no_real_roots_l1032_103262


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l1032_103275

noncomputable def f (x : ℝ) := x^3 + 3*x^2 - 9*x + 1

theorem f_extrema_on_interval :
  let a := -4
  let b := 4
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 77 ∧ f x_min = -4 :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l1032_103275


namespace NUMINAMATH_CALUDE_older_ate_twelve_l1032_103210

/-- Represents the pancake eating scenario --/
structure PancakeScenario where
  initial_pancakes : ℕ
  final_pancakes : ℕ
  younger_eats : ℕ
  older_eats : ℕ
  grandma_bakes : ℕ

/-- Calculates the number of pancakes eaten by the older grandchild --/
def older_grandchild_pancakes (scenario : PancakeScenario) : ℕ :=
  let net_reduction := scenario.younger_eats + scenario.older_eats - scenario.grandma_bakes
  let cycles := (scenario.initial_pancakes - scenario.final_pancakes) / net_reduction
  scenario.older_eats * cycles

/-- Theorem stating that the older grandchild ate 12 pancakes in the given scenario --/
theorem older_ate_twelve (scenario : PancakeScenario) 
  (h1 : scenario.initial_pancakes = 19)
  (h2 : scenario.final_pancakes = 11)
  (h3 : scenario.younger_eats = 1)
  (h4 : scenario.older_eats = 3)
  (h5 : scenario.grandma_bakes = 2) :
  older_grandchild_pancakes scenario = 12 := by
  sorry

end NUMINAMATH_CALUDE_older_ate_twelve_l1032_103210
