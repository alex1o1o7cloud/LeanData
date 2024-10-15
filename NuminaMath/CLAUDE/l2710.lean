import Mathlib

namespace NUMINAMATH_CALUDE_solution_value_l2710_271098

theorem solution_value (x y m : ℝ) : x - 2*y = m → x = 2 → y = 1 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2710_271098


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l2710_271008

/-- The average speed of a round trip given different speeds for each direction -/
theorem round_trip_average_speed (speed_to_school speed_from_school : ℝ) :
  speed_to_school > 0 →
  speed_from_school > 0 →
  let average_speed := 2 / (1 / speed_to_school + 1 / speed_from_school)
  average_speed = 4.8 ↔ speed_to_school = 6 ∧ speed_from_school = 4 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l2710_271008


namespace NUMINAMATH_CALUDE_tamika_always_greater_l2710_271036

def tamika_set : Set ℕ := {6, 7, 8}
def carlos_set : Set ℕ := {2, 4, 5}

def tamika_product (a b : ℕ) : Prop := a ∈ tamika_set ∧ b ∈ tamika_set ∧ a ≠ b
def carlos_product (c d : ℕ) : Prop := c ∈ carlos_set ∧ d ∈ carlos_set ∧ c ≠ d

theorem tamika_always_greater :
  ∀ (a b c d : ℕ), tamika_product a b → carlos_product c d →
    a * b > c * d :=
sorry

end NUMINAMATH_CALUDE_tamika_always_greater_l2710_271036


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l2710_271001

-- Define the function f(x) = |x + 2|
def f (x : ℝ) : ℝ := |x + 2|

-- State the theorem
theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
  (x₁ - x₂) * (f x₁ - f x₂) > 0 :=
by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l2710_271001


namespace NUMINAMATH_CALUDE_not_always_greater_than_original_l2710_271077

theorem not_always_greater_than_original : ¬ (∀ x : ℝ, 1.25 * x > x) := by sorry

end NUMINAMATH_CALUDE_not_always_greater_than_original_l2710_271077


namespace NUMINAMATH_CALUDE_complement_A_union_B_l2710_271049

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B : Set ℝ := {y | ∃ x, y = |x|}

-- State the theorem
theorem complement_A_union_B :
  (Aᶜ ∪ B) = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l2710_271049


namespace NUMINAMATH_CALUDE_penny_nickel_dime_heads_prob_l2710_271029

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (half_dollar : CoinOutcome)

/-- The probability of getting heads on the penny, nickel, and dime when flipping five coins -/
def prob_penny_nickel_dime_heads : ℚ :=
  1 / 8

/-- Theorem stating that the probability of getting heads on the penny, nickel, and dime
    when flipping five coins simultaneously is 1/8 -/
theorem penny_nickel_dime_heads_prob :
  prob_penny_nickel_dime_heads = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_penny_nickel_dime_heads_prob_l2710_271029


namespace NUMINAMATH_CALUDE_cracked_to_broken_ratio_l2710_271086

/-- Represents the number of eggs in each category --/
structure EggCounts where
  total : ℕ
  broken : ℕ
  perfect : ℕ
  cracked : ℕ

/-- Theorem stating the ratio of cracked to broken eggs --/
theorem cracked_to_broken_ratio (e : EggCounts) : 
  e.total = 24 →
  e.broken = 3 →
  e.perfect - e.cracked = 9 →
  e.total = e.perfect + e.cracked + e.broken →
  (e.cracked : ℚ) / e.broken = 2 := by
  sorry

#check cracked_to_broken_ratio

end NUMINAMATH_CALUDE_cracked_to_broken_ratio_l2710_271086


namespace NUMINAMATH_CALUDE_square_area_increase_when_side_tripled_l2710_271063

theorem square_area_increase_when_side_tripled :
  ∀ (s : ℝ), s > 0 →
  (3 * s)^2 = 9 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_when_side_tripled_l2710_271063


namespace NUMINAMATH_CALUDE_pinecone_count_l2710_271099

theorem pinecone_count (initial : ℕ) : 
  (initial : ℝ) * 0.2 = initial * 0.2 ∧  -- 20% eaten by reindeer
  (initial : ℝ) * 0.4 = 2 * (initial * 0.2) ∧  -- Twice as many eaten by squirrels
  (initial : ℝ) * 0.25 * 0.4 = initial * 0.1 ∧  -- 25% of remainder collected for fires
  (initial : ℝ) * 0.3 = 600 →  -- 600 pinecones left
  initial = 2000 := by
sorry

end NUMINAMATH_CALUDE_pinecone_count_l2710_271099


namespace NUMINAMATH_CALUDE_round_trip_distance_l2710_271064

/-- Calculates the total distance of a round trip journey given speeds and times -/
theorem round_trip_distance 
  (speed_to : ℝ) 
  (speed_from : ℝ) 
  (time_to : ℝ) 
  (time_from : ℝ) 
  (h1 : speed_to = 4)
  (h2 : speed_from = 3)
  (h3 : time_to = 30 / 60)
  (h4 : time_from = 40 / 60) :
  speed_to * time_to + speed_from * time_from = 4 := by
  sorry

#check round_trip_distance

end NUMINAMATH_CALUDE_round_trip_distance_l2710_271064


namespace NUMINAMATH_CALUDE_price_increase_percentage_l2710_271050

/-- Proves that the percentage increase in prices is 15% given the problem conditions -/
theorem price_increase_percentage (orange_price : ℝ) (mango_price : ℝ) (new_total_cost : ℝ) :
  orange_price = 40 →
  mango_price = 50 →
  new_total_cost = 1035 →
  10 * (orange_price * (1 + 15 / 100)) + 10 * (mango_price * (1 + 15 / 100)) = new_total_cost :=
by sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l2710_271050


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2710_271070

/-- Given a quadratic function y = ax^2 + bx - 1 where a ≠ 0 and 
    the graph passes through the point (1, 1), prove that 1 - a - b = -1 -/
theorem quadratic_function_property (a b : ℝ) (h1 : a ≠ 0) 
    (h2 : a * 1^2 + b * 1 - 1 = 1) : 1 - a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2710_271070


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l2710_271067

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  geometric_sequence a → a 3 = -3 → a 4 = 6 → a 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l2710_271067


namespace NUMINAMATH_CALUDE_pentagon_area_theorem_l2710_271054

/-- A pentagon is a polygon with 5 sides -/
structure Pentagon where
  sides : Fin 5 → ℝ

/-- The area of a pentagon -/
noncomputable def Pentagon.area (p : Pentagon) : ℝ := sorry

/-- Theorem: There exists a pentagon with sides 18, 25, 30, 28, and 25 units, and its area is 950 square units -/
theorem pentagon_area_theorem : 
  ∃ (p : Pentagon), 
    p.sides 0 = 18 ∧ 
    p.sides 1 = 25 ∧ 
    p.sides 2 = 30 ∧ 
    p.sides 3 = 28 ∧ 
    p.sides 4 = 25 ∧ 
    p.area = 950 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_theorem_l2710_271054


namespace NUMINAMATH_CALUDE_number_of_students_in_class_l2710_271012

/-- Proves that the number of students in a class is 23 given certain grade conditions --/
theorem number_of_students_in_class 
  (recorded_biology : ℝ) 
  (recorded_chemistry : ℝ)
  (actual_biology : ℝ) 
  (actual_chemistry : ℝ)
  (subject_weight : ℝ)
  (class_average_increase : ℝ)
  (initial_class_average : ℝ)
  (h1 : recorded_biology = 83)
  (h2 : recorded_chemistry = 85)
  (h3 : actual_biology = 70)
  (h4 : actual_chemistry = 75)
  (h5 : subject_weight = 0.5)
  (h6 : class_average_increase = 0.5)
  (h7 : initial_class_average = 80) :
  ∃ n : ℕ, n = 23 ∧ n * class_average_increase = (recorded_biology * subject_weight + recorded_chemistry * subject_weight) - (actual_biology * subject_weight + actual_chemistry * subject_weight) := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_in_class_l2710_271012


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2710_271094

theorem absolute_value_inequality (x : ℝ) :
  |x + 2| + |x + 3| ≤ 2 ↔ -7/2 ≤ x ∧ x ≤ -3/2 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2710_271094


namespace NUMINAMATH_CALUDE_graduating_class_size_l2710_271079

theorem graduating_class_size :
  let num_boys : ℕ := 138
  let girls_more_than_boys : ℕ := 69
  let num_girls : ℕ := num_boys + girls_more_than_boys
  let total_students : ℕ := num_boys + num_girls
  total_students = 345 := by sorry

end NUMINAMATH_CALUDE_graduating_class_size_l2710_271079


namespace NUMINAMATH_CALUDE_square_sum_inequality_l2710_271005

theorem square_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  a^2 + b^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l2710_271005


namespace NUMINAMATH_CALUDE_range_of_a_l2710_271026

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing on [1,5]
def IsIncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 1 5 → y ∈ Set.Icc 1 5 → x < y → f x < f y

-- Define the theorem
theorem range_of_a (h1 : IsIncreasingOn f) 
  (h2 : ∀ a, f (a + 1) < f (2 * a - 1)) :
  ∃ a, a ∈ Set.Ioo 2 3 ∧ 
    (∀ x, x ∈ Set.Ioo 2 3 → 
      (f (x + 1) < f (2 * x - 1) ∧ 
       x + 1 ∈ Set.Icc 1 5 ∧ 
       2 * x - 1 ∈ Set.Icc 1 5)) :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_l2710_271026


namespace NUMINAMATH_CALUDE_smallest_composite_with_large_factors_l2710_271027

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_with_large_factors : 
  (is_composite 323) ∧ 
  (has_no_small_prime_factors 323) ∧ 
  (∀ m : ℕ, m < 323 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_with_large_factors_l2710_271027


namespace NUMINAMATH_CALUDE_min_sum_distances_l2710_271017

/-- The minimum value of PA + PB for a point P on the parabola y² = 4x -/
theorem min_sum_distances (y : ℝ) : 
  let x := y^2 / 4
  let PA := x
  let PB := |x - y + 4| / Real.sqrt 2
  (∀ y', (y'^2 / 4 - y' / Real.sqrt 2 + 2 * Real.sqrt 2) ≥ 
         (y^2 / 4 - y / Real.sqrt 2 + 2 * Real.sqrt 2)) →
  PA + PB = 5 * Real.sqrt 2 / 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_min_sum_distances_l2710_271017


namespace NUMINAMATH_CALUDE_french_students_count_l2710_271080

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 87)
  (h_german : german = 22)
  (h_both : both = 9)
  (h_neither : neither = 33) :
  ∃ french : ℕ, french = total - german + both - neither :=
by
  sorry

end NUMINAMATH_CALUDE_french_students_count_l2710_271080


namespace NUMINAMATH_CALUDE_zoo_animals_l2710_271066

theorem zoo_animals (M B L : ℕ) : 
  (26 ≤ M + B + L) ∧ (M + B + L ≤ 32) ∧
  (M + L > B) ∧
  (B + L = 2 * M) ∧
  (M + B = 3 * L + 3) ∧
  (2 * B = L) →
  B = 13 := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_l2710_271066


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l2710_271031

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ f' x = 2 ↔ (x = 1 ∧ y = 3) ∨ (x = -1 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l2710_271031


namespace NUMINAMATH_CALUDE_right_triangle_existence_l2710_271097

noncomputable def f (x : ℝ) : ℝ :=
  if x < Real.exp 1 then -x^3 + x^2 else Real.log x

theorem right_triangle_existence (a : ℝ) :
  (∃ t : ℝ, t ≥ Real.exp 1 ∧
    ((-t^2 + f t * (-t^3 + t^2) = 0) ∧
     (∃ P Q : ℝ × ℝ, P = (t, f t) ∧ Q = (-t, f (-t)) ∧
       (P.1 * Q.1 + P.2 * Q.2 = 0) ∧
       ((P.1 + Q.1) / 2 = 0))))
  ↔ (0 < a ∧ a ≤ 1 / (Real.exp 1 * Real.log (Real.exp 1) + 1)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l2710_271097


namespace NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l2710_271065

theorem min_sum_given_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 4/b = 2) : a + b ≥ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l2710_271065


namespace NUMINAMATH_CALUDE_bike_shop_profit_l2710_271013

/-- Jim's bike shop problem -/
theorem bike_shop_profit (tire_repair_price : ℕ) (tire_repair_cost : ℕ) (tire_repairs : ℕ)
  (complex_repair_price : ℕ) (complex_repair_cost : ℕ) (complex_repairs : ℕ)
  (fixed_expenses : ℕ) (total_profit : ℕ) :
  tire_repair_price = 20 →
  tire_repair_cost = 5 →
  tire_repairs = 300 →
  complex_repair_price = 300 →
  complex_repair_cost = 50 →
  complex_repairs = 2 →
  fixed_expenses = 4000 →
  total_profit = 3000 →
  (tire_repairs * (tire_repair_price - tire_repair_cost) +
   complex_repairs * (complex_repair_price - complex_repair_cost) -
   fixed_expenses + 2000) = total_profit :=
by sorry

end NUMINAMATH_CALUDE_bike_shop_profit_l2710_271013


namespace NUMINAMATH_CALUDE_janabel_widget_sales_l2710_271014

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem janabel_widget_sales : arithmetic_sequence_sum 2 3 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_janabel_widget_sales_l2710_271014


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2710_271051

theorem hemisphere_surface_area (base_area : ℝ) (h : base_area = 225 * Real.pi) :
  let radius := Real.sqrt (base_area / Real.pi)
  let curved_area := 2 * Real.pi * radius^2
  let total_area := curved_area + base_area
  total_area = 675 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2710_271051


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_18_27_45_l2710_271071

theorem arithmetic_mean_of_18_27_45 (S : Finset ℕ) :
  S = {18, 27, 45} →
  (S.sum id) / S.card = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_18_27_45_l2710_271071


namespace NUMINAMATH_CALUDE_reflection_properties_l2710_271047

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a figure as a set of points
def Figure := Set Point2D

-- Define the reflection of a point about a line
def reflect (p : Point2D) (l : Line2D) : Point2D :=
  sorry

-- Define the reflection of a figure about a line
def reflectFigure (f : Figure) (l : Line2D) : Figure :=
  sorry

-- Define a predicate to check if a point is on a specific side of a line
def onSide (p : Point2D) (l : Line2D) (side : Bool) : Prop :=
  sorry

-- Define a predicate to check if a figure is on a specific side of a line
def figureOnSide (f : Figure) (l : Line2D) (side : Bool) : Prop :=
  sorry

-- Define a predicate to check if two figures have the same shape
def sameShape (f1 f2 : Figure) : Prop :=
  sorry

-- Define a predicate to check if a figure touches a line at given points
def touchesAt (f : Figure) (l : Line2D) (p q : Point2D) : Prop :=
  sorry

theorem reflection_properties 
  (f : Figure) (l : Line2D) (p q : Point2D) (side : Bool) :
  figureOnSide f l side →
  touchesAt f l p q →
  let f' := reflectFigure f l
  figureOnSide f' l (!side) ∧
  sameShape f f' ∧
  touchesAt f' l p q :=
by
  sorry

end NUMINAMATH_CALUDE_reflection_properties_l2710_271047


namespace NUMINAMATH_CALUDE_solution_value_l2710_271042

theorem solution_value (x a : ℝ) : 
  2 * (x - 6) = -16 →
  a * (x + 3) = (1/2) * a + x →
  a^2 - (a/2) + 1 = 19 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l2710_271042


namespace NUMINAMATH_CALUDE_sandy_earnings_l2710_271007

/-- Calculates the total earnings for a given hourly rate and hours worked over three days -/
def total_earnings (hourly_rate : ℝ) (hours_day1 hours_day2 hours_day3 : ℝ) : ℝ :=
  hourly_rate * (hours_day1 + hours_day2 + hours_day3)

/-- Sandy's earnings problem -/
theorem sandy_earnings : 
  let hourly_rate : ℝ := 15
  let hours_friday : ℝ := 10
  let hours_saturday : ℝ := 6
  let hours_sunday : ℝ := 14
  total_earnings hourly_rate hours_friday hours_saturday hours_sunday = 450 := by
  sorry

end NUMINAMATH_CALUDE_sandy_earnings_l2710_271007


namespace NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l2710_271020

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 + 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l2710_271020


namespace NUMINAMATH_CALUDE_largest_n_for_unique_k_l2710_271057

theorem largest_n_for_unique_k : ∃ (n : ℕ), n > 0 ∧ 
  (∃! (k : ℤ), (9 : ℚ)/17 < n/(n + k) ∧ n/(n + k) < 8/15) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (9 : ℚ)/17 < m/(m + k) ∧ m/(m + k) < 8/15)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_unique_k_l2710_271057


namespace NUMINAMATH_CALUDE_max_candies_eaten_l2710_271025

theorem max_candies_eaten (n : ℕ) (h : n = 32) : 
  (n * (n - 1)) / 2 = 496 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_eaten_l2710_271025


namespace NUMINAMATH_CALUDE_steven_shirt_count_l2710_271023

def brian_shirts : ℕ := 3

def andrew_shirts : ℕ := 6 * brian_shirts

def steven_shirts : ℕ := 4 * andrew_shirts

theorem steven_shirt_count : steven_shirts = 72 := by
  sorry

end NUMINAMATH_CALUDE_steven_shirt_count_l2710_271023


namespace NUMINAMATH_CALUDE_largest_divisor_when_square_divisible_by_50_l2710_271018

theorem largest_divisor_when_square_divisible_by_50 (n : ℕ) (h1 : n > 0) (h2 : 50 ∣ n^2) :
  ∃ (d : ℕ), d ∣ n ∧ d = 10 ∧ ∀ (k : ℕ), k ∣ n → k ≤ d :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_when_square_divisible_by_50_l2710_271018


namespace NUMINAMATH_CALUDE_x_equals_four_l2710_271095

/-- Custom operation € -/
def euro (x y : ℝ) : ℝ := 2 * x * y

/-- Theorem stating that x = 4 given the conditions -/
theorem x_equals_four :
  ∃ x : ℝ, euro 9 (euro x 5) = 720 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_x_equals_four_l2710_271095


namespace NUMINAMATH_CALUDE_half_plus_five_equals_thirteen_l2710_271039

theorem half_plus_five_equals_thirteen (n : ℕ) (h : n = 16) : n / 2 + 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_thirteen_l2710_271039


namespace NUMINAMATH_CALUDE_ferry_speed_proof_l2710_271002

/-- The speed of ferry P in km/h -/
def speed_P : ℝ := 8

/-- The speed of ferry Q in km/h -/
def speed_Q : ℝ := speed_P + 1

/-- The travel time of ferry P in hours -/
def time_P : ℝ := 3

/-- The travel time of ferry Q in hours -/
def time_Q : ℝ := time_P + 5

/-- The distance traveled by ferry P in km -/
def distance_P : ℝ := speed_P * time_P

/-- The distance traveled by ferry Q in km -/
def distance_Q : ℝ := speed_Q * time_Q

theorem ferry_speed_proof :
  speed_P = 8 ∧
  speed_Q = speed_P + 1 ∧
  time_P = 3 ∧
  time_Q = time_P + 5 ∧
  distance_Q = 3 * distance_P :=
by sorry

end NUMINAMATH_CALUDE_ferry_speed_proof_l2710_271002


namespace NUMINAMATH_CALUDE_only_setC_in_proportion_l2710_271059

-- Define a structure for a set of four line segments
structure FourSegments where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the property of being in proportion
def isInProportion (segments : FourSegments) : Prop :=
  segments.a * segments.d = segments.b * segments.c

-- Define the four sets of line segments
def setA : FourSegments := ⟨3, 5, 6, 9⟩
def setB : FourSegments := ⟨3, 5, 8, 9⟩
def setC : FourSegments := ⟨3, 9, 10, 30⟩
def setD : FourSegments := ⟨3, 6, 7, 9⟩

-- State the theorem
theorem only_setC_in_proportion :
  isInProportion setC ∧
  ¬isInProportion setA ∧
  ¬isInProportion setB ∧
  ¬isInProportion setD :=
sorry

end NUMINAMATH_CALUDE_only_setC_in_proportion_l2710_271059


namespace NUMINAMATH_CALUDE_positive_number_equality_l2710_271083

theorem positive_number_equality (x : ℝ) (h1 : x > 0) 
  (h2 : (2/3) * x = (64/216) * (1/x)) : x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equality_l2710_271083


namespace NUMINAMATH_CALUDE_line_inclination_angle_l2710_271056

-- Define the parametric equations
def x (t : ℝ) : ℝ := 1 + t
def y (t : ℝ) : ℝ := 1 - t

-- Define the line using the parametric equations
def line : Set (ℝ × ℝ) := {(x t, y t) | t : ℝ}

-- State the theorem
theorem line_inclination_angle :
  let slope := (y 1 - y 0) / (x 1 - x 0)
  let inclination_angle := Real.arctan slope
  inclination_angle = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l2710_271056


namespace NUMINAMATH_CALUDE_marvelous_divisible_by_five_infinitely_many_marvelous_numbers_l2710_271075

def is_marvelous (n : ℕ+) : Prop :=
  ∃ (a b c d e : ℕ+),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    (n : ℕ) % a = 0 ∧ (n : ℕ) % b = 0 ∧ (n : ℕ) % c = 0 ∧ (n : ℕ) % d = 0 ∧ (n : ℕ) % e = 0 ∧
    n = a^4 + b^4 + c^4 + d^4 + e^4

theorem marvelous_divisible_by_five (n : ℕ+) (h : is_marvelous n) :
  (n : ℕ) % 5 = 0 :=
sorry

theorem infinitely_many_marvelous_numbers :
  ∀ k : ℕ, ∃ n : ℕ+, n > k ∧ is_marvelous n :=
sorry

end NUMINAMATH_CALUDE_marvelous_divisible_by_five_infinitely_many_marvelous_numbers_l2710_271075


namespace NUMINAMATH_CALUDE_system_solution_set_system_solutions_l2710_271061

def system_has_solution (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x - 4 = a * (y^3 - 2)) ∧ (2 * x / (|y^3| + y^3) = Real.sqrt x)

theorem system_solution_set :
  {a : ℝ | system_has_solution a} = Set.Ioi 2 ∪ Set.Iic 0 :=
sorry

theorem system_solutions (a : ℝ) (h : system_has_solution a) :
  (∃ x y : ℝ, x = 4 ∧ y^3 = 2) ∨
  (∃ x y : ℝ, x = 0 ∧ y^3 = 2*a - 4) :=
sorry

end NUMINAMATH_CALUDE_system_solution_set_system_solutions_l2710_271061


namespace NUMINAMATH_CALUDE_root_magnitude_theorem_l2710_271032

theorem root_magnitude_theorem (p : ℝ) (r₁ r₂ : ℝ) :
  (r₁ ≠ r₂) →
  (r₁^2 + p*r₁ + 12 = 0) →
  (r₂^2 + p*r₂ + 12 = 0) →
  (abs r₁ > 4 ∨ abs r₂ > 4) :=
by sorry

end NUMINAMATH_CALUDE_root_magnitude_theorem_l2710_271032


namespace NUMINAMATH_CALUDE_coat_original_price_l2710_271016

/-- Proves that if a coat is sold for 135 yuan after a 25% discount, its original price was 180 yuan -/
theorem coat_original_price (discounted_price : ℝ) (discount_percent : ℝ) 
  (h1 : discounted_price = 135)
  (h2 : discount_percent = 25) : 
  discounted_price / (1 - discount_percent / 100) = 180 := by
sorry

end NUMINAMATH_CALUDE_coat_original_price_l2710_271016


namespace NUMINAMATH_CALUDE_toucan_problem_l2710_271019

theorem toucan_problem (initial_toucans : ℕ) : 
  (initial_toucans + 1 = 3) → initial_toucans = 2 := by
sorry

end NUMINAMATH_CALUDE_toucan_problem_l2710_271019


namespace NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l2710_271009

theorem least_positive_integer_for_multiple_of_five : 
  ∀ n : ℕ, n > 0 → (725 + n) % 5 = 0 → n ≥ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_five_l2710_271009


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_line_l2710_271081

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition that a circle's center is on the parabola
def circle_center_on_parabola (c : Circle) : Prop :=
  parabola c.center.1 c.center.2

-- Define the condition that a circle passes through a point
def circle_passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

-- Define a line by its equation y = mx + b
structure Line where
  m : ℝ
  b : ℝ

-- Define the condition that a circle is tangent to a line
def circle_tangent_to_line (c : Circle) (l : Line) : Prop :=
  ∃ (x y : ℝ), y = l.m * x + l.b ∧
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ∧
  (c.center.1 - x)^2 + (c.center.2 - y)^2 = c.radius^2

theorem parabola_circle_tangent_line :
  ∀ (c : Circle) (l : Line),
  circle_center_on_parabola c →
  circle_passes_through c (0, 1) →
  circle_tangent_to_line c l →
  l.m = 0 ∧ l.b = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_line_l2710_271081


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_17_l2710_271024

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (n : ℕ), (n ≥ 10000 ∧ n < 100000) ∧ 
             (n % 17 = 6) ∧ 
             (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m) ∧
             n = 10002 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_17_l2710_271024


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2710_271076

/-- The sum of the infinite series ∑_{k=1}^∞ (k^3 / 3^k) is equal to 6 -/
theorem infinite_series_sum : 
  (∑' k : ℕ+, (k : ℝ)^3 / 3^(k : ℝ)) = 6 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2710_271076


namespace NUMINAMATH_CALUDE_unique_function_solution_l2710_271082

theorem unique_function_solution (f : ℕ → ℕ) :
  (∀ a b : ℕ, f (f a + f b) = a + b) ↔ (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_solution_l2710_271082


namespace NUMINAMATH_CALUDE_prob_at_least_one_red_l2710_271033

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := red_balls + white_balls

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one red ball when drawing 2 balls from a bag containing 2 red and 2 white balls -/
theorem prob_at_least_one_red : 
  (Nat.choose total_balls drawn_balls - Nat.choose white_balls drawn_balls) / Nat.choose total_balls drawn_balls = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_red_l2710_271033


namespace NUMINAMATH_CALUDE_no_solution_functional_equation_l2710_271093

theorem no_solution_functional_equation :
  ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x * y) = f x * f y + 2 * x * y :=
by sorry

end NUMINAMATH_CALUDE_no_solution_functional_equation_l2710_271093


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l2710_271003

/-- Given an ellipse and a hyperbola with specific foci, prove the product of their semi-axes lengths -/
theorem ellipse_hyperbola_product (a b : ℝ) : 
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ (y = 5 ∨ y = -5))) → 
  (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (y = 0 ∧ (x = 7 ∨ x = -7))) → 
  |a * b| = 2 * Real.sqrt 111 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l2710_271003


namespace NUMINAMATH_CALUDE_bookstore_change_percentage_l2710_271085

def book_prices : List ℝ := [10, 8, 6, 4, 3, 5]
def discount_rate : ℝ := 0.1
def payment_amount : ℝ := 50

theorem bookstore_change_percentage :
  let total_price := book_prices.sum
  let discounted_price := total_price * (1 - discount_rate)
  let change := payment_amount - discounted_price
  let change_percentage := (change / payment_amount) * 100
  change_percentage = 35.2 := by sorry

end NUMINAMATH_CALUDE_bookstore_change_percentage_l2710_271085


namespace NUMINAMATH_CALUDE_petes_flag_problem_l2710_271091

/-- The number of stars on the US flag -/
def us_stars : ℕ := 50

/-- The number of stripes on the US flag -/
def us_stripes : ℕ := 13

/-- The number of circles on Pete's flag -/
def petes_circles : ℕ := us_stars / 2 - 3

/-- The number of squares on Pete's flag -/
def petes_squares (x : ℕ) : ℕ := 2 * us_stripes + x

/-- The total number of shapes on Pete's flag -/
def total_shapes : ℕ := 54

theorem petes_flag_problem :
  ∃ x : ℕ, petes_circles + petes_squares x = total_shapes ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_petes_flag_problem_l2710_271091


namespace NUMINAMATH_CALUDE_min_pizzas_cover_scooter_cost_l2710_271035

def scooter_cost : ℕ := 8000
def earning_per_pizza : ℕ := 12
def cost_per_delivery : ℕ := 4

def min_pizzas : ℕ := 1000

theorem min_pizzas_cover_scooter_cost :
  ∀ p : ℕ, p ≥ min_pizzas →
  p * (earning_per_pizza - cost_per_delivery) ≥ scooter_cost :=
by sorry

end NUMINAMATH_CALUDE_min_pizzas_cover_scooter_cost_l2710_271035


namespace NUMINAMATH_CALUDE_value_of_a_l2710_271011

theorem value_of_a (x y z a : ℝ) 
  (h1 : 2 * x^2 + 3 * y^2 + 6 * z^2 = a) 
  (h2 : a > 0)
  (h3 : ∀ (x' y' z' : ℝ), 2 * x'^2 + 3 * y'^2 + 6 * z'^2 = a → x' + y' + z' ≤ 1) 
  (h4 : ∃ (x' y' z' : ℝ), 2 * x'^2 + 3 * y'^2 + 6 * z'^2 = a ∧ x' + y' + z' = 1) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l2710_271011


namespace NUMINAMATH_CALUDE_track_width_l2710_271034

theorem track_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 10 * Real.pi) : 
  r₁ - r₂ = 5 := by
  sorry

end NUMINAMATH_CALUDE_track_width_l2710_271034


namespace NUMINAMATH_CALUDE_certain_number_exists_l2710_271037

theorem certain_number_exists : ∃ N : ℝ, (7/13) * N = (5/16) * N + 500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l2710_271037


namespace NUMINAMATH_CALUDE_cube_surface_area_l2710_271090

theorem cube_surface_area (a : ℝ) (h : a > 0) : 
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 3 = a ∧ 6 * s^2 = 2 * a^2 :=
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2710_271090


namespace NUMINAMATH_CALUDE_complex_sum_equals_negative_two_l2710_271052

theorem complex_sum_equals_negative_two (z : ℂ) 
  (h1 : z = Complex.exp (6 * Real.pi * Complex.I / 11))
  (h2 : z^11 = 1) : 
  (z / (1 + z^3)) + (z^2 / (1 + z^6)) + (z^4 / (1 + z^9)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_negative_two_l2710_271052


namespace NUMINAMATH_CALUDE_expression_evaluation_l2710_271087

theorem expression_evaluation : (2^3 - 2^2) - (3^3 - 3^2) + (4^3 - 4^2) - (5^3 - 5^2) = -66 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2710_271087


namespace NUMINAMATH_CALUDE_early_arrival_time_l2710_271048

/-- Proves that a boy walking at 5/4 of his usual rate arrives 4 minutes early when his usual time is 20 minutes. -/
theorem early_arrival_time (usual_time : ℝ) (usual_rate : ℝ) (faster_rate : ℝ) :
  usual_time = 20 →
  faster_rate = (5 / 4) * usual_rate →
  usual_time - (usual_time * usual_rate / faster_rate) = 4 := by
sorry

end NUMINAMATH_CALUDE_early_arrival_time_l2710_271048


namespace NUMINAMATH_CALUDE_union_complement_problem_l2710_271053

def U : Set ℤ := {x : ℤ | |x| < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_complement_problem :
  A ∪ (U \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_problem_l2710_271053


namespace NUMINAMATH_CALUDE_polygon_side_length_l2710_271069

theorem polygon_side_length (n : ℕ) (h : n > 0) :
  ∃ (side_length : ℝ),
    side_length ≥ Real.sqrt (1/2 * (1 - Real.cos (π / n))) ∧
    (∃ (vertices : Fin (2*n) → ℝ × ℝ),
      (∀ i : Fin (2*n), ∃ j : Fin (2*n), i ≠ j ∧
        ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 1) ∧
      (∃ i j : Fin (2*n), i ≠ j ∧
        ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 ≥ side_length^2)) :=
by sorry

end NUMINAMATH_CALUDE_polygon_side_length_l2710_271069


namespace NUMINAMATH_CALUDE_basic_computer_price_is_correct_l2710_271058

/-- The price of the basic computer -/
def basic_computer_price : ℝ := 1040

/-- The price of the printer -/
def printer_price : ℝ := 2500 - basic_computer_price

/-- The total price of the basic computer and printer -/
def total_price : ℝ := 2500

/-- The price of the first enhanced computer -/
def enhanced_computer1_price : ℝ := basic_computer_price + 800

/-- The price of the second enhanced computer -/
def enhanced_computer2_price : ℝ := basic_computer_price + 1100

/-- The price of the third enhanced computer -/
def enhanced_computer3_price : ℝ := basic_computer_price + 1500

theorem basic_computer_price_is_correct :
  basic_computer_price + printer_price = total_price ∧
  enhanced_computer1_price + (1/5) * (enhanced_computer1_price + printer_price) = total_price ∧
  enhanced_computer2_price + (1/8) * (enhanced_computer2_price + printer_price) = total_price ∧
  enhanced_computer3_price + (1/10) * (enhanced_computer3_price + printer_price) = total_price :=
by sorry

end NUMINAMATH_CALUDE_basic_computer_price_is_correct_l2710_271058


namespace NUMINAMATH_CALUDE_cube_side_ratio_l2710_271045

/-- Given two cubes of the same material, if one cube weighs 5 pounds and the other weighs 40 pounds,
    then the ratio of the side length of the heavier cube to the side length of the lighter cube is 2:1. -/
theorem cube_side_ratio (s S : ℝ) (h1 : s > 0) (h2 : S > 0) : 
  (5 : ℝ) / s^3 = (40 : ℝ) / S^3 → S / s = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l2710_271045


namespace NUMINAMATH_CALUDE_expression_simplification_l2710_271092

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 / (x - 2) + 1) / ((x^2 - 2*x + 1) / (x - 2)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2710_271092


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l2710_271096

/-- The distance between Maxwell's and Brad's homes in kilometers -/
def total_distance : ℝ := 40

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 3

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 5

/-- The distance traveled by Maxwell when they meet -/
def maxwell_distance : ℝ := 15

theorem meeting_point_theorem :
  maxwell_distance = total_distance * maxwell_speed / (maxwell_speed + brad_speed) :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l2710_271096


namespace NUMINAMATH_CALUDE_larger_integer_is_48_l2710_271028

theorem larger_integer_is_48 (x y : ℤ) : 
  y = 4 * x →                           -- Two integers are in the ratio of 1 to 4
  (x + 12) * 2 = y →                    -- If 12 is added to the smaller number, the ratio becomes 1 to 2
  y = 48 :=                             -- The larger integer is 48
by
  sorry


end NUMINAMATH_CALUDE_larger_integer_is_48_l2710_271028


namespace NUMINAMATH_CALUDE_negation_equivalence_l2710_271010

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2710_271010


namespace NUMINAMATH_CALUDE_jaco_gift_budget_l2710_271030

/-- Given a total budget, number of friends, and cost of parent gifts, 
    calculate the budget for each friend's gift -/
def friend_gift_budget (total_budget : ℕ) (num_friends : ℕ) (parent_gift_cost : ℕ) : ℕ :=
  (total_budget - 2 * parent_gift_cost) / num_friends

/-- Proof that Jaco's budget for each friend's gift is $9 -/
theorem jaco_gift_budget :
  friend_gift_budget 100 8 14 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jaco_gift_budget_l2710_271030


namespace NUMINAMATH_CALUDE_building_height_l2710_271068

/-- Given a flagpole and a building casting shadows under similar conditions,
    this theorem proves that the height of the building can be determined
    using the principle of similar triangles. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_shadow_pos : 0 < building_shadow)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 70) :
  (flagpole_height / flagpole_shadow) * building_shadow = 28 :=
by sorry

end NUMINAMATH_CALUDE_building_height_l2710_271068


namespace NUMINAMATH_CALUDE_multiply_divide_sqrt_l2710_271022

theorem multiply_divide_sqrt (x : ℝ) (y : ℝ) (h1 : x = 0.42857142857142855) (h2 : x ≠ 0) :
  Real.sqrt ((x * y) / 7) = x → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_sqrt_l2710_271022


namespace NUMINAMATH_CALUDE_at_least_one_black_certain_l2710_271073

/-- Represents the color of a ball -/
inductive BallColor
  | Black
  | White

/-- Represents the composition of balls in the bag -/
structure BagComposition where
  blackBalls : Nat
  whiteBalls : Nat

/-- Represents the result of drawing two balls -/
structure DrawResult where
  firstBall : BallColor
  secondBall : BallColor

/-- Defines the event of drawing at least one black ball -/
def AtLeastOneBlack (result : DrawResult) : Prop :=
  result.firstBall = BallColor.Black ∨ result.secondBall = BallColor.Black

/-- The theorem to be proved -/
theorem at_least_one_black_certain (bag : BagComposition) 
    (h1 : bag.blackBalls = 2) 
    (h2 : bag.whiteBalls = 1) : 
    ∀ (result : DrawResult), AtLeastOneBlack result :=
  sorry

end NUMINAMATH_CALUDE_at_least_one_black_certain_l2710_271073


namespace NUMINAMATH_CALUDE_pen_price_calculation_l2710_271046

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 630 ∧ num_pens = 30 ∧ num_pencils = 75 ∧ pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l2710_271046


namespace NUMINAMATH_CALUDE_mary_number_is_14_l2710_271062

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem mary_number_is_14 :
  ∃! x : ℕ, is_two_digit x ∧
    91 ≤ switch_digits (4 * x - 7) ∧
    switch_digits (4 * x - 7) ≤ 95 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_number_is_14_l2710_271062


namespace NUMINAMATH_CALUDE_money_division_l2710_271074

theorem money_division (p q r : ℕ) (total : ℕ) :
  p + q + r = total →
  p = 3 * (total / 22) →
  q = 7 * (total / 22) →
  r = 12 * (total / 22) →
  r - q = 3000 →
  q - p = 2400 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l2710_271074


namespace NUMINAMATH_CALUDE_ethanol_mixture_optimization_l2710_271021

theorem ethanol_mixture_optimization (initial_volume : ℝ) (initial_ethanol_percentage : ℝ)
  (added_ethanol : ℝ) (final_ethanol_percentage : ℝ) :
  initial_volume = 45 →
  initial_ethanol_percentage = 0.05 →
  added_ethanol = 2.5 →
  final_ethanol_percentage = 0.1 →
  (initial_volume * initial_ethanol_percentage + added_ethanol) /
    (initial_volume + added_ethanol) = final_ethanol_percentage :=
by sorry

end NUMINAMATH_CALUDE_ethanol_mixture_optimization_l2710_271021


namespace NUMINAMATH_CALUDE_faster_train_speed_l2710_271055

theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 100)  -- Length of each train in meters
  (h2 : slower_speed = 36)   -- Speed of slower train in km/hr
  (h3 : passing_time = 72)   -- Time taken to pass in seconds
  : ∃ (faster_speed : ℝ), faster_speed = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l2710_271055


namespace NUMINAMATH_CALUDE_min_bound_sqrt_two_l2710_271000

theorem min_bound_sqrt_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  min x (min (y + 1/x) (1/y)) ≤ Real.sqrt 2 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ min x (min (y + 1/x) (1/y)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_bound_sqrt_two_l2710_271000


namespace NUMINAMATH_CALUDE_square_sum_quadruple_l2710_271043

theorem square_sum_quadruple (n : ℕ) (h : n ≥ 8) :
  ∃ (a b c d : ℕ),
    a = 3*n^2 - 18*n - 39 ∧
    b = 3*n^2 + 6 ∧
    c = 3*n^2 + 18*n + 33 ∧
    d = 3*n^2 + 36*n + 42 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (w x y z : ℕ),
      a + b + c = w^2 ∧
      a + b + d = x^2 ∧
      a + c + d = y^2 ∧
      b + c + d = z^2 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_quadruple_l2710_271043


namespace NUMINAMATH_CALUDE_area_between_specific_lines_l2710_271044

/-- Line passing through two points -/
structure Line where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Calculate the area between two lines from x = 0 to x = 5 -/
def areaBetweenLines (l1 l2 : Line) : ℝ :=
  sorry

/-- The main theorem -/
theorem area_between_specific_lines :
  let line1 : Line := ⟨0, 3, 6, 0⟩
  let line2 : Line := ⟨0, 5, 10, 2⟩
  areaBetweenLines line1 line2 = 10 := by sorry

end NUMINAMATH_CALUDE_area_between_specific_lines_l2710_271044


namespace NUMINAMATH_CALUDE_students_in_grade_l2710_271006

theorem students_in_grade (n : ℕ) (misha : ℕ) : 
  (misha = n - 59) ∧ (misha = 60) → n = 119 :=
by sorry

end NUMINAMATH_CALUDE_students_in_grade_l2710_271006


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_is_correct_l2710_271089

/-- The smallest positive integer b for which x^2 + bx + 1760 factors into (x + p)(x + q) with integer p and q -/
def smallest_factorizable_b : ℕ := 84

/-- A polynomial of the form x^2 + bx + 1760 -/
def polynomial (b : ℕ) (x : ℤ) : ℤ := x^2 + b * x + 1760

/-- Checks if a polynomial can be factored into (x + p)(x + q) with integer p and q -/
def is_factorizable (b : ℕ) : Prop :=
  ∃ (p q : ℤ), ∀ x, polynomial b x = (x + p) * (x + q)

theorem smallest_factorizable_b_is_correct :
  (is_factorizable smallest_factorizable_b) ∧
  (∀ b : ℕ, b < smallest_factorizable_b → ¬(is_factorizable b)) :=
sorry

end NUMINAMATH_CALUDE_smallest_factorizable_b_is_correct_l2710_271089


namespace NUMINAMATH_CALUDE_parallel_line_equation_line_K_equation_l2710_271040

/-- Given a line with equation y = mx + b, this function returns the y-intercept of a parallel line
    that is d units away from the original line. -/
def parallelLineYIntercept (m : ℝ) (b : ℝ) (d : ℝ) : Set ℝ :=
  {y | ∃ (sign : ℝ), sign = 1 ∨ sign = -1 ∧ y = b + sign * d * Real.sqrt (m^2 + 1)}

theorem parallel_line_equation (m b d : ℝ) :
  parallelLineYIntercept m b d = {b + d * Real.sqrt (m^2 + 1), b - d * Real.sqrt (m^2 + 1)} := by
  sorry

/-- The equation of line K, which is parallel to y = 1/2x + 3 and 5 units away from it. -/
theorem line_K_equation :
  parallelLineYIntercept (1/2) 3 5 = {3 + 5 * Real.sqrt 5 / 2, 3 - 5 * Real.sqrt 5 / 2} := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_line_K_equation_l2710_271040


namespace NUMINAMATH_CALUDE_max_value_abc_l2710_271088

theorem max_value_abc (a b c : ℝ) (h : a + 3 * b + c = 6) :
  (∀ x y z : ℝ, x + 3 * y + z = 6 → a * b + a * c + b * c ≥ x * y + x * z + y * z) →
  a * b + a * c + b * c = 12 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l2710_271088


namespace NUMINAMATH_CALUDE_sally_nickels_l2710_271060

-- Define the initial state and gifts
def initial_nickels : ℕ := 7
def dad_gift : ℕ := 9
def mom_gift : ℕ := 2

-- Theorem to prove
theorem sally_nickels : initial_nickels + dad_gift + mom_gift = 18 := by
  sorry

end NUMINAMATH_CALUDE_sally_nickels_l2710_271060


namespace NUMINAMATH_CALUDE_calculate_expression_l2710_271038

theorem calculate_expression : 20.17 * 69 + 201.7 * 1.3 - 8.2 * 1.7 = 1640 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2710_271038


namespace NUMINAMATH_CALUDE_april_sales_calculation_l2710_271072

def january_sales : ℕ := 90
def february_sales : ℕ := 50
def march_sales : ℕ := 70
def average_sales : ℕ := 72
def total_months : ℕ := 5

theorem april_sales_calculation :
  ∃ (april_sales may_sales : ℕ),
    (january_sales + february_sales + march_sales + april_sales + may_sales) / total_months = average_sales ∧
    april_sales = 75 := by
  sorry

end NUMINAMATH_CALUDE_april_sales_calculation_l2710_271072


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2710_271004

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, 280 * r = a ∧ a * r = 35 / 8) : a = 35 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2710_271004


namespace NUMINAMATH_CALUDE_cake_recipe_flour_flour_in_recipe_l2710_271078

theorem cake_recipe_flour (salt_cups : ℕ) (flour_added : ℕ) (flour_salt_diff : ℕ) : ℕ :=
  let total_flour := salt_cups + flour_salt_diff
  total_flour

theorem flour_in_recipe :
  cake_recipe_flour 7 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cake_recipe_flour_flour_in_recipe_l2710_271078


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l2710_271041

theorem quadratic_equation_condition (m : ℝ) : 
  (m^2 - 2 = 2 ∧ m + 2 ≠ 0) ↔ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l2710_271041


namespace NUMINAMATH_CALUDE_movie_savings_theorem_l2710_271084

/-- Represents the savings calculation for a movie outing --/
def movie_savings (regular_price : ℚ) (student_discount : ℚ) (senior_discount : ℚ) 
  (early_discount : ℚ) (popcorn_price : ℚ) (popcorn_discount : ℚ) 
  (nachos_price : ℚ) (nachos_discount : ℚ) (hotdog_price : ℚ) 
  (hotdog_discount : ℚ) (combo_discount : ℚ) : ℚ :=
  let regular_tickets := 2 * regular_price
  let student_ticket := regular_price - student_discount
  let senior_ticket := regular_price - senior_discount
  let early_factor := 1 - early_discount
  let early_tickets := (regular_tickets + student_ticket + senior_ticket) * early_factor
  let ticket_savings := regular_tickets + student_ticket + senior_ticket - early_tickets
  let food_regular := popcorn_price + nachos_price + hotdog_price
  let food_discounted := popcorn_price * (1 - popcorn_discount) + 
                         nachos_price * (1 - nachos_discount) + 
                         hotdog_price * (1 - hotdog_discount)
  let food_combo := popcorn_price * (1 - popcorn_discount) + 
                    nachos_price * (1 - nachos_discount) + 
                    hotdog_price * (1 - hotdog_discount) * (1 - combo_discount)
  let food_savings := food_regular - food_combo
  ticket_savings + food_savings

/-- The total savings for the movie outing is $16.80 --/
theorem movie_savings_theorem : 
  movie_savings 10 2 3 (1/5) 10 (1/2) 8 (3/10) 6 (1/5) (1/4) = 84/5 := by
  sorry

end NUMINAMATH_CALUDE_movie_savings_theorem_l2710_271084


namespace NUMINAMATH_CALUDE_collinear_vectors_product_l2710_271015

/-- Given two non-collinear vectors i and j in a vector space V over ℝ,
    if AB = i + m*j, AD = n*i + j, m ≠ 1, and points A, B, and D are collinear,
    then mn = 1 -/
theorem collinear_vectors_product (V : Type*) [AddCommGroup V] [Module ℝ V]
  (i j : V) (m n : ℝ) (A B D : V) :
  LinearIndependent ℝ ![i, j] →
  B - A = i + m • j →
  D - A = n • i + j →
  m ≠ 1 →
  ∃ (k : ℝ), B - A = k • (D - A) →
  m * n = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_product_l2710_271015
