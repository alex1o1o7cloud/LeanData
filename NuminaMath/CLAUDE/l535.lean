import Mathlib

namespace NUMINAMATH_CALUDE_rhombus_area_l535_53526

/-- The area of a rhombus with side length 13 and diagonals differing by 10 units is 208 square units. -/
theorem rhombus_area (s d₁ d₂ : ℝ) (h₁ : s = 13) (h₂ : d₂ - d₁ = 10) 
    (h₃ : s^2 = (d₁/2)^2 + (d₂/2)^2) : d₁ * d₂ / 2 = 208 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l535_53526


namespace NUMINAMATH_CALUDE_amy_school_year_hours_l535_53592

/-- Amy's summer work and earnings information -/
structure SummerWork where
  hours_per_week : ℕ
  weeks : ℕ
  total_earnings : ℕ

/-- Amy's school year work plan -/
structure SchoolYearPlan where
  weeks : ℕ
  target_earnings : ℕ

/-- Calculate required weekly hours for school year -/
def required_weekly_hours (summer : SummerWork) (school : SchoolYearPlan) : ℕ :=
  15

/-- Theorem: Amy must work 15 hours per week during the school year -/
theorem amy_school_year_hours 
  (summer : SummerWork) 
  (school : SchoolYearPlan) 
  (h1 : summer.hours_per_week = 45)
  (h2 : summer.weeks = 8)
  (h3 : summer.total_earnings = 3600)
  (h4 : school.weeks = 24)
  (h5 : school.target_earnings = 3600) :
  required_weekly_hours summer school = 15 := by
  sorry

#check amy_school_year_hours

end NUMINAMATH_CALUDE_amy_school_year_hours_l535_53592


namespace NUMINAMATH_CALUDE_range_of_f_l535_53576

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l535_53576


namespace NUMINAMATH_CALUDE_closest_reps_20_eq_12_or_13_l535_53531

def weight_25 : ℕ := 25
def weight_20 : ℕ := 20
def reps_25 : ℕ := 10

def total_weight : ℕ := 2 * weight_25 * reps_25

def closest_reps (w : ℕ) : Set ℕ :=
  {n : ℕ | n * 2 * w ≥ total_weight ∧ 
    ∀ m : ℕ, m * 2 * w ≥ total_weight → n ≤ m}

theorem closest_reps_20_eq_12_or_13 : 
  closest_reps weight_20 = {12, 13} :=
sorry

end NUMINAMATH_CALUDE_closest_reps_20_eq_12_or_13_l535_53531


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l535_53509

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ i : ℕ, i > 0 ∧ i ≤ 10 → i ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ i : ℕ, i > 0 ∧ i ≤ 10 → i ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l535_53509


namespace NUMINAMATH_CALUDE_sum_of_angles_equals_360_l535_53516

-- Define the angles as real numbers
variable (A B C D F G : ℝ)

-- Define the property of being a quadrilateral
def is_quadrilateral (A B C D : ℝ) : Prop :=
  A + B + C + D = 360

-- State the theorem
theorem sum_of_angles_equals_360 
  (h : is_quadrilateral A B C D) : A + B + C + D + F + G = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_equals_360_l535_53516


namespace NUMINAMATH_CALUDE_target_line_is_correct_l535_53519

-- Define the line we're looking for
def target_line (x y : ℝ) : Prop := y = x + 1

-- Define the given line x + y = 0
def given_line (x y : ℝ) : Prop := x + y = 0

-- Define perpendicularity of two lines
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄, 
    f x₁ y₁ ∧ f x₂ y₂ ∧ g x₃ y₃ ∧ g x₄ y₄ ∧ 
    x₁ ≠ x₂ ∧ x₃ ≠ x₄ → 
    (y₂ - y₁) / (x₂ - x₁) * (y₄ - y₃) / (x₄ - x₃) = -1

-- Theorem statement
theorem target_line_is_correct : 
  target_line (-1) 0 ∧ 
  perpendicular target_line given_line :=
sorry

end NUMINAMATH_CALUDE_target_line_is_correct_l535_53519


namespace NUMINAMATH_CALUDE_sin_cos_identity_l535_53514

theorem sin_cos_identity (x : ℝ) : 
  (Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x))^2 = 2 - 2 * Real.cos ((2 / 3) * Real.pi - x) ↔ 
  (∃ n : ℤ, x = (2 * Real.pi / 5) * ↑n) ∨ 
  (∃ k : ℤ, x = (2 * Real.pi / 9) * (3 * ↑k + 1)) :=
sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l535_53514


namespace NUMINAMATH_CALUDE_race_finish_times_l535_53553

/-- Race parameters and results -/
structure RaceData where
  malcolm_speed : ℝ  -- Malcolm's speed in minutes per mile
  joshua_speed : ℝ   -- Joshua's speed in minutes per mile
  ellie_speed : ℝ    -- Ellie's speed in minutes per mile
  race_distance : ℝ  -- Race distance in miles

def finish_time (speed : ℝ) (distance : ℝ) : ℝ := speed * distance

/-- Theorem stating the time differences for Joshua and Ellie compared to Malcolm -/
theorem race_finish_times (data : RaceData) 
  (h_malcolm : data.malcolm_speed = 5)
  (h_joshua : data.joshua_speed = 7)
  (h_ellie : data.ellie_speed = 6)
  (h_distance : data.race_distance = 15) :
  let malcolm_time := finish_time data.malcolm_speed data.race_distance
  let joshua_time := finish_time data.joshua_speed data.race_distance
  let ellie_time := finish_time data.ellie_speed data.race_distance
  (joshua_time - malcolm_time = 30 ∧ ellie_time - malcolm_time = 15) := by
  sorry


end NUMINAMATH_CALUDE_race_finish_times_l535_53553


namespace NUMINAMATH_CALUDE_lily_bouquet_cost_l535_53502

/-- The cost of a bouquet of lilies given the number of lilies -/
def bouquet_cost (n : ℕ) : ℚ :=
  sorry

/-- The property that the price is directly proportional to the number of lilies -/
axiom price_proportional (n m : ℕ) :
  n ≠ 0 → m ≠ 0 → bouquet_cost n / n = bouquet_cost m / m

theorem lily_bouquet_cost :
  bouquet_cost 18 = 30 →
  bouquet_cost 45 = 75 :=
by sorry

end NUMINAMATH_CALUDE_lily_bouquet_cost_l535_53502


namespace NUMINAMATH_CALUDE_integral_sqrt_plus_xcosx_equals_pi_half_l535_53559

open Real MeasureTheory Interval

theorem integral_sqrt_plus_xcosx_equals_pi_half :
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x * Real.cos x) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_plus_xcosx_equals_pi_half_l535_53559


namespace NUMINAMATH_CALUDE_vacation_cost_l535_53545

theorem vacation_cost (C : ℝ) : 
  (C / 3 - C / 5 = 50) → C = 375 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l535_53545


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l535_53566

theorem common_root_quadratic_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + x + a = 0 ∧ x^2 + a*x + 1 = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l535_53566


namespace NUMINAMATH_CALUDE_peters_horse_food_l535_53510

/-- Calculates the total food required for horses over a given number of days -/
def total_food_required (num_horses : ℕ) (oats_per_meal : ℕ) (oats_meals_per_day : ℕ) 
                        (grain_per_day : ℕ) (num_days : ℕ) : ℕ :=
  let total_oats := num_horses * oats_per_meal * oats_meals_per_day * num_days
  let total_grain := num_horses * grain_per_day * num_days
  total_oats + total_grain

/-- Theorem: Peter needs 132 pounds of food to feed his horses for 3 days -/
theorem peters_horse_food : total_food_required 4 4 2 3 3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_peters_horse_food_l535_53510


namespace NUMINAMATH_CALUDE_books_read_difference_result_l535_53500

/-- The number of books Peter has read more than his brother and Sarah combined -/
def books_read_difference (total_books : ℕ) (peter_percent : ℚ) (brother_percent : ℚ) (sarah_percent : ℚ) : ℚ :=
  (peter_percent * total_books) - ((brother_percent + sarah_percent) * total_books)

/-- Theorem stating the difference in books read -/
theorem books_read_difference_result :
  books_read_difference 50 (60 / 100) (25 / 100) (15 / 100) = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_read_difference_result_l535_53500


namespace NUMINAMATH_CALUDE_point_on_segment_vector_relation_l535_53506

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points
variable (A B M O C D : V)

-- Define conditions
variable (h1 : M ∈ closedSegment A B)
variable (h2 : O ∉ line_through A B)
variable (h3 : C = 2 • O - A)  -- C is symmetric to A with respect to O
variable (h4 : D = 2 • C - B)  -- D is symmetric to B with respect to C
variable (x y : ℝ)
variable (h5 : O - M = x • (O - C) + y • (O - D))

-- Theorem statement
theorem point_on_segment_vector_relation :
  x + 3 * y = -1 :=
sorry

end NUMINAMATH_CALUDE_point_on_segment_vector_relation_l535_53506


namespace NUMINAMATH_CALUDE_quadratic_equation_prime_roots_l535_53532

theorem quadratic_equation_prime_roots (p q : ℕ) 
  (h1 : ∃ (x y : ℕ), Prime x ∧ Prime y ∧ p * x^2 - q * x + 1985 = 0 ∧ p * y^2 - q * y + 1985 = 0) :
  12 * p^2 + q = 414 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_prime_roots_l535_53532


namespace NUMINAMATH_CALUDE_half_three_abs_diff_squares_l535_53572

theorem half_three_abs_diff_squares : (1/2 : ℝ) * 3 * |20^2 - 15^2| = 262.5 := by
  sorry

end NUMINAMATH_CALUDE_half_three_abs_diff_squares_l535_53572


namespace NUMINAMATH_CALUDE_line_equation_proof_l535_53598

/-- Given a point (2, 1) and a slope of -2, prove that the equation 2x + y - 5 = 0 represents the line passing through this point with the given slope. -/
theorem line_equation_proof (x y : ℝ) :
  let point : ℝ × ℝ := (2, 1)
  let slope : ℝ := -2
  (2 * x + y - 5 = 0) ↔ (y - point.2 = slope * (x - point.1)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l535_53598


namespace NUMINAMATH_CALUDE_correct_ways_to_spend_l535_53565

/-- Represents the number of magazines costing 2 yuan -/
def magazines_2yuan : ℕ := 8

/-- Represents the number of magazines costing 1 yuan -/
def magazines_1yuan : ℕ := 3

/-- Represents the total budget in yuan -/
def budget : ℕ := 10

/-- Calculates the number of ways to select magazines to spend exactly the budget -/
def ways_to_spend_budget : ℕ := sorry

theorem correct_ways_to_spend : ways_to_spend_budget = 266 := by sorry

end NUMINAMATH_CALUDE_correct_ways_to_spend_l535_53565


namespace NUMINAMATH_CALUDE_probability_of_double_l535_53523

/-- The number of integers on the dominoes (from 0 to 12, inclusive) -/
def n : ℕ := 12

/-- The total number of dominoes in the set -/
def total_dominoes : ℕ := (n + 1) * (n + 2) / 2

/-- The number of doubles in the set -/
def num_doubles : ℕ := n + 1

/-- The probability of selecting a double -/
def prob_double : ℚ := num_doubles / total_dominoes

/-- Theorem stating the probability of selecting a double -/
theorem probability_of_double : prob_double = 13 / 91 := by sorry

end NUMINAMATH_CALUDE_probability_of_double_l535_53523


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l535_53591

-- Define the rectangle
def rectangle_width : ℝ := 11
def rectangle_height : ℝ := 7

-- Define the circles
def circle_diameter : ℝ := rectangle_height

-- Theorem statement
theorem distance_between_circle_centers : 
  let circle_radius : ℝ := circle_diameter / 2
  let distance : ℝ := rectangle_width - 2 * circle_radius
  distance = 4 := by sorry

end NUMINAMATH_CALUDE_distance_between_circle_centers_l535_53591


namespace NUMINAMATH_CALUDE_pascal_row20_sum_l535_53580

theorem pascal_row20_sum : Nat.choose 20 4 + Nat.choose 20 5 = 20349 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row20_sum_l535_53580


namespace NUMINAMATH_CALUDE_complement_intersection_equals_singleton_l535_53578

def U : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def A : Set ℤ := {x | -1 < x ∧ x < 3}
def B : Set ℤ := {x | x^2 - x - 2 ≤ 0}

theorem complement_intersection_equals_singleton :
  (U \ A) ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_singleton_l535_53578


namespace NUMINAMATH_CALUDE_unique_general_term_implies_m_eq_one_third_l535_53581

/-- Two geometric sequences satisfying given conditions -/
structure GeometricSequences (m : ℝ) :=
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (a_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (b_geom : ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q)
  (a_first : a 1 = m)
  (b_minus_a_1 : b 1 - a 1 = 1)
  (b_minus_a_2 : b 2 - a 2 = 2)
  (b_minus_a_3 : b 3 - a 3 = 3)
  (m_pos : m > 0)

/-- The uniqueness of the general term formula for sequence a -/
def uniqueGeneralTerm (m : ℝ) (gs : GeometricSequences m) :=
  ∃! q : ℝ, ∀ n : ℕ, gs.a (n + 1) = gs.a n * q

/-- Main theorem: If the general term formula of a_n is unique, then m = 1/3 -/
theorem unique_general_term_implies_m_eq_one_third (m : ℝ) (gs : GeometricSequences m) :
  uniqueGeneralTerm m gs → m = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_general_term_implies_m_eq_one_third_l535_53581


namespace NUMINAMATH_CALUDE_monthly_income_A_l535_53586

/-- Given the average monthly incomes of pairs of individuals, 
    prove that the monthly income of A is 4000. -/
theorem monthly_income_A (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 5050)
  (avg_bc : (b + c) / 2 = 6250)
  (avg_ac : (a + c) / 2 = 5200) :
  a = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_A_l535_53586


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_is_correct_l535_53584

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallest_divisible_by_1_to_10 : ℕ := 27720

/-- Proposition: smallest_divisible_by_1_to_10 is the smallest positive integer 
    divisible by all integers from 1 to 10 -/
theorem smallest_divisible_by_1_to_10_is_correct :
  (∀ n : ℕ, n > 0 ∧ n < smallest_divisible_by_1_to_10 → 
    ∃ m : ℕ, m ∈ Finset.range 10 ∧ n % (m + 1) ≠ 0) ∧
  (∀ m : ℕ, m ∈ Finset.range 10 → smallest_divisible_by_1_to_10 % (m + 1) = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_is_correct_l535_53584


namespace NUMINAMATH_CALUDE_mount_pilot_snow_amount_l535_53542

/-- The amount of snow on Mount Pilot in centimeters -/
def mount_pilot_snow (bald_snow billy_snow : ℝ) : ℝ :=
  (billy_snow * 100 + (billy_snow * 100 + bald_snow * 100 + 326) - bald_snow * 100) - billy_snow * 100

/-- Theorem stating that Mount Pilot received 326 cm of snow -/
theorem mount_pilot_snow_amount :
  mount_pilot_snow 1.5 3.5 = 326 := by
  sorry

#eval mount_pilot_snow 1.5 3.5

end NUMINAMATH_CALUDE_mount_pilot_snow_amount_l535_53542


namespace NUMINAMATH_CALUDE_set_relationship_l535_53551

-- Define the sets
def set1 : Set ℝ := {x | (1 : ℝ) / x ≤ 1}
def set2 : Set ℝ := {x | Real.log x ≥ 0}

-- Theorem statement
theorem set_relationship : Set.Subset set2 set1 ∧ ¬(set1 = set2) := by
  sorry

end NUMINAMATH_CALUDE_set_relationship_l535_53551


namespace NUMINAMATH_CALUDE_four_distinct_roots_l535_53505

theorem four_distinct_roots (m : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∀ (x : ℝ), x^2 - 4*|x| + 5 - m = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)))
  ↔ (1 < m ∧ m < 5) :=
sorry

end NUMINAMATH_CALUDE_four_distinct_roots_l535_53505


namespace NUMINAMATH_CALUDE_unique_odd_natural_from_primes_l535_53535

theorem unique_odd_natural_from_primes :
  ∃! (n : ℕ), 
    n % 2 = 1 ∧ 
    ∃ (p q : ℕ), 
      Prime p ∧ Prime q ∧ p > q ∧ 
      n = (p + q) / (p - q) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_odd_natural_from_primes_l535_53535


namespace NUMINAMATH_CALUDE_inequality_proof_l535_53529

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a / (b * (1 + c)) + b / (c * (1 + a)) + c / (a * (1 + b)) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l535_53529


namespace NUMINAMATH_CALUDE_bottles_to_buy_promotion_l535_53593

/-- Calculates the number of bottles to buy given a promotion and total bottles needed -/
def bottlesToBuy (bottlesNeeded : ℕ) (buyQuantity : ℕ) (freeQuantity : ℕ) : ℕ :=
  bottlesNeeded - (bottlesNeeded / (buyQuantity + freeQuantity)) * freeQuantity

/-- Proves that 8 bottles need to be bought given the promotion and number of people -/
theorem bottles_to_buy_promotion (numPeople : ℕ) (buyQuantity : ℕ) (freeQuantity : ℕ) :
  numPeople = 10 → buyQuantity = 4 → freeQuantity = 1 →
  bottlesToBuy numPeople buyQuantity freeQuantity = 8 :=
by
  sorry

#eval bottlesToBuy 10 4 1  -- Should output 8

end NUMINAMATH_CALUDE_bottles_to_buy_promotion_l535_53593


namespace NUMINAMATH_CALUDE_min_distance_line_circle_l535_53550

/-- The minimum distance between a point on the given line and a point on the given circle is √5/5 -/
theorem min_distance_line_circle :
  let line := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = t ∧ p.2 = 6 - 2*t}
  let circle := {q : ℝ × ℝ | (q.1 - 1)^2 + (q.2 + 2)^2 = 5}
  ∃ d : ℝ, d = Real.sqrt 5 / 5 ∧
    ∀ p ∈ line, ∀ q ∈ circle,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d ∧
      ∃ p' ∈ line, ∃ q' ∈ circle,
        Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_circle_l535_53550


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l535_53503

theorem fahrenheit_to_celsius (C F : ℝ) : 
  C = (4 / 7) * (F - 40) → C = 35 → F = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l535_53503


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l535_53525

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 180 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 42 + 2 * Real.sqrt 261 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l535_53525


namespace NUMINAMATH_CALUDE_travis_apple_sales_l535_53540

/-- Calculates the total money Travis takes home from selling apples -/
def total_money (total_apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  (total_apples / apples_per_box) * price_per_box

/-- Proves that Travis will take home $7000 -/
theorem travis_apple_sales : total_money 10000 50 35 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_travis_apple_sales_l535_53540


namespace NUMINAMATH_CALUDE_profit_maximized_at_70_l535_53511

/-- Represents the store's helmet sales scenario -/
structure HelmetStore where
  initialPrice : ℝ
  initialSales : ℝ
  priceReductionEffect : ℝ
  costPrice : ℝ

/-- Calculates the monthly profit for a given selling price -/
def monthlyProfit (store : HelmetStore) (sellingPrice : ℝ) : ℝ :=
  let salesVolume := store.initialSales + (store.initialPrice - sellingPrice) * store.priceReductionEffect
  (sellingPrice - store.costPrice) * salesVolume

/-- Theorem stating that 70 yuan maximizes the monthly profit -/
theorem profit_maximized_at_70 (store : HelmetStore) 
    (h1 : store.initialPrice = 80)
    (h2 : store.initialSales = 200)
    (h3 : store.priceReductionEffect = 20)
    (h4 : store.costPrice = 50) :
    ∀ x, monthlyProfit store 70 ≥ monthlyProfit store x := by
  sorry

#check profit_maximized_at_70

end NUMINAMATH_CALUDE_profit_maximized_at_70_l535_53511


namespace NUMINAMATH_CALUDE_four_circles_max_regions_l535_53524

/-- The maximum number of regions that n circles can divide a plane into -/
def max_regions (n : ℕ) : ℕ :=
  n * (n - 1) + 2

/-- Assumption that for n = 1, 2, 3, n circles divide the plane into at most 2^n parts -/
axiom max_regions_small (n : ℕ) (h : n ≤ 3) : max_regions n ≤ 2^n

/-- Theorem: The maximum number of regions that four circles can divide a plane into is 14 -/
theorem four_circles_max_regions : max_regions 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_four_circles_max_regions_l535_53524


namespace NUMINAMATH_CALUDE_equation_positive_root_implies_m_eq_neg_one_l535_53585

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x - 1) - m / (1 - x) = 2

-- Define the theorem
theorem equation_positive_root_implies_m_eq_neg_one :
  (∃ x : ℝ, x > 0 ∧ equation x m) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_implies_m_eq_neg_one_l535_53585


namespace NUMINAMATH_CALUDE_puppy_weight_l535_53543

/-- Represents the weight of animals in pounds -/
structure AnimalWeights where
  puppy : ℝ
  smaller_cat : ℝ
  larger_cat : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (w : AnimalWeights) : Prop :=
  w.puppy + 2 * w.smaller_cat + w.larger_cat = 38 ∧
  w.puppy + w.larger_cat = 3 * w.smaller_cat ∧
  w.puppy + 2 * w.smaller_cat = w.larger_cat

/-- The theorem stating the puppy's weight -/
theorem puppy_weight (w : AnimalWeights) (h : satisfies_conditions w) : w.puppy = 3.8 := by
  sorry

#check puppy_weight

end NUMINAMATH_CALUDE_puppy_weight_l535_53543


namespace NUMINAMATH_CALUDE_fraction_equality_l535_53569

theorem fraction_equality (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / (x + y) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l535_53569


namespace NUMINAMATH_CALUDE_nth_term_from_sum_l535_53589

/-- Given a sequence {a_n} where S_n = 3n^2 - 2n is the sum of its first n terms,
    prove that the n-th term of the sequence is a_n = 6n - 5 for all natural numbers n. -/
theorem nth_term_from_sum (n : ℕ) (a : ℕ → ℚ) (S : ℕ → ℚ) 
    (h : ∀ k, S k = 3 * k^2 - 2 * k) : 
  a n = 6 * n - 5 := by
  sorry

end NUMINAMATH_CALUDE_nth_term_from_sum_l535_53589


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l535_53560

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → 1 / a < 1) ∧ 
  (∃ a, 1 / a < 1 ∧ ¬(a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l535_53560


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l535_53562

/-- Given a car traveling for two hours with an average speed of 82.5 km/h
    and a speed of 90 km/h in the first hour, the speed in the second hour is 75 km/h. -/
theorem car_speed_second_hour
  (average_speed : ℝ)
  (first_hour_speed : ℝ)
  (h_average : average_speed = 82.5)
  (h_first : first_hour_speed = 90)
  : ∃ (second_hour_speed : ℝ),
    second_hour_speed = 75 ∧
    average_speed = (first_hour_speed + second_hour_speed) / 2 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l535_53562


namespace NUMINAMATH_CALUDE_best_player_hits_l535_53534

/-- Represents a baseball team -/
structure BaseballTeam where
  totalPlayers : ℕ
  averageHitsPerGame : ℕ
  gamesPlayed : ℕ
  otherPlayersAverageHits : ℕ
  otherPlayersGames : ℕ

/-- Calculates the total hits of the best player -/
def bestPlayerTotalHits (team : BaseballTeam) : ℕ :=
  team.averageHitsPerGame * team.gamesPlayed - 
  (team.totalPlayers - 1) * team.otherPlayersAverageHits

/-- Theorem stating the best player's total hits -/
theorem best_player_hits (team : BaseballTeam) 
  (h1 : team.totalPlayers = 11)
  (h2 : team.averageHitsPerGame = 15)
  (h3 : team.gamesPlayed = 5)
  (h4 : team.otherPlayersAverageHits = 6)
  (h5 : team.otherPlayersGames = 6) :
  bestPlayerTotalHits team = 25 := by
  sorry

#eval bestPlayerTotalHits { 
  totalPlayers := 11, 
  averageHitsPerGame := 15, 
  gamesPlayed := 5, 
  otherPlayersAverageHits := 6, 
  otherPlayersGames := 6
}

end NUMINAMATH_CALUDE_best_player_hits_l535_53534


namespace NUMINAMATH_CALUDE_probability_not_losing_l535_53527

theorem probability_not_losing (p_draw p_win : ℝ) :
  p_draw = 1/2 →
  p_win = 1/3 →
  p_draw + p_win = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_probability_not_losing_l535_53527


namespace NUMINAMATH_CALUDE_missing_number_proof_l535_53521

theorem missing_number_proof (x : ℝ) (y : ℝ) : 
  (12 + x + 42 + 78 + y) / 5 = 62 →
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 →
  y = 104 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l535_53521


namespace NUMINAMATH_CALUDE_bread_cost_l535_53564

/-- The cost of a loaf of bread, given the costs of ham and cake, and that the combined cost of ham and bread equals the cost of cake. -/
theorem bread_cost (ham_cost cake_cost : ℕ) (h1 : ham_cost = 150) (h2 : cake_cost = 200)
  (h3 : ∃ (bread_cost : ℕ), bread_cost + ham_cost = cake_cost) : 
  ∃ (bread_cost : ℕ), bread_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_l535_53564


namespace NUMINAMATH_CALUDE_binomial_7_2_l535_53577

theorem binomial_7_2 : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_2_l535_53577


namespace NUMINAMATH_CALUDE_ice_cream_truck_expenses_l535_53554

/-- Proves that for an ice cream truck business where each cone costs $5, 
    if 200 cones are sold and a $200 profit is made, 
    then the expenses are 80% of the total sales. -/
theorem ice_cream_truck_expenses (cone_price : ℝ) (cones_sold : ℕ) (profit : ℝ) :
  cone_price = 5 →
  cones_sold = 200 →
  profit = 200 →
  let total_sales := cone_price * cones_sold
  let expenses := total_sales - profit
  expenses / total_sales = 0.8 := by sorry

end NUMINAMATH_CALUDE_ice_cream_truck_expenses_l535_53554


namespace NUMINAMATH_CALUDE_smaug_hoard_value_l535_53558

/-- Calculates the total value of Smaug's hoard in copper coins -/
def smaugsHoardValue (goldCoins silverCoins copperCoins : ℕ) 
  (silverToCopperRatio goldToSilverRatio : ℕ) : ℕ :=
  goldCoins * goldToSilverRatio * silverToCopperRatio + 
  silverCoins * silverToCopperRatio + 
  copperCoins

/-- Proves that Smaug's hoard has a total value of 2913 copper coins -/
theorem smaug_hoard_value : 
  smaugsHoardValue 100 60 33 8 3 = 2913 := by
  sorry

end NUMINAMATH_CALUDE_smaug_hoard_value_l535_53558


namespace NUMINAMATH_CALUDE_divisible_by_fifteen_l535_53528

theorem divisible_by_fifteen (n : ℤ) : 15 ∣ (7*n + 5*n^3 + 3*n^5) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_fifteen_l535_53528


namespace NUMINAMATH_CALUDE_volleyball_team_theorem_l535_53541

def volleyball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) : ℕ :=
  quadruplets * (Nat.choose (total_players - quadruplets) (starters - 1))

theorem volleyball_team_theorem :
  volleyball_team_selection 16 4 6 = 3168 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_theorem_l535_53541


namespace NUMINAMATH_CALUDE_arrangement_count_is_518400_l535_53508

/-- The number of ways to arrange 4 math books and 6 history books with specific conditions -/
def arrangement_count : ℕ :=
  let math_books : ℕ := 4
  let history_books : ℕ := 6
  let math_ends : ℕ := math_books * (math_books - 1)
  let consecutive_history : ℕ := Nat.choose history_books 2
  let remaining_units : ℕ := 5  -- 4 single history books + 1 double-history unit
  let middle_arrangements : ℕ := Nat.factorial remaining_units
  let remaining_math_placements : ℕ := Nat.choose remaining_units 2 * Nat.factorial 2
  math_ends * consecutive_history * middle_arrangements * remaining_math_placements

/-- Theorem stating that the number of arrangements is 518,400 -/
theorem arrangement_count_is_518400 : arrangement_count = 518400 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_518400_l535_53508


namespace NUMINAMATH_CALUDE_equal_incircle_radii_of_original_triangles_l535_53504

/-- A structure representing a triangle with an inscribed circle -/
structure TriangleWithIncircle where
  vertices : Fin 3 → ℝ × ℝ
  incircle_center : ℝ × ℝ
  incircle_radius : ℝ

/-- A structure representing the configuration of two intersecting triangles -/
structure IntersectingTriangles where
  triangle1 : TriangleWithIncircle
  triangle2 : TriangleWithIncircle
  hexagon_vertices : Fin 6 → ℝ × ℝ
  small_triangles : Fin 6 → TriangleWithIncircle

/-- The theorem statement -/
theorem equal_incircle_radii_of_original_triangles 
  (config : IntersectingTriangles)
  (h_equal_small_radii : ∀ i j : Fin 6, (config.small_triangles i).incircle_radius = (config.small_triangles j).incircle_radius) :
  config.triangle1.incircle_radius = config.triangle2.incircle_radius :=
sorry

end NUMINAMATH_CALUDE_equal_incircle_radii_of_original_triangles_l535_53504


namespace NUMINAMATH_CALUDE_divisibility_by_nine_highest_power_of_three_in_M_l535_53563

/-- The integer formed by concatenating 2-digit integers from 15 to 95 -/
def M : ℕ := sorry

/-- The sum of digits of M -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is divisible by 9 if and only if the sum of its digits is divisible by 9 -/
theorem divisibility_by_nine (n : ℕ) : n % 9 = 0 ↔ sum_of_digits n % 9 = 0 := sorry

/-- The highest power of 3 that divides M is 3^2 -/
theorem highest_power_of_three_in_M : 
  ∃ (k : ℕ), M % (3^3) ≠ 0 ∧ M % (3^2) = 0 := sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_highest_power_of_three_in_M_l535_53563


namespace NUMINAMATH_CALUDE_three_true_propositions_l535_53571

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

theorem three_true_propositions
  (a : Line3D) (α β : Plane3D) (h_diff : α ≠ β) :
  (perpendicular a α ∧ perpendicular a β → parallel α β) ∧
  (perpendicular a α ∧ parallel α β → perpendicular a β) ∧
  (perpendicular a β ∧ parallel α β → perpendicular a α) :=
by sorry

end NUMINAMATH_CALUDE_three_true_propositions_l535_53571


namespace NUMINAMATH_CALUDE_contrapositive_example_l535_53588

theorem contrapositive_example (a b : ℝ) :
  (¬(a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l535_53588


namespace NUMINAMATH_CALUDE_coefficient_x2y3z2_is_120_l535_53597

/-- The coefficient of x^2 * y^3 * z^2 in the expansion of (x-y)(x+2y+z)^6 -/
def coefficient_x2y3z2 (x y z : ℤ) : ℤ :=
  let expansion := (x - y) * (x + 2*y + z)^6
  -- The actual computation of the coefficient would go here
  120

/-- Theorem stating that the coefficient of x^2 * y^3 * z^2 in the expansion of (x-y)(x+2y+z)^6 is 120 -/
theorem coefficient_x2y3z2_is_120 (x y z : ℤ) :
  coefficient_x2y3z2 x y z = 120 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y3z2_is_120_l535_53597


namespace NUMINAMATH_CALUDE_reciprocal_of_proper_fraction_greater_than_one_l535_53515

-- Define a proper fraction
def ProperFraction (n d : ℕ) : Prop := 0 < n ∧ n < d

-- Theorem statement
theorem reciprocal_of_proper_fraction_greater_than_one {n d : ℕ} (h : ProperFraction n d) :
  (d : ℝ) / (n : ℝ) > 1 := by
  sorry


end NUMINAMATH_CALUDE_reciprocal_of_proper_fraction_greater_than_one_l535_53515


namespace NUMINAMATH_CALUDE_donuts_per_box_l535_53538

/-- Proves that the number of donuts per box is 10 given the conditions of Jeff's donut-making and eating scenario. -/
theorem donuts_per_box :
  let total_donuts := 10 * 12
  let jeff_eaten := 1 * 12
  let chris_eaten := 8
  let boxes := 10
  let remaining_donuts := total_donuts - jeff_eaten - chris_eaten
  remaining_donuts / boxes = 10 := by
  sorry

end NUMINAMATH_CALUDE_donuts_per_box_l535_53538


namespace NUMINAMATH_CALUDE_senior_citizens_average_age_l535_53512

theorem senior_citizens_average_age
  (total_members : ℕ)
  (overall_average_age : ℚ)
  (num_women : ℕ)
  (num_men : ℕ)
  (num_seniors : ℕ)
  (women_average_age : ℚ)
  (men_average_age : ℚ)
  (h1 : total_members = 60)
  (h2 : overall_average_age = 30)
  (h3 : num_women = 25)
  (h4 : num_men = 20)
  (h5 : num_seniors = 15)
  (h6 : women_average_age = 28)
  (h7 : men_average_age = 35)
  (h8 : total_members = num_women + num_men + num_seniors) :
  (total_members * overall_average_age - num_women * women_average_age - num_men * men_average_age) / num_seniors = 80 / 3 :=
by sorry

end NUMINAMATH_CALUDE_senior_citizens_average_age_l535_53512


namespace NUMINAMATH_CALUDE_min_odd_integers_l535_53555

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 28)
  (sum2 : a + b + c + d = 46)
  (sum3 : a + b + c + d + e + f = 65) :
  ∃ (x : Finset ℤ), x ⊆ {a, b, c, d, e, f} ∧ x.card = 1 ∧ ∀ i ∈ x, Odd i ∧
  ∀ (y : Finset ℤ), y ⊆ {a, b, c, d, e, f} ∧ (∀ i ∈ y, Odd i) → y.card ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_odd_integers_l535_53555


namespace NUMINAMATH_CALUDE_smallest_integer_divisible_l535_53539

theorem smallest_integer_divisible (x : ℤ) : x = 36629 ↔ 
  (∀ y : ℤ, y < x → ¬(∃ k₁ k₂ k₃ k₄ : ℤ, 
    2 * y + 2 = 33 * k₁ ∧ 
    2 * y + 2 = 44 * k₂ ∧ 
    2 * y + 2 = 55 * k₃ ∧ 
    2 * y + 2 = 666 * k₄)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℤ, 
    2 * x + 2 = 33 * k₁ ∧ 
    2 * x + 2 = 44 * k₂ ∧ 
    2 * x + 2 = 55 * k₃ ∧ 
    2 * x + 2 = 666 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_divisible_l535_53539


namespace NUMINAMATH_CALUDE_divisibility_by_19_l535_53595

theorem divisibility_by_19 (n : ℕ) : ∃ k : ℤ, 
  120 * 10^(n+2) + 3 * ((10^(n+1) - 1) / 9) * 100 + 8 = 19 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_19_l535_53595


namespace NUMINAMATH_CALUDE_unique_square_multiple_of_five_in_range_l535_53530

theorem unique_square_multiple_of_five_in_range : 
  ∃! x : ℕ, 
    (∃ n : ℕ, x = n^2) ∧ 
    (x % 5 = 0) ∧ 
    (50 < x) ∧ 
    (x < 120) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_square_multiple_of_five_in_range_l535_53530


namespace NUMINAMATH_CALUDE_triangle_interior_lines_sum_bound_l535_53575

-- Define a triangle with side lengths x, y, z
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  hxy : x ≤ y
  hyz : y ≤ z

-- Define the sum s
def s (t : Triangle) (XX' YY' ZZ' : ℝ) : ℝ := XX' + YY' + ZZ'

-- Theorem statement
theorem triangle_interior_lines_sum_bound (t : Triangle) 
  (XX' YY' ZZ' : ℝ) (hXX' : XX' ≥ 0) (hYY' : YY' ≥ 0) (hZZ' : ZZ' ≥ 0) : 
  s t XX' YY' ZZ' ≤ t.x + t.y + t.z := by
  sorry

end NUMINAMATH_CALUDE_triangle_interior_lines_sum_bound_l535_53575


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l535_53596

/-- A cone with an isosceles right triangle cross-section and volume 8π/3 has lateral surface area 4√2π -/
theorem cone_lateral_surface_area (V : ℝ) (r h l : ℝ) : 
  V = (8 / 3) * Real.pi →  -- Volume condition
  V = (1 / 3) * Real.pi * r^2 * h →  -- Volume formula
  r = (Real.sqrt 2 * l) / 2 →  -- Relationship between radius and slant height
  h = r →  -- Height equals radius in isosceles right triangle
  (Real.pi * r * l) = 4 * Real.sqrt 2 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l535_53596


namespace NUMINAMATH_CALUDE_count_valid_integers_l535_53547

def is_valid_integer (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 100 = 45 ∧ n % 90 = 0

theorem count_valid_integers :
  ∃! (count : ℕ), ∃ (S : Finset ℕ),
    S.card = count ∧
    (∀ n, n ∈ S ↔ is_valid_integer n) ∧
    count = 9 := by sorry

end NUMINAMATH_CALUDE_count_valid_integers_l535_53547


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l535_53549

theorem chess_tournament_participants (n : ℕ) : 
  (∃ (y : ℚ), 2 * y + n * y = (n + 2) * (n + 1) / 2) → 
  (n = 7 ∨ n = 14) := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l535_53549


namespace NUMINAMATH_CALUDE_eliminate_denominators_l535_53590

theorem eliminate_denominators (x : ℝ) : 
  (3 * x + (2 * x - 1) / 3 = 3 - (x + 1) / 2) ↔ 
  (18 * x + 2 * (2 * x - 1) = 18 - 3 * (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l535_53590


namespace NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_parameter_l535_53507

/-- Given an ellipse and a hyperbola that are tangent, prove that the parameter m of the hyperbola is 5/9 -/
theorem tangent_ellipse_hyperbola_parameter (x y m : ℝ) : 
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 4) →  -- Existence of points satisfying both equations
  (∀ x y, x^2 + 9*y^2 = 9 → x^2 - m*(y+3)^2 ≥ 4) →  -- Hyperbola does not intersect interior of ellipse
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 4) →  -- Existence of a common point
  m = 5/9 := by sorry

end NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_parameter_l535_53507


namespace NUMINAMATH_CALUDE_males_band_not_orchestra_l535_53583

/-- Represents the number of students in various categories of the school's music program -/
structure MusicProgram where
  female_band : ℕ
  male_band : ℕ
  female_orchestra : ℕ
  male_orchestra : ℕ
  female_both : ℕ
  left_band : ℕ
  total_either : ℕ

/-- Theorem stating the number of males in the band who are not in the orchestra -/
theorem males_band_not_orchestra (mp : MusicProgram) : 
  mp.female_band = 120 →
  mp.male_band = 90 →
  mp.female_orchestra = 70 →
  mp.male_orchestra = 110 →
  mp.female_both = 55 →
  mp.left_band = 10 →
  mp.total_either = 250 →
  mp.male_band - (mp.male_band + mp.male_orchestra - (mp.total_either - ((mp.female_band + mp.female_orchestra - mp.female_both) + mp.left_band))) = 15 := by
  sorry


end NUMINAMATH_CALUDE_males_band_not_orchestra_l535_53583


namespace NUMINAMATH_CALUDE_min_seats_for_adjacent_seating_l535_53546

/-- Represents a seating arrangement on a train -/
def SeatingArrangement (total_seats : ℕ) (occupied_seats : ℕ) : Prop :=
  occupied_seats ≤ total_seats

/-- Checks if the next person must sit next to someone already seated -/
def ForceAdjacentSeat (total_seats : ℕ) (occupied_seats : ℕ) : Prop :=
  ∀ (empty_seat : ℕ), empty_seat ≤ total_seats - occupied_seats →
    ∃ (adjacent_seat : ℕ), adjacent_seat ≤ total_seats ∧
      (adjacent_seat = empty_seat + 1 ∨ adjacent_seat = empty_seat - 1) ∧
      (adjacent_seat ≤ occupied_seats)

/-- The main theorem to prove -/
theorem min_seats_for_adjacent_seating :
  ∃ (min_occupied : ℕ),
    SeatingArrangement 150 min_occupied ∧
    ForceAdjacentSeat 150 min_occupied ∧
    (∀ (n : ℕ), n < min_occupied → ¬ForceAdjacentSeat 150 n) ∧
    min_occupied = 37 := by
  sorry

end NUMINAMATH_CALUDE_min_seats_for_adjacent_seating_l535_53546


namespace NUMINAMATH_CALUDE_gelato_sundae_combinations_l535_53599

theorem gelato_sundae_combinations :
  (Finset.univ.filter (fun s : Finset (Fin 8) => s.card = 3)).card = 56 := by
  sorry

end NUMINAMATH_CALUDE_gelato_sundae_combinations_l535_53599


namespace NUMINAMATH_CALUDE_mom_bought_14_packages_l535_53570

/-- The number of packages Mom bought -/
def num_packages (total_shirts : ℕ) (shirts_per_package : ℕ) : ℕ :=
  total_shirts / shirts_per_package

/-- Proof that Mom bought 14 packages of white t-shirts -/
theorem mom_bought_14_packages :
  num_packages 70 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_14_packages_l535_53570


namespace NUMINAMATH_CALUDE_quadratic_inequality_l535_53587

theorem quadratic_inequality (a b : ℝ) : ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |x₀^2 + a*x₀ + b| + a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l535_53587


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l535_53567

/-- Proves that -0.000008691 is equal to -8.691×10^(-6) in scientific notation -/
theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  -0.000008691 = a * 10^n ∧ 
  1 ≤ |a| ∧ 
  |a| < 10 ∧ 
  a = -8.691 ∧ 
  n = -6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l535_53567


namespace NUMINAMATH_CALUDE_cube_volume_problem_l535_53582

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) →  -- Ensure positive side length
  (a^3 - ((a - 1) * a * (a + 1)) = 5) →
  (a^3 = 125) :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l535_53582


namespace NUMINAMATH_CALUDE_bicycle_not_in_motion_time_l535_53556

-- Define the constants
def total_distance : ℝ := 22.5
def bert_ride_speed : ℝ := 8
def bert_walk_speed : ℝ := 5
def al_walk_speed : ℝ := 4
def al_ride_speed : ℝ := 10

-- Define the theorem
theorem bicycle_not_in_motion_time :
  ∃ (x : ℝ),
    (x / bert_ride_speed + (total_distance - x) / bert_walk_speed =
     x / al_walk_speed + (total_distance - x) / al_ride_speed) ∧
    ((x / al_walk_speed - x / bert_ride_speed) * 60 = 75) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_not_in_motion_time_l535_53556


namespace NUMINAMATH_CALUDE_base4_division_theorem_l535_53513

/-- Converts a base 4 number to base 10 --/
def base4_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

/-- Converts a base 10 number to base 4 --/
def base10_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem base4_division_theorem :
  let dividend := [3, 1, 2, 3]  -- 3213₄ in reverse order
  let divisor := [3, 1]         -- 13₄ in reverse order
  let quotient := [1, 0, 2]     -- 201₄ in reverse order
  (base4_to_base10 dividend) / (base4_to_base10 divisor) = base4_to_base10 quotient :=
by sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l535_53513


namespace NUMINAMATH_CALUDE_amy_bob_games_l535_53594

theorem amy_bob_games (n : ℕ) (h : n = 9) :
  let total_combinations := Nat.choose n 3
  let games_per_player := total_combinations / n
  let games_together := games_per_player / 4
  games_together = 7 := by
  sorry

end NUMINAMATH_CALUDE_amy_bob_games_l535_53594


namespace NUMINAMATH_CALUDE_sqrt_three_squared_four_to_fourth_l535_53561

theorem sqrt_three_squared_four_to_fourth : Real.sqrt (3^2 * 4^4) = 48 := by sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_four_to_fourth_l535_53561


namespace NUMINAMATH_CALUDE_burger_slices_l535_53522

theorem burger_slices (total_burgers : ℕ) (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) (friend4_slices : ℕ) (era_slices : ℕ) :
  total_burgers = 5 →
  friend1_slices = 1 →
  friend2_slices = 2 →
  friend3_slices = 3 →
  friend4_slices = 3 →
  era_slices = 1 →
  (friend1_slices + friend2_slices + friend3_slices + friend4_slices + era_slices) / total_burgers = 2 :=
by sorry

end NUMINAMATH_CALUDE_burger_slices_l535_53522


namespace NUMINAMATH_CALUDE_courtyard_width_prove_courtyard_width_l535_53574

/-- The width of a rectangular courtyard given specific conditions -/
theorem courtyard_width : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (length width stone_side num_stones : ℝ) =>
    length = 30 ∧
    stone_side = 2 ∧
    num_stones = 135 ∧
    length * width = num_stones * stone_side * stone_side →
    width = 18

/-- Proof of the courtyard width theorem -/
theorem prove_courtyard_width :
  ∃ (length width stone_side num_stones : ℝ),
    courtyard_width length width stone_side num_stones :=
by
  sorry

end NUMINAMATH_CALUDE_courtyard_width_prove_courtyard_width_l535_53574


namespace NUMINAMATH_CALUDE_benny_turnips_l535_53544

theorem benny_turnips (melanie_turnips benny_turnips total_turnips : ℕ) : 
  melanie_turnips = 139 → total_turnips = 252 → benny_turnips = total_turnips - melanie_turnips → 
  benny_turnips = 113 := by
  sorry

end NUMINAMATH_CALUDE_benny_turnips_l535_53544


namespace NUMINAMATH_CALUDE_swimmer_speed_is_4_l535_53520

/-- The swimmer's speed in still water -/
def swimmer_speed : ℝ := 4

/-- The speed of the water current -/
def current_speed : ℝ := 1

/-- The time taken to swim against the current -/
def swim_time : ℝ := 2

/-- The distance swum against the current -/
def swim_distance : ℝ := 6

/-- Theorem stating that the swimmer's speed in still water is 4 km/h -/
theorem swimmer_speed_is_4 :
  swimmer_speed = 4 ∧
  current_speed = 1 ∧
  swim_time = 2 ∧
  swim_distance = 6 →
  swimmer_speed = swim_distance / swim_time + current_speed :=
by sorry

end NUMINAMATH_CALUDE_swimmer_speed_is_4_l535_53520


namespace NUMINAMATH_CALUDE_opposite_of_2023_l535_53568

theorem opposite_of_2023 : 
  ∀ (x : ℤ), x = 2023 → -x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l535_53568


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_fourth_quadrant_condition_l535_53573

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6)

theorem purely_imaginary_condition (a : ℝ) :
  z a = Complex.I * (z a).im → a = 1 := by sorry

theorem fourth_quadrant_condition (a : ℝ) :
  (z a).re > 0 ∧ (z a).im < 0 → a > -1 ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_fourth_quadrant_condition_l535_53573


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l535_53548

theorem arccos_one_over_sqrt_two (π : Real) : 
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l535_53548


namespace NUMINAMATH_CALUDE_triangle_max_side_length_l535_53518

theorem triangle_max_side_length (P Q R : Real) (a b : Real) :
  -- Triangle angles
  P + Q + R = Real.pi →
  -- Given condition
  Real.cos (3 * P) + Real.cos (3 * Q) + Real.cos (3 * R) = 1 →
  -- Two sides have lengths 12 and 15
  a = 12 ∧ b = 15 →
  -- Maximum length of the third side
  ∃ c : Real, c ≤ 27 ∧ 
    ∀ c' : Real, (c' ^ 2 ≤ a ^ 2 + b ^ 2 - 2 * a * b * Real.cos R) → c' ≤ c :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_side_length_l535_53518


namespace NUMINAMATH_CALUDE_box_volume_less_than_500_l535_53533

def box_volume (x : ℕ) : ℕ := (x + 3) * (x - 3) * (x^2 + 9)

theorem box_volume_less_than_500 :
  ∀ x : ℕ, x > 0 → (box_volume x < 500 ↔ x = 4 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_box_volume_less_than_500_l535_53533


namespace NUMINAMATH_CALUDE_units_digit_of_18_power_l535_53557

theorem units_digit_of_18_power : ∃ n : ℕ, (18^(18*(7^7))) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_18_power_l535_53557


namespace NUMINAMATH_CALUDE_complex_sum_simplification_l535_53517

theorem complex_sum_simplification : 
  ((-2 + Complex.I * Real.sqrt 7) / 3) ^ 4 + ((-2 - Complex.I * Real.sqrt 7) / 3) ^ 4 = 242 / 81 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_simplification_l535_53517


namespace NUMINAMATH_CALUDE_no_p_q_for_all_x_divisible_by_3_l535_53552

theorem no_p_q_for_all_x_divisible_by_3 : 
  ¬ ∃ (p q : ℤ), ∀ (x : ℤ), (3 : ℤ) ∣ (x^2 + p*x + q) := by
  sorry

end NUMINAMATH_CALUDE_no_p_q_for_all_x_divisible_by_3_l535_53552


namespace NUMINAMATH_CALUDE_quadratic_factorization_l535_53537

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l535_53537


namespace NUMINAMATH_CALUDE_smallest_constant_term_l535_53501

theorem smallest_constant_term (a b c d e : ℤ) :
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = 1 ∨ x = -3 ∨ x = 7 ∨ x = -2/5) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∃ a' b' c' d' : ℤ, ∀ x : ℚ, a' * x^4 + b' * x^3 + c' * x^2 + d' * x + e' = 0 ↔ 
      x = 1 ∨ x = -3 ∨ x = 7 ∨ x = -2/5) →
    e ≤ e') →
  e = 42 := by
sorry

end NUMINAMATH_CALUDE_smallest_constant_term_l535_53501


namespace NUMINAMATH_CALUDE_first_group_size_l535_53579

/-- Represents the work rate of a group of people -/
structure WorkRate where
  people : ℕ
  work : ℕ
  days : ℕ

/-- The work rate of the first group -/
def first_group : WorkRate :=
  { people := 0,  -- We don't know this value yet
    work := 3,
    days := 3 }

/-- The work rate of the second group -/
def second_group : WorkRate :=
  { people := 9,
    work := 9,
    days := 3 }

/-- Calculates the daily work rate -/
def daily_rate (wr : WorkRate) : ℚ :=
  wr.work / wr.days

theorem first_group_size :
  first_group.people = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l535_53579


namespace NUMINAMATH_CALUDE_smallest_b_is_correct_l535_53536

/-- A function that checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

/-- The smallest integer b > 5 for which 43_b is a perfect cube -/
def smallest_b : ℕ := 6

theorem smallest_b_is_correct :
  (smallest_b > 5) ∧ 
  (is_perfect_cube (4 * smallest_b + 3)) ∧ 
  (∀ b : ℕ, b > 5 ∧ b < smallest_b → ¬(is_perfect_cube (4 * b + 3))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_is_correct_l535_53536
