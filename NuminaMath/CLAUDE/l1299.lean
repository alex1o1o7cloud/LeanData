import Mathlib

namespace min_value_abc_l1299_129909

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 → (a + b) / (a * b * c) ≤ (x + y) / (x * y * z)) →
  (a + b) / (a * b * c) = 16 :=
sorry

end min_value_abc_l1299_129909


namespace apple_sale_percentage_l1299_129920

/-- The percentage of apples sold by a fruit seller -/
theorem apple_sale_percentage (original : Real) (remaining : Real) 
  (h1 : original = 2499.9987500006246)
  (h2 : remaining = 500) :
  let sold := original - remaining
  let percentage := (sold / original) * 100
  ∃ ε > 0, abs (percentage - 80) < ε :=
by sorry

end apple_sale_percentage_l1299_129920


namespace swimming_speed_is_15_l1299_129979

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  /-- The man's swimming speed in still water (km/h) -/
  v : ℝ
  /-- The speed of the stream (km/h) -/
  s : ℝ
  /-- The time it takes to swim downstream (hours) -/
  t : ℝ
  /-- Assertion that it takes twice as long to swim upstream -/
  upstream_time : (v - s) * (2 * t) = (v + s) * t
  /-- The speed of the stream is 5 km/h -/
  stream_speed : s = 5

/-- Theorem stating that the man's swimming speed in still water is 15 km/h -/
theorem swimming_speed_is_15 (scenario : SwimmingScenario) : scenario.v = 15 := by
  sorry

end swimming_speed_is_15_l1299_129979


namespace product_of_roots_quadratic_l1299_129922

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) :
  x₁^2 - 3*x₁ + 2 = 0 → x₂^2 - 3*x₂ + 2 = 0 → x₁ * x₂ = 2 := by
  sorry

end product_of_roots_quadratic_l1299_129922


namespace length_AB_trajectory_C_l1299_129972

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 5*y^2 = 5

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ellipse_foci : A = (-2, 0) ∧ B = (2, 0))
  (angle_relation : ∀ (θA θB θC : ℝ), 
    Real.sin θB - Real.sin θA = Real.sin θC → 
    θA + θB + θC = π)

-- Statement 1: Length of AB is 4
theorem length_AB (t : Triangle) : 
  Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) = 4 :=
sorry

-- Statement 2: Trajectory of C
theorem trajectory_C (t : Triangle) (x y : ℝ) :
  (∃ (C : ℝ × ℝ), t.C = C ∧ x > 1) →
  (x^2 - y^2/3 = 1) :=
sorry

end length_AB_trajectory_C_l1299_129972


namespace regression_analysis_l1299_129912

/-- Unit prices -/
def unit_prices : List ℝ := [4, 5, 6, 7, 8, 9]

/-- Sales volumes -/
def sales_volumes : List ℝ := [90, 84, 83, 80, 75, 68]

/-- Empirical regression equation -/
def regression_equation (x : ℝ) (a : ℝ) : ℝ := -4 * x + a

theorem regression_analysis :
  let avg_sales := (sales_volumes.sum) / (sales_volumes.length : ℝ)
  let slope := -4
  let a := avg_sales + 4 * ((unit_prices.sum) / (unit_prices.length : ℝ))
  (avg_sales = 80) ∧ 
  (slope = -4) ∧
  (regression_equation 10 a = 66) := by sorry

end regression_analysis_l1299_129912


namespace inscribed_circle_diameter_right_triangle_l1299_129984

/-- In a right-angled triangle with legs a and b, hypotenuse c, and inscribed circle of radius r,
    the diameter of the inscribed circle is a + b - c. -/
theorem inscribed_circle_diameter_right_triangle 
  (a b c r : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0) 
  (h_inscribed : r = (a + b - c) / 2) : 
  2 * r = a + b - c := by
  sorry

end inscribed_circle_diameter_right_triangle_l1299_129984


namespace polynomial_simplification_l1299_129933

theorem polynomial_simplification (w : ℝ) : 
  2 * w^2 + 3 - 4 * w^2 + 2 * w - 6 * w + 4 = -2 * w^2 - 4 * w + 7 := by
  sorry

end polynomial_simplification_l1299_129933


namespace male_attendees_fraction_l1299_129925

theorem male_attendees_fraction (M F : ℚ) : 
  M + F = 1 →
  (3/4 : ℚ) * M + (5/6 : ℚ) * F = 7/9 →
  M = 2/3 := by
sorry

end male_attendees_fraction_l1299_129925


namespace son_age_problem_l1299_129990

theorem son_age_problem (father_age : ℕ) (son_age : ℕ) : 
  father_age = 40 ∧ 
  father_age = 4 * son_age ∧ 
  father_age + 20 = 2 * (son_age + 20) → 
  son_age = 10 :=
by sorry

end son_age_problem_l1299_129990


namespace triangle_properties_l1299_129923

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c)
  (h2 : t.c = Real.sqrt 7)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2) :
  t.C = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry

end triangle_properties_l1299_129923


namespace f_symmetry_l1299_129955

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^7 + a*x^5 + b*x - 5

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b (-3) = 5 → f a b 3 = -15 := by
  sorry

end f_symmetry_l1299_129955


namespace proposition_uses_or_l1299_129921

-- Define the equation
def equation (x : ℝ) : Prop := x^2 = 4

-- Define the solution set
def solution_set : Set ℝ := {2, -2}

-- Define the proposition
def proposition : Prop := ∀ x, equation x ↔ x ∈ solution_set

-- Theorem: The proposition uses the "or" conjunction
theorem proposition_uses_or : 
  (∀ x, equation x ↔ (x = 2 ∨ x = -2)) ↔ proposition := by sorry

end proposition_uses_or_l1299_129921


namespace original_equals_scientific_l1299_129970

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The original number -/
def original_number : ℕ := 346000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation := {
  coefficient := 3.46
  exponent := 8
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end original_equals_scientific_l1299_129970


namespace problem_solution_l1299_129981

theorem problem_solution : 
  (Real.sqrt 8 - Real.sqrt 2 + 2 * Real.sqrt (1/2) = 2 * Real.sqrt 2) ∧
  (Real.sqrt 12 - 9 * Real.sqrt (1/3) + |2 - Real.sqrt 3| = 2 - 2 * Real.sqrt 3) := by
  sorry

end problem_solution_l1299_129981


namespace geometric_sequence_tan_property_l1299_129998

/-- Given a geometric sequence {a_n} where a₂a₆ + 2a₄² = π, prove that tan(a₃a₅) = √3 -/
theorem geometric_sequence_tan_property (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n)
  (h_condition : a 2 * a 6 + 2 * (a 4)^2 = Real.pi) :
  Real.tan (a 3 * a 5) = Real.sqrt 3 := by
  sorry

end geometric_sequence_tan_property_l1299_129998


namespace train_length_calculation_l1299_129942

/-- Calculates the length of a train given its speed, the platform length, and the time taken to cross the platform. -/
theorem train_length_calculation (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 55 → 
  platform_length = 620 → 
  crossing_time = 71.99424046076314 → 
  ∃ (train_length : ℝ), (train_length ≥ 479.9 ∧ train_length ≤ 480.1) := by
  sorry

end train_length_calculation_l1299_129942


namespace compare_base_6_and_base_2_l1299_129928

def base_6_to_decimal (n : ℕ) : ℕ := 
  (n / 10) * 6 + (n % 10)

def base_2_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 4 + ((n / 10) % 10) * 2 + (n % 10)

theorem compare_base_6_and_base_2 : 
  base_6_to_decimal 12 > base_2_to_decimal 101 := by
  sorry

end compare_base_6_and_base_2_l1299_129928


namespace sector_area_l1299_129916

theorem sector_area (θ : Real) (r : Real) (h1 : θ = 72 * π / 180) (h2 : r = 20) :
  (1 / 2) * θ * r^2 = 80 * π := by
  sorry

end sector_area_l1299_129916


namespace probability_play_one_instrument_l1299_129963

/-- Given a population with the following properties:
  * The total population is 10000
  * One-third of the population plays at least one instrument
  * 450 people play two or more instruments
  This theorem states that the probability of a randomly selected person
  playing exactly one instrument is 0.2883 -/
theorem probability_play_one_instrument (total_population : ℕ)
  (plays_at_least_one : ℕ) (plays_two_or_more : ℕ) :
  total_population = 10000 →
  plays_at_least_one = total_population / 3 →
  plays_two_or_more = 450 →
  (plays_at_least_one - plays_two_or_more : ℚ) / total_population = 2883 / 10000 := by
  sorry

end probability_play_one_instrument_l1299_129963


namespace polynomial_expansion_value_l1299_129918

/-- The value of a in the expansion of (x+y)^7 -/
theorem polynomial_expansion_value (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 
  (21 * a^5 * b^2 = 35 * a^4 * b^3) →
  a = 5/8 := by
sorry

end polynomial_expansion_value_l1299_129918


namespace total_migration_l1299_129961

/-- The number of bird families that flew away for the winter -/
def total_migrated : ℕ := 118

/-- The number of bird families that flew to Africa -/
def africa_migrated : ℕ := 38

/-- The number of bird families that flew to Asia -/
def asia_migrated : ℕ := 80

/-- Theorem: The total number of bird families that migrated is equal to the sum of those that flew to Africa and Asia -/
theorem total_migration :
  total_migrated = africa_migrated + asia_migrated := by
sorry

end total_migration_l1299_129961


namespace trapezoid_area_is_80_l1299_129982

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid :=
  (long_base : ℝ)
  (base_angle : ℝ)
  (h : 0 < long_base)
  (angle_h : 0 < base_angle ∧ base_angle < π / 2)

/-- The area of an isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem trapezoid_area_is_80 (t : IsoscelesTrapezoid) 
  (h1 : t.long_base = 16)
  (h2 : t.base_angle = Real.arcsin 0.8) :
  trapezoid_area t = 80 :=
sorry

end trapezoid_area_is_80_l1299_129982


namespace total_desks_is_1776_total_desks_within_capacity_l1299_129948

/-- Represents the total number of classrooms in the school. -/
def total_classrooms : ℕ := 50

/-- Represents the number of desks in classrooms of type 1. -/
def desks_type1 : ℕ := 45

/-- Represents the number of desks in classrooms of type 2. -/
def desks_type2 : ℕ := 38

/-- Represents the number of desks in classrooms of type 3. -/
def desks_type3 : ℕ := 32

/-- Represents the number of desks in classrooms of type 4. -/
def desks_type4 : ℕ := 25

/-- Represents the fraction of classrooms of type 1. -/
def fraction_type1 : ℚ := 3 / 10

/-- Represents the fraction of classrooms of type 2. -/
def fraction_type2 : ℚ := 1 / 4

/-- Represents the fraction of classrooms of type 3. -/
def fraction_type3 : ℚ := 1 / 5

/-- Represents the maximum student capacity allowed by regulations. -/
def max_capacity : ℕ := 1800

/-- Theorem stating that the total number of desks in the school is 1776. -/
theorem total_desks_is_1776 : 
  (↑total_classrooms * fraction_type1).floor * desks_type1 +
  (↑total_classrooms * fraction_type2).floor * desks_type2 +
  (↑total_classrooms * fraction_type3).floor * desks_type3 +
  (total_classrooms - 
    (↑total_classrooms * fraction_type1).floor - 
    (↑total_classrooms * fraction_type2).floor - 
    (↑total_classrooms * fraction_type3).floor) * desks_type4 = 1776 :=
by sorry

/-- Theorem stating that the total number of desks does not exceed the maximum capacity. -/
theorem total_desks_within_capacity : 
  (↑total_classrooms * fraction_type1).floor * desks_type1 +
  (↑total_classrooms * fraction_type2).floor * desks_type2 +
  (↑total_classrooms * fraction_type3).floor * desks_type3 +
  (total_classrooms - 
    (↑total_classrooms * fraction_type1).floor - 
    (↑total_classrooms * fraction_type2).floor - 
    (↑total_classrooms * fraction_type3).floor) * desks_type4 ≤ max_capacity :=
by sorry

end total_desks_is_1776_total_desks_within_capacity_l1299_129948


namespace cube_face_sum_l1299_129901

theorem cube_face_sum (a b c d e f : ℕ+) :
  (a * b * c + a * e * c + a * b * f + a * e * f +
   d * b * c + d * e * c + d * b * f + d * e * f) = 1491 →
  (a + b + c + d + e + f : ℕ) = 41 := by
sorry

end cube_face_sum_l1299_129901


namespace file_app_difference_l1299_129915

/-- Given initial and final counts of apps and files on a phone, 
    prove the difference between final files and apps --/
theorem file_app_difference 
  (initial_apps : ℕ) 
  (initial_files : ℕ) 
  (final_apps : ℕ) 
  (final_files : ℕ) 
  (h1 : initial_apps = 11) 
  (h2 : initial_files = 3) 
  (h3 : final_apps = 2) 
  (h4 : final_files = 24) : 
  final_files - final_apps = 22 := by
  sorry

end file_app_difference_l1299_129915


namespace calculate_rates_l1299_129940

/-- Represents the rates and quantities in the problem -/
structure Rates where
  b : ℕ  -- number of bananas Charles cooked
  d : ℕ  -- number of dishes Sandrine washed
  r1 : ℚ  -- rate at which Charles picks pears (pears per hour)
  r2 : ℚ  -- rate at which Charles cooks bananas (bananas per hour)
  r3 : ℚ  -- rate at which Sandrine washes dishes (dishes per hour)

/-- The main theorem representing the problem -/
theorem calculate_rates (rates : Rates) : 
  rates.d = rates.b + 10 ∧ 
  rates.b = 3 * 50 ∧ 
  rates.r1 = 50 / 4 ∧ 
  rates.r2 = rates.b / 2 ∧ 
  rates.r3 = rates.d / 5 → 
  rates.r1 = 12.5 ∧ rates.r2 = 75 ∧ rates.r3 = 32 := by
  sorry

end calculate_rates_l1299_129940


namespace ferris_wheel_seats_l1299_129941

/-- The number of people each seat can hold -/
def seat_capacity : ℕ := 6

/-- The total number of people the Ferris wheel can hold -/
def total_capacity : ℕ := 84

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := total_capacity / seat_capacity

theorem ferris_wheel_seats : num_seats = 14 := by
  sorry

end ferris_wheel_seats_l1299_129941


namespace glass_bowl_selling_price_l1299_129904

theorem glass_bowl_selling_price 
  (total_bowls : ℕ) 
  (cost_per_bowl : ℚ) 
  (bowls_sold : ℕ) 
  (percentage_gain : ℚ) 
  (h1 : total_bowls = 118) 
  (h2 : cost_per_bowl = 12) 
  (h3 : bowls_sold = 102) 
  (h4 : percentage_gain = 8050847457627118 / 100000000000000000) : 
  ∃ (selling_price : ℚ), selling_price = 15 ∧ 
  (total_bowls * cost_per_bowl * (1 + percentage_gain) / bowls_sold).floor = selling_price := by
  sorry

#check glass_bowl_selling_price

end glass_bowl_selling_price_l1299_129904


namespace ratio_bounds_l1299_129962

theorem ratio_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 5 - 3*a ≤ b) (h2 : b ≤ 4 - a) (h3 : Real.log b ≥ a) :
  e ≤ b/a ∧ b/a ≤ 7 := by
sorry

end ratio_bounds_l1299_129962


namespace samuel_journey_length_l1299_129978

/-- Represents a journey divided into three parts -/
structure Journey where
  first_part : ℚ  -- Fraction of the total journey
  middle_part : ℚ  -- Length in miles
  last_part : ℚ  -- Fraction of the total journey

/-- Calculates the total length of a journey -/
def journey_length (j : Journey) : ℚ :=
  j.middle_part / (1 - j.first_part - j.last_part)

theorem samuel_journey_length :
  let j : Journey := {
    first_part := 1/4,
    middle_part := 30,
    last_part := 1/6
  }
  journey_length j = 360/7 := by
  sorry

end samuel_journey_length_l1299_129978


namespace roots_of_equation_l1299_129966

def equation (x : ℝ) : ℝ := x * (2*x - 5)^2 * (x + 3) * (7 - x)

theorem roots_of_equation :
  {x : ℝ | equation x = 0} = {0, 2.5, -3, 7} := by sorry

end roots_of_equation_l1299_129966


namespace part1_part2_l1299_129926

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 12 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 3*a + 2}

-- Part 1
theorem part1 : A ∩ (Set.univ \ (B 1)) = {x | -2 ≤ x ∧ x ≤ 0 ∨ 5 ≤ x ∧ x ≤ 6} := by sorry

-- Part 2
theorem part2 : ∀ a : ℝ, A ∩ (B a) = B a ↔ a ∈ Set.Iic (-3/2) ∪ Set.Icc (-1) (4/3) := by sorry

end part1_part2_l1299_129926


namespace subcommittee_formation_count_l1299_129997

def senate_committee_size : ℕ := 18
def num_republicans : ℕ := 10
def num_democrats : ℕ := 8
def subcommittee_size : ℕ := 5
def min_republicans : ℕ := 2

theorem subcommittee_formation_count :
  (Finset.range (subcommittee_size - min_republicans + 1)).sum (λ k =>
    Nat.choose num_republicans (min_republicans + k) *
    Nat.choose num_democrats (subcommittee_size - (min_republicans + k))
  ) = 7812 := by
  sorry

end subcommittee_formation_count_l1299_129997


namespace correct_calculation_l1299_129900

-- Define variables
variable (x y : ℝ)

-- Theorem statement
theorem correct_calculation : -4 * x * y + 3 * x * y = -x * y := by
  sorry

end correct_calculation_l1299_129900


namespace sin_squared_simplification_l1299_129947

theorem sin_squared_simplification (x y : ℝ) : 
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end sin_squared_simplification_l1299_129947


namespace abc_inequality_l1299_129932

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : Real.sqrt a ^ 3 + Real.sqrt b ^ 3 + Real.sqrt c ^ 3 = 1) :
  a * b * c ≤ 1 / 9 ∧
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end abc_inequality_l1299_129932


namespace unique_swap_pair_l1299_129903

/-- A quadratic polynomial function -/
def QuadraticPolynomial (α : Type) [Ring α] := α → α

theorem unique_swap_pair
  (f : QuadraticPolynomial ℝ)
  (a b : ℝ)
  (h_distinct : a ≠ b)
  (h_swap : f a = b ∧ f b = a) :
  ¬∃ c d, c ≠ d ∧ (c, d) ≠ (a, b) ∧ f c = d ∧ f d = c :=
sorry

end unique_swap_pair_l1299_129903


namespace first_day_visitors_l1299_129949

/-- Given the initial stock and restock amount, calculate the number of people who showed up on the first day -/
theorem first_day_visitors (initial_stock : ℕ) (first_restock : ℕ) (cans_per_person : ℕ) : 
  initial_stock = 2000 →
  first_restock = 1500 →
  cans_per_person = 1 →
  (initial_stock - first_restock) / cans_per_person = 500 := by
  sorry

#check first_day_visitors

end first_day_visitors_l1299_129949


namespace problem_statement_l1299_129959

theorem problem_statement (x y : ℝ) : 
  x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3) →
  y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3) →
  x^4 + y^4 + (x + y)^4 = 1152 := by
sorry

end problem_statement_l1299_129959


namespace intersection_when_a_is_4_subset_condition_l1299_129927

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a - 1}
def B : Set ℝ := {x | x ≤ 3 ∨ x > 5}

-- Theorem 1: When a = 4, A ∩ B = {6, 7}
theorem intersection_when_a_is_4 : A 4 ∩ B = {6, 7} := by sorry

-- Theorem 2: A ⊆ B if and only if a < 2 or a > 4
theorem subset_condition (a : ℝ) : A a ⊆ B ↔ a < 2 ∨ a > 4 := by sorry

end intersection_when_a_is_4_subset_condition_l1299_129927


namespace sophie_bought_four_boxes_l1299_129930

/-- The number of boxes of donuts Sophie bought -/
def boxes_bought : ℕ := sorry

/-- The number of donuts in each box -/
def donuts_per_box : ℕ := 12

/-- The number of boxes Sophie gave to her mom -/
def boxes_to_mom : ℕ := 1

/-- The number of donuts Sophie gave to her sister -/
def donuts_to_sister : ℕ := 6

/-- The number of donuts Sophie had left for herself -/
def donuts_left : ℕ := 30

theorem sophie_bought_four_boxes : boxes_bought = 4 := by sorry

end sophie_bought_four_boxes_l1299_129930


namespace marble_color_convergence_l1299_129936

/-- Represents the number of marbles of each color -/
structure MarbleState :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- The total number of marbles -/
def totalMarbles : ℕ := 2015

/-- Possible operations on the marble state -/
inductive MarbleOperation
  | RedGreenToBlue
  | RedBlueToGreen
  | GreenBlueToRed

/-- Apply a marble operation to a state -/
def applyOperation (state : MarbleState) (op : MarbleOperation) : MarbleState :=
  match op with
  | MarbleOperation.RedGreenToBlue => 
      { red := state.red - 1, green := state.green - 1, blue := state.blue + 2 }
  | MarbleOperation.RedBlueToGreen => 
      { red := state.red - 1, green := state.green + 2, blue := state.blue - 1 }
  | MarbleOperation.GreenBlueToRed => 
      { red := state.red + 2, green := state.green - 1, blue := state.blue - 1 }

/-- Check if all marbles are the same color -/
def allSameColor (state : MarbleState) : Prop :=
  (state.red = totalMarbles ∧ state.green = 0 ∧ state.blue = 0) ∨
  (state.red = 0 ∧ state.green = totalMarbles ∧ state.blue = 0) ∨
  (state.red = 0 ∧ state.green = 0 ∧ state.blue = totalMarbles)

/-- The main theorem to prove -/
theorem marble_color_convergence 
  (initial : MarbleState) 
  (h_total : initial.red + initial.green + initial.blue = totalMarbles) :
  ∃ (operations : List MarbleOperation), 
    allSameColor (operations.foldl applyOperation initial) :=
sorry

end marble_color_convergence_l1299_129936


namespace total_basketball_cost_l1299_129968

/-- Represents a basketball team -/
structure Team where
  players : Nat
  basketballs_per_player : Nat
  price_per_basketball : Nat

/-- Calculates the total cost of basketballs for a team -/
def team_cost (t : Team) : Nat :=
  t.players * t.basketballs_per_player * t.price_per_basketball

/-- The Spurs basketball team -/
def spurs : Team :=
  { players := 22
    basketballs_per_player := 11
    price_per_basketball := 15 }

/-- The Dynamos basketball team -/
def dynamos : Team :=
  { players := 18
    basketballs_per_player := 9
    price_per_basketball := 20 }

/-- The Lions basketball team -/
def lions : Team :=
  { players := 26
    basketballs_per_player := 7
    price_per_basketball := 12 }

/-- Theorem stating the total cost of basketballs for all three teams -/
theorem total_basketball_cost :
  team_cost spurs + team_cost dynamos + team_cost lions = 9054 := by
  sorry

end total_basketball_cost_l1299_129968


namespace lunch_calories_l1299_129971

/-- The total calories for a kid's lunch -/
def total_calories (burger_calories : ℕ) (carrot_stick_calories : ℕ) (cookie_calories : ℕ) : ℕ :=
  burger_calories + 5 * carrot_stick_calories + 5 * cookie_calories

/-- Theorem stating that the total calories for each kid's lunch is 750 -/
theorem lunch_calories :
  total_calories 400 20 50 = 750 := by
  sorry

end lunch_calories_l1299_129971


namespace parallel_vectors_m_value_l1299_129986

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (m, 2)
  parallel a b → m = 4 := by
sorry

end parallel_vectors_m_value_l1299_129986


namespace negation_equivalence_l1299_129945

theorem negation_equivalence (a b : ℝ) :
  ¬(a ≤ 2 ∧ b ≤ 2) ↔ (a > 2 ∨ b > 2) := by
  sorry

end negation_equivalence_l1299_129945


namespace polynomial_characterization_l1299_129999

/-- A polynomial satisfying the given condition -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), a*b + b*c + c*a = 0 →
    P (a-b) + P (b-c) + P (c-a) = 2 * P (a+b+c)

/-- The form of the polynomial that satisfies the condition -/
def PolynomialForm (P : ℝ → ℝ) : Prop :=
  ∃ (r s : ℝ), ∀ x, P x = r * x^4 + s * x^2

theorem polynomial_characterization :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P ↔ PolynomialForm P :=
sorry

end polynomial_characterization_l1299_129999


namespace rectangular_prism_sum_l1299_129960

theorem rectangular_prism_sum (a b c : ℕ+) : 
  a * b * c = 21 → a ≠ b → b ≠ c → a ≠ c → a + b + c = 11 := by
sorry

end rectangular_prism_sum_l1299_129960


namespace vector_magnitude_problem_l1299_129951

/-- Given two vectors a and b in a real inner product space such that 
    |a| = |b| = |a - 2b| = 1, prove that |a + 2b| = 3. -/
theorem vector_magnitude_problem (a b : EuclideanSpace ℝ (Fin n)) 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a - 2 • b‖ = 1) : 
  ‖a + 2 • b‖ = 3 := by
  sorry

end vector_magnitude_problem_l1299_129951


namespace quadratic_shift_and_roots_l1299_129935

/-- A quadratic function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_shift_and_roots (a b c : ℝ) (h : a > 0) :
  (∀ k > 0, ∀ x, quadratic a b (c - k) x < quadratic a b c x) ∧
  (∀ x, quadratic a b c x ≠ 0 →
    ∃ k > 0, ∃ x, quadratic a b (c - k) x = 0) := by
  sorry

end quadratic_shift_and_roots_l1299_129935


namespace sqrt_four_fifths_simplification_l1299_129953

theorem sqrt_four_fifths_simplification :
  Real.sqrt (4 / 5) = (2 * Real.sqrt 5) / 5 := by sorry

end sqrt_four_fifths_simplification_l1299_129953


namespace quadruple_solution_l1299_129919

-- Define the condition function
def condition (a b c d : ℝ) : Prop :=
  a + b * c * d = b + c * d * a ∧
  a + b * c * d = c + d * a * b ∧
  a + b * c * d = d + a * b * c

-- Define the solution set
def solution_set (a b c d : ℝ) : Prop :=
  (a = b ∧ b = c ∧ c = d) ∨
  (a = b ∧ c = d ∧ c = 1 / a ∧ a ≠ 0) ∨
  (a = 1 ∧ b = 1 ∧ c = 1) ∨
  (a = -1 ∧ b = -1 ∧ c = -1)

-- Theorem statement
theorem quadruple_solution (a b c d : ℝ) :
  condition a b c d → solution_set a b c d :=
sorry

end quadruple_solution_l1299_129919


namespace water_tower_height_l1299_129907

/-- Given a bamboo pole and a water tower under the same lighting conditions,
    this theorem proves the height of the water tower based on the similar triangles concept. -/
theorem water_tower_height
  (bamboo_height : ℝ)
  (bamboo_shadow : ℝ)
  (tower_shadow : ℝ)
  (h_bamboo_height : bamboo_height = 2)
  (h_bamboo_shadow : bamboo_shadow = 1.5)
  (h_tower_shadow : tower_shadow = 24) :
  bamboo_height / bamboo_shadow * tower_shadow = 32 :=
by sorry

end water_tower_height_l1299_129907


namespace sum_is_three_or_seven_l1299_129913

theorem sum_is_three_or_seven (x y z : ℝ) 
  (eq1 : x + y / z = 2)
  (eq2 : y + z / x = 2)
  (eq3 : z + x / y = 2) :
  let S := x + y + z
  S = 3 ∨ S = 7 := by
sorry

end sum_is_three_or_seven_l1299_129913


namespace smallest_variance_l1299_129989

def minimumVariance (n : ℕ) (s : Finset ℝ) : Prop :=
  n ≥ 2 ∧
  s.card = n ∧
  (0 : ℝ) ∈ s ∧
  (1 : ℝ) ∈ s ∧
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 1) →
  ∀ ε > 0, ∃ (v : ℝ), v ≥ 1 / (2 * n) ∧
    v = (s.sum (λ x => (x - s.sum (λ y => y) / n) ^ 2)) / n

theorem smallest_variance (n : ℕ) (s : Finset ℝ) (h : minimumVariance n s) :
  ∃ (v : ℝ), v = 1 / (2 * n) ∧
    v = (s.sum (λ x => (x - s.sum (λ y => y) / n) ^ 2)) / n :=
sorry

end smallest_variance_l1299_129989


namespace area_equality_l1299_129956

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the shapes
def is_convex_quadrilateral (A C D F : ℝ × ℝ) : Prop := sorry
def is_equilateral_triangle (A B E : ℝ × ℝ) : Prop := sorry
def is_square (A C D F : ℝ × ℝ) : Prop := sorry
def is_rectangle (A C D F : ℝ × ℝ) : Prop := sorry

-- Define the point on side condition
def point_on_side (P Q R : ℝ × ℝ) : Prop := sorry

-- Define the area calculation function
def area_triangle (P Q R : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem area_equality 
  (h_quad : is_convex_quadrilateral A C D F)
  (h_tri : is_equilateral_triangle A B E)
  (h_common : A = A)  -- Common vertex
  (h_B_on_CF : point_on_side B C F)
  (h_E_on_FD : point_on_side E F D)
  (h_shape : is_square A C D F ∨ is_rectangle A C D F) :
  area_triangle A D E + area_triangle A B C = area_triangle B E F := by
  sorry

end area_equality_l1299_129956


namespace f_max_and_inequality_l1299_129987

def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

theorem f_max_and_inequality :
  (∃ (a : ℝ), ∀ x, f x ≤ a ∧ ∃ y, f y = a) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → 1/m + 1/(2*n) = 2 → 2*m + n > 2) :=
sorry

end f_max_and_inequality_l1299_129987


namespace root_sum_reciprocals_l1299_129988

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 - 4*p^3 + 7*p^2 - 3*p + 2 = 0) →
  (q^4 - 4*q^3 + 7*q^2 - 3*q + 2 = 0) →
  (r^4 - 4*r^3 + 7*r^2 - 3*r + 2 = 0) →
  (s^4 - 4*s^3 + 7*s^2 - 3*s + 2 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 7/2 := by
sorry

end root_sum_reciprocals_l1299_129988


namespace peanuts_in_box_l1299_129992

/-- The number of peanuts in a box after adding more -/
theorem peanuts_in_box (initial : ℕ) (added : ℕ) : 
  initial = 4 → added = 12 → initial + added = 16 := by
  sorry

end peanuts_in_box_l1299_129992


namespace probability_white_ball_specific_l1299_129967

/-- The probability of drawing a white ball from a bag -/
def probability_white_ball (black white red : ℕ) : ℚ :=
  white / (black + white + red)

/-- Theorem: The probability of drawing a white ball from a bag with 3 black, 2 white, and 1 red ball is 1/3 -/
theorem probability_white_ball_specific : probability_white_ball 3 2 1 = 1/3 := by
  sorry

end probability_white_ball_specific_l1299_129967


namespace sandy_grew_eight_carrots_l1299_129976

/-- The number of carrots Sandy grew -/
def sandys_carrots : ℕ := sorry

/-- The number of carrots Mary grew -/
def marys_carrots : ℕ := 6

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := 14

/-- Theorem stating that Sandy grew 8 carrots -/
theorem sandy_grew_eight_carrots : sandys_carrots = 8 := by
  sorry

end sandy_grew_eight_carrots_l1299_129976


namespace cassies_nail_trimming_l1299_129991

/-- The number of nails/claws Cassie needs to cut -/
def total_nails_to_cut (num_dogs : ℕ) (num_parrots : ℕ) (nails_per_dog_foot : ℕ) 
  (feet_per_dog : ℕ) (claws_per_parrot_leg : ℕ) (legs_per_parrot : ℕ) 
  (extra_claw : ℕ) : ℕ :=
  (num_dogs * nails_per_dog_foot * feet_per_dog) + 
  (num_parrots * claws_per_parrot_leg * legs_per_parrot) + 
  extra_claw

/-- Theorem stating the total number of nails/claws Cassie needs to cut -/
theorem cassies_nail_trimming :
  total_nails_to_cut 4 8 4 4 3 2 1 = 113 := by
  sorry

end cassies_nail_trimming_l1299_129991


namespace calculation_proof_l1299_129939

theorem calculation_proof : 0.25^2005 * 4^2006 - 8^100 * 0.5^300 = 3 := by
  sorry

end calculation_proof_l1299_129939


namespace inverse_81_mod_101_l1299_129931

theorem inverse_81_mod_101 (h : (9 : ZMod 101)⁻¹ = 90) : (81 : ZMod 101)⁻¹ = 20 := by
  sorry

end inverse_81_mod_101_l1299_129931


namespace inessa_is_cleverest_l1299_129910

-- Define the foxes
inductive Fox : Type
  | Alisa : Fox
  | Larisa : Fox
  | Inessa : Fox

-- Define a relation for "is cleverer than"
def is_cleverer_than : Fox → Fox → Prop := sorry

-- Define a property for being the cleverest
def is_cleverest : Fox → Prop := sorry

-- Define a function to check if a fox is telling the truth
def tells_truth : Fox → Prop := sorry

-- State the theorem
theorem inessa_is_cleverest :
  -- The cleverest fox lies, others tell the truth
  (∀ f : Fox, is_cleverest f ↔ ¬(tells_truth f)) →
  -- Larisa's statement
  (tells_truth Fox.Larisa ↔ ¬(is_cleverest Fox.Alisa)) →
  -- Alisa's statement
  (tells_truth Fox.Alisa ↔ is_cleverer_than Fox.Alisa Fox.Larisa) →
  -- Inessa's statement
  (tells_truth Fox.Inessa ↔ is_cleverer_than Fox.Alisa Fox.Inessa) →
  -- There is exactly one cleverest fox
  (∃! f : Fox, is_cleverest f) →
  -- Conclusion: Inessa is the cleverest
  is_cleverest Fox.Inessa :=
by
  sorry

end inessa_is_cleverest_l1299_129910


namespace triangle_shape_l1299_129917

theorem triangle_shape (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hSum : A + B + C = π) (hSine : 2 * Real.sin B * Real.cos C = Real.sin A) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  b = c :=
sorry

end triangle_shape_l1299_129917


namespace perimeter_of_parallelogram_l1299_129964

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB AC BC : ℝ)
  (angleBAC : ℝ)

-- Define the parallelogram ADEF
structure Parallelogram :=
  (A D E F : ℝ × ℝ)

-- Define the problem statement
theorem perimeter_of_parallelogram (t : Triangle) (p : Parallelogram) : 
  t.AB = 20 →
  t.AC = 24 →
  t.BC = 18 →
  t.angleBAC = 60 * π / 180 →
  (p.D.1 - t.A.1) / (t.B.1 - t.A.1) = (p.D.2 - t.A.2) / (t.B.2 - t.A.2) →
  (p.E.1 - t.B.1) / (t.C.1 - t.B.1) = (p.E.2 - t.B.2) / (t.C.2 - t.B.2) →
  (p.F.1 - t.A.1) / (t.C.1 - t.A.1) = (p.F.2 - t.A.2) / (t.C.2 - t.A.2) →
  (p.E.1 - p.D.1) / (t.C.1 - t.A.1) = (p.E.2 - p.D.2) / (t.C.2 - t.A.2) →
  (p.F.1 - p.E.1) / (t.B.1 - t.A.1) = (p.F.2 - p.E.2) / (t.B.2 - t.A.2) →
  Real.sqrt ((p.A.1 - p.D.1)^2 + (p.A.2 - p.D.2)^2) +
  Real.sqrt ((p.D.1 - p.E.1)^2 + (p.D.2 - p.E.2)^2) +
  Real.sqrt ((p.E.1 - p.F.1)^2 + (p.E.2 - p.F.2)^2) +
  Real.sqrt ((p.F.1 - p.A.1)^2 + (p.F.2 - p.A.2)^2) = 44 :=
by sorry


end perimeter_of_parallelogram_l1299_129964


namespace expression_evaluation_l1299_129924

theorem expression_evaluation : 
  |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.cos (π / 4) - Real.sqrt 2 * Real.sqrt 6 = -Real.sqrt 3 := by
sorry

end expression_evaluation_l1299_129924


namespace abc_sum_sqrt_l1299_129983

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17) 
  (h2 : c + a = 18) 
  (h3 : a + b = 19) : 
  Real.sqrt (a * b * c * (a + b + c)) = 54 * Real.sqrt 5 := by
  sorry

end abc_sum_sqrt_l1299_129983


namespace estimate_fish_population_l1299_129929

/-- Estimates the number of fish in a lake using the mark-recapture method. -/
theorem estimate_fish_population (n m k : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) (h4 : k ≤ m) (h5 : k ≤ n) :
  (n * m : ℚ) / k = (m : ℚ) / (k : ℚ) * n :=
by sorry

end estimate_fish_population_l1299_129929


namespace shaded_triangle_area_l1299_129969

/-- Given a rectangle with sides 12 units long and a square with sides 4 units long
    placed in one corner of the rectangle, the area of the triangle formed by
    the diagonal of the rectangle and two sides of the rectangle is 54 square units. -/
theorem shaded_triangle_area (rectangle_side : ℝ) (square_side : ℝ) : 
  rectangle_side = 12 →
  square_side = 4 →
  let triangle_base := rectangle_side - (rectangle_side - square_side) * (square_side / rectangle_side)
  let triangle_height := rectangle_side
  (1/2) * triangle_base * triangle_height = 54 := by
sorry

end shaded_triangle_area_l1299_129969


namespace carlsons_land_cost_l1299_129996

/-- Proves that the cost of Carlson's first land is $8000 --/
theorem carlsons_land_cost (initial_area : ℝ) (final_area : ℝ) (new_land_cost_per_sqm : ℝ) (additional_cost : ℝ) : ℝ :=
  by
  -- Define the given conditions
  have h1 : initial_area = 300 := by sorry
  have h2 : final_area = 900 := by sorry
  have h3 : new_land_cost_per_sqm = 20 := by sorry
  have h4 : additional_cost = 4000 := by sorry

  -- Define the first land cost
  let first_land_cost : ℝ := 8000

  -- Prove that the first land cost is $8000
  sorry

end carlsons_land_cost_l1299_129996


namespace license_plate_count_l1299_129946

/-- The number of consonants in the English alphabet (excluding Y) -/
def num_consonants : ℕ := 20

/-- The number of vowels in the English alphabet (including Y) -/
def num_vowels : ℕ := 6

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of special symbols allowed (!, #, $) -/
def num_special_symbols : ℕ := 3

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits * num_special_symbols

theorem license_plate_count : total_license_plates = 72000 := by
  sorry

end license_plate_count_l1299_129946


namespace parabola_point_x_coordinate_l1299_129965

/-- A point on a parabola with a specific distance to the focus -/
structure ParabolaPoint where
  y : ℝ
  x : ℝ
  parabola_eq : x = 4 * y^2
  focus_distance : Real.sqrt ((x - 1/4)^2 + y^2) = 1/2

/-- The x-coordinate of a point on a parabola with a specific distance to the focus -/
theorem parabola_point_x_coordinate (M : ParabolaPoint) : M.x = 7/16 := by
  sorry

end parabola_point_x_coordinate_l1299_129965


namespace farm_animals_l1299_129973

theorem farm_animals (total_legs : ℕ) (total_animals : ℕ) (chicken_legs : ℕ) (sheep_legs : ℕ) :
  total_legs = 60 →
  total_animals = 20 →
  chicken_legs = 2 →
  sheep_legs = 4 →
  ∃ (num_chickens num_sheep : ℕ),
    num_chickens + num_sheep = total_animals ∧
    num_chickens * chicken_legs + num_sheep * sheep_legs = total_legs ∧
    num_sheep = 10 := by
  sorry

end farm_animals_l1299_129973


namespace number_of_students_l1299_129914

/-- Given a class where:
    1. The initial average marks of students is 100.
    2. A student's mark is wrongly noted as 50 instead of 10.
    3. The correct average marks is 96.
    Prove that the number of students in the class is 10. -/
theorem number_of_students (n : ℕ) 
    (h1 : (100 * n) / n = 100)  -- Initial average is 100
    (h2 : (100 * n - 40) / n = 96)  -- Correct average is 96
    : n = 10 := by
  sorry


end number_of_students_l1299_129914


namespace inverse_square_relation_l1299_129952

/-- Given that x varies inversely as the square of y, and y = 2 when x = 1,
    prove that x = 1/9 when y = 6 -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) (h1 : x = k / y^2) 
    (h2 : 1 = k / 2^2) : 
    (y = 6) → (x = 1/9) := by
  sorry

end inverse_square_relation_l1299_129952


namespace complex_fraction_equality_l1299_129944

theorem complex_fraction_equality : 2 * (1 + 1 / (1 - 1 / (2 + 2))) = 10 / 3 := by
  sorry

end complex_fraction_equality_l1299_129944


namespace median_and_mode_of_scores_l1299_129905

def scores : List Nat := [74, 84, 84, 84, 87, 92, 92]

def median (l : List Nat) : Nat := sorry

def mode (l : List Nat) : Nat := sorry

theorem median_and_mode_of_scores :
  median scores = 84 ∧ mode scores = 84 := by sorry

end median_and_mode_of_scores_l1299_129905


namespace independence_day_banana_distribution_l1299_129937

theorem independence_day_banana_distribution :
  ∀ (total_children : ℕ) (total_bananas : ℕ),
    (2 * total_children = total_bananas) →
    (4 * (total_children - 390) = total_bananas) →
    total_children = 780 := by
  sorry

end independence_day_banana_distribution_l1299_129937


namespace solve_for_q_l1299_129993

theorem solve_for_q (n m q : ℚ) 
  (h1 : 7/9 = n/108)
  (h2 : 7/9 = (m+n)/126)
  (h3 : 7/9 = (q-m)/162) : 
  q = 140 := by
sorry

end solve_for_q_l1299_129993


namespace ages_problem_l1299_129958

/-- The present ages of individuals A, B, C, and D satisfy the given conditions. -/
theorem ages_problem (A B C D : ℕ) : 
  (C + 10 = 3 * (A + 10)) →  -- In 10 years, C will be 3 times as old as A
  (A = 2 * (B - 10)) →       -- A will be twice as old as B was 10 years ago
  (A = B + 12) →             -- A is now 12 years older than B
  (B = D + 5) →              -- B is 5 years older than D
  (D = C / 2) →              -- D is half the age of C
  (A = 88 ∧ B = 76 ∧ C = 142 ∧ D = 71) :=
by sorry

end ages_problem_l1299_129958


namespace percentage_failed_hindi_l1299_129977

theorem percentage_failed_hindi (failed_english : Real) (failed_both : Real) (passed_both : Real)
  (h1 : failed_english = 48)
  (h2 : failed_both = 27)
  (h3 : passed_both = 54) :
  failed_english + (100 - passed_both) - failed_both = 25 := by
  sorry

end percentage_failed_hindi_l1299_129977


namespace m_intersect_n_equals_m_l1299_129994

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}

-- Theorem statement
theorem m_intersect_n_equals_m : M ∩ N = M := by sorry

end m_intersect_n_equals_m_l1299_129994


namespace square_with_seven_in_tens_place_l1299_129943

theorem square_with_seven_in_tens_place (a : ℕ) (b : Fin 10) :
  ∃ k : ℕ, ((10 * a + b) ^ 2) % 100 = 70 + k ∧ k < 10 →
  (b = 4 ∨ b = 6) := by
sorry

end square_with_seven_in_tens_place_l1299_129943


namespace magic_8_ball_probability_l1299_129906

/-- The probability of getting a positive answer from the Magic 8 Ball -/
def p : ℚ := 1/3

/-- The number of questions asked -/
def n : ℕ := 7

/-- The number of positive answers we're interested in -/
def k : ℕ := 3

/-- The probability of getting exactly k positive answers out of n questions -/
def probability_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem magic_8_ball_probability :
  probability_k_successes n k p = 560/2187 := by
  sorry

end magic_8_ball_probability_l1299_129906


namespace expand_product_l1299_129975

theorem expand_product (x : ℝ) : (x + 5) * (x + 9) = x^2 + 14*x + 45 := by
  sorry

end expand_product_l1299_129975


namespace stock_sale_total_amount_l1299_129954

/-- Calculates the total amount including brokerage for a stock sale -/
theorem stock_sale_total_amount 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 104.25)
  (h2 : brokerage_rate = 1/4) : 
  ∃ (total_amount : ℝ), total_amount = 104.51 ∧ 
  total_amount = cash_realized + (brokerage_rate / 100) * cash_realized := by
  sorry

end stock_sale_total_amount_l1299_129954


namespace simplify_and_rationalize_l1299_129902

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 9) = Real.sqrt 15 / 9 := by
  sorry

end simplify_and_rationalize_l1299_129902


namespace seven_mile_taxi_cost_l1299_129957

/-- Calculates the total cost of a taxi ride given the fixed cost, cost per mile, and distance traveled. -/
def taxi_cost (fixed_cost : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  fixed_cost + cost_per_mile * distance

/-- Theorem stating that a 7-mile taxi ride with $2.00 fixed cost and $0.30 per mile costs $4.10 -/
theorem seven_mile_taxi_cost :
  taxi_cost 2.00 0.30 7 = 4.10 := by
  sorry

end seven_mile_taxi_cost_l1299_129957


namespace lion_path_angles_l1299_129911

theorem lion_path_angles (r : ℝ) (path_length : ℝ) (turn_angles : List ℝ) : 
  r = 10 →
  path_length = 30000 →
  path_length ≤ 2 * r + r * (turn_angles.sum) →
  turn_angles.sum ≥ 2998 := by
sorry

end lion_path_angles_l1299_129911


namespace curve_W_properties_l1299_129934

-- Define the curve W
def W (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y + 1)^2) * Real.sqrt (x^2 + (y - 1)^2) = 3

-- Theorem stating the properties of curve W
theorem curve_W_properties :
  -- 1. x = 0 is an axis of symmetry
  (∀ y : ℝ, W 0 y ↔ W 0 (-y)) ∧
  -- 2. (0, 2) and (0, -2) are points on W
  W 0 2 ∧ W 0 (-2) ∧
  -- 3. The range of y-coordinates is [-2, 2]
  (∀ x y : ℝ, W x y → -2 ≤ y ∧ y ≤ 2) ∧
  (∀ y : ℝ, -2 ≤ y ∧ y ≤ 2 → ∃ x : ℝ, W x y) :=
by sorry


end curve_W_properties_l1299_129934


namespace average_mark_proof_l1299_129980

/-- Given an examination with 50 candidates and a total of 2000 marks,
    prove that the average mark obtained by each candidate is 40. -/
theorem average_mark_proof (candidates : ℕ) (total_marks : ℕ) :
  candidates = 50 →
  total_marks = 2000 →
  (total_marks : ℚ) / (candidates : ℚ) = 40 := by
  sorry

end average_mark_proof_l1299_129980


namespace tangent_lines_imply_a_range_l1299_129908

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.sqrt x

def has_two_tangent_lines (f g : ℝ → ℝ) : Prop :=
  ∃ (l₁ l₂ : ℝ → ℝ), l₁ ≠ l₂ ∧
    (∃ (x₁ : ℝ), l₁ x₁ = f x₁ ∧ (∀ x, l₁ x ≤ f x)) ∧
    (∃ (x₂ : ℝ), l₂ x₂ = f x₂ ∧ (∀ x, l₂ x ≤ f x)) ∧
    (∃ (y₁ : ℝ), l₁ y₁ = g y₁ ∧ (∀ y, l₁ y ≤ g y)) ∧
    (∃ (y₂ : ℝ), l₂ y₂ = g y₂ ∧ (∀ y, l₂ y ≤ g y))

theorem tangent_lines_imply_a_range (a : ℝ) :
  has_two_tangent_lines f (g a) → 0 < a ∧ a < 2 :=
by sorry

end tangent_lines_imply_a_range_l1299_129908


namespace two_is_only_22_2_sum_of_squares_l1299_129995

/-- A number is of the form 22...2 if it consists of one or more 2's. -/
def is_form_22_2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 + 10 * (2 * (10^k - 1) / 9)

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

/-- The main theorem stating that 2 is the only number of the form 22...2
    that can be expressed as the sum of two perfect squares. -/
theorem two_is_only_22_2_sum_of_squares :
  ∀ n : ℕ, is_form_22_2 n ∧ (∃ a b : ℕ, n = a^2 + b^2) ↔ n = 2 := by
  sorry


end two_is_only_22_2_sum_of_squares_l1299_129995


namespace isosceles_triangles_in_square_l1299_129938

theorem isosceles_triangles_in_square (s : ℝ) (h : s = 2) :
  let square_area := s^2
  let triangle_area := square_area / 4
  let half_base := s / 2
  let height := triangle_area / half_base
  let side_length := Real.sqrt (half_base^2 + height^2)
  side_length = Real.sqrt 2 := by sorry

end isosceles_triangles_in_square_l1299_129938


namespace sum_of_squares_and_products_l1299_129974

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 75)
  (h5 : x*y + y*z + z*x = 28) :
  x + y + z = Real.sqrt 131 :=
by sorry

end sum_of_squares_and_products_l1299_129974


namespace lines_perpendicular_to_plane_are_parallel_l1299_129950

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end lines_perpendicular_to_plane_are_parallel_l1299_129950


namespace profit_distribution_l1299_129985

theorem profit_distribution (total_profit : ℕ) (a_prop b_prop c_prop : ℕ) 
  (h_total : total_profit = 20000)
  (h_prop : a_prop = 2 ∧ b_prop = 3 ∧ c_prop = 5) :
  let total_parts := a_prop + b_prop + c_prop
  let part_value := total_profit / total_parts
  let b_share := b_prop * part_value
  let c_share := c_prop * part_value
  c_share - b_share = 4000 := by
sorry

end profit_distribution_l1299_129985
