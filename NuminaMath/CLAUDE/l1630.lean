import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l1630_163038

theorem system_solution :
  ∀ (x y z : ℝ),
    (x + 1) * y * z = 12 ∧
    (y + 1) * z * x = 4 ∧
    (z + 1) * x * y = 4 →
    ((x = 1/3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1630_163038


namespace NUMINAMATH_CALUDE_point_distance_to_line_l1630_163025

theorem point_distance_to_line (a : ℝ) (h1 : a > 0) : 
  (|a - 2 + 3| / Real.sqrt 2 = 1) → a = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_to_line_l1630_163025


namespace NUMINAMATH_CALUDE_class_one_is_correct_l1630_163010

/-- Represents the correct way to refer to a numbered class -/
inductive ClassReference
  | CardinalNumber (n : Nat)
  | OrdinalNumber (n : Nat)

/-- Checks if a class reference is correct -/
def is_correct_reference (ref : ClassReference) : Prop :=
  match ref with
  | ClassReference.CardinalNumber n => true
  | ClassReference.OrdinalNumber n => false

/-- The statement that "Class One" is the correct way to refer to the first class -/
theorem class_one_is_correct :
  is_correct_reference (ClassReference.CardinalNumber 1) = true :=
sorry


end NUMINAMATH_CALUDE_class_one_is_correct_l1630_163010


namespace NUMINAMATH_CALUDE_special_triangle_sum_squares_is_square_l1630_163099

/-- A triangle with integer side lengths where one altitude equals the sum of the other two -/
structure SpecialTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  altitude_sum : ℝ
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  altitude_relation : h₁ = h₂ + h₃
  area_equality : (a : ℝ) * h₁ = (b : ℝ) * h₂ ∧ (b : ℝ) * h₂ = (c : ℝ) * h₃

theorem special_triangle_sum_squares_is_square (t : SpecialTriangle) :
  ∃ n : ℤ, t.a^2 + t.b^2 + t.c^2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sum_squares_is_square_l1630_163099


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_435_l1630_163018

/-- The sum of the digits in the binary representation of 435 is 6 -/
theorem sum_of_binary_digits_435 : 
  (Nat.digits 2 435).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_435_l1630_163018


namespace NUMINAMATH_CALUDE_power_product_evaluation_l1630_163076

theorem power_product_evaluation :
  let a : ℕ := 3
  a^2 * a^5 = 2187 :=
by sorry

end NUMINAMATH_CALUDE_power_product_evaluation_l1630_163076


namespace NUMINAMATH_CALUDE_system_solutions_l1630_163087

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  (x + y - 2018 = (x - 2019) * y) ∧
  (x + z - 2014 = (x - 2019) * z) ∧
  (y + z + 2 = y * z)

-- State the theorem
theorem system_solutions :
  (∃ (x y z : ℝ), system x y z ∧ x = 2022 ∧ y = 2 ∧ z = 4) ∧
  (∃ (x y z : ℝ), system x y z ∧ x = 2017 ∧ y = 0 ∧ z = -2) ∧
  (∀ (x y z : ℝ), system x y z → (x = 2022 ∧ y = 2 ∧ z = 4) ∨ (x = 2017 ∧ y = 0 ∧ z = -2)) :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l1630_163087


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1630_163048

theorem sqrt_equation_solution (x : ℝ) :
  x ≥ 1 →
  (Real.sqrt (x + 3 - 4 * Real.sqrt (x - 1)) + Real.sqrt (x + 8 - 6 * Real.sqrt (x - 1)) = 1) ↔
  (5 ≤ x ∧ x ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1630_163048


namespace NUMINAMATH_CALUDE_sum_of_constants_l1630_163091

theorem sum_of_constants (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / x) →
  (2 = a + b / (-2)) →
  (7 = a + b / (-4)) →
  a + b = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l1630_163091


namespace NUMINAMATH_CALUDE_chairs_to_hall_l1630_163020

theorem chairs_to_hall (num_students : ℕ) (chairs_per_trip : ℕ) (num_trips : ℕ) :
  num_students = 5 →
  chairs_per_trip = 5 →
  num_trips = 10 →
  num_students * chairs_per_trip * num_trips = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_chairs_to_hall_l1630_163020


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1630_163080

theorem intersection_of_sets : 
  let M : Set ℤ := {-1, 1, 3, 5}
  let N : Set ℤ := {-3, 1, 5}
  M ∩ N = {1, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1630_163080


namespace NUMINAMATH_CALUDE_reciprocal_of_one_twentieth_l1630_163037

theorem reciprocal_of_one_twentieth (x : ℚ) : x = 1 / 20 → 1 / x = 20 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_one_twentieth_l1630_163037


namespace NUMINAMATH_CALUDE_max_profit_computer_sales_profit_per_computer_type_l1630_163035

/-- Profit function for computer sales -/
def profit_function (m : ℕ) : ℝ := -50 * m + 15000

/-- Constraint on the number of type B computers -/
def type_b_constraint (m : ℕ) : Prop := 100 - m ≤ 2 * m

/-- Theorem stating the maximum profit and optimal purchase strategy -/
theorem max_profit_computer_sales :
  ∃ (m : ℕ),
    m = 34 ∧
    type_b_constraint m ∧
    profit_function m = 13300 ∧
    ∀ (n : ℕ), type_b_constraint n → profit_function n ≤ profit_function m :=
by
  sorry

/-- Theorem verifying the profit for each computer type -/
theorem profit_per_computer_type :
  ∃ (a b : ℝ),
    a = 100 ∧
    b = 150 ∧
    10 * a + 20 * b = 4000 ∧
    20 * a + 10 * b = 3500 :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_computer_sales_profit_per_computer_type_l1630_163035


namespace NUMINAMATH_CALUDE_cube_surface_area_l1630_163069

/-- Given a cube with volume 27 cubic cm, its surface area is 54 square cm. -/
theorem cube_surface_area (cube : Set ℝ) (volume : ℝ) (surface_area : ℝ) : 
  volume = 27 →
  surface_area = 54 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1630_163069


namespace NUMINAMATH_CALUDE_cost_of_5_spoons_l1630_163040

-- Define the cost of a set of 7 spoons
def cost_7_spoons : ℝ := 21

-- Define the number of spoons in a set
def spoons_in_set : ℕ := 7

-- Define the number of spoons we want to buy
def spoons_to_buy : ℕ := 5

-- Theorem: The cost of 5 spoons is $15
theorem cost_of_5_spoons :
  (cost_7_spoons / spoons_in_set) * spoons_to_buy = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_5_spoons_l1630_163040


namespace NUMINAMATH_CALUDE_halfway_fraction_l1630_163095

theorem halfway_fraction (a b : ℚ) (ha : a = 1/6) (hb : b = 1/4) :
  (a + b) / 2 = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1630_163095


namespace NUMINAMATH_CALUDE_rectangle_area_l1630_163074

/-- Given a rectangle with perimeter 176 inches and length 8 inches more than its width,
    prove that its area is 1920 square inches. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 4 * w + 16 = 176) : w * (w + 8) = 1920 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1630_163074


namespace NUMINAMATH_CALUDE_inequality_proof_l1630_163093

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1630_163093


namespace NUMINAMATH_CALUDE_max_odd_integers_in_even_product_l1630_163088

theorem max_odd_integers_in_even_product (integers : Finset ℕ) :
  integers.card = 6 ∧
  (∀ n ∈ integers, n > 0) ∧
  Even (integers.prod id) →
  (integers.filter Odd).card ≤ 5 ∧
  ∃ (subset : Finset ℕ),
    subset ⊆ integers ∧
    subset.card = 5 ∧
    ∀ n ∈ subset, Odd n :=
by sorry

end NUMINAMATH_CALUDE_max_odd_integers_in_even_product_l1630_163088


namespace NUMINAMATH_CALUDE_french_fries_cost_is_ten_l1630_163071

/-- Represents the cost of a meal at Wendy's -/
structure WendysMeal where
  taco_salad : ℕ
  hamburgers : ℕ
  lemonade : ℕ
  friends : ℕ
  individual_payment : ℕ

/-- Calculates the total cost of french fries in a Wendy's meal -/
def french_fries_cost (meal : WendysMeal) : ℕ :=
  meal.friends * meal.individual_payment -
  (meal.taco_salad + 5 * meal.hamburgers + 5 * meal.lemonade)

/-- Theorem stating that the total cost of french fries is $10 -/
theorem french_fries_cost_is_ten (meal : WendysMeal)
  (h1 : meal.taco_salad = 10)
  (h2 : meal.hamburgers = 5)
  (h3 : meal.lemonade = 2)
  (h4 : meal.friends = 5)
  (h5 : meal.individual_payment = 11) :
  french_fries_cost meal = 10 := by
  sorry

#eval french_fries_cost { taco_salad := 10, hamburgers := 5, lemonade := 2, friends := 5, individual_payment := 11 }

end NUMINAMATH_CALUDE_french_fries_cost_is_ten_l1630_163071


namespace NUMINAMATH_CALUDE_expression_evaluation_l1630_163015

theorem expression_evaluation (b : ℕ) (h : b = 2) : (b^3 * b^4) - b^2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1630_163015


namespace NUMINAMATH_CALUDE_largest_two_digit_multiple_of_17_l1630_163073

theorem largest_two_digit_multiple_of_17 : ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, m ≤ 99 → m ≥ 10 → m % 17 = 0 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_multiple_of_17_l1630_163073


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1630_163046

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  (∀ n : ℕ, a n > 0) → a 4 * a 10 = 16 → a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1630_163046


namespace NUMINAMATH_CALUDE_equation_holds_iff_l1630_163097

theorem equation_holds_iff (k y : ℝ) : 
  (∀ x : ℝ, -x^2 - (k+10)*x - 8 = -(x - 2)*(x - 4) + (y - 3)*(y - 6)) ↔ 
  (k = -16 ∧ False) :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_iff_l1630_163097


namespace NUMINAMATH_CALUDE_skyscraper_window_installation_time_l1630_163030

/-- The time required to install the remaining windows in a skyscraper -/
theorem skyscraper_window_installation_time 
  (total_windows : ℕ) 
  (installed_windows : ℕ) 
  (installation_time_per_window : ℕ) 
  (h1 : total_windows = 200)
  (h2 : installed_windows = 65)
  (h3 : installation_time_per_window = 12) :
  (total_windows - installed_windows) * installation_time_per_window = 1620 := by
  sorry

end NUMINAMATH_CALUDE_skyscraper_window_installation_time_l1630_163030


namespace NUMINAMATH_CALUDE_museum_ticket_fraction_l1630_163081

theorem museum_ticket_fraction (total_money : ℚ) (sandwich_fraction : ℚ) (book_fraction : ℚ) (leftover : ℚ) : 
  total_money = 120 →
  sandwich_fraction = 1/5 →
  book_fraction = 1/2 →
  leftover = 16 →
  (total_money - (sandwich_fraction * total_money + book_fraction * total_money + leftover)) / total_money = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_fraction_l1630_163081


namespace NUMINAMATH_CALUDE_angle_double_complement_measure_l1630_163013

theorem angle_double_complement_measure : ∀ x : ℝ, 
  (x = 2 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_double_complement_measure_l1630_163013


namespace NUMINAMATH_CALUDE_peaches_left_l1630_163051

def total_peaches : ℕ := 250
def fresh_percentage : ℚ := 60 / 100
def small_peaches : ℕ := 15

theorem peaches_left : 
  (total_peaches * fresh_percentage).floor - small_peaches = 135 := by
  sorry

end NUMINAMATH_CALUDE_peaches_left_l1630_163051


namespace NUMINAMATH_CALUDE_eve_ran_distance_l1630_163047

/-- The distance Eve walked in miles -/
def distance_walked : ℝ := 0.6

/-- The additional distance Eve ran compared to what she walked, in miles -/
def additional_distance : ℝ := 0.1

/-- The total distance Eve ran in miles -/
def distance_ran : ℝ := distance_walked + additional_distance

theorem eve_ran_distance : distance_ran = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_eve_ran_distance_l1630_163047


namespace NUMINAMATH_CALUDE_equal_prob_conditions_l1630_163000

/-- Represents an urn with two compartments -/
structure Urn :=
  (v₁ f₁ v₂ f₂ : ℕ)

/-- Probability of drawing a red ball from a partitioned urn -/
def probPartitioned (u : Urn) : ℚ :=
  (u.v₁ * (u.v₂ + u.f₂) + u.v₂ * (u.v₁ + u.f₁)) / (2 * (u.v₁ + u.f₁) * (u.v₂ + u.f₂))

/-- Probability of drawing a red ball from a non-partitioned urn -/
def probNonPartitioned (u : Urn) : ℚ :=
  (u.v₁ + u.v₂) / ((u.v₁ + u.f₁) + (u.v₂ + u.f₂))

/-- Theorem stating the conditions for equal probabilities -/
theorem equal_prob_conditions (u : Urn) :
  probPartitioned u = probNonPartitioned u ↔ u.v₁ * u.f₂ = u.v₂ * u.f₁ ∨ u.v₁ + u.f₁ = u.v₂ + u.f₂ :=
sorry

end NUMINAMATH_CALUDE_equal_prob_conditions_l1630_163000


namespace NUMINAMATH_CALUDE_total_votes_is_129_l1630_163003

/-- The number of votes for each cake type in a baking contest. -/
structure CakeVotes where
  witch : ℕ
  unicorn : ℕ
  dragon : ℕ
  mermaid : ℕ
  fairy : ℕ
  phoenix : ℕ

/-- The conditions for the cake voting contest. -/
def contestConditions (votes : CakeVotes) : Prop :=
  votes.witch = 15 ∧
  votes.unicorn = 3 * votes.witch ∧
  votes.dragon = votes.witch + 7 ∧
  votes.dragon = (votes.mermaid * 5) / 4 ∧
  votes.mermaid = votes.dragon - 3 ∧
  votes.mermaid = 2 * votes.fairy ∧
  votes.fairy = votes.witch - 5 ∧
  votes.phoenix = votes.dragon - (votes.dragon / 5) ∧
  votes.phoenix = votes.fairy + 15

/-- The theorem stating that given the contest conditions, the total number of votes is 129. -/
theorem total_votes_is_129 (votes : CakeVotes) :
  contestConditions votes → votes.witch + votes.unicorn + votes.dragon + votes.mermaid + votes.fairy + votes.phoenix = 129 := by
  sorry


end NUMINAMATH_CALUDE_total_votes_is_129_l1630_163003


namespace NUMINAMATH_CALUDE_dividend_divisible_by_divisor_l1630_163089

/-- The dividend polynomial -/
def dividend (x : ℂ) : ℂ := x^55 + x^44 + x^33 + x^22 + x^11 + 1

/-- The divisor polynomial -/
def divisor (x : ℂ) : ℂ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

/-- Theorem stating that the dividend is divisible by the divisor -/
theorem dividend_divisible_by_divisor :
  ∃ q : ℂ → ℂ, ∀ x, dividend x = (divisor x) * (q x) := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisible_by_divisor_l1630_163089


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1630_163044

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 70 = (X - 7) * q + 63 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1630_163044


namespace NUMINAMATH_CALUDE_rent_equation_l1630_163075

/-- The monthly rent of Janet's apartment -/
def monthly_rent : ℝ := 1250

/-- Janet's savings -/
def savings : ℝ := 2225

/-- Additional amount Janet needs -/
def additional_amount : ℝ := 775

/-- Deposit required by the landlord -/
def deposit : ℝ := 500

/-- Number of months' rent required in advance -/
def months_in_advance : ℕ := 2

theorem rent_equation :
  2 * monthly_rent + deposit = savings + additional_amount :=
by sorry

end NUMINAMATH_CALUDE_rent_equation_l1630_163075


namespace NUMINAMATH_CALUDE_fifteenth_recalibration_in_march_l1630_163070

/-- Calculates the month of the nth recalibration given a start month and recalibration interval -/
def recalibrationMonth (startMonth : Nat) (interval : Nat) (n : Nat) : Nat :=
  ((startMonth - 1) + (n - 1) * interval) % 12 + 1

/-- The month of the 15th recalibration is March (month 3) -/
theorem fifteenth_recalibration_in_march :
  recalibrationMonth 1 7 15 = 3 := by
  sorry

#eval recalibrationMonth 1 7 15

end NUMINAMATH_CALUDE_fifteenth_recalibration_in_march_l1630_163070


namespace NUMINAMATH_CALUDE_magical_elixir_combinations_l1630_163028

/-- The number of magical herbs. -/
def num_herbs : ℕ := 4

/-- The number of enchanted crystals. -/
def num_crystals : ℕ := 6

/-- The number of incompatible herb-crystal pairs. -/
def num_incompatible : ℕ := 3

/-- The number of valid combinations for the magical elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible

theorem magical_elixir_combinations :
  valid_combinations = 21 :=
by sorry

end NUMINAMATH_CALUDE_magical_elixir_combinations_l1630_163028


namespace NUMINAMATH_CALUDE_diet_soda_count_l1630_163077

/-- The number of diet soda bottles in a grocery store -/
def diet_soda : ℕ := sorry

/-- The number of regular soda bottles in the grocery store -/
def regular_soda : ℕ := 60

/-- The difference between regular and diet soda bottles -/
def difference : ℕ := 41

theorem diet_soda_count : diet_soda = 19 :=
  by
  have h1 : regular_soda = diet_soda + difference := sorry
  sorry

end NUMINAMATH_CALUDE_diet_soda_count_l1630_163077


namespace NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_equals_four_l1630_163056

theorem a_squared_b_plus_ab_squared_equals_four :
  let a : ℝ := 2 + Real.sqrt 3
  let b : ℝ := 2 - Real.sqrt 3
  a^2 * b + a * b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_equals_four_l1630_163056


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l1630_163052

theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, x ≥ 4 ∧ Real.sqrt (x + 2 - 2 * Real.sqrt (x - 4)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 4)) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l1630_163052


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1630_163012

theorem geometric_sequence_common_ratio 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 27) 
  (h₂ : a₂ = 54) 
  (h₃ : a₃ = 108) 
  (h_geom : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : 
  ∃ r : ℝ, r = 2 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1630_163012


namespace NUMINAMATH_CALUDE_distance_to_origin_of_point_on_parabola_l1630_163086

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * C.p * x

theorem distance_to_origin_of_point_on_parabola
  (C : Parabola)
  (A : PointOnParabola C)
  (h1 : Real.sqrt ((A.x - C.p/2)^2 + A.y^2) = 6)  -- Distance from A to focus is 6
  (h2 : A.x = 3)  -- Distance from A to y-axis is 3
  : Real.sqrt (A.x^2 + A.y^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_point_on_parabola_l1630_163086


namespace NUMINAMATH_CALUDE_greatest_common_measure_l1630_163053

theorem greatest_common_measure (a b c : ℕ) (ha : a = 700) (hb : b = 385) (hc : c = 1295) :
  Nat.gcd a (Nat.gcd b c) = 35 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_measure_l1630_163053


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1630_163066

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∃ (q : ℝ) (a₁ : ℝ), ∀ n, a n = a₁ * q^(n-1))
  (h_condition : 2 * a 4 = a 6 - a 5) :
  ∃ (q : ℝ), (q = -1 ∨ q = 2) ∧ 
    (∃ (a₁ : ℝ), ∀ n, a n = a₁ * q^(n-1)) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1630_163066


namespace NUMINAMATH_CALUDE_ab_equality_l1630_163061

theorem ab_equality (a b : ℚ) (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * a * b = 800 := by
  sorry

end NUMINAMATH_CALUDE_ab_equality_l1630_163061


namespace NUMINAMATH_CALUDE_opposite_face_in_cube_net_l1630_163024

-- Define the faces of the cube
inductive Face : Type
  | x | A | B | C | D | Z

-- Define the cube net structure
structure CubeNet where
  faces : List Face
  surrounding : List Face
  not_connected : Face

-- Define the property of being opposite in a cube
def opposite (f1 f2 : Face) : Prop := sorry

-- Theorem statement
theorem opposite_face_in_cube_net (net : CubeNet) :
  net.faces = [Face.x, Face.A, Face.B, Face.C, Face.D, Face.Z] →
  net.surrounding = [Face.A, Face.B, Face.Z, Face.C] →
  net.not_connected = Face.D →
  opposite Face.x Face.D :=
by sorry

end NUMINAMATH_CALUDE_opposite_face_in_cube_net_l1630_163024


namespace NUMINAMATH_CALUDE_circle_parabola_height_difference_l1630_163068

/-- Given a circle inside the parabola y = 4x^2, tangent at two points,
    prove the height difference between the circle's center and tangency points. -/
theorem circle_parabola_height_difference (a : ℝ) : 
  let parabola (x : ℝ) := 4 * x^2
  let tangency_point := (a, parabola a)
  let circle_center := (0, a^2 + 1/8)
  circle_center.2 - tangency_point.2 = -3 * a^2 + 1/8 :=
by sorry

end NUMINAMATH_CALUDE_circle_parabola_height_difference_l1630_163068


namespace NUMINAMATH_CALUDE_find_number_l1630_163045

theorem find_number : ∃ N : ℕ,
  (N = (555 + 445) * (2 * (555 - 445)) + 30) ∧ 
  (N = 220030) := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1630_163045


namespace NUMINAMATH_CALUDE_gcd_lcm_360_possibilities_l1630_163008

theorem gcd_lcm_360_possibilities (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), S.card = 23 ∧ (∀ x, x ∈ S ↔ ∃ (a b : ℕ+), Nat.gcd a b = x ∧ Nat.gcd a b * Nat.lcm a b = 360)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_360_possibilities_l1630_163008


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_nine_l1630_163005

theorem sum_of_squares_zero_implies_sum_nine (a b c : ℝ) 
  (h : 2 * (a - 2)^2 + 3 * (b - 3)^2 + 4 * (c - 4)^2 = 0) : 
  a + b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_nine_l1630_163005


namespace NUMINAMATH_CALUDE_cans_left_to_load_l1630_163029

/-- Given a supplier packing cartons of canned juice, this theorem proves
    the number of cans left to be loaded on the truck. -/
theorem cans_left_to_load (cans_per_carton : ℕ) (total_cartons : ℕ) (loaded_cartons : ℕ)
    (h1 : cans_per_carton = 20)
    (h2 : total_cartons = 50)
    (h3 : loaded_cartons = 40) :
    (total_cartons - loaded_cartons) * cans_per_carton = 200 := by
  sorry

end NUMINAMATH_CALUDE_cans_left_to_load_l1630_163029


namespace NUMINAMATH_CALUDE_function_characterization_l1630_163082

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : f 1 = 1)
  (h2 : ∀ x y : ℝ, f (x + y) = 3^y * f x + 2^x * f y) :
  ∀ x : ℝ, f x = 3^x - 2^x := by
sorry

end NUMINAMATH_CALUDE_function_characterization_l1630_163082


namespace NUMINAMATH_CALUDE_smaller_part_is_4000_l1630_163027

/-- Represents an investment split into two parts -/
structure Investment where
  total : ℝ
  greater_part : ℝ
  smaller_part : ℝ
  greater_rate : ℝ
  smaller_rate : ℝ

/-- Conditions for the investment problem -/
def investment_conditions (i : Investment) : Prop :=
  i.total = 10000 ∧
  i.greater_part + i.smaller_part = i.total ∧
  i.greater_rate = 0.06 ∧
  i.smaller_rate = 0.05 ∧
  i.greater_rate * i.greater_part = i.smaller_rate * i.smaller_part + 160

/-- Theorem stating that under the given conditions, the smaller part of the investment is 4000 -/
theorem smaller_part_is_4000 (i : Investment) 
  (h : investment_conditions i) : i.smaller_part = 4000 := by
  sorry

end NUMINAMATH_CALUDE_smaller_part_is_4000_l1630_163027


namespace NUMINAMATH_CALUDE_bus_truck_speed_ratio_l1630_163023

theorem bus_truck_speed_ratio :
  ∀ (distance : ℝ) (bus_time truck_time : ℝ),
    bus_time = 10 →
    truck_time = 15 →
    (distance / bus_time) / (distance / truck_time) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bus_truck_speed_ratio_l1630_163023


namespace NUMINAMATH_CALUDE_find_M_l1630_163016

theorem find_M : ∃ (M : ℕ), M > 0 ∧ 18^2 * 45^2 = 15^2 * M^2 ∧ M = 54 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l1630_163016


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l1630_163022

theorem complex_number_coordinates (z : ℂ) : z = (2 * Complex.I) / (1 + Complex.I) → z.re = 1 ∧ z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l1630_163022


namespace NUMINAMATH_CALUDE_adults_eaten_correct_l1630_163085

/-- Represents the number of adults who had their meal -/
def adults_eaten : ℕ := 42

/-- Represents the total number of adults in the group -/
def total_adults : ℕ := 55

/-- Represents the total number of children in the group -/
def total_children : ℕ := 70

/-- Represents the meal capacity for adults -/
def meal_capacity_adults : ℕ := 70

/-- Represents the meal capacity for children -/
def meal_capacity_children : ℕ := 90

/-- Represents the number of children that can be catered with the remaining food -/
def remaining_children : ℕ := 36

theorem adults_eaten_correct : 
  adults_eaten = 42 ∧
  total_adults = 55 ∧
  total_children = 70 ∧
  meal_capacity_adults = 70 ∧
  meal_capacity_children = 90 ∧
  remaining_children = 36 ∧
  meal_capacity_children - (adults_eaten * meal_capacity_children / meal_capacity_adults) = remaining_children :=
by sorry

end NUMINAMATH_CALUDE_adults_eaten_correct_l1630_163085


namespace NUMINAMATH_CALUDE_mode_and_median_of_data_set_l1630_163063

def data_set : List ℝ := [1, 1, 4, 5, 5, 5]

/-- The mode of a list of real numbers -/
def mode (l : List ℝ) : ℝ := sorry

/-- The median of a list of real numbers -/
def median (l : List ℝ) : ℝ := sorry

theorem mode_and_median_of_data_set :
  mode data_set = 5 ∧ median data_set = 4.5 := by sorry

end NUMINAMATH_CALUDE_mode_and_median_of_data_set_l1630_163063


namespace NUMINAMATH_CALUDE_arrangement_count_l1630_163090

def arrange_people (n : ℕ) (k : ℕ) (m : ℕ) : Prop :=
  (n = 6) ∧ (k = 2) ∧ (m = 4)

theorem arrangement_count (n k m : ℕ) (h : arrange_people n k m) : 
  (Nat.choose n 2 * 2) + (Nat.choose n 3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1630_163090


namespace NUMINAMATH_CALUDE_jacket_final_price_l1630_163034

/-- The final price of a jacket after two successive discounts -/
theorem jacket_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  initial_price = 20 ∧ discount1 = 0.4 ∧ discount2 = 0.25 →
  initial_price * (1 - discount1) * (1 - discount2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_jacket_final_price_l1630_163034


namespace NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l1630_163031

theorem quadratic_reciprocal_roots (p q : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x*y = 1) →
  ((p ≥ 2 ∨ p ≤ -2) ∧ q = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l1630_163031


namespace NUMINAMATH_CALUDE_race_outcomes_count_l1630_163055

/-- The number of participants in the race -/
def total_participants : ℕ := 6

/-- The number of participants eligible for top three positions -/
def eligible_participants : ℕ := total_participants - 1

/-- The number of top positions to be filled -/
def top_positions : ℕ := 3

/-- Calculates the number of permutations for selecting k items from n items -/
def permutations (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

/-- The main theorem stating the number of possible race outcomes -/
theorem race_outcomes_count : 
  permutations eligible_participants top_positions = 60 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_count_l1630_163055


namespace NUMINAMATH_CALUDE_collinear_probability_4x5_l1630_163098

/-- Represents a grid of dots -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Counts the number of sets of 4 collinear dots in a grid -/
def collinearSets (g : Grid) : ℕ := sorry

/-- The probability of selecting 4 collinear dots from a grid -/
def collinearProbability (g : Grid) : ℚ :=
  (collinearSets g : ℚ) / choose (g.rows * g.cols) 4

theorem collinear_probability_4x5 :
  let g : Grid := ⟨4, 5⟩
  collinearProbability g = 9 / 4845 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_4x5_l1630_163098


namespace NUMINAMATH_CALUDE_lanas_roses_l1630_163033

theorem lanas_roses (tulips : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) 
  (h1 : tulips = 36)
  (h2 : used_flowers = 70)
  (h3 : extra_flowers = 3) :
  tulips + (used_flowers + extra_flowers - tulips) = 73 :=
by sorry

end NUMINAMATH_CALUDE_lanas_roses_l1630_163033


namespace NUMINAMATH_CALUDE_mi_gu_li_fen_problem_l1630_163039

/-- The "Mi-Gu-Li-Fen" problem from the "Mathematical Treatise in Nine Sections" -/
theorem mi_gu_li_fen_problem (total_mixture : ℚ) (sample_size : ℕ) (wheat_in_sample : ℕ) 
  (h1 : total_mixture = 1512)
  (h2 : sample_size = 216)
  (h3 : wheat_in_sample = 27) :
  (total_mixture * (wheat_in_sample : ℚ) / (sample_size : ℚ)) = 189 := by
  sorry

end NUMINAMATH_CALUDE_mi_gu_li_fen_problem_l1630_163039


namespace NUMINAMATH_CALUDE_jane_sum_minus_liam_sum_l1630_163041

def jane_list : List Nat := List.range 50

def replace_3_with_2 (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def liam_list : List Nat := jane_list.map replace_3_with_2

theorem jane_sum_minus_liam_sum : 
  jane_list.sum - liam_list.sum = 105 := by sorry

end NUMINAMATH_CALUDE_jane_sum_minus_liam_sum_l1630_163041


namespace NUMINAMATH_CALUDE_b_present_age_l1630_163043

/-- Given two people A and B, prove that B's present age is 34 years -/
theorem b_present_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
  (a = b + 4) →              -- A is now 4 years older than B
  b = 34 := by
sorry

end NUMINAMATH_CALUDE_b_present_age_l1630_163043


namespace NUMINAMATH_CALUDE_brothers_age_difference_l1630_163006

/-- The age difference between two brothers -/
def age_difference (mark_age john_age : ℕ) : ℕ :=
  mark_age - john_age

theorem brothers_age_difference :
  ∀ (mark_age john_age parents_age : ℕ),
    mark_age = 18 →
    parents_age = 5 * john_age →
    parents_age = 40 →
    age_difference mark_age john_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_difference_l1630_163006


namespace NUMINAMATH_CALUDE_rohan_salary_l1630_163017

/-- Rohan's monthly expenses and savings -/
structure RohanFinances where
  salary : ℝ
  food_percent : ℝ
  rent_percent : ℝ
  entertainment_percent : ℝ
  conveyance_percent : ℝ
  education_percent : ℝ
  utilities_percent : ℝ
  savings : ℝ

/-- Theorem stating Rohan's monthly salary given his expenses and savings -/
theorem rohan_salary (r : RohanFinances) : 
  r.food_percent = 0.30 →
  r.rent_percent = 0.20 →
  r.entertainment_percent = r.food_percent / 2 →
  r.conveyance_percent = r.entertainment_percent * 1.25 →
  r.education_percent = 0.05 →
  r.utilities_percent = 0.10 →
  r.savings = 2500 →
  r.salary = 200000 := by
  sorry

end NUMINAMATH_CALUDE_rohan_salary_l1630_163017


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1630_163057

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 + (k + 1) * x + (k^2 - 3)

-- Define the interval for k
def k_interval : Set ℝ := {k | (1 - 2 * Real.sqrt 10) / 3 ≤ k ∧ k ≤ (1 + 2 * Real.sqrt 10) / 3}

-- Theorem statement
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, quadratic k x = 0) ↔ k ∈ k_interval :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1630_163057


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_l1630_163042

theorem cubic_sum_of_roots (r s : ℝ) : 
  r^2 - 5*r + 6 = 0 → 
  s^2 - 5*s + 6 = 0 → 
  r^3 + s^3 = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_l1630_163042


namespace NUMINAMATH_CALUDE_problem_solution_l1630_163083

theorem problem_solution (a b c : ℝ) (h1 : a - b = 2) (h2 : a + c = 6) :
  (2*a + b + c) - 2*(a - b - c) = 12 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1630_163083


namespace NUMINAMATH_CALUDE_limit_at_neg_three_is_zero_l1630_163062

/-- The limit of (x^2 + 2x - 3)^2 / (x^3 + 4x^2 + 3x) as x approaches -3 is 0 -/
theorem limit_at_neg_three_is_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x + 3| ∧ |x + 3| < δ → 
    |(x^2 + 2*x - 3)^2 / (x^3 + 4*x^2 + 3*x) - 0| < ε :=
by
  sorry

#check limit_at_neg_three_is_zero

end NUMINAMATH_CALUDE_limit_at_neg_three_is_zero_l1630_163062


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1630_163058

theorem sandwich_combinations (meat_types cheese_types : ℕ) 
  (h1 : meat_types = 12) 
  (h2 : cheese_types = 8) : 
  meat_types * (cheese_types.choose 3) = 672 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1630_163058


namespace NUMINAMATH_CALUDE_jellybean_probability_l1630_163054

def total_jellybeans : ℕ := 15
def red_jellybeans : ℕ := 6
def blue_jellybeans : ℕ := 3
def green_jellybeans : ℕ := 6
def picked_jellybeans : ℕ := 4

theorem jellybean_probability :
  let total_combinations := Nat.choose total_jellybeans picked_jellybeans
  let successful_combinations := Nat.choose red_jellybeans 2 * Nat.choose green_jellybeans 2
  (successful_combinations : ℚ) / total_combinations = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l1630_163054


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1630_163096

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x : ℝ | 3 < x ∧ x < 9}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 9} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1630_163096


namespace NUMINAMATH_CALUDE_root_difference_equals_1993_l1630_163032

theorem root_difference_equals_1993 : ∃ m n : ℝ,
  (1992 * m)^2 - 1991 * 1993 * m - 1 = 0 ∧
  n^2 + 1991 * n - 1992 = 0 ∧
  (∀ x : ℝ, (1992 * x)^2 - 1991 * 1993 * x - 1 = 0 → x ≤ m) ∧
  (∀ y : ℝ, y^2 + 1991 * y - 1992 = 0 → y ≤ n) ∧
  m - n = 1993 :=
sorry

end NUMINAMATH_CALUDE_root_difference_equals_1993_l1630_163032


namespace NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l1630_163026

theorem sqrt_sum_greater_than_sqrt_of_sum : Real.sqrt 2 + Real.sqrt 3 > Real.sqrt (2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l1630_163026


namespace NUMINAMATH_CALUDE_rectangle_ratio_problem_l1630_163084

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- Represents a regular pentagon with side length a -/
structure Pentagon where
  a : ℝ

/-- Theorem statement for the rectangle ratio problem -/
theorem rectangle_ratio_problem (p : Pentagon) (r : Rectangle) : 
  -- The pentagon is regular and has side length a
  p.a > 0 →
  -- Five congruent rectangles are placed around the pentagon
  -- The shorter side of each rectangle lies against a side of the inner pentagon
  r.y = p.a →
  -- The area of the outer pentagon is 5 times that of the inner pentagon
  -- (We use this as an assumption without deriving it geometrically)
  r.x + r.y = Real.sqrt 5 * p.a →
  -- The ratio of the longer side to the shorter side of each rectangle is √5 - 1
  r.x / r.y = Real.sqrt 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_problem_l1630_163084


namespace NUMINAMATH_CALUDE_bc_values_l1630_163092

theorem bc_values (a b c : ℝ) 
  (sum_eq : a + b + c = 100)
  (prod_sum_eq : a * b + b * c + c * a = 20)
  (mixed_prod_eq : (a + b) * (a + c) = 24) :
  b * c = -176 ∨ b * c = 224 :=
by sorry

end NUMINAMATH_CALUDE_bc_values_l1630_163092


namespace NUMINAMATH_CALUDE_cos_x_plus_pi_sixth_l1630_163036

theorem cos_x_plus_pi_sixth (x : ℝ) (h : Real.sin (π / 3 - x) = 3 / 5) :
  Real.cos (x + π / 6) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_pi_sixth_l1630_163036


namespace NUMINAMATH_CALUDE_smallest_odd_triangle_perimeter_l1630_163009

/-- A triangle with consecutive odd integer side lengths. -/
structure OddTriangle where
  a : ℕ
  is_odd : Odd a
  satisfies_inequality : a + (a + 2) > (a + 4) ∧ a + (a + 4) > (a + 2) ∧ (a + 2) + (a + 4) > a

/-- The perimeter of an OddTriangle. -/
def perimeter (t : OddTriangle) : ℕ := t.a + (t.a + 2) + (t.a + 4)

/-- The statement to be proven. -/
theorem smallest_odd_triangle_perimeter :
  (∃ t : OddTriangle, ∀ t' : OddTriangle, perimeter t ≤ perimeter t') ∧
  (∃ t : OddTriangle, perimeter t = 15) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_triangle_perimeter_l1630_163009


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1630_163065

theorem binomial_expansion_coefficient (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a₀ + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a₀ = 32 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1630_163065


namespace NUMINAMATH_CALUDE_infinite_solutions_l1630_163067

theorem infinite_solutions (b : ℝ) :
  (∀ x, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_l1630_163067


namespace NUMINAMATH_CALUDE_all_expressions_correct_l1630_163004

theorem all_expressions_correct (x y : ℚ) (h : x / y = 5 / 3) :
  (2 * x + y) / y = 13 / 3 ∧
  y / (y - 2 * x) = 3 / (-7) ∧
  (x + y) / x = 8 / 5 ∧
  x / (3 * y) = 5 / 9 ∧
  (x - 2 * y) / y = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_all_expressions_correct_l1630_163004


namespace NUMINAMATH_CALUDE_line_moved_up_two_units_l1630_163007

/-- Represents a line in the 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Moves a line vertically by a given amount --/
def moveLine (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

/-- The theorem stating that moving y = 4x - 1 up by 2 units results in y = 4x + 1 --/
theorem line_moved_up_two_units :
  let original_line : Line := { slope := 4, intercept := -1 }
  let moved_line := moveLine original_line 2
  moved_line = { slope := 4, intercept := 1 } := by
  sorry

end NUMINAMATH_CALUDE_line_moved_up_two_units_l1630_163007


namespace NUMINAMATH_CALUDE_fertilizing_to_mowing_ratio_l1630_163059

def mowing_time : ℕ := 40
def total_time : ℕ := 120

def fertilizing_time : ℕ := total_time - mowing_time

theorem fertilizing_to_mowing_ratio :
  (fertilizing_time : ℚ) / mowing_time = 2 := by sorry

end NUMINAMATH_CALUDE_fertilizing_to_mowing_ratio_l1630_163059


namespace NUMINAMATH_CALUDE_imaginary_part_reciprocal_l1630_163078

theorem imaginary_part_reciprocal (a : ℝ) (h1 : a > 0) :
  let z : ℂ := a + Complex.I
  (Complex.abs z = Real.sqrt 5) →
  Complex.im (z⁻¹) = -1/5 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_reciprocal_l1630_163078


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_three_l1630_163064

/-- Given a polynomial ax^5 - bx^3 + cx - 7 that equals 65 when x = 3,
    prove that it equals -79 when x = -3 -/
theorem polynomial_value_at_negative_three 
  (a b c : ℝ) 
  (h : a * 3^5 - b * 3^3 + c * 3 - 7 = 65) :
  a * (-3)^5 - b * (-3)^3 + c * (-3) - 7 = -79 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_value_at_negative_three_l1630_163064


namespace NUMINAMATH_CALUDE_sphere_surface_area_in_cube_l1630_163050

theorem sphere_surface_area_in_cube (edge_length : Real) (surface_area : Real) :
  edge_length = 2 →
  surface_area = 4 * Real.pi →
  ∃ (r : Real),
    r = edge_length / 2 ∧
    surface_area = 4 * Real.pi * r^2 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_in_cube_l1630_163050


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l1630_163011

theorem cot_thirty_degrees : 
  let θ : Real := 30 * π / 180 -- Convert 30 degrees to radians
  let cot (x : Real) := 1 / Real.tan x -- Definition of cotangent
  (Real.tan θ = 1 / Real.sqrt 3) → -- Given condition
  (cot θ = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l1630_163011


namespace NUMINAMATH_CALUDE_remainder_problem_l1630_163021

theorem remainder_problem : 2851 * 7347 * 419^2 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1630_163021


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1630_163060

theorem geometric_series_sum : 
  let a₁ : ℚ := 1 / 4
  let r : ℚ := -1 / 4
  let n : ℕ := 6
  let series_sum := a₁ * (1 - r^n) / (1 - r)
  series_sum = 81 / 405 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1630_163060


namespace NUMINAMATH_CALUDE_two_valid_positions_l1630_163019

/-- Represents a square in the polygon arrangement -/
structure Square :=
  (id : Char)

/-- Represents the flat arrangement of squares -/
def FlatArrangement := List Square

/-- Represents a position where an additional square can be attached -/
inductive AttachmentPosition
  | Right : Square → AttachmentPosition
  | Left : Square → AttachmentPosition

/-- Checks if a given attachment position allows folding into a cube with two opposite faces missing -/
def allows_cube_folding (arrangement : FlatArrangement) (pos : AttachmentPosition) : Prop :=
  sorry

/-- The main theorem stating that there are exactly two valid attachment positions -/
theorem two_valid_positions (arrangement : FlatArrangement) :
  (arrangement.length = 4) →
  (∃ A B C D : Square, arrangement = [A, B, C, D]) →
  (∃! (pos1 pos2 : AttachmentPosition),
    pos1 ≠ pos2 ∧
    allows_cube_folding arrangement pos1 ∧
    allows_cube_folding arrangement pos2 ∧
    (∀ pos, allows_cube_folding arrangement pos → (pos = pos1 ∨ pos = pos2))) :=
  sorry

end NUMINAMATH_CALUDE_two_valid_positions_l1630_163019


namespace NUMINAMATH_CALUDE_triangle_inequality_satisfied_l1630_163014

theorem triangle_inequality_satisfied (a b c : ℝ) (ha : a = 8) (hb : b = 8) (hc : c = 15) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_satisfied_l1630_163014


namespace NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l1630_163072

theorem freshmen_in_liberal_arts (total_students : ℝ) (freshmen_percent : ℝ) 
  (psych_majors_percent : ℝ) (freshmen_psych_liberal_arts_percent : ℝ) :
  freshmen_percent = 80 →
  psych_majors_percent = 50 →
  freshmen_psych_liberal_arts_percent = 24 →
  (freshmen_psych_liberal_arts_percent * total_students) / 
    (psych_majors_percent / 100 * freshmen_percent * total_students / 100) = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l1630_163072


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1630_163001

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1630_163001


namespace NUMINAMATH_CALUDE_fraction_addition_l1630_163002

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 1 / a + 2 / a = 3 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1630_163002


namespace NUMINAMATH_CALUDE_abc_sum_mod_seven_l1630_163049

theorem abc_sum_mod_seven (a b c : ℤ) : 
  a ∈ ({1, 2, 3, 4, 5, 6} : Set ℤ) →
  b ∈ ({1, 2, 3, 4, 5, 6} : Set ℤ) →
  c ∈ ({1, 2, 3, 4, 5, 6} : Set ℤ) →
  (a * b * c) % 7 = 1 →
  (2 * c) % 7 = 5 →
  (3 * b) % 7 = (4 + b) % 7 →
  (a + b + c) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_seven_l1630_163049


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1630_163079

theorem quadratic_equation_solution (r : ℝ) : 
  (r^2 - 3) / 3 = (5 - r) / 2 ↔ r = (-3 + Real.sqrt 177) / 4 ∨ r = (-3 - Real.sqrt 177) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1630_163079


namespace NUMINAMATH_CALUDE_bridge_length_at_least_train_length_l1630_163094

/-- Proves that the length of a bridge is at least as long as a train, given the train's length,
    speed, and time to cross the bridge. -/
theorem bridge_length_at_least_train_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 200)
  (h2 : train_speed_kmh = 32)
  (h3 : crossing_time = 20)
  : ∃ (bridge_length : ℝ), bridge_length ≥ train_length :=
by
  sorry

#check bridge_length_at_least_train_length

end NUMINAMATH_CALUDE_bridge_length_at_least_train_length_l1630_163094
