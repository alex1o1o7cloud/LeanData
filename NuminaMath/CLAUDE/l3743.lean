import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_integer_expression_l3743_374330

theorem quadratic_integer_expression (A B C : ℤ) :
  ∃ (k l m : ℚ), 
    (∀ x, A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m) ∧
    ((∀ x : ℤ, ∃ y : ℤ, A * x^2 + B * x + C = y) ↔ 
      (∃ (k' l' m' : ℤ), k = k' ∧ l = l' ∧ m = m')) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_expression_l3743_374330


namespace NUMINAMATH_CALUDE_total_road_signs_l3743_374361

/-- The number of road signs at four intersections -/
def road_signs (first second third fourth : ℕ) : Prop :=
  (second = first + first / 4) ∧
  (third = 2 * second) ∧
  (fourth = third - 20) ∧
  (first + second + third + fourth = 270)

/-- Theorem: There are 270 road signs in total given the conditions -/
theorem total_road_signs : ∃ (first second third fourth : ℕ),
  first = 40 ∧ road_signs first second third fourth := by
  sorry

end NUMINAMATH_CALUDE_total_road_signs_l3743_374361


namespace NUMINAMATH_CALUDE_peanut_butter_weight_calculation_l3743_374332

-- Define the ratio of oil to peanuts
def oil_to_peanuts_ratio : ℚ := 2 / 8

-- Define the amount of oil used
def oil_used : ℚ := 4

-- Define the function to calculate the total weight of peanut butter
def peanut_butter_weight (oil_amount : ℚ) : ℚ :=
  oil_amount + (oil_amount / oil_to_peanuts_ratio) * 8

-- Theorem statement
theorem peanut_butter_weight_calculation :
  peanut_butter_weight oil_used = 20 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_weight_calculation_l3743_374332


namespace NUMINAMATH_CALUDE_sugar_price_increase_l3743_374366

theorem sugar_price_increase (initial_price : ℝ) (consumption_reduction : ℝ) (new_price : ℝ) :
  initial_price = 2 →
  consumption_reduction = 0.6 →
  (1 - consumption_reduction) * new_price = initial_price →
  new_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_sugar_price_increase_l3743_374366


namespace NUMINAMATH_CALUDE_investment_value_after_two_years_l3743_374316

/-- Calculates the value of an investment after a given period --/
def investment_value (income : ℝ) (income_expenditure_ratio : ℝ × ℝ) 
  (savings_rate : ℝ) (tax_rate : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  let expenditure := income * income_expenditure_ratio.2 / income_expenditure_ratio.1
  let savings := income - expenditure
  let amount_saved := income * savings_rate
  let tax_deductions := income * tax_rate
  let net_investment := amount_saved - tax_deductions
  net_investment * (1 + interest_rate) ^ years

/-- Theorem stating the value of the investment after two years --/
theorem investment_value_after_two_years :
  investment_value 19000 (5, 4) 0.15 0.10 0.08 2 = 1108.08 := by
  sorry

end NUMINAMATH_CALUDE_investment_value_after_two_years_l3743_374316


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_in_set_l3743_374352

theorem sqrt_2_irrational_in_set (S : Set ℝ) : 
  S = {1/7, Real.sqrt 2, (8 : ℝ) ^ (1/3), 1.010010001} → 
  ∃ x ∈ S, Irrational x ∧ x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_in_set_l3743_374352


namespace NUMINAMATH_CALUDE_harvester_problem_l3743_374399

/-- Represents the number of harvesters of each type -/
structure HarvesterCount where
  typeA : ℕ
  typeB : ℕ

/-- Represents a plan for introducing additional harvesters -/
structure IntroductionPlan where
  additionalTypeA : ℕ
  additionalTypeB : ℕ

/-- The problem statement -/
theorem harvester_problem 
  (total_harvesters : ℕ)
  (typeA_capacity : ℕ)
  (typeB_capacity : ℕ)
  (total_daily_harvest : ℕ)
  (new_target : ℕ)
  (additional_harvesters : ℕ)
  (h1 : total_harvesters = 20)
  (h2 : typeA_capacity = 80)
  (h3 : typeB_capacity = 120)
  (h4 : total_daily_harvest = 2080)
  (h5 : new_target > 2900)
  (h6 : additional_harvesters = 8) :
  ∃ (initial : HarvesterCount) (plans : List IntroductionPlan),
    initial.typeA + initial.typeB = total_harvesters ∧
    initial.typeA * typeA_capacity + initial.typeB * typeB_capacity = total_daily_harvest ∧
    initial.typeA = 8 ∧
    initial.typeB = 12 ∧
    plans.length = 3 ∧
    ∀ plan ∈ plans, 
      plan.additionalTypeA + plan.additionalTypeB = additional_harvesters ∧
      (initial.typeA + plan.additionalTypeA) * typeA_capacity + 
      (initial.typeB + plan.additionalTypeB) * typeB_capacity > new_target :=
by sorry

end NUMINAMATH_CALUDE_harvester_problem_l3743_374399


namespace NUMINAMATH_CALUDE_quadratic_integer_solutions_count_l3743_374390

theorem quadratic_integer_solutions_count : 
  ∃! (S : Finset ℚ), 
    (∀ k ∈ S, |k| < 100 ∧ 
      ∃! (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 
        3 * (x₁ : ℚ)^2 + k * x₁ + 8 = 0 ∧ 
        3 * (x₂ : ℚ)^2 + k * x₂ + 8 = 0) ∧
    Finset.card S = 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_solutions_count_l3743_374390


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l3743_374397

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- The origin point -/
def origin : Point := ⟨0, 0⟩

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Main theorem -/
theorem parabola_intersection_distance 
  (A B : Point) 
  (hA : A ∈ Parabola) 
  (hB : B ∈ Parabola) 
  (hline : ∃ (m c : ℝ), A.y = m * A.x + c ∧ B.y = m * B.x + c ∧ focus.y = m * focus.x + c) 
  (harea : triangleArea A origin focus = 3 * triangleArea B origin focus) :
  distance A B = 16/3 := 
sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l3743_374397


namespace NUMINAMATH_CALUDE_more_24_than_32_placements_l3743_374381

/-- Represents a chessboard configuration --/
structure Chessboard :=
  (size : Nat)
  (dominoes : Nat)

/-- Represents the number of ways to place dominoes on a chessboard --/
def PlacementCount (board : Chessboard) : Nat := sorry

/-- The 8x8 chessboard with 32 dominoes --/
def board32 : Chessboard :=
  { size := 8, dominoes := 32 }

/-- The 8x8 chessboard with 24 dominoes --/
def board24 : Chessboard :=
  { size := 8, dominoes := 24 }

/-- Theorem stating that there are more ways to place 24 dominoes than 32 dominoes --/
theorem more_24_than_32_placements : PlacementCount board24 > PlacementCount board32 := by
  sorry

end NUMINAMATH_CALUDE_more_24_than_32_placements_l3743_374381


namespace NUMINAMATH_CALUDE_plane_speed_with_wind_l3743_374317

theorem plane_speed_with_wind (distance : ℝ) (wind_speed : ℝ) (time_with_wind : ℝ) (time_against_wind : ℝ) :
  wind_speed = 24 ∧ time_with_wind = 5.5 ∧ time_against_wind = 6 →
  ∃ (plane_speed : ℝ),
    distance / time_with_wind = plane_speed + wind_speed ∧
    distance / time_against_wind = plane_speed - wind_speed ∧
    plane_speed + wind_speed = 576 ∧
    plane_speed - wind_speed = 528 := by
  sorry

end NUMINAMATH_CALUDE_plane_speed_with_wind_l3743_374317


namespace NUMINAMATH_CALUDE_locus_of_point_M_l3743_374309

/-- The locus of point M given an ellipse and conditions on point P -/
theorem locus_of_point_M (x₀ y₀ x y : ℝ) : 
  (4 * x₀^2 + y₀^2 = 4) →  -- P(x₀, y₀) is on the ellipse
  ((0, -y₀) = (2*(x - x₀), -2*y)) →  -- PD = 2MD condition
  (x^2 + y^2 = 1) -- M(x, y) is on the unit circle
  := by sorry

end NUMINAMATH_CALUDE_locus_of_point_M_l3743_374309


namespace NUMINAMATH_CALUDE_prime_cube_plus_one_l3743_374346

theorem prime_cube_plus_one (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p^x = y^3 + 1) ↔ p = 2 ∨ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_plus_one_l3743_374346


namespace NUMINAMATH_CALUDE_catering_weight_calculation_l3743_374335

/-- Calculates the total weight of silverware and plates for a catering event --/
theorem catering_weight_calculation 
  (silverware_weight : ℕ) 
  (silverware_per_setting : ℕ) 
  (plate_weight : ℕ) 
  (plates_per_setting : ℕ) 
  (tables : ℕ) 
  (settings_per_table : ℕ) 
  (backup_settings : ℕ) : 
  silverware_weight = 4 →
  silverware_per_setting = 3 →
  plate_weight = 12 →
  plates_per_setting = 2 →
  tables = 15 →
  settings_per_table = 8 →
  backup_settings = 20 →
  (silverware_weight * silverware_per_setting + 
   plate_weight * plates_per_setting) * 
  (tables * settings_per_table + backup_settings) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_catering_weight_calculation_l3743_374335


namespace NUMINAMATH_CALUDE_min_value_theorem_l3743_374324

theorem min_value_theorem (x : ℝ) (hx : x > 0) :
  (x^2 + 6 - Real.sqrt (x^4 + 36)) / x ≥ 12 / (2 * (Real.sqrt 6 + Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3743_374324


namespace NUMINAMATH_CALUDE_full_price_tickets_l3743_374378

theorem full_price_tickets (total : ℕ) (reduced : ℕ) (h1 : total = 25200) (h2 : reduced = 5400) :
  total - reduced = 19800 := by
  sorry

end NUMINAMATH_CALUDE_full_price_tickets_l3743_374378


namespace NUMINAMATH_CALUDE_youngest_child_age_l3743_374348

/-- Represents the age of the youngest child -/
def youngest_age : ℕ → Prop := λ x =>
  -- There are 5 children
  -- Children are born at intervals of 3 years each
  -- The sum of their ages is 60 years
  x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 60

/-- Proves that the age of the youngest child is 6 years -/
theorem youngest_child_age : youngest_age 6 := by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l3743_374348


namespace NUMINAMATH_CALUDE_second_division_divisor_l3743_374350

theorem second_division_divisor (x y : ℕ) (h1 : x > 0) (h2 : x % 10 = 3) (h3 : x / 10 = y)
  (h4 : ∃ k : ℕ, (2 * x) % k = 1 ∧ (2 * x) / k = 3 * y) (h5 : 11 * y - x = 2) :
  ∃ k : ℕ, (2 * x) % k = 1 ∧ (2 * x) / k = 3 * y ∧ k = 7 :=
by sorry

end NUMINAMATH_CALUDE_second_division_divisor_l3743_374350


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l3743_374387

theorem quadratic_root_transformation (p q r u v : ℝ) : 
  (p * u^2 + q * u + r = 0) → 
  (p * v^2 + q * v + r = 0) → 
  ((2*p*u + q)^2 - q^2 + 4*p*r = 0) ∧ 
  ((2*p*v + q)^2 - q^2 + 4*p*r = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l3743_374387


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l3743_374304

/-- Proves that given a car's speed in the second hour and its average speed over two hours, we can determine its speed in the first hour. -/
theorem car_speed_first_hour 
  (speed_second_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_second_hour = 80) 
  (h2 : average_speed = 90) : 
  ∃ (speed_first_hour : ℝ), 
    speed_first_hour = 100 ∧ 
    average_speed = (speed_first_hour + speed_second_hour) / 2 := by
  sorry

#check car_speed_first_hour

end NUMINAMATH_CALUDE_car_speed_first_hour_l3743_374304


namespace NUMINAMATH_CALUDE_baking_time_proof_l3743_374341

-- Define the total baking time for 4 pans
def total_time : ℕ := 28

-- Define the number of pans
def num_pans : ℕ := 4

-- Define the time for one pan
def time_per_pan : ℕ := total_time / num_pans

-- Theorem to prove
theorem baking_time_proof : time_per_pan = 7 := by
  sorry

end NUMINAMATH_CALUDE_baking_time_proof_l3743_374341


namespace NUMINAMATH_CALUDE_product_97_squared_l3743_374375

theorem product_97_squared : 97 * 97 = 9409 := by
  sorry

end NUMINAMATH_CALUDE_product_97_squared_l3743_374375


namespace NUMINAMATH_CALUDE_additional_savings_when_combined_l3743_374365

/-- The regular price of a window -/
def window_price : ℕ := 120

/-- The number of windows that need to be bought to get one free -/
def windows_for_free : ℕ := 6

/-- The number of windows Dave needs -/
def dave_windows : ℕ := 9

/-- The number of windows Doug needs -/
def doug_windows : ℕ := 10

/-- Calculate the cost of windows with the offer -/
def cost_with_offer (n : ℕ) : ℕ :=
  ((n + windows_for_free - 1) / windows_for_free * windows_for_free) * window_price

/-- Calculate the savings for a given number of windows -/
def savings (n : ℕ) : ℕ :=
  n * window_price - cost_with_offer n

/-- The theorem to be proved -/
theorem additional_savings_when_combined :
  savings (dave_windows + doug_windows) - (savings dave_windows + savings doug_windows) = 240 := by
  sorry

end NUMINAMATH_CALUDE_additional_savings_when_combined_l3743_374365


namespace NUMINAMATH_CALUDE_k_range_theorem_l3743_374315

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - x * log x + 2

theorem k_range_theorem (a b : ℝ) (h1 : 1/2 ≤ a) (h2 : a < b) 
  (h3 : ∀ x ∈ Set.Icc a b, ∃ k : ℝ, f x = k * (x + 2)) :
  ∃ k : ℝ, 1 < k ∧ k ≤ (9 + 2 * log 2) / 10 :=
sorry

end NUMINAMATH_CALUDE_k_range_theorem_l3743_374315


namespace NUMINAMATH_CALUDE_intersection_set_characterization_l3743_374329

/-- The set of positive real numbers m for which the graphs of y = (mx-1)^2 and y = √x + m 
    have exactly one intersection point on the interval [0,1] -/
def IntersectionSet : Set ℝ :=
  {m : ℝ | m > 0 ∧ ∃! x : ℝ, x ∈ [0, 1] ∧ (m * x - 1)^2 = Real.sqrt x + m}

/-- The theorem stating that the IntersectionSet is equal to (0,1] ∪ [3, +∞) -/
theorem intersection_set_characterization :
  IntersectionSet = Set.Ioo 0 1 ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_set_characterization_l3743_374329


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l3743_374323

theorem quadratic_function_minimum (a b c : ℝ) (h₁ : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let f' : ℝ → ℝ := λ x ↦ 2 * a * x + b
  (f' 0 > 0) →
  (∀ x : ℝ, f x ≥ 0) →
  f 1 / f' 0 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l3743_374323


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3743_374344

theorem quadratic_root_property (m : ℝ) : 
  (∃ α β : ℝ, (3 * α^2 + m * α - 4 = 0) ∧ 
              (3 * β^2 + m * β - 4 = 0) ∧ 
              (α * β = 2 * (α^3 + β^3))) ↔ 
  (m = -1.5 ∨ m = 6 ∨ m = -2.4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3743_374344


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_min_dot_product_midpoint_locus_l3743_374340

-- Define the line l: mx - y - m + 2 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y - m + 2 = 0

-- Define the circle C: x^2 + y^2 = 9
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

-- Define the intersection points A and B
def intersection_points (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  line_l m A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l m B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem 1: Line l passes through (1, 2) for all m
theorem line_passes_through_fixed_point (m : ℝ) :
  line_l m 1 2 := by sorry

-- Theorem 2: Minimum value of AC · AB is 8
theorem min_dot_product (m : ℝ) (A B : ℝ × ℝ) :
  intersection_points m A B →
  ∃ (C : ℝ × ℝ), circle_C C.1 C.2 ∧
  (∀ (D : ℝ × ℝ), circle_C D.1 D.2 →
    (A.1 - C.1) * (B.1 - A.1) + (A.2 - C.2) * (B.2 - A.2) ≥ 8) := by sorry

-- Theorem 3: Locus of midpoint of AB is a circle
theorem midpoint_locus (m : ℝ) (A B : ℝ × ℝ) :
  intersection_points m A B →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ((A.1 + B.1) / 2 - center.1)^2 + ((A.2 + B.2) / 2 - center.2)^2 = radius^2 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_min_dot_product_midpoint_locus_l3743_374340


namespace NUMINAMATH_CALUDE_trapezoid_not_constructible_l3743_374384

/-- Represents a quadrilateral with sides a, b, c, d where a is parallel to c -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_parallel_c : True  -- We use True here as a placeholder for the parallel condition

/-- The triangle inequality: the sum of any two sides of a triangle must be greater than the third side -/
def triangle_inequality (x y z : ℝ) : Prop := x + y > z ∧ y + z > x ∧ z + x > y

/-- Theorem stating that a trapezoid with the given side lengths cannot be formed -/
theorem trapezoid_not_constructible : ¬ ∃ (t : Trapezoid), t.a = 16 ∧ t.b = 13 ∧ t.c = 10 ∧ t.d = 6 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_not_constructible_l3743_374384


namespace NUMINAMATH_CALUDE_right_triangle_segment_relation_l3743_374371

/-- Given a right-angled triangle with legs of lengths a and b, and a segment of length d
    connecting the right angle vertex to the hypotenuse forming an angle δ with leg a,
    prove that 1/d = (cos δ)/a + (sin δ)/b. -/
theorem right_triangle_segment_relation (a b d : ℝ) (δ : ℝ) 
    (ha : a > 0) (hb : b > 0) (hd : d > 0) (hδ : 0 < δ ∧ δ < π / 2) :
    1 / d = (Real.cos δ) / a + (Real.sin δ) / b := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_segment_relation_l3743_374371


namespace NUMINAMATH_CALUDE_fraction_simplification_l3743_374343

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3743_374343


namespace NUMINAMATH_CALUDE_division_problem_l3743_374368

theorem division_problem (h : (7125 : ℝ) / 1.25 = 5700) : 
  ∃ x : ℝ, 712.5 / x = 57 ∧ x = 12.5 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3743_374368


namespace NUMINAMATH_CALUDE_hockey_skates_fraction_l3743_374310

/-- Proves that the fraction of money spent on hockey skates is 1/2 --/
theorem hockey_skates_fraction (initial_amount pad_cost remaining : ℚ)
  (h1 : initial_amount = 150)
  (h2 : pad_cost = 50)
  (h3 : remaining = 25) :
  (initial_amount - pad_cost - remaining) / initial_amount = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hockey_skates_fraction_l3743_374310


namespace NUMINAMATH_CALUDE_auction_theorem_l3743_374334

def auction_problem (starting_price : ℕ) (harry_first_bid : ℕ) (harry_final_bid : ℕ) : Prop :=
  let harry_bid := starting_price + harry_first_bid
  let second_bid := 2 * harry_bid
  let third_bid := second_bid + 3 * harry_first_bid
  harry_final_bid - third_bid = 2400

theorem auction_theorem : 
  auction_problem 300 200 4000 := by
  sorry

end NUMINAMATH_CALUDE_auction_theorem_l3743_374334


namespace NUMINAMATH_CALUDE_cubic_solution_sum_l3743_374395

theorem cubic_solution_sum (k : ℝ) (a b c : ℝ) : 
  (a^3 - 6*a^2 + 8*a + k = 0) →
  (b^3 - 6*b^2 + 8*b + k = 0) →
  (c^3 - 6*c^2 + 8*c + k = 0) →
  (k ≠ 0) →
  (a*b/c + b*c/a + c*a/b = 64/k - 12) := by sorry

end NUMINAMATH_CALUDE_cubic_solution_sum_l3743_374395


namespace NUMINAMATH_CALUDE_walking_speed_proof_l3743_374357

def jack_speed (x : ℝ) := x^2 - 11*x - 22
def jill_distance (x : ℝ) := x^2 - 4*x - 12
def jill_time (x : ℝ) := x + 6

theorem walking_speed_proof :
  ∃ (x : ℝ), 
    (jack_speed x = jill_distance x / jill_time x) ∧
    (jack_speed x = 10) :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_proof_l3743_374357


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l3743_374325

theorem unique_number_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 11))) →
  11 ∣ x →
  x = 59048 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l3743_374325


namespace NUMINAMATH_CALUDE_james_share_is_six_l3743_374303

/-- Calculates James' share of the cost for stickers -/
def james_share (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ) : ℚ :=
  (packs * stickers_per_pack * cost_per_sticker) / 2

/-- Proves that James' share of the cost is $6.00 given the problem conditions -/
theorem james_share_is_six :
  james_share 4 30 (1/10) = 6 := by
  sorry

end NUMINAMATH_CALUDE_james_share_is_six_l3743_374303


namespace NUMINAMATH_CALUDE_proportional_set_l3743_374391

/-- A set of four positive real numbers is proportional if and only if
    the product of the extremes equals the product of the means. -/
def IsProportional (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * d = b * c

/-- The set of line segments (2, 3, 4, 6) is proportional. -/
theorem proportional_set : IsProportional 2 3 4 6 := by
  sorry

end NUMINAMATH_CALUDE_proportional_set_l3743_374391


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l3743_374396

/-- 
Given a man's speed against the current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 12.4)
  (h2 : current_speed = 4.3) : 
  speed_against_current + 2 * current_speed = 21 := by
sorry

end NUMINAMATH_CALUDE_mans_speed_with_current_l3743_374396


namespace NUMINAMATH_CALUDE_eggs_remaining_l3743_374398

theorem eggs_remaining (original : ℝ) (removed : ℝ) (remaining : ℝ) : 
  original = 35.3 → removed = 4.5 → remaining = original - removed → remaining = 30.8 := by
  sorry

end NUMINAMATH_CALUDE_eggs_remaining_l3743_374398


namespace NUMINAMATH_CALUDE_negation_ab_zero_implies_a_zero_contrapositive_equilateral_60_degrees_l3743_374302

-- Define the basic concepts
def is_equilateral (t : Triangle) : Prop := sorry
def angle_measure (a : Angle) : ℝ := sorry

-- Statement for the negation of "If ab=0, then a=0"
theorem negation_ab_zero_implies_a_zero : 
  ∀ (a b : ℝ), (a * b ≠ 0) → (a ≠ 0) := by sorry

-- Statement for the contrapositive of "All angles in an equilateral triangle are 60°"
theorem contrapositive_equilateral_60_degrees :
  ∀ (t : Triangle) (a : Angle), 
    (angle_measure a ≠ 60) → ¬(is_equilateral t) := by sorry

end NUMINAMATH_CALUDE_negation_ab_zero_implies_a_zero_contrapositive_equilateral_60_degrees_l3743_374302


namespace NUMINAMATH_CALUDE_largest_square_area_l3743_374385

-- Define a right-angled triangle with squares on each side
structure RightTriangleWithSquares where
  xy : ℝ  -- Length of side XY
  yz : ℝ  -- Length of side YZ
  xz : ℝ  -- Length of hypotenuse XZ
  right_angle : xz^2 = xy^2 + yz^2  -- Pythagorean theorem

-- Theorem statement
theorem largest_square_area
  (t : RightTriangleWithSquares)
  (sum_of_squares : t.xy^2 + t.yz^2 + t.xz^2 = 450) :
  t.xz^2 = 225 := by
sorry

end NUMINAMATH_CALUDE_largest_square_area_l3743_374385


namespace NUMINAMATH_CALUDE_hanna_erasers_count_l3743_374383

/-- The number of erasers Tanya has -/
def tanya_erasers : ℕ := 20

/-- The number of red erasers Tanya has -/
def tanya_red_erasers : ℕ := tanya_erasers / 2

/-- The number of erasers Rachel has -/
def rachel_erasers : ℕ := tanya_red_erasers / 2 - 3

/-- The number of erasers Hanna has -/
def hanna_erasers : ℕ := rachel_erasers * 2

theorem hanna_erasers_count : hanna_erasers = 4 := by
  sorry

end NUMINAMATH_CALUDE_hanna_erasers_count_l3743_374383


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3743_374314

/-- Quadratic function y = ax² - 4ax + 3a -/
def quadratic_function (a x : ℝ) : ℝ := a * x^2 - 4 * a * x + 3 * a

theorem quadratic_function_properties :
  (∀ x, quadratic_function 1 x ≥ -1) ∧
  (∃ x, quadratic_function 1 x = -1) ∧
  (∀ x ∈ Set.Icc 1 4, quadratic_function (4/3) x ≤ 4) ∧
  (∃ x ∈ Set.Icc 1 4, quadratic_function (4/3) x = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3743_374314


namespace NUMINAMATH_CALUDE_company_picnic_teams_l3743_374322

theorem company_picnic_teams (managers : ℕ) (employees : ℕ) (teams : ℕ) :
  managers = 3 →
  employees = 3 →
  teams = 3 →
  (managers + employees) / teams = 2 := by
sorry

end NUMINAMATH_CALUDE_company_picnic_teams_l3743_374322


namespace NUMINAMATH_CALUDE_distinct_roots_find_m_l3743_374313

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 9

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (-2*m)^2 - 4*(m^2 - 9)

-- Theorem 1: The quadratic equation always has two distinct real roots
theorem distinct_roots (m : ℝ) : discriminant m > 0 := by sorry

-- Define the roots of the quadratic equation
noncomputable def x₁ (m : ℝ) : ℝ := sorry
noncomputable def x₂ (m : ℝ) : ℝ := sorry

-- Theorem 2: When x₂ = 3x₁, m = ±6
theorem find_m : 
  ∃ m : ℝ, (x₂ m = 3 * x₁ m) ∧ (m = 6 ∨ m = -6) := by sorry

end NUMINAMATH_CALUDE_distinct_roots_find_m_l3743_374313


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3743_374319

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3743_374319


namespace NUMINAMATH_CALUDE_starting_number_of_range_l3743_374326

/-- Given a sequence of 10 consecutive multiples of 5 ending with 65,
    prove that the first number in the sequence is 15. -/
theorem starting_number_of_range (seq : Fin 10 → ℕ) : 
  (∀ i : Fin 10, seq i % 5 = 0) →  -- All numbers are divisible by 5
  (∀ i : Fin 9, seq i.succ = seq i + 5) →  -- Consecutive multiples of 5
  seq 9 = 65 →  -- The last number is 65
  seq 0 = 15 := by  -- The first number is 15
sorry


end NUMINAMATH_CALUDE_starting_number_of_range_l3743_374326


namespace NUMINAMATH_CALUDE_cobbler_weekly_shoes_l3743_374312

/-- The number of pairs of shoes a cobbler can mend per hour -/
def shoes_per_hour : ℕ := 3

/-- The number of hours the cobbler works from Monday to Thursday each day -/
def hours_per_day : ℕ := 8

/-- The number of days the cobbler works full hours (Monday to Thursday) -/
def full_days : ℕ := 4

/-- The number of hours the cobbler works on Friday -/
def friday_hours : ℕ := 3

/-- The total number of pairs of shoes the cobbler can mend in a week -/
def total_shoes : ℕ := shoes_per_hour * (hours_per_day * full_days + friday_hours)

theorem cobbler_weekly_shoes : total_shoes = 105 := by
  sorry

end NUMINAMATH_CALUDE_cobbler_weekly_shoes_l3743_374312


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3743_374301

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem tenth_term_of_sequence
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 2)
  (h3 : a 1 = 1) :
  a 10 = 19 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3743_374301


namespace NUMINAMATH_CALUDE_even_iff_mod_two_eq_zero_l3743_374351

theorem even_iff_mod_two_eq_zero (x : Int) : Even x ↔ x % 2 = 0 := by sorry

end NUMINAMATH_CALUDE_even_iff_mod_two_eq_zero_l3743_374351


namespace NUMINAMATH_CALUDE_function_range_l3743_374336

theorem function_range (x : ℝ) :
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, x^2 + (a - 4)*x + 4 - 2*a > 0) →
  x < 1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l3743_374336


namespace NUMINAMATH_CALUDE_d_72_eq_22_l3743_374362

/-- D(n) is the number of ways to write n as an ordered product of integers greater than 1 -/
def D (n : ℕ) : ℕ := sorry

/-- The main theorem: D(72) = 22 -/
theorem d_72_eq_22 : D 72 = 22 := by sorry

end NUMINAMATH_CALUDE_d_72_eq_22_l3743_374362


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3743_374308

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 16 * Real.sqrt 13 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l3743_374308


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3743_374359

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 20 → c^2 = a^2 + b^2 → c = 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3743_374359


namespace NUMINAMATH_CALUDE_equation_solution_l3743_374377

theorem equation_solution : ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3743_374377


namespace NUMINAMATH_CALUDE_tuesday_equals_friday_l3743_374356

def total_weekly_time : ℝ := 5
def monday_time : ℝ := 1.5
def wednesday_time : ℝ := 1.5
def friday_time : ℝ := 1

def tuesday_time : ℝ := total_weekly_time - (monday_time + wednesday_time + friday_time)

theorem tuesday_equals_friday : tuesday_time = friday_time := by
  sorry

end NUMINAMATH_CALUDE_tuesday_equals_friday_l3743_374356


namespace NUMINAMATH_CALUDE_second_number_solution_l3743_374389

theorem second_number_solution (x : ℝ) :
  12.1212 + x - 9.1103 = 20.011399999999995 →
  x = 18.000499999999995 := by
sorry

end NUMINAMATH_CALUDE_second_number_solution_l3743_374389


namespace NUMINAMATH_CALUDE_function_properties_l3743_374393

theorem function_properties (f : ℝ → ℝ) (h1 : f (-2) > f (-1)) (h2 : f (-1) < f 0) :
  ¬ (
    (∀ x y, -2 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≥ f y) ∧
    (∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≤ f y)
  ) ∧
  ¬ (
    (∀ x y, -2 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≤ f y) ∧
    (∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≥ f y)
  ) ∧
  ¬ (∀ x, -2 ≤ x ∧ x ≤ 0 → f x ≥ f (-1)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3743_374393


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l3743_374306

-- Define the conditions
def condition_p (x : ℝ) : Prop := -1 < x ∧ x < 3
def condition_q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- Theorem statement
theorem condition_p_sufficient_not_necessary :
  (∀ x, condition_p x → condition_q x) ∧
  (∃ x, condition_q x ∧ ¬condition_p x) :=
sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l3743_374306


namespace NUMINAMATH_CALUDE_smallest_number_l3743_374318

theorem smallest_number (A B C : ℤ) : 
  A = 18 + 38 →
  B = A - 26 →
  C = B / 3 →
  C < A ∧ C < B :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3743_374318


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3743_374311

theorem imaginary_part_of_z (z : ℂ) (h : z + 3 - 4*I = 1) : z.im = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3743_374311


namespace NUMINAMATH_CALUDE_triangle_area_l3743_374369

/-- Given a triangle with perimeter 20 cm and inradius 2.5 cm, its area is 25 cm². -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
    (h1 : perimeter = 20) 
    (h2 : inradius = 2.5) 
    (h3 : area = inradius * (perimeter / 2)) : 
  area = 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3743_374369


namespace NUMINAMATH_CALUDE_difference_of_squares_divided_l3743_374307

theorem difference_of_squares_divided : (113^2 - 107^2) / 6 = 220 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_divided_l3743_374307


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l3743_374337

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (((x^2 + y^2 + z^2) * (4*x^2 + 2*y^2 + 3*z^2)).sqrt) / (x*y*z) ≥ 2 + Real.sqrt 2 + Real.sqrt 3 :=
sorry

theorem lower_bound_achievable :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (((x^2 + y^2 + z^2) * (4*x^2 + 2*y^2 + 3*z^2)).sqrt) / (x*y*z) = 2 + Real.sqrt 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l3743_374337


namespace NUMINAMATH_CALUDE_games_within_division_is_48_l3743_374354

/-- Represents a baseball league with two divisions -/
structure BaseballLeague where
  /-- Number of games played against each team in the same division -/
  N : ℕ
  /-- Number of games played against each team in the other division -/
  M : ℕ
  /-- N is greater than 2M -/
  h1 : N > 2 * M
  /-- M is greater than 4 -/
  h2 : M > 4
  /-- Total number of games in the schedule is 76 -/
  h3 : 3 * N + 4 * M = 76

/-- The number of games a team plays within its own division -/
def gamesWithinDivision (league : BaseballLeague) : ℕ := 3 * league.N

/-- Theorem stating that the number of games within division is 48 -/
theorem games_within_division_is_48 (league : BaseballLeague) :
  gamesWithinDivision league = 48 := by
  sorry


end NUMINAMATH_CALUDE_games_within_division_is_48_l3743_374354


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3743_374376

theorem fraction_equation_solution (n : ℚ) : 
  (2 / (n + 1) + 3 / (n + 1) + n / (n + 1) = 4) → n = 1/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3743_374376


namespace NUMINAMATH_CALUDE_paul_school_supplies_l3743_374333

/-- Given Paul's initial crayons and erasers, and the number of crayons left,
    prove the difference between erasers and crayons left is 70. -/
theorem paul_school_supplies (initial_crayons : ℕ) (initial_erasers : ℕ) (crayons_left : ℕ)
    (h1 : initial_crayons = 601)
    (h2 : initial_erasers = 406)
    (h3 : crayons_left = 336) :
    initial_erasers - crayons_left = 70 := by
  sorry

end NUMINAMATH_CALUDE_paul_school_supplies_l3743_374333


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l3743_374355

theorem intersection_point_of_lines (x y : ℝ) :
  (x - 2*y + 7 = 0) ∧ (2*x + y - 1 = 0) ↔ (x = -1 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l3743_374355


namespace NUMINAMATH_CALUDE_subcommittee_formation_ways_senate_subcommittee_formation_l3743_374367

theorem subcommittee_formation_ways (total_republicans : Nat) (total_democrats : Nat) 
  (subcommittee_republicans : Nat) (subcommittee_democrats : Nat) : Nat :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem senate_subcommittee_formation : 
  subcommittee_formation_ways 10 8 4 3 = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_ways_senate_subcommittee_formation_l3743_374367


namespace NUMINAMATH_CALUDE_triangle_max_area_l3743_374382

/-- Given a triangle ABC with sides a, b, c, where S = a² - (b-c)² and b + c = 8,
    the maximum value of S is 64/17 -/
theorem triangle_max_area (a b c : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
    (h2 : b + c = 8) (h3 : ∀ S : ℝ, S = a^2 - (b-c)^2) : 
    ∃ (S : ℝ), S ≤ 64/17 ∧ ∃ (a' b' c' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ b' + c' = 8 ∧ 
    64/17 = a'^2 - (b'-c')^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3743_374382


namespace NUMINAMATH_CALUDE_percent_relation_l3743_374331

theorem percent_relation (x y : ℝ) (h : 0.7 * (x - y) = 0.3 * (x + y)) : y / x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3743_374331


namespace NUMINAMATH_CALUDE_expression_value_at_three_l3743_374347

theorem expression_value_at_three :
  let x : ℝ := 3
  let expr := (Real.sqrt (x - 2 * Real.sqrt 2) / Real.sqrt (x^2 - 4 * Real.sqrt 2 * x + 8)) -
              (Real.sqrt (x + 2 * Real.sqrt 2) / Real.sqrt (x^2 + 4 * Real.sqrt 2 * x + 8))
  expr = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l3743_374347


namespace NUMINAMATH_CALUDE_largest_coin_distribution_l3743_374394

theorem largest_coin_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 12 * k + 3) ∧ 
  n < 100 ∧ 
  (∀ m : ℕ, (∃ j : ℕ, m = 12 * j + 3) → m < 100 → m ≤ n) → 
  n = 99 := by
sorry

end NUMINAMATH_CALUDE_largest_coin_distribution_l3743_374394


namespace NUMINAMATH_CALUDE_square_floor_theorem_l3743_374345

/-- Represents a square floor tiled with congruent square tiles. -/
structure SquareFloor :=
  (side_length : ℕ)

/-- Calculates the number of black tiles on the diagonals of a square floor. -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- Calculates the total number of tiles on a square floor. -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem stating that a square floor with 101 black diagonal tiles has 2601 total tiles. -/
theorem square_floor_theorem (floor : SquareFloor) :
  black_tiles floor = 101 → total_tiles floor = 2601 :=
by sorry

end NUMINAMATH_CALUDE_square_floor_theorem_l3743_374345


namespace NUMINAMATH_CALUDE_parabola_and_intersection_properties_l3743_374386

/-- Parabola C with directrix x = -1/4 -/
def ParabolaC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = p.1}

/-- Line l passing through P(t, 0) -/
def LineL (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ m : ℝ, p.1 = m * p.2 + t}

/-- Points A and B are the intersections of ParabolaC and LineL -/
def IntersectionPoints (t : ℝ) : Set (ℝ × ℝ) :=
  ParabolaC ∩ LineL t

/-- Circle with diameter AB passes through the origin -/
def CircleThroughOrigin (t : ℝ) : Prop :=
  ∀ A B : ℝ × ℝ, A ∈ IntersectionPoints t → B ∈ IntersectionPoints t →
    A.1 * B.1 + A.2 * B.2 = 0

theorem parabola_and_intersection_properties :
  (∀ p : ℝ × ℝ, p ∈ ParabolaC ↔ p.2^2 = p.1) ∧
  (∀ t : ℝ, CircleThroughOrigin t → t = 0 ∨ t = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_intersection_properties_l3743_374386


namespace NUMINAMATH_CALUDE_atlantic_charge_proof_l3743_374353

/-- United Telephone's base rate in dollars -/
def united_base : ℚ := 8

/-- United Telephone's per minute charge in dollars -/
def united_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℚ := 12

/-- Number of minutes at which the bills are equal -/
def equal_minutes : ℚ := 80

/-- Atlantic Call's per minute charge in dollars -/
def atlantic_per_minute : ℚ := 1/5

theorem atlantic_charge_proof :
  united_base + united_per_minute * equal_minutes =
  atlantic_base + atlantic_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_atlantic_charge_proof_l3743_374353


namespace NUMINAMATH_CALUDE_sequence_120th_term_l3743_374327

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth term of the sequence of positive integers 
    whose digits sum to 10, arranged in ascending order -/
def sequence_term (n : ℕ) : ℕ := sorry

/-- The main theorem: The 120th term of the sequence is 2017 -/
theorem sequence_120th_term : sequence_term 120 = 2017 := by sorry

end NUMINAMATH_CALUDE_sequence_120th_term_l3743_374327


namespace NUMINAMATH_CALUDE_probability_is_one_fourteenth_l3743_374342

/-- Represents a cube with side length 4 and two adjacent painted faces -/
structure PaintedCube :=
  (side_length : ℕ)
  (total_cubes : ℕ)
  (two_face_cubes : ℕ)
  (no_face_cubes : ℕ)

/-- The probability of selecting one cube with two painted faces and one with no painted faces -/
def select_probability (c : PaintedCube) : ℚ :=
  (c.two_face_cubes * c.no_face_cubes) / (c.total_cubes.choose 2)

/-- The theorem stating the probability is 1/14 -/
theorem probability_is_one_fourteenth (c : PaintedCube) 
  (h1 : c.side_length = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.two_face_cubes = 4)
  (h4 : c.no_face_cubes = 36) :
  select_probability c = 1 / 14 := by
  sorry

#eval select_probability { side_length := 4, total_cubes := 64, two_face_cubes := 4, no_face_cubes := 36 }

end NUMINAMATH_CALUDE_probability_is_one_fourteenth_l3743_374342


namespace NUMINAMATH_CALUDE_binomial_prob_one_third_l3743_374374

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_prob_one_third 
  (X : BinomialRV) 
  (h_expect : expectation X = 30)
  (h_var : variance X = 20) : 
  X.p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_binomial_prob_one_third_l3743_374374


namespace NUMINAMATH_CALUDE_gcd_problem_l3743_374339

theorem gcd_problem (A B : ℕ) (h1 : Nat.lcm A B = 180) (h2 : ∃ (k : ℕ), A = 4 * k ∧ B = 5 * k) : 
  Nat.gcd A B = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3743_374339


namespace NUMINAMATH_CALUDE_expression_evaluation_l3743_374388

theorem expression_evaluation : 12 - 10 + 8 * 7 + 6 - 5 * 4 + 3 / 3 - 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3743_374388


namespace NUMINAMATH_CALUDE_catia_speed_theorem_l3743_374320

/-- The speed at which Cátia should travel to reach home at 5:00 PM -/
def required_speed : ℝ := 12

/-- The time Cátia leaves school every day -/
def departure_time : ℝ := 3.75 -- 3:45 PM in decimal hours

/-- The distance from school to Cátia's home -/
def distance : ℝ := 15

/-- Arrival time when traveling at 20 km/h -/
def arrival_time_fast : ℝ := 4.5 -- 4:30 PM in decimal hours

/-- Arrival time when traveling at 10 km/h -/
def arrival_time_slow : ℝ := 5.25 -- 5:15 PM in decimal hours

/-- The desired arrival time -/
def desired_arrival_time : ℝ := 5 -- 5:00 PM in decimal hours

theorem catia_speed_theorem :
  (distance / (arrival_time_fast - departure_time) = 20) →
  (distance / (arrival_time_slow - departure_time) = 10) →
  (distance / (desired_arrival_time - departure_time) = required_speed) :=
by sorry

end NUMINAMATH_CALUDE_catia_speed_theorem_l3743_374320


namespace NUMINAMATH_CALUDE_minimum_peanuts_l3743_374379

theorem minimum_peanuts (N A₁ A₂ A₃ A₄ A₅ : ℕ) : 
  N = 5 * A₁ + 1 ∧
  4 * A₁ = 5 * A₂ + 1 ∧
  4 * A₂ = 5 * A₃ + 1 ∧
  4 * A₃ = 5 * A₄ + 1 ∧
  4 * A₄ = 5 * A₅ + 1 →
  N ≥ 3121 ∧ (N = 3121 → 
    A₁ = 624 ∧ A₂ = 499 ∧ A₃ = 399 ∧ A₄ = 319 ∧ A₅ = 255) :=
by sorry

#check minimum_peanuts

end NUMINAMATH_CALUDE_minimum_peanuts_l3743_374379


namespace NUMINAMATH_CALUDE_intersection_possibilities_l3743_374358

-- Define the sets P and Q
variable (P Q : Set ℕ)

-- Define the function f
def f (t : ℕ) : ℕ := t^2

-- State the theorem
theorem intersection_possibilities (h1 : Q = {1, 4}) 
  (h2 : ∀ t ∈ P, f t ∈ Q) : 
  P ∩ Q = {1} ∨ P ∩ Q = ∅ := by
sorry

end NUMINAMATH_CALUDE_intersection_possibilities_l3743_374358


namespace NUMINAMATH_CALUDE_radical_simplification_l3743_374392

-- Define the statement
theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (11 * q) * Real.sqrt (8 * q^3) * Real.sqrt (9 * q^5) = 28 * q^4 * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l3743_374392


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l3743_374300

theorem factorial_ratio_equals_sixty_sevenths :
  (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l3743_374300


namespace NUMINAMATH_CALUDE_max_lambda_inequality_l3743_374338

theorem max_lambda_inequality (a b x y : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0)
  (h_sum : a + b = 27) :
  (a * x^2 + b * y^2 + 4 * x * y)^3 ≥ 2916 * (a * x^2 * y + b * x * y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_max_lambda_inequality_l3743_374338


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3743_374372

/-- Theorem: When each edge of a cube is increased by p%, 
    the surface area of the cube is increased by 2p + (p^2/100)%. -/
theorem cube_surface_area_increase (p : ℝ) :
  let original_edge : ℝ → ℝ := λ s => s
  let increased_edge : ℝ → ℝ := λ s => s * (1 + p / 100)
  let original_surface_area : ℝ → ℝ := λ s => 6 * s^2
  let increased_surface_area : ℝ → ℝ := λ s => 6 * (increased_edge s)^2
  let percent_increase : ℝ → ℝ := λ s => 
    (increased_surface_area s - original_surface_area s) / original_surface_area s * 100
  ∀ s > 0, percent_increase s = 2 * p + p^2 / 100 :=
by sorry


end NUMINAMATH_CALUDE_cube_surface_area_increase_l3743_374372


namespace NUMINAMATH_CALUDE_min_value_of_S_l3743_374370

theorem min_value_of_S (x y : ℝ) : 2 * x^2 - x*y + y^2 + 2*x + 3*y ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_S_l3743_374370


namespace NUMINAMATH_CALUDE_odd_square_minus_one_divisible_by_24_l3743_374305

theorem odd_square_minus_one_divisible_by_24 (n : ℤ) : 
  Odd (n^2) → (n^2 % 9 ≠ 0) → (n^2 - 1) % 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_divisible_by_24_l3743_374305


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_inequality_l3743_374321

theorem positive_integer_solutions_of_inequality :
  {x : ℕ+ | (3 : ℝ) * x.val < x.val + 3} = {1} := by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_inequality_l3743_374321


namespace NUMINAMATH_CALUDE_pizza_and_toppings_count_l3743_374360

/-- Calculates the total number of pieces of pizza and toppings carried by fourth-graders --/
theorem pizza_and_toppings_count : 
  let pieces_per_pizza : ℕ := 6
  let num_fourth_graders : ℕ := 10
  let pizzas_per_child : ℕ := 20
  let pepperoni_per_pizza : ℕ := 5
  let mushrooms_per_pizza : ℕ := 3
  let olives_per_pizza : ℕ := 8

  let total_pizzas : ℕ := num_fourth_graders * pizzas_per_child
  let total_pieces : ℕ := total_pizzas * pieces_per_pizza
  let total_pepperoni : ℕ := total_pizzas * pepperoni_per_pizza
  let total_mushrooms : ℕ := total_pizzas * mushrooms_per_pizza
  let total_olives : ℕ := total_pizzas * olives_per_pizza
  let total_toppings : ℕ := total_pepperoni + total_mushrooms + total_olives

  total_pieces + total_toppings = 4400 := by
  sorry

end NUMINAMATH_CALUDE_pizza_and_toppings_count_l3743_374360


namespace NUMINAMATH_CALUDE_percentage_less_than_50k_l3743_374364

/-- Represents the percentage of counties in each population category -/
structure PopulationDistribution :=
  (less_than_50k : ℝ)
  (between_50k_and_150k : ℝ)
  (more_than_150k : ℝ)

/-- The given population distribution from the pie chart -/
def given_distribution : PopulationDistribution :=
  { less_than_50k := 35,
    between_50k_and_150k := 40,
    more_than_150k := 25 }

/-- Theorem stating that the percentage of counties with fewer than 50,000 residents is 35% -/
theorem percentage_less_than_50k (dist : PopulationDistribution) 
  (h1 : dist = given_distribution) : 
  dist.less_than_50k = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_less_than_50k_l3743_374364


namespace NUMINAMATH_CALUDE_sampling_probabilities_l3743_374363

/-- Simple random sampling without replacement -/
def SimpleRandomSampling (population_size : ℕ) (sample_size : ℕ) : Prop :=
  sample_size ≤ population_size

/-- Probability of selecting a specific individual on the first draw -/
def ProbFirstDraw (population_size : ℕ) : ℚ :=
  1 / population_size

/-- Probability of selecting a specific individual on the second draw -/
def ProbSecondDraw (population_size : ℕ) : ℚ :=
  (population_size - 1) / population_size * (1 / (population_size - 1))

/-- Theorem stating the probabilities for the given scenario -/
theorem sampling_probabilities 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 10) 
  (h2 : sample_size = 3) 
  (h3 : SimpleRandomSampling population_size sample_size) :
  ProbFirstDraw population_size = 1/10 ∧ 
  ProbSecondDraw population_size = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_sampling_probabilities_l3743_374363


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l3743_374373

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 3 (1/3) + 2 →
  x^3 - 6*x^2 + 12*x - 11 = 0 := by sorry

#check cubic_polynomial_root

end NUMINAMATH_CALUDE_cubic_polynomial_root_l3743_374373


namespace NUMINAMATH_CALUDE_forty_people_skating_wheels_l3743_374349

/-- The number of wheels on the floor when a given number of people are roller skating. -/
def wheels_on_floor (people : ℕ) : ℕ :=
  people * 2 * 4

/-- Theorem: When 40 people are roller skating, there are 320 wheels on the floor. -/
theorem forty_people_skating_wheels : wheels_on_floor 40 = 320 := by
  sorry

#eval wheels_on_floor 40

end NUMINAMATH_CALUDE_forty_people_skating_wheels_l3743_374349


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3743_374380

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ)^((a + b)/2) = Real.sqrt 3 → 
  (1/a + 1/b) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3743_374380


namespace NUMINAMATH_CALUDE_foreign_trade_income_2007_2009_l3743_374328

/-- Represents the foreign trade income equation given the initial value,
    final value, and growth rate over a two-year period. -/
def foreign_trade_equation (initial : ℝ) (final : ℝ) (rate : ℝ) : Prop :=
  initial * (1 + rate)^2 = final

/-- Theorem stating that the foreign trade income equation holds for the given values. -/
theorem foreign_trade_income_2007_2009 :
  foreign_trade_equation 2.5 3.6 x = true :=
sorry

end NUMINAMATH_CALUDE_foreign_trade_income_2007_2009_l3743_374328
