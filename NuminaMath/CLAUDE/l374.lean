import Mathlib

namespace NUMINAMATH_CALUDE_jan_extra_distance_l374_37438

/-- Represents the driving scenario of Ian, Han, and Jan -/
structure DrivingScenario where
  ian_time : ℝ
  ian_speed : ℝ
  han_time : ℝ
  han_speed : ℝ
  jan_time : ℝ
  jan_speed : ℝ
  han_extra_distance : ℝ

/-- The conditions of the driving scenario -/
def scenario_conditions (s : DrivingScenario) : Prop :=
  s.han_time = s.ian_time + 2 ∧
  s.han_speed = s.ian_speed + 10 ∧
  s.jan_time = s.ian_time + 3 ∧
  s.jan_speed = s.ian_speed + 15 ∧
  s.han_extra_distance = 120

/-- The theorem stating that Jan drove 195 miles more than Ian -/
theorem jan_extra_distance (s : DrivingScenario) 
  (h : scenario_conditions s) : 
  s.jan_speed * s.jan_time - s.ian_speed * s.ian_time = 195 :=
sorry


end NUMINAMATH_CALUDE_jan_extra_distance_l374_37438


namespace NUMINAMATH_CALUDE_binomial_probability_one_l374_37483

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The probability of a binomial random variable taking a specific value -/
def probability (X : BinomialRV) (k : ℕ) : ℝ :=
  (X.n.choose k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

theorem binomial_probability_one (η : BinomialRV) 
  (h_p : η.p = 0.6) 
  (h_expectation : expectation η = 3) :
  probability η 1 = 3 * 0.4^4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_one_l374_37483


namespace NUMINAMATH_CALUDE_three_cubes_volume_l374_37414

theorem three_cubes_volume (s₁ s₂ : ℝ) (h₁ : s₁ > 0) (h₂ : s₂ > 0) : 
  6 * (s₁ + s₂)^2 = 864 → 2 * s₁^3 + s₂^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_three_cubes_volume_l374_37414


namespace NUMINAMATH_CALUDE_twelfth_nine_position_l374_37477

/-- The position of the nth occurrence of a digit in the sequence of natural numbers written without spaces -/
def digitPosition (n : ℕ) (digit : ℕ) : ℕ :=
  sorry

/-- The sequence of natural numbers written without spaces -/
def naturalNumberSequence : List ℕ :=
  sorry

theorem twelfth_nine_position :
  digitPosition 12 9 = 174 :=
sorry

end NUMINAMATH_CALUDE_twelfth_nine_position_l374_37477


namespace NUMINAMATH_CALUDE_binomial_7_4_l374_37401

theorem binomial_7_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_4_l374_37401


namespace NUMINAMATH_CALUDE_special_triangle_rs_distance_l374_37449

/-- Triangle ABC with altitude CH and inscribed circles in ACH and BCH -/
structure SpecialTriangle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  -- CH is an altitude
  altitude : (C.1 - H.1) * (B.1 - A.1) + (C.2 - H.2) * (B.2 - A.2) = 0
  -- R is on CH
  r_on_ch : ∃ t : ℝ, R = (C.1 + t * (H.1 - C.1), C.2 + t * (H.2 - C.2))
  -- S is on CH
  s_on_ch : ∃ t : ℝ, S = (C.1 + t * (H.1 - C.1), C.2 + t * (H.2 - C.2))
  -- Side lengths
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2000
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 1997
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 1998

/-- The distance between R and S is 2001/4000 -/
theorem special_triangle_rs_distance (t : SpecialTriangle) :
  Real.sqrt ((t.R.1 - t.S.1)^2 + (t.R.2 - t.S.2)^2) = 2001 / 4000 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_rs_distance_l374_37449


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l374_37427

/-- Given two circles A and B, where A is inside B, this theorem proves the diameter of A
    given the diameter of B, the distance between centers, and the ratio of areas. -/
theorem circle_diameter_ratio (dB : ℝ) (d : ℝ) (r : ℝ) : 
  dB = 20 →  -- Diameter of circle B
  d = 4 →    -- Distance between centers
  r = 5 →    -- Ratio of shaded area to area of circle A
  ∃ (dA : ℝ), dA = 2 * Real.sqrt (50 / 3) ∧ 
    (π * (dA / 2)^2) * (1 + r) = π * (dB / 2)^2 ∧ 
    d ≤ (dB - dA) / 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l374_37427


namespace NUMINAMATH_CALUDE_complex_repair_charge_is_300_l374_37453

/-- Represents the financial details of Jim's bike shop --/
structure BikeShop where
  tire_repair_charge : ℕ
  tire_repair_cost : ℕ
  tire_repairs_count : ℕ
  complex_repairs_count : ℕ
  complex_repair_cost : ℕ
  retail_profit : ℕ
  fixed_expenses : ℕ
  total_profit : ℕ

/-- Calculates the charge for a complex repair --/
def complex_repair_charge (shop : BikeShop) : ℕ :=
  let tire_repair_profit := shop.tire_repair_charge - shop.tire_repair_cost
  let total_tire_profit := tire_repair_profit * shop.tire_repairs_count
  let profit_before_complex := total_tire_profit + shop.retail_profit - shop.fixed_expenses
  let complex_repairs_profit := shop.total_profit - profit_before_complex
  let single_complex_repair_profit := complex_repairs_profit / shop.complex_repairs_count
  single_complex_repair_profit + shop.complex_repair_cost

/-- Theorem stating that the complex repair charge is $300 --/
theorem complex_repair_charge_is_300 (shop : BikeShop) 
    (h1 : shop.tire_repair_charge = 20)
    (h2 : shop.tire_repair_cost = 5)
    (h3 : shop.tire_repairs_count = 300)
    (h4 : shop.complex_repairs_count = 2)
    (h5 : shop.complex_repair_cost = 50)
    (h6 : shop.retail_profit = 2000)
    (h7 : shop.fixed_expenses = 4000)
    (h8 : shop.total_profit = 3000) :
    complex_repair_charge shop = 300 := by
  sorry

end NUMINAMATH_CALUDE_complex_repair_charge_is_300_l374_37453


namespace NUMINAMATH_CALUDE_zayne_bracelet_count_l374_37416

/-- Calculates the number of bracelets Zayne started with given the sales conditions -/
def bracelets_count (single_price : ℕ) (pair_price : ℕ) (single_sales : ℕ) (total_revenue : ℕ) : ℕ :=
  let single_revenue := single_price * single_sales
  let pair_revenue := total_revenue - single_revenue
  let pair_sales := pair_revenue / pair_price
  single_sales + 2 * pair_sales

/-- Theorem stating that Zayne started with 30 bracelets given the sales conditions -/
theorem zayne_bracelet_count :
  bracelets_count 5 8 (60 / 5) 132 = 30 := by
  sorry

end NUMINAMATH_CALUDE_zayne_bracelet_count_l374_37416


namespace NUMINAMATH_CALUDE_discount_comparison_l374_37484

/-- The original bill amount in dollars -/
def original_bill : ℝ := 8000

/-- The single discount rate -/
def single_discount_rate : ℝ := 0.3

/-- The first successive discount rate -/
def first_successive_discount_rate : ℝ := 0.2

/-- The second successive discount rate -/
def second_successive_discount_rate : ℝ := 0.1

/-- The difference between the two discount scenarios -/
def discount_difference : ℝ := 160

theorem discount_comparison :
  let single_discounted := original_bill * (1 - single_discount_rate)
  let successive_discounted := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)
  successive_discounted - single_discounted = discount_difference := by
  sorry

end NUMINAMATH_CALUDE_discount_comparison_l374_37484


namespace NUMINAMATH_CALUDE_no_real_solution_l374_37464

theorem no_real_solution :
  ¬∃ (x : ℝ), 4 * (3 * x)^2 + (3 * x) + 3 = 2 * (9 * x^2 + (3 * x) + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l374_37464


namespace NUMINAMATH_CALUDE_sphere_surface_area_l374_37498

theorem sphere_surface_area (r₁ r₂ d R : ℝ) : 
  r₁ > 0 → r₂ > 0 → d > 0 → R > 0 →
  r₁^2 * π = 9 * π →
  r₂^2 * π = 16 * π →
  d = 1 →
  R^2 = r₂^2 + (R - d)^2 →
  R^2 = r₁^2 + R^2 →
  4 * π * R^2 = 100 * π := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l374_37498


namespace NUMINAMATH_CALUDE_car_speed_problem_l374_37431

/-- Proves that the speed at which a car travels 1 kilometer in 12 seconds less time
    than it does at 60 km/h is 50 km/h. -/
theorem car_speed_problem (v : ℝ) : 
  (1000 / (60000 / 3600) - 1000 / (v / 3600) = 12) → v = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l374_37431


namespace NUMINAMATH_CALUDE_linear_function_k_value_l374_37480

/-- Given a linear function y = kx + 3 passing through the point (2, 5), prove that k = 1 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3) → 
  (5 : ℝ) = k * 2 + 3 → 
  k = 1 := by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l374_37480


namespace NUMINAMATH_CALUDE_problem_statement_l374_37472

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a / (1 + a) + b / (1 + b) = 1) : 
  a / (1 + b^2) - b / (1 + a^2) = a - b := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l374_37472


namespace NUMINAMATH_CALUDE_fir_trees_count_l374_37413

/-- Represents the statements made by each child -/
inductive Statement
| anya : Statement
| borya : Statement
| vera : Statement
| gena : Statement

/-- Represents the gender of each child -/
inductive Gender
| boy : Gender
| girl : Gender

/-- Checks if a statement is true given the number of trees -/
def isTrue (s : Statement) (n : Nat) : Prop :=
  match s with
  | .anya => n = 15
  | .borya => n % 11 = 0
  | .vera => n < 25
  | .gena => n % 22 = 0

/-- Assigns a gender to each child -/
def gender (s : Statement) : Gender :=
  match s with
  | .anya => .girl
  | .borya => .boy
  | .vera => .girl
  | .gena => .boy

/-- The main theorem to prove -/
theorem fir_trees_count : 
  ∃ (n : Nat), n = 11 ∧ 
  ∃ (s1 s2 : Statement), s1 ≠ s2 ∧ 
  gender s1 ≠ gender s2 ∧
  isTrue s1 n ∧ isTrue s2 n ∧
  ∀ (s : Statement), s ≠ s1 ∧ s ≠ s2 → ¬(isTrue s n) :=
by sorry

end NUMINAMATH_CALUDE_fir_trees_count_l374_37413


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l374_37423

def polynomial (x : ℤ) : ℤ := x^3 - 4*x^2 - 14*x + 24

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = {-4, -3, 3} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l374_37423


namespace NUMINAMATH_CALUDE_target_number_scientific_notation_l374_37439

/-- The number we want to express in scientific notation -/
def target_number : ℕ := 1200000000

/-- Definition of scientific notation for positive integers -/
def scientific_notation (n : ℕ) (a : ℚ) (b : ℤ) : Prop :=
  (1 ≤ a) ∧ (a < 10) ∧ (n = (a * 10^b).floor)

/-- Theorem stating that 1,200,000,000 is equal to 1.2 × 10^9 in scientific notation -/
theorem target_number_scientific_notation :
  scientific_notation target_number (12/10) 9 := by
  sorry

end NUMINAMATH_CALUDE_target_number_scientific_notation_l374_37439


namespace NUMINAMATH_CALUDE_pentagon_area_fraction_is_five_eighths_l374_37491

/-- Represents the tiling pattern of a large square -/
structure TilingPattern where
  total_divisions : Nat
  pentagon_count : Nat
  square_count : Nat

/-- Calculates the fraction of area covered by pentagons in the tiling pattern -/
def pentagon_area_fraction (pattern : TilingPattern) : Rat :=
  pattern.pentagon_count / pattern.total_divisions

/-- Theorem stating that the fraction of area covered by pentagons is 5/8 -/
theorem pentagon_area_fraction_is_five_eighths (pattern : TilingPattern) :
  pattern.total_divisions = 16 →
  pattern.pentagon_count = 10 →
  pattern.square_count = 6 →
  pentagon_area_fraction pattern = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_fraction_is_five_eighths_l374_37491


namespace NUMINAMATH_CALUDE_empty_set_subset_of_all_l374_37404

theorem empty_set_subset_of_all (A : Set α) : ∅ ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_empty_set_subset_of_all_l374_37404


namespace NUMINAMATH_CALUDE_water_in_pool_is_34_l374_37459

/-- Calculates the amount of water in Carol's pool after five hours of filling and leaking -/
def water_in_pool : ℕ :=
  let first_hour : ℕ := 8
  let next_two_hours : ℕ := 10 * 2
  let fourth_hour : ℕ := 14
  let leak : ℕ := 8
  (first_hour + next_two_hours + fourth_hour) - leak

/-- Theorem stating that the amount of water in the pool after five hours is 34 gallons -/
theorem water_in_pool_is_34 : water_in_pool = 34 := by
  sorry

end NUMINAMATH_CALUDE_water_in_pool_is_34_l374_37459


namespace NUMINAMATH_CALUDE_inscribed_square_area_l374_37455

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A square inscribed in the region bound by the parabola and the x-axis -/
structure InscribedSquare where
  center : ℝ
  side_half : ℝ
  lower_left_on_axis : f (center - side_half) = 0
  upper_right_on_parabola : f (center + side_half) = 2 * side_half

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area (s : InscribedSquare) : 
  (2 * s.side_half)^2 = 24 - 16 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l374_37455


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l374_37458

theorem circle_area_from_circumference : ∀ (r : ℝ), 
  (2 * π * r = 18 * π) → (π * r^2 = 81 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l374_37458


namespace NUMINAMATH_CALUDE_rabbit_distribution_theorem_l374_37429

/-- Represents the number of pet stores --/
def num_stores : ℕ := 5

/-- Represents the number of parent rabbits --/
def num_parents : ℕ := 2

/-- Represents the number of offspring rabbits --/
def num_offspring : ℕ := 4

/-- Represents the total number of rabbits --/
def total_rabbits : ℕ := num_parents + num_offspring

/-- 
Represents the number of ways to distribute rabbits to pet stores 
such that no store gets both a parent and a child
--/
def distribution_ways : ℕ := sorry

theorem rabbit_distribution_theorem : 
  distribution_ways = 560 := by sorry

end NUMINAMATH_CALUDE_rabbit_distribution_theorem_l374_37429


namespace NUMINAMATH_CALUDE_circle_range_theorem_l374_37471

/-- The range of 'a' for a circle (x-a)^2 + (y-a)^2 = 8 with a point at distance √2 from origin -/
theorem circle_range_theorem (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 8 ∧ x^2 + y^2 = 2) ↔ 
  (a ∈ Set.Icc (-3) (-1) ∪ Set.Icc 1 3) :=
sorry

end NUMINAMATH_CALUDE_circle_range_theorem_l374_37471


namespace NUMINAMATH_CALUDE_raduzhny_residents_l374_37417

/-- The number of villages in Solar Valley -/
def num_villages : ℕ := 10

/-- The population of Znoynoe village -/
def znoynoe_population : ℕ := 1000

/-- The amount by which Znoynoe's population exceeds the average -/
def excess_population : ℕ := 90

/-- The total population of all villages in Solar Valley -/
def total_population : ℕ := znoynoe_population + (num_villages - 1) * (znoynoe_population - excess_population)

/-- The population of Raduzhny village -/
def raduzhny_population : ℕ := znoynoe_population - excess_population

theorem raduzhny_residents : raduzhny_population = 900 := by
  sorry

end NUMINAMATH_CALUDE_raduzhny_residents_l374_37417


namespace NUMINAMATH_CALUDE_quadratic_minimum_l374_37475

theorem quadratic_minimum (x : ℝ) : x^2 + 10*x ≥ -25 ∧ ∃ y : ℝ, y^2 + 10*y = -25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l374_37475


namespace NUMINAMATH_CALUDE_triangle_ratio_equals_two_l374_37492

noncomputable def triangle_ratio (A B C : ℝ) (a b c : ℝ) : ℝ :=
  (a + b - c) / (Real.sin A + Real.sin B - Real.sin C)

theorem triangle_ratio_equals_two (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = π / 3)  -- 60° in radians
  (h2 : a = Real.sqrt 3) :
  triangle_ratio A B C a b c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_equals_two_l374_37492


namespace NUMINAMATH_CALUDE_hyperbola_center_correct_l374_37479

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * y + 8)^2 / 16^2 - (5 * x - 15)^2 / 9^2 = 1

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, -2)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem hyperbola_center_correct :
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    hyperbola_equation (x - hyperbola_center.1) (y - hyperbola_center.2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_correct_l374_37479


namespace NUMINAMATH_CALUDE_age_problem_l374_37493

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 42 →
  b = 16 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l374_37493


namespace NUMINAMATH_CALUDE_parallelepiped_dimensions_l374_37412

theorem parallelepiped_dimensions (n : ℕ) (h1 : n > 6) : 
  (n - 2 : ℕ) > 0 ∧ (n - 4 : ℕ) > 0 ∧
  (n - 2 : ℕ) * (n - 4 : ℕ) * (n - 6 : ℕ) = 2 * n * (n - 2 : ℕ) * (n - 4 : ℕ) / 3 →
  n = 18 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_dimensions_l374_37412


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l374_37433

theorem rectangular_solid_diagonal (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l374_37433


namespace NUMINAMATH_CALUDE_real_part_of_z_l374_37489

theorem real_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2) : z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l374_37489


namespace NUMINAMATH_CALUDE_distance_to_tangent_l374_37436

/-- The curve function -/
def f (x : ℝ) : ℝ := -x^3 + 2*x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := -3*x^2 + 2

/-- The x-coordinate of the tangent point -/
def x₀ : ℝ := -1

/-- The y-coordinate of the tangent point -/
def y₀ : ℝ := f x₀

/-- The slope of the tangent line -/
def m : ℝ := f' x₀

/-- The point we're measuring the distance from -/
def P : ℝ × ℝ := (3, 2)

/-- Theorem: The distance from P to the tangent line is 7√2/2 -/
theorem distance_to_tangent :
  let A : ℝ := 1
  let B : ℝ := 1
  let C : ℝ := -(m * x₀ - y₀)
  (A * P.1 + B * P.2 + C) / Real.sqrt (A^2 + B^2) = 7 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_tangent_l374_37436


namespace NUMINAMATH_CALUDE_folded_perimeter_not_greater_l374_37419

/-- Represents a polygon in 2D space -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.length ≥ 3

/-- Represents a line in 2D space -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculates the perimeter of a polygon -/
def perimeter (p : Polygon) : ℝ := sorry

/-- Folds a polygon along a line and glues the halves together -/
def fold_and_glue (p : Polygon) (l : Line) : Polygon := sorry

/-- Theorem: The perimeter of a folded and glued polygon is not greater than the original -/
theorem folded_perimeter_not_greater (p : Polygon) (l : Line) :
  perimeter (fold_and_glue p l) ≤ perimeter p := by sorry

end NUMINAMATH_CALUDE_folded_perimeter_not_greater_l374_37419


namespace NUMINAMATH_CALUDE_packs_per_box_l374_37499

/-- Given that Jenny sold 24.0 boxes of Trefoils and 192 packs in total,
    prove that there are 8 packs in each box. -/
theorem packs_per_box (boxes : ℝ) (total_packs : ℕ) 
    (h1 : boxes = 24.0) 
    (h2 : total_packs = 192) : 
  (total_packs : ℝ) / boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_packs_per_box_l374_37499


namespace NUMINAMATH_CALUDE_set_intersection_example_l374_37410

theorem set_intersection_example : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4, 6}
  A ∩ B = {2, 4} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l374_37410


namespace NUMINAMATH_CALUDE_three_cats_meowing_l374_37451

/-- The number of meows for three cats in a given time period -/
def total_meows (cat1_freq : ℕ) (time : ℕ) : ℕ :=
  let cat2_freq := 2 * cat1_freq
  let cat3_freq := cat2_freq / 3
  (cat1_freq + cat2_freq + cat3_freq) * time

/-- Theorem stating that the total number of meows for three cats in 5 minutes is 55 -/
theorem three_cats_meowing (cat1_freq : ℕ) (h : cat1_freq = 3) :
  total_meows cat1_freq 5 = 55 := by
  sorry

#eval total_meows 3 5

end NUMINAMATH_CALUDE_three_cats_meowing_l374_37451


namespace NUMINAMATH_CALUDE_original_light_wattage_l374_37474

theorem original_light_wattage (new_wattage : ℝ) (increase_percentage : ℝ) 
  (h1 : new_wattage = 67.2)
  (h2 : increase_percentage = 0.12) : 
  ∃ (original_wattage : ℝ), 
    new_wattage = original_wattage * (1 + increase_percentage) ∧ 
    original_wattage = 60 :=
by sorry

end NUMINAMATH_CALUDE_original_light_wattage_l374_37474


namespace NUMINAMATH_CALUDE_hyperbola_equation_l374_37485

/-- Represents a hyperbola -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on the hyperbola -/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  h.equation x y

/-- Checks if a line is an asymptote of the hyperbola -/
def Hyperbola.hasAsymptote (h : Hyperbola) (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x y, h.contains x y → |y - f x| < ε ∨ |x| > δ

theorem hyperbola_equation (h : Hyperbola) :
  (h.hasAsymptote (fun x ↦ 2 * x) ∧ h.hasAsymptote (fun x ↦ -2 * x)) →
  h.contains 1 (2 * Real.sqrt 5) →
  h.equation = fun x y ↦ y^2 / 16 - x^2 / 4 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l374_37485


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_47_plus_one_l374_37487

theorem gcd_of_powers_of_47_plus_one (h : Nat.Prime 47) :
  Nat.gcd (47^6 + 1) (47^6 + 47^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_47_plus_one_l374_37487


namespace NUMINAMATH_CALUDE_kates_hair_length_l374_37450

theorem kates_hair_length :
  ∀ (kate emily logan : ℝ),
  kate = (1/2) * emily →
  emily = logan + 6 →
  logan = 20 →
  kate = 13 := by
sorry

end NUMINAMATH_CALUDE_kates_hair_length_l374_37450


namespace NUMINAMATH_CALUDE_tangent_intersection_monotonicity_intervals_m_range_l374_37405

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + m / x

theorem tangent_intersection (m : ℝ) :
  (∃ y, y = f m 1 ∧ y - f m 1 = (1 - m) * (0 - 1) ∧ y = 1) → m = 1 := by sorry

theorem monotonicity_intervals (m : ℝ) :
  (m ≤ 0 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) ∧
  (m > 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < m → f m x₁ > f m x₂) ∧
           (∀ x₁ x₂, m < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂)) := by sorry

theorem m_range (m : ℝ) :
  (∀ a b, 0 < a ∧ a < b → (f m b - f m a) / (b - a) < 1) → m ≥ 1/4 := by sorry

end NUMINAMATH_CALUDE_tangent_intersection_monotonicity_intervals_m_range_l374_37405


namespace NUMINAMATH_CALUDE_min_players_distinct_scores_l374_37408

/-- A round robin chess tournament where each player plays every other player exactly once. -/
structure Tournament (n : ℕ) where
  scores : Fin n → ℚ

/-- Property P(m) for a tournament -/
def hasPropertyP (t : Tournament n) (m : ℕ) : Prop :=
  ∀ (S : Finset (Fin n)), S.card = m →
    (∃ (w : Fin n), w ∈ S ∧ ∀ (x : Fin n), x ∈ S ∧ x ≠ w → t.scores w > t.scores x) ∧
    (∃ (l : Fin n), l ∈ S ∧ ∀ (x : Fin n), x ∈ S ∧ x ≠ l → t.scores l < t.scores x)

/-- All scores in the tournament are distinct -/
def hasDistinctScores (t : Tournament n) : Prop :=
  ∀ (i j : Fin n), i ≠ j → t.scores i ≠ t.scores j

/-- The main theorem -/
theorem min_players_distinct_scores (m : ℕ) (h : m ≥ 4) :
  (∀ (n : ℕ), n ≥ 2*m - 3 →
    ∀ (t : Tournament n), hasPropertyP t m → hasDistinctScores t) ∧
  (∃ (t : Tournament (2*m - 4)), hasPropertyP t m ∧ ¬hasDistinctScores t) :=
sorry

end NUMINAMATH_CALUDE_min_players_distinct_scores_l374_37408


namespace NUMINAMATH_CALUDE_equivalence_of_functional_equations_l374_37468

theorem equivalence_of_functional_equations (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ↔
  (∀ x y : ℝ, f (x * y + x + y) = f (x * y) + f x + f y) := by
  sorry

end NUMINAMATH_CALUDE_equivalence_of_functional_equations_l374_37468


namespace NUMINAMATH_CALUDE_sqrt_square_equals_abs_l374_37415

theorem sqrt_square_equals_abs (a : ℝ) : Real.sqrt (a^2) = |a| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_equals_abs_l374_37415


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l374_37447

theorem min_value_expression (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * Real.cos (2 * x) ≥ 2017 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * Real.cos (2 * x) = 2017 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l374_37447


namespace NUMINAMATH_CALUDE_cross_country_winning_scores_l374_37460

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  runners_per_team : ℕ
  total_runners : ℕ
  min_score : ℕ
  max_winning_score : ℕ

/-- The number of different possible winning scores in a cross country meet -/
def winning_scores (meet : CrossCountryMeet) : ℕ :=
  meet.max_winning_score - meet.min_score + 1

/-- Theorem stating the number of different possible winning scores in the specific meet conditions -/
theorem cross_country_winning_scores :
  ∃ (meet : CrossCountryMeet),
    meet.runners_per_team = 6 ∧
    meet.total_runners = 12 ∧
    meet.min_score = 21 ∧
    meet.max_winning_score = 38 ∧
    winning_scores meet = 18 :=
  sorry

end NUMINAMATH_CALUDE_cross_country_winning_scores_l374_37460


namespace NUMINAMATH_CALUDE_cube_inequality_l374_37409

theorem cube_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l374_37409


namespace NUMINAMATH_CALUDE_unique_digit_subtraction_l374_37461

theorem unique_digit_subtraction (A B C D : Nat) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 →
  1000 * A + 100 * B + 10 * C + D - 989 = 109 →
  1000 * A + 100 * B + 10 * C + D = 1908 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_subtraction_l374_37461


namespace NUMINAMATH_CALUDE_polynomial_value_at_8_l374_37430

def is_monic_degree_7 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d e f : ℝ, p = λ x => x^7 + a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + (p 0)

theorem polynomial_value_at_8 
  (p : ℝ → ℝ) 
  (h_monic : is_monic_degree_7 p)
  (h1 : p 1 = 1) (h2 : p 2 = 2) (h3 : p 3 = 3) (h4 : p 4 = 4)
  (h5 : p 5 = 5) (h6 : p 6 = 6) (h7 : p 7 = 7) :
  p 8 = 5048 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_8_l374_37430


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l374_37441

/-- For a positive integer k, the equation kx^2 + 30x + k = 0 has rational solutions
    if and only if k = 9 or k = 15. -/
theorem quadratic_rational_solutions (k : ℕ+) :
  (∃ x : ℚ, k * x^2 + 30 * x + k = 0) ↔ k = 9 ∨ k = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l374_37441


namespace NUMINAMATH_CALUDE_pascal_triangle_prob_one_or_two_l374_37403

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Type := Unit

/-- Total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (pt : PascalTriangle n) : ℕ := n * (n + 1) / 2

/-- Number of elements equal to 1 in the first n rows of Pascal's Triangle -/
def countOnes (pt : PascalTriangle n) : ℕ := 1 + 2 * (n - 1)

/-- Number of elements equal to 2 in the first n rows of Pascal's Triangle -/
def countTwos (pt : PascalTriangle n) : ℕ := 2 * (n - 3)

/-- Probability of selecting 1 or 2 from the first n rows of Pascal's Triangle -/
def probOneOrTwo (pt : PascalTriangle n) : ℚ :=
  (countOnes pt + countTwos pt : ℚ) / totalElements pt

theorem pascal_triangle_prob_one_or_two :
  ∃ (pt : PascalTriangle 20), probOneOrTwo pt = 73 / 210 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_prob_one_or_two_l374_37403


namespace NUMINAMATH_CALUDE_total_good_vegetables_l374_37426

def carrots_day1 : ℕ := 23
def carrots_day2 : ℕ := 47
def rotten_carrots_day1 : ℕ := 10
def rotten_carrots_day2 : ℕ := 15

def tomatoes_day1 : ℕ := 34
def tomatoes_day2 : ℕ := 50
def rotten_tomatoes_day1 : ℕ := 5
def rotten_tomatoes_day2 : ℕ := 7

def cucumbers_day1 : ℕ := 42
def cucumbers_day2 : ℕ := 38
def rotten_cucumbers_day1 : ℕ := 7
def rotten_cucumbers_day2 : ℕ := 12

theorem total_good_vegetables :
  (carrots_day1 - rotten_carrots_day1) + (carrots_day2 - rotten_carrots_day2) +
  (tomatoes_day1 - rotten_tomatoes_day1) + (tomatoes_day2 - rotten_tomatoes_day2) +
  (cucumbers_day1 - rotten_cucumbers_day1) + (cucumbers_day2 - rotten_cucumbers_day2) = 178 :=
by sorry

end NUMINAMATH_CALUDE_total_good_vegetables_l374_37426


namespace NUMINAMATH_CALUDE_arc_length_radius_l374_37420

/-- Given an arc length of 2.5π cm and a central angle of 75°, the radius of the circle is 6 cm. -/
theorem arc_length_radius (L : ℝ) (θ : ℝ) (R : ℝ) : 
  L = 2.5 * π ∧ θ = 75 → R = 6 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_radius_l374_37420


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l374_37446

open Real

theorem function_inequality_implies_parameter_bound 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) :
  (∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → f x = a / x + x * log x) →
  (∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → g x = x^3 - x^2 - 5) →
  (∀ x₁ x₂, x₁ ∈ Set.Icc (1/2 : ℝ) 2 → x₂ ∈ Set.Icc (1/2 : ℝ) 2 → f x₁ - g x₂ ≥ 2) →
  a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l374_37446


namespace NUMINAMATH_CALUDE_citrus_grove_total_orchards_l374_37463

/-- Represents the number of orchards for each fruit type and the total -/
structure CitrusGrove where
  lemons : ℕ
  oranges : ℕ
  grapefruits : ℕ
  limes : ℕ
  total : ℕ

/-- Theorem stating the total number of orchards in the citrus grove -/
theorem citrus_grove_total_orchards (cg : CitrusGrove) : cg.total = 16 :=
  by
  have h1 : cg.lemons = 8 := by sorry
  have h2 : cg.oranges = cg.lemons / 2 := by sorry
  have h3 : cg.grapefruits = 2 := by sorry
  have h4 : cg.limes + cg.grapefruits = cg.total - (cg.lemons + cg.oranges) := by sorry
  sorry

#check citrus_grove_total_orchards

end NUMINAMATH_CALUDE_citrus_grove_total_orchards_l374_37463


namespace NUMINAMATH_CALUDE_clothing_purchase_problem_l374_37442

/-- The problem of determining the number of clothing pieces bought --/
theorem clothing_purchase_problem (total_spent : ℕ) (price1 price2 other_price : ℕ) :
  total_spent = 610 →
  price1 = 49 →
  price2 = 81 →
  other_price = 96 →
  ∃ (n : ℕ), total_spent = price1 + price2 + n * other_price ∧ n + 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_clothing_purchase_problem_l374_37442


namespace NUMINAMATH_CALUDE_clothing_selection_probability_l374_37454

/-- The probability of selecting at least one shirt, exactly one pair of shorts, and exactly one pair of socks
    when choosing 4 articles of clothing from a drawer containing 6 shirts, 3 pairs of shorts, and 8 pairs of socks -/
theorem clothing_selection_probability : 
  let total_items := 6 + 3 + 8
  let shirts := 6
  let shorts := 3
  let socks := 8
  let items_chosen := 4
  Nat.choose total_items items_chosen ≠ 0 →
  (Nat.choose shorts 1 * Nat.choose socks 1 * 
   (Nat.choose shirts 2 + Nat.choose shirts 1)) / 
  Nat.choose total_items items_chosen = 84 / 397 := by
sorry

end NUMINAMATH_CALUDE_clothing_selection_probability_l374_37454


namespace NUMINAMATH_CALUDE_place_value_ratio_l374_37462

/-- The number we're analyzing -/
def number : ℚ := 25684.2057

/-- The place value of the digit 6 in the number -/
def place_value_6 : ℚ := 1000

/-- The place value of the digit 2 in the number -/
def place_value_2 : ℚ := 0.1

/-- Theorem stating the relationship between the place values -/
theorem place_value_ratio : place_value_6 / place_value_2 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l374_37462


namespace NUMINAMATH_CALUDE_subtracted_value_l374_37424

theorem subtracted_value (x : ℤ) (h : 282 = x + 133) : x - 11 = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l374_37424


namespace NUMINAMATH_CALUDE_common_element_in_sets_l374_37482

theorem common_element_in_sets (n : ℕ) (S : Finset (Finset ℕ)) : 
  n = 50 →
  S.card = n →
  (∀ s ∈ S, s.card = 30) →
  (∀ T ⊆ S, T.card = 30 → ∃ x, ∀ s ∈ T, x ∈ s) →
  ∃ x, ∀ s ∈ S, x ∈ s :=
by sorry

end NUMINAMATH_CALUDE_common_element_in_sets_l374_37482


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l374_37443

theorem fraction_equation_solution :
  ∃ (x : ℚ), (x + 7) / (x - 4) = (x - 3) / (x + 6) ∧ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l374_37443


namespace NUMINAMATH_CALUDE_complex_number_location_l374_37452

theorem complex_number_location (z : ℂ) (h : (2 - 3*Complex.I)*z = 1 + Complex.I) :
  z.re < 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l374_37452


namespace NUMINAMATH_CALUDE_roots_exist_when_q_positive_no_integer_roots_when_q_negative_l374_37425

-- Define the quadratic equations
def equation1 (p q x : ℤ) : Prop := x^2 - p*x + q = 0
def equation2 (p q x : ℤ) : Prop := x^2 - (p+1)*x + q = 0

-- Theorem for q > 0
theorem roots_exist_when_q_positive (q : ℤ) (hq : q > 0) :
  ∃ (p x1 x2 x3 x4 : ℤ), 
    equation1 p q x1 ∧ equation1 p q x2 ∧
    equation2 p q x3 ∧ equation2 p q x4 :=
sorry

-- Theorem for q < 0
theorem no_integer_roots_when_q_negative (q : ℤ) (hq : q < 0) :
  ¬∃ (p x1 x2 x3 x4 : ℤ), 
    equation1 p q x1 ∧ equation1 p q x2 ∧
    equation2 p q x3 ∧ equation2 p q x4 :=
sorry

end NUMINAMATH_CALUDE_roots_exist_when_q_positive_no_integer_roots_when_q_negative_l374_37425


namespace NUMINAMATH_CALUDE_expected_min_swaps_value_l374_37497

/-- Represents a pair of twins -/
structure TwinPair :=
  (twin1 : ℕ)
  (twin2 : ℕ)

/-- Represents an arrangement of twin pairs around a circle -/
def Arrangement := List TwinPair

/-- Computes whether an arrangement has adjacent twins -/
def has_adjacent_twins (arr : Arrangement) : Prop :=
  sorry

/-- Performs a swap between two adjacent positions in the arrangement -/
def swap (arr : Arrangement) (pos : ℕ) : Arrangement :=
  sorry

/-- Computes the minimum number of swaps needed to separate all twins -/
def min_swaps (arr : Arrangement) : ℕ :=
  sorry

/-- Generates all possible arrangements of 5 pairs of twins -/
def all_arrangements : List Arrangement :=
  sorry

/-- Computes the expected value of the minimum number of swaps -/
def expected_min_swaps : ℚ :=
  sorry

theorem expected_min_swaps_value : 
  expected_min_swaps = 926 / 945 :=
sorry

end NUMINAMATH_CALUDE_expected_min_swaps_value_l374_37497


namespace NUMINAMATH_CALUDE_kids_at_camp_difference_l374_37495

theorem kids_at_camp_difference (camp_kids home_kids : ℕ) 
  (h1 : camp_kids = 819058) 
  (h2 : home_kids = 668278) : 
  camp_kids - home_kids = 150780 := by
  sorry

end NUMINAMATH_CALUDE_kids_at_camp_difference_l374_37495


namespace NUMINAMATH_CALUDE_smallest_divisor_after_361_l374_37467

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem smallest_divisor_after_361 (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m ≤ 9999)  -- m is a 4-digit number
  (h2 : is_even m)             -- m is even
  (h3 : m % 361 = 0)           -- m is divisible by 361
  : (∃ d : ℕ, d ∣ m ∧ 361 < d ∧ ∀ d' : ℕ, d' ∣ m → 361 < d' → d ≤ d') → 
    (∃ d : ℕ, d ∣ m ∧ 361 < d ∧ ∀ d' : ℕ, d' ∣ m → 361 < d' → d ≤ d' ∧ d = 380) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_361_l374_37467


namespace NUMINAMATH_CALUDE_triangle_rectangle_ratio_l374_37488

theorem triangle_rectangle_ratio : 
  ∀ (t w l : ℝ),
  t > 0 → w > 0 → l > 0 →
  3 * t = 24 →           -- Perimeter of equilateral triangle
  2 * (w + l) = 24 →     -- Perimeter of rectangle
  l = 2 * w →            -- Length is twice the width
  t / w = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_ratio_l374_37488


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l374_37435

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : IsEven g) : IsEven (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l374_37435


namespace NUMINAMATH_CALUDE_problem_solution_l374_37490

noncomputable def f (m n x : ℝ) : ℝ := m * x + 1 / (n * x) + 1 / 2

theorem problem_solution (m n : ℝ) 
  (h1 : f m n 1 = 2) 
  (h2 : f m n 2 = 11 / 4) :
  (m = 1 ∧ n = 2) ∧ 
  (∀ x y, 1 ≤ x → x < y → f m n x < f m n y) ∧
  (∀ x : ℝ, f m n (1 + 2 * x^2) > f m n (x^2 - 2 * x + 4) ↔ x < -3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l374_37490


namespace NUMINAMATH_CALUDE_investment_ratio_l374_37469

/-- Given two investors p and q, where p invested 60000 and the profit is divided in the ratio 4:6,
    prove that q invested 90000. -/
theorem investment_ratio (p q : ℕ) (h1 : p = 60000) (h2 : 4 * q = 6 * p) : q = 90000 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l374_37469


namespace NUMINAMATH_CALUDE_video_card_upgrade_multiple_l374_37486

theorem video_card_upgrade_multiple (computer_cost monitor_peripheral_ratio base_video_card_cost total_spent : ℚ) :
  computer_cost = 1500 →
  monitor_peripheral_ratio = 1/5 →
  base_video_card_cost = 300 →
  total_spent = 2100 →
  let monitor_peripheral_cost := computer_cost * monitor_peripheral_ratio
  let total_without_upgrade := computer_cost + monitor_peripheral_cost
  let upgraded_video_card_cost := total_spent - total_without_upgrade
  upgraded_video_card_cost / base_video_card_cost = 1 := by
  sorry

end NUMINAMATH_CALUDE_video_card_upgrade_multiple_l374_37486


namespace NUMINAMATH_CALUDE_badminton_probabilities_l374_37481

/-- Represents the state of the badminton game -/
inductive GameState
  | Playing (a b c : ℕ) -- number of consecutive losses for each player
  | Winner (player : Fin 3)

/-- Represents a single game outcome -/
inductive GameOutcome
  | Win
  | Lose

/-- The rules of the badminton game -/
def next_state (s : GameState) (outcome : GameOutcome) : GameState :=
  sorry

/-- The probability of a player winning a single game -/
def win_probability : ℚ := 1/2

/-- The probability of A winning four consecutive games -/
def prob_a_wins_four : ℚ := sorry

/-- The probability of needing a fifth game -/
def prob_fifth_game : ℚ := sorry

/-- The probability of C being the ultimate winner -/
def prob_c_wins : ℚ := sorry

theorem badminton_probabilities :
  prob_a_wins_four = 1/16 ∧
  prob_fifth_game = 3/4 ∧
  prob_c_wins = 7/16 :=
sorry

end NUMINAMATH_CALUDE_badminton_probabilities_l374_37481


namespace NUMINAMATH_CALUDE_shopping_money_l374_37411

theorem shopping_money (initial_amount : ℝ) : 
  0.7 * initial_amount = 2800 → initial_amount = 4000 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_l374_37411


namespace NUMINAMATH_CALUDE_nancys_water_intake_l374_37428

/-- Nancy's weight in pounds -/
def nancys_weight : ℝ := 90

/-- The percentage of body weight Nancy drinks in water -/
def water_percentage : ℝ := 0.6

/-- The amount of water Nancy drinks daily in pounds -/
def water_intake : ℝ := nancys_weight * water_percentage

theorem nancys_water_intake : water_intake = 54 := by
  sorry

end NUMINAMATH_CALUDE_nancys_water_intake_l374_37428


namespace NUMINAMATH_CALUDE_negation_of_implication_l374_37437

theorem negation_of_implication (a b c : ℝ) : 
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
sorry

end NUMINAMATH_CALUDE_negation_of_implication_l374_37437


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l374_37456

/-- Systematic sampling function -/
def systematicSample (totalEmployees : ℕ) (sampleSize : ℕ) (startingNumber : ℕ) (sampleIndex : ℕ) : ℕ :=
  startingNumber + (sampleIndex - 1) * (totalEmployees / sampleSize)

/-- Theorem: If the 5th sample is 23 in a systematic sampling of 40 from 200, then the 8th sample is 38 -/
theorem systematic_sampling_theorem (totalEmployees sampleSize startingNumber : ℕ) 
    (h1 : totalEmployees = 200)
    (h2 : sampleSize = 40)
    (h3 : systematicSample totalEmployees sampleSize startingNumber 5 = 23) :
  systematicSample totalEmployees sampleSize startingNumber 8 = 38 := by
  sorry

#eval systematicSample 200 40 3 5  -- Should output 23
#eval systematicSample 200 40 3 8  -- Should output 38

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l374_37456


namespace NUMINAMATH_CALUDE_xyz_equals_five_l374_37494

-- Define the variables
variable (x y z : ℝ)

-- Define the theorem
theorem xyz_equals_five
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
  x * y * z = 5 := by
  sorry

end NUMINAMATH_CALUDE_xyz_equals_five_l374_37494


namespace NUMINAMATH_CALUDE_average_temperature_l374_37465

def temperatures : List ℤ := [-36, 13, -15, -10]

theorem average_temperature : 
  (temperatures.sum : ℚ) / temperatures.length = -12 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l374_37465


namespace NUMINAMATH_CALUDE_angle_with_special_complement_supplement_l374_37422

theorem angle_with_special_complement_supplement : ∀ x : ℝ,
  (90 - x = (1 / 3) * (180 - x)) → x = 45 := by sorry

end NUMINAMATH_CALUDE_angle_with_special_complement_supplement_l374_37422


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l374_37400

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + y = b₁ ↔ m₂ * x + y = b₂) ↔ m₁ = m₂

/-- Given two lines l₁ and l₂, prove that if they are parallel, then m = -2 -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, m * x + 2 * y - 3 = 0 ↔ 3 * x + (m - 1) * y + m - 6 = 0) →
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l374_37400


namespace NUMINAMATH_CALUDE_fraction_evaluation_l374_37448

theorem fraction_evaluation : (3 : ℚ) / (1 - 3 / 4) = 12 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l374_37448


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l374_37440

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 
    Real.sin (Real.exp (x^2 * Real.sin (5/x)) - 1) + x
  else 
    0

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l374_37440


namespace NUMINAMATH_CALUDE_polynomial_simplification_l374_37432

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (3 * x^4 + 2 * x^3 - 4 * x^2 + 3 * x - 7) =
  2 * x^5 + 3 * x^4 - x^3 + x^2 - 5 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l374_37432


namespace NUMINAMATH_CALUDE_rachel_math_problems_l374_37470

theorem rachel_math_problems (minutes_before_bed : ℕ) (problems_next_day : ℕ) (total_problems : ℕ)
  (h1 : minutes_before_bed = 12)
  (h2 : problems_next_day = 16)
  (h3 : total_problems = 76) :
  ∃ (problems_per_minute : ℕ),
    problems_per_minute * minutes_before_bed + problems_next_day = total_problems ∧
    problems_per_minute = 5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_math_problems_l374_37470


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l374_37402

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 4) : 2*x^2 - 6*x + 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l374_37402


namespace NUMINAMATH_CALUDE_trucks_meeting_l374_37434

/-- Two trucks meeting on a highway --/
theorem trucks_meeting 
  (initial_distance : ℝ) 
  (speed_A speed_B : ℝ) 
  (delay : ℝ) :
  initial_distance = 940 →
  speed_A = 90 →
  speed_B = 80 →
  delay = 1 →
  ∃ (t : ℝ), 
    t > 0 ∧ 
    speed_A * (t + delay) + speed_B * t = initial_distance ∧ 
    speed_A * (t + delay) - speed_B * t = 140 :=
by sorry

end NUMINAMATH_CALUDE_trucks_meeting_l374_37434


namespace NUMINAMATH_CALUDE_republicans_count_l374_37407

/-- Given the total number of representatives and the difference between Republicans and Democrats,
    calculate the number of Republicans. -/
def calculateRepublicans (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

theorem republicans_count :
  calculateRepublicans 434 30 = 232 := by
  sorry

end NUMINAMATH_CALUDE_republicans_count_l374_37407


namespace NUMINAMATH_CALUDE_chocolate_bar_revenue_increase_l374_37496

theorem chocolate_bar_revenue_increase 
  (original_weight : ℝ) (original_price : ℝ) 
  (new_weight : ℝ) (new_price : ℝ) :
  original_weight = 400 →
  original_price = 150 →
  new_weight = 300 →
  new_price = 180 →
  let original_revenue := original_price / original_weight
  let new_revenue := new_price / new_weight
  let revenue_increase := (new_revenue - original_revenue) / original_revenue
  revenue_increase = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_revenue_increase_l374_37496


namespace NUMINAMATH_CALUDE_max_area_rectangular_enclosure_l374_37418

/-- The maximum area of a rectangular enclosure with given constraints -/
theorem max_area_rectangular_enclosure 
  (perimeter : ℝ) 
  (min_length : ℝ) 
  (min_width : ℝ) 
  (h_perimeter : perimeter = 400) 
  (h_min_length : min_length = 100) 
  (h_min_width : min_width = 50) : 
  ∃ (length width : ℝ), 
    length ≥ min_length ∧ 
    width ≥ min_width ∧ 
    2 * (length + width) = perimeter ∧ 
    ∀ (l w : ℝ), 
      l ≥ min_length → 
      w ≥ min_width → 
      2 * (l + w) = perimeter → 
      length * width ≥ l * w ∧ 
      length * width = 10000 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangular_enclosure_l374_37418


namespace NUMINAMATH_CALUDE_divide_fractions_l374_37457

theorem divide_fractions : (7 : ℚ) / 3 / ((5 : ℚ) / 4) = 28 / 15 := by sorry

end NUMINAMATH_CALUDE_divide_fractions_l374_37457


namespace NUMINAMATH_CALUDE_typist_salary_problem_l374_37421

theorem typist_salary_problem (original_salary : ℝ) : 
  let increased_salary := original_salary * 1.1
  let final_salary := increased_salary * 0.95
  final_salary = 5225 →
  original_salary = 5000 := by
sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l374_37421


namespace NUMINAMATH_CALUDE_largest_two_digit_satisfying_property_l374_37466

/-- Given a two-digit number n = 10a + b, where a and b are single digits,
    we define a function that switches the digits and adds 5. -/
def switchAndAdd5 (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a + 5

/-- The property that switching digits and adding 5 results in 3n -/
def satisfiesProperty (n : ℕ) : Prop :=
  switchAndAdd5 n = 3 * n

theorem largest_two_digit_satisfying_property :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ satisfiesProperty n ∧
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ satisfiesProperty m → m ≤ n :=
by
  use 13
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_satisfying_property_l374_37466


namespace NUMINAMATH_CALUDE_ratio_of_55_to_11_l374_37473

theorem ratio_of_55_to_11 : 
  let certain_number : ℚ := 55
  let ratio := certain_number / 11
  ratio = 5 / 1 := by sorry

end NUMINAMATH_CALUDE_ratio_of_55_to_11_l374_37473


namespace NUMINAMATH_CALUDE_williams_tickets_l374_37476

theorem williams_tickets (initial_tickets : ℕ) : 
  initial_tickets + 3 = 18 → initial_tickets = 15 := by
  sorry

end NUMINAMATH_CALUDE_williams_tickets_l374_37476


namespace NUMINAMATH_CALUDE_doubleBracket_two_l374_37444

-- Define the double bracket notation
def doubleBracket (x : ℝ) : ℝ := x^2 + 2*x + 4

-- State the theorem
theorem doubleBracket_two : doubleBracket 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_doubleBracket_two_l374_37444


namespace NUMINAMATH_CALUDE_problem_statement_l374_37445

theorem problem_statement (a b : ℤ) : 
  ({1, a, b / a} : Set ℤ) = {0, a^2, a + b} → a^2017 + b^2017 = -1 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l374_37445


namespace NUMINAMATH_CALUDE_parallelogram_area_l374_37406

/-- The area of a parallelogram with a diagonal of length 30 meters and a perpendicular height to that diagonal of 20 meters is 600 square meters. -/
theorem parallelogram_area (d h : ℝ) (hd : d = 30) (hh : h = 20) :
  d * h = 600 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l374_37406


namespace NUMINAMATH_CALUDE_chess_tournament_games_l374_37478

theorem chess_tournament_games (n : ℕ) (h : n = 24) : 
  n * (n - 1) / 2 = 552 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l374_37478
