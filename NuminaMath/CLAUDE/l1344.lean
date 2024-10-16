import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_1_l1344_134440

theorem system_solution_1 (x y : ℚ) : 
  (3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33) ↔ (x = 6 ∧ y = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_1_l1344_134440


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1344_134477

theorem min_value_sum_of_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (hsum : p + q + r + s + t + u = 10) :
  2/p + 3/q + 5/r + 7/s + 11/t + 13/u ≥ 23.875 ∧ 
  ∃ (p' q' r' s' t' u' : ℝ), 
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 10 ∧
    2/p' + 3/q' + 5/r' + 7/s' + 11/t' + 13/u' = 23.875 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1344_134477


namespace NUMINAMATH_CALUDE_initial_number_of_kids_l1344_134447

theorem initial_number_of_kids (kids_left : ℕ) (kids_gone_home : ℕ) : 
  kids_left = 8 ∧ kids_gone_home = 14 → kids_left + kids_gone_home = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_kids_l1344_134447


namespace NUMINAMATH_CALUDE_mortdecai_mall_delivery_l1344_134442

/-- Represents the egg collection and distribution for Mortdecai in a week -/
structure EggDistribution where
  collected_per_day : ℕ  -- dozens of eggs collected on Tuesday and Thursday
  market_delivery : ℕ    -- dozens of eggs delivered to the market
  pie_usage : ℕ          -- dozens of eggs used for pie
  charity_donation : ℕ   -- dozens of eggs donated to charity

/-- Calculates the number of dozens of eggs delivered to the mall -/
def mall_delivery (ed : EggDistribution) : ℕ :=
  2 * ed.collected_per_day - (ed.market_delivery + ed.pie_usage + ed.charity_donation)

/-- Theorem stating that Mortdecai delivers 5 dozen eggs to the mall -/
theorem mortdecai_mall_delivery :
  let ed : EggDistribution := {
    collected_per_day := 8,
    market_delivery := 3,
    pie_usage := 4,
    charity_donation := 4
  }
  mall_delivery ed = 5 := by sorry

end NUMINAMATH_CALUDE_mortdecai_mall_delivery_l1344_134442


namespace NUMINAMATH_CALUDE_disjoint_sets_cardinality_relation_l1344_134494

theorem disjoint_sets_cardinality_relation (a b : ℕ+) (A B : Finset ℤ) :
  Disjoint A B →
  (∀ i : ℤ, i ∈ A ∪ B → (i + a) ∈ A ∨ (i - b) ∈ B) →
  a * A.card = b * B.card := by
  sorry

end NUMINAMATH_CALUDE_disjoint_sets_cardinality_relation_l1344_134494


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1344_134441

theorem fraction_equation_solution :
  ∃ (x : ℚ), x ≠ 3 ∧ x ≠ -2 ∧ (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1344_134441


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1344_134450

def p (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 2
def q (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem coefficient_of_x_cubed (x : ℝ) : 
  ∃ (a b c d e : ℝ), p x * q x = a * x^5 + b * x^4 - 25 * x^3 + c * x^2 + d * x + e :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1344_134450


namespace NUMINAMATH_CALUDE_remainder_product_l1344_134465

theorem remainder_product (n : ℤ) : n % 24 = 19 → (n % 3) * (n % 8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_product_l1344_134465


namespace NUMINAMATH_CALUDE_teacher_selection_and_assignment_l1344_134492

-- Define the number of male and female teachers
def num_male_teachers : ℕ := 5
def num_female_teachers : ℕ := 4

-- Define the number of male and female teachers to be selected
def selected_male_teachers : ℕ := 3
def selected_female_teachers : ℕ := 2

-- Define the total number of teachers to be selected
def total_selected_teachers : ℕ := selected_male_teachers + selected_female_teachers

-- Define the number of villages
def num_villages : ℕ := 5

-- Theorem statement
theorem teacher_selection_and_assignment :
  (Nat.choose num_male_teachers selected_male_teachers) *
  (Nat.choose num_female_teachers selected_female_teachers) *
  (Nat.factorial total_selected_teachers) = 7200 :=
sorry

end NUMINAMATH_CALUDE_teacher_selection_and_assignment_l1344_134492


namespace NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l1344_134470

theorem smallest_x_multiple_of_53 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → (3 * y + 28)^2 % 53 = 0 → x ≤ y) ∧ 
  (3 * x + 28)^2 % 53 = 0 ∧ 
  x = 26 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l1344_134470


namespace NUMINAMATH_CALUDE_inequality_solution_length_l1344_134496

theorem inequality_solution_length (c d : ℝ) : 
  (∀ x : ℝ, d ≤ x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 ≤ c) →
  (∃ a b : ℝ, ∀ x : ℝ, (d ≤ x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 ≤ c) ↔ (a ≤ x ∧ x ≤ b)) →
  (∃ a b : ℝ, b - a = 8 ∧ ∀ x : ℝ, (d ≤ x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 ≤ c) ↔ (a ≤ x ∧ x ≤ b)) →
  c - d = 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_length_l1344_134496


namespace NUMINAMATH_CALUDE_unique_prime_between_squares_l1344_134479

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 6 ∧ 
  ∃ m : ℕ, p = (m + 1)^2 - 10 ∧
  m^2 < p ∧ p < (m + 1)^2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_squares_l1344_134479


namespace NUMINAMATH_CALUDE_symbol_equation_solution_l1344_134474

theorem symbol_equation_solution (triangle circle : ℕ) 
  (h1 : triangle + circle + circle = 55)
  (h2 : triangle + circle = 40) :
  circle = 15 ∧ triangle = 25 := by
  sorry

end NUMINAMATH_CALUDE_symbol_equation_solution_l1344_134474


namespace NUMINAMATH_CALUDE_dream_car_cost_proof_l1344_134412

/-- Calculates the cost of a dream car given monthly earnings, savings, and total earnings before purchase. -/
def dream_car_cost (monthly_earnings : ℕ) (monthly_savings : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings / monthly_earnings) * monthly_savings

/-- Proves that the cost of the dream car is £45,000 given the specified conditions. -/
theorem dream_car_cost_proof :
  dream_car_cost 4000 500 360000 = 45000 := by
  sorry

end NUMINAMATH_CALUDE_dream_car_cost_proof_l1344_134412


namespace NUMINAMATH_CALUDE_min_value_problem_l1344_134405

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  ∀ x y, x > 0 ∧ y > 1 ∧ x + y = 2 → (4 / a + 1 / (b - 1) ≤ 4 / x + 1 / (y - 1)) ∧
  (∃ x y, x > 0 ∧ y > 1 ∧ x + y = 2 ∧ 4 / x + 1 / (y - 1) = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1344_134405


namespace NUMINAMATH_CALUDE_two_pencils_length_l1344_134437

/-- The length of a pencil in cubes -/
def PencilLength : ℕ := 12

/-- The total length of two pencils -/
def TotalLength : ℕ := PencilLength + PencilLength

/-- Theorem: The total length of two pencils, each 12 cubes long, is 24 cubes -/
theorem two_pencils_length : TotalLength = 24 := by
  sorry

end NUMINAMATH_CALUDE_two_pencils_length_l1344_134437


namespace NUMINAMATH_CALUDE_xyz_equals_one_l1344_134410

theorem xyz_equals_one
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = (2 * b + 3 * c) / (x - 3))
  (eq_b : b = (3 * a + 2 * c) / (y - 3))
  (eq_c : c = (2 * a + 2 * b) / (z - 3))
  (sum_product : x * y + x * z + y * z = -1)
  (sum : x + y + z = 1) :
  x * y * z = 1 := by
sorry


end NUMINAMATH_CALUDE_xyz_equals_one_l1344_134410


namespace NUMINAMATH_CALUDE_surface_area_increase_after_cube_removal_l1344_134438

/-- Represents a rectangular solid with given dimensions -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cube with given side length -/
structure Cube where
  sideLength : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Calculates the surface area of a cube -/
def cubeSurfaceArea (cube : Cube) : ℝ :=
  6 * cube.sideLength^2

/-- Theorem: Removing a cube from the center of a rectangular solid increases surface area -/
theorem surface_area_increase_after_cube_removal 
  (solid : RectangularSolid) 
  (cube : Cube) 
  (h1 : solid.length = 4) 
  (h2 : solid.width = 3) 
  (h3 : solid.height = 5) 
  (h4 : cube.sideLength = 2) :
  surfaceArea solid + cubeSurfaceArea cube = surfaceArea solid + 24 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_increase_after_cube_removal_l1344_134438


namespace NUMINAMATH_CALUDE_rhombus_area_theorem_l1344_134476

/-- Represents a rhombus with side length and diagonal -/
structure Rhombus where
  side_length : ℝ
  diagonal1 : ℝ

/-- Calculates the area of a rhombus given its side length and one diagonal -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

theorem rhombus_area_theorem (r : Rhombus) :
  r.side_length = 2 * Real.sqrt 5 →
  r.diagonal1 = 4 →
  rhombus_area r = 16 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_theorem_l1344_134476


namespace NUMINAMATH_CALUDE_developed_countries_modern_pattern_l1344_134426

/-- Represents different types of countries --/
inductive CountryType
| Developed
| Developing

/-- Represents different population growth patterns --/
inductive GrowthPattern
| Traditional
| Modern

/-- Represents the growth rate of a country --/
structure GrowthRate where
  rate : ℝ

/-- A country with its properties --/
structure Country where
  type : CountryType
  growthPattern : GrowthPattern
  growthRate : GrowthRate
  hasImplementedFamilyPlanning : Bool

/-- Axiom: Developed countries have slow growth rates --/
axiom developed_country_slow_growth (c : Country) :
  c.type = CountryType.Developed → c.growthRate.rate ≤ 0

/-- Axiom: Developing countries have faster growth rates --/
axiom developing_country_faster_growth (c : Country) :
  c.type = CountryType.Developing → c.growthRate.rate > 0

/-- Axiom: Most developing countries are in the traditional growth pattern --/
axiom most_developing_traditional (c : Country) :
  c.type = CountryType.Developing → c.growthPattern = GrowthPattern.Traditional

/-- Axiom: Countries with family planning are in the modern growth pattern --/
axiom family_planning_modern_pattern (c : Country) :
  c.hasImplementedFamilyPlanning → c.growthPattern = GrowthPattern.Modern

/-- Theorem: Developed countries are in the modern population growth pattern --/
theorem developed_countries_modern_pattern (c : Country) :
  c.type = CountryType.Developed → c.growthPattern = GrowthPattern.Modern := by
  sorry

end NUMINAMATH_CALUDE_developed_countries_modern_pattern_l1344_134426


namespace NUMINAMATH_CALUDE_least_number_of_sweets_l1344_134451

theorem least_number_of_sweets (s : ℕ) : s > 0 ∧ 
  s % 6 = 5 ∧ 
  s % 8 = 3 ∧ 
  s % 9 = 6 ∧ 
  s % 11 = 10 ∧ 
  (∀ t : ℕ, t > 0 → t % 6 = 5 → t % 8 = 3 → t % 9 = 6 → t % 11 = 10 → s ≤ t) → 
  s = 2095 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_sweets_l1344_134451


namespace NUMINAMATH_CALUDE_x_value_proof_l1344_134497

theorem x_value_proof (x y : ℚ) : x / y = 12 / 5 → y = 20 → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1344_134497


namespace NUMINAMATH_CALUDE_smaller_cube_edge_length_l1344_134409

theorem smaller_cube_edge_length :
  ∀ (s : ℝ),
  (8 : ℝ) * s^3 = 1000 →
  s = 5 :=
by sorry

end NUMINAMATH_CALUDE_smaller_cube_edge_length_l1344_134409


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1344_134462

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, 7 * x^2 - 2 * x + 45 = 0 ↔ x = p + q * I) → 
  p + q^2 = 321/49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1344_134462


namespace NUMINAMATH_CALUDE_optimal_rental_plan_minimum_transportation_cost_l1344_134452

/-- Represents the rental plan for trucks -/
structure RentalPlan where
  truckA : ℕ
  truckB : ℕ

/-- Checks if a rental plan is valid according to the problem constraints -/
def isValidPlan (plan : RentalPlan) : Prop :=
  plan.truckA + plan.truckB = 6 ∧
  45 * plan.truckA + 30 * plan.truckB ≥ 240 ∧
  400 * plan.truckA + 300 * plan.truckB ≤ 2300

/-- Calculates the total cost of a rental plan -/
def totalCost (plan : RentalPlan) : ℕ :=
  400 * plan.truckA + 300 * plan.truckB

/-- Theorem stating that the optimal plan is 4 Truck A and 2 Truck B -/
theorem optimal_rental_plan :
  ∀ (plan : RentalPlan),
    isValidPlan plan →
    totalCost plan ≥ totalCost { truckA := 4, truckB := 2 } :=
by sorry

/-- Corollary stating the minimum transportation cost -/
theorem minimum_transportation_cost :
  totalCost { truckA := 4, truckB := 2 } = 2200 :=
by sorry

end NUMINAMATH_CALUDE_optimal_rental_plan_minimum_transportation_cost_l1344_134452


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1344_134486

theorem fraction_subtraction (m : ℝ) (h : m ≠ 1) : m / (1 - m) - 1 / (1 - m) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1344_134486


namespace NUMINAMATH_CALUDE_lcm_220_504_l1344_134422

theorem lcm_220_504 : Nat.lcm 220 504 = 27720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_220_504_l1344_134422


namespace NUMINAMATH_CALUDE_population_change_theorem_l1344_134434

/-- Calculates the population after three years of changes --/
def population_after_three_years (initial_population : ℕ) : ℕ :=
  let year1 := (initial_population * 80) / 100
  let year2_increase := (year1 * 110) / 100
  let year2 := (year2_increase * 95) / 100
  let year3_increase := (year2 * 108) / 100
  (year3_increase * 75) / 100

/-- Theorem stating that the population after three years of changes is 10157 --/
theorem population_change_theorem :
  population_after_three_years 15000 = 10157 := by
  sorry

end NUMINAMATH_CALUDE_population_change_theorem_l1344_134434


namespace NUMINAMATH_CALUDE_f_composed_eq_6_has_three_solutions_l1344_134406

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 4 else 3*x - 7

-- Define the composite function f(f(x))
noncomputable def f_composed (x : ℝ) : ℝ := f (f x)

-- Theorem statement
theorem f_composed_eq_6_has_three_solutions :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ f_composed x = 6 :=
sorry

end NUMINAMATH_CALUDE_f_composed_eq_6_has_three_solutions_l1344_134406


namespace NUMINAMATH_CALUDE_number_equation_l1344_134439

theorem number_equation (x : ℝ) : 2500 - (x / 20.04) = 2450 ↔ x = 1002 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1344_134439


namespace NUMINAMATH_CALUDE_football_team_progress_l1344_134480

def football_progress (first_play : Int) (second_play : Int) : Int :=
  let third_play := -2 * (-first_play)
  let fourth_play := third_play / 2
  first_play + second_play + third_play + fourth_play

theorem football_team_progress :
  football_progress (-5) 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l1344_134480


namespace NUMINAMATH_CALUDE_rectangular_table_capacity_l1344_134498

/-- The number of rectangular tables in the library -/
def num_rectangular_tables : ℕ := 7

/-- The number of pupils a square table can seat -/
def pupils_per_square_table : ℕ := 4

/-- The number of square tables in the library -/
def num_square_tables : ℕ := 5

/-- The total number of pupils that can be seated -/
def total_pupils : ℕ := 90

/-- The number of pupils a rectangular table can seat -/
def pupils_per_rectangular_table : ℕ := 10

theorem rectangular_table_capacity :
  pupils_per_rectangular_table * num_rectangular_tables +
  pupils_per_square_table * num_square_tables = total_pupils :=
by sorry

end NUMINAMATH_CALUDE_rectangular_table_capacity_l1344_134498


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1344_134449

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Calculates the number of elderly employees to sample given the number of young employees sampled -/
def elderlyToSample (ec : EmployeeCount) (youngSampled : ℕ) : ℕ :=
  (youngSampled * ec.elderly) / ec.young

/-- Theorem stating that given the specific employee counts and 7 young employees sampled, 
    3 elderly employees should be sampled -/
theorem stratified_sampling_theorem (ec : EmployeeCount) 
  (h1 : ec.total = 750)
  (h2 : ec.young = 350)
  (h3 : ec.middleAged = 250)
  (h4 : ec.elderly = 150)
  (h5 : ec.total = ec.young + ec.middleAged + ec.elderly) :
  elderlyToSample ec 7 = 3 := by
  sorry

#eval elderlyToSample { total := 750, young := 350, middleAged := 250, elderly := 150 } 7

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1344_134449


namespace NUMINAMATH_CALUDE_first_lift_weight_l1344_134404

/-- Given two lifts with a total weight of 600 pounds, where twice the weight of the first lift
    is 300 pounds more than the weight of the second lift, prove that the weight of the first lift
    is 300 pounds. -/
theorem first_lift_weight (first_lift second_lift : ℕ) 
  (total_weight : first_lift + second_lift = 600)
  (lift_relation : 2 * first_lift = second_lift + 300) : 
  first_lift = 300 := by
  sorry

end NUMINAMATH_CALUDE_first_lift_weight_l1344_134404


namespace NUMINAMATH_CALUDE_function_value_at_2009_l1344_134478

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (2 * x * y + 3) + 3 * f (x + y) - 3 * f x = -6 * x

/-- The main theorem stating that for a function satisfying the functional equation, f(2009) = 4021 -/
theorem function_value_at_2009 (f : ℝ → ℝ) (h : FunctionalEquation f) : f 2009 = 4021 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_2009_l1344_134478


namespace NUMINAMATH_CALUDE_max_value_on_curve_l1344_134468

theorem max_value_on_curve :
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4 →
  ∃ M : ℝ, M = 17 ∧ ∀ x' y' : ℝ, (x' - 1)^2 + (y' - 1)^2 = 4 → 3*x' + 4*y' ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l1344_134468


namespace NUMINAMATH_CALUDE_mary_trip_time_and_cost_l1344_134425

-- Define the problem parameters
def uber_to_house : ℕ := 10
def uber_cost : ℚ := 15
def airport_time_factor : ℕ := 5
def bag_check_time : ℕ := 15
def luggage_fee_eur : ℚ := 20
def security_time_factor : ℕ := 3
def boarding_wait : ℕ := 20
def takeoff_wait_factor : ℕ := 2
def first_layover : ℕ := 205  -- 3 hours 25 minutes in minutes
def flight_delay : ℕ := 45
def second_layover : ℕ := 110  -- 1 hour 50 minutes in minutes
def time_zone_change : ℕ := 3
def usd_to_eur : ℚ := 0.85
def usd_to_gbp : ℚ := 0.75
def meal_cost_gbp : ℚ := 10

-- Define the theorem
theorem mary_trip_time_and_cost :
  let total_time : ℕ := uber_to_house + (uber_to_house * airport_time_factor) + 
                        bag_check_time + (bag_check_time * security_time_factor) + 
                        boarding_wait + (boarding_wait * takeoff_wait_factor) + 
                        first_layover + flight_delay + second_layover
  let total_time_hours : ℕ := total_time / 60 + time_zone_change
  let total_cost : ℚ := uber_cost + (luggage_fee_eur / usd_to_eur) + (meal_cost_gbp / usd_to_gbp)
  total_time_hours = 12 ∧ total_cost = 51.86 := by sorry

end NUMINAMATH_CALUDE_mary_trip_time_and_cost_l1344_134425


namespace NUMINAMATH_CALUDE_g_at_negative_two_l1344_134485

def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 5 * x^2 - x + 8

theorem g_at_negative_two : g (-2) = -186 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_two_l1344_134485


namespace NUMINAMATH_CALUDE_waiting_by_tree_only_random_l1344_134435

/-- Represents an idiom --/
inductive Idiom
  | CatchingTurtleInJar
  | WaitingByTreeForRabbit
  | RisingTideLiftAllBoats
  | FishingForMoonInWater

/-- Predicate to determine if an idiom describes a random event --/
def is_random_event (i : Idiom) : Prop :=
  match i with
  | Idiom.WaitingByTreeForRabbit => true
  | _ => false

/-- Theorem stating that "Waiting by a tree for a rabbit" is the only idiom
    among the given options that describes a random event --/
theorem waiting_by_tree_only_random :
  ∀ (i : Idiom), is_random_event i ↔ i = Idiom.WaitingByTreeForRabbit :=
by sorry

end NUMINAMATH_CALUDE_waiting_by_tree_only_random_l1344_134435


namespace NUMINAMATH_CALUDE_certain_number_l1344_134475

theorem certain_number : ∃ x : ℝ, x + 0.675 = 0.8 ∧ x = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_l1344_134475


namespace NUMINAMATH_CALUDE_replaced_person_weight_l1344_134443

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating the weight of the replaced person under the given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 85 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l1344_134443


namespace NUMINAMATH_CALUDE_hexagon_star_perimeter_constant_l1344_134431

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- Checks if a hexagon is equilateral -/
def isEquilateral (h : Hexagon) : Prop := sorry

/-- Calculates the perimeter of a hexagon -/
def perimeter (h : Hexagon) : ℝ := sorry

/-- Calculates the perimeter of the star formed by extending the sides of the hexagon -/
def starPerimeter (h : Hexagon) : ℝ := sorry

theorem hexagon_star_perimeter_constant 
  (h : Hexagon) 
  (equilateral : isEquilateral h) 
  (unit_perimeter : perimeter h = 1) :
  ∀ (h' : Hexagon), 
    isEquilateral h' → 
    perimeter h' = 1 → 
    starPerimeter h = starPerimeter h' :=
sorry

end NUMINAMATH_CALUDE_hexagon_star_perimeter_constant_l1344_134431


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1344_134420

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < C → C < π →
  0 < A → A < π / 3 →
  (2 * a + b) / Real.cos B = -c / Real.cos C →
  (C = 2 * π / 3 ∧ 
   ∀ A' B', 0 < A' → A' < π / 3 → 
             Real.sin A' * Real.sin B' ≤ 1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1344_134420


namespace NUMINAMATH_CALUDE_age_ratio_is_eleven_eighths_l1344_134418

/-- Represents the ages and relationships of Rehana, Phoebe, Jacob, and Xander -/
structure AgeGroup where
  rehana_age : ℕ
  phoebe_age : ℕ
  jacob_age : ℕ
  xander_age : ℕ

/-- Conditions for the age group -/
def valid_age_group (ag : AgeGroup) : Prop :=
  ag.rehana_age = 25 ∧
  ag.rehana_age + 5 = 3 * (ag.phoebe_age + 5) ∧
  ag.jacob_age = (3 * ag.phoebe_age) / 5 ∧
  ag.xander_age = ag.rehana_age + ag.jacob_age - 4

/-- The ratio of combined ages to Xander's age -/
def age_ratio (ag : AgeGroup) : ℚ :=
  (ag.rehana_age + ag.phoebe_age + ag.jacob_age : ℚ) / ag.xander_age

/-- Theorem stating the age ratio is 11/8 for a valid age group -/
theorem age_ratio_is_eleven_eighths (ag : AgeGroup) (h : valid_age_group ag) :
  age_ratio ag = 11/8 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_eleven_eighths_l1344_134418


namespace NUMINAMATH_CALUDE_faster_train_speed_l1344_134453

-- Define the lengths of the trains in meters
def train1_length : ℝ := 200
def train2_length : ℝ := 160

-- Define the time taken to cross in seconds
def crossing_time : ℝ := 11.999040076793857

-- Define the speed of the slower train in km/h
def slower_train_speed : ℝ := 40

-- Define the conversion factor from m/s to km/h
def ms_to_kmh : ℝ := 3.6

-- Theorem statement
theorem faster_train_speed : 
  ∃ (faster_speed : ℝ),
    faster_speed = 68 ∧ 
    (train1_length + train2_length) / crossing_time * ms_to_kmh = faster_speed + slower_train_speed :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l1344_134453


namespace NUMINAMATH_CALUDE_power_sum_prime_l1344_134493

theorem power_sum_prime (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → (2^p + 3^p = a^n) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_prime_l1344_134493


namespace NUMINAMATH_CALUDE_range_of_a_l1344_134481

-- Define the propositions p and q
def p (x : ℝ) : Prop := 2 * x^2 - 5 * x + 3 < 0
def q (x a : ℝ) : Prop := (x - (2 * a + 1)) * (x - 2 * a) ≤ 0

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ (∃ x, p x ∧ ¬q x a)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, necessary_not_sufficient a ↔ (1/4 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1344_134481


namespace NUMINAMATH_CALUDE_correct_average_l1344_134417

theorem correct_average (n : ℕ) (initial_avg : ℚ) 
  (correct_numbers incorrect_numbers : List ℚ) :
  n = 15 ∧ 
  initial_avg = 25 ∧ 
  correct_numbers = [86, 92, 48] ∧ 
  incorrect_numbers = [26, 62, 24] →
  (n * initial_avg + (correct_numbers.sum - incorrect_numbers.sum)) / n = 32.6 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l1344_134417


namespace NUMINAMATH_CALUDE_number_value_l1344_134472

theorem number_value (number y : ℝ) 
  (h1 : (number + 5) * (y - 5) = 0)
  (h2 : ∀ a b : ℝ, (a + 5) * (b - 5) = 0 → a^2 + b^2 ≥ number^2 + y^2)
  (h3 : number^2 + y^2 = 25) : 
  number = -5 := by
sorry

end NUMINAMATH_CALUDE_number_value_l1344_134472


namespace NUMINAMATH_CALUDE_smallest_alpha_beta_inequality_optimal_alpha_beta_optimal_beta_value_l1344_134469

theorem smallest_alpha_beta_inequality (α : ℝ) (β : ℝ) :
  (α > 0 ∧ β > 0 ∧
   ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^α / β) →
  α ≥ 2 :=
by sorry

theorem optimal_alpha_beta :
  ∃ β : ℝ, β > 0 ∧
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 →
    Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β :=
by sorry

theorem optimal_beta_value (β : ℝ) :
  (β > 0 ∧
   ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 →
     Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β) →
  β ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_alpha_beta_inequality_optimal_alpha_beta_optimal_beta_value_l1344_134469


namespace NUMINAMATH_CALUDE_fraction_square_product_l1344_134491

theorem fraction_square_product : (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by sorry

end NUMINAMATH_CALUDE_fraction_square_product_l1344_134491


namespace NUMINAMATH_CALUDE_x_twelfth_power_l1344_134463

theorem x_twelfth_power (x : ℂ) (h : x + 1/x = -Real.sqrt 3) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_l1344_134463


namespace NUMINAMATH_CALUDE_function_equivalence_and_coefficient_sum_l1344_134432

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 - 4*x - 12) / (x + 3)

def g (x : ℝ) : ℝ := x^2 - 4

def A : ℝ := 1
def B : ℝ := 0
def C : ℝ := -4
def D : ℝ := -3

theorem function_equivalence_and_coefficient_sum :
  (∀ x : ℝ, x ≠ D → f x = g x) ∧
  A + B + C + D = -6 := by sorry

end NUMINAMATH_CALUDE_function_equivalence_and_coefficient_sum_l1344_134432


namespace NUMINAMATH_CALUDE_grasshopper_return_to_origin_l1344_134454

def jump_length (n : ℕ) : ℕ := n

def is_horizontal (n : ℕ) : Bool :=
  n % 2 = 1

theorem grasshopper_return_to_origin :
  let horizontal_jumps := List.range 31 |>.filter is_horizontal |>.map jump_length
  let vertical_jumps := List.range 31 |>.filter (fun n => ¬ is_horizontal n) |>.map jump_length
  (List.sum horizontal_jumps = 0) ∧ (List.sum vertical_jumps = 0) := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_return_to_origin_l1344_134454


namespace NUMINAMATH_CALUDE_disaster_relief_team_selection_l1344_134414

/-- The number of internal medicine doctors -/
def internal_doctors : ℕ := 12

/-- The number of surgeons -/
def surgeons : ℕ := 8

/-- The size of the disaster relief medical team -/
def team_size : ℕ := 5

/-- Doctor A is an internal medicine doctor -/
def doctor_A : Fin internal_doctors := sorry

/-- Doctor B is a surgeon -/
def doctor_B : Fin surgeons := sorry

/-- The number of ways to select 5 doctors including A and B -/
def selection_with_A_and_B : ℕ := sorry

/-- The number of ways to select 5 doctors excluding both A and B -/
def selection_without_A_and_B : ℕ := sorry

/-- The number of ways to select 5 doctors including at least one of A or B -/
def selection_with_A_or_B : ℕ := sorry

/-- The number of ways to select 5 doctors with at least one internal medicine doctor and one surgeon -/
def selection_with_both_specialties : ℕ := sorry

theorem disaster_relief_team_selection :
  selection_with_A_and_B = 816 ∧
  selection_without_A_and_B = 8568 ∧
  selection_with_A_or_B = 6936 ∧
  selection_with_both_specialties = 14656 := by sorry

end NUMINAMATH_CALUDE_disaster_relief_team_selection_l1344_134414


namespace NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l1344_134424

def total_flights : ℕ := 8
def late_flights : ℕ := 1
def initial_on_time_flights : ℕ := 3
def subsequent_on_time_flights : ℕ := 4
def target_rate : ℚ := 4/5

def on_time_rate (total : ℕ) (on_time : ℕ) : ℚ :=
  (on_time : ℚ) / (total : ℚ)

theorem phoenix_airport_on_time_rate :
  on_time_rate total_flights (initial_on_time_flights + subsequent_on_time_flights) > target_rate := by
  sorry

end NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l1344_134424


namespace NUMINAMATH_CALUDE_complex_modulus_equality_not_implies_square_equality_l1344_134433

theorem complex_modulus_equality_not_implies_square_equality :
  ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_not_implies_square_equality_l1344_134433


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1344_134444

theorem sum_of_squares_of_roots (p q r : ℂ) : 
  (3 * p^3 - 3 * p^2 - 3 * p - 9 = 0) →
  (3 * q^3 - 3 * q^2 - 3 * q - 9 = 0) →
  (3 * r^3 - 3 * r^2 - 3 * r - 9 = 0) →
  p^2 + q^2 + r^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1344_134444


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1344_134416

theorem max_value_of_expression (x : ℝ) (h : -1 ≤ x ∧ x ≤ 2) :
  ∃ (max : ℝ), max = 5 ∧ ∀ y, -1 ≤ y ∧ y ≤ 2 → 2 + |y - 2| ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1344_134416


namespace NUMINAMATH_CALUDE_complementary_angles_can_be_both_acute_l1344_134427

-- Define what it means for two angles to be complementary
def complementary (a b : ℝ) : Prop := a + b = 90

-- Define what it means for an angle to be acute
def acute (a : ℝ) : Prop := 0 < a ∧ a < 90

theorem complementary_angles_can_be_both_acute :
  ∃ (a b : ℝ), complementary a b ∧ acute a ∧ acute b :=
sorry

end NUMINAMATH_CALUDE_complementary_angles_can_be_both_acute_l1344_134427


namespace NUMINAMATH_CALUDE_pedestrian_speed_ratio_l1344_134436

/-- Two pedestrians depart simultaneously from point A in the same direction.
    The first pedestrian meets a tourist 20 minutes after leaving point A.
    The second pedestrian meets the tourist 5 minutes after the first pedestrian.
    The tourist arrives at point A 10 minutes after the second meeting. -/
theorem pedestrian_speed_ratio 
  (v₁ : ℝ) -- speed of the first pedestrian
  (v₂ : ℝ) -- speed of the second pedestrian
  (v : ℝ)  -- speed of the tourist
  (h₁ : v₁ > 0)
  (h₂ : v₂ > 0)
  (h₃ : v > 0)
  (h₄ : (1/3) * v₁ = (1/4) * v) -- first meeting point equation
  (h₅ : (5/12) * v₂ = (1/6) * v) -- second meeting point equation
  : v₁ / v₂ = 15 / 8 := by
  sorry


end NUMINAMATH_CALUDE_pedestrian_speed_ratio_l1344_134436


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l1344_134402

/-- Calculate simple interest for a loan where the time period equals the interest rate -/
theorem simple_interest_calculation (principal : ℝ) (rate : ℝ) : 
  principal = 1800 →
  rate = 5.93 →
  let interest := principal * rate * rate / 100
  ∃ ε > 0, |interest - 632.61| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l1344_134402


namespace NUMINAMATH_CALUDE_tangent_angle_range_l1344_134467

theorem tangent_angle_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  let f : ℝ → ℝ := λ x => Real.log x + x / b
  let θ := Real.arctan (((1 / a) + (1 / b)) : ℝ)
  π / 4 ≤ θ ∧ θ < π / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_angle_range_l1344_134467


namespace NUMINAMATH_CALUDE_max_product_sum_300_l1344_134495

theorem max_product_sum_300 :
  ∃ (x : ℤ), x * (300 - x) = 22500 ∧ ∀ (y : ℤ), y * (300 - y) ≤ 22500 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l1344_134495


namespace NUMINAMATH_CALUDE_problem_1_l1344_134430

theorem problem_1 : 
  Real.sqrt ((-2)^2) + Real.sqrt 2 * (1 - Real.sqrt (1/2)) + |(-Real.sqrt 8)| = 1 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1344_134430


namespace NUMINAMATH_CALUDE_element_in_complement_l1344_134421

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {1, 5}

-- Define set P
def P : Set Nat := {2, 4}

-- Theorem statement
theorem element_in_complement : 3 ∈ (U \ (M ∪ P)) := by sorry

end NUMINAMATH_CALUDE_element_in_complement_l1344_134421


namespace NUMINAMATH_CALUDE_books_read_total_l1344_134456

/-- The number of books read by Megan, Kelcie, and Greg -/
def total_books (megan_books kelcie_books greg_books : ℕ) : ℕ :=
  megan_books + kelcie_books + greg_books

/-- Theorem stating the total number of books read by Megan, Kelcie, and Greg -/
theorem books_read_total :
  ∃ (megan_books kelcie_books greg_books : ℕ),
    megan_books = 32 ∧
    kelcie_books = megan_books / 4 ∧
    greg_books = 2 * kelcie_books + 9 ∧
    total_books megan_books kelcie_books greg_books = 65 :=
by
  sorry


end NUMINAMATH_CALUDE_books_read_total_l1344_134456


namespace NUMINAMATH_CALUDE_solve_for_y_l1344_134482

theorem solve_for_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1344_134482


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1344_134487

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set {x | -1 < x < 1/3},
    prove that a + b = -5 -/
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a + b = -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1344_134487


namespace NUMINAMATH_CALUDE_fraction_problem_l1344_134413

theorem fraction_problem (f : ℚ) : 
  (1 / 5 : ℚ)^4 * f^2 = 1 / (10 : ℚ)^4 → f = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1344_134413


namespace NUMINAMATH_CALUDE_original_price_calculation_l1344_134446

def selling_price : ℝ := 1220
def gain_percentage : ℝ := 45.23809523809524

theorem original_price_calculation :
  let original_price := selling_price / (1 + gain_percentage / 100)
  ∃ ε > 0, |original_price - 840| < ε :=
by sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1344_134446


namespace NUMINAMATH_CALUDE_heart_self_not_always_zero_l1344_134459

-- Define the heart operation
def heart (x y : ℝ) : ℝ := |x - 2*y|

-- Theorem stating that "x ♡ x = 0 for all x" is false
theorem heart_self_not_always_zero : ¬ ∀ x : ℝ, heart x x = 0 := by
  sorry

end NUMINAMATH_CALUDE_heart_self_not_always_zero_l1344_134459


namespace NUMINAMATH_CALUDE_subtracted_number_l1344_134464

theorem subtracted_number (x y : ℤ) (h1 : x = 30) (h2 : 8 * x - y = 102) : y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l1344_134464


namespace NUMINAMATH_CALUDE_inequality_proof_l1344_134466

theorem inequality_proof (x y z : ℝ) (h : x^4 + y^4 + z^4 + x*y*z = 4) :
  x ≤ 2 ∧ Real.sqrt (2 - x) ≥ (y + z) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1344_134466


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l1344_134455

/-- In a right-angled triangle ABC, given the measures of its angles, prove the relationship between x and y. -/
theorem triangle_angle_relation (x y : ℝ) : 
  x > 0 → y > 0 → x + 3 * y = 90 → x + y = 90 - 2 * y := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l1344_134455


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_l1344_134445

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 < a + b) (h2 : a + b < 3) 
  (h3 : 2 < a - b) (h4 : a - b < 4) :
  ∀ x, (2*a + 3*b = x) → (-9/2 < x ∧ x < 13/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_l1344_134445


namespace NUMINAMATH_CALUDE_specific_truck_toll_l1344_134484

/-- Calculates the toll for a truck crossing a bridge -/
def calculate_toll (x : ℕ) (w : ℝ) (peak_hours : Bool) : ℝ :=
  let y : ℝ := if peak_hours then 2 else 0
  3.50 + 0.50 * (x - 2 : ℝ) + 0.10 * w + y

/-- Theorem: The toll for a specific truck is $8.50 -/
theorem specific_truck_toll :
  calculate_toll 5 15 true = 8.50 := by
  sorry

end NUMINAMATH_CALUDE_specific_truck_toll_l1344_134484


namespace NUMINAMATH_CALUDE_apple_stack_theorem_l1344_134488

/-- Calculates the number of apples in a pyramid-like stack --/
def appleStack (baseWidth : Nat) (baseLength : Nat) : Nat :=
  let layers := min baseWidth baseLength
  List.range layers
  |>.map (fun i => (baseWidth - i) * (baseLength - i))
  |>.sum

/-- Theorem: A pyramid-like stack of apples with a 4x6 base contains 50 apples --/
theorem apple_stack_theorem :
  appleStack 4 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_stack_theorem_l1344_134488


namespace NUMINAMATH_CALUDE_not_divisible_by_169_l1344_134407

theorem not_divisible_by_169 (x : ℤ) : ¬(169 ∣ (x^2 + 5*x + 16)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_169_l1344_134407


namespace NUMINAMATH_CALUDE_system_solution_range_l1344_134473

theorem system_solution_range (x y k : ℝ) : 
  (2 * x + y = 3 * k - 1) →
  (x + 2 * y = -2) →
  (x - y ≤ 5) →
  (k ≤ 4/3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_range_l1344_134473


namespace NUMINAMATH_CALUDE_unique_three_digit_odd_sum_27_l1344_134458

/-- A three-digit number is a natural number between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- The digit sum of a natural number is the sum of its digits. -/
def DigitSum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- A number is odd if it leaves a remainder of 1 when divided by 2. -/
def IsOdd (n : ℕ) : Prop :=
  n % 2 = 1

theorem unique_three_digit_odd_sum_27 :
  ∃! n : ℕ, ThreeDigitNumber n ∧ DigitSum n = 27 ∧ IsOdd n := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_odd_sum_27_l1344_134458


namespace NUMINAMATH_CALUDE_january_display_144_l1344_134429

/-- Rose display sequence with a constant increase -/
structure RoseSequence where
  october : ℕ
  november : ℕ
  december : ℕ
  february : ℕ
  constant_increase : ℕ
  increase_consistent : 
    november - october = constant_increase ∧
    december - november = constant_increase ∧
    february - (december + constant_increase) = constant_increase

/-- The number of roses displayed in January given a rose sequence -/
def january_roses (seq : RoseSequence) : ℕ :=
  seq.december + seq.constant_increase

/-- Theorem stating that for the given rose sequence, January displays 144 roses -/
theorem january_display_144 (seq : RoseSequence) 
  (h_oct : seq.october = 108)
  (h_nov : seq.november = 120)
  (h_dec : seq.december = 132)
  (h_feb : seq.february = 156) :
  january_roses seq = 144 := by
  sorry


end NUMINAMATH_CALUDE_january_display_144_l1344_134429


namespace NUMINAMATH_CALUDE_game_ends_in_finite_steps_l1344_134428

/-- Represents the state of a bowl in the game -/
inductive BowlState
| Empty : BowlState
| NonEmpty : BowlState

/-- Represents the game state -/
def GameState (n : ℕ) := Fin n → BowlState

/-- Function to place a bean in a bowl -/
def placeBeanInBowl (k : ℕ) (n : ℕ) : Fin n :=
  ⟨k * (k + 1) / 2, sorry⟩

/-- Predicate to check if a number is a power of 2 -/
def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- Theorem stating the condition for the game to end in finite steps -/
theorem game_ends_in_finite_steps (n : ℕ) :
  (∃ k : ℕ, ∀ i : Fin n, (placeBeanInBowl k n).val = i.val → 
    ∃ m : ℕ, m ≤ k ∧ (placeBeanInBowl m n).val = i.val) ↔ 
  isPowerOfTwo n :=
sorry


end NUMINAMATH_CALUDE_game_ends_in_finite_steps_l1344_134428


namespace NUMINAMATH_CALUDE_complex_magnitude_l1344_134499

theorem complex_magnitude (z : ℂ) (h : (1 - Complex.I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1344_134499


namespace NUMINAMATH_CALUDE_range_of_m_l1344_134490

-- Define propositions p and q
def p (x : ℝ) : Prop := x < -2 ∨ x > 10
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m^2

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x : ℝ, ¬(p x) → q x m) ∧ ¬(∀ x : ℝ, q x m → ¬(p x))

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, sufficient_not_necessary m ↔ m > 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1344_134490


namespace NUMINAMATH_CALUDE_hexagon_circumscribable_l1344_134403

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon defined by six points -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Checks if two line segments are parallel -/
def parallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if two line segments have equal length -/
def equal_length (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a circle can be circumscribed around a set of points -/
def can_circumscribe (points : List Point) : Prop := sorry

/-- Theorem: A circle can be circumscribed around a hexagon with the given properties -/
theorem hexagon_circumscribable (h : Hexagon) :
  parallel h.A h.B h.D h.E →
  parallel h.B h.C h.E h.F →
  parallel h.C h.D h.F h.A →
  equal_length h.A h.D h.B h.E →
  equal_length h.A h.D h.C h.F →
  can_circumscribe [h.A, h.B, h.C, h.D, h.E, h.F] := by
  sorry

end NUMINAMATH_CALUDE_hexagon_circumscribable_l1344_134403


namespace NUMINAMATH_CALUDE_speed_calculation_l1344_134489

theorem speed_calculation (v : ℝ) (t : ℝ) (h1 : t > 0) :
  v * t = (v + 18) * (2/3 * t) → v = 36 :=
by sorry

end NUMINAMATH_CALUDE_speed_calculation_l1344_134489


namespace NUMINAMATH_CALUDE_future_years_calculation_l1344_134448

/-- The number of years in the future when Shekhar will be 26 years old -/
def future_years : ℕ := 6

/-- Shekhar's current age -/
def shekhar_current_age : ℕ := 20

/-- Shobha's current age -/
def shobha_current_age : ℕ := 15

/-- The ratio of Shekhar's age to Shobha's age -/
def age_ratio : ℚ := 4 / 3

theorem future_years_calculation :
  (shekhar_current_age + future_years = 26) ∧
  (shekhar_current_age : ℚ) / shobha_current_age = age_ratio :=
by sorry

end NUMINAMATH_CALUDE_future_years_calculation_l1344_134448


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1344_134423

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 3 = 0) ↔ (k ≤ 1/3 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1344_134423


namespace NUMINAMATH_CALUDE_proportion_problem_l1344_134460

theorem proportion_problem (hours_per_day : ℝ) (h : hours_per_day = 24) :
  ∃ x : ℝ, (24 : ℝ) / (6 / hours_per_day) = x / 8 ∧ x = 768 :=
by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l1344_134460


namespace NUMINAMATH_CALUDE_candy_bar_fundraiser_profit_l1344_134419

/-- Calculates the profit from selling candy bars in a fundraiser -/
theorem candy_bar_fundraiser_profit
  (boxes : ℕ)
  (bars_per_box : ℕ)
  (selling_price : ℚ)
  (cost_price : ℚ)
  (h1 : boxes = 5)
  (h2 : bars_per_box = 10)
  (h3 : selling_price = 3/2)
  (h4 : cost_price = 1) :
  (boxes * bars_per_box * selling_price) - (boxes * bars_per_box * cost_price) = 25 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_fundraiser_profit_l1344_134419


namespace NUMINAMATH_CALUDE_calculation_proof_l1344_134461

theorem calculation_proof :
  (2 * (Real.sqrt 3 - Real.sqrt 5) + 3 * (Real.sqrt 3 + Real.sqrt 5) = 5 * Real.sqrt 3 + Real.sqrt 5) ∧
  (-1^2 - |1 - Real.sqrt 3| + Real.rpow 8 (1/3) - (-3) * Real.sqrt 9 = 11 - Real.sqrt 3) := by
sorry


end NUMINAMATH_CALUDE_calculation_proof_l1344_134461


namespace NUMINAMATH_CALUDE_seats_formula_l1344_134457

/-- The number of seats in the n-th row of a cinema -/
def seats (n : ℕ) : ℕ :=
  18 + 3 * (n - 1)

/-- Theorem: The number of seats in the n-th row is 3n + 15 -/
theorem seats_formula (n : ℕ) : seats n = 3 * n + 15 := by
  sorry

end NUMINAMATH_CALUDE_seats_formula_l1344_134457


namespace NUMINAMATH_CALUDE_sum_product_identity_l1344_134408

theorem sum_product_identity (a b : ℝ) (h : a + b = a * b) :
  (a^3 + b^3 - a^3 * b^3)^3 + 27 * a^6 * b^6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_identity_l1344_134408


namespace NUMINAMATH_CALUDE_equation_solution_l1344_134400

theorem equation_solution : ∃ (x₁ x₂ : ℚ), 
  (x₁ = 1 ∧ x₂ = 2/3) ∧ 
  (∀ x : ℚ, 3*x*(x-1) = 2*x-2 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1344_134400


namespace NUMINAMATH_CALUDE_complex_product_equals_24_plus_18i_l1344_134411

/-- Complex number multiplication -/
def complex_mult (a b c d : ℤ) : ℤ × ℤ :=
  (a * c - b * d, a * d + b * c)

/-- The imaginary unit i -/
def i : ℤ × ℤ := (0, 1)

theorem complex_product_equals_24_plus_18i : 
  complex_mult 3 (-4) 0 6 = (24, 18) := by sorry

end NUMINAMATH_CALUDE_complex_product_equals_24_plus_18i_l1344_134411


namespace NUMINAMATH_CALUDE_digit_452_of_7_19_is_6_l1344_134483

/-- The decimal representation of 7/19 is repeating -/
def decimal_rep_7_19_repeating : Prop := 
  ∃ (s : List Nat), s.length > 0 ∧ (7 : ℚ) / 19 = (s.map (λ n => (n : ℚ) / 10^s.length)).sum

/-- The 452nd digit after the decimal point in the decimal representation of 7/19 -/
def digit_452_of_7_19 : Nat := sorry

theorem digit_452_of_7_19_is_6 (h : decimal_rep_7_19_repeating) : 
  digit_452_of_7_19 = 6 := by sorry

end NUMINAMATH_CALUDE_digit_452_of_7_19_is_6_l1344_134483


namespace NUMINAMATH_CALUDE_sqrt2_irrational_bound_l1344_134401

theorem sqrt2_irrational_bound (p q : ℤ) (hq : q ≠ 0) :
  |Real.sqrt 2 - (p : ℝ) / (q : ℝ)| > 1 / (3 * (q : ℝ)^2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_irrational_bound_l1344_134401


namespace NUMINAMATH_CALUDE_max_value_of_f_l1344_134415

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ 
  (∀ x > 0, f x ≤ f c) ∧
  f c = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1344_134415


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l1344_134471

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (45/23, -64/23)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8 * x - 3 * y = 24

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 10 * x + 2 * y = 14

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l1344_134471
