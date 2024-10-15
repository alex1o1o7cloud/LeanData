import Mathlib

namespace NUMINAMATH_CALUDE_pig_count_l3880_388033

theorem pig_count (initial_pigs joining_pigs : ℕ) 
  (h1 : initial_pigs = 64) 
  (h2 : joining_pigs = 22) : 
  initial_pigs + joining_pigs = 86 := by
  sorry

end NUMINAMATH_CALUDE_pig_count_l3880_388033


namespace NUMINAMATH_CALUDE_joana_shopping_problem_l3880_388019

theorem joana_shopping_problem :
  ∃! (b c : ℕ), 15 * b + 17 * c = 143 :=
by sorry

end NUMINAMATH_CALUDE_joana_shopping_problem_l3880_388019


namespace NUMINAMATH_CALUDE_marbles_started_with_l3880_388026

def marbles_bought : Real := 489.0
def total_marbles : Real := 2778.0

theorem marbles_started_with : total_marbles - marbles_bought = 2289.0 := by
  sorry

end NUMINAMATH_CALUDE_marbles_started_with_l3880_388026


namespace NUMINAMATH_CALUDE_two_students_per_section_l3880_388085

/-- Represents a school bus with a given number of rows and total capacity. -/
structure SchoolBus where
  rows : ℕ
  capacity : ℕ

/-- Calculates the number of students allowed per section in a school bus. -/
def studentsPerSection (bus : SchoolBus) : ℚ :=
  bus.capacity / (2 * bus.rows)

/-- Theorem stating that for a bus with 13 rows and capacity of 52 students,
    the number of students per section is 2. -/
theorem two_students_per_section :
  let bus : SchoolBus := { rows := 13, capacity := 52 }
  studentsPerSection bus = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_students_per_section_l3880_388085


namespace NUMINAMATH_CALUDE_equation_solution_l3880_388070

theorem equation_solution : 
  ∃ x : ℚ, (17 / 60 + 7 / x = 21 / x + 1 / 15) ∧ (x = 840 / 13) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3880_388070


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3880_388011

/-- A rhombus with given diagonal lengths has a specific perimeter -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l3880_388011


namespace NUMINAMATH_CALUDE_watermelon_theorem_l3880_388059

def watermelon_problem (initial_watermelons : ℕ) (consumption_pattern : List ℕ) : Prop :=
  let total_consumption := consumption_pattern.sum
  let complete_cycles := initial_watermelons / total_consumption
  let remaining_watermelons := initial_watermelons % total_consumption
  complete_cycles * consumption_pattern.length = 3 ∧
  remaining_watermelons < consumption_pattern.head!

theorem watermelon_theorem :
  watermelon_problem 30 [7, 8, 9] :=
by sorry

end NUMINAMATH_CALUDE_watermelon_theorem_l3880_388059


namespace NUMINAMATH_CALUDE_x_value_proof_l3880_388075

theorem x_value_proof (x : ℝ) (h : 9 / x^3 = x / 27) : x = 3 * (3 ^ (1/4)) := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3880_388075


namespace NUMINAMATH_CALUDE_tissue_cost_with_discount_l3880_388052

/-- Calculate the total cost of tissues with discount --/
theorem tissue_cost_with_discount
  (num_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (price_per_tissue : ℚ)
  (discount_rate : ℚ)
  (h_num_boxes : num_boxes = 25)
  (h_packs_per_box : packs_per_box = 18)
  (h_tissues_per_pack : tissues_per_pack = 150)
  (h_price_per_tissue : price_per_tissue = 6 / 100)
  (h_discount_rate : discount_rate = 1 / 10) :
  (num_boxes : ℚ) * (packs_per_box : ℚ) * (tissues_per_pack : ℚ) * price_per_tissue *
    (1 - discount_rate) = 3645 := by
  sorry

#check tissue_cost_with_discount

end NUMINAMATH_CALUDE_tissue_cost_with_discount_l3880_388052


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1997_l3880_388036

def f (x : ℤ) : ℤ := 3 * x + 2

def f_iter : ℕ → (ℤ → ℤ)
| 0 => id
| n + 1 => f ∘ f_iter n

theorem exists_m_divisible_by_1997 : 
  ∃ m : ℕ+, (1997 : ℤ) ∣ f_iter 99 m.val :=
sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1997_l3880_388036


namespace NUMINAMATH_CALUDE_bad_carrots_count_l3880_388081

theorem bad_carrots_count (nancy_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  nancy_carrots = 38 → mom_carrots = 47 → good_carrots = 71 →
  nancy_carrots + mom_carrots - good_carrots = 14 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l3880_388081


namespace NUMINAMATH_CALUDE_weight_qualification_l3880_388097

/-- A weight is qualified if it falls within the acceptable range -/
def is_qualified (weight : ℝ) : Prop :=
  24.75 ≤ weight ∧ weight ≤ 25.25

/-- The labeled weight of the flour -/
def labeled_weight : ℝ := 25

/-- The tolerance of the weight -/
def tolerance : ℝ := 0.25

theorem weight_qualification (weight : ℝ) :
  is_qualified weight ↔ labeled_weight - tolerance ≤ weight ∧ weight ≤ labeled_weight + tolerance :=
by sorry

end NUMINAMATH_CALUDE_weight_qualification_l3880_388097


namespace NUMINAMATH_CALUDE_vector_magnitude_l3880_388067

-- Define the vectors a and b
def a (t : ℝ) : Fin 2 → ℝ := ![t - 2, 3]
def b : Fin 2 → ℝ := ![3, -1]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 2, v i = k * w i

-- State the theorem
theorem vector_magnitude (t : ℝ) :
  (parallel (λ i => a t i + 2 * b i) b) →
  Real.sqrt ((a t 0) ^ 2 + (a t 1) ^ 2) = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3880_388067


namespace NUMINAMATH_CALUDE_two_thirds_of_fifteen_fourths_l3880_388043

theorem two_thirds_of_fifteen_fourths (x : ℚ) : x = 15 / 4 → (2 / 3) * x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_fifteen_fourths_l3880_388043


namespace NUMINAMATH_CALUDE_tobys_breakfast_calories_l3880_388061

-- Define the calorie content of bread and peanut butter
def bread_calories : ℕ := 100
def peanut_butter_calories : ℕ := 200

-- Define Toby's breakfast composition
def bread_pieces : ℕ := 1
def peanut_butter_servings : ℕ := 2

-- Theorem to prove
theorem tobys_breakfast_calories :
  bread_calories * bread_pieces + peanut_butter_calories * peanut_butter_servings = 500 := by
  sorry

end NUMINAMATH_CALUDE_tobys_breakfast_calories_l3880_388061


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3880_388049

/-- The value of r for which the line x + y = 4 is tangent to the circle (x-2)^2 + (y+1)^2 = r -/
theorem tangent_line_to_circle (x y : ℝ) :
  (x + y = 4) →
  ((x - 2)^2 + (y + 1)^2 = (9:ℝ)/2) →
  ∃ (r : ℝ), r = (9:ℝ)/2 ∧ 
    (∀ (x' y' : ℝ), (x' + y' = 4) → ((x' - 2)^2 + (y' + 1)^2 ≤ r)) ∧
    (∃ (x₀ y₀ : ℝ), (x₀ + y₀ = 4) ∧ ((x₀ - 2)^2 + (y₀ + 1)^2 = r)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3880_388049


namespace NUMINAMATH_CALUDE_perpendicular_planes_through_perpendicular_line_non_perpendicular_line_in_perpendicular_planes_l3880_388072

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (passes_through : Plane → Line → Prop)
variable (perpendicular_line : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (line_of_intersection : Plane → Plane → Line)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorems
theorem perpendicular_planes_through_perpendicular_line 
  (P Q : Plane) (l : Line) :
  passes_through Q l → perpendicular_line l P → perpendicular P Q := by sorry

theorem non_perpendicular_line_in_perpendicular_planes 
  (P Q : Plane) (l : Line) :
  perpendicular P Q → 
  in_plane l P → 
  ¬ perpendicular_lines l (line_of_intersection P Q) → 
  ¬ perpendicular_line l Q := by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_through_perpendicular_line_non_perpendicular_line_in_perpendicular_planes_l3880_388072


namespace NUMINAMATH_CALUDE_art_book_cost_is_two_l3880_388082

/-- The cost of each art book given the number of books and their prices --/
def cost_of_art_book (math_books science_books art_books : ℕ) 
                     (total_cost : ℚ) (math_science_cost : ℚ) : ℚ :=
  (total_cost - (math_books + science_books : ℚ) * math_science_cost) / art_books

/-- Theorem stating that the cost of each art book is $2 --/
theorem art_book_cost_is_two :
  cost_of_art_book 2 6 3 30 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_art_book_cost_is_two_l3880_388082


namespace NUMINAMATH_CALUDE_work_completion_time_l3880_388076

theorem work_completion_time (b a_and_b : ℚ) (hb : b = 35) (hab : a_and_b = 20 / 11) :
  let a : ℚ := (1 / a_and_b - 1 / b)⁻¹
  a = 700 / 365 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3880_388076


namespace NUMINAMATH_CALUDE_alphabet_composition_l3880_388094

theorem alphabet_composition (total : ℕ) (both : ℕ) (line_only : ℕ) (dot_only : ℕ) : 
  total = 40 →
  both = 8 →
  line_only = 24 →
  total = both + line_only + dot_only →
  dot_only = 8 := by
sorry

end NUMINAMATH_CALUDE_alphabet_composition_l3880_388094


namespace NUMINAMATH_CALUDE_money_left_calculation_l3880_388074

theorem money_left_calculation (initial_amount spent_on_sweets given_to_each_friend : ℚ) 
  (number_of_friends : ℕ) (h1 : initial_amount = 200.50) 
  (h2 : spent_on_sweets = 35.25) (h3 : given_to_each_friend = 25.20) 
  (h4 : number_of_friends = 2) : 
  initial_amount - spent_on_sweets - (given_to_each_friend * number_of_friends) = 114.85 := by
  sorry

end NUMINAMATH_CALUDE_money_left_calculation_l3880_388074


namespace NUMINAMATH_CALUDE_grid_division_exists_l3880_388028

/-- Represents a figure cut from the grid -/
structure Figure where
  area : ℕ
  externalPerimeter : ℕ
  internalPerimeter : ℕ

/-- Represents the division of the 9x9 grid -/
structure GridDivision where
  a : Figure
  b : Figure
  c : Figure

/-- The proposition to be proved -/
theorem grid_division_exists : ∃ (d : GridDivision),
  -- The grid is 9x9
  (9 * 9 = d.a.area + d.b.area + d.c.area) ∧
  -- All figures have equal area
  (d.a.area = d.b.area) ∧ (d.b.area = d.c.area) ∧
  -- The perimeter of c equals the sum of perimeters of a and b
  (d.c.externalPerimeter + d.c.internalPerimeter = 
   d.a.externalPerimeter + d.a.internalPerimeter + 
   d.b.externalPerimeter + d.b.internalPerimeter) ∧
  -- The sum of external perimeters is the perimeter of the 9x9 grid
  (d.a.externalPerimeter + d.b.externalPerimeter + d.c.externalPerimeter = 4 * 9) ∧
  -- The sum of a and b's internal perimeters equals c's internal perimeter
  (d.a.internalPerimeter + d.b.internalPerimeter = d.c.internalPerimeter) :=
sorry

end NUMINAMATH_CALUDE_grid_division_exists_l3880_388028


namespace NUMINAMATH_CALUDE_integral_tangent_sine_l3880_388015

open Real MeasureTheory

theorem integral_tangent_sine (f : ℝ → ℝ) :
  (∫ x in Set.Icc (π/4) (arctan 3), 1 / ((3 * tan x + 5) * sin (2 * x))) = (1/10) * log (12/7) := by
  sorry

end NUMINAMATH_CALUDE_integral_tangent_sine_l3880_388015


namespace NUMINAMATH_CALUDE_first_expression_value_l3880_388048

theorem first_expression_value (a : ℝ) (E : ℝ) : 
  a = 28 → (E + (3 * a - 8)) / 2 = 74 → E = 72 := by
  sorry

end NUMINAMATH_CALUDE_first_expression_value_l3880_388048


namespace NUMINAMATH_CALUDE_divisibility_conditions_solutions_l3880_388089

theorem divisibility_conditions_solutions (a b : ℕ+) : 
  (a ∣ b^2) → (b ∣ a^2) → ((a + 1) ∣ (b^2 + 1)) → 
  (∃ q : ℕ+, (a = q^2 ∧ b = q) ∨ 
             (a = q^2 ∧ b = q^3) ∨ 
             (a = (q^2 - 1) * q^2 ∧ b = q * (q^2 - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_conditions_solutions_l3880_388089


namespace NUMINAMATH_CALUDE_koi_fish_count_l3880_388073

/-- Calculates the number of koi fish after three weeks given the initial conditions and final number of goldfish --/
theorem koi_fish_count (initial_total : ℕ) (koi_added_per_day : ℕ) (goldfish_added_per_day : ℕ) (days : ℕ) (final_goldfish : ℕ) : 
  initial_total = 280 →
  days = 21 →
  koi_added_per_day = 2 →
  goldfish_added_per_day = 5 →
  final_goldfish = 200 →
  initial_total + days * (koi_added_per_day + goldfish_added_per_day) - final_goldfish = 227 :=
by
  sorry

#check koi_fish_count

end NUMINAMATH_CALUDE_koi_fish_count_l3880_388073


namespace NUMINAMATH_CALUDE_oscar_swag_bag_scarf_cost_l3880_388040

/-- The cost of each designer scarf in the Oscar swag bag -/
def scarf_cost (total_value earring_cost iphone_cost num_earrings num_scarves : ℕ) : ℕ :=
  (total_value - (num_earrings * earring_cost + iphone_cost)) / num_scarves

/-- Theorem: The cost of each designer scarf in the Oscar swag bag is $1,500 -/
theorem oscar_swag_bag_scarf_cost :
  scarf_cost 20000 6000 2000 2 4 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_oscar_swag_bag_scarf_cost_l3880_388040


namespace NUMINAMATH_CALUDE_markov_equation_solution_l3880_388047

/-- Markov equation -/
def markov_equation (x y z : ℕ+) : Prop :=
  x^2 + y^2 + z^2 = 3*x*y*z

/-- Definition of coprime positive integers -/
def coprime (a b : ℕ+) : Prop :=
  Nat.gcd a.val b.val = 1

/-- Definition of sum of squares of two coprime integers -/
def sum_of_coprime_squares (a : ℕ+) : Prop :=
  ∃ (p q : ℕ+), coprime p q ∧ a = p^2 + q^2

/-- Main theorem -/
theorem markov_equation_solution :
  ∀ (a b c : ℕ+), markov_equation a b c →
    (coprime a b ∧ coprime b c ∧ coprime a c) ∧
    (a ≠ 1 → sum_of_coprime_squares a) :=
sorry

end NUMINAMATH_CALUDE_markov_equation_solution_l3880_388047


namespace NUMINAMATH_CALUDE_root_equation_implies_d_equals_eight_l3880_388009

theorem root_equation_implies_d_equals_eight 
  (a b c d : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) (hd : d > 1) 
  (h : ∀ (M : ℝ), M ≠ 1 → (M^(1/a + 1/(a*b) + 1/(a*b*c) + 1/(a*b*c*d))) = M^(17/24)) : 
  d = 8 := by
sorry

end NUMINAMATH_CALUDE_root_equation_implies_d_equals_eight_l3880_388009


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3880_388038

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 * x - 2) + 12 / Real.sqrt (3 * x - 2) = 8 ↔ x = 2 ∨ x = 38 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3880_388038


namespace NUMINAMATH_CALUDE_system_solution_l3880_388090

theorem system_solution (x y : ℝ) : 
  (x + 2*y = 0 ∧ 3*x - 4*y = 5) ↔ (x = 1 ∧ y = -1/2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3880_388090


namespace NUMINAMATH_CALUDE_prob_different_ranks_value_l3880_388050

/-- The number of cards in a standard deck --/
def deck_size : ℕ := 52

/-- The number of ranks in a standard deck --/
def num_ranks : ℕ := 13

/-- The number of suits in a standard deck --/
def num_suits : ℕ := 4

/-- The probability of drawing two cards of different ranks from a standard deck --/
def prob_different_ranks : ℚ :=
  (deck_size * (deck_size - 1) - num_ranks * (num_suits * (num_suits - 1))) /
  (deck_size * (deck_size - 1))

theorem prob_different_ranks_value : prob_different_ranks = 208 / 221 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_ranks_value_l3880_388050


namespace NUMINAMATH_CALUDE_exists_multi_illuminated_point_l3880_388053

/-- Represents a street light in City A -/
structure StreetLight where
  position : ℝ × ℝ
  batteryReplacementTime : ℝ

/-- The city configuration -/
structure CityA where
  streetLights : Set StreetLight
  cityRadius : ℝ
  newBatteryRadius : ℝ
  radiusDecreaseRate : ℝ
  batteryLifespan : ℝ
  dailyBatteryUsage : ℕ

/-- The illumination area of a street light at a given time -/
def illuminationArea (light : StreetLight) (time : ℝ) (city : CityA) : Set (ℝ × ℝ) :=
  sorry

/-- Theorem: There exists a point illuminated by multiple street lights -/
theorem exists_multi_illuminated_point (city : CityA) 
  (h1 : city.cityRadius = 10000)
  (h2 : city.newBatteryRadius = 200)
  (h3 : city.radiusDecreaseRate = 10)
  (h4 : city.batteryLifespan = 20)
  (h5 : city.dailyBatteryUsage = 18000) :
  ∃ (point : ℝ × ℝ) (time : ℝ), 
    ∃ (light1 light2 : StreetLight), light1 ≠ light2 ∧ 
    point ∈ illuminationArea light1 time city ∧ 
    point ∈ illuminationArea light2 time city :=
  sorry

end NUMINAMATH_CALUDE_exists_multi_illuminated_point_l3880_388053


namespace NUMINAMATH_CALUDE_visitors_in_scientific_notation_l3880_388088

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number of visitors -/
def visitors : ℕ := 876000

/-- The scientific notation representation of the number of visitors -/
def visitors_scientific : ScientificNotation :=
  { coefficient := 8.76
  , exponent := 5
  , h1 := by sorry }

theorem visitors_in_scientific_notation :
  (visitors : ℝ) = visitors_scientific.coefficient * (10 : ℝ) ^ visitors_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_visitors_in_scientific_notation_l3880_388088


namespace NUMINAMATH_CALUDE_dinner_pizzas_count_l3880_388055

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := 15

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := total_pizzas - lunch_pizzas

theorem dinner_pizzas_count : dinner_pizzas = 6 := by
  sorry

end NUMINAMATH_CALUDE_dinner_pizzas_count_l3880_388055


namespace NUMINAMATH_CALUDE_units_produced_today_l3880_388008

theorem units_produced_today (past_average : ℝ) (new_average : ℝ) (past_days : ℕ) :
  past_average = 40 →
  new_average = 45 →
  past_days = 9 →
  (past_days + 1) * new_average - past_days * past_average = 90 := by
  sorry

end NUMINAMATH_CALUDE_units_produced_today_l3880_388008


namespace NUMINAMATH_CALUDE_galaxy_first_chinese_supercomputer_l3880_388018

/-- Represents a supercomputer -/
structure Supercomputer where
  name : String
  country : String
  performance : ℕ  -- calculations per second
  year_introduced : ℕ
  month_introduced : ℕ

/-- The Galaxy supercomputer -/
def galaxy : Supercomputer :=
  { name := "Galaxy"
  , country := "China"
  , performance := 100000000  -- 100 million
  , year_introduced := 1983
  , month_introduced := 12 }

/-- Predicate to check if a supercomputer meets the criteria -/
def meets_criteria (sc : Supercomputer) : Prop :=
  sc.country = "China" ∧
  sc.performance ≥ 100000000 ∧
  sc.year_introduced = 1983 ∧
  sc.month_introduced = 12

/-- Theorem stating that Galaxy was China's first supercomputer meeting the criteria -/
theorem galaxy_first_chinese_supercomputer :
  meets_criteria galaxy ∧
  ∀ (sc : Supercomputer), meets_criteria sc → sc.name = galaxy.name :=
by sorry


end NUMINAMATH_CALUDE_galaxy_first_chinese_supercomputer_l3880_388018


namespace NUMINAMATH_CALUDE_factorization_d_is_valid_l3880_388013

/-- Represents a polynomial factorization -/
def IsFactorization (left right : ℝ → ℝ) : Prop :=
  ∀ x, left x = right x ∧ 
       ∃ p q : ℝ → ℝ, right = fun y ↦ p y * q y

/-- The specific factorization we want to prove -/
def FactorizationD (x : ℝ) : ℝ := x^2 + 4*x + 4

/-- The factored form -/
def FactoredFormD (x : ℝ) : ℝ := (x + 2)^2

/-- Theorem stating that FactorizationD is a valid factorization -/
theorem factorization_d_is_valid : IsFactorization FactorizationD FactoredFormD := by
  sorry

end NUMINAMATH_CALUDE_factorization_d_is_valid_l3880_388013


namespace NUMINAMATH_CALUDE_equation_solution_l3880_388080

theorem equation_solution :
  ∃ y : ℚ, (3 / y - (5 / y) / (7 / y) = 1.2) ∧ (y = 105 / 67) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3880_388080


namespace NUMINAMATH_CALUDE_tire_usage_l3880_388001

/-- Proves that each tire is used for 32,000 miles given the conditions of the problem -/
theorem tire_usage (total_miles : ℕ) (total_tires : ℕ) (tires_in_use : ℕ) 
  (h1 : total_miles = 40000)
  (h2 : total_tires = 5)
  (h3 : tires_in_use = 4)
  (h4 : tires_in_use < total_tires) :
  (total_miles * tires_in_use) / total_tires = 32000 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_l3880_388001


namespace NUMINAMATH_CALUDE_seven_digit_increasing_numbers_l3880_388077

theorem seven_digit_increasing_numbers (n : ℕ) (h : n = 7) :
  (Nat.choose (9 + n - 1) n) % 1000 = 435 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_increasing_numbers_l3880_388077


namespace NUMINAMATH_CALUDE_auto_finance_fraction_l3880_388045

theorem auto_finance_fraction (total_credit : ℝ) (auto_credit_percentage : ℝ) (finance_company_credit : ℝ) :
  total_credit = 475 →
  auto_credit_percentage = 0.36 →
  finance_company_credit = 57 →
  finance_company_credit / (auto_credit_percentage * total_credit) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_auto_finance_fraction_l3880_388045


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3880_388096

/-- 
For a quadratic equation (k-2)x^2 - 2kx + k = 6 to have real roots,
k must satisfy k ≥ 1.5 and k ≠ 2.
-/
theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ 1.5 ∧ k ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3880_388096


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3880_388021

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  /-- Distance from center to focus -/
  c : ℝ
  /-- Ratio of b to a in the standard equation -/
  b_over_a : ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    h.b_over_a = b / a ∧
    h.c^2 = a^2 + b^2 ∧
    x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbola_equation (h : Hyperbola) (h_focus : h.c = 10) (h_asymptote : h.b_over_a = 4/3) :
  standard_equation h x y ↔ x^2 / 36 - y^2 / 64 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3880_388021


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3880_388078

theorem container_volume_ratio (C D : ℚ) 
  (h : C > 0 ∧ D > 0) 
  (transfer : (3 / 4 : ℚ) * C = (2 / 3 : ℚ) * D) : 
  C / D = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3880_388078


namespace NUMINAMATH_CALUDE_sqrt_2x_plus_1_domain_l3880_388084

theorem sqrt_2x_plus_1_domain (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x + 1) ↔ x ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_plus_1_domain_l3880_388084


namespace NUMINAMATH_CALUDE_stating_two_cookies_per_guest_l3880_388064

/-- 
Given a total number of cookies and guests, calculates the number of cookies per guest,
assuming each guest receives the same number of cookies.
-/
def cookiesPerGuest (totalCookies guests : ℕ) : ℚ :=
  totalCookies / guests

/-- 
Theorem stating that when there are 10 cookies and 5 guests,
each guest receives 2 cookies.
-/
theorem two_cookies_per_guest :
  cookiesPerGuest 10 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_two_cookies_per_guest_l3880_388064


namespace NUMINAMATH_CALUDE_delta_sum_bound_l3880_388056

/-- The greatest odd divisor of a positive integer -/
def greatest_odd_divisor (n : ℕ+) : ℕ+ :=
  sorry

/-- The sum of δ(n)/n from 1 to x -/
def delta_sum (x : ℕ+) : ℚ :=
  sorry

/-- Theorem: For any positive integer x, |∑(n=1 to x) [δ(n)/n] - (2/3)x| < 1 -/
theorem delta_sum_bound (x : ℕ+) :
  |delta_sum x - (2/3 : ℚ) * x.val| < 1 :=
sorry

end NUMINAMATH_CALUDE_delta_sum_bound_l3880_388056


namespace NUMINAMATH_CALUDE_tangent_curves_alpha_l3880_388037

theorem tangent_curves_alpha (f g : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, g x = α * x^2) →
  (∃ x₀, f x₀ = g x₀ ∧ deriv f x₀ = deriv g x₀) →
  α = Real.exp 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_tangent_curves_alpha_l3880_388037


namespace NUMINAMATH_CALUDE_market_price_calculation_l3880_388063

/-- Proves that given a reduction in sales tax from 3.5% to 3 1/3% resulting in a
    difference of Rs. 12.99999999999999 in tax amount, the market price of the article is Rs. 7800. -/
theorem market_price_calculation (initial_tax : ℚ) (reduced_tax : ℚ) (tax_difference : ℚ) 
  (h1 : initial_tax = 7/200)  -- 3.5%
  (h2 : reduced_tax = 1/30)   -- 3 1/3%
  (h3 : tax_difference = 12999999999999999/1000000000000000) : -- 12.99999999999999
  ∃ (market_price : ℕ), 
    (initial_tax - reduced_tax) * market_price = tax_difference ∧ 
    market_price = 7800 := by
sorry

end NUMINAMATH_CALUDE_market_price_calculation_l3880_388063


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l3880_388093

theorem fraction_equality_solution :
  ∀ m n : ℕ+, 
  (m : ℚ) / ((n : ℚ) + m) = (n : ℚ) / ((n : ℚ) - m) →
  (∃ h : ℕ, m = (2*h + 1)*h ∧ n = (2*h + 1)*(h + 1)) ∨
  (∃ h : ℕ+, m = 2*h*(4*h^2 - 1) ∧ n = 2*h*(4*h^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l3880_388093


namespace NUMINAMATH_CALUDE_scale_model_height_l3880_388098

/-- The scale ratio used for the model -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the Eiffel Tower in feet -/
def actual_height : ℕ := 984

/-- The height of the scale model before rounding -/
def model_height : ℚ := actual_height / scale_ratio

/-- Function to round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem scale_model_height :
  round_to_nearest model_height = 39 := by
  sorry

end NUMINAMATH_CALUDE_scale_model_height_l3880_388098


namespace NUMINAMATH_CALUDE_square_binomial_coefficient_l3880_388069

theorem square_binomial_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 24 * x + 9 = (r * x + s)^2) → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_coefficient_l3880_388069


namespace NUMINAMATH_CALUDE_circle_equation_l3880_388030

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the line x - y = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = p.2}

-- State the theorem
theorem circle_equation :
  -- C passes through the origin
  (0, 0) ∈ C ∧
  -- The center of C is on the positive x-axis
  (∃ a : ℝ, a > 0 ∧ (a, 0) ∈ C) ∧
  -- The chord intercepted by the line x-y=0 on C has a length of 2√2
  (∃ p q : ℝ × ℝ, p ∈ C ∧ q ∈ C ∧ p ∈ L ∧ q ∈ L ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 2) →
  -- Then the equation of C is (x-2)^2 + y^2 = 4
  C = {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4} :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l3880_388030


namespace NUMINAMATH_CALUDE_bella_steps_l3880_388017

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℕ := 15840

/-- Bella's speed relative to Ella's -/
def speed_ratio : ℕ := 4

/-- The number of feet Bella covers in one step -/
def feet_per_step : ℕ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps : ℕ := 1056

theorem bella_steps :
  distance * (speed_ratio + 1) / speed_ratio / feet_per_step = steps := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_l3880_388017


namespace NUMINAMATH_CALUDE_topsoil_cost_l3880_388010

-- Define the cost per cubic foot of topsoil
def cost_per_cubic_foot : ℝ := 8

-- Define the conversion factor from cubic yards to cubic feet
def cubic_yards_to_cubic_feet : ℝ := 27

-- Define the volume in cubic yards
def volume_in_cubic_yards : ℝ := 7

-- Theorem statement
theorem topsoil_cost : 
  volume_in_cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l3880_388010


namespace NUMINAMATH_CALUDE_total_time_wasted_l3880_388044

def traffic_wait_time : ℝ := 2
def freeway_exit_time_multiplier : ℝ := 4

theorem total_time_wasted : 
  traffic_wait_time + freeway_exit_time_multiplier * traffic_wait_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_time_wasted_l3880_388044


namespace NUMINAMATH_CALUDE_f_min_value_inequality_solution_l3880_388003

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 5|

-- Theorem 1: The minimum value of f(x) is 6
theorem f_min_value : ∀ x : ℝ, f x ≥ 6 := by sorry

-- Theorem 2: Solution to the inequality when m = 6
theorem inequality_solution :
  ∀ x : ℝ, (|x - 3| - 2*x ≤ 4) ↔ (x ≥ -1/3) := by sorry

end NUMINAMATH_CALUDE_f_min_value_inequality_solution_l3880_388003


namespace NUMINAMATH_CALUDE_rainstorm_multiple_rainstorm_multiple_proof_l3880_388029

/-- Given the conditions of a rainstorm, prove that the multiple of the first hour's
    rain amount that determines the second hour's rain (minus 7 inches) is equal to 2. -/
theorem rainstorm_multiple : ℝ → Prop :=
  fun x =>
    let first_hour_rain := 5
    let second_hour_rain := x * first_hour_rain + 7
    let total_rain := 22
    first_hour_rain + second_hour_rain = total_rain →
    x = 2

/-- Proof of the rainstorm_multiple theorem -/
theorem rainstorm_multiple_proof : rainstorm_multiple 2 := by
  sorry

end NUMINAMATH_CALUDE_rainstorm_multiple_rainstorm_multiple_proof_l3880_388029


namespace NUMINAMATH_CALUDE_sixth_root_of_six_sqrt_2_over_sqrt_3_of_6_l3880_388014

theorem sixth_root_of_six (x : ℝ) (h : x > 0) : 
  (x^(1/2)) / (x^(1/3)) = x^(1/6) := by
  sorry

-- The specific case for x = 6
theorem sqrt_2_over_sqrt_3_of_6 : 
  (6^(1/2)) / (6^(1/3)) = 6^(1/6) := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_six_sqrt_2_over_sqrt_3_of_6_l3880_388014


namespace NUMINAMATH_CALUDE_exponent_division_23_l3880_388032

theorem exponent_division_23 : (23 ^ 11) / (23 ^ 8) = 12167 := by sorry

end NUMINAMATH_CALUDE_exponent_division_23_l3880_388032


namespace NUMINAMATH_CALUDE_courier_packages_l3880_388091

theorem courier_packages (x : ℕ) (h1 : x + 2*x = 240) : x = 80 := by
  sorry

end NUMINAMATH_CALUDE_courier_packages_l3880_388091


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l3880_388071

/-- A coloring of the edges of a complete graph on 10 vertices using two colors -/
def TwoColoring : Type := Fin 10 → Fin 10 → Bool

/-- A triangle in a graph is represented by three distinct vertices -/
structure Triangle (n : Nat) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n
  distinct : v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3

/-- A triangle is monochromatic if all its edges have the same color -/
def isMonochromatic (c : TwoColoring) (t : Triangle 10) : Prop :=
  c t.v1 t.v2 = c t.v2 t.v3 ∧ c t.v2 t.v3 = c t.v3 t.v1

/-- The main theorem: every two-coloring of K_10 contains a monochromatic triangle -/
theorem monochromatic_triangle_exists (c : TwoColoring) : 
  ∃ t : Triangle 10, isMonochromatic c t := by
  sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l3880_388071


namespace NUMINAMATH_CALUDE_intersecting_quadratic_properties_l3880_388060

/-- A quadratic function that intersects both coordinate axes at three points -/
structure IntersectingQuadratic where
  b : ℝ
  intersects_axes : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ -x₁^2 - 2*x₁ + b = 0 ∧ -x₂^2 - 2*x₂ + b = 0
  intersects_y : b ≠ 0

/-- The range of possible values for b -/
def valid_b_range (q : IntersectingQuadratic) : Prop :=
  q.b > -1 ∧ q.b ≠ 0

/-- The equation of the circle passing through the three intersection points -/
def circle_equation (q : IntersectingQuadratic) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + (1 - q.b)*y - q.b = 0

theorem intersecting_quadratic_properties (q : IntersectingQuadratic) :
  valid_b_range q ∧
  ∀ (x y : ℝ), circle_equation q x y ↔ 
    (x = 0 ∧ y = q.b) ∨ 
    (y = 0 ∧ -x^2 - 2*x + q.b = 0) :=
sorry

end NUMINAMATH_CALUDE_intersecting_quadratic_properties_l3880_388060


namespace NUMINAMATH_CALUDE_field_goal_missed_fraction_l3880_388079

theorem field_goal_missed_fraction 
  (total_attempts : ℕ) 
  (wide_right_percentage : ℚ) 
  (wide_right_count : ℕ) 
  (h1 : total_attempts = 60) 
  (h2 : wide_right_percentage = 1/5) 
  (h3 : wide_right_count = 3) : 
  (wide_right_count / wide_right_percentage) / total_attempts = 1/4 :=
sorry

end NUMINAMATH_CALUDE_field_goal_missed_fraction_l3880_388079


namespace NUMINAMATH_CALUDE_square_side_length_l3880_388020

theorem square_side_length (A : ℝ) (s : ℝ) (h : A = 9) (h₁ : A = s^2) :
  s = Real.sqrt 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3880_388020


namespace NUMINAMATH_CALUDE_inscribed_circle_larger_than_sphere_l3880_388062

structure Tetrahedron where
  inscribedSphereRadius : ℝ
  faceInscribedCircleRadius : ℝ
  inscribedSphereRadiusPositive : 0 < inscribedSphereRadius
  faceInscribedCircleRadiusPositive : 0 < faceInscribedCircleRadius

theorem inscribed_circle_larger_than_sphere (t : Tetrahedron) :
  t.faceInscribedCircleRadius > t.inscribedSphereRadius := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_larger_than_sphere_l3880_388062


namespace NUMINAMATH_CALUDE_area_of_ABC_l3880_388007

-- Define the triangle ABC and point P
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def P : ℝ × ℝ := sorry

-- Define the conditions
def is_scalene_right_triangle (A B C : ℝ × ℝ) : Prop := sorry
def point_on_hypotenuse (A C P : ℝ × ℝ) : Prop := sorry
def angle_ABP_45 (A B P : ℝ × ℝ) : Prop := sorry
def AP_equals_1 (A P : ℝ × ℝ) : Prop := sorry
def CP_equals_2 (C P : ℝ × ℝ) : Prop := sorry

-- Define the area function
def area (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_ABC :
  is_scalene_right_triangle A B C →
  point_on_hypotenuse A C P →
  angle_ABP_45 A B P →
  AP_equals_1 A P →
  CP_equals_2 C P →
  area A B C = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_ABC_l3880_388007


namespace NUMINAMATH_CALUDE_foreign_language_speakers_l3880_388041

theorem foreign_language_speakers (total_students : ℕ) (male_students : ℕ) (female_students : ℕ) :
  male_students = female_students →
  (3 : ℚ) / 5 * male_students + (2 : ℚ) / 3 * female_students = (19 : ℚ) / 30 * (male_students + female_students) :=
by sorry

end NUMINAMATH_CALUDE_foreign_language_speakers_l3880_388041


namespace NUMINAMATH_CALUDE_digit_sum_power_equality_l3880_388095

-- Define the sum of digits function
def S (m : ℕ) : ℕ := sorry

-- Define the set of solutions
def solution_set : Set (ℕ × ℕ) :=
  {p | ∃ (b : ℕ), p = (1, b + 1)} ∪ {(3, 2), (9, 1)}

-- State the theorem
theorem digit_sum_power_equality :
  ∀ a b : ℕ, a > 0 → b > 0 →
  (S (a^(b+1)) = a^b ↔ (a, b) ∈ solution_set) := by sorry

end NUMINAMATH_CALUDE_digit_sum_power_equality_l3880_388095


namespace NUMINAMATH_CALUDE_opposite_numbers_absolute_value_l3880_388022

theorem opposite_numbers_absolute_value (a b : ℝ) : 
  a + b = 0 → |a - 2014 + b| = 2014 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_absolute_value_l3880_388022


namespace NUMINAMATH_CALUDE_square_EC_dot_ED_l3880_388039

/-- Square ABCD with side length 2 and E as midpoint of AB -/
structure Square2D where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  is_square : A.1 = B.1 ∧ A.2 = D.2 ∧ C.1 = D.1 ∧ C.2 = B.2
  side_length : ‖B - A‖ = 2
  E_midpoint : E = (A + B) / 2

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem square_EC_dot_ED (s : Square2D) :
  dot_product (s.C - s.E) (s.D - s.E) = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_EC_dot_ED_l3880_388039


namespace NUMINAMATH_CALUDE_cards_left_l3880_388025

def basketball_boxes : ℕ := 4
def basketball_cards_per_box : ℕ := 10
def baseball_boxes : ℕ := 5
def baseball_cards_per_box : ℕ := 8
def cards_given_away : ℕ := 58

theorem cards_left : 
  basketball_boxes * basketball_cards_per_box + 
  baseball_boxes * baseball_cards_per_box - 
  cards_given_away = 22 := by sorry

end NUMINAMATH_CALUDE_cards_left_l3880_388025


namespace NUMINAMATH_CALUDE_expression_simplification_l3880_388066

theorem expression_simplification (x : ℝ) : 7*x + 9 - 3*x + 15 * 2 = 4*x + 39 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3880_388066


namespace NUMINAMATH_CALUDE_bugs_eating_flowers_l3880_388065

/-- Given 2.5 bugs eating 4.5 flowers in total, the number of flowers consumed per bug is 1.8 -/
theorem bugs_eating_flowers (num_bugs : ℝ) (total_flowers : ℝ) 
    (h1 : num_bugs = 2.5) 
    (h2 : total_flowers = 4.5) : 
  total_flowers / num_bugs = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eating_flowers_l3880_388065


namespace NUMINAMATH_CALUDE_fault_line_current_movement_l3880_388024

/-- Represents the movement of a fault line over two years -/
structure FaultLineMovement where
  total : ℝ  -- Total movement over two years
  previous : ℝ  -- Movement in the previous year
  current : ℝ  -- Movement in the current year

/-- Theorem stating the movement of the fault line in the current year -/
theorem fault_line_current_movement (f : FaultLineMovement)
  (h1 : f.total = 6.5)
  (h2 : f.previous = 5.25)
  (h3 : f.total = f.previous + f.current) :
  f.current = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_current_movement_l3880_388024


namespace NUMINAMATH_CALUDE_range_of_a_l3880_388092

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → (a ∈ Set.Icc (-1) 1 ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3880_388092


namespace NUMINAMATH_CALUDE_rook_placement_on_colored_board_l3880_388006

theorem rook_placement_on_colored_board :
  let board_size : ℕ := 64
  let num_rooks : ℕ := 8
  let num_colors : ℕ := 32
  let cells_per_color : ℕ := 2

  let total_placements : ℕ := num_rooks.factorial
  let same_color_placements : ℕ := num_colors * (num_rooks - 2).factorial

  total_placements > same_color_placements :=
by sorry

end NUMINAMATH_CALUDE_rook_placement_on_colored_board_l3880_388006


namespace NUMINAMATH_CALUDE_moss_pollen_scientific_notation_l3880_388099

/-- The diameter of a moss flower's pollen in meters -/
def moss_pollen_diameter : ℝ := 0.0000084

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Theorem stating that the moss pollen diameter is equal to its scientific notation representation -/
theorem moss_pollen_scientific_notation :
  ∃ (sn : ScientificNotation), moss_pollen_diameter = sn.coefficient * (10 : ℝ) ^ sn.exponent ∧
  sn.coefficient = 8.4 ∧ sn.exponent = -6 := by
  sorry

end NUMINAMATH_CALUDE_moss_pollen_scientific_notation_l3880_388099


namespace NUMINAMATH_CALUDE_sum_of_products_l3880_388000

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 27)
  (eq2 : y^2 + y*z + z^2 = 25)
  (eq3 : z^2 + x*z + x^2 = 52) :
  x*y + y*z + x*z = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l3880_388000


namespace NUMINAMATH_CALUDE_distance_between_foci_l3880_388035

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 10)^2) = 24

-- Define the foci
def focus1 : ℝ × ℝ := (4, 3)
def focus2 : ℝ × ℝ := (-6, 10)

-- Theorem: The distance between foci is √149
theorem distance_between_foci :
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = Real.sqrt 149 := by
  sorry

#check distance_between_foci

end NUMINAMATH_CALUDE_distance_between_foci_l3880_388035


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3880_388031

/-- A point in a 2D Cartesian plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant in a Cartesian plane. -/
def FourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem stating that the point (1, -1) lies in the fourth quadrant. -/
theorem point_in_fourth_quadrant :
  let A : Point := ⟨1, -1⟩
  FourthQuadrant A := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3880_388031


namespace NUMINAMATH_CALUDE_peach_count_theorem_l3880_388002

def audrey_initial : ℕ := 26
def paul_initial : ℕ := 48
def maya_initial : ℕ := 57

def audrey_multiplier : ℕ := 3
def paul_multiplier : ℕ := 2
def maya_additional : ℕ := 20

def total_peaches : ℕ := 
  (audrey_initial + audrey_initial * audrey_multiplier) +
  (paul_initial + paul_initial * paul_multiplier) +
  (maya_initial + maya_additional)

theorem peach_count_theorem : total_peaches = 325 := by
  sorry

end NUMINAMATH_CALUDE_peach_count_theorem_l3880_388002


namespace NUMINAMATH_CALUDE_function_not_in_third_quadrant_l3880_388054

theorem function_not_in_third_quadrant
  (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : b > -1) :
  ¬∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ y = a^x + b :=
by sorry

end NUMINAMATH_CALUDE_function_not_in_third_quadrant_l3880_388054


namespace NUMINAMATH_CALUDE_functional_polynomial_form_l3880_388051

/-- A polynomial that satisfies the given functional equation. -/
structure FunctionalPolynomial where
  P : ℝ → ℝ
  nonzero : P ≠ 0
  satisfies_equation : ∀ x : ℝ, P (x^2 - 2*x) = (P (x - 2))^2

/-- The theorem stating the form of polynomials satisfying the functional equation. -/
theorem functional_polynomial_form (fp : FunctionalPolynomial) :
  ∃ n : ℕ, n > 0 ∧ ∀ x : ℝ, fp.P x = (x + 1)^n :=
sorry

end NUMINAMATH_CALUDE_functional_polynomial_form_l3880_388051


namespace NUMINAMATH_CALUDE_max_sum_hexagonal_prism_with_pyramid_l3880_388005

/-- Represents a three-dimensional geometric shape --/
structure Shape3D where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- A hexagonal prism --/
def hexagonal_prism : Shape3D :=
  { faces := 8, vertices := 12, edges := 18 }

/-- Adds a pyramid to a hexagonal face of the prism --/
def add_pyramid_to_hexagonal_face (s : Shape3D) : Shape3D :=
  { faces := s.faces + 5,
    vertices := s.vertices + 1,
    edges := s.edges + 6 }

/-- Adds a pyramid to a rectangular face of the prism --/
def add_pyramid_to_rectangular_face (s : Shape3D) : Shape3D :=
  { faces := s.faces + 3,
    vertices := s.vertices + 1,
    edges := s.edges + 4 }

/-- Calculates the sum of faces, vertices, and edges --/
def sum_features (s : Shape3D) : ℕ :=
  s.faces + s.vertices + s.edges

/-- Theorem: The maximum sum of exterior faces, vertices, and edges 
    when adding a pyramid to a hexagonal prism is 50 --/
theorem max_sum_hexagonal_prism_with_pyramid : 
  max 
    (sum_features (add_pyramid_to_hexagonal_face hexagonal_prism))
    (sum_features (add_pyramid_to_rectangular_face hexagonal_prism)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_hexagonal_prism_with_pyramid_l3880_388005


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l3880_388087

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  let d : ℝ := 10  -- diameter in meters
  let r : ℝ := d / 2  -- radius in meters
  let area : ℝ := π * r^2  -- area formula
  area = 25 * π :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l3880_388087


namespace NUMINAMATH_CALUDE_combined_salaries_l3880_388057

/-- Given the salary of E and the average salary of five individuals including E,
    calculate the combined salaries of the other four individuals. -/
theorem combined_salaries 
  (salary_E : ℕ) 
  (average_salary : ℕ) 
  (num_individuals : ℕ) :
  salary_E = 9000 →
  average_salary = 8800 →
  num_individuals = 5 →
  (num_individuals * average_salary) - salary_E = 35000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l3880_388057


namespace NUMINAMATH_CALUDE_at_least_one_root_exists_l3880_388023

theorem at_least_one_root_exists (c m a n : ℝ) : 
  (m^2 + 4*a*c ≥ 0) ∨ (n^2 - 4*a*c ≥ 0) := by sorry

end NUMINAMATH_CALUDE_at_least_one_root_exists_l3880_388023


namespace NUMINAMATH_CALUDE_exactly_one_zero_two_zeros_greater_than_neg_one_l3880_388012

-- Define the function f(x) in terms of m
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 3*m + 4

-- Theorem for condition 1
theorem exactly_one_zero (m : ℝ) :
  (∃! x, f m x = 0) ↔ (m = 4 ∨ m = -1) :=
sorry

-- Theorem for condition 2
theorem two_zeros_greater_than_neg_one (m : ℝ) :
  (∃ x y, x > -1 ∧ y > -1 ∧ x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ 
  (m > -5 ∧ m < -1) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_zero_two_zeros_greater_than_neg_one_l3880_388012


namespace NUMINAMATH_CALUDE_vote_intersection_l3880_388086

theorem vote_intersection (U A B : Finset Nat) : 
  Finset.card U = 250 →
  Finset.card A = 172 →
  Finset.card B = 143 →
  Finset.card (U \ (A ∪ B)) = 37 →
  Finset.card (A ∩ B) = 102 := by
sorry

end NUMINAMATH_CALUDE_vote_intersection_l3880_388086


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3880_388046

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 + 2 * Complex.I → z = -2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3880_388046


namespace NUMINAMATH_CALUDE_simplest_form_count_l3880_388083

-- Define the fractions
def fraction1 (a b : ℚ) : ℚ := b / (8 * a)
def fraction2 (a b : ℚ) : ℚ := (a + b) / (a - b)
def fraction3 (x y : ℚ) : ℚ := (x - y) / (x^2 - y^2)
def fraction4 (x y : ℚ) : ℚ := (x - y) / (x^2 + 2*x*y + y^2)

-- Define a function to check if a fraction is in simplest form
def isSimplestForm (f : ℚ → ℚ → ℚ) : Prop := 
  ∀ a b, a ≠ 0 → b ≠ 0 → (∃ c, f a b = c) → 
    ¬∃ d e, d ≠ 0 ∧ e ≠ 0 ∧ f (a*d) (b*e) = f a b

-- Theorem statement
theorem simplest_form_count : 
  (isSimplestForm fraction1) ∧ 
  (isSimplestForm fraction2) ∧ 
  ¬(isSimplestForm fraction3) ∧
  (isSimplestForm fraction4) := by sorry

end NUMINAMATH_CALUDE_simplest_form_count_l3880_388083


namespace NUMINAMATH_CALUDE_crew_member_count_l3880_388058

/-- The number of crew members working on all islands in a country -/
def total_crew_members (num_islands : ℕ) (ships_per_island : ℕ) (crew_per_ship : ℕ) : ℕ :=
  num_islands * ships_per_island * crew_per_ship

/-- Theorem stating the total number of crew members in the given scenario -/
theorem crew_member_count :
  total_crew_members 3 12 24 = 864 := by
  sorry

end NUMINAMATH_CALUDE_crew_member_count_l3880_388058


namespace NUMINAMATH_CALUDE_time_after_3250_minutes_l3880_388016

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting date and time -/
def startDateTime : DateTime :=
  { year := 2020, month := 1, day := 1, hour := 3, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3250

/-- The resulting date and time -/
def resultDateTime : DateTime :=
  { year := 2020, month := 1, day := 3, hour := 9, minute := 10 }

theorem time_after_3250_minutes :
  addMinutes startDateTime minutesToAdd = resultDateTime :=
sorry

end NUMINAMATH_CALUDE_time_after_3250_minutes_l3880_388016


namespace NUMINAMATH_CALUDE_square_sum_xy_l3880_388027

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : 1 / x^2 + 1 / y^2 = 7)
  (h3 : x - y = Real.sqrt 10) :
  (x + y)^2 = 264 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l3880_388027


namespace NUMINAMATH_CALUDE_cows_eating_husk_l3880_388042

/-- The number of bags of husk eaten by a group of cows in 30 days -/
def bags_eaten (num_cows : ℕ) (bags_per_cow : ℕ) : ℕ :=
  num_cows * bags_per_cow

/-- Theorem: 30 cows eat 30 bags of husk in 30 days -/
theorem cows_eating_husk :
  bags_eaten 30 1 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cows_eating_husk_l3880_388042


namespace NUMINAMATH_CALUDE_function_range_theorem_l3880_388004

/-- Given a function f(x) = |2x - 1| + |x - 2a|, if for all x ∈ [1, 2], f(x) ≤ 4,
    then the range of real values for a is [1/2, 3/2]. -/
theorem function_range_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f x = |2 * x - 1| + |x - 2 * a|) →
  (∀ x ∈ Set.Icc 1 2, f x ≤ 4) →
  a ∈ Set.Icc (1/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_function_range_theorem_l3880_388004


namespace NUMINAMATH_CALUDE_tan_alpha_equals_three_implies_ratio_equals_five_l3880_388068

theorem tan_alpha_equals_three_implies_ratio_equals_five (α : Real) 
  (h : Real.tan α = 3) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_three_implies_ratio_equals_five_l3880_388068


namespace NUMINAMATH_CALUDE_max_tied_teams_seven_team_tournament_l3880_388034

/-- Represents a round-robin tournament --/
structure Tournament :=
  (num_teams : Nat)
  (no_draws : Bool)
  (round_robin : Bool)

/-- Calculates the total number of games in a round-robin tournament --/
def total_games (t : Tournament) : Nat :=
  (t.num_teams * (t.num_teams - 1)) / 2

/-- Represents the maximum number of teams that can be tied for the most wins --/
def max_tied_teams (t : Tournament) : Nat :=
  sorry

/-- The theorem to be proved --/
theorem max_tied_teams_seven_team_tournament :
  ∀ t : Tournament, t.num_teams = 7 → t.no_draws = true → t.round_robin = true →
  max_tied_teams t = 6 :=
sorry

end NUMINAMATH_CALUDE_max_tied_teams_seven_team_tournament_l3880_388034
