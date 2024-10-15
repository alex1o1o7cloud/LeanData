import Mathlib

namespace NUMINAMATH_CALUDE_total_candy_cases_l314_31427

/-- The Sweet Shop's candy inventory -/
structure CandyInventory where
  chocolate_cases : ℕ
  lollipop_cases : ℕ

/-- The total number of candy cases in the inventory -/
def total_cases (inventory : CandyInventory) : ℕ :=
  inventory.chocolate_cases + inventory.lollipop_cases

/-- Theorem: The total number of candy cases is 80 -/
theorem total_candy_cases :
  ∃ (inventory : CandyInventory),
    inventory.chocolate_cases = 25 ∧
    inventory.lollipop_cases = 55 ∧
    total_cases inventory = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_candy_cases_l314_31427


namespace NUMINAMATH_CALUDE_card_problem_l314_31435

theorem card_problem (w x y z : ℝ) : 
  x = w / 2 →
  y = w + x →
  z = 400 →
  w + x + y + z = 1000 →
  w = 200 := by
sorry

end NUMINAMATH_CALUDE_card_problem_l314_31435


namespace NUMINAMATH_CALUDE_sector_central_angle_l314_31457

theorem sector_central_angle (circumference area : ℝ) (h_circ : circumference = 6) (h_area : area = 2) :
  ∃ (r l : ℝ), r > 0 ∧ l > 0 ∧ 2 * r + l = circumference ∧ (1 / 2) * r * l = area ∧
  (l / r = 1 ∨ l / r = 4) :=
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l314_31457


namespace NUMINAMATH_CALUDE_maria_car_trip_l314_31477

theorem maria_car_trip (D : ℝ) : 
  (D / 2 + (D / 2) / 4 + 150 = D) → D = 400 := by sorry

end NUMINAMATH_CALUDE_maria_car_trip_l314_31477


namespace NUMINAMATH_CALUDE_sum_of_units_digits_equals_zero_l314_31421

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the problem
theorem sum_of_units_digits_equals_zero :
  (unitsDigit (17 * 34) + unitsDigit (19 * 28)) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_units_digits_equals_zero_l314_31421


namespace NUMINAMATH_CALUDE_xiao_hua_seat_l314_31488

structure Classroom where
  rows : Nat
  columns : Nat

structure Seat where
  row : Nat
  column : Nat

def is_valid_seat (c : Classroom) (s : Seat) : Prop :=
  s.row ≤ c.rows ∧ s.column ≤ c.columns

theorem xiao_hua_seat (c : Classroom) (s : Seat) :
  c.rows = 7 →
  c.columns = 8 →
  is_valid_seat c s →
  s.row = 5 →
  s.column = 2 →
  s = ⟨5, 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_xiao_hua_seat_l314_31488


namespace NUMINAMATH_CALUDE_product_sum_squares_l314_31491

theorem product_sum_squares (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 + (a*d - b*c)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_squares_l314_31491


namespace NUMINAMATH_CALUDE_data_set_property_l314_31475

theorem data_set_property (m n : ℝ) : 
  (m + n + 9 + 8 + 10) / 5 = 9 →
  ((m^2 + n^2 + 9^2 + 8^2 + 10^2) / 5) - 9^2 = 2 →
  |m - n| = 4 :=
by sorry

end NUMINAMATH_CALUDE_data_set_property_l314_31475


namespace NUMINAMATH_CALUDE_dog_treat_cost_theorem_l314_31467

/-- Calculates the total cost of dog treats for a month -/
def total_treat_cost (treats_per_day : ℕ) (cost_per_treat : ℚ) (days_in_month : ℕ) : ℚ :=
  (treats_per_day * days_in_month : ℚ) * cost_per_treat

/-- Proves that the total cost of dog treats for a month with given parameters is $6 -/
theorem dog_treat_cost_theorem (treats_per_day : ℕ) (cost_per_treat : ℚ) (days_in_month : ℕ)
  (h1 : treats_per_day = 2)
  (h2 : cost_per_treat = 1/10)
  (h3 : days_in_month = 30) :
  total_treat_cost treats_per_day cost_per_treat days_in_month = 6 := by
  sorry

end NUMINAMATH_CALUDE_dog_treat_cost_theorem_l314_31467


namespace NUMINAMATH_CALUDE_square_area_relation_l314_31414

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I : ℝ := 2*a + 3*b
  let area_I : ℝ := (diagonal_I^2) / 2
  let area_II : ℝ := area_I^3
  area_II = (diagonal_I^6) / 8 := by
sorry

end NUMINAMATH_CALUDE_square_area_relation_l314_31414


namespace NUMINAMATH_CALUDE_special_function_properties_l314_31452

/-- An increasing function f: ℝ₊ → ℝ₊ satisfying f(xy) = f(x)f(y) for all x, y > 0, and f(2) = 4 -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 0 → y > 0 → f (x * y) = f x * f y) ∧
  (∀ x y, x > y → x > 0 → y > 0 → f x > f y) ∧
  f 2 = 4

theorem special_function_properties (f : ℝ → ℝ) (h : SpecialFunction f) :
  f 1 = 1 ∧ f 8 = 64 ∧ Set.Ioo 3 (7/2) = {x | 16 * f (1 / (x - 3)) ≥ f (2 * x + 1)} := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l314_31452


namespace NUMINAMATH_CALUDE_unique_divisor_function_exists_l314_31426

open Nat

/-- The divisor function τ(n) counts the number of positive divisors of n. -/
noncomputable def tau (n : ℕ) : ℕ := (divisors n).card

/-- 
Given a finite set of natural numbers, there exists a number x such that 
the divisor function τ applied to the product of x and any element of the set 
yields a unique result for each element of the set.
-/
theorem unique_divisor_function_exists (S : Finset ℕ) : 
  ∃ x : ℕ, ∀ s₁ s₂ : ℕ, s₁ ∈ S → s₂ ∈ S → s₁ ≠ s₂ → tau (s₁ * x) ≠ tau (s₂ * x) := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_function_exists_l314_31426


namespace NUMINAMATH_CALUDE_power_of_power_l314_31416

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l314_31416


namespace NUMINAMATH_CALUDE_max_profit_theorem_l314_31469

/-- Represents the online store's sales and profit model -/
structure OnlineStore where
  initialPrice : ℕ
  initialSales : ℕ
  cost : ℕ
  salesIncrease : ℕ
  priceReduction : ℕ

/-- Calculates the monthly sales volume based on the price -/
def monthlySales (store : OnlineStore) (price : ℕ) : ℤ :=
  store.initialSales + store.salesIncrease * (store.initialPrice - price)

/-- Calculates the monthly profit based on the price -/
def monthlyProfit (store : OnlineStore) (price : ℕ) : ℤ :=
  (price - store.cost) * (monthlySales store price)

/-- Theorem stating the maximum profit and optimal price reduction -/
theorem max_profit_theorem (store : OnlineStore) :
  store.initialPrice = 80 ∧
  store.initialSales = 100 ∧
  store.cost = 40 ∧
  store.salesIncrease = 5 →
  ∃ (optimalReduction : ℕ),
    optimalReduction = 10 ∧
    monthlyProfit store (store.initialPrice - optimalReduction) = 4500 ∧
    ∀ (price : ℕ), monthlyProfit store price ≤ 4500 :=
by sorry

#check max_profit_theorem

end NUMINAMATH_CALUDE_max_profit_theorem_l314_31469


namespace NUMINAMATH_CALUDE_road_length_l314_31431

theorem road_length (trees : ℕ) (tree_space : ℕ) (between_space : ℕ) : 
  trees = 13 → tree_space = 1 → between_space = 12 → 
  trees * tree_space + (trees - 1) * between_space = 157 := by
  sorry

end NUMINAMATH_CALUDE_road_length_l314_31431


namespace NUMINAMATH_CALUDE_min_removal_for_given_structure_l314_31497

/-- Represents the structure of triangles made with toothpicks -/
structure TriangleStructure where
  totalToothpicks : ℕ
  baseTriangles : ℕ
  rows : ℕ

/-- Calculates the number of toothpicks needed to be removed to eliminate all triangles -/
def minRemovalCount (ts : TriangleStructure) : ℕ :=
  ts.rows

/-- Theorem stating that for the given structure, 5 toothpicks need to be removed -/
theorem min_removal_for_given_structure :
  let ts : TriangleStructure := {
    totalToothpicks := 50,
    baseTriangles := 5,
    rows := 5
  }
  minRemovalCount ts = 5 := by
  sorry

#check min_removal_for_given_structure

end NUMINAMATH_CALUDE_min_removal_for_given_structure_l314_31497


namespace NUMINAMATH_CALUDE_christine_speed_l314_31422

/-- Given a distance of 80 miles traveled in 4 hours, prove that the speed is 20 miles per hour. -/
theorem christine_speed (distance : ℝ) (time : ℝ) (h1 : distance = 80) (h2 : time = 4) :
  distance / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_christine_speed_l314_31422


namespace NUMINAMATH_CALUDE_remaining_numbers_are_even_l314_31438

def last_digit (n : ℕ) : ℕ := n % 10
def second_last_digit (n : ℕ) : ℕ := (n / 10) % 10

def is_removed (n : ℕ) : Prop :=
  (last_digit n % 2 = 1 ∧ second_last_digit n % 2 = 0) ∨
  (last_digit n % 2 = 1 ∧ last_digit n % 3 ≠ 0) ∨
  (second_last_digit n % 2 = 1 ∧ n % 3 = 0)

theorem remaining_numbers_are_even (n : ℕ) :
  ¬(is_removed n) → Even n :=
by sorry

end NUMINAMATH_CALUDE_remaining_numbers_are_even_l314_31438


namespace NUMINAMATH_CALUDE_dress_price_difference_l314_31408

theorem dress_price_difference : 
  let original_price := 78.2 / 0.85
  let discounted_price := 78.2
  let final_price := discounted_price * 1.25
  final_price - original_price = 5.75 := by
sorry

end NUMINAMATH_CALUDE_dress_price_difference_l314_31408


namespace NUMINAMATH_CALUDE_unique_positive_number_l314_31482

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x - 4 = 21 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l314_31482


namespace NUMINAMATH_CALUDE_annual_population_change_l314_31409

def town_population (initial_pop : ℕ) (new_people : ℕ) (moved_out : ℕ) (years : ℕ) (final_pop : ℕ) : ℤ :=
  let pop_after_changes : ℤ := initial_pop + new_people - moved_out
  let total_change : ℤ := pop_after_changes - final_pop
  total_change / years

theorem annual_population_change :
  town_population 780 100 400 4 60 = -105 :=
sorry

end NUMINAMATH_CALUDE_annual_population_change_l314_31409


namespace NUMINAMATH_CALUDE_lucy_has_19_snowballs_l314_31439

-- Define the number of snowballs Charlie and Lucy have
def charlie_snowballs : ℕ := 50
def lucy_snowballs : ℕ := charlie_snowballs - 31

-- Theorem statement
theorem lucy_has_19_snowballs : lucy_snowballs = 19 := by
  sorry

end NUMINAMATH_CALUDE_lucy_has_19_snowballs_l314_31439


namespace NUMINAMATH_CALUDE_employee_transfer_theorem_l314_31440

/-- Represents the number of employees transferred to the tertiary industry -/
def x : ℕ+ := sorry

/-- Represents the profit multiplier for transferred employees -/
def a : ℝ := sorry

/-- The total number of employees -/
def total_employees : ℕ := 1000

/-- The initial average profit per employee per year in ten thousands of yuan -/
def initial_profit : ℝ := 10

/-- The profit increase rate for remaining employees -/
def profit_increase_rate : ℝ := 0.002

/-- The condition that the total profit after transfer is not less than the initial total profit -/
def profit_condition (x : ℕ+) : Prop :=
  (initial_profit * (total_employees - x) * (1 + profit_increase_rate * x) ≥ initial_profit * total_employees)

/-- The condition that the profit from transferred employees is not more than the profit from remaining employees -/
def transfer_condition (x : ℕ+) (a : ℝ) : Prop :=
  (initial_profit * a * x ≤ initial_profit * (total_employees - x) * (1 + profit_increase_rate * x))

theorem employee_transfer_theorem :
  (∃ max_x : ℕ+, max_x = 500 ∧ 
    (∀ y : ℕ+, y > max_x → ¬profit_condition y) ∧
    (∀ y : ℕ+, y ≤ max_x → profit_condition y)) ∧
  (∀ x : ℕ+, x ≤ 500 →
    (∀ a : ℝ, 0 < a ∧ a ≤ 5 → transfer_condition x a) ∧
    (∀ a : ℝ, a > 5 → ¬transfer_condition x a)) :=
sorry

end NUMINAMATH_CALUDE_employee_transfer_theorem_l314_31440


namespace NUMINAMATH_CALUDE_original_equals_scientific_l314_31458

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℝ := 0.056

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 5.6
    exponent := -2
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  original_number = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l314_31458


namespace NUMINAMATH_CALUDE_art_class_price_l314_31418

/-- Represents the price of Claudia's one-hour art class -/
def class_price : ℝ := 10

/-- Number of kids attending Saturday's class -/
def saturday_attendance : ℕ := 20

/-- Number of kids attending Sunday's class -/
def sunday_attendance : ℕ := saturday_attendance / 2

/-- Total earnings for both days -/
def total_earnings : ℝ := 300

theorem art_class_price :
  class_price * (saturday_attendance + sunday_attendance) = total_earnings :=
sorry

end NUMINAMATH_CALUDE_art_class_price_l314_31418


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l314_31493

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l314_31493


namespace NUMINAMATH_CALUDE_business_trip_distance_l314_31407

/-- Calculates the total distance traveled during a business trip -/
theorem business_trip_distance (total_duration : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_duration = 8 →
  speed1 = 70 →
  speed2 = 85 →
  (total_duration / 2 * speed1) + (total_duration / 2 * speed2) = 620 := by
  sorry

#check business_trip_distance

end NUMINAMATH_CALUDE_business_trip_distance_l314_31407


namespace NUMINAMATH_CALUDE_parabola_equation_l314_31478

/-- A parabola with vertex at the origin -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 = kx -/
  k : ℝ
  /-- The focus of the parabola -/
  focus : ℝ × ℝ

/-- The condition that the focus is on the x-axis -/
def focus_on_x_axis (p : Parabola) : Prop :=
  p.focus.2 = 0

/-- The condition that a perpendicular line from the origin to a line passing through the focus has its foot at (2, 1) -/
def perpendicular_foot_condition (p : Parabola) : Prop :=
  ∃ (m : ℝ), m * p.focus.1 = p.focus.2 ∧ 2 * m = 1

/-- The theorem stating that if the two conditions are met, the parabola's equation is y^2 = 10x -/
theorem parabola_equation (p : Parabola) :
  focus_on_x_axis p → perpendicular_foot_condition p → p.k = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l314_31478


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l314_31415

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 1)^5 = a₅*(x + 1)^5 + a₄*(x + 1)^4 + a₃*(x + 1)^3 + a₂*(x + 1)^2 + a₁*(x + 1) + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l314_31415


namespace NUMINAMATH_CALUDE_subtracted_value_l314_31444

theorem subtracted_value (x y : ℝ) : 
  (x - 5) / 7 = 7 → 
  (x - y) / 10 = 2 → 
  y = 34 := by
sorry

end NUMINAMATH_CALUDE_subtracted_value_l314_31444


namespace NUMINAMATH_CALUDE_water_level_rise_l314_31447

/-- Calculates the rise in water level when a cube is immersed in a rectangular vessel. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length vessel_width : ℝ) : 
  cube_edge = 5 →
  vessel_length = 10 →
  vessel_width = 5 →
  (cube_edge^3) / (vessel_length * vessel_width) = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_water_level_rise_l314_31447


namespace NUMINAMATH_CALUDE_halloween_candy_count_l314_31476

/-- Calculates the final candy count given initial count, eaten count, and received count. -/
def finalCandyCount (initial eaten received : ℕ) : ℕ :=
  initial - eaten + received

/-- Theorem stating that given the specific values from the problem, 
    the final candy count is 62. -/
theorem halloween_candy_count : 
  finalCandyCount 47 25 40 = 62 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_count_l314_31476


namespace NUMINAMATH_CALUDE_concentric_circles_radii_product_l314_31428

theorem concentric_circles_radii_product (r₁ r₂ r₃ : ℝ) : 
  r₁ = 2 →
  (r₂^2 - r₁^2 = r₁^2) →
  (r₃^2 - r₂^2 = r₁^2) →
  (r₁ * r₂ * r₃)^2 = 384 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_product_l314_31428


namespace NUMINAMATH_CALUDE_fraction_equality_l314_31463

theorem fraction_equality : (1 : ℚ) / 2 + (1 : ℚ) / 4 = 9 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l314_31463


namespace NUMINAMATH_CALUDE_second_number_value_l314_31489

theorem second_number_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b = 1.2 * a) (h4 : a / b = 5 / 6) : b = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l314_31489


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l314_31403

theorem greatest_integer_inequality (y : ℤ) : (8 : ℚ) / 11 > (y : ℚ) / 15 ↔ y ≤ 10 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l314_31403


namespace NUMINAMATH_CALUDE_sqrt_2023_between_40_and_45_l314_31479

theorem sqrt_2023_between_40_and_45 : 40 < Real.sqrt 2023 ∧ Real.sqrt 2023 < 45 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2023_between_40_and_45_l314_31479


namespace NUMINAMATH_CALUDE_flower_count_proof_l314_31450

theorem flower_count_proof (total : ℕ) (red green blue yellow purple orange : ℕ) : 
  total = 180 →
  red = (30 * total) / 100 →
  green = (10 * total) / 100 →
  blue = green / 2 →
  yellow = red + 5 →
  3 * purple = 7 * orange →
  red + green + blue + yellow + purple + orange = total →
  red = 54 ∧ green = 18 ∧ blue = 9 ∧ yellow = 59 ∧ purple = 12 ∧ orange = 28 :=
by sorry

end NUMINAMATH_CALUDE_flower_count_proof_l314_31450


namespace NUMINAMATH_CALUDE_complex_magnitude_l314_31442

theorem complex_magnitude (r s : ℝ) (z : ℂ) 
  (h1 : |r| < 4) 
  (h2 : s ≠ 0) 
  (h3 : s * z + 1 / z = r) : 
  Complex.abs z = Real.sqrt (2 * (r^2 - 2*s) + 2*r * Real.sqrt (r^2 - 4*s)) / (2 * |s|) :=
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l314_31442


namespace NUMINAMATH_CALUDE_parabola_coefficient_l314_31471

/-- A parabola passing through a specific point -/
def parabola_through_point (a : ℝ) (x y : ℝ) : Prop :=
  a ≠ 0 ∧ y = a * x^2

/-- Theorem: The parabola y = ax^2 passing through (2, -8) has a = -2 -/
theorem parabola_coefficient :
  ∀ a : ℝ, parabola_through_point a 2 (-8) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l314_31471


namespace NUMINAMATH_CALUDE_focus_of_specific_parabola_l314_31406

/-- A parabola is defined by its quadratic equation coefficients -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (x, y) -/
def focus (p : Parabola) : ℝ × ℝ :=
  sorry

/-- Theorem: The focus of the parabola y = 9x^2 + 6x - 2 is at (-1/3, -107/36) -/
theorem focus_of_specific_parabola :
  let p : Parabola := { a := 9, b := 6, c := -2 }
  focus p = (-1/3, -107/36) := by
  sorry

end NUMINAMATH_CALUDE_focus_of_specific_parabola_l314_31406


namespace NUMINAMATH_CALUDE_largest_domain_of_g_l314_31425

def is_valid_domain (S : Set ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x ∈ S, x^2 ∈ S ∧ 1/x^2 ∈ S ∧ g x + g (1/x^2) = x^2

theorem largest_domain_of_g :
  ∃! S : Set ℝ, is_valid_domain S g ∧
    ∀ T : Set ℝ, is_valid_domain T g → T ⊆ S :=
by
  sorry

#check largest_domain_of_g

end NUMINAMATH_CALUDE_largest_domain_of_g_l314_31425


namespace NUMINAMATH_CALUDE_triangle_free_edge_bound_l314_31449

/-- A graph with n vertices and k edges, where no three edges form a triangle -/
structure TriangleFreeGraph where
  n : ℕ  -- number of vertices
  k : ℕ  -- number of edges
  no_triangle : True  -- represents the condition that no three edges form a triangle

/-- Theorem: In a triangle-free graph, the number of edges is at most ⌊n²/4⌋ -/
theorem triangle_free_edge_bound (G : TriangleFreeGraph) : G.k ≤ (G.n^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_free_edge_bound_l314_31449


namespace NUMINAMATH_CALUDE_r_div_p_equals_1100_l314_31486

/-- The number of cards in the box -/
def total_cards : ℕ := 60

/-- The number of different numbers on the cards -/
def distinct_numbers : ℕ := 12

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def drawn_cards : ℕ := 5

/-- The probability of drawing five cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / Nat.choose total_cards drawn_cards

/-- The probability of drawing three cards with one number and two with another -/
def r : ℚ := (13200 : ℚ) / Nat.choose total_cards drawn_cards

/-- Theorem stating the ratio of r to p -/
theorem r_div_p_equals_1100 : r / p = 1100 := by sorry

end NUMINAMATH_CALUDE_r_div_p_equals_1100_l314_31486


namespace NUMINAMATH_CALUDE_friendly_number_F_formula_max_friendly_N_l314_31424

def is_friendly_number (M : ℕ) : Prop :=
  ∃ (a b c d : ℕ), M = 1000 * a + 100 * b + 10 * c + d ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ a - b = c - d

def F (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  100 * a - 100 * b - 10 * b + c - d

theorem friendly_number_F_formula (M : ℕ) (h : is_friendly_number M) :
  F M = 100 * (M / 1000) - 110 * ((M / 100) % 10) + (M / 10) % 10 - M % 10 :=
sorry

def N (x y m n : ℕ) : ℕ := 1000 * x + 100 * y + 30 * m + n + 1001

theorem max_friendly_N (x y m n : ℕ) 
  (h1 : 0 ≤ y ∧ y < x ∧ x ≤ 8) 
  (h2 : 0 ≤ m ∧ m ≤ 3) 
  (h3 : 0 ≤ n ∧ n ≤ 8) 
  (h4 : is_friendly_number (N x y m n)) 
  (h5 : F (N x y m n) % 5 = 1) :
  N x y m n ≤ 9696 :=
sorry

end NUMINAMATH_CALUDE_friendly_number_F_formula_max_friendly_N_l314_31424


namespace NUMINAMATH_CALUDE_fruit_mix_grapes_l314_31419

theorem fruit_mix_grapes (b r g c : ℚ) : 
  b + r + g + c = 400 →
  r = 3 * b →
  g = 2 * c →
  c = 5 * r →
  g = 12000 / 49 := by
sorry

end NUMINAMATH_CALUDE_fruit_mix_grapes_l314_31419


namespace NUMINAMATH_CALUDE_base_seven_divisibility_l314_31446

theorem base_seven_divisibility (y : ℕ) : 
  y ≤ 6 → (∃! y, (5 * 7^2 + y * 7 + 2) % 19 = 0 ∧ y ≤ 6) → y = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_divisibility_l314_31446


namespace NUMINAMATH_CALUDE_jenny_activities_lcm_l314_31448

theorem jenny_activities_lcm : Nat.lcm (Nat.lcm 6 12) 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jenny_activities_lcm_l314_31448


namespace NUMINAMATH_CALUDE_parabola_sum_l314_31492

/-- A parabola with equation y = px^2 + qx + r -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.yCoord (para : Parabola) (x : ℝ) : ℝ :=
  para.p * x^2 + para.q * x + para.r

theorem parabola_sum (para : Parabola) :
  para.yCoord 3 = 2 →   -- Vertex at (3, 2)
  para.yCoord 1 = 6 →   -- Passes through (1, 6)
  para.p + para.q + para.r = 6 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l314_31492


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_inradius_and_area_l314_31411

/-- Given a triangle with inradius 2.0 cm and area 28 cm², its perimeter is 28 cm. -/
theorem triangle_perimeter_from_inradius_and_area :
  ∀ (p : ℝ), 
    (2.0 : ℝ) * p / 2 = 28 →
    p = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_inradius_and_area_l314_31411


namespace NUMINAMATH_CALUDE_girls_in_school_l314_31453

theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girl_boy_diff : ℕ) :
  total_students = 1600 →
  sample_size = 200 →
  girl_boy_diff = 20 →
  ∃ (girls : ℕ) (boys : ℕ),
    girls + boys = total_students ∧
    girls * sample_size = (total_students - girls) * (sample_size - girl_boy_diff) ∧
    girls = 720 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_school_l314_31453


namespace NUMINAMATH_CALUDE_largest_odd_equal_cost_l314_31485

/-- Calculates the sum of digits in decimal representation -/
def sumDigitsDecimal (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumDigitsDecimal (n / 10)

/-- Calculates the sum of digits in binary representation with two trailing zeros -/
def sumDigitsBinary (n : Nat) : Nat :=
  if n < 4 then 0 else (n % 2) + sumDigitsBinary (n / 2)

/-- Checks if a number is odd -/
def isOdd (n : Nat) : Prop := n % 2 = 1

/-- Theorem statement -/
theorem largest_odd_equal_cost :
  ∃ (n : Nat), n < 2000 ∧ isOdd n ∧
    sumDigitsDecimal n = sumDigitsBinary n ∧
    ∀ (m : Nat), m < 2000 ∧ isOdd m ∧ sumDigitsDecimal m = sumDigitsBinary m → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_odd_equal_cost_l314_31485


namespace NUMINAMATH_CALUDE_set_relationships_evaluation_l314_31462

theorem set_relationships_evaluation :
  let s1 : Set (Set ℕ) := {{0}, {2, 3, 4}}
  let s2 : Set ℕ := {0}
  let s3 : Set ℤ := {-1, 0, 1}
  let s4 : Set ℤ := {0, -1, 1}
  ({0} ∈ s1) = false ∧
  (∅ ⊆ s2) = true ∧
  (s3 = s4) = true ∧
  (0 ∈ (∅ : Set ℕ)) = false :=
by sorry

end NUMINAMATH_CALUDE_set_relationships_evaluation_l314_31462


namespace NUMINAMATH_CALUDE_intersection_sum_l314_31412

theorem intersection_sum (c d : ℝ) : 
  (2 = (1/5) * 3 + c) → 
  (3 = (1/5) * 2 + d) → 
  c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l314_31412


namespace NUMINAMATH_CALUDE_star_polygon_points_l314_31405

/-- A regular star polygon with n points, where each point has two types of angles -/
structure StarPolygon where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ
  h1 : angle_A = angle_B - 15
  h2 : 0 < n

/-- The sum of all exterior angles in a polygon is 360° -/
axiom sum_of_exterior_angles : ∀ (p : StarPolygon), p.n * (p.angle_B - p.angle_A) = 360

/-- The number of points in the star polygon is 24 -/
theorem star_polygon_points (p : StarPolygon) : p.n = 24 := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_points_l314_31405


namespace NUMINAMATH_CALUDE_profit_percentage_is_30_percent_l314_31474

def cost_per_dog : ℝ := 1000
def selling_price_two_dogs : ℝ := 2600

theorem profit_percentage_is_30_percent :
  let cost_two_dogs := 2 * cost_per_dog
  let profit := selling_price_two_dogs - cost_two_dogs
  let profit_percentage := (profit / cost_two_dogs) * 100
  profit_percentage = 30 := by sorry

end NUMINAMATH_CALUDE_profit_percentage_is_30_percent_l314_31474


namespace NUMINAMATH_CALUDE_polynomial_has_non_real_root_l314_31490

def is_valid_polynomial (P : Polynomial ℝ) : Prop :=
  (P.degree ≥ 4) ∧
  (∀ i, P.coeff i ∈ ({-1, 0, 1} : Set ℝ)) ∧
  (P.eval 0 ≠ 0)

theorem polynomial_has_non_real_root (P : Polynomial ℝ) 
  (h : is_valid_polynomial P) : 
  ∃ z : ℂ, z.im ≠ 0 ∧ P.eval (z.re : ℝ) = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_has_non_real_root_l314_31490


namespace NUMINAMATH_CALUDE_carpenter_logs_l314_31461

/-- Proves that the carpenter currently has 8 logs given the conditions of the problem -/
theorem carpenter_logs :
  ∀ (total_woodblocks : ℕ) (woodblocks_per_log : ℕ) (additional_logs_needed : ℕ),
    total_woodblocks = 80 →
    woodblocks_per_log = 5 →
    additional_logs_needed = 8 →
    (total_woodblocks - additional_logs_needed * woodblocks_per_log) / woodblocks_per_log = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_carpenter_logs_l314_31461


namespace NUMINAMATH_CALUDE_ratio_problem_l314_31464

theorem ratio_problem (x : ℝ) : (20 / 1 = x / 10) → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l314_31464


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l314_31443

theorem ellipse_foci_distance (a b : ℝ) (h1 : a = 6) (h2 : b = 2) :
  2 * Real.sqrt (a^2 - b^2) = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l314_31443


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l314_31433

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + 2*y^2)).sqrt) / (x*y) ≥ 2 + Real.sqrt 2 :=
sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (((x^2 + y^2) * (4*x^2 + 2*y^2)).sqrt) / (x*y) = 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l314_31433


namespace NUMINAMATH_CALUDE_inequality_solution_set_l314_31441

theorem inequality_solution_set :
  {x : ℝ | x * |x - 1| > 0} = (Set.Ioo 0 1) ∪ (Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l314_31441


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l314_31459

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Define the condition for a point to be in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem linear_function_not_in_third_quadrant 
  (k b : ℝ) 
  (h : ∀ x y : ℝ, y = linear_function k b x → ¬in_third_quadrant x y) : 
  k < 0 ∧ b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l314_31459


namespace NUMINAMATH_CALUDE_expand_binomials_l314_31436

theorem expand_binomials (x : ℝ) : (2*x + 3) * (4*x - 7) = 8*x^2 - 2*x - 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l314_31436


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_product_of_fractions_l314_31456

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem product_of_fractions :
  (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_product_of_fractions_l314_31456


namespace NUMINAMATH_CALUDE_birdseed_mixture_cost_per_pound_l314_31429

theorem birdseed_mixture_cost_per_pound
  (millet_weight : ℝ)
  (millet_cost_per_lb : ℝ)
  (sunflower_weight : ℝ)
  (sunflower_cost_per_lb : ℝ)
  (h1 : millet_weight = 100)
  (h2 : millet_cost_per_lb = 0.60)
  (h3 : sunflower_weight = 25)
  (h4 : sunflower_cost_per_lb = 1.10) :
  (millet_weight * millet_cost_per_lb + sunflower_weight * sunflower_cost_per_lb) /
  (millet_weight + sunflower_weight) = 0.70 := by
  sorry

#check birdseed_mixture_cost_per_pound

end NUMINAMATH_CALUDE_birdseed_mixture_cost_per_pound_l314_31429


namespace NUMINAMATH_CALUDE_candy_count_l314_31430

theorem candy_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 3409) 
  (h2 : red = 145) 
  (h3 : total = red + blue) : blue = 3264 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l314_31430


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l314_31460

/-- Given vectors a and b, where a is parallel to b, prove the minimum value of 1/m + 8/n is 9/2 -/
theorem min_value_parallel_vectors (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4 - n, 2)
  (∃ (k : ℝ), a = k • b) → 
  (∀ (x y : ℝ), x > 0 → y > 0 → 2 * x + y = 4 → 1/x + 8/y ≥ 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l314_31460


namespace NUMINAMATH_CALUDE_correct_sunset_time_l314_31472

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define addition for Time
def Time.add (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hours * 60 + t1.minutes + t2.hours * 60 + t2.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Convert 24-hour format to 12-hour format
def to12HourFormat (t : Time) : Time :=
  if t.hours ≥ 12 then
    { hours := if t.hours = 12 then 12 else t.hours - 12, minutes := t.minutes }
  else
    { hours := if t.hours = 0 then 12 else t.hours, minutes := t.minutes }

theorem correct_sunset_time :
  let sunrise : Time := { hours := 6, minutes := 57 }
  let daylight : Time := { hours := 10, minutes := 24 }
  let sunset := to12HourFormat (Time.add sunrise daylight)
  sunset = { hours := 5, minutes := 21 } := by sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l314_31472


namespace NUMINAMATH_CALUDE_nonreal_cube_root_sum_l314_31484

/-- Given that ω is a nonreal root of x^3 = 1, prove that 
    (2 - 2ω + 2ω^2)^3 + (2 + 2ω - 2ω^2)^3 = 0 -/
theorem nonreal_cube_root_sum (ω : ℂ) 
  (h1 : ω^3 = 1) 
  (h2 : ω ≠ 1) : 
  (2 - 2*ω + 2*ω^2)^3 + (2 + 2*ω - 2*ω^2)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nonreal_cube_root_sum_l314_31484


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l314_31410

/-- Given a line segment with midpoint (2, -3) and one endpoint (3, 1),
    prove that the other endpoint is (1, -7) -/
theorem line_segment_endpoint (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁ = 3 ∧ y₁ = 1) →  -- One endpoint is (3, 1)
  ((x₁ + x₂) / 2 = 2 ∧ (y₁ + y₂) / 2 = -3) →  -- Midpoint is (2, -3)
  x₂ = 1 ∧ y₂ = -7  -- Other endpoint is (1, -7)
  := by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l314_31410


namespace NUMINAMATH_CALUDE_jorge_goals_this_season_l314_31455

/-- Given that Jorge scored 156 goals last season and his total goals are 343,
    prove that the number of goals he scored this season is 343 - 156. -/
theorem jorge_goals_this_season 
  (goals_last_season : ℕ) 
  (total_goals : ℕ) 
  (h1 : goals_last_season = 156) 
  (h2 : total_goals = 343) : 
  total_goals - goals_last_season = 343 - 156 :=
by sorry

end NUMINAMATH_CALUDE_jorge_goals_this_season_l314_31455


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l314_31487

theorem least_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (525 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (525 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l314_31487


namespace NUMINAMATH_CALUDE_product_prices_and_savings_l314_31413

-- Define the discount rates
def discount_A : ℚ := 0.2
def discount_B : ℚ := 0.25

-- Define the equations from the conditions
def equation1 (x y : ℚ) : Prop := 6 * x + 3 * y = 600
def equation2 (x y : ℚ) : Prop := 50 * (1 - discount_A) * x + 40 * (1 - discount_B) * y = 5200

-- Define the prices we want to prove
def price_A : ℚ := 40
def price_B : ℚ := 120

-- Define the savings calculation
def savings (x y : ℚ) : ℚ :=
  80 * x + 100 * y - (80 * (1 - discount_A) * x + 100 * (1 - discount_B) * y)

-- Theorem statement
theorem product_prices_and_savings :
  equation1 price_A price_B ∧
  equation2 price_A price_B ∧
  savings price_A price_B = 3640 := by
  sorry

end NUMINAMATH_CALUDE_product_prices_and_savings_l314_31413


namespace NUMINAMATH_CALUDE_sheets_from_jane_l314_31480

theorem sheets_from_jane (initial_sheets final_sheets given_sheets : ℕ) 
  (h1 : initial_sheets = 212)
  (h2 : given_sheets = 156)
  (h3 : final_sheets = 363) :
  initial_sheets + (final_sheets + given_sheets - initial_sheets) - given_sheets = final_sheets := by
  sorry

#check sheets_from_jane

end NUMINAMATH_CALUDE_sheets_from_jane_l314_31480


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l314_31495

theorem quadratic_inequality_always_nonnegative : ∀ x : ℝ, x^2 - x + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l314_31495


namespace NUMINAMATH_CALUDE_roots_quadratic_sum_l314_31470

theorem roots_quadratic_sum (a b : ℝ) : 
  (a^2 - 3*a + 1 = 0) → 
  (b^2 - 3*b + 1 = 0) → 
  (1 / (a^2 + 1) + 1 / (b^2 + 1) = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_sum_l314_31470


namespace NUMINAMATH_CALUDE_quadratic_root_value_l314_31401

theorem quadratic_root_value (d : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 14 * x + d = 0 ↔ x = (-14 + Real.sqrt 14) / 4 ∨ x = (-14 - Real.sqrt 14) / 4) →
  d = 91/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l314_31401


namespace NUMINAMATH_CALUDE_smallest_3_4_cut_is_14_l314_31496

/-- A positive integer n is m-cut if n-2 is divisible by m -/
def is_m_cut (n m : ℕ) : Prop :=
  n > 2 ∧ m > 2 ∧ (n - 2) % m = 0

/-- The smallest positive integer that is both 3-cut and 4-cut -/
def smallest_3_4_cut : ℕ := 14

/-- Theorem stating that 14 is the smallest positive integer that is both 3-cut and 4-cut -/
theorem smallest_3_4_cut_is_14 :
  (∀ n : ℕ, n < smallest_3_4_cut → ¬(is_m_cut n 3 ∧ is_m_cut n 4)) ∧
  (is_m_cut smallest_3_4_cut 3 ∧ is_m_cut smallest_3_4_cut 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_3_4_cut_is_14_l314_31496


namespace NUMINAMATH_CALUDE_bleacher_runs_theorem_l314_31454

/-- The number of times a player runs up and down the bleachers -/
def number_of_trips (stairs_one_way : ℕ) (calories_per_stair : ℕ) (total_calories_burned : ℕ) : ℕ :=
  total_calories_burned / (4 * stairs_one_way * calories_per_stair)

/-- Theorem stating the number of times players run up and down the bleachers -/
theorem bleacher_runs_theorem (stairs_one_way : ℕ) (calories_per_stair : ℕ) (total_calories_burned : ℕ)
    (h1 : stairs_one_way = 32)
    (h2 : calories_per_stair = 2)
    (h3 : total_calories_burned = 5120) :
    number_of_trips stairs_one_way calories_per_stair total_calories_burned = 40 := by
  sorry

end NUMINAMATH_CALUDE_bleacher_runs_theorem_l314_31454


namespace NUMINAMATH_CALUDE_cos_sum_max_min_points_l314_31494

/-- Given a function f(x) = cos(2x) + sin(x), prove that the cosine of the sum of
    the abscissas of its maximum and minimum points equals 1/4. -/
theorem cos_sum_max_min_points (f : ℝ → ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, f x = Real.cos (2 * x) + Real.sin x) →
  (∀ x, f x ≤ f x₁) →
  (∀ x, f x ≥ f x₂) →
  Real.cos (x₁ + x₂) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_max_min_points_l314_31494


namespace NUMINAMATH_CALUDE_painters_work_days_l314_31465

/-- Represents the number of work-days required for a given number of painters to complete a job -/
noncomputable def workDays (numPainters : ℕ) : ℝ :=
  sorry

theorem painters_work_days :
  (workDays 6 = 1.5) →
  (∀ (n m : ℕ), n * workDays n = m * workDays m) →
  workDays 4 = 2.25 :=
by
  sorry

end NUMINAMATH_CALUDE_painters_work_days_l314_31465


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percent_l314_31481

/-- Calculates the loss percent for a shopkeeper given profit margin and theft percentage -/
theorem shopkeeper_loss_percent 
  (profit_margin : ℝ) 
  (theft_percent : ℝ) 
  (hprofit : profit_margin = 0.1) 
  (htheft : theft_percent = 0.4) : 
  (1 - (1 - theft_percent) * (1 + profit_margin)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percent_l314_31481


namespace NUMINAMATH_CALUDE_chocolates_left_first_method_candies_left_second_method_chocolates_left_second_method_total_items_is_35_l314_31402

/-- Represents the number of bags packed using the first method -/
def x : ℕ := 1

/-- Represents the number of bags packed using the second method -/
def y : ℕ := 0

/-- The total number of chocolates initially -/
def total_chocolates : ℕ := 3 * x + 5 * y + 25

/-- The total number of fruit candies initially -/
def total_candies : ℕ := 7 * x + 5 * y

/-- Condition: When fruit candies are used up in the first method, 25 chocolates are left -/
theorem chocolates_left_first_method : total_chocolates - (3 * x + 5 * y) = 25 := by sorry

/-- Condition: In the second method, 4 fruit candies are left in the end -/
theorem candies_left_second_method : total_candies - (7 * x + 5 * y) = 4 := by sorry

/-- Condition: In the second method, 1 chocolate is left in the end -/
theorem chocolates_left_second_method : total_chocolates - (3 * x + 5 * y) - 4 = 1 := by sorry

/-- The main theorem: The total number of chocolates and fruit candies is 35 -/
theorem total_items_is_35 : total_chocolates + total_candies = 35 := by sorry

end NUMINAMATH_CALUDE_chocolates_left_first_method_candies_left_second_method_chocolates_left_second_method_total_items_is_35_l314_31402


namespace NUMINAMATH_CALUDE_aloks_age_l314_31423

theorem aloks_age (alok_age bipin_age chandan_age : ℕ) : 
  bipin_age = 6 * alok_age →
  bipin_age + 10 = 2 * (chandan_age + 10) →
  chandan_age = 10 →
  alok_age = 5 := by
sorry

end NUMINAMATH_CALUDE_aloks_age_l314_31423


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l314_31417

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -2 ∧ x₂ > -2 ∧
    x₁^2 + (2*m + 6)*x₁ + 4*m + 12 = 0 ∧
    x₂^2 + (2*m + 6)*x₂ + 4*m + 12 = 0) ↔
  m ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l314_31417


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l314_31420

theorem cube_sum_of_roots (a b c : ℝ) : 
  (3 * a^3 - 2 * a^2 + 5 * a - 7 = 0) ∧ 
  (3 * b^3 - 2 * b^2 + 5 * b - 7 = 0) ∧ 
  (3 * c^3 - 2 * c^2 + 5 * c - 7 = 0) → 
  a^3 + b^3 + c^3 = 137 / 27 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l314_31420


namespace NUMINAMATH_CALUDE_dog_max_distance_l314_31400

/-- The maximum distance a dog can be from the origin when tied to a post -/
theorem dog_max_distance (post_x post_y rope_length : ℝ) :
  post_x = 6 ∧ post_y = 8 ∧ rope_length = 15 →
  ∃ (max_distance : ℝ),
    max_distance = 25 ∧
    ∀ (x y : ℝ),
      (x - post_x)^2 + (y - post_y)^2 ≤ rope_length^2 →
      x^2 + y^2 ≤ max_distance^2 :=
by sorry

end NUMINAMATH_CALUDE_dog_max_distance_l314_31400


namespace NUMINAMATH_CALUDE_complex_equation_solution_l314_31432

theorem complex_equation_solution (z : ℂ) : z = Complex.I * (2 - z) → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l314_31432


namespace NUMINAMATH_CALUDE_fraction_decomposition_sum_l314_31498

theorem fraction_decomposition_sum : ∃ (C D : ℝ), 
  (∀ x : ℝ, x ≠ 2 → x ≠ 4 → (D * x - 17) / ((x - 2) * (x - 4)) = C / (x - 2) + 4 / (x - 4)) ∧
  C + D = 8.5 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_sum_l314_31498


namespace NUMINAMATH_CALUDE_tangent_line_equation_l314_31466

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Define the point of tangency
def point : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l314_31466


namespace NUMINAMATH_CALUDE_hernandez_state_tax_l314_31434

/-- Calculates the state tax for a resident given their taxable income and months of residence --/
def calculate_state_tax (taxable_income : ℕ) (months_of_residence : ℕ) : ℕ :=
  let adjusted_income := taxable_income - 5000
  let tax_bracket1 := min adjusted_income 10000
  let tax_bracket2 := min (max (adjusted_income - 10000) 0) 20000
  let tax_bracket3 := min (max (adjusted_income - 30000) 0) 30000
  let tax_bracket4 := max (adjusted_income - 60000) 0
  let total_tax := tax_bracket1 / 100 + tax_bracket2 * 3 / 100 + tax_bracket3 * 5 / 100 + tax_bracket4 * 7 / 100
  let tax_credit := if months_of_residence < 10 then 500 else 0
  total_tax - tax_credit

/-- The theorem stating that Mr. Hernandez's state tax is $575 --/
theorem hernandez_state_tax :
  calculate_state_tax 42500 9 = 575 :=
by sorry

end NUMINAMATH_CALUDE_hernandez_state_tax_l314_31434


namespace NUMINAMATH_CALUDE_burger_length_l314_31451

theorem burger_length (share : ℝ) (h1 : share = 6) : 2 * share = 12 := by
  sorry

#check burger_length

end NUMINAMATH_CALUDE_burger_length_l314_31451


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l314_31483

theorem reciprocal_sum_of_quadratic_roots (α β : ℝ) : 
  (∃ r s : ℝ, 7 * r^2 - 8 * r + 6 = 0 ∧ 
               7 * s^2 - 8 * s + 6 = 0 ∧ 
               α = 1 / r ∧ 
               β = 1 / s) → 
  α + β = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l314_31483


namespace NUMINAMATH_CALUDE_watch_cost_price_l314_31499

/-- Proves that the cost price of a watch is 3000, given the conditions of the problem -/
theorem watch_cost_price (loss_percentage : ℚ) (gain_percentage : ℚ) (price_difference : ℚ) :
  loss_percentage = 10 / 100 →
  gain_percentage = 8 / 100 →
  price_difference = 540 →
  ∃ (cost_price : ℚ),
    cost_price * (1 - loss_percentage) + price_difference = cost_price * (1 + gain_percentage) ∧
    cost_price = 3000 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l314_31499


namespace NUMINAMATH_CALUDE_rice_and_husk_division_l314_31468

/-- Calculates the approximate amount of husks in a batch of grain --/
def calculate_husks (total_grain : ℕ) (sample_husks : ℕ) (sample_total : ℕ) : ℕ :=
  (total_grain * sample_husks) / sample_total

/-- The Rice and Husk Division problem from "The Nine Chapters on the Mathematical Art" --/
theorem rice_and_husk_division :
  let total_grain : ℕ := 1524
  let sample_husks : ℕ := 28
  let sample_total : ℕ := 254
  calculate_husks total_grain sample_husks sample_total = 168 := by
  sorry

#eval calculate_husks 1524 28 254

end NUMINAMATH_CALUDE_rice_and_husk_division_l314_31468


namespace NUMINAMATH_CALUDE_game_score_product_l314_31437

def g (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 0

def allie_rolls : List ℕ := [6, 3, 2, 4]
def betty_rolls : List ℕ := [5, 2, 3, 6]

theorem game_score_product : 
  (allie_rolls.map g).sum * (betty_rolls.map g).sum = 143 := by
  sorry

end NUMINAMATH_CALUDE_game_score_product_l314_31437


namespace NUMINAMATH_CALUDE_chemistry_class_section_size_l314_31404

theorem chemistry_class_section_size :
  let section1_size : ℕ := 65
  let section2_size : ℕ := 35
  let section4_size : ℕ := 42
  let section1_mean : ℚ := 50
  let section2_mean : ℚ := 60
  let section3_mean : ℚ := 55
  let section4_mean : ℚ := 45
  let overall_mean : ℚ := 5195 / 100

  ∃ (section3_size : ℕ),
    (section1_size * section1_mean + section2_size * section2_mean + 
     section3_size * section3_mean + section4_size * section4_mean) / 
    (section1_size + section2_size + section3_size + section4_size : ℚ) = overall_mean ∧
    section3_size = 45
  := by sorry

end NUMINAMATH_CALUDE_chemistry_class_section_size_l314_31404


namespace NUMINAMATH_CALUDE_school_raffle_earnings_l314_31445

/-- The amount of money Zoe's school made from selling raffle tickets -/
def total_money_made (cost_per_ticket : ℕ) (num_tickets_sold : ℕ) : ℕ :=
  cost_per_ticket * num_tickets_sold

/-- Theorem stating that Zoe's school made 620 dollars from selling raffle tickets -/
theorem school_raffle_earnings :
  total_money_made 4 155 = 620 := by
  sorry

end NUMINAMATH_CALUDE_school_raffle_earnings_l314_31445


namespace NUMINAMATH_CALUDE_parity_of_D_2021_2022_2023_l314_31473

def D : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | n+3 => D (n+2) + D n

theorem parity_of_D_2021_2022_2023 :
  (D 2021 % 2 = 0) ∧ (D 2022 % 2 = 1) ∧ (D 2023 % 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_parity_of_D_2021_2022_2023_l314_31473
