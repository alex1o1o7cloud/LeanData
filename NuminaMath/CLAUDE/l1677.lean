import Mathlib

namespace NUMINAMATH_CALUDE_candies_equalization_l1677_167757

theorem candies_equalization (basket_a basket_b added : ℕ) : 
  basket_a = 8 → basket_b = 17 → basket_a + added = basket_b → added = 9 := by
sorry

end NUMINAMATH_CALUDE_candies_equalization_l1677_167757


namespace NUMINAMATH_CALUDE_value_of_expression_l1677_167741

theorem value_of_expression (x y : ℝ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1677_167741


namespace NUMINAMATH_CALUDE_wine_cork_price_difference_l1677_167746

/-- 
Given:
- The price of a bottle of wine with a cork
- The price of the cork
Prove that the difference in price between a bottle of wine with a cork and without a cork
is equal to the price of the cork.
-/
theorem wine_cork_price_difference 
  (price_with_cork : ℝ) 
  (price_cork : ℝ) 
  (h1 : price_with_cork = 2.10)
  (h2 : price_cork = 0.05) :
  price_with_cork - (price_with_cork - price_cork) = price_cork :=
by sorry

end NUMINAMATH_CALUDE_wine_cork_price_difference_l1677_167746


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1677_167723

theorem complex_equation_solution (z : ℂ) : z = Complex.I * (2 - z) → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1677_167723


namespace NUMINAMATH_CALUDE_esperanza_savings_l1677_167764

theorem esperanza_savings :
  let rent : ℕ := 600
  let food_cost : ℕ := (3 * rent) / 5
  let mortgage : ℕ := 3 * food_cost
  let gross_salary : ℕ := 4840
  let expenses : ℕ := rent + food_cost + mortgage
  let pre_tax_savings : ℕ := gross_salary - expenses
  let taxes : ℕ := (2 * pre_tax_savings) / 5
  let savings : ℕ := pre_tax_savings - taxes
  savings = 1680 := by sorry

end NUMINAMATH_CALUDE_esperanza_savings_l1677_167764


namespace NUMINAMATH_CALUDE_smartphone_savings_plan_l1677_167743

theorem smartphone_savings_plan (smartphone_cost initial_savings : ℕ) 
  (saving_months weeks_per_month : ℕ) : 
  smartphone_cost = 160 →
  initial_savings = 40 →
  saving_months = 2 →
  weeks_per_month = 4 →
  (smartphone_cost - initial_savings) / (saving_months * weeks_per_month) = 15 := by
sorry

end NUMINAMATH_CALUDE_smartphone_savings_plan_l1677_167743


namespace NUMINAMATH_CALUDE_divisibility_of_concatenated_integers_l1677_167709

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of concatenating integers from 1 to n
  sorry

theorem divisibility_of_concatenated_integers :
  ∃ M : ℕ, M = concatenate_integers 50 ∧ M % 51 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_concatenated_integers_l1677_167709


namespace NUMINAMATH_CALUDE_solve_for_x_l1677_167797

theorem solve_for_x : ∃ x : ℤ, x + 1315 + 9211 - 1569 = 11901 ∧ x = 2944 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1677_167797


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l1677_167763

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by_9 (529000 + d * 100 + 46) ∧
    ∀ (d' : ℕ), d' < d → ¬is_divisible_by_9 (529000 + d' * 100 + 46) :=
by
  use 1
  sorry

#check smallest_digit_for_divisibility

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l1677_167763


namespace NUMINAMATH_CALUDE_find_set_M_l1677_167795

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}

def complement_M : Finset ℕ := {1, 2, 4}

theorem find_set_M : 
  ∀ M : Finset ℕ, (∀ x : ℕ, x ∈ U → (x ∈ M ↔ x ∉ complement_M)) → 
  M = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_find_set_M_l1677_167795


namespace NUMINAMATH_CALUDE_bus_stoppage_time_l1677_167713

/-- Proves that a bus with given speeds stops for 30 minutes per hour -/
theorem bus_stoppage_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 32)
  (h2 : speed_with_stops = 16) : 
  (1 - speed_with_stops / speed_without_stops) * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_bus_stoppage_time_l1677_167713


namespace NUMINAMATH_CALUDE_russel_carousel_rides_l1677_167733

/-- The number of times Russel rode the carousel -/
def carousel_rides (total_tickets jen_games shooting_cost carousel_cost : ℕ) : ℕ :=
  (total_tickets - jen_games * shooting_cost) / carousel_cost

/-- Proof that Russel rode the carousel 3 times -/
theorem russel_carousel_rides : 
  carousel_rides 19 2 5 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_russel_carousel_rides_l1677_167733


namespace NUMINAMATH_CALUDE_only_paintable_integer_l1677_167796

/-- Represents a painting configuration for the fence. -/
structure PaintingConfig where
  h : ℕ+  -- Harold's interval
  t : ℕ+  -- Tanya's interval
  u : ℕ+  -- Ulysses' interval
  v : ℕ+  -- Victor's interval

/-- Checks if a picket is painted by Harold. -/
def paintedByHarold (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.h.val = 1

/-- Checks if a picket is painted by Tanya. -/
def paintedByTanya (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.t.val = 2

/-- Checks if a picket is painted by Ulysses. -/
def paintedByUlysses (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.u.val = 3

/-- Checks if a picket is painted by Victor. -/
def paintedByVictor (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.v.val = 4

/-- Checks if a picket is painted by exactly one person. -/
def paintedOnce (config : PaintingConfig) (picket : ℕ) : Prop :=
  (paintedByHarold config picket ∨ paintedByTanya config picket ∨
   paintedByUlysses config picket ∨ paintedByVictor config picket) ∧
  ¬(paintedByHarold config picket ∧ paintedByTanya config picket) ∧
  ¬(paintedByHarold config picket ∧ paintedByUlysses config picket) ∧
  ¬(paintedByHarold config picket ∧ paintedByVictor config picket) ∧
  ¬(paintedByTanya config picket ∧ paintedByUlysses config picket) ∧
  ¬(paintedByTanya config picket ∧ paintedByVictor config picket) ∧
  ¬(paintedByUlysses config picket ∧ paintedByVictor config picket)

/-- Checks if a configuration is paintable. -/
def isPaintable (config : PaintingConfig) : Prop :=
  ∀ picket : ℕ, picket > 0 → paintedOnce config picket

/-- Calculates the paintable integer for a configuration. -/
def paintableInteger (config : PaintingConfig) : ℕ :=
  1000 * config.h.val + 100 * config.t.val + 10 * config.u.val + config.v.val

/-- The main theorem stating that 4812 is the only paintable integer. -/
theorem only_paintable_integer :
  ∀ config : PaintingConfig, isPaintable config → paintableInteger config = 4812 := by
  sorry


end NUMINAMATH_CALUDE_only_paintable_integer_l1677_167796


namespace NUMINAMATH_CALUDE_max_servings_emily_l1677_167736

/-- Represents the recipe requirements for 4 servings -/
structure Recipe :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (water : ℚ)
  (milk : ℚ)

/-- Represents Emily's available ingredients -/
structure Available :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (milk : ℚ)

def recipe : Recipe :=
  { chocolate := 3
  , sugar := 1/2
  , water := 2
  , milk := 3 }

def emily : Available :=
  { chocolate := 9
  , sugar := 3
  , milk := 10 }

/-- Calculates the number of servings possible for a given ingredient -/
def servings_for_ingredient (recipe_amount : ℚ) (available_amount : ℚ) : ℚ :=
  (available_amount / recipe_amount) * 4

theorem max_servings_emily :
  let chocolate_servings := servings_for_ingredient recipe.chocolate emily.chocolate
  let sugar_servings := servings_for_ingredient recipe.sugar emily.sugar
  let milk_servings := servings_for_ingredient recipe.milk emily.milk
  min chocolate_servings (min sugar_servings milk_servings) = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_emily_l1677_167736


namespace NUMINAMATH_CALUDE_bicycle_average_speed_l1677_167726

theorem bicycle_average_speed (total_distance : ℝ) (first_distance : ℝ) (second_distance : ℝ)
  (first_speed : ℝ) (second_speed : ℝ) (h1 : total_distance = 250)
  (h2 : first_distance = 100) (h3 : second_distance = 150)
  (h4 : first_speed = 20) (h5 : second_speed = 15)
  (h6 : total_distance = first_distance + second_distance) :
  (total_distance / (first_distance / first_speed + second_distance / second_speed)) =
  (250 : ℝ) / ((100 : ℝ) / 20 + (150 : ℝ) / 15) := by
  sorry

end NUMINAMATH_CALUDE_bicycle_average_speed_l1677_167726


namespace NUMINAMATH_CALUDE_solve_for_z_l1677_167742

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem solve_for_z : ∃ z : ℝ, euro (euro 4 5) z = 560 ∧ z = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l1677_167742


namespace NUMINAMATH_CALUDE_rectangle_fit_impossibility_l1677_167782

theorem rectangle_fit_impossibility : 
  ∀ (a b c d : ℝ), 
    a = 5 ∧ b = 6 ∧ c = 3 ∧ d = 8 → 
    (c^2 + d^2 : ℝ) > (a^2 + b^2 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_fit_impossibility_l1677_167782


namespace NUMINAMATH_CALUDE_dog_bones_total_l1677_167767

theorem dog_bones_total (initial_bones dug_up_bones : ℕ) 
  (h1 : initial_bones = 493) 
  (h2 : dug_up_bones = 367) : 
  initial_bones + dug_up_bones = 860 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_total_l1677_167767


namespace NUMINAMATH_CALUDE_max_profit_price_l1677_167756

/-- Represents the profit function for a product -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 280 * x - 1600

/-- The initial purchase price of the product -/
def initial_purchase_price : ℝ := 8

/-- The initial selling price of the product -/
def initial_selling_price : ℝ := 10

/-- The initial daily sales volume -/
def initial_daily_sales : ℝ := 100

/-- The decrease in daily sales for each yuan increase in price -/
def sales_decrease_rate : ℝ := 10

/-- Theorem: The selling price that maximizes profit is 14 yuan -/
theorem max_profit_price : 
  ∃ (x : ℝ), x > initial_selling_price ∧ 
  ∀ (y : ℝ), y > initial_selling_price → profit_function x ≥ profit_function y :=
sorry

end NUMINAMATH_CALUDE_max_profit_price_l1677_167756


namespace NUMINAMATH_CALUDE_coral_age_conversion_l1677_167761

/-- Converts an octal digit to decimal --/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- Converts an octal number to decimal --/
def octal_to_decimal_number (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + octal_to_decimal digit * 8^i) 0

theorem coral_age_conversion :
  octal_to_decimal_number [7, 3, 4] = 476 := by
  sorry

end NUMINAMATH_CALUDE_coral_age_conversion_l1677_167761


namespace NUMINAMATH_CALUDE_shaded_fraction_is_seven_sixteenths_l1677_167719

/-- Represents a square divided into smaller squares and triangles -/
structure DividedSquare where
  /-- The number of smaller squares the large square is divided into -/
  num_small_squares : ℕ
  /-- The number of triangles each smaller square is divided into -/
  triangles_per_small_square : ℕ
  /-- The total number of shaded triangles -/
  shaded_triangles : ℕ

/-- Calculates the fraction of the square that is shaded -/
def shaded_fraction (s : DividedSquare) : ℚ :=
  s.shaded_triangles / (s.num_small_squares * s.triangles_per_small_square)

/-- Theorem stating that the shaded fraction of the given square is 7/16 -/
theorem shaded_fraction_is_seven_sixteenths (s : DividedSquare) 
  (h1 : s.num_small_squares = 4)
  (h2 : s.triangles_per_small_square = 4)
  (h3 : s.shaded_triangles = 7) : 
  shaded_fraction s = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_is_seven_sixteenths_l1677_167719


namespace NUMINAMATH_CALUDE_bacteria_growth_rate_l1677_167787

/-- The growth rate of a bacteria colony -/
def growth_rate : ℝ := 2

/-- The number of days for a single colony to reach the habitat's limit -/
def single_colony_days : ℕ := 22

/-- The number of days for two colonies to reach the habitat's limit -/
def double_colony_days : ℕ := 21

/-- The theorem stating the growth rate of the bacteria colony -/
theorem bacteria_growth_rate :
  (growth_rate ^ single_colony_days : ℝ) = 2 * (growth_rate ^ double_colony_days : ℝ) :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_rate_l1677_167787


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1677_167715

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2)
  (h_x : x > 1)
  (h_cos : Real.cos (θ / 2) = Real.sqrt ((x + 1) / (2 * x))) :
  Real.tan θ = Real.sqrt (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1677_167715


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1677_167707

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a < 0 ∧ -1 < b ∧ b < 0) →
  (a + a * b < 0) ∧
  ∃ (x y : ℝ), x + x * y < 0 ∧ ¬(x < 0 ∧ -1 < y ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1677_167707


namespace NUMINAMATH_CALUDE_plot_width_l1677_167770

/-- Proves that a rectangular plot with given conditions has a width of 47.5 meters -/
theorem plot_width (length : ℝ) (poles : ℕ) (pole_distance : ℝ) (width : ℝ) :
  length = 90 →
  poles = 56 →
  pole_distance = 5 →
  (poles - 1 : ℝ) * pole_distance = 2 * (length + width) →
  width = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_plot_width_l1677_167770


namespace NUMINAMATH_CALUDE_new_yellow_tint_percentage_l1677_167734

/-- Calculates the new percentage of yellow tint after adding more yellow tint to a mixture -/
theorem new_yellow_tint_percentage
  (initial_volume : ℝ)
  (initial_yellow_percentage : ℝ)
  (added_yellow_volume : ℝ)
  (h1 : initial_volume = 40)
  (h2 : initial_yellow_percentage = 0.25)
  (h3 : added_yellow_volume = 10) :
  let initial_yellow_volume := initial_volume * initial_yellow_percentage
  let new_yellow_volume := initial_yellow_volume + added_yellow_volume
  let new_total_volume := initial_volume + added_yellow_volume
  new_yellow_volume / new_total_volume = 0.4 := by
sorry


end NUMINAMATH_CALUDE_new_yellow_tint_percentage_l1677_167734


namespace NUMINAMATH_CALUDE_win_sector_area_l1677_167783

theorem win_sector_area (radius : ℝ) (win_probability : ℝ) (win_area : ℝ) : 
  radius = 8 → win_probability = 1/4 → win_area = 16 * Real.pi → 
  win_area = win_probability * Real.pi * radius^2 := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1677_167783


namespace NUMINAMATH_CALUDE_mnp_value_l1677_167712

theorem mnp_value (a b x z : ℝ) (m n p : ℤ) 
  (h : a^12 * x * z - a^10 * z - a^9 * x = a^8 * (b^6 - 1)) 
  (h_equiv : (a^m * x - a^n) * (a^p * z - a^3) = a^8 * b^6) : 
  m * n * p = 4 := by
sorry

end NUMINAMATH_CALUDE_mnp_value_l1677_167712


namespace NUMINAMATH_CALUDE_product_of_one_plus_roots_l1677_167753

theorem product_of_one_plus_roots (u v w : ℝ) : 
  u^3 - 15*u^2 + 25*u - 12 = 0 ∧ 
  v^3 - 15*v^2 + 25*v - 12 = 0 ∧ 
  w^3 - 15*w^2 + 25*w - 12 = 0 → 
  (1 + u) * (1 + v) * (1 + w) = 29 := by
sorry

end NUMINAMATH_CALUDE_product_of_one_plus_roots_l1677_167753


namespace NUMINAMATH_CALUDE_max_value_of_J_l1677_167718

def consecutive_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (· + 1)

def sum_equals_21 (a b c d : ℕ) : Prop :=
  a + b + c + d = 21

theorem max_value_of_J (nums : List ℕ) (A B C D E F G H I J K : ℕ) :
  nums = consecutive_numbers 11 →
  D ∈ nums → G ∈ nums → I ∈ nums → F ∈ nums → A ∈ nums →
  B ∈ nums → C ∈ nums → E ∈ nums → H ∈ nums → J ∈ nums → K ∈ nums →
  D > G → G > I → I > F → F > A →
  sum_equals_21 A B C D →
  sum_equals_21 D E F G →
  sum_equals_21 G H F I →
  sum_equals_21 I J K A →
  J ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_J_l1677_167718


namespace NUMINAMATH_CALUDE_quadratic_inequality_boundary_l1677_167717

theorem quadratic_inequality_boundary (c : ℝ) : 
  (∀ x : ℝ, x * (4 * x + 1) < c ↔ -5/2 < x ∧ x < 3) → c = 27 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_boundary_l1677_167717


namespace NUMINAMATH_CALUDE_rice_and_flour_consumption_l1677_167716

theorem rice_and_flour_consumption (initial_rice initial_flour consumed : ℕ) : 
  initial_rice = 500 →
  initial_flour = 200 →
  initial_rice - consumed = 7 * (initial_flour - consumed) →
  consumed = 150 := by
sorry

end NUMINAMATH_CALUDE_rice_and_flour_consumption_l1677_167716


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_l1677_167777

theorem greatest_four_digit_multiple : ∃ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (15 ∣ n) ∧ (25 ∣ n) ∧ (40 ∣ n) ∧ (75 ∣ n) ∧
  (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) ∧ (15 ∣ m) ∧ (25 ∣ m) ∧ (40 ∣ m) ∧ (75 ∣ m) → m ≤ n) ∧
  n = 9600 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_l1677_167777


namespace NUMINAMATH_CALUDE_school_fundraising_admin_fee_percentage_l1677_167780

/-- Proves that the percentage deducted for administration fees is 2% --/
theorem school_fundraising_admin_fee_percentage 
  (johnson_amount : ℝ)
  (sutton_amount : ℝ)
  (rollin_amount : ℝ)
  (total_amount : ℝ)
  (remaining_amount : ℝ)
  (h1 : johnson_amount = 2300)
  (h2 : johnson_amount = 2 * sutton_amount)
  (h3 : rollin_amount = 8 * sutton_amount)
  (h4 : rollin_amount = total_amount / 3)
  (h5 : remaining_amount = 27048) :
  (total_amount - remaining_amount) / total_amount * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_school_fundraising_admin_fee_percentage_l1677_167780


namespace NUMINAMATH_CALUDE_dog_reachable_area_l1677_167745

/-- The area outside a regular hexagon reachable by a tethered dog -/
theorem dog_reachable_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 2 → rope_length = 5 → 
  (π * rope_length^2 : ℝ) = 25 * π := by
  sorry

#check dog_reachable_area

end NUMINAMATH_CALUDE_dog_reachable_area_l1677_167745


namespace NUMINAMATH_CALUDE_problem_statement_l1677_167781

theorem problem_statement (x y : ℚ) : 
  x = 3/4 → y = 4/3 → (3/5 : ℚ) * x^5 * y^8 = 897/1000 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1677_167781


namespace NUMINAMATH_CALUDE_starting_lineup_count_l1677_167744

def total_players : ℕ := 20
def point_guards : ℕ := 1
def other_players : ℕ := 7

def starting_lineup_combinations : ℕ := total_players * (Nat.choose (total_players - point_guards) other_players)

theorem starting_lineup_count :
  starting_lineup_combinations = 1007760 :=
sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l1677_167744


namespace NUMINAMATH_CALUDE_technicians_count_l1677_167755

/-- Proves the number of technicians in a workshop with given salary conditions -/
theorem technicians_count (total_workers : ℕ) (avg_salary : ℕ) (tech_salary : ℕ) (rest_salary : ℕ) :
  total_workers = 14 ∧ 
  avg_salary = 8000 ∧ 
  tech_salary = 10000 ∧ 
  rest_salary = 6000 → 
  ∃ (tech_count : ℕ),
    tech_count = 7 ∧ 
    tech_count ≤ total_workers ∧
    tech_count * tech_salary + (total_workers - tech_count) * rest_salary = total_workers * avg_salary :=
by sorry

end NUMINAMATH_CALUDE_technicians_count_l1677_167755


namespace NUMINAMATH_CALUDE_max_eccentricity_ellipse_l1677_167739

/-- The maximum eccentricity of an ellipse with given properties -/
theorem max_eccentricity_ellipse :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (2, 0)
  let P : ℝ → ℝ × ℝ := λ x => (x, x + 3)
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let c : ℝ := dist A B / 2
  let a (x : ℝ) : ℝ := (dist (P x) A + dist (P x) B) / 2
  let e (x : ℝ) : ℝ := c / a x
  ∃ (x : ℝ), ∀ (y : ℝ), e y ≤ e x ∧ e x = 2 * Real.sqrt 26 / 13 :=
sorry

end NUMINAMATH_CALUDE_max_eccentricity_ellipse_l1677_167739


namespace NUMINAMATH_CALUDE_waiter_customers_l1677_167799

/-- Calculates the total number of customers for a waiter given the number of tables and customers per table. -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Theorem stating that a waiter with 5 tables, each having 5 women and 3 men, has a total of 40 customers. -/
theorem waiter_customers :
  total_customers 5 5 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l1677_167799


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l1677_167752

theorem divisibility_by_seven (a b : ℕ) : 
  (7 ∣ (a * b)) → (7 ∣ a) ∨ (7 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l1677_167752


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l1677_167704

theorem range_of_a_minus_b (a b : ℝ) (ha : -2 < a ∧ a < 1) (hb : 0 < b ∧ b < 4) :
  ∀ x, (∃ y z, -2 < y ∧ y < 1 ∧ 0 < z ∧ z < 4 ∧ x = y - z) ↔ -6 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l1677_167704


namespace NUMINAMATH_CALUDE_find_divisor_l1677_167776

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 100)
  (h2 : quotient = 9)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 11 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1677_167776


namespace NUMINAMATH_CALUDE_books_read_during_travel_l1677_167711

theorem books_read_during_travel (total_distance : ℝ) (distance_per_book : ℝ) : 
  total_distance = 6987.5 → 
  distance_per_book = 482.3 → 
  ⌊total_distance / distance_per_book⌋ = 14 := by
sorry

end NUMINAMATH_CALUDE_books_read_during_travel_l1677_167711


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l1677_167710

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 2) :
  2 * x^2 + 3 * y^2 + z^2 ≥ 24 / 11 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l1677_167710


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l1677_167730

theorem partial_fraction_sum (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l1677_167730


namespace NUMINAMATH_CALUDE_largest_inscribed_right_triangle_area_l1677_167784

/-- The area of the largest inscribed right triangle in a circle -/
theorem largest_inscribed_right_triangle_area (r : ℝ) (h : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_triangle_area := (diameter * r) / 2
  max_triangle_area = 64 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_right_triangle_area_l1677_167784


namespace NUMINAMATH_CALUDE_square_area_ratio_l1677_167765

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 48) (h2 : side_D = 60) :
  (side_C ^ 2) / (side_D ^ 2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1677_167765


namespace NUMINAMATH_CALUDE_ab_difference_l1677_167705

theorem ab_difference (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_difference_l1677_167705


namespace NUMINAMATH_CALUDE_smallest_ending_9_div_13_proof_l1677_167785

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

def smallest_ending_9_div_13 : ℕ := 129

theorem smallest_ending_9_div_13_proof :
  (ends_in_9 smallest_ending_9_div_13) ∧
  (smallest_ending_9_div_13 % 13 = 0) ∧
  (∀ m : ℕ, m < smallest_ending_9_div_13 → ¬(ends_in_9 m ∧ m % 13 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_ending_9_div_13_proof_l1677_167785


namespace NUMINAMATH_CALUDE_unique_delivery_exists_l1677_167714

/-- Represents the amount of cargo delivered to each warehouse -/
structure Delivery where
  first : Int
  second : Int
  third : Int

/-- Checks if a delivery satisfies the given conditions -/
def satisfiesConditions (d : Delivery) : Prop :=
  d.first + d.second = 400 ∧
  d.second + d.third = -300 ∧
  d.first + d.third = -440

/-- The theorem stating that there is a unique delivery satisfying the conditions -/
theorem unique_delivery_exists : ∃! d : Delivery, satisfiesConditions d ∧ 
  d.first = -130 ∧ d.second = -270 ∧ d.third = 230 := by
  sorry

end NUMINAMATH_CALUDE_unique_delivery_exists_l1677_167714


namespace NUMINAMATH_CALUDE_exists_k_no_carry_l1677_167720

/-- 
There exists a positive integer k such that 3993·k is a number 
consisting only of the digit 9.
-/
theorem exists_k_no_carry : ∃ k : ℕ+, 
  ∃ n : ℕ+, (3993 * k.val : ℕ) = (10^n.val - 1) := by sorry

end NUMINAMATH_CALUDE_exists_k_no_carry_l1677_167720


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l1677_167791

theorem sum_of_coefficients_zero (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 - x - 2)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                               a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l1677_167791


namespace NUMINAMATH_CALUDE_complete_square_constant_l1677_167735

theorem complete_square_constant (a h k : ℚ) :
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) →
  k = -49/4 := by
sorry

end NUMINAMATH_CALUDE_complete_square_constant_l1677_167735


namespace NUMINAMATH_CALUDE_cos_300_degrees_l1677_167750

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l1677_167750


namespace NUMINAMATH_CALUDE_milk_remaining_l1677_167728

theorem milk_remaining (initial : ℚ) (given : ℚ) (remaining : ℚ) : 
  initial = 8 → given = 18/7 → remaining = initial - given → remaining = 38/7 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l1677_167728


namespace NUMINAMATH_CALUDE_committee_count_l1677_167774

/-- Represents a department in the division of sciences -/
inductive Department
| Mathematics
| Statistics
| ComputerScience
| Physics

/-- Represents the gender of a professor -/
inductive Gender
| Male
| Female

/-- Represents the number of professors in each department by gender -/
def professors_count (d : Department) (g : Gender) : Nat :=
  match d, g with
  | Department.Physics, _ => 1
  | _, _ => 3

/-- Represents the total number of professors to be selected from each department -/
def selection_count (d : Department) : Nat :=
  match d with
  | Department.Physics => 1
  | _ => 2

/-- Calculates the number of ways to select professors from a department -/
def department_selection_ways (d : Department) : Nat :=
  (professors_count d Gender.Male).choose (selection_count d) *
  (professors_count d Gender.Female).choose (selection_count d)

/-- Theorem: The number of possible committees is 729 -/
theorem committee_count : 
  (department_selection_ways Department.Mathematics) *
  (department_selection_ways Department.Statistics) *
  (department_selection_ways Department.ComputerScience) *
  (department_selection_ways Department.Physics) = 729 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_l1677_167774


namespace NUMINAMATH_CALUDE_certain_number_proof_l1677_167792

theorem certain_number_proof (x : ℝ) (h : x / 3 = 3 * 134) : x = 1206 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1677_167792


namespace NUMINAMATH_CALUDE_u_floor_formula_l1677_167771

def u : ℕ → ℚ
  | 0 => 2
  | 1 => 5/2
  | (n+2) => u (n+1) * (u n ^ 2 - 2) - u 1

theorem u_floor_formula (n : ℕ) (h : n ≥ 1) :
  ⌊u n⌋ = (2 * (2^n - (-1)^n)) / 3 :=
sorry

end NUMINAMATH_CALUDE_u_floor_formula_l1677_167771


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1677_167758

theorem rhombus_longer_diagonal (side_length shorter_diagonal : ℝ) :
  side_length = 65 ∧ shorter_diagonal = 72 →
  ∃ longer_diagonal : ℝ, longer_diagonal = 108 ∧
  longer_diagonal^2 + shorter_diagonal^2 = 4 * side_length^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1677_167758


namespace NUMINAMATH_CALUDE_minimum_additional_stickers_l1677_167790

def initial_stickers : ℕ := 29
def row_size : ℕ := 4
def group_size : ℕ := 5

theorem minimum_additional_stickers :
  let total_stickers := initial_stickers + 11
  (total_stickers % row_size = 0) ∧
  (total_stickers % group_size = 0) ∧
  (∀ n : ℕ, n < 11 →
    let test_total := initial_stickers + n
    (test_total % row_size ≠ 0) ∨ (test_total % group_size ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_additional_stickers_l1677_167790


namespace NUMINAMATH_CALUDE_vector_subtraction_l1677_167779

def a : Fin 2 → ℝ := ![-1, 3]
def b : Fin 2 → ℝ := ![2, -1]

theorem vector_subtraction : a - 2 • b = ![-5, 5] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1677_167779


namespace NUMINAMATH_CALUDE_painted_cube_problem_l1677_167740

theorem painted_cube_problem (n : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 →
  n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l1677_167740


namespace NUMINAMATH_CALUDE_job_completion_time_l1677_167788

/-- Given a job that A and B can complete together in 5 days, and B can complete alone in 10 days,
    prove that A can complete the job alone in 10 days. -/
theorem job_completion_time (rate_A rate_B : ℝ) : 
  rate_A + rate_B = 1 / 5 →  -- A and B together complete the job in 5 days
  rate_B = 1 / 10 →          -- B alone completes the job in 10 days
  rate_A = 1 / 10            -- A alone completes the job in 10 days
:= by sorry

end NUMINAMATH_CALUDE_job_completion_time_l1677_167788


namespace NUMINAMATH_CALUDE_toonie_is_two_dollar_coin_l1677_167737

/-- Represents the types of coins in Antonella's purse -/
inductive Coin
  | Loonie
  | Toonie

/-- The value of a coin in dollars -/
def coin_value (c : Coin) : ℕ :=
  match c with
  | Coin.Loonie => 1
  | Coin.Toonie => 2

/-- Antonella's coin situation -/
structure AntoniellaPurse where
  coins : List Coin
  initial_toonies : ℕ
  spent : ℕ
  remaining : ℕ

/-- The conditions of Antonella's coins -/
def antonellas_coins : AntoniellaPurse :=
  { coins := List.replicate 10 Coin.Toonie,  -- placeholder, actual distribution doesn't matter
    initial_toonies := 4,
    spent := 3,
    remaining := 11 }

theorem toonie_is_two_dollar_coin (purse : AntoniellaPurse := antonellas_coins) :
  ∃ (c : Coin), coin_value c = 2 ∧ c = Coin.Toonie :=
sorry

end NUMINAMATH_CALUDE_toonie_is_two_dollar_coin_l1677_167737


namespace NUMINAMATH_CALUDE_negation_equivalence_l1677_167721

def exactly_one_even (a b c : ℕ) : Prop :=
  (Even a ∧ Odd b ∧ Odd c) ∨ (Odd a ∧ Even b ∧ Odd c) ∨ (Odd a ∧ Odd b ∧ Even c)

def negation_statement (a b c : ℕ) : Prop :=
  (Even a ∧ Even b) ∨ (Even a ∧ Even c) ∨ (Even b ∧ Even c) ∨ (Odd a ∧ Odd b ∧ Odd c)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ negation_statement a b c :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1677_167721


namespace NUMINAMATH_CALUDE_max_value_trig_sum_l1677_167789

open Real

theorem max_value_trig_sum (α β γ δ ε : ℝ) : 
  (∀ α β γ δ ε : ℝ, cos α * sin β + cos β * sin γ + cos γ * sin δ + cos δ * sin ε + cos ε * sin α ≤ 5) ∧ 
  (∃ α β γ δ ε : ℝ, cos α * sin β + cos β * sin γ + cos γ * sin δ + cos δ * sin ε + cos ε * sin α = 5) := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_sum_l1677_167789


namespace NUMINAMATH_CALUDE_symmetry_implies_linear_plus_periodic_l1677_167729

/-- A function has two centers of symmetry if there exist two distinct points
    such that reflecting the graph through these points leaves it unchanged. -/
def has_two_centers_of_symmetry (f : ℝ → ℝ) : Prop :=
  ∃ (C₁ C₂ : ℝ × ℝ), C₁ ≠ C₂ ∧
  ∀ (x y : ℝ), f y = x ↔ f (2 * C₁.1 - y) = 2 * C₁.2 - x ∧
                      f (2 * C₂.1 - y) = 2 * C₂.2 - x

/-- A function is the sum of a linear function and a periodic function if
    there exist real numbers b and a ≠ 0, and a periodic function g with period a,
    such that f(x) = bx + g(x) for all x. -/
def is_sum_of_linear_and_periodic (f : ℝ → ℝ) : Prop :=
  ∃ (b : ℝ) (a : ℝ) (g : ℝ → ℝ), a ≠ 0 ∧
  (∀ x, g (x + a) = g x) ∧
  (∀ x, f x = b * x + g x)

/-- Theorem: If a function has two centers of symmetry,
    then it can be expressed as the sum of a linear function and a periodic function. -/
theorem symmetry_implies_linear_plus_periodic (f : ℝ → ℝ) :
  has_two_centers_of_symmetry f → is_sum_of_linear_and_periodic f := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_linear_plus_periodic_l1677_167729


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1677_167775

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem intersection_complement_equality : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1677_167775


namespace NUMINAMATH_CALUDE_gcd_372_684_l1677_167748

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_372_684_l1677_167748


namespace NUMINAMATH_CALUDE_gcd_90_405_l1677_167769

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_405_l1677_167769


namespace NUMINAMATH_CALUDE_gcf_of_36_and_60_l1677_167738

theorem gcf_of_36_and_60 : Nat.gcd 36 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_36_and_60_l1677_167738


namespace NUMINAMATH_CALUDE_prob_sum_less_than_10_given_first_6_l1677_167786

/-- The probability that the sum of two dice is less than 10, given that the first die shows 6 -/
theorem prob_sum_less_than_10_given_first_6 :
  let outcomes : Finset ℕ := Finset.range 6
  let favorable_outcomes : Finset ℕ := Finset.filter (λ x => x + 6 < 10) outcomes
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_10_given_first_6_l1677_167786


namespace NUMINAMATH_CALUDE_tree_house_wood_theorem_l1677_167768

/-- The total amount of wood needed for John's tree house -/
def total_wood_needed : ℝ :=
  let pillar_short := 4
  let pillar_long := 5 * pillar_short
  let wall_long := 6
  let wall_short := wall_long - 3
  let floor_wood := 5.5
  let roof_long := 2 * floor_wood
  let roof_short := 1.5 * floor_wood
  4 * pillar_short + 4 * pillar_long +
  10 * wall_long + 10 * wall_short +
  8 * floor_wood +
  6 * roof_long + 6 * roof_short

/-- Theorem stating the total amount of wood needed for John's tree house -/
theorem tree_house_wood_theorem : total_wood_needed = 345.5 := by
  sorry

end NUMINAMATH_CALUDE_tree_house_wood_theorem_l1677_167768


namespace NUMINAMATH_CALUDE_ivy_cupcakes_l1677_167727

/-- The number of cupcakes Ivy baked in the morning -/
def morning_cupcakes : ℕ := 20

/-- The additional number of cupcakes Ivy baked in the afternoon compared to the morning -/
def afternoon_extra : ℕ := 15

/-- The total number of cupcakes Ivy baked -/
def total_cupcakes : ℕ := morning_cupcakes + (morning_cupcakes + afternoon_extra)

theorem ivy_cupcakes : total_cupcakes = 55 := by
  sorry

end NUMINAMATH_CALUDE_ivy_cupcakes_l1677_167727


namespace NUMINAMATH_CALUDE_wrapping_paper_area_is_8lh_l1677_167793

/-- Represents a rectangular box -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of wrapping paper needed for a given box -/
def wrappingPaperArea (box : Box) : ℝ :=
  8 * box.length * box.height

/-- Theorem stating that the area of wrapping paper needed is 8lh -/
theorem wrapping_paper_area_is_8lh (box : Box) :
  wrappingPaperArea box = 8 * box.length * box.height :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_is_8lh_l1677_167793


namespace NUMINAMATH_CALUDE_expression_simplification_l1677_167773

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 - 1) :
  (x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2*x + 1)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1677_167773


namespace NUMINAMATH_CALUDE_range_of_a_for_p_necessary_not_sufficient_for_q_l1677_167708

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 ≤ 5*x - 4
def q (x a : ℝ) : Prop := x^2 - (a + 2)*x + 2*a ≤ 0

-- Define the set A corresponding to proposition p
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

-- Define the set B corresponding to proposition q
def B (a : ℝ) : Set ℝ := {x | q x a}

-- Theorem statement
theorem range_of_a_for_p_necessary_not_sufficient_for_q :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ B a → x ∈ A) ∧ (∃ x : ℝ, x ∈ A ∧ x ∉ B a) ↔ 1 ≤ a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_p_necessary_not_sufficient_for_q_l1677_167708


namespace NUMINAMATH_CALUDE_multiples_of_seven_l1677_167772

theorem multiples_of_seven (a b : ℤ) (q : Set ℤ) : 
  (∃ k₁ k₂ : ℤ, a = 14 * k₁ ∧ b = 14 * k₂) →
  q = {x : ℤ | a ≤ x ∧ x ≤ b} →
  (Finset.filter (fun x => x % 14 = 0) (Finset.Icc a b)).card = 12 →
  (Finset.filter (fun x => x % 7 = 0) (Finset.Icc a b)).card = 24 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_seven_l1677_167772


namespace NUMINAMATH_CALUDE_bike_price_l1677_167702

theorem bike_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 ∧ upfront_percentage = 20 ∧ upfront_payment = (upfront_percentage / 100) * total_price →
  total_price = 1200 :=
by sorry

end NUMINAMATH_CALUDE_bike_price_l1677_167702


namespace NUMINAMATH_CALUDE_specific_bike_ride_north_distance_l1677_167706

/-- Represents a bike ride with given distances and final position -/
structure BikeRide where
  west : ℝ
  initialNorth : ℝ
  east : ℝ
  finalDistance : ℝ

/-- Calculates the final northward distance after going east for a given bike ride -/
def finalNorthDistance (ride : BikeRide) : ℝ :=
  sorry

/-- Theorem stating that for the specific bike ride described, the final northward distance after going east is 15 miles -/
theorem specific_bike_ride_north_distance :
  let ride : BikeRide := {
    west := 8,
    initialNorth := 5,
    east := 4,
    finalDistance := 20.396078054371138
  }
  finalNorthDistance ride = 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_bike_ride_north_distance_l1677_167706


namespace NUMINAMATH_CALUDE_adjacent_supplementary_angles_l1677_167778

/-- Given two adjacent supplementary angles, if one is 60°, then the other is 120°. -/
theorem adjacent_supplementary_angles (angle1 angle2 : ℝ) : 
  angle1 = 60 → 
  angle1 + angle2 = 180 → 
  angle2 = 120 := by
sorry

end NUMINAMATH_CALUDE_adjacent_supplementary_angles_l1677_167778


namespace NUMINAMATH_CALUDE_equation_solution_l1677_167759

theorem equation_solution (x : ℝ) : 
  (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 ↔ 
  (∃ k : ℤ, x = 2 * Real.pi / 3 + 2 * k * Real.pi ∨ 
            x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ 
            x = -Real.pi / 6 + 2 * k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1677_167759


namespace NUMINAMATH_CALUDE_solve_bracket_equation_l1677_167794

-- Define the bracket function
def bracket (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

-- State the theorem
theorem solve_bracket_equation :
  ∃ x : ℤ, (bracket 6) * (bracket x) = 28 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_bracket_equation_l1677_167794


namespace NUMINAMATH_CALUDE_two_out_of_three_accurate_l1677_167766

/-- The probability of an accurate forecast -/
def p_accurate : ℝ := 0.9

/-- The probability of an inaccurate forecast -/
def p_inaccurate : ℝ := 1 - p_accurate

/-- The probability of exactly 2 out of 3 forecasts being accurate -/
def p_two_accurate : ℝ := 3 * (p_accurate ^ 2 * p_inaccurate)

theorem two_out_of_three_accurate :
  p_two_accurate = 0.243 := by sorry

end NUMINAMATH_CALUDE_two_out_of_three_accurate_l1677_167766


namespace NUMINAMATH_CALUDE_sqrt_17_property_l1677_167731

theorem sqrt_17_property (a b : ℝ) : 
  (∀ x : ℤ, (x : ℝ) ≤ Real.sqrt 17 → (x + 1 : ℝ) > Real.sqrt 17 → a = x) →
  b = Real.sqrt 17 - a →
  b ^ 2020 * (a + Real.sqrt 17) ^ 2021 = Real.sqrt 17 + 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_17_property_l1677_167731


namespace NUMINAMATH_CALUDE_watch_cost_price_l1677_167754

theorem watch_cost_price (loss_percentage : ℚ) (gain_percentage : ℚ) (price_difference : ℚ) :
  loss_percentage = 21/100 →
  gain_percentage = 4/100 →
  price_difference = 140 →
  ∃ (cost_price : ℚ),
    cost_price * (1 - loss_percentage) + price_difference = cost_price * (1 + gain_percentage) ∧
    cost_price = 560 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1677_167754


namespace NUMINAMATH_CALUDE_otimes_difference_l1677_167760

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := a^3 / b^2

-- State the theorem
theorem otimes_difference : 
  (otimes (otimes 2 4) 6) - (otimes 2 (otimes 4 6)) = -23327/288 := by
  sorry

end NUMINAMATH_CALUDE_otimes_difference_l1677_167760


namespace NUMINAMATH_CALUDE_gp_solution_and_sum_l1677_167700

/-- Given a real number x, returns true if 10+x, 30+x, and 90+x form a geometric progression -/
def isGeometricProgression (x : ℝ) : Prop :=
  (30 + x)^2 = (10 + x) * (90 + x)

/-- Computes the sum of the terms in the progression for a given x -/
def sumOfProgression (x : ℝ) : ℝ :=
  (10 + x) + (30 + x) + (90 + x)

theorem gp_solution_and_sum :
  ∃! x : ℝ, isGeometricProgression x ∧ sumOfProgression x = 130 :=
sorry

end NUMINAMATH_CALUDE_gp_solution_and_sum_l1677_167700


namespace NUMINAMATH_CALUDE_speeding_ticket_problem_l1677_167747

theorem speeding_ticket_problem (total_motorists : ℝ) 
  (h1 : total_motorists > 0) 
  (h2 : total_motorists * 0.4 = total_motorists * 0.5 - (total_motorists * 0.5 - total_motorists * 0.4)) :
  (total_motorists * 0.5 - total_motorists * 0.4) / (total_motorists * 0.5) = 0.2 := by
  sorry

#check speeding_ticket_problem

end NUMINAMATH_CALUDE_speeding_ticket_problem_l1677_167747


namespace NUMINAMATH_CALUDE_expression_simplification_l1677_167725

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 * a^2 - 9 * b^2) / (3 * a * b) - (6 * a * b - 9 * b^2) / (4 * a * b - 3 * a^2) =
  2 * (a^2 - 9 * b^2) / (3 * a * b) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1677_167725


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_l1677_167749

def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_intersection :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Aᶜ ∩ B) = {x | (8 ≤ x ∧ x < 10) ∨ (2 < x ∧ x < 4)}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty ↔ a > 4) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_l1677_167749


namespace NUMINAMATH_CALUDE_fathers_age_l1677_167732

theorem fathers_age (father daughter : ℕ) 
  (h1 : father = 4 * daughter)
  (h2 : father + daughter + 10 = 50) : 
  father = 32 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l1677_167732


namespace NUMINAMATH_CALUDE_square_sum_ge_double_product_l1677_167762

theorem square_sum_ge_double_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_double_product_l1677_167762


namespace NUMINAMATH_CALUDE_fraction_simplification_l1677_167798

/-- The number of quarters Sarah has -/
def total_quarters : ℕ := 30

/-- The number of states that joined the union from 1790 to 1799 -/
def states_1790_1799 : ℕ := 8

/-- The fraction of Sarah's quarters representing states that joined from 1790 to 1799 -/
def fraction_1790_1799 : ℚ := states_1790_1799 / total_quarters

theorem fraction_simplification :
  fraction_1790_1799 = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1677_167798


namespace NUMINAMATH_CALUDE_sample_size_major_C_l1677_167751

/-- Represents the number of students in each major -/
structure CollegeMajors where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Calculates the total number of students across all majors -/
def totalStudents (majors : CollegeMajors) : Nat :=
  majors.A + majors.B + majors.C + majors.D

/-- Calculates the number of students to be sampled from a specific major -/
def sampleSize (majors : CollegeMajors) (totalSample : Nat) (majorSize : Nat) : Nat :=
  (majorSize * totalSample) / totalStudents majors

/-- Theorem: The number of students to be sampled from major C is 16 -/
theorem sample_size_major_C :
  let majors : CollegeMajors := { A := 150, B := 150, C := 400, D := 300 }
  let totalSample : Nat := 40
  sampleSize majors totalSample majors.C = 16 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_major_C_l1677_167751


namespace NUMINAMATH_CALUDE_overall_correct_percent_l1677_167703

def math_problems : ℕ := 30
def science_problems : ℕ := 20
def history_problems : ℕ := 50

def math_correct_percent : ℚ := 85 / 100
def science_correct_percent : ℚ := 75 / 100
def history_correct_percent : ℚ := 65 / 100

def total_problems : ℕ := math_problems + science_problems + history_problems

def total_correct : ℚ := 
  math_problems * math_correct_percent + 
  science_problems * science_correct_percent + 
  history_problems * history_correct_percent

theorem overall_correct_percent : 
  (total_correct / total_problems) * 100 = 73 := by sorry

end NUMINAMATH_CALUDE_overall_correct_percent_l1677_167703


namespace NUMINAMATH_CALUDE_intersection_reciprocals_sum_l1677_167724

/-- Circle C with equation x^2 + y^2 + 2x - 3 = 0 -/
def CircleC (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0

/-- Line l passing through the origin with slope k -/
def LineL (k x y : ℝ) : Prop := y = k * x

/-- Theorem: For any line passing through the origin and intersecting CircleC, 
    the sum of reciprocals of x-coordinates of intersection points is 2/3 -/
theorem intersection_reciprocals_sum (k : ℝ) (hk : k ≠ 0) : 
  ∃ x₁ x₂ y₁ y₂ : ℝ, 
    CircleC x₁ y₁ ∧ CircleC x₂ y₂ ∧ 
    LineL k x₁ y₁ ∧ LineL k x₂ y₂ ∧
    x₁ ≠ x₂ ∧ 
    1 / x₁ + 1 / x₂ = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_reciprocals_sum_l1677_167724


namespace NUMINAMATH_CALUDE_x_squared_gt_1_sufficient_not_necessary_for_reciprocal_lt_1_l1677_167722

theorem x_squared_gt_1_sufficient_not_necessary_for_reciprocal_lt_1 :
  (∀ x : ℝ, x^2 > 1 → 1/x < 1) ∧
  (∃ x : ℝ, 1/x < 1 ∧ ¬(x^2 > 1)) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_gt_1_sufficient_not_necessary_for_reciprocal_lt_1_l1677_167722


namespace NUMINAMATH_CALUDE_alice_bushes_l1677_167701

/-- The number of bushes Alice needs to buy for her yard -/
def bushes_needed (sides : ℕ) (side_length : ℕ) (bush_length : ℕ) : ℕ :=
  (sides * side_length) / bush_length

/-- Theorem: Alice needs to buy 12 bushes -/
theorem alice_bushes :
  bushes_needed 3 16 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_alice_bushes_l1677_167701
