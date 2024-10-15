import Mathlib

namespace NUMINAMATH_CALUDE_total_sales_equals_205_l1387_138793

def apple_price : ℝ := 1.50
def orange_price : ℝ := 1.00

def morning_apples : ℕ := 40
def morning_oranges : ℕ := 30
def afternoon_apples : ℕ := 50
def afternoon_oranges : ℕ := 40

def total_sales : ℝ :=
  apple_price * (morning_apples + afternoon_apples) +
  orange_price * (morning_oranges + afternoon_oranges)

theorem total_sales_equals_205 : total_sales = 205 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_equals_205_l1387_138793


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1387_138749

theorem quadratic_inequality_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1387_138749


namespace NUMINAMATH_CALUDE_union_equals_universal_l1387_138730

def U : Set ℕ := {2, 3, 4, 5, 6, 7}
def M : Set ℕ := {3, 4, 5, 7}
def N : Set ℕ := {2, 4, 5, 6}

theorem union_equals_universal : M ∪ N = U := by
  sorry

end NUMINAMATH_CALUDE_union_equals_universal_l1387_138730


namespace NUMINAMATH_CALUDE_complex_quotient_plus_modulus_l1387_138701

theorem complex_quotient_plus_modulus :
  let z₁ : ℂ := 2 - I
  let z₂ : ℂ := -I
  z₁ / z₂ + Complex.abs z₂ = 2 + 2 * I := by sorry

end NUMINAMATH_CALUDE_complex_quotient_plus_modulus_l1387_138701


namespace NUMINAMATH_CALUDE_output_for_twelve_l1387_138787

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 25 then
    step1 - 5
  else
    step1 * 2

theorem output_for_twelve : function_machine 12 = 31 := by
  sorry

end NUMINAMATH_CALUDE_output_for_twelve_l1387_138787


namespace NUMINAMATH_CALUDE_bicycle_trip_time_l1387_138716

theorem bicycle_trip_time (adam_speed simon_speed separation_distance : ℝ) 
  (adam_speed_pos : adam_speed > 0)
  (simon_speed_pos : simon_speed > 0)
  (separation_distance_pos : separation_distance > 0)
  (h_adam : adam_speed = 12)
  (h_simon : simon_speed = 9)
  (h_separation : separation_distance = 90) : 
  ∃ t : ℝ, t > 0 ∧ t * (adam_speed ^ 2 + simon_speed ^ 2) ^ (1/2 : ℝ) = separation_distance ∧ t = 6 :=
sorry

end NUMINAMATH_CALUDE_bicycle_trip_time_l1387_138716


namespace NUMINAMATH_CALUDE_distance_between_first_and_last_trees_l1387_138745

/-- Given 30 trees along a straight road with 3 meters between each adjacent pair of trees,
    the distance between the first and last trees is 87 meters. -/
theorem distance_between_first_and_last_trees (num_trees : ℕ) (distance_between_trees : ℝ) :
  num_trees = 30 →
  distance_between_trees = 3 →
  (num_trees - 1) * distance_between_trees = 87 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_first_and_last_trees_l1387_138745


namespace NUMINAMATH_CALUDE_translated_quadratic_increases_l1387_138724

/-- Original quadratic function -/
def f (x : ℝ) : ℝ := -x^2 + 1

/-- Translated quadratic function -/
def g (x : ℝ) : ℝ := f (x - 2)

/-- Theorem stating that the translated function increases for x < 2 -/
theorem translated_quadratic_increases (x1 x2 : ℝ) 
  (h1 : x1 < 2) (h2 : x2 < 2) (h3 : x1 < x2) : 
  g x1 < g x2 := by
  sorry

end NUMINAMATH_CALUDE_translated_quadratic_increases_l1387_138724


namespace NUMINAMATH_CALUDE_airplane_average_speed_l1387_138746

/-- The average speed of an airplane -/
theorem airplane_average_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 1584) 
  (h2 : time = 24) 
  (h3 : speed = distance / time) : speed = 66 := by
  sorry

end NUMINAMATH_CALUDE_airplane_average_speed_l1387_138746


namespace NUMINAMATH_CALUDE_piglet_growth_period_l1387_138740

/-- Represents the problem of determining the growth period for piglets --/
theorem piglet_growth_period (num_piglets : ℕ) (sale_price : ℕ) (feed_cost : ℕ) 
  (num_sold_early : ℕ) (num_sold_late : ℕ) (late_sale_months : ℕ) (total_profit : ℕ) :
  num_piglets = 6 →
  sale_price = 300 →
  feed_cost = 10 →
  num_sold_early = 3 →
  num_sold_late = 3 →
  late_sale_months = 16 →
  total_profit = 960 →
  ∃ x : ℕ, 
    x = 12 ∧
    (num_sold_early * sale_price + num_sold_late * sale_price) - 
    (num_sold_early * feed_cost * x + num_sold_late * feed_cost * late_sale_months) = total_profit :=
by sorry

end NUMINAMATH_CALUDE_piglet_growth_period_l1387_138740


namespace NUMINAMATH_CALUDE_square_hexagon_side_ratio_l1387_138709

theorem square_hexagon_side_ratio :
  ∀ (s_s s_h : ℝ),
  s_s > 0 → s_h > 0 →
  s_s^2 = (3 * s_h^2 * Real.sqrt 3) / 2 →
  s_s / s_h = Real.sqrt ((3 * Real.sqrt 3) / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_square_hexagon_side_ratio_l1387_138709


namespace NUMINAMATH_CALUDE_triangle_side_length_l1387_138753

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that when a = 2, b = 3, and angle C = 60°, the length of side c (AB) is √7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  b = 3 →
  C = Real.pi / 3 →  -- 60° in radians
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1387_138753


namespace NUMINAMATH_CALUDE_total_apples_picked_l1387_138732

/-- Given that Benny picked 2 apples and Dan picked 9 apples, 
    prove that the total number of apples picked is 11. -/
theorem total_apples_picked (benny_apples dan_apples : ℕ) 
  (h1 : benny_apples = 2) (h2 : dan_apples = 9) : 
  benny_apples + dan_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_picked_l1387_138732


namespace NUMINAMATH_CALUDE_tetrakis_hexahedron_colorings_l1387_138760

/-- The number of faces in a regular tetrakis hexahedron -/
def num_faces : ℕ := 16

/-- The number of available colors -/
def num_colors : ℕ := 12

/-- The order of the rotational symmetry group of a tetrakis hexahedron -/
def symmetry_order : ℕ := 24

/-- Calculates the number of permutations of k items chosen from n items -/
def permutations (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

/-- The number of distinguishable colorings of a tetrakis hexahedron -/
def distinguishable_colorings : ℕ :=
  permutations num_colors (num_faces - 1) / symmetry_order

theorem tetrakis_hexahedron_colorings :
  distinguishable_colorings = 479001600 := by
  sorry

end NUMINAMATH_CALUDE_tetrakis_hexahedron_colorings_l1387_138760


namespace NUMINAMATH_CALUDE_vector_equation_l1387_138777

variable {V : Type*} [AddCommGroup V]

theorem vector_equation (A B C : V) : (C - A) - (C - B) = B - A := by sorry

end NUMINAMATH_CALUDE_vector_equation_l1387_138777


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_I_simplify_trigonometric_expression_II_l1387_138789

-- Part I
theorem simplify_trigonometric_expression_I :
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (20 * π / 180))) /
  (Real.sin (160 * π / 180) - Real.sqrt (1 - Real.sin (20 * π / 180) ^ 2)) = -1 := by sorry

-- Part II
theorem simplify_trigonometric_expression_II (α : Real) (h : π / 2 < α ∧ α < π) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) +
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) =
  Real.sin α - Real.cos α := by sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_I_simplify_trigonometric_expression_II_l1387_138789


namespace NUMINAMATH_CALUDE_simplify_fraction_l1387_138788

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  x / (x - 1)^2 - 1 / (x - 1)^2 = 1 / (x - 1) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1387_138788


namespace NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l1387_138718

theorem prime_divisors_of_50_factorial (n : ℕ) :
  (n = 50) →
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card =
  (Finset.filter (λ p => p.Prime ∧ p ∣ n!) (Finset.range (n + 1))).card :=
sorry

end NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l1387_138718


namespace NUMINAMATH_CALUDE_frog_dog_ratio_l1387_138710

theorem frog_dog_ratio (dogs : ℕ) (cats : ℕ) (frogs : ℕ) : 
  cats = (80 * dogs) / 100 →
  frogs = 160 →
  dogs + cats + frogs = 304 →
  frogs = 2 * dogs :=
by sorry

end NUMINAMATH_CALUDE_frog_dog_ratio_l1387_138710


namespace NUMINAMATH_CALUDE_base_9_digits_of_2500_l1387_138774

/-- The number of digits in the base-9 representation of a positive integer -/
def num_digits_base_9 (n : ℕ+) : ℕ :=
  Nat.log 9 n.val + 1

/-- Theorem: The number of digits in the base-9 representation of 2500 is 4 -/
theorem base_9_digits_of_2500 : num_digits_base_9 2500 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_9_digits_of_2500_l1387_138774


namespace NUMINAMATH_CALUDE_binomial_coefficient_relation_l1387_138731

theorem binomial_coefficient_relation (n : ℕ) : 
  (Nat.choose n 3 = 7 * Nat.choose n 1) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_relation_l1387_138731


namespace NUMINAMATH_CALUDE_expansion_coefficient_equation_l1387_138714

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^2 in the expansion of (√x + a)^6
def coefficient_x_squared (a : ℝ) : ℝ := binomial 6 2 * a^2

-- State the theorem
theorem expansion_coefficient_equation :
  ∃ a : ℝ, coefficient_x_squared a = 60 ∧ (a = 2 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_equation_l1387_138714


namespace NUMINAMATH_CALUDE_brick_height_l1387_138706

/-- Represents a rectangular solid brick made of unit cubes -/
structure RectangularBrick where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of a rectangular brick in unit cubes -/
def RectangularBrick.volume (brick : RectangularBrick) : ℕ :=
  brick.length * brick.width * brick.height

/-- The perimeter of the base of a rectangular brick -/
def RectangularBrick.basePerimeter (brick : RectangularBrick) : ℕ :=
  2 * (brick.length + brick.width)

theorem brick_height (brick : RectangularBrick) :
  brick.volume = 42 ∧
  brick.basePerimeter = 18 →
  brick.height = 3 := by
  sorry

end NUMINAMATH_CALUDE_brick_height_l1387_138706


namespace NUMINAMATH_CALUDE_division_of_fractions_l1387_138743

theorem division_of_fractions : 
  (-1/24) / ((1/3) - (1/6) + (3/8)) = -1/13 := by sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1387_138743


namespace NUMINAMATH_CALUDE_min_daily_pages_for_given_plan_l1387_138758

/-- Represents a reading plan for a book -/
structure ReadingPlan where
  total_pages : ℕ
  total_days : ℕ
  initial_days : ℕ
  initial_pages : ℕ

/-- Calculates the minimum pages to read daily for the remaining days -/
def min_daily_pages (plan : ReadingPlan) : ℕ :=
  ((plan.total_pages - plan.initial_pages) + (plan.total_days - plan.initial_days - 1)) / (plan.total_days - plan.initial_days)

/-- Theorem stating the minimum daily pages for the given reading plan -/
theorem min_daily_pages_for_given_plan :
  let plan := ReadingPlan.mk 400 10 5 100
  min_daily_pages plan = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_daily_pages_for_given_plan_l1387_138758


namespace NUMINAMATH_CALUDE_increasing_function_inequality_range_l1387_138770

/-- Given an increasing function f defined on [0,+∞), 
    prove that the range of x satisfying f(2x-1) < f(1/3) is [1/2, 2/3). -/
theorem increasing_function_inequality_range 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_domain : ∀ x, x ∈ Set.Ici (0 : ℝ) → f x ∈ Set.univ) :
  {x : ℝ | f (2*x - 1) < f (1/3)} = Set.Icc (1/2 : ℝ) (2/3) := by
sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_range_l1387_138770


namespace NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l1387_138705

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ × ℝ)

-- Define properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- Right angle at F (we don't need to specify this explicitly in Lean)
  true

def angle_D_is_60_degrees (t : Triangle) : Prop :=
  -- Angle D is 60 degrees
  true

def DF_length (t : Triangle) : ℝ :=
  12

-- Define the incircle radius function
noncomputable def incircle_radius (t : Triangle) : ℝ :=
  sorry

-- Theorem statement
theorem incircle_radius_of_special_triangle (t : Triangle) 
  (h1 : is_right_triangle t)
  (h2 : angle_D_is_60_degrees t)
  (h3 : DF_length t = 12) :
  incircle_radius t = 6 * (Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l1387_138705


namespace NUMINAMATH_CALUDE_angelina_speed_to_library_l1387_138783

-- Define the distances
def home_to_grocery : ℝ := 150
def grocery_to_gym : ℝ := 200
def gym_to_park : ℝ := 250
def park_to_library : ℝ := 300

-- Define Angelina's initial speed
def v : ℝ := 5

-- Define the theorem
theorem angelina_speed_to_library :
  let time_home_to_grocery := home_to_grocery / v
  let time_grocery_to_gym := grocery_to_gym / (2 * v)
  let time_gym_to_park := gym_to_park / (v / 2)
  let time_park_to_library := park_to_library / (6 * v)
  time_grocery_to_gym = time_home_to_grocery - 10 ∧
  time_gym_to_park = time_park_to_library + 20 →
  6 * v = 30 := by
  sorry

end NUMINAMATH_CALUDE_angelina_speed_to_library_l1387_138783


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l1387_138729

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (x + 5)^2 = (4*y - 3)^2 - 140

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

/-- Theorem stating that the given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l1387_138729


namespace NUMINAMATH_CALUDE_average_temperature_proof_l1387_138794

/-- Given the average temperature for four days and individual temperatures for two days,
    prove that the average temperature for a different set of four days is as calculated. -/
theorem average_temperature_proof
  (avg_mon_to_thu : ℝ)
  (temp_mon : ℝ)
  (temp_fri : ℝ)
  (h1 : avg_mon_to_thu = 48)
  (h2 : temp_mon = 40)
  (h3 : temp_fri = 32) :
  (4 * avg_mon_to_thu - temp_mon + temp_fri) / 4 = 46 := by
  sorry


end NUMINAMATH_CALUDE_average_temperature_proof_l1387_138794


namespace NUMINAMATH_CALUDE_container_capacity_l1387_138767

theorem container_capacity (initial_fill : Real) (added_water : Real) (final_fill : Real) :
  initial_fill = 0.3 →
  added_water = 18 →
  final_fill = 0.75 →
  ∃ capacity : Real, 
    capacity * final_fill - capacity * initial_fill = added_water ∧
    capacity = 40 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l1387_138767


namespace NUMINAMATH_CALUDE_graces_age_fraction_l1387_138766

theorem graces_age_fraction (mother_age : ℕ) (grace_age : ℕ) :
  mother_age = 80 →
  grace_age = 60 →
  (grace_age : ℚ) / ((2 * mother_age) : ℚ) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_graces_age_fraction_l1387_138766


namespace NUMINAMATH_CALUDE_sixteen_letters_with_both_l1387_138781

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total : ℕ)
  (only_line : ℕ)
  (only_dot : ℕ)
  (both : ℕ)
  (all_have_feature : only_line + only_dot + both = total)

/-- The number of letters with both a dot and a straight line in the given alphabet -/
def letters_with_both (a : Alphabet) : ℕ := a.both

/-- Theorem stating that in the given alphabet, 16 letters contain both a dot and a straight line -/
theorem sixteen_letters_with_both (a : Alphabet) 
  (h1 : a.total = 50)
  (h2 : a.only_line = 30)
  (h3 : a.only_dot = 4) :
  letters_with_both a = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_letters_with_both_l1387_138781


namespace NUMINAMATH_CALUDE_book_sale_problem_l1387_138720

theorem book_sale_problem (total_cost book1_cost book2_cost selling_price : ℚ) :
  total_cost = 300 ∧
  book1_cost + book2_cost = total_cost ∧
  selling_price = book1_cost * (1 - 15/100) ∧
  selling_price = book2_cost * (1 + 19/100) →
  book1_cost = 175 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_problem_l1387_138720


namespace NUMINAMATH_CALUDE_cube_surface_area_l1387_138772

-- Define the volume of the cube
def cube_volume : ℝ := 4913

-- Define the surface area we want to prove
def target_surface_area : ℝ := 1734

-- Theorem statement
theorem cube_surface_area :
  let side := (cube_volume ^ (1/3 : ℝ))
  6 * side^2 = target_surface_area := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1387_138772


namespace NUMINAMATH_CALUDE_coin_difference_l1387_138712

/-- Represents the denominations of coins available --/
inductive Coin
  | Five : Coin
  | Ten : Coin
  | Twenty : Coin

/-- The value of a coin in cents --/
def coinValue : Coin → Nat
  | Coin.Five => 5
  | Coin.Ten => 10
  | Coin.Twenty => 20

/-- The target amount to be paid in cents --/
def targetAmount : Nat := 40

/-- A function that calculates the minimum number of coins needed --/
def minCoins : Nat := sorry

/-- A function that calculates the maximum number of coins needed --/
def maxCoins : Nat := sorry

/-- Theorem stating the difference between max and min number of coins --/
theorem coin_difference : maxCoins - minCoins = 6 := by sorry

end NUMINAMATH_CALUDE_coin_difference_l1387_138712


namespace NUMINAMATH_CALUDE_principal_calculation_l1387_138775

def interest_rates : List ℚ := [6/100, 75/1000, 8/100, 85/1000, 9/100]

theorem principal_calculation (total_interest : ℚ) (rates : List ℚ) :
  total_interest = 6016.75 ∧ rates = interest_rates →
  (total_interest / rates.sum) = 15430 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1387_138775


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l1387_138735

/-- Two points are symmetric about the y-axis if their y-coordinates are the same and their x-coordinates are opposites -/
def symmetric_about_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = y₂ ∧ x₁ = -x₂

/-- The problem statement -/
theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis a 3 4 b →
  (a + b)^2012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l1387_138735


namespace NUMINAMATH_CALUDE_solution_pairs_l1387_138771

theorem solution_pairs : 
  {(x, y) : ℝ × ℝ | (x^2 + y + 1) * (y^2 + x + 1) = 4 ∧ (x^2 + y)^2 + (y^2 + x)^2 = 2} = 
  {(0, 1), (1, 0), ((Real.sqrt 5 - 1) / 2, (Real.sqrt 5 - 1) / 2), 
   (-(Real.sqrt 5 + 1) / 2, -(Real.sqrt 5 + 1) / 2)} := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l1387_138771


namespace NUMINAMATH_CALUDE_retail_price_calculation_l1387_138703

def wholesale_price : ℝ := 90

def discount_rate : ℝ := 0.10

def profit_rate : ℝ := 0.20

def retail_price : ℝ := 120

theorem retail_price_calculation :
  let profit := profit_rate * wholesale_price
  let selling_price := wholesale_price + profit
  selling_price = retail_price * (1 - discount_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l1387_138703


namespace NUMINAMATH_CALUDE_classroom_notebooks_l1387_138739

theorem classroom_notebooks :
  ∀ (x : ℕ),
    (28 : ℕ) / 2 * x + (28 : ℕ) / 2 * 3 = 112 →
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_classroom_notebooks_l1387_138739


namespace NUMINAMATH_CALUDE_round_table_seats_l1387_138747

/-- Represents a round table with equally spaced seats -/
structure RoundTable where
  total_seats : ℕ
  seat_numbers : Fin total_seats → ℕ

/-- Two seats are opposite if they are half the total number of seats apart -/
def opposite (t : RoundTable) (s1 s2 : Fin t.total_seats) : Prop :=
  (t.seat_numbers s2 - t.seat_numbers s1) % t.total_seats = t.total_seats / 2

theorem round_table_seats (t : RoundTable) (s1 s2 : Fin t.total_seats) :
  t.seat_numbers s1 = 10 ∧ t.seat_numbers s2 = 29 ∧ opposite t s1 s2 → t.total_seats = 38 :=
by
  sorry


end NUMINAMATH_CALUDE_round_table_seats_l1387_138747


namespace NUMINAMATH_CALUDE_line_intercepts_l1387_138708

/-- Given a line with equation x - 2y - 2 = 0, prove that its x-intercept is 2 and y-intercept is -1 -/
theorem line_intercepts :
  let line := {(x, y) : ℝ × ℝ | x - 2*y - 2 = 0}
  let x_intercept := {x : ℝ | ∃ y, (x, y) ∈ line ∧ y = 0}
  let y_intercept := {y : ℝ | ∃ x, (x, y) ∈ line ∧ x = 0}
  x_intercept = {2} ∧ y_intercept = {-1} := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_l1387_138708


namespace NUMINAMATH_CALUDE_max_profit_theorem_additional_cost_range_l1387_138738

/-- Represents the monthly sales and profit model for a product. -/
structure SalesModel where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ
  max_price : ℝ

/-- Calculates the monthly sales volume given a price increase. -/
def sales_volume (model : SalesModel) (price_increase : ℝ) : ℝ :=
  model.initial_sales - model.price_sensitivity * price_increase

/-- Calculates the monthly profit given a price increase. -/
def monthly_profit (model : SalesModel) (price_increase : ℝ) : ℝ :=
  (sales_volume model price_increase) * (model.initial_price + price_increase - model.cost_price)

/-- Theorem stating the maximum monthly profit and optimal selling price. -/
theorem max_profit_theorem (model : SalesModel) 
  (h_cost : model.cost_price = 40)
  (h_initial_price : model.initial_price = 50)
  (h_initial_sales : model.initial_sales = 210)
  (h_price_sensitivity : model.price_sensitivity = 10)
  (h_max_price : model.max_price = 65) :
  ∃ (x : ℝ), x ∈ Set.Icc 5 6 ∧ 
  ∀ (y : ℝ), y > 0 ∧ y ≤ 15 → monthly_profit model x ≥ monthly_profit model y ∧
  monthly_profit model x = 2400 := by sorry

/-- Theorem stating the range of additional costs. -/
theorem additional_cost_range (model : SalesModel) (a : ℝ)
  (h_cost : model.cost_price = 40)
  (h_initial_price : model.initial_price = 50)
  (h_initial_sales : model.initial_sales = 210)
  (h_price_sensitivity : model.price_sensitivity = 10)
  (h_max_price : model.max_price = 65) :
  (∀ (x y : ℝ), 8 ≤ x ∧ x < y ∧ y ≤ 15 → 
    monthly_profit model x - (sales_volume model x * a) > 
    monthly_profit model y - (sales_volume model y * a)) 
  ↔ 0 < a ∧ a < 6 := by sorry

end NUMINAMATH_CALUDE_max_profit_theorem_additional_cost_range_l1387_138738


namespace NUMINAMATH_CALUDE_jamie_grape_juice_theorem_l1387_138761

/-- The amount of grape juice Jamie had at recess -/
def grape_juice_amount (max_liquid bathroom_threshold planned_water milk_amount : ℕ) : ℕ :=
  max_liquid - bathroom_threshold - planned_water - milk_amount

theorem jamie_grape_juice_theorem :
  grape_juice_amount 32 0 8 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_jamie_grape_juice_theorem_l1387_138761


namespace NUMINAMATH_CALUDE_interest_rate_problem_l1387_138734

/-- Prove that for given conditions, the interest rate is 4% --/
theorem interest_rate_problem (P t : ℝ) (diff : ℝ) (h1 : P = 2000) (h2 : t = 2) (h3 : diff = 3.20) :
  ∃ r : ℝ, r = 4 ∧ 
    P * ((1 + r / 100) ^ t - 1) - (P * r * t / 100) = diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l1387_138734


namespace NUMINAMATH_CALUDE_prob_one_black_in_three_draws_l1387_138784

-- Define the number of balls
def total_balls : ℕ := 6
def black_balls : ℕ := 2
def white_balls : ℕ := 4

-- Define the number of draws
def num_draws : ℕ := 3

-- Define the probability of drawing a black ball
def prob_black : ℚ := black_balls / total_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Define the number of ways to choose 1 draw out of 3
def ways_to_choose : ℕ := 3

-- Theorem to prove
theorem prob_one_black_in_three_draws : 
  ways_to_choose * prob_black * prob_white^2 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_black_in_three_draws_l1387_138784


namespace NUMINAMATH_CALUDE_derivative_at_two_l1387_138700

theorem derivative_at_two (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h_eq : ∀ x, f x = 2 * f (2 - x) - x^2 + 8*x - 8) : 
  deriv f 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_two_l1387_138700


namespace NUMINAMATH_CALUDE_parabola_focus_l1387_138792

/-- A parabola is defined by its equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of a parabola is a point -/
def Focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola y^2 = 4x is (1, 0) -/
theorem parabola_focus :
  ∀ (p : ℝ × ℝ), p ∈ Parabola → Focus = (1, 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1387_138792


namespace NUMINAMATH_CALUDE_sampling_methods_correct_l1387_138779

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Stratified
| Systematic

-- Define the tasks
structure Task1 where
  total_products : Nat
  sample_size : Nat

structure Task2 where
  total_students : Nat
  first_year : Nat
  second_year : Nat
  third_year : Nat
  sample_size : Nat

structure Task3 where
  rows : Nat
  seats_per_row : Nat
  sample_size : Nat

-- Define the function to determine the most reasonable sampling method
def most_reasonable_sampling_method (task1 : Task1) (task2 : Task2) (task3 : Task3) : 
  (SamplingMethod × SamplingMethod × SamplingMethod) :=
  (SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic)

-- Theorem statement
theorem sampling_methods_correct (task1 : Task1) (task2 : Task2) (task3 : Task3) :
  task1.total_products = 30 ∧ task1.sample_size = 3 ∧
  task2.total_students = 2460 ∧ task2.first_year = 890 ∧ task2.second_year = 820 ∧ 
  task2.third_year = 810 ∧ task2.sample_size = 300 ∧
  task3.rows = 28 ∧ task3.seats_per_row = 32 ∧ task3.sample_size = 28 →
  most_reasonable_sampling_method task1 task2 task3 = 
    (SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic) :=
by
  sorry

end NUMINAMATH_CALUDE_sampling_methods_correct_l1387_138779


namespace NUMINAMATH_CALUDE_egg_price_per_dozen_l1387_138773

/-- Calculates the price per dozen eggs given the number of hens, eggs laid per hen per week,
    number of weeks, and total revenue. -/
theorem egg_price_per_dozen 
  (num_hens : ℕ) 
  (eggs_per_hen_per_week : ℕ) 
  (num_weeks : ℕ) 
  (total_revenue : ℚ) : 
  num_hens = 10 →
  eggs_per_hen_per_week = 12 →
  num_weeks = 4 →
  total_revenue = 120 →
  (total_revenue / (↑(num_hens * eggs_per_hen_per_week * num_weeks) / 12)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_egg_price_per_dozen_l1387_138773


namespace NUMINAMATH_CALUDE_only_set2_forms_triangle_l1387_138742

-- Define a structure for a set of three line segments
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle inequality check
def satisfiesTriangleInequality (s : LineSegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

-- Define the given sets of line segments
def set1 : LineSegmentSet := ⟨1, 2, 3⟩
def set2 : LineSegmentSet := ⟨2, 3, 4⟩
def set3 : LineSegmentSet := ⟨4, 4, 8⟩
def set4 : LineSegmentSet := ⟨5, 6, 12⟩

-- Theorem stating that only set2 satisfies the triangle inequality
theorem only_set2_forms_triangle :
  ¬(satisfiesTriangleInequality set1) ∧
  (satisfiesTriangleInequality set2) ∧
  ¬(satisfiesTriangleInequality set3) ∧
  ¬(satisfiesTriangleInequality set4) :=
sorry

end NUMINAMATH_CALUDE_only_set2_forms_triangle_l1387_138742


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l1387_138750

theorem unique_solution_lcm_gcd_equation :
  ∃! (n : ℕ), n > 0 ∧ Nat.lcm n 60 = 2 * Nat.gcd n 60 + 300 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l1387_138750


namespace NUMINAMATH_CALUDE_negation_equivalence_l1387_138776

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2016 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1387_138776


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1387_138778

/-- Two circles with equations x^2 + y^2 + 2ax + a^2 - 9 = 0 and x^2 + y^2 - 4by - 1 + 4b^2 = 0 -/
def Circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 9 = 0
def Circle2 (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The circles have three shared tangents -/
axiom three_shared_tangents (a b : ℝ) : ∃ (t1 t2 t3 : ℝ × ℝ → ℝ), 
  (∀ x y, Circle1 a x y → t1 (x, y) = 0) ∧
  (∀ x y, Circle2 b x y → t1 (x, y) = 0) ∧
  (∀ x y, Circle1 a x y → t2 (x, y) = 0) ∧
  (∀ x y, Circle2 b x y → t2 (x, y) = 0) ∧
  (∀ x y, Circle1 a x y → t3 (x, y) = 0) ∧
  (∀ x y, Circle2 b x y → t3 (x, y) = 0)

/-- The theorem to be proved -/
theorem min_value_of_expression (a b : ℝ) (h : a * b ≠ 0) : 
  (∃ (x y : ℝ), Circle1 a x y) → 
  (∃ (x y : ℝ), Circle2 b x y) → 
  (∀ c, c ≥ 1 → 4 / a^2 + 1 / b^2 ≥ c) ∧ 
  (∃ a0 b0, a0 * b0 ≠ 0 ∧ 4 / a0^2 + 1 / b0^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1387_138778


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l1387_138795

theorem rationalize_and_simplify :
  ∃ (A B C : ℤ), 
    (3 + Real.sqrt 2) / (2 - Real.sqrt 5) = 
      A + B * Real.sqrt C ∧ A * B * C = -24 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l1387_138795


namespace NUMINAMATH_CALUDE_sequence_sum_l1387_138737

/-- Given a sequence {a_n} with a₁ = 1 and S_{n+1} = ((n+1)a_n)/n + S_n, 
    prove that S_n = n(n+1)/2 for all positive integers n. -/
theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  a 1 = 1 → 
  (∀ n : ℕ, n > 0 → S (n + 1) = ((n + 1) * a n) / n + S n) → 
  (∀ n : ℕ, n > 0 → S n = n * (n + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1387_138737


namespace NUMINAMATH_CALUDE_triangle_area_l1387_138715

theorem triangle_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = Real.sqrt 19) :
  (1/2 : ℝ) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1387_138715


namespace NUMINAMATH_CALUDE_trinomial_zeros_l1387_138728

theorem trinomial_zeros (a b : ℝ) (ha : a > 4) (hb : b > 4) :
  (a^2 - 4*b > 0) ∨ (b^2 - 4*a > 0) := by sorry

end NUMINAMATH_CALUDE_trinomial_zeros_l1387_138728


namespace NUMINAMATH_CALUDE_trapezoid_de_length_l1387_138785

/-- Represents a trapezoid ABCD formed by a rectangle ABCE and a right triangle EDF -/
structure Trapezoid where
  /-- Length of side AB of the rectangle -/
  ab : ℝ
  /-- Length of side BC of the rectangle -/
  bc : ℝ
  /-- Length of side DE of the trapezoid -/
  de : ℝ
  /-- Length of side EF of the triangle -/
  ef : ℝ
  /-- Condition that AB = 7 -/
  ab_eq : ab = 7
  /-- Condition that BC = 8 -/
  bc_eq : bc = 8
  /-- Condition that DE is twice EF -/
  de_twice_ef : de = 2 * ef
  /-- Condition that the areas of the rectangle and triangle are equal -/
  areas_equal : ab * bc = (1 / 2) * de * ef

/-- Theorem stating that the length of DE in the described trapezoid is 4√14 -/
theorem trapezoid_de_length (t : Trapezoid) : t.de = 4 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_de_length_l1387_138785


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1387_138799

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 30 →
  boys + girls = total_students →
  2 * girls = 3 * boys →
  girls - boys = 6 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1387_138799


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l1387_138782

theorem tangent_circles_radius (r₁ r₂ d : ℝ) : 
  r₁ = 2 →
  d = 5 →
  (r₁ + r₂ = d ∨ |r₁ - r₂| = d) →
  r₂ = 3 ∨ r₂ = 7 := by
sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l1387_138782


namespace NUMINAMATH_CALUDE_smallest_x_abs_equation_l1387_138722

theorem smallest_x_abs_equation : ∃ x : ℝ, x = -7 ∧ 
  (∀ y : ℝ, |4*y + 8| = 20 → y ≥ x) ∧ |4*x + 8| = 20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_abs_equation_l1387_138722


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1387_138719

theorem rectangle_ratio (w : ℝ) : 
  w > 0 ∧ 2 * (w + 10) = 30 → w / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1387_138719


namespace NUMINAMATH_CALUDE_go_relay_match_sequences_l1387_138757

/-- Represents the number of players in each team -/
def team_size : ℕ := 7

/-- Represents the maximum number of matches possible -/
def max_matches : ℕ := 2 * team_size - 1

/-- Represents the number of matches the winning team must win -/
def required_wins : ℕ := team_size

/-- The number of possible match sequences in a Go relay match -/
def match_sequences : ℕ := 2 * (Nat.choose max_matches required_wins)

theorem go_relay_match_sequences :
  match_sequences = 3432 :=
sorry

end NUMINAMATH_CALUDE_go_relay_match_sequences_l1387_138757


namespace NUMINAMATH_CALUDE_difficult_vs_easy_problems_l1387_138754

/-- Represents the number of problems solved by different combinations of students -/
structure ProblemDistribution where
  x₁ : ℕ  -- problems solved only by student 1
  x₂ : ℕ  -- problems solved only by student 2
  x₃ : ℕ  -- problems solved only by student 3
  y₁₂ : ℕ -- problems solved only by students 1 and 2
  y₁₃ : ℕ -- problems solved only by students 1 and 3
  y₂₃ : ℕ -- problems solved only by students 2 and 3
  z : ℕ   -- problems solved by all three students

/-- The main theorem stating the relationship between difficult and easy problems -/
theorem difficult_vs_easy_problems (d : ProblemDistribution) :
  d.x₁ + d.x₂ + d.x₃ + d.y₁₂ + d.y₁₃ + d.y₂₃ + d.z = 100 →
  d.x₁ + d.y₁₂ + d.y₁₃ + d.z = 60 →
  d.x₂ + d.y₁₂ + d.y₂₃ + d.z = 60 →
  d.x₃ + d.y₁₃ + d.y₂₃ + d.z = 60 →
  d.x₁ + d.x₂ + d.x₃ - d.z = 20 :=
by sorry

end NUMINAMATH_CALUDE_difficult_vs_easy_problems_l1387_138754


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_neg_two_l1387_138717

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of real number a. -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 1)

/-- If z(a) is a pure imaginary number, then a = -2. -/
theorem pure_imaginary_implies_a_eq_neg_two :
  ∀ a : ℝ, is_pure_imaginary (z a) → a = -2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_neg_two_l1387_138717


namespace NUMINAMATH_CALUDE_maximize_x_cubed_y_fifth_l1387_138796

theorem maximize_x_cubed_y_fifth (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  x^3 * y^5 ≤ (7.5^3) * (12.5^5) ∧ 
  (x^3 * y^5 = (7.5^3) * (12.5^5) ↔ x = 7.5 ∧ y = 12.5) := by
sorry

end NUMINAMATH_CALUDE_maximize_x_cubed_y_fifth_l1387_138796


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l1387_138798

theorem subtraction_multiplication_equality : (3.625 - 1.047) * 4 = 10.312 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l1387_138798


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1387_138707

theorem inequality_equivalence (x y : ℝ) :
  y - x > Real.sqrt (x^2 + 9) ↔ y > x + Real.sqrt (x^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1387_138707


namespace NUMINAMATH_CALUDE_fraction_equality_l1387_138744

theorem fraction_equality (p q r s : ℚ) 
  (h1 : p / q = 8)
  (h2 : r / q = 5)
  (h3 : r / s = 3 / 4) :
  s / p = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1387_138744


namespace NUMINAMATH_CALUDE_problem_solution_l1387_138764

theorem problem_solution (k : ℕ) (y : ℚ) 
  (h1 : (1/2)^18 * (1/81)^k = y)
  (h2 : k = 9) : 
  y = 1 / (2^18 * 3^36) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1387_138764


namespace NUMINAMATH_CALUDE_solve_equation_l1387_138797

theorem solve_equation (x : ℝ) (h : 8 * (2 + 1 / x) = 18) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1387_138797


namespace NUMINAMATH_CALUDE_speed_increases_with_height_l1387_138702

/-- Represents a data point of height and time -/
structure DataPoint where
  height : ℝ
  time : ℝ

/-- The data set from the experiment -/
def dataSet : List DataPoint := [
  ⟨10, 4.23⟩, ⟨20, 3.00⟩, ⟨30, 2.45⟩, ⟨40, 2.13⟩, 
  ⟨50, 1.89⟩, ⟨60, 1.71⟩, ⟨70, 1.59⟩
]

/-- Theorem stating that average speed increases with height -/
theorem speed_increases_with_height :
  ∀ (d1 d2 : DataPoint), 
    d1 ∈ dataSet → d2 ∈ dataSet →
    d2.height > d1.height → 
    d2.height / d2.time > d1.height / d1.time :=
by sorry

end NUMINAMATH_CALUDE_speed_increases_with_height_l1387_138702


namespace NUMINAMATH_CALUDE_abs_negative_2010_l1387_138752

theorem abs_negative_2010 : |(-2010 : ℤ)| = 2010 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2010_l1387_138752


namespace NUMINAMATH_CALUDE_three_different_days_probability_three_different_days_probability_value_l1387_138721

/-- The probability of three group members working on exactly three different days in a week -/
theorem three_different_days_probability : ℝ :=
  let total_outcomes := 7^3
  let favorable_outcomes := 7 * 6 * 5
  favorable_outcomes / total_outcomes

/-- The probability of three group members working on exactly three different days in a week is 30/49 -/
theorem three_different_days_probability_value : three_different_days_probability = 30 / 49 := by
  sorry

end NUMINAMATH_CALUDE_three_different_days_probability_three_different_days_probability_value_l1387_138721


namespace NUMINAMATH_CALUDE_election_vote_count_l1387_138765

theorem election_vote_count 
  (total_votes : ℕ) 
  (candidate_a_votes : ℕ) 
  (candidate_b_votes : ℕ) :
  (candidate_a_votes = candidate_b_votes + (15 * total_votes) / 100) →
  (candidate_b_votes = 3159) →
  ((80 * total_votes) / 100 = candidate_a_votes + candidate_b_votes) →
  (total_votes = 9720) :=
by sorry

end NUMINAMATH_CALUDE_election_vote_count_l1387_138765


namespace NUMINAMATH_CALUDE_distinct_weights_count_l1387_138755

def weights : List ℕ := [1, 2, 3, 4]

def possible_combinations (weights : List ℕ) : List (List ℕ) :=
  sorry

def distinct_weights (combinations : List (List ℕ)) : List ℕ :=
  sorry

theorem distinct_weights_count :
  weights.length = 4 →
  (distinct_weights (possible_combinations weights)).length = 10 :=
by sorry

end NUMINAMATH_CALUDE_distinct_weights_count_l1387_138755


namespace NUMINAMATH_CALUDE_exists_permutation_satisfying_average_condition_l1387_138713

/-- A permutation of the first n natural numbers. -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Predicate to check if a permutation satisfies the average condition. -/
def SatisfiesAverageCondition (n : ℕ) (p : Permutation n) : Prop :=
  ∀ i j k : Fin n, i < j → j < k →
    (p i).val + (p k).val ≠ 2 * (p j).val

/-- Theorem stating that for any n, there exists a permutation satisfying the average condition. -/
theorem exists_permutation_satisfying_average_condition (n : ℕ) :
  ∃ p : Permutation n, SatisfiesAverageCondition n p :=
sorry

end NUMINAMATH_CALUDE_exists_permutation_satisfying_average_condition_l1387_138713


namespace NUMINAMATH_CALUDE_chessboard_invariant_l1387_138780

/-- Represents a chessboard configuration -/
def Chessboard := Matrix (Fin 8) (Fin 8) Int

/-- Initial chessboard configuration -/
def initialBoard : Chessboard :=
  fun i j => if i = 1 ∧ j = 7 then -1 else 1

/-- Represents a move (changing signs in a row or column) -/
inductive Move
  | row (i : Fin 8)
  | col (j : Fin 8)

/-- Apply a move to a chessboard -/
def applyMove (b : Chessboard) (m : Move) : Chessboard :=
  match m with
  | Move.row i => fun r c => if r = i then -b r c else b r c
  | Move.col j => fun r c => if c = j then -b r c else b r c

/-- Apply a sequence of moves to a chessboard -/
def applyMoves (b : Chessboard) : List Move → Chessboard
  | [] => b
  | m :: ms => applyMoves (applyMove b m) ms

/-- Product of all numbers on the board -/
def boardProduct (b : Chessboard) : Int :=
  (Finset.univ.prod fun i => Finset.univ.prod fun j => b i j)

/-- Main theorem -/
theorem chessboard_invariant (moves : List Move) :
    boardProduct (applyMoves initialBoard moves) = -1 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_invariant_l1387_138780


namespace NUMINAMATH_CALUDE_circle_through_AB_with_center_on_line_l1387_138733

-- Define the points A and B
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (1, -3)

-- Define the line on which the center lies
def centerLine (x y : ℝ) : Prop := y = x + 3

-- Define the standard form of a circle
def isCircle (h k r x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

-- State the theorem
theorem circle_through_AB_with_center_on_line :
  ∃ (h k r : ℝ), 
    centerLine h k ∧
    isCircle h k r A.1 A.2 ∧
    isCircle h k r B.1 B.2 ∧
    h = -2 ∧ k = 1 ∧ r = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_through_AB_with_center_on_line_l1387_138733


namespace NUMINAMATH_CALUDE_middle_number_proof_l1387_138726

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
    (h3 : a + b = 15) (h4 : a + c = 18) (h5 : b + c = 21) : b = 9 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1387_138726


namespace NUMINAMATH_CALUDE_initial_typists_count_l1387_138751

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 20

/-- The number of letters typed by the initial group in 20 minutes -/
def letters_20min : ℕ := 40

/-- The number of typists in the second group -/
def second_typists : ℕ := 30

/-- The number of letters typed by the second group in 1 hour -/
def letters_1hour : ℕ := 180

/-- The rate of typing (letters per hour per typist) is consistent between groups -/
axiom typing_rate_consistent : 
  (letters_20min : ℚ) / initial_typists * 3 = (letters_1hour : ℚ) / second_typists

theorem initial_typists_count : initial_typists = 20 := by
  sorry

end NUMINAMATH_CALUDE_initial_typists_count_l1387_138751


namespace NUMINAMATH_CALUDE_square_roots_problem_l1387_138704

theorem square_roots_problem (a : ℝ) :
  (∃ x > 0, (2*a - 1)^2 = x ∧ (a - 2)^2 = x) → (2*a - 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1387_138704


namespace NUMINAMATH_CALUDE_two_distinct_roots_iff_p_condition_l1387_138786

theorem two_distinct_roots_iff_p_condition (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    ((x ≥ 0 → x^2 - 2*x - p = 0) ∧ (x < 0 → x^2 + 2*x - p = 0)) ∧
    ((y ≥ 0 → y^2 - 2*y - p = 0) ∧ (y < 0 → y^2 + 2*y - p = 0)))
  ↔ 
  (p > 0 ∨ p = -1) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_iff_p_condition_l1387_138786


namespace NUMINAMATH_CALUDE_base7_523_equals_base10_262_l1387_138769

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (a b c : ℕ) : ℕ :=
  a * 7^2 + b * 7^1 + c * 7^0

/-- The theorem stating that 523 in base-7 is equal to 262 in base-10 --/
theorem base7_523_equals_base10_262 : base7ToBase10 5 2 3 = 262 := by
  sorry

end NUMINAMATH_CALUDE_base7_523_equals_base10_262_l1387_138769


namespace NUMINAMATH_CALUDE_bag_volume_proof_l1387_138756

/-- The volume of a cuboid-shaped bag -/
def bag_volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a cuboid-shaped bag with width 9 cm, length 4 cm, and height 7 cm is 252 cm³ -/
theorem bag_volume_proof : bag_volume 9 4 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_bag_volume_proof_l1387_138756


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1387_138759

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (-2, 4)
  are_parallel a b → x = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1387_138759


namespace NUMINAMATH_CALUDE_proposition_p_true_q_false_l1387_138768

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define a triangle
structure Triangle :=
(A B C : ℝ)
(angle_sum : A + B + C = π)
(positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

theorem proposition_p_true_q_false :
  (∀ x : ℝ, 0 < x → x < 1 → lg (x * (1 - x) + 1) > 0) ∧
  (∃ t : Triangle, t.A > t.B ∧ Real.cos (t.A / 2)^2 ≥ Real.cos (t.B / 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_true_q_false_l1387_138768


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l1387_138741

theorem isosceles_triangle_sides (p a : ℝ) (h1 : p = 14) (h2 : a = 4) :
  (∃ b c : ℝ, (a + b + c = p ∧ (b = c ∨ a = b ∨ a = c)) →
    ((b = 5 ∧ c = 5) ∨ (b = 4 ∧ c = 6) ∨ (b = 6 ∧ c = 4))) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l1387_138741


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l1387_138791

/-- Represents a right circular cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Given two cylinders with equal volumes and the second radius 20% larger than the first,
    prove that the height of the first cylinder is 44% more than the height of the second --/
theorem cylinder_height_relationship (c1 c2 : Cylinder) 
    (h_volume : c1.radius^2 * c1.height = c2.radius^2 * c2.height)
    (h_radius : c2.radius = 1.2 * c1.radius) :
    c1.height = 1.44 * c2.height := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l1387_138791


namespace NUMINAMATH_CALUDE_sixth_quiz_score_for_target_mean_l1387_138725

def quiz_scores : List ℕ := [92, 96, 87, 89, 100]
def target_mean : ℕ := 94
def num_quizzes : ℕ := 6

theorem sixth_quiz_score_for_target_mean :
  ∃ (x : ℕ), (quiz_scores.sum + x) / num_quizzes = target_mean ∧ x = 100 := by
sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_for_target_mean_l1387_138725


namespace NUMINAMATH_CALUDE_exam_score_problem_l1387_138763

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 80)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 120) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * (correct_answers : ℤ) + wrong_score * ((total_questions - correct_answers) : ℤ) = total_score ∧
    correct_answers = 40 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l1387_138763


namespace NUMINAMATH_CALUDE_left_seats_count_l1387_138727

/-- Represents the seating configuration of a bus -/
structure BusSeats where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeat : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- The bus seating configuration satisfies the given conditions -/
def validBusConfig (bus : BusSeats) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.backSeat = 8 ∧
  bus.seatCapacity = 3 ∧
  bus.totalCapacity = 89 ∧
  bus.leftSeats * bus.seatCapacity + bus.rightSeats * bus.seatCapacity + bus.backSeat = bus.totalCapacity

/-- The number of seats on the left side of the bus is 15 -/
theorem left_seats_count (bus : BusSeats) (h : validBusConfig bus) : bus.leftSeats = 15 := by
  sorry

end NUMINAMATH_CALUDE_left_seats_count_l1387_138727


namespace NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l1387_138790

theorem four_digit_multiples_of_seven (n : ℕ) : 
  (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 7 = 0) ↔ 
  (n ∈ Finset.range 1286 ∧ ∃ k : ℕ, n = 7 * k + 1001) :=
sorry

end NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l1387_138790


namespace NUMINAMATH_CALUDE_initial_bushes_count_l1387_138711

/-- The number of orchid bushes planted today -/
def bushes_planted_today : ℕ := 37

/-- The number of orchid bushes planted tomorrow -/
def bushes_planted_tomorrow : ℕ := 25

/-- The total number of orchid bushes after planting -/
def total_bushes_after_planting : ℕ := 109

/-- The number of workers who finished the planting -/
def number_of_workers : ℕ := 35

/-- The initial number of orchid bushes in the park -/
def initial_bushes : ℕ := total_bushes_after_planting - (bushes_planted_today + bushes_planted_tomorrow)

theorem initial_bushes_count : initial_bushes = 47 := by
  sorry

end NUMINAMATH_CALUDE_initial_bushes_count_l1387_138711


namespace NUMINAMATH_CALUDE_inequality_implies_a_nonpositive_l1387_138723

theorem inequality_implies_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, x ∈ [1, 2] → 4^x - 2^(x+1) - a ≥ 0) →
  a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_nonpositive_l1387_138723


namespace NUMINAMATH_CALUDE_distance_equation_l1387_138736

/-- The distance between the boy's house and school -/
def D : ℝ := sorry

/-- The speed from house to library (km/hr) -/
def speed_to_library : ℝ := 3

/-- The speed from library to school (km/hr) -/
def speed_library_to_school : ℝ := 2.5

/-- The speed from school to house (km/hr) -/
def speed_return : ℝ := 2

/-- The time spent at the library (hours) -/
def library_time : ℝ := 0.5

/-- The total trip time (hours) -/
def total_time : ℝ := 5.5

theorem distance_equation : 
  (D / 2) / speed_to_library + library_time + 
  (D / 2) / speed_library_to_school + 
  D / speed_return = total_time := by sorry

end NUMINAMATH_CALUDE_distance_equation_l1387_138736


namespace NUMINAMATH_CALUDE_mira_sticker_arrangement_l1387_138748

/-- The number of stickers Mira currently has -/
def current_stickers : ℕ := 31

/-- The number of stickers required in each row -/
def stickers_per_row : ℕ := 7

/-- The function to calculate the number of additional stickers needed -/
def additional_stickers_needed (current : ℕ) (per_row : ℕ) : ℕ :=
  (per_row - (current % per_row)) % per_row

theorem mira_sticker_arrangement :
  additional_stickers_needed current_stickers stickers_per_row = 4 :=
by sorry

end NUMINAMATH_CALUDE_mira_sticker_arrangement_l1387_138748


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_simplify_resistance_formula_compare_time_taken_l1387_138762

-- 1. Simplify complex fraction
theorem simplify_complex_fraction (x y : ℝ) (h : y ≠ x) :
  (1 + x / y) / (1 - x / y) = (y + x) / (y - x) := by sorry

-- 2. Simplify resistance formula
theorem simplify_resistance_formula (R R₁ R₂ : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) :
  (1 / R = 1 / R₁ + 1 / R₂) → R = (R₁ * R₂) / (R₁ + R₂) := by sorry

-- 3. Compare time taken
theorem compare_time_taken (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) :
  x / (1 / (1 / y + 1 / z)) = (x * y + x * z) / (y * z) := by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_simplify_resistance_formula_compare_time_taken_l1387_138762
