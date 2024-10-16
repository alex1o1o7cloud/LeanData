import Mathlib

namespace NUMINAMATH_CALUDE_max_sales_and_profit_l16_1626

-- Define the sales volume function for the first 4 days
def sales_volume_early (x : ℝ) : ℝ := 20 * x + 80

-- Define the sales volume function for days 6 to 20
def sales_volume_late (x : ℝ) : ℝ := -x^2 + 50*x - 100

-- Define the selling price function for the first 5 days
def selling_price (x : ℝ) : ℝ := 2 * x + 28

-- Define the cost price
def cost_price : ℝ := 22

-- Define the profit function for days 1 to 5
def profit_early (x : ℝ) : ℝ := (selling_price x - cost_price) * sales_volume_early x

-- Define the profit function for days 6 to 20
def profit_late (x : ℝ) : ℝ := (28 - cost_price) * sales_volume_late x

theorem max_sales_and_profit :
  (∀ x ∈ Set.Icc 6 20, sales_volume_late x ≤ sales_volume_late 20) ∧
  sales_volume_late 20 = 500 ∧
  (∀ x ∈ Set.Icc 1 20, profit_early x ≤ profit_late 20 ∧ profit_late x ≤ profit_late 20) ∧
  profit_late 20 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_max_sales_and_profit_l16_1626


namespace NUMINAMATH_CALUDE_remaining_eggs_l16_1650

theorem remaining_eggs (initial_eggs : ℕ) (morning_eaten : ℕ) (afternoon_eaten : ℕ) :
  initial_eggs = 20 → morning_eaten = 4 → afternoon_eaten = 3 →
  initial_eggs - (morning_eaten + afternoon_eaten) = 13 := by
  sorry

end NUMINAMATH_CALUDE_remaining_eggs_l16_1650


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_l16_1654

/-- A parabola passing through a specific point has a specific latus rectum equation -/
theorem parabola_latus_rectum (p : ℝ) (h1 : p > 0) :
  (∀ x y, y^2 = 2*p*x → x = 1 ∧ y = 1/2) →
  (∃ x, x = -1/16 ∧ ∀ y, y^2 = 2*p*x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_l16_1654


namespace NUMINAMATH_CALUDE_total_wood_pieces_l16_1692

/-- The number of pieces of wood that can be contained in one sack -/
def sack_capacity : ℕ := 20

/-- The number of sacks filled with wood -/
def filled_sacks : ℕ := 4

/-- Theorem stating that the total number of wood pieces gathered is equal to
    the product of sack capacity and the number of filled sacks -/
theorem total_wood_pieces :
  sack_capacity * filled_sacks = 80 := by sorry

end NUMINAMATH_CALUDE_total_wood_pieces_l16_1692


namespace NUMINAMATH_CALUDE_tetrahedron_inference_is_logical_l16_1656

/-- Represents the concept of logical reasoning -/
def LogicalReasoning : Type := Unit

/-- Represents the concept of analogical reasoning -/
def AnalogicalReasoning : Type := Unit

/-- Represents the act of inferring properties of a spatial tetrahedron from a plane triangle -/
def InferTetrahedronFromTriangle : Type := Unit

/-- Analogical reasoning is a type of logical reasoning -/
axiom analogical_is_logical : AnalogicalReasoning → LogicalReasoning

/-- Inferring tetrahedron properties from triangle properties is analogical reasoning -/
axiom tetrahedron_inference_is_analogical : InferTetrahedronFromTriangle → AnalogicalReasoning

/-- Theorem: Inferring properties of a spatial tetrahedron from properties of a plane triangle
    is a kind of logical reasoning -/
theorem tetrahedron_inference_is_logical : InferTetrahedronFromTriangle → LogicalReasoning := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inference_is_logical_l16_1656


namespace NUMINAMATH_CALUDE_train_combined_speed_l16_1662

/-- The combined speed of two trains crossing a bridge simultaneously -/
theorem train_combined_speed
  (bridge_length : ℝ)
  (train1_length train1_time : ℝ)
  (train2_length train2_time : ℝ)
  (h1 : bridge_length = 300)
  (h2 : train1_length = 100)
  (h3 : train1_time = 30)
  (h4 : train2_length = 150)
  (h5 : train2_time = 45) :
  (train1_length + bridge_length) / train1_time +
  (train2_length + bridge_length) / train2_time =
  23.33 :=
sorry

end NUMINAMATH_CALUDE_train_combined_speed_l16_1662


namespace NUMINAMATH_CALUDE_no_divisible_by_three_for_all_x_l16_1668

theorem no_divisible_by_three_for_all_x : ¬∃ (p q : ℤ), ∀ (x : ℤ), 3 ∣ (x^2 + p*x + q) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_three_for_all_x_l16_1668


namespace NUMINAMATH_CALUDE_unique_value_at_two_l16_1697

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y^2

/-- The theorem stating that f(2) = 5 for any function satisfying the functional equation -/
theorem unique_value_at_two (f : ℝ → ℝ) (h : FunctionalEquation f) : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_value_at_two_l16_1697


namespace NUMINAMATH_CALUDE_restaurant_outdoor_area_l16_1691

/-- The area of a rectangular section with width 4 feet and length 6 feet is 24 square feet. -/
theorem restaurant_outdoor_area : 
  ∀ (width length area : ℝ), 
    width = 4 → 
    length = 6 → 
    area = width * length → 
    area = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_outdoor_area_l16_1691


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l16_1671

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → a ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l16_1671


namespace NUMINAMATH_CALUDE_product_with_decimals_l16_1636

theorem product_with_decimals (x y : ℚ) (z : ℕ) :
  x = 0.075 → y = 2.56 → z = 19200 →
  (↑75 : ℚ) * 256 = z →
  x * y = 0.192 := by
sorry

end NUMINAMATH_CALUDE_product_with_decimals_l16_1636


namespace NUMINAMATH_CALUDE_ravenswood_remaining_gnomes_l16_1657

/-- The number of gnomes in Westerville woods -/
def westerville_gnomes : ℕ := 20

/-- The ratio of gnomes in Ravenswood forest compared to Westerville woods -/
def ravenswood_ratio : ℕ := 4

/-- The percentage of gnomes taken from Ravenswood forest -/
def taken_percentage : ℚ := 40 / 100

/-- The number of gnomes remaining in Ravenswood forest after some are taken -/
def remaining_gnomes : ℕ := 48

theorem ravenswood_remaining_gnomes :
  (ravenswood_ratio * westerville_gnomes : ℚ) * (1 - taken_percentage) = remaining_gnomes := by
  sorry

end NUMINAMATH_CALUDE_ravenswood_remaining_gnomes_l16_1657


namespace NUMINAMATH_CALUDE_eighteen_games_equation_l16_1631

/-- The number of games in a competition where each pair of teams plays once. -/
def numGames (x : ℕ) : ℕ := x * (x - 1) / 2

/-- Theorem stating that for x teams, 18 total games is equivalent to the equation 1/2 * x * (x-1) = 18 -/
theorem eighteen_games_equation (x : ℕ) :
  numGames x = 18 ↔ (x * (x - 1)) / 2 = 18 := by sorry

end NUMINAMATH_CALUDE_eighteen_games_equation_l16_1631


namespace NUMINAMATH_CALUDE_science_fiction_section_pages_l16_1687

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_section_pages :
  total_pages = 3824 := by sorry

end NUMINAMATH_CALUDE_science_fiction_section_pages_l16_1687


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l16_1602

/-- A quadratic function f(x) = 3ax^2 + 2bx + c satisfying certain conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_zero : a + b + c = 0
  f_zero_pos : c > 0
  f_one_pos : 3*a + 2*b + c > 0

/-- The main theorem about the properties of the quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  f.a > 0 ∧ -2 < f.b / f.a ∧ f.b / f.a < -1 ∧
  (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 ∧
    3*f.a*x^2 + 2*f.b*x + f.c = 0 ∧
    3*f.a*y^2 + 2*f.b*y + f.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l16_1602


namespace NUMINAMATH_CALUDE_intersection_points_f_squared_f_sixth_l16_1637

theorem intersection_points_f_squared_f_sixth (f : ℝ → ℝ) (h_inj : Function.Injective f) :
  (∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x : ℝ, f (x^2) = f (x^6) ↔ x ∈ s)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_f_squared_f_sixth_l16_1637


namespace NUMINAMATH_CALUDE_negation_of_implication_l16_1620

theorem negation_of_implication :
  (¬(∀ x : ℝ, x ≥ 1 → x^2 - 4*x + 2 ≥ -1)) ↔ (∃ x : ℝ, x < 1 ∧ x^2 - 4*x + 2 < -1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l16_1620


namespace NUMINAMATH_CALUDE_pamelas_initial_skittles_l16_1628

/-- The number of Skittles Pamela gave away -/
def skittles_given : ℕ := 7

/-- The number of Skittles Pamela has now -/
def skittles_remaining : ℕ := 43

/-- Pamela's initial number of Skittles -/
def initial_skittles : ℕ := skittles_given + skittles_remaining

theorem pamelas_initial_skittles : initial_skittles = 50 := by
  sorry

end NUMINAMATH_CALUDE_pamelas_initial_skittles_l16_1628


namespace NUMINAMATH_CALUDE_equation_describes_cylinder_l16_1643

-- Define cylindrical coordinates
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define a cylinder
def IsCylinder (S : Set CylindricalCoord) (c : ℝ) : Prop :=
  c > 0 ∧ ∀ p : CylindricalCoord, p ∈ S ↔ p.r = c

-- Theorem statement
theorem equation_describes_cylinder (c : ℝ) :
  IsCylinder {p : CylindricalCoord | p.r = c} c :=
by
  sorry

end NUMINAMATH_CALUDE_equation_describes_cylinder_l16_1643


namespace NUMINAMATH_CALUDE_nine_seats_six_people_arrangement_l16_1615

/-- The number of ways to arrange people and empty seats in a row -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  (Nat.factorial people) * (Nat.choose (people - 1) (total_seats - people))

/-- Theorem: There are 7200 ways to arrange 6 people and 3 empty seats in a row of 9 seats,
    where every empty seat is flanked by people on both sides -/
theorem nine_seats_six_people_arrangement :
  seating_arrangements 9 6 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_nine_seats_six_people_arrangement_l16_1615


namespace NUMINAMATH_CALUDE_simple_interest_problem_l16_1632

/-- Given a principal P and an interest rate R, if increasing the rate by 6% 
    results in $90 more interest over 5 years, then P = $300. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * R * 5 / 100 + 90 = P * (R + 6) * 5 / 100) → P = 300 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l16_1632


namespace NUMINAMATH_CALUDE_negation_square_positive_l16_1625

theorem negation_square_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_square_positive_l16_1625


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_line_slope_intercept_sum_proof_l16_1623

/-- Given two points A(1, 4) and B(5, 16) on a line, 
    the sum of the line's slope and y-intercept is 4. -/
theorem line_slope_intercept_sum : ℝ → ℝ → Prop :=
  fun (slope : ℝ) (y_intercept : ℝ) =>
    (slope * 1 + y_intercept = 4) ∧  -- Point A satisfies the line equation
    (slope * 5 + y_intercept = 16) ∧ -- Point B satisfies the line equation
    (slope + y_intercept = 4)        -- Sum of slope and y-intercept is 4

/-- Proof of the theorem -/
theorem line_slope_intercept_sum_proof : ∃ (slope : ℝ) (y_intercept : ℝ), 
  line_slope_intercept_sum slope y_intercept := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_line_slope_intercept_sum_proof_l16_1623


namespace NUMINAMATH_CALUDE_average_licks_to_center_l16_1649

def dan_licks : ℕ := 58
def michael_licks : ℕ := 63
def sam_licks : ℕ := 70
def david_licks : ℕ := 70
def lance_licks : ℕ := 39

def total_licks : ℕ := dan_licks + michael_licks + sam_licks + david_licks + lance_licks
def num_people : ℕ := 5

theorem average_licks_to_center (h : total_licks = dan_licks + michael_licks + sam_licks + david_licks + lance_licks) :
  (total_licks : ℚ) / num_people = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_licks_to_center_l16_1649


namespace NUMINAMATH_CALUDE_divisibility_implication_l16_1673

theorem divisibility_implication (n : ℕ+) :
  (∃ k : ℤ, n.val^2 + 3*n.val + 51 = 13*k) →
  (∃ m : ℤ, 21*n.val^2 + 89*n.val + 44 = 169*m) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implication_l16_1673


namespace NUMINAMATH_CALUDE_intersection_points_product_l16_1605

-- Define the curve C in Cartesian coordinates
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l
def line_l (x y m : ℝ) : Prop := x - Real.sqrt 3 * y - m = 0

-- Define the intersection condition
def intersects (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧
    (x₁ - m)^2 + y₁^2 * (x₂ - m)^2 + y₂^2 = 1

-- Theorem statement
theorem intersection_points_product (m : ℝ) :
  intersects m ↔ m = 1 ∨ m = 1 + Real.sqrt 2 ∨ m = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_product_l16_1605


namespace NUMINAMATH_CALUDE_poverty_decline_rate_l16_1630

/-- The annual average decline rate of impoverished people -/
def annual_decline_rate : ℝ := 0.5

/-- The initial number of impoverished people in 2018 -/
def initial_population : ℕ := 40000

/-- The number of impoverished people in 2020 -/
def final_population : ℕ := 10000

/-- The time period in years -/
def time_period : ℕ := 2

theorem poverty_decline_rate :
  (↑initial_population * (1 - annual_decline_rate) ^ time_period = ↑final_population) ∧
  (0 < annual_decline_rate) ∧
  (annual_decline_rate < 1) := by
  sorry

end NUMINAMATH_CALUDE_poverty_decline_rate_l16_1630


namespace NUMINAMATH_CALUDE_maddie_makeup_palettes_l16_1608

/-- The number of makeup palettes Maddie bought -/
def num_palettes : ℕ := 3

/-- The cost of each makeup palette in dollars -/
def palette_cost : ℚ := 15

/-- The total cost of lipsticks in dollars -/
def lipstick_cost : ℚ := 10

/-- The total cost of hair color boxes in dollars -/
def hair_color_cost : ℚ := 12

/-- The total amount Maddie paid in dollars -/
def total_paid : ℚ := 67

/-- Theorem stating that the number of makeup palettes Maddie bought is correct -/
theorem maddie_makeup_palettes : 
  (num_palettes : ℚ) * palette_cost + lipstick_cost + hair_color_cost = total_paid := by
  sorry

#check maddie_makeup_palettes

end NUMINAMATH_CALUDE_maddie_makeup_palettes_l16_1608


namespace NUMINAMATH_CALUDE_prob_two_threes_correct_l16_1679

/-- The probability of rolling exactly two 3s when rolling eight standard 6-sided dice -/
def prob_two_threes : ℚ :=
  (28 : ℚ) * 15625 / 559872

/-- The probability calculated using binomial distribution -/
def prob_two_threes_calc : ℚ :=
  (Nat.choose 8 2 : ℚ) * (1/6)^2 * (5/6)^6

theorem prob_two_threes_correct : prob_two_threes = prob_two_threes_calc := by
  sorry

end NUMINAMATH_CALUDE_prob_two_threes_correct_l16_1679


namespace NUMINAMATH_CALUDE_extremum_condition_l16_1699

/-- The function f(x) = ax^3 + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The function f has an extremum -/
def has_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y ∨ f a x ≥ f a y

/-- The necessary and sufficient condition for f to have an extremum is a < 0 -/
theorem extremum_condition (a : ℝ) :
  has_extremum a ↔ a < 0 :=
sorry

end NUMINAMATH_CALUDE_extremum_condition_l16_1699


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l16_1689

/-- Given a line L1 with equation 3x - 4y + 6 = 0 and a point P(4, -1),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 4x + 3y - 13 = 0 -/
theorem perpendicular_line_equation :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 4 * y + 6 = 0
  let P : ℝ × ℝ := (4, -1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 3 * y - 13 = 0
  (∀ x y, L2 x y ↔ (y - P.2 = -(3/4) * (x - P.1))) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((y₂ - y₁) / (x₂ - x₁)) * ((x₂ - x₁) / (y₂ - y₁)) = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l16_1689


namespace NUMINAMATH_CALUDE_smallest_n_is_correct_smallest_n_satisfies_property_l16_1642

/-- The smallest positive integer n with the given divisibility property -/
def smallest_n : ℕ := 13

/-- Proposition stating that smallest_n is the correct answer -/
theorem smallest_n_is_correct :
  ∀ (n : ℕ), n > 0 → 
  (∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
    x ∣ y^3 → y ∣ z^3 → z ∣ x^3 →
    x * y * z ∣ (x + y + z)^n) →
  n ≥ smallest_n :=
by sorry

/-- Proposition stating that smallest_n satisfies the required property -/
theorem smallest_n_satisfies_property :
  ∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
    x ∣ y^3 → y ∣ z^3 → z ∣ x^3 →
    x * y * z ∣ (x + y + z)^smallest_n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_is_correct_smallest_n_satisfies_property_l16_1642


namespace NUMINAMATH_CALUDE_unique_solution_iff_k_eq_six_l16_1619

/-- The equation (x+5)(x+2) = k + 3x has exactly one real solution if and only if k = 6 -/
theorem unique_solution_iff_k_eq_six (k : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_k_eq_six_l16_1619


namespace NUMINAMATH_CALUDE_shelter_cats_l16_1686

theorem shelter_cats (total_animals : ℕ) (dogs : ℕ) (cats : ℕ) : 
  total_animals = 1212 → dogs = 567 → total_animals = cats + dogs → cats = 645 := by
  sorry

end NUMINAMATH_CALUDE_shelter_cats_l16_1686


namespace NUMINAMATH_CALUDE_rectangular_field_area_l16_1613

/-- Proves that a rectangular field with width one-third of length and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_field_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  2 * (width + length) = 72 →
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l16_1613


namespace NUMINAMATH_CALUDE_alloy_fourth_metal_mass_l16_1658

/-- Given an alloy of four metals with a total mass of 20 kg, where:
    - The mass of the first metal is 1.5 times the mass of the second metal
    - The ratio of the mass of the second metal to the third metal is 3:4
    - The ratio of the mass of the third metal to the fourth metal is 5:6
    Prove that the mass of the fourth metal is 960/163 kg. -/
theorem alloy_fourth_metal_mass (m₁ m₂ m₃ m₄ : ℝ) 
  (h_total : m₁ + m₂ + m₃ + m₄ = 20)
  (h_first_second : m₁ = 1.5 * m₂)
  (h_second_third : m₂ / m₃ = 3 / 4)
  (h_third_fourth : m₃ / m₄ = 5 / 6) :
  m₄ = 960 / 163 := by
sorry

end NUMINAMATH_CALUDE_alloy_fourth_metal_mass_l16_1658


namespace NUMINAMATH_CALUDE_expression_evaluation_l16_1682

theorem expression_evaluation :
  let f (x : ℚ) := (3 * x + 2) / (2 * x - 1)
  let g (x : ℚ) := (3 * f x + 2) / (2 * f x - 1)
  g (1/3) = 113/31 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l16_1682


namespace NUMINAMATH_CALUDE_laura_shirt_count_l16_1607

def pants_count : ℕ := 2
def pants_price : ℕ := 54
def shirt_price : ℕ := 33
def money_given : ℕ := 250
def change_received : ℕ := 10

theorem laura_shirt_count :
  (money_given - change_received - pants_count * pants_price) / shirt_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_laura_shirt_count_l16_1607


namespace NUMINAMATH_CALUDE_min_value_on_circle_l16_1670

theorem min_value_on_circle (x y : ℝ) (h : (x - 3)^2 + y^2 = 9) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a - 3)^2 + b^2 = 9 → -2*b - 3*a ≥ m) ∧
             (∃ (c d : ℝ), (c - 3)^2 + d^2 = 9 ∧ -2*d - 3*c = m) ∧
             m = -3 * Real.sqrt 13 - 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l16_1670


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l16_1601

/-- Given a bus that stops for 10 minutes per hour and travels at 45 kmph including stoppages,
    prove that its speed excluding stoppages is 54 kmph. -/
theorem bus_speed_excluding_stoppages :
  let stop_time : ℚ := 10 / 60  -- 10 minutes per hour
  let speed_with_stops : ℚ := 45  -- 45 kmph including stoppages
  let actual_travel_time : ℚ := 1 - stop_time  -- fraction of hour bus is moving
  speed_with_stops / actual_travel_time = 54 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l16_1601


namespace NUMINAMATH_CALUDE_room_occupancy_l16_1614

theorem room_occupancy (people stools chairs : ℕ) : 
  people > stools ∧ 
  people > chairs ∧ 
  people < stools + chairs ∧ 
  2 * people + 3 * stools + 4 * chairs = 32 →
  people = 5 ∧ stools = 2 ∧ chairs = 4 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l16_1614


namespace NUMINAMATH_CALUDE_incorrect_transformations_count_l16_1621

-- Define the structure of an equation transformation
structure EquationTransformation where
  initial : String
  final : String

-- Define a function to check if a transformation is correct
def isCorrectTransformation (t : EquationTransformation) : Bool :=
  match t with
  | ⟨"(3-2x)/3 - (x-2)/2 = 1", "2(3-2x) - 3(x-2) = 6"⟩ => false
  | ⟨"3x + 8 = -4x - 7", "3x + 4x = 7 - 8"⟩ => false
  | ⟨"7(3-x) - 5(x-3) = 8", "21 - 7x - 5x + 15 = 8"⟩ => true
  | ⟨"3/7 * x = 7/3", "x = 49/9"⟩ => false
  | _ => false

-- Define the set of given transformations
def givenTransformations : List EquationTransformation := [
  ⟨"(3-2x)/3 - (x-2)/2 = 1", "2(3-2x) - 3(x-2) = 6"⟩,
  ⟨"3x + 8 = -4x - 7", "3x + 4x = 7 - 8"⟩,
  ⟨"7(3-x) - 5(x-3) = 8", "21 - 7x - 5x + 15 = 8"⟩,
  ⟨"3/7 * x = 7/3", "x = 49/9"⟩
]

-- Theorem statement
theorem incorrect_transformations_count :
  (givenTransformations.filter (fun t => ¬(isCorrectTransformation t))).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_transformations_count_l16_1621


namespace NUMINAMATH_CALUDE_twelfth_team_games_l16_1600

/-- Represents a football tournament -/
structure Tournament where
  teams : Fin 12 → ℕ
  first_team_games : teams 0 = 11
  three_teams_nine_games : ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ teams i = 9 ∧ teams j = 9 ∧ teams k = 9
  one_team_five_games : ∃ i, teams i = 5
  four_teams_four_games : ∃ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
                          teams i = 4 ∧ teams j = 4 ∧ teams k = 4 ∧ teams l = 4
  two_teams_one_game : ∃ i j, i ≠ j ∧ teams i = 1 ∧ teams j = 1
  no_repeat_games : ∀ i j, i ≠ j → teams i + teams j ≤ 12

theorem twelfth_team_games (t : Tournament) : 
  ∃ i, t.teams i = 5 ∧ ∀ j, j ≠ i → t.teams j ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_twelfth_team_games_l16_1600


namespace NUMINAMATH_CALUDE_sprint_medal_awards_l16_1640

/-- The number of ways to award medals in the international sprinting event -/
def medal_award_ways (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) 
  (americans_winning : ℕ) : ℕ :=
  -- The actual calculation would go here
  216

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem sprint_medal_awards : 
  medal_award_ways 10 4 3 2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sprint_medal_awards_l16_1640


namespace NUMINAMATH_CALUDE_initial_average_marks_l16_1616

theorem initial_average_marks (n : ℕ) (wrong_mark correct_mark : ℝ) (correct_avg : ℝ) :
  n = 10 →
  wrong_mark = 50 →
  correct_mark = 10 →
  correct_avg = 96 →
  (n * correct_avg * n - (wrong_mark - correct_mark)) / n = 92 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_average_marks_l16_1616


namespace NUMINAMATH_CALUDE_complex_calculation_l16_1638

theorem complex_calculation (z : ℂ) (h : z = 1 + I) : z^2 + 2/z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l16_1638


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l16_1624

theorem arithmetic_calculations :
  (7 + (-14) - (-9) - 12 = -10) ∧
  (25 / (-5) * (1 / 5) / (3 / 4) = -4 / 3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l16_1624


namespace NUMINAMATH_CALUDE_trailing_zeros_of_square_l16_1659

/-- The number of trailing zeros in (10^10 - 2)^2 is 17 -/
theorem trailing_zeros_of_square : ∃ n : ℕ, (10^10 - 2)^2 = n * 10^17 ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_square_l16_1659


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l16_1690

def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l16_1690


namespace NUMINAMATH_CALUDE_x_equals_y_l16_1663

theorem x_equals_y (x y : ℝ) : x = 2 + Real.sqrt 3 → y = 1 / (2 - Real.sqrt 3) → x = y := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_l16_1663


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_26_l16_1627

theorem modular_inverse_of_5_mod_26 :
  ∃! x : ℕ, x ∈ Finset.range 26 ∧ (5 * x) % 26 = 1 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_26_l16_1627


namespace NUMINAMATH_CALUDE_wings_temperature_l16_1646

/-- Given an initial oven temperature and a required temperature increase,
    calculate the final required temperature. -/
def required_temperature (initial_temp increase : ℕ) : ℕ :=
  initial_temp + increase

/-- Theorem: The required temperature for the wings is 546 degrees,
    given an initial temperature of 150 degrees and a needed increase of 396 degrees. -/
theorem wings_temperature : required_temperature 150 396 = 546 := by
  sorry

end NUMINAMATH_CALUDE_wings_temperature_l16_1646


namespace NUMINAMATH_CALUDE_incenter_distance_l16_1674

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB = 15 ∧ AC = 17 ∧ BC = 16

-- Define the incenter
def Incenter (I : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (Real.sqrt ((I.1 - A.1)^2 + (I.2 - A.2)^2) = r) ∧
  (Real.sqrt ((I.1 - B.1)^2 + (I.2 - B.2)^2) = r) ∧
  (Real.sqrt ((I.1 - C.1)^2 + (I.2 - C.2)^2) = r)

theorem incenter_distance (A B C I : ℝ × ℝ) :
  Triangle A B C → Incenter I A B C →
  Real.sqrt ((I.1 - B.1)^2 + (I.2 - B.2)^2) = Real.sqrt 85 :=
by sorry

end NUMINAMATH_CALUDE_incenter_distance_l16_1674


namespace NUMINAMATH_CALUDE_square_sum_value_l16_1639

theorem square_sum_value (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : 
  x^2 + 9 * y^2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l16_1639


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l16_1676

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  7 * p^6 * q = 21 * p^5 * q^2 → 
  p = 3/4 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l16_1676


namespace NUMINAMATH_CALUDE_product_441_sum_32_l16_1665

theorem product_441_sum_32 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 441 →
  a + b + c + d = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_441_sum_32_l16_1665


namespace NUMINAMATH_CALUDE_inverse_square_relation_l16_1644

/-- Given that x varies inversely as the square of y and y = 3 when x = 1,
    prove that x = 1/9 when y = 9. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) : 
  (∀ (x y : ℝ), x = k / (y^2)) →  -- x varies inversely as square of y
  (1 = k / (3^2)) →               -- y = 3 when x = 1
  (x = k / (9^2)) →               -- condition for y = 9
  x = 1/9 := by
sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l16_1644


namespace NUMINAMATH_CALUDE_age_problem_l16_1664

theorem age_problem (A B C : ℤ) : 
  A + B = 2 * C →
  B = A / 2 + 4 →
  A - C = B - C + 16 →
  A = 40 ∧ B = 24 ∧ C = 32 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l16_1664


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l16_1680

/-- Given points A, B, C on the inverse proportion function y = -2/x, 
    prove the relationship between their y-coordinates. -/
theorem inverse_proportion_y_relationship (y₁ y₂ y₃ : ℝ) : 
  y₁ = -2 / (-2) → y₂ = -2 / 2 → y₃ = -2 / 3 → y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

#check inverse_proportion_y_relationship

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l16_1680


namespace NUMINAMATH_CALUDE_sequence_sum_property_l16_1652

theorem sequence_sum_property (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) :
  (∀ n : ℕ+, S n = 1 - n * a n) →
  (∀ n : ℕ+, a n = 1 / (n * (n + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l16_1652


namespace NUMINAMATH_CALUDE_fraction_negative_exponent_l16_1611

theorem fraction_negative_exponent :
  (2 / 3 : ℚ) ^ (-2 : ℤ) = 9 / 4 := by sorry

end NUMINAMATH_CALUDE_fraction_negative_exponent_l16_1611


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l16_1677

theorem pure_imaginary_complex_number (x : ℝ) : 
  (Complex.I * (x + 3) = (x^2 + 2*x - 3) + Complex.I * (x + 3)) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l16_1677


namespace NUMINAMATH_CALUDE_percent_of_x_l16_1694

theorem percent_of_x (x y z : ℝ) 
  (h1 : 0.6 * (x - y) = 0.3 * (x + y + z)) 
  (h2 : 0.4 * (y - z) = 0.2 * (y + x - z)) : 
  y - z = x := by sorry

end NUMINAMATH_CALUDE_percent_of_x_l16_1694


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l16_1684

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 14 ∧ (9679 - k) % 15 = 0 ∧ ∀ (m : ℕ), m < k → (9679 - m) % 15 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l16_1684


namespace NUMINAMATH_CALUDE_rotation_equivalence_l16_1651

/-- Given two rotations about the same point Q:
    1. A 735-degree clockwise rotation of point P to point R
    2. A y-degree counterclockwise rotation of point P to the same point R
    where y < 360, prove that y = 345 degrees. -/
theorem rotation_equivalence (y : ℝ) (h1 : y < 360) : 
  (735 % 360 : ℝ) + y = 360 → y = 345 := by sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l16_1651


namespace NUMINAMATH_CALUDE_line_equation_45_degree_slope_2_intercept_l16_1666

/-- The equation of a line with a slope angle of 45° and a y-intercept of 2 is y = x + 2 -/
theorem line_equation_45_degree_slope_2_intercept :
  let slope_angle : Real := 45 * (π / 180)  -- Convert 45° to radians
  let y_intercept : Real := 2
  let slope : Real := Real.tan slope_angle
  ∀ x y : Real, y = slope * x + y_intercept ↔ y = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_45_degree_slope_2_intercept_l16_1666


namespace NUMINAMATH_CALUDE_min_sum_dimensions_for_volume_2184_l16_1678

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the volume of a rectangular box -/
def volume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Calculates the sum of dimensions of a rectangular box -/
def sumDimensions (d : BoxDimensions) : ℕ := d.length + d.width + d.height

/-- Theorem: The minimum sum of dimensions for a box with volume 2184 is 36 -/
theorem min_sum_dimensions_for_volume_2184 :
  (∃ (d : BoxDimensions), volume d = 2184) →
  (∀ (d : BoxDimensions), volume d = 2184 → sumDimensions d ≥ 36) ∧
  (∃ (d : BoxDimensions), volume d = 2184 ∧ sumDimensions d = 36) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_for_volume_2184_l16_1678


namespace NUMINAMATH_CALUDE_negative_integer_solution_to_inequality_l16_1698

theorem negative_integer_solution_to_inequality :
  ∀ x : ℤ, (x < 0 ∧ -2 * x < 4) ↔ x = -1 :=
sorry

end NUMINAMATH_CALUDE_negative_integer_solution_to_inequality_l16_1698


namespace NUMINAMATH_CALUDE_stating_regular_ngon_diagonal_difference_l16_1696

/-- 
Given a regular n-gon with n > 5, this function calculates the length of its longest diagonal.
-/
noncomputable def longest_diagonal (n : ℕ) (side_length : ℝ) : ℝ := sorry

/-- 
Given a regular n-gon with n > 5, this function calculates the length of its shortest diagonal.
-/
noncomputable def shortest_diagonal (n : ℕ) (side_length : ℝ) : ℝ := sorry

/-- 
Theorem stating that for a regular n-gon with n > 5, the difference between 
the longest diagonal and the shortest diagonal is equal to the side length 
if and only if n = 9.
-/
theorem regular_ngon_diagonal_difference (n : ℕ) (side_length : ℝ) : 
  n > 5 → 
  (longest_diagonal n side_length - shortest_diagonal n side_length = side_length ↔ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_stating_regular_ngon_diagonal_difference_l16_1696


namespace NUMINAMATH_CALUDE_trig_identity_l16_1655

theorem trig_identity (α β : ℝ) :
  (Real.sin (2 * α + β) / Real.sin α) - 2 * Real.cos (α + β) = Real.sin β / Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l16_1655


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l16_1675

def selling_price : ℝ := 900
def profit : ℝ := 100

theorem profit_percentage_calculation :
  (profit / (selling_price - profit)) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l16_1675


namespace NUMINAMATH_CALUDE_fraction_product_l16_1660

theorem fraction_product : (2 : ℚ) / 9 * 5 / 11 = 10 / 99 := by sorry

end NUMINAMATH_CALUDE_fraction_product_l16_1660


namespace NUMINAMATH_CALUDE_max_value_of_expression_l16_1635

theorem max_value_of_expression (x y : ℝ) 
  (hx : -4 ≤ x ∧ x ≤ -2) (hy : 2 ≤ y ∧ y ≤ 4) :
  (∀ z w : ℝ, -4 ≤ z ∧ z ≤ -2 ∧ 2 ≤ w ∧ w ≤ 4 → (z + w) / z ≤ (x + y) / x) →
  (x + y) / x = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l16_1635


namespace NUMINAMATH_CALUDE_product_expansion_sum_l16_1661

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 14 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l16_1661


namespace NUMINAMATH_CALUDE_soccer_games_played_l16_1634

theorem soccer_games_played (total_players : ℕ) (total_goals : ℕ) (goals_by_others : ℕ) :
  total_players = 24 →
  total_goals = 150 →
  goals_by_others = 30 →
  ∃ (games_played : ℕ),
    games_played = 15 ∧
    games_played * (total_players / 3) + goals_by_others = total_goals :=
by sorry

end NUMINAMATH_CALUDE_soccer_games_played_l16_1634


namespace NUMINAMATH_CALUDE_tonya_lego_sets_l16_1672

/-- Represents the cost of a single doll in dollars -/
def doll_cost : ℕ := 15

/-- Represents the number of dolls bought for the younger sister -/
def num_dolls : ℕ := 4

/-- Represents the cost of a single lego set in dollars -/
def lego_cost : ℕ := 20

/-- Calculates the total spent on the younger sister -/
def younger_sister_total : ℕ := doll_cost * num_dolls

/-- Represents the number of lego sets bought for the older sister -/
def num_lego_sets : ℕ := younger_sister_total / lego_cost

theorem tonya_lego_sets : num_lego_sets = 3 := by
  sorry

end NUMINAMATH_CALUDE_tonya_lego_sets_l16_1672


namespace NUMINAMATH_CALUDE_trapezoid_area_three_squares_l16_1641

/-- The area of the trapezoid formed by three squares with sides 3, 5, and 7 units -/
theorem trapezoid_area_three_squares :
  let square1 : ℝ := 3
  let square2 : ℝ := 5
  let square3 : ℝ := 7
  let total_base : ℝ := square1 + square2 + square3
  let height_ratio : ℝ := square3 / total_base
  let trapezoid_height : ℝ := square2
  let trapezoid_base1 : ℝ := square1 * height_ratio
  let trapezoid_base2 : ℝ := (square1 + square2) * height_ratio
  let trapezoid_area : ℝ := (trapezoid_base1 + trapezoid_base2) * trapezoid_height / 2
  trapezoid_area = 12.825 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_three_squares_l16_1641


namespace NUMINAMATH_CALUDE_tangent_condition_intersection_condition_l16_1669

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Tangent condition
theorem tangent_condition (m : ℝ) : 
  (∃! p : ℝ × ℝ, ellipse p.1 p.2 ∧ line p.1 p.2 m) ↔ m^2 = 5 :=
sorry

-- Intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ p q : ℝ × ℝ, p ≠ q ∧ 
   ellipse p.1 p.2 ∧ ellipse q.1 q.2 ∧ 
   line p.1 p.2 m ∧ line q.1 q.2 m ∧
   (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4) ↔ 16 * m^2 = 30 :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_intersection_condition_l16_1669


namespace NUMINAMATH_CALUDE_carolyn_practice_time_l16_1695

/-- Calculates the total practice time for Carolyn in a month -/
def total_practice_time (piano_time : ℕ) (violin_multiplier : ℕ) (practice_days : ℕ) (weeks : ℕ) : ℕ :=
  let daily_total := piano_time + violin_multiplier * piano_time
  let weekly_total := daily_total * practice_days
  weekly_total * weeks

/-- Proves that Carolyn's total practice time in a month is 1920 minutes -/
theorem carolyn_practice_time :
  total_practice_time 20 3 6 4 = 1920 :=
by sorry

end NUMINAMATH_CALUDE_carolyn_practice_time_l16_1695


namespace NUMINAMATH_CALUDE_union_of_sets_l16_1647

theorem union_of_sets : 
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l16_1647


namespace NUMINAMATH_CALUDE_tower_arrangements_eq_4200_l16_1629

/-- The number of ways to arrange 9 cubes out of 10 cubes (3 red, 3 blue, 4 green) -/
def tower_arrangements : ℕ := 
  Nat.choose 10 9 * (Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))

/-- Theorem stating that the number of tower arrangements is 4200 -/
theorem tower_arrangements_eq_4200 : tower_arrangements = 4200 := by
  sorry

end NUMINAMATH_CALUDE_tower_arrangements_eq_4200_l16_1629


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l16_1612

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 20) 
  (h_a6 : a 6 = 11) : 
  a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l16_1612


namespace NUMINAMATH_CALUDE_circular_track_catchup_l16_1683

/-- The time (in minutes) for Person A to catch up with Person B on a circular track -/
def catchUpTime (trackCircumference : ℝ) (speedA speedB : ℝ) (restInterval : ℝ) (restDuration : ℝ) : ℝ :=
  sorry

theorem circular_track_catchup :
  let trackCircumference : ℝ := 400
  let speedA : ℝ := 52
  let speedB : ℝ := 46
  let restInterval : ℝ := 100
  let restDuration : ℝ := 1
  catchUpTime trackCircumference speedA speedB restInterval restDuration = 147 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_catchup_l16_1683


namespace NUMINAMATH_CALUDE_class_transfer_equation_l16_1609

theorem class_transfer_equation (x : ℕ) : 
  (∀ (total : ℕ), total = 98 → 
    (∀ (transfer : ℕ), transfer = 3 →
      (total - x) + transfer = x - transfer)) ↔ 
  (98 - x) + 3 = x - 3 :=
sorry

end NUMINAMATH_CALUDE_class_transfer_equation_l16_1609


namespace NUMINAMATH_CALUDE_negation_of_for_all_positive_negation_of_specific_quadratic_l16_1688

theorem negation_of_for_all_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) := by sorry

theorem negation_of_specific_quadratic :
  (¬ ∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 3 * x + 4 ≤ 0) := by
  apply negation_of_for_all_positive (fun x ↦ 2 * x^2 - 3 * x + 4)

end NUMINAMATH_CALUDE_negation_of_for_all_positive_negation_of_specific_quadratic_l16_1688


namespace NUMINAMATH_CALUDE_prop_one_prop_two_l16_1667

-- Proposition 1
theorem prop_one (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a > b) :
  a - 1 / a > b - 1 / b :=
sorry

-- Proposition 2
theorem prop_two (a b : ℝ) (hb : b ≠ 0) :
  b * (b - a) ≤ 0 ↔ a / b ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_prop_one_prop_two_l16_1667


namespace NUMINAMATH_CALUDE_all_positive_iff_alpha_geq_three_l16_1622

-- Define the sequence a_n
def a : ℕ → ℝ → ℝ
  | 0, α => α
  | n + 1, α => 2 * a n α - n^2

-- Theorem statement
theorem all_positive_iff_alpha_geq_three (α : ℝ) :
  (∀ n : ℕ, a n α > 0) ↔ α ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_all_positive_iff_alpha_geq_three_l16_1622


namespace NUMINAMATH_CALUDE_coefficient_corresponds_to_20th_term_l16_1604

/-- The general term of the arithmetic sequence -/
def a (n : ℕ) : ℤ := 3 * n - 5

/-- The coefficient of x^4 in the expansion of (1+x)^k -/
def coeff (k : ℕ) : ℕ := Nat.choose k 4

/-- The theorem stating that the 20th term of the sequence corresponds to
    the coefficient of x^4 in the given expansion -/
theorem coefficient_corresponds_to_20th_term :
  a 20 = (coeff 5 + coeff 6 + coeff 7) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_corresponds_to_20th_term_l16_1604


namespace NUMINAMATH_CALUDE_square_area_measurement_error_l16_1681

theorem square_area_measurement_error :
  let actual_length : ℝ := L
  let measured_side1 : ℝ := L * (1 + 0.02)
  let measured_side2 : ℝ := L * (1 - 0.03)
  let calculated_area : ℝ := measured_side1 * measured_side2
  let actual_area : ℝ := L * L
  let error : ℝ := actual_area - calculated_area
  let percentage_error : ℝ := (error / actual_area) * 100
  percentage_error = 1.06 := by
sorry

end NUMINAMATH_CALUDE_square_area_measurement_error_l16_1681


namespace NUMINAMATH_CALUDE_milk_pumping_time_l16_1603

theorem milk_pumping_time (initial_milk : ℝ) (pump_rate : ℝ) (add_rate : ℝ) (add_time : ℝ) (final_milk : ℝ) :
  initial_milk = 30000 ∧
  pump_rate = 2880 ∧
  add_rate = 1500 ∧
  add_time = 7 ∧
  final_milk = 28980 →
  ∃ (h : ℝ), h = 4 ∧ initial_milk - pump_rate * h + add_rate * add_time = final_milk :=
by sorry

end NUMINAMATH_CALUDE_milk_pumping_time_l16_1603


namespace NUMINAMATH_CALUDE_equivalent_discount_l16_1648

theorem equivalent_discount (original_price : ℝ) (first_discount second_discount : ℝ) :
  first_discount = 0.2 →
  second_discount = 0.25 →
  original_price * (1 - first_discount) * (1 - second_discount) = original_price * (1 - 0.4) :=
by
  sorry

end NUMINAMATH_CALUDE_equivalent_discount_l16_1648


namespace NUMINAMATH_CALUDE_min_balls_to_draw_l16_1610

theorem min_balls_to_draw (black white red : ℕ) (h1 : black = 10) (h2 : white = 9) (h3 : red = 8) :
  black + white + 1 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_l16_1610


namespace NUMINAMATH_CALUDE_song_storage_size_l16_1693

-- Define the given values
def total_storage : ℕ := 16  -- in GB
def used_storage : ℕ := 4    -- in GB
def num_songs : ℕ := 400
def mb_per_gb : ℕ := 1000

-- Define the theorem
theorem song_storage_size :
  let available_storage : ℕ := total_storage - used_storage
  let available_storage_mb : ℕ := available_storage * mb_per_gb
  available_storage_mb / num_songs = 30 := by sorry

end NUMINAMATH_CALUDE_song_storage_size_l16_1693


namespace NUMINAMATH_CALUDE_max_value_theorem_l16_1633

theorem max_value_theorem (a b c d e : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) (pos_e : e > 0)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 504) :
  ac + 3*b*c + 4*c*d + 6*c*e ≤ 252 * Real.sqrt 62 ∧
  (a = 2 ∧ b = 6 ∧ c = 6 * Real.sqrt 7 ∧ d = 8 ∧ e = 12) →
  ac + 3*b*c + 4*c*d + 6*c*e = 252 * Real.sqrt 62 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l16_1633


namespace NUMINAMATH_CALUDE_product_mod_seven_l16_1606

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l16_1606


namespace NUMINAMATH_CALUDE_pattern_1005th_row_l16_1685

/-- Represents the number of items in the nth row of the pattern -/
def num_items (n : ℕ) : ℕ := n

/-- Represents the sum of items in the nth row of the pattern -/
def sum_items (n : ℕ) : ℕ := n * (n + 1) / 2 + (n - 1) * n / 2

/-- Theorem stating that the 1005th row is the one where the number of items
    and their sum equals 20092 -/
theorem pattern_1005th_row :
  num_items 1005 + sum_items 1005 = 20092 := by sorry

end NUMINAMATH_CALUDE_pattern_1005th_row_l16_1685


namespace NUMINAMATH_CALUDE_triangle_isosceles_l16_1645

theorem triangle_isosceles (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : Real.sin C = 2 * Real.cos A * Real.sin B) : A = B := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l16_1645


namespace NUMINAMATH_CALUDE_copy_pages_theorem_l16_1618

/-- Given a cost per page in cents and a budget in dollars, 
    calculate the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Theorem: With a cost of 3 cents per page and a budget of $15, 
    the maximum number of pages that can be copied is 500. -/
theorem copy_pages_theorem : max_pages_copied 3 15 = 500 := by
  sorry

end NUMINAMATH_CALUDE_copy_pages_theorem_l16_1618


namespace NUMINAMATH_CALUDE_doug_fires_count_l16_1653

theorem doug_fires_count (doug kai eli total : ℕ) 
  (h1 : kai = 3 * doug)
  (h2 : eli = kai / 2)
  (h3 : doug + kai + eli = total)
  (h4 : total = 110) : doug = 20 := by
  sorry

end NUMINAMATH_CALUDE_doug_fires_count_l16_1653


namespace NUMINAMATH_CALUDE_probability_largest_smaller_theorem_l16_1617

/-- The probability that the largest number in each row is smaller than the largest number in each row with more numbers, given n rows arranged as described. -/
def probability_largest_smaller (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / (n + 1).factorial

/-- Theorem stating the probability for the arrangement of numbers in rows. -/
theorem probability_largest_smaller_theorem (n : ℕ) :
  let total_numbers := n * (n + 1) / 2
  let row_sizes := List.range n.succ
  probability_largest_smaller n =
    (2 ^ n : ℚ) / (n + 1).factorial :=
by
  sorry

end NUMINAMATH_CALUDE_probability_largest_smaller_theorem_l16_1617
