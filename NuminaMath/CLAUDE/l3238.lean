import Mathlib

namespace NUMINAMATH_CALUDE_cube_surface_area_l3238_323835

/-- Given three vertices of a cube, prove that its surface area is 150 -/
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) : 
  A = (5, 9, 6) → B = (5, 14, 6) → C = (5, 14, 11) → 
  (let surface_area := 6 * (B.2 - A.2)^2
   surface_area = 150) :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3238_323835


namespace NUMINAMATH_CALUDE_silver_car_percentage_l3238_323820

theorem silver_car_percentage (initial_cars : ℕ) (initial_silver_percent : ℚ)
  (new_cars : ℕ) (new_non_silver_percent : ℚ) :
  initial_cars = 40 →
  initial_silver_percent = 1/5 →
  new_cars = 80 →
  new_non_silver_percent = 1/2 →
  let total_cars := initial_cars + new_cars
  let initial_silver := initial_cars * initial_silver_percent
  let new_silver := new_cars * (1 - new_non_silver_percent)
  let total_silver := initial_silver + new_silver
  (total_silver / total_cars) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_silver_car_percentage_l3238_323820


namespace NUMINAMATH_CALUDE_volume_maximized_at_two_l3238_323811

/-- The volume function of the box -/
def volume (x : ℝ) : ℝ := 4 * x * (6 - x)^2

/-- The side length of the original square sheet -/
def original_side : ℝ := 12

/-- Theorem stating that the volume is maximized when x = 2 -/
theorem volume_maximized_at_two :
  ∃ (max_x : ℝ), max_x = 2 ∧
  ∀ (x : ℝ), 0 < x ∧ x < original_side / 2 → volume x ≤ volume max_x :=
sorry

end NUMINAMATH_CALUDE_volume_maximized_at_two_l3238_323811


namespace NUMINAMATH_CALUDE_total_rooms_to_paint_l3238_323854

theorem total_rooms_to_paint 
  (time_per_room : ℕ) 
  (rooms_painted : ℕ) 
  (time_remaining : ℕ) : 
  time_per_room = 7 → 
  rooms_painted = 2 → 
  time_remaining = 63 → 
  rooms_painted + (time_remaining / time_per_room) = 11 :=
by sorry

end NUMINAMATH_CALUDE_total_rooms_to_paint_l3238_323854


namespace NUMINAMATH_CALUDE_amber_max_ounces_l3238_323894

/-- Represents the amount of money Amber has to spend -/
def amberMoney : ℚ := 7

/-- Represents the cost of a bag of candy in dollars -/
def candyCost : ℚ := 1

/-- Represents the number of ounces in a bag of candy -/
def candyOunces : ℚ := 12

/-- Represents the cost of a bag of chips in dollars -/
def chipsCost : ℚ := 1.4

/-- Represents the number of ounces in a bag of chips -/
def chipsOunces : ℚ := 17

/-- Calculates the maximum number of ounces Amber can get -/
def maxOunces : ℚ := max (amberMoney / candyCost * candyOunces) (amberMoney / chipsCost * chipsOunces)

theorem amber_max_ounces : maxOunces = 85 := by sorry

end NUMINAMATH_CALUDE_amber_max_ounces_l3238_323894


namespace NUMINAMATH_CALUDE_shipping_cost_formula_l3238_323845

/-- The shipping cost function for a premium service -/
def shippingCost (P : ℕ) : ℕ :=
  12 + 5 * (P - 1)

/-- Theorem: The shipping cost for a parcel of P pounds is 12 + 5(P-1) cents -/
theorem shipping_cost_formula (P : ℕ) (h : P ≥ 1) :
  shippingCost P = 12 + 5 * (P - 1) := by
  sorry

#check shipping_cost_formula

end NUMINAMATH_CALUDE_shipping_cost_formula_l3238_323845


namespace NUMINAMATH_CALUDE_modular_inverse_40_mod_61_l3238_323824

theorem modular_inverse_40_mod_61 :
  (∃ x : ℤ, 21 * x ≡ 1 [ZMOD 61] ∧ x ≡ 15 [ZMOD 61]) →
  (∃ y : ℤ, 40 * y ≡ 1 [ZMOD 61] ∧ y ≡ 46 [ZMOD 61]) :=
by sorry

end NUMINAMATH_CALUDE_modular_inverse_40_mod_61_l3238_323824


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_equation_l3238_323822

theorem perpendicular_lines_slope_equation (k₁ k₂ n : ℝ) : 
  (2 * k₁^2 + 8 * k₁ + n = 0) →
  (2 * k₂^2 + 8 * k₂ + n = 0) →
  (k₁ * k₂ = -1) →
  n = -2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_equation_l3238_323822


namespace NUMINAMATH_CALUDE_simplify_expression_l3238_323852

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 9) - (x + 6)*(3*x + 2) = 3*x - 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3238_323852


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3238_323823

/-- The speed of a boat in still water given its travel distances with and against a stream -/
theorem boat_speed_in_still_water 
  (along_stream : ℝ) 
  (against_stream : ℝ) 
  (h1 : along_stream = 16) 
  (h2 : against_stream = 6) : 
  ∃ (boat_speed stream_speed : ℝ), 
    boat_speed + stream_speed = along_stream ∧ 
    boat_speed - stream_speed = against_stream ∧ 
    boat_speed = 11 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3238_323823


namespace NUMINAMATH_CALUDE_system_stable_l3238_323838

-- Define the system of differential equations
def system (x y : ℝ → ℝ) : Prop :=
  ∀ t, (deriv x t = -y t) ∧ (deriv y t = x t)

-- Define Lyapunov stability for the zero solution
def lyapunov_stable (x y : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x₀ y₀ : ℝ,
    x₀^2 + y₀^2 < δ^2 →
    (∀ t ≥ 0, x t^2 + y t^2 < ε^2) ∧
    (x 0 = x₀) ∧ (y 0 = y₀) ∧ system x y

-- Theorem statement
theorem system_stable :
  ∃ x y : ℝ → ℝ, lyapunov_stable x y ∧ system x y ∧ x 0 = 0 ∧ y 0 = 0 :=
sorry

end NUMINAMATH_CALUDE_system_stable_l3238_323838


namespace NUMINAMATH_CALUDE_f_continuous_iff_l3238_323836

noncomputable def f (b c x : ℝ) : ℝ :=
  if x ≤ 4 then 2 * x^2 + 5
  else if x ≤ 6 then b * x + 3
  else c * x^2 - 2 * x + 9

theorem f_continuous_iff (b c : ℝ) :
  Continuous (f b c) ↔ b = 8.5 ∧ c = 19/12 := by sorry

end NUMINAMATH_CALUDE_f_continuous_iff_l3238_323836


namespace NUMINAMATH_CALUDE_toll_formula_correct_l3238_323851

/-- Represents the toll formula for a truck crossing a bridge -/
def toll_formula (x : ℕ) : ℚ := 0.50 + 0.30 * x

/-- Represents an 18-wheel truck with 2 wheels on its front axle and 4 wheels on each other axle -/
def eighteen_wheel_truck : ℕ := 5

theorem toll_formula_correct : 
  toll_formula eighteen_wheel_truck = 2 := by sorry

end NUMINAMATH_CALUDE_toll_formula_correct_l3238_323851


namespace NUMINAMATH_CALUDE_andrew_bought_62_eggs_l3238_323805

/-- Represents the number of eggs Andrew has at different points -/
structure EggCount where
  initial : Nat
  final : Nat

/-- Calculates the number of eggs bought -/
def eggsBought (e : EggCount) : Nat :=
  e.final - e.initial

/-- Theorem stating that Andrew bought 62 eggs -/
theorem andrew_bought_62_eggs :
  let e : EggCount := { initial := 8, final := 70 }
  eggsBought e = 62 := by
  sorry

end NUMINAMATH_CALUDE_andrew_bought_62_eggs_l3238_323805


namespace NUMINAMATH_CALUDE_distance_after_four_hours_l3238_323848

/-- The distance between two students walking in opposite directions -/
def distance_between_students (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem: The distance between two students walking in opposite directions for 4 hours,
    with speeds of 6 km/hr and 9 km/hr respectively, is 60 km. -/
theorem distance_after_four_hours :
  distance_between_students 6 9 4 = 60 := by
  sorry

#eval distance_between_students 6 9 4

end NUMINAMATH_CALUDE_distance_after_four_hours_l3238_323848


namespace NUMINAMATH_CALUDE_first_quarter_2016_has_91_days_l3238_323825

/-- The number of days in the first quarter of 2016 -/
def first_quarter_days_2016 : ℕ :=
  let year := 2016
  let is_leap_year := year % 4 = 0
  let february_days := if is_leap_year then 29 else 28
  let january_days := 31
  let march_days := 31
  january_days + february_days + march_days

/-- Theorem stating that the first quarter of 2016 has 91 days -/
theorem first_quarter_2016_has_91_days :
  first_quarter_days_2016 = 91 := by
  sorry

end NUMINAMATH_CALUDE_first_quarter_2016_has_91_days_l3238_323825


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3238_323879

theorem simplify_fraction_product : 5 * (14 / 9) * (27 / -63) = -30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3238_323879


namespace NUMINAMATH_CALUDE_fraction_equality_l3238_323892

theorem fraction_equality : (2 + 4 - 8 + 16 + 32 - 64) / (4 + 8 - 16 + 32 + 64 - 128) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3238_323892


namespace NUMINAMATH_CALUDE_basketball_team_selection_count_l3238_323813

/-- The number of ways to select a basketball team lineup with specific roles. -/
theorem basketball_team_selection_count :
  let total_members : ℕ := 15
  let leadership_material : ℕ := 6
  let positions_to_fill : ℕ := 5
  
  -- Number of ways to select captain and vice-captain
  let leadership_selection : ℕ := leadership_material * (leadership_material - 1)
  
  -- Number of ways to select 5 position players from remaining members
  let position_selection : ℕ := 
    (total_members - 2) * (total_members - 3) * (total_members - 4) * 
    (total_members - 5) * (total_members - 6)
  
  leadership_selection * position_selection = 3326400 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_count_l3238_323813


namespace NUMINAMATH_CALUDE_g_of_5_equals_15_l3238_323870

def g (x : ℝ) : ℝ := x^2 - 2*x

theorem g_of_5_equals_15 : g 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_15_l3238_323870


namespace NUMINAMATH_CALUDE_farm_leg_count_l3238_323882

def farm_animals : ℕ := 13
def chickens : ℕ := 4
def chicken_legs : ℕ := 2
def buffalo_legs : ℕ := 4

theorem farm_leg_count : 
  (chickens * chicken_legs) + ((farm_animals - chickens) * buffalo_legs) = 44 := by
  sorry

end NUMINAMATH_CALUDE_farm_leg_count_l3238_323882


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3238_323871

theorem solve_exponential_equation : ∃ x : ℝ, (100 : ℝ) ^ 4 = 5 ^ x ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3238_323871


namespace NUMINAMATH_CALUDE_product_greater_than_sum_minus_one_l3238_323895

theorem product_greater_than_sum_minus_one {a₁ a₂ : ℝ} 
  (h₁ : 0 < a₁) (h₂ : a₁ < 1) (h₃ : 0 < a₂) (h₄ : a₂ < 1) : 
  a₁ * a₂ > a₁ + a₂ - 1 := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_minus_one_l3238_323895


namespace NUMINAMATH_CALUDE_max_triangle_perimeter_l3238_323846

/-- Given a triangle with two sides of length 8 and 15 units, and the third side
    length x being an integer, the maximum perimeter of the triangle is 45 units. -/
theorem max_triangle_perimeter :
  ∀ x : ℤ,
  (8 : ℝ) + 15 > (x : ℝ) →
  (8 : ℝ) + (x : ℝ) > 15 →
  (15 : ℝ) + (x : ℝ) > 8 →
  (∀ y : ℤ, (8 : ℝ) + 15 > (y : ℝ) →
             (8 : ℝ) + (y : ℝ) > 15 →
             (15 : ℝ) + (y : ℝ) > 8 →
             8 + 15 + (x : ℝ) ≥ 8 + 15 + (y : ℝ)) →
  8 + 15 + (x : ℝ) = 45 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_perimeter_l3238_323846


namespace NUMINAMATH_CALUDE_ring_price_is_7_10_l3238_323887

/-- Represents the sales at a craft fair -/
structure CraftFairSales where
  necklace_price : ℝ
  earrings_price : ℝ
  ring_price : ℝ
  necklaces_sold : ℕ
  rings_sold : ℕ
  earrings_sold : ℕ
  bracelets_sold : ℕ
  total_sales : ℝ

/-- The cost of a bracelet is twice the cost of a ring -/
def bracelet_price (sales : CraftFairSales) : ℝ := 2 * sales.ring_price

/-- Theorem stating that the ring price is $7.10 given the conditions -/
theorem ring_price_is_7_10 (sales : CraftFairSales) 
  (h1 : sales.necklace_price = 12)
  (h2 : sales.earrings_price = 10)
  (h3 : sales.necklaces_sold = 4)
  (h4 : sales.rings_sold = 8)
  (h5 : sales.earrings_sold = 5)
  (h6 : sales.bracelets_sold = 6)
  (h7 : sales.total_sales = 240)
  (h8 : sales.necklace_price * sales.necklaces_sold + 
        sales.ring_price * sales.rings_sold + 
        sales.earrings_price * sales.earrings_sold + 
        bracelet_price sales * sales.bracelets_sold = sales.total_sales) :
  sales.ring_price = 7.1 := by
  sorry


end NUMINAMATH_CALUDE_ring_price_is_7_10_l3238_323887


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l3238_323832

/-- Given a quadratic inequality ax^2 + bx + 2 > 0 with solution set (-1/2, 1/3),
    prove that a + b = -14 -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l3238_323832


namespace NUMINAMATH_CALUDE_smallest_frood_number_l3238_323886

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_frood_number : 
  ∀ n : ℕ, n > 0 → (n < 10 → sum_first_n n ≤ 5 * n) ∧ (sum_first_n 10 > 5 * 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_frood_number_l3238_323886


namespace NUMINAMATH_CALUDE_maddie_tshirt_cost_l3238_323891

/-- Calculates the total cost of T-shirts bought by Maddie -/
def total_cost (white_packs blue_packs white_per_pack blue_per_pack cost_per_shirt : ℕ) : ℕ :=
  ((white_packs * white_per_pack + blue_packs * blue_per_pack) * cost_per_shirt)

/-- Proves that Maddie spent $66 on T-shirts -/
theorem maddie_tshirt_cost :
  total_cost 2 4 5 3 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_maddie_tshirt_cost_l3238_323891


namespace NUMINAMATH_CALUDE_remainder_99_power_36_mod_100_l3238_323817

theorem remainder_99_power_36_mod_100 : 99^36 ≡ 1 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_remainder_99_power_36_mod_100_l3238_323817


namespace NUMINAMATH_CALUDE_distance_probability_l3238_323802

-- Define the points and distances
def A : ℝ × ℝ := (0, -10)
def B : ℝ × ℝ := (0, 0)
def AB : ℝ := 10
def BC : ℝ := 6
def AC_max : ℝ := 8

-- Define the angle range
def angle_range : Set ℝ := Set.Ioo 0 Real.pi

-- Define the probability function
noncomputable def probability_AC_less_than_8 : ℝ :=
  (30 : ℝ) / 180

-- State the theorem
theorem distance_probability :
  probability_AC_less_than_8 = 1/6 := by sorry

end NUMINAMATH_CALUDE_distance_probability_l3238_323802


namespace NUMINAMATH_CALUDE_toy_shop_problem_l3238_323867

/-- Toy shop problem -/
theorem toy_shop_problem 
  (total_A : ℝ) (total_B : ℝ) (diff : ℕ) (ratio : ℝ) 
  (sell_A : ℝ) (sell_B : ℝ) (total_toys : ℕ) (min_profit : ℝ) :
  total_A = 1200 →
  total_B = 1500 →
  diff = 20 →
  ratio = 1.5 →
  sell_A = 12 →
  sell_B = 20 →
  total_toys = 75 →
  min_profit = 300 →
  ∃ (cost_A cost_B : ℝ) (max_A : ℕ),
    -- Part 1: Cost of toys
    cost_A = 10 ∧ 
    cost_B = 15 ∧
    total_A / cost_A - total_B / cost_B = diff ∧
    cost_B = ratio * cost_A ∧
    -- Part 2: Maximum number of type A toys
    max_A = 25 ∧
    ∀ m : ℕ, 
      m ≤ total_toys →
      (sell_A - cost_A) * m + (sell_B - cost_B) * (total_toys - m) ≥ min_profit →
      m ≤ max_A := by
  sorry

end NUMINAMATH_CALUDE_toy_shop_problem_l3238_323867


namespace NUMINAMATH_CALUDE_birthday_cake_icing_l3238_323839

/-- Represents a rectangular cake with given dimensions -/
structure Cake where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a smaller cuboid piece of the cake -/
structure Piece where
  size : ℕ

/-- Calculates the number of pieces with icing on exactly two sides -/
def pieces_with_two_sided_icing (c : Cake) (p : Piece) : ℕ :=
  sorry

/-- Theorem stating that a 6 × 4 × 4 cake cut into 2 × 2 × 2 pieces has 16 pieces with icing on two sides -/
theorem birthday_cake_icing (c : Cake) (p : Piece) :
  c.length = 6 ∧ c.width = 4 ∧ c.height = 4 ∧ p.size = 2 →
  pieces_with_two_sided_icing c p = 16 :=
by sorry

end NUMINAMATH_CALUDE_birthday_cake_icing_l3238_323839


namespace NUMINAMATH_CALUDE_quadrilateral_fixed_point_theorem_l3238_323861

-- Define the plane
variable (Plane : Type)

-- Define points in the plane
variable (Point : Type)
variable (A B C D P : Point)

-- Define the distance function
variable (distance : Point → Point → ℝ)

-- Define the angle function
variable (angle : Point → Point → Point → ℝ)

-- Define the line through two points
variable (line_through : Point → Point → Set Point)

-- Define the "lies on" relation
variable (lies_on : Point → Set Point → Prop)

-- Theorem statement
theorem quadrilateral_fixed_point_theorem :
  ∃ P : Point,
    ∀ C D : Point,
      distance A B = distance B C →
      distance A D = distance D C →
      angle A D C = Real.pi / 2 →
      lies_on P (line_through C D) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_fixed_point_theorem_l3238_323861


namespace NUMINAMATH_CALUDE_mountain_height_l3238_323890

/-- Given a mountain where a person makes 10 round trips, reaching 3/4 of the height each time,
    and covering a total distance of 600,000 feet, the height of the mountain is 80,000 feet. -/
theorem mountain_height (trips : ℕ) (fraction_reached : ℚ) (total_distance : ℕ) 
    (h1 : trips = 10)
    (h2 : fraction_reached = 3/4)
    (h3 : total_distance = 600000) :
  (total_distance : ℚ) / (2 * trips * fraction_reached) = 80000 := by
  sorry

end NUMINAMATH_CALUDE_mountain_height_l3238_323890


namespace NUMINAMATH_CALUDE_wire_length_l3238_323878

/-- The length of a wire cut into two pieces, where one piece is 2/3 of the other --/
theorem wire_length (shorter_piece : ℝ) (h : shorter_piece = 27.999999999999993) : 
  ∃ (longer_piece total_length : ℝ),
    longer_piece = (2/3) * shorter_piece ∧
    total_length = shorter_piece + longer_piece ∧
    total_length = 46.66666666666666 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_l3238_323878


namespace NUMINAMATH_CALUDE_isosceles_from_cosine_relation_l3238_323883

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  angle_bounds : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π

/-- A triangle is isosceles if it has at least two equal sides -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The main theorem: if a = 2b cos C, then the triangle is isosceles -/
theorem isosceles_from_cosine_relation (t : Triangle) (h : t.a = 2 * t.b * Real.cos t.C) :
  t.isIsosceles := by
  sorry

end NUMINAMATH_CALUDE_isosceles_from_cosine_relation_l3238_323883


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l3238_323806

/-- A line represented by y = mx + b -/
structure Line where
  m : ℚ
  b : ℚ

/-- The point of intersection of two lines -/
def intersection (l1 l2 : Line) : ℚ × ℚ :=
  let x := (l2.b - l1.b) / (l1.m - l2.m)
  let y := l1.m * x + l1.b
  (x, y)

/-- Theorem: If three lines intersect at a single point, and two of them are
    y = 3x + 5 and y = -5x + 20, then the third line y = 4x + p must have p = 25/8 -/
theorem intersection_of_three_lines
  (l1 : Line)
  (l2 : Line)
  (l3 : Line)
  (h1 : l1 = ⟨3, 5⟩)
  (h2 : l2 = ⟨-5, 20⟩)
  (h3 : l3.m = 4)
  (h_intersect : intersection l1 l2 = intersection l2 l3) :
  l3.b = 25/8 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l3238_323806


namespace NUMINAMATH_CALUDE_total_cotton_is_sixty_l3238_323899

/-- The amount of cotton needed for one tee-shirt in feet -/
def cotton_per_shirt : ℕ := 4

/-- The number of tee-shirts to be made -/
def num_shirts : ℕ := 15

/-- The total amount of cotton needed for all tee-shirts in feet -/
def total_cotton : ℕ := cotton_per_shirt * num_shirts

/-- Theorem stating that the total amount of cotton needed is 60 feet -/
theorem total_cotton_is_sixty : total_cotton = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_cotton_is_sixty_l3238_323899


namespace NUMINAMATH_CALUDE_mias_socks_theorem_l3238_323808

/-- Represents the number of pairs of socks at each price point --/
structure SockInventory where
  one_dollar : ℕ
  two_dollar : ℕ
  three_dollar : ℕ
  four_dollar : ℕ

/-- Calculates the total number of pairs of socks --/
def total_pairs (s : SockInventory) : ℕ :=
  s.one_dollar + s.two_dollar + s.three_dollar + s.four_dollar

/-- Calculates the total cost of all socks --/
def total_cost (s : SockInventory) : ℕ :=
  s.one_dollar + 2 * s.two_dollar + 3 * s.three_dollar + 4 * s.four_dollar

/-- Checks if at least one pair of each type was bought --/
def at_least_one_each (s : SockInventory) : Prop :=
  s.one_dollar ≥ 1 ∧ s.two_dollar ≥ 1 ∧ s.three_dollar ≥ 1 ∧ s.four_dollar ≥ 1

theorem mias_socks_theorem (s : SockInventory) 
  (h1 : total_pairs s = 16)
  (h2 : total_cost s = 36)
  (h3 : at_least_one_each s) :
  s.one_dollar = 3 := by
  sorry

end NUMINAMATH_CALUDE_mias_socks_theorem_l3238_323808


namespace NUMINAMATH_CALUDE_gcd_180_450_l3238_323807

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_450_l3238_323807


namespace NUMINAMATH_CALUDE_boys_average_weight_l3238_323853

/-- Proves that given a group of 10 students with 5 girls and 5 boys, where the average weight of
    the girls is 45 kg and the average weight of all students is 50 kg, then the average weight
    of the boys is 55 kg. -/
theorem boys_average_weight 
  (num_students : Nat) 
  (num_girls : Nat) 
  (num_boys : Nat) 
  (girls_avg_weight : ℝ) 
  (total_avg_weight : ℝ) : ℝ :=
by
  have h1 : num_students = 10 := by sorry
  have h2 : num_girls = 5 := by sorry
  have h3 : num_boys = 5 := by sorry
  have h4 : girls_avg_weight = 45 := by sorry
  have h5 : total_avg_weight = 50 := by sorry

  -- The average weight of the boys
  let boys_avg_weight : ℝ := 55

  -- Proof that boys_avg_weight = 55
  sorry

end NUMINAMATH_CALUDE_boys_average_weight_l3238_323853


namespace NUMINAMATH_CALUDE_kopek_payment_l3238_323884

theorem kopek_payment (n : ℕ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end NUMINAMATH_CALUDE_kopek_payment_l3238_323884


namespace NUMINAMATH_CALUDE_new_average_age_l3238_323828

def initial_people : ℕ := 8
def initial_average_age : ℚ := 25
def leaving_person_age : ℕ := 20
def remaining_people : ℕ := 7

theorem new_average_age :
  (initial_people * initial_average_age - leaving_person_age) / remaining_people = 180 / 7 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l3238_323828


namespace NUMINAMATH_CALUDE_object_distances_l3238_323855

-- Define the parameters
def speed1 : ℝ := 3
def speed2 : ℝ := 4
def initial_distance : ℝ := 20
def final_distance : ℝ := 10
def time_elapsed : ℝ := 2

-- Define the theorem
theorem object_distances (x y : ℝ) :
  -- Conditions
  (x^2 + y^2 = initial_distance^2) →
  ((x - speed1 * time_elapsed)^2 + (y - speed2 * time_elapsed)^2 = final_distance^2) →
  -- Conclusion
  (x = 12 ∧ y = 16) :=
by sorry

end NUMINAMATH_CALUDE_object_distances_l3238_323855


namespace NUMINAMATH_CALUDE_correct_factorization_l3238_323842

theorem correct_factorization (x : ℝ) : 1 - 2*x + x^2 = (1 - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l3238_323842


namespace NUMINAMATH_CALUDE_area_of_triangle_abc_l3238_323864

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup : ℕ → Circle
| 0 => ⟨(-7, 3), 3⟩  -- Circle A
| 1 => ⟨(0, 4), 4⟩   -- Circle B
| 2 => ⟨(9, 5), 5⟩   -- Circle C
| _ => ⟨(0, 0), 0⟩   -- Default case

-- Define the line l (y = 0 in this case)
def line_l : ℝ → ℝ := λ _ => 0

-- Theorem statement
theorem area_of_triangle_abc :
  let a := problem_setup 0
  let b := problem_setup 1
  let c := problem_setup 2
  (∀ i, (problem_setup i).center.2 - (problem_setup i).radius = line_l (problem_setup i).center.1) →
  ((b.center.1 - a.center.1)^2 + (b.center.2 - a.center.2)^2 = (b.radius + a.radius)^2) →
  ((c.center.1 - b.center.1)^2 + (c.center.2 - b.center.2)^2 = (c.radius + b.radius)^2) →
  let area := abs ((a.center.1 * (b.center.2 - c.center.2) + 
                    b.center.1 * (c.center.2 - a.center.2) + 
                    c.center.1 * (a.center.2 - b.center.2)) / 2)
  area = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_abc_l3238_323864


namespace NUMINAMATH_CALUDE_complex_sum_modulus_l3238_323888

theorem complex_sum_modulus (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs (z₁ - z₂) = Real.sqrt 3) : 
  Complex.abs (z₁ + z₂) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_modulus_l3238_323888


namespace NUMINAMATH_CALUDE_min_value_of_function_l3238_323857

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  ∃ (m : ℝ), m = 8 ∧ ∀ y, y = x + 1/x + 16*x/(x^2 + 1) → y ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3238_323857


namespace NUMINAMATH_CALUDE_not_product_of_consecutive_numbers_l3238_323803

theorem not_product_of_consecutive_numbers (n k : ℕ) :
  ¬ ∃ x : ℕ, x * (x + 1) = 2 * n^(3*k) + 4 * n^k + 10 := by
  sorry

end NUMINAMATH_CALUDE_not_product_of_consecutive_numbers_l3238_323803


namespace NUMINAMATH_CALUDE_circumscribed_iff_similar_when_moved_l3238_323897

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- A function to check if a polygon is convex -/
def isConvex (p : Polygon) : Prop :=
  sorry

/-- A function to move all sides of a polygon outward by a distance -/
def moveOutward (p : Polygon) (distance : ℝ) : Polygon :=
  sorry

/-- A function to check if two polygons are similar -/
def areSimilar (p1 p2 : Polygon) : Prop :=
  sorry

/-- A function to check if a polygon is circumscribed -/
def isCircumscribed (p : Polygon) : Prop :=
  sorry

/-- Theorem: A convex polygon is circumscribed if and only if 
    moving all its sides outward by a distance of 1 results 
    in a polygon similar to the original one -/
theorem circumscribed_iff_similar_when_moved (p : Polygon) :
  isConvex p →
  isCircumscribed p ↔ areSimilar p (moveOutward p 1) :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_iff_similar_when_moved_l3238_323897


namespace NUMINAMATH_CALUDE_solve_cookies_problem_l3238_323889

def cookies_problem (total_cookies : ℕ) (cookies_per_guest : ℕ) : Prop :=
  total_cookies = 10 ∧ cookies_per_guest = 2 →
  total_cookies / cookies_per_guest = 5

theorem solve_cookies_problem : cookies_problem 10 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_cookies_problem_l3238_323889


namespace NUMINAMATH_CALUDE_probability_selecting_two_types_l3238_323847

theorem probability_selecting_two_types (total : ℕ) (type_c : ℕ) (type_r : ℕ) (type_a : ℕ) :
  total = type_c + type_r + type_a →
  type_c = type_r →
  type_a = 1 →
  (type_c : ℚ) * type_r / (total * (total - 1)) = 5 / 11 :=
by sorry

end NUMINAMATH_CALUDE_probability_selecting_two_types_l3238_323847


namespace NUMINAMATH_CALUDE_ice_skating_skiing_probability_l3238_323896

theorem ice_skating_skiing_probability (P_ice_skating P_skiing P_either : ℝ)
  (h1 : P_ice_skating = 0.6)
  (h2 : P_skiing = 0.5)
  (h3 : P_either = 0.7)
  (h4 : 0 ≤ P_ice_skating ∧ P_ice_skating ≤ 1)
  (h5 : 0 ≤ P_skiing ∧ P_skiing ≤ 1)
  (h6 : 0 ≤ P_either ∧ P_either ≤ 1) :
  (P_ice_skating + P_skiing - P_either) / P_skiing = 0.8 :=
by sorry

end NUMINAMATH_CALUDE_ice_skating_skiing_probability_l3238_323896


namespace NUMINAMATH_CALUDE_aunt_gift_amount_l3238_323830

def birthday_money_problem (grandmother_gift aunt_gift uncle_gift total_money game_cost games_bought remaining_money : ℕ) : Prop :=
  grandmother_gift = 20 ∧
  uncle_gift = 30 ∧
  total_money = 125 ∧
  game_cost = 35 ∧
  games_bought = 3 ∧
  remaining_money = 20 ∧
  total_money = grandmother_gift + aunt_gift + uncle_gift ∧
  total_money = game_cost * games_bought + remaining_money

theorem aunt_gift_amount :
  ∀ (grandmother_gift aunt_gift uncle_gift total_money game_cost games_bought remaining_money : ℕ),
    birthday_money_problem grandmother_gift aunt_gift uncle_gift total_money game_cost games_bought remaining_money →
    aunt_gift = 75 := by
  sorry

end NUMINAMATH_CALUDE_aunt_gift_amount_l3238_323830


namespace NUMINAMATH_CALUDE_smallest_N_satisfying_condition_l3238_323866

def P (N : ℕ) : ℚ := (4 * N + 2) / (5 * N + 1)

theorem smallest_N_satisfying_condition :
  ∃ (N : ℕ), N > 0 ∧ N % 5 = 0 ∧ P N < 321 / 400 ∧
  ∀ (M : ℕ), M > 0 → M % 5 = 0 → P M < 321 / 400 → N ≤ M ∧
  N = 480 := by
  sorry

end NUMINAMATH_CALUDE_smallest_N_satisfying_condition_l3238_323866


namespace NUMINAMATH_CALUDE_vector_addition_l3238_323885

def vector1 : Fin 2 → ℤ := ![5, -9]
def vector2 : Fin 2 → ℤ := ![-8, 14]

theorem vector_addition :
  (vector1 + vector2) = ![(-3 : ℤ), 5] := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l3238_323885


namespace NUMINAMATH_CALUDE_monthly_payment_calculation_l3238_323804

def original_price : ℝ := 480
def discount_percentage : ℝ := 5
def first_installment : ℝ := 150
def num_installments : ℕ := 3

theorem monthly_payment_calculation :
  let discounted_price := original_price * (1 - discount_percentage / 100)
  let remaining_balance := discounted_price - first_installment
  let monthly_payment := remaining_balance / num_installments
  monthly_payment = 102 := by
  sorry

end NUMINAMATH_CALUDE_monthly_payment_calculation_l3238_323804


namespace NUMINAMATH_CALUDE_line_MN_tangent_to_circle_l3238_323800

-- Define the necessary types
variable (Point Line Circle : Type)

-- Define the necessary relations and functions
variable (on_line : Point → Line → Prop)
variable (on_circle : Point → Circle → Prop)
variable (center : Circle → Point)
variable (tangent_line : Line → Circle → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Point)
variable (line_through : Point → Point → Line)

-- Define the given points, lines, and circle
variable (A B C O E F R P N M : Point)
variable (AB AC PR EF MN : Line)
variable (ω : Circle)

-- State the theorem
theorem line_MN_tangent_to_circle (h1 : ¬ on_line A (line_through B C))
  (h2 : center ω = O)
  (h3 : tangent_line AC ω)
  (h4 : tangent_line AB ω)
  (h5 : on_line E AC)
  (h6 : on_line F AB)
  (h7 : on_line R EF)
  (h8 : parallel (line_through O P) EF)
  (h9 : on_line P AB)
  (h10 : N = intersect PR AC)
  (h11 : M = intersect AB (line_through R C))
  (h12 : parallel (line_through R C) AC) :
  tangent_line MN ω :=
sorry

end NUMINAMATH_CALUDE_line_MN_tangent_to_circle_l3238_323800


namespace NUMINAMATH_CALUDE_triangle_shape_l3238_323843

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : 0 < B) (h6 : 0 < C) 
  (h7 : A + B + C = Real.pi)
  (h8 : 2 * a * Real.cos B = c)
  (h9 : a * Real.sin B = b * Real.sin A)
  (h10 : b * Real.sin C = c * Real.sin B)
  (h11 : c * Real.sin A = a * Real.sin C) :
  A = B ∨ B = C ∨ A = C :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l3238_323843


namespace NUMINAMATH_CALUDE_thursday_beef_sales_l3238_323827

/-- Given a store's beef sales over three days, prove that Thursday's sales were 210 pounds -/
theorem thursday_beef_sales (x : ℝ) : 
  (x + 2*x + 150) / 3 = 260 → x = 210 := by sorry

end NUMINAMATH_CALUDE_thursday_beef_sales_l3238_323827


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3238_323829

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 - a - 2) (a + 1)
  (z.re = 0 ∧ z.im ≠ 0) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3238_323829


namespace NUMINAMATH_CALUDE_unique_solution_congruence_system_l3238_323821

theorem unique_solution_congruence_system :
  ∀ x y z : ℤ,
  2 ≤ x ∧ x ≤ y ∧ y ≤ z →
  (x * y) % z = 1 →
  (x * z) % y = 1 →
  (y * z) % x = 1 →
  x = 2 ∧ y = 3 ∧ z = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_congruence_system_l3238_323821


namespace NUMINAMATH_CALUDE_class_size_l3238_323877

theorem class_size (total : ℕ) (girls : ℕ) (boys : ℕ) :
  girls = total * 52 / 100 →
  girls = boys + 1 →
  total = girls + boys →
  total = 25 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l3238_323877


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l3238_323881

theorem negation_of_existence_inequality (p : Prop) :
  (p ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) →
  (¬p ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l3238_323881


namespace NUMINAMATH_CALUDE_total_wine_age_l3238_323837

-- Define the ages of the wines
def carlo_rosi_age : ℕ := 40
def franzia_age : ℕ := 3 * carlo_rosi_age
def twin_valley_age : ℕ := carlo_rosi_age / 4

-- Theorem statement
theorem total_wine_age :
  franzia_age + carlo_rosi_age + twin_valley_age = 170 :=
by sorry

end NUMINAMATH_CALUDE_total_wine_age_l3238_323837


namespace NUMINAMATH_CALUDE_triangle_value_is_five_l3238_323818

/-- Represents a digit in base 6 --/
def Base6Digit := Fin 6

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : List Base6Digit := sorry

/-- Performs addition in base 6 --/
def addBase6 (a b : List Base6Digit) : List Base6Digit := sorry

/-- Theorem: The value of ▲ in the given base-6 addition problem is 5 --/
theorem triangle_value_is_five :
  ∃ (triangle : Base6Digit),
    (triangle.val = 5) ∧
    (addBase6 (toBase6 321 ++ [triangle])
      (addBase6 (triangle :: toBase6 40) (triangle :: toBase6 2)) =
        toBase6 425 ++ [triangle]) := by sorry

end NUMINAMATH_CALUDE_triangle_value_is_five_l3238_323818


namespace NUMINAMATH_CALUDE_odd_digits_base7_528_l3238_323810

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-7 representation of 528 (base 10) is 4 -/
theorem odd_digits_base7_528 : countOddDigits (toBase7 528) = 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base7_528_l3238_323810


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3238_323869

theorem regular_polygon_interior_angle_sum :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 0 →
    exterior_angle = 20 →
    360 / exterior_angle = n →
    (n - 2) * 180 = 2880 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3238_323869


namespace NUMINAMATH_CALUDE_total_ants_l3238_323858

def ants_problem (abe beth cece duke emily frances : ℕ) : Prop :=
  abe = 4 ∧
  beth = 2 * abe ∧
  cece = 3 * abe ∧
  duke = abe / 2 ∧
  emily = abe + (75 * abe) / 100 ∧
  frances = 2 * cece ∧
  abe + beth + cece + duke + emily + frances = 57

theorem total_ants : ∃ (abe beth cece duke emily frances : ℕ),
  ants_problem abe beth cece duke emily frances :=
sorry

end NUMINAMATH_CALUDE_total_ants_l3238_323858


namespace NUMINAMATH_CALUDE_minimum_pages_required_l3238_323840

-- Define the types of cards and pages
inductive CardType
| Rare
| LimitedEdition
| Regular

inductive PageType
| NineCard
| SevenCard
| FiveCard

-- Define the card counts
def rareCardCount : Nat := 18
def limitedEditionCardCount : Nat := 21
def regularCardCount : Nat := 45

-- Define the page capacities
def pageCapacity (pt : PageType) : Nat :=
  match pt with
  | PageType.NineCard => 9
  | PageType.SevenCard => 7
  | PageType.FiveCard => 5

-- Define a function to check if a page type is valid for a card type
def isValidPageType (ct : CardType) (pt : PageType) : Bool :=
  match ct, pt with
  | CardType.Rare, PageType.NineCard => true
  | CardType.Rare, PageType.SevenCard => true
  | CardType.LimitedEdition, PageType.NineCard => true
  | CardType.LimitedEdition, PageType.SevenCard => true
  | CardType.Regular, _ => true
  | _, _ => false

-- Define the theorem
theorem minimum_pages_required :
  ∃ (rarePages limitedPages regularPages : Nat),
    rarePages * pageCapacity PageType.NineCard = rareCardCount ∧
    limitedPages * pageCapacity PageType.SevenCard = limitedEditionCardCount ∧
    regularPages * pageCapacity PageType.NineCard = regularCardCount ∧
    rarePages + limitedPages + regularPages = 10 ∧
    (∀ (rp lp regalp : Nat),
      rp * pageCapacity PageType.NineCard ≥ rareCardCount →
      lp * pageCapacity PageType.SevenCard ≥ limitedEditionCardCount →
      regalp * pageCapacity PageType.NineCard ≥ regularCardCount →
      isValidPageType CardType.Rare PageType.NineCard →
      isValidPageType CardType.LimitedEdition PageType.SevenCard →
      isValidPageType CardType.Regular PageType.NineCard →
      rp + lp + regalp ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_minimum_pages_required_l3238_323840


namespace NUMINAMATH_CALUDE_quinn_reading_rate_l3238_323856

/-- A reading challenge that lasts for a certain number of weeks -/
structure ReadingChallenge where
  duration : ℕ  -- Duration of the challenge in weeks
  books_per_coupon : ℕ  -- Number of books required for one coupon

/-- A participant in the reading challenge -/
structure Participant where
  challenge : ReadingChallenge
  coupons_earned : ℕ  -- Number of coupons earned

def books_per_week (p : Participant) : ℚ :=
  (p.coupons_earned * p.challenge.books_per_coupon : ℚ) / p.challenge.duration

theorem quinn_reading_rate (c : ReadingChallenge) (p : Participant) :
    c.duration = 10 ∧ c.books_per_coupon = 5 ∧ p.challenge = c ∧ p.coupons_earned = 4 →
    books_per_week p = 2 := by
  sorry

end NUMINAMATH_CALUDE_quinn_reading_rate_l3238_323856


namespace NUMINAMATH_CALUDE_min_cos_B_angle_A_values_l3238_323801

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.c = 6 * Real.sqrt 3 ∧ t.b = 6

-- Theorem for the minimum value of cos B
theorem min_cos_B (t : Triangle) (h : triangle_conditions t) :
  ∃ (min_cos_B : ℝ), min_cos_B = 1/3 ∧ ∀ (cos_B : ℝ), cos_B = (t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c) → cos_B ≥ min_cos_B :=
sorry

-- Theorem for the possible values of angle A
theorem angle_A_values (t : Triangle) (h1 : triangle_conditions t) (h2 : t.a * t.b * Real.cos t.C = 12) :
  t.A = π/2 ∨ t.A = π/6 :=
sorry

end NUMINAMATH_CALUDE_min_cos_B_angle_A_values_l3238_323801


namespace NUMINAMATH_CALUDE_donna_card_shop_days_l3238_323873

/-- Represents Donna's work schedule and earnings --/
structure DonnaWork where
  dog_walking_hours : ℕ
  dog_walking_rate : ℚ
  card_shop_hours : ℕ
  card_shop_rate : ℚ
  babysitting_hours : ℕ
  babysitting_rate : ℚ
  total_earnings : ℚ
  total_days : ℕ

/-- Calculates the number of days Donna worked at the card shop --/
def card_shop_days (work : DonnaWork) : ℚ :=
  let dog_walking_earnings := ↑work.dog_walking_hours * work.dog_walking_rate * ↑work.total_days
  let babysitting_earnings := ↑work.babysitting_hours * work.babysitting_rate
  let card_shop_earnings := work.total_earnings - dog_walking_earnings - babysitting_earnings
  card_shop_earnings / (↑work.card_shop_hours * work.card_shop_rate)

/-- Theorem stating that Donna worked 5 days at the card shop --/
theorem donna_card_shop_days :
  ∀ (work : DonnaWork),
  work.dog_walking_hours = 2 ∧
  work.dog_walking_rate = 10 ∧
  work.card_shop_hours = 2 ∧
  work.card_shop_rate = 25/2 ∧
  work.babysitting_hours = 4 ∧
  work.babysitting_rate = 10 ∧
  work.total_earnings = 305 ∧
  work.total_days = 7 →
  card_shop_days work = 5 := by
  sorry


end NUMINAMATH_CALUDE_donna_card_shop_days_l3238_323873


namespace NUMINAMATH_CALUDE_aquarium_count_l3238_323849

theorem aquarium_count (total_animals : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : animals_per_aquarium = 2) :
  total_animals / animals_per_aquarium = 20 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_count_l3238_323849


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3238_323831

theorem rectangular_to_polar_conversion :
  ∀ (x y : ℝ),
  x = -3 ∧ y = 1 →
  ∃ (r θ : ℝ),
  r > 0 ∧
  0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = Real.sqrt 10 ∧
  θ = Real.pi - Real.arctan (1 / 3) ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3238_323831


namespace NUMINAMATH_CALUDE_seeds_per_medium_row_is_twenty_l3238_323875

/-- Represents the garden setup with large and medium beds -/
structure GardenSetup where
  largeBeds : Nat
  mediumBeds : Nat
  largeRowsPerBed : Nat
  mediumRowsPerBed : Nat
  seedsPerLargeRow : Nat
  totalSeeds : Nat

/-- Calculates the number of seeds per row in the medium bed -/
def seedsPerMediumRow (setup : GardenSetup) : Nat :=
  let largeSeeds := setup.largeBeds * setup.largeRowsPerBed * setup.seedsPerLargeRow
  let mediumSeeds := setup.totalSeeds - largeSeeds
  let totalMediumRows := setup.mediumBeds * setup.mediumRowsPerBed
  mediumSeeds / totalMediumRows

/-- Theorem stating that the number of seeds per row in the medium bed is 20 -/
theorem seeds_per_medium_row_is_twenty :
  let setup : GardenSetup := {
    largeBeds := 2,
    mediumBeds := 2,
    largeRowsPerBed := 4,
    mediumRowsPerBed := 3,
    seedsPerLargeRow := 25,
    totalSeeds := 320
  }
  seedsPerMediumRow setup = 20 := by sorry

end NUMINAMATH_CALUDE_seeds_per_medium_row_is_twenty_l3238_323875


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l3238_323844

theorem sin_cos_difference_equals_half : 
  Real.sin (57 * π / 180) * Real.cos (27 * π / 180) - 
  Real.cos (57 * π / 180) * Real.sin (27 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l3238_323844


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3238_323812

theorem right_triangle_sides (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  (a + b + c) / 2 - a = 2/3 * r →  -- Relation derived from circle touching sides
  c = 5/3 * r →  -- Hypotenuse relation
  a * b / 2 = 2 * r →  -- Area of the triangle
  (a = 4/3 * r ∧ b = r) ∨ (a = r ∧ b = 4/3 * r) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3238_323812


namespace NUMINAMATH_CALUDE_block_division_l3238_323816

theorem block_division (n : ℕ) (weights : List ℕ) : 
  weights.length = n →
  (∀ w ∈ weights, w > 0 ∧ w < n) →
  weights.sum < 2 * n →
  ∃ subset : List ℕ, subset ⊆ weights ∧ subset.sum = n :=
by sorry

end NUMINAMATH_CALUDE_block_division_l3238_323816


namespace NUMINAMATH_CALUDE_transformed_area_is_63_l3238_323826

/-- The transformation matrix --/
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 1, -1]

/-- The original region's area --/
def original_area : ℝ := 9

/-- Theorem stating the area of the transformed region --/
theorem transformed_area_is_63 : 
  |A.det| * original_area = 63 := by sorry

end NUMINAMATH_CALUDE_transformed_area_is_63_l3238_323826


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l3238_323876

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l3238_323876


namespace NUMINAMATH_CALUDE_M_equality_l3238_323874

theorem M_equality : 
  let M := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2) - Real.sqrt (4 - 2 * Real.sqrt 3)
  M = (5/2) * Real.sqrt 2 - Real.sqrt 3 + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_M_equality_l3238_323874


namespace NUMINAMATH_CALUDE_negation_equivalence_l3238_323819

theorem negation_equivalence :
  (¬ (∀ x : ℝ, x^2 + x + 1 > 0)) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3238_323819


namespace NUMINAMATH_CALUDE_smallest_value_for_x_greater_than_one_l3238_323893

theorem smallest_value_for_x_greater_than_one (x : ℝ) (hx : x > 1) :
  (1 / x < x) ∧ (1 / x < x^2) ∧ (1 / x < 2*x) ∧ (1 / x < Real.sqrt x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_greater_than_one_l3238_323893


namespace NUMINAMATH_CALUDE_valid_basis_l3238_323815

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (3, -1)
def a : ℝ × ℝ := (3, 4)

theorem valid_basis :
  ∃ (x y : ℝ), x • e₁ + y • e₂ = a ∧ ¬(∃ (k : ℝ), e₁ = k • e₂) :=
sorry

end NUMINAMATH_CALUDE_valid_basis_l3238_323815


namespace NUMINAMATH_CALUDE_final_number_is_172_l3238_323880

/-- Represents the state of the board at any given time -/
structure BoardState where
  numbers : List Nat
  deriving Repr

/-- The operation of erasing two numbers and replacing them with their sum minus 1 -/
def boardOperation (state : BoardState) (i j : Nat) : BoardState :=
  { numbers := 
      (state.numbers.removeNth i).removeNth j ++ 
      [state.numbers[i]! + state.numbers[j]! - 1] }

/-- The invariant of the board state -/
def boardInvariant (state : BoardState) : Int :=
  state.numbers.sum - state.numbers.length

/-- Initial board state with numbers 1 to 20 -/
def initialBoard : BoardState :=
  { numbers := List.range 20 |>.map (· + 1) }

/-- Theorem stating that after 19 operations, the final number on the board is 172 -/
theorem final_number_is_172 : 
  ∃ (operations : List (Nat × Nat)),
    operations.length = 19 ∧
    (operations.foldl 
      (fun state (i, j) => boardOperation state i j) 
      initialBoard).numbers = [172] := by
  sorry

end NUMINAMATH_CALUDE_final_number_is_172_l3238_323880


namespace NUMINAMATH_CALUDE_problem_solution_l3238_323834

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 25) : 
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 82.1762 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3238_323834


namespace NUMINAMATH_CALUDE_square_diagonal_double_area_l3238_323814

theorem square_diagonal_double_area (d₁ : ℝ) (d₂ : ℝ) : 
  d₁ = 4 * Real.sqrt 2 → 
  d₂ * d₂ = 2 * (d₁ * d₁ / 2) → 
  d₂ = 8 := by
sorry

end NUMINAMATH_CALUDE_square_diagonal_double_area_l3238_323814


namespace NUMINAMATH_CALUDE_work_problem_solution_l3238_323898

def work_problem (a_days b_days remaining_days : ℚ) : Prop :=
  let a_rate : ℚ := 1 / a_days
  let b_rate : ℚ := 1 / b_days
  let combined_rate : ℚ := a_rate + b_rate
  let x : ℚ := 2  -- Days A and B worked together
  combined_rate * x + b_rate * remaining_days = 1

theorem work_problem_solution :
  work_problem 4 8 2 = true :=
sorry

end NUMINAMATH_CALUDE_work_problem_solution_l3238_323898


namespace NUMINAMATH_CALUDE_special_triangle_angles_l3238_323841

/-- A triangle with excircle radii and circumradius satisfying certain conditions -/
structure SpecialTriangle where
  /-- Excircle radius opposite to side a -/
  r_a : ℝ
  /-- Excircle radius opposite to side b -/
  r_b : ℝ
  /-- Excircle radius opposite to side c -/
  r_c : ℝ
  /-- Circumradius of the triangle -/
  R : ℝ
  /-- First condition: r_a + r_b = 3R -/
  cond1 : r_a + r_b = 3 * R
  /-- Second condition: r_b + r_c = 2R -/
  cond2 : r_b + r_c = 2 * R

/-- The angles of a SpecialTriangle are 30°, 60°, and 90° -/
theorem special_triangle_angles (t : SpecialTriangle) :
  ∃ (A B C : Real),
    A = 30 * π / 180 ∧
    B = 60 * π / 180 ∧
    C = 90 * π / 180 ∧
    A + B + C = π :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_angles_l3238_323841


namespace NUMINAMATH_CALUDE_annas_pencils_l3238_323833

theorem annas_pencils (anna_pencils : ℕ) (harry_pencils : ℕ) : 
  (harry_pencils = 2 * anna_pencils) → -- Harry has twice Anna's pencils initially
  (harry_pencils - 19 = 81) → -- Harry lost 19 pencils and now has 81 left
  anna_pencils = 50 := by
sorry

end NUMINAMATH_CALUDE_annas_pencils_l3238_323833


namespace NUMINAMATH_CALUDE_circle_symmetry_axis_l3238_323872

/-- Given a circle and a line that is its axis of symmetry, prove that the parameter a in the line equation equals 1 -/
theorem circle_symmetry_axis (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y - 3 = 0 → 
    (∃ c : ℝ, ∀ x' y' : ℝ, (x' - 2*a*y' - 3 = 0 ∧ 
      x'^2 + y'^2 - 2*x' + 2*y' - 3 = 0) ↔ 
      (2*c - x' - 2*a*y' - 3 = 0 ∧ 
       (2*c - x')^2 + y'^2 - 2*(2*c - x') + 2*y' - 3 = 0))) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_axis_l3238_323872


namespace NUMINAMATH_CALUDE_simplify_expression_l3238_323863

theorem simplify_expression (a b c : ℝ) : a - (b - c) = a - b + c := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3238_323863


namespace NUMINAMATH_CALUDE_water_percentage_in_container_l3238_323868

/-- Proves that the percentage of a container's capacity filled with 8 liters of water is 20%,
    given that the total capacity of 40 such containers is 1600 liters. -/
theorem water_percentage_in_container (container_capacity : ℝ) : 
  (40 * container_capacity = 1600) → (8 / container_capacity * 100 = 20) := by
  sorry

end NUMINAMATH_CALUDE_water_percentage_in_container_l3238_323868


namespace NUMINAMATH_CALUDE_pie_shop_revenue_calculation_l3238_323850

/-- Represents the revenue calculation for a pie shop --/
def pie_shop_revenue (apple_price blueberry_price cherry_price : ℕ) 
                     (slices_per_pie : ℕ) 
                     (apple_pies blueberry_pies cherry_pies : ℕ) : ℕ :=
  (apple_price * slices_per_pie * apple_pies) + 
  (blueberry_price * slices_per_pie * blueberry_pies) + 
  (cherry_price * slices_per_pie * cherry_pies)

/-- Theorem stating the revenue of the pie shop --/
theorem pie_shop_revenue_calculation : 
  pie_shop_revenue 5 6 7 6 12 8 10 = 1068 := by
  sorry

end NUMINAMATH_CALUDE_pie_shop_revenue_calculation_l3238_323850


namespace NUMINAMATH_CALUDE_bottle_cap_collection_l3238_323862

/-- Given that 7 bottle caps weigh one ounce and a collection of bottle caps weighs 18 pounds,
    prove that the number of bottle caps in the collection is 2016. -/
theorem bottle_cap_collection (caps_per_ounce : ℕ) (collection_weight_pounds : ℕ) 
  (h1 : caps_per_ounce = 7)
  (h2 : collection_weight_pounds = 18) :
  caps_per_ounce * (collection_weight_pounds * 16) = 2016 := by
  sorry

#check bottle_cap_collection

end NUMINAMATH_CALUDE_bottle_cap_collection_l3238_323862


namespace NUMINAMATH_CALUDE_average_and_difference_l3238_323865

theorem average_and_difference (y : ℝ) : 
  (45 + y) / 2 = 37 → |45 - y| = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l3238_323865


namespace NUMINAMATH_CALUDE_right_triangles_count_l3238_323859

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a line of points -/
def Line := List Point

/-- Creates a line of points with given y-coordinate -/
def createLine (y : ℕ) : Line :=
  List.map (fun x => ⟨x, y⟩) (List.range 73)

/-- Checks if three points form a right triangle -/
def isRightTriangle (p1 p2 p3 : Point) : Bool :=
  -- Implementation omitted for brevity
  sorry

/-- Counts the number of right triangles formed by points from two lines -/
def countRightTriangles (line1 line2 : Line) : ℕ :=
  -- Implementation omitted for brevity
  sorry

/-- The main theorem to prove -/
theorem right_triangles_count :
  let line1 := createLine 3
  let line2 := createLine 4
  countRightTriangles line1 line2 = 10654 := by
  sorry

end NUMINAMATH_CALUDE_right_triangles_count_l3238_323859


namespace NUMINAMATH_CALUDE_common_difference_is_five_l3238_323809

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_is_five 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum1 : a 2 + a 6 = 8) 
  (h_sum2 : a 3 + a 4 = 3) : 
  ∃ d : ℝ, d = 5 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_five_l3238_323809


namespace NUMINAMATH_CALUDE_molecular_weight_N2O3_l3238_323860

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms in N2O3
def N_count : ℕ := 2
def O_count : ℕ := 3

-- Define the number of moles
def moles : ℝ := 4

-- Theorem statement
theorem molecular_weight_N2O3 : 
  moles * (N_count * atomic_weight_N + O_count * atomic_weight_O) = 304.08 := by
  sorry


end NUMINAMATH_CALUDE_molecular_weight_N2O3_l3238_323860
