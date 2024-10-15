import Mathlib

namespace NUMINAMATH_CALUDE_unique_prime_digit_product_l2196_219628

def is_prime_digit (d : Nat) : Prop :=
  d ∈ [2, 3, 5, 7]

def all_prime_digits (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_prime_digit d

theorem unique_prime_digit_product : 
  ∃! (a b : Nat), 
    100 ≤ a ∧ a < 1000 ∧
    10 ≤ b ∧ b < 100 ∧
    all_prime_digits a ∧
    all_prime_digits b ∧
    1000 ≤ a * b ∧ a * b < 10000 ∧
    all_prime_digits (a * b) ∧
    a = 775 ∧ b = 33 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_digit_product_l2196_219628


namespace NUMINAMATH_CALUDE_roller_coaster_cost_l2196_219605

/-- The cost of a roller coaster ride in tickets, given the total number of tickets needed,
    the cost of a Ferris wheel ride, and the cost of a log ride. -/
theorem roller_coaster_cost
  (total_tickets : ℕ)
  (ferris_wheel_cost : ℕ)
  (log_ride_cost : ℕ)
  (h1 : total_tickets = 10)
  (h2 : ferris_wheel_cost = 2)
  (h3 : log_ride_cost = 1)
  : total_tickets - (ferris_wheel_cost + log_ride_cost) = 7 := by
  sorry

#check roller_coaster_cost

end NUMINAMATH_CALUDE_roller_coaster_cost_l2196_219605


namespace NUMINAMATH_CALUDE_linear_function_monotonicity_and_inequality_l2196_219668

variables (a b c : ℝ)

def f (x : ℝ) := a * x + b

theorem linear_function_monotonicity_and_inequality (a b c : ℝ) :
  (a > 0 → Monotone (f a b)) ∧
  (b^2 - 4*a*c < 0 → a^3 + a*b + c ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_linear_function_monotonicity_and_inequality_l2196_219668


namespace NUMINAMATH_CALUDE_star_sum_minus_emilio_sum_l2196_219608

def star_list : List Nat := List.range 50

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem star_sum_minus_emilio_sum : 
  star_list.sum - emilio_list.sum = 105 := by
  sorry

end NUMINAMATH_CALUDE_star_sum_minus_emilio_sum_l2196_219608


namespace NUMINAMATH_CALUDE_line_segment_intersection_range_l2196_219631

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := a * x - y - 2 * a - 1 = 0

-- Define the endpoints of the line segment
def point_A : ℝ × ℝ := (-2, 3)
def point_B : ℝ × ℝ := (5, 2)

-- Define the intersection condition
def intersects_segment (a : ℝ) : Prop :=
  ∃ (x y : ℝ), line_equation a x y ∧
    ((x - point_A.1) / (point_B.1 - point_A.1) =
     (y - point_A.2) / (point_B.2 - point_A.2)) ∧
    0 ≤ (x - point_A.1) / (point_B.1 - point_A.1) ∧
    (x - point_A.1) / (point_B.1 - point_A.1) ≤ 1

-- State the theorem
theorem line_segment_intersection_range :
  ∀ a : ℝ, intersects_segment a ↔ a ≤ -1 ∨ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_line_segment_intersection_range_l2196_219631


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_in_fraction_l2196_219649

theorem zeros_before_first_nonzero_digit_in_fraction :
  ∃ (n : ℕ) (d : ℚ), 
    (d = 7 / 800) ∧ 
    (∃ (m : ℕ), d * (10 ^ n) = m ∧ m ≥ 100 ∧ m < 1000) ∧
    n = 3 :=
by sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_in_fraction_l2196_219649


namespace NUMINAMATH_CALUDE_gymnastics_students_count_l2196_219614

/-- The position of a student in a rectangular formation. -/
structure Position where
  column_from_right : ℕ
  column_from_left : ℕ
  row_from_back : ℕ
  row_from_front : ℕ

/-- The gymnastics formation. -/
structure GymnasticsFormation where
  eunji_position : Position
  equal_students_per_row : Bool

/-- Calculate the total number of students in the gymnastics formation. -/
def total_students (formation : GymnasticsFormation) : ℕ :=
  let total_columns := formation.eunji_position.column_from_right +
                       formation.eunji_position.column_from_left - 1
  let total_rows := formation.eunji_position.row_from_back +
                    formation.eunji_position.row_from_front - 1
  total_columns * total_rows

/-- The main theorem stating the total number of students in the given formation. -/
theorem gymnastics_students_count :
  ∀ (formation : GymnasticsFormation),
    formation.eunji_position.column_from_right = 8 →
    formation.eunji_position.column_from_left = 14 →
    formation.eunji_position.row_from_back = 7 →
    formation.eunji_position.row_from_front = 15 →
    formation.equal_students_per_row = true →
    total_students formation = 441 := by
  sorry

end NUMINAMATH_CALUDE_gymnastics_students_count_l2196_219614


namespace NUMINAMATH_CALUDE_colored_plane_theorem_l2196_219671

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point in the plane
def colorAssignment : Point → Color := sorry

-- Define what it means for three points to form an equilateral triangle
def isEquilateralTriangle (A B C : Point) : Prop := sorry

-- Define what it means for a point to be the midpoint of two other points
def isMidpoint (M A C : Point) : Prop := sorry

theorem colored_plane_theorem :
  -- Part (a)
  (¬ ∃ A B C : Point, isEquilateralTriangle A B C ∧ colorAssignment A = colorAssignment B ∧ colorAssignment B = colorAssignment C →
   ∃ A B C : Point, colorAssignment A = colorAssignment B ∧ colorAssignment B = colorAssignment C ∧ isMidpoint B A C) ∧
  -- Part (b)
  ∃ A B C : Point, isEquilateralTriangle A B C ∧ colorAssignment A = colorAssignment B ∧ colorAssignment B = colorAssignment C :=
by sorry

end NUMINAMATH_CALUDE_colored_plane_theorem_l2196_219671


namespace NUMINAMATH_CALUDE_expected_sides_after_cutting_l2196_219603

/-- The expected number of sides of a randomly picked polygon after cutting -/
def expected_sides (n : ℕ) : ℚ :=
  (n + 7200 : ℚ) / 3601

/-- Theorem stating the expected number of sides after cutting an n-sided polygon for 3600 seconds -/
theorem expected_sides_after_cutting (n : ℕ) :
  let initial_sides := n
  let num_cuts := 3600
  let total_sides := initial_sides + 2 * num_cuts
  let num_polygons := num_cuts + 1
  expected_sides n = total_sides / num_polygons :=
by
  sorry

#eval expected_sides 3  -- For a triangle
#eval expected_sides 4  -- For a quadrilateral

end NUMINAMATH_CALUDE_expected_sides_after_cutting_l2196_219603


namespace NUMINAMATH_CALUDE_nancy_chip_distribution_l2196_219690

/-- The number of tortilla chips Nancy initially had -/
def initial_chips : ℕ := 22

/-- The number of tortilla chips Nancy gave to her brother -/
def brother_chips : ℕ := 7

/-- The number of tortilla chips Nancy kept for herself -/
def nancy_chips : ℕ := 10

/-- The number of tortilla chips Nancy gave to her sister -/
def sister_chips : ℕ := initial_chips - brother_chips - nancy_chips

theorem nancy_chip_distribution :
  sister_chips = 5 := by sorry

end NUMINAMATH_CALUDE_nancy_chip_distribution_l2196_219690


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_after_2023_seconds_l2196_219626

/-- Represents the number of positions in the cat's path -/
def cat_path_length : ℕ := 8

/-- Represents the number of positions in the mouse's path -/
def mouse_path_length : ℕ := 12

/-- Calculates the position of an object after a given number of seconds,
    given the length of its path -/
def position_after_time (path_length : ℕ) (time : ℕ) : ℕ :=
  time % path_length

/-- The main theorem stating the positions of the cat and mouse after 2023 seconds -/
theorem cat_and_mouse_positions_after_2023_seconds :
  position_after_time cat_path_length 2023 = 7 ∧
  position_after_time mouse_path_length 2023 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cat_and_mouse_positions_after_2023_seconds_l2196_219626


namespace NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l2196_219617

-- First expression
theorem simplify_first_expression (a b : ℝ) :
  6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1/2) * a * b) = -a * b := by
  sorry

-- Second expression
theorem simplify_second_expression (t : ℝ) :
  -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l2196_219617


namespace NUMINAMATH_CALUDE_volume_T_coefficients_qr_ps_ratio_l2196_219683

/-- A right rectangular prism with edge lengths 2, 4, and 6 units -/
structure RectangularPrism where
  length : ℝ := 2
  width : ℝ := 4
  height : ℝ := 6

/-- The set of points within distance r from any point in the prism -/
def T (C : RectangularPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume function of T(r) -/
def volume_T (C : RectangularPrism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume function -/
structure VolumeCoefficients where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ

theorem volume_T_coefficients (C : RectangularPrism) :
  ∃ (coeff : VolumeCoefficients),
    ∀ r : ℝ, volume_T C r = coeff.P * r^3 + coeff.Q * r^2 + coeff.R * r + coeff.S :=
  sorry

theorem qr_ps_ratio (C : RectangularPrism) (coeff : VolumeCoefficients)
    (h : ∀ r : ℝ, volume_T C r = coeff.P * r^3 + coeff.Q * r^2 + coeff.R * r + coeff.S) :
    coeff.Q * coeff.R / (coeff.P * coeff.S) = 16.5 :=
  sorry

end NUMINAMATH_CALUDE_volume_T_coefficients_qr_ps_ratio_l2196_219683


namespace NUMINAMATH_CALUDE_angle_triple_supplement_measure_l2196_219634

theorem angle_triple_supplement_measure : 
  ∀ x : ℝ, (x = 3 * (180 - x)) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_measure_l2196_219634


namespace NUMINAMATH_CALUDE_roots_real_implies_ab_nonpositive_l2196_219676

/-- The polynomial x^4 + ax^3 + bx + c has all real roots -/
def has_all_real_roots (a b c : ℝ) : Prop :=
  ∀ x : ℂ, x^4 + a*x^3 + b*x + c = 0 → x.im = 0

/-- If all roots of the polynomial x^4 + ax^3 + bx + c are real numbers, then ab ≤ 0 -/
theorem roots_real_implies_ab_nonpositive (a b c : ℝ) :
  has_all_real_roots a b c → a * b ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_roots_real_implies_ab_nonpositive_l2196_219676


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2196_219679

theorem two_numbers_problem :
  ∃! (a b : ℕ), a > b ∧ (a / b : ℚ) * 6 = 10 ∧ (a - b : ℤ) + 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2196_219679


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l2196_219633

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | x^2 + 2*x = 0}
def T : Set ℝ := {x : ℝ | x^2 - 2*x = 0}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l2196_219633


namespace NUMINAMATH_CALUDE_a_4_equals_zero_l2196_219657

def sequence_a (n : ℕ) : ℤ := n^2 - 2*n - 8

theorem a_4_equals_zero : sequence_a 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_zero_l2196_219657


namespace NUMINAMATH_CALUDE_chicken_count_l2196_219682

theorem chicken_count (east : ℕ) (west : ℕ) : 
  east = 40 → 
  (east : ℚ) + west * (1 - 1/4 - 1/3) = (1/2 : ℚ) * (east + west) → 
  east + west = 280 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l2196_219682


namespace NUMINAMATH_CALUDE_water_bottles_fourth_game_l2196_219600

/-- Represents the number of bottles in a case -/
structure CaseSize where
  water : ℕ
  sports_drink : ℕ

/-- Represents the number of cases purchased -/
structure CasesPurchased where
  water : ℕ
  sports_drink : ℕ

/-- Represents the consumption of bottles in a game -/
structure GameConsumption where
  water : ℕ
  sports_drink : ℕ

/-- Calculates the total number of bottles initially available -/
def totalInitialBottles (caseSize : CaseSize) (casesPurchased : CasesPurchased) : ℕ × ℕ :=
  (caseSize.water * casesPurchased.water, caseSize.sports_drink * casesPurchased.sports_drink)

/-- Calculates the total consumption for the first three games -/
def totalConsumptionFirstThreeGames (game1 game2 game3 : GameConsumption) : ℕ × ℕ :=
  (game1.water + game2.water + game3.water, game1.sports_drink + game2.sports_drink + game3.sports_drink)

/-- Theorem: The number of water bottles used in the fourth game is 20 -/
theorem water_bottles_fourth_game 
  (caseSize : CaseSize)
  (casesPurchased : CasesPurchased)
  (game1 game2 game3 : GameConsumption)
  (remainingBottles : ℕ × ℕ) :
  caseSize.water = 20 →
  caseSize.sports_drink = 15 →
  casesPurchased.water = 10 →
  casesPurchased.sports_drink = 5 →
  game1 = { water := 70, sports_drink := 30 } →
  game2 = { water := 40, sports_drink := 20 } →
  game3 = { water := 50, sports_drink := 25 } →
  remainingBottles = (20, 10) →
  let (initialWater, _) := totalInitialBottles caseSize casesPurchased
  let (consumedWater, _) := totalConsumptionFirstThreeGames game1 game2 game3
  initialWater - consumedWater - remainingBottles.1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_fourth_game_l2196_219600


namespace NUMINAMATH_CALUDE_min_price_theorem_l2196_219669

/-- Represents the manufacturing scenario with two components -/
structure ManufacturingScenario where
  prod_cost_A : ℝ  -- Production cost for component A
  ship_cost_A : ℝ  -- Shipping cost for component A
  prod_cost_B : ℝ  -- Production cost for component B
  ship_cost_B : ℝ  -- Shipping cost for component B
  fixed_costs : ℝ  -- Fixed costs per month
  units_A : ℕ      -- Number of units of component A produced and sold
  units_B : ℕ      -- Number of units of component B produced and sold

/-- Calculates the total cost for the given manufacturing scenario -/
def total_cost (s : ManufacturingScenario) : ℝ :=
  s.fixed_costs +
  s.units_A * (s.prod_cost_A + s.ship_cost_A) +
  s.units_B * (s.prod_cost_B + s.ship_cost_B)

/-- Theorem: The minimum price per unit that ensures total revenue is at least equal to total costs is $103 -/
theorem min_price_theorem (s : ManufacturingScenario)
  (h1 : s.prod_cost_A = 80)
  (h2 : s.ship_cost_A = 2)
  (h3 : s.prod_cost_B = 60)
  (h4 : s.ship_cost_B = 3)
  (h5 : s.fixed_costs = 16200)
  (h6 : s.units_A = 200)
  (h7 : s.units_B = 300) :
  ∃ (P : ℝ), P ≥ 103 ∧ P * (s.units_A + s.units_B) ≥ total_cost s ∧
  ∀ (Q : ℝ), Q * (s.units_A + s.units_B) ≥ total_cost s → Q ≥ P :=
sorry


end NUMINAMATH_CALUDE_min_price_theorem_l2196_219669


namespace NUMINAMATH_CALUDE_circle_reflection_minimum_l2196_219641

/-- Given a circle and a line, if reflection about the line keeps points on the circle,
    then there's a minimum value for a certain expression involving the line's parameters. -/
theorem circle_reflection_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 → 
   ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 1 = 0 ∧ 
              ((x + x')/2, (y + y')/2) ∈ {(x, y) | 2*a*x - b*y + 2 = 0}) →
  (∃ m : ℝ, m = 1/a + 2/b ∧ ∀ k : ℝ, k = 1/a + 2/b → m ≤ k) →
  1/a + 2/b = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_circle_reflection_minimum_l2196_219641


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2196_219635

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with S₁₀ = 5 and a₇ = 1, a₁ = -1 -/
theorem arithmetic_sequence_first_term
  (seq : ArithmeticSequence)
  (h1 : seq.S 10 = 5)
  (h2 : seq.a 7 = 1) :
  seq.a 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2196_219635


namespace NUMINAMATH_CALUDE_candy_distribution_l2196_219625

theorem candy_distribution (total_candy : ℕ) (candy_per_bag : ℕ) (h1 : total_candy = 648) (h2 : candy_per_bag = 81) :
  total_candy / candy_per_bag = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2196_219625


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2196_219672

/-- The distance between the vertices of a hyperbola with equation x^2/16 - y^2/25 = 1 is 8 -/
theorem hyperbola_vertex_distance : 
  ∀ (x y : ℝ), x^2/16 - y^2/25 = 1 → ∃ (v₁ v₂ : ℝ × ℝ), 
    (v₁.1^2/16 - v₁.2^2/25 = 1) ∧ 
    (v₂.1^2/16 - v₂.2^2/25 = 1) ∧ 
    (v₁.2 = 0) ∧ (v₂.2 = 0) ∧
    (v₁.1 + v₂.1 = 0) ∧
    (|v₁.1 - v₂.1| = 8) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2196_219672


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l2196_219681

/-- The sum of the digits of 10^93 - 93 -/
def sum_of_digits : ℕ := 826

/-- The number represented by 10^93 - 93 -/
def large_number : ℕ := 10^93 - 93

/-- Function to calculate the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem sum_of_digits_of_large_number :
  digit_sum large_number = sum_of_digits :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l2196_219681


namespace NUMINAMATH_CALUDE_sum_of_integers_l2196_219699

theorem sum_of_integers : 7 + (-19) + 13 + (-31) = -30 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2196_219699


namespace NUMINAMATH_CALUDE_pacific_ocean_area_rounded_l2196_219673

/-- Rounds a number to the nearest multiple of 10000 -/
def roundToNearestTenThousand (n : ℕ) : ℕ :=
  ((n + 5000) / 10000) * 10000

theorem pacific_ocean_area_rounded :
  roundToNearestTenThousand 17996800 = 18000000 := by sorry

end NUMINAMATH_CALUDE_pacific_ocean_area_rounded_l2196_219673


namespace NUMINAMATH_CALUDE_three_hundred_percent_of_forty_l2196_219638

-- Define 300 percent as 3 in decimal form
def three_hundred_percent : ℝ := 3

-- Define the operation of taking a percentage of a number
def percentage_of (percent : ℝ) (number : ℝ) : ℝ := percent * number

-- Theorem statement
theorem three_hundred_percent_of_forty :
  percentage_of three_hundred_percent 40 = 120 := by
  sorry

end NUMINAMATH_CALUDE_three_hundred_percent_of_forty_l2196_219638


namespace NUMINAMATH_CALUDE_belts_count_l2196_219695

/-- The number of ties in the store -/
def ties : ℕ := 34

/-- The number of black shirts in the store -/
def black_shirts : ℕ := 63

/-- The number of white shirts in the store -/
def white_shirts : ℕ := 42

/-- The number of jeans in the store -/
def jeans : ℕ := (2 * (black_shirts + white_shirts)) / 3

/-- The number of scarves in the store -/
def scarves (belts : ℕ) : ℕ := (ties + belts) / 2

/-- The relationship between jeans and scarves -/
def jeans_scarves_relation (belts : ℕ) : Prop :=
  jeans = scarves belts + 33

theorem belts_count : ∃ (belts : ℕ), jeans_scarves_relation belts ∧ belts = 40 :=
sorry

end NUMINAMATH_CALUDE_belts_count_l2196_219695


namespace NUMINAMATH_CALUDE_meghan_coffee_order_cost_l2196_219650

/-- Represents the cost of a coffee order with given quantities and prices --/
def coffee_order_cost (drip_coffee_price : ℚ) (drip_coffee_qty : ℕ)
                      (espresso_price : ℚ) (espresso_qty : ℕ)
                      (latte_price : ℚ) (latte_qty : ℕ)
                      (vanilla_syrup_price : ℚ) (vanilla_syrup_qty : ℕ)
                      (cold_brew_price : ℚ) (cold_brew_qty : ℕ)
                      (cappuccino_price : ℚ) (cappuccino_qty : ℕ) : ℚ :=
  drip_coffee_price * drip_coffee_qty +
  espresso_price * espresso_qty +
  latte_price * latte_qty +
  vanilla_syrup_price * vanilla_syrup_qty +
  cold_brew_price * cold_brew_qty +
  cappuccino_price * cappuccino_qty

/-- The total cost of Meghan's coffee order is $25.00 --/
theorem meghan_coffee_order_cost :
  coffee_order_cost (25/10) 2
                    (35/10) 1
                    4 2
                    (1/2) 1
                    (25/10) 2
                    (35/10) 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_meghan_coffee_order_cost_l2196_219650


namespace NUMINAMATH_CALUDE_abs_value_properties_l2196_219621

-- Define the absolute value function
def f (x : ℝ) := abs x

-- State the theorem
theorem abs_value_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_abs_value_properties_l2196_219621


namespace NUMINAMATH_CALUDE_triangle_area_with_base_12_height_15_l2196_219610

/-- The area of a triangle with base 12 and height 15 is 90 -/
theorem triangle_area_with_base_12_height_15 :
  let base : ℝ := 12
  let height : ℝ := 15
  let area : ℝ := (1 / 2) * base * height
  area = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_base_12_height_15_l2196_219610


namespace NUMINAMATH_CALUDE_total_shirts_produced_l2196_219615

/-- Represents the number of shirts produced per minute -/
def shirts_per_minute : ℕ := 6

/-- Represents the number of minutes the machine operates -/
def operation_time : ℕ := 6

/-- Theorem stating that the total number of shirts produced is 36 -/
theorem total_shirts_produced :
  shirts_per_minute * operation_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_shirts_produced_l2196_219615


namespace NUMINAMATH_CALUDE_lemonade_proportion_l2196_219666

theorem lemonade_proportion (lemons_small : ℕ) (gallons_small : ℕ) (gallons_large : ℕ) :
  lemons_small = 36 →
  gallons_small = 48 →
  gallons_large = 100 →
  (lemons_small * gallons_large) / gallons_small = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_lemonade_proportion_l2196_219666


namespace NUMINAMATH_CALUDE_steve_final_marbles_l2196_219629

/-- Represents the initial and final marble counts for each person --/
structure MarbleCounts where
  steve_initial : ℕ
  sam_initial : ℕ
  sally_initial : ℕ
  sarah_initial : ℕ
  steve_final : ℕ

/-- Defines the marble distribution scenario --/
def marble_distribution (counts : MarbleCounts) : Prop :=
  counts.sam_initial = 2 * counts.steve_initial ∧
  counts.sally_initial = counts.sam_initial - 5 ∧
  counts.sarah_initial = counts.steve_initial + 3 ∧
  counts.sam_initial - (3 + 3 + 4) = 6 ∧
  counts.steve_final = counts.steve_initial + 3

theorem steve_final_marbles (counts : MarbleCounts) :
  marble_distribution counts → counts.steve_final = 11 := by
  sorry

end NUMINAMATH_CALUDE_steve_final_marbles_l2196_219629


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2196_219667

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age_diff : ℕ) 
  (remaining_players_age_diff : ℕ) : 
  team_size = 11 → 
  captain_age = 28 → 
  wicket_keeper_age_diff = 3 → 
  remaining_players_age_diff = 1 → 
  ∃ (team_avg_age : ℚ), 
    team_avg_age = 25 ∧ 
    team_size * team_avg_age = 
      captain_age + (captain_age + wicket_keeper_age_diff) + 
      (team_size - 2) * (team_avg_age - remaining_players_age_diff) :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l2196_219667


namespace NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l2196_219660

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem ten_factorial_mod_thirteen : factorial 10 % 13 = 6 := by sorry

end NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l2196_219660


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2196_219647

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 where a > b > 0,
    and points A(-a,0), B(a,0), M(x,y), N(x,-y) on C,
    prove that if the product of slopes of AM and BN is 4/9,
    then the eccentricity of C is √5/3 -/
theorem ellipse_eccentricity (a b : ℝ) (x y : ℝ) :
  a > b → b > 0 →
  x^2 / a^2 + y^2 / b^2 = 1 →
  (y / (x + a)) * (-y / (x - a)) = 4/9 →
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2196_219647


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l2196_219622

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 2 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1/2

-- Main theorem
theorem ellipse_line_intersection :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  midpoint_condition A B →
  (∀ x y : ℝ, line x y ↔ x + 2*y - 2 = 0) ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l2196_219622


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_segment_ratio_l2196_219651

theorem right_triangle_hypotenuse_segment_ratio 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_ratio : a / b = 3 / 4) 
  (d : ℝ) 
  (h_d : d * c = a * b) : 
  (c - d) / d = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_segment_ratio_l2196_219651


namespace NUMINAMATH_CALUDE_ben_win_probability_l2196_219607

/-- The probability of Ben winning a game, given the probability of losing and that tying is impossible -/
theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5/7) (h2 : lose_prob + win_prob = 1) : win_prob = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ben_win_probability_l2196_219607


namespace NUMINAMATH_CALUDE_longest_watching_time_l2196_219678

structure Show where
  episodes : ℕ
  minutesPerEpisode : ℕ
  speed : ℚ

def watchingTimePerDay (s : Show) (days : ℕ) : ℚ :=
  (s.episodes * s.minutesPerEpisode : ℚ) / (s.speed * (days * 60))

theorem longest_watching_time (showA showB showC : Show) (days : ℕ) :
  showA.episodes = 20 ∧ 
  showA.minutesPerEpisode = 30 ∧ 
  showA.speed = 1.2 ∧
  showB.episodes = 25 ∧ 
  showB.minutesPerEpisode = 45 ∧ 
  showB.speed = 1 ∧
  showC.episodes = 30 ∧ 
  showC.minutesPerEpisode = 40 ∧ 
  showC.speed = 0.9 ∧
  days = 5 →
  watchingTimePerDay showC days > watchingTimePerDay showA days ∧
  watchingTimePerDay showC days > watchingTimePerDay showB days :=
sorry

end NUMINAMATH_CALUDE_longest_watching_time_l2196_219678


namespace NUMINAMATH_CALUDE_percentage_theorem_l2196_219639

theorem percentage_theorem (x y : ℝ) (P : ℝ) 
  (h1 : 0.6 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.5 * x) : 
  P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_theorem_l2196_219639


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l2196_219654

/-- For an ellipse with equation x²/m + y²/4 = 1 and focal distance 2, m = 5 -/
theorem ellipse_focal_distance (m : ℝ) : 
  (∀ x y : ℝ, x^2/m + y^2/4 = 1) →  -- Ellipse equation
  2 = 2 * 1 →                      -- Focal distance is 2
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_distance_l2196_219654


namespace NUMINAMATH_CALUDE_cos_double_alpha_l2196_219612

theorem cos_double_alpha (α : ℝ) : 
  (Real.cos α)^2 + (Real.sqrt 2 / 2)^2 = (Real.sqrt 3 / 2)^2 → 
  Real.cos (2 * α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_alpha_l2196_219612


namespace NUMINAMATH_CALUDE_inverse_z_minus_z_inv_l2196_219697

/-- Given a complex number z = 1 + i where i² = -1, prove that (z - z⁻¹)⁻¹ = (1 - 3i) / 5 -/
theorem inverse_z_minus_z_inv (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := 1 + i
  (z - z⁻¹)⁻¹ = (1 - 3*i) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_z_minus_z_inv_l2196_219697


namespace NUMINAMATH_CALUDE_correct_linear_regression_l2196_219696

-- Define the variables and constants
variable (x y : ℝ)
def x_mean : ℝ := 2.5
def y_mean : ℝ := 3.5

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := 0.4 * x + 2.5

-- State the theorem
theorem correct_linear_regression :
  (∃ r : ℝ, r > 0 ∧ (∀ x y : ℝ, y - y_mean = r * (x - x_mean))) →  -- Positive correlation
  (linear_regression x_mean = y_mean) →                           -- Passes through (x̄, ȳ)
  (∀ x : ℝ, linear_regression x = 0.4 * x + 2.5) :=               -- The equation is correct
by sorry

end NUMINAMATH_CALUDE_correct_linear_regression_l2196_219696


namespace NUMINAMATH_CALUDE_storm_damage_conversion_l2196_219692

/-- Converts Canadian dollars to American dollars given exchange rates -/
def storm_damage_in_usd (damage_cad : ℝ) (cad_to_eur : ℝ) (eur_to_usd : ℝ) : ℝ :=
  damage_cad * cad_to_eur * eur_to_usd

/-- Theorem: The storm damage in USD is 40.5 million given the conditions -/
theorem storm_damage_conversion :
  storm_damage_in_usd 45000000 0.75 1.2 = 40500000 := by
  sorry

end NUMINAMATH_CALUDE_storm_damage_conversion_l2196_219692


namespace NUMINAMATH_CALUDE_least_common_period_is_30_l2196_219664

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least positive period of a function -/
def is_least_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_period f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬ is_period f q

/-- The main theorem -/
theorem least_common_period_is_30 :
  ∃ p : ℝ, p = 30 ∧
    (∀ f : ℝ → ℝ, satisfies_condition f → is_least_positive_period f p) ∧
    (∀ q : ℝ, q ≠ p →
      ∃ f : ℝ → ℝ, satisfies_condition f ∧ ¬ is_least_positive_period f q) :=
sorry

end NUMINAMATH_CALUDE_least_common_period_is_30_l2196_219664


namespace NUMINAMATH_CALUDE_min_value_inequality_l2196_219630

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 9) : 
  (x^2 + y^2 + 1) / (x + y) + (x^2 + z^2 + 1) / (x + z) + (y^2 + z^2 + 1) / (y + z) ≥ 4.833 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2196_219630


namespace NUMINAMATH_CALUDE_flower_shop_optimal_strategy_l2196_219619

/-- Represents the flower shop's sales and profit model -/
structure FlowerShop where
  cost : ℝ := 50
  max_margin : ℝ := 0.52
  sales : ℝ → ℝ
  profit : ℝ → ℝ
  profit_after_donation : ℝ → ℝ → ℝ

/-- The main theorem about the flower shop's optimal pricing and donation strategy -/
theorem flower_shop_optimal_strategy (shop : FlowerShop) 
  (h_sales : ∀ x, shop.sales x = -6 * x + 600) 
  (h_profit : ∀ x, shop.profit x = (x - shop.cost) * shop.sales x) 
  (h_profit_donation : ∀ x n, shop.profit_after_donation x n = shop.profit x - n * shop.sales x) 
  (h_price_range : ∀ x, x ≥ shop.cost ∧ x ≤ shop.cost * (1 + shop.max_margin)) :
  (∃ max_profit : ℝ, max_profit = 3750 ∧ 
    ∀ x, shop.profit x ≤ max_profit ∧ 
    (shop.profit 75 = max_profit)) ∧
  (∀ n, (∀ x₁ x₂, x₁ < x₂ → shop.profit_after_donation x₁ n < shop.profit_after_donation x₂ n) 
    ↔ (1 < n ∧ n < 2)) :=
sorry

end NUMINAMATH_CALUDE_flower_shop_optimal_strategy_l2196_219619


namespace NUMINAMATH_CALUDE_simplified_fraction_ratio_l2196_219662

theorem simplified_fraction_ratio (k : ℤ) : 
  ∃ (a b : ℤ), (6 * k + 18) / 3 = a * k + b ∧ a / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_ratio_l2196_219662


namespace NUMINAMATH_CALUDE_expression_evaluation_l2196_219643

theorem expression_evaluation :
  let x : ℚ := 2
  (x^2 + 2*x + 1) / (x^2 - 1) / ((x / (x - 1)) - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2196_219643


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l2196_219656

theorem complex_magnitude_theorem : 
  let i : ℂ := Complex.I
  let T : ℂ := 3 * ((1 + i)^15 - (1 - i)^15)
  Complex.abs T = 768 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l2196_219656


namespace NUMINAMATH_CALUDE_probability_receive_one_l2196_219677

/-- Probability of receiving a signal as 1 in a digital communication system with given error rates --/
theorem probability_receive_one (p_receive_zero_given_send_zero : ℝ)
                                (p_receive_one_given_send_zero : ℝ)
                                (p_receive_one_given_send_one : ℝ)
                                (p_receive_zero_given_send_one : ℝ)
                                (p_send_zero : ℝ)
                                (p_send_one : ℝ)
                                (h1 : p_receive_zero_given_send_zero = 0.9)
                                (h2 : p_receive_one_given_send_zero = 0.1)
                                (h3 : p_receive_one_given_send_one = 0.95)
                                (h4 : p_receive_zero_given_send_one = 0.05)
                                (h5 : p_send_zero = 0.5)
                                (h6 : p_send_one = 0.5) :
  p_send_zero * p_receive_one_given_send_zero + p_send_one * p_receive_one_given_send_one = 0.525 :=
by sorry

end NUMINAMATH_CALUDE_probability_receive_one_l2196_219677


namespace NUMINAMATH_CALUDE_lesser_number_problem_l2196_219616

theorem lesser_number_problem (x y : ℝ) 
  (sum_eq : x + y = 60) 
  (diff_eq : 4 * y - x = 10) : 
  y = 14 := by sorry

end NUMINAMATH_CALUDE_lesser_number_problem_l2196_219616


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l2196_219611

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an additional innings -/
def newAverage (bp : BatsmanPerformance) (newScore : ℕ) : ℚ :=
  (bp.totalScore + newScore) / (bp.innings + 1)

theorem batsman_average_after_17th_innings 
  (bp : BatsmanPerformance) 
  (h1 : bp.innings = 16) 
  (h2 : newAverage bp 85 = bp.average + 3) :
  newAverage bp 85 = 37 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l2196_219611


namespace NUMINAMATH_CALUDE_statue_weight_proof_l2196_219687

/-- Given a set of statues carved from a marble block, prove the weight of each remaining statue. -/
theorem statue_weight_proof (initial_weight discarded_weight first_statue_weight second_statue_weight : ℝ)
  (h1 : initial_weight = 80)
  (h2 : discarded_weight = 22)
  (h3 : first_statue_weight = 10)
  (h4 : second_statue_weight = 18)
  (h5 : initial_weight ≥ first_statue_weight + second_statue_weight + discarded_weight) :
  let remaining_weight := initial_weight - first_statue_weight - second_statue_weight - discarded_weight
  (remaining_weight / 2 : ℝ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_proof_l2196_219687


namespace NUMINAMATH_CALUDE_beyonce_songs_count_l2196_219613

/-- The number of songs Beyonce has released in total -/
def total_songs : ℕ :=
  let singles := 12
  let albums := 4
  let songs_first_cd := 18
  let songs_second_cd := 14
  let songs_per_album := songs_first_cd + songs_second_cd
  singles + albums * songs_per_album

/-- Theorem stating that Beyonce has released 140 songs in total -/
theorem beyonce_songs_count : total_songs = 140 := by
  sorry

end NUMINAMATH_CALUDE_beyonce_songs_count_l2196_219613


namespace NUMINAMATH_CALUDE_solve_for_a_l2196_219685

theorem solve_for_a : ∃ a : ℝ, (2 - 3 * (a + 1) = 2 * 1) ∧ (a = -1) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2196_219685


namespace NUMINAMATH_CALUDE_marks_buttons_l2196_219691

theorem marks_buttons (x : ℕ) : 
  (x + 3*x) / 2 = 28 → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_marks_buttons_l2196_219691


namespace NUMINAMATH_CALUDE_discount_profit_calculation_l2196_219686

/-- Calculates the profit percentage with discount given the discount rate and profit without discount -/
def profit_with_discount (discount : ℝ) (profit_without_discount : ℝ) : ℝ :=
  let marked_price := 1 + profit_without_discount
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem stating that with a 5% discount and 30% profit without discount, 
    the profit with discount is 23.5% -/
theorem discount_profit_calculation :
  profit_with_discount 0.05 0.30 = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_discount_profit_calculation_l2196_219686


namespace NUMINAMATH_CALUDE_floor_ceiling_calculation_l2196_219644

theorem floor_ceiling_calculation : 
  ⌊(15 : ℝ) / 8 * (-34 : ℝ) / 4⌋ - ⌈(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌉ = 0 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_calculation_l2196_219644


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_l2196_219609

/-- Conversion from spherical coordinates to rectangular coordinates -/
theorem spherical_to_rectangular (ρ θ φ : ℝ) :
  ρ = 5 ∧ θ = π/4 ∧ φ = π/3 →
  (ρ * Real.sin φ * Real.cos θ = 5 * Real.sqrt 6 / 4) ∧
  (ρ * Real.sin φ * Real.sin θ = 5 * Real.sqrt 6 / 4) ∧
  (ρ * Real.cos φ = 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_l2196_219609


namespace NUMINAMATH_CALUDE_residue_of_negative_1237_mod_29_l2196_219698

theorem residue_of_negative_1237_mod_29 : Int.mod (-1237) 29 = 10 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_negative_1237_mod_29_l2196_219698


namespace NUMINAMATH_CALUDE_square_coloring_l2196_219655

/-- The number of triangles in the square -/
def n : ℕ := 18

/-- The number of triangles to be colored -/
def k : ℕ := 6

/-- Binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem square_coloring :
  binomial n k = 18564 := by
  sorry

end NUMINAMATH_CALUDE_square_coloring_l2196_219655


namespace NUMINAMATH_CALUDE_units_digit_of_31_cubed_plus_13_cubed_l2196_219689

theorem units_digit_of_31_cubed_plus_13_cubed : ∃ n : ℕ, 31^3 + 13^3 = 10 * n + 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_31_cubed_plus_13_cubed_l2196_219689


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2196_219658

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 11 ∧ 
  (∀ (y : ℕ), (2000 - y : ℤ) % 17 = 0 → y ≥ x) ∧ 
  (2000 - x : ℤ) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2196_219658


namespace NUMINAMATH_CALUDE_cans_difference_l2196_219663

/-- The number of cans Sarah collected yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of additional cans Lara collected compared to Sarah yesterday -/
def lara_extra_yesterday : ℕ := 30

/-- The number of cans Sarah collected today -/
def sarah_today : ℕ := 40

/-- The number of cans Lara collected today -/
def lara_today : ℕ := 70

/-- Theorem stating the difference in total cans collected between yesterday and today -/
theorem cans_difference : 
  (sarah_yesterday + (sarah_yesterday + lara_extra_yesterday)) - (sarah_today + lara_today) = 20 :=
by sorry

end NUMINAMATH_CALUDE_cans_difference_l2196_219663


namespace NUMINAMATH_CALUDE_negative_one_greater_than_negative_two_l2196_219684

theorem negative_one_greater_than_negative_two : -1 > -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_greater_than_negative_two_l2196_219684


namespace NUMINAMATH_CALUDE_cupcakes_eaten_correct_l2196_219645

/-- Calculates the number of cupcakes Todd ate given the initial number of cupcakes,
    the number of packages, and the number of cupcakes per package. -/
def cupcakes_eaten (initial : ℕ) (packages : ℕ) (per_package : ℕ) : ℕ :=
  initial - (packages * per_package)

/-- Proves that the number of cupcakes Todd ate is correct -/
theorem cupcakes_eaten_correct (initial : ℕ) (packages : ℕ) (per_package : ℕ) :
  cupcakes_eaten initial packages per_package = initial - (packages * per_package) :=
by
  sorry

#eval cupcakes_eaten 39 6 3  -- Should evaluate to 21

end NUMINAMATH_CALUDE_cupcakes_eaten_correct_l2196_219645


namespace NUMINAMATH_CALUDE_sum_of_p_and_q_l2196_219688

theorem sum_of_p_and_q (p q : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ y : ℝ, 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 14)) →
  p + q = 69 := by
sorry

end NUMINAMATH_CALUDE_sum_of_p_and_q_l2196_219688


namespace NUMINAMATH_CALUDE_bell_size_ratio_l2196_219642

theorem bell_size_ratio (first_bell : ℝ) (second_bell : ℝ) (third_bell : ℝ) 
  (h1 : first_bell = 50)
  (h2 : third_bell = 4 * second_bell)
  (h3 : first_bell + second_bell + third_bell = 550) :
  second_bell / first_bell = 2 := by
sorry

end NUMINAMATH_CALUDE_bell_size_ratio_l2196_219642


namespace NUMINAMATH_CALUDE_third_consecutive_odd_integer_l2196_219661

/-- Given three consecutive odd integers where 3 times the first is 3 more than twice the third, 
    prove that the third integer is 15. -/
theorem third_consecutive_odd_integer (x : ℤ) : 
  (∃ y z : ℤ, 
    y = x + 2 ∧ 
    z = x + 4 ∧ 
    Odd x ∧ 
    Odd y ∧ 
    Odd z ∧ 
    3 * x = 2 * z + 3) →
  x + 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_third_consecutive_odd_integer_l2196_219661


namespace NUMINAMATH_CALUDE_three_cakes_cooking_time_l2196_219674

/-- Represents the cooking process for cakes -/
structure CookingProcess where
  pot_capacity : ℕ
  cooking_time_per_cake : ℕ
  num_cakes : ℕ

/-- Calculates the minimum time needed to cook all cakes -/
def min_cooking_time (process : CookingProcess) : ℕ :=
  sorry

/-- Theorem stating the minimum time to cook 3 cakes -/
theorem three_cakes_cooking_time :
  ∀ (process : CookingProcess),
    process.pot_capacity = 2 →
    process.cooking_time_per_cake = 5 →
    process.num_cakes = 3 →
    min_cooking_time process = 15 :=
by sorry

end NUMINAMATH_CALUDE_three_cakes_cooking_time_l2196_219674


namespace NUMINAMATH_CALUDE_bodies_of_revolution_l2196_219652

-- Define the type for geometric solids
inductive GeometricSolid
  | Cylinder
  | HexagonalPyramid
  | Cube
  | Sphere
  | Tetrahedron

-- Define what it means to be a body of revolution
def isBodyOfRevolution : GeometricSolid → Prop :=
  fun solid => match solid with
    | GeometricSolid.Cylinder => True
    | GeometricSolid.Sphere => True
    | _ => False

-- Theorem statement
theorem bodies_of_revolution :
  ∀ (solid : GeometricSolid),
    isBodyOfRevolution solid ↔ (solid = GeometricSolid.Cylinder ∨ solid = GeometricSolid.Sphere) :=
by sorry

end NUMINAMATH_CALUDE_bodies_of_revolution_l2196_219652


namespace NUMINAMATH_CALUDE_find_value_of_A_l2196_219693

theorem find_value_of_A : ∃ A : ℚ, 
  (∃ B : ℚ, B - A = 0.99 ∧ B = 10 * A) → A = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_find_value_of_A_l2196_219693


namespace NUMINAMATH_CALUDE_range_of_difference_l2196_219601

theorem range_of_difference (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_difference_l2196_219601


namespace NUMINAMATH_CALUDE_second_cat_brown_kittens_count_l2196_219648

/-- The number of brown-eyed kittens the second cat has -/
def second_cat_brown_kittens : ℕ := sorry

/-- The total number of kittens from both cats -/
def total_kittens : ℕ := 14 + second_cat_brown_kittens

/-- The total number of blue-eyed kittens from both cats -/
def blue_eyed_kittens : ℕ := 7

/-- The percentage of blue-eyed kittens -/
def blue_eyed_percentage : ℚ := 35 / 100

theorem second_cat_brown_kittens_count : second_cat_brown_kittens = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_cat_brown_kittens_count_l2196_219648


namespace NUMINAMATH_CALUDE_proportion_solution_l2196_219606

theorem proportion_solution (x : ℝ) : (0.75 / x = 5 / 11) → x = 1.65 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2196_219606


namespace NUMINAMATH_CALUDE_circle_radius_c_value_l2196_219675

theorem circle_radius_c_value (c : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + c = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25) → 
  c = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_c_value_l2196_219675


namespace NUMINAMATH_CALUDE_increasing_interval_implies_a_l2196_219624

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a|

-- State the theorem
theorem increasing_interval_implies_a (a : ℝ) :
  (∀ x ≥ 3, ∀ y > x, f a y > f a x) ∧
  (∀ x < 3, ∃ y > x, f a y ≤ f a x) →
  a = -6 := by
  sorry

end NUMINAMATH_CALUDE_increasing_interval_implies_a_l2196_219624


namespace NUMINAMATH_CALUDE_train_speed_l2196_219618

/-- The speed of a train given its length, the speed of a man running in the opposite direction,
    and the time it takes for the train to pass the man. -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_length = 140 →
  man_speed = 6 →
  passing_time = 6 →
  (train_length / passing_time) * 3.6 - man_speed = 78 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2196_219618


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l2196_219659

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l2196_219659


namespace NUMINAMATH_CALUDE_function_inequality_l2196_219637

-- Define a real-valued function f on ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
variable (hf' : ∀ x, HasDerivAt f (f' x) x)

-- State that f'(x) < f(x) for all x ∈ ℝ
variable (h : ∀ x, f' x < f x)

-- Theorem statement
theorem function_inequality (f f' : ℝ → ℝ) (hf' : ∀ x, HasDerivAt f (f' x) x) (h : ∀ x, f' x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2001 < Real.exp 2001 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2196_219637


namespace NUMINAMATH_CALUDE_magnitude_z_l2196_219636

theorem magnitude_z (w z : ℂ) (h1 : w * z = 16 - 30 * I) (h2 : Complex.abs w = 5) : 
  Complex.abs z = 6.8 := by
sorry

end NUMINAMATH_CALUDE_magnitude_z_l2196_219636


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2196_219670

theorem triangle_perimeter (a b c : ℝ) : 
  (a - 2)^2 + |b - 4| = 0 → 
  c > 0 →
  c < a + b →
  c > |a - b| →
  ∃ (n : ℕ), c = 2 * n →
  a + b + c = 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2196_219670


namespace NUMINAMATH_CALUDE_difference_X_Y_cost_per_capsule_l2196_219653

/-- Represents a bottle of capsules -/
structure Bottle where
  capsules : ℕ
  cost : ℚ

/-- Calculates the cost per capsule for a given bottle -/
def costPerCapsule (b : Bottle) : ℚ :=
  b.cost / b.capsules

/-- Theorem stating the difference in cost per capsule between bottles X and Y -/
theorem difference_X_Y_cost_per_capsule :
  let R : Bottle := { capsules := 250, cost := 25/4 }
  let T : Bottle := { capsules := 100, cost := 3 }
  let X : Bottle := { capsules := 300, cost := 15/2 }
  let Y : Bottle := { capsules := 120, cost := 4 }
  abs (costPerCapsule X - costPerCapsule Y) = 83/10000 := by
  sorry

end NUMINAMATH_CALUDE_difference_X_Y_cost_per_capsule_l2196_219653


namespace NUMINAMATH_CALUDE_salem_poem_word_count_l2196_219694

/-- Represents a poem with a specific structure -/
structure Poem where
  stanzas : Nat
  lines_per_stanza : Nat
  words_per_line : Nat

/-- Calculates the total number of words in a poem -/
def total_words (p : Poem) : Nat :=
  p.stanzas * p.lines_per_stanza * p.words_per_line

/-- Theorem: A poem with 35 stanzas, 15 lines per stanza, and 12 words per line has 6300 words -/
theorem salem_poem_word_count :
  let p : Poem := { stanzas := 35, lines_per_stanza := 15, words_per_line := 12 }
  total_words p = 6300 := by
  sorry

#eval total_words { stanzas := 35, lines_per_stanza := 15, words_per_line := 12 }

end NUMINAMATH_CALUDE_salem_poem_word_count_l2196_219694


namespace NUMINAMATH_CALUDE_prob_select_AB_correct_l2196_219604

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of students to be selected
def selected_students : ℕ := 3

-- Define the probability of selecting both A and B
def prob_select_AB : ℚ := 3 / 10

-- Theorem statement
theorem prob_select_AB_correct :
  (Nat.choose (total_students - 2) (selected_students - 2)) / (Nat.choose total_students selected_students) = prob_select_AB :=
sorry

end NUMINAMATH_CALUDE_prob_select_AB_correct_l2196_219604


namespace NUMINAMATH_CALUDE_parabola_intersection_points_l2196_219623

/-- The parabola y = x^2 + 2x + a - 2 has exactly two intersection points with the coordinate axes if and only if a = 2 or a = 3 -/
theorem parabola_intersection_points (a : ℝ) : 
  (∃! (x y : ℝ), y = x^2 + 2*x + a - 2 ∧ (x = 0 ∨ y = 0)) ∧ 
  (∃ (x1 x2 y1 y2 : ℝ), (x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
    (y1 = x1^2 + 2*x1 + a - 2) ∧ (y2 = x2^2 + 2*x2 + a - 2) ∧ 
    ((x1 = 0 ∨ y1 = 0) ∧ (x2 = 0 ∨ y2 = 0))) ↔ 
  (a = 2 ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_points_l2196_219623


namespace NUMINAMATH_CALUDE_discount_apple_price_l2196_219632

/-- Given a discount percentage and the discounted total price for a certain quantity of apples,
    calculate the original price per kilogram. -/
def original_price_per_kg (discount_percent : ℚ) (discounted_total_price : ℚ) (quantity_kg : ℚ) : ℚ :=
  (discounted_total_price / quantity_kg) / (1 - discount_percent)

/-- Theorem stating that if a 40% discount results in $30 for 10 kg of apples, 
    then the original price was $5 per kg. -/
theorem discount_apple_price : 
  original_price_per_kg (40/100) 30 10 = 5 := by
  sorry

#eval original_price_per_kg (40/100) 30 10

end NUMINAMATH_CALUDE_discount_apple_price_l2196_219632


namespace NUMINAMATH_CALUDE_minimum_final_percentage_is_60_percent_l2196_219646

def total_points : ℕ := 700
def passing_threshold : ℚ := 70 / 100
def problem_set_score : ℕ := 100
def midterm1_score : ℚ := 60 / 100
def midterm2_score : ℚ := 70 / 100
def midterm3_score : ℚ := 80 / 100
def final_exam_points : ℕ := 300

def minimum_final_percentage (total : ℕ) (threshold : ℚ) (problem_set : ℕ) 
  (mid1 mid2 mid3 : ℚ) (final_points : ℕ) : ℚ :=
  -- Definition of the function to calculate the minimum final percentage
  sorry

theorem minimum_final_percentage_is_60_percent :
  minimum_final_percentage total_points passing_threshold problem_set_score
    midterm1_score midterm2_score midterm3_score final_exam_points = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_minimum_final_percentage_is_60_percent_l2196_219646


namespace NUMINAMATH_CALUDE_circle_inequality_l2196_219602

theorem circle_inequality (a b c d : ℝ) (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (hab : a * b + c * d = 1)
    (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
    (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) : 
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₁ + b * x₃ + c * x₂ + d * x₁)^2 
    ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_l2196_219602


namespace NUMINAMATH_CALUDE_matrix_power_identity_l2196_219640

/-- Given a 2x2 matrix B, prove that B^4 = 51*B + 52*I --/
theorem matrix_power_identity (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B = !![1, 2; 3, 1]) : 
  B^4 = 51 • B + 52 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_identity_l2196_219640


namespace NUMINAMATH_CALUDE_carol_first_six_probability_l2196_219627

/-- The probability of rolling a 6 on any single roll. -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on any single roll. -/
def prob_not_six : ℚ := 1 - prob_six

/-- The sequence of rolls, where 0 represents Alice, 1 represents Bob, and 2 represents Carol. -/
def roll_sequence : ℕ → Fin 3
  | n => n % 3

/-- The probability that Carol is the first to roll a 6. -/
def prob_carol_first_six : ℚ :=
  let a : ℚ := prob_not_six ^ 2 * prob_six
  let r : ℚ := prob_not_six ^ 3
  a / (1 - r)

theorem carol_first_six_probability :
  prob_carol_first_six = 25 / 91 := by
  sorry

end NUMINAMATH_CALUDE_carol_first_six_probability_l2196_219627


namespace NUMINAMATH_CALUDE_remaining_money_l2196_219665

-- Define the initial amount, amount spent on sweets, and amount given to each friend
def initial_amount : ℚ := 20.10
def sweets_cost : ℚ := 1.05
def friend_gift : ℚ := 1.00
def num_friends : ℕ := 2

-- Define the theorem
theorem remaining_money :
  initial_amount - sweets_cost - (friend_gift * num_friends) = 17.05 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l2196_219665


namespace NUMINAMATH_CALUDE_parallelogram_z_range_l2196_219680

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -2)

-- Define the function z
def z (x y : ℝ) : ℝ := 2*x - 5*y

-- Statement of the theorem
theorem parallelogram_z_range :
  ∀ (x y : ℝ), 
  (∃ (t₁ t₂ t₃ : ℝ), 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ 0 ≤ t₃ ∧ t₁ + t₂ + t₃ ≤ 1 ∧
    (x, y) = t₁ • A + t₂ • B + t₃ • C + (1 - t₁ - t₂ - t₃) • (C + A - B)) →
  -14 ≤ z x y ∧ z x y ≤ 20 :=
by sorry


end NUMINAMATH_CALUDE_parallelogram_z_range_l2196_219680


namespace NUMINAMATH_CALUDE_rose_bundle_price_l2196_219620

theorem rose_bundle_price (rose_price : ℕ) (total_roses : ℕ) (num_bundles : ℕ) 
  (h1 : rose_price = 500)
  (h2 : total_roses = 200)
  (h3 : num_bundles = 25) :
  (rose_price * total_roses) / num_bundles = 4000 :=
by sorry

end NUMINAMATH_CALUDE_rose_bundle_price_l2196_219620
