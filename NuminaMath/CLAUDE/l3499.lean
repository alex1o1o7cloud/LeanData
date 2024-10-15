import Mathlib

namespace NUMINAMATH_CALUDE_overlapping_squares_area_l3499_349978

/-- Represents a square sheet of paper --/
structure Square :=
  (side_length : ℝ)

/-- Represents the configuration of three overlapping squares --/
structure OverlappingSquares :=
  (base : Square)
  (middle_rotation : ℝ)
  (top_rotation : ℝ)

/-- Calculates the area of the resulting polygon --/
def polygon_area (config : OverlappingSquares) : ℝ :=
  sorry

/-- The main theorem --/
theorem overlapping_squares_area :
  let config := OverlappingSquares.mk (Square.mk 6) (30 * π / 180) (60 * π / 180)
  polygon_area config = 108 - 36 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l3499_349978


namespace NUMINAMATH_CALUDE_min_pigs_on_farm_l3499_349968

theorem min_pigs_on_farm (P : ℕ) (T : ℕ) : 
  (P > 0) → 
  (T > 0) → 
  (P ≤ T) → 
  (54 * T ≤ 100 * P) → 
  (100 * P ≤ 57 * T) → 
  (∀ Q : ℕ, Q > 0 ∧ Q < P → ¬(54 * T ≤ 100 * Q ∧ 100 * Q ≤ 57 * T)) →
  P = 5 :=
by sorry

#check min_pigs_on_farm

end NUMINAMATH_CALUDE_min_pigs_on_farm_l3499_349968


namespace NUMINAMATH_CALUDE_square_sum_difference_equals_338_l3499_349910

theorem square_sum_difference_equals_338 :
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 338 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_equals_338_l3499_349910


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3499_349936

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 3) - y^2 / (k + 3) = 1

-- Define the sufficient condition
def sufficient_condition (k : ℝ) : Prop :=
  k > 3 → is_hyperbola k

-- Define the necessary condition
def necessary_condition (k : ℝ) : Prop :=
  is_hyperbola k → k > 3

-- Theorem statement
theorem hyperbola_condition :
  (∀ k : ℝ, sufficient_condition k) ∧ ¬(∀ k : ℝ, necessary_condition k) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3499_349936


namespace NUMINAMATH_CALUDE_sum_mod_five_zero_l3499_349938

theorem sum_mod_five_zero : (4283 + 4284 + 4285 + 4286 + 4287) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_five_zero_l3499_349938


namespace NUMINAMATH_CALUDE_sum_of_solutions_l3499_349954

-- Define the equation
def equation (x : ℝ) : Prop :=
  x / 3 + x / Real.sqrt (x^2 - 9) = 35 / 12

-- Define the set of solutions
def solution_set : Set ℝ :=
  {x | equation x ∧ x^2 > 9}

-- Theorem statement
theorem sum_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solution_set ∧ s.sum id = 35 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l3499_349954


namespace NUMINAMATH_CALUDE_total_blue_balloons_l3499_349973

theorem total_blue_balloons (joan_balloons melanie_balloons : ℕ) 
  (h1 : joan_balloons = 40) 
  (h2 : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l3499_349973


namespace NUMINAMATH_CALUDE_max_x_minus_y_l3499_349942

theorem max_x_minus_y (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (prod_eq : x*y + y*z + z*x = 1) : 
  ∃ (max : ℝ), max = 2 / Real.sqrt 3 ∧ 
  ∀ (a b c : ℝ), a + b + c = 2 → a*b + b*c + c*a = 1 → 
  |a - b| ≤ max := by
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l3499_349942


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3499_349903

/-- The complex number i(2-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I * (2 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3499_349903


namespace NUMINAMATH_CALUDE_greatest_int_with_gcd_18_6_l3499_349958

theorem greatest_int_with_gcd_18_6 : 
  (∀ n : ℕ, n < 200 ∧ n > 174 → Nat.gcd n 18 ≠ 6) ∧ 
  Nat.gcd 174 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_int_with_gcd_18_6_l3499_349958


namespace NUMINAMATH_CALUDE_tiles_arrangement_exists_l3499_349972

/-- Represents a tile with a diagonal -/
inductive Tile
| LeftDiagonal
| RightDiagonal

/-- Represents the 8x8 grid -/
def Grid := Fin 8 → Fin 8 → Tile

/-- Checks if two adjacent tiles have non-overlapping diagonals -/
def compatible (t1 t2 : Tile) : Prop :=
  t1 ≠ t2

/-- Checks if the entire grid is valid (no overlapping diagonals) -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j, i < 7 → compatible (g i j) (g (i+1) j) ∧
         j < 7 → compatible (g i j) (g i (j+1))

/-- The main theorem stating that a valid arrangement exists -/
theorem tiles_arrangement_exists : ∃ g : Grid, valid_grid g :=
  sorry

end NUMINAMATH_CALUDE_tiles_arrangement_exists_l3499_349972


namespace NUMINAMATH_CALUDE_number_of_possible_lists_l3499_349923

def number_of_balls : ℕ := 15
def list_length : ℕ := 4

theorem number_of_possible_lists :
  (number_of_balls ^ list_length : ℕ) = 50625 := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_lists_l3499_349923


namespace NUMINAMATH_CALUDE_expected_balls_in_original_position_l3499_349927

/-- Represents the number of balls arranged in a circle -/
def numBalls : ℕ := 6

/-- Represents the number of people performing swaps -/
def numSwaps : ℕ := 3

/-- Probability that a specific ball is not involved in a single swap -/
def probNotSwapped : ℚ := 4 / 6

/-- Probability that a ball remains in its original position after all swaps -/
def probInOriginalPosition : ℚ := probNotSwapped ^ numSwaps

/-- Expected number of balls in their original positions after all swaps -/
def expectedBallsInOriginalPosition : ℚ := numBalls * probInOriginalPosition

/-- Theorem stating the expected number of balls in their original positions -/
theorem expected_balls_in_original_position :
  expectedBallsInOriginalPosition = 48 / 27 := by
  sorry

end NUMINAMATH_CALUDE_expected_balls_in_original_position_l3499_349927


namespace NUMINAMATH_CALUDE_hancho_drank_03L_l3499_349905

/-- The amount of milk Hancho drank -/
def hancho_consumption (initial_amount yeseul_consumption gayoung_extra remaining : ℝ) : ℝ :=
  initial_amount - (yeseul_consumption + (yeseul_consumption + gayoung_extra) + remaining)

/-- Theorem stating that Hancho drank 0.3 L of milk given the initial conditions -/
theorem hancho_drank_03L (initial_amount yeseul_consumption gayoung_extra remaining : ℝ) 
  (h1 : initial_amount = 1)
  (h2 : yeseul_consumption = 0.1)
  (h3 : gayoung_extra = 0.2)
  (h4 : remaining = 0.3) :
  hancho_consumption initial_amount yeseul_consumption gayoung_extra remaining = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_hancho_drank_03L_l3499_349905


namespace NUMINAMATH_CALUDE_workbook_selection_cases_l3499_349900

/-- The number of cases to choose either a Korean workbook or a math workbook -/
def total_cases (korean_books : ℕ) (math_books : ℕ) : ℕ :=
  korean_books + math_books

/-- Theorem: Given 2 types of Korean workbooks and 4 types of math workbooks,
    the total number of cases to choose either a Korean workbook or a math workbook is 6 -/
theorem workbook_selection_cases : total_cases 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_workbook_selection_cases_l3499_349900


namespace NUMINAMATH_CALUDE_beads_per_necklace_l3499_349953

/-- Given that Emily made 6 necklaces and used a total of 18 beads,
    prove that each necklace needs 3 beads. -/
theorem beads_per_necklace :
  let total_necklaces : ℕ := 6
  let total_beads : ℕ := 18
  total_beads / total_necklaces = 3 := by sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l3499_349953


namespace NUMINAMATH_CALUDE_quartic_roots_arithmetic_sequence_l3499_349957

theorem quartic_roots_arithmetic_sequence (m n : ℚ) : 
  (∃ a b c d : ℚ, 
    (a^2 - 2*a + m) * (a^2 - 2*a + n) = 0 ∧
    (b^2 - 2*b + m) * (b^2 - 2*b + n) = 0 ∧
    (c^2 - 2*c + m) * (c^2 - 2*c + n) = 0 ∧
    (d^2 - 2*d + m) * (d^2 - 2*d + n) = 0 ∧
    a = 1/4 ∧
    b - a = c - b ∧
    c - b = d - c) →
  |m - n| = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quartic_roots_arithmetic_sequence_l3499_349957


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l3499_349901

/-- Proves that given two rectangles with equal area, where one rectangle has dimensions 12 inches 
    by W inches, and the other rectangle has dimensions 6 inches by 30 inches, the value of W is 15 inches. -/
theorem equal_area_rectangles_width (W : ℝ) : 
  (12 * W = 6 * 30) → W = 15 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l3499_349901


namespace NUMINAMATH_CALUDE_weightlifting_winner_l3499_349914

theorem weightlifting_winner (A B C : ℕ) 
  (sum_AB : A + B = 220)
  (sum_AC : A + C = 240)
  (sum_BC : B + C = 250) :
  max A (max B C) = 135 := by
sorry

end NUMINAMATH_CALUDE_weightlifting_winner_l3499_349914


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l3499_349944

/-- The sum of digits of a natural number in base 10 -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digitSum (n / 10)

/-- The number 4444^444 -/
def bigNumber : ℕ := 4444^444

/-- A is the sum of digits of bigNumber -/
def A : ℕ := digitSum bigNumber

/-- B is the sum of digits of A -/
def B : ℕ := digitSum A

theorem sum_of_digits_of_B_is_seven : digitSum B = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l3499_349944


namespace NUMINAMATH_CALUDE_a_minus_b_values_l3499_349939

theorem a_minus_b_values (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a + b > 0) :
  a - b = -1 ∨ a - b = -7 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_values_l3499_349939


namespace NUMINAMATH_CALUDE_segment_length_l3499_349995

/-- Given two points P and Q on a line segment AB, prove that AB has length 336/11 -/
theorem segment_length (A B P Q : ℝ) : 
  (0 < A ∧ A < P ∧ P < Q ∧ Q < B) →  -- P and Q are on AB and on the same side of midpoint
  (P - A) / (B - P) = 3 / 4 →        -- P divides AB in ratio 3:4
  (Q - A) / (B - Q) = 5 / 7 →        -- Q divides AB in ratio 5:7
  Q - P = 4 →                        -- PQ = 4
  B - A = 336 / 11 := by             -- AB has length 336/11
sorry


end NUMINAMATH_CALUDE_segment_length_l3499_349995


namespace NUMINAMATH_CALUDE_painter_week_total_l3499_349974

/-- Represents the painter's work schedule and productivity --/
structure PainterSchedule where
  monday_speed : ℝ
  normal_speed : ℝ
  friday_speed : ℝ
  normal_hours : ℝ
  friday_hours : ℝ
  friday_monday_diff : ℝ

/-- Calculates the total length of fence painted over the week --/
def total_painted (schedule : PainterSchedule) : ℝ :=
  let monday_length := schedule.monday_speed * schedule.normal_hours
  let normal_day_length := schedule.normal_speed * schedule.normal_hours
  let friday_length := schedule.friday_speed * schedule.friday_hours
  monday_length + 3 * normal_day_length + friday_length

/-- Theorem stating the total length of fence painted over the week --/
theorem painter_week_total (schedule : PainterSchedule)
  (h1 : schedule.monday_speed = 0.5 * schedule.normal_speed)
  (h2 : schedule.friday_speed = 2 * schedule.normal_speed)
  (h3 : schedule.friday_hours = 6)
  (h4 : schedule.normal_hours = 8)
  (h5 : schedule.friday_speed * schedule.friday_hours - 
        schedule.monday_speed * schedule.normal_hours = schedule.friday_monday_diff)
  (h6 : schedule.friday_monday_diff = 300) :
  total_painted schedule = 1500 := by
  sorry


end NUMINAMATH_CALUDE_painter_week_total_l3499_349974


namespace NUMINAMATH_CALUDE_fraction_product_l3499_349983

theorem fraction_product : (2 : ℚ) / 3 * 5 / 7 * 9 / 11 * 4 / 13 = 360 / 3003 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3499_349983


namespace NUMINAMATH_CALUDE_sum_of_eleventh_powers_l3499_349956

theorem sum_of_eleventh_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_eleventh_powers_l3499_349956


namespace NUMINAMATH_CALUDE_trig_identities_l3499_349962

theorem trig_identities (α : ℝ) (h : Real.tan α = 7) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8/13 ∧
  Real.sin α * Real.cos α = 7/50 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l3499_349962


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l3499_349902

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ / b₁ = a₂ / b₂

/-- Definition of line l₁ -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 3) * x + 4 * y + 3 * m - 5 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 5) * y - 8 = 0

/-- Theorem: If l₁ and l₂ are parallel, then m = -7 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel (m + 3) 4 2 (m + 5) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l3499_349902


namespace NUMINAMATH_CALUDE_min_value_of_g_l3499_349997

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^3

-- State the theorem
theorem min_value_of_g :
  ∃ (min : ℝ), min = 16 * Real.sqrt 3 / 9 ∧
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → g x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_g_l3499_349997


namespace NUMINAMATH_CALUDE_duck_percentage_among_non_herons_l3499_349982

theorem duck_percentage_among_non_herons (total : ℝ) (geese swan heron duck : ℝ) :
  geese = 0.28 * total →
  swan = 0.20 * total →
  heron = 0.15 * total →
  duck = 0.32 * total →
  (duck / (total - heron)) * 100 = 37.6 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_percentage_among_non_herons_l3499_349982


namespace NUMINAMATH_CALUDE_pizza_dough_flour_calculation_l3499_349960

theorem pizza_dough_flour_calculation 
  (original_doughs : ℕ) 
  (original_flour_per_dough : ℚ) 
  (new_doughs : ℕ) 
  (total_flour : ℚ) 
  (h1 : original_doughs = 45)
  (h2 : original_flour_per_dough = 1/9)
  (h3 : new_doughs = 15)
  (h4 : total_flour = original_doughs * original_flour_per_dough)
  (h5 : total_flour = new_doughs * (total_flour / new_doughs)) :
  total_flour / new_doughs = 1/3 := by
sorry

end NUMINAMATH_CALUDE_pizza_dough_flour_calculation_l3499_349960


namespace NUMINAMATH_CALUDE_pants_price_problem_l3499_349947

theorem pants_price_problem (total_cost belt_price pants_price : ℝ) :
  total_cost = 70.93 →
  pants_price = belt_price - 2.93 →
  total_cost = pants_price + belt_price →
  pants_price = 34.00 := by
sorry

end NUMINAMATH_CALUDE_pants_price_problem_l3499_349947


namespace NUMINAMATH_CALUDE_unique_symmetric_shape_l3499_349906

-- Define a type for the shapes
inductive Shape : Type
  | A | B | C | D | E

-- Define a function to represent symmetry with respect to the vertical line
def isSymmetric (s : Shape) : Prop :=
  match s with
  | Shape.D => True
  | _ => False

-- Theorem statement
theorem unique_symmetric_shape :
  ∃! s : Shape, isSymmetric s :=
by
  sorry

end NUMINAMATH_CALUDE_unique_symmetric_shape_l3499_349906


namespace NUMINAMATH_CALUDE_glass_bowls_problem_l3499_349975

/-- The number of glass bowls initially bought -/
def initial_bowls : ℕ := 139

/-- The cost per bowl in Rupees -/
def cost_per_bowl : ℚ := 13

/-- The selling price per bowl in Rupees -/
def selling_price : ℚ := 17

/-- The number of bowls sold -/
def bowls_sold : ℕ := 108

/-- The percentage gain -/
def percentage_gain : ℚ := 23.88663967611336

theorem glass_bowls_problem :
  (percentage_gain / 100 * (initial_bowls * cost_per_bowl) = 
   bowls_sold * selling_price - bowls_sold * cost_per_bowl) ∧
  (initial_bowls ≥ bowls_sold) := by
  sorry

end NUMINAMATH_CALUDE_glass_bowls_problem_l3499_349975


namespace NUMINAMATH_CALUDE_original_lettuce_price_l3499_349986

/-- Grocery order with item substitutions -/
def grocery_order (original_total delivery_tip new_total original_tomatoes new_tomatoes
                   original_celery new_celery new_lettuce : ℚ) : Prop :=
  -- Original order total before changes
  original_total = 25 ∧
  -- Delivery and tip
  delivery_tip = 8 ∧
  -- New total after changes and delivery/tip
  new_total = 35 ∧
  -- Original and new prices for tomatoes
  original_tomatoes = 0.99 ∧
  new_tomatoes = 2.20 ∧
  -- Original and new prices for celery
  original_celery = 1.96 ∧
  new_celery = 2 ∧
  -- New price for lettuce
  new_lettuce = 1.75

/-- The cost of the original lettuce -/
def original_lettuce_cost (original_total delivery_tip new_total original_tomatoes new_tomatoes
                           original_celery new_celery new_lettuce : ℚ) : ℚ :=
  new_lettuce - ((new_total - delivery_tip) - (original_total + (new_tomatoes - original_tomatoes) + (new_celery - original_celery)))

theorem original_lettuce_price
  (original_total delivery_tip new_total original_tomatoes new_tomatoes
   original_celery new_celery new_lettuce : ℚ)
  (h : grocery_order original_total delivery_tip new_total original_tomatoes new_tomatoes
                     original_celery new_celery new_lettuce) :
  original_lettuce_cost original_total delivery_tip new_total original_tomatoes new_tomatoes
                        original_celery new_celery new_lettuce = 1 := by
  sorry

end NUMINAMATH_CALUDE_original_lettuce_price_l3499_349986


namespace NUMINAMATH_CALUDE_four_digit_number_with_two_schemes_l3499_349961

/-- Represents a division scheme for a four-digit number -/
structure DivisionScheme where
  divisor : Nat
  quotient : Nat
  remainder : Nat

/-- Checks if a number satisfies a given division scheme -/
def satisfiesScheme (n : Nat) (scheme : DivisionScheme) : Prop :=
  n / scheme.divisor = scheme.quotient ∧ n % scheme.divisor = scheme.remainder

/-- Theorem stating the existence of a four-digit number satisfying two division schemes -/
theorem four_digit_number_with_two_schemes :
  ∃ (n : Nat) (scheme1 scheme2 : DivisionScheme),
    1000 ≤ n ∧ n < 10000 ∧
    scheme1.divisor ≠ scheme2.divisor ∧
    scheme1.divisor < 10 ∧ scheme2.divisor < 10 ∧
    satisfiesScheme n scheme1 ∧
    satisfiesScheme n scheme2 := by
  sorry

#check four_digit_number_with_two_schemes

end NUMINAMATH_CALUDE_four_digit_number_with_two_schemes_l3499_349961


namespace NUMINAMATH_CALUDE_solution_k_value_l3499_349966

theorem solution_k_value (x y k : ℝ) : 
  x = -3 ∧ y = 2 ∧ 2*x + k*y = 0 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_k_value_l3499_349966


namespace NUMINAMATH_CALUDE_meaningful_expression_l3499_349937

theorem meaningful_expression (x : ℝ) : 
  (10 - x ≥ 0 ∧ x ≠ 4) ↔ x = 8 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3499_349937


namespace NUMINAMATH_CALUDE_ram_money_l3499_349916

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Krishan's amount,
    calculate the amount of money Ram has. -/
theorem ram_money (ram gopal krishan : ℚ) : 
  ram / gopal = 7 / 17 →
  gopal / krishan = 7 / 17 →
  krishan = 3468 →
  ram = 588 := by
sorry

end NUMINAMATH_CALUDE_ram_money_l3499_349916


namespace NUMINAMATH_CALUDE_liquid_film_radius_l3499_349999

/-- Given a box with dimensions and a liquid that partially fills it, 
    calculate the radius of the circular film formed when poured on water. -/
theorem liquid_film_radius 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (fill_percentage : ℝ) 
  (film_thickness : ℝ) : 
  box_length = 5 → 
  box_width = 4 → 
  box_height = 10 → 
  fill_percentage = 0.8 → 
  film_thickness = 0.05 → 
  ∃ (r : ℝ), r = Real.sqrt (3200 / Real.pi) ∧ 
  r^2 * Real.pi * film_thickness = box_length * box_width * box_height * fill_percentage :=
by sorry

end NUMINAMATH_CALUDE_liquid_film_radius_l3499_349999


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l3499_349951

theorem percentage_of_percentage (y : ℝ) (h : y ≠ 0) :
  (30 / 100) * (80 / 100) * y = (24 / 100) * y := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l3499_349951


namespace NUMINAMATH_CALUDE_sum_of_special_integers_l3499_349950

theorem sum_of_special_integers (x y : ℕ) 
  (h1 : x > y) 
  (h2 : x - y = 8) 
  (h3 : x * y = 168) : 
  x + y = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_integers_l3499_349950


namespace NUMINAMATH_CALUDE_steak_weight_l3499_349949

/-- Given 15 pounds of beef cut into 20 equal steaks, prove that each steak weighs 12 ounces. -/
theorem steak_weight (total_pounds : ℕ) (num_steaks : ℕ) (ounces_per_pound : ℕ) : 
  total_pounds = 15 → 
  num_steaks = 20 → 
  ounces_per_pound = 16 → 
  (total_pounds * ounces_per_pound) / num_steaks = 12 := by
  sorry

end NUMINAMATH_CALUDE_steak_weight_l3499_349949


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l3499_349904

/-- Represents the watermelon vendor's business model -/
structure WatermelonVendor where
  initialPurchasePrice : ℝ
  initialSellingPrice : ℝ
  initialDailySales : ℝ
  salesIncreaseRate : ℝ
  fixedCosts : ℝ

/-- Calculates the daily sales volume based on price reduction -/
def dailySalesVolume (w : WatermelonVendor) (priceReduction : ℝ) : ℝ :=
  w.initialDailySales + w.salesIncreaseRate * priceReduction * 10

/-- Calculates the daily profit based on price reduction -/
def dailyProfit (w : WatermelonVendor) (priceReduction : ℝ) : ℝ :=
  (w.initialSellingPrice - priceReduction - w.initialPurchasePrice) * 
  (dailySalesVolume w priceReduction) - w.fixedCosts

/-- Theorem stating the optimal price reduction for maximum sales and 200 yuan profit -/
theorem optimal_price_reduction (w : WatermelonVendor) 
  (h1 : w.initialPurchasePrice = 2)
  (h2 : w.initialSellingPrice = 3)
  (h3 : w.initialDailySales = 200)
  (h4 : w.salesIncreaseRate = 40)
  (h5 : w.fixedCosts = 24) :
  ∃ (x : ℝ), x = 0.3 ∧ 
  dailyProfit w x = 200 ∧ 
  ∀ (y : ℝ), dailyProfit w y = 200 → dailySalesVolume w x ≥ dailySalesVolume w y := by
  sorry


end NUMINAMATH_CALUDE_optimal_price_reduction_l3499_349904


namespace NUMINAMATH_CALUDE_second_third_smallest_average_l3499_349915

theorem second_third_smallest_average (a b c d e : ℕ+) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧  -- five different positive integers
  (a + b + c + d + e : ℚ) / 5 = 5 ∧  -- average is 5
  ∀ x y z w v : ℕ+, x < y ∧ y < z ∧ z < w ∧ w < v → 
    (x + y + z + w + v : ℚ) / 5 = 5 → (v - x : ℚ) ≤ (e - a) →  -- difference is maximized
  (b + c : ℚ) / 2 = 5/2 :=  -- average of second and third smallest is 2.5
sorry

end NUMINAMATH_CALUDE_second_third_smallest_average_l3499_349915


namespace NUMINAMATH_CALUDE_parabola_intersection_condition_l3499_349940

-- Define the parabola function
def parabola (m : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(m-1)*x + m + 1

-- Theorem statement
theorem parabola_intersection_condition (m : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b < 0 ∧ parabola m a = 0 ∧ parabola m b = 0) ↔ m > -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_condition_l3499_349940


namespace NUMINAMATH_CALUDE_peach_expense_l3499_349963

theorem peach_expense (total berries apples : ℝ) 
  (h_total : total = 34.72)
  (h_berries : berries = 11.08)
  (h_apples : apples = 14.33) :
  total - (berries + apples) = 9.31 := by sorry

end NUMINAMATH_CALUDE_peach_expense_l3499_349963


namespace NUMINAMATH_CALUDE_sum_of_cubes_and_cube_of_sum_l3499_349929

theorem sum_of_cubes_and_cube_of_sum : (3 + 6 + 9)^3 + (3^3 + 6^3 + 9^3) = 6804 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_and_cube_of_sum_l3499_349929


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3499_349977

theorem inequality_system_solution_set :
  {x : ℝ | 3 * x + 9 > 0 ∧ 2 * x < 6} = {x : ℝ | -3 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3499_349977


namespace NUMINAMATH_CALUDE_russian_tennis_pairing_probability_l3499_349993

theorem russian_tennis_pairing_probability :
  let total_players : ℕ := 10
  let russian_players : ℕ := 4
  let total_pairs : ℕ := total_players / 2
  let russian_pairs : ℕ := russian_players / 2
  let favorable_outcomes : ℕ := Nat.choose total_pairs russian_pairs
  let total_outcomes : ℕ := Nat.choose total_players russian_players
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 21 :=
by sorry

end NUMINAMATH_CALUDE_russian_tennis_pairing_probability_l3499_349993


namespace NUMINAMATH_CALUDE_find_number_l3499_349919

theorem find_number (k : ℝ) (x : ℝ) (h1 : x / k = 4) (h2 : k = 16) : x = 64 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3499_349919


namespace NUMINAMATH_CALUDE_pin_sequence_solution_l3499_349935

def pin_sequence (k : ℕ) (n : ℕ) : ℕ := 2 + k * (n - 1)

theorem pin_sequence_solution :
  ∀ k : ℕ, (pin_sequence k 10 > 45 ∧ pin_sequence k 15 < 90) ↔ (k = 5 ∨ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_pin_sequence_solution_l3499_349935


namespace NUMINAMATH_CALUDE_remainder_problem_l3499_349909

theorem remainder_problem (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3499_349909


namespace NUMINAMATH_CALUDE_min_value_of_f_l3499_349913

/-- The quadratic function f(x) = 2(x-3)^2 + 2 -/
def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

/-- Theorem: The minimum value of f(x) = 2(x-3)^2 + 2 is 2 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3499_349913


namespace NUMINAMATH_CALUDE_remainder_of_Q_l3499_349964

-- Define the polynomial Q
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom Q_div_21 : ∃ P₁ : ℝ → ℝ, ∀ x, Q x = (x - 21) * (P₁ x) + 105
axiom Q_div_105 : ∃ P₂ : ℝ → ℝ, ∀ x, Q x = (x - 105) * (P₂ x) + 21

-- Theorem statement
theorem remainder_of_Q : 
  ∃ P : ℝ → ℝ, ∀ x, Q x = (x - 21) * (x - 105) * (P x) + (-x + 126) := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_Q_l3499_349964


namespace NUMINAMATH_CALUDE_percentage_invalid_votes_l3499_349992

/-- The percentage of invalid votes in an election --/
theorem percentage_invalid_votes 
  (total_votes : ℕ) 
  (candidate_a_percentage : ℚ) 
  (candidate_a_valid_votes : ℕ) 
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 75 / 100)
  (h3 : candidate_a_valid_votes = 357000) :
  (1 - (candidate_a_valid_votes : ℚ) / (candidate_a_percentage * total_votes)) * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_invalid_votes_l3499_349992


namespace NUMINAMATH_CALUDE_boys_playing_basketball_l3499_349989

/-- Given a class with the following properties:
  * There are 30 students in total
  * One-third of the students are girls
  * Three-quarters of the boys play basketball
  Prove that the number of boys who play basketball is 15 -/
theorem boys_playing_basketball (total_students : ℕ) (girls : ℕ) (boys : ℕ) (boys_playing : ℕ) : 
  total_students = 30 →
  girls = total_students / 3 →
  boys = total_students - girls →
  boys_playing = (3 * boys) / 4 →
  boys_playing = 15 := by
sorry

end NUMINAMATH_CALUDE_boys_playing_basketball_l3499_349989


namespace NUMINAMATH_CALUDE_radius_of_2003rd_circle_l3499_349971

/-- The radius of the nth circle in a sequence of circles tangent to the sides of a 60° angle -/
def radius (n : ℕ) : ℝ :=
  3^(n - 1)

/-- The number of circles in the sequence -/
def num_circles : ℕ := 2003

theorem radius_of_2003rd_circle :
  radius num_circles = 3^2002 :=
by sorry

end NUMINAMATH_CALUDE_radius_of_2003rd_circle_l3499_349971


namespace NUMINAMATH_CALUDE_modulus_one_plus_i_to_sixth_l3499_349925

theorem modulus_one_plus_i_to_sixth (i : ℂ) : i * i = -1 → Complex.abs ((1 + i)^6) = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulus_one_plus_i_to_sixth_l3499_349925


namespace NUMINAMATH_CALUDE_second_concert_proof_l3499_349928

/-- The attendance of the first concert -/
def first_concert_attendance : ℕ := 65899

/-- The additional attendance at the second concert -/
def additional_attendance : ℕ := 119

/-- The attendance of the second concert -/
def second_concert_attendance : ℕ := first_concert_attendance + additional_attendance

theorem second_concert_proof : second_concert_attendance = 66018 := by
  sorry

end NUMINAMATH_CALUDE_second_concert_proof_l3499_349928


namespace NUMINAMATH_CALUDE_sum_220_is_5500_div_3_l3499_349985

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  /-- The first term of the progression -/
  a : ℚ
  /-- The common difference of the progression -/
  d : ℚ
  /-- The sum of the first 20 terms is 500 -/
  sum_20 : (20 : ℚ) / 2 * (2 * a + (19 : ℚ) * d) = 500
  /-- The sum of the first 200 terms is 2000 -/
  sum_200 : (200 : ℚ) / 2 * (2 * a + (199 : ℚ) * d) = 2000

/-- The sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℚ) : ℚ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

/-- Theorem: The sum of the first 220 terms is 5500/3 -/
theorem sum_220_is_5500_div_3 (ap : ArithmeticProgression) :
  sum_n ap 220 = 5500 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_220_is_5500_div_3_l3499_349985


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l3499_349907

/-- 
Proves that in a rhombus with an area of 120 cm² and one diagonal of 20 cm, 
the length of the other diagonal is 12 cm.
-/
theorem rhombus_diagonal_length 
  (area : ℝ) 
  (diagonal1 : ℝ) 
  (diagonal2 : ℝ) 
  (h1 : area = 120) 
  (h2 : diagonal1 = 20) 
  (h3 : area = (diagonal1 * diagonal2) / 2) : 
  diagonal2 = 12 := by
sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l3499_349907


namespace NUMINAMATH_CALUDE_bead_calculation_l3499_349922

theorem bead_calculation (blue_beads yellow_beads : ℕ) 
  (h1 : blue_beads = 23)
  (h2 : yellow_beads = 16) : 
  let total_beads := blue_beads + yellow_beads
  let parts := 3
  let beads_per_part := total_beads / parts
  let removed_beads := 10
  let remaining_beads := beads_per_part - removed_beads
  remaining_beads * 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_bead_calculation_l3499_349922


namespace NUMINAMATH_CALUDE_ceo_dividends_calculation_l3499_349946

/-- Calculates the CEO's dividends based on company financial data -/
theorem ceo_dividends_calculation (revenue : ℝ) (expenses : ℝ) (tax_rate : ℝ) 
  (monthly_loan_payment : ℝ) (months_in_year : ℕ) (total_shares : ℕ) (ceo_ownership : ℝ) 
  (h1 : revenue = 2500000)
  (h2 : expenses = 1576250)
  (h3 : tax_rate = 0.2)
  (h4 : monthly_loan_payment = 25000)
  (h5 : months_in_year = 12)
  (h6 : total_shares = 1600)
  (h7 : ceo_ownership = 0.35) :
  ∃ (ceo_dividends : ℝ),
    ceo_dividends = 153440 ∧
    ceo_dividends = 
      ((revenue - expenses - (revenue - expenses) * tax_rate - 
        (monthly_loan_payment * months_in_year)) / total_shares) * 
      ceo_ownership * total_shares :=
by
  sorry

end NUMINAMATH_CALUDE_ceo_dividends_calculation_l3499_349946


namespace NUMINAMATH_CALUDE_peach_pie_customers_l3499_349988

/-- Represents the number of slices in an apple pie -/
def apple_slices : ℕ := 8

/-- Represents the number of slices in a peach pie -/
def peach_slices : ℕ := 6

/-- Represents the number of customers who ordered apple pie slices -/
def apple_customers : ℕ := 56

/-- Represents the total number of pies sold -/
def total_pies : ℕ := 15

/-- Theorem stating that the number of customers who ordered peach pie slices is 48 -/
theorem peach_pie_customers : 
  (total_pies * peach_slices) - (apple_customers / apple_slices * peach_slices) = 48 := by
  sorry

end NUMINAMATH_CALUDE_peach_pie_customers_l3499_349988


namespace NUMINAMATH_CALUDE_flagpole_height_l3499_349952

/-- Given a flagpole and a building under similar shadow-casting conditions,
    prove that the flagpole's height is 18 meters. -/
theorem flagpole_height 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_height : building_height = 28)
  (h_building_shadow : building_shadow = 70)
  (h_similar_conditions : True)  -- This represents the similar conditions
  : ∃ (flagpole_height : ℝ), flagpole_height = 18 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_height_l3499_349952


namespace NUMINAMATH_CALUDE_parabola_translation_leftward_shift_by_2_l3499_349990

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x + 2)^2

-- Theorem stating the translation
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) :=
by
  sorry

-- Theorem stating the leftward shift by 2 units
theorem leftward_shift_by_2 :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_leftward_shift_by_2_l3499_349990


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3499_349908

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 5*x + 6 < 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3499_349908


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l3499_349911

theorem cubic_root_sum_product (p q r : ℂ) : 
  (2 * p^3 - 4 * p^2 + 7 * p - 3 = 0) →
  (2 * q^3 - 4 * q^2 + 7 * q - 3 = 0) →
  (2 * r^3 - 4 * r^2 + 7 * r - 3 = 0) →
  p * q + q * r + r * p = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l3499_349911


namespace NUMINAMATH_CALUDE_toy_car_speed_l3499_349991

theorem toy_car_speed (t s : ℝ) (h1 : t = 15 * s^2) (h2 : t = 3) : s = Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_toy_car_speed_l3499_349991


namespace NUMINAMATH_CALUDE_distribute_four_balls_three_boxes_l3499_349931

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Stirling number of the second kind: number of ways to partition a set of n objects into k non-empty subsets -/
def stirling_second (n k : ℕ) : ℕ := sorry

theorem distribute_four_balls_three_boxes : 
  distribute_balls 4 3 = 14 := by sorry

end NUMINAMATH_CALUDE_distribute_four_balls_three_boxes_l3499_349931


namespace NUMINAMATH_CALUDE_order_of_expressions_l3499_349926

theorem order_of_expressions : 
  let a : ℝ := (1/2)^(1/2)
  let b : ℝ := Real.log 2015 / Real.log 2014
  let c : ℝ := Real.log 2 / Real.log 4
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l3499_349926


namespace NUMINAMATH_CALUDE_unique_lowest_degree_polynomial_l3499_349967

def f (n : ℕ) : ℕ := n^3 + 2*n^2 + n + 3

theorem unique_lowest_degree_polynomial :
  (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 21 ∧ f 3 = 51) ∧
  (∀ g : ℕ → ℕ, (g 0 = 3 ∧ g 1 = 7 ∧ g 2 = 21 ∧ g 3 = 51) →
    (∃ a b c d : ℕ, ∀ n, g n = a*n^3 + b*n^2 + c*n + d) →
    (∀ n, f n = g n)) :=
by sorry

end NUMINAMATH_CALUDE_unique_lowest_degree_polynomial_l3499_349967


namespace NUMINAMATH_CALUDE_vegetable_bins_l3499_349917

theorem vegetable_bins (total_bins soup_bins pasta_bins : ℚ)
  (h1 : total_bins = 0.75)
  (h2 : soup_bins = 0.12)
  (h3 : pasta_bins = 0.5)
  (h4 : total_bins = soup_bins + pasta_bins + (total_bins - soup_bins - pasta_bins)) :
  total_bins - soup_bins - pasta_bins = 0.13 := by
sorry

end NUMINAMATH_CALUDE_vegetable_bins_l3499_349917


namespace NUMINAMATH_CALUDE_bill_face_value_l3499_349921

/-- Calculates the face value of a bill given true discount, time, and interest rate. -/
def face_value (true_discount : ℚ) (time_months : ℚ) (interest_rate : ℚ) : ℚ :=
  (true_discount * 100) / (interest_rate * (time_months / 12))

/-- Theorem stating that given the specified conditions, the face value of the bill is 1575. -/
theorem bill_face_value :
  let true_discount : ℚ := 189
  let time_months : ℚ := 9
  let interest_rate : ℚ := 16
  face_value true_discount time_months interest_rate = 1575 := by
  sorry


end NUMINAMATH_CALUDE_bill_face_value_l3499_349921


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l3499_349980

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters : ℕ := 8
  let probability : ℚ := unique_letters / alphabet_size
  probability = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l3499_349980


namespace NUMINAMATH_CALUDE_D_72_l3499_349994

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where the order of factors matters. -/
def D (n : ℕ+) : ℕ := sorry

/-- Theorem stating that D(72) = 35 -/
theorem D_72 : D 72 = 35 := by sorry

end NUMINAMATH_CALUDE_D_72_l3499_349994


namespace NUMINAMATH_CALUDE_survey_result_l3499_349970

theorem survey_result (total : ℕ) (lentils : ℕ) (chickpeas : ℕ) (neither : ℕ) 
  (h1 : total = 100)
  (h2 : lentils = 68)
  (h3 : chickpeas = 53)
  (h4 : neither = 6) :
  ∃ both : ℕ, both = 27 ∧ 
    total = lentils + chickpeas - both + neither :=
by sorry

end NUMINAMATH_CALUDE_survey_result_l3499_349970


namespace NUMINAMATH_CALUDE_wage_payment_period_l3499_349945

/-- Given a sum of money that can pay three workers' wages for different periods,
    prove that it can pay their combined wages for a specific period when working together. -/
theorem wage_payment_period (M : ℝ) (p q r : ℝ) : 
  M = 24 * p ∧ M = 40 * q ∧ M = 30 * r → M = 10 * (p + q + r) := by
sorry

end NUMINAMATH_CALUDE_wage_payment_period_l3499_349945


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3499_349955

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3499_349955


namespace NUMINAMATH_CALUDE_class_mean_calculation_l3499_349996

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_mean : ℚ)
  (group2_students : ℕ) (group2_mean : ℚ) :
  total_students = 28 →
  group1_students = 24 →
  group2_students = 4 →
  group1_mean = 68 / 100 →
  group2_mean = 82 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 70 / 100 := by
sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l3499_349996


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l3499_349943

/-- A quadrilateral with perpendicular diagonals, two adjacent equal sides, and one pair of equal opposite angles -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perp_diagonals : Bool
  /-- Two adjacent sides of the quadrilateral are equal -/
  adj_sides_equal : Bool
  /-- One pair of opposite angles are equal -/
  opp_angles_equal : Bool

/-- Definition of a kite -/
def is_kite (q : SpecialQuadrilateral) : Prop :=
  q.perp_diagonals ∧ q.adj_sides_equal

/-- Theorem stating that a quadrilateral with the given properties is a kite -/
theorem special_quadrilateral_is_kite (q : SpecialQuadrilateral) 
  (h1 : q.perp_diagonals = true) 
  (h2 : q.adj_sides_equal = true) 
  (h3 : q.opp_angles_equal = true) : 
  is_kite q :=
sorry

end NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l3499_349943


namespace NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l3499_349969

theorem ceiling_negative_seven_fourths_squared : ⌈(-(7/4))^2⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l3499_349969


namespace NUMINAMATH_CALUDE_additive_function_negative_on_positive_properties_l3499_349998

/-- A function satisfying f(x+y) = f(x) + f(y) for all x, y and f(x) < 0 for x > 0 -/
def AdditiveFunctionNegativeOnPositive (f : ℝ → ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y) ∧ (∀ x, x > 0 → f x < 0)

/-- Theorem stating that such a function is odd and monotonically decreasing -/
theorem additive_function_negative_on_positive_properties
    (f : ℝ → ℝ) (h : AdditiveFunctionNegativeOnPositive f) :
    (∀ x, f (-x) = -f x) ∧ (∀ x₁ x₂, x₁ > x₂ → f x₁ < f x₂) := by
  sorry


end NUMINAMATH_CALUDE_additive_function_negative_on_positive_properties_l3499_349998


namespace NUMINAMATH_CALUDE_trigonometric_equation_proof_l3499_349930

theorem trigonometric_equation_proof (α : ℝ) : 
  (Real.sin (2 * α) + Real.sin (5 * α) - Real.sin (3 * α)) / 
  (Real.cos α + 1 - 2 * (Real.sin (2 * α))^2) = 2 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_proof_l3499_349930


namespace NUMINAMATH_CALUDE_min_trees_chopped_is_270_l3499_349920

def min_trees_chopped (axe_resharpen_interval : ℕ) (saw_regrind_interval : ℕ)
  (axe_sharpen_cost : ℕ) (saw_regrind_cost : ℕ)
  (total_axe_sharpen_cost : ℕ) (total_saw_regrind_cost : ℕ) : ℕ :=
  let axe_sharpenings := (total_axe_sharpen_cost + axe_sharpen_cost - 1) / axe_sharpen_cost
  let saw_regrindings := total_saw_regrind_cost / saw_regrind_cost
  axe_sharpenings * axe_resharpen_interval + saw_regrindings * saw_regrind_interval

theorem min_trees_chopped_is_270 :
  min_trees_chopped 25 20 8 10 46 60 = 270 := by
  sorry

end NUMINAMATH_CALUDE_min_trees_chopped_is_270_l3499_349920


namespace NUMINAMATH_CALUDE_oprah_car_collection_reduction_l3499_349987

def reduce_car_collection (initial_cars : ℕ) (target_cars : ℕ) (cars_given_per_year : ℕ) : ℕ :=
  (initial_cars - target_cars) / cars_given_per_year

theorem oprah_car_collection_reduction :
  reduce_car_collection 3500 500 50 = 60 := by
  sorry

end NUMINAMATH_CALUDE_oprah_car_collection_reduction_l3499_349987


namespace NUMINAMATH_CALUDE_twelfth_term_is_fifteen_l3499_349984

/-- An arithmetic sequence {a_n} with the given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 + a 4 + a 5 = 3 ∧
  a 8 = 8

/-- Theorem stating that for an arithmetic sequence satisfying the given conditions, 
    the 12th term is equal to 15 -/
theorem twelfth_term_is_fifteen (a : ℕ → ℚ) 
  (h : arithmetic_sequence a) : a 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_is_fifteen_l3499_349984


namespace NUMINAMATH_CALUDE_greatest_common_divisor_and_digit_sum_l3499_349976

def a : ℕ := 1305
def b : ℕ := 4665
def c : ℕ := 6905

def diff1 : ℕ := b - a
def diff2 : ℕ := c - b
def diff3 : ℕ := c - a

def n : ℕ := Nat.gcd diff1 (Nat.gcd diff2 diff3)

def sum_of_digits (k : ℕ) : ℕ :=
  if k < 10 then k else (k % 10) + sum_of_digits (k / 10)

theorem greatest_common_divisor_and_digit_sum :
  n = 1120 ∧ sum_of_digits n = 4 := by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_and_digit_sum_l3499_349976


namespace NUMINAMATH_CALUDE_largest_angle_in_18_sided_polygon_l3499_349912

theorem largest_angle_in_18_sided_polygon (n : ℕ) (sum_other_angles : ℝ) :
  n = 18 ∧ sum_other_angles = 2754 →
  (n - 2) * 180 - sum_other_angles = 126 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_18_sided_polygon_l3499_349912


namespace NUMINAMATH_CALUDE_sin_150_degrees_l3499_349941

theorem sin_150_degrees : Real.sin (150 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l3499_349941


namespace NUMINAMATH_CALUDE_loan_repayment_months_l3499_349924

/-- Represents the monthly income in ten thousands of yuan -/
def monthlyIncome : ℕ → ℚ
  | 0 => 20  -- First month's income
  | n + 1 => if n < 5 then monthlyIncome n * 1.2 else monthlyIncome n + 2

/-- Calculates the cumulative income up to month n -/
def cumulativeIncome (n : ℕ) : ℚ :=
  (List.range n).map monthlyIncome |>.sum

/-- The loan amount in ten thousands of yuan -/
def loanAmount : ℚ := 400

theorem loan_repayment_months :
  (∀ k < 10, cumulativeIncome k < loanAmount) ∧
  cumulativeIncome 10 ≥ loanAmount := by
  sorry

#eval cumulativeIncome 10  -- For verification

end NUMINAMATH_CALUDE_loan_repayment_months_l3499_349924


namespace NUMINAMATH_CALUDE_melissa_points_per_game_l3499_349933

/-- The number of points Melissa scored in total -/
def total_points : ℕ := 91

/-- The number of games Melissa played -/
def num_games : ℕ := 13

/-- The number of points Melissa scored in each game -/
def points_per_game : ℕ := total_points / num_games

/-- Theorem stating that Melissa scored 7 points in each game -/
theorem melissa_points_per_game : points_per_game = 7 := by
  sorry

end NUMINAMATH_CALUDE_melissa_points_per_game_l3499_349933


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3499_349918

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSeq (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 1 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_incr_geom : IsIncreasingGeometricSeq a)
  (h_sum : a 4 + a 6 = 6)
  (h_prod : a 2 * a 8 = 8) :
  a 3 = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3499_349918


namespace NUMINAMATH_CALUDE_root_shift_theorem_l3499_349979

theorem root_shift_theorem (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 6*x^2 + 11*x - 6 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 - 15*x^2 + 74*x - 120 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_theorem_l3499_349979


namespace NUMINAMATH_CALUDE_intersection_when_m_neg_one_subset_iff_m_leq_neg_two_l3499_349959

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m < x ∧ x < 1 - m}

-- Theorem 1: When m = -1, A ∩ B = {x | 1 < x < 2}
theorem intersection_when_m_neg_one :
  A ∩ B (-1) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 2: A ⊆ B if and only if m ≤ -2
theorem subset_iff_m_leq_neg_two :
  ∀ m : ℝ, A ⊆ B m ↔ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_neg_one_subset_iff_m_leq_neg_two_l3499_349959


namespace NUMINAMATH_CALUDE_evening_temperature_l3499_349948

def initial_temp : Int := -7
def temp_rise : Int := 11
def temp_drop : Int := 9

theorem evening_temperature :
  initial_temp + temp_rise - temp_drop = -5 :=
by sorry

end NUMINAMATH_CALUDE_evening_temperature_l3499_349948


namespace NUMINAMATH_CALUDE_multiply_and_add_l3499_349981

theorem multiply_and_add : 12 * 25 + 16 * 15 = 540 := by sorry

end NUMINAMATH_CALUDE_multiply_and_add_l3499_349981


namespace NUMINAMATH_CALUDE_range_of_m_for_two_distinct_zeros_l3499_349934

/-- A quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + (m + 3)

/-- The discriminant of the quadratic function -/
def discriminant (m : ℝ) : ℝ := m^2 - 4*(m + 3)

/-- The theorem stating the range of m for which the quadratic function has two distinct zeros -/
theorem range_of_m_for_two_distinct_zeros :
  ∀ m : ℝ, (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m ∈ Set.Ioi 6 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_two_distinct_zeros_l3499_349934


namespace NUMINAMATH_CALUDE_dima_places_more_berries_l3499_349932

/-- The total number of berries on the bush -/
def total_berries : ℕ := 450

/-- Dima's picking pattern: fraction of berries that go into the basket -/
def dima_basket_ratio : ℚ := 1/2

/-- Sergei's picking pattern: fraction of berries that go into the basket -/
def sergei_basket_ratio : ℚ := 2/3

/-- Dima's picking speed relative to Sergei -/
def dima_speed_ratio : ℕ := 2

/-- The number of berries Dima puts in the basket -/
def dima_basket_berries : ℕ := 150

/-- The number of berries Sergei puts in the basket -/
def sergei_basket_berries : ℕ := 100

/-- Theorem stating that Dima places 50 more berries into the basket than Sergei -/
theorem dima_places_more_berries :
  dima_basket_berries - sergei_basket_berries = 50 :=
sorry

end NUMINAMATH_CALUDE_dima_places_more_berries_l3499_349932


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3499_349965

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 150 →
  crossing_time = 12.499 →
  ∃ (speed : ℝ), abs (speed - 72) < 0.1 ∧ speed = (train_length + bridge_length) / crossing_time * 3.6 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3499_349965
