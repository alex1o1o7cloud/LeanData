import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_solve_inequality_system_l814_81424

-- Problem 1
theorem simplify_expression (x : ℝ) : (2*x + 1)^2 + x*(x - 4) = 5*x^2 + 1 := by
  sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) : 
  (3*x - 6 > 0 ∧ (5 - x)/2 < 1) ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_inequality_system_l814_81424


namespace NUMINAMATH_CALUDE_nyc_streetlights_l814_81483

/-- The number of streetlights bought by the New York City Council -/
theorem nyc_streetlights (num_squares : ℕ) (lights_per_square : ℕ) (unused_lights : ℕ) :
  num_squares = 15 →
  lights_per_square = 12 →
  unused_lights = 20 →
  num_squares * lights_per_square + unused_lights = 200 := by
  sorry


end NUMINAMATH_CALUDE_nyc_streetlights_l814_81483


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l814_81484

theorem roots_sum_of_squares (a b : ℝ) : 
  (∀ x, x^2 - 4*x + 4 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l814_81484


namespace NUMINAMATH_CALUDE_dinner_set_cost_calculation_l814_81459

/-- The cost calculation for John's dinner set purchase --/
theorem dinner_set_cost_calculation :
  let fork_cost : ℚ := 25
  let knife_cost : ℚ := 30
  let spoon_cost : ℚ := 20
  let silverware_cost : ℚ := fork_cost + knife_cost + spoon_cost
  let plate_cost : ℚ := silverware_cost * (1/2)
  let total_cost : ℚ := silverware_cost + plate_cost
  let discount_rate : ℚ := 1/10
  let final_cost : ℚ := total_cost * (1 - discount_rate)
  final_cost = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_dinner_set_cost_calculation_l814_81459


namespace NUMINAMATH_CALUDE_value_of_x_l814_81446

theorem value_of_x (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 48 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l814_81446


namespace NUMINAMATH_CALUDE_solve_system_l814_81404

theorem solve_system (x y : ℤ) 
  (h1 : x + y = 270) 
  (h2 : x - y = 200) : 
  y = 35 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l814_81404


namespace NUMINAMATH_CALUDE_solution_characterization_l814_81406

def is_solution (x y z w : ℝ) : Prop :=
  x + y + z + w = 10 ∧
  x^2 + y^2 + z^2 + w^2 = 30 ∧
  x^3 + y^3 + z^3 + w^3 = 100 ∧
  x * y * z * w = 24

def is_permutation_of_1234 (x y z w : ℝ) : Prop :=
  ({x, y, z, w} : Set ℝ) = {1, 2, 3, 4}

theorem solution_characterization :
  ∀ x y z w : ℝ, is_solution x y z w ↔ is_permutation_of_1234 x y z w :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l814_81406


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l814_81460

theorem sum_of_squares_zero_implies_sum (a b c : ℝ) :
  (a - 5)^2 + (b - 6)^2 + (c - 7)^2 = 0 → a + b + c = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l814_81460


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l814_81461

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequences_ratio :
  let num_sum := arithmetic_sum 4 4 72
  let den_sum := arithmetic_sum 5 5 90
  num_sum / den_sum = 76 / 95 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l814_81461


namespace NUMINAMATH_CALUDE_additive_function_properties_l814_81414

/-- A function satisfying f(x + y) = f(x) + f(y) for all real x and y -/
def AdditiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem additive_function_properties (f : ℝ → ℝ) (h : AdditiveFunction f) :
  (∀ x : ℝ, f (-x) = -f x) ∧ f 24 = -8 * f (-3) := by
  sorry

end NUMINAMATH_CALUDE_additive_function_properties_l814_81414


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l814_81472

theorem rectangle_area (square_area : Real) (rectangle_length_factor : Real) : Real :=
  let square_side := Real.sqrt square_area
  let rectangle_width := square_side
  let rectangle_length := rectangle_length_factor * rectangle_width
  rectangle_width * rectangle_length

theorem rectangle_area_proof :
  rectangle_area 36 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l814_81472


namespace NUMINAMATH_CALUDE_length_units_ordering_l814_81443

-- Define an enumeration for length units
inductive LengthUnit
  | Kilometer
  | Meter
  | Centimeter
  | Millimeter

-- Define a function to compare two length units
def isLargerThan (a b : LengthUnit) : Prop :=
  match a, b with
  | LengthUnit.Kilometer, _ => a ≠ b
  | LengthUnit.Meter, LengthUnit.Centimeter => True
  | LengthUnit.Meter, LengthUnit.Millimeter => True
  | LengthUnit.Centimeter, LengthUnit.Millimeter => True
  | _, _ => False

-- Theorem to prove the correct ordering of length units
theorem length_units_ordering :
  isLargerThan LengthUnit.Kilometer LengthUnit.Meter ∧
  isLargerThan LengthUnit.Meter LengthUnit.Centimeter ∧
  isLargerThan LengthUnit.Centimeter LengthUnit.Millimeter :=
by sorry


end NUMINAMATH_CALUDE_length_units_ordering_l814_81443


namespace NUMINAMATH_CALUDE_root_product_l814_81428

theorem root_product (d e : ℤ) : 
  (∀ s : ℂ, s^2 - 2*s - 1 = 0 → s^5 - d*s - e = 0) → 
  d * e = 348 := by
sorry

end NUMINAMATH_CALUDE_root_product_l814_81428


namespace NUMINAMATH_CALUDE_compute_expression_l814_81435

theorem compute_expression : 
  2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / (10 * Real.sqrt 2) = 3 * Real.sqrt 2 / 20 := by
sorry

end NUMINAMATH_CALUDE_compute_expression_l814_81435


namespace NUMINAMATH_CALUDE_total_earnings_proof_l814_81480

structure LaundryShop where
  regular_rate : ℝ
  delicate_rate : ℝ
  bulky_rate : ℝ

def day_earnings (shop : LaundryShop) (regular_kilos delicate_kilos : ℝ) (bulky_items : ℕ) (delicate_discount : ℝ := 0) : ℝ :=
  shop.regular_rate * regular_kilos +
  shop.delicate_rate * delicate_kilos * (1 - delicate_discount) +
  shop.bulky_rate * (bulky_items : ℝ)

theorem total_earnings_proof (shop : LaundryShop)
  (h1 : shop.regular_rate = 3)
  (h2 : shop.delicate_rate = 4)
  (h3 : shop.bulky_rate = 5)
  (h4 : day_earnings shop 7 4 2 = 47)
  (h5 : day_earnings shop 10 6 3 = 69)
  (h6 : day_earnings shop 20 4 0 0.2 = 72.8) :
  day_earnings shop 7 4 2 + day_earnings shop 10 6 3 + day_earnings shop 20 4 0 0.2 = 188.8 := by
  sorry

#eval day_earnings ⟨3, 4, 5⟩ 7 4 2 + day_earnings ⟨3, 4, 5⟩ 10 6 3 + day_earnings ⟨3, 4, 5⟩ 20 4 0 0.2

end NUMINAMATH_CALUDE_total_earnings_proof_l814_81480


namespace NUMINAMATH_CALUDE_prob_same_length_is_one_fifth_l814_81494

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The number of diagonals of each distinct length -/
def num_diagonals_per_length : ℕ := 3

/-- The probability of selecting two elements of the same length from T -/
def prob_same_length : ℚ := sorry

theorem prob_same_length_is_one_fifth :
  prob_same_length = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_same_length_is_one_fifth_l814_81494


namespace NUMINAMATH_CALUDE_mortdecai_donation_l814_81434

/-- Represents the number of eggs in a dozen --/
def eggsPerDozen : ℕ := 12

/-- Represents the number of days Mortdecai collects eggs in a week --/
def collectionDays : ℕ := 2

/-- Represents the number of dozens of eggs Mortdecai collects each collection day --/
def collectedDozens : ℕ := 8

/-- Represents the number of dozens of eggs Mortdecai delivers to the market --/
def marketDeliveryDozens : ℕ := 3

/-- Represents the number of dozens of eggs Mortdecai delivers to the mall --/
def mallDeliveryDozens : ℕ := 5

/-- Represents the number of dozens of eggs Mortdecai uses for pie --/
def pieDozens : ℕ := 4

/-- Calculates the number of eggs Mortdecai donates to charity --/
def donatedEggs : ℕ :=
  (collectedDozens * collectionDays - marketDeliveryDozens - mallDeliveryDozens - pieDozens) * eggsPerDozen

theorem mortdecai_donation :
  donatedEggs = 48 := by
  sorry

end NUMINAMATH_CALUDE_mortdecai_donation_l814_81434


namespace NUMINAMATH_CALUDE_gel_pen_price_ratio_l814_81486

variables (x y : ℕ) (b g : ℝ)

def total_cost := x * b + y * g

theorem gel_pen_price_ratio :
  (∀ (x y : ℕ) (b g : ℝ),
    (x + y) * g = 4 * (x * b + y * g) ∧
    (x + y) * b = (1 / 2) * (x * b + y * g)) →
  g = 8 * b :=
sorry

end NUMINAMATH_CALUDE_gel_pen_price_ratio_l814_81486


namespace NUMINAMATH_CALUDE_tens_digit_of_19_pow_2023_l814_81416

theorem tens_digit_of_19_pow_2023 : ∃ n : ℕ, 19^2023 ≡ 50 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_pow_2023_l814_81416


namespace NUMINAMATH_CALUDE_sculpture_area_is_62_l814_81432

/-- Represents a cube with a given edge length -/
structure Cube where
  edge : ℝ

/-- Represents a layer of the sculpture -/
structure Layer where
  cubes : ℕ
  exposed_top : ℕ
  exposed_sides : ℕ

/-- Represents the entire sculpture -/
structure Sculpture where
  bottom : Layer
  middle : Layer
  top : Layer

/-- Calculates the exposed surface area of a layer -/
def layer_area (c : Cube) (l : Layer) : ℝ :=
  (l.exposed_top + l.exposed_sides) * c.edge ^ 2

/-- Calculates the total exposed surface area of the sculpture -/
def total_area (c : Cube) (s : Sculpture) : ℝ :=
  layer_area c s.bottom + layer_area c s.middle + layer_area c s.top

/-- The sculpture described in the problem -/
def problem_sculpture : Sculpture :=
  { bottom := { cubes := 12, exposed_top := 12, exposed_sides := 24 },
    middle := { cubes := 6, exposed_top := 6, exposed_sides := 10 },
    top := { cubes := 2, exposed_top := 2, exposed_sides := 8 } }

/-- The cube used in the sculpture -/
def unit_cube : Cube :=
  { edge := 1 }

theorem sculpture_area_is_62 :
  total_area unit_cube problem_sculpture = 62 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_area_is_62_l814_81432


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l814_81453

theorem simplify_and_evaluate (x y : ℚ) 
  (hx : x = 1/2) (hy : y = 1/3) : 
  (3*x - y)^2 - (3*x + 2*y)*(3*x - 2*y) = -4/9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l814_81453


namespace NUMINAMATH_CALUDE_valid_triplets_l814_81478

def is_valid_triplet (m n p : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ Nat.Prime p ∧ (Nat.choose m 3 - 4 = p^n)

theorem valid_triplets :
  ∀ m n p : ℕ, is_valid_triplet m n p ↔ (m = 7 ∧ n = 1 ∧ p = 31) ∨ (m = 6 ∧ n = 4 ∧ p = 2) :=
sorry

end NUMINAMATH_CALUDE_valid_triplets_l814_81478


namespace NUMINAMATH_CALUDE_min_box_value_l814_81488

theorem min_box_value (a b : ℤ) (box : ℤ) :
  (∀ x : ℝ, (a * x + b) * (b * x + a) = 30 * x^2 + box * x + 30) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  ∃ (min_box : ℤ), (min_box = 61 ∧ box ≥ min_box) := by
  sorry

end NUMINAMATH_CALUDE_min_box_value_l814_81488


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l814_81420

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ : ℝ} : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + b₁ ∧ y = m₂ * x + b₂) → m₁ = m₂

/-- The first line equation: 3y - 3b = 9x -/
def line1 (b : ℝ) (x y : ℝ) : Prop := 3 * y - 3 * b = 9 * x

/-- The second line equation: y - 2 = (b + 9)x -/
def line2 (b : ℝ) (x y : ℝ) : Prop := y - 2 = (b + 9) * x

theorem parallel_lines_b_value :
  ∀ b : ℝ, (∀ x y : ℝ, line1 b x y ∧ line2 b x y) → b = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l814_81420


namespace NUMINAMATH_CALUDE_days_without_email_is_244_l814_81448

/-- Represents the number of days in a year -/
def days_in_year : ℕ := 365

/-- Represents the email frequency of the first niece -/
def niece1_frequency : ℕ := 4

/-- Represents the email frequency of the second niece -/
def niece2_frequency : ℕ := 6

/-- Represents the email frequency of the third niece -/
def niece3_frequency : ℕ := 8

/-- Calculates the number of days Mr. Thompson did not receive an email from any niece -/
def days_without_email : ℕ :=
  days_in_year - 
  (days_in_year / niece1_frequency + 
   days_in_year / niece2_frequency + 
   days_in_year / niece3_frequency - 
   days_in_year / (niece1_frequency * niece2_frequency) - 
   days_in_year / (niece1_frequency * niece3_frequency) - 
   days_in_year / (niece2_frequency * niece3_frequency) + 
   days_in_year / (niece1_frequency * niece2_frequency * niece3_frequency))

theorem days_without_email_is_244 : days_without_email = 244 := by
  sorry

end NUMINAMATH_CALUDE_days_without_email_is_244_l814_81448


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l814_81470

/-- 
Given an isosceles triangle with height H that is twice as long as 
its projection on the lateral side, prove that its area is H^2 * √3.
-/
theorem isosceles_triangle_area (H : ℝ) (h : H > 0) : 
  let projection := H / 2
  let base := 2 * H * Real.sqrt 3
  let area := (1 / 2) * base * H
  area = H^2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l814_81470


namespace NUMINAMATH_CALUDE_exactly_three_tangent_lines_l814_81487

/-- A line passing through (0, 1) that intersects the parabola y^2 = 4x at only one point -/
structure TangentLine where
  slope : ℝ
  intersects_once : ∃! (x y : ℝ), y^2 = 4*x ∧ y = slope * x + 1

/-- The number of lines passing through (0, 1) that intersect y^2 = 4x at only one point -/
def num_tangent_lines : ℕ := sorry

/-- Theorem stating that there are exactly 3 such lines -/
theorem exactly_three_tangent_lines : num_tangent_lines = 3 := by sorry

end NUMINAMATH_CALUDE_exactly_three_tangent_lines_l814_81487


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l814_81490

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 5*x = 4) ↔ (∃ x : ℝ, x^2 + 5*x ≠ 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l814_81490


namespace NUMINAMATH_CALUDE_min_value_expression_l814_81431

theorem min_value_expression (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x^2 ≥ 4 * (4:ℝ)^(1/5) ∧
  (3 * Real.sqrt x + 4 / x^2 = 4 * (4:ℝ)^(1/5) ↔ x = (4:ℝ)^(2/5)) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l814_81431


namespace NUMINAMATH_CALUDE_one_fourth_of_8_4_l814_81469

theorem one_fourth_of_8_4 : (8.4 : ℚ) / 4 = 21 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_4_l814_81469


namespace NUMINAMATH_CALUDE_equation_graph_is_x_axis_l814_81418

theorem equation_graph_is_x_axis : 
  ∀ (x y : ℝ), (x - y)^2 = x^2 - y^2 - 2*x*y ↔ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_graph_is_x_axis_l814_81418


namespace NUMINAMATH_CALUDE_equipment_maintenance_cost_calculation_l814_81497

def equipment_maintenance_cost (initial_balance cheque_payment received_payment final_balance : ℕ) : ℕ :=
  initial_balance - cheque_payment + received_payment - final_balance

theorem equipment_maintenance_cost_calculation :
  equipment_maintenance_cost 2000 600 800 1000 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_equipment_maintenance_cost_calculation_l814_81497


namespace NUMINAMATH_CALUDE_solution_difference_l814_81496

/-- Given that r and s are distinct solutions to the equation (6x-18)/(x^2+4x-21) = x+3,
    and r > s, prove that r - s = 10. -/
theorem solution_difference (r s : ℝ) : 
  r ≠ s →
  (6*r - 18) / (r^2 + 4*r - 21) = r + 3 →
  (6*s - 18) / (s^2 + 4*s - 21) = s + 3 →
  r > s →
  r - s = 10 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l814_81496


namespace NUMINAMATH_CALUDE_paulo_children_ages_l814_81407

theorem paulo_children_ages :
  ∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 12 ∧ a * b * c = 30 :=
by sorry

end NUMINAMATH_CALUDE_paulo_children_ages_l814_81407


namespace NUMINAMATH_CALUDE_schoolClubProfit_l814_81449

/-- Represents the candy bar sale scenario for a school club -/
structure CandyBarSale where
  totalBars : ℕ
  purchaseRate : ℚ
  regularSellRate : ℚ
  bulkSellRate : ℚ

/-- Calculates the profit for the candy bar sale -/
def calculateProfit (sale : CandyBarSale) : ℚ :=
  let costPerBar := sale.purchaseRate / 4
  let totalCost := costPerBar * sale.totalBars
  let revenuePerBar := sale.regularSellRate / 3
  let totalRevenue := revenuePerBar * sale.totalBars
  totalRevenue - totalCost

/-- The given candy bar sale scenario -/
def schoolClubSale : CandyBarSale :=
  { totalBars := 1200
  , purchaseRate := 3
  , regularSellRate := 2
  , bulkSellRate := 3/5 }

/-- Theorem stating that the profit for the school club is -100 dollars -/
theorem schoolClubProfit :
  calculateProfit schoolClubSale = -100 := by
  sorry


end NUMINAMATH_CALUDE_schoolClubProfit_l814_81449


namespace NUMINAMATH_CALUDE_c_investment_is_10500_l814_81423

/-- Calculates the investment of partner C given the investments of A and B, 
    the total profit, and A's share of the profit. -/
def calculate_c_investment (a_investment b_investment total_profit a_profit : ℚ) : ℚ :=
  (a_investment * total_profit / a_profit) - a_investment - b_investment

/-- Theorem stating that given the specified conditions, C's investment is 10500. -/
theorem c_investment_is_10500 :
  let a_investment : ℚ := 6300
  let b_investment : ℚ := 4200
  let total_profit : ℚ := 12700
  let a_profit : ℚ := 3810
  calculate_c_investment a_investment b_investment total_profit a_profit = 10500 := by
  sorry

#eval calculate_c_investment 6300 4200 12700 3810

end NUMINAMATH_CALUDE_c_investment_is_10500_l814_81423


namespace NUMINAMATH_CALUDE_sin_negative_sixty_degrees_l814_81457

theorem sin_negative_sixty_degrees : Real.sin (-(60 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_sixty_degrees_l814_81457


namespace NUMINAMATH_CALUDE_total_cost_is_17_l814_81427

/-- The total cost of ingredients for Pauline's tacos -/
def total_cost (taco_shells_cost : ℝ) (bell_pepper_cost : ℝ) (bell_pepper_quantity : ℕ) (meat_cost_per_pound : ℝ) (meat_quantity : ℝ) : ℝ :=
  taco_shells_cost + bell_pepper_cost * bell_pepper_quantity + meat_cost_per_pound * meat_quantity

/-- Proof that the total cost of ingredients for Pauline's tacos is $17 -/
theorem total_cost_is_17 :
  total_cost 5 1.5 4 3 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_17_l814_81427


namespace NUMINAMATH_CALUDE_minimal_additional_squares_l814_81465

/-- A point on the grid --/
structure Point where
  x : Nat
  y : Nat

/-- The grid configuration --/
structure Grid where
  size : Nat
  shaded : List Point

/-- Check if a point is within the grid --/
def inGrid (p : Point) (g : Grid) : Prop :=
  p.x < g.size ∧ p.y < g.size

/-- Check if a point is shaded --/
def isShaded (p : Point) (g : Grid) : Prop :=
  p ∈ g.shaded

/-- Reflect a point horizontally --/
def reflectHorizontal (p : Point) (g : Grid) : Point :=
  ⟨p.x, g.size - 1 - p.y⟩

/-- Reflect a point vertically --/
def reflectVertical (p : Point) (g : Grid) : Point :=
  ⟨g.size - 1 - p.x, p.y⟩

/-- Check if the grid has horizontal symmetry --/
def hasHorizontalSymmetry (g : Grid) : Prop :=
  ∀ p, inGrid p g → (isShaded p g ↔ isShaded (reflectHorizontal p g) g)

/-- Check if the grid has vertical symmetry --/
def hasVerticalSymmetry (g : Grid) : Prop :=
  ∀ p, inGrid p g → (isShaded p g ↔ isShaded (reflectVertical p g) g)

/-- The initial grid configuration --/
def initialGrid : Grid :=
  { size := 6
  , shaded := [⟨0,5⟩, ⟨2,3⟩, ⟨3,2⟩, ⟨5,0⟩] }

/-- The theorem to prove --/
theorem minimal_additional_squares :
  ∃ (additionalSquares : List Point),
    additionalSquares.length = 1 ∧
    let newGrid : Grid := { size := initialGrid.size, shaded := initialGrid.shaded ++ additionalSquares }
    hasHorizontalSymmetry newGrid ∧ hasVerticalSymmetry newGrid ∧
    ∀ (otherSquares : List Point),
      otherSquares.length < additionalSquares.length →
      let otherGrid : Grid := { size := initialGrid.size, shaded := initialGrid.shaded ++ otherSquares }
      ¬(hasHorizontalSymmetry otherGrid ∧ hasVerticalSymmetry otherGrid) :=
by sorry

end NUMINAMATH_CALUDE_minimal_additional_squares_l814_81465


namespace NUMINAMATH_CALUDE_min_dials_for_lighting_l814_81451

/-- Represents a stack of 12-sided dials -/
def DialStack := ℕ → Fin 12 → Fin 12

/-- The sum of numbers in a column of the dial stack -/
def columnSum (stack : DialStack) (column : Fin 12) : ℕ :=
  sorry

/-- Predicate that checks if all column sums have the same remainder mod 12 -/
def allColumnSumsEqualMod12 (stack : DialStack) : Prop :=
  ∀ i j : Fin 12, columnSum stack i % 12 = columnSum stack j % 12

/-- The minimum number of dials required for the Christmas tree to light up -/
theorem min_dials_for_lighting : 
  ∃ (n : ℕ), n = 12 ∧ 
  (∃ (stack : DialStack), (∀ i : ℕ, i < n → ∃ (dial : Fin 12 → Fin 12), stack i = dial) ∧ 
   allColumnSumsEqualMod12 stack) ∧
  (∀ (m : ℕ), m < n → 
   ∀ (stack : DialStack), (∀ i : ℕ, i < m → ∃ (dial : Fin 12 → Fin 12), stack i = dial) → 
   ¬allColumnSumsEqualMod12 stack) :=
sorry

end NUMINAMATH_CALUDE_min_dials_for_lighting_l814_81451


namespace NUMINAMATH_CALUDE_teaspoons_per_tablespoon_l814_81477

/-- Given the following definitions:
  * One cup contains 480 grains of rice
  * Half a cup is 8 tablespoons
  * One teaspoon contains 10 grains of rice
  Prove that there are 3 teaspoons in one tablespoon -/
theorem teaspoons_per_tablespoon 
  (grains_per_cup : ℕ) 
  (tablespoons_per_half_cup : ℕ) 
  (grains_per_teaspoon : ℕ) 
  (h1 : grains_per_cup = 480)
  (h2 : tablespoons_per_half_cup = 8)
  (h3 : grains_per_teaspoon = 10) : 
  (grains_per_cup / 2) / tablespoons_per_half_cup / grains_per_teaspoon = 3 :=
by sorry

end NUMINAMATH_CALUDE_teaspoons_per_tablespoon_l814_81477


namespace NUMINAMATH_CALUDE_pool_filling_buckets_l814_81485

theorem pool_filling_buckets 
  (george_buckets : ℕ) 
  (harry_buckets : ℕ) 
  (total_rounds : ℕ) :
  george_buckets = 2 →
  harry_buckets = 3 →
  total_rounds = 22 →
  (george_buckets + harry_buckets) * total_rounds = 110 := by
sorry

end NUMINAMATH_CALUDE_pool_filling_buckets_l814_81485


namespace NUMINAMATH_CALUDE_shopping_spending_l814_81466

/-- The total spending of Elizabeth, Emma, and Elsa given their spending relationships -/
theorem shopping_spending (emma_spending : ℕ) : emma_spending = 58 →
  (emma_spending + 2 * emma_spending + 4 * (2 * emma_spending) = 638) := by
  sorry

#check shopping_spending

end NUMINAMATH_CALUDE_shopping_spending_l814_81466


namespace NUMINAMATH_CALUDE_weight_of_b_l814_81409

/-- Given three weights a, b, and c, prove that b = 33 under the given conditions. -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 41 →
  (b + c) / 2 = 43 →
  b = 33 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l814_81409


namespace NUMINAMATH_CALUDE_root_difference_indeterminate_l814_81402

/-- A function with the property f(1 + x) = f(1 - x) for all real x -/
def symmetric_around_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 + x) = f (1 - x)

/-- A function has exactly two distinct real roots -/
def has_two_distinct_roots (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧
  ∀ z : ℝ, f z = 0 → z = x ∨ z = y

theorem root_difference_indeterminate (f : ℝ → ℝ) 
  (h1 : symmetric_around_one f) 
  (h2 : has_two_distinct_roots f) : 
  ¬ ∃ d : ℝ, ∀ x y : ℝ, f x = 0 → f y = 0 → x ≠ y → |x - y| = d :=
sorry

end NUMINAMATH_CALUDE_root_difference_indeterminate_l814_81402


namespace NUMINAMATH_CALUDE_multiple_problem_l814_81415

theorem multiple_problem (m : ℤ) : 17 = m * (2625 / 1000) - 4 ↔ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l814_81415


namespace NUMINAMATH_CALUDE_quadratic_roots_and_triangle_l814_81421

-- Define the quadratic equation
def quadratic_eq (k x : ℝ) : Prop := x^2 - (2*k + 1)*x + k^2 + k = 0

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + k)

-- Define a right triangle with sides a, b, c
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

-- Main theorem
theorem quadratic_roots_and_triangle (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_eq k x ∧ quadratic_eq k y) ∧
  (∃ a b : ℝ, quadratic_eq k a ∧ quadratic_eq k b ∧ is_right_triangle a b 5 → k = 3 ∨ k = 12) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_triangle_l814_81421


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l814_81436

/-- Given a hyperbola with equation x²/a² - y²/3 = 1 (a > 0) that passes through (-2, 0),
    its eccentricity is √7/2 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) :
  (4 / a^2 - 0 / 3 = 1) →
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l814_81436


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l814_81426

theorem quadratic_equation_m_value (x₁ x₂ m : ℝ) : 
  (∀ x, x^2 - 4*x + m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + 3*x₂ = 5 →
  m = 7/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l814_81426


namespace NUMINAMATH_CALUDE_trigonometric_identity_l814_81467

theorem trigonometric_identity :
  Real.cos (17 * π / 180) * Real.sin (43 * π / 180) +
  Real.sin (163 * π / 180) * Real.sin (47 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l814_81467


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l814_81438

theorem max_digits_product_5_4 : 
  ∃ (a b : ℕ), 
    (10000 ≤ a ∧ a < 100000) ∧ 
    (1000 ≤ b ∧ b < 10000) ∧ 
    (∀ x y : ℕ, (10000 ≤ x ∧ x < 100000) → (1000 ≤ y ∧ y < 10000) → x * y ≤ a * b) ∧
    (Nat.digits 10 (a * b)).length = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l814_81438


namespace NUMINAMATH_CALUDE_least_cans_for_given_volumes_l814_81405

/-- The least number of cans required to pack drinks -/
def leastCans (maaza pepsi sprite : ℕ) : ℕ :=
  let gcd := Nat.gcd (Nat.gcd maaza pepsi) sprite
  (maaza / gcd) + (pepsi / gcd) + (sprite / gcd)

/-- Theorem stating the least number of cans required for given volumes -/
theorem least_cans_for_given_volumes :
  leastCans 40 144 368 = 69 := by
  sorry

end NUMINAMATH_CALUDE_least_cans_for_given_volumes_l814_81405


namespace NUMINAMATH_CALUDE_remainder_divisibility_l814_81476

theorem remainder_divisibility (N : ℤ) : 
  ∃ k : ℤ, N = 142 * k + 110 → ∃ m : ℤ, N = 14 * m + 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l814_81476


namespace NUMINAMATH_CALUDE_smallest_k_for_zero_diff_l814_81433

def u (n : ℕ) : ℕ := n^3 + 2*n^2 + n

def diff (f : ℕ → ℕ) (n : ℕ) : ℕ := f (n + 1) - f n

def diff_k (f : ℕ → ℕ) : ℕ → (ℕ → ℕ)
  | 0 => f
  | k + 1 => diff (diff_k f k)

theorem smallest_k_for_zero_diff (n : ℕ) : 
  (∀ n, diff_k u 4 n = 0) ∧ 
  (∀ k < 4, ∃ n, diff_k u k n ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_zero_diff_l814_81433


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l814_81481

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 13 = 40) →
  (a 8 + a 9 + a 10 = 60) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l814_81481


namespace NUMINAMATH_CALUDE_sum_of_terms_l814_81450

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_terms (a : ℕ → ℕ) : 
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 31 →
  a 4 + a 5 + a 7 = 93 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_terms_l814_81450


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_probability_l814_81440

/-- Represents an isosceles right-angled triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The probability of choosing a point within distance 1 from the right angle -/
def probability_within_distance (t : IsoscelesRightTriangle) : ℝ :=
  sorry

theorem isosceles_right_triangle_probability 
  (t : IsoscelesRightTriangle) 
  (h : t.leg_length = 2) : 
  probability_within_distance t = π / 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_probability_l814_81440


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l814_81475

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_decr : ∀ n, a (n + 1) < a n)
  (h_geom : geometric_sequence a)
  (h_prod : a 2 * a 8 = 6)
  (h_sum : a 4 + a 6 = 5) :
  a 5 / a 7 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l814_81475


namespace NUMINAMATH_CALUDE_min_contestants_solved_all_l814_81468

theorem min_contestants_solved_all (total : ℕ) (solved1 solved2 solved3 solved4 : ℕ) 
  (h_total : total = 100)
  (h_solved1 : solved1 = 90)
  (h_solved2 : solved2 = 85)
  (h_solved3 : solved3 = 80)
  (h_solved4 : solved4 = 75) :
  ∃ (min_solved_all : ℕ), 
    min_solved_all ≤ solved1 ∧
    min_solved_all ≤ solved2 ∧
    min_solved_all ≤ solved3 ∧
    min_solved_all ≤ solved4 ∧
    min_solved_all ≥ solved1 + solved2 + solved3 + solved4 - 3 * total ∧
    min_solved_all = 30 :=
sorry

end NUMINAMATH_CALUDE_min_contestants_solved_all_l814_81468


namespace NUMINAMATH_CALUDE_neighbor_took_twelve_eggs_l814_81445

/-- Represents the egg-laying scenario with Myrtle's hens -/
structure EggScenario where
  hens : ℕ
  eggs_per_hen_per_day : ℕ
  days_gone : ℕ
  dropped_eggs : ℕ
  remaining_eggs : ℕ

/-- Calculates the number of eggs taken by the neighbor -/
def neighbor_eggs (scenario : EggScenario) : ℕ :=
  scenario.hens * scenario.eggs_per_hen_per_day * scenario.days_gone -
  (scenario.remaining_eggs + scenario.dropped_eggs)

/-- Theorem stating that the neighbor took 12 eggs -/
theorem neighbor_took_twelve_eggs :
  let scenario : EggScenario := {
    hens := 3,
    eggs_per_hen_per_day := 3,
    days_gone := 7,
    dropped_eggs := 5,
    remaining_eggs := 46
  }
  neighbor_eggs scenario = 12 := by sorry

end NUMINAMATH_CALUDE_neighbor_took_twelve_eggs_l814_81445


namespace NUMINAMATH_CALUDE_fish_count_l814_81408

theorem fish_count (num_bowls : ℕ) (fish_per_bowl : ℕ) (h1 : num_bowls = 261) (h2 : fish_per_bowl = 23) :
  num_bowls * fish_per_bowl = 6003 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l814_81408


namespace NUMINAMATH_CALUDE_meals_neither_kosher_nor_vegan_l814_81464

/-- Proves the number of meals that are neither kosher nor vegan -/
theorem meals_neither_kosher_nor_vegan 
  (total_clients : ℕ) 
  (vegan_meals : ℕ) 
  (kosher_meals : ℕ) 
  (both_vegan_and_kosher : ℕ) 
  (h1 : total_clients = 30)
  (h2 : vegan_meals = 7)
  (h3 : kosher_meals = 8)
  (h4 : both_vegan_and_kosher = 3) :
  total_clients - (vegan_meals + kosher_meals - both_vegan_and_kosher) = 18 :=
by
  sorry

#check meals_neither_kosher_nor_vegan

end NUMINAMATH_CALUDE_meals_neither_kosher_nor_vegan_l814_81464


namespace NUMINAMATH_CALUDE_max_value_of_e_l814_81444

def b (n : ℕ) : ℕ := (10^n - 9) / 3

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n+1))

theorem max_value_of_e :
  (∀ n : ℕ, e n ≤ 3) ∧ (∃ n : ℕ, e n = 3) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_e_l814_81444


namespace NUMINAMATH_CALUDE_negative_quarter_and_negative_four_power_l814_81403

theorem negative_quarter_and_negative_four_power :
  (-0.25)^11 * (-4)^12 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_quarter_and_negative_four_power_l814_81403


namespace NUMINAMATH_CALUDE_total_questions_answered_l814_81493

/-- Represents a tour group with the number of tourists asking different amounts of questions -/
structure TourGroup where
  usual : ℕ  -- number of tourists asking the usual 2 questions
  zero : ℕ   -- number of tourists asking 0 questions
  one : ℕ    -- number of tourists asking 1 question
  three : ℕ  -- number of tourists asking 3 questions
  five : ℕ   -- number of tourists asking 5 questions
  double : ℕ -- number of tourists asking double the usual (4 questions)
  triple : ℕ -- number of tourists asking triple the usual (6 questions)
  quad : ℕ   -- number of tourists asking quadruple the usual (8 questions)

/-- Calculates the total number of questions for a tour group -/
def questionsForGroup (g : TourGroup) : ℕ :=
  2 * g.usual + 0 * g.zero + 1 * g.one + 3 * g.three + 5 * g.five +
  4 * g.double + 6 * g.triple + 8 * g.quad

/-- The six tour groups as described in the problem -/
def tourGroups : List TourGroup := [
  ⟨3, 0, 2, 0, 1, 0, 0, 0⟩,  -- Group A
  ⟨4, 1, 0, 6, 0, 0, 0, 0⟩,  -- Group B
  ⟨4, 2, 1, 0, 0, 0, 1, 0⟩,  -- Group C
  ⟨3, 1, 0, 0, 0, 0, 0, 1⟩,  -- Group D
  ⟨3, 2, 0, 0, 1, 3, 0, 0⟩,  -- Group E
  ⟨4, 1, 0, 2, 0, 0, 0, 0⟩   -- Group F
]

theorem total_questions_answered (groups := tourGroups) :
  (groups.map questionsForGroup).sum = 105 := by sorry

end NUMINAMATH_CALUDE_total_questions_answered_l814_81493


namespace NUMINAMATH_CALUDE_picture_processing_time_l814_81498

/-- Given 960 pictures and a processing time of 2 minutes per picture, 
    the total processing time in hours is equal to 32. -/
theorem picture_processing_time : 
  let num_pictures : ℕ := 960
  let processing_time_per_picture : ℕ := 2
  let minutes_per_hour : ℕ := 60
  (num_pictures * processing_time_per_picture) / minutes_per_hour = 32 := by
sorry

end NUMINAMATH_CALUDE_picture_processing_time_l814_81498


namespace NUMINAMATH_CALUDE_inequality_range_l814_81452

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2*x + 1 - a^2 < 0) → 
  (a > 3 ∨ a < -3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l814_81452


namespace NUMINAMATH_CALUDE_quadratic_expression_sum_l814_81410

theorem quadratic_expression_sum (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ), (15 * d + 16 + 17 * d^2) + (3 * d + 2) = a * d^2 + b * d + c ∧ a + b + c = 53 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_sum_l814_81410


namespace NUMINAMATH_CALUDE_square_side_length_l814_81495

/-- The side length of a square with area equal to a 3 cm × 27 cm rectangle is 9 cm. -/
theorem square_side_length (square_area rectangle_area : ℝ) (square_side : ℝ) : 
  square_area = rectangle_area →
  rectangle_area = 3 * 27 →
  square_area = square_side ^ 2 →
  square_side = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l814_81495


namespace NUMINAMATH_CALUDE_log_expression_equality_l814_81492

theorem log_expression_equality : 
  2 * Real.log 10 / Real.log 5 + Real.log (1/4) / Real.log 5 + (2 : ℝ) ^ (Real.log 3 / Real.log 4) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l814_81492


namespace NUMINAMATH_CALUDE_nested_root_simplification_l814_81422

theorem nested_root_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x^9)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l814_81422


namespace NUMINAMATH_CALUDE_unique_root_range_l814_81411

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a * Real.exp x else -Real.log x

theorem unique_root_range (a : ℝ) :
  (∃! x, f a (f a x) = 0) → a ∈ Set.Ioi 0 ∪ Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_unique_root_range_l814_81411


namespace NUMINAMATH_CALUDE_mushroom_collection_problem_l814_81463

/-- Represents the mushroom distribution pattern for a given girl --/
def mushroom_distribution (total : ℕ) (girl_number : ℕ) : ℚ :=
  (girl_number + 19) + 0.04 * (total - (girl_number + 19))

/-- Theorem stating the solution to the mushroom collection problem --/
theorem mushroom_collection_problem :
  ∃ (n : ℕ) (total : ℕ), 
    (∀ i j, i ≤ n → j ≤ n → mushroom_distribution total i = mushroom_distribution total j) ∧
    n = 5 ∧
    total = 120 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_problem_l814_81463


namespace NUMINAMATH_CALUDE_unique_value_at_half_l814_81473

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x * y + f x) = 2 * x * f y + f x

theorem unique_value_at_half (f : ℝ → ℝ) (hf : special_function f) :
  ∃! v : ℝ, f (1/2) = v ∧ v = -1 :=
sorry

end NUMINAMATH_CALUDE_unique_value_at_half_l814_81473


namespace NUMINAMATH_CALUDE_parallelogram_has_multiple_altitudes_l814_81456

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- An altitude of a parallelogram is a line segment from a vertex perpendicular to the opposite side or its extension. -/
structure Altitude (p : Parallelogram) where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- A parallelogram has more than one altitude. -/
theorem parallelogram_has_multiple_altitudes (p : Parallelogram) : ∃ (a b : Altitude p), a ≠ b := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_has_multiple_altitudes_l814_81456


namespace NUMINAMATH_CALUDE_novel_pages_count_prove_novel_pages_l814_81419

theorem novel_pages_count : ℕ → Prop :=
  fun total_pages =>
    let day1_read := total_pages / 6 + 10
    let day1_remaining := total_pages - day1_read
    let day2_read := day1_remaining / 5 + 20
    let day2_remaining := day1_remaining - day2_read
    let day3_read := day2_remaining / 4 + 25
    let day3_remaining := day2_remaining - day3_read
    day3_remaining = 80 ∧ total_pages = 252

theorem prove_novel_pages : novel_pages_count 252 := by
  sorry

end NUMINAMATH_CALUDE_novel_pages_count_prove_novel_pages_l814_81419


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_neg_one_l814_81417

theorem sqrt_meaningful_iff_geq_neg_one (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_neg_one_l814_81417


namespace NUMINAMATH_CALUDE_rectangle_area_l814_81482

theorem rectangle_area (x : ℝ) : 
  x > 0 → 
  ∃ w l : ℝ, w > 0 ∧ l > 0 ∧ 
  l = 3 * w ∧ 
  x^2 = l^2 + w^2 ∧
  w * l = (3/10) * x^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l814_81482


namespace NUMINAMATH_CALUDE_hungry_bear_purchase_cost_l814_81458

/-- Represents the cost of items at Hungry Bear Diner -/
structure DinerCost where
  sandwich_price : ℕ
  soda_price : ℕ
  cookie_price : ℕ

/-- Calculates the total cost of a purchase at Hungry Bear Diner -/
def total_cost (prices : DinerCost) (num_sandwiches num_sodas num_cookies : ℕ) : ℕ :=
  prices.sandwich_price * num_sandwiches +
  prices.soda_price * num_sodas +
  prices.cookie_price * num_cookies

/-- Theorem stating that the total cost of 3 sandwiches, 5 sodas, and 4 cookies is $35 -/
theorem hungry_bear_purchase_cost :
  ∃ (prices : DinerCost),
    prices.sandwich_price = 4 ∧
    prices.soda_price = 3 ∧
    prices.cookie_price = 2 ∧
    total_cost prices 3 5 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_hungry_bear_purchase_cost_l814_81458


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l814_81442

-- Define an isosceles triangle with sides a, b, and c
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)
  validTriangle : a + b > c ∧ b + c > a ∧ a + c > b

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle,
    (t.a = 4 ∧ t.b = 7) ∨ (t.a = 7 ∧ t.b = 4) →
    perimeter t = 15 ∨ perimeter t = 18 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l814_81442


namespace NUMINAMATH_CALUDE_arc_length_30_degree_sector_l814_81471

/-- The length of an arc in a sector with radius 1 cm and central angle 30° is π/6 cm. -/
theorem arc_length_30_degree_sector (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 1 → θ = 30 * π / 180 → l = r * θ → l = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_30_degree_sector_l814_81471


namespace NUMINAMATH_CALUDE_cyclic_fraction_product_l814_81479

theorem cyclic_fraction_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_fraction_product_l814_81479


namespace NUMINAMATH_CALUDE_line_point_k_value_l814_81401

/-- Given a line containing points (7, 10), (1, k), and (-5, 3), prove that k = 6.5 -/
theorem line_point_k_value : ∀ k : ℝ,
  (∃ (line : Set (ℝ × ℝ)),
    (7, 10) ∈ line ∧ (1, k) ∈ line ∧ (-5, 3) ∈ line ∧
    (∀ p q r : ℝ × ℝ, p ∈ line → q ∈ line → r ∈ line →
      (p.2 - q.2) * (q.1 - r.1) = (q.2 - r.2) * (p.1 - q.1))) →
  k = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_line_point_k_value_l814_81401


namespace NUMINAMATH_CALUDE_new_triangle_is_right_triangle_l814_81462

/-- Given a right triangle with legs a and b, hypotenuse c, and altitude h on the hypotenuse,
    prove that the triangle formed by sides c+h, a+b, and h is also a right triangle. -/
theorem new_triangle_is_right_triangle
  (a b c h : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (altitude_relation : a * b = c * h)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (h_pos : 0 < h) :
  (c + h)^2 = (a + b)^2 + h^2 :=
sorry

end NUMINAMATH_CALUDE_new_triangle_is_right_triangle_l814_81462


namespace NUMINAMATH_CALUDE_box_width_is_48_l814_81425

/-- Represents the dimensions of a box and the number of cubes that fill it -/
structure BoxWithCubes where
  length : ℕ
  width : ℕ
  depth : ℕ
  num_cubes : ℕ

/-- The box is completely filled by the cubes -/
def is_filled (box : BoxWithCubes) : Prop :=
  ∃ (cube_side : ℕ), 
    cube_side > 0 ∧
    box.length % cube_side = 0 ∧
    box.width % cube_side = 0 ∧
    box.depth % cube_side = 0 ∧
    box.length * box.width * box.depth = box.num_cubes * (cube_side ^ 3)

/-- The main theorem: if a box with given dimensions is filled with 80 cubes, its width is 48 inches -/
theorem box_width_is_48 (box : BoxWithCubes) : 
  box.length = 30 → box.depth = 12 → box.num_cubes = 80 → is_filled box → box.width = 48 := by
  sorry

end NUMINAMATH_CALUDE_box_width_is_48_l814_81425


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l814_81447

/-- The slope of the original line -/
def original_slope : ℚ := 3 / 2

/-- The slope of the perpendicular line -/
def perpendicular_slope : ℚ := -2 / 3

/-- The y-intercept of the perpendicular line -/
def y_intercept : ℚ := 5

/-- The x-intercept of the perpendicular line -/
def x_intercept : ℚ := 15 / 2

theorem perpendicular_line_x_intercept :
  let line := fun (x : ℚ) => perpendicular_slope * x + y_intercept
  (∀ x, line x = 0 ↔ x = x_intercept) ∧
  perpendicular_slope * original_slope = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l814_81447


namespace NUMINAMATH_CALUDE_count_valid_sequences_l814_81499

/-- Represents a sequence of non-negative integers -/
def Sequence := ℕ → ℕ

/-- Checks if a sequence satisfies the given conditions -/
def ValidSequence (a : Sequence) : Prop :=
  a 0 = 2016 ∧
  (∀ n, a (n + 1) ≤ Real.sqrt (a n)) ∧
  (∀ m n, m ≠ n → a m ≠ a n)

/-- Counts the number of valid sequences -/
def CountValidSequences : ℕ := sorry

/-- The main theorem stating that the count of valid sequences is 948 -/
theorem count_valid_sequences :
  CountValidSequences = 948 := by sorry

end NUMINAMATH_CALUDE_count_valid_sequences_l814_81499


namespace NUMINAMATH_CALUDE_max_piles_is_30_l814_81454

/-- Represents a configuration of stone piles -/
structure StonePiles where
  piles : List Nat
  sum_stones : (piles.sum = 660)
  valid_ratio : ∀ i j, i < piles.length → j < piles.length → 2 * piles[i]! > piles[j]!

/-- The maximum number of piles that can be formed -/
def max_piles : Nat := 30

/-- Theorem stating that 30 is the maximum number of piles -/
theorem max_piles_is_30 :
  ∀ sp : StonePiles, sp.piles.length ≤ max_piles :=
by sorry

end NUMINAMATH_CALUDE_max_piles_is_30_l814_81454


namespace NUMINAMATH_CALUDE_madeline_homework_hours_l814_81412

/-- Calculates the number of hours Madeline spends on homework per day -/
theorem madeline_homework_hours (class_hours_per_week : ℕ) 
                                 (sleep_hours_per_day : ℕ) 
                                 (work_hours_per_week : ℕ) 
                                 (leftover_hours : ℕ) 
                                 (days_per_week : ℕ) 
                                 (hours_per_day : ℕ) :
  class_hours_per_week = 18 →
  sleep_hours_per_day = 8 →
  work_hours_per_week = 20 →
  leftover_hours = 46 →
  days_per_week = 7 →
  hours_per_day = 24 →
  (hours_per_day * days_per_week - 
   (class_hours_per_week + sleep_hours_per_day * days_per_week + 
    work_hours_per_week + leftover_hours)) / days_per_week = 4 := by
  sorry

end NUMINAMATH_CALUDE_madeline_homework_hours_l814_81412


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l814_81413

theorem slope_angle_of_line (x y : ℝ) : 
  x - y + 3 = 0 → Real.arctan 1 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l814_81413


namespace NUMINAMATH_CALUDE_trapezoid_DG_length_l814_81439

-- Define the trapezoids and their properties
structure Trapezoid where
  BC : ℝ
  AD : ℝ
  CT : ℝ
  TD : ℝ
  DG : ℝ

-- Define the theorem
theorem trapezoid_DG_length (ABCD AEFG : Trapezoid) : 
  ABCD.BC = 4 →
  ABCD.AD = 7 →
  ABCD.CT = 1 →
  ABCD.TD = 2 →
  -- ABCD and AEFG are right trapezoids with BC ∥ EF and CD ∥ FG (assumed)
  -- ABCD and AEFG have the same area (assumed)
  AEFG.DG = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_DG_length_l814_81439


namespace NUMINAMATH_CALUDE_convex_ngon_divided_into_equal_triangles_l814_81489

/-- A convex n-gon that is circumscribed and divided into equal triangles by non-intersecting diagonals -/
structure ConvexNGon (n : ℕ) :=
  (convex : Bool)
  (circumscribed : Bool)
  (equal_triangles : Bool)
  (non_intersecting_diagonals : Bool)

/-- Theorem stating that the only possible value for n is 4 -/
theorem convex_ngon_divided_into_equal_triangles
  (n : ℕ) (ngon : ConvexNGon n) (h1 : n > 3)
  (h2 : ngon.convex = true)
  (h3 : ngon.circumscribed = true)
  (h4 : ngon.equal_triangles = true)
  (h5 : ngon.non_intersecting_diagonals = true) :
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_convex_ngon_divided_into_equal_triangles_l814_81489


namespace NUMINAMATH_CALUDE_angle_through_point_l814_81491

theorem angle_through_point (α : Real) :
  0 ≤ α → α < 2 * Real.pi →
  let P : ℝ × ℝ := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  (Real.cos α = P.1 ∧ Real.sin α = P.2) →
  α = 11 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_through_point_l814_81491


namespace NUMINAMATH_CALUDE_infinite_series_sum_l814_81441

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 55/12 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n - 2) / (n * (n + 1) * (n + 3))) = 55 / 12 :=
by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l814_81441


namespace NUMINAMATH_CALUDE_modulo_17_residue_l814_81455

theorem modulo_17_residue : (342 + 6 * 47 + 8 * 157 + 3^3 * 21) % 17 = 10 := by
  sorry

end NUMINAMATH_CALUDE_modulo_17_residue_l814_81455


namespace NUMINAMATH_CALUDE_zero_is_monomial_l814_81429

/-- A monomial is a polynomial with a single term. -/
def IsMonomial (p : Polynomial ℝ) : Prop :=
  ∃ c a, p = c * Polynomial.X ^ a

/-- Zero is a monomial. -/
theorem zero_is_monomial : IsMonomial (0 : Polynomial ℝ) := by
  sorry

end NUMINAMATH_CALUDE_zero_is_monomial_l814_81429


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l814_81437

theorem smallest_distance_between_complex_numbers (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 4*I) = 2)
  (hw : Complex.abs (w - 6 - 7*I) = 4) :
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), Complex.abs (z' + 2 + 4*I) = 2 → Complex.abs (w' - 6 - 7*I) = 4 
      → Complex.abs (z' - w') ≥ min_dist) ∧ 
    (∃ (z₀ w₀ : ℂ), Complex.abs (z₀ + 2 + 4*I) = 2 ∧ Complex.abs (w₀ - 6 - 7*I) = 4 
      ∧ Complex.abs (z₀ - w₀) = min_dist) ∧
    min_dist = Real.sqrt 185 - 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l814_81437


namespace NUMINAMATH_CALUDE_even_function_order_l814_81474

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem even_function_order (f : ℝ → ℝ) (h1 : is_even f) 
  (h2 : ∀ x, f (2 + x) = f (2 - x)) 
  (h3 : is_monotone_decreasing f (-2) 0) :
  f 5 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f (-1.5) := by
  sorry

end NUMINAMATH_CALUDE_even_function_order_l814_81474


namespace NUMINAMATH_CALUDE_no_solutions_exist_l814_81400

theorem no_solutions_exist : ¬∃ (a b : ℕ), 2019 * a^2018 = 2017 + b^2016 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l814_81400


namespace NUMINAMATH_CALUDE_x_squared_plus_3x_plus_4_range_l814_81430

theorem x_squared_plus_3x_plus_4_range :
  ∀ x : ℝ, x^2 - 8*x + 15 < 0 → 22 < x^2 + 3*x + 4 ∧ x^2 + 3*x + 4 < 44 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_3x_plus_4_range_l814_81430
