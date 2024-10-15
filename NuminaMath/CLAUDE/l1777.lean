import Mathlib

namespace NUMINAMATH_CALUDE_removed_triangles_area_l1777_177730

theorem removed_triangles_area (s : ℝ) (h1 : s > 0) : 
  let x := (s - 8) / 2
  4 * (1/2 * x^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l1777_177730


namespace NUMINAMATH_CALUDE_disconnected_circuit_scenarios_l1777_177731

/-- Represents a circuit with solder points -/
structure Circuit where
  total_points : ℕ
  is_disconnected : Bool

/-- Calculates the number of scenarios where solder points can fall off -/
def scenarios_with_fallen_points (c : Circuit) : ℕ :=
  2^c.total_points - 1

/-- Theorem: For a disconnected circuit with 6 solder points, there are 63 scenarios of fallen points -/
theorem disconnected_circuit_scenarios :
  ∀ (c : Circuit), c.total_points = 6 → c.is_disconnected = true →
  scenarios_with_fallen_points c = 63 := by
  sorry

#check disconnected_circuit_scenarios

end NUMINAMATH_CALUDE_disconnected_circuit_scenarios_l1777_177731


namespace NUMINAMATH_CALUDE_raul_shopping_spree_l1777_177707

def initial_amount : ℚ := 87
def comic_price : ℚ := 4
def comic_quantity : ℕ := 8
def novel_price : ℚ := 7
def novel_quantity : ℕ := 3
def magazine_price : ℚ := 5.5
def magazine_quantity : ℕ := 2

def total_spent : ℚ :=
  comic_price * comic_quantity +
  novel_price * novel_quantity +
  magazine_price * magazine_quantity

def remaining_amount : ℚ := initial_amount - total_spent

theorem raul_shopping_spree :
  remaining_amount = 23 := by sorry

end NUMINAMATH_CALUDE_raul_shopping_spree_l1777_177707


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l1777_177746

theorem rhombus_diagonal (area : ℝ) (ratio_long : ℝ) (ratio_short : ℝ) :
  area = 135 →
  ratio_long = 5 →
  ratio_short = 3 →
  (ratio_long * ratio_short * (longer_diagonal ^ 2)) / (2 * (ratio_long ^ 2)) = area →
  longer_diagonal = 15 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l1777_177746


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l1777_177751

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse a b) : ℝ := sorry

/-- The right focus of an ellipse -/
def right_focus (e : Ellipse a b) : Point := sorry

/-- The intersection points of a line and an ellipse -/
def intersection_points (e : Ellipse a b) (l : Line) : (Point × Point) := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if two line segments are perpendicular -/
def perpendicular (p1 p2 p3 : Point) : Prop := sorry

theorem ellipse_eccentricity_theorem 
  (a b : ℝ) (e : Ellipse a b) (l : Line) :
  let F := right_focus e
  let (A, B) := intersection_points e l
  perpendicular A F B ∧ distance A F = 3 * distance B F →
  eccentricity e = Real.sqrt 10 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l1777_177751


namespace NUMINAMATH_CALUDE_exactly_three_correct_implies_B_false_l1777_177714

-- Define the function f over ℝ
variable (f : ℝ → ℝ)

-- Define the properties stated by each student
def property_A (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≥ f y

def property_B (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≤ f y

def property_C (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + (1 - x)) = f x

def property_D (f : ℝ → ℝ) : Prop :=
  ∃ x, f x < f 0

-- Theorem stating that if exactly three properties are true, then B must be false
theorem exactly_three_correct_implies_B_false (f : ℝ → ℝ) :
  ((property_A f ∧ property_C f ∧ property_D f) ∨
   (property_A f ∧ property_B f ∧ property_C f) ∨
   (property_A f ∧ property_B f ∧ property_D f) ∨
   (property_B f ∧ property_C f ∧ property_D f)) →
  ¬ property_B f :=
sorry

end NUMINAMATH_CALUDE_exactly_three_correct_implies_B_false_l1777_177714


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l1777_177767

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l1777_177767


namespace NUMINAMATH_CALUDE_non_negativity_and_extrema_l1777_177756

theorem non_negativity_and_extrema :
  (∀ x y : ℝ, (x - 1)^2 ≥ 0 ∧ x^2 + 1 > 0 ∧ |3*x + 2*y| ≥ 0) ∧
  (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0 ∧ (∃ x₀ : ℝ, x₀^2 - 2*x₀ + 1 = 0)) ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 + x*y →
    (x - 3*y)^2 + 4*(y + x)*(x - y) ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_non_negativity_and_extrema_l1777_177756


namespace NUMINAMATH_CALUDE_line_segment_parameterization_l1777_177705

/-- Given a line segment connecting points (1,3) and (4,9), parameterized by x = at + b and y = ct + d,
    where t = 0 corresponds to (1,3) and t = 1 corresponds to (4,9),
    prove that a^2 + b^2 + c^2 + d^2 = 55. -/
theorem line_segment_parameterization (a b c d : ℝ) : 
  (∀ t : ℝ, (a * t + b, c * t + d) = (1 - t, 3 - 3*t) + t • (4, 9)) →
  a^2 + b^2 + c^2 + d^2 = 55 := by
sorry

end NUMINAMATH_CALUDE_line_segment_parameterization_l1777_177705


namespace NUMINAMATH_CALUDE_tshirt_shop_weekly_earnings_l1777_177702

/-- Represents the T-shirt shop's operations and calculates weekly earnings -/
def TShirtShopEarnings : ℕ :=
  let women_shirt_price : ℕ := 18
  let men_shirt_price : ℕ := 15
  let women_shirt_interval : ℕ := 30  -- in minutes
  let men_shirt_interval : ℕ := 40    -- in minutes
  let daily_open_hours : ℕ := 12
  let days_per_week : ℕ := 7
  let minutes_per_hour : ℕ := 60

  let women_shirts_per_day : ℕ := (minutes_per_hour / women_shirt_interval) * daily_open_hours
  let men_shirts_per_day : ℕ := (minutes_per_hour / men_shirt_interval) * daily_open_hours
  
  let daily_earnings : ℕ := women_shirts_per_day * women_shirt_price + men_shirts_per_day * men_shirt_price
  
  daily_earnings * days_per_week

/-- Theorem stating that the weekly earnings of the T-shirt shop is $4914 -/
theorem tshirt_shop_weekly_earnings : TShirtShopEarnings = 4914 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_shop_weekly_earnings_l1777_177702


namespace NUMINAMATH_CALUDE_onion_bag_cost_l1777_177795

/-- The cost of one bag of onions -/
def cost_of_one_bag (price_per_onion : ℕ) (total_onions : ℕ) (num_bags : ℕ) : ℕ :=
  (price_per_onion * total_onions) / num_bags

/-- Theorem stating the cost of one bag of onions -/
theorem onion_bag_cost :
  let price_per_onion := 200
  let total_onions := 180
  let num_bags := 6
  cost_of_one_bag price_per_onion total_onions num_bags = 6000 := by
  sorry

end NUMINAMATH_CALUDE_onion_bag_cost_l1777_177795


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1777_177796

theorem fractional_equation_solution (k : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) ↔ k ≠ -3 ∧ k ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1777_177796


namespace NUMINAMATH_CALUDE_inequality_solutions_range_l1777_177758

theorem inequality_solutions_range (a : ℝ) : 
  (∃! (x₁ x₂ : ℕ), x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
   (∀ (x : ℕ), x > 0 → (3 * ↑x + a ≤ 2 ↔ (x = x₁ ∨ x = x₂)))) →
  -7 < a ∧ a ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_range_l1777_177758


namespace NUMINAMATH_CALUDE_special_triangle_bc_length_l1777_177776

/-- A triangle with special properties -/
structure SpecialTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Length of side AB is 1
  ab_length : dist A B = 1
  -- Length of side AC is 2
  ac_length : dist A C = 2
  -- Median from A to BC has same length as BC
  median_eq_bc : dist A ((B + C) / 2) = dist B C

/-- The length of BC in a SpecialTriangle is √2 -/
theorem special_triangle_bc_length (t : SpecialTriangle) : dist t.B t.C = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_bc_length_l1777_177776


namespace NUMINAMATH_CALUDE_krystiana_monthly_earnings_l1777_177765

/-- Represents the apartment building owned by Krystiana -/
structure ApartmentBuilding where
  firstFloorRate : ℕ
  secondFloorRate : ℕ
  thirdFloorRate : ℕ
  roomsPerFloor : ℕ
  occupiedThirdFloorRooms : ℕ

/-- Calculates the monthly earnings from Krystiana's apartment building -/
def calculateMonthlyEarnings (building : ApartmentBuilding) : ℕ :=
  building.firstFloorRate * building.roomsPerFloor +
  building.secondFloorRate * building.roomsPerFloor +
  building.thirdFloorRate * building.occupiedThirdFloorRooms

/-- Theorem stating that Krystiana's monthly earnings are $165 -/
theorem krystiana_monthly_earnings :
  ∀ (building : ApartmentBuilding),
    building.firstFloorRate = 15 →
    building.secondFloorRate = 20 →
    building.thirdFloorRate = 2 * building.firstFloorRate →
    building.roomsPerFloor = 3 →
    building.occupiedThirdFloorRooms = 2 →
    calculateMonthlyEarnings building = 165 := by
  sorry

#eval calculateMonthlyEarnings {
  firstFloorRate := 15,
  secondFloorRate := 20,
  thirdFloorRate := 30,
  roomsPerFloor := 3,
  occupiedThirdFloorRooms := 2
}

end NUMINAMATH_CALUDE_krystiana_monthly_earnings_l1777_177765


namespace NUMINAMATH_CALUDE_guess_number_in_seven_questions_l1777_177700

theorem guess_number_in_seven_questions :
  ∃ (f : Fin 7 → (Nat × Nat)),
    (∀ i, (f i).1 < 100 ∧ (f i).2 < 100) →
    ∀ X ≤ 100,
      ∀ Y ≤ 100,
      (∀ i, Nat.gcd (X + (f i).1) (f i).2 = Nat.gcd (Y + (f i).1) (f i).2) →
      X = Y :=
by sorry

end NUMINAMATH_CALUDE_guess_number_in_seven_questions_l1777_177700


namespace NUMINAMATH_CALUDE_angle_sum_proof_l1777_177781

theorem angle_sum_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = Real.sqrt 5 / 5) (h4 : Real.sin β = Real.sqrt 10 / 10) :
  α + β = π/4 := by sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l1777_177781


namespace NUMINAMATH_CALUDE_parallel_line_slope_l1777_177718

/-- Given a line with equation 3x + 6y = -12, this theorem states that
    the slope of any line parallel to it is -1/2. -/
theorem parallel_line_slope (x y : ℝ) :
  (3 : ℝ) * x + 6 * y = -12 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l1777_177718


namespace NUMINAMATH_CALUDE_no_functions_satisfying_condition_l1777_177715

theorem no_functions_satisfying_condition : 
  ¬∃ (f g : ℝ → ℝ), ∀ x y : ℝ, x ≠ y → |f x - f y| + |g x - g y| > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_functions_satisfying_condition_l1777_177715


namespace NUMINAMATH_CALUDE_roots_condition_l1777_177798

-- Define the quadratic function F(x)
def F (R l a x : ℝ) := 2 * R * x^2 - (l^2 + 4 * a * R) * x + 2 * R * a^2

-- Define the conditions for the roots to be between 0 and 2R
def roots_between_0_and_2R (R l a : ℝ) : Prop :=
  (0 < a ∧ a < 2 * R ∧ l^2 < (2 * R - a)^2) ∨
  (-2 * R < a ∧ a < 0 ∧ l^2 < (2 * R - a)^2)

-- Theorem statement
theorem roots_condition (R l a : ℝ) (hR : R > 0) (hl : l > 0) (ha : a ≠ 0) :
  (∀ x, F R l a x = 0 → 0 < x ∧ x < 2 * R) ↔ roots_between_0_and_2R R l a := by
  sorry

end NUMINAMATH_CALUDE_roots_condition_l1777_177798


namespace NUMINAMATH_CALUDE_johnny_closed_days_l1777_177777

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the number of crab dishes Johnny makes per day -/
def dishes_per_day : ℕ := 40

/-- Represents the amount of crab meat used per dish in pounds -/
def crab_per_dish : ℚ := 3/2

/-- Represents the cost of crab meat per pound in dollars -/
def crab_cost_per_pound : ℕ := 8

/-- Represents Johnny's weekly expenditure on crab meat in dollars -/
def weekly_expenditure : ℕ := 1920

/-- Theorem stating that Johnny is closed 3 days a week -/
theorem johnny_closed_days : 
  days_in_week - (weekly_expenditure / (dishes_per_day * crab_per_dish * crab_cost_per_pound)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_johnny_closed_days_l1777_177777


namespace NUMINAMATH_CALUDE_alex_age_l1777_177766

theorem alex_age (charlie_age alex_age : ℕ) : 
  charlie_age = 2 * alex_age + 8 → 
  charlie_age = 22 → 
  alex_age = 7 := by
sorry

end NUMINAMATH_CALUDE_alex_age_l1777_177766


namespace NUMINAMATH_CALUDE_pennys_bakery_revenue_l1777_177762

/-- Represents the price and quantity of a type of cheesecake -/
structure Cheesecake where
  price_per_slice : ℕ
  pies_sold : ℕ

/-- Calculates the total revenue from a type of cheesecake -/
def revenue (c : Cheesecake) (slices_per_pie : ℕ) : ℕ :=
  c.price_per_slice * c.pies_sold * slices_per_pie

/-- The main theorem about Penny's bakery revenue -/
theorem pennys_bakery_revenue : 
  let slices_per_pie : ℕ := 6
  let blueberry : Cheesecake := { price_per_slice := 7, pies_sold := 7 }
  let strawberry : Cheesecake := { price_per_slice := 8, pies_sold := 5 }
  let chocolate : Cheesecake := { price_per_slice := 9, pies_sold := 3 }
  revenue blueberry slices_per_pie + revenue strawberry slices_per_pie + revenue chocolate slices_per_pie = 696 := by
  sorry


end NUMINAMATH_CALUDE_pennys_bakery_revenue_l1777_177762


namespace NUMINAMATH_CALUDE_hydrangea_price_l1777_177728

def pansy_price : ℝ := 2.50
def petunia_price : ℝ := 1.00
def num_pansies : ℕ := 5
def num_petunias : ℕ := 5
def discount_rate : ℝ := 0.10
def paid_amount : ℝ := 50.00
def change_received : ℝ := 23.00

theorem hydrangea_price (hydrangea_cost : ℝ) : hydrangea_cost = 12.50 := by
  sorry

#check hydrangea_price

end NUMINAMATH_CALUDE_hydrangea_price_l1777_177728


namespace NUMINAMATH_CALUDE_tangent_curve_sum_l1777_177721

/-- A curve y = -2x^2 + bx + c is tangent to the line y = x - 3 at the point (2, -1). -/
theorem tangent_curve_sum (b c : ℝ) : 
  (∀ x, -2 * x^2 + b * x + c = x - 3 → x = 2) → 
  (-2 * 2^2 + b * 2 + c = -1) →
  ((-4 * 2 + b) = 1) →
  b + c = -2 := by sorry

end NUMINAMATH_CALUDE_tangent_curve_sum_l1777_177721


namespace NUMINAMATH_CALUDE_ellipse_and_circle_problem_l1777_177760

theorem ellipse_and_circle_problem 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 2^2 = a^2 - b^2) -- condition for right focus at (2,0)
  : 
  (∀ x y : ℝ, x^2/6 + y^2/2 = 1 ↔ x^2/a^2 + y^2/b^2 = 1) ∧ 
  (∃ m : ℝ, ∃ c : Set (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∈ c ↔ (p.1^2 + (p.2 - 1/3)^2 = (1/3)^2)) ∧
    (∃ p1 p2 p3 p4 : ℝ × ℝ, 
      p1 ∈ c ∧ p2 ∈ c ∧ p3 ∈ c ∧ p4 ∈ c ∧
      p1.2 = p1.1^2 + m ∧ p2.2 = p2.1^2 + m ∧ p3.2 = p3.1^2 + m ∧ p4.2 = p4.1^2 + m ∧
      p1.1^2/6 + p1.2^2/2 = 1 ∧ p2.1^2/6 + p2.2^2/2 = 1 ∧ p3.1^2/6 + p3.2^2/2 = 1 ∧ p4.1^2/6 + p4.2^2/2 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_problem_l1777_177760


namespace NUMINAMATH_CALUDE_inverse_256_mod_101_l1777_177783

theorem inverse_256_mod_101 (h : (16⁻¹ : ZMod 101) = 31) :
  (256⁻¹ : ZMod 101) = 52 := by
  sorry

end NUMINAMATH_CALUDE_inverse_256_mod_101_l1777_177783


namespace NUMINAMATH_CALUDE_enclosed_area_calculation_l1777_177712

/-- The area enclosed by a curve consisting of 9 congruent circular arcs, 
    each of length π/2, whose centers are at the vertices of a regular hexagon 
    with side length 3. -/
def enclosed_area (num_arcs : ℕ) (arc_length : ℝ) (hexagon_side : ℝ) : ℝ :=
  sorry

/-- Theorem stating the enclosed area for the specific problem -/
theorem enclosed_area_calculation : 
  enclosed_area 9 (π/2) 3 = (27 * Real.sqrt 3) / 2 + 9 * π / 8 :=
sorry

end NUMINAMATH_CALUDE_enclosed_area_calculation_l1777_177712


namespace NUMINAMATH_CALUDE_rungs_on_twenty_ladders_eq_1200_l1777_177774

/-- Calculates the number of rungs on 20 ladders given the following conditions:
  * There are 10 ladders with 50 rungs each
  * There are 20 additional ladders with an unknown number of rungs
  * Each rung costs $2
  * The total cost for all ladders is $3,400
-/
def rungs_on_twenty_ladders : ℕ :=
  let ladders_with_fifty_rungs : ℕ := 10
  let rungs_per_ladder : ℕ := 50
  let cost_per_rung : ℕ := 2
  let total_cost : ℕ := 3400
  let remaining_ladders : ℕ := 20
  
  let cost_of_fifty_rung_ladders : ℕ := ladders_with_fifty_rungs * rungs_per_ladder * cost_per_rung
  let remaining_cost : ℕ := total_cost - cost_of_fifty_rung_ladders
  remaining_cost / cost_per_rung

theorem rungs_on_twenty_ladders_eq_1200 : rungs_on_twenty_ladders = 1200 := by
  sorry

end NUMINAMATH_CALUDE_rungs_on_twenty_ladders_eq_1200_l1777_177774


namespace NUMINAMATH_CALUDE_lemonade_stand_problem_l1777_177754

/-- Represents the lemonade stand problem -/
theorem lemonade_stand_problem 
  (total_days : ℕ) 
  (hot_days : ℕ) 
  (cups_per_day : ℕ) 
  (total_profit : ℚ) 
  (cost_per_cup : ℚ) 
  (hot_day_price_increase : ℚ) :
  total_days = 10 →
  hot_days = 4 →
  cups_per_day = 32 →
  total_profit = 350 →
  cost_per_cup = 3/4 →
  hot_day_price_increase = 1/4 →
  ∃ (regular_price : ℚ),
    regular_price > 0 ∧
    (total_days - hot_days) * cups_per_day * regular_price +
    hot_days * cups_per_day * (regular_price * (1 + hot_day_price_increase)) -
    total_days * cups_per_day * cost_per_cup = total_profit ∧
    regular_price * (1 + hot_day_price_increase) = 15/8 :=
by sorry

end NUMINAMATH_CALUDE_lemonade_stand_problem_l1777_177754


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1777_177742

theorem shopkeeper_profit (CP : ℝ) (CP_pos : CP > 0) : 
  let LP := CP * 1.3
  let SP := LP * 0.9
  let profit := SP - CP
  let percent_profit := (profit / CP) * 100
  percent_profit = 17 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1777_177742


namespace NUMINAMATH_CALUDE_derivative_symmetry_l1777_177743

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Theorem statement
theorem derivative_symmetry (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_symmetry_l1777_177743


namespace NUMINAMATH_CALUDE_mode_of_team_ages_l1777_177749

def team_ages : List Nat := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_team_ages :
  mode team_ages = 18 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_team_ages_l1777_177749


namespace NUMINAMATH_CALUDE_g_expression_l1777_177759

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g using the given condition
def g (x : ℝ) : ℝ := f (x - 3)

-- Theorem statement
theorem g_expression : ∀ x : ℝ, g x = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l1777_177759


namespace NUMINAMATH_CALUDE_flower_bed_max_area_l1777_177778

/-- Given a rectangular flower bed with one side against a house,
    using 450 feet of total fencing with 150 feet along the house,
    the maximum area of the flower bed is 22500 square feet. -/
theorem flower_bed_max_area :
  ∀ (l w : ℝ),
  l = 150 →
  l + 2 * w = 450 →
  l * w ≤ 22500 :=
by sorry

end NUMINAMATH_CALUDE_flower_bed_max_area_l1777_177778


namespace NUMINAMATH_CALUDE_chessboard_tiling_l1777_177701

/-- A chessboard of size a × b can be tiled with n-ominoes of size 1 × n if and only if n divides a or n divides b. -/
theorem chessboard_tiling (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (∃ (tiling : ℕ → ℕ → Fin n), ∀ (i j : ℕ), i < a ∧ j < b → 
    (∀ (k : ℕ), k < n → tiling i (j + k) = tiling i j + k ∨ tiling (i + k) j = tiling i j + k)) ↔ 
  (n ∣ a ∨ n ∣ b) :=
by sorry

end NUMINAMATH_CALUDE_chessboard_tiling_l1777_177701


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1777_177797

theorem fraction_evaluation (a b : ℝ) (h1 : a = 7) (h2 : b = 4) :
  5 / (a - b)^2 = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1777_177797


namespace NUMINAMATH_CALUDE_projection_equals_negative_two_l1777_177772

def a : Fin 2 → ℝ
| 0 => 4
| 1 => -7

def b : Fin 2 → ℝ
| 0 => 3
| 1 => -4

theorem projection_equals_negative_two :
  let proj := (((a - 2 • b) • b) / (b • b)) • b
  proj = (-2 : ℝ) • b :=
by sorry

end NUMINAMATH_CALUDE_projection_equals_negative_two_l1777_177772


namespace NUMINAMATH_CALUDE_f_nonpositive_implies_k_geq_one_l1777_177782

open Real

/-- Given a function f(x) = ln(ex) - kx defined on (0, +∞), 
    if f(x) ≤ 0 for all x > 0, then k ≥ 1 -/
theorem f_nonpositive_implies_k_geq_one (k : ℝ) : 
  (∀ x > 0, Real.log (Real.exp 1 * x) - k * x ≤ 0) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_nonpositive_implies_k_geq_one_l1777_177782


namespace NUMINAMATH_CALUDE_binomial_product_nine_two_seven_two_l1777_177719

theorem binomial_product_nine_two_seven_two :
  Nat.choose 9 2 * Nat.choose 7 2 = 756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_nine_two_seven_two_l1777_177719


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l1777_177786

/-- Given a sequence {a_n} with S_n being the sum of its first n terms,
    if S_n^2 - 2S_n - a_nS_n + 1 = 0 for all positive integers n,
    then S_n = n / (n + 1) for all positive integers n. -/
theorem sequence_sum_theorem (a : ℕ+ → ℚ) (S : ℕ+ → ℚ)
    (h : ∀ n : ℕ+, S n ^ 2 - 2 * S n - a n * S n + 1 = 0) :
  ∀ n : ℕ+, S n = n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l1777_177786


namespace NUMINAMATH_CALUDE_certain_number_is_two_l1777_177717

theorem certain_number_is_two :
  ∃ x : ℚ, (287 * 287) + (269 * 269) - x * (287 * 269) = 324 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_two_l1777_177717


namespace NUMINAMATH_CALUDE_custom_op_five_two_l1777_177791

-- Define the custom operation
def custom_op (a b : ℕ) : ℕ := 3*a + 4*b - a*b

-- State the theorem
theorem custom_op_five_two : custom_op 5 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_five_two_l1777_177791


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l1777_177735

theorem abc_sum_sqrt (a b c : ℝ) 
  (eq1 : b + c = 17) 
  (eq2 : c + a = 18) 
  (eq3 : a + b = 19) : 
  Real.sqrt (a * b * c * (a + b + c)) = 60 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l1777_177735


namespace NUMINAMATH_CALUDE_power_expression_l1777_177726

theorem power_expression (a x y : ℝ) (ha : a > 0) (hx : a^x = 3) (hy : a^y = 5) :
  a^(2*x + y/2) = 9 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_l1777_177726


namespace NUMINAMATH_CALUDE_expression_equals_185_l1777_177761

theorem expression_equals_185 : (-4)^7 / 4^5 + 5^3 * 2 - 7^2 = 185 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_185_l1777_177761


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l1777_177769

/-- Given that f(x) = (x + a)(x - 4) is an even function, prove that a = 4 --/
theorem even_function_implies_a_equals_four (a : ℝ) :
  (∀ x : ℝ, (x + a) * (x - 4) = (-x + a) * (-x - 4)) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l1777_177769


namespace NUMINAMATH_CALUDE_equation_solutions_l1777_177764

theorem equation_solutions :
  (∀ x : ℝ, (x + 1) / (x - 1) - 4 / (x^2 - 1) ≠ 1) ∧
  (∀ x : ℝ, x^2 + 3*x - 2 = 0 ↔ x = -3/2 - Real.sqrt 17/2 ∨ x = -3/2 + Real.sqrt 17/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1777_177764


namespace NUMINAMATH_CALUDE_apple_fractions_l1777_177739

/-- Given that Simone ate 1/2 of an apple each day for 16 days,
    Lauri ate x fraction of an apple each day for 15 days,
    and the total number of apples eaten by both girls is 13,
    prove that x = 1/3 -/
theorem apple_fractions (x : ℚ) : 
  (16 * (1/2 : ℚ)) + (15 * x) = 13 → x = 1/3 := by sorry

end NUMINAMATH_CALUDE_apple_fractions_l1777_177739


namespace NUMINAMATH_CALUDE_new_parabola_equation_l1777_177713

/-- The original quadratic function -/
def original_function (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- The vertex of the original parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- The axis of symmetry of the original parabola -/
def axis_of_symmetry : ℝ := 1

/-- The line that intersects the new parabola -/
def intersecting_line (m : ℝ) (x : ℝ) : ℝ := m * x - 2

/-- The point of intersection between the new parabola and the line -/
def intersection_point : ℝ × ℝ := (2, 4)

/-- The equation of the new parabola -/
def new_parabola (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 4

theorem new_parabola_equation :
  ∃ (t : ℝ), ∀ (x : ℝ),
    new_parabola x = -3 * (x - axis_of_symmetry)^2 + vertex.2 + t ∧
    new_parabola intersection_point.1 = intersection_point.2 :=
by sorry

end NUMINAMATH_CALUDE_new_parabola_equation_l1777_177713


namespace NUMINAMATH_CALUDE_fish_value_in_rice_l1777_177745

/-- Represents the value of items in a barter system -/
structure BarterValue where
  fish : ℚ
  bread : ℚ
  rice : ℚ

/-- The barter system with given exchange rates -/
def barterSystem : BarterValue where
  fish := 1
  bread := 3/5
  rice := 1/10

theorem fish_value_in_rice (b : BarterValue) 
  (h1 : 5 * b.fish = 3 * b.bread) 
  (h2 : b.bread = 6 * b.rice) : 
  b.fish = 18/5 * b.rice := by
  sorry

#check fish_value_in_rice barterSystem

end NUMINAMATH_CALUDE_fish_value_in_rice_l1777_177745


namespace NUMINAMATH_CALUDE_babysitter_earnings_correct_l1777_177780

/-- Calculates the babysitter's earnings for a given number of hours worked -/
def babysitter_earnings (regular_rate : ℕ) (regular_hours : ℕ) (overtime_rate : ℕ) (total_hours : ℕ) : ℕ :=
  let regular_pay := min regular_hours total_hours * regular_rate
  let overtime_pay := max 0 (total_hours - regular_hours) * overtime_rate
  regular_pay + overtime_pay

theorem babysitter_earnings_correct :
  let regular_rate : ℕ := 16
  let regular_hours : ℕ := 30
  let overtime_rate : ℕ := 28  -- 16 + (75% of 16)
  let total_hours : ℕ := 40
  babysitter_earnings regular_rate regular_hours overtime_rate total_hours = 760 :=
by sorry

end NUMINAMATH_CALUDE_babysitter_earnings_correct_l1777_177780


namespace NUMINAMATH_CALUDE_complex_sum_real_parts_l1777_177733

theorem complex_sum_real_parts (a b : ℝ) (h : Complex.mk a b = Complex.I * (1 - Complex.I)) : a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_real_parts_l1777_177733


namespace NUMINAMATH_CALUDE_total_meows_in_five_minutes_l1777_177748

/-- The number of meows per minute for the first cat -/
def first_cat_meows : ℕ := 3

/-- The number of meows per minute for the second cat -/
def second_cat_meows : ℕ := 2 * first_cat_meows

/-- The number of meows per minute for the third cat -/
def third_cat_meows : ℕ := second_cat_meows / 3

/-- The duration in minutes -/
def duration : ℕ := 5

/-- Theorem: The total number of meows from all three cats in 5 minutes is 55 -/
theorem total_meows_in_five_minutes :
  first_cat_meows * duration + second_cat_meows * duration + third_cat_meows * duration = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_meows_in_five_minutes_l1777_177748


namespace NUMINAMATH_CALUDE_word_count_correct_l1777_177793

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the word -/
def word_length : ℕ := 5

/-- The number of positions that can vary (middle letters) -/
def varying_positions : ℕ := word_length - 2

/-- The number of five-letter words where the first and last letters are the same -/
def num_words : ℕ := alphabet_size * (alphabet_size ^ varying_positions)

theorem word_count_correct : num_words = 456976 := by sorry

end NUMINAMATH_CALUDE_word_count_correct_l1777_177793


namespace NUMINAMATH_CALUDE_equation_solution_l1777_177737

theorem equation_solution : 
  ∃! x : ℚ, (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3 ∧ x = 9 / 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1777_177737


namespace NUMINAMATH_CALUDE_dollar_three_neg_two_l1777_177727

-- Define the operation $
def dollar (a b : ℤ) : ℤ := a * (b - 1) + a * b

-- Theorem statement
theorem dollar_three_neg_two : dollar 3 (-2) = -15 := by
  sorry

end NUMINAMATH_CALUDE_dollar_three_neg_two_l1777_177727


namespace NUMINAMATH_CALUDE_intersection_M_N_l1777_177732

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1777_177732


namespace NUMINAMATH_CALUDE_collinear_vectors_perpendicular_vectors_l1777_177740

-- Problem 1
def point_A : ℝ × ℝ := (5, 4)
def point_C : ℝ × ℝ := (12, -2)

def vector_AB (k : ℝ) : ℝ × ℝ := (k - 5, 6)
def vector_BC (k : ℝ) : ℝ × ℝ := (12 - k, -12)

def are_collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem collinear_vectors :
  are_collinear (vector_AB (-2)) (vector_BC (-2)) := by sorry

-- Problem 2
def vector_OA : ℝ × ℝ := (-7, 6)
def vector_OC : ℝ × ℝ := (5, 7)

def vector_AB' (k : ℝ) : ℝ × ℝ := (10, k - 6)
def vector_BC' (k : ℝ) : ℝ × ℝ := (2, 7 - k)

def are_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem perpendicular_vectors :
  (are_perpendicular (vector_AB' 2) (vector_BC' 2)) ∧
  (are_perpendicular (vector_AB' 11) (vector_BC' 11)) := by sorry

end NUMINAMATH_CALUDE_collinear_vectors_perpendicular_vectors_l1777_177740


namespace NUMINAMATH_CALUDE_three_solutions_when_a_is_9_l1777_177773

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 6

-- Theorem statement
theorem three_solutions_when_a_is_9 :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f 9 x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_when_a_is_9_l1777_177773


namespace NUMINAMATH_CALUDE_tory_sold_to_grandmother_l1777_177738

/-- Represents the cookie sales problem for Tory's school fundraiser -/
def cookie_sales (grandmother_packs : ℕ) : Prop :=
  let total_packs : ℕ := 50
  let uncle_packs : ℕ := 7
  let neighbor_packs : ℕ := 5
  let remaining_packs : ℕ := 26
  grandmother_packs + uncle_packs + neighbor_packs + remaining_packs = total_packs

/-- Proves that Tory sold 12 packs of cookies to his grandmother -/
theorem tory_sold_to_grandmother :
  ∃ (x : ℕ), cookie_sales x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_tory_sold_to_grandmother_l1777_177738


namespace NUMINAMATH_CALUDE_abs_minus_self_nonnegative_l1777_177775

theorem abs_minus_self_nonnegative (x : ℝ) : |x| - x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_minus_self_nonnegative_l1777_177775


namespace NUMINAMATH_CALUDE_function_properties_l1777_177710

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (f a 1 + f a (-1) = 5/2 → f a 2 + f a (-2) = 17/4) ∧
  (∃ (max min : ℝ), (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ max ∧ f a x ≥ min) ∧
    max - min = 8/3 → a = 3 ∨ a = 1/3) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1777_177710


namespace NUMINAMATH_CALUDE_books_written_proof_l1777_177753

/-- The number of books Zig wrote -/
def zig_books : ℕ := 60

/-- The number of books Flo wrote -/
def flo_books : ℕ := zig_books / 4

/-- The total number of books written by Zig and Flo -/
def total_books : ℕ := zig_books + flo_books

theorem books_written_proof :
  (zig_books = 4 * flo_books) → total_books = 75 := by
  sorry

end NUMINAMATH_CALUDE_books_written_proof_l1777_177753


namespace NUMINAMATH_CALUDE_competition_earnings_difference_l1777_177747

/-- Represents the earnings of a seller for a single day -/
structure DayEarnings where
  regular_sales : ℝ
  discounted_sales : ℝ
  tax_rate : ℝ
  exchange_rate : ℝ

/-- Calculates the total earnings for a day after tax and currency conversion -/
def calculate_day_earnings (e : DayEarnings) : ℝ :=
  let total_sales := e.regular_sales + e.discounted_sales
  let after_tax := total_sales * (1 - e.tax_rate)
  after_tax * e.exchange_rate

/-- Represents the earnings of a seller for the two-day competition -/
structure CompetitionEarnings where
  day1 : DayEarnings
  day2 : DayEarnings

/-- Calculates the total earnings for the two-day competition -/
def calculate_total_earnings (e : CompetitionEarnings) : ℝ :=
  calculate_day_earnings e.day1 + calculate_day_earnings e.day2

/-- Theorem statement for the competition earnings -/
theorem competition_earnings_difference
  (bert_earnings tory_earnings : CompetitionEarnings)
  (h_bert_day1 : bert_earnings.day1 = {
    regular_sales := 9 * 18,
    discounted_sales := 3 * (18 * 0.85),
    tax_rate := 0.05,
    exchange_rate := 1
  })
  (h_bert_day2 : bert_earnings.day2 = {
    regular_sales := 10 * 15,
    discounted_sales := 0,
    tax_rate := 0.05,
    exchange_rate := 1.4
  })
  (h_tory_day1 : tory_earnings.day1 = {
    regular_sales := 10 * 20,
    discounted_sales := 5 * (20 * 0.9),
    tax_rate := 0.05,
    exchange_rate := 1
  })
  (h_tory_day2 : tory_earnings.day2 = {
    regular_sales := 8 * 18,
    discounted_sales := 0,
    tax_rate := 0.05,
    exchange_rate := 1.4
  }) :
  calculate_total_earnings tory_earnings - calculate_total_earnings bert_earnings = 71.82 := by
  sorry


end NUMINAMATH_CALUDE_competition_earnings_difference_l1777_177747


namespace NUMINAMATH_CALUDE_probability_one_blue_one_white_l1777_177704

def total_marbles : ℕ := 8
def blue_marbles : ℕ := 3
def white_marbles : ℕ := 5
def marbles_left : ℕ := 2

def favorable_outcomes : ℕ := blue_marbles * white_marbles
def total_outcomes : ℕ := Nat.choose total_marbles marbles_left

theorem probability_one_blue_one_white :
  (favorable_outcomes : ℚ) / total_outcomes = 15 / 28 := by sorry

end NUMINAMATH_CALUDE_probability_one_blue_one_white_l1777_177704


namespace NUMINAMATH_CALUDE_painted_subcubes_count_l1777_177790

def cube_size : ℕ := 4

-- Define a function to calculate the number of subcubes with at least two painted faces
def subcubes_with_two_or_more_painted_faces (n : ℕ) : ℕ :=
  8 + 12 * (n - 2)

-- Theorem statement
theorem painted_subcubes_count :
  subcubes_with_two_or_more_painted_faces cube_size = 32 := by
  sorry

end NUMINAMATH_CALUDE_painted_subcubes_count_l1777_177790


namespace NUMINAMATH_CALUDE_decimal_to_base_conversion_l1777_177741

theorem decimal_to_base_conversion (x : ℕ) : 
  (4 * x + 7 = 71) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_base_conversion_l1777_177741


namespace NUMINAMATH_CALUDE_sqrt_six_equality_l1777_177784

theorem sqrt_six_equality (r : ℝ) (h : r = Real.sqrt 2 + Real.sqrt 3) :
  Real.sqrt 6 = (r^2 - 5) / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_six_equality_l1777_177784


namespace NUMINAMATH_CALUDE_cats_sold_during_sale_l1777_177755

theorem cats_sold_during_sale
  (initial_siamese : ℕ)
  (initial_house : ℕ)
  (cats_remaining : ℕ)
  (h1 : initial_siamese = 12)
  (h2 : initial_house = 20)
  (h3 : cats_remaining = 12) :
  initial_siamese + initial_house - cats_remaining = 20 :=
by sorry

end NUMINAMATH_CALUDE_cats_sold_during_sale_l1777_177755


namespace NUMINAMATH_CALUDE_tree_height_l1777_177763

theorem tree_height (hop_distance : ℕ) (slip_distance : ℕ) (total_hours : ℕ) (tree_height : ℕ) : 
  hop_distance = 3 →
  slip_distance = 2 →
  total_hours = 17 →
  tree_height = (total_hours - 1) * (hop_distance - slip_distance) + hop_distance := by
sorry

#eval (17 - 1) * (3 - 2) + 3

end NUMINAMATH_CALUDE_tree_height_l1777_177763


namespace NUMINAMATH_CALUDE_combination_equation_solution_l1777_177787

theorem combination_equation_solution : 
  ∃! (n : ℕ), n > 0 ∧ Nat.choose (n + 1) (n - 1) = 21 := by sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l1777_177787


namespace NUMINAMATH_CALUDE_candy_probability_theorem_l1777_177794

/-- The probability of selecting the same candy type for the first and last candy -/
def same_type_probability (lollipops chocolate jelly : ℕ) : ℚ :=
  let total := lollipops + chocolate + jelly
  let p_lollipop := (lollipops : ℚ) / total * (lollipops - 1) / (total - 1)
  let p_chocolate := (chocolate : ℚ) / total * (chocolate - 1) / (total - 1)
  let p_jelly := (jelly : ℚ) / total * (jelly - 1) / (total - 1)
  p_lollipop + p_chocolate + p_jelly

theorem candy_probability_theorem :
  same_type_probability 2 3 5 = 14 / 45 := by
  sorry

#eval same_type_probability 2 3 5

end NUMINAMATH_CALUDE_candy_probability_theorem_l1777_177794


namespace NUMINAMATH_CALUDE_factorial_sum_calculation_l1777_177720

theorem factorial_sum_calculation : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 2 * Nat.factorial 5 = 5160 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_calculation_l1777_177720


namespace NUMINAMATH_CALUDE_john_remaining_money_l1777_177723

/-- The amount of money John has left after purchasing pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let initial_money : ℝ := 50
  let drink_cost : ℝ := q
  let small_pizza_cost : ℝ := q
  let large_pizza_cost : ℝ := 4 * q
  let num_drinks : ℕ := 4
  let num_small_pizzas : ℕ := 2
  let num_large_pizzas : ℕ := 1
  initial_money - (num_drinks * drink_cost + num_small_pizzas * small_pizza_cost + num_large_pizzas * large_pizza_cost)

/-- Theorem stating that John's remaining money is equal to 50 - 10q -/
theorem john_remaining_money (q : ℝ) : money_left q = 50 - 10 * q := by
  sorry

end NUMINAMATH_CALUDE_john_remaining_money_l1777_177723


namespace NUMINAMATH_CALUDE_final_peanut_count_l1777_177789

def peanut_problem (initial_peanuts : ℕ) (mary_adds : ℕ) (john_takes : ℕ) (friends : ℕ) : ℕ :=
  initial_peanuts + mary_adds - john_takes

theorem final_peanut_count :
  peanut_problem 4 4 2 2 = 6 := by sorry

end NUMINAMATH_CALUDE_final_peanut_count_l1777_177789


namespace NUMINAMATH_CALUDE_customers_before_rush_count_l1777_177785

/-- The number of customers who didn't leave a tip -/
def no_tip : ℕ := 49

/-- The number of customers who left a tip -/
def left_tip : ℕ := 2

/-- The number of additional customers during lunch rush -/
def additional_customers : ℕ := 12

/-- The total number of customers after the lunch rush -/
def total_after_rush : ℕ := no_tip + left_tip

/-- The number of customers before the lunch rush -/
def customers_before_rush : ℕ := total_after_rush - additional_customers

theorem customers_before_rush_count : customers_before_rush = 39 := by
  sorry

end NUMINAMATH_CALUDE_customers_before_rush_count_l1777_177785


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l1777_177799

theorem quadratic_complete_square (x : ℝ) : 
  (∃ (b c : ℤ), (x + b : ℝ)^2 = c ∧ x^2 - 10*x + 15 = 0) → 
  (∃ (b c : ℤ), (x + b : ℝ)^2 = c ∧ b + c = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l1777_177799


namespace NUMINAMATH_CALUDE_trivia_team_distribution_l1777_177771

theorem trivia_team_distribution (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total_students = 58)
  (h2 : not_picked = 10)
  (h3 : num_groups = 8) :
  (total_students - not_picked) / num_groups = 6 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_distribution_l1777_177771


namespace NUMINAMATH_CALUDE_one_fifth_of_five_times_nine_l1777_177706

theorem one_fifth_of_five_times_nine : (1 / 5 : ℚ) * (5 * 9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_five_times_nine_l1777_177706


namespace NUMINAMATH_CALUDE_range_of_S_l1777_177770

theorem range_of_S (x : ℝ) (y : ℝ) (S : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : 0 ≤ x ∧ x ≤ 1/2) 
  (h3 : S = x * y) : 
  -1/8 ≤ S ∧ S ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_S_l1777_177770


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_range_l1777_177716

theorem quadratic_roots_imply_m_range (m : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ∈ Set.Ioo 0 1 ∧ r₂ ∈ Set.Ioo 2 3 ∧ 
   r₁^2 - 2*m*r₁ + m^2 - 1 = 0 ∧ r₂^2 - 2*m*r₂ + m^2 - 1 = 0) →
  m ∈ Set.Ioo 1 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_range_l1777_177716


namespace NUMINAMATH_CALUDE_eighth_term_is_negative_22_l1777_177722

/-- An arithmetic sequence with a2 = -4 and common difference -3 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  -4 + (n - 2) * (-3)

/-- Theorem: The 8th term of the arithmetic sequence is -22 -/
theorem eighth_term_is_negative_22 : arithmetic_sequence 8 = -22 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_negative_22_l1777_177722


namespace NUMINAMATH_CALUDE_midpoint_line_slope_zero_l1777_177708

/-- The slope of the line containing the midpoints of the segments [(1, 1), (3, 4)] and [(4, 1), (7, 4)] is 0. -/
theorem midpoint_line_slope_zero : 
  let midpoint1 := ((1 + 3) / 2, (1 + 4) / 2)
  let midpoint2 := ((4 + 7) / 2, (1 + 4) / 2)
  let slope := (midpoint2.2 - midpoint1.2) / (midpoint2.1 - midpoint1.1)
  slope = 0 := by
sorry

end NUMINAMATH_CALUDE_midpoint_line_slope_zero_l1777_177708


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l1777_177709

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l1777_177709


namespace NUMINAMATH_CALUDE_min_omega_value_l1777_177725

/-- Given a function f(x) = sin(ω(x - π/4)) where ω > 0, if f(3π/4) = 0, then the minimum value of ω is 2. -/
theorem min_omega_value (ω : ℝ) (h₁ : ω > 0) :
  (fun x => Real.sin (ω * (x - π / 4))) (3 * π / 4) = 0 → ω ≥ 2 ∧ ∀ ω' > 0, (fun x => Real.sin (ω' * (x - π / 4))) (3 * π / 4) = 0 → ω' ≥ ω :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l1777_177725


namespace NUMINAMATH_CALUDE_flag_arrangement_remainder_l1777_177703

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def M : ℕ :=
  13 * Nat.choose 14 10 - 2 * Nat.choose 13 10

/-- The theorem stating the remainder when M is divided by 1000 -/
theorem flag_arrangement_remainder :
  M % 1000 = 441 := by
  sorry

end NUMINAMATH_CALUDE_flag_arrangement_remainder_l1777_177703


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l1777_177744

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / b₁ = a₂ / b₂ ∧ a₁ / b₁ ≠ c₁ / c₂

/-- The value of m for which the lines x + my + 6 = 0 and (m-2)x + 3y + 2m = 0 are parallel -/
theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel 1 m 6 (m-2) 3 (2*m) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l1777_177744


namespace NUMINAMATH_CALUDE_total_pizzas_eaten_l1777_177750

/-- The number of pizzas eaten by class A -/
def pizzas_class_a : ℕ := 8

/-- The number of pizzas eaten by class B -/
def pizzas_class_b : ℕ := 7

/-- The total number of pizzas eaten by both classes -/
def total_pizzas : ℕ := pizzas_class_a + pizzas_class_b

theorem total_pizzas_eaten : total_pizzas = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_pizzas_eaten_l1777_177750


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l1777_177768

theorem simplify_and_ratio : ∀ m : ℝ, 
  (6 * m + 12) / 3 = 2 * m + 4 ∧ 2 / 4 = (1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l1777_177768


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1777_177757

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 512 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 60 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1777_177757


namespace NUMINAMATH_CALUDE_reflect_point_D_l1777_177792

/-- Reflect a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflect a point across the line y = x - 1 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 + 1, p.1 - 1)

/-- The main theorem stating that reflecting D(5,0) across y-axis and then y=x-1 results in (-1,4) -/
theorem reflect_point_D : 
  let D : ℝ × ℝ := (5, 0)
  let D' := reflect_y_axis D
  let D'' := reflect_line D'
  D'' = (-1, 4) := by sorry

end NUMINAMATH_CALUDE_reflect_point_D_l1777_177792


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_36_l1777_177779

theorem ceiling_neg_sqrt_36 : ⌈-Real.sqrt 36⌉ = -6 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_36_l1777_177779


namespace NUMINAMATH_CALUDE_shampoo_duration_l1777_177752

theorem shampoo_duration (rose_shampoo : Rat) (jasmine_shampoo : Rat) (daily_usage : Rat) : 
  rose_shampoo = 1/3 → jasmine_shampoo = 1/4 → daily_usage = 1/12 →
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end NUMINAMATH_CALUDE_shampoo_duration_l1777_177752


namespace NUMINAMATH_CALUDE_exist_three_aliens_common_language_l1777_177724

/-- The number of aliens -/
def num_aliens : ℕ := 3 * Nat.factorial 2005

/-- The number of languages -/
def num_languages : ℕ := 2005

/-- A function representing the language used between two aliens -/
def communication_language : Fin num_aliens → Fin num_aliens → Fin num_languages := sorry

/-- The main theorem -/
theorem exist_three_aliens_common_language :
  ∃ (a b c : Fin num_aliens),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    communication_language a b = communication_language b c ∧
    communication_language b c = communication_language a c :=
by sorry

end NUMINAMATH_CALUDE_exist_three_aliens_common_language_l1777_177724


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1777_177729

theorem shaded_area_theorem (square_side : ℝ) (total_beans : ℕ) (shaded_beans : ℕ) :
  square_side = 2 →
  total_beans = 200 →
  shaded_beans = 120 →
  (shaded_beans : ℝ) / (total_beans : ℝ) * (square_side ^ 2) = 12 / 5 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l1777_177729


namespace NUMINAMATH_CALUDE_only_jia_can_formulate_quadratic_l1777_177736

/-- Represents a person in the problem -/
inductive Person
  | jia
  | yi
  | bing
  | ding

/-- Checks if a number is congruent to 1 modulo 3 -/
def is_cong_1_mod_3 (n : ℤ) : Prop := n % 3 = 1

/-- Checks if a number is congruent to 2 modulo 3 -/
def is_cong_2_mod_3 (n : ℤ) : Prop := n % 3 = 2

/-- Represents the conditions for each person's quadratic equation -/
def satisfies_conditions (person : Person) (p q x₁ x₂ : ℤ) : Prop :=
  match person with
  | Person.jia => is_cong_1_mod_3 p ∧ is_cong_1_mod_3 q ∧ is_cong_1_mod_3 x₁ ∧ is_cong_1_mod_3 x₂
  | Person.yi => is_cong_2_mod_3 p ∧ is_cong_2_mod_3 q ∧ is_cong_2_mod_3 x₁ ∧ is_cong_2_mod_3 x₂
  | Person.bing => is_cong_1_mod_3 p ∧ is_cong_1_mod_3 q ∧ is_cong_2_mod_3 x₁ ∧ is_cong_2_mod_3 x₂
  | Person.ding => is_cong_2_mod_3 p ∧ is_cong_2_mod_3 q ∧ is_cong_1_mod_3 x₁ ∧ is_cong_1_mod_3 x₂

/-- Represents a valid quadratic equation -/
def is_valid_quadratic (p q x₁ x₂ : ℤ) : Prop :=
  x₁ + x₂ = -p ∧ x₁ * x₂ = q

/-- The main theorem stating that only Jia can formulate a valid quadratic equation -/
theorem only_jia_can_formulate_quadratic :
  ∀ person : Person,
    (∃ p q x₁ x₂ : ℤ, satisfies_conditions person p q x₁ x₂ ∧ is_valid_quadratic p q x₁ x₂) ↔
    person = Person.jia :=
sorry


end NUMINAMATH_CALUDE_only_jia_can_formulate_quadratic_l1777_177736


namespace NUMINAMATH_CALUDE_fuel_cost_calculation_l1777_177734

theorem fuel_cost_calculation (original_cost : ℝ) : 
  (2 * original_cost * 1.2 = 480) → original_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_calculation_l1777_177734


namespace NUMINAMATH_CALUDE_jiangsu_income_scientific_notation_l1777_177711

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Rounds a real number to a specified number of significant figures -/
def roundToSignificantFigures (x : ℝ) (sigFigs : ℕ) : ℝ :=
  sorry

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The original amount in yuan -/
def originalAmount : ℝ := 26341

/-- The number of significant figures required -/
def requiredSigFigs : ℕ := 3

theorem jiangsu_income_scientific_notation :
  toScientificNotation originalAmount requiredSigFigs =
    ScientificNotation.mk 2.63 4 := by sorry

end NUMINAMATH_CALUDE_jiangsu_income_scientific_notation_l1777_177711


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1777_177788

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem: The tangent line to y = x^3 - 3x at (0, 0) is y = -3x
theorem tangent_line_at_origin : 
  ∀ x : ℝ, (f' 0) * x = -3 * x := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l1777_177788
