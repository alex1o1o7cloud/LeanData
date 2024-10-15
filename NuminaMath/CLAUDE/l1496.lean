import Mathlib

namespace NUMINAMATH_CALUDE_megan_pop_albums_l1496_149636

/-- The number of songs on each album -/
def songs_per_album : ℕ := 7

/-- The number of country albums bought -/
def country_albums : ℕ := 2

/-- The total number of songs bought -/
def total_songs : ℕ := 70

/-- The number of pop albums bought -/
def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem megan_pop_albums : pop_albums = 8 := by sorry

end NUMINAMATH_CALUDE_megan_pop_albums_l1496_149636


namespace NUMINAMATH_CALUDE_chord_distance_from_center_l1496_149698

theorem chord_distance_from_center (R : ℝ) (chord_length : ℝ) (h1 : R = 13) (h2 : chord_length = 10) :
  ∃ d : ℝ, d = 12 ∧ d^2 + (chord_length/2)^2 = R^2 :=
by sorry

end NUMINAMATH_CALUDE_chord_distance_from_center_l1496_149698


namespace NUMINAMATH_CALUDE_zoo_ticket_cost_is_correct_l1496_149624

/-- The cost of a zoo entry ticket per person -/
def zoo_ticket_cost : ℝ := 5

/-- The one-way bus fare per person -/
def bus_fare : ℝ := 1.5

/-- The total amount of money brought -/
def total_amount : ℝ := 40

/-- The amount left after buying tickets and paying for bus fare -/
def amount_left : ℝ := 24

/-- The number of people -/
def num_people : ℕ := 2

theorem zoo_ticket_cost_is_correct : 
  zoo_ticket_cost = (total_amount - amount_left - 2 * num_people * bus_fare) / num_people := by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_cost_is_correct_l1496_149624


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1496_149685

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 3}

theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1496_149685


namespace NUMINAMATH_CALUDE_night_crew_ratio_l1496_149680

theorem night_crew_ratio (day_workers : ℝ) (night_workers : ℝ) (boxes_per_day_worker : ℝ) 
  (h1 : day_workers > 0)
  (h2 : night_workers > 0)
  (h3 : boxes_per_day_worker > 0)
  (h4 : day_workers * boxes_per_day_worker = 0.7 * (day_workers * boxes_per_day_worker + night_workers * (3/4 * boxes_per_day_worker))) :
  night_workers / day_workers = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_night_crew_ratio_l1496_149680


namespace NUMINAMATH_CALUDE_remainder_of_difference_l1496_149634

theorem remainder_of_difference (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (ha_mod : a % 6 = 2) (hb_mod : b % 6 = 3) (hab : a > b) : 
  (a - b) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_difference_l1496_149634


namespace NUMINAMATH_CALUDE_min_value_of_x_l1496_149646

theorem min_value_of_x (x : ℝ) : 
  (∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) → 
  x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_min_value_of_x_l1496_149646


namespace NUMINAMATH_CALUDE_xy_sum_problem_l1496_149639

theorem xy_sum_problem (x y : ℕ) 
  (pos_x : x > 0) (pos_y : y > 0)
  (bound_x : x < 30) (bound_y : y < 30)
  (eq_condition : x + y + x * y = 119) :
  x + y = 24 ∨ x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_problem_l1496_149639


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l1496_149651

/-- The area of a rhombus formed by two equilateral triangles in a square -/
theorem rhombus_area_in_square (square_side : ℝ) (h_square_side : square_side = 4) :
  let triangle_height := square_side * (Real.sqrt 3) / 2
  let vertical_overlap := 2 * triangle_height - square_side
  let rhombus_area := (vertical_overlap * square_side) / 2
  rhombus_area = 8 * Real.sqrt 3 - 8 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_in_square_l1496_149651


namespace NUMINAMATH_CALUDE_smallest_perimeter_consecutive_sides_l1496_149622

theorem smallest_perimeter_consecutive_sides (a b c : ℕ) : 
  a > 2 →
  b = a + 1 →
  c = a + 2 →
  (∀ x y z : ℕ, x > 2 ∧ y = x + 1 ∧ z = x + 2 → a + b + c ≤ x + y + z) →
  a + b + c = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_consecutive_sides_l1496_149622


namespace NUMINAMATH_CALUDE_olympic_medal_awards_l1496_149689

/-- The number of ways to award medals in the Olympic 100-meter finals -/
def medal_awards (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medals := non_american_sprinters.descFactorial medals
  let one_american_medal := american_sprinters * medals * (non_american_sprinters.descFactorial (medals - 1))
  no_american_medals + one_american_medal

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem olympic_medal_awards :
  medal_awards 10 4 3 = 480 :=
by sorry

end NUMINAMATH_CALUDE_olympic_medal_awards_l1496_149689


namespace NUMINAMATH_CALUDE_special_functions_at_zero_l1496_149621

/-- Two non-constant functions satisfying specific addition formulas -/
class SpecialFunctions (f g : ℝ → ℝ) : Prop where
  add_f : ∀ x y, f (x + y) = f x * g y + g x * f y
  add_g : ∀ x y, g (x + y) = g x * g y - f x * f y
  non_constant_f : ∃ x y, f x ≠ f y
  non_constant_g : ∃ x y, g x ≠ g y

/-- The values of f(0) and g(0) for special functions f and g -/
theorem special_functions_at_zero {f g : ℝ → ℝ} [SpecialFunctions f g] :
  f 0 = 0 ∧ g 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_functions_at_zero_l1496_149621


namespace NUMINAMATH_CALUDE_nested_fourth_root_solution_l1496_149691

/-- The positive solution to the nested fourth root equation --/
noncomputable def x : ℝ := 3.1412

/-- The left-hand side of the equation --/
noncomputable def lhs (x : ℝ) : ℝ := Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

/-- The right-hand side of the equation --/
noncomputable def rhs (x : ℝ) : ℝ := Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

/-- Theorem stating that x is the positive solution to the equation --/
theorem nested_fourth_root_solution :
  lhs x = rhs x ∧ x > 0 := by sorry

end NUMINAMATH_CALUDE_nested_fourth_root_solution_l1496_149691


namespace NUMINAMATH_CALUDE_largest_parabolic_slice_l1496_149606

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a circle in 3D space -/
structure Circle3D where
  center : Point3D
  radius : ℝ

/-- Represents a cone in 3D space -/
structure Cone where
  vertex : Point3D
  base : Circle3D

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Calculates the area of a parabolic slice -/
def parabolicSliceArea (cone : Cone) (plane : Plane) : ℝ := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (p : Point3D) (a : Point3D) (b : Point3D) : Prop := sorry

/-- Theorem: The largest area parabolic slice is obtained when the midpoint of
    the intersection of the cutting plane with the base circle bisects AO -/
theorem largest_parabolic_slice (cone : Cone) (plane : Plane) :
  let A := sorry -- Point on base circle
  let O := cone.base.center
  let E := sorry -- Midpoint of intersection of plane with base circle
  (∀ p : Plane, parabolicSliceArea cone p ≤ parabolicSliceArea cone plane) ↔
  isMidpoint E A O := by sorry

end NUMINAMATH_CALUDE_largest_parabolic_slice_l1496_149606


namespace NUMINAMATH_CALUDE_tom_car_distribution_l1496_149612

theorem tom_car_distribution (total_packages : ℕ) (cars_per_package : ℕ) (num_nephews : ℕ) (cars_remaining : ℕ) :
  total_packages = 10 →
  cars_per_package = 5 →
  num_nephews = 2 →
  cars_remaining = 30 →
  (total_packages * cars_per_package - cars_remaining) / (num_nephews * (total_packages * cars_per_package)) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tom_car_distribution_l1496_149612


namespace NUMINAMATH_CALUDE_count_satisfying_integers_l1496_149672

-- Define the function f(n)
def f (n : ℤ) : ℤ := ⌈(99 * n : ℚ) / 100⌉ - ⌊(100 * n : ℚ) / 101⌋

-- State the theorem
theorem count_satisfying_integers :
  (∃ (S : Finset ℤ), (∀ n ∈ S, f n = 1) ∧ S.card = 10100 ∧
    (∀ n : ℤ, f n = 1 → n ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_count_satisfying_integers_l1496_149672


namespace NUMINAMATH_CALUDE_rd_participation_and_optimality_l1496_149658

/-- Represents a firm engaged in R&D -/
structure Firm where
  participates : Bool

/-- Represents the R&D scenario in country A -/
structure RDScenario where
  V : ℝ  -- Value of successful solo development
  α : ℝ  -- Probability of success
  IC : ℝ  -- Investment cost
  firms : Fin 2 → Firm

/-- Expected revenue for a firm when both participate -/
def expectedRevenueBoth (s : RDScenario) : ℝ :=
  s.α * (1 - s.α) * s.V + 0.5 * s.α^2 * s.V

/-- Expected revenue for a firm when only one participates -/
def expectedRevenueOne (s : RDScenario) : ℝ :=
  s.α * s.V

/-- Condition for both firms to participate -/
def bothParticipateCondition (s : RDScenario) : Prop :=
  s.V * s.α * (1 - 0.5 * s.α) ≥ s.IC

/-- Total expected profit when both firms participate -/
def totalProfitBoth (s : RDScenario) : ℝ :=
  2 * (expectedRevenueBoth s - s.IC)

/-- Total expected profit when only one firm participates -/
def totalProfitOne (s : RDScenario) : ℝ :=
  expectedRevenueOne s - s.IC

/-- Theorem stating the conditions for participation and social optimality -/
theorem rd_participation_and_optimality (s : RDScenario) 
    (h_α_pos : 0 < s.α) (h_α_lt_one : s.α < 1) :
  bothParticipateCondition s ↔ 
    expectedRevenueBoth s ≥ s.IC ∧
    (s.V = 16 ∧ s.α = 0.5 ∧ s.IC = 5 → 
      bothParticipateCondition s ∧ totalProfitOne s > totalProfitBoth s) :=
sorry

end NUMINAMATH_CALUDE_rd_participation_and_optimality_l1496_149658


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1496_149679

def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1496_149679


namespace NUMINAMATH_CALUDE_cos_squared_pi_third_minus_x_l1496_149661

theorem cos_squared_pi_third_minus_x (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.cos (π/3 - x) ^ 2 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_pi_third_minus_x_l1496_149661


namespace NUMINAMATH_CALUDE_expression_factorization_l1496_149601

theorem expression_factorization (b : ℝ) : 
  (8 * b^3 + 45 * b^2 - 10) - (-12 * b^3 + 5 * b^2 - 10) = 20 * b^2 * (b + 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1496_149601


namespace NUMINAMATH_CALUDE_widest_strip_width_l1496_149665

theorem widest_strip_width (w1 w2 w3 : ℕ) (hw1 : w1 = 45) (hw2 : w2 = 60) (hw3 : w3 = 70) :
  Nat.gcd w1 (Nat.gcd w2 w3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_widest_strip_width_l1496_149665


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1496_149655

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℝ

/-- The problem statement -/
theorem ellipse_intersection_theorem (C : Ellipse) (l₁ : Line) :
  -- The ellipse passes through (2, 1)
  (2 / C.a)^2 + (1 / C.b)^2 = 1 →
  -- The eccentricity is √3/2
  (C.a^2 - C.b^2) / C.a^2 = 3/4 →
  -- There exists a point M on x - y + 2√6 = 0 such that MPQ is equilateral
  ∃ (M : ℝ × ℝ), M.1 - M.2 + 2 * Real.sqrt 6 = 0 ∧
    -- (Condition for equilateral triangle, simplified)
    (M.1^2 + M.2^2) = 3 * ((C.a * C.b * l₁.slope / Real.sqrt (C.a^2 * l₁.slope^2 + C.b^2))^2 + 
                           (C.a * C.b / Real.sqrt (C.a^2 * l₁.slope^2 + C.b^2))^2) →
  -- Then l₁ is either y = 0 or y = 2x/7
  l₁.slope = 0 ∨ l₁.slope = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1496_149655


namespace NUMINAMATH_CALUDE_book_loss_percentage_l1496_149620

/-- If the cost price of 8 books equals the selling price of 16 books, then the loss percentage is 50%. -/
theorem book_loss_percentage (C S : ℝ) (h : 8 * C = 16 * S) : 
  (C - S) / C * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_loss_percentage_l1496_149620


namespace NUMINAMATH_CALUDE_asterisk_replacement_l1496_149670

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 20) * (x / 80) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l1496_149670


namespace NUMINAMATH_CALUDE_jenny_max_sales_l1496_149682

/-- Represents a neighborhood where Jenny can sell cookies. -/
structure Neighborhood where
  homes : ℕ
  boxesPerHome : ℕ

/-- Calculates the total sales for a given neighborhood. -/
def totalSales (n : Neighborhood) (pricePerBox : ℕ) : ℕ :=
  n.homes * n.boxesPerHome * pricePerBox

/-- Theorem stating that the maximum amount Jenny can make is $50. -/
theorem jenny_max_sales : 
  let neighborhoodA : Neighborhood := { homes := 10, boxesPerHome := 2 }
  let neighborhoodB : Neighborhood := { homes := 5, boxesPerHome := 5 }
  let pricePerBox : ℕ := 2
  max (totalSales neighborhoodA pricePerBox) (totalSales neighborhoodB pricePerBox) = 50 := by
  sorry

end NUMINAMATH_CALUDE_jenny_max_sales_l1496_149682


namespace NUMINAMATH_CALUDE_quarterly_charge_is_80_l1496_149644

/-- The Kwik-e-Tax Center pricing structure and sales data -/
structure TaxCenter where
  federal_charge : ℕ
  state_charge : ℕ
  federal_sold : ℕ
  state_sold : ℕ
  quarterly_sold : ℕ
  total_revenue : ℕ

/-- The charge for quarterly business taxes -/
def quarterly_charge (tc : TaxCenter) : ℕ :=
  (tc.total_revenue - (tc.federal_charge * tc.federal_sold + tc.state_charge * tc.state_sold)) / tc.quarterly_sold

/-- Theorem stating the charge for quarterly business taxes is $80 -/
theorem quarterly_charge_is_80 (tc : TaxCenter) 
  (h1 : tc.federal_charge = 50)
  (h2 : tc.state_charge = 30)
  (h3 : tc.federal_sold = 60)
  (h4 : tc.state_sold = 20)
  (h5 : tc.quarterly_sold = 10)
  (h6 : tc.total_revenue = 4400) :
  quarterly_charge tc = 80 := by
  sorry

#eval quarterly_charge { federal_charge := 50, state_charge := 30, federal_sold := 60, state_sold := 20, quarterly_sold := 10, total_revenue := 4400 }

end NUMINAMATH_CALUDE_quarterly_charge_is_80_l1496_149644


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_plus_one_less_than_zero_l1496_149654

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_squared_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_plus_one_less_than_zero_l1496_149654


namespace NUMINAMATH_CALUDE_cube_side_length_is_three_l1496_149663

/-- Represents a cube with side length n -/
structure Cube where
  n : ℕ

/-- Calculates the total number of faces of all unit cubes after slicing -/
def totalFaces (c : Cube) : ℕ := 6 * c.n^3

/-- Calculates the number of blue faces (surface area of the original cube) -/
def blueFaces (c : Cube) : ℕ := 6 * c.n^2

/-- Theorem: If one-third of all faces are blue, then the cube's side length is 3 -/
theorem cube_side_length_is_three (c : Cube) :
  3 * blueFaces c = totalFaces c → c.n = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_is_three_l1496_149663


namespace NUMINAMATH_CALUDE_intersect_at_two_points_l1496_149649

/-- The first function representing y = 2x^2 - x + 3 --/
def f (x : ℝ) : ℝ := 2 * x^2 - x + 3

/-- The second function representing y = -x^2 + x + 5 --/
def g (x : ℝ) : ℝ := -x^2 + x + 5

/-- The difference function between f and g --/
def h (x : ℝ) : ℝ := f x - g x

/-- Theorem stating that the graphs of f and g intersect at exactly two distinct points --/
theorem intersect_at_two_points : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ h x₁ = 0 ∧ h x₂ = 0 ∧ ∀ x, h x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_intersect_at_two_points_l1496_149649


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1496_149626

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x : ℝ), x = 6 ∧ x^2 = a^2 + b^2) →
  (∃ (x y : ℝ), y = Real.sqrt 3 * x ∧ b / a = Real.sqrt 3) →
  a^2 = 9 ∧ b^2 = 27 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1496_149626


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1496_149688

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (h1 : r ≠ 1) (h2 : n > 0) :
  let last_term := a * r^(n - 1)
  let series_sum := a * (r^n - 1) / (r - 1)
  (a = 2 ∧ r = 3 ∧ last_term = 4374) → series_sum = 6560 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1496_149688


namespace NUMINAMATH_CALUDE_soup_ratio_l1496_149648

/-- Given the amount of beef bought, unused beef, and vegetables used, 
    calculate the ratio of vegetables to beef used in the soup -/
theorem soup_ratio (beef_bought : ℚ) (unused_beef : ℚ) (vegetables : ℚ) : 
  beef_bought = 4 → unused_beef = 1 → vegetables = 6 →
  vegetables / (beef_bought - unused_beef) = 2 := by sorry

end NUMINAMATH_CALUDE_soup_ratio_l1496_149648


namespace NUMINAMATH_CALUDE_price_reduction_order_invariance_l1496_149603

theorem price_reduction_order_invariance :
  let reduction1 := 0.1
  let reduction2 := 0.15
  let total_reduction1 := 1 - (1 - reduction1) * (1 - reduction2)
  let total_reduction2 := 1 - (1 - reduction2) * (1 - reduction1)
  total_reduction1 = total_reduction2 ∧ total_reduction1 = 0.235 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_order_invariance_l1496_149603


namespace NUMINAMATH_CALUDE_kerman_triple_49_64_15_l1496_149693

/-- Definition of a Kerman triple -/
def is_kerman_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: (49, 64, 15) is a Kerman triple -/
theorem kerman_triple_49_64_15 :
  is_kerman_triple 49 64 15 := by
  sorry

end NUMINAMATH_CALUDE_kerman_triple_49_64_15_l1496_149693


namespace NUMINAMATH_CALUDE_last_digit_of_A_l1496_149674

theorem last_digit_of_A (A : ℕ) : 
  A = (2+1)*(2^2+1)*(2^4+1)*(2^8+1)+1 → 
  A % 10 = (2^16) % 10 := by
sorry

end NUMINAMATH_CALUDE_last_digit_of_A_l1496_149674


namespace NUMINAMATH_CALUDE_gas_measurement_l1496_149611

/-- Represents the ratio of inches to liters per minute for liquid -/
def liquid_ratio : ℚ := 2.5 / 60

/-- Represents the movement ratio of gas compared to liquid -/
def gas_movement_ratio : ℚ := 1 / 2

/-- Represents the amount of gas that passed through the rotameter in liters -/
def gas_volume : ℚ := 192

/-- Calculates the number of inches measured for the gas phase -/
def gas_inches : ℚ := (gas_volume * liquid_ratio) / gas_movement_ratio

/-- Theorem stating that the number of inches measured for the gas phase is 4 -/
theorem gas_measurement :
  gas_inches = 4 := by sorry

end NUMINAMATH_CALUDE_gas_measurement_l1496_149611


namespace NUMINAMATH_CALUDE_divisibility_of_245245_by_35_l1496_149653

theorem divisibility_of_245245_by_35 : 35 ∣ 245245 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_245245_by_35_l1496_149653


namespace NUMINAMATH_CALUDE_balloon_solution_l1496_149619

/-- The number of balloons Allan and Jake have in the park -/
def balloon_problem (allan_balloons jake_initial_balloons jake_bought_balloons : ℕ) : Prop :=
  allan_balloons - (jake_initial_balloons + jake_bought_balloons) = 1

/-- Theorem stating the solution to the balloon problem -/
theorem balloon_solution :
  balloon_problem 6 2 3 := by
  sorry

end NUMINAMATH_CALUDE_balloon_solution_l1496_149619


namespace NUMINAMATH_CALUDE_min_pieces_for_rearrangement_l1496_149696

/-- Represents a shape made of small squares -/
structure Shape :=
  (squares : Nat)

/-- Represents the goal configuration -/
structure GoalSquare :=
  (side : Nat)

/-- Represents a cutting of the shape into pieces -/
structure Cutting :=
  (num_pieces : Nat)

/-- Predicate to check if a cutting is valid for rearrangement -/
def is_valid_cutting (s : Shape) (g : GoalSquare) (c : Cutting) : Prop :=
  c.num_pieces ≥ 1 ∧ c.num_pieces ≤ s.squares

/-- Predicate to check if a cutting allows rearrangement into the goal square -/
def allows_rearrangement (s : Shape) (g : GoalSquare) (c : Cutting) : Prop :=
  is_valid_cutting s g c ∧ s.squares = g.side * g.side

/-- The main theorem stating the minimum number of pieces required -/
theorem min_pieces_for_rearrangement (s : Shape) (g : GoalSquare) :
  s.squares = 9 → g.side = 3 →
  ∃ (c : Cutting), 
    c.num_pieces = 3 ∧ 
    allows_rearrangement s g c ∧
    ∀ (c' : Cutting), allows_rearrangement s g c' → c'.num_pieces ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_pieces_for_rearrangement_l1496_149696


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1496_149683

theorem complex_equation_solution : ∃ (a : ℝ), 
  (1 - Complex.I : ℂ) = (2 + a * Complex.I) / (1 + Complex.I) ∧ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1496_149683


namespace NUMINAMATH_CALUDE_largest_band_members_l1496_149687

theorem largest_band_members : ∃ (m r x : ℕ),
  m < 100 ∧
  r * x + 3 = m ∧
  (r - 3) * (x + 1) = m ∧
  ∀ (m' r' x' : ℕ),
    m' < 100 →
    r' * x' + 3 = m' →
    (r' - 3) * (x' + 1) = m' →
    m' ≤ m ∧
  m = 87 := by
sorry

end NUMINAMATH_CALUDE_largest_band_members_l1496_149687


namespace NUMINAMATH_CALUDE_set_intersection_and_complement_l1496_149678

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}

-- State the theorem
theorem set_intersection_and_complement :
  (A ∩ B = {x | 1 < x ∧ x ≤ 2}) ∧
  ((Aᶜ ∩ Bᶜ) = {x | -3 ≤ x ∧ x ≤ 0}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_complement_l1496_149678


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l1496_149610

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l1496_149610


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1496_149690

theorem arithmetic_sequence_first_term
  (a d : ℝ)
  (sum_100 : (100 : ℝ) / 2 * (2 * a + 99 * d) = 1800)
  (sum_51_to_150 : (100 : ℝ) / 2 * (2 * a + 199 * d) = 6300) :
  a = -26.55 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1496_149690


namespace NUMINAMATH_CALUDE_pipe_length_is_35_l1496_149643

/-- The length of the pipe in meters -/
def pipe_length : ℝ := 35

/-- The length of Yura's step in meters -/
def step_length : ℝ := 1

/-- The number of steps Yura took against the movement of the tractor -/
def steps_against : ℕ := 20

/-- The number of steps Yura took with the movement of the tractor -/
def steps_with : ℕ := 140

/-- Theorem stating that the pipe length is 35 meters -/
theorem pipe_length_is_35 : 
  ∃ (x : ℝ), 
    (step_length * steps_against : ℝ) = pipe_length - x ∧ 
    (step_length * steps_with : ℝ) = pipe_length + 7 * x ∧
    pipe_length = 35 := by sorry

end NUMINAMATH_CALUDE_pipe_length_is_35_l1496_149643


namespace NUMINAMATH_CALUDE_share_of_b_l1496_149656

theorem share_of_b (A B C : ℕ) : 
  A = 3 * B → 
  B = C + 25 → 
  A + B + C = 645 → 
  B = 134 := by
sorry

end NUMINAMATH_CALUDE_share_of_b_l1496_149656


namespace NUMINAMATH_CALUDE_floor_ceiling_product_l1496_149686

theorem floor_ceiling_product : ⌊(3.999 : ℝ)⌋ * ⌈(0.002 : ℝ)⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_product_l1496_149686


namespace NUMINAMATH_CALUDE_restaurant_menu_combinations_l1496_149659

theorem restaurant_menu_combinations : 
  (12 * 11) * (12 * 10) = 15840 := by sorry

end NUMINAMATH_CALUDE_restaurant_menu_combinations_l1496_149659


namespace NUMINAMATH_CALUDE_girl_multiplication_problem_l1496_149684

theorem girl_multiplication_problem (incorrect_multiplier : ℕ) (difference : ℕ) (base_number : ℕ) :
  incorrect_multiplier = 34 →
  difference = 1242 →
  base_number = 138 →
  ∃ (correct_multiplier : ℕ), 
    base_number * correct_multiplier = base_number * incorrect_multiplier + difference ∧
    correct_multiplier = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_girl_multiplication_problem_l1496_149684


namespace NUMINAMATH_CALUDE_correct_ways_to_leave_shop_l1496_149632

/-- The number of different flavors of oreos --/
def num_oreo_flavors : ℕ := 6

/-- The number of different flavors of milk --/
def num_milk_flavors : ℕ := 4

/-- The total number of product types (oreos + milk) --/
def total_product_types : ℕ := num_oreo_flavors + num_milk_flavors

/-- The number of products they leave the shop with --/
def num_products : ℕ := 4

/-- Function to calculate the number of ways Alpha and Beta can leave the shop --/
def ways_to_leave_shop : ℕ := sorry

/-- Theorem stating the correct number of ways to leave the shop --/
theorem correct_ways_to_leave_shop : ways_to_leave_shop = 2546 := by sorry

end NUMINAMATH_CALUDE_correct_ways_to_leave_shop_l1496_149632


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1496_149666

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1496_149666


namespace NUMINAMATH_CALUDE_focal_lengths_equal_l1496_149673

/-- Focal length of a hyperbola with equation 15y^2 - x^2 = 15 -/
def hyperbola_focal_length : ℝ := 4

/-- Focal length of an ellipse with equation x^2/25 + y^2/9 = 1 -/
def ellipse_focal_length : ℝ := 4

/-- The focal lengths of the given hyperbola and ellipse are equal -/
theorem focal_lengths_equal : hyperbola_focal_length = ellipse_focal_length := by sorry

end NUMINAMATH_CALUDE_focal_lengths_equal_l1496_149673


namespace NUMINAMATH_CALUDE_average_of_abc_l1496_149600

theorem average_of_abc (A B C : ℝ) 
  (eq1 : 1001 * C - 2002 * A = 4004)
  (eq2 : 1001 * B + 3003 * A = 5005) : 
  (A + B + C) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_abc_l1496_149600


namespace NUMINAMATH_CALUDE_triangle_exterior_angle_theorem_l1496_149633

/-- 
Given a triangle where one side is extended:
- ext_angle is the exterior angle
- int_angle1 is one of the non-adjacent interior angles
- int_angle2 is the other non-adjacent interior angle
-/
theorem triangle_exterior_angle_theorem 
  (ext_angle int_angle1 int_angle2 : ℝ) : 
  ext_angle = 154 ∧ int_angle1 = 58 → int_angle2 = 96 := by
sorry

end NUMINAMATH_CALUDE_triangle_exterior_angle_theorem_l1496_149633


namespace NUMINAMATH_CALUDE_part1_part2_l1496_149629

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - (a-1)*x + a-2

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ -2) ↔ (3 - 2*Real.sqrt 2 ≤ a ∧ a ≤ 3 + 2*Real.sqrt 2) :=
sorry

-- Part 2
theorem part2 (a x : ℝ) :
  (a < 3 → (f a x < 0 ↔ a-2 < x ∧ x < 1)) ∧
  (a = 3 → ¬∃ x, f a x < 0) ∧
  (a > 3 → (f a x < 0 ↔ 1 < x ∧ x < a-2)) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l1496_149629


namespace NUMINAMATH_CALUDE_distribute_teachers_count_l1496_149681

/-- The number of ways to distribute 6 teachers across 4 neighborhoods --/
def distribute_teachers : ℕ :=
  let n_teachers : ℕ := 6
  let n_neighborhoods : ℕ := 4
  let distribution_3111 : ℕ := (Nat.choose n_teachers 3) * (Nat.factorial n_neighborhoods)
  let distribution_2211 : ℕ := 
    (Nat.choose n_teachers 2) * (Nat.choose (n_teachers - 2) 2) * 
    (Nat.factorial n_neighborhoods) / (Nat.factorial 2)
  distribution_3111 + distribution_2211

/-- Theorem stating that the number of distribution schemes is 1560 --/
theorem distribute_teachers_count : distribute_teachers = 1560 := by
  sorry

end NUMINAMATH_CALUDE_distribute_teachers_count_l1496_149681


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l1496_149608

/-- The x-coordinate of the point on the x-axis equidistant from A(-4, 0) and B(2, 6) is 2 -/
theorem equidistant_point_x_coordinate : 
  ∃ (x : ℝ), (x + 4)^2 = (x - 2)^2 + 36 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l1496_149608


namespace NUMINAMATH_CALUDE_gas_diffusion_rate_and_molar_mass_l1496_149609

theorem gas_diffusion_rate_and_molar_mass 
  (r_unknown r_O2 : ℝ) 
  (M_unknown M_O2 : ℝ) 
  (h1 : r_unknown / r_O2 = 1 / 3) 
  (h2 : r_unknown / r_O2 = Real.sqrt (M_O2 / M_unknown)) :
  M_unknown = 9 * M_O2 := by
  sorry

end NUMINAMATH_CALUDE_gas_diffusion_rate_and_molar_mass_l1496_149609


namespace NUMINAMATH_CALUDE_log_base_six_two_point_five_l1496_149605

theorem log_base_six_two_point_five (x : ℝ) :
  (Real.log x) / (Real.log 6) = 2.5 → x = 36 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_log_base_six_two_point_five_l1496_149605


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1496_149623

theorem solution_set_of_inequality (x : ℝ) :
  (((x - 2) / (x + 3) > 0) ↔ (x ∈ Set.Iio (-3) ∪ Set.Ioi 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1496_149623


namespace NUMINAMATH_CALUDE_zoe_songs_total_l1496_149602

theorem zoe_songs_total (country_albums : ℕ) (pop_albums : ℕ) (songs_per_album : ℕ) : 
  country_albums = 3 → pop_albums = 5 → songs_per_album = 3 →
  (country_albums + pop_albums) * songs_per_album = 24 := by
sorry

end NUMINAMATH_CALUDE_zoe_songs_total_l1496_149602


namespace NUMINAMATH_CALUDE_children_fed_theorem_l1496_149630

/-- Represents the number of people a meal can feed -/
structure MealCapacity where
  adults : ℕ
  children : ℕ

/-- Calculates the number of children that can be fed with the remaining food -/
def remainingChildrenFed (totalAdults totalChildren consumedAdultMeals : ℕ) (capacity : MealCapacity) : ℕ :=
  let remainingAdultMeals := capacity.adults - consumedAdultMeals
  let childrenPerAdultMeal := capacity.children / capacity.adults
  remainingAdultMeals * childrenPerAdultMeal

/-- Theorem stating that given the conditions, 63 children can be fed with the remaining food -/
theorem children_fed_theorem (totalAdults totalChildren consumedAdultMeals : ℕ) (capacity : MealCapacity) :
  totalAdults = 55 →
  totalChildren = 70 →
  capacity.adults = 70 →
  capacity.children = 90 →
  consumedAdultMeals = 21 →
  remainingChildrenFed totalAdults totalChildren consumedAdultMeals capacity = 63 := by
  sorry

end NUMINAMATH_CALUDE_children_fed_theorem_l1496_149630


namespace NUMINAMATH_CALUDE_special_function_range_l1496_149675

/-- A monotonically increasing function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x → 0 < y → f (x * y) = f x + f y) ∧
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y) ∧
  (f 3 = 1)

/-- The theorem statement -/
theorem special_function_range (f : ℝ → ℝ) (hf : SpecialFunction f) :
  {x : ℝ | 0 < x ∧ f x + f (x - 8) ≤ 2} = Set.Ioo 8 9 := by
  sorry

end NUMINAMATH_CALUDE_special_function_range_l1496_149675


namespace NUMINAMATH_CALUDE_triangle_property_l1496_149631

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- S is the area of triangle ABC -/
def area (t : Triangle) : ℝ := sorry

/-- Main theorem -/
theorem triangle_property (t : Triangle) (h : 4 * Real.sqrt 3 * area t = t.a^2 - (t.b - t.c)^2) :
  t.A = 2 * Real.pi / 3 ∧ 2 / 3 ≤ (t.b^2 + t.c^2) / t.a^2 ∧ (t.b^2 + t.c^2) / t.a^2 < 1 :=
by sorry

end

end NUMINAMATH_CALUDE_triangle_property_l1496_149631


namespace NUMINAMATH_CALUDE_complex_square_l1496_149640

theorem complex_square (z : ℂ) (i : ℂ) (h1 : z = 5 - 3 * i) (h2 : i^2 = -1) :
  z^2 = 34 - 30 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l1496_149640


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1496_149695

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1496_149695


namespace NUMINAMATH_CALUDE_school_trip_probabilities_l1496_149662

/-- Represents the setup of a school trip with students and a teacher assigned to cities. -/
structure SchoolTrip where
  numStudents : Nat
  numCities : Nat
  studentsPerCity : Nat

/-- Defines the probability of event A: student a and the teacher go to the same city. -/
def probA (trip : SchoolTrip) : ℚ :=
  1 / trip.numCities

/-- Defines the probability of event B: students a and b go to the same city. -/
def probB (trip : SchoolTrip) : ℚ :=
  1 / (trip.numStudents - 1)

/-- Defines the expected value of ξ, the total number of occurrences of events A and B. -/
def expectedXi (trip : SchoolTrip) : ℚ :=
  8 / 15

/-- Theorem stating the probabilities and expected value for the given school trip scenario. -/
theorem school_trip_probabilities (trip : SchoolTrip) :
  trip.numStudents = 6 ∧ trip.numCities = 3 ∧ trip.studentsPerCity = 2 →
  probA trip = 1/3 ∧ probB trip = 1/5 ∧ expectedXi trip = 8/15 := by
  sorry


end NUMINAMATH_CALUDE_school_trip_probabilities_l1496_149662


namespace NUMINAMATH_CALUDE_fixed_points_of_f_squared_l1496_149604

def X := ℤ × ℤ × ℤ

def f (x : X) : X :=
  let (a, b, c) := x
  (a + b + c, a * b + b * c + c * a, a * b * c)

theorem fixed_points_of_f_squared (a b c : ℤ) :
  f (f (a, b, c)) = (a, b, c) ↔ 
    ((∃ k : ℤ, (a, b, c) = (k, 0, 0)) ∨ (a, b, c) = (-1, -1, 1)) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_squared_l1496_149604


namespace NUMINAMATH_CALUDE_program_attendance_l1496_149694

/-- The total number of people present at the program -/
def total_people (parents pupils teachers staff family_members : ℕ) : ℕ :=
  parents + pupils + teachers + staff + family_members

/-- The number of family members accompanying the pupils -/
def accompanying_family_members (pupils : ℕ) : ℕ :=
  (pupils / 6) * 2

theorem program_attendance : 
  let parents : ℕ := 83
  let pupils : ℕ := 956
  let teachers : ℕ := 154
  let staff : ℕ := 27
  let family_members : ℕ := accompanying_family_members pupils
  total_people parents pupils teachers staff family_members = 1379 := by
sorry

end NUMINAMATH_CALUDE_program_attendance_l1496_149694


namespace NUMINAMATH_CALUDE_simplify_expression_l1496_149697

theorem simplify_expression (y : ℝ) : 5*y - 3*y + 7*y - 2*y + 6*y = 13*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1496_149697


namespace NUMINAMATH_CALUDE_inequality_not_true_l1496_149642

theorem inequality_not_true : Real.sqrt 2 + Real.sqrt 10 ≤ 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_true_l1496_149642


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1496_149676

theorem inequality_solution_set (x : ℝ) : 
  abs (2 * x - 1) + abs (2 * x + 1) ≤ 6 ↔ x ∈ Set.Icc (-3/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1496_149676


namespace NUMINAMATH_CALUDE_shirt_cost_l1496_149618

theorem shirt_cost (jeans_cost shirt_cost : ℚ) : 
  (3 * jeans_cost + 2 * shirt_cost = 69) →
  (2 * jeans_cost + 3 * shirt_cost = 61) →
  shirt_cost = 9 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l1496_149618


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l1496_149647

-- Define the triangle
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.angleA = 2 * t.angleB ∧
  t.angleC > Real.pi / 2 ∧
  t.angleA + t.angleB + t.angleC = Real.pi

-- Define the perimeter
def perimeter (t : Triangle) : ℕ :=
  t.a.val + t.b.val + t.c.val

-- Theorem statement
theorem min_perimeter_triangle :
  ∃ (t : Triangle), validTriangle t ∧
    (∀ (t' : Triangle), validTriangle t' → perimeter t ≤ perimeter t') ∧
    perimeter t = 77 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l1496_149647


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1496_149638

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4
  let θ : ℝ := 3 * Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = -2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1496_149638


namespace NUMINAMATH_CALUDE_a_value_when_A_equals_B_l1496_149664

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + 1 = 0}

-- Define the set B
def B : Set ℝ := {1, 2}

-- Theorem statement
theorem a_value_when_A_equals_B (a : ℝ) : A a = B → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_value_when_A_equals_B_l1496_149664


namespace NUMINAMATH_CALUDE_sum_of_squares_reciprocals_l1496_149652

theorem sum_of_squares_reciprocals (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_reciprocals_l1496_149652


namespace NUMINAMATH_CALUDE_jack_classic_collection_l1496_149635

/-- The number of books each author has in Jack's classic collection -/
def books_per_author (total_books : ℕ) (num_authors : ℕ) : ℕ :=
  total_books / num_authors

/-- Theorem stating that each author has 33 books in Jack's classic collection -/
theorem jack_classic_collection :
  let total_books : ℕ := 198
  let num_authors : ℕ := 6
  books_per_author total_books num_authors = 33 := by
sorry

end NUMINAMATH_CALUDE_jack_classic_collection_l1496_149635


namespace NUMINAMATH_CALUDE_light_distance_250_years_l1496_149625

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℝ := 250

/-- The theorem stating the distance light travels in 250 years -/
theorem light_distance_250_years : 
  light_year_distance * years = 1.4675 * (10 : ℝ) ^ 15 := by
  sorry

end NUMINAMATH_CALUDE_light_distance_250_years_l1496_149625


namespace NUMINAMATH_CALUDE_smallest_winning_number_l1496_149613

theorem smallest_winning_number : ∃ N : ℕ, 
  (N = 6) ∧ 
  (8 * N + 450 < 500) ∧ 
  (N ≤ 499) ∧ 
  (∀ m : ℕ, m < N → (8 * m + 450 ≥ 500) ∨ m > 499) :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l1496_149613


namespace NUMINAMATH_CALUDE_gary_remaining_money_l1496_149677

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem: Gary's remaining money -/
theorem gary_remaining_money :
  remaining_money 73 55 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gary_remaining_money_l1496_149677


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1496_149645

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x-1) + 2
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1496_149645


namespace NUMINAMATH_CALUDE_product_of_specific_numbers_l1496_149660

theorem product_of_specific_numbers (x y : ℝ) 
  (h1 : x - y = 6) 
  (h2 : x^3 - y^3 = 198) : 
  x * y = 5 := by
sorry

end NUMINAMATH_CALUDE_product_of_specific_numbers_l1496_149660


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1496_149669

theorem imaginary_part_of_z (z : ℂ) : z - Complex.I = (4 - 2 * Complex.I) / (1 + 2 * Complex.I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1496_149669


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l1496_149671

-- Define the number to factorize
def n : ℕ := 65535

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor n) = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l1496_149671


namespace NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l1496_149699

/-- A natural number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬(Nat.Prime n)

/-- The number of factors of a natural number -/
def numFactors (n : ℕ) : ℕ :=
  (Nat.divisors n).card

/-- Theorem: Any composite number has at least 3 factors -/
theorem composite_has_at_least_three_factors (n : ℕ) (h : IsComposite n) :
  numFactors n ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l1496_149699


namespace NUMINAMATH_CALUDE_stating_anoop_join_time_l1496_149641

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Represents Arjun's investment in rupees -/
def arjunInvestment : ℕ := 20000

/-- Represents Anoop's investment in rupees -/
def anoopInvestment : ℕ := 4000

/-- 
Theorem stating that if Arjun invests for 12 months and Anoop invests for (12 - x) months,
and their profits are divided equally, then Anoop must have joined after 7 months.
-/
theorem anoop_join_time (x : ℕ) : 
  (arjunInvestment * monthsInYear) / (anoopInvestment * (monthsInYear - x)) = 1 → x = 7 := by
  sorry


end NUMINAMATH_CALUDE_stating_anoop_join_time_l1496_149641


namespace NUMINAMATH_CALUDE_triangular_difference_2015_l1496_149607

theorem triangular_difference_2015 : ∃ (n k : ℕ), 
  1000 ≤ n * (n + 1) / 2 ∧ n * (n + 1) / 2 < 10000 ∧
  1000 ≤ k * (k + 1) / 2 ∧ k * (k + 1) / 2 < 10000 ∧
  n * (n + 1) / 2 - k * (k + 1) / 2 = 2015 :=
by sorry


end NUMINAMATH_CALUDE_triangular_difference_2015_l1496_149607


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l1496_149627

def f (x : ℝ) := x^2 - 1

theorem inequality_holds_iff_m_in_range :
  ∀ m : ℝ, (∀ x ≥ 3, f (x / m) - 4 * m^2 * f x ≤ f (x - 1) + 4 * f m) ↔
    m ≤ -Real.sqrt 2 / 2 ∨ m ≥ Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l1496_149627


namespace NUMINAMATH_CALUDE_fir_trees_count_l1496_149667

theorem fir_trees_count : ∃ (n : ℕ), 
  (n ≠ 15) ∧ 
  (n % 11 = 0) ∧ 
  (n < 25) ∧ 
  (n % 22 ≠ 0) ∧
  (n = 11) := by
  sorry

end NUMINAMATH_CALUDE_fir_trees_count_l1496_149667


namespace NUMINAMATH_CALUDE_rectangle_containment_exists_l1496_149650

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : Nat
  height : Nat

/-- The set of all rectangles with positive integer dimensions -/
def RectangleSet : Set Rectangle :=
  {r : Rectangle | r.width > 0 ∧ r.height > 0}

/-- Predicate to check if one rectangle is contained within another -/
def contains (r1 r2 : Rectangle) : Prop :=
  r1.width ≤ r2.width ∧ r1.height ≤ r2.height

theorem rectangle_containment_exists :
  ∃ r1 r2 : Rectangle, r1 ∈ RectangleSet ∧ r2 ∈ RectangleSet ∧ r1 ≠ r2 ∧ contains r1 r2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_containment_exists_l1496_149650


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_800_l1496_149692

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_800 :
  units_digit (factorial_sum 800) = 3 := by
sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_800_l1496_149692


namespace NUMINAMATH_CALUDE_jeans_pricing_markup_l1496_149657

theorem jeans_pricing_markup (cost : ℝ) (h : cost > 0) :
  let retailer_price := cost * 1.4
  let customer_price := retailer_price * 1.3
  (customer_price - cost) / cost = 0.82 := by
sorry

end NUMINAMATH_CALUDE_jeans_pricing_markup_l1496_149657


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1496_149616

-- Define the markup percentage
def markup : ℝ := 0.15

-- Define the selling price
def selling_price : ℝ := 6400

-- Theorem statement
theorem cost_price_calculation :
  ∃ (cost_price : ℝ), cost_price * (1 + markup) = selling_price :=
by
  sorry


end NUMINAMATH_CALUDE_cost_price_calculation_l1496_149616


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1496_149668

/-- Two planar vectors are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem states that for given planar vectors a = (-6, 2) and b = (3, m),
    if they are perpendicular, then m = 9 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (-6, 2)
  let b : ℝ × ℝ := (3, m)
  perpendicular a b → m = 9 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1496_149668


namespace NUMINAMATH_CALUDE_highest_score_for_given_stats_l1496_149614

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  totalInnings : ℕ
  overallAverage : ℚ
  scoreDifference : ℕ
  averageExcludingExtremes : ℚ

/-- Calculates the highest score given a batsman's statistics -/
def highestScore (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating the highest score for the given conditions -/
theorem highest_score_for_given_stats :
  let stats : BatsmanStats := {
    totalInnings := 46,
    overallAverage := 59,
    scoreDifference := 150,
    averageExcludingExtremes := 58
  }
  highestScore stats = 151 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_for_given_stats_l1496_149614


namespace NUMINAMATH_CALUDE_ralphs_tv_time_l1496_149637

/-- The number of hours Ralph watches TV in one week -/
def total_tv_hours (weekday_hours weekday_days weekend_hours weekend_days : ℕ) : ℕ :=
  weekday_hours * weekday_days + weekend_hours * weekend_days

theorem ralphs_tv_time : total_tv_hours 4 5 6 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ralphs_tv_time_l1496_149637


namespace NUMINAMATH_CALUDE_tetrachloromethane_formation_l1496_149617

-- Define the chemical species
structure ChemicalSpecies where
  name : String
  moles : ℝ

-- Define the reaction equation
structure ReactionEquation where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

-- Define the problem parameters
def methane : ChemicalSpecies := ⟨"CH4", 1⟩
def chlorine : ChemicalSpecies := ⟨"Cl2", 4⟩
def tetrachloromethane : ChemicalSpecies := ⟨"CCl4", 0⟩ -- Initial amount is 0
def hydrogenChloride : ChemicalSpecies := ⟨"HCl", 0⟩ -- Initial amount is 0

-- Define the balanced reaction equation
def balancedEquation : ReactionEquation :=
  ⟨[methane, chlorine], [tetrachloromethane, hydrogenChloride]⟩

-- Theorem statement
theorem tetrachloromethane_formation
  (reactionEq : ReactionEquation)
  (h1 : reactionEq = balancedEquation)
  (h2 : methane.moles = 1)
  (h3 : chlorine.moles = 4) :
  tetrachloromethane.moles = 1 :=
sorry

end NUMINAMATH_CALUDE_tetrachloromethane_formation_l1496_149617


namespace NUMINAMATH_CALUDE_white_ducks_count_l1496_149628

theorem white_ducks_count (fish_per_white : ℕ) (fish_per_black : ℕ) (fish_per_multi : ℕ)
  (black_ducks : ℕ) (multi_ducks : ℕ) (total_fish : ℕ)
  (h1 : fish_per_white = 5)
  (h2 : fish_per_black = 10)
  (h3 : fish_per_multi = 12)
  (h4 : black_ducks = 7)
  (h5 : multi_ducks = 6)
  (h6 : total_fish = 157) :
  ∃ white_ducks : ℕ, white_ducks * fish_per_white + black_ducks * fish_per_black + multi_ducks * fish_per_multi = total_fish ∧ white_ducks = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_white_ducks_count_l1496_149628


namespace NUMINAMATH_CALUDE_jersey_profit_calculation_l1496_149615

/-- The amount of money made from each jersey -/
def jersey_profit : ℝ := 165

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 156

/-- The total money made from selling jerseys -/
def total_jersey_profit : ℝ := jersey_profit * (jerseys_sold : ℝ)

theorem jersey_profit_calculation : total_jersey_profit = 25740 := by
  sorry

end NUMINAMATH_CALUDE_jersey_profit_calculation_l1496_149615
