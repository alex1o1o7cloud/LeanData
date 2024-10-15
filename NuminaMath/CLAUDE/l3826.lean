import Mathlib

namespace NUMINAMATH_CALUDE_jane_calculation_l3826_382682

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - y - z = 7)
  (h2 : x - (y + z) = 19) : 
  x - y = 13 := by sorry

end NUMINAMATH_CALUDE_jane_calculation_l3826_382682


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3826_382672

theorem complex_equation_solution (z : ℂ) : (Complex.I * (z - 1) = 1 - Complex.I) → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3826_382672


namespace NUMINAMATH_CALUDE_wendy_walked_distance_l3826_382606

/-- The number of miles Wendy ran -/
def miles_ran : ℝ := 19.83

/-- The difference between miles ran and walked -/
def difference : ℝ := 10.67

/-- The number of miles Wendy walked -/
def miles_walked : ℝ := miles_ran - difference

theorem wendy_walked_distance : miles_walked = 9.16 := by
  sorry

end NUMINAMATH_CALUDE_wendy_walked_distance_l3826_382606


namespace NUMINAMATH_CALUDE_class_average_marks_l3826_382604

theorem class_average_marks (students1 students2 : ℕ) (avg2 combined_avg : ℚ) :
  students1 = 12 →
  students2 = 28 →
  avg2 = 60 →
  combined_avg = 54 →
  (students1 : ℚ) * (40 : ℚ) + (students2 : ℚ) * avg2 = (students1 + students2 : ℚ) * combined_avg :=
by sorry

end NUMINAMATH_CALUDE_class_average_marks_l3826_382604


namespace NUMINAMATH_CALUDE_intersection_k_value_l3826_382617

-- Define the lines
def line1 (x y k : ℝ) : Prop := 2*x + 3*y - k = 0
def line2 (x y k : ℝ) : Prop := x - k*y + 12 = 0

-- Define the condition that the intersection point lies on the y-axis
def intersection_on_y_axis (k : ℝ) : Prop :=
  ∃ y : ℝ, line1 0 y k ∧ line2 0 y k

-- Theorem statement
theorem intersection_k_value :
  ∀ k : ℝ, intersection_on_y_axis k → (k = 6 ∨ k = -6) :=
by sorry

end NUMINAMATH_CALUDE_intersection_k_value_l3826_382617


namespace NUMINAMATH_CALUDE_correct_lineup_count_l3826_382645

-- Define the total number of players
def total_players : ℕ := 15

-- Define the number of players in the starting lineup
def lineup_size : ℕ := 6

-- Define the number of guaranteed players (All-Stars)
def guaranteed_players : ℕ := 3

-- Define the number of goalkeepers
def goalkeepers : ℕ := 1

-- Define the function to calculate the number of possible lineups
def possible_lineups : ℕ := Nat.choose (total_players - guaranteed_players - goalkeepers) (lineup_size - guaranteed_players - goalkeepers)

-- Theorem statement
theorem correct_lineup_count : possible_lineups = 55 := by
  sorry

end NUMINAMATH_CALUDE_correct_lineup_count_l3826_382645


namespace NUMINAMATH_CALUDE_probability_of_shared_character_l3826_382696

/-- Represents an idiom card -/
structure IdiomCard where
  idiom : String

/-- The set of all idiom cards -/
def idiomCards : Finset IdiomCard := sorry

/-- Two cards share a character -/
def shareCharacter (card1 card2 : IdiomCard) : Prop := sorry

/-- The number of ways to choose 2 cards from the set -/
def totalChoices : Nat := Nat.choose idiomCards.card 2

/-- The number of ways to choose 2 cards that share a character -/
def favorableChoices : Nat := sorry

theorem probability_of_shared_character :
  (favorableChoices : ℚ) / totalChoices = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_shared_character_l3826_382696


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l3826_382662

-- Define the function g(x)
def g (x : ℝ) : ℝ := 18 * x^4 - 20 * x^2 + 3

-- State the theorem
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = 1 ∧ g r = 0 ∧ ∀ x : ℝ, g x = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l3826_382662


namespace NUMINAMATH_CALUDE_binary_10101_is_21_l3826_382642

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_is_21_l3826_382642


namespace NUMINAMATH_CALUDE_jacket_savings_percentage_l3826_382681

/-- Calculates the percentage saved on a purchase given the original price and total savings. -/
def percentage_saved (original_price savings : ℚ) : ℚ :=
  (savings / original_price) * 100

/-- Proves that the total percentage saved on a jacket purchase is 22.5% given the specified conditions. -/
theorem jacket_savings_percentage :
  let original_price : ℚ := 160
  let store_discount : ℚ := 20
  let coupon_savings : ℚ := 16
  let total_savings : ℚ := store_discount + coupon_savings
  percentage_saved original_price total_savings = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_jacket_savings_percentage_l3826_382681


namespace NUMINAMATH_CALUDE_no_equal_sums_l3826_382669

theorem no_equal_sums : ¬∃ (n : ℕ+), 
  (5 * n * (n + 1) : ℚ) / 2 = (5 * n * (n + 7) : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_equal_sums_l3826_382669


namespace NUMINAMATH_CALUDE_simplify_expression_l3826_382647

theorem simplify_expression (a b : ℝ) : (22*a + 60*b) + (10*a + 29*b) - (9*a + 50*b) = 23*a + 39*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3826_382647


namespace NUMINAMATH_CALUDE_trajectory_equation_l3826_382653

/-- The circle C -/
def C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

/-- The fixed point F -/
def F : ℝ × ℝ := (2, 0)

/-- Predicate for a point being on the trajectory of Q -/
def on_trajectory (x y : ℝ) : Prop :=
  ∃ (px py : ℝ),
    C px py ∧
    ∃ (qx qy : ℝ),
      -- Q is on the perpendicular bisector of PF
      (qx - px)^2 + (qy - py)^2 = (qx - F.1)^2 + (qy - F.2)^2 ∧
      -- Q is on the line CP
      (qx + 2) * py = (qy) * (px + 2) ∧
      -- Q is the point we're considering
      qx = x ∧ qy = y

/-- The main theorem -/
theorem trajectory_equation :
  ∀ x y : ℝ, on_trajectory x y ↔ x^2 - y^2/3 = 1 := by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l3826_382653


namespace NUMINAMATH_CALUDE_divisibility_by_forty_l3826_382685

theorem divisibility_by_forty (p : ℕ) (h_prime : Prime p) (h_ge_seven : p ≥ 7) :
  (∃ q : ℕ, Prime q ∧ q ≥ 7 ∧ 40 ∣ (q^2 - 1)) ∧
  (∃ r : ℕ, Prime r ∧ r ≥ 7 ∧ ¬(40 ∣ (r^2 - 1))) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_forty_l3826_382685


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l3826_382699

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (m n : Line) (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : subset m α)
  (h4 : perp m β) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l3826_382699


namespace NUMINAMATH_CALUDE_min_sum_triangular_grid_l3826_382654

/-- Represents a triangular grid with 16 cells --/
structure TriangularGrid :=
  (cells : Fin 16 → ℕ)

/-- Checks if a number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Represents the layers of the triangular grid --/
def layers : List (List (Fin 16)) := sorry

/-- The sum of numbers in each layer is prime --/
def layerSumsPrime (grid : TriangularGrid) : Prop :=
  ∀ layer ∈ layers, isPrime (layer.map grid.cells).sum

/-- The theorem stating the minimum sum of all numbers in the grid --/
theorem min_sum_triangular_grid :
  ∀ grid : TriangularGrid, layerSumsPrime grid →
  (Finset.univ.sum (λ i => grid.cells i) ≥ 22) :=
sorry

end NUMINAMATH_CALUDE_min_sum_triangular_grid_l3826_382654


namespace NUMINAMATH_CALUDE_trapezoid_sides_l3826_382658

-- Define the right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  is_345 : a = 3 ∧ b = 4 ∧ c = 5

-- Define the perpendicular line
def perpendicular_line (t : RightTriangle) (d : ℝ) : Prop :=
  d = 1 ∨ d = t.c - 1

-- Define the trapezoid formed by the perpendicular line
structure Trapezoid where
  s1 : ℝ
  s2 : ℝ
  s3 : ℝ
  s4 : ℝ

-- Theorem statement
theorem trapezoid_sides (t : RightTriangle) (d : ℝ) (trap : Trapezoid) 
  (h1 : perpendicular_line t d) :
  (trap.s1 = trap.s4 ∧ trap.s2 = trap.s3) ∧
  ((trap.s1 = 3 ∧ trap.s2 = 3/2) ∨ (trap.s1 = 4 ∧ trap.s2 = 4/3)) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_sides_l3826_382658


namespace NUMINAMATH_CALUDE_cannot_simplify_further_l3826_382625

theorem cannot_simplify_further (x : ℝ) : 
  Real.sqrt (x^6 + x^4 + 1) = Real.sqrt (x^6 + x^4 + 1) := by sorry

end NUMINAMATH_CALUDE_cannot_simplify_further_l3826_382625


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3826_382667

/-- An arithmetic sequence with 12 terms -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of odd-numbered terms is 10 -/
def SumOddTerms (a : ℕ → ℚ) : Prop :=
  (a 1) + (a 3) + (a 5) + (a 7) + (a 9) + (a 11) = 10

/-- The sum of even-numbered terms is 22 -/
def SumEvenTerms (a : ℕ → ℚ) : Prop :=
  (a 2) + (a 4) + (a 6) + (a 8) + (a 10) + (a 12) = 22

/-- The common difference of the arithmetic sequence is 2 -/
def CommonDifferenceIsTwo (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h1 : ArithmeticSequence a)
  (h2 : SumOddTerms a)
  (h3 : SumEvenTerms a) :
  CommonDifferenceIsTwo a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3826_382667


namespace NUMINAMATH_CALUDE_space_geometry_theorem_l3826_382695

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)

-- Define the theorem
theorem space_geometry_theorem 
  (m n : Line) (α β : Plane) 
  (hm_neq_n : m ≠ n) (hα_neq_β : α ≠ β) :
  (perpendicularLP m α ∧ perpendicularLP n β ∧ perpendicular m n → perpendicularPP α β) ∧
  (perpendicularLP m α ∧ parallelLP n β ∧ parallelPP α β → perpendicular m n) :=
sorry

end NUMINAMATH_CALUDE_space_geometry_theorem_l3826_382695


namespace NUMINAMATH_CALUDE_device_marked_price_device_marked_price_is_59_l3826_382614

theorem device_marked_price (initial_price : ℝ) (purchase_discount : ℝ) 
  (profit_percentage : ℝ) (sale_discount : ℝ) : ℝ :=
  let purchase_price := initial_price * (1 - purchase_discount)
  let selling_price := purchase_price * (1 + profit_percentage)
  selling_price / (1 - sale_discount)

theorem device_marked_price_is_59 :
  device_marked_price 50 0.15 0.25 0.10 = 59 := by
  sorry

end NUMINAMATH_CALUDE_device_marked_price_device_marked_price_is_59_l3826_382614


namespace NUMINAMATH_CALUDE_billboard_perimeter_l3826_382631

theorem billboard_perimeter (area : ℝ) (short_side : ℝ) (perimeter : ℝ) : 
  area = 104 → 
  short_side = 8 → 
  perimeter = 2 * (area / short_side + short_side) →
  perimeter = 42 := by
sorry


end NUMINAMATH_CALUDE_billboard_perimeter_l3826_382631


namespace NUMINAMATH_CALUDE_tennis_ball_ratio_l3826_382678

/-- Given the number of tennis balls for Lily, Brian, and Frodo, prove the ratio of Brian's to Frodo's tennis balls -/
theorem tennis_ball_ratio :
  ∀ (lily_balls brian_balls frodo_balls : ℕ),
    lily_balls = 3 →
    brian_balls = 22 →
    frodo_balls = lily_balls + 8 →
    (brian_balls : ℚ) / (frodo_balls : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_tennis_ball_ratio_l3826_382678


namespace NUMINAMATH_CALUDE_expression_simplification_l3826_382623

theorem expression_simplification (a b c k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c = 0) :
  (k * a^2 * b^2 + k * a^2 * c^2 + k * b^2 * c^2) / 
  ((a^2 - b*c)*(b^2 - a*c) + (a^2 - b*c)*(c^2 - a*b) + (b^2 - a*c)*(c^2 - a*b)) = k/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3826_382623


namespace NUMINAMATH_CALUDE_jane_current_age_jane_age_is_40_l3826_382628

theorem jane_current_age : ℕ → Prop :=
  fun jane_age =>
    ∃ (babysitting_start_age babysitting_end_age oldest_babysat_age : ℕ),
      babysitting_start_age = 18 ∧
      babysitting_end_age = jane_age - 10 ∧
      oldest_babysat_age = 25 ∧
      (∀ child_age : ℕ, child_age ≤ oldest_babysat_age - 10 → 2 * child_age ≤ babysitting_end_age) ∧
      jane_age = 40

theorem jane_age_is_40 : jane_current_age 40 := by
  sorry

end NUMINAMATH_CALUDE_jane_current_age_jane_age_is_40_l3826_382628


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l3826_382624

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l3826_382624


namespace NUMINAMATH_CALUDE_hyperbola_from_ellipse_l3826_382688

/-- Given an ellipse with equation x²/24 + y²/49 = 1, 
    prove that the equation of the hyperbola whose vertices are the foci of this ellipse 
    and whose foci are the vertices of this ellipse is y²/25 - x²/24 = 1 -/
theorem hyperbola_from_ellipse (x y : ℝ) :
  (x^2 / 24 + y^2 / 49 = 1) →
  ∃ (x' y' : ℝ), (y'^2 / 25 - x'^2 / 24 = 1 ∧ 
    (∀ (a b c : ℝ), (a^2 = 49 ∧ b^2 = 24 ∧ c^2 = a^2 - b^2) →
      ((x' = 0 ∧ y' = c) ∨ (x' = 0 ∧ y' = -c)) ∧
      ((x' = 0 ∧ y' = a) ∨ (x' = 0 ∧ y' = -a)))) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_from_ellipse_l3826_382688


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_products_l3826_382644

theorem sum_of_reciprocal_products (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 2022*a^2 + 1011 = 0 →
  b^3 - 2022*b^2 + 1011 = 0 →
  c^3 - 2022*c^2 + 1011 = 0 →
  1/(a*b) + 1/(b*c) + 1/(a*c) = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_products_l3826_382644


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3826_382603

-- Define the isosceles triangle
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  area : ℝ

-- Define the conditions of the problem
def triangle : IsoscelesTriangle :=
  { side1 := 6,
    side2 := 8,
    area := 12 }

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) 
  (h1 : t = triangle) : 
  (2 * t.side1 + t.side2 = 20) ∨ (2 * t.side2 + t.side1 = 20) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3826_382603


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3826_382680

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3826_382680


namespace NUMINAMATH_CALUDE_product_of_special_integers_l3826_382677

theorem product_of_special_integers (A B C D : ℕ+) 
  (sum_eq : A + B + C + D = 70)
  (def_A : A = 3 * C + 1)
  (def_B : B = 3 * C + 5)
  (def_D : D = 3 * C * C) :
  A * B * C * D = 16896 := by
  sorry

end NUMINAMATH_CALUDE_product_of_special_integers_l3826_382677


namespace NUMINAMATH_CALUDE_count_leap_years_l3826_382640

def is_leap_year (year : ℕ) : Bool :=
  if year % 100 = 0 then year % 400 = 0 else year % 4 = 0

def years : List ℕ := [1964, 1978, 1995, 1996, 2001, 2100]

theorem count_leap_years : (years.filter is_leap_year).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_leap_years_l3826_382640


namespace NUMINAMATH_CALUDE_johns_journey_cost_l3826_382615

/-- Calculates the total cost of John's journey given the specified conditions. -/
theorem johns_journey_cost : 
  let rental_cost : ℚ := 150
  let rental_discount : ℚ := 0.15
  let gas_cost_per_gallon : ℚ := 3.5
  let gas_gallons : ℚ := 8
  let driving_cost_per_mile : ℚ := 0.5
  let initial_distance : ℚ := 320
  let additional_distance : ℚ := 50
  let toll_fees : ℚ := 15
  let parking_cost_per_day : ℚ := 20
  let parking_days : ℚ := 3
  let meals_lodging_cost_per_day : ℚ := 70
  let meals_lodging_days : ℚ := 2

  let discounted_rental := rental_cost * (1 - rental_discount)
  let total_gas_cost := gas_cost_per_gallon * gas_gallons
  let total_distance := initial_distance + additional_distance
  let total_driving_cost := driving_cost_per_mile * total_distance
  let total_parking_cost := parking_cost_per_day * parking_days
  let total_meals_lodging := meals_lodging_cost_per_day * meals_lodging_days

  let total_cost := discounted_rental + total_gas_cost + total_driving_cost + 
                    toll_fees + total_parking_cost + total_meals_lodging

  total_cost = 555.5 := by sorry

end NUMINAMATH_CALUDE_johns_journey_cost_l3826_382615


namespace NUMINAMATH_CALUDE_prime_sum_product_93_178_l3826_382691

theorem prime_sum_product_93_178 : 
  ∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ p + q = 93 ∧ p * q = 178 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_93_178_l3826_382691


namespace NUMINAMATH_CALUDE_point_product_theorem_l3826_382613

theorem point_product_theorem : 
  ∀ y₁ y₂ : ℝ, 
    ((-2 - 4)^2 + (y₁ - (-1))^2 = 8^2) → 
    ((-2 - 4)^2 + (y₂ - (-1))^2 = 8^2) → 
    y₁ ≠ y₂ →
    y₁ * y₂ = -27 := by
  sorry

end NUMINAMATH_CALUDE_point_product_theorem_l3826_382613


namespace NUMINAMATH_CALUDE_puzzle_solution_l3826_382621

theorem puzzle_solution (x y : ℤ) (h1 : 3 * x + 4 * y = 150) (h2 : x = 15 ∨ y = 15) : 
  (x ≠ 15 → x = 30) ∧ (y ≠ 15 → y = 30) :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3826_382621


namespace NUMINAMATH_CALUDE_sin_450_degrees_l3826_382661

theorem sin_450_degrees : Real.sin (450 * π / 180) = 1 := by sorry

end NUMINAMATH_CALUDE_sin_450_degrees_l3826_382661


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3826_382663

theorem quadratic_always_positive (k : ℝ) : ∀ x : ℝ, x^2 - (k - 4)*x + k - 7 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3826_382663


namespace NUMINAMATH_CALUDE_first_part_length_l3826_382650

/-- Given a trip with the following conditions:
  * Total distance is 50 km
  * First part is traveled at 66 km/h
  * Second part is traveled at 33 km/h
  * Average speed of the entire trip is 44 km/h
  Prove that the length of the first part of the trip is 25 km -/
theorem first_part_length (total_distance : ℝ) (speed1 speed2 avg_speed : ℝ) 
  (h1 : total_distance = 50)
  (h2 : speed1 = 66)
  (h3 : speed2 = 33)
  (h4 : avg_speed = 44)
  (h5 : ∃ x : ℝ, x > 0 ∧ x < total_distance ∧ 
        avg_speed = total_distance / (x / speed1 + (total_distance - x) / speed2)) :
  ∃ x : ℝ, x = 25 ∧ x > 0 ∧ x < total_distance ∧ 
    avg_speed = total_distance / (x / speed1 + (total_distance - x) / speed2) := by
  sorry

end NUMINAMATH_CALUDE_first_part_length_l3826_382650


namespace NUMINAMATH_CALUDE_rectangle_width_equal_to_square_area_l3826_382609

theorem rectangle_width_equal_to_square_area (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ) :
  square_side = 8 →
  rect_length = 16 →
  square_side * square_side = rect_length * rect_width →
  rect_width = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_equal_to_square_area_l3826_382609


namespace NUMINAMATH_CALUDE_inequality_proof_l3826_382622

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  a^2 / (b + c + d) + b^2 / (c + d + a) + c^2 / (d + a + b) + d^2 / (a + b + c) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3826_382622


namespace NUMINAMATH_CALUDE_pentagon_area_l3826_382648

/-- A point on a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- A pentagon on a grid -/
structure GridPentagon where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint
  v4 : GridPoint
  v5 : GridPoint

/-- Count the number of integer points strictly inside a polygon -/
def countInteriorPoints (p : GridPentagon) : Int :=
  sorry

/-- Count the number of integer points on the boundary of a polygon -/
def countBoundaryPoints (p : GridPentagon) : Int :=
  sorry

/-- Calculate the area of a polygon using Pick's theorem -/
def polygonArea (p : GridPentagon) : Int :=
  countInteriorPoints p + (countBoundaryPoints p / 2) - 1

theorem pentagon_area :
  let p : GridPentagon := {
    v1 := { x := 0, y := 1 },
    v2 := { x := 2, y := 5 },
    v3 := { x := 6, y := 3 },
    v4 := { x := 5, y := 0 },
    v5 := { x := 1, y := 0 }
  }
  polygonArea p = 17 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l3826_382648


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_at_least_two_zeros_l3826_382643

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 9 * 10^5

/-- The number of 6-digit numbers with no zeros -/
def numbers_with_no_zeros : ℕ := 9^6

/-- The number of 6-digit numbers with exactly one zero -/
def numbers_with_one_zero : ℕ := 5 * 9^5

/-- The number of 6-digit numbers with at least two zeros -/
def numbers_with_at_least_two_zeros : ℕ := total_six_digit_numbers - (numbers_with_no_zeros + numbers_with_one_zero)

theorem six_digit_numbers_with_at_least_two_zeros :
  numbers_with_at_least_two_zeros = 73314 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_at_least_two_zeros_l3826_382643


namespace NUMINAMATH_CALUDE_opposite_number_theorem_l3826_382637

theorem opposite_number_theorem (a : ℝ) : -a = -1 → a + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_theorem_l3826_382637


namespace NUMINAMATH_CALUDE_hexagon_percentage_is_62_5_l3826_382639

/-- Represents the tiling pattern of the plane -/
structure TilingPattern where
  /-- The number of smaller squares in each large square -/
  total_squares : ℕ
  /-- The number of smaller squares used to form hexagons in each large square -/
  hexagon_squares : ℕ

/-- Calculates the percentage of the plane enclosed by hexagons -/
def hexagon_percentage (pattern : TilingPattern) : ℚ :=
  (pattern.hexagon_squares : ℚ) / (pattern.total_squares : ℚ) * 100

/-- The theorem stating that the percentage of the plane enclosed by hexagons is 62.5% -/
theorem hexagon_percentage_is_62_5 (pattern : TilingPattern) 
  (h1 : pattern.total_squares = 16)
  (h2 : pattern.hexagon_squares = 10) : 
  hexagon_percentage pattern = 62.5 := by
  sorry

#eval hexagon_percentage { total_squares := 16, hexagon_squares := 10 }

end NUMINAMATH_CALUDE_hexagon_percentage_is_62_5_l3826_382639


namespace NUMINAMATH_CALUDE_greater_number_problem_l3826_382666

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : 
  max x y = 35 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l3826_382666


namespace NUMINAMATH_CALUDE_quadratic_root_relationship_l3826_382683

/-- Given two quadratic equations and their relationship, prove the ratio of their coefficients -/
theorem quadratic_root_relationship (m n p : ℝ) : 
  m ≠ 0 → n ≠ 0 → p ≠ 0 →
  (∃ r₁ r₂ : ℝ, (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧
               (3*r₁ + 3*r₂ = -m ∧ 9*r₁*r₂ = n)) →
  n/p = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relationship_l3826_382683


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3826_382673

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3826_382673


namespace NUMINAMATH_CALUDE_cattle_problem_l3826_382690

/-- Represents the problem of determining the number of cattle that died --/
theorem cattle_problem (initial_cattle : ℕ) (initial_price : ℕ) (price_reduction : ℕ) (total_loss : ℕ) : 
  initial_cattle = 340 →
  initial_price = 204000 →
  price_reduction = 150 →
  total_loss = 25200 →
  ∃ (dead_cattle : ℕ), 
    dead_cattle = 57 ∧ 
    (initial_cattle - dead_cattle) * (initial_price / initial_cattle - price_reduction) = initial_price - total_loss := by
  sorry


end NUMINAMATH_CALUDE_cattle_problem_l3826_382690


namespace NUMINAMATH_CALUDE_total_study_time_is_three_l3826_382651

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time Sam spends studying Science in minutes -/
def science_time : ℕ := 60

/-- The time Sam spends studying Math in minutes -/
def math_time : ℕ := 80

/-- The time Sam spends studying Literature in minutes -/
def literature_time : ℕ := 40

/-- The total time Sam spends studying in hours -/
def total_study_time : ℚ :=
  (science_time + math_time + literature_time) / minutes_per_hour

theorem total_study_time_is_three : total_study_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_total_study_time_is_three_l3826_382651


namespace NUMINAMATH_CALUDE_mixed_fraction_product_l3826_382610

theorem mixed_fraction_product (X Y : ℤ) : 
  (5 + 1 / X : ℚ) * (Y + 1 / 2 : ℚ) = 43 →
  5 < 5 + 1 / X →
  5 + 1 / X ≤ 11 / 2 →
  X = 17 ∧ Y = 8 := by
sorry

end NUMINAMATH_CALUDE_mixed_fraction_product_l3826_382610


namespace NUMINAMATH_CALUDE_f_of_2_equals_negative_2_l3826_382607

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x

-- State the theorem
theorem f_of_2_equals_negative_2 : f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_negative_2_l3826_382607


namespace NUMINAMATH_CALUDE_two_apples_per_slice_l3826_382618

/-- The number of apples in each slice of pie -/
def apples_per_slice (total_apples : ℕ) (num_pies : ℕ) (slices_per_pie : ℕ) : ℚ :=
  total_apples / (num_pies * slices_per_pie)

/-- Theorem: Given the conditions, prove that there are 2 apples in each slice of pie -/
theorem two_apples_per_slice :
  let total_apples : ℕ := 4 * 12  -- 4 dozen apples
  let num_pies : ℕ := 4
  let slices_per_pie : ℕ := 6
  apples_per_slice total_apples num_pies slices_per_pie = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_apples_per_slice_l3826_382618


namespace NUMINAMATH_CALUDE_birthday_number_proof_l3826_382692

theorem birthday_number_proof : ∃! T : ℕ+,
  (∃ x y : ℕ, 4 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
    T ^ 2 = 40000 + x * 1000 + y * 100 + 29) ∧
  T = 223 := by
  sorry

end NUMINAMATH_CALUDE_birthday_number_proof_l3826_382692


namespace NUMINAMATH_CALUDE_nori_crayon_problem_l3826_382602

/-- Given the initial number of crayon boxes, crayons per box, crayons given to Mae, and crayons left,
    calculate the difference between crayons given to Lea and Mae. -/
def crayon_difference (boxes : ℕ) (crayons_per_box : ℕ) (given_to_mae : ℕ) (crayons_left : ℕ) : ℕ :=
  boxes * crayons_per_box - given_to_mae - crayons_left - given_to_mae

theorem nori_crayon_problem :
  crayon_difference 4 8 5 15 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nori_crayon_problem_l3826_382602


namespace NUMINAMATH_CALUDE_puzzle_piece_ratio_l3826_382659

theorem puzzle_piece_ratio (total pieces : ℕ) (border : ℕ) (trevor : ℕ) (missing : ℕ) :
  total = 500 →
  border = 75 →
  trevor = 105 →
  missing = 5 →
  ∃ (joe : ℕ), joe = total - border - trevor - missing ∧ joe = 3 * trevor :=
by sorry

end NUMINAMATH_CALUDE_puzzle_piece_ratio_l3826_382659


namespace NUMINAMATH_CALUDE_bob_walking_distance_l3826_382627

/-- Proves that Bob walked 4 miles before meeting Yolanda given the problem conditions -/
theorem bob_walking_distance (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) 
  (head_start : ℝ) (h1 : total_distance = 10) 
  (h2 : yolanda_rate = 3) (h3 : bob_rate = 4) (h4 : head_start = 1) : 
  ∃ t : ℝ, t > head_start ∧ yolanda_rate * t + bob_rate * (t - head_start) = total_distance ∧ 
  bob_rate * (t - head_start) = 4 :=
by sorry

end NUMINAMATH_CALUDE_bob_walking_distance_l3826_382627


namespace NUMINAMATH_CALUDE_average_annual_growth_rate_l3826_382687

/-- Given growth rates a and b for two consecutive years, 
    the average annual growth rate over these two years 
    is equal to √((a+1)(b+1)) - 1 -/
theorem average_annual_growth_rate 
  (a b : ℝ) : 
  ∃ x : ℝ, x = Real.sqrt ((a + 1) * (b + 1)) - 1 ∧ 
  (1 + x)^2 = (1 + a) * (1 + b) :=
by sorry

end NUMINAMATH_CALUDE_average_annual_growth_rate_l3826_382687


namespace NUMINAMATH_CALUDE_mark_kangaroos_l3826_382619

theorem mark_kangaroos (num_kangaroos num_goats : ℕ) : 
  num_goats = 3 * num_kangaroos →
  2 * num_kangaroos + 4 * num_goats = 322 →
  num_kangaroos = 23 := by
sorry

end NUMINAMATH_CALUDE_mark_kangaroos_l3826_382619


namespace NUMINAMATH_CALUDE_min_value_expression_l3826_382698

theorem min_value_expression (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq_one : a + b + c = 1) :
  a + (a * b) ^ (1/3 : ℝ) + (a * b * c) ^ (1/4 : ℝ) ≥ 1/3 + 1/(3 * 3^(1/3 : ℝ)) + 1/(3 * 3^(1/4 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3826_382698


namespace NUMINAMATH_CALUDE_equal_earnings_l3826_382601

theorem equal_earnings (t : ℝ) : 
  (t - 6) * (2 * t - 5) = (2 * t - 8) * (t - 5) → t = 10 := by
  sorry

end NUMINAMATH_CALUDE_equal_earnings_l3826_382601


namespace NUMINAMATH_CALUDE_annual_rent_per_square_foot_l3826_382693

/-- Calculates the annual rent per square foot for a shop -/
theorem annual_rent_per_square_foot
  (length : ℝ) (width : ℝ) (monthly_rent : ℝ)
  (h1 : length = 18)
  (h2 : width = 22)
  (h3 : monthly_rent = 2244) :
  (monthly_rent * 12) / (length * width) = 68 := by
  sorry

end NUMINAMATH_CALUDE_annual_rent_per_square_foot_l3826_382693


namespace NUMINAMATH_CALUDE_inequality_implication_l3826_382634

theorem inequality_implication (x y : ℝ) (h : x > y) : 2*x - 1 > 2*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3826_382634


namespace NUMINAMATH_CALUDE_share_ratio_l3826_382632

theorem share_ratio (total amount : ℕ) (a_share b_share c_share : ℕ) 
  (h1 : total = 595)
  (h2 : a_share = 420)
  (h3 : b_share = 105)
  (h4 : c_share = 70)
  (h5 : total = a_share + b_share + c_share)
  (h6 : b_share = c_share / 4) :
  a_share / b_share = 4 := by
sorry

end NUMINAMATH_CALUDE_share_ratio_l3826_382632


namespace NUMINAMATH_CALUDE_units_digit_of_power_of_three_l3826_382635

theorem units_digit_of_power_of_three (n : ℕ) : (3^(4*n + 2) % 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_of_three_l3826_382635


namespace NUMINAMATH_CALUDE_incorrect_value_calculation_l3826_382636

/-- Given a set of values with an incorrect mean due to a copying error,
    calculate the incorrect value that was used. -/
theorem incorrect_value_calculation
  (n : ℕ)
  (initial_mean correct_mean : ℚ)
  (correct_value : ℚ)
  (h_n : n = 30)
  (h_initial_mean : initial_mean = 250)
  (h_correct_mean : correct_mean = 251)
  (h_correct_value : correct_value = 165) :
  ∃ (incorrect_value : ℚ),
    incorrect_value = 195 ∧
    n * correct_mean = n * initial_mean - correct_value + incorrect_value :=
by sorry

end NUMINAMATH_CALUDE_incorrect_value_calculation_l3826_382636


namespace NUMINAMATH_CALUDE_circle_to_bar_graph_correspondence_l3826_382679

/-- Represents the proportions of a circle graph -/
structure CircleGraph where
  white : ℝ
  black : ℝ
  gray : ℝ
  sum_to_one : white + black + gray = 1
  white_twice_others : white = 2 * black ∧ white = 2 * gray
  black_gray_equal : black = gray

/-- Represents the heights of bars in a bar graph -/
structure BarGraph where
  white : ℝ
  black : ℝ
  gray : ℝ

/-- Theorem stating that a bar graph correctly represents a circle graph -/
theorem circle_to_bar_graph_correspondence (cg : CircleGraph) (bg : BarGraph) :
  (bg.white = 2 * bg.black ∧ bg.white = 2 * bg.gray) ∧ bg.black = bg.gray :=
by sorry

end NUMINAMATH_CALUDE_circle_to_bar_graph_correspondence_l3826_382679


namespace NUMINAMATH_CALUDE_chicken_duck_count_l3826_382600

theorem chicken_duck_count : ∃ (chickens ducks : ℕ),
  chickens + ducks = 239 ∧
  ducks = 3 * chickens + 15 ∧
  chickens = 56 ∧
  ducks = 183 := by
  sorry

end NUMINAMATH_CALUDE_chicken_duck_count_l3826_382600


namespace NUMINAMATH_CALUDE_final_tree_count_l3826_382664

/-- 
Given:
- T: The initial number of trees
- P: The percentage of trees cut (as a whole number, e.g., 20 for 20%)
- R: The number of new trees planted for each tree cut

Prove that the final number of trees F is equal to T - (P/100 * T) + (P/100 * T * R)
-/
theorem final_tree_count (T P R : ℕ) (h1 : P ≤ 100) : 
  ∃ F : ℕ, F = T - (P * T / 100) + (P * T * R / 100) :=
sorry

end NUMINAMATH_CALUDE_final_tree_count_l3826_382664


namespace NUMINAMATH_CALUDE_sequence_divergence_criterion_l3826_382608

/-- Given a sequence xₙ and a limit point a, prove that for every ε > 0,
    there exists a number k such that for all n > k, |xₙ - a| ≥ ε -/
theorem sequence_divergence_criterion 
  (x : ℕ → ℝ) (a : ℝ) : 
  ∀ ε > 0, ∃ k : ℕ, ∀ n > k, |x n - a| ≥ ε := by
  sorry

end NUMINAMATH_CALUDE_sequence_divergence_criterion_l3826_382608


namespace NUMINAMATH_CALUDE_dianas_age_dianas_age_is_eight_l3826_382689

/-- Diana's age today, given that she is twice as old as Grace and Grace turned 3 a year ago -/
theorem dianas_age : ℕ :=
  let graces_age_last_year : ℕ := 3
  let graces_age_today : ℕ := graces_age_last_year + 1
  let dianas_age : ℕ := 2 * graces_age_today
  8

/-- Proof that Diana's age is 8 years old today -/
theorem dianas_age_is_eight : dianas_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_dianas_age_dianas_age_is_eight_l3826_382689


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_equals_sqrt_six_l3826_382670

theorem sqrt_two_times_sqrt_three_equals_sqrt_six :
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_equals_sqrt_six_l3826_382670


namespace NUMINAMATH_CALUDE_cubic_unique_solution_iff_l3826_382684

/-- The cubic equation in x with parameter a -/
def cubic_equation (a x : ℝ) : ℝ := x^3 - a*x^2 - 3*a*x + a^2 - 1

/-- The property that the cubic equation has exactly one real solution -/
def has_unique_real_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation a x = 0

theorem cubic_unique_solution_iff (a : ℝ) :
  has_unique_real_solution a ↔ a < -5/4 :=
sorry

end NUMINAMATH_CALUDE_cubic_unique_solution_iff_l3826_382684


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_height_l3826_382655

/-- Represents a right hexagonal pyramid with three parallel cross sections. -/
structure HexagonalPyramid where
  /-- Height of the smallest cross section from the apex -/
  x : ℝ
  /-- Area of the smallest cross section -/
  area₁ : ℝ
  /-- Area of the middle cross section -/
  area₂ : ℝ
  /-- Area of the largest cross section -/
  area₃ : ℝ
  /-- The areas are in the correct ratio -/
  area_ratio₁ : area₁ / area₂ = 9 / 20
  /-- The areas are in the correct ratio -/
  area_ratio₂ : area₂ / area₃ = 5 / 9
  /-- The heights are in arithmetic progression -/
  height_progression : x + 20 - (x + 10) = (x + 10) - x

/-- The height of the smallest cross section from the apex in a right hexagonal pyramid
    with specific cross-sectional areas at specific heights. -/
theorem hexagonal_pyramid_height (p : HexagonalPyramid) : p.x = 100 / (10 - 3 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_height_l3826_382655


namespace NUMINAMATH_CALUDE_sum_of_y_coordinates_on_y_axis_l3826_382629

-- Define the circle
def circle_center : ℝ × ℝ := (-3, 5)
def circle_radius : ℝ := 15

-- Define a function to represent the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem sum_of_y_coordinates_on_y_axis :
  ∃ y₁ y₂ : ℝ, 
    circle_equation 0 y₁ ∧ 
    circle_equation 0 y₂ ∧ 
    y₁ ≠ y₂ ∧ 
    y₁ + y₂ = 10 :=
sorry

end NUMINAMATH_CALUDE_sum_of_y_coordinates_on_y_axis_l3826_382629


namespace NUMINAMATH_CALUDE_mike_pears_count_l3826_382694

/-- The number of pears picked by Jason -/
def jason_pears : ℕ := 46

/-- The number of pears picked by Keith -/
def keith_pears : ℕ := 47

/-- The total number of pears picked -/
def total_pears : ℕ := 105

/-- The number of pears picked by Mike -/
def mike_pears : ℕ := total_pears - (jason_pears + keith_pears)

theorem mike_pears_count : mike_pears = 12 := by
  sorry

end NUMINAMATH_CALUDE_mike_pears_count_l3826_382694


namespace NUMINAMATH_CALUDE_number_equation_solution_l3826_382660

theorem number_equation_solution :
  ∃ x : ℚ, (35 + 3 * x = 51) ∧ (x = 16 / 3) :=
by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3826_382660


namespace NUMINAMATH_CALUDE_performing_arts_school_l3826_382656

theorem performing_arts_school (total : ℕ) (cant_sing cant_dance cant_act : ℕ) :
  total = 150 ∧
  cant_sing = 80 ∧
  cant_dance = 110 ∧
  cant_act = 60 →
  ∃ (all_talents : ℕ),
    all_talents = total - ((total - cant_sing) + (total - cant_dance) + (total - cant_act) - total) ∧
    all_talents = 50 := by
  sorry

end NUMINAMATH_CALUDE_performing_arts_school_l3826_382656


namespace NUMINAMATH_CALUDE_remove_five_for_target_average_l3826_382652

def original_list : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def target_average : Rat := 41/5  -- 8.2 as a rational number

theorem remove_five_for_target_average :
  let remaining_list := original_list.filter (· ≠ 5)
  (remaining_list.sum : Rat) / remaining_list.length = target_average := by
  sorry

end NUMINAMATH_CALUDE_remove_five_for_target_average_l3826_382652


namespace NUMINAMATH_CALUDE_bowling_ball_difference_l3826_382611

theorem bowling_ball_difference :
  ∀ (red green : ℕ),
  red = 30 →
  green > red →
  red + green = 66 →
  green - red = 6 :=
by sorry

end NUMINAMATH_CALUDE_bowling_ball_difference_l3826_382611


namespace NUMINAMATH_CALUDE_projection_result_l3826_382686

def a : ℝ × ℝ := (-4, 2)
def b : ℝ × ℝ := (3, 4)

theorem projection_result (v : ℝ × ℝ) (p : ℝ × ℝ) 
  (h1 : v ≠ (0, 0)) 
  (h2 : p = (v.1 * (a.1 * v.1 + a.2 * v.2) / (v.1^2 + v.2^2), 
             v.2 * (a.1 * v.1 + a.2 * v.2) / (v.1^2 + v.2^2)))
  (h3 : p = (v.1 * (b.1 * v.1 + b.2 * v.2) / (v.1^2 + v.2^2), 
             v.2 * (b.1 * v.1 + b.2 * v.2) / (v.1^2 + v.2^2))) :
  p = (-44/53, 154/53) := by
sorry

end NUMINAMATH_CALUDE_projection_result_l3826_382686


namespace NUMINAMATH_CALUDE_min_candies_removed_correct_l3826_382649

/-- Represents the number of candies of each flavor -/
structure CandyCounts where
  chocolate : Nat
  mint : Nat
  butterscotch : Nat

/-- The initial candy counts in the bag -/
def initialCandies : CandyCounts :=
  { chocolate := 4, mint := 6, butterscotch := 10 }

/-- The total number of candies in the bag -/
def totalCandies : Nat := 20

/-- The minimum number of candies that must be removed to ensure
    at least two of each flavor have been eaten -/
def minCandiesRemoved : Nat := 18

theorem min_candies_removed_correct :
  minCandiesRemoved = totalCandies - (initialCandies.chocolate - 2) - (initialCandies.mint - 2) - (initialCandies.butterscotch - 2) :=
by sorry

end NUMINAMATH_CALUDE_min_candies_removed_correct_l3826_382649


namespace NUMINAMATH_CALUDE_phone_number_A_value_l3826_382665

def phone_number (A B C D E F G H I J : ℕ) : Prop :=
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  D % 2 = 0 ∧ E % 2 = 0 ∧ F % 2 = 0 ∧
  D = E + 2 ∧ E = F + 2 ∧
  G % 2 = 1 ∧ H % 2 = 1 ∧ I % 2 = 1 ∧ J % 2 = 1 ∧
  G = H + 2 ∧ H = I + 2 ∧ I = J + 4 ∧
  J = 1 ∧
  A + B + C = 11 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J

theorem phone_number_A_value :
  ∀ A B C D E F G H I J : ℕ,
  phone_number A B C D E F G H I J →
  A = 8 := by
sorry

end NUMINAMATH_CALUDE_phone_number_A_value_l3826_382665


namespace NUMINAMATH_CALUDE_only_zhong_symmetric_l3826_382675

-- Define a type for Chinese characters
inductive ChineseCharacter : Type
  | ai   : ChineseCharacter  -- 爱
  | wo   : ChineseCharacter  -- 我
  | zhong : ChineseCharacter  -- 中
  | hua  : ChineseCharacter  -- 华

-- Define a property for vertical symmetry
def hasVerticalSymmetry (c : ChineseCharacter) : Prop :=
  match c with
  | ChineseCharacter.zhong => True
  | _ => False

-- Theorem stating that only 中 (zhong) has vertical symmetry
theorem only_zhong_symmetric :
  ∀ (c : ChineseCharacter),
    hasVerticalSymmetry c ↔ c = ChineseCharacter.zhong :=
by
  sorry


end NUMINAMATH_CALUDE_only_zhong_symmetric_l3826_382675


namespace NUMINAMATH_CALUDE_eventually_point_difference_exceeds_50_l3826_382630

/-- Represents a player in the tournament -/
structure Player where
  id : Nat
  points : Int

/-- Represents the state of the tournament on a given day -/
structure TournamentDay where
  players : Vector Player 200
  day : Nat

/-- Function to sort players by their points -/
def sortPlayers (players : Vector Player 200) : Vector Player 200 := sorry

/-- Function to play matches for a day and update points -/
def playMatches (t : TournamentDay) : TournamentDay := sorry

/-- Predicate to check if the point difference exceeds 50 -/
def pointDifferenceExceeds50 (t : TournamentDay) : Prop :=
  ∃ i j, i < 200 ∧ j < 200 ∧ (t.players.get i).points - (t.players.get j).points > 50

/-- The main theorem to be proved -/
theorem eventually_point_difference_exceeds_50 :
  ∃ n : Nat, ∃ t : TournamentDay, t.day = n ∧ pointDifferenceExceeds50 t :=
sorry

end NUMINAMATH_CALUDE_eventually_point_difference_exceeds_50_l3826_382630


namespace NUMINAMATH_CALUDE_brianna_reread_books_l3826_382674

/-- The number of old books Brianna needs to reread in a year --/
def old_books_to_reread (books_per_month : ℕ) (months_in_year : ℕ) (gifted_books : ℕ) (bought_books : ℕ) (borrowed_books_difference : ℕ) : ℕ :=
  let total_books_needed := books_per_month * months_in_year
  let new_books := gifted_books + bought_books + (bought_books - borrowed_books_difference)
  total_books_needed - new_books

/-- Theorem stating the number of old books Brianna needs to reread --/
theorem brianna_reread_books : 
  old_books_to_reread 2 12 6 8 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_brianna_reread_books_l3826_382674


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3826_382676

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

def monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n m, n < m → a n < a m

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (¬ (q > 1 → monotonically_increasing a) ∧ ¬ (monotonically_increasing a → q > 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3826_382676


namespace NUMINAMATH_CALUDE_exists_unique_q_l3826_382657

/-- Polynomial function g(x) -/
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p*x^4 + q*x^3 + r*x^2 + s*x + t

/-- Theorem stating the existence and uniqueness of q -/
theorem exists_unique_q :
  ∃! q : ℝ, ∃ p r s t : ℝ,
    g p q r s t 0 = 3 ∧
    g p q r s t (-2) = 0 ∧
    g p q r s t 1 = 0 ∧
    g p q r s t (-1) = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_exists_unique_q_l3826_382657


namespace NUMINAMATH_CALUDE_courtyard_paving_l3826_382638

theorem courtyard_paving (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (brick_width : ℝ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  ⌈(courtyard_length * courtyard_width) / (brick_length * brick_width)⌉ = 20000 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l3826_382638


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l3826_382641

theorem rectangle_area_perimeter_sum (w : ℕ) (h : w > 0) : 
  let l := 2 * w
  let A := l * w
  let P := 2 * (l + w)
  A + P ≠ 110 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l3826_382641


namespace NUMINAMATH_CALUDE_tangerine_consumption_change_l3826_382671

/-- Demand function before embargo -/
def initial_demand (p : ℝ) : ℝ := 50 - p

/-- Demand function after embargo -/
def new_demand (p : ℝ) : ℝ := 2.5 * (50 - p)

/-- Marginal cost (constant) -/
def marginal_cost : ℝ := 5

/-- Initial equilibrium quantity under perfect competition -/
def initial_equilibrium_quantity : ℝ := initial_demand marginal_cost

/-- New equilibrium quantity under monopoly -/
noncomputable def new_equilibrium_quantity : ℝ := 56.25

theorem tangerine_consumption_change :
  new_equilibrium_quantity / initial_equilibrium_quantity = 1.25 := by sorry

end NUMINAMATH_CALUDE_tangerine_consumption_change_l3826_382671


namespace NUMINAMATH_CALUDE_carbonated_water_in_solution2_l3826_382620

/-- Represents a solution mixture of lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  sum_to_one : lemonade + carbonated_water = 1

/-- Represents the final mixture of two solutions -/
structure Mixture where
  solution1 : Solution
  solution2 : Solution
  proportion1 : ℝ
  proportion2 : ℝ
  sum_to_one : proportion1 + proportion2 = 1
  carbonated_water_percent : ℝ

/-- The main theorem to prove -/
theorem carbonated_water_in_solution2 
  (mix : Mixture)
  (h1 : mix.solution1.lemonade = 0.2)
  (h2 : mix.solution2.lemonade = 0.45)
  (h3 : mix.carbonated_water_percent = 0.72)
  (h4 : mix.proportion1 = 0.6799999999999997) :
  mix.solution2.carbonated_water = 0.55 := by
  sorry

#eval 1 - 0.45 -- Expected output: 0.55

end NUMINAMATH_CALUDE_carbonated_water_in_solution2_l3826_382620


namespace NUMINAMATH_CALUDE_jack_piggy_bank_total_l3826_382605

/-- Calculates the final amount in Jack's piggy bank after a given number of weeks -/
def piggy_bank_total (initial_amount : ℝ) (weekly_allowance : ℝ) (savings_rate : ℝ) (weeks : ℕ) : ℝ :=
  initial_amount + (weekly_allowance * savings_rate * weeks)

/-- Proves that Jack will have $83 in his piggy bank after 8 weeks -/
theorem jack_piggy_bank_total :
  piggy_bank_total 43 10 0.5 8 = 83 := by
  sorry

end NUMINAMATH_CALUDE_jack_piggy_bank_total_l3826_382605


namespace NUMINAMATH_CALUDE_prove_scotts_golf_score_drop_l3826_382697

def scotts_golf_problem (first_four_average : ℝ) (fifth_round_score : ℝ) : Prop :=
  let first_four_total := first_four_average * 4
  let five_round_total := first_four_total + fifth_round_score
  let new_average := five_round_total / 5
  first_four_average - new_average = 2

theorem prove_scotts_golf_score_drop :
  scotts_golf_problem 78 68 :=
sorry

end NUMINAMATH_CALUDE_prove_scotts_golf_score_drop_l3826_382697


namespace NUMINAMATH_CALUDE_spinner_sections_l3826_382616

theorem spinner_sections (n : ℕ) (n_pos : n > 0) : 
  (1 - 1 / n : ℚ) ^ 2 = 559 / 1000 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_sections_l3826_382616


namespace NUMINAMATH_CALUDE_find_divisor_l3826_382626

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 997) (h2 : quotient = 43) (h3 : remainder = 8) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 23 :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l3826_382626


namespace NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_l3826_382633

/-- Represents a discount as a fraction between 0 and 1 -/
def Discount := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- Apply a discount to a price -/
def applyDiscount (price : ℝ) (discount : Discount) : ℝ :=
  price * (1 - discount.val)

/-- Apply two successive discounts -/
def applySuccessiveDiscounts (price : ℝ) (d1 d2 : Discount) : ℝ :=
  applyDiscount (applyDiscount price d1) d2

theorem successive_discounts_equivalent_to_single (price : ℝ) :
  let d1 : Discount := ⟨0.1, by norm_num⟩
  let d2 : Discount := ⟨0.2, by norm_num⟩
  let singleDiscount : Discount := ⟨0.28, by norm_num⟩
  applySuccessiveDiscounts price d1 d2 = applyDiscount price singleDiscount := by
  sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_l3826_382633


namespace NUMINAMATH_CALUDE_inverse_g_90_l3826_382612

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_g_90 : g 3 = 90 := by sorry

end NUMINAMATH_CALUDE_inverse_g_90_l3826_382612


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l3826_382668

theorem mod_equivalence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l3826_382668


namespace NUMINAMATH_CALUDE_divisibility_implies_value_l3826_382646

theorem divisibility_implies_value (p q r : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, x^4 + 6*x^3 + 8*p*x^2 + 6*q*x + r = (x^3 + 4*x^2 + 16*x + 4) * k) →
  (p + q) * r = 56 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_value_l3826_382646
