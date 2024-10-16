import Mathlib

namespace NUMINAMATH_CALUDE_abs_inequality_range_l1347_134775

theorem abs_inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 3| ≥ a^2 + a) ↔ -2 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_range_l1347_134775


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1347_134794

theorem quadratic_inequality_solution (x : ℝ) :
  -3 * x^2 + 5 * x + 4 < 0 ↔ x < 3/4 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1347_134794


namespace NUMINAMATH_CALUDE_sets_are_equal_l1347_134792

-- Define the sets A and B
def A : Set ℝ := {1, Real.sqrt 3, Real.pi}
def B : Set ℝ := {Real.pi, 1, |-(Real.sqrt 3)|}

-- State the theorem
theorem sets_are_equal : A = B := by sorry

end NUMINAMATH_CALUDE_sets_are_equal_l1347_134792


namespace NUMINAMATH_CALUDE_evaluate_expression_l1347_134787

theorem evaluate_expression : 2 * ((3^4)^3 - (4^3)^4) = -32471550 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1347_134787


namespace NUMINAMATH_CALUDE_divisibility_by_101_l1347_134738

theorem divisibility_by_101 (a b : ℕ) : 
  a < 10 → b < 10 → 
  (12 * 10^10 + a * 10^9 + b * 10^8 + 9876543) % 101 = 0 → 
  10 * a + b = 58 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_101_l1347_134738


namespace NUMINAMATH_CALUDE_square_and_hexagon_symmetric_l1347_134788

-- Define the set of polygons
inductive Polygon
| EquilateralTriangle
| Square
| RegularPentagon
| RegularHexagon

-- Define properties
def isAxiSymmetric : Polygon → Prop
| Polygon.EquilateralTriangle => True
| Polygon.Square => True
| Polygon.RegularPentagon => True
| Polygon.RegularHexagon => True

def isCentrallySymmetric : Polygon → Prop
| Polygon.EquilateralTriangle => False
| Polygon.Square => True
| Polygon.RegularPentagon => False
| Polygon.RegularHexagon => True

-- Theorem statement
theorem square_and_hexagon_symmetric :
  ∀ p : Polygon, (isAxiSymmetric p ∧ isCentrallySymmetric p) ↔ (p = Polygon.Square ∨ p = Polygon.RegularHexagon) :=
by sorry

end NUMINAMATH_CALUDE_square_and_hexagon_symmetric_l1347_134788


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l1347_134762

theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 10) (h2 : DF = 5) (h3 : EF = 9) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let diameter := 2 * area / s
  diameter = Real.sqrt 14 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l1347_134762


namespace NUMINAMATH_CALUDE_complex_equation_l1347_134769

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Given that m/(1+i) = 1 - ni, where m and n are real numbers,
    prove that m + ni = 2 + i -/
theorem complex_equation (m n : ℝ) (h : (m : ℂ) / (1 + i) = 1 - n * i) :
  (m : ℂ) + n * i = 2 + i := by sorry

end NUMINAMATH_CALUDE_complex_equation_l1347_134769


namespace NUMINAMATH_CALUDE_taxi_distance_calculation_l1347_134745

/-- Calculates the total distance of a taxi ride given the fare structure and total charge --/
theorem taxi_distance_calculation (initial_charge : ℚ) (initial_distance : ℚ) 
  (additional_charge : ℚ) (additional_distance : ℚ) (total_charge : ℚ) :
  initial_charge = 2.5 →
  initial_distance = 1/5 →
  additional_charge = 0.4 →
  additional_distance = 1/5 →
  total_charge = 18.1 →
  ∃ (total_distance : ℚ), total_distance = 8 ∧
    total_charge = initial_charge + 
      (total_distance - initial_distance) / additional_distance * additional_charge :=
by
  sorry


end NUMINAMATH_CALUDE_taxi_distance_calculation_l1347_134745


namespace NUMINAMATH_CALUDE_rest_area_location_l1347_134772

/-- Represents a highway with exits and a rest area -/
structure Highway where
  fifth_exit : ℝ
  seventh_exit : ℝ
  rest_area : ℝ

/-- The rest area is located halfway between the fifth and seventh exits -/
def is_halfway (h : Highway) : Prop :=
  h.rest_area = (h.fifth_exit + h.seventh_exit) / 2

/-- Theorem: Given the conditions, prove that the rest area is at milepost 65 -/
theorem rest_area_location (h : Highway) 
    (h_fifth : h.fifth_exit = 35)
    (h_seventh : h.seventh_exit = 95)
    (h_halfway : is_halfway h) : 
    h.rest_area = 65 := by
  sorry

#check rest_area_location

end NUMINAMATH_CALUDE_rest_area_location_l1347_134772


namespace NUMINAMATH_CALUDE_cut_polygon_perimeter_decrease_l1347_134702

/-- Represents a polygon -/
structure Polygon where
  perimeter : ℝ
  perim_pos : perimeter > 0

/-- Represents the result of cutting a polygon with a straight line -/
structure CutPolygon where
  original : Polygon
  part1 : Polygon
  part2 : Polygon

/-- Theorem: The perimeter of each part resulting from cutting a polygon
    with a straight line is less than the perimeter of the original polygon -/
theorem cut_polygon_perimeter_decrease (cp : CutPolygon) :
  cp.part1.perimeter < cp.original.perimeter ∧
  cp.part2.perimeter < cp.original.perimeter := by
  sorry

end NUMINAMATH_CALUDE_cut_polygon_perimeter_decrease_l1347_134702


namespace NUMINAMATH_CALUDE_chubby_checkerboard_black_squares_l1347_134784

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard where
  rows : Nat
  cols : Nat

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard) : Nat :=
  ((board.cols + 1) / 2) * board.rows

/-- Theorem: A 31x29 checkerboard has 465 black squares -/
theorem chubby_checkerboard_black_squares :
  let board : Checkerboard := ⟨31, 29⟩
  count_black_squares board = 465 := by
  sorry

#eval count_black_squares ⟨31, 29⟩

end NUMINAMATH_CALUDE_chubby_checkerboard_black_squares_l1347_134784


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l1347_134756

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 < x ≤ 8}
theorem union_A_B : A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8} := by sorry

-- Theorem 2: (∁ᵤA) ∩ B = {x | 1 < x < 2}
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 3: A ∩ C ≠ ∅ if and only if a < 8
theorem intersection_A_C_nonempty (a : ℝ) : A ∩ C a ≠ ∅ ↔ a < 8 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l1347_134756


namespace NUMINAMATH_CALUDE_karen_has_128_crayons_l1347_134717

/-- The number of crayons in Judah's box -/
def judah_crayons : ℕ := 8

/-- The number of crayons in Gilbert's box -/
def gilbert_crayons : ℕ := 4 * judah_crayons

/-- The number of crayons in Beatrice's box -/
def beatrice_crayons : ℕ := 2 * gilbert_crayons

/-- The number of crayons in Karen's box -/
def karen_crayons : ℕ := 2 * beatrice_crayons

/-- Theorem stating that Karen's box contains 128 crayons -/
theorem karen_has_128_crayons : karen_crayons = 128 := by
  sorry

end NUMINAMATH_CALUDE_karen_has_128_crayons_l1347_134717


namespace NUMINAMATH_CALUDE_complex_equality_l1347_134715

theorem complex_equality (z₁ z₂ : ℂ) (h : Complex.abs (z₁ + 2 * z₂) = Complex.abs (2 * z₁ + z₂)) :
  ∀ a : ℝ, Complex.abs (z₁ + a * z₂) = Complex.abs (a * z₁ + z₂) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l1347_134715


namespace NUMINAMATH_CALUDE_f_has_no_boundary_point_l1347_134729

-- Define the concept of a boundary point
def has_boundary_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ ≠ 0 ∧
    (∃ x₁ x₂ : ℝ, x₁ < x₀ ∧ x₀ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0)

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem stating that f does not have a boundary point
theorem f_has_no_boundary_point : ¬ has_boundary_point f := by
  sorry


end NUMINAMATH_CALUDE_f_has_no_boundary_point_l1347_134729


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1347_134752

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ a ∈ Set.Ioc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1347_134752


namespace NUMINAMATH_CALUDE_cricket_average_l1347_134742

/-- Calculates the average score for the last 4 matches of a cricket series -/
theorem cricket_average (total_matches : ℕ) (first_matches : ℕ) (total_average : ℚ) (first_average : ℚ) :
  total_matches = 10 →
  first_matches = 6 →
  total_average = 389/10 →
  first_average = 42 →
  (total_matches * total_average - first_matches * first_average) / (total_matches - first_matches) = 137/4 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l1347_134742


namespace NUMINAMATH_CALUDE_a_upper_bound_l1347_134734

theorem a_upper_bound (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x * y = 2 → 
    2 - x ≥ a / (4 - y)) → 
  a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_a_upper_bound_l1347_134734


namespace NUMINAMATH_CALUDE_system_solution_unique_l1347_134791

theorem system_solution_unique :
  ∃! (x y : ℚ), x = 2 * y ∧ 2 * x - y = 5 :=
by
  -- The unique solution is x = 10/3 and y = 5/3
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1347_134791


namespace NUMINAMATH_CALUDE_jacks_total_money_l1347_134723

/-- Calculates the total amount of money in dollars given an amount in dollars and euros, with a fixed exchange rate. -/
def total_money_in_dollars (dollars : ℕ) (euros : ℕ) (exchange_rate : ℕ) : ℕ :=
  dollars + euros * exchange_rate

/-- Theorem stating that Jack's total money in dollars is 117 given the problem conditions. -/
theorem jacks_total_money :
  total_money_in_dollars 45 36 2 = 117 := by
  sorry

end NUMINAMATH_CALUDE_jacks_total_money_l1347_134723


namespace NUMINAMATH_CALUDE_paper_flowers_per_hour_l1347_134793

/-- The number of paper flowers Person B makes per hour -/
def flowers_per_hour_B : ℕ := 80

/-- The number of paper flowers Person A makes per hour -/
def flowers_per_hour_A : ℕ := flowers_per_hour_B - 20

/-- The time it takes Person A to make 120 flowers -/
def time_A : ℚ := 120 / flowers_per_hour_A

/-- The time it takes Person B to make 160 flowers -/
def time_B : ℚ := 160 / flowers_per_hour_B

theorem paper_flowers_per_hour :
  (flowers_per_hour_A = flowers_per_hour_B - 20) ∧
  (time_A = time_B) →
  flowers_per_hour_B = 80 := by
  sorry

end NUMINAMATH_CALUDE_paper_flowers_per_hour_l1347_134793


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l1347_134709

/-- Calculates the dividend percentage given investment details --/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_rate : ℝ)
  (total_dividend : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : total_dividend = 600) :
  let share_cost := share_face_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := total_dividend / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 5 := by
sorry


end NUMINAMATH_CALUDE_dividend_percentage_calculation_l1347_134709


namespace NUMINAMATH_CALUDE_triangle_properties_l1347_134732

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the following properties when certain conditions are met. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a - b = 2 →
  c = 4 →
  Real.sin A = 2 * Real.sin B →
  (a = 4 ∧ b = 2 ∧ Real.cos B = 7/8) ∧
  Real.sin (2*B - π/6) = (21 * Real.sqrt 5 - 17) / 64 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1347_134732


namespace NUMINAMATH_CALUDE_complex_square_equality_l1347_134744

theorem complex_square_equality (a b : ℝ) (i : ℂ) (h : i^2 = -1) 
  (eq : (a : ℂ) + b*i - 2*i = 2 - b*i) : 
  (a + b*i)^2 = 3 + 4*i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equality_l1347_134744


namespace NUMINAMATH_CALUDE_prop_2_prop_4_prop_1_counter_prop_3_counter_l1347_134761

-- Define basic geometric concepts
def Line : Type := sorry
def Plane : Type := sorry
def Point : Type := sorry

-- Define geometric relations
def parallel (a b : Plane) : Prop := sorry
def perpendicular (a b : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_of_intersection (p1 p2 : Plane) : Line := sorry

-- Proposition 2
theorem prop_2 (p1 p2 : Plane) (l : Line) : 
  perpendicular_line_plane l p1 → line_in_plane l p2 → perpendicular p1 p2 := by sorry

-- Proposition 4
theorem prop_4 (p1 p2 : Plane) (l : Line) :
  perpendicular p1 p2 → 
  line_in_plane l p1 → 
  ¬perpendicular_line_plane l (line_of_intersection p1 p2) → 
  ¬perpendicular_line_plane l p2 := by sorry

-- Proposition 1 (counterexample)
theorem prop_1_counter : ∃ (p1 p2 p3 : Plane) (l1 l2 : Line),
  line_in_plane l1 p1 ∧ line_in_plane l2 p1 ∧
  parallel p2 p1 ∧ parallel p3 p1 ∧
  ¬parallel p2 p3 := by sorry

-- Proposition 3 (counterexample)
theorem prop_3_counter : ∃ (l1 l2 l3 : Line),
  perpendicular_line_plane l1 l3 ∧ 
  perpendicular_line_plane l2 l3 ∧
  ¬parallel l1 l2 := by sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_prop_1_counter_prop_3_counter_l1347_134761


namespace NUMINAMATH_CALUDE_square_difference_ratio_l1347_134786

theorem square_difference_ratio : (2045^2 - 2030^2) / (2050^2 - 2025^2) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_ratio_l1347_134786


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l1347_134703

theorem friend_lunch_cost (total : ℕ) (difference : ℕ) (friend_cost : ℕ) : 
  total = 15 → difference = 5 → 
  (∃ (your_cost : ℕ), your_cost + friend_cost = total ∧ friend_cost = your_cost + difference) →
  friend_cost = 10 := by sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l1347_134703


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1347_134740

theorem trigonometric_identity : 
  let cos_45 : ℝ := Real.sqrt 2 / 2
  let tan_30 : ℝ := Real.sqrt 3 / 3
  let sin_60 : ℝ := Real.sqrt 3 / 2
  cos_45^2 + tan_30 * sin_60 = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1347_134740


namespace NUMINAMATH_CALUDE_chinese_digit_mapping_l1347_134708

/-- A function that maps Chinese characters to unique digits 1-9 -/
def ChineseToDigit : Type := Char → Fin 9

/-- The condition that the function maps different characters to different digits -/
def isInjective (f : ChineseToDigit) : Prop :=
  ∀ (c1 c2 : Char), f c1 = f c2 → c1 = c2

/-- The theorem statement -/
theorem chinese_digit_mapping (f : ChineseToDigit) 
  (h_injective : isInjective f)
  (h_zhu : f '祝' = 4)
  (h_he : f '贺' = 8) :
  (f '华') * 100 + (f '杯') * 10 + (f '赛') = 7632 := by
  sorry


end NUMINAMATH_CALUDE_chinese_digit_mapping_l1347_134708


namespace NUMINAMATH_CALUDE_percent_change_equality_l1347_134737

theorem percent_change_equality (x y : ℝ) (p : ℝ) 
  (h1 : x ≠ 0)
  (h2 : y = x * (1 + 0.15) * (1 - p / 100))
  (h3 : y = x) : 
  p = 15 := by
sorry

end NUMINAMATH_CALUDE_percent_change_equality_l1347_134737


namespace NUMINAMATH_CALUDE_highest_y_coordinate_zero_is_highest_y_l1347_134758

theorem highest_y_coordinate (x y : ℝ) : 
  (x - 4)^2 / 25 + y^2 / 49 = 0 → y ≤ 0 :=
by sorry

theorem zero_is_highest_y (x y : ℝ) : 
  (x - 4)^2 / 25 + y^2 / 49 = 0 → ∃ (x₀ y₀ : ℝ), (x₀ - 4)^2 / 25 + y₀^2 / 49 = 0 ∧ y₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_highest_y_coordinate_zero_is_highest_y_l1347_134758


namespace NUMINAMATH_CALUDE_min_value_x_plus_9y_l1347_134770

theorem min_value_x_plus_9y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  x + 9*y ≥ 16 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_9y_l1347_134770


namespace NUMINAMATH_CALUDE_count_l_shapes_l1347_134724

/-- The number of ways to select an L-shaped piece from an m × n chessboard -/
def lShapeCount (m n : ℕ) : ℕ :=
  4 * (m - 1) * (n - 1)

/-- Theorem stating that the number of ways to select an L-shaped piece
    from an m × n chessboard is equal to 4(m-1)(n-1) -/
theorem count_l_shapes (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  lShapeCount m n = 4 * (m - 1) * (n - 1) := by
  sorry

#check count_l_shapes

end NUMINAMATH_CALUDE_count_l_shapes_l1347_134724


namespace NUMINAMATH_CALUDE_min_value_problem_l1347_134781

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := |x + a| + |x - b| + c

-- State the theorem
theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, f a b c x ≥ 4) 
  (hmin_exists : ∃ x, f a b c x = 4) : 
  (a + b + c = 4) ∧ 
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) ∧
  (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 = 8/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1347_134781


namespace NUMINAMATH_CALUDE_bunny_count_l1347_134779

/-- The number of bunnies coming out of their burrows -/
def num_bunnies : ℕ := 
  let times_per_minute : ℕ := 3
  let hours : ℕ := 10
  let minutes_per_hour : ℕ := 60
  let total_times : ℕ := 36000
  total_times / (times_per_minute * hours * minutes_per_hour)

theorem bunny_count : num_bunnies = 20 := by
  sorry

end NUMINAMATH_CALUDE_bunny_count_l1347_134779


namespace NUMINAMATH_CALUDE_reciprocal_sum_inequality_l1347_134774

theorem reciprocal_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / a) + (1 / b) ≥ 4 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_inequality_l1347_134774


namespace NUMINAMATH_CALUDE_crayons_theorem_l1347_134701

/-- The number of crayons in the drawer at the end of Thursday. -/
def crayons_at_end_of_thursday (initial : ℕ) (mary_adds : ℕ) (john_removes : ℕ) (lisa_adds : ℕ) (jeremy_adds : ℕ) (sarah_removes : ℕ) : ℕ :=
  initial + mary_adds - john_removes + lisa_adds + jeremy_adds - sarah_removes

/-- Theorem stating that the number of crayons at the end of Thursday is 13. -/
theorem crayons_theorem : crayons_at_end_of_thursday 7 3 5 4 6 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_crayons_theorem_l1347_134701


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l1347_134731

theorem at_least_one_non_negative (x : ℝ) : 
  let m := x^2 - 1
  let n := 2*x + 2
  max m n ≥ 0 := by sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l1347_134731


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1347_134725

/-- Given an item with a cost price, this theorem proves that if the item is priced at 1.5 times
    its cost price and sold with a 40% profit after a 20 yuan discount, then the cost price
    of the item is 200 yuan. -/
theorem cost_price_calculation (cost_price : ℝ) : 
  (1.5 * cost_price - 20 - cost_price = 0.4 * cost_price) → cost_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1347_134725


namespace NUMINAMATH_CALUDE_probability_ratio_l1347_134798

def num_balls : ℕ := 20
def num_bins : ℕ := 5

def distribution_A : List ℕ := [2, 4, 4, 3, 7]
def distribution_B : List ℕ := [3, 3, 4, 4, 4]

def probability_A : ℚ := (Nat.choose 5 1) * (Nat.choose 4 2) * (Nat.choose 2 1) * (Nat.factorial 20) / 
  ((Nat.factorial 2) * (Nat.factorial 4) * (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 7) * (Nat.choose (num_balls + num_bins - 1) (num_bins - 1)))

def probability_B : ℚ := (Nat.choose 5 2) * (Nat.choose 3 3) * (Nat.factorial 20) / 
  ((Nat.factorial 3) * (Nat.factorial 3) * (Nat.factorial 4) * (Nat.factorial 4) * (Nat.factorial 4) * (Nat.choose (num_balls + num_bins - 1) (num_bins - 1)))

theorem probability_ratio : probability_A / probability_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l1347_134798


namespace NUMINAMATH_CALUDE_cos_double_angle_on_graph_l1347_134746

-- Define the angle α
variable (α : Real)

-- Define the condition that the terminal side of α lies on y = -3x
def terminal_side_on_graph (α : Real) : Prop :=
  ∃ x : Real, Real.tan α = -3 ∧ x ≠ 0

-- State the theorem
theorem cos_double_angle_on_graph (α : Real) 
  (h : terminal_side_on_graph α) : Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_on_graph_l1347_134746


namespace NUMINAMATH_CALUDE_scout_weekend_earnings_l1347_134790

def base_pay : ℝ := 10.00
def tip_per_customer : ℝ := 5.00
def saturday_hours : ℝ := 4
def saturday_customers : ℕ := 5
def sunday_hours : ℝ := 5
def sunday_customers : ℕ := 8

theorem scout_weekend_earnings :
  let saturday_earnings := base_pay * saturday_hours + tip_per_customer * saturday_customers
  let sunday_earnings := base_pay * sunday_hours + tip_per_customer * sunday_customers
  saturday_earnings + sunday_earnings = 155.00 := by
  sorry

end NUMINAMATH_CALUDE_scout_weekend_earnings_l1347_134790


namespace NUMINAMATH_CALUDE_quarters_spent_l1347_134799

def initial_quarters : ℕ := 760
def remaining_quarters : ℕ := 342

theorem quarters_spent : initial_quarters - remaining_quarters = 418 := by
  sorry

end NUMINAMATH_CALUDE_quarters_spent_l1347_134799


namespace NUMINAMATH_CALUDE_hannah_kids_stockings_l1347_134713

theorem hannah_kids_stockings (total_stuffers : ℕ) 
  (candy_canes_per_kid : ℕ) (beanie_babies_per_kid : ℕ) (books_per_kid : ℕ) :
  total_stuffers = 21 ∧ 
  candy_canes_per_kid = 4 ∧ 
  beanie_babies_per_kid = 2 ∧ 
  books_per_kid = 1 →
  ∃ (num_kids : ℕ), 
    num_kids * (candy_canes_per_kid + beanie_babies_per_kid + books_per_kid) = total_stuffers ∧
    num_kids = 3 := by
  sorry

end NUMINAMATH_CALUDE_hannah_kids_stockings_l1347_134713


namespace NUMINAMATH_CALUDE_break_even_price_l1347_134783

/-- Calculates the minimum selling price per component to break even -/
def minimum_selling_price (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (volume : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / volume)

theorem break_even_price 
  (production_cost : ℚ) 
  (shipping_cost : ℚ) 
  (fixed_costs : ℚ) 
  (volume : ℕ) 
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 2)
  (h3 : fixed_costs = 16200)
  (h4 : volume = 150) :
  minimum_selling_price production_cost shipping_cost fixed_costs volume = 190 := by
  sorry

#eval minimum_selling_price 80 2 16200 150

end NUMINAMATH_CALUDE_break_even_price_l1347_134783


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l1347_134707

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l1347_134707


namespace NUMINAMATH_CALUDE_sum_of_abc_is_zero_l1347_134765

theorem sum_of_abc_is_zero 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hnot_all_equal : ¬(a = b ∧ b = c))
  (heq : a^2 / (2*a^2 + b*c) + b^2 / (2*b^2 + c*a) + c^2 / (2*c^2 + a*b) = 1) :
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_abc_is_zero_l1347_134765


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1347_134735

theorem quadratic_equation_solution :
  let a : ℚ := -2
  let b : ℚ := 1
  let c : ℚ := 3
  let x₁ : ℚ := -1
  let x₂ : ℚ := 3/2
  (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1347_134735


namespace NUMINAMATH_CALUDE_total_interest_calculation_l1347_134700

/-- Calculate the total interest for two principal amounts -/
def totalInterest (principal1 principal2 : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal1 + principal2) * rate * time

theorem total_interest_calculation :
  let principal1 : ℝ := 1000
  let principal2 : ℝ := 1400
  let rate : ℝ := 0.03
  let time : ℝ := 4.861111111111111
  abs (totalInterest principal1 principal2 rate time - 350) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l1347_134700


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_a_greater_than_one_l1347_134704

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the first part of the problem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x | 7 < x ∧ x < 10} := by sorry

-- Theorem for the second part of the problem
theorem a_greater_than_one (h : A ∩ C a ≠ ∅) : a > 1 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_a_greater_than_one_l1347_134704


namespace NUMINAMATH_CALUDE_band_member_earnings_l1347_134736

theorem band_member_earnings 
  (attendees : ℕ) 
  (band_share : ℚ) 
  (ticket_price : ℕ) 
  (band_members : ℕ) 
  (h1 : attendees = 500) 
  (h2 : band_share = 70 / 100) 
  (h3 : ticket_price = 30) 
  (h4 : band_members = 4) : 
  (attendees * ticket_price * band_share) / band_members = 2625 := by
sorry

end NUMINAMATH_CALUDE_band_member_earnings_l1347_134736


namespace NUMINAMATH_CALUDE_juice_distribution_l1347_134711

theorem juice_distribution (C : ℝ) (h : C > 0) : 
  let juice_volume := (2/3) * C
  let cups := 4
  let juice_per_cup := juice_volume / cups
  juice_per_cup / C = 1/6 := by sorry

end NUMINAMATH_CALUDE_juice_distribution_l1347_134711


namespace NUMINAMATH_CALUDE_pizza_delivery_gas_theorem_l1347_134782

/-- The amount of gas remaining after a pizza delivery route. -/
def gas_remaining (start : Float) (used : Float) : Float :=
  start - used

/-- Theorem stating that given the starting amount and used amount of gas,
    the remaining amount is correctly calculated. -/
theorem pizza_delivery_gas_theorem :
  gas_remaining 0.5 0.3333333333333333 = 0.1666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_pizza_delivery_gas_theorem_l1347_134782


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1347_134706

/-- The number of ways to choose 2 items from 5 items -/
def choose_2_from_5 : ℕ := 10

/-- The number of rectangles in a 5x5 grid -/
def num_rectangles : ℕ := choose_2_from_5 * choose_2_from_5

theorem rectangles_in_5x5_grid :
  num_rectangles = 100 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1347_134706


namespace NUMINAMATH_CALUDE_probability_is_three_eighths_l1347_134766

/-- Represents a circular field with 8 roads -/
structure CircularField :=
  (radius : ℝ)
  (num_roads : ℕ := 8)

/-- Represents a geologist on the field -/
structure Geologist :=
  (speed : ℝ)
  (time : ℝ)
  (road : ℕ)

/-- Calculates the distance between two geologists -/
def distance_between (g1 g2 : Geologist) (field : CircularField) : ℝ :=
  sorry

/-- Determines if two geologists are more than 8 km apart -/
def more_than_8km_apart (g1 g2 : Geologist) (field : CircularField) : Prop :=
  distance_between g1 g2 field > 8

/-- Calculates the probability of two geologists being more than 8 km apart -/
def probability_more_than_8km_apart (field : CircularField) : ℝ :=
  sorry

theorem probability_is_three_eighths (field : CircularField) 
  (g1 g2 : Geologist) 
  (h1 : field.num_roads = 8) 
  (h2 : g1.speed = 5) 
  (h3 : g2.speed = 5) 
  (h4 : g1.time = 1) 
  (h5 : g2.time = 1) :
  probability_more_than_8km_apart field = 3/8 :=
sorry

end NUMINAMATH_CALUDE_probability_is_three_eighths_l1347_134766


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l1347_134719

theorem sum_of_four_numbers : 2468 + 8642 + 6824 + 4286 = 22220 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l1347_134719


namespace NUMINAMATH_CALUDE_sum_less_than_sum_of_roots_l1347_134778

theorem sum_less_than_sum_of_roots (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : |a - b| < 2) (hbc : |b - c| < 2) (hca : |c - a| < 2) :
  a + b + c < Real.sqrt (a * b + 1) + Real.sqrt (a * c + 1) + Real.sqrt (b * c + 1) := by
  sorry


end NUMINAMATH_CALUDE_sum_less_than_sum_of_roots_l1347_134778


namespace NUMINAMATH_CALUDE_addington_average_temperature_l1347_134759

/-- The average of the daily low temperatures in Addington from September 15th, 2008 through September 19th, 2008, inclusive, is 42.4 degrees Fahrenheit. -/
theorem addington_average_temperature : 
  let temperatures : List ℝ := [40, 47, 45, 41, 39]
  (temperatures.sum / temperatures.length : ℝ) = 42.4 := by
  sorry

end NUMINAMATH_CALUDE_addington_average_temperature_l1347_134759


namespace NUMINAMATH_CALUDE_half_plus_five_equals_thirteen_l1347_134763

theorem half_plus_five_equals_thirteen (n : ℝ) : (1/2) * n + 5 = 13 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_thirteen_l1347_134763


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1347_134777

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (a b : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a * b = k

/-- Given two inversely proportional numbers a and b, 
    if a + b = 60 and a = 3b, then when a = 12, b = 56.25 -/
theorem inverse_proportion_problem (a b : ℝ) 
  (h_inv : InverselyProportional a b) 
  (h_sum : a + b = 60) 
  (h_prop : a = 3 * b) : 
  ∃ b' : ℝ, InverselyProportional 12 b' ∧ b' = 56.25 := by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_problem_l1347_134777


namespace NUMINAMATH_CALUDE_inequality_iff_solution_set_l1347_134748

def inequality (x : ℝ) : Prop :=
  2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 20

def solution_set (x : ℝ) : Prop :=
  x < -3 ∨ (-2 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ 8 < x

theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequality_iff_solution_set_l1347_134748


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1347_134722

theorem triangle_angle_sum (x : ℝ) : 
  36 + 90 + x = 180 → x = 54 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1347_134722


namespace NUMINAMATH_CALUDE_exactly_five_false_propositions_l1347_134757

-- Define the geometric objects
def Line : Type := sorry
def Plane : Type := sorry
def Point : Type := sorry
def Angle : Type := sorry

-- Define geometric relations
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def coplanar (l1 l2 l3 : Line) : Prop := sorry
def collinear (p1 p2 p3 : Point) : Prop := sorry
def onPlane (p : Point) (pl : Plane) : Prop := sorry
def commonPoint (pl1 pl2 : Plane) (p : Point) : Prop := sorry
def sidesParallel (a1 a2 : Angle) : Prop := sorry

-- Define the propositions
def prop1 : Prop := ∀ l1 l2 l3 : Line, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2
def prop2 : Prop := ∀ l1 l2 l3 : Line, intersect l1 l2 ∧ intersect l2 l3 ∧ intersect l3 l1 → coplanar l1 l2 l3
def prop3 : Prop := ∀ p1 p2 p3 p4 : Point, (∃ pl : Plane, ¬(onPlane p1 pl ∧ onPlane p2 pl ∧ onPlane p3 pl ∧ onPlane p4 pl)) → ¬collinear p1 p2 p3 ∧ ¬collinear p1 p2 p4 ∧ ¬collinear p1 p3 p4 ∧ ¬collinear p2 p3 p4
def prop4 : Prop := ∀ pl1 pl2 : Plane, (∃ p1 p2 p3 : Point, commonPoint pl1 pl2 p1 ∧ commonPoint pl1 pl2 p2 ∧ commonPoint pl1 pl2 p3) → pl1 = pl2
def prop5 : Prop := ∃ α β : Plane, ∃! p : Point, commonPoint α β p
def prop6 : Prop := ∀ a1 a2 : Angle, sidesParallel a1 a2 → a1 = a2

-- Theorem statement
theorem exactly_five_false_propositions : 
  ¬prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 ∧ ¬prop5 ∧ ¬prop6 := by sorry

end NUMINAMATH_CALUDE_exactly_five_false_propositions_l1347_134757


namespace NUMINAMATH_CALUDE_total_animals_hunted_l1347_134710

theorem total_animals_hunted (sam rob mark peter : ℕ) : 
  sam = 6 →
  rob = sam / 2 →
  mark = (sam + rob) / 3 →
  peter = 3 * mark →
  sam + rob + mark + peter = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_total_animals_hunted_l1347_134710


namespace NUMINAMATH_CALUDE_range_of_m_l1347_134718

-- Define the function y in terms of x, k, and m
def y (x k m : ℝ) : ℝ := k * x - k + m

-- State the theorem
theorem range_of_m (k m : ℝ) : 
  (∃ x, y x k m = 3 ∧ x = -2) →  -- When x = -2, y = 3
  (k ≠ 0) →  -- k is non-zero (implied by direct proportionality)
  (k < 0) →  -- Slope is negative (passes through 2nd, 3rd, and 4th quadrants)
  (-k + m < 0) →  -- y-intercept is negative (passes through 2nd, 3rd, and 4th quadrants)
  m < -3/2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1347_134718


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1347_134776

theorem quadratic_rewrite :
  ∃ (a b c : ℤ), a > 0 ∧
  (∀ x, 64 * x^2 + 80 * x - 72 = 0 ↔ (a * x + b)^2 = c) ∧
  a + b + c = 110 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1347_134776


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l1347_134797

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Translates a point vertically -/
def translateY (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

/-- The main theorem -/
theorem circle_reflection_translation :
  let Q : Point := { x := 3, y := -4 }
  let Q' := translateY (reflectX Q) 5
  Q'.x = 3 ∧ Q'.y = 9 := by sorry

end NUMINAMATH_CALUDE_circle_reflection_translation_l1347_134797


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1347_134764

theorem sqrt_inequality (a : ℝ) (h : a > 2) : Real.sqrt (a + 2) + Real.sqrt (a - 2) < 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1347_134764


namespace NUMINAMATH_CALUDE_apple_distribution_l1347_134768

/-- Represents the number of apples each person has -/
structure Apples where
  greg : ℕ
  sarah : ℕ
  susan : ℕ
  mark : ℕ

/-- The ratio of Susan's apples to Greg's apples -/
def apple_ratio (a : Apples) : ℚ :=
  a.susan / a.greg

theorem apple_distribution (a : Apples) :
  a.greg = a.sarah ∧
  a.greg + a.sarah = 18 ∧
  a.mark = a.susan - 5 ∧
  a.greg + a.sarah + a.susan + a.mark = 49 →
  apple_ratio a = 2 := by
sorry

end NUMINAMATH_CALUDE_apple_distribution_l1347_134768


namespace NUMINAMATH_CALUDE_intersection_A_B_l1347_134747

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | x^2 - 1 > 0}

theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1347_134747


namespace NUMINAMATH_CALUDE_exactly_one_defective_probability_l1347_134771

theorem exactly_one_defective_probability
  (pass_rate_1 : ℝ)
  (pass_rate_2 : ℝ)
  (h1 : pass_rate_1 = 0.90)
  (h2 : pass_rate_2 = 0.95)
  : (pass_rate_1 * (1 - pass_rate_2)) + ((1 - pass_rate_1) * pass_rate_2) = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_defective_probability_l1347_134771


namespace NUMINAMATH_CALUDE_min_sum_at_6_l1347_134755

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = -14
  sum_5_6 : a 5 + a 6 = -4
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The value of n for which the sum reaches its minimum -/
def min_sum_index (seq : ArithmeticSequence) : ℕ :=
  6

/-- Theorem: The sum of the arithmetic sequence reaches its minimum when n = 6 -/
theorem min_sum_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, sum_n seq (min_sum_index seq) ≤ sum_n seq n :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_6_l1347_134755


namespace NUMINAMATH_CALUDE_line_equation_from_intercepts_l1347_134760

theorem line_equation_from_intercepts (x y : ℝ) :
  (x = -2 ∧ y = 0) ∨ (x = 0 ∧ y = 3) → 3 * x - 2 * y + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_from_intercepts_l1347_134760


namespace NUMINAMATH_CALUDE_volume_weight_proportion_l1347_134714

/-- Given a substance where volume is directly proportional to weight,
    if 48 cubic inches of the substance weigh 112 ounces,
    then 56 ounces of the substance will have a volume of 24 cubic inches. -/
theorem volume_weight_proportion (volume weight : ℝ → ℝ) :
  (∀ w₁ w₂, volume w₁ / volume w₂ = w₁ / w₂) →  -- volume is directly proportional to weight
  volume 112 = 48 →                            -- 48 cubic inches weigh 112 ounces
  volume 56 = 24                               -- 56 ounces have a volume of 24 cubic inches
:= by sorry

end NUMINAMATH_CALUDE_volume_weight_proportion_l1347_134714


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1347_134767

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - k*x + 1 > 0) → -2 < k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1347_134767


namespace NUMINAMATH_CALUDE_battle_station_staffing_ways_l1347_134730

/-- Represents the number of job openings -/
def num_jobs : ℕ := 5

/-- Represents the total number of candidates considered -/
def total_candidates : ℕ := 18

/-- Represents the number of candidates skilled in one area only -/
def specialized_candidates : ℕ := 6

/-- Represents the number of versatile candidates -/
def versatile_candidates : ℕ := total_candidates - specialized_candidates

/-- Represents the number of ways to select the specialized candidates -/
def specialized_selection_ways : ℕ := 2 * 2 * 1 * 1

/-- The main theorem stating the number of ways to staff the battle station -/
theorem battle_station_staffing_ways :
  specialized_selection_ways * versatile_candidates * (versatile_candidates - 1) = 528 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_ways_l1347_134730


namespace NUMINAMATH_CALUDE_minimum_employees_to_hire_l1347_134741

theorem minimum_employees_to_hire (S H : Finset Nat) 
  (h1 : S.card = 120)
  (h2 : H.card = 90)
  (h3 : (S ∩ H).card = 40) :
  (S ∪ H).card = 170 := by
sorry

end NUMINAMATH_CALUDE_minimum_employees_to_hire_l1347_134741


namespace NUMINAMATH_CALUDE_two_tangent_circles_l1347_134727

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent to each other -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Counts the number of circles with radius 4 that are tangent to both given circles -/
def count_tangent_circles (c1 c2 : Circle) : ℕ :=
  sorry

theorem two_tangent_circles 
  (c1 c2 : Circle) 
  (h1 : c1.radius = 2) 
  (h2 : c2.radius = 2) 
  (h3 : are_tangent c1 c2) :
  count_tangent_circles c1 c2 = 2 :=
sorry

end NUMINAMATH_CALUDE_two_tangent_circles_l1347_134727


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1347_134789

theorem polynomial_expansion (z : ℂ) :
  (z^2 - 3*z + 1) * (4*z^4 + z^3 - 2*z^2 + 3) = 4*z^6 - 12*z^5 + 3*z^4 + 4*z^3 - z^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1347_134789


namespace NUMINAMATH_CALUDE_binomial_expectation_from_variance_l1347_134796

/-- 
Given a binomial distribution with 4 trials and probability p of success on each trial,
if the variance of the distribution is 1, then the expected value is 2.
-/
theorem binomial_expectation_from_variance 
  (p : ℝ) 
  (h_prob : 0 ≤ p ∧ p ≤ 1) 
  (h_var : 4 * p * (1 - p) = 1) : 
  4 * p = 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expectation_from_variance_l1347_134796


namespace NUMINAMATH_CALUDE_function_inequality_l1347_134705

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (x - 1) * (deriv f x) ≤ 0)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- Define the theorem
theorem function_inequality (x₁ x₂ : ℝ) (h3 : |x₁ - 1| < |x₂ - 1|) : 
  f (2 - x₁) ≥ f (2 - x₂) := by sorry

end NUMINAMATH_CALUDE_function_inequality_l1347_134705


namespace NUMINAMATH_CALUDE_rotate_from_one_to_six_l1347_134785

/-- Represents a face of a standard six-sided die -/
inductive DieFace
| one
| two
| three
| four
| five
| six

/-- Represents the state of a die with visible faces -/
structure DieState where
  top : DieFace
  front : DieFace
  right : DieFace

/-- Defines the opposite face relation for a standard die -/
def opposite_face (f : DieFace) : DieFace :=
  match f with
  | DieFace.one => DieFace.six
  | DieFace.two => DieFace.five
  | DieFace.three => DieFace.four
  | DieFace.four => DieFace.three
  | DieFace.five => DieFace.two
  | DieFace.six => DieFace.one

/-- Simulates a 90° clockwise rotation of the die -/
def rotate_90_clockwise (s : DieState) : DieState :=
  { top := s.right
  , front := s.top
  , right := opposite_face s.front }

/-- Theorem: After a 90° clockwise rotation from a state where 1 is visible,
    the opposite face (6) becomes visible -/
theorem rotate_from_one_to_six (initial : DieState) 
    (h : initial.top = DieFace.one) : 
    ∃ (rotated : DieState), rotated = rotate_90_clockwise initial ∧ 
    (rotated.top = DieFace.six ∨ rotated.front = DieFace.six ∨ rotated.right = DieFace.six) :=
  sorry


end NUMINAMATH_CALUDE_rotate_from_one_to_six_l1347_134785


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1347_134743

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2022
  let y : ℝ := -Real.sqrt 2
  4 * x * y + (2 * x - y) * (2 * x + y) - (2 * x + y)^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1347_134743


namespace NUMINAMATH_CALUDE_cube_root_of_3x_plus_4y_is_3_l1347_134780

theorem cube_root_of_3x_plus_4y_is_3 (x y : ℝ) (h : y = Real.sqrt (x - 5) + Real.sqrt (5 - x) + 3) :
  (3 * x + 4 * y) ^ (1/3 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_3x_plus_4y_is_3_l1347_134780


namespace NUMINAMATH_CALUDE_simplify_like_terms_l1347_134712

theorem simplify_like_terms (x : ℝ) : 5*x + 2*x + 7*x = 14*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_like_terms_l1347_134712


namespace NUMINAMATH_CALUDE_number_problem_l1347_134721

theorem number_problem (x : ℝ) : (10 * x = x + 34.65) → x = 3.85 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1347_134721


namespace NUMINAMATH_CALUDE_base6_addition_l1347_134773

/-- Converts a base 6 number represented as a list of digits to its decimal (base 10) equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal (base 10) number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The main theorem stating that 3454₆ + 12345₆ = 142042₆ in base 6 -/
theorem base6_addition :
  decimalToBase6 (base6ToDecimal [3, 4, 5, 4] + base6ToDecimal [1, 2, 3, 4, 5]) =
  [1, 4, 2, 0, 4, 2] := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_l1347_134773


namespace NUMINAMATH_CALUDE_fifteen_mangoes_make_120_lassis_l1347_134726

/-- Given that 3 mangoes can make 24 lassis, this function calculates
    the number of lassis that can be made from a given number of mangoes. -/
def lassisFromMangoes (mangoes : ℕ) : ℕ :=
  (24 * mangoes) / 3

/-- Theorem stating that 15 mangoes will produce 120 lassis -/
theorem fifteen_mangoes_make_120_lassis :
  lassisFromMangoes 15 = 120 := by
  sorry

#eval lassisFromMangoes 15

end NUMINAMATH_CALUDE_fifteen_mangoes_make_120_lassis_l1347_134726


namespace NUMINAMATH_CALUDE_sum_three_x_square_y_correct_l1347_134754

/-- The sum of three times x and the square of y -/
def sum_three_x_square_y (x y : ℝ) : ℝ := 3 * x + y^2

theorem sum_three_x_square_y_correct (x y : ℝ) : 
  sum_three_x_square_y x y = 3 * x + y^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_three_x_square_y_correct_l1347_134754


namespace NUMINAMATH_CALUDE_scientific_notation_of_830_billion_l1347_134733

theorem scientific_notation_of_830_billion :
  (830 : ℝ) * (10 ^ 9) = 8.3 * (10 ^ 11) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_830_billion_l1347_134733


namespace NUMINAMATH_CALUDE_melanie_dimes_count_l1347_134749

/-- The total number of dimes Melanie has after receiving dimes from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem stating that Melanie's total dimes is the sum of her initial dimes and those received from her parents -/
theorem melanie_dimes_count (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) :
  total_dimes initial from_dad from_mom = initial + from_dad + from_mom :=
by
  sorry

#eval total_dimes 19 39 25

end NUMINAMATH_CALUDE_melanie_dimes_count_l1347_134749


namespace NUMINAMATH_CALUDE_complex_magnitude_real_part_l1347_134750

theorem complex_magnitude_real_part (t : ℝ) : 
  t > 0 → Complex.abs (9 + t * Complex.I) = 15 → Complex.re (9 + t * Complex.I) = 9 → t = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_real_part_l1347_134750


namespace NUMINAMATH_CALUDE_range_of_z_l1347_134728

theorem range_of_z (α β z : ℝ) 
  (h1 : -2 < α ∧ α ≤ 3) 
  (h2 : 2 < β ∧ β ≤ 4) 
  (h3 : z = 2*α - (1/2)*β) : 
  -6 < z ∧ z < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_z_l1347_134728


namespace NUMINAMATH_CALUDE_volumes_equal_l1347_134716

/-- The region bounded by x² = 4y, x² = -4y, x = 4, x = -4 -/
def Region1 (x y : ℝ) : Prop :=
  (x^2 = 4*y ∨ x^2 = -4*y) ∧ (x ≤ 4 ∧ x ≥ -4)

/-- The region defined by x²y² ≤ 16, x² + (y-2)² ≥ 4, x² + (y+2)² ≥ 4 -/
def Region2 (x y : ℝ) : Prop :=
  x^2 * y^2 ≤ 16 ∧ x^2 + (y-2)^2 ≥ 4 ∧ x^2 + (y+2)^2 ≥ 4

/-- The volume of the solid obtained by rotating Region1 around the y-axis -/
noncomputable def V1 : ℝ := sorry

/-- The volume of the solid obtained by rotating Region2 around the y-axis -/
noncomputable def V2 : ℝ := sorry

/-- The volumes of the two solids are equal -/
theorem volumes_equal : V1 = V2 := by sorry

end NUMINAMATH_CALUDE_volumes_equal_l1347_134716


namespace NUMINAMATH_CALUDE_products_from_equipment_B_l1347_134739

/-- Given a total number of products and a stratified sample, 
    calculate the number of products produced by equipment B -/
theorem products_from_equipment_B 
  (total : ℕ) 
  (sample_size : ℕ) 
  (sample_A : ℕ) 
  (h1 : total = 4800)
  (h2 : sample_size = 80)
  (h3 : sample_A = 50) : 
  total - (total * sample_A / sample_size) = 1800 :=
by sorry

end NUMINAMATH_CALUDE_products_from_equipment_B_l1347_134739


namespace NUMINAMATH_CALUDE_cos_x_value_l1347_134795

theorem cos_x_value (x : Real) (h1 : Real.tan (x + Real.pi / 4) = 2) 
  (h2 : x ∈ Set.Icc (Real.pi) (3 * Real.pi / 2)) : 
  Real.cos x = - (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_value_l1347_134795


namespace NUMINAMATH_CALUDE_first_quarter_profit_determination_l1347_134751

/-- Represents the quarterly profits of a store in dollars. -/
structure QuarterlyProfits where
  first : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the annual profit given quarterly profits. -/
def annualProfit (q : QuarterlyProfits) : ℕ :=
  q.first + q.third + q.fourth

/-- Theorem stating that given the annual profit and profits from the third and fourth quarters,
    the first quarter profit can be determined. -/
theorem first_quarter_profit_determination
  (annual_profit : ℕ)
  (third_quarter : ℕ)
  (fourth_quarter : ℕ)
  (h1 : third_quarter = 3000)
  (h2 : fourth_quarter = 2000)
  (h3 : annual_profit = 8000)
  (h4 : ∃ q : QuarterlyProfits, q.third = third_quarter ∧ q.fourth = fourth_quarter ∧ annualProfit q = annual_profit) :
  ∃ q : QuarterlyProfits, q.first = 3000 ∧ q.third = third_quarter ∧ q.fourth = fourth_quarter ∧ annualProfit q = annual_profit :=
by sorry

end NUMINAMATH_CALUDE_first_quarter_profit_determination_l1347_134751


namespace NUMINAMATH_CALUDE_max_sales_on_day_40_l1347_134753

def salesVolume (t : ℕ) : ℝ := -t + 110

def price (t : ℕ) : ℝ :=
  if t ≤ 40 then t + 8 else -0.5 * t + 69

def salesAmount (t : ℕ) : ℝ := salesVolume t * price t

theorem max_sales_on_day_40 :
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 100 → salesAmount t ≤ salesAmount 40 ∧ salesAmount 40 = 3360 :=
by sorry

end NUMINAMATH_CALUDE_max_sales_on_day_40_l1347_134753


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l1347_134720

theorem arithmetic_progression_problem (a d : ℝ) : 
  (a - 2*d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2*d)^3 = 0 ∧
  (a - 2*d)^4 + (a - d)^4 + a^4 + (a + d)^4 + (a + 2*d)^4 = 136 →
  a - 2*d = -2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l1347_134720
