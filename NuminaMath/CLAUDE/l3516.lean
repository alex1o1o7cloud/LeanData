import Mathlib

namespace NUMINAMATH_CALUDE_vegetables_per_week_l3516_351648

theorem vegetables_per_week (total_points : ℕ) (points_per_vegetable : ℕ) 
  (num_students : ℕ) (num_weeks : ℕ) 
  (h1 : total_points = 200)
  (h2 : points_per_vegetable = 2)
  (h3 : num_students = 25)
  (h4 : num_weeks = 2) :
  (total_points / points_per_vegetable / num_students) / num_weeks = 2 :=
by
  sorry

#check vegetables_per_week

end NUMINAMATH_CALUDE_vegetables_per_week_l3516_351648


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l3516_351670

theorem right_triangle_max_ratio (a b c A : ℝ) : 
  a > 0 → b > 0 → c > 0 → A > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  A = (1/2) * a * b →  -- Area formula
  (a + b + A) / c ≤ (5/4) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l3516_351670


namespace NUMINAMATH_CALUDE_cost_of_900_candies_l3516_351686

/-- The cost of buying a given number of chocolate candies -/
def cost_of_candies (num_candies : ℕ) : ℚ :=
  let candies_per_box : ℕ := 30
  let cost_per_box : ℚ := 7.5
  let discount_threshold : ℕ := 500
  let discount_rate : ℚ := 0.1
  let num_boxes : ℕ := num_candies / candies_per_box
  let discounted_cost_per_box : ℚ := if num_candies > discount_threshold then cost_per_box * (1 - discount_rate) else cost_per_box
  (num_boxes : ℚ) * discounted_cost_per_box

/-- The cost of 900 chocolate candies is $202.50 -/
theorem cost_of_900_candies : cost_of_candies 900 = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_900_candies_l3516_351686


namespace NUMINAMATH_CALUDE_triangle_inequality_with_circumradius_l3516_351682

-- Define a structure for a triangle with its circumradius
structure Triangle :=
  (a b c : ℝ)
  (R : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hR : R > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (circumradius : R = (a * b * c) / (4 * area))
  (area : ℝ)
  (area_positive : area > 0)

-- State the theorem
theorem triangle_inequality_with_circumradius (t : Triangle) :
  1 / (t.a * t.b) + 1 / (t.b * t.c) + 1 / (t.c * t.a) ≥ 1 / (t.R ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_circumradius_l3516_351682


namespace NUMINAMATH_CALUDE_sphere_enclosed_by_truncated_cone_l3516_351621

theorem sphere_enclosed_by_truncated_cone (R r r' ζ : ℝ) 
  (h_positive : R > 0)
  (h_volume : (4/3) * π * R^3 * 2 = (4/3) * π * (r^2 + r * r' + r'^2) * R)
  (h_generator : (r + r')^2 = 4 * R^2 + (r - r')^2)
  (h_contact : ζ = (2 * r * r') / (r + r')) :
  r = (R/2) * (Real.sqrt 5 + 1) ∧ 
  r' = (R/2) * (Real.sqrt 5 - 1) ∧ 
  ζ = (2 * R * Real.sqrt 5) / 5 := by
sorry

end NUMINAMATH_CALUDE_sphere_enclosed_by_truncated_cone_l3516_351621


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3516_351650

theorem book_arrangement_count : ℕ := by
  -- Define the number of math books and English books
  let math_books : ℕ := 4
  let english_books : ℕ := 4

  -- Define the number of ways to arrange math books
  let math_arrangements : ℕ := Nat.factorial math_books

  -- Define the number of ways to arrange English books
  let english_arrangements : ℕ := Nat.factorial english_books

  -- Define the number of ways to arrange the two blocks (always 1 in this case)
  let block_arrangements : ℕ := 1

  -- Calculate the total number of arrangements
  let total_arrangements : ℕ := block_arrangements * math_arrangements * english_arrangements

  -- Prove that the total number of arrangements is 576
  sorry

-- The final statement to be proven
#check book_arrangement_count

end NUMINAMATH_CALUDE_book_arrangement_count_l3516_351650


namespace NUMINAMATH_CALUDE_angle_120_degrees_is_200_vens_l3516_351628

/-- Represents the number of vens in a full circle -/
def vens_in_full_circle : ℕ := 600

/-- Represents the number of degrees in a full circle -/
def degrees_in_full_circle : ℕ := 360

/-- Represents the angle in degrees we want to convert to vens -/
def angle_in_degrees : ℕ := 120

/-- Theorem stating that 120 degrees is equivalent to 200 vens -/
theorem angle_120_degrees_is_200_vens :
  (angle_in_degrees : ℚ) * vens_in_full_circle / degrees_in_full_circle = 200 := by
  sorry


end NUMINAMATH_CALUDE_angle_120_degrees_is_200_vens_l3516_351628


namespace NUMINAMATH_CALUDE_symmetric_line_values_l3516_351619

/-- Two lines are symmetric with respect to the origin if for any point (x, y) on one line,
    the point (-x, -y) lies on the other line. -/
def symmetric_lines (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a * x + 3 * y - 9 = 0 ↔ x - 3 * y + b = 0

/-- If the line ax + 3y - 9 = 0 is symmetric to the line x - 3y + b = 0
    with respect to the origin, then a = -1 and b = -9. -/
theorem symmetric_line_values (a b : ℝ) (h : symmetric_lines a b) : a = -1 ∧ b = -9 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_values_l3516_351619


namespace NUMINAMATH_CALUDE_square_difference_l3516_351683

theorem square_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l3516_351683


namespace NUMINAMATH_CALUDE_equation_solution_l3516_351615

theorem equation_solution : 
  {x : ℝ | (16:ℝ)^x - (5/2) * (2:ℝ)^(2*x+1) + 4 = 0} = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3516_351615


namespace NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_in_triangle_l3516_351602

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A centrally symmetric polygon -/
structure CentrallySymmetricPolygon where
  vertices : List (ℝ × ℝ)
  center : ℝ × ℝ
  isSymmetric : ∀ v ∈ vertices, ∃ v' ∈ vertices, v' = (2 * center.1 - v.1, 2 * center.2 - v.2)

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isPointInTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Check if a polygon is inside a triangle -/
def isPolygonInTriangle (p : CentrallySymmetricPolygon) (t : Triangle) : Prop :=
  ∀ v ∈ p.vertices, isPointInTriangle v t

/-- The area of a centrally symmetric polygon -/
def polygonArea (p : CentrallySymmetricPolygon) : ℝ := sorry

/-- The theorem stating that the largest centrally symmetric polygon 
    inscribed in a triangle has 2/3 the area of the triangle -/
theorem largest_centrally_symmetric_polygon_in_triangle 
  (t : Triangle) : 
  (∃ p : CentrallySymmetricPolygon, 
    isPolygonInTriangle p t ∧ 
    (∀ q : CentrallySymmetricPolygon, 
      isPolygonInTriangle q t → polygonArea q ≤ polygonArea p)) → 
  (∃ p : CentrallySymmetricPolygon, 
    isPolygonInTriangle p t ∧ 
    polygonArea p = (2/3) * triangleArea t) := by
  sorry

end NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_in_triangle_l3516_351602


namespace NUMINAMATH_CALUDE_cans_recycling_l3516_351629

theorem cans_recycling (total_cans : ℕ) (saturday_bags : ℕ) (cans_per_bag : ℕ) : 
  total_cans = 42 →
  saturday_bags = 4 →
  cans_per_bag = 6 →
  (total_cans - saturday_bags * cans_per_bag) / cans_per_bag = 3 :=
by sorry

end NUMINAMATH_CALUDE_cans_recycling_l3516_351629


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l3516_351666

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : k = 2012^2 + 2^2012 → (k^2 + 2^k) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l3516_351666


namespace NUMINAMATH_CALUDE_compound_interest_rate_l3516_351692

theorem compound_interest_rate (P : ℝ) (r : ℝ) : 
  P * (1 + r)^2 = 240 → 
  P * (1 + r) = 217.68707482993196 → 
  r = 0.1025 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l3516_351692


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3516_351646

def f (x : ℝ) : ℝ := (x - 2)^2 - 3

theorem vertex_of_quadratic :
  ∃ (a b c : ℝ), f x = a * (x - b)^2 + c ∧ f b = c ∧ b = 2 ∧ c = -3 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3516_351646


namespace NUMINAMATH_CALUDE_total_repair_time_l3516_351660

/-- Represents the time in minutes required for each repair task for different shoe types -/
structure ShoeRepairTime where
  buckle : ℕ
  strap : ℕ
  sole : ℕ

/-- Represents the number of shoes repaired in a session -/
structure SessionRepair where
  flat : ℕ
  sandal : ℕ
  highHeel : ℕ

/-- Calculates the total repair time for a given shoe type and quantity -/
def repairTime (time : ShoeRepairTime) (quantity : ℕ) : ℕ :=
  (time.buckle + time.strap + time.sole) * quantity

/-- Calculates the total repair time for a session -/
def sessionTime (flat : ShoeRepairTime) (sandal : ShoeRepairTime) (highHeel : ShoeRepairTime) (session : SessionRepair) : ℕ :=
  repairTime flat session.flat + repairTime sandal session.sandal + repairTime highHeel session.highHeel

theorem total_repair_time :
  let flat := ShoeRepairTime.mk 3 8 9
  let sandal := ShoeRepairTime.mk 4 5 0
  let highHeel := ShoeRepairTime.mk 6 12 10
  let session1 := SessionRepair.mk 6 4 3
  let session2 := SessionRepair.mk 4 7 5
  let breakTime := 15
  sessionTime flat sandal highHeel session1 + sessionTime flat sandal highHeel session2 + breakTime = 538 := by
  sorry

end NUMINAMATH_CALUDE_total_repair_time_l3516_351660


namespace NUMINAMATH_CALUDE_quadratic_form_simplification_l3516_351639

theorem quadratic_form_simplification 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ K : ℝ, ∀ x : ℝ, 
    (x + a)^2 / ((a - b) * (a - c)) + 
    (x + b)^2 / ((b - a) * (b - c + 2)) + 
    (x + c)^2 / ((c - a) * (c - b)) = 
    x^2 - (a + b + c) * x + K :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_simplification_l3516_351639


namespace NUMINAMATH_CALUDE_random_events_count_l3516_351654

-- Define the type for events
inductive Event
| DiceRoll : Event
| PearFall : Event
| LotteryWin : Event
| SecondChild : Event
| WaterBoil : Event

-- Define a function to check if an event is random
def isRandom (e : Event) : Bool :=
  match e with
  | Event.DiceRoll => true
  | Event.PearFall => false
  | Event.LotteryWin => true
  | Event.SecondChild => true
  | Event.WaterBoil => false

-- Define the list of events
def eventList : List Event := [
  Event.DiceRoll,
  Event.PearFall,
  Event.LotteryWin,
  Event.SecondChild,
  Event.WaterBoil
]

-- Theorem: The number of random events in the list is 3
theorem random_events_count : 
  (eventList.filter isRandom).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_random_events_count_l3516_351654


namespace NUMINAMATH_CALUDE_price_two_bracelets_is_eight_l3516_351647

/-- Represents the bracelet selling scenario -/
structure BraceletSale where
  initialStock : ℕ
  singlePrice : ℕ
  singleRevenue : ℕ
  totalRevenue : ℕ

/-- Calculates the price for two bracelets -/
def priceTwoBracelets (sale : BraceletSale) : ℕ :=
  let singleSold := sale.singleRevenue / sale.singlePrice
  let remainingBracelets := sale.initialStock - singleSold
  let pairRevenue := sale.totalRevenue - sale.singleRevenue
  let pairsSold := remainingBracelets / 2
  pairRevenue / pairsSold

/-- Theorem stating that the price for two bracelets is 8 -/
theorem price_two_bracelets_is_eight (sale : BraceletSale) 
  (h1 : sale.initialStock = 30)
  (h2 : sale.singlePrice = 5)
  (h3 : sale.singleRevenue = 60)
  (h4 : sale.totalRevenue = 132) : 
  priceTwoBracelets sale = 8 := by
  sorry

end NUMINAMATH_CALUDE_price_two_bracelets_is_eight_l3516_351647


namespace NUMINAMATH_CALUDE_probability_log3_is_integer_l3516_351600

/-- A three-digit number is a natural number between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The count of three-digit numbers that are powers of 3. -/
def CountPowersOfThree : ℕ := 2

/-- The total count of three-digit numbers. -/
def TotalThreeDigitNumbers : ℕ := 900

/-- The probability of a randomly chosen three-digit number being a power of 3. -/
def ProbabilityPowerOfThree : ℚ := CountPowersOfThree / TotalThreeDigitNumbers

theorem probability_log3_is_integer :
  ProbabilityPowerOfThree = 1 / 450 := by sorry

end NUMINAMATH_CALUDE_probability_log3_is_integer_l3516_351600


namespace NUMINAMATH_CALUDE_strings_needed_is_302_l3516_351607

/-- Calculates the total number of strings needed for a set of instruments, including extra strings due to machine malfunction --/
def total_strings_needed (num_basses : ℕ) (strings_per_bass : ℕ) (guitar_multiplier : ℕ) 
  (strings_per_guitar : ℕ) (eight_string_guitar_reduction : ℕ) (strings_per_eight_string_guitar : ℕ)
  (strings_per_twelve_string_guitar : ℕ) (nylon_strings_per_eight_string_guitar : ℕ)
  (nylon_strings_per_twelve_string_guitar : ℕ) (malfunction_rate : ℕ) : ℕ :=
  let num_guitars := num_basses * guitar_multiplier
  let num_eight_string_guitars := num_guitars - eight_string_guitar_reduction
  let num_twelve_string_guitars := num_basses
  let total_strings := 
    num_basses * strings_per_bass +
    num_guitars * strings_per_guitar +
    num_eight_string_guitars * strings_per_eight_string_guitar +
    num_twelve_string_guitars * strings_per_twelve_string_guitar
  let extra_strings := (total_strings + malfunction_rate - 1) / malfunction_rate
  total_strings + extra_strings

/-- Theorem stating that given the specific conditions, the total number of strings needed is 302 --/
theorem strings_needed_is_302 : 
  total_strings_needed 5 4 3 6 2 8 12 2 6 10 = 302 := by
  sorry

end NUMINAMATH_CALUDE_strings_needed_is_302_l3516_351607


namespace NUMINAMATH_CALUDE_inequality_solution_l3516_351624

theorem inequality_solution (x : ℝ) :
  (x + 2) / ((x + 1)^2) < 0 ↔ x < -2 ∧ x ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3516_351624


namespace NUMINAMATH_CALUDE_f_zero_points_range_l3516_351627

/-- The function f(x) = ax^2 + x - 1 + 3a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - 1 + 3 * a

/-- The set of a values for which f has zero points in [-1, 1] -/
def A : Set ℝ := {a : ℝ | ∃ x, x ∈ Set.Icc (-1 : ℝ) 1 ∧ f a x = 0}

theorem f_zero_points_range :
  A = Set.Icc (0 : ℝ) (1/2) :=
sorry

end NUMINAMATH_CALUDE_f_zero_points_range_l3516_351627


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_37_l3516_351665

theorem sum_of_divisors_of_37 (h : Nat.Prime 37) : 
  (Finset.filter (· ∣ 37) (Finset.range 38)).sum id = 38 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_37_l3516_351665


namespace NUMINAMATH_CALUDE_sector_central_angle_l3516_351611

/-- Given a sector with perimeter 6 and area 2, its central angle in radians is either 4 or 1 -/
theorem sector_central_angle (r l : ℝ) : 
  2 * r + l = 6 →
  1 / 2 * l * r = 2 →
  l / r = 4 ∨ l / r = 1 :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3516_351611


namespace NUMINAMATH_CALUDE_none_always_true_l3516_351643

/-- Given r > 0 and x^2 + y^2 > x^2y^2 for x, y ≠ 0, none of the following statements are true for all x and y -/
theorem none_always_true (r : ℝ) (x y : ℝ) (hr : r > 0) (hxy : x ≠ 0 ∧ y ≠ 0) (h : x^2 + y^2 > x^2 * y^2) :
  ¬(∀ x y : ℝ, -x > -y) ∧
  ¬(∀ x y : ℝ, -x > y) ∧
  ¬(∀ x y : ℝ, 1 > -y/x) ∧
  ¬(∀ x y : ℝ, 1 < x/y) :=
by sorry

end NUMINAMATH_CALUDE_none_always_true_l3516_351643


namespace NUMINAMATH_CALUDE_rent_is_840_l3516_351638

/-- The total rent for a pasture shared by three people --/
def total_rent (a_horses b_horses c_horses : ℕ) (a_months b_months c_months : ℕ) (b_rent : ℕ) : ℕ :=
  let a_horse_months := a_horses * a_months
  let b_horse_months := b_horses * b_months
  let c_horse_months := c_horses * c_months
  let total_horse_months := a_horse_months + b_horse_months + c_horse_months
  (b_rent * total_horse_months) / b_horse_months

/-- Theorem stating that the total rent is 840 given the problem conditions --/
theorem rent_is_840 :
  total_rent 12 16 18 8 9 6 348 = 840 := by
  sorry

end NUMINAMATH_CALUDE_rent_is_840_l3516_351638


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3516_351669

theorem contrapositive_equivalence (a b : ℝ) :
  (∀ a b, a > b → a - 1 > b - 1) ↔ (∀ a b, a - 1 ≤ b - 1 → a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3516_351669


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3516_351642

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → Complex.im (5 * i / (1 - 2 * i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3516_351642


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l3516_351664

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The perimeter of the trapezoid -/
  perimeter : ℝ
  /-- The diagonal bisects the obtuse angle -/
  diagonalBisectsObtuseAngle : Prop

/-- The area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific isosceles trapezoid is 96 -/
theorem area_of_specific_trapezoid :
  ∀ (t : IsoscelesTrapezoid),
    t.smallerBase = 3 ∧
    t.perimeter = 42 ∧
    t.diagonalBisectsObtuseAngle →
    areaOfTrapezoid t = 96 :=
  sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l3516_351664


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3516_351603

/-- Given two concentric circles where a chord is tangent to the smaller circle,
    this theorem proves the area of the region between the circles. -/
theorem area_between_concentric_circles
  (outer_radius inner_radius chord_length : ℝ)
  (h_outer_positive : 0 < outer_radius)
  (h_inner_positive : 0 < inner_radius)
  (h_outer_greater : inner_radius < outer_radius)
  (h_chord_tangent : chord_length^2 = outer_radius^2 - inner_radius^2)
  (h_chord_length : chord_length = 100) :
  (outer_radius^2 - inner_radius^2) * π = 2000 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3516_351603


namespace NUMINAMATH_CALUDE_triangle_incenter_inequality_l3516_351684

theorem triangle_incenter_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1/4 < ((a+b)*(b+c)*(c+a)) / ((a+b+c)^3) ∧ ((a+b)*(b+c)*(c+a)) / ((a+b+c)^3) ≤ 8/27 := by
  sorry

end NUMINAMATH_CALUDE_triangle_incenter_inequality_l3516_351684


namespace NUMINAMATH_CALUDE_fraction_simplification_l3516_351632

theorem fraction_simplification (x y : ℚ) (hx : x = 4/3) (hy : y = 8/6) : 
  (6 * x^2 + 4 * y) / (36 * x * y) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3516_351632


namespace NUMINAMATH_CALUDE_paul_initial_books_l3516_351606

/-- Represents the number of books and pens Paul has -/
structure PaulsItems where
  books : ℕ
  pens : ℕ

/-- Represents the change in Paul's items after the garage sale -/
structure GarageSale where
  booksRemaining : ℕ
  pensRemaining : ℕ
  booksSold : ℕ

def initialItems : PaulsItems where
  books := 0  -- Unknown initial number of books
  pens := 55

def afterSale : GarageSale where
  booksRemaining := 66
  pensRemaining := 59
  booksSold := 42

theorem paul_initial_books :
  initialItems.books = afterSale.booksRemaining + afterSale.booksSold :=
by sorry

end NUMINAMATH_CALUDE_paul_initial_books_l3516_351606


namespace NUMINAMATH_CALUDE_drawing_probability_comparison_l3516_351604

theorem drawing_probability_comparison : 
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let draws : ℕ := 3

  let prob_with_replacement : ℚ := 3 / 8
  let prob_without_replacement : ℚ := 5 / 12

  prob_without_replacement > prob_with_replacement := by
  sorry

end NUMINAMATH_CALUDE_drawing_probability_comparison_l3516_351604


namespace NUMINAMATH_CALUDE_extremum_and_intersection_implies_m_range_l3516_351662

def f (x : ℝ) := x^3 - 3*x - 1

theorem extremum_and_intersection_implies_m_range :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≤ f (-1) ∨ f x ≥ f (-1)) →
  (∃ m : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) →
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) → 
  -3 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_extremum_and_intersection_implies_m_range_l3516_351662


namespace NUMINAMATH_CALUDE_number_of_cube_nets_l3516_351673

/-- A net is a 2D arrangement of squares that can be folded to form a polyhedron -/
def Net : Type := Unit

/-- A cube net is a specific type of net that can be folded to form a cube -/
def CubeNet : Type := Net

/-- Function to count the number of distinct cube nets -/
def count_distinct_cube_nets : ℕ := sorry

/-- Theorem stating that the number of distinct cube nets is 11 -/
theorem number_of_cube_nets : count_distinct_cube_nets = 11 := by sorry

end NUMINAMATH_CALUDE_number_of_cube_nets_l3516_351673


namespace NUMINAMATH_CALUDE_abs_m_minus_n_equals_three_l3516_351689

theorem abs_m_minus_n_equals_three (m n : ℝ) 
  (h1 : m * n = 4) 
  (h2 : m + n = 5) : 
  |m - n| = 3 := by
sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_equals_three_l3516_351689


namespace NUMINAMATH_CALUDE_knights_selection_ways_l3516_351687

/-- Represents the number of knights at the round table -/
def total_knights : ℕ := 12

/-- Represents the number of knights to be chosen -/
def knights_to_choose : ℕ := 5

/-- Represents the number of ways to choose knights in a linear arrangement -/
def linear_arrangements : ℕ := Nat.choose (total_knights - knights_to_choose + 1) knights_to_choose

/-- Represents the number of invalid arrangements (where first and last knights are adjacent) -/
def invalid_arrangements : ℕ := Nat.choose (total_knights - knights_to_choose - 1) (knights_to_choose - 2)

/-- Theorem stating the number of ways to choose knights under the given conditions -/
theorem knights_selection_ways : 
  linear_arrangements - invalid_arrangements = 36 := by sorry

end NUMINAMATH_CALUDE_knights_selection_ways_l3516_351687


namespace NUMINAMATH_CALUDE_park_width_l3516_351667

/-- Given a rectangular park with specified length, tree density, and total number of trees,
    prove that the width of the park is as calculated. -/
theorem park_width (length : ℝ) (tree_density : ℝ) (total_trees : ℝ) (width : ℝ) : 
  length = 1000 →
  tree_density = 1 / 20 →
  total_trees = 100000 →
  width = total_trees / (length * tree_density) →
  width = 2000 :=
by sorry

end NUMINAMATH_CALUDE_park_width_l3516_351667


namespace NUMINAMATH_CALUDE_mans_running_speed_l3516_351645

/-- A proof that calculates a man's running speed given his walking speed and times. -/
theorem mans_running_speed (walking_speed : ℝ) (walking_time : ℝ) (running_time : ℝ) :
  walking_speed = 8 →
  walking_time = 3 →
  running_time = 1 →
  walking_speed * walking_time / running_time = 24 := by
  sorry

#check mans_running_speed

end NUMINAMATH_CALUDE_mans_running_speed_l3516_351645


namespace NUMINAMATH_CALUDE_inequality_theorem_l3516_351671

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop :=
  |x^2 - 2*x - 8| ≤ 2 * |x - 4| * |x + 2|

-- Define the second condition for x > 1
def second_condition (x m : ℝ) : Prop :=
  x > 1 → x^2 - 2*x - 8 ≥ (m + 2)*x - m - 15

-- Theorem statement
theorem inequality_theorem :
  (∀ x : ℝ, inequality_condition x) ∧
  (∀ m : ℝ, (∀ x : ℝ, second_condition x m) → m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3516_351671


namespace NUMINAMATH_CALUDE_cut_cube_properties_l3516_351691

/-- A cube with one corner cut off -/
structure CutCube where
  vertices : Finset (ℝ × ℝ × ℝ)
  faces : Finset (Finset (ℝ × ℝ × ℝ))

/-- Properties of a cube with one corner cut off -/
def is_valid_cut_cube (c : CutCube) : Prop :=
  c.vertices.card = 10 ∧ c.faces.card = 9

/-- Theorem: A cube with one corner cut off has 10 vertices and 9 faces -/
theorem cut_cube_properties (c : CutCube) (h : is_valid_cut_cube c) :
  c.vertices.card = 10 ∧ c.faces.card = 9 := by
  sorry


end NUMINAMATH_CALUDE_cut_cube_properties_l3516_351691


namespace NUMINAMATH_CALUDE_disneyland_attractions_permutations_l3516_351612

theorem disneyland_attractions_permutations :
  Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_disneyland_attractions_permutations_l3516_351612


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3516_351698

theorem quadratic_rewrite (b : ℝ) (h1 : b > 0) :
  (∃ m : ℝ, ∀ x : ℝ, x^2 + b*x + 108 = (x + m)^2 - 4) →
  b = 8 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3516_351698


namespace NUMINAMATH_CALUDE_expression_equal_to_five_l3516_351634

theorem expression_equal_to_five : 3^2 - 2^2 = 5 := by
  sorry

#check expression_equal_to_five

end NUMINAMATH_CALUDE_expression_equal_to_five_l3516_351634


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l3516_351680

theorem orange_juice_fraction (pitcher_capacity : ℚ) 
  (pitcher1_orange : ℚ) (pitcher1_apple : ℚ)
  (pitcher2_orange : ℚ) (pitcher2_apple : ℚ) :
  pitcher_capacity = 800 →
  pitcher1_orange = 1/4 →
  pitcher1_apple = 1/8 →
  pitcher2_orange = 1/5 →
  pitcher2_apple = 1/10 →
  (pitcher_capacity * pitcher1_orange + pitcher_capacity * pitcher2_orange) / (2 * pitcher_capacity) = 9/40 := by
  sorry

#check orange_juice_fraction

end NUMINAMATH_CALUDE_orange_juice_fraction_l3516_351680


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3516_351613

theorem contrapositive_equivalence :
  (∀ x : ℝ, x < 3 → x^2 ≤ 9) ↔ (∀ x : ℝ, x^2 > 9 → x ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3516_351613


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3516_351699

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

theorem inequality_solution_set (x : ℝ) :
  (0 < x ∧ f (Real.log x) + f (Real.log (1/x)) < 2 * f 1) ↔ (1/Real.exp 1 < x ∧ x < Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3516_351699


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3516_351652

/-- Given two lines in the form ax + by + c = 0, this function returns true if they are parallel -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1

/-- Given a line ax + by + c = 0 and a point (x0, y0), this function returns true if the point lies on the line -/
def point_on_line (a b c x0 y0 : ℝ) : Prop :=
  a * x0 + b * y0 + c = 0

theorem line_through_point_parallel_to_line :
  are_parallel 2 (-3) 12 2 (-3) 4 ∧
  point_on_line 2 (-3) 12 (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3516_351652


namespace NUMINAMATH_CALUDE_yuna_position_l3516_351620

/-- Given Eunji's position and Yuna's relative position after Eunji, 
    calculate Yuna's absolute position on the train. -/
theorem yuna_position (eunji_pos yuna_after : ℕ) : 
  eunji_pos = 100 → yuna_after = 11 → eunji_pos + yuna_after = 111 := by
  sorry

#check yuna_position

end NUMINAMATH_CALUDE_yuna_position_l3516_351620


namespace NUMINAMATH_CALUDE_platform_length_l3516_351649

/-- Given a train of length 200 meters that crosses a platform in 50 seconds
    and a signal pole in 42 seconds, the length of the platform is 38 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) 
  (h1 : train_length = 200)
  (h2 : time_platform = 50)
  (h3 : time_pole = 42) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 38 := by
  sorry


end NUMINAMATH_CALUDE_platform_length_l3516_351649


namespace NUMINAMATH_CALUDE_range_of_f_l3516_351630

-- Define the function
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of the function
theorem range_of_f :
  ∀ y : ℝ, (y ≥ -8) ↔ ∃ x : ℝ, f x = y :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l3516_351630


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3516_351676

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3516_351676


namespace NUMINAMATH_CALUDE_fly_probabilities_l3516_351693

def fly_walk (n m : ℕ) : ℚ := (Nat.choose (n + m) n : ℚ) / (2 ^ (n + m))

def fly_walk_through (n₁ m₁ n₂ m₂ : ℕ) : ℚ :=
  (Nat.choose (n₁ + m₁) n₁ : ℚ) * (Nat.choose (n₂ + m₂) n₂) / (2 ^ (n₁ + m₁ + n₂ + m₂))

def fly_walk_circle : ℚ :=
  (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + Nat.choose 9 4 * Nat.choose 9 4 : ℚ) / 2^18

theorem fly_probabilities :
  (fly_walk 8 10 = (Nat.choose 18 8 : ℚ) / 2^18) ∧
  (fly_walk_through 5 6 2 4 = ((Nat.choose 11 5 : ℚ) * Nat.choose 6 2) / 2^18) ∧
  (fly_walk_circle = (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + Nat.choose 9 4 * Nat.choose 9 4 : ℚ) / 2^18) := by
  sorry

end NUMINAMATH_CALUDE_fly_probabilities_l3516_351693


namespace NUMINAMATH_CALUDE_units_digit_of_8429_pow_1246_l3516_351623

theorem units_digit_of_8429_pow_1246 :
  (8429^1246) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_8429_pow_1246_l3516_351623


namespace NUMINAMATH_CALUDE_min_marbles_needed_l3516_351663

/-- The minimum number of additional marbles needed --/
def min_additional_marbles (n : ℕ) (current : ℕ) : ℕ :=
  (n * (n + 1)) / 2 - current

/-- Theorem stating the minimum number of additional marbles needed --/
theorem min_marbles_needed (n : ℕ) (current : ℕ) 
  (h_n : n = 12) (h_current : current = 40) : 
  min_additional_marbles n current = 38 := by
  sorry

#eval min_additional_marbles 12 40

end NUMINAMATH_CALUDE_min_marbles_needed_l3516_351663


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3516_351635

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 6 * x + c = 0) →
  a + c = 12 →
  a < c →
  (a, c) = (6 - 3 * Real.sqrt 3, 6 + 3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3516_351635


namespace NUMINAMATH_CALUDE_correct_number_of_pupils_l3516_351681

/-- The number of pupils in a class where an error in one pupil's marks
    caused the class average to increase by half. -/
def number_of_pupils : ℕ :=
  let mark_increase : ℕ := 85 - 45
  let average_increase : ℚ := 1/2
  (2 * mark_increase : ℕ)

theorem correct_number_of_pupils :
  number_of_pupils = 80 :=
sorry

end NUMINAMATH_CALUDE_correct_number_of_pupils_l3516_351681


namespace NUMINAMATH_CALUDE_smallest_nat_greater_than_12_l3516_351633

theorem smallest_nat_greater_than_12 :
  ∀ n : ℕ, n > 12 → n ≥ 13 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_nat_greater_than_12_l3516_351633


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l3516_351659

-- Define the propositions p and q
def p (x : ℝ) : Prop := 4 / (x - 1) ≤ -1
def q (x a : ℝ) : Prop := x^2 - x < a^2 - a

-- Define the condition that ¬q is sufficient but not necessary for ¬p
def sufficient_not_necessary (a : ℝ) : Prop :=
  ∀ x, ¬(q x a) → ¬(p x) ∧ ∃ y, ¬(p y) ∧ q y a

-- Define the range of a
def range_of_a : Set ℝ := {a | a ∈ [0, 1] ∧ a ≠ 1/2}

-- State the theorem
theorem range_of_a_theorem :
  ∀ a, sufficient_not_necessary a ↔ a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l3516_351659


namespace NUMINAMATH_CALUDE_wall_area_l3516_351677

theorem wall_area (small_tile_area small_tile_proportion total_wall_area : ℝ) 
  (h1 : small_tile_proportion = 1 / 2)
  (h2 : small_tile_area = 80)
  (h3 : small_tile_area = small_tile_proportion * total_wall_area) :
  total_wall_area = 160 := by
  sorry

end NUMINAMATH_CALUDE_wall_area_l3516_351677


namespace NUMINAMATH_CALUDE_wheel_distance_l3516_351697

/-- Proves that a wheel rotating 20 times per minute and moving 35 cm per rotation will travel 420 meters in one hour -/
theorem wheel_distance (rotations_per_minute : ℕ) (distance_per_rotation_cm : ℕ) :
  rotations_per_minute = 20 →
  distance_per_rotation_cm = 35 →
  (rotations_per_minute * 60 * distance_per_rotation_cm : ℚ) / 100 = 420 := by
  sorry

#check wheel_distance

end NUMINAMATH_CALUDE_wheel_distance_l3516_351697


namespace NUMINAMATH_CALUDE_kite_long_diagonal_angle_in_circular_arrangement_l3516_351609

/-- Represents a symmetrical kite in a circular arrangement -/
structure Kite where
  long_diagonal_angle : ℝ
  short_diagonal_angle : ℝ

/-- Represents a circular arrangement of kites -/
structure CircularArrangement where
  num_kites : ℕ
  kites : Fin num_kites → Kite
  covers_circle : Bool
  long_diagonals_meet_center : Bool

/-- The theorem stating the long diagonal angle in the specific arrangement -/
theorem kite_long_diagonal_angle_in_circular_arrangement 
  (arr : CircularArrangement) 
  (h1 : arr.num_kites = 10) 
  (h2 : arr.covers_circle = true) 
  (h3 : arr.long_diagonals_meet_center = true) :
  ∀ i, (arr.kites i).long_diagonal_angle = 162 :=
sorry

end NUMINAMATH_CALUDE_kite_long_diagonal_angle_in_circular_arrangement_l3516_351609


namespace NUMINAMATH_CALUDE_wine_bottle_prices_l3516_351678

-- Define the prices as real numbers
variable (A B C X Y : ℝ)

-- Define the conditions from the problem
def condition1 : Prop := A + X = 3.50
def condition2 : Prop := B + X = 4.20
def condition3 : Prop := C + Y = 6.10
def condition4 : Prop := A = X + 1.50
def condition5 : Prop := B = X + 2.20
def condition6 : Prop := C = Y + 3.40

-- State the theorem to be proved
theorem wine_bottle_prices 
  (h1 : condition1 A X)
  (h2 : condition2 B X)
  (h3 : condition3 C Y)
  (h4 : condition4 A X)
  (h5 : condition5 B X)
  (h6 : condition6 C Y) :
  A = 2.50 ∧ B = 3.20 ∧ C = 4.75 ∧ X = 1.00 ∧ Y = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_wine_bottle_prices_l3516_351678


namespace NUMINAMATH_CALUDE_new_total_bill_l3516_351610

def original_order_cost : ℝ := 25
def tomatoes_old_price : ℝ := 0.99
def tomatoes_new_price : ℝ := 2.20
def lettuce_old_price : ℝ := 1.00
def lettuce_new_price : ℝ := 1.75
def celery_old_price : ℝ := 1.96
def celery_new_price : ℝ := 2.00
def delivery_tip_cost : ℝ := 8.00

theorem new_total_bill :
  let price_increase := (tomatoes_new_price - tomatoes_old_price) +
                        (lettuce_new_price - lettuce_old_price) +
                        (celery_new_price - celery_old_price)
  let new_food_cost := original_order_cost + price_increase
  let total_bill := new_food_cost + delivery_tip_cost
  total_bill = 35 := by
  sorry

end NUMINAMATH_CALUDE_new_total_bill_l3516_351610


namespace NUMINAMATH_CALUDE_number_of_paths_l3516_351657

def grid_width : ℕ := 6
def grid_height : ℕ := 5
def path_length : ℕ := 8
def steps_right : ℕ := grid_width - 1
def steps_up : ℕ := grid_height - 1

theorem number_of_paths : 
  Nat.choose path_length steps_up = Nat.choose path_length (path_length - steps_right) := by
  sorry

end NUMINAMATH_CALUDE_number_of_paths_l3516_351657


namespace NUMINAMATH_CALUDE_gasoline_distribution_impossibility_l3516_351674

theorem gasoline_distribution_impossibility :
  ¬∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x + y + z = 50 ∧
  x = y + 10 ∧
  z + 26 = y :=
by sorry

end NUMINAMATH_CALUDE_gasoline_distribution_impossibility_l3516_351674


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3516_351661

/-- The equation of the conic section --/
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt ((x-6)^2 + (y-4)^2) = 14

/-- The two focal points of the conic section --/
def focal_point1 : ℝ × ℝ := (0, -2)
def focal_point2 : ℝ × ℝ := (6, 4)

/-- Theorem stating that the given equation describes an ellipse --/
theorem conic_is_ellipse : ∃ (a b : ℝ) (center : ℝ × ℝ), 
  a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : ℝ), conic_equation x y ↔ 
    ((x - center.1) / a)^2 + ((y - center.2) / b)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3516_351661


namespace NUMINAMATH_CALUDE_edward_initial_money_l3516_351614

def books_cost : ℕ := 6
def pens_cost : ℕ := 16
def notebook_cost : ℕ := 5
def pencil_case_cost : ℕ := 3
def money_left : ℕ := 19

theorem edward_initial_money :
  books_cost + pens_cost + notebook_cost + pencil_case_cost + money_left = 49 := by
  sorry

end NUMINAMATH_CALUDE_edward_initial_money_l3516_351614


namespace NUMINAMATH_CALUDE_identify_alkali_metal_l3516_351656

/-- Represents an alkali metal with its atomic mass -/
structure AlkaliMetal where
  atomic_mass : ℝ

/-- Represents a mixture of an alkali metal and its oxide -/
structure Mixture (R : AlkaliMetal) where
  initial_mass : ℝ
  final_mass : ℝ

/-- Theorem: If a mixture of alkali metal R and its oxide R₂O weighs 10.8 grams,
    and after reaction with water and drying, the resulting solid weighs 16 grams,
    then the atomic mass of R is 23. -/
theorem identify_alkali_metal (R : AlkaliMetal) (mix : Mixture R) :
  mix.initial_mass = 10.8 ∧ mix.final_mass = 16 → R.atomic_mass = 23 := by
  sorry

#check identify_alkali_metal

end NUMINAMATH_CALUDE_identify_alkali_metal_l3516_351656


namespace NUMINAMATH_CALUDE_solve_equation_l3516_351694

theorem solve_equation : 
  ∃ x : ℚ, 64 + 5 * x / (180 / 3) = 65 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3516_351694


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3516_351641

/-- 
For a quadratic equation x^2 - x + n = 0, if it has two equal real roots,
then n = 1/4.
-/
theorem equal_roots_quadratic (n : ℝ) : 
  (∃ x : ℝ, x^2 - x + n = 0 ∧ (∀ y : ℝ, y^2 - y + n = 0 → y = x)) → n = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3516_351641


namespace NUMINAMATH_CALUDE_events_independent_prob_A_or_B_l3516_351631

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The set of all ball numbers -/
def ball_numbers : Finset ℕ := Finset.range total_balls

/-- Event A: selecting a ball with an odd number -/
def event_A : Finset ℕ := ball_numbers.filter (λ n => n % 2 = 1)

/-- Event B: selecting a ball with a number that is a multiple of 3 -/
def event_B : Finset ℕ := ball_numbers.filter (λ n => n % 3 = 0)

/-- The probability of an event occurring -/
def prob (event : Finset ℕ) : ℚ := (event.card : ℚ) / total_balls

/-- The intersection of events A and B -/
def event_AB : Finset ℕ := event_A ∩ event_B

/-- Theorem: Events A and B are independent -/
theorem events_independent : prob event_AB = prob event_A * prob event_B := by sorry

/-- Theorem: The probability of A or B occurring is 5/8 -/
theorem prob_A_or_B : prob (event_A ∪ event_B) = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_events_independent_prob_A_or_B_l3516_351631


namespace NUMINAMATH_CALUDE_race_fourth_part_length_l3516_351651

/-- Given a 4-part race with specified lengths for the first three parts,
    calculate the length of the fourth part. -/
theorem race_fourth_part_length 
  (total_length : ℝ) 
  (first_part : ℝ) 
  (second_part : ℝ) 
  (third_part : ℝ) 
  (h1 : total_length = 74.5)
  (h2 : first_part = 15.5)
  (h3 : second_part = 21.5)
  (h4 : third_part = 21.5) :
  total_length - (first_part + second_part + third_part) = 16 := by
sorry

end NUMINAMATH_CALUDE_race_fourth_part_length_l3516_351651


namespace NUMINAMATH_CALUDE_raise_time_on_hoop_l3516_351668

/-- Time required to raise an object by a certain distance when wrapped around a rotating hoop -/
theorem raise_time_on_hoop (r : ℝ) (rpm : ℝ) (distance : ℝ) : 
  r > 0 → rpm > 0 → distance > 0 → 
  (distance / (2 * π * r)) * (60 / rpm) = 15 / π := by
  sorry

end NUMINAMATH_CALUDE_raise_time_on_hoop_l3516_351668


namespace NUMINAMATH_CALUDE_multiply_power_rule_l3516_351622

theorem multiply_power_rule (x : ℝ) : x * x^4 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_power_rule_l3516_351622


namespace NUMINAMATH_CALUDE_parabola_intersection_l3516_351640

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 4
def parabola2 (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 8

theorem parabola_intersection :
  ∀ x y : ℝ, parabola1 x = parabola2 x ∧ y = parabola1 x ↔ (x = 3 ∧ y = 20) ∨ (x = 4 ∧ y = 32) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3516_351640


namespace NUMINAMATH_CALUDE_cubic_roots_inequality_l3516_351618

theorem cubic_roots_inequality (a b c r s t : ℝ) :
  (∀ x, x^3 + a*x^2 + b*x + c = 0 ↔ x = r ∨ x = s ∨ x = t) →
  r ≥ s →
  s ≥ t →
  (a^2 - 3*b ≥ 0) ∧ (Real.sqrt (a^2 - 3*b) ≤ r - t) := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_inequality_l3516_351618


namespace NUMINAMATH_CALUDE_largest_divisor_of_even_squares_sum_l3516_351655

theorem largest_divisor_of_even_squares_sum (m n : ℕ) : 
  Even m → Even n → n < m → (∀ k : ℕ, k > 4 → ∃ m' n' : ℕ, 
    Even m' ∧ Even n' ∧ n' < m' ∧ ¬(k ∣ m'^2 + n'^2)) ∧ 
  (∀ m' n' : ℕ, Even m' → Even n' → n' < m' → (4 ∣ m'^2 + n'^2)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_even_squares_sum_l3516_351655


namespace NUMINAMATH_CALUDE_round_trip_percentage_l3516_351688

theorem round_trip_percentage (total_passengers : ℝ) 
  (h1 : total_passengers > 0) :
  let round_trip_with_car := 0.15 * total_passengers
  let round_trip_without_car := 0.6 * (round_trip_with_car / 0.4)
  (round_trip_with_car + round_trip_without_car) / total_passengers = 0.375 := by
sorry

end NUMINAMATH_CALUDE_round_trip_percentage_l3516_351688


namespace NUMINAMATH_CALUDE_g_in_terms_of_f_l3516_351636

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_in_terms_of_f : ∀ x : ℝ, g x = f (6 - x) := by sorry

end NUMINAMATH_CALUDE_g_in_terms_of_f_l3516_351636


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3516_351685

theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_eq : a^2 - c^2 + b^2 = -Real.sqrt 3 * a * b) : 
  Real.cos (Real.pi / 6) = (a^2 + b^2 - c^2) / (2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3516_351685


namespace NUMINAMATH_CALUDE_statement_a_statement_b_statement_c_incorrect_statement_d_main_theorem_l3516_351690

-- Statement A
theorem statement_a (x y : ℝ) (hx : x > 0) (hy : y > 0) : x / y + y / x ≥ 2 := by sorry

-- Statement B
theorem statement_b (x : ℝ) : (x^2 + 2) / Real.sqrt (x^2 + 1) ≥ 2 := by sorry

-- Statement C (incorrect)
theorem statement_c_incorrect : ∃ x : ℝ, x > 0 ∧ x < 1 ∧ Real.log x / Real.log 10 + Real.log 10 / Real.log x < 2 := by sorry

-- Statement D
theorem statement_d (a : ℝ) (ha : a > 0) : (1 + a) * (1 + 1 / a) ≥ 4 := by sorry

-- Main theorem
theorem main_theorem : 
  (∀ x y : ℝ, x > 0 → y > 0 → x / y + y / x ≥ 2) ∧
  (∀ x : ℝ, (x^2 + 2) / Real.sqrt (x^2 + 1) ≥ 2) ∧
  (∃ x : ℝ, x > 0 ∧ x < 1 ∧ Real.log x / Real.log 10 + Real.log 10 / Real.log x < 2) ∧
  (∀ a : ℝ, a > 0 → (1 + a) * (1 + 1 / a) ≥ 4) := by sorry

end NUMINAMATH_CALUDE_statement_a_statement_b_statement_c_incorrect_statement_d_main_theorem_l3516_351690


namespace NUMINAMATH_CALUDE_determinant_calculation_l3516_351672

def determinant (a b c d : Int) : Int :=
  a * d - b * c

theorem determinant_calculation : determinant 2 3 (-6) (-5) = 8 := by
  sorry

end NUMINAMATH_CALUDE_determinant_calculation_l3516_351672


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3516_351601

theorem fraction_equivalence (x y : ℝ) (h1 : y ≠ 0) (h2 : x + 2*y ≠ 0) :
  (x + y) / (x + 2*y) = y / (2*y) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3516_351601


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l3516_351626

theorem complex_magnitude_one (z : ℂ) (p : ℕ) (h : 11 * z^10 + 10 * Complex.I * z^p + 10 * Complex.I * z - 11 = 0) : 
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l3516_351626


namespace NUMINAMATH_CALUDE_equilateral_triangle_circle_radius_l3516_351625

theorem equilateral_triangle_circle_radius (r : ℝ) 
  (h : r > 0) : 
  (3 * (r * Real.sqrt 3) = π * r^2) → 
  r = (3 * Real.sqrt 3) / π := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circle_radius_l3516_351625


namespace NUMINAMATH_CALUDE_determinant_transformation_l3516_351637

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = -3 →
  Matrix.det !![p, 5*p + 2*q; r, 5*r + 2*s] = -6 := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l3516_351637


namespace NUMINAMATH_CALUDE_min_value_theorem_l3516_351675

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 5) :
  9/x + 4/y + 25/z ≥ 20 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' + y' + z' = 5 ∧ 9/x' + 4/y' + 25/z' = 20 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3516_351675


namespace NUMINAMATH_CALUDE_debby_water_bottles_l3516_351644

/-- The number of water bottles Debby drank in one day -/
def bottles_drank : ℕ := 144

/-- The number of water bottles Debby has left -/
def bottles_left : ℕ := 157

/-- The initial number of water bottles Debby bought -/
def initial_bottles : ℕ := bottles_drank + bottles_left

theorem debby_water_bottles : initial_bottles = 301 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l3516_351644


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l3516_351696

/-- Given information about children's emotions and gender distribution -/
structure ChildrenData where
  total : Nat
  happy : Nat
  sad : Nat
  neither : Nat
  boys : Nat
  girls : Nat
  happy_boys : Nat
  sad_girls : Nat

/-- Theorem stating the number of boys who are neither happy nor sad -/
theorem boys_neither_happy_nor_sad (data : ChildrenData)
  (h1 : data.total = 60)
  (h2 : data.happy = 30)
  (h3 : data.sad = 10)
  (h4 : data.neither = 20)
  (h5 : data.boys = 19)
  (h6 : data.girls = 41)
  (h7 : data.happy_boys = 6)
  (h8 : data.sad_girls = 4)
  (h9 : data.total = data.happy + data.sad + data.neither)
  (h10 : data.total = data.boys + data.girls) :
  data.boys - data.happy_boys - (data.sad - data.sad_girls) = 7 := by
  sorry


end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l3516_351696


namespace NUMINAMATH_CALUDE_phone_repair_amount_is_10_l3516_351605

/-- The amount earned from repairing a phone -/
def phone_repair_amount : ℝ := sorry

/-- The amount earned from repairing a laptop -/
def laptop_repair_amount : ℝ := 20

/-- The total number of phones repaired -/
def total_phones : ℕ := 3 + 5

/-- The total number of laptops repaired -/
def total_laptops : ℕ := 2 + 4

/-- The total amount earned -/
def total_earned : ℝ := 200

theorem phone_repair_amount_is_10 :
  phone_repair_amount * total_phones + laptop_repair_amount * total_laptops = total_earned ∧
  phone_repair_amount = 10 := by sorry

end NUMINAMATH_CALUDE_phone_repair_amount_is_10_l3516_351605


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3516_351653

theorem decimal_to_fraction :
  (3.68 : ℚ) = 92 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3516_351653


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l3516_351616

theorem simplify_nested_expression (x : ℝ) : 1 - (1 - (1 + (1 - (1 + (1 - x))))) = x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l3516_351616


namespace NUMINAMATH_CALUDE_mixture_composition_l3516_351658

theorem mixture_composition (alcohol_volume : ℚ) (water_volume : ℚ) 
  (h1 : alcohol_volume = 3/5)
  (h2 : alcohol_volume / water_volume = 3/4) :
  water_volume = 4/5 := by
sorry

end NUMINAMATH_CALUDE_mixture_composition_l3516_351658


namespace NUMINAMATH_CALUDE_optimal_group_division_l3516_351679

theorem optimal_group_division (total_members : ℕ) (large_group_size : ℕ) (small_group_size : ℕ) 
  (h1 : total_members = 90)
  (h2 : large_group_size = 7)
  (h3 : small_group_size = 3) :
  ∃ (large_groups small_groups : ℕ),
    large_groups * large_group_size + small_groups * small_group_size = total_members ∧
    large_groups = 12 ∧
    ∀ (lg sg : ℕ), lg * large_group_size + sg * small_group_size = total_members → lg ≤ large_groups :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_group_division_l3516_351679


namespace NUMINAMATH_CALUDE_sphere_radius_is_six_l3516_351608

/-- A truncated cone with horizontal bases of radii 12 and 3, and a sphere tangent to its top, bottom, and lateral surface. -/
structure TruncatedConeWithSphere where
  lower_radius : ℝ
  upper_radius : ℝ
  sphere_radius : ℝ
  lower_radius_eq : lower_radius = 12
  upper_radius_eq : upper_radius = 3
  sphere_tangent : True  -- We can't directly express tangency in this simple structure

/-- The radius of the sphere in the TruncatedConeWithSphere is 6. -/
theorem sphere_radius_is_six (cone : TruncatedConeWithSphere) : cone.sphere_radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_six_l3516_351608


namespace NUMINAMATH_CALUDE_odd_prime_congruence_l3516_351617

theorem odd_prime_congruence (p : Nat) (c : Int) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ a : Int, (a^((p+1)/2) + (a+c)^((p+1)/2)) % p = c % p := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_congruence_l3516_351617


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l3516_351695

-- Define the complex number
def z : ℂ := Complex.I * (2 - Complex.I)

-- Theorem statement
theorem point_in_first_quadrant :
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l3516_351695
