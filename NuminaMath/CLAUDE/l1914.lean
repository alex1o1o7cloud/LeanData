import Mathlib

namespace NUMINAMATH_CALUDE_two_bedroom_units_l1914_191428

theorem two_bedroom_units (total_units : ℕ) (cost_one_bedroom : ℕ) (cost_two_bedroom : ℕ) (total_cost : ℕ) :
  total_units = 12 →
  cost_one_bedroom = 360 →
  cost_two_bedroom = 450 →
  total_cost = 4950 →
  ∃ (one_bedroom_units two_bedroom_units : ℕ),
    one_bedroom_units + two_bedroom_units = total_units ∧
    cost_one_bedroom * one_bedroom_units + cost_two_bedroom * two_bedroom_units = total_cost ∧
    two_bedroom_units = 7 :=
by sorry

end NUMINAMATH_CALUDE_two_bedroom_units_l1914_191428


namespace NUMINAMATH_CALUDE_art_gallery_display_ratio_l1914_191451

/-- Theorem about the ratio of displayed art pieces to total pieces in a gallery -/
theorem art_gallery_display_ratio 
  (total_pieces : ℕ)
  (sculptures_not_displayed : ℕ)
  (h_total : total_pieces = 3150)
  (h_sculptures_not_displayed : sculptures_not_displayed = 1400)
  (h_display_ratio : ∀ d : ℕ, d > 0 → (d : ℚ) / 6 = (sculptures_not_displayed : ℚ))
  (h_not_display_ratio : ∀ n : ℕ, n > 0 → (n : ℚ) / 3 = ((total_pieces - sculptures_not_displayed) : ℚ)) :
  (total_pieces : ℚ) / 3 = (total_pieces - sculptures_not_displayed * 3 / 2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_art_gallery_display_ratio_l1914_191451


namespace NUMINAMATH_CALUDE_bridget_apples_l1914_191447

theorem bridget_apples (x : ℕ) : 
  (x : ℚ) / 3 + 4 + 6 = x → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_l1914_191447


namespace NUMINAMATH_CALUDE_rhombus_count_in_triangle_l1914_191414

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a rhombus composed of smaller triangles -/
structure Rhombus where
  smallTrianglesCount : ℕ

/-- Counts the number of rhombuses in a given equilateral triangle -/
def countRhombuses (triangle : EquilateralTriangle) (rhombusSize : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem rhombus_count_in_triangle :
  let largeTriangle := EquilateralTriangle.mk 10
  let rhombusType := Rhombus.mk 8
  countRhombuses largeTriangle rhombusType.smallTrianglesCount = 84 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_count_in_triangle_l1914_191414


namespace NUMINAMATH_CALUDE_quartic_roots_sum_product_l1914_191473

theorem quartic_roots_sum_product (a b : ℝ) : 
  a^4 - 6*a - 2 = 0 → b^4 - 6*b - 2 = 0 → a * b + a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quartic_roots_sum_product_l1914_191473


namespace NUMINAMATH_CALUDE_remainder_777_pow_777_mod_13_l1914_191480

theorem remainder_777_pow_777_mod_13 : 777^777 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_777_pow_777_mod_13_l1914_191480


namespace NUMINAMATH_CALUDE_negation_of_all_divisible_by_two_are_even_l1914_191499

theorem negation_of_all_divisible_by_two_are_even :
  (¬ ∀ n : ℕ, 2 ∣ n → Even n) ↔ (∃ n : ℕ, 2 ∣ n ∧ ¬Even n) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_divisible_by_two_are_even_l1914_191499


namespace NUMINAMATH_CALUDE_flower_count_l1914_191487

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of different candles --/
def num_candles : ℕ := 4

/-- The number of candles to choose --/
def candles_to_choose : ℕ := 2

/-- The number of flowers to choose --/
def flowers_to_choose : ℕ := 8

/-- The total number of candle + flower groupings --/
def total_groupings : ℕ := 54

theorem flower_count :
  ∃ (F : ℕ), 
    F > 0 ∧
    choose num_candles candles_to_choose * choose F flowers_to_choose = total_groupings ∧
    F = 9 :=
by sorry

end NUMINAMATH_CALUDE_flower_count_l1914_191487


namespace NUMINAMATH_CALUDE_a_grazing_months_l1914_191426

/-- Represents the number of months 'a' put his oxen for grazing -/
def a_months : ℕ := sorry

/-- Represents the number of oxen 'a' put for grazing -/
def a_oxen : ℕ := 10

/-- Represents the number of oxen 'b' put for grazing -/
def b_oxen : ℕ := 12

/-- Represents the number of months 'b' put his oxen for grazing -/
def b_months : ℕ := 5

/-- Represents the number of oxen 'c' put for grazing -/
def c_oxen : ℕ := 15

/-- Represents the number of months 'c' put his oxen for grazing -/
def c_months : ℕ := 3

/-- Represents the total rent of the pasture in Rs. -/
def total_rent : ℕ := 105

/-- Represents 'c's share of the rent in Rs. -/
def c_share : ℕ := 27

/-- Theorem stating that 'a' put his oxen for grazing for 7 months -/
theorem a_grazing_months : a_months = 7 := by sorry

end NUMINAMATH_CALUDE_a_grazing_months_l1914_191426


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1914_191475

/-- The quadratic function f(x) = -(x+1)^2 - 8 has vertex coordinates (-1, -8) -/
theorem quadratic_vertex (x : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ -(x + 1)^2 - 8
  (∀ x, f x ≤ f (-1)) ∧ f (-1) = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1914_191475


namespace NUMINAMATH_CALUDE_xy_equals_one_l1914_191437

theorem xy_equals_one (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 25) (h4 : x^2 * y^3 + y^2 * x^3 = 25) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_one_l1914_191437


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1914_191453

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 2*X^2 + 1 = (X^2 - 2*X + 4) * q + (-4*X - 7) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1914_191453


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_l1914_191458

theorem right_triangle_acute_angle (α : ℝ) : 
  α > 0 ∧ α < 90 → -- α is an acute angle
  α + (α - 10) + 90 = 180 → -- sum of angles in the triangle
  α = 50 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_l1914_191458


namespace NUMINAMATH_CALUDE_calculation_proof_l1914_191405

theorem calculation_proof : 10 - 9 * 8 / 4 + 7 - 6 * 5 + 3 - 2 * 1 = -30 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1914_191405


namespace NUMINAMATH_CALUDE_marble_bag_count_l1914_191465

/-- Given a bag of marbles with red, blue, and green marbles in the ratio 2:4:6,
    and 36 green marbles, prove that the total number of marbles is 72. -/
theorem marble_bag_count (red blue green total : ℕ) : 
  red + blue + green = total →
  2 * blue = 4 * red →
  3 * blue = 2 * green →
  green = 36 →
  total = 72 := by
sorry

end NUMINAMATH_CALUDE_marble_bag_count_l1914_191465


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l1914_191407

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_proof (a b c : ℝ) :
  (∀ x, f a b c x = 0 ↔ x = -2 ∨ x = 1) →  -- Roots condition
  (∃ m, ∀ x, f a b c x ≤ m) →              -- Maximum value condition
  (∀ x, f a b c x = -x^2 - x + 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l1914_191407


namespace NUMINAMATH_CALUDE_no_real_solutions_l1914_191461

theorem no_real_solutions : ¬∃ x : ℝ, (2*x^2 - 6*x + 5)^2 + 1 = -|x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1914_191461


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_events_not_complementary_l1914_191445

-- Define the sample space for a standard six-sided die
def DieOutcome : Type := Fin 6

-- Define the event "the number is odd"
def isOdd (n : DieOutcome) : Prop := n.val % 2 = 1

-- Define the event "the number is greater than 5"
def isGreaterThan5 (n : DieOutcome) : Prop := n.val = 6

-- Theorem stating that the events are mutually exclusive
theorem events_mutually_exclusive :
  ∀ (n : DieOutcome), ¬(isOdd n ∧ isGreaterThan5 n) :=
sorry

-- Theorem stating that the events are not complementary
theorem events_not_complementary :
  ¬(∀ (n : DieOutcome), isOdd n ↔ ¬isGreaterThan5 n) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_events_not_complementary_l1914_191445


namespace NUMINAMATH_CALUDE_billy_sam_money_difference_l1914_191479

theorem billy_sam_money_difference (sam_money : ℕ) (total_money : ℕ) : 
  sam_money = 75 →
  total_money = 200 →
  let billy_money := total_money - sam_money
  billy_money < 2 * sam_money →
  2 * sam_money - billy_money = 25 := by
  sorry

end NUMINAMATH_CALUDE_billy_sam_money_difference_l1914_191479


namespace NUMINAMATH_CALUDE_betty_bracelets_l1914_191452

/-- Given that Betty has 88.0 pink flower stones and each bracelet requires 11 stones,
    prove that the number of bracelets she can make is 8. -/
theorem betty_bracelets :
  let total_stones : ℝ := 88.0
  let stones_per_bracelet : ℕ := 11
  (total_stones / stones_per_bracelet : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_betty_bracelets_l1914_191452


namespace NUMINAMATH_CALUDE_jasmine_solution_concentration_l1914_191494

theorem jasmine_solution_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_jasmine : ℝ) 
  (added_water : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 100)
  (h2 : initial_concentration = 0.1)
  (h3 : added_jasmine = 5)
  (h4 : added_water = 67.5)
  (h5 : final_concentration = 0.08695652173913043) :
  let initial_jasmine := initial_volume * initial_concentration
  let total_jasmine := initial_jasmine + added_jasmine
  let final_volume := initial_volume + added_jasmine + added_water
  total_jasmine / final_volume = final_concentration :=
sorry

end NUMINAMATH_CALUDE_jasmine_solution_concentration_l1914_191494


namespace NUMINAMATH_CALUDE_cookie_difference_l1914_191482

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_choc : ℕ := 19

/-- The number of raisin cookies Helen baked this morning -/
def morning_raisin : ℕ := 231

/-- The number of chocolate chip cookies Helen baked this morning -/
def morning_choc : ℕ := 237

/-- The total number of chocolate chip cookies Helen baked -/
def total_choc : ℕ := yesterday_choc + morning_choc

/-- The difference between chocolate chip cookies and raisin cookies -/
theorem cookie_difference : total_choc - morning_raisin = 25 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l1914_191482


namespace NUMINAMATH_CALUDE_students_only_english_l1914_191444

theorem students_only_english (total : ℕ) (both : ℕ) (german : ℕ) (h1 : total = 52) (h2 : both = 12) (h3 : german = 22) (h4 : total ≥ german) : total - german + both = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_only_english_l1914_191444


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l1914_191496

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startingFee : ℝ
  ratePerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.startingFee + tf.ratePerMile * distance

theorem taxi_fare_calculation (tf : TaxiFare) 
  (h1 : tf.startingFee = 20)
  (h2 : totalFare tf 80 = 160)
  : totalFare tf 120 = 230 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l1914_191496


namespace NUMINAMATH_CALUDE_no_solution_implies_a_range_l1914_191460

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem no_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(otimes (x - a) (x + 1) ≥ 1)) → -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_range_l1914_191460


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1914_191462

theorem diophantine_equation_solution (x y : ℤ) :
  y^2 = x^3 + 16 → x = 0 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1914_191462


namespace NUMINAMATH_CALUDE_bertha_family_females_without_daughters_l1914_191438

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  total_descendants : ℕ
  each_daughter_has_equal_children : Bool
  no_great_grandchildren : Bool

/-- Calculates the number of females with no daughters in Bertha's family -/
def females_without_daughters (family : BerthaFamily) : ℕ :=
  family.total_descendants - family.daughters

/-- Theorem stating that the number of females with no daughters in Bertha's family is 32 -/
theorem bertha_family_females_without_daughters :
  ∀ (family : BerthaFamily),
    family.daughters = 8 ∧
    family.total_descendants = 40 ∧
    family.each_daughter_has_equal_children = true ∧
    family.no_great_grandchildren = true →
    females_without_daughters family = 32 := by
  sorry

end NUMINAMATH_CALUDE_bertha_family_females_without_daughters_l1914_191438


namespace NUMINAMATH_CALUDE_regular_polygon_on_grid_l1914_191489

/-- A grid in the plane formed by two families of equally spaced parallel lines -/
structure Grid where
  -- We don't need to define the internal structure of the grid

/-- A point in the plane -/
structure Point where
  -- We don't need to define the internal structure of the point

/-- A regular convex n-gon -/
structure RegularPolygon where
  vertices : List Point
  n : Nat
  is_regular : Bool
  is_convex : Bool

/-- Predicate to check if a point is on the grid -/
def Point.on_grid (p : Point) (g : Grid) : Prop := sorry

/-- Predicate to check if all vertices of a polygon are on the grid -/
def RegularPolygon.vertices_on_grid (p : RegularPolygon) (g : Grid) : Prop :=
  ∀ v ∈ p.vertices, v.on_grid g

/-- The main theorem -/
theorem regular_polygon_on_grid (g : Grid) (p : RegularPolygon) :
  p.n ≥ 3 ∧ p.is_regular ∧ p.is_convex ∧ p.vertices_on_grid g → p.n = 3 ∨ p.n = 4 ∨ p.n = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_on_grid_l1914_191489


namespace NUMINAMATH_CALUDE_area_of_overlapping_squares_l1914_191423

/-- Represents a square in a 2D plane -/
structure Square where
  sideLength : ℝ
  center : ℝ × ℝ

/-- Calculates the area of overlap between two squares -/
def overlapArea (s1 s2 : Square) : ℝ :=
  sorry

/-- Calculates the total area covered by two squares -/
def totalCoveredArea (s1 s2 : Square) : ℝ :=
  sorry

theorem area_of_overlapping_squares :
  let s1 : Square := { sideLength := 20, center := (0, 0) }
  let s2 : Square := { sideLength := 20, center := (10, 0) }
  totalCoveredArea s1 s2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_area_of_overlapping_squares_l1914_191423


namespace NUMINAMATH_CALUDE_power_multiplication_equality_l1914_191459

theorem power_multiplication_equality : (-0.25)^2023 * 4^2024 = -4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_equality_l1914_191459


namespace NUMINAMATH_CALUDE_right_triangle_x_values_l1914_191466

/-- A right-angled triangle ABC with vectors AB and AC -/
structure RightTriangle where
  AB : ℝ × ℝ
  AC : ℝ × ℝ
  is_right_angled : Bool

/-- The possible x values for the right-angled triangle -/
def possible_x_values : Set ℝ := {3/2, 4}

/-- Theorem: In a right-angled triangle ABC with AB = (2, -1) and AC = (x, 3), x = 3/2 or x = 4 -/
theorem right_triangle_x_values (ABC : RightTriangle) 
  (h1 : ABC.AB = (2, -1)) 
  (h2 : ∃ x : ℝ, ABC.AC = (x, 3)) 
  (h3 : ABC.is_right_angled = true) :
  ∃ x : ℝ, ABC.AC.1 = x ∧ x ∈ possible_x_values := by
  sorry

#check right_triangle_x_values

end NUMINAMATH_CALUDE_right_triangle_x_values_l1914_191466


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_seven_mod_nineteen_l1914_191408

theorem largest_four_digit_congruent_to_seven_mod_nineteen :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 19 = 7 → n ≤ 9982 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_seven_mod_nineteen_l1914_191408


namespace NUMINAMATH_CALUDE_apple_harvest_per_section_l1914_191446

/-- The number of sections in the apple orchard -/
def num_sections : ℕ := 8

/-- The total number of sacks of apples harvested daily -/
def total_sacks : ℕ := 360

/-- The number of sacks of apples harvested from each section daily -/
def sacks_per_section : ℕ := total_sacks / num_sections

/-- Theorem stating that the number of sacks of apples harvested from each section daily is 45 -/
theorem apple_harvest_per_section : sacks_per_section = 45 := by
  sorry

end NUMINAMATH_CALUDE_apple_harvest_per_section_l1914_191446


namespace NUMINAMATH_CALUDE_smallest_fraction_l1914_191441

theorem smallest_fraction (x : ℝ) (h : x > 2022) :
  min (x / 2022) (min (2022 / (x - 1)) (min ((x + 1) / 2022) (min (2022 / x) (2022 / (x + 1))))) = 2022 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l1914_191441


namespace NUMINAMATH_CALUDE_line_equation_from_parametric_l1914_191411

/-- The equation of a line parameterized by (3t + 6, 5t - 7) is y = (5/3)x - 17 -/
theorem line_equation_from_parametric : 
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 7 → y = (5/3) * x - 17 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_from_parametric_l1914_191411


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l1914_191469

theorem cubic_expression_evaluation : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l1914_191469


namespace NUMINAMATH_CALUDE_phantom_needs_more_money_l1914_191420

/-- The amount of additional money Phantom needs to buy printer inks -/
def additional_money_needed (initial_money : ℚ) 
  (black_price red_price yellow_price blue_price magenta_price cyan_price : ℚ)
  (black_count red_count yellow_count blue_count magenta_count cyan_count : ℕ)
  (tax_rate : ℚ) : ℚ :=
  let subtotal := black_price * black_count + red_price * red_count + 
                  yellow_price * yellow_count + blue_price * blue_count + 
                  magenta_price * magenta_count + cyan_price * cyan_count
  let total_cost := subtotal + subtotal * tax_rate
  total_cost - initial_money

theorem phantom_needs_more_money :
  additional_money_needed 50 12 16 14 17 15 18 3 4 3 2 2 1 (5/100) = 185.20 := by
  sorry

end NUMINAMATH_CALUDE_phantom_needs_more_money_l1914_191420


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_minimum_l1914_191421

theorem arithmetic_geometric_sequence_minimum (n : ℕ) (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  d > 0 →
  (∀ k, a k = a 1 + (k - 1) * d) →
  a 1 = 5 →
  (a 5 - 1)^2 = a 2 * a 10 →
  (∀ k, S k = (k / 2) * (2 * a 1 + (k - 1) * d)) →
  (∀ k, (2 * S k + k + 32) / (a k + 1) ≥ 20 / 3) ∧
  (∃ k, (2 * S k + k + 32) / (a k + 1) = 20 / 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_minimum_l1914_191421


namespace NUMINAMATH_CALUDE_inequality_condition_l1914_191427

theorem inequality_condition :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1914_191427


namespace NUMINAMATH_CALUDE_thelmas_tomato_slices_l1914_191450

def slices_per_meal : ℕ := 20
def people_to_feed : ℕ := 8
def tomatoes_needed : ℕ := 20

def slices_per_tomato : ℕ := (slices_per_meal * people_to_feed) / tomatoes_needed

theorem thelmas_tomato_slices : slices_per_tomato = 8 := by
  sorry

end NUMINAMATH_CALUDE_thelmas_tomato_slices_l1914_191450


namespace NUMINAMATH_CALUDE_wooden_strip_triangle_l1914_191418

theorem wooden_strip_triangle (x : ℝ) : 
  (0 < x ∧ x < 5 ∧ 
   x + x > 10 - 2*x ∧
   10 - 2*x > 0) ↔ 
  (2.5 < x ∧ x < 5) :=
sorry

end NUMINAMATH_CALUDE_wooden_strip_triangle_l1914_191418


namespace NUMINAMATH_CALUDE_least_integer_with_deletion_property_l1914_191403

theorem least_integer_with_deletion_property : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n = 17) ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (m / 10 : ℚ) ≠ m / 17) ∧
  ((n / 10 : ℚ) = n / 17) := by
sorry

end NUMINAMATH_CALUDE_least_integer_with_deletion_property_l1914_191403


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_tangent_sum_l1914_191483

/-- If the sum of tangents of angle differences in a triangle is zero, then the triangle is isosceles. -/
theorem isosceles_triangle_from_tangent_sum (A B C : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_tangent_sum : Real.tan (A - B) + Real.tan (B - C) + Real.tan (C - A) = 0) : 
  A = B ∨ B = C ∨ C = A :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_tangent_sum_l1914_191483


namespace NUMINAMATH_CALUDE_shaded_square_area_ratio_l1914_191455

theorem shaded_square_area_ratio (n : ℕ) (shaded_area : ℕ) : 
  n = 5 → shaded_area = 5 → (shaded_area : ℚ) / (n^2 : ℚ) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_square_area_ratio_l1914_191455


namespace NUMINAMATH_CALUDE_triangle_problem_l1914_191429

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  (Real.sqrt 3 * t.c = 2 * t.a * Real.sin t.C) →  -- Condition 2
  (t.A < π / 2) →  -- Condition 3: A is acute
  (t.a = 2 * Real.sqrt 3) →  -- Condition 4
  (1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3) →  -- Condition 5: Area
  (t.A = π / 3 ∧ 
   ((t.b = 4 ∧ t.c = 2) ∨ (t.b = 2 ∧ t.c = 4))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1914_191429


namespace NUMINAMATH_CALUDE_inverse_mod_53_l1914_191493

theorem inverse_mod_53 (h : (21⁻¹ : ZMod 53) = 17) : (32⁻¹ : ZMod 53) = 36 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l1914_191493


namespace NUMINAMATH_CALUDE_complex_subtraction_zero_implies_equality_l1914_191439

theorem complex_subtraction_zero_implies_equality (a b : ℂ) : a - b = 0 → a = b := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_zero_implies_equality_l1914_191439


namespace NUMINAMATH_CALUDE_exists_solution_l1914_191481

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + Real.exp (x - a)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem exists_solution (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) → a = -Real.log 2 - 1 := by sorry

end NUMINAMATH_CALUDE_exists_solution_l1914_191481


namespace NUMINAMATH_CALUDE_complement_of_M_l1914_191416

def M : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 5}

theorem complement_of_M :
  (Set.univ : Set ℝ) \ M = {x : ℝ | x < -3 ∨ x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1914_191416


namespace NUMINAMATH_CALUDE_initial_orchids_l1914_191425

theorem initial_orchids (initial_roses : ℕ) (final_orchids : ℕ) (final_roses : ℕ) :
  initial_roses = 9 →
  final_orchids = 13 →
  final_roses = 3 →
  final_orchids - final_roses = 10 →
  ∃ initial_orchids : ℕ, initial_orchids = 3 :=
by sorry

end NUMINAMATH_CALUDE_initial_orchids_l1914_191425


namespace NUMINAMATH_CALUDE_total_bike_rides_l1914_191433

theorem total_bike_rides (billy_rides : ℕ) (john_rides : ℕ) (mother_rides : ℕ) : 
  billy_rides = 17 →
  john_rides = 2 * billy_rides →
  mother_rides = john_rides + 10 →
  billy_rides + john_rides + mother_rides = 95 := by
sorry

end NUMINAMATH_CALUDE_total_bike_rides_l1914_191433


namespace NUMINAMATH_CALUDE_probability_of_six_l1914_191422

/-- A fair die with 6 faces -/
structure FairDie :=
  (faces : Nat)
  (is_fair : Bool)
  (h_faces : faces = 6)
  (h_fair : is_fair = true)

/-- The probability of getting a specific face on a fair die -/
def probability_of_face (d : FairDie) : ℚ :=
  1 / d.faces

/-- Theorem: The probability of getting any specific face on a fair 6-faced die is 1/6 -/
theorem probability_of_six (d : FairDie) : probability_of_face d = 1 / 6 := by
  sorry

#eval (1 : ℚ) / 6  -- To show that 1/6 ≈ 0.17

end NUMINAMATH_CALUDE_probability_of_six_l1914_191422


namespace NUMINAMATH_CALUDE_parabola_vertex_l1914_191472

/-- A parabola with vertex (h, k) has the general form y = (x - h)² + k -/
def is_parabola_with_vertex (f : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, f x = (x - h)^2 + k

/-- The specific parabola we're considering -/
def f (x : ℝ) : ℝ := (x - 4)^2 - 3

/-- Theorem stating that f is a parabola with vertex (4, -3) -/
theorem parabola_vertex : is_parabola_with_vertex f 4 (-3) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1914_191472


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l1914_191431

theorem lcm_gcf_ratio : (Nat.lcm 240 630) / (Nat.gcd 240 630) = 168 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l1914_191431


namespace NUMINAMATH_CALUDE_perpendicular_diagonals_not_sufficient_for_rhombus_l1914_191413

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define diagonals of a quadrilateral
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((q.A.1 - q.C.1, q.A.2 - q.C.2), (q.B.1 - q.D.1, q.B.2 - q.D.2))

-- Define perpendicularity of two vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  let (AC, BD) := diagonals q
  perpendicular AC BD ∧ 
  AC.1^2 + AC.2^2 = BD.1^2 + BD.2^2 ∧
  (AC.1 / 2, AC.2 / 2) = (BD.1 / 2, BD.2 / 2)

-- Statement to prove
theorem perpendicular_diagonals_not_sufficient_for_rhombus :
  ∃ (q : Quadrilateral), 
    (let (AC, BD) := diagonals q; perpendicular AC BD) ∧ 
    ¬is_rhombus q :=
sorry

end NUMINAMATH_CALUDE_perpendicular_diagonals_not_sufficient_for_rhombus_l1914_191413


namespace NUMINAMATH_CALUDE_apple_cost_is_14_l1914_191476

/-- Represents the cost of groceries in dollars -/
structure GroceryCost where
  total : ℕ
  bananas : ℕ
  bread : ℕ
  milk : ℕ

/-- Calculates the cost of apples given the total cost and costs of other items -/
def appleCost (g : GroceryCost) : ℕ :=
  g.total - (g.bananas + g.bread + g.milk)

/-- Theorem stating that the cost of apples is 14 dollars given the specific grocery costs -/
theorem apple_cost_is_14 (g : GroceryCost) 
    (h1 : g.total = 42)
    (h2 : g.bananas = 12)
    (h3 : g.bread = 9)
    (h4 : g.milk = 7) : 
  appleCost g = 14 := by
  sorry

#eval appleCost { total := 42, bananas := 12, bread := 9, milk := 7 }

end NUMINAMATH_CALUDE_apple_cost_is_14_l1914_191476


namespace NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l1914_191463

/-- A quadratic polynomial with non-negative coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c

/-- The evaluation of a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (P : QuadraticPolynomial) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- The statement of the theorem -/
theorem quadratic_polynomial_inequality (P : QuadraticPolynomial) (x y : ℝ) :
  (P.eval (x * y))^2 ≤ (P.eval (x^2)) * (P.eval (y^2)) := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l1914_191463


namespace NUMINAMATH_CALUDE_square_brush_ratio_l1914_191442

theorem square_brush_ratio (s w : ℝ) (h_positive_s : 0 < s) (h_positive_w : 0 < w) :
  (w^2 + 2 * (s^2 / 2 - w^2) = s^2 / 3) → (s / w = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_square_brush_ratio_l1914_191442


namespace NUMINAMATH_CALUDE_parallelogram_not_symmetrical_l1914_191464

-- Define a type for shapes
inductive Shape
  | Circle
  | Rectangle
  | Parallelogram
  | IsoscelesTrapezoid

-- Define a property for symmetry
def isSymmetrical (s : Shape) : Prop :=
  match s with
  | Shape.Circle => True
  | Shape.Rectangle => True
  | Shape.Parallelogram => False
  | Shape.IsoscelesTrapezoid => True

-- Theorem statement
theorem parallelogram_not_symmetrical :
  ¬(isSymmetrical Shape.Parallelogram) :=
by
  sorry

#check parallelogram_not_symmetrical

end NUMINAMATH_CALUDE_parallelogram_not_symmetrical_l1914_191464


namespace NUMINAMATH_CALUDE_football_tournament_max_points_l1914_191415

theorem football_tournament_max_points (num_teams : ℕ) (points_win : ℕ) (points_draw : ℕ) (points_loss : ℕ) :
  num_teams = 15 →
  points_win = 3 →
  points_draw = 1 →
  points_loss = 0 →
  ∃ (N : ℕ), N = 34 ∧ 
    (∀ (M : ℕ), (∃ (teams : Finset (Fin num_teams)), teams.card ≥ 6 ∧ 
      (∀ t ∈ teams, ∃ (score : ℕ), score ≥ M)) → M ≤ N) :=
by sorry

end NUMINAMATH_CALUDE_football_tournament_max_points_l1914_191415


namespace NUMINAMATH_CALUDE_product_ab_l1914_191491

theorem product_ab (a b : ℚ) (h1 : 2 * a + 5 * b = 40) (h2 : 4 * a + 3 * b = 41) :
  a * b = 3315 / 98 := by
sorry

end NUMINAMATH_CALUDE_product_ab_l1914_191491


namespace NUMINAMATH_CALUDE_side_is_one_third_perimeter_l1914_191440

-- Define a triangle with an inscribed circle
structure TriangleWithInscribedCircle where
  -- We don't need to explicitly define the triangle or circle, 
  -- just the properties we're interested in
  side_length : ℝ
  perimeter : ℝ
  midpoint : ℝ × ℝ
  altitude_foot : ℝ × ℝ
  tangency_point : ℝ × ℝ

-- Define the symmetry condition
def is_symmetrical (t : TriangleWithInscribedCircle) : Prop :=
  let midpoint_distance := (t.midpoint.1 - t.tangency_point.1)^2 + (t.midpoint.2 - t.tangency_point.2)^2
  let altitude_foot_distance := (t.altitude_foot.1 - t.tangency_point.1)^2 + (t.altitude_foot.2 - t.tangency_point.2)^2
  midpoint_distance = altitude_foot_distance

-- State the theorem
theorem side_is_one_third_perimeter (t : TriangleWithInscribedCircle) 
  (h : is_symmetrical t) : t.side_length = t.perimeter / 3 := by
  sorry

end NUMINAMATH_CALUDE_side_is_one_third_perimeter_l1914_191440


namespace NUMINAMATH_CALUDE_museum_visit_permutations_l1914_191470

theorem museum_visit_permutations : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_museum_visit_permutations_l1914_191470


namespace NUMINAMATH_CALUDE_area_of_B_l1914_191488

-- Define set A
def A : Set ℝ := {a : ℝ | -1 ≤ a ∧ a ≤ 2}

-- Define set B
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 + p.2 ≥ 0}

-- Theorem statement
theorem area_of_B : MeasureTheory.volume B = 7 := by
  sorry

end NUMINAMATH_CALUDE_area_of_B_l1914_191488


namespace NUMINAMATH_CALUDE_goat_max_distance_l1914_191456

theorem goat_max_distance (center : ℝ × ℝ) (radius : ℝ) :
  center = (6, 8) →
  radius = 15 →
  let dist_to_center := Real.sqrt ((center.1 - 0)^2 + (center.2 - 0)^2)
  let max_distance := dist_to_center + radius
  max_distance = 25 := by sorry

end NUMINAMATH_CALUDE_goat_max_distance_l1914_191456


namespace NUMINAMATH_CALUDE_trigonometric_equality_l1914_191477

theorem trigonometric_equality : 
  (Real.cos (10 * π / 180) + Real.sqrt 3 * Real.sin (10 * π / 180)) / 
  Real.sqrt (1 - Real.sin (50 * π / 180) ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l1914_191477


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1914_191454

theorem min_value_sum_reciprocals (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 3) : 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
    1 / (x + y) + 1 / z ≥ 1 / (a + b) + 1 / c) → 
  1 / (a + b) + 1 / c = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1914_191454


namespace NUMINAMATH_CALUDE_distance_from_origin_l1914_191430

theorem distance_from_origin (x y : ℝ) (h1 : |x| = 8) 
  (h2 : Real.sqrt ((x - 7)^2 + (y - 3)^2) = 8) (h3 : y > 3) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (136 + 6 * Real.sqrt 63) :=
by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1914_191430


namespace NUMINAMATH_CALUDE_student_number_problem_l1914_191478

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 102 → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1914_191478


namespace NUMINAMATH_CALUDE_acme_vowel_soup_strings_l1914_191400

/-- Represents the number of times each vowel appears in the soup -/
def vowel_count : Fin 5 → ℕ
  | 0 => 6  -- A
  | 1 => 6  -- E
  | 2 => 6  -- I
  | 3 => 6  -- O
  | 4 => 3  -- U

/-- The length of the strings to be formed -/
def string_length : ℕ := 6

/-- Calculates the number of possible strings -/
def count_strings : ℕ :=
  (Finset.range 4).sum (λ k =>
    (Nat.choose string_length k) * (4 * vowel_count 0) ^ (string_length - k))

theorem acme_vowel_soup_strings :
  count_strings = 117072 :=
sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_strings_l1914_191400


namespace NUMINAMATH_CALUDE_common_chord_length_l1914_191424

theorem common_chord_length (r : ℝ) (h : r = 15) :
  let chord_length := 2 * r * Real.sqrt 3
  chord_length = 15 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_l1914_191424


namespace NUMINAMATH_CALUDE_jungkook_has_bigger_number_l1914_191404

theorem jungkook_has_bigger_number : 
  let jungkook_number := 3 + 6
  let yoongi_number := 4
  jungkook_number > yoongi_number := by
sorry

end NUMINAMATH_CALUDE_jungkook_has_bigger_number_l1914_191404


namespace NUMINAMATH_CALUDE_cases_needed_l1914_191485

def boxes_sold : ℕ := 10
def boxes_per_case : ℕ := 2

theorem cases_needed : boxes_sold / boxes_per_case = 5 := by
  sorry

end NUMINAMATH_CALUDE_cases_needed_l1914_191485


namespace NUMINAMATH_CALUDE_f_is_even_l1914_191474

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_is_even (g : ℝ → ℝ) (h : isOdd g) :
  isEven (fun x ↦ |g (x^4)|) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1914_191474


namespace NUMINAMATH_CALUDE_y_at_64_l1914_191417

-- Define the function y in terms of k and x
def y (k : ℝ) (x : ℝ) : ℝ := k * x^(1/3)

-- State the theorem
theorem y_at_64 (k : ℝ) :
  y k 8 = 4 → y k 64 = 8 := by
  sorry

end NUMINAMATH_CALUDE_y_at_64_l1914_191417


namespace NUMINAMATH_CALUDE_speedster_convertibles_count_l1914_191497

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  nonSpeedsters : ℕ
  speedsterConvertibles : ℕ

/-- Theorem stating the number of Speedster convertibles in the inventory -/
theorem speedster_convertibles_count (inv : Inventory) :
  inv.nonSpeedsters = 30 ∧
  inv.speedsters = 3 * inv.total / 4 ∧
  inv.nonSpeedsters = inv.total - inv.speedsters ∧
  inv.speedsterConvertibles = 3 * inv.speedsters / 5 →
  inv.speedsterConvertibles = 54 := by
  sorry


end NUMINAMATH_CALUDE_speedster_convertibles_count_l1914_191497


namespace NUMINAMATH_CALUDE_mistaken_divisor_problem_l1914_191401

theorem mistaken_divisor_problem (dividend : ℕ) (mistaken_divisor : ℕ) :
  dividend % 21 = 0 →
  dividend / 21 = 28 →
  dividend / mistaken_divisor = 49 →
  mistaken_divisor = 12 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_divisor_problem_l1914_191401


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1914_191409

theorem min_value_theorem (a : ℝ) (h : a > 2) : a + 4 / (a - 2) ≥ 6 :=
sorry

theorem equality_condition (a : ℝ) (h : a > 2) : 
  ∃ a₀ > 2, a₀ + 4 / (a₀ - 2) = 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1914_191409


namespace NUMINAMATH_CALUDE_man_crossing_bridge_l1914_191406

/-- Proves that a man walking at 10 km/hr takes 10 minutes to cross a 1666.6666666666665 meter bridge -/
theorem man_crossing_bridge 
  (walking_rate : ℝ) 
  (bridge_length : ℝ) 
  (h1 : walking_rate = 10) -- km/hr
  (h2 : bridge_length = 1666.6666666666665) -- meters
  : (bridge_length / (walking_rate * 1000 / 60)) = 10 := by
  sorry

#check man_crossing_bridge

end NUMINAMATH_CALUDE_man_crossing_bridge_l1914_191406


namespace NUMINAMATH_CALUDE_sugar_profit_percentage_l1914_191402

/-- Proves that given 1000 kg of sugar, with 400 kg sold at 8% profit and 600 kg sold at x% profit,
    if the overall profit is 14%, then x = 18. -/
theorem sugar_profit_percentage 
  (total_sugar : ℝ) 
  (sugar_at_8_percent : ℝ) 
  (sugar_at_x_percent : ℝ) 
  (x : ℝ) :
  total_sugar = 1000 →
  sugar_at_8_percent = 400 →
  sugar_at_x_percent = 600 →
  sugar_at_8_percent * 0.08 + sugar_at_x_percent * (x / 100) = total_sugar * 0.14 →
  x = 18 := by
  sorry

end NUMINAMATH_CALUDE_sugar_profit_percentage_l1914_191402


namespace NUMINAMATH_CALUDE_tank_capacity_l1914_191432

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 1/4 →
  final_fraction = 2/3 →
  added_amount = 180 →
  (final_fraction - initial_fraction) * (added_amount / (final_fraction - initial_fraction)) = 432 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1914_191432


namespace NUMINAMATH_CALUDE_min_distance_ellipse_to_Q_l1914_191498

/-- The ellipse with semi-major axis 4 and semi-minor axis 2 -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

/-- The point Q -/
def Q : ℝ × ℝ := (2, 0)

/-- The squared distance between two points -/
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem min_distance_ellipse_to_Q :
  ∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧
    (∀ (P' : ℝ × ℝ), ellipse P'.1 P'.2 →
      distance_squared P Q ≤ distance_squared P' Q) ∧
    distance_squared P Q = (2*Real.sqrt 6/3)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_to_Q_l1914_191498


namespace NUMINAMATH_CALUDE_bounds_per_meter_proof_l1914_191419

/-- Represents the number of bounds in one meter -/
def bounds_per_meter : ℚ :=
  21 / 100

/-- The number of leaps that equal 3 bounds -/
def leaps_to_bounds : ℕ := 4

/-- The number of bounds that equal 4 leaps -/
def bounds_to_leaps : ℕ := 3

/-- The number of strides that equal 2 leaps -/
def strides_to_leaps : ℕ := 5

/-- The number of leaps that equal 5 strides -/
def leaps_to_strides : ℕ := 2

/-- The number of strides that equal 10 meters -/
def strides_to_meters : ℕ := 7

/-- The number of meters that equal 7 strides -/
def meters_to_strides : ℕ := 10

theorem bounds_per_meter_proof :
  bounds_per_meter = 21 / 100 :=
by sorry

end NUMINAMATH_CALUDE_bounds_per_meter_proof_l1914_191419


namespace NUMINAMATH_CALUDE_park_area_l1914_191436

/-- The area of a rectangular park with a given length-to-breadth ratio and perimeter -/
theorem park_area (length breadth perimeter : ℝ) : 
  length > 0 →
  breadth > 0 →
  length / breadth = 1 / 3 →
  perimeter = 2 * (length + breadth) →
  length * breadth = 30000 := by
  sorry

#check park_area

end NUMINAMATH_CALUDE_park_area_l1914_191436


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1914_191434

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (1 / a + 2 / b) ≥ 3 / 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1914_191434


namespace NUMINAMATH_CALUDE_min_value_theorem_l1914_191435

theorem min_value_theorem (x y z : ℝ) (h : 2 * x + 2 * y + z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1914_191435


namespace NUMINAMATH_CALUDE_terminating_decimal_thirteen_over_sixtwentyfive_l1914_191449

theorem terminating_decimal_thirteen_over_sixtwentyfive :
  (13 : ℚ) / 625 = (208 : ℚ) / 10000 :=
by sorry

end NUMINAMATH_CALUDE_terminating_decimal_thirteen_over_sixtwentyfive_l1914_191449


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1914_191448

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = n * (a 1 + a n) / 2) →  -- Definition of S_n
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Property of arithmetic sequence
  a 1 = -2018 →
  (S 2016 / 2016 - S 10 / 10 = 2006) →
  S 2018 = -2018 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1914_191448


namespace NUMINAMATH_CALUDE_pens_cost_after_discount_and_tax_l1914_191471

/-- The cost of one pen in terms of the cost of one pencil -/
def pen_cost (pencil_cost : ℚ) : ℚ := 5 * pencil_cost

/-- The total cost of pens and pencils -/
def total_cost (pencil_cost : ℚ) : ℚ := 3 * pen_cost pencil_cost + 5 * pencil_cost

/-- The cost of one dozen pens -/
def dozen_pens_cost (pencil_cost : ℚ) : ℚ := 12 * pen_cost pencil_cost

/-- The discount rate applied to one dozen pens -/
def discount_rate : ℚ := 1 / 10

/-- The tax rate applied after the discount -/
def tax_rate : ℚ := 18 / 100

/-- The final cost of one dozen pens after discount and tax -/
def final_cost (pencil_cost : ℚ) : ℚ :=
  let discounted_cost := dozen_pens_cost pencil_cost * (1 - discount_rate)
  discounted_cost * (1 + tax_rate)

theorem pens_cost_after_discount_and_tax :
  ∃ (pencil_cost : ℚ),
    total_cost pencil_cost = 260 ∧
    final_cost pencil_cost = 828.36 := by
  sorry


end NUMINAMATH_CALUDE_pens_cost_after_discount_and_tax_l1914_191471


namespace NUMINAMATH_CALUDE_arc_length_problem_l1914_191484

theorem arc_length_problem (r : ℝ) (θ : ℝ) (a : ℝ) :
  r = 18 →
  θ = π / 3 →
  r * θ = a * π →
  a = 6 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_problem_l1914_191484


namespace NUMINAMATH_CALUDE_octahedron_volume_with_unit_inscribed_sphere_l1914_191468

/-- An octahedron is a polyhedron with 8 equilateral triangular faces. -/
structure Octahedron where
  -- We don't need to define the full structure, just what we need for this problem
  volume : ℝ

/-- A sphere is a three-dimensional geometric object. -/
structure Sphere where
  radius : ℝ

/-- An octahedron with an inscribed sphere. -/
structure OctahedronWithInscribedSphere where
  octahedron : Octahedron
  sphere : Sphere
  inscribed : sphere.radius = 1  -- The sphere is inscribed and has radius 1

/-- The volume of an octahedron with an inscribed sphere of radius 1 is √6. -/
theorem octahedron_volume_with_unit_inscribed_sphere
  (o : OctahedronWithInscribedSphere) :
  o.octahedron.volume = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_volume_with_unit_inscribed_sphere_l1914_191468


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1914_191490

theorem cubic_equation_solution (m : ℝ) (h : m^2 + m - 1 = 0) :
  m^3 + 2*m^2 + 2005 = 2006 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1914_191490


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l1914_191467

theorem least_positive_integer_congruence :
  ∃! x : ℕ+, x.val + 3567 ≡ 1543 [ZMOD 14] ∧
  ∀ y : ℕ+, y.val + 3567 ≡ 1543 [ZMOD 14] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l1914_191467


namespace NUMINAMATH_CALUDE_unique_prime_satisfying_condition_l1914_191486

theorem unique_prime_satisfying_condition : 
  ∀ p : ℕ, Prime p → (Prime (p^3 + p^2 + 11*p + 2) ↔ p = 3) := by sorry

end NUMINAMATH_CALUDE_unique_prime_satisfying_condition_l1914_191486


namespace NUMINAMATH_CALUDE_cricketer_average_score_l1914_191410

theorem cricketer_average_score (score1 score2 : ℝ) (matches1 matches2 : ℕ) 
  (h1 : score1 = 20)
  (h2 : score2 = 30)
  (h3 : matches1 = 2)
  (h4 : matches2 = 3) :
  let total_matches := matches1 + matches2
  let total_score := score1 * matches1 + score2 * matches2
  total_score / total_matches = 26 := by
sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l1914_191410


namespace NUMINAMATH_CALUDE_expression_evaluation_l1914_191457

theorem expression_evaluation :
  let x : ℚ := -2/5
  let y : ℚ := 2
  2 * (x^2 * y - 2 * x * y) - 3 * (x^2 * y - 3 * x * y) + x^2 * y = -4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1914_191457


namespace NUMINAMATH_CALUDE_joan_quarters_l1914_191412

def total_cents : ℕ := 150
def cents_per_quarter : ℕ := 25

theorem joan_quarters : total_cents / cents_per_quarter = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_quarters_l1914_191412


namespace NUMINAMATH_CALUDE_abc_value_l1914_191495

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 30 * Real.sqrt 5)
  (hac : a * c = 45 * Real.sqrt 5)
  (hbc : b * c = 40 * Real.sqrt 5) :
  a * b * c = 300 * Real.sqrt 3 * (5 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_abc_value_l1914_191495


namespace NUMINAMATH_CALUDE_ratio_theorem_max_coeff_theorem_l1914_191443

open Real

/-- The ratio of the sum of all coefficients to the sum of all binomial coefficients
    in the expansion of (x^(2/3) + 3x^2)^n is 32 -/
def ratio_condition (n : ℕ) : Prop :=
  (4 : ℝ)^n / (2 : ℝ)^n = 32

/-- The value of n that satisfies the ratio condition -/
def n_value : ℕ := 5

/-- Theorem stating that n_value satisfies the ratio condition -/
theorem ratio_theorem : ratio_condition n_value := by
  sorry

/-- The terms with maximum binomial coefficient in the expansion -/
def max_coeff_terms (x : ℝ) : ℝ × ℝ :=
  (90 * x^6, 270 * x^(22/3))

/-- Theorem stating that max_coeff_terms gives the correct terms -/
theorem max_coeff_theorem (x : ℝ) :
  max_coeff_terms x = (90 * x^6, 270 * x^(22/3)) := by
  sorry

end NUMINAMATH_CALUDE_ratio_theorem_max_coeff_theorem_l1914_191443


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1914_191492

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 1 / a 0

theorem fourth_term_of_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_first : a 0 = 2^(1/2 : ℝ))
  (h_second : a 1 = 2^(1/4 : ℝ))
  (h_third : a 2 = 2^(1/8 : ℝ)) :
  a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1914_191492
