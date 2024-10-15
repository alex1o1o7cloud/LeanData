import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l517_51780

theorem system_solution : ∃! (x y : ℚ), 
  2 * x - 3 * y = 5 ∧ 
  4 * x - 6 * y = 10 ∧ 
  x + y = 7 ∧ 
  x = 26 / 5 ∧ 
  y = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l517_51780


namespace NUMINAMATH_CALUDE_unique_determination_from_sums_and_products_l517_51779

theorem unique_determination_from_sums_and_products 
  (x y z : ℝ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ y ≠ z) 
  (sum_xy sum_xz sum_yz : ℝ) 
  (prod_xy prod_xz prod_yz : ℝ) 
  (h_sums : sum_xy = x + y ∧ sum_xz = x + z ∧ sum_yz = y + z) 
  (h_prods : prod_xy = x * y ∧ prod_xz = x * z ∧ prod_yz = y * z) :
  ∃! (a b c : ℝ), (a = x ∧ b = y ∧ c = z) ∨ (a = x ∧ b = z ∧ c = y) ∨ 
                   (a = y ∧ b = x ∧ c = z) ∨ (a = y ∧ b = z ∧ c = x) ∨ 
                   (a = z ∧ b = x ∧ c = y) ∨ (a = z ∧ b = y ∧ c = x) :=
by sorry

end NUMINAMATH_CALUDE_unique_determination_from_sums_and_products_l517_51779


namespace NUMINAMATH_CALUDE_area_of_region_l517_51712

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 37 ∧ 
   A = Real.pi * (Real.sqrt ((x + 3)^2 + (y - 4)^2))^2 ∧
   x^2 + y^2 + 6*x - 8*y - 12 = 0) := by
sorry

end NUMINAMATH_CALUDE_area_of_region_l517_51712


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_not_factorization_expansion_not_complete_factorization_not_factorization_expansion_2_l517_51791

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 4 = (x - 2) * (x + 2) :=
by sorry

theorem not_factorization_expansion (a : ℝ) :
  (a - 1)^2 = a^2 - 2*a + 1 :=
by sorry

theorem not_complete_factorization (x : ℝ) :
  x^2 - 2*x - 6 = x*(x - 2) - 6 :=
by sorry

theorem not_factorization_expansion_2 (x : ℝ) :
  x*(x - 1) = x^2 - x :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_not_factorization_expansion_not_complete_factorization_not_factorization_expansion_2_l517_51791


namespace NUMINAMATH_CALUDE_soccer_tournament_matches_l517_51728

/-- The number of matches in a round-robin tournament with n teams -/
def numMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of matches between two groups of teams -/
def numMatchesBetweenGroups (n m : ℕ) : ℕ := n * m

theorem soccer_tournament_matches :
  (numMatches 3 = 3) ∧
  (numMatches 4 = 6) ∧
  (numMatchesBetweenGroups 3 4 = 12) := by
  sorry

#eval numMatches 3  -- Expected output: 3
#eval numMatches 4  -- Expected output: 6
#eval numMatchesBetweenGroups 3 4  -- Expected output: 12

end NUMINAMATH_CALUDE_soccer_tournament_matches_l517_51728


namespace NUMINAMATH_CALUDE_compound_composition_l517_51768

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  h : ℕ
  cl : ℕ
  o : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  h : ℝ
  cl : ℝ
  o : ℝ

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (comp : CompoundComposition) (weights : AtomicWeights) : ℝ :=
  comp.h * weights.h + comp.cl * weights.cl + comp.o * weights.o

/-- The main theorem to prove -/
theorem compound_composition (weights : AtomicWeights) :
  let comp := CompoundComposition.mk 1 1 2
  weights.h = 1 ∧ weights.cl = 35.5 ∧ weights.o = 16 →
  molecularWeight comp weights = 68 := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l517_51768


namespace NUMINAMATH_CALUDE_product_constraint_sum_l517_51729

theorem product_constraint_sum (w x y z : ℕ) : 
  w * x * y * z = 720 → 
  0 < w → w < x → x < y → y < z → z < 20 → 
  w + z = 14 := by
sorry

end NUMINAMATH_CALUDE_product_constraint_sum_l517_51729


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l517_51716

theorem greatest_integer_difference (x y : ℚ) 
  (hx : 3 < x) (hxy : x < (3/2)^3) (hyz : (3/2)^3 < y) (hy : y < 7) :
  ∃ (n : ℕ), n = 2 ∧ ∀ (m : ℕ), (m : ℚ) ≤ y - x → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l517_51716


namespace NUMINAMATH_CALUDE_sum_equals_1300_l517_51733

/-- Converts a number from base 15 to base 10 -/
def base15ToBase10 (n : Nat) : Nat :=
  (n / 100) * 225 + ((n / 10) % 10) * 15 + (n % 10)

/-- Converts a number from base 7 to base 10, where 'A' represents 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 49 + ((n / 10) % 10) * 7 + (n % 10)

/-- Theorem stating that the sum of 537 (base 15) and 1A4 (base 7) equals 1300 in base 10 -/
theorem sum_equals_1300 : 
  base15ToBase10 537 + base7ToBase10 194 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_1300_l517_51733


namespace NUMINAMATH_CALUDE_fraction_multiplication_result_l517_51788

theorem fraction_multiplication_result : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5040 = 1512 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_result_l517_51788


namespace NUMINAMATH_CALUDE_counterexample_exists_l517_51706

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l517_51706


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l517_51732

theorem complex_fraction_simplification :
  (1/2 * 1/3 * 1/4 * 1/5 + 3/2 * 3/4 * 3/5) / (1/2 * 2/3 * 2/5) = 41/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l517_51732


namespace NUMINAMATH_CALUDE_election_votes_l517_51722

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (60 * total_votes) / 100 - (40 * total_votes) / 100 = 240) : 
  (60 * total_votes) / 100 = 720 :=
sorry

end NUMINAMATH_CALUDE_election_votes_l517_51722


namespace NUMINAMATH_CALUDE_resulting_polygon_has_18_sides_l517_51769

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ

/-- Represents the arrangement of polygons. -/
structure PolygonArrangement where
  pentagon : RegularPolygon
  triangle : RegularPolygon
  octagon : RegularPolygon
  hexagon : RegularPolygon
  square : RegularPolygon

/-- The number of sides exposed to the outside for polygons adjacent to one other shape. -/
def exposedSidesOneAdjacent (p1 p2 : RegularPolygon) : ℕ :=
  p1.sides + p2.sides - 2

/-- The number of sides exposed to the outside for polygons adjacent to two other shapes. -/
def exposedSidesTwoAdjacent (p1 p2 p3 : RegularPolygon) : ℕ :=
  p1.sides + p2.sides + p3.sides - 6

/-- The total number of sides in the resulting polygon. -/
def totalSides (arrangement : PolygonArrangement) : ℕ :=
  exposedSidesOneAdjacent arrangement.pentagon arrangement.square +
  exposedSidesTwoAdjacent arrangement.triangle arrangement.octagon arrangement.hexagon

/-- Theorem stating that the resulting polygon has 18 sides. -/
theorem resulting_polygon_has_18_sides (arrangement : PolygonArrangement)
  (h1 : arrangement.pentagon.sides = 5)
  (h2 : arrangement.triangle.sides = 3)
  (h3 : arrangement.octagon.sides = 8)
  (h4 : arrangement.hexagon.sides = 6)
  (h5 : arrangement.square.sides = 4) :
  totalSides arrangement = 18 := by
  sorry

end NUMINAMATH_CALUDE_resulting_polygon_has_18_sides_l517_51769


namespace NUMINAMATH_CALUDE_fraction_addition_l517_51754

theorem fraction_addition (c : ℝ) : (6 + 5 * c) / 9 + 3 = (33 + 5 * c) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l517_51754


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l517_51771

theorem distance_between_complex_points :
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l517_51771


namespace NUMINAMATH_CALUDE_danivan_drugstore_inventory_l517_51760

/-- Calculates the remaining inventory of sanitizer gel at Danivan Drugstore -/
def remaining_inventory (initial_inventory : ℕ) 
  (daily_sales : List ℕ) (supplier_deliveries : List ℕ) : ℕ :=
  initial_inventory - (daily_sales.sum) + (supplier_deliveries.sum)

theorem danivan_drugstore_inventory : 
  let initial_inventory : ℕ := 4500
  let daily_sales : List ℕ := [2445, 906, 215, 457, 312, 239, 188]
  let supplier_deliveries : List ℕ := [350, 750, 981]
  remaining_inventory initial_inventory daily_sales supplier_deliveries = 819 := by
  sorry

end NUMINAMATH_CALUDE_danivan_drugstore_inventory_l517_51760


namespace NUMINAMATH_CALUDE_smallest_with_property_l517_51735

/-- A function that returns the list of digits of a natural number. -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if two lists of natural numbers are permutations of each other. -/
def is_permutation (l1 l2 : List ℕ) : Prop := sorry

/-- The property we're looking for: when multiplied by 9, the result has the same digits in a different order. -/
def has_property (n : ℕ) : Prop :=
  is_permutation (digits n) (digits (9 * n))

/-- The theorem stating that 1089 is the smallest natural number with the desired property. -/
theorem smallest_with_property :
  has_property 1089 ∧ ∀ m : ℕ, m < 1089 → ¬(has_property m) := by sorry

end NUMINAMATH_CALUDE_smallest_with_property_l517_51735


namespace NUMINAMATH_CALUDE_min_troupe_size_l517_51757

def is_valid_troupe_size (n : ℕ) : Prop :=
  n % 4 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n > 50

theorem min_troupe_size :
  ∃ (n : ℕ), is_valid_troupe_size n ∧ ∀ (m : ℕ), is_valid_troupe_size m → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_min_troupe_size_l517_51757


namespace NUMINAMATH_CALUDE_village_population_distribution_l517_51789

theorem village_population_distribution (pop_20k_to_50k : ℝ) (pop_under_20k : ℝ) (pop_50k_and_above : ℝ) :
  pop_20k_to_50k = 45 →
  pop_under_20k = 30 →
  pop_50k_and_above = 25 →
  pop_20k_to_50k + pop_under_20k = 75 :=
by sorry

end NUMINAMATH_CALUDE_village_population_distribution_l517_51789


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l517_51718

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 90) (h2 : B = 50) : C = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l517_51718


namespace NUMINAMATH_CALUDE_inner_tangent_circle_radius_l517_51750

/-- Given a right triangle with legs 3 and 4 units, the radius of the circle
    tangent to both legs and the circumcircle internally is 2 units. -/
theorem inner_tangent_circle_radius (a b c r : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → r = a + b - c → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_inner_tangent_circle_radius_l517_51750


namespace NUMINAMATH_CALUDE_trig_sum_equals_two_l517_51751

theorem trig_sum_equals_two : Real.cos (π / 4) ^ 2 + Real.tan (π / 3) * Real.cos (π / 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_two_l517_51751


namespace NUMINAMATH_CALUDE_total_water_intake_l517_51708

/-- Calculates the total water intake throughout the day given specific drinking patterns --/
theorem total_water_intake (morning : ℝ) : morning = 1.5 →
  let early_afternoon := 2 * morning
  let late_afternoon := 3 * morning
  let evening := late_afternoon * (1 - 0.25)
  let night := 2 * evening
  morning + early_afternoon + late_afternoon + evening + night = 19.125 := by
  sorry

end NUMINAMATH_CALUDE_total_water_intake_l517_51708


namespace NUMINAMATH_CALUDE_eiffel_tower_height_difference_l517_51761

/-- The height difference between two structures -/
def height_difference (taller_height shorter_height : ℝ) : ℝ :=
  taller_height - shorter_height

/-- The heights of the Burj Khalifa and Eiffel Tower -/
def burj_khalifa_height : ℝ := 830
def eiffel_tower_height : ℝ := 324

/-- Theorem: The Eiffel Tower is 506 meters lower than the Burj Khalifa -/
theorem eiffel_tower_height_difference : 
  height_difference burj_khalifa_height eiffel_tower_height = 506 := by
  sorry

end NUMINAMATH_CALUDE_eiffel_tower_height_difference_l517_51761


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l517_51759

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) :
  z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l517_51759


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l517_51711

/-- A line with equal x and y intercepts passing through (-1, 2) -/
structure EqualInterceptLine where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (-1, 2)
  point_condition : 2 = m * (-1) + b
  -- The line has equal x and y intercepts
  equal_intercepts : b ≠ 0 → -b/m = b

/-- The equation of an EqualInterceptLine is either 2x + y = 0 or x + y - 1 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = -2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 1) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l517_51711


namespace NUMINAMATH_CALUDE_intersection_range_l517_51770

/-- The set M representing an ellipse -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}

/-- The set N representing a line with slope m and y-intercept b -/
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

/-- Theorem stating the range of b for which M and N always intersect -/
theorem intersection_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) ↔ b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

#check intersection_range

end NUMINAMATH_CALUDE_intersection_range_l517_51770


namespace NUMINAMATH_CALUDE_circle_graph_proportion_l517_51724

theorem circle_graph_proportion (angle : ℝ) (percentage : ℝ) :
  angle = 180 →
  angle / 360 = percentage / 100 →
  percentage = 50 := by
sorry

end NUMINAMATH_CALUDE_circle_graph_proportion_l517_51724


namespace NUMINAMATH_CALUDE_max_profit_is_45_6_l517_51717

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2 * x

/-- Total profit function -/
def S (x : ℝ) : ℝ := L₁ x + L₂ (15 - x)

/-- The maximum total profit is 45.6 when selling 15 cars across both locations -/
theorem max_profit_is_45_6 :
  ∃ x : ℕ, x ≤ 15 ∧ S x = 45.6 ∧ ∀ y : ℕ, y ≤ 15 → S y ≤ S x := by
  sorry

end NUMINAMATH_CALUDE_max_profit_is_45_6_l517_51717


namespace NUMINAMATH_CALUDE_quadratic_trinomial_with_integral_roots_l517_51748

theorem quadratic_trinomial_with_integral_roots :
  ∃ (a b c : ℕ+),
    (∃ (x : ℤ), a * x^2 + b * x + c = 0) ∧
    (∃ (y : ℤ), (a + 1) * y^2 + (b + 1) * y + (c + 1) = 0) ∧
    (∃ (z : ℤ), (a + 2) * z^2 + (b + 2) * z + (c + 2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_with_integral_roots_l517_51748


namespace NUMINAMATH_CALUDE_price_decrease_units_sold_ratio_l517_51746

theorem price_decrease_units_sold_ratio (P U : ℝ) (h : P > 0) (k : U > 0) :
  let new_price := 0.25 * P
  let new_units := U / 0.25
  let revenue_unchanged := P * U = new_price * new_units
  let percent_decrease_price := 75
  let percent_increase_units := (new_units - U) / U * 100
  revenue_unchanged →
  percent_increase_units / percent_decrease_price = 4 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_units_sold_ratio_l517_51746


namespace NUMINAMATH_CALUDE_determine_b_l517_51726

theorem determine_b (a b : ℝ) (h1 : 3 * a + 4 = 1) (h2 : b - 2 * a = 5) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_determine_b_l517_51726


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l517_51738

theorem quadrilateral_angle_measure (E F G H : ℝ) : 
  E = 3 * F ∧ E = 4 * G ∧ E = 6 * H ∧ E + F + G + H = 360 → E = 540 / 7 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l517_51738


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l517_51742

def total_potatoes : ℕ := 13
def cooked_potatoes : ℕ := 5
def cooking_time_per_potato : ℕ := 6

theorem remaining_cooking_time : (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 48 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l517_51742


namespace NUMINAMATH_CALUDE_hyperbola_equation_for_given_conditions_l517_51772

/-- A hyperbola with given eccentricity and foci -/
structure Hyperbola where
  eccentricity : ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- Theorem: A hyperbola with eccentricity 2 and foci at (-4,0) and (4,0) has the equation x^2/4 - y^2/12 = 1 -/
theorem hyperbola_equation_for_given_conditions (h : Hyperbola) 
    (h_ecc : h.eccentricity = 2)
    (h_foci : h.focus1 = (-4, 0) ∧ h.focus2 = (4, 0)) :
    ∀ x y : ℝ, hyperbola_equation h x y :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_for_given_conditions_l517_51772


namespace NUMINAMATH_CALUDE_complex_power_simplification_l517_51773

theorem complex_power_simplification :
  ((2 + Complex.I) / (2 - Complex.I)) ^ 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l517_51773


namespace NUMINAMATH_CALUDE_discount_difference_l517_51741

def initial_amount : ℝ := 15000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_amount 0.25) 0.1) 0.05

def option2_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_amount 0.3) 0.1) 0.1

theorem discount_difference :
  option1_price - option2_price = 1113.75 := by sorry

end NUMINAMATH_CALUDE_discount_difference_l517_51741


namespace NUMINAMATH_CALUDE_racetrack_circumference_difference_l517_51702

/-- The difference in circumferences of two concentric circles -/
theorem racetrack_circumference_difference (inner_diameter outer_diameter : ℝ) 
  (h1 : inner_diameter = 55)
  (h2 : outer_diameter = inner_diameter + 2 * 15) :
  π * outer_diameter - π * inner_diameter = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_racetrack_circumference_difference_l517_51702


namespace NUMINAMATH_CALUDE_evaluate_expression_l517_51734

theorem evaluate_expression (b x : ℝ) (h : x = b + 10) : 2*x - b + 5 = b + 25 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l517_51734


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l517_51782

theorem sqrt_expression_equality (x : ℝ) :
  (Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt 0.81 / Real.sqrt x = 2.507936507936508) →
  x = 0.49 := by
sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l517_51782


namespace NUMINAMATH_CALUDE_largest_four_digit_with_product_72_l517_51785

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_with_product_72 :
  ∃ M : ℕ, is_four_digit M ∧ 
    digit_product M = 72 ∧
    (∀ n : ℕ, is_four_digit n → digit_product n = 72 → n ≤ M) ∧
    digit_sum M = 17 := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_with_product_72_l517_51785


namespace NUMINAMATH_CALUDE_sum_f_positive_l517_51719

/-- The function f(x) = x + x³ -/
def f (x : ℝ) : ℝ := x + x^3

/-- Theorem: For any real numbers x₁, x₂, x₃ satisfying the given conditions,
    the sum f(x₁) + f(x₂) + f(x₃) is always positive -/
theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
    (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
    f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l517_51719


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l517_51740

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℤ) + (⌈x⌉ : ℤ) = 7 ↔ 3 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l517_51740


namespace NUMINAMATH_CALUDE_midpoint_sum_after_doubling_x_l517_51796

/-- Given a segment with endpoints (10, 3) and (-4, 7), prove that the sum of the doubled x-coordinate
and the y-coordinate of the midpoint is 11. -/
theorem midpoint_sum_after_doubling_x : 
  let p1 : ℝ × ℝ := (10, 3)
  let p2 : ℝ × ℝ := (-4, 7)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let doubled_x : ℝ := 2 * midpoint.1
  doubled_x + midpoint.2 = 11 := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_after_doubling_x_l517_51796


namespace NUMINAMATH_CALUDE_complex_expression_equals_one_l517_51755

theorem complex_expression_equals_one 
  (x y : ℂ) 
  (h_nonzero : x ≠ 0 ∧ y ≠ 0) 
  (h_equation : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^2005 + (y / (x + y))^2005 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_one_l517_51755


namespace NUMINAMATH_CALUDE_pit_width_is_five_l517_51709

/-- Represents the dimensions and conditions of the field and pit problem -/
structure FieldPitProblem where
  field_length : ℝ
  field_width : ℝ
  pit_length : ℝ
  pit_depth : ℝ
  field_rise : ℝ

/-- Calculates the width of the pit given the problem conditions -/
def calculate_pit_width (problem : FieldPitProblem) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the pit width is 5 meters given the specified conditions -/
theorem pit_width_is_five (problem : FieldPitProblem) 
  (h1 : problem.field_length = 20)
  (h2 : problem.field_width = 10)
  (h3 : problem.pit_length = 8)
  (h4 : problem.pit_depth = 2)
  (h5 : problem.field_rise = 0.5) :
  calculate_pit_width problem = 5 := by
  sorry

end NUMINAMATH_CALUDE_pit_width_is_five_l517_51709


namespace NUMINAMATH_CALUDE_z_plus_two_over_z_traces_ellipse_l517_51794

/-- Given a complex number z with |z| = 3, prove that z + 2/z traces an ellipse -/
theorem z_plus_two_over_z_traces_ellipse (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  ∀ (w : ℂ), w = z + 2 / z → (w.re / a)^2 + (w.im / b)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_z_plus_two_over_z_traces_ellipse_l517_51794


namespace NUMINAMATH_CALUDE_solution_product_l517_51725

theorem solution_product (r s : ℝ) : 
  (r - 3) * (3 * r + 11) = r^2 - 16 * r + 52 →
  (s - 3) * (3 * s + 11) = s^2 - 16 * s + 52 →
  r ≠ s →
  (r + 4) * (s + 4) = -62.5 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l517_51725


namespace NUMINAMATH_CALUDE_alternating_arrangements_adjacent_ab_arrangements_l517_51778

/-- Represents the number of male students -/
def num_male : Nat := 2

/-- Represents the number of female students -/
def num_female : Nat := 3

/-- Represents the total number of students -/
def total_students : Nat := num_male + num_female

/-- Calculates the number of ways to arrange n distinct objects -/
def arrangements (n : Nat) : Nat := Nat.factorial n

/-- Theorem stating the number of alternating arrangements -/
theorem alternating_arrangements : 
  arrangements num_male * arrangements num_female = 12 := by sorry

/-- Theorem stating the number of arrangements with A and B adjacent -/
theorem adjacent_ab_arrangements : 
  arrangements (total_students - 1) * 2 = 48 := by sorry

end NUMINAMATH_CALUDE_alternating_arrangements_adjacent_ab_arrangements_l517_51778


namespace NUMINAMATH_CALUDE_heights_academy_music_problem_l517_51737

/-- The Heights Academy music problem -/
theorem heights_academy_music_problem
  (total_students : ℕ)
  (females_band : ℕ)
  (males_band : ℕ)
  (females_orchestra : ℕ)
  (males_orchestra : ℕ)
  (females_both : ℕ)
  (h1 : total_students = 260)
  (h2 : females_band = 120)
  (h3 : males_band = 90)
  (h4 : females_orchestra = 100)
  (h5 : males_orchestra = 130)
  (h6 : females_both = 80) :
  males_band - (males_band + males_orchestra - (total_students - (females_band + females_orchestra - females_both))) = 30 := by
  sorry


end NUMINAMATH_CALUDE_heights_academy_music_problem_l517_51737


namespace NUMINAMATH_CALUDE_filter_kit_cost_difference_l517_51793

/-- Proves that buying the camera lens filter kit costs more than buying filters individually -/
theorem filter_kit_cost_difference : 
  let kit_price : ℚ := 87.5
  let filter_price_1 : ℚ := 16.45
  let filter_price_2 : ℚ := 14.05
  let filter_price_3 : ℚ := 19.5
  let individual_total : ℚ := 2 * filter_price_1 + 2 * filter_price_2 + filter_price_3
  kit_price - individual_total = 7 :=
by sorry

end NUMINAMATH_CALUDE_filter_kit_cost_difference_l517_51793


namespace NUMINAMATH_CALUDE_factor_proof_l517_51730

theorem factor_proof (x y z : ℝ) : 
  ∃ (k : ℝ), x^2 - y^2 - z^2 + 2*y*z + x - y - z + 2 = (x - y + z + 1) * k :=
by sorry

end NUMINAMATH_CALUDE_factor_proof_l517_51730


namespace NUMINAMATH_CALUDE_smallest_positive_integer_2002m_44444n_l517_51752

theorem smallest_positive_integer_2002m_44444n : 
  (∃ (k : ℕ+), ∀ (a : ℕ+), (∃ (m n : ℤ), a.val = 2002 * m + 44444 * n) → k ≤ a) ∧ 
  (∃ (m n : ℤ), (2 : ℕ+).val = 2002 * m + 44444 * n) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_2002m_44444n_l517_51752


namespace NUMINAMATH_CALUDE_percent_relationship_l517_51798

theorem percent_relationship (x y z : ℝ) (h1 : x = 1.20 * y) (h2 : y = 0.70 * z) :
  x = 0.84 * z := by sorry

end NUMINAMATH_CALUDE_percent_relationship_l517_51798


namespace NUMINAMATH_CALUDE_window_area_l517_51762

-- Define the number of panes
def num_panes : ℕ := 8

-- Define the length of each pane in inches
def pane_length : ℕ := 12

-- Define the width of each pane in inches
def pane_width : ℕ := 8

-- Theorem statement
theorem window_area :
  (num_panes * pane_length * pane_width) = 768 := by
  sorry

end NUMINAMATH_CALUDE_window_area_l517_51762


namespace NUMINAMATH_CALUDE_yearly_calls_cost_is_78_l517_51721

/-- The total cost of weekly calls for a year -/
def total_cost_yearly_calls (weeks_per_year : ℕ) (call_duration_minutes : ℕ) (cost_per_minute : ℚ) : ℚ :=
  (weeks_per_year : ℚ) * (call_duration_minutes : ℚ) * cost_per_minute

/-- Theorem stating that the total cost for a year of weekly calls is $78 -/
theorem yearly_calls_cost_is_78 :
  total_cost_yearly_calls 52 30 (5 / 100) = 78 := by
  sorry

end NUMINAMATH_CALUDE_yearly_calls_cost_is_78_l517_51721


namespace NUMINAMATH_CALUDE_chosen_number_l517_51765

theorem chosen_number (x : ℝ) : (x / 6) - 15 = 5 → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l517_51765


namespace NUMINAMATH_CALUDE_common_days_off_l517_51783

/-- Earl's work cycle in days -/
def earl_cycle : ℕ := 4

/-- Bob's work cycle in days -/
def bob_cycle : ℕ := 10

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Number of common rest days in one LCM period -/
def common_rest_days_per_lcm : ℕ := 2

/-- Theorem stating the number of common days off for Earl and Bob -/
theorem common_days_off : ℕ := by
  sorry

end NUMINAMATH_CALUDE_common_days_off_l517_51783


namespace NUMINAMATH_CALUDE_population_falls_below_threshold_l517_51705

/-- The annual decrease rate of the finch population -/
def annual_decrease_rate : ℝ := 0.3

/-- The threshold below which we consider the population to have significantly decreased -/
def threshold : ℝ := 0.15

/-- The function that calculates the population after a given number of years -/
def population_after_years (initial_population : ℝ) (years : ℕ) : ℝ :=
  initial_population * (1 - annual_decrease_rate) ^ years

/-- Theorem stating that it takes 6 years for the population to fall below the threshold -/
theorem population_falls_below_threshold (initial_population : ℝ) (h : initial_population > 0) :
  population_after_years initial_population 6 < threshold * initial_population ∧
  population_after_years initial_population 5 ≥ threshold * initial_population :=
by sorry

end NUMINAMATH_CALUDE_population_falls_below_threshold_l517_51705


namespace NUMINAMATH_CALUDE_angle_C_in_right_triangle_l517_51790

-- Define a right triangle ABC
structure RightTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  right_angle : A = 90
  angle_sum : A + B + C = 180

-- Theorem statement
theorem angle_C_in_right_triangle (t : RightTriangle) (h : t.B = 50) : t.C = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_in_right_triangle_l517_51790


namespace NUMINAMATH_CALUDE_expand_polynomial_l517_51720

theorem expand_polynomial (x : ℝ) : (5*x^2 + 3*x - 4) * 3*x^3 = 15*x^5 + 9*x^4 - 12*x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l517_51720


namespace NUMINAMATH_CALUDE_incorrect_negation_l517_51776

theorem incorrect_negation : 
  ¬(¬(∀ x : ℝ, x^2 - x = 0 → x = 0 ∨ x = 1) ↔ 
    (∀ x : ℝ, x^2 - x = 0 → x ≠ 0 ∧ x ≠ 1)) := by sorry

end NUMINAMATH_CALUDE_incorrect_negation_l517_51776


namespace NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l517_51700

/-- Represents a system of two linear equations with two unknowns,
    where the coefficients form an arithmetic progression. -/
structure ArithmeticProgressionSystem where
  a : ℝ
  d : ℝ

/-- The solution to the system of linear equations. -/
def solution : ℝ × ℝ := (-1, 2)

/-- Checks if the given pair (x, y) satisfies the first equation of the system. -/
def satisfies_equation1 (sys : ArithmeticProgressionSystem) (sol : ℝ × ℝ) : Prop :=
  sys.a * sol.1 + (sys.a + sys.d) * sol.2 = sys.a + 2 * sys.d

/-- Checks if the given pair (x, y) satisfies the second equation of the system. -/
def satisfies_equation2 (sys : ArithmeticProgressionSystem) (sol : ℝ × ℝ) : Prop :=
  (sys.a + 3 * sys.d) * sol.1 + (sys.a + 4 * sys.d) * sol.2 = sys.a + 5 * sys.d

/-- Theorem stating that the solution satisfies both equations of the system. -/
theorem solution_satisfies_system (sys : ArithmeticProgressionSystem) :
  satisfies_equation1 sys solution ∧ satisfies_equation2 sys solution :=
sorry

/-- Theorem stating that the solution is unique. -/
theorem solution_is_unique (sys : ArithmeticProgressionSystem) (sol : ℝ × ℝ) :
  satisfies_equation1 sys sol ∧ satisfies_equation2 sys sol → sol = solution :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l517_51700


namespace NUMINAMATH_CALUDE_three_numbers_sum_and_ratio_l517_51743

theorem three_numbers_sum_and_ratio (A B C : ℝ) : 
  A + B + C = 36 →
  (A + B) / (B + C) = 2 / 3 →
  (B + C) / (A + C) = 3 / 4 →
  A = 12 ∧ B = 4 ∧ C = 20 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_and_ratio_l517_51743


namespace NUMINAMATH_CALUDE_conditional_probability_l517_51731

/-- The total number of products in the box -/
def total_products : ℕ := 4

/-- The number of first-class products in the box -/
def first_class_products : ℕ := 3

/-- The number of second-class products in the box -/
def second_class_products : ℕ := 1

/-- Event A: "the first draw is a first-class product" -/
def event_A : Set ℕ := {1, 2, 3}

/-- Event B: "the second draw is a first-class product" -/
def event_B : Set ℕ := {1, 2}

/-- The probability of event A -/
def prob_A : ℚ := first_class_products / total_products

/-- The probability of event B given event A has occurred -/
def prob_B_given_A : ℚ := (first_class_products - 1) / (total_products - 1)

/-- The conditional probability of event B given event A -/
theorem conditional_probability :
  prob_B_given_A = 2/3 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_l517_51731


namespace NUMINAMATH_CALUDE_M_intersect_N_l517_51763

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 = x}

theorem M_intersect_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l517_51763


namespace NUMINAMATH_CALUDE_min_sum_of_a_and_b_l517_51745

/-- Given a line x/a + y/b = 1 where a > 0 and b > 0, and the line passes through (2, 2),
    the minimum value of a + b is 8. -/
theorem min_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : 2/a + 2/b = 1) : 
  ∀ (x y : ℝ), x/a + y/b = 1 → a + b ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ + b₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_a_and_b_l517_51745


namespace NUMINAMATH_CALUDE_cyclist_round_trip_l517_51786

/-- A cyclist's round trip with given conditions -/
theorem cyclist_round_trip (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (distance1 : ℝ) (distance2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = distance1 + distance2)
  (h2 : distance1 = 12)
  (h3 : distance2 = 24)
  (h4 : speed1 = 8)
  (h5 : speed2 = 12)
  (h6 : total_time = 7.5) :
  (2 * total_distance) / (total_time - (distance1 / speed1 + distance2 / speed2)) = 9 := by
sorry

end NUMINAMATH_CALUDE_cyclist_round_trip_l517_51786


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_three_l517_51707

theorem absolute_value_of_negative_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_three_l517_51707


namespace NUMINAMATH_CALUDE_cubic_function_properties_l517_51704

-- Define the cubic function
def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 + b*x + c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem cubic_function_properties (a b c : ℝ) :
  -- Part 1: If there exists a point where the tangent line is parallel to the x-axis
  (∃ x : ℝ, f' a b x = 0) →
  a^2 ≥ 3*b ∧
  -- Part 2: If f(x) has extreme values at x = -1 and x = 3
  (f' a b (-1) = 0 ∧ f' a b 3 = 0) →
  a = 3 ∧ b = -9 ∧
  -- Part 3: If f(x) < 2c for all x ∈ [-2, 6], then c > 54
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 6 → f a b c x < 2*c) →
  c > 54 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l517_51704


namespace NUMINAMATH_CALUDE_functional_equation_solution_l517_51710

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f x * f y) = x * f (x + y)

/-- The theorem stating that any function satisfying the functional equation
    must be one of the three specified functions -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) ∨ (∀ x, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l517_51710


namespace NUMINAMATH_CALUDE_hyperbola_focus_l517_51723

/-- Definition of the hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  -x^2 + 2*y^2 - 10*x - 16*y + 1 = 0

/-- Theorem stating that one of the foci of the hyperbola is at (-5, 7) or (-5, 1) -/
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ ((x = -5 ∧ y = 7) ∨ (x = -5 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l517_51723


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l517_51792

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 + 3*y^2 - 6*x - 12*y + 9 = 0

/-- The standard form of an ellipse equation -/
def is_ellipse (h k a b : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem stating that the given equation represents an ellipse -/
theorem conic_is_ellipse :
  ∃ (h k a b : ℝ), ∀ (x y : ℝ),
    conic_equation x y ↔ is_ellipse h k a b x y :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l517_51792


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l517_51787

theorem square_ratio_side_length (area_ratio : ℚ) : 
  area_ratio = 75 / 128 →
  ∃ (a b c : ℕ), 
    (a = 5 ∧ b = 6 ∧ c = 16) ∧
    (Real.sqrt area_ratio * c = a * Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l517_51787


namespace NUMINAMATH_CALUDE_cube_ending_with_ones_l517_51775

theorem cube_ending_with_ones (k : ℕ) : ∃ n : ℤ, ∃ m : ℕ, n^3 = m * 10^k + (10^k - 1) := by
  sorry

end NUMINAMATH_CALUDE_cube_ending_with_ones_l517_51775


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l517_51701

-- Define the points
variable (A B C D E F : EuclideanSpace ℝ (Fin 2))

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define E as the intersection of angle bisectors of ∠B and ∠C
def is_angle_bisector_intersection (E B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define F as the intersection of AB and CD
def is_line_intersection (F A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the condition AB + CD = BC
def sum_equals_side (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := 
  dist A B + dist C D = dist B C

-- Define cyclic quadrilateral
def is_cyclic (A D E F : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Theorem statement
theorem cyclic_quadrilateral_theorem 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_angle_bisector_intersection E B C)
  (h3 : is_line_intersection F A B C D)
  (h4 : sum_equals_side A B C D) :
  is_cyclic A D E F := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l517_51701


namespace NUMINAMATH_CALUDE_conference_tables_needed_l517_51784

-- Define the base 7 number
def base7_number : ℕ := 312

-- Define the base conversion function
def base7_to_decimal (n : ℕ) : ℕ :=
  3 * 7^2 + 1 * 7^1 + 2 * 7^0

-- Define the number of attendees per table
def attendees_per_table : ℕ := 3

-- Theorem statement
theorem conference_tables_needed :
  (base7_to_decimal base7_number) / attendees_per_table = 52 := by
  sorry

end NUMINAMATH_CALUDE_conference_tables_needed_l517_51784


namespace NUMINAMATH_CALUDE_intersection_in_second_quadrant_l517_51736

/-- 
If the intersection point of the lines y = 2x + 4 and y = -2x + m 
is in the second quadrant, then -4 < m < 4.
-/
theorem intersection_in_second_quadrant (m : ℝ) : 
  (∃ x y : ℝ, y = 2*x + 4 ∧ y = -2*x + m ∧ x < 0 ∧ y > 0) → 
  -4 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_second_quadrant_l517_51736


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_five_plus_sqrt_two_l517_51744

theorem sum_of_fractions_equals_five_plus_sqrt_two :
  let S := 1 / (5 - Real.sqrt 19) - 1 / (Real.sqrt 19 - Real.sqrt 18) + 
           1 / (Real.sqrt 18 - Real.sqrt 17) - 1 / (Real.sqrt 17 - 3) + 
           1 / (3 - Real.sqrt 2)
  S = 5 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_five_plus_sqrt_two_l517_51744


namespace NUMINAMATH_CALUDE_entertainment_unit_theorem_l517_51758

/-- A structure representing the entertainment unit -/
structure EntertainmentUnit where
  singers : ℕ
  dancers : ℕ
  total : ℕ
  both : ℕ
  h_singers : singers = 4
  h_dancers : dancers = 5
  h_total : total = singers + dancers - both
  h_all_can : total ≤ singers + dancers

/-- The probability of selecting at least one person who can both sing and dance -/
def prob_at_least_one (u : EntertainmentUnit) : ℚ :=
  1 - (Nat.choose (u.total - u.both) 2 : ℚ) / (Nat.choose u.total 2 : ℚ)

/-- The probability distribution of ξ -/
def prob_dist (u : EntertainmentUnit) : ℕ → ℚ
| 0 => (Nat.choose (u.total - u.both) 2 : ℚ) / (Nat.choose u.total 2 : ℚ)
| 1 => (u.both * (u.total - u.both) : ℚ) / (Nat.choose u.total 2 : ℚ)
| 2 => (Nat.choose u.both 2 : ℚ) / (Nat.choose u.total 2 : ℚ)
| _ => 0

/-- The expected value of ξ -/
def expected_value (u : EntertainmentUnit) : ℚ :=
  0 * prob_dist u 0 + 1 * prob_dist u 1 + 2 * prob_dist u 2

/-- The main theorem -/
theorem entertainment_unit_theorem (u : EntertainmentUnit) 
  (h_prob : prob_at_least_one u = 11/21) : 
  u.total = 7 ∧ 
  prob_dist u 0 = 10/21 ∧ 
  prob_dist u 1 = 10/21 ∧ 
  prob_dist u 2 = 1/21 ∧
  expected_value u = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_entertainment_unit_theorem_l517_51758


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_for_specific_pyramid_l517_51713

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_edge : ℝ
  side_edge : ℝ

/-- Radius of the circumscribed sphere of a regular triangular pyramid -/
def circumscribed_sphere_radius (p : RegularTriangularPyramid) : ℝ :=
  -- Definition to be proved
  sorry

/-- Theorem: The radius of the circumscribed sphere of a regular triangular pyramid
    with base edge 6 and side edge 4 is 4 -/
theorem circumscribed_sphere_radius_for_specific_pyramid :
  let p : RegularTriangularPyramid := ⟨6, 4⟩
  circumscribed_sphere_radius p = 4 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_for_specific_pyramid_l517_51713


namespace NUMINAMATH_CALUDE_trig_identity_l517_51797

theorem trig_identity (α : Real) (h : (1 + Real.tan α) / (1 - Real.tan α) = 2012) :
  1 / Real.cos (2 * α) + Real.tan (2 * α) = 2012 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l517_51797


namespace NUMINAMATH_CALUDE_function_satisfying_equation_l517_51774

theorem function_satisfying_equation (r s : ℚ) :
  ∀ f : ℚ → ℚ, (∀ x y : ℚ, f (x + f y) = f (x + r) + y + s) →
  (∀ x : ℚ, f x = x + r + s) ∨ (∀ x : ℚ, f x = -x + r - s) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_equation_l517_51774


namespace NUMINAMATH_CALUDE_abs_inequality_l517_51703

theorem abs_inequality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l517_51703


namespace NUMINAMATH_CALUDE_distance_A_B_l517_51767

/-- The distance between points A(0, 0, 1) and B(0, 1, 0) in a spatial Cartesian coordinate system is √2. -/
theorem distance_A_B : Real.sqrt 2 = (Real.sqrt ((0 - 0)^2 + (1 - 0)^2 + (0 - 1)^2)) := by
  sorry

end NUMINAMATH_CALUDE_distance_A_B_l517_51767


namespace NUMINAMATH_CALUDE_sandys_phone_bill_l517_51799

theorem sandys_phone_bill (kim_age : ℕ) (sandy_age : ℕ) (sandy_bill : ℕ) : 
  kim_age = 10 →
  sandy_age + 2 = 3 * (kim_age + 2) →
  sandy_bill = 10 * sandy_age →
  sandy_bill = 340 :=
by
  sorry

end NUMINAMATH_CALUDE_sandys_phone_bill_l517_51799


namespace NUMINAMATH_CALUDE_rectangle_length_problem_l517_51727

theorem rectangle_length_problem (b : ℝ) (h1 : b > 0) : 
  (2 * b - 5) * (b + 5) - 2 * b * b = 75 → 2 * b = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_problem_l517_51727


namespace NUMINAMATH_CALUDE_water_depth_in_cylinder_l517_51764

/-- Represents the depth of water in a horizontal cylindrical tank. -/
def water_depth (tank_length tank_diameter water_surface_area : ℝ) : Set ℝ :=
  {h : ℝ | ∃ (w : ℝ), 
    tank_length > 0 ∧ 
    tank_diameter > 0 ∧ 
    water_surface_area > 0 ∧
    w > 0 ∧ 
    h > 0 ∧ 
    h < tank_diameter ∧
    w * tank_length = water_surface_area ∧
    w = 2 * Real.sqrt (tank_diameter * h - h^2)}

/-- The main theorem stating the depth of water in the given cylindrical tank. -/
theorem water_depth_in_cylinder : 
  water_depth 12 4 24 = {2 - Real.sqrt 3, 2 + Real.sqrt 3} := by
  sorry


end NUMINAMATH_CALUDE_water_depth_in_cylinder_l517_51764


namespace NUMINAMATH_CALUDE_mean_temperature_is_negative_point_six_l517_51747

def temperatures : List ℝ := [-8, -5, -5, -2, 0, 4, 5, 3, 6, 1]

theorem mean_temperature_is_negative_point_six :
  (temperatures.sum / temperatures.length : ℝ) = -0.6 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_negative_point_six_l517_51747


namespace NUMINAMATH_CALUDE_xyz_inequality_l517_51739

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z ≥ x * y + y * z + z * x) : 
  Real.sqrt (x * y * z) ≥ Real.sqrt x + Real.sqrt y + Real.sqrt z := by
sorry

end NUMINAMATH_CALUDE_xyz_inequality_l517_51739


namespace NUMINAMATH_CALUDE_fourth_root_equation_l517_51753

theorem fourth_root_equation (x : ℝ) (h : x > 0) :
  (x^3 * x^(1/2))^(1/4) = 4 → x = 4^(8/7) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_l517_51753


namespace NUMINAMATH_CALUDE_max_sum_of_integers_l517_51777

theorem max_sum_of_integers (a c d : ℤ) (b : ℕ+) 
  (eq1 : a + b = c) 
  (eq2 : b + c = d) 
  (eq3 : c + d = a) : 
  a + b + c + d ≤ -5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_integers_l517_51777


namespace NUMINAMATH_CALUDE_mass_percentage_K_is_23_81_l517_51795

/-- The mass percentage of K in a compound -/
def mass_percentage_K : ℝ := 23.81

/-- Theorem stating that the mass percentage of K in the compound is 23.81% -/
theorem mass_percentage_K_is_23_81 :
  mass_percentage_K = 23.81 := by sorry

end NUMINAMATH_CALUDE_mass_percentage_K_is_23_81_l517_51795


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l517_51714

theorem shaded_area_calculation (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 40 →
  triangle_base = 15 →
  triangle_height = 15 →
  square_side * square_side - 2 * (1/2 * triangle_base * triangle_height) = 1375 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l517_51714


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l517_51781

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  /-- The length of the altitude to the base -/
  altitude : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True

/-- The area of an isosceles triangle -/
def triangle_area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area : 
  ∀ (t : IsoscelesTriangle), t.altitude = 10 ∧ t.perimeter = 40 → triangle_area t = 75 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l517_51781


namespace NUMINAMATH_CALUDE_no_square_in_triangle_grid_l517_51756

/-- A point in the plane represented by its coordinates -/
structure Point where
  x : ℚ
  y : ℝ

/-- The grid of equilateral triangles -/
structure TriangleGrid where
  side_length : ℝ
  is_valid : side_length > 0

/-- A function that checks if a point is a valid vertex in the triangle grid -/
def is_vertex (grid : TriangleGrid) (p : Point) : Prop :=
  ∃ (k l : ℤ), p.x = k * (grid.side_length / 2) ∧ p.y = l * (Real.sqrt 3 * grid.side_length / 2)

/-- A function that checks if four points form a square -/
def is_square (a b c d : Point) : Prop :=
  let dist (p q : Point) := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)
  (dist a b = dist b c) ∧ (dist b c = dist c d) ∧ (dist c d = dist d a) ∧
  (dist a c = dist b d) ∧
  ((b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y) = 0)

/-- The main theorem stating that it's impossible to form a square from vertices of the triangle grid -/
theorem no_square_in_triangle_grid (grid : TriangleGrid) :
  ¬∃ (a b c d : Point), is_vertex grid a ∧ is_vertex grid b ∧ is_vertex grid c ∧ is_vertex grid d ∧ is_square a b c d :=
sorry

end NUMINAMATH_CALUDE_no_square_in_triangle_grid_l517_51756


namespace NUMINAMATH_CALUDE_octagon_side_length_l517_51766

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (P Q R S : Point)

/-- Represents an octagon -/
structure Octagon :=
  (A B C D E F G H : Point)

/-- Checks if an octagon is equilateral -/
def is_equilateral (oct : Octagon) : Prop := sorry

/-- Checks if an octagon is convex -/
def is_convex (oct : Octagon) : Prop := sorry

/-- Checks if an octagon is inscribed in a rectangle -/
def is_inscribed (oct : Octagon) (rect : Rectangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_square_of_prime (n : ℕ) : Prop := sorry

theorem octagon_side_length (rect : Rectangle) (oct : Octagon) :
  distance rect.P rect.Q = 8 →
  distance rect.Q rect.R = 6 →
  is_inscribed oct rect →
  is_equilateral oct →
  is_convex oct →
  distance oct.A rect.P = distance oct.B rect.Q →
  distance oct.A rect.P < 4 →
  ∃ (k m n : ℕ), 
    distance oct.A oct.B = k + m * Real.sqrt n ∧ 
    not_divisible_by_square_of_prime n ∧
    k + m + n = 7 :=
by sorry

end NUMINAMATH_CALUDE_octagon_side_length_l517_51766


namespace NUMINAMATH_CALUDE_gold_quarter_weight_l517_51749

/-- The weight of a gold quarter in ounces -/
def quarter_weight : ℝ := 0.2

/-- The value of a quarter in dollars when spent in a store -/
def quarter_store_value : ℝ := 0.25

/-- The value of an ounce of melted gold in dollars -/
def melted_gold_value_per_ounce : ℝ := 100

/-- The ratio of melted value to store value -/
def melted_to_store_ratio : ℕ := 80

theorem gold_quarter_weight :
  quarter_weight * melted_gold_value_per_ounce = melted_to_store_ratio * quarter_store_value := by
  sorry

end NUMINAMATH_CALUDE_gold_quarter_weight_l517_51749


namespace NUMINAMATH_CALUDE_cubic_sum_l517_51715

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_l517_51715
