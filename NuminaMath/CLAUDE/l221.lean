import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_results_l221_22111

theorem stratified_sampling_results (total_sample : ℕ) (junior_students senior_students : ℕ) :
  total_sample = 60 ∧ junior_students = 400 ∧ senior_students = 200 →
  (Nat.choose junior_students ((total_sample * junior_students) / (junior_students + senior_students))) *
  (Nat.choose senior_students ((total_sample * senior_students) / (junior_students + senior_students))) =
  Nat.choose 400 40 * Nat.choose 200 20 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_results_l221_22111


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l221_22172

theorem shaded_fraction_of_rectangle (rectangle_length rectangle_width : ℝ) 
  (h_length : rectangle_length = 15)
  (h_width : rectangle_width = 20)
  (h_triangle_area : ∃ (triangle_area : ℝ), triangle_area = (1/3) * rectangle_length * rectangle_width)
  (h_shaded_area : ∃ (shaded_area : ℝ), shaded_area = (1/2) * (1/3) * rectangle_length * rectangle_width) :
  (∃ (shaded_area : ℝ), shaded_area = (1/6) * rectangle_length * rectangle_width) :=
by sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l221_22172


namespace NUMINAMATH_CALUDE_abc_product_is_one_l221_22158

theorem abc_product_is_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h1 : a + 1 / b^2 = b + 1 / c^2) (h2 : b + 1 / c^2 = c + 1 / a^2) :
  |a * b * c| = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_product_is_one_l221_22158


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l221_22138

/-- The first line parameterized by t -/
def line1 (t : ℚ) : ℚ × ℚ := (3 - t, 2 + 4*t)

/-- The second line parameterized by u -/
def line2 (u : ℚ) : ℚ × ℚ := (-1 + 3*u, 3 + 5*u)

/-- The proposed intersection point -/
def intersection_point : ℚ × ℚ := (39/17, 74/17)

theorem lines_intersect_at_point :
  ∃! (t u : ℚ), line1 t = line2 u ∧ line1 t = intersection_point :=
sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l221_22138


namespace NUMINAMATH_CALUDE_vendor_drink_problem_l221_22182

theorem vendor_drink_problem (maaza sprite cans : ℕ) (pepsi : ℕ) : 
  maaza = 50 →
  sprite = 368 →
  cans = 281 →
  (maaza + sprite + pepsi) % cans = 0 →
  pepsi = 144 :=
by sorry

end NUMINAMATH_CALUDE_vendor_drink_problem_l221_22182


namespace NUMINAMATH_CALUDE_subset_count_l221_22179

theorem subset_count : ℕ := by
  -- Define the universal set U
  let U : Finset ℕ := {1, 2, 3, 4, 5, 6}
  
  -- Define the required subset A
  let A : Finset ℕ := {1, 2, 3}
  
  -- Define the count of subsets X such that A ⊆ X ⊆ U
  let count := Finset.filter (fun X => A ⊆ X) U.powerset |>.card
  
  -- Assert that this count is equal to 8
  have h : count = 8 := by sorry
  
  -- Return the result
  exact 8

end NUMINAMATH_CALUDE_subset_count_l221_22179


namespace NUMINAMATH_CALUDE_dani_pants_after_five_years_l221_22174

/-- The number of pants Dani will have after a given number of years -/
def total_pants (initial_pants : ℕ) (pairs_per_year : ℕ) (years : ℕ) : ℕ :=
  initial_pants + pairs_per_year * 2 * years

/-- Theorem stating that Dani will have 90 pants after 5 years -/
theorem dani_pants_after_five_years :
  total_pants 50 4 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_dani_pants_after_five_years_l221_22174


namespace NUMINAMATH_CALUDE_seventeen_flavors_l221_22146

/-- Represents the number of different flavors possible given blue and orange candies. -/
def number_of_flavors (blue : ℕ) (orange : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that given 5 blue candies and 4 orange candies, 
    the number of different possible flavors is 17. -/
theorem seventeen_flavors : number_of_flavors 5 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_flavors_l221_22146


namespace NUMINAMATH_CALUDE_max_area_rectangle_l221_22115

/-- The parabola function y = 4 - x^2 --/
def parabola (x : ℝ) : ℝ := 4 - x^2

/-- The area function of the rectangle --/
def area (x : ℝ) : ℝ := 2 * x * (4 - x^2)

/-- The theorem stating the maximum area of the rectangle --/
theorem max_area_rectangle :
  ∃ (x : ℝ), x > 0 ∧ x < 2 ∧
  (∀ (y : ℝ), y > 0 ∧ y < 2 → area x ≥ area y) ∧
  (2 * x = (4 / 3) * Real.sqrt 3) := by
  sorry

#check max_area_rectangle

end NUMINAMATH_CALUDE_max_area_rectangle_l221_22115


namespace NUMINAMATH_CALUDE_larger_number_l221_22117

theorem larger_number (x y : ℝ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l221_22117


namespace NUMINAMATH_CALUDE_max_product_constraint_l221_22163

theorem max_product_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_constraint : 5 * x + 8 * y + 3 * z = 90) :
  x * y * z ≤ 225 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    5 * x₀ + 8 * y₀ + 3 * z₀ = 90 ∧ x₀ * y₀ * z₀ = 225 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l221_22163


namespace NUMINAMATH_CALUDE_woodwind_to_brass_ratio_l221_22132

/-- Represents the composition of a marching band -/
structure MarchingBand where
  total : ℕ
  percussion : ℕ
  woodwind : ℕ
  brass : ℕ

/-- Checks if the marching band satisfies the given conditions -/
def validBand (band : MarchingBand) : Prop :=
  band.total = 110 ∧
  band.percussion = 4 * band.woodwind ∧
  band.brass = 10 ∧
  band.total = band.percussion + band.woodwind + band.brass

/-- Theorem stating the ratio of woodwind to brass players -/
theorem woodwind_to_brass_ratio (band : MarchingBand) 
  (h : validBand band) : 
  band.woodwind = 2 * band.brass :=
sorry

end NUMINAMATH_CALUDE_woodwind_to_brass_ratio_l221_22132


namespace NUMINAMATH_CALUDE_power_equation_solution_l221_22191

theorem power_equation_solution (m : ℕ) : 5^m = 5 * 25^2 * 125^3 → m = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l221_22191


namespace NUMINAMATH_CALUDE_east_west_convention_l221_22189

-- Define the direction type
inductive Direction
| West
| East

-- Define a function to convert distance and direction to a signed number
def signedDistance (dist : ℝ) (dir : Direction) : ℝ :=
  match dir with
  | Direction.West => dist
  | Direction.East => -dist

-- State the theorem
theorem east_west_convention (westDistance : ℝ) (eastDistance : ℝ) :
  westDistance > 0 →
  signedDistance westDistance Direction.West = westDistance →
  signedDistance eastDistance Direction.East = -eastDistance :=
by
  sorry

-- Example with the given values
example : signedDistance 3 Direction.East = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_east_west_convention_l221_22189


namespace NUMINAMATH_CALUDE_horner_method_for_f_at_3_l221_22170

/-- Horner's method for a polynomial of degree 4 -/
def horner_method (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₄ * x + a₃) * x + a₂) * x + a₁) * x + a₀)

/-- The polynomial f(x) = 2x⁴ - x³ + 3x² + 7 -/
def f (x : ℝ) : ℝ := 2 * x^4 - x^3 + 3 * x^2 + 7

theorem horner_method_for_f_at_3 :
  horner_method 2 (-1) 3 0 7 3 = f 3 ∧ horner_method 2 (-1) 3 0 7 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_at_3_l221_22170


namespace NUMINAMATH_CALUDE_sues_waiting_time_l221_22113

/-- Proves that Sue's waiting time in New York is 16 hours given the travel conditions -/
theorem sues_waiting_time (total_time : ℝ) (ny_to_sf_time : ℝ) (no_to_ny_ratio : ℝ) 
  (h1 : total_time = 58)
  (h2 : ny_to_sf_time = 24)
  (h3 : no_to_ny_ratio = 3/4)
  : total_time - (no_to_ny_ratio * ny_to_sf_time) - ny_to_sf_time = 16 := by
  sorry

#check sues_waiting_time

end NUMINAMATH_CALUDE_sues_waiting_time_l221_22113


namespace NUMINAMATH_CALUDE_large_painting_area_is_150_l221_22175

/-- Represents Davonte's art collection --/
structure ArtCollection where
  square_paintings : Nat
  small_paintings : Nat
  large_painting : Nat
  square_side : Nat
  small_width : Nat
  small_height : Nat
  total_area : Nat

/-- Calculates the area of the large painting in Davonte's collection --/
def large_painting_area (collection : ArtCollection) : Nat :=
  collection.total_area -
  (collection.square_paintings * collection.square_side * collection.square_side +
   collection.small_paintings * collection.small_width * collection.small_height)

/-- Theorem stating that the area of the large painting is 150 square feet --/
theorem large_painting_area_is_150 (collection : ArtCollection)
  (h1 : collection.square_paintings = 3)
  (h2 : collection.small_paintings = 4)
  (h3 : collection.square_side = 6)
  (h4 : collection.small_width = 2)
  (h5 : collection.small_height = 3)
  (h6 : collection.total_area = 282) :
  large_painting_area collection = 150 := by
  sorry

#eval large_painting_area { square_paintings := 3, small_paintings := 4, large_painting := 1,
                            square_side := 6, small_width := 2, small_height := 3, total_area := 282 }

end NUMINAMATH_CALUDE_large_painting_area_is_150_l221_22175


namespace NUMINAMATH_CALUDE_table_movement_l221_22193

theorem table_movement (table_length table_width : ℝ) 
  (hl : table_length = 12) (hw : table_width = 9) : 
  let diagonal := Real.sqrt (table_length^2 + table_width^2)
  ∀ L W : ℕ, 
    (L ≥ diagonal ∧ W ≥ diagonal ∧ L ≥ table_length) → 
    (∀ L' W' : ℕ, (L' < L ∨ W' < W) → 
      ¬(L' ≥ diagonal ∧ W' ≥ diagonal ∧ L' ≥ table_length)) → 
    L = 15 ∧ W = 15 :=
by sorry

end NUMINAMATH_CALUDE_table_movement_l221_22193


namespace NUMINAMATH_CALUDE_triangle_nth_part_area_l221_22104

theorem triangle_nth_part_area (b h n : ℝ) (h_pos : 0 < h) (n_pos : 0 < n) :
  let original_area := (1 / 2) * b * h
  let cut_height := h / Real.sqrt n
  let cut_area := (1 / 2) * b * cut_height
  cut_area = (1 / n) * original_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_nth_part_area_l221_22104


namespace NUMINAMATH_CALUDE_set_operations_l221_22141

def A (x : ℝ) : Set ℝ := {0, |x|}
def B : Set ℝ := {1, 0, -1}

theorem set_operations (x : ℝ) (h : A x ⊆ B) :
  (A x ∩ B = {0, 1}) ∧
  (A x ∪ B = {-1, 0, 1}) ∧
  (B \ A x = {-1}) := by
sorry

end NUMINAMATH_CALUDE_set_operations_l221_22141


namespace NUMINAMATH_CALUDE_square_area_is_169_l221_22145

/-- Square with intersecting segments --/
structure SquareWithIntersection where
  -- Side length of the square
  s : ℝ
  -- Length of BR
  br : ℝ
  -- Length of PR
  pr : ℝ
  -- Length of CQ
  cq : ℝ
  -- Conditions
  br_positive : br > 0
  pr_positive : pr > 0
  cq_positive : cq > 0
  right_angle : True  -- Represents that BP and CQ intersect at right angles
  br_eq : br = 8
  pr_eq : pr = 5
  cq_eq : cq = 12

/-- The area of the square is 169 --/
theorem square_area_is_169 (square : SquareWithIntersection) : square.s^2 = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_169_l221_22145


namespace NUMINAMATH_CALUDE_chicken_adventure_feathers_l221_22107

/-- Calculates the number of feathers remaining after a chicken's thrill-seeking adventure. -/
def remaining_feathers (initial_feathers : ℕ) (cars_dodged : ℕ) : ℕ :=
  initial_feathers - 2 * cars_dodged

/-- Theorem stating the number of feathers remaining after the chicken's adventure. -/
theorem chicken_adventure_feathers :
  remaining_feathers 5263 23 = 5217 := by
  sorry

#eval remaining_feathers 5263 23

end NUMINAMATH_CALUDE_chicken_adventure_feathers_l221_22107


namespace NUMINAMATH_CALUDE_no_real_roots_iff_m_gt_one_l221_22101

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Theorem statement
theorem no_real_roots_iff_m_gt_one (m : ℝ) :
  (∀ x : ℝ, quadratic x m ≠ 0) ↔ m > 1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_m_gt_one_l221_22101


namespace NUMINAMATH_CALUDE_road_repair_workers_l221_22103

theorem road_repair_workers (group1_people : ℕ) (group1_days : ℕ) (group1_hours : ℕ)
                             (group2_days : ℕ) (group2_hours : ℕ) :
  group1_people = 69 →
  group1_days = 12 →
  group1_hours = 5 →
  group2_days = 23 →
  group2_hours = 6 →
  group1_people * group1_days * group1_hours = 
    ((group1_people * group1_days * group1_hours) / (group2_days * group2_hours) : ℕ) * group2_days * group2_hours →
  ((group1_people * group1_days * group1_hours) / (group2_days * group2_hours) : ℕ) = 30 :=
by sorry

end NUMINAMATH_CALUDE_road_repair_workers_l221_22103


namespace NUMINAMATH_CALUDE_calculate_gratuity_percentage_l221_22171

/-- Calculate the gratuity percentage for a restaurant bill -/
theorem calculate_gratuity_percentage
  (num_people : ℕ)
  (total_bill : ℚ)
  (avg_cost_before_gratuity : ℚ)
  (h_num_people : num_people = 9)
  (h_total_bill : total_bill = 756)
  (h_avg_cost : avg_cost_before_gratuity = 70) :
  (total_bill - num_people * avg_cost_before_gratuity) / (num_people * avg_cost_before_gratuity) = 1/5 :=
sorry

end NUMINAMATH_CALUDE_calculate_gratuity_percentage_l221_22171


namespace NUMINAMATH_CALUDE_find_X_l221_22148

theorem find_X : ∃ X : ℕ, X = 555 * 465 * (3 * (555 - 465)) + (555 - 465)^2 ∧ X = 69688350 := by
  sorry

end NUMINAMATH_CALUDE_find_X_l221_22148


namespace NUMINAMATH_CALUDE_resulting_polygon_has_16_sides_l221_22194

/-- Represents a regular polygon --/
structure RegularPolygon where
  sides : ℕ
  sides_positive : sides > 0

/-- The resulting polygon formed by connecting the given regular polygons --/
def resulting_polygon (triangle square pentagon heptagon hexagon octagon : RegularPolygon) : ℕ :=
  2 + 2 + (4 * 3)

/-- Theorem stating that the resulting polygon has 16 sides --/
theorem resulting_polygon_has_16_sides 
  (triangle : RegularPolygon) 
  (square : RegularPolygon)
  (pentagon : RegularPolygon)
  (heptagon : RegularPolygon)
  (hexagon : RegularPolygon)
  (octagon : RegularPolygon)
  (h1 : triangle.sides = 3)
  (h2 : square.sides = 4)
  (h3 : pentagon.sides = 5)
  (h4 : heptagon.sides = 7)
  (h5 : hexagon.sides = 6)
  (h6 : octagon.sides = 8) :
  resulting_polygon triangle square pentagon heptagon hexagon octagon = 16 := by
  sorry

end NUMINAMATH_CALUDE_resulting_polygon_has_16_sides_l221_22194


namespace NUMINAMATH_CALUDE_square_difference_l221_22143

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 80) (h2 : x * y = 12) : 
  (x - y)^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l221_22143


namespace NUMINAMATH_CALUDE_certain_number_solution_l221_22187

theorem certain_number_solution (x : ℝ) : 
  8 * 5.4 - (x * 10) / 1.2 = 31.000000000000004 → x = 1.464 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_solution_l221_22187


namespace NUMINAMATH_CALUDE_line_increase_percentage_l221_22139

/-- Given an increase of 450 lines resulting in a total of 1350 lines, 
    prove that the percentage increase is 50%. -/
theorem line_increase_percentage (increase : ℕ) (total : ℕ) : 
  increase = 450 → total = 1350 → (increase : ℚ) / ((total - increase) : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_line_increase_percentage_l221_22139


namespace NUMINAMATH_CALUDE_function_decomposition_l221_22169

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (g h : ℝ → ℝ), 
    (∀ x, g (-x) = g x) ∧ 
    (∀ x, h (-x) = -h x) ∧ 
    (∀ x, f x = g x + h x) := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l221_22169


namespace NUMINAMATH_CALUDE_ellipse_equation_l221_22198

/-- The standard equation of an ellipse with given major axis and eccentricity -/
theorem ellipse_equation (major_axis : ℝ) (eccentricity : ℝ) :
  major_axis = 8 ∧ eccentricity = 3/4 →
  ∃ (x y : ℝ), (x^2/16 + y^2/7 = 1) ∨ (x^2/7 + y^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l221_22198


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l221_22118

/-- The distance function of a particle moving in a straight line -/
def s (t : ℝ) : ℝ := 3 * t^2 + t

/-- The instantaneous velocity of the particle at time t -/
def v (t : ℝ) : ℝ := 6 * t + 1

theorem instantaneous_velocity_at_2 : v 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l221_22118


namespace NUMINAMATH_CALUDE_inscribed_square_area_l221_22167

theorem inscribed_square_area (x y : ℝ) (h1 : x = 18) (h2 : y = 30) :
  let s := Real.sqrt ((x * y) / (x + y))
  s ^ 2 = 540 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l221_22167


namespace NUMINAMATH_CALUDE_value_of_fraction_difference_l221_22123

theorem value_of_fraction_difference (x y : ℝ) 
  (hx : x = Real.sqrt 5 - 1) 
  (hy : y = Real.sqrt 5 + 1) : 
  1 / x - 1 / y = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_value_of_fraction_difference_l221_22123


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l221_22130

/-- Given a geometric sequence {a_n} with the specified conditions, 
    prove that the sum of the 6th, 7th, and 8th terms is 32. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 1 + a 2 + a 3 = 1 →                             -- first given condition
  a 2 + a 3 + a 4 = 2 →                             -- second given condition
  a 6 + a 7 + a 8 = 32 :=                           -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l221_22130


namespace NUMINAMATH_CALUDE_f_geq_4_iff_valid_a_range_f_3_geq_4_l221_22188

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 3/a| + |x - a|

def valid_a_range (a : ℝ) : Prop :=
  a ∈ Set.Iic (-3) ∪ Set.Icc (-1) 0 ∪ Set.Ioc 0 1 ∪ Set.Ici 3

theorem f_geq_4_iff_valid_a_range (a : ℝ) (h : a ≠ 0) :
  (∀ x, f a x ≥ 4) ↔ valid_a_range a := by sorry

theorem f_3_geq_4 (a : ℝ) (h : a ≠ 0) : f a 3 ≥ 4 := by sorry

end NUMINAMATH_CALUDE_f_geq_4_iff_valid_a_range_f_3_geq_4_l221_22188


namespace NUMINAMATH_CALUDE_pizza_toppings_l221_22102

/-- Given a pizza with the following properties:
  * Total slices: 16
  * Slices with pepperoni: 8
  * Slices with mushrooms: 12
  * Plain slices: 2
  Prove that the number of slices with both pepperoni and mushrooms is 6. -/
theorem pizza_toppings (total : Nat) (pepperoni : Nat) (mushrooms : Nat) (plain : Nat)
    (h_total : total = 16)
    (h_pepperoni : pepperoni = 8)
    (h_mushrooms : mushrooms = 12)
    (h_plain : plain = 2) :
    ∃ (both : Nat), both = 6 ∧
      pepperoni + mushrooms - both = total - plain :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l221_22102


namespace NUMINAMATH_CALUDE_percentage_both_languages_l221_22119

/-- Represents the number of diplomats speaking both French and Russian -/
def both_languages (total french not_russian neither : ℕ) : ℕ :=
  french + (total - not_russian) - (total - neither)

/-- Theorem stating the percentage of diplomats speaking both French and Russian -/
theorem percentage_both_languages :
  let total := 100
  let french := 22
  let not_russian := 32
  let neither := 20
  (both_languages total french not_russian neither : ℚ) / total * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_both_languages_l221_22119


namespace NUMINAMATH_CALUDE_inequality_represents_lower_right_l221_22160

/-- Represents a point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The line defined by the equation x - 2y + 6 = 0 -/
def line (p : Point) : Prop :=
  p.x - 2 * p.y + 6 = 0

/-- The area defined by the inequality x - 2y + 6 > 0 -/
def inequality_area (p : Point) : Prop :=
  p.x - 2 * p.y + 6 > 0

/-- A point is on the lower right side of the line if it satisfies the inequality -/
def is_lower_right (p : Point) : Prop :=
  inequality_area p

theorem inequality_represents_lower_right :
  ∀ p : Point, is_lower_right p ↔ inequality_area p :=
sorry

end NUMINAMATH_CALUDE_inequality_represents_lower_right_l221_22160


namespace NUMINAMATH_CALUDE_statement_A_statement_C_statement_D_l221_22134

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 1)^3 - a*x - b + 1

-- Define the function g
def g (a b x : ℝ) : ℝ := f a b x - 3*x + a*x + b

-- Statement A
theorem statement_A (a b : ℝ) :
  a = 3 → (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b x = 0 ∧ f a b y = 0 ∧ f a b z = 0) →
  -4 < b ∧ b < 0 := by sorry

-- Statement C
theorem statement_C (a b m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ∃ k₁ k₂ k₃ : ℝ, 
      k₁ * (2 - x) + m = g a b x ∧
      k₂ * (2 - y) + m = g a b y ∧
      k₃ * (2 - z) + m = g a b z) →
  -5 < m ∧ m < -4 := by sorry

-- Statement D
theorem statement_D (a b : ℝ) :
  (∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧
    (∀ x : ℝ, f a b x₀ ≤ f a b x ∨ f a b x₀ ≥ f a b x) ∧
    f a b x₀ = f a b x₁) →
  ∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧ x₁ + 2*x₀ = 3 := by sorry

end NUMINAMATH_CALUDE_statement_A_statement_C_statement_D_l221_22134


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l221_22116

theorem polynomial_divisibility (m : ℚ) :
  (∀ x, (x^4 - 5*x^2 + 4*x - m) % (2*x + 1) = 0) → m = -51/16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l221_22116


namespace NUMINAMATH_CALUDE_inequality_proof_l221_22110

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  (|(1/3 : ℝ) * a + (1/6 : ℝ) * b| < (1/4 : ℝ)) ∧ 
  (|1 - 4 * a * b| > 2 * |a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l221_22110


namespace NUMINAMATH_CALUDE_min_sum_squares_on_parabola_l221_22100

/-- The parabola equation y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line passing through P(4, 0) and (x, y) -/
def line_through_P (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x - 4)

/-- Theorem: The minimum value of y₁² + y₂² is 32 for points on the parabola
    intersected by a line through P(4, 0) -/
theorem min_sum_squares_on_parabola (x₁ y₁ x₂ y₂ : ℝ) :
  parabola x₁ y₁ →
  parabola x₂ y₂ →
  line_through_P x₁ y₁ →
  line_through_P x₂ y₂ →
  y₁^2 + y₂^2 ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_on_parabola_l221_22100


namespace NUMINAMATH_CALUDE_complex_inequality_l221_22120

theorem complex_inequality (z : ℂ) (n : ℕ) (h1 : z.re ≥ 1) (h2 : n ≥ 4) :
  Complex.abs (z^(n+1) - 1) ≥ Complex.abs (z^n) * Complex.abs (z - 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l221_22120


namespace NUMINAMATH_CALUDE_power_problem_l221_22154

theorem power_problem (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3*a + b) = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_problem_l221_22154


namespace NUMINAMATH_CALUDE_min_ceiling_sum_squares_l221_22122

theorem min_ceiling_sum_squares (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z : ℝ) 
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0) (hE : E ≠ 0) (hF : F ≠ 0) 
  (hG : G ≠ 0) (hH : H ≠ 0) (hI : I ≠ 0) (hJ : J ≠ 0) (hK : K ≠ 0) (hL : L ≠ 0) 
  (hM : M ≠ 0) (hN : N ≠ 0) (hO : O ≠ 0) (hP : P ≠ 0) (hQ : Q ≠ 0) (hR : R ≠ 0) 
  (hS : S ≠ 0) (hT : T ≠ 0) (hU : U ≠ 0) (hV : V ≠ 0) (hW : W ≠ 0) (hX : X ≠ 0) 
  (hY : Y ≠ 0) (hZ : Z ≠ 0) : 
  26 = ⌈(A^2 + B^2 + C^2 + D^2 + E^2 + F^2 + G^2 + H^2 + I^2 + J^2 + K^2 + L^2 + 
         M^2 + N^2 + O^2 + P^2 + Q^2 + R^2 + S^2 + T^2 + U^2 + V^2 + W^2 + X^2 + Y^2 + Z^2)⌉ :=
by sorry

end NUMINAMATH_CALUDE_min_ceiling_sum_squares_l221_22122


namespace NUMINAMATH_CALUDE_no_meetings_before_return_l221_22186

/-- The number of times two boys meet on a circular track before returning to their starting point -/
def number_of_meetings (circumference : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℕ :=
  sorry

theorem no_meetings_before_return :
  let circumference : ℝ := 120
  let speed1 : ℝ := 6
  let speed2 : ℝ := 10
  number_of_meetings circumference speed1 speed2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_meetings_before_return_l221_22186


namespace NUMINAMATH_CALUDE_total_bikes_l221_22157

theorem total_bikes (jungkook_bikes : ℕ) (yoongi_bikes : ℕ) 
  (h1 : jungkook_bikes = 3) (h2 : yoongi_bikes = 4) : 
  jungkook_bikes + yoongi_bikes = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_bikes_l221_22157


namespace NUMINAMATH_CALUDE_largest_integer_less_than_80_remainder_3_mod_5_l221_22151

theorem largest_integer_less_than_80_remainder_3_mod_5 : ∃ n : ℕ, 
  (n < 80 ∧ n % 5 = 3 ∧ ∀ m : ℕ, m < 80 ∧ m % 5 = 3 → m ≤ n) ∧ n = 78 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_80_remainder_3_mod_5_l221_22151


namespace NUMINAMATH_CALUDE_fraction_simplification_l221_22176

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (18 - 13 * x) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l221_22176


namespace NUMINAMATH_CALUDE_bamboo_pole_is_ten_feet_l221_22195

/-- The length of a bamboo pole satisfying specific conditions relative to a door --/
def bamboo_pole_length : ℝ → Prop := fun x =>
  ∃ (door_width door_height : ℝ),
    door_width > 0 ∧ 
    door_height > 0 ∧ 
    x = door_width + 4 ∧ 
    x = door_height + 2 ∧ 
    x^2 = door_width^2 + door_height^2

/-- Theorem stating that the bamboo pole length is 10 feet --/
theorem bamboo_pole_is_ten_feet : 
  bamboo_pole_length 10 := by
  sorry

#check bamboo_pole_is_ten_feet

end NUMINAMATH_CALUDE_bamboo_pole_is_ten_feet_l221_22195


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l221_22180

theorem lcm_gcd_product (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h_lcm : Nat.lcm a b = 60) (h_gcd : Nat.gcd a b = 5) : 
  a * b = 300 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l221_22180


namespace NUMINAMATH_CALUDE_cubic_system_solution_l221_22185

theorem cubic_system_solution :
  ∃ (x y z : ℝ), 
    x + y + z = 1 ∧
    x^2 + y^2 + z^2 = 1 ∧
    x^3 + y^3 + z^3 = 89/125 ∧
    ((x = 2/5 ∧ y = (3 + Real.sqrt 33)/10 ∧ z = (3 - Real.sqrt 33)/10) ∨
     (x = 2/5 ∧ y = (3 - Real.sqrt 33)/10 ∧ z = (3 + Real.sqrt 33)/10) ∨
     (x = (3 + Real.sqrt 33)/10 ∧ y = 2/5 ∧ z = (3 - Real.sqrt 33)/10) ∨
     (x = (3 + Real.sqrt 33)/10 ∧ y = (3 - Real.sqrt 33)/10 ∧ z = 2/5) ∨
     (x = (3 - Real.sqrt 33)/10 ∧ y = 2/5 ∧ z = (3 + Real.sqrt 33)/10) ∨
     (x = (3 - Real.sqrt 33)/10 ∧ y = (3 + Real.sqrt 33)/10 ∧ z = 2/5)) :=
by
  sorry


end NUMINAMATH_CALUDE_cubic_system_solution_l221_22185


namespace NUMINAMATH_CALUDE_units_digit_of_factorial_sum_l221_22155

def factorial (n : ℕ) : ℕ := sorry

def sum_factorials (n : ℕ) : ℕ := sorry

theorem units_digit_of_factorial_sum : 
  ∃ k : ℕ, (sum_factorials 500 + factorial 2 * factorial 4 + factorial 3 * factorial 7) % 10 = 1 + 10 * k :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_factorial_sum_l221_22155


namespace NUMINAMATH_CALUDE_green_face_box_dimensions_l221_22168

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given dimensions satisfy the green face condition -/
def satisfiesGreenFaceCondition (dim : BoxDimensions) : Prop :=
  3 * ((dim.a - 2) * (dim.b - 2) * (dim.c - 2)) = dim.a * dim.b * dim.c

/-- List of valid box dimensions -/
def validDimensions : List BoxDimensions := [
  ⟨7, 30, 4⟩, ⟨8, 18, 4⟩, ⟨9, 14, 4⟩, ⟨10, 12, 4⟩,
  ⟨5, 27, 5⟩, ⟨6, 12, 5⟩, ⟨7, 9, 5⟩, ⟨6, 8, 6⟩
]

theorem green_face_box_dimensions :
  ∀ dim : BoxDimensions,
    satisfiesGreenFaceCondition dim ↔ dim ∈ validDimensions :=
by sorry

end NUMINAMATH_CALUDE_green_face_box_dimensions_l221_22168


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l221_22112

theorem parabola_line_intersection (a k b x₁ x₂ x₃ : ℝ) 
  (ha : a > 0)
  (h₁ : a * x₁^2 = k * x₁ + b)
  (h₂ : a * x₂^2 = k * x₂ + b)
  (h₃ : 0 = k * x₃ + b) :
  x₁ * x₂ = x₂ * x₃ + x₁ * x₃ := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l221_22112


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l221_22149

/-- The perpendicular bisector of a line segment connecting two points -/
theorem perpendicular_bisector_equation (A B : ℝ × ℝ) :
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let m_AB : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let m_perp : ℝ := -1 / m_AB
  A = (0, 1) →
  B = (4, 3) →
  (λ (x y : ℝ) => 2 * x + y - 6 = 0) =
    (λ (x y : ℝ) => y - M.2 = m_perp * (x - M.1)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l221_22149


namespace NUMINAMATH_CALUDE_other_color_counts_l221_22165

def total_students : ℕ := 800

def blue_shirt_percent : ℚ := 45/100
def red_shirt_percent : ℚ := 23/100
def green_shirt_percent : ℚ := 15/100

def black_pants_percent : ℚ := 30/100
def khaki_pants_percent : ℚ := 25/100
def jeans_percent : ℚ := 10/100

def white_shoes_percent : ℚ := 40/100
def black_shoes_percent : ℚ := 20/100
def brown_shoes_percent : ℚ := 15/100

theorem other_color_counts :
  let other_shirt_count := total_students - (blue_shirt_percent + red_shirt_percent + green_shirt_percent) * total_students
  let other_pants_count := total_students - (black_pants_percent + khaki_pants_percent + jeans_percent) * total_students
  let other_shoes_count := total_students - (white_shoes_percent + black_shoes_percent + brown_shoes_percent) * total_students
  (other_shirt_count : ℚ) = 136 ∧ (other_pants_count : ℚ) = 280 ∧ (other_shoes_count : ℚ) = 200 :=
by sorry

end NUMINAMATH_CALUDE_other_color_counts_l221_22165


namespace NUMINAMATH_CALUDE_unique_number_with_triple_property_l221_22128

/-- A six-digit number with 1 as its leftmost digit -/
def sixDigitNumberStartingWith1 (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 1

/-- Function to move the leftmost digit to the rightmost position -/
def moveFirstDigitToEnd (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

/-- The main theorem statement -/
theorem unique_number_with_triple_property :
  ∃! n : ℕ, sixDigitNumberStartingWith1 n ∧ moveFirstDigitToEnd n = 3 * n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_triple_property_l221_22128


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l221_22164

/-- Given a point P(-3, -5) in the Cartesian coordinate system,
    its coordinates with respect to the origin are (3, 5). -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-3, -5)
  (|P.1|, |P.2|) = (3, 5) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l221_22164


namespace NUMINAMATH_CALUDE_function_symmetry_and_translation_l221_22137

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define the translation operation
def translate (f : RealFunction) (h : ℝ) : RealFunction :=
  λ x => f (x + h)

-- Define symmetry with respect to y-axis
def symmetricToYAxis (f g : RealFunction) : Prop :=
  ∀ x, f x = g (-x)

-- State the theorem
theorem function_symmetry_and_translation (f : RealFunction) :
  (symmetricToYAxis (translate f 1) (λ x => 2^x)) →
  (f = λ x => (1/2)^(x-1)) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_and_translation_l221_22137


namespace NUMINAMATH_CALUDE_apple_tree_production_ratio_l221_22156

theorem apple_tree_production_ratio : 
  ∀ (first_season second_season third_season : ℕ),
  first_season = 200 →
  second_season = first_season - first_season / 5 →
  first_season + second_season + third_season = 680 →
  third_season / second_season = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_tree_production_ratio_l221_22156


namespace NUMINAMATH_CALUDE_sum_parity_of_nine_consecutive_naturals_l221_22127

theorem sum_parity_of_nine_consecutive_naturals (n : ℕ) :
  Even (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) ↔ Even n :=
by sorry

end NUMINAMATH_CALUDE_sum_parity_of_nine_consecutive_naturals_l221_22127


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l221_22166

theorem opposite_of_negative_three : 
  ∃ y : ℤ, ((-3 : ℤ) + y = 0) ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l221_22166


namespace NUMINAMATH_CALUDE_kyle_paper_delivery_l221_22197

/-- The number of papers Kyle delivers each week -/
def weekly_papers (
  regular_houses : ℕ
  ) (sunday_skip : ℕ) (sunday_extra : ℕ) : ℕ :=
  (6 * regular_houses) + (regular_houses - sunday_skip + sunday_extra)

theorem kyle_paper_delivery :
  weekly_papers 100 10 30 = 720 := by
  sorry

end NUMINAMATH_CALUDE_kyle_paper_delivery_l221_22197


namespace NUMINAMATH_CALUDE_largest_negative_integer_l221_22181

theorem largest_negative_integer :
  ∀ n : ℤ, n < 0 → n ≤ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_l221_22181


namespace NUMINAMATH_CALUDE_alternating_series_sum_equals_minus_30_l221_22124

def alternatingSeriesSum (a₁ : ℤ) (d : ℤ) (lastTerm : ℤ) : ℤ :=
  -- Definition of the sum of the alternating series
  sorry

theorem alternating_series_sum_equals_minus_30 :
  alternatingSeriesSum 2 6 59 = -30 := by
  sorry

end NUMINAMATH_CALUDE_alternating_series_sum_equals_minus_30_l221_22124


namespace NUMINAMATH_CALUDE_acute_angle_inequality_l221_22142

theorem acute_angle_inequality (α : Real) (h : 0 < α ∧ α < π / 2) :
  α < (Real.sin α + Real.tan α) / 2 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_inequality_l221_22142


namespace NUMINAMATH_CALUDE_fourth_power_sum_l221_22136

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a^2 + b^2 + c^2 = 5) 
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^4 + b^4 + c^4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l221_22136


namespace NUMINAMATH_CALUDE_yellow_parrot_count_l221_22121

theorem yellow_parrot_count (total : ℕ) (red_fraction : ℚ) : 
  total = 120 → red_fraction = 5/8 → (total : ℚ) * (1 - red_fraction) = 45 := by
  sorry

end NUMINAMATH_CALUDE_yellow_parrot_count_l221_22121


namespace NUMINAMATH_CALUDE_free_fall_time_l221_22131

-- Define the relationship between height and time
def height_time_relation (t : ℝ) : ℝ := 4.9 * t^2

-- Define the initial height
def initial_height : ℝ := 490

-- Theorem statement
theorem free_fall_time : 
  ∃ (t : ℝ), t > 0 ∧ height_time_relation t = initial_height ∧ t = 10 := by
  sorry

end NUMINAMATH_CALUDE_free_fall_time_l221_22131


namespace NUMINAMATH_CALUDE_rug_length_is_25_l221_22177

/-- Represents a rectangular rug with integer dimensions -/
structure Rug where
  length : ℕ
  width : ℕ

/-- Represents a rectangular room -/
structure Room where
  width : ℕ
  length : ℕ

/-- Checks if a rug fits perfectly in a room -/
def fitsInRoom (rug : Rug) (room : Room) : Prop :=
  rug.length ^ 2 + rug.width ^ 2 = room.width ^ 2 + room.length ^ 2

theorem rug_length_is_25 :
  ∃ (rug : Rug) (room1 room2 : Room),
    room1.width = 38 ∧
    room2.width = 50 ∧
    room1.length = room2.length ∧
    fitsInRoom rug room1 ∧
    fitsInRoom rug room2 →
    rug.length = 25 := by
  sorry

end NUMINAMATH_CALUDE_rug_length_is_25_l221_22177


namespace NUMINAMATH_CALUDE_penguin_colony_growth_l221_22150

/-- Represents the penguin colony growth over three years -/
structure PenguinColony where
  initial_size : ℕ
  first_year_growth : ℕ → ℕ
  second_year_growth : ℕ → ℕ
  third_year_gain : ℕ
  current_size : ℕ
  fish_per_penguin : ℚ
  initial_fish_caught : ℕ

/-- Theorem stating the number of penguins gained in the third year -/
theorem penguin_colony_growth (colony : PenguinColony) : colony.third_year_gain = 129 :=
  by
  have h1 : colony.initial_size = 158 := by sorry
  have h2 : colony.first_year_growth colony.initial_size = 2 * colony.initial_size := by sorry
  have h3 : colony.second_year_growth (colony.first_year_growth colony.initial_size) = 
            3 * (colony.first_year_growth colony.initial_size) := by sorry
  have h4 : colony.current_size = 1077 := by sorry
  have h5 : colony.fish_per_penguin = 3/2 := by sorry
  have h6 : colony.initial_fish_caught = 237 := by sorry
  have h7 : colony.initial_size * colony.fish_per_penguin = colony.initial_fish_caught := by sorry
  sorry

end NUMINAMATH_CALUDE_penguin_colony_growth_l221_22150


namespace NUMINAMATH_CALUDE_greek_cross_dissection_l221_22125

/-- Represents a Greek cross -/
structure GreekCross where
  area : ℝ
  squares : Fin 5 → Square

/-- Represents a square piece of a Greek cross -/
structure Square where
  side_length : ℝ

/-- Represents a piece obtained from cutting a Greek cross -/
inductive Piece
| Square : Square → Piece
| Composite : List Square → Piece

/-- Theorem stating that a Greek cross can be dissected into 12 pieces 
    to form three identical smaller Greek crosses -/
theorem greek_cross_dissection (original : GreekCross) :
  ∃ (pieces : List Piece) (small_crosses : Fin 3 → GreekCross),
    (pieces.length = 12) ∧
    (∀ i : Fin 3, (small_crosses i).area = original.area / 3) ∧
    (∀ i j : Fin 3, i ≠ j → small_crosses i = small_crosses j) ∧
    (∃ (reassembly : List Piece → Fin 3 → GreekCross), 
      reassembly pieces = small_crosses) :=
sorry

end NUMINAMATH_CALUDE_greek_cross_dissection_l221_22125


namespace NUMINAMATH_CALUDE_complex_real_part_l221_22161

theorem complex_real_part (z : ℂ) (h : (z^2 + z).im = 0) : z.re = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_part_l221_22161


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_10_l221_22173

theorem binomial_coefficient_19_10 : 
  (Nat.choose 17 7 = 19448) → (Nat.choose 17 9 = 24310) → (Nat.choose 19 10 = 87516) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_10_l221_22173


namespace NUMINAMATH_CALUDE_max_attendance_days_l221_22140

structure Day where
  name : String
  dan_available : Bool
  eve_available : Bool
  frank_available : Bool
  grace_available : Bool

def schedule : List Day := [
  { name := "Monday",    dan_available := false, eve_available := true,  frank_available := false, grace_available := true  },
  { name := "Tuesday",   dan_available := true,  eve_available := false, frank_available := false, grace_available := true  },
  { name := "Wednesday", dan_available := false, eve_available := true,  frank_available := true,  grace_available := false },
  { name := "Thursday",  dan_available := true,  eve_available := false, frank_available := true,  grace_available := false },
  { name := "Friday",    dan_available := false, eve_available := false, frank_available := false, grace_available := true  }
]

def count_available (day : Day) : Nat :=
  (if day.dan_available then 1 else 0) +
  (if day.eve_available then 1 else 0) +
  (if day.frank_available then 1 else 0) +
  (if day.grace_available then 1 else 0)

def max_available (schedule : List Day) : Nat :=
  schedule.map count_available |>.maximum?
    |>.getD 0

theorem max_attendance_days (schedule : List Day) :
  max_available schedule = 2 ∧
  schedule.filter (fun day => count_available day = 2) =
    schedule.filter (fun day => day.name ∈ ["Monday", "Tuesday", "Wednesday", "Thursday"]) :=
by sorry

end NUMINAMATH_CALUDE_max_attendance_days_l221_22140


namespace NUMINAMATH_CALUDE_trigonometric_identity_l221_22114

theorem trigonometric_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l221_22114


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l221_22126

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 11/15 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 11/15) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l221_22126


namespace NUMINAMATH_CALUDE_inequality_solution_set_inequality_positive_reals_l221_22199

-- Part 1: Inequality solution set
theorem inequality_solution_set (x : ℝ) :
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1/2 < x ∧ x ≤ 1 :=
sorry

-- Part 2: Inequality with positive real numbers
theorem inequality_positive_reals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/(b + c)) ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_inequality_positive_reals_l221_22199


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_line_equation_with_equal_intercepts_l221_22147

-- Define the line equation
def line_equation (m x y : ℝ) : Prop :=
  (m + 2) * x - (m + 1) * y - 3 * m - 7 = 0

-- Theorem 1: The line passes through the point (4, 1) for all real m
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m 4 1 :=
sorry

-- Theorem 2: When x and y intercepts are equal, the line equation becomes x+y-5=0
theorem line_equation_with_equal_intercepts :
  ∀ m : ℝ, 
    (∃ k : ℝ, k ≠ 0 ∧ line_equation m k 0 ∧ line_equation m 0 (-k)) →
    ∃ c : ℝ, ∀ x y : ℝ, line_equation m x y ↔ x + y - 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_line_equation_with_equal_intercepts_l221_22147


namespace NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l221_22159

-- Problem 1
theorem solution_set_inequality_1 (x : ℝ) :
  (x + 2) / (x - 3) ≤ 0 ↔ -2 ≤ x ∧ x < 3 :=
sorry

-- Problem 2
theorem solution_set_inequality_2 (x a : ℝ) :
  (x + a) * (x - 1) > 0 ↔
    (a > -1 ∧ (x < -a ∨ x > 1)) ∨
    (a = -1 ∧ x ≠ 1) ∨
    (a < -1 ∧ (x < 1 ∨ x > -a)) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l221_22159


namespace NUMINAMATH_CALUDE_basketball_not_tabletennis_l221_22178

theorem basketball_not_tabletennis (U A B : Finset ℕ) : 
  Finset.card U = 42 →
  Finset.card A = 20 →
  Finset.card B = 25 →
  Finset.card (U \ (A ∪ B)) = 12 →
  Finset.card (A \ B) = 5 := by
sorry

end NUMINAMATH_CALUDE_basketball_not_tabletennis_l221_22178


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l221_22109

def P : Set ℝ := {0, 1, 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = 3^x}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l221_22109


namespace NUMINAMATH_CALUDE_third_smallest_sum_is_four_l221_22135

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 10) % 10 = 1

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + (n % 10)

theorem third_smallest_sum_is_four :
  ∃ (n : ℕ), is_valid_number n ∧
  (∀ m, is_valid_number m → m < n) ∧
  (∃ k₁ k₂, is_valid_number k₁ ∧ is_valid_number k₂ ∧ k₁ < k₂ ∧ k₂ < n) ∧
  digit_sum n = 4 :=
sorry

end NUMINAMATH_CALUDE_third_smallest_sum_is_four_l221_22135


namespace NUMINAMATH_CALUDE_binary_1100111_to_decimal_l221_22133

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1100111₂ -/
def binary_1100111 : List Bool := [true, true, false, false, true, true, true]

/-- Theorem stating that the decimal equivalent of 1100111₂ is 103 -/
theorem binary_1100111_to_decimal :
  binary_to_decimal binary_1100111 = 103 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100111_to_decimal_l221_22133


namespace NUMINAMATH_CALUDE_exponent_multiplication_l221_22144

theorem exponent_multiplication (a b : ℝ) : -a^2 * 2*a^4*b = -2*a^6*b := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l221_22144


namespace NUMINAMATH_CALUDE_line_translation_l221_22190

/-- Represents a line in the 2D Cartesian plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The vertical translation distance between two lines -/
def vertical_translation (l1 l2 : Line) : ℝ :=
  l2.y_intercept - l1.y_intercept

theorem line_translation (l1 l2 : Line) :
  l1.slope = -2 ∧ l1.y_intercept = -2 ∧ 
  l2.slope = -2 ∧ l2.y_intercept = 4 →
  vertical_translation l1 l2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_l221_22190


namespace NUMINAMATH_CALUDE_tan_150_degrees_l221_22162

theorem tan_150_degrees : 
  Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l221_22162


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l221_22192

theorem fraction_equals_zero (x : ℝ) : 
  (x^2 - 1) / (1 - x) = 0 ∧ 1 - x ≠ 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l221_22192


namespace NUMINAMATH_CALUDE_long_jump_difference_l221_22129

/-- Represents the long jump event results for Ricciana and Margarita -/
structure LongJumpEvent where
  ricciana_total : ℕ
  ricciana_run : ℕ
  ricciana_jump : ℕ
  margarita_run : ℕ
  h_ricciana_total : ricciana_total = ricciana_run + ricciana_jump
  h_margarita_jump : ℕ

/-- The difference in total distance between Margarita and Ricciana is 1 foot -/
theorem long_jump_difference (event : LongJumpEvent)
  (h_ricciana_total : event.ricciana_total = 24)
  (h_ricciana_run : event.ricciana_run = 20)
  (h_ricciana_jump : event.ricciana_jump = 4)
  (h_margarita_run : event.margarita_run = 18)
  (h_margarita_jump : event.h_margarita_jump = 2 * event.ricciana_jump - 1) :
  event.margarita_run + event.h_margarita_jump - event.ricciana_total = 1 := by
  sorry

end NUMINAMATH_CALUDE_long_jump_difference_l221_22129


namespace NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l221_22183

theorem modulo_equivalence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 123456 [ZMOD 9] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l221_22183


namespace NUMINAMATH_CALUDE_green_minus_blue_equals_twenty_l221_22152

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green
  | Red

/-- Represents the distribution of disks in the bag -/
structure DiskDistribution where
  blue : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ

/-- The total number of disks in the bag -/
def totalDisks : ℕ := 108

/-- The ratio of blue:yellow:green:red disks -/
def colorRatio : DiskDistribution :=
  { blue := 3, yellow := 7, green := 8, red := 9 }

/-- The sum of all parts in the ratio -/
def totalRatioParts : ℕ :=
  colorRatio.blue + colorRatio.yellow + colorRatio.green + colorRatio.red

/-- Calculates the actual distribution of disks based on the ratio and total number of disks -/
def actualDistribution : DiskDistribution :=
  let disksPerPart := totalDisks / totalRatioParts
  { blue := colorRatio.blue * disksPerPart,
    yellow := colorRatio.yellow * disksPerPart,
    green := colorRatio.green * disksPerPart,
    red := colorRatio.red * disksPerPart }

/-- Theorem: There are 20 more green disks than blue disks in the bag -/
theorem green_minus_blue_equals_twenty :
  actualDistribution.green - actualDistribution.blue = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_minus_blue_equals_twenty_l221_22152


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l221_22106

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℤ := sorry

-- Define the expansion of (x-1)^6
def expansion_x_minus_1_power_6 (r : ℕ) : ℤ := binomial 6 r * (-1)^r

-- Theorem statement
theorem coefficient_x_squared_in_expansion :
  expansion_x_minus_1_power_6 3 = -20 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l221_22106


namespace NUMINAMATH_CALUDE_amithab_january_expenditure_l221_22184

/-- Amithab's monthly expenditure problem -/
theorem amithab_january_expenditure
  (avg_jan_to_jun : ℝ)
  (july_expenditure : ℝ)
  (avg_feb_to_jul : ℝ)
  (h1 : avg_jan_to_jun = 4200)
  (h2 : july_expenditure = 1500)
  (h3 : avg_feb_to_jul = 4250) :
  6 * avg_jan_to_jun + july_expenditure = 6 * avg_feb_to_jul + 1800 :=
by sorry

end NUMINAMATH_CALUDE_amithab_january_expenditure_l221_22184


namespace NUMINAMATH_CALUDE_least_common_denominator_l221_22153

theorem least_common_denominator : 
  let denominators := [5, 6, 8, 9, 10, 11]
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9) 10) 11 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l221_22153


namespace NUMINAMATH_CALUDE_smallest_four_digit_prime_divisible_proof_l221_22108

def smallest_four_digit_prime_divisible : ℕ := 2310

theorem smallest_four_digit_prime_divisible_proof :
  (smallest_four_digit_prime_divisible ≥ 1000) ∧
  (smallest_four_digit_prime_divisible < 10000) ∧
  (smallest_four_digit_prime_divisible % 2 = 0) ∧
  (smallest_four_digit_prime_divisible % 3 = 0) ∧
  (smallest_four_digit_prime_divisible % 5 = 0) ∧
  (smallest_four_digit_prime_divisible % 7 = 0) ∧
  (smallest_four_digit_prime_divisible % 11 = 0) ∧
  (∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 → n ≥ smallest_four_digit_prime_divisible) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_prime_divisible_proof_l221_22108


namespace NUMINAMATH_CALUDE_farmer_cows_problem_l221_22196

theorem farmer_cows_problem (initial_cows : ℕ) : 
  (3 / 4 : ℚ) * (initial_cows + 5 : ℚ) = 42 → initial_cows = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_cows_problem_l221_22196


namespace NUMINAMATH_CALUDE_probability_one_girl_two_boys_l221_22105

/-- The probability of having a boy or a girl for each child -/
def child_probability : ℝ := 0.5

/-- The number of children in the family -/
def num_children : ℕ := 3

/-- The number of ways to arrange 1 girl and 2 boys in 3 positions -/
def num_arrangements : ℕ := 3

/-- Theorem: The probability of having exactly 1 girl and 2 boys in a family with 3 children,
    where each child has an equal probability of being a boy or a girl, is 0.375 -/
theorem probability_one_girl_two_boys :
  (child_probability ^ num_children) * num_arrangements = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_girl_two_boys_l221_22105
