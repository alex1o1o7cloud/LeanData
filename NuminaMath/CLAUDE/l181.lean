import Mathlib

namespace NUMINAMATH_CALUDE_negative_a_fifth_times_a_l181_18116

theorem negative_a_fifth_times_a (a : ℝ) : (-a)^5 * a = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_fifth_times_a_l181_18116


namespace NUMINAMATH_CALUDE_parabola_point_distance_l181_18152

theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  y₀^2 = 8 * x₀ →  -- Point (x₀, y₀) is on the parabola y² = 8x
  (x₀ - 2)^2 + y₀^2 = 3^2 →  -- Distance from (x₀, y₀) to focus (2, 0) is 3
  |y₀| = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l181_18152


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l181_18118

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15) ∧ 
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 30 :=
by sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l181_18118


namespace NUMINAMATH_CALUDE_cone_height_l181_18124

/-- Given a cone with slant height 10 and base radius 5, its height is 5√3 -/
theorem cone_height (l r h : ℝ) (hl : l = 10) (hr : r = 5) 
  (h_def : h = Real.sqrt (l^2 - r^2)) : h = 5 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_cone_height_l181_18124


namespace NUMINAMATH_CALUDE_problem_solution_l181_18148

theorem problem_solution (x : ℝ) (h : (1 : ℝ) / 4 + 4 * ((1 : ℝ) / 2013 + 1 / x) = 7 / 4) :
  1872 + 48 * ((2013 : ℝ) * x / (x + 2013)) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l181_18148


namespace NUMINAMATH_CALUDE_inequality_system_solution_l181_18186

theorem inequality_system_solution :
  {x : ℝ | x + 3 ≥ 2 ∧ (3 * x - 1) / 2 < 4} = {x : ℝ | -1 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l181_18186


namespace NUMINAMATH_CALUDE_carton_height_theorem_l181_18103

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dim : BoxDimensions) : ℕ :=
  dim.length * dim.width * dim.height

/-- Calculates the area of a rectangle given its length and width -/
def rectangleArea (length width : ℕ) : ℕ :=
  length * width

/-- Calculates the number of smaller rectangles that can fit in a larger rectangle -/
def fitRectangles (largeLength largeWidth smallLength smallWidth : ℕ) : ℕ :=
  (largeLength / smallLength) * (largeWidth / smallWidth)

/-- The main theorem about the carton height -/
theorem carton_height_theorem 
  (cartonLength cartonWidth : ℕ)
  (soapBox : BoxDimensions)
  (maxSoapBoxes : ℕ) :
  cartonLength = 30 →
  cartonWidth = 42 →
  soapBox.length = 7 →
  soapBox.width = 6 →
  soapBox.height = 5 →
  maxSoapBoxes = 360 →
  ∃ (cartonHeight : ℕ), cartonHeight = 60 ∧
    cartonHeight * fitRectangles cartonLength cartonWidth soapBox.length soapBox.width = 
    maxSoapBoxes * soapBox.height :=
sorry

end NUMINAMATH_CALUDE_carton_height_theorem_l181_18103


namespace NUMINAMATH_CALUDE_monomial_2023_matches_pattern_l181_18134

/-- Represents a monomial in the sequence -/
def monomial (n : ℕ) : ℚ × ℕ := ((2 * n + 1) / n, n)

/-- The 2023rd monomial in the sequence -/
def monomial_2023 : ℚ × ℕ := (4047 / 2023, 2023)

/-- Theorem stating that the 2023rd monomial matches the pattern -/
theorem monomial_2023_matches_pattern : monomial 2023 = monomial_2023 := by
  sorry

end NUMINAMATH_CALUDE_monomial_2023_matches_pattern_l181_18134


namespace NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l181_18192

/-- Given a point P with coordinates (3,2), prove that its symmetrical point
    with respect to the x-axis has coordinates (3,-2) -/
theorem symmetry_wrt_x_axis :
  let P : ℝ × ℝ := (3, 2)
  let symmetry_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetry_x P = (3, -2) := by sorry

end NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l181_18192


namespace NUMINAMATH_CALUDE_tourist_group_size_proof_l181_18185

/-- Represents the number of people a large room can accommodate -/
def large_room_capacity : ℕ := 3

/-- Represents the number of large rooms rented -/
def large_rooms_rented : ℕ := 8

/-- Represents the total number of people in the tourist group -/
def tourist_group_size : ℕ := large_rooms_rented * large_room_capacity

theorem tourist_group_size_proof :
  (∀ n : ℕ, n ≠ tourist_group_size → 
    (∃ m k : ℕ, n = 3 * m + 2 * k ∧ m + k < large_rooms_rented) ∨
    (∃ m k : ℕ, n = 3 * m + 2 * k ∧ m > large_rooms_rented)) →
  tourist_group_size = 24 := by sorry

end NUMINAMATH_CALUDE_tourist_group_size_proof_l181_18185


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l181_18114

theorem area_between_concentric_circles (r : ℝ) (h1 : r = 2) (h2 : r > 0) : 
  π * (5 * r)^2 - π * r^2 = 96 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l181_18114


namespace NUMINAMATH_CALUDE_function_properties_l181_18122

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Define the theorem
theorem function_properties (a m n : ℝ) 
  (h_m_pos : m > 0) (h_n_pos : n > 0) (h_m_neq_n : m ≠ n)
  (h_fm : f a m = 3) (h_fn : f a n = 3) :
  0 < a ∧ a < Real.exp 2 ∧ a^2 < m * n ∧ m * n < a * Real.exp 2 := by
  sorry

end

end NUMINAMATH_CALUDE_function_properties_l181_18122


namespace NUMINAMATH_CALUDE_equation_solution_l181_18166

theorem equation_solution : 
  ∃! x : ℚ, (x + 1) / 3 - 1 = (5 * x - 1) / 6 :=
by
  use -1
  constructor
  · -- Prove that -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l181_18166


namespace NUMINAMATH_CALUDE_fruit_basket_combinations_l181_18193

/-- The number of possible fruit baskets given the constraints -/
def fruitBaskets (totalApples totalOranges : ℕ) (minApples minOranges : ℕ) : ℕ :=
  (totalApples - minApples + 1) * (totalOranges - minOranges + 1)

/-- Theorem stating the number of possible fruit baskets under given conditions -/
theorem fruit_basket_combinations :
  fruitBaskets 6 12 1 2 = 66 := by
  sorry

#eval fruitBaskets 6 12 1 2

end NUMINAMATH_CALUDE_fruit_basket_combinations_l181_18193


namespace NUMINAMATH_CALUDE_cone_apex_angle_l181_18156

theorem cone_apex_angle (r : ℝ) (h : ℝ) (l : ℝ) (θ : ℝ) : 
  r > 0 → h > 0 → l > 0 →
  l = 2 * r →  -- ratio of lateral area to base area is 2
  h = r * Real.sqrt 3 →  -- derived from Pythagorean theorem
  θ = 2 * Real.arctan (1 / Real.sqrt 3) →  -- definition of apex angle
  θ = π / 3  -- 60 degrees in radians
:= by sorry

end NUMINAMATH_CALUDE_cone_apex_angle_l181_18156


namespace NUMINAMATH_CALUDE_power_of_three_mod_nineteen_l181_18178

theorem power_of_three_mod_nineteen : 3^17 % 19 = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_nineteen_l181_18178


namespace NUMINAMATH_CALUDE_adam_final_spend_l181_18188

/-- Represents a purchased item with its weight and price per kilogram -/
structure Item where
  weight : Float
  price_per_kg : Float

/-- Calculates the total cost of purchases before discounts -/
def total_cost (items : List Item) : Float :=
  items.foldl (λ acc item => acc + item.weight * item.price_per_kg) 0

/-- Applies the almonds and walnuts discount if eligible -/
def apply_nuts_discount (almonds_cost cashews_cost total : Float) : Float :=
  if almonds_cost + cashews_cost ≥ 2.5 * 10 then
    total - 0.1 * (almonds_cost + cashews_cost)
  else
    total

/-- Applies the overall purchase discount if eligible -/
def apply_overall_discount (total : Float) : Float :=
  if total > 100 then total * 0.95 else total

/-- Theorem stating that Adam's final spend is $69.1 -/
theorem adam_final_spend :
  let items : List Item := [
    { weight := 1.5, price_per_kg := 12 },  -- almonds
    { weight := 1,   price_per_kg := 10 },  -- walnuts
    { weight := 0.5, price_per_kg := 20 },  -- cashews
    { weight := 1,   price_per_kg := 8 },   -- raisins
    { weight := 1.5, price_per_kg := 6 },   -- apricots
    { weight := 0.8, price_per_kg := 15 },  -- pecans
    { weight := 0.7, price_per_kg := 7 }    -- dates
  ]
  let initial_total := total_cost items
  let almonds_cost := 1.5 * 12
  let walnuts_cost := 1 * 10
  let after_nuts_discount := apply_nuts_discount almonds_cost walnuts_cost initial_total
  let final_total := apply_overall_discount after_nuts_discount
  final_total = 69.1 := by
  sorry

end NUMINAMATH_CALUDE_adam_final_spend_l181_18188


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_squares_with_diagonal_ratio_l181_18139

theorem perimeter_ratio_of_squares_with_diagonal_ratio (d : ℝ) :
  let d1 := d
  let d2 := 4 * d
  let s1 := d1 / Real.sqrt 2
  let s2 := d2 / Real.sqrt 2
  let p1 := 4 * s1
  let p2 := 4 * s2
  p2 / p1 = 8 := by sorry

end NUMINAMATH_CALUDE_perimeter_ratio_of_squares_with_diagonal_ratio_l181_18139


namespace NUMINAMATH_CALUDE_base4_division_theorem_l181_18137

/-- Convert a number from base 4 to base 10 -/
def base4To10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (4 ^ i)) 0

/-- Convert a number from base 10 to base 4 -/
def base10To4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Division in base 4 -/
def divBase4 (a b : List Nat) : List Nat :=
  base10To4 (base4To10 a / base4To10 b)

theorem base4_division_theorem :
  divBase4 [3, 1, 2, 2] [1, 2] = [2, 0, 1] := by sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l181_18137


namespace NUMINAMATH_CALUDE_ellipse_min_major_axis_l181_18154

/-- Given an ellipse where the maximum area of a triangle formed by a point
    on the ellipse and its two foci is 1, the minimum value of its major axis is 2√2. -/
theorem ellipse_min_major_axis (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_ellipse : a^2 = b^2 + c^2) (h_area : b * c = 1) :
  2 * a ≥ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_min_major_axis_l181_18154


namespace NUMINAMATH_CALUDE_sqrt_sum_floor_equality_l181_18194

theorem sqrt_sum_floor_equality (n : ℤ) : 
  ⌊Real.sqrt (n : ℝ) + Real.sqrt ((n + 1) : ℝ)⌋ = ⌊Real.sqrt ((4 * n + 2) : ℝ)⌋ :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_floor_equality_l181_18194


namespace NUMINAMATH_CALUDE_function_composition_ratio_l181_18129

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l181_18129


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l181_18106

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: For a geometric sequence with first term a₁ and common ratio q, 
    the general term a_n is equal to a₁qⁿ⁻¹. -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) (q : ℝ) (a₁ : ℝ) (h : GeometricSequence a q) (h₁ : a 1 = a₁) :
  ∀ n : ℕ, a n = a₁ * q ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l181_18106


namespace NUMINAMATH_CALUDE_smallest_positive_a_l181_18183

theorem smallest_positive_a (a : ℝ) : 
  a > 0 ∧ 
  (⌊2016 * a⌋ : ℤ) - (⌈a⌉ : ℤ) + 1 = 2016 ∧ 
  ∀ b : ℝ, b > 0 → (⌊2016 * b⌋ : ℤ) - (⌈b⌉ : ℤ) + 1 = 2016 → a ≤ b → 
  a = 2017 / 2016 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_a_l181_18183


namespace NUMINAMATH_CALUDE_expression_equality_l181_18142

theorem expression_equality (x : ℝ) (Q : ℝ) (h : 2 * (5 * x + 3 * Real.sqrt 2) = Q) :
  4 * (10 * x + 6 * Real.sqrt 2) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l181_18142


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l181_18167

theorem simplify_fraction_product : 16 * (-24 / 5) * (45 / 56) = -2160 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l181_18167


namespace NUMINAMATH_CALUDE_max_y_value_l181_18162

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : 
  ∃ (max_y : ℤ), (∀ (z : ℤ), ∃ (w : ℤ), w * z + 3 * w + 2 * z = -6 → z ≤ max_y) ∧ max_y = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l181_18162


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l181_18158

theorem sum_of_squares_problem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 50)
  (sum_of_products : x*y + y*z + z*x = 28) :
  x + y + z = Real.sqrt 106 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l181_18158


namespace NUMINAMATH_CALUDE_art_collection_unique_paintings_l181_18149

theorem art_collection_unique_paintings
  (shared : ℕ)
  (andrew_total : ℕ)
  (john_unique : ℕ)
  (h1 : shared = 15)
  (h2 : andrew_total = 25)
  (h3 : john_unique = 8) :
  andrew_total - shared + john_unique = 18 :=
by sorry

end NUMINAMATH_CALUDE_art_collection_unique_paintings_l181_18149


namespace NUMINAMATH_CALUDE_evaluate_expression_l181_18195

theorem evaluate_expression : (3^3)^2 + 1 = 730 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l181_18195


namespace NUMINAMATH_CALUDE_circle_center_distance_l181_18100

theorem circle_center_distance (x y : ℝ) :
  x^2 + y^2 = 8*x - 2*y + 23 →
  Real.sqrt ((4 - (-3))^2 + (-1 - 4)^2) = Real.sqrt 74 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_distance_l181_18100


namespace NUMINAMATH_CALUDE_box_width_l181_18160

/-- The width of a rectangular box given specific conditions -/
theorem box_width (num_cubes : ℕ) (cube_volume length height : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  length = 8 →
  height = 12 →
  (num_cubes : ℝ) * cube_volume / (length * height) = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_box_width_l181_18160


namespace NUMINAMATH_CALUDE_ellipse_properties_l181_18113

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the line l -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- Theorem stating the properties of the ellipse and its intersections -/
theorem ellipse_properties :
  ∀ (k m : ℝ),
  m > 0 →
  (∃ (A B : ℝ × ℝ),
    ellipse_C A.1 A.2 ∧
    ellipse_C B.1 B.2 ∧
    line_l k m A.1 A.2 ∧
    line_l k m B.1 B.2 ∧
    (k = 1/2 ∨ k = -1/2) →
    (∃ (c : ℝ), A.1^2 + A.2^2 + B.1^2 + B.2^2 = c) ∧
    (∃ (area : ℝ), area ≤ 1 ∧
      (k = 1/2 ∨ k = -1/2) →
      area = 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l181_18113


namespace NUMINAMATH_CALUDE_min_a_value_l181_18151

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a*x + 1 ≥ 0) →
  a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l181_18151


namespace NUMINAMATH_CALUDE_prime_power_cube_plus_one_l181_18177

def is_solution (x y z : ℕ+) : Prop :=
  z.val.Prime ∧ z^(x.val) = y^3 + 1

theorem prime_power_cube_plus_one :
  ∀ x y z : ℕ+, is_solution x y z ↔ (x, y, z) = (1, 1, 2) ∨ (x, y, z) = (2, 2, 3) :=
sorry

end NUMINAMATH_CALUDE_prime_power_cube_plus_one_l181_18177


namespace NUMINAMATH_CALUDE_conditional_probability_l181_18150

theorem conditional_probability (P_AB P_A : ℝ) (h1 : P_AB = 2/15) (h2 : P_A = 2/5) :
  P_AB / P_A = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_l181_18150


namespace NUMINAMATH_CALUDE_subtraction_of_negatives_l181_18131

theorem subtraction_of_negatives : -1 - 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negatives_l181_18131


namespace NUMINAMATH_CALUDE_count_valid_pairs_l181_18119

def is_valid_pair (a b : ℂ) : Prop :=
  a^4 * b^7 = 1 ∧ a^8 * b^3 = 1

theorem count_valid_pairs :
  ∃! (n : ℕ), ∃ (S : Finset (ℂ × ℂ)),
    Finset.card S = n ∧
    (∀ (p : ℂ × ℂ), p ∈ S ↔ is_valid_pair p.1 p.2) ∧
    n = 16 :=
by sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l181_18119


namespace NUMINAMATH_CALUDE_max_value_of_S_l181_18163

theorem max_value_of_S (a b : ℝ) :
  3 * a^2 + 5 * abs b = 7 →
  let S := 2 * a^2 - 3 * abs b
  ∀ x y : ℝ, 3 * x^2 + 5 * abs y = 7 → 2 * x^2 - 3 * abs y ≤ |14| / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_max_value_of_S_l181_18163


namespace NUMINAMATH_CALUDE_cabin_price_calculation_l181_18198

/-- The price of Alfonso's cabin that Gloria wants to buy -/
def cabin_price : ℕ := sorry

/-- Gloria's initial cash -/
def initial_cash : ℕ := 150

/-- Number of cypress trees Gloria has -/
def cypress_trees : ℕ := 20

/-- Number of pine trees Gloria has -/
def pine_trees : ℕ := 600

/-- Number of maple trees Gloria has -/
def maple_trees : ℕ := 24

/-- Price per cypress tree -/
def cypress_price : ℕ := 100

/-- Price per pine tree -/
def pine_price : ℕ := 200

/-- Price per maple tree -/
def maple_price : ℕ := 300

/-- Amount Gloria wants to have left after buying the cabin -/
def leftover_amount : ℕ := 350

/-- Total amount Gloria can get from selling her trees and her initial cash -/
def total_amount : ℕ :=
  initial_cash +
  cypress_trees * cypress_price +
  pine_trees * pine_price +
  maple_trees * maple_price

theorem cabin_price_calculation :
  cabin_price = total_amount - leftover_amount :=
by sorry

end NUMINAMATH_CALUDE_cabin_price_calculation_l181_18198


namespace NUMINAMATH_CALUDE_walter_chores_l181_18182

/-- The number of days Walter worked -/
def total_days : ℕ := 10

/-- Walter's earnings for a regular day -/
def regular_pay : ℕ := 3

/-- Walter's earnings for an exceptional day -/
def exceptional_pay : ℕ := 5

/-- Walter's total earnings -/
def total_earnings : ℕ := 36

/-- The number of days Walter did chores exceptionally well -/
def exceptional_days : ℕ := 3

/-- The number of days Walter did regular chores -/
def regular_days : ℕ := total_days - exceptional_days

theorem walter_chores :
  regular_days * regular_pay + exceptional_days * exceptional_pay = total_earnings ∧
  regular_days + exceptional_days = total_days :=
by sorry

end NUMINAMATH_CALUDE_walter_chores_l181_18182


namespace NUMINAMATH_CALUDE_total_cleaner_needed_l181_18107

/-- Amount of cleaner needed for a dog stain in ounces -/
def dog_cleaner : ℕ := 6

/-- Amount of cleaner needed for a cat stain in ounces -/
def cat_cleaner : ℕ := 4

/-- Amount of cleaner needed for a rabbit stain in ounces -/
def rabbit_cleaner : ℕ := 1

/-- Number of dogs -/
def num_dogs : ℕ := 6

/-- Number of cats -/
def num_cats : ℕ := 3

/-- Number of rabbits -/
def num_rabbits : ℕ := 1

/-- Theorem stating the total amount of cleaner needed -/
theorem total_cleaner_needed : 
  dog_cleaner * num_dogs + cat_cleaner * num_cats + rabbit_cleaner * num_rabbits = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_cleaner_needed_l181_18107


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_m_range_l181_18135

/-- An ellipse with equation x²/5 + y²/m = 1 -/
def isEllipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2/5 + y^2/m = 1 ∧ m ≠ 0 ∧ m ≠ 5

/-- A hyperbola with equation x²/5 + y²/(m-6) = 1 -/
def isHyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2/5 + y^2/(m-6) = 1 ∧ m ≠ 6

/-- The range of valid m values -/
def validRange (m : ℝ) : Prop :=
  (0 < m ∧ m < 5) ∨ (5 < m ∧ m < 6)

theorem ellipse_hyperbola_m_range :
  ∀ m : ℝ, (isEllipse m ∧ isHyperbola m) ↔ validRange m :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_m_range_l181_18135


namespace NUMINAMATH_CALUDE_solve_system_for_p_l181_18174

theorem solve_system_for_p (p q : ℚ) 
  (eq1 : 2 * p + 5 * q = 10)
  (eq2 : 5 * p + 2 * q = 20) : 
  p = 80 / 21 := by sorry

end NUMINAMATH_CALUDE_solve_system_for_p_l181_18174


namespace NUMINAMATH_CALUDE_square_sum_proof_l181_18146

theorem square_sum_proof (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 32) : a^2 + b^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_proof_l181_18146


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l181_18190

theorem base_10_to_base_7 : ∃ (a b c d : ℕ), 
  803 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
  a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
  a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 5 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l181_18190


namespace NUMINAMATH_CALUDE_bells_lcm_l181_18189

def church_interval : ℕ := 18
def school_interval : ℕ := 24
def city_hall_interval : ℕ := 30

theorem bells_lcm :
  Nat.lcm (Nat.lcm church_interval school_interval) city_hall_interval = 360 := by
  sorry

end NUMINAMATH_CALUDE_bells_lcm_l181_18189


namespace NUMINAMATH_CALUDE_john_repair_results_l181_18181

/-- Represents the repair job details for John --/
structure RepairJob where
  totalCars : ℕ
  standardRepairCars : ℕ
  standardRepairTime : ℕ
  longerRepairPercent : ℚ
  hourlyRate : ℚ

/-- Calculates the total repair time and money earned for a given repair job --/
def calculateRepairResults (job : RepairJob) : ℚ × ℚ :=
  let standardTime := job.standardRepairCars * job.standardRepairTime
  let longerRepairTime := job.standardRepairTime * (1 + job.longerRepairPercent)
  let longerRepairCars := job.totalCars - job.standardRepairCars
  let longerTime := longerRepairCars * longerRepairTime
  let totalMinutes := standardTime + longerTime
  let totalHours := totalMinutes / 60
  let moneyEarned := totalHours * job.hourlyRate
  (totalHours, moneyEarned)

/-- Theorem stating that for John's specific repair job, the total repair time is 11 hours and he earns $330 --/
theorem john_repair_results :
  let job : RepairJob := {
    totalCars := 10,
    standardRepairCars := 6,
    standardRepairTime := 50,
    longerRepairPercent := 4/5,
    hourlyRate := 30
  }
  calculateRepairResults job = (11, 330) := by sorry

end NUMINAMATH_CALUDE_john_repair_results_l181_18181


namespace NUMINAMATH_CALUDE_library_purchase_theorem_l181_18165

-- Define the types of books
inductive BookType
| SocialScience
| Children

-- Define the price function
def price : BookType → ℕ
| BookType.SocialScience => 40
| BookType.Children => 20

-- Define the total cost function
def totalCost (ss_count : ℕ) (c_count : ℕ) : ℕ :=
  ss_count * price BookType.SocialScience + c_count * price BookType.Children

-- Define the valid purchase plan predicate
def isValidPurchasePlan (ss_count : ℕ) (c_count : ℕ) : Prop :=
  ss_count + c_count ≥ 70 ∧
  c_count = ss_count + 20 ∧
  totalCost ss_count c_count ≤ 2000

-- State the theorem
theorem library_purchase_theorem :
  (totalCost 20 40 = 1600) ∧
  (20 * price BookType.SocialScience = 30 * price BookType.Children + 200) ∧
  (∀ ss_count c_count : ℕ, isValidPurchasePlan ss_count c_count ↔ 
    (ss_count = 25 ∧ c_count = 45) ∨ (ss_count = 26 ∧ c_count = 46)) :=
sorry

end NUMINAMATH_CALUDE_library_purchase_theorem_l181_18165


namespace NUMINAMATH_CALUDE_victor_stickers_l181_18101

theorem victor_stickers (flower_stickers : ℕ) (total_stickers : ℕ) (animal_stickers : ℕ) : 
  flower_stickers = 8 → 
  total_stickers = 14 → 
  animal_stickers < flower_stickers → 
  flower_stickers + animal_stickers = total_stickers →
  animal_stickers = 6 := by
sorry

end NUMINAMATH_CALUDE_victor_stickers_l181_18101


namespace NUMINAMATH_CALUDE_exists_compound_interest_l181_18145

/-- Represents the compound interest scenario -/
def compound_interest (P : ℝ) : Prop :=
  let r : ℝ := 0.06  -- annual interest rate
  let n : ℝ := 12    -- number of compounding periods per year
  let t : ℝ := 0.25  -- time in years (3 months)
  let A : ℝ := 1014.08  -- final amount after 3 months
  let two_month_amount : ℝ := P * (1 + r / n) ^ (2 * n * (t / 3))
  A = P * (1 + r / n) ^ (n * t) ∧ 
  (A - two_month_amount) * 100 = 13

/-- Theorem stating the existence of an initial investment satisfying the compound interest scenario -/
theorem exists_compound_interest : ∃ P : ℝ, compound_interest P :=
  sorry

end NUMINAMATH_CALUDE_exists_compound_interest_l181_18145


namespace NUMINAMATH_CALUDE_abs_neg_2023_l181_18143

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l181_18143


namespace NUMINAMATH_CALUDE_seven_at_eight_equals_28_div_9_l181_18132

/-- The '@' operation for positive integers -/
def at_op (a b : ℕ+) : ℚ :=
  (a.val * b.val : ℚ) / (a.val + b.val + 3 : ℚ)

/-- Theorem: 7 @ 8 = 28/9 -/
theorem seven_at_eight_equals_28_div_9 : 
  at_op ⟨7, by norm_num⟩ ⟨8, by norm_num⟩ = 28 / 9 := by
  sorry

end NUMINAMATH_CALUDE_seven_at_eight_equals_28_div_9_l181_18132


namespace NUMINAMATH_CALUDE_f_properties_l181_18170

/-- The function f(m, n) represents the absolute difference between 
    the areas of black and white parts in a right triangle with legs m and n. -/
def f (m n : ℕ+) : ℝ :=
  sorry

theorem f_properties :
  (∀ m n : ℕ+, Even m.val → Even n.val → f m n = 0) ∧
  (∀ m n : ℕ+, Odd m.val → Odd n.val → f m n = 1/2) ∧
  (∀ m n : ℕ+, f m n ≤ (1/2 : ℝ) * max m.val n.val) ∧
  (∀ c : ℝ, ∃ m n : ℕ+, f m n ≥ c) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l181_18170


namespace NUMINAMATH_CALUDE_arccos_one_half_l181_18112

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l181_18112


namespace NUMINAMATH_CALUDE_remainder_of_binary_division_l181_18115

def binary_number : ℕ := 101110100101

theorem remainder_of_binary_division (n : ℕ) (h : n = binary_number) :
  n % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_division_l181_18115


namespace NUMINAMATH_CALUDE_system_solution_inequality_solution_l181_18108

theorem system_solution :
  ∃! (x y : ℝ), (6 * x - 2 * y = 1 ∧ 2 * x + y = 2) ∧
  x = (1/2 : ℝ) ∧ y = 1 := by sorry

theorem inequality_solution :
  ∀ x : ℝ, (2 * x - 10 < 0 ∧ (x + 1) / 3 < x - 1) ↔ (2 < x ∧ x < 5) := by sorry

end NUMINAMATH_CALUDE_system_solution_inequality_solution_l181_18108


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l181_18184

theorem product_of_four_consecutive_integers (X : ℤ) :
  X * (X + 1) * (X + 2) * (X + 3) = (X^2 + 3*X + 1)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l181_18184


namespace NUMINAMATH_CALUDE_game_a_higher_prob_l181_18128

def prob_heads : ℚ := 3/4
def prob_tails : ℚ := 1/4

def game_a_win_prob : ℚ := prob_heads^4 + prob_tails^4

def game_b_win_prob : ℚ := prob_heads^3 * prob_tails^2 + prob_tails^3 * prob_heads^2

theorem game_a_higher_prob : game_a_win_prob = game_b_win_prob + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_game_a_higher_prob_l181_18128


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l181_18144

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- If the points (2, 3), (7, k), and (15, 4) are collinear, then k = 44/13. -/
theorem collinear_points_k_value :
  collinear 2 3 7 k 15 4 → k = 44 / 13 :=
by sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l181_18144


namespace NUMINAMATH_CALUDE_complex_determinant_solution_l181_18147

/-- Definition of the determinant operation -/
def det (a b c d : ℂ) : ℂ := a * d - b * c

/-- Theorem stating that z = 2 - i satisfies the given condition -/
theorem complex_determinant_solution :
  ∃ z : ℂ, det z (1 + 2*I) (1 - I) (1 + I) = 0 ∧ z = 2 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_determinant_solution_l181_18147


namespace NUMINAMATH_CALUDE_fraction_cubed_l181_18197

theorem fraction_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cubed_l181_18197


namespace NUMINAMATH_CALUDE_partition_large_rectangle_l181_18138

/-- Definition of a "good" rectangle -/
inductive GoodRectangle
  | square : GoodRectangle
  | rectangle : GoodRectangle

/-- Predicate to check if a rectangle can be partitioned into good rectangles -/
def can_partition (a b : ℕ) : Prop :=
  ∃ (num_squares num_rectangles : ℕ),
    2 * 2 * num_squares + 1 * 11 * num_rectangles = a * b

/-- Theorem: Any rectangle with integer sides greater than 100 can be partitioned into good rectangles -/
theorem partition_large_rectangle (a b : ℕ) (ha : a > 100) (hb : b > 100) :
  can_partition a b := by
  sorry


end NUMINAMATH_CALUDE_partition_large_rectangle_l181_18138


namespace NUMINAMATH_CALUDE_no_upper_bound_for_positive_second_order_ratio_increasing_l181_18199

open Set Real

-- Define the type for functions from (0, +∞) to ℝ
def PosRealFunc := { f : ℝ → ℝ // ∀ x, x > 0 → f x ≠ 0 }

-- Define second-order ratio increasing function
def SecondOrderRatioIncreasing (f : PosRealFunc) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f.val x / x^2 < f.val y / y^2

-- Define the theorem
theorem no_upper_bound_for_positive_second_order_ratio_increasing
  (f : PosRealFunc)
  (h1 : SecondOrderRatioIncreasing f)
  (h2 : ∀ x, x > 0 → f.val x > 0) :
  ¬∃ k, ∀ x, x > 0 → f.val x < k :=
sorry

end NUMINAMATH_CALUDE_no_upper_bound_for_positive_second_order_ratio_increasing_l181_18199


namespace NUMINAMATH_CALUDE_total_silver_dollars_l181_18127

/-- The number of silver dollars owned by Mr. Chiu -/
def chiu_dollars : ℕ := 56

/-- The number of silver dollars owned by Mr. Phung -/
def phung_dollars : ℕ := chiu_dollars + 16

/-- The number of silver dollars owned by Mr. Ha -/
def ha_dollars : ℕ := phung_dollars + 5

/-- The total number of silver dollars owned by all three -/
def total_dollars : ℕ := chiu_dollars + phung_dollars + ha_dollars

theorem total_silver_dollars :
  total_dollars = 205 := by sorry

end NUMINAMATH_CALUDE_total_silver_dollars_l181_18127


namespace NUMINAMATH_CALUDE_common_area_rectangle_circle_l181_18173

/-- The area of the region common to a rectangle and a circle with the same center -/
theorem common_area_rectangle_circle (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_radius : ℝ) : 
  rectangle_width = 10 →
  rectangle_height = 4 →
  circle_radius = 5 →
  (rectangle_width / 2 = circle_radius) →
  (rectangle_height / 2 < circle_radius) →
  let common_area := rectangle_width * rectangle_height + 2 * π * (rectangle_height / 2)^2
  common_area = 40 + 4 * π := by
  sorry


end NUMINAMATH_CALUDE_common_area_rectangle_circle_l181_18173


namespace NUMINAMATH_CALUDE_fraction_powers_equality_l181_18105

theorem fraction_powers_equality : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_fraction_powers_equality_l181_18105


namespace NUMINAMATH_CALUDE_not_necessarily_divisible_by_twenty_l181_18164

theorem not_necessarily_divisible_by_twenty (k : ℤ) (n : ℤ) : 
  n = k * (k + 1) * (k + 2) → (∃ m : ℤ, n = 5 * m) → 
  ¬(∀ (k : ℤ), ∃ (m : ℤ), n = 20 * m) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_divisible_by_twenty_l181_18164


namespace NUMINAMATH_CALUDE_function_periodicity_l181_18155

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_periodicity (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f 1 := by
sorry

end NUMINAMATH_CALUDE_function_periodicity_l181_18155


namespace NUMINAMATH_CALUDE_water_tank_capacity_l181_18123

theorem water_tank_capacity (x : ℚ) : 
  (1 / 3 : ℚ) * x + 16 = x → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l181_18123


namespace NUMINAMATH_CALUDE_odd_function_property_l181_18126

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f 3 - f 2 = 1) :
  f (-2) - f (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l181_18126


namespace NUMINAMATH_CALUDE_quadratic_inequality_l181_18157

/-- Given non-zero numbers a, b, c such that ax^2 + bx + c > cx for all real x,
    prove that cx^2 - bx + a > cx - b for all real x. -/
theorem quadratic_inequality (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_given : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l181_18157


namespace NUMINAMATH_CALUDE_first_1000_decimals_are_zero_l181_18125

theorem first_1000_decimals_are_zero (a : ℕ) (n : ℕ) 
    (ha : a = 35 ∨ a = 37) (hn : n = 1999 ∨ n = 2000) :
  ∃ (k : ℕ), (6 + Real.sqrt a)^n = k + (1 / 10^1000) * (Real.sqrt a) := by
  sorry

end NUMINAMATH_CALUDE_first_1000_decimals_are_zero_l181_18125


namespace NUMINAMATH_CALUDE_prob_not_blue_marble_l181_18161

/-- Given odds ratio for an event --/
structure OddsRatio :=
  (for_event : ℕ)
  (against_event : ℕ)

/-- Calculates the probability of an event not occurring given its odds ratio --/
def probability_of_not_occurring (odds : OddsRatio) : ℚ :=
  odds.against_event / (odds.for_event + odds.against_event)

/-- Theorem: The probability of not pulling a blue marble is 6/11 given odds of 5:6 --/
theorem prob_not_blue_marble (odds : OddsRatio) 
  (h : odds = OddsRatio.mk 5 6) : 
  probability_of_not_occurring odds = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_blue_marble_l181_18161


namespace NUMINAMATH_CALUDE_gcd_n4_plus_16_n_plus_3_l181_18133

theorem gcd_n4_plus_16_n_plus_3 (n : ℕ) (h : n > 16) :
  Nat.gcd (n^4 + 16) (n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n4_plus_16_n_plus_3_l181_18133


namespace NUMINAMATH_CALUDE_any_proof_to_contradiction_l181_18136

theorem any_proof_to_contradiction (P : Prop) : P → ∃ (proof : ¬P → False), P :=
  sorry

end NUMINAMATH_CALUDE_any_proof_to_contradiction_l181_18136


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l181_18187

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

/-- Theorem: The 12th term of the arithmetic sequence with a₁ = 1/4 and d = 1/2 is 23/4 -/
theorem twelfth_term_of_sequence :
  arithmetic_sequence (1/4) (1/2) 12 = 23/4 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l181_18187


namespace NUMINAMATH_CALUDE_exponential_inequality_l181_18120

theorem exponential_inequality (x a b : ℝ) 
  (h_x_pos : x > 0) 
  (h_ineq : 0 < b^x ∧ b^x < a^x ∧ a^x < 1) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) : 
  1 > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l181_18120


namespace NUMINAMATH_CALUDE_dividing_line_equation_l181_18111

/-- Represents a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the region S formed by the union of nine unit circles -/
def region_S : Set (ℝ × ℝ) :=
  sorry

/-- The line with slope 4 that divides region S into two equal areas -/
def dividing_line : ℝ → ℝ :=
  sorry

/-- Theorem stating that the dividing line has the equation 4x - y = 3 -/
theorem dividing_line_equation :
  ∀ x y, dividing_line y = x ↔ 4 * x - y = 3 :=
sorry

end NUMINAMATH_CALUDE_dividing_line_equation_l181_18111


namespace NUMINAMATH_CALUDE_size_relationship_l181_18172

theorem size_relationship (a₁ a₂ b₁ b₂ : ℝ) (h1 : a₁ < a₂) (h2 : b₁ < b₂) :
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l181_18172


namespace NUMINAMATH_CALUDE_prob_sum_three_two_dice_l181_18168

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of ways to roll a sum of 3 with two dice -/
def favorable_outcomes : ℕ := 2

/-- The probability of an event occurring -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Theorem: The probability of rolling a sum of 3 with two fair dice is 1/18 -/
theorem prob_sum_three_two_dice : 
  probability favorable_outcomes total_outcomes = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_prob_sum_three_two_dice_l181_18168


namespace NUMINAMATH_CALUDE_square_difference_65_35_l181_18130

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_65_35_l181_18130


namespace NUMINAMATH_CALUDE_tailwind_speed_l181_18102

/-- Given a plane's ground speeds with and against a tailwind, calculate the speed of the tailwind. -/
theorem tailwind_speed (speed_with_wind speed_against_wind : ℝ) 
  (h1 : speed_with_wind = 460)
  (h2 : speed_against_wind = 310) :
  ∃ (plane_speed wind_speed : ℝ),
    plane_speed + wind_speed = speed_with_wind ∧
    plane_speed - wind_speed = speed_against_wind ∧
    wind_speed = 75 := by
  sorry

end NUMINAMATH_CALUDE_tailwind_speed_l181_18102


namespace NUMINAMATH_CALUDE_intersection_implies_z_equals_i_l181_18159

theorem intersection_implies_z_equals_i : 
  let i : ℂ := Complex.I
  let P : Set ℂ := {1, -1}
  let Q : Set ℂ := {i, i^2}
  ∀ z : ℂ, (P ∩ Q = {z * i}) → z = i := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_z_equals_i_l181_18159


namespace NUMINAMATH_CALUDE_classroom_gpa_l181_18175

/-- Given a classroom where one-third of the students have a GPA of 54 and the remaining two-thirds have a GPA of 45, the GPA of the whole class is 48. -/
theorem classroom_gpa : 
  ∀ (n : ℕ) (total_gpa : ℝ),
  n > 0 →
  total_gpa = (n / 3 : ℝ) * 54 + (2 * n / 3 : ℝ) * 45 →
  total_gpa / n = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_gpa_l181_18175


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l181_18153

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (2 - a * I) / (1 + I) = b * I) ↔ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l181_18153


namespace NUMINAMATH_CALUDE_infinitely_many_polynomials_l181_18141

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that x, y, and z must satisfy -/
def SphereCondition (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 + 2*x*y*z = 1

/-- The condition that the polynomial P must satisfy -/
def PolynomialCondition (P : RealPolynomial) : Prop :=
  ∀ x y z : ℝ, SphereCondition x y z →
    P x^2 + P y^2 + P z^2 + 2*(P x)*(P y)*(P z) = 1

/-- The main theorem stating that there are infinitely many polynomials satisfying the condition -/
theorem infinitely_many_polynomials :
  ∃ (S : Set RealPolynomial), (Set.Infinite S) ∧ (∀ P ∈ S, PolynomialCondition P) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_polynomials_l181_18141


namespace NUMINAMATH_CALUDE_initial_children_on_bus_l181_18117

/-- Given that 14 more children got on a bus at a bus stop, 
    resulting in a total of 78 children, prove that there were 
    initially 64 children on the bus. -/
theorem initial_children_on_bus : 
  ∀ (initial : ℕ), initial + 14 = 78 → initial = 64 := by
  sorry

end NUMINAMATH_CALUDE_initial_children_on_bus_l181_18117


namespace NUMINAMATH_CALUDE_ali_baba_strategy_l181_18121

/-- A game with diamonds where players split piles. -/
structure DiamondGame where
  total_diamonds : ℕ
  
/-- The number of moves required to end the game. -/
def moves_to_end (game : DiamondGame) : ℕ :=
  game.total_diamonds - 1

/-- Determines if the second player wins the game. -/
def second_player_wins (game : DiamondGame) : Prop :=
  Even (moves_to_end game)

/-- Theorem: In a game with 2017 diamonds, the second player wins. -/
theorem ali_baba_strategy (game : DiamondGame) (h : game.total_diamonds = 2017) :
  second_player_wins game := by
  sorry

#eval moves_to_end { total_diamonds := 2017 }

end NUMINAMATH_CALUDE_ali_baba_strategy_l181_18121


namespace NUMINAMATH_CALUDE_second_polygon_sides_l181_18110

/-- 
Given two regular polygons with the same perimeter, where the first polygon has 24 sides
and a side length that is three times as long as the second polygon,
prove that the second polygon has 72 sides.
-/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 → 
  24 * (3 * s) = n * s → 
  n = 72 :=
by sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l181_18110


namespace NUMINAMATH_CALUDE_chef_potato_problem_l181_18104

theorem chef_potato_problem (cooked : ℕ) (cook_time : ℕ) (remaining_time : ℕ) : 
  cooked = 7 → 
  cook_time = 5 → 
  remaining_time = 45 → 
  cooked + remaining_time / cook_time = 16 := by
sorry

end NUMINAMATH_CALUDE_chef_potato_problem_l181_18104


namespace NUMINAMATH_CALUDE_vacation_savings_l181_18196

def total_income : ℝ := 72800
def total_expenses : ℝ := 54200
def deposit_rate : ℝ := 0.1

theorem vacation_savings : 
  let remaining := total_income - total_expenses
  let deposit := deposit_rate * remaining
  total_income - total_expenses - deposit = 16740 := by
  sorry

end NUMINAMATH_CALUDE_vacation_savings_l181_18196


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l181_18180

theorem closest_integer_to_cube_root_250 :
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (250 : ℝ)^(1/3)| ≤ |m - (250 : ℝ)^(1/3)| ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l181_18180


namespace NUMINAMATH_CALUDE_simplify_expression_solve_fractional_equation_l181_18169

-- Problem 1
theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 := by
  sorry

-- Problem 2
theorem solve_fractional_equation :
  ∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 ∧ 5 / (x^2 + x) - 1 / (x^2 - x) = 0 ∧ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_fractional_equation_l181_18169


namespace NUMINAMATH_CALUDE_complex_equation_solution_l181_18109

theorem complex_equation_solution (a b : ℝ) :
  (1 + 2 * Complex.I) * a + b = 2 * Complex.I → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l181_18109


namespace NUMINAMATH_CALUDE_lottery_ambo_probability_l181_18191

theorem lottery_ambo_probability (n : ℕ) : 
  (n ≥ 5) →
  (Nat.choose 5 2 : ℚ) / (Nat.choose n 2 : ℚ) = 5 / 473 →
  n = 44 :=
by sorry

end NUMINAMATH_CALUDE_lottery_ambo_probability_l181_18191


namespace NUMINAMATH_CALUDE_optimal_timing_problem_l181_18171

/-- Represents the optimal timing problem for three people traveling between two points. -/
theorem optimal_timing_problem (distance : ℝ) (walking_speed : ℝ) (bicycle_speed : ℝ) 
  (h_distance : distance = 15)
  (h_walking_speed : walking_speed = 6)
  (h_bicycle_speed : bicycle_speed = 15) :
  ∃ (optimal_time : ℝ),
    optimal_time = 3 / 11 ∧
    (∀ (t : ℝ), 
      let time_A := distance / walking_speed + (distance - walking_speed * t) / bicycle_speed
      let time_B := t + (distance - bicycle_speed * t) / walking_speed
      let time_C := distance / walking_speed - t
      (time_A = time_B ∧ time_B = time_C) → t = optimal_time) :=
by sorry

end NUMINAMATH_CALUDE_optimal_timing_problem_l181_18171


namespace NUMINAMATH_CALUDE_laticia_socks_l181_18176

/-- Proves that Laticia knitted 13 pairs of socks in the first week -/
theorem laticia_socks (x : ℕ) : x + (x + 4) + (x + 2) + (x - 1) = 57 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_laticia_socks_l181_18176


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l181_18140

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  parabola_eq : (y - 3)^2 = 8 * (x - 2)

/-- A circle tangent to the y-axis -/
structure TangentCircle where
  center : ParabolaPoint
  radius : ℝ
  tangent_to_y_axis : radius = center.x

theorem circle_passes_through_fixed_point (P : ParabolaPoint) (C : TangentCircle) 
  (h : C.center = P) : 
  (C.center.x - 4)^2 + (C.center.y - 3)^2 = C.radius^2 := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l181_18140


namespace NUMINAMATH_CALUDE_number_problem_l181_18179

theorem number_problem (n : ℚ) : (1/2 : ℚ) * (3/5 : ℚ) * n = 36 → n = 120 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l181_18179
