import Mathlib

namespace NUMINAMATH_CALUDE_trapezium_area_l1931_193129

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 15) (hh : h = 14) :
  (a + b) * h / 2 = 245 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_area_l1931_193129


namespace NUMINAMATH_CALUDE_tank_unoccupied_volume_l1931_193195

/-- Calculates the unoccupied volume in a cube-shaped tank --/
def unoccupied_volume (tank_side : ℝ) (water_fraction : ℝ) (ice_cube_side : ℝ) (num_ice_cubes : ℕ) : ℝ :=
  let tank_volume := tank_side ^ 3
  let water_volume := water_fraction * tank_volume
  let ice_cube_volume := ice_cube_side ^ 3
  let total_ice_volume := (num_ice_cubes : ℝ) * ice_cube_volume
  let occupied_volume := water_volume + total_ice_volume
  tank_volume - occupied_volume

/-- Theorem stating the unoccupied volume in the tank --/
theorem tank_unoccupied_volume :
  unoccupied_volume 12 (1/3) 1.5 15 = 1101.375 := by
  sorry

end NUMINAMATH_CALUDE_tank_unoccupied_volume_l1931_193195


namespace NUMINAMATH_CALUDE_tax_difference_is_correct_l1931_193194

-- Define the item price
def item_price : ℝ := 50

-- Define the tax rates
def high_tax_rate : ℝ := 0.075
def low_tax_rate : ℝ := 0.05

-- Define the tax difference function
def tax_difference (price : ℝ) (high_rate : ℝ) (low_rate : ℝ) : ℝ :=
  price * high_rate - price * low_rate

-- Theorem statement
theorem tax_difference_is_correct : 
  tax_difference item_price high_tax_rate low_tax_rate = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_tax_difference_is_correct_l1931_193194


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1931_193135

theorem geometric_sequence_sum (n : ℕ) : 
  let a : ℚ := 1/3
  let r : ℚ := 2/3
  let sum : ℚ := a * (1 - r^n) / (1 - r)
  sum = 80/243 → n = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1931_193135


namespace NUMINAMATH_CALUDE_six_digit_divisibility_difference_l1931_193143

def six_digit_lower_bound : Nat := 100000
def six_digit_upper_bound : Nat := 999999

def count_divisible (n : Nat) : Nat :=
  (six_digit_upper_bound / n) - (six_digit_lower_bound / n)

def a : Nat := count_divisible 13 - count_divisible (13 * 17)
def b : Nat := count_divisible 17 - count_divisible (13 * 17)

theorem six_digit_divisibility_difference : a - b = 16290 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_difference_l1931_193143


namespace NUMINAMATH_CALUDE_page_shoe_collection_l1931_193199

theorem page_shoe_collection (initial_shoes : ℕ) (donation_percentage : ℚ) (new_shoes : ℕ) : 
  initial_shoes = 120 →
  donation_percentage = 45 / 100 →
  new_shoes = 15 →
  initial_shoes - (initial_shoes * donation_percentage).floor + new_shoes = 81 :=
by sorry

end NUMINAMATH_CALUDE_page_shoe_collection_l1931_193199


namespace NUMINAMATH_CALUDE_complex_modulus_l1931_193196

theorem complex_modulus (z : ℂ) : z = -6 + (3 - 5/3*I)*I → Complex.abs z = 5*Real.sqrt 10/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1931_193196


namespace NUMINAMATH_CALUDE_problem_solution_l1931_193128

theorem problem_solution (a b c : ℕ) 
  (ha : a > 0 ∧ a < 10) 
  (hb : b > 0 ∧ b < 10) 
  (hc : c > 0 ∧ c < 10) 
  (h_prob : (1/a + 1/b + 1/c) - (1/a * 1/b + 1/a * 1/c + 1/b * 1/c) + (1/a * 1/b * 1/c) = 7/15) : 
  (1 - 1/a) * (1 - 1/b) * (1 - 1/c) = 8/15 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1931_193128


namespace NUMINAMATH_CALUDE_parameterized_line_matches_equation_l1931_193175

/-- A line parameterized by a point and a direction vector -/
structure ParametricLine (n : Type*) [NormedAddCommGroup n] where
  point : n
  direction : n

/-- The equation of a line in slope-intercept form -/
structure SlopeInterceptLine (α : Type*) [Field α] where
  slope : α
  intercept : α

def line_equation (l : SlopeInterceptLine ℝ) (x : ℝ) : ℝ :=
  l.slope * x + l.intercept

theorem parameterized_line_matches_equation 
  (r k : ℝ) 
  (param_line : ParametricLine (Fin 2 → ℝ))
  (slope_intercept_line : SlopeInterceptLine ℝ) :
  param_line.point = ![r, 2] ∧ 
  param_line.direction = ![3, k] ∧
  slope_intercept_line.slope = 2 ∧
  slope_intercept_line.intercept = -5 →
  r = 7/2 ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_parameterized_line_matches_equation_l1931_193175


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1931_193107

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1931_193107


namespace NUMINAMATH_CALUDE_g_sum_lower_bound_l1931_193148

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2

noncomputable def g (x : ℝ) : ℝ := f x + 3 * x + 1

theorem g_sum_lower_bound (x₁ x₂ : ℝ) (h : x₁ + x₂ ≥ 0) :
  g x₁ + g x₂ ≥ 4 := by sorry

end NUMINAMATH_CALUDE_g_sum_lower_bound_l1931_193148


namespace NUMINAMATH_CALUDE_greatest_third_term_arithmetic_sequence_l1931_193158

theorem greatest_third_term_arithmetic_sequence :
  ∀ (a d : ℕ+), 
  (a : ℕ) + (a + d : ℕ) + (a + 2 * d : ℕ) + (a + 3 * d : ℕ) = 58 →
  ∀ (b e : ℕ+),
  (b : ℕ) + (b + e : ℕ) + (b + 2 * e : ℕ) + (b + 3 * e : ℕ) = 58 →
  (a + 2 * d : ℕ) ≤ 19 :=
by sorry

end NUMINAMATH_CALUDE_greatest_third_term_arithmetic_sequence_l1931_193158


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1931_193138

/-- The equation of an ellipse with parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 + k) + y^2 / (2 - k) = 1 ∧
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

/-- The range of k for which the equation represents an ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ Set.Ioo (-3 : ℝ) (-1/2) ∪ Set.Ioo (-1/2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1931_193138


namespace NUMINAMATH_CALUDE_population_ratio_x_to_z_l1931_193173

/-- Represents the population of a city. -/
structure CityPopulation where
  value : ℕ

/-- Represents the ratio between two city populations. -/
structure PopulationRatio where
  numerator : ℕ
  denominator : ℕ

/-- Given three cities X, Y, and Z, where X's population is 8 times Y's,
    and Y's population is twice Z's, prove that the ratio of X's population
    to Z's population is 16:1. -/
theorem population_ratio_x_to_z
  (pop_x pop_y pop_z : CityPopulation)
  (h1 : pop_x.value = 8 * pop_y.value)
  (h2 : pop_y.value = 2 * pop_z.value) :
  PopulationRatio.mk 16 1 = PopulationRatio.mk (pop_x.value / pop_z.value) 1 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_x_to_z_l1931_193173


namespace NUMINAMATH_CALUDE_equation_system_solutions_l1931_193172

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(1, 5, 2, 3), (1, 5, 3, 2), (5, 1, 2, 3), (5, 1, 3, 2),
   (2, 3, 1, 5), (2, 3, 5, 1), (3, 2, 1, 5), (3, 2, 5, 1),
   (2, 2, 2, 2)}

theorem equation_system_solutions :
  ∀ x y z t : ℕ,
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 →
    x + y = z * t ∧ z + t = x * y ↔ (x, y, z, t) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_system_solutions_l1931_193172


namespace NUMINAMATH_CALUDE_propositions_are_false_l1931_193126

-- Define a type for planes
def Plane : Type := Unit

-- Define a relation for "is in"
def is_in (α β : Plane) : Prop := sorry

-- Define a relation for "is parallel to"
def is_parallel (α β : Plane) : Prop := sorry

-- Define a type for points
def Point : Type := Unit

-- Define a property for three points being non-collinear
def non_collinear (p q r : Point) : Prop := sorry

-- Define a property for a point being on a plane
def on_plane (p : Point) (α : Plane) : Prop := sorry

-- Define a property for a point being equidistant from a plane
def equidistant_from_plane (p : Point) (β : Plane) : Prop := sorry

theorem propositions_are_false :
  (∃ α β γ : Plane, is_in α β ∧ is_in β γ ∧ ¬is_parallel α γ) ∧
  (∃ α β : Plane, ∃ p q r : Point,
    non_collinear p q r ∧
    on_plane p α ∧ on_plane q α ∧ on_plane r α ∧
    equidistant_from_plane p β ∧ equidistant_from_plane q β ∧ equidistant_from_plane r β ∧
    ¬is_parallel α β) :=
by sorry

end NUMINAMATH_CALUDE_propositions_are_false_l1931_193126


namespace NUMINAMATH_CALUDE_trigonometric_equality_l1931_193153

theorem trigonometric_equality (θ α β γ x y z : ℝ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : Real.tan (θ + α) / x = Real.tan (θ + β) / y)
  (h8 : Real.tan (θ + β) / y = Real.tan (θ + γ) / z) : 
  (x + y) / (x - y) * Real.sin (α - β) ^ 2 + 
  (y + z) / (y - z) * Real.sin (β - γ) ^ 2 + 
  (z + x) / (z - x) * Real.sin (γ - α) ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l1931_193153


namespace NUMINAMATH_CALUDE_range_of_a_l1931_193112

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → 
  -1 ≤ a ∧ a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1931_193112


namespace NUMINAMATH_CALUDE_pet_store_cages_l1931_193188

def total_cages (num_snakes num_parrots num_rabbits : ℕ)
                (snakes_per_cage parrots_per_cage rabbits_per_cage : ℕ) : ℕ :=
  (num_snakes / snakes_per_cage) + (num_parrots / parrots_per_cage) + (num_rabbits / rabbits_per_cage)

theorem pet_store_cages :
  total_cages 4 6 8 2 3 4 = 6 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1931_193188


namespace NUMINAMATH_CALUDE_cannot_determine_read_sonnets_l1931_193182

/-- Represents the number of lines in a sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of unread lines -/
def unread_lines : ℕ := 70

/-- Represents the number of sonnets not read -/
def unread_sonnets : ℕ := unread_lines / lines_per_sonnet

theorem cannot_determine_read_sonnets (total_sonnets : ℕ) :
  ∀ n : ℕ, n < total_sonnets → n ≥ unread_sonnets →
  ∃ m : ℕ, m ≠ n ∧ m < total_sonnets ∧ m ≥ unread_sonnets :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_read_sonnets_l1931_193182


namespace NUMINAMATH_CALUDE_equal_area_rectangle_width_l1931_193174

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangle_width (r1 r2 : Rectangle) 
  (h1 : r1.length = 12)
  (h2 : r1.width = 10)
  (h3 : r2.length = 24)
  (h4 : area r1 = area r2) :
  r2.width = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangle_width_l1931_193174


namespace NUMINAMATH_CALUDE_equation_solutions_l1931_193120

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 81 = 0 ↔ x = 9/2 ∨ x = -9/2) ∧
  (∀ x l : ℝ, 64 * (x + l)^3 = 27 → x = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1931_193120


namespace NUMINAMATH_CALUDE_tic_tac_toe_rounds_l1931_193134

/-- Given that William won 10 rounds of tic-tac-toe and 5 more rounds than Harry,
    prove that the total number of rounds played is 15. -/
theorem tic_tac_toe_rounds (william_rounds harry_rounds total_rounds : ℕ) 
  (h1 : william_rounds = 10)
  (h2 : william_rounds = harry_rounds + 5) : 
  total_rounds = 15 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_rounds_l1931_193134


namespace NUMINAMATH_CALUDE_value_of_q_l1931_193132

theorem value_of_q (a q : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * q) : q = 49 := by
  sorry

end NUMINAMATH_CALUDE_value_of_q_l1931_193132


namespace NUMINAMATH_CALUDE_sum_f_negative_l1931_193115

def f (x : ℝ) := -x - x^3

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l1931_193115


namespace NUMINAMATH_CALUDE_henry_returned_half_l1931_193192

/-- The portion of catch Henry returned -/
def henryReturnedPortion (willCatfish : ℕ) (willEels : ℕ) (henryTroutPerCatfish : ℕ) (totalFishAfterReturn : ℕ) : ℚ :=
  let willTotal := willCatfish + willEels
  let henryTotal := willCatfish * henryTroutPerCatfish
  let totalBeforeReturn := willTotal + henryTotal
  let returnedFish := totalBeforeReturn - totalFishAfterReturn
  returnedFish / henryTotal

/-- Theorem stating that Henry returned half of his catch -/
theorem henry_returned_half :
  henryReturnedPortion 16 10 3 50 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_henry_returned_half_l1931_193192


namespace NUMINAMATH_CALUDE_squirrels_and_nuts_l1931_193102

theorem squirrels_and_nuts (squirrels : ℕ) (nuts : ℕ) : 
  squirrels = 4 → squirrels = nuts + 2 → nuts = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_and_nuts_l1931_193102


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_max_area_l1931_193152

/-- Given a right triangle with legs of length a and hypotenuse of length c,
    the area of the triangle is maximized when the legs are equal. -/
theorem isosceles_right_triangle_max_area (a c : ℝ) (h1 : 0 < a) (h2 : 0 < c) :
  let area := (1/2) * a * (c^2 - a^2).sqrt
  ∀ b, 0 < b → b^2 + a^2 = c^2 → area ≥ (1/2) * a * b :=
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_max_area_l1931_193152


namespace NUMINAMATH_CALUDE_decreasing_function_condition_l1931_193198

-- Define the function f(x)
def f (k x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

-- Define the derivative of f(x)
def f_derivative (k x : ℝ) : ℝ := 3 * k * x^2 + 6 * (k - 1) * x

-- Theorem statement
theorem decreasing_function_condition (k : ℝ) :
  (∀ x ∈ Set.Ioo 0 4, f_derivative k x ≤ 0) ↔ k ≤ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_condition_l1931_193198


namespace NUMINAMATH_CALUDE_room_dimension_l1931_193190

/-- Proves that a square room with an area of 14400 square inches has sides of length 10 feet, given that there are 12 inches in a foot. -/
theorem room_dimension (inches_per_foot : ℕ) (area_sq_inches : ℕ) : 
  inches_per_foot = 12 → 
  area_sq_inches = 14400 → 
  ∃ (side_length : ℕ), side_length * side_length * (inches_per_foot * inches_per_foot) = area_sq_inches ∧ 
                        side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_room_dimension_l1931_193190


namespace NUMINAMATH_CALUDE_shaded_area_sum_l1931_193125

/-- Represents the shaded area between a circle and an inscribed equilateral triangle --/
structure ShadedArea where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the shaded area for a given circle and inscribed equilateral triangle --/
def calculateShadedArea (sideLength : ℝ) : ShadedArea :=
  { a := 18.75,
    b := 21,
    c := 3 }

/-- Theorem stating the sum of a, b, and c for the given problem --/
theorem shaded_area_sum (sideLength : ℝ) :
  sideLength = 15 →
  let area := calculateShadedArea sideLength
  area.a + area.b + area.c = 42.75 := by
  sorry

#check shaded_area_sum

end NUMINAMATH_CALUDE_shaded_area_sum_l1931_193125


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l1931_193144

theorem paper_clip_distribution (total_clips : ℕ) (clips_per_box : ℕ) (boxes_needed : ℕ) :
  total_clips = 81 →
  clips_per_box = 9 →
  total_clips = clips_per_box * boxes_needed →
  boxes_needed = 9 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l1931_193144


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l1931_193166

-- Define complex numbers a and b
variable (a b : ℂ)

-- Define real number t
variable (t : ℝ)

-- State the theorem
theorem complex_product_magnitude 
  (h1 : Complex.abs a = 2)
  (h2 : Complex.abs b = 5)
  (h3 : a * b = t - 3 * Complex.I)
  (h4 : t > 0) :
  t = Real.sqrt 91 := by
sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l1931_193166


namespace NUMINAMATH_CALUDE_average_after_exclusion_l1931_193186

theorem average_after_exclusion (numbers : Finset ℕ) (sum : ℕ) (excluded : ℕ) :
  numbers.card = 5 →
  sum / numbers.card = 27 →
  excluded ∈ numbers →
  excluded = 35 →
  (sum - excluded) / (numbers.card - 1) = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_after_exclusion_l1931_193186


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l1931_193111

theorem power_of_two_divisibility (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ r : ℕ, n.val = 2^r :=
sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l1931_193111


namespace NUMINAMATH_CALUDE_colored_integers_theorem_l1931_193176

def ColoredInteger := ℤ → Bool

theorem colored_integers_theorem (color : ColoredInteger) 
  (h1 : color 1 = true)
  (h2 : ∀ a b : ℤ, color a = true → color b = true → color (a + b) ≠ color (a - b)) :
  color 2011 = true := by sorry

end NUMINAMATH_CALUDE_colored_integers_theorem_l1931_193176


namespace NUMINAMATH_CALUDE_number_2009_in_group_31_l1931_193122

/-- The sum of squares of the first n odd numbers -/
def O (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 3

/-- The number we're looking for -/
def target : ℕ := 2009

/-- The group number we're proving -/
def group_number : ℕ := 31

theorem number_2009_in_group_31 :
  O (group_number - 1) < target ∧ target ≤ O group_number :=
sorry

end NUMINAMATH_CALUDE_number_2009_in_group_31_l1931_193122


namespace NUMINAMATH_CALUDE_line_properties_l1931_193185

/-- A line passing through a point with given conditions -/
structure Line where
  P : ℝ × ℝ
  α : ℝ
  intersects_positive_axes : Bool
  PA_PB_product : ℝ

/-- The main theorem stating the properties of the line -/
theorem line_properties (l : Line) 
  (h1 : l.P = (2, 1))
  (h2 : l.intersects_positive_axes = true)
  (h3 : l.PA_PB_product = 4) :
  (l.α = 3 * Real.pi / 4) ∧ 
  (∃ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 3) := by
  sorry

#check line_properties

end NUMINAMATH_CALUDE_line_properties_l1931_193185


namespace NUMINAMATH_CALUDE_scalene_polygon_existence_l1931_193161

theorem scalene_polygon_existence (n : ℕ) : 
  (n ≥ 13) → 
  (∀ (S : Finset ℝ), 
    (S.card = n) → 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 2013) → 
    ∃ (a b c : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
      a + b > c ∧ b + c > a ∧ a + c > b) ∧
  (n = 13) :=
sorry

end NUMINAMATH_CALUDE_scalene_polygon_existence_l1931_193161


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1931_193103

/-- Given a natural number, returns true if it ends with 56 -/
def ends_with_56 (n : ℕ) : Prop :=
  n % 100 = 56

/-- Given a natural number, returns the sum of its digits -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem smallest_number_with_conditions :
  ∃ (n : ℕ), 
    ends_with_56 n ∧ 
    n % 56 = 0 ∧ 
    digit_sum n = 56 ∧
    (∀ m : ℕ, m < n → ¬(ends_with_56 m ∧ m % 56 = 0 ∧ digit_sum m = 56)) ∧
    n = 29899856 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1931_193103


namespace NUMINAMATH_CALUDE_triple_sharp_fifty_l1931_193140

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N - 2

-- Theorem statement
theorem triple_sharp_fifty : sharp (sharp (sharp 50)) = 6.88 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_fifty_l1931_193140


namespace NUMINAMATH_CALUDE_bing_duan_duan_properties_l1931_193184

/-- Represents the production and sales of "Bing Duan Duan" mascots --/
structure BingDuanDuan where
  feb_production : ℕ
  apr_production : ℕ
  daily_sales : ℕ
  profit_per_item : ℕ
  sales_increase : ℕ
  max_price_reduction : ℕ
  target_daily_profit : ℕ

/-- Calculates the monthly growth rate given February and April production --/
def monthly_growth_rate (b : BingDuanDuan) : ℚ :=
  ((b.apr_production : ℚ) / b.feb_production) ^ (1/2) - 1

/-- Calculates the optimal price reduction --/
def optimal_price_reduction (b : BingDuanDuan) : ℕ :=
  sorry -- The actual calculation would go here

/-- Theorem stating the properties of BingDuanDuan production and sales --/
theorem bing_duan_duan_properties (b : BingDuanDuan) 
  (h1 : b.feb_production = 500)
  (h2 : b.apr_production = 720)
  (h3 : b.daily_sales = 20)
  (h4 : b.profit_per_item = 40)
  (h5 : b.sales_increase = 5)
  (h6 : b.max_price_reduction = 10)
  (h7 : b.target_daily_profit = 1440) :
  monthly_growth_rate b = 1/5 ∧ 
  optimal_price_reduction b = 4 ∧ 
  optimal_price_reduction b ≤ b.max_price_reduction :=
by sorry


end NUMINAMATH_CALUDE_bing_duan_duan_properties_l1931_193184


namespace NUMINAMATH_CALUDE_det_special_matrix_l1931_193133

theorem det_special_matrix (x : ℝ) : 
  Matrix.det (![![x + 3, x, x], ![x, x + 3, x], ![x, x, x + 3]]) = 27 * x + 27 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l1931_193133


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1931_193117

/-- The area of a triangle given its three altitudes --/
def triangle_area_from_altitudes (h₁ h₂ h₃ : ℝ) : ℝ := sorry

/-- A triangle with altitudes 36.4, 39, and 42 has an area of 3549/4 --/
theorem triangle_area_theorem :
  triangle_area_from_altitudes 36.4 39 42 = 3549 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1931_193117


namespace NUMINAMATH_CALUDE_ticket_sales_total_l1931_193124

/-- Calculates the total money collected from ticket sales -/
def total_money_collected (student_price general_price : ℕ) (total_tickets general_tickets : ℕ) : ℕ :=
  let student_tickets := total_tickets - general_tickets
  student_tickets * student_price + general_tickets * general_price

/-- Theorem stating that the total money collected is 2876 given the specific conditions -/
theorem ticket_sales_total :
  total_money_collected 4 6 525 388 = 2876 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_total_l1931_193124


namespace NUMINAMATH_CALUDE_line_equation_l1931_193109

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a line has equal intercepts on both axes -/
def Line.equalIntercepts (l : Line) : Prop :=
  l.a * l.c = -l.b * l.c

theorem line_equation (P : Point) (l : Line) :
  P.x = 2 ∧ P.y = 1 ∧
  P.onLine l ∧
  l.perpendicular { a := 1, b := -1, c := 1 } ∧
  l.equalIntercepts →
  (l = { a := 1, b := 1, c := -3 } ∨ l = { a := 1, b := -2, c := 0 }) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1931_193109


namespace NUMINAMATH_CALUDE_sqrt_72_plus_24sqrt6_l1931_193181

theorem sqrt_72_plus_24sqrt6 :
  ∃ (a b c : ℤ), (c > 0) ∧ 
  (∀ (n : ℕ), n > 1 → ¬(∃ (k : ℕ), c = n^2 * k)) ∧
  Real.sqrt (72 + 24 * Real.sqrt 6) = a + b * Real.sqrt c ∧
  a = 6 ∧ b = 3 ∧ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_72_plus_24sqrt6_l1931_193181


namespace NUMINAMATH_CALUDE_point_coordinates_l1931_193156

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ := |p.y|

/-- The distance of a point from the y-axis -/
def distanceFromYAxis (p : Point) : ℝ := |p.x|

/-- Determines if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y > 0

/-- Theorem: Given the conditions, the point M has coordinates (-2, 3) -/
theorem point_coordinates (M : Point)
    (h1 : distanceFromXAxis M = 3)
    (h2 : distanceFromYAxis M = 2)
    (h3 : isInSecondQuadrant M) :
    M = Point.mk (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1931_193156


namespace NUMINAMATH_CALUDE_karen_tagalongs_sales_l1931_193108

/-- The number of cases Karen picked up -/
def num_cases : ℕ := 3

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 12

/-- The total number of boxes Karen sold -/
def total_boxes : ℕ := num_cases * boxes_per_case

theorem karen_tagalongs_sales : total_boxes = 36 := by
  sorry

end NUMINAMATH_CALUDE_karen_tagalongs_sales_l1931_193108


namespace NUMINAMATH_CALUDE_no_common_values_under_180_l1931_193179

theorem no_common_values_under_180 : 
  ¬ ∃ x : ℕ, x < 180 ∧ x % 13 = 2 ∧ x % 8 = 5 := by
sorry

end NUMINAMATH_CALUDE_no_common_values_under_180_l1931_193179


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1931_193155

theorem sum_of_solutions (x : ℝ) : 
  (Real.sqrt x + Real.sqrt (9 / x) + Real.sqrt (x + 9 / x) = 7) → 
  (∃ y : ℝ, x^2 - (49/4) * x + 9 = 0 ∧ y^2 - (49/4) * y + 9 = 0 ∧ x + y = 49/4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1931_193155


namespace NUMINAMATH_CALUDE_dimes_per_quarter_l1931_193193

/-- Represents the number of coins traded for a quarter -/
structure TradeRatio :=
  (dimes : ℚ)
  (nickels : ℚ)

/-- Calculates the total value of coins traded -/
def totalValue (ratio : TradeRatio) : ℚ :=
  20 * (ratio.dimes * (1/10) + ratio.nickels * (1/20))

/-- Theorem: The number of dimes traded for each quarter is 4 -/
theorem dimes_per_quarter :
  ∃ (ratio : TradeRatio),
    totalValue ratio = 10 + 3 ∧
    ratio.nickels = 5 ∧
    ratio.dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_dimes_per_quarter_l1931_193193


namespace NUMINAMATH_CALUDE_binomial_60_2_l1931_193142

theorem binomial_60_2 : Nat.choose 60 2 = 1770 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_2_l1931_193142


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l1931_193141

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 12) (h2 : n2 = 28) (h3 : avg1 = 40) (h4 : avg2 = 60) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 54 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l1931_193141


namespace NUMINAMATH_CALUDE_logans_father_cartons_l1931_193136

/-- The number of cartons Logan's father usually receives -/
def usual_cartons : ℕ := 50

/-- The number of jars in each carton -/
def jars_per_carton : ℕ := 20

/-- The number of cartons received in the particular week -/
def received_cartons : ℕ := usual_cartons - 20

/-- The number of damaged jars from partially damaged cartons -/
def partially_damaged_jars : ℕ := 5 * 3

/-- The number of damaged jars from the totally damaged carton -/
def totally_damaged_jars : ℕ := jars_per_carton

/-- The total number of damaged jars -/
def total_damaged_jars : ℕ := partially_damaged_jars + totally_damaged_jars

/-- The number of jars good for sale in the particular week -/
def good_jars : ℕ := 565

theorem logans_father_cartons :
  jars_per_carton * received_cartons - total_damaged_jars = good_jars :=
by sorry

end NUMINAMATH_CALUDE_logans_father_cartons_l1931_193136


namespace NUMINAMATH_CALUDE_train_passing_jogger_l1931_193180

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger (v_jogger v_train : ℝ) (train_length initial_distance : ℝ) :
  v_jogger = 10 * 1000 / 3600 →
  v_train = 46 * 1000 / 3600 →
  train_length = 120 →
  initial_distance = 340 →
  (initial_distance + train_length) / (v_train - v_jogger) = 46 := by
  sorry

#check train_passing_jogger

end NUMINAMATH_CALUDE_train_passing_jogger_l1931_193180


namespace NUMINAMATH_CALUDE_lg_sum_equals_zero_l1931_193118

-- Define lg as the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_zero : lg 5 + lg 0.2 = 0 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_zero_l1931_193118


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l1931_193163

/-- The general term formula for the sequence -1/2, 1/4, -1/8, 1/16, ... -/
def sequence_formula (n : ℕ) : ℚ := (-1)^(n+1) / (2^n)

/-- The nth term of the sequence -1/2, 1/4, -1/8, 1/16, ... -/
def sequence_term (n : ℕ) : ℚ := 
  if n % 2 = 1 
  then -1 / (2^n) 
  else 1 / (2^n)

theorem sequence_formula_correct : 
  ∀ n : ℕ, n > 0 → sequence_formula n = sequence_term n :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l1931_193163


namespace NUMINAMATH_CALUDE_total_steps_in_week_l1931_193167

/-- Represents the number of steps taken to school and back for each day -/
structure DailySteps where
  toSchool : ℕ
  fromSchool : ℕ

/-- Calculates the total steps for a given day -/
def totalSteps (day : DailySteps) : ℕ := day.toSchool + day.fromSchool

/-- Represents Raine's walking data for the week -/
structure WeeklyWalk where
  monday : DailySteps
  tuesday : DailySteps
  wednesday : DailySteps
  thursday : DailySteps
  friday : DailySteps

/-- The actual walking data for Raine's week -/
def rainesWeek : WeeklyWalk := {
  monday := { toSchool := 150, fromSchool := 170 }
  tuesday := { toSchool := 140, fromSchool := 170 }  -- 140 + 30 rest stop
  wednesday := { toSchool := 160, fromSchool := 210 }
  thursday := { toSchool := 150, fromSchool := 170 }  -- 140 + 30 rest stop
  friday := { toSchool := 180, fromSchool := 200 }
}

/-- Theorem: The total number of steps Raine takes in five days is 1700 -/
theorem total_steps_in_week (w : WeeklyWalk := rainesWeek) :
  totalSteps w.monday + totalSteps w.tuesday + totalSteps w.wednesday +
  totalSteps w.thursday + totalSteps w.friday = 1700 := by
  sorry

end NUMINAMATH_CALUDE_total_steps_in_week_l1931_193167


namespace NUMINAMATH_CALUDE_book_pages_book_has_120_pages_l1931_193145

theorem book_pages : ℕ → Prop :=
  fun total_pages =>
    let pages_yesterday : ℕ := 12
    let pages_today : ℕ := 2 * pages_yesterday
    let pages_read : ℕ := pages_yesterday + pages_today
    let pages_tomorrow : ℕ := 42
    let remaining_pages : ℕ := 2 * pages_tomorrow
    total_pages = pages_read + remaining_pages ∧ total_pages = 120

-- The proof of the theorem
theorem book_has_120_pages : ∃ (n : ℕ), book_pages n := by
  sorry

end NUMINAMATH_CALUDE_book_pages_book_has_120_pages_l1931_193145


namespace NUMINAMATH_CALUDE_transformation_of_point_l1931_193100

/-- Given a point A and a transformation φ, prove that the transformed point A' has specific coordinates -/
theorem transformation_of_point (x y x' y' : ℚ) : 
  x = 1/3 ∧ y = -2 ∧ x' = 3*x ∧ 2*y' = y → x' = 1 ∧ y' = -1 := by
  sorry

end NUMINAMATH_CALUDE_transformation_of_point_l1931_193100


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourths_l1931_193116

theorem tan_seven_pi_fourths : Real.tan (7 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourths_l1931_193116


namespace NUMINAMATH_CALUDE_valid_word_count_mod_2000_l1931_193119

/-- Represents a letter in Zuminglish --/
inductive ZuminglishLetter
| M
| O
| P

/-- Represents whether a letter is a vowel or consonant --/
def isVowel : ZuminglishLetter → Bool
| ZuminglishLetter.O => true
| _ => false

/-- A Zuminglish word is a list of Zuminglish letters --/
def ZuminglishWord := List ZuminglishLetter

/-- Checks if a Zuminglish word is valid (no two O's are adjacent without at least two consonants in between) --/
def isValidWord : ZuminglishWord → Bool := sorry

/-- Counts the number of valid 12-letter Zuminglish words --/
def countValidWords : Nat := sorry

/-- The main theorem: The number of valid 12-letter Zuminglish words is congruent to 192 modulo 2000 --/
theorem valid_word_count_mod_2000 : countValidWords % 2000 = 192 := by sorry

end NUMINAMATH_CALUDE_valid_word_count_mod_2000_l1931_193119


namespace NUMINAMATH_CALUDE_undefined_values_count_l1931_193123

theorem undefined_values_count : ∃! (s : Finset ℤ), 
  (∀ x ∈ s, (x^2 - x - 6) * (x - 4) = 0) ∧ 
  (∀ x ∉ s, (x^2 - x - 6) * (x - 4) ≠ 0) ∧ 
  s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_values_count_l1931_193123


namespace NUMINAMATH_CALUDE_initial_toys_count_l1931_193165

/-- 
Given that Emily sold some toys and has some left, this theorem proves
the initial number of toys she had.
-/
theorem initial_toys_count 
  (sold : ℕ) -- Number of toys sold
  (remaining : ℕ) -- Number of toys remaining
  (h1 : sold = 3) -- Condition: Emily sold 3 toys
  (h2 : remaining = 4) -- Condition: Emily now has 4 toys left
  : sold + remaining = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_toys_count_l1931_193165


namespace NUMINAMATH_CALUDE_sqrt_five_decomposition_l1931_193189

theorem sqrt_five_decomposition (a : ℤ) (b : ℝ) 
  (h1 : Real.sqrt 5 = a + b) 
  (h2 : 0 < b) 
  (h3 : b < 1) : 
  (a - b) * (4 + Real.sqrt 5) = 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_decomposition_l1931_193189


namespace NUMINAMATH_CALUDE_orange_balls_count_l1931_193101

theorem orange_balls_count (total green red blue yellow pink orange purple : ℕ) :
  total = 120 ∧
  green = 5 ∧
  red = 30 ∧
  blue = 20 ∧
  yellow = 10 ∧
  pink = 2 * green ∧
  orange = 3 * pink ∧
  purple = orange - pink ∧
  total = red + blue + yellow + green + pink + orange + purple →
  orange = 30 := by
sorry

end NUMINAMATH_CALUDE_orange_balls_count_l1931_193101


namespace NUMINAMATH_CALUDE_morley_theorem_l1931_193187

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a ray (half-line) -/
structure Ray :=
  (origin : Point) (direction : Point)

/-- Defines a trisector of an angle -/
def is_trisector (r : Ray) (A B C : Point) : Prop := sorry

/-- Defines the intersection point of two rays -/
def intersection (r1 r2 : Ray) : Point := sorry

/-- Morley's theorem -/
theorem morley_theorem (T : Triangle) :
  let A := T.A
  let B := T.B
  let C := T.C
  let trisector_B1 := Ray.mk B (sorry : Point)
  let trisector_B2 := Ray.mk B (sorry : Point)
  let trisector_C1 := Ray.mk C (sorry : Point)
  let trisector_C2 := Ray.mk C (sorry : Point)
  let trisector_A1 := Ray.mk A (sorry : Point)
  let trisector_A2 := Ray.mk A (sorry : Point)
  let A1 := intersection trisector_B1 trisector_C1
  let B1 := intersection trisector_C2 trisector_A1
  let C1 := intersection trisector_A2 trisector_B2
  is_trisector trisector_B1 B A C ∧
  is_trisector trisector_B2 B A C ∧
  is_trisector trisector_C1 C B A ∧
  is_trisector trisector_C2 C B A ∧
  is_trisector trisector_A1 A C B ∧
  is_trisector trisector_A2 A C B →
  -- A1B1 = B1C1 = C1A1
  (A1.x - B1.x)^2 + (A1.y - B1.y)^2 =
  (B1.x - C1.x)^2 + (B1.y - C1.y)^2 ∧
  (B1.x - C1.x)^2 + (B1.y - C1.y)^2 =
  (C1.x - A1.x)^2 + (C1.y - A1.y)^2 :=
sorry

end NUMINAMATH_CALUDE_morley_theorem_l1931_193187


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1931_193131

theorem algebraic_expression_value (m n : ℝ) (h : 2*m - 3*n = -2) :
  4*m - 6*n + 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1931_193131


namespace NUMINAMATH_CALUDE_three_digit_reverse_double_l1931_193121

theorem three_digit_reverse_double (g : ℕ) (a b c : ℕ) : 
  (0 < g) → 
  (a < g) → (b < g) → (c < g) →
  (a * g^2 + b * g + c = 2 * (c * g^2 + b * g + a)) →
  ∃ k : ℕ, (k > 0) ∧ (g = 3 * k + 2) := by
sorry


end NUMINAMATH_CALUDE_three_digit_reverse_double_l1931_193121


namespace NUMINAMATH_CALUDE_dot_product_sum_l1931_193171

theorem dot_product_sum (a b : ℝ × ℝ × ℝ) (h1 : a = (0, 2, 0)) (h2 : b = (1, 0, -1)) :
  (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2) • b = 2 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_sum_l1931_193171


namespace NUMINAMATH_CALUDE_snail_return_whole_hours_l1931_193154

/-- Represents the snail's movement on a 2D plane -/
structure SnailMovement where
  speed : ℝ
  turnInterval : ℝ
  turnAngle : ℝ

/-- Represents the snail's position on a 2D plane -/
structure Position where
  x : ℝ
  y : ℝ

/-- Calculates the snail's position after a given time -/
def snailPosition (movement : SnailMovement) (time : ℝ) : Position :=
  sorry

/-- Theorem: The snail returns to its starting point only after a whole number of hours -/
theorem snail_return_whole_hours (movement : SnailMovement) 
    (h1 : movement.speed > 0)
    (h2 : movement.turnInterval = 1/4)
    (h3 : movement.turnAngle = π/2) :
  ∀ t : ℝ, snailPosition movement t = snailPosition movement 0 → ∃ n : ℕ, t = n :=
  sorry

end NUMINAMATH_CALUDE_snail_return_whole_hours_l1931_193154


namespace NUMINAMATH_CALUDE_problem_statement_l1931_193146

theorem problem_statement (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) :
  (b > 1) ∧ 
  (∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → a + b ≤ x + y) ∧ 
  (∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → a * b ≤ x * y) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1931_193146


namespace NUMINAMATH_CALUDE_f_neg_nine_eq_neg_one_l1931_193114

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the properties of the function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  ∃ b : ℝ, 
    (∀ x : ℝ, f (-x) = -f x) ∧ 
    (∀ x : ℝ, x ≥ 0 → f x = lg (x + 1) - b)

-- State the theorem
theorem f_neg_nine_eq_neg_one (f : ℝ → ℝ) (h : is_valid_f f) : f (-9) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_nine_eq_neg_one_l1931_193114


namespace NUMINAMATH_CALUDE_manufacturing_sector_degrees_l1931_193177

theorem manufacturing_sector_degrees (total_degrees : ℝ) (total_percent : ℝ) 
  (manufacturing_percent : ℝ) (h1 : total_degrees = 360) 
  (h2 : total_percent = 100) (h3 : manufacturing_percent = 60) : 
  (manufacturing_percent / total_percent) * total_degrees = 216 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_sector_degrees_l1931_193177


namespace NUMINAMATH_CALUDE_grid_sum_equality_l1931_193160

theorem grid_sum_equality (row1 row2 : List ℕ) (x : ℕ) :
  row1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1050] →
  row2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, x] →
  row1.sum = row2.sum →
  x = 950 := by
  sorry

end NUMINAMATH_CALUDE_grid_sum_equality_l1931_193160


namespace NUMINAMATH_CALUDE_petes_number_l1931_193169

theorem petes_number : ∃ x : ℝ, 4 * (2 * x + 20) = 200 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l1931_193169


namespace NUMINAMATH_CALUDE_shortest_distance_to_circle_l1931_193147

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 6*y + 9 = 0

/-- The shortest distance from the origin to the circle -/
def shortest_distance : ℝ := 1

/-- Theorem: The shortest distance from the origin to the circle defined by
    x^2 - 8x + y^2 + 6y + 9 = 0 is equal to 1 -/
theorem shortest_distance_to_circle :
  ∀ (x y : ℝ), circle_equation x y →
  ∃ (p : ℝ × ℝ), p ∈ {(x, y) | circle_equation x y} ∧
  ∀ (q : ℝ × ℝ), q ∈ {(x, y) | circle_equation x y} →
  Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) ∧
  Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) = shortest_distance :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_circle_l1931_193147


namespace NUMINAMATH_CALUDE_today_is_thursday_l1931_193162

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define the vehicles
inductive Vehicle
  | A
  | B
  | C
  | D
  | E

def is_weekday (d : Day) : Prop :=
  d ≠ Day.Saturday ∧ d ≠ Day.Sunday

def next_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

def can_operate (v : Vehicle) (d : Day) : Prop := sorry

theorem today_is_thursday 
  (h1 : ∀ (d : Day), is_weekday d → ∃ (v : Vehicle), ¬can_operate v d)
  (h2 : ∀ (d : Day), is_weekday d → (∃ (v1 v2 v3 v4 : Vehicle), can_operate v1 d ∧ can_operate v2 d ∧ can_operate v3 d ∧ can_operate v4 d))
  (h3 : ¬can_operate Vehicle.E Day.Thursday)
  (h4 : ¬can_operate Vehicle.B (next_day today))
  (h5 : ∀ (d : Day), d = today ∨ d = next_day today ∨ d = next_day (next_day today) ∨ d = next_day (next_day (next_day today)) → can_operate Vehicle.A d ∧ can_operate Vehicle.C d)
  (h6 : can_operate Vehicle.E (next_day today))
  : today = Day.Thursday :=
sorry

end NUMINAMATH_CALUDE_today_is_thursday_l1931_193162


namespace NUMINAMATH_CALUDE_min_value_of_f_l1931_193104

def f (x : ℝ) : ℝ := |2*x + 1| + |x - 1|

theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = 3/2 ∧ ∀ (x : ℝ), f x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1931_193104


namespace NUMINAMATH_CALUDE_intersection_distance_approx_l1931_193113

-- Define the centers of the circles
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (4, 0)
def D : ℝ × ℝ := (5, 0)

-- Define the radii of the circles
def radius_A : ℝ := 2
def radius_B : ℝ := 2
def radius_C : ℝ := 3
def radius_D : ℝ := 3

-- Define the equations of the circles
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = radius_A^2
def circle_C (x y : ℝ) : Prop := (x - C.1)^2 + y^2 = radius_C^2
def circle_D (x y : ℝ) : Prop := (x - D.1)^2 + y^2 = radius_D^2

-- Define the intersection points
def B' : ℝ × ℝ := sorry
def D' : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_distance_approx :
  ∃ ε > 0, abs (Real.sqrt ((B'.1 - D'.1)^2 + (B'.2 - D'.2)^2) - 0.8) < ε :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_approx_l1931_193113


namespace NUMINAMATH_CALUDE_total_kids_l1931_193105

theorem total_kids (girls : ℕ) (boys : ℕ) (h1 : girls = 3) (h2 : boys = 6) :
  girls + boys = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_kids_l1931_193105


namespace NUMINAMATH_CALUDE_a_plus_b_equals_zero_l1931_193159

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M (a : ℝ) : Set ℝ := {x | x^2 + a*x ≤ 0}

-- Define the complement of M in U
def C_U_M (a b : ℝ) : Set ℝ := {x | x > b ∨ x < 0}

-- Theorem statement
theorem a_plus_b_equals_zero (a b : ℝ) : 
  (∀ x, x ∈ M a ↔ x ∉ C_U_M a b) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_zero_l1931_193159


namespace NUMINAMATH_CALUDE_stating_adjacent_probability_in_grid_l1931_193130

/-- The number of students -/
def num_students : ℕ := 8

/-- The number of rows in the seating arrangement -/
def num_rows : ℕ := 2

/-- The number of columns in the seating arrangement -/
def num_columns : ℕ := 4

/-- The probability of two specific students being adjacent -/
def adjacent_probability : ℚ := 5/14

/-- 
Theorem stating that the probability of two specific students 
being adjacent in a random seating arrangement is 5/14
-/
theorem adjacent_probability_in_grid : 
  let total_arrangements := Nat.factorial num_students
  let row_adjacent_pairs := num_rows * (num_columns - 1)
  let column_adjacent_pairs := num_columns
  let ways_to_arrange_pair := 2
  let remaining_arrangements := Nat.factorial (num_students - 2)
  let favorable_outcomes := (row_adjacent_pairs + column_adjacent_pairs) * 
                            ways_to_arrange_pair * 
                            remaining_arrangements
  (favorable_outcomes : ℚ) / total_arrangements = adjacent_probability := by
  sorry

end NUMINAMATH_CALUDE_stating_adjacent_probability_in_grid_l1931_193130


namespace NUMINAMATH_CALUDE_angle_equality_l1931_193197

-- Define the types for points and angles
variable (Point Angle : Type)

-- Define the triangle ABC
variable (A B C : Point)

-- Define the points on the sides of the triangle
variable (P₁ P₂ Q₁ Q₂ R S M : Point)

-- Define the necessary geometric predicates
variable (lies_on : Point → Point → Point → Prop)
variable (is_midpoint : Point → Point → Point → Prop)
variable (angle : Point → Point → Point → Angle)
variable (length_eq : Point → Point → Point → Point → Prop)
variable (intersects : Point → Point → Point → Point → Point → Prop)
variable (on_circumcircle : Point → Point → Point → Point → Prop)
variable (inside_triangle : Point → Point → Point → Point → Prop)

-- State the theorem
theorem angle_equality 
  (h1 : lies_on P₁ A B) (h2 : lies_on P₂ A B) (h3 : lies_on P₂ B P₁)
  (h4 : length_eq A P₁ B P₂)
  (h5 : lies_on Q₁ B C) (h6 : lies_on Q₂ B C) (h7 : lies_on Q₂ B Q₁)
  (h8 : length_eq B Q₁ C Q₂)
  (h9 : intersects P₁ Q₂ P₂ Q₁ R)
  (h10 : on_circumcircle S P₁ P₂ R) (h11 : on_circumcircle S Q₁ Q₂ R)
  (h12 : inside_triangle S P₁ Q₁ R)
  (h13 : is_midpoint M A C) :
  angle P₁ R S = angle Q₁ R M :=
by sorry

end NUMINAMATH_CALUDE_angle_equality_l1931_193197


namespace NUMINAMATH_CALUDE_tube_length_doubles_pressure_l1931_193150

/-- The length of the tube that doubles the pressure at the bottom of a water-filled barrel. -/
theorem tube_length_doubles_pressure (h₁ : ℝ) (m : ℝ) (ρ : ℝ) (g : ℝ) :
  h₁ = 1.5 →  -- height of the barrel in meters
  m = 1000 →  -- mass of water in the barrel in kg
  ρ = 1000 →  -- density of water in kg/m³
  g = 9.8 →   -- acceleration due to gravity in m/s²
  ∃ h₂ : ℝ,   -- height of water in the tube
    h₂ = 1.5 ∧ ρ * g * (h₁ + h₂) = 2 * (ρ * g * h₁) :=
by sorry

end NUMINAMATH_CALUDE_tube_length_doubles_pressure_l1931_193150


namespace NUMINAMATH_CALUDE_probability_is_four_twentysevenths_l1931_193168

/-- A regular tetrahedron with painted stripes -/
structure StripedTetrahedron where
  /-- The number of faces in a tetrahedron -/
  num_faces : Nat
  /-- The number of possible stripe configurations per face -/
  stripes_per_face : Nat
  /-- The total number of possible stripe configurations -/
  total_configurations : Nat
  /-- The number of configurations that form a continuous stripe -/
  continuous_configurations : Nat

/-- The probability of a continuous stripe encircling the tetrahedron -/
def probability_continuous_stripe (t : StripedTetrahedron) : Rat :=
  t.continuous_configurations / t.total_configurations

/-- Theorem stating the probability of a continuous stripe encircling the tetrahedron -/
theorem probability_is_four_twentysevenths (t : StripedTetrahedron) 
  (h1 : t.num_faces = 4)
  (h2 : t.stripes_per_face = 3)
  (h3 : t.total_configurations = t.stripes_per_face ^ t.num_faces)
  (h4 : t.continuous_configurations = 12) : 
  probability_continuous_stripe t = 4 / 27 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_four_twentysevenths_l1931_193168


namespace NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l1931_193157

/-- Represents the number of free ends after k iterations of drawing segments -/
def freeEnds (k : ℕ) : ℕ := 2 + 4 * k

/-- Theorem stating that there exists a positive integer k such that
    the number of free ends after k iterations is 1001 -/
theorem exists_k_for_1001_free_ends :
  ∃ k : ℕ, k > 0 ∧ freeEnds k = 1001 :=
sorry

end NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l1931_193157


namespace NUMINAMATH_CALUDE_mountain_bike_price_l1931_193127

theorem mountain_bike_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 → 
  upfront_percentage = 20 → 
  upfront_payment = (upfront_percentage / 100) * total_price → 
  total_price = 1200 := by
sorry

end NUMINAMATH_CALUDE_mountain_bike_price_l1931_193127


namespace NUMINAMATH_CALUDE_prime_expressions_l1931_193137

theorem prime_expressions (p : ℤ) : 
  Prime p ∧ Prime (2*p + 1) ∧ Prime (4*p + 1) ∧ Prime (6*p + 1) ↔ p = -2 ∨ p = -3 ∨ p = 3 :=
sorry

end NUMINAMATH_CALUDE_prime_expressions_l1931_193137


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1931_193139

/-- 
Given a rectangular field with one side uncovered and three sides fenced,
prove that the area of the field is 720 square feet when the uncovered side
is 20 feet and the total fencing is 92 feet.
-/
theorem rectangular_field_area (L W : ℝ) : 
  L = 20 →  -- The uncovered side is 20 feet
  2 * W + L = 92 →  -- Total fencing equation
  L * W = 720  -- Area of the field
:= by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1931_193139


namespace NUMINAMATH_CALUDE_largest_increase_2011_2012_l1931_193110

/-- Represents the number of students participating in AMC 12 for each year from 2010 to 2016 --/
def amc_participants : Fin 7 → ℕ
  | 0 => 120  -- 2010
  | 1 => 130  -- 2011
  | 2 => 150  -- 2012
  | 3 => 155  -- 2013
  | 4 => 160  -- 2014
  | 5 => 140  -- 2015
  | 6 => 150  -- 2016

/-- Calculates the percentage increase between two consecutive years --/
def percentage_increase (year : Fin 6) : ℚ :=
  (amc_participants (year.succ) - amc_participants year : ℚ) / amc_participants year * 100

/-- Theorem stating that the percentage increase between 2011 and 2012 is the largest --/
theorem largest_increase_2011_2012 :
  ∀ year : Fin 6, percentage_increase 1 ≥ percentage_increase year :=
by sorry

#eval percentage_increase 1  -- Should output the largest percentage increase

end NUMINAMATH_CALUDE_largest_increase_2011_2012_l1931_193110


namespace NUMINAMATH_CALUDE_net_pay_calculation_l1931_193149

/-- Calculate net pay given gross pay and tax paid -/
def netPay (grossPay : ℕ) (taxPaid : ℕ) : ℕ :=
  grossPay - taxPaid

theorem net_pay_calculation (grossPay : ℕ) (taxPaid : ℕ) 
  (h1 : grossPay = 450)
  (h2 : taxPaid = 135) :
  netPay grossPay taxPaid = 315 := by
  sorry

end NUMINAMATH_CALUDE_net_pay_calculation_l1931_193149


namespace NUMINAMATH_CALUDE_min_distance_b_to_c_l1931_193151

/-- Calculates the minimum distance between points B and C given boat and river conditions -/
theorem min_distance_b_to_c 
  (boat_speed : ℝ) 
  (downstream_current : ℝ) 
  (upstream_current : ℝ) 
  (time_a_to_b : ℝ) 
  (max_time_b_to_c : ℝ) 
  (h1 : boat_speed = 42) 
  (h2 : downstream_current = 5) 
  (h3 : upstream_current = 7) 
  (h4 : time_a_to_b = 1 + 10/60) 
  (h5 : max_time_b_to_c = 2.5) : 
  ∃ (min_distance : ℝ), min_distance = 87.5 := by
  sorry

#check min_distance_b_to_c

end NUMINAMATH_CALUDE_min_distance_b_to_c_l1931_193151


namespace NUMINAMATH_CALUDE_point_move_result_l1931_193170

def point_move (initial_position : ℤ) (move_distance : ℤ) : Set ℤ :=
  {initial_position - move_distance, initial_position + move_distance}

theorem point_move_result :
  point_move (-5) 3 = {-8, -2} := by sorry

end NUMINAMATH_CALUDE_point_move_result_l1931_193170


namespace NUMINAMATH_CALUDE_x_minus_y_equals_ten_l1931_193183

theorem x_minus_y_equals_ten (x y : ℝ) 
  (h1 : 2 = 0.10 * x) 
  (h2 : 2 = 0.20 * y) : 
  x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_ten_l1931_193183


namespace NUMINAMATH_CALUDE_histogram_frequency_l1931_193106

theorem histogram_frequency (sample_size : ℕ) (num_groups : ℕ) (class_interval : ℕ) (rectangle_height : ℝ) : 
  sample_size = 100 →
  num_groups = 10 →
  class_interval = 10 →
  rectangle_height = 0.03 →
  (rectangle_height * class_interval * sample_size : ℝ) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_histogram_frequency_l1931_193106


namespace NUMINAMATH_CALUDE_expression_evaluation_l1931_193178

theorem expression_evaluation : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1931_193178


namespace NUMINAMATH_CALUDE_triangle_inequality_l1931_193164

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1931_193164


namespace NUMINAMATH_CALUDE_element_in_set_l1931_193191

open Set

universe u

def U : Set ℕ := {1, 3, 5, 7, 9}

theorem element_in_set (M : Set ℕ) (h : (U \ M) = {1, 3, 5}) : 7 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l1931_193191
