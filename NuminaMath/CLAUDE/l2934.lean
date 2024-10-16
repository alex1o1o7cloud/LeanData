import Mathlib

namespace NUMINAMATH_CALUDE_percent_difference_z_w_l2934_293475

theorem percent_difference_z_w (y x w z : ℝ) 
  (hw : w = 0.6 * x)
  (hx : x = 0.6 * y)
  (hz : z = 0.54 * y) :
  (z - w) / w * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_difference_z_w_l2934_293475


namespace NUMINAMATH_CALUDE_gumball_machine_problem_l2934_293435

theorem gumball_machine_problem (red blue green : ℕ) : 
  blue = red / 2 →
  green = 4 * blue →
  red + blue + green = 56 →
  red = 16 := by
  sorry

end NUMINAMATH_CALUDE_gumball_machine_problem_l2934_293435


namespace NUMINAMATH_CALUDE_photographs_eighteen_hours_ago_l2934_293479

theorem photographs_eighteen_hours_ago (photos_18h_ago : ℕ) : 
  (photos_18h_ago : ℚ) + 0.8 * (photos_18h_ago : ℚ) = 180 →
  photos_18h_ago = 100 := by
sorry

end NUMINAMATH_CALUDE_photographs_eighteen_hours_ago_l2934_293479


namespace NUMINAMATH_CALUDE_expand_product_l2934_293457

theorem expand_product (x : ℝ) : (x + 6) * (x + 8) * (x - 3) = x^3 + 11*x^2 + 6*x - 144 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2934_293457


namespace NUMINAMATH_CALUDE_min_sequence_length_l2934_293473

def S : Finset Nat := {1, 2, 3, 4}

def is_valid_sequence (a : List Nat) : Prop :=
  ∀ b : List Nat, b.length = 4 ∧ b.toFinset = S ∧ b.getLast? ≠ some 1 →
    ∃ i₁ i₂ i₃ i₄, i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ ≤ a.length ∧
      (a.get? i₁, a.get? i₂, a.get? i₃, a.get? i₄) = (b.get? 0, b.get? 1, b.get? 2, b.get? 3)

theorem min_sequence_length :
  ∃ a : List Nat, a.length = 11 ∧ is_valid_sequence a ∧
    ∀ a' : List Nat, is_valid_sequence a' → a'.length ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_min_sequence_length_l2934_293473


namespace NUMINAMATH_CALUDE_larger_interior_angle_measure_l2934_293489

/-- A circular pavilion constructed with congruent isosceles trapezoids -/
structure CircularPavilion where
  /-- The number of trapezoids in the pavilion -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of a typical trapezoid in degrees -/
  larger_interior_angle : ℝ
  /-- Assertion that the bottom sides of the two end trapezoids are horizontal -/
  horizontal_bottom_sides : Prop

/-- Theorem stating the measure of the larger interior angle in a circular pavilion with 12 trapezoids -/
theorem larger_interior_angle_measure (p : CircularPavilion) 
  (h1 : p.num_trapezoids = 12)
  (h2 : p.horizontal_bottom_sides) :
  p.larger_interior_angle = 97.5 := by
  sorry

end NUMINAMATH_CALUDE_larger_interior_angle_measure_l2934_293489


namespace NUMINAMATH_CALUDE_sum_in_base_b_l2934_293486

/-- Given a base b, converts a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, converts a number from base 10 to base b -/
def fromBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if a given base b satisfies the condition (14)(17)(18) = 5404 in base b -/
def isValidBase (b : ℕ) : Prop :=
  (toBase10 14 b) * (toBase10 17 b) * (toBase10 18 b) = toBase10 5404 b

theorem sum_in_base_b (b : ℕ) (h : isValidBase b) :
  fromBase10 ((toBase10 14 b) + (toBase10 17 b) + (toBase10 18 b)) b = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base_b_l2934_293486


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l2934_293402

/-- The line that is the perpendicular bisector of two points -/
def perpendicular_bisector (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P A = dist P B}

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point satisfies a line equation -/
def satisfies_equation (P : ℝ × ℝ) (L : LineEquation) : Prop :=
  L.a * P.1 + L.b * P.2 + L.c = 0

theorem perpendicular_bisector_equation 
  (A B : ℝ × ℝ) 
  (hA : A = (7, -4)) 
  (hB : B = (-5, 6)) :
  ∃ L : LineEquation, 
    L.a = 6 ∧ L.b = -5 ∧ L.c = -1 ∧
    ∀ P, P ∈ perpendicular_bisector A B ↔ satisfies_equation P L :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l2934_293402


namespace NUMINAMATH_CALUDE_total_stars_is_10_pow_22_l2934_293445

/-- The number of galaxies in the universe -/
def num_galaxies : ℕ := 10^11

/-- The number of stars in each galaxy -/
def stars_per_galaxy : ℕ := 10^11

/-- The total number of stars in the universe -/
def total_stars : ℕ := num_galaxies * stars_per_galaxy

/-- Theorem stating that the total number of stars is 10^22 -/
theorem total_stars_is_10_pow_22 : total_stars = 10^22 := by
  sorry

end NUMINAMATH_CALUDE_total_stars_is_10_pow_22_l2934_293445


namespace NUMINAMATH_CALUDE_smallest_binary_multiple_of_60_l2934_293420

def is_binary (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_60 :
  ∃ (X : ℕ), X > 0 ∧ is_binary (60 * X) ∧
  (∀ (Y : ℕ), Y > 0 → is_binary (60 * Y) → X ≤ Y) ∧
  X = 185 := by
sorry

end NUMINAMATH_CALUDE_smallest_binary_multiple_of_60_l2934_293420


namespace NUMINAMATH_CALUDE_billys_age_l2934_293474

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe)
  (h2 : billy + joe = 60)
  (h3 : billy > 30) :
  billy = 45 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l2934_293474


namespace NUMINAMATH_CALUDE_distance_when_in_step_l2934_293469

/-- The stride length of Jack in centimeters. -/
def jackStride : ℕ := 64

/-- The stride length of Jill in centimeters. -/
def jillStride : ℕ := 56

/-- The theorem states that the distance walked when Jack and Jill are next in step
    is equal to the least common multiple of their stride lengths. -/
theorem distance_when_in_step :
  Nat.lcm jackStride jillStride = 448 := by sorry

end NUMINAMATH_CALUDE_distance_when_in_step_l2934_293469


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2934_293417

theorem inequality_equivalence (x : ℝ) : 3 * x + 4 < 5 * x - 6 ↔ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2934_293417


namespace NUMINAMATH_CALUDE_OPRQ_shape_l2934_293481

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Represents the figure OPRQ -/
structure OPRQ :=
  (O : Point)
  (P : Point)
  (Q : Point)
  (R : Point)
  (h_distinct : P ≠ Q)
  (h_R : R.x = P.x + Q.x ∧ R.y = P.y + Q.y)
  (h_O : O.x = 0 ∧ O.y = 0)

/-- The figure OPRQ is a parallelogram -/
def is_parallelogram (f : OPRQ) : Prop :=
  f.O.x + f.R.x = f.P.x + f.Q.x ∧ f.O.y + f.R.y = f.P.y + f.Q.y

/-- The figure OPRQ is a straight line -/
def is_straight_line (f : OPRQ) : Prop :=
  collinear f.O f.P f.Q ∧ collinear f.O f.P f.R

theorem OPRQ_shape (f : OPRQ) :
  is_parallelogram f ∨ is_straight_line f :=
sorry

end NUMINAMATH_CALUDE_OPRQ_shape_l2934_293481


namespace NUMINAMATH_CALUDE_extreme_values_of_f_max_min_on_interval_parallel_tangents_midpoint_l2934_293476

/-- The function f(x) = x^3 - 12x + 12 --/
def f (x : ℝ) : ℝ := x^3 - 12*x + 12

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 12

theorem extreme_values_of_f :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 2 ∧ 
   f x₁ = 28 ∧ f x₂ = -4 ∧
   ∀ x : ℝ, f x ≤ f x₁ ∧ f x₂ ≤ f x) :=
sorry

theorem max_min_on_interval :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 4 → f x ≤ 28) ∧
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 4 → -4 ≤ f x) ∧
  (∃ x₁ x₂ : ℝ, -3 ≤ x₁ ∧ x₁ ≤ 4 ∧ -3 ≤ x₂ ∧ x₂ ≤ 4 ∧ f x₁ = 28 ∧ f x₂ = -4) :=
sorry

theorem parallel_tangents_midpoint :
  ∀ a b : ℝ, f' a = f' b →
  (a + b) / 2 = 0 ∧ (f a + f b) / 2 = 12 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_max_min_on_interval_parallel_tangents_midpoint_l2934_293476


namespace NUMINAMATH_CALUDE_construction_material_sum_l2934_293412

theorem construction_material_sum : 
  12.468 + 4.6278 + 7.9101 + 8.3103 + 5.6327 = 38.9499 := by
  sorry

end NUMINAMATH_CALUDE_construction_material_sum_l2934_293412


namespace NUMINAMATH_CALUDE_fencing_calculation_l2934_293436

/-- Represents a rectangular field with given dimensions and fencing requirements -/
structure RectangularField where
  length : ℝ
  width : ℝ
  uncoveredSide : ℝ
  area : ℝ

/-- Calculates the required fencing for a rectangular field -/
def requiredFencing (field : RectangularField) : ℝ :=
  2 * field.width + field.length

/-- Theorem stating the required fencing for the given field specifications -/
theorem fencing_calculation (field : RectangularField) 
  (h1 : field.length = 20)
  (h2 : field.area = 390)
  (h3 : field.area = field.length * field.width)
  (h4 : field.uncoveredSide = field.length) :
  requiredFencing field = 59 := by
  sorry

end NUMINAMATH_CALUDE_fencing_calculation_l2934_293436


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_plus_one_l2934_293466

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest possible sum of digits of n+1 is 2, given that the sum of digits of n is 2017 -/
theorem smallest_sum_of_digits_plus_one (n : ℕ) (h : sum_of_digits n = 2017) :
  ∃ m : ℕ, sum_of_digits (n + 1) = 2 ∧ ∀ k : ℕ, sum_of_digits (k + 1) < 2 → sum_of_digits k ≠ 2017 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_plus_one_l2934_293466


namespace NUMINAMATH_CALUDE_wendy_accounting_career_percentage_l2934_293429

/-- Represents the number of years Wendy spent as an accountant. -/
def years_as_accountant : ℕ := 25

/-- Represents the number of years Wendy spent as an accounting manager. -/
def years_as_manager : ℕ := 15

/-- Represents Wendy's total lifespan in years. -/
def total_lifespan : ℕ := 80

/-- Calculates the percentage of Wendy's life spent in accounting-related jobs. -/
def accounting_career_percentage : ℚ :=
  (years_as_accountant + years_as_manager : ℚ) / total_lifespan * 100

/-- Proves that the percentage of Wendy's life spent in accounting-related jobs is 50%. -/
theorem wendy_accounting_career_percentage :
  accounting_career_percentage = 50 := by sorry

end NUMINAMATH_CALUDE_wendy_accounting_career_percentage_l2934_293429


namespace NUMINAMATH_CALUDE_power_of_power_three_l2934_293414

theorem power_of_power_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l2934_293414


namespace NUMINAMATH_CALUDE_stuffed_animal_cost_l2934_293453

theorem stuffed_animal_cost (coloring_books_cost peanuts_cost total_spent : ℚ) : 
  coloring_books_cost = 8 →
  peanuts_cost = 6 →
  total_spent = 25 →
  total_spent - (coloring_books_cost + peanuts_cost) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_stuffed_animal_cost_l2934_293453


namespace NUMINAMATH_CALUDE_rectangle_x_value_l2934_293454

/-- A rectangle with specified side lengths -/
structure Rectangle where
  top_left : ℝ
  top_middle : ℝ
  top_right : ℝ
  bottom_left : ℝ
  bottom_middle : ℝ
  bottom_right : ℝ

/-- The theorem stating that X must be 7 in the given rectangle -/
theorem rectangle_x_value (r : Rectangle) 
    (h1 : r.top_left = 1)
    (h2 : r.top_middle = 2)
    (h3 : r.top_right = 3)
    (h4 : r.bottom_left = 4)
    (h5 : r.bottom_middle = 2)
    (h6 : r.bottom_right = 7)
    (h_rect : r.top_left + r.top_middle + X + r.top_right = 
              r.bottom_left + r.bottom_middle + r.bottom_right) : 
  X = 7 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_x_value_l2934_293454


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2934_293498

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_valid : 1 ≤ tens ∧ tens ≤ 9
  units_valid : units ≤ 9

/-- The value of a two-digit number -/
def value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- The reverse of a two-digit number -/
def reverse (n : TwoDigitNumber) : Nat :=
  10 * n.units + n.tens

/-- The sum of digits of a two-digit number -/
def digitSum (n : TwoDigitNumber) : Nat :=
  n.tens + n.units

theorem two_digit_number_property (n : TwoDigitNumber) :
  value n - reverse n = 7 * digitSum n →
  value n + reverse n = 99 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2934_293498


namespace NUMINAMATH_CALUDE_lcm_36_84_l2934_293465

theorem lcm_36_84 : Nat.lcm 36 84 = 252 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_84_l2934_293465


namespace NUMINAMATH_CALUDE_simplify_expression_l2934_293461

theorem simplify_expression (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2934_293461


namespace NUMINAMATH_CALUDE_restaurant_cooks_count_l2934_293459

/-- Proves that the number of cooks is 9 given the initial and final ratios of cooks to waiters -/
theorem restaurant_cooks_count (cooks waiters : ℕ) : 
  (cooks : ℚ) / waiters = 3 / 10 →
  cooks / (waiters + 12) = 3 / 14 →
  cooks = 9 := by
sorry

end NUMINAMATH_CALUDE_restaurant_cooks_count_l2934_293459


namespace NUMINAMATH_CALUDE_quadratic_increasing_after_vertex_l2934_293447

def f (x : ℝ) : ℝ := (x - 1)^2 + 5

theorem quadratic_increasing_after_vertex (x1 x2 : ℝ) (h1 : x1 > 1) (h2 : x2 > x1) : 
  f x2 > f x1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_after_vertex_l2934_293447


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l2934_293455

theorem angle_sum_theorem (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 1/3) (h4 : Real.cos β = 3/5) :
  α + 2*β = π - Real.arctan (13/9) := by sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l2934_293455


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2934_293446

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, Real.log (x^2 + 1) > 0) ↔ (∃ x : ℝ, Real.log (x^2 + 1) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2934_293446


namespace NUMINAMATH_CALUDE_equation_solution_l2934_293401

theorem equation_solution : 
  ∃ y : ℚ, (40 / 70)^2 = Real.sqrt (y / 70) → y = 17920 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2934_293401


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2934_293424

theorem geometric_arithmetic_sequence_problem 
  (a b : ℕ → ℝ)
  (h_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n)
  (h_arithmetic : ∀ n : ℕ, b (n + 2) - b (n + 1) = b (n + 1) - b n)
  (h_a_product : a 1 * a 5 * a 9 = -8)
  (h_b_sum : b 2 + b 5 + b 8 = 6 * Real.pi)
  : Real.cos ((b 4 + b 6) / (1 - a 3 * a 7)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2934_293424


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l2934_293464

def is_root (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem correct_quadratic_equation (a b c : ℝ) :
  (∃ b₁ c₁, is_root 1 b₁ c₁ 7 ∧ is_root 1 b₁ c₁ 3) →
  (∃ b₂ c₂, is_root 1 b₂ c₂ 11 ∧ is_root 1 b₂ c₂ (-1)) →
  (a = 1 ∧ b = -10 ∧ c = 32) :=
sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l2934_293464


namespace NUMINAMATH_CALUDE_slope_of_line_l2934_293495

/-- The slope of a line given by the equation y/4 - x/5 = 2 is 4/5 -/
theorem slope_of_line (x y : ℝ) :
  y / 4 - x / 5 = 2 → (∃ b : ℝ, y = 4 / 5 * x + b) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l2934_293495


namespace NUMINAMATH_CALUDE_psychology_lecture_first_probability_l2934_293458

-- Define the type for lectures
inductive Lecture
| Morality
| Psychology
| Safety

-- Define a function to calculate the number of permutations
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the theorem
theorem psychology_lecture_first_probability :
  let total_arrangements := factorial 3
  let favorable_arrangements := factorial 2
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 3 := by
sorry


end NUMINAMATH_CALUDE_psychology_lecture_first_probability_l2934_293458


namespace NUMINAMATH_CALUDE_actual_distance_calculation_l2934_293415

/-- Calculates the actual distance between two towns given map distance, scale, and conversion factor. -/
theorem actual_distance_calculation (map_distance : ℝ) (scale : ℝ) (mile_to_km : ℝ) : 
  map_distance = 20 →
  scale = 5 →
  mile_to_km = 1.60934 →
  map_distance * scale * mile_to_km = 160.934 := by
sorry

end NUMINAMATH_CALUDE_actual_distance_calculation_l2934_293415


namespace NUMINAMATH_CALUDE_power_of_power_l2934_293428

theorem power_of_power (a : ℝ) : (a^5)^2 = a^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2934_293428


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt35_l2934_293442

theorem rationalize_denominator_sqrt35 : 
  (35 : ℝ) / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt35_l2934_293442


namespace NUMINAMATH_CALUDE_simplify_expression_l2934_293410

theorem simplify_expression : ((-Real.sqrt 3)^2)^(-1/2 : ℝ) = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2934_293410


namespace NUMINAMATH_CALUDE_max_fourth_power_sum_l2934_293433

theorem max_fourth_power_sum (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 8) :
  ∃ (m : ℝ), m = 16 ∧ a^4 + b^4 + c^4 + d^4 ≤ m ∧
  ∃ (a' b' c' d' : ℝ), a'^3 + b'^3 + c'^3 + d'^3 = 8 ∧ a'^4 + b'^4 + c'^4 + d'^4 = m :=
by sorry

end NUMINAMATH_CALUDE_max_fourth_power_sum_l2934_293433


namespace NUMINAMATH_CALUDE_remaining_three_digit_numbers_l2934_293439

/-- The count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two identical adjacent digits and a different third digit -/
def excluded_numbers : ℕ := 162

/-- The remaining count of three-digit numbers after exclusion -/
def remaining_numbers : ℕ := total_three_digit_numbers - excluded_numbers

theorem remaining_three_digit_numbers : remaining_numbers = 738 := by
  sorry

end NUMINAMATH_CALUDE_remaining_three_digit_numbers_l2934_293439


namespace NUMINAMATH_CALUDE_improve_shooting_average_l2934_293419

/-- Represents a basketball player's shooting statistics -/
structure ShootingStats :=
  (initial_shots : ℕ)
  (initial_made : ℕ)
  (additional_shots : ℕ)
  (additional_made : ℕ)

/-- Calculates the shooting average as a rational number -/
def shooting_average (stats : ShootingStats) : ℚ :=
  (stats.initial_made + stats.additional_made : ℚ) / (stats.initial_shots + stats.additional_shots)

theorem improve_shooting_average 
  (stats : ShootingStats) 
  (h1 : stats.initial_shots = 40)
  (h2 : stats.initial_made = 18)
  (h3 : stats.additional_shots = 15)
  (h4 : shooting_average {initial_shots := stats.initial_shots, 
                          initial_made := stats.initial_made, 
                          additional_shots := 0, 
                          additional_made := 0} = 45/100)
  : shooting_average {initial_shots := stats.initial_shots,
                      initial_made := stats.initial_made,
                      additional_shots := stats.additional_shots,
                      additional_made := 12} = 55/100 := by
  sorry

end NUMINAMATH_CALUDE_improve_shooting_average_l2934_293419


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2934_293487

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5678 * 10 + N) % 6 = 0 → N ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2934_293487


namespace NUMINAMATH_CALUDE_limit_at_infinity_limit_at_point_l2934_293438

-- Part 1
theorem limit_at_infinity (ε : ℝ) (hε : ε > 0) :
  ∃ M : ℝ, ∀ x : ℝ, x > M → |(2*x + 3)/(3*x) - 2/3| < ε :=
sorry

-- Part 2
theorem limit_at_point (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x - 3| ∧ |x - 3| < δ → |(2*x + 1) - 7| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_at_infinity_limit_at_point_l2934_293438


namespace NUMINAMATH_CALUDE_two_valid_inequalities_l2934_293480

theorem two_valid_inequalities : 
  (∃ (f₁ f₂ f₃ : Prop), 
    (f₁ ↔ ∀ x : ℝ, Real.sqrt 5 + Real.sqrt 9 > 2 * Real.sqrt 7) ∧ 
    (f₂ ↔ ∀ a b c : ℝ, a^2 + 2*b^2 + 3*c^2 ≥ (1/6) * (a + 2*b + 3*c)^2) ∧ 
    (f₃ ↔ ∀ x : ℝ, Real.exp x ≥ x + 1) ∧ 
    (f₁ ∨ f₂ ∨ f₃) ∧ 
    (f₁ ∧ f₂ ∨ f₁ ∧ f₃ ∨ f₂ ∧ f₃) ∧ 
    ¬(f₁ ∧ f₂ ∧ f₃)) :=
by sorry

end NUMINAMATH_CALUDE_two_valid_inequalities_l2934_293480


namespace NUMINAMATH_CALUDE_f_properties_l2934_293493

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (2 * ω * x - Real.pi / 6) + 2 * (Real.cos (ω * x))^2 - 1

def is_interval_of_increase (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_range (f : ℝ → ℝ) (S : Set ℝ) (R : Set ℝ) : Prop :=
  ∀ y ∈ R, ∃ x ∈ S, f x = y

theorem f_properties (ω : ℝ) (h : ω > 0) :
  (∀ k : ℤ, is_interval_of_increase (f 1) (-Real.pi/3 + k*Real.pi) (Real.pi/6 + k*Real.pi)) ∧
  (ω = 8/3 → is_range (f ω) (Set.Icc 0 (Real.pi/8)) (Set.Icc (1/2) 1)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2934_293493


namespace NUMINAMATH_CALUDE_projection_matrix_values_l2934_293413

def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  Q * Q = Q

theorem projection_matrix_values :
  ∀ (b d : ℝ),
  let Q : Matrix (Fin 2) (Fin 2) ℝ := !![b, 1/5; d, 4/5]
  is_projection_matrix Q ↔ b = 1 ∧ d = 1 := by
sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l2934_293413


namespace NUMINAMATH_CALUDE_sample_capacity_proof_l2934_293472

theorem sample_capacity_proof (n : ℕ) (frequency : ℕ) (relative_frequency : ℚ) 
  (h1 : frequency = 30)
  (h2 : relative_frequency = 1/4)
  (h3 : relative_frequency = frequency / n) :
  n = 120 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_proof_l2934_293472


namespace NUMINAMATH_CALUDE_m_mobile_additional_line_cost_l2934_293441

/-- Represents a mobile phone plan with a base cost and additional line cost -/
structure MobilePlan where
  baseCost : ℕ  -- Cost for first two lines
  addLineCost : ℕ  -- Cost for each additional line

/-- Calculates the total cost for a given number of lines -/
def totalCost (plan : MobilePlan) (lines : ℕ) : ℕ :=
  plan.baseCost + plan.addLineCost * (lines - 2)

theorem m_mobile_additional_line_cost :
  ∃ (mMobileAddCost : ℕ),
    let tMobile : MobilePlan := ⟨50, 16⟩
    let mMobile : MobilePlan := ⟨45, mMobileAddCost⟩
    totalCost tMobile 5 - totalCost mMobile 5 = 11 →
    mMobileAddCost = 14 := by
  sorry


end NUMINAMATH_CALUDE_m_mobile_additional_line_cost_l2934_293441


namespace NUMINAMATH_CALUDE_pythagorean_triple_7_24_25_l2934_293468

theorem pythagorean_triple_7_24_25 : 
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a = 7 ∧ b = 24 ∧ c = 25 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_7_24_25_l2934_293468


namespace NUMINAMATH_CALUDE_complex_product_PRS_l2934_293411

theorem complex_product_PRS : 
  let P : ℂ := 3 + 4 * Complex.I
  let R : ℂ := 2 * Complex.I
  let S : ℂ := 3 - 4 * Complex.I
  P * R * S = 50 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_PRS_l2934_293411


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2934_293451

theorem cubic_equation_root (k : ℚ) : 
  (∃ x : ℚ, 10 * k * x^3 - x - 9 = 0 ∧ x = -1) → k = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2934_293451


namespace NUMINAMATH_CALUDE_valid_numbers_characterization_l2934_293452

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 2 = 0 ∧  -- even
  (n / 10 + n % 10) > 6 ∧  -- sum of digits greater than 6
  (n / 10) ≥ (n % 10 + 4)  -- tens digit at least 4 greater than units digit

theorem valid_numbers_characterization :
  {n : ℕ | is_valid_number n} = {70, 80, 90, 62, 72, 82, 92, 84, 94} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_characterization_l2934_293452


namespace NUMINAMATH_CALUDE_adam_marbles_l2934_293400

theorem adam_marbles (greg_marbles : ℕ) (greg_more_than_adam : ℕ) 
  (h1 : greg_marbles = 43)
  (h2 : greg_more_than_adam = 14) :
  greg_marbles - greg_more_than_adam = 29 := by
  sorry

end NUMINAMATH_CALUDE_adam_marbles_l2934_293400


namespace NUMINAMATH_CALUDE_parabola_translation_l2934_293422

-- Define the original parabola
def original_parabola (x y : ℝ) : Prop := y = x^2 + 3

-- Define the translated parabola
def translated_parabola (x y : ℝ) : Prop := y = (x + 1)^2 + 3

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, original_parabola (x + 1) y ↔ translated_parabola x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2934_293422


namespace NUMINAMATH_CALUDE_image_of_3_4_preimages_of_1_neg6_l2934_293405

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

-- Theorem for the image of (3, 4)
theorem image_of_3_4 : f (3, 4) = (7, 12) := by sorry

-- Theorem for the pre-images of (1, -6)
theorem preimages_of_1_neg6 : 
  {p : ℝ × ℝ | f p = (1, -6)} = {(-2, 3), (3, -2)} := by sorry

end NUMINAMATH_CALUDE_image_of_3_4_preimages_of_1_neg6_l2934_293405


namespace NUMINAMATH_CALUDE_wrong_multiplication_correction_l2934_293462

theorem wrong_multiplication_correction (x : ℝ) (h : x * 2.4 = 288) : (x / 2.4) / 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_wrong_multiplication_correction_l2934_293462


namespace NUMINAMATH_CALUDE_set_D_is_empty_l2934_293491

def set_D : Set ℝ := {x : ℝ | x^2 - x + 1 = 0}

theorem set_D_is_empty : set_D = ∅ := by
  sorry

end NUMINAMATH_CALUDE_set_D_is_empty_l2934_293491


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2934_293499

theorem complex_fraction_evaluation : 
  (1 : ℚ) / (1 - 1 / (3 + 1 / 4)) = 13 / 9 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2934_293499


namespace NUMINAMATH_CALUDE_f_2x_equals_x_plus_1_over_x_minus_1_l2934_293409

theorem f_2x_equals_x_plus_1_over_x_minus_1 
  (x : ℝ) 
  (h : x^2 ≠ 4) : 
  let f := fun (y : ℝ) => (y + 2) / (y - 2)
  f (2 * x) = (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_2x_equals_x_plus_1_over_x_minus_1_l2934_293409


namespace NUMINAMATH_CALUDE_cubic_inequality_l2934_293407

theorem cubic_inequality (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2934_293407


namespace NUMINAMATH_CALUDE_employment_percentage_l2934_293456

theorem employment_percentage (total_population : ℝ) (employed_population : ℝ) :
  (employed_population / total_population = 0.5 / 0.78125) ↔
  (0.5 * total_population = employed_population * (1 - 0.21875)) :=
by sorry

end NUMINAMATH_CALUDE_employment_percentage_l2934_293456


namespace NUMINAMATH_CALUDE_tan_addition_special_case_l2934_293427

theorem tan_addition_special_case (x : Real) (h : Real.tan x = 1/2) :
  Real.tan (x + π/3) = 7 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_addition_special_case_l2934_293427


namespace NUMINAMATH_CALUDE_min_questions_100_boxes_l2934_293460

/-- Represents the setup of the box guessing game -/
structure BoxGame where
  num_boxes : ℕ
  num_questions : ℕ

/-- Checks if the number of questions is sufficient to determine the prize box -/
def is_sufficient (game : BoxGame) : Prop :=
  game.num_questions + 1 ≥ game.num_boxes

/-- The minimum number of questions needed for a given number of boxes -/
def min_questions (n : ℕ) : ℕ :=
  n - 1

/-- Theorem stating the minimum number of questions needed for 100 boxes -/
theorem min_questions_100_boxes :
  ∃ (game : BoxGame), game.num_boxes = 100 ∧ game.num_questions = 99 ∧ 
  is_sufficient game ∧ 
  ∀ (g : BoxGame), g.num_boxes = 100 → g.num_questions < 99 → ¬is_sufficient g :=
by sorry


end NUMINAMATH_CALUDE_min_questions_100_boxes_l2934_293460


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l2934_293423

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem stating that 499 is the smallest prime whose digits sum to 23 -/
theorem smallest_prime_digit_sum_23 : 
  (is_prime 499 ∧ digit_sum 499 = 23) ∧ 
  ∀ n : ℕ, n < 499 → ¬(is_prime n ∧ digit_sum n = 23) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l2934_293423


namespace NUMINAMATH_CALUDE_find_value_of_b_l2934_293406

/-- Given a configuration of numbers in circles with specific properties, prove the value of b. -/
theorem find_value_of_b (circle_sum : ℕ) (total_circles : ℕ) (total_sum : ℕ) 
  (overlap_sum : ℕ → ℕ → ℕ) (d_circle_sum : ℕ → ℕ) 
  (h1 : circle_sum = 21)
  (h2 : total_circles = 5)
  (h3 : total_sum = 69)
  (h4 : ∀ (b d : ℕ), overlap_sum b d = 2 + 8 + 9 + b + d)
  (h5 : ∀ (d : ℕ), d_circle_sum d = d + 5 + 9)
  (h6 : ∀ (d : ℕ), d_circle_sum d = circle_sum) :
  ∃ (b : ℕ), b = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_value_of_b_l2934_293406


namespace NUMINAMATH_CALUDE_modulus_of_3_minus_4i_l2934_293470

theorem modulus_of_3_minus_4i : Complex.abs (3 - 4*I) = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_3_minus_4i_l2934_293470


namespace NUMINAMATH_CALUDE_integer_sqrt_divisibility_l2934_293477

theorem integer_sqrt_divisibility (n : ℕ) (h1 : n ≥ 4) :
  (Int.floor (Real.sqrt n) + 1 ∣ n - 1) ∧
  (Int.floor (Real.sqrt n) - 1 ∣ n + 1) →
  n = 4 ∨ n = 7 ∨ n = 9 ∨ n = 13 ∨ n = 31 := by
  sorry

end NUMINAMATH_CALUDE_integer_sqrt_divisibility_l2934_293477


namespace NUMINAMATH_CALUDE_parallel_lines_interior_alternate_angles_l2934_293437

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- A line intersects two other lines -/
def intersects (l : Line) (l1 l2 : Line) : Prop := sorry

/-- Interior alternate angles between two lines and a transversal -/
def interior_alternate_angles (l1 l2 l : Line) (α β : Angle) : Prop := sorry

/-- The proposition about parallel lines and interior alternate angles -/
theorem parallel_lines_interior_alternate_angles 
  (l1 l2 l : Line) (α β : Angle) :
  parallel l1 l2 → 
  intersects l l1 l2 → 
  interior_alternate_angles l1 l2 l α β → 
  α = β := 
sorry

end NUMINAMATH_CALUDE_parallel_lines_interior_alternate_angles_l2934_293437


namespace NUMINAMATH_CALUDE_right_triangle_inequalities_l2934_293418

-- Define a structure for a right-angled triangle with height to hypotenuse
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  right_angle : a^2 + b^2 = c^2
  height_def : 2 * h * c = a * b

theorem right_triangle_inequalities (t : RightTriangle) :
  (t.a^2 + t.b^2 < t.c^2 + t.h^2) ∧ (t.a^4 + t.b^4 < t.c^4 + t.h^4) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inequalities_l2934_293418


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2934_293444

theorem unique_four_digit_number :
  ∃! N : ℕ,
    N ≡ N^2 [ZMOD 10000] ∧
    N ≡ 7 [ZMOD 16] ∧
    1000 ≤ N ∧ N < 10000 ∧
    N = 3751 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l2934_293444


namespace NUMINAMATH_CALUDE_range_of_a_l2934_293494

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, 0 < a * x^2 - x + 1/(16*a)

def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → Real.sqrt (2*x + 1) < 1 + a*x

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2934_293494


namespace NUMINAMATH_CALUDE_mardi_gras_necklaces_mardi_gras_necklaces_proof_l2934_293404

theorem mardi_gras_necklaces : Int → Int → Int → Prop :=
  fun boudreaux rhonda latch =>
    boudreaux = 12 ∧
    rhonda = boudreaux / 2 ∧
    latch = 3 * rhonda - 4 →
    latch = 14

-- The proof is omitted
theorem mardi_gras_necklaces_proof : mardi_gras_necklaces 12 6 14 := by
  sorry

end NUMINAMATH_CALUDE_mardi_gras_necklaces_mardi_gras_necklaces_proof_l2934_293404


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l2934_293449

theorem midpoint_sum_equals_vertex_sum (a b : ℝ) 
  (h : a + b + (a + 5) = 15) : 
  (a + b) / 2 + (2 * a + 5) / 2 + (b + a + 5) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l2934_293449


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l2934_293450

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 3*x - 4 > 0}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | x ≤ 3 ∨ x > 4} := by sorry

-- Theorem for A ∩ (U \ B)
theorem intersection_A_complement_B : A ∩ (U \ B) = {x | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l2934_293450


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l2934_293434

theorem smallest_factorization_coefficient (b : ℕ+) : 
  (∃ (r s : ℤ), (∀ x : ℝ, x^2 + b.val*x + 3258 = (x + r) * (x + s))) →
  b.val ≥ 1089 :=
sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l2934_293434


namespace NUMINAMATH_CALUDE_kiarra_age_l2934_293482

/-- Given the ages of several people and their relationships, prove Kiarra's age --/
theorem kiarra_age (bea job figaro harry kiarra : ℕ) 
  (h1 : kiarra = 2 * bea)
  (h2 : job = 3 * bea)
  (h3 : figaro = job + 7)
  (h4 : harry * 2 = figaro)
  (h5 : harry = 26) :
  kiarra = 30 := by
  sorry

end NUMINAMATH_CALUDE_kiarra_age_l2934_293482


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2934_293463

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-3) 2 = {x : ℝ | a * x^2 - 5*x + b > 0}) : 
  {x : ℝ | b * x^2 - 5*x + a > 0} = Set.Iic (-1/3) ∪ Set.Ici (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2934_293463


namespace NUMINAMATH_CALUDE_seventy_fifth_term_is_298_l2934_293403

def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem seventy_fifth_term_is_298 : arithmetic_sequence 2 4 75 = 298 := by
  sorry

end NUMINAMATH_CALUDE_seventy_fifth_term_is_298_l2934_293403


namespace NUMINAMATH_CALUDE_locus_and_max_dot_product_l2934_293425

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 8

-- Define point N
def N : ℝ × ℝ := (0, -1)

-- Define the locus C
def locus_C (x y : ℝ) : Prop := y^2 / 2 + x^2 = 1

-- Define the dot product OA · AN
def dot_product (x y : ℝ) : ℝ := -x^2 - y^2 - y

-- Theorem statement
theorem locus_and_max_dot_product :
  ∀ (x y : ℝ),
    (∃ (px py : ℝ), circle_M px py ∧
      (x - (px + 0) / 2)^2 + (y - (py + -1) / 2)^2 = ((px - 0)^2 + (py - -1)^2) / 4) →
    locus_C x y ∧
    (∀ (ax ay : ℝ), locus_C ax ay → dot_product ax ay ≤ -1/2) ∧
    (∃ (ax ay : ℝ), locus_C ax ay ∧ dot_product ax ay = -1/2) :=
by sorry


end NUMINAMATH_CALUDE_locus_and_max_dot_product_l2934_293425


namespace NUMINAMATH_CALUDE_davids_mowing_hours_l2934_293485

theorem davids_mowing_hours (rate : ℝ) (days : ℕ) (remaining : ℝ) : 
  rate = 14 → days = 7 → remaining = 49 → 
  ∃ (hours : ℝ), 
    hours * rate * days / 2 / 2 = remaining ∧ 
    hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_davids_mowing_hours_l2934_293485


namespace NUMINAMATH_CALUDE_abs_neg_a_eq_five_implies_a_eq_plus_minus_five_l2934_293490

theorem abs_neg_a_eq_five_implies_a_eq_plus_minus_five (a : ℝ) :
  |(-a)| = 5 → (a = 5 ∨ a = -5) := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_a_eq_five_implies_a_eq_plus_minus_five_l2934_293490


namespace NUMINAMATH_CALUDE_lyssa_incorrect_percentage_is_12_l2934_293408

def exam_items : ℕ := 75
def precious_mistakes : ℕ := 12
def lyssa_additional_correct : ℕ := 3

def lyssa_incorrect_percentage : ℚ :=
  (exam_items - (exam_items - precious_mistakes + lyssa_additional_correct)) / exam_items * 100

theorem lyssa_incorrect_percentage_is_12 :
  lyssa_incorrect_percentage = 12 := by sorry

end NUMINAMATH_CALUDE_lyssa_incorrect_percentage_is_12_l2934_293408


namespace NUMINAMATH_CALUDE_brigade_plowing_rates_l2934_293497

/-- Represents the daily plowing rate and work duration of a brigade --/
structure Brigade where
  daily_rate : ℝ
  days_worked : ℝ

/-- Proves that given the problem conditions, the brigades' daily rates are 24 and 27 hectares --/
theorem brigade_plowing_rates 
  (first_brigade second_brigade : Brigade)
  (h1 : first_brigade.daily_rate * first_brigade.days_worked = 240)
  (h2 : second_brigade.daily_rate * second_brigade.days_worked = 240 * 1.35)
  (h3 : second_brigade.daily_rate = first_brigade.daily_rate + 3)
  (h4 : second_brigade.days_worked = first_brigade.days_worked + 2)
  (h5 : first_brigade.daily_rate > 20)
  (h6 : second_brigade.daily_rate > 20)
  : first_brigade.daily_rate = 24 ∧ second_brigade.daily_rate = 27 := by
  sorry

#check brigade_plowing_rates

end NUMINAMATH_CALUDE_brigade_plowing_rates_l2934_293497


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_l2934_293478

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Digit
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Converts a FourDigitNumber to a natural number -/
def FourDigitNumber.toNat (n : FourDigitNumber) : Nat :=
  1000 * (n.thousands.val + 1) + 100 * (n.hundreds.val + 1) + 10 * (n.tens.val + 1) + (n.ones.val + 1)

/-- Represents the equation AB・CD = EFGH -/
structure Equation where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  E : Digit
  F : Digit
  G : Digit
  H : Digit
  allDistinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧
                B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧
                C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧
                D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧
                E ≠ F ∧ E ≠ G ∧ E ≠ H ∧
                F ≠ G ∧ F ≠ H ∧
                G ≠ H
  equationHolds : (A.val + 1) * 10 + (B.val + 1) * (C.val + 1) * 10 + (D.val + 1) =
                  (E.val + 1) * 1000 + (F.val + 1) * 100 + (G.val + 1) * 10 + (H.val + 1)

theorem smallest_four_digit_number (eq : Equation) :
  ∃ (n : FourDigitNumber), n.toNat = 4396 ∧ 
  (∀ (m : FourDigitNumber), m.thousands = eq.E ∧ m.hundreds = eq.F ∧ m.tens = eq.G ∧ m.ones = eq.H →
    n.toNat ≤ m.toNat) :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_l2934_293478


namespace NUMINAMATH_CALUDE_conditional_without_else_l2934_293426

-- Define the structure of conditional statements
inductive ConditionalStatement
  | ifThenElse (condition : Prop) (thenStmt : Prop) (elseStmt : Prop)
  | ifThen (condition : Prop) (thenStmt : Prop)

-- Define a property that checks if a conditional statement has an ELSE part
def hasElsePart : ConditionalStatement → Prop
  | ConditionalStatement.ifThenElse _ _ _ => true
  | ConditionalStatement.ifThen _ _ => false

-- Theorem stating that there exists a conditional statement without an ELSE part
theorem conditional_without_else : ∃ (stmt : ConditionalStatement), ¬(hasElsePart stmt) := by
  sorry


end NUMINAMATH_CALUDE_conditional_without_else_l2934_293426


namespace NUMINAMATH_CALUDE_sum_factorials_6_mod_20_l2934_293432

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_6_mod_20 :
  sum_factorials 6 % 20 = 13 := by sorry

end NUMINAMATH_CALUDE_sum_factorials_6_mod_20_l2934_293432


namespace NUMINAMATH_CALUDE_paper_towel_package_rolls_l2934_293467

/-- Given a package of paper towels with the following properties:
  * The package price is $9
  * The individual roll price is $1
  * The savings per roll in the package is 25% compared to individual purchase
  Prove that the number of rolls in the package is 12 -/
theorem paper_towel_package_rolls : 
  ∀ (package_price individual_price : ℚ) (savings_percent : ℚ) (num_rolls : ℕ),
  package_price = 9 →
  individual_price = 1 →
  savings_percent = 25 / 100 →
  package_price = num_rolls * (individual_price * (1 - savings_percent)) →
  num_rolls = 12 := by
sorry

end NUMINAMATH_CALUDE_paper_towel_package_rolls_l2934_293467


namespace NUMINAMATH_CALUDE_expected_value_Z_l2934_293484

/-- The probability mass function for the random variable Z --/
def pmf_Z (P : ℝ) (k : ℕ) : ℝ :=
  if k ≥ 2 then P * (1 - P)^(k - 1) + (1 - P) * P^(k - 1) else 0

/-- The expected value of Z --/
noncomputable def E_Z (P : ℝ) : ℝ :=
  ∑' k, k * pmf_Z P k

/-- Theorem stating the expected value of Z --/
theorem expected_value_Z (P : ℝ) (hP : 0 < P ∧ P < 1) :
  E_Z P = 1 / (P * (1 - P)) - 1 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_Z_l2934_293484


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2934_293431

def polynomial (x : ℝ) : ℝ := 4*x^8 - 3*x^7 + 2*x^6 - 8*x^4 + 5*x^3 - 9

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + 671 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2934_293431


namespace NUMINAMATH_CALUDE_quadratic_solution_l2934_293488

theorem quadratic_solution (y : ℝ) : 
  y > 0 ∧ 6 * y^2 + 5 * y - 12 = 0 ↔ y = (-5 + Real.sqrt 313) / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2934_293488


namespace NUMINAMATH_CALUDE_tangent_line_at_2_sum_formula_min_value_nSn_l2934_293492

/-- The original function -/
def g (x : ℝ) : ℝ := x^2 - 2*x - 11

/-- The tangent line to g(x) at x = 2 -/
def f (x : ℝ) : ℝ := 2*x - 15

/-- The sequence a_n -/
def a (n : ℕ) : ℝ := f n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := n^2 - 14*n

theorem tangent_line_at_2 : 
  ∀ x, f x = (2 : ℝ) * (x - 2) + g 2 :=
sorry

theorem sum_formula : 
  ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2 :=
sorry

theorem min_value_nSn : 
  ∃ n : ℕ, ∀ m : ℕ, m ≥ 1 → (n : ℝ) * S n ≤ (m : ℝ) * S m ∧ 
  (n : ℝ) * S n = -405 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_sum_formula_min_value_nSn_l2934_293492


namespace NUMINAMATH_CALUDE_midpoint_x_coordinate_l2934_293448

/-- Given two points P and Q, prove that if their midpoint has x-coordinate 18, then the x-coordinate of P is 6. -/
theorem midpoint_x_coordinate (a : ℝ) : 
  let P : ℝ × ℝ := (a, 2)
  let Q : ℝ × ℝ := (30, -6)
  let midpoint : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  midpoint.1 = 18 → a = 6 := by
  sorry

#check midpoint_x_coordinate

end NUMINAMATH_CALUDE_midpoint_x_coordinate_l2934_293448


namespace NUMINAMATH_CALUDE_angle_equivalence_l2934_293440

theorem angle_equivalence :
  ∃ (α : ℝ) (k : ℤ), -27/4 * π = α + 2*k*π ∧ 0 ≤ α ∧ α < 2*π ∧ α = 5*π/4 ∧ k = -8 :=
by sorry

end NUMINAMATH_CALUDE_angle_equivalence_l2934_293440


namespace NUMINAMATH_CALUDE_sphere_remaining_volume_l2934_293421

/-- The remaining volume of a sphere after drilling a cylindrical hole -/
theorem sphere_remaining_volume (R : ℝ) (h : R > 3) : 
  (4 / 3 * π * R^3) - (6 * π * (R^2 - 9)) - (2 * π * 3^2 * (R - 3 / 3)) = 36 * π :=
sorry

end NUMINAMATH_CALUDE_sphere_remaining_volume_l2934_293421


namespace NUMINAMATH_CALUDE_carpet_area_l2934_293471

theorem carpet_area : 
  ∀ (length width : ℝ) (shoe_length : ℝ),
    shoe_length = 28 →
    length = 15 * shoe_length →
    width = 10 * shoe_length →
    length * width = 117600 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_l2934_293471


namespace NUMINAMATH_CALUDE_value_of_expression_l2934_293416

theorem value_of_expression (x : ℝ) (h : x^2 - 2*x = 3) : 3*x^2 - 6*x - 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2934_293416


namespace NUMINAMATH_CALUDE_pairings_equal_twenty_l2934_293496

/-- The number of items in the first set -/
def set1_size : ℕ := 5

/-- The number of items in the second set -/
def set2_size : ℕ := 4

/-- The total number of possible pairings -/
def total_pairings : ℕ := set1_size * set2_size

/-- Theorem: The total number of possible pairings is 20 -/
theorem pairings_equal_twenty : total_pairings = 20 := by
  sorry

end NUMINAMATH_CALUDE_pairings_equal_twenty_l2934_293496


namespace NUMINAMATH_CALUDE_turban_price_turban_price_proof_l2934_293430

/-- The price of a turban given the following conditions:
  - The total salary for one year is Rs. 90 plus one turban
  - The servant leaves after 9 months
  - The servant receives Rs. 40 and the turban after 9 months
-/
theorem turban_price : ℝ → Prop :=
  fun price =>
    let total_salary : ℝ := 90 + price
    let months_worked : ℝ := 9
    let total_months : ℝ := 12
    let received_amount : ℝ := 40 + price
    (months_worked / total_months) * total_salary = received_amount →
    price = 110

/-- Proof of the turban price theorem -/
theorem turban_price_proof : ∃ price, turban_price price := by
  sorry

end NUMINAMATH_CALUDE_turban_price_turban_price_proof_l2934_293430


namespace NUMINAMATH_CALUDE_line_equations_proof_l2934_293443

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) is on a given line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Checks if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Checks if two lines are perpendicular -/
def Line.isPerpendicularTo (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_equations_proof :
  let l1 : Line := { a := 3, b := -2, c := 1 }
  let l2 : Line := { a := 3, b := -2, c := 5 }
  let l3 : Line := { a := 3, b := -2, c := -5 }
  let l4 : Line := { a := 2, b := 3, c := 1 }
  (l1.containsPoint 1 2 ∧ l1.isParallelTo l2) ∧
  (l3.containsPoint 1 (-1) ∧ l3.isPerpendicularTo l4) := by sorry

end NUMINAMATH_CALUDE_line_equations_proof_l2934_293443


namespace NUMINAMATH_CALUDE_point_not_on_ln_graph_l2934_293483

theorem point_not_on_ln_graph (a b : ℝ) (h : b = Real.log a) :
  ¬(1 + b = Real.log (a + Real.exp 1)) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_ln_graph_l2934_293483
