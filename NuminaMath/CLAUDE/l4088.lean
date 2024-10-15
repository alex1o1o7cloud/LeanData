import Mathlib

namespace NUMINAMATH_CALUDE_product_104_96_l4088_408894

theorem product_104_96 : 104 * 96 = 9984 := by
  sorry

end NUMINAMATH_CALUDE_product_104_96_l4088_408894


namespace NUMINAMATH_CALUDE_quadratic_intersection_theorem_l4088_408889

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + m

-- Define the condition for three intersection points
def has_three_intersections (m : ℝ) : Prop :=
  m ≠ 0 ∧ ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  ((f m x₁ = 0 ∧ x₁ ≠ 0) ∨ (x₁ = 0 ∧ f m 0 = m)) ∧
  ((f m x₂ = 0 ∧ x₂ ≠ 0) ∨ (x₂ = 0 ∧ f m 0 = m)) ∧
  ((f m x₃ = 0 ∧ x₃ ≠ 0) ∨ (x₃ = 0 ∧ f m 0 = m))

-- Define the circle passing through the three intersection points
def circle_through_intersections (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - (m + 1)*y + m = 0

-- The main theorem
theorem quadratic_intersection_theorem (m : ℝ) :
  has_three_intersections m →
  (m < 4 ∧
   circle_through_intersections m 0 1 ∧
   circle_through_intersections m (-4) 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_theorem_l4088_408889


namespace NUMINAMATH_CALUDE_difference_of_squares_153_147_l4088_408809

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_153_147_l4088_408809


namespace NUMINAMATH_CALUDE_range_of_m_l4088_408885

/-- Given two predicates p and q on real numbers, where p states that there exists a real x such that
    mx² + 1 ≤ 0, and q states that for all real x, x² + mx + 1 > 0, if the disjunction of p and q
    is false, then m is greater than or equal to 2. -/
theorem range_of_m (m : ℝ) : 
  let p := ∃ x : ℝ, m * x^2 + 1 ≤ 0
  let q := ∀ x : ℝ, x^2 + m * x + 1 > 0
  ¬(p ∨ q) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l4088_408885


namespace NUMINAMATH_CALUDE_total_discount_calculation_l4088_408882

theorem total_discount_calculation (tshirt_price jeans_price : ℝ)
  (tshirt_discount jeans_discount : ℝ) :
  tshirt_price = 25 →
  jeans_price = 75 →
  tshirt_discount = 0.3 →
  jeans_discount = 0.1 →
  tshirt_price * tshirt_discount + jeans_price * jeans_discount = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_discount_calculation_l4088_408882


namespace NUMINAMATH_CALUDE_problem_statement_l4088_408826

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 6) :
  -- Part 1: Maximum value of x + 2y + z is 6
  (∃ (max : ℝ), max = 6 ∧ x + 2*y + z ≤ max ∧ 
    ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
      x₀^2 + y₀^2 + z₀^2 = 6 ∧ x₀ + 2*y₀ + z₀ = max) ∧
  -- Part 2: If |a+1| - 2a ≥ x + 2y + z for all valid x, y, z, then a ≤ -7/3
  (∀ (a : ℝ), (∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → 
    x'^2 + y'^2 + z'^2 = 6 → |a + 1| - 2*a ≥ x' + 2*y' + z') → a ≤ -7/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l4088_408826


namespace NUMINAMATH_CALUDE_triangle_theorem_l4088_408812

-- Define a triangle with side lengths and angles
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.A = π/3)  -- A = 60° in radians
  (h2 : t.a = Real.sqrt 13)
  (h3 : t.b = 1) :
  t.c = 4 ∧ (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 2 * Real.sqrt 39 / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l4088_408812


namespace NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l4088_408854

theorem definite_integral_exp_plus_2x : 
  ∫ x in (-1)..1, (Real.exp x + 2 * x) = Real.exp 1 - Real.exp (-1) := by sorry

end NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l4088_408854


namespace NUMINAMATH_CALUDE_sampling_consistency_l4088_408853

def systematic_sampling (n : ℕ) (k : ℕ) (i : ℕ) : Prop :=
  ∃ (r : ℕ), i = r * k ∧ r ≤ n / k

theorem sampling_consistency 
  (total : ℕ) (sample_size : ℕ) (selected : ℕ) (h_total : total = 800) (h_sample : sample_size = 50)
  (h_selected : selected = 39) (h_interval : total / sample_size = 16) :
  systematic_sampling total (total / sample_size) selected → 
  systematic_sampling total (total / sample_size) 7 :=
by sorry

end NUMINAMATH_CALUDE_sampling_consistency_l4088_408853


namespace NUMINAMATH_CALUDE_max_distance_M_to_N_l4088_408866

-- Define the circles and point
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*a*y + 2*a^2 - 2 = 0
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 18
def point_N : ℝ × ℝ := (1, 2)

-- Define the theorem
theorem max_distance_M_to_N :
  ∀ a : ℝ,
  (∀ x y : ℝ, ∃ x' y' : ℝ, circle_M a x' y' ∧ circle_O x' y') →
  ∃ a_max : ℝ,
    (∀ a' : ℝ, (∃ x y : ℝ, circle_M a' x y ∧ circle_O x y) →
      Real.sqrt ((a' - point_N.1)^2 + (a' - point_N.2)^2) ≤ Real.sqrt ((a_max - point_N.1)^2 + (a_max - point_N.2)^2)) ∧
    Real.sqrt ((a_max - point_N.1)^2 + (a_max - point_N.2)^2) = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_M_to_N_l4088_408866


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l4088_408833

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x - 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l4088_408833


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4088_408891

theorem absolute_value_equation_solution (x : ℝ) : 
  |24 / x + 4| = 4 → x = -3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4088_408891


namespace NUMINAMATH_CALUDE_pascal_triangle_cube_sum_l4088_408804

/-- Pascal's Triangle interior numbers sum for row n -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- Sum of cubes of interior numbers in the fifth row -/
def fifth_row_cube_sum : ℕ := 468

/-- Sum of cubes of interior numbers in the sixth row -/
def sixth_row_cube_sum : ℕ := 14750

/-- Theorem: If the sum of cubes of interior numbers in the fifth row is 468,
    then the sum of cubes of interior numbers in the sixth row is 14750 -/
theorem pascal_triangle_cube_sum :
  fifth_row_cube_sum = 468 → sixth_row_cube_sum = 14750 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_cube_sum_l4088_408804


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l4088_408830

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x - f y) = (x - y)^2 * f (x + y)

/-- The theorem stating the possible forms of functions satisfying the equation. -/
theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
    (∀ x, f x = 0) ∨ (∀ x, f x = x^2) ∨ (∀ x, f x = -x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l4088_408830


namespace NUMINAMATH_CALUDE_snowdrift_final_depth_l4088_408808

/-- Calculates the final depth of a snowdrift after four days of weather events. -/
def snowdrift_depth (initial_depth : ℝ) (day2_melt_fraction : ℝ) (day3_snow : ℝ) (day4_snow : ℝ) : ℝ :=
  ((initial_depth * (1 - day2_melt_fraction)) + day3_snow) + day4_snow

/-- Theorem stating that given specific weather conditions over four days,
    the final depth of a snowdrift will be 34 inches. -/
theorem snowdrift_final_depth :
  snowdrift_depth 20 0.5 6 18 = 34 := by
  sorry

end NUMINAMATH_CALUDE_snowdrift_final_depth_l4088_408808


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_relation_l4088_408838

theorem isosceles_triangle_angle_relation (A B C C₁ C₂ θ : Real) :
  -- Isosceles triangle condition
  A = B →
  -- Altitude divides angle C into C₁ and C₂
  A + C₁ = 90 →
  B + C₂ = 90 →
  -- External angle θ
  θ = 30 →
  θ = A + B →
  -- Conclusion
  C₁ = 75 ∧ C₂ = 75 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_relation_l4088_408838


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l4088_408861

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a - b) / (b + c) + (b - c) / (c + d) + (c - d) / (d + a) + (d - a) / (a + b) ≥ 0 ∧
  ((a - b) / (b + c) + (b - c) / (c + d) + (c - d) / (d + a) + (d - a) / (a + b) = 0 ↔ a = c ∧ b = d) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l4088_408861


namespace NUMINAMATH_CALUDE_equation_system_solution_l4088_408823

def solution_set (a b c x y z : ℝ) : Prop :=
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = c) ∨
  (x = a ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = b ∧ z = 0)

theorem equation_system_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ x y z : ℝ,
    (a * x + b * y = (x - y)^2 ∧
     b * y + c * z = (y - z)^2 ∧
     c * z + a * x = (z - x)^2) ↔
    solution_set a b c x y z :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l4088_408823


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4088_408884

-- Define the proposition
def proposition (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a*x - 4*a ≤ 0

-- Define the condition
def condition (a : ℝ) : Prop :=
  -16 ≤ a ∧ a ≤ 0

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, condition a → ¬proposition a) ∧
  ¬(∀ a : ℝ, ¬proposition a → condition a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4088_408884


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l4088_408821

theorem fenced_area_calculation (length width cutout_side : ℕ) : 
  length = 20 → width = 18 → cutout_side = 4 →
  (length * width) - (cutout_side * cutout_side) = 344 := by
sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l4088_408821


namespace NUMINAMATH_CALUDE_system_solution_l4088_408856

theorem system_solution (a b c d : ℚ) 
  (eq1 : 4 * a + 2 * b + 6 * c + 8 * d = 48)
  (eq2 : 4 * d + 2 * c = 2 * b)
  (eq3 : 4 * b + 2 * c = 2 * a)
  (eq4 : c + 2 = d) :
  a * b * c * d = -11033 / 1296 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l4088_408856


namespace NUMINAMATH_CALUDE_infinitely_many_n_congruent_to_sum_of_digits_l4088_408820

/-- Sum of digits in base r -/
def S_r (r : ℕ) (n : ℕ) : ℕ := sorry

/-- There are infinitely many n such that S_r(n) ≡ n (mod p) -/
theorem infinitely_many_n_congruent_to_sum_of_digits 
  (r : ℕ) (p : ℕ) (hr : r > 1) (hp : Nat.Prime p) :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, S_r r (f k) ≡ f k [MOD p] := by sorry

end NUMINAMATH_CALUDE_infinitely_many_n_congruent_to_sum_of_digits_l4088_408820


namespace NUMINAMATH_CALUDE_breakfast_cost_theorem_l4088_408848

/-- Represents the cost of breakfast items and the special offer. -/
structure BreakfastPrices where
  toast : ℚ
  egg : ℚ
  coffee : ℚ
  orange_juice : ℚ
  special_offer : ℚ

/-- Represents an individual's breakfast order. -/
structure BreakfastOrder where
  toast : ℕ
  egg : ℕ
  coffee : ℕ
  orange_juice : ℕ

/-- Calculates the cost of a breakfast order given the prices. -/
def orderCost (prices : BreakfastPrices) (order : BreakfastOrder) : ℚ :=
  prices.toast * order.toast +
  prices.egg * order.egg +
  (if order.coffee ≥ 2 then prices.special_offer else prices.coffee * order.coffee) +
  prices.orange_juice * order.orange_juice

/-- Calculates the total cost of all breakfast orders with service charge. -/
def totalCost (prices : BreakfastPrices) (orders : List BreakfastOrder) (serviceCharge : ℚ) : ℚ :=
  let subtotal := (orders.map (orderCost prices)).sum
  subtotal + subtotal * serviceCharge

/-- Theorem stating that the total breakfast cost is £48.40. -/
theorem breakfast_cost_theorem (prices : BreakfastPrices)
    (dale andrew melanie kevin : BreakfastOrder) :
    prices.toast = 1 →
    prices.egg = 3 →
    prices.coffee = 2 →
    prices.orange_juice = 3/2 →
    prices.special_offer = 7/2 →
    dale = { toast := 2, egg := 2, coffee := 1, orange_juice := 0 } →
    andrew = { toast := 1, egg := 2, coffee := 0, orange_juice := 1 } →
    melanie = { toast := 3, egg := 1, coffee := 0, orange_juice := 2 } →
    kevin = { toast := 4, egg := 3, coffee := 2, orange_juice := 0 } →
    totalCost prices [dale, andrew, melanie, kevin] (1/10) = 484/10 := by
  sorry


end NUMINAMATH_CALUDE_breakfast_cost_theorem_l4088_408848


namespace NUMINAMATH_CALUDE_symmetric_point_example_l4088_408852

/-- Given a line ax + by + c = 0 and two points P and Q, this function checks if Q is symmetric to P with respect to the line. -/
def is_symmetric_point (a b c : ℝ) (px py qx qy : ℝ) : Prop :=
  let mx := (px + qx) / 2
  let my := (py + qy) / 2
  (a * mx + b * my + c = 0) ∧ (a * (qx - px) + b * (qy - py) = 0)

/-- The point (3, 2) is symmetric to (-1, -2) with respect to the line x + y = 1 -/
theorem symmetric_point_example : is_symmetric_point 1 1 (-1) (-1) (-2) 3 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_example_l4088_408852


namespace NUMINAMATH_CALUDE_remainder_sum_l4088_408840

theorem remainder_sum (n : ℤ) (h : n % 12 = 5) : (n % 4 + n % 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l4088_408840


namespace NUMINAMATH_CALUDE_valentines_count_l4088_408851

theorem valentines_count (initial : ℕ) (given_away : ℕ) (received : ℕ) : 
  initial = 60 → given_away = 16 → received = 5 → 
  initial - given_away + received = 49 := by
  sorry

end NUMINAMATH_CALUDE_valentines_count_l4088_408851


namespace NUMINAMATH_CALUDE_area_ratio_EFWZ_ZWGH_l4088_408824

-- Define the points
variable (E F G H O Q Z W : ℝ × ℝ)

-- Define the lengths
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the conditions
axiom EF_eq_EO : length E F = length E O
axiom EO_eq_OG : length E O = length O G
axiom OG_eq_GH : length O G = length G H
axiom EF_eq_12 : length E F = 12
axiom FG_eq_18 : length F G = 18
axiom EH_eq_18 : length E H = 18
axiom OH_eq_18 : length O H = 18

-- Define Q as the point on FG such that OQ is perpendicular to FG
axiom Q_on_FG : sorry
axiom OQ_perp_FG : sorry

-- Define Z as midpoint of EF
axiom Z_midpoint_EF : sorry

-- Define W as midpoint of GH
axiom W_midpoint_GH : sorry

-- Define the area function for trapezoids
def area_trapezoid (A B C D : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_EFWZ_ZWGH : 
  area_trapezoid E F W Z = area_trapezoid Z W G H := by sorry

end NUMINAMATH_CALUDE_area_ratio_EFWZ_ZWGH_l4088_408824


namespace NUMINAMATH_CALUDE_pizza_theorem_l4088_408895

/-- Calculates the number of pizza slices remaining after a series of consumption events. -/
def remainingSlices (initialSlices : ℕ) : ℕ :=
  let afterLunch := initialSlices / 2
  let afterDinner := afterLunch - (afterLunch / 3)
  let afterSharing := afterDinner - (afterDinner / 4)
  afterSharing - (afterSharing / 5)

/-- Theorem stating that given 12 initial slices, 3 slices remain after the described events. -/
theorem pizza_theorem : remainingSlices 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l4088_408895


namespace NUMINAMATH_CALUDE_parallelogram_area_l4088_408857

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area : 
  ∀ (base height : ℝ), 
  base = 20 → 
  height = 4 → 
  base * height = 80 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l4088_408857


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l4088_408887

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of possible four-ring arrangements on four fingers of one hand,
    given seven distinguishable rings, where the order matters and not all
    fingers need to have a ring -/
def ring_arrangements : ℕ :=
  choose 7 4 * factorial 4 * choose 7 3

theorem ring_arrangements_count :
  ring_arrangements = 29400 := by sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l4088_408887


namespace NUMINAMATH_CALUDE_classroom_children_l4088_408828

theorem classroom_children (total_pencils : ℕ) (pencils_per_student : ℕ) (h1 : total_pencils = 8) (h2 : pencils_per_student = 2) :
  total_pencils / pencils_per_student = 4 :=
by sorry

end NUMINAMATH_CALUDE_classroom_children_l4088_408828


namespace NUMINAMATH_CALUDE_sum_greater_than_four_necessary_not_sufficient_l4088_408863

theorem sum_greater_than_four_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, (a > 1 ∧ b > 3) → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 1 ∧ b > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_necessary_not_sufficient_l4088_408863


namespace NUMINAMATH_CALUDE_three_people_selection_l4088_408876

-- Define the number of people in the group
def n : ℕ := 30

-- Define the number of enemies each person has
def enemies_per_person : ℕ := 6

-- Define the function to calculate the number of ways to select 3 people
-- such that any two of them are either friends or enemies
def select_three_people (n : ℕ) (enemies_per_person : ℕ) : ℕ :=
  -- The actual calculation is not implemented, as per instructions
  sorry

-- The theorem to prove
theorem three_people_selection :
  select_three_people n enemies_per_person = 1990 := by
  sorry

end NUMINAMATH_CALUDE_three_people_selection_l4088_408876


namespace NUMINAMATH_CALUDE_alpha_plus_two_beta_eq_pi_over_four_l4088_408859

theorem alpha_plus_two_beta_eq_pi_over_four 
  (α β : Real) 
  (acute_α : 0 < α ∧ α < π / 2) 
  (acute_β : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 1 / 7) 
  (sin_β : Real.sin β = Real.sqrt 10 / 10) : 
  α + 2 * β = π / 4 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_two_beta_eq_pi_over_four_l4088_408859


namespace NUMINAMATH_CALUDE_line_equation_sum_l4088_408819

/-- Given a line with slope -3 passing through the point (5, 2),
    prove that m + b = 14 where y = mx + b is the equation of the line. -/
theorem line_equation_sum (m b : ℝ) : 
  m = -3 → 
  2 = m * 5 + b → 
  m + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_sum_l4088_408819


namespace NUMINAMATH_CALUDE_sequence_formula_l4088_408807

theorem sequence_formula (a : ℕ → ℚ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 1 → (n + 1) * a n = 2 * n * a (n + 1)) :
    ∀ n : ℕ, n ≥ 1 → a n = n / (2^(n-1)) := by sorry

end NUMINAMATH_CALUDE_sequence_formula_l4088_408807


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_div_11_l4088_408880

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_number_with_digit_sum_div_11 (N : ℕ) : 
  ∃ k ∈ Finset.range 39, 11 ∣ sum_of_digits (N + k) := by sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_div_11_l4088_408880


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4088_408874

theorem complex_modulus_problem (z : ℂ) (h : (3 - I) / z = 1 + I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4088_408874


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4088_408843

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + 5 * x^3 + x + 20) - (x^6 + 4 * x^5 - 2 * x^4 + x^3 + 15) =
  x^6 - x^5 + 3 * x^4 + 4 * x^3 + x + 5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4088_408843


namespace NUMINAMATH_CALUDE_roots_opposite_signs_n_value_l4088_408805

/-- 
Given an equation of the form (x^2 - (a+1)x) / ((b+1)x - d) = (n-2) / (n+2),
if the roots of this equation are numerically equal but of opposite signs,
then n = 2(b-a) / (a+b+2).
-/
theorem roots_opposite_signs_n_value 
  (a b d n : ℝ) 
  (eq : ∀ x, (x^2 - (a+1)*x) / ((b+1)*x - d) = (n-2) / (n+2)) 
  (roots_opposite : ∃ r : ℝ, (r^2 - (a+1)*r) / ((b+1)*r - d) = (n-2) / (n+2) ∧ 
                              ((-r)^2 - (a+1)*(-r)) / ((b+1)*(-r) - d) = (n-2) / (n+2)) :
  n = 2*(b-a) / (a+b+2) := by
sorry

end NUMINAMATH_CALUDE_roots_opposite_signs_n_value_l4088_408805


namespace NUMINAMATH_CALUDE_parabola_directrix_l4088_408832

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the points
def origin : ℝ × ℝ := (0, 0)
def point_D : ℝ × ℝ := (1, 2)

-- Define the perpendicularity condition
def perpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p4.1 - p3.1) + (p2.2 - p1.2) * (p4.2 - p3.2) = 0

-- State the theorem
theorem parabola_directrix (p : ℝ) (A B : ℝ × ℝ) :
  parabola p A.1 A.2 ∧ 
  parabola p B.1 B.2 ∧ 
  perpendicular origin A origin B ∧
  perpendicular origin point_D A B →
  ∃ (x : ℝ), x = -5/4 ∧ ∀ (y : ℝ), parabola p x y → x = -p/2 :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4088_408832


namespace NUMINAMATH_CALUDE_grant_baseball_gear_sale_total_l4088_408872

theorem grant_baseball_gear_sale_total (cards_price bat_price glove_original_price glove_discount cleats_price cleats_count : ℝ) :
  cards_price = 25 →
  bat_price = 10 →
  glove_original_price = 30 →
  glove_discount = 0.2 →
  cleats_price = 10 →
  cleats_count = 2 →
  cards_price + bat_price + (glove_original_price * (1 - glove_discount)) + (cleats_price * cleats_count) = 79 := by
  sorry

end NUMINAMATH_CALUDE_grant_baseball_gear_sale_total_l4088_408872


namespace NUMINAMATH_CALUDE_number_puzzle_l4088_408862

theorem number_puzzle : ∃ x : ℤ, x + 3*12 + 3*13 + 3*16 = 134 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l4088_408862


namespace NUMINAMATH_CALUDE_larger_number_problem_l4088_408817

theorem larger_number_problem (S L : ℕ) 
  (h1 : L - S = 50000)
  (h2 : L = 13 * S + 317) :
  L = 54140 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l4088_408817


namespace NUMINAMATH_CALUDE_win_sector_area_l4088_408855

/-- Given a circular spinner with radius 8 cm and a probability of winning 3/8,
    the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * π * r^2 = 24 * π := by
sorry

end NUMINAMATH_CALUDE_win_sector_area_l4088_408855


namespace NUMINAMATH_CALUDE_total_sum_is_120_rupees_l4088_408849

/-- Represents the division of money among three people -/
structure MoneyDivision where
  a_share : ℕ  -- A's share in paisa per rupee
  b_share : ℕ  -- B's share in paisa per rupee
  c_share : ℕ  -- C's share in paisa per rupee

/-- The given problem setup -/
def problem_setup : MoneyDivision :=
  { a_share := 0,  -- We don't know A's exact share, so we leave it as 0
    b_share := 65,
    c_share := 40 }

/-- Theorem stating the total sum of money -/
theorem total_sum_is_120_rupees (md : MoneyDivision) 
  (h1 : md.b_share = 65)
  (h2 : md.c_share = 40)
  (h3 : md.a_share + md.b_share + md.c_share = 100)  -- Total per rupee is 100 paisa
  (h4 : md.c_share * 120 = 4800)  -- C's share is Rs. 48 (4800 paisa)
  : (4800 / md.c_share) * 100 = 12000 := by
  sorry

#check total_sum_is_120_rupees

end NUMINAMATH_CALUDE_total_sum_is_120_rupees_l4088_408849


namespace NUMINAMATH_CALUDE_train_speed_l4088_408897

/-- Proves that a train with given passing times has a specific speed -/
theorem train_speed (pole_passing_time : ℝ) (stationary_train_length : ℝ) (stationary_train_passing_time : ℝ) :
  pole_passing_time = 10 →
  stationary_train_length = 300 →
  stationary_train_passing_time = 40 →
  ∃ (train_length : ℝ),
    train_length > 0 ∧
    train_length / pole_passing_time = (train_length + stationary_train_length) / stationary_train_passing_time ∧
    train_length / pole_passing_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l4088_408897


namespace NUMINAMATH_CALUDE_base_10_to_base_8_l4088_408893

theorem base_10_to_base_8 : 
  ∃ (a b c d : ℕ), 
    947 = a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0 ∧ 
    a = 1 ∧ b = 6 ∧ c = 6 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_8_l4088_408893


namespace NUMINAMATH_CALUDE_sin_cos_pi_twelve_eq_one_fourth_l4088_408879

theorem sin_cos_pi_twelve_eq_one_fourth : 
  Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sin_cos_pi_twelve_eq_one_fourth_l4088_408879


namespace NUMINAMATH_CALUDE_identical_permutations_of_increasing_sum_l4088_408822

/-- A strictly increasing finite sequence of real numbers -/
def StrictlyIncreasingSeq (a : Fin n → ℝ) : Prop :=
  ∀ i j : Fin n, i < j → a i < a j

/-- A permutation of indices -/
def IsPermutation (σ : Fin n → Fin n) : Prop :=
  Function.Bijective σ

theorem identical_permutations_of_increasing_sum
  (a : Fin n → ℝ) (σ : Fin n → Fin n)
  (h_inc : StrictlyIncreasingSeq a)
  (h_perm : IsPermutation σ)
  (h_sum_inc : StrictlyIncreasingSeq (fun i => a i + a (σ i))) :
  ∀ i, a i = a (σ i) := by
sorry

end NUMINAMATH_CALUDE_identical_permutations_of_increasing_sum_l4088_408822


namespace NUMINAMATH_CALUDE_min_sum_of_distances_l4088_408842

/-- The curve on which point P moves -/
def curve (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The first line l₁ -/
def line1 (y : ℝ) : Prop := y = 2

/-- The second line l₂ -/
def line2 (x : ℝ) : Prop := x = -1

/-- The distance from a point (x, y) to line1 -/
def dist_to_line1 (y : ℝ) : ℝ := |y - 2|

/-- The distance from a point (x, y) to line2 -/
def dist_to_line2 (x : ℝ) : ℝ := |x + 1|

/-- The sum of distances from a point (x, y) to both lines -/
def sum_of_distances (x y : ℝ) : ℝ := dist_to_line1 y + dist_to_line2 x

/-- The theorem stating the minimum value of the sum of distances -/
theorem min_sum_of_distances :
  ∃ (min : ℝ), min = 4 - Real.sqrt 2 ∧
  ∀ (x y : ℝ), curve x y → sum_of_distances x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_l4088_408842


namespace NUMINAMATH_CALUDE_t_equality_l4088_408883

theorem t_equality (t : ℝ) : t = 1 / (1 - 2^(1/4)) → t = -(1 + 2^(1/4)) * (1 + 2^(1/2)) := by
  sorry

end NUMINAMATH_CALUDE_t_equality_l4088_408883


namespace NUMINAMATH_CALUDE_base_k_addition_l4088_408818

/-- Represents a digit in base k -/
def Digit (k : ℕ) := Fin k

/-- Converts a natural number to its representation in base k -/
def toBaseK (n : ℕ) (k : ℕ) : List (Digit k) :=
  sorry

/-- Adds two numbers represented in base k -/
def addBaseK (a b : List (Digit k)) : List (Digit k) :=
  sorry

/-- Checks if two lists of digits are equal -/
def digitListEq (a b : List (Digit k)) : Prop :=
  sorry

theorem base_k_addition :
  ∃ k : ℕ, k > 1 ∧
    digitListEq
      (addBaseK (toBaseK 8374 k) (toBaseK 9423 k))
      (toBaseK 20397 k) ∧
    k = 18 :=
  sorry

end NUMINAMATH_CALUDE_base_k_addition_l4088_408818


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l4088_408829

theorem arithmetic_sequence_length :
  ∀ (a₁ aₙ d n : ℤ),
    a₁ = -38 →
    aₙ = 69 →
    d = 6 →
    aₙ = a₁ + (n - 1) * d →
    n = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l4088_408829


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_equal_distances_m_value_l4088_408846

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def A : Point := { x := -3, y := -4 }
def B : Point := { x := 6, y := 3 }

def perpendicular_bisector (p1 p2 : Point) : Line := sorry

def distance_to_line (p : Point) (l : Line) : ℝ := sorry

theorem perpendicular_bisector_equation :
  perpendicular_bisector A B = { a := 9, b := 7, c := -10 } := by sorry

theorem equal_distances_m_value (m : ℝ) :
  let l : Line := { a := 1, b := m, c := 1 }
  distance_to_line A l = distance_to_line B l → m = 5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_equal_distances_m_value_l4088_408846


namespace NUMINAMATH_CALUDE_hyperbola_sum_theorem_l4088_408892

def F₁ : ℝ × ℝ := (2, -1)
def F₂ : ℝ × ℝ := (2, 3)

def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  abs (dist P F₁ - dist P F₂) = 2

def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

theorem hyperbola_sum_theorem (h k a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x y, hyperbola_equation x y h k a b ↔ is_on_hyperbola (x, y)) :
  h + k + a + b = 4 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_theorem_l4088_408892


namespace NUMINAMATH_CALUDE_unique_numbers_satisfying_condition_l4088_408873

theorem unique_numbers_satisfying_condition : ∃! (a b : ℕ), 
  100 ≤ a ∧ a < 1000 ∧ 
  1000 ≤ b ∧ b < 10000 ∧ 
  10000 * a + b = 7 * a * b ∧ 
  a + b = 1458 := by sorry

end NUMINAMATH_CALUDE_unique_numbers_satisfying_condition_l4088_408873


namespace NUMINAMATH_CALUDE_representation_of_2015_l4088_408802

theorem representation_of_2015 : ∃ (a b c : ℤ),
  a + b + c = 2015 ∧
  Nat.Prime a.natAbs ∧
  ∃ (k : ℤ), b = 3 * k ∧
  400 < c ∧ c < 500 ∧
  ¬∃ (m : ℤ), c = 3 * m :=
by sorry

end NUMINAMATH_CALUDE_representation_of_2015_l4088_408802


namespace NUMINAMATH_CALUDE_derek_average_increase_l4088_408831

def derek_scores : List ℝ := [92, 86, 89, 94, 91]

theorem derek_average_increase :
  let first_three := derek_scores.take 3
  let all_five := derek_scores
  (all_five.sum / all_five.length) - (first_three.sum / first_three.length) = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_derek_average_increase_l4088_408831


namespace NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l4088_408801

/-- Given a quadratic equation x^2 + 2x - k = 0 with real roots, prove that k ≥ -1 -/
theorem quadratic_real_roots_k_range (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*x - k = 0) →
  k ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l4088_408801


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_two_min_value_achieved_l4088_408806

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x + y) / (x*y) = 7/2 + Real.sqrt 6) : 
  ∀ a b : ℝ, a > 0 → b > 0 → (2*a + b) / (a*b) = 7/2 + Real.sqrt 6 → x + 3*y ≤ a + 3*b :=
sorry

theorem min_value_is_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x + y) / (x*y) = 7/2 + Real.sqrt 6) : 
  x + 3*y ≥ 2 :=
sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x + y) / (x*y) = 7/2 + Real.sqrt 6) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (2*a + b) / (a*b) = 7/2 + Real.sqrt 6 ∧ a + 3*b = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_two_min_value_achieved_l4088_408806


namespace NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_l4088_408814

-- Define a type for lines in 3D space
structure Line3D where
  -- You might represent a line by a point and a direction vector
  -- But for this abstract proof, we don't need to specify the internals
  mk :: 

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop := sorry

-- The theorem to prove
theorem perpendicular_parallel_perpendicular 
  (l1 l2 l3 : Line3D) : 
  perpendicular l1 l2 → parallel l2 l3 → perpendicular l1 l3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_l4088_408814


namespace NUMINAMATH_CALUDE_dog_weight_ratio_l4088_408810

theorem dog_weight_ratio (chihuahua pitbull great_dane : ℝ) : 
  chihuahua + pitbull + great_dane = 439 →
  great_dane = 307 →
  great_dane = 3 * pitbull + 10 →
  pitbull / chihuahua = 3 := by
  sorry

end NUMINAMATH_CALUDE_dog_weight_ratio_l4088_408810


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l4088_408845

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l4088_408845


namespace NUMINAMATH_CALUDE_gcd_78_36_l4088_408888

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_36_l4088_408888


namespace NUMINAMATH_CALUDE_jemma_grasshopper_count_l4088_408860

/-- The number of grasshoppers Jemma found on the African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found on the grass -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshopper_count : total_grasshoppers = 31 := by
  sorry

end NUMINAMATH_CALUDE_jemma_grasshopper_count_l4088_408860


namespace NUMINAMATH_CALUDE_cube_volume_problem_l4088_408878

theorem cube_volume_problem (x : ℝ) (h : x > 0) (eq : x^3 + 6*x^2 = 16*x) :
  27 * x^3 = 216 := by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l4088_408878


namespace NUMINAMATH_CALUDE_power_of_seven_mod_2000_l4088_408835

theorem power_of_seven_mod_2000 : 7^2023 % 2000 = 1849 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_2000_l4088_408835


namespace NUMINAMATH_CALUDE_f_sum_equals_six_l4088_408800

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 9
  else 4^(-x) + 3/2

-- Theorem statement
theorem f_sum_equals_six :
  f 27 + f (-Real.log 3 / Real.log 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_six_l4088_408800


namespace NUMINAMATH_CALUDE_remainder_of_repeated_sequence_l4088_408871

/-- The sequence of digits that is repeated to form the number -/
def digit_sequence : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- The number of digits in the large number -/
def total_digits : Nat := 2012

/-- The theorem stating that the remainder when the 2012-digit number
    formed by repeating the sequence 1, 2, 3, 4, 5, 6, 7, 8, 9
    is divided by 9 is equal to 6 -/
theorem remainder_of_repeated_sequence :
  (List.sum (List.take (total_digits % digit_sequence.length) digit_sequence)) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_repeated_sequence_l4088_408871


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l4088_408815

theorem degree_to_radian_conversion :
  ∃ (k : ℤ) (α : ℝ), 
    -885 * (π / 180) = 2 * k * π + α ∧
    0 ≤ α ∧ α ≤ 2 * π ∧
    2 * k * π + α = -6 * π + 13 * π / 12 :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l4088_408815


namespace NUMINAMATH_CALUDE_circle_equation_specific_l4088_408881

/-- The equation of a circle with center (h, k) and radius r -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The equation of a circle with center (2, -3) and radius 4 -/
theorem circle_equation_specific : 
  CircleEquation 2 (-3) 4 x y ↔ (x - 2)^2 + (y + 3)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_specific_l4088_408881


namespace NUMINAMATH_CALUDE_student_heights_average_l4088_408864

theorem student_heights_average :
  ∀ (h1 h2 h3 h4 : ℝ),
    h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4 →
    max h1 (max h2 (max h3 h4)) = 152 →
    min h1 (min h2 (min h3 h4)) = 137 →
    ∃ (avg : ℝ), avg = 145 ∧ (h1 + h2 + h3 + h4) / 4 = avg :=
by sorry

end NUMINAMATH_CALUDE_student_heights_average_l4088_408864


namespace NUMINAMATH_CALUDE_third_face_area_is_60_l4088_408841

/-- Represents a cuboidal box with given dimensions -/
structure CuboidalBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The area of the first adjacent face -/
def first_face_area (box : CuboidalBox) : ℝ := box.length * box.width

/-- The area of the second adjacent face -/
def second_face_area (box : CuboidalBox) : ℝ := box.width * box.height

/-- The area of the third adjacent face -/
def third_face_area (box : CuboidalBox) : ℝ := box.length * box.height

/-- The volume of the box -/
def volume (box : CuboidalBox) : ℝ := box.length * box.width * box.height

/-- Theorem stating the area of the third face given the conditions -/
theorem third_face_area_is_60 (box : CuboidalBox) 
  (h1 : first_face_area box = 120)
  (h2 : second_face_area box = 72)
  (h3 : volume box = 720) :
  third_face_area box = 60 := by
  sorry


end NUMINAMATH_CALUDE_third_face_area_is_60_l4088_408841


namespace NUMINAMATH_CALUDE_triangle_side_length_l4088_408850

theorem triangle_side_length (A B C : ℝ) (AC : ℝ) (sinA sinB : ℝ) (cosC : ℝ) :
  AC = 3 →
  3 * sinA = 2 * sinB →
  cosC = 1 / 4 →
  ∃ (AB : ℝ), AB = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4088_408850


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l4088_408870

theorem quadratic_roots_sum (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 2023 = 0 → x₂^2 + x₂ - 2023 = 0 → x₁^2 + 2*x₁ + x₂ = 2022 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l4088_408870


namespace NUMINAMATH_CALUDE_equal_potato_distribution_l4088_408847

theorem equal_potato_distribution (total_potatoes : ℕ) (family_members : ℕ) 
  (h1 : total_potatoes = 60) (h2 : family_members = 6) :
  total_potatoes / family_members = 10 := by
  sorry

end NUMINAMATH_CALUDE_equal_potato_distribution_l4088_408847


namespace NUMINAMATH_CALUDE_two_distinct_real_roots_l4088_408886

def polynomial (a x : ℝ) : ℝ := x^4 + 3*a*x^3 + a*(1-5*a^2)*x - 3*a^4 + a^2 + 1

theorem two_distinct_real_roots (a : ℝ) :
  (∃ x : ℝ, polynomial a x = 0) ∧ 
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ polynomial a x₁ = 0 ∧ polynomial a x₂ = 0) →
  a = 2 * Real.sqrt 26 / 13 ∨ a = -2 * Real.sqrt 26 / 13 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_real_roots_l4088_408886


namespace NUMINAMATH_CALUDE_farm_animals_feet_count_l4088_408816

theorem farm_animals_feet_count (total_heads : Nat) (hen_count : Nat) : 
  total_heads = 60 → hen_count = 20 → (total_heads - hen_count) * 4 + hen_count * 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_feet_count_l4088_408816


namespace NUMINAMATH_CALUDE_complex_division_result_l4088_408825

theorem complex_division_result (z : ℂ) (h : z = 1 - Complex.I * Real.sqrt 3) : 
  4 / z = 1 + Complex.I * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_division_result_l4088_408825


namespace NUMINAMATH_CALUDE_minutes_to_seconds_l4088_408803

theorem minutes_to_seconds (minutes : ℝ) : minutes * 60 = 750 → minutes = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_seconds_l4088_408803


namespace NUMINAMATH_CALUDE_pizza_order_l4088_408839

theorem pizza_order (total_slices : ℕ) (slices_per_pizza : ℕ) (h1 : total_slices = 14) (h2 : slices_per_pizza = 2) :
  total_slices / slices_per_pizza = 7 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l4088_408839


namespace NUMINAMATH_CALUDE_min_value_theorem_l4088_408858

/-- The minimum value of (x₁² + x₂²) / (x₁ - x₂) given the conditions -/
theorem min_value_theorem (a c m n x₁ x₂ : ℝ) : 
  (2 * a * m + (a + c) * n + 2 * c = 0) →  -- line passes through (m, n)
  (x₁ + x₂ + m + n = 15) →                 -- sum condition
  (x₁ > x₂) →                              -- ordering condition
  (∀ y₁ y₂ : ℝ, (y₁ + y₂ + m + n = 15) → (y₁ > y₂) → 
    (x₁^2 + x₂^2) / (x₁ - x₂) ≤ (y₁^2 + y₂^2) / (y₁ - y₂)) →
  (x₁^2 + x₂^2) / (x₁ - x₂) = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4088_408858


namespace NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l4088_408834

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Determines if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Given point A in the second quadrant, prove that point B is in the fourth quadrant -/
theorem point_B_in_fourth_quadrant (m n : ℝ) (h : isInSecondQuadrant ⟨m, n⟩) :
  isInFourthQuadrant ⟨2*n - m, -n + m⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l4088_408834


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4088_408844

/-- Given a geometric sequence {a_n} with all positive terms, where 3a_1, (1/2)a_3, 2a_2 form an arithmetic sequence, (a_11 + a_13) / (a_8 + a_10) = 27. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  (3 * a 1 - (1/2) * a 3 = (1/2) * a 3 - 2 * a 2) →  -- arithmetic sequence condition
  (a 11 + a 13) / (a 8 + a 10) = 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4088_408844


namespace NUMINAMATH_CALUDE_expression_evaluation_l4088_408898

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b - 11)
  (h2 : b = a + 3)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4088_408898


namespace NUMINAMATH_CALUDE_statements_correctness_l4088_408869

theorem statements_correctness : 
  (∃! n : ℕ, n = 3 ∧ 
    (2^3 = 8) ∧ 
    (∀ r : ℚ, ∃ s : ℚ, s < r) ∧ 
    (∀ x : ℝ, x + x = 0 → x = 0) ∧ 
    (Real.sqrt ((-4)^2) ≠ 4) ∧ 
    (∃ x : ℝ, x ≠ 1 ∧ 1 / x ≠ 1)) :=
by sorry

end NUMINAMATH_CALUDE_statements_correctness_l4088_408869


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_congruences_l4088_408836

theorem smallest_n_satisfying_congruences : 
  ∃ n : ℕ, n > 20 ∧ n % 6 = 4 ∧ n % 7 = 5 ∧ 
  ∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 5 → n ≤ m :=
by
  use 40
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_congruences_l4088_408836


namespace NUMINAMATH_CALUDE_field_trip_students_l4088_408865

theorem field_trip_students (van_capacity : ℕ) (num_vans : ℕ) (num_adults : ℕ) 
  (h1 : van_capacity = 9)
  (h2 : num_vans = 6)
  (h3 : num_adults = 14) :
  num_vans * van_capacity - num_adults = 40 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l4088_408865


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4088_408813

theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (3 * x^2 * y = 3 * 3^2 * 15) → (y = 6750) → (x = Real.sqrt 2 / 10) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4088_408813


namespace NUMINAMATH_CALUDE_marbles_in_container_l4088_408899

/-- Given that a container with volume 24 cm³ holds 75 marbles, 
    prove that a container with volume 72 cm³ holds 225 marbles, 
    assuming the number of marbles is proportional to the volume. -/
theorem marbles_in_container (v₁ v₂ : ℝ) (m₁ m₂ : ℕ) 
  (h₁ : v₁ = 24) (h₂ : v₂ = 72) (h₃ : m₁ = 75) 
  (h₄ : v₁ * m₂ = v₂ * m₁) : m₂ = 225 := by
  sorry

end NUMINAMATH_CALUDE_marbles_in_container_l4088_408899


namespace NUMINAMATH_CALUDE_total_animals_hunted_l4088_408811

/- Define the number of animals hunted by each person -/
def sam_hunt : ℕ := 6

def rob_hunt : ℕ := sam_hunt / 2

def rob_sam_total : ℕ := sam_hunt + rob_hunt

def mark_hunt : ℕ := rob_sam_total / 3

def peter_hunt : ℕ := 3 * mark_hunt

/- Theorem to prove -/
theorem total_animals_hunted : sam_hunt + rob_hunt + mark_hunt + peter_hunt = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_hunted_l4088_408811


namespace NUMINAMATH_CALUDE_parallel_resistor_calculation_l4088_408827

/-- Calculates the resistance of the second resistor in a parallel circuit -/
theorem parallel_resistor_calculation (R1 R_total : ℝ) (h1 : R1 = 9) (h2 : R_total = 4.235294117647059) :
  ∃ R2 : ℝ, R2 = 8 ∧ 1 / R_total = 1 / R1 + 1 / R2 := by
sorry

end NUMINAMATH_CALUDE_parallel_resistor_calculation_l4088_408827


namespace NUMINAMATH_CALUDE_cut_rectangle_properties_l4088_408868

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the result of cutting a smaller rectangle from a larger one -/
structure CutRectangle where
  original : Rectangle
  cut : Rectangle

/-- The resulting figure after cutting -/
def resultingFigure (cr : CutRectangle) : ℝ := area cr.original - area cr.cut

theorem cut_rectangle_properties (R : Rectangle) (S : CutRectangle) 
    (h1 : S.original = R) 
    (h2 : area S.cut > 0) 
    (h3 : S.cut.length < R.length ∧ S.cut.width < R.width) :
  resultingFigure S < area R ∧ perimeter R = perimeter S.original :=
by sorry

end NUMINAMATH_CALUDE_cut_rectangle_properties_l4088_408868


namespace NUMINAMATH_CALUDE_circle_area_when_six_reciprocal_circumference_equals_diameter_l4088_408867

/-- Given a circle where six times the reciprocal of its circumference equals its diameter, the area of the circle is 3/2 -/
theorem circle_area_when_six_reciprocal_circumference_equals_diameter (r : ℝ) (h : 6 * (1 / (2 * Real.pi * r)) = 2 * r) : π * r^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_when_six_reciprocal_circumference_equals_diameter_l4088_408867


namespace NUMINAMATH_CALUDE_inequality_proof_l4088_408837

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4088_408837


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l4088_408890

theorem system_of_inequalities_solution (x : ℝ) :
  (x < x / 5 + 4 ∧ 4 * x + 1 > 3 * (2 * x - 1)) → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l4088_408890


namespace NUMINAMATH_CALUDE_decimal_77_to_octal_l4088_408877

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

theorem decimal_77_to_octal :
  decimal_to_octal 77 = [5, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_77_to_octal_l4088_408877


namespace NUMINAMATH_CALUDE_distinct_remainders_l4088_408875

theorem distinct_remainders (n : ℕ+) :
  ∀ (i j : Fin n), i ≠ j →
    (2 * i.val + 1) ^ (2 * i.val + 1) % (2 ^ n.val) ≠
    (2 * j.val + 1) ^ (2 * j.val + 1) % (2 ^ n.val) := by
  sorry

end NUMINAMATH_CALUDE_distinct_remainders_l4088_408875


namespace NUMINAMATH_CALUDE_circle_equation_l4088_408896

/-- A circle with center on the x-axis, passing through the origin, and tangent to the line y = 4 has the general equation x^2 + y^2 ± 8x = 0. -/
theorem circle_equation (x y : ℝ) : 
  ∃ (a : ℝ), 
    (∀ (x₀ y₀ : ℝ), (x₀ - a)^2 + y₀^2 = 16 → y₀ ≤ 4) ∧  -- Circle is tangent to y = 4
    ((0 - a)^2 + 0^2 = 16) ∧                             -- Circle passes through origin
    (a^2 = 16) →                                         -- Center is on x-axis at distance 4 from origin
  (x^2 + y^2 + 8*x = 0 ∨ x^2 + y^2 - 8*x = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l4088_408896
