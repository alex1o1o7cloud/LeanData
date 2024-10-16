import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2645_264514

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 2*x₀ + 4 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2645_264514


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l2645_264573

/-- The value of m for which the ellipse x^2 + 9y^2 = 9 is tangent to the hyperbola x^2 - m(y + 3)^2 = 1 -/
theorem ellipse_hyperbola_tangent : ∃ (m : ℝ), 
  (∀ (x y : ℝ), x^2 + 9*y^2 = 9 ∧ x^2 - m*(y + 3)^2 = 1) →
  (∃! (x y : ℝ), x^2 + 9*y^2 = 9 ∧ x^2 - m*(y + 3)^2 = 1) →
  m = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l2645_264573


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l2645_264593

/-- The volume of a cone formed by rolling up a three-quarter sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 4) :
  let sector_angle : ℝ := 3 * π / 2
  let base_radius : ℝ := sector_angle * r / (2 * π)
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  (1/3) * π * base_radius^2 * cone_height = 3 * π * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l2645_264593


namespace NUMINAMATH_CALUDE_train_length_l2645_264548

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 12 → speed_kmh * (1000 / 3600) * time_s = 240 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2645_264548


namespace NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l2645_264508

theorem sum_reciprocals_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 1/y ≥ 2 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ 1/a + 1/b = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l2645_264508


namespace NUMINAMATH_CALUDE_painting_time_relation_l2645_264524

/-- Time taken by Taylor to paint the room alone -/
def taylor_time : ℝ := 12

/-- Time taken by Taylor and Jennifer together to paint the room -/
def combined_time : ℝ := 5.45454545455

/-- Time taken by Jennifer to paint the room alone -/
def jennifer_time : ℝ := 10.1538461538

/-- Theorem stating the relationship between individual and combined painting times -/
theorem painting_time_relation :
  1 / taylor_time + 1 / jennifer_time = 1 / combined_time :=
sorry

end NUMINAMATH_CALUDE_painting_time_relation_l2645_264524


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_four_l2645_264590

theorem smallest_n_divisible_by_four :
  ∃ (n : ℕ), (7 * (n - 3)^5 - n^2 + 16*n - 30) % 4 = 0 ∧
  ∀ (m : ℕ), m < n → (7 * (m - 3)^5 - m^2 + 16*m - 30) % 4 ≠ 0 ∧
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_four_l2645_264590


namespace NUMINAMATH_CALUDE_cost_calculation_l2645_264512

theorem cost_calculation (N P M : ℚ) 
  (eq1 : 13 * N + 26 * P + 19 * M = 25)
  (eq2 : 27 * N + 18 * P + 31 * M = 31) :
  24 * N + 120 * P + 52 * M = 88 := by
sorry

end NUMINAMATH_CALUDE_cost_calculation_l2645_264512


namespace NUMINAMATH_CALUDE_decimal_521_to_octal_l2645_264536

-- Define a function to convert decimal to octal
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

-- Theorem statement
theorem decimal_521_to_octal :
  decimal_to_octal 521 = [1, 0, 1, 1] := by sorry

end NUMINAMATH_CALUDE_decimal_521_to_octal_l2645_264536


namespace NUMINAMATH_CALUDE_equation_solution_l2645_264528

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 4.5 ∧ x₂ = -3) ∧ 
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ -3 → (18 / (x^2 - 9) - 3 / (x - 3) = 2 ↔ (x = x₁ ∨ x = x₂))) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2645_264528


namespace NUMINAMATH_CALUDE_sum_product_equality_l2645_264521

theorem sum_product_equality : 3 * 12 + 3 * 13 + 3 * 16 + 11 = 134 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l2645_264521


namespace NUMINAMATH_CALUDE_no_formula_fits_all_pairs_l2645_264513

-- Define the pairs of x and y values
def xy_pairs : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 35), (4, 69), (5, 119)]

-- Define the formulas
def formula_A (x : ℕ) : ℕ := x^3 + x^2 + x + 2
def formula_B (x : ℕ) : ℕ := 3*x^2 + 2*x + 1
def formula_C (x : ℕ) : ℕ := 2*x^3 - x + 4
def formula_D (x : ℕ) : ℕ := 3*x^3 + 2*x^2 + x + 1

-- Theorem statement
theorem no_formula_fits_all_pairs :
  ∀ (pair : ℕ × ℕ), pair ∈ xy_pairs →
    (formula_A pair.1 ≠ pair.2) ∧
    (formula_B pair.1 ≠ pair.2) ∧
    (formula_C pair.1 ≠ pair.2) ∧
    (formula_D pair.1 ≠ pair.2) :=
by sorry

end NUMINAMATH_CALUDE_no_formula_fits_all_pairs_l2645_264513


namespace NUMINAMATH_CALUDE_jeff_fills_ten_boxes_l2645_264586

/-- The number of boxes Jeff can fill with his donuts -/
def boxes_filled (donuts_per_day : ℕ) (days : ℕ) (jeff_eats_per_day : ℕ) (chris_eats : ℕ) (donuts_per_box : ℕ) : ℕ :=
  ((donuts_per_day * days) - (jeff_eats_per_day * days) - chris_eats) / donuts_per_box

/-- Theorem stating that Jeff can fill 10 boxes with his donuts -/
theorem jeff_fills_ten_boxes :
  boxes_filled 10 12 1 8 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jeff_fills_ten_boxes_l2645_264586


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l2645_264526

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧ 
  N = 23 ∧ 
  (1743 % N = 2019 % N) ∧ 
  (2019 % N = 3008 % N) ∧ 
  ∀ (M : ℕ), M > N → (1743 % M ≠ 2019 % M ∨ 2019 % M ≠ 3008 % M) := by
  sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l2645_264526


namespace NUMINAMATH_CALUDE_unique_integer_product_l2645_264511

/-- A function that returns true if the given number uses each digit from the given list exactly once -/
def uses_digits_once (n : ℕ) (digits : List ℕ) : Prop :=
  sorry

/-- A function that combines two natural numbers into a single number -/
def combine_numbers (a b : ℕ) : ℕ :=
  sorry

theorem unique_integer_product : ∃! n : ℕ, 
  uses_digits_once (combine_numbers (4 * n) (5 * n)) [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧ 
  n = 2469 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_product_l2645_264511


namespace NUMINAMATH_CALUDE_cube_sum_product_l2645_264544

def is_even_or_prime (n : ℕ) : Prop :=
  Even n ∨ Nat.Prime n

theorem cube_sum_product : ∃ (a b : ℕ), 
  a^3 + b^3 = 91 ∧ 
  is_even_or_prime a ∧ 
  is_even_or_prime b ∧ 
  a * b = 12 := by sorry

end NUMINAMATH_CALUDE_cube_sum_product_l2645_264544


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2645_264531

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) :
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3 / 2 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) :
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) = 3 / 2 ↔
  a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2645_264531


namespace NUMINAMATH_CALUDE_diamond_four_three_l2645_264500

/-- Diamond operation: a ◇ b = 4a + 3b - ab + a² + b² -/
def diamond (a b : ℝ) : ℝ := 4*a + 3*b - a*b + a^2 + b^2

theorem diamond_four_three : diamond 4 3 = 38 := by sorry

end NUMINAMATH_CALUDE_diamond_four_three_l2645_264500


namespace NUMINAMATH_CALUDE_median_length_l2645_264569

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10
  right_angle : a^2 + b^2 = c^2

/-- The length of the median to the longest side of the triangle -/
def median_to_longest_side (t : RightTriangle) : ℝ := 5

/-- Theorem: The length of the median to the longest side is 5 -/
theorem median_length (t : RightTriangle) : median_to_longest_side t = 5 := by
  sorry

end NUMINAMATH_CALUDE_median_length_l2645_264569


namespace NUMINAMATH_CALUDE_berry_box_problem_l2645_264572

/-- Represents the number of berries in different colored boxes -/
structure BerryBoxes where
  blue : ℕ  -- number of blueberries in a blue box
  red : ℕ   -- number of strawberries in a red box
  green : ℕ -- number of raspberries in a green box

/-- Theorem representing the berry box problem -/
theorem berry_box_problem (boxes : BerryBoxes) : 
  (boxes.red - boxes.blue = 12) ∧ 
  (boxes.red - boxes.green = 25) := by
  sorry

#check berry_box_problem

end NUMINAMATH_CALUDE_berry_box_problem_l2645_264572


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2645_264598

theorem child_ticket_cost (num_adults num_children : ℕ) (adult_ticket_cost : ℚ) (extra_cost : ℚ) :
  num_adults = 9 →
  num_children = 7 →
  adult_ticket_cost = 11 →
  extra_cost = 50 →
  ∃ (child_ticket_cost : ℚ),
    num_adults * adult_ticket_cost = num_children * child_ticket_cost + extra_cost ∧
    child_ticket_cost = 7 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2645_264598


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l2645_264520

theorem tangent_line_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : ∃ x y : ℝ, y = x - 2*a ∧ y = Real.log (x + b) ∧ 
    (Real.exp y) * (1 / (x + b)) = 1) :
  (1/a + 2/b) ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l2645_264520


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l2645_264539

theorem consecutive_integers_problem (a b c d e : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  b = a + 1 → c = b + 1 → d = c + 1 → e = d + 1 →
  a < b → b < c → c < d → d < e →
  a + b = e - 1 →
  a * b = d + 1 →
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l2645_264539


namespace NUMINAMATH_CALUDE_range_of_a_given_max_value_l2645_264534

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x - a| + a

/-- The theorem stating the range of a given the maximum value of f(x) -/
theorem range_of_a_given_max_value :
  (∃ (a : ℝ), (∀ x ∈ Set.Icc (-1) 3, f a x ≤ 3) ∧ 
   (∃ x ∈ Set.Icc (-1) 3, f a x = 3)) ↔ 
  (∀ a : ℝ, a ≤ -1) := by sorry

end NUMINAMATH_CALUDE_range_of_a_given_max_value_l2645_264534


namespace NUMINAMATH_CALUDE_family_composition_l2645_264538

theorem family_composition :
  ∀ (boys girls : ℕ),
  (boys > 0 ∧ girls > 0) →
  (boys - 1 = girls) →
  (boys = 2 * (girls - 1)) →
  (boys = 4 ∧ girls = 3) :=
by sorry

end NUMINAMATH_CALUDE_family_composition_l2645_264538


namespace NUMINAMATH_CALUDE_product_of_D_coordinates_l2645_264599

-- Define the points
def C : ℝ × ℝ := (-2, -7)
def M : ℝ × ℝ := (4, -3)

-- Define D as a variable point
variable (D : ℝ × ℝ)

-- State the theorem
theorem product_of_D_coordinates : 
  (M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) → D.1 * D.2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_of_D_coordinates_l2645_264599


namespace NUMINAMATH_CALUDE_function_symmetry_theorem_l2645_264543

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define the concept of symmetry about y-axis
def symmetric_about_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_symmetry_theorem :
  symmetric_about_y_axis (fun x ↦ f (x - 1)) exp →
  f = fun x ↦ exp (-x - 1) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_theorem_l2645_264543


namespace NUMINAMATH_CALUDE_complex_square_equation_l2645_264589

theorem complex_square_equation (a b : ℕ+) :
  (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I →
  a + b * Complex.I = 4 + 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_equation_l2645_264589


namespace NUMINAMATH_CALUDE_circle_passes_through_origin_circle_passes_through_four_zero_circle_passes_through_neg_one_one_is_circle_equation_l2645_264507

/-- A circle passing through the points (0,0), (4,0), and (-1,1) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- The circle passes through the point (0,0) -/
theorem circle_passes_through_origin :
  circle_equation 0 0 := by sorry

/-- The circle passes through the point (4,0) -/
theorem circle_passes_through_four_zero :
  circle_equation 4 0 := by sorry

/-- The circle passes through the point (-1,1) -/
theorem circle_passes_through_neg_one_one :
  circle_equation (-1) 1 := by sorry

/-- The equation represents a circle -/
theorem is_circle_equation :
  ∃ (h k r : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_origin_circle_passes_through_four_zero_circle_passes_through_neg_one_one_is_circle_equation_l2645_264507


namespace NUMINAMATH_CALUDE_cos_180_degrees_l2645_264591

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l2645_264591


namespace NUMINAMATH_CALUDE_cube_side_area_l2645_264563

/-- Given a cube with a total surface area of 54.3 square centimeters,
    the area of one side is 9.05 square centimeters. -/
theorem cube_side_area (total_area : ℝ) (h1 : total_area = 54.3) : ∃ (side_area : ℝ), 
  side_area = 9.05 ∧ 6 * side_area = total_area := by
  sorry

end NUMINAMATH_CALUDE_cube_side_area_l2645_264563


namespace NUMINAMATH_CALUDE_simplify_expression_l2645_264582

theorem simplify_expression : 
  ((5^1010)^2 - (5^1008)^2) / ((5^1009)^2 - (5^1007)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2645_264582


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2645_264566

/-- Given a hyperbola with equation x²/4 - y²/9 = 1, its asymptotes are y = ±(3/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 4 - y^2 / 9 = 1 →
  ∃ (k : ℝ), k = 3/2 ∧ (y = k*x ∨ y = -k*x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2645_264566


namespace NUMINAMATH_CALUDE_cover_plane_with_ten_usage_cannot_cover_plane_with_single_usage_l2645_264581

/-- Represents a square tile with a side length that is a power of 2 -/
structure SquareTile where
  sideLength : ℕ
  is_power_of_two : ∃ n : ℕ, sideLength = 2^n

/-- Represents a tiling of the plane using square tiles -/
structure PlaneTiling where
  tiles : List SquareTile
  max_usage : ℕ
  covers_plane : Bool
  no_overlap : Bool

/-- Theorem stating that it's possible to cover the plane with squares used up to 10 times each -/
theorem cover_plane_with_ten_usage :
  ∃ (t : PlaneTiling), t.max_usage = 10 ∧ t.covers_plane = true ∧ t.no_overlap = true :=
sorry

/-- Theorem stating that it's impossible to cover the plane with each square used only once -/
theorem cannot_cover_plane_with_single_usage :
  ¬ ∃ (t : PlaneTiling), t.max_usage = 1 ∧ t.covers_plane = true ∧ t.no_overlap = true :=
sorry

end NUMINAMATH_CALUDE_cover_plane_with_ten_usage_cannot_cover_plane_with_single_usage_l2645_264581


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l2645_264540

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) := by sorry

theorem negation_of_inequality :
  (¬∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l2645_264540


namespace NUMINAMATH_CALUDE_max_rectangle_area_l2645_264551

/-- The maximum area of a rectangle with perimeter 156 feet and natural number sides --/
theorem max_rectangle_area (l w : ℕ) : 
  (2 * (l + w) = 156) → l * w ≤ 1521 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l2645_264551


namespace NUMINAMATH_CALUDE_students_not_participating_l2645_264587

-- Define the sets and their cardinalities
def totalStudents : ℕ := 45
def volleyballParticipants : ℕ := 12
def trackFieldParticipants : ℕ := 20
def bothParticipants : ℕ := 6

-- Define the theorem
theorem students_not_participating : 
  totalStudents - volleyballParticipants - trackFieldParticipants + bothParticipants = 19 :=
by sorry

end NUMINAMATH_CALUDE_students_not_participating_l2645_264587


namespace NUMINAMATH_CALUDE_al_mass_percentage_in_mixture_l2645_264504

/-- The mass percentage of aluminum in a mixture of AlCl3, Al2(SO4)3, and Al(OH)3 --/
theorem al_mass_percentage_in_mixture (m_AlCl3 m_Al2SO4_3 m_AlOH3 : ℝ)
  (molar_mass_Al molar_mass_AlCl3 molar_mass_Al2SO4_3 molar_mass_AlOH3 : ℝ)
  (h1 : m_AlCl3 = 50)
  (h2 : m_Al2SO4_3 = 70)
  (h3 : m_AlOH3 = 40)
  (h4 : molar_mass_Al = 26.98)
  (h5 : molar_mass_AlCl3 = 133.33)
  (h6 : molar_mass_Al2SO4_3 = 342.17)
  (h7 : molar_mass_AlOH3 = 78.01) :
  let m_Al_AlCl3 := m_AlCl3 / molar_mass_AlCl3 * molar_mass_Al
  let m_Al_Al2SO4_3 := m_Al2SO4_3 / molar_mass_Al2SO4_3 * (2 * molar_mass_Al)
  let m_Al_AlOH3 := m_AlOH3 / molar_mass_AlOH3 * molar_mass_Al
  let total_m_Al := m_Al_AlCl3 + m_Al_Al2SO4_3 + m_Al_AlOH3
  let total_m_mixture := m_AlCl3 + m_Al2SO4_3 + m_AlOH3
  let mass_percentage := total_m_Al / total_m_mixture * 100
  ∃ ε > 0, |mass_percentage - 21.87| < ε :=
by sorry

end NUMINAMATH_CALUDE_al_mass_percentage_in_mixture_l2645_264504


namespace NUMINAMATH_CALUDE_smallest_solution_cubic_equation_l2645_264592

theorem smallest_solution_cubic_equation :
  ∃ (x : ℝ), x = 2/3 ∧ 24 * x^3 - 106 * x^2 + 116 * x - 70 = 0 ∧
  ∀ (y : ℝ), 24 * y^3 - 106 * y^2 + 116 * y - 70 = 0 → y ≥ 2/3 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_cubic_equation_l2645_264592


namespace NUMINAMATH_CALUDE_ellipse_range_l2645_264506

-- Define the set of real numbers m for which the equation represents an ellipse
def ellipse_set (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m + 1) = 1 ∧ 
  (m + 2 > 0) ∧ (-m - 1 > 0) ∧ (m + 2 ≠ -m - 1)

-- Define the target range for m
def target_range (m : ℝ) : Prop :=
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1)

-- Theorem statement
theorem ellipse_range :
  ∀ m : ℝ, ellipse_set m ↔ target_range m :=
sorry

end NUMINAMATH_CALUDE_ellipse_range_l2645_264506


namespace NUMINAMATH_CALUDE_polynomial_root_implication_l2645_264553

theorem polynomial_root_implication (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - 3 * Complex.I : ℂ) ^ 3 + a * (2 - 3 * Complex.I : ℂ) ^ 2 + 3 * (2 - 3 * Complex.I : ℂ) + b = 0 →
  a = -3/2 ∧ b = 65/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_implication_l2645_264553


namespace NUMINAMATH_CALUDE_concert_drive_l2645_264519

/-- Given a total distance and a distance already driven, calculate the remaining distance. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem: Given a total distance of 78 miles and having driven 32 miles, 
    the remaining distance to drive is 46 miles. -/
theorem concert_drive : remaining_distance 78 32 = 46 := by
  sorry

end NUMINAMATH_CALUDE_concert_drive_l2645_264519


namespace NUMINAMATH_CALUDE_sum_of_square_roots_geq_one_l2645_264522

theorem sum_of_square_roots_geq_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a / (a + 3 * b)) + Real.sqrt (b / (b + 3 * a)) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_geq_one_l2645_264522


namespace NUMINAMATH_CALUDE_bus_rows_count_l2645_264571

theorem bus_rows_count (total_capacity : ℕ) (row_capacity : ℕ) (h1 : total_capacity = 80) (h2 : row_capacity = 4) :
  total_capacity / row_capacity = 20 :=
by sorry

end NUMINAMATH_CALUDE_bus_rows_count_l2645_264571


namespace NUMINAMATH_CALUDE_fraction_problem_l2645_264567

theorem fraction_problem (x : ℚ) : 
  x / (4 * x - 4) = 3 / 7 → x = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2645_264567


namespace NUMINAMATH_CALUDE_floor_of_5_7_l2645_264518

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_of_5_7_l2645_264518


namespace NUMINAMATH_CALUDE_blue_face_probability_l2645_264516

theorem blue_face_probability (total_faces : ℕ) (blue_faces : ℕ)
  (h1 : total_faces = 12)
  (h2 : blue_faces = 4) :
  (blue_faces : ℚ) / total_faces = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_blue_face_probability_l2645_264516


namespace NUMINAMATH_CALUDE_total_distance_walked_l2645_264597

/-- Given a constant walking pace and duration, calculate the total distance walked. -/
theorem total_distance_walked (pace : ℝ) (duration : ℝ) (total_distance : ℝ) : 
  pace = 2 → duration = 8 → total_distance = pace * duration → total_distance = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l2645_264597


namespace NUMINAMATH_CALUDE_oreo_cheesecake_solution_l2645_264578

def oreo_cheesecake_problem (graham_boxes_bought : ℕ) (oreo_packets_bought : ℕ) 
  (graham_boxes_per_cake : ℕ) (graham_boxes_leftover : ℕ) : ℕ :=
  let cakes_made := (graham_boxes_bought - graham_boxes_leftover) / graham_boxes_per_cake
  oreo_packets_bought / cakes_made

theorem oreo_cheesecake_solution :
  oreo_cheesecake_problem 14 15 2 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_oreo_cheesecake_solution_l2645_264578


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2645_264527

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) : 
  total = 150 → badminton = 75 → tennis = 85 → neither = 15 → 
  badminton + tennis - (total - neither) = 25 := by
sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2645_264527


namespace NUMINAMATH_CALUDE_min_value_sum_product_l2645_264525

theorem min_value_sum_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * ((a + b)⁻¹ + (a + c)⁻¹ + (b + c)⁻¹) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l2645_264525


namespace NUMINAMATH_CALUDE_problem_solution_l2645_264530

theorem problem_solution (a b : ℝ) (h1 : a + b = 8) (h2 : a^2 * b^2 = 4) :
  (a^2 + b^2)/2 - a*b = 28 ∨ (a^2 + b^2)/2 - a*b = 36 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2645_264530


namespace NUMINAMATH_CALUDE_square_expression_l2645_264559

theorem square_expression (a b : ℝ) (square : ℝ) :
  square * (2 * a * b) = 4 * a^2 * b → square = 2 * a :=
by
  sorry

end NUMINAMATH_CALUDE_square_expression_l2645_264559


namespace NUMINAMATH_CALUDE_camp_participants_equality_l2645_264501

structure CampParticipants where
  mathOrange : ℕ
  mathPurple : ℕ
  physicsOrange : ℕ
  physicsPurple : ℕ

theorem camp_participants_equality (p : CampParticipants) 
  (h : p.physicsOrange = p.mathPurple) : 
  p.mathOrange + p.mathPurple = p.mathOrange + p.physicsOrange :=
by
  sorry

#check camp_participants_equality

end NUMINAMATH_CALUDE_camp_participants_equality_l2645_264501


namespace NUMINAMATH_CALUDE_inequality_proof_l2645_264568

theorem inequality_proof (a b c : ℝ) (h1 : c > 0) (h2 : a ≠ b) 
  (h3 : a^4 - 2019*a = c) (h4 : b^4 - 2019*b = c) : 
  -Real.sqrt c < a * b ∧ a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2645_264568


namespace NUMINAMATH_CALUDE_largest_n_with_triangle_property_l2645_264584

/-- A set of consecutive positive integers has the triangle property for all 9-element subsets -/
def has_triangle_property (s : Set ℕ) : Prop :=
  ∀ (x y z : ℕ), x ∈ s → y ∈ s → z ∈ s → x < y → y < z → z < x + y

/-- The set of consecutive positive integers from 6 to n -/
def consecutive_set (n : ℕ) : Set ℕ :=
  {x : ℕ | 6 ≤ x ∧ x ≤ n}

/-- The theorem stating that 224 is the largest possible value of n -/
theorem largest_n_with_triangle_property :
  ∀ n : ℕ, (has_triangle_property (consecutive_set n)) → n ≤ 224 :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_triangle_property_l2645_264584


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l2645_264564

theorem tangent_line_to_parabola (b : ℝ) :
  (∀ x y : ℝ, y = -2*x + b → y^2 = 8*x → (∀ ε > 0, ∃ δ > 0, ∀ x' y', 
    ((x' - x)^2 + (y' - y)^2 < δ^2) → (y' + 2*x' - b)^2 > 0 ∨ 
    ((y')^2 - 8*x')^2 > 0)) →
  b = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l2645_264564


namespace NUMINAMATH_CALUDE_sqrt_difference_simplification_l2645_264576

theorem sqrt_difference_simplification (x : ℝ) (h : -1 < x ∧ x < 0) :
  Real.sqrt (x^2) - Real.sqrt ((x+1)^2) = -2*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_simplification_l2645_264576


namespace NUMINAMATH_CALUDE_swap_numbers_l2645_264596

theorem swap_numbers (a b : ℕ) : 
  let c := b
  let b' := a
  let a' := c
  (a' = b ∧ b' = a) :=
by
  sorry

end NUMINAMATH_CALUDE_swap_numbers_l2645_264596


namespace NUMINAMATH_CALUDE_min_value_of_product_l2645_264550

theorem min_value_of_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 47 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_product_l2645_264550


namespace NUMINAMATH_CALUDE_triangle_properties_l2645_264547

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) (BD : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  c * Real.sin ((A + C) / 2) = b * Real.sin C ∧
  BD = 1 ∧
  b = Real.sqrt 3 ∧
  BD * (a * Real.sin C) = b * c * Real.sin (π / 2) →
  B = π / 3 ∧ 
  a + b + c = 3 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2645_264547


namespace NUMINAMATH_CALUDE_rain_given_northeast_wind_l2645_264557

/-- Probability of northeast winds blowing -/
def P_A : ℝ := 0.7

/-- Probability of rain -/
def P_B : ℝ := 0.8

/-- Probability of both northeast winds blowing and rain -/
def P_AB : ℝ := 0.65

/-- Theorem: The conditional probability of rain given northeast winds is 13/14 -/
theorem rain_given_northeast_wind :
  P_AB / P_A = 13 / 14 := by sorry

end NUMINAMATH_CALUDE_rain_given_northeast_wind_l2645_264557


namespace NUMINAMATH_CALUDE_running_percentage_is_fifty_percent_l2645_264579

/-- Represents a cricket batsman's score -/
structure BatsmanScore where
  total_runs : ℕ
  boundaries : ℕ
  sixes : ℕ

/-- Calculates the percentage of runs made by running between wickets -/
def runningPercentage (score : BatsmanScore) : ℚ :=
  let boundary_runs := 4 * score.boundaries
  let six_runs := 6 * score.sixes
  let running_runs := score.total_runs - (boundary_runs + six_runs)
  (running_runs : ℚ) / score.total_runs * 100

/-- Theorem: The percentage of runs made by running is 50% for the given score -/
theorem running_percentage_is_fifty_percent (score : BatsmanScore) 
    (h_total : score.total_runs = 120)
    (h_boundaries : score.boundaries = 3)
    (h_sixes : score.sixes = 8) : 
  runningPercentage score = 50 := by
  sorry

end NUMINAMATH_CALUDE_running_percentage_is_fifty_percent_l2645_264579


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2645_264588

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ k : ℝ, m^2 + m - 2 + (m^2 - 1) * Complex.I = k * Complex.I) ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2645_264588


namespace NUMINAMATH_CALUDE_janous_inequality_l2645_264552

theorem janous_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5 ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l2645_264552


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2645_264583

/-- Given vectors a and b, find k such that k*a - 2*b is perpendicular to a -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) : 
  a = (1, 1) → b = (2, -3) → 
  (k * a.1 - 2 * b.1, k * a.2 - 2 * b.2) • a = 0 → 
  k = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2645_264583


namespace NUMINAMATH_CALUDE_male_listeners_count_l2645_264570

/-- Represents the survey results of radio station XYZ -/
structure SurveyResults where
  total_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ
  total_non_listeners : ℕ

/-- Calculates the number of male listeners given the survey results -/
def male_listeners (survey : SurveyResults) : ℕ :=
  survey.total_listeners - survey.female_listeners

/-- Theorem stating that the number of male listeners is 85 -/
theorem male_listeners_count (survey : SurveyResults) 
  (h1 : survey.total_listeners = 160)
  (h2 : survey.female_listeners = 75) :
  male_listeners survey = 85 := by
  sorry

#eval male_listeners { total_listeners := 160, female_listeners := 75, male_non_listeners := 85, total_non_listeners := 180 }

end NUMINAMATH_CALUDE_male_listeners_count_l2645_264570


namespace NUMINAMATH_CALUDE_indeterminate_roots_of_related_quadratic_l2645_264562

/-- Given positive numbers a, b, c, and a quadratic equation with two equal real roots,
    the nature of the roots of a related quadratic equation cannot be determined. -/
theorem indeterminate_roots_of_related_quadratic
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_equal_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x)) :
  ∃ (r₁ r₂ : ℝ), (a + 1) * r₁^2 + (b + 2) * r₁ + (c + 1) = 0 ∧
                 (a + 1) * r₂^2 + (b + 2) * r₂ + (c + 1) = 0 ∧
                 (r₁ = r₂ ∨ r₁ ≠ r₂) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_roots_of_related_quadratic_l2645_264562


namespace NUMINAMATH_CALUDE_two_int_points_probability_l2645_264560

/-- Square S with diagonal endpoints (1/2, 3/2) and (-1/2, -3/2) -/
def S : Set (ℝ × ℝ) := sorry

/-- Random point v = (x,y) where 0 ≤ x ≤ 1006 and 0 ≤ y ≤ 1006 -/
def v : ℝ × ℝ := sorry

/-- T(v) is a translated copy of S centered at v -/
def T (v : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The probability that T(v) contains exactly two integer points in its interior -/
def prob_two_int_points : ℝ := sorry

theorem two_int_points_probability :
  prob_two_int_points = 2 / 25 := by sorry

end NUMINAMATH_CALUDE_two_int_points_probability_l2645_264560


namespace NUMINAMATH_CALUDE_a_less_than_b_plus_one_l2645_264595

theorem a_less_than_b_plus_one (a b : ℝ) (h : a < b) : a < b + 1 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_plus_one_l2645_264595


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2645_264545

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3*x + y = 5) 
  (h2 : x + 3*y = 6) : 
  10*x^2 + 13*x*y + 10*y^2 = 97 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2645_264545


namespace NUMINAMATH_CALUDE_ral_to_suri_age_ratio_l2645_264555

def suri_future_age : ℕ := 16
def years_to_future : ℕ := 3
def ral_current_age : ℕ := 26

def suri_current_age : ℕ := suri_future_age - years_to_future

theorem ral_to_suri_age_ratio :
  ral_current_age / suri_current_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_ral_to_suri_age_ratio_l2645_264555


namespace NUMINAMATH_CALUDE_vasyas_number_l2645_264529

theorem vasyas_number (n : ℕ) 
  (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : (n / 100) + (n % 10) = 1)
  (h3 : (n / 100) * ((n / 10) % 10) = 4) :
  n = 140 := by sorry

end NUMINAMATH_CALUDE_vasyas_number_l2645_264529


namespace NUMINAMATH_CALUDE_interest_rate_difference_l2645_264546

theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_diff : ℝ) 
  (h1 : principal = 2100) 
  (h2 : time = 3) 
  (h3 : interest_diff = 63) : 
  ∃ (rate1 rate2 : ℝ), rate2 - rate1 = 0.01 ∧ 
    principal * rate2 * time - principal * rate1 * time = interest_diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l2645_264546


namespace NUMINAMATH_CALUDE_rectangle_area_is_eight_l2645_264558

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*y^2 + 8*x - 16*y + 32 = 0

/-- The rectangle's height is twice the diameter of the circle -/
def rectangle_height_condition (height diameter : ℝ) : Prop :=
  height = 2 * diameter

/-- One pair of sides of the rectangle is parallel to the x-axis -/
def rectangle_orientation : Prop :=
  True  -- This condition is implicitly assumed and doesn't affect the calculation

/-- The area of the rectangle given its height and width -/
def rectangle_area (height width : ℝ) : ℝ :=
  height * width

/-- The main theorem stating that the area of the rectangle is 8 square units -/
theorem rectangle_area_is_eight :
  ∃ (x y height width diameter : ℝ),
    circle_equation x y ∧
    rectangle_height_condition height diameter ∧
    rectangle_orientation ∧
    rectangle_area height width = 8 :=
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_eight_l2645_264558


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_l2645_264523

theorem mean_equality_implies_z (z : ℚ) : 
  (7 + 10 + 23) / 3 = (18 + z) / 2 → z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_l2645_264523


namespace NUMINAMATH_CALUDE_discounted_milk_price_is_correct_l2645_264542

/-- The discounted price of a gallon of whole milk -/
def discounted_milk_price : ℝ := 2

/-- The normal price of a gallon of whole milk -/
def normal_milk_price : ℝ := 3

/-- The discount on a box of cereal -/
def cereal_discount : ℝ := 1

/-- The total savings when buying 3 gallons of milk and 5 boxes of cereal -/
def total_savings : ℝ := 8

/-- The number of gallons of milk bought -/
def milk_quantity : ℕ := 3

/-- The number of boxes of cereal bought -/
def cereal_quantity : ℕ := 5

theorem discounted_milk_price_is_correct :
  (milk_quantity : ℝ) * (normal_milk_price - discounted_milk_price) + 
  (cereal_quantity : ℝ) * cereal_discount = total_savings := by
  sorry

end NUMINAMATH_CALUDE_discounted_milk_price_is_correct_l2645_264542


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l2645_264503

theorem last_digit_of_sum (n : ℕ) : 
  (5^555 + 6^666 + 7^777) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_sum_l2645_264503


namespace NUMINAMATH_CALUDE_proportion_equality_l2645_264549

/-- Given a proportion x : 6 :: 2 : 0.19999999999999998, prove that x = 60 -/
theorem proportion_equality : 
  ∀ x : ℝ, (x / 6 = 2 / 0.19999999999999998) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l2645_264549


namespace NUMINAMATH_CALUDE_expression_evaluation_l2645_264509

theorem expression_evaluation (x : ℝ) : x = 2 → 2 * x^2 - 3 * x + 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2645_264509


namespace NUMINAMATH_CALUDE_negation_of_existence_power_of_two_less_than_1000_l2645_264577

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem power_of_two_less_than_1000 :
  (¬ ∃ n : ℕ, 2^n < 1000) ↔ (∀ n : ℕ, 2^n ≥ 1000) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_power_of_two_less_than_1000_l2645_264577


namespace NUMINAMATH_CALUDE_second_day_speed_l2645_264580

/-- Given a journey with specific conditions, prove the speed on the second day -/
theorem second_day_speed 
  (distance : ℝ) 
  (first_day_speed : ℝ) 
  (normal_time : ℝ) 
  (first_day_delay : ℝ) 
  (second_day_early : ℝ) 
  (h1 : distance = 60) 
  (h2 : first_day_speed = 10) 
  (h3 : normal_time = distance / first_day_speed) 
  (h4 : first_day_delay = 2) 
  (h5 : second_day_early = 1) : 
  distance / (normal_time - second_day_early) = 12 := by
sorry

end NUMINAMATH_CALUDE_second_day_speed_l2645_264580


namespace NUMINAMATH_CALUDE_count_happy_license_plates_l2645_264537

/-- The set of allowed letters on the license plate -/
def allowed_letters : Finset Char := {'А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х'}

/-- The set of consonant letters from the allowed letters -/
def consonant_letters : Finset Char := {'В', 'К', 'М', 'Н', 'Р', 'С', 'Т', 'Х'}

/-- The set of odd digits -/
def odd_digits : Finset Nat := {1, 3, 5, 7, 9}

/-- A license plate is a tuple of 3 letters and 3 digits -/
structure LicensePlate :=
  (letter1 : Char)
  (letter2 : Char)
  (digit1 : Nat)
  (digit2 : Nat)
  (digit3 : Nat)
  (letter3 : Char)

/-- A license plate is happy if the first two letters are consonants and the third digit is odd -/
def is_happy (plate : LicensePlate) : Prop :=
  plate.letter1 ∈ consonant_letters ∧
  plate.letter2 ∈ consonant_letters ∧
  plate.digit3 ∈ odd_digits

/-- The set of all valid license plates -/
def all_license_plates : Finset LicensePlate :=
  sorry

/-- The set of all happy license plates -/
def happy_license_plates : Finset LicensePlate :=
  sorry

/-- The main theorem: there are 384000 happy license plates -/
theorem count_happy_license_plates :
  Finset.card happy_license_plates = 384000 :=
sorry

end NUMINAMATH_CALUDE_count_happy_license_plates_l2645_264537


namespace NUMINAMATH_CALUDE_inscribed_circle_square_side_length_l2645_264565

theorem inscribed_circle_square_side_length 
  (circle_area : ℝ) 
  (h_area : circle_area = 3848.4510006474966) : 
  let square_side := 70
  let π := Real.pi
  circle_area = π * (square_side / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_side_length_l2645_264565


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l2645_264505

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r^n

theorem sum_of_fourth_and_fifth_terms (a₀ : ℝ) (r : ℝ) :
  (geometric_sequence a₀ r 5 = 4) →
  (geometric_sequence a₀ r 6 = 1) →
  (geometric_sequence a₀ r 2 = 256) →
  (geometric_sequence a₀ r 3 + geometric_sequence a₀ r 4 = 80) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l2645_264505


namespace NUMINAMATH_CALUDE_final_score_calculation_l2645_264574

theorem final_score_calculation (innovation_score comprehensive_score language_score : ℝ)
  (innovation_weight comprehensive_weight language_weight : ℝ) :
  innovation_score = 88 →
  comprehensive_score = 80 →
  language_score = 75 →
  innovation_weight = 5 →
  comprehensive_weight = 3 →
  language_weight = 2 →
  (innovation_score * innovation_weight + comprehensive_score * comprehensive_weight + language_score * language_weight) /
    (innovation_weight + comprehensive_weight + language_weight) = 83 := by
  sorry

end NUMINAMATH_CALUDE_final_score_calculation_l2645_264574


namespace NUMINAMATH_CALUDE_roots_sum_squares_l2645_264517

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 3*x - 2

-- Define the theorem
theorem roots_sum_squares (p q r : ℝ) : 
  f p = 0 → f q = 0 → f r = 0 → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_squares_l2645_264517


namespace NUMINAMATH_CALUDE_no_rational_solution_and_unique_perfect_square_l2645_264515

theorem no_rational_solution_and_unique_perfect_square :
  (∀ a : ℕ, ¬∃ x y z : ℚ, x^2 + y^2 + z^2 = 8 * a + 7) ∧
  (∀ n : ℕ, (∃ k : ℤ, 7^n + 8 = k^2) ↔ n = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_rational_solution_and_unique_perfect_square_l2645_264515


namespace NUMINAMATH_CALUDE_find_y_l2645_264502

theorem find_y (n x y : ℝ) : 
  (100 + 200 + n + x) / 4 = 250 ∧ 
  (n + 150 + 100 + x + y) / 5 = 200 → 
  y = 50 := by
sorry

end NUMINAMATH_CALUDE_find_y_l2645_264502


namespace NUMINAMATH_CALUDE_saltwater_concentration_l2645_264575

/-- The final concentration of saltwater in a cup after partial overflow and refilling -/
theorem saltwater_concentration (initial_concentration : ℝ) 
  (overflow_ratio : ℝ) (h1 : initial_concentration = 0.16) 
  (h2 : overflow_ratio = 0.1) : 
  initial_concentration * (1 - overflow_ratio) = 8/75 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_concentration_l2645_264575


namespace NUMINAMATH_CALUDE_zu_chongzhi_pi_calculation_l2645_264533

/-- Represents a historical mathematician -/
structure Mathematician where
  name : String
  calculating_pi : Bool
  decimal_places : ℕ
  father_of_pi : Bool

/-- The mathematician who calculated π to the 9th decimal place in ancient China -/
def ancient_chinese_pi_calculator : Mathematician :=
  { name := "Zu Chongzhi",
    calculating_pi := true,
    decimal_places := 9,
    father_of_pi := true }

/-- Theorem stating that Zu Chongzhi calculated π to the 9th decimal place and is known as the "Father of π" -/
theorem zu_chongzhi_pi_calculation :
  ∃ (m : Mathematician), m.calculating_pi ∧ m.decimal_places = 9 ∧ m.father_of_pi ∧ m.name = "Zu Chongzhi" :=
by
  sorry

end NUMINAMATH_CALUDE_zu_chongzhi_pi_calculation_l2645_264533


namespace NUMINAMATH_CALUDE_opposite_face_of_B_is_H_l2645_264556

-- Define a cube type
structure Cube where
  faces : Fin 6 → Char

-- Define the set of valid labels
def ValidLabels : Set Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Define a property that all faces have valid labels
def has_valid_labels (c : Cube) : Prop :=
  ∀ i : Fin 6, c.faces i ∈ ValidLabels

-- Define a property that all faces are unique
def has_unique_faces (c : Cube) : Prop :=
  ∀ i j : Fin 6, i ≠ j → c.faces i ≠ c.faces j

-- Define the theorem
theorem opposite_face_of_B_is_H (c : Cube) 
  (h1 : has_valid_labels c) 
  (h2 : has_unique_faces c) 
  (h3 : ∃ i : Fin 6, c.faces i = 'B') : 
  ∃ j : Fin 6, c.faces j = 'H' ∧ 
  (∀ k : Fin 6, c.faces k = 'B' → k.val + j.val = 5) :=
sorry

end NUMINAMATH_CALUDE_opposite_face_of_B_is_H_l2645_264556


namespace NUMINAMATH_CALUDE_misha_max_cities_l2645_264594

/-- The maximum number of cities Misha can visit -/
def max_cities_visited (n k : ℕ) : ℕ :=
  if k ≥ n - 3 then min (n - k) 2 else n - k

/-- Theorem stating the maximum number of cities Misha can visit -/
theorem misha_max_cities (n k : ℕ) (h1 : n ≥ 2) (h2 : k ≥ 1) :
  max_cities_visited n k = 
    if k ≥ n - 3 then min (n - k) 2 else n - k :=
by sorry

end NUMINAMATH_CALUDE_misha_max_cities_l2645_264594


namespace NUMINAMATH_CALUDE_brown_rabbit_hop_distance_l2645_264510

/-- Proves that given a white rabbit hopping 15 meters per minute and a total distance of 135 meters
    hopped by both rabbits in 5 minutes, the brown rabbit hops 12 meters per minute. -/
theorem brown_rabbit_hop_distance
  (white_rabbit_speed : ℝ)
  (total_distance : ℝ)
  (time : ℝ)
  (h1 : white_rabbit_speed = 15)
  (h2 : total_distance = 135)
  (h3 : time = 5) :
  (total_distance - white_rabbit_speed * time) / time = 12 := by
  sorry

#check brown_rabbit_hop_distance

end NUMINAMATH_CALUDE_brown_rabbit_hop_distance_l2645_264510


namespace NUMINAMATH_CALUDE_kyunghoon_descent_time_l2645_264561

/-- Proves that given the conditions of Kyunghoon's mountain hike, the time it took him to go down is 2 hours. -/
theorem kyunghoon_descent_time :
  ∀ (d : ℝ), -- distance up the mountain
  d > 0 →
  d / 3 + (d + 2) / 4 = 4 → -- total time equation
  (d + 2) / 4 = 2 -- time to go down
  := by sorry

end NUMINAMATH_CALUDE_kyunghoon_descent_time_l2645_264561


namespace NUMINAMATH_CALUDE_inscribed_squares_segment_product_l2645_264585

theorem inscribed_squares_segment_product :
  ∀ (small_area large_area : ℝ) (x : ℝ),
    small_area = 16 →
    large_area = 25 →
    x + 3*x = Real.sqrt large_area →
    x * (3*x) = 75/16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_segment_product_l2645_264585


namespace NUMINAMATH_CALUDE_barry_sotter_magic_l2645_264535

/-- The length factor after n days of growth -/
def length_factor (n : ℕ) : ℚ :=
  (n + 2 : ℚ) / 2

theorem barry_sotter_magic (n : ℕ) :
  length_factor n = 50 ↔ n = 98 := by
  sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_l2645_264535


namespace NUMINAMATH_CALUDE_same_heads_probability_is_three_sixteenths_l2645_264541

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes when tossing n pennies -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- The number of favorable outcomes where Ephraim gets the same number of heads as Keiko -/
def favorable_outcomes : ℕ := 6

/-- The probability of Ephraim getting the same number of heads as Keiko -/
def same_heads_probability : ℚ :=
  favorable_outcomes / (total_outcomes keiko_pennies * total_outcomes ephraim_pennies)

theorem same_heads_probability_is_three_sixteenths :
  same_heads_probability = 3 / 16 := by sorry

end NUMINAMATH_CALUDE_same_heads_probability_is_three_sixteenths_l2645_264541


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l2645_264554

theorem circle_line_intersection_range (k : ℝ) : 
  (∃ (a b : ℝ), (b = k * a - 2) ∧ 
    (∃ (x y : ℝ), (x^2 + y^2 + 8*x + 15 = 0) ∧ 
      ((x - a)^2 + (y - b)^2 = 1))) →
  -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l2645_264554


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2645_264532

theorem shaded_area_calculation (grid_height grid_width triangle_base triangle_height : ℝ) 
  (h1 : grid_height = 3)
  (h2 : grid_width = 15)
  (h3 : triangle_base = 5)
  (h4 : triangle_height = 3) :
  grid_height * grid_width - (1/2 * triangle_base * triangle_height) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2645_264532
