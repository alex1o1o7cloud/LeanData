import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l2437_243799

theorem system_solution (x y z : ℝ) : 
  (x + y + z = 6 ∧ x*y + y*z + z*x = 11 ∧ x*y*z = 6) ↔ 
  ((x = 1 ∧ y = 2 ∧ z = 3) ∨
   (x = 1 ∧ y = 3 ∧ z = 2) ∨
   (x = 2 ∧ y = 1 ∧ z = 3) ∨
   (x = 2 ∧ y = 3 ∧ z = 1) ∨
   (x = 3 ∧ y = 1 ∧ z = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2437_243799


namespace NUMINAMATH_CALUDE_distinct_roots_sum_abs_gt_six_l2437_243704

theorem distinct_roots_sum_abs_gt_six (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 9 = 0 → 
  r₂^2 + p*r₂ + 9 = 0 → 
  |r₁ + r₂| > 6 := by
sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_abs_gt_six_l2437_243704


namespace NUMINAMATH_CALUDE_toy_car_spending_l2437_243762

theorem toy_car_spending
  (A B C D E F G H : ℝ)
  (last_month : ℝ := A + B + C + D + E)
  (this_month_new : ℝ := F + G + H)
  (discount : ℝ := 0.2)
  (total_before_discount : ℝ := 2 * last_month + this_month_new)
  (total_after_discount : ℝ := (1 - discount) * total_before_discount) :
  total_after_discount = 1.6 * A + 1.6 * B + 1.6 * C + 1.6 * D + 1.6 * E + 0.8 * F + 0.8 * G + 0.8 * H :=
by sorry

end NUMINAMATH_CALUDE_toy_car_spending_l2437_243762


namespace NUMINAMATH_CALUDE_fruit_display_total_l2437_243729

/-- Represents the number of fruits on a display -/
structure FruitDisplay where
  bananas : ℕ
  oranges : ℕ
  apples : ℕ
  lemons : ℕ

/-- Calculates the total number of fruits on the display -/
def totalFruits (d : FruitDisplay) : ℕ :=
  d.bananas + d.oranges + d.apples + d.lemons

/-- Theorem stating the total number of fruits on the display -/
theorem fruit_display_total (d : FruitDisplay) 
  (h1 : d.bananas = 5)
  (h2 : d.oranges = 2 * d.bananas)
  (h3 : d.apples = 2 * d.oranges)
  (h4 : d.lemons = (d.apples + d.bananas) / 2) :
  totalFruits d = 47 := by
  sorry

#eval totalFruits { bananas := 5, oranges := 10, apples := 20, lemons := 12 }

end NUMINAMATH_CALUDE_fruit_display_total_l2437_243729


namespace NUMINAMATH_CALUDE_arithmetic_progression_equality_l2437_243746

/-- An arithmetic progression with first term and difference as natural numbers -/
structure ArithmeticProgression :=
  (first : ℕ)
  (diff : ℕ)
  (coprime : Nat.Coprime first diff)

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℕ :=
  ap.first + (n - 1) * ap.diff

theorem arithmetic_progression_equality (ap1 ap2 : ArithmeticProgression) :
  (∀ n : ℕ, 
    (ArithmeticProgression.nthTerm ap1 n ^ 2 + ArithmeticProgression.nthTerm ap1 (n + 1) ^ 2) *
    (ArithmeticProgression.nthTerm ap2 n ^ 2 + ArithmeticProgression.nthTerm ap2 (n + 1) ^ 2) = m ^ 2 ∨
    (ArithmeticProgression.nthTerm ap1 n ^ 2 + ArithmeticProgression.nthTerm ap2 n ^ 2) *
    (ArithmeticProgression.nthTerm ap1 (n + 1) ^ 2 + ArithmeticProgression.nthTerm ap2 (n + 1) ^ 2) = k ^ 2) →
  ∀ n : ℕ, ArithmeticProgression.nthTerm ap1 n = ArithmeticProgression.nthTerm ap2 n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equality_l2437_243746


namespace NUMINAMATH_CALUDE_prism_18_edges_8_faces_l2437_243715

/-- A prism is a polyhedron with two congruent parallel faces (bases) and other faces (lateral faces) that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  2 + (p.edges / 3)

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ p : Prism, p.edges = 18 → num_faces p = 8 := by
  sorry


end NUMINAMATH_CALUDE_prism_18_edges_8_faces_l2437_243715


namespace NUMINAMATH_CALUDE_area_of_specific_triangle_l2437_243738

/-- The line equation ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A triangle bounded by the x-axis, y-axis, and a line --/
structure AxisAlignedTriangle where
  boundingLine : Line

/-- The area of an axis-aligned triangle --/
def areaOfAxisAlignedTriangle (t : AxisAlignedTriangle) : ℝ :=
  sorry

theorem area_of_specific_triangle : 
  let t : AxisAlignedTriangle := { boundingLine := { a := 3, b := 4, c := 12 } }
  areaOfAxisAlignedTriangle t = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_triangle_l2437_243738


namespace NUMINAMATH_CALUDE_min_route_length_5x5_city_l2437_243764

/-- Represents a square grid city -/
structure City where
  size : ℕ
  streets : ℕ

/-- Calculates the minimum route length for an Eulerian circuit in the city -/
def minRouteLength (c : City) : ℕ :=
  2 * c.streets + 8

theorem min_route_length_5x5_city :
  ∃ (c : City), c.size = 5 ∧ c.streets = 30 ∧ minRouteLength c = 68 :=
by sorry

end NUMINAMATH_CALUDE_min_route_length_5x5_city_l2437_243764


namespace NUMINAMATH_CALUDE_area_difference_square_rectangle_l2437_243775

theorem area_difference_square_rectangle :
  ∀ (square_side : ℝ) (rect_length rect_width : ℝ),
  square_side * 4 = 52 →
  rect_length = 15 →
  rect_length * 2 + rect_width * 2 = 52 →
  square_side * square_side - rect_length * rect_width = 4 := by
sorry

end NUMINAMATH_CALUDE_area_difference_square_rectangle_l2437_243775


namespace NUMINAMATH_CALUDE_complex_quadrant_l2437_243767

theorem complex_quadrant (z : ℂ) (h : (z - 1) * Complex.I = 1 + 2 * Complex.I) :
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l2437_243767


namespace NUMINAMATH_CALUDE_min_value_of_a_l2437_243744

theorem min_value_of_a (a b c : ℝ) : 
  a + b + c = 3 → 
  a ≥ b → 
  b ≥ c → 
  ∃ x : ℝ, a * x^2 + b * x + c = 0 → 
  a ≥ 4/3 ∧ ∀ a' : ℝ, (∃ b' c' : ℝ, 
    a' + b' + c' = 3 ∧ 
    a' ≥ b' ∧ 
    b' ≥ c' ∧ 
    (∃ x : ℝ, a' * x^2 + b' * x + c' = 0)) → 
  a' ≥ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2437_243744


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2437_243792

def U : Set ℝ := {x | x < 2}
def A : Set ℝ := {x | x^2 < x}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2437_243792


namespace NUMINAMATH_CALUDE_green_peaches_count_l2437_243706

/-- The number of baskets -/
def num_baskets : ℕ := 7

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 2

/-- The total number of green peaches -/
def total_green_peaches : ℕ := num_baskets * green_peaches_per_basket

theorem green_peaches_count : total_green_peaches = 14 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2437_243706


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2437_243758

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B : A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2437_243758


namespace NUMINAMATH_CALUDE_factorization_equality_l2437_243784

theorem factorization_equality (x y : ℝ) : 
  (x + y)^2 - 14*(x + y) + 49 = (x + y - 7)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l2437_243784


namespace NUMINAMATH_CALUDE_function_identity_l2437_243732

def f_condition (f : ℝ → ℝ) : Prop :=
  f 0 = 1 ∧ ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2

theorem function_identity (f : ℝ → ℝ) (h : f_condition f) :
  ∀ x : ℝ, f x = x + 1 :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l2437_243732


namespace NUMINAMATH_CALUDE_perpendicular_to_plane_false_l2437_243750

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- line is subset of plane
variable (perp : Line → Line → Prop)     -- line is perpendicular to line
variable (perpPlane : Line → Plane → Prop)  -- line is perpendicular to plane

-- State the theorem
theorem perpendicular_to_plane_false
  (l m n : Line) (α : Plane)
  (h1 : subset m α)
  (h2 : subset n α)
  (h3 : perp l m)
  (h4 : perp l n) :
  ¬ (perpPlane l α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_plane_false_l2437_243750


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2437_243751

theorem quadratic_factorization (a b : ℕ) : 
  (∀ x, x^2 - 20*x + 96 = (x - a)*(x - b)) →
  a > b →
  4*b - a = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2437_243751


namespace NUMINAMATH_CALUDE_bills_final_money_bills_final_money_is_3180_l2437_243725

/-- Calculates Bill's final amount of money after Frank and Bill's pizza purchase --/
theorem bills_final_money (initial_money : ℝ) (pizza_cost : ℝ) (num_pizzas : ℕ) 
  (topping1_cost : ℝ) (topping2_cost : ℝ) (discount_rate : ℝ) (bills_initial_money : ℝ) : ℝ :=
  let total_pizza_cost := pizza_cost * num_pizzas
  let total_topping_cost := (topping1_cost + topping2_cost) * num_pizzas
  let total_cost_before_discount := total_pizza_cost + total_topping_cost
  let discount := discount_rate * total_pizza_cost
  let final_cost := total_cost_before_discount - discount
  let remaining_money := initial_money - final_cost
  bills_initial_money + remaining_money

/-- Proves that Bill's final amount of money is $31.80 --/
theorem bills_final_money_is_3180 : 
  bills_final_money 42 11 3 1.5 2 0.1 30 = 31.80 := by
  sorry

end NUMINAMATH_CALUDE_bills_final_money_bills_final_money_is_3180_l2437_243725


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l2437_243766

theorem smallest_solution_abs_equation :
  let f : ℝ → ℝ := λ x => x * |x| - (2 * x^2 + 3 * x + 1)
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y ∧ x = (3 + Real.sqrt 13) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l2437_243766


namespace NUMINAMATH_CALUDE_valid_numbers_l2437_243723

def is_valid_number (N : ℕ) : Prop :=
  ∃ (a b : ℕ) (q : ℚ),
    N = 10 * a + b ∧
    0 < a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    b = a * q ∧
    N = 3 * (a * q^2)

theorem valid_numbers :
  {N : ℕ | is_valid_number N} = {12, 24, 36, 48} :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l2437_243723


namespace NUMINAMATH_CALUDE_no_adjacent_same_probability_correct_probability_between_zero_and_one_l2437_243722

def number_of_people : ℕ := 6
def die_sides : ℕ := 6

/-- The probability of no two adjacent people rolling the same number on a six-sided die 
    when six people are sitting around a circular table. -/
def no_adjacent_same_probability : ℚ :=
  625 / 1944

/-- Theorem stating that the calculated probability is correct. -/
theorem no_adjacent_same_probability_correct : 
  no_adjacent_same_probability = 625 / 1944 := by
  sorry

/-- Theorem stating that the probability is between 0 and 1. -/
theorem probability_between_zero_and_one :
  0 ≤ no_adjacent_same_probability ∧ no_adjacent_same_probability ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_same_probability_correct_probability_between_zero_and_one_l2437_243722


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2437_243741

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := y^2 / a^2 - x^2 / 3 = 1

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Define asymptotes
def asymptotes (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x

-- Define the trajectory of midpoint M
def trajectory (x y : ℝ) : Prop := x^2 / 75 + 3 * y^2 / 25 = 1

-- Define the theorem
theorem hyperbola_properties :
  ∀ (a : ℝ), 
  (∃ (x y : ℝ), hyperbola x y a) →
  eccentricity 2 →
  (∀ (x y : ℝ), asymptotes x y) ∧
  (∀ (x y : ℝ), trajectory x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2437_243741


namespace NUMINAMATH_CALUDE_angle_conversions_correct_l2437_243753

theorem angle_conversions_correct :
  let deg_to_rad (d : ℝ) := d * (π / 180)
  let rad_to_deg (r : ℝ) := r * (180 / π)
  (deg_to_rad 60 = π / 3) ∧
  (rad_to_deg (-10 * π / 3) = -600) ∧
  (deg_to_rad (-150) = -5 * π / 6) ∧
  (rad_to_deg (π / 12) = 15) := by
  sorry

end NUMINAMATH_CALUDE_angle_conversions_correct_l2437_243753


namespace NUMINAMATH_CALUDE_stratified_sample_o_blood_type_l2437_243794

/-- Calculates the number of students with blood type O in a stratified sample -/
def stratifiedSampleO (totalStudents : ℕ) (oTypeStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  (oTypeStudents * sampleSize) / totalStudents

/-- Theorem: In a stratified sample of 40 students from a population of 500 students, 
    where 200 students have blood type O, the number of students with blood type O 
    in the sample should be 16. -/
theorem stratified_sample_o_blood_type 
  (totalStudents : ℕ) 
  (oTypeStudents : ℕ) 
  (sampleSize : ℕ) 
  (h1 : totalStudents = 500) 
  (h2 : oTypeStudents = 200) 
  (h3 : sampleSize = 40) :
  stratifiedSampleO totalStudents oTypeStudents sampleSize = 16 := by
  sorry

#eval stratifiedSampleO 500 200 40

end NUMINAMATH_CALUDE_stratified_sample_o_blood_type_l2437_243794


namespace NUMINAMATH_CALUDE_minimum_garden_width_minimum_garden_width_is_ten_l2437_243759

theorem minimum_garden_width (w : ℝ) : w > 0 → w * (w + 10) ≥ 150 → w ≥ 10 := by
  sorry

theorem minimum_garden_width_is_ten : ∃ w : ℝ, w > 0 ∧ w * (w + 10) ≥ 150 ∧ ∀ x : ℝ, x > 0 → x * (x + 10) ≥ 150 → x ≥ w := by
  sorry

end NUMINAMATH_CALUDE_minimum_garden_width_minimum_garden_width_is_ten_l2437_243759


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2437_243783

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 8)
  let b : ℝ × ℝ := (x, 4)
  parallel a b → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2437_243783


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2437_243763

theorem rectangular_field_area (m : ℕ) : 
  (3 * m + 8) * (m - 3) = 76 → m = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2437_243763


namespace NUMINAMATH_CALUDE_box_area_is_679_l2437_243712

/-- The surface area of the interior of a box formed by removing square corners from a rectangular sheet --/
def box_interior_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Theorem stating that the surface area of the interior of the box is 679 square units --/
theorem box_area_is_679 :
  box_interior_area 25 35 7 = 679 :=
by sorry

end NUMINAMATH_CALUDE_box_area_is_679_l2437_243712


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2437_243778

theorem quadratic_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≠ 0) → -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2437_243778


namespace NUMINAMATH_CALUDE_final_amount_calculation_l2437_243720

/-- Calculates the final amount paid after applying a discount based on complete hundreds spent. -/
theorem final_amount_calculation (purchase_amount : ℕ) (discount_per_hundred : ℕ) : 
  purchase_amount = 250 ∧ discount_per_hundred = 10 →
  purchase_amount - (purchase_amount / 100) * discount_per_hundred = 230 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_calculation_l2437_243720


namespace NUMINAMATH_CALUDE_most_colored_pencils_l2437_243772

theorem most_colored_pencils (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow :=
by
  sorry

end NUMINAMATH_CALUDE_most_colored_pencils_l2437_243772


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l2437_243770

theorem no_simultaneous_squares (n : ℕ+) : ¬∃ (x y : ℕ+), (n + 1 : ℕ) = x^2 ∧ (4*n + 1 : ℕ) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l2437_243770


namespace NUMINAMATH_CALUDE_expression_evaluation_l2437_243702

/-- Given a = 3, b = 2, and c = 1, prove that (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 -/
theorem expression_evaluation (a b c : ℕ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2437_243702


namespace NUMINAMATH_CALUDE_binomial_difference_divisibility_l2437_243777

theorem binomial_difference_divisibility (k : ℕ) (h : k ≥ 2) :
  ∃ n : ℕ, (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) = n * 2^(3*k) ∧
           (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) % 2^(3*k+1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_binomial_difference_divisibility_l2437_243777


namespace NUMINAMATH_CALUDE_factor_expression_l2437_243721

theorem factor_expression (x : ℝ) : 5*x*(x+2) + 11*(x+2) = (x+2)*(5*x+11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2437_243721


namespace NUMINAMATH_CALUDE_problem_solution_l2437_243755

theorem problem_solution (a b : ℝ) : (a + b)^2 + Real.sqrt (2 * b - 4) = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2437_243755


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2437_243754

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = -9 ∧ x₂ = -3 ∧ 
  (∀ x : ℝ, x^2 + 12*x + 27 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2437_243754


namespace NUMINAMATH_CALUDE_equal_triangle_areas_l2437_243779

-- Define the points
variable (A B C D E F K L : Point)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the intersection points E and F
def E_is_intersection (A B C D E : Point) : Prop := sorry
def F_is_intersection (A B C D F : Point) : Prop := sorry

-- Define K and L as midpoints of diagonals
def K_is_midpoint (A C K : Point) : Prop := sorry
def L_is_midpoint (B D L : Point) : Prop := sorry

-- Define the area of a triangle
def area (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem equal_triangle_areas 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : E_is_intersection A B C D E)
  (h3 : F_is_intersection A B C D F)
  (h4 : K_is_midpoint A C K)
  (h5 : L_is_midpoint B D L) :
  area E K L = area F K L := by sorry

end NUMINAMATH_CALUDE_equal_triangle_areas_l2437_243779


namespace NUMINAMATH_CALUDE_surface_area_of_cube_with_holes_l2437_243705

/-- Represents a cube with square holes cut through each face -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the surface area of a cube with square holes cut through each face -/
def surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length^2
  let hole_area := 6 * cube.hole_side_length^2
  let inner_surface_area := 6 * 4 * cube.hole_side_length^2
  original_surface_area - hole_area + inner_surface_area

/-- Theorem stating that the surface area of the specified cube with holes is 168 square meters -/
theorem surface_area_of_cube_with_holes :
  let cube := CubeWithHoles.mk 4 2
  surface_area cube = 168 := by
  sorry


end NUMINAMATH_CALUDE_surface_area_of_cube_with_holes_l2437_243705


namespace NUMINAMATH_CALUDE_club_equation_solution_l2437_243774

def club (A B : ℝ) : ℝ := 3 * A + 2 * B + 5

theorem club_equation_solution :
  ∃! A : ℝ, club A 4 = 58 ∧ A = 15 := by sorry

end NUMINAMATH_CALUDE_club_equation_solution_l2437_243774


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2437_243769

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2437_243769


namespace NUMINAMATH_CALUDE_dog_count_l2437_243768

theorem dog_count (num_puppies : ℕ) (dog_meal_frequency : ℕ) (dog_meal_amount : ℕ) (total_food : ℕ) : 
  num_puppies = 4 →
  dog_meal_frequency = 3 →
  dog_meal_amount = 4 →
  total_food = 108 →
  (∃ (num_dogs : ℕ),
    num_dogs * (dog_meal_frequency * dog_meal_amount) + 
    num_puppies * (3 * dog_meal_frequency) * (dog_meal_amount / 2) = total_food ∧
    num_dogs = 3) := by
  sorry

end NUMINAMATH_CALUDE_dog_count_l2437_243768


namespace NUMINAMATH_CALUDE_cylinder_surface_area_and_volume_l2437_243727

/-- Given a cylinder with cross-sectional area M and axial section area N,
    prove its surface area and volume. -/
theorem cylinder_surface_area_and_volume (M N : ℝ) (M_pos : M > 0) (N_pos : N > 0) :
  ∃ (surface_area volume : ℝ),
    surface_area = N * Real.pi + 2 * M ∧
    volume = (N / 2) * Real.sqrt (M * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_and_volume_l2437_243727


namespace NUMINAMATH_CALUDE_machine_x_production_rate_l2437_243765

/-- Production rates and times for two machines -/
structure MachineProduction where
  x_rate : ℝ  -- Production rate of Machine X (widgets per hour)
  y_rate : ℝ  -- Production rate of Machine Y (widgets per hour)
  x_time : ℝ  -- Time taken by Machine X to produce 1080 widgets
  y_time : ℝ  -- Time taken by Machine Y to produce 1080 widgets

/-- Theorem stating the production rate of Machine X -/
theorem machine_x_production_rate (m : MachineProduction) :
  m.x_rate = 18 :=
by
  have h1 : m.x_time = m.y_time + 10 := by sorry
  have h2 : m.y_rate = 1.2 * m.x_rate := by sorry
  have h3 : m.x_rate * m.x_time = 1080 := by sorry
  have h4 : m.y_rate * m.y_time = 1080 := by sorry
  sorry

end NUMINAMATH_CALUDE_machine_x_production_rate_l2437_243765


namespace NUMINAMATH_CALUDE_percentage_problem_l2437_243773

theorem percentage_problem (x y : ℝ) (P : ℝ) 
  (h1 : 0.7 * (x - y) = (P / 100) * (x + y))
  (h2 : y = 0.4 * x) : 
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2437_243773


namespace NUMINAMATH_CALUDE_max_value_abc_l2437_243791

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  a^2 * b^3 * c ≤ 27/16 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_l2437_243791


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l2437_243749

/-- Given the initial length of Isabella's hair, the amount it grew, and the final length,
    prove that the initial length plus the growth equals the final length. -/
theorem isabellas_hair_growth (initial_length growth final_length : ℝ) 
    (h1 : growth = 6)
    (h2 : final_length = 24)
    (h3 : initial_length + growth = final_length) : 
  initial_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l2437_243749


namespace NUMINAMATH_CALUDE_train_speed_fraction_l2437_243713

theorem train_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 12 → delay = 9 → 
  (usual_time / (usual_time + delay)) = (4 : ℝ) / 7 := by
sorry

end NUMINAMATH_CALUDE_train_speed_fraction_l2437_243713


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l2437_243771

theorem stratified_sampling_female_count 
  (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) 
  (h_male : total_male = 810)
  (h_female : total_female = 540)
  (h_sample : sample_size = 200) :
  (sample_size : ℚ) * total_female / (total_male + total_female) = 80 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l2437_243771


namespace NUMINAMATH_CALUDE_wen_family_theater_cost_l2437_243719

/-- Represents the cost of tickets for a family theater outing -/
def theater_cost (regular_price : ℚ) : ℚ :=
  let senior_price := regular_price * (1 - 0.2)
  let child_price := regular_price * (1 - 0.4)
  let total_before_discount := 2 * senior_price + 2 * regular_price + 2 * child_price
  total_before_discount * (1 - 0.1)

/-- Theorem stating the total cost for the Wen family's theater tickets -/
theorem wen_family_theater_cost :
  ∃ (regular_price : ℚ),
    (regular_price * (1 - 0.2) = 7.5) ∧
    (theater_cost regular_price = 40.5) := by
  sorry


end NUMINAMATH_CALUDE_wen_family_theater_cost_l2437_243719


namespace NUMINAMATH_CALUDE_inequality_proof_l2437_243703

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  Real.sqrt (3 * x^2 + x * y) + Real.sqrt (3 * y^2 + y * z) + Real.sqrt (3 * z^2 + z * x) ≤ 2 * (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2437_243703


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l2437_243734

theorem greatest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, 8 - 3 * (2 * y + 1) > 26 → y ≤ x) ∧ (8 - 3 * (2 * x + 1) > 26) ↔ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l2437_243734


namespace NUMINAMATH_CALUDE_a_investment_l2437_243718

/-- Represents the investment scenario and proves A's investment amount -/
theorem a_investment (a_time b_time : ℕ) (b_investment total_profit a_share : ℚ) :
  a_time = 12 →
  b_time = 6 →
  b_investment = 200 →
  total_profit = 100 →
  a_share = 75 →
  ∃ (a_investment : ℚ),
    a_investment * a_time / (a_investment * a_time + b_investment * b_time) * total_profit = a_share ∧
    a_investment = 300 := by
  sorry


end NUMINAMATH_CALUDE_a_investment_l2437_243718


namespace NUMINAMATH_CALUDE_school_pairing_fraction_l2437_243731

theorem school_pairing_fraction :
  ∀ (s n : ℕ), 
    s > 0 → n > 0 →
    (n : ℚ) / 4 = (s : ℚ) / 3 →
    ((s : ℚ) / 3 + (n : ℚ) / 4) / ((s : ℚ) + (n : ℚ)) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_school_pairing_fraction_l2437_243731


namespace NUMINAMATH_CALUDE_junior_score_l2437_243747

theorem junior_score (n : ℝ) (junior_score : ℝ) :
  n > 0 →
  0.15 * n * junior_score + 0.85 * n * 87 = n * 88 →
  junior_score = 94 := by
  sorry

end NUMINAMATH_CALUDE_junior_score_l2437_243747


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l2437_243782

theorem smallest_gcd_bc (a b c : ℕ+) (hab : Nat.gcd a b = 210) (hac : Nat.gcd a c = 770) :
  (∀ d : ℕ+, ∃ a' b' c' : ℕ+, Nat.gcd a' b' = 210 ∧ Nat.gcd a' c' = 770 ∧ Nat.gcd b' c' = d) →
  10 ≤ Nat.gcd b c :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l2437_243782


namespace NUMINAMATH_CALUDE_bridge_crossing_time_l2437_243742

/-- Proves that a man walking at 9 km/hr takes 15 minutes to cross a bridge of 2250 meters in length -/
theorem bridge_crossing_time (walking_speed : ℝ) (bridge_length : ℝ) :
  walking_speed = 9 →
  bridge_length = 2250 →
  (bridge_length / (walking_speed * 1000 / 60)) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_crossing_time_l2437_243742


namespace NUMINAMATH_CALUDE_bird_population_theorem_l2437_243788

theorem bird_population_theorem (total : ℝ) (total_pos : total > 0) : 
  let hawks := 0.3 * total
  let non_hawks := total - hawks
  let paddyfield_warblers := 0.4 * non_hawks
  let kingfishers := 0.25 * paddyfield_warblers
  let other_birds := total - (hawks + paddyfield_warblers + kingfishers)
  (other_birds / total) * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_bird_population_theorem_l2437_243788


namespace NUMINAMATH_CALUDE_q_at_zero_l2437_243709

-- Define polynomials p, q, and r
variable (p q r : ℝ[X])

-- Define the relationship between p, q, and r
axiom poly_product : r = p * q

-- Define the constant terms of p and r
axiom p_constant : p.coeff 0 = 5
axiom r_constant : r.coeff 0 = -10

-- Theorem to prove
theorem q_at_zero : q.eval 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_q_at_zero_l2437_243709


namespace NUMINAMATH_CALUDE_max_value_product_sum_l2437_243790

theorem max_value_product_sum (X Y Z : ℕ) (h : X + Y + Z = 15) :
  (∀ A B C : ℕ, A + B + C = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ A * B * C + A * B + B * C + C * A) →
  X * Y * Z + X * Y + Y * Z + Z * X = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l2437_243790


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l2437_243789

theorem largest_integer_inequality : 
  ∀ y : ℤ, y ≤ 0 ↔ (y : ℚ) / 4 + 3 / 7 < 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l2437_243789


namespace NUMINAMATH_CALUDE_clever_calculation_l2437_243786

theorem clever_calculation :
  (46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 = 56.056) ∧
  (101 * 92 - 92 = 9200) ∧
  (36000 / 125 / 8 = 36) := by
  sorry

end NUMINAMATH_CALUDE_clever_calculation_l2437_243786


namespace NUMINAMATH_CALUDE_car_dealership_count_l2437_243796

theorem car_dealership_count :
  ∀ (total_cars : ℕ),
    (total_cars : ℝ) * 0.6 * 0.6 = 216 →
    total_cars = 600 := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_count_l2437_243796


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2437_243745

/-- Proves that given the conditions of the age problem, the ratio of Michael's age to Monica's age is 3:5 -/
theorem age_ratio_problem (patrick_age michael_age monica_age : ℕ) : 
  patrick_age * 5 = michael_age * 3 →  -- Patrick and Michael's ages are in ratio 3:5
  patrick_age + michael_age + monica_age = 196 →  -- Sum of ages is 196
  monica_age - patrick_age = 64 →  -- Difference between Monica's and Patrick's ages is 64
  michael_age * 5 = monica_age * 3  -- Conclusion: Michael and Monica's ages are in ratio 3:5
:= by sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2437_243745


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l2437_243752

theorem solution_satisfies_equations :
  ∃ (x y : ℝ), 3 * x - 7 * y = 2 ∧ 4 * y - x = 6 ∧ x = 10 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l2437_243752


namespace NUMINAMATH_CALUDE_three_positions_from_six_people_l2437_243717

/-- The number of ways to select 3 distinct positions from a group of 6 people. -/
def select_positions (n : ℕ) (k : ℕ) : ℕ :=
  n * (n - 1) * (n - 2)

/-- Theorem stating that selecting 3 distinct positions from 6 people results in 120 ways. -/
theorem three_positions_from_six_people :
  select_positions 6 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_three_positions_from_six_people_l2437_243717


namespace NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2437_243700

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of decks used -/
def num_decks : ℕ := 2

/-- The total number of cards in the combined deck -/
def total_cards : ℕ := standard_deck_size * num_decks

/-- The number of red cards in the combined deck -/
def red_cards : ℕ := standard_deck_size

/-- The expected number of pairs of adjacent red cards -/
def expected_red_pairs : ℚ := 2652 / 103

theorem expected_adjacent_red_pairs :
  let p := red_cards / total_cards
  expected_red_pairs = red_cards * (red_cards - 1) / (total_cards - 1) := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2437_243700


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2437_243736

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 1 > 0}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- State the theorem
theorem intersection_complement_equality :
  (Aᶜ ∩ B) = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2437_243736


namespace NUMINAMATH_CALUDE_A_value_l2437_243760

theorem A_value (a : ℝ) (h : a * (a + 2) = 8 ∨ a^2 + a = 8 - a) :
  2 / (a^2 - 4) - 1 / (a * (a - 2)) = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_A_value_l2437_243760


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l2437_243780

theorem least_positive_angle_theorem : ∃ θ : Real,
  θ > 0 ∧
  θ < 360 ∧
  Real.cos (15 * π / 180) = Real.sin (45 * π / 180) + Real.sin θ ∧
  θ = 195 * π / 180 ∧
  ∀ φ, 0 < φ ∧ φ < θ → Real.cos (15 * π / 180) ≠ Real.sin (45 * π / 180) + Real.sin φ :=
by sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l2437_243780


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2437_243739

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2437_243739


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2437_243716

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2 * Real.sqrt 3) :
  ((a^2 + b^2) / (2 * a) - b) * (a / (a - b)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2437_243716


namespace NUMINAMATH_CALUDE_seokgi_candies_l2437_243797

theorem seokgi_candies :
  ∀ (original : ℕ),
  (original : ℚ) * (1/2 : ℚ) * (2/3 : ℚ) = 12 →
  original = 36 := by
  sorry

end NUMINAMATH_CALUDE_seokgi_candies_l2437_243797


namespace NUMINAMATH_CALUDE_basketball_tryouts_l2437_243701

theorem basketball_tryouts (girls boys callback : ℕ) 
  (h1 : girls = 17)
  (h2 : boys = 32)
  (h3 : callback = 10) :
  girls + boys - callback = 39 := by
sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l2437_243701


namespace NUMINAMATH_CALUDE_choose_three_from_five_l2437_243737

theorem choose_three_from_five : Nat.choose 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_five_l2437_243737


namespace NUMINAMATH_CALUDE_egyptian_fraction_representation_l2437_243707

theorem egyptian_fraction_representation : ∃! (b₂ b₃ b₄ b₅ b₆ b₇ : ℕ),
  (17 : ℚ) / 23 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11 := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_representation_l2437_243707


namespace NUMINAMATH_CALUDE_three_zeros_properties_l2437_243795

variable (a : ℝ) (x₁ x₂ x₃ : ℝ)

def f (x : ℝ) := a * (2 * x - 1) * abs (x + 1) - 2 * x - 1

theorem three_zeros_properties 
  (h_zeros : f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0)
  (h_order : x₁ < x₂ ∧ x₂ < x₃) :
  (1 / a < x₃ ∧ x₃ < 1 / a + 1 / x₃) ∧ a * (x₂ - x₁) < 1 := by
  sorry

end NUMINAMATH_CALUDE_three_zeros_properties_l2437_243795


namespace NUMINAMATH_CALUDE_expected_wins_equal_l2437_243733

/-- The total number of balls in the lottery box -/
def total_balls : ℕ := 8

/-- The number of red balls in the lottery box -/
def red_balls : ℕ := 4

/-- The number of black balls in the lottery box -/
def black_balls : ℕ := 4

/-- The number of draws made -/
def num_draws : ℕ := 2

/-- Represents the outcome of a single lottery draw -/
inductive DrawResult
| Red
| Black

/-- Represents the result of two draws -/
inductive TwoDrawResult
| Win  -- Two balls of the same color
| Lose -- Two balls of different colors

/-- The probability of winning in a single draw with replacement -/
def prob_win_with_replacement : ℚ :=
  (red_balls.choose 2 + black_balls.choose 2) / total_balls.choose 2

/-- The expected number of wins with replacement -/
def expected_wins_with_replacement : ℚ :=
  num_draws * prob_win_with_replacement

/-- The probability of winning in a single draw without replacement -/
def prob_win_without_replacement : ℚ :=
  (red_balls.choose 2 + black_balls.choose 2) / total_balls.choose 2

/-- The expected number of wins without replacement -/
def expected_wins_without_replacement : ℚ :=
  (0 * (12 / 35) + 1 * (16 / 35) + 2 * (7 / 35))

/-- Theorem stating that the expected number of wins is 6/7 for both cases -/
theorem expected_wins_equal :
  expected_wins_with_replacement = 6/7 ∧
  expected_wins_without_replacement = 6/7 := by
  sorry


end NUMINAMATH_CALUDE_expected_wins_equal_l2437_243733


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2437_243761

/-- Given a stratified sample where the ratio of product A to total production
    is 1/5 and 18 products of type A are sampled, prove the total sample size is 90. -/
theorem stratified_sample_size (sample_A : ℕ) (ratio_A : ℚ) (total_sample : ℕ) :
  sample_A = 18 →
  ratio_A = 1 / 5 →
  (sample_A : ℚ) / total_sample = ratio_A →
  total_sample = 90 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2437_243761


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l2437_243708

/-- Given a line passing through the points (-2, -3) and (3, 4),
    the product of its slope and y-intercept is equal to -7/25. -/
theorem line_slope_intercept_product : 
  let p₁ : ℝ × ℝ := (-2, -3)
  let p₂ : ℝ × ℝ := (3, 4)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)  -- slope
  let b : ℝ := p₁.2 - m * p₁.1  -- y-intercept
  m * b = -7/25 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l2437_243708


namespace NUMINAMATH_CALUDE_gcd_10010_15015_l2437_243798

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10010_15015_l2437_243798


namespace NUMINAMATH_CALUDE_potato_bag_weight_l2437_243726

theorem potato_bag_weight : ∀ w : ℝ, w = 12 / (w / 2) → w = 12 := by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l2437_243726


namespace NUMINAMATH_CALUDE_m_range_l2437_243785

def p (m : ℝ) : Prop := ∀ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x > m

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m*x + 1 ≤ 0

theorem m_range (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m = -2 ∨ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_m_range_l2437_243785


namespace NUMINAMATH_CALUDE_max_regions_formula_l2437_243710

/-- The maximum number of regions delimited by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem stating the maximum number of regions delimited by n lines in a plane -/
theorem max_regions_formula (n : ℕ) :
  max_regions n = 1 + n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_formula_l2437_243710


namespace NUMINAMATH_CALUDE_debora_has_twelve_more_dresses_l2437_243714

/-- The number of dresses each person has -/
structure Dresses where
  emily : ℕ
  melissa : ℕ
  debora : ℕ

/-- The conditions of the problem -/
def problem_conditions (d : Dresses) : Prop :=
  d.emily = 16 ∧
  d.melissa = d.emily / 2 ∧
  d.debora > d.melissa ∧
  d.emily + d.melissa + d.debora = 44

/-- The theorem to prove -/
theorem debora_has_twelve_more_dresses (d : Dresses) 
  (h : problem_conditions d) : d.debora = d.melissa + 12 := by
  sorry

#check debora_has_twelve_more_dresses

end NUMINAMATH_CALUDE_debora_has_twelve_more_dresses_l2437_243714


namespace NUMINAMATH_CALUDE_correct_system_l2437_243787

/-- Represents the price of a horse in taels -/
def horse_price : ℝ := sorry

/-- Represents the price of a head of cattle in taels -/
def cattle_price : ℝ := sorry

/-- The total price of 4 horses and 6 heads of cattle is 48 taels -/
axiom eq1 : 4 * horse_price + 6 * cattle_price = 48

/-- The total price of 3 horses and 5 heads of cattle is 38 taels -/
axiom eq2 : 3 * horse_price + 5 * cattle_price = 38

/-- The system of equations correctly represents the given conditions -/
theorem correct_system : 
  (4 * horse_price + 6 * cattle_price = 48) ∧ 
  (3 * horse_price + 5 * cattle_price = 38) :=
sorry

end NUMINAMATH_CALUDE_correct_system_l2437_243787


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l2437_243756

/-- The number of steps Cozy takes to climb the stairs -/
def cozy_jumps (n : ℕ) : ℕ := (n + 2) / 3

/-- The number of steps Dash takes to climb the stairs -/
def dash_jumps (n : ℕ) : ℕ := (n + 6) / 7

/-- Theorem stating the smallest number of steps in the staircase -/
theorem smallest_staircase_steps : 
  ∃ (n : ℕ), 
    n % 11 = 0 ∧ 
    cozy_jumps n - dash_jumps n = 13 ∧ 
    ∀ (m : ℕ), m < n → (m % 11 ≠ 0 ∨ cozy_jumps m - dash_jumps m ≠ 13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l2437_243756


namespace NUMINAMATH_CALUDE_jungkook_age_relation_l2437_243757

theorem jungkook_age_relation :
  ∃ (x : ℕ), 
    (46 - x : ℤ) = 4 * (16 - x : ℤ) ∧ 
    x ≤ 16 ∧ 
    x ≤ 46 ∧ 
    x = 6 :=
by sorry

end NUMINAMATH_CALUDE_jungkook_age_relation_l2437_243757


namespace NUMINAMATH_CALUDE_sum_of_integers_l2437_243781

theorem sum_of_integers (a b : ℕ) (h1 : a > b) (h2 : a - b = 5) (h3 : a * b = 84) : a + b = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2437_243781


namespace NUMINAMATH_CALUDE_paper_strip_dimensions_l2437_243730

theorem paper_strip_dimensions (a b c : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) : 
  (a = 1 ∧ b + c = 22) ∨ (a = 22 ∧ b + c = 1) :=
sorry

end NUMINAMATH_CALUDE_paper_strip_dimensions_l2437_243730


namespace NUMINAMATH_CALUDE_gcd_of_180_270_450_l2437_243743

theorem gcd_of_180_270_450 : Nat.gcd 180 (Nat.gcd 270 450) = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_180_270_450_l2437_243743


namespace NUMINAMATH_CALUDE_f_always_positive_l2437_243793

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + 4*x^3 + a*x^2 - 4*x + 1

/-- Theorem stating that f(x) is always positive if and only if a > 2 -/
theorem f_always_positive (a : ℝ) : (∀ x : ℝ, f a x > 0) ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_l2437_243793


namespace NUMINAMATH_CALUDE_jiaotong_primary_school_students_l2437_243748

theorem jiaotong_primary_school_students (b g : ℕ) : 
  b = 7 * g ∧ b = g + 900 → b + g = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jiaotong_primary_school_students_l2437_243748


namespace NUMINAMATH_CALUDE_razorback_tshirt_revenue_l2437_243776

/-- The amount of money made from selling t-shirts at the Razorback shop -/
theorem razorback_tshirt_revenue :
  let profit_per_tshirt : ℕ := 62
  let tshirts_sold : ℕ := 183
  let total_profit : ℕ := profit_per_tshirt * tshirts_sold
  total_profit = 11346 := by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_revenue_l2437_243776


namespace NUMINAMATH_CALUDE_first_player_wins_l2437_243735

/-- Represents a stick with a certain length -/
structure Stick :=
  (length : ℝ)

/-- Represents the state of the game -/
structure GameState :=
  (sticks : List Stick)

/-- Represents a player's move, breaking a stick into two parts -/
def breakStick (s : Stick) : Stick × Stick :=
  sorry

/-- Checks if three sticks can form a triangle -/
def canFormTriangle (s1 s2 s3 : Stick) : Prop :=
  sorry

/-- Represents a player's strategy -/
def Strategy := GameState → Option (Stick × (Stick × Stick))

/-- The first player's strategy -/
def firstPlayerStrategy : Strategy :=
  sorry

/-- The second player's strategy -/
def secondPlayerStrategy : Strategy :=
  sorry

/-- Simulates the game for three moves -/
def gameSimulation (s1 : Strategy) (s2 : Strategy) : GameState :=
  sorry

/-- Checks if the given game state allows forming two triangles -/
def canFormTwoTriangles (gs : GameState) : Prop :=
  sorry

/-- The main theorem stating that the first player can guarantee a win -/
theorem first_player_wins :
  ∀ (s2 : Strategy),
  ∃ (s1 : Strategy),
  canFormTwoTriangles (gameSimulation s1 s2) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l2437_243735


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l2437_243740

theorem logarithmic_equation_solution :
  ∃ x : ℝ, (Real.log x / Real.log 4) - 3 * (Real.log 8 / Real.log 2) = 1 - (Real.log 2 / Real.log 2) ∧ x = 262144 := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l2437_243740


namespace NUMINAMATH_CALUDE_range_of_a_in_p_a_neither_necessary_nor_sufficient_for_b_l2437_243711

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define sets A and B
def A : Set ℝ := {a | a ≤ 1}
def B : Set ℝ := {a | a ≥ 1 ∨ a ≤ -2}

-- Theorem for the range of a in proposition p
theorem range_of_a_in_p : ∀ a : ℝ, p a ↔ a ∈ A := by sorry

-- Theorem for the relationship between A and B
theorem a_neither_necessary_nor_sufficient_for_b :
  (¬∀ a : ℝ, a ∈ B → a ∈ A) ∧ (¬∀ a : ℝ, a ∈ A → a ∈ B) := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_p_a_neither_necessary_nor_sufficient_for_b_l2437_243711


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l2437_243724

theorem factorial_equation_solutions :
  ∀ x y z : ℕ+, 2^x.val + 3^y.val - 7 = Nat.factorial z.val →
    ((x = 2 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 4)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l2437_243724


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2437_243728

theorem quadratic_minimum (x : ℝ) : 
  let f : ℝ → ℝ := fun x => x^2 - 12*x + 35
  ∃ (min_x : ℝ), ∀ y, f y ≥ f min_x ∧ min_x = 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2437_243728
