import Mathlib

namespace NUMINAMATH_CALUDE_chairs_for_play_l469_46973

theorem chairs_for_play (rows : ℕ) (chairs_per_row : ℕ) 
  (h1 : rows = 27) (h2 : chairs_per_row = 16) : 
  rows * chairs_per_row = 432 := by
  sorry

end NUMINAMATH_CALUDE_chairs_for_play_l469_46973


namespace NUMINAMATH_CALUDE_min_value_expression_l469_46920

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 + 8*a*b + 32*b^2 + 24*b*c + 8*c^2 ≥ 36 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 1 ∧
    a₀^2 + 8*a₀*b₀ + 32*b₀^2 + 24*b₀*c₀ + 8*c₀^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l469_46920


namespace NUMINAMATH_CALUDE_octagon_cannot_cover_floor_l469_46967

/-- Calculate the interior angle of a regular polygon with n sides -/
def interiorAngle (n : ℕ) : ℚ :=
  (n - 2 : ℚ) * 180 / n

/-- Check if a given angle divides 360° evenly -/
def divides360 (angle : ℚ) : Prop :=
  ∃ k : ℕ, k * angle = 360

/-- Theorem: Among equilateral triangles, squares, hexagons, and octagons,
    only the octagon's interior angle does not divide 360° evenly -/
theorem octagon_cannot_cover_floor :
  divides360 (interiorAngle 3) ∧
  divides360 (interiorAngle 4) ∧
  divides360 (interiorAngle 6) ∧
  ¬divides360 (interiorAngle 8) :=
sorry

end NUMINAMATH_CALUDE_octagon_cannot_cover_floor_l469_46967


namespace NUMINAMATH_CALUDE_joan_grilled_cheese_sandwiches_l469_46932

/-- Represents the number of cheese slices required for one ham sandwich. -/
def ham_cheese_slices : ℕ := 2

/-- Represents the number of cheese slices required for one grilled cheese sandwich. -/
def grilled_cheese_slices : ℕ := 3

/-- Represents the total number of cheese slices Joan uses. -/
def total_cheese_slices : ℕ := 50

/-- Represents the number of ham sandwiches Joan makes. -/
def ham_sandwiches : ℕ := 10

/-- Proves that Joan makes 10 grilled cheese sandwiches. -/
theorem joan_grilled_cheese_sandwiches : 
  (total_cheese_slices - ham_cheese_slices * ham_sandwiches) / grilled_cheese_slices = 10 := by
  sorry

end NUMINAMATH_CALUDE_joan_grilled_cheese_sandwiches_l469_46932


namespace NUMINAMATH_CALUDE_stating_number_of_passed_candidates_l469_46986

/-- Represents the number of candidates who passed the examination. -/
def passed_candidates : ℕ := 346

/-- Represents the total number of candidates. -/
def total_candidates : ℕ := 500

/-- Represents the average marks of all candidates. -/
def average_marks : ℚ := 60

/-- Represents the average marks of passed candidates. -/
def average_marks_passed : ℚ := 80

/-- Represents the average marks of failed candidates. -/
def average_marks_failed : ℚ := 15

/-- 
Theorem stating that the number of candidates who passed the examination is 346,
given the total number of candidates, average marks of all candidates,
average marks of passed candidates, and average marks of failed candidates.
-/
theorem number_of_passed_candidates : 
  passed_candidates = 346 ∧
  passed_candidates + (total_candidates - passed_candidates) = total_candidates ∧
  (passed_candidates * average_marks_passed + 
   (total_candidates - passed_candidates) * average_marks_failed) / total_candidates = average_marks :=
by sorry

end NUMINAMATH_CALUDE_stating_number_of_passed_candidates_l469_46986


namespace NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l469_46959

/-- 
Given:
- Joel is currently 5 years old
- Joel's dad is currently 32 years old

Prove that Joel will be 27 years old when his dad is twice as old as him.
-/
theorem joel_age_when_dad_twice_as_old (joel_current_age : ℕ) (dad_current_age : ℕ) :
  joel_current_age = 5 →
  dad_current_age = 32 →
  ∃ (future_joel_age : ℕ), 
    future_joel_age + joel_current_age = dad_current_age ∧
    2 * future_joel_age = future_joel_age + dad_current_age ∧
    future_joel_age = 27 :=
by sorry

end NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l469_46959


namespace NUMINAMATH_CALUDE_g_f_neg_3_l469_46910

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 7

-- Define g(f(3)) = 15 as a hypothesis
axiom g_f_3 : ∃ g : ℝ → ℝ, g (f 3) = 15

-- Theorem to prove
theorem g_f_neg_3 : ∃ g : ℝ → ℝ, g (f (-3)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_g_f_neg_3_l469_46910


namespace NUMINAMATH_CALUDE_fraction_equality_l469_46956

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - 4*b ≠ 0) (h4 : 4*a - b ≠ 0)
  (h5 : (4*a + b) / (a - 4*b) = 3) : (a + 4*b) / (4*a - b) = 9/53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l469_46956


namespace NUMINAMATH_CALUDE_set_A_is_empty_l469_46987

theorem set_A_is_empty (a : ℝ) : {x : ℝ | |x - 1| ≤ 2*a - a^2 - 2} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_set_A_is_empty_l469_46987


namespace NUMINAMATH_CALUDE_grade_students_ratio_l469_46907

theorem grade_students_ratio (sixth_grade seventh_grade : ℕ) : 
  (sixth_grade : ℚ) / seventh_grade = 3 / 4 →
  seventh_grade - sixth_grade = 13 →
  sixth_grade = 39 ∧ seventh_grade = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_grade_students_ratio_l469_46907


namespace NUMINAMATH_CALUDE_some_club_members_not_committee_members_l469_46976

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (ClubMember : U → Prop)
variable (CommitteeMember : U → Prop)
variable (Punctual : U → Prop)

-- State the theorem
theorem some_club_members_not_committee_members :
  (∃ x, ClubMember x ∧ ¬Punctual x) →
  (∀ x, CommitteeMember x → Punctual x) →
  ∃ x, ClubMember x ∧ ¬CommitteeMember x :=
by
  sorry


end NUMINAMATH_CALUDE_some_club_members_not_committee_members_l469_46976


namespace NUMINAMATH_CALUDE_tangent_line_properties_l469_46977

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2

theorem tangent_line_properties :
  (∃ x : ℝ, (deriv f) x = 3) ∧
  (∃! t : ℝ, (f t - 2) / t = (deriv f) t) ∧
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
    (f t₁ - 4) / (t₁ - 1) = (deriv f) t₁ ∧
    (f t₂ - 4) / (t₂ - 1) = (deriv f) t₂ ∧
    ∀ t : ℝ, t ≠ t₁ → t ≠ t₂ →
      (f t - 4) / (t - 1) ≠ (deriv f) t) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l469_46977


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l469_46961

theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ (n : ℕ), a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 1/2 →                                -- First term is 1/2
  a 5 = 8 →                                  -- Fifth term is 8
  a 2 * a 3 * a 4 = 8 :=                     -- Product of middle terms is 8
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l469_46961


namespace NUMINAMATH_CALUDE_rectangle_max_area_l469_46916

/-- A rectangle with integer dimensions and perimeter 30 has a maximum area of 56 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 15 →
  ∀ a b : ℕ,
  a + b = 15 →
  l * w ≤ 56 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l469_46916


namespace NUMINAMATH_CALUDE_sara_quarters_count_l469_46994

theorem sara_quarters_count (initial_quarters final_quarters dad_quarters : ℕ) : 
  initial_quarters = 21 → dad_quarters = 49 → final_quarters = initial_quarters + dad_quarters → 
  final_quarters = 70 := by
sorry

end NUMINAMATH_CALUDE_sara_quarters_count_l469_46994


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l469_46901

theorem gcd_of_three_numbers : Nat.gcd 12357 (Nat.gcd 15498 21726) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l469_46901


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l469_46980

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (4, -2)
def b : ℝ → ℝ × ℝ := λ x ↦ (x, 5)

/-- Parallel vectors in ℝ² -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  parallel a (b x) → x = -10 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l469_46980


namespace NUMINAMATH_CALUDE_reflection_x_axis_l469_46952

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The reflection of (-2, -3) across the x-axis is (-2, 3) -/
theorem reflection_x_axis : reflect_x (-2, -3) = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_reflection_x_axis_l469_46952


namespace NUMINAMATH_CALUDE_thermal_underwear_sales_l469_46946

def cost_price : ℕ := 50
def standard_price : ℕ := 70
def price_adjustments : List ℤ := [5, 2, 1, 0, -2]
def sets_sold : List ℕ := [7, 10, 15, 20, 23]

theorem thermal_underwear_sales :
  (List.sum (List.zipWith (· * ·) price_adjustments (List.map Int.ofNat sets_sold)) = 24) ∧
  ((standard_price - cost_price) * (List.sum sets_sold) + 24 = 1524) := by
  sorry

end NUMINAMATH_CALUDE_thermal_underwear_sales_l469_46946


namespace NUMINAMATH_CALUDE_water_usage_difference_l469_46984

/-- Proves the difference in daily water usage before and after installing a water recycling device -/
theorem water_usage_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (b / a) - (b / (a + 4)) = (4 * b) / (a * (a + 4)) := by
  sorry

end NUMINAMATH_CALUDE_water_usage_difference_l469_46984


namespace NUMINAMATH_CALUDE_range_of_c_over_a_l469_46992

theorem range_of_c_over_a (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 1) (h3 : a + b + c = 0) :
  ∀ x, (x = c / a) → -2 < x ∧ x < -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_c_over_a_l469_46992


namespace NUMINAMATH_CALUDE_eleven_girls_l469_46918

/-- Represents a circular arrangement of girls -/
structure CircularArrangement where
  girls : ℕ  -- Total number of girls in the circle

/-- Defines the position of one girl relative to another in the circle -/
def position (c : CircularArrangement) (left right : ℕ) : Prop :=
  left + right + 2 = c.girls

/-- Theorem: If Florence is the 4th on the left and 7th on the right from Jess,
    then there are 11 girls in total -/
theorem eleven_girls (c : CircularArrangement) :
  position c 3 6 → c.girls = 11 := by
  sorry

#check eleven_girls

end NUMINAMATH_CALUDE_eleven_girls_l469_46918


namespace NUMINAMATH_CALUDE_betty_oranges_l469_46969

/-- Given 3 boxes and 8 oranges per box, the total number of oranges is 24. -/
theorem betty_oranges (num_boxes : ℕ) (oranges_per_box : ℕ) 
  (h1 : num_boxes = 3) 
  (h2 : oranges_per_box = 8) : 
  num_boxes * oranges_per_box = 24 := by
  sorry

end NUMINAMATH_CALUDE_betty_oranges_l469_46969


namespace NUMINAMATH_CALUDE_stating_inscribed_triangle_area_bound_l469_46995

/-- A parallelogram in a 2D plane. -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ

/-- A triangle in a 2D plane. -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Checks if a point is inside or on the perimeter of a parallelogram. -/
def isInOrOnParallelogram (p : ℝ × ℝ) (pgram : Parallelogram) : Prop :=
  sorry

/-- Checks if a triangle is inscribed in a parallelogram. -/
def isInscribed (t : Triangle) (pgram : Parallelogram) : Prop :=
  ∀ i, isInOrOnParallelogram (t.vertices i) pgram

/-- Calculates the area of a parallelogram. -/
noncomputable def areaParallelogram (pgram : Parallelogram) : ℝ :=
  sorry

/-- Calculates the area of a triangle. -/
noncomputable def areaTriangle (t : Triangle) : ℝ :=
  sorry

/-- 
Theorem stating that the area of any triangle inscribed in a parallelogram
is less than or equal to half the area of the parallelogram.
-/
theorem inscribed_triangle_area_bound
  (pgram : Parallelogram) (t : Triangle) (h : isInscribed t pgram) :
  areaTriangle t ≤ (1/2) * areaParallelogram pgram :=
by sorry

end NUMINAMATH_CALUDE_stating_inscribed_triangle_area_bound_l469_46995


namespace NUMINAMATH_CALUDE_gcd_187_253_l469_46991

theorem gcd_187_253 : Nat.gcd 187 253 = 11 := by sorry

end NUMINAMATH_CALUDE_gcd_187_253_l469_46991


namespace NUMINAMATH_CALUDE_f_is_odd_sum_greater_than_two_l469_46963

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x + x / (x^2 + 1)

-- Theorem 1: f is an odd function
theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by sorry

-- Theorem 2: If x₁ > 0, x₂ > 0, x₁ ≠ x₂, and f(x₁) = f(x₂), then x₁ + x₂ > 2
theorem sum_greater_than_two (x₁ x₂ : ℝ) (h1 : x₁ > 0) (h2 : x₂ > 0) (h3 : x₁ ≠ x₂) (h4 : f x₁ = f x₂) : x₁ + x₂ > 2 := by sorry

end NUMINAMATH_CALUDE_f_is_odd_sum_greater_than_two_l469_46963


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l469_46933

/-- Calculates the final tax percentage given spending percentages and tax rates --/
def final_tax_percentage (clothing_percent : ℝ) (food_percent : ℝ) (electronics_percent : ℝ) 
  (other_percent : ℝ) (clothing_tax : ℝ) (food_tax : ℝ) (electronics_tax : ℝ) 
  (other_tax : ℝ) (loyalty_discount : ℝ) : ℝ :=
  let total_tax := clothing_percent * clothing_tax + food_percent * food_tax + 
                   electronics_percent * electronics_tax + other_percent * other_tax
  let discounted_tax := total_tax * (1 - loyalty_discount)
  discounted_tax * 100

theorem shopping_tax_calculation :
  final_tax_percentage 0.4 0.25 0.2 0.15 0.05 0.02 0.1 0.08 0.03 = 5.529 := by
  sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l469_46933


namespace NUMINAMATH_CALUDE_subtraction_problem_l469_46968

theorem subtraction_problem (x : ℝ) (h : 40 / x = 5) : 20 - x = 12 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l469_46968


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_l469_46944

theorem smallest_sum_of_factors (x y z w : ℕ+) : 
  x * y * z * w = 362880 → 
  ∀ a b c d : ℕ+, a * b * c * d = 362880 → x + y + z + w ≤ a + b + c + d →
  x + y + z + w = 69 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_l469_46944


namespace NUMINAMATH_CALUDE_triangle_square_diagonal_l469_46921

/-- Given a triangle with base 6 and height 4, the length of the diagonal of a square 
    with the same area as the triangle is √24. -/
theorem triangle_square_diagonal : 
  ∀ (triangle_base triangle_height : ℝ),
  triangle_base = 6 →
  triangle_height = 4 →
  ∃ (square_diagonal : ℝ),
    (1/2 * triangle_base * triangle_height) = square_diagonal^2 / 2 ∧
    square_diagonal = Real.sqrt 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_square_diagonal_l469_46921


namespace NUMINAMATH_CALUDE_triangle_angle_from_sides_and_area_l469_46917

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    prove that if a = 2√3, b = 2, and the area S = √3, then C = π/6 -/
theorem triangle_angle_from_sides_and_area 
  (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  a = 2 * Real.sqrt 3 →
  b = 2 →
  S = Real.sqrt 3 →
  S = 1/2 * a * b * Real.sin C →
  C = π/6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_from_sides_and_area_l469_46917


namespace NUMINAMATH_CALUDE_vector_collinear_opposite_direction_l469_46948

/-- Two vectors in ℝ² -/
def Vector2D : Type := ℝ × ℝ

/-- Check if two vectors are collinear -/
def collinear (v w : Vector2D) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

/-- Check if two vectors have opposite directions -/
def opposite_directions (v w : Vector2D) : Prop :=
  ∃ k : ℝ, k < 0 ∧ v = (k * w.1, k * w.2)

/-- The main theorem -/
theorem vector_collinear_opposite_direction (m : ℝ) :
  let a : Vector2D := (m, 1)
  let b : Vector2D := (1, m)
  collinear a b → opposite_directions a b → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinear_opposite_direction_l469_46948


namespace NUMINAMATH_CALUDE_truck_max_load_l469_46929

/-- The maximum load a truck can carry, given the mass of lemon bags and remaining capacity -/
theorem truck_max_load (mass_per_bag : ℕ) (num_bags : ℕ) (remaining_capacity : ℕ) :
  mass_per_bag = 8 →
  num_bags = 100 →
  remaining_capacity = 100 →
  mass_per_bag * num_bags + remaining_capacity = 900 := by
  sorry

end NUMINAMATH_CALUDE_truck_max_load_l469_46929


namespace NUMINAMATH_CALUDE_inequality_proof_l469_46937

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l469_46937


namespace NUMINAMATH_CALUDE_least_months_to_triple_l469_46908

def interest_rate : ℝ := 1.06

theorem least_months_to_triple (t : ℕ) : t = 19 ↔ 
  (∀ n : ℕ, n < 19 → interest_rate ^ n ≤ 3) ∧ 
  interest_rate ^ 19 > 3 := by
  sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l469_46908


namespace NUMINAMATH_CALUDE_apple_ratio_l469_46900

theorem apple_ratio (jim_apples jane_apples jerry_apples : ℕ) 
  (h1 : jim_apples = 20)
  (h2 : jane_apples = 60)
  (h3 : jerry_apples = 40) :
  (jim_apples + jane_apples + jerry_apples) / 3 / jim_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_ratio_l469_46900


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l469_46915

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + |x| ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 + |x₀| < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l469_46915


namespace NUMINAMATH_CALUDE_complex_sum_and_reciprocal_l469_46997

theorem complex_sum_and_reciprocal : 
  let z : ℂ := 1 - I
  (z⁻¹ + z) = (3/2 : ℂ) - (1/2 : ℂ) * I := by sorry

end NUMINAMATH_CALUDE_complex_sum_and_reciprocal_l469_46997


namespace NUMINAMATH_CALUDE_student_marks_calculation_l469_46951

theorem student_marks_calculation 
  (max_marks : ℕ) 
  (passing_percentage : ℚ) 
  (fail_margin : ℕ) 
  (h1 : max_marks = 400)
  (h2 : passing_percentage = 36 / 100)
  (h3 : fail_margin = 14) :
  ∃ (student_marks : ℕ), 
    student_marks = max_marks * passing_percentage - fail_margin ∧
    student_marks = 130 :=
by sorry

end NUMINAMATH_CALUDE_student_marks_calculation_l469_46951


namespace NUMINAMATH_CALUDE_fraction_irreducible_l469_46935

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l469_46935


namespace NUMINAMATH_CALUDE_stating_consecutive_sum_equals_odd_divisors_l469_46943

/-- 
Given a positive integer n, count_consecutive_sum n returns the number of ways
n can be represented as a sum of one or more consecutive positive integers.
-/
def count_consecutive_sum (n : ℕ+) : ℕ := sorry

/-- 
Given a positive integer n, count_odd_divisors n returns the number of odd
divisors of n.
-/
def count_odd_divisors (n : ℕ+) : ℕ := sorry

/-- 
Theorem stating that for any positive integer n, the number of ways n can be
represented as a sum of one or more consecutive positive integers is equal to
the number of odd divisors of n.
-/
theorem consecutive_sum_equals_odd_divisors (n : ℕ+) :
  count_consecutive_sum n = count_odd_divisors n := by sorry

end NUMINAMATH_CALUDE_stating_consecutive_sum_equals_odd_divisors_l469_46943


namespace NUMINAMATH_CALUDE_toys_sold_proof_l469_46978

/-- The number of toys sold by a man -/
def number_of_toys : ℕ := 18

/-- The selling price of the toys -/
def selling_price : ℕ := 23100

/-- The cost price of one toy -/
def cost_price : ℕ := 1100

/-- The gain from the sale -/
def gain : ℕ := 3 * cost_price

theorem toys_sold_proof :
  number_of_toys * cost_price + gain = selling_price :=
by sorry

end NUMINAMATH_CALUDE_toys_sold_proof_l469_46978


namespace NUMINAMATH_CALUDE_farm_animal_ratio_l469_46903

theorem farm_animal_ratio (cows sheep pigs : ℕ) : 
  cows = 12 →
  sheep = 2 * cows →
  cows + sheep + pigs = 108 →
  pigs / sheep = 3 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_ratio_l469_46903


namespace NUMINAMATH_CALUDE_fraction_equality_l469_46971

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : y^2 - 1/x ≠ 0) :
  (x^2 - 1/y) / (y^2 - 1/x) = x/y := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l469_46971


namespace NUMINAMATH_CALUDE_folded_quadrilateral_l469_46954

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if two points coincide after folding -/
def coincide (p1 p2 : Point) : Prop :=
  ∃ (m : Point), (m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2) ∧
  (p2.y - p1.y) * (m.x - p1.x) = (p1.x - p2.x) * (m.y - p1.y)

/-- Calculates the area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Main theorem -/
theorem folded_quadrilateral :
  ∀ (m n : ℝ),
  let q := Quadrilateral.mk
    (Point.mk 0 2)  -- A
    (Point.mk 4 0)  -- B
    (Point.mk 7 3)  -- C
    (Point.mk m n)  -- D
  coincide q.A q.B ∧ coincide q.C q.D →
  m = 3/5 ∧ n = 31/5 ∧ area q = 117/5 := by
  sorry

end NUMINAMATH_CALUDE_folded_quadrilateral_l469_46954


namespace NUMINAMATH_CALUDE_two_distinct_roots_condition_l469_46912

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : ℝ := x^2 - 4*x + 2*m

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m = 0 ∧ quadratic_equation x₂ m = 0

-- Theorem statement
theorem two_distinct_roots_condition (m : ℝ) :
  has_two_distinct_real_roots m ↔ m < 2 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_condition_l469_46912


namespace NUMINAMATH_CALUDE_max_coverage_of_two_inch_card_l469_46953

/-- A checkerboard square -/
structure CheckerboardSquare where
  size : Real
  (size_positive : size > 0)

/-- A square card -/
structure SquareCard where
  side_length : Real
  (side_length_positive : side_length > 0)

/-- Represents the coverage of a card on a checkerboard -/
def Coverage (card : SquareCard) (square : CheckerboardSquare) : Nat :=
  sorry

/-- Theorem: The maximum number of one-inch squares on a checkerboard 
    that can be covered by a 2-inch square card is 12 -/
theorem max_coverage_of_two_inch_card : 
  ∀ (board_square : CheckerboardSquare) (card : SquareCard),
    board_square.size = 1 → 
    card.side_length = 2 → 
    ∃ (n : Nat), Coverage card board_square = n ∧ 
      ∀ (m : Nat), Coverage card board_square ≤ m → n ≤ m ∧ n = 12 :=
sorry

end NUMINAMATH_CALUDE_max_coverage_of_two_inch_card_l469_46953


namespace NUMINAMATH_CALUDE_jerry_spent_difference_l469_46966

/-- Jerry's initial amount of money in dollars -/
def initial_amount : ℕ := 18

/-- Jerry's remaining amount of money in dollars -/
def remaining_amount : ℕ := 12

/-- The amount Jerry spent on video games -/
def amount_spent : ℕ := initial_amount - remaining_amount

theorem jerry_spent_difference :
  amount_spent = initial_amount - remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_jerry_spent_difference_l469_46966


namespace NUMINAMATH_CALUDE_salt_mixture_proof_l469_46982

theorem salt_mixture_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_volume : ℝ) (added_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 40 ∧ 
  initial_concentration = 0.2 ∧ 
  added_volume = 40 ∧ 
  added_concentration = 0.6 ∧ 
  final_concentration = 0.4 →
  (initial_volume * initial_concentration + added_volume * added_concentration) / 
  (initial_volume + added_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_mixture_proof_l469_46982


namespace NUMINAMATH_CALUDE_both_products_not_qualified_l469_46939

-- Define the qualification rates for Factory A and Factory B
def qualification_rate_A : ℝ := 0.9
def qualification_rate_B : ℝ := 0.8

-- Define the probability that both products are not qualified
def both_not_qualified : ℝ := (1 - qualification_rate_A) * (1 - qualification_rate_B)

-- Theorem statement
theorem both_products_not_qualified :
  both_not_qualified = 0.02 :=
sorry

end NUMINAMATH_CALUDE_both_products_not_qualified_l469_46939


namespace NUMINAMATH_CALUDE_polynomial_roots_l469_46970

def polynomial (x : ℝ) : ℝ := x^3 - 5*x^2 + 3*x + 9

theorem polynomial_roots : 
  (polynomial (-1) = 0) ∧ 
  (polynomial 3 = 0) ∧ 
  (∃ (f : ℝ → ℝ), ∀ x, polynomial x = (x + 1) * (x - 3)^2 * f x) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l469_46970


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l469_46942

theorem cara_seating_arrangements (n : ℕ) (h : n = 6) : 
  Nat.choose n 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l469_46942


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l469_46985

/-- The area of the region between two concentric circles, where the radius of the larger circle
    is three times the radius of the smaller circle, and the diameter of the smaller circle is 6 units. -/
theorem shaded_area_between_circles (π : ℝ) : ℝ := by
  -- Define the diameter of the smaller circle
  let small_diameter : ℝ := 6
  -- Define the radius of the smaller circle
  let small_radius : ℝ := small_diameter / 2
  -- Define the radius of the larger circle
  let large_radius : ℝ := 3 * small_radius
  -- Define the area of the shaded region
  let shaded_area : ℝ := π * large_radius^2 - π * small_radius^2
  -- Prove that the shaded area equals 72π
  have : shaded_area = 72 * π := by sorry
  -- Return the result
  exact 72 * π

end NUMINAMATH_CALUDE_shaded_area_between_circles_l469_46985


namespace NUMINAMATH_CALUDE_age_difference_proof_l469_46931

theorem age_difference_proof (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → 
  (10 * a + b + 3 = 3 * (10 * b + a + 3)) → 
  (10 * a + b) - (10 * b + a) = 36 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l469_46931


namespace NUMINAMATH_CALUDE_evans_needed_amount_l469_46913

/-- The amount Evan still needs to buy the watch -/
def amount_needed (david_found : ℕ) (evan_initial : ℕ) (watch_cost : ℕ) : ℕ :=
  watch_cost - (evan_initial + david_found)

/-- Theorem stating the amount Evan still needs -/
theorem evans_needed_amount :
  amount_needed 12 1 20 = 7 := by
  sorry

end NUMINAMATH_CALUDE_evans_needed_amount_l469_46913


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l469_46975

theorem trigonometric_simplification (α : ℝ) : 
  (Real.tan ((5 / 4) * Real.pi - 4 * α) * (Real.sin ((5 / 4) * Real.pi + 4 * α))^2) / 
  (1 - 2 * (Real.cos (4 * α))^2) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l469_46975


namespace NUMINAMATH_CALUDE_infinite_sum_equals_ln2_squared_l469_46962

/-- The infinite sum of the given series is equal to ln(2)² -/
theorem infinite_sum_equals_ln2_squared :
  ∑' k : ℕ, (3 * Real.log (4 * k + 2) / (4 * k + 2) -
             Real.log (4 * k + 3) / (4 * k + 3) -
             Real.log (4 * k + 4) / (4 * k + 4) -
             Real.log (4 * k + 5) / (4 * k + 5)) = (Real.log 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_ln2_squared_l469_46962


namespace NUMINAMATH_CALUDE_square_partition_impossibility_l469_46999

theorem square_partition_impossibility :
  ¬ ∃ (partition : List (ℕ × ℕ)),
    (∀ (rect : ℕ × ℕ), rect ∈ partition →
      (2 * (rect.1 + rect.2) = 18 ∨ 2 * (rect.1 + rect.2) = 22 ∨ 2 * (rect.1 + rect.2) = 26)) ∧
    (List.sum (partition.map (λ rect => rect.1 * rect.2)) = 35 * 35) :=
by
  sorry


end NUMINAMATH_CALUDE_square_partition_impossibility_l469_46999


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l469_46911

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →
  (a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1) →
  ((a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l469_46911


namespace NUMINAMATH_CALUDE_journey_speed_proof_l469_46923

theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 672 ∧ total_time = 30 ∧ second_half_speed = 24 →
  ∃ first_half_speed : ℝ,
    first_half_speed = 21 ∧
    first_half_speed * (total_time / 2) + second_half_speed * (total_time / 2) = total_distance :=
by
  sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l469_46923


namespace NUMINAMATH_CALUDE_center_square_side_length_l469_46998

theorem center_square_side_length 
  (main_square_side : ℝ) 
  (l_shape_area_fraction : ℝ) 
  (num_l_shapes : ℕ) :
  main_square_side = 120 →
  l_shape_area_fraction = 1 / 5 →
  num_l_shapes = 4 →
  let total_area := main_square_side ^ 2
  let l_shapes_area := num_l_shapes * l_shape_area_fraction * total_area
  let center_square_area := total_area - l_shapes_area
  Real.sqrt center_square_area = 60 := by sorry

end NUMINAMATH_CALUDE_center_square_side_length_l469_46998


namespace NUMINAMATH_CALUDE_norwich_carriages_l469_46960

/-- The number of carriages in each town --/
structure Carriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flying_scotsman : ℕ

/-- The conditions of the carriage problem --/
def carriage_problem (c : Carriages) : Prop :=
  c.euston = c.norfolk + 20 ∧
  c.flying_scotsman = c.norwich + 20 ∧
  c.euston = 130 ∧
  c.euston + c.norfolk + c.norwich + c.flying_scotsman = 460

/-- The theorem stating that Norwich had 100 carriages --/
theorem norwich_carriages :
  ∃ c : Carriages, carriage_problem c ∧ c.norwich = 100 := by
  sorry

end NUMINAMATH_CALUDE_norwich_carriages_l469_46960


namespace NUMINAMATH_CALUDE_boys_camp_total_l469_46989

theorem boys_camp_total (total : ℕ) : 
  (total : ℝ) * 0.2 * 0.7 = 63 → total = 450 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_total_l469_46989


namespace NUMINAMATH_CALUDE_complex_expression_equals_one_l469_46928

theorem complex_expression_equals_one : 
  (((4.5 * (1 + 2/3) - 6.75) * (2/3)) / 
   ((3 + 1/3) * 0.3 + (5 + 1/3) * (1/8)) / (2 + 2/3)) + 
  ((1 + 4/11) * 0.22 / 0.3 - 0.96) / 
   ((0.2 - 3/40) * 1.6) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_one_l469_46928


namespace NUMINAMATH_CALUDE_ellipse_sum_specific_l469_46981

/-- The sum of the center coordinates and axis lengths of an ellipse -/
def ellipse_sum (h k a b : ℝ) : ℝ := h + k + a + b

/-- Theorem: The sum of center coordinates and axis lengths for a specific ellipse -/
theorem ellipse_sum_specific : ∃ (h k a b : ℝ), 
  (∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ∧ 
  h = 3 ∧ 
  k = -5 ∧ 
  a = 7 ∧ 
  b = 4 ∧ 
  ellipse_sum h k a b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_specific_l469_46981


namespace NUMINAMATH_CALUDE_even_function_monotonicity_l469_46904

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

theorem even_function_monotonicity (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  ∃ a b c : ℝ, a < b ∧ b < c ∧
    (∀ x ∈ Set.Ioo (-3) 1, f m x = -x^2 + 3) ∧
    (∀ x y, -3 < x ∧ x < y ∧ y < a → f m x < f m y) ∧
    (∀ x y, a < x ∧ x < y ∧ y < 1 → f m x > f m y) :=
by sorry

end NUMINAMATH_CALUDE_even_function_monotonicity_l469_46904


namespace NUMINAMATH_CALUDE_zach_ben_score_difference_l469_46974

theorem zach_ben_score_difference :
  ∀ (zach_score ben_score : ℕ),
    zach_score = 42 →
    ben_score = 21 →
    zach_score - ben_score = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_zach_ben_score_difference_l469_46974


namespace NUMINAMATH_CALUDE_nontrivial_solution_iff_l469_46909

/-- A system of linear equations with coefficients a, b, c has a non-trivial solution -/
def has_nontrivial_solution (a b c : ℝ) : Prop :=
  ∃ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    a * x + b * y + c * z = 0 ∧
    b * x + c * y + a * z = 0 ∧
    c * x + a * y + b * z = 0

/-- The main theorem characterizing when the system has a non-trivial solution -/
theorem nontrivial_solution_iff (a b c : ℝ) :
  has_nontrivial_solution a b c ↔ a + b + c = 0 ∨ (a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_nontrivial_solution_iff_l469_46909


namespace NUMINAMATH_CALUDE_product_xy_equals_one_l469_46990

theorem product_xy_equals_one (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (1 + x + x^2) + 1 / (1 + y + y^2) + 1 / (1 + x + y) = 1) :
  x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_equals_one_l469_46990


namespace NUMINAMATH_CALUDE_football_game_attendance_difference_l469_46988

theorem football_game_attendance_difference :
  let saturday : ℕ := 80
  let wednesday (monday : ℕ) : ℕ := monday + 50
  let friday (monday : ℕ) : ℕ := saturday + monday
  let total : ℕ := 390
  ∀ monday : ℕ,
    monday < saturday →
    saturday + monday + wednesday monday + friday monday = total →
    saturday - monday = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_football_game_attendance_difference_l469_46988


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l469_46930

theorem max_area_rectangle_with_fixed_perimeter :
  ∀ (width height : ℝ),
  width > 0 → height > 0 →
  width + height = 50 →
  width * height ≤ 625 :=
by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l469_46930


namespace NUMINAMATH_CALUDE_binary_100_is_4_binary_101_is_5_binary_1100_is_12_l469_46919

-- Define binary to decimal conversion function
def binaryToDecimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

-- Define decimal numbers
def four : ℕ := 4
def five : ℕ := 5
def twelve : ℕ := 12

-- Define binary numbers
def binary_100 : List Bool := [true, false, false]
def binary_101 : List Bool := [true, false, true]
def binary_1100 : List Bool := [true, true, false, false]

-- Theorem statements
theorem binary_100_is_4 : binaryToDecimal binary_100 = four := by sorry

theorem binary_101_is_5 : binaryToDecimal binary_101 = five := by sorry

theorem binary_1100_is_12 : binaryToDecimal binary_1100 = twelve := by sorry

end NUMINAMATH_CALUDE_binary_100_is_4_binary_101_is_5_binary_1100_is_12_l469_46919


namespace NUMINAMATH_CALUDE_coefficient_of_y_squared_l469_46940

def polynomial (x : ℝ) : ℝ :=
  (1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11 + x^12 - x^13 + x^14 - x^15 + x^16 - x^17)

def y (x : ℝ) : ℝ := x + 1

theorem coefficient_of_y_squared (x : ℝ) : 
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ a₁₅ a₁₆ a₁₇ : ℝ), 
    polynomial x = a₀ + a₁ * y x + a₂ * (y x)^2 + a₃ * (y x)^3 + a₄ * (y x)^4 + 
                   a₅ * (y x)^5 + a₆ * (y x)^6 + a₇ * (y x)^7 + a₈ * (y x)^8 + 
                   a₉ * (y x)^9 + a₁₀ * (y x)^10 + a₁₁ * (y x)^11 + a₁₂ * (y x)^12 + 
                   a₁₃ * (y x)^13 + a₁₄ * (y x)^14 + a₁₅ * (y x)^15 + a₁₆ * (y x)^16 + 
                   a₁₇ * (y x)^17 ∧
    a₂ = 816 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_y_squared_l469_46940


namespace NUMINAMATH_CALUDE_alberts_cabbage_rows_l469_46936

/-- Represents Albert's cabbage patch -/
structure CabbagePatch where
  total_heads : ℕ
  heads_per_row : ℕ

/-- Calculates the number of rows in the cabbage patch -/
def number_of_rows (patch : CabbagePatch) : ℕ :=
  patch.total_heads / patch.heads_per_row

/-- Theorem stating the number of rows in Albert's cabbage patch -/
theorem alberts_cabbage_rows :
  let patch : CabbagePatch := { total_heads := 180, heads_per_row := 15 }
  number_of_rows patch = 12 := by
  sorry

end NUMINAMATH_CALUDE_alberts_cabbage_rows_l469_46936


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l469_46922

theorem number_exceeding_fraction (x : ℝ) : x = (3/7 + 0.8 * (3/7)) * x → x = (35/27) * x := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l469_46922


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l469_46972

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 1 / a^2 → a^2 > 1 / a) ∧ 
  (∃ a, a^2 > 1 / a ∧ ¬(a > 1 / a^2)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l469_46972


namespace NUMINAMATH_CALUDE_football_yardage_l469_46950

theorem football_yardage (total_yardage running_yardage : ℕ) 
  (h1 : total_yardage = 150)
  (h2 : running_yardage = 90) :
  total_yardage - running_yardage = 60 := by
  sorry

end NUMINAMATH_CALUDE_football_yardage_l469_46950


namespace NUMINAMATH_CALUDE_correct_equation_representation_l469_46914

/-- Represents the boat distribution problem from the ancient Chinese text --/
def boat_distribution_problem (total_boats : ℕ) (large_boat_capacity : ℕ) (small_boat_capacity : ℕ) (total_students : ℕ) : Prop :=
  ∃ (small_boats : ℕ),
    small_boats ≤ total_boats ∧
    (small_boats * small_boat_capacity + (total_boats - small_boats) * large_boat_capacity = total_students)

/-- Theorem stating that the equation 4x + 6(8 - x) = 38 correctly represents the boat distribution problem --/
theorem correct_equation_representation :
  boat_distribution_problem 8 6 4 38 ↔ ∃ x : ℕ, 4 * x + 6 * (8 - x) = 38 :=
sorry

end NUMINAMATH_CALUDE_correct_equation_representation_l469_46914


namespace NUMINAMATH_CALUDE_cubic_identity_l469_46938

theorem cubic_identity (x : ℝ) : 
  (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l469_46938


namespace NUMINAMATH_CALUDE_middle_income_sample_size_l469_46905

/-- Calculates the number of households to be drawn from a specific income group in a stratified sample. -/
def stratifiedSampleSize (totalHouseholds : ℕ) (groupHouseholds : ℕ) (sampleSize : ℕ) : ℕ :=
  (groupHouseholds * sampleSize) / totalHouseholds

/-- Proves that the number of middle-income households in the stratified sample is 60. -/
theorem middle_income_sample_size :
  let totalHouseholds : ℕ := 600
  let middleIncomeHouseholds : ℕ := 360
  let sampleSize : ℕ := 100
  stratifiedSampleSize totalHouseholds middleIncomeHouseholds sampleSize = 60 := by
  sorry


end NUMINAMATH_CALUDE_middle_income_sample_size_l469_46905


namespace NUMINAMATH_CALUDE_abs_cubic_inequality_l469_46924

theorem abs_cubic_inequality (x : ℝ) : |x| ≤ 2 → |3*x - x^3| ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_cubic_inequality_l469_46924


namespace NUMINAMATH_CALUDE_min_value_problem_min_value_attained_l469_46902

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (x^2 / (x + 2) + y^2 / (y + 1)) ≥ 1/4 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧
    (x^2 / (x + 2) + y^2 / (y + 1)) < 1/4 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_min_value_attained_l469_46902


namespace NUMINAMATH_CALUDE_number_ratio_l469_46993

theorem number_ratio (x : ℚ) (h : 3 * (2 * x + 9) = 81) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l469_46993


namespace NUMINAMATH_CALUDE_checker_moves_10_l469_46964

/-- Represents the number of ways a checker can move n cells -/
def checkerMoves : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => checkerMoves (n + 1) + checkerMoves n

/-- Theorem stating that the number of ways a checker can move 10 cells is 89 -/
theorem checker_moves_10 : checkerMoves 10 = 89 := by
  sorry

#eval checkerMoves 10

end NUMINAMATH_CALUDE_checker_moves_10_l469_46964


namespace NUMINAMATH_CALUDE_min_blocks_for_wall_l469_46965

/-- Represents the dimensions of a block -/
structure Block where
  length : Nat
  height : Nat

/-- Represents the dimensions of the wall -/
structure Wall where
  length : Nat
  height : Nat

/-- Calculates the minimum number of blocks needed to build the wall -/
def minBlocksNeeded (wall : Wall) (blocks : List Block) : Nat :=
  sorry

/-- The theorem to be proven -/
theorem min_blocks_for_wall :
  let wall : Wall := { length := 120, height := 9 }
  let blocks : List Block := [
    { length := 3, height := 1 },
    { length := 2, height := 1 },
    { length := 1, height := 1 }
  ]
  minBlocksNeeded wall blocks = 365 := by sorry

end NUMINAMATH_CALUDE_min_blocks_for_wall_l469_46965


namespace NUMINAMATH_CALUDE_maria_friends_money_l469_46926

def problem (maria_total rene_amount : ℚ) : Prop :=
  let isha_amount := maria_total / 4
  let florence_amount := isha_amount / 2
  let john_amount := florence_amount / 3
  florence_amount = 4 * rene_amount ∧
  rene_amount = 450 ∧
  isha_amount + florence_amount + rene_amount + john_amount = 6450

theorem maria_friends_money :
  ∃ maria_total : ℚ, problem maria_total 450 :=
sorry

end NUMINAMATH_CALUDE_maria_friends_money_l469_46926


namespace NUMINAMATH_CALUDE_all_digits_appear_as_cube_units_l469_46941

theorem all_digits_appear_as_cube_units : ∀ d : Nat, d < 10 → ∃ n : Nat, n^3 % 10 = d := by
  sorry

end NUMINAMATH_CALUDE_all_digits_appear_as_cube_units_l469_46941


namespace NUMINAMATH_CALUDE_primitive_pythagorean_triple_ab_div_12_l469_46947

/-- A primitive Pythagorean triple is a tuple of positive integers (a, b, c) where a² + b² = c² and gcd(a, b, c) = 1 -/
def isPrimitivePythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧ Nat.gcd a (Nat.gcd b c) = 1

/-- For any primitive Pythagorean triple (a, b, c), ab is divisible by 12 -/
theorem primitive_pythagorean_triple_ab_div_12 (a b c : ℕ) 
  (h : isPrimitivePythagoreanTriple a b c) : 
  12 ∣ (a * b) := by
  sorry


end NUMINAMATH_CALUDE_primitive_pythagorean_triple_ab_div_12_l469_46947


namespace NUMINAMATH_CALUDE_new_speed_calculation_l469_46957

/-- Theorem: Given a distance of 630 km and an original time of 6 hours,
    if the new time is 3/2 times the original time,
    then the new speed required to cover the same distance is 70 km/h. -/
theorem new_speed_calculation (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 630 →
  original_time = 6 →
  new_time_factor = 3 / 2 →
  let new_time := original_time * new_time_factor
  let new_speed := distance / new_time
  new_speed = 70 := by
  sorry

#check new_speed_calculation

end NUMINAMATH_CALUDE_new_speed_calculation_l469_46957


namespace NUMINAMATH_CALUDE_pizza_combinations_l469_46927

theorem pizza_combinations (n k : ℕ) (h1 : n = 8) (h2 : k = 5) :
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l469_46927


namespace NUMINAMATH_CALUDE_inequality_proof_l469_46934

theorem inequality_proof (x y : ℝ) (n : ℕ+) 
  (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  (x^n.val / (1 - x^2) + y^n.val / (1 - y^2)) ≥ ((x^n.val + y^n.val) / (1 - x*y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l469_46934


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l469_46958

theorem sum_of_fifth_powers (a b u v : ℝ) 
  (h1 : a * u + b * v = 5)
  (h2 : a * u^2 + b * v^2 = 11)
  (h3 : a * u^3 + b * v^3 = 30)
  (h4 : a * u^4 + b * v^4 = 76) :
  a * u^5 + b * v^5 = 8264 / 319 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l469_46958


namespace NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_500_l469_46906

theorem percentage_of_amount (amount : ℝ) (percentage : ℝ) :
  (percentage / 100) * amount = (percentage * amount) / 100 := by sorry

theorem twenty_five_percent_of_500 :
  (25 : ℝ) / 100 * 500 = 125 := by sorry

end NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_500_l469_46906


namespace NUMINAMATH_CALUDE_circle_passes_through_points_circle_equation_l469_46949

/-- A circle passing through three given points -/
def CircleThroughThreePoints (p1 p2 p3 : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | x^2 + y^2 + D*x + E*y + F = 0}
  where
    D : ℝ := -8
    E : ℝ := 6
    F : ℝ := 0

/-- The circle passes through the given points -/
theorem circle_passes_through_points :
  let C := CircleThroughThreePoints (0, 0) (1, 1) (4, 2)
  (0, 0) ∈ C ∧ (1, 1) ∈ C ∧ (4, 2) ∈ C := by
  sorry

/-- The equation of the circle is x^2 + y^2 - 8x + 6y = 0 -/
theorem circle_equation (x y : ℝ) :
  let C := CircleThroughThreePoints (0, 0) (1, 1) (4, 2)
  (x, y) ∈ C ↔ x^2 + y^2 - 8*x + 6*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_circle_equation_l469_46949


namespace NUMINAMATH_CALUDE_milk_dilution_l469_46955

theorem milk_dilution (whole_milk : ℝ) (added_skimmed_milk : ℝ) 
  (h1 : whole_milk = 1) 
  (h2 : added_skimmed_milk = 1/4) : 
  let initial_cream := 0.05 * whole_milk
  let initial_skimmed := 0.95 * whole_milk
  let total_volume := whole_milk + added_skimmed_milk
  let final_cream_percentage := initial_cream / total_volume
  final_cream_percentage = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_milk_dilution_l469_46955


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l469_46979

theorem no_solution_quadratic_inequality :
  ∀ x : ℝ, ¬(-x^2 + 2*x - 2 > 0) := by
sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l469_46979


namespace NUMINAMATH_CALUDE_at_least_one_intersection_l469_46925

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line lying in a plane
variable (lies_in : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem at_least_one_intersection 
  (a b c : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : lies_in a α)
  (h3 : lies_in b β)
  (h4 : c = intersect α β) :
  intersects c a ∨ intersects c b :=
sorry

end NUMINAMATH_CALUDE_at_least_one_intersection_l469_46925


namespace NUMINAMATH_CALUDE_sum_of_solutions_abs_equation_l469_46996

theorem sum_of_solutions_abs_equation : 
  ∃ (x₁ x₂ : ℝ), 
    (|3 * x₁ - 5| = 8) ∧ 
    (|3 * x₂ - 5| = 8) ∧ 
    (x₁ + x₂ = 10 / 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_abs_equation_l469_46996


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l469_46945

/-- Given a boat that travels 6 km/hr along a stream and 2 km/hr against the same stream,
    its speed in still water is 4 km/hr. -/
theorem boat_speed_in_still_water (boat_speed : ℝ) (stream_speed : ℝ) : 
  (boat_speed + stream_speed = 6) → 
  (boat_speed - stream_speed = 2) → 
  boat_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l469_46945


namespace NUMINAMATH_CALUDE_total_peaches_is_273_l469_46983

/-- The number of monkeys in the zoo --/
def num_monkeys : ℕ := 36

/-- The number of peaches each monkey receives in the first scenario --/
def peaches_per_monkey_scenario1 : ℕ := 6

/-- The number of peaches left over in the first scenario --/
def peaches_left_scenario1 : ℕ := 57

/-- The number of peaches each monkey should receive in the second scenario --/
def peaches_per_monkey_scenario2 : ℕ := 9

/-- The number of monkeys that get nothing in the second scenario --/
def monkeys_with_no_peaches : ℕ := 5

/-- The number of peaches the last monkey gets in the second scenario --/
def peaches_for_last_monkey : ℕ := 3

/-- The total number of peaches --/
def total_peaches : ℕ := num_monkeys * peaches_per_monkey_scenario1 + peaches_left_scenario1

theorem total_peaches_is_273 : total_peaches = 273 := by
  sorry

#eval total_peaches

end NUMINAMATH_CALUDE_total_peaches_is_273_l469_46983
