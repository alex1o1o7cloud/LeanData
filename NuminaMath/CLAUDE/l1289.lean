import Mathlib

namespace NUMINAMATH_CALUDE_distance_product_l1289_128933

theorem distance_product (a₁ a₂ : ℝ) : 
  let p₁ := (3 * a₁, 2 * a₁ - 5)
  let p₂ := (6, -2)
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 = (3 * Real.sqrt 17)^2 →
  let p₁ := (3 * a₂, 2 * a₂ - 5)
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 = (3 * Real.sqrt 17)^2 →
  a₁ * a₂ = -2880 / 169 := by
sorry

end NUMINAMATH_CALUDE_distance_product_l1289_128933


namespace NUMINAMATH_CALUDE_custom_mul_one_one_eq_neg_eleven_l1289_128997

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 1

/-- Theorem: Given the conditions, 1 * 1 = -11 -/
theorem custom_mul_one_one_eq_neg_eleven 
  (a b : ℝ) 
  (h1 : custom_mul a b 3 5 = 15) 
  (h2 : custom_mul a b 4 7 = 28) : 
  custom_mul a b 1 1 = -11 :=
by sorry

end NUMINAMATH_CALUDE_custom_mul_one_one_eq_neg_eleven_l1289_128997


namespace NUMINAMATH_CALUDE_absolute_value_zero_l1289_128999

theorem absolute_value_zero (y : ℚ) : |2 * y - 3| = 0 ↔ y = 3/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_zero_l1289_128999


namespace NUMINAMATH_CALUDE_angle_PQT_measure_l1289_128988

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The measure of an angle in a regular octagon -/
def regular_octagon_angle : ℝ := 135

/-- The measure of angle PQT in a regular octagon -/
def angle_PQT (octagon : RegularOctagon) : ℝ :=
  22.5

theorem angle_PQT_measure (octagon : RegularOctagon) :
  angle_PQT octagon = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_PQT_measure_l1289_128988


namespace NUMINAMATH_CALUDE_cookout_attendance_l1289_128959

theorem cookout_attendance (kids_2004 kids_2005 kids_2006 : ℕ) : 
  kids_2005 = kids_2004 / 2 →
  kids_2006 = (2 * kids_2005) / 3 →
  kids_2006 = 20 →
  kids_2004 = 60 := by
  sorry

end NUMINAMATH_CALUDE_cookout_attendance_l1289_128959


namespace NUMINAMATH_CALUDE_fraction_of_as_l1289_128991

theorem fraction_of_as (total : ℝ) (as : ℝ) (bs : ℝ) : 
  total > 0 → 
  bs / total = 0.2 → 
  (as + bs) / total = 0.9 → 
  as / total = 0.7 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_as_l1289_128991


namespace NUMINAMATH_CALUDE_find_b_value_l1289_128977

theorem find_b_value (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l1289_128977


namespace NUMINAMATH_CALUDE_tan_half_angle_l1289_128929

theorem tan_half_angle (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.tan (α / 2) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_l1289_128929


namespace NUMINAMATH_CALUDE_additional_license_plates_l1289_128910

theorem additional_license_plates 
  (initial_first : Nat) 
  (initial_second : Nat) 
  (initial_third : Nat) 
  (added_letters : Nat) 
  (h1 : initial_first = 5) 
  (h2 : initial_second = 3) 
  (h3 : initial_third = 4) 
  (h4 : added_letters = 1) : 
  (initial_first + added_letters) * (initial_second + added_letters) * (initial_third + added_letters) - 
  (initial_first * initial_second * initial_third) = 60 := by
sorry

end NUMINAMATH_CALUDE_additional_license_plates_l1289_128910


namespace NUMINAMATH_CALUDE_cubic_division_theorem_l1289_128904

theorem cubic_division_theorem (c d : ℝ) (hc : c = 7) (hd : d = 3) :
  (c^3 + d^3) / (c^2 - c*d + d^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_division_theorem_l1289_128904


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l1289_128946

theorem cube_sum_divisibility (x y z t : ℤ) (h : x^3 + y^3 = 3*(z^3 + t^3)) :
  3 ∣ x ∧ 3 ∣ y := by sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l1289_128946


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_l1289_128994

/-- The plane region defined by the given inequalities -/
def PlaneRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ 4 * p.1 + 3 * p.2 - 12 ≤ 0}

/-- The circle with center (1,1) and radius 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

/-- The circle is inscribed in the plane region -/
def IsInscribed (c : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Prop :=
  c ⊆ r ∧ ∃ p q s : ℝ × ℝ, p ∈ c ∧ p ∈ r ∧ q ∈ c ∧ q ∈ r ∧ s ∈ c ∧ s ∈ r ∧
    p.1 = 0 ∧ q.2 = 0 ∧ 4 * s.1 + 3 * s.2 = 12

/-- The circle is the largest inscribed circle -/
theorem largest_inscribed_circle :
  IsInscribed Circle PlaneRegion ∧
  ∀ c : Set (ℝ × ℝ), IsInscribed c PlaneRegion → MeasureTheory.volume c ≤ MeasureTheory.volume Circle :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_l1289_128994


namespace NUMINAMATH_CALUDE_no_obtuse_right_triangle_l1289_128917

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of triangles
def Triangle.isRight (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: An obtuse right triangle cannot exist
theorem no_obtuse_right_triangle :
  ∀ t : Triangle,
  (t.angle1 + t.angle2 + t.angle3 = 180) →
  ¬(t.isRight ∧ t.isObtuse) :=
by
  sorry


end NUMINAMATH_CALUDE_no_obtuse_right_triangle_l1289_128917


namespace NUMINAMATH_CALUDE_sample_size_proof_l1289_128940

/-- Given a sample divided into 3 groups with specific frequencies, prove the sample size. -/
theorem sample_size_proof (f1 f2 f3 : ℝ) (n1 : ℝ) (h1 : f2 = 0.35) (h2 : f3 = 0.45) (h3 : n1 = 10) :
  ∃ M : ℝ, M = 50 ∧ f1 + f2 + f3 = 1 ∧ n1 / M = f1 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_proof_l1289_128940


namespace NUMINAMATH_CALUDE_compound_propositions_truth_l1289_128989

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x < y → x^2 > y^2

-- Theorem statement
theorem compound_propositions_truth (hp : p) (hq : ¬q) : 
  (p ∧ q = False) ∧ 
  (p ∨ q = True) ∧ 
  (p ∧ (¬q) = True) ∧ 
  ((¬p) ∨ q = False) := by
  sorry

end NUMINAMATH_CALUDE_compound_propositions_truth_l1289_128989


namespace NUMINAMATH_CALUDE_zeros_of_quadratic_function_l1289_128907

theorem zeros_of_quadratic_function (x : ℝ) :
  x^2 = 0 → x ∈ Set.Ioo (-1 : ℝ) 1 :=
by
  sorry

end NUMINAMATH_CALUDE_zeros_of_quadratic_function_l1289_128907


namespace NUMINAMATH_CALUDE_edward_initial_money_l1289_128958

def toy_car_cost : ℚ := 0.95
def race_track_cost : ℚ := 6.00
def num_toy_cars : ℕ := 4
def remaining_money : ℚ := 8.00

theorem edward_initial_money :
  ∃ (initial_money : ℚ),
    initial_money = num_toy_cars * toy_car_cost + race_track_cost + remaining_money :=
by
  sorry

end NUMINAMATH_CALUDE_edward_initial_money_l1289_128958


namespace NUMINAMATH_CALUDE_red_box_position_l1289_128931

/-- Given a collection of boxes with a red box among them, this function
    calculates the position of the red box from the right when arranged
    from largest to smallest, given its position from the right when
    arranged from smallest to largest. -/
def position_from_right_largest_to_smallest (total_boxes : ℕ) (position_smallest_to_largest : ℕ) : ℕ :=
  total_boxes - (position_smallest_to_largest - 1)

/-- Theorem stating that for 45 boxes with the red box 29th from the right
    when arranged smallest to largest, it will be 17th from the right
    when arranged largest to smallest. -/
theorem red_box_position (total_boxes : ℕ) (position_smallest_to_largest : ℕ) 
    (h1 : total_boxes = 45)
    (h2 : position_smallest_to_largest = 29) :
    position_from_right_largest_to_smallest total_boxes position_smallest_to_largest = 17 := by
  sorry

#eval position_from_right_largest_to_smallest 45 29

end NUMINAMATH_CALUDE_red_box_position_l1289_128931


namespace NUMINAMATH_CALUDE_increasing_sequence_count_l1289_128955

def sequence_count (n m : ℕ) : ℕ := Nat.choose (n + m - 1) m

theorem increasing_sequence_count : 
  let n := 675
  let m := 15
  sequence_count n m = Nat.choose 689 15 ∧ 689 % 1000 = 689 := by sorry

#eval sequence_count 675 15
#eval 689 % 1000

end NUMINAMATH_CALUDE_increasing_sequence_count_l1289_128955


namespace NUMINAMATH_CALUDE_enclosed_area_theorem_l1289_128938

/-- The common area enclosed by 4 equilateral triangles with side length 1, 
    each sharing a side with one of the 4 sides of a unit square. -/
def commonAreaEnclosedByTriangles : ℝ := -1

/-- The side length of the square -/
def squareSideLength : ℝ := 1

/-- The side length of each equilateral triangle -/
def triangleSideLength : ℝ := 1

/-- The number of equilateral triangles -/
def numberOfTriangles : ℕ := 4

theorem enclosed_area_theorem :
  commonAreaEnclosedByTriangles = -1 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_theorem_l1289_128938


namespace NUMINAMATH_CALUDE_smallest_number_square_and_cube_l1289_128972

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 3

theorem smallest_number_square_and_cube :
  ∃ n : ℕ, n = 72 ∧
    is_perfect_square (n * 2) ∧
    is_perfect_cube (n * 3) ∧
    ∀ m : ℕ, m < n →
      ¬(is_perfect_square (m * 2) ∧ is_perfect_cube (m * 3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_square_and_cube_l1289_128972


namespace NUMINAMATH_CALUDE_triangle_special_angle_relation_l1289_128925

/-- In a triangle ABC where α = 3β = 6γ, the equation bc² = (a+b)(a-b)² holds true. -/
theorem triangle_special_angle_relation (a b c : ℝ) (α β γ : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < α ∧ 0 < β ∧ 0 < γ →  -- positive angles
  α + β + γ = π →         -- sum of angles in a triangle
  α = 3*β →               -- given condition
  α = 6*γ →               -- given condition
  b*c^2 = (a+b)*(a-b)^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_angle_relation_l1289_128925


namespace NUMINAMATH_CALUDE_area_ABDE_l1289_128906

/-- A regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- The quadrilateral ABDE formed by four vertices of the regular hexagon -/
def ABDE (h : RegularHexagon) : Set (ℝ × ℝ) :=
  {(2, 0), (1, Real.sqrt 3), (-2, 0), (-1, -Real.sqrt 3)}

/-- The area of a set of points in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of ABDE in a regular hexagon with side length 2 is 4√3 -/
theorem area_ABDE (h : RegularHexagon) : area (ABDE h) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_ABDE_l1289_128906


namespace NUMINAMATH_CALUDE_unique_functional_equation_l1289_128921

/-- Given g: ℂ → ℂ, w ∈ ℂ, a ∈ ℂ, where w³ = 1 and w ≠ 1, 
    prove that the unique function f: ℂ → ℂ satisfying 
    f(z) + f(wz + a) = g(z) for all z ∈ ℂ 
    is given by f(z) = (g(z) + g(w²z + wa + a) - g(wz + a)) / 2 -/
theorem unique_functional_equation (g : ℂ → ℂ) (w a : ℂ) 
    (hw : w^3 = 1) (hw_neq : w ≠ 1) :
    ∃! f : ℂ → ℂ, ∀ z : ℂ, f z + f (w * z + a) = g z ∧
    f = fun z ↦ (g z + g (w^2 * z + w * a + a) - g (w * z + a)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_l1289_128921


namespace NUMINAMATH_CALUDE_equation_solution_l1289_128918

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 3 ∧ x₂ = 3/5 ∧ 
  ∀ (x : ℝ), (x - 3)^2 + 4*x*(x - 3) = 0 ↔ (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1289_128918


namespace NUMINAMATH_CALUDE_least_number_with_remainder_two_five_six_satisfies_conditions_least_number_is_256_l1289_128902

theorem least_number_with_remainder (n : ℕ) : 
  (n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4) →
  n ≥ 256 :=
by sorry

theorem two_five_six_satisfies_conditions : 
  (256 % 7 = 4) ∧ (256 % 9 = 4) ∧ (256 % 12 = 4) ∧ (256 % 18 = 4) :=
by sorry

theorem least_number_is_256 : 
  ∀ n : ℕ, n < 256 → 
  ¬((n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4)) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_two_five_six_satisfies_conditions_least_number_is_256_l1289_128902


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1289_128916

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 5) * (Real.sqrt 8 / Real.sqrt 10) * (Real.sqrt 15 / Real.sqrt 21) = 
  (2 * Real.sqrt 105) / 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1289_128916


namespace NUMINAMATH_CALUDE_star_running_back_yardage_l1289_128919

/-- Represents the yardage gained by a player in a football game -/
structure Yardage where
  running : ℕ
  catching : ℕ

/-- Calculates the total yardage for a player -/
def totalYardage (y : Yardage) : ℕ :=
  y.running + y.catching

/-- Theorem: The total yardage of a player who gained 90 yards running and 60 yards catching is 150 yards -/
theorem star_running_back_yardage :
  let y : Yardage := { running := 90, catching := 60 }
  totalYardage y = 150 := by
  sorry

end NUMINAMATH_CALUDE_star_running_back_yardage_l1289_128919


namespace NUMINAMATH_CALUDE_fraction_equality_l1289_128951

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := a * b - b^3

-- Define the # operation
def hash_op (a b : ℚ) : ℚ := a^2 + b - a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 4 3) / (hash_op 4 3) = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1289_128951


namespace NUMINAMATH_CALUDE_min_value_of_function_l1289_128901

theorem min_value_of_function (x : ℝ) (h : x ≥ 2) :
  x + 5 / (x + 1) ≥ 11 / 3 ∧
  (x + 5 / (x + 1) = 11 / 3 ↔ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1289_128901


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l1289_128930

theorem quadratic_form_minimum (x y : ℝ) : 3*x^2 + 2*x*y + y^2 - 6*x + 4*y + 9 ≥ 0 ∧
  ∃ (x₀ y₀ : ℝ), 3*x₀^2 + 2*x₀*y₀ + y₀^2 - 6*x₀ + 4*y₀ + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l1289_128930


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_is_three_tenths_l1289_128922

def total_students : ℕ := 5
def students_to_select : ℕ := 3

def probability_A_and_B_selected : ℚ :=
  (total_students - students_to_select + 1 : ℚ) / total_students.choose students_to_select

theorem probability_A_and_B_selected_is_three_tenths :
  probability_A_and_B_selected = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_and_B_selected_is_three_tenths_l1289_128922


namespace NUMINAMATH_CALUDE_seller_deception_l1289_128939

theorem seller_deception (a w : ℝ) (ha : a > 0) (hw : w > 0) (ha_neq_1 : a ≠ 1) :
  (a * w + w / a) / 2 ≥ w ∧
  ((a * w + w / a) / 2 = w ↔ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_seller_deception_l1289_128939


namespace NUMINAMATH_CALUDE_no_solution_equation_l1289_128965

theorem no_solution_equation : ¬∃ (x : ℝ), x - 9 / (x - 5) = 5 - 9 / (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l1289_128965


namespace NUMINAMATH_CALUDE_largest_k_value_l1289_128962

/-- A function that splits the whole numbers from 1 to 2k into two groups -/
def split_numbers (k : ℕ) : (Fin (2 * k) → Bool) := sorry

/-- A predicate that checks if two numbers share more than two distinct prime factors -/
def share_more_than_two_prime_factors (a b : ℕ) : Prop := sorry

/-- The main theorem stating that 44 is the largest possible value of k -/
theorem largest_k_value : 
  ∀ k : ℕ, k > 44 → 
  ¬∃ (f : Fin (2 * k) → Bool), 
    (∀ i j : Fin (2 * k), i.val < j.val ∧ f i = f j → 
      ¬share_more_than_two_prime_factors (i.val + 1) (j.val + 1)) ∧
    (Fintype.card {i : Fin (2 * k) | f i = true} = k) :=
sorry

end NUMINAMATH_CALUDE_largest_k_value_l1289_128962


namespace NUMINAMATH_CALUDE_price_of_pants_l1289_128913

/-- Given Iris's shopping trip to the mall, this theorem proves the price of each pair of pants. -/
theorem price_of_pants (jacket_price : ℕ) (shorts_price : ℕ) (total_spent : ℕ) 
  (jacket_count : ℕ) (shorts_count : ℕ) (pants_count : ℕ) :
  jacket_price = 10 →
  shorts_price = 6 →
  jacket_count = 3 →
  shorts_count = 2 →
  pants_count = 4 →
  total_spent = 90 →
  ∃ (pants_price : ℕ), 
    pants_price * pants_count + jacket_price * jacket_count + shorts_price * shorts_count = total_spent ∧
    pants_price = 12 :=
by sorry

end NUMINAMATH_CALUDE_price_of_pants_l1289_128913


namespace NUMINAMATH_CALUDE_point_distance_ratio_l1289_128985

theorem point_distance_ratio (x : ℝ) : 
  let P : ℝ × ℝ := (x, -5)
  (P.1)^2 + (P.2)^2 = 10^2 → 
  (abs P.2) / 10 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_point_distance_ratio_l1289_128985


namespace NUMINAMATH_CALUDE_cosine_of_inclination_angle_l1289_128974

/-- A line in 2D space represented by its parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The inclination angle of a line -/
def inclinationAngle (l : ParametricLine) : ℝ := sorry

/-- The given line with parametric equations x = -2 + 3t and y = 3 - 4t -/
def givenLine : ParametricLine := {
  x := λ t => -2 + 3*t,
  y := λ t => 3 - 4*t
}

/-- Theorem stating that the cosine of the inclination angle of the given line is -3/5 -/
theorem cosine_of_inclination_angle :
  Real.cos (inclinationAngle givenLine) = -3/5 := by sorry

end NUMINAMATH_CALUDE_cosine_of_inclination_angle_l1289_128974


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1289_128996

theorem sufficient_not_necessary_condition (a : ℝ) (h : a > 0) :
  (∀ a, a ≥ 1 → a + 1/a ≥ 2) ∧
  (∃ a, 0 < a ∧ a < 1 ∧ a + 1/a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1289_128996


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l1289_128995

/-- The number of red jelly beans -/
def red_beans : ℕ := 10

/-- The number of green jelly beans -/
def green_beans : ℕ := 12

/-- The number of yellow jelly beans -/
def yellow_beans : ℕ := 13

/-- The number of blue jelly beans -/
def blue_beans : ℕ := 15

/-- The number of purple jelly beans -/
def purple_beans : ℕ := 5

/-- The total number of jelly beans -/
def total_beans : ℕ := red_beans + green_beans + yellow_beans + blue_beans + purple_beans

/-- The number of blue and purple jelly beans combined -/
def blue_and_purple : ℕ := blue_beans + purple_beans

/-- The probability of selecting either a blue or purple jelly bean -/
def probability : ℚ := blue_and_purple / total_beans

theorem jelly_bean_probability : probability = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l1289_128995


namespace NUMINAMATH_CALUDE_profit_percentage_l1289_128963

theorem profit_percentage (C P : ℝ) (h : (2/3) * P = 0.95 * C) : 
  (P - C) / C * 100 = 42.5 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l1289_128963


namespace NUMINAMATH_CALUDE_class_size_l1289_128979

theorem class_size (mini_cupcakes : ℕ) (donut_holes : ℕ) (desserts_per_student : ℕ) : 
  mini_cupcakes = 14 → 
  donut_holes = 12 → 
  desserts_per_student = 2 → 
  (mini_cupcakes + donut_holes) / desserts_per_student = 13 := by
sorry

end NUMINAMATH_CALUDE_class_size_l1289_128979


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l1289_128956

theorem estimate_sqrt_expression :
  5 < Real.sqrt (1/3) * Real.sqrt 27 + Real.sqrt 7 ∧
  Real.sqrt (1/3) * Real.sqrt 27 + Real.sqrt 7 < 6 :=
by sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l1289_128956


namespace NUMINAMATH_CALUDE_arccos_cos_gt_arcsin_sin_iff_l1289_128915

theorem arccos_cos_gt_arcsin_sin_iff (x : ℝ) : 
  (∃ k : ℤ, 2 * k * Real.pi + Real.pi / 2 < x ∧ x < 2 * (k + 1) * Real.pi) ↔ 
  Real.arccos (Real.cos x) > Real.arcsin (Real.sin x) :=
sorry

end NUMINAMATH_CALUDE_arccos_cos_gt_arcsin_sin_iff_l1289_128915


namespace NUMINAMATH_CALUDE_ab_value_l1289_128912

theorem ab_value (a b : ℝ) 
  (h : ∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) : 
  a * b = -1 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l1289_128912


namespace NUMINAMATH_CALUDE_calculate_expression_l1289_128927

theorem calculate_expression : (10^10 / (2 * 10^6)) * 3 = 15000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1289_128927


namespace NUMINAMATH_CALUDE_conic_eccentricity_l1289_128944

/-- The eccentricity of a conic section x + y^2/m = 1, where m is the geometric mean of 2 and 8 -/
theorem conic_eccentricity (m : ℝ) : 
  (m^2 = 2 * 8) →  -- m is the geometric mean of 2 and 8
  (∃ (e : ℝ), (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧ 
    ∃ (a b c : ℝ), (a > 0 ∧ b > 0 ∧ c > 0) ∧
      ((x + y^2/m = 1) → (e = c/a ∧ (a^2 = b^2 + c^2 ∨ a^2 + b^2 = c^2)))) :=
by sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l1289_128944


namespace NUMINAMATH_CALUDE_tempo_insurance_premium_l1289_128960

/-- Calculate the premium amount for a tempo insurance --/
theorem tempo_insurance_premium 
  (original_value : ℝ) 
  (insurance_extent : ℝ) 
  (premium_rate : ℝ) 
  (h1 : original_value = 14000)
  (h2 : insurance_extent = 5/7)
  (h3 : premium_rate = 3/100) : 
  original_value * insurance_extent * premium_rate = 300 := by
  sorry

end NUMINAMATH_CALUDE_tempo_insurance_premium_l1289_128960


namespace NUMINAMATH_CALUDE_range_of_m_when_not_two_distinct_positive_roots_l1289_128983

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + m*x + 1 = 0

-- Define the condition for two distinct positive real roots
def has_two_distinct_positive_roots (m : ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y > 0 ∧ x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- The theorem to prove
theorem range_of_m_when_not_two_distinct_positive_roots :
  {m : ℝ | ¬(has_two_distinct_positive_roots m)} = Set.Ici (-2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_when_not_two_distinct_positive_roots_l1289_128983


namespace NUMINAMATH_CALUDE_military_unit_reorganization_l1289_128982

theorem military_unit_reorganization (x : ℕ) : 
  (x * (x + 5) = 5 * (x + 845)) → 
  (x * (x + 5) = 4550) := by
  sorry

end NUMINAMATH_CALUDE_military_unit_reorganization_l1289_128982


namespace NUMINAMATH_CALUDE_triangle_side_length_l1289_128984

/-- Given an acute triangle ABC with sides a, b, and c, 
    if a = 4, b = 3, and the area is 3√3, then c = √13 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  a = 4 → 
  b = 3 → 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 →
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  c = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1289_128984


namespace NUMINAMATH_CALUDE_sum_of_extrema_l1289_128942

theorem sum_of_extrema (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a^2 + b^2 + c^2 = 11) :
  ∃ (m M : ℝ), (∀ x, (∃ y z, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 11) → m ≤ x ∧ x ≤ M) ∧
                m + M = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_extrema_l1289_128942


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1289_128941

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | 0 ≤ x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1289_128941


namespace NUMINAMATH_CALUDE_james_stickers_after_birthday_l1289_128936

/-- The number of stickers James had after his birthday -/
def total_stickers (initial : ℕ) (birthday : ℕ) : ℕ :=
  initial + birthday

/-- Theorem stating that James had 61 stickers after his birthday -/
theorem james_stickers_after_birthday :
  total_stickers 39 22 = 61 := by
  sorry

end NUMINAMATH_CALUDE_james_stickers_after_birthday_l1289_128936


namespace NUMINAMATH_CALUDE_function_properties_l1289_128980

def f (x m : ℝ) : ℝ := |x - m| - |x + 3 * m|

theorem function_properties (m : ℝ) (h : m > 0) :
  (m = 1 → {x : ℝ | f x m ≥ 1} = {x : ℝ | x ≤ -3/2}) ∧
  ({m : ℝ | ∀ x t : ℝ, f x m < |2 + t| + |t - 1|} = {m : ℝ | 0 < m ∧ m < 3/4}) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1289_128980


namespace NUMINAMATH_CALUDE_total_glue_blobs_is_96_l1289_128961

/-- Represents a layer in the pyramid --/
structure Layer where
  size : Nat
  deriving Repr

/-- Calculates the number of internal glue blobs within a layer --/
def internalGlueBlobs (layer : Layer) : Nat :=
  2 * layer.size * (layer.size - 1)

/-- Calculates the number of glue blobs between two adjacent layers --/
def interlayerGlueBlobs (upper : Layer) (lower : Layer) : Nat :=
  upper.size * upper.size * 4

/-- The pyramid structure --/
def pyramid : List Layer := [
  { size := 4 },
  { size := 3 },
  { size := 2 },
  { size := 1 }
]

/-- Theorem: The total number of glue blobs in the pyramid is 96 --/
theorem total_glue_blobs_is_96 : 
  (pyramid.map internalGlueBlobs).sum + 
  (List.zipWith interlayerGlueBlobs pyramid.tail pyramid).sum = 96 := by
  sorry

#eval (pyramid.map internalGlueBlobs).sum + 
      (List.zipWith interlayerGlueBlobs pyramid.tail pyramid).sum

end NUMINAMATH_CALUDE_total_glue_blobs_is_96_l1289_128961


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1289_128969

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 + a 8 = 2 →
  a 6 * a 7 = -8 →
  a 2 + a 11 = -7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1289_128969


namespace NUMINAMATH_CALUDE_identify_tasty_candies_l1289_128949

/-- Represents a candy on the table. -/
structure Candy where
  tasty : Bool

/-- Represents the state of the game. -/
structure GameState where
  candies : Finset Candy
  moves_left : Nat

/-- Represents a query about a subset of candies. -/
def Query := Finset Candy → Nat

/-- The main theorem stating that all tasty candies can be identified within the given number of moves. -/
theorem identify_tasty_candies 
  (n : Nat) 
  (candies : Finset Candy) 
  (h1 : candies.card = 28) 
  (query : Query) : 
  (∃ (strategy : GameState → Finset Candy), 
    (∀ (gs : GameState), 
      gs.candies = candies → 
      gs.moves_left ≥ 21 → 
      strategy gs = {c ∈ candies | c.tasty})) ∧ 
    (∃ (strategy : GameState → Finset Candy), 
      (∀ (gs : GameState), 
        gs.candies = candies → 
        gs.moves_left ≥ 20 → 
        strategy gs = {c ∈ candies | c.tasty})) :=
by sorry

end NUMINAMATH_CALUDE_identify_tasty_candies_l1289_128949


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1289_128952

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  d^2 / 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1289_128952


namespace NUMINAMATH_CALUDE_expected_hit_targets_bound_l1289_128924

theorem expected_hit_targets_bound (n : ℕ) (hn : n > 0) :
  let p := 1 - (1 - 1 / n)^n
  n * p ≥ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_hit_targets_bound_l1289_128924


namespace NUMINAMATH_CALUDE_sock_pair_count_l1289_128975

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_pairs (white brown blue : ℕ) : ℕ :=
  white * brown + white * blue + brown * blue

/-- Theorem: The number of ways to choose a pair of socks of different colors
    from a drawer containing 5 white, 4 brown, and 3 blue distinguishable socks
    is equal to 47. -/
theorem sock_pair_count :
  different_color_pairs 5 4 3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l1289_128975


namespace NUMINAMATH_CALUDE_arithmetic_progression_product_l1289_128945

theorem arithmetic_progression_product (a₁ a₂ a₃ a₄ d : ℕ) : 
  a₁ * a₂ * a₃ = 6 ∧ 
  a₁ * a₂ * a₃ * a₄ = 24 ∧ 
  a₂ = a₁ + d ∧ 
  a₃ = a₁ + 2 * d ∧ 
  a₄ = a₁ + 3 * d ↔ 
  a₁ = 1 ∧ a₂ = 2 ∧ a₃ = 3 ∧ a₄ = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_product_l1289_128945


namespace NUMINAMATH_CALUDE_max_area_between_lines_l1289_128953

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 16

-- Define the area function
def area (x₀ : ℝ) : ℝ :=
  2 * (-2 * x₀^2 + 32 - 4 * x₀)

-- State the theorem
theorem max_area_between_lines :
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-3) 1 ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc (-3) 1 → area x ≤ area x₀) ∧
  area x₀ = 68 := by
  sorry

end NUMINAMATH_CALUDE_max_area_between_lines_l1289_128953


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1289_128950

theorem complex_expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1.22 * (((Real.sqrt a + Real.sqrt b)^2 - 4*b) / ((a - b) / (Real.sqrt (1/b) + 3 * Real.sqrt (1/a)))) / 
  ((a + 9*b + 6 * Real.sqrt (a*b)) / (1 / Real.sqrt a + 1 / Real.sqrt b))) = 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1289_128950


namespace NUMINAMATH_CALUDE_identify_liars_in_two_questions_l1289_128978

/-- Represents a person who can be either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- Represents a position on a regular decagon -/
structure Position :=
  (angle : ℝ)

/-- Represents the state of the problem -/
structure DecagonState :=
  (people : Fin 10 → Person)
  (positions : Fin 10 → Position)

/-- Represents a question asked by the traveler -/
structure Question :=
  (position : Position)

/-- Represents an answer given by a person -/
structure Answer :=
  (distance : ℝ)

/-- Function to determine the answer given by a person -/
def getAnswer (state : DecagonState) (person : Fin 10) (q : Question) : Answer :=
  sorry

/-- Function to determine if a person is a liar based on their answer -/
def isLiar (state : DecagonState) (person : Fin 10) (q : Question) (a : Answer) : Bool :=
  sorry

/-- Theorem stating that at most 2 questions are needed to identify all liars -/
theorem identify_liars_in_two_questions (state : DecagonState) :
  ∃ (q1 q2 : Question), ∀ (person : Fin 10),
    isLiar state person q1 (getAnswer state person q1) ∨
    isLiar state person q2 (getAnswer state person q2) =
    (state.people person = Person.Liar) :=
  sorry

end NUMINAMATH_CALUDE_identify_liars_in_two_questions_l1289_128978


namespace NUMINAMATH_CALUDE_ice_cream_cost_l1289_128971

theorem ice_cream_cost (two_cones_cost : ℕ) (h : two_cones_cost = 198) : 
  two_cones_cost / 2 = 99 := by sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l1289_128971


namespace NUMINAMATH_CALUDE_probability_both_selected_l1289_128928

/-- The probability of both Ram and Ravi being selected in an exam -/
theorem probability_both_selected (prob_ram prob_ravi : ℚ) 
  (h_ram : prob_ram = 4 / 7)
  (h_ravi : prob_ravi = 1 / 5) :
  prob_ram * prob_ravi = 4 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l1289_128928


namespace NUMINAMATH_CALUDE_fifteenth_term_is_101_l1289_128966

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The 15th term of the arithmetic sequence with first term 3 and common difference 7 is 101 -/
theorem fifteenth_term_is_101 : arithmeticSequence 3 7 15 = 101 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_is_101_l1289_128966


namespace NUMINAMATH_CALUDE_percentage_of_english_books_l1289_128943

theorem percentage_of_english_books (total_books : ℕ) 
  (english_books_outside : ℕ) (percentage_published_in_country : ℚ) :
  total_books = 2300 →
  english_books_outside = 736 →
  percentage_published_in_country = 60 / 100 →
  (english_books_outside / (1 - percentage_published_in_country)) / total_books = 80 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_english_books_l1289_128943


namespace NUMINAMATH_CALUDE_club_group_size_theorem_l1289_128914

theorem club_group_size_theorem (N : ℕ) (x : ℕ) 
  (h1 : 20 < N ∧ N < 50) 
  (h2 : (N - 5) % 6 = 0 ∧ (N - 5) % 7 = 0) 
  (h3 : N % x = 7) : 
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_club_group_size_theorem_l1289_128914


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1289_128970

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : a + b = 7 * (a - b)) (h5 : a^2 + b^2 = 85) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1289_128970


namespace NUMINAMATH_CALUDE_mixed_repeating_decimal_denominator_divisibility_l1289_128968

/-- Represents a mixed repeating decimal -/
structure MixedRepeatingDecimal where
  non_repeating : ℕ
  repeating : ℕ

/-- Theorem: For any mixed repeating decimal that can be expressed as an irreducible fraction p/q,
    the denominator q is divisible by 2 or 5, or both. -/
theorem mixed_repeating_decimal_denominator_divisibility
  (x : MixedRepeatingDecimal)
  (p q : ℕ)
  (h_irreducible : Nat.Coprime p q)
  (h_fraction : (p : ℚ) / q = x.non_repeating + (x.repeating : ℚ) / (10^x.non_repeating.succ * (10^x.repeating.succ - 1))) :
  2 ∣ q ∨ 5 ∣ q :=
sorry

end NUMINAMATH_CALUDE_mixed_repeating_decimal_denominator_divisibility_l1289_128968


namespace NUMINAMATH_CALUDE_function_properties_l1289_128964

def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - m

theorem function_properties (m : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f m y ≤ f m x ∧ f m x = 0) →
  (m = 0 ∨ m = 4) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, ∀ y ∈ Set.Icc (-1 : ℝ) 0, x ≤ y → f m x ≥ f m y) →
  (m ≤ -2) ∧
  (Set.range (f m) = Set.Icc 2 3) ↔ m = 6 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1289_128964


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1289_128954

theorem interest_rate_calculation (total : ℝ) (part2 : ℝ) (years1 : ℝ) (years2 : ℝ) (rate2 : ℝ) :
  total = 2665 →
  part2 = 1332.5 →
  years1 = 5 →
  years2 = 3 →
  rate2 = 0.05 →
  let part1 := total - part2
  let r := (part2 * rate2 * years2) / (part1 * years1)
  r = 0.03 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1289_128954


namespace NUMINAMATH_CALUDE_vector_sum_example_l1289_128990

theorem vector_sum_example : 
  (5 : ℝ) • (1 : Fin 3 → ℝ) 0 + (-3 : ℝ) • (1 : Fin 3 → ℝ) 1 + (2 : ℝ) • (1 : Fin 3 → ℝ) 2 + 
  (-4 : ℝ) • (1 : Fin 3 → ℝ) 0 + (8 : ℝ) • (1 : Fin 3 → ℝ) 1 + (-1 : ℝ) • (1 : Fin 3 → ℝ) 2 = 
  (1 : ℝ) • (1 : Fin 3 → ℝ) 0 + (5 : ℝ) • (1 : Fin 3 → ℝ) 1 + (1 : ℝ) • (1 : Fin 3 → ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_example_l1289_128990


namespace NUMINAMATH_CALUDE_pencil_sales_problem_l1289_128932

/-- The number of pencils initially sold for a rupee -/
def initial_pencils : ℝ := 11

/-- The number of pencils sold for a rupee to achieve a 20% gain -/
def gain_pencils : ℝ := 8.25

/-- The loss percentage when selling the initial number of pencils -/
def loss_percentage : ℝ := 10

/-- The gain percentage when selling 8.25 pencils -/
def gain_percentage : ℝ := 20

theorem pencil_sales_problem :
  (1 = (1 - loss_percentage / 100) * initial_pencils * (1 / gain_pencils)) ∧
  (1 = (1 + gain_percentage / 100) * 1) ∧
  initial_pencils = 11 := by sorry

end NUMINAMATH_CALUDE_pencil_sales_problem_l1289_128932


namespace NUMINAMATH_CALUDE_minimum_boxes_required_l1289_128993

structure BoxType where
  capacity : ℕ
  quantity : ℕ

def total_brochures : ℕ := 10000

def small_box : BoxType := ⟨50, 40⟩
def medium_box : BoxType := ⟨200, 25⟩
def large_box : BoxType := ⟨500, 10⟩

def box_types : List BoxType := [small_box, medium_box, large_box]

def can_ship (boxes : List (BoxType × ℕ)) : Prop :=
  (boxes.map (λ (b, n) => b.capacity * n)).sum ≥ total_brochures

theorem minimum_boxes_required :
  ∃ (boxes : List (BoxType × ℕ)),
    (boxes.map Prod.snd).sum = 35 ∧
    can_ship boxes ∧
    ∀ (other_boxes : List (BoxType × ℕ)),
      can_ship other_boxes →
      (other_boxes.map Prod.snd).sum ≥ 35 :=
sorry

end NUMINAMATH_CALUDE_minimum_boxes_required_l1289_128993


namespace NUMINAMATH_CALUDE_modified_short_bingo_first_column_l1289_128935

/-- The number of elements in the set from which we select numbers -/
def n : ℕ := 15

/-- The number of elements we select -/
def k : ℕ := 5

/-- The number of ways to select k distinct numbers from a set of n numbers, where order matters -/
def permutations (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

theorem modified_short_bingo_first_column : permutations n k = 360360 := by
  sorry

end NUMINAMATH_CALUDE_modified_short_bingo_first_column_l1289_128935


namespace NUMINAMATH_CALUDE_transformation_result_l1289_128992

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Translates a point to the right by a given amount -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

/-- The initial point -/
def initial_point : ℝ × ℝ := (3, -4)

/-- The final point after transformations -/
def final_point : ℝ × ℝ := (2, 4)

theorem transformation_result :
  translate_right (reflect_x (reflect_y initial_point)) 5 = final_point := by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l1289_128992


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1289_128937

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular nine-sided polygon contains 36 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 36 := by sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1289_128937


namespace NUMINAMATH_CALUDE_ellipse_constant_dot_product_l1289_128967

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the moving line
def moving_line (k x y : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product of vectors MA and MB
def dot_product_MA_MB (m x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 - m) * (x2 - m) + y1 * y2

-- Statement of the theorem
theorem ellipse_constant_dot_product :
  ∃ (m : ℝ), 
    (∀ (k x1 y1 x2 y2 : ℝ), k ≠ 0 →
      ellipse_C x1 y1 → ellipse_C x2 y2 →
      moving_line k x1 y1 → moving_line k x2 y2 →
      dot_product_MA_MB m x1 y1 x2 y2 = -7/16) ∧
    m = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_constant_dot_product_l1289_128967


namespace NUMINAMATH_CALUDE_solution_in_fourth_quadrant_l1289_128900

-- Define the equation system
def equation_system (x y : ℝ) : Prop :=
  y = 2 * x - 5 ∧ y = -x + 1

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Theorem statement
theorem solution_in_fourth_quadrant :
  ∃ x y : ℝ, equation_system x y ∧ fourth_quadrant x y :=
sorry

end NUMINAMATH_CALUDE_solution_in_fourth_quadrant_l1289_128900


namespace NUMINAMATH_CALUDE_trapezoid_BE_length_l1289_128986

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a trapezoid ABCD with point F outside -/
structure Trapezoid :=
  (A B C D F : Point)
  (is_trapezoid : sorry)  -- Condition that ABCD is a trapezoid
  (F_on_AD_extension : sorry)  -- Condition that F is on the extension of AD

/-- Given a trapezoid, find point E on AC such that E is on BF -/
def find_E (t : Trapezoid) : Point :=
  sorry

/-- Given a trapezoid, find point G on the extension of DC such that FG is parallel to BC -/
def find_G (t : Trapezoid) : Point :=
  sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

theorem trapezoid_BE_length (t : Trapezoid) :
  let E := find_E t
  let G := find_G t
  distance t.B E = 30 :=
  sorry

end NUMINAMATH_CALUDE_trapezoid_BE_length_l1289_128986


namespace NUMINAMATH_CALUDE_coin_piles_theorem_l1289_128976

/-- Represents the number of coins in each pile -/
structure CoinPiles :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Performs the coin transfers as described in the problem -/
def transfer (piles : CoinPiles) : CoinPiles :=
  let step1 := CoinPiles.mk (piles.first - piles.second) (piles.second + piles.second) piles.third
  let step2 := CoinPiles.mk step1.first (step1.second - step1.third) (step1.third + step1.third)
  CoinPiles.mk (step2.first + step2.third) step2.second (step2.third - step2.first)

/-- The main theorem stating the original number of coins in each pile -/
theorem coin_piles_theorem (piles : CoinPiles) :
  transfer piles = CoinPiles.mk 16 16 16 →
  piles = CoinPiles.mk 22 14 12 :=
by sorry

end NUMINAMATH_CALUDE_coin_piles_theorem_l1289_128976


namespace NUMINAMATH_CALUDE_hyperbola_center_is_correct_l1289_128973

/-- The center of a hyperbola given by the equation 9x^2 - 54x - 36y^2 + 288y - 576 = 0 -/
def hyperbola_center : ℝ × ℝ := (3, 4)

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

/-- Theorem stating that the center of the hyperbola is (3, 4) -/
theorem hyperbola_center_is_correct :
  let (h₁, h₂) := hyperbola_center
  ∀ (ε : ℝ), ε ≠ 0 →
    ∃ (δ : ℝ), δ > 0 ∧
      ∀ (x y : ℝ),
        hyperbola_equation x y →
        (x - h₁)^2 + (y - h₂)^2 < δ^2 →
        (x - h₁)^2 + (y - h₂)^2 < ε^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_correct_l1289_128973


namespace NUMINAMATH_CALUDE_correct_transformation_l1289_128911

theorem correct_transformation (x : ℝ) : 2*x - 5 = 3*x + 3 → 2*x - 3*x = 3 + 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l1289_128911


namespace NUMINAMATH_CALUDE_suv_max_distance_l1289_128903

/-- Represents the fuel efficiency of an SUV in miles per gallon -/
structure FuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
def max_distance (efficiency : FuelEfficiency) (fuel : ℝ) : ℝ :=
  efficiency.highway * fuel

/-- Theorem: The maximum distance an SUV with 12.2 mpg highway efficiency can travel on 23 gallons of fuel is 280.6 miles -/
theorem suv_max_distance :
  let suv_efficiency : FuelEfficiency := { highway := 12.2, city := 7.6 }
  let available_fuel : ℝ := 23
  max_distance suv_efficiency available_fuel = 280.6 := by
  sorry


end NUMINAMATH_CALUDE_suv_max_distance_l1289_128903


namespace NUMINAMATH_CALUDE_gcd_powers_of_two_l1289_128926

theorem gcd_powers_of_two : Nat.gcd (2^2025 - 1) (2^2007 - 1) = 2^18 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_powers_of_two_l1289_128926


namespace NUMINAMATH_CALUDE_angle_with_complement_33_percent_of_supplement_is_45_degrees_l1289_128998

theorem angle_with_complement_33_percent_of_supplement_is_45_degrees (x : ℝ) :
  (90 - x = (1 / 3) * (180 - x)) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_33_percent_of_supplement_is_45_degrees_l1289_128998


namespace NUMINAMATH_CALUDE_vector_operation_result_unique_linear_combination_l1289_128948

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (5, 6)

-- Theorem for part 1
theorem vector_operation_result : 
  (3 • a) + b - (2 • c) = (-2, -4) := by sorry

-- Theorem for part 2
theorem unique_linear_combination :
  ∃! (m n : ℝ), c = m • a + n • b ∧ m = 2 ∧ n = 1 := by sorry

-- Note: • is used for scalar multiplication in Lean

end NUMINAMATH_CALUDE_vector_operation_result_unique_linear_combination_l1289_128948


namespace NUMINAMATH_CALUDE_x_minus_y_equals_negative_three_l1289_128957

theorem x_minus_y_equals_negative_three
  (eq1 : 2020 * x + 2024 * y = 2028)
  (eq2 : 2022 * x + 2026 * y = 2030)
  : x - y = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_negative_three_l1289_128957


namespace NUMINAMATH_CALUDE_perpendicular_implies_m_eq_neg_one_parallel_implies_m_eq_neg_one_l1289_128981

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0

-- Define perpendicularity of lines
def perpendicular (m : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ m x₁ y₁ → l₂ m x₂ y₂ → (m - 2) / 3 * m = -1

-- Define parallelism of lines
def parallel (m : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ m x₁ y₁ → l₂ m x₂ y₂ → (m - 2) / 3 = m

-- Theorem for perpendicular case
theorem perpendicular_implies_m_eq_neg_one :
  ∀ m : ℝ, perpendicular m → m = -1 := by sorry

-- Theorem for parallel case
theorem parallel_implies_m_eq_neg_one :
  ∀ m : ℝ, parallel m → m = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_implies_m_eq_neg_one_parallel_implies_m_eq_neg_one_l1289_128981


namespace NUMINAMATH_CALUDE_shaded_area_in_square_l1289_128909

/-- The area of a symmetric shaded region in a square -/
theorem shaded_area_in_square (square_side : ℝ) (point_A_x : ℝ) (point_B_x : ℝ) : 
  square_side = 10 →
  point_A_x = 7.5 →
  point_B_x = 7.5 →
  let shaded_area := 2 * (1/2 * (square_side/4) * (square_side/2))
  shaded_area = 28.125 := by sorry

end NUMINAMATH_CALUDE_shaded_area_in_square_l1289_128909


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1289_128905

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | x < 1}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1289_128905


namespace NUMINAMATH_CALUDE_one_fourth_of_ten_times_twelve_divided_by_two_l1289_128923

theorem one_fourth_of_ten_times_twelve_divided_by_two : (1 / 4 : ℚ) * ((10 * 12) / 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_ten_times_twelve_divided_by_two_l1289_128923


namespace NUMINAMATH_CALUDE_area_equality_l1289_128908

-- Define the points
variable (A B C D M N K L : Plane)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : Plane) : Prop := sorry

-- Define that M is on AB and N is on CD
def on_segment (P Q R : Plane) : Prop := sorry

-- Define the ratio condition
def ratio_condition (A M B C N D : Plane) : Prop := sorry

-- Define the intersection points
def intersect_at (P Q R S T : Plane) : Prop := sorry

-- Define the area of a polygon
def area (points : List Plane) : ℝ := sorry

-- Theorem statement
theorem area_equality 
  (h1 : is_quadrilateral A B C D)
  (h2 : on_segment A M B)
  (h3 : on_segment C N D)
  (h4 : ratio_condition A M B C N D)
  (h5 : intersect_at A N D M K)
  (h6 : intersect_at B N C M L) :
  area [K, M, L, N] = area [A, D, K] + area [B, C, L] := by sorry

end NUMINAMATH_CALUDE_area_equality_l1289_128908


namespace NUMINAMATH_CALUDE_tax_percentage_calculation_l1289_128987

def annual_salary : ℝ := 40000
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800
def take_home_pay : ℝ := 27200

theorem tax_percentage_calculation :
  let healthcare_deduction := annual_salary * healthcare_rate
  let total_non_tax_deductions := healthcare_deduction + union_dues
  let amount_before_taxes := annual_salary - total_non_tax_deductions
  let tax_deduction := amount_before_taxes - take_home_pay
  let tax_percentage := (tax_deduction / annual_salary) * 100
  tax_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_tax_percentage_calculation_l1289_128987


namespace NUMINAMATH_CALUDE_max_player_salary_max_salary_is_512000_l1289_128934

/-- The maximum possible salary for a single player in a minor league soccer team -/
theorem max_player_salary (n : ℕ) (min_salary : ℕ) (max_total : ℕ) : ℕ :=
  let max_single_salary := max_total - (n - 1) * min_salary
  max_single_salary

/-- The maximum possible salary for a single player in the given scenario is $512,000 -/
theorem max_salary_is_512000 :
  max_player_salary 25 12000 800000 = 512000 := by
  sorry

end NUMINAMATH_CALUDE_max_player_salary_max_salary_is_512000_l1289_128934


namespace NUMINAMATH_CALUDE_ice_cream_bar_price_l1289_128947

theorem ice_cream_bar_price 
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℚ)
  (sundae_price : ℚ)
  (h1 : num_ice_cream_bars = 125)
  (h2 : num_sundaes = 125)
  (h3 : total_price = 225)
  (h4 : sundae_price = 6/5) :
  (total_price - num_sundaes * sundae_price) / num_ice_cream_bars = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_bar_price_l1289_128947


namespace NUMINAMATH_CALUDE_no_trapezoid_solution_l1289_128920

theorem no_trapezoid_solution : 
  ¬ ∃ (b₁ b₂ : ℕ), 
    (b₁ % 10 = 0) ∧ 
    (b₂ % 10 = 0) ∧ 
    ((b₁ + b₂) * 30 / 2 = 1080) :=
by sorry

end NUMINAMATH_CALUDE_no_trapezoid_solution_l1289_128920
