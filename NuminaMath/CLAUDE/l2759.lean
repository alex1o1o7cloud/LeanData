import Mathlib

namespace NUMINAMATH_CALUDE_abs_sum_inequality_range_l2759_275909

theorem abs_sum_inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3*a) ↔ a ∈ Set.Icc (-1) 4 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_range_l2759_275909


namespace NUMINAMATH_CALUDE_temple_shop_cost_l2759_275969

/-- The cost per object at the shop --/
def cost_per_object : ℕ := 11

/-- The number of people in Nathan's group --/
def number_of_people : ℕ := 3

/-- The number of shoes per person --/
def shoes_per_person : ℕ := 2

/-- The number of socks per person --/
def socks_per_person : ℕ := 2

/-- The number of mobiles per person --/
def mobiles_per_person : ℕ := 1

/-- The total cost for Nathan and his parents to store their belongings --/
def total_cost : ℕ := number_of_people * (shoes_per_person + socks_per_person + mobiles_per_person) * cost_per_object

theorem temple_shop_cost : total_cost = 165 := by
  sorry

end NUMINAMATH_CALUDE_temple_shop_cost_l2759_275969


namespace NUMINAMATH_CALUDE_billy_bobbi_probability_zero_l2759_275928

def billy_number (n : ℕ) : Prop := n > 0 ∧ n < 150 ∧ 15 ∣ n
def bobbi_number (n : ℕ) : Prop := n > 0 ∧ n < 150 ∧ 20 ∣ n
def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem billy_bobbi_probability_zero :
  ∀ (b₁ b₂ : ℕ), 
    billy_number b₁ → 
    bobbi_number b₂ → 
    (is_square b₁ ∨ is_square b₂) →
    b₁ = b₂ → 
    False :=
sorry

end NUMINAMATH_CALUDE_billy_bobbi_probability_zero_l2759_275928


namespace NUMINAMATH_CALUDE_modified_riemann_zeta_sum_l2759_275948

noncomputable def ξ (x : ℝ) : ℝ := ∑' n, (1 : ℝ) / (2 * n) ^ x

theorem modified_riemann_zeta_sum (h : ∀ x > 2, ξ x = ∑' n, (1 : ℝ) / (2 * n) ^ x) :
  ∑' k, ξ (2 * k + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_modified_riemann_zeta_sum_l2759_275948


namespace NUMINAMATH_CALUDE_nuts_in_third_box_l2759_275934

-- Define the weights of nuts in each box
def box1 (x y z : ℝ) : ℝ := y + z - 6
def box2 (x y z : ℝ) : ℝ := x + z - 10

-- Theorem statement
theorem nuts_in_third_box (x y z : ℝ) 
  (h1 : x = box1 x y z) 
  (h2 : y = box2 x y z) : 
  z = 16 := by
sorry

end NUMINAMATH_CALUDE_nuts_in_third_box_l2759_275934


namespace NUMINAMATH_CALUDE_special_ellipse_major_axis_length_l2759_275995

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The ellipse is tangent to the line y = 1 -/
  tangent_to_y1 : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_to_yaxis : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ

/-- The length of the major axis of the special ellipse -/
def majorAxisLength (e : SpecialEllipse) : ℝ := sorry

/-- Theorem stating that the length of the major axis is 2 for the given ellipse -/
theorem special_ellipse_major_axis_length :
  ∀ (e : SpecialEllipse),
    e.tangent_to_y1 = true →
    e.tangent_to_yaxis = true →
    e.focus1 = (3, 2 + Real.sqrt 2) →
    e.focus2 = (3, 2 - Real.sqrt 2) →
    majorAxisLength e = 2 := by sorry

end NUMINAMATH_CALUDE_special_ellipse_major_axis_length_l2759_275995


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2759_275957

theorem polynomial_simplification (q : ℝ) : 
  (4 * q^3 - 7 * q^2 + 3 * q - 2) + (5 * q^2 - 9 * q + 8) = 4 * q^3 - 2 * q^2 - 6 * q + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2759_275957


namespace NUMINAMATH_CALUDE_polynomial_sum_equality_l2759_275942

-- Define the two polynomials
def p1 (x : ℝ) : ℝ := 3*x^4 + 2*x^3 - 5*x^2 + 9*x - 2
def p2 (x : ℝ) : ℝ := -3*x^4 - 5*x^3 + 7*x^2 - 9*x + 4

-- Define the sum of the polynomials
def sum_poly (x : ℝ) : ℝ := p1 x + p2 x

-- Define the result polynomial
def result (x : ℝ) : ℝ := -3*x^3 + 2*x^2 + 2

-- Theorem statement
theorem polynomial_sum_equality : 
  ∀ x : ℝ, sum_poly x = result x := by sorry

end NUMINAMATH_CALUDE_polynomial_sum_equality_l2759_275942


namespace NUMINAMATH_CALUDE_sara_has_108_golf_balls_l2759_275997

/-- The number of dozens of golf balls Sara has -/
def saras_dozens : ℕ := 9

/-- The number of items in one dozen -/
def items_per_dozen : ℕ := 12

/-- The total number of golf balls Sara has -/
def saras_golf_balls : ℕ := saras_dozens * items_per_dozen

theorem sara_has_108_golf_balls : saras_golf_balls = 108 := by
  sorry

end NUMINAMATH_CALUDE_sara_has_108_golf_balls_l2759_275997


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2759_275972

theorem fraction_evaluation : (3^4 - 3^2) / (3^(-2) + 3^(-4)) = 583.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2759_275972


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l2759_275962

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 29 = 17 % 29 ∧ 
  ∀ (y : ℕ), y > 0 → (5 * y) % 29 = 17 % 29 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l2759_275962


namespace NUMINAMATH_CALUDE_min_value_theorem_l2759_275980

/-- The circle C: (x-2)^2+(y+1)^2=5 is symmetric with respect to the line ax-by-1=0 -/
def symmetric_circle (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 5 ∧ a * x - b * y - 1 = 0

/-- The theorem stating the minimum value of 3/b + 2/a -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_sym : symmetric_circle a b) : 
    (∀ x y : ℝ, x > 0 → y > 0 → symmetric_circle x y → 3/y + 2/x ≥ 7 + 4 * Real.sqrt 3) ∧ 
    (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ symmetric_circle x y ∧ 3/y + 2/x = 7 + 4 * Real.sqrt 3) :=
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2759_275980


namespace NUMINAMATH_CALUDE_logarithm_inequality_l2759_275946

theorem logarithm_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  Real.log (Real.sqrt (a * b)) = (Real.log a + Real.log b) / 2 ∧ 
  Real.log (Real.sqrt (a * b)) < Real.log ((a + b) / 2) ∧
  Real.log ((a + b) / 2) < Real.log ((a^2 + b^2) / 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l2759_275946


namespace NUMINAMATH_CALUDE_smallest_number_l2759_275930

theorem smallest_number (A B C : ℚ) (hA : A = 1/2) (hB : B = 9/10) (hC : C = 2/5) :
  C ≤ A ∧ C ≤ B := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2759_275930


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_abs_min_value_is_achievable_l2759_275939

theorem min_value_of_sum_of_abs (x y : ℝ) : 
  |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 :=
by sorry

theorem min_value_is_achievable : 
  ∃ (x y : ℝ), |x - 1| + |x| + |y - 1| + |y + 1| = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_abs_min_value_is_achievable_l2759_275939


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2759_275915

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 64

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 36 - y^2 / 12 = 1

-- Theorem statement
theorem hyperbola_equation 
  (h1 : ∀ x y, ellipse x y ↔ hyperbola x y)  -- Same foci condition
  (h2 : ∃ x y, hyperbola x y ∧ asymptote x y)  -- Asymptote condition
  : ∀ x y, hyperbola x y ↔ x^2 / 36 - y^2 / 12 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2759_275915


namespace NUMINAMATH_CALUDE_base7_305_eq_base5_1102_l2759_275983

/-- Converts a base-7 number to its decimal (base-10) representation -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a decimal (base-10) number to its base-5 representation -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

/-- States that the base-7 number 305 is equal to the base-5 number 1102 -/
theorem base7_305_eq_base5_1102 :
  decimalToBase5 (base7ToDecimal [5, 0, 3]) = [1, 1, 0, 2] := by
  sorry

#eval base7ToDecimal [5, 0, 3]
#eval decimalToBase5 152

end NUMINAMATH_CALUDE_base7_305_eq_base5_1102_l2759_275983


namespace NUMINAMATH_CALUDE_triangle_inequality_and_equality_l2759_275984

/-- Triangle ABC with side lengths a, b, c opposite to vertices A, B, C respectively,
    and h being the height from vertex C onto side AB -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_h : 0 < h
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Main theorem about the inequality and equality condition -/
theorem triangle_inequality_and_equality (t : Triangle) :
  t.a + t.b ≥ Real.sqrt (t.c^2 + 4*t.h^2) ∧
  (t.a + t.b = Real.sqrt (t.c^2 + 4*t.h^2) ↔ t.a = t.b ∧ t.a^2 + t.b^2 = t.c^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_equality_l2759_275984


namespace NUMINAMATH_CALUDE_arrangement_problem_l2759_275981

/-- The number of ways to arrange people in a row -/
def arrange (n : ℕ) (m : ℕ) : ℕ :=
  n.factorial * m.factorial * (n + 1).factorial

/-- The problem statement -/
theorem arrangement_problem : arrange 5 2 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_problem_l2759_275981


namespace NUMINAMATH_CALUDE_tree_height_difference_l2759_275902

-- Define the heights of the trees
def birch_height : ℚ := 12 + 1/4
def maple_height : ℚ := 20 + 2/5

-- Define the height difference
def height_difference : ℚ := maple_height - birch_height

-- Theorem to prove
theorem tree_height_difference :
  height_difference = 8 + 3/20 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l2759_275902


namespace NUMINAMATH_CALUDE_circle_equation_l2759_275910

/-- Given a circle with center (0, -2) and a chord intercepted by the line 2x - y + 3 = 0
    with length 4√5, prove that the equation of the circle is x² + (y+2)² = 25. -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (0, -2)
  let chord_line (x y : ℝ) := 2 * x - y + 3 = 0
  let chord_length : ℝ := 4 * Real.sqrt 5
  ∃ (r : ℝ), r > 0 ∧
    (∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 ↔
      x^2 + (y + 2)^2 = 25) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_equation_l2759_275910


namespace NUMINAMATH_CALUDE_expression_evaluation_l2759_275954

theorem expression_evaluation (x : ℝ) (h : x = 4) :
  (x - 1 - 3 / (x + 1)) / ((x^2 - 2*x) / (x + 1)) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2759_275954


namespace NUMINAMATH_CALUDE_lego_sales_triple_pieces_l2759_275919

/-- Represents the number of Lego pieces sold for each type --/
structure LegoSales where
  single : ℕ
  double : ℕ
  triple : ℕ
  quadruple : ℕ

/-- Calculates the total earnings in cents from Lego sales --/
def totalEarnings (sales : LegoSales) : ℕ :=
  sales.single * 1 + sales.double * 2 + sales.triple * 3 + sales.quadruple * 4

/-- The main theorem to prove --/
theorem lego_sales_triple_pieces : 
  ∃ (sales : LegoSales), 
    sales.single = 100 ∧ 
    sales.double = 45 ∧ 
    sales.quadruple = 165 ∧ 
    totalEarnings sales = 1000 ∧ 
    sales.triple = 50 := by
  sorry


end NUMINAMATH_CALUDE_lego_sales_triple_pieces_l2759_275919


namespace NUMINAMATH_CALUDE_frac_greater_than_one_solution_set_l2759_275992

theorem frac_greater_than_one_solution_set (x : ℝ) : 
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_frac_greater_than_one_solution_set_l2759_275992


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l2759_275912

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l2759_275912


namespace NUMINAMATH_CALUDE_digital_earth_capabilities_l2759_275988

-- Define the possible capabilities
inductive Capability
  | ReceiveDistanceEducation
  | ShopOnline
  | SeekMedicalAdviceOnline
  | TravelAroundWorld

-- Define Digital Earth
def DigitalEarth : Type := Set Capability

-- Define the correct set of capabilities
def CorrectCapabilities : Set Capability :=
  {Capability.ReceiveDistanceEducation, Capability.ShopOnline, Capability.SeekMedicalAdviceOnline}

-- Theorem stating that Digital Earth capabilities are exactly the correct ones
theorem digital_earth_capabilities :
  ∃ (de : DigitalEarth), de = CorrectCapabilities :=
sorry

end NUMINAMATH_CALUDE_digital_earth_capabilities_l2759_275988


namespace NUMINAMATH_CALUDE_symbol_equations_l2759_275907

theorem symbol_equations :
  ∀ (triangle circle square star : ℤ),
  triangle = circle + 2 →
  square = triangle + triangle →
  star = triangle + square + 5 →
  star = circle + 31 →
  triangle = 12 ∧ circle = 10 ∧ square = 24 ∧ star = 41 := by
sorry

end NUMINAMATH_CALUDE_symbol_equations_l2759_275907


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2759_275965

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 - 2 * x + 5 = 0 ∧ m * y^2 - 2 * y + 5 = 0) ↔ 
  (m < (1 : ℝ) / 5 ∧ m ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2759_275965


namespace NUMINAMATH_CALUDE_theater_ticket_pricing_l2759_275996

theorem theater_ticket_pricing (adult_price : ℝ) 
  (h1 : 4 * adult_price + 3 * (adult_price / 2) + 2 * (0.75 * adult_price) = 35) :
  10 * adult_price + 8 * (adult_price / 2) + 5 * (0.75 * adult_price) = 88.75 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_pricing_l2759_275996


namespace NUMINAMATH_CALUDE_circle_diameter_and_circumference_l2759_275985

theorem circle_diameter_and_circumference (A : ℝ) (h : A = 16 * Real.pi) :
  ∃ (d c : ℝ), d = 8 ∧ c = 8 * Real.pi ∧ A = Real.pi * (d / 2)^2 ∧ c = Real.pi * d := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_and_circumference_l2759_275985


namespace NUMINAMATH_CALUDE_license_plate_count_l2759_275976

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible characters for the second position (letters + digits) -/
def num_second_choices : ℕ := num_letters + num_digits

/-- The length of the license plate -/
def plate_length : ℕ := 4

/-- Calculates the number of possible license plates given the constraints -/
def num_license_plates : ℕ :=
  num_letters * num_second_choices * 1 * num_digits

/-- Theorem stating that the number of possible license plates is 9360 -/
theorem license_plate_count :
  num_license_plates = 9360 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2759_275976


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2759_275916

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≤ 1}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2759_275916


namespace NUMINAMATH_CALUDE_total_people_in_program_l2759_275900

theorem total_people_in_program : 
  let parents : ℕ := 105
  let pupils : ℕ := 698
  let staff : ℕ := 45
  let performers : ℕ := 32
  parents + pupils + staff + performers = 880 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l2759_275900


namespace NUMINAMATH_CALUDE_parabola_tangent_condition_l2759_275991

/-- A parabola is tangent to a line if and only if their intersection has exactly one solution --/
def is_tangent (a b : ℝ) : Prop :=
  ∃! x, a * x^2 + b * x + 12 = 2 * x + 3

/-- The main theorem stating the conditions for the parabola to be tangent to the line --/
theorem parabola_tangent_condition (a b : ℝ) :
  is_tangent a b ↔ (b = 2 + 6 * Real.sqrt a ∨ b = 2 - 6 * Real.sqrt a) ∧ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_condition_l2759_275991


namespace NUMINAMATH_CALUDE_parabola_vertex_problem_parabola_vertex_l2759_275921

/-- The coordinates of the vertex of a parabola in the form y = a(x-h)^2 + k are (h,k) -/
theorem parabola_vertex (a h k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k
  (∀ x, f x ≥ f h) ∧ f h = k := by sorry

/-- The coordinates of the vertex of the parabola y = 3(x-7)^2 + 5 are (7,5) -/
theorem problem_parabola_vertex : 
  let f : ℝ → ℝ := λ x ↦ 3 * (x - 7)^2 + 5
  (∀ x, f x ≥ f 7) ∧ f 7 = 5 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_problem_parabola_vertex_l2759_275921


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2759_275929

theorem quadratic_always_positive (m : ℝ) (h : m > 3) :
  ∀ x : ℝ, m * x^2 - (m + 3) * x + m > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2759_275929


namespace NUMINAMATH_CALUDE_distance_between_points_l2759_275993

/-- The distance between two points when two vehicles move towards each other -/
theorem distance_between_points (v1 v2 t : ℝ) (h1 : v1 > 0) (h2 : v2 > 0) (h3 : t > 0) :
  let d := (v1 + v2) * t
  d = v1 * t + v2 * t :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l2759_275993


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2759_275973

theorem not_sufficient_not_necessary (a b : ℝ) : 
  ¬(∀ a b : ℝ, (a < 0 ∧ b < 0) → a * b * (a - b) > 0) ∧ 
  ¬(∀ a b : ℝ, a * b * (a - b) > 0 → (a < 0 ∧ b < 0)) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2759_275973


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2759_275982

theorem sum_of_squares_of_roots (r s t : ℝ) : 
  (2 * r^3 + 3 * r^2 - 5 * r + 1 = 0) →
  (2 * s^3 + 3 * s^2 - 5 * s + 1 = 0) →
  (2 * t^3 + 3 * t^2 - 5 * t + 1 = 0) →
  (r ≠ s) → (r ≠ t) → (s ≠ t) →
  r^2 + s^2 + t^2 = -11/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2759_275982


namespace NUMINAMATH_CALUDE_product_repeating_decimal_and_seven_l2759_275986

theorem product_repeating_decimal_and_seven (x : ℚ) : 
  (x = 1/3) → (x * 7 = 7/3) := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_and_seven_l2759_275986


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2759_275967

theorem complex_expression_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  7 * (4 - i) + 4 * i * (7 - i) + 2 * (3 + i) = 38 + 23 * i := by
sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2759_275967


namespace NUMINAMATH_CALUDE_puppies_sold_l2759_275998

theorem puppies_sold (initial_puppies initial_kittens kittens_sold remaining_pets : ℕ) :
  initial_puppies = 7 →
  initial_kittens = 6 →
  kittens_sold = 3 →
  remaining_pets = 8 →
  initial_puppies + initial_kittens - kittens_sold - remaining_pets = 2 := by
  sorry

#check puppies_sold

end NUMINAMATH_CALUDE_puppies_sold_l2759_275998


namespace NUMINAMATH_CALUDE_tetrahedron_division_possible_l2759_275990

theorem tetrahedron_division_possible (edge_length : ℝ) (target_length : ℝ) : 
  edge_length > 0 → target_length > 0 → target_length < edge_length →
  ∃ n : ℕ, (1/2 : ℝ)^n * edge_length < target_length := by
  sorry

#check tetrahedron_division_possible 1 (1/100)

end NUMINAMATH_CALUDE_tetrahedron_division_possible_l2759_275990


namespace NUMINAMATH_CALUDE_rectangle_area_l2759_275952

theorem rectangle_area (a b : ℝ) :
  let diagonal := a + b
  let ratio := 2 / 1
  let longer_side := Real.sqrt ((4 * diagonal^2) / 5)
  let shorter_side := longer_side / 2
  let area := longer_side * shorter_side
  area = (2 * diagonal^2) / 5 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2759_275952


namespace NUMINAMATH_CALUDE_action_figure_cost_l2759_275940

theorem action_figure_cost (current : ℕ) (total : ℕ) (cost : ℕ) : current = 7 → total = 16 → cost = 72 → (cost / (total - current) : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_action_figure_cost_l2759_275940


namespace NUMINAMATH_CALUDE_average_weight_abc_l2759_275931

/-- Given the weights of three individuals a, b, and c, prove that their average weight is 45 kg -/
theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 40 →   -- average weight of a and b is 40 kg
  (b + c) / 2 = 47 →   -- average weight of b and c is 47 kg
  b = 39 →             -- weight of b is 39 kg
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_abc_l2759_275931


namespace NUMINAMATH_CALUDE_point_on_circle_after_rotation_l2759_275961

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def arc_length (θ : ℝ) : ℝ := θ

theorem point_on_circle_after_rotation 
  (start_x start_y : ℝ) 
  (θ : ℝ) 
  (h_start : unit_circle start_x start_y) 
  (h_θ : arc_length θ = 2 * Real.pi / 3) :
  ∃ (end_x end_y : ℝ), 
    unit_circle end_x end_y ∧ 
    end_x = -1/2 ∧ 
    end_y = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_point_on_circle_after_rotation_l2759_275961


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2759_275975

-- Problem 1
theorem problem_1 : 27 - 16 + (-7) - 18 = -14 := by sorry

-- Problem 2
theorem problem_2 : (-6) * (-3/4) / (-3/2) = -3 := by sorry

-- Problem 3
theorem problem_3 : (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81 := by sorry

-- Problem 4
theorem problem_4 : -2^4 + 3 * (-1)^4 - (-2)^3 = -5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2759_275975


namespace NUMINAMATH_CALUDE_ivy_stripping_l2759_275999

/-- The number of feet of ivy Cary strips daily -/
def daily_strip : ℝ := 6

/-- The initial ivy coverage in feet -/
def initial_coverage : ℝ := 40

/-- The number of days it takes to remove all ivy -/
def days_to_remove : ℝ := 10

/-- The number of feet the ivy grows each night -/
def nightly_growth : ℝ := 2

theorem ivy_stripping :
  daily_strip * days_to_remove - nightly_growth * days_to_remove = initial_coverage :=
sorry

end NUMINAMATH_CALUDE_ivy_stripping_l2759_275999


namespace NUMINAMATH_CALUDE_larger_number_l2759_275943

theorem larger_number (x y : ℝ) (h1 : x - y = 5) (h2 : x + y = 20) : max x y = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l2759_275943


namespace NUMINAMATH_CALUDE_line_slope_through_points_l2759_275924

/-- The slope of a line passing through points (1, 0) and (2, √3) is √3. -/
theorem line_slope_through_points : 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (2, Real.sqrt 3)
  (B.2 - A.2) / (B.1 - A.1) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_through_points_l2759_275924


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2759_275951

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 = 12) →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2759_275951


namespace NUMINAMATH_CALUDE_no_roots_implication_l2759_275956

theorem no_roots_implication (p q b c : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + p*x + q ≠ 0)
  (h2 : ∀ x : ℝ, x^2 + b*x + c ≠ 0) :
  ∀ x : ℝ, 7*x^2 + (2*p + 3*b + 4)*x + 2*q + 3*c + 2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_roots_implication_l2759_275956


namespace NUMINAMATH_CALUDE_cosine_power_sum_l2759_275935

theorem cosine_power_sum (α : ℝ) (n : ℤ) (x : ℝ) (hx : x ≠ 0) :
  x + 1/x = 2 * Real.cos α →
  x^n + 1/x^n = 2 * Real.cos (n * α) := by
sorry

end NUMINAMATH_CALUDE_cosine_power_sum_l2759_275935


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2759_275944

theorem polynomial_factorization (b : ℝ) : 
  (8 * b^4 - 100 * b^3 + 18) - (3 * b^4 - 11 * b^3 + 18) = b^3 * (5 * b - 89) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2759_275944


namespace NUMINAMATH_CALUDE_percent_of_percent_l2759_275970

theorem percent_of_percent (x : ℝ) : (0.3 * (0.6 * x)) = (0.18 * x) := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l2759_275970


namespace NUMINAMATH_CALUDE_probability_both_red_correct_l2759_275925

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def blue_balls : ℕ := 4
def green_balls : ℕ := 2
def balls_picked : ℕ := 2

def probability_both_red : ℚ := 2 / 15

theorem probability_both_red_correct :
  (Nat.choose red_balls balls_picked : ℚ) / (Nat.choose total_balls balls_picked : ℚ) = probability_both_red :=
sorry

end NUMINAMATH_CALUDE_probability_both_red_correct_l2759_275925


namespace NUMINAMATH_CALUDE_sixtieth_pair_l2759_275904

/-- Definition of the sequence of integer pairs -/
def sequence_pair : ℕ → ℕ × ℕ
| 0 => (1, 1)
| n + 1 => 
  let (a, b) := sequence_pair n
  if a = 1 then (b + 1, 1) else (a - 1, b + 1)

/-- The 60th pair in the sequence is (5, 7) -/
theorem sixtieth_pair : sequence_pair 59 = (5, 7) := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_pair_l2759_275904


namespace NUMINAMATH_CALUDE_middle_digit_zero_l2759_275906

theorem middle_digit_zero (a b c : Nat) (M : Nat) :
  (0 ≤ a ∧ a < 6) →
  (0 ≤ b ∧ b < 6) →
  (0 ≤ c ∧ c < 6) →
  M = 36 * a + 6 * b + c →
  M = 64 * a + 8 * b + c →
  b = 0 := by
  sorry

end NUMINAMATH_CALUDE_middle_digit_zero_l2759_275906


namespace NUMINAMATH_CALUDE_value_in_scientific_notation_l2759_275966

-- Define a billion
def billion : ℝ := 10^9

-- Define the value in question
def value : ℝ := 101.49 * billion

-- Theorem statement
theorem value_in_scientific_notation : value = 1.0149 * 10^10 := by
  sorry

end NUMINAMATH_CALUDE_value_in_scientific_notation_l2759_275966


namespace NUMINAMATH_CALUDE_ax5_plus_by5_l2759_275937

theorem ax5_plus_by5 (a b x y : ℝ) 
  (eq1 : a*x + b*y = 1)
  (eq2 : a*x^2 + b*y^2 = 2)
  (eq3 : a*x^3 + b*y^3 = 5)
  (eq4 : a*x^4 + b*y^4 = 15) :
  a*x^5 + b*y^5 = -40 := by
  sorry

end NUMINAMATH_CALUDE_ax5_plus_by5_l2759_275937


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l2759_275933

theorem chord_length_concentric_circles (area_ring : ℝ) (r_small : ℝ) (r_large : ℝ) (chord_length : ℝ) : 
  area_ring = 18.75 * Real.pi ∧ 
  r_large = 2 * r_small ∧ 
  area_ring = Real.pi * (r_large^2 - r_small^2) ∧
  chord_length^2 = 4 * (r_large^2 - r_small^2) →
  chord_length = Real.sqrt 75 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l2759_275933


namespace NUMINAMATH_CALUDE_perfect_square_sum_l2759_275979

theorem perfect_square_sum : 529 + 2 * 23 * 7 + 49 = 900 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l2759_275979


namespace NUMINAMATH_CALUDE_six_digit_integers_count_l2759_275949

/-- The number of different positive, six-digit integers that can be formed
    using the digits 2, 2, 2, 5, 5, and 9 -/
def six_digit_integers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)

/-- Theorem stating that the number of different positive, six-digit integers
    that can be formed using the digits 2, 2, 2, 5, 5, and 9 is equal to 60 -/
theorem six_digit_integers_count : six_digit_integers = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_integers_count_l2759_275949


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l2759_275936

/-- Circle C with equation x^2+y^2-8x+6y+21=0 -/
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2 - 8*p.1 + 6*p.2 + 21) = 0}

/-- Point A with coordinates (-6, 7) -/
def point_A : ℝ × ℝ := (-6, 7)

/-- A line is tangent to a circle if it intersects the circle at exactly one point -/
def is_tangent_line (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ l ∩ c

/-- The set of all lines passing through point A -/
def lines_through_A : Set (Set (ℝ × ℝ)) :=
  {l | point_A ∈ l ∧ ∃ k, l = {p | p.2 - 7 = k * (p.1 + 6)}}

theorem circle_and_tangent_line :
  ∃ l ∈ lines_through_A,
    is_tangent_line l circle_C ∧
    (∃ c r, c = (4, -3) ∧ r = 2 ∧
      ∀ p ∈ circle_C, (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    (l = {p | 3*p.1 + 4*p.2 - 10 = 0} ∨ l = {p | 4*p.1 + 3*p.2 + 3 = 0}) :=
  sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l2759_275936


namespace NUMINAMATH_CALUDE_simplify_fraction_l2759_275945

theorem simplify_fraction : 21 * (8 / 15) * (1 / 14) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2759_275945


namespace NUMINAMATH_CALUDE_bookstore_new_releases_fraction_l2759_275963

theorem bookstore_new_releases_fraction 
  (total_books : ℕ) 
  (historical_fiction_ratio : ℚ) 
  (historical_fiction_new_release_ratio : ℚ) 
  (other_new_release_ratio : ℚ) 
  (h1 : historical_fiction_ratio = 30 / 100)
  (h2 : historical_fiction_new_release_ratio = 40 / 100)
  (h3 : other_new_release_ratio = 50 / 100)
  (h4 : total_books > 0) :
  let historical_fiction_books := total_books * historical_fiction_ratio
  let historical_fiction_new_releases := historical_fiction_books * historical_fiction_new_release_ratio
  let other_books := total_books - historical_fiction_books
  let other_new_releases := other_books * other_new_release_ratio
  let total_new_releases := historical_fiction_new_releases + other_new_releases
  (historical_fiction_new_releases / total_new_releases : ℚ) = 12 / 47 := by
sorry

end NUMINAMATH_CALUDE_bookstore_new_releases_fraction_l2759_275963


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2759_275964

theorem remainder_divisibility (x : ℤ) : x % 72 = 19 → x % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2759_275964


namespace NUMINAMATH_CALUDE_binomial_1409_1_l2759_275903

theorem binomial_1409_1 : (1409 : ℕ).choose 1 = 1409 := by sorry

end NUMINAMATH_CALUDE_binomial_1409_1_l2759_275903


namespace NUMINAMATH_CALUDE_amusement_park_expenses_l2759_275989

theorem amusement_park_expenses (total brought food tshirt left ticket : ℕ) : 
  brought = 75 ∧ 
  food = 13 ∧ 
  tshirt = 23 ∧ 
  left = 9 ∧ 
  brought = food + tshirt + left + ticket → 
  ticket = 30 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_expenses_l2759_275989


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2759_275917

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 7

theorem volleyball_team_selection :
  (Nat.choose (total_players - quadruplets) starters) +
  (quadruplets * Nat.choose (total_players - quadruplets) (starters - 1)) = 4488 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2759_275917


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2759_275978

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {2, 4, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2759_275978


namespace NUMINAMATH_CALUDE_sqrt_of_16_l2759_275913

theorem sqrt_of_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_16_l2759_275913


namespace NUMINAMATH_CALUDE_rope_length_for_second_post_l2759_275932

theorem rope_length_for_second_post
  (total_rope : ℕ)
  (first_post : ℕ)
  (third_post : ℕ)
  (fourth_post : ℕ)
  (h1 : total_rope = 70)
  (h2 : first_post = 24)
  (h3 : third_post = 14)
  (h4 : fourth_post = 12) :
  total_rope - (first_post + third_post + fourth_post) = 20 := by
  sorry

#check rope_length_for_second_post

end NUMINAMATH_CALUDE_rope_length_for_second_post_l2759_275932


namespace NUMINAMATH_CALUDE_yella_computer_usage_l2759_275947

def days_in_week : ℕ := 7
def hours_per_day_this_week : ℕ := 8
def hours_difference : ℕ := 35

def computer_usage_last_week : ℕ := days_in_week * hours_per_day_this_week + hours_difference

theorem yella_computer_usage :
  computer_usage_last_week = 91 := by
sorry

end NUMINAMATH_CALUDE_yella_computer_usage_l2759_275947


namespace NUMINAMATH_CALUDE_evaluate_expression_l2759_275994

theorem evaluate_expression : -(16 / 4 * 12 - 100 + 2^3 * 6) = 4 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2759_275994


namespace NUMINAMATH_CALUDE_balloon_count_l2759_275977

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 41

/-- The total number of blue balloons Joan and Melanie have together -/
def total_balloons : ℕ := joan_balloons + melanie_balloons

theorem balloon_count : total_balloons = 81 := by sorry

end NUMINAMATH_CALUDE_balloon_count_l2759_275977


namespace NUMINAMATH_CALUDE_pascals_triangle_56th_row_second_to_last_l2759_275908

theorem pascals_triangle_56th_row_second_to_last : Nat.choose 56 55 = 56 := by
  sorry

end NUMINAMATH_CALUDE_pascals_triangle_56th_row_second_to_last_l2759_275908


namespace NUMINAMATH_CALUDE_december_gas_consumption_l2759_275920

/-- Gas fee structure and consumption for a user in December --/
structure GasConsumption where
  baseRate : ℝ  -- Rate for the first 60 cubic meters
  excessRate : ℝ  -- Rate for consumption above 60 cubic meters
  baseVolume : ℝ  -- Volume threshold for base rate
  averageCost : ℝ  -- Average cost per cubic meter for the user
  consumption : ℝ  -- Total gas consumption

/-- The gas consumption satisfies the given fee structure and average cost --/
def validConsumption (g : GasConsumption) : Prop :=
  g.baseRate * g.baseVolume + g.excessRate * (g.consumption - g.baseVolume) = g.averageCost * g.consumption

/-- Theorem stating that given the fee structure and average cost, 
    the gas consumption in December was 100 cubic meters --/
theorem december_gas_consumption :
  ∃ (g : GasConsumption), 
    g.baseRate = 1 ∧ 
    g.excessRate = 1.5 ∧ 
    g.baseVolume = 60 ∧ 
    g.averageCost = 1.2 ∧ 
    g.consumption = 100 ∧
    validConsumption g :=
  sorry

end NUMINAMATH_CALUDE_december_gas_consumption_l2759_275920


namespace NUMINAMATH_CALUDE_shorter_train_length_l2759_275926

/-- Calculates the length of the shorter train given the speeds of two trains,
    the time they take to cross each other, and the length of the longer train. -/
theorem shorter_train_length
  (speed1 : ℝ) (speed2 : ℝ) (crossing_time : ℝ) (longer_train_length : ℝ)
  (h1 : speed1 = 60) -- km/hr
  (h2 : speed2 = 40) -- km/hr
  (h3 : crossing_time = 10.799136069114471) -- seconds
  (h4 : longer_train_length = 160) -- meters
  : ∃ (shorter_train_length : ℝ),
    shorter_train_length = 140 ∧ 
    shorter_train_length = 
      (speed1 + speed2) * (5 / 18) * crossing_time - longer_train_length :=
by
  sorry

end NUMINAMATH_CALUDE_shorter_train_length_l2759_275926


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2759_275911

theorem water_tank_capacity (w c : ℝ) (h1 : w / c = 1 / 5) (h2 : (w + 3) / c = 1 / 4) : c = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2759_275911


namespace NUMINAMATH_CALUDE_trigonometric_equality_l2759_275959

theorem trigonometric_equality (x y : ℝ) 
  (h : (Real.sin x ^ 2 - Real.cos x ^ 2 + Real.cos x ^ 2 * Real.cos y ^ 2 - Real.sin x ^ 2 * Real.sin y ^ 2) / Real.sin (x + y) = 1) :
  ∃ k : ℤ, x - y = 2 * k * Real.pi + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l2759_275959


namespace NUMINAMATH_CALUDE_correct_plates_removed_l2759_275918

/-- The number of plates that need to be removed to reach the acceptable weight -/
def plates_to_remove : ℕ :=
  let initial_plates : ℕ := 38
  let plate_weight : ℕ := 10  -- in ounces
  let max_weight_lbs : ℕ := 20
  let max_weight_oz : ℕ := max_weight_lbs * 16
  let total_weight : ℕ := initial_plates * plate_weight
  let excess_weight : ℕ := total_weight - max_weight_oz
  excess_weight / plate_weight

theorem correct_plates_removed : plates_to_remove = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_plates_removed_l2759_275918


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l2759_275905

theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 14) (h3 : EF = 15) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let radius := area / s
  2 * radius = 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l2759_275905


namespace NUMINAMATH_CALUDE_absolute_value_and_quadratic_equation_l2759_275923

theorem absolute_value_and_quadratic_equation :
  ∀ (b c : ℝ),
  (∀ x : ℝ, |x - 4| = 3 ↔ x^2 + b*x + c = 0) →
  b = -8 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_and_quadratic_equation_l2759_275923


namespace NUMINAMATH_CALUDE_smoking_health_correlation_l2759_275938

-- Define smoking and health as variables
variable (smoking health : ℝ)

-- Define the concept of "harmful to health"
def is_harmful_to_health (x y : ℝ) : Prop := 
  ∀ δ > 0, ∃ ε > 0, ∀ x' y', |x' - x| < ε → |y' - y| < δ → y' < y

-- Define negative correlation
def negative_correlation (x y : ℝ) : Prop :=
  ∀ δ > 0, ∃ ε > 0, ∀ x₁ x₂ y₁ y₂, 
    |x₁ - x| < ε → |x₂ - x| < ε → |y₁ - y| < δ → |y₂ - y| < δ →
    (x₁ < x₂ → y₁ > y₂) ∧ (x₁ > x₂ → y₁ < y₂)

-- Theorem statement
theorem smoking_health_correlation 
  (h : is_harmful_to_health smoking health) : 
  negative_correlation smoking health :=
sorry

end NUMINAMATH_CALUDE_smoking_health_correlation_l2759_275938


namespace NUMINAMATH_CALUDE_symmetric_line_l2759_275922

/-- Given a line L1 with equation x - 2y + 1 = 0 and a line of symmetry x = 1,
    the symmetric line L2 has the equation x + 2y - 3 = 0 -/
theorem symmetric_line (x y : ℝ) :
  (x - 2*y + 1 = 0) →  -- Original line L1
  (x = 1) →            -- Line of symmetry
  (x + 2*y - 3 = 0)    -- Symmetric line L2
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_l2759_275922


namespace NUMINAMATH_CALUDE_electronic_product_failure_probability_l2759_275960

theorem electronic_product_failure_probability
  (p_working : ℝ)
  (h_working : p_working = 0.992)
  (h_probability : 0 ≤ p_working ∧ p_working ≤ 1) :
  1 - p_working = 0.008 := by
sorry

end NUMINAMATH_CALUDE_electronic_product_failure_probability_l2759_275960


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_12_l2759_275955

theorem cos_alpha_plus_pi_12 (α : Real) (h : Real.tan (α + π/3) = -2) :
  Real.cos (α + π/12) = Real.sqrt 10 / 10 ∨ Real.cos (α + π/12) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_12_l2759_275955


namespace NUMINAMATH_CALUDE_carrie_pays_94_l2759_275958

/-- The amount Carrie pays for clothes given the quantities and prices of items, and that her mom pays half the total cost. -/
def carriePays (shirtQuantity pantQuantity jacketQuantity : ℕ) 
               (shirtPrice pantPrice jacketPrice : ℚ) : ℚ :=
  let totalCost := shirtQuantity * shirtPrice + 
                   pantQuantity * pantPrice + 
                   jacketQuantity * jacketPrice
  totalCost / 2

/-- Theorem stating that Carrie pays $94 for the clothes. -/
theorem carrie_pays_94 : 
  carriePays 4 2 2 8 18 60 = 94 := by
  sorry

end NUMINAMATH_CALUDE_carrie_pays_94_l2759_275958


namespace NUMINAMATH_CALUDE_max_value_expression_l2759_275971

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0) 
  (sum_condition : x + y + z = 2) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 256/243 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2759_275971


namespace NUMINAMATH_CALUDE_isosceles_triangle_circumradius_l2759_275927

/-- The radius of a circle circumscribing an isosceles triangle -/
theorem isosceles_triangle_circumradius (a b c : ℝ) (h_isosceles : a = b) (h_sides : a = 13 ∧ c = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * area) = 169 / 24 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_circumradius_l2759_275927


namespace NUMINAMATH_CALUDE_subtract_like_terms_l2759_275974

theorem subtract_like_terms (a : ℝ) : 3 * a^2 - 2 * a^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l2759_275974


namespace NUMINAMATH_CALUDE_overlap_area_is_nine_l2759_275950

/-- Regular hexagon with area 36 -/
structure RegularHexagon :=
  (area : ℝ)
  (is_regular : Bool)
  (area_eq : area = 36)

/-- Equilateral triangle formed by connecting every other vertex of the hexagon -/
structure EquilateralTriangle (hex : RegularHexagon) :=
  (vertices : Fin 3 → Fin 6)
  (is_equilateral : Bool)
  (area : ℝ)
  (area_eq : area = hex.area / 2)

/-- The overlapping region of two equilateral triangles in the hexagon -/
def overlap_area (hex : RegularHexagon) (t1 t2 : EquilateralTriangle hex) : ℝ := sorry

/-- Theorem stating that the overlap area is 9 -/
theorem overlap_area_is_nine (hex : RegularHexagon) 
  (t1 t2 : EquilateralTriangle hex) : overlap_area hex t1 t2 = 9 := by sorry

end NUMINAMATH_CALUDE_overlap_area_is_nine_l2759_275950


namespace NUMINAMATH_CALUDE_train_passing_jogger_l2759_275941

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger
  (jogger_speed : Real)
  (train_speed : Real)
  (train_length : Real)
  (initial_distance : Real)
  (h1 : jogger_speed = 9 * (1000 / 3600))
  (h2 : train_speed = 45 * (1000 / 3600))
  (h3 : train_length = 120)
  (h4 : initial_distance = 250) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 37 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_l2759_275941


namespace NUMINAMATH_CALUDE_computer_price_increase_l2759_275987

/-- The new price of a computer after a 30% increase, given initial conditions -/
theorem computer_price_increase (b : ℝ) (h : 2 * b = 540) : b * 1.3 = 351 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2759_275987


namespace NUMINAMATH_CALUDE_probability_sum_17_three_dice_l2759_275968

-- Define a die as a finite type with 6 faces
def Die := Fin 6

-- Define a function to represent the value on a die face (1 to 6)
def dieValue (d : Die) : Nat := d.val + 1

-- Define a function to calculate the sum of three dice rolls
def sumOfThreeDice (d1 d2 d3 : Die) : Nat :=
  dieValue d1 + dieValue d2 + dieValue d3

-- Define the total number of possible outcomes when rolling three dice
def totalOutcomes : Nat := 6 * 6 * 6

-- Define the number of favorable outcomes (sum of 17)
def favorableOutcomes : Nat := 3

-- Theorem statement
theorem probability_sum_17_three_dice :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_17_three_dice_l2759_275968


namespace NUMINAMATH_CALUDE_sum_of_xy_l2759_275901

theorem sum_of_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (hx_bound : x < 30) (hy_bound : y < 30) 
  (h_eq : x + y + x * y = 119) : x + y = 24 ∨ x + y = 21 ∨ x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l2759_275901


namespace NUMINAMATH_CALUDE_pgcd_and_divisibility_properties_l2759_275953

/-- Given a ≥ 2 and m ≥ n ≥ 1, prove three statements about PGCD and divisibility -/
theorem pgcd_and_divisibility_properties (a m n : ℕ) (ha : a ≥ 2) (hmn : m ≥ n) (hn : n ≥ 1) :
  (gcd (a^m - 1) (a^n - 1) = gcd (a^(m-n) - 1) (a^n - 1)) ∧
  (gcd (a^m - 1) (a^n - 1) = a^(gcd m n) - 1) ∧
  ((a^m - 1) ∣ (a^n - 1) ↔ m ∣ n) := by
  sorry


end NUMINAMATH_CALUDE_pgcd_and_divisibility_properties_l2759_275953


namespace NUMINAMATH_CALUDE_min_students_l2759_275914

/-- Represents a student in the math competition -/
structure Student where
  solved : Finset (Fin 6)

/-- Represents the math competition -/
structure MathCompetition where
  students : Finset Student
  problem_count : Nat
  students_per_problem : Nat

/-- The conditions of the math competition -/
def validCompetition (c : MathCompetition) : Prop :=
  c.problem_count = 6 ∧
  c.students_per_problem = 500 ∧
  (∀ p : Fin 6, (c.students.filter (fun s => p ∈ s.solved)).card = c.students_per_problem) ∧
  (∀ s₁ s₂ : Student, s₁ ∈ c.students → s₂ ∈ c.students → s₁ ≠ s₂ → 
    ∃ p : Fin 6, p ∉ s₁.solved ∧ p ∉ s₂.solved)

/-- The theorem to be proved -/
theorem min_students (c : MathCompetition) (h : validCompetition c) : 
  c.students.card ≥ 1000 := by
  sorry

end NUMINAMATH_CALUDE_min_students_l2759_275914
