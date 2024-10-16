import Mathlib

namespace NUMINAMATH_CALUDE_parabola_two_axis_intersections_l3596_359603

/-- A parabola has only two common points with the coordinate axes if and only if m is 0 or 8 --/
theorem parabola_two_axis_intersections (m : ℝ) : 
  (∃! x y : ℝ, (y = 2*x^2 + 8*x + m ∧ (x = 0 ∨ y = 0)) ∧ 
   (∃ x' y' : ℝ, (y' = 2*x'^2 + 8*x' + m ∧ (x' = 0 ∨ y' = 0)) ∧ (x ≠ x' ∨ y ≠ y'))) ↔ 
  (m = 0 ∨ m = 8) :=
sorry

end NUMINAMATH_CALUDE_parabola_two_axis_intersections_l3596_359603


namespace NUMINAMATH_CALUDE_max_inscribed_circle_area_l3596_359629

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Left focus of the ellipse -/
def F1 : ℝ × ℝ := (-1, 0)

/-- Right focus of the ellipse -/
def F2 : ℝ × ℝ := (1, 0)

/-- A line passing through the right focus -/
def line_through_F2 (m : ℝ) (y : ℝ) : ℝ := m * y + 1

/-- Points of intersection between the line and the ellipse -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ y, ellipse (line_through_F2 m y) y ∧ p = (line_through_F2 m y, y)}

/-- Triangle formed by F1 and two intersection points -/
def triangle_F1PQ (m : ℝ) : Set (ℝ × ℝ) :=
  {F1} ∪ intersection_points m

/-- The inscribed circle of a triangle -/
def inscribed_circle (t : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry  -- Definition of inscribed circle

/-- The area of a circle -/
def circle_area (c : Set (ℝ × ℝ)) : ℝ :=
  sorry  -- Definition of circle area

/-- The theorem to be proved -/
theorem max_inscribed_circle_area :
  ∃ (m : ℝ), ∀ (n : ℝ),
    circle_area (inscribed_circle (triangle_F1PQ m)) ≥
    circle_area (inscribed_circle (triangle_F1PQ n)) ∧
    circle_area (inscribed_circle (triangle_F1PQ m)) = 9 * Real.pi / 16 :=
sorry

end NUMINAMATH_CALUDE_max_inscribed_circle_area_l3596_359629


namespace NUMINAMATH_CALUDE_min_value_expression_l3596_359683

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 1) (hc : c > 1) :
  (((a^2 + 1) / (2*a*b) - 1) * c + (Real.sqrt 2 / (c - 1))) ≥ 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3596_359683


namespace NUMINAMATH_CALUDE_fraction_calculation_l3596_359642

theorem fraction_calculation : (1 / 4 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 144 + (1 / 2 : ℚ) = (5 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3596_359642


namespace NUMINAMATH_CALUDE_tile_square_side_length_l3596_359600

/-- Given tiles with width 16 and length 24, proves that the side length of a square
    formed by a minimum of 6 tiles is 48. -/
theorem tile_square_side_length
  (tile_width : ℕ) (tile_length : ℕ) (min_tiles : ℕ)
  (hw : tile_width = 16)
  (hl : tile_length = 24)
  (hm : min_tiles = 6) :
  2 * tile_length = 3 * tile_width ∧ 2 * tile_length = 48 := by
  sorry

#check tile_square_side_length

end NUMINAMATH_CALUDE_tile_square_side_length_l3596_359600


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3596_359635

/-- Given a geometric sequence of 9 terms where the first term is 4 and the last term is 2097152,
    prove that the 7th term is 1048576 -/
theorem seventh_term_of_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, 1 ≤ n → n < 9 → a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 4 →                                            -- first term
  a 9 = 2097152 →                                      -- last term
  a 7 = 1048576 :=                                     -- seventh term to prove
by sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3596_359635


namespace NUMINAMATH_CALUDE_parabola_b_value_l3596_359602

/-- Given a parabola y = ax^2 + bx + c with vertex (p, p) and y-intercept (0, -2p), where p ≠ 0, 
    the value of b is 6/p. -/
theorem parabola_b_value (a b c p : ℝ) (h_p : p ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 + p) →
  (a * 0^2 + b * 0 + c = -2 * p) →
  b = 6 / p := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3596_359602


namespace NUMINAMATH_CALUDE_train_passing_time_l3596_359691

theorem train_passing_time (slower_speed faster_speed : ℝ) (train_length : ℝ) : 
  slower_speed = 36 →
  faster_speed = 45 →
  train_length = 90.0072 →
  (train_length / ((slower_speed + faster_speed) * (1000 / 3600))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l3596_359691


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3596_359676

theorem cube_sum_theorem (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3596_359676


namespace NUMINAMATH_CALUDE_radians_to_degrees_l3596_359684

theorem radians_to_degrees (π : ℝ) (h : π > 0) :
  (8 * π / 5) * (180 / π) = 288 := by
  sorry

end NUMINAMATH_CALUDE_radians_to_degrees_l3596_359684


namespace NUMINAMATH_CALUDE_remainder_sum_l3596_359646

theorem remainder_sum (x y : ℤ) (hx : x % 90 = 75) (hy : y % 120 = 115) :
  (x + y) % 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3596_359646


namespace NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l3596_359681

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y - 7 = 0
def line2 (x y : ℝ) : Prop := 7*x + 15*y + 1 = 0
def line3 (x y : ℝ) : Prop := x + 2*y - 3 = 0
def result_line (x y : ℝ) : Prop := 3*x + 6*y - 2 = 0

-- Theorem statement
theorem line_through_intersection_and_parallel :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧  -- Intersection point satisfies both line1 and line2
    (∀ (x y : ℝ), line3 x y ↔ ∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ y - y₀ = -1/2 * (x - x₀)) ∧  -- line3 has slope -1/2
    (∀ (x y : ℝ), result_line x y ↔ ∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ y - y₀ = -1/2 * (x - x₀)) ∧  -- result_line has slope -1/2
    result_line x₀ y₀  -- result_line passes through the intersection point
  := by sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l3596_359681


namespace NUMINAMATH_CALUDE_sesame_seed_weight_scientific_notation_l3596_359639

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem sesame_seed_weight_scientific_notation :
  toScientificNotation 0.00000201 = ScientificNotation.mk 2.01 (-6) (by sorry) :=
sorry

end NUMINAMATH_CALUDE_sesame_seed_weight_scientific_notation_l3596_359639


namespace NUMINAMATH_CALUDE_minimum_pastries_for_trick_l3596_359648

/-- Represents a pastry with two fillings -/
structure Pastry where
  filling1 : Fin 10
  filling2 : Fin 10
  h : filling1 ≠ filling2

/-- The set of all possible pastries -/
def allPastries : Finset Pastry :=
  sorry

theorem minimum_pastries_for_trick :
  ∀ n : ℕ,
    (n < 36 →
      ∃ (remaining : Finset Pastry),
        remaining ⊆ allPastries ∧
        remaining.card = 45 - n ∧
        ∀ (p : Pastry),
          p ∈ remaining →
            ∃ (q : Pastry),
              q ∈ remaining ∧ q ≠ p ∧
              (p.filling1 = q.filling1 ∨ p.filling1 = q.filling2 ∨
               p.filling2 = q.filling1 ∨ p.filling2 = q.filling2)) ∧
    (n = 36 →
      ∀ (remaining : Finset Pastry),
        remaining ⊆ allPastries →
        remaining.card = 45 - n →
        ∀ (p : Pastry),
          p ∈ remaining →
            ∃ (broken : Finset Pastry),
              broken ⊆ allPastries ∧
              broken.card = n ∧
              (p.filling1 ∈ broken.image Pastry.filling1 ∪ broken.image Pastry.filling2 ∨
               p.filling2 ∈ broken.image Pastry.filling1 ∪ broken.image Pastry.filling2)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_pastries_for_trick_l3596_359648


namespace NUMINAMATH_CALUDE_canada_population_1998_l3596_359671

/-- The population of Canada in 1998 in millions -/
def canada_population_millions : ℝ := 30.3

/-- One million in standard form -/
def million : ℕ := 1000000

/-- Theorem: The population of Canada in 1998 was 30,300,000 -/
theorem canada_population_1998 : 
  (canada_population_millions * million : ℝ) = 30300000 := by sorry

end NUMINAMATH_CALUDE_canada_population_1998_l3596_359671


namespace NUMINAMATH_CALUDE_max_value_x_plus_1000y_l3596_359605

theorem max_value_x_plus_1000y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x + 2018 / y = 1000) (eq2 : 9 / x + y = 1) :
  ∃ (x' y' : ℝ), x' + 2018 / y' = 1000 ∧ 9 / x' + y' = 1 ∧
  ∀ (a b : ℝ), a + 2018 / b = 1000 → 9 / a + b = 1 → x' + 1000 * y' ≥ a + 1000 * b ∧
  x' + 1000 * y' = 1991 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_1000y_l3596_359605


namespace NUMINAMATH_CALUDE_jo_equals_alex_sum_l3596_359622

def roundToNearestMultipleOf5 (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def joSum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def alexSum (n : ℕ) : ℕ :=
  (Finset.range n).sum (roundToNearestMultipleOf5 ∘ (· + 1))

theorem jo_equals_alex_sum :
  joSum 100 = alexSum 100 := by
  sorry

end NUMINAMATH_CALUDE_jo_equals_alex_sum_l3596_359622


namespace NUMINAMATH_CALUDE_unique_number_property_l3596_359609

theorem unique_number_property : ∃! x : ℚ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l3596_359609


namespace NUMINAMATH_CALUDE_maria_age_l3596_359643

theorem maria_age (maria ann : ℕ) : 
  maria = ann - 3 →
  maria - 4 = (ann - 4) / 2 →
  maria = 7 := by sorry

end NUMINAMATH_CALUDE_maria_age_l3596_359643


namespace NUMINAMATH_CALUDE_keystone_arch_angle_l3596_359682

/-- Represents a keystone arch composed of congruent isosceles trapezoids -/
structure KeystoneArch where
  num_trapezoids : ℕ
  trapezoids_congruent : Bool
  trapezoids_isosceles : Bool
  bottom_sides_horizontal : Bool

/-- Calculates the smaller interior angle of a trapezoid in a keystone arch -/
def smaller_interior_angle (arch : KeystoneArch) : ℝ :=
  if arch.num_trapezoids = 8 ∧ 
     arch.trapezoids_congruent ∧ 
     arch.trapezoids_isosceles ∧ 
     arch.bottom_sides_horizontal
  then 78.75
  else 0

/-- Theorem stating that the smaller interior angle of each trapezoid in the specified keystone arch is 78.75° -/
theorem keystone_arch_angle (arch : KeystoneArch) :
  arch.num_trapezoids = 8 ∧ 
  arch.trapezoids_congruent ∧ 
  arch.trapezoids_isosceles ∧ 
  arch.bottom_sides_horizontal →
  smaller_interior_angle arch = 78.75 := by
  sorry

end NUMINAMATH_CALUDE_keystone_arch_angle_l3596_359682


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3596_359689

theorem quadratic_root_relation (a b : ℝ) (h : a ≠ 0) :
  (a * 2019^2 + b * 2019 - 1 = 0) →
  (a * (2020 - 1)^2 + b * (2020 - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3596_359689


namespace NUMINAMATH_CALUDE_university_diploma_percentage_l3596_359677

theorem university_diploma_percentage
  (no_diploma_with_job : Real)
  (diploma_without_job : Real)
  (job_of_choice : Real)
  (h1 : no_diploma_with_job = 0.18)
  (h2 : diploma_without_job = 0.25)
  (h3 : job_of_choice = 0.4) :
  (job_of_choice - no_diploma_with_job) + (diploma_without_job * (1 - job_of_choice)) = 0.37 := by
sorry

end NUMINAMATH_CALUDE_university_diploma_percentage_l3596_359677


namespace NUMINAMATH_CALUDE_four_digit_multiple_of_65_l3596_359674

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_number (n : ℕ) : ℕ :=
  let d := n % 10
  let c := (n / 10) % 10
  let b := (n / 100) % 10
  let a := n / 1000
  1000 * d + 100 * c + 10 * b + a

theorem four_digit_multiple_of_65 :
  ∃! n : ℕ, is_four_digit n ∧ 
            65 ∣ n ∧ 
            65 ∣ (reverse_number n) ∧
            n = 5005 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_multiple_of_65_l3596_359674


namespace NUMINAMATH_CALUDE_second_square_area_equal_l3596_359624

/-- An isosceles right triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- The length of a leg of the isosceles right triangle -/
  leg : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed with two vertices on one leg, one on the hypotenuse, and one on the other leg -/
  inscribed : square_side > 0 ∧ square_side < leg
  /-- The area of the inscribed square is 625 cm² -/
  area_condition : square_side ^ 2 = 625

/-- The area of another inscribed square in the same triangle -/
def second_square_area (triangle : IsoscelesRightTriangleWithSquare) : ℝ :=
  triangle.square_side ^ 2

theorem second_square_area_equal (triangle : IsoscelesRightTriangleWithSquare) :
  second_square_area triangle = 625 := by
  sorry

end NUMINAMATH_CALUDE_second_square_area_equal_l3596_359624


namespace NUMINAMATH_CALUDE_percent_of_percent_l3596_359693

theorem percent_of_percent (y : ℝ) : (21 / 100) * y = (30 / 100) * ((70 / 100) * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l3596_359693


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l3596_359615

/-- Calculates the surface area of a cuboid given its length, breadth, and height. -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + length * height + breadth * height)

/-- Theorem stating that the surface area of a cuboid with length 12, breadth 14, and height 7 is 700. -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 12 14 7 = 700 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l3596_359615


namespace NUMINAMATH_CALUDE_apple_price_correct_l3596_359621

/-- The price of one apple in dollars -/
def apple_price : ℚ := 49/30

/-- The price of one orange in dollars -/
def orange_price : ℚ := 3/4

/-- The number of apples that equal the price of 2 watermelons or 3 pineapples -/
def apple_equiv : ℕ := 6

/-- The number of watermelons that equal the price of 6 apples or 3 pineapples -/
def watermelon_equiv : ℕ := 2

/-- The number of pineapples that equal the price of 6 apples or 2 watermelons -/
def pineapple_equiv : ℕ := 3

/-- The number of oranges bought -/
def oranges_bought : ℕ := 24

/-- The number of apples bought -/
def apples_bought : ℕ := 18

/-- The number of watermelons bought -/
def watermelons_bought : ℕ := 12

/-- The number of pineapples bought -/
def pineapples_bought : ℕ := 18

/-- The total bill in dollars -/
def total_bill : ℚ := 165

theorem apple_price_correct :
  apple_price * apple_equiv = apple_price * watermelon_equiv * 3 ∧
  apple_price * 2 * pineapple_equiv = apple_price * watermelon_equiv * 3 ∧
  orange_price * oranges_bought + apple_price * apples_bought + 
  (apple_price * 3) * watermelons_bought + (apple_price * 2) * pineapples_bought = total_bill :=
by sorry

end NUMINAMATH_CALUDE_apple_price_correct_l3596_359621


namespace NUMINAMATH_CALUDE_ratio_ac_to_bd_l3596_359608

/-- Given points A, B, C, D, and E on a line in that order, with given distances between consecutive points,
    prove that the ratio of AC to BD is 7/6. -/
theorem ratio_ac_to_bd (A B C D E : ℝ) 
  (h_order : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_ab : B - A = 3)
  (h_bc : C - B = 4)
  (h_cd : D - C = 2)
  (h_de : E - D = 3) :
  (C - A) / (D - B) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_ac_to_bd_l3596_359608


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3596_359634

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : a 2 = 1/2) 
  (h2 : a 5 = 4) 
  (h_geom : ∀ n : ℕ, n ≥ 1 → ∃ q : ℝ, a (n + 1) = a n * q) :
  ∃ q : ℝ, (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * q) ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3596_359634


namespace NUMINAMATH_CALUDE_cone_volume_with_inscribed_sphere_l3596_359618

/-- The volume of a cone with an inscribed sphere -/
theorem cone_volume_with_inscribed_sphere (r α : ℝ) (hr : r > 0) (hα : 0 < α ∧ α < π / 2) :
  ∃ V : ℝ, V = -π * r^3 * Real.tan (2 * α) / (24 * Real.cos α ^ 6) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_with_inscribed_sphere_l3596_359618


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l3596_359660

theorem jerrys_action_figures (initial : ℕ) : 
  initial + 11 - 10 = 8 → initial = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l3596_359660


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l3596_359644

theorem girls_to_boys_ratio (total_students : ℕ) 
  (girls boys : ℕ) 
  (girls_with_dogs : ℚ) 
  (boys_with_dogs : ℚ) 
  (total_with_dogs : ℕ) :
  total_students = 100 →
  girls + boys = total_students →
  girls_with_dogs = 1/5 →
  boys_with_dogs = 1/10 →
  total_with_dogs = 15 →
  girls_with_dogs * girls + boys_with_dogs * boys = total_with_dogs →
  girls = boys :=
by sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l3596_359644


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3596_359679

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  l1 : ℝ → ℝ → ℝ := λ x y => a * x + (a + 1) * y + 1
  l2 : ℝ → ℝ → ℝ := λ x y => x + a * y + 2

/-- Perpendicularity condition for two lines -/
def isPerpendicular (lines : TwoLines) : Prop :=
  lines.a * 1 + (lines.a + 1) * lines.a = 0

/-- Theorem stating that a = -2 is a sufficient but not necessary condition for perpendicularity -/
theorem sufficient_not_necessary (lines : TwoLines) :
  (lines.a = -2 → isPerpendicular lines) ∧
  ¬(isPerpendicular lines → lines.a = -2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3596_359679


namespace NUMINAMATH_CALUDE_area_of_large_rectangle_l3596_359656

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of a square -/
def Square.area (s : Square) : ℝ := s.side * s.side

/-- The theorem to be proved -/
theorem area_of_large_rectangle (shaded_square : Square) 
  (bottom_rect left_rect : Rectangle) :
  shaded_square.area = 4 →
  bottom_rect.width = shaded_square.side →
  bottom_rect.height + left_rect.height = shaded_square.side →
  left_rect.width + bottom_rect.width = shaded_square.side →
  (shaded_square.area + bottom_rect.area + left_rect.area = 12) := by
  sorry

end NUMINAMATH_CALUDE_area_of_large_rectangle_l3596_359656


namespace NUMINAMATH_CALUDE_equation_solution_l3596_359672

theorem equation_solution (x y : ℝ) :
  (x^4 + 1) * (y^4 + 1) = 4 * x^2 * y^2 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3596_359672


namespace NUMINAMATH_CALUDE_amanda_candy_bars_l3596_359632

/-- Amanda's candy bar problem -/
theorem amanda_candy_bars :
  let initial_bars : ℕ := 7
  let first_gift : ℕ := 3
  let new_bars : ℕ := 30
  let second_gift : ℕ := 4 * first_gift
  let kept_bars : ℕ := (initial_bars - first_gift) + (new_bars - second_gift)
  kept_bars = 22 := by sorry

end NUMINAMATH_CALUDE_amanda_candy_bars_l3596_359632


namespace NUMINAMATH_CALUDE_math_club_election_l3596_359664

theorem math_club_election (total_candidates : ℕ) (positions : ℕ) (past_officers : ℕ) 
  (h1 : total_candidates = 20)
  (h2 : positions = 5)
  (h3 : past_officers = 10) :
  (Nat.choose total_candidates positions) - (Nat.choose (total_candidates - past_officers) positions) = 15252 := by
sorry

end NUMINAMATH_CALUDE_math_club_election_l3596_359664


namespace NUMINAMATH_CALUDE_exponent_simplification_l3596_359638

theorem exponent_simplification :
  ((-5^2)^4 * (-5)^11) / ((-5)^3) = 5^16 := by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l3596_359638


namespace NUMINAMATH_CALUDE_awake_cats_l3596_359699

theorem awake_cats (total : ℕ) (asleep : ℕ) (awake : ℕ) : 
  total = 98 → asleep = 92 → awake = total - asleep → awake = 6 := by
  sorry

end NUMINAMATH_CALUDE_awake_cats_l3596_359699


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3596_359678

-- System 1
theorem system_one_solution (x y : ℚ) : 
  2 * x - y = 5 ∧ x - 1 = (2 * y - 1) / 2 → x = 9/2 ∧ y = 4 := by sorry

-- System 2
theorem system_two_solution (x y : ℚ) :
  3 * x + 2 * y = 1 ∧ 2 * x - 3 * y = 5 → x = 1 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3596_359678


namespace NUMINAMATH_CALUDE_complex_calculation_l3596_359627

theorem complex_calculation : (2 - I) / (1 - I) - I = 3/2 - 1/2 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l3596_359627


namespace NUMINAMATH_CALUDE_square_perimeter_proof_l3596_359698

theorem square_perimeter_proof (p1 p2 p3 : ℝ) : 
  p1 = 60 ∧ p2 = 48 ∧ p3 = 36 →
  (p1 / 4)^2 - (p2 / 4)^2 = (p3 / 4)^2 →
  p3 = 36 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_proof_l3596_359698


namespace NUMINAMATH_CALUDE_one_fourth_of_8_4_l3596_359686

theorem one_fourth_of_8_4 : (8.4 : ℚ) / 4 = 21 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_4_l3596_359686


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l3596_359685

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t2.a = k * t1.a ∧
    t2.b = k * t1.b ∧
    t2.c = k * t1.c

theorem similar_triangle_perimeter (t1 t2 : Triangle) :
  t1.isIsosceles ∧
  t1.a = 30 ∧ t1.b = 30 ∧ t1.c = 15 ∧
  t2.isSimilar t1 ∧
  min t2.a (min t2.b t2.c) = 45 →
  t2.perimeter = 225 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l3596_359685


namespace NUMINAMATH_CALUDE_fraction_simplification_l3596_359666

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (5 * Real.sqrt 2) / 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3596_359666


namespace NUMINAMATH_CALUDE_car_travel_distance_l3596_359649

theorem car_travel_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (travel_time_minutes : ℝ) :
  train_speed = 90 →
  car_speed_ratio = 5/6 →
  travel_time_minutes = 45 →
  let car_speed := car_speed_ratio * train_speed
  let travel_time_hours := travel_time_minutes / 60
  car_speed * travel_time_hours = 56.25 := by
sorry

end NUMINAMATH_CALUDE_car_travel_distance_l3596_359649


namespace NUMINAMATH_CALUDE_expansion_term_count_l3596_359640

/-- The number of terms in the expansion of (a+b+c)(a+d+e+f+g) -/
def expansion_terms : ℕ := 15

/-- The first polynomial (a+b+c) has 3 terms -/
def first_poly_terms : ℕ := 3

/-- The second polynomial (a+d+e+f+g) has 5 terms -/
def second_poly_terms : ℕ := 5

/-- Theorem stating that the expansion of (a+b+c)(a+d+e+f+g) has 15 terms -/
theorem expansion_term_count :
  expansion_terms = first_poly_terms * second_poly_terms := by
  sorry

end NUMINAMATH_CALUDE_expansion_term_count_l3596_359640


namespace NUMINAMATH_CALUDE_smallest_primes_satisfying_conditions_l3596_359636

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_primes_satisfying_conditions (p q : ℕ) :
  is_prime p ∧ is_prime q ∧ is_prime (p * q + 1) ∧ p - q > 40 →
  p = 53 ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_primes_satisfying_conditions_l3596_359636


namespace NUMINAMATH_CALUDE_arrangement_count_l3596_359633

/-- The number of volunteers --/
def num_volunteers : ℕ := 5

/-- The number of elderly people --/
def num_elderly : ℕ := 2

/-- The total number of units to arrange (volunteers + elderly unit) --/
def total_units : ℕ := num_volunteers + 1

/-- The number of possible positions for the elderly unit --/
def elderly_positions : ℕ := total_units - 2

theorem arrangement_count :
  (elderly_positions * Nat.factorial num_volunteers * Nat.factorial num_elderly) = 960 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3596_359633


namespace NUMINAMATH_CALUDE_pet_shelter_problem_l3596_359614

theorem pet_shelter_problem (total : ℕ) (apples chicken cheese : ℕ)
  (apples_chicken apples_cheese chicken_cheese : ℕ) (all_three : ℕ)
  (h_total : total = 100)
  (h_apples : apples = 20)
  (h_chicken : chicken = 70)
  (h_cheese : cheese = 10)
  (h_apples_chicken : apples_chicken = 7)
  (h_apples_cheese : apples_cheese = 3)
  (h_chicken_cheese : chicken_cheese = 5)
  (h_all_three : all_three = 2) :
  total - (apples + chicken + cheese
          - apples_chicken - apples_cheese - chicken_cheese
          + all_three) = 13 := by
  sorry

end NUMINAMATH_CALUDE_pet_shelter_problem_l3596_359614


namespace NUMINAMATH_CALUDE_reading_time_calculation_l3596_359652

def total_homework_time : ℕ := 120
def math_time : ℕ := 25
def spelling_time : ℕ := 30
def history_time : ℕ := 20
def science_time : ℕ := 15

theorem reading_time_calculation :
  total_homework_time - (math_time + spelling_time + history_time + science_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l3596_359652


namespace NUMINAMATH_CALUDE_abs_neg_2023_l3596_359613

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l3596_359613


namespace NUMINAMATH_CALUDE_inequality_sum_l3596_359645

theorem inequality_sum (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : x < a) (h2 : y < b) : x + y < a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_sum_l3596_359645


namespace NUMINAMATH_CALUDE_check_error_l3596_359667

theorem check_error (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 →
  100 * y + x - (100 * x + y) = 2376 →
  y = 2 * x + 12 →
  x = 12 ∧ y = 36 := by
sorry

end NUMINAMATH_CALUDE_check_error_l3596_359667


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l3596_359610

theorem p_or_q_is_true : 
  let p : Prop := 2 + 3 = 5
  let q : Prop := 5 < 4
  p ∨ q := by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l3596_359610


namespace NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l3596_359601

/-- Represents the dimensions and areas of a yard with a pool and flower beds. -/
structure YardLayout where
  yard_length : ℝ
  yard_width : ℝ
  pool_length : ℝ
  pool_width : ℝ
  trapezoid_side1 : ℝ
  trapezoid_side2 : ℝ

/-- Calculates the fraction of usable yard area occupied by flower beds. -/
def flower_bed_fraction (layout : YardLayout) : ℚ :=
  sorry

/-- Theorem stating that the fraction of usable yard occupied by flower beds is 9/260. -/
theorem flower_bed_fraction_is_correct (layout : YardLayout) : 
  layout.yard_length = 30 ∧ 
  layout.yard_width = 10 ∧ 
  layout.pool_length = 10 ∧ 
  layout.pool_width = 4 ∧
  layout.trapezoid_side1 = 16 ∧ 
  layout.trapezoid_side2 = 22 →
  flower_bed_fraction layout = 9 / 260 :=
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l3596_359601


namespace NUMINAMATH_CALUDE_bakers_sales_l3596_359631

/-- Baker's cake and pastry sales problem -/
theorem bakers_sales (cakes_sold : ℕ) (pastries_sold : ℕ) 
  (h1 : cakes_sold = 158) 
  (h2 : cakes_sold = pastries_sold + 11) : 
  pastries_sold = 147 := by
  sorry

end NUMINAMATH_CALUDE_bakers_sales_l3596_359631


namespace NUMINAMATH_CALUDE_initial_balance_is_20_l3596_359687

def football_club_balance (initial_balance : ℝ) : Prop :=
  let players_sold := 2
  let price_per_sold_player := 10
  let players_bought := 4
  let price_per_bought_player := 15
  let final_balance := 60
  
  initial_balance + players_sold * price_per_sold_player - 
  players_bought * price_per_bought_player = final_balance

theorem initial_balance_is_20 : 
  ∃ (initial_balance : ℝ), football_club_balance initial_balance ∧ initial_balance = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_balance_is_20_l3596_359687


namespace NUMINAMATH_CALUDE_complement_intersection_empty_l3596_359619

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 3, 4}
def N : Set Nat := {2, 4, 5}

theorem complement_intersection_empty :
  (U \ M) ∩ (U \ N) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_empty_l3596_359619


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3596_359659

theorem inequality_system_solutions :
  {x : ℕ | 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4} = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3596_359659


namespace NUMINAMATH_CALUDE_gcd_lcm_identity_l3596_359662

theorem gcd_lcm_identity (a b c : ℕ+) :
  (Nat.lcm (Nat.lcm a b) c)^2 / (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a) =
  (Nat.gcd (Nat.gcd a b) c)^2 / (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_identity_l3596_359662


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_882_l3596_359647

def sum_of_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_distinct_prime_factors_882 :
  sum_of_distinct_prime_factors 882 = 12 := by sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_882_l3596_359647


namespace NUMINAMATH_CALUDE_casey_pumping_time_l3596_359653

/-- The number of minutes Casey needs to spend pumping water -/
def pumpingTime (pumpRate : ℚ) (cornRows : ℕ) (cornPerRow : ℕ) (cornWater : ℚ) 
                (pigs : ℕ) (pigWater : ℚ) (ducks : ℕ) (duckWater : ℚ) : ℚ :=
  let totalCornWater := (cornRows * cornPerRow : ℚ) * cornWater
  let totalPigWater := (pigs : ℚ) * pigWater
  let totalDuckWater := (ducks : ℚ) * duckWater
  let totalWater := totalCornWater + totalPigWater + totalDuckWater
  totalWater / pumpRate

/-- Casey needs to spend 25 minutes pumping water -/
theorem casey_pumping_time :
  pumpingTime 3 4 15 (1/2) 10 4 20 (1/4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_casey_pumping_time_l3596_359653


namespace NUMINAMATH_CALUDE_sum_three_consecutive_not_prime_l3596_359650

theorem sum_three_consecutive_not_prime (n : ℕ) : ¬ Prime (3 * (n + 1)) := by
  sorry

#check sum_three_consecutive_not_prime

end NUMINAMATH_CALUDE_sum_three_consecutive_not_prime_l3596_359650


namespace NUMINAMATH_CALUDE_employed_females_percentage_l3596_359675

/-- Given the employment statistics of town X, calculate the percentage of employed females among all employed people. -/
theorem employed_females_percentage (total_employed : ℝ) (employed_males : ℝ) 
  (h1 : total_employed = 60) 
  (h2 : employed_males = 48) : 
  (total_employed - employed_males) / total_employed * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l3596_359675


namespace NUMINAMATH_CALUDE_iphone_defects_l3596_359669

theorem iphone_defects (
  initial_samsung : ℕ)
  (initial_iphone : ℕ)
  (final_samsung : ℕ)
  (final_iphone : ℕ)
  (total_sold : ℕ)
  (h1 : initial_samsung = 14)
  (h2 : initial_iphone = 8)
  (h3 : final_samsung = 10)
  (h4 : final_iphone = 5)
  (h5 : total_sold = 4)
  : initial_iphone - final_iphone - (total_sold - (initial_samsung - final_samsung)) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_iphone_defects_l3596_359669


namespace NUMINAMATH_CALUDE_simplify_expression_l3596_359673

theorem simplify_expression : (7^5 + 2^7) * (2^3 - (-1)^3)^8 = 729000080835 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3596_359673


namespace NUMINAMATH_CALUDE_regression_slope_l3596_359680

-- Define the linear function
def f (x : ℝ) : ℝ := 2 - 3 * x

-- Theorem statement
theorem regression_slope (x : ℝ) :
  f (x + 1) = f x - 3 := by
  sorry

end NUMINAMATH_CALUDE_regression_slope_l3596_359680


namespace NUMINAMATH_CALUDE_root_square_condition_l3596_359606

theorem root_square_condition (q : ℝ) : 
  (∃ a b : ℝ, a^2 - 12*a + q = 0 ∧ b^2 - 12*b + q = 0 ∧ (a = b^2 ∨ b = a^2)) ↔ 
  (q = -64 ∨ q = 27) := by
sorry

end NUMINAMATH_CALUDE_root_square_condition_l3596_359606


namespace NUMINAMATH_CALUDE_mason_tables_theorem_l3596_359623

/-- The number of tables Mason needs settings for -/
def num_tables : ℕ :=
  let silverware_weight : ℕ := 4  -- weight of one piece of silverware in ounces
  let silverware_per_setting : ℕ := 3  -- number of silverware pieces per setting
  let plate_weight : ℕ := 12  -- weight of one plate in ounces
  let plates_per_setting : ℕ := 2  -- number of plates per setting
  let settings_per_table : ℕ := 8  -- number of settings per table
  let backup_settings : ℕ := 20  -- number of backup settings
  let total_weight : ℕ := 5040  -- total weight of all settings in ounces

  -- Calculate the result
  (total_weight / (silverware_weight * silverware_per_setting + plate_weight * plates_per_setting) - backup_settings) / settings_per_table

theorem mason_tables_theorem : num_tables = 15 := by
  sorry

end NUMINAMATH_CALUDE_mason_tables_theorem_l3596_359623


namespace NUMINAMATH_CALUDE_symmetry_y_axis_values_l3596_359695

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal -/
def symmetric_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

theorem symmetry_y_axis_values :
  ∀ a b : ℝ, symmetric_y_axis a (-3) 2 b → a = -2 ∧ b = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_y_axis_values_l3596_359695


namespace NUMINAMATH_CALUDE_expression_defined_iff_l3596_359655

def expression_defined (x : ℝ) : Prop :=
  x > 2 ∧ x < 5

theorem expression_defined_iff (x : ℝ) :
  expression_defined x ↔ (∃ y : ℝ, y = (Real.log (5 - x)) / Real.sqrt (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l3596_359655


namespace NUMINAMATH_CALUDE_average_after_discarding_specific_case_l3596_359690

def average_after_discarding (n : ℕ) (initial_avg : ℚ) (discarded1 discarded2 : ℚ) : ℚ :=
  let initial_sum := n * initial_avg
  let remaining_sum := initial_sum - (discarded1 + discarded2)
  let remaining_count := n - 2
  remaining_sum / remaining_count

theorem average_after_discarding_specific_case :
  average_after_discarding 50 62 45 55 = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_average_after_discarding_specific_case_l3596_359690


namespace NUMINAMATH_CALUDE_min_value_expression_l3596_359607

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (5 * z) / (2 * x + y) + (5 * x) / (y + 2 * z) + (2 * y) / (x + z) + (x + y + z) / (x * y + y * z + z * x) ≥ 9 ∧
  ((5 * z) / (2 * x + y) + (5 * x) / (y + 2 * z) + (2 * y) / (x + z) + (x + y + z) / (x * y + y * z + z * x) = 9 ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3596_359607


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3596_359658

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) * (x + 3) > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ x > 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3596_359658


namespace NUMINAMATH_CALUDE_table_stool_equation_correctness_l3596_359617

/-- Represents a scenario with tables and stools -/
structure TableStoolScenario where
  numTables : ℕ
  numStools : ℕ
  totalItems : ℕ
  totalLegs : ℕ
  h_totalItems : numTables + numStools = totalItems
  h_totalLegs : 4 * numTables + 3 * numStools = totalLegs

/-- The correct system of equations for the given scenario -/
def correctSystem (x y : ℕ) : Prop :=
  x + y = 12 ∧ 4 * x + 3 * y = 40

theorem table_stool_equation_correctness :
  ∀ (scenario : TableStoolScenario),
    scenario.totalItems = 12 →
    scenario.totalLegs = 40 →
    correctSystem scenario.numTables scenario.numStools :=
by sorry

end NUMINAMATH_CALUDE_table_stool_equation_correctness_l3596_359617


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l3596_359657

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/5 + 1/6, 1/5 + 1/7, 1/5 + 1/3, 1/5 + 1/8, 1/5 + 1/9]
  (∀ x ∈ sums, x ≤ (1/5 + 1/3)) ∧ (1/5 + 1/3 = 8/15) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l3596_359657


namespace NUMINAMATH_CALUDE_xiao_ming_age_problem_l3596_359612

/-- Proves that Xiao Ming was 7 years old when his father's age was 5 times Xiao Ming's age -/
theorem xiao_ming_age_problem (current_age : ℕ) (father_current_age : ℕ) 
  (h1 : current_age = 12) (h2 : father_current_age = 40) : 
  ∃ (past_age : ℕ), past_age = 7 ∧ father_current_age - (current_age - past_age) = 5 * past_age :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_age_problem_l3596_359612


namespace NUMINAMATH_CALUDE_incorrect_statement_proof_l3596_359626

structure VisionSurvey where
  total_students : Nat
  sample_size : Nat
  is_about_vision : Bool

def is_correct_statement (s : VisionSurvey) (statement : String) : Prop :=
  match statement with
  | "The sample size is correct" => s.sample_size = 40
  | "The sample is about vision of selected students" => s.is_about_vision
  | "The population is about vision of all students" => s.is_about_vision
  | "The individual refers to each student" => false
  | _ => false

theorem incorrect_statement_proof (s : VisionSurvey) 
  (h1 : s.total_students = 400) 
  (h2 : s.sample_size = 40) 
  (h3 : s.is_about_vision = true) :
  ¬(is_correct_statement s "The individual refers to each student") := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_proof_l3596_359626


namespace NUMINAMATH_CALUDE_frustum_surface_area_l3596_359663

/-- The surface area of a frustum of a regular pyramid with square bases -/
theorem frustum_surface_area (top_side : ℝ) (bottom_side : ℝ) (slant_height : ℝ) :
  top_side = 2 →
  bottom_side = 4 →
  slant_height = 2 →
  let lateral_area := (top_side + bottom_side) * slant_height * 2
  let top_area := top_side ^ 2
  let bottom_area := bottom_side ^ 2
  lateral_area + top_area + bottom_area = 12 * Real.sqrt 3 + 20 := by
  sorry

end NUMINAMATH_CALUDE_frustum_surface_area_l3596_359663


namespace NUMINAMATH_CALUDE_floor_length_percentage_l3596_359651

theorem floor_length_percentage (length width area : ℝ) : 
  length = 23 ∧ 
  area = 529 / 3 ∧ 
  area = length * width → 
  length = width * 3 :=
by sorry

end NUMINAMATH_CALUDE_floor_length_percentage_l3596_359651


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_attained_l3596_359625

theorem min_value_of_expression (x : ℝ) : 
  (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 ≥ 2008 :=
by sorry

theorem equality_attained : 
  ∃ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 = 2008 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_attained_l3596_359625


namespace NUMINAMATH_CALUDE_fraction_inequality_range_l3596_359637

theorem fraction_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) ↔ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_range_l3596_359637


namespace NUMINAMATH_CALUDE_intersection_of_P_and_M_l3596_359668

-- Define the sets P and M
def P : Set ℝ := {y | ∃ x, y = x^2 - 6*x + 10}
def M : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + 8}

-- State the theorem
theorem intersection_of_P_and_M : P ∩ M = {y | 1 ≤ y ∧ y ≤ 9} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_M_l3596_359668


namespace NUMINAMATH_CALUDE_count_sevens_1_to_100_l3596_359630

/-- Count of digit 7 in numbers from 1 to 100 -/
def countSevens : ℕ → ℕ
| 0 => 0
| (n + 1) => (if n + 1 < 101 then (if (n + 1) % 10 = 7 || (n + 1) / 10 = 7 then 1 else 0) else 0) + countSevens n

theorem count_sevens_1_to_100 : countSevens 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_count_sevens_1_to_100_l3596_359630


namespace NUMINAMATH_CALUDE_hyperbola_circle_range_l3596_359641

theorem hyperbola_circle_range (a : ℝ) : 
  let P := (a > 1 ∨ a < -3)
  let Q := (-1 < a ∧ a < 3)
  (¬(P ∧ Q) ∧ ¬(¬Q)) → (-1 < a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_circle_range_l3596_359641


namespace NUMINAMATH_CALUDE_hair_brushing_ratio_l3596_359654

/-- The number of hairs washed down the drain -/
def hairs_washed : ℕ := 32

/-- The number of hairs grown back -/
def hairs_grown : ℕ := 49

/-- The ratio of hairs brushed out to hairs washed down the drain -/
def hair_ratio : ℚ := 1 / 2

theorem hair_brushing_ratio : 
  ∃ (hairs_brushed : ℕ), 
    hairs_brushed + hairs_washed + 1 = hairs_grown ∧ 
    hair_ratio = hairs_brushed / hairs_washed := by
  sorry

end NUMINAMATH_CALUDE_hair_brushing_ratio_l3596_359654


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3596_359694

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3596_359694


namespace NUMINAMATH_CALUDE_gcd_of_squares_gcd_130_215_310_131_216_309_l3596_359688

theorem gcd_of_squares (a b c d e f : ℤ) : 
  Int.gcd (a^2 + b^2 + c^2) (d^2 + e^2 + f^2) = 
  Int.gcd ((d^2 + e^2 + f^2) : ℤ) (|((a - d) * (a + d) + (b - e) * (b + e) + (c - f) * (c + f))|) :=
by sorry

theorem gcd_130_215_310_131_216_309 : 
  Int.gcd (130^2 + 215^2 + 310^2) (131^2 + 216^2 + 309^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_squares_gcd_130_215_310_131_216_309_l3596_359688


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3596_359665

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 4/3 of a right angle
  a + b = 4/3 * 90 →
  -- One angle is 40° larger than the other
  b = a + 40 →
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c →
  -- Sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 80°
  max a (max b c) = 80 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3596_359665


namespace NUMINAMATH_CALUDE_pairing_count_l3596_359670

/-- The number of bowls -/
def num_bowls : ℕ := 6

/-- The number of glasses -/
def num_glasses : ℕ := 4

/-- The number of fixed pairings -/
def num_fixed_pairings : ℕ := 1

/-- The number of remaining bowls after fixed pairing -/
def num_remaining_bowls : ℕ := num_bowls - num_fixed_pairings

/-- The number of remaining glasses after fixed pairing -/
def num_remaining_glasses : ℕ := num_glasses - num_fixed_pairings

/-- The total number of possible pairings -/
def total_pairings : ℕ := num_remaining_bowls * num_remaining_glasses + num_fixed_pairings

theorem pairing_count : total_pairings = 16 := by
  sorry

end NUMINAMATH_CALUDE_pairing_count_l3596_359670


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3596_359696

theorem max_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 14*x + 6*y + 6) :
  3*x + 4*y ≤ 73 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3596_359696


namespace NUMINAMATH_CALUDE_sports_club_overlap_l3596_359697

/-- Given a sports club with the following properties:
  * There are 30 total members
  * 17 members play badminton
  * 21 members play tennis
  * 2 members play neither badminton nor tennis
  This theorem proves that 10 members play both badminton and tennis. -/
theorem sports_club_overlap :
  ∀ (total badminton tennis neither : ℕ),
  total = 30 →
  badminton = 17 →
  tennis = 21 →
  neither = 2 →
  badminton + tennis - total + neither = 10 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l3596_359697


namespace NUMINAMATH_CALUDE_first_term_of_sequence_l3596_359604

/-- Given a sequence of points scored in a game where the second to sixth terms
    are 3, 5, 8, 12, and 17, and the differences between consecutive terms
    form an arithmetic sequence, prove that the first term of the sequence is 2. -/
theorem first_term_of_sequence (a : ℕ → ℕ) : 
  a 2 = 3 ∧ a 3 = 5 ∧ a 4 = 8 ∧ a 5 = 12 ∧ a 6 = 17 ∧ 
  (∃ d : ℕ, ∀ n : ℕ, n ≥ 2 → a (n+1) - a n = d + n - 2) →
  a 1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_first_term_of_sequence_l3596_359604


namespace NUMINAMATH_CALUDE_function_characterization_l3596_359661

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the conditions
def Condition1 (f : RealFunction) : Prop :=
  ∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v)

def Condition2 (f : RealFunction) : Prop :=
  ∀ u : ℝ, f u ≥ 0

-- State the theorem
theorem function_characterization (f : RealFunction) 
  (h1 : Condition1 f) (h2 : Condition2 f) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1/2) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3596_359661


namespace NUMINAMATH_CALUDE_prob_one_success_in_three_trials_l3596_359616

/-- The probability of exactly one success in three independent trials with success probability 3/4 -/
theorem prob_one_success_in_three_trials : 
  let p : ℚ := 3/4  -- Probability of success in each trial
  let n : ℕ := 3    -- Number of trials
  let k : ℕ := 1    -- Number of successes we're interested in
  Nat.choose n k * p^k * (1-p)^(n-k) = 9/64 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_success_in_three_trials_l3596_359616


namespace NUMINAMATH_CALUDE_lisa_savings_l3596_359620

theorem lisa_savings (x : ℚ) : 
  (x + 3/5 * x + 2 * (3/5 * x) = 3760 - 400) → x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_lisa_savings_l3596_359620


namespace NUMINAMATH_CALUDE_triangle_transformation_l3596_359628

-- Define the initial triangle
def initial_triangle : List (ℝ × ℝ) := [(0, 0), (1, 0), (0, 1)]

-- Define the transformation functions
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def translate_right (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2, p.2)

-- Define the composite transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_right (reflect_x_axis (rotate_180 p))

-- Theorem statement
theorem triangle_transformation :
  List.map transform initial_triangle = [(2, 0), (1, 0), (2, 1)] := by
  sorry

end NUMINAMATH_CALUDE_triangle_transformation_l3596_359628


namespace NUMINAMATH_CALUDE_alternating_color_probability_l3596_359692

/-- The probability of drawing 10 balls from a box containing 5 white and 5 black balls
    such that the colors alternate is equal to 1/126. -/
theorem alternating_color_probability (n : ℕ) (white_balls black_balls : ℕ) : 
  n = 10 → white_balls = 5 → black_balls = 5 →
  (Nat.choose n white_balls : ℚ)⁻¹ * 2 = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l3596_359692


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l3596_359611

/-- A quadratic function f(x) = a + bx - x^2 satisfying certain conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a + b * x - x^2

/-- The theorem stating the range of m for the given conditions -/
theorem quadratic_function_m_range (a b m : ℝ) :
  (∀ x, f a b (1 + x) = f a b (1 - x)) →
  (∀ x ≤ 4, Monotone (fun x => f a b (x + m))) →
  m ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l3596_359611
