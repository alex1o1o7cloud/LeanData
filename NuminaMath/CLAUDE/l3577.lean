import Mathlib

namespace NUMINAMATH_CALUDE_incentive_savings_l3577_357734

/-- Calculates the amount saved given an initial amount and spending percentages -/
def calculate_savings (initial_amount : ℝ) (food_percent : ℝ) (clothes_percent : ℝ) 
  (household_percent : ℝ) (savings_percent : ℝ) : ℝ :=
  let remaining_after_food := initial_amount * (1 - food_percent)
  let remaining_after_clothes := remaining_after_food * (1 - clothes_percent)
  let remaining_after_household := remaining_after_clothes * (1 - household_percent)
  remaining_after_household * savings_percent

/-- Theorem stating that given the specified spending pattern, 
    the amount saved from a $600 incentive is $171.36 -/
theorem incentive_savings : 
  calculate_savings 600 0.3 0.2 0.15 0.6 = 171.36 := by
  sorry

end NUMINAMATH_CALUDE_incentive_savings_l3577_357734


namespace NUMINAMATH_CALUDE_mike_unbroken_seashells_l3577_357786

/-- The number of unbroken seashells Mike found -/
def unbroken_seashells (total : ℕ) (broken : ℕ) : ℕ :=
  total - broken

/-- Theorem stating that Mike found 2 unbroken seashells -/
theorem mike_unbroken_seashells :
  unbroken_seashells 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mike_unbroken_seashells_l3577_357786


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l3577_357787

/-- Given an arithmetic sequence with 5 terms, first term 8, and common difference 8,
    prove that the average (mean) of the sequence is 24. -/
theorem arithmetic_sequence_average (a : Fin 5 → ℕ) 
  (h1 : a 0 = 8)
  (h2 : ∀ i : Fin 4, a (i + 1) = a i + 8) :
  (Finset.sum Finset.univ a) / 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l3577_357787


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3577_357712

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = 1 + Complex.I) :
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3577_357712


namespace NUMINAMATH_CALUDE_range_of_a_l3577_357773

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def B : Set ℝ := {x : ℝ | x ≥ 2}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (Set.univ \ B) ∪ A a = A a → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3577_357773


namespace NUMINAMATH_CALUDE_probability_not_adjacent_seats_l3577_357764

-- Define the number of seats
def num_seats : ℕ := 10

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Define the number of ways two people can sit next to each other in a row of seats
def adjacent_seats (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem probability_not_adjacent_seats :
  let total_ways := choose num_seats 2
  let adjacent_ways := adjacent_seats num_seats
  (total_ways - adjacent_ways) / total_ways = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_seats_l3577_357764


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3577_357779

theorem smaller_number_problem (x y : ℤ) : 
  x + y = 56 → y = x + 12 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3577_357779


namespace NUMINAMATH_CALUDE_system_solution_l3577_357749

theorem system_solution (a b c x y z : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : x * y = a) (h2 : y * z = b) (h3 : z * x = c) :
  (x = Real.sqrt (a * c / b) ∨ x = -Real.sqrt (a * c / b)) ∧
  (y = Real.sqrt (a * b / c) ∨ y = -Real.sqrt (a * b / c)) ∧
  (z = Real.sqrt (b * c / a) ∨ z = -Real.sqrt (b * c / a)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3577_357749


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3577_357789

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 5)
  parallel a b → x = -10 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3577_357789


namespace NUMINAMATH_CALUDE_wax_for_feathers_l3577_357723

/-- The total amount of wax required for feathers given the current amount and additional amount needed -/
theorem wax_for_feathers (current_amount additional_amount : ℕ) : 
  current_amount = 20 → additional_amount = 146 → 
  current_amount + additional_amount = 166 := by
  sorry

end NUMINAMATH_CALUDE_wax_for_feathers_l3577_357723


namespace NUMINAMATH_CALUDE_laundry_detergent_price_l3577_357782

/-- Calculates the initial price of laundry detergent given grocery shopping conditions --/
theorem laundry_detergent_price
  (initial_amount : ℝ)
  (milk_price : ℝ)
  (bread_price : ℝ)
  (banana_price_per_pound : ℝ)
  (banana_quantity : ℝ)
  (detergent_coupon : ℝ)
  (amount_left : ℝ)
  (h1 : initial_amount = 20)
  (h2 : milk_price = 4)
  (h3 : bread_price = 3.5)
  (h4 : banana_price_per_pound = 0.75)
  (h5 : banana_quantity = 2)
  (h6 : detergent_coupon = 1.25)
  (h7 : amount_left = 4) :
  let discounted_milk_price := milk_price / 2
  let banana_total := banana_price_per_pound * banana_quantity
  let other_items_cost := discounted_milk_price + bread_price + banana_total
  let total_spent := initial_amount - amount_left
  let detergent_price_with_coupon := total_spent - other_items_cost
  let initial_detergent_price := detergent_price_with_coupon + detergent_coupon
  initial_detergent_price = 10.25 := by
sorry

end NUMINAMATH_CALUDE_laundry_detergent_price_l3577_357782


namespace NUMINAMATH_CALUDE_total_colorings_l3577_357772

/-- Represents the number of ways to color a 2x2 square with 3 colors,
    such that adjacent cells have different colors -/
def ways_to_color_2x2 : ℕ := 18

/-- Represents the number of ways to color a figure consisting of 6 cells
    adjacent to a central cell, such that adjacent cells have different colors -/
def ways_to_color_figure : ℕ := 48

/-- Represents the total number of cells in the entire figure -/
def total_cells : ℕ := 25

/-- Represents the number of identical figures surrounding the central cell -/
def num_surrounding_figures : ℕ := 4

/-- Represents the number of available colors -/
def num_colors : ℕ := 3

/-- Theorem stating the total number of ways to color the entire figure -/
theorem total_colorings :
  (num_colors : ℕ) * (ways_to_color_figure ^ num_surrounding_figures) =
  3 * 48^4 := by sorry

end NUMINAMATH_CALUDE_total_colorings_l3577_357772


namespace NUMINAMATH_CALUDE_surface_area_cube_with_holes_l3577_357785

/-- The surface area of a cube with smaller cubes dug out from each face -/
theorem surface_area_cube_with_holes (edge_length : ℝ) (hole_length : ℝ) : 
  edge_length = 10 →
  hole_length = 2 →
  (6 * edge_length^2) - (6 * hole_length^2) + (6 * 5 * hole_length^2) = 696 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_cube_with_holes_l3577_357785


namespace NUMINAMATH_CALUDE_factorial_ratio_l3577_357701

theorem factorial_ratio : Nat.factorial 45 / Nat.factorial 42 = 85140 := by sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3577_357701


namespace NUMINAMATH_CALUDE_slope_of_line_intersecting_ellipse_l3577_357739

/-- Given an ellipse and a line that intersects it, this theorem proves
    that if (1,1) is the midpoint of the chord formed by the intersection,
    then the slope of the line is -1/4. -/
theorem slope_of_line_intersecting_ellipse 
  (x₁ y₁ x₂ y₂ : ℝ) : 
  x₁^2/36 + y₁^2/9 = 1 →   -- Point (x₁, y₁) is on the ellipse
  x₂^2/36 + y₂^2/9 = 1 →   -- Point (x₂, y₂) is on the ellipse
  (x₁ + x₂)/2 = 1 →        -- x-coordinate of midpoint is 1
  (y₁ + y₂)/2 = 1 →        -- y-coordinate of midpoint is 1
  (y₂ - y₁)/(x₂ - x₁) = -1/4 :=  -- Slope of the line
by sorry

end NUMINAMATH_CALUDE_slope_of_line_intersecting_ellipse_l3577_357739


namespace NUMINAMATH_CALUDE_cubic_system_solution_l3577_357742

theorem cubic_system_solution (a b c : ℝ) : 
  a + b + c = 3 ∧ 
  a^2 + b^2 + c^2 = 35 ∧ 
  a^3 + b^3 + c^3 = 99 → 
  ({a, b, c} : Set ℝ) = {1, -3, 5} :=
sorry

end NUMINAMATH_CALUDE_cubic_system_solution_l3577_357742


namespace NUMINAMATH_CALUDE_midpoint_sum_scaled_triangle_l3577_357781

theorem midpoint_sum_scaled_triangle (a b c : ℝ) (h : a + b + c = 18) :
  let scaled_midpoint_sum := (2*a + 2*b) + (2*a + 2*c) + (2*b + 2*c)
  scaled_midpoint_sum = 36 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_scaled_triangle_l3577_357781


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_lines_l3577_357713

def vector1 : ℝ × ℝ := (4, -1)
def vector2 : ℝ × ℝ := (2, 5)

theorem cosine_of_angle_between_lines :
  let v1 := vector1
  let v2 := vector2
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2
  let magnitude1 := Real.sqrt (v1.1^2 + v1.2^2)
  let magnitude2 := Real.sqrt (v2.1^2 + v2.2^2)
  dot_product / (magnitude1 * magnitude2) = 3 / Real.sqrt 493 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_lines_l3577_357713


namespace NUMINAMATH_CALUDE_james_purchase_cost_l3577_357721

def shirts_count : ℕ := 10
def shirt_price : ℕ := 6
def pants_price : ℕ := 8

def pants_count : ℕ := shirts_count / 2

def total_cost : ℕ := shirts_count * shirt_price + pants_count * pants_price

theorem james_purchase_cost : total_cost = 100 := by
  sorry

end NUMINAMATH_CALUDE_james_purchase_cost_l3577_357721


namespace NUMINAMATH_CALUDE_arithmetic_progression_theorem_l3577_357755

/-- An arithmetic progression with n terms -/
structure ArithmeticProgression where
  n : ℕ
  a : ℕ → ℕ
  d : ℕ
  progression : ∀ i, i < n → a (i + 1) = a i + d

/-- The sum of an arithmetic progression -/
def sum (ap : ArithmeticProgression) : ℕ :=
  (ap.n * (2 * ap.a 0 + (ap.n - 1) * ap.d)) / 2

theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  sum ap = 112 ∧
  ap.a 1 * ap.d = 30 ∧
  ap.a 2 + ap.a 4 = 32 →
  ap.n = 7 ∧
  ((ap.a 0 = 7 ∧ ap.a 1 = 10 ∧ ap.a 2 = 13) ∨
   (ap.a 0 = 1 ∧ ap.a 1 = 6 ∧ ap.a 2 = 11)) :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_progression_theorem_l3577_357755


namespace NUMINAMATH_CALUDE_tom_average_increase_l3577_357725

def tom_scores : List ℝ := [92, 89, 91, 93]

theorem tom_average_increase :
  let first_three := tom_scores.take 3
  let all_four := tom_scores
  let avg_first_three := first_three.sum / first_three.length
  let avg_all_four := all_four.sum / all_four.length
  avg_all_four - avg_first_three = 0.58 := by
  sorry

end NUMINAMATH_CALUDE_tom_average_increase_l3577_357725


namespace NUMINAMATH_CALUDE_solution_set_of_trig_equation_l3577_357756

theorem solution_set_of_trig_equation :
  let S : Set ℝ := {x | 5 * Real.sin x = 4 + 2 * Real.cos (2 * x)}
  S = {x | ∃ k : ℤ, x = Real.arcsin (3/4) + 2 * k * Real.pi ∨ 
                    x = Real.pi - Real.arcsin (3/4) + 2 * k * Real.pi} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_trig_equation_l3577_357756


namespace NUMINAMATH_CALUDE_sqrt_square_negative_two_l3577_357708

theorem sqrt_square_negative_two : Real.sqrt ((-2)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_negative_two_l3577_357708


namespace NUMINAMATH_CALUDE_solve_equations_l3577_357780

theorem solve_equations :
  (∀ x : ℝ, 4 * x = 20 → x = 5) ∧
  (∀ x : ℝ, x - 18 = 40 → x = 58) ∧
  (∀ x : ℝ, x / 7 = 12 → x = 84) ∧
  (∀ n : ℝ, 8 * n / 2 = 15 → n = 15 / 4) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l3577_357780


namespace NUMINAMATH_CALUDE_divide_into_triominoes_l3577_357784

/-- An L-shaped triomino is a shape consisting of three connected cells in an L shape -/
def LShapedTriomino : Type := Unit

/-- A grid is represented by its size, which is always of the form 6n+1 for some natural number n -/
structure Grid :=
  (n : ℕ)

/-- A cell in the grid, represented by its row and column coordinates -/
structure Cell :=
  (row : ℕ)
  (col : ℕ)

/-- A configuration is a grid with one cell removed -/
structure Configuration :=
  (grid : Grid)
  (removed_cell : Cell)

/-- A division of a configuration into L-shaped triominoes -/
def Division (config : Configuration) : Type := Unit

/-- The main theorem: any configuration can be divided into L-shaped triominoes -/
theorem divide_into_triominoes (config : Configuration) : 
  ∃ (d : Division config), True :=
sorry

end NUMINAMATH_CALUDE_divide_into_triominoes_l3577_357784


namespace NUMINAMATH_CALUDE_orchard_fruit_sales_l3577_357776

theorem orchard_fruit_sales (total_fruit : ℕ) (frozen_fruit : ℕ) (fresh_fruit : ℕ) :
  total_fruit = 9792 →
  frozen_fruit = 3513 →
  fresh_fruit = total_fruit - frozen_fruit →
  fresh_fruit = 6279 := by
sorry

end NUMINAMATH_CALUDE_orchard_fruit_sales_l3577_357776


namespace NUMINAMATH_CALUDE_bill_throws_21_objects_l3577_357792

/-- The number of sticks Ted throws -/
def ted_sticks : ℕ := 10

/-- The number of rocks Ted throws -/
def ted_rocks : ℕ := 10

/-- The number of sticks Bill throws -/
def bill_sticks : ℕ := ted_sticks + 6

/-- The number of rocks Bill throws -/
def bill_rocks : ℕ := ted_rocks / 2

/-- The total number of objects Bill throws -/
def bill_total : ℕ := bill_sticks + bill_rocks

theorem bill_throws_21_objects : bill_total = 21 := by
  sorry

end NUMINAMATH_CALUDE_bill_throws_21_objects_l3577_357792


namespace NUMINAMATH_CALUDE_max_area_triangle_l3577_357797

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the properties
def isAcute (t : Triangle) : Prop := sorry

def isSimilar (t1 t2 : Triangle) : Prop := sorry

def circumscribes (t1 t2 : Triangle) : Prop := sorry

def area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem max_area_triangle 
  (A₀B₀C₀ : Triangle) 
  (A'B'C' : Triangle) 
  (h1 : isAcute A₀B₀C₀) 
  (h2 : isAcute A'B'C') :
  ∃ (A₁B₁C₁ : Triangle),
    isSimilar A₁B₁C₁ A'B'C' ∧ 
    circumscribes A₁B₁C₁ A₀B₀C₀ ∧
    ∀ (ABC : Triangle),
      isSimilar ABC A'B'C' → 
      circumscribes ABC A₀B₀C₀ → 
      area ABC ≤ area A₁B₁C₁ :=
sorry

end NUMINAMATH_CALUDE_max_area_triangle_l3577_357797


namespace NUMINAMATH_CALUDE_inverse_of_i_minus_three_inverse_i_l3577_357741

-- Define i as a complex number with i^2 = -1
def i : ℂ := Complex.I

-- State the theorem
theorem inverse_of_i_minus_three_inverse_i (h : i^2 = -1) :
  (i - 3 * i⁻¹)⁻¹ = -i/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_i_minus_three_inverse_i_l3577_357741


namespace NUMINAMATH_CALUDE_joes_kids_haircuts_l3577_357731

-- Define the time it takes for each type of haircut
def womens_haircut_time : ℕ := 50
def mens_haircut_time : ℕ := 15
def kids_haircut_time : ℕ := 25

-- Define the number of women's and men's haircuts
def num_womens_haircuts : ℕ := 3
def num_mens_haircuts : ℕ := 2

-- Define the total time spent cutting hair
def total_time : ℕ := 255

-- Define a function to calculate the number of kids' haircuts
def num_kids_haircuts (w m k : ℕ) : ℕ :=
  (total_time - (w * womens_haircut_time + m * mens_haircut_time)) / k

-- Theorem statement
theorem joes_kids_haircuts :
  num_kids_haircuts num_womens_haircuts num_mens_haircuts kids_haircut_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_joes_kids_haircuts_l3577_357731


namespace NUMINAMATH_CALUDE_math_competition_problem_l3577_357788

theorem math_competition_problem (a b : ℝ) 
  (ha : 4 / a^4 - 2 / a^2 - 3 = 0) 
  (hb : b^4 + b^2 - 3 = 0) : 
  (a^4 * b^4 + 4) / a^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_problem_l3577_357788


namespace NUMINAMATH_CALUDE_pythagoras_academy_olympiad_students_l3577_357762

/-- The number of distinct students taking the Math Olympiad at Pythagoras Academy -/
def distinctStudents (eulerStudents gaussStudents fibonacciStudents doubleCountedStudents : ℕ) : ℕ :=
  eulerStudents + gaussStudents + fibonacciStudents - doubleCountedStudents

/-- Theorem stating the number of distinct students taking the Math Olympiad -/
theorem pythagoras_academy_olympiad_students :
  distinctStudents 15 10 12 3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_pythagoras_academy_olympiad_students_l3577_357762


namespace NUMINAMATH_CALUDE_zebra_crossing_distance_l3577_357707

/-- Given a boulevard with zebra crossing, calculate the distance between stripes --/
theorem zebra_crossing_distance (boulevard_width : ℝ) (stripe_length : ℝ) (gate_distance : ℝ)
  (h1 : boulevard_width = 60)
  (h2 : stripe_length = 65)
  (h3 : gate_distance = 22) :
  (boulevard_width * gate_distance) / stripe_length = 20.31 := by
  sorry

end NUMINAMATH_CALUDE_zebra_crossing_distance_l3577_357707


namespace NUMINAMATH_CALUDE_infinitely_many_a_for_perfect_cube_l3577_357767

theorem infinitely_many_a_for_perfect_cube (n : ℕ) :
  ∃ (f : ℕ → ℤ), Function.Injective f ∧ ∀ (k : ℕ), ∃ (m : ℕ), (n^6 + 3 * (f k) : ℤ) = m^3 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_a_for_perfect_cube_l3577_357767


namespace NUMINAMATH_CALUDE_triangle_area_l3577_357748

/-- Given a triangle with perimeter 36 and inradius 2.5, prove its area is 45 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
    (h1 : perimeter = 36) 
    (h2 : inradius = 2.5) 
    (h3 : area = inradius * (perimeter / 2)) : 
  area = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3577_357748


namespace NUMINAMATH_CALUDE_similar_quadrilaterals_rectangle_areas_l3577_357703

/-- Given two similar quadrilaterals with sides (a, b, c, d) and (a', b', c', d') respectively,
    prove that the areas of rectangles formed by pairs of corresponding sides
    are in proportion to the squares of the sides of the original quadrilaterals. -/
theorem similar_quadrilaterals_rectangle_areas
  (a b c d a' b' c' d' : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0)
  (h_similar : ∃ (k : ℝ), k > 0 ∧ a' = k * a ∧ b' = k * b ∧ c' = k * c ∧ d' = k * d) :
  ∃ (m : ℝ), m > 0 ∧
    a * a' / (b * b') = a^2 / b^2 ∧
    b * b' / (c * c') = b^2 / c^2 ∧
    c * c' / (d * d') = c^2 / d^2 ∧
    d * d' / (a * a') = d^2 / a^2 :=
by sorry

end NUMINAMATH_CALUDE_similar_quadrilaterals_rectangle_areas_l3577_357703


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3577_357794

/-- If a point P with coordinates (m-3, 2+m) lies on the x-axis, then its coordinates are (-5, 0). -/
theorem point_on_x_axis (m : ℝ) :
  (∃ P : ℝ × ℝ, P = (m - 3, 2 + m) ∧ P.2 = 0) →
  (∃ P : ℝ × ℝ, P = (m - 3, 2 + m) ∧ P = (-5, 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3577_357794


namespace NUMINAMATH_CALUDE_find_B_l3577_357706

theorem find_B (x y A : ℕ) (hx : x > 1) (hy : y > 1) (hxy : x > y) 
  (heq : x * y = x + y + A) : x / y = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l3577_357706


namespace NUMINAMATH_CALUDE_mans_speed_with_current_is_15_l3577_357711

/-- Given a man's speed against a current and the speed of the current,
    calculate the man's speed with the current. -/
def mans_speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions,
    the man's speed with the current is 15 km/hr. -/
theorem mans_speed_with_current_is_15
  (speed_against_current : ℝ)
  (current_speed : ℝ)
  (h1 : speed_against_current = 8.6)
  (h2 : current_speed = 3.2) :
  mans_speed_with_current speed_against_current current_speed = 15 := by
  sorry

#eval mans_speed_with_current 8.6 3.2

end NUMINAMATH_CALUDE_mans_speed_with_current_is_15_l3577_357711


namespace NUMINAMATH_CALUDE_circle_tangent_point_relation_l3577_357774

/-- Given a circle C and a point A satisfying certain conditions, prove that a + (3/2)b = 3 -/
theorem circle_tangent_point_relation (a b : ℝ) : 
  (∃ (x y : ℝ), (x - 2)^2 + (y - 3)^2 = 1) →  -- Circle C equation
  (∃ (m_x m_y : ℝ), (m_x - 2)^2 + (m_y - 3)^2 = 1 ∧ 
    ((m_x - a) * (m_x - 2) + (m_y - b) * (m_y - 3) = 0)) →  -- AM is tangent to C at M
  ((a - 2)^2 + (b - 3)^2 - 1 = a^2 + b^2) →  -- |AM| = |AO|
  a + (3/2) * b = 3 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_point_relation_l3577_357774


namespace NUMINAMATH_CALUDE_juice_left_in_cup_l3577_357765

theorem juice_left_in_cup (consumed : Rat) (h : consumed = 4/6) :
  1 - consumed = 2/6 ∨ 1 - consumed = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_juice_left_in_cup_l3577_357765


namespace NUMINAMATH_CALUDE_star_sum_theorem_l3577_357702

/-- The star operation defined for positive integers a and b -/
def star (a b : ℕ+) : ℕ := a.val^b.val - a.val * b.val + 5

/-- Theorem stating that if a ★ b = 13 for a ≥ 2 and b ≥ 3, then a + b = 6 -/
theorem star_sum_theorem (a b : ℕ+) (ha : 2 ≤ a.val) (hb : 3 ≤ b.val) 
  (h : star a b = 13) : a.val + b.val = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_sum_theorem_l3577_357702


namespace NUMINAMATH_CALUDE_peggy_bought_three_folders_l3577_357740

/-- Represents the number of sheets in each folder -/
def sheets_per_folder : ℕ := 10

/-- Represents the number of stickers per sheet in the red folder -/
def red_stickers_per_sheet : ℕ := 3

/-- Represents the number of stickers per sheet in the green folder -/
def green_stickers_per_sheet : ℕ := 2

/-- Represents the number of stickers per sheet in the blue folder -/
def blue_stickers_per_sheet : ℕ := 1

/-- Represents the total number of stickers used -/
def total_stickers : ℕ := 60

/-- Theorem stating that Peggy bought 3 folders -/
theorem peggy_bought_three_folders :
  (sheets_per_folder * red_stickers_per_sheet) +
  (sheets_per_folder * green_stickers_per_sheet) +
  (sheets_per_folder * blue_stickers_per_sheet) = total_stickers :=
by sorry

end NUMINAMATH_CALUDE_peggy_bought_three_folders_l3577_357740


namespace NUMINAMATH_CALUDE_race_time_difference_l3577_357796

/-- Race parameters and runner speeds -/
def race_distance : ℕ := 12
def malcolm_speed : ℕ := 7
def joshua_speed : ℕ := 8

/-- Theorem stating the time difference between Malcolm and Joshua finishing the race -/
theorem race_time_difference : 
  joshua_speed * race_distance - malcolm_speed * race_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l3577_357796


namespace NUMINAMATH_CALUDE_democrat_ratio_l3577_357738

/-- Proves that the ratio of democrats to total participants is 1:3 given the specified conditions -/
theorem democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) :
  total_participants = 750 →
  female_democrats = 125 →
  (∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    2 * female_democrats = female_participants ∧
    4 * female_democrats = male_participants) →
  (3 * (2 * female_democrats) : ℚ) / total_participants = 1 := by
  sorry

end NUMINAMATH_CALUDE_democrat_ratio_l3577_357738


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l3577_357770

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) : 
  n1 = 22 →
  n2 = 28 →
  avg1 = 40 →
  avg2 = 60 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l3577_357770


namespace NUMINAMATH_CALUDE_triangle_area_change_l3577_357714

theorem triangle_area_change (base height : ℝ) (base_new height_new : ℝ) :
  base_new = base * 1.1 →
  height_new = height * 0.95 →
  (base_new * height_new) / (base * height) - 1 = 0.045 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_change_l3577_357714


namespace NUMINAMATH_CALUDE_football_game_spectators_l3577_357795

/-- Represents the number of spectators at a football game --/
structure Spectators :=
  (adults : ℕ)
  (children : ℕ)
  (vips : ℕ)

/-- Conditions of the football game spectator problem --/
def football_game_conditions (s : Spectators) : Prop :=
  s.vips = 20 ∧
  s.children = s.adults / 2 ∧
  2 * s.adults + 2 * s.children + 2 * s.vips = 310

/-- Theorem stating the correct number of spectators --/
theorem football_game_spectators :
  ∃ (s : Spectators), football_game_conditions s ∧
    s.adults = 90 ∧ s.children = 45 ∧ s.vips = 20 ∧
    s.adults + s.children + s.vips = 155 :=
sorry


end NUMINAMATH_CALUDE_football_game_spectators_l3577_357795


namespace NUMINAMATH_CALUDE_wage_increase_percentage_l3577_357747

theorem wage_increase_percentage (original_wage new_wage : ℝ) 
  (h1 : original_wage = 28)
  (h2 : new_wage = 42) :
  (new_wage - original_wage) / original_wage * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_percentage_l3577_357747


namespace NUMINAMATH_CALUDE_root_lines_are_tangents_l3577_357724

/-- The Opq plane -/
structure Opq_plane where
  p : ℝ
  q : ℝ

/-- The line given by a^2 + ap + q = 0 for a real number a -/
def root_line (a : ℝ) : Set Opq_plane :=
  {point : Opq_plane | a^2 + a * point.p + point.q = 0}

/-- The discriminant parabola p^2 - 4q = 0 -/
def discriminant_parabola : Set Opq_plane :=
  {point : Opq_plane | point.p^2 - 4 * point.q = 0}

/-- A line is tangent to the parabola if it intersects the parabola at exactly one point -/
def is_tangent (line : Set Opq_plane) : Prop :=
  ∃! point : Opq_plane, point ∈ line ∩ discriminant_parabola

/-- The set of all tangents to the discriminant parabola -/
def all_tangents : Set (Set Opq_plane) :=
  {line : Set Opq_plane | is_tangent line}

/-- The theorem to be proved -/
theorem root_lines_are_tangents :
  {line | ∃ a : ℝ, line = root_line a} = all_tangents :=
sorry

end NUMINAMATH_CALUDE_root_lines_are_tangents_l3577_357724


namespace NUMINAMATH_CALUDE_regular_tetrahedron_volume_l3577_357777

/-- A regular tetrahedron with given base and lateral edge lengths -/
structure RegularTetrahedron where
  base_edge : ℝ
  lateral_edge : ℝ

/-- The volume of a regular tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of a regular tetrahedron with base edge length 6 and lateral edge length 9 is 9 -/
theorem regular_tetrahedron_volume :
  let t : RegularTetrahedron := ⟨6, 9⟩
  volume t = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_volume_l3577_357777


namespace NUMINAMATH_CALUDE_vector_problem_l3577_357790

/-- Given vectors in R^2 -/
def OA : Fin 2 → ℝ := ![3, -4]
def OB : Fin 2 → ℝ := ![6, -3]
def OC (m : ℝ) : Fin 2 → ℝ := ![5 - m, -3 - m]

/-- A, B, and C are collinear if and only if the cross product of AB and AC is zero -/
def collinear (m : ℝ) : Prop :=
  let AB := OB - OA
  let AC := OC m - OA
  AB 0 * AC 1 = AB 1 * AC 0

/-- ABC is a right-angled triangle if and only if one of its angles is 90 degrees -/
def right_angled (m : ℝ) : Prop :=
  let AB := OB - OA
  let BC := OC m - OB
  let AC := OC m - OA
  AB • BC = 0 ∨ BC • AC = 0 ∨ AC • AB = 0

/-- Main theorem -/
theorem vector_problem (m : ℝ) :
  (collinear m → m = 1/2) ∧
  (right_angled m → m = 7/4 ∨ m = -3/4 ∨ m = (1 + Real.sqrt 5)/2 ∨ m = (1 - Real.sqrt 5)/2) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l3577_357790


namespace NUMINAMATH_CALUDE_farm_tax_percentage_l3577_357778

theorem farm_tax_percentage (total_tax : ℝ) (willam_tax : ℝ) 
  (h1 : total_tax = 3840)
  (h2 : willam_tax = 480) :
  willam_tax / total_tax * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_farm_tax_percentage_l3577_357778


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3577_357732

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a + 1) = (a^2 - 2*a - 3) + Complex.I * (a + 1)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3577_357732


namespace NUMINAMATH_CALUDE_feet_quadrilateral_similar_l3577_357766

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The feet of perpendiculars from vertices to diagonals -/
def feet_of_perpendiculars (q : Quadrilateral) : Quadrilateral :=
  sorry

/-- Similarity relation between two quadrilaterals -/
def is_similar (q1 q2 : Quadrilateral) : Prop :=
  sorry

/-- Theorem: The quadrilateral formed by the feet of perpendiculars 
    is similar to the original quadrilateral -/
theorem feet_quadrilateral_similar (q : Quadrilateral) :
  is_similar q (feet_of_perpendiculars q) :=
sorry

end NUMINAMATH_CALUDE_feet_quadrilateral_similar_l3577_357766


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3577_357750

theorem power_fraction_simplification :
  (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by
sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3577_357750


namespace NUMINAMATH_CALUDE_last_segment_speed_l3577_357700

theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 96)
  (h2 : total_time = 90 / 60)
  (h3 : speed1 = 60)
  (h4 : speed2 = 65)
  (h5 : (speed1 + speed2 + (3 * total_distance / total_time - speed1 - speed2)) / 3 = total_distance / total_time) :
  3 * total_distance / total_time - speed1 - speed2 = 67 := by
  sorry

end NUMINAMATH_CALUDE_last_segment_speed_l3577_357700


namespace NUMINAMATH_CALUDE_regular_ngon_rotation_forms_regular_2ngon_l3577_357715

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (vertices : Fin n → V)
  (center : V)
  (is_regular : ∀ i j : Fin n, ‖vertices i - center‖ = ‖vertices j - center‖)

/-- Rotation of a vector about a point -/
def rotate (θ : ℝ) (center : V) (v : V) [NormedAddCommGroup V] [InnerProductSpace ℝ V] : V :=
  sorry

/-- The theorem statement -/
theorem regular_ngon_rotation_forms_regular_2ngon
  (n : ℕ) (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (ngon : RegularNGon n V) (θ : ℝ) :
  θ < 2 * Real.pi / n →
  (∃ (m : ℕ), θ = 2 * Real.pi / m) →
  ∃ (circle_center : V) (radius : ℝ),
    ∀ (i : Fin n),
      ‖circle_center - ngon.vertices i‖ = radius ∧
      ‖circle_center - rotate θ ngon.center (ngon.vertices i)‖ = radius :=
sorry

end NUMINAMATH_CALUDE_regular_ngon_rotation_forms_regular_2ngon_l3577_357715


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l3577_357717

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^4 - 5 * X^3 + 3 * X^2 + 11 * X - 6
  let divisor : Polynomial ℚ := 5 * X^2 + 7
  let quotient : Polynomial ℚ := 2 * X^2 - X - 11/5
  (dividend : Polynomial ℚ).div divisor = quotient := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l3577_357717


namespace NUMINAMATH_CALUDE_angle_215_in_third_quadrant_l3577_357753

def angle_in_third_quadrant (angle : ℝ) : Prop :=
  180 < angle ∧ angle ≤ 270

theorem angle_215_in_third_quadrant :
  angle_in_third_quadrant 215 :=
sorry

end NUMINAMATH_CALUDE_angle_215_in_third_quadrant_l3577_357753


namespace NUMINAMATH_CALUDE_rainy_days_count_l3577_357722

theorem rainy_days_count (n : ℕ) : 
  (∃ (R NR : ℕ), 
    R + NR = 7 ∧ 
    n * R + 4 * NR = 26 ∧ 
    4 * NR - n * R = 14) → 
  (∃ (R : ℕ), R = 2 ∧ 
    (∃ (NR : ℕ), R + NR = 7 ∧ 
      n * R + 4 * NR = 26 ∧ 
      4 * NR - n * R = 14)) :=
by sorry

end NUMINAMATH_CALUDE_rainy_days_count_l3577_357722


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l3577_357727

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l3577_357727


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3577_357718

theorem quadratic_roots_properties (a b : ℝ) : 
  a^2 + 5*a + 2 = 0 → 
  b^2 + 5*b + 2 = 0 → 
  a ≠ b →
  (1/a + 1/b = -5/2) ∧ ((a^2 + 7*a) * (b^2 + 7*b) = 32) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3577_357718


namespace NUMINAMATH_CALUDE_geometric_sum_n1_l3577_357716

theorem geometric_sum_n1 (a : ℝ) (h : a ≠ 1) :
  1 + a + a^2 + a^3 = (1 - a^4) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_n1_l3577_357716


namespace NUMINAMATH_CALUDE_dog_group_arrangements_count_l3577_357757

/-- The number of ways to divide 12 dogs into three groups -/
def dog_group_arrangements : ℕ :=
  let total_dogs : ℕ := 12
  let group_1_size : ℕ := 4
  let group_2_size : ℕ := 6
  let group_3_size : ℕ := 2
  let dogs_to_distribute : ℕ := total_dogs - 2  -- Fluffy and Nipper are pre-assigned
  let remaining_group_1_size : ℕ := group_1_size - 1  -- Fluffy is already in group 1
  let remaining_group_2_size : ℕ := group_2_size - 1  -- Nipper is already in group 2
  (Nat.choose dogs_to_distribute remaining_group_1_size) * 
  (Nat.choose (dogs_to_distribute - remaining_group_1_size) remaining_group_2_size)

/-- Theorem stating the number of ways to arrange the dogs -/
theorem dog_group_arrangements_count : dog_group_arrangements = 2520 := by
  sorry

end NUMINAMATH_CALUDE_dog_group_arrangements_count_l3577_357757


namespace NUMINAMATH_CALUDE_basketball_cost_l3577_357752

/-- The cost of each basketball given the total cost and soccer ball cost -/
theorem basketball_cost (total_cost : ℕ) (soccer_cost : ℕ) : 
  total_cost = 920 ∧ soccer_cost = 65 → (total_cost - 8 * soccer_cost) / 5 = 80 := by
  sorry

#check basketball_cost

end NUMINAMATH_CALUDE_basketball_cost_l3577_357752


namespace NUMINAMATH_CALUDE_number_problem_l3577_357761

theorem number_problem (x : ℝ) : (258/100 * x) / 6 = 543.95 → x = 1265 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3577_357761


namespace NUMINAMATH_CALUDE_third_year_sample_size_l3577_357769

/-- The number of third-year students to be sampled in a stratified sampling scenario -/
theorem third_year_sample_size 
  (total_students : ℕ) 
  (first_year_students : ℕ) 
  (sophomore_probability : ℚ) 
  (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : first_year_students = 760)
  (h3 : sophomore_probability = 37/100)
  (h4 : sample_size = 20) :
  let sophomore_students : ℕ := (sophomore_probability * total_students).num.toNat
  let third_year_students : ℕ := total_students - first_year_students - sophomore_students
  (sample_size * third_year_students) / total_students = 5 :=
by sorry

end NUMINAMATH_CALUDE_third_year_sample_size_l3577_357769


namespace NUMINAMATH_CALUDE_equation_solution_l3577_357719

theorem equation_solution (y : ℝ) : 
  (y / 6) / 3 = 6 / (y / 3) → y = 18 ∨ y = -18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3577_357719


namespace NUMINAMATH_CALUDE_carrot_weight_problem_l3577_357760

/-- Given 20 carrots weighing 3.64 kg, if 4 carrots are removed and the average weight
    of the remaining 16 carrots is 180 grams, then the average weight of the 4 removed
    carrots is 190 grams. -/
theorem carrot_weight_problem (total_weight : Real) (remaining_avg : Real) :
  total_weight = 3.64 →
  remaining_avg = 180 →
  let removed := 4
  let remaining := 20 - removed
  let removed_weight := total_weight * 1000 - remaining * remaining_avg
  removed_weight / removed = 190 := by
  sorry

end NUMINAMATH_CALUDE_carrot_weight_problem_l3577_357760


namespace NUMINAMATH_CALUDE_probability_second_white_ball_l3577_357745

theorem probability_second_white_ball (white_balls black_balls : ℕ) : 
  white_balls = 8 → black_balls = 7 → 
  (white_balls : ℚ) / (white_balls + black_balls - 1 : ℚ) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_white_ball_l3577_357745


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l3577_357758

theorem subset_implies_a_value (A B : Set ℤ) (a : ℤ) :
  A = {0, 1} →
  B = {-1, 0, a + 3} →
  A ⊆ B →
  a = -2 := by sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l3577_357758


namespace NUMINAMATH_CALUDE_suv_wash_price_l3577_357710

/-- The price of a car wash in dollars -/
def car_price : ℕ := 5

/-- The price of a truck wash in dollars -/
def truck_price : ℕ := 6

/-- The total amount raised in dollars -/
def total_raised : ℕ := 100

/-- The number of SUVs washed -/
def num_suvs : ℕ := 5

/-- The number of trucks washed -/
def num_trucks : ℕ := 5

/-- The number of cars washed -/
def num_cars : ℕ := 7

/-- The price of an SUV wash in dollars -/
def suv_price : ℕ := 9

theorem suv_wash_price :
  car_price * num_cars + truck_price * num_trucks + suv_price * num_suvs = total_raised :=
by sorry

end NUMINAMATH_CALUDE_suv_wash_price_l3577_357710


namespace NUMINAMATH_CALUDE_polar_bear_salmon_consumption_l3577_357709

def daily_fish_consumption : ℝ := 0.6
def daily_trout_consumption : ℝ := 0.2

theorem polar_bear_salmon_consumption :
  daily_fish_consumption - daily_trout_consumption = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_polar_bear_salmon_consumption_l3577_357709


namespace NUMINAMATH_CALUDE_square_less_than_triple_l3577_357744

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l3577_357744


namespace NUMINAMATH_CALUDE_final_game_score_l3577_357704

/-- Represents the points scored by each player in the basketball game -/
structure PlayerPoints where
  bailey : ℕ
  michiko : ℕ
  akiko : ℕ
  chandra : ℕ

/-- Calculates the total points scored by the team -/
def total_points (p : PlayerPoints) : ℕ :=
  p.bailey + p.michiko + p.akiko + p.chandra

/-- Proves that the team scored 54 points in the final game -/
theorem final_game_score :
  ∃ (p : PlayerPoints),
    p.bailey = 14 ∧
    p.michiko = p.bailey / 2 ∧
    p.akiko = p.michiko + 4 ∧
    p.chandra = 2 * p.akiko ∧
    total_points p = 54 := by
  sorry

end NUMINAMATH_CALUDE_final_game_score_l3577_357704


namespace NUMINAMATH_CALUDE_donut_selection_count_donut_selection_theorem_l3577_357775

theorem donut_selection_count : Nat → ℕ
  | n => 
    let total_donuts := 6
    let donut_types := 4
    let remaining_donuts := total_donuts - donut_types
    Nat.choose (remaining_donuts + donut_types - 1) (donut_types - 1)

theorem donut_selection_theorem : 
  donut_selection_count 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_donut_selection_count_donut_selection_theorem_l3577_357775


namespace NUMINAMATH_CALUDE_triangle_problem_l3577_357793

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove the angle B and area of the triangle under specific conditions. -/
theorem triangle_problem (a b c A B C : ℝ) : 
  (a + b) / Real.sin (A + B) = (a - c) / (Real.sin A - Real.sin B) →
  b = 3 →
  Real.sin A = Real.sqrt 3 / 3 →
  B = π / 3 ∧ 
  (1/2 * a * b * Real.sin C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3577_357793


namespace NUMINAMATH_CALUDE_binary_representation_of_51_l3577_357729

/-- Converts a natural number to its binary representation as a list of booleans -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 51

/-- The expected binary representation -/
def expectedBinary : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary representation of 51 is [true, true, false, false, true, true] -/
theorem binary_representation_of_51 :
  toBinary decimalNumber = expectedBinary := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_51_l3577_357729


namespace NUMINAMATH_CALUDE_inverse_proposition_absolute_values_l3577_357783

theorem inverse_proposition_absolute_values (a b : ℝ) :
  (∀ x y : ℝ, x = y → |x| = |y|) →
  (∀ x y : ℝ, |x| = |y| → x = y) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_absolute_values_l3577_357783


namespace NUMINAMATH_CALUDE_q_of_one_equals_five_l3577_357735

/-- Given a function q : ℝ → ℝ that passes through the point (1, 5), prove that q(1) = 5 -/
theorem q_of_one_equals_five (q : ℝ → ℝ) (h : q 1 = 5) : q 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_q_of_one_equals_five_l3577_357735


namespace NUMINAMATH_CALUDE_intersection_count_is_two_l3577_357705

/-- The number of intersection points between two circles -/
def intersection_count (c1 c2 : ℝ × ℝ → Prop) : ℕ :=
  sorry

/-- First circle: (x - 2.5)² + y² = 6.25 -/
def circle1 (p : ℝ × ℝ) : Prop :=
  (p.1 - 2.5)^2 + p.2^2 = 6.25

/-- Second circle: x² + (y - 5)² = 25 -/
def circle2 (p : ℝ × ℝ) : Prop :=
  p.1^2 + (p.2 - 5)^2 = 25

/-- Theorem stating that the number of intersection points between the two circles is 2 -/
theorem intersection_count_is_two :
  intersection_count circle1 circle2 = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_count_is_two_l3577_357705


namespace NUMINAMATH_CALUDE_boat_men_count_l3577_357720

/-- The number of men in the boat -/
def n : ℕ := 8

/-- The weight of the man being replaced -/
def old_weight : ℕ := 60

/-- The weight of the new man -/
def new_weight : ℕ := 68

/-- The increase in average weight after replacement -/
def avg_increase : ℕ := 1

theorem boat_men_count :
  ∀ W : ℕ,
  (W + (new_weight - old_weight)) / n = W / n + avg_increase →
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_boat_men_count_l3577_357720


namespace NUMINAMATH_CALUDE_f_value_at_2_l3577_357754

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
    (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l3577_357754


namespace NUMINAMATH_CALUDE_abc_sum_888_l3577_357751

theorem abc_sum_888 : 
  ∃! (a b c : Nat), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    (100 * a + 10 * b + c) + (100 * a + 10 * b + c) + (100 * a + 10 * b + c) = 888 ∧
    100 * a + 10 * b + c = 296 :=
by sorry

end NUMINAMATH_CALUDE_abc_sum_888_l3577_357751


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3577_357759

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem union_of_M_and_N :
  M ∪ N = {x | -1 ≤ x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3577_357759


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l3577_357791

/-- The quadratic function in general form -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The quadratic function in vertex form -/
def g (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating that f and g are equivalent -/
theorem quadratic_equivalence : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l3577_357791


namespace NUMINAMATH_CALUDE_multiple_of_ten_implies_multiple_of_five_l3577_357768

theorem multiple_of_ten_implies_multiple_of_five 
  (h1 : ∀ n : ℕ, 10 ∣ n → 5 ∣ n) 
  (a : ℕ) 
  (h2 : 10 ∣ a) : 
  5 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_ten_implies_multiple_of_five_l3577_357768


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_is_rectangle_l3577_357733

-- Define a circle
def Circle : Type := Unit

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define the property of being inscribed in a circle
def inscribed_in_circle (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem inscribed_quadrilateral_is_rectangle 
  (q : Quadrilateral) (c : Circle) : 
  inscribed_in_circle q c → is_rectangle q := by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_is_rectangle_l3577_357733


namespace NUMINAMATH_CALUDE_red_pens_count_l3577_357799

theorem red_pens_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 240)
  (h2 : red + blue = total)
  (h3 : blue = red - 2) : 
  red = 121 := by
  sorry

end NUMINAMATH_CALUDE_red_pens_count_l3577_357799


namespace NUMINAMATH_CALUDE_smallest_sum_4x4x4_cube_l3577_357728

/-- Represents a 4x4x4 cube made of dice -/
structure LargeCube where
  size : Nat
  dice_count : Nat
  opposite_sides_sum : Nat

/-- Calculates the smallest possible sum of visible faces on the large cube -/
def smallest_visible_sum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the smallest possible sum for a 4x4x4 cube of dice -/
theorem smallest_sum_4x4x4_cube (cube : LargeCube) 
  (h1 : cube.size = 4)
  (h2 : cube.dice_count = 64)
  (h3 : cube.opposite_sides_sum = 7) :
  smallest_visible_sum cube = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_4x4x4_cube_l3577_357728


namespace NUMINAMATH_CALUDE_cos_195_plus_i_sin_195_to_60_l3577_357726

-- Define DeMoivre's Theorem
axiom deMoivre (θ : ℝ) (n : ℕ) : 
  (Complex.exp (Complex.I * θ)) ^ n = Complex.exp (Complex.I * (n * θ))

-- Define the problem
theorem cos_195_plus_i_sin_195_to_60 :
  (Complex.exp (Complex.I * (195 * π / 180))) ^ 60 = -1 := by sorry

end NUMINAMATH_CALUDE_cos_195_plus_i_sin_195_to_60_l3577_357726


namespace NUMINAMATH_CALUDE_linear_function_third_quadrant_l3577_357730

/-- A linear function y = (m-2)x + m-1 does not pass through the third quadrant
    if and only if 1 ≤ m < 2. -/
theorem linear_function_third_quadrant (m : ℝ) :
  (∀ x y : ℝ, y = (m - 2) * x + m - 1 → (x < 0 ∧ y < 0 → False)) ↔ 1 ≤ m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_third_quadrant_l3577_357730


namespace NUMINAMATH_CALUDE_young_pioneers_tree_planting_l3577_357736

theorem young_pioneers_tree_planting (x : ℕ) : 
  (5 * x + 3 = 6 * x - 7) → 
  (x = 10 ∧ 5 * x + 3 = 53) := by
  sorry

end NUMINAMATH_CALUDE_young_pioneers_tree_planting_l3577_357736


namespace NUMINAMATH_CALUDE_rick_cheese_servings_l3577_357743

/-- Calculates the number of cheese servings eaten given the remaining calories -/
def servingsEaten (caloriesPerServing : ℕ) (servingsPerBlock : ℕ) (remainingCalories : ℕ) : ℕ :=
  (caloriesPerServing * servingsPerBlock - remainingCalories) / caloriesPerServing

theorem rick_cheese_servings :
  servingsEaten 110 16 1210 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rick_cheese_servings_l3577_357743


namespace NUMINAMATH_CALUDE_perpendicular_planes_not_always_true_l3577_357746

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection relation between planes
variable (intersects : Plane → Plane → Line → Prop)

-- Define the subset relation between lines and planes
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_not_always_true 
  (α β : Plane) (l a b : Line) 
  (h1 : intersects α β l)
  (h2 : subset a α)
  (h3 : subset b β) :
  ¬ (∀ (h4 : perp_lines a l) (h5 : perp_lines b l), perp_planes α β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_not_always_true_l3577_357746


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l3577_357737

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 48, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 8, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l3577_357737


namespace NUMINAMATH_CALUDE_hamburger_sales_solution_l3577_357763

/-- Represents the hamburger sales problem. -/
def HamburgerSales (total_goal : ℕ) (price : ℕ) (first_group : ℕ) (remaining : ℕ) : Prop :=
  let total_hamburgers := total_goal / price
  let accounted_for := first_group + remaining
  total_hamburgers - accounted_for = 2

/-- Theorem stating the solution to the hamburger sales problem. -/
theorem hamburger_sales_solution :
  HamburgerSales 50 5 4 4 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_sales_solution_l3577_357763


namespace NUMINAMATH_CALUDE_equal_candies_after_sharing_l3577_357798

/-- Proves that Minyoung and Taehyung will have the same number of candies
    if Minyoung gives 3 candies to Taehyung. -/
theorem equal_candies_after_sharing (minyoung_initial : ℕ) (taehyung_initial : ℕ) 
  (candies_shared : ℕ) : 
  minyoung_initial = 9 →
  taehyung_initial = 3 →
  candies_shared = 3 →
  minyoung_initial - candies_shared = taehyung_initial + candies_shared :=
by
  sorry

#check equal_candies_after_sharing

end NUMINAMATH_CALUDE_equal_candies_after_sharing_l3577_357798


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3577_357771

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3577_357771
