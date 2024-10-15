import Mathlib

namespace NUMINAMATH_CALUDE_aftershave_dilution_l524_52454

theorem aftershave_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 12 →
  initial_concentration = 0.6 →
  target_concentration = 0.4 →
  water_added = 6 →
  initial_volume * initial_concentration = 
    target_concentration * (initial_volume + water_added) :=
by
  sorry

#check aftershave_dilution

end NUMINAMATH_CALUDE_aftershave_dilution_l524_52454


namespace NUMINAMATH_CALUDE_log_seven_eighteen_l524_52446

theorem log_seven_eighteen (a b : ℝ) 
  (h1 : Real.log 2 / Real.log 10 = a) 
  (h2 : Real.log 3 / Real.log 10 = b) : 
  Real.log 18 / Real.log 7 = (2*a + 4*b) / (1 + 2*a) := by
  sorry

end NUMINAMATH_CALUDE_log_seven_eighteen_l524_52446


namespace NUMINAMATH_CALUDE_total_reduction_proof_l524_52471

-- Define the original price and reduction percentages
def original_price : ℝ := 500
def first_reduction : ℝ := 0.07
def second_reduction : ℝ := 0.05
def third_reduction : ℝ := 0.03

-- Define the function to calculate the price after reductions
def price_after_reductions (p : ℝ) (r1 r2 r3 : ℝ) : ℝ :=
  p * (1 - r1) * (1 - r2) * (1 - r3)

-- Theorem statement
theorem total_reduction_proof :
  original_price - price_after_reductions original_price first_reduction second_reduction third_reduction = 71.5025 := by
  sorry


end NUMINAMATH_CALUDE_total_reduction_proof_l524_52471


namespace NUMINAMATH_CALUDE_susan_chairs_l524_52427

/-- The number of chairs in Susan's house -/
def total_chairs : ℕ :=
  let red_chairs : ℕ := 5
  let yellow_chairs : ℕ := 4 * red_chairs
  let blue_chairs : ℕ := yellow_chairs - 2
  let green_chairs : ℕ := (red_chairs + blue_chairs) / 2
  red_chairs + yellow_chairs + blue_chairs + green_chairs

/-- Theorem stating the total number of chairs in Susan's house -/
theorem susan_chairs : total_chairs = 54 := by
  sorry

end NUMINAMATH_CALUDE_susan_chairs_l524_52427


namespace NUMINAMATH_CALUDE_quadratic_expression_values_l524_52439

theorem quadratic_expression_values (m n : ℤ) 
  (hm : |m| = 3)
  (hn : |n| = 2)
  (hmn : m < n) :
  m^2 + m*n + n^2 = 7 ∨ m^2 + m*n + n^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_values_l524_52439


namespace NUMINAMATH_CALUDE_f_properties_l524_52484

-- Define the function f(x) = x^2 + ln|x|
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log (abs x)

-- State the theorem
theorem f_properties :
  -- f is defined for all non-zero real numbers
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) ∧
  -- f is an even function
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  -- f is increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l524_52484


namespace NUMINAMATH_CALUDE_parabola_equation_l524_52401

/-- Given a parabola y^2 = 2px where p > 0, if a point P(2, y_0) on the parabola
    has a distance of 4 from the directrix, then the equation of the parabola is y^2 = 8x -/
theorem parabola_equation (p : ℝ) (y_0 : ℝ) (h1 : p > 0) (h2 : y_0^2 = 2*p*2) 
  (h3 : p/2 + 2 = 4) : 
  ∀ x y : ℝ, y^2 = 2*p*x ↔ y^2 = 8*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l524_52401


namespace NUMINAMATH_CALUDE_sams_money_l524_52443

/-- Given that Sam and Erica have $91 together and Erica has $53, 
    prove that Sam has $38. -/
theorem sams_money (total : ℕ) (ericas_money : ℕ) (sams_money : ℕ) : 
  total = 91 → ericas_money = 53 → sams_money = total - ericas_money → sams_money = 38 := by
  sorry

end NUMINAMATH_CALUDE_sams_money_l524_52443


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l524_52482

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  -- We don't need to define any specific properties here, as we're only interested in the general structure

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

theorem rectangular_prism_sum (rp : RectangularPrism) : 
  num_faces rp + num_edges rp + num_vertices rp = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l524_52482


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l524_52459

theorem imaginary_part_of_reciprocal (z : ℂ) (h : z = 1 - 2*I) : 
  Complex.im (z⁻¹) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l524_52459


namespace NUMINAMATH_CALUDE_certain_number_proof_l524_52486

theorem certain_number_proof (y : ℝ) : 
  (0.20 * 1050 = 0.15 * y - 15) → y = 1500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l524_52486


namespace NUMINAMATH_CALUDE_proportion_of_dogs_l524_52402

theorem proportion_of_dogs (C G : ℝ) 
  (h1 : 0.8 * G + 0.25 * C = 0.3 * (G + C)) 
  (h2 : C > 0) 
  (h3 : G > 0) : 
  C / (C + G) = 10 / 11 := by
  sorry

end NUMINAMATH_CALUDE_proportion_of_dogs_l524_52402


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l524_52474

theorem geometric_sequence_first_term 
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r = 18) -- second term is 18
  (h2 : a * r^2 = 24) -- third term is 24
  : a = 27/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l524_52474


namespace NUMINAMATH_CALUDE_tip_is_24_dollars_l524_52417

/-- The cost of a woman's haircut in dollars -/
def womens_haircut_cost : ℚ := 48

/-- The cost of a child's haircut in dollars -/
def childrens_haircut_cost : ℚ := 36

/-- The number of women getting haircuts -/
def num_women : ℕ := 1

/-- The number of children getting haircuts -/
def num_children : ℕ := 2

/-- The tip percentage as a decimal -/
def tip_percentage : ℚ := 0.20

/-- The total cost of haircuts before tip -/
def total_cost : ℚ := womens_haircut_cost * num_women + childrens_haircut_cost * num_children

/-- The tip amount in dollars -/
def tip_amount : ℚ := total_cost * tip_percentage

theorem tip_is_24_dollars : tip_amount = 24 := by
  sorry

end NUMINAMATH_CALUDE_tip_is_24_dollars_l524_52417


namespace NUMINAMATH_CALUDE_course_length_is_300_l524_52467

/-- Represents the dogsled race scenario -/
structure DogsledRace where
  teamT_speed : ℝ
  teamA_speed_diff : ℝ
  teamT_time : ℝ
  teamA_time_diff : ℝ

/-- Calculates the length of the dogsled race course -/
def course_length (race : DogsledRace) : ℝ :=
  race.teamT_speed * race.teamT_time

/-- Theorem stating that the course length is 300 miles given the race conditions -/
theorem course_length_is_300 (race : DogsledRace)
  (h1 : race.teamT_speed = 20)
  (h2 : race.teamA_speed_diff = 5)
  (h3 : race.teamA_time_diff = 3)
  (h4 : race.teamT_time * race.teamT_speed = (race.teamT_time - race.teamA_time_diff) * (race.teamT_speed + race.teamA_speed_diff)) :
  course_length race = 300 := by
  sorry

#eval course_length { teamT_speed := 20, teamA_speed_diff := 5, teamT_time := 15, teamA_time_diff := 3 }

end NUMINAMATH_CALUDE_course_length_is_300_l524_52467


namespace NUMINAMATH_CALUDE_xy_neq_one_condition_l524_52407

theorem xy_neq_one_condition (x y : ℝ) :
  (∃ x y : ℝ, (x ≠ 1 ∨ y ≠ 1) ∧ x * y = 1) ∧
  (x * y ≠ 1 → (x ≠ 1 ∨ y ≠ 1)) :=
by sorry

end NUMINAMATH_CALUDE_xy_neq_one_condition_l524_52407


namespace NUMINAMATH_CALUDE_distance_between_points_l524_52494

/-- The distance between two points given rowing speed, stream speed, and round trip time -/
theorem distance_between_points (rowing_speed stream_speed : ℝ) (round_trip_time : ℝ) :
  rowing_speed = 10 →
  stream_speed = 2 →
  round_trip_time = 5 →
  ∃ (distance : ℝ),
    distance / (rowing_speed + stream_speed) + distance / (rowing_speed - stream_speed) = round_trip_time ∧
    distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l524_52494


namespace NUMINAMATH_CALUDE_fraction_calculation_l524_52469

theorem fraction_calculation :
  (1 / 5 + 1 / 7) / (3 / 8 + 2 / 9) = 864 / 1505 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l524_52469


namespace NUMINAMATH_CALUDE_tree_spacing_l524_52472

theorem tree_spacing (yard_length : ℕ) (num_trees : ℕ) (spacing : ℕ) :
  yard_length = 434 →
  num_trees = 32 →
  spacing * (num_trees - 1) = yard_length →
  spacing = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l524_52472


namespace NUMINAMATH_CALUDE_ellipse_geometric_sequence_l524_52404

/-- Given an ellipse E with equation x²/a² + y²/b² = 1 (a > b > 0),
    eccentricity e = √2/2, and left vertex at (-2,0),
    prove that for points B and C on E, where AB is parallel to OC
    and AB intersects the y-axis at D, |AB|, √2|OC|, and |AD|
    form a geometric sequence. -/
theorem ellipse_geometric_sequence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (E : Set (ℝ × ℝ))
  (hE : E = {(x, y) | x^2/a^2 + y^2/b^2 = 1})
  (he : (a^2 - b^2)/a^2 = 1/2)
  (hA : (-2, 0) ∈ E)
  (B C : ℝ × ℝ) (hB : B ∈ E) (hC : C ∈ E)
  (hparallel : ∃ (k : ℝ), (B.2 + 2*k = B.1 ∧ C.2 = k*C.1))
  (D : ℝ × ℝ) (hD : D.1 = 0 ∧ D.2 = B.2 - B.1/2*B.2) :
  ∃ (r : ℝ), abs (B.1 - (-2)) * abs (D.2) = r * (abs (C.1) * abs (C.1) + abs (C.2) * abs (C.2))
    ∧ abs (D.2)^2 = r * (abs (B.1 - (-2)) * abs (D.2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_geometric_sequence_l524_52404


namespace NUMINAMATH_CALUDE_cone_radii_sum_l524_52449

/-- Given a circle with radius 5 divided into three sectors with area ratios 1:2:3,
    used as lateral surfaces of three cones with base radii r₁, r₂, and r₃ respectively,
    prove that r₁ + r₂ + r₃ = 5. -/
theorem cone_radii_sum (r₁ r₂ r₃ : ℝ) : 
  (2 * π * r₁ = (1 / 6) * 2 * π * 5) → 
  (2 * π * r₂ = (2 / 6) * 2 * π * 5) → 
  (2 * π * r₃ = (3 / 6) * 2 * π * 5) → 
  r₁ + r₂ + r₃ = 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_radii_sum_l524_52449


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l524_52422

def f (x : ℝ) := x^3 - 3*x^2

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (∀ y : ℝ, x < y → f x > f y) ↔ x ∈ Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l524_52422


namespace NUMINAMATH_CALUDE_line_through_circle_center_l524_52440

theorem line_through_circle_center (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0 ∧ 
   ∀ (x' y' : ℝ), x'^2 + y'^2 + 2*x' - 4*y' = 0 → 
   (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l524_52440


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l524_52437

theorem trigonometric_equation_solution (x : ℝ) :
  8.483 * Real.tan x - Real.sin (2 * x) - Real.cos (2 * x) + 2 * (2 * Real.cos x - 1 / Real.cos x) = 0 ↔
  ∃ k : ℤ, x = π / 4 * (2 * k + 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l524_52437


namespace NUMINAMATH_CALUDE_chessboard_one_color_l524_52461

/-- Represents the color of a square on the chessboard -/
inductive Color
| Black
| White

/-- Represents the chessboard as a function from coordinates to colors -/
def Chessboard := Fin 8 → Fin 8 → Color

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  x1 : Fin 8
  y1 : Fin 8
  x2 : Fin 8
  y2 : Fin 8

/-- Checks if a rectangle is adjacent to a corner of the board -/
def isCornerRectangle (r : Rectangle) : Prop :=
  (r.x1 = 0 ∧ r.y1 = 0) ∨
  (r.x1 = 0 ∧ r.y2 = 7) ∨
  (r.x2 = 7 ∧ r.y1 = 0) ∨
  (r.x2 = 7 ∧ r.y2 = 7)

/-- The operation of changing colors in a rectangle -/
def applyRectangle (board : Chessboard) (r : Rectangle) : Chessboard :=
  sorry

/-- Theorem stating that any chessboard can be made one color -/
theorem chessboard_one_color :
  ∀ (initial : Chessboard),
  ∃ (final : Chessboard) (steps : List Rectangle),
    (∀ r ∈ steps, isCornerRectangle r) ∧
    (final = steps.foldl applyRectangle initial) ∧
    (∃ c : Color, ∀ x y : Fin 8, final x y = c) :=
  sorry

end NUMINAMATH_CALUDE_chessboard_one_color_l524_52461


namespace NUMINAMATH_CALUDE_candy_bar_cost_l524_52426

/-- The cost of candy bars purchased by Dan -/
def total_cost : ℚ := 6

/-- The number of candy bars Dan bought -/
def number_of_bars : ℕ := 2

/-- The cost of each candy bar -/
def cost_per_bar : ℚ := total_cost / number_of_bars

/-- Theorem stating that the cost of each candy bar is $3 -/
theorem candy_bar_cost : cost_per_bar = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l524_52426


namespace NUMINAMATH_CALUDE_area_AXYD_area_AXYD_is_72_l524_52499

/-- Rectangle ABCD with given dimensions and point E -/
structure Rectangle :=
  (A B C D E : ℝ × ℝ)
  (AB : ℝ)
  (BC : ℝ)

/-- Point Z on the extension of BC -/
def Z (rect : Rectangle) : ℝ × ℝ := (rect.C.1, rect.C.2 + 18)

/-- Conditions for the rectangle and point E -/
def validRectangle (rect : Rectangle) : Prop :=
  rect.AB = 20 ∧
  rect.BC = 12 ∧
  rect.A = (0, 0) ∧
  rect.B = (20, 0) ∧
  rect.C = (20, 12) ∧
  rect.D = (0, 12) ∧
  rect.E = (6, 6)

/-- Theorem: Area of quadrilateral AXYD is 72 -/
theorem area_AXYD (rect : Rectangle) (h : validRectangle rect) : ℝ :=
  72

/-- Main theorem: If the rectangle satisfies the conditions, then the area of AXYD is 72 -/
theorem area_AXYD_is_72 (rect : Rectangle) (h : validRectangle rect) : 
  area_AXYD rect h = 72 := by
  sorry

end NUMINAMATH_CALUDE_area_AXYD_area_AXYD_is_72_l524_52499


namespace NUMINAMATH_CALUDE_max_sphere_radius_in_intersecting_cones_l524_52405

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum radius of a sphere that can fit within two intersecting cones -/
def maxSphereRadius (ic : IntersectingCones) : ℝ := sorry

/-- Theorem stating the maximum sphere radius for the given configuration -/
theorem max_sphere_radius_in_intersecting_cones :
  let ic : IntersectingCones := {
    cone1 := { baseRadius := 5, height := 12 },
    cone2 := { baseRadius := 5, height := 12 },
    intersectionDistance := 4
  }
  maxSphereRadius ic = 40 / 13 := by sorry

end NUMINAMATH_CALUDE_max_sphere_radius_in_intersecting_cones_l524_52405


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l524_52495

/-- Given vectors a and b, where a is parallel to b, prove the minimum value of 3^x + 9^y + 2 -/
theorem min_value_parallel_vectors (x y : ℝ) :
  let a : Fin 2 → ℝ := ![3 - x, y]
  let b : Fin 2 → ℝ := ![2, 1]
  (∃ (k : ℝ), a = k • b) →
  (∀ (x' y' : ℝ), 3^x' + 9^y' + 2 ≥ 6 * Real.sqrt 3 + 2) ∧
  (∃ (x₀ y₀ : ℝ), 3^x₀ + 9^y₀ + 2 = 6 * Real.sqrt 3 + 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l524_52495


namespace NUMINAMATH_CALUDE_nickel_difference_formula_l524_52444

/-- The number of nickels equivalent to one quarter -/
def nickels_per_quarter : ℕ := 5

/-- Alice's quarters as a function of q -/
def alice_quarters (q : ℕ) : ℕ := 10 * q + 2

/-- Bob's quarters as a function of q -/
def bob_quarters (q : ℕ) : ℕ := 2 * q + 10

/-- The difference in nickels between Alice and Bob -/
def nickel_difference (q : ℕ) : ℤ :=
  (alice_quarters q - bob_quarters q) * nickels_per_quarter

theorem nickel_difference_formula (q : ℕ) :
  nickel_difference q = 40 * (q - 1) := by sorry

end NUMINAMATH_CALUDE_nickel_difference_formula_l524_52444


namespace NUMINAMATH_CALUDE_negative_sqrt_four_equals_negative_two_l524_52429

theorem negative_sqrt_four_equals_negative_two : -Real.sqrt 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_four_equals_negative_two_l524_52429


namespace NUMINAMATH_CALUDE_factorization_3x2_minus_12_factorization_ax2_4axy_4ay2_l524_52416

-- Statement 1
theorem factorization_3x2_minus_12 (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

-- Statement 2
theorem factorization_ax2_4axy_4ay2 (a x y : ℝ) : a * x^2 - 4 * a * x * y + 4 * a * y^2 = a * (x - 2 * y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x2_minus_12_factorization_ax2_4axy_4ay2_l524_52416


namespace NUMINAMATH_CALUDE_perpendicular_tangents_imply_m_value_l524_52477

/-- The original function F1 -/
def F1 (x : ℝ) : ℝ := x^2

/-- The translated function F2 -/
def F2 (m : ℝ) (x : ℝ) : ℝ := (x - m)^2 - 1

/-- The derivative of F1 -/
def F1_derivative (x : ℝ) : ℝ := 2 * x

/-- The derivative of F2 -/
def F2_derivative (m : ℝ) (x : ℝ) : ℝ := 2 * (x - m)

theorem perpendicular_tangents_imply_m_value :
  ∀ m : ℝ, (F1_derivative 1 * F2_derivative m 1 = -1) → m = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_imply_m_value_l524_52477


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_5_range_of_a_for_f_greater_than_abs_1_minus_a_l524_52406

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 3|

-- Theorem for part I
theorem solution_set_f_less_than_5 :
  {x : ℝ | f x < 5} = Set.Ioo (-7/4) (3/4) :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_greater_than_abs_1_minus_a :
  {a : ℝ | ∀ x, f x > |1 - a|} = Set.Ioo (-3) 5 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_5_range_of_a_for_f_greater_than_abs_1_minus_a_l524_52406


namespace NUMINAMATH_CALUDE_man_and_son_work_time_l524_52441

/-- The time taken for a man and his son to complete a task together, given their individual completion times -/
theorem man_and_son_work_time (man_time son_time : ℝ) (h1 : man_time = 5) (h2 : son_time = 20) :
  1 / (1 / man_time + 1 / son_time) = 4 := by
  sorry

end NUMINAMATH_CALUDE_man_and_son_work_time_l524_52441


namespace NUMINAMATH_CALUDE_sum_of_max_min_is_10_l524_52435

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem sum_of_max_min_is_10 :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a + f b = 10 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_is_10_l524_52435


namespace NUMINAMATH_CALUDE_binomial_permutation_equality_l524_52436

theorem binomial_permutation_equality (n : ℕ+) :
  3 * (Nat.choose (n.val - 1) (n.val - 5)) = 5 * (Nat.factorial (n.val - 2) / Nat.factorial (n.val - 4)) →
  n.val = 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_permutation_equality_l524_52436


namespace NUMINAMATH_CALUDE_ten_factorial_minus_nine_factorial_l524_52434

-- Define factorial function
def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

-- State the theorem
theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_ten_factorial_minus_nine_factorial_l524_52434


namespace NUMINAMATH_CALUDE_sum_of_coefficients_eq_64_l524_52475

/-- The sum of the numerical coefficients in the complete expansion of (x^2 - 3xy + y^2)^6 -/
def sum_of_coefficients : ℕ :=
  (1 - 3)^6

theorem sum_of_coefficients_eq_64 : sum_of_coefficients = 64 := by
  sorry

#eval sum_of_coefficients

end NUMINAMATH_CALUDE_sum_of_coefficients_eq_64_l524_52475


namespace NUMINAMATH_CALUDE_annies_final_crayons_l524_52476

/-- The number of crayons Annie has at the end, given the initial conditions. -/
def anniesCrayons : ℕ :=
  let initialCrayons : ℕ := 4
  let samsCrayons : ℕ := 36
  let matthewsCrayons : ℕ := 5 * samsCrayons
  initialCrayons + samsCrayons + matthewsCrayons

/-- Theorem stating that Annie will have 220 crayons at the end. -/
theorem annies_final_crayons : anniesCrayons = 220 := by
  sorry

end NUMINAMATH_CALUDE_annies_final_crayons_l524_52476


namespace NUMINAMATH_CALUDE_probability_theorem_l524_52460

/-- Represents a brother with a name of a certain length -/
structure Brother where
  name : String
  name_length : Nat

/-- Represents the problem setup -/
structure LetterCardProblem where
  adam : Brother
  brian : Brother
  total_letters : Nat
  (total_is_sum : total_letters = adam.name_length + brian.name_length)
  (total_is_twelve : total_letters = 12)

/-- The probability of selecting one letter from each brother's name -/
def probability_one_from_each (problem : LetterCardProblem) : Rat :=
  4 / 11

theorem probability_theorem (problem : LetterCardProblem) :
  probability_one_from_each problem = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l524_52460


namespace NUMINAMATH_CALUDE_percentage_difference_l524_52432

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.25)) :
  y = x * (1 + 0.25) := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l524_52432


namespace NUMINAMATH_CALUDE_range_of_a_l524_52498

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x + 1| > 2) → a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l524_52498


namespace NUMINAMATH_CALUDE_percent_equality_l524_52438

theorem percent_equality (y : ℝ) : (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l524_52438


namespace NUMINAMATH_CALUDE_initial_cabinets_l524_52492

theorem initial_cabinets (total : ℕ) (additional : ℕ) (counters : ℕ) : 
  total = 26 → 
  additional = 5 → 
  counters = 3 → 
  ∃ initial : ℕ, initial + counters * (2 * initial) + additional = total ∧ initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_cabinets_l524_52492


namespace NUMINAMATH_CALUDE_matchstick_sequence_l524_52464

/-- 
Given a sequence where:
- The first term is 4
- Each subsequent term increases by 3
This theorem proves that the 20th term of the sequence is 61.
-/
theorem matchstick_sequence (n : ℕ) : 
  let sequence : ℕ → ℕ := λ k => 4 + 3 * (k - 1)
  sequence 20 = 61 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_sequence_l524_52464


namespace NUMINAMATH_CALUDE_simplify_fraction_integer_decimal_parts_l524_52430

-- Part 1
theorem simplify_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  2 / (Real.sqrt x + Real.sqrt y) = Real.sqrt x - Real.sqrt y :=
sorry

-- Part 2
theorem integer_decimal_parts (a : ℤ) (b : ℝ) 
  (h : 1 / (2 - Real.sqrt 3) = ↑a + b) (h_b : 0 ≤ b ∧ b < 1) :
  (a : ℝ)^2 + b^2 = 13 - 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_simplify_fraction_integer_decimal_parts_l524_52430


namespace NUMINAMATH_CALUDE_missy_claims_count_l524_52431

/-- The number of insurance claims that can be handled by three agents --/
def insurance_claims (jan_claims : ℕ) : ℕ × ℕ × ℕ :=
  let john_claims := jan_claims + (jan_claims * 30 / 100)
  let missy_claims := john_claims + 15
  (jan_claims, john_claims, missy_claims)

/-- Theorem stating that Missy can handle 41 claims given the conditions --/
theorem missy_claims_count :
  let (jan, john, missy) := insurance_claims 20
  missy = 41 := by sorry

end NUMINAMATH_CALUDE_missy_claims_count_l524_52431


namespace NUMINAMATH_CALUDE_markup_calculation_l524_52489

theorem markup_calculation (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) : 
  purchase_price = 48 →
  overhead_percentage = 0.25 →
  net_profit = 12 →
  purchase_price + purchase_price * overhead_percentage + net_profit - purchase_price = 24 := by
sorry

end NUMINAMATH_CALUDE_markup_calculation_l524_52489


namespace NUMINAMATH_CALUDE_jack_age_l524_52458

/-- Given that Jack's age is 20 years less than twice Jane's age,
    and the sum of their ages is 60, prove that Jack is 33 years old. -/
theorem jack_age (j a : ℕ) 
  (h1 : j = 2 * a - 20)  -- Jack's age is 20 years less than twice Jane's age
  (h2 : j + a = 60)      -- The sum of their ages is 60
  : j = 33 := by
  sorry

end NUMINAMATH_CALUDE_jack_age_l524_52458


namespace NUMINAMATH_CALUDE_fraction_comparison_l524_52421

theorem fraction_comparison : 
  (14 / 10 : ℚ) = 7 / 5 ∧ 
  (1 + 2 / 5 : ℚ) = 7 / 5 ∧ 
  (1 + 4 / 20 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 2 / 6 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 28 / 20 : ℚ) ≠ 7 / 5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l524_52421


namespace NUMINAMATH_CALUDE_octal_to_decimal_l524_52420

theorem octal_to_decimal (n : ℕ) (h : n = 246) : 
  2 * 8^2 + 4 * 8^1 + 6 * 8^0 = 166 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_l524_52420


namespace NUMINAMATH_CALUDE_original_denominator_proof_l524_52408

theorem original_denominator_proof (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = (1 : ℚ) / 3 →
  d = 23 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l524_52408


namespace NUMINAMATH_CALUDE_b_plus_c_equals_six_l524_52415

theorem b_plus_c_equals_six (a b c d : ℝ) 
  (h1 : a + b = 5) 
  (h2 : c + d = 3) 
  (h3 : a + d = 2) : 
  b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_b_plus_c_equals_six_l524_52415


namespace NUMINAMATH_CALUDE_inverse_inequality_l524_52462

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l524_52462


namespace NUMINAMATH_CALUDE_darks_drying_time_l524_52423

/-- Represents the time in minutes for washing and drying a load of laundry -/
structure LaundryTime where
  wash : Nat
  dry : Nat

/-- Calculates the total time for a load of laundry -/
def totalTime (lt : LaundryTime) : Nat :=
  lt.wash + lt.dry

theorem darks_drying_time (whites : LaundryTime) (darks_wash : Nat) (colors : LaundryTime) 
    (total_time : Nat) (h1 : whites.wash = 72) (h2 : whites.dry = 50)
    (h3 : darks_wash = 58) (h4 : colors.wash = 45) (h5 : colors.dry = 54)
    (h6 : total_time = 344) :
    ∃ (darks_dry : Nat), darks_dry = 65 ∧ 
    total_time = totalTime whites + totalTime colors + darks_wash + darks_dry := by
  sorry

#check darks_drying_time

end NUMINAMATH_CALUDE_darks_drying_time_l524_52423


namespace NUMINAMATH_CALUDE_reciprocal_lcm_24_221_l524_52413

theorem reciprocal_lcm_24_221 :
  let a : ℕ := 24
  let b : ℕ := 221
  Nat.gcd a b = 1 →
  (1 : ℚ) / (Nat.lcm a b) = 1 / 5304 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_lcm_24_221_l524_52413


namespace NUMINAMATH_CALUDE_pencil_cost_l524_52442

/-- Calculates the cost of a pencil given shopping information -/
theorem pencil_cost (initial_amount : ℚ) (hat_cost : ℚ) (num_cookies : ℕ) (cookie_cost : ℚ) (remaining_amount : ℚ) : 
  initial_amount = 20 →
  hat_cost = 10 →
  num_cookies = 4 →
  cookie_cost = 5/4 →
  remaining_amount = 3 →
  initial_amount - (hat_cost + num_cookies * cookie_cost + remaining_amount) = 2 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_l524_52442


namespace NUMINAMATH_CALUDE_max_expression_l524_52451

theorem max_expression (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hy : 0 < y₁ ∧ y₁ < y₂) 
  (hsum_x : x₁ + x₂ = 1) 
  (hsum_y : y₁ + y₂ = 1) : 
  x₁ * y₁ + x₂ * y₂ ≥ max (x₁ * x₂ + y₁ * y₂) (max (x₁ * y₂ + x₂ * y₁) (1/2)) :=
sorry

end NUMINAMATH_CALUDE_max_expression_l524_52451


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l524_52418

-- Define the function f
def f (x : ℝ) : ℝ := (1 - 2*x)^10

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 20 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l524_52418


namespace NUMINAMATH_CALUDE_birds_reduced_correct_l524_52478

/-- The number of birds reduced on the third day, given the initial number of birds,
    the doubling on the second day, and the total number of birds seen in three days. -/
def birds_reduced (initial : ℕ) (total : ℕ) : ℕ :=
  initial * 2 - (total - (initial + initial * 2))

/-- Theorem stating that the number of birds reduced on the third day is 200,
    given the conditions from the problem. -/
theorem birds_reduced_correct : birds_reduced 300 1300 = 200 := by
  sorry

end NUMINAMATH_CALUDE_birds_reduced_correct_l524_52478


namespace NUMINAMATH_CALUDE_plot_length_is_52_l524_52463

/-- Represents a rectangular plot with specific fencing conditions -/
structure Plot where
  breadth : ℝ
  length : ℝ
  flatCost : ℝ
  risePercent : ℝ
  totalRise : ℝ
  totalCost : ℝ

/-- Calculates the length of the plot given the conditions -/
def calculateLength (p : Plot) : ℝ :=
  p.breadth + 20

/-- Theorem stating the length of the plot under given conditions -/
theorem plot_length_is_52 (p : Plot) 
  (h1 : p.length = p.breadth + 20)
  (h2 : p.flatCost = 26.5)
  (h3 : p.risePercent = 0.1)
  (h4 : p.totalRise = 5)
  (h5 : p.totalCost = 5300)
  (h6 : p.totalCost = 2 * (p.breadth + 20) * p.flatCost + 
        2 * p.breadth * (p.flatCost * (1 + p.risePercent * p.totalRise))) :
  calculateLength p = 52 := by
  sorry

#eval calculateLength { breadth := 32, length := 52, flatCost := 26.5, 
                        risePercent := 0.1, totalRise := 5, totalCost := 5300 }

end NUMINAMATH_CALUDE_plot_length_is_52_l524_52463


namespace NUMINAMATH_CALUDE_smallest_cut_length_l524_52412

theorem smallest_cut_length (x : ℕ) : x > 0 ∧ x ≤ 13 →
  (∀ y : ℕ, y > 0 ∧ y ≤ 13 → (13 - y) + (20 - y) ≤ 25 - y → y ≥ x) →
  (13 - x) + (20 - x) ≤ 25 - x →
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cut_length_l524_52412


namespace NUMINAMATH_CALUDE_largest_6k_plus_1_factor_of_11_factorial_l524_52410

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_factor (a b : ℕ) : Prop := b % a = 0

def is_of_form_6k_plus_1 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k + 1

theorem largest_6k_plus_1_factor_of_11_factorial :
  ∀ n : ℕ, is_factor n (factorial 11) → is_of_form_6k_plus_1 n → n ≤ 385 :=
by sorry

end NUMINAMATH_CALUDE_largest_6k_plus_1_factor_of_11_factorial_l524_52410


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l524_52425

theorem fractional_equation_solution :
  ∃ x : ℚ, x = -3/4 ∧ x / (x + 1) = 2 * x / (3 * x + 3) - 1 :=
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l524_52425


namespace NUMINAMATH_CALUDE_paige_homework_problems_l524_52411

/-- The number of math problems Paige had for homework -/
def math_problems : ℕ := 43

/-- The number of science problems Paige had for homework -/
def science_problems : ℕ := 12

/-- The number of problems Paige finished at school -/
def finished_problems : ℕ := 44

/-- The number of problems Paige had to do for homework -/
def homework_problems : ℕ := math_problems + science_problems - finished_problems

theorem paige_homework_problems :
  homework_problems = 11 := by sorry

end NUMINAMATH_CALUDE_paige_homework_problems_l524_52411


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l524_52481

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l524_52481


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l524_52445

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x + 8) - (4 / Real.sqrt (x + 8)) = 3 ∧ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l524_52445


namespace NUMINAMATH_CALUDE_family_reunion_attendance_l524_52480

/-- Calculates the number of people served given the amount of pasta used,
    based on a recipe where 2 pounds of pasta serves 7 people. -/
def people_served (pasta_pounds : ℚ) : ℚ :=
  (pasta_pounds / 2) * 7

/-- Theorem stating that 10 pounds of pasta will serve 35 people,
    given a recipe where 2 pounds of pasta serves 7 people. -/
theorem family_reunion_attendance :
  people_served 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_attendance_l524_52480


namespace NUMINAMATH_CALUDE_fraction_upper_bound_l524_52485

theorem fraction_upper_bound (x : ℝ) (h : x > 0) : x / (x^2 + 3*x + 1) ≤ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_upper_bound_l524_52485


namespace NUMINAMATH_CALUDE_total_students_in_schools_l524_52473

theorem total_students_in_schools (capacity1 capacity2 : ℕ) 
  (h1 : capacity1 = 400) 
  (h2 : capacity2 = 340) : 
  2 * capacity1 + 2 * capacity2 = 1480 :=
by sorry

end NUMINAMATH_CALUDE_total_students_in_schools_l524_52473


namespace NUMINAMATH_CALUDE_min_value_theorem_l524_52433

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z > 0 → z + y = 1 → (2 / (z + 3 * y) + 1 / (z - y)) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l524_52433


namespace NUMINAMATH_CALUDE_f_min_value_l524_52403

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

-- State the theorem
theorem f_min_value (a b : ℝ) :
  (∀ x > 0, f a b x ≤ 5) ∧ (∃ x > 0, f a b x = 5) →
  (∀ x < 0, f a b x ≥ -1) ∧ (∃ x < 0, f a b x = -1) :=
by sorry

end NUMINAMATH_CALUDE_f_min_value_l524_52403


namespace NUMINAMATH_CALUDE_sports_lottery_combinations_and_cost_l524_52400

/-- The number of ways to choose 3 consecutive numbers from 01 to 17 -/
def consecutive_three : ℕ := 15

/-- The number of ways to choose 2 consecutive numbers from 19 to 29 -/
def consecutive_two : ℕ := 10

/-- The number of ways to choose 1 number from 30 to 36 -/
def single_number : ℕ := 7

/-- The cost of each bet in yuan -/
def bet_cost : ℕ := 2

/-- The total number of combinations -/
def total_combinations : ℕ := consecutive_three * consecutive_two * single_number

/-- The total cost in yuan -/
def total_cost : ℕ := total_combinations * bet_cost

theorem sports_lottery_combinations_and_cost :
  total_combinations = 1050 ∧ total_cost = 2100 := by
  sorry

end NUMINAMATH_CALUDE_sports_lottery_combinations_and_cost_l524_52400


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l524_52456

theorem sum_of_three_consecutive_integers (a b c : ℤ) : 
  (a + 1 = b ∧ b + 1 = c) → c = 14 → a + b + c = 39 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l524_52456


namespace NUMINAMATH_CALUDE_jills_net_salary_l524_52414

/-- Calculates the net monthly salary given the discretionary income percentage and the amount left after allocations -/
def calculate_net_salary (discretionary_income_percentage : ℚ) (amount_left : ℚ) : ℚ :=
  (amount_left / (discretionary_income_percentage * (1 - 0.3 - 0.2 - 0.35))) * 100

/-- Proves that under the given conditions, Jill's net monthly salary is $3700 -/
theorem jills_net_salary :
  let discretionary_income_percentage : ℚ := 1/5
  let amount_left : ℚ := 111
  calculate_net_salary discretionary_income_percentage amount_left = 3700 := by
  sorry

#eval calculate_net_salary (1/5) 111

end NUMINAMATH_CALUDE_jills_net_salary_l524_52414


namespace NUMINAMATH_CALUDE_pet_store_dogs_l524_52468

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs is 3:5 and there are 18 cats, prove there are 30 dogs -/
theorem pet_store_dogs :
  let cat_ratio : ℕ := 3
  let dog_ratio : ℕ := 5
  let num_cats : ℕ := 18
  calculate_dogs cat_ratio dog_ratio num_cats = 30 := by
  sorry

#eval calculate_dogs 3 5 18

end NUMINAMATH_CALUDE_pet_store_dogs_l524_52468


namespace NUMINAMATH_CALUDE_randy_initial_money_l524_52452

theorem randy_initial_money :
  ∀ M : ℝ,
  (M - 10 - (M - 10) / 4 = 15) →
  M = 30 := by
sorry

end NUMINAMATH_CALUDE_randy_initial_money_l524_52452


namespace NUMINAMATH_CALUDE_bake_sale_pastries_sold_l524_52450

/-- Represents the number of pastries sold at a bake sale. -/
def pastries_sold (cupcakes cookies taken_home : ℕ) : ℕ :=
  cupcakes + cookies - taken_home

/-- Proves that the number of pastries sold is correct given the conditions. -/
theorem bake_sale_pastries_sold :
  pastries_sold 4 29 24 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_pastries_sold_l524_52450


namespace NUMINAMATH_CALUDE_point_not_on_line_l524_52466

theorem point_not_on_line (m b : ℝ) (h : m + b < 0) : 
  ¬(∃ (x y : ℝ), y = m * x + b ∧ x = 0 ∧ y = 20) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l524_52466


namespace NUMINAMATH_CALUDE_abc_inequalities_l524_52448

theorem abc_inequalities (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : 4*a^2 + b^2 + 16*c^2 = 1) : 
  (0 < a*b ∧ a*b < 1/4) ∧ 
  (1/a^2 + 1/b^2 + 1/(4*a*b*c^2) > 49) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l524_52448


namespace NUMINAMATH_CALUDE_parallelogram_area_l524_52453

/-- The area of a parallelogram with given dimensions -/
theorem parallelogram_area (base slant_height horiz_diff : ℝ) 
  (h_base : base = 20)
  (h_slant : slant_height = 6)
  (h_diff : horiz_diff = 5) :
  base * Real.sqrt (slant_height^2 - horiz_diff^2) = 20 * Real.sqrt 11 := by
  sorry

#check parallelogram_area

end NUMINAMATH_CALUDE_parallelogram_area_l524_52453


namespace NUMINAMATH_CALUDE_tom_total_weight_l524_52479

/-- Calculates the total weight Tom is moving with given his body weight, the weight he holds in each hand, and the weight of his vest. -/
def total_weight (tom_weight : ℝ) (hand_multiplier : ℝ) (vest_multiplier : ℝ) : ℝ :=
  tom_weight * hand_multiplier * 2 + tom_weight * vest_multiplier

/-- Theorem stating that Tom's total weight moved is 525 kg given the problem conditions. -/
theorem tom_total_weight :
  let tom_weight : ℝ := 150
  let hand_multiplier : ℝ := 1.5
  let vest_multiplier : ℝ := 0.5
  total_weight tom_weight hand_multiplier vest_multiplier = 525 := by
  sorry

end NUMINAMATH_CALUDE_tom_total_weight_l524_52479


namespace NUMINAMATH_CALUDE_new_student_weight_is_62_l524_52447

/-- The weight of the new student given the conditions of the problem -/
def new_student_weight (n : ℕ) (avg_decrease : ℚ) (old_student_weight : ℚ) : ℚ :=
  old_student_weight - n * avg_decrease

/-- Theorem stating that the weight of the new student is 62 kg -/
theorem new_student_weight_is_62 :
  new_student_weight 6 3 80 = 62 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_is_62_l524_52447


namespace NUMINAMATH_CALUDE_total_pencils_l524_52424

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l524_52424


namespace NUMINAMATH_CALUDE_convince_jury_l524_52487

-- Define the types of people
inductive PersonType
| Knight
| Liar
| Normal

-- Define the properties of a person
structure Person where
  type : PersonType
  guilty : Bool

-- Define the statement made by the person
def statement (p : Person) : Prop :=
  p.guilty ∧ p.type = PersonType.Liar

-- Define what it means for a person to be consistent with their statement
def consistent (p : Person) : Prop :=
  (p.type = PersonType.Knight ∧ statement p) ∨
  (p.type = PersonType.Liar ∧ ¬statement p) ∨
  (p.type = PersonType.Normal ∧ statement p)

-- Theorem to prove
theorem convince_jury :
  ∃ (p : Person), consistent p ∧ ¬p.guilty ∧ p.type ≠ PersonType.Knight :=
sorry

end NUMINAMATH_CALUDE_convince_jury_l524_52487


namespace NUMINAMATH_CALUDE_system_equation_solution_l524_52457

theorem system_equation_solution (x y some_number : ℝ) : 
  (2 * x + y = 7) → 
  (x + 2 * y = 5) → 
  (2 * x * y / some_number = 2) →
  some_number = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l524_52457


namespace NUMINAMATH_CALUDE_final_selling_price_l524_52465

/-- Calculate the final selling price of items with given conditions -/
theorem final_selling_price :
  let cycle_price : ℚ := 1400
  let helmet_price : ℚ := 400
  let safety_light_price : ℚ := 200
  let cycle_discount : ℚ := 0.1
  let helmet_discount : ℚ := 0.05
  let tax_rate : ℚ := 0.05
  let cycle_loss : ℚ := 0.12
  let helmet_profit : ℚ := 0.25
  let lock_price : ℚ := 300
  let transaction_fee : ℚ := 0.03

  let discounted_cycle := cycle_price * (1 - cycle_discount)
  let discounted_helmet := helmet_price * (1 - helmet_discount)
  let total_safety_lights := 2 * safety_light_price

  let total_before_tax := discounted_cycle + discounted_helmet + total_safety_lights
  let total_after_tax := total_before_tax * (1 + tax_rate)

  let selling_cycle := discounted_cycle * (1 - cycle_loss)
  let selling_helmet := discounted_helmet * (1 + helmet_profit)
  let selling_safety_lights := total_safety_lights

  let total_selling_before_fee := selling_cycle + selling_helmet + selling_safety_lights + lock_price
  let fee_amount := total_selling_before_fee * transaction_fee
  let total_selling_after_fee := total_selling_before_fee - fee_amount

  let final_price := ⌊total_selling_after_fee⌋

  final_price = 2215 := by sorry

end NUMINAMATH_CALUDE_final_selling_price_l524_52465


namespace NUMINAMATH_CALUDE_identity_is_unique_solution_l524_52428

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (x + f (y + x*y)) = (y + 1) * f (x + 1) - 1

/-- The main theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
  SatisfiesEquation f →
  ∀ x, x > 0 → f x = x :=
sorry

end NUMINAMATH_CALUDE_identity_is_unique_solution_l524_52428


namespace NUMINAMATH_CALUDE_correct_result_l524_52491

variables (a b c : ℝ)

def A : ℝ := 3 * a * b - 2 * a * c + 5 * b * c + 2 * (a * b + 2 * b * c - 4 * a * c)

theorem correct_result :
  A a b c - 2 * (a * b + 2 * b * c - 4 * a * c) = -a * b + 14 * a * c - 3 * b * c := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l524_52491


namespace NUMINAMATH_CALUDE_half_of_half_equals_half_l524_52455

theorem half_of_half_equals_half (x : ℝ) : (1/2 * (1/2 * x) = 1/2) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_half_of_half_equals_half_l524_52455


namespace NUMINAMATH_CALUDE_speedster_fraction_l524_52470

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  convertibles : ℕ
  non_speedsters : ℕ

/-- Conditions for the inventory -/
def inventory_conditions (inv : Inventory) : Prop :=
  inv.convertibles = (4 * inv.speedsters) / 5 ∧
  inv.non_speedsters = 60 ∧
  inv.convertibles = 96 ∧
  inv.total = inv.speedsters + inv.non_speedsters

/-- Theorem: The fraction of Speedsters in the inventory is 2/3 -/
theorem speedster_fraction (inv : Inventory) 
  (h : inventory_conditions inv) : 
  (inv.speedsters : ℚ) / inv.total = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_speedster_fraction_l524_52470


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l524_52496

theorem quadratic_real_roots (a b : ℝ) : 
  (∃ x : ℝ, x^2 + 2*(1+a)*x + (3*a^2 + 4*a*b + 4*b^2 + 2) = 0) ↔ (a = 1 ∧ b = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l524_52496


namespace NUMINAMATH_CALUDE_inequality_proof_l524_52409

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l524_52409


namespace NUMINAMATH_CALUDE_expression_equals_one_l524_52490

theorem expression_equals_one : 
  (150^2 - 9^2) / (110^2 - 13^2) * ((110 - 13) * (110 + 13)) / ((150 - 9) * (150 + 9)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l524_52490


namespace NUMINAMATH_CALUDE_train_speed_l524_52419

/-- The speed of a train given specific conditions involving a jogger --/
theorem train_speed (jogger_speed : ℝ) (jogger_ahead : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  jogger_ahead = 120 →
  train_length = 120 →
  passing_time = 24 →
  ∃ (train_speed : ℝ), train_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l524_52419


namespace NUMINAMATH_CALUDE_janes_reading_speed_l524_52483

theorem janes_reading_speed (total_pages : ℕ) (first_half_speed : ℕ) (total_days : ℕ) 
  (h1 : total_pages = 500)
  (h2 : first_half_speed = 10)
  (h3 : total_days = 75) :
  (total_pages / 2) / (total_days - (total_pages / 2) / first_half_speed) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_reading_speed_l524_52483


namespace NUMINAMATH_CALUDE_f_derivative_l524_52497

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem f_derivative :
  deriv f = λ x => -2 * Real.sin (2 * x) := by sorry

end NUMINAMATH_CALUDE_f_derivative_l524_52497


namespace NUMINAMATH_CALUDE_count_six_digit_integers_l524_52493

def digit_set : Multiset ℕ := {1, 1, 2, 3, 3, 3}

/-- The number of different positive, six-digit integers formed from the given digit set -/
def num_six_digit_integers : ℕ := sorry

theorem count_six_digit_integers : num_six_digit_integers = 60 := by sorry

end NUMINAMATH_CALUDE_count_six_digit_integers_l524_52493


namespace NUMINAMATH_CALUDE_no_real_solutions_l524_52488

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x*y - z^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l524_52488
