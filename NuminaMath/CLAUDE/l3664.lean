import Mathlib

namespace NUMINAMATH_CALUDE_triangle_angle_B_l3664_366426

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides

-- Define the theorem
theorem triangle_angle_B (t : Triangle) :
  t.A = π/4 ∧ t.a = Real.sqrt 2 ∧ t.b = Real.sqrt 3 →
  t.B = π/3 ∨ t.B = 2*π/3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l3664_366426


namespace NUMINAMATH_CALUDE_age_difference_l3664_366472

theorem age_difference (x y z : ℕ) (h1 : z = x - 19) : x + y - (y + z) = 19 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3664_366472


namespace NUMINAMATH_CALUDE_different_course_selections_count_l3664_366456

/-- The number of courses available to choose from -/
def num_courses : ℕ := 4

/-- The number of courses each person must choose -/
def courses_per_person : ℕ := 2

/-- The number of people choosing courses -/
def num_people : ℕ := 2

/-- Represents the ways two people can choose courses differently -/
def different_course_selections : ℕ := 30

/-- Theorem stating the number of ways two people can choose courses differently -/
theorem different_course_selections_count :
  (num_courses = 4) →
  (courses_per_person = 2) →
  (num_people = 2) →
  (different_course_selections = 30) :=
by sorry

end NUMINAMATH_CALUDE_different_course_selections_count_l3664_366456


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l3664_366427

/-- Calculates the total surface area of a cube with holes --/
def total_surface_area (cube_edge : ℝ) (hole_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_side^2
  let exposed_area := 6 * 4 * hole_side^2
  original_surface_area - hole_area + exposed_area

/-- Theorem: The total surface area of a cube with edge length 4 meters and square holes
    of side 2 meters cut through each face is 168 square meters --/
theorem cube_with_holes_surface_area :
  total_surface_area 4 2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l3664_366427


namespace NUMINAMATH_CALUDE_yan_distance_ratio_l3664_366453

/-- Represents the scenario of Yan's journey between home and stadium. -/
structure YanJourney where
  w : ℝ  -- Yan's walking speed
  x : ℝ  -- Distance from Yan to his home
  y : ℝ  -- Distance from Yan to the stadium
  h_positive : w > 0 -- Assumption that walking speed is positive
  h_between : x > 0 ∧ y > 0 -- Assumption that Yan is between home and stadium

/-- The theorem stating the ratio of Yan's distances. -/
theorem yan_distance_ratio (j : YanJourney) : 
  j.y / j.w = j.x / j.w + (j.x + j.y) / (7 * j.w) → j.x / j.y = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_yan_distance_ratio_l3664_366453


namespace NUMINAMATH_CALUDE_expression_evaluation_l3664_366477

theorem expression_evaluation :
  let a : ℚ := -1/3
  let expr := (3 - a) / (2*a - 4) / (a + 2 - 5/(a - 2))
  expr = 3/16 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3664_366477


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3664_366478

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ) ^ (3 * x) = 27 ^ (1/3) ∧ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3664_366478


namespace NUMINAMATH_CALUDE_correct_oranges_to_put_back_l3664_366447

/-- The number of oranges Mary must put back to achieve an average price of 50¢ -/
def oranges_to_put_back (apple_price orange_price : ℚ) (total_fruit : ℕ) (initial_avg_price : ℚ) : ℕ :=
  6

theorem correct_oranges_to_put_back 
  (apple_price : ℚ) 
  (orange_price : ℚ) 
  (total_fruit : ℕ) 
  (initial_avg_price : ℚ) 
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruit = 10)
  (h4 : initial_avg_price = 56/100) :
  let x := oranges_to_put_back apple_price orange_price total_fruit initial_avg_price
  let remaining_fruit := total_fruit - x
  let remaining_avg_price := 50/100
  (apple_price * (total_fruit - (total_fruit - remaining_fruit)) + 
   orange_price * (remaining_fruit - (total_fruit - remaining_fruit))) / remaining_fruit = remaining_avg_price :=
by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_to_put_back_l3664_366447


namespace NUMINAMATH_CALUDE_circle_radius_from_spherical_coords_l3664_366425

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/4) is √2/2 -/
theorem circle_radius_from_spherical_coords :
  let r : ℝ := Real.sqrt 2 / 2
  ∀ θ : ℝ,
  let x : ℝ := Real.sin (π/4 : ℝ) * Real.cos θ
  let y : ℝ := Real.sin (π/4 : ℝ) * Real.sin θ
  Real.sqrt (x^2 + y^2) = r :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_spherical_coords_l3664_366425


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3664_366458

theorem unique_integer_solution :
  ∃! z : ℤ, (5 * z ≤ 2 * z - 8) ∧ (-3 * z ≥ 18) ∧ (7 * z ≤ -3 * z - 21) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3664_366458


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3664_366445

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + (m-2)*x + m + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 + (m-2)*y + m + 1 = 0 → y = x) ↔ 
  (m = 0 ∨ m = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3664_366445


namespace NUMINAMATH_CALUDE_walkers_meet_at_calculated_point_l3664_366424

/-- Two people walking around a loop -/
structure WalkersOnLoop where
  loop_length : ℕ
  speed_ratio : ℕ

/-- The meeting point of two walkers -/
def meeting_point (w : WalkersOnLoop) : ℕ × ℕ :=
  (w.loop_length / (w.speed_ratio + 1), w.speed_ratio * w.loop_length / (w.speed_ratio + 1))

/-- Theorem: Walkers meet at the calculated point -/
theorem walkers_meet_at_calculated_point (w : WalkersOnLoop) 
  (h1 : w.loop_length = 24) 
  (h2 : w.speed_ratio = 3) : 
  meeting_point w = (6, 18) := by
  sorry

#eval meeting_point ⟨24, 3⟩

end NUMINAMATH_CALUDE_walkers_meet_at_calculated_point_l3664_366424


namespace NUMINAMATH_CALUDE_quadratic_point_distance_l3664_366493

/-- Given a quadratic function f(x) = ax² - 2ax + b where a > 0,
    and two points (x₁, y₁) and (x₂, y₂) on its graph where y₁ > y₂,
    prove that |x₁ - 1| > |x₂ - 1| -/
theorem quadratic_point_distance (a b x₁ y₁ x₂ y₂ : ℝ) 
  (ha : a > 0)
  (hf₁ : y₁ = a * x₁^2 - 2 * a * x₁ + b)
  (hf₂ : y₂ = a * x₂^2 - 2 * a * x₂ + b)
  (hy : y₁ > y₂) :
  |x₁ - 1| > |x₂ - 1| := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_distance_l3664_366493


namespace NUMINAMATH_CALUDE_animals_food_consumption_l3664_366469

/-- The total food consumption for a group of animals in one month -/
def total_food_consumption (num_animals : ℕ) (food_per_animal : ℕ) : ℕ :=
  num_animals * food_per_animal

/-- Theorem: Given 6 animals, with each animal eating 4 kg of food in one month,
    the total food consumption for all animals in one month is 24 kg -/
theorem animals_food_consumption :
  total_food_consumption 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_animals_food_consumption_l3664_366469


namespace NUMINAMATH_CALUDE_power_of_sum_squares_and_abs_l3664_366462

theorem power_of_sum_squares_and_abs (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : a^b = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_power_of_sum_squares_and_abs_l3664_366462


namespace NUMINAMATH_CALUDE_square_plus_one_ge_twice_abs_l3664_366420

theorem square_plus_one_ge_twice_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_ge_twice_abs_l3664_366420


namespace NUMINAMATH_CALUDE_complex_equality_l3664_366465

theorem complex_equality (z : ℂ) : z = -1 + (7/2) * I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧
  Complex.abs (z - 2) = Complex.abs (z + I) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3664_366465


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3664_366418

def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, 2, 3; 0, 1, 4; 5, 0, 1]

theorem matrix_equation_solution :
  ∃ (p q r : ℤ), 
    B^3 + p • B^2 + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℤ) = (0 : Matrix (Fin 3) (Fin 3) ℤ) ∧
    p = -41 ∧ q = -80 ∧ r = -460 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3664_366418


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l3664_366441

theorem smallest_n_satisfying_inequality : ∃ n : ℕ, 
  (∀ m : ℕ, 2006^1003 < m^2006 → n ≤ m) ∧ 2006^1003 < n^2006 ∧ n = 45 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l3664_366441


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3664_366401

/-- The lateral surface area of a cone with base area 25π and height 12 is 65π. -/
theorem cone_lateral_surface_area :
  ∀ (base_area height radius slant_height lateral_area : ℝ),
  base_area = 25 * Real.pi →
  height = 12 →
  base_area = Real.pi * radius^2 →
  slant_height^2 = radius^2 + height^2 →
  lateral_area = Real.pi * radius * slant_height →
  lateral_area = 65 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3664_366401


namespace NUMINAMATH_CALUDE_special_triangle_properties_l3664_366431

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the specific triangle in the problem -/
def SpecialTriangle (t : Triangle) : Prop :=
  3 * t.a * Real.cos t.C = 2 * t.c * Real.cos t.A ∧ 
  Real.tan t.C = 1/2 ∧
  t.b = 5

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.B = 3 * Real.pi / 4 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C = 5/2) := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_properties_l3664_366431


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_and_custom_definition_l3664_366496

/-- The diameter of a circle with area 400π cm² is 1600 cm, given that the diameter is defined as four times the square of the radius. -/
theorem circle_diameter_from_area_and_custom_definition :
  ∀ (r d : ℝ),
  r > 0 →
  400 * Real.pi = Real.pi * r^2 →
  d = 4 * r^2 →
  d = 1600 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_and_custom_definition_l3664_366496


namespace NUMINAMATH_CALUDE_square_sum_solution_l3664_366466

theorem square_sum_solution (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 130) : 
  x^2 + y^2 = 3049 / 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_solution_l3664_366466


namespace NUMINAMATH_CALUDE_quarter_circle_sum_approaches_semi_circumference_l3664_366481

/-- The sum of quarter-circle arc lengths approaches the semi-circumference as n approaches infinity --/
theorem quarter_circle_sum_approaches_semi_circumference (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |2 * n * (π * D / (4 * n)) - π * D / 2| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circle_sum_approaches_semi_circumference_l3664_366481


namespace NUMINAMATH_CALUDE_bryan_stones_sale_l3664_366421

/-- The total money Bryan received from selling his precious stones collection -/
def total_money (num_emeralds num_rubies num_sapphires : ℕ) 
  (price_emerald price_ruby price_sapphire : ℕ) : ℕ :=
  num_emeralds * price_emerald + num_rubies * price_ruby + num_sapphires * price_sapphire

/-- Theorem stating that Bryan received $17555 for his precious stones collection -/
theorem bryan_stones_sale : 
  total_money 3 2 3 1785 2650 2300 = 17555 := by
  sorry

#eval total_money 3 2 3 1785 2650 2300

end NUMINAMATH_CALUDE_bryan_stones_sale_l3664_366421


namespace NUMINAMATH_CALUDE_square_diff_squared_l3664_366443

theorem square_diff_squared : (7^2 - 5^2)^2 = 576 := by sorry

end NUMINAMATH_CALUDE_square_diff_squared_l3664_366443


namespace NUMINAMATH_CALUDE_sum_of_digits_nine_times_ascending_l3664_366468

/-- A function that checks if a natural number has digits in ascending order -/
def has_ascending_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem: For any number A with digits in ascending order, 
    the sum of digits of 9 * A is always 9 -/
theorem sum_of_digits_nine_times_ascending (A : ℕ) 
  (h : has_ascending_digits A) : sum_of_digits (9 * A) = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_nine_times_ascending_l3664_366468


namespace NUMINAMATH_CALUDE_max_sum_for_2029_product_l3664_366402

theorem max_sum_for_2029_product (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → 
  A * B * C = 2029 → 
  A + B + C ≤ 297 := by
sorry

end NUMINAMATH_CALUDE_max_sum_for_2029_product_l3664_366402


namespace NUMINAMATH_CALUDE_straight_line_shortest_l3664_366407

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line segment between two points
def LineSegment (p1 p2 : Point2D) : Set Point2D :=
  {p : Point2D | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y)}

-- Define the length of a path between two points
def PathLength (path : Set Point2D) : ℝ := sorry

-- Theorem: The straight line segment between two points has the shortest length among all paths between those points
theorem straight_line_shortest (p1 p2 : Point2D) :
  ∀ path : Set Point2D, p1 ∈ path ∧ p2 ∈ path →
    PathLength (LineSegment p1 p2) ≤ PathLength path :=
sorry

end NUMINAMATH_CALUDE_straight_line_shortest_l3664_366407


namespace NUMINAMATH_CALUDE_mixed_nuts_price_per_pound_l3664_366473

/-- Calculates the price per pound of mixed nuts given the following conditions:
  * Total weight of mixed nuts is 100 pounds
  * Price of peanuts is $3.50 per pound
  * Price of cashews is $4.00 per pound
  * Amount of cashews used is 60 pounds
-/
theorem mixed_nuts_price_per_pound 
  (total_weight : ℝ) 
  (peanut_price : ℝ) 
  (cashew_price : ℝ) 
  (cashew_weight : ℝ) 
  (h1 : total_weight = 100) 
  (h2 : peanut_price = 3.5) 
  (h3 : cashew_price = 4) 
  (h4 : cashew_weight = 60) : 
  (cashew_price * cashew_weight + peanut_price * (total_weight - cashew_weight)) / total_weight = 3.8 := by
  sorry

end NUMINAMATH_CALUDE_mixed_nuts_price_per_pound_l3664_366473


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l3664_366432

theorem systematic_sampling_probability
  (total_parts : Nat)
  (first_grade : Nat)
  (second_grade : Nat)
  (third_grade : Nat)
  (sample_size : Nat)
  (h1 : total_parts = 120)
  (h2 : first_grade = 24)
  (h3 : second_grade = 36)
  (h4 : third_grade = 60)
  (h5 : sample_size = 20)
  (h6 : total_parts = first_grade + second_grade + third_grade) :
  (sample_size : ℚ) / (total_parts : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l3664_366432


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_product_l3664_366440

theorem largest_divisor_of_consecutive_even_product : ∃ (n : ℕ), 
  (∀ (k : ℕ), k > 0 → 16 ∣ (2*k) * (2*k + 2) * (2*k + 4)) ∧ 
  (∀ (m : ℕ), m > 16 → ∃ (j : ℕ), j > 0 ∧ ¬(m ∣ (2*j) * (2*j + 2) * (2*j + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_product_l3664_366440


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l3664_366457

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(p ∣ n)

theorem smallest_non_prime_non_square_no_small_factors :
  (∀ m : ℕ, m < 4091 →
    is_prime m ∨
    is_perfect_square m ∨
    ¬(has_no_prime_factor_less_than m 60)) ∧
  ¬(is_prime 4091) ∧
  ¬(is_perfect_square 4091) ∧
  has_no_prime_factor_less_than 4091 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l3664_366457


namespace NUMINAMATH_CALUDE_perimeter_of_equilateral_triangle_with_base_8_l3664_366480

-- Define an equilateral triangle
structure EquilateralTriangle where
  base : ℝ
  is_positive : base > 0

-- Define the perimeter of an equilateral triangle
def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.base

-- Theorem statement
theorem perimeter_of_equilateral_triangle_with_base_8 :
  ∀ t : EquilateralTriangle, t.base = 8 → perimeter t = 24 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_equilateral_triangle_with_base_8_l3664_366480


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3664_366436

/-- Proves that the speed of a boat in still water is 16 km/hr, given the rate of the stream and the time and distance traveled downstream. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 8)
  (h3 : downstream_distance = 168) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 16 ∧ 
    downstream_distance = (still_water_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3664_366436


namespace NUMINAMATH_CALUDE_symmetric_curve_equation_l3664_366490

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line of symmetry
def symmetry_line : ℝ := 2

-- Define the symmetric point
def symmetric_point (x y : ℝ) : ℝ × ℝ := (4 - x, y)

-- Theorem statement
theorem symmetric_curve_equation :
  ∀ x y : ℝ, original_curve (4 - x) y → y^2 = 16 - 4*x :=
by sorry

end NUMINAMATH_CALUDE_symmetric_curve_equation_l3664_366490


namespace NUMINAMATH_CALUDE_rectangle_area_l3664_366488

theorem rectangle_area (w : ℝ) (h : w > 0) : 
  (4 * w = 4 * w) ∧ (2 * (4 * w) + 2 * w = 200) → 4 * w * w = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3664_366488


namespace NUMINAMATH_CALUDE_janet_walk_results_l3664_366406

/-- Represents Janet's walk in the city --/
structure JanetWalk where
  blocks_north : ℕ
  blocks_west : ℕ
  blocks_south : ℕ
  walking_speed : ℕ

/-- Calculates the time Janet needs to get home --/
def time_to_home (walk : JanetWalk) : ℚ :=
  (walk.blocks_west : ℚ) / walk.walking_speed

/-- Calculates the ratio of blocks walked east to south --/
def east_south_ratio (walk : JanetWalk) : ℚ × ℚ :=
  (walk.blocks_west, walk.blocks_south)

/-- Theorem stating the results of Janet's walk --/
theorem janet_walk_results (walk : JanetWalk) 
  (h1 : walk.blocks_north = 3)
  (h2 : walk.blocks_west = 7 * walk.blocks_north)
  (h3 : walk.blocks_south = 8)
  (h4 : walk.walking_speed = 2) :
  time_to_home walk = 21/2 ∧ east_south_ratio walk = (21, 8) := by
  sorry

#eval time_to_home { blocks_north := 3, blocks_west := 21, blocks_south := 8, walking_speed := 2 }
#eval east_south_ratio { blocks_north := 3, blocks_west := 21, blocks_south := 8, walking_speed := 2 }

end NUMINAMATH_CALUDE_janet_walk_results_l3664_366406


namespace NUMINAMATH_CALUDE_tangent_circle_area_l3664_366454

/-- A circle passing through two given points with tangent lines intersecting on x-axis --/
structure TangentCircle where
  /-- The center of the circle --/
  center : ℝ × ℝ
  /-- The radius of the circle --/
  radius : ℝ
  /-- The circle passes through point A --/
  passes_through_A : (center.1 - 7)^2 + (center.2 - 14)^2 = radius^2
  /-- The circle passes through point B --/
  passes_through_B : (center.1 - 13)^2 + (center.2 - 12)^2 = radius^2
  /-- The tangent lines at A and B intersect on the x-axis --/
  tangents_intersect_x_axis : ∃ x : ℝ, 
    (x - 7) * (center.2 - 14) = (center.1 - 7) * 14 ∧
    (x - 13) * (center.2 - 12) = (center.1 - 13) * 12

/-- The theorem stating that the area of the circle is 196π --/
theorem tangent_circle_area (ω : TangentCircle) : π * ω.radius^2 = 196 * π :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_area_l3664_366454


namespace NUMINAMATH_CALUDE_susan_jen_time_difference_l3664_366495

/-- A relay race with 4 runners -/
structure RelayRace where
  mary_time : ℝ
  susan_time : ℝ
  jen_time : ℝ
  tiffany_time : ℝ

/-- The conditions of the relay race -/
def race_conditions (race : RelayRace) : Prop :=
  race.mary_time = 2 * race.susan_time ∧
  race.susan_time > race.jen_time ∧
  race.jen_time = 30 ∧
  race.tiffany_time = race.mary_time - 7 ∧
  race.mary_time + race.susan_time + race.jen_time + race.tiffany_time = 223

/-- The theorem stating that Susan's time is 10 seconds longer than Jen's time -/
theorem susan_jen_time_difference (race : RelayRace) 
  (h : race_conditions race) : race.susan_time - race.jen_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_susan_jen_time_difference_l3664_366495


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3664_366467

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3664_366467


namespace NUMINAMATH_CALUDE_work_completion_time_l3664_366422

/-- The number of days A takes to complete the work alone -/
def a_days : ℝ := 12

/-- The number of days B takes to complete the work alone -/
def b_days : ℝ := 27.99999999999998

/-- The number of days A worked alone before B joined -/
def a_solo_days : ℝ := 2

/-- The total number of days it takes to complete the work when A and B work together -/
def total_days : ℝ := 9

theorem work_completion_time :
  let a_rate : ℝ := 1 / a_days
  let b_rate : ℝ := 1 / b_days
  let combined_rate : ℝ := a_rate + b_rate
  let work_done_by_a_solo : ℝ := a_rate * a_solo_days
  let remaining_work : ℝ := 1 - work_done_by_a_solo
  remaining_work / combined_rate + a_solo_days = total_days := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3664_366422


namespace NUMINAMATH_CALUDE_catfish_count_l3664_366423

theorem catfish_count (C : ℕ) (total_fish : ℕ) : 
  C + 10 + (3 * C / 2) = total_fish ∧ total_fish = 50 → C = 16 := by
  sorry

end NUMINAMATH_CALUDE_catfish_count_l3664_366423


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3664_366459

/-- The surface area of a sphere, given properties of a hemisphere. -/
theorem sphere_surface_area (r : ℝ) (h_base_area : π * r^2 = 3) (h_hemisphere_area : 3 * π * r^2 = 9) :
  ∃ A : ℝ → ℝ, A r = 4 * π * r^2 := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3664_366459


namespace NUMINAMATH_CALUDE_least_n_triple_f_not_one_digit_l3664_366400

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Function f as defined in the problem -/
def f (n : ℕ) : ℕ := sumOfDigits n

/-- Theorem stating that 19999999999999999999999 is the least natural number n 
    such that f(f(f(n))) is not a one-digit number -/
theorem least_n_triple_f_not_one_digit :
  ∀ k : ℕ, k < 19999999999999999999999 → f (f (f k)) < 10 ∧ 
  f (f (f 19999999999999999999999)) ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_least_n_triple_f_not_one_digit_l3664_366400


namespace NUMINAMATH_CALUDE_sqrt_80_bound_l3664_366455

theorem sqrt_80_bound (k : ℤ) : k < Real.sqrt 80 ∧ Real.sqrt 80 < k + 1 → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_80_bound_l3664_366455


namespace NUMINAMATH_CALUDE_average_weight_problem_l3664_366484

/-- Given the average weights of pairs and the weight of one individual, 
    prove the average weight of all three. -/
theorem average_weight_problem (a b c : ℝ) 
    (h1 : (a + b) / 2 = 25) 
    (h2 : (b + c) / 2 = 28) 
    (h3 : b = 16) : 
    (a + b + c) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l3664_366484


namespace NUMINAMATH_CALUDE_sum_of_compositions_l3664_366486

def p (x : ℝ) : ℝ := x^2 - 3

def q (x : ℝ) : ℝ := x - 2

def x_values : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_compositions : 
  (x_values.map (λ x => q (p x))).sum = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_compositions_l3664_366486


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l3664_366444

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nth_number_with_digit_sum_13 (n : ℕ+) : ℕ+ := sorry

/-- The theorem stating that the 11th number with digit sum 13 is 175 -/
theorem eleventh_number_with_digit_sum_13 : 
  nth_number_with_digit_sum_13 11 = 175 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l3664_366444


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l3664_366438

theorem product_zero_implies_factor_zero (a b : ℝ) : a * b = 0 → a = 0 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l3664_366438


namespace NUMINAMATH_CALUDE_arg_cube_quotient_complex_l3664_366474

theorem arg_cube_quotient_complex (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 3)
  (h2 : Complex.abs z₂ = 5)
  (h3 : Complex.abs (z₁ + z₂) = 7) :
  Complex.arg ((z₂ / z₁) ^ 3) = π :=
sorry

end NUMINAMATH_CALUDE_arg_cube_quotient_complex_l3664_366474


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3664_366461

theorem container_volume_ratio : 
  ∀ (A B : ℝ), A > 0 → B > 0 →
  (3/5 * A + 1/4 * B = 4/5 * B) →
  A / B = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3664_366461


namespace NUMINAMATH_CALUDE_power_calculation_l3664_366428

theorem power_calculation (n : ℝ) : 
  (3/5 : ℝ) * (14.500000000000002 : ℝ)^n = 126.15 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3664_366428


namespace NUMINAMATH_CALUDE_four_appears_after_24_terms_l3664_366492

/-- An arithmetic sequence with first term 100 and common difference -4 -/
def arithmeticSequence : ℕ → ℤ := λ n => 100 - 4 * (n - 1)

/-- The number of terms in the sequence before 4 appears -/
def termsBeforeFour : ℕ := 24

theorem four_appears_after_24_terms :
  arithmeticSequence (termsBeforeFour + 1) = 4 ∧
  ∀ k : ℕ, k ≤ termsBeforeFour → arithmeticSequence k > 4 :=
sorry

end NUMINAMATH_CALUDE_four_appears_after_24_terms_l3664_366492


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3664_366451

def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}

theorem complement_of_A_in_U : Aᶜ = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3664_366451


namespace NUMINAMATH_CALUDE_tanya_work_days_l3664_366446

/-- Given that Sakshi can complete a piece of work in 20 days and Tanya is 25% more efficient than Sakshi,
    prove that Tanya can complete the same piece of work in 16 days. -/
theorem tanya_work_days (sakshi_days : ℝ) (tanya_efficiency : ℝ) :
  sakshi_days = 20 →
  tanya_efficiency = 1.25 →
  (sakshi_days / tanya_efficiency : ℝ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_tanya_work_days_l3664_366446


namespace NUMINAMATH_CALUDE_evaluate_expression_l3664_366498

theorem evaluate_expression : 5000 * (5000 ^ 1000) = 5000 ^ 1001 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3664_366498


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3664_366489

theorem sum_of_x_and_y (x y : ℝ) 
  (eq1 : x^2 + x*y + y = 14) 
  (eq2 : y^2 + x*y + x = 28) : 
  x + y = -7 ∨ x + y = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3664_366489


namespace NUMINAMATH_CALUDE_section_area_theorem_l3664_366464

/-- Regular quadrilateral pyramid with given properties -/
structure RegularPyramid where
  -- Base side length
  base_side : ℝ
  -- Distance from apex to cutting plane
  apex_distance : ℝ

/-- Area of the section formed by a plane in the pyramid -/
def section_area (p : RegularPyramid) : ℝ := sorry

/-- Theorem stating the area of the section for the given pyramid -/
theorem section_area_theorem (p : RegularPyramid) 
  (h1 : p.base_side = 8 / Real.sqrt 7)
  (h2 : p.apex_distance = 2 / 3) : 
  section_area p = 6 := by sorry

end NUMINAMATH_CALUDE_section_area_theorem_l3664_366464


namespace NUMINAMATH_CALUDE_blackboard_remainder_l3664_366404

theorem blackboard_remainder (a : ℕ) : 
  a < 10 → (a + 100) % 7 = 5 → a = 5 := by sorry

end NUMINAMATH_CALUDE_blackboard_remainder_l3664_366404


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3664_366476

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3664_366476


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3664_366450

theorem rectangle_perimeter (area : ℝ) (length : ℝ) (h1 : area = 192) (h2 : length = 24) :
  2 * (length + area / length) = 64 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3664_366450


namespace NUMINAMATH_CALUDE_subtract_problem_l3664_366417

theorem subtract_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtract_problem_l3664_366417


namespace NUMINAMATH_CALUDE_blueberry_pies_l3664_366405

theorem blueberry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) :
  total_pies = 36 →
  apple_ratio = 3 →
  blueberry_ratio = 4 →
  cherry_ratio = 5 →
  blueberry_ratio * (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_blueberry_pies_l3664_366405


namespace NUMINAMATH_CALUDE_square_value_l3664_366439

theorem square_value : ∃ (square : ℤ), 9210 - 9124 = 210 - square ∧ square = 124 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l3664_366439


namespace NUMINAMATH_CALUDE_range_of_f_l3664_366435

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Define the domain
def D : Set ℝ := {x | -3 ≤ x ∧ x < 2}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ D, f x = y} = {y | 2 ≤ y ∧ y < 11} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3664_366435


namespace NUMINAMATH_CALUDE_plan_a_monthly_fee_is_9_l3664_366460

/-- Represents the cost per text message for Plan A -/
def plan_a_cost_per_text : ℚ := 25 / 100

/-- Represents the cost per text message for the other plan -/
def other_plan_cost_per_text : ℚ := 40 / 100

/-- Represents the number of text messages at which both plans cost the same -/
def equal_cost_messages : ℕ := 60

/-- The monthly fee for Plan A makes both plans cost the same at 60 messages -/
theorem plan_a_monthly_fee_is_9 :
  ∃ (monthly_fee : ℚ),
    plan_a_cost_per_text * equal_cost_messages + monthly_fee =
    other_plan_cost_per_text * equal_cost_messages ∧
    monthly_fee = 9 := by
  sorry

end NUMINAMATH_CALUDE_plan_a_monthly_fee_is_9_l3664_366460


namespace NUMINAMATH_CALUDE_mn_pq_ratio_l3664_366442

-- Define the points on a real line
variable (A B C M N P Q : ℝ)

-- Define the conditions
variable (h1 : A ≤ B ∧ B ≤ C)  -- B is on line segment AC
variable (h2 : M = (A + B) / 2)  -- M is midpoint of AB
variable (h3 : N = (A + C) / 2)  -- N is midpoint of AC
variable (h4 : P = (N + A) / 2)  -- P is midpoint of NA
variable (h5 : Q = (M + A) / 2)  -- Q is midpoint of MA

-- State the theorem
theorem mn_pq_ratio :
  |N - M| / |P - Q| = 2 :=
sorry

end NUMINAMATH_CALUDE_mn_pq_ratio_l3664_366442


namespace NUMINAMATH_CALUDE_matrix_determinant_l3664_366437

theorem matrix_determinant : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 0, 1; -5, 5, -4; 3, 3, 6]
  Matrix.det A = 96 := by sorry

end NUMINAMATH_CALUDE_matrix_determinant_l3664_366437


namespace NUMINAMATH_CALUDE_language_study_difference_l3664_366449

theorem language_study_difference (total : ℕ) (german_min german_max russian_min russian_max : ℕ) :
  total = 2500 →
  german_min = 1750 →
  german_max = 1875 →
  russian_min = 1000 →
  russian_max = 1125 →
  let m := german_min + russian_min - total
  let M := german_max + russian_max - total
  M - m = 250 := by sorry

end NUMINAMATH_CALUDE_language_study_difference_l3664_366449


namespace NUMINAMATH_CALUDE_number_of_arrangements_l3664_366414

/-- Represents a checkout lane with two checkout points -/
structure CheckoutLane :=
  (point1 : Bool)
  (point2 : Bool)

/-- Represents the arrangement of checkout lanes -/
def Arrangement := List CheckoutLane

/-- The total number of checkout lanes -/
def totalLanes : Nat := 6

/-- The number of lanes to be selected -/
def selectedLanes : Nat := 3

/-- Checks if the lanes in an arrangement are non-adjacent -/
def areNonAdjacent (arr : Arrangement) : Bool :=
  sorry

/-- Checks if at least one checkout point is open in each lane -/
def hasOpenPoint (lane : CheckoutLane) : Bool :=
  lane.point1 || lane.point2

/-- Counts the number of valid arrangements -/
def countArrangements : Nat :=
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem number_of_arrangements :
  countArrangements = 108 :=
sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l3664_366414


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l3664_366415

theorem investment_interest_calculation
  (total_investment : ℝ)
  (investment_at_6_percent : ℝ)
  (interest_rate_6_percent : ℝ)
  (interest_rate_9_percent : ℝ)
  (h1 : total_investment = 10000)
  (h2 : investment_at_6_percent = 7200)
  (h3 : interest_rate_6_percent = 0.06)
  (h4 : interest_rate_9_percent = 0.09) :
  let investment_at_9_percent := total_investment - investment_at_6_percent
  let interest_from_6_percent := investment_at_6_percent * interest_rate_6_percent
  let interest_from_9_percent := investment_at_9_percent * interest_rate_9_percent
  let total_interest := interest_from_6_percent + interest_from_9_percent
  total_interest = 684 := by sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l3664_366415


namespace NUMINAMATH_CALUDE_tissues_cost_is_two_l3664_366403

def cost_of_tissues (toilet_paper_rolls : ℕ) (paper_towel_rolls : ℕ) (tissue_boxes : ℕ)
                    (toilet_paper_cost : ℚ) (paper_towel_cost : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - (toilet_paper_rolls * toilet_paper_cost + paper_towel_rolls * paper_towel_cost)) / tissue_boxes

theorem tissues_cost_is_two :
  cost_of_tissues 10 7 3 (3/2) 2 35 = 2 :=
by sorry

end NUMINAMATH_CALUDE_tissues_cost_is_two_l3664_366403


namespace NUMINAMATH_CALUDE_west_movement_notation_l3664_366430

/-- Represents the direction of movement -/
inductive Direction
  | East
  | West

/-- Represents a movement with distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Converts a movement to its numerical representation -/
def toNumber (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.distance
  | Direction.West => -m.distance

theorem west_movement_notation :
  let eastMovement : Movement := ⟨5, Direction.East⟩
  let westMovement : Movement := ⟨3, Direction.West⟩
  toNumber eastMovement = 5 →
  toNumber westMovement = -3 := by
  sorry

end NUMINAMATH_CALUDE_west_movement_notation_l3664_366430


namespace NUMINAMATH_CALUDE_equation_comparison_l3664_366433

theorem equation_comparison : 
  (abs (-2))^3 = abs 2^3 ∧ 
  (-2)^2 = 2^2 ∧ 
  (-2)^3 = -(2^3) ∧ 
  (-2)^4 ≠ -(2^4) := by
sorry

end NUMINAMATH_CALUDE_equation_comparison_l3664_366433


namespace NUMINAMATH_CALUDE_m_range_l3664_366499

/-- A function f satisfying the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc (-1) 1 → f (-x) = -f x) ∧
  (∀ a b, a ∈ Set.Ioo (-1) 0 → b ∈ Set.Ioo (-1) 0 → a ≠ b → (f a - f b) / (a - b) > 0)

theorem m_range (f : ℝ → ℝ) (m : ℝ) (hf : f_conditions f) (h : f (m + 1) > f (2 * m)) :
  -1/2 ≤ m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l3664_366499


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3664_366487

/-- Given an arithmetic sequence {a_n} where a_1 = 13 and a_4 = 1,
    prove that the common difference d is -4. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)  -- The sequence a_n
  (h1 : a 1 = 13)  -- a_1 = 13
  (h4 : a 4 = 1)   -- a_4 = 1
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 2 - a 1 = -4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3664_366487


namespace NUMINAMATH_CALUDE_shoe_cost_comparison_l3664_366416

/-- Calculates the percentage increase in average cost per year of new shoes compared to repaired used shoes -/
theorem shoe_cost_comparison (used_repair_cost : ℝ) (used_lifespan : ℝ) (new_cost : ℝ) (new_lifespan : ℝ)
  (h1 : used_repair_cost = 11.50)
  (h2 : used_lifespan = 1)
  (h3 : new_cost = 28.00)
  (h4 : new_lifespan = 2)
  : (((new_cost / new_lifespan) - (used_repair_cost / used_lifespan)) / (used_repair_cost / used_lifespan)) * 100 = 21.74 := by
  sorry

end NUMINAMATH_CALUDE_shoe_cost_comparison_l3664_366416


namespace NUMINAMATH_CALUDE_checkerboard_probability_l3664_366491

/-- Represents a rectangular checkerboard -/
structure Checkerboard where
  length : ℕ
  width : ℕ

/-- Calculates the total number of squares on the checkerboard -/
def totalSquares (board : Checkerboard) : ℕ :=
  board.length * board.width

/-- Calculates the number of squares not touching or adjacent to any edge -/
def innerSquares (board : Checkerboard) : ℕ :=
  (board.length - 4) * (board.width - 4)

/-- The probability of choosing a square not touching or adjacent to any edge -/
def innerSquareProbability (board : Checkerboard) : ℚ :=
  innerSquares board / totalSquares board

theorem checkerboard_probability :
  ∃ (board : Checkerboard), board.length = 10 ∧ board.width = 6 ∧
  innerSquareProbability board = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l3664_366491


namespace NUMINAMATH_CALUDE_simplify_expression_l3664_366475

theorem simplify_expression (w : ℝ) : 2*w + 3 - 4*w - 5 + 6*w + 7 - 8*w - 9 = -4*w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3664_366475


namespace NUMINAMATH_CALUDE_marco_running_time_l3664_366497

-- Define the constants
def laps : ℕ := 7
def track_length : ℝ := 500
def first_segment : ℝ := 150
def second_segment : ℝ := 350
def speed_first : ℝ := 3
def speed_second : ℝ := 4

-- Define the theorem
theorem marco_running_time :
  let time_first := first_segment / speed_first
  let time_second := second_segment / speed_second
  let time_per_lap := time_first + time_second
  laps * time_per_lap = 962.5 := by sorry

end NUMINAMATH_CALUDE_marco_running_time_l3664_366497


namespace NUMINAMATH_CALUDE_integer_sqrt_pair_l3664_366413

theorem integer_sqrt_pair : ∃! (x y : ℕ), 
  ((x = 88209 ∧ y = 90288) ∨
   (x = 82098 ∧ y = 89028) ∨
   (x = 28098 ∧ y = 89082) ∨
   (x = 90882 ∧ y = 28809)) ∧
  ∃ (z : ℕ), z^2 = x^2 + y^2 := by
sorry

end NUMINAMATH_CALUDE_integer_sqrt_pair_l3664_366413


namespace NUMINAMATH_CALUDE_vasyas_numbers_l3664_366409

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l3664_366409


namespace NUMINAMATH_CALUDE_swimming_pool_area_l3664_366411

/-- Represents a rectangular swimming pool with given properties -/
structure SwimmingPool where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_condition : length = 2 * width + 40
  perimeter_condition : perimeter = 2 * (length + width)
  perimeter_value : perimeter = 800

/-- Calculates the area of a rectangular swimming pool -/
def pool_area (pool : SwimmingPool) : ℝ :=
  pool.width * pool.length

/-- Theorem stating that a swimming pool with the given properties has an area of 33600 square feet -/
theorem swimming_pool_area (pool : SwimmingPool) : pool_area pool = 33600 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_area_l3664_366411


namespace NUMINAMATH_CALUDE_pool_capacity_is_12000_l3664_366412

/-- Represents the capacity of a pool and its filling rates. -/
structure PoolSystem where
  capacity : ℝ
  bothValvesTime : ℝ
  firstValveTime : ℝ
  secondValveExtraRate : ℝ

/-- Theorem stating that under given conditions, the pool capacity is 12000 cubic meters. -/
theorem pool_capacity_is_12000 (p : PoolSystem)
  (h1 : p.bothValvesTime = 48)
  (h2 : p.firstValveTime = 120)
  (h3 : p.secondValveExtraRate = 50)
  (h4 : p.capacity / p.firstValveTime + (p.capacity / p.firstValveTime + p.secondValveExtraRate) = p.capacity / p.bothValvesTime) :
  p.capacity = 12000 := by
  sorry

#check pool_capacity_is_12000

end NUMINAMATH_CALUDE_pool_capacity_is_12000_l3664_366412


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l3664_366419

theorem complex_modulus_theorem : 
  let i : ℂ := Complex.I
  let T : ℂ := (1 + i)^19 - (1 - i)^19
  Complex.abs T = 512 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l3664_366419


namespace NUMINAMATH_CALUDE_knight_moves_equality_on_7x7_l3664_366470

/-- Represents a position on a chessboard --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a knight's move on a chessboard --/
inductive KnightMove : Position → Position → Prop
  | move_1 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 1, y + 2⟩
  | move_2 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 2, y + 1⟩
  | move_3 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 2, y - 1⟩
  | move_4 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 1, y - 2⟩
  | move_5 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 1, y - 2⟩
  | move_6 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 2, y - 1⟩
  | move_7 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 2, y + 1⟩
  | move_8 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 1, y + 2⟩

/-- Minimum number of moves for a knight to reach a target position from a start position --/
def minMoves (start target : Position) : Nat :=
  sorry

theorem knight_moves_equality_on_7x7 :
  let start := ⟨0, 0⟩
  let topRight := ⟨6, 6⟩
  let bottomRight := ⟨6, 0⟩
  minMoves start topRight = minMoves start bottomRight :=
by sorry

end NUMINAMATH_CALUDE_knight_moves_equality_on_7x7_l3664_366470


namespace NUMINAMATH_CALUDE_triangle_area_l3664_366429

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x)^2 + Real.sin x * Real.cos x

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  f (A / 2) = Real.sqrt 3 ∧
  a = 4 ∧
  b + c = 5 →
  (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3664_366429


namespace NUMINAMATH_CALUDE_g_15_equals_274_l3664_366482

/-- The function g defined for all natural numbers -/
def g (n : ℕ) : ℕ := n^2 + 2*n + 19

/-- Theorem stating that g(15) equals 274 -/
theorem g_15_equals_274 : g 15 = 274 := by
  sorry

end NUMINAMATH_CALUDE_g_15_equals_274_l3664_366482


namespace NUMINAMATH_CALUDE_triangle_properties_l3664_366463

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

def Triangle.isArithmetic (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

theorem triangle_properties (t : Triangle) 
    (h1 : t.C = 2 * t.A) 
    (h2 : Real.cos t.A = 3/4) : 
    (t.c / t.a = 3/2) ∧ t.isArithmetic := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3664_366463


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l3664_366479

/-- Triangle inequality for side lengths a, b, c -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The main inequality to be proved -/
def main_inequality (a b c : ℝ) : ℝ :=
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a)

theorem triangle_side_inequality (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (tri : triangle_inequality a b c) : 
  main_inequality a b c ≥ 0 ∧ 
  (main_inequality a b c = 0 ↔ a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_inequality_l3664_366479


namespace NUMINAMATH_CALUDE_train_length_l3664_366434

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 4 → speed * time * (1000 / 3600) = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3664_366434


namespace NUMINAMATH_CALUDE_tan_a_plus_pi_third_l3664_366452

theorem tan_a_plus_pi_third (a : Real) (h : Real.tan a = Real.sqrt 3) : 
  Real.tan (a + π/3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_a_plus_pi_third_l3664_366452


namespace NUMINAMATH_CALUDE_cooking_mode_and_median_l3664_366494

def cooking_data : List Nat := [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8]

def mode (data : List Nat) : Nat :=
  sorry

def median (data : List Nat) : Nat :=
  sorry

theorem cooking_mode_and_median :
  mode cooking_data = 6 ∧ median cooking_data = 6 := by
  sorry

end NUMINAMATH_CALUDE_cooking_mode_and_median_l3664_366494


namespace NUMINAMATH_CALUDE_painting_cost_in_cny_l3664_366448

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 5

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 8

/-- Cost of the painting in Namibian dollars -/
def painting_cost_nad : ℝ := 150

/-- Theorem stating the cost of the painting in Chinese yuan -/
theorem painting_cost_in_cny :
  (painting_cost_nad / usd_to_nad) * usd_to_cny = 240 := by
  sorry

end NUMINAMATH_CALUDE_painting_cost_in_cny_l3664_366448


namespace NUMINAMATH_CALUDE_star_seven_five_l3664_366471

/-- The star operation for positive integers -/
def star (a b : ℕ+) : ℚ :=
  (a * b - (a - b)) / (a + b)

/-- Theorem stating that 7 ★ 5 = 11/4 -/
theorem star_seven_five : star 7 5 = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_five_l3664_366471


namespace NUMINAMATH_CALUDE_triangle_area_l3664_366408

theorem triangle_area (base height : ℝ) (h1 : base = 4) (h2 : height = 6) :
  (base * height) / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3664_366408


namespace NUMINAMATH_CALUDE_max_phi_symmetric_sine_l3664_366410

/-- Given a function f(x) = 2sin(4x + φ) where φ < 0, if the graph of f(x) is symmetric
    about the line x = π/24, then the maximum value of φ is -2π/3. -/
theorem max_phi_symmetric_sine (φ : ℝ) (hφ : φ < 0) :
  (∀ x : ℝ, 2 * Real.sin (4 * x + φ) = 2 * Real.sin (4 * (π / 12 - x) + φ)) →
  (∃ (φ_max : ℝ), φ_max = -2 * π / 3 ∧ φ ≤ φ_max ∧ ∀ ψ, ψ < 0 → ψ ≤ φ_max) :=
by sorry

end NUMINAMATH_CALUDE_max_phi_symmetric_sine_l3664_366410


namespace NUMINAMATH_CALUDE_tan_value_from_trig_equation_l3664_366485

theorem tan_value_from_trig_equation (θ : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 2 * Real.sin θ * Real.sin (θ + π / 4) = 5 * Real.cos (2 * θ)) : 
  Real.tan θ = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_equation_l3664_366485


namespace NUMINAMATH_CALUDE_katie_new_games_l3664_366483

/-- Given that Katie has some new games and 39 old games,
    her friends have 34 new games, and Katie has 62 more games than her friends,
    prove that Katie has 57 new games. -/
theorem katie_new_games :
  ∀ (new_games : ℕ),
  new_games + 39 = 34 + 62 →
  new_games = 57 :=
by
  sorry

end NUMINAMATH_CALUDE_katie_new_games_l3664_366483
