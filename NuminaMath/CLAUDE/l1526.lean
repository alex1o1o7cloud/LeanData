import Mathlib

namespace NUMINAMATH_CALUDE_bill_fraction_l1526_152679

theorem bill_fraction (total_stickers : ℕ) (andrew_fraction : ℚ) (total_given : ℕ) 
  (h1 : total_stickers = 100)
  (h2 : andrew_fraction = 1 / 5)
  (h3 : total_given = 44) :
  let andrew_stickers := andrew_fraction * total_stickers
  let remaining_after_andrew := total_stickers - andrew_stickers
  let bill_stickers := total_given - andrew_stickers
  bill_stickers / remaining_after_andrew = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_bill_fraction_l1526_152679


namespace NUMINAMATH_CALUDE_special_polyhedron_interior_segments_l1526_152670

/-- Represents a convex polyhedron with specific face types -/
structure SpecialPolyhedron where
  square_faces : ℕ
  hexagon_faces : ℕ
  octagon_faces : ℕ
  vertex_configuration : Bool  -- True if each vertex meets one square, one hexagon, and one octagon

/-- Calculates the number of interior segments in the special polyhedron -/
def interior_segments (p : SpecialPolyhedron) : ℕ :=
  sorry

/-- Theorem stating the number of interior segments in the specific polyhedron -/
theorem special_polyhedron_interior_segments :
  let p : SpecialPolyhedron := {
    square_faces := 12,
    hexagon_faces := 8,
    octagon_faces := 6,
    vertex_configuration := true
  }
  interior_segments p = 840 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_interior_segments_l1526_152670


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l1526_152650

/-- The amount Joan spent on shorts -/
def shorts_cost : ℚ := 15

/-- The amount Joan spent on a shirt -/
def shirt_cost : ℚ := 12.51

/-- The total amount Joan spent on clothing -/
def total_cost : ℚ := 42.33

/-- The amount Joan spent on the jacket -/
def jacket_cost : ℚ := total_cost - shorts_cost - shirt_cost

theorem jacket_cost_calculation : jacket_cost = 14.82 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l1526_152650


namespace NUMINAMATH_CALUDE_chessboard_coverage_l1526_152681

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a domino -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Checks if a chessboard can be covered by dominoes -/
def can_cover (board : Chessboard) (domino : Domino) : Prop :=
  sorry

/-- Checks if a chessboard with one corner removed can be covered by dominoes -/
def can_cover_one_corner_removed (board : Chessboard) (domino : Domino) : Prop :=
  sorry

/-- Checks if a chessboard with two opposite corners removed can be covered by dominoes -/
def can_cover_two_corners_removed (board : Chessboard) (domino : Domino) : Prop :=
  sorry

theorem chessboard_coverage (board : Chessboard) (domino : Domino) :
  board.rows = 8 ∧ board.cols = 8 ∧ domino.length = 2 ∧ domino.width = 1 →
  (can_cover board domino) ∧
  ¬(can_cover_one_corner_removed board domino) ∧
  ¬(can_cover_two_corners_removed board domino) :=
sorry

end NUMINAMATH_CALUDE_chessboard_coverage_l1526_152681


namespace NUMINAMATH_CALUDE_park_visitors_l1526_152620

theorem park_visitors (total : ℕ) (men_fraction : ℚ) (women_student_fraction : ℚ) 
  (h1 : total = 1260)
  (h2 : men_fraction = 7 / 18)
  (h3 : women_student_fraction = 6 / 11) :
  (total : ℚ) * (1 - men_fraction) * women_student_fraction = 420 := by
  sorry

end NUMINAMATH_CALUDE_park_visitors_l1526_152620


namespace NUMINAMATH_CALUDE_bridge_length_l1526_152608

/-- The length of a bridge given specific train and crossing conditions -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length ∧
    bridge_length = 240 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1526_152608


namespace NUMINAMATH_CALUDE_y_coordinate_of_P_l1526_152699

/-- The y-coordinate of a point P satisfying certain conditions -/
theorem y_coordinate_of_P (A B C D P : ℝ × ℝ) : 
  A = (-4, 0) →
  B = (-1, 2) →
  C = (1, 2) →
  D = (4, 0) →
  dist P A + dist P D = 10 →
  dist P B + dist P C = 10 →
  P.2 = (-12 + 16 * Real.sqrt 16.5) / 5 :=
by sorry

end NUMINAMATH_CALUDE_y_coordinate_of_P_l1526_152699


namespace NUMINAMATH_CALUDE_quadratic_inequality_has_solution_l1526_152600

theorem quadratic_inequality_has_solution : ∃ x : ℝ, x^2 + 2*x - 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_has_solution_l1526_152600


namespace NUMINAMATH_CALUDE_point_A_coordinates_l1526_152685

theorem point_A_coordinates :
  ∀ a : ℤ,
  (a + 1 < 0) →
  (2 * a + 6 > 0) →
  (a + 1, 2 * a + 6) = (-1, 2) :=
by
  sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l1526_152685


namespace NUMINAMATH_CALUDE_adam_tattoo_count_l1526_152622

/-- The number of tattoos on each of Jason's arms -/
def jason_arm_tattoos : ℕ := 2

/-- The number of tattoos on each of Jason's legs -/
def jason_leg_tattoos : ℕ := 3

/-- The number of arms Jason has -/
def jason_arms : ℕ := 2

/-- The number of legs Jason has -/
def jason_legs : ℕ := 2

/-- The total number of tattoos Jason has -/
def jason_total_tattoos : ℕ := jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs

/-- Adam has three more than twice as many tattoos as Jason -/
def adam_tattoos : ℕ := 2 * jason_total_tattoos + 3

theorem adam_tattoo_count : adam_tattoos = 23 := by sorry

end NUMINAMATH_CALUDE_adam_tattoo_count_l1526_152622


namespace NUMINAMATH_CALUDE_garden_length_is_40_l1526_152651

/-- Represents a rectangular garden with given properties -/
structure Garden where
  total_distance : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ
  width_ratio : ℝ
  length : ℝ

/-- Theorem stating that the garden's length is 40 meters given the conditions -/
theorem garden_length_is_40 (g : Garden)
  (h1 : g.total_distance = 960)
  (h2 : g.length_walks = 24)
  (h3 : g.perimeter_walks = 8)
  (h4 : g.width_ratio = 1/2)
  (h5 : g.length * g.length_walks = g.total_distance)
  (h6 : (2 * g.length + 2 * (g.width_ratio * g.length)) * g.perimeter_walks = g.total_distance) :
  g.length = 40 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_is_40_l1526_152651


namespace NUMINAMATH_CALUDE_f_odd_g_even_l1526_152606

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the main property
axiom main_property : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y

-- Define f(0) = 0
axiom f_zero : f 0 = 0

-- Define f is not identically zero
axiom f_not_zero : ∃ x : ℝ, f x ≠ 0

-- Theorem to prove
theorem f_odd_g_even :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ y : ℝ, g (-y) = g y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_g_even_l1526_152606


namespace NUMINAMATH_CALUDE_second_car_speed_theorem_l1526_152680

/-- Two cars traveling on perpendicular roads towards an intersection -/
structure TwoCars where
  s₁ : ℝ  -- Initial distance of first car from intersection
  s₂ : ℝ  -- Initial distance of second car from intersection
  v₁ : ℝ  -- Speed of first car
  s  : ℝ  -- Distance between cars when first car reaches intersection

/-- The speed of the second car in the TwoCars scenario -/
def second_car_speed (cars : TwoCars) : Set ℝ :=
  {v₂ | v₂ = 12 ∨ v₂ = 16}

/-- Theorem stating the possible speeds of the second car -/
theorem second_car_speed_theorem (cars : TwoCars) 
    (h₁ : cars.s₁ = 500)
    (h₂ : cars.s₂ = 700)
    (h₃ : cars.v₁ = 10)  -- 36 km/h converted to m/s
    (h₄ : cars.s = 100) :
  second_car_speed cars = {12, 16} := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_theorem_l1526_152680


namespace NUMINAMATH_CALUDE_class_size_l1526_152611

theorem class_size :
  let both := 5  -- number of people who like both baseball and football
  let baseball_only := 2  -- number of people who only like baseball
  let football_only := 3  -- number of people who only like football
  let neither := 6  -- number of people who like neither baseball nor football
  both + baseball_only + football_only + neither = 16 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1526_152611


namespace NUMINAMATH_CALUDE_three_digit_combinations_l1526_152696

def set1 : Finset Nat := {0, 2, 4}
def set2 : Finset Nat := {1, 3, 5}

theorem three_digit_combinations : 
  (Finset.card set1) * (Finset.card set2) * (Finset.card set2 - 1) = 48 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_combinations_l1526_152696


namespace NUMINAMATH_CALUDE_rational_function_equation_l1526_152697

theorem rational_function_equation (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x + y) →
  (∀ x : ℚ, f x = x ∨ f x = -x) := by
sorry

end NUMINAMATH_CALUDE_rational_function_equation_l1526_152697


namespace NUMINAMATH_CALUDE_inverse_function_problem_l1526_152692

theorem inverse_function_problem (f : ℝ → ℝ) (f_inv : ℝ → ℝ) :
  (∀ x, f_inv x = 2^(x + 1)) → f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l1526_152692


namespace NUMINAMATH_CALUDE_point_on_line_l1526_152682

/-- Given points A and B in the Cartesian plane, if point C satisfies the vector equation,
    then C lies on the line passing through A and B. -/
theorem point_on_line (A B C : ℝ × ℝ) (α β : ℝ) :
  A = (3, 1) →
  B = (-1, 3) →
  α + β = 1 →
  C = (α * A.1 + β * B.1, α * A.2 + β * B.2) →
  C.1 + 2 * C.2 - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1526_152682


namespace NUMINAMATH_CALUDE_sum_of_max_min_S_l1526_152686

theorem sum_of_max_min_S (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h1 : x + y = 10) (h2 : y + z = 8) : 
  let S := x + z
  ∃ (S_min S_max : ℝ), 
    (∀ s, (∃ x' z', x' ≥ 0 ∧ z' ≥ 0 ∧ ∃ y', y' ≥ 0 ∧ x' + y' = 10 ∧ y' + z' = 8 ∧ s = x' + z') → s ≥ S_min) ∧
    (∀ s, (∃ x' z', x' ≥ 0 ∧ z' ≥ 0 ∧ ∃ y', y' ≥ 0 ∧ x' + y' = 10 ∧ y' + z' = 8 ∧ s = x' + z') → s ≤ S_max) ∧
    S_min + S_max = 20 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_S_l1526_152686


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1526_152614

theorem water_tank_capacity (initial_fraction : Rat) (final_fraction : Rat) (removed_liters : ℕ) 
  (h1 : initial_fraction = 2/3)
  (h2 : final_fraction = 1/3)
  (h3 : removed_liters = 20)
  (h4 : initial_fraction * tank_capacity - removed_liters = final_fraction * tank_capacity) :
  tank_capacity = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1526_152614


namespace NUMINAMATH_CALUDE_range_of_2alpha_minus_beta_l1526_152630

theorem range_of_2alpha_minus_beta (α β : ℝ) 
  (h : -π/2 < α ∧ α < β ∧ β < π/2) : 
  -3*π/2 < 2*α - β ∧ 2*α - β < π/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2alpha_minus_beta_l1526_152630


namespace NUMINAMATH_CALUDE_f_composition_result_l1526_152655

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else 2^x

theorem f_composition_result : f (f (1/9)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_result_l1526_152655


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1526_152629

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_prod : a 4 * a 5 * a 6 = 27) :
  a 1 * a 9 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1526_152629


namespace NUMINAMATH_CALUDE_jaden_final_car_count_l1526_152621

/-- The number of toy cars Jaden has after all transactions -/
def final_car_count (initial : ℕ) (bought : ℕ) (birthday : ℕ) (to_sister : ℕ) (to_friend : ℕ) : ℕ :=
  initial + bought + birthday - to_sister - to_friend

/-- Theorem stating that Jaden's final car count is 43 -/
theorem jaden_final_car_count :
  final_car_count 14 28 12 8 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_jaden_final_car_count_l1526_152621


namespace NUMINAMATH_CALUDE_max_travel_distance_proof_l1526_152649

/-- The distance (in km) a tire can travel on the front wheel before wearing out -/
def front_tire_lifespan : ℝ := 25000

/-- The distance (in km) a tire can travel on the rear wheel before wearing out -/
def rear_tire_lifespan : ℝ := 15000

/-- The maximum distance (in km) a motorcycle can travel before its tires are completely worn out,
    given that the tires are exchanged between front and rear wheels at the optimal time -/
def max_travel_distance : ℝ := 18750

/-- Theorem stating that the calculated maximum travel distance is correct -/
theorem max_travel_distance_proof :
  max_travel_distance = (front_tire_lifespan * rear_tire_lifespan) / (front_tire_lifespan / 2 + rear_tire_lifespan / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_travel_distance_proof_l1526_152649


namespace NUMINAMATH_CALUDE_constant_m_value_l1526_152613

theorem constant_m_value (x y z m : ℝ) :
  (5^2 / (x + y) = m / (x + 2*z)) ∧ (m / (x + 2*z) = 7^2 / (y - 2*z)) →
  m = 74 := by
  sorry

end NUMINAMATH_CALUDE_constant_m_value_l1526_152613


namespace NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l1526_152623

def fibonacci_factorial_series : List Nat := [1, 1, 2, 3, 5, 8, 13, 21]

def last_two_digits (n : Nat) : Nat :=
  n % 100

def sum_last_two_digits (series : List Nat) : Nat :=
  (series.map (λ x => last_two_digits (Nat.factorial x))).sum % 100

theorem fibonacci_factorial_last_two_digits_sum :
  sum_last_two_digits fibonacci_factorial_series = 5 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l1526_152623


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1526_152636

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 1)) ↔ (∃ x₀ : ℝ, x₀^2 + 1 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1526_152636


namespace NUMINAMATH_CALUDE_exponential_function_property_l1526_152672

theorem exponential_function_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x - 2
  f 0 = -1 := by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l1526_152672


namespace NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l1526_152610

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- The sum of the measures of the six interior angles of a hexagon is 720 degrees -/
theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l1526_152610


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1526_152643

/-- Given a real number m and a complex number z defined as z = (2m² + m - 1) + (-m² - 3m - 2)i,
    if z is purely imaginary, then m = 1/2. -/
theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (2 * m^2 + m - 1) (-m^2 - 3*m - 2)
  (z.re = 0 ∧ z.im ≠ 0) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1526_152643


namespace NUMINAMATH_CALUDE_recurrence_solution_l1526_152640

def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n - 3 * a (n - 1) - 10 * a (n - 2) = 28 * (5 ^ n)

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 0 = 25 ∧ a 1 = 120

def general_term (n : ℕ) : ℝ :=
  (20 * n + 10) * (5 ^ n) + 15 * ((-2) ^ n)

theorem recurrence_solution (a : ℕ → ℝ) :
  recurrence_relation a ∧ initial_conditions a →
  ∀ n : ℕ, a n = general_term n := by
  sorry

end NUMINAMATH_CALUDE_recurrence_solution_l1526_152640


namespace NUMINAMATH_CALUDE_range_of_k_l1526_152691

/-- Given x ∈ (0, 2), prove that x/(e^x) < 1/(k + 2x - x^2) holds if and only if k ∈ [0, e-1) -/
theorem range_of_k (x : ℝ) (hx : x ∈ Set.Ioo 0 2) :
  (∀ k : ℝ, x / Real.exp x < 1 / (k + 2 * x - x^2)) ↔ k ∈ Set.Icc 0 (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l1526_152691


namespace NUMINAMATH_CALUDE_no_first_quadrant_intersection_l1526_152664

/-- A linear function y = -3x + m -/
def linear_function (x : ℝ) (m : ℝ) : ℝ := -3 * x + m

/-- The first quadrant of the coordinate plane -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem no_first_quadrant_intersection :
  ∀ x y : ℝ, first_quadrant x y → linear_function x (-1) ≠ y := by
  sorry

end NUMINAMATH_CALUDE_no_first_quadrant_intersection_l1526_152664


namespace NUMINAMATH_CALUDE_smallest_d_for_inverse_l1526_152695

def g (x : ℝ) : ℝ := (x - 3)^2 - 8

theorem smallest_d_for_inverse :
  ∀ d : ℝ, (∀ x y, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_for_inverse_l1526_152695


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_iff_a_eq_one_l1526_152665

/-- The quadratic function f(x) = ax^2 - (a+1)x + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

/-- The solution set of f(x) < 0 is empty --/
def has_empty_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≥ 0

theorem quadratic_inequality_empty_iff_a_eq_one :
  ∀ a : ℝ, has_empty_solution_set a ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_iff_a_eq_one_l1526_152665


namespace NUMINAMATH_CALUDE_a_1998_value_l1526_152653

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n m, n < m → a n < a m) ∧ 
  (∀ k : ℕ, ∃! (i j k : ℕ), k = a i + 2 * a j + 4 * a k)

theorem a_1998_value (a : ℕ → ℕ) (h : is_valid_sequence a) : a 1998 = 1227096648 := by
  sorry

end NUMINAMATH_CALUDE_a_1998_value_l1526_152653


namespace NUMINAMATH_CALUDE_donna_marcia_pencils_l1526_152669

/-- The number of pencils Cindi bought -/
def cindi_pencils : ℕ := 60

/-- The number of pencils Marcia bought -/
def marcia_pencils : ℕ := 2 * cindi_pencils

/-- The number of pencils Donna bought -/
def donna_pencils : ℕ := 3 * marcia_pencils

/-- The total number of pencils bought by Donna and Marcia -/
def total_pencils : ℕ := donna_pencils + marcia_pencils

theorem donna_marcia_pencils :
  total_pencils = 480 :=
sorry

end NUMINAMATH_CALUDE_donna_marcia_pencils_l1526_152669


namespace NUMINAMATH_CALUDE_game_score_l1526_152602

/-- Calculates the total score of a father and son in a game where the son scores three times more than the father. -/
def totalScore (fatherScore : ℕ) : ℕ :=
  fatherScore + 3 * fatherScore

/-- Theorem stating that when the father scores 7 points, the total score is 28 points. -/
theorem game_score : totalScore 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_game_score_l1526_152602


namespace NUMINAMATH_CALUDE_area_is_54_height_is_7_2_l1526_152677

/-- A triangle with side lengths 9, 12, and 15 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : a = 9
  h_b : b = 12
  h_c : c = 15
  h_right : a ^ 2 + b ^ 2 = c ^ 2

/-- The area of the triangle is 54 square units -/
theorem area_is_54 (t : RightTriangle) : (1 / 2) * t.a * t.b = 54 := by sorry

/-- The height from the right angle vertex to the hypotenuse is 7.2 units -/
theorem height_is_7_2 (t : RightTriangle) : (t.a * t.b) / t.c = 7.2 := by sorry

end NUMINAMATH_CALUDE_area_is_54_height_is_7_2_l1526_152677


namespace NUMINAMATH_CALUDE_inequality_coverage_l1526_152627

theorem inequality_coverage (a : ℝ) : 
  (∀ x : ℝ, (2 * a - x > 1 ∧ 2 * x + 5 > 3 * a) → (1 ≤ x ∧ x ≤ 6)) →
  (7/3 ≤ a ∧ a ≤ 7/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_coverage_l1526_152627


namespace NUMINAMATH_CALUDE_point_on_number_line_l1526_152635

theorem point_on_number_line (A : ℝ) : (|A| = 3) ↔ (A = 3 ∨ A = -3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_number_line_l1526_152635


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1526_152662

theorem complex_equation_solution (a b c : ℤ) : 
  (a * (3 - Complex.I)^4 + b * (3 - Complex.I)^3 + c * (3 - Complex.I)^2 + b * (3 - Complex.I) + a = 0) →
  (Int.gcd a (Int.gcd b c) = 1) →
  (abs c = 109) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1526_152662


namespace NUMINAMATH_CALUDE_sin_2012_degrees_l1526_152654

theorem sin_2012_degrees : Real.sin (2012 * π / 180) = -Real.sin (32 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_2012_degrees_l1526_152654


namespace NUMINAMATH_CALUDE_annular_ring_area_l1526_152690

/-- Given a circle and a chord AB divided by point C such that AC = a and BC = b,
    the area of the annular ring formed when C traces another circle as AB's position changes
    is π(a + b)²/4. -/
theorem annular_ring_area (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let chord_length := a + b
  ∃ (R : ℝ), R > chord_length / 2 →
    (π * (chord_length ^ 2) / 4 : ℝ) =
      π * R ^ 2 - π * (R ^ 2 - chord_length ^ 2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_annular_ring_area_l1526_152690


namespace NUMINAMATH_CALUDE_solve_jerichos_money_problem_l1526_152625

def jerichos_money_problem (initial_amount debt_to_annika : ℕ) : Prop :=
  let debt_to_manny := 2 * debt_to_annika
  let total_debt := debt_to_annika + debt_to_manny
  let remaining_amount := initial_amount - total_debt
  (initial_amount = 3 * 90) ∧ 
  (debt_to_annika = 20) ∧
  (remaining_amount = 210)

theorem solve_jerichos_money_problem :
  jerichos_money_problem 270 20 := by sorry

end NUMINAMATH_CALUDE_solve_jerichos_money_problem_l1526_152625


namespace NUMINAMATH_CALUDE_min_k_existence_l1526_152673

open Real

theorem min_k_existence (k : ℕ) : (∃ x₀ : ℝ, x₀ > 2 ∧ k * (x₀ - 2) > x₀ * (log x₀ + 1)) ↔ k ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_min_k_existence_l1526_152673


namespace NUMINAMATH_CALUDE_september_birth_percentage_l1526_152604

theorem september_birth_percentage (total_authors : ℕ) (september_authors : ℕ) :
  total_authors = 120 →
  september_authors = 15 →
  (september_authors : ℚ) / (total_authors : ℚ) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_september_birth_percentage_l1526_152604


namespace NUMINAMATH_CALUDE_decimal_point_error_l1526_152675

theorem decimal_point_error (actual_amount : ℚ) : 
  (actual_amount * 10 - actual_amount = 153) → actual_amount = 17 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_error_l1526_152675


namespace NUMINAMATH_CALUDE_special_function_at_2021_l1526_152660

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2))

/-- The main theorem stating that any function satisfying SpecialFunction has f(2021) = 2 -/
theorem special_function_at_2021 (f : ℝ → ℝ) (h : SpecialFunction f) : f 2021 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_2021_l1526_152660


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1526_152641

theorem rectangle_ratio (w l : ℝ) (h1 : w = 5) (h2 : l * w = 75) : l / w = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1526_152641


namespace NUMINAMATH_CALUDE_ducks_in_lake_l1526_152666

/-- The number of ducks swimming in a lake after multiple groups join -/
def total_ducks (initial : ℕ) (first_group : ℕ) (additional : ℕ) : ℕ :=
  initial + first_group + additional

/-- Theorem stating the total number of ducks in the lake -/
theorem ducks_in_lake : 
  ∀ x : ℕ, total_ducks 13 20 x = 33 + x :=
by
  sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l1526_152666


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l1526_152687

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + m) - a n = m * (a (n + 1) - a n)

theorem ninth_term_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3/8)
  (h_seventeenth : a 17 = 2/3) :
  a 9 = 25/48 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l1526_152687


namespace NUMINAMATH_CALUDE_pencils_taken_l1526_152644

theorem pencils_taken (initial_pencils : ℕ) (pencils_left : ℕ) (h1 : initial_pencils = 34) (h2 : pencils_left = 12) :
  initial_pencils - pencils_left = 22 := by
sorry

end NUMINAMATH_CALUDE_pencils_taken_l1526_152644


namespace NUMINAMATH_CALUDE_day_of_week_N_minus_1_l1526_152676

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  value : ℕ
  is_leap : Bool

/-- Function to get the day of the week for a given day in a year -/
def day_of_week (year : Year) (day : ℕ) : DayOfWeek := sorry

/-- Function to get the next day of the week -/
def next_day (d : DayOfWeek) : DayOfWeek := sorry

/-- Function to get the previous day of the week -/
def prev_day (d : DayOfWeek) : DayOfWeek := sorry

theorem day_of_week_N_minus_1 
  (N : Year)
  (h1 : N.is_leap = true)
  (h2 : day_of_week N 250 = DayOfWeek.Friday)
  (h3 : (Year.mk (N.value + 1) true).is_leap = true)
  (h4 : day_of_week (Year.mk (N.value + 1) true) 150 = DayOfWeek.Friday) :
  day_of_week (Year.mk (N.value - 1) false) 50 = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_day_of_week_N_minus_1_l1526_152676


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1526_152658

-- Define the hyperbola
structure Hyperbola where
  asymptote_slope : ℝ

-- Define eccentricity
def eccentricity (h : Hyperbola) : Set ℝ :=
  {e : ℝ | e = 2 ∨ e = (2 * Real.sqrt 3) / 3}

-- Theorem statement
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = Real.sqrt 3) : 
  ∃ e : ℝ, e ∈ eccentricity h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1526_152658


namespace NUMINAMATH_CALUDE_correct_admin_in_sample_l1526_152648

/-- Represents the composition of staff in a school -/
structure StaffComposition where
  total : ℕ
  administrative : ℕ
  teaching : ℕ
  teaching_support : ℕ

/-- Represents a stratified sample from the staff -/
structure StratifiedSample where
  size : ℕ
  administrative : ℕ

/-- Calculates the correct number of administrative personnel in a stratified sample -/
def calculate_admin_in_sample (staff : StaffComposition) (sample : StratifiedSample) : ℕ :=
  (staff.administrative * sample.size) / staff.total

/-- The theorem to be proved -/
theorem correct_admin_in_sample (staff : StaffComposition) (sample : StratifiedSample) : 
  staff.total = 200 →
  staff.administrative = 24 →
  staff.teaching = 10 * staff.teaching_support →
  sample.size = 50 →
  calculate_admin_in_sample staff sample = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_admin_in_sample_l1526_152648


namespace NUMINAMATH_CALUDE_range_of_m_l1526_152657

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - 3| ≤ 2 → (x - m + 1) * (x - m - 1) ≤ 0) ∧
        ((x < 1 ∨ x > 5) → (x < m - 1 ∨ x > m + 1)) ∧
        (∃ x, (x < m - 1 ∨ x > m + 1) ∧ ¬(x < 1 ∨ x > 5)))
  → 2 ≤ m ∧ m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1526_152657


namespace NUMINAMATH_CALUDE_sufficient_conditions_for_quadratic_inequality_l1526_152647

theorem sufficient_conditions_for_quadratic_inequality :
  (∀ x, x < 4 → x^2 - 2*x - 8 < 0) ∨
  (∀ x, 0 < x ∧ x < 4 → x^2 - 2*x - 8 < 0) ∨
  (∀ x, -2 < x ∧ x < 4 → x^2 - 2*x - 8 < 0) ∨
  (∀ x, -2 < x ∧ x < 3 → x^2 - 2*x - 8 < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_conditions_for_quadratic_inequality_l1526_152647


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1526_152693

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the point P
def point_P : ℝ × ℝ := (1, 2)

-- Define a function to check if a line is tangent to the circle C at a point
def is_tangent_to_C (a b c : ℝ) (x y : ℝ) : Prop :=
  circle_C x y ∧ a*x + b*y + c = 0 ∧
  ∀ x' y', circle_C x' y' → (a*x' + b*y' + c)^2 ≥ (a^2 + b^2) * (x'^2 + y'^2 - 2)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    is_tangent_to_C (point_P.1 - x₁) (point_P.2 - y₁) (x₁*(x₁ - point_P.1) + y₁*(y₁ - point_P.2)) x₁ y₁ ∧
    is_tangent_to_C (point_P.1 - x₂) (point_P.2 - y₂) (x₂*(x₂ - point_P.1) + y₂*(y₂ - point_P.2)) x₂ y₂ ∧
    x₁ + 2*y₁ - 2 = 0 ∧ x₂ + 2*y₂ - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1526_152693


namespace NUMINAMATH_CALUDE_system_equations_range_l1526_152616

theorem system_equations_range (a b x y : ℝ) : 
  3 * x - y = 2 * a - 5 →
  x + 2 * y = 3 * a + 3 →
  x > 0 →
  y > 0 →
  a - b = 4 →
  b < 2 →
  a > 1 ∧ -2 < a + b ∧ a + b < 8 := by
sorry

end NUMINAMATH_CALUDE_system_equations_range_l1526_152616


namespace NUMINAMATH_CALUDE_lemonade_water_calculation_l1526_152659

/-- Represents the ratio of ingredients in the lemonade recipe -/
structure LemonadeRatio where
  water : ℚ
  lemon_juice : ℚ
  sugar : ℚ

/-- Calculates the amount of water needed for a given lemonade recipe and total volume -/
def water_needed (ratio : LemonadeRatio) (total_volume : ℚ) (quarts_per_gallon : ℚ) : ℚ :=
  let total_parts := ratio.water + ratio.lemon_juice + ratio.sugar
  let water_fraction := ratio.water / total_parts
  water_fraction * total_volume * quarts_per_gallon

/-- Proves that the amount of water needed for the given recipe and volume is 4 quarts -/
theorem lemonade_water_calculation (ratio : LemonadeRatio) 
  (h1 : ratio.water = 6)
  (h2 : ratio.lemon_juice = 2)
  (h3 : ratio.sugar = 1)
  (h4 : quarts_per_gallon = 4) :
  water_needed ratio (3/2) quarts_per_gallon = 4 := by
  sorry

#eval water_needed ⟨6, 2, 1⟩ (3/2) 4

end NUMINAMATH_CALUDE_lemonade_water_calculation_l1526_152659


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l1526_152661

open Real

/-- The minimum distance between a point on the line y = (12/5)x - 3 and a point on the parabola y = x^2 is 3/5. -/
theorem min_distance_line_parabola :
  let line := fun x => (12/5) * x - 3
  let parabola := fun x => x^2
  ∃ (a b : ℝ),
    (∀ x y : ℝ, 
      (y = line x ∨ y = parabola x) → 
      (a - x)^2 + (line a - y)^2 ≥ (3/5)^2) ∧
    line a = parabola b ∧
    (a - b)^2 + (line a - parabola b)^2 = (3/5)^2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l1526_152661


namespace NUMINAMATH_CALUDE_max_value_function_l1526_152624

theorem max_value_function (a b : ℝ) (h1 : a > b) (h2 : b ≥ 0) :
  ∃ M : ℝ, M = Real.sqrt ((a - b)^2 + a^2) ∧
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 →
    (a - b) * Real.sqrt (1 - x^2) + a * x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_function_l1526_152624


namespace NUMINAMATH_CALUDE_empty_set_cardinality_zero_l1526_152609

theorem empty_set_cardinality_zero : Finset.card (∅ : Finset α) = 0 := by sorry

end NUMINAMATH_CALUDE_empty_set_cardinality_zero_l1526_152609


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_l1526_152656

-- Define the function f(x) = x^3 - x + a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x + a

-- Define the property of being an increasing function
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem not_necessary_not_sufficient :
  ¬(∀ a : ℝ, (a^2 - a = 0 ↔ is_increasing (f a))) :=
sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_l1526_152656


namespace NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l1526_152607

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in the problem -/
def problemCounts : BallCounts :=
  { red := 30, green := 25, yellow := 25, blue := 18, white := 15, black := 12 }

/-- The theorem stating the minimum number of balls to draw -/
theorem min_balls_for_twenty_of_one_color :
    minBallsForColor problemCounts 20 = 103 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l1526_152607


namespace NUMINAMATH_CALUDE_common_tangents_count_l1526_152674

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 4*y + 4 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 4 = 0

-- Define the function to count common tangents
def count_common_tangents (C1 C2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle_C1 circle_C2 = 3 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l1526_152674


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l1526_152694

theorem mixed_number_calculation : 
  72 * ((2 + 3/4) - (3 + 1/2)) / ((3 + 1/3) + (1 + 1/4)) = -(13 + 1/11) := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l1526_152694


namespace NUMINAMATH_CALUDE_tangent_parallel_to_chord_l1526_152628

/-- The curve function -/
def f (x : ℝ) : ℝ := 4*x - x^2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 - 2*x

theorem tangent_parallel_to_chord :
  let A : ℝ × ℝ := (4, 0)
  let B : ℝ × ℝ := (2, 4)
  let P : ℝ × ℝ := (3, 3)
  let chord_slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  P.2 = f P.1 ∧ f' P.1 = chord_slope := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_chord_l1526_152628


namespace NUMINAMATH_CALUDE_pqr_value_l1526_152637

theorem pqr_value (p q r : ℂ) 
  (eq1 : p * q + 5 * q = -20)
  (eq2 : q * r + 5 * r = -20)
  (eq3 : r * p + 5 * p = -20) :
  p * q * r = 80 := by
sorry

end NUMINAMATH_CALUDE_pqr_value_l1526_152637


namespace NUMINAMATH_CALUDE_sixth_point_equals_initial_l1526_152639

/-- Triangle in a plane --/
structure Triangle where
  A₀ : ℝ × ℝ
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ

/-- Symmetric point with respect to a given point --/
def symmetric_point (P : ℝ × ℝ) (A : ℝ × ℝ) : ℝ × ℝ :=
  (2 * A.1 - P.1, 2 * A.2 - P.2)

/-- Generate the next point in the sequence --/
def next_point (P : ℝ × ℝ) (i : ℕ) (T : Triangle) : ℝ × ℝ :=
  match i % 3 with
  | 0 => symmetric_point P T.A₀
  | 1 => symmetric_point P T.A₁
  | _ => symmetric_point P T.A₂

/-- Generate the i-th point in the sequence --/
def P (i : ℕ) (P₀ : ℝ × ℝ) (T : Triangle) : ℝ × ℝ :=
  match i with
  | 0 => P₀
  | n + 1 => next_point (P n P₀ T) (n + 1) T

theorem sixth_point_equals_initial (P₀ : ℝ × ℝ) (T : Triangle) :
  P 6 P₀ T = P₀ := by
  sorry

end NUMINAMATH_CALUDE_sixth_point_equals_initial_l1526_152639


namespace NUMINAMATH_CALUDE_dvaneft_percentage_range_l1526_152605

/-- Represents the share packages in the auction --/
structure SharePackages where
  razneft : ℕ
  dvaneft : ℕ
  trineft : ℕ

/-- Represents the prices of individual shares --/
structure SharePrices where
  razneft : ℝ
  dvaneft : ℝ
  trineft : ℝ

/-- Main theorem about the percentage range of Dvaneft shares --/
theorem dvaneft_percentage_range 
  (packages : SharePackages) 
  (prices : SharePrices) : 
  /- Total shares of Razneft and Dvaneft equals shares of Trineft -/
  (packages.razneft + packages.dvaneft = packages.trineft) → 
  /- Dvaneft package is 3 times cheaper than Razneft package -/
  (3 * prices.dvaneft * packages.dvaneft = prices.razneft * packages.razneft) → 
  /- Total cost of Razneft and Dvaneft equals cost of Trineft -/
  (prices.razneft * packages.razneft + prices.dvaneft * packages.dvaneft = 
   prices.trineft * packages.trineft) → 
  /- Price difference between Razneft and Dvaneft share is between 10,000 and 18,000 -/
  (10000 ≤ prices.razneft - prices.dvaneft ∧ prices.razneft - prices.dvaneft ≤ 18000) → 
  /- Price of Trineft share is between 18,000 and 42,000 -/
  (18000 ≤ prices.trineft ∧ prices.trineft ≤ 42000) → 
  /- The percentage of Dvaneft shares is between 15% and 25% -/
  (15 ≤ 100 * packages.dvaneft / (packages.razneft + packages.dvaneft + packages.trineft) ∧
   100 * packages.dvaneft / (packages.razneft + packages.dvaneft + packages.trineft) ≤ 25) :=
by sorry

end NUMINAMATH_CALUDE_dvaneft_percentage_range_l1526_152605


namespace NUMINAMATH_CALUDE_range_of_a_l1526_152612

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ 
    Real.exp x * (y₁ - x) - a * Real.exp (2 * y₁ - x) = 0 ∧
    Real.exp x * (y₂ - x) - a * Real.exp (2 * y₂ - x) = 0) →
  0 < a ∧ a < 1 / (2 * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1526_152612


namespace NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l1526_152667

theorem negation_of_existence (f : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ f x) ↔ (∀ x : ℝ, x > 0 → ¬ f x) :=
by sorry

theorem cubic_inequality_negation :
  (¬ ∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l1526_152667


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1526_152688

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1526_152688


namespace NUMINAMATH_CALUDE_window_purchase_savings_l1526_152698

/-- Calculates the cost of windows given the quantity and the store's offer -/
def windowCost (quantity : ℕ) : ℕ :=
  let regularPrice := 100
  let freeWindowsPer4 := quantity / 4
  (quantity - freeWindowsPer4) * regularPrice

/-- Calculates the savings when purchasing windows together vs separately -/
def calculateSavings (dave_windows : ℕ) (doug_windows : ℕ) : ℕ :=
  let separate_cost := windowCost dave_windows + windowCost doug_windows
  let joint_cost := windowCost (dave_windows + doug_windows)
  separate_cost - joint_cost

theorem window_purchase_savings :
  calculateSavings 7 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l1526_152698


namespace NUMINAMATH_CALUDE_twelve_bushes_for_sixty_zucchinis_l1526_152626

/-- The number of blueberry bushes needed to obtain a given number of zucchinis -/
def bushes_needed (zucchinis : ℕ) (containers_per_bush : ℕ) (containers_per_trade : ℕ) (zucchinis_per_trade : ℕ) : ℕ :=
  (zucchinis * containers_per_trade) / (zucchinis_per_trade * containers_per_bush)

/-- Theorem: 12 bushes are needed to obtain 60 zucchinis -/
theorem twelve_bushes_for_sixty_zucchinis :
  bushes_needed 60 10 6 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_bushes_for_sixty_zucchinis_l1526_152626


namespace NUMINAMATH_CALUDE_sin_2alpha_l1526_152603

theorem sin_2alpha (α : ℝ) (h : Real.sin (α + π/4) = 1/3) : Real.sin (2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_l1526_152603


namespace NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l1526_152618

theorem x_less_than_2_necessary_not_sufficient :
  (∃ x : ℝ, x < 2 ∧ x^2 - 2*x ≥ 0) ∧
  (∀ x : ℝ, x^2 - 2*x < 0 → x < 2) :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l1526_152618


namespace NUMINAMATH_CALUDE_divisibility_implication_l1526_152646

theorem divisibility_implication (a b : ℕ) (h : a < 1000) :
  (a^21 ∣ b^10) → (a^2 ∣ b) :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1526_152646


namespace NUMINAMATH_CALUDE_hostel_provisions_l1526_152634

/-- Proves that given the initial conditions of a hostel's food provisions,
    the initial number of days the provisions were planned for is 28. -/
theorem hostel_provisions (initial_men : ℕ) (left_men : ℕ) (days_after_leaving : ℕ) 
  (h1 : initial_men = 250)
  (h2 : left_men = 50)
  (h3 : days_after_leaving = 35) :
  (initial_men * ((initial_men - left_men) * days_after_leaving / initial_men) : ℚ) = 
  (initial_men * 28 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_hostel_provisions_l1526_152634


namespace NUMINAMATH_CALUDE_A_in_B_l1526_152671

-- Define the set A
def A : Set ℕ := {0, 1}

-- Define the set B
def B : Set (Set ℕ) := {x | x ⊆ A}

-- Theorem statement
theorem A_in_B : A ∈ B := by sorry

end NUMINAMATH_CALUDE_A_in_B_l1526_152671


namespace NUMINAMATH_CALUDE_intersection_symmetry_l1526_152632

/-- Prove that if a line y = kx intersects a circle (x-1)^2 + y^2 = 1 at two points 
    symmetric with respect to the line x - y + b = 0, then k = -1 and b = -1. -/
theorem intersection_symmetry (k b : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Line equation
    y₁ = k * x₁ ∧ y₂ = k * x₂ ∧
    -- Circle equation
    (x₁ - 1)^2 + y₁^2 = 1 ∧ (x₂ - 1)^2 + y₂^2 = 1 ∧
    -- Symmetry condition
    (x₁ + x₂) / 2 - (y₁ + y₂) / 2 + b = 0) →
  k = -1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_symmetry_l1526_152632


namespace NUMINAMATH_CALUDE_average_annual_growth_rate_l1526_152638

/-- Proves that the average annual growth rate is 20% given the initial and final revenues --/
theorem average_annual_growth_rate 
  (initial_revenue : ℝ) 
  (final_revenue : ℝ) 
  (years : ℕ) 
  (h1 : initial_revenue = 280)
  (h2 : final_revenue = 403.2)
  (h3 : years = 2) :
  ∃ (growth_rate : ℝ), 
    growth_rate = 0.2 ∧ 
    final_revenue = initial_revenue * (1 + growth_rate) ^ years :=
by sorry


end NUMINAMATH_CALUDE_average_annual_growth_rate_l1526_152638


namespace NUMINAMATH_CALUDE_age_problem_l1526_152615

theorem age_problem (a₁ a₂ a₃ a₄ a₅ : ℕ) : 
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ →
  (a₁ + a₂ + a₃) / 3 = 18 →
  a₅ - a₄ = 5 →
  (a₃ + a₄ + a₅) / 3 = 26 →
  a₂ - a₁ = 7 →
  (a₁ + a₅) / 2 = 22 →
  a₁ = 13 ∧ a₂ = 20 ∧ a₃ = 21 ∧ a₄ = 26 ∧ a₅ = 31 :=
by sorry

#check age_problem

end NUMINAMATH_CALUDE_age_problem_l1526_152615


namespace NUMINAMATH_CALUDE_vector_simplification_l1526_152678

/-- Given points A, B, C, and O in 3D space, 
    prove that AB + OC - OB = AC -/
theorem vector_simplification 
  (A B C O : EuclideanSpace ℝ (Fin 3)) : 
  (B - A) + (C - O) - (B - O) = C - A := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l1526_152678


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1526_152631

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane defined by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The specific line l: x + y - 1 = 0 -/
def lineL : Line := { a := 1, b := 1, c := -1 }

/-- The specific point condition x = 2 and y = -1 -/
def specificPoint : Point := { x := 2, y := -1 }

theorem sufficient_not_necessary_condition :
  (∀ p : Point, p = specificPoint → isOnLine p lineL) ∧
  ¬(∀ p : Point, isOnLine p lineL → p = specificPoint) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1526_152631


namespace NUMINAMATH_CALUDE_sin_15_cos_15_value_l1526_152663

theorem sin_15_cos_15_value : (1/4) * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_value_l1526_152663


namespace NUMINAMATH_CALUDE_units_digit_of_power_plus_six_l1526_152642

theorem units_digit_of_power_plus_six (y : ℕ+) :
  (7^y.val + 6) % 10 = 9 ↔ y.val % 4 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_power_plus_six_l1526_152642


namespace NUMINAMATH_CALUDE_liza_age_is_14_liza_older_than_nastya_liza_triple_nastya_two_years_ago_l1526_152683

/-- The age difference between Liza and Nastya -/
def age_difference : ℕ := 8

/-- Liza's current age -/
def liza_age : ℕ := 14

/-- Nastya's current age -/
def nastya_age : ℕ := liza_age - age_difference

theorem liza_age_is_14 : liza_age = 14 := by sorry

theorem liza_older_than_nastya : liza_age = nastya_age + age_difference := by sorry

theorem liza_triple_nastya_two_years_ago : 
  liza_age - 2 = 3 * (nastya_age - 2) := by sorry

end NUMINAMATH_CALUDE_liza_age_is_14_liza_older_than_nastya_liza_triple_nastya_two_years_ago_l1526_152683


namespace NUMINAMATH_CALUDE_data_transmission_time_l1526_152668

theorem data_transmission_time : 
  let num_blocks : ℕ := 60
  let chunks_per_block : ℕ := 512
  let transmission_rate : ℕ := 120  -- chunks per second
  let total_chunks : ℕ := num_blocks * chunks_per_block
  let transmission_time_seconds : ℕ := total_chunks / transmission_rate
  let transmission_time_minutes : ℕ := transmission_time_seconds / 60
  transmission_time_minutes = 4 := by
  sorry

end NUMINAMATH_CALUDE_data_transmission_time_l1526_152668


namespace NUMINAMATH_CALUDE_coffee_package_size_l1526_152619

theorem coffee_package_size (total_coffee : ℕ) (large_package_size : ℕ) (large_package_count : ℕ) (small_package_count_diff : ℕ) :
  total_coffee = 115 →
  large_package_size = 10 →
  large_package_count = 7 →
  small_package_count_diff = 2 →
  ∃ (small_package_size : ℕ),
    small_package_size = 5 ∧
    total_coffee = (large_package_size * large_package_count) + (small_package_size * (large_package_count + small_package_count_diff)) :=
by sorry

end NUMINAMATH_CALUDE_coffee_package_size_l1526_152619


namespace NUMINAMATH_CALUDE_initial_candies_l1526_152633

theorem initial_candies : ∃ x : ℕ, (x - 29) / 13 = 15 ∧ x = 224 := by
  sorry

end NUMINAMATH_CALUDE_initial_candies_l1526_152633


namespace NUMINAMATH_CALUDE_right_triangle_circles_radius_l1526_152689

/-- Represents a right triangle with a circle tangent to one side and another circle --/
structure RightTriangleWithCircles where
  -- The length of side AC
  ac : ℝ
  -- The length of side AB
  ab : ℝ
  -- The radius of circle C
  rc : ℝ
  -- The radius of circle A
  ra : ℝ
  -- Circle C is tangent to AB
  c_tangent_ab : True
  -- Circle A and circle C are tangent
  a_tangent_c : True
  -- Angle C is 90 degrees
  angle_c_90 : True

/-- The main theorem --/
theorem right_triangle_circles_radius 
  (t : RightTriangleWithCircles) 
  (h1 : t.ac = 6) 
  (h2 : t.ab = 10) : 
  t.ra = 1.2 ∨ t.ra = 10.8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circles_radius_l1526_152689


namespace NUMINAMATH_CALUDE_chicken_feathers_after_crossing_l1526_152684

/-- Represents the number of feathers a chicken has after crossing a road twice -/
def feathers_after_crossing (initial_feathers : ℕ) (cars_dodged : ℕ) : ℕ :=
  initial_feathers - 2 * cars_dodged

/-- Theorem stating the number of feathers remaining after the chicken's adventure -/
theorem chicken_feathers_after_crossing :
  feathers_after_crossing 5263 23 = 5217 := by
  sorry

#eval feathers_after_crossing 5263 23

end NUMINAMATH_CALUDE_chicken_feathers_after_crossing_l1526_152684


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l1526_152645

theorem cube_sum_of_roots (p q r : ℝ) : 
  (p^3 - p^2 + p - 2 = 0) → 
  (q^3 - q^2 + q - 2 = 0) → 
  (r^3 - r^2 + r - 2 = 0) → 
  p^3 + q^3 + r^3 = 4 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l1526_152645


namespace NUMINAMATH_CALUDE_cyclist_distance_l1526_152652

/-- The distance traveled by a cyclist moving between two people walking towards each other -/
theorem cyclist_distance (distance : ℝ) (speed_vasya : ℝ) (speed_roma : ℝ) 
  (h1 : distance > 0)
  (h2 : speed_vasya > 0)
  (h3 : speed_roma > 0) :
  let speed_dima := speed_vasya + speed_roma
  let time := distance / (speed_vasya + speed_roma)
  speed_dima * time = distance :=
by sorry

end NUMINAMATH_CALUDE_cyclist_distance_l1526_152652


namespace NUMINAMATH_CALUDE_total_distance_ran_l1526_152601

/-- The length of a football field in meters -/
def football_field_length : ℕ := 168

/-- The distance Nate ran in the first part, in terms of football field lengths -/
def initial_distance_in_fields : ℕ := 4

/-- The additional distance Nate ran in meters -/
def additional_distance : ℕ := 500

/-- Theorem: The total distance Nate ran is 1172 meters -/
theorem total_distance_ran : 
  football_field_length * initial_distance_in_fields + additional_distance = 1172 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_ran_l1526_152601


namespace NUMINAMATH_CALUDE_apple_distribution_l1526_152617

theorem apple_distribution (total_apples : ℕ) (new_people : ℕ) (apple_decrease : ℕ) :
  total_apples = 1430 →
  new_people = 45 →
  apple_decrease = 9 →
  ∃ (original_people : ℕ),
    original_people > 0 ∧
    (total_apples / original_people : ℚ) - (total_apples / (original_people + new_people) : ℚ) = apple_decrease ∧
    total_apples / original_people = 22 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l1526_152617
