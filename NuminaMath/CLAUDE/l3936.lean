import Mathlib

namespace NUMINAMATH_CALUDE_three_heads_probability_l3936_393665

/-- The probability of getting heads on a single flip of a fair coin -/
def prob_heads : ℚ := 1/2

/-- The probability of getting three heads in a row when flipping a fair coin -/
def prob_three_heads : ℚ := prob_heads * prob_heads * prob_heads

theorem three_heads_probability : prob_three_heads = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_probability_l3936_393665


namespace NUMINAMATH_CALUDE_apollonius_circle_l3936_393694

/-- The Apollonius Circle Theorem -/
theorem apollonius_circle (x y : ℝ) : 
  let A : ℝ × ℝ := (2, 0)
  let B : ℝ × ℝ := (8, 0)
  let P : ℝ × ℝ := (x, y)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist P A / dist P B = 1/2 → x^2 + y^2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_apollonius_circle_l3936_393694


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l3936_393604

/-- A point on a 2D grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A set of black cells on the grid --/
def BlackCells := Set GridPoint

/-- A line on the grid (vertical, horizontal, or diagonal) --/
inductive GridLine
  | Vertical (x : ℤ)
  | Horizontal (y : ℤ)
  | Diagonal (m : ℤ) (b : ℤ)

/-- The number of black cells on a given line --/
def blackCellsOnLine (cells : BlackCells) (line : GridLine) : ℕ :=
  sorry

/-- The property that a set of black cells satisfies the k-cell condition --/
def satisfiesKCellCondition (cells : BlackCells) (k : ℕ) : Prop :=
  ∀ line : GridLine, blackCellsOnLine cells line = k ∨ blackCellsOnLine cells line = 0

theorem exists_valid_coloring (k : ℕ) : 
  ∃ (cells : BlackCells), cells.Nonempty ∧ Set.Finite cells ∧ satisfiesKCellCondition cells k :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l3936_393604


namespace NUMINAMATH_CALUDE_remainder_123456789012_mod_252_l3936_393607

theorem remainder_123456789012_mod_252 : 123456789012 % 252 = 108 := by
  sorry

end NUMINAMATH_CALUDE_remainder_123456789012_mod_252_l3936_393607


namespace NUMINAMATH_CALUDE_alice_exam_score_l3936_393633

theorem alice_exam_score (exam1 exam2 exam3 : ℕ) 
  (h1 : exam1 = 85) (h2 : exam2 = 76) (h3 : exam3 = 83)
  (h4 : ∀ exam, exam ≤ 100) : 
  ∃ (exam4 exam5 : ℕ), 
    exam4 ≤ 100 ∧ exam5 ≤ 100 ∧ 
    (exam1 + exam2 + exam3 + exam4 + exam5) / 5 = 80 ∧
    (exam4 = 56 ∨ exam5 = 56) ∧
    ∀ (x : ℕ), x < 56 → 
      ¬∃ (y : ℕ), y ≤ 100 ∧ (exam1 + exam2 + exam3 + x + y) / 5 = 80 :=
by sorry

end NUMINAMATH_CALUDE_alice_exam_score_l3936_393633


namespace NUMINAMATH_CALUDE_certain_number_exists_l3936_393689

theorem certain_number_exists : ∃ (n : ℕ), n > 0 ∧ 49 % n = 4 ∧ 66 % n = 6 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l3936_393689


namespace NUMINAMATH_CALUDE_inequality_reversal_l3936_393662

theorem inequality_reversal (a b c : ℝ) (h1 : a < b) (h2 : c < 0) : ¬(a * c < b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l3936_393662


namespace NUMINAMATH_CALUDE_square_side_length_l3936_393613

theorem square_side_length (perimeter : ℝ) (h : perimeter = 17.8) :
  let side_length := perimeter / 4
  side_length = 4.45 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3936_393613


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3936_393634

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, (a - b) * a^2 < 0 → a < b) ∧
  ¬(∀ a b, a < b → (a - b) * a^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3936_393634


namespace NUMINAMATH_CALUDE_function_symmetry_about_origin_l3936_393690

/-- The function f(x) = x^5 + x^3 is odd, implying symmetry about the origin -/
theorem function_symmetry_about_origin (x : ℝ) : 
  ((-x)^5 + (-x)^3) = -(x^5 + x^3) := by sorry

end NUMINAMATH_CALUDE_function_symmetry_about_origin_l3936_393690


namespace NUMINAMATH_CALUDE_sin_seven_halves_pi_plus_theta_l3936_393677

theorem sin_seven_halves_pi_plus_theta (θ : Real) 
  (h : Real.cos (3 * Real.pi + θ) = -(2 * Real.sqrt 2) / 3) : 
  Real.sin ((7 / 2) * Real.pi + θ) = -(2 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_halves_pi_plus_theta_l3936_393677


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l3936_393630

-- Define the sets A, B, and M
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}
def M (k : ℝ) : Set ℝ := {x | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

-- Theorem statement
theorem set_operations_and_subset :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) ∧
  (∀ k, M k ⊆ A ↔ k < -5/2 ∨ k > 1) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l3936_393630


namespace NUMINAMATH_CALUDE_line_condition_perpendicular_to_x_axis_equal_intercepts_l3936_393600

-- Define the equation coefficients as functions of m
def a (m : ℝ) := m^2 - 2*m - 3
def b (m : ℝ) := 2*m^2 + m - 1
def c (m : ℝ) := 5 - 2*m

-- Theorem 1: Condition for the equation to represent a line
theorem line_condition (m : ℝ) : 
  (a m = 0 ∧ b m = 0) ↔ m = -1 :=
sorry

-- Theorem 2: Condition for the line to be perpendicular to x-axis
theorem perpendicular_to_x_axis (m : ℝ) :
  (a m ≠ 0 ∧ b m = 0) ↔ (m^2 - 2*m - 3 ≠ 0 ∧ 2*m^2 + m - 1 = 0) :=
sorry

-- Theorem 3: Condition for equal intercepts on both axes
theorem equal_intercepts (m : ℝ) :
  (m ≠ 5/2 → (2*m - 5)/(m^2 - 2*m - 3) = (2*m - 5)/(2*m^2 + m - 1)) ↔ m = 5/2 :=
sorry

end NUMINAMATH_CALUDE_line_condition_perpendicular_to_x_axis_equal_intercepts_l3936_393600


namespace NUMINAMATH_CALUDE_opposite_angles_equal_l3936_393698

/-- Two angles are opposite if they are formed by two intersecting lines and are not adjacent. -/
def are_opposite_angles (α β : Real) : Prop := sorry

/-- The measure of an angle in radians. -/
def angle_measure (α : Real) : ℝ := sorry

theorem opposite_angles_equal (α β : Real) :
  are_opposite_angles α β → angle_measure α = angle_measure β := by sorry

end NUMINAMATH_CALUDE_opposite_angles_equal_l3936_393698


namespace NUMINAMATH_CALUDE_probability_of_listening_second_class_l3936_393651

/-- Represents the duration of a class in minutes -/
def class_duration : ℕ := 40

/-- Represents the duration of a break between classes in minutes -/
def break_duration : ℕ := 10

/-- Represents the start time of the first class in minutes after midnight -/
def first_class_start : ℕ := 8 * 60

/-- Represents the earliest arrival time of the student in minutes after midnight -/
def earliest_arrival : ℕ := 9 * 60 + 10

/-- Represents the latest arrival time of the student in minutes after midnight -/
def latest_arrival : ℕ := 10 * 60

/-- Represents the duration of the arrival window in minutes -/
def arrival_window : ℕ := latest_arrival - earliest_arrival

/-- Represents the duration of the favorable arrival window in minutes -/
def favorable_window : ℕ := 10

/-- The probability of the student listening to the second class for no less than 10 minutes -/
theorem probability_of_listening_second_class :
  (favorable_window : ℚ) / arrival_window = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_listening_second_class_l3936_393651


namespace NUMINAMATH_CALUDE_right_triangle_min_std_dev_l3936_393615

theorem right_triangle_min_std_dev (a b c : ℝ) : 
  a > 0 → b > 0 → c = 3 → a^2 + b^2 = c^2 →
  let s := Real.sqrt ((a^2 + b^2 + c^2) / 3 - ((a + b + c) / 3)^2)
  s ≥ Real.sqrt 2 - 1 ∧ 
  (s = Real.sqrt 2 - 1 ↔ a = 3 * Real.sqrt 2 / 2 ∧ b = 3 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_min_std_dev_l3936_393615


namespace NUMINAMATH_CALUDE_square_sum_given_squared_sum_and_product_l3936_393679

theorem square_sum_given_squared_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 49) 
  (h2 : x * y = 10) : 
  x^2 + y^2 = 29 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_squared_sum_and_product_l3936_393679


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3936_393680

/-- The coefficient of x^(3/2) in the expansion of (√x - a/√x)^5 -/
def coefficient_x_3_2 (a : ℝ) : ℝ := 
  (5 : ℝ) * (-a)

theorem expansion_coefficient (a : ℝ) : 
  coefficient_x_3_2 a = 30 → a = -6 := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3936_393680


namespace NUMINAMATH_CALUDE_broadcast_orders_count_l3936_393692

/-- The number of ways to arrange 6 commercial ads and 2 public service ads 
    with specific constraints -/
def broadcast_orders : ℕ :=
  let n_commercials : ℕ := 6
  let n_public_service : ℕ := 2
  let n_spaces : ℕ := n_commercials - 1
  let ways_to_place_public_service : ℕ := n_spaces * (n_spaces - 2)
  Nat.factorial n_commercials * ways_to_place_public_service

/-- Theorem stating the number of different broadcast orders -/
theorem broadcast_orders_count :
  broadcast_orders = 10800 := by
  sorry

end NUMINAMATH_CALUDE_broadcast_orders_count_l3936_393692


namespace NUMINAMATH_CALUDE_max_value_of_f_l3936_393603

noncomputable def f (x : ℝ) : ℝ := 2 * (-1) * Real.log x - 1 / x

theorem max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x ∧ f x = 2 * Real.log 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3936_393603


namespace NUMINAMATH_CALUDE_grandmother_rolls_l3936_393611

def total_rolls : ℕ := 12
def uncle_rolls : ℕ := 4
def neighbor_rolls : ℕ := 3
def remaining_rolls : ℕ := 2

theorem grandmother_rolls : 
  total_rolls - (uncle_rolls + neighbor_rolls + remaining_rolls) = 3 := by
  sorry

end NUMINAMATH_CALUDE_grandmother_rolls_l3936_393611


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l3936_393649

theorem middle_part_of_proportional_division (total : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 ∧ a = 2 ∧ b = (1 : ℝ) / 2 ∧ c = (1 : ℝ) / 4 →
  ∃ x : ℝ, a * x + b * x + c * x = total ∧ b * x = 20.8 :=
by sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l3936_393649


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3936_393623

theorem quadratic_equation_solutions : {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3936_393623


namespace NUMINAMATH_CALUDE_picture_area_l3936_393650

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h_frame_area : (2 * x + 5) * (y + 4) = 60) : x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l3936_393650


namespace NUMINAMATH_CALUDE_midpoint_area_in_square_l3936_393697

/-- The area enclosed by midpoints of line segments in a square --/
theorem midpoint_area_in_square (s : ℝ) (h : s = 3) : 
  let midpoint_area := s^2 - (s^2 * Real.pi) / 4
  midpoint_area = 9 - (9 * Real.pi) / 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_area_in_square_l3936_393697


namespace NUMINAMATH_CALUDE_halfway_fraction_l3936_393643

theorem halfway_fraction : (3 : ℚ) / 4 + ((5 : ℚ) / 6 - (3 : ℚ) / 4) / 2 = (19 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3936_393643


namespace NUMINAMATH_CALUDE_courier_package_ratio_l3936_393653

theorem courier_package_ratio : 
  ∀ (total_packages yesterday_packages today_packages : ℕ),
    total_packages = 240 →
    yesterday_packages = 80 →
    total_packages = yesterday_packages + today_packages →
    (today_packages : ℚ) / (yesterday_packages : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_courier_package_ratio_l3936_393653


namespace NUMINAMATH_CALUDE_skew_quadrilateral_angle_sum_less_than_360_l3936_393636

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The angle between three points in 3D space -/
noncomputable def angle (A B C : Point3D) : ℝ := sorry

/-- Four points are non-coplanar if they do not lie in the same plane -/
def nonCoplanar (A B C D : Point3D) : Prop := sorry

/-- A skew quadrilateral is formed by four non-coplanar points -/
structure SkewQuadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  nonCoplanar : nonCoplanar A B C D

theorem skew_quadrilateral_angle_sum_less_than_360 (quad : SkewQuadrilateral) :
  angle quad.A quad.B quad.C + angle quad.B quad.C quad.D +
  angle quad.C quad.D quad.A + angle quad.D quad.A quad.B < 2 * π :=
sorry

end NUMINAMATH_CALUDE_skew_quadrilateral_angle_sum_less_than_360_l3936_393636


namespace NUMINAMATH_CALUDE_benjamins_house_paintable_area_l3936_393673

/-- Represents the dimensions of a room --/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total paintable area in Benjamin's house --/
def total_paintable_area (
  num_bedrooms : ℕ
  ) (room_dims : RoomDimensions)
  (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (room_dims.length * room_dims.height + room_dims.width * room_dims.height)
  let paintable_area_per_room := wall_area - unpaintable_area
  num_bedrooms * paintable_area_per_room

/-- Theorem stating the total paintable area in Benjamin's house --/
theorem benjamins_house_paintable_area :
  total_paintable_area 4 ⟨14, 12, 9⟩ 70 = 1592 := by
  sorry

end NUMINAMATH_CALUDE_benjamins_house_paintable_area_l3936_393673


namespace NUMINAMATH_CALUDE_fraction_well_defined_l3936_393617

theorem fraction_well_defined (x : ℝ) (h : x ≠ 2) : 2 * x - 4 ≠ 0 := by
  sorry

#check fraction_well_defined

end NUMINAMATH_CALUDE_fraction_well_defined_l3936_393617


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3936_393640

/-- The number of games in a chess tournament -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- The theorem stating the number of games in the specific tournament -/
theorem chess_tournament_games :
  tournament_games 19 * 2 = 684 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3936_393640


namespace NUMINAMATH_CALUDE_polygon_sides_l3936_393661

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3936_393661


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l3936_393629

def a (m : ℝ) : Fin 2 → ℝ := ![2*m, 3]
def b (m : ℝ) : Fin 2 → ℝ := ![m-1, 1]

theorem collinear_vectors_m_value (m : ℝ) :
  (∃ (k : ℝ), a m = k • b m) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l3936_393629


namespace NUMINAMATH_CALUDE_double_plus_five_positive_l3936_393699

theorem double_plus_five_positive (m : ℝ) :
  (2 * m + 5 > 0) ↔ (∃ x > 0, x = 2 * m + 5) :=
by sorry

end NUMINAMATH_CALUDE_double_plus_five_positive_l3936_393699


namespace NUMINAMATH_CALUDE_remaining_note_denomination_l3936_393608

theorem remaining_note_denomination 
  (total_amount : ℕ) 
  (total_notes : ℕ) 
  (fifty_notes : ℕ) 
  (h1 : total_amount = 10350)
  (h2 : total_notes = 36)
  (h3 : fifty_notes = 17) :
  (total_amount - 50 * fifty_notes) / (total_notes - fifty_notes) = 500 := by
  sorry

end NUMINAMATH_CALUDE_remaining_note_denomination_l3936_393608


namespace NUMINAMATH_CALUDE_root_implies_b_value_l3936_393601

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 15

-- State the theorem
theorem root_implies_b_value (a : ℚ) :
  (∃ b : ℚ, f a b (3 + Real.sqrt 5) = 0) →
  (∃ b : ℚ, b = -37/2) :=
by sorry

end NUMINAMATH_CALUDE_root_implies_b_value_l3936_393601


namespace NUMINAMATH_CALUDE_min_abs_sum_of_quadratic_roots_l3936_393678

theorem min_abs_sum_of_quadratic_roots : ∃ (α β : ℝ), 
  (∀ y : ℝ, y^2 - 6*y + 5 = 0 ↔ y = α ∨ y = β) ∧
  (∀ x : ℝ, |x - α| + |x - β| ≥ 4) ∧
  (∃ x : ℝ, |x - α| + |x - β| = 4) := by
sorry

end NUMINAMATH_CALUDE_min_abs_sum_of_quadratic_roots_l3936_393678


namespace NUMINAMATH_CALUDE_registration_scientific_notation_equality_l3936_393668

/-- The number of people registered for the national college entrance examination in 2023 -/
def registration_number : ℕ := 12910000

/-- The scientific notation representation of the registration number -/
def scientific_notation : ℝ := 1.291 * (10 ^ 7)

/-- Theorem stating that the registration number is equal to its scientific notation representation -/
theorem registration_scientific_notation_equality :
  (registration_number : ℝ) = scientific_notation :=
sorry

end NUMINAMATH_CALUDE_registration_scientific_notation_equality_l3936_393668


namespace NUMINAMATH_CALUDE_no_roots_geq_two_l3936_393619

theorem no_roots_geq_two : ∀ x : ℝ, x ≥ 2 → 4 * x^3 - 5 * x^2 - 6 * x + 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_roots_geq_two_l3936_393619


namespace NUMINAMATH_CALUDE_fabric_width_l3936_393696

/-- Given a rectangular piece of fabric with area 24 square centimeters and length 8 centimeters,
    prove that its width is 3 centimeters. -/
theorem fabric_width (area : ℝ) (length : ℝ) (width : ℝ) 
    (h1 : area = 24) 
    (h2 : length = 8) 
    (h3 : area = length * width) : width = 3 := by
  sorry

end NUMINAMATH_CALUDE_fabric_width_l3936_393696


namespace NUMINAMATH_CALUDE_stream_speed_l3936_393642

/-- Proves that the speed of the stream is 8 kmph given the conditions of the problem -/
theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 24 →
  (1 / (boat_speed - stream_speed)) = (2 / (boat_speed + stream_speed)) →
  stream_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3936_393642


namespace NUMINAMATH_CALUDE_stating_calculate_total_applicants_l3936_393655

/-- Represents the proportion of students who applied to first-tier colleges in a sample -/
def sample_proportion (sample_size : ℕ) (applicants_in_sample : ℕ) : ℚ :=
  applicants_in_sample / sample_size

/-- Represents the proportion of students who applied to first-tier colleges in the population -/
def population_proportion (population_size : ℕ) (total_applicants : ℕ) : ℚ :=
  total_applicants / population_size

/-- 
Theorem stating that if the sample proportion equals the population proportion,
then the total number of applicants in the population can be calculated.
-/
theorem calculate_total_applicants 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (applicants_in_sample : ℕ) 
  (h1 : population_size = 1000)
  (h2 : sample_size = 150)
  (h3 : applicants_in_sample = 60) :
  ∃ (total_applicants : ℕ),
    sample_proportion sample_size applicants_in_sample = 
    population_proportion population_size total_applicants ∧ 
    total_applicants = 400 := by
  sorry

end NUMINAMATH_CALUDE_stating_calculate_total_applicants_l3936_393655


namespace NUMINAMATH_CALUDE_inequality_proof_l3936_393683

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 + c^2 ≥ a*b + b*c + c*a) ∧ ((a + b + c)^2 ≥ 3*(a*b + b*c + c*a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3936_393683


namespace NUMINAMATH_CALUDE_circle_on_parabola_passes_through_focus_l3936_393605

/-- A circle with center on a parabola y^2 = 4x and tangent to x = -1 passes through (1, 0) -/
theorem circle_on_parabola_passes_through_focus (C : ℝ × ℝ) (r : ℝ) :
  C.2^2 = 4 * C.1 →  -- Center C is on the parabola y^2 = 4x
  abs (C.1 + 1) = r →  -- Circle is tangent to x = -1
  (1 - C.1)^2 + C.2^2 = r^2  -- Circle passes through (1, 0)
  := by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_passes_through_focus_l3936_393605


namespace NUMINAMATH_CALUDE_simple_interest_difference_l3936_393606

/-- Simple interest calculation and comparison --/
theorem simple_interest_difference (principal rate time : ℕ) : 
  principal = 3000 → 
  rate = 4 → 
  time = 5 → 
  principal - (principal * rate * time) / 100 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_difference_l3936_393606


namespace NUMINAMATH_CALUDE_complex_fraction_squared_l3936_393618

theorem complex_fraction_squared (i : ℂ) : i * i = -1 → ((1 - i) / (1 + i))^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_squared_l3936_393618


namespace NUMINAMATH_CALUDE_correct_purchase_ways_l3936_393658

/-- The number of oreo flavors available -/
def num_oreo_flavors : ℕ := 6

/-- The number of milk flavors available -/
def num_milk_flavors : ℕ := 4

/-- The total number of products they purchase collectively -/
def total_products : ℕ := 3

/-- Function to calculate the number of ways Alpha and Beta can purchase products -/
def purchase_ways : ℕ := sorry

/-- Theorem stating the correct number of ways to purchase products -/
theorem correct_purchase_ways : purchase_ways = 656 := by sorry

end NUMINAMATH_CALUDE_correct_purchase_ways_l3936_393658


namespace NUMINAMATH_CALUDE_geometry_statements_l3936_393686

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)

variable (m n : Line)
variable (α β : Plane)

-- Assume m and n are distinct, α and β are different
variable (h_distinct_lines : m ≠ n)
variable (h_different_planes : α ≠ β)

theorem geometry_statements :
  (parallel_line_plane m α ∧ perpendicular_line_plane n β ∧ parallel_lines m n → perpendicular_planes α β) ∧
  (perpendicular_line_plane m α ∧ parallel_lines m n → perpendicular_line_plane n α) ∧
  ¬(perpendicular_lines m n ∧ line_in_plane n α ∧ line_in_plane m β → perpendicular_planes α β) ∧
  (parallel_line_plane m β ∧ line_in_plane m α ∧ plane_intersection α β = n → parallel_lines m n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_statements_l3936_393686


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l3936_393695

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l3936_393695


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3936_393610

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x : ℝ | x^2 - p*x + 15 = 0}
def B (q : ℝ) : Set ℝ := {x : ℝ | x^2 - 5*x + q = 0}

-- State the theorem
theorem intersection_implies_sum (p q : ℝ) : A p ∩ B q = {3} → p + q = 14 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3936_393610


namespace NUMINAMATH_CALUDE_smallest_max_sum_l3936_393624

theorem smallest_max_sum (p q r s t : ℕ+) 
  (sum_condition : p + q + r + s + t = 4020) : 
  (∃ (N : ℕ), 
    N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ 
    (∀ (M : ℕ), M = max (p + q) (max (q + r) (max (r + s) (s + t))) → N ≤ M) ∧
    N = 1005) := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l3936_393624


namespace NUMINAMATH_CALUDE_total_eggs_l3936_393674

theorem total_eggs (num_students : ℕ) (eggs_per_student : ℕ) (h1 : num_students = 7) (h2 : eggs_per_student = 8) :
  num_students * eggs_per_student = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_l3936_393674


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3936_393675

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : x + |y| - y = 12) : 
  x + y = 18/5 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3936_393675


namespace NUMINAMATH_CALUDE_basket_capacity_l3936_393663

/-- The number of placards taken by each person -/
def placards_per_person : ℕ := 2

/-- The number of people who entered the stadium -/
def people_entered : ℕ := 2317

/-- The total number of placards taken -/
def total_placards : ℕ := people_entered * placards_per_person

theorem basket_capacity : total_placards = 4634 := by
  sorry

end NUMINAMATH_CALUDE_basket_capacity_l3936_393663


namespace NUMINAMATH_CALUDE_least_common_solution_l3936_393647

theorem least_common_solution : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 8 = 7 ∧ 
  x % 7 = 6 ∧
  (∀ y : ℕ, y > 0 ∧ y % 6 = 5 ∧ y % 8 = 7 ∧ y % 7 = 6 → x ≤ y) ∧
  x = 167 := by
sorry

end NUMINAMATH_CALUDE_least_common_solution_l3936_393647


namespace NUMINAMATH_CALUDE_at_least_two_primes_in_sequence_l3936_393656

theorem at_least_two_primes_in_sequence : ∃ (m n : ℕ), 
  2 ≤ m ∧ 2 ≤ n ∧ m ≠ n ∧ 
  Nat.Prime (m^3 + m + 1) ∧ 
  Nat.Prime (n^3 + n + 1) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_primes_in_sequence_l3936_393656


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3936_393671

theorem sqrt_equation_solution :
  ∃ x : ℝ, 3 * Real.sqrt (x + 15) = 36 ∧ x = 129 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3936_393671


namespace NUMINAMATH_CALUDE_fraction_calculation_l3936_393682

theorem fraction_calculation : (2 / 3 * 4 / 7 * 5 / 8) + 1 / 6 = 17 / 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3936_393682


namespace NUMINAMATH_CALUDE_inequality_proof_l3936_393676

open Real

theorem inequality_proof (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x > 0, f x = Real.log x - 3 * x) →
  (∀ x > 0, f x ≤ x * (a * Real.exp x - 4) + b) →
  a + b ≥ 0 := by
    sorry

end NUMINAMATH_CALUDE_inequality_proof_l3936_393676


namespace NUMINAMATH_CALUDE_pizza_flour_calculation_l3936_393625

theorem pizza_flour_calculation (bases : ℕ) (total_flour : ℚ) : 
  bases = 15 → total_flour = 8 → (total_flour / bases : ℚ) = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_pizza_flour_calculation_l3936_393625


namespace NUMINAMATH_CALUDE_instant_noodle_price_reduction_l3936_393616

theorem instant_noodle_price_reduction 
  (original_weight : ℝ) 
  (original_price : ℝ) 
  (weight_increase_percentage : ℝ) 
  (h1 : weight_increase_percentage = 0.25) 
  (h2 : original_weight > 0) 
  (h3 : original_price > 0) : 
  let new_weight := original_weight * (1 + weight_increase_percentage)
  let original_price_per_unit := original_price / original_weight
  let new_price_per_unit := original_price / new_weight
  (original_price_per_unit - new_price_per_unit) / original_price_per_unit = 0.2
  := by sorry

end NUMINAMATH_CALUDE_instant_noodle_price_reduction_l3936_393616


namespace NUMINAMATH_CALUDE_expression_evaluation_l3936_393687

theorem expression_evaluation :
  let a : ℚ := 1/3
  let b : ℤ := -1
  4 * (3 * a^2 * b - a * b^2) - (2 * a * b^2 + 3 * a^2 * b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3936_393687


namespace NUMINAMATH_CALUDE_ellipse_properties_l3936_393670

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the line
def line (m x y : ℝ) : Prop := y = m * (x - 1)

-- Define the intersection points
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line m p.1 p.2}

-- Theorem statement
theorem ellipse_properties :
  -- Part 1: Standard equation of the ellipse
  (∀ x y : ℝ, ellipse x y ↔ x^2 / 3 + y^2 / 2 = 1) ∧
  -- Part 2: Line intersects ellipse at two distinct points
  (∀ m : ℝ, ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ A ≠ B) ∧
  -- Part 3: No real m exists such that the circle with diameter AB passes through origin
  ¬(∃ m : ℝ, ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧
    A.1 * B.1 + A.2 * B.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3936_393670


namespace NUMINAMATH_CALUDE_min_sum_distances_l3936_393631

-- Define a rectangle in 2D space
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_rectangle : sorry -- Condition ensuring ABCD forms a rectangle

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the center of a rectangle
def center (r : Rectangle) : ℝ × ℝ := sorry

-- Define the sum of distances from a point to the corners
def sum_distances (r : Rectangle) (p : ℝ × ℝ) : ℝ :=
  distance p r.A + distance p r.B + distance p r.C + distance p r.D

-- Theorem statement
theorem min_sum_distances (r : Rectangle) :
  ∀ p : ℝ × ℝ, sum_distances r (center r) ≤ sum_distances r p :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_l3936_393631


namespace NUMINAMATH_CALUDE_ultra_high_yield_interest_l3936_393659

/-- The compound interest formula -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The interest earned from an investment -/
def interest_earned (principal : ℝ) (final_amount : ℝ) : ℝ :=
  final_amount - principal

/-- Theorem: The interest earned on a 500-dollar investment compounded annually at 3% for 10 years is approximately 172 dollars -/
theorem ultra_high_yield_interest :
  let principal : ℝ := 500
  let rate : ℝ := 0.03
  let years : ℕ := 10
  let final_amount := compound_interest principal rate years
  let earned := interest_earned principal final_amount
  ∃ ε > 0, |earned - 172| < ε :=
by sorry

end NUMINAMATH_CALUDE_ultra_high_yield_interest_l3936_393659


namespace NUMINAMATH_CALUDE_cosine_difference_equals_negative_seven_thousandths_l3936_393645

theorem cosine_difference_equals_negative_seven_thousandths :
  let α := Real.arcsin (3/5)
  let β := Real.arcsin (4/5)
  (Real.cos (3*Real.pi/2 - α/2))^6 - (Real.cos (5*Real.pi/2 + β/2))^6 = -7/1000 := by
sorry

end NUMINAMATH_CALUDE_cosine_difference_equals_negative_seven_thousandths_l3936_393645


namespace NUMINAMATH_CALUDE_intensity_after_three_plates_l3936_393688

/-- The intensity of light after passing through a number of glass plates -/
def intensity (a : ℝ) (n : ℕ) : ℝ :=
  a * (0.9 ^ n)

/-- Theorem: The intensity of light with original intensity a after passing through 3 glass plates is 0.729a -/
theorem intensity_after_three_plates (a : ℝ) :
  intensity a 3 = 0.729 * a := by
  sorry

end NUMINAMATH_CALUDE_intensity_after_three_plates_l3936_393688


namespace NUMINAMATH_CALUDE_carrie_turnip_mixture_l3936_393632

-- Define the ratio of potatoes to turnips
def potatoTurnipRatio : ℚ := 5 / 2

-- Define the total amount of potatoes
def totalPotatoes : ℚ := 20

-- Define the amount of turnips that can be added
def turnipsToAdd : ℚ := totalPotatoes / potatoTurnipRatio

-- Theorem statement
theorem carrie_turnip_mixture :
  turnipsToAdd = 8 := by sorry

end NUMINAMATH_CALUDE_carrie_turnip_mixture_l3936_393632


namespace NUMINAMATH_CALUDE_ravi_work_time_l3936_393672

-- Define the work completion times
def prakash_time : ℝ := 75
def combined_time : ℝ := 30

-- Define Ravi's time as a variable
def ravi_time : ℝ := 50

-- Theorem statement
theorem ravi_work_time :
  (1 / ravi_time + 1 / prakash_time = 1 / combined_time) →
  ravi_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_ravi_work_time_l3936_393672


namespace NUMINAMATH_CALUDE_lucas_addition_example_l3936_393641

/-- Lucas's notation for integers -/
def lucas_notation (n : ℤ) : ℕ :=
  if n ≥ 0 then n.natAbs else n.natAbs + 1

/-- Addition in Lucas's notation -/
def lucas_add (a b : ℕ) : ℕ :=
  lucas_notation (-(a : ℤ) + -(b : ℤ))

/-- Theorem: 000 + 0000 = 000000 in Lucas's notation -/
theorem lucas_addition_example : lucas_add 3 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_lucas_addition_example_l3936_393641


namespace NUMINAMATH_CALUDE_line_configuration_theorem_l3936_393648

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  n : ℕ  -- number of lines
  total_intersections : ℕ  -- total number of intersection points
  triple_intersections : ℕ  -- number of points where three lines intersect

/-- The theorem statement -/
theorem line_configuration_theorem (config : LineConfiguration) :
  config.n > 0 ∧
  config.total_intersections = 16 ∧
  config.triple_intersections = 6 ∧
  (∀ (i j : ℕ), i < config.n → j < config.n → i ≠ j → ∃ (p : ℕ), p < config.total_intersections) ∧
  (∀ (i j k l : ℕ), i < config.n → j < config.n → k < config.n → l < config.n →
    i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
    ¬∃ (p : ℕ), p < config.total_intersections) →
  config.n = 8 :=
sorry

end NUMINAMATH_CALUDE_line_configuration_theorem_l3936_393648


namespace NUMINAMATH_CALUDE_student_distribution_l3936_393685

theorem student_distribution (total : ℕ) (a b : ℕ) : 
  total = 81 →
  a + b = total →
  a = b - 9 →
  a = 36 ∧ b = 45 := by
sorry

end NUMINAMATH_CALUDE_student_distribution_l3936_393685


namespace NUMINAMATH_CALUDE_vacuum_cleaner_cost_l3936_393627

theorem vacuum_cleaner_cost (dishwasher_cost coupon_value total_spent : ℕ) 
  (h1 : dishwasher_cost = 450)
  (h2 : coupon_value = 75)
  (h3 : total_spent = 625) :
  ∃ (vacuum_cost : ℕ), vacuum_cost = 250 ∧ vacuum_cost + dishwasher_cost - coupon_value = total_spent :=
by sorry

end NUMINAMATH_CALUDE_vacuum_cleaner_cost_l3936_393627


namespace NUMINAMATH_CALUDE_function_transformation_l3936_393666

theorem function_transformation (f : ℝ → ℝ) :
  (∀ x, f (x - 1) = x^2 + 6*x) →
  (∀ x, f x = x^2 + 8*x + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l3936_393666


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l3936_393652

theorem sum_of_squares_problem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 52)
  (sum_of_products : x*y + y*z + z*x = 27) :
  x + y + z = Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l3936_393652


namespace NUMINAMATH_CALUDE_junior_count_l3936_393660

theorem junior_count (total : ℕ) (junior_percent : ℚ) (senior_percent : ℚ) :
  total = 28 →
  junior_percent = 1/4 →
  senior_percent = 1/10 →
  ∃ (juniors seniors : ℕ),
    juniors + seniors = total ∧
    junior_percent * juniors = senior_percent * seniors ∧
    juniors = 8 := by
  sorry

end NUMINAMATH_CALUDE_junior_count_l3936_393660


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l3936_393681

theorem shortest_altitude_right_triangle (a b c h : ℝ) : 
  a = 8 ∧ b = 15 ∧ c = 17 →
  a^2 + b^2 = c^2 →
  h * c = 2 * (1/2 * a * b) →
  h = 120/17 := by
sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l3936_393681


namespace NUMINAMATH_CALUDE_parabola_directrix_l3936_393664

/-- The directrix of the parabola y = -1/4 * x^2 is y = 1 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = -1/4 * x^2 → (∃ (d : ℝ), d = 1 ∧ 
    ∀ (p : ℝ × ℝ), p.2 = -1/4 * p.1^2 → 
      ∃ (f : ℝ), (p.1 - 0)^2 + (p.2 - f)^2 = (p.2 - d)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3936_393664


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3936_393684

theorem cubic_equation_solution : 
  ∀ x y : ℕ, x^3 - y^3 = x * y + 61 → x = 6 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3936_393684


namespace NUMINAMATH_CALUDE_andy_remaining_demerits_l3936_393669

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def max_demerits : ℕ := 50

/-- The number of demerits Andy gets per instance of being late -/
def demerits_per_late : ℕ := 2

/-- The number of times Andy was late -/
def times_late : ℕ := 6

/-- The number of demerits Andy got for making an inappropriate joke -/
def demerits_for_joke : ℕ := 15

/-- The number of additional demerits Andy can get before being fired -/
def remaining_demerits : ℕ := max_demerits - (demerits_per_late * times_late + demerits_for_joke)

theorem andy_remaining_demerits : remaining_demerits = 23 := by
  sorry

end NUMINAMATH_CALUDE_andy_remaining_demerits_l3936_393669


namespace NUMINAMATH_CALUDE_fifth_month_sales_l3936_393667

def sales_1 : ℕ := 5420
def sales_2 : ℕ := 5660
def sales_3 : ℕ := 6200
def sales_4 : ℕ := 6350
def sales_6 : ℕ := 6470
def average_sale : ℕ := 6100
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    sales_5 = num_months * average_sale - (sales_1 + sales_2 + sales_3 + sales_4 + sales_6) ∧
    sales_5 = 6500 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l3936_393667


namespace NUMINAMATH_CALUDE_max_value_of_a_max_value_is_tight_l3936_393602

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x - a ≥ 0) → a ≤ -1 := by
  sorry

theorem max_value_is_tight : ∃ a : ℝ, a = -1 ∧ (∀ x : ℝ, x^2 - 2*x - a ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_a_max_value_is_tight_l3936_393602


namespace NUMINAMATH_CALUDE_mens_wages_l3936_393637

/-- Proves that given the conditions in the problem, the total wages for 9 men is Rs. 72 -/
theorem mens_wages (total_earnings : ℕ) (num_men num_boys : ℕ) (W : ℕ) :
  total_earnings = 216 →
  num_men = 9 →
  num_boys = 7 →
  num_men * W = num_men * num_boys →
  (3 * num_men : ℕ) * (total_earnings / (3 * num_men)) = 72 :=
by sorry

end NUMINAMATH_CALUDE_mens_wages_l3936_393637


namespace NUMINAMATH_CALUDE_farmer_randy_planting_rate_l3936_393628

/-- Represents the cotton planting problem for Farmer Randy -/
structure CottonPlanting where
  total_acres : ℕ
  total_days : ℕ
  first_crew_tractors : ℕ
  first_crew_days : ℕ
  second_crew_tractors : ℕ
  second_crew_days : ℕ

/-- Calculates the acres per tractor per day needed to meet the planting deadline -/
def acres_per_tractor_per_day (cp : CottonPlanting) : ℚ :=
  cp.total_acres / (cp.first_crew_tractors * cp.first_crew_days + cp.second_crew_tractors * cp.second_crew_days)

/-- Theorem stating that for Farmer Randy's specific situation, each tractor needs to plant 68 acres per day -/
theorem farmer_randy_planting_rate :
  let cp : CottonPlanting := {
    total_acres := 1700,
    total_days := 5,
    first_crew_tractors := 2,
    first_crew_days := 2,
    second_crew_tractors := 7,
    second_crew_days := 3
  }
  acres_per_tractor_per_day cp = 68 := by
  sorry

end NUMINAMATH_CALUDE_farmer_randy_planting_rate_l3936_393628


namespace NUMINAMATH_CALUDE_playground_area_l3936_393626

/-- Given a rectangular playground with perimeter 90 feet and length three times the width,
    prove that its area is 380.625 square feet. -/
theorem playground_area (w : ℝ) (l : ℝ) :
  (2 * l + 2 * w = 90) →  -- Perimeter is 90 feet
  (l = 3 * w) →           -- Length is three times the width
  (l * w = 380.625) :=    -- Area is 380.625 square feet
by sorry

end NUMINAMATH_CALUDE_playground_area_l3936_393626


namespace NUMINAMATH_CALUDE_max_sum_is_21_l3936_393691

/-- Represents a nonzero digit (1-9) -/
def NonzeroDigit := { d : ℕ // 1 ≤ d ∧ d ≤ 9 }

/-- Calculates An for a given nonzero digit a and positive integer n -/
def An (a : NonzeroDigit) (n : ℕ+) : ℕ :=
  a.val * (10^n.val - 1) / 9

/-- Calculates Bn for a given nonzero digit b and positive integer n -/
def Bn (b : NonzeroDigit) (n : ℕ+) : ℕ :=
  b.val * (10^n.val - 1) / 9

/-- Calculates Cn for a given nonzero digit c and positive integer n -/
def Cn (c : NonzeroDigit) (n : ℕ+) : ℕ :=
  c.val * (10^(n.val + 1) - 1) / 9

/-- Checks if the equation Cn - Bn = An^2 holds for given a, b, c, and n -/
def EquationHolds (a b c : NonzeroDigit) (n : ℕ+) : Prop :=
  Cn c n - Bn b n = (An a n)^2

/-- Checks if there exist at least two distinct positive integers n for which the equation holds -/
def ExistTwoDistinctN (a b c : NonzeroDigit) : Prop :=
  ∃ n₁ n₂ : ℕ+, n₁ ≠ n₂ ∧ EquationHolds a b c n₁ ∧ EquationHolds a b c n₂

/-- The main theorem stating that the maximum value of a + b + c is 21 -/
theorem max_sum_is_21 :
  ∀ a b c : NonzeroDigit,
  ExistTwoDistinctN a b c →
  a.val + b.val + c.val ≤ 21 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_21_l3936_393691


namespace NUMINAMATH_CALUDE_divisibility_power_increase_l3936_393609

theorem divisibility_power_increase (k m n : ℕ) (a : ℕ → ℕ) :
  (m^n ∣ a k) → (m^(n+1) ∣ a (k*m)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_power_increase_l3936_393609


namespace NUMINAMATH_CALUDE_abs_function_symmetric_about_y_axis_l3936_393646

def f (x : ℝ) : ℝ := |x|

theorem abs_function_symmetric_about_y_axis :
  ∀ x : ℝ, f (-x) = f x :=
by
  sorry

end NUMINAMATH_CALUDE_abs_function_symmetric_about_y_axis_l3936_393646


namespace NUMINAMATH_CALUDE_total_triangles_is_twenty_l3936_393638

/-- A rectangle with diagonals and midpoint segments. -/
structure RectangleWithDiagonals where
  /-- The rectangle has different length sides. -/
  different_sides : Bool
  /-- The diagonals intersect at the center. -/
  diagonals_intersect_center : Bool
  /-- Segments join midpoints of opposite sides. -/
  midpoint_segments : Bool

/-- Count the number of triangles in the rectangle configuration. -/
def count_triangles (r : RectangleWithDiagonals) : ℕ :=
  sorry

/-- Theorem stating that the total number of triangles is 20. -/
theorem total_triangles_is_twenty (r : RectangleWithDiagonals) 
  (h1 : r.different_sides = true)
  (h2 : r.diagonals_intersect_center = true)
  (h3 : r.midpoint_segments = true) : 
  count_triangles r = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_twenty_l3936_393638


namespace NUMINAMATH_CALUDE_grape_juice_theorem_l3936_393622

/-- Represents a fruit drink composition -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ

/-- Calculates the amount of grape juice in the drink -/
def grape_juice_amount (drink : FruitDrink) : ℝ :=
  drink.total - (drink.orange_percent * drink.total + drink.watermelon_percent * drink.total)

/-- Theorem: The amount of grape juice in the specified drink is 70 ounces -/
theorem grape_juice_theorem (drink : FruitDrink) 
    (h1 : drink.total = 200)
    (h2 : drink.orange_percent = 0.25)
    (h3 : drink.watermelon_percent = 0.40) : 
  grape_juice_amount drink = 70 := by
  sorry

#eval grape_juice_amount { total := 200, orange_percent := 0.25, watermelon_percent := 0.40 }

end NUMINAMATH_CALUDE_grape_juice_theorem_l3936_393622


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l3936_393639

theorem abs_neg_three_eq_three : |(-3 : ℝ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l3936_393639


namespace NUMINAMATH_CALUDE_fraction_equality_l3936_393635

theorem fraction_equality (x : ℝ) : (4 + x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3936_393635


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3936_393654

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (5 * Complex.I) / (1 + 2 * Complex.I) → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3936_393654


namespace NUMINAMATH_CALUDE_v2_equals_5_l3936_393614

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Definition of V₂ in Horner's method -/
def V₂ (a₅ a₄ a₃ a₂ a₁ a₀ : ℝ) (x : ℝ) : ℝ :=
  (a₅ * x + a₄) * x - a₃

/-- Theorem: V₂ equals 5 for the given polynomial when x = 2 -/
theorem v2_equals_5 :
  let f : ℝ → ℝ := fun x => 2 * x^5 - 3 * x^3 + 2 * x^2 - x + 5
  V₂ 2 0 (-3) 2 (-1) 5 2 = 5 := by
  sorry

#eval V₂ 2 0 (-3) 2 (-1) 5 2

end NUMINAMATH_CALUDE_v2_equals_5_l3936_393614


namespace NUMINAMATH_CALUDE_angle_equality_l3936_393621

-- Define angles A, B, and C
variable (A B C : ℝ)

-- Define the conditions
axiom angle_sum_1 : A + B = 180
axiom angle_sum_2 : B + C = 180

-- State the theorem
theorem angle_equality : A = C := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l3936_393621


namespace NUMINAMATH_CALUDE_victors_flower_stickers_l3936_393620

theorem victors_flower_stickers :
  ∀ (flower_stickers animal_stickers : ℕ),
    animal_stickers = flower_stickers - 2 →
    flower_stickers + animal_stickers = 14 →
    flower_stickers = 8 := by
  sorry

end NUMINAMATH_CALUDE_victors_flower_stickers_l3936_393620


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l3936_393612

theorem complex_modulus_equation (t : ℝ) (h : t > 0) :
  Complex.abs (8 + t * Complex.I) = 12 → t = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l3936_393612


namespace NUMINAMATH_CALUDE_total_cost_proof_l3936_393693

/-- The cost of a single ticket in dollars -/
def ticket_cost : ℝ := 44

/-- The number of tickets purchased -/
def num_tickets : ℕ := 7

/-- The total cost of tickets in dollars -/
def total_cost : ℝ := ticket_cost * num_tickets

theorem total_cost_proof : total_cost = 308 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l3936_393693


namespace NUMINAMATH_CALUDE_baseball_cards_per_page_l3936_393644

theorem baseball_cards_per_page : 
  ∀ (cards_per_page : ℕ+) (full_pages : ℕ+),
  cards_per_page.val * full_pages.val + 1 = 7 →
  cards_per_page = 2 := by
sorry

end NUMINAMATH_CALUDE_baseball_cards_per_page_l3936_393644


namespace NUMINAMATH_CALUDE_correct_field_equation_l3936_393657

/-- Represents a rectangular field with given area and width-length relationship -/
structure RectangularField where
  area : ℕ
  lengthWidthDiff : ℕ

/-- The equation representing the relationship between length and area for the given field -/
def fieldEquation (field : RectangularField) (x : ℕ) : Prop :=
  x * (x - field.lengthWidthDiff) = field.area

/-- Theorem stating that the equation correctly represents the given field properties -/
theorem correct_field_equation (field : RectangularField) 
    (h1 : field.area = 864) (h2 : field.lengthWidthDiff = 12) :
    ∃ x : ℕ, fieldEquation field x :=
  sorry

end NUMINAMATH_CALUDE_correct_field_equation_l3936_393657
