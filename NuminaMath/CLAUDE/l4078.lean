import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_variables_l4078_407849

theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 15 - 4*x)
  (eq2 : x + z = -17 - 4*y)
  (eq3 : x + y = 8 - 4*z) :
  2*x + 2*y + 2*z = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l4078_407849


namespace NUMINAMATH_CALUDE_range_of_b_minus_a_l4078_407872

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem range_of_b_minus_a (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, -1 ≤ f x ∧ f x ≤ 3) →
  (∃ x ∈ Set.Icc a b, f x = -1) →
  (∃ x ∈ Set.Icc a b, f x = 3) →
  2 ≤ b - a ∧ b - a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_minus_a_l4078_407872


namespace NUMINAMATH_CALUDE_cube_coverage_tape_pieces_correct_l4078_407805

/-- Represents the number of tape pieces needed to cover a cube --/
def tape_pieces (n : ℕ) : ℕ := 2 * n

/-- Theorem stating that the number of tape pieces needed to cover a cube with edge length n is 2n --/
theorem cube_coverage (n : ℕ) :
  tape_pieces n = 2 * n :=
by sorry

/-- Represents the properties of the tape coverage method --/
structure TapeCoverage where
  edge_length : ℕ
  tape_width : ℕ
  parallel_to_edge : Bool
  can_cross_edges : Bool
  no_overhang : Bool

/-- Theorem stating that the tape_pieces function gives the correct number of pieces
    for a cube coverage satisfying the given constraints --/
theorem tape_pieces_correct (coverage : TapeCoverage) 
  (h1 : coverage.tape_width = 1)
  (h2 : coverage.parallel_to_edge = true)
  (h3 : coverage.can_cross_edges = true)
  (h4 : coverage.no_overhang = true) :
  tape_pieces coverage.edge_length = 2 * coverage.edge_length :=
by sorry

end NUMINAMATH_CALUDE_cube_coverage_tape_pieces_correct_l4078_407805


namespace NUMINAMATH_CALUDE_equal_chore_time_l4078_407874

/-- The time in minutes it takes to sweep one room -/
def sweep_time : ℕ := 3

/-- The time in minutes it takes to wash one dish -/
def dish_time : ℕ := 2

/-- The time in minutes it takes to do one load of laundry -/
def laundry_time : ℕ := 9

/-- The number of rooms Anna sweeps -/
def anna_rooms : ℕ := 10

/-- The number of laundry loads Billy does -/
def billy_laundry : ℕ := 2

/-- The number of dishes Billy should wash -/
def billy_dishes : ℕ := 6

theorem equal_chore_time : 
  anna_rooms * sweep_time = billy_laundry * laundry_time + billy_dishes * dish_time := by
  sorry

end NUMINAMATH_CALUDE_equal_chore_time_l4078_407874


namespace NUMINAMATH_CALUDE_only_D_in_second_quadrant_l4078_407894

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def point_A : ℝ × ℝ := (2, 3)
def point_B : ℝ × ℝ := (2, -3)
def point_C : ℝ × ℝ := (-2, -3)
def point_D : ℝ × ℝ := (-2, 3)

theorem only_D_in_second_quadrant :
  ¬(second_quadrant point_A.1 point_A.2) ∧
  ¬(second_quadrant point_B.1 point_B.2) ∧
  ¬(second_quadrant point_C.1 point_C.2) ∧
  second_quadrant point_D.1 point_D.2 := by sorry

end NUMINAMATH_CALUDE_only_D_in_second_quadrant_l4078_407894


namespace NUMINAMATH_CALUDE_largest_number_in_set_l4078_407859

def S (a : ℝ) : Set ℝ := {-3*a, 4*a, 24/a, a^2, 2*a+6, 1}

theorem largest_number_in_set (a : ℝ) (h : a = 3) :
  (∀ x ∈ S a, x ≤ 4*a) ∧ (∀ x ∈ S a, x ≤ 2*a+6) ∧ (4*a ∈ S a) ∧ (2*a+6 ∈ S a) :=
sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l4078_407859


namespace NUMINAMATH_CALUDE_OPRQ_shape_l4078_407804

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A figure formed by four points -/
structure Quadrilateral where
  O : Point
  P : Point
  R : Point
  Q : Point

/-- Check if three points are collinear -/
def collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) = (C.x - A.x) * (B.y - A.y)

/-- Check if two line segments are parallel -/
def parallel (A B C D : Point) : Prop :=
  (B.x - A.x) * (D.y - C.y) = (D.x - C.x) * (B.y - A.y)

/-- Check if a quadrilateral is a straight line -/
def isStraightLine (quad : Quadrilateral) : Prop :=
  collinear quad.O quad.P quad.Q ∧ collinear quad.O quad.R quad.Q

/-- Check if a quadrilateral is a trapezoid -/
def isTrapezoid (quad : Quadrilateral) : Prop :=
  (parallel quad.O quad.P quad.Q quad.R ∧ ¬parallel quad.O quad.Q quad.P quad.R) ∨
  (¬parallel quad.O quad.P quad.Q quad.R ∧ parallel quad.O quad.Q quad.P quad.R)

/-- Check if a quadrilateral is a parallelogram -/
def isParallelogram (quad : Quadrilateral) : Prop :=
  parallel quad.O quad.P quad.Q quad.R ∧ parallel quad.O quad.Q quad.P quad.R

/-- The main theorem -/
theorem OPRQ_shape (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂ ∨ y₁ ≠ y₂) :
  let P : Point := ⟨x₁, y₁⟩
  let Q : Point := ⟨x₂, y₂⟩
  let R : Point := ⟨x₁ - x₂, y₁ - y₂⟩
  let O : Point := ⟨0, 0⟩
  let quad : Quadrilateral := ⟨O, P, R, Q⟩
  (isStraightLine quad ∨ isTrapezoid quad) ∧ ¬isParallelogram quad := by
  sorry


end NUMINAMATH_CALUDE_OPRQ_shape_l4078_407804


namespace NUMINAMATH_CALUDE_total_playing_hours_l4078_407865

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of hours Nathan plays per day -/
def nathan_hours_per_day : ℕ := 3

/-- The number of weeks Nathan plays -/
def nathan_weeks : ℕ := 2

/-- The number of hours Tobias plays per day -/
def tobias_hours_per_day : ℕ := 5

/-- The number of weeks Tobias plays -/
def tobias_weeks : ℕ := 1

/-- The total number of hours Nathan and Tobias played -/
def total_hours : ℕ := 
  nathan_hours_per_day * days_per_week * nathan_weeks + 
  tobias_hours_per_day * days_per_week * tobias_weeks

theorem total_playing_hours : total_hours = 77 := by
  sorry

end NUMINAMATH_CALUDE_total_playing_hours_l4078_407865


namespace NUMINAMATH_CALUDE_final_number_independent_of_operations_l4078_407844

/-- Represents the state of the blackboard with counts of 0, 1, and 2 -/
structure Board :=
  (count_0 : ℕ)
  (count_1 : ℕ)
  (count_2 : ℕ)

/-- Represents a single operation on the board -/
inductive Operation
  | replace_0_1_with_2
  | replace_1_2_with_0
  | replace_0_2_with_1

/-- Applies an operation to the board -/
def apply_operation (b : Board) (op : Operation) : Board :=
  match op with
  | Operation.replace_0_1_with_2 => ⟨b.count_0 - 1, b.count_1 - 1, b.count_2 + 1⟩
  | Operation.replace_1_2_with_0 => ⟨b.count_0 + 1, b.count_1 - 1, b.count_2 - 1⟩
  | Operation.replace_0_2_with_1 => ⟨b.count_0 - 1, b.count_1 + 1, b.count_2 - 1⟩

/-- Checks if the board has only one number left -/
def is_final (b : Board) : Prop :=
  (b.count_0 = 1 ∧ b.count_1 = 0 ∧ b.count_2 = 0) ∨
  (b.count_0 = 0 ∧ b.count_1 = 1 ∧ b.count_2 = 0) ∨
  (b.count_0 = 0 ∧ b.count_1 = 0 ∧ b.count_2 = 1)

/-- The final number on the board -/
def final_number (b : Board) : ℕ :=
  if b.count_0 = 1 then 0
  else if b.count_1 = 1 then 1
  else 2

/-- Theorem: The final number is determined by initial parity, regardless of operations -/
theorem final_number_independent_of_operations (initial : Board) 
  (ops1 ops2 : List Operation) (h1 : is_final (ops1.foldl apply_operation initial))
  (h2 : is_final (ops2.foldl apply_operation initial)) :
  final_number (ops1.foldl apply_operation initial) = 
  final_number (ops2.foldl apply_operation initial) :=
sorry

end NUMINAMATH_CALUDE_final_number_independent_of_operations_l4078_407844


namespace NUMINAMATH_CALUDE_lune_area_specific_case_l4078_407808

/-- Represents a semicircle with a given diameter -/
structure Semicircle where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- Represents a lune formed by two semicircles -/
structure Lune where
  upper : Semicircle
  lower : Semicircle
  upper_on_lower : upper.diameter < lower.diameter

/-- Calculates the area of a lune -/
noncomputable def lune_area (l : Lune) : ℝ :=
  sorry

theorem lune_area_specific_case :
  let upper := Semicircle.mk 3 (by norm_num)
  let lower := Semicircle.mk 4 (by norm_num)
  let l := Lune.mk upper lower (by norm_num)
  lune_area l = (9 * Real.sqrt 3) / 4 - (55 / 24) * Real.pi :=
sorry

end NUMINAMATH_CALUDE_lune_area_specific_case_l4078_407808


namespace NUMINAMATH_CALUDE_second_hand_large_division_time_l4078_407866

/-- The number of large divisions on a clock face -/
def large_divisions : ℕ := 12

/-- The number of small divisions in each large division -/
def small_divisions_per_large : ℕ := 5

/-- The time (in seconds) it takes for the second hand to move one small division -/
def time_per_small_division : ℕ := 1

/-- The time it takes for the second hand to move one large division -/
def time_for_large_division : ℕ := small_divisions_per_large * time_per_small_division

theorem second_hand_large_division_time :
  time_for_large_division = 5 := by sorry

end NUMINAMATH_CALUDE_second_hand_large_division_time_l4078_407866


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l4078_407895

/-- The volume of a cylinder minus two congruent cones -/
theorem cylinder_minus_cones_volume (r h_cylinder h_cone : ℝ) 
  (hr : r = 10)
  (hh_cylinder : h_cylinder = 20)
  (hh_cone : h_cone = 9) :
  π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 1400 * π := by
sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l4078_407895


namespace NUMINAMATH_CALUDE_remaining_distance_is_4430_l4078_407848

/-- Represents the state of the race between Alex and Max -/
structure RaceState where
  total_distance : ℕ
  alex_lead : ℤ

/-- Calculates the final race state after all lead changes -/
def final_race_state : RaceState :=
  let initial_state : RaceState := { total_distance := 5000, alex_lead := 0 }
  let after_uphill : RaceState := { initial_state with alex_lead := 300 }
  let after_downhill : RaceState := { after_uphill with alex_lead := after_uphill.alex_lead - 170 }
  { after_downhill with alex_lead := after_downhill.alex_lead + 440 }

/-- Calculates the remaining distance for Max to catch up -/
def remaining_distance (state : RaceState) : ℕ :=
  state.total_distance - state.alex_lead.toNat

/-- Theorem stating the remaining distance for Max to catch up -/
theorem remaining_distance_is_4430 :
  remaining_distance final_race_state = 4430 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_is_4430_l4078_407848


namespace NUMINAMATH_CALUDE_maxwell_current_age_l4078_407861

/-- Maxwell's current age -/
def maxwell_age : ℕ := sorry

/-- Maxwell's sister's current age -/
def sister_age : ℕ := 2

/-- In 2 years, Maxwell will be twice his sister's age -/
axiom maxwell_twice_sister : maxwell_age + 2 = 2 * (sister_age + 2)

theorem maxwell_current_age : maxwell_age = 6 := by sorry

end NUMINAMATH_CALUDE_maxwell_current_age_l4078_407861


namespace NUMINAMATH_CALUDE_probability_same_tune_is_one_fourth_l4078_407833

/-- A defective toy train that produces two different tunes at random -/
structure DefectiveToyTrain :=
  (tunes : Fin 2 → String)

/-- The probability of the defective toy train producing 3 music tunes of the same type -/
def probability_same_tune (train : DefectiveToyTrain) : ℚ :=
  1 / 4

/-- Theorem stating that the probability of producing 3 music tunes of the same type is 1/4 -/
theorem probability_same_tune_is_one_fourth (train : DefectiveToyTrain) :
  probability_same_tune train = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_tune_is_one_fourth_l4078_407833


namespace NUMINAMATH_CALUDE_locus_and_fixed_points_l4078_407803

-- Define the points and lines
def F : ℝ × ℝ := (1, 0)
def H : ℝ × ℝ := (1, 2)
def l : Set (ℝ × ℝ) := {p | p.1 = -1}

-- Define the locus C
def C : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

-- Define a function to represent a line passing through F and not perpendicular to x-axis
def line_through_F (m : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m * (p.1 - 1)}

-- Define the circle with diameter MN
def circle_MN (m : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.1 - 3 + p.2^2 + (4/m)*p.2 = 0}

-- State the theorem
theorem locus_and_fixed_points :
  ∀ (m : ℝ), m ≠ 0 →
  (∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ A ∈ line_through_F m ∧ B ∈ line_through_F m) →
  ((-3, 0) ∈ circle_MN m ∧ (1, 0) ∈ circle_MN m) :=
sorry

end NUMINAMATH_CALUDE_locus_and_fixed_points_l4078_407803


namespace NUMINAMATH_CALUDE_system_solution_unique_l4078_407815

theorem system_solution_unique (x y : ℝ) : 
  (x - y = -5 ∧ 3*x + 2*y = 10) ↔ (x = 0 ∧ y = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l4078_407815


namespace NUMINAMATH_CALUDE_books_checked_out_thursday_l4078_407889

theorem books_checked_out_thursday (initial_books : ℕ) (wednesday_checkout : ℕ) 
  (thursday_return : ℕ) (friday_return : ℕ) (final_books : ℕ) :
  initial_books = 98 →
  wednesday_checkout = 43 →
  thursday_return = 23 →
  friday_return = 7 →
  final_books = 80 →
  ∃ (thursday_checkout : ℕ),
    final_books = initial_books - wednesday_checkout + thursday_return - thursday_checkout + friday_return ∧
    thursday_checkout = 5 :=
by sorry

end NUMINAMATH_CALUDE_books_checked_out_thursday_l4078_407889


namespace NUMINAMATH_CALUDE_athena_snack_spending_l4078_407812

/-- Calculates the total amount spent by Athena on snacks -/
def total_spent (sandwich_price : ℚ) (sandwich_qty : ℕ)
                (drink_price : ℚ) (drink_qty : ℕ)
                (cookie_price : ℚ) (cookie_qty : ℕ)
                (chips_price : ℚ) (chips_qty : ℕ) : ℚ :=
  sandwich_price * sandwich_qty +
  drink_price * drink_qty +
  cookie_price * cookie_qty +
  chips_price * chips_qty

/-- Proves that Athena spent $33.95 on snacks -/
theorem athena_snack_spending :
  total_spent (325/100) 4 (275/100) 3 (150/100) 6 (185/100) 2 = 3395/100 := by
  sorry

end NUMINAMATH_CALUDE_athena_snack_spending_l4078_407812


namespace NUMINAMATH_CALUDE_pentagon_area_is_14_l4078_407801

/-- Represents a trapezoid segmented into two triangles and a pentagon -/
structure SegmentedTrapezoid where
  triangle1_area : ℝ
  triangle2_area : ℝ
  base_ratio : ℝ
  total_area : ℝ

/-- The area of the pentagon in a segmented trapezoid -/
def pentagon_area (t : SegmentedTrapezoid) : ℝ :=
  t.total_area - t.triangle1_area - t.triangle2_area

/-- Theorem stating that the area of the pentagon is 14 under given conditions -/
theorem pentagon_area_is_14 (t : SegmentedTrapezoid) 
  (h1 : t.triangle1_area = 8)
  (h2 : t.triangle2_area = 18)
  (h3 : t.base_ratio = 2)
  (h4 : t.total_area = 40) :
  pentagon_area t = 14 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_area_is_14_l4078_407801


namespace NUMINAMATH_CALUDE_five_by_seven_not_tileable_l4078_407827

/-- Represents a rectangular board -/
structure Board :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a domino -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- Checks if a board can be tiled with dominos -/
def can_be_tiled (b : Board) (d : Domino) : Prop :=
  (b.length * b.width) % (d.length * d.width) = 0

/-- The theorem stating that a 5×7 board cannot be tiled with 2×1 dominos -/
theorem five_by_seven_not_tileable :
  ¬(can_be_tiled (Board.mk 5 7) (Domino.mk 2 1)) :=
sorry

end NUMINAMATH_CALUDE_five_by_seven_not_tileable_l4078_407827


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l4078_407854

theorem cycle_price_calculation (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 1080) 
  (h2 : gain_percent = 20) : 
  ∃ original_price : ℝ, 
    original_price * (1 + gain_percent / 100) = selling_price ∧ 
    original_price = 900 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l4078_407854


namespace NUMINAMATH_CALUDE_reach_destination_in_time_l4078_407875

/-- The distance to the destination in kilometers -/
def destination_distance : ℝ := 62

/-- The walking speed in km/hr -/
def walking_speed : ℝ := 5

/-- The car speed in km/hr -/
def car_speed : ℝ := 50

/-- The maximum time allowed to reach the destination in hours -/
def max_time : ℝ := 3

/-- A strategy represents a plan for A, B, and C to reach the destination -/
structure Strategy where
  -- Add necessary fields to represent the strategy
  dummy : Unit

/-- Calculates the time taken to execute a given strategy -/
def time_taken (s : Strategy) : ℝ :=
  -- Implement the calculation of time taken for the strategy
  sorry

/-- Theorem stating that there exists a strategy to reach the destination in less than the maximum allowed time -/
theorem reach_destination_in_time :
  ∃ (s : Strategy), time_taken s < max_time :=
sorry

end NUMINAMATH_CALUDE_reach_destination_in_time_l4078_407875


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l4078_407862

theorem sum_of_reciprocal_squares (p q r : ℝ) : 
  p^3 - 9*p^2 + 8*p + 2 = 0 →
  q^3 - 9*q^2 + 8*q + 2 = 0 →
  r^3 - 9*r^2 + 8*r + 2 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  1/p^2 + 1/q^2 + 1/r^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l4078_407862


namespace NUMINAMATH_CALUDE_notecard_area_theorem_l4078_407831

/-- Given a rectangle with original dimensions 5 × 7 inches, prove that if shortening one side
    by 2 inches results in an area of 21 square inches, then shortening the other side
    by 2 inches instead will result in an area of 25 square inches. -/
theorem notecard_area_theorem :
  ∀ (original_width original_length : ℝ),
    original_width = 5 →
    original_length = 7 →
    (∃ (new_width new_length : ℝ),
      (new_width = original_width - 2 ∧ new_length = original_length ∨
       new_width = original_width ∧ new_length = original_length - 2) ∧
      new_width * new_length = 21) →
    ∃ (other_width other_length : ℝ),
      (other_width = original_width - 2 ∧ other_length = original_length ∨
       other_width = original_width ∧ other_length = original_length - 2) ∧
      other_width ≠ new_width ∧
      other_length ≠ new_length ∧
      other_width * other_length = 25 :=
by sorry

end NUMINAMATH_CALUDE_notecard_area_theorem_l4078_407831


namespace NUMINAMATH_CALUDE_range_of_a_l4078_407819

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x - 4 < 0) ↔ -4 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4078_407819


namespace NUMINAMATH_CALUDE_circle_radius_l4078_407863

/-- The radius of the circle described by the equation x^2 + y^2 - 6x + 8y = 0 is 5 -/
theorem circle_radius (x y : ℝ) : x^2 + y^2 - 6*x + 8*y = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 5^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l4078_407863


namespace NUMINAMATH_CALUDE_right_triangle_area_l4078_407873

theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = c^2) (h3 : c = 3) :
  (1/2) * a * b = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4078_407873


namespace NUMINAMATH_CALUDE_other_number_is_31_l4078_407817

theorem other_number_is_31 (a b : ℤ) (h1 : 3 * a + 2 * b = 140) (h2 : a = 26 ∨ b = 26) : (a = 26 ∧ b = 31) ∨ (a = 31 ∧ b = 26) :=
sorry

end NUMINAMATH_CALUDE_other_number_is_31_l4078_407817


namespace NUMINAMATH_CALUDE_solution_set_inequality_range_of_a_range_of_m_l4078_407880

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Statement 1
theorem solution_set_inequality (a : ℝ) :
  (∀ x, f a x ≤ 0 ↔ x ∈ Set.Icc 1 2) →
  (∀ x, f a x ≥ 1 - x^2 ↔ x ∈ Set.Iic (1/2) ∪ Set.Ici 1) :=
sorry

-- Statement 2
theorem range_of_a :
  (∀ a, (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*a*(x-1) + 4) →
    a ∈ Set.Iic (1/3)) :=
sorry

-- Statement 3
def g (m : ℝ) (x : ℝ) : ℝ := -x + m

theorem range_of_m :
  (∀ m, (∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Ioo 1 8, f (-3) x₁ = g m x₂) →
    m ∈ Set.Ioo 7 (31/4)) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_range_of_a_range_of_m_l4078_407880


namespace NUMINAMATH_CALUDE_annika_hans_age_multiple_l4078_407823

/-- Proves that in four years, Annika's age will be 3 times Hans' age -/
theorem annika_hans_age_multiple :
  ∀ (hans_current_age annika_current_age years_elapsed : ℕ),
    hans_current_age = 8 →
    annika_current_age = 32 →
    years_elapsed = 4 →
    (annika_current_age + years_elapsed) = 3 * (hans_current_age + years_elapsed) :=
by sorry

end NUMINAMATH_CALUDE_annika_hans_age_multiple_l4078_407823


namespace NUMINAMATH_CALUDE_binomial_distribution_p_value_l4078_407852

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: For a binomial distribution with E(ξ) = 7 and D(ξ) = 6, p = 1/7 -/
theorem binomial_distribution_p_value (ξ : BinomialDistribution) 
  (h_exp : expectedValue ξ = 7)
  (h_var : variance ξ = 6) : 
  ξ.p = 1/7 := by
  sorry


end NUMINAMATH_CALUDE_binomial_distribution_p_value_l4078_407852


namespace NUMINAMATH_CALUDE_weed_pulling_rate_is_11_l4078_407832

-- Define the hourly rates and hours worked
def mowing_rate : ℝ := 6
def mulch_rate : ℝ := 9
def mowing_hours : ℝ := 63
def weed_hours : ℝ := 9
def mulch_hours : ℝ := 10
def total_earnings : ℝ := 567

-- Define the function to calculate total earnings
def calculate_earnings (weed_rate : ℝ) : ℝ :=
  mowing_rate * mowing_hours + weed_rate * weed_hours + mulch_rate * mulch_hours

-- Theorem statement
theorem weed_pulling_rate_is_11 :
  ∃ (weed_rate : ℝ), calculate_earnings weed_rate = total_earnings ∧ weed_rate = 11 := by
  sorry

end NUMINAMATH_CALUDE_weed_pulling_rate_is_11_l4078_407832


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l4078_407822

theorem merry_go_round_revolutions 
  (outer_radius inner_radius : ℝ) 
  (outer_revolutions : ℕ) 
  (h1 : outer_radius = 40)
  (h2 : inner_radius = 10)
  (h3 : outer_revolutions = 15) :
  ∃ inner_revolutions : ℕ,
    inner_revolutions = 60 ∧
    outer_radius * outer_revolutions = inner_radius * inner_revolutions :=
by sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l4078_407822


namespace NUMINAMATH_CALUDE_solution_y_composition_l4078_407811

/-- Represents a chemical solution --/
structure Solution where
  a : ℝ  -- Percentage of chemical a
  b : ℝ  -- Percentage of chemical b

/-- Represents a mixture of two solutions --/
structure Mixture where
  x : Solution  -- First solution
  y : Solution  -- Second solution
  x_ratio : ℝ   -- Ratio of solution x in the mixture

def is_valid_solution (s : Solution) : Prop :=
  s.a + s.b = 100 ∧ s.a ≥ 0 ∧ s.b ≥ 0

def is_valid_mixture (m : Mixture) : Prop :=
  m.x_ratio ≥ 0 ∧ m.x_ratio ≤ 1

theorem solution_y_composition 
  (x : Solution)
  (y : Solution)
  (m : Mixture)
  (hx : is_valid_solution x)
  (hy : is_valid_solution y)
  (hm : is_valid_mixture m)
  (hx_comp : x.a = 40 ∧ x.b = 60)
  (hy_comp : y.a = y.b)
  (hm_comp : m.x = x ∧ m.y = y)
  (hm_ratio : m.x_ratio = 0.3)
  (hm_a : m.x_ratio * x.a + (1 - m.x_ratio) * y.a = 47) :
  y.a = 50 := by
    sorry

end NUMINAMATH_CALUDE_solution_y_composition_l4078_407811


namespace NUMINAMATH_CALUDE_largest_number_l4078_407820

theorem largest_number (a b c d : ℝ) (h1 : a = -3) (h2 : b = 0) (h3 : c = Real.sqrt 5) (h4 : d = 2) :
  c = max a (max b (max c d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l4078_407820


namespace NUMINAMATH_CALUDE_sum_of_decimals_equals_fraction_l4078_407870

theorem sum_of_decimals_equals_fraction :
  (∃ (x y : ℚ), x = 1/3 ∧ y = 7/9 ∧ x + y + (1/4 : ℚ) = 49/36) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_equals_fraction_l4078_407870


namespace NUMINAMATH_CALUDE_problem_statement_l4078_407838

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (n : ℕ), a^5 % 10 = b^5 % 10 → a - b = 10 * n) ∧
  (a^2 - b^2 = 1940 → a = 102 ∧ b = 92) ∧
  (a^2 - b^2 = 1920 → 
    ((a = 101 ∧ b = 91) ∨ 
     (a = 58 ∧ b = 38) ∨ 
     (a = 47 ∧ b = 17) ∨ 
     (a = 44 ∧ b = 4))) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4078_407838


namespace NUMINAMATH_CALUDE_divisor_totient_sum_bound_l4078_407893

/-- d(n) represents the number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- φ(n) represents Euler's totient function -/
def φ (n : ℕ+) : ℕ := sorry

/-- Theorem stating that c must be less than or equal to 1 -/
theorem divisor_totient_sum_bound (n : ℕ+) (c : ℕ) (h : d n + φ n = n + c) : c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_totient_sum_bound_l4078_407893


namespace NUMINAMATH_CALUDE_science_fair_girls_fraction_l4078_407824

theorem science_fair_girls_fraction :
  let pine_grove_total : ℕ := 300
  let pine_grove_ratio_boys : ℕ := 3
  let pine_grove_ratio_girls : ℕ := 2
  let maple_town_total : ℕ := 240
  let maple_town_ratio_boys : ℕ := 5
  let maple_town_ratio_girls : ℕ := 3
  let total_students := pine_grove_total + maple_town_total
  let pine_grove_girls := (pine_grove_total * pine_grove_ratio_girls) / (pine_grove_ratio_boys + pine_grove_ratio_girls)
  let maple_town_girls := (maple_town_total * maple_town_ratio_girls) / (maple_town_ratio_boys + maple_town_ratio_girls)
  let total_girls := pine_grove_girls + maple_town_girls
  (total_girls : ℚ) / total_students = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_science_fair_girls_fraction_l4078_407824


namespace NUMINAMATH_CALUDE_tenth_root_of_unity_l4078_407878

theorem tenth_root_of_unity (n : ℕ) (h : n = 3) :
  (Complex.tan (π / 4) + Complex.I) / (Complex.tan (π / 4) - Complex.I) =
  Complex.exp (Complex.I * (2 * ↑n * π / 10)) :=
by sorry

end NUMINAMATH_CALUDE_tenth_root_of_unity_l4078_407878


namespace NUMINAMATH_CALUDE_event_probability_l4078_407842

theorem event_probability (n : ℕ) (p_at_least_once : ℚ) (p_single : ℚ) : 
  n = 4 →
  p_at_least_once = 65 / 81 →
  (1 - p_single) ^ n = 1 - p_at_least_once →
  p_single = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_event_probability_l4078_407842


namespace NUMINAMATH_CALUDE_invisible_dots_sum_l4078_407846

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The number of dice -/
def num_dice : ℕ := 3

/-- The sum of visible numbers -/
def visible_sum : ℕ := 1 + 2 + 3 + 3 + 4

/-- The number of visible faces -/
def num_visible_faces : ℕ := 5

theorem invisible_dots_sum : 
  num_dice * die_sum - visible_sum = 50 := by sorry

end NUMINAMATH_CALUDE_invisible_dots_sum_l4078_407846


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l4078_407868

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_problem : Nat.gcd (factorial 7) ((factorial 10) / (factorial 5)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l4078_407868


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l4078_407839

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (initial_average : ℚ) 
  (incorrect_value : ℚ) 
  (correct_value : ℚ) : 
  n = 10 → 
  initial_average = 15 → 
  incorrect_value = 26 → 
  correct_value = 36 → 
  (n : ℚ) * initial_average + (correct_value - incorrect_value) = n * 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l4078_407839


namespace NUMINAMATH_CALUDE_horner_method_v2_equals_6_l4078_407876

def f (x : ℝ) : ℝ := 1 + 2*x + x^2 - 3*x^3 + 2*x^4

def horner_v2 (a₀ a₁ a₂ a₃ a₄ x : ℝ) : ℝ :=
  let v₁ := a₄ * x + a₃
  v₁ * x + a₂

theorem horner_method_v2_equals_6 :
  horner_v2 1 2 1 (-3) 2 (-1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v2_equals_6_l4078_407876


namespace NUMINAMATH_CALUDE_flower_baskets_count_l4078_407899

/-- The number of baskets used to hold flowers --/
def num_baskets (initial_flowers_per_daughter : ℕ) (additional_flowers : ℕ) (dead_flowers : ℕ) (flowers_per_basket : ℕ) : ℕ :=
  ((2 * initial_flowers_per_daughter + additional_flowers - dead_flowers) / flowers_per_basket)

/-- Theorem stating the number of baskets in the given scenario --/
theorem flower_baskets_count : num_baskets 5 20 10 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_flower_baskets_count_l4078_407899


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l4078_407858

theorem solve_exponential_equation :
  ∃! x : ℝ, (8 : ℝ)^(x - 1) / (2 : ℝ)^(x - 1) = (64 : ℝ)^(2 * x) ∧ x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l4078_407858


namespace NUMINAMATH_CALUDE_complex_root_modulus_l4078_407837

open Complex

theorem complex_root_modulus (c d : ℝ) (h : (1 + I)^2 + c*(1 + I) + d = 0) : 
  abs (c + d*I) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_modulus_l4078_407837


namespace NUMINAMATH_CALUDE_added_amount_proof_l4078_407888

theorem added_amount_proof (n x : ℝ) : n = 20 → (1/2) * n + x = 15 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_proof_l4078_407888


namespace NUMINAMATH_CALUDE_roots_difference_l4078_407814

theorem roots_difference (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 3 = 0 → 
  x₂^2 + x₂ - 3 = 0 → 
  |x₁ - x₂| = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_roots_difference_l4078_407814


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l4078_407864

def k : ℕ := 2010^2 + 2^2010

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := k) :
  (k^2 + 2^k) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l4078_407864


namespace NUMINAMATH_CALUDE_largest_integer_m_l4078_407867

theorem largest_integer_m (x y m : ℝ) : 
  x + 2*y = 2*m + 1 →
  2*x + y = m + 2 →
  x - y > 2 →
  ∀ k : ℤ, k > m → k ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_m_l4078_407867


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l4078_407890

theorem parabola_line_intersection (k : ℝ) : 
  (∃! x : ℝ, -2 = x^2 + k*x - 1) → (k = 2 ∨ k = -2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l4078_407890


namespace NUMINAMATH_CALUDE_parametric_to_standard_hyperbola_l4078_407879

theorem parametric_to_standard_hyperbola 
  (a b t x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (ht : t ≠ 0) :
  x = (a / 2) * (t + 1 / t) ∧ y = (b / 2) * (t - 1 / t) → 
  x^2 / a^2 - y^2 / b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_standard_hyperbola_l4078_407879


namespace NUMINAMATH_CALUDE_eight_digit_number_theorem_l4078_407850

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def move_last_to_first (n : ℕ) : ℕ :=
  let last_digit := n % 10
  let rest := n / 10
  last_digit * 10^7 + rest

theorem eight_digit_number_theorem (B : ℕ) (hB1 : is_coprime B 36) (hB2 : B > 7777777) :
  let A := move_last_to_first B
  (∃ A_min A_max : ℕ, 
    (∀ A' : ℕ, (∃ B' : ℕ, A' = move_last_to_first B' ∧ is_coprime B' 36 ∧ B' > 7777777) → 
      A_min ≤ A' ∧ A' ≤ A_max) ∧
    A_min = 17777779 ∧ 
    A_max = 99999998) :=
sorry

end NUMINAMATH_CALUDE_eight_digit_number_theorem_l4078_407850


namespace NUMINAMATH_CALUDE_final_sum_after_fillings_l4078_407896

/-- Represents the state of the blackboard after each filling -/
structure BoardState :=
  (numbers : List Int)
  (sum : Int)

/-- Perform one filling operation on the board -/
def fill (state : BoardState) : BoardState :=
  sorry

/-- The initial state of the board -/
def initial_state : BoardState :=
  { numbers := [2, 0, 2, 3], sum := 7 }

/-- Theorem stating the final sum after 2023 fillings -/
theorem final_sum_after_fillings :
  (Nat.iterate fill 2023 initial_state).sum = 2030 :=
sorry

end NUMINAMATH_CALUDE_final_sum_after_fillings_l4078_407896


namespace NUMINAMATH_CALUDE_calculator_game_sum_l4078_407845

/-- Represents the operation to be performed on a calculator --/
inductive Operation
  | Square
  | Negate

/-- Performs the specified operation on a number --/
def applyOperation (op : Operation) (x : Int) : Int :=
  match op with
  | Operation.Square => x * x
  | Operation.Negate => -x

/-- Determines the operation for the third calculator based on the pass number --/
def thirdOperation (pass : Nat) : Operation :=
  if pass % 2 = 0 then Operation.Negate else Operation.Square

/-- Performs one round of operations on the three calculators --/
def performRound (a b c : Int) (pass : Nat) : (Int × Int × Int) :=
  (applyOperation Operation.Square a,
   applyOperation Operation.Square b,
   applyOperation (thirdOperation pass) c)

/-- Performs n rounds of operations on the three calculators --/
def performNRounds (n : Nat) (a b c : Int) : (Int × Int × Int) :=
  match n with
  | 0 => (a, b, c)
  | n + 1 => 
    let (a', b', c') := performRound a b c n
    performNRounds n a' b' c'

theorem calculator_game_sum :
  let (a, b, c) := performNRounds 50 1 0 (-1)
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculator_game_sum_l4078_407845


namespace NUMINAMATH_CALUDE_approximation_place_l4078_407802

def number : ℕ := 345000000

theorem approximation_place (n : ℕ) (h : n = number) : 
  ∃ (k : ℕ), n ≥ 10^6 ∧ n < 10^7 ∧ k * 10^6 = n ∧ k < 1000 :=
by sorry

end NUMINAMATH_CALUDE_approximation_place_l4078_407802


namespace NUMINAMATH_CALUDE_ellipse_chord_through_focus_l4078_407834

/-- The x-coordinate of point A on the ellipse satisfies a specific quadratic equation --/
theorem ellipse_chord_through_focus (x y : ℝ) : 
  (x^2 / 36 + y^2 / 16 = 1) →  -- ellipse equation
  ((x - 2 * Real.sqrt 5)^2 + y^2 = 9) →  -- AF = 3
  (84 * x^2 - 400 * x + 552 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_through_focus_l4078_407834


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l4078_407887

theorem max_value_cos_sin (θ : Real) (h : -π/2 < θ ∧ θ < π/2) :
  ∃ (M : Real), M = Real.sqrt 2 ∧ 
  ∀ θ', -π/2 < θ' ∧ θ' < π/2 → 
    Real.cos (θ'/2) * (1 + Real.sin θ') ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l4078_407887


namespace NUMINAMATH_CALUDE_tenth_term_value_l4078_407813

def sequence_term (n : ℕ+) : ℚ :=
  (-1)^(n + 1 : ℕ) * (2 * n - 1 : ℚ) / ((n : ℚ)^2 + 1)

theorem tenth_term_value : sequence_term 10 = -19 / 101 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_value_l4078_407813


namespace NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l4078_407816

theorem absolute_value_square_sum_zero (x y : ℝ) :
  |x + 5| + (y - 2)^2 = 0 → x = -5 ∧ y = 2 ∧ x^y = 25 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l4078_407816


namespace NUMINAMATH_CALUDE_seven_digit_multiples_of_three_l4078_407830

theorem seven_digit_multiples_of_three (D B C : ℕ) : 
  D < 10 → B < 10 → C < 10 →
  (8 * 1000000 + 5 * 100000 + D * 10000 + 6 * 1000 + 3 * 100 + B * 10 + 2) % 3 = 0 →
  (4 * 1000000 + 1 * 100000 + 7 * 10000 + D * 1000 + B * 100 + 5 * 10 + C) % 3 = 0 →
  C = 2 := by
sorry

end NUMINAMATH_CALUDE_seven_digit_multiples_of_three_l4078_407830


namespace NUMINAMATH_CALUDE_cuboid_to_cube_l4078_407828

-- Define the dimensions of the original cuboid
def cuboid_length : ℝ := 27
def cuboid_width : ℝ := 18
def cuboid_height : ℝ := 12

-- Define the volume to be added
def added_volume : ℝ := 17.999999999999996

-- Define the edge length of the resulting cube in centimeters
def cube_edge_cm : ℕ := 1802

-- Theorem statement
theorem cuboid_to_cube :
  let original_volume := cuboid_length * cuboid_width * cuboid_height
  let total_volume := original_volume + added_volume
  let cube_edge_m := (total_volume ^ (1/3 : ℝ))
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 1 ∧ cube_edge_cm = ⌊cube_edge_m * 100 + ε⌋ :=
sorry

end NUMINAMATH_CALUDE_cuboid_to_cube_l4078_407828


namespace NUMINAMATH_CALUDE_unique_square_solution_l4078_407825

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digits_match (abc adeff : ℕ) : Prop :=
  let abc_digits := [abc / 100, (abc / 10) % 10, abc % 10]
  let adeff_digits := [adeff / 10000, (adeff / 1000) % 10, (adeff / 100) % 10, (adeff / 10) % 10, adeff % 10]
  (abc_digits.head? = adeff_digits.head?) ∧
  (abc_digits.get? 2 = adeff_digits.get? 3) ∧
  (abc_digits.get? 2 = adeff_digits.get? 4)

theorem unique_square_solution :
  ∀ abc adeff : ℕ,
    is_three_digit abc →
    is_five_digit adeff →
    abc ^ 2 = adeff →
    digits_match abc adeff →
    abc = 138 ∧ adeff = 19044 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_solution_l4078_407825


namespace NUMINAMATH_CALUDE_hank_bake_sale_earnings_l4078_407886

/-- Prove that Hank made $80 in the bake sale given the conditions of his fundraising activities. -/
theorem hank_bake_sale_earnings :
  let carwash_earnings : ℚ := 100
  let carwash_donation_rate : ℚ := 90 / 100
  let bake_sale_donation_rate : ℚ := 75 / 100
  let lawn_mowing_earnings : ℚ := 50
  let lawn_mowing_donation_rate : ℚ := 1
  let total_donation : ℚ := 200
  ∃ bake_sale_earnings : ℚ,
    bake_sale_earnings * bake_sale_donation_rate +
    carwash_earnings * carwash_donation_rate +
    lawn_mowing_earnings * lawn_mowing_donation_rate = total_donation ∧
    bake_sale_earnings = 80 :=
by sorry

end NUMINAMATH_CALUDE_hank_bake_sale_earnings_l4078_407886


namespace NUMINAMATH_CALUDE_marias_apple_sales_l4078_407881

/-- Given Maria's apple sales, prove the amount sold in the second hour -/
theorem marias_apple_sales (first_hour_sales second_hour_sales : ℝ) 
  (h1 : first_hour_sales = 10)
  (h2 : (first_hour_sales + second_hour_sales) / 2 = 6) : 
  second_hour_sales = 2 := by
  sorry

end NUMINAMATH_CALUDE_marias_apple_sales_l4078_407881


namespace NUMINAMATH_CALUDE_dice_sum_pigeonhole_l4078_407877

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- Represents the sum of four dice rolls -/
def DiceSum := Fin 21

/-- The minimum number of throws required to guarantee a repeated sum -/
def minThrows : Nat := 22

theorem dice_sum_pigeonhole :
  ∀ (rolls : Fin minThrows → DiceSum),
  ∃ (i j : Fin minThrows), i ≠ j ∧ rolls i = rolls j :=
sorry

end NUMINAMATH_CALUDE_dice_sum_pigeonhole_l4078_407877


namespace NUMINAMATH_CALUDE_total_games_is_30_l4078_407860

/-- The number of Monopoly games won by Betsy, Helen, and Susan -/
def monopoly_games (betsy helen susan : ℕ) : Prop :=
  betsy = 5 ∧ helen = 2 * betsy ∧ susan = 3 * betsy

/-- The total number of games won by all three players -/
def total_games (betsy helen susan : ℕ) : ℕ :=
  betsy + helen + susan

/-- Theorem stating that the total number of games won is 30 -/
theorem total_games_is_30 :
  ∀ betsy helen susan : ℕ,
  monopoly_games betsy helen susan →
  total_games betsy helen susan = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_total_games_is_30_l4078_407860


namespace NUMINAMATH_CALUDE_stock_value_change_l4078_407840

theorem stock_value_change (x : ℝ) (h : x > 0) : 
  let day1_value := x * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  (day2_value - x) / x * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_stock_value_change_l4078_407840


namespace NUMINAMATH_CALUDE_subway_speed_comparison_l4078_407882

-- Define the speed function
def speed (s : ℝ) : ℝ := s^2 + 2*s

-- Define the theorem
theorem subway_speed_comparison :
  ∃! t : ℝ, 0 ≤ t ∧ t ≤ 7 ∧ speed 5 = speed t + 20 ∧ t = 3 := by
  sorry

end NUMINAMATH_CALUDE_subway_speed_comparison_l4078_407882


namespace NUMINAMATH_CALUDE_specific_pairing_probability_l4078_407800

/-- The probability of a specific pairing in a class of 50 students -/
theorem specific_pairing_probability (n : ℕ) (h : n = 50) :
  (1 : ℚ) / (n - 1) = 1 / 49 := by
  sorry

#check specific_pairing_probability

end NUMINAMATH_CALUDE_specific_pairing_probability_l4078_407800


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l4078_407807

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (a * 10 + b) / 99

/-- The fraction 0.overline{72} divided by 0.overline{27} is equal to 8/3 -/
theorem repeating_decimal_ratio : 
  (RepeatingDecimal 7 2) / (RepeatingDecimal 2 7) = 8 / 3 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l4078_407807


namespace NUMINAMATH_CALUDE_intersection_of_lines_l4078_407809

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The intersection of lines AB and CD --/
theorem intersection_of_lines 
  (A : ℝ × ℝ × ℝ) 
  (B : ℝ × ℝ × ℝ) 
  (C : ℝ × ℝ × ℝ) 
  (D : ℝ × ℝ × ℝ) 
  (h1 : A = (6, -7, 7)) 
  (h2 : B = (15, -16, 11)) 
  (h3 : C = (0, 3, -6)) 
  (h4 : D = (2, -5, 10)) : 
  intersection_point A B C D = (144/27, -171/27, 181/27) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l4078_407809


namespace NUMINAMATH_CALUDE_wendy_accounting_percentage_l4078_407829

/-- Calculates the percentage of life spent in accounting-related jobs -/
def accounting_percentage (years_accountant : ℕ) (years_manager : ℕ) (total_lifespan : ℕ) : ℚ :=
  (years_accountant + years_manager : ℚ) / total_lifespan * 100

/-- Wendy's accounting career percentage theorem -/
theorem wendy_accounting_percentage :
  accounting_percentage 25 15 80 = 50 := by
  sorry

end NUMINAMATH_CALUDE_wendy_accounting_percentage_l4078_407829


namespace NUMINAMATH_CALUDE_consecutive_pages_product_l4078_407810

theorem consecutive_pages_product (n : ℕ) : 
  n > 0 ∧ n + (n + 1) = 217 → n * (n + 1) = 11772 := by
sorry

end NUMINAMATH_CALUDE_consecutive_pages_product_l4078_407810


namespace NUMINAMATH_CALUDE_number_problem_l4078_407826

theorem number_problem : ∃ x : ℚ, (35 / 100) * x = (40 / 100) * 50 ∧ x = 400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4078_407826


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l4078_407897

theorem min_value_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  Real.sqrt (x + 1/x) + Real.sqrt (y + 1/y) ≥ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l4078_407897


namespace NUMINAMATH_CALUDE_clothing_expense_l4078_407857

theorem clothing_expense (total_spent adidas_original nike skechers puma adidas clothes : ℝ) 
  (h_total : total_spent = 12000)
  (h_nike : nike = 2 * adidas)
  (h_skechers : adidas = 1/3 * skechers)
  (h_puma : puma = 3/4 * nike)
  (h_adidas_original : adidas_original = 900)
  (h_adidas_discount : adidas = adidas_original * 0.9)
  (h_sum : total_spent = nike + adidas + skechers + puma + clothes) :
  clothes = 5925 := by
sorry


end NUMINAMATH_CALUDE_clothing_expense_l4078_407857


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l4078_407892

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the limit condition
variable (h : ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((f 1 - f (1 + 2*x)) / (2*x)) - 1| < ε)

-- State the theorem
theorem tangent_slope_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((f 1 - f (1 + 2*x)) / (2*x)) - 1| < ε) : 
  deriv f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l4078_407892


namespace NUMINAMATH_CALUDE_cubic_equation_special_case_l4078_407885

/-- Given a cubic equation and parameters, prove it's a special case of a model equation. -/
theorem cubic_equation_special_case 
  (x a b : ℝ) 
  (h_b_nonneg : b ≥ 0) :
  6.266 * x^3 - 3 * a * x^2 + (3 * a^2 - b) * x - (a^3 - a * b) = 0 ↔ 
  ∃ (v u w : ℝ), 
    v = a ∧ 
    u = a ∧ 
    w^2 = b ∧
    6.266 * x^3 - 3 * v * x^2 + (3 * u^2 - w^2) * x - (v^3 - v * w^2) = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_special_case_l4078_407885


namespace NUMINAMATH_CALUDE_negative_exponent_division_l4078_407851

theorem negative_exponent_division (m : ℝ) :
  (-m)^7 / (-m)^2 = -m^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_exponent_division_l4078_407851


namespace NUMINAMATH_CALUDE_odd_natural_not_divisible_by_square_l4078_407835

theorem odd_natural_not_divisible_by_square (n : ℕ) : 
  Odd n → (¬(Nat.factorial (n - 1) % (n^2) = 0) ↔ Nat.Prime n ∨ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_odd_natural_not_divisible_by_square_l4078_407835


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l4078_407843

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (5 * x + 2) → y = 2 → x = -3/10 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l4078_407843


namespace NUMINAMATH_CALUDE_translated_graph_minimum_point_l4078_407855

/-- The function f representing the translated graph -/
def f (x : ℝ) : ℝ := 2 * |x - 4| - 2

/-- The minimum point of the translated graph -/
def min_point : ℝ × ℝ := (4, -2)

theorem translated_graph_minimum_point :
  ∀ x : ℝ, f x ≥ f (min_point.1) ∧ f (min_point.1) = min_point.2 :=
by sorry

end NUMINAMATH_CALUDE_translated_graph_minimum_point_l4078_407855


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l4078_407891

-- Define the repeating decimals
def repeating_decimal_2 : ℚ := 2 / 9
def repeating_decimal_02 : ℚ := 2 / 99

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_2 + repeating_decimal_02 = 8 / 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l4078_407891


namespace NUMINAMATH_CALUDE_yellow_two_days_ago_count_l4078_407869

/-- Represents the count of dandelions for a specific day -/
structure DandelionCount where
  yellow : ℕ
  white : ℕ

/-- Represents the dandelion lifecycle and counts for three consecutive days -/
structure DandelionMeadow where
  twoDaysAgo : DandelionCount
  yesterday : DandelionCount
  today : DandelionCount

/-- Theorem stating the relationship between yellow dandelions two days ago and white dandelions on subsequent days -/
theorem yellow_two_days_ago_count (meadow : DandelionMeadow) 
  (h1 : meadow.yesterday.yellow = 20)
  (h2 : meadow.yesterday.white = 14)
  (h3 : meadow.today.yellow = 15)
  (h4 : meadow.today.white = 11) :
  meadow.twoDaysAgo.yellow = meadow.yesterday.white + meadow.today.white :=
sorry

end NUMINAMATH_CALUDE_yellow_two_days_ago_count_l4078_407869


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l4078_407818

theorem consecutive_even_integers_sum (n : ℤ) : 
  n % 2 = 0 ∧ n * (n + 2) * (n + 4) = 3360 → n + (n + 2) + (n + 4) = 48 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l4078_407818


namespace NUMINAMATH_CALUDE_arccos_arcsin_equation_l4078_407821

theorem arccos_arcsin_equation : ∃ x : ℝ, Real.arccos (3 * x) - Real.arcsin (2 * x) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_arcsin_equation_l4078_407821


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l4078_407883

theorem arithmetic_mean_of_fractions (x b : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((2*x + b) / x + (2*x - b) / x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l4078_407883


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4078_407856

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- First term of the sequence -/
def a : ℚ := 1/3

/-- Common ratio of the sequence -/
def r : ℚ := 1/3

/-- Number of terms to sum -/
def n : ℕ := 8

theorem geometric_sequence_sum :
  geometric_sum a r n = 3280/6561 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4078_407856


namespace NUMINAMATH_CALUDE_vanessa_recycled_20_pounds_l4078_407884

/-- The number of pounds that earn one point -/
def pounds_per_point : ℕ := 9

/-- The number of pounds Vanessa's friends recycled -/
def friends_pounds : ℕ := 16

/-- The total number of points earned -/
def total_points : ℕ := 4

/-- Vanessa's recycled pounds -/
def vanessa_pounds : ℕ := total_points * pounds_per_point - friends_pounds

theorem vanessa_recycled_20_pounds : vanessa_pounds = 20 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_recycled_20_pounds_l4078_407884


namespace NUMINAMATH_CALUDE_expression_not_simplifiable_l4078_407836

theorem expression_not_simplifiable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a + 2*b + 2*c = 0) : 
  ∃ (f : ℝ → ℝ → ℝ → ℝ), f a b c = 
    (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) ∧
    ∀ (g : ℝ → ℝ), (∀ x y z, f x y z = g (f x y z)) → g = id := by
  sorry

end NUMINAMATH_CALUDE_expression_not_simplifiable_l4078_407836


namespace NUMINAMATH_CALUDE_distance_between_cities_l4078_407853

/-- The distance between two cities given the meeting points of three vehicles --/
theorem distance_between_cities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (horder : b < c ∧ c < a) :
  ∃ (s : ℝ), s > 0 ∧ s = Real.sqrt ((a * b * c) / (a + c - b)) ∧
  ∃ (v₁ v₂ v₃ : ℝ), v₁ > v₂ ∧ v₂ > v₃ ∧ v₃ > 0 ∧
  (v₁ / v₂ = (s + a) / (s - a)) ∧
  (v₁ / v₃ = (s + b) / (s - b)) ∧
  (v₂ / v₃ = (s + c) / (s - c)) := by
sorry

end NUMINAMATH_CALUDE_distance_between_cities_l4078_407853


namespace NUMINAMATH_CALUDE_unique_valid_swap_l4078_407898

/-- Represents a time between 6 and 7 o'clock -/
structure Time6To7 where
  hour : ℝ
  minute : ℝ
  h_range : 6 < hour ∧ hour < 7
  m_range : 0 ≤ minute ∧ minute < 60

/-- Checks if swapping hour and minute hands results in a valid time -/
def is_valid_swap (t : Time6To7) : Prop :=
  ∃ (t' : Time6To7), t.hour = t'.minute / 5 ∧ t.minute = t'.hour * 5

/-- The main theorem stating there's exactly one time where swapping hands is valid -/
theorem unique_valid_swap : ∃! (t : Time6To7), is_valid_swap t :=
sorry

end NUMINAMATH_CALUDE_unique_valid_swap_l4078_407898


namespace NUMINAMATH_CALUDE_river_depth_l4078_407806

/-- Proves that given a river with specified width, flow rate, and discharge, its depth is 2 meters -/
theorem river_depth (width : ℝ) (flow_rate : ℝ) (discharge : ℝ) : 
  width = 45 ∧ 
  flow_rate = 6 ∧ 
  discharge = 9000 → 
  discharge = width * 2 * (flow_rate * 1000 / 60) := by
  sorry

#check river_depth

end NUMINAMATH_CALUDE_river_depth_l4078_407806


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l4078_407841

theorem reciprocal_sum_of_roots (m n : ℝ) : 
  m^2 - 4*m - 2 = 0 → n^2 - 4*n - 2 = 0 → m ≠ n → 1/m + 1/n = -2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l4078_407841


namespace NUMINAMATH_CALUDE_total_units_is_34_l4078_407871

/-- The number of apartment units in two identical buildings with specific floor configurations -/
def total_apartment_units : ℕ := by
  -- Define the number of buildings
  let num_buildings : ℕ := 2

  -- Define the number of floors in each building
  let num_floors : ℕ := 4

  -- Define the number of units on the first floor
  let units_first_floor : ℕ := 2

  -- Define the number of units on each of the other floors
  let units_other_floors : ℕ := 5

  -- Calculate the total number of units in one building
  let units_per_building : ℕ := units_first_floor + (num_floors - 1) * units_other_floors

  -- Calculate the total number of units in all buildings
  exact num_buildings * units_per_building

/-- Theorem stating that the total number of apartment units is 34 -/
theorem total_units_is_34 : total_apartment_units = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_units_is_34_l4078_407871


namespace NUMINAMATH_CALUDE_samia_walking_distance_l4078_407847

/-- Proves that Samia walked 4.0 km given the journey conditions --/
theorem samia_walking_distance :
  ∀ (total_distance : ℝ) (biking_distance : ℝ),
    -- Samia's average biking speed is 15 km/h
    -- Samia bikes for 30 minutes (0.5 hours)
    biking_distance = 15 * 0.5 →
    -- The entire journey took 90 minutes (1.5 hours)
    0.5 + ((total_distance - biking_distance) / 4) = 1.5 →
    -- Prove that the walking distance is 4.0 km
    total_distance - biking_distance = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_samia_walking_distance_l4078_407847
