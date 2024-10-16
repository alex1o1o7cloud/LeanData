import Mathlib

namespace NUMINAMATH_CALUDE_yellow_pairs_count_l604_60445

theorem yellow_pairs_count (blue_count : ℕ) (yellow_count : ℕ) (total_count : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_count = 63 →
  yellow_count = 69 →
  total_count = blue_count + yellow_count →
  total_pairs = 66 →
  blue_blue_pairs = 27 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 30 ∧ 
    yellow_yellow_pairs = (yellow_count - (total_pairs - blue_blue_pairs - (yellow_count - blue_count) / 2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_yellow_pairs_count_l604_60445


namespace NUMINAMATH_CALUDE_vector_operation_result_l604_60460

/-- Prove that the result of 3 * (-3, 2, 6) + (4, -5, 2) is (-5, 1, 20) -/
theorem vector_operation_result :
  (3 : ℝ) • ((-3 : ℝ), (2 : ℝ), (6 : ℝ)) + ((4 : ℝ), (-5 : ℝ), (2 : ℝ)) = ((-5 : ℝ), (1 : ℝ), (20 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_result_l604_60460


namespace NUMINAMATH_CALUDE_range_of_m_l604_60434

def P (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m : ∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m ∈ Set.Ioc 1 2 ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l604_60434


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l604_60400

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 72 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l604_60400


namespace NUMINAMATH_CALUDE_larger_number_is_72_l604_60489

theorem larger_number_is_72 (a b : ℝ) : 
  5 * b = 6 * a ∧ b - a = 12 → b = 72 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_72_l604_60489


namespace NUMINAMATH_CALUDE_sum_of_numbers_l604_60454

theorem sum_of_numbers : 0.45 + 0.003 + (1/4 : ℚ) = 0.703 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l604_60454


namespace NUMINAMATH_CALUDE_original_number_problem_l604_60420

theorem original_number_problem (x : ℝ) : 3 * (2 * x + 9) = 69 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l604_60420


namespace NUMINAMATH_CALUDE_fraction_change_with_addition_l604_60480

theorem fraction_change_with_addition (a b n : ℕ) (h_b_pos : b > 0) :
  (a / b < 1 → (a + n) / (b + n) > a / b) ∧
  (a / b > 1 → (a + n) / (b + n) < a / b) := by
sorry

end NUMINAMATH_CALUDE_fraction_change_with_addition_l604_60480


namespace NUMINAMATH_CALUDE_problem_solution_l604_60438

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (x : ℝ) : ℝ := 2*x - 3

-- Define the interval [0, 2]
def interval : Set ℝ := Set.Icc 0 2

theorem problem_solution :
  -- 1. Tangent line equation
  (∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 2*x - y - 2 = 0) ∧
  (∀ x, x ≠ 1 → (f x - f 1) / (x - 1) < 2) ∧
  (∀ x, x ≠ 1 → (f x - f 1) / (x - 1) > 2) ∧
  
  -- 2. Maximum value on the interval
  (∀ x ∈ interval, f x ≤ 6) ∧
  (∃ x ∈ interval, f x = 6) ∧
  
  -- 3. Existence of unique x₀
  (∃! x₀, f x₀ = g x₀) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l604_60438


namespace NUMINAMATH_CALUDE_tangent_equality_solution_l604_60402

theorem tangent_equality_solution (x : Real) : 
  0 < x ∧ x < 360 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 100 ∨ x = 220 := by
sorry

end NUMINAMATH_CALUDE_tangent_equality_solution_l604_60402


namespace NUMINAMATH_CALUDE_range_of_m_l604_60466

theorem range_of_m : ∀ m : ℝ, 
  (¬∃ x : ℝ, 1 < x ∧ x < 3 ∧ x^2 - m*x - 1 = 0) ↔ 
  (m ≤ 0 ∨ m ≥ 8/3) := by sorry

end NUMINAMATH_CALUDE_range_of_m_l604_60466


namespace NUMINAMATH_CALUDE_expression_evaluation_l604_60432

theorem expression_evaluation (c : ℕ) (h : c = 4) :
  (c^c + c*(c+1)^c)^c = 5750939763536 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l604_60432


namespace NUMINAMATH_CALUDE_count_positive_area_triangles_l604_60416

/-- The number of points in each row or column of the grid -/
def gridSize : ℕ := 6

/-- The total number of points in the grid -/
def totalPoints : ℕ := gridSize * gridSize

/-- The number of ways to choose 3 points from the total points -/
def totalCombinations : ℕ := Nat.choose totalPoints 3

/-- The number of ways to choose 3 points from a single row or column -/
def lineCombo : ℕ := Nat.choose gridSize 3

/-- The number of straight lines (rows and columns) -/
def numLines : ℕ := 2 * gridSize

/-- The number of main diagonals -/
def numMainDiagonals : ℕ := 2

/-- The number of triangles with positive area on the grid -/
def positiveAreaTriangles : ℕ := 
  totalCombinations - (numLines * lineCombo) - (numMainDiagonals * lineCombo)

theorem count_positive_area_triangles : positiveAreaTriangles = 6860 := by
  sorry

end NUMINAMATH_CALUDE_count_positive_area_triangles_l604_60416


namespace NUMINAMATH_CALUDE_intern_teacher_arrangements_l604_60447

def num_teachers : ℕ := 5
def num_classes : ℕ := 3

def arrangements (n m : ℕ) : ℕ := sorry

theorem intern_teacher_arrangements :
  let remaining_teachers := num_teachers - 1
  arrangements remaining_teachers num_classes = 50 :=
by sorry

end NUMINAMATH_CALUDE_intern_teacher_arrangements_l604_60447


namespace NUMINAMATH_CALUDE_solve_for_z_l604_60463

theorem solve_for_z (x y z : ℤ) (h1 : x^2 = y - 4) (h2 : x = -6) (h3 : y = z + 2) : z = 38 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l604_60463


namespace NUMINAMATH_CALUDE_curtain_length_for_given_room_l604_60419

/-- Calculates the required curtain length in inches given the room height in feet and additional material in inches. -/
def curtain_length (room_height_feet : ℕ) (additional_inches : ℕ) : ℕ :=
  room_height_feet * 12 + additional_inches

/-- Theorem stating that for a room height of 8 feet and 5 inches of additional material, the required curtain length is 101 inches. -/
theorem curtain_length_for_given_room : curtain_length 8 5 = 101 := by
  sorry

end NUMINAMATH_CALUDE_curtain_length_for_given_room_l604_60419


namespace NUMINAMATH_CALUDE_sequence_sum_equals_63_l604_60450

theorem sequence_sum_equals_63 : 
  (Finset.range 9).sum (fun i => (i + 4) * (1 - 1 / (i + 2))) = 63 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_63_l604_60450


namespace NUMINAMATH_CALUDE_thirty_day_month_equal_tuesdays_thursdays_l604_60425

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Counts the number of occurrences of a specific day in a 30-day month starting from a given day -/
def countDayInMonth (startDay : DayOfWeek) (dayToCount : DayOfWeek) : Nat :=
  sorry

/-- Checks if a 30-day month starting from a given day has equal Tuesdays and Thursdays -/
def hasEqualTuesdaysThursdays (startDay : DayOfWeek) : Bool :=
  countDayInMonth startDay DayOfWeek.Tuesday = countDayInMonth startDay DayOfWeek.Thursday

/-- Counts the number of possible start days for a 30-day month with equal Tuesdays and Thursdays -/
def countValidStartDays : Nat :=
  sorry

theorem thirty_day_month_equal_tuesdays_thursdays :
  countValidStartDays = 4 :=
sorry

end NUMINAMATH_CALUDE_thirty_day_month_equal_tuesdays_thursdays_l604_60425


namespace NUMINAMATH_CALUDE_positive_expression_l604_60428

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 2) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  0 < y + x^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l604_60428


namespace NUMINAMATH_CALUDE_max_value_x_1plusx_3minusx_l604_60487

theorem max_value_x_1plusx_3minusx (x : ℝ) (h : x > 0) :
  x * (1 + x) * (3 - x) ≤ (70 + 26 * Real.sqrt 13) / 27 ∧
  ∃ y > 0, y * (1 + y) * (3 - y) = (70 + 26 * Real.sqrt 13) / 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_1plusx_3minusx_l604_60487


namespace NUMINAMATH_CALUDE_omega_sum_equals_one_l604_60492

theorem omega_sum_equals_one (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 + ω^48 + ω^51 + ω^54 + ω^57 + ω^60 + ω^63 = 1 := by
  sorry

end NUMINAMATH_CALUDE_omega_sum_equals_one_l604_60492


namespace NUMINAMATH_CALUDE_equation_solutions_l604_60493

theorem equation_solutions :
  (∀ x : ℝ, (x - 3)^2 + 2*x*(x - 3) = 0 ↔ x = 3 ∨ x = 1) ∧
  (∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l604_60493


namespace NUMINAMATH_CALUDE_product_sum_not_1001_l604_60403

theorem product_sum_not_1001 (a b c d : ℤ) (h1 : a + b = 100) (h2 : c + d = 100) : 
  a * b + c * d ≠ 1001 := by
sorry

end NUMINAMATH_CALUDE_product_sum_not_1001_l604_60403


namespace NUMINAMATH_CALUDE_factorization_proof_l604_60461

theorem factorization_proof (z : ℝ) :
  45 * z^12 + 180 * z^24 = 45 * z^12 * (1 + 4 * z^12) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l604_60461


namespace NUMINAMATH_CALUDE_graph_not_in_second_quadrant_implies_a_nonnegative_l604_60490

-- Define the function f(x) = x^3 - a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a

-- Define the condition that the graph does not pass through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f a x ≤ 0

-- Theorem statement
theorem graph_not_in_second_quadrant_implies_a_nonnegative (a : ℝ) :
  not_in_second_quadrant a → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_graph_not_in_second_quadrant_implies_a_nonnegative_l604_60490


namespace NUMINAMATH_CALUDE_chessboard_tiling_l604_60443

/-- A type representing a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- A type representing a tiling piece -/
structure TilingPiece :=
  (coverage : ℕ)

/-- Function to check if a chessboard can be tiled with given pieces -/
def can_tile (board : Chessboard) (piece : TilingPiece) : Prop :=
  (board.size * board.size) % piece.coverage = 0

theorem chessboard_tiling :
  (∃ (piece : TilingPiece), piece.coverage = 4 ∧ can_tile ⟨8⟩ piece) ∧
  (∀ (piece : TilingPiece), piece.coverage = 4 → ¬can_tile ⟨10⟩ piece) :=
sorry

end NUMINAMATH_CALUDE_chessboard_tiling_l604_60443


namespace NUMINAMATH_CALUDE_floor_length_is_ten_l604_60414

/-- Represents a rectangular floor with a rug -/
structure FloorWithRug where
  length : ℝ
  width : ℝ
  strip_width : ℝ
  rug_area : ℝ

/-- Theorem: Given the conditions, the floor length is 10 meters -/
theorem floor_length_is_ten (floor : FloorWithRug)
  (h1 : floor.width = 8)
  (h2 : floor.strip_width = 2)
  (h3 : floor.rug_area = 24)
  (h4 : floor.rug_area = (floor.length - 2 * floor.strip_width) * (floor.width - 2 * floor.strip_width)) :
  floor.length = 10 := by
  sorry

#check floor_length_is_ten

end NUMINAMATH_CALUDE_floor_length_is_ten_l604_60414


namespace NUMINAMATH_CALUDE_apple_crates_delivered_l604_60485

/-- The number of crates delivered to a factory, given the conditions of the apple delivery problem. -/
theorem apple_crates_delivered : ℕ := by
  -- Define the number of apples per crate
  let apples_per_crate : ℕ := 180

  -- Define the number of rotten apples
  let rotten_apples : ℕ := 160

  -- Define the number of boxes and apples per box for the remaining apples
  let num_boxes : ℕ := 100
  let apples_per_box : ℕ := 20

  -- Calculate the total number of good apples
  let good_apples : ℕ := num_boxes * apples_per_box

  -- Calculate the total number of apples delivered
  let total_apples : ℕ := good_apples + rotten_apples

  -- Calculate the number of crates delivered
  let crates_delivered : ℕ := total_apples / apples_per_crate

  -- Prove that the number of crates delivered is 12
  have : crates_delivered = 12 := by sorry

  -- Return the result
  exact 12


end NUMINAMATH_CALUDE_apple_crates_delivered_l604_60485


namespace NUMINAMATH_CALUDE_expression_is_equation_l604_60462

/-- Definition of an equation -/
def is_equation (e : Prop) : Prop :=
  ∃ (x : ℝ), ∃ (f g : ℝ → ℝ), e = (f x = g x)

/-- The expression 2x - 1 = 3 is an equation -/
theorem expression_is_equation : is_equation (∃ x : ℝ, 2 * x - 1 = 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_is_equation_l604_60462


namespace NUMINAMATH_CALUDE_projection_onto_xoy_plane_l604_60444

/-- Given a space orthogonal coordinate system Oxyz, prove that the projection
    of point P(1, 2, 3) onto the xOy plane has coordinates (1, 2, 0). -/
theorem projection_onto_xoy_plane :
  let P : ℝ × ℝ × ℝ := (1, 2, 3)
  let xoy_plane : Set (ℝ × ℝ × ℝ) := {v | v.2.2 = 0}
  let projection (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, p.2.1, 0)
  projection P ∈ xoy_plane ∧ projection P = (1, 2, 0) := by
  sorry

end NUMINAMATH_CALUDE_projection_onto_xoy_plane_l604_60444


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l604_60486

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fourth_term_of_geometric_progression :
  ∀ (a r : ℝ),
  (geometric_progression a r 1 = 6^(1/2)) →
  (geometric_progression a r 2 = 6^(1/6)) →
  (geometric_progression a r 3 = 6^(1/12)) →
  (geometric_progression a r 4 = 6^0) :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l604_60486


namespace NUMINAMATH_CALUDE_grape_juice_mixture_l604_60427

/-- Given an initial mixture with 10% grape juice, adding 20 gallons of pure grape juice
    to create a new mixture with 40% grape juice, prove that the initial mixture
    must have been 40 gallons. -/
theorem grape_juice_mixture (initial_volume : ℝ) : 
  (0.1 * initial_volume + 20) / (initial_volume + 20) = 0.4 → initial_volume = 40 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_l604_60427


namespace NUMINAMATH_CALUDE_greatest_integer_jo_l604_60451

theorem greatest_integer_jo (n : ℕ) : 
  n > 0 ∧ 
  n < 150 ∧ 
  ∃ k : ℕ, n = 9 * k - 2 ∧
  ∃ l : ℕ, n = 8 * l - 4 →
  n ≤ 124 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_jo_l604_60451


namespace NUMINAMATH_CALUDE_expense_equalization_l604_60408

/-- Given three people's expenses A, B, and C, where A < B < C, 
    prove that the amount the person who paid A needs to give to each of the others 
    to equalize the costs is (B + C - 2A) / 3 -/
theorem expense_equalization (A B C : ℝ) (h1 : A < B) (h2 : B < C) :
  let total := A + B + C
  let equal_share := total / 3
  let amount_to_give := equal_share - A
  amount_to_give = (B + C - 2 * A) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expense_equalization_l604_60408


namespace NUMINAMATH_CALUDE_linda_travel_distance_l604_60442

/-- Represents the travel data for one day -/
structure DayTravel where
  minutes_per_mile : ℕ
  distance : ℕ

/-- Calculates the distance traveled in one hour given the minutes per mile -/
def distance_traveled (minutes_per_mile : ℕ) : ℕ :=
  60 / minutes_per_mile

/-- Generates the travel data for four days -/
def generate_four_days (initial_minutes_per_mile : ℕ) : List DayTravel :=
  [0, 1, 2, 3].map (λ i =>
    { minutes_per_mile := initial_minutes_per_mile + i * 5,
      distance := distance_traveled (initial_minutes_per_mile + i * 5) })

theorem linda_travel_distance :
  ∃ (initial_minutes_per_mile : ℕ),
    let four_days := generate_four_days initial_minutes_per_mile
    four_days.length = 4 ∧
    (∀ day ∈ four_days, day.minutes_per_mile > 0 ∧ day.minutes_per_mile ≤ 60) ∧
    (∀ day ∈ four_days, day.distance > 0) ∧
    (List.sum (four_days.map (λ day => day.distance)) = 25) := by
  sorry

end NUMINAMATH_CALUDE_linda_travel_distance_l604_60442


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l604_60401

theorem hot_dogs_remainder : 25197638 % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l604_60401


namespace NUMINAMATH_CALUDE_tan_two_theta_l604_60453

theorem tan_two_theta (θ : Real) 
  (h1 : π / 2 < θ ∧ θ < π) -- θ is an obtuse angle
  (h2 : Real.cos (2 * θ) - Real.sin (2 * θ) = (Real.cos θ)^2) :
  Real.tan (2 * θ) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_two_theta_l604_60453


namespace NUMINAMATH_CALUDE_total_dog_legs_l604_60471

/-- Proves that the total number of dog legs on a street is 400, given the conditions. -/
theorem total_dog_legs (total_animals : ℕ) (cat_fraction : ℚ) (dog_legs : ℕ) : 
  total_animals = 300 →
  cat_fraction = 2/3 →
  dog_legs = 4 →
  (total_animals * (1 - cat_fraction) : ℚ).num * dog_legs = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_dog_legs_l604_60471


namespace NUMINAMATH_CALUDE_outfit_count_l604_60418

/-- The number of shirts -/
def num_shirts : ℕ := 8

/-- The number of ties that can be paired with each shirt -/
def ties_per_shirt : ℕ := 4

/-- The total number of shirt-and-tie outfits -/
def total_outfits : ℕ := num_shirts * ties_per_shirt

theorem outfit_count : total_outfits = 32 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l604_60418


namespace NUMINAMATH_CALUDE_bench_press_theorem_l604_60467

def bench_press_problem (initial_weight : ℝ) (injury_reduction : ℝ) (training_multiplier : ℝ) : Prop :=
  let after_injury := initial_weight * (1 - injury_reduction)
  let final_weight := after_injury * training_multiplier
  final_weight = 300

theorem bench_press_theorem :
  bench_press_problem 500 0.8 3 := by
  sorry

end NUMINAMATH_CALUDE_bench_press_theorem_l604_60467


namespace NUMINAMATH_CALUDE_quarters_collected_per_month_l604_60446

/-- Represents the number of quarters Phil collected each month during the second year -/
def quarters_per_month : ℕ := sorry

/-- The initial number of quarters Phil had -/
def initial_quarters : ℕ := 50

/-- The number of quarters Phil had after doubling his initial collection -/
def after_doubling : ℕ := 2 * initial_quarters

/-- The number of quarters Phil collected in the third year -/
def third_year_quarters : ℕ := 4

/-- The number of quarters Phil had before losing some -/
def before_loss : ℕ := 140

/-- The number of quarters Phil had after losing some -/
def after_loss : ℕ := 105

/-- Theorem stating that the number of quarters collected each month in the second year is 3 -/
theorem quarters_collected_per_month : 
  quarters_per_month = 3 ∧
  after_doubling + 12 * quarters_per_month + third_year_quarters = before_loss ∧
  before_loss * 3 = after_loss * 4 := by sorry

end NUMINAMATH_CALUDE_quarters_collected_per_month_l604_60446


namespace NUMINAMATH_CALUDE_linear_function_property_l604_60429

theorem linear_function_property :
  ∀ x y : ℝ, y = -2 * x + 1 → x > (1/2 : ℝ) → y < 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l604_60429


namespace NUMINAMATH_CALUDE_not_prime_a_l604_60498

theorem not_prime_a (a b : ℕ+) (h : ∃ k : ℤ, k * (b.val^4 + 3*b.val^2 + 4) = 5*a.val^4 + a.val^2) : 
  ¬ Nat.Prime a.val := by
sorry

end NUMINAMATH_CALUDE_not_prime_a_l604_60498


namespace NUMINAMATH_CALUDE_helly_theorem_2d_l604_60437

-- Define a type for points in the plane
variable (Point : Type)

-- Define a type for convex sets in the plane
variable (ConvexSet : Type)

-- Define a function to check if a point is in a convex set
variable (isIn : Point → ConvexSet → Prop)

-- Define a function to check if a set is convex
variable (isConvex : ConvexSet → Prop)

-- Define the theorem
theorem helly_theorem_2d 
  (n : ℕ) 
  (h_n : n ≥ 4) 
  (A : Fin n → ConvexSet) 
  (h_convex : ∀ i, isConvex (A i)) 
  (h_intersection : ∀ i j k, ∃ p, isIn p (A i) ∧ isIn p (A j) ∧ isIn p (A k)) :
  ∃ p, ∀ i, isIn p (A i) :=
sorry

end NUMINAMATH_CALUDE_helly_theorem_2d_l604_60437


namespace NUMINAMATH_CALUDE_katy_brownies_l604_60436

theorem katy_brownies (x : ℕ) : 
  x + 2 * x = 15 → x = 5 := by sorry

end NUMINAMATH_CALUDE_katy_brownies_l604_60436


namespace NUMINAMATH_CALUDE_original_number_proof_l604_60475

theorem original_number_proof :
  ∃ x : ℝ, x * 1.1 = 550 ∧ x = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l604_60475


namespace NUMINAMATH_CALUDE_largest_number_l604_60491

theorem largest_number (π : ℝ) (h : 3 < π ∧ π < 4) : 
  π = max π (max 3 (max (1 - π) (-π^2))) := by
sorry

end NUMINAMATH_CALUDE_largest_number_l604_60491


namespace NUMINAMATH_CALUDE_cut_string_theorem_l604_60456

/-- Represents the number of pieces resulting from cutting a string at all points marked by two different equal-spacing schemes. -/
def cut_string_pieces (total_length : ℝ) (divisions1 divisions2 : ℕ) : ℕ :=
  let marks1 := divisions1 - 1
  let marks2 := divisions2 - 1
  marks1 + marks2 + 1

/-- Theorem stating that cutting a string at points marked for 9 equal pieces and 8 equal pieces results in 16 pieces. -/
theorem cut_string_theorem : cut_string_pieces 1 9 8 = 16 := by
  sorry

#eval cut_string_pieces 1 9 8

end NUMINAMATH_CALUDE_cut_string_theorem_l604_60456


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l604_60476

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem fiftieth_term_of_sequence :
  let a₁ := 3
  let d := 6
  let n := 50
  arithmetic_sequence a₁ d n = 297 := by sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l604_60476


namespace NUMINAMATH_CALUDE_total_disks_is_126_l604_60497

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- The ratio of blue to yellow to green disks -/
def diskRatio : DiskBag → Prop
  | ⟨b, y, g⟩ => ∃ (x : ℕ), b = 3 * x ∧ y = 7 * x ∧ g = 8 * x

/-- The difference between green and blue disks -/
def greenBlueDifference (bag : DiskBag) : Prop :=
  bag.green = bag.blue + 35

/-- The total number of disks in the bag -/
def totalDisks (bag : DiskBag) : ℕ :=
  bag.blue + bag.yellow + bag.green

/-- Theorem: Given the conditions, the total number of disks is 126 -/
theorem total_disks_is_126 (bag : DiskBag) 
  (h1 : diskRatio bag) 
  (h2 : greenBlueDifference bag) : 
  totalDisks bag = 126 := by
  sorry

end NUMINAMATH_CALUDE_total_disks_is_126_l604_60497


namespace NUMINAMATH_CALUDE_fraction_equality_l604_60484

theorem fraction_equality (a b : ℚ) (h : b ≠ 0) (h1 : a / b = 2 / 3) :
  (a - b) / b = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l604_60484


namespace NUMINAMATH_CALUDE_parallel_line_slope_parallel_line_y_intercept_exists_l604_60409

/-- Given a line parallel to 3x - 6y = 12, prove its slope is 1/2 --/
theorem parallel_line_slope (a b c : ℝ) (h : ∃ k : ℝ, a * x + b * y = c ∧ k ≠ 0 ∧ 3 * (a / b) = -1 / 2) :
  a / b = 1 / 2 := by sorry

/-- The y-intercept of a line parallel to 3x - 6y = 12 can be any real number --/
theorem parallel_line_y_intercept_exists : ∀ k : ℝ, ∃ (a b c : ℝ), a * x + b * y = c ∧ a / b = 1 / 2 ∧ c / b = k := by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_parallel_line_y_intercept_exists_l604_60409


namespace NUMINAMATH_CALUDE_border_length_is_even_l604_60474

/-- Represents a domino on the board -/
inductive Domino
| Horizontal
| Vertical

/-- Represents the board -/
def Board := Fin 2010 → Fin 2011 → Domino

/-- The border length between horizontal and vertical dominoes -/
def borderLength (board : Board) : ℕ := sorry

/-- Theorem stating that the border length is even -/
theorem border_length_is_even (board : Board) : 
  Even (borderLength board) := by sorry

end NUMINAMATH_CALUDE_border_length_is_even_l604_60474


namespace NUMINAMATH_CALUDE_weaving_increase_l604_60481

/-- The sum of an arithmetic sequence with n terms, first term a, and common difference d -/
def arithmetic_sum (n : ℕ) (a d : ℚ) : ℚ := n * a + n * (n - 1) / 2 * d

/-- The problem of finding the daily increase in weaving -/
theorem weaving_increase (a₁ : ℚ) (n : ℕ) (S : ℚ) (h1 : a₁ = 5) (h2 : n = 30) (h3 : S = 390) :
  ∃ d : ℚ, arithmetic_sum n a₁ d = S ∧ d = 16/29 := by
  sorry

end NUMINAMATH_CALUDE_weaving_increase_l604_60481


namespace NUMINAMATH_CALUDE_sin_2x_equals_cos_2x_minus_pi_over_4_l604_60426

theorem sin_2x_equals_cos_2x_minus_pi_over_4 (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (2 * (x - π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_equals_cos_2x_minus_pi_over_4_l604_60426


namespace NUMINAMATH_CALUDE_find_number_l604_60417

theorem find_number (x : ℚ) : ((x / 9) - 13) / 7 - 8 = 13 → x = 1440 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l604_60417


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l604_60449

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (l m n : Line) (α : Plane) 
  (h1 : intersect l m)
  (h2 : parallel l α)
  (h3 : parallel m α)
  (h4 : perpendicular n l)
  (h5 : perpendicular n m) :
  perpendicularToPlane n α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l604_60449


namespace NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l604_60457

theorem product_divisible_by_sum_implies_inequality (m n : ℕ+) 
  (h : (m + n : ℕ) ∣ (m * n : ℕ)) : 
  (m : ℕ) + n ≤ n^2 := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l604_60457


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_sum_l604_60423

theorem binomial_coefficient_ratio_sum (n k : ℕ) : 
  (2 : ℚ) / 5 = (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) →
  (∃ m l : ℕ, m ≠ l ∧ (m = n ∧ l = k ∨ l = n ∧ m = k) ∧ m + l = 23) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_sum_l604_60423


namespace NUMINAMATH_CALUDE_album_difference_l604_60410

/-- Represents the number of albums each person has -/
structure AlbumCounts where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ

/-- The conditions of the problem -/
def problem_conditions (counts : AlbumCounts) : Prop :=
  counts.miriam = 5 * counts.katrina ∧
  counts.katrina = 6 * counts.bridget ∧
  counts.bridget < counts.adele ∧
  counts.adele + counts.bridget + counts.katrina + counts.miriam = 585 ∧
  counts.adele = 30

/-- The theorem to be proved -/
theorem album_difference (counts : AlbumCounts) 
  (h : problem_conditions counts) : 
  counts.adele - counts.bridget = 15 := by
  sorry

end NUMINAMATH_CALUDE_album_difference_l604_60410


namespace NUMINAMATH_CALUDE_inequalities_theorem_l604_60473

theorem inequalities_theorem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > -a) (h4 : c < d) (h5 : d < 0) : 
  (a * d ≤ b * c) ∧ 
  (a / d + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l604_60473


namespace NUMINAMATH_CALUDE_trapezoid_max_area_l604_60469

theorem trapezoid_max_area (r d : ℝ) (hr : r = 13) (hd : d = 5) :
  let diag := 2 * Real.sqrt (r^2 - d^2)
  ∃ (area : ℝ), area ≤ (diag^2) / 2 ∧
    ∀ (a : ℝ), a ≤ (diag^2) / 2 → a ≤ area :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_max_area_l604_60469


namespace NUMINAMATH_CALUDE_walnut_price_l604_60430

/-- Represents the price of a nut in Forints -/
structure NutPrice where
  price : ℕ
  is_two_digit : price ≥ 10 ∧ price < 100

/-- Checks if two digits are consecutive -/
def consecutive_digits (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (a = b + 1 ∨ b = a + 1) ∧ n = 10 * a + b

/-- Checks if two prices are digit swaps of each other -/
def is_digit_swap (p1 p2 : NutPrice) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ p1.price = 10 * a + b ∧ p2.price = 10 * b + a

theorem walnut_price (walnut hazelnut : NutPrice)
  (total_value : ℕ) (total_weight : ℕ)
  (h1 : total_value = 1978)
  (h2 : total_weight = 55)
  (h3 : walnut.price > hazelnut.price)
  (h4 : is_digit_swap walnut hazelnut)
  (h5 : consecutive_digits walnut.price)
  (h6 : ∃ (w h : ℕ), w * walnut.price + h * hazelnut.price = total_value ∧ w + h = total_weight) :
  walnut.price = 43 := by
  sorry

end NUMINAMATH_CALUDE_walnut_price_l604_60430


namespace NUMINAMATH_CALUDE_pencil_color_fraction_l604_60499

theorem pencil_color_fraction (total_length : ℝ) (green_fraction : ℝ) (white_fraction : ℝ) :
  total_length = 2 →
  green_fraction = 7 / 10 →
  white_fraction = 1 / 2 →
  (total_length - green_fraction * total_length) / 2 = 
  (1 - white_fraction) * (total_length - green_fraction * total_length) :=
by sorry

end NUMINAMATH_CALUDE_pencil_color_fraction_l604_60499


namespace NUMINAMATH_CALUDE_motorcycle_trip_time_difference_specific_motorcycle_problem_l604_60478

/-- Given a motorcycle traveling at a constant speed, prove the time difference between two trips -/
theorem motorcycle_trip_time_difference (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ) : 
  speed > 0 → 
  distance1 > 0 →
  distance2 > 0 →
  distance1 > distance2 →
  (distance1 / speed - distance2 / speed) * 60 = (distance1 - distance2) / speed * 60 := by
  sorry

/-- Specific instance of the theorem for the given problem -/
theorem specific_motorcycle_problem : 
  (400 / 40 - 360 / 40) * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_trip_time_difference_specific_motorcycle_problem_l604_60478


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l604_60440

theorem contrapositive_equivalence (x y : ℝ) :
  (¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ↔ (x^2 + y^2 = 0 → x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l604_60440


namespace NUMINAMATH_CALUDE_odd_function_equivalence_l604_60452

theorem odd_function_equivalence (f : ℝ → ℝ) : 
  (∀ x, f x + f (-x) = 0) ↔ (∀ x, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_odd_function_equivalence_l604_60452


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l604_60422

-- Define the complex number z
def z : ℂ := (3 + Complex.I) * (1 - Complex.I)

-- Theorem stating that z is in the fourth quadrant
theorem z_in_fourth_quadrant :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l604_60422


namespace NUMINAMATH_CALUDE_grocery_shop_sales_l604_60412

theorem grocery_shop_sales (sales1 sales3 sales4 sales5 sales6 : ℕ) 
  (h1 : sales1 = 6335)
  (h3 : sales3 = 7230)
  (h4 : sales4 = 6562)
  (h5 : sales5 = 6855)
  (h6 : sales6 = 5091)
  (h_avg : (sales1 + sales3 + sales4 + sales5 + sales6 + 6927) / 6 = 6500) :
  ∃ sales2 : ℕ, sales2 = 6927 := by
  sorry

end NUMINAMATH_CALUDE_grocery_shop_sales_l604_60412


namespace NUMINAMATH_CALUDE_angle_difference_range_l604_60488

theorem angle_difference_range (α β : Real) (h1 : -π/2 < α) (h2 : α < β) (h3 : β < π/2) :
  ∃ (x : Real), -π < x ∧ x < 0 ∧ x = α - β :=
sorry

end NUMINAMATH_CALUDE_angle_difference_range_l604_60488


namespace NUMINAMATH_CALUDE_five_valid_configurations_l604_60433

/-- Represents a square in the figure -/
structure Square :=
  (label : Char)

/-- Represents the L-shaped figure -/
structure LShape :=
  (squares : Finset Square)
  (size : Nat)
  (h_size : size = 4)

/-- Represents the set of additional squares -/
structure AdditionalSquares :=
  (squares : Finset Square)
  (size : Nat)
  (h_size : size = 8)

/-- Represents a configuration formed by adding one square to the L-shape -/
structure Configuration :=
  (base : LShape)
  (added : Square)

/-- Predicate to determine if a configuration can be folded into a topless cubical box -/
def canFoldIntoCube (config : Configuration) : Prop :=
  sorry

/-- The main theorem stating that exactly 5 configurations can be folded into a topless cubical box -/
theorem five_valid_configurations
  (l : LShape)
  (extras : AdditionalSquares) :
  ∃! (validConfigs : Finset Configuration),
    validConfigs.card = 5 ∧
    ∀ (config : Configuration),
      config ∈ validConfigs ↔
        (config.base = l ∧
         config.added ∈ extras.squares ∧
         canFoldIntoCube config) :=
sorry

end NUMINAMATH_CALUDE_five_valid_configurations_l604_60433


namespace NUMINAMATH_CALUDE_fb_is_80_l604_60458

/-- A right-angled triangle ABC with a point F on BC -/
structure TriangleABCF where
  /-- The length of side AB -/
  ab : ℝ
  /-- The length of side AC -/
  ac : ℝ
  /-- The length of side BC -/
  bc : ℝ
  /-- The length of BF -/
  bf : ℝ
  /-- The length of CF -/
  cf : ℝ
  /-- AB is 120 meters -/
  hab : ab = 120
  /-- AC is 160 meters -/
  hac : ac = 160
  /-- ABC is a right-angled triangle -/
  hright : ab^2 + ac^2 = bc^2
  /-- F is on BC -/
  hf_on_bc : bf + cf = bc
  /-- Jack and Jill jog the same distance -/
  heq_dist : ac + cf = ab + bf

/-- The main theorem: FB is 80 meters -/
theorem fb_is_80 (t : TriangleABCF) : t.bf = 80 := by
  sorry

end NUMINAMATH_CALUDE_fb_is_80_l604_60458


namespace NUMINAMATH_CALUDE_integral_ln_sin_x_l604_60405

theorem integral_ln_sin_x (x : ℝ) : 
  ∫ x in (0)..(π/2), Real.log (Real.sin x) = -(π/2) * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_integral_ln_sin_x_l604_60405


namespace NUMINAMATH_CALUDE_lottery_blank_probability_l604_60406

theorem lottery_blank_probability :
  let num_prizes : ℕ := 10
  let num_blanks : ℕ := 25
  let total_outcomes : ℕ := num_prizes + num_blanks
  (num_blanks : ℚ) / (total_outcomes : ℚ) = 5 / 7 :=
by sorry

end NUMINAMATH_CALUDE_lottery_blank_probability_l604_60406


namespace NUMINAMATH_CALUDE_least_integer_y_l604_60441

theorem least_integer_y : ∃ y : ℤ, (∀ z : ℤ, |3*z - 4| ≤ 25 → y ≤ z) ∧ |3*y - 4| ≤ 25 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_y_l604_60441


namespace NUMINAMATH_CALUDE_triangle_side_length_l604_60407

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = 10 →
  b = 5 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l604_60407


namespace NUMINAMATH_CALUDE_squareable_numbers_l604_60421

def isSquareable (n : ℕ) : Prop :=
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ i : Fin n, ∃ k : ℕ, (p i).val + 1 + i.val = k^2

theorem squareable_numbers : 
  (¬ isSquareable 7) ∧ 
  (isSquareable 9) ∧ 
  (¬ isSquareable 11) ∧ 
  (isSquareable 15) :=
sorry

end NUMINAMATH_CALUDE_squareable_numbers_l604_60421


namespace NUMINAMATH_CALUDE_smallest_irrational_distance_points_theorem_l604_60477

/-- The smallest number of points in ℝⁿ such that every point of ℝⁿ is an irrational distance from at least one of the points -/
def smallest_irrational_distance_points (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3

/-- Theorem stating the smallest number of points in ℝⁿ such that every point of ℝⁿ is an irrational distance from at least one of the points -/
theorem smallest_irrational_distance_points_theorem (n : ℕ) (hn : n > 0) :
  smallest_irrational_distance_points n = if n = 1 then 2 else 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_irrational_distance_points_theorem_l604_60477


namespace NUMINAMATH_CALUDE_abc_product_l604_60459

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) :
  a * b * c = 720 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l604_60459


namespace NUMINAMATH_CALUDE_complex_sum_equality_l604_60464

theorem complex_sum_equality : 
  let z₁ : ℂ := -1/2 + 3/4 * I
  let z₂ : ℂ := 7/3 - 5/6 * I
  z₁ + z₂ = 11/6 - 1/12 * I := by
sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l604_60464


namespace NUMINAMATH_CALUDE_comparison_of_roots_l604_60431

theorem comparison_of_roots : 
  let a := (16 : ℝ) ^ (1/4)
  let b := (27 : ℝ) ^ (1/3)
  let c := (25 : ℝ) ^ (1/2)
  let d := (32 : ℝ) ^ (1/5)
  (c > b ∧ b > a ∧ b > d) := by sorry

end NUMINAMATH_CALUDE_comparison_of_roots_l604_60431


namespace NUMINAMATH_CALUDE_triangle_circles_radius_sum_l604_60482

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ) (AC : ℝ) (BC : ℝ)

/-- Circle with given radius -/
structure Circle :=
  (radius : ℝ)

/-- Represents the radius of circle Q in the form m - n√k -/
structure RadiusForm :=
  (m : ℕ) (n : ℕ) (k : ℕ)

/-- Main theorem statement -/
theorem triangle_circles_radius_sum (ABC : Triangle) (P Q : Circle) (r : RadiusForm) :
  ABC.AB = 130 →
  ABC.AC = 130 →
  ABC.BC = 78 →
  P.radius = 25 →
  -- Circle P is tangent to AC and BC
  -- Circle Q is externally tangent to P and tangent to AB and BC
  -- No point of circle Q lies outside of triangle ABC
  Q.radius = r.m - r.n * Real.sqrt r.k →
  r.m > 0 →
  r.n > 0 →
  r.k > 0 →
  -- k is the product of distinct primes
  r.m + r.n * r.k = 131 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circles_radius_sum_l604_60482


namespace NUMINAMATH_CALUDE_three_hour_charge_l604_60483

/-- Represents the pricing structure and total charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ  -- Price of the first hour
  additional_hour : ℕ  -- Price of each additional hour
  first_hour_premium : first_hour = additional_hour + 30  -- First hour costs $30 more
  five_hour_total : first_hour + 4 * additional_hour = 400  -- Total for 5 hours is $400

/-- Theorem stating that given the pricing structure, the total charge for 3 hours is $252. -/
theorem three_hour_charge (p : TherapyPricing) : 
  p.first_hour + 2 * p.additional_hour = 252 := by
  sorry


end NUMINAMATH_CALUDE_three_hour_charge_l604_60483


namespace NUMINAMATH_CALUDE_fraction_addition_subtraction_l604_60448

theorem fraction_addition_subtraction :
  (1 / 4 : ℚ) + (3 / 8 : ℚ) - (1 / 8 : ℚ) = (1 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_subtraction_l604_60448


namespace NUMINAMATH_CALUDE_debbie_number_l604_60465

def alice_skips (n : ℕ) : Bool :=
  n % 4 = 3

def barbara_says (n : ℕ) : Bool :=
  alice_skips n ∧ ¬(n % 12 = 7)

def candice_says (n : ℕ) : Bool :=
  alice_skips n ∧ barbara_says n ∧ ¬(n % 24 = 11)

def debbie_says (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 1200 ∧ ¬(alice_skips n) ∧ ¬(barbara_says n) ∧ ¬(candice_says n)

theorem debbie_number : ∃! n : ℕ, debbie_says n ∧ n = 1187 := by
  sorry

end NUMINAMATH_CALUDE_debbie_number_l604_60465


namespace NUMINAMATH_CALUDE_mauras_seashells_l604_60404

/-- Represents the number of seashells Maura found during her summer vacation. -/
def total_seashells : ℕ := 75

/-- Represents the number of seashells Maura kept after giving some to her sister. -/
def kept_seashells : ℕ := 57

/-- Represents the number of seashells Maura gave to her sister. -/
def given_seashells : ℕ := 18

/-- Represents the number of days Maura's family stayed at the beach house. -/
def beach_days : ℕ := 21

/-- Proves that the total number of seashells Maura found is equal to the sum of
    the seashells she kept and the seashells she gave away. -/
theorem mauras_seashells : total_seashells = kept_seashells + given_seashells := by
  sorry

end NUMINAMATH_CALUDE_mauras_seashells_l604_60404


namespace NUMINAMATH_CALUDE_linear_equation_solution_l604_60435

theorem linear_equation_solution (x y : ℝ) : x - 3 * y = 4 ↔ x = 1 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l604_60435


namespace NUMINAMATH_CALUDE_three_digit_square_sum_l604_60495

theorem three_digit_square_sum (N : ℕ) : 
  (100 ≤ N ∧ N ≤ 999) →
  (∃ (a b c : ℕ), 
    0 ≤ a ∧ a ≤ 9 ∧ a ≠ 0 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    N = 100 * a + 10 * b + c ∧
    N = 11 * (a^2 + b^2 + c^2)) →
  (N = 550 ∨ N = 803) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_square_sum_l604_60495


namespace NUMINAMATH_CALUDE_total_people_count_l604_60468

theorem total_people_count (cannoneers : ℕ) (women : ℕ) (men : ℕ) : 
  women = 2 * cannoneers →
  cannoneers = 63 →
  men = 2 * women →
  cannoneers + women + men = 378 := by
sorry

end NUMINAMATH_CALUDE_total_people_count_l604_60468


namespace NUMINAMATH_CALUDE_gem_purchase_theorem_l604_60472

/-- Proves that given the conditions of gem purchasing and bonuses, 
    the amount spent to obtain 30,000 gems is $250. -/
theorem gem_purchase_theorem (gems_per_dollar : ℕ) (bonus_rate : ℚ) (final_gems : ℕ) : 
  gems_per_dollar = 100 →
  bonus_rate = 1/5 →
  final_gems = 30000 →
  (final_gems : ℚ) / (gems_per_dollar : ℚ) / (1 + bonus_rate) = 250 := by
  sorry

end NUMINAMATH_CALUDE_gem_purchase_theorem_l604_60472


namespace NUMINAMATH_CALUDE_quoted_price_calculation_l604_60424

/-- Calculates the quoted price of shares given investment details -/
theorem quoted_price_calculation (investment : ℚ) (face_value : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : 
  investment = 4455 ∧ 
  face_value = 10 ∧ 
  dividend_rate = 12 / 100 ∧ 
  annual_income = 648 → 
  (investment / (annual_income / (dividend_rate * face_value))) = 33 / 4 :=
by sorry

end NUMINAMATH_CALUDE_quoted_price_calculation_l604_60424


namespace NUMINAMATH_CALUDE_unique_root_when_b_zero_c_positive_odd_function_when_c_zero_symmetric_about_zero_one_iff_c_one_l604_60496

-- Define the function f
def f (x b c : ℝ) : ℝ := |x| * x + b * x + c

-- Theorem 1: When b=0 and c>0, f(x) = 0 has only one root
theorem unique_root_when_b_zero_c_positive (c : ℝ) (hc : c > 0) :
  ∃! x : ℝ, f x 0 c = 0 :=
sorry

-- Theorem 2: When c=0, y=f(x) is an odd function
theorem odd_function_when_c_zero (b : ℝ) :
  ∀ x : ℝ, f (-x) b 0 = -f x b 0 :=
sorry

-- Theorem 3: The graph of y=f(x) is symmetric about (0,1) iff c=1
theorem symmetric_about_zero_one_iff_c_one (b : ℝ) :
  (∀ x : ℝ, f x b 1 = 2 - f (-x) b 1) ↔ c = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_root_when_b_zero_c_positive_odd_function_when_c_zero_symmetric_about_zero_one_iff_c_one_l604_60496


namespace NUMINAMATH_CALUDE_larger_part_of_sum_and_product_l604_60411

theorem larger_part_of_sum_and_product (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ x + y = 20 ∧ x * y = 96 → max x y = 12 := by
  sorry

end NUMINAMATH_CALUDE_larger_part_of_sum_and_product_l604_60411


namespace NUMINAMATH_CALUDE_samuel_coaching_discontinue_date_l604_60479

/-- Represents a date in a non-leap year -/
structure Date where
  month : Nat
  day : Nat

/-- Calculates the number of days from January 1st to a given date in a non-leap year -/
def daysFromNewYear (d : Date) : Nat :=
  sorry

/-- The date Samuel discontinued coaching -/
def discontinueDate : Date :=
  { month := 11, day := 3 }

theorem samuel_coaching_discontinue_date 
  (totalCost : Nat) 
  (dailyCharge : Nat) 
  (nonLeapYear : Bool) :
  totalCost = 7038 →
  dailyCharge = 23 →
  nonLeapYear = true →
  daysFromNewYear discontinueDate = totalCost / dailyCharge :=
by sorry

end NUMINAMATH_CALUDE_samuel_coaching_discontinue_date_l604_60479


namespace NUMINAMATH_CALUDE_journey_speed_l604_60415

theorem journey_speed (D : ℝ) (V : ℝ) (h1 : D > 0) (h2 : V > 0) : 
  (2 * D) / (D / V + D / 30) = 40 → V = 60 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_l604_60415


namespace NUMINAMATH_CALUDE_base_eight_representation_l604_60455

theorem base_eight_representation (a b c d e f : ℕ) 
  (h1 : 208208 = 8^5 * a + 8^4 * b + 8^3 * c + 8^2 * d + 8 * e + f)
  (h2 : a ≤ 7 ∧ b ≤ 7 ∧ c ≤ 7 ∧ d ≤ 7 ∧ e ≤ 7 ∧ f ≤ 7) :
  a * b * c + d * e * f = 72 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_representation_l604_60455


namespace NUMINAMATH_CALUDE_trigonometric_identity_l604_60439

theorem trigonometric_identity (α : ℝ) :
  2 * (Real.sin (3 * π - 2 * α))^2 * (Real.cos (5 * π + 2 * α))^2 =
  1/4 - 1/4 * Real.sin (5/2 * π - 8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l604_60439


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l604_60470

theorem lcm_hcf_problem (a b : ℕ+) 
  (h1 : Nat.lcm a b = 2310)
  (h2 : Nat.gcd a b = 30)
  (h3 : a = 330) : 
  b = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l604_60470


namespace NUMINAMATH_CALUDE_clubsuit_not_commutative_l604_60494

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Define the clubsuit operation
def clubsuit (x y : ℝ) : ℝ := heartsuit x (y + 1)

-- Theorem stating that the equality is false
theorem clubsuit_not_commutative : ¬ (∀ x y : ℝ, clubsuit x y = clubsuit y x) := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_not_commutative_l604_60494


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l604_60413

/-- A function that satisfies f(xy) = f(x) + f(y) + 1 for all x and y -/
def f (x : ℝ) : ℝ := -1

/-- Theorem stating that f satisfies the given functional equation -/
theorem f_satisfies_equation (x y : ℝ) : f (x * y) = f x + f y + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l604_60413
