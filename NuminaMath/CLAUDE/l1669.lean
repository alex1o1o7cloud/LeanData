import Mathlib

namespace NUMINAMATH_CALUDE_prime_triplet_divisibility_l1669_166928

theorem prime_triplet_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧
  (q * r - 1) % p = 0 ∧
  (p * r - 1) % q = 0 ∧
  (p * q - 1) % r = 0 →
  ({p, q, r} : Set ℕ) = {2, 3, 5} :=
by sorry

end NUMINAMATH_CALUDE_prime_triplet_divisibility_l1669_166928


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1669_166979

theorem fraction_to_decimal (numerator denominator : ℕ) (decimal : ℚ) : 
  numerator = 16 → denominator = 50 → decimal = 0.32 → 
  (numerator : ℚ) / (denominator : ℚ) = decimal :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1669_166979


namespace NUMINAMATH_CALUDE_train_crossing_time_l1669_166938

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length platform_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 250 →
  platform_length = 200 →
  train_speed_kmph = 90 →
  (train_length + platform_length) / (train_speed_kmph * 1000 / 3600) = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1669_166938


namespace NUMINAMATH_CALUDE_friday_temperature_l1669_166983

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem friday_temperature 
  (M T W Th F : ℤ) 
  (avg_mon_to_thu : (M + T + W + Th) / 4 = 48)
  (avg_tue_to_fri : (T + W + Th + F) / 4 = 46)
  (monday_temp : M = 43)
  (all_odd : is_odd M ∧ is_odd T ∧ is_odd W ∧ is_odd Th ∧ is_odd F) :
  F = 35 := by
  sorry

end NUMINAMATH_CALUDE_friday_temperature_l1669_166983


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1669_166965

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^3 + 3•X^2 - 4) = (X^2 + X - 2) * q + 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1669_166965


namespace NUMINAMATH_CALUDE_painter_problem_l1669_166910

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculates the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Proves that for the given scenario, the time to paint the remaining rooms is 32 hours. -/
theorem painter_problem :
  let total_rooms : ℕ := 9
  let time_per_room : ℕ := 8
  let painted_rooms : ℕ := 5
  time_to_paint_remaining total_rooms time_per_room painted_rooms = 32 :=
by
  sorry


end NUMINAMATH_CALUDE_painter_problem_l1669_166910


namespace NUMINAMATH_CALUDE_max_intersection_points_ellipse_three_lines_l1669_166977

/-- Represents a line in a 2D plane -/
structure Line :=
  (a b c : ℝ)

/-- Represents an ellipse in a 2D plane -/
structure Ellipse :=
  (a b c d e f : ℝ)

/-- Counts the maximum number of intersection points between an ellipse and a line -/
def maxIntersectionPointsEllipseLine : ℕ := 2

/-- Counts the maximum number of intersection points between two distinct lines -/
def maxIntersectionPointsTwoLines : ℕ := 1

/-- The number of distinct pairs of lines given 3 lines -/
def numLinePairs : ℕ := 3

/-- The number of lines -/
def numLines : ℕ := 3

theorem max_intersection_points_ellipse_three_lines :
  ∀ (e : Ellipse) (l₁ l₂ l₃ : Line),
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ →
    (maxIntersectionPointsEllipseLine * numLines) + 
    (maxIntersectionPointsTwoLines * numLinePairs) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_ellipse_three_lines_l1669_166977


namespace NUMINAMATH_CALUDE_deepthi_material_usage_l1669_166972

theorem deepthi_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 4 / 17)
  (h2 : material2 = 3 / 10)
  (h3 : leftover = 9 / 30) :
  material1 + material2 - leftover = 4 / 17 := by
sorry

end NUMINAMATH_CALUDE_deepthi_material_usage_l1669_166972


namespace NUMINAMATH_CALUDE_f_is_odd_l1669_166994

noncomputable def f (x : ℝ) : ℝ := Real.log ((2 / (1 - x)) - 1) / Real.log 10

theorem f_is_odd : ∀ x : ℝ, x ≠ 1 → f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_l1669_166994


namespace NUMINAMATH_CALUDE_linear_function_two_points_l1669_166907

/-- A linear function passing through exactly two of three given points -/
theorem linear_function_two_points :
  ∃ (f : ℝ → ℝ) (a b : ℝ), 
    (∀ x, f x = a * x + b) ∧ 
    (f 0 = 0 ∧ f 1 = 1 ∧ f 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_two_points_l1669_166907


namespace NUMINAMATH_CALUDE_distinct_residues_count_l1669_166918

theorem distinct_residues_count (n m : ℕ) (a b : ℕ → ℝ) :
  (∀ j ∈ Finset.range n, ∀ k ∈ Finset.range m, a j + b k ≠ 1) →
  (∀ j ∈ Finset.range (n-1), a j < a (j+1)) →
  (∀ k ∈ Finset.range (m-1), b k < b (k+1)) →
  a 0 = 0 →
  b 0 = 0 →
  (∀ j ∈ Finset.range n, 0 < a j ∧ a j < 1) →
  (∀ k ∈ Finset.range m, 0 < b k ∧ b k < 1) →
  Finset.card (Finset.image (λ (p : ℕ × ℕ) => (a p.1 + b p.2) % 1) (Finset.product (Finset.range n) (Finset.range m))) ≥ m + n - 1 :=
by sorry


end NUMINAMATH_CALUDE_distinct_residues_count_l1669_166918


namespace NUMINAMATH_CALUDE_ten_thousand_squared_l1669_166988

theorem ten_thousand_squared (x : ℕ) (h : x = 10^4) : x * x = 10^8 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_squared_l1669_166988


namespace NUMINAMATH_CALUDE_cosine_function_properties_l1669_166978

/-- 
Given a cosine function y = a * cos(b * x + c) where:
1. The minimum occurs at x = 0
2. The peak-to-peak amplitude is 6
Prove that c = π
-/
theorem cosine_function_properties (a b c : ℝ) : 
  (∀ x, a * Real.cos (b * x + c) ≥ a * Real.cos c) →  -- minimum at x = 0
  (2 * |a| = 6) →                                     -- peak-to-peak amplitude is 6
  c = π :=
by sorry

end NUMINAMATH_CALUDE_cosine_function_properties_l1669_166978


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1669_166948

theorem inequality_system_solution :
  ∃ (x : ℤ),
    (3 * (2 * x - 1) < 2 * x + 8) ∧
    (2 + (3 * (x + 1)) / 8 > 3 - (x - 1) / 4) ∧
    (x = 2) ∧
    (∀ a : ℝ, (a * x + 6 ≤ x - 2 * a) → (|a + 1| - |a - 1| = -2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1669_166948


namespace NUMINAMATH_CALUDE_wipes_count_l1669_166964

/-- The number of wipes initially in the container -/
def initial_wipes : ℕ := 70

/-- The number of wipes used during the day -/
def wipes_used : ℕ := 20

/-- The number of wipes added after using some -/
def wipes_added : ℕ := 10

/-- The number of wipes left at night -/
def wipes_at_night : ℕ := 60

theorem wipes_count : initial_wipes - wipes_used + wipes_added = wipes_at_night := by
  sorry

end NUMINAMATH_CALUDE_wipes_count_l1669_166964


namespace NUMINAMATH_CALUDE_village_population_l1669_166990

/-- The number of residents who speak Bashkir -/
def bashkir_speakers : ℕ := 912

/-- The number of residents who speak Russian -/
def russian_speakers : ℕ := 653

/-- The number of residents who speak both Bashkir and Russian -/
def bilingual_speakers : ℕ := 435

/-- The total number of residents in the village -/
def total_residents : ℕ := bashkir_speakers + russian_speakers - bilingual_speakers

theorem village_population :
  total_residents = 1130 :=
by sorry

end NUMINAMATH_CALUDE_village_population_l1669_166990


namespace NUMINAMATH_CALUDE_hillary_activities_lcm_l1669_166966

theorem hillary_activities_lcm : Nat.lcm (Nat.lcm 6 4) 16 = 48 := by
  sorry

end NUMINAMATH_CALUDE_hillary_activities_lcm_l1669_166966


namespace NUMINAMATH_CALUDE_inequality_proof_l1669_166911

theorem inequality_proof (x : ℝ) : 1 + 2 * x^2 ≥ 2 * x + x^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1669_166911


namespace NUMINAMATH_CALUDE_class_artworks_l1669_166902

theorem class_artworks (total_students : ℕ) (total_kits : ℕ) 
  (students_one_kit : ℕ) (students_two_kits : ℕ)
  (students_five_works : ℕ) (students_six_works : ℕ) (students_seven_works : ℕ) :
  total_students = 24 →
  total_kits = 36 →
  students_one_kit = 12 →
  students_two_kits = 12 →
  students_five_works = 8 →
  students_six_works = 10 →
  students_seven_works = 6 →
  students_one_kit + students_two_kits = total_students →
  students_five_works + students_six_works + students_seven_works = total_students →
  students_five_works * 5 + students_six_works * 6 + students_seven_works * 7 = 142 :=
by sorry

end NUMINAMATH_CALUDE_class_artworks_l1669_166902


namespace NUMINAMATH_CALUDE_erikas_savings_l1669_166953

theorem erikas_savings (gift_cost cake_cost leftover : ℕ) 
  (h1 : gift_cost = 250)
  (h2 : cake_cost = 25)
  (h3 : leftover = 5)
  (ricks_savings : ℕ) (h4 : ricks_savings = gift_cost / 2)
  (total_savings : ℕ) (h5 : total_savings = gift_cost + cake_cost + leftover) :
  total_savings - ricks_savings = 155 := by
sorry

end NUMINAMATH_CALUDE_erikas_savings_l1669_166953


namespace NUMINAMATH_CALUDE_susan_ate_six_candies_l1669_166997

/-- The number of candies Susan ate during the week -/
def candies_eaten (bought_tuesday bought_thursday bought_friday remaining : ℕ) : ℕ :=
  bought_tuesday + bought_thursday + bought_friday - remaining

/-- Theorem stating that Susan ate 6 candies during the week -/
theorem susan_ate_six_candies :
  candies_eaten 3 5 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_ate_six_candies_l1669_166997


namespace NUMINAMATH_CALUDE_equation_solution_l1669_166980

theorem equation_solution :
  ∀ x y : ℝ, y = 3 * x →
  (5 * y^2 + 3 * y + 2 = 3 * (8 * x^2 + y + 1)) ↔ 
  (x = 1 / Real.sqrt 21 ∨ x = -(1 / Real.sqrt 21)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1669_166980


namespace NUMINAMATH_CALUDE_brick_length_calculation_l1669_166927

theorem brick_length_calculation (courtyard_length courtyard_width : ℝ)
  (brick_width : ℝ) (total_bricks : ℕ) (h1 : courtyard_length = 25)
  (h2 : courtyard_width = 16) (h3 : brick_width = 0.1) (h4 : total_bricks = 20000) :
  (courtyard_length * courtyard_width * 10000) / (brick_width * total_bricks) = 20 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l1669_166927


namespace NUMINAMATH_CALUDE_triangle_side_length_l1669_166932

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 9 → b = 2 * Real.sqrt 3 → C = 150 * π / 180 → c = 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1669_166932


namespace NUMINAMATH_CALUDE_quotient_sum_difference_forty_percent_less_than_36_l1669_166939

-- Problem 1
theorem quotient_sum_difference : (0.4 + 1/3) / (0.4 - 1/3) = 11 := by sorry

-- Problem 2
theorem forty_percent_less_than_36 : ∃ x : ℝ, x - 0.4 * x = 36 ∧ x = 60 := by sorry

end NUMINAMATH_CALUDE_quotient_sum_difference_forty_percent_less_than_36_l1669_166939


namespace NUMINAMATH_CALUDE_find_p_l1669_166941

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (p w : ℂ) : Prop := 10 * p - w = 50000

-- State the theorem
theorem find_p :
  ∀ (p w : ℂ),
  equation p w →
  (10 : ℂ) = 2 →
  w = 10 + 250 * i →
  p = 5001 + 25 * i :=
by sorry

end NUMINAMATH_CALUDE_find_p_l1669_166941


namespace NUMINAMATH_CALUDE_worker_earnings_worker_earnings_proof_l1669_166921

/-- Calculates the total earnings of a worker based on regular and cellphone survey rates -/
theorem worker_earnings (regular_rate : ℕ) (total_surveys : ℕ) (cellphone_rate_increase : ℚ) 
  (cellphone_surveys : ℕ) (h1 : regular_rate = 30) (h2 : total_surveys = 100) 
  (h3 : cellphone_rate_increase = 1/5) (h4 : cellphone_surveys = 50) : ℕ :=
  let regular_surveys := total_surveys - cellphone_surveys
  let cellphone_rate := regular_rate + (regular_rate * cellphone_rate_increase).floor
  let regular_pay := regular_surveys * regular_rate
  let cellphone_pay := cellphone_surveys * cellphone_rate
  let total_pay := regular_pay + cellphone_pay
  3300

/-- The worker's total earnings for the week are Rs. 3300 -/
theorem worker_earnings_proof : worker_earnings 30 100 (1/5) 50 rfl rfl rfl rfl = 3300 := by
  sorry

end NUMINAMATH_CALUDE_worker_earnings_worker_earnings_proof_l1669_166921


namespace NUMINAMATH_CALUDE_f_properties_l1669_166943

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 2 * Real.sin x

theorem f_properties :
  (∃ (max : ℝ), ∀ (x : ℝ), f x ≤ max ∧ max = Real.sqrt 3) ∧
  (∃ (θ : ℝ), ∀ (x : ℝ), f x ≤ f θ ∧ Real.cos (θ - π/6) = (3 + Real.sqrt 6) / 6) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1669_166943


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficients_l1669_166905

theorem cubic_expansion_coefficients (a b : ℤ) : 
  (3 * b + 3 * a^2 = 99) ∧ (3 * a * b^2 = 162) → (a = 6 ∧ b = -3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficients_l1669_166905


namespace NUMINAMATH_CALUDE_parabola_intersection_value_l1669_166903

theorem parabola_intersection_value (a : ℝ) : 
  a^2 - a - 1 = 0 → a^2 - a + 2014 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_value_l1669_166903


namespace NUMINAMATH_CALUDE_correct_assignment_count_l1669_166967

/-- Represents an assignment statement --/
inductive AssignmentStatement
  | Constant : ℕ → String → AssignmentStatement
  | Variable : String → String → AssignmentStatement
  | Expression : String → String → AssignmentStatement
  | SelfAssignment : String → AssignmentStatement

/-- Checks if an assignment statement is valid --/
def isValidAssignment (stmt : AssignmentStatement) : Bool :=
  match stmt with
  | AssignmentStatement.Constant _ _ => false
  | AssignmentStatement.Variable _ _ => true
  | AssignmentStatement.Expression _ _ => false
  | AssignmentStatement.SelfAssignment _ => true

/-- The list of given assignment statements --/
def givenStatements : List AssignmentStatement :=
  [AssignmentStatement.Constant 2 "A",
   AssignmentStatement.Expression "x_+_y" "2",
   AssignmentStatement.Expression "A_-_B" "-2",
   AssignmentStatement.SelfAssignment "A"]

/-- Counts the number of valid assignment statements in a list --/
def countValidAssignments (stmts : List AssignmentStatement) : ℕ :=
  (stmts.filter isValidAssignment).length

theorem correct_assignment_count :
  countValidAssignments givenStatements = 1 := by sorry

end NUMINAMATH_CALUDE_correct_assignment_count_l1669_166967


namespace NUMINAMATH_CALUDE_system_of_inequalities_l1669_166900

theorem system_of_inequalities (x : ℝ) :
  (2 * x + 1 < 3) ∧ (x / 2 + (1 - 3 * x) / 4 ≤ 1) → -3 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l1669_166900


namespace NUMINAMATH_CALUDE_track_length_is_360_l1669_166998

/-- Represents a circular running track -/
structure Track where
  length : ℝ
  start_points_opposite : Bool
  runners_opposite_directions : Bool

/-- Represents a runner on the track -/
structure Runner where
  speed : ℝ
  distance_to_first_meeting : ℝ
  distance_between_meetings : ℝ

/-- The main theorem statement -/
theorem track_length_is_360 (track : Track) (brenda sally : Runner) : 
  track.start_points_opposite ∧ 
  track.runners_opposite_directions ∧
  brenda.distance_to_first_meeting = 120 ∧
  sally.speed = 2 * brenda.speed ∧
  sally.distance_between_meetings = 180 →
  track.length = 360 := by sorry

end NUMINAMATH_CALUDE_track_length_is_360_l1669_166998


namespace NUMINAMATH_CALUDE_prime_divisor_of_3n_minus_1_and_n_minus_10_l1669_166922

theorem prime_divisor_of_3n_minus_1_and_n_minus_10 (n : ℕ) (p : ℕ) (h_prime : Prime p) 
  (h_div_3n_minus_1 : p ∣ (3 * n - 1)) (h_div_n_minus_10 : p ∣ (n - 10)) : p = 29 :=
sorry

end NUMINAMATH_CALUDE_prime_divisor_of_3n_minus_1_and_n_minus_10_l1669_166922


namespace NUMINAMATH_CALUDE_sum_of_angles_convex_polygon_l1669_166962

/-- The sum of interior angles of a convex polygon with n sides, where n ≥ 3 -/
def sumOfAngles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- Theorem: For any convex polygon with n sides, where n ≥ 3,
    the sum of its interior angles is equal to (n-2) * 180° -/
theorem sum_of_angles_convex_polygon (n : ℕ) (h : n ≥ 3) :
  sumOfAngles n = (n - 2) * 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_convex_polygon_l1669_166962


namespace NUMINAMATH_CALUDE_quadratic_expression_evaluation_l1669_166906

theorem quadratic_expression_evaluation :
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  2 * x^2 + 3 * y^2 - z^2 + 4 * x * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_evaluation_l1669_166906


namespace NUMINAMATH_CALUDE_next_two_pythagorean_triples_l1669_166919

/-- Given a sequence of Pythagorean triples, find the next two triples -/
theorem next_two_pythagorean_triples 
  (h1 : 3^2 + 4^2 = 5^2)
  (h2 : 5^2 + 12^2 = 13^2)
  (h3 : 7^2 + 24^2 = 25^2) :
  (9^2 + 40^2 = 41^2) ∧ (11^2 + 60^2 = 61^2) := by
  sorry

end NUMINAMATH_CALUDE_next_two_pythagorean_triples_l1669_166919


namespace NUMINAMATH_CALUDE_inequality_and_existence_l1669_166950

theorem inequality_and_existence : 
  (∀ x y z : ℝ, x^2 + 2*y^2 + 3*z^2 ≥ Real.sqrt 3 * (x*y + y*z + z*x)) ∧ 
  (∃ k : ℝ, k > Real.sqrt 3 ∧ (∀ x y z : ℝ, x^2 + 2*y^2 + 3*z^2 ≥ k * (x*y + y*z + z*x))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_existence_l1669_166950


namespace NUMINAMATH_CALUDE_derivative_x_ln_x_l1669_166931

open Real

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) :
  deriv (fun x => x * log x) x = log x + 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_ln_x_l1669_166931


namespace NUMINAMATH_CALUDE_gift_bags_total_l1669_166982

theorem gift_bags_total (daily_rate : ℕ) (days_needed : ℕ) (h1 : daily_rate = 42) (h2 : days_needed = 13) :
  daily_rate * days_needed = 546 := by
  sorry

end NUMINAMATH_CALUDE_gift_bags_total_l1669_166982


namespace NUMINAMATH_CALUDE_point_transformation_l1669_166947

/-- Given a point P(a,b) in the xy-plane, this theorem proves that if P is first rotated
    clockwise by 180° around the origin (0,0) and then reflected about the line y = -x,
    resulting in the point (9,-4), then b - a = -13. -/
theorem point_transformation (a b : ℝ) : 
  (∃ (x y : ℝ), ((-a) = x ∧ (-b) = y) ∧ (y = x ∧ -x = 9 ∧ -y = -4)) → b - a = -13 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1669_166947


namespace NUMINAMATH_CALUDE_min_moves_for_checkerboard_l1669_166908

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents a 6x6 grid -/
def Grid := Fin 6 → Fin 6 → Cell

/-- Represents a move (changing color of two adjacent cells) -/
structure Move where
  row : Fin 6
  col : Fin 6
  horizontal : Bool

/-- Defines a checkerboard pattern -/
def isCheckerboard (g : Grid) : Prop :=
  ∀ i j, g i j = if (i.val + j.val) % 2 = 0 then Cell.White else Cell.Black

/-- Applies a move to a grid -/
def applyMove (g : Grid) (m : Move) : Grid :=
  sorry

/-- Counts the number of black cells in a grid -/
def blackCellCount (g : Grid) : Nat :=
  sorry

theorem min_moves_for_checkerboard :
  ∀ (initial : Grid) (moves : List Move),
    (∀ i j, initial i j = Cell.White) →
    isCheckerboard (moves.foldl applyMove initial) →
    moves.length ≥ 18 :=
  sorry

end NUMINAMATH_CALUDE_min_moves_for_checkerboard_l1669_166908


namespace NUMINAMATH_CALUDE_matrix_power_4_l1669_166992

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_power_4 : A^4 = !![(-8), 8; 0, 3] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l1669_166992


namespace NUMINAMATH_CALUDE_chessboard_polygon_theorem_l1669_166924

/-- A polygon cut out from an infinite chessboard -/
structure ChessboardPolygon where
  black_cells : ℕ              -- number of black cells
  white_cells : ℕ              -- number of white cells
  black_perimeter : ℕ          -- number of black perimeter segments
  white_perimeter : ℕ          -- number of white perimeter segments

/-- Theorem stating the relationship between perimeter segments and cells -/
theorem chessboard_polygon_theorem (p : ChessboardPolygon) :
  p.black_perimeter - p.white_perimeter = 4 * (p.black_cells - p.white_cells) := by
  sorry

end NUMINAMATH_CALUDE_chessboard_polygon_theorem_l1669_166924


namespace NUMINAMATH_CALUDE_remainder_problem_l1669_166912

theorem remainder_problem (n : ℤ) (h : n % 5 = 3) : (n + 1) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1669_166912


namespace NUMINAMATH_CALUDE_equal_share_of_sweets_l1669_166951

/-- Represents the number of sweets Jennifer has of each color -/
structure Sweets where
  green : Nat
  blue : Nat
  yellow : Nat

/-- The total number of people sharing the sweets -/
def totalPeople : Nat := 4

/-- Jennifer's sweets -/
def jenniferSweets : Sweets := { green := 212, blue := 310, yellow := 502 }

/-- Theorem stating that each person gets 256 sweets when Jennifer shares equally -/
theorem equal_share_of_sweets (s : Sweets) (h : s = jenniferSweets) :
  (s.green + s.blue + s.yellow) / totalPeople = 256 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_of_sweets_l1669_166951


namespace NUMINAMATH_CALUDE_volume_ratio_cylinders_capacity_ratio_64_percent_l1669_166913

/-- The volume ratio of two right circular cylinders with the same height
    is equal to the square of the ratio of their circumferences. -/
theorem volume_ratio_cylinders (h C_A C_B : ℝ) (h_pos : h > 0) (C_A_pos : C_A > 0) (C_B_pos : C_B > 0) :
  (h * (C_A / (2 * Real.pi))^2) / (h * (C_B / (2 * Real.pi))^2) = (C_A / C_B)^2 := by
  sorry

/-- The capacity of a cylinder with circumference 8 is 64% of the capacity
    of a cylinder with circumference 10, given the same height. -/
theorem capacity_ratio_64_percent (h : ℝ) (h_pos : h > 0) :
  (h * (8 / (2 * Real.pi))^2) / (h * (10 / (2 * Real.pi))^2) = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_cylinders_capacity_ratio_64_percent_l1669_166913


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1669_166945

theorem quadratic_root_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 3) ↔ x ≥ -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1669_166945


namespace NUMINAMATH_CALUDE_percentage_calculation_l1669_166934

theorem percentage_calculation (P : ℝ) : 
  (P / 100) * 1265 / 5.96 = 377.8020134228188 → P = 178 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1669_166934


namespace NUMINAMATH_CALUDE_cupcake_distribution_exists_l1669_166956

theorem cupcake_distribution_exists (total_cupcakes : ℕ) 
  (cupcakes_per_cousin : ℕ) (cupcakes_per_friend : ℕ) : 
  total_cupcakes = 42 → cupcakes_per_cousin = 3 → cupcakes_per_friend = 2 →
  ∃ (n : ℕ), ∃ (cousins : ℕ), ∃ (friends : ℕ),
    n = cousins + friends ∧ 
    cousins * cupcakes_per_cousin + friends * cupcakes_per_friend = total_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_cupcake_distribution_exists_l1669_166956


namespace NUMINAMATH_CALUDE_star_3_5_l1669_166973

/-- The star operation defined for real numbers -/
def star (x y : ℝ) : ℝ := x^2 + x*y + y^2

/-- Theorem stating that 3 ⋆ 5 = 49 -/
theorem star_3_5 : star 3 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_star_3_5_l1669_166973


namespace NUMINAMATH_CALUDE_sum_between_13_and_14_l1669_166989

theorem sum_between_13_and_14 : ∃ x : ℚ, 
  13 < x ∧ x < 14 ∧ 
  x = (3 + 3/8) + (4 + 2/5) + (6 + 1/11) := by
  sorry

end NUMINAMATH_CALUDE_sum_between_13_and_14_l1669_166989


namespace NUMINAMATH_CALUDE_jason_total_cost_l1669_166920

def stove_cost : ℚ := 1200
def wall_cost : ℚ := stove_cost / 6
def repair_cost : ℚ := stove_cost + wall_cost
def labor_fee_rate : ℚ := 1/5  -- 20% as a fraction

def total_cost : ℚ := repair_cost + (labor_fee_rate * repair_cost)

theorem jason_total_cost : total_cost = 1680 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_cost_l1669_166920


namespace NUMINAMATH_CALUDE_total_choices_is_64_l1669_166942

/-- The number of tour routes available -/
def num_routes : ℕ := 4

/-- The number of tour groups -/
def num_groups : ℕ := 3

/-- The total number of different possible choices -/
def total_choices : ℕ := num_routes ^ num_groups

/-- Theorem stating that the total number of different choices is 64 -/
theorem total_choices_is_64 : total_choices = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_choices_is_64_l1669_166942


namespace NUMINAMATH_CALUDE_exists_team_rating_l1669_166995

variable {Team : Type}
variable (d : Team → Team → ℝ)

axiom goal_difference_symmetry :
  ∀ (A B : Team), d A B + d B A = 0

axiom goal_difference_transitivity :
  ∀ (A B C : Team), d A B + d B C + d C A = 0

theorem exists_team_rating :
  ∃ (f : Team → ℝ), ∀ (A B : Team), d A B = f A - f B :=
sorry

end NUMINAMATH_CALUDE_exists_team_rating_l1669_166995


namespace NUMINAMATH_CALUDE_investment_growth_l1669_166991

/-- Represents the investment growth over a two-year period -/
theorem investment_growth 
  (initial_investment : ℝ) 
  (final_investment : ℝ) 
  (growth_rate : ℝ) 
  (h1 : initial_investment = 800) 
  (h2 : final_investment = 960) 
  (h3 : initial_investment * (1 + growth_rate)^2 = final_investment) : 
  800 * (1 + growth_rate)^2 = 960 :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l1669_166991


namespace NUMINAMATH_CALUDE_tv_watching_time_l1669_166975

/-- The number of episodes of Jeopardy watched -/
def jeopardy_episodes : ℕ := 2

/-- The number of episodes of Wheel of Fortune watched -/
def wheel_episodes : ℕ := 2

/-- The duration of one episode of Jeopardy in minutes -/
def jeopardy_duration : ℕ := 20

/-- The duration of one episode of Wheel of Fortune in minutes -/
def wheel_duration : ℕ := 2 * jeopardy_duration

/-- The total time spent watching TV in minutes -/
def total_time : ℕ := jeopardy_episodes * jeopardy_duration + wheel_episodes * wheel_duration

/-- Conversion factor from minutes to hours -/
def minutes_per_hour : ℕ := 60

/-- Theorem: James watched TV for 2 hours -/
theorem tv_watching_time : total_time / minutes_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_tv_watching_time_l1669_166975


namespace NUMINAMATH_CALUDE_wages_payment_duration_l1669_166987

/-- Given a sum of money that can pay two workers' wages separately for different periods,
    this theorem proves how long it can pay both workers together. -/
theorem wages_payment_duration (S : ℝ) (p q : ℝ) (hp : S = 24 * p) (hq : S = 40 * q) :
  ∃ D : ℝ, D = 15 ∧ S = D * (p + q) := by
  sorry

end NUMINAMATH_CALUDE_wages_payment_duration_l1669_166987


namespace NUMINAMATH_CALUDE_sum_product_bounds_l1669_166949

theorem sum_product_bounds (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (min max : ℝ), min = -1/4 ∧ max = 1/2 ∧
  (xy + xz + yz ≥ min ∧ xy + xz + yz ≤ max) ∧
  ∀ t, min ≤ t ∧ t ≤ max → ∃ (a b c : ℝ), a + b + c = 1 ∧ ab + ac + bc = t :=
sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l1669_166949


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l1669_166946

theorem system_of_inequalities_solution (x : ℝ) : 
  ((x - 1) / 2 < 2 * x + 1 ∧ -3 * (1 - x) ≥ -4) ↔ x ≥ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l1669_166946


namespace NUMINAMATH_CALUDE_triangle_area_l1669_166986

def a : Fin 2 → ℝ := ![4, -1]
def b : Fin 2 → ℝ := ![3, 5]

theorem triangle_area : 
  (1/2 : ℝ) * |a 0 * b 1 - a 1 * b 0| = 23/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1669_166986


namespace NUMINAMATH_CALUDE_annulus_area_l1669_166954

/-- The area of an annulus with specific properties -/
theorem annulus_area (r s t : ℝ) (h1 : r > s) (h2 : t = 2 * s) (h3 : r^2 = s^2 + (t/2)^2) :
  π * (r^2 - s^2) = π * s^2 := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l1669_166954


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_three_l1669_166940

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_three :
  log10 5^2 + 2/3 * log10 8 + log10 5 * log10 20 + (log10 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_three_l1669_166940


namespace NUMINAMATH_CALUDE_henrikh_walk_time_per_block_l1669_166976

/-- The time it takes Henrikh to walk one block to work -/
def walkTimePerBlock : ℝ := 60

/-- The number of blocks from Henrikh's home to his office -/
def distanceInBlocks : ℕ := 12

/-- The time it takes Henrikh to ride his bicycle for one block -/
def bikeTimePerBlock : ℝ := 20

/-- The additional time it takes to walk compared to riding a bicycle for the entire distance -/
def additionalWalkTime : ℝ := 8 * 60  -- 8 minutes in seconds

theorem henrikh_walk_time_per_block :
  walkTimePerBlock * distanceInBlocks = 
  bikeTimePerBlock * distanceInBlocks + additionalWalkTime :=
by sorry

end NUMINAMATH_CALUDE_henrikh_walk_time_per_block_l1669_166976


namespace NUMINAMATH_CALUDE_problem_solution_l1669_166959

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - 4*x^3 + a*x^2 - 1

-- Define the function g
def g (b : ℝ) (x : ℝ) : ℝ := b*x^2 - 1

theorem problem_solution :
  -- Part 1
  (∀ x y, (0 ≤ x ∧ x < y ∧ y ≤ 1) → f 4 x < f 4 y) ∧
  (∀ x y, (1 ≤ x ∧ x < y ∧ y ≤ 2) → f 4 x > f 4 y) →
  -- Part 2
  (∃ b₁ b₂, b₁ ≠ b₂ ∧
    (∀ b, (∃! x₁ x₂, x₁ ≠ x₂ ∧ f 4 x₁ = g b x₁ ∧ f 4 x₂ = g b x₂) ↔ (b = b₁ ∨ b = b₂))) ∧
  -- Part 3
  (∀ m n, m ∈ Set.Icc (-6 : ℝ) (-2) →
    ((∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f 4 x ≤ m*x^3 + 2*x^2 - n) →
      n ∈ Set.Iic (-4 : ℝ))) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1669_166959


namespace NUMINAMATH_CALUDE_tomato_ratio_l1669_166914

def total_tomatoes : ℕ := 127
def eaten_by_birds : ℕ := 19
def tomatoes_left : ℕ := 54

theorem tomato_ratio :
  let picked := total_tomatoes - eaten_by_birds
  let given_to_friend := picked - tomatoes_left
  (given_to_friend : ℚ) / picked = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tomato_ratio_l1669_166914


namespace NUMINAMATH_CALUDE_solve_for_b_l1669_166944

theorem solve_for_b (a b : ℝ) (h1 : 3 * a + 2 = 5) (h2 : b - 4 * a = 2) : b = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l1669_166944


namespace NUMINAMATH_CALUDE_cyclic_sum_root_l1669_166971

theorem cyclic_sum_root {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := (a * b * c) / (a * b + b * c + c * a + 2 * Real.sqrt (a * b * c * (a + b + c)))
  (Real.sqrt (a * b * x * (a + b + x)) + 
   Real.sqrt (b * c * x * (b + c + x)) + 
   Real.sqrt (c * a * x * (c + a + x))) = 
  Real.sqrt (a * b * c * (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_root_l1669_166971


namespace NUMINAMATH_CALUDE_a_minus_c_equals_three_l1669_166952

theorem a_minus_c_equals_three (a b c d : ℤ) 
  (h1 : a - b = c + d + 9) 
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_equals_three_l1669_166952


namespace NUMINAMATH_CALUDE_circle_symmetry_axis_l1669_166961

/-- Given a circle and a line that is its axis of symmetry, prove that the parameter a in the line equation equals 1 -/
theorem circle_symmetry_axis (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y - 3 = 0 → 
    (∃ c : ℝ, ∀ x' y' : ℝ, (x' - 2*a*y' - 3 = 0 ∧ 
      x'^2 + y'^2 - 2*x' + 2*y' - 3 = 0) ↔ 
      (2*c - x' - 2*a*y' - 3 = 0 ∧ 
       (2*c - x')^2 + y'^2 - 2*(2*c - x') + 2*y' - 3 = 0))) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_axis_l1669_166961


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1669_166963

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1669_166963


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l1669_166926

/-- For the equation (a-2)x^2 + (a+2)x + 3 = 0 to be a quadratic equation in one variable, a ≠ 2 -/
theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, ∃ y, y = (a - 2) * x^2 + (a + 2) * x + 3) → a ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l1669_166926


namespace NUMINAMATH_CALUDE_cuts_through_examples_l1669_166936

-- Define what it means for a line to cut through a curve at a point
def cuts_through (l : ℝ → ℝ) (c : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  -- The line is tangent to the curve at the point
  (∀ x, l x = c x + (l p.1 - c p.1)) ∧
  -- The curve is on both sides of the line near the point
  ∃ δ > 0, ∀ x ∈ Set.Ioo (p.1 - δ) (p.1 + δ), 
    (x < p.1 → c x < l x) ∧ (x > p.1 → c x > l x)

-- Theorem statement
theorem cuts_through_examples :
  cuts_through (λ _ => 0) (λ x => x^3) (0, 0) ∧
  cuts_through (λ x => x) Real.sin (0, 0) ∧
  cuts_through (λ x => x) Real.tan (0, 0) := by
  sorry

end NUMINAMATH_CALUDE_cuts_through_examples_l1669_166936


namespace NUMINAMATH_CALUDE_opposite_sqrt_nine_is_negative_three_l1669_166930

theorem opposite_sqrt_nine_is_negative_three :
  -(Real.sqrt 9) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sqrt_nine_is_negative_three_l1669_166930


namespace NUMINAMATH_CALUDE_ln_ln_pi_lt_ln_pi_lt_exp_ln_pi_l1669_166970

theorem ln_ln_pi_lt_ln_pi_lt_exp_ln_pi : 
  Real.log (Real.log Real.pi) < Real.log Real.pi ∧ Real.log Real.pi < 2 ^ Real.log Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ln_ln_pi_lt_ln_pi_lt_exp_ln_pi_l1669_166970


namespace NUMINAMATH_CALUDE_courtyard_paving_l1669_166916

/-- Represents the dimensions of a rectangular area in centimeters -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular shape given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts meters to centimeters -/
def meters_to_cm (m : ℕ) : ℕ := m * 100

/-- The dimensions of the courtyard in meters -/
def courtyard_m : Dimensions := ⟨30, 16⟩

/-- The dimensions of the courtyard in centimeters -/
def courtyard_cm : Dimensions := ⟨meters_to_cm courtyard_m.length, meters_to_cm courtyard_m.width⟩

/-- The dimensions of a single brick in centimeters -/
def brick : Dimensions := ⟨20, 10⟩

/-- Calculates the number of bricks needed to cover an area -/
def bricks_needed (area_to_cover : ℕ) (brick_size : ℕ) : ℕ := area_to_cover / brick_size

theorem courtyard_paving :
  bricks_needed (area courtyard_cm) (area brick) = 24000 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l1669_166916


namespace NUMINAMATH_CALUDE_vector_magnitude_l1669_166917

theorem vector_magnitude (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (2, -1)
  (a.1 * b.1 + a.2 * b.2 = 5) →
  ((a.1 + b.1)^2 + (a.2 + b.2)^2 = 8^2) →
  (b.1^2 + b.2^2 = 7^2) :=
by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1669_166917


namespace NUMINAMATH_CALUDE_P_and_S_not_third_l1669_166935

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T | U

-- Define the finish order relation
def finishes_before (a b : Runner) : Prop := sorry

-- Define the race conditions
axiom P_beats_Q : finishes_before Runner.P Runner.Q
axiom P_beats_R : finishes_before Runner.P Runner.R
axiom Q_beats_S : finishes_before Runner.Q Runner.S
axiom U_after_P_before_T : finishes_before Runner.P Runner.U ∧ finishes_before Runner.U Runner.T
axiom T_after_P_before_Q : finishes_before Runner.P Runner.T ∧ finishes_before Runner.T Runner.Q

-- Define a function to represent the finishing position of a runner
def finish_position (r : Runner) : ℕ := sorry

-- State the theorem
theorem P_and_S_not_third :
  ¬(finish_position Runner.P = 3 ∨ finish_position Runner.S = 3) :=
sorry

end NUMINAMATH_CALUDE_P_and_S_not_third_l1669_166935


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1669_166901

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z + w) = 6) :
  Complex.abs z = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1669_166901


namespace NUMINAMATH_CALUDE_park_playgroups_l1669_166925

theorem park_playgroups (girls boys parents playgroups : ℕ) 
  (h1 : girls = 14)
  (h2 : boys = 11)
  (h3 : parents = 50)
  (h4 : playgroups = 3)
  (h5 : (girls + boys + parents) % playgroups = 0) :
  (girls + boys + parents) / playgroups = 25 := by
  sorry

end NUMINAMATH_CALUDE_park_playgroups_l1669_166925


namespace NUMINAMATH_CALUDE_unique_solution_l1669_166909

/-- The function F as defined in the problem -/
def F (t : ℝ) : ℝ := 32 * t^5 + 48 * t^3 + 17 * t - 15

/-- The system of equations -/
def system_equations (x y z : ℝ) : Prop :=
  1/x = 32/y^5 + 48/y^3 + 17/y - 15 ∧
  1/y = 32/z^5 + 48/z^3 + 17/z - 15 ∧
  1/z = 32/x^5 + 48/x^3 + 17/x - 15

/-- The theorem stating the unique solution -/
theorem unique_solution :
  ∃! (x y z : ℝ), system_equations x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1669_166909


namespace NUMINAMATH_CALUDE_average_age_problem_l1669_166904

theorem average_age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 27 →
  b = 23 →
  (a + c) / 2 = 29 := by
sorry

end NUMINAMATH_CALUDE_average_age_problem_l1669_166904


namespace NUMINAMATH_CALUDE_certain_number_proof_l1669_166981

theorem certain_number_proof : ∃ x : ℤ, (9823 + x = 13200) ∧ (x = 3377) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1669_166981


namespace NUMINAMATH_CALUDE_box_properties_l1669_166933

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  white : ℕ
  red : ℕ
  yellow : ℕ

/-- The given ball counts in the box -/
def box : BallCounts := { white := 1, red := 2, yellow := 3 }

/-- The total number of balls in the box -/
def totalBalls (b : BallCounts) : ℕ := b.white + b.red + b.yellow

/-- The number of possible outcomes when drawing 1 ball -/
def possibleOutcomes (b : BallCounts) : ℕ := 
  (if b.white > 0 then 1 else 0) + 
  (if b.red > 0 then 1 else 0) + 
  (if b.yellow > 0 then 1 else 0)

/-- The probability of drawing a ball of a specific color -/
def probability (b : BallCounts) (color : ℕ) : ℚ :=
  color / (totalBalls b : ℚ)

theorem box_properties : 
  (possibleOutcomes box = 3) ∧ 
  (probability box box.yellow > probability box box.red ∧ 
   probability box box.yellow > probability box box.white) ∧
  (probability box box.white + probability box box.yellow = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_box_properties_l1669_166933


namespace NUMINAMATH_CALUDE_towel_area_decrease_l1669_166974

theorem towel_area_decrease :
  ∀ (L B : ℝ), L > 0 → B > 0 →
  let new_length := 0.7 * L
  let new_breadth := 0.75 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.475 :=
by sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l1669_166974


namespace NUMINAMATH_CALUDE_complement_of_union_equals_open_interval_l1669_166937

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_of_union_equals_open_interval :
  (A ∪ B)ᶜ = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_open_interval_l1669_166937


namespace NUMINAMATH_CALUDE_average_and_difference_l1669_166984

theorem average_and_difference (y : ℝ) : 
  (45 + y) / 2 = 37 → |45 - y| = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l1669_166984


namespace NUMINAMATH_CALUDE_first_repeat_l1669_166915

/-- The number of points on the circle -/
def n : ℕ := 2021

/-- The function that calculates the position of the nth marked point -/
def f (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The theorem stating that 66 is the smallest positive integer b such that
    there exists an a < b where f(a) ≡ f(b) (mod n) -/
theorem first_repeat : 
  ∀ b < 66, ¬∃ a < b, f a % n = f b % n ∧ 
  ∃ a < 66, f a % n = f 66 % n :=
sorry

end NUMINAMATH_CALUDE_first_repeat_l1669_166915


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1669_166985

theorem basketball_lineup_combinations (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 6) :
  (n.factorial / (n - k).factorial) = 360360 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1669_166985


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1669_166999

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem arithmetic_sequence_difference :
  let C := arithmetic_sequence 20 15
  let D := arithmetic_sequence 20 (-15)
  |C 31 - D 31| = 900 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1669_166999


namespace NUMINAMATH_CALUDE_base3_10212_equals_104_l1669_166969

/-- Converts a base 3 number to base 10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- Theorem: The base 10 representation of 10212 in base 3 is 104 --/
theorem base3_10212_equals_104 : base3ToBase10 [1, 0, 2, 1, 2] = 104 := by
  sorry

end NUMINAMATH_CALUDE_base3_10212_equals_104_l1669_166969


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l1669_166929

theorem smaller_root_of_equation (x : ℚ) :
  (x - 4/5)^2 + (x - 4/5) * (x - 2/5) + (x - 1/2)^2 = 0 →
  x = 14/15 ∨ x = 4/5 ∧ 14/15 < 4/5 := by
sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l1669_166929


namespace NUMINAMATH_CALUDE_total_pitchers_is_one_and_half_l1669_166923

/-- The total number of pitchers of lemonade served during a school play -/
def total_pitchers (first second third fourth : ℚ) : ℚ :=
  first + second + third + fourth

/-- Theorem stating that the total number of pitchers served is 1.5 -/
theorem total_pitchers_is_one_and_half :
  total_pitchers 0.25 0.4166666666666667 0.25 0.5833333333333334 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_total_pitchers_is_one_and_half_l1669_166923


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1669_166955

theorem sufficient_condition_for_inequality (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a < b) (h3 : b < 0) :
  1 / a^2 > 1 / b^2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1669_166955


namespace NUMINAMATH_CALUDE_algae_free_day_l1669_166968

/-- The number of days it takes for the pond to be completely covered in algae -/
def total_days : ℕ := 20

/-- The fraction of the pond covered by algae on a given day -/
def algae_coverage (day : ℕ) : ℚ :=
  if day ≥ total_days then 1
  else (1 / 2) ^ (total_days - day)

/-- The day on which the pond is 87.5% algae-free -/
def target_day : ℕ :=
  total_days - 3

theorem algae_free_day :
  algae_coverage target_day = 1 - (7 / 8) :=
sorry

end NUMINAMATH_CALUDE_algae_free_day_l1669_166968


namespace NUMINAMATH_CALUDE_emissions_2019_safe_m_range_l1669_166996

/-- Represents the carbon emissions of City A over years -/
def CarbonEmissions (m : ℝ) : ℕ → ℝ
  | 0 => 400  -- 2017 emissions
  | n + 1 => 0.9 * CarbonEmissions m n + m

/-- The maximum allowed annual carbon emissions -/
def MaxEmissions : ℝ := 550

/-- Theorem stating the carbon emissions of City A in 2019 -/
theorem emissions_2019 (m : ℝ) (h : m > 0) : 
  CarbonEmissions m 2 = 324 + 1.9 * m := by sorry

/-- Theorem stating the range of m for which emergency measures are never needed -/
theorem safe_m_range : 
  ∀ m : ℝ, (m > 0 ∧ m ≤ 55) ↔ 
    (∀ n : ℕ, CarbonEmissions m n ≤ MaxEmissions) := by sorry

end NUMINAMATH_CALUDE_emissions_2019_safe_m_range_l1669_166996


namespace NUMINAMATH_CALUDE_extremum_at_one_l1669_166958

def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

theorem extremum_at_one (a b : ℝ) :
  f a b 1 = 10 ∧ (deriv (f a b)) 1 = 0 → a = -4 ∧ b = 11 :=
by sorry

end NUMINAMATH_CALUDE_extremum_at_one_l1669_166958


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1669_166960

theorem solve_exponential_equation : ∃ x : ℝ, (100 : ℝ) ^ 4 = 5 ^ x ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1669_166960


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_4x4x4_cube_l1669_166993

/-- Represents a 4x4x4 cube composed of unit cubes -/
structure Cube :=
  (size : Nat)
  (total_units : Nat)
  (painted_corners : Nat)

/-- The number of unpainted unit cubes in a cube with painted corners -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - c.painted_corners

/-- Theorem stating the number of unpainted cubes in the specific 4x4x4 cube -/
theorem unpainted_cubes_in_4x4x4_cube :
  ∃ (c : Cube), c.size = 4 ∧ c.total_units = 64 ∧ c.painted_corners = 8 ∧ unpainted_cubes c = 56 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_4x4x4_cube_l1669_166993


namespace NUMINAMATH_CALUDE_olivia_friday_hours_l1669_166957

/-- Calculates the number of hours Olivia worked on Friday given her hourly rate, work hours on Monday and Wednesday, and total earnings for the week. -/
def fridayHours (hourlyRate : ℚ) (mondayHours wednesdayHours : ℚ) (totalEarnings : ℚ) : ℚ :=
  (totalEarnings - hourlyRate * (mondayHours + wednesdayHours)) / hourlyRate

/-- Proves that Olivia worked 6 hours on Friday given the specified conditions. -/
theorem olivia_friday_hours :
  fridayHours 9 4 3 117 = 6 := by
  sorry

end NUMINAMATH_CALUDE_olivia_friday_hours_l1669_166957
