import Mathlib

namespace NUMINAMATH_CALUDE_parabola_intersection_through_focus_l1152_115291

/-- The parabola type -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  m : ℝ
  b : ℝ

/-- Theorem statement -/
theorem parabola_intersection_through_focus 
  (para : Parabola) 
  (l : Line)
  (A B : Point)
  (N : ℝ) -- x-coordinate of N
  (h_not_perpendicular : l.m ≠ 0)
  (h_intersect : A.y^2 = 2*para.p*A.x ∧ B.y^2 = 2*para.p*B.x)
  (h_on_line : A.y = l.m * A.x + l.b ∧ B.y = l.m * B.x + l.b)
  (h_different_quadrants : A.y * B.y < 0)
  (h_bisect : abs ((A.y / (A.x - N)) + (B.y / (B.x - N))) = abs (A.y / (A.x - N) - B.y / (B.x - N))) :
  ∃ (t : ℝ), l.m * (para.p / 2) + l.b = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_through_focus_l1152_115291


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l1152_115254

/-- The number of ways to divide 12 dogs into groups of 4, 6, and 2,
    with Fluffy in the 4-dog group and Nipper in the 6-dog group -/
def dog_grouping_ways : ℕ := 2520

/-- The total number of dogs -/
def total_dogs : ℕ := 12

/-- The size of the first group (including Fluffy) -/
def group1_size : ℕ := 4

/-- The size of the second group (including Nipper) -/
def group2_size : ℕ := 6

/-- The size of the third group -/
def group3_size : ℕ := 2

theorem dog_grouping_theorem :
  dog_grouping_ways =
    Nat.choose (total_dogs - 2) (group1_size - 1) *
    Nat.choose (total_dogs - group1_size - 1) (group2_size - 1) :=
by sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l1152_115254


namespace NUMINAMATH_CALUDE_charlottes_schedule_is_correct_l1152_115265

/-- Represents the number of hours it takes to walk each type of dog -/
structure WalkingTime where
  poodle : ℕ
  chihuahua : ℕ
  labrador : ℕ

/-- Represents the schedule for the week -/
structure Schedule where
  monday_poodles : ℕ
  monday_chihuahuas : ℕ
  tuesday_chihuahuas : ℕ
  wednesday_labradors : ℕ

/-- The total available hours for dog-walking in the week -/
def total_hours : ℕ := 32

/-- The walking times for each type of dog -/
def walking_times : WalkingTime := {
  poodle := 2,
  chihuahua := 1,
  labrador := 3
}

/-- Charlotte's schedule for the week -/
def charlottes_schedule : Schedule := {
  monday_poodles := 8,  -- This is what we want to prove
  monday_chihuahuas := 2,
  tuesday_chihuahuas := 2,
  wednesday_labradors := 4
}

/-- Calculate the total hours spent walking dogs based on the schedule and walking times -/
def calculate_total_hours (s : Schedule) (w : WalkingTime) : ℕ :=
  s.monday_poodles * w.poodle +
  s.monday_chihuahuas * w.chihuahua +
  s.tuesday_chihuahuas * w.chihuahua +
  s.wednesday_labradors * w.labrador

/-- Theorem stating that Charlotte's schedule is correct -/
theorem charlottes_schedule_is_correct :
  calculate_total_hours charlottes_schedule walking_times = total_hours :=
by sorry

end NUMINAMATH_CALUDE_charlottes_schedule_is_correct_l1152_115265


namespace NUMINAMATH_CALUDE_smallest_number_l1152_115239

-- Define the numbers
def A : ℝ := 5.67823
def B : ℝ := 5.678333333 -- Approximation of 5.678̅3
def C : ℝ := 5.678383838 -- Approximation of 5.67̅83
def D : ℝ := 5.678378378 -- Approximation of 5.6̅783
def E : ℝ := 5.678367836 -- Approximation of 5.̅6783

-- Theorem statement
theorem smallest_number : E < A ∧ E < B ∧ E < C ∧ E < D :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1152_115239


namespace NUMINAMATH_CALUDE_triangle_ratio_l1152_115286

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  a / (Real.sin A) = c / (Real.sin C) ∧
  A = π / 3 ∧
  a = Real.sqrt 3 →
  (a + b) / (Real.sin A + Real.sin B) = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1152_115286


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1152_115229

theorem complex_magnitude_problem (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = (1 - i) / (2 + i) →
  Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1152_115229


namespace NUMINAMATH_CALUDE_smallest_a_l1152_115234

/-- A parabola with vertex at (1/3, -25/27) described by y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a > 0 → b = -2*a/3
  vertex_y : a > 0 → c = a/9 - 25/27
  integer_sum : ∃ k : ℤ, 3*a + 2*b + 4*c = k

/-- The smallest possible value of a for the given parabola conditions -/
theorem smallest_a (p : Parabola) : 
  (∀ q : Parabola, q.a > 0 → p.a ≤ q.a) → p.a = 300/19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_l1152_115234


namespace NUMINAMATH_CALUDE_circle_tangent_problem_l1152_115226

-- Define the circles
def Circle := ℝ × ℝ → Prop

-- Define the tangent line
def TangentLine (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Define the property of being square-free
def SquareFree (n : ℕ) : Prop := ∀ p : ℕ, Prime p → (p^2 ∣ n) → False

-- Define the property of being relatively prime
def RelativelyPrime (a c : ℕ) : Prop := Nat.gcd a c = 1

theorem circle_tangent_problem (C₁ C₂ : Circle) (m : ℝ) (a b c : ℕ) :
  (∃ x y : ℝ, C₁ (x, y) ∧ C₂ (x, y)) →  -- Circles intersect
  C₁ (8, 6) ∧ C₂ (8, 6) →  -- Intersection point at (8,6)
  (∃ r₁ r₂ : ℝ, r₁ * r₂ = 75) →  -- Product of radii is 75
  (∀ x : ℝ, C₁ (x, 0) → x = 0) ∧ (∀ x : ℝ, C₂ (x, 0) → x = 0) →  -- x-axis is tangent
  (∀ x y : ℝ, C₁ (x, y) ∧ TangentLine m x y → x = 0) ∧ 
  (∀ x y : ℝ, C₂ (x, y) ∧ TangentLine m x y → x = 0) →  -- y = mx is tangent
  m > 0 →  -- m is positive
  m = (a : ℝ) * Real.sqrt (b : ℝ) / (c : ℝ) →  -- m in the form a√b/c
  a > 0 ∧ b > 0 ∧ c > 0 →  -- a, b, c are positive
  SquareFree b →  -- b is square-free
  RelativelyPrime a c →  -- a and c are relatively prime
  a + b + c = 282 := by  -- Conclusion
sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_circle_tangent_problem_l1152_115226


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1152_115260

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, (3 * x₁^2 = 4 - 2 * x₁ ∧ 3 * x₂^2 = 4 - 2 * x₂) ∧ 
    x₁ = (-1 + Real.sqrt 13) / 3 ∧ x₂ = (-1 - Real.sqrt 13) / 3) ∧
  (∃ y₁ y₂ : ℝ, (y₁ * (y₁ - 7) = 8 * (7 - y₁) ∧ y₂ * (y₂ - 7) = 8 * (7 - y₂)) ∧
    y₁ = 7 ∧ y₂ = -8) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1152_115260


namespace NUMINAMATH_CALUDE_jerry_payment_l1152_115267

/-- Calculates the total payment for Jerry's work --/
theorem jerry_payment (painting_time counter_time_multiplier lawn_mowing_time hourly_rate : ℕ) 
  (h1 : counter_time_multiplier = 3)
  (h2 : painting_time = 8)
  (h3 : lawn_mowing_time = 6)
  (h4 : hourly_rate = 15) :
  (painting_time + counter_time_multiplier * painting_time + lawn_mowing_time) * hourly_rate = 570 :=
by sorry

end NUMINAMATH_CALUDE_jerry_payment_l1152_115267


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l1152_115247

-- Define the points
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (0, 6)
def C : ℝ × ℝ := (0, -2)
def D : ℝ × ℝ := (0, 2)

-- Define the moving point P
def P : ℝ × ℝ → Prop :=
  λ p => ‖p - A‖ / ‖p - B‖ = 1 / 2

-- Define the perpendicular bisector of PC
def perpBisector (p : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ q => ‖q - p‖ = ‖q - C‖

-- Define point Q
def Q (p : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ q => perpBisector p q ∧ ∃ t : ℝ, q = p + t • (D - p)

-- State the theorem
theorem trajectory_of_Q :
  ∀ p : ℝ × ℝ, P p →
    ∀ q : ℝ × ℝ, Q p q →
      q.2^2 - q.1^2 / 3 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l1152_115247


namespace NUMINAMATH_CALUDE_trapezium_shorter_side_length_l1152_115227

theorem trapezium_shorter_side_length 
  (longer_side : ℝ) 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : longer_side = 30) 
  (h2 : height = 16) 
  (h3 : area = 336) : 
  ∃ (shorter_side : ℝ), 
    area = (1 / 2) * (shorter_side + longer_side) * height ∧ 
    shorter_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_shorter_side_length_l1152_115227


namespace NUMINAMATH_CALUDE_quadratic_expression_equality_l1152_115289

theorem quadratic_expression_equality (a b : ℝ) : 
  ((-11 * -8)^(3/2) + 5 * Real.sqrt 16) * ((a - 2) + (b + 3)) = 
  ((176 * Real.sqrt 22) + 20) * (a + b + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equality_l1152_115289


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1152_115206

theorem complex_number_in_third_quadrant :
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  ∃ (a b : ℝ), z = Complex.mk a b ∧ a < 0 ∧ b < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1152_115206


namespace NUMINAMATH_CALUDE_daily_food_cost_l1152_115288

theorem daily_food_cost (purchase_price : ℕ) (vaccination_cost : ℕ) (selling_price : ℕ) (num_days : ℕ) (profit : ℕ) :
  purchase_price = 600 →
  vaccination_cost = 500 →
  selling_price = 2500 →
  num_days = 40 →
  profit = 600 →
  (selling_price - (purchase_price + vaccination_cost) - profit) / num_days = 20 := by
  sorry

end NUMINAMATH_CALUDE_daily_food_cost_l1152_115288


namespace NUMINAMATH_CALUDE_hole_large_enough_for_person_l1152_115278

/-- Represents a two-dimensional shape --/
structure Shape :=
  (perimeter : ℝ)

/-- Represents a hole cut in a shape --/
structure Hole :=
  (opening_size : ℝ)

/-- Represents a person --/
structure Person :=
  (size : ℝ)

/-- Function to create a hole in a shape --/
def cut_hole (s : Shape) : Hole :=
  sorry

/-- Theorem stating that it's possible to cut a hole in a shape that a person can fit through --/
theorem hole_large_enough_for_person (s : Shape) (p : Person) :
  ∃ (h : Hole), h = cut_hole s ∧ h.opening_size > p.size :=
sorry

end NUMINAMATH_CALUDE_hole_large_enough_for_person_l1152_115278


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l1152_115251

theorem mean_of_remaining_numbers (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 90 →
  (a + b + c + d) / 4 = 86.25 := by
sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l1152_115251


namespace NUMINAMATH_CALUDE_inequality_holds_for_nonzero_reals_l1152_115248

theorem inequality_holds_for_nonzero_reals (x : ℝ) (h : x ≠ 0) :
  (x^3 - 2*x^5 + x^6) / (x - 2*x^2 + x^4) ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_nonzero_reals_l1152_115248


namespace NUMINAMATH_CALUDE_product_112_54_l1152_115290

theorem product_112_54 : 112 * 54 = 6048 := by
  sorry

end NUMINAMATH_CALUDE_product_112_54_l1152_115290


namespace NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_l1152_115211

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : Nat :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The giraffe beauty contest voting structure -/
def giraffe_contest : VotingStructure :=
  { total_voters := 135
  , num_districts := 5
  , precincts_per_district := 9
  , voters_per_precinct := 3 }

theorem min_voters_for_tall_giraffe :
  min_voters_to_win giraffe_contest = 30 := by
  sorry

#eval min_voters_to_win giraffe_contest

end NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_l1152_115211


namespace NUMINAMATH_CALUDE_a_leq_0_necessary_not_sufficient_l1152_115200

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 * x^2 + a * x - 3/2
  else 2 * a * x^2 + x

-- Define what it means for a function to be monotonically decreasing
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≥ f y

-- Theorem statement
theorem a_leq_0_necessary_not_sufficient :
  (∃ a : ℝ, a ≤ 0 ∧ ¬(MonotonicallyDecreasing (f a))) ∧
  (∀ a : ℝ, MonotonicallyDecreasing (f a) → a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_a_leq_0_necessary_not_sufficient_l1152_115200


namespace NUMINAMATH_CALUDE_unique_solution_implies_equal_absolute_values_l1152_115277

theorem unique_solution_implies_equal_absolute_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃! x, a * (x - a)^2 + b * (x - b)^2 = 0) → |a| = |b| :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_equal_absolute_values_l1152_115277


namespace NUMINAMATH_CALUDE_quotient_digits_of_203_div_single_digit_l1152_115282

theorem quotient_digits_of_203_div_single_digit :
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 →
  ∃ q : ℕ, 203 / d = q ∧ (100 ≤ q ∧ q ≤ 999 ∨ 10 ≤ q ∧ q ≤ 99) :=
by sorry

end NUMINAMATH_CALUDE_quotient_digits_of_203_div_single_digit_l1152_115282


namespace NUMINAMATH_CALUDE_car_speed_problem_l1152_115240

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 270 →
  original_time = 6 →
  new_time_factor = 3 / 2 →
  let new_time := new_time_factor * original_time
  let new_speed := distance / new_time
  new_speed = 30 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1152_115240


namespace NUMINAMATH_CALUDE_melissa_points_per_game_l1152_115295

theorem melissa_points_per_game 
  (total_points : ℕ) 
  (num_games : ℕ) 
  (points_per_game : ℕ) 
  (h1 : total_points = 21) 
  (h2 : num_games = 3) 
  (h3 : total_points = num_games * points_per_game) : 
  points_per_game = 7 := by
  sorry

end NUMINAMATH_CALUDE_melissa_points_per_game_l1152_115295


namespace NUMINAMATH_CALUDE_meadow_area_is_24_l1152_115215

/-- The area of a meadow that was mowed in two days -/
def meadow_area : ℝ → Prop :=
  fun x => 
    -- Day 1: Half of the meadow plus 3 hectares
    let day1 := x / 2 + 3
    -- Remaining area after day 1
    let remaining := x - day1
    -- Day 2: One-third of the remaining area plus 6 hectares
    let day2 := remaining / 3 + 6
    -- The entire meadow is mowed after two days
    day1 + day2 = x

/-- Theorem: The area of the meadow is 24 hectares -/
theorem meadow_area_is_24 : meadow_area 24 := by
  sorry

#check meadow_area_is_24

end NUMINAMATH_CALUDE_meadow_area_is_24_l1152_115215


namespace NUMINAMATH_CALUDE_student_line_arrangements_l1152_115272

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of students who refuse to stand next to each other
def num_refusing_adjacent : ℕ := 2

-- Define the number of students who must stand at an end
def num_at_end : ℕ := 1

-- Function to calculate the number of arrangements
def num_arrangements (n : ℕ) (r : ℕ) (e : ℕ) : ℕ :=
  2 * (n.factorial - (n - r + 1).factorial * r.factorial)

-- Theorem statement
theorem student_line_arrangements :
  num_arrangements num_students num_refusing_adjacent num_at_end = 144 :=
by sorry

end NUMINAMATH_CALUDE_student_line_arrangements_l1152_115272


namespace NUMINAMATH_CALUDE_lighthouse_angle_elevation_l1152_115205

/-- Given a lighthouse and two ships, proves that the angle of elevation from one ship is 30 degrees -/
theorem lighthouse_angle_elevation 
  (h : ℝ) -- height of the lighthouse
  (d : ℝ) -- distance between the ships
  (θ₁ : ℝ) -- angle of elevation from the first ship
  (θ₂ : ℝ) -- angle of elevation from the second ship
  (h_height : h = 100) -- lighthouse height is 100 m
  (h_distance : d = 273.2050807568877) -- distance between ships
  (h_angle₂ : θ₂ = 45 * π / 180) -- angle from second ship is 45°
  : θ₁ = 30 * π / 180 := by 
sorry


end NUMINAMATH_CALUDE_lighthouse_angle_elevation_l1152_115205


namespace NUMINAMATH_CALUDE_base_8_representation_of_512_l1152_115235

/-- Converts a natural number to its base-8 representation as a list of digits (least significant first) -/
def to_base_8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 8) :: aux (m / 8)
    aux n

theorem base_8_representation_of_512 :
  to_base_8 512 = [0, 0, 0, 1] := by
sorry

end NUMINAMATH_CALUDE_base_8_representation_of_512_l1152_115235


namespace NUMINAMATH_CALUDE_probability_theorem_l1152_115255

/-- A permutation of the first n natural numbers -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The property that a permutation satisfies iₖ ≥ k - 3 for all k -/
def SatisfiesInequality (n : ℕ) (p : Permutation n) : Prop :=
  ∀ k : Fin n, (p k : ℕ) + 1 ≥ k.val - 2

/-- The number of permutations satisfying the inequality -/
def CountSatisfyingPermutations (n : ℕ) : ℕ :=
  (4 ^ (n - 3)) * 6

/-- The probability theorem -/
theorem probability_theorem (n : ℕ) (h : n > 3) :
  (CountSatisfyingPermutations n : ℚ) / (Nat.factorial n) =
  (↑(4 ^ (n - 3) * 6) : ℚ) / (Nat.factorial n) := by
  sorry


end NUMINAMATH_CALUDE_probability_theorem_l1152_115255


namespace NUMINAMATH_CALUDE_min_at_five_l1152_115296

/-- The function to be minimized -/
def f (c : ℝ) : ℝ := (c - 3)^2 + (c - 4)^2 + (c - 8)^2

/-- The theorem stating that 5 minimizes the function f -/
theorem min_at_five : 
  ∀ x : ℝ, f 5 ≤ f x :=
sorry

end NUMINAMATH_CALUDE_min_at_five_l1152_115296


namespace NUMINAMATH_CALUDE_angela_jacob_insect_ratio_l1152_115279

/-- Proves that the ratio of Angela's insects to Jacob's insects is 1:2 -/
theorem angela_jacob_insect_ratio :
  let dean_insects : ℕ := 30
  let jacob_insects : ℕ := 5 * dean_insects
  let angela_insects : ℕ := 75
  (angela_insects : ℚ) / jacob_insects = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angela_jacob_insect_ratio_l1152_115279


namespace NUMINAMATH_CALUDE_middle_group_frequency_l1152_115236

theorem middle_group_frequency 
  (sample_size : ℕ) 
  (num_rectangles : ℕ) 
  (middle_area_ratio : ℚ) : 
  sample_size = 300 →
  num_rectangles = 9 →
  middle_area_ratio = 1/5 →
  (middle_area_ratio * (1 - middle_area_ratio / (1 + middle_area_ratio))) * sample_size = 50 :=
by sorry

end NUMINAMATH_CALUDE_middle_group_frequency_l1152_115236


namespace NUMINAMATH_CALUDE_total_students_is_150_l1152_115238

/-- In a school, when there are 60 boys, girls become 60% of the total number of students. -/
def school_condition (total_students : ℕ) : Prop :=
  (60 : ℝ) / total_students + 0.6 = 1

/-- The theorem states that under the given condition, the total number of students is 150. -/
theorem total_students_is_150 : ∃ (total_students : ℕ), 
  school_condition total_students ∧ total_students = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_150_l1152_115238


namespace NUMINAMATH_CALUDE_liar_identification_l1152_115216

def original_number : ℕ := 2014315

def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

def statement_A (cards : Finset ℕ) : Prop :=
  ∃ (i j : ℕ), i ∈ cards ∧ j ∈ cards ∧ i ≠ j ∧
  is_divisible_by (swap_digits original_number i j) 8

def statement_B (cards : Finset ℕ) : Prop :=
  ∀ (i j : ℕ), i ∈ cards → j ∈ cards → i ≠ j →
  ¬is_divisible_by (swap_digits original_number i j) 9

def statement_C (cards : Finset ℕ) : Prop :=
  ∃ (i j : ℕ), i ∈ cards ∧ j ∈ cards ∧ i ≠ j ∧
  is_divisible_by (swap_digits original_number i j) 10

def statement_D (cards : Finset ℕ) : Prop :=
  ∃ (i j : ℕ), i ∈ cards ∧ j ∈ cards ∧ i ≠ j ∧
  is_divisible_by (swap_digits original_number i j) 11

theorem liar_identification :
  ∃ (cards_A cards_B cards_C cards_D : Finset ℕ),
    cards_A.card ≤ 2 ∧ cards_B.card ≤ 2 ∧ cards_C.card ≤ 2 ∧ cards_D.card ≤ 2 ∧
    cards_A ∪ cards_B ∪ cards_C ∪ cards_D = {0, 1, 2, 3, 4, 5} ∧
    cards_A ∩ cards_B = ∅ ∧ cards_A ∩ cards_C = ∅ ∧ cards_A ∩ cards_D = ∅ ∧
    cards_B ∩ cards_C = ∅ ∧ cards_B ∩ cards_D = ∅ ∧ cards_C ∩ cards_D = ∅ ∧
    statement_A cards_A ∧ statement_B cards_B ∧ ¬statement_C cards_C ∧ statement_D cards_D :=
by sorry

end NUMINAMATH_CALUDE_liar_identification_l1152_115216


namespace NUMINAMATH_CALUDE_function_properties_l1152_115298

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem function_properties (a : ℝ) (h : a > 1) :
  (∀ x, x ∈ Set.Icc 1 a → f a x ∈ Set.Icc 1 a) ∧
  (∀ x, x ∈ Set.Icc 1 a → f a x = x) →
  a = 2 ∧
  (∀ x ≤ 2, ∀ y ≤ x, f a x ≤ f a y) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 1 (a+1) → x₂ ∈ Set.Icc 1 (a+1) → |f a x₁ - f a x₂| ≤ 4) →
  2 ≤ a ∧ a ≤ 3 ∧
  (∃ x ∈ Set.Icc 1 3, f a x = 0) →
  Real.sqrt 5 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1152_115298


namespace NUMINAMATH_CALUDE_drain_rate_calculation_l1152_115207

/-- Represents the filling and draining system of a tank -/
structure TankSystem where
  capacity : ℝ
  fill_rate_A : ℝ
  fill_rate_B : ℝ
  drain_rate_C : ℝ
  cycle_time : ℝ
  total_time : ℝ

/-- Theorem stating the drain rate of pipe C given the system conditions -/
theorem drain_rate_calculation (s : TankSystem)
  (h1 : s.capacity = 950)
  (h2 : s.fill_rate_A = 40)
  (h3 : s.fill_rate_B = 30)
  (h4 : s.cycle_time = 3)
  (h5 : s.total_time = 57)
  (h6 : (s.total_time / s.cycle_time) * (s.fill_rate_A + s.fill_rate_B - s.drain_rate_C) = s.capacity) :
  s.drain_rate_C = 20 := by
  sorry

#check drain_rate_calculation

end NUMINAMATH_CALUDE_drain_rate_calculation_l1152_115207


namespace NUMINAMATH_CALUDE_circle_tangent_properties_l1152_115274

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 3 = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define tangent lines PA and PB
def tangent_PA (xa ya xp yp : ℝ) : Prop := 
  circle_M xa ya ∧ point_P xp yp ∧ (xa - xp) * (xa + 1) + (ya - yp) * ya = 0

def tangent_PB (xb yb xp yp : ℝ) : Prop := 
  circle_M xb yb ∧ point_P xp yp ∧ (xb - xp) * (xb + 1) + (yb - yp) * yb = 0

-- Theorem statement
theorem circle_tangent_properties :
  ∃ (min_area : ℝ) (chord_length : ℝ) (fixed_point : ℝ × ℝ),
    (min_area = 2 * Real.sqrt 3) ∧
    (chord_length = Real.sqrt 6) ∧
    (fixed_point = (-1/2, -1/2)) ∧
    (∀ xa ya xb yb xp yp : ℝ,
      tangent_PA xa ya xp yp →
      tangent_PB xb yb xp yp →
      -- 1. Minimum area of quadrilateral PAMB
      (xa - xp)^2 + (ya - yp)^2 + (xb - xp)^2 + (yb - yp)^2 ≥ min_area^2 ∧
      -- 2. Length of chord AB when |PA| is shortest
      ((xa - xp)^2 + (ya - yp)^2 = (xb - xp)^2 + (yb - yp)^2 →
        (xa - xb)^2 + (ya - yb)^2 = chord_length^2) ∧
      -- 3. Line AB passes through the fixed point
      (ya - yb) * (fixed_point.1 - xa) = (xa - xb) * (fixed_point.2 - ya)) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_properties_l1152_115274


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1152_115224

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) (exterior_angle : ℝ) : 
  interior_angle = 150 →
  exterior_angle = 180 - interior_angle →
  n * exterior_angle = 360 →
  n = 12 := by
  sorry

#check regular_polygon_sides

end NUMINAMATH_CALUDE_regular_polygon_sides_l1152_115224


namespace NUMINAMATH_CALUDE_function_inequalities_l1152_115252

/-- Given a function f(x) = x^2 - (a + 2)x + 4, where a is a real number -/
def f (a x : ℝ) : ℝ := x^2 - (a + 2)*x + 4

theorem function_inequalities (a : ℝ) :
  (∀ x, a < 2 → (f a x ≤ -2*a + 4 ↔ a ≤ x ∧ x ≤ 2)) ∧
  (∀ x, a = 2 → (f a x ≤ -2*a + 4 ↔ x = 2)) ∧
  (∀ x, a > 2 → (f a x ≤ -2*a + 4 ↔ 2 ≤ x ∧ x ≤ a)) ∧
  (∀ x, x ∈ Set.Icc 1 4 → f a x + a + 1 ≥ 0 ↔ a ∈ Set.Iic 4) :=
by sorry


end NUMINAMATH_CALUDE_function_inequalities_l1152_115252


namespace NUMINAMATH_CALUDE_final_amount_proof_l1152_115217

/-- Calculates the final amount after two years of compound interest with different rates each year. -/
def final_amount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount1 := initial * (1 + rate1)
  amount1 * (1 + rate2)

/-- Theorem stating that given the specific initial amount and interest rates, 
    the final amount after two years is as calculated. -/
theorem final_amount_proof :
  final_amount 6552 0.04 0.05 = 7154.784 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_proof_l1152_115217


namespace NUMINAMATH_CALUDE_tea_consumption_discrepancy_l1152_115284

theorem tea_consumption_discrepancy 
  (box_size : ℕ) 
  (cups_per_bag_min cups_per_bag_max : ℕ) 
  (darya_cups marya_cups : ℕ) :
  cups_per_bag_min = 3 →
  cups_per_bag_max = 4 →
  darya_cups = 74 →
  marya_cups = 105 →
  (∃ n : ℕ, n * cups_per_bag_min ≤ darya_cups ∧ darya_cups < (n + 1) * cups_per_bag_min ∧
            n * cups_per_bag_min ≤ marya_cups ∧ marya_cups < (n + 1) * cups_per_bag_min) →
  (∃ m : ℕ, m * cups_per_bag_max ≤ darya_cups ∧ darya_cups < (m + 1) * cups_per_bag_max ∧
            m * cups_per_bag_max ≤ marya_cups ∧ marya_cups < (m + 1) * cups_per_bag_max) →
  False :=
by sorry

end NUMINAMATH_CALUDE_tea_consumption_discrepancy_l1152_115284


namespace NUMINAMATH_CALUDE_mitch_family_milk_consumption_l1152_115287

/-- The total milk consumption in cartons for Mitch's family in one week -/
def total_milk_consumption (regular_milk soy_milk : ℝ) : ℝ :=
  regular_milk + soy_milk

/-- Proof that Mitch's family's total milk consumption is 0.6 cartons in one week -/
theorem mitch_family_milk_consumption :
  let regular_milk : ℝ := 0.5
  let soy_milk : ℝ := 0.1
  total_milk_consumption regular_milk soy_milk = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_mitch_family_milk_consumption_l1152_115287


namespace NUMINAMATH_CALUDE_multiply_inverse_square_equals_cube_l1152_115280

theorem multiply_inverse_square_equals_cube (x : ℝ) : x * (1/7)^2 = 7^3 ↔ x = 16807 := by
  sorry

end NUMINAMATH_CALUDE_multiply_inverse_square_equals_cube_l1152_115280


namespace NUMINAMATH_CALUDE_maggie_total_spent_l1152_115293

def plant_books : ℕ := 20
def fish_books : ℕ := 7
def magazines : ℕ := 25
def book_cost : ℕ := 25
def magazine_cost : ℕ := 5

theorem maggie_total_spent : 
  (plant_books + fish_books) * book_cost + magazines * magazine_cost = 800 := by
sorry

end NUMINAMATH_CALUDE_maggie_total_spent_l1152_115293


namespace NUMINAMATH_CALUDE_total_cost_is_55_l1152_115222

/-- The total cost of two pairs of shoes, where the first pair costs $22 and the second pair is 50% more expensive than the first pair. -/
def total_cost : ℝ :=
  let first_pair_cost : ℝ := 22
  let second_pair_cost : ℝ := first_pair_cost * 1.5
  first_pair_cost + second_pair_cost

/-- Theorem stating that the total cost of the two pairs of shoes is $55. -/
theorem total_cost_is_55 : total_cost = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_55_l1152_115222


namespace NUMINAMATH_CALUDE_clock_angle_at_3_37_clock_angle_proof_l1152_115269

/-- The acute angle between clock hands at 3:37 -/
theorem clock_angle_at_3_37 : ℝ :=
  let hours : ℕ := 3
  let minutes : ℕ := 37
  let total_hours : ℕ := 12
  let degrees_per_hour : ℝ := 30

  let minute_angle : ℝ := (minutes : ℝ) / 60 * 360
  let hour_angle : ℝ := (hours : ℝ) * degrees_per_hour + (minutes : ℝ) / 60 * degrees_per_hour

  let angle_diff : ℝ := |minute_angle - hour_angle|
  let acute_angle : ℝ := min angle_diff (360 - angle_diff)

  113.5

/-- Proof of the clock angle theorem -/
theorem clock_angle_proof : clock_angle_at_3_37 = 113.5 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_37_clock_angle_proof_l1152_115269


namespace NUMINAMATH_CALUDE_mikes_age_l1152_115258

theorem mikes_age (claire_age jessica_age mike_age : ℕ) : 
  jessica_age = claire_age + 6 →
  claire_age + 2 = 20 →
  mike_age = 2 * (jessica_age - 3) →
  mike_age = 42 := by
  sorry

end NUMINAMATH_CALUDE_mikes_age_l1152_115258


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1152_115246

/-- An arithmetic sequence {a_n} with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 + a 7 = 22)
  (h3 : a 4 + a 10 = 40) :
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1152_115246


namespace NUMINAMATH_CALUDE_bella_pizza_consumption_l1152_115264

theorem bella_pizza_consumption 
  (rachel_pizza : ℕ) 
  (total_pizza : ℕ) 
  (h1 : rachel_pizza = 598)
  (h2 : total_pizza = 952) :
  total_pizza - rachel_pizza = 354 := by
sorry

end NUMINAMATH_CALUDE_bella_pizza_consumption_l1152_115264


namespace NUMINAMATH_CALUDE_square_with_tens_digit_seven_l1152_115292

/-- Given a number A with more than one digit, if the tens digit of A^2 is 7, 
    then the units digit of A^2 is 6. -/
theorem square_with_tens_digit_seven (A : ℕ) : 
  A > 9 → 
  (A^2 / 10) % 10 = 7 → 
  A^2 % 10 = 6 :=
by sorry

end NUMINAMATH_CALUDE_square_with_tens_digit_seven_l1152_115292


namespace NUMINAMATH_CALUDE_class_size_proof_l1152_115210

theorem class_size_proof (n : ℕ) : 
  (n / 6 : ℚ) = (n / 18 : ℚ) + 4 →  -- One-sixth wear glasses, split into girls and boys
  n = 36 :=
by
  sorry

#check class_size_proof

end NUMINAMATH_CALUDE_class_size_proof_l1152_115210


namespace NUMINAMATH_CALUDE_factorization_equality_l1152_115263

theorem factorization_equality (x : ℝ) :
  (x^2 + 5*x + 2) * (x^2 + 5*x + 3) - 12 = (x + 2) * (x + 3) * (x^2 + 5*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1152_115263


namespace NUMINAMATH_CALUDE_bigger_part_problem_l1152_115281

theorem bigger_part_problem (x y : ℝ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) 
  (h3 : x > 0) (h4 : y > 0) : max x y = 34 := by
  sorry

end NUMINAMATH_CALUDE_bigger_part_problem_l1152_115281


namespace NUMINAMATH_CALUDE_subset_implies_m_squared_l1152_115253

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {-1, 3, m^2}
def B : Set ℝ := {3, 4}

-- State the theorem
theorem subset_implies_m_squared (m : ℝ) : B ⊆ A m → (m = 2 ∨ m = -2) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_squared_l1152_115253


namespace NUMINAMATH_CALUDE_gumball_calculation_l1152_115219

/-- The number of gumballs originally in the dispenser -/
def original_gumballs : ℝ := 100

/-- The fraction of gumballs remaining after each day -/
def daily_remaining_fraction : ℝ := 0.7

/-- The number of days that have passed -/
def days : ℕ := 3

/-- The number of gumballs remaining after 3 days -/
def remaining_gumballs : ℝ := 34.3

/-- Theorem stating that the original number of gumballs is correct -/
theorem gumball_calculation :
  original_gumballs * daily_remaining_fraction ^ days = remaining_gumballs := by
  sorry

end NUMINAMATH_CALUDE_gumball_calculation_l1152_115219


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l1152_115209

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]

theorem matrix_inverse_proof :
  A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l1152_115209


namespace NUMINAMATH_CALUDE_min_value_iff_lower_bound_l1152_115231

/-- Given a function f: ℝ → ℝ and a constant M, prove that the following are equivalent:
    1) For all x ∈ ℝ, f(x) ≥ M
    2) M is the minimum value of f -/
theorem min_value_iff_lower_bound (f : ℝ → ℝ) (M : ℝ) :
  (∀ x, f x ≥ M) ↔ (∀ x, f x ≥ M ∧ ∃ y, f y = M) :=
by sorry

end NUMINAMATH_CALUDE_min_value_iff_lower_bound_l1152_115231


namespace NUMINAMATH_CALUDE_range_of_f_l1152_115242

def f (x : ℤ) : ℤ := x + 1

def domain : Set ℤ := {-1, 1, 2}

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ domain, f x = y} = {0, 2, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1152_115242


namespace NUMINAMATH_CALUDE_car_meeting_problem_l1152_115218

/-- Represents a car with a speed and initial position -/
structure Car where
  speed : ℝ
  initial_position : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  car_x : Car
  car_y : Car
  first_meeting_time : ℝ
  distance_between_meetings : ℝ

/-- The theorem statement -/
theorem car_meeting_problem (setup : ProblemSetup)
  (h1 : setup.car_x.speed = 50)
  (h2 : setup.first_meeting_time = 1)
  (h3 : setup.distance_between_meetings = 20)
  (h4 : setup.car_x.initial_position = 0)
  (h5 : setup.car_y.initial_position = setup.car_x.initial_position + 
        setup.car_x.speed * setup.first_meeting_time + 
        setup.car_y.speed * setup.first_meeting_time) :
  setup.car_y.initial_position - setup.car_x.initial_position = 110 ∧
  setup.car_y.speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_meeting_problem_l1152_115218


namespace NUMINAMATH_CALUDE_tiling_condition_l1152_115268

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a square -/
inductive Color
  | Black
  | White

/-- Determines the color of a square based on its position -/
def squareColor (s : Square) : Color :=
  if (s.row.val + s.col.val) % 2 = 0 then Color.Black else Color.White

/-- Represents a chessboard with two squares removed -/
structure ChessboardWithRemovedSquares where
  removed1 : Square
  removed2 : Square
  different : removed1 ≠ removed2

/-- Represents the possibility of tiling the chessboard with dominoes -/
def canTile (board : ChessboardWithRemovedSquares) : Prop :=
  squareColor board.removed1 ≠ squareColor board.removed2

/-- Theorem stating the condition for possible tiling -/
theorem tiling_condition (board : ChessboardWithRemovedSquares) :
  canTile board ↔ squareColor board.removed1 ≠ squareColor board.removed2 := by sorry

end NUMINAMATH_CALUDE_tiling_condition_l1152_115268


namespace NUMINAMATH_CALUDE_total_cost_proof_l1152_115201

def squat_rack_cost : ℕ := 2500
def barbell_cost_ratio : ℚ := 1 / 10

theorem total_cost_proof :
  squat_rack_cost + (squat_rack_cost : ℚ) * barbell_cost_ratio = 2750 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l1152_115201


namespace NUMINAMATH_CALUDE_fourth_team_odd_l1152_115270

/-- Calculates the odd for the fourth team in a soccer bet -/
theorem fourth_team_odd (odd1 odd2 odd3 : ℝ) (bet_amount expected_winnings : ℝ) :
  odd1 = 1.28 →
  odd2 = 5.23 →
  odd3 = 3.25 →
  bet_amount = 5.00 →
  expected_winnings = 223.0072 →
  ∃ (odd4 : ℝ), abs (odd4 - 2.061) < 0.001 ∧ 
    odd1 * odd2 * odd3 * odd4 = expected_winnings / bet_amount :=
by
  sorry

#check fourth_team_odd

end NUMINAMATH_CALUDE_fourth_team_odd_l1152_115270


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l1152_115221

theorem quadratic_equation_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ -x₁^2 - 3*x₁ + 3 = 0 ∧ -x₂^2 - 3*x₂ + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l1152_115221


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l1152_115237

/-- The probability of picking 2 red balls from a bag with 3 red, 2 blue, and 3 green balls -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) :
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 3 →
  blue_balls = 2 →
  green_balls = 3 →
  (Nat.choose red_balls 2 : ℚ) / (Nat.choose total_balls 2) = 3 / 28 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l1152_115237


namespace NUMINAMATH_CALUDE_unique_zero_point_condition_l1152_115213

def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * x - a

theorem unique_zero_point_condition (a : ℝ) :
  (∃! x : ℝ, x ∈ Set.Ioo (-1) 1 ∧ f a x = 0) ↔ (1 < a ∧ a < 5) ∨ a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_point_condition_l1152_115213


namespace NUMINAMATH_CALUDE_triathlon_problem_l1152_115262

/-- Triathlon problem -/
theorem triathlon_problem 
  (swim_distance : ℝ) 
  (cycle_distance : ℝ) 
  (run_distance : ℝ)
  (total_time : ℝ)
  (practice_swim_time : ℝ)
  (practice_cycle_time : ℝ)
  (practice_run_time : ℝ)
  (practice_total_distance : ℝ)
  (h_swim_distance : swim_distance = 1)
  (h_cycle_distance : cycle_distance = 25)
  (h_run_distance : run_distance = 4)
  (h_total_time : total_time = 5/4)
  (h_practice_swim_time : practice_swim_time = 1/16)
  (h_practice_cycle_time : practice_cycle_time = 1/49)
  (h_practice_run_time : practice_run_time = 1/49)
  (h_practice_total_distance : practice_total_distance = 5/4)
  (h_positive_speeds : ∀ v : ℝ, v > 0 → v + 1/v ≥ 2) :
  ∃ (cycle_time cycle_speed : ℝ),
    cycle_time = 5/7 ∧ 
    cycle_speed = 35 ∧
    cycle_distance / cycle_speed = cycle_time ∧
    swim_distance / (swim_distance / practice_swim_time) + 
    cycle_distance / cycle_speed + 
    run_distance / (run_distance / practice_run_time) = total_time ∧
    practice_swim_time * (swim_distance / practice_swim_time) + 
    practice_cycle_time * cycle_speed + 
    practice_run_time * (run_distance / practice_run_time) = practice_total_distance :=
by sorry


end NUMINAMATH_CALUDE_triathlon_problem_l1152_115262


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1152_115249

theorem simplify_and_evaluate (x : ℝ) (h : x = 2) :
  (1 + 1/x) / ((x^2 - 1) / x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1152_115249


namespace NUMINAMATH_CALUDE_rectangle_sides_and_solvability_l1152_115297

/-- Given a rectangle with perimeter k and area t, this theorem proves the lengths of its sides
    and the condition for solvability. -/
theorem rectangle_sides_and_solvability (k t : ℝ) (k_pos : k > 0) (t_pos : t > 0) :
  let a := (k + Real.sqrt (k^2 - 16*t)) / 4
  let b := (k - Real.sqrt (k^2 - 16*t)) / 4
  (k^2 ≥ 16*t) →
  (a + b = k/2 ∧ a * b = t ∧ a > 0 ∧ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_sides_and_solvability_l1152_115297


namespace NUMINAMATH_CALUDE_incorrect_multiplication_l1152_115228

theorem incorrect_multiplication : (79133 * 111107) % 9 ≠ 8792240231 % 9 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_multiplication_l1152_115228


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1152_115250

theorem cricket_team_average_age (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) :
  team_size = 11 →
  captain_age = 26 →
  wicket_keeper_age_diff = 3 →
  let total_age := team_size * (captain_age + wicket_keeper_age_diff + 2) / 2
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + captain_age + wicket_keeper_age_diff)
  (remaining_age / remaining_players) + 1 = total_age / team_size →
  total_age / team_size = 32 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l1152_115250


namespace NUMINAMATH_CALUDE_stamps_per_page_l1152_115230

theorem stamps_per_page (a b c : ℕ) (ha : a = 1200) (hb : b = 1800) (hc : c = 2400) :
  Nat.gcd a (Nat.gcd b c) = 600 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_page_l1152_115230


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l1152_115225

/-- Represents the maximum distance a car can travel with tire swapping -/
def max_distance (front_tire_life rear_tire_life : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum distance for the given tire lifespans -/
theorem max_distance_for_given_tires :
  max_distance 24000 36000 = 28800 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l1152_115225


namespace NUMINAMATH_CALUDE_inequality_proof_l1152_115259

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) : 
  1/(1+a*b) + 1/(1+b*c) + 1/(1+c*a) ≥ 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1152_115259


namespace NUMINAMATH_CALUDE_worker_travel_time_l1152_115275

theorem worker_travel_time (normal_speed : ℝ) (slower_speed : ℝ) (usual_time : ℝ) (delay : ℝ) :
  slower_speed = (5 / 6) * normal_speed →
  delay = 12 →
  slower_speed * (usual_time + delay) = normal_speed * usual_time →
  usual_time = 60 := by
sorry

end NUMINAMATH_CALUDE_worker_travel_time_l1152_115275


namespace NUMINAMATH_CALUDE_fraction_sum_l1152_115204

theorem fraction_sum (m n : ℕ) (hcoprime : Nat.Coprime m n) 
  (heq : (2013 * 2013) / (2014 * 2014 + 2012) = n / m) : 
  m + n = 1343 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1152_115204


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1152_115299

theorem quadratic_equation_solution (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let x₁ : ℝ := 4*a/(3*b)
  let x₂ : ℝ := -3*b/(4*a)
  (12*a*b*x₁^2 - (16*a^2 - 9*b^2)*x₁ - 12*a*b = 0) ∧
  (12*a*b*x₂^2 - (16*a^2 - 9*b^2)*x₂ - 12*a*b = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1152_115299


namespace NUMINAMATH_CALUDE_tom_and_michael_have_nine_robots_l1152_115208

/-- The number of car robots Bob has -/
def bob_robots : ℕ := 81

/-- The factor by which Bob's robots outnumber Tom and Michael's combined -/
def factor : ℕ := 9

/-- The number of car robots Tom and Michael have combined -/
def tom_and_michael_robots : ℕ := bob_robots / factor

theorem tom_and_michael_have_nine_robots : tom_and_michael_robots = 9 := by
  sorry

end NUMINAMATH_CALUDE_tom_and_michael_have_nine_robots_l1152_115208


namespace NUMINAMATH_CALUDE_roses_distribution_l1152_115244

theorem roses_distribution (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) 
  (h1 : initial_roses = 40)
  (h2 : stolen_roses = 4)
  (h3 : people = 9)
  : (initial_roses - stolen_roses) / people = 4 := by
  sorry

end NUMINAMATH_CALUDE_roses_distribution_l1152_115244


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1152_115241

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ), A = -1/2 ∧ B = 5/2 ∧ C = -5 ∧
  ∀ (x : ℚ), x ≠ 0 → x^2 ≠ 2 →
  (2*x^2 - 5*x + 1) / (x^3 - 2*x) = A / x + (B*x + C) / (x^2 - 2) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1152_115241


namespace NUMINAMATH_CALUDE_factorize_nine_minus_a_squared_l1152_115203

theorem factorize_nine_minus_a_squared (a : ℝ) : 9 - a^2 = (3 + a) * (3 - a) := by
  sorry

end NUMINAMATH_CALUDE_factorize_nine_minus_a_squared_l1152_115203


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l1152_115223

/-- Proves that given the initial profit is 320% of the cost, and after a cost increase
    (with constant selling price) the profit becomes 66.67% of the selling price,
    then the cost increase percentage is 40%. -/
theorem cost_increase_percentage (C : ℝ) (X : ℝ) : 
  C > 0 →                           -- Assuming positive initial cost
  let S := 4.2 * C                  -- Initial selling price
  let new_profit := 3.2 * C - (X / 100) * C  -- New profit after cost increase
  3.2 * C = 320 / 100 * C →         -- Initial profit is 320% of cost
  new_profit = 2 / 3 * S →          -- New profit is 66.67% of selling price
  X = 40 :=                         -- Cost increase percentage is 40%
by
  sorry


end NUMINAMATH_CALUDE_cost_increase_percentage_l1152_115223


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1152_115261

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + Real.sin x + 1 < 0) ↔ (∃ x : ℝ, x^2 + Real.sin x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1152_115261


namespace NUMINAMATH_CALUDE_sum_is_composite_l1152_115276

theorem sum_is_composite (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b + c + d = x * y :=
sorry

end NUMINAMATH_CALUDE_sum_is_composite_l1152_115276


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1152_115256

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a₁ = -16 and a₄ = 8, prove that a₇ = -4 -/
theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_a1 : a 1 = -16)
  (h_a4 : a 4 = 8) :
  a 7 = -4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1152_115256


namespace NUMINAMATH_CALUDE_train_crossing_time_l1152_115271

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 150 ∧ 
  train_speed_kmh = 90 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 6 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1152_115271


namespace NUMINAMATH_CALUDE_strawberry_pies_count_l1152_115212

/-- Given a total number of pies and a ratio for different types of pies,
    calculate the number of pies of a specific type. -/
theorem strawberry_pies_count
  (total_pies : ℕ)
  (apple_ratio blueberry_ratio cherry_ratio strawberry_ratio : ℕ)
  (h_total : total_pies = 48)
  (h_ratios : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4 ∧ strawberry_ratio = 1) :
  (strawberry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio + strawberry_ratio) = 4 :=
by sorry

end NUMINAMATH_CALUDE_strawberry_pies_count_l1152_115212


namespace NUMINAMATH_CALUDE_plywood_width_l1152_115257

theorem plywood_width (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 24 →
  length = 4 →
  area = length * width →
  width = 6 := by
sorry

end NUMINAMATH_CALUDE_plywood_width_l1152_115257


namespace NUMINAMATH_CALUDE_largest_angle_measure_l1152_115220

/-- A triangle XYZ is obtuse and isosceles with one of the equal angles measuring 30 degrees. -/
structure ObtuseIsoscelesTriangle where
  X : ℝ
  Y : ℝ
  Z : ℝ
  sum_180 : X + Y + Z = 180
  obtuse : Z > 90
  isosceles : X = Y
  x_measure : X = 30

/-- The largest interior angle of an obtuse isosceles triangle with one equal angle measuring 30 degrees is 120 degrees. -/
theorem largest_angle_measure (t : ObtuseIsoscelesTriangle) : t.Z = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_measure_l1152_115220


namespace NUMINAMATH_CALUDE_nested_f_application_l1152_115202

def f (x : ℝ) : ℝ := x + 1

theorem nested_f_application : f (f (f (f (f 3)))) = 8 := by sorry

end NUMINAMATH_CALUDE_nested_f_application_l1152_115202


namespace NUMINAMATH_CALUDE_basketball_score_proof_l1152_115273

theorem basketball_score_proof :
  ∀ (S : ℕ) (x : ℕ),
    S > 0 →
    S % 4 = 0 →
    S % 7 = 0 →
    S / 4 + 2 * S / 7 + 15 + x = S →
    x ≤ 14 →
    x = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l1152_115273


namespace NUMINAMATH_CALUDE_parallel_iff_m_eq_neg_three_l1152_115266

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Vector a as defined in the problem -/
def a : ℝ × ℝ := (1, -2)

/-- Vector b as defined in the problem -/
def b (m : ℝ) : ℝ × ℝ := (1 + m, 1 - m)

/-- The main theorem: vectors a and b are parallel if and only if m = -3 -/
theorem parallel_iff_m_eq_neg_three :
  ∀ m : ℝ, are_parallel a (b m) ↔ m = -3 := by sorry

end NUMINAMATH_CALUDE_parallel_iff_m_eq_neg_three_l1152_115266


namespace NUMINAMATH_CALUDE_inequality_solution_l1152_115245

theorem inequality_solution (x : ℝ) :
  (3 - x) / (5 + 2*x) ≤ 0 ↔ x < -5/2 ∨ x ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1152_115245


namespace NUMINAMATH_CALUDE_cube_sum_zero_l1152_115233

theorem cube_sum_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum_zero : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_zero_l1152_115233


namespace NUMINAMATH_CALUDE_quadratic_equation_b_range_l1152_115243

theorem quadratic_equation_b_range :
  ∀ (b c : ℝ),
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ x^2 + b*x + c = 0) →
  (0 ≤ 3*b + c) →
  (3*b + c ≤ 3) →
  b ∈ Set.Icc 0 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_b_range_l1152_115243


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1152_115232

theorem geometric_sequence_property (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ r > 0, ∀ n, a (n + 1) = r * a n)
  (h_sum : a 1 + a 2 + a 3 = 18)
  (h_inv_sum : 1 / a 1 + 1 / a 2 + 1 / a 3 = 2) :
  a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1152_115232


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_ratio_range_l1152_115214

theorem triangle_geometric_sequence_ratio_range (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : b^2 = a*c) : 2 ≤ (b/a + a/b) ∧ (b/a + a/b) < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_geometric_sequence_ratio_range_l1152_115214


namespace NUMINAMATH_CALUDE_bus_trip_difference_l1152_115283

def bus_trip (initial : ℕ) 
             (stop1_off stop1_on : ℕ) 
             (stop2_off stop2_on : ℕ) 
             (stop3_off stop3_on : ℕ) 
             (stop4_off stop4_on : ℕ) : ℕ :=
  let after_stop1 := initial - stop1_off + stop1_on
  let after_stop2 := after_stop1 - stop2_off + stop2_on
  let after_stop3 := after_stop2 - stop3_off + stop3_on
  let final := after_stop3 - stop4_off + stop4_on
  initial - final

theorem bus_trip_difference :
  bus_trip 41 12 5 7 10 14 3 9 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_difference_l1152_115283


namespace NUMINAMATH_CALUDE_crate_tower_probability_l1152_115285

def crate_dimensions := (3, 4, 6)
def num_crates := 11
def target_height := 50

def valid_arrangements (a b c : ℕ) : ℕ :=
  if a + b + c = num_crates ∧ 3 * a + 4 * b + 6 * c = target_height
  then Nat.factorial num_crates / (Nat.factorial a * Nat.factorial b * Nat.factorial c)
  else 0

def total_valid_arrangements : ℕ :=
  valid_arrangements 4 2 5 + valid_arrangements 2 5 4 + valid_arrangements 0 8 3

def total_possible_arrangements : ℕ := 3^num_crates

theorem crate_tower_probability : 
  (total_valid_arrangements : ℚ) / total_possible_arrangements = 72 / 115 := by
  sorry

end NUMINAMATH_CALUDE_crate_tower_probability_l1152_115285


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_l1152_115294

/-- Configuration of five spheres with a tetrahedron -/
structure SpheresTetrahedron where
  /-- Radius of each sphere -/
  radius : ℝ
  /-- Distance between centers of adjacent spheres on the square -/
  square_side : ℝ
  /-- Height of the top sphere's center above the square -/
  height : ℝ
  /-- Edge length of the tetrahedron -/
  tetra_edge : ℝ
  /-- The radius is 2 -/
  radius_eq : radius = 2
  /-- The square side is twice the diameter -/
  square_side_eq : square_side = 4 * radius
  /-- The height is equal to the diameter -/
  height_eq : height = 2 * radius
  /-- The tetrahedron edge is the distance from a lower sphere to the top sphere -/
  tetra_edge_eq : tetra_edge ^ 2 = square_side ^ 2 + height ^ 2

/-- Theorem: The edge length of the tetrahedron is 4√2 -/
theorem tetrahedron_edge_length (config : SpheresTetrahedron) : 
  config.tetra_edge = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_l1152_115294
