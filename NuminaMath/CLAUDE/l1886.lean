import Mathlib

namespace NUMINAMATH_CALUDE_blackboard_final_product_l1886_188615

/-- Represents the count of each number on the blackboard -/
structure BoardState :=
  (ones : ℕ)
  (twos : ℕ)
  (threes : ℕ)
  (fours : ℕ)

/-- Represents a single operation on the board -/
inductive Operation
  | erase_123_add_4
  | erase_124_add_3
  | erase_134_add_2
  | erase_234_add_1

/-- Applies an operation to a board state -/
def apply_operation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.erase_123_add_4 => ⟨state.ones - 1, state.twos - 1, state.threes - 1, state.fours + 2⟩
  | Operation.erase_124_add_3 => ⟨state.ones - 1, state.twos - 1, state.threes + 2, state.fours - 1⟩
  | Operation.erase_134_add_2 => ⟨state.ones - 1, state.twos + 2, state.threes - 1, state.fours - 1⟩
  | Operation.erase_234_add_1 => ⟨state.ones + 2, state.twos - 1, state.threes - 1, state.fours - 1⟩

/-- Checks if a board state is in the final condition (only 3 numbers left) -/
def is_final_state (state : BoardState) : Prop :=
  (state.ones = 0 ∧ state.twos = 2 ∧ state.threes = 1 ∧ state.fours = 0) ∨
  (state.ones = 0 ∧ state.twos = 1 ∧ state.threes = 2 ∧ state.fours = 0) ∨
  (state.ones = 0 ∧ state.twos = 2 ∧ state.threes = 0 ∧ state.fours = 1) ∨
  (state.ones = 1 ∧ state.twos = 0 ∧ state.threes = 2 ∧ state.fours = 0) ∨
  (state.ones = 1 ∧ state.twos = 2 ∧ state.threes = 0 ∧ state.fours = 0) ∨
  (state.ones = 2 ∧ state.twos = 0 ∧ state.threes = 1 ∧ state.fours = 0) ∨
  (state.ones = 2 ∧ state.twos = 1 ∧ state.threes = 0 ∧ state.fours = 0)

/-- Calculates the product of the last three remaining numbers -/
def final_product (state : BoardState) : ℕ :=
  if state.ones > 0 then state.ones else 1 *
  if state.twos > 0 then state.twos else 1 *
  if state.threes > 0 then state.threes else 1 *
  if state.fours > 0 then state.fours else 1

/-- The main theorem to prove -/
theorem blackboard_final_product :
  ∀ (operations : List Operation),
  let initial_state : BoardState := ⟨11, 22, 33, 44⟩
  let final_state := operations.foldl apply_operation initial_state
  is_final_state final_state → final_product final_state = 12 :=
sorry

end NUMINAMATH_CALUDE_blackboard_final_product_l1886_188615


namespace NUMINAMATH_CALUDE_investment_proof_l1886_188623

/-- Represents the total amount invested -/
def total_investment : ℝ := 10000

/-- Represents the amount invested at 6% interest -/
def investment_at_6_percent : ℝ := 7200

/-- Represents the annual interest rate for the first part of the investment -/
def interest_rate_1 : ℝ := 0.06

/-- Represents the annual interest rate for the second part of the investment -/
def interest_rate_2 : ℝ := 0.09

/-- Represents the total interest received after one year -/
def total_interest : ℝ := 684

theorem investment_proof : 
  interest_rate_1 * investment_at_6_percent + 
  interest_rate_2 * (total_investment - investment_at_6_percent) = 
  total_interest :=
by sorry

end NUMINAMATH_CALUDE_investment_proof_l1886_188623


namespace NUMINAMATH_CALUDE_fourth_red_ball_is_24_l1886_188645

/-- Represents a random number table --/
def RandomTable : List (List Nat) :=
  [[2, 9, 7, 63, 4, 1, 32, 8, 4, 14, 2, 4, 1],
   [8, 3, 0, 39, 8, 2, 25, 8, 8, 82, 4, 1, 0],
   [5, 5, 5, 68, 5, 2, 66, 1, 6, 68, 2, 3, 1]]

/-- Checks if a number is a valid red ball number --/
def isValidRedBall (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 33

/-- Selects valid red ball numbers from a list --/
def selectValidNumbers (numbers : List Nat) : List Nat :=
  numbers.filter isValidRedBall

/-- Flattens the random table into a single list, starting from the specified position --/
def flattenTableFrom (table : List (List Nat)) (startRow startCol : Nat) : List Nat :=
  let rowsFromStart := table.drop startRow
  let firstRow := (rowsFromStart.head!).drop startCol
  firstRow ++ (rowsFromStart.tail!).join

/-- The main theorem to prove --/
theorem fourth_red_ball_is_24 :
  let flattenedTable := flattenTableFrom RandomTable 0 8
  let validNumbers := selectValidNumbers flattenedTable
  validNumbers[3] = 24 := by sorry

end NUMINAMATH_CALUDE_fourth_red_ball_is_24_l1886_188645


namespace NUMINAMATH_CALUDE_age_difference_l1886_188672

theorem age_difference (A B : ℕ) : B = 35 → A + 10 = 2 * (B - 10) → A - B = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1886_188672


namespace NUMINAMATH_CALUDE_interval_sum_theorem_l1886_188652

open Real

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

/-- The function g as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := (floor x : ℝ) * (2013^(frac x) - 2)

/-- The theorem statement -/
theorem interval_sum_theorem :
  ∃ (S : Set ℝ), S = {x : ℝ | 1 ≤ x ∧ x < 2013 ∧ g x ≤ 0} ∧
  (∫ x in S, 1) = 2012 * (log 2 / log 2013) := by sorry

end NUMINAMATH_CALUDE_interval_sum_theorem_l1886_188652


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_three_l1886_188699

theorem sqrt_difference_equals_two_sqrt_three :
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_three_l1886_188699


namespace NUMINAMATH_CALUDE_mark_reading_time_l1886_188694

/-- Calculates Mark's total weekly reading time given his daily reading time and weekly increase. -/
def weekly_reading_time (x : ℝ) (y : ℝ) : ℝ :=
  7 * x + y

/-- Theorem stating that Mark's total weekly reading time is 7x + y hours -/
theorem mark_reading_time (x y : ℝ) :
  weekly_reading_time x y = 7 * x + y := by
  sorry

end NUMINAMATH_CALUDE_mark_reading_time_l1886_188694


namespace NUMINAMATH_CALUDE_nth_equation_pattern_l1886_188636

theorem nth_equation_pattern (n : ℕ) : 1 + 6 * n = (3 * n + 1)^2 - 9 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_pattern_l1886_188636


namespace NUMINAMATH_CALUDE_length_of_CF_l1886_188668

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle ABCD with given properties -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point
  ab_length : ℝ
  bc_length : ℝ
  cd_length : ℝ
  da_length : ℝ
  is_rectangle : ab_length = cd_length ∧ bc_length = da_length

/-- Triangle DEF with B as its centroid -/
structure TriangleDEF where
  D : Point
  E : Point
  F : Point
  B : Point
  is_centroid : B.x = (2 * D.x + E.x) / 3 ∧ B.y = (2 * D.y + E.y) / 3

/-- The main theorem -/
theorem length_of_CF (rect : Rectangle) (tri : TriangleDEF) :
  rect.A = tri.D ∧
  rect.B = tri.B ∧
  rect.C.x = tri.F.x ∧
  rect.da_length = 7 ∧
  rect.ab_length = 6 ∧
  rect.cd_length = 8 →
  Real.sqrt ((rect.C.x - tri.F.x)^2 + (rect.C.y - tri.F.y)^2) = 10.66 := by
  sorry


end NUMINAMATH_CALUDE_length_of_CF_l1886_188668


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1886_188620

theorem cube_root_equation_solution :
  ∀ x : ℝ, (7 - 3 / (3 + x))^(1/3) = -2 → x = -14/5 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1886_188620


namespace NUMINAMATH_CALUDE_inequality_solution_transformation_l1886_188667

theorem inequality_solution_transformation (a c : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + c > 0 ↔ -1/3 < x ∧ x < 1/2) →
  (∀ x : ℝ, -c*x^2 + 2*x - a > 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_transformation_l1886_188667


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1886_188630

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) + sequence_a n

theorem divisibility_equivalence (k n : ℕ) :
  (2^k : ℤ) ∣ sequence_a n ↔ 2^k ∣ n := by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1886_188630


namespace NUMINAMATH_CALUDE_count_polygons_l1886_188686

/-- The number of distinct convex polygons with 3 or more sides
    that can be drawn from 12 points on a circle -/
def num_polygons : ℕ := 4017

/-- The number of points marked on the circle -/
def num_points : ℕ := 12

/-- Theorem stating that the number of distinct convex polygons
    with 3 or more sides drawn from 12 points on a circle is 4017 -/
theorem count_polygons :
  (2^num_points : ℕ) - (Nat.choose num_points 0) - (Nat.choose num_points 1) - (Nat.choose num_points 2) = num_polygons :=
by sorry

end NUMINAMATH_CALUDE_count_polygons_l1886_188686


namespace NUMINAMATH_CALUDE_central_angle_A_B_l1886_188631

noncomputable def earthRadius : ℝ := 1 -- Normalized Earth radius

/-- Represents a point on the Earth's surface using latitude and longitude -/
structure EarthPoint where
  latitude : ℝ
  longitude : ℝ

/-- Calculates the angle at the Earth's center between two points on the surface -/
noncomputable def centralAngle (p1 p2 : EarthPoint) : ℝ := sorry

/-- Point A on Earth's surface -/
def pointA : EarthPoint := { latitude := 0, longitude := 90 }

/-- Point B on Earth's surface -/
def pointB : EarthPoint := { latitude := 30, longitude := -80 }

/-- Theorem stating that the central angle between points A and B is 140 degrees -/
theorem central_angle_A_B :
  centralAngle pointA pointB = 140 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_central_angle_A_B_l1886_188631


namespace NUMINAMATH_CALUDE_exist_cubes_sum_100_power_100_l1886_188653

theorem exist_cubes_sum_100_power_100 : ∃ (a b c d : ℕ+), (a.val ^ 3 + b.val ^ 3 + c.val ^ 3 + d.val ^ 3 : ℕ) = 100 ^ 100 := by
  sorry

end NUMINAMATH_CALUDE_exist_cubes_sum_100_power_100_l1886_188653


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1886_188659

theorem quadratic_equation_solutions : 
  ∀ x : ℝ, x^2 = 6*x ↔ x = 0 ∨ x = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1886_188659


namespace NUMINAMATH_CALUDE_inequality_preservation_l1886_188633

theorem inequality_preservation (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1886_188633


namespace NUMINAMATH_CALUDE_polynomial_subtraction_l1886_188628

theorem polynomial_subtraction (x : ℝ) :
  (4*x - 3) * (x + 6) - (2*x + 1) * (x - 4) = 2*x^2 + 28*x - 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_subtraction_l1886_188628


namespace NUMINAMATH_CALUDE_dividend_calculation_l1886_188665

theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h_divisor : divisor = 127.5)
  (h_quotient : quotient = 238)
  (h_remainder : remainder = 53.2) :
  divisor * quotient + remainder = 30398.2 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1886_188665


namespace NUMINAMATH_CALUDE_internally_tangent_circles_distance_l1886_188646

theorem internally_tangent_circles_distance (r₁ r₂ d : ℝ) : 
  r₁ = 3 → r₂ = 6 → d = r₂ - r₁ → d = 3 := by sorry

end NUMINAMATH_CALUDE_internally_tangent_circles_distance_l1886_188646


namespace NUMINAMATH_CALUDE_jay_and_paul_distance_l1886_188678

/-- The distance between two people walking in opposite directions -/
def distance_apart (jay_speed : ℚ) (paul_speed : ℚ) (time : ℚ) : ℚ :=
  jay_speed * time + paul_speed * time

/-- Theorem: Jay and Paul's distance apart after 2 hours -/
theorem jay_and_paul_distance : 
  let jay_speed : ℚ := 1 / 20 -- miles per minute
  let paul_speed : ℚ := 3 / 40 -- miles per minute
  let time : ℚ := 120 -- minutes (2 hours)
  distance_apart jay_speed paul_speed time = 15
  := by sorry

end NUMINAMATH_CALUDE_jay_and_paul_distance_l1886_188678


namespace NUMINAMATH_CALUDE_circle_radius_with_inscribed_square_l1886_188616

/-- Given a circle with a chord of length 6 and an inscribed square of side length 2 in the segment
    corresponding to the chord, prove that the radius of the circle is √10. -/
theorem circle_radius_with_inscribed_square (r : ℝ) 
  (h1 : ∃ (chord : ℝ), chord = 6 ∧ chord ≤ 2 * r)
  (h2 : ∃ (square_side : ℝ), square_side = 2 ∧ 
        square_side ≤ (r + r - chord) ∧ 
        square_side * square_side ≤ chord * (2 * r - chord)) :
  r = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_inscribed_square_l1886_188616


namespace NUMINAMATH_CALUDE_systematic_sampling_l1886_188669

theorem systematic_sampling (total_students : ℕ) (sample_size : ℕ) (interval : ℕ) (start : ℕ) :
  total_students = 800 →
  sample_size = 50 →
  interval = 16 →
  start = 7 →
  ∃ (n : ℕ), n ≤ 4 ∧ 
    (start + (n - 1) * interval ≥ 49) ∧ 
    (start + (n - 1) * interval ≤ 64) ∧
    (start + (n - 1) * interval = 55) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l1886_188669


namespace NUMINAMATH_CALUDE_y_range_l1886_188642

theorem y_range (a b y : ℝ) (h1 : a + b = 2) (h2 : b ≤ 2) (h3 : y - a^2 - 2*a + 2 = 0) :
  y ≥ -2 := by
sorry

end NUMINAMATH_CALUDE_y_range_l1886_188642


namespace NUMINAMATH_CALUDE_apple_pear_box_difference_l1886_188604

theorem apple_pear_box_difference :
  ∀ (initial_apples initial_pears additional : ℕ),
    initial_apples = 25 →
    initial_pears = 12 →
    additional = 8 →
    (initial_apples + additional) - (initial_pears + additional) = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_pear_box_difference_l1886_188604


namespace NUMINAMATH_CALUDE_line_slope_l1886_188640

theorem line_slope (x y : ℝ) : 4 * y + 2 * x = 10 → (y - 2.5) / x = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_line_slope_l1886_188640


namespace NUMINAMATH_CALUDE_rabbit_pairs_rabbit_pairs_base_cases_rabbit_pairs_recurrence_l1886_188621

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem rabbit_pairs (n : ℕ) : 
  fib n = if n = 0 then 0 
          else if n = 1 then 1 
          else fib (n - 1) + fib (n - 2) := by
  sorry

theorem rabbit_pairs_base_cases :
  fib 1 = 1 ∧ fib 2 = 1 := by
  sorry

theorem rabbit_pairs_recurrence (n : ℕ) (h : n > 2) :
  fib n = fib (n - 1) + fib (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_rabbit_pairs_rabbit_pairs_base_cases_rabbit_pairs_recurrence_l1886_188621


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1886_188609

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1886_188609


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l1886_188607

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l1886_188607


namespace NUMINAMATH_CALUDE_wrong_mark_correction_l1886_188658

theorem wrong_mark_correction (n : ℕ) (initial_avg correct_avg : ℚ) (correct_mark : ℚ) (x : ℚ) 
  (h1 : n = 30)
  (h2 : initial_avg = 100)
  (h3 : correct_avg = 98)
  (h4 : correct_mark = 10) :
  (n : ℚ) * initial_avg - x + correct_mark = n * correct_avg → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_correction_l1886_188658


namespace NUMINAMATH_CALUDE_triangle_transformation_l1886_188698

theorem triangle_transformation (n : ℕ) (remaining_fraction : ℚ) :
  n = 3 ∧ 
  remaining_fraction = (8 / 9 : ℚ)^n → 
  remaining_fraction = 512 / 729 := by
sorry

end NUMINAMATH_CALUDE_triangle_transformation_l1886_188698


namespace NUMINAMATH_CALUDE_equation_solutions_l1886_188641

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 13*x - 8) + 1 / (x^2 + 3*x - 8) + 1 / (x^2 - 15*x - 8) = 0)} = {8, 1, -1, -8} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1886_188641


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l1886_188685

theorem arithmetic_simplification :
  (100 - 25 * 4 = 0) ∧
  (20 / 5 * 2 = 8) ∧
  (360 - 200 / 4 = 310) ∧
  (36 / 3 + 27 = 39) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l1886_188685


namespace NUMINAMATH_CALUDE_radius_of_combined_lead_spheres_l1886_188688

/-- The radius of a sphere formed by combining the volume of multiple smaller spheres -/
def radiusOfCombinedSphere (n : ℕ) (r : ℝ) : ℝ :=
  ((n : ℝ) * r^3)^(1/3)

/-- Theorem: The radius of a sphere formed by combining 12 spheres of radius 2 cm is ∛96 cm -/
theorem radius_of_combined_lead_spheres :
  radiusOfCombinedSphere 12 2 = (96 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_radius_of_combined_lead_spheres_l1886_188688


namespace NUMINAMATH_CALUDE_symmetric_difference_A_B_l1886_188618

-- Define the set difference operation
def set_difference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define the symmetric difference operation
def symmetric_difference (M N : Set ℝ) : Set ℝ := 
  set_difference M N ∪ set_difference N M

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 3^x}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -(x-1)^2 + 2}

-- State the theorem
theorem symmetric_difference_A_B : 
  symmetric_difference A B = {y | y ≤ 0 ∨ y > 2} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_A_B_l1886_188618


namespace NUMINAMATH_CALUDE_schoolchildren_mushroom_picking_l1886_188614

theorem schoolchildren_mushroom_picking (n : ℕ) 
  (h_max : ∃ (child : ℕ), child ≤ n ∧ child * 5 = n) 
  (h_min : ∃ (child : ℕ), child ≤ n ∧ child * 7 = n) : 
  5 < n ∧ n < 7 := by
  sorry

#check schoolchildren_mushroom_picking

end NUMINAMATH_CALUDE_schoolchildren_mushroom_picking_l1886_188614


namespace NUMINAMATH_CALUDE_checkered_board_division_l1886_188662

theorem checkered_board_division (n : ℕ) : 
  (∃ m : ℕ, n^2 = 9 + 7*m) ∧ 
  (∃ k : ℕ, n = 7*k + 3) ↔ 
  n % 7 = 3 :=
sorry

end NUMINAMATH_CALUDE_checkered_board_division_l1886_188662


namespace NUMINAMATH_CALUDE_derivative_f_l1886_188644

noncomputable def f (x : ℝ) : ℝ := (x + 1/x)^5

theorem derivative_f (x : ℝ) (hx : x ≠ 0) :
  deriv f x = 5 * (x + 1/x)^4 * (1 - 1/x^2) :=
by sorry

end NUMINAMATH_CALUDE_derivative_f_l1886_188644


namespace NUMINAMATH_CALUDE_min_omega_value_l1886_188627

theorem min_omega_value (ω : Real) (x₁ x₂ : Real) :
  ω > 0 →
  (fun x ↦ Real.sin (ω * x + π / 3) + Real.sin (ω * x)) x₁ = 0 →
  (fun x ↦ Real.sin (ω * x + π / 3) + Real.sin (ω * x)) x₂ = Real.sqrt 3 →
  |x₁ - x₂| = π →
  ∃ (ω_min : Real), ω_min = 1/2 ∧ ∀ (ω' : Real), ω' > 0 ∧
    (∃ (y₁ y₂ : Real), 
      (fun x ↦ Real.sin (ω' * x + π / 3) + Real.sin (ω' * x)) y₁ = 0 ∧
      (fun x ↦ Real.sin (ω' * x + π / 3) + Real.sin (ω' * x)) y₂ = Real.sqrt 3 ∧
      |y₁ - y₂| = π) →
    ω' ≥ ω_min :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l1886_188627


namespace NUMINAMATH_CALUDE_trisha_walk_distance_l1886_188664

theorem trisha_walk_distance (total_distance : ℝ) (tshirt_to_hotel : ℝ) (hotel_to_postcard : ℝ) :
  total_distance = 0.89 →
  tshirt_to_hotel = 0.67 →
  total_distance = hotel_to_postcard + hotel_to_postcard + tshirt_to_hotel →
  hotel_to_postcard = 0.11 := by
sorry

end NUMINAMATH_CALUDE_trisha_walk_distance_l1886_188664


namespace NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l1886_188649

theorem complex_fraction_equals_negative_two :
  let z : ℂ := 1 + I
  (z^2) / (1 - z) = -2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l1886_188649


namespace NUMINAMATH_CALUDE_total_students_count_l1886_188679

/-- The number of students who wish to go on a scavenger hunting trip -/
def scavenger_hunting_students : ℕ := 4000

/-- The number of students who wish to go on a skiing trip -/
def skiing_students : ℕ := 2 * scavenger_hunting_students

/-- The total number of students -/
def total_students : ℕ := scavenger_hunting_students + skiing_students

theorem total_students_count : total_students = 12000 := by
  sorry

end NUMINAMATH_CALUDE_total_students_count_l1886_188679


namespace NUMINAMATH_CALUDE_tiffany_miles_per_day_l1886_188661

theorem tiffany_miles_per_day (T : ℚ) : 
  (7 : ℚ) = 3 * T + 3 * (1/3 : ℚ) + 0 → T = 2 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_miles_per_day_l1886_188661


namespace NUMINAMATH_CALUDE_equation_solution_l1886_188683

theorem equation_solution :
  ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 57 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1886_188683


namespace NUMINAMATH_CALUDE_system_real_solutions_l1886_188691

theorem system_real_solutions (k : ℝ) : 
  (∃ x y : ℝ, x - k * y = 0 ∧ x^2 + y = -1) ↔ -1/2 ≤ k ∧ k ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_real_solutions_l1886_188691


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l1886_188666

theorem parallel_angles_theorem (angle1 angle2 : ℝ) : 
  (angle1 + angle2 = 180 ∨ angle1 = angle2) →  -- parallel sides condition
  angle2 = 3 * angle1 - 20 →                   -- angle relationship
  (angle1 = 50 ∧ angle2 = 130) :=              -- conclusion
by sorry

end NUMINAMATH_CALUDE_parallel_angles_theorem_l1886_188666


namespace NUMINAMATH_CALUDE_sqrt_twelve_equals_two_sqrt_three_l1886_188681

theorem sqrt_twelve_equals_two_sqrt_three : Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_equals_two_sqrt_three_l1886_188681


namespace NUMINAMATH_CALUDE_greatest_common_length_l1886_188671

theorem greatest_common_length (a b c d : ℕ) 
  (ha : a = 72) (hb : b = 48) (hc : c = 120) (hd : d = 96) : 
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_length_l1886_188671


namespace NUMINAMATH_CALUDE_water_remaining_in_bucket_l1886_188619

theorem water_remaining_in_bucket (initial_water : ℚ) (poured_out : ℚ) : 
  initial_water = 3/4 → poured_out = 1/3 → initial_water - poured_out = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_in_bucket_l1886_188619


namespace NUMINAMATH_CALUDE_line_equation_from_parabola_intersections_l1886_188626

/-- Given a parabola y^2 = 2x and a point G, prove that the line AB formed by
    the intersection of two lines from G to the parabola has a specific equation. -/
theorem line_equation_from_parabola_intersections
  (G : ℝ × ℝ)
  (k₁ k₂ : ℝ)
  (h_G : G = (2, 2))
  (h_parabola : ∀ x y, y^2 = 2*x → (∃ A B : ℝ × ℝ, 
    (A.1 = x ∧ A.2 = y) ∨ (B.1 = x ∧ B.2 = y)))
  (h_slopes : ∀ A B : ℝ × ℝ, 
    (A.2^2 = 2*A.1 ∧ B.2^2 = 2*B.1) → 
    k₁ = (A.2 - G.2) / (A.1 - G.1) ∧
    k₂ = (B.2 - G.2) / (B.1 - G.1))
  (h_sum : k₁ + k₂ = 5)
  (h_product : k₁ * k₂ = -2) :
  ∃ A B : ℝ × ℝ, 2 * A.1 + 9 * A.2 + 12 = 0 ∧
                 2 * B.1 + 9 * B.2 + 12 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_parabola_intersections_l1886_188626


namespace NUMINAMATH_CALUDE_money_distribution_l1886_188673

theorem money_distribution (x : ℚ) : 
  x > 0 →
  let moe_initial := 6 * x
  let loki_initial := 5 * x
  let nick_initial := 4 * x
  let ott_received := 3 * x
  let total_money := moe_initial + loki_initial + nick_initial
  ott_received / total_money = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1886_188673


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1886_188606

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x > 0, Real.exp x - a * x < 1

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(a - 1)^x) > (-(a - 1)^y)

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ (∃ a : ℝ, p a ∧ ¬(q a)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1886_188606


namespace NUMINAMATH_CALUDE_sphere_volume_l1886_188635

theorem sphere_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 36 * Real.pi) :
  (4 / 3) * Real.pi * r^3 = 36 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_volume_l1886_188635


namespace NUMINAMATH_CALUDE_cookie_baking_problem_l1886_188625

theorem cookie_baking_problem (x : ℚ) : 
  x > 0 → 
  x + x/2 + (3*x/2 - 4) = 92 → 
  x = 32 := by
sorry

end NUMINAMATH_CALUDE_cookie_baking_problem_l1886_188625


namespace NUMINAMATH_CALUDE_f_composition_equals_sqrt2_over_2_l1886_188632

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 - 2^x else Real.sqrt x

theorem f_composition_equals_sqrt2_over_2 : f (f (-1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_sqrt2_over_2_l1886_188632


namespace NUMINAMATH_CALUDE_f_properties_l1886_188674

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x + 1) + 2 * a * x - 4 * a * Real.exp x + 4

theorem f_properties (a : ℝ) (h : a > 0) :
  (∃ x, f 1 x ≤ f 1 0) ∧
  ((0 < a ∧ a < 1 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
   (a = 1 → ∃! x, f a x = 0) ∧
   (a > 1 → ∀ x, f a x ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1886_188674


namespace NUMINAMATH_CALUDE_percentage_of_800_l1886_188638

theorem percentage_of_800 : (25 / 100) * 800 = 200 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_800_l1886_188638


namespace NUMINAMATH_CALUDE_length_AG_is_3_sqrt_10_over_2_l1886_188689

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define properties of the triangle
def isRightAngled (t : Triangle) : Prop :=
  -- Right angle at A
  sorry

def hasGivenSides (t : Triangle) : Prop :=
  -- AB = 3 and AC = 3√5
  sorry

-- Define altitude AD
def altitudeAD (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define median BE
def medianBE (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define intersection point G
def intersectionG (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define length of AG
def lengthAG (t : Triangle) : ℝ :=
  sorry

-- Theorem statement
theorem length_AG_is_3_sqrt_10_over_2 (t : Triangle) :
  isRightAngled t → hasGivenSides t →
  lengthAG t = (3 * Real.sqrt 10) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_length_AG_is_3_sqrt_10_over_2_l1886_188689


namespace NUMINAMATH_CALUDE_polynomial_equality_l1886_188629

theorem polynomial_equality (x : ℝ) : 
  (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1 = (2*x)^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1886_188629


namespace NUMINAMATH_CALUDE_honey_distribution_l1886_188613

theorem honey_distribution (bottles : ℕ) (weight_per_bottle : ℚ) (share_per_neighbor : ℚ) :
  bottles = 4 →
  weight_per_bottle = 3 →
  share_per_neighbor = 3/4 →
  (bottles * weight_per_bottle) / share_per_neighbor = 16 := by
sorry

end NUMINAMATH_CALUDE_honey_distribution_l1886_188613


namespace NUMINAMATH_CALUDE_inequality_proof_l1886_188680

theorem inequality_proof (a b c d : ℝ) (h : a > b ∧ b > c ∧ c > d) :
  c < (c * d - a * b) / (c - a + d - b) ∧ (c * d - a * b) / (c - a + d - b) < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1886_188680


namespace NUMINAMATH_CALUDE_cubic_minimum_condition_l1886_188687

/-- A cubic function with parameters p, q, and r -/
def cubic_function (p q r x : ℝ) : ℝ := x^3 + 3*p*x^2 + 3*q*x + r

/-- The derivative of the cubic function with respect to x -/
def cubic_derivative (p q x : ℝ) : ℝ := 3*x^2 + 6*p*x + 3*q

theorem cubic_minimum_condition (p q r : ℝ) :
  (∀ x : ℝ, cubic_function p q r x ≥ cubic_function p q r (-p)) ∧
  cubic_function p q r (-p) = -27 →
  r = -27 - 2*p^3 + 3*p*q :=
by sorry

end NUMINAMATH_CALUDE_cubic_minimum_condition_l1886_188687


namespace NUMINAMATH_CALUDE_parabola_c_value_l1886_188684

/-- A parabola with equation y = 2x^2 + bx + c passes through the points (1,5) and (3,17).
    This theorem proves that the value of c is 5. -/
theorem parabola_c_value :
  ∀ b c : ℝ,
  (5 : ℝ) = 2 * (1 : ℝ)^2 + b * (1 : ℝ) + c →
  (17 : ℝ) = 2 * (3 : ℝ)^2 + b * (3 : ℝ) + c →
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1886_188684


namespace NUMINAMATH_CALUDE_simons_score_l1886_188650

theorem simons_score (n : ℕ) (avg_before avg_after simons_score : ℚ) : 
  n = 21 →
  avg_before = 86 →
  avg_after = 88 →
  n * avg_after = (n - 1) * avg_before + simons_score →
  simons_score = 128 :=
by sorry

end NUMINAMATH_CALUDE_simons_score_l1886_188650


namespace NUMINAMATH_CALUDE_johns_former_apartment_cost_l1886_188643

/-- Proves that the cost per square foot of John's former apartment was $2 -/
theorem johns_former_apartment_cost (former_size : ℝ) (new_rent : ℝ) (savings : ℝ) : 
  former_size = 750 →
  new_rent = 2800 →
  savings = 1200 →
  ∃ (cost_per_sqft : ℝ), 
    cost_per_sqft = 2 ∧ 
    former_size * cost_per_sqft * 12 = (new_rent / 2) * 12 + savings :=
by sorry

end NUMINAMATH_CALUDE_johns_former_apartment_cost_l1886_188643


namespace NUMINAMATH_CALUDE_p_and_q_true_l1886_188697

theorem p_and_q_true (h1 : p ∨ q) (h2 : p ∧ q) : p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_l1886_188697


namespace NUMINAMATH_CALUDE_number_of_bags_l1886_188657

theorem number_of_bags (students : ℕ) (nuts_per_student : ℕ) (nuts_per_bag : ℕ) : 
  students = 13 → nuts_per_student = 75 → nuts_per_bag = 15 →
  (students * nuts_per_student) / nuts_per_bag = 65 := by
  sorry

end NUMINAMATH_CALUDE_number_of_bags_l1886_188657


namespace NUMINAMATH_CALUDE_contradictory_implies_mutually_exclusive_but_not_conversely_l1886_188656

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (A B : Set Ω) : Prop := A ∩ B = ∅

/-- Two events are contradictory if one event is the complement of the other -/
def contradictory (A B : Set Ω) : Prop := A = Bᶜ

/-- Theorem: Contradictory events are mutually exclusive, but mutually exclusive events are not necessarily contradictory -/
theorem contradictory_implies_mutually_exclusive_but_not_conversely :
  (∀ A B : Set Ω, contradictory A B → mutually_exclusive A B) ∧
  ¬(∀ A B : Set Ω, mutually_exclusive A B → contradictory A B) := by
  sorry

end NUMINAMATH_CALUDE_contradictory_implies_mutually_exclusive_but_not_conversely_l1886_188656


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1886_188670

/-- The circle with center on y = -4x and tangent to x + y - 1 = 0 at (3, -2) -/
def special_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 4)^2 = 8

/-- The line y = -4x -/
def center_line (x y : ℝ) : Prop :=
  y = -4 * x

/-- The line x + y - 1 = 0 -/
def tangent_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- The point P(3, -2) -/
def point_P : ℝ × ℝ :=
  (3, -2)

theorem circle_equation_proof :
  ∃ (c : ℝ × ℝ), 
    (center_line c.1 c.2) ∧ 
    (∀ (x y : ℝ), tangent_line x y → 
      ((x - c.1)^2 + (y - c.2)^2 = (c.1 - point_P.1)^2 + (c.2 - point_P.2)^2)) ↔
    (∀ (x y : ℝ), special_circle x y ↔ 
      ((x - 1)^2 + (y + 4)^2 = (1 - point_P.1)^2 + (-4 - point_P.2)^2)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1886_188670


namespace NUMINAMATH_CALUDE_faster_train_speed_l1886_188651

theorem faster_train_speed
  (train_length : ℝ)
  (speed_difference : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 37.5)
  (h2 : speed_difference = 36)
  (h3 : passing_time = 27)
  : ∃ (faster_speed : ℝ),
    faster_speed = 46 ∧
    (faster_speed - speed_difference) * 1000 / 3600 * passing_time = 2 * train_length :=
by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l1886_188651


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1886_188634

/-- Given that P(1, -3) is the midpoint of line segment CD and C is located at (7, 5),
    prove that the sum of the coordinates of point D is -16. -/
theorem midpoint_coordinate_sum (C D : ℝ × ℝ) : 
  C = (7, 5) →
  (1, -3) = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1886_188634


namespace NUMINAMATH_CALUDE_sheet_reduction_percentage_l1886_188663

def original_sheets : ℕ := 20
def original_lines_per_sheet : ℕ := 55
def original_chars_per_line : ℕ := 65

def new_lines_per_sheet : ℕ := 65
def new_chars_per_line : ℕ := 70

def total_chars : ℕ := original_sheets * original_lines_per_sheet * original_chars_per_line
def new_chars_per_sheet : ℕ := new_lines_per_sheet * new_chars_per_line
def new_sheets : ℕ := (total_chars + new_chars_per_sheet - 1) / new_chars_per_sheet

theorem sheet_reduction_percentage : 
  (original_sheets - new_sheets : ℚ) / original_sheets * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_sheet_reduction_percentage_l1886_188663


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_determination_l1886_188660

/-- Given an arithmetic sequence b₁, b₂, b₃, ..., we define:
    S'ₙ = b₁ + b₂ + b₃ + ... + bₙ
    T'ₙ = S'₁ + S'₂ + S'₃ + ... + S'ₙ
    This theorem states that if we know the value of S'₃₀₂₈, 
    then 4543 is the smallest positive integer n for which 
    T'ₙ can be uniquely determined. -/
theorem arithmetic_sequence_unique_determination (b₁ : ℚ) (d : ℚ) (S'₃₀₂₈ : ℚ) :
  let b : ℕ → ℚ := λ n => b₁ + (n - 1) * d
  let S' : ℕ → ℚ := λ n => (n : ℚ) * (2 * b₁ + (n - 1) * d) / 2
  let T' : ℕ → ℚ := λ n => (n * (n + 1) * (3 * b₁ + (n - 1) * d)) / 6
  ∃! (T'₄₅₄₃ : ℚ), S'₃₀₂₈ = S' 3028 ∧ T'₄₅₄₃ = T' 4543 ∧
    ∀ m : ℕ, m < 4543 → ¬∃! (T'ₘ : ℚ), S'₃₀₂₈ = S' 3028 ∧ T'ₘ = T' m :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_determination_l1886_188660


namespace NUMINAMATH_CALUDE_no_lattice_polygon1994_l1886_188677

/-- A polygon with 1994 sides where side lengths are √(i^2 + 4) -/
def Polygon1994 : Type :=
  { vertices : Fin 1995 → ℤ × ℤ // 
    ∀ i : Fin 1994, 
      let (x₁, y₁) := vertices i
      let (x₂, y₂) := vertices (i + 1)
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = i^2 + 4 ∧
    vertices 0 = vertices 1994 }

/-- Theorem stating that such a polygon cannot exist with all vertices on lattice points -/
theorem no_lattice_polygon1994 : ¬ ∃ (p : Polygon1994), True := by
  sorry

end NUMINAMATH_CALUDE_no_lattice_polygon1994_l1886_188677


namespace NUMINAMATH_CALUDE_percentage_in_quarters_l1886_188693

def dimes : ℕ := 80
def quarters : ℕ := 30
def nickels : ℕ := 40

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def total_value : ℕ := dimes * dime_value + quarters * quarter_value + nickels * nickel_value
def quarters_value : ℕ := quarters * quarter_value

theorem percentage_in_quarters : 
  (quarters_value : ℚ) / total_value * 100 = 3/7 * 100 := by sorry

end NUMINAMATH_CALUDE_percentage_in_quarters_l1886_188693


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1886_188690

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcome of two shots -/
def TwoShotOutcome := ShotOutcome × ShotOutcome

/-- The event of hitting the target at least once in two shots -/
def hitAtLeastOnce (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Hit ∨ outcome.2 = ShotOutcome.Hit

/-- The event of missing the target both times in two shots -/
def missBothTimes (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

theorem mutually_exclusive_events :
  ∀ (outcome : TwoShotOutcome), ¬(hitAtLeastOnce outcome ∧ missBothTimes outcome) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l1886_188690


namespace NUMINAMATH_CALUDE_smallest_square_area_for_two_rectangles_l1886_188601

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum square side length needed to fit two rectangles -/
def minSquareSideLength (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r2.width) (r1.height + r2.height)

/-- Theorem: The smallest square area to fit 2×4 and 3×5 rectangles is 25 -/
theorem smallest_square_area_for_two_rectangles :
  let r1 : Rectangle := ⟨2, 4⟩
  let r2 : Rectangle := ⟨3, 5⟩
  (minSquareSideLength r1 r2)^2 = 25 := by
  sorry

#eval (minSquareSideLength ⟨2, 4⟩ ⟨3, 5⟩)^2

end NUMINAMATH_CALUDE_smallest_square_area_for_two_rectangles_l1886_188601


namespace NUMINAMATH_CALUDE_find_A_l1886_188612

theorem find_A (A B C : ℚ) 
  (h1 : A = (1 / 2) * B) 
  (h2 : B = (3 / 4) * C) 
  (h3 : A + C = 55) : 
  A = 15 := by
sorry

end NUMINAMATH_CALUDE_find_A_l1886_188612


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1886_188675

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + a * c) = 54)
  (edge_length : 4 * (a + b + c) = 40) :
  ∃ d : ℝ, d^2 = a^2 + b^2 + c^2 ∧ d = Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1886_188675


namespace NUMINAMATH_CALUDE_system_is_linear_l1886_188603

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- A system of two equations is linear if both equations are linear and they involve exactly two variables. -/
def is_linear_system (f g : ℝ → ℝ → ℝ) : Prop :=
  is_linear_equation f ∧ is_linear_equation g

/-- The given system of equations -/
def equation1 (x y : ℝ) : ℝ := x - y - 11
def equation2 (x y : ℝ) : ℝ := 4 * x - y - 1

theorem system_is_linear : is_linear_system equation1 equation2 := by
  sorry

end NUMINAMATH_CALUDE_system_is_linear_l1886_188603


namespace NUMINAMATH_CALUDE_sqrt_88200_simplification_l1886_188647

theorem sqrt_88200_simplification : Real.sqrt 88200 = 70 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_88200_simplification_l1886_188647


namespace NUMINAMATH_CALUDE_marked_price_calculation_l1886_188676

theorem marked_price_calculation (purchase_price : ℝ) (discount_percentage : ℝ) : 
  purchase_price = 50 ∧ discount_percentage = 60 → 
  (purchase_price / ((100 - discount_percentage) / 100)) / 2 = 62.50 := by
sorry

end NUMINAMATH_CALUDE_marked_price_calculation_l1886_188676


namespace NUMINAMATH_CALUDE_median_in_third_interval_l1886_188617

/-- Represents the distribution of students across score intervals --/
structure ScoreDistribution where
  total_students : ℕ
  intervals : List ℕ
  h_total : total_students = intervals.sum

/-- The index of the interval containing the median --/
def median_interval_index (sd : ScoreDistribution) : ℕ :=
  sd.intervals.foldl
    (λ acc count =>
      if acc.1 < sd.total_students / 2 then (acc.1 + count, acc.2 + 1)
      else acc)
    (0, 0)
  |>.2

theorem median_in_third_interval (sd : ScoreDistribution) :
  sd.total_students = 100 ∧
  sd.intervals = [20, 18, 15, 22, 14, 11] →
  median_interval_index sd = 3 := by
  sorry

#eval median_interval_index ⟨100, [20, 18, 15, 22, 14, 11], rfl⟩

end NUMINAMATH_CALUDE_median_in_third_interval_l1886_188617


namespace NUMINAMATH_CALUDE_chimney_bricks_total_bricks_correct_l1886_188682

-- Define the problem parameters
def brenda_time : ℝ := 8
def brandon_time : ℝ := 12
def combined_decrease : ℝ := 12
def combined_time : ℝ := 6

-- Define the theorem
theorem chimney_bricks : ∃ (h : ℝ),
  h > 0 ∧
  h / brenda_time + h / brandon_time - combined_decrease = h / combined_time :=
by
  -- The proof goes here
  sorry

-- Define the final answer
def total_bricks : ℕ := 288

-- Prove that the total_bricks satisfies the theorem
theorem total_bricks_correct : 
  ∃ (h : ℝ), h = total_bricks ∧
  h > 0 ∧
  h / brenda_time + h / brandon_time - combined_decrease = h / combined_time :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_chimney_bricks_total_bricks_correct_l1886_188682


namespace NUMINAMATH_CALUDE_min_value_theorem_l1886_188600

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : Real.log x + Real.log y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log a + Real.log b = 1 → 2/a + 5/b ≥ 2/x + 5/y) ∧ 2/x + 5/y = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1886_188600


namespace NUMINAMATH_CALUDE_sum_of_nine_and_number_l1886_188622

theorem sum_of_nine_and_number (x : ℝ) : 
  (9 - x = 1) → (x < 10) → (9 + x = 17) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_nine_and_number_l1886_188622


namespace NUMINAMATH_CALUDE_range_of_m_l1886_188624

/-- A quadratic function f(x) = ax^2 - 2ax + c -/
def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + c

/-- The statement that f is monotonically decreasing on [0,1] -/
def is_monotone_decreasing (a c : ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a c x > f a c y

/-- The main theorem -/
theorem range_of_m (a c : ℝ) :
  is_monotone_decreasing a c →
  (∃ m, f a c m ≤ f a c 0) →
  ∃ m, 0 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1886_188624


namespace NUMINAMATH_CALUDE_binomial_equation_solutions_l1886_188608

theorem binomial_equation_solutions :
  ∀ m r : ℕ, 2014 ≥ m → m ≥ r → r ≥ 1 →
  (Nat.choose 2014 m + Nat.choose m r = Nat.choose 2014 r + Nat.choose (2014 - r) (m - r)) ↔
  ((m = r ∧ m ≤ 2014) ∨
   (m = 2014 - r ∧ r ≤ 1006) ∨
   (m = 2014 ∧ r ≤ 2013)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_equation_solutions_l1886_188608


namespace NUMINAMATH_CALUDE_power_five_plus_five_mod_eight_l1886_188695

theorem power_five_plus_five_mod_eight : (5^123 + 5) % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_five_plus_five_mod_eight_l1886_188695


namespace NUMINAMATH_CALUDE_min_value_of_D_l1886_188655

noncomputable def D (x a : ℝ) : ℝ :=
  Real.sqrt ((x - a^2) + (Real.log x - a^2 / 4)^2) + a^2 / 4 + 1

theorem min_value_of_D :
  ∃ (m : ℝ), ∀ (x a : ℝ), D x a ≥ m ∧ ∃ (x₀ a₀ : ℝ), D x₀ a₀ = m ∧ m = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_D_l1886_188655


namespace NUMINAMATH_CALUDE_triangle_angle_bisector_theorem_l1886_188639

/-- Given a triangle ABC with AB = 16 and AC = 5, where the angle bisectors of ∠ABC and ∠BCA 
    meet at point P inside the triangle such that AP = 4, prove that BC = 14. -/
theorem triangle_angle_bisector_theorem (A B C P : ℝ × ℝ) : 
  let d (X Y : ℝ × ℝ) := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  -- AB = 16
  d A B = 16 →
  -- AC = 5
  d A C = 5 →
  -- P is on the angle bisector of ∠ABC
  (d A P / d B P = d A C / d B C) →
  -- P is on the angle bisector of ∠BCA
  (d C P / d A P = d C B / d A B) →
  -- P is inside the triangle
  (0 < d A P ∧ d A P < d A B ∧ d A P < d A C) →
  -- AP = 4
  d A P = 4 →
  -- BC = 14
  d B C = 14 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_bisector_theorem_l1886_188639


namespace NUMINAMATH_CALUDE_distance_from_blast_site_l1886_188610

/-- Proves the distance a man is from a blast site when he hears a second blast -/
theorem distance_from_blast_site (speed_of_sound : ℝ) (time_between_blasts : ℝ) (time_heard_second_blast : ℝ) : 
  speed_of_sound = 330 →
  time_between_blasts = 30 →
  time_heard_second_blast = 30 + 12 / 60 →
  speed_of_sound * (time_heard_second_blast - time_between_blasts) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_blast_site_l1886_188610


namespace NUMINAMATH_CALUDE_investment_percentage_l1886_188648

/-- Given an investment scenario, prove that the unknown percentage is 7% -/
theorem investment_percentage (total_investment : ℝ) (known_rate : ℝ) (total_interest : ℝ) (amount_at_unknown_rate : ℝ) (unknown_rate : ℝ) :
  total_investment = 12000 ∧
  known_rate = 0.09 ∧
  total_interest = 970 ∧
  amount_at_unknown_rate = 5500 ∧
  amount_at_unknown_rate * unknown_rate + (total_investment - amount_at_unknown_rate) * known_rate = total_interest →
  unknown_rate = 0.07 := by
  sorry

end NUMINAMATH_CALUDE_investment_percentage_l1886_188648


namespace NUMINAMATH_CALUDE_inequality_solution_set_function_domain_set_l1886_188602

-- Part 1: Inequality solution
def inequality_solution (x : ℝ) : Prop :=
  x * (x + 2) > x * (3 - x) + 6

theorem inequality_solution_set :
  ∀ x : ℝ, inequality_solution x ↔ (x < -3/2 ∨ x > 2) :=
sorry

-- Part 2: Function domain
def function_domain (x : ℝ) : Prop :=
  x + 1 ≥ 0 ∧ x ≠ 1 ∧ -x^2 - x + 6 > 0

theorem function_domain_set :
  ∀ x : ℝ, function_domain x ↔ (-1 ≤ x ∧ x < 2 ∧ x ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_function_domain_set_l1886_188602


namespace NUMINAMATH_CALUDE_students_not_in_biology_l1886_188696

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880) 
  (h2 : biology_percentage = 275 / 1000) : 
  total_students - (total_students * biology_percentage).floor = 638 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l1886_188696


namespace NUMINAMATH_CALUDE_smallest_set_size_existence_of_set_smallest_set_size_is_eight_l1886_188611

theorem smallest_set_size (n : ℕ) (h : n ≥ 5) :
  (∃ (S : Finset (ℕ × ℕ)),
    S.card = n ∧
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
    (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
      ∃ (r : ℕ × ℕ), r ∈ S ∧
        4 ∣ (p.1 + q.1 - r.1) ∧
        4 ∣ (p.2 + q.2 - r.2))) →
  n ≥ 8 := by sorry

theorem existence_of_set :
  ∃ (S : Finset (ℕ × ℕ)),
    S.card = 8 ∧
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
    (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
      ∃ (r : ℕ × ℕ), r ∈ S ∧
        4 ∣ (p.1 + q.1 - r.1) ∧
        4 ∣ (p.2 + q.2 - r.2)) := by sorry

theorem smallest_set_size_is_eight :
  (∃ (n : ℕ), n ≥ 5 ∧
    (∃ (S : Finset (ℕ × ℕ)),
      S.card = n ∧
      (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
      (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
        ∃ (r : ℕ × ℕ), r ∈ S ∧
          4 ∣ (p.1 + q.1 - r.1) ∧
          4 ∣ (p.2 + q.2 - r.2)))) →
  (∀ (m : ℕ), m ≥ 5 →
    (∃ (S : Finset (ℕ × ℕ)),
      S.card = m ∧
      (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
      (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
        ∃ (r : ℕ × ℕ), r ∈ S ∧
          4 ∣ (p.1 + q.1 - r.1) ∧
          4 ∣ (p.2 + q.2 - r.2))) →
    m ≥ 8) ∧
  (∃ (S : Finset (ℕ × ℕ)),
    S.card = 8 ∧
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 ∈ Finset.range 4 ∧ p.2 ∈ Finset.range 4) ∧
    (∀ (p q : ℕ × ℕ), p ∈ S → q ∈ S →
      ∃ (r : ℕ × ℕ), r ∈ S ∧
        4 ∣ (p.1 + q.1 - r.1) ∧
        4 ∣ (p.2 + q.2 - r.2))) := by sorry

end NUMINAMATH_CALUDE_smallest_set_size_existence_of_set_smallest_set_size_is_eight_l1886_188611


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1886_188654

/-- Given a rhombus with side length 60 units and shorter diagonal 56 units,
    the longer diagonal has a length of 32√11 units. -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) 
    (h1 : side = 60) 
    (h2 : shorter_diag = 56) 
    (h3 : side^2 = (shorter_diag/2)^2 + (longer_diag/2)^2) : 
  longer_diag = 32 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1886_188654


namespace NUMINAMATH_CALUDE_chessboard_separating_edges_l1886_188692

/-- Represents a square on the chessboard -/
inductive Square
| White
| Black

/-- Represents the chessboard -/
def Chessboard (n : ℕ) := Fin n → Fin n → Square

/-- Counts the number of white squares on the border of the chessboard -/
def countWhiteBorderSquares (n : ℕ) (board : Chessboard n) : ℕ := sorry

/-- Counts the number of black squares on the border of the chessboard -/
def countBlackBorderSquares (n : ℕ) (board : Chessboard n) : ℕ := sorry

/-- Counts the number of edges inside the board that separate squares of different colors -/
def countSeparatingEdges (n : ℕ) (board : Chessboard n) : ℕ := sorry

/-- Main theorem: If a chessboard has at least n white and n black squares on its border,
    then there are at least n edges inside the board separating different colors -/
theorem chessboard_separating_edges (n : ℕ) (board : Chessboard n) :
  countWhiteBorderSquares n board ≥ n →
  countBlackBorderSquares n board ≥ n →
  countSeparatingEdges n board ≥ n := by sorry

end NUMINAMATH_CALUDE_chessboard_separating_edges_l1886_188692


namespace NUMINAMATH_CALUDE_no_five_two_digit_coprime_composites_l1886_188637

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem no_five_two_digit_coprime_composites :
  ¬ ∃ a b c d e : ℕ,
    is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ is_two_digit d ∧ is_two_digit e ∧
    is_composite a ∧ is_composite b ∧ is_composite c ∧ is_composite d ∧ is_composite e ∧
    are_coprime a b ∧ are_coprime a c ∧ are_coprime a d ∧ are_coprime a e ∧
    are_coprime b c ∧ are_coprime b d ∧ are_coprime b e ∧
    are_coprime c d ∧ are_coprime c e ∧
    are_coprime d e :=
by
  sorry

end NUMINAMATH_CALUDE_no_five_two_digit_coprime_composites_l1886_188637


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1886_188605

theorem regular_polygon_sides (n : ℕ) (central_angle : ℝ) : 
  n > 0 ∧ central_angle = 72 → (360 : ℝ) / n = central_angle → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1886_188605
