import Mathlib

namespace NUMINAMATH_CALUDE_experimental_fields_yield_l1823_182386

theorem experimental_fields_yield (x : ℝ) : 
  x > 0 →
  (900 : ℝ) / x = (1500 : ℝ) / (x + 300) ↔
  (∃ (area : ℝ), 
    area > 0 ∧
    area * x = 900 ∧
    area * (x + 300) = 1500) :=
by sorry

end NUMINAMATH_CALUDE_experimental_fields_yield_l1823_182386


namespace NUMINAMATH_CALUDE_fibonacci_like_invariant_l1823_182358

def fibonacci_like_sequence (u : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, u (n + 2) = u n + u (n + 1)

theorem fibonacci_like_invariant (u : ℕ → ℤ) (h : fibonacci_like_sequence u) :
  ∃ c : ℕ, ∀ n : ℕ, n ≥ 1 → |u (n - 1) * u (n + 2) - u n * u (n + 1)| = c :=
sorry

end NUMINAMATH_CALUDE_fibonacci_like_invariant_l1823_182358


namespace NUMINAMATH_CALUDE_girls_in_algebra_class_l1823_182308

theorem girls_in_algebra_class (total : ℕ) (girls boys : ℕ) : 
  total = 84 →
  girls + boys = total →
  4 * boys = 3 * girls →
  girls = 48 := by
sorry

end NUMINAMATH_CALUDE_girls_in_algebra_class_l1823_182308


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l1823_182314

/-- The trajectory of point P(x,y) moving such that its distance from the line x=-4 
    is twice its distance from the fixed point F(-1,0) -/
def trajectory (x y : ℝ) : Prop :=
  let F := ((-1 : ℝ), (0 : ℝ))
  let d := |x + 4|
  let PF := Real.sqrt ((x + 1)^2 + y^2)
  d = 2 * PF ∧ x^2 / 4 + y^2 / 3 = 1

/-- The theorem stating that the trajectory satisfies the ellipse equation -/
theorem trajectory_is_ellipse (x y : ℝ) : 
  trajectory x y ↔ x^2 / 4 + y^2 / 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l1823_182314


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1823_182370

theorem simplify_fraction_product : (210 : ℚ) / 18 * 6 / 150 * 9 / 4 = 21 / 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1823_182370


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l1823_182388

/-- Calculates the distance traveled downstream by a boat given its speed in still water,
    the stream speed, and the time taken. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat with a speed of 24 km/hr in still water, traveling downstream
    in a stream with a speed of 4 km/hr for 6 hours, covers a distance of 168 km. -/
theorem boat_downstream_distance :
  distance_downstream 24 4 6 = 168 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l1823_182388


namespace NUMINAMATH_CALUDE_convex_polyhedron_structure_l1823_182323

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  convex : Bool

/-- Represents a face of a polyhedron -/
structure Face where
  sides : Nat

/-- Represents a vertex of a polyhedron -/
structure Vertex where
  edges : Nat

/-- Definition of a convex polyhedron with its faces and vertices -/
def ConvexPolyhedronWithFacesAndVertices (p : ConvexPolyhedron) (faces : List Face) (vertices : List Vertex) : Prop :=
  p.convex ∧ faces.length > 0 ∧ vertices.length > 0

/-- Theorem stating that not all faces can have more than 3 sides 
    and not all vertices can have more than 3 edges simultaneously -/
theorem convex_polyhedron_structure 
  (p : ConvexPolyhedron) 
  (faces : List Face) 
  (vertices : List Vertex) 
  (h : ConvexPolyhedronWithFacesAndVertices p faces vertices) :
  ¬(∀ f ∈ faces, f.sides > 3 ∧ ∀ v ∈ vertices, v.edges > 3) :=
by sorry

end NUMINAMATH_CALUDE_convex_polyhedron_structure_l1823_182323


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1823_182311

theorem complex_equation_solution (i : ℂ) (z : ℂ) 
  (h1 : i * i = -1) 
  (h2 : i * z = (1 - 2*i)^2) : 
  z = -4 + 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1823_182311


namespace NUMINAMATH_CALUDE_alcohol_dilution_l1823_182357

/-- Proves that adding 30 ml of pure water to 50 ml of 30% alcohol solution results in 18.75% alcohol concentration -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.30 →
  water_added = 30 →
  final_concentration = 0.1875 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check alcohol_dilution

end NUMINAMATH_CALUDE_alcohol_dilution_l1823_182357


namespace NUMINAMATH_CALUDE_first_student_completion_time_l1823_182380

/-- Given a race with 4 students, prove that if the average completion time of the last 3 students
    is 35 seconds, and the average completion time of all 4 students is 30 seconds,
    then the completion time of the first student is 15 seconds. -/
theorem first_student_completion_time
  (n : ℕ)
  (avg_last_three : ℝ)
  (avg_all : ℝ)
  (h1 : n = 4)
  (h2 : avg_last_three = 35)
  (h3 : avg_all = 30)
  : (n : ℝ) * avg_all - (n - 1 : ℝ) * avg_last_three = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_first_student_completion_time_l1823_182380


namespace NUMINAMATH_CALUDE_joannas_reading_time_l1823_182344

/-- Joanna's reading problem -/
theorem joannas_reading_time (
  total_pages : ℕ)
  (pages_per_hour : ℕ)
  (monday_hours : ℕ)
  (remaining_hours : ℕ)
  (h1 : total_pages = 248)
  (h2 : pages_per_hour = 16)
  (h3 : monday_hours = 3)
  (h4 : remaining_hours = 6)
  : (total_pages - (monday_hours * pages_per_hour + remaining_hours * pages_per_hour)) / pages_per_hour = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_joannas_reading_time_l1823_182344


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l1823_182306

/-- Two vectors are parallel if and only if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, y)
  parallel a b → y = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l1823_182306


namespace NUMINAMATH_CALUDE_certain_number_exists_and_unique_l1823_182366

theorem certain_number_exists_and_unique : 
  ∃! x : ℝ, 22030 = (x + 445) * (2 * (x - 445)) + 30 := by
sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_unique_l1823_182366


namespace NUMINAMATH_CALUDE_min_value_theorem_l1823_182365

theorem min_value_theorem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 * x₀ + y₀ = 5 * x₀ * y₀ ∧ 4 * x₀ + 3 * y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1823_182365


namespace NUMINAMATH_CALUDE_multiple_of_sum_and_smaller_l1823_182383

theorem multiple_of_sum_and_smaller (s l : ℕ) : 
  s + l = 84 →  -- sum of two numbers is 84
  l = s * (l / s) →  -- one number is a multiple of the other
  s = 21 →  -- the smaller number is 21
  l / s = 3 :=  -- the multiple (ratio) is 3
by
  sorry

end NUMINAMATH_CALUDE_multiple_of_sum_and_smaller_l1823_182383


namespace NUMINAMATH_CALUDE_amanda_remaining_money_l1823_182351

/-- Calculates the remaining amount after purchases -/
def remaining_amount (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Proves that Amanda will have $7 left after her purchases -/
theorem amanda_remaining_money :
  remaining_amount 50 9 2 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_amanda_remaining_money_l1823_182351


namespace NUMINAMATH_CALUDE_cricket_theorem_l1823_182337

def cricket_problem (team_scores : List Nat) : Prop :=
  let n := team_scores.length
  let lost_matches := 6
  let won_matches := n - lost_matches
  let opponent_scores_lost := List.map (λ x => x + 2) (team_scores.take lost_matches)
  let opponent_scores_won := List.map (λ x => (x + 2) / 3) (team_scores.drop lost_matches)
  let total_opponent_score := opponent_scores_lost.sum + opponent_scores_won.sum
  
  n = 12 ∧
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] ∧
  total_opponent_score = 54

theorem cricket_theorem : 
  ∃ (team_scores : List Nat), cricket_problem team_scores :=
sorry

end NUMINAMATH_CALUDE_cricket_theorem_l1823_182337


namespace NUMINAMATH_CALUDE_investment_problem_investment_problem_proof_l1823_182315

/-- The investment problem -/
theorem investment_problem (a_investment : ℕ) (b_join_time : ℚ) (profit_ratio : ℚ × ℚ) : ℕ :=
  let a_investment := 27000
  let b_join_time := 7.5
  let profit_ratio := (2, 1)
  let total_months := 12
  let b_investment := a_investment * (total_months / (total_months - b_join_time)) * (profit_ratio.2 / profit_ratio.1)
  36000

/-- Proof of the investment problem -/
theorem investment_problem_proof : investment_problem 27000 (15/2) (2, 1) = 36000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_investment_problem_proof_l1823_182315


namespace NUMINAMATH_CALUDE_evaluate_expression_l1823_182317

theorem evaluate_expression : (47^2 - 28^2) + 100 = 1525 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1823_182317


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l1823_182330

/-- Given an infinite geometric series with common ratio 1/4 and sum 48,
    the second term of the sequence is 9. -/
theorem second_term_of_geometric_series :
  ∀ (a : ℝ), -- first term of the series
  let r : ℝ := (1 : ℝ) / 4 -- common ratio
  let S : ℝ := 48 -- sum of the series
  (S = a / (1 - r)) → -- formula for sum of infinite geometric series
  (a * r = 9) -- second term of the sequence
  := by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l1823_182330


namespace NUMINAMATH_CALUDE_expression_simplification_l1823_182376

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(2 + x) - 5*(1 - 3*x) = 24*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1823_182376


namespace NUMINAMATH_CALUDE_right_triangle_area_l1823_182359

theorem right_triangle_area (a b c m : ℝ) : 
  a = 10 →                -- One leg is 10
  m = 13 →                -- Shortest median is 13
  m^2 = (2*a^2 + 2*c^2 - b^2) / 4 →  -- Apollonius's theorem
  a^2 + b^2 = c^2 →       -- Pythagorean theorem
  a * b / 2 = 10 * Real.sqrt 69 :=   -- Area of the triangle
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1823_182359


namespace NUMINAMATH_CALUDE_zoo_field_trip_count_l1823_182333

/-- Represents the number of individuals at the zoo during the field trip -/
def ZooFieldTrip : Type :=
  { n : ℕ // n ≤ 100 }

/-- The initial class size -/
def initial_class_size : ℕ := 10

/-- The number of parents who volunteered as chaperones -/
def parent_chaperones : ℕ := 5

/-- The number of teachers who joined -/
def teachers : ℕ := 2

/-- The number of students who left -/
def students_left : ℕ := 10

/-- The number of chaperones who left -/
def chaperones_left : ℕ := 2

/-- Function to calculate the final number of individuals at the zoo -/
def final_zoo_count (init_class : ℕ) (parents : ℕ) (teachers : ℕ) (students_gone : ℕ) (chaperones_gone : ℕ) : ZooFieldTrip :=
  ⟨2 * init_class + parents + teachers - students_gone - chaperones_gone, by sorry⟩

/-- Theorem stating that the final number of individuals at the zoo is 15 -/
theorem zoo_field_trip_count :
  (final_zoo_count initial_class_size parent_chaperones teachers students_left chaperones_left).val = 15 := by
  sorry

end NUMINAMATH_CALUDE_zoo_field_trip_count_l1823_182333


namespace NUMINAMATH_CALUDE_min_m_value_l1823_182309

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(|x - a|)

theorem min_m_value (a : ℝ) :
  (∀ x, f a (1 + x) = f a (1 - x)) →
  (∃ m, ∀ x y, m ≤ x → x < y → f a x < f a y) →
  (∀ m', (∀ x y, m' ≤ x → x < y → f a x < f a y) → 1 ≤ m') :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l1823_182309


namespace NUMINAMATH_CALUDE_cos_105_degrees_l1823_182347

theorem cos_105_degrees : 
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l1823_182347


namespace NUMINAMATH_CALUDE_black_haired_girls_l1823_182385

/-- Represents the number of girls in the choir -/
def initial_total : ℕ := 80

/-- Represents the number of blonde-haired girls added -/
def blonde_added : ℕ := 10

/-- Represents the initial number of blonde-haired girls -/
def initial_blonde : ℕ := 30

/-- Theorem stating the number of black-haired girls in the choir -/
theorem black_haired_girls : 
  initial_total - (initial_blonde + blonde_added) = 50 := by
  sorry

end NUMINAMATH_CALUDE_black_haired_girls_l1823_182385


namespace NUMINAMATH_CALUDE_solve_equation_l1823_182390

theorem solve_equation (x : ℝ) : (3 * x - 7) / 4 = 14 → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1823_182390


namespace NUMINAMATH_CALUDE_system_solution_l1823_182338

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 17 - 2*x)
  (eq2 : x + z = -11 - 2*y)
  (eq3 : x + y = 9 - 2*z) :
  3*x + 3*y + 3*z = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1823_182338


namespace NUMINAMATH_CALUDE_circle_area_sum_l1823_182397

/-- The sum of the areas of an infinite series of circles, where the radius of the first
    circle is 2 inches and each subsequent circle's radius is one-third of its predecessor,
    is equal to 9π/2 square inches. -/
theorem circle_area_sum : 
  let radius : ℕ → ℝ := fun n => 2 * (1/3)^(n-1)
  let area : ℕ → ℝ := fun n => π * (radius n)^2
  (∑' n, area n) = (9 * π) / 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_sum_l1823_182397


namespace NUMINAMATH_CALUDE_dalton_needs_four_dollars_l1823_182325

/-- The amount of additional money Dalton needs to buy all items -/
def additional_money_needed (jump_rope_cost board_game_cost ball_cost saved_allowance uncle_gift : ℕ) : ℕ :=
  let total_cost := jump_rope_cost + board_game_cost + ball_cost
  let available_money := saved_allowance + uncle_gift
  if total_cost > available_money then
    total_cost - available_money
  else
    0

/-- Theorem stating that Dalton needs $4 more to buy all items -/
theorem dalton_needs_four_dollars : 
  additional_money_needed 7 12 4 6 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_dalton_needs_four_dollars_l1823_182325


namespace NUMINAMATH_CALUDE_basketball_team_callback_l1823_182373

/-- The number of students called back for the basketball team. -/
def students_called_back (girls boys not_called : ℕ) : ℕ :=
  girls + boys - not_called

/-- Theorem stating that 26 students were called back for the basketball team. -/
theorem basketball_team_callback : students_called_back 39 4 17 = 26 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_callback_l1823_182373


namespace NUMINAMATH_CALUDE_sports_club_intersection_l1823_182319

theorem sports_club_intersection (N B T X : ℕ) : 
  N = 30 ∧ B = 18 ∧ T = 19 ∧ (N - (B + T - X) = 2) → X = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_intersection_l1823_182319


namespace NUMINAMATH_CALUDE_median_squares_sum_l1823_182310

/-- For a triangle with sides a, b, c, medians m_a, m_b, m_c, and circumcircle diameter D,
    the sum of squares of medians equals 3/4 of the sum of squares of sides plus 3/4 of the square of the diameter. -/
theorem median_squares_sum (a b c m_a m_b m_c D : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_m_a : 0 < m_a) (h_pos_m_b : 0 < m_b) (h_pos_m_c : 0 < m_c)
  (h_pos_D : 0 < D)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : 4 * m_a^2 + a^2 = 2 * b^2 + 2 * c^2)
  (h_median_b : 4 * m_b^2 + b^2 = 2 * c^2 + 2 * a^2)
  (h_median_c : 4 * m_c^2 + c^2 = 2 * a^2 + 2 * b^2)
  (h_D : D ≥ max a (max b c)) :
  m_a^2 + m_b^2 + m_c^2 = 3/4 * (a^2 + b^2 + c^2) + 3/4 * D^2 :=
sorry

end NUMINAMATH_CALUDE_median_squares_sum_l1823_182310


namespace NUMINAMATH_CALUDE_factorization_equality_l1823_182352

theorem factorization_equality (a x y : ℝ) :
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1823_182352


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l1823_182336

/-- Represents the problem from "The Compendious Book on Calculation by Completion and Balancing" --/
theorem ancient_chinese_math_problem (x y : ℕ) : 
  (∀ (room_capacity : ℕ), 
    (room_capacity = 7 → 7 * x + 7 = y) ∧ 
    (room_capacity = 9 → 9 * (x - 1) = y)) ↔ 
  (7 * x + 7 = y ∧ 9 * (x - 1) = y) :=
sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l1823_182336


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l1823_182356

theorem isosceles_triangle_angles (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  (a = 40 ∧ b = c) ∨ (b = 40 ∧ a = c) ∨ (c = 40 ∧ a = b) →  -- One angle is 40° and it's an isosceles triangle
  ((b = 70 ∧ c = 70) ∨ (a = 100 ∧ b = 40) ∨ (a = 100 ∧ c = 40)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l1823_182356


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_8_l1823_182368

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (1 : ℕ)^k * (1 : ℕ)^(8 - k)) = 256 ∧
  Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_8_l1823_182368


namespace NUMINAMATH_CALUDE_distance_between_anastasia_and_bananastasia_l1823_182300

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 343

/-- The time difference in seconds between hearing Anastasia and Bananastasia when they yell simultaneously -/
def simultaneous_time_diff : ℝ := 5

/-- The time difference in seconds between hearing Bananastasia and Anastasia when Bananastasia yells first -/
def sequential_time_diff : ℝ := 5

/-- The distance between Anastasia and Bananastasia in meters -/
def distance : ℝ := 1715

theorem distance_between_anastasia_and_bananastasia :
  ∀ (d : ℝ),
  (d / speed_of_sound = simultaneous_time_diff) ∧
  (2 * d / speed_of_sound - d / speed_of_sound = sequential_time_diff) →
  d = distance := by
  sorry

end NUMINAMATH_CALUDE_distance_between_anastasia_and_bananastasia_l1823_182300


namespace NUMINAMATH_CALUDE_pauls_crayons_l1823_182322

theorem pauls_crayons (birthday_crayons : Float) (school_year_crayons : Float) (neighbor_crayons : Float)
  (h1 : birthday_crayons = 479.0)
  (h2 : school_year_crayons = 134.0)
  (h3 : neighbor_crayons = 256.0) :
  birthday_crayons + school_year_crayons + neighbor_crayons = 869.0 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_l1823_182322


namespace NUMINAMATH_CALUDE_class_size_l1823_182364

theorem class_size (chorus : ℕ) (band : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : chorus = 18)
  (h2 : band = 26)
  (h3 : both = 2)
  (h4 : neither = 8) :
  chorus + band - both + neither = 50 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1823_182364


namespace NUMINAMATH_CALUDE_opposite_of_six_l1823_182346

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of 6 is -6
theorem opposite_of_six : opposite 6 = -6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_six_l1823_182346


namespace NUMINAMATH_CALUDE_division_result_l1823_182307

theorem division_result : (-1/20) / (-1/4 - 2/5 + 9/10 - 3/2) = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1823_182307


namespace NUMINAMATH_CALUDE_circle_segment_angle_l1823_182320

theorem circle_segment_angle (r₁ r₂ r₃ : ℝ) (shaded_ratio : ℝ) :
  r₁ = 4 →
  r₂ = 3 →
  r₃ = 2 →
  shaded_ratio = 3 / 5 →
  ∃ θ : ℝ,
    θ > 0 ∧
    θ < π / 2 ∧
    (θ * (r₁^2 + r₂^2 + r₃^2)) / ((π - θ) * (r₁^2 + r₂^2 + r₃^2)) = shaded_ratio ∧
    θ = 3 * π / 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_segment_angle_l1823_182320


namespace NUMINAMATH_CALUDE_kyle_spent_one_third_l1823_182327

def dave_money : ℕ := 46
def kyle_initial_money : ℕ := 3 * dave_money - 12
def kyle_remaining_money : ℕ := 84

theorem kyle_spent_one_third : 
  (kyle_initial_money - kyle_remaining_money) / kyle_initial_money = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_kyle_spent_one_third_l1823_182327


namespace NUMINAMATH_CALUDE_pole_height_pole_height_is_8_5_l1823_182372

/-- The height of a pole given specific cable and person measurements -/
theorem pole_height (cable_length : ℝ) (cable_ground_distance : ℝ) 
  (person_height : ℝ) (person_distance : ℝ) : ℝ :=
  cable_length * person_height / (cable_ground_distance - person_distance)

/-- Proof that a pole is 8.5 meters tall given specific measurements -/
theorem pole_height_is_8_5 :
  pole_height 5 5 1.7 4 = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_pole_height_pole_height_is_8_5_l1823_182372


namespace NUMINAMATH_CALUDE_tenRowTrianglePieces_l1823_182398

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a1 n : ℕ) : ℕ := n * (2 * a1 + (n - 1)) / 2

/-- Represents a triangle structure with rods and connectors -/
structure Triangle where
  rows : ℕ
  rodSequence : ℕ → ℕ
  connectorSequence : ℕ → ℕ

/-- Calculates the total number of pieces in the triangle -/
def totalPieces (t : Triangle) : ℕ :=
  (arithmeticSum (t.rodSequence 1) t.rows) + (arithmeticSum (t.connectorSequence 1) (t.rows + 1))

/-- The specific 10-row triangle described in the problem -/
def tenRowTriangle : Triangle :=
  { rows := 10
  , rodSequence := fun n => 3 * n
  , connectorSequence := fun n => n }

/-- Theorem stating that the total number of pieces in the 10-row triangle is 231 -/
theorem tenRowTrianglePieces : totalPieces tenRowTriangle = 231 := by
  sorry

end NUMINAMATH_CALUDE_tenRowTrianglePieces_l1823_182398


namespace NUMINAMATH_CALUDE_fraction_simplification_l1823_182360

theorem fraction_simplification :
  (5 : ℝ) / (3 * Real.sqrt 50 + Real.sqrt 18 + 4 * Real.sqrt 8) = (5 * Real.sqrt 2) / 52 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1823_182360


namespace NUMINAMATH_CALUDE_horner_rule_operations_l1823_182379

/-- Horner's Rule evaluation for a polynomial -/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ × ℕ × ℕ :=
  let rec go : List ℤ → ℤ → ℕ → ℕ → ℤ × ℕ × ℕ
    | [], acc, mults, adds => (acc, mults, adds)
    | c :: cs, acc, mults, adds => go cs (c + x * acc) (mults + 1) (adds + 1)
  go (coeffs.reverse.tail) (coeffs.reverse.head!) 0 0

/-- The polynomial f(x) = 3x^6 + 4x^5 + 5x^4 + 6x^3 + 7x^2 + 8x + 1 -/
def f_coeffs : List ℤ := [1, 8, 7, 6, 5, 4, 3]

theorem horner_rule_operations :
  let (_, mults, adds) := horner_eval f_coeffs 4
  mults = 6 ∧ adds = 6 := by sorry

end NUMINAMATH_CALUDE_horner_rule_operations_l1823_182379


namespace NUMINAMATH_CALUDE_four_spheres_cover_all_rays_l1823_182304

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray in 3D space
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a ray intersects a sphere
def rayIntersectsSphere (r : Ray) (s : Sphere) : Prop :=
  sorry

-- Theorem statement
theorem four_spheres_cover_all_rays :
  ∃ (lightSource : Point3D) (s₁ s₂ s₃ s₄ : Sphere),
    ∀ (r : Ray),
      r.origin = lightSource →
      rayIntersectsSphere r s₁ ∨
      rayIntersectsSphere r s₂ ∨
      rayIntersectsSphere r s₃ ∨
      rayIntersectsSphere r s₄ :=
sorry

end NUMINAMATH_CALUDE_four_spheres_cover_all_rays_l1823_182304


namespace NUMINAMATH_CALUDE_sum_O_eq_321_l1823_182371

/-- O(n) represents the sum of odd digits in number n -/
def O (n : ℕ) : ℕ := sorry

/-- The sum of O(n) from 1 to 75 -/
def sum_O : ℕ := (Finset.range 75).sum (λ n => O (n + 1))

/-- Theorem: The sum of O(n) from 1 to 75 equals 321 -/
theorem sum_O_eq_321 : sum_O = 321 := by sorry

end NUMINAMATH_CALUDE_sum_O_eq_321_l1823_182371


namespace NUMINAMATH_CALUDE_constant_width_interior_angle_ge_120_l1823_182348

/-- A curve of constant width. -/
class ConstantWidthCurve (α : Type*) [MetricSpace α] where
  width : ℝ
  is_constant_width : ∀ (x y : α), dist x y ≤ width

/-- The interior angle at a point on a curve. -/
def interior_angle {α : Type*} [MetricSpace α] (c : ConstantWidthCurve α) (p : α) : ℝ := sorry

/-- Theorem: The interior angle at any corner point of a curve of constant width is at least 120 degrees. -/
theorem constant_width_interior_angle_ge_120 
  {α : Type*} [MetricSpace α] (c : ConstantWidthCurve α) (p : α) :
  interior_angle c p ≥ 120 := by sorry

end NUMINAMATH_CALUDE_constant_width_interior_angle_ge_120_l1823_182348


namespace NUMINAMATH_CALUDE_other_divisor_problem_l1823_182328

theorem other_divisor_problem (n : Nat) (d1 d2 : Nat) : 
  (n = 386) →
  (d1 = 35) →
  (n % d1 = 1) →
  (n % d2 = 1) →
  (∀ m : Nat, m < n → (m % d1 = 1 ∧ m % d2 = 1) → False) →
  (d2 = 11) := by
  sorry

end NUMINAMATH_CALUDE_other_divisor_problem_l1823_182328


namespace NUMINAMATH_CALUDE_fraction_simplification_l1823_182382

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : y - 1/x ≠ 0) : 
  (x - 1/y) / (y - 1/x) = x / y := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1823_182382


namespace NUMINAMATH_CALUDE_middle_terms_equal_l1823_182387

/-- Given two geometric progressions with positive terms satisfying certain conditions,
    prove that the middle terms are equal. -/
theorem middle_terms_equal (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
    (h_pos_a : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0)
    (h_pos_b : b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0)
    (h_geom_a : ∃ q : ℝ, q > 0 ∧ a₂ = a₁ * q ∧ a₃ = a₂ * q)
    (h_geom_b : ∃ r : ℝ, r > 0 ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r)
    (h_sum_eq : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
    (h_arith_prog : ∃ d : ℝ, a₂ * b₂ - a₁ * b₁ = d ∧ a₃ * b₃ - a₂ * b₂ = d) :
  a₂ = b₂ := by
  sorry

end NUMINAMATH_CALUDE_middle_terms_equal_l1823_182387


namespace NUMINAMATH_CALUDE_arc_length_unit_circle_30_degrees_l1823_182377

theorem arc_length_unit_circle_30_degrees :
  let r : ℝ := 1  -- radius of unit circle
  let θ : ℝ := 30 -- central angle in degrees
  let l : ℝ := θ * π * r / 180 -- arc length formula
  l = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_unit_circle_30_degrees_l1823_182377


namespace NUMINAMATH_CALUDE_number_difference_l1823_182399

theorem number_difference (S L : ℕ) (h1 : S = 476) (h2 : L = 6 * S + 15) :
  L - S = 2395 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1823_182399


namespace NUMINAMATH_CALUDE_pears_in_D_l1823_182369

/-- The number of baskets --/
def num_baskets : ℕ := 5

/-- The average number of fruits per basket --/
def avg_fruits_per_basket : ℕ := 25

/-- The number of apples in basket A --/
def apples_in_A : ℕ := 15

/-- The number of mangoes in basket B --/
def mangoes_in_B : ℕ := 30

/-- The number of peaches in basket C --/
def peaches_in_C : ℕ := 20

/-- The number of bananas in basket E --/
def bananas_in_E : ℕ := 35

/-- The theorem stating the number of pears in basket D --/
theorem pears_in_D : 
  (num_baskets * avg_fruits_per_basket) - (apples_in_A + mangoes_in_B + peaches_in_C + bananas_in_E) = 25 := by
  sorry

end NUMINAMATH_CALUDE_pears_in_D_l1823_182369


namespace NUMINAMATH_CALUDE_band_size_correct_l1823_182394

/-- The number of flutes that tried out -/
def flutes : ℕ := 20

/-- The number of clarinets that tried out -/
def clarinets : ℕ := 30

/-- The number of trumpets that tried out -/
def trumpets : ℕ := 60

/-- The number of pianists that tried out -/
def pianists : ℕ := 20

/-- The fraction of flutes that got in -/
def flute_acceptance : ℚ := 4/5

/-- The fraction of clarinets that got in -/
def clarinet_acceptance : ℚ := 1/2

/-- The fraction of trumpets that got in -/
def trumpet_acceptance : ℚ := 1/3

/-- The fraction of pianists that got in -/
def pianist_acceptance : ℚ := 1/10

/-- The total number of people in the band -/
def band_total : ℕ := 53

theorem band_size_correct :
  (flutes : ℚ) * flute_acceptance +
  (clarinets : ℚ) * clarinet_acceptance +
  (trumpets : ℚ) * trumpet_acceptance +
  (pianists : ℚ) * pianist_acceptance = band_total := by
  sorry

end NUMINAMATH_CALUDE_band_size_correct_l1823_182394


namespace NUMINAMATH_CALUDE_fraction_zero_implies_a_neg_two_l1823_182350

theorem fraction_zero_implies_a_neg_two (a : ℝ) :
  (a^2 - 4) / (a - 2) = 0 → a = -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_a_neg_two_l1823_182350


namespace NUMINAMATH_CALUDE_total_students_eq_920_l1823_182316

/-- The number of students in the third school -/
def students_third_school : ℕ := 200

/-- The number of students in the second school -/
def students_second_school : ℕ := students_third_school + 40

/-- The number of students in the first school -/
def students_first_school : ℕ := 2 * students_second_school

/-- The total number of students from all three schools -/
def total_students : ℕ := students_first_school + students_second_school + students_third_school

theorem total_students_eq_920 : total_students = 920 := by
  sorry

end NUMINAMATH_CALUDE_total_students_eq_920_l1823_182316


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l1823_182329

open Real

theorem tangent_perpendicular_to_line (a : ℝ) : 
  let f (x : ℝ) := (2 - cos x) / sin x
  let x₀ : ℝ := π / 2
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  (y₀ = 2) → (m * (-1/a) = -1) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l1823_182329


namespace NUMINAMATH_CALUDE_michaels_house_paint_area_l1823_182349

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a house -/
def totalPaintArea (numRooms : ℕ) (dimensions : RoomDimensions) (windowDoorArea : ℝ) : ℝ :=
  let wallArea := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintableArea := wallArea - windowDoorArea
  numRooms * paintableArea

/-- Theorem: The total area to be painted in Michael's house is 1600 square feet -/
theorem michaels_house_paint_area :
  let dimensions : RoomDimensions := ⟨14, 11, 9⟩
  totalPaintArea 4 dimensions 50 = 1600 := by sorry

end NUMINAMATH_CALUDE_michaels_house_paint_area_l1823_182349


namespace NUMINAMATH_CALUDE_christian_initial_savings_l1823_182334

/-- The price of the perfume in dollars -/
def perfume_price : ℚ := 50

/-- Sue's initial savings in dollars -/
def sue_initial : ℚ := 7

/-- The number of yards Christian mowed -/
def yards_mowed : ℕ := 4

/-- The price Christian charged per yard in dollars -/
def price_per_yard : ℚ := 5

/-- The number of dogs Sue walked -/
def dogs_walked : ℕ := 6

/-- The price Sue charged per dog in dollars -/
def price_per_dog : ℚ := 2

/-- The additional amount needed in dollars -/
def additional_needed : ℚ := 6

/-- Christian's earnings from mowing yards -/
def christian_earnings : ℚ := yards_mowed * price_per_yard

/-- Sue's earnings from walking dogs -/
def sue_earnings : ℚ := dogs_walked * price_per_dog

/-- Total money they have after their work -/
def total_after_work : ℚ := christian_earnings + sue_earnings + sue_initial

/-- Christian's initial savings -/
def christian_initial : ℚ := perfume_price - total_after_work - additional_needed

theorem christian_initial_savings : christian_initial = 5 := by
  sorry

end NUMINAMATH_CALUDE_christian_initial_savings_l1823_182334


namespace NUMINAMATH_CALUDE_ice_palace_staircase_steps_l1823_182332

theorem ice_palace_staircase_steps 
  (time_for_20_steps : ℕ) 
  (steps_20 : ℕ) 
  (total_time : ℕ) 
  (h1 : time_for_20_steps = 120)
  (h2 : steps_20 = 20)
  (h3 : total_time = 180) :
  (total_time * steps_20) / time_for_20_steps = 30 :=
by sorry

end NUMINAMATH_CALUDE_ice_palace_staircase_steps_l1823_182332


namespace NUMINAMATH_CALUDE_f_extrema_l1823_182305

def f (x : ℝ) := 3 * x^4 - 6 * x^2 + 4

theorem f_extrema :
  (∀ x ∈ Set.Icc (-1) 3, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 1) ∧
  (∀ x ∈ Set.Icc (-1) 3, f x ≤ 193) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 193) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l1823_182305


namespace NUMINAMATH_CALUDE_clear_denominators_l1823_182339

theorem clear_denominators (x : ℝ) : 
  (2*x + 1) / 3 - (10*x + 1) / 6 = 1 ↔ 4*x + 2 - 10*x - 1 = 6 := by
sorry

end NUMINAMATH_CALUDE_clear_denominators_l1823_182339


namespace NUMINAMATH_CALUDE_set_membership_problem_l1823_182391

theorem set_membership_problem (n : ℕ) (x y z w : ℕ) 
  (hn : n ≥ 4)
  (hx : x ∈ Finset.range n)
  (hy : y ∈ Finset.range n)
  (hz : z ∈ Finset.range n)
  (hw : w ∈ Finset.range n)
  (hS : Set.Mem (x, y, z) S ∧ Set.Mem (z, w, x) S) :
  Set.Mem (y, z, w) S ∧ Set.Mem (x, y, w) S :=
by
  sorry
where
  X : Finset ℕ := Finset.range n
  S : Set (ℕ × ℕ × ℕ) := 
    {p | p.1 ∈ X ∧ p.2.1 ∈ X ∧ p.2.2 ∈ X ∧
      ((p.1 < p.2.1 ∧ p.2.1 < p.2.2) ∨
       (p.2.1 < p.2.2 ∧ p.2.2 < p.1) ∨
       (p.2.2 < p.1 ∧ p.1 < p.2.1)) ∧
      ¬((p.1 < p.2.1 ∧ p.2.1 < p.2.2) ∧
        (p.2.1 < p.2.2 ∧ p.2.2 < p.1) ∧
        (p.2.2 < p.1 ∧ p.1 < p.2.1))}

end NUMINAMATH_CALUDE_set_membership_problem_l1823_182391


namespace NUMINAMATH_CALUDE_unique_lcm_gcd_relation_l1823_182345

theorem unique_lcm_gcd_relation : 
  ∃! (n : ℕ), n > 0 ∧ Nat.lcm n 100 = Nat.gcd n 100 + 450 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_lcm_gcd_relation_l1823_182345


namespace NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l1823_182335

/-- Represents the number of flies eaten daily in a swamp ecosystem -/
def flies_eaten_daily (gharials : ℕ) (fish_per_gharial : ℕ) (frogs_per_fish : ℕ) (flies_per_frog : ℕ) : ℕ :=
  gharials * fish_per_gharial * frogs_per_fish * flies_per_frog

/-- Theorem stating the number of flies eaten daily in the given swamp ecosystem -/
theorem swamp_ecosystem_flies_eaten :
  flies_eaten_daily 9 15 8 30 = 32400 := by
  sorry

end NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l1823_182335


namespace NUMINAMATH_CALUDE_diagonal_intersection_probability_l1823_182342

theorem diagonal_intersection_probability (n : ℕ) (h : n > 0) :
  let vertices := 2 * n + 1
  let total_diagonals := vertices * (vertices - 3) / 2
  let intersecting_diagonals := vertices.choose 4
  intersecting_diagonals / (total_diagonals.choose 2 : ℚ) = 
    n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersection_probability_l1823_182342


namespace NUMINAMATH_CALUDE_system_solution_l1823_182378

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Calculates the distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Calculates the distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Represents the system of equations -/
def satisfiesSystem (p : Point) (m : ℝ) : Prop :=
  2 * p.x - p.y = m ∧ 3 * p.x + 2 * p.y = m + 7

theorem system_solution :
  (∃ p : Point, satisfiesSystem p 0 ∧ p.x = 1 ∧ p.y = 2) ∧
  (∃ p : Point, ∃ m : ℝ,
    satisfiesSystem p m ∧
    isInSecondQuadrant p ∧
    distanceToXAxis p = 3 ∧
    distanceToYAxis p = 2 ∧
    m = -7) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l1823_182378


namespace NUMINAMATH_CALUDE_inequalities_proof_l1823_182363

theorem inequalities_proof (n : ℕ) (a : ℝ) (h1 : n ≥ 1) (h2 : a > 0) :
  2^(n-1) ≤ n! ∧ 
  n! ≤ n^n ∧ 
  (n+3)^2 ≤ 2^(n+3) ∧ 
  1 + n * a ≤ (1+a)^n := by
  sorry


end NUMINAMATH_CALUDE_inequalities_proof_l1823_182363


namespace NUMINAMATH_CALUDE_multiples_of_seven_ending_in_five_l1823_182341

/-- The count of positive multiples of 7 less than 2000 that end with the digit 5 -/
theorem multiples_of_seven_ending_in_five (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k ∧ n < 2000 ∧ n % 10 = 5) ↔ n ∈ Finset.range 29 :=
sorry

end NUMINAMATH_CALUDE_multiples_of_seven_ending_in_five_l1823_182341


namespace NUMINAMATH_CALUDE_range_of_a_l1823_182353

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (a + 6) + y^2 / (a - 7) = 1 ∧ 
  (∃ (b c : ℝ), (x = 0 ∧ y = b) ∨ (x = c ∧ y = 0))

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 4*x + a < 0

-- Define the theorem
theorem range_of_a : 
  (∀ a : ℝ, p a ∨ ¬(q a)) → 
  ∀ a : ℝ, a ∈ Set.Ioi (-6) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1823_182353


namespace NUMINAMATH_CALUDE_ratio_of_sums_eleven_l1823_182343

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 0 + seq.a (n - 1)) / 2

theorem ratio_of_sums_eleven (a b : ArithmeticSequence)
    (h : ∀ n, a.a n / b.a n = (2 * n - 1) / (n + 1)) :
  sum_n a 11 / sum_n b 11 = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sums_eleven_l1823_182343


namespace NUMINAMATH_CALUDE_car_distance_l1823_182395

/-- Proves that a car traveling 3/4 as fast as a train going 80 miles per hour will cover 20 miles in 20 minutes -/
theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (travel_time : ℝ) :
  train_speed = 80 →
  car_speed_ratio = 3 / 4 →
  travel_time = 20 / 60 →
  car_speed_ratio * train_speed * travel_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l1823_182395


namespace NUMINAMATH_CALUDE_product_of_numbers_l1823_182301

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1823_182301


namespace NUMINAMATH_CALUDE_total_fruits_is_107_l1823_182321

/-- The number of fruits picked by George and Amelia -/
def total_fruits (george_oranges amelia_apples : ℕ) : ℕ :=
  let george_apples := amelia_apples + 5
  let amelia_oranges := george_oranges - 18
  (george_oranges + amelia_oranges) + (george_apples + amelia_apples)

/-- Theorem stating that the total number of fruits picked is 107 -/
theorem total_fruits_is_107 :
  total_fruits 45 15 = 107 := by sorry

end NUMINAMATH_CALUDE_total_fruits_is_107_l1823_182321


namespace NUMINAMATH_CALUDE_three_correct_propositions_l1823_182389

theorem three_correct_propositions (a b c d : ℝ) : 
  (∃! n : ℕ, n = 3 ∧ 
    (((a * b > 0 ∧ b * c - a * d > 0) → (c / a - d / b > 0)) ∧
     ((a * b > 0 ∧ c / a - d / b > 0) → (b * c - a * d > 0)) ∧
     ((b * c - a * d > 0 ∧ c / a - d / b > 0) → (a * b > 0)))) := by
  sorry

end NUMINAMATH_CALUDE_three_correct_propositions_l1823_182389


namespace NUMINAMATH_CALUDE_geometric_series_product_sum_limit_l1823_182362

/-- The limit of the sum of the product of corresponding terms from two geometric series --/
theorem geometric_series_product_sum_limit (a r s : ℝ) 
  (hr : |r| < 1) (hs : |s| < 1) : 
  (∑' n, a^2 * (r*s)^n) = a^2 / (1 - r*s) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_product_sum_limit_l1823_182362


namespace NUMINAMATH_CALUDE_cosine_product_equals_quarter_l1823_182384

theorem cosine_product_equals_quarter : 
  (1 + Real.cos (π/4)) * (1 + Real.cos (3*π/4)) * (1 + Real.cos (π/2)) * (1 - Real.cos (π/4)^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_equals_quarter_l1823_182384


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l1823_182312

theorem cylinder_radius_problem (h : ℝ) (r : ℝ) :
  h = 2 →
  (π * (r + 5)^2 * h - π * r^2 * h = π * r^2 * (h + 4) - π * r^2 * h) →
  r = (5 + 5 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l1823_182312


namespace NUMINAMATH_CALUDE_min_points_guarantee_victory_min_points_is_smallest_l1823_182396

/-- Represents the possible points a racer can earn in a single race -/
inductive RacePoints
  | first  : RacePoints
  | second : RacePoints
  | third  : RacePoints

/-- Converts RacePoints to its numerical value -/
def points_value : RacePoints → Nat
  | RacePoints.first  => 6
  | RacePoints.second => 4
  | RacePoints.third  => 2

/-- The total number of races in the championship -/
def num_races : Nat := 5

/-- Calculates the total points for a list of race results -/
def total_points (results : List RacePoints) : Nat :=
  results.map points_value |>.sum

/-- Checks if a list of race results is valid (has exactly num_races races) -/
def valid_results (results : List RacePoints) : Prop :=
  results.length = num_races

/-- The minimum points needed to guarantee victory -/
def min_points_for_victory : Nat := 26

theorem min_points_guarantee_victory :
  ∀ (results : List RacePoints),
    valid_results results →
    total_points results ≥ min_points_for_victory →
    ∀ (other_results : List RacePoints),
      valid_results other_results →
      total_points results > total_points other_results :=
sorry

theorem min_points_is_smallest :
  ∀ (n : Nat),
    n < min_points_for_victory →
    ∃ (results other_results : List RacePoints),
      valid_results results ∧
      valid_results other_results ∧
      total_points results = n ∧
      total_points other_results ≥ n :=
sorry

end NUMINAMATH_CALUDE_min_points_guarantee_victory_min_points_is_smallest_l1823_182396


namespace NUMINAMATH_CALUDE_isosceles_60_is_equilateral_l1823_182374

-- Define an isosceles triangle with one 60° angle
def IsoscelesTriangleWith60Degree (α β γ : ℝ) : Prop :=
  (α = β ∨ β = γ ∨ γ = α) ∧ (α = 60 ∨ β = 60 ∨ γ = 60)

-- Theorem statement
theorem isosceles_60_is_equilateral (α β γ : ℝ) :
  IsoscelesTriangleWith60Degree α β γ →
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_60_is_equilateral_l1823_182374


namespace NUMINAMATH_CALUDE_max_correct_is_38_l1823_182313

/-- Represents the scoring system and result of a multiple-choice test -/
structure TestScoring where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correct answers possible given a TestScoring -/
def max_correct_answers (ts : TestScoring) : ℕ :=
  sorry

/-- Theorem stating that for the given test conditions, the maximum number of correct answers is 38 -/
theorem max_correct_is_38 : 
  let ts : TestScoring := {
    total_questions := 60,
    correct_points := 5,
    blank_points := 0,
    incorrect_points := -2,
    total_score := 150
  }
  max_correct_answers ts = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_is_38_l1823_182313


namespace NUMINAMATH_CALUDE_article_sale_price_l1823_182355

/-- Given an article with cost price CP, prove that the selling price SP
    that yields the same percentage profit as the percentage loss when
    sold for 1280 is 1820, given that selling it for 1937.5 gives a 25% profit. -/
theorem article_sale_price (CP : ℝ) 
    (h1 : 1937.5 = CP * 1.25)  -- 25% profit condition
    (h2 : ∃ SP, (SP - CP) / CP = (CP - 1280) / CP)  -- Equal percentage condition
    : ∃ SP, SP = 1820 ∧ (SP - CP) / CP = (CP - 1280) / CP := by
  sorry

end NUMINAMATH_CALUDE_article_sale_price_l1823_182355


namespace NUMINAMATH_CALUDE_pie_difference_l1823_182340

/-- The number of pies baked per day -/
def pies_per_day : ℕ := 12

/-- The number of days apple pies are baked per week -/
def apple_pie_days : ℕ := 3

/-- The number of days cherry pies are baked per week -/
def cherry_pie_days : ℕ := 2

/-- Theorem: The difference between apple pies and cherry pies baked in one week is 12 -/
theorem pie_difference : 
  apple_pie_days * pies_per_day - cherry_pie_days * pies_per_day = 12 := by
  sorry

end NUMINAMATH_CALUDE_pie_difference_l1823_182340


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1823_182393

-- Problem 1
theorem problem_one (a b c : ℝ) (ha : |a| = 1) (hb : |b| = 2) (hc : |c| = 3) (horder : a > b ∧ b > c) :
  a + b - c = 2 ∨ a + b - c = 0 := by sorry

-- Problem 2
theorem problem_two (a b c d : ℚ) (hab : |a - b| ≤ 9) (hcd : |c - d| ≤ 16) (habcd : |a - b - c + d| = 25) :
  |b - a| - |d - c| = -7 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1823_182393


namespace NUMINAMATH_CALUDE_calculation_proof_l1823_182302

theorem calculation_proof : 3^2 + Real.sqrt 25 - (64 : ℝ)^(1/3) + abs (-9) = 19 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1823_182302


namespace NUMINAMATH_CALUDE_license_plate_count_l1823_182392

/-- The number of letters in the Rotokas alphabet -/
def rotokas_alphabet_size : ℕ := 12

/-- The set of allowed first letters -/
def first_letters : Finset Char := {'G', 'K', 'P'}

/-- The required last letter -/
def last_letter : Char := 'T'

/-- The forbidden letter -/
def forbidden_letter : Char := 'R'

/-- The length of the license plate -/
def license_plate_length : ℕ := 5

/-- Calculates the number of valid license plates -/
def count_license_plates : ℕ :=
  first_letters.card * (rotokas_alphabet_size - 5) * (rotokas_alphabet_size - 6) * (rotokas_alphabet_size - 7)

theorem license_plate_count :
  count_license_plates = 630 :=
sorry

end NUMINAMATH_CALUDE_license_plate_count_l1823_182392


namespace NUMINAMATH_CALUDE_tangent_sum_l1823_182367

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_l1823_182367


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1823_182381

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 2) :
  (1 - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1823_182381


namespace NUMINAMATH_CALUDE_pablo_blocks_sum_l1823_182331

/-- The number of blocks in Pablo's toy block stacks -/
def pablo_blocks : ℕ → ℕ
| 0 => 5  -- First stack
| 1 => pablo_blocks 0 + 2  -- Second stack
| 2 => pablo_blocks 1 - 5  -- Third stack
| 3 => pablo_blocks 2 + 5  -- Fourth stack
| _ => 0  -- No more stacks

/-- The total number of blocks used by Pablo -/
def total_blocks : ℕ := pablo_blocks 0 + pablo_blocks 1 + pablo_blocks 2 + pablo_blocks 3

theorem pablo_blocks_sum : total_blocks = 21 := by
  sorry

end NUMINAMATH_CALUDE_pablo_blocks_sum_l1823_182331


namespace NUMINAMATH_CALUDE_expression_evaluation_l1823_182303

theorem expression_evaluation :
  (3 : ℚ)^3010 * 2^3008 / 6^3009 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1823_182303


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1823_182326

theorem line_tangent_to_circle (k : ℝ) : 
  (∀ x y : ℝ, y = k * (x + Real.sqrt 3) → x^2 + (y - 1)^2 = 1 → 
    ∀ x' y' : ℝ, y' = k * (x' + Real.sqrt 3) → x'^2 + (y' - 1)^2 ≥ 1) →
  k = Real.sqrt 3 ∨ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1823_182326


namespace NUMINAMATH_CALUDE_reciprocal_location_l1823_182375

/-- A complex number is in the third quadrant if its real and imaginary parts are both negative -/
def in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- A complex number is inside the unit circle if its norm is less than 1 -/
def inside_unit_circle (z : ℂ) : Prop :=
  Complex.abs z < 1

/-- A complex number is in the second quadrant if its real part is negative and imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- A complex number is outside the unit circle if its norm is greater than 1 -/
def outside_unit_circle (z : ℂ) : Prop :=
  Complex.abs z > 1

theorem reciprocal_location (F : ℂ) :
  in_third_quadrant F ∧ inside_unit_circle F →
  in_second_quadrant (1 / F) ∧ outside_unit_circle (1 / F) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_location_l1823_182375


namespace NUMINAMATH_CALUDE_union_complement_problem_l1823_182324

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {2, 3, 4}
def B : Finset Nat := {2, 5}

theorem union_complement_problem : B ∪ (U \ A) = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_union_complement_problem_l1823_182324


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1823_182361

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 552) : x + (x + 1) = 47 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1823_182361


namespace NUMINAMATH_CALUDE_hours_until_visit_l1823_182354

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days until Joy sees her grandma -/
def days_until_visit : ℕ := 2

/-- Theorem: The number of hours until Joy sees her grandma is 48 -/
theorem hours_until_visit : days_until_visit * hours_per_day = 48 := by
  sorry

end NUMINAMATH_CALUDE_hours_until_visit_l1823_182354


namespace NUMINAMATH_CALUDE_lower_limit_of_b_l1823_182318

theorem lower_limit_of_b (a b : ℤ) (h1 : 10 ≤ a ∧ a ≤ 25) (h2 : b < 31) 
  (h3 : (a : ℚ) / b ≤ 4/3) : 19 ≤ b := by
  sorry

end NUMINAMATH_CALUDE_lower_limit_of_b_l1823_182318
