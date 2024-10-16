import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_l2156_215633

/-- The area of the triangle formed by two lines intersecting at (3,3) with slopes 1/3 and 3, 
    and a third line x + y = 12 -/
theorem triangle_area : ℝ := by
  -- Define the lines
  let line1 : ℝ → ℝ := fun x ↦ (1/3) * x + 2
  let line2 : ℝ → ℝ := fun x ↦ 3 * x - 6
  let line3 : ℝ → ℝ := fun x ↦ 12 - x

  -- Define the intersection points
  let A : ℝ × ℝ := (3, 3)
  let B : ℝ × ℝ := (4.5, 7.5)
  let C : ℝ × ℝ := (7.5, 4.5)

  -- Calculate the area of the triangle
  have area_formula : ℝ :=
    (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

  -- Assert that the area is equal to 9
  have area_eq_9 : area_formula = 9 := by sorry

  exact 9

end NUMINAMATH_CALUDE_triangle_area_l2156_215633


namespace NUMINAMATH_CALUDE_speed_limit_violation_percentage_l2156_215661

theorem speed_limit_violation_percentage
  (total_motorists : ℝ)
  (h1 : total_motorists > 0)
  (ticketed_percentage : ℝ)
  (h2 : ticketed_percentage = 40)
  (unticketed_speeders_percentage : ℝ)
  (h3 : unticketed_speeders_percentage = 20)
  : (ticketed_percentage / (100 - unticketed_speeders_percentage)) * 100 = 50 := by
  sorry

#check speed_limit_violation_percentage

end NUMINAMATH_CALUDE_speed_limit_violation_percentage_l2156_215661


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l2156_215696

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear --/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_b_value :
  ∀ b : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, b⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → b = -9 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l2156_215696


namespace NUMINAMATH_CALUDE_line_point_x_value_l2156_215639

/-- Given a line passing through points (x, -4) and (4, 1) with a slope of 1, prove that x = -1 -/
theorem line_point_x_value (x : ℝ) : 
  let p1 : ℝ × ℝ := (x, -4)
  let p2 : ℝ × ℝ := (4, 1)
  let slope : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  slope = 1 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_point_x_value_l2156_215639


namespace NUMINAMATH_CALUDE_sum_odd_sequence_to_99_l2156_215603

/-- Sum of arithmetic sequence -/
def sum_arithmetic_sequence (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: Sum of sequence 1+3+5+...+99 -/
theorem sum_odd_sequence_to_99 :
  sum_arithmetic_sequence 1 99 2 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_sequence_to_99_l2156_215603


namespace NUMINAMATH_CALUDE_no_real_solution_system_l2156_215636

theorem no_real_solution_system :
  ¬∃ (x y z : ℝ), (x + y + 2 + 4*x*y = 0) ∧ 
                  (y + z + 2 + 4*y*z = 0) ∧ 
                  (z + x + 2 + 4*z*x = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_system_l2156_215636


namespace NUMINAMATH_CALUDE_extended_segment_coordinates_l2156_215690

/-- Given two points A and B on a plane, and a point C such that BC = 2/3 * AB,
    this theorem proves that the coordinates of C can be determined. -/
theorem extended_segment_coordinates
  (A B : ℝ × ℝ)
  (hA : A = (-1, 3))
  (hB : B = (11, 7))
  (hC : ∃ C : ℝ × ℝ, (C.1 - B.1)^2 + (C.2 - B.2)^2 = (2/3)^2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2)) :
  ∃ C : ℝ × ℝ, C = (19, 29/3) :=
sorry

end NUMINAMATH_CALUDE_extended_segment_coordinates_l2156_215690


namespace NUMINAMATH_CALUDE_trajectory_equation_l2156_215608

/-- The trajectory of points equidistant from A(-1, 1, 0) and B(2, -1, -1) in 3D space -/
theorem trajectory_equation :
  ∀ (x y z : ℝ),
  (x + 1)^2 + (y - 1)^2 + z^2 = (x - 2)^2 + (y + 1)^2 + (z + 1)^2 →
  3*x - 2*y - z = 2 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2156_215608


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2156_215605

theorem negation_of_proposition (p : ∀ x : ℝ, x^2 - x + 1 > 0) :
  (∃ x_0 : ℝ, x_0^2 - x_0 + 1 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 - x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2156_215605


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_squared_l2156_215669

theorem sum_of_fourth_powers_squared (x y z : ℤ) (h : x + y + z = 0) :
  ∃ (n : ℤ), 2 * (x^4 + y^4 + z^4) = n^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_squared_l2156_215669


namespace NUMINAMATH_CALUDE_students_exceed_pets_l2156_215683

/-- The number of third-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 22

/-- The number of rabbits in each classroom -/
def rabbits_per_classroom : ℕ := 3

/-- The number of hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 5

/-- The theorem stating the difference between students and pets -/
theorem students_exceed_pets : 
  (num_classrooms * students_per_classroom) - 
  (num_classrooms * (rabbits_per_classroom + hamsters_per_classroom)) = 70 := by
  sorry

end NUMINAMATH_CALUDE_students_exceed_pets_l2156_215683


namespace NUMINAMATH_CALUDE_crossing_point_distance_less_than_one_l2156_215680

/-- Represents a ladder in the ditch -/
structure Ladder :=
  (length : ℝ)
  (base_point : ℝ × ℝ)
  (top_point : ℝ × ℝ)

/-- Represents the ditch setup -/
structure DitchSetup :=
  (width : ℝ)
  (height : ℝ)
  (ladder1 : Ladder)
  (ladder2 : Ladder)

/-- The crossing point of two ladders -/
def crossing_point (l1 l2 : Ladder) : ℝ × ℝ := sorry

/-- Distance from a point to the left wall of the ditch -/
def distance_to_left_wall (p : ℝ × ℝ) : ℝ := p.1

/-- Main theorem: The crossing point is less than 1m from the left wall -/
theorem crossing_point_distance_less_than_one (setup : DitchSetup) :
  setup.ladder1.length = 3 →
  setup.ladder2.length = 2 →
  setup.ladder1.base_point.1 = 0 →
  setup.ladder2.base_point.1 = setup.width →
  setup.ladder1.top_point.2 = setup.height →
  setup.ladder2.top_point.2 = setup.height →
  distance_to_left_wall (crossing_point setup.ladder1 setup.ladder2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_crossing_point_distance_less_than_one_l2156_215680


namespace NUMINAMATH_CALUDE_salary_decrease_equivalence_l2156_215649

-- Define the pay cuts
def first_cut : ℝ := 0.05
def second_cut : ℝ := 0.10
def third_cut : ℝ := 0.15

-- Define the function to calculate the equivalent single percentage decrease
def equivalent_decrease (c1 c2 c3 : ℝ) : ℝ :=
  (1 - (1 - c1) * (1 - c2) * (1 - c3)) * 100

-- State the theorem
theorem salary_decrease_equivalence :
  equivalent_decrease first_cut second_cut third_cut = 27.325 := by
  sorry

end NUMINAMATH_CALUDE_salary_decrease_equivalence_l2156_215649


namespace NUMINAMATH_CALUDE_bees_in_hive_l2156_215657

/-- The total number of bees in a hive after more bees fly in -/
def total_bees (initial : ℕ) (flew_in : ℕ) : ℕ :=
  initial + flew_in

/-- Theorem: Given 16 initial bees and 7 more flying in, the total is 23 -/
theorem bees_in_hive : total_bees 16 7 = 23 := by
  sorry

end NUMINAMATH_CALUDE_bees_in_hive_l2156_215657


namespace NUMINAMATH_CALUDE_books_on_shelves_l2156_215635

/-- The number of ways to place n distinct books onto k shelves with no empty shelf -/
def place_books (n : ℕ) (k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

theorem books_on_shelves :
  place_books 10 3 * Nat.factorial 10 = 55980 * Nat.factorial 10 :=
by sorry

end NUMINAMATH_CALUDE_books_on_shelves_l2156_215635


namespace NUMINAMATH_CALUDE_abs_difference_symmetry_l2156_215614

theorem abs_difference_symmetry (a b : ℚ) : |a - b| = |b - a| := by sorry

end NUMINAMATH_CALUDE_abs_difference_symmetry_l2156_215614


namespace NUMINAMATH_CALUDE_rectangle_rotation_path_length_l2156_215629

/-- The length of the path traveled by point A in a rectangle ABCD undergoing three 90° rotations -/
theorem rectangle_rotation_path_length (AB BC : ℝ) (hAB : AB = 3) (hBC : BC = 8) :
  let diagonal := Real.sqrt (AB^2 + BC^2)
  let first_rotation := (1/2) * π * diagonal
  let second_rotation := (3/2) * π
  let third_rotation := 4 * π
  first_rotation + second_rotation + third_rotation = ((1/2) * Real.sqrt 73 + 11/2) * π :=
sorry

end NUMINAMATH_CALUDE_rectangle_rotation_path_length_l2156_215629


namespace NUMINAMATH_CALUDE_abs_two_minus_sqrt_five_l2156_215625

theorem abs_two_minus_sqrt_five : |2 - Real.sqrt 5| = Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_minus_sqrt_five_l2156_215625


namespace NUMINAMATH_CALUDE_welcoming_and_planning_committees_l2156_215686

theorem welcoming_and_planning_committees 
  (n : ℕ) -- Number of students
  (h1 : Nat.choose n 2 = 10) -- There are 10 ways to choose 2 from n
  : Nat.choose n 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_welcoming_and_planning_committees_l2156_215686


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2156_215634

theorem sum_of_squares_and_square_of_sum : (5 + 7)^2 + (5^2 + 7^2) = 218 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2156_215634


namespace NUMINAMATH_CALUDE_valid_grid_has_twelve_red_cells_l2156_215685

/-- Represents the color of a cell -/
inductive Color
| Red
| Blue

/-- Represents a 4x4 grid of colored cells -/
def Grid := Fin 4 → Fin 4 → Color

/-- Returns the list of neighboring cells for a given position -/
def neighbors (i j : Fin 4) : List (Fin 4 × Fin 4) :=
  sorry

/-- Counts the number of neighbors of a given color -/
def countNeighbors (g : Grid) (i j : Fin 4) (c : Color) : Nat :=
  sorry

/-- Checks if the grid satisfies the conditions for red cells -/
def validRedCells (g : Grid) : Prop :=
  ∀ i j, g i j = Color.Red →
    countNeighbors g i j Color.Red > countNeighbors g i j Color.Blue

/-- Checks if the grid satisfies the conditions for blue cells -/
def validBlueCells (g : Grid) : Prop :=
  ∀ i j, g i j = Color.Blue →
    countNeighbors g i j Color.Red = countNeighbors g i j Color.Blue

/-- Counts the total number of red cells in the grid -/
def countRedCells (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that a valid grid has exactly 12 red cells -/
theorem valid_grid_has_twelve_red_cells (g : Grid)
  (h_red : validRedCells g)
  (h_blue : validBlueCells g)
  (h_both_colors : ∃ i j, g i j = Color.Red ∧ ∃ i' j', g i' j' = Color.Blue) :
  countRedCells g = 12 :=
sorry

end NUMINAMATH_CALUDE_valid_grid_has_twelve_red_cells_l2156_215685


namespace NUMINAMATH_CALUDE_dot_product_PA_PB_is_negative_one_l2156_215692

/-- The dot product of vectors PA and PB is always -1, where P is any point on the curve y = x + 2/x (x > 0),
    A is the foot of the perpendicular from P to y = x, and B is the foot of the perpendicular from P to the y-axis. -/
theorem dot_product_PA_PB_is_negative_one :
  ∀ (x₀ : ℝ), x₀ > 0 →
  let P : ℝ × ℝ := (x₀, x₀ + 2 / x₀)
  let A : ℝ × ℝ := (x₀ + 1 / x₀, x₀ + 1 / x₀)
  let B : ℝ × ℝ := (0, x₀ + 2 / x₀)
  let PA : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)
  let PB : ℝ × ℝ := (B.1 - P.1, B.2 - P.2)
  (PA.1 * PB.1 + PA.2 * PB.2 : ℝ) = -1 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_PA_PB_is_negative_one_l2156_215692


namespace NUMINAMATH_CALUDE_gcd_n4_plus_125_and_n_plus_5_l2156_215699

theorem gcd_n4_plus_125_and_n_plus_5 (n : ℕ) (h1 : n > 0) (h2 : ¬ 7 ∣ n) :
  (Nat.gcd (n^4 + 5^3) (n + 5) = 1) ∨ (Nat.gcd (n^4 + 5^3) (n + 5) = 3) := by
sorry

end NUMINAMATH_CALUDE_gcd_n4_plus_125_and_n_plus_5_l2156_215699


namespace NUMINAMATH_CALUDE_snack_eaters_final_count_l2156_215654

/-- Calculates the final number of snack eaters after a series of events -/
def final_snack_eaters (initial_gathering : ℕ) (initial_snackers : ℕ) 
  (first_outsiders : ℕ) (second_outsiders : ℕ) (third_leavers : ℕ) : ℕ :=
  let total_after_first_join := initial_snackers + first_outsiders
  let after_half_left := total_after_first_join / 2
  let after_second_join := after_half_left + second_outsiders
  let after_more_left := after_second_join - third_leavers
  after_more_left / 2

/-- Theorem stating that given the initial conditions, the final number of snack eaters is 20 -/
theorem snack_eaters_final_count :
  final_snack_eaters 200 100 20 10 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_snack_eaters_final_count_l2156_215654


namespace NUMINAMATH_CALUDE_max_a_value_l2156_215638

theorem max_a_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + 6 = 4 * x * y) :
  ∃ (a : ℝ), ∀ (b : ℝ), (∀ (u v : ℝ), u > 0 → v > 0 → u + v + 6 = 4 * u * v →
    u^2 + 2 * u * v + v^2 - b * u - b * v + 1 ≥ 0) → b ≤ a ∧ a = 10 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2156_215638


namespace NUMINAMATH_CALUDE_separation_leads_to_growth_and_blessing_l2156_215637

/-- Represents the separation experience between a child and their mother --/
structure SeparationExperience where
  duration : ℕ
  communication_frequency : ℕ
  visits : ℕ
  child_attitude : Bool
  mother_attitude : Bool

/-- Represents the outcome of the separation experience --/
inductive Outcome
  | PersonalGrowth
  | Blessing
  | Negative

/-- Function to determine the outcome of a separation experience --/
def determine_outcome (exp : SeparationExperience) : Outcome := sorry

/-- Theorem stating that a positive separation experience leads to personal growth and can be a blessing --/
theorem separation_leads_to_growth_and_blessing 
  (exp : SeparationExperience) 
  (h1 : exp.duration ≥ 3) 
  (h2 : exp.communication_frequency ≥ 300) 
  (h3 : exp.visits ≥ 1) 
  (h4 : exp.child_attitude = true) 
  (h5 : exp.mother_attitude = true) : 
  determine_outcome exp = Outcome.PersonalGrowth ∧ 
  determine_outcome exp = Outcome.Blessing := 
sorry

end NUMINAMATH_CALUDE_separation_leads_to_growth_and_blessing_l2156_215637


namespace NUMINAMATH_CALUDE_equation_solution_l2156_215648

theorem equation_solution (x : ℝ) : 
  (x^3 + 2*x + 1 > 0) → 
  ((16 * 5^(2*x - 1) - 2 * 5^(x - 1) - 0.048) * Real.log (x^3 + 2*x + 1) = 0) ↔ 
  (x = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2156_215648


namespace NUMINAMATH_CALUDE_handshake_theorem_l2156_215623

def number_of_people : ℕ := 12
def handshakes_per_person : ℕ := 3

def handshake_arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

theorem handshake_theorem :
  let M := handshake_arrangements number_of_people handshakes_per_person
  M = 6100940 ∧ M % 1000 = 940 := by sorry

end NUMINAMATH_CALUDE_handshake_theorem_l2156_215623


namespace NUMINAMATH_CALUDE_team_sports_count_l2156_215615

theorem team_sports_count (total_score : ℕ) : ∃ (n : ℕ), 
  (n > 0) ∧ 
  ((97 + total_score) / n = 90) ∧ 
  ((73 + total_score) / n = 87) → 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_team_sports_count_l2156_215615


namespace NUMINAMATH_CALUDE_equation_holds_iff_sum_twelve_l2156_215695

theorem equation_holds_iff_sum_twelve (a b c : ℕ+) 
  (ha : a < 12) (hb : b < 12) (hc : c < 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b + c ↔ b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_iff_sum_twelve_l2156_215695


namespace NUMINAMATH_CALUDE_inequality_proof_l2156_215630

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2156_215630


namespace NUMINAMATH_CALUDE_tetrahedron_plane_distance_l2156_215656

/-- Regular tetrahedron with side length 15 -/
def tetrahedron_side_length : ℝ := 15

/-- Heights of three vertices above the plane -/
def vertex_heights : Fin 3 → ℝ
  | 0 => 15
  | 1 => 17
  | 2 => 20
  | _ => 0  -- This case should never occur due to Fin 3

/-- The theorem stating the properties of the tetrahedron and plane -/
theorem tetrahedron_plane_distance :
  ∃ (r s t : ℕ), 
    r > 0 ∧ s > 0 ∧ t > 0 ∧
    (∃ (d : ℝ), d = (r - Real.sqrt s) / t ∧
      d > 0 ∧ 
      d < tetrahedron_side_length ∧
      (∀ i, d < vertex_heights i) ∧
      r + s + t = 930) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_plane_distance_l2156_215656


namespace NUMINAMATH_CALUDE_opposite_reciprocal_expression_l2156_215664

theorem opposite_reciprocal_expression (m n p q : ℝ) 
  (h1 : m + n = 0) 
  (h2 : p * q = 1) : 
  -2023 * m + 3 / (p * q) - 2023 * n = 3 := by
sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_expression_l2156_215664


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2156_215662

theorem contrapositive_equivalence (p q : Prop) :
  (p → ¬q) ↔ (q → ¬p) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2156_215662


namespace NUMINAMATH_CALUDE_max_x_plus_y_l2156_215622

theorem max_x_plus_y (x y : ℝ) (h : x^2 + 3*y^2 = 1) :
  ∃ (max_x max_y : ℝ), max_x^2 + 3*max_y^2 = 1 ∧
  ∀ (a b : ℝ), a^2 + 3*b^2 = 1 → a + b ≤ max_x + max_y ∧
  max_x = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_x_plus_y_l2156_215622


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_ratio_l2156_215647

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (2, sin θ) and b = (1, cos θ), then sin²θ / (1 + cos²θ) = 2/3 -/
theorem parallel_vectors_trig_ratio 
  (θ : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (2, Real.sin θ)) 
  (hb : b = (1, Real.cos θ)) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  (Real.sin θ)^2 / (1 + (Real.cos θ)^2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_ratio_l2156_215647


namespace NUMINAMATH_CALUDE_ball_box_theorem_l2156_215672

/-- Given a box with 60 balls where the probability of picking a white ball is 0.25,
    this theorem proves the number of white and black balls, and the number of
    additional white balls needed to change the probability to 2/5. -/
theorem ball_box_theorem (total_balls : ℕ) (prob_white : ℚ) :
  total_balls = 60 →
  prob_white = 1/4 →
  ∃ (white_balls black_balls additional_balls : ℕ),
    white_balls = 15 ∧
    black_balls = 45 ∧
    additional_balls = 15 ∧
    white_balls + black_balls = total_balls ∧
    (white_balls : ℚ) / total_balls = prob_white ∧
    ((white_balls + additional_balls : ℚ) / (total_balls + additional_balls) = 2/5) :=
by sorry

end NUMINAMATH_CALUDE_ball_box_theorem_l2156_215672


namespace NUMINAMATH_CALUDE_not_in_sample_l2156_215601

/-- Represents a systematic sampling problem -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_seats : Finset ℕ
  h_total : total_students = 60
  h_sample : sample_size = 5
  h_known : known_seats = {3, 15, 45, 53}

/-- The interval between sampled items in systematic sampling -/
def sample_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- Checks if a given seat number could be in the sample -/
def could_be_in_sample (s : SystematicSampling) (seat : ℕ) : Prop :=
  ∃ k, 0 < k ∧ k ≤ s.sample_size ∧ seat = k * (sample_interval s)

/-- The main theorem stating that 37 cannot be the remaining seat in the sample -/
theorem not_in_sample (s : SystematicSampling) : ¬(could_be_in_sample s 37) := by
  sorry

end NUMINAMATH_CALUDE_not_in_sample_l2156_215601


namespace NUMINAMATH_CALUDE_evaluate_expression_l2156_215663

theorem evaluate_expression (x y z w : ℚ) 
  (hx : x = 1/4)
  (hy : y = 1/3)
  (hz : z = -12)
  (hw : w = 5) :
  x^2 * y^3 * z + w = 179/36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2156_215663


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2156_215677

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (2 * x^2 - 12 * x + 1 = 0) ↔ ((x - 3)^2 = 17/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2156_215677


namespace NUMINAMATH_CALUDE_square_root_extraction_scheme_l2156_215665

theorem square_root_extraction_scheme (n : Nat) (root : Nat) : 
  n = 418089 ∧ root = 647 → root * root = n := by
  sorry

end NUMINAMATH_CALUDE_square_root_extraction_scheme_l2156_215665


namespace NUMINAMATH_CALUDE_chromatic_number_upper_bound_l2156_215691

-- Define a graph type
structure Graph :=
  (V : Type) -- Vertex set
  (E : V → V → Prop) -- Edge relation

-- Define the number of edges in a graph
def num_edges (G : Graph) : ℕ := sorry

-- Define the chromatic number of a graph
def chromatic_number (G : Graph) : ℕ := sorry

-- State the theorem
theorem chromatic_number_upper_bound (G : Graph) :
  chromatic_number G ≤ (1/2 : ℝ) + Real.sqrt (2 * (num_edges G : ℝ) + 1/4) := by
  sorry

end NUMINAMATH_CALUDE_chromatic_number_upper_bound_l2156_215691


namespace NUMINAMATH_CALUDE_hexagon_side_length_l2156_215688

/-- The side length of a regular hexagon given the distance between opposite sides -/
theorem hexagon_side_length (opposite_distance : ℝ) : 
  opposite_distance = 18 → 
  ∃ (side_length : ℝ), side_length = 12 * Real.sqrt 3 ∧ 
    opposite_distance = (Real.sqrt 3 / 2) * side_length :=
by sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l2156_215688


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2156_215676

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| = x^2 - 5*x + 6 ∧ x = 2 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2156_215676


namespace NUMINAMATH_CALUDE_product_of_powers_equals_thousand_l2156_215613

theorem product_of_powers_equals_thousand :
  (10 ^ 0.25) * (10 ^ 0.25) * (10 ^ 0.5) * (10 ^ 0.5) * (10 ^ 0.75) * (10 ^ 0.75) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_thousand_l2156_215613


namespace NUMINAMATH_CALUDE_area_ratio_is_one_twentieth_l2156_215606

/-- Represents the areas of regions A, B, and C in a shop --/
structure ShopAreas where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the shop after misallocation and reallocation --/
def FinalShop (initial : ShopAreas) : Prop :=
  -- After initial misallocation
  initial.b + 0.1 * initial.a - 0.5 * initial.b = 0.4 * (initial.a + initial.b) ∧
  -- Final ratio of A:B:C is 2:1:1
  ∃ (m : ℝ),
    2 * (initial.b + 0.1 * initial.a - 0.5 * initial.b + (0.5 * initial.c - m)) =
    initial.a - 0.1 * initial.a + 0.5 * initial.b + m ∧
    initial.a - 0.1 * initial.a + 0.5 * initial.b + m =
    2 * (initial.c - 0.5 * initial.c)

/-- The main theorem to prove --/
theorem area_ratio_is_one_twentieth (initial : ShopAreas) 
    (h : FinalShop initial) : 
    (0.5 * initial.c - (initial.a - 0.1 * initial.a + 0.5 * initial.b + 
    (2 * (initial.b + 0.1 * initial.a - 0.5 * initial.b + 
    (0.5 * initial.c - (initial.a - 0.1 * initial.a + 0.5 * initial.b))) - 
    (initial.b + 0.1 * initial.a - 0.5 * initial.b)))) / 
    (initial.a + initial.b + initial.c) = 1 / 20 := by
  sorry


end NUMINAMATH_CALUDE_area_ratio_is_one_twentieth_l2156_215606


namespace NUMINAMATH_CALUDE_swimmer_distance_l2156_215658

/-- Calculates the distance traveled by a swimmer against a current. -/
theorem swimmer_distance (still_water_speed : ℝ) (current_speed : ℝ) (time : ℝ) :
  still_water_speed > current_speed →
  still_water_speed = 20 →
  current_speed = 12 →
  time = 5 →
  (still_water_speed - current_speed) * time = 40 := by
sorry

end NUMINAMATH_CALUDE_swimmer_distance_l2156_215658


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l2156_215698

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center : ℝ × ℝ := reflect_about_y_neg_x original_center
  reflected_center = (3, -8) := by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l2156_215698


namespace NUMINAMATH_CALUDE_geometric_sequence_arithmetic_mean_l2156_215667

/-- The arithmetic mean of the first three terms of a geometric sequence 
    with first term 4 and common ratio 3 is 52/3. -/
theorem geometric_sequence_arithmetic_mean : 
  let a : ℝ := 4  -- First term
  let r : ℝ := 3  -- Common ratio
  let term1 : ℝ := a
  let term2 : ℝ := a * r
  let term3 : ℝ := a * r^2
  (term1 + term2 + term3) / 3 = 52 / 3 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_arithmetic_mean_l2156_215667


namespace NUMINAMATH_CALUDE_sum_of_digits_equality_l2156_215679

def num1 : ℕ := (10^100 - 1) / 9
def num2 : ℕ := 4 * ((10^50 - 1) / 9)

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem sum_of_digits_equality :
  sumOfDigits (num1 * num2) = sumOfDigits (4 * (10^150 - 10^100 - 10^50 + 1) / 81) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_equality_l2156_215679


namespace NUMINAMATH_CALUDE_sports_classes_theorem_l2156_215609

/-- The number of students in different sports classes -/
def sports_classes (x : ℕ) : ℕ × ℕ × ℕ :=
  let basketball := x
  let soccer := 2 * x - 2
  let volleyball := (soccer / 2) + 2
  (basketball, soccer, volleyball)

theorem sports_classes_theorem (x : ℕ) (h : 2 * x - 6 = 34) :
  sports_classes x = (20, 34, 19) := by
  sorry

end NUMINAMATH_CALUDE_sports_classes_theorem_l2156_215609


namespace NUMINAMATH_CALUDE_ellen_lego_problem_l2156_215655

/-- Represents Ellen's Lego collection and calculations -/
theorem ellen_lego_problem (initial : ℕ) (lost : ℕ) (found : ℕ) 
  (h1 : initial = 12560) (h2 : lost = 478) (h3 : found = 342) :
  let current := initial - lost + found
  (current = 12424) ∧ 
  (((lost : ℚ) / (initial : ℚ)) * 100 = 381 / 100) := by
  sorry

#check ellen_lego_problem

end NUMINAMATH_CALUDE_ellen_lego_problem_l2156_215655


namespace NUMINAMATH_CALUDE_students_history_not_statistics_l2156_215673

/-- Given a group of students with the following properties:
  * There are 89 students in total
  * 36 students are taking history
  * 32 students are taking statistics
  * 59 students are taking history or statistics or both
  This theorem proves that 27 students are taking history but not statistics. -/
theorem students_history_not_statistics 
  (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ)
  (h_total : total = 89)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_or_statistics : history_or_statistics = 59) :
  history - (history + statistics - history_or_statistics) = 27 := by
  sorry

end NUMINAMATH_CALUDE_students_history_not_statistics_l2156_215673


namespace NUMINAMATH_CALUDE_opposite_to_83_l2156_215640

/-- Represents a circle with 100 equally spaced points -/
def Circle := Fin 100

/-- A function assigning numbers 1 to 100 to the points on the circle -/
def numbering : Circle → Nat :=
  sorry

/-- Predicate to check if a number is opposite to another on the circle -/
def is_opposite (a b : Circle) : Prop :=
  sorry

/-- Predicate to check if numbers less than k are evenly distributed -/
def evenly_distributed (k : Nat) : Prop :=
  sorry

theorem opposite_to_83 (h : ∀ k, evenly_distributed k) :
  ∃ n : Circle, numbering n = 84 ∧ is_opposite n (⟨82, sorry⟩ : Circle) :=
sorry

end NUMINAMATH_CALUDE_opposite_to_83_l2156_215640


namespace NUMINAMATH_CALUDE_birthday_candle_cost_l2156_215607

/-- The cost of a box of candles given Kerry's birthday setup -/
def cost_of_candle_box (num_cakes : ℕ) (age : ℕ) (candles_per_box : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost

theorem birthday_candle_cost :
  let num_cakes : ℕ := 3
  let age : ℕ := 8
  let candles_per_box : ℕ := 12
  let total_cost : ℚ := 5
  cost_of_candle_box num_cakes age candles_per_box total_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_birthday_candle_cost_l2156_215607


namespace NUMINAMATH_CALUDE_book_pages_theorem_l2156_215684

/-- Calculate the number of digits used to number pages in a book -/
def digits_used (num_pages : ℕ) : ℕ :=
  let single_digit := min num_pages 9
  let double_digit := min (num_pages - 9) 90
  let triple_digit := max (num_pages - 99) 0
  single_digit + 2 * double_digit + 3 * triple_digit

theorem book_pages_theorem :
  ∃ (num_pages : ℕ), digits_used num_pages = 636 ∧ num_pages = 248 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l2156_215684


namespace NUMINAMATH_CALUDE_fishing_competition_l2156_215617

/-- Fishing Competition Problem -/
theorem fishing_competition (days : ℕ) (jackson_per_day : ℕ) (george_per_day : ℕ) (total_catch : ℕ) :
  days = 5 →
  jackson_per_day = 6 →
  george_per_day = 8 →
  total_catch = 90 →
  ∃ (jonah_per_day : ℕ),
    jonah_per_day = 4 ∧
    total_catch = days * (jackson_per_day + george_per_day + jonah_per_day) :=
by
  sorry


end NUMINAMATH_CALUDE_fishing_competition_l2156_215617


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2156_215616

/-- The focal length of the hyperbola x²/10 - y²/2 = 1 is 4√3 -/
theorem hyperbola_focal_length : 
  ∃ (f : ℝ), f = 4 * Real.sqrt 3 ∧ 
  f = 2 * Real.sqrt ((10 : ℝ) + 2) ∧
  ∀ (x y : ℝ), x^2 / 10 - y^2 / 2 = 1 → 
    ∃ (c : ℝ), c = Real.sqrt ((10 : ℝ) + 2) ∧ 
    f = 2 * c :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2156_215616


namespace NUMINAMATH_CALUDE_grandmothers_age_is_77_l2156_215697

/-- The grandmother's age is obtained by writing the Latin grade twice in a row -/
def grandmothers_age (latin_grade : ℕ) : ℕ := 11 * latin_grade

/-- The morning grade is obtained by dividing the grandmother's age by the number of kittens and subtracting fourteen-thirds -/
def morning_grade (age : ℕ) (kittens : ℕ) : ℚ := age / kittens - 14 / 3

theorem grandmothers_age_is_77 :
  ∃ (latin_grade : ℕ) (kittens : ℕ),
    latin_grade < 10 ∧
    kittens % 3 = 0 ∧
    grandmothers_age latin_grade = 77 ∧
    morning_grade (grandmothers_age latin_grade) kittens = latin_grade :=
by sorry

end NUMINAMATH_CALUDE_grandmothers_age_is_77_l2156_215697


namespace NUMINAMATH_CALUDE_equation_solution_l2156_215652

theorem equation_solution : 
  ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ (x = 105) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2156_215652


namespace NUMINAMATH_CALUDE_integral_x_plus_sqrt_one_minus_x_squared_l2156_215610

open Set
open MeasureTheory
open Interval
open Real

theorem integral_x_plus_sqrt_one_minus_x_squared : 
  ∫ x in (-1 : ℝ)..1, (x + Real.sqrt (1 - x^2)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_plus_sqrt_one_minus_x_squared_l2156_215610


namespace NUMINAMATH_CALUDE_sum_of_special_integers_l2156_215651

/-- A positive integer with exactly two positive divisors -/
def smallest_two_divisor_integer : ℕ+ := sorry

/-- The largest integer less than 150 with exactly three positive divisors -/
def largest_three_divisor_integer_below_150 : ℕ+ := sorry

/-- The sum of the smallest integer with two positive divisors and 
    the largest integer less than 150 with three positive divisors -/
theorem sum_of_special_integers : 
  (smallest_two_divisor_integer : ℕ) + (largest_three_divisor_integer_below_150 : ℕ) = 123 := by sorry

end NUMINAMATH_CALUDE_sum_of_special_integers_l2156_215651


namespace NUMINAMATH_CALUDE_book_problem_solution_l2156_215668

def book_problem (cost_loss : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : Prop :=
  let selling_price := cost_loss * (1 - loss_percent)
  let cost_gain := selling_price / (1 + gain_percent)
  cost_loss + cost_gain = 360

theorem book_problem_solution :
  book_problem 210 0.15 0.19 :=
sorry

end NUMINAMATH_CALUDE_book_problem_solution_l2156_215668


namespace NUMINAMATH_CALUDE_minimum_other_sales_l2156_215604

/-- Represents the sales distribution of a stationery store -/
structure SalesDistribution where
  pens : ℝ
  pencils : ℝ
  other : ℝ

/-- The sales distribution meets the store's goals -/
def MeetsGoals (s : SalesDistribution) : Prop :=
  s.pens = 40 ∧
  s.pencils = 28 ∧
  s.other ≥ 20 ∧
  s.pens + s.pencils + s.other = 100

theorem minimum_other_sales (s : SalesDistribution) (h : MeetsGoals s) :
  s.other = 32 ∧ s.pens + s.pencils + s.other = 100 := by
  sorry

#check minimum_other_sales

end NUMINAMATH_CALUDE_minimum_other_sales_l2156_215604


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2156_215600

/-- The total surface area of a rectangular solid. -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 9 meters, 
    width 8 meters, and depth 5 meters is 314 square meters. -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 9 8 5 = 314 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2156_215600


namespace NUMINAMATH_CALUDE_x_value_and_upper_bound_l2156_215687

theorem x_value_and_upper_bound :
  ∀ (x : ℤ) (u : ℚ),
    0 < x ∧ x < 7 ∧
    0 < x ∧ x < 15 ∧
    -1 < x ∧ x < 5 ∧
    0 < x ∧ x < u ∧
    x + 2 < 4 →
    x = 1 ∧ 1 < u ∧ u < 2 :=
by sorry

end NUMINAMATH_CALUDE_x_value_and_upper_bound_l2156_215687


namespace NUMINAMATH_CALUDE_division_and_addition_l2156_215632

theorem division_and_addition : (-75) / (-25) + (1 / 2) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l2156_215632


namespace NUMINAMATH_CALUDE_vector_sum_zero_l2156_215689

variable {V : Type*} [AddCommGroup V]

def closed_polygon (a b c f : V) : Prop :=
  a + (c - b) + (f - c) + (b - f) = 0

theorem vector_sum_zero (a b c f : V) (h : closed_polygon a b c f) :
  (b - a) + (f - c) + (c - b) + (a - f) = 0 := by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l2156_215689


namespace NUMINAMATH_CALUDE_arc_length_proof_l2156_215678

open Real

noncomputable def curve (x : ℝ) : ℝ := Real.log (5 / (2 * x))

theorem arc_length_proof (a b : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 8) :
  ∫ x in a..b, sqrt (1 + (deriv curve x) ^ 2) = 1 + (1 / 2) * log (3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_arc_length_proof_l2156_215678


namespace NUMINAMATH_CALUDE_solution_sum_l2156_215620

/-- Given a system of equations, prove that the sum of its solutions is 2020 -/
theorem solution_sum (x₀ y₀ : ℝ) 
  (eq1 : x₀^3 - 2023*x₀ = 2023*y₀ - y₀^3 - 2020)
  (eq2 : x₀^2 - x₀*y₀ + y₀^2 = 2022) :
  x₀ + y₀ = 2020 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l2156_215620


namespace NUMINAMATH_CALUDE_square_perimeter_l2156_215631

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 392) (h2 : side^2 = area) : 
  4 * side = 112 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2156_215631


namespace NUMINAMATH_CALUDE_article_cost_l2156_215624

/-- Calculates the final cost of an article after two years of inflation and price changes -/
def finalCost (originalCost : ℝ) (inflationRate : ℝ) 
  (year1Increase year1Decrease year2Increase year2Decrease : ℝ) : ℝ :=
  let adjustedCost1 := originalCost * (1 + inflationRate)
  let afterYear1 := adjustedCost1 * (1 + year1Increase) * (1 - year1Decrease)
  let adjustedCost2 := afterYear1 * (1 + inflationRate)
  adjustedCost2 * (1 + year2Increase) * (1 - year2Decrease)

/-- Theorem stating the final cost of the article after two years -/
theorem article_cost : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |finalCost 75 0.05 0.20 0.20 0.30 0.25 - 77.40| < ε :=
sorry

end NUMINAMATH_CALUDE_article_cost_l2156_215624


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2156_215666

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+1)(x+a) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ (x + 1) * (x + a)

theorem even_function_implies_a_eq_neg_one :
  IsEven (f a) → a = -1 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2156_215666


namespace NUMINAMATH_CALUDE_fruit_sales_problem_l2156_215653

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (ponkan_cost fuji_cost : ℚ)
  (h1 : 30 * ponkan_cost + 20 * fuji_cost = 2700)
  (h2 : 50 * ponkan_cost + 40 * fuji_cost = 4800)
  (ponkan_price fuji_price : ℚ)
  (h3 : ponkan_price = 80)
  (h4 : fuji_price = 60)
  (fuji_price_red1 fuji_price_red2 : ℚ)
  (h5 : fuji_price_red1 = fuji_price * (1 - 1/10))
  (h6 : fuji_price_red2 = fuji_price_red1 * (1 - 1/10))
  (profit : ℚ)
  (h7 : profit = 50 * (ponkan_price - ponkan_cost) + 
                 20 * (fuji_price - fuji_cost) +
                 10 * (fuji_price_red1 - fuji_cost) +
                 10 * (fuji_price_red2 - fuji_cost)) :
  ponkan_cost = 60 ∧ fuji_cost = 45 ∧ profit = 1426 := by
sorry

end NUMINAMATH_CALUDE_fruit_sales_problem_l2156_215653


namespace NUMINAMATH_CALUDE_square_sum_identity_l2156_215643

theorem square_sum_identity (y : ℝ) :
  (y - 2)^2 + 2*(y - 2)*(4 + y) + (4 + y)^2 = 4*(y + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l2156_215643


namespace NUMINAMATH_CALUDE_at_least_one_leq_neg_four_l2156_215642

theorem at_least_one_leq_neg_four (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 4 / b ≤ -4) ∨ (b + 4 / c ≤ -4) ∨ (c + 4 / a ≤ -4) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_leq_neg_four_l2156_215642


namespace NUMINAMATH_CALUDE_jacket_purchase_price_l2156_215619

/-- Proves that the purchase price of a jacket is $56 given the specified conditions --/
theorem jacket_purchase_price :
  ∀ (purchase_price selling_price sale_price : ℝ),
  selling_price = purchase_price + 0.3 * selling_price →
  sale_price = 0.8 * selling_price →
  sale_price - purchase_price = 8 →
  purchase_price = 56 := by
sorry

end NUMINAMATH_CALUDE_jacket_purchase_price_l2156_215619


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2156_215660

theorem cubic_equation_solution (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (9 * p^3 - 9 * q^3) * (p - q)⁻¹ = 49 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2156_215660


namespace NUMINAMATH_CALUDE_officer_selection_theorem_l2156_215645

def club_size : ℕ := 25
def num_officers : ℕ := 3

def ways_to_choose_officers : ℕ :=
  let ways_without_alice_bob := (club_size - 2) * (club_size - 3) * (club_size - 4)
  let ways_with_alice_bob := 3 * 2 * (club_size - 2)
  ways_without_alice_bob + ways_with_alice_bob

theorem officer_selection_theorem :
  ways_to_choose_officers = 10764 := by sorry

end NUMINAMATH_CALUDE_officer_selection_theorem_l2156_215645


namespace NUMINAMATH_CALUDE_violet_percentage_l2156_215612

/-- Represents a flower bouquet with yellow and purple flowers -/
structure Bouquet where
  total : ℕ
  yellow : ℕ
  purple : ℕ
  yellow_daisies : ℕ
  purple_violets : ℕ

/-- Conditions for the flower bouquet -/
def bouquet_conditions (b : Bouquet) : Prop :=
  b.total > 0 ∧
  b.yellow + b.purple = b.total ∧
  b.yellow = b.total / 2 ∧
  b.yellow_daisies = b.yellow / 5 ∧
  b.purple_violets = b.purple / 2

/-- Theorem: The percentage of violets in the bouquet is 25% -/
theorem violet_percentage (b : Bouquet) (h : bouquet_conditions b) :
  (b.purple_violets : ℚ) / b.total = 1/4 := by
  sorry

#check violet_percentage

end NUMINAMATH_CALUDE_violet_percentage_l2156_215612


namespace NUMINAMATH_CALUDE_chairs_left_is_54_l2156_215675

/-- The number of chairs left in Rodrigo's classroom after borrowing -/
def chairs_left : ℕ :=
  let red_chairs : ℕ := 4
  let yellow_chairs : ℕ := 2 * red_chairs
  let blue_chairs : ℕ := 3 * yellow_chairs
  let green_chairs : ℕ := blue_chairs / 2
  let orange_chairs : ℕ := green_chairs + 2
  let total_chairs : ℕ := red_chairs + yellow_chairs + blue_chairs + green_chairs + orange_chairs
  let borrowed_chairs : ℕ := 5 + 3
  total_chairs - borrowed_chairs

theorem chairs_left_is_54 : chairs_left = 54 := by
  sorry

end NUMINAMATH_CALUDE_chairs_left_is_54_l2156_215675


namespace NUMINAMATH_CALUDE_tank_drainage_rate_l2156_215628

/-- Prove that given the conditions of the tank filling problem, 
    the drainage rate of pipe C is 20 liters per minute. -/
theorem tank_drainage_rate 
  (tank_capacity : ℕ) 
  (fill_rate_A : ℕ) 
  (fill_rate_B : ℕ) 
  (total_time : ℕ) 
  (h1 : tank_capacity = 800)
  (h2 : fill_rate_A = 40)
  (h3 : fill_rate_B = 30)
  (h4 : total_time = 48)
  : ∃ (drain_rate_C : ℕ), 
    drain_rate_C = 20 ∧ 
    (total_time / 3) * (fill_rate_A + fill_rate_B - drain_rate_C) = tank_capacity :=
by sorry

end NUMINAMATH_CALUDE_tank_drainage_rate_l2156_215628


namespace NUMINAMATH_CALUDE_count_divisible_by_seven_l2156_215694

theorem count_divisible_by_seven : 
  let e₁ (b : ℕ) := b^3 + 3^b + b * 3^((b+1)/2)
  let e₂ (b : ℕ) := b^3 + 3^b - b * 3^((b+1)/2)
  (Finset.filter (fun b => (e₁ b * e₂ b) % 7 = 0) (Finset.range 500)).card = 71 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_seven_l2156_215694


namespace NUMINAMATH_CALUDE_probability_two_painted_faces_is_three_eighths_l2156_215641

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  total_small_cubes : ℕ
  small_cubes_with_two_painted_faces : ℕ

/-- The probability of selecting a small cube with exactly two painted faces -/
def probability_two_painted_faces (c : CutCube) : ℚ :=
  c.small_cubes_with_two_painted_faces / c.total_small_cubes

/-- A cube cut into 64 smaller cubes -/
def cube_64 : CutCube :=
  { total_small_cubes := 64,
    small_cubes_with_two_painted_faces := 24 }

theorem probability_two_painted_faces_is_three_eighths :
  probability_two_painted_faces cube_64 = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_painted_faces_is_three_eighths_l2156_215641


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l2156_215602

/-- Triangle DEF with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Tetrahedron ODEF -/
structure Tetrahedron where
  O : Point3D
  D : Point3D
  E : Point3D
  F : Point3D

def origin : Point3D := ⟨0, 0, 0⟩

/-- Volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: Volume of tetrahedron ODEF is 110/3 -/
theorem volume_of_specific_tetrahedron (tri : Triangle) (t : Tetrahedron) :
  tri.a = 8 ∧ tri.b = 10 ∧ tri.c = 12 ∧
  t.O = origin ∧
  t.D.y = 0 ∧ t.D.z = 0 ∧
  t.E.x = 0 ∧ t.E.z = 0 ∧
  t.F.x = 0 ∧ t.F.y = 0 →
  tetrahedronVolume t = 110 / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l2156_215602


namespace NUMINAMATH_CALUDE_jungkook_has_biggest_number_l2156_215650

def jungkook_number : ℕ := 6 * 3
def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5

theorem jungkook_has_biggest_number :
  jungkook_number > yoongi_number ∧ jungkook_number > yuna_number := by
  sorry

end NUMINAMATH_CALUDE_jungkook_has_biggest_number_l2156_215650


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2156_215644

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) ↔ (2 ≤ m ∧ m ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2156_215644


namespace NUMINAMATH_CALUDE_gps_primary_benefit_l2156_215611

/-- Represents the capabilities of GPS technology -/
structure GPSTechnology where
  navigation : Bool
  routeOptimization : Bool
  costReduction : Bool

/-- Represents the uses of GPS in mobile phones -/
structure GPSUses where
  travel : Bool
  tourism : Bool
  exploration : Bool

/-- Represents the primary benefit of GPS technology in daily life -/
def primaryBenefit (tech : GPSTechnology) : Prop :=
  tech.routeOptimization ∧ tech.costReduction

/-- The theorem stating that given GPS is used for travel, tourism, and exploration,
    its primary benefit is route optimization and cost reduction -/
theorem gps_primary_benefit (uses : GPSUses) (tech : GPSTechnology) 
  (h1 : uses.travel = true)
  (h2 : uses.tourism = true)
  (h3 : uses.exploration = true)
  (h4 : tech.navigation = true) :
  primaryBenefit tech :=
sorry

end NUMINAMATH_CALUDE_gps_primary_benefit_l2156_215611


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2156_215693

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 6}
def B : Set ℝ := {x | 3 * x^2 + x - 8 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Icc 0 (4/3) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2156_215693


namespace NUMINAMATH_CALUDE_linear_equation_condition_l2156_215627

theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ k m, (a + 3) * x^(|a| - 2) + 5 = k * x + m) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l2156_215627


namespace NUMINAMATH_CALUDE_only_C_is_certain_l2156_215646

-- Define the event type
inductive Event
  | A  -- The temperature in Aojiang on June 1st this year is 30 degrees
  | B  -- There are 10 red balls in a box, and any ball taken out must be a white ball
  | C  -- Throwing a stone, the stone will eventually fall
  | D  -- In this math competition, every participating student will score full marks

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.C => True
  | _ => False

-- Theorem statement
theorem only_C_is_certain :
  ∀ e : Event, is_certain e ↔ e = Event.C :=
by sorry

end NUMINAMATH_CALUDE_only_C_is_certain_l2156_215646


namespace NUMINAMATH_CALUDE_inequality_solution_l2156_215626

theorem inequality_solution (x : ℝ) : (x - 2) / (x + 5) ≤ 1 / 2 ↔ -5 < x ∧ x ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2156_215626


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l2156_215659

theorem chocolate_bar_cost (total_cost : ℚ) (num_chocolate_bars : ℕ) (num_gummy_packs : ℕ) (num_chip_bags : ℕ) 
  (gummy_pack_cost : ℚ) (chip_bag_cost : ℚ) :
  total_cost = 150 →
  num_chocolate_bars = 10 →
  num_gummy_packs = 10 →
  num_chip_bags = 20 →
  gummy_pack_cost = 2 →
  chip_bag_cost = 5 →
  (total_cost - (num_gummy_packs * gummy_pack_cost + num_chip_bags * chip_bag_cost)) / num_chocolate_bars = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l2156_215659


namespace NUMINAMATH_CALUDE_candy_distribution_l2156_215682

theorem candy_distribution (total_candy : ℕ) (total_bags : ℕ) (heart_bags : ℕ) (kiss_bags : ℕ) (jelly_bags : ℕ) :
  total_candy = 260 →
  total_bags = 13 →
  heart_bags = 4 →
  kiss_bags = 5 →
  jelly_bags = 3 →
  total_candy % total_bags = 0 →
  let pieces_per_bag := total_candy / total_bags
  let chew_bags := total_bags - heart_bags - kiss_bags - jelly_bags
  heart_bags * pieces_per_bag + chew_bags * pieces_per_bag + jelly_bags * pieces_per_bag = total_candy :=
by sorry

#check candy_distribution

end NUMINAMATH_CALUDE_candy_distribution_l2156_215682


namespace NUMINAMATH_CALUDE_triangle_inequality_l2156_215621

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2156_215621


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l2156_215670

theorem repeating_decimal_division : 
  let a := (36 : ℚ) / 99
  let b := (12 : ℚ) / 99
  a / b = 3 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l2156_215670


namespace NUMINAMATH_CALUDE_special_function_form_l2156_215681

/-- A positive continuous function satisfying the given inequality. -/
structure SpecialFunction where
  f : ℝ → ℝ
  continuous : Continuous f
  positive : ∀ x, f x > 0
  inequality : ∀ x y, f x - f y ≥ (x - y) * f ((x + y) / 2) * a
  a : ℝ

/-- The theorem stating that any function satisfying the SpecialFunction properties
    must be of the form c * exp(a * x) for some positive c. -/
theorem special_function_form (sf : SpecialFunction) :
  ∃ c : ℝ, c > 0 ∧ ∀ x, sf.f x = c * Real.exp (sf.a * x) := by
  sorry

end NUMINAMATH_CALUDE_special_function_form_l2156_215681


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2156_215618

/-- Given a line segment with one endpoint at (7, 4) and midpoint at (5, -8),
    the sum of coordinates of the other endpoint is -17. -/
theorem midpoint_coordinate_sum :
  ∀ x y : ℝ,
  (5 : ℝ) = (7 + x) / 2 →
  (-8 : ℝ) = (4 + y) / 2 →
  x + y = -17 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2156_215618


namespace NUMINAMATH_CALUDE_unique_solution_square_equation_l2156_215674

theorem unique_solution_square_equation :
  ∃! x : ℝ, (10 - x)^2 = x^2 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_square_equation_l2156_215674


namespace NUMINAMATH_CALUDE_month_mean_profit_l2156_215671

/-- Calculates the mean daily profit for a month given the mean profits of two equal periods -/
def mean_daily_profit (days : ℕ) (mean_profit1 : ℚ) (mean_profit2 : ℚ) : ℚ :=
  (mean_profit1 + mean_profit2) / 2

theorem month_mean_profit : 
  let days : ℕ := 30
  let first_half_mean : ℚ := 275
  let second_half_mean : ℚ := 425
  mean_daily_profit days first_half_mean second_half_mean = 350 := by
sorry

end NUMINAMATH_CALUDE_month_mean_profit_l2156_215671
