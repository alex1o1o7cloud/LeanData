import Mathlib

namespace NUMINAMATH_CALUDE_intersection_when_a_is_quarter_b_necessary_condition_for_a_l2627_262741

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 3*a + 1}

-- Theorem for part 1
theorem intersection_when_a_is_quarter :
  A ∩ B (1/4) = {x | 1 < x ∧ x < 7/4} := by sorry

-- Theorem for part 2
theorem b_necessary_condition_for_a (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ↔ 1/3 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_quarter_b_necessary_condition_for_a_l2627_262741


namespace NUMINAMATH_CALUDE_complex_sum_reciprocal_squared_l2627_262733

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem complex_sum_reciprocal_squared (x : ℂ) :
  x + (1 / x) = 5 → x^2 + (1 / x)^2 = (7 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_reciprocal_squared_l2627_262733


namespace NUMINAMATH_CALUDE_slope_condition_implies_y_value_l2627_262706

/-- Given two points P and Q in a coordinate plane, where P has coordinates (-3, 5) and Q has coordinates (5, y), prove that if the slope of the line through P and Q is -4/3, then y = -17/3. -/
theorem slope_condition_implies_y_value :
  let P : ℝ × ℝ := (-3, 5)
  let Q : ℝ × ℝ := (5, y)
  let slope := (Q.2 - P.2) / (Q.1 - P.1)
  slope = -4/3 → y = -17/3 :=
by
  sorry

end NUMINAMATH_CALUDE_slope_condition_implies_y_value_l2627_262706


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l2627_262709

theorem cement_mixture_weight :
  ∀ (total_weight : ℝ),
    (1/5 : ℝ) * total_weight +     -- Sand
    (3/4 : ℝ) * total_weight +     -- Water
    6 = total_weight →             -- Gravel
    total_weight = 120 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l2627_262709


namespace NUMINAMATH_CALUDE_sector_area_l2627_262752

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 16) (h2 : central_angle = 2) :
  let radius := perimeter / (2 + central_angle)
  (1 / 2) * radius^2 * central_angle = 16 := by sorry

end NUMINAMATH_CALUDE_sector_area_l2627_262752


namespace NUMINAMATH_CALUDE_baseball_games_per_month_l2627_262707

theorem baseball_games_per_month 
  (total_games : ℕ) 
  (season_months : ℕ) 
  (h1 : total_games = 14) 
  (h2 : season_months = 2) : 
  total_games / season_months = 7 := by
sorry

end NUMINAMATH_CALUDE_baseball_games_per_month_l2627_262707


namespace NUMINAMATH_CALUDE_vampire_blood_consumption_l2627_262710

/-- The amount of blood a vampire needs per week in gallons -/
def blood_needed_per_week : ℚ := 7

/-- The number of people the vampire sucks blood from each day -/
def people_per_day : ℕ := 4

/-- The number of pints in a gallon -/
def pints_per_gallon : ℕ := 8

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: Given a vampire who needs 7 gallons of blood per week and sucks blood from 4 people each day, 
    the amount of blood sucked per person is 2 pints. -/
theorem vampire_blood_consumption :
  (blood_needed_per_week * pints_per_gallon) / (people_per_day * days_per_week) = 2 := by
  sorry

end NUMINAMATH_CALUDE_vampire_blood_consumption_l2627_262710


namespace NUMINAMATH_CALUDE_arthur_baked_115_muffins_l2627_262738

/-- The number of muffins Arthur baked -/
def arthur_muffins : ℕ := 115

/-- The number of muffins James baked -/
def james_muffins : ℕ := 1380

/-- James baked 12 times as many muffins as Arthur -/
axiom james_baked_12_times : james_muffins = 12 * arthur_muffins

theorem arthur_baked_115_muffins : arthur_muffins = 115 := by
  sorry

end NUMINAMATH_CALUDE_arthur_baked_115_muffins_l2627_262738


namespace NUMINAMATH_CALUDE_P_lower_bound_and_equality_l2627_262794

/-- The number of 4k-digit numbers composed of digits 2 and 0 (not starting with 0) that are divisible by 2020 -/
def P (k : ℕ+) : ℕ := sorry

/-- The theorem stating the inequality and the condition for equality -/
theorem P_lower_bound_and_equality (k : ℕ+) :
  P k ≥ Nat.choose (2 * k - 1) k ^ 2 ∧
  (P k = Nat.choose (2 * k - 1) k ^ 2 ↔ k ≤ 9) :=
sorry

end NUMINAMATH_CALUDE_P_lower_bound_and_equality_l2627_262794


namespace NUMINAMATH_CALUDE_linearly_dependent_implies_k_equals_six_l2627_262771

/-- Two vectors in ℝ² are linearly dependent if there exist non-zero scalars such that their linear combination is zero. -/
def linearlyDependent (v w : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ a • v + b • w = (0, 0)

/-- The theorem states that if the vectors (2, 3) and (4, k) are linearly dependent, then k must equal 6. -/
theorem linearly_dependent_implies_k_equals_six :
  linearlyDependent (2, 3) (4, k) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_linearly_dependent_implies_k_equals_six_l2627_262771


namespace NUMINAMATH_CALUDE_abs_inequality_iff_gt_l2627_262724

theorem abs_inequality_iff_gt (a b : ℝ) (h : a * b > 0) :
  a * |a| > b * |b| ↔ a > b :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_iff_gt_l2627_262724


namespace NUMINAMATH_CALUDE_complement_of_angle_A_l2627_262791

-- Define the angle A
def angle_A : ℝ := 76

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Theorem statement
theorem complement_of_angle_A : complement angle_A = 14 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle_A_l2627_262791


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2627_262767

theorem sum_of_solutions (x : ℝ) : 
  (x^2 + 2023*x = 2025) → 
  (∃ y : ℝ, y^2 + 2023*y = 2025 ∧ x + y = -2023) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2627_262767


namespace NUMINAMATH_CALUDE_bela_winning_strategy_l2627_262701

/-- The game state, representing the current player and the list of chosen numbers -/
inductive GameState
  | bela (choices : List ℝ)
  | jenn (choices : List ℝ)

/-- The game rules -/
def GameRules (n : ℕ) : GameState → Prop :=
  λ state =>
    n > 10 ∧
    match state with
    | GameState.bela choices => choices.all (λ x => 0 ≤ x ∧ x ≤ n)
    | GameState.jenn choices => choices.all (λ x => 0 ≤ x ∧ x ≤ n)

/-- A valid move in the game -/
def ValidMove (n : ℕ) (state : GameState) (move : ℝ) : Prop :=
  0 ≤ move ∧ move ≤ n ∧
  match state with
  | GameState.bela choices => choices.all (λ x => |x - move| > 2)
  | GameState.jenn choices => choices.all (λ x => |x - move| > 2)

/-- Bela has a winning strategy -/
theorem bela_winning_strategy (n : ℕ) :
  n > 10 →
  ∃ (strategy : GameState → ℝ),
    ∀ (state : GameState),
      GameRules n state →
      (∃ (move : ℝ), ValidMove n state move) →
      ValidMove n state (strategy state) :=
sorry

end NUMINAMATH_CALUDE_bela_winning_strategy_l2627_262701


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l2627_262787

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 1) 
  (h2 : 1/x - 1/y = 9) : 
  x + y = -1/20 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l2627_262787


namespace NUMINAMATH_CALUDE_box_volume_increase_l2627_262788

theorem box_volume_increase (l w h : ℝ) : 
  l * w * h = 5000 →
  2 * (l * w + w * h + l * h) = 1850 →
  4 * (l + w + h) = 240 →
  (l + 3) * (w + 3) * (h + 3) = 8342 := by
sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2627_262788


namespace NUMINAMATH_CALUDE_james_shirts_count_l2627_262730

/-- The number of shirts James has -/
def num_shirts : ℕ := 10

/-- The number of pairs of pants James has -/
def num_pants : ℕ := 12

/-- The time it takes to fix a shirt (in hours) -/
def shirt_time : ℚ := 3/2

/-- The hourly rate charged by the tailor (in dollars) -/
def hourly_rate : ℕ := 30

/-- The total cost for fixing all shirts and pants (in dollars) -/
def total_cost : ℕ := 1530

theorem james_shirts_count :
  num_shirts = 10 ∧
  num_pants = 12 ∧
  shirt_time = 3/2 ∧
  hourly_rate = 30 ∧
  total_cost = 1530 →
  num_shirts * (shirt_time * hourly_rate) + num_pants * (2 * shirt_time * hourly_rate) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_james_shirts_count_l2627_262730


namespace NUMINAMATH_CALUDE_real_part_of_z_l2627_262732

theorem real_part_of_z (z : ℂ) : z = (2 + I) / (1 + I)^2 → z.re = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2627_262732


namespace NUMINAMATH_CALUDE_wire_service_reporters_l2627_262711

theorem wire_service_reporters (total : ℕ) (h_total : total > 0) :
  let local_politics := (18 : ℚ) / 100 * total
  let no_politics := (70 : ℚ) / 100 * total
  let cover_politics := total - no_politics
  let cover_not_local := cover_politics - local_politics
  (cover_not_local / cover_politics) = (2 : ℚ) / 5 := by
sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l2627_262711


namespace NUMINAMATH_CALUDE_negation_of_positive_square_plus_two_is_false_l2627_262713

theorem negation_of_positive_square_plus_two_is_false : 
  ¬(∃ x : ℝ, x^2 + 2 ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_negation_of_positive_square_plus_two_is_false_l2627_262713


namespace NUMINAMATH_CALUDE_complement_A_complement_B_intersection_A_complement_B_l2627_262786

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5 ∨ x = 6}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- State the theorems to be proved
theorem complement_A : (Set.univ \ A) = {x | x ≤ -1 ∨ (5 < x ∧ x < 6) ∨ x > 6} := by sorry

theorem complement_B : (Set.univ \ B) = {x | x < 2 ∨ x ≥ 5} := by sorry

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x | -1 < x ∧ x < 2 ∨ x = 5 ∨ x = 6} := by sorry

end NUMINAMATH_CALUDE_complement_A_complement_B_intersection_A_complement_B_l2627_262786


namespace NUMINAMATH_CALUDE_point_transformation_l2627_262760

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the transformations
def rotateY90 (p : Point3D) : Point3D :=
  { x := p.z, y := p.y, z := -p.x }

def reflectYZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

def reflectXZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

def reflectXY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

-- Define the sequence of transformations
def transformSequence (p : Point3D) : Point3D :=
  p |> rotateY90
    |> reflectYZ
    |> reflectXZ
    |> rotateY90
    |> reflectXZ
    |> reflectXY

-- Theorem statement
theorem point_transformation :
  let initial := Point3D.mk 2 2 2
  transformSequence initial = Point3D.mk (-2) 2 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l2627_262760


namespace NUMINAMATH_CALUDE_sum_of_m_and_n_l2627_262736

theorem sum_of_m_and_n (m n : ℝ) (h : m^2 + n^2 - 6*m + 10*n + 34 = 0) : m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_and_n_l2627_262736


namespace NUMINAMATH_CALUDE_inscribed_square_existence_uniqueness_l2627_262768

/-- A sector in a plane --/
structure Sector where
  center : Point
  p : Point
  q : Point

/-- Angle of a sector --/
def Sector.angle (s : Sector) : ℝ := sorry

/-- A square in a plane --/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Predicate to check if a square is inscribed in a sector according to the problem conditions --/
def isInscribed (sq : Square) (s : Sector) : Prop := sorry

/-- Theorem stating the existence and uniqueness of the inscribed square --/
theorem inscribed_square_existence_uniqueness (s : Sector) :
  (∃! sq : Square, isInscribed sq s) ↔ s.angle ≤ 180 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_existence_uniqueness_l2627_262768


namespace NUMINAMATH_CALUDE_x_value_proof_l2627_262742

theorem x_value_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^3 / y = 2)
  (h2 : y^3 / z = 6)
  (h3 : z^3 / x = 9) :
  x = (559872 : ℝ) ^ (1 / 38) :=
by sorry

end NUMINAMATH_CALUDE_x_value_proof_l2627_262742


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2627_262727

theorem quadratic_inequality_solution_sets (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : a₁ ≠ 0) (h₂ : b₁ ≠ 0) (h₃ : c₁ ≠ 0) 
  (h₄ : a₂ ≠ 0) (h₅ : b₂ ≠ 0) (h₆ : c₂ ≠ 0) :
  ¬(((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)) ↔
    ({x : ℝ | a₁ * x^2 + b₁ * x + c₁ > 0} = {x : ℝ | a₂ * x^2 + b₂ * x + c₂ > 0})) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2627_262727


namespace NUMINAMATH_CALUDE_corresponding_angles_equal_l2627_262745

theorem corresponding_angles_equal (α β γ : ℝ) : 
  α + β + γ = 180 → 
  (180 - α) + β + γ = 180 → 
  α = 180 - α ∧ β = β ∧ γ = γ := by
sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_l2627_262745


namespace NUMINAMATH_CALUDE_license_plate_count_l2627_262756

/-- The number of possible first letters for the license plate -/
def first_letter_choices : ℕ := 3

/-- The number of choices for each digit position -/
def digit_choices : ℕ := 10

/-- The number of digit positions after the letter -/
def num_digits : ℕ := 5

/-- The total number of possible license plates -/
def total_license_plates : ℕ := first_letter_choices * digit_choices ^ num_digits

theorem license_plate_count : total_license_plates = 300000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2627_262756


namespace NUMINAMATH_CALUDE_ramp_cost_calculation_l2627_262714

def ramp_installation_cost (permits_cost : ℝ) (contractor_labor_rate : ℝ) 
  (contractor_materials_rate : ℝ) (contractor_days : ℕ) (contractor_hours_per_day : ℝ) 
  (contractor_lunch_break : ℝ) (inspector_rate_discount : ℝ) (inspector_hours_per_day : ℝ) : ℝ :=
  let contractor_work_hours := (contractor_hours_per_day - contractor_lunch_break) * contractor_days
  let contractor_labor_cost := contractor_work_hours * contractor_labor_rate
  let materials_cost := contractor_work_hours * contractor_materials_rate
  let inspector_rate := contractor_labor_rate * (1 - inspector_rate_discount)
  let inspector_cost := inspector_rate * inspector_hours_per_day * contractor_days
  permits_cost + contractor_labor_cost + materials_cost + inspector_cost

theorem ramp_cost_calculation :
  ramp_installation_cost 250 150 50 3 5 0.5 0.8 2 = 3130 := by
  sorry

end NUMINAMATH_CALUDE_ramp_cost_calculation_l2627_262714


namespace NUMINAMATH_CALUDE_problem_statement_l2627_262743

theorem problem_statement (a : ℝ) (h_pos : a > 0) (h_eq : a^2 / (a^4 - a^2 + 1) = 4/37) :
  a^3 / (a^6 - a^3 + 1) = 8/251 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2627_262743


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2627_262782

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y : ℝ, (m + 2) * x + 3 * m * y + 7 = 0 ∧ 
               (m - 2) * x + (m + 2) * y - 5 = 0 → 
               ((m + 2) * (m - 2) + 3 * m * (m + 2) = 0)) → 
  m = 1/2 ∨ m = -2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2627_262782


namespace NUMINAMATH_CALUDE_line_circle_intersection_m_values_l2627_262793

/-- A line intersecting a circle -/
structure LineCircleIntersection where
  /-- The parameter m in the line equation x - y + m = 0 -/
  m : ℝ
  /-- The line intersects the circle x^2 + y^2 = 4 at two points -/
  intersects : ∃ (A B : ℝ × ℝ), A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4 ∧
                                 A.1 - A.2 + m = 0 ∧ B.1 - B.2 + m = 0
  /-- The length of the chord AB is 2√3 -/
  chord_length : ∃ (A B : ℝ × ℝ), (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12

/-- The theorem stating the possible values of m -/
theorem line_circle_intersection_m_values (lci : LineCircleIntersection) :
  lci.m = Real.sqrt 2 ∨ lci.m = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_m_values_l2627_262793


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2627_262747

-- Define the color type
inductive Color
  | White
  | Red
  | Black

-- Define the coloring function type
def ColoringFunction := ℤ × ℤ → Color

-- Define the property of a color appearing on infinitely many lines
def InfiniteLines (f : ColoringFunction) (c : Color) : Prop :=
  ∀ N : ℕ, ∃ y > N, ∀ M : ℕ, ∃ x > M, f (x, y) = c

-- Define the parallelogram property
def ParallelogramProperty (f : ColoringFunction) : Prop :=
  ∀ a b c : ℤ × ℤ,
    f a = Color.White → f b = Color.Red → f c = Color.Black →
    ∃ d : ℤ × ℤ, f d = Color.Red ∧ d = (a.1 + c.1 - b.1, a.2 + c.2 - b.2)

-- The main theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction,
  (InfiniteLines f Color.White) ∧
  (InfiniteLines f Color.Red) ∧
  (InfiniteLines f Color.Black) ∧
  ParallelogramProperty f :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2627_262747


namespace NUMINAMATH_CALUDE_rope_division_l2627_262749

theorem rope_division (initial_length : ℝ) (initial_cuts : ℕ) (final_cuts : ℕ) : 
  initial_length = 200 →
  initial_cuts = 4 →
  final_cuts = 2 →
  (initial_length / initial_cuts) / final_cuts = 25 := by
  sorry

end NUMINAMATH_CALUDE_rope_division_l2627_262749


namespace NUMINAMATH_CALUDE_initial_diaries_count_l2627_262797

theorem initial_diaries_count (initial : ℕ) : 
  (2 * initial - (2 * initial) / 4 = 18) → initial = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_diaries_count_l2627_262797


namespace NUMINAMATH_CALUDE_find_a_value_l2627_262739

theorem find_a_value (A B : Set ℝ) (a : ℝ) :
  A = {2^a, 3} →
  B = {2, 3} →
  A ∪ B = {1, 2, 3} →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l2627_262739


namespace NUMINAMATH_CALUDE_expected_different_faces_correct_l2627_262789

/-- A fair six-sided die is rolled six times. -/
def num_rolls : ℕ := 6

/-- The number of faces on the die. -/
def num_faces : ℕ := 6

/-- The expected number of different faces that will appear when rolling a fair six-sided die six times. -/
def expected_different_faces : ℚ :=
  (num_faces ^ num_rolls - (num_faces - 1) ^ num_rolls) / (num_faces ^ (num_rolls - 1))

/-- Theorem stating that the expected number of different faces is correct. -/
theorem expected_different_faces_correct :
  expected_different_faces = (6^6 - 5^6) / 6^5 := by
  sorry


end NUMINAMATH_CALUDE_expected_different_faces_correct_l2627_262789


namespace NUMINAMATH_CALUDE_parabola_directrix_l2627_262774

/-- The directrix of the parabola y = x^2 -/
theorem parabola_directrix : ∃ (k : ℝ), ∀ (x y : ℝ),
  y = x^2 → (4 * y + 1 = 0 ↔ y = k) := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2627_262774


namespace NUMINAMATH_CALUDE_total_cost_is_94_l2627_262762

/-- The cost of goods A, B, and C -/
structure GoodsCost where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (cost : GoodsCost) : Prop :=
  cost.A + 2 * cost.B + 3 * cost.C = 136 ∧
  3 * cost.A + 2 * cost.B + cost.C = 240

/-- The theorem to be proved -/
theorem total_cost_is_94 (cost : GoodsCost) (h : satisfies_conditions cost) : 
  cost.A + cost.B + cost.C = 94 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_94_l2627_262762


namespace NUMINAMATH_CALUDE_n_squared_divisible_by_144_l2627_262721

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∀ d : ℕ+, d ∣ n → d ≤ 12) : 144 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_n_squared_divisible_by_144_l2627_262721


namespace NUMINAMATH_CALUDE_exam_attendance_l2627_262795

theorem exam_attendance (passed_percentage : ℚ) (failed_count : ℕ) : 
  passed_percentage = 35/100 →
  failed_count = 546 →
  (failed_count : ℚ) / (1 - passed_percentage) = 840 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_attendance_l2627_262795


namespace NUMINAMATH_CALUDE_line_equation_through_point_parallel_to_line_l2627_262700

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_through_point_parallel_to_line 
  (given_line : Line) 
  (point : Point) 
  (h_point : point.x = 2 ∧ point.y = 1) 
  (h_given_line : given_line.a = 2 ∧ given_line.b = -1 ∧ given_line.c = 2) :
  ∃ (result_line : Line), 
    result_line.a = 2 ∧ 
    result_line.b = -1 ∧ 
    result_line.c = -3 ∧
    parallel result_line given_line ∧
    on_line point result_line :=
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_parallel_to_line_l2627_262700


namespace NUMINAMATH_CALUDE_product_of_four_expressions_l2627_262726

theorem product_of_four_expressions (A B C D : ℝ) : 
  A = (Real.sqrt 2018 + Real.sqrt 2019 + 1) →
  B = (-Real.sqrt 2018 - Real.sqrt 2019 - 1) →
  C = (Real.sqrt 2018 - Real.sqrt 2019 + 1) →
  D = (Real.sqrt 2019 - Real.sqrt 2018 + 1) →
  A * B * C * D = 9 := by sorry

end NUMINAMATH_CALUDE_product_of_four_expressions_l2627_262726


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l2627_262758

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ (∀ x, f x ≠ 0) := by sorry

theorem negation_of_cubic_equation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l2627_262758


namespace NUMINAMATH_CALUDE_trick_decks_spending_l2627_262792

/-- The total amount spent by Frank and his friend on trick decks -/
def total_spent (deck_price : ℕ) (frank_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * frank_decks + deck_price * friend_decks

/-- Theorem: Frank and his friend spent 35 dollars on trick decks -/
theorem trick_decks_spending : total_spent 7 3 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_spending_l2627_262792


namespace NUMINAMATH_CALUDE_lcm_of_36_and_132_l2627_262740

theorem lcm_of_36_and_132 (hcf : ℕ) (lcm : ℕ) :
  hcf = 12 →
  lcm = 36 * 132 / hcf →
  lcm = 396 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_132_l2627_262740


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2627_262783

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + f y) - f x = (x + f y)^4 - x^4) :
  (∀ x : ℝ, f x = 0) ∨ (∃ k : ℝ, ∀ x : ℝ, f x = x^4 + k) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2627_262783


namespace NUMINAMATH_CALUDE_right_angle_implies_acute_fraction_inequality_l2627_262702

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define angle measures
def angle_measure (T : Triangle) (vertex : ℕ) : ℝ := sorry

-- Statement 1
theorem right_angle_implies_acute (T : Triangle) :
  angle_measure T 3 = π / 2 → angle_measure T 2 < π / 2 := by sorry

-- Statement 2
theorem fraction_inequality (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hm : m > 0) :
  (b + m) / (a + m) > b / a := by sorry

end NUMINAMATH_CALUDE_right_angle_implies_acute_fraction_inequality_l2627_262702


namespace NUMINAMATH_CALUDE_find_r_l2627_262775

-- Define the polynomials f and g
def f (r a x : ℝ) : ℝ := (x - (r + 2)) * (x - (r + 6)) * (x - a)
def g (r b x : ℝ) : ℝ := (x - (r + 4)) * (x - (r + 8)) * (x - b)

-- State the theorem
theorem find_r : ∃ (r a b : ℝ), 
  (∀ x, f r a x - g r b x = 2 * r) → r = 48 / 17 := by
  sorry

end NUMINAMATH_CALUDE_find_r_l2627_262775


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2627_262798

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (2*x + y)⁻¹ + 4*(2*x + 3*y)⁻¹ = 1) :
  x + y ≥ 9/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    (2*x₀ + y₀)⁻¹ + 4*(2*x₀ + 3*y₀)⁻¹ = 1 ∧ x₀ + y₀ = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2627_262798


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2627_262753

theorem trigonometric_identities (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π) -- θ is in the second quadrant
  (h2 : Real.tan (2*θ) = -2*Real.sqrt 2) : -- tan 2θ = -2√2
  (Real.tan θ = -Real.sqrt 2 / 2) ∧ 
  ((2 * (Real.cos (θ/2))^2 - Real.sin θ - Real.tan (5*π/4)) / 
   (Real.sqrt 2 * Real.sin (θ + π/4)) = 4 + 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2627_262753


namespace NUMINAMATH_CALUDE_max_k_logarithmic_inequality_l2627_262708

theorem max_k_logarithmic_inequality (x₀ x₁ x₂ x₃ : ℝ) (h₀ : x₀ > x₁) (h₁ : x₁ > x₂) (h₂ : x₂ > x₃) (h₃ : x₃ > 0) :
  ∃ k : ℝ, k = 9 ∧ 
  ∀ k' : ℝ, k' > k → 
  ∃ x₀' x₁' x₂' x₃' : ℝ, x₀' > x₁' ∧ x₁' > x₂' ∧ x₂' > x₃' ∧ x₃' > 0 ∧
  (Real.log (x₀' / x₁') / Real.log (x₀ / x₁) + 
   Real.log (x₁' / x₂') / Real.log (x₁ / x₂) + 
   Real.log (x₂' / x₃') / Real.log (x₂ / x₃) ≤ 
   k' * Real.log (x₀' / x₃') / Real.log (x₀ / x₃)) :=
by sorry

end NUMINAMATH_CALUDE_max_k_logarithmic_inequality_l2627_262708


namespace NUMINAMATH_CALUDE_total_books_proof_l2627_262764

/-- The number of books taken by the librarian -/
def books_taken_by_librarian : ℕ := 7

/-- The number of books Jerry can fit on one shelf -/
def books_per_shelf : ℕ := 3

/-- The number of shelves Jerry needs -/
def shelves_needed : ℕ := 9

/-- The total number of books to put away -/
def total_books : ℕ := books_per_shelf * shelves_needed + books_taken_by_librarian

theorem total_books_proof : total_books = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_books_proof_l2627_262764


namespace NUMINAMATH_CALUDE_proposition_analysis_l2627_262703

theorem proposition_analysis (a b : ℝ) : 
  (∃ a b, a * b > 0 ∧ (a ≤ 0 ∨ b ≤ 0)) ∧ 
  (∃ a b, (a ≤ 0 ∨ b ≤ 0) ∧ a * b > 0) ∧
  (∀ a b, a * b ≤ 0 → a ≤ 0 ∨ b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_analysis_l2627_262703


namespace NUMINAMATH_CALUDE_operations_result_l2627_262715

-- Define operation S
def S (a b : ℤ) : ℤ := 4*a + 6*b

-- Define operation T
def T (a b : ℤ) : ℤ := 2*a - 3*b

-- Theorem statement
theorem operations_result : T (S 8 3) 4 = 88 := by
  sorry

end NUMINAMATH_CALUDE_operations_result_l2627_262715


namespace NUMINAMATH_CALUDE_egg_difference_is_thirteen_l2627_262734

/-- Represents the egg problem with given conditions --/
structure EggProblem where
  total_dozens : Nat
  trays : Nat
  dropped_trays : Nat
  first_tray_broken : Nat
  first_tray_cracked : Nat
  first_tray_slightly_cracked : Nat
  second_tray_shattered : Nat
  second_tray_cracked : Nat
  second_tray_slightly_cracked : Nat

/-- Calculates the difference between perfect eggs in undropped trays and cracked eggs in dropped trays --/
def egg_difference (p : EggProblem) : Nat :=
  let total_eggs := p.total_dozens * 12
  let eggs_per_tray := total_eggs / p.trays
  let undropped_trays := p.trays - p.dropped_trays
  let perfect_eggs := undropped_trays * eggs_per_tray
  let cracked_eggs := p.first_tray_cracked + p.second_tray_cracked
  perfect_eggs - cracked_eggs

/-- Theorem stating the difference is 13 for the given problem conditions --/
theorem egg_difference_is_thirteen : egg_difference {
  total_dozens := 4
  trays := 4
  dropped_trays := 2
  first_tray_broken := 3
  first_tray_cracked := 5
  first_tray_slightly_cracked := 2
  second_tray_shattered := 4
  second_tray_cracked := 6
  second_tray_slightly_cracked := 1
} = 13 := by
  sorry

end NUMINAMATH_CALUDE_egg_difference_is_thirteen_l2627_262734


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2627_262785

theorem quadratic_roots_sum (u v : ℝ) : 
  (u^2 - 5*u + 6 = 0) → 
  (v^2 - 5*v + 6 = 0) → 
  u^2 + v^2 + u + v = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2627_262785


namespace NUMINAMATH_CALUDE_laylas_track_distance_l2627_262799

/-- The distance Layla rode around the running track, given her total mileage and the distance to the high school. -/
theorem laylas_track_distance (total_mileage : ℝ) (distance_to_school : ℝ) 
  (h1 : total_mileage = 10)
  (h2 : distance_to_school = 3) :
  total_mileage - 2 * distance_to_school = 4 :=
by sorry

end NUMINAMATH_CALUDE_laylas_track_distance_l2627_262799


namespace NUMINAMATH_CALUDE_inverse_proportion_through_point_l2627_262769

/-- An inverse proportion function passing through (-2, 3) has the equation y = -6/x -/
theorem inverse_proportion_through_point (f : ℝ → ℝ) :
  (∀ x ≠ 0, ∃ k, f x = k / x) →  -- f is an inverse proportion function
  f (-2) = 3 →                   -- f passes through the point (-2, 3)
  ∀ x ≠ 0, f x = -6 / x :=       -- The equation of f is y = -6/x
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_through_point_l2627_262769


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2627_262720

theorem quadratic_inequality (d : ℝ) : 
  (∀ x : ℝ, x * (2 * x + 3) < d ↔ -5/2 < x ∧ x < 1) ↔ d = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2627_262720


namespace NUMINAMATH_CALUDE_bobby_blocks_l2627_262761

def total_blocks (initial_blocks : ℕ) (factor : ℕ) : ℕ :=
  initial_blocks + factor * initial_blocks

theorem bobby_blocks : total_blocks 2 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bobby_blocks_l2627_262761


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2627_262729

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∃ (x y : ℝ), ax - y + 2*a = 0 ∧ (2*a - 1)*x + a*y + a = 0) →
  (a*(2*a - 1) + (-1)*a = 0) →
  (a = 0 ∨ a = 1) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2627_262729


namespace NUMINAMATH_CALUDE_amount_with_r_l2627_262751

/-- Given three people sharing a total amount of money, where one person has
    two-thirds of what the other two have combined, this theorem proves
    the amount held by that person. -/
theorem amount_with_r (total : ℝ) (amount_r : ℝ) : 
  total = 7000 →
  amount_r = (2/3) * (total - amount_r) →
  amount_r = 2800 := by
sorry


end NUMINAMATH_CALUDE_amount_with_r_l2627_262751


namespace NUMINAMATH_CALUDE_number_of_boys_l2627_262778

theorem number_of_boys (total : ℕ) (x : ℕ) : 
  total = 150 → 
  x + (x * total) / 100 = total → 
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l2627_262778


namespace NUMINAMATH_CALUDE_perfect_square_solutions_l2627_262737

theorem perfect_square_solutions : 
  {n : ℤ | ∃ k : ℤ, n^2 + 8*n + 44 = k^2} = {2, -10} := by sorry

end NUMINAMATH_CALUDE_perfect_square_solutions_l2627_262737


namespace NUMINAMATH_CALUDE_total_pencils_l2627_262779

/-- Given 4.0 pencil boxes, each filled with 648.0 pencils, prove that the total number of pencils is 2592.0 -/
theorem total_pencils (num_boxes : Float) (pencils_per_box : Float) 
  (h1 : num_boxes = 4.0) 
  (h2 : pencils_per_box = 648.0) : 
  num_boxes * pencils_per_box = 2592.0 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l2627_262779


namespace NUMINAMATH_CALUDE_exist_x_y_different_squares_no_x_y_different_squares_in_range_l2627_262728

-- Define the property for two numbers to be different perfect squares
def areDifferentPerfectSquares (a b : ℕ) : Prop :=
  ∃ m n : ℕ, m ≠ n ∧ a = m^2 ∧ b = n^2

-- Theorem 1: Existence of x and y satisfying the condition
theorem exist_x_y_different_squares :
  ∃ x y : ℕ, areDifferentPerfectSquares (x*y + x) (x*y + y) :=
sorry

-- Theorem 2: Non-existence of x and y between 988 and 1991 satisfying the condition
theorem no_x_y_different_squares_in_range :
  ¬∃ x y : ℕ, 988 ≤ x ∧ x ≤ 1991 ∧ 988 ≤ y ∧ y ≤ 1991 ∧
    areDifferentPerfectSquares (x*y + x) (x*y + y) :=
sorry

end NUMINAMATH_CALUDE_exist_x_y_different_squares_no_x_y_different_squares_in_range_l2627_262728


namespace NUMINAMATH_CALUDE_area_to_paint_is_15_l2627_262777

/-- The area of the wall to be painted -/
def area_to_paint (wall_length wall_width blackboard_length blackboard_width : ℝ) : ℝ :=
  wall_length * wall_width - blackboard_length * blackboard_width

/-- Theorem: The area to be painted is 15 square meters -/
theorem area_to_paint_is_15 :
  area_to_paint 6 3 3 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_is_15_l2627_262777


namespace NUMINAMATH_CALUDE_trigonometric_ratio_equality_l2627_262763

theorem trigonometric_ratio_equality 
  (a b c α β : ℝ) 
  (eq1 : a * Real.cos α + b * Real.sin α = c)
  (eq2 : a * Real.cos β + b * Real.sin β = c) :
  ∃ (k : ℝ), k ≠ 0 ∧ 
    a = k * Real.cos ((α + β) / 2) ∧
    b = k * Real.sin ((α + β) / 2) ∧
    c = k * Real.cos ((α - β) / 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_ratio_equality_l2627_262763


namespace NUMINAMATH_CALUDE_magnitude_of_3_plus_i_squared_l2627_262746

theorem magnitude_of_3_plus_i_squared : 
  Complex.abs ((3 : ℂ) + Complex.I) ^ 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_3_plus_i_squared_l2627_262746


namespace NUMINAMATH_CALUDE_convex_polygon_mean_inequality_l2627_262712

/-- For a convex n-gon, the arithmetic mean of side lengths is less than the arithmetic mean of diagonal lengths -/
theorem convex_polygon_mean_inequality {n : ℕ} (hn : n ≥ 3) 
  (P : ℝ) (D : ℝ) (hP : P > 0) (hD : D > 0) :
  P / n < (2 * D) / (n * (n - 3)) := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_mean_inequality_l2627_262712


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l2627_262790

/-- The cost price of the bicycle for A -/
def cost_price_A : ℝ := sorry

/-- The selling price from A to B -/
def selling_price_B : ℝ := 1.20 * cost_price_A

/-- The selling price from B to C before tax -/
def selling_price_C : ℝ := 1.25 * selling_price_B

/-- The total cost for C including tax -/
def total_cost_C : ℝ := 1.15 * selling_price_C

/-- The selling price from C to D before discount -/
def selling_price_D1 : ℝ := 1.30 * total_cost_C

/-- The final selling price from C to D after discount -/
def selling_price_D2 : ℝ := 0.90 * selling_price_D1

/-- The final price D pays for the bicycle -/
def final_price_D : ℝ := 350

theorem bicycle_cost_price :
  cost_price_A = final_price_D / 2.01825 := by sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l2627_262790


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2627_262770

theorem ellipse_m_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2/m + y^2/(2*m-1) = 1 ∧ 
   ∃ (a b : ℝ), a > b ∧ a^2 = m ∧ b^2 = 2*m-1) ↔ 
  (1/2 < m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2627_262770


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2627_262755

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (0 < x ∧ x < 4) → (x^2 - 3*x < 0 → 0 < x ∧ x < 4) ∧ ¬(0 < x ∧ x < 4 → x^2 - 3*x < 0) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2627_262755


namespace NUMINAMATH_CALUDE_sum_of_products_l2627_262744

theorem sum_of_products (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x^2 + x*y + y^2 = 75)
  (h2 : y^2 + y*z + z^2 = 16)
  (h3 : z^2 + x*z + x^2 = 91) :
  x*y + y*z + x*z = 40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2627_262744


namespace NUMINAMATH_CALUDE_solve_equation_l2627_262796

theorem solve_equation (x y : ℝ) (h : 3 * x^2 - 2 * y = 1) :
  2025 + 2 * y - 3 * x^2 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2627_262796


namespace NUMINAMATH_CALUDE_rectangle_folding_cutting_perimeter_ratio_l2627_262781

theorem rectangle_folding_cutting_perimeter_ratio :
  let initial_length : ℚ := 6
  let initial_width : ℚ := 4
  let folded_length : ℚ := initial_length / 2
  let folded_width : ℚ := initial_width
  let cut_length : ℚ := folded_length
  let cut_width : ℚ := folded_width / 2
  let small_perimeter : ℚ := 2 * (cut_length + cut_width)
  let large_perimeter : ℚ := 2 * (folded_length + folded_width)
  small_perimeter / large_perimeter = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_folding_cutting_perimeter_ratio_l2627_262781


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2627_262718

theorem sin_300_degrees : 
  Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2627_262718


namespace NUMINAMATH_CALUDE_green_percentage_is_25_l2627_262780

def amber_pieces : ℕ := 20
def green_pieces : ℕ := 35
def clear_pieces : ℕ := 85

def total_pieces : ℕ := amber_pieces + green_pieces + clear_pieces

def percentage_green : ℚ := (green_pieces : ℚ) / (total_pieces : ℚ) * 100

theorem green_percentage_is_25 : percentage_green = 25 := by
  sorry

end NUMINAMATH_CALUDE_green_percentage_is_25_l2627_262780


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2627_262705

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statement
theorem intersection_complement_equality :
  N ∩ (U \ M) = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2627_262705


namespace NUMINAMATH_CALUDE_triangle_problem_l2627_262719

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (a + c = 6) →
  (b = 2) →
  (Real.cos B = 7/9) →
  (a = Real.sqrt (b^2 + c^2 - 2*b*c*(Real.cos A))) →
  (b = Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B))) →
  (c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos C))) →
  (Real.sin A / a = Real.sin B / b) →
  (Real.sin B / b = Real.sin C / c) →
  (A + B + C = Real.pi) →
  (a = 3 ∧ c = 3 ∧ Real.sin (A - B) = (10 * Real.sqrt 2) / 27) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2627_262719


namespace NUMINAMATH_CALUDE_average_L_value_l2627_262759

/-- Represents a coin configuration with H and T sides -/
def Configuration (n : ℕ) := Fin n → Bool

/-- The number of operations before stopping for a given configuration -/
def L (n : ℕ) (c : Configuration n) : ℕ :=
  sorry  -- Definition of L would go here

/-- The average value of L(C) over all 2^n possible initial configurations -/
def averageLValue (n : ℕ) : ℚ :=
  sorry  -- Definition of average L value would go here

/-- Theorem stating that the average L value is n(n+1)/4 -/
theorem average_L_value (n : ℕ) : 
  averageLValue n = ↑n * (↑n + 1) / 4 :=
sorry

end NUMINAMATH_CALUDE_average_L_value_l2627_262759


namespace NUMINAMATH_CALUDE_max_modest_number_l2627_262722

def is_modest_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧
  5 * a = b + c + d

def G (a b c d : ℕ) : ℤ :=
  10 * a + b - 10 * c - d

theorem max_modest_number :
  ∀ (a b c d : ℕ),
    is_modest_number a b c d →
    d % 2 = 0 →
    (G a b c d) % 11 = 0 →
    (a + b + c) % 3 = 0 →
    a * 1000 + b * 100 + c * 10 + d ≤ 3816 :=
by sorry

end NUMINAMATH_CALUDE_max_modest_number_l2627_262722


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2627_262773

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (-(-8) + Real.sqrt ((-8)^2 - 4*1*(-12))) / (2*1)
  let r₂ := (-(-8) - Real.sqrt ((-8)^2 - 4*1*(-12))) / (2*1)
  r₁ + r₂ = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2627_262773


namespace NUMINAMATH_CALUDE_interior_angle_regular_octagon_l2627_262725

theorem interior_angle_regular_octagon :
  let sum_exterior_angles : ℝ := 360
  let num_sides : ℕ := 8
  let exterior_angle : ℝ := sum_exterior_angles / num_sides
  let interior_angle : ℝ := 180 - exterior_angle
  interior_angle = 135 := by
sorry

end NUMINAMATH_CALUDE_interior_angle_regular_octagon_l2627_262725


namespace NUMINAMATH_CALUDE_total_fish_count_l2627_262717

theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1 : num_fishbowls = 261) 
  (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := by
sorry

end NUMINAMATH_CALUDE_total_fish_count_l2627_262717


namespace NUMINAMATH_CALUDE_area_ratio_small_large_triangles_l2627_262716

/-- The ratio of areas between four small equilateral triangles and one large equilateral triangle -/
theorem area_ratio_small_large_triangles : 
  let small_side : ℝ := 10
  let small_perimeter : ℝ := 3 * small_side
  let total_perimeter : ℝ := 4 * small_perimeter
  let large_side : ℝ := total_perimeter / 3
  let small_area : ℝ := (Real.sqrt 3 / 4) * small_side ^ 2
  let large_area : ℝ := (Real.sqrt 3 / 4) * large_side ^ 2
  (4 * small_area) / large_area = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_small_large_triangles_l2627_262716


namespace NUMINAMATH_CALUDE_rhett_salary_l2627_262784

/-- Rhett's monthly salary calculation --/
theorem rhett_salary (monthly_rent : ℝ) (tax_rate : ℝ) (late_payments : ℕ) 
  (after_tax_fraction : ℝ) (salary : ℝ) :
  monthly_rent = 1350 →
  tax_rate = 0.1 →
  late_payments = 2 →
  after_tax_fraction = 3/5 →
  after_tax_fraction * (1 - tax_rate) * salary = late_payments * monthly_rent →
  salary = 5000 := by
sorry

end NUMINAMATH_CALUDE_rhett_salary_l2627_262784


namespace NUMINAMATH_CALUDE_expression_value_l2627_262766

theorem expression_value : 12 * (1 / 15) * 30 - 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2627_262766


namespace NUMINAMATH_CALUDE_line_vertical_translation_l2627_262731

/-- The equation of a line after vertical translation -/
theorem line_vertical_translation (x y : ℝ) :
  (y = x) → (y = x + 2) ↔ (∀ point : ℝ × ℝ, point.2 = point.1 + 2 ↔ point.2 = point.1 + 2) :=
by sorry

end NUMINAMATH_CALUDE_line_vertical_translation_l2627_262731


namespace NUMINAMATH_CALUDE_tan_alpha_negative_three_l2627_262704

theorem tan_alpha_negative_three (α : Real) (h : Real.tan α = -3) :
  (3 * Real.sin α - 3 * Real.cos α) / (6 * Real.cos α + Real.sin α) = -4 ∧
  1 / (Real.sin α * Real.cos α + 1 + Real.cos (2 * α)) = -10 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_negative_three_l2627_262704


namespace NUMINAMATH_CALUDE_unique_root_continuous_monotonic_l2627_262776

theorem unique_root_continuous_monotonic {α : Type*} [LinearOrder α] [TopologicalSpace α] {f : α → ℝ} {a b : α} (h_cont : Continuous f) (h_mono : Monotone f) (h_sign : f a * f b < 0) : ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_root_continuous_monotonic_l2627_262776


namespace NUMINAMATH_CALUDE_rectangular_field_area_difference_l2627_262735

theorem rectangular_field_area_difference : 
  let stan_length : ℕ := 30
  let stan_width : ℕ := 50
  let isla_length : ℕ := 35
  let isla_width : ℕ := 55
  let stan_area := stan_length * stan_width
  let isla_area := isla_length * isla_width
  isla_area - stan_area = 425 ∧ isla_area > stan_area := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_difference_l2627_262735


namespace NUMINAMATH_CALUDE_trapezoid_area_l2627_262723

/-- The area of a trapezoid bounded by y = 2x, y = 10, y = 5, and the y-axis -/
theorem trapezoid_area : ∃ (A : ℝ), A = 18.75 ∧ 
  A = ((5 - 0) + (10 - 5)) / 2 * 5 ∧
  (∀ x y : ℝ, (y = 2*x ∨ y = 10 ∨ y = 5 ∨ x = 0) → 
    0 ≤ x ∧ x ≤ 5 ∧ 5 ≤ y ∧ y ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2627_262723


namespace NUMINAMATH_CALUDE_soccer_team_captains_l2627_262754

theorem soccer_team_captains (n : ℕ) (k : ℕ) (h1 : n = 14) (h2 : k = 3) :
  Nat.choose n k = 364 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_captains_l2627_262754


namespace NUMINAMATH_CALUDE_sum_interior_angles_30_vertices_l2627_262748

/-- The sum of interior angles of faces in a convex polyhedron with given number of vertices -/
def sum_interior_angles (vertices : ℕ) : ℝ :=
  (vertices - 2) * 180

/-- Theorem: The sum of interior angles of faces in a convex polyhedron with 30 vertices is 5040° -/
theorem sum_interior_angles_30_vertices :
  sum_interior_angles 30 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_30_vertices_l2627_262748


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2627_262750

theorem inequality_system_solution (p : ℝ) : 19 * p < 10 ∧ p > (1/2 : ℝ) → (1/2 : ℝ) < p ∧ p < 10/19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2627_262750


namespace NUMINAMATH_CALUDE_total_boxes_theorem_l2627_262765

/-- Calculates the total number of boxes sold over four days given specific sales conditions --/
def total_boxes_sold (thursday_boxes : ℕ) : ℕ :=
  let friday_boxes : ℕ := thursday_boxes + (thursday_boxes * 50) / 100
  let saturday_boxes : ℕ := friday_boxes + (friday_boxes * 80) / 100
  let sunday_boxes : ℕ := saturday_boxes - (saturday_boxes * 30) / 100
  thursday_boxes + friday_boxes + saturday_boxes + sunday_boxes

/-- Theorem stating that given the specific sales conditions, the total number of boxes sold is 425 --/
theorem total_boxes_theorem : total_boxes_sold 60 = 425 := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_theorem_l2627_262765


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l2627_262757

theorem consecutive_integers_problem (x y z : ℤ) : 
  (y = x - 1) → (z = y - 1) → (x > y) → (y > z) → 
  (2 * x + 3 * y + 3 * z = 5 * y + 11) → (z = 3) → 
  (2 * x = 10) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l2627_262757


namespace NUMINAMATH_CALUDE_quadrangle_area_inequality_quadrangle_area_equality_quadrangle_area_equality_converse_l2627_262772

-- Define a quadrangle
structure Quadrangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ

-- State the theorem
theorem quadrangle_area_inequality (q : Quadrangle) :
  q.area ≤ (1/2) * (q.a * q.c + q.b * q.d) := by sorry

-- Define a convex orthodiagonal cyclic quadrilateral
structure ConvexOrthoDiagonalCyclicQuad extends Quadrangle where
  is_convex : Bool
  is_orthodiagonal : Bool
  is_cyclic : Bool

-- State the equality condition
theorem quadrangle_area_equality (q : ConvexOrthoDiagonalCyclicQuad) :
  q.is_convex = true → q.is_orthodiagonal = true → q.is_cyclic = true →
  q.area = (1/2) * (q.a * q.c + q.b * q.d) := by sorry

-- State the converse of the equality condition
theorem quadrangle_area_equality_converse (q : Quadrangle) :
  q.area = (1/2) * (q.a * q.c + q.b * q.d) →
  ∃ (cq : ConvexOrthoDiagonalCyclicQuad),
    cq.a = q.a ∧ cq.b = q.b ∧ cq.c = q.c ∧ cq.d = q.d ∧
    cq.area = q.area ∧
    cq.is_convex = true ∧ cq.is_orthodiagonal = true ∧ cq.is_cyclic = true := by sorry

end NUMINAMATH_CALUDE_quadrangle_area_inequality_quadrangle_area_equality_quadrangle_area_equality_converse_l2627_262772
