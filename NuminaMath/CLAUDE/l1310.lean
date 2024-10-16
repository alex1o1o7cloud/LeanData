import Mathlib

namespace NUMINAMATH_CALUDE_road_trip_driving_hours_l1310_131076

/-- Proves that in a 3-day road trip where one person drives 6 hours each day
    and the total driving time is 42 hours, the other person drives 8 hours each day. -/
theorem road_trip_driving_hours (total_days : ℕ) (krista_hours_per_day : ℕ) (total_hours : ℕ)
    (h1 : total_days = 3)
    (h2 : krista_hours_per_day = 6)
    (h3 : total_hours = 42) :
    (total_hours - krista_hours_per_day * total_days) / total_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_driving_hours_l1310_131076


namespace NUMINAMATH_CALUDE_johnny_weekly_earnings_l1310_131028

/-- Represents Johnny's dog walking business --/
structure DogWalker where
  dogs_per_walk : ℕ
  pay_30min : ℕ
  pay_60min : ℕ
  hours_per_day : ℕ
  long_walks_per_day : ℕ
  work_days_per_week : ℕ

/-- Calculates Johnny's weekly earnings --/
def weekly_earnings (dw : DogWalker) : ℕ :=
  sorry

/-- Johnny's specific situation --/
def johnny : DogWalker := {
  dogs_per_walk := 3,
  pay_30min := 15,
  pay_60min := 20,
  hours_per_day := 4,
  long_walks_per_day := 6,
  work_days_per_week := 5
}

/-- Theorem stating Johnny's weekly earnings --/
theorem johnny_weekly_earnings : weekly_earnings johnny = 1500 :=
  sorry

end NUMINAMATH_CALUDE_johnny_weekly_earnings_l1310_131028


namespace NUMINAMATH_CALUDE_sum_of_composite_function_evaluations_l1310_131015

def p (x : ℝ) : ℝ := 2 * |x| - 4

def q (x : ℝ) : ℝ := -|x|

def evaluation_points : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composite_function_evaluations :
  (evaluation_points.map (λ x => q (p x))).sum = -20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_evaluations_l1310_131015


namespace NUMINAMATH_CALUDE_project_completion_proof_l1310_131056

/-- The number of days Person A takes to complete the project alone -/
def person_a_days : ℕ := 20

/-- The number of days Person B takes to complete the project alone -/
def person_b_days : ℕ := 10

/-- The total number of days taken to complete the project -/
def total_days : ℕ := 12

/-- The number of days Person B worked alone -/
def person_b_worked_days : ℕ := 8

theorem project_completion_proof :
  (1 : ℚ) = (total_days - person_b_worked_days : ℚ) / person_a_days + 
            (person_b_worked_days : ℚ) / person_b_days :=
by sorry

end NUMINAMATH_CALUDE_project_completion_proof_l1310_131056


namespace NUMINAMATH_CALUDE_gray_eyed_black_haired_count_l1310_131052

theorem gray_eyed_black_haired_count : ∀ (total red_haired black_haired green_eyed gray_eyed green_eyed_red_haired : ℕ),
  total = 60 →
  red_haired + black_haired = total →
  green_eyed + gray_eyed = total →
  green_eyed_red_haired = 20 →
  black_haired = 40 →
  gray_eyed = 25 →
  gray_eyed - (red_haired - green_eyed_red_haired) = 25 := by
  sorry

end NUMINAMATH_CALUDE_gray_eyed_black_haired_count_l1310_131052


namespace NUMINAMATH_CALUDE_max_revenue_l1310_131066

/-- The revenue function for the bookstore --/
def revenue (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- Theorem stating the maximum revenue and the price at which it occurs --/
theorem max_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  revenue p = 140.625 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → revenue q ≤ revenue p :=
by
  sorry

end NUMINAMATH_CALUDE_max_revenue_l1310_131066


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1310_131010

theorem complex_equation_solution (a b c : ℕ+) 
  (h : (a - b * Complex.I) ^ 2 + c = 13 - 8 * Complex.I) :
  a = 2 ∧ b = 2 ∧ c = 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1310_131010


namespace NUMINAMATH_CALUDE_deck_card_count_l1310_131001

theorem deck_card_count : ∀ (r n : ℕ), 
  (n = 2 * r) →                           -- Initially, black cards are twice red cards
  (n + 4 = 3 * r) →                       -- After adding 4 black cards, black is triple red
  (r + n = 12) :=                         -- Initial total number of cards is 12
by
  sorry

end NUMINAMATH_CALUDE_deck_card_count_l1310_131001


namespace NUMINAMATH_CALUDE_share_of_a_l1310_131088

def total : ℕ := 366

def shares (a b c : ℕ) : Prop :=
  a + b + c = total ∧
  a = (b + c) / 2 ∧
  b = (a + c) * 2 / 3

theorem share_of_a : ∃ a b c : ℕ, shares a b c ∧ a = 122 := by sorry

end NUMINAMATH_CALUDE_share_of_a_l1310_131088


namespace NUMINAMATH_CALUDE_circle_C_equation_range_of_a_symmetry_condition_l1310_131063

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the chord line
def chord_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the intersecting line
def intersecting_line (a x y : ℝ) : Prop := a * x - y + 5 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, 4)

-- Theorem 1: Prove that the equation represents circle C
theorem circle_C_equation :
  ∃ (m : ℝ), m > 0 ∧
  (∀ (x y : ℝ), circle_C x y ↔ (x - m)^2 + y^2 = 25) ∧
  (∃ (x y : ℝ), chord_line x y ∧ circle_C x y ∧
    ∃ (x' y' : ℝ), chord_line x' y' ∧ circle_C x' y' ∧
    (x - x')^2 + (y - y')^2 = 4 * 17) :=
sorry

-- Theorem 2: Prove the range of a
theorem range_of_a :
  ∀ (a : ℝ), (∃ (x y : ℝ), intersecting_line a x y ∧ circle_C x y) ↔
  (a < 0 ∨ a > 5/12) :=
sorry

-- Theorem 3: Prove the symmetry condition
theorem symmetry_condition :
  ∃ (a : ℝ), a = 3/4 ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    intersecting_line a x₁ y₁ ∧ circle_C x₁ y₁ ∧
    intersecting_line a x₂ y₂ ∧ circle_C x₂ y₂ ∧
    x₁ ≠ x₂ →
    (x₁ + x₂) * (point_P.1 + 2) + (y₁ + y₂) * (point_P.2 - 4) = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_range_of_a_symmetry_condition_l1310_131063


namespace NUMINAMATH_CALUDE_correct_transformation_l1310_131008

theorem correct_transformation (x : ℝ) : (2/3 * x - 1 = x) ↔ (2*x - 3 = 3*x) := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l1310_131008


namespace NUMINAMATH_CALUDE_binomial_coefficient_22_5_l1310_131041

theorem binomial_coefficient_22_5 
  (h1 : Nat.choose 20 3 = 1140)
  (h2 : Nat.choose 20 4 = 4845)
  (h3 : Nat.choose 20 5 = 15504) : 
  Nat.choose 22 5 = 26334 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_22_5_l1310_131041


namespace NUMINAMATH_CALUDE_rabbit_log_sawing_l1310_131068

theorem rabbit_log_sawing (cuts pieces : ℕ) (h1 : cuts = 10) (h2 : pieces = 16) :
  pieces - cuts = 6 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_log_sawing_l1310_131068


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1310_131086

theorem fraction_multiplication :
  (2 : ℚ) / 3 * (5 : ℚ) / 7 * (11 : ℚ) / 13 = (110 : ℚ) / 273 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1310_131086


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l1310_131069

/-- The volume difference between a sphere and an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere : ℝ) (r_cylinder : ℝ) 
  (h_sphere : r_sphere = 7)
  (h_cylinder : r_cylinder = 4) :
  (4 / 3 * π * r_sphere^3) - (π * r_cylinder^2 * Real.sqrt (4 * r_sphere^2 - 4 * r_cylinder^2)) = 
  1372 * π / 3 - 32 * π * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l1310_131069


namespace NUMINAMATH_CALUDE_l_structure_surface_area_l1310_131012

/-- Represents the L-shaped structure composed of unit cubes -/
structure LStructure where
  bottom_row : Nat
  first_stack : Nat
  second_stack : Nat

/-- Calculates the surface area of the L-shaped structure -/
def surface_area (l : LStructure) : Nat :=
  let bottom_area := 2 * l.bottom_row + 2
  let first_stack_area := 1 + 1 + 3 + 3 + 2
  let second_stack_area := 1 + 5 + 5 + 2
  bottom_area + first_stack_area + second_stack_area

/-- Theorem stating that the surface area of the specific L-shaped structure is 39 square units -/
theorem l_structure_surface_area :
  surface_area { bottom_row := 7, first_stack := 3, second_stack := 5 } = 39 := by
  sorry

#eval surface_area { bottom_row := 7, first_stack := 3, second_stack := 5 }

end NUMINAMATH_CALUDE_l_structure_surface_area_l1310_131012


namespace NUMINAMATH_CALUDE_three_disjoint_edges_exist_l1310_131016

/-- A graph with 6 vertices where each vertex has degree at least 3 -/
structure SixVertexGraph where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  vertex_count : vertices.card = 6
  edge_symmetry : ∀ (u v : Fin 6), (u, v) ∈ edges → (v, u) ∈ edges
  min_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 3

/-- A set of 3 disjoint edges that cover all vertices -/
def ThreeDisjointEdges (G : SixVertexGraph) : Prop :=
  ∃ (e₁ e₂ e₃ : Fin 6 × Fin 6),
    e₁ ∈ G.edges ∧ e₂ ∈ G.edges ∧ e₃ ∈ G.edges ∧
    e₁.1 ≠ e₁.2 ∧ e₂.1 ≠ e₂.2 ∧ e₃.1 ≠ e₃.2 ∧
    e₁.1 ≠ e₂.1 ∧ e₁.1 ≠ e₂.2 ∧ e₁.1 ≠ e₃.1 ∧ e₁.1 ≠ e₃.2 ∧
    e₁.2 ≠ e₂.1 ∧ e₁.2 ≠ e₂.2 ∧ e₁.2 ≠ e₃.1 ∧ e₁.2 ≠ e₃.2 ∧
    e₂.1 ≠ e₃.1 ∧ e₂.1 ≠ e₃.2 ∧ e₂.2 ≠ e₃.1 ∧ e₂.2 ≠ e₃.2

/-- Theorem: In a graph with 6 vertices where each vertex has degree at least 3,
    there exists a set of 3 disjoint edges that cover all vertices -/
theorem three_disjoint_edges_exist (G : SixVertexGraph) : ThreeDisjointEdges G :=
sorry

end NUMINAMATH_CALUDE_three_disjoint_edges_exist_l1310_131016


namespace NUMINAMATH_CALUDE_meeting_arrangements_count_l1310_131023

/-- Represents the number of schools -/
def num_schools : ℕ := 4

/-- Represents the number of members per school -/
def members_per_school : ℕ := 6

/-- Represents the total number of members -/
def total_members : ℕ := num_schools * members_per_school

/-- Represents the number of representatives sent by the host school -/
def host_representatives : ℕ := 1

/-- Represents the number of schools (excluding the host) that send representatives -/
def schools_sending_representatives : ℕ := 2

/-- Represents the number of representatives sent by each non-host school -/
def representatives_per_school : ℕ := 2

/-- Theorem stating the number of ways to arrange the meeting -/
theorem meeting_arrangements_count : 
  (num_schools) * (members_per_school.choose host_representatives) * 
  ((num_schools - 1).choose schools_sending_representatives) * 
  (members_per_school.choose representatives_per_school)^schools_sending_representatives = 16200 := by
  sorry

end NUMINAMATH_CALUDE_meeting_arrangements_count_l1310_131023


namespace NUMINAMATH_CALUDE_stratified_sampling_group_b_l1310_131093

-- Define the total number of cities and the number in Group B
def total_cities : ℕ := 48
def group_b_cities : ℕ := 18

-- Define the total sample size
def sample_size : ℕ := 16

-- Define the function to calculate the number of cities to sample from Group B
def cities_to_sample_from_b : ℕ := 
  (group_b_cities * sample_size) / total_cities

-- Theorem statement
theorem stratified_sampling_group_b :
  cities_to_sample_from_b = 6 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_b_l1310_131093


namespace NUMINAMATH_CALUDE_simplify_expression_l1310_131091

theorem simplify_expression (a b : ℝ) :
  (33 * a + 75 * b + 12) + (15 * a + 44 * b + 7) - (12 * a + 65 * b + 5) = 36 * a + 54 * b + 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1310_131091


namespace NUMINAMATH_CALUDE_train_crossing_time_l1310_131051

/-- The time taken for a train to cross a man moving in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 210 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 28 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1310_131051


namespace NUMINAMATH_CALUDE_stating_club_truncator_probability_l1310_131073

/-- The number of matches played by Club Truncator -/
def num_matches : ℕ := 8

/-- The probability of winning, losing, or tying a single match -/
def single_match_prob : ℚ := 1/3

/-- The probability of finishing with more wins than losses -/
def more_wins_prob : ℚ := 2741/6561

/-- 
Theorem stating that given 8 matches where the probability of winning, 
losing, or tying each match is 1/3, the probability of finishing with 
more wins than losses is 2741/6561.
-/
theorem club_truncator_probability : 
  (num_matches = 8) → 
  (single_match_prob = 1/3) → 
  (more_wins_prob = 2741/6561) :=
by sorry

end NUMINAMATH_CALUDE_stating_club_truncator_probability_l1310_131073


namespace NUMINAMATH_CALUDE_carbon_emissions_solution_l1310_131089

theorem carbon_emissions_solution :
  ∃! (x y : ℝ), x + y = 70 ∧ x = 5 * y - 8 ∧ x = 57 ∧ y = 13 := by
  sorry

end NUMINAMATH_CALUDE_carbon_emissions_solution_l1310_131089


namespace NUMINAMATH_CALUDE_exists_valid_nail_configuration_l1310_131027

/-- Represents a nail configuration for hanging a painting -/
structure NailConfiguration where
  nails : Fin 4 → Unit

/-- Represents the state of the painting (hanging or fallen) -/
inductive PaintingState
  | Hanging
  | Fallen

/-- Determines the state of the painting given a nail configuration and a set of removed nails -/
def paintingState (config : NailConfiguration) (removed : Set (Fin 4)) : PaintingState :=
  sorry

/-- Theorem stating the existence of a nail configuration satisfying the given conditions -/
theorem exists_valid_nail_configuration :
  ∃ (config : NailConfiguration),
    (∀ (i : Fin 4), paintingState config {i} = PaintingState.Hanging) ∧
    (∀ (i j : Fin 4), i ≠ j → paintingState config {i, j} = PaintingState.Fallen) :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_nail_configuration_l1310_131027


namespace NUMINAMATH_CALUDE_investment_average_rate_l1310_131009

theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) 
  (h1 : total = 6000)
  (h2 : rate1 = 0.035)
  (h3 : rate2 = 0.055)
  (h4 : ∃ x : ℝ, x > 0 ∧ x < total ∧ rate1 * (total - x) = rate2 * x) :
  ∃ avg_rate : ℝ, abs (avg_rate - 0.043) < 0.0001 ∧ 
  avg_rate * total = rate1 * (total - x) + rate2 * x :=
sorry

end NUMINAMATH_CALUDE_investment_average_rate_l1310_131009


namespace NUMINAMATH_CALUDE_half_reporters_not_cover_politics_l1310_131045

/-- Represents the percentage of reporters covering local politics in country X -/
def local_politics_coverage : ℝ := 35

/-- Represents the percentage of political reporters not covering local politics in country X -/
def non_local_political_coverage : ℝ := 30

/-- Theorem stating that 50% of reporters do not cover politics -/
theorem half_reporters_not_cover_politics : 
  local_politics_coverage = 35 ∧ 
  non_local_political_coverage = 30 → 
  (100 : ℝ) - (local_politics_coverage / ((100 : ℝ) - non_local_political_coverage) * 100) = 50 := by
  sorry

end NUMINAMATH_CALUDE_half_reporters_not_cover_politics_l1310_131045


namespace NUMINAMATH_CALUDE_conditional_probability_A_given_B_l1310_131075

def group_A : List Nat := [76, 90, 84, 86, 81, 87, 86, 82, 85, 83]
def group_B : List Nat := [82, 84, 85, 89, 79, 80, 91, 89, 79, 74]

def total_students : Nat := group_A.length + group_B.length

def students_A_above_85 : Nat := (group_A.filter (λ x => x ≥ 85)).length
def students_B_above_85 : Nat := (group_B.filter (λ x => x ≥ 85)).length
def total_above_85 : Nat := students_A_above_85 + students_B_above_85

def P_B : Rat := total_above_85 / total_students
def P_AB : Rat := students_A_above_85 / total_students

theorem conditional_probability_A_given_B :
  P_AB / P_B = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_A_given_B_l1310_131075


namespace NUMINAMATH_CALUDE_abs_S_equals_512_l1310_131038

-- Define the complex number i
def i : ℂ := Complex.I

-- Define S
def S : ℂ := (1 + i)^17 - (1 - i)^17

-- Theorem statement
theorem abs_S_equals_512 : Complex.abs S = 512 := by sorry

end NUMINAMATH_CALUDE_abs_S_equals_512_l1310_131038


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1310_131072

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℚ  -- Sum function
  is_arithmetic : ∀ n : ℕ, S (n + 1) - S n = S (n + 2) - S (n + 1)

/-- Theorem: If S_2 / S_4 = 1/3, then S_4 / S_8 = 3/10 for an arithmetic sequence -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) 
  (h : seq.S 2 / seq.S 4 = 1 / 3) : 
  seq.S 4 / seq.S 8 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1310_131072


namespace NUMINAMATH_CALUDE_f_at_negative_two_l1310_131046

def f (x : ℝ) : ℝ := 2*x^5 + 5*x^4 + 5*x^3 + 10*x^2 + 6*x + 1

theorem f_at_negative_two : f (-2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_two_l1310_131046


namespace NUMINAMATH_CALUDE_fraction_arithmetic_l1310_131092

theorem fraction_arithmetic : (2 : ℚ) / 9 * 4 / 5 - 1 / 45 = 7 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_arithmetic_l1310_131092


namespace NUMINAMATH_CALUDE_shirt_selection_theorem_l1310_131090

/-- Represents the number of shirts of each color in the drawer -/
structure ShirtDrawer :=
  (blue : ℕ)
  (gray : ℕ)
  (red : ℕ)

/-- The total number of shirts in the drawer -/
def total_shirts (d : ShirtDrawer) : ℕ := d.blue + d.gray + d.red

/-- The number of different colors of shirts -/
def num_colors (d : ShirtDrawer) : ℕ := 3

/-- The minimum number of shirts that must be selected to ensure a certain number of the same color -/
def min_shirts_same_color (d : ShirtDrawer) (n : ℕ) : ℕ := 
  (n - 1) * num_colors d + 1

/-- The given drawer configuration -/
def drawer : ShirtDrawer := ⟨4, 7, 9⟩

theorem shirt_selection_theorem :
  (total_shirts drawer = 20) →
  (min_shirts_same_color drawer 4 = 10) ∧
  (min_shirts_same_color drawer 5 = 13) ∧
  (min_shirts_same_color drawer 6 = 16) ∧
  (min_shirts_same_color drawer 7 = 17) ∧
  (min_shirts_same_color drawer 8 = 19) ∧
  (min_shirts_same_color drawer 9 = 20) := by
  sorry

end NUMINAMATH_CALUDE_shirt_selection_theorem_l1310_131090


namespace NUMINAMATH_CALUDE_max_area_rectangular_frame_l1310_131040

/-- Represents the maximum area of a rectangular frame given budget constraints. -/
theorem max_area_rectangular_frame :
  ∃ (L W : ℕ),
    (3 * L + 5 * W ≤ 100) ∧
    (∀ (L' W' : ℕ), (3 * L' + 5 * W' ≤ 100) → L * W ≥ L' * W') ∧
    L * W = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_frame_l1310_131040


namespace NUMINAMATH_CALUDE_fraction_value_l1310_131032

/-- Given that x is four times y, y is three times z, and z is five times w,
    prove that (x * z) / (y * w) = 20 -/
theorem fraction_value (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) : 
  (x * z) / (y * w) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1310_131032


namespace NUMINAMATH_CALUDE_min_value_f_neg_three_range_of_a_l1310_131071

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem 1: Minimum value of f when a = -3
theorem min_value_f_neg_three :
  ∃ (m : ℝ), m = 4 ∧ ∀ (x : ℝ), f (-3) x ≥ m :=
sorry

-- Theorem 2: Range of a when f(x) ≤ 2a + 2|x-1| for all x
theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), f a x ≤ 2 * a + 2 * |x - 1|) → a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_neg_three_range_of_a_l1310_131071


namespace NUMINAMATH_CALUDE_six_by_six_grid_squares_l1310_131047

/-- The number of squares of size n×n in a grid of size m×m -/
def count_squares (n m : ℕ) : ℕ := (m - n + 1) * (m - n + 1)

/-- The total number of squares in a 6×6 grid -/
def total_squares : ℕ :=
  count_squares 1 6 + count_squares 2 6 + count_squares 3 6 + count_squares 4 6

theorem six_by_six_grid_squares :
  total_squares = 86 :=
sorry

end NUMINAMATH_CALUDE_six_by_six_grid_squares_l1310_131047


namespace NUMINAMATH_CALUDE_variety_promotion_criterion_variety_B_more_suitable_l1310_131098

/-- Represents a rice variety with its yield statistics -/
structure RiceVariety where
  mean_yield : ℝ
  variance : ℝ

/-- Determines if a rice variety is more suitable for promotion based on yield stability -/
def more_suitable_for_promotion (a b : RiceVariety) : Prop :=
  a.mean_yield = b.mean_yield ∧ a.variance < b.variance

/-- Theorem stating that given two varieties with equal mean yields, 
    the one with lower variance is more suitable for promotion -/
theorem variety_promotion_criterion 
  (a b : RiceVariety) 
  (h_equal_means : a.mean_yield = b.mean_yield) 
  (h_lower_variance : a.variance < b.variance) : 
  more_suitable_for_promotion b a := by
  sorry

/-- The specific rice varieties from the problem -/
def variety_A : RiceVariety := ⟨1042, 6.5⟩
def variety_B : RiceVariety := ⟨1042, 1.2⟩

/-- Theorem applying the general criterion to the specific varieties -/
theorem variety_B_more_suitable : 
  more_suitable_for_promotion variety_B variety_A := by
  sorry

end NUMINAMATH_CALUDE_variety_promotion_criterion_variety_B_more_suitable_l1310_131098


namespace NUMINAMATH_CALUDE_coin_flip_problem_l1310_131025

theorem coin_flip_problem (n : ℕ) : 
  (1 + n : ℚ) / 2^n = 3/16 ↔ n = 5 := by sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l1310_131025


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_l1310_131064

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_l1310_131064


namespace NUMINAMATH_CALUDE_polynomial_simplification_and_evaluation_l1310_131029

theorem polynomial_simplification_and_evaluation (a b : ℝ) :
  (-3 * a * (a - 2 * b)^5 + 6 * b * (a - 2 * b)^5 - 5 * (-a + 2 * b)^3 = -3 * (a - 2 * b)^6 + 5 * (a - 2 * b)^3) ∧
  (a - 2 * b = -1 → -3 * (a - 2 * b)^6 + 5 * (a - 2 * b)^3 = -8) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_and_evaluation_l1310_131029


namespace NUMINAMATH_CALUDE_greatest_common_divisor_546_126_under_30_l1310_131061

theorem greatest_common_divisor_546_126_under_30 : 
  ∃ (n : ℕ), n = 21 ∧ 
  n ∣ 546 ∧ 
  n < 30 ∧ 
  n ∣ 126 ∧
  ∀ (m : ℕ), m ∣ 546 → m < 30 → m ∣ 126 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_546_126_under_30_l1310_131061


namespace NUMINAMATH_CALUDE_carnival_tickets_l1310_131087

def ticket_distribution (n : Nat) : Nat :=
  let ratio := [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
  let total_parts := ratio.sum
  let tickets_per_part := n / total_parts
  tickets_per_part * total_parts

theorem carnival_tickets :
  let friends : Nat := 17
  let initial_tickets : Nat := 865
  let ratio := [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
  let total_parts := ratio.sum
  let next_multiple := (initial_tickets / total_parts + 1) * total_parts
  next_multiple - initial_tickets = 26 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_l1310_131087


namespace NUMINAMATH_CALUDE_donnys_remaining_money_l1310_131065

/-- Calculates the remaining money after purchases -/
def remaining_money (initial : ℕ) (kite_cost : ℕ) (frisbee_cost : ℕ) : ℕ :=
  initial - (kite_cost + frisbee_cost)

/-- Theorem: Donny's remaining money after purchases -/
theorem donnys_remaining_money :
  remaining_money 78 8 9 = 61 := by
  sorry

end NUMINAMATH_CALUDE_donnys_remaining_money_l1310_131065


namespace NUMINAMATH_CALUDE_line_and_circle_properties_l1310_131067

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - 7*y + 8 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := y = -7/2*x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x-3)^2 + (y-2)^2 = 13

-- Define points A and B
def point_A : ℝ × ℝ := (6, 0)
def point_B : ℝ × ℝ := (1, 5)

-- Theorem statement
theorem line_and_circle_properties :
  -- Line l passes through (3,2)
  line_l 3 2 ∧
  -- Line l is perpendicular to y = -7/2x + 1
  (∀ x y : ℝ, line_l x y → perp_line x y → x = y) ∧
  -- The center of circle C lies on line l
  (∃ x y : ℝ, line_l x y ∧ circle_C x y) ∧
  -- Circle C passes through points A and B
  circle_C point_A.1 point_A.2 ∧ circle_C point_B.1 point_B.2 →
  -- Conclusion 1: The equation of line l is 2x - 7y + 8 = 0
  (∀ x y : ℝ, line_l x y ↔ 2*x - 7*y + 8 = 0) ∧
  -- Conclusion 2: The standard equation of circle C is (x-3)^2 + (y-2)^2 = 13
  (∀ x y : ℝ, circle_C x y ↔ (x-3)^2 + (y-2)^2 = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_line_and_circle_properties_l1310_131067


namespace NUMINAMATH_CALUDE_electric_guitar_price_l1310_131096

theorem electric_guitar_price (total_guitars : ℕ) (total_revenue : ℕ) 
  (acoustic_price : ℕ) (electric_count : ℕ) : 
  total_guitars = 9 → 
  total_revenue = 3611 → 
  acoustic_price = 339 → 
  electric_count = 4 → 
  (total_revenue - (total_guitars - electric_count) * acoustic_price) / electric_count = 479 :=
by sorry

end NUMINAMATH_CALUDE_electric_guitar_price_l1310_131096


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1310_131018

/-- A rectangular field with given area and width has a specific perimeter -/
theorem rectangle_perimeter (area width : ℝ) (h_area : area = 750) (h_width : width = 25) :
  2 * (area / width + width) = 110 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1310_131018


namespace NUMINAMATH_CALUDE_minimal_cost_proof_l1310_131037

/-- Represents an entity that can clean -/
inductive Cleaner
| Janitor
| Student
| Company

/-- Represents a location to be cleaned -/
inductive Location
| Classes
| Gym

/-- Time (in hours) it takes for a cleaner to clean a location -/
def cleaning_time (c : Cleaner) (l : Location) : ℕ :=
  match c, l with
  | Cleaner.Janitor, Location.Classes => 8
  | Cleaner.Janitor, Location.Gym => 6
  | Cleaner.Student, Location.Classes => 20
  | Cleaner.Student, Location.Gym => 0  -- Student cannot clean the gym
  | Cleaner.Company, Location.Classes => 10
  | Cleaner.Company, Location.Gym => 5

/-- Hourly rate (in dollars) for each cleaner -/
def hourly_rate (c : Cleaner) : ℕ :=
  match c with
  | Cleaner.Janitor => 21
  | Cleaner.Student => 7
  | Cleaner.Company => 60

/-- Cost for a cleaner to clean a location -/
def cleaning_cost (c : Cleaner) (l : Location) : ℕ :=
  (cleaning_time c l) * (hourly_rate c)

/-- The minimal cost to clean both the classes and the gym -/
def minimal_cleaning_cost : ℕ := 266

theorem minimal_cost_proof :
  ∀ (c1 c2 : Cleaner) (l1 l2 : Location),
    l1 ≠ l2 →
    cleaning_cost c1 l1 + cleaning_cost c2 l2 ≥ minimal_cleaning_cost :=
by sorry

end NUMINAMATH_CALUDE_minimal_cost_proof_l1310_131037


namespace NUMINAMATH_CALUDE_unique_n_for_total_digits_l1310_131053

/-- Sum of digits function for a natural number -/
def sumOfDigits (k : ℕ) : ℕ := sorry

/-- Total sum of digits for all numbers from 1 to n -/
def totalSumOfDigits (n : ℕ) : ℕ := 
  (Finset.range n).sum (fun i => sumOfDigits (i + 1))

/-- The theorem statement -/
theorem unique_n_for_total_digits : 
  ∃! n : ℕ, totalSumOfDigits n = 777 := by sorry

end NUMINAMATH_CALUDE_unique_n_for_total_digits_l1310_131053


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1310_131081

theorem sqrt_equation_solution (z : ℝ) : 
  (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt z) / (Real.sqrt 0.49) = 2.9365079365079367 → 
  z = 1.44 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1310_131081


namespace NUMINAMATH_CALUDE_solve_for_c_l1310_131060

theorem solve_for_c : ∃ C : ℝ, (4 * C + 5 = 25) ∧ (C = 5) := by sorry

end NUMINAMATH_CALUDE_solve_for_c_l1310_131060


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l1310_131099

theorem trigonometric_expression_evaluation : 3 * Real.cos 0 + 4 * Real.sin (3 * Real.pi / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l1310_131099


namespace NUMINAMATH_CALUDE_oats_per_meal_is_four_l1310_131095

/-- The amount of oats each horse eats per meal, in pounds -/
def oats_per_meal (num_horses : ℕ) (grain_per_horse : ℕ) (total_food : ℕ) (num_days : ℕ) : ℚ :=
  let total_food_per_day := total_food / num_days
  let grain_per_day := num_horses * grain_per_horse
  let oats_per_day := total_food_per_day - grain_per_day
  oats_per_day / (2 * num_horses)

theorem oats_per_meal_is_four :
  oats_per_meal 4 3 132 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oats_per_meal_is_four_l1310_131095


namespace NUMINAMATH_CALUDE_common_chord_equation_l1310_131048

/-- The equation of the line containing the common chord of two circles -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 4 = 0) →
  (x^2 + y^2 - 4*x + 4*y - 12 = 0) →
  (x - y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l1310_131048


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1310_131085

theorem real_part_of_complex_fraction :
  let i : ℂ := Complex.I
  (2 * i / (1 + i)).re = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1310_131085


namespace NUMINAMATH_CALUDE_cake_segment_length_squared_l1310_131030

theorem cake_segment_length_squared (d : ℝ) (n : ℕ) (m : ℝ) : 
  d = 20 → n = 4 → m = (d / 2) * Real.sqrt 2 → m^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_cake_segment_length_squared_l1310_131030


namespace NUMINAMATH_CALUDE_polygon_exterior_angle_72_l1310_131050

theorem polygon_exterior_angle_72 (n : ℕ) (exterior_angle : ℝ) :
  exterior_angle = 72 →
  (360 : ℝ) / exterior_angle = n →
  n = 5 ∧ (n - 2) * 180 = 540 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angle_72_l1310_131050


namespace NUMINAMATH_CALUDE_birds_on_fence_l1310_131019

theorem birds_on_fence (initial_birds : ℕ) (initial_storks : ℕ) (additional_birds : ℕ) : 
  initial_birds = 2 →
  initial_storks = 6 →
  initial_storks = (initial_birds + additional_birds + 1) →
  additional_birds = 3 := by
sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1310_131019


namespace NUMINAMATH_CALUDE_carrots_picked_first_day_l1310_131082

theorem carrots_picked_first_day (carrots_thrown_out carrots_second_day total_carrots : ℕ) 
  (h1 : carrots_thrown_out = 4)
  (h2 : carrots_second_day = 46)
  (h3 : total_carrots = 61) :
  ∃ carrots_first_day : ℕ, 
    carrots_first_day + carrots_second_day - carrots_thrown_out = total_carrots ∧ 
    carrots_first_day = 19 := by
  sorry

end NUMINAMATH_CALUDE_carrots_picked_first_day_l1310_131082


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1310_131004

/-- An isosceles triangle with two sides of length 8 cm and perimeter 26 cm has a base of length 10 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side : ℝ),
  congruent_side = 8 →
  base + 2 * congruent_side = 26 →
  base = 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1310_131004


namespace NUMINAMATH_CALUDE_sum_denominator_divisible_by_prime_l1310_131011

theorem sum_denominator_divisible_by_prime (p : ℕ) (n : ℕ) (b : Fin n → ℕ) :
  Prime p →
  (∃! i : Fin n, p ∣ b i) →
  (∀ i : Fin n, 0 < b i) →
  ∃ (num den : ℕ), 
    (0 < den) ∧
    (Nat.gcd num den = 1) ∧
    (p ∣ den) ∧
    (Finset.sum Finset.univ (λ i => 1 / (b i : ℚ)) = num / den) :=
by sorry

end NUMINAMATH_CALUDE_sum_denominator_divisible_by_prime_l1310_131011


namespace NUMINAMATH_CALUDE_tuesday_bags_count_l1310_131036

/-- The number of bags of leaves raked on Tuesday -/
def bags_on_tuesday (price_per_bag : ℕ) (bags_monday : ℕ) (bags_other_day : ℕ) (total_money : ℕ) : ℕ :=
  (total_money - price_per_bag * (bags_monday + bags_other_day)) / price_per_bag

/-- Theorem stating that given the conditions, the number of bags raked on Tuesday is 3 -/
theorem tuesday_bags_count :
  bags_on_tuesday 4 5 9 68 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_bags_count_l1310_131036


namespace NUMINAMATH_CALUDE_point_placement_l1310_131083

theorem point_placement (x : ℕ) : 
  (9 * x - 8 = 82) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_point_placement_l1310_131083


namespace NUMINAMATH_CALUDE_habitable_land_area_l1310_131002

/-- Calculates the area of habitable land in a rectangular field with a circular pond. -/
theorem habitable_land_area (length width diagonal pond_radius : ℝ) 
  (h_length : length = 23)
  (h_diagonal : diagonal = 33)
  (h_width : width^2 = diagonal^2 - length^2)
  (h_pond_radius : pond_radius = 3) : 
  ∃ (area : ℝ), abs (area - 515.91) < 0.01 ∧ 
  area = length * width - π * pond_radius^2 := by
sorry

end NUMINAMATH_CALUDE_habitable_land_area_l1310_131002


namespace NUMINAMATH_CALUDE_tina_sold_26_more_than_katya_l1310_131080

/-- The number of glasses of lemonade sold by Katya -/
def katya_sales : ℕ := 8

/-- The number of glasses of lemonade sold by Ricky -/
def ricky_sales : ℕ := 9

/-- The number of glasses of lemonade sold by Tina -/
def tina_sales : ℕ := 2 * (katya_sales + ricky_sales)

/-- Theorem: Tina sold 26 more glasses of lemonade than Katya -/
theorem tina_sold_26_more_than_katya : tina_sales - katya_sales = 26 := by
  sorry

end NUMINAMATH_CALUDE_tina_sold_26_more_than_katya_l1310_131080


namespace NUMINAMATH_CALUDE_line_parameterization_l1310_131055

/-- Given a line y = 2x - 40 parameterized by (x, y) = (g(t), 20t - 14),
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t x y : ℝ, y = 2*x - 40 ∧ x = g t ∧ y = 20*t - 14) →
  (∀ t : ℝ, g t = 10*t + 13) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l1310_131055


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l1310_131058

def f (x : ℤ) : ℤ := x^3 - 3*x^2 - 13*x + 15

theorem integer_roots_of_cubic :
  {x : ℤ | f x = 0} = {-3, 1, 5} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l1310_131058


namespace NUMINAMATH_CALUDE_book_purchase_equation_l1310_131084

/-- Represents a book purchase with total cost and quantity -/
structure BookPurchase where
  cost : ℝ
  quantity : ℝ

/-- Represents two book purchases with a quantity difference -/
structure TwoBookPurchases where
  first : BookPurchase
  second : BookPurchase
  quantity_difference : ℝ

/-- The equation correctly represents the situation of two book purchases with equal price per set -/
theorem book_purchase_equation (purchases : TwoBookPurchases) 
    (h1 : purchases.first.cost = 500)
    (h2 : purchases.second.cost = 700)
    (h3 : purchases.quantity_difference = 4)
    (h4 : purchases.first.quantity > 0)
    (h5 : purchases.second.quantity = purchases.first.quantity + purchases.quantity_difference) :
  purchases.first.cost / purchases.first.quantity = 
  purchases.second.cost / purchases.second.quantity :=
by sorry

end NUMINAMATH_CALUDE_book_purchase_equation_l1310_131084


namespace NUMINAMATH_CALUDE_simplify_radical_l1310_131000

theorem simplify_radical (a b : ℝ) (h : b > 0) :
  Real.sqrt (20 * a^3 * b^2) = 2 * a * b * Real.sqrt (5 * a) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_l1310_131000


namespace NUMINAMATH_CALUDE_ninth_observation_l1310_131054

theorem ninth_observation (n : ℕ) (original_avg new_avg : ℚ) (decrease : ℚ) :
  n = 8 →
  original_avg = 15 →
  decrease = 2 →
  new_avg = original_avg - decrease →
  (n * original_avg + (n + 1) * new_avg) / (2 * n + 1) - original_avg = -3 :=
by sorry

end NUMINAMATH_CALUDE_ninth_observation_l1310_131054


namespace NUMINAMATH_CALUDE_dog_pickup_duration_l1310_131079

-- Define the time in minutes for each activity
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_cleaning_time : ℕ := 10
def cooking_time : ℕ := 90

-- Define the total time from work end to dinner
def total_time : ℕ := 180

-- Define the time to pick up the dog (unknown)
def dog_pickup_time : ℕ := total_time - (commute_time + grocery_time + dry_cleaning_time + cooking_time)

-- Theorem to prove
theorem dog_pickup_duration : dog_pickup_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_dog_pickup_duration_l1310_131079


namespace NUMINAMATH_CALUDE_x_is_perfect_square_l1310_131074

theorem x_is_perfect_square (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ n : ℕ+, x = n^2 := by
sorry

end NUMINAMATH_CALUDE_x_is_perfect_square_l1310_131074


namespace NUMINAMATH_CALUDE_first_investment_interest_rate_l1310_131022

/-- Prove that the annual simple interest rate of the first investment is 8.5% --/
theorem first_investment_interest_rate 
  (total_income : ℝ) 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_rate : ℝ) 
  (h1 : total_income = 575)
  (h2 : total_investment = 8000)
  (h3 : first_investment = 3000)
  (h4 : second_rate = 0.064)
  (h5 : total_income = first_investment * x + (total_investment - first_investment) * second_rate) :
  x = 0.085 := by
sorry

end NUMINAMATH_CALUDE_first_investment_interest_rate_l1310_131022


namespace NUMINAMATH_CALUDE_movie_start_time_l1310_131031

-- Define the movie duration in minutes
def movie_duration : ℕ := 3 * 60

-- Define the remaining time in minutes
def remaining_time : ℕ := 36

-- Define the end time (5:44 pm) in minutes since midnight
def end_time : ℕ := 17 * 60 + 44

-- Define the start time (to be proven) in minutes since midnight
def start_time : ℕ := 15 * 60 + 20

-- Theorem statement
theorem movie_start_time :
  movie_duration - remaining_time = end_time - start_time :=
by sorry

end NUMINAMATH_CALUDE_movie_start_time_l1310_131031


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_inclination_l1310_131014

/-- Given a hyperbola mx^2 - y^2 = m where m > 0, if one of its asymptotes has an angle of inclination
    that is twice the angle of inclination of the line x - √3y = 0, then m = 3. -/
theorem hyperbola_asymptote_inclination (m : ℝ) (h1 : m > 0) : 
  (∃ θ : ℝ, θ = 2 * Real.arctan (1 / Real.sqrt 3) ∧ 
             Real.tan θ = Real.sqrt m) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_inclination_l1310_131014


namespace NUMINAMATH_CALUDE_time_difference_per_question_l1310_131042

/-- Calculates the difference in time per question between Math and English exams -/
theorem time_difference_per_question 
  (english_questions : ℕ) 
  (math_questions : ℕ)
  (english_time : ℕ) 
  (math_time : ℕ)
  (h1 : english_questions = 50)
  (h2 : math_questions = 20)
  (h3 : english_time = 80)
  (h4 : math_time = 110) : 
  (math_time : ℚ) / math_questions - (english_time : ℚ) / english_questions = 39/10 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_per_question_l1310_131042


namespace NUMINAMATH_CALUDE_grandmother_age_problem_l1310_131035

theorem grandmother_age_problem (yuna_initial_age grandmother_initial_age : ℕ) 
  (h1 : yuna_initial_age = 12)
  (h2 : grandmother_initial_age = 72) :
  ∃ (years_passed : ℕ), 
    grandmother_initial_age + years_passed = 5 * (yuna_initial_age + years_passed) ∧
    grandmother_initial_age + years_passed = 75 := by
  sorry

end NUMINAMATH_CALUDE_grandmother_age_problem_l1310_131035


namespace NUMINAMATH_CALUDE_no_integer_solution_l1310_131006

theorem no_integer_solution : ¬∃ (n : ℕ+), (20 * n + 2) ∣ (2003 * n + 2002) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1310_131006


namespace NUMINAMATH_CALUDE_square_side_length_from_rectangle_l1310_131062

theorem square_side_length_from_rectangle (width height : ℝ) (h1 : width = 10) (h2 : height = 20) :
  ∃ y : ℝ, y^2 = width * height ∧ y = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_from_rectangle_l1310_131062


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1310_131033

theorem sqrt_product_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1310_131033


namespace NUMINAMATH_CALUDE_marsh_birds_count_l1310_131078

theorem marsh_birds_count (geese ducks : ℕ) (h1 : geese = 58) (h2 : ducks = 37) :
  geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_marsh_birds_count_l1310_131078


namespace NUMINAMATH_CALUDE_quarters_in_jar_l1310_131077

/-- Represents the number of coins of each type in the jar -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  half_dollars : ℕ
  dollar_coins : ℕ
  two_dollar_coins : ℕ

/-- Represents the cost of a sundae and its modifications -/
structure SundaeCost where
  base : ℚ
  special_topping : ℚ
  featured_flavor : ℚ

/-- Represents the family's ice cream trip details -/
structure IceCreamTrip where
  family_size : ℕ
  special_toppings : ℕ
  featured_flavors : ℕ
  leftover : ℚ

def count_quarters (coins : CoinCounts) (sundae : SundaeCost) (trip : IceCreamTrip) : ℕ :=
  sorry

theorem quarters_in_jar 
  (coins : CoinCounts)
  (sundae : SundaeCost)
  (trip : IceCreamTrip) :
  coins.pennies = 123 ∧ 
  coins.nickels = 85 ∧ 
  coins.dimes = 35 ∧ 
  coins.half_dollars = 15 ∧ 
  coins.dollar_coins = 5 ∧ 
  coins.two_dollar_coins = 4 ∧
  sundae.base = 5.25 ∧
  sundae.special_topping = 0.5 ∧
  sundae.featured_flavor = 0.25 ∧
  trip.family_size = 8 ∧
  trip.special_toppings = 3 ∧
  trip.featured_flavors = 5 ∧
  trip.leftover = 0.97 →
  count_quarters coins sundae trip = 54 :=
by sorry

end NUMINAMATH_CALUDE_quarters_in_jar_l1310_131077


namespace NUMINAMATH_CALUDE_locus_of_centers_l1310_131043

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ with equation (x - 1)² + (y - 1)² = 81 -/
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 81

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 1)^2 + (b - 1)^2 = (9 - r)^2

/-- The locus of centers (a, b) of circles externally tangent to C₁ and internally tangent to C₂ -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) → 
  a^2 + b^2 - (2*a*b)/63 - (66*a)/63 - (66*b)/63 + 17 = 0 := by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l1310_131043


namespace NUMINAMATH_CALUDE_min_distance_exp_to_line_l1310_131017

/-- The minimum distance from a point on y = e^x to y = x is √2/2 -/
theorem min_distance_exp_to_line :
  let f : ℝ → ℝ := fun x ↦ Real.exp x
  let g : ℝ → ℝ := fun x ↦ x
  ∃ (x₀ : ℝ), ∀ (x : ℝ),
    Real.sqrt ((x - g x)^2 + (f x - g x)^2) ≥ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_exp_to_line_l1310_131017


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1310_131005

/-- Given a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1, where a > 0, b > 0,
    if one focus is at (2,0) and one asymptote has a slope of √3,
    then the equation of the hyperbola is x^2 - (y^2 / 3) = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (2 : ℝ)^2 = a^2 + b^2 →
  b / a = Real.sqrt 3 →
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1310_131005


namespace NUMINAMATH_CALUDE_betty_bracelets_l1310_131013

/-- The number of bracelets that can be made given a total number of stones and stones per bracelet -/
def num_bracelets (total_stones : ℕ) (stones_per_bracelet : ℕ) : ℕ :=
  total_stones / stones_per_bracelet

/-- Theorem: Given 140 stones and 14 stones per bracelet, the number of bracelets is 10 -/
theorem betty_bracelets :
  num_bracelets 140 14 = 10 := by
  sorry

end NUMINAMATH_CALUDE_betty_bracelets_l1310_131013


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1310_131044

theorem simplify_trig_expression :
  let sin30 : ℝ := 1 / 2
  let cos30 : ℝ := Real.sqrt 3 / 2
  ∀ (sin10 sin20 cos10 : ℝ),
    (sin10 + sin20 * cos30) / (cos10 - sin20 * sin30) = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1310_131044


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1310_131026

def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - (a + 1) * x + 1 < 0

def solution_set (a : ℝ) : Set ℝ :=
  {x | quadratic_inequality a x}

theorem solution_set_characterization (a : ℝ) (h : a > 0) :
  (a = 2 → solution_set a = Set.Ioo (1/2) 1) ∧
  (0 < a ∧ a < 1 → solution_set a = Set.Ioo 1 (1/a)) ∧
  (a = 1 → solution_set a = ∅) ∧
  (a > 1 → solution_set a = Set.Ioo (1/a) 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1310_131026


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l1310_131094

/-- A line in 3D space -/
structure Line3D where
  -- Define the line structure (omitted for brevity)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane structure (omitted for brevity)

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def lineParallelToPlane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def linePerpendicularToPlane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to another line -/
def linePerpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to a plane and another line is parallel to the plane,
    then the two lines are perpendicular to each other -/
theorem perpendicular_parallel_implies_perpendicular
  (a b : Line3D) (α : Plane3D)
  (h1 : linePerpendicularToPlane a α)
  (h2 : lineParallelToPlane b α) :
  linePerpendicular a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l1310_131094


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_l1310_131034

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (n : ℕ) (abc_group : ℕ) (de_pair : ℕ) : ℕ :=
  factorial n - (factorial (n - abc_group + 1) * factorial abc_group +
                 factorial (n - de_pair + 1) * factorial de_pair -
                 factorial (n - abc_group - de_pair + 2) * factorial abc_group * factorial de_pair)

theorem valid_seating_arrangements :
  seating_arrangements 10 3 2 = 3144960 :=
by sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_l1310_131034


namespace NUMINAMATH_CALUDE_adams_game_rounds_l1310_131097

/-- Given Adam's total score and points per round, prove the number of rounds played --/
theorem adams_game_rounds (total_points : ℕ) (points_per_round : ℕ) 
  (h1 : total_points = 283) 
  (h2 : points_per_round = 71) : 
  total_points / points_per_round = 4 := by
  sorry

end NUMINAMATH_CALUDE_adams_game_rounds_l1310_131097


namespace NUMINAMATH_CALUDE_product_of_solutions_l1310_131003

theorem product_of_solutions (x : ℝ) : 
  (3 * x^2 + 5 * x - 40 = 0) → 
  (∃ y : ℝ, 3 * y^2 + 5 * y - 40 = 0 ∧ x * y = -40/3) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l1310_131003


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1310_131021

-- Define the time to fill without leak
def T : ℝ := 12

-- Define the time to fill with leak
def time_with_leak : ℝ := T + 2

-- Define the time to empty when full
def time_to_empty : ℝ := 84

-- State the theorem
theorem cistern_fill_time :
  (1 / T - 1 / time_to_empty = 1 / time_with_leak) ∧
  (T > 0) :=
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1310_131021


namespace NUMINAMATH_CALUDE_work_multiple_l1310_131039

/-- Given that P persons can complete a work W in 12 days, 
    and mP persons can complete half of the work (W/2) in 3 days,
    prove that the multiple m is 2. -/
theorem work_multiple (P : ℕ) (W : ℝ) (m : ℝ) 
  (h1 : P > 0) (h2 : W > 0) (h3 : m > 0)
  (complete_full : P * 12 * (W / (P * 12)) = W)
  (complete_half : m * P * 3 * (W / (2 * m * P * 3)) = W / 2) : 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_work_multiple_l1310_131039


namespace NUMINAMATH_CALUDE_normal_equation_for_given_conditions_l1310_131024

def normal_equation (p : ℝ) (α : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x * Real.cos α + y * Real.sin α - p = 0

theorem normal_equation_for_given_conditions :
  let p : ℝ := 3
  let α₁ : ℝ := π / 4  -- 45°
  let α₂ : ℝ := 7 * π / 4  -- 315°
  (∀ x y, normal_equation p α₁ x y ↔ Real.sqrt 2 / 2 * x + Real.sqrt 2 / 2 * y - 3 = 0) ∧
  (∀ x y, normal_equation p α₂ x y ↔ Real.sqrt 2 / 2 * x - Real.sqrt 2 / 2 * y - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_normal_equation_for_given_conditions_l1310_131024


namespace NUMINAMATH_CALUDE_waysToSum1800_eq_45651_l1310_131059

/-- The number of ways to write 1800 as the sum of ones, twos, and threes, ignoring order -/
def waysToSum1800 : ℕ := sorry

/-- The target number we're considering -/
def targetNumber : ℕ := 1800

theorem waysToSum1800_eq_45651 : waysToSum1800 = 45651 := by sorry

end NUMINAMATH_CALUDE_waysToSum1800_eq_45651_l1310_131059


namespace NUMINAMATH_CALUDE_equal_intercepts_equation_not_in_second_quadrant_range_l1310_131057

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 1) * x + y + 2 + a = 0

-- Define the condition for equal intercepts
def equal_intercepts (a : ℝ) : Prop :=
  ∃ k, k = -a - 2 ∧ k = (-a - 2) / (a + 1)

-- Define the condition for not passing through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop :=
  a = -1 ∨ (-(a + 1) > 0 ∧ -a - 2 ≤ 0)

-- Theorem for equal intercepts
theorem equal_intercepts_equation (a : ℝ) :
  equal_intercepts a → (a = 0 ∨ a = -2) :=
sorry

-- Theorem for not passing through the second quadrant
theorem not_in_second_quadrant_range (a : ℝ) :
  not_in_second_quadrant a → -2 ≤ a ∧ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_equal_intercepts_equation_not_in_second_quadrant_range_l1310_131057


namespace NUMINAMATH_CALUDE_sandwiches_problem_l1310_131070

def sandwiches_left (initial : ℕ) (ruth_ate : ℕ) (brother_given : ℕ) (first_cousin_ate : ℕ) (other_cousins_ate : ℕ) : ℕ :=
  initial - ruth_ate - brother_given - first_cousin_ate - other_cousins_ate

theorem sandwiches_problem :
  sandwiches_left 10 1 2 2 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_problem_l1310_131070


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1310_131007

theorem digit_sum_problem (p q r : ℕ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 →
  p < 10 → q < 10 → r < 10 →
  100 * p + 10 * q + r + 10 * q + r + r = 912 →
  q = 5 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1310_131007


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1310_131020

theorem perfect_square_condition (A B : ℤ) : 
  (800 < A ∧ A < 1300) → 
  B > 1 → 
  A = B^4 → 
  (∃ n : ℤ, A = n^2) ↔ (B = 5 ∨ B = 6) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1310_131020


namespace NUMINAMATH_CALUDE_gcd_of_squares_l1310_131049

theorem gcd_of_squares : Nat.gcd (111^2 + 222^2 + 333^2) (110^2 + 221^2 + 334^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l1310_131049
