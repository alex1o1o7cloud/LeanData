import Mathlib

namespace NUMINAMATH_CALUDE_circle_tangency_radius_sum_l2874_287423

/-- A circle with center D(r, r) is tangent to the positive x and y-axes
    and externally tangent to a circle centered at (5,0) with radius 1.
    The sum of all possible radii of the circle with center D is 12. -/
theorem circle_tangency_radius_sum : 
  ∀ r : ℝ, 
    (r > 0) →
    ((r - 5)^2 + r^2 = (r + 1)^2) →
    (∃ s : ℝ, (s > 0) ∧ ((s - 5)^2 + s^2 = (s + 1)^2) ∧ (r + s = 12)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangency_radius_sum_l2874_287423


namespace NUMINAMATH_CALUDE_consecutive_squares_equality_l2874_287452

theorem consecutive_squares_equality :
  ∃ (a b c d : ℝ), (b = a + 1 ∧ c = b + 1 ∧ d = c + 1) ∧ (a^2 + b^2 = c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_equality_l2874_287452


namespace NUMINAMATH_CALUDE_circle_tangency_sum_of_radii_l2874_287448

/-- A circle with center C(r, r) is tangent to the positive x and y-axes
    and externally tangent to a circle centered at (5,0) with radius 2.
    The sum of all possible values of r is 14. -/
theorem circle_tangency_sum_of_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ s : ℝ, (s > 0) ∧ ((s - 5)^2 + s^2 = (s + 2)^2) ∧ (r + s = 14)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangency_sum_of_radii_l2874_287448


namespace NUMINAMATH_CALUDE_translation_of_points_l2874_287426

/-- Given two points A and B in ℝ², if A is translated to A₁, 
    then B translated by the same vector results in B₁ -/
theorem translation_of_points (A B A₁ B₁ : ℝ × ℝ) : 
  A = (-1, 0) → 
  B = (1, 2) → 
  A₁ = (2, -1) → 
  B₁.1 = B.1 + (A₁.1 - A.1) ∧ B₁.2 = B.2 + (A₁.2 - A.2) → 
  B₁ = (4, 1) := by
  sorry

end NUMINAMATH_CALUDE_translation_of_points_l2874_287426


namespace NUMINAMATH_CALUDE_card_number_solution_l2874_287420

theorem card_number_solution : ∃ (L O M N S V : ℕ), 
  (L < 10) ∧ (O < 10) ∧ (M < 10) ∧ (N < 10) ∧ (S < 10) ∧ (V < 10) ∧
  (L ≠ O) ∧ (L ≠ M) ∧ (L ≠ N) ∧ (L ≠ S) ∧ (L ≠ V) ∧
  (O ≠ M) ∧ (O ≠ N) ∧ (O ≠ S) ∧ (O ≠ V) ∧
  (M ≠ N) ∧ (M ≠ S) ∧ (M ≠ V) ∧
  (N ≠ S) ∧ (N ≠ V) ∧
  (S ≠ V) ∧
  (0 < O) ∧ (O < M) ∧ (O < S) ∧
  (L + O * S + O * M + N * M * S + O * M = 10 * M * S + V * M * S) :=
by sorry


end NUMINAMATH_CALUDE_card_number_solution_l2874_287420


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l2874_287492

/-- Represents the number of frogs of each color -/
structure FrogCounts where
  green : Nat
  red : Nat
  blue : Nat

/-- Represents the arrangement rules for frogs -/
structure FrogRules where
  green_red_adjacent : Bool
  green_blue_adjacent : Bool
  red_blue_adjacent : Bool
  blue_blue_adjacent : Bool

/-- Calculates the number of valid frog arrangements -/
def countFrogArrangements (counts : FrogCounts) (rules : FrogRules) : Nat :=
  sorry

/-- The main theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  let counts : FrogCounts := ⟨2, 3, 2⟩
  let rules : FrogRules := ⟨false, true, true, true⟩
  countFrogArrangements counts rules = 72 := by sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l2874_287492


namespace NUMINAMATH_CALUDE_blue_faces_cube_l2874_287437

theorem blue_faces_cube (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_faces_cube_l2874_287437


namespace NUMINAMATH_CALUDE_side_margin_width_l2874_287436

/-- Given a sheet of paper with dimensions and margin constraints, prove the side margin width. -/
theorem side_margin_width (sheet_width sheet_length top_bottom_margin : ℝ)
  (typing_area_percentage : ℝ) (h1 : sheet_width = 20)
  (h2 : sheet_length = 30) (h3 : top_bottom_margin = 3)
  (h4 : typing_area_percentage = 0.64) :
  ∃ (side_margin : ℝ),
    side_margin = 2 ∧
    (sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin) =
      typing_area_percentage * sheet_width * sheet_length :=
by sorry

end NUMINAMATH_CALUDE_side_margin_width_l2874_287436


namespace NUMINAMATH_CALUDE_joan_balloons_l2874_287425

/-- The number of blue balloons Joan has now -/
def total_balloons (initial : ℕ) (gained : ℕ) : ℕ :=
  initial + gained

/-- Proof that Joan has 11 blue balloons now -/
theorem joan_balloons : total_balloons 9 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l2874_287425


namespace NUMINAMATH_CALUDE_stock_price_calculation_l2874_287421

def stock_price_evolution (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

theorem stock_price_calculation :
  stock_price_evolution 150 0.5 0.3 = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l2874_287421


namespace NUMINAMATH_CALUDE_solution_pairs_l2874_287435

theorem solution_pairs (a b : ℝ) :
  2 * (a^2 + 1) * (b^2 + 1) = (a + 1) * (b + 1) * (a * b + 1) →
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1)) := by
sorry

end NUMINAMATH_CALUDE_solution_pairs_l2874_287435


namespace NUMINAMATH_CALUDE_swimmer_speed_l2874_287428

/-- The speed of a swimmer in still water, given downstream and upstream swim data -/
theorem swimmer_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 30) (h2 : upstream_distance = 20) 
  (h3 : time = 5) : ∃ (v_man v_stream : ℝ),
  downstream_distance / time = v_man + v_stream ∧
  upstream_distance / time = v_man - v_stream ∧
  v_man = 5 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_l2874_287428


namespace NUMINAMATH_CALUDE_ratio_of_bases_l2874_287463

/-- An isosceles trapezoid with a point inside dividing it into four triangles -/
structure IsoscelesTrapezoidWithPoint where
  /-- Length of the larger base AB -/
  AB : ℝ
  /-- Length of the smaller base CD -/
  CD : ℝ
  /-- Area of triangle with base CD -/
  area_CD : ℝ
  /-- Area of triangle adjacent to CD (clockwise) -/
  area_adj_CD : ℝ
  /-- Area of triangle with base AB -/
  area_AB : ℝ
  /-- Area of triangle adjacent to AB (counter-clockwise) -/
  area_adj_AB : ℝ
  /-- AB is longer than CD -/
  h_AB_gt_CD : AB > CD
  /-- The trapezoid is isosceles -/
  h_isosceles : True  -- We don't need to specify this condition explicitly for the proof
  /-- The bases are parallel -/
  h_parallel : True   -- We don't need to specify this condition explicitly for the proof
  /-- Areas of triangles -/
  h_areas : area_CD = 5 ∧ area_adj_CD = 7 ∧ area_AB = 9 ∧ area_adj_AB = 3

/-- The ratio of bases in the isosceles trapezoid with given triangle areas -/
theorem ratio_of_bases (t : IsoscelesTrapezoidWithPoint) : t.AB / t.CD = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_bases_l2874_287463


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2874_287410

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ (m ≤ 4/3 ∧ m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2874_287410


namespace NUMINAMATH_CALUDE_trees_along_road_l2874_287495

theorem trees_along_road (road_length : ℕ) (tree_spacing : ℕ) (h1 : road_length = 1000) (h2 : tree_spacing = 5) :
  road_length / tree_spacing + 1 = 201 := by
  sorry

end NUMINAMATH_CALUDE_trees_along_road_l2874_287495


namespace NUMINAMATH_CALUDE_max_consecutive_expressible_l2874_287409

/-- A function that represents the expression x^3 + 2y^2 --/
def f (x y : ℤ) : ℤ := x^3 + 2*y^2

/-- The property of being expressible in the form x^3 + 2y^2 --/
def expressible (n : ℤ) : Prop := ∃ x y : ℤ, f x y = n

/-- A sequence of consecutive integers starting from a given integer --/
def consecutive_seq (start : ℤ) (length : ℕ) : Set ℤ :=
  {n : ℤ | start ≤ n ∧ n < start + length}

/-- The main theorem stating the maximal length of consecutive expressible integers --/
theorem max_consecutive_expressible :
  (∃ start : ℤ, ∀ n ∈ consecutive_seq start 5, expressible n) ∧
  (∀ start : ℤ, ∀ length : ℕ, length > 5 →
    ∃ n ∈ consecutive_seq start length, ¬expressible n) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_expressible_l2874_287409


namespace NUMINAMATH_CALUDE_vectors_not_parallel_l2874_287443

def vector_a : Fin 2 → ℝ := ![2, 0]
def vector_b : Fin 2 → ℝ := ![0, 2]

theorem vectors_not_parallel : ¬ (∃ (k : ℝ), vector_a = k • vector_b) := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_parallel_l2874_287443


namespace NUMINAMATH_CALUDE_intersection_equals_specific_set_l2874_287479

-- Define the set P
def P : Set ℝ := {x | ∃ k : ℤ, 2 * k * Real.pi ≤ x ∧ x ≤ (2 * k + 1) * Real.pi}

-- Define the set Q
def Q : Set ℝ := {α | -4 ≤ α ∧ α ≤ 4}

-- Define the intersection set
def intersection_set : Set ℝ := {α | (-4 ≤ α ∧ α ≤ -Real.pi) ∨ (0 ≤ α ∧ α ≤ Real.pi)}

-- Theorem statement
theorem intersection_equals_specific_set : P ∩ Q = intersection_set := by sorry

end NUMINAMATH_CALUDE_intersection_equals_specific_set_l2874_287479


namespace NUMINAMATH_CALUDE_solution_set_inequality_inequality_for_negative_a_l2874_287499

-- Define the function f(x) = |x - 5|
def f (x : ℝ) : ℝ := |x - 5|

-- Theorem 1: Solution set of f(x) + f(x + 2) ≤ 3
theorem solution_set_inequality (x : ℝ) :
  (f x + f (x + 2) ≤ 3) ↔ (5/2 ≤ x ∧ x ≤ 11/2) := by sorry

-- Theorem 2: f(ax) - f(5a) ≥ af(x) for a < 0
theorem inequality_for_negative_a (a x : ℝ) (h : a < 0) :
  f (a * x) - f (5 * a) ≥ a * f x := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_inequality_for_negative_a_l2874_287499


namespace NUMINAMATH_CALUDE_cargo_loaded_in_bahamas_l2874_287439

/-- The amount of cargo loaded in the Bahamas is equal to the difference between the final amount of cargo and the initial amount of cargo. -/
theorem cargo_loaded_in_bahamas (initial_cargo final_cargo : ℕ) 
  (h1 : initial_cargo = 5973)
  (h2 : final_cargo = 14696) :
  final_cargo - initial_cargo = 8723 := by
  sorry

end NUMINAMATH_CALUDE_cargo_loaded_in_bahamas_l2874_287439


namespace NUMINAMATH_CALUDE_peters_newspaper_delivery_l2874_287491

/-- Peter's newspaper delivery problem -/
theorem peters_newspaper_delivery :
  let total_weekend := 110
  let saturday := 45
  let sunday := 65
  sunday > saturday →
  sunday - saturday = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_peters_newspaper_delivery_l2874_287491


namespace NUMINAMATH_CALUDE_six_stairs_ways_l2874_287487

/-- The number of ways to climb n stairs, taking 1, 2, or 3 stairs at a time -/
def stairClimbWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => stairClimbWays (n + 2) + stairClimbWays (n + 1) + stairClimbWays n

theorem six_stairs_ways :
  stairClimbWays 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_six_stairs_ways_l2874_287487


namespace NUMINAMATH_CALUDE_expression_evaluation_l2874_287416

theorem expression_evaluation (b : ℚ) (h : b = 4/3) :
  (7*b^2 - 15*b + 5) * (3*b - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2874_287416


namespace NUMINAMATH_CALUDE_field_trip_buses_l2874_287408

/-- The number of classrooms in the school -/
def num_classrooms : ℕ := 67

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 66

/-- The number of seats in each bus -/
def seats_per_bus : ℕ := 6

/-- The function to calculate the minimum number of buses needed -/
def min_buses_needed (classrooms : ℕ) (students : ℕ) (seats : ℕ) : ℕ :=
  (classrooms * students + seats - 1) / seats

/-- Theorem stating the minimum number of buses needed for the field trip -/
theorem field_trip_buses :
  min_buses_needed num_classrooms students_per_classroom seats_per_bus = 738 := by
  sorry


end NUMINAMATH_CALUDE_field_trip_buses_l2874_287408


namespace NUMINAMATH_CALUDE_partnership_profit_l2874_287419

/-- Calculates the total profit of a partnership business given the investments and one partner's share of the profit -/
theorem partnership_profit 
  (investment_A investment_B investment_C : ℕ) 
  (profit_share_A : ℕ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : profit_share_A = 4260) :
  (investment_A + investment_B + investment_C) * profit_share_A / investment_A = 14200 := by
  sorry

#check partnership_profit

end NUMINAMATH_CALUDE_partnership_profit_l2874_287419


namespace NUMINAMATH_CALUDE_median_line_equation_circle_equation_l2874_287461

/-- Triangle ABC with vertices A(-3,0), B(2,0), and C(0,-4) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Define the specific triangle ABC -/
def triangleABC : Triangle :=
  { A := (-3, 0),
    B := (2, 0),
    C := (0, -4) }

/-- General form of a line equation: ax + by + c = 0 -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- General form of a circle equation: x^2 + y^2 + dx + ey + f = 0 -/
structure Circle :=
  (d : ℝ)
  (e : ℝ)
  (f : ℝ)

/-- Theorem: The median line of side BC in triangle ABC has equation x + 2y + 3 = 0 -/
theorem median_line_equation (t : Triangle) (l : Line) : t = triangleABC → l = { a := 1, b := 2, c := 3 } := by sorry

/-- Theorem: The circle passing through points A, B, and C has equation x^2 + y^2 + x + (5/2)y - 6 = 0 -/
theorem circle_equation (t : Triangle) (c : Circle) : t = triangleABC → c = { d := 1, e := 5/2, f := -6 } := by sorry

end NUMINAMATH_CALUDE_median_line_equation_circle_equation_l2874_287461


namespace NUMINAMATH_CALUDE_milk_problem_l2874_287404

theorem milk_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (sam_fraction : ℚ) : 
  initial_milk = 3/4 →
  rachel_fraction = 1/2 →
  sam_fraction = 1/3 →
  sam_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_milk_problem_l2874_287404


namespace NUMINAMATH_CALUDE_ant_collision_theorem_l2874_287431

/-- Represents the possible numbers of ants on the track -/
def PossibleAntCounts : Set ℕ := {10, 11, 14, 25}

/-- Represents a configuration of ants on the track -/
structure AntConfiguration where
  clockwise : ℕ
  counterclockwise : ℕ

/-- Checks if a given configuration is valid -/
def isValidConfiguration (config : AntConfiguration) : Prop :=
  config.clockwise * config.counterclockwise = 24

theorem ant_collision_theorem
  (track_length : ℕ)
  (ant_speed : ℕ)
  (collision_pairs : ℕ)
  (h1 : track_length = 60)
  (h2 : ant_speed = 1)
  (h3 : collision_pairs = 48) :
  ∀ (total_ants : ℕ),
    (∃ (config : AntConfiguration),
      config.clockwise + config.counterclockwise = total_ants ∧
      isValidConfiguration config) →
    total_ants ∈ PossibleAntCounts :=
sorry

end NUMINAMATH_CALUDE_ant_collision_theorem_l2874_287431


namespace NUMINAMATH_CALUDE_apples_eaten_l2874_287470

theorem apples_eaten (total : ℕ) (eaten : ℕ) : 
  total = 6 → 
  eaten + 2 * eaten = total → 
  eaten = 2 := by
sorry

end NUMINAMATH_CALUDE_apples_eaten_l2874_287470


namespace NUMINAMATH_CALUDE_tangent_two_identities_l2874_287445

open Real

theorem tangent_two_identities (α : ℝ) (h : tan α = 2) :
  (2 * sin α + 2 * cos α) / (sin α - cos α) = 8 ∧
  (cos (π - α) * cos (π / 2 + α) * sin (α - 3 * π / 2)) /
  (sin (3 * π + α) * sin (α - π) * cos (π + α)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_two_identities_l2874_287445


namespace NUMINAMATH_CALUDE_lolita_milk_consumption_l2874_287432

/-- Lolita's weekly milk consumption --/
theorem lolita_milk_consumption :
  let weekday_consumption : ℕ := 3
  let saturday_consumption : ℕ := 2 * weekday_consumption
  let sunday_consumption : ℕ := 3 * weekday_consumption
  let weekdays : ℕ := 5
  weekdays * weekday_consumption + saturday_consumption + sunday_consumption = 30 := by
  sorry

end NUMINAMATH_CALUDE_lolita_milk_consumption_l2874_287432


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2874_287400

/-- Given a quadratic equation that can be factored into two linear factors, prove the value of m -/
theorem quadratic_factorization (m : ℝ) : 
  (∃ (a b : ℝ), ∀ (x y : ℝ), 
    x^2 + 7*x*y + m*y^2 - 5*x + 43*y - 24 = (x + a*y + 3) * (x + b*y - 8)) → 
  m = -18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2874_287400


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2874_287466

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 1 ≤ 0) ↔ (∃ x : ℝ, x^2 - 3*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2874_287466


namespace NUMINAMATH_CALUDE_angle_AOF_is_118_l2874_287403

/-- Given a configuration of angles where:
    ∠AOB = ∠BOC
    ∠COD = ∠DOE = ∠EOF
    ∠AOD = 82°
    ∠BOE = 68°
    Prove that ∠AOF = 118° -/
theorem angle_AOF_is_118 (AOB BOC COD DOE EOF AOD BOE : ℝ) : 
  AOB = BOC ∧ 
  COD = DOE ∧ DOE = EOF ∧
  AOD = 82 ∧
  BOE = 68 →
  AOB + BOC + COD + DOE + EOF = 118 := by
  sorry

end NUMINAMATH_CALUDE_angle_AOF_is_118_l2874_287403


namespace NUMINAMATH_CALUDE_vector_addition_l2874_287447

theorem vector_addition :
  let v1 : Fin 2 → ℝ := ![5, -9]
  let v2 : Fin 2 → ℝ := ![-8, 14]
  v1 + v2 = ![(-3), 5] := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l2874_287447


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2874_287472

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 2 + a 6) / 2 = 5 ∧
  (a 3 + a 7) / 2 = 7

/-- The general term of the arithmetic sequence -/
def GeneralTerm (n : ℕ) : ℝ := 2 * n - 3

theorem arithmetic_sequence_general_term (a : ℕ → ℝ) :
  ArithmeticSequence a → ∀ n : ℕ, a n = GeneralTerm n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2874_287472


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2874_287414

theorem pure_imaginary_product (a : ℝ) : 
  (Complex.I * (6 - a) : ℂ) = (3 - Complex.I) * (a + 2 * Complex.I) → a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2874_287414


namespace NUMINAMATH_CALUDE_race_length_is_1000_l2874_287412

/-- The length of a race, given the positions of two runners at the end. -/
def race_length (jack_position : ℕ) (distance_apart : ℕ) : ℕ :=
  jack_position + distance_apart

/-- Theorem stating that the race length is 1000 meters given the conditions -/
theorem race_length_is_1000 :
  race_length 152 848 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_race_length_is_1000_l2874_287412


namespace NUMINAMATH_CALUDE_sector_area_l2874_287477

/-- The area of a circular sector with radius 6 cm and central angle 30° is 3π cm². -/
theorem sector_area : 
  let r : ℝ := 6
  let α : ℝ := 30 * π / 180  -- Convert degrees to radians
  (1/2) * r^2 * α = 3 * π := by sorry

end NUMINAMATH_CALUDE_sector_area_l2874_287477


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l2874_287433

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 2

-- Define point M
def point_M : ℝ × ℝ := (-2, 1)

-- Define trajectory E
def trajectory_E (x y : ℝ) : Prop := 4*x + 2*y - 3 = 0

-- Theorem statement
theorem min_distance_circle_to_line :
  ∃ (min_dist : ℝ),
    min_dist = (11 * Real.sqrt 5) / 10 - Real.sqrt 2 ∧
    ∀ (a b : ℝ × ℝ),
      circle_C a.1 a.2 →
      trajectory_E b.1 b.2 →
      Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l2874_287433


namespace NUMINAMATH_CALUDE_max_term_of_sequence_l2874_287483

theorem max_term_of_sequence (n : ℕ) : 
  let a : ℕ → ℤ := λ k => -2 * k^2 + 9 * k + 3
  ∀ k, a k ≤ a 2 := by
  sorry

end NUMINAMATH_CALUDE_max_term_of_sequence_l2874_287483


namespace NUMINAMATH_CALUDE_tank_fill_time_l2874_287449

def fill_time_A : ℝ := 60
def fill_time_B : ℝ := 40

theorem tank_fill_time :
  let total_time : ℝ := 30
  let first_half_time : ℝ := total_time / 2
  let second_half_time : ℝ := total_time / 2
  let fill_rate_A : ℝ := 1 / fill_time_A
  let fill_rate_B : ℝ := 1 / fill_time_B
  (fill_rate_B * first_half_time) + ((fill_rate_A + fill_rate_B) * second_half_time) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tank_fill_time_l2874_287449


namespace NUMINAMATH_CALUDE_prob_same_color_is_139_435_l2874_287480

-- Define the number of socks for each color
def blue_socks : ℕ := 12
def gray_socks : ℕ := 10
def white_socks : ℕ := 8

-- Define the total number of socks
def total_socks : ℕ := blue_socks + gray_socks + white_socks

-- Define the probability of picking two socks of the same color
def prob_same_color : ℚ :=
  (Nat.choose blue_socks 2 + Nat.choose gray_socks 2 + Nat.choose white_socks 2) /
  Nat.choose total_socks 2

-- Theorem statement
theorem prob_same_color_is_139_435 : prob_same_color = 139 / 435 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_139_435_l2874_287480


namespace NUMINAMATH_CALUDE_car_selection_problem_l2874_287430

theorem car_selection_problem (num_cars : ℕ) (selections_per_car : ℕ) (cars_per_client : ℕ)
  (h_num_cars : num_cars = 15)
  (h_selections_per_car : selections_per_car = 3)
  (h_cars_per_client : cars_per_client = 3) :
  (num_cars * selections_per_car) / cars_per_client = 15 := by
  sorry

end NUMINAMATH_CALUDE_car_selection_problem_l2874_287430


namespace NUMINAMATH_CALUDE_digit_sum_problem_l2874_287454

theorem digit_sum_problem (p q r : ℕ) : 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  0 < p ∧ p < 10 ∧ 0 < q ∧ q < 10 ∧ 0 < r ∧ r < 10 ∧
  (10 * p + q) * (10 * p + r) = 221 →
  p + q + r = 11 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l2874_287454


namespace NUMINAMATH_CALUDE_john_gave_twenty_l2874_287438

/-- The amount of money John gave to the store for buying Slurpees -/
def money_given (cost_per_slurpee : ℕ) (num_slurpees : ℕ) (change_received : ℕ) : ℕ :=
  cost_per_slurpee * num_slurpees + change_received

/-- Proof that John gave $20 to the store -/
theorem john_gave_twenty :
  money_given 2 6 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_gave_twenty_l2874_287438


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_cube_eq_x_l2874_287496

theorem x_eq_one_sufficient_not_necessary_for_cube_eq_x (x : ℝ) :
  (x = 1 → x^3 = x) ∧ ¬(x^3 = x → x = 1) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_cube_eq_x_l2874_287496


namespace NUMINAMATH_CALUDE_equation_solution_l2874_287415

theorem equation_solution : ∃ x : ℚ, 5*x + 9*x = 360 - 7*(x + 4) ∧ x = 332/21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2874_287415


namespace NUMINAMATH_CALUDE_point_translation_and_line_l2874_287450

/-- Given a point (5,3) translated 4 units left and 1 unit down,
    if the resulting point lies on y = kx - 2, then k = 4 -/
theorem point_translation_and_line (k : ℝ) : 
  let original_point : ℝ × ℝ := (5, 3)
  let translated_point : ℝ × ℝ := (original_point.1 - 4, original_point.2 - 1)
  (translated_point.2 = k * translated_point.1 - 2) → k = 4 := by
sorry

end NUMINAMATH_CALUDE_point_translation_and_line_l2874_287450


namespace NUMINAMATH_CALUDE_star_composition_l2874_287406

-- Define the star operation
def star (x y : ℝ) : ℝ := x^3 - x*y

-- Theorem statement
theorem star_composition (j : ℝ) : star j (star j j) = 2*j^3 - j^4 := by
  sorry

end NUMINAMATH_CALUDE_star_composition_l2874_287406


namespace NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l2874_287407

-- Define the function f(x) = |2x-a| + 5x
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + 5*x

-- Part I: Solution set when a = 3
theorem solution_set_part_i :
  ∀ x : ℝ, f 3 x ≥ 5*x + 1 ↔ x ≤ 1 ∨ x ≥ 2 := by sorry

-- Part II: Value of a for given solution set
theorem solution_set_part_ii :
  (∀ x : ℝ, f 3 x ≤ 0 ↔ x ≤ -1) := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l2874_287407


namespace NUMINAMATH_CALUDE_sufficient_condition_for_monotonic_decrease_l2874_287478

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the property of being monotonic decreasing on an interval
def monotonic_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y ≤ f x

-- Theorem statement
theorem sufficient_condition_for_monotonic_decrease :
  ∃ (f : ℝ → ℝ), (∀ x, deriv f x = f' x) →
    (monotonic_decreasing_on (fun x ↦ f (x + 1)) 0 1) ∧
    ¬(∀ g : ℝ → ℝ, (∀ x, deriv g x = f' x) → 
      monotonic_decreasing_on (fun x ↦ g (x + 1)) 0 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_monotonic_decrease_l2874_287478


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2874_287493

theorem line_passes_through_point :
  ∀ (t : ℝ), (t + 1) * (-4) - (2 * t + 5) * (-2) - 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2874_287493


namespace NUMINAMATH_CALUDE_triangles_in_hexagon_count_l2874_287427

/-- The number of vertices in a hexagon -/
def hexagon_vertices : ℕ := 6

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of different triangles that can be formed using the vertices of a hexagon -/
def triangles_in_hexagon : ℕ := Nat.choose hexagon_vertices triangle_vertices

theorem triangles_in_hexagon_count :
  triangles_in_hexagon = 20 := by sorry

end NUMINAMATH_CALUDE_triangles_in_hexagon_count_l2874_287427


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2874_287498

theorem bowling_ball_weight (canoe_weight : ℕ) (num_canoes num_balls : ℕ) :
  canoe_weight = 35 →
  num_canoes = 4 →
  num_balls = 10 →
  num_canoes * canoe_weight = num_balls * (num_canoes * canoe_weight / num_balls) →
  (num_canoes * canoe_weight / num_balls : ℕ) = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2874_287498


namespace NUMINAMATH_CALUDE_line_through_point_l2874_287417

/-- Given a line equation bx - (b+2)y = b-3 that passes through the point (3, -5), prove that b = -13/7 --/
theorem line_through_point (b : ℚ) : 
  (b * 3 - (b + 2) * (-5) = b - 3) → b = -13/7 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2874_287417


namespace NUMINAMATH_CALUDE_y_squared_equals_zx_sufficient_not_necessary_l2874_287402

-- Define a function to check if three numbers form an arithmetic sequence
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

-- Define the theorem
theorem y_squared_equals_zx_sufficient_not_necessary 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z) → y^2 = z*x) ∧
  ¬(y^2 = z*x → is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z)) :=
by sorry

end NUMINAMATH_CALUDE_y_squared_equals_zx_sufficient_not_necessary_l2874_287402


namespace NUMINAMATH_CALUDE_smallest_value_of_complex_sum_l2874_287494

theorem smallest_value_of_complex_sum (a b c d : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_omega_power : ω^4 = 1)
  (h_omega_not_one : ω ≠ 1) :
  ∃ (x y z w : ℤ), 
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
    ∀ (p q r s : ℤ), p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    Complex.abs (x + y*ω + z*ω^2 + w*ω^3) ≤ Complex.abs (p + q*ω + r*ω^2 + s*ω^3) ∧
    Complex.abs (x + y*ω + z*ω^2 + w*ω^3) = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_complex_sum_l2874_287494


namespace NUMINAMATH_CALUDE_luke_total_points_l2874_287484

/-- 
Given that Luke gained 11 points in each round and played 14 rounds,
prove that the total points he scored is 154.
-/
theorem luke_total_points : 
  let points_per_round : ℕ := 11
  let number_of_rounds : ℕ := 14
  points_per_round * number_of_rounds = 154 := by sorry

end NUMINAMATH_CALUDE_luke_total_points_l2874_287484


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_78_l2874_287458

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the point J (intersection of angle bisectors)
def J : ℝ × ℝ := sorry

-- Define the condition that PQR has positive integer side lengths
def has_positive_integer_sides (t : Triangle) : Prop :=
  ∃ (a b c : ℕ+), 
    dist t.P t.Q = a ∧ 
    dist t.Q t.R = b ∧ 
    dist t.R t.P = c

-- Define the condition that PQR is isosceles with PQ = PR
def is_isosceles (t : Triangle) : Prop :=
  dist t.P t.Q = dist t.P t.R

-- Define the condition that J is on the angle bisectors of ∠Q and ∠R
def J_on_angle_bisectors (t : Triangle) : Prop :=
  sorry

-- Define the condition that QJ = 10
def QJ_equals_10 (t : Triangle) : Prop :=
  dist t.Q J = 10

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  dist t.P t.Q + dist t.Q t.R + dist t.R t.P

-- Theorem statement
theorem smallest_perimeter_is_78 :
  ∀ t : Triangle,
    has_positive_integer_sides t →
    is_isosceles t →
    J_on_angle_bisectors t →
    QJ_equals_10 t →
    ∀ t' : Triangle,
      has_positive_integer_sides t' →
      is_isosceles t' →
      J_on_angle_bisectors t' →
      QJ_equals_10 t' →
      perimeter t ≤ perimeter t' →
      perimeter t = 78 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_78_l2874_287458


namespace NUMINAMATH_CALUDE_exists_integer_square_one_l2874_287474

theorem exists_integer_square_one : ∃ x : ℤ, x^2 = 1 := by sorry

end NUMINAMATH_CALUDE_exists_integer_square_one_l2874_287474


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2874_287475

theorem reciprocal_sum_theorem (a b c : ℕ+) : 
  a < b ∧ b < c → 
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1 → 
  (a : ℕ) + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2874_287475


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l2874_287473

theorem positive_integer_solutions_count : 
  (Finset.filter (fun (x : ℕ × ℕ × ℕ × ℕ) => x.1 + x.2.1 + x.2.2.1 + x.2.2.2 = 10) (Finset.product (Finset.range 10) (Finset.product (Finset.range 10) (Finset.product (Finset.range 10) (Finset.range 10))))).card = 84 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l2874_287473


namespace NUMINAMATH_CALUDE_one_sixth_star_neg_one_l2874_287405

-- Define the ※ operation for rational numbers
def star_op (m n : ℚ) : ℚ := (3*m + n) * (3*m - n) + n

-- State the theorem
theorem one_sixth_star_neg_one :
  star_op (1/6) (-1) = -7/4 := by sorry

end NUMINAMATH_CALUDE_one_sixth_star_neg_one_l2874_287405


namespace NUMINAMATH_CALUDE_tan_sum_pi_eighths_l2874_287446

theorem tan_sum_pi_eighths : Real.tan (π / 8) + Real.tan (3 * π / 8) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_eighths_l2874_287446


namespace NUMINAMATH_CALUDE_seven_factorial_divisors_l2874_287451

/-- The number of positive divisors of n! -/
def num_divisors_factorial (n : ℕ) : ℕ := sorry

/-- 7! has 60 positive divisors -/
theorem seven_factorial_divisors : num_divisors_factorial 7 = 60 := by sorry

end NUMINAMATH_CALUDE_seven_factorial_divisors_l2874_287451


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2874_287486

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2874_287486


namespace NUMINAMATH_CALUDE_multiple_between_factorials_l2874_287411

theorem multiple_between_factorials (n : ℕ) (h : n ≥ 4) :
  ∃ k : ℕ, n.factorial < k * n^3 ∧ k * n^3 < (n + 1).factorial := by
  sorry

end NUMINAMATH_CALUDE_multiple_between_factorials_l2874_287411


namespace NUMINAMATH_CALUDE_count_integer_segments_specific_triangle_l2874_287441

/-- Represents a right triangle ABC with integer leg lengths -/
structure RightTriangle where
  ab : ℕ  -- Length of leg AB
  bc : ℕ  -- Length of leg BC

/-- Calculates the number of distinct integer lengths of line segments 
    that can be drawn from vertex B to a point on hypotenuse AC -/
def count_integer_segments (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem count_integer_segments_specific_triangle : 
  let t : RightTriangle := { ab := 20, bc := 21 }
  count_integer_segments t = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_segments_specific_triangle_l2874_287441


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l2874_287468

/-- Represents the systematic sampling of students for a dental health check. -/
def systematicSampling (totalStudents : ℕ) (sampleSize : ℕ) (interval : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => start + i * interval)

/-- Theorem stating that the systematic sampling with given parameters results in the expected list of student numbers. -/
theorem systematic_sampling_result :
  systematicSampling 50 5 10 6 = [6, 16, 26, 36, 46] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l2874_287468


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l2874_287455

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l2874_287455


namespace NUMINAMATH_CALUDE_novel_reading_time_difference_l2874_287462

/-- The number of pages in the novel -/
def pages : ℕ := 760

/-- The time in seconds Bob takes to read one page -/
def bob_time : ℕ := 45

/-- The time in seconds Chandra takes to read one page -/
def chandra_time : ℕ := 30

/-- The difference in reading time between Bob and Chandra for the entire novel -/
def reading_time_difference : ℕ := pages * bob_time - pages * chandra_time

theorem novel_reading_time_difference :
  reading_time_difference = 11400 := by
  sorry

end NUMINAMATH_CALUDE_novel_reading_time_difference_l2874_287462


namespace NUMINAMATH_CALUDE_division_37_by_8_l2874_287459

theorem division_37_by_8 (A B : ℕ) : 37 = 8 * A + B ∧ B < 8 → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_37_by_8_l2874_287459


namespace NUMINAMATH_CALUDE_hyperbola_point_comparison_l2874_287476

theorem hyperbola_point_comparison 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2023 / x₁) 
  (h2 : y₂ = 2023 / x₂) 
  (h3 : y₁ > y₂) 
  (h4 : y₂ > 0) : 
  x₁ < x₂ := by
sorry

end NUMINAMATH_CALUDE_hyperbola_point_comparison_l2874_287476


namespace NUMINAMATH_CALUDE_move_right_2_units_l2874_287489

/-- Moving a point to the right in a 2D coordinate system -/
def move_right (x y dx : ℝ) : ℝ × ℝ :=
  (x + dx, y)

theorem move_right_2_units :
  let A : ℝ × ℝ := (1, 2)
  let A' : ℝ × ℝ := move_right A.1 A.2 2
  A' = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_move_right_2_units_l2874_287489


namespace NUMINAMATH_CALUDE_min_value_theorem_l2874_287457

theorem min_value_theorem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧ 
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → 1/a' + 4/b' + 9/c' ≥ 9) ∧
  (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 1/a' + 4/b' + 9/c' = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2874_287457


namespace NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l2874_287456

-- Define the point type
variable {Point : Type*}

-- Define the distance function
variable (dist : Point → Point → ℝ)

-- Define the volume function for tetrahedrons
variable (volume : Point → Point → Point → Point → ℝ)

-- Theorem statement
theorem tetrahedron_volume_ratio
  (A B C D B' C' D' : Point) :
  volume A B C D / volume A B' C' D' =
  (dist A B * dist A C * dist A D) / (dist A B' * dist A C' * dist A D') :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l2874_287456


namespace NUMINAMATH_CALUDE_exponent_for_28_decimal_places_l2874_287442

def base : ℝ := 10^4 * 3.456789

theorem exponent_for_28_decimal_places :
  ∀ n : ℕ, (∃ m : ℕ, base^n * 10^28 = m ∧ m < base^n * 10^29) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_exponent_for_28_decimal_places_l2874_287442


namespace NUMINAMATH_CALUDE_bricks_A_is_40_l2874_287471

/-- Represents the number of bricks of type A -/
def bricks_A : ℕ := sorry

/-- Represents the number of bricks of type B -/
def bricks_B : ℕ := sorry

/-- The number of bricks of type B is half the number of bricks of type A -/
axiom half_relation : bricks_B = bricks_A / 2

/-- The total number of bricks of type A and B is 60 -/
axiom total_bricks : bricks_A + bricks_B = 60

/-- Theorem stating that the number of bricks of type A is 40 -/
theorem bricks_A_is_40 : bricks_A = 40 := by sorry

end NUMINAMATH_CALUDE_bricks_A_is_40_l2874_287471


namespace NUMINAMATH_CALUDE_equation_solution_l2874_287418

theorem equation_solution : 
  ∃ (x : ℚ), x ≠ 1 ∧ x ≠ (1/2 : ℚ) ∧ (x / (x - 1) = 3 / (2*x - 2) - 2) ∧ x = (7/6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2874_287418


namespace NUMINAMATH_CALUDE_correct_operation_l2874_287440

theorem correct_operation (a : ℝ) : 2 * a^3 - a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2874_287440


namespace NUMINAMATH_CALUDE_apple_purchase_problem_l2874_287453

theorem apple_purchase_problem (x : ℕ) : 
  (12 : ℚ) / x - (12 : ℚ) / (x + 2) = 1 / 12 → x + 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_problem_l2874_287453


namespace NUMINAMATH_CALUDE_product_difference_sum_problem_l2874_287464

theorem product_difference_sum_problem : 
  ∃ (a b : ℕ+), (a * b = 18) ∧ (max a b - min a b = 3) → (a + b = 9) :=
by sorry

end NUMINAMATH_CALUDE_product_difference_sum_problem_l2874_287464


namespace NUMINAMATH_CALUDE_range_of_m_l2874_287444

/-- The range of m that satisfies the given conditions -/
theorem range_of_m (m : ℝ) : m ≥ 9 ↔ 
  (∀ x : ℝ, (|1 - x| > 2 → (x^2 - 2*x + 1 - m^2 > 0))) ∧ 
  (∃ x : ℝ, |1 - x| > 2 ∧ x^2 - 2*x + 1 - m^2 ≤ 0) ∧
  m > 0 :=
by sorry


end NUMINAMATH_CALUDE_range_of_m_l2874_287444


namespace NUMINAMATH_CALUDE_janes_breakfast_l2874_287429

theorem janes_breakfast (b m : ℕ) : 
  b + m = 7 →
  (90 * b + 40 * m) % 100 = 0 →
  b = 4 :=
by sorry

end NUMINAMATH_CALUDE_janes_breakfast_l2874_287429


namespace NUMINAMATH_CALUDE_paper_I_maximum_marks_l2874_287482

theorem paper_I_maximum_marks :
  ∀ (max_marks passing_mark secured_marks deficit : ℕ),
    passing_mark = (max_marks * 40) / 100 →
    secured_marks = 40 →
    deficit = 20 →
    passing_mark = secured_marks + deficit →
    max_marks = 150 := by
  sorry

end NUMINAMATH_CALUDE_paper_I_maximum_marks_l2874_287482


namespace NUMINAMATH_CALUDE_total_difference_l2874_287485

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def correct_discount : ℝ := 0.25
def charlie_discount : ℝ := 0.15

def anne_total : ℝ := original_price * (1 + sales_tax_rate) * (1 - correct_discount)
def ben_total : ℝ := original_price * (1 - correct_discount) * (1 + sales_tax_rate)
def charlie_total : ℝ := original_price * (1 - charlie_discount) * (1 + sales_tax_rate)

theorem total_difference : anne_total - ben_total - charlie_total = -12.96 := by
  sorry

end NUMINAMATH_CALUDE_total_difference_l2874_287485


namespace NUMINAMATH_CALUDE_not_monotone_decreasing_if_f2_gt_f1_l2874_287434

theorem not_monotone_decreasing_if_f2_gt_f1 
  (f : ℝ → ℝ) (h : f 2 > f 1) : 
  ¬(∀ x y : ℝ, x ≤ y → f x ≥ f y) := by
  sorry

end NUMINAMATH_CALUDE_not_monotone_decreasing_if_f2_gt_f1_l2874_287434


namespace NUMINAMATH_CALUDE_negation_equivalence_l2874_287401

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, Real.exp x - x - 1 < 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2874_287401


namespace NUMINAMATH_CALUDE_same_grade_probability_l2874_287497

-- Define the number of student volunteers in each grade
def grade_A_volunteers : ℕ := 240
def grade_B_volunteers : ℕ := 160
def grade_C_volunteers : ℕ := 160

-- Define the total number of student volunteers
def total_volunteers : ℕ := grade_A_volunteers + grade_B_volunteers + grade_C_volunteers

-- Define the number of students to be selected using stratified sampling
def selected_students : ℕ := 7

-- Define the number of students to be chosen for sanitation work
def sanitation_workers : ℕ := 2

-- Define the function to calculate the number of students selected from each grade
def students_per_grade (grade_volunteers : ℕ) : ℕ :=
  (grade_volunteers * selected_students) / total_volunteers

-- Theorem: The probability of selecting 2 students from the same grade is 5/21
theorem same_grade_probability :
  (students_per_grade grade_A_volunteers) * (students_per_grade grade_A_volunteers - 1) / 2 +
  (students_per_grade grade_B_volunteers) * (students_per_grade grade_B_volunteers - 1) / 2 +
  (students_per_grade grade_C_volunteers) * (students_per_grade grade_C_volunteers - 1) / 2 =
  5 * (selected_students * (selected_students - 1) / 2) / 21 :=
by sorry

end NUMINAMATH_CALUDE_same_grade_probability_l2874_287497


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2874_287465

theorem two_numbers_problem (x y : ℝ) : 
  x^2 + y^2 = 45/4 ∧ x - y = x * y → 
  (x = -3 ∧ y = 3/2) ∨ (x = -3/2 ∧ y = 3) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2874_287465


namespace NUMINAMATH_CALUDE_total_selling_price_proof_l2874_287467

def calculate_selling_price (cost : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost + cost * profit_percentage / 100

theorem total_selling_price_proof (cost_A cost_B cost_C : ℕ)
  (profit_percentage_A profit_percentage_B profit_percentage_C : ℕ)
  (h1 : cost_A = 400)
  (h2 : cost_B = 600)
  (h3 : cost_C = 800)
  (h4 : profit_percentage_A = 40)
  (h5 : profit_percentage_B = 35)
  (h6 : profit_percentage_C = 25) :
  calculate_selling_price cost_A profit_percentage_A +
  calculate_selling_price cost_B profit_percentage_B +
  calculate_selling_price cost_C profit_percentage_C = 2370 :=
by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_proof_l2874_287467


namespace NUMINAMATH_CALUDE_evaluate_eight_to_nine_thirds_l2874_287424

theorem evaluate_eight_to_nine_thirds : 8^(9/3) = 512 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_eight_to_nine_thirds_l2874_287424


namespace NUMINAMATH_CALUDE_expression_value_l2874_287460

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)    -- absolute value of m is 2
  : m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2874_287460


namespace NUMINAMATH_CALUDE_absolute_difference_sequence_l2874_287490

/-- Given three non-negative real numbers x, y, z, where z = 1, and after n steps of taking pairwise
    absolute differences, the sequence stabilizes with x_n = x, y_n = y, z_n = z, 
    then (x, y) = (0, 1) or (1, 0). -/
theorem absolute_difference_sequence (x y z : ℝ) (n : ℕ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z = 1 →
  (∃ (x_seq y_seq z_seq : ℕ → ℝ),
    (∀ k, k < n → 
      x_seq (k+1) = |x_seq k - y_seq k| ∧
      y_seq (k+1) = |y_seq k - z_seq k| ∧
      z_seq (k+1) = |z_seq k - x_seq k|) ∧
    x_seq 0 = x ∧ y_seq 0 = y ∧ z_seq 0 = z ∧
    x_seq n = x ∧ y_seq n = y ∧ z_seq n = z) →
  (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_absolute_difference_sequence_l2874_287490


namespace NUMINAMATH_CALUDE_lcm_18_24_30_l2874_287422

theorem lcm_18_24_30 : Nat.lcm 18 (Nat.lcm 24 30) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_30_l2874_287422


namespace NUMINAMATH_CALUDE_unsprinkled_bricks_count_l2874_287488

/-- Represents a rectangular solid pile of bricks -/
structure BrickPile where
  length : Nat
  width : Nat
  height : Nat

/-- Calculates the number of bricks not sprinkled with lime water -/
def unsprinkledBricks (pile : BrickPile) : Nat :=
  pile.length * pile.width * pile.height - 
  (pile.length - 2) * (pile.width - 2) * (pile.height - 2)

/-- Theorem stating that the number of unsprinkled bricks in a 30x20x10 pile is 4032 -/
theorem unsprinkled_bricks_count :
  let pile : BrickPile := { length := 30, width := 20, height := 10 }
  unsprinkledBricks pile = 4032 := by
  sorry

end NUMINAMATH_CALUDE_unsprinkled_bricks_count_l2874_287488


namespace NUMINAMATH_CALUDE_plane_distance_l2874_287481

/-- Proves that a plane flying east at 300 km/h and west at 400 km/h for a total of 7 hours travels 1200 km from the airport -/
theorem plane_distance (speed_east speed_west total_time : ℝ) 
  (h_speed_east : speed_east = 300)
  (h_speed_west : speed_west = 400)
  (h_total_time : total_time = 7) :
  ∃ (time_east time_west distance : ℝ),
    time_east + time_west = total_time ∧
    speed_east * time_east = distance ∧
    speed_west * time_west = distance ∧
    distance = 1200 := by
  sorry

end NUMINAMATH_CALUDE_plane_distance_l2874_287481


namespace NUMINAMATH_CALUDE_rals_current_age_l2874_287469

/-- Given that Ral is three times as old as Suri, and in 6 years Suri's age will be 25,
    prove that Ral's current age is 57 years. -/
theorem rals_current_age (suri_age suri_future_age ral_age : ℕ) 
    (h1 : ral_age = 3 * suri_age)
    (h2 : suri_future_age = suri_age + 6)
    (h3 : suri_future_age = 25) : 
  ral_age = 57 := by
  sorry

end NUMINAMATH_CALUDE_rals_current_age_l2874_287469


namespace NUMINAMATH_CALUDE_travis_cereal_cost_l2874_287413

/-- The amount Travis spends on cereal in a year -/
def cereal_cost (boxes_per_week : ℕ) (cost_per_box : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (boxes_per_week * weeks_per_year : ℚ) * cost_per_box

/-- Theorem: Travis spends $312.00 on cereal in a year -/
theorem travis_cereal_cost :
  cereal_cost 2 3 52 = 312 := by
  sorry

#eval cereal_cost 2 3 52

end NUMINAMATH_CALUDE_travis_cereal_cost_l2874_287413
