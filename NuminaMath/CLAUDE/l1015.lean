import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l1015_101502

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x * f y - f (x * y)) / 3 = x + y + 2

/-- The main theorem stating that the function f(x) = x + 3 is the unique solution -/
theorem unique_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → ∀ x : ℝ, f x = x + 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1015_101502


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1015_101596

theorem arithmetic_expression_evaluation : (8 * 6) - (4 / 2) = 46 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1015_101596


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l1015_101532

/-- Given a rectangle formed from a rope of length 100 cm with shorter sides of 22 cm each,
    prove that the length of each longer side is 28 cm. -/
theorem rectangle_longer_side_length (total_length : ℝ) (short_side : ℝ) (long_side : ℝ) :
  total_length = 100 ∧ short_side = 22 →
  2 * short_side + 2 * long_side = total_length →
  long_side = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l1015_101532


namespace NUMINAMATH_CALUDE_inequality_proof_l1015_101583

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1015_101583


namespace NUMINAMATH_CALUDE_residue_13_2045_mod_19_l1015_101562

theorem residue_13_2045_mod_19 : (13 ^ 2045 : ℕ) % 19 = 9 := by sorry

end NUMINAMATH_CALUDE_residue_13_2045_mod_19_l1015_101562


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l1015_101550

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l1015_101550


namespace NUMINAMATH_CALUDE_reciprocal_problem_l1015_101534

theorem reciprocal_problem (x : ℝ) (h : 6 * x = 12) : 150 * (1 / x) = 75 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l1015_101534


namespace NUMINAMATH_CALUDE_negation_equivalence_l1015_101580

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x < 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1015_101580


namespace NUMINAMATH_CALUDE_sine_double_angle_special_l1015_101514

theorem sine_double_angle_special (α : Real) : 
  α ∈ Set.Ioo 0 (Real.pi / 2) → 
  Real.cos (α + Real.pi / 6) = 3 / 5 → 
  Real.sin (2 * α + Real.pi / 3) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sine_double_angle_special_l1015_101514


namespace NUMINAMATH_CALUDE_quadratic_solution_l1015_101524

theorem quadratic_solution : ∃! x : ℝ, x > 0 ∧ 3 * x^2 + 13 * x - 10 = 0 :=
by
  use 2/3
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1015_101524


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l1015_101515

/-- The cost of one dozen pens given the cost of one pen and the ratio of pen to pencil cost -/
theorem cost_of_dozen_pens 
  (cost_of_one_pen : ℕ) 
  (ratio_pen_to_pencil : ℚ) 
  (h1 : cost_of_one_pen = 65) 
  (h2 : ratio_pen_to_pencil = 5 / 1) : 
  12 * cost_of_one_pen = 780 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l1015_101515


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1015_101539

/-- The polynomial f(x) = x^5 - 5x^4 + 8x^3 + 25x^2 - 14x - 40 -/
def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 8*x^3 + 25*x^2 - 14*x - 40

/-- The remainder when f(x) is divided by (x-2) -/
def remainder : ℝ := f 2

theorem polynomial_remainder : remainder = 48 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1015_101539


namespace NUMINAMATH_CALUDE_associate_professor_pencils_l1015_101574

theorem associate_professor_pencils 
  (total_people : ℕ) 
  (total_pencils : ℕ) 
  (total_charts : ℕ) 
  (associate_profs : ℕ) 
  (assistant_profs : ℕ) 
  (associate_prof_charts : ℕ) 
  (assistant_prof_pencils : ℕ) 
  (assistant_prof_charts : ℕ) :
  total_people = 6 →
  total_pencils = 7 →
  total_charts = 11 →
  associate_profs + assistant_profs = total_people →
  associate_prof_charts = 1 →
  assistant_prof_pencils = 1 →
  assistant_prof_charts = 2 →
  ∃ (associate_prof_pencils : ℕ),
    associate_prof_pencils * associate_profs + assistant_prof_pencils * assistant_profs = total_pencils ∧
    associate_prof_charts * associate_profs + assistant_prof_charts * assistant_profs = total_charts ∧
    associate_prof_pencils = 2 :=
by sorry

end NUMINAMATH_CALUDE_associate_professor_pencils_l1015_101574


namespace NUMINAMATH_CALUDE_matthew_initial_crackers_l1015_101544

/-- The number of crackers Matthew gave to each friend -/
def crackers_per_friend : ℕ := 2

/-- The number of friends Matthew gave crackers to -/
def number_of_friends : ℕ := 4

/-- The total number of crackers Matthew gave away -/
def total_crackers_given : ℕ := crackers_per_friend * number_of_friends

/-- Theorem stating that Matthew had at least 8 crackers initially -/
theorem matthew_initial_crackers :
  ∃ (initial_crackers : ℕ), initial_crackers ≥ total_crackers_given ∧ initial_crackers ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_matthew_initial_crackers_l1015_101544


namespace NUMINAMATH_CALUDE_chess_matches_to_reach_target_percentage_l1015_101597

theorem chess_matches_to_reach_target_percentage 
  (initial_matches : ℕ) 
  (initial_wins : ℕ) 
  (target_percentage : ℚ) : 
  initial_matches = 20 → 
  initial_wins = 19 → 
  target_percentage = 96/100 → 
  ∃ (additional_matches : ℕ), 
    additional_matches = 5 ∧ 
    (initial_wins + additional_matches) / (initial_matches + additional_matches) = target_percentage :=
by sorry

end NUMINAMATH_CALUDE_chess_matches_to_reach_target_percentage_l1015_101597


namespace NUMINAMATH_CALUDE_square_roots_problem_l1015_101512

theorem square_roots_problem (x m : ℝ) : 
  x > 0 ∧ 
  (m + 3)^2 = x ∧ 
  (2*m - 15)^2 = x ∧ 
  m + 3 ≠ 2*m - 15 → 
  x = 49 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1015_101512


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1015_101551

-- Define the triangle vertices
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 4)
def C : ℝ × ℝ := (-2, 4)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the line equation 4x + 3y + m = 0
def line_equation (x y m : ℝ) : Prop := 4 * x + 3 * y + m = 0

-- Define the circumcircle equation (x-3)^2 + (y-4)^2 = 25
def circumcircle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

-- Define the chord length
def chord_length : ℝ := 6

theorem triangle_abc_properties :
  (dot_product AB AC = 0) ∧ 
  (∃ m : ℝ, (m = -4 ∨ m = -44) ∧
    ∃ x y : ℝ, line_equation x y m ∧ circumcircle_equation x y) :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1015_101551


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l1015_101510

/-- The number of terms with integer exponents in the expansion of (√x + 1/(2∛x))^n -/
def integer_exponent_terms (n : ℕ) : ℕ :=
  (Finset.filter (fun r => (2 * n - 3 * r) % 3 = 0) (Finset.range (n + 1))).card

/-- The coefficients of the first three terms in the expansion -/
def first_three_coeffs (n : ℕ) : Fin 3 → ℚ
  | 0 => 1
  | 1 => n / 2
  | 2 => n * (n - 1) / 8

/-- The condition that the first three coefficients form an arithmetic sequence -/
def arithmetic_sequence_condition (n : ℕ) : Prop :=
  2 * (first_three_coeffs n 1) = (first_three_coeffs n 0) + (first_three_coeffs n 2)

theorem binomial_expansion_theorem (n : ℕ) :
  arithmetic_sequence_condition n → integer_exponent_terms n = 3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l1015_101510


namespace NUMINAMATH_CALUDE_right_triangle_has_three_altitudes_l1015_101570

/-- A triangle is a geometric figure with three vertices and three sides. -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side or its extension. -/
def Altitude (t : Triangle) (v : Fin 3) : Set (ℝ × ℝ) :=
  sorry

/-- A right triangle is a triangle with one right angle. -/
def IsRightTriangle (t : Triangle) : Prop :=
  sorry

/-- The number of altitudes in a triangle. -/
def NumberOfAltitudes (t : Triangle) : ℕ :=
  sorry

/-- Theorem: A right triangle has three altitudes. -/
theorem right_triangle_has_three_altitudes (t : Triangle) :
  IsRightTriangle t → NumberOfAltitudes t = 3 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_has_three_altitudes_l1015_101570


namespace NUMINAMATH_CALUDE_maciek_purchase_l1015_101561

/-- The cost of a pack of pretzels in dollars -/
def pretzel_cost : ℚ := 4

/-- The cost of a pack of chips in dollars -/
def chip_cost : ℚ := pretzel_cost * (1 + 3/4)

/-- The total amount Maciek spent in dollars -/
def total_spent : ℚ := 22

/-- The number of packets of each type (chips and pretzels) that Maciek bought -/
def num_packets : ℚ := total_spent / (pretzel_cost + chip_cost)

theorem maciek_purchase : num_packets = 2 := by
  sorry

end NUMINAMATH_CALUDE_maciek_purchase_l1015_101561


namespace NUMINAMATH_CALUDE_soccer_team_subjects_l1015_101517

theorem soccer_team_subjects (total : ℕ) (history : ℕ) (both : ℕ) (geography : ℕ) : 
  total = 18 → 
  history = 10 → 
  both = 6 → 
  geography = total - (history - both) → 
  geography = 14 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_subjects_l1015_101517


namespace NUMINAMATH_CALUDE_D_72_eq_45_l1015_101582

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where order matters. -/
def D (n : ℕ+) : ℕ := sorry

/-- Theorem stating that D(72) is equal to 45 -/
theorem D_72_eq_45 : D 72 = 45 := by sorry

end NUMINAMATH_CALUDE_D_72_eq_45_l1015_101582


namespace NUMINAMATH_CALUDE_expression_not_constant_l1015_101520

theorem expression_not_constant : 
  ¬∀ (x y : ℝ), x ≠ -1 → x ≠ 3 → y ≠ -1 → y ≠ 3 → 
  (3*x^2 + 4*x - 5) / ((x+1)*(x-3)) - (8 + x) / ((x+1)*(x-3)) = 
  (3*y^2 + 4*y - 5) / ((y+1)*(y-3)) - (8 + y) / ((y+1)*(y-3)) :=
by sorry

end NUMINAMATH_CALUDE_expression_not_constant_l1015_101520


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1015_101527

/-- Given that (x-1)^2 + |y+2| = 0, prove that 3/2*x^2*y - (x^2*y - 3*(2*x*y - x^2*y) - x*y) = -9 -/
theorem simplify_and_evaluate (x y : ℝ) (h : (x - 1)^2 + |y + 2| = 0) :
  3/2 * x^2 * y - (x^2 * y - 3 * (2 * x * y - x^2 * y) - x * y) = -9 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1015_101527


namespace NUMINAMATH_CALUDE_circle_E_equation_line_circle_disjoint_l1015_101541

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define point D
def point_D : ℝ × ℝ := (-2, 0)

-- Define line l passing through D with slope k
def line_l (k x y : ℝ) : Prop := y = k * (x + 2)

-- Theorem for the equation of circle E
theorem circle_E_equation : ∀ x y : ℝ,
  (∃ k : ℝ, line_l k x y) →
  ((x + 1)^2 + (y - 2)^2 = 5) ↔ 
  (∃ t : ℝ, x = -2 * (1 - t) + 0 * t ∧ y = 0 * (1 - t) + 4 * t) :=
sorry

-- Theorem for the condition of line l and circle C being disjoint
theorem line_circle_disjoint : ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y → ¬circle_C x y) ↔ k < 3/4 :=
sorry

end NUMINAMATH_CALUDE_circle_E_equation_line_circle_disjoint_l1015_101541


namespace NUMINAMATH_CALUDE_tangent_line_at_1_0_l1015_101554

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_1_0 :
  let p : ℝ × ℝ := (1, 0)
  let m : ℝ := f' p.1
  let tangent_line (x : ℝ) : ℝ := m * (x - p.1) + p.2
  (∀ x, tangent_line x = x - 1) ∧ f p.1 = p.2 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_at_1_0_l1015_101554


namespace NUMINAMATH_CALUDE_line_equation_l1015_101579

/-- A line passing through (2, 3) with slope -1 has equation x + y - 5 = 0 -/
theorem line_equation (x y : ℝ) :
  (∀ t : ℝ, x = 2 - t ∧ y = 3 + t) →  -- Parametric form of line through (2, 3) with slope -1
  x + y - 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l1015_101579


namespace NUMINAMATH_CALUDE_english_only_enrollment_l1015_101530

theorem english_only_enrollment (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 32)
  (h2 : both = 12)
  (h3 : german = 22)
  (h4 : total ≥ german)
  (h5 : german ≥ both) :
  total - german + both = 10 := by
  sorry

end NUMINAMATH_CALUDE_english_only_enrollment_l1015_101530


namespace NUMINAMATH_CALUDE_smallest_n_below_threshold_l1015_101567

/-- The number of boxes -/
def num_boxes : ℕ := 1005

/-- The probability of stopping at box n -/
def P (n : ℕ) : ℚ := 2 / (2 * n + 1)

/-- The threshold probability -/
def threshold : ℚ := 1 / num_boxes

theorem smallest_n_below_threshold :
  ∀ k : ℕ, k < num_boxes → P k ≥ threshold ∧
  P num_boxes < threshold ∧
  ∀ m : ℕ, m > num_boxes → P m < threshold :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_below_threshold_l1015_101567


namespace NUMINAMATH_CALUDE_school_students_count_l1015_101593

theorem school_students_count : ℕ :=
  let initial_bananas_per_student : ℕ := 2
  let initial_apples_per_student : ℕ := 1
  let initial_oranges_per_student : ℕ := 1
  let absent_students : ℕ := 420
  let final_bananas_per_student : ℕ := 6
  let final_apples_per_student : ℕ := 3
  let final_oranges_per_student : ℕ := 2

  have h1 : ∀ (S : ℕ), S * initial_bananas_per_student = (S - absent_students) * final_bananas_per_student →
    S = 840 :=
    sorry

  840

/- Proof omitted -/

end NUMINAMATH_CALUDE_school_students_count_l1015_101593


namespace NUMINAMATH_CALUDE_rock_splash_width_l1015_101553

theorem rock_splash_width 
  (num_pebbles num_rocks num_boulders : ℕ)
  (total_width pebble_splash_width boulder_splash_width : ℝ)
  (h1 : num_pebbles = 6)
  (h2 : num_rocks = 3)
  (h3 : num_boulders = 2)
  (h4 : total_width = 7)
  (h5 : pebble_splash_width = 1/4)
  (h6 : boulder_splash_width = 2)
  : (total_width - num_pebbles * pebble_splash_width - num_boulders * boulder_splash_width) / num_rocks = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_rock_splash_width_l1015_101553


namespace NUMINAMATH_CALUDE_linda_win_probability_is_two_thirty_first_l1015_101507

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a player in the game -/
inductive Player
| Sara
| Peter
| Linda

/-- The game state -/
structure GameState where
  currentPlayer : Player
  saraLastFlip : Option CoinFlip
  
/-- The result of a game round -/
inductive RoundResult
| Continue (newState : GameState)
| SaraWins
| LindaWins

/-- Simulates a single round of the game -/
def playRound (state : GameState) (flip : CoinFlip) : RoundResult := sorry

/-- Calculates the probability of Linda winning given the game rules -/
def lindaWinProbability : ℚ := sorry

/-- Theorem stating that the probability of Linda winning is 2/31 -/
theorem linda_win_probability_is_two_thirty_first :
  lindaWinProbability = 2 / 31 := by sorry

end NUMINAMATH_CALUDE_linda_win_probability_is_two_thirty_first_l1015_101507


namespace NUMINAMATH_CALUDE_helpers_count_l1015_101584

theorem helpers_count (pouches_per_pack : ℕ) (team_members : ℕ) (coaches : ℕ) (packs_bought : ℕ) :
  pouches_per_pack = 6 →
  team_members = 13 →
  coaches = 3 →
  packs_bought = 3 →
  (pouches_per_pack * packs_bought) - (team_members + coaches) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_helpers_count_l1015_101584


namespace NUMINAMATH_CALUDE_circle_centers_locus_l1015_101529

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 16

-- Define the property of being externally tangent to C₁
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

-- Define the property of being internally tangent to C₂
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 2)^2 + b^2 = (4 - r)^2

-- Define the locus equation
def locus_equation (a b : ℝ) : Prop := 84 * a^2 + 100 * b^2 - 168 * a - 441 = 0

-- State the theorem
theorem circle_centers_locus (a b : ℝ) :
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) ↔ locus_equation a b :=
sorry

end NUMINAMATH_CALUDE_circle_centers_locus_l1015_101529


namespace NUMINAMATH_CALUDE_prime_divisor_of_2p_minus_1_l1015_101577

theorem prime_divisor_of_2p_minus_1 (p : ℕ) (hp : Prime p) :
  ∀ q : ℕ, Prime q → q ∣ (2^p - 1) → q > p :=
by sorry

end NUMINAMATH_CALUDE_prime_divisor_of_2p_minus_1_l1015_101577


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l1015_101557

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m^2 - 1 = 0 ∧ x = 0) → m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l1015_101557


namespace NUMINAMATH_CALUDE_line_segment_lattice_points_l1015_101535

/-- The number of lattice points on a line segment --/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem stating that the number of lattice points on the given line segment is 3 --/
theorem line_segment_lattice_points :
  latticePointCount 5 23 47 297 = 3 := by sorry

end NUMINAMATH_CALUDE_line_segment_lattice_points_l1015_101535


namespace NUMINAMATH_CALUDE_unique_pair_no_real_solutions_l1015_101598

theorem unique_pair_no_real_solutions : 
  ∃! (b c : ℕ+), 
    (∀ x : ℝ, x^2 + 2*(b:ℝ)*x + (c:ℝ) ≠ 0) ∧ 
    (∀ x : ℝ, x^2 + 2*(c:ℝ)*x + (b:ℝ) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_no_real_solutions_l1015_101598


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_geq_neg_eight_l1015_101537

-- Define set A
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a ≥ 0}

-- Theorem statement
theorem intersection_nonempty_iff_a_geq_neg_eight :
  ∀ a : ℝ, (A ∩ B a).Nonempty ↔ a ≥ -8 := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_geq_neg_eight_l1015_101537


namespace NUMINAMATH_CALUDE_catering_weight_calculation_catering_weight_proof_l1015_101578

theorem catering_weight_calculation (silverware_weight : ℕ) (silverware_per_setting : ℕ)
  (plate_weight : ℕ) (plates_per_setting : ℕ) (num_tables : ℕ) (settings_per_table : ℕ)
  (backup_settings : ℕ) : ℕ :=
  let weight_per_setting := silverware_weight * silverware_per_setting + plate_weight * plates_per_setting
  let total_settings := num_tables * settings_per_table + backup_settings
  weight_per_setting * total_settings

theorem catering_weight_proof :
  catering_weight_calculation 4 3 12 2 15 8 20 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_catering_weight_calculation_catering_weight_proof_l1015_101578


namespace NUMINAMATH_CALUDE_rainy_days_count_l1015_101508

theorem rainy_days_count (n : ℤ) : 
  (∃ (R NR : ℤ),
    R + NR = 7 ∧ 
    n * R + 4 * NR = 26 ∧ 
    4 * NR - n * R = 14 ∧ 
    R ≥ 0 ∧ NR ≥ 0) → 
  (∃ (R : ℤ), R = 2 ∧ 
    (∃ (NR : ℤ), 
      R + NR = 7 ∧ 
      n * R + 4 * NR = 26 ∧ 
      4 * NR - n * R = 14 ∧ 
      R ≥ 0 ∧ NR ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_rainy_days_count_l1015_101508


namespace NUMINAMATH_CALUDE_evaluate_expression_l1015_101523

theorem evaluate_expression : (3 + 2) * (3^2 + 2^2) * (3^4 + 2^4) = 6255 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1015_101523


namespace NUMINAMATH_CALUDE_factorial_sum_equals_power_of_two_l1015_101511

theorem factorial_sum_equals_power_of_two (a b c : ℕ+) : 
  (Nat.factorial a.val + Nat.factorial b.val = 2^(Nat.factorial c.val)) ↔ 
  ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 2, 2)) := by
sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_power_of_two_l1015_101511


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1015_101509

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 - 2*x - 1 = 0) ↔ ((x - 1)^2 = 2) := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1015_101509


namespace NUMINAMATH_CALUDE_max_siskins_achievable_24_siskins_l1015_101548

/-- Represents a row of poles with siskins -/
structure PoleRow :=
  (num_poles : Nat)
  (occupied : Finset Nat)

/-- The rules for siskin movement on poles -/
def valid_configuration (pr : PoleRow) : Prop :=
  pr.num_poles = 25 ∧
  pr.occupied.card ≤ pr.num_poles ∧
  ∀ i ∈ pr.occupied, i ≤ pr.num_poles ∧
  ∀ i ∈ pr.occupied, (i + 1 ∉ pr.occupied ∨ i = pr.num_poles) ∧
                     (i - 1 ∉ pr.occupied ∨ i = 1)

/-- The theorem stating the maximum number of siskins -/
theorem max_siskins (pr : PoleRow) :
  valid_configuration pr → pr.occupied.card ≤ 24 :=
sorry

/-- The theorem stating that 24 siskins is achievable -/
theorem achievable_24_siskins :
  ∃ pr : PoleRow, valid_configuration pr ∧ pr.occupied.card = 24 :=
sorry

end NUMINAMATH_CALUDE_max_siskins_achievable_24_siskins_l1015_101548


namespace NUMINAMATH_CALUDE_triangle_tangent_range_l1015_101586

theorem triangle_tangent_range (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A ∧ A < π) (h5 : 0 < B ∧ B < π) (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π) (h8 : a^2 + b^2 + Real.sqrt 2 * a * b = c^2) :
  0 < Real.tan A * Real.tan (2 * B) ∧ Real.tan A * Real.tan (2 * B) < 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_range_l1015_101586


namespace NUMINAMATH_CALUDE_exists_player_reaching_all_l1015_101518

/-- Represents a tournament where every player plays against every other player once with no draws -/
structure Tournament (α : Type) :=
  (players : Set α)
  (defeated : α → α → Prop)
  (complete : ∀ a b : α, a ≠ b → (defeated a b ∨ defeated b a))
  (irreflexive : ∀ a : α, ¬ defeated a a)

/-- A player can reach another player within two steps of the defeated relation -/
def can_reach_in_two_steps {α : Type} (t : Tournament α) (a b : α) : Prop :=
  t.defeated a b ∨ ∃ c, t.defeated a c ∧ t.defeated c b

/-- The main theorem: there exists a player who can reach all others within two steps -/
theorem exists_player_reaching_all {α : Type} (t : Tournament α) :
  ∃ a : α, ∀ b : α, b ∈ t.players → a ≠ b → can_reach_in_two_steps t a b :=
sorry

end NUMINAMATH_CALUDE_exists_player_reaching_all_l1015_101518


namespace NUMINAMATH_CALUDE_line_plane_intersection_l1015_101528

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (intersects : Line → Plane → Prop)

-- State the theorem
theorem line_plane_intersection
  (a b : Line) (α : Plane)
  (h1 : parallel a α)
  (h2 : perpendicular b a) :
  intersects b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l1015_101528


namespace NUMINAMATH_CALUDE_gcd_lcm_product_18_42_l1015_101506

theorem gcd_lcm_product_18_42 : Nat.gcd 18 42 * Nat.lcm 18 42 = 756 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_18_42_l1015_101506


namespace NUMINAMATH_CALUDE_translation_result_l1015_101525

def point_translation (x y dx dy : ℝ) : ℝ × ℝ :=
  (x + dx, y - dy)

theorem translation_result :
  point_translation (-2) 3 3 1 = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l1015_101525


namespace NUMINAMATH_CALUDE_problem_solution_l1015_101571

theorem problem_solution (x y : ℝ) (hx_pos : x > 0) :
  (2/3 : ℝ) * x = (144/216 : ℝ) * (1/x) ∧ y * (1/x) = Real.sqrt x → x = 1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1015_101571


namespace NUMINAMATH_CALUDE_pythagorean_theorem_for_triples_scaled_right_triangle_is_pythagorean_l1015_101521

/-- Pythagorean numbers are positive integers that can be the lengths of the sides of a right triangle. -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_theorem_for_triples :
  ∀ (a b c : ℕ), isPythagoreanTriple a b c → a * a + b * b = c * c :=
sorry

theorem scaled_right_triangle_is_pythagorean :
  ∀ (a b c : ℕ), isPythagoreanTriple a b c → isPythagoreanTriple (2*a) (2*b) (2*c) :=
sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_for_triples_scaled_right_triangle_is_pythagorean_l1015_101521


namespace NUMINAMATH_CALUDE_correction_is_15x_l1015_101556

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "half-dollar" => 50
  | "nickel" => 5
  | _ => 0

/-- Calculates the correction needed for miscounted coins -/
def correction_needed (x : ℕ) : ℤ :=
  let quarter_dime_diff := (coin_value "quarter" - coin_value "dime") * (2 * x)
  let half_dollar_nickel_diff := (coin_value "half-dollar" - coin_value "nickel") * x
  quarter_dime_diff - half_dollar_nickel_diff

theorem correction_is_15x (x : ℕ) : correction_needed x = 15 * x := by
  sorry

end NUMINAMATH_CALUDE_correction_is_15x_l1015_101556


namespace NUMINAMATH_CALUDE_stating_election_cases_l1015_101555

/-- Represents the number of candidates for the election -/
def num_candidates : ℕ := 3

/-- Represents the number of positions to be filled -/
def num_positions : ℕ := 2

/-- 
Theorem stating that the number of ways to select a president and vice president 
from a group of three people, where one person cannot hold both positions, is equal to 6.
-/
theorem election_cases : 
  num_candidates * (num_candidates - 1) = 6 :=
sorry

end NUMINAMATH_CALUDE_stating_election_cases_l1015_101555


namespace NUMINAMATH_CALUDE_no_natural_power_pair_l1015_101546

theorem no_natural_power_pair : ¬∃ (x y : ℕ), 
  (∃ (k : ℕ), x^2 + x + 1 = y^k) ∧ 
  (∃ (m : ℕ), y^2 + y + 1 = x^m) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_power_pair_l1015_101546


namespace NUMINAMATH_CALUDE_x_y_relation_existence_of_k_l1015_101566

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 4
  | (n + 2) => 3 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 3 * y (n + 1) - y n

theorem x_y_relation (n : ℕ) : (x n)^2 - 5*(y n)^2 + 4 = 0 := by
  sorry

theorem existence_of_k (a b : ℕ) (h : a^2 - 5*b^2 + 4 = 0) :
  ∃ k : ℕ, x k = a ∧ y k = b := by
  sorry

end NUMINAMATH_CALUDE_x_y_relation_existence_of_k_l1015_101566


namespace NUMINAMATH_CALUDE_juice_consumption_l1015_101547

theorem juice_consumption (refrigerator pantry bought left : ℕ) 
  (h1 : refrigerator = 4)
  (h2 : pantry = 4)
  (h3 : bought = 5)
  (h4 : left = 10) :
  refrigerator + pantry + bought - left = 3 := by
  sorry

end NUMINAMATH_CALUDE_juice_consumption_l1015_101547


namespace NUMINAMATH_CALUDE_sin_minus_cos_with_tan_one_third_l1015_101581

theorem sin_minus_cos_with_tan_one_third 
  (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_with_tan_one_third_l1015_101581


namespace NUMINAMATH_CALUDE_normal_commute_time_l1015_101572

/-- A worker's commute scenario -/
structure WorkerCommute where
  normal_speed : ℝ
  normal_distance : ℝ
  normal_time : ℝ
  inclined_speed : ℝ
  inclined_distance : ℝ
  inclined_time : ℝ

/-- The conditions of the worker's commute -/
def commute_conditions (w : WorkerCommute) : Prop :=
  w.inclined_speed = 3 / 4 * w.normal_speed ∧
  w.inclined_distance = 5 / 4 * w.normal_distance ∧
  w.inclined_time = w.normal_time + 20 ∧
  w.normal_distance = w.normal_speed * w.normal_time ∧
  w.inclined_distance = w.inclined_speed * w.inclined_time

/-- The theorem stating that under the given conditions, the normal commute time is 30 minutes -/
theorem normal_commute_time (w : WorkerCommute) 
  (h : commute_conditions w) : w.normal_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_normal_commute_time_l1015_101572


namespace NUMINAMATH_CALUDE_log_equation_solution_l1015_101563

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log 216 / Real.log (2 * x) = x →
  x = 3 ∧ ¬∃ (n : ℕ), x = n^2 ∧ ¬∃ (n : ℕ), x = n^3 ∧ ∃ (n : ℕ), x = n := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1015_101563


namespace NUMINAMATH_CALUDE_valerie_stamps_l1015_101573

/-- The number of stamps Valerie needs for all her envelopes --/
def total_stamps : ℕ :=
  let thank_you_cards := 3
  let water_bill := 1
  let electric_bill := 2
  let internet_bill := 3
  let bills := water_bill + electric_bill + internet_bill
  let rebates := bills + 3
  let job_applications := 2 * rebates
  thank_you_cards + bills + 2 * rebates + job_applications

/-- Theorem stating that Valerie needs 33 stamps in total --/
theorem valerie_stamps : total_stamps = 33 := by
  sorry

end NUMINAMATH_CALUDE_valerie_stamps_l1015_101573


namespace NUMINAMATH_CALUDE_parabola_line_intersection_slope_l1015_101589

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- A line with slope k passing through the focus -/
def line (k x y : ℝ) : Prop := y = k * (x - 1)

/-- The distance between two points on the parabola -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ + x₂ + 2

theorem parabola_line_intersection_slope 
  (k : ℝ) 
  (h_k : k ≠ 0) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_A : parabola x₁ y₁) 
  (h_B : parabola x₂ y₂) 
  (h_l₁ : line k x₁ y₁) 
  (h_l₂ : line k x₂ y₂) 
  (h_dist : distance x₁ y₁ x₂ y₂ = 5) : 
  k = 2 ∨ k = -2 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_slope_l1015_101589


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1015_101587

/-- Given a quadratic equation 2x^2 - 1 = 6x, prove its coefficients -/
theorem quadratic_equation_coefficients :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 - 1 - 6 * x
  ∃ (a b c : ℝ), (∀ x, f x = a * x^2 + b * x + c) ∧ a = 2 ∧ b = -6 ∧ c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1015_101587


namespace NUMINAMATH_CALUDE_paint_intensity_problem_l1015_101591

theorem paint_intensity_problem (original_intensity new_intensity replacement_fraction : ℚ)
  (h1 : original_intensity = 15 / 100)
  (h2 : new_intensity = 30 / 100)
  (h3 : replacement_fraction = 3 / 2)
  : ∃ added_intensity : ℚ,
    added_intensity = 40 / 100 ∧
    (original_intensity * (1 / (1 + replacement_fraction)) + 
     added_intensity * (replacement_fraction / (1 + replacement_fraction)) = new_intensity) :=
by sorry

end NUMINAMATH_CALUDE_paint_intensity_problem_l1015_101591


namespace NUMINAMATH_CALUDE_steve_pages_per_month_l1015_101526

def days_in_month : ℕ := 30
def letter_frequency : ℕ := 3
def regular_letter_time : ℕ := 20
def time_per_page : ℕ := 10
def long_letter_time : ℕ := 80

theorem steve_pages_per_month :
  let regular_letters := days_in_month / letter_frequency
  let regular_pages := regular_letters * regular_letter_time / time_per_page
  let long_letter_pages := long_letter_time / (time_per_page * 2)
  regular_pages + long_letter_pages = 24 := by
sorry

end NUMINAMATH_CALUDE_steve_pages_per_month_l1015_101526


namespace NUMINAMATH_CALUDE_probability_two_face_cards_total_20_l1015_101513

-- Define the deck
def deck_size : ℕ := 52

-- Define the number of face cards
def face_cards : ℕ := 12

-- Define the value of a face card
def face_card_value : ℕ := 10

-- Define the theorem
theorem probability_two_face_cards_total_20 :
  (face_cards : ℚ) * (face_cards - 1) / (deck_size * (deck_size - 1)) = 11 / 221 :=
sorry

end NUMINAMATH_CALUDE_probability_two_face_cards_total_20_l1015_101513


namespace NUMINAMATH_CALUDE_reciprocal_equal_self_l1015_101552

theorem reciprocal_equal_self (x : ℝ) : x ≠ 0 ∧ x = 1 / x ↔ x = 1 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_reciprocal_equal_self_l1015_101552


namespace NUMINAMATH_CALUDE_early_arrival_l1015_101558

/-- A boy's journey to school -/
def school_journey (usual_time : ℕ) (speed_factor : ℚ) : Prop :=
  let new_time := (usual_time : ℚ) * (1 / speed_factor)
  (usual_time : ℚ) - new_time = 7

theorem early_arrival : school_journey 49 (7/6) := by
  sorry

end NUMINAMATH_CALUDE_early_arrival_l1015_101558


namespace NUMINAMATH_CALUDE_favorite_numbers_l1015_101545

def is_favorite_number (n : ℕ) : Prop :=
  100 < n ∧ n < 150 ∧
  n % 13 = 0 ∧
  n % 3 ≠ 0 ∧
  (n / 100 + (n / 10) % 10 + n % 10) % 4 = 0

theorem favorite_numbers :
  ∀ n : ℕ, is_favorite_number n ↔ n = 130 ∨ n = 143 :=
by sorry

end NUMINAMATH_CALUDE_favorite_numbers_l1015_101545


namespace NUMINAMATH_CALUDE_pen_price_theorem_l1015_101516

/-- Given the conditions of the pen and pencil purchase, prove the average price of a pen. -/
theorem pen_price_theorem (total_pens : ℕ) (total_pencils : ℕ) (total_cost : ℚ) (avg_pencil_price : ℚ) :
  total_pens = 30 →
  total_pencils = 75 →
  total_cost = 510 →
  avg_pencil_price = 2 →
  (total_cost - total_pencils * avg_pencil_price) / total_pens = 12 :=
by sorry

end NUMINAMATH_CALUDE_pen_price_theorem_l1015_101516


namespace NUMINAMATH_CALUDE_expression_simplification_l1015_101519

theorem expression_simplification (x y k : ℝ) 
  (hk : k ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = k * y) :
  (x - k / x) * (y + 1 / (k * y)) = (x^2 * k - k^3) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1015_101519


namespace NUMINAMATH_CALUDE_line_and_segment_properties_l1015_101568

-- Define the points and lines
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (1, -1)
def C : ℝ × ℝ := (0, 2)

def line_l (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line_m (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define the theorem
theorem line_and_segment_properties :
  -- Given conditions
  (∃ a : ℝ, A = (2, a) ∧ B = (a, -1)) →
  (∀ x y : ℝ, line_l x y ↔ ∃ t : ℝ, x = 2 * (1 - t) + t ∧ y = 1 * (1 - t) + (-1) * t) →
  (∀ x y : ℝ, line_l x y → line_m (x + 1) (y + 1)) →
  -- Conclusions
  (∀ x y : ℝ, line_l x y ↔ 2 * x - y - 3 = 0) ∧
  Real.sqrt 10 = Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_line_and_segment_properties_l1015_101568


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l1015_101594

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the foci of an ellipse -/
structure Foci (a b : ℝ) where
  left : Point
  right : Point
  h_ellipse : Ellipse a b

/-- Represents a triangle formed by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Defines an equilateral triangle -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Defines a line perpendicular to the x-axis passing through a point -/
def perpendicular_to_x_axis (p : Point) (A B : Point) : Prop := sorry

/-- Defines points on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse a b) : Prop := sorry

/-- Defines the eccentricity of an ellipse -/
def eccentricity (e : Ellipse a b) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_eccentricity_theorem 
  (a b : ℝ) 
  (e : Ellipse a b) 
  (f : Foci a b) 
  (A B : Point) 
  (t : Triangle) :
  perpendicular_to_x_axis f.right A B →
  on_ellipse A e →
  on_ellipse B e →
  t = Triangle.mk A B f.left →
  is_equilateral t →
  eccentricity e = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l1015_101594


namespace NUMINAMATH_CALUDE_set_equality_l1015_101504

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l1015_101504


namespace NUMINAMATH_CALUDE_cab_driver_income_proof_l1015_101500

def cab_driver_problem (day1 day3 day4 day5 average_income : ℕ) : Prop :=
  let total_income := 5 * average_income
  let known_income := day1 + day3 + day4 + day5
  let day2 := total_income - known_income
  (day1 = 300) ∧ (day3 = 750) ∧ (day4 = 400) ∧ (day5 = 500) ∧ (average_income = 420) →
  day2 = 150

theorem cab_driver_income_proof :
  cab_driver_problem 300 750 400 500 420 :=
by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_proof_l1015_101500


namespace NUMINAMATH_CALUDE_new_person_weight_is_68_l1015_101569

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 68 kg -/
theorem new_person_weight_is_68 :
  new_person_weight 6 3.5 47 = 68 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_68_l1015_101569


namespace NUMINAMATH_CALUDE_f_neg_one_eq_zero_f_is_even_x_range_l1015_101543

noncomputable section

variable (f : ℝ → ℝ)

-- Define the functional equation
axiom functional_eq : ∀ (x₁ x₂ : ℝ), x₁ ≠ 0 → x₂ ≠ 0 → f (x₁ * x₂) = f x₁ + f x₂

-- Define that f is increasing on (0, +∞)
axiom f_increasing : ∀ (x y : ℝ), 0 < x → x < y → f x < f y

-- Define the inequality condition
axiom f_inequality : ∀ (x : ℝ), f (2 * x - 1) < f x

-- Theorem 1: f(-1) = 0
theorem f_neg_one_eq_zero : f (-1) = 0 := by sorry

-- Theorem 2: f is an even function
theorem f_is_even : ∀ (x : ℝ), f (-x) = f x := by sorry

-- Theorem 3: Range of x
theorem x_range : ∀ (x : ℝ), (1/3 < x ∧ x < 1) ↔ (f (2*x - 1) < f x ∧ ∀ (y z : ℝ), 0 < y → y < z → f y < f z) := by sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_zero_f_is_even_x_range_l1015_101543


namespace NUMINAMATH_CALUDE_inequality_proof_l1015_101505

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 * b + a + b^2) * (a * b^2 + a^2 + b) > 9 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1015_101505


namespace NUMINAMATH_CALUDE_veronica_ring_removal_ways_l1015_101538

/-- Represents the number of rings on each finger --/
structure RingDistribution :=
  (little : Nat)
  (middle : Nat)
  (ring : Nat)

/-- Calculates the number of ways to remove rings given a distribution --/
def removalWays (dist : RingDistribution) (fixedOrderOnRingFinger : Bool) : Nat :=
  if fixedOrderOnRingFinger then
    dist.little * dist.middle
  else
    sorry

/-- The specific ring distribution in the problem --/
def veronicaRings : RingDistribution :=
  { little := 1, middle := 1, ring := 3 }

theorem veronica_ring_removal_ways :
  removalWays veronicaRings true = 20 := by sorry

end NUMINAMATH_CALUDE_veronica_ring_removal_ways_l1015_101538


namespace NUMINAMATH_CALUDE_circle_intersection_properties_l1015_101590

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Theorem statement
theorem circle_intersection_properties :
  -- 1. The equation of the line containing AB is x - y = 0
  (∀ (x y : ℝ), (x - y = 0) ↔ (∃ (t : ℝ), x = t * A.1 + (1 - t) * B.1 ∧ y = t * A.2 + (1 - t) * B.2)) ∧
  -- 2. The equation of the perpendicular bisector of AB is x + y - 1 = 0
  (∀ (x y : ℝ), (x + y - 1 = 0) ↔ ((x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2)) ∧
  -- 3. The maximum distance from a point P on O₂ to the line AB is (3√2)/2 + √5
  (∃ (P : ℝ × ℝ), circle_O2 P.1 P.2 ∧
    ∀ (Q : ℝ × ℝ), circle_O2 Q.1 Q.2 →
      abs ((Q.1 - Q.2) / Real.sqrt 2) ≤ (3 * Real.sqrt 2) / 2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_properties_l1015_101590


namespace NUMINAMATH_CALUDE_quadratic_function_ordering_l1015_101565

theorem quadratic_function_ordering (m y₁ y₂ y₃ : ℝ) : 
  m < -2 →
  y₁ = (m - 1)^2 - 2*(m - 1) →
  y₂ = m^2 - 2*m →
  y₃ = (m + 1)^2 - 2*(m + 1) →
  y₃ < y₂ ∧ y₂ < y₁ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_ordering_l1015_101565


namespace NUMINAMATH_CALUDE_phone_not_answered_probability_l1015_101549

theorem phone_not_answered_probability 
  (p1 p2 p3 p4 : ℝ) 
  (h1 : p1 = 0.1) 
  (h2 : p2 = 0.3) 
  (h3 : p3 = 0.4) 
  (h4 : p4 = 0.1) : 
  1 - (p1 + p2 + p3 + p4) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_phone_not_answered_probability_l1015_101549


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1015_101564

/-- Given a circle with equation x^2 + y^2 + 4x - 12y + 20 = 0, 
    the sum of the x and y coordinates of its center is 4 -/
theorem circle_center_coordinate_sum :
  ∀ (x y : ℝ), x^2 + y^2 + 4*x - 12*y + 20 = 0 →
  ∃ (h k : ℝ), (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = (x^2 + y^2 + 4*x - 12*y + 20)) ∧ 
                h + k = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1015_101564


namespace NUMINAMATH_CALUDE_inscribed_squares_inequality_l1015_101560

/-- Given a triangle ABC with sides a, b, and c, and inscribed squares with side lengths x, y, and z
    on sides BC, AC, and AB respectively, prove that (a/x) + (b/y) + (c/z) ≥ 3 + 2√3. -/
theorem inscribed_squares_inequality (a b c x y z : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_square_x : x ≤ b ∧ x ≤ c)
    (h_square_y : y ≤ c ∧ y ≤ a)
    (h_square_z : z ≤ a ∧ z ≤ b) :
  a / x + b / y + c / z ≥ 3 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_inequality_l1015_101560


namespace NUMINAMATH_CALUDE_find_m_l1015_101585

theorem find_m : ∃ m : ℕ, (1/5 : ℚ)^m * (1/4 : ℚ)^2 = 1/(10^4 : ℚ) ∧ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1015_101585


namespace NUMINAMATH_CALUDE_molecular_weight_CaCO3_is_100_l1015_101501

/-- The molecular weight of CaCO3 in grams per mole -/
def molecular_weight_CaCO3 : ℝ := 100

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 9

/-- The total molecular weight of the given number of moles in grams -/
def given_total_weight : ℝ := 900

/-- Theorem stating that the molecular weight of CaCO3 is 100 grams/mole -/
theorem molecular_weight_CaCO3_is_100 :
  molecular_weight_CaCO3 = given_total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_CaCO3_is_100_l1015_101501


namespace NUMINAMATH_CALUDE_not_square_difference_l1015_101595

-- Define the square difference formula
def square_difference (a b : ℝ → ℝ) : ℝ → ℝ := λ x => (a x)^2 - (b x)^2

-- Define the expression we want to prove doesn't fit the square difference formula
def expression : ℝ → ℝ := λ x => (x + 1) * (1 + x)

-- Theorem statement
theorem not_square_difference :
  ¬ ∃ (a b : ℝ → ℝ), ∀ x, expression x = square_difference a b x :=
sorry

end NUMINAMATH_CALUDE_not_square_difference_l1015_101595


namespace NUMINAMATH_CALUDE_number_of_green_balls_l1015_101503

/-- Given a total of 40 balls with red, blue, and green colors, where there are 11 blue balls
and the number of red balls is twice the number of blue balls, prove that there are 7 green balls. -/
theorem number_of_green_balls (total : ℕ) (blue : ℕ) (red : ℕ) (green : ℕ) : 
  total = 40 →
  blue = 11 →
  red = 2 * blue →
  total = red + blue + green →
  green = 7 := by
sorry

end NUMINAMATH_CALUDE_number_of_green_balls_l1015_101503


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l1015_101533

theorem binomial_coefficient_n_minus_two (n : ℕ+) : 
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l1015_101533


namespace NUMINAMATH_CALUDE_boat_half_speed_time_and_distance_l1015_101599

/-- Represents the motion of a boat experiencing water resistance -/
noncomputable def boat_motion (v₀ : ℝ) (m : ℝ) : ℝ → ℝ × ℝ := fun t =>
  let v := 50 / (t + 10)
  let s := 50 * Real.log ((t + 10) / 10)
  (v, s)

/-- Theorem stating the time and distance for the boat's speed to halve -/
theorem boat_half_speed_time_and_distance :
  let v₀ : ℝ := 5
  let m : ℝ := 1  -- We don't need a specific mass value for this problem
  let (v, s) := boat_motion v₀ m 10
  v = v₀ / 2 ∧ s = 50 * Real.log 2 := by sorry


end NUMINAMATH_CALUDE_boat_half_speed_time_and_distance_l1015_101599


namespace NUMINAMATH_CALUDE_roses_left_unsold_l1015_101576

theorem roses_left_unsold (price : ℕ) (initial : ℕ) (earned : ℕ) : 
  price = 7 → initial = 9 → earned = 35 → initial - (earned / price) = 4 := by
  sorry

end NUMINAMATH_CALUDE_roses_left_unsold_l1015_101576


namespace NUMINAMATH_CALUDE_parallel_planes_line_condition_l1015_101559

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (lineParallel : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_condition
  (α β : Plane) (m : Line)
  (h_different : α ≠ β)
  (h_subset : subset m α) :
  (∀ α β m, parallel α β → lineParallel m β) ∧
  (∃ α β m, lineParallel m β ∧ ¬parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_line_condition_l1015_101559


namespace NUMINAMATH_CALUDE_harmonious_equations_have_real_roots_l1015_101531

/-- A harmonious equation is a quadratic equation ax² + bx + c = 0 where a ≠ 0 and b = a + c -/
def HarmoniousEquation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b = a + c

/-- The discriminant of a quadratic equation ax² + bx + c = 0 -/
def Discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4*a*c

/-- A quadratic equation has real roots if and only if its discriminant is non-negative -/
def HasRealRoots (a b c : ℝ) : Prop :=
  Discriminant a b c ≥ 0

/-- Theorem: Harmonious equations always have real roots -/
theorem harmonious_equations_have_real_roots (a b c : ℝ) :
  HarmoniousEquation a b c → HasRealRoots a b c :=
by sorry

end NUMINAMATH_CALUDE_harmonious_equations_have_real_roots_l1015_101531


namespace NUMINAMATH_CALUDE_money_distribution_l1015_101540

theorem money_distribution (a b c d : ℕ) : 
  a + b + c + d = 2000 →
  a + c = 900 →
  b + c = 1100 →
  a + d = 700 →
  c = 200 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1015_101540


namespace NUMINAMATH_CALUDE_sixteen_not_valid_l1015_101588

/-- Represents a set of lines in a plane -/
structure LineSet where
  numLines : ℕ
  intersectionCount : ℕ

/-- Checks if a LineSet is valid according to the problem conditions -/
def isValidLineSet (ls : LineSet) : Prop :=
  ls.intersectionCount = 10 ∧
  ∃ (n k : ℕ), n > 1 ∧ k > 0 ∧ ls.numLines = n * k ∧ (n - 1) * k = ls.intersectionCount

/-- Theorem stating that 16 cannot be a valid number of lines in the set -/
theorem sixteen_not_valid : ¬ (∃ (ls : LineSet), ls.numLines = 16 ∧ isValidLineSet ls) := by
  sorry


end NUMINAMATH_CALUDE_sixteen_not_valid_l1015_101588


namespace NUMINAMATH_CALUDE_proposition_properties_l1015_101575

theorem proposition_properties :
  -- 1. Negation of existential quantifier
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  -- 2. Sufficient but not necessary condition
  (∃ x : ℝ, x = 1 → x^2 - 4*x + 3 = 0) ∧
  (∃ x : ℝ, x^2 - 4*x + 3 = 0 ∧ x ≠ 1) ∧
  -- 3. Converse of implication
  ((∀ x : ℝ, x^2 - 4*x + 3 = 0 → x = 1) →
   (∀ x : ℝ, x ≠ 1 → x^2 - 4*x + 3 ≠ 0)) ∧
  -- 4. Falsity of conjunction doesn't imply falsity of both propositions
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_properties_l1015_101575


namespace NUMINAMATH_CALUDE_trig_product_equals_one_sixteenth_l1015_101542

theorem trig_product_equals_one_sixteenth :
  (1 - Real.sin (π / 12)) * (1 - Real.sin (5 * π / 12)) *
  (1 - Real.sin (7 * π / 12)) * (1 - Real.sin (11 * π / 12)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_sixteenth_l1015_101542


namespace NUMINAMATH_CALUDE_acquaintance_theorem_l1015_101536

/-- A graph with 9 vertices where every subset of 3 vertices contains at least 2 connected vertices -/
def AcquaintanceGraph : Type :=
  { g : SimpleGraph (Fin 9) // ∀ (s : Finset (Fin 9)), s.card = 3 →
    ∃ (v w : Fin 9), v ∈ s ∧ w ∈ s ∧ v ≠ w ∧ g.Adj v w }

/-- The existence of a complete subgraph of 4 vertices in the AcquaintanceGraph -/
theorem acquaintance_theorem (g : AcquaintanceGraph) :
  ∃ (s : Finset (Fin 9)), s.card = 4 ∧ ∀ (v w : Fin 9), v ∈ s → w ∈ s → v ≠ w → g.val.Adj v w :=
sorry

end NUMINAMATH_CALUDE_acquaintance_theorem_l1015_101536


namespace NUMINAMATH_CALUDE_repair_time_is_30_minutes_l1015_101592

/-- The time it takes to replace the buckle on one shoe (in minutes) -/
def buckle_time : ℕ := 5

/-- The time it takes to even out the heel on one shoe (in minutes) -/
def heel_time : ℕ := 10

/-- The number of shoes Melissa is repairing -/
def num_shoes : ℕ := 2

/-- The total time Melissa spends repairing her shoes -/
def total_repair_time : ℕ := (buckle_time + heel_time) * num_shoes

theorem repair_time_is_30_minutes : total_repair_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_repair_time_is_30_minutes_l1015_101592


namespace NUMINAMATH_CALUDE_probability_of_event_B_l1015_101522

theorem probability_of_event_B 
  (A B : Set ℝ) 
  (P : Set ℝ → ℝ) 
  (h1 : P (A ∩ B) = 0.25)
  (h2 : P (A ∪ B) = 0.6)
  (h3 : P A = 0.45) :
  P B = 0.4 := by
sorry

end NUMINAMATH_CALUDE_probability_of_event_B_l1015_101522
