import Mathlib

namespace NUMINAMATH_CALUDE_same_type_as_reference_l3422_342231

-- Define the type of polynomial expressions
def PolynomialExpr (α : Type) := List (α × ℕ)

-- Function to get the type of a polynomial expression
def exprType (expr : PolynomialExpr ℚ) : PolynomialExpr ℚ :=
  expr.map (λ (c, e) ↦ (1, e))

-- Define the reference expression 3a²b
def reference : PolynomialExpr ℚ := [(3, 2), (1, 1)]

-- Define the given expressions
def expr1 : PolynomialExpr ℚ := [(-2, 2), (1, 1)]  -- -2a²b
def expr2 : PolynomialExpr ℚ := [(-2, 1), (1, 1)]  -- -2ab
def expr3 : PolynomialExpr ℚ := [(2, 1), (2, 1)]   -- 2ab²
def expr4 : PolynomialExpr ℚ := [(2, 2)]           -- 2a²

theorem same_type_as_reference :
  (exprType expr1 = exprType reference) ∧
  (exprType expr2 ≠ exprType reference) ∧
  (exprType expr3 ≠ exprType reference) ∧
  (exprType expr4 ≠ exprType reference) :=
by sorry

end NUMINAMATH_CALUDE_same_type_as_reference_l3422_342231


namespace NUMINAMATH_CALUDE_bisection_method_root_interval_l3422_342221

-- Define the function f(x) = x^3 - 2x - 1
def f (x : ℝ) : ℝ := x^3 - 2*x - 1

-- State the theorem
theorem bisection_method_root_interval :
  f 1 < 0 → f 2 > 0 → f 1.5 < 0 →
  ∃ r ∈ Set.Ioo 1.5 2, f r = 0 :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_root_interval_l3422_342221


namespace NUMINAMATH_CALUDE_pizza_piece_volume_l3422_342208

/-- The volume of a piece of pizza -/
theorem pizza_piece_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) : 
  thickness = 1/3 →
  diameter = 12 →
  num_pieces = 12 →
  (π * (diameter/2)^2 * thickness) / num_pieces = π := by
  sorry

#check pizza_piece_volume

end NUMINAMATH_CALUDE_pizza_piece_volume_l3422_342208


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l3422_342251

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l3422_342251


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3422_342247

-- Define the conditions
def condition_p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 > 0

def condition_q (m : ℝ) : Prop := ∀ x y, x < y → (m+3)^x < (m+3)^y

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, condition_p m ∧ condition_q m) ∧
  (∃ m : ℝ, ¬condition_p m ∧ condition_q m) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3422_342247


namespace NUMINAMATH_CALUDE_model_fit_relationships_l3422_342289

-- Define the model and its properties
structure Model where
  ssr : ℝ  -- Sum of squared residuals
  r_squared : ℝ  -- Coefficient of determination
  fit_quality : ℝ  -- Model fit quality (higher is better)

-- Define the relationships
axiom ssr_r_squared_inverse (m : Model) : m.ssr < 0 → m.r_squared > 0
axiom r_squared_fit_quality_direct (m : Model) : m.r_squared > 0 → m.fit_quality > 0

-- Theorem statement
theorem model_fit_relationships (m1 m2 : Model) :
  (m1.ssr < m2.ssr → m1.r_squared > m2.r_squared ∧ m1.fit_quality > m2.fit_quality) ∧
  (m1.ssr > m2.ssr → m1.r_squared < m2.r_squared ∧ m1.fit_quality < m2.fit_quality) :=
sorry

end NUMINAMATH_CALUDE_model_fit_relationships_l3422_342289


namespace NUMINAMATH_CALUDE_parabola_count_equals_intersection_count_l3422_342201

-- Define the basic geometric objects
structure Line :=
  (a b c : ℝ)

structure Point :=
  (x y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Parabola :=
  (focus : Point)
  (directrix : Line)

-- Define the given lines
def t₁ : Line := sorry
def t₂ : Line := sorry
def t₃ : Line := sorry
def e : Line := sorry

-- Define the circumcircle of the triangle formed by t₁, t₂, t₃
def circumcircle : Circle := sorry

-- Function to count intersection points between a circle and a line
def intersectionCount (c : Circle) (l : Line) : Nat := sorry

-- Function to count parabolas touching t₁, t₂, t₃ with focus on e
def parabolaCount : Nat := sorry

-- Theorem statement
theorem parabola_count_equals_intersection_count :
  parabolaCount = intersectionCount circumcircle e :=
sorry

end NUMINAMATH_CALUDE_parabola_count_equals_intersection_count_l3422_342201


namespace NUMINAMATH_CALUDE_dot_product_special_vectors_l3422_342207

/-- The dot product of vectors a = (sin 55°, sin 35°) and b = (sin 25°, sin 65°) is equal to √3/2 -/
theorem dot_product_special_vectors :
  let a : ℝ × ℝ := (Real.sin (55 * π / 180), Real.sin (35 * π / 180))
  let b : ℝ × ℝ := (Real.sin (25 * π / 180), Real.sin (65 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_special_vectors_l3422_342207


namespace NUMINAMATH_CALUDE_rational_inequality_l3422_342271

theorem rational_inequality (x : ℝ) : (x^2 - 9) / (x + 3) < 0 ↔ x ∈ Set.Ioo (-3 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_rational_inequality_l3422_342271


namespace NUMINAMATH_CALUDE_kevin_cards_l3422_342291

/-- The number of cards Kevin has at the end of the day -/
def final_cards (initial : ℕ) (found : ℕ) (lost1 : ℕ) (lost2 : ℕ) (won : ℕ) : ℕ :=
  initial + found - lost1 - lost2 + won

/-- Theorem stating that Kevin ends up with 63 cards given the problem conditions -/
theorem kevin_cards : final_cards 20 47 7 12 15 = 63 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l3422_342291


namespace NUMINAMATH_CALUDE_first_pipe_fill_time_l3422_342230

def cistern_problem (x : ℝ) : Prop :=
  let second_pipe_time : ℝ := 15
  let both_pipes_time : ℝ := 6
  let remaining_time : ℝ := 1.5
  (both_pipes_time / x + both_pipes_time / second_pipe_time + remaining_time / second_pipe_time) = 1

theorem first_pipe_fill_time :
  ∃ x : ℝ, cistern_problem x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_first_pipe_fill_time_l3422_342230


namespace NUMINAMATH_CALUDE_inequality_proof_l3422_342222

theorem inequality_proof (x y : ℝ) : 
  |((x + y) * (1 - x * y)) / ((1 + x^2) * (1 + y^2))| ≤ (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3422_342222


namespace NUMINAMATH_CALUDE_distinct_cube_constructions_l3422_342264

/-- The group of rotational symmetries of a cube -/
def CubeRotationGroup : Type := Unit

/-- The number of elements in the cube rotation group -/
def CubeRotationGroup.order : ℕ := 24

/-- The number of ways to place 5 white cubes in a 2x2x2 cube -/
def WhiteCubePlacements : ℕ := Nat.choose 8 5

/-- The number of fixed points under the identity rotation -/
def FixedPointsUnderIdentity : ℕ := WhiteCubePlacements

/-- The number of fixed points under all non-identity rotations -/
def FixedPointsUnderNonIdentity : ℕ := 0

/-- The total number of fixed points under all rotations -/
def TotalFixedPoints : ℕ := FixedPointsUnderIdentity + 23 * FixedPointsUnderNonIdentity

theorem distinct_cube_constructions :
  (TotalFixedPoints : ℚ) / CubeRotationGroup.order = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_distinct_cube_constructions_l3422_342264


namespace NUMINAMATH_CALUDE_sum_of_angles_l3422_342215

theorem sum_of_angles (angle1 angle2 angle3 angle4 angle5 angle6 angleA angleB angleC : ℝ) :
  angle1 + angle3 + angle5 = 180 →
  angle2 + angle4 + angle6 = 180 →
  angleA + angleB + angleC = 180 →
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angleA + angleB + angleC = 540 := by
sorry

end NUMINAMATH_CALUDE_sum_of_angles_l3422_342215


namespace NUMINAMATH_CALUDE_sum_zero_not_all_negative_l3422_342295

theorem sum_zero_not_all_negative (a b c : ℝ) (h : a + b + c = 0) :
  ¬(a < 0 ∧ b < 0 ∧ c < 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_not_all_negative_l3422_342295


namespace NUMINAMATH_CALUDE_unique_prime_in_range_l3422_342218

theorem unique_prime_in_range : ∃! (n : ℕ), 
  50 < n ∧ n < 60 ∧ 
  Nat.Prime n ∧ 
  n % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_in_range_l3422_342218


namespace NUMINAMATH_CALUDE_average_of_combined_results_l3422_342294

theorem average_of_combined_results :
  let n₁ : ℕ := 30
  let avg₁ : ℚ := 20
  let n₂ : ℕ := 20
  let avg₂ : ℚ := 30
  let total_sum : ℚ := n₁ * avg₁ + n₂ * avg₂
  let total_count : ℕ := n₁ + n₂
  total_sum / total_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l3422_342294


namespace NUMINAMATH_CALUDE_circus_tent_seating_l3422_342273

theorem circus_tent_seating (total_capacity : ℕ) (num_sections : ℕ) : 
  total_capacity = 984 → num_sections = 4 → 
  (total_capacity / num_sections : ℕ) = 246 := by
  sorry

end NUMINAMATH_CALUDE_circus_tent_seating_l3422_342273


namespace NUMINAMATH_CALUDE_peace_numbers_examples_l3422_342227

/-- Two numbers are peace numbers about 3 if their sum is 3 -/
def PeaceNumbersAbout3 (a b : ℝ) : Prop := a + b = 3

theorem peace_numbers_examples :
  (PeaceNumbersAbout3 4 (-1)) ∧
  (∀ x : ℝ, PeaceNumbersAbout3 (8 - x) (-5 + x)) ∧
  (∀ x : ℝ, PeaceNumbersAbout3 (x^2 - 4*x - 1) (x^2 - 2*(x^2 - 2*x - 2))) ∧
  (∀ k : ℕ, (∃ x : ℕ, x > 0 ∧ PeaceNumbersAbout3 (k * x + 1) (x - 2)) ↔ (k = 1 ∨ k = 3)) :=
by sorry

end NUMINAMATH_CALUDE_peace_numbers_examples_l3422_342227


namespace NUMINAMATH_CALUDE_seven_by_seven_grid_shaded_percentage_l3422_342214

/-- Represents a square grid -/
structure SquareGrid :=
  (size : ℕ)
  (shaded : ℕ)

/-- Calculates the percentage of shaded area in a square grid -/
def shadedPercentage (grid : SquareGrid) : ℚ :=
  (grid.shaded : ℚ) / (grid.size * grid.size : ℚ) * 100

/-- Theorem: The percentage of shaded area in a 7x7 grid with 7 shaded squares is (1/7) * 100% -/
theorem seven_by_seven_grid_shaded_percentage :
  let grid : SquareGrid := ⟨7, 7⟩
  shadedPercentage grid = 100 / 7 := by sorry

end NUMINAMATH_CALUDE_seven_by_seven_grid_shaded_percentage_l3422_342214


namespace NUMINAMATH_CALUDE_perimeter_of_AMN_l3422_342280

-- Define the triangle ABC
structure Triangle :=
  (AB BC CA : ℝ)

-- Define the properties of triangle AMN
structure TriangleAMN (ABC : Triangle) :=
  (M : ℝ) -- Distance BM
  (N : ℝ) -- Distance CN
  (parallel_to_BC : True) -- MN is parallel to BC

-- Theorem statement
theorem perimeter_of_AMN (ABC : Triangle) (AMN : TriangleAMN ABC) :
  ABC.AB = 26 ∧ ABC.BC = 17 ∧ ABC.CA = 19 →
  (ABC.AB - AMN.M) + (ABC.CA - AMN.N) + 
    ((AMN.M / ABC.AB) * ABC.BC) = 45 :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_AMN_l3422_342280


namespace NUMINAMATH_CALUDE_train_speed_l3422_342268

/-- Proves that a train of given length crossing a platform of given length in a given time has a specific speed in km/hr -/
theorem train_speed (train_length platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 230 ∧ 
  platform_length = 290 ∧ 
  crossing_time = 26 →
  (train_length + platform_length) / crossing_time * 3.6 = 72 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3422_342268


namespace NUMINAMATH_CALUDE_number_difference_and_division_l3422_342228

theorem number_difference_and_division (S L : ℕ) : 
  L - S = 8327 → L = 21 * S + 125 → S = 410 ∧ L = 8735 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_and_division_l3422_342228


namespace NUMINAMATH_CALUDE_max_sum_constrained_l3422_342288

theorem max_sum_constrained (x y : ℝ) (h : x^2 + y^2 + x*y = 1) :
  x + y ≤ 2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_sum_constrained_l3422_342288


namespace NUMINAMATH_CALUDE_stratified_sample_teachers_under_40_l3422_342249

/-- Calculates the number of teachers under 40 in a stratified sample -/
def stratified_sample_size (total_population : ℕ) (under_40_population : ℕ) (sample_size : ℕ) : ℕ :=
  (under_40_population * sample_size) / total_population

/-- Theorem: The stratified sample size for teachers under 40 is 50 -/
theorem stratified_sample_teachers_under_40 :
  stratified_sample_size 490 350 70 = 50 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_teachers_under_40_l3422_342249


namespace NUMINAMATH_CALUDE_bryans_score_l3422_342203

/-- Represents the math exam scores for Bryan, Jen, and Sammy -/
structure ExamScores where
  bryan : ℕ
  jen : ℕ
  sammy : ℕ

/-- The total points possible on the exam -/
def totalPoints : ℕ := 35

/-- Defines the relationship between the scores based on the given conditions -/
def validScores (scores : ExamScores) : Prop :=
  scores.jen = scores.bryan + 10 ∧
  scores.sammy = scores.jen - 2 ∧
  scores.sammy = totalPoints - 7

/-- Theorem stating Bryan's score on the exam -/
theorem bryans_score (scores : ExamScores) (h : validScores scores) : scores.bryan = 20 := by
  sorry

end NUMINAMATH_CALUDE_bryans_score_l3422_342203


namespace NUMINAMATH_CALUDE_super_ball_distance_l3422_342233

def initial_height : ℚ := 80
def rebound_factor : ℚ := 2/3
def num_bounces : ℕ := 4

def bounce_sequence (n : ℕ) : ℚ :=
  initial_height * (rebound_factor ^ n)

def total_distance : ℚ :=
  2 * (initial_height * (1 - rebound_factor^(num_bounces + 1)) / (1 - rebound_factor)) - initial_height

theorem super_ball_distance :
  total_distance = 11280/81 :=
sorry

end NUMINAMATH_CALUDE_super_ball_distance_l3422_342233


namespace NUMINAMATH_CALUDE_triangle_circumcircle_identity_l3422_342270

/-- Given a triangle inscribed in a circle, this theorem states the relationship
    between the sides, angles, and the radius of the circumscribed circle. -/
theorem triangle_circumcircle_identity 
  (R : ℝ) (A B C : ℝ) (a b c : ℝ)
  (h_triangle : A + B + C = π)
  (h_a : a = 2 * R * Real.sin A)
  (h_b : b = 2 * R * Real.sin B)
  (h_c : c = 2 * R * Real.sin C) :
  a * Real.cos A + b * Real.cos B + c * Real.cos C = 4 * R * Real.sin A * Real.sin B * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_identity_l3422_342270


namespace NUMINAMATH_CALUDE_profit_ratio_theorem_l3422_342254

/-- Represents a partner's investment details -/
structure Partner where
  investment : ℚ
  time : ℕ

/-- Calculates the profit factor for a partner -/
def profitFactor (p : Partner) : ℚ :=
  p.investment * p.time

/-- Theorem: Given the investment ratio and time periods, prove the profit ratio -/
theorem profit_ratio_theorem (p q : Partner) 
  (h1 : p.investment / q.investment = 7 / 5)
  (h2 : p.time = 20)
  (h3 : q.time = 40) :
  profitFactor p / profitFactor q = 7 / 10 := by
  sorry

#check profit_ratio_theorem

end NUMINAMATH_CALUDE_profit_ratio_theorem_l3422_342254


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3422_342245

theorem fraction_product_simplification :
  (150 : ℚ) / 12 * 7 / 140 * 6 / 5 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3422_342245


namespace NUMINAMATH_CALUDE_axis_of_symmetry_can_be_left_of_y_axis_l3422_342248

theorem axis_of_symmetry_can_be_left_of_y_axis :
  ∃ (a : ℝ), a > 0 ∧ ∃ (x : ℝ), x < 0 ∧
    x = -(1 - 2*a) / (2*a) ∧
    ∀ (y : ℝ), y = a*x^2 + (1 - 2*a)*x :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_can_be_left_of_y_axis_l3422_342248


namespace NUMINAMATH_CALUDE_power_relation_l3422_342236

theorem power_relation (a x y : ℝ) (ha : a > 0) (hx : a^x = 2) (hy : a^y = 3) :
  a^(x - 2*y) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l3422_342236


namespace NUMINAMATH_CALUDE_mitten_knitting_time_l3422_342279

/-- Represents the time (in hours) to knit each item -/
structure KnittingTimes where
  hat : ℝ
  scarf : ℝ
  sweater : ℝ
  sock : ℝ
  mitten : ℝ

/-- Represents the number of each item in a set -/
structure SetComposition where
  hats : ℕ
  scarves : ℕ
  sweaters : ℕ
  mittens : ℕ
  socks : ℕ

def numGrandchildren : ℕ := 3

def knittingTimes : KnittingTimes := {
  hat := 2,
  scarf := 3,
  sweater := 6,
  sock := 1.5,
  mitten := 0  -- We'll solve for this
}

def setComposition : SetComposition := {
  hats := 1,
  scarves := 1,
  sweaters := 1,
  mittens := 2,
  socks := 2
}

def totalTime : ℝ := 48

theorem mitten_knitting_time :
  ∃ (mittenTime : ℝ),
    mittenTime > 0 ∧
    (let kt := { knittingTimes with mitten := mittenTime };
     (kt.hat * setComposition.hats +
      kt.scarf * setComposition.scarves +
      kt.sweater * setComposition.sweaters +
      kt.mitten * setComposition.mittens +
      kt.sock * setComposition.socks) * numGrandchildren = totalTime) ∧
    mittenTime = 1 := by sorry

end NUMINAMATH_CALUDE_mitten_knitting_time_l3422_342279


namespace NUMINAMATH_CALUDE_carly_bbq_cooking_time_l3422_342299

/-- Represents the cooking scenario for Carly's BBQ --/
structure BBQScenario where
  cook_time_per_side : ℕ
  burgers_per_batch : ℕ
  total_guests : ℕ
  guests_wanting_two : ℕ
  guests_wanting_one : ℕ

/-- Calculates the total cooking time for all burgers --/
def total_cooking_time (scenario : BBQScenario) : ℕ :=
  let total_burgers := 2 * scenario.guests_wanting_two + scenario.guests_wanting_one
  let num_batches := (total_burgers + scenario.burgers_per_batch - 1) / scenario.burgers_per_batch
  num_batches * (2 * scenario.cook_time_per_side)

/-- Theorem stating that the total cooking time for Carly's scenario is 72 minutes --/
theorem carly_bbq_cooking_time :
  total_cooking_time {
    cook_time_per_side := 4,
    burgers_per_batch := 5,
    total_guests := 30,
    guests_wanting_two := 15,
    guests_wanting_one := 15
  } = 72 := by
  sorry

end NUMINAMATH_CALUDE_carly_bbq_cooking_time_l3422_342299


namespace NUMINAMATH_CALUDE_problem_solution_l3422_342229

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : 
  x = 50 ∨ x = 16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3422_342229


namespace NUMINAMATH_CALUDE_lamp_game_solvable_l3422_342246

/-- Represents a move in the lamp game -/
inductive Move
  | row (r : Nat) (start : Nat)
  | col (c : Nat) (start : Nat)

/-- The lamp game state -/
def LampGame (n m : Nat) :=
  { grid : Fin n → Fin n → Bool // n > 0 ∧ m > 0 }

/-- Applies a move to the game state -/
def applyMove (game : LampGame n m) (move : Move) : LampGame n m :=
  sorry

/-- Checks if all lamps are on -/
def allOn (game : LampGame n m) : Prop :=
  ∀ i j, game.val i j = true

/-- Main theorem: all lamps can be turned on iff m divides n -/
theorem lamp_game_solvable (n m : Nat) :
  (∃ (game : LampGame n m) (moves : List Move), allOn (moves.foldl applyMove game)) ↔ m ∣ n :=
  sorry

end NUMINAMATH_CALUDE_lamp_game_solvable_l3422_342246


namespace NUMINAMATH_CALUDE_gcd_fraction_equality_l3422_342261

theorem gcd_fraction_equality (a b c d : ℕ) (h : a * b = c * d) :
  (Nat.gcd a c * Nat.gcd a d) / Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = a := by
  sorry

end NUMINAMATH_CALUDE_gcd_fraction_equality_l3422_342261


namespace NUMINAMATH_CALUDE_square_diagonal_characterization_l3422_342235

/-- A quadrilateral with vertices A, B, C, and D in 2D space. -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The diagonals of a quadrilateral. -/
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((q.C.1 - q.A.1, q.C.2 - q.A.2), (q.D.1 - q.B.1, q.D.2 - q.B.2))

/-- Check if two vectors are perpendicular. -/
def are_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Check if a vector bisects another vector. -/
def bisects (v w : ℝ × ℝ) : Prop :=
  v.1 = w.1 / 2 ∧ v.2 = w.2 / 2

/-- Check if two vectors have equal length. -/
def equal_length (v w : ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 = w.1^2 + w.2^2

/-- A square is a quadrilateral with all sides equal and all angles right angles. -/
def is_square (q : Quadrilateral) : Prop :=
  let (AC, BD) := diagonals q
  are_perpendicular AC BD ∧
  bisects AC BD ∧
  bisects BD AC ∧
  equal_length AC BD

theorem square_diagonal_characterization (q : Quadrilateral) :
  is_square q ↔
    let (AC, BD) := diagonals q
    are_perpendicular AC BD ∧
    bisects AC BD ∧
    bisects BD AC ∧
    equal_length AC BD :=
  sorry

end NUMINAMATH_CALUDE_square_diagonal_characterization_l3422_342235


namespace NUMINAMATH_CALUDE_hypotenuse_length_l3422_342267

-- Define a right triangle with legs 3 and 5
def right_triangle (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 5 ∧ c^2 = a^2 + b^2

-- Theorem statement
theorem hypotenuse_length :
  ∀ a b c : ℝ, right_triangle a b c → c = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l3422_342267


namespace NUMINAMATH_CALUDE_share_difference_l3422_342292

/-- Given four shares in the ratio 3:3:7:4, where the second share is 1500
    and the fourth share is 2000, the difference between the largest share
    and the second-largest share is 1500. -/
theorem share_difference (shares : Fin 4 → ℕ) : 
  (∃ x : ℕ, shares 0 = 3*x ∧ shares 1 = 3*x ∧ shares 2 = 7*x ∧ shares 3 = 4*x) →
  shares 1 = 1500 →
  shares 3 = 2000 →
  (shares 2 - (max (shares 0) (shares 3))) = 1500 := by
sorry

end NUMINAMATH_CALUDE_share_difference_l3422_342292


namespace NUMINAMATH_CALUDE_cars_meeting_time_l3422_342287

/-- Two cars driving towards each other meet after a certain time -/
theorem cars_meeting_time (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) : 
  speed1 = 100 →
  speed1 = 1.25 * speed2 →
  distance = 720 →
  (distance / (speed1 + speed2)) = 4 := by
sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l3422_342287


namespace NUMINAMATH_CALUDE_gas_pressure_calculation_l3422_342200

/-- Represents the pressure-volume relationship for a gas at constant temperature -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  inv_prop : volume * pressure = volume * pressure

/-- The initial state of the gas -/
def initial_state : GasState where
  volume := 3
  pressure := 8
  inv_prop := by sorry

/-- The final state of the gas -/
def final_state : GasState where
  volume := 7.5
  pressure := 3.2
  inv_prop := by sorry

/-- Theorem stating that the final pressure is correct given the initial conditions -/
theorem gas_pressure_calculation (initial : GasState) (final : GasState)
    (h_initial : initial = initial_state)
    (h_final_volume : final.volume = 7.5)
    (h_const : initial.volume * initial.pressure = final.volume * final.pressure) :
    final.pressure = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_gas_pressure_calculation_l3422_342200


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3422_342257

theorem arithmetic_mean_problem (a b c d : ℝ) :
  (a + b + c + d + 120) / 5 = 100 →
  (a + b + c + d) / 4 = 95 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3422_342257


namespace NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_find_k_value_l3422_342225

/-- A quadratic equation with parameter k -/
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 + (2*k - 1)*x - k - 2

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ := (2*k - 1)^2 - 4*1*(-k - 2)

theorem quadratic_has_two_distinct_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 :=
sorry

theorem find_k_value (k : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic k x₁ = 0)
  (h₂ : quadratic k x₂ = 0)
  (h₃ : x₁ + x₂ - 4*x₁*x₂ = 1) :
  k = -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_find_k_value_l3422_342225


namespace NUMINAMATH_CALUDE_special_circle_equation_l3422_342259

/-- A circle with center on y = 2x and specific chord lengths -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.2 = 2 * center.1
  x_chord_length : 4 = 2 * (radius ^ 2 - center.1 ^ 2).sqrt
  y_chord_length : 8 = 2 * (radius ^ 2 - center.2 ^ 2).sqrt

/-- The equation of the circle is one of two specific forms -/
theorem special_circle_equation (c : SpecialCircle) :
  (∀ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = 5 ↔ (x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2) ∨
  (∀ x y : ℝ, (x + 1) ^ 2 + (y + 2) ^ 2 = 5 ↔ (x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2) :=
sorry

end NUMINAMATH_CALUDE_special_circle_equation_l3422_342259


namespace NUMINAMATH_CALUDE_linear_expr_pythagorean_relation_l3422_342212

-- Define linear expressions
def LinearExpr (α : Type*) [Ring α] := α → α

-- Theorem statement
theorem linear_expr_pythagorean_relation
  {α : Type*} [Field α]
  (A B C : LinearExpr α)
  (h : ∀ x, (A x)^2 + (B x)^2 = (C x)^2) :
  ∃ (k₁ k₂ : α), ∀ x, A x = k₁ * (C x) ∧ B x = k₂ * (C x) := by
  sorry

end NUMINAMATH_CALUDE_linear_expr_pythagorean_relation_l3422_342212


namespace NUMINAMATH_CALUDE_quadratic_and_inequality_system_solution_l3422_342278

theorem quadratic_and_inequality_system_solution :
  (∃ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) ∧
  (∀ x : ℝ, 3*x + 5 ≥ 2 ∧ (x - 1) / 2 < (x + 1) / 4 ↔ -1 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_inequality_system_solution_l3422_342278


namespace NUMINAMATH_CALUDE_exists_multiple_with_digit_sum_l3422_342223

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number that is a multiple of 2015 and whose sum of digits equals 2015 -/
theorem exists_multiple_with_digit_sum :
  ∃ (n : ℕ), (n % 2015 = 0) ∧ (sum_of_digits n = 2015) := by sorry

end NUMINAMATH_CALUDE_exists_multiple_with_digit_sum_l3422_342223


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l3422_342224

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (12*x + 3) * (12*x + 9) * (6*x + 6) = 324 * k) ∧
  (∀ (m : ℤ), m > 324 → ¬(∀ (x : ℤ), Odd x → ∃ (k : ℤ), (12*x + 3) * (12*x + 9) * (6*x + 6) = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l3422_342224


namespace NUMINAMATH_CALUDE_f_even_not_odd_implies_a_gt_one_l3422_342277

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 - 1) + Real.sqrt (a - x^2)

theorem f_even_not_odd_implies_a_gt_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) ∧ 
  (∃ x, f a x ≠ -(f a (-x))) →
  a > 1 := by sorry

end NUMINAMATH_CALUDE_f_even_not_odd_implies_a_gt_one_l3422_342277


namespace NUMINAMATH_CALUDE_oliver_seashell_difference_l3422_342260

/-- The number of seashells Oliver collected on Monday -/
def monday_shells : ℕ := 2

/-- The total number of seashells Oliver collected -/
def total_shells : ℕ := 4

/-- The number of seashells Oliver collected on Tuesday -/
def tuesday_shells : ℕ := total_shells - monday_shells

/-- Theorem: Oliver collected 2 more seashells on Tuesday compared to Monday -/
theorem oliver_seashell_difference : tuesday_shells - monday_shells = 2 := by
  sorry

end NUMINAMATH_CALUDE_oliver_seashell_difference_l3422_342260


namespace NUMINAMATH_CALUDE_family_income_and_tax_calculation_l3422_342266

/-- Family income and tax calculation -/
theorem family_income_and_tax_calculation 
  (father_monthly_income mother_monthly_income grandmother_monthly_pension mikhail_monthly_scholarship : ℕ)
  (property_cadastral_value property_area : ℕ)
  (lada_priora_hp lada_priora_months lada_xray_hp lada_xray_months : ℕ)
  (land_cadastral_value land_area : ℕ)
  (tour_cost_per_person : ℕ)
  (h1 : father_monthly_income = 50000)
  (h2 : mother_monthly_income = 28000)
  (h3 : grandmother_monthly_pension = 15000)
  (h4 : mikhail_monthly_scholarship = 3000)
  (h5 : property_cadastral_value = 6240000)
  (h6 : property_area = 78)
  (h7 : lada_priora_hp = 106)
  (h8 : lada_priora_months = 3)
  (h9 : lada_xray_hp = 122)
  (h10 : lada_xray_months = 8)
  (h11 : land_cadastral_value = 420300)
  (h12 : land_area = 10)
  (h13 : tour_cost_per_person = 17900) :
  ∃ (january_income annual_income property_tax transport_tax land_tax remaining_funds : ℕ),
    january_income = 86588 ∧
    annual_income = 137236 ∧
    property_tax = 4640 ∧
    transport_tax = 3775 ∧
    land_tax = 504 ∧
    remaining_funds = 38817 :=
by sorry


end NUMINAMATH_CALUDE_family_income_and_tax_calculation_l3422_342266


namespace NUMINAMATH_CALUDE_vector_b_components_l3422_342209

def vector_a : ℝ × ℝ := (2, -1)

theorem vector_b_components :
  ∀ (b : ℝ × ℝ),
  (∃ (k : ℝ), k < 0 ∧ b = (k * vector_a.1, k * vector_a.2)) →
  (b.1 * b.1 + b.2 * b.2 = 20) →
  b = (-4, 2) := by sorry

end NUMINAMATH_CALUDE_vector_b_components_l3422_342209


namespace NUMINAMATH_CALUDE_min_value_f1_div_f2prime0_l3422_342210

/-- A quadratic function f(x) = ax² + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  f_prime_0_pos : 2 * a * 0 + b > 0
  range_nonneg : ∀ x, a * x^2 + b * x + c ≥ 0

/-- The theorem stating the minimum value of f(1) / f''(0) for quadratic functions with specific properties -/
theorem min_value_f1_div_f2prime0 (f : QuadraticFunction) :
  (∀ g : QuadraticFunction, (g.a + g.b + g.c) / (2 * g.a) ≥ (f.a + f.b + f.c) / (2 * f.a)) →
  (f.a + f.b + f.c) / (2 * f.a) = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_f1_div_f2prime0_l3422_342210


namespace NUMINAMATH_CALUDE_min_value_w_l3422_342293

theorem min_value_w (x y z : ℝ) :
  x^2 + 4*y^2 + 8*x - 6*y + z - 20 ≥ z - 38.25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_w_l3422_342293


namespace NUMINAMATH_CALUDE_base10_to_base5_453_l3422_342213

-- Define a function to convert from base 10 to base 5
def toBase5 (n : ℕ) : List ℕ :=
  sorry

-- Theorem stating that 453 in base 10 is equal to 3303 in base 5
theorem base10_to_base5_453 : toBase5 453 = [3, 3, 0, 3] :=
  sorry

end NUMINAMATH_CALUDE_base10_to_base5_453_l3422_342213


namespace NUMINAMATH_CALUDE_cubic_difference_l3422_342216

theorem cubic_difference (x : ℝ) (h : x - 1/x = 3) : x^3 - 1/x^3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l3422_342216


namespace NUMINAMATH_CALUDE_students_after_three_stops_l3422_342286

/-- Calculates the number of students on the bus after three stops --/
def studentsOnBusAfterThreeStops (initial : ℕ) 
  (firstOff firstOn : ℕ) 
  (secondOff secondOn : ℕ) 
  (thirdOff thirdOn : ℕ) : ℕ :=
  initial - firstOff + firstOn - secondOff + secondOn - thirdOff + thirdOn

/-- Theorem stating the number of students on the bus after three stops --/
theorem students_after_three_stops :
  studentsOnBusAfterThreeStops 10 3 4 2 5 6 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_students_after_three_stops_l3422_342286


namespace NUMINAMATH_CALUDE_households_with_appliances_l3422_342202

theorem households_with_appliances 
  (total : ℕ) 
  (tv : ℕ) 
  (fridge : ℕ) 
  (both : ℕ) 
  (h1 : total = 100) 
  (h2 : tv = 65) 
  (h3 : fridge = 84) 
  (h4 : both = 53) : 
  tv + fridge - both = 96 := by
  sorry

end NUMINAMATH_CALUDE_households_with_appliances_l3422_342202


namespace NUMINAMATH_CALUDE_ellipse_parabola_line_equations_l3422_342242

/-- Given an ellipse and a parabola with specific properties, prove the equations of both curves and a line. -/
theorem ellipse_parabola_line_equations :
  ∀ (a b c p : ℝ) (F A B P Q D : ℝ × ℝ),
  a > 0 → b > 0 → a > b →
  c / a = 1 / 2 →
  A.1 - F.1 = a →
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 →
  ∀ (x y : ℝ), y^2 = 2 * p * x →
  A.1 - F.1 = 1 / 2 →
  P.1 = Q.1 ∧ P.2 = -Q.2 →
  B ≠ A →
  D.2 = 0 →
  abs ((A.1 - P.1) * (D.2 - P.2) - (A.2 - P.2) * (D.1 - P.1)) / 2 = Real.sqrt 6 / 2 →
  ((∀ (x y : ℝ), x^2 + 4 * y^2 / 3 = 1) ∧
   (∀ (x y : ℝ), y^2 = 4 * x) ∧
   ((3 * P.1 + Real.sqrt 6 * P.2 - 3 = 0) ∨ (3 * P.1 - Real.sqrt 6 * P.2 - 3 = 0))) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_line_equations_l3422_342242


namespace NUMINAMATH_CALUDE_train_length_problem_l3422_342275

theorem train_length_problem (speed1 speed2 time length2 : Real) 
  (h1 : speed1 = 120)
  (h2 : speed2 = 80)
  (h3 : time = 9)
  (h4 : length2 = 210.04)
  : ∃ length1 : Real, length1 = 290 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l3422_342275


namespace NUMINAMATH_CALUDE_painting_price_change_l3422_342276

/-- The percentage increase in the first year given the conditions of the problem -/
def first_year_increase : ℝ := 30

/-- The percentage decrease in the second year -/
def second_year_decrease : ℝ := 15

/-- The final price as a percentage of the original price -/
def final_price_percentage : ℝ := 110.5

theorem painting_price_change : 
  (100 + first_year_increase) * (100 - second_year_decrease) / 100 = final_price_percentage := by
  sorry

#check painting_price_change

end NUMINAMATH_CALUDE_painting_price_change_l3422_342276


namespace NUMINAMATH_CALUDE_boat_distance_proof_l3422_342250

-- Define the given constants
def boat_speed : ℝ := 10
def stream_speed : ℝ := 2
def time_difference : ℝ := 1.5  -- 90 minutes in hours

-- Define the theorem
theorem boat_distance_proof :
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let upstream_time := (downstream_speed * time_difference) / (downstream_speed - upstream_speed)
  let distance := upstream_speed * upstream_time
  distance = 36 := by sorry

end NUMINAMATH_CALUDE_boat_distance_proof_l3422_342250


namespace NUMINAMATH_CALUDE_base_conversion_536_7_to_6_l3422_342269

/-- Converts a number from base b1 to base b2 -/
def convert_base (n : ℕ) (b1 b2 : ℕ) : ℕ :=
  sorry

/-- Checks if a number n in base b has the given digits -/
def check_digits (n : ℕ) (b : ℕ) (digits : List ℕ) : Prop :=
  sorry

theorem base_conversion_536_7_to_6 :
  convert_base 536 7 6 = 1132 ∧ 
  check_digits 536 7 [6, 3, 5] ∧
  check_digits 1132 6 [2, 3, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_base_conversion_536_7_to_6_l3422_342269


namespace NUMINAMATH_CALUDE_fiona_reaches_food_l3422_342206

/-- Represents a lily pad --/
structure LilyPad :=
  (number : ℕ)

/-- Represents Fiona the frog --/
structure Frog :=
  (position : LilyPad)

/-- Represents the probability of a jump --/
def JumpProbability : ℚ := 1/3

/-- The total number of lily pads --/
def TotalPads : ℕ := 16

/-- The position of the first predator --/
def Predator1 : LilyPad := ⟨4⟩

/-- The position of the second predator --/
def Predator2 : LilyPad := ⟨9⟩

/-- The position of the food --/
def FoodPosition : LilyPad := ⟨14⟩

/-- Fiona's starting position --/
def StartPosition : LilyPad := ⟨0⟩

/-- Function to calculate the probability of Fiona reaching the food --/
noncomputable def probabilityToReachFood (f : Frog) : ℚ :=
  sorry

theorem fiona_reaches_food :
  probabilityToReachFood ⟨StartPosition⟩ = 52/59049 :=
sorry

end NUMINAMATH_CALUDE_fiona_reaches_food_l3422_342206


namespace NUMINAMATH_CALUDE_prob_a_prob_b_prob_c_prob_d_prob_e_chess_probabilities_l3422_342241

/-- The total number of chess pieces -/
def total_pieces : ℕ := 32

/-- The number of pieces of each color -/
def pieces_per_color : ℕ := total_pieces / 2

/-- The number of pawns of each color -/
def pawns_per_color : ℕ := 8

/-- The number of bishops of each color -/
def bishops_per_color : ℕ := 2

/-- The number of rooks of each color -/
def rooks_per_color : ℕ := 2

/-- The number of knights of each color -/
def knights_per_color : ℕ := 2

/-- The number of kings of each color -/
def kings_per_color : ℕ := 1

/-- The number of queens of each color -/
def queens_per_color : ℕ := 1

/-- The probability of drawing 2 dark pieces or 2 pieces of different colors -/
theorem prob_a : ℚ :=
  47 / 62

/-- The probability of drawing 1 bishop and 1 pawn or 2 pieces of different colors -/
theorem prob_b : ℚ :=
  18 / 31

/-- The probability of drawing 2 different-colored rooks or 2 pieces of the same color but different sizes -/
theorem prob_c : ℚ :=
  91 / 248

/-- The probability of drawing 1 king and one knight of the same color, or two pieces of the same color -/
theorem prob_d : ℚ :=
  15 / 31

/-- The probability of drawing 2 pieces of the same size or 2 pieces of the same color -/
theorem prob_e : ℚ :=
  159 / 248

/-- The main theorem combining all probabilities -/
theorem chess_probabilities :
  (prob_a = 47 / 62) ∧
  (prob_b = 18 / 31) ∧
  (prob_c = 91 / 248) ∧
  (prob_d = 15 / 31) ∧
  (prob_e = 159 / 248) :=
by sorry

end NUMINAMATH_CALUDE_prob_a_prob_b_prob_c_prob_d_prob_e_chess_probabilities_l3422_342241


namespace NUMINAMATH_CALUDE_perfect_square_function_characterization_l3422_342285

theorem perfect_square_function_characterization (g : ℕ → ℕ) : 
  (∀ n m : ℕ, ∃ k : ℕ, (g n + m) * (g m + n) = k^2) →
  ∃ c : ℕ, ∀ n : ℕ, g n = n + c := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_function_characterization_l3422_342285


namespace NUMINAMATH_CALUDE_square_sum_product_equality_l3422_342284

theorem square_sum_product_equality : (6 + 10)^2 + (6^2 + 10^2 + 6 * 10) = 452 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_equality_l3422_342284


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l3422_342252

theorem min_values_xy_and_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1/x + 9/y = 1) :
  xy ≥ 36 ∧ x + 2*y ≥ 19 + 6*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l3422_342252


namespace NUMINAMATH_CALUDE_total_cost_theorem_l3422_342205

/-- The cost of cherries per pound in yuan -/
def cherry_cost : ℝ := sorry

/-- The cost of apples per pound in yuan -/
def apple_cost : ℝ := sorry

/-- The total cost of 2 pounds of cherries and 3 pounds of apples is 58 yuan -/
axiom condition1 : 2 * cherry_cost + 3 * apple_cost = 58

/-- The total cost of 3 pounds of cherries and 2 pounds of apples is 72 yuan -/
axiom condition2 : 3 * cherry_cost + 2 * apple_cost = 72

/-- The theorem states that the total cost of 3 pounds of cherries and 3 pounds of apples is 78 yuan -/
theorem total_cost_theorem : 3 * cherry_cost + 3 * apple_cost = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l3422_342205


namespace NUMINAMATH_CALUDE_sin_four_thirds_pi_l3422_342226

theorem sin_four_thirds_pi : Real.sin (4 / 3 * Real.pi) = -(Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_four_thirds_pi_l3422_342226


namespace NUMINAMATH_CALUDE_sequence_property_l3422_342255

def arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b ∧ a > 0 ∧ b > 0 ∧ c > 0

def geometric_sequence (a b c : ℝ) : Prop :=
  b / a = c / b ∧ a ≠ 0 ∧ b ≠ 0

def general_term (n : ℕ) : ℝ := 2^(n - 1)

theorem sequence_property :
  ∀ a b c : ℝ,
    arithmetic_sequence a b c →
    a + b + c = 6 →
    geometric_sequence (a + 3) (b + 6) (c + 13) →
    (∀ n : ℕ, n ≥ 3 → general_term n = (a + 3) * 2^(n - 3)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l3422_342255


namespace NUMINAMATH_CALUDE_parabola_translation_l3422_342281

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * dx + p.b
    c := p.a * dx^2 - p.b * dx + p.c + dy }

/-- The original parabola y = x² -/
def original : Parabola :=
  { a := 1, b := 0, c := 0 }

/-- The resulting parabola after translation -/
def translated : Parabola :=
  translate original 2 1

theorem parabola_translation :
  translated = { a := 1, b := -4, c := 5 } :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l3422_342281


namespace NUMINAMATH_CALUDE_circle_chord_segments_l3422_342234

theorem circle_chord_segments (r : ℝ) (chord_length : ℝ) : 
  r = 6 → 
  chord_length = 10 → 
  ∃ (m n : ℝ), m + n = 2*r ∧ m*n = (chord_length/2)^2 ∧ 
  ((m = 6 + Real.sqrt 11 ∧ n = 6 - Real.sqrt 11) ∨ 
   (m = 6 - Real.sqrt 11 ∧ n = 6 + Real.sqrt 11)) :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_segments_l3422_342234


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3422_342282

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 + 3 * x^3 - 5 * x^2 + 8 * x - 6) + (-6 * x^5 + x^3 + 4 * x^2 - 8 * x + 7) =
  -4 * x^5 + 4 * x^3 - x^2 + 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3422_342282


namespace NUMINAMATH_CALUDE_tiled_square_theorem_l3422_342243

/-- A square area tiled with identical square tiles -/
structure TiledSquare where
  /-- The number of tiles adjoining the four sides -/
  perimeter_tiles : ℕ
  /-- The total number of tiles in the square -/
  total_tiles : ℕ

/-- Theorem stating that a square area with 20 tiles adjoining its sides contains 36 tiles in total -/
theorem tiled_square_theorem (ts : TiledSquare) (h : ts.perimeter_tiles = 20) : ts.total_tiles = 36 := by
  sorry

end NUMINAMATH_CALUDE_tiled_square_theorem_l3422_342243


namespace NUMINAMATH_CALUDE_inverse_product_positive_implies_one_greater_than_one_l3422_342211

theorem inverse_product_positive_implies_one_greater_than_one 
  (a b c : ℝ) (h : (a⁻¹) * (b⁻¹) * (c⁻¹) > 0) : 
  (a > 1) ∨ (b > 1) ∨ (c > 1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_product_positive_implies_one_greater_than_one_l3422_342211


namespace NUMINAMATH_CALUDE_max_value_sum_of_square_roots_l3422_342298

theorem max_value_sum_of_square_roots (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) 
  (h_sum : a + b + c = 7) : 
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ 6 ∧
  (∃ (a₀ b₀ c₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀ + b₀ + c₀ = 7 ∧
    Real.sqrt (3 * a₀ + 1) + Real.sqrt (3 * b₀ + 1) + Real.sqrt (3 * c₀ + 1) = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_square_roots_l3422_342298


namespace NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l3422_342239

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The 10th term of a geometric sequence with first term 2 and second term 5/2 -/
theorem tenth_term_of_geometric_sequence :
  let a : ℚ := 2
  let second_term : ℚ := 5/2
  let r : ℚ := second_term / a
  geometric_term a r 10 = 3906250/262144 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l3422_342239


namespace NUMINAMATH_CALUDE_max_value_of_function_l3422_342296

theorem max_value_of_function (x : ℝ) : 1 + 1 / (x^2 + 2*x + 2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3422_342296


namespace NUMINAMATH_CALUDE_fish_per_family_member_l3422_342290

def fish_distribution (family_size : ℕ) (eyes_eaten : ℕ) (eyes_to_dog : ℕ) (eyes_per_fish : ℕ) : ℕ :=
  let total_eyes := eyes_eaten + eyes_to_dog
  let total_fish := total_eyes / eyes_per_fish
  total_fish / family_size

theorem fish_per_family_member :
  fish_distribution 3 22 2 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fish_per_family_member_l3422_342290


namespace NUMINAMATH_CALUDE_no_xyz_single_double_triple_digits_l3422_342262

theorem no_xyz_single_double_triple_digits :
  ¬∃ (x y z : ℕ+),
    (1 : ℚ) / x = 1 / y + 1 / z ∧
    ((1 ≤ x ∧ x < 10) ∨ (1 ≤ y ∧ y < 10) ∨ (1 ≤ z ∧ z < 10)) ∧
    ((10 ≤ x ∧ x < 100) ∨ (10 ≤ y ∧ y < 100) ∨ (10 ≤ z ∧ z < 100)) ∧
    ((100 ≤ x ∧ x < 1000) ∨ (100 ≤ y ∧ y < 1000) ∨ (100 ≤ z ∧ z < 1000)) :=
by sorry

end NUMINAMATH_CALUDE_no_xyz_single_double_triple_digits_l3422_342262


namespace NUMINAMATH_CALUDE_gcd_3869_6497_l3422_342204

theorem gcd_3869_6497 : Nat.gcd 3869 6497 = 73 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3869_6497_l3422_342204


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3422_342238

variable (V : Type) [AddCommGroup V] [Module ℝ V]

variable (cross : V → V → V)

variable (crossProperties : 
  (∀ a b c : V, cross (a + b) c = cross a c + cross b c) ∧ 
  (∀ a b : V, cross a b = -cross b a) ∧
  (∀ a : V, cross a a = 0))

theorem vector_equation_solution :
  ∃! (k m : ℝ), ∀ (a b c d : V), 
    a + b + c + d = 0 → 
    k • (cross b a) + m • (cross d c) + cross b c + cross c a + cross d b = 0 :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3422_342238


namespace NUMINAMATH_CALUDE_main_theorem_l3422_342263

/-- A function with the property that (f x + y) * (f y + x) > 0 implies f x + y = f y + x -/
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x

/-- The main theorem: if f has the property, then f x + y ≤ f y + x whenever x > y -/
theorem main_theorem (f : ℝ → ℝ) (hf : has_property f) :
  ∀ x y : ℝ, x > y → f x + y ≤ f y + x :=
sorry

end NUMINAMATH_CALUDE_main_theorem_l3422_342263


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l3422_342240

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l3422_342240


namespace NUMINAMATH_CALUDE_correct_answers_l3422_342297

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  totalQuestions : ℕ
  correctScore : ℤ
  wrongScore : ℤ

/-- Represents a student's exam attempt. -/
structure ExamAttempt where
  exam : Exam
  totalScore : ℤ
  attemptedAll : Bool

/-- Theorem stating the number of correctly answered questions given the exam conditions. -/
theorem correct_answers (e : Exam) (a : ExamAttempt) 
    (h1 : e.totalQuestions = 60)
    (h2 : e.correctScore = 4)
    (h3 : e.wrongScore = -1)
    (h4 : a.exam = e)
    (h5 : a.totalScore = 150)
    (h6 : a.attemptedAll = true) :
    ∃ (c : ℕ), c = 42 ∧ 
    c * e.correctScore + (e.totalQuestions - c) * e.wrongScore = a.totalScore :=
  sorry

end NUMINAMATH_CALUDE_correct_answers_l3422_342297


namespace NUMINAMATH_CALUDE_cylinder_diameter_from_sphere_surface_area_l3422_342220

theorem cylinder_diameter_from_sphere_surface_area (r_sphere : ℝ) (h_cylinder : ℝ) :
  r_sphere = 3 →
  h_cylinder = 6 →
  4 * Real.pi * r_sphere^2 = 2 * Real.pi * (6 / 2) * h_cylinder →
  6 = 2 * (6 / 2) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_diameter_from_sphere_surface_area_l3422_342220


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3422_342258

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3*a) → 
  a ∈ Set.Icc (-2 : ℝ) 5 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3422_342258


namespace NUMINAMATH_CALUDE_angle_sum_quarter_range_l3422_342232

-- Define acute and obtuse angles
def is_acute (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2
def is_obtuse (β : Real) : Prop := Real.pi / 2 < β ∧ β < Real.pi

-- Main theorem
theorem angle_sum_quarter_range (α β : Real) 
  (h_acute : is_acute α) (h_obtuse : is_obtuse β) :
  Real.pi / 8 < 0.25 * (α + β) ∧ 0.25 * (α + β) < 3 * Real.pi / 8 := by
  sorry

#check angle_sum_quarter_range

end NUMINAMATH_CALUDE_angle_sum_quarter_range_l3422_342232


namespace NUMINAMATH_CALUDE_fraction_integer_iff_p_equals_three_l3422_342283

theorem fraction_integer_iff_p_equals_three (p : ℕ+) :
  (↑p : ℚ) > 0 →
  (∃ (n : ℕ), n > 0 ∧ (5 * p + 45 : ℚ) / (3 * p - 8 : ℚ) = ↑n) ↔ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_p_equals_three_l3422_342283


namespace NUMINAMATH_CALUDE_sine_function_properties_l3422_342272

/-- Given a function y = a * sin(x) + 2b where a > 0, with maximum value 4 and minimum value 0,
    prove that a + b = 3 and the minimum positive period of y = b * sin(ax) is π -/
theorem sine_function_properties (a b : ℝ) (h_a_pos : a > 0)
  (h_max : ∀ x, a * Real.sin x + 2 * b ≤ 4)
  (h_min : ∀ x, a * Real.sin x + 2 * b ≥ 0)
  (h_max_achievable : ∃ x, a * Real.sin x + 2 * b = 4)
  (h_min_achievable : ∃ x, a * Real.sin x + 2 * b = 0) :
  (a + b = 3) ∧
  (∀ T > 0, (∀ x, b * Real.sin (a * x) = b * Real.sin (a * (x + T))) → T ≥ π) ∧
  (∀ x, b * Real.sin (a * x) = b * Real.sin (a * (x + π))) := by
  sorry


end NUMINAMATH_CALUDE_sine_function_properties_l3422_342272


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l3422_342256

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat
  firstNotFriday : firstDay ≠ DayOfWeek.Friday
  lastNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : fridayCount = 5
  validDayCount : dayCount = 30 ∨ dayCount = 31

/-- Function to get the day of the week for a given day number -/
def getDayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Theorem stating that the 12th day is a Monday -/
theorem twelfth_day_is_monday (m : Month) : 
  getDayOfWeek m 12 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l3422_342256


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3422_342265

/-- Given a quadratic function f(x) = ax² + bx + c where 2a + 3b + 6c = 0,
    there exists an x in the interval (0,1) such that f(x) = 0. -/
theorem quadratic_root_in_unit_interval (a b c : ℝ) 
  (h : 2*a + 3*b + 6*c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3422_342265


namespace NUMINAMATH_CALUDE_system_solution_l3422_342274

theorem system_solution (x y z u : ℚ) (a b c d : ℚ) :
  (x * y) / (x + y) = 1 / a ∧
  (y * z) / (y + z) = 1 / b ∧
  (z * u) / (z + u) = 1 / c ∧
  (x * y * z * u) / (x + y + z + u) = 1 / d →
  ((a = 1 ∧ b = 2 ∧ c = -1 ∧ d = 1 →
    x = -4/3 ∧ y = 4/7 ∧ z = 4 ∧ u = -4/5) ∧
   (a = 1 ∧ b = 3 ∧ c = -2 ∧ d = 1 →
    (x = -1 ∧ y = 1/2 ∧ z = 1 ∧ u = -1/3) ∨
    (x = 1/9 ∧ y = -1/8 ∧ z = 1/11 ∧ u = -1/13))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3422_342274


namespace NUMINAMATH_CALUDE_cord_cutting_problem_l3422_342244

theorem cord_cutting_problem (cord1 : ℕ) (cord2 : ℕ) 
  (h1 : cord1 = 15) (h2 : cord2 = 12) : 
  Nat.gcd cord1 cord2 = 3 := by
sorry

end NUMINAMATH_CALUDE_cord_cutting_problem_l3422_342244


namespace NUMINAMATH_CALUDE_at_least_one_chooses_probability_l3422_342219

-- Define the probabilities for Students A and B
def prob_A : ℚ := 1/3
def prob_B : ℚ := 1/4

-- Define the event that at least one student chooses the "Inequality Lecture"
def at_least_one_chooses : ℚ := 1 - (1 - prob_A) * (1 - prob_B)

-- Theorem statement
theorem at_least_one_chooses_probability :
  at_least_one_chooses = 1/2 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_chooses_probability_l3422_342219


namespace NUMINAMATH_CALUDE_correct_sum_after_reversing_tens_digit_l3422_342253

/-- Represents a three-digit number with digits a, b, and c --/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Represents the same number with tens digit reversed --/
def reversed_tens_digit (a b c : ℕ) : ℕ := 100 * a + 10 * c + b

theorem correct_sum_after_reversing_tens_digit 
  (m n : ℕ) 
  (a b c : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : m = three_digit_number a b c) 
  (h4 : reversed_tens_digit a b c + n = 128) :
  m + n = 128 := by
sorry

end NUMINAMATH_CALUDE_correct_sum_after_reversing_tens_digit_l3422_342253


namespace NUMINAMATH_CALUDE_restaurant_group_children_correct_number_of_children_l3422_342237

theorem restaurant_group_children (adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : ℕ :=
  let children := (total_bill - adults * meal_cost) / meal_cost
  children

theorem correct_number_of_children :
  restaurant_group_children 2 3 21 = 5 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_children_correct_number_of_children_l3422_342237


namespace NUMINAMATH_CALUDE_build_time_relation_l3422_342217

/-- Represents the time taken to build a cottage given the number of builders and their rate -/
def build_time (builders : ℕ) (rate : ℚ) : ℚ :=
  1 / (builders.cast * rate)

/-- Theorem stating the relationship between build times for different numbers of builders -/
theorem build_time_relation (n : ℕ) (rate : ℚ) :
  n > 0 → 6 > 0 → build_time n rate = 8 → 
  build_time 6 rate = (n.cast / 6 : ℚ) * 8 := by
  sorry

#check build_time_relation

end NUMINAMATH_CALUDE_build_time_relation_l3422_342217
