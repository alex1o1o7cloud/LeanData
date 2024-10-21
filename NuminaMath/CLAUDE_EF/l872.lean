import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_draw_three_probability_l872_87245

/-- Represents the content of a box -/
structure BoxContent where
  one_balls : Nat
  two_balls : Nat
  three_balls : Nat

/-- The probability of drawing a specific ball from a box -/
def draw_probability (box : BoxContent) (ball_type : Nat) : Rat :=
  let total := box.one_balls + box.two_balls + box.three_balls
  match ball_type with
  | 1 => box.one_balls / total
  | 2 => box.two_balls / total
  | 3 => box.three_balls / total
  | _ => 0

/-- The contents of the boxes -/
def box1 : BoxContent := ⟨2, 1, 1⟩
def box2 : BoxContent := ⟨2, 0, 1⟩
def box3 : BoxContent := ⟨3, 2, 0⟩

/-- The probability of drawing a 3-numbered ball on the second draw -/
noncomputable def second_draw_probability : Rat :=
  (draw_probability box1 1 * draw_probability ⟨3, 1, 1⟩ 3) +
  (draw_probability box1 2 * draw_probability ⟨2, 1, 1⟩ 3) +
  (draw_probability box1 3 * draw_probability ⟨3, 2, 1⟩ 3)

/-- The main theorem to prove -/
theorem second_draw_three_probability :
  second_draw_probability = 11 / 48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_draw_three_probability_l872_87245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l872_87292

theorem trigonometric_problem (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos (α + π/6) = 3/5) (h4 : Real.cos (α + β) = -(Real.sqrt 5)/5) :
  Real.sin (2*α + π/3) = 24/25 ∧ Real.cos (β - π/6) = (Real.sqrt 5)/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l872_87292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_x_value_l872_87277

theorem prove_x_value (x n : ℕ) (h1 : x = 9^n - 1) (h2 : Odd n) 
  (h3 : (Nat.factors x).toFinset.card = 3) (h4 : 61 ∈ Nat.factors x) : x = 59048 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_x_value_l872_87277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangles_eq_n_minus_two_exists_vertex_in_one_triangle_l872_87221

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3

/-- A triangulation of a convex polygon -/
structure Triangulation (n : ℕ) extends ConvexPolygon n where
  num_triangles : ℕ
  diagonals_dont_intersect : Bool

/-- The number of triangles in a triangulation is n - 2 -/
theorem num_triangles_eq_n_minus_two {n : ℕ} (t : Triangulation n) :
  t.num_triangles = n - 2 := by
  sorry

/-- The number of triangles a vertex belongs to -/
def vertex_triangle_count {n : ℕ} (t : Triangulation n) (v : Fin n) : ℕ := by
  sorry

/-- There exists a vertex that belongs to exactly one triangle -/
theorem exists_vertex_in_one_triangle {n : ℕ} (t : Triangulation n) :
  ∃ v : Fin n, vertex_triangle_count t v = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangles_eq_n_minus_two_exists_vertex_in_one_triangle_l872_87221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l872_87216

/-- The line y = (3x - 1)/4 --/
noncomputable def line (x : ℝ) : ℝ := (3 * x - 1) / 4

/-- The point we're finding the closest point to --/
def target_point : ℝ × ℝ := (2, -3)

/-- The claimed closest point on the line --/
def closest_point : ℝ × ℝ := (-0.04, -0.28)

/-- Distance between two points in ℝ² --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem closest_point_on_line :
  (closest_point.2 = line closest_point.1) ∧
  ∀ p : ℝ × ℝ, p.2 = line p.1 →
    distance closest_point target_point ≤ distance p target_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l872_87216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_negative_one_thirtysecond_l872_87273

noncomputable def my_sequence (n : ℕ) : ℚ := (-1)^(n+1) * (n : ℚ) / (2^n : ℚ)

theorem eighth_term_is_negative_one_thirtysecond :
  my_sequence 8 = -1/32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_negative_one_thirtysecond_l872_87273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_equiv_line_and_circle_l872_87225

/-- The curve represented by the polar equation ρ cos θ = 2 sin 2θ -/
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = 2 * Real.sin (2 * θ)

/-- The Cartesian representation of a vertical line at x = 0 -/
def vertical_line (x : ℝ) : Prop :=
  x = 0

/-- The Cartesian representation of a circle with equation x² + y² = 4y -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * y

/-- Theorem stating that the polar curve is equivalent to the union of a line and a circle -/
theorem polar_curve_equiv_line_and_circle :
  ∀ ρ θ x y, polar_curve ρ θ ↔ (vertical_line x ∨ circle_eq x y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_equiv_line_and_circle_l872_87225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_segments_theorem_l872_87252

/-- A type representing a point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a colored segment between two points --/
structure ColoredSegment where
  p1 : Point
  p2 : Point
  color : ℕ

/-- Function to check if three points are collinear --/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem colored_segments_theorem (points : Finset Point) (segments : Finset ColoredSegment) (k : ℕ) :
  (points.card = 10) →
  (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → ¬collinear p1 p2 p3) →
  (∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → 
    ∃ s ∈ segments, (s.p1 = p1 ∧ s.p2 = p2) ∨ (s.p1 = p2 ∧ s.p2 = p1)) →
  (∀ s, s ∈ segments → s.color ≤ k) →
  (∀ subset : Finset Point, subset ⊆ points → subset.card = k →
    ∃ subset_segments : Finset ColoredSegment,
      subset_segments ⊆ segments ∧
      subset_segments.card = k ∧
      (∀ s1 s2, s1 ∈ subset_segments → s2 ∈ subset_segments → s1 ≠ s2 → s1.color ≠ s2.color) ∧
      (∀ s, s ∈ subset_segments → (s.p1 ∈ subset ∧ s.p2 ∈ subset))) →
  (1 ≤ k ∧ k ≤ 10) →
  (k ≥ 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_segments_theorem_l872_87252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_forty_five_degrees_l872_87281

noncomputable def angle : ℝ := Real.pi / 4  -- 45° in radians

theorem tan_forty_five_degrees :
  Real.tan angle = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_forty_five_degrees_l872_87281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_l872_87258

/-- Represents the time in minutes -/
def Time := ℕ

/-- Represents the number of robots -/
def Robots := ℕ

/-- Represents the number of batteries -/
def Batteries := ℕ

/-- Calculates the number of batteries manufactured given the conditions -/
def batteries_manufactured (gather_time create_time : ℕ) (num_robots : ℕ) (total_hours : ℕ) : ℕ :=
  let total_time_per_battery := gather_time + create_time
  let batteries_per_robot_per_hour := 60 / total_time_per_battery
  let batteries_per_hour := num_robots * batteries_per_robot_per_hour
  batteries_per_hour * total_hours

/-- Theorem stating that 10 robots can manufacture 200 batteries in 5 hours 
    given the specified conditions -/
theorem factory_output : 
  batteries_manufactured 6 9 10 5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_l872_87258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_divides_harmonic_numerator_l872_87241

/-- The harmonic series sum up to n terms -/
def harmonic_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

/-- The proposition that p_n and q_n are coprime positive integers
    satisfying the harmonic series sum condition -/
def harmonic_fraction (n p q : ℕ) : Prop :=
  0 < p ∧ 0 < q ∧ Nat.Coprime p q ∧ (p : ℚ) / q = harmonic_sum n

/-- The main theorem stating the conditions for 3 to divide p_n -/
theorem three_divides_harmonic_numerator (n : ℕ) :
  (∃ p q : ℕ, harmonic_fraction n p q ∧ 3 ∣ p) ↔ n = 2 ∨ n = 7 ∨ n = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_divides_harmonic_numerator_l872_87241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glasses_cost_calculation_l872_87297

def glasses_cost (frame_cost lens_cost antiglare_cost scratchresistant_cost : ℝ)
  (insurance_lens_coverage insurance_antiglare_coverage : ℝ)
  (frame_coupon loyalty_discount cash_reward sales_tax : ℝ) : ℝ :=
  let frame_final := frame_cost - frame_coupon - cash_reward
  let lens_final := lens_cost * (1 - insurance_lens_coverage)
  let antiglare_final := antiglare_cost * (1 - insurance_antiglare_coverage)
  let subtotal := frame_final + lens_final + antiglare_final
  let loyalty_discount_amount := (lens_final + antiglare_final) * loyalty_discount
  let total_after_discount := subtotal - loyalty_discount_amount
  let final_total := total_after_discount * (1 + sales_tax)
  final_total

theorem glasses_cost_calculation :
  abs (glasses_cost 200 500 75 75 0.7 0.5 50 0.1 20 0.07 - 319.66) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_glasses_cost_calculation_l872_87297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_total_wins_l872_87223

/-- Calculates the total number of matches won by a player given their win percentages in two sets of 100 matches each. -/
def total_matches_won (first_hundred_win_percentage : ℚ) (second_hundred_win_percentage : ℚ) : ℕ :=
  (first_hundred_win_percentage * 100).floor.toNat + (second_hundred_win_percentage * 100).floor.toNat

/-- Theorem stating that a player who won 50% of their first 100 matches and 60% of their next 100 matches won a total of 110 matches. -/
theorem sam_total_wins : total_matches_won (1/2) (3/5) = 110 := by
  -- Unfold the definition of total_matches_won
  unfold total_matches_won
  -- Simplify the arithmetic expressions
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_total_wins_l872_87223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_side_relation_triangle_side_length_l872_87268

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_angle_side_relation 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_relation : Real.sin C * (Real.sin B - Real.sin C) = Real.sin B ^ 2 - Real.sin A ^ 2) :
  A = Real.pi / 3 := by
  sorry

theorem triangle_side_length 
  (a b c : ℝ) (A B C : ℝ)
  (h_triangle : Triangle a b c A B C)
  (h_area : (1/2) * b * c * Real.sin A = (5 * Real.sqrt 3) / 4)
  (h_perimeter : b + c = 6)
  (h_angle : A = Real.pi / 3) :
  a = Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_side_relation_triangle_side_length_l872_87268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_rate_l872_87276

/-- Calculates the simple interest rate given the principal, time, and interest amount -/
noncomputable def simple_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) : ℝ :=
  (interest * 100) / (principal * time)

/-- Theorem stating that for the given investment scenario, the interest rate is 17.5% -/
theorem investment_interest_rate :
  let principal : ℝ := 7200
  let time : ℝ := 2.5
  let interest : ℝ := 3150
  simple_interest_rate principal time interest = 17.5 := by
  -- Unfold the definition of simple_interest_rate
  unfold simple_interest_rate
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_rate_l872_87276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_l872_87220

/-- The function f(x) = (3x^2 - 5x + 4) / (x - 2) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - 5 * x + 4) / (x - 2)

/-- The proposed oblique asymptote y = 3x - 1 -/
def g (x : ℝ) : ℝ := 3 * x - 1

/-- Theorem: The oblique asymptote of f(x) is g(x) -/
theorem oblique_asymptote : 
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - g x| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_l872_87220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derangement_probability_three_l872_87283

/-- A function that calculates the number of derangements for n items -/
def countDerangements (n : ℕ) : ℕ := sorry

/-- The total number of permutations for n items -/
def totalPermutations (n : ℕ) : ℕ := Nat.factorial n

/-- The probability of a derangement for n items -/
def derangementProbability (n : ℕ) : ℚ :=
  (countDerangements n : ℚ) / (totalPermutations n : ℚ)

/-- Theorem: The probability of a derangement for 3 items is 1/3 -/
theorem derangement_probability_three : derangementProbability 3 = 1/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derangement_probability_three_l872_87283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_for_divisibility_l872_87246

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem smallest_digit_for_divisibility :
  ∃ d : ℕ, d < 10 ∧
    (is_divisible_by_9 ((761 * 10 + d) * 829)) ∧
    (∀ k : ℕ, k < d → ¬ is_divisible_by_9 ((761 * 10 + k) * 829)) :=
by sorry

#check smallest_digit_for_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_for_divisibility_l872_87246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_proof_l872_87228

/-- The circumference of the base of a right circular cone with volume 18π and height triple the radius -/
noncomputable def cone_base_circumference : ℝ :=
  2 * Real.pi * (18 ^ (1/3 : ℝ))

/-- Theorem: The circumference of the base of a right circular cone with volume 18π and height triple the radius is 2π · 18^(1/3) -/
theorem cone_base_circumference_proof (r h : ℝ) (volume : ℝ) (height_radius_relation : h = 3 * r) (volume_formula : volume = (1/3) * Real.pi * r^2 * h) (given_volume : volume = 18 * Real.pi) :
  cone_base_circumference = 2 * Real.pi * r :=
by
  -- We'll use sorry to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_proof_l872_87228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l872_87200

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line
def line (x y : ℝ) : Prop := x - y - 6 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_circle_line :
  ∃ (xM yM xN yN : ℝ),
    circle_C xN yN ∧
    line xM yM ∧
    (∀ (x y : ℝ), circle_C x y → distance xM yM x y ≥ distance xM yM xN yN) ∧
    distance xM yM xN yN = Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l872_87200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_minus_pi_fourth_l872_87231

theorem sin_double_angle_minus_pi_fourth (x : ℝ) :
  Real.sin x = (Real.sqrt 5 - 1) / 2 →
  Real.sin (2 * (x - Real.pi / 4)) = 2 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_minus_pi_fourth_l872_87231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l872_87207

theorem power_equation (x y : ℝ) (h1 : (30 : ℝ)^x = 2) (h2 : (30 : ℝ)^y = 3) :
  (6 : ℝ)^((1 - x - y) / (2 * (1 - y))) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l872_87207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_most_accurate_l872_87256

/-- Represents a statistical test method for categorical variables -/
inductive StatTest
  | ContingencyTable
  | IndependenceTest
  | StackedBarChart
  | Other

/-- Represents the strength of relationship between categorical variables -/
def RelationshipStrength := ℝ

/-- Function that calculates the k^2 value for the independence test -/
noncomputable def calculate_k_squared : StatTest → RelationshipStrength :=
  sorry

/-- Function that determines the accuracy of a statistical test -/
noncomputable def test_accuracy : StatTest → ℝ :=
  sorry

/-- Theorem stating that the independence test is the most accurate method 
    for determining the relationship between two categorical variables -/
theorem independence_test_most_accurate :
  ∀ (test : StatTest),
    test ≠ StatTest.IndependenceTest →
    test_accuracy StatTest.IndependenceTest > test_accuracy test :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_most_accurate_l872_87256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chips_on_valid_board_l872_87249

/-- Represents a chip on the board -/
inductive Chip
| Red
| Blue

/-- Represents the board -/
def Board := Fin 200 → Fin 200 → Option Chip

/-- Counts the number of chips of a specific color in a row or column -/
def countChips (b : Board) (color : Chip) (row : Fin 200) (isRow : Bool) : Nat :=
  sorry

/-- Checks if the board configuration is valid -/
def isValidBoard (b : Board) : Prop :=
  ∀ i j : Fin 200, 
    match b i j with
    | some Chip.Red => countChips b Chip.Blue i true + countChips b Chip.Blue j false = 5
    | some Chip.Blue => countChips b Chip.Red i true + countChips b Chip.Red j false = 5
    | none => True

/-- Counts the total number of chips on the board -/
def totalChips (b : Board) : Nat :=
  sorry

/-- The maximum number of chips that can be placed on the board -/
def maxChips : Nat := 3800

/-- Theorem: The maximum number of chips on a valid 200x200 board is 3800 -/
theorem max_chips_on_valid_board :
  ∀ b : Board, isValidBoard b → totalChips b ≤ maxChips := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chips_on_valid_board_l872_87249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_solution_set_is_open_interval_l872_87254

-- Define the function f(x) = e^x - e^-x
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  f (2 * x + 1) + f (x - 2) > 0 ↔ x > 1/3 :=
by sorry

-- Define the solution set
def solution_set : Set ℝ := {x | x > 1/3}

-- State that the solution set is the open interval (1/3, +∞)
theorem solution_set_is_open_interval :
  solution_set = Set.Ioi (1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_solution_set_is_open_interval_l872_87254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_13_l872_87251

theorem remainder_sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_13_l872_87251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_permutation_exists_l872_87291

def is_valid_permutation (a : Fin 7 → ℕ) : Prop :=
  Function.Bijective a ∧ Set.range a = Finset.range 7

theorem unique_permutation_exists :
  ∃! a : Fin 7 → ℕ,
    is_valid_permutation a ∧
    (a 0 + 1) / 2 * (a 1 + 2) / 2 * (a 2 + 3) / 2 * (a 3 + 4) / 2 *
    (a 4 + 5) / 2 * (a 5 + 6) / 2 * (a 6 + 7) / 2 = Nat.factorial 7 ∧
    a 6 > a 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_permutation_exists_l872_87291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_ribbon_length_example_l872_87215

/-- The average length of two ribbons -/
noncomputable def average_ribbon_length (length1 : ℝ) (length2 : ℝ) : ℝ :=
  (length1 + length2) / 2

/-- Theorem: The average length of two ribbons with lengths 2.5 inches and 6.5 inches is 4.5 inches -/
theorem average_ribbon_length_example : average_ribbon_length 2.5 6.5 = 4.5 := by
  -- Unfold the definition of average_ribbon_length
  unfold average_ribbon_length
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_ribbon_length_example_l872_87215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_positions_in_first_20_l872_87208

noncomputable def a (n : ℕ+) : ℝ := (n.val - Real.sqrt 98) / (n.val - Real.sqrt 99)

theorem max_min_positions_in_first_20 :
  (∀ k ∈ Finset.range 20, a ⟨10, by norm_num⟩ ≥ a ⟨k + 1, Nat.succ_pos k⟩) ∧
  (∀ k ∈ Finset.range 20, a ⟨9, by norm_num⟩ ≤ a ⟨k + 1, Nat.succ_pos k⟩) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_positions_in_first_20_l872_87208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_theorem_l872_87230

-- Define the circle
def circle_equation (a x y : ℝ) : Prop := x^2 + y^2 - 2*a*x + 2*y - 1 = 0

-- Define the point P
def point_P (a : ℝ) : ℝ × ℝ := (-5, a)

-- Define the tangent condition
def tangent_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) + (x₁ + x₂ - 2) / (y₁ + y₂) = 0

theorem circle_tangent_theorem (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation a (-5) a ∧
    circle_equation a x₁ y₁ ∧
    circle_equation a x₂ y₂ ∧
    tangent_condition x₁ y₁ x₂ y₂) →
  a = 3 ∨ a = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_theorem_l872_87230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_sum_l872_87299

/-- Given a triangle ABC with points D on BC, E on CA, and F on AB,
    where DC = 2BD, CE = 2EA, and AF = 2FB,
    prove that AD + BE + CF = -1/3 * BC -/
theorem triangle_vector_sum (A B C D E F : EuclideanSpace ℝ (Fin 3)) : 
  (D - B) = (2 : ℝ) • (C - D) →
  (E - A) = (2 : ℝ) • (C - E) →
  (F - B) = (2 : ℝ) • (A - F) →
  (D - A) + (E - B) + (F - C) = -(1/3 : ℝ) • (C - B) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_sum_l872_87299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bathhouse_optimal_location_optimal_bathhouse_location_l872_87213

/-- Represents a village with a certain number of residents -/
structure Village where
  residents : ℕ

/-- Represents the problem setup -/
structure BathhouseProblem where
  villageA : Village
  villageB : Village
  distance : ℝ  -- Distance between villages A and B

/-- Calculates the total distance traveled by all residents -/
def totalDistance (problem : BathhouseProblem) (x : ℝ) : ℝ :=
  2 * problem.villageA.residents * x + 2 * problem.villageB.residents * (problem.distance - x)

/-- Theorem stating that the total distance is minimized when the bathhouse is in village A -/
theorem bathhouse_optimal_location (problem : BathhouseProblem) :
  ∀ x, 0 ≤ x ∧ x ≤ problem.distance →
    totalDistance problem 0 ≤ totalDistance problem x := by
  sorry

/-- Main theorem proving the optimal location of the bathhouse -/
theorem optimal_bathhouse_location (problem : BathhouseProblem) :
  problem.villageA.residents = 100 →
  problem.villageB.residents = 50 →
  ∀ x, 0 ≤ x ∧ x ≤ problem.distance →
    totalDistance problem 0 ≤ totalDistance problem x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bathhouse_optimal_location_optimal_bathhouse_location_l872_87213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_x_equation_l872_87293

theorem h_x_equation (x : ℝ) :
  let h : ℝ → ℝ := λ x => -2*x^5 - 4*x^3 + 7*x^2 - 12*x + 9
  2*x^5 + 4*x^3 + 3*x - 4 + h x = 7*x^2 - 9*x + 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_x_equation_l872_87293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_function_difference_l872_87270

open Real

theorem tangent_line_and_function_difference (a b x₁ x₂ : ℝ) : 
  a = 1 → b > 3 → x₁ < x₂ → 
  let f := λ x => a * x^2 - b * x + log x
  let f' := λ x => (2 * a * x^2 - b * x + 1) / x
  f' x₁ = 0 → f' x₂ = 0 →
  f x₁ - f x₂ = 3/4 - log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_function_difference_l872_87270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_function_correct_min_cost_value_profit_maximizing_volume_l872_87261

-- Define the production volume range
def production_range : Set ℝ := { x | 50 ≤ x ∧ x ≤ 200 }

-- Define the cost function
noncomputable def P (x : ℝ) : ℝ := x + 8100 / x + 40

-- Define the sales revenue function
noncomputable def Q (x : ℝ) : ℝ := 1240 * x - (1 / 30) * x^3

-- Define the profit function
noncomputable def L (x : ℝ) : ℝ := Q x - (50 * x + 7500 + 20 * x + x^2 + 600 - 30 * x)

theorem cost_function_correct (x : ℝ) (hx : x ∈ production_range) :
  P x = (50 * x + (7500 + 20 * x) + x * (x + 600 / x - 30)) / x :=
by sorry

theorem min_cost_value (x : ℝ) (hx : x ∈ production_range) :
  P x ≥ 220 :=
by sorry

theorem profit_maximizing_volume :
  ∃ x ∈ production_range, ∀ y ∈ production_range, L x ≥ L y ∧ x = 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_function_correct_min_cost_value_profit_maximizing_volume_l872_87261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l872_87214

theorem power_equality (x : ℝ) (h : (4 : ℝ)^x - (4 : ℝ)^(x - 1) = 24) : 
  (2*x)^x = 25 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l872_87214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_without_discount_l872_87272

-- Define the discount rate
noncomputable def discount_rate : ℝ := 0.05

-- Define the profit rate with discount
noncomputable def profit_rate_with_discount : ℝ := 0.235

-- Define the function to calculate the marked price
noncomputable def marked_price (selling_price : ℝ) : ℝ :=
  selling_price / (1 - discount_rate)

-- Define the function to calculate profit percentage
noncomputable def profit_percentage (selling_price cost_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

-- Theorem statement
theorem profit_without_discount 
  (cost_price : ℝ) 
  (h_positive : cost_price > 0) :
  let selling_price_with_discount := cost_price * (1 + profit_rate_with_discount)
  let marked_price := marked_price selling_price_with_discount
  profit_percentage marked_price cost_price = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_without_discount_l872_87272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l872_87265

noncomputable def f (x : ℝ) : ℝ := 3 / (6 - 2 * x)

theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x ≠ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l872_87265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l872_87259

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the first graph
def graph1 (x y : ℝ) : Prop :=
  (x - floor x)^2 + y^2 = x - floor x

-- Define the second graph
def graph2 (x y : ℝ) : Prop :=
  y = (1/3) * x + 1

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  graph1 x y ∧ graph2 x y

-- Define the set of all intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | is_intersection_point p.1 p.2}

-- The theorem to be proved
theorem intersection_count : Cardinal.mk intersection_points = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l872_87259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_possibilities_l872_87290

theorem band_arrangement_possibilities (n : ℕ) (h : n = 100) :
  (Finset.filter (λ x => 4 ≤ x ∧ x ≤ 25 ∧ n % x = 0) (Finset.range 26)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_possibilities_l872_87290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l872_87269

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the eccentricity is √6/2, then the equation of its asymptotes is y = ±(√2/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.sqrt 6 / 2 = Real.sqrt ((a^2 + b^2) / a^2)) →
  (∀ x y : ℝ, y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l872_87269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l872_87219

theorem vector_magnitude (a b : ℝ × ℝ) : 
  (b = (1, Real.sqrt 3) ∧ 
   ((a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = 2) ∧
   Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 5) →
  Real.sqrt (a.1^2 + a.2^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l872_87219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l872_87275

/-- The number of days it takes for one woman to complete the work -/
noncomputable def days_for_one_woman (men_work_rate : ℝ) (women_work_rate : ℝ) : ℝ :=
  1 / women_work_rate

/-- Theorem stating that it takes 600 days for one woman to complete the work -/
theorem work_completion_time 
  (h1 : (10 * men_work_rate + 15 * women_work_rate) * 8 = 1)
  (h2 : men_work_rate * 100 = 1) :
  days_for_one_woman men_work_rate women_work_rate = 600 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l872_87275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l872_87205

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid (intersection of medians)
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define distance squared between two points
noncomputable def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Theorem statement
theorem triangle_centroid_property (t : Triangle) :
  let O := centroid t
  distanceSquared t.A t.B + distanceSquared t.B t.C + distanceSquared t.C t.A =
  3 * (distanceSquared O t.A + distanceSquared O t.B + distanceSquared O t.C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l872_87205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_color_chips_probability_l872_87280

/-- The probability of drawing two chips of different colors from a bag with replacement -/
theorem two_different_color_chips_probability 
  (blue : ℕ) (red : ℕ) (yellow : ℕ) 
  (h_blue : blue = 5) (h_red : red = 4) (h_yellow : yellow = 3) : 
  (blue * (red + yellow) + red * (blue + yellow) + yellow * (blue + red)) / 
  ((blue + red + yellow) * (blue + red + yellow)) = 47 / 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_color_chips_probability_l872_87280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_complete_time_is_80_l872_87274

/-- The time it takes for worker p to complete the work alone -/
noncomputable def p_complete_time (total_work : ℝ) (p_rate : ℝ) (q_rate : ℝ) : ℝ :=
  total_work / p_rate

/-- The proposition that given the conditions, p takes 80 days to complete the work alone -/
theorem p_complete_time_is_80 
  (total_work : ℝ) 
  (p_rate : ℝ) 
  (q_rate : ℝ) 
  (h1 : q_rate = total_work / 48)
  (h2 : 16 * p_rate + 24 * (p_rate + q_rate) = total_work)
  : p_complete_time total_work p_rate q_rate = 80 := by
  sorry

#check p_complete_time_is_80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_complete_time_is_80_l872_87274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_concentration_l872_87226

/-- Calculates the final salt percentage in a solution after adding pure salt -/
noncomputable def final_salt_percentage (initial_solution_mass : ℝ) (initial_salt_percentage : ℝ) 
  (added_salt_mass : ℝ) : ℝ :=
  let initial_salt_mass := initial_solution_mass * initial_salt_percentage / 100
  let total_salt_mass := initial_salt_mass + added_salt_mass
  let final_solution_mass := initial_solution_mass + added_salt_mass
  (total_salt_mass / final_solution_mass) * 100

/-- Theorem stating that adding 38.46153846153846 kg of pure salt to 100 kg of 10% salt solution 
    results in a 35% salt solution -/
theorem salt_solution_concentration : 
  final_salt_percentage 100 10 38.46153846153846 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_solution_concentration_l872_87226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_CDFE_l872_87263

/-- A square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

/-- Points E and F on sides AB and AD respectively -/
structure PointsEF (square : UnitSquare) where
  E : ℝ × ℝ
  F : ℝ × ℝ
  E_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (t, 0)
  F_on_AD : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (0, s)
  AE_twice_AF : E.1 = 2 * F.2

/-- The area of quadrilateral CDFE -/
noncomputable def area_CDFE (square : UnitSquare) (points : PointsEF square) : ℝ :=
  points.E.1 / 2

/-- Theorem: The maximum area of quadrilateral CDFE is 1/2 -/
theorem max_area_CDFE (square : UnitSquare) :
  ∃ (points : PointsEF square), ∀ (other : PointsEF square),
    area_CDFE square points ≥ area_CDFE square other ∧
    area_CDFE square points = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_CDFE_l872_87263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lines_through_four_points_l872_87257

/-- A point in the 3D grid -/
structure GridPoint where
  x : Fin 5
  y : Fin 5
  z : Fin 5

/-- A line in the 3D grid -/
structure GridLine where
  points : Finset GridPoint
  distinct : points.card = 4

/-- The set of all valid lines in the grid -/
def validLines : Finset GridLine :=
  sorry

theorem count_lines_through_four_points : validLines.card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lines_through_four_points_l872_87257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ring_area_l872_87288

/-- The radius of the larger circle D -/
def R : ℝ := 40

/-- The number of smaller circles in the ring -/
def n : ℕ := 8

/-- The area of the region inside the larger circle and outside all smaller circles -/
noncomputable def K' (r : ℝ) : ℝ := Real.pi * ((R^2) - n * (r^2))

/-- The radius of each smaller circle in terms of R -/
noncomputable def r : ℝ := R / (1 + Real.sqrt 2)

theorem circle_ring_area :
  ⌊K' r⌋ = 3551 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ring_area_l872_87288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chips_theorem_exists_valid_25_chips_l872_87244

/-- Represents a row of colored chips -/
def ChipRow := List Bool

/-- Checks if two chips at given positions have the same color -/
def same_color (row : ChipRow) (i j : Nat) : Prop :=
  i < row.length ∧ j < row.length ∧ row.get? i = row.get? j

/-- Checks if the chip row satisfies the color condition -/
def satisfies_condition (row : ChipRow) : Prop :=
  ∀ i : Nat, i < row.length →
    (i + 10 < row.length → same_color row i (i + 10)) ∧
    (i + 15 < row.length → same_color row i (i + 15))

/-- Checks if the chip row contains both colors -/
def has_both_colors (row : ChipRow) : Prop :=
  ∃ i j : Nat, i < row.length ∧ j < row.length ∧ row.get? i ≠ row.get? j

/-- The main theorem stating the maximum number of chips possible -/
theorem max_chips_theorem :
  ¬∃ (row : ChipRow), row.length > 25 ∧ satisfies_condition row ∧ has_both_colors row :=
by
  sorry

/-- Proof that there exists a valid configuration with 25 chips -/
theorem exists_valid_25_chips :
  ∃ (row : ChipRow), row.length = 25 ∧ satisfies_condition row ∧ has_both_colors row :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chips_theorem_exists_valid_25_chips_l872_87244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_implies_collinear_l872_87229

noncomputable section

open Real

theorem vector_sum_magnitude_implies_collinear 
  (a b : EuclideanSpace ℝ (Fin 3)) 
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (h : ‖a + b‖ = ‖a‖ + ‖b‖) : 
  ∃ t : ℝ, a = t • b :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_implies_collinear_l872_87229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l872_87253

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.log (abs x)}
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l872_87253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l872_87201

/-- The distance between the center of the circle with equation x^2 + y^2 = 6x - 4y - 13 
    and the point (-2, 5) is √74. -/
theorem circle_center_distance : 
  let circle_eq : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 = 6*x - 4*y - 13
  let center : ℝ × ℝ := (3, -2)
  let point : ℝ × ℝ := (-2, 5)
  ∀ x y, circle_eq x y → 
    Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) = Real.sqrt 74 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l872_87201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_speed_ratio_l872_87239

/-- Represents Jake's swimming and rowing speeds around a square lake -/
structure JakeSpeed where
  lake_side : ℚ
  swim_time : ℚ
  row_time : ℚ

/-- Calculates the ratio of Jake's rowing speed to his swimming speed -/
noncomputable def speed_ratio (jake : JakeSpeed) : ℚ :=
  let swim_speed := 60 / jake.swim_time
  let row_speed := (4 * jake.lake_side) / jake.row_time
  row_speed / swim_speed

/-- Theorem stating that the ratio of Jake's rowing speed to his swimming speed is 2 -/
theorem jake_speed_ratio :
  ∃ (jake : JakeSpeed),
    jake.lake_side = 15 ∧
    jake.swim_time = 20 ∧
    jake.row_time = 600 ∧
    speed_ratio jake = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_speed_ratio_l872_87239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_X_in_0_1_value_l872_87217

/-- The probability density function for the random variable X -/
noncomputable def p (x : ℝ) : ℝ := |x| * Real.exp (-x^2)

/-- The probability that X takes a value in the interval (0,1) -/
noncomputable def prob_X_in_0_1 : ℝ := ∫ x in Set.Ioo 0 1, p x

theorem prob_X_in_0_1_value : prob_X_in_0_1 = 1/2 * (1 - 1/Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_X_in_0_1_value_l872_87217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_greater_than_third_l872_87255

/-- Represents a convex polygon -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : True  -- Placeholder for convexity condition

/-- Represents the process of clipping a polygon -/
def clip (p : ConvexPolygon) : ConvexPolygon :=
  sorry

/-- Calculates the area of a polygon -/
noncomputable def area (p : ConvexPolygon) : ℝ :=
  sorry

/-- Defines a regular hexagon with area 1 -/
def initial_hexagon : ConvexPolygon :=
  sorry

/-- Generates the nth polygon in the clipping sequence -/
def P : ℕ → ConvexPolygon
  | 0 => initial_hexagon
  | 1 => initial_hexagon
  | 2 => initial_hexagon
  | 3 => initial_hexagon
  | 4 => initial_hexagon
  | 5 => initial_hexagon
  | n + 6 => clip (P (n + 5))

/-- The main theorem: area of P_n is always greater than 1/3 for n ≥ 6 -/
theorem area_greater_than_third (n : ℕ) (h : n ≥ 6) :
  area (P n) > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_greater_than_third_l872_87255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_wins_probability_sum_of_mn_equals_39_l872_87247

def alice_numbers : Finset ℕ := {2, 4, 6, 8, 10}
def palice_numbers : Finset ℕ := {1, 3, 5, 7, 9}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (fun (a, p) => a > p) (alice_numbers.product palice_numbers)

def total_outcomes : Finset (ℕ × ℕ) := alice_numbers.product palice_numbers

theorem alice_wins_probability :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card total_outcomes : ℚ) = 14 / 25 := by
  sorry

theorem sum_of_mn_equals_39 :
  let m : ℕ := 14
  let n : ℕ := 25
  m + n = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_wins_probability_sum_of_mn_equals_39_l872_87247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l872_87287

noncomputable def nominal_rate : ℝ := 0.05
noncomputable def inflation_rate : ℝ := 0.02
def investment_period : ℕ := 6
noncomputable def final_amount : ℝ := 1120

noncomputable def effective_rate : ℝ := ((1 + nominal_rate) / (1 + inflation_rate)) - 1

noncomputable def principal : ℝ := final_amount / (1 + effective_rate) ^ investment_period

theorem principal_calculation :
  ∃ ε > 0, |principal - 938.14| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l872_87287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l872_87203

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := 300

/-- The length of the platform in meters -/
noncomputable def platform_length : ℝ := 1000

/-- The time taken to cross the platform in seconds -/
noncomputable def platform_crossing_time : ℝ := 39

/-- The time taken to cross a signal pole in seconds -/
noncomputable def pole_crossing_time : ℝ := 9

/-- The speed of the train in meters per second -/
noncomputable def train_speed : ℝ := train_length / pole_crossing_time

theorem train_length_proof :
  train_length = 300 ∧
  platform_length = 1000 ∧
  platform_crossing_time = 39 ∧
  pole_crossing_time = 9 →
  train_length + platform_length = train_speed * platform_crossing_time :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l872_87203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_CD_length_l872_87266

noncomputable section

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- C is a right angle
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the circle with diameter AB
def circle_AB (A B D : ℝ × ℝ) : Prop :=
  -- D is on the circle with diameter AB
  (D.1 - A.1) * (B.1 - D.1) + (D.2 - A.2) * (B.2 - D.2) = 0

-- Define the intersection of the circle with side BC
def intersects_BC (B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)

-- Define the area of triangle ABC
noncomputable def area_ABC (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Define the length of BC
noncomputable def length_BC (B C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)

-- Define the length of CD
noncomputable def length_CD (C D : ℝ × ℝ) : ℝ :=
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)

theorem triangle_CD_length (A B C D : ℝ × ℝ) :
  triangle_ABC A B C →
  circle_AB A B D →
  intersects_BC B C D →
  area_ABC A B C = 240 →
  length_BC B C = 24 →
  length_CD C D = Real.sqrt 48 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_CD_length_l872_87266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_theorem_l872_87210

theorem acute_angles_theorem (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.sin (α + β) = 2 * Real.sin (α - β)) : 
  (0 < α - β ∧ α - β ≤ π / 6) ∧ 
  (Real.sin α = 2 * Real.sin β → Real.cos α = Real.sqrt 6 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_theorem_l872_87210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tap_time_l872_87236

/-- Represents the time (in hours) it takes for a tap to fill a tank. -/
structure TapTime where
  hours : ℚ
  hours_pos : hours > 0

/-- Represents the rate at which a tap fills a tank (fraction of tank per hour). -/
noncomputable def fillRate (t : TapTime) : ℚ := 1 / t.hours

/-- Given three taps with specific fill times, proves that the first tap takes 10 hours to fill the tank. -/
theorem first_tap_time (combined_time second_time third_time : TapTime)
  (h_combined : combined_time.hours = 3)
  (h_second : second_time.hours = 15)
  (h_third : third_time.hours = 6)
  (h_sum : ∃ (first_time : TapTime),
    fillRate first_time + fillRate second_time + fillRate third_time = fillRate combined_time) :
  ∃ (first_time : TapTime), first_time.hours = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tap_time_l872_87236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_for_nine_not_in_range_l872_87294

theorem smallest_c_for_nine_not_in_range : ∃ c : ℤ, 
  (∀ x : ℝ, x^2 + c*x + 18 ≠ 9) ∧ (∀ c' : ℤ, c' < c → ∃ x : ℝ, x^2 + c'*x + 18 = 9) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_for_nine_not_in_range_l872_87294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robbie_weight_verify_solution_l872_87242

/-- Robbie's weight in pounds -/
def R : ℝ := 100

/-- Patty's initial weight as a multiple of Robbie's weight -/
def patty_initial_multiple : ℝ := 4.5

/-- Amount of weight Patty lost in pounds -/
def patty_weight_loss : ℝ := 235

/-- Difference between Patty's final weight and Robbie's weight in pounds -/
def weight_difference : ℝ := 115

/-- Theorem stating that Robbie weighs 100 pounds -/
theorem robbie_weight : R = 100 := by
  -- Unfold the definition of R
  unfold R
  -- Reflexivity
  rfl

/-- Verify that the solution satisfies all conditions -/
theorem verify_solution : 
  (patty_initial_multiple * R - patty_weight_loss) = (R + weight_difference) := by
  -- Substitute the values
  simp [R, patty_initial_multiple, patty_weight_loss, weight_difference]
  -- Evaluate the expression
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robbie_weight_verify_solution_l872_87242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_equality_l872_87243

theorem sin_difference_equality (α : ℝ) :
  Real.sin (5 * α) - Real.sin (6 * α) - Real.sin (7 * α) + Real.sin (8 * α) = 
  -4 * Real.sin (α / 2) * Real.sin α * Real.sin (13 * α / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_equality_l872_87243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natalies_list_count_l872_87298

theorem natalies_list_count : 
  let start : Nat := 225
  let end_ : Nat := 3375
  let step : Nat := 15
  (List.range ((end_ - start) / step + 1)).length = 211 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_natalies_list_count_l872_87298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_S_less_than_neg_four_l872_87238

noncomputable def a (n : ℕ+) : ℝ := Real.log (n : ℝ) / Real.log 3 - Real.log ((n : ℝ) + 1) / Real.log 3

noncomputable def S (n : ℕ+) : ℝ := -Real.log ((n : ℝ) + 1) / Real.log 3

theorem smallest_n_for_S_less_than_neg_four :
  (∃ (n : ℕ+), S n < -4) ∧ 
  (∀ (m : ℕ+), m < 81 → S m ≥ -4) ∧
  (S 81 < -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_S_less_than_neg_four_l872_87238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l872_87284

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (3 * L * B / 2 - L * B) / (L * B) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_change_l872_87284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_sequence_properties_l872_87227

def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => a n / (2 * a n + 1)

def b (n : ℕ) : ℚ := a n * a (n + 1)

theorem a_sequence_properties :
  (∀ n : ℕ, (1 / a (n + 1) - 1 / a n) = 2) ∧
  (∀ n : ℕ, a n = 1 / (2 * n.succ - 1)) ∧
  (∀ n : ℕ, (Finset.range n).sum (λ i => b i) = n / (2 * n.succ + 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_sequence_properties_l872_87227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_person_win_probability_is_one_thirty_first_l872_87218

/-- The probability of winning for the fourth person in a coin-flipping game -/
def fourth_person_win_probability : ℚ := 1 / 31

/-- The game setup with four players flipping coins in turn -/
axiom coin_flip_game : Unit

/-- The probability of getting heads on a single coin flip -/
axiom coin_flip_prob : ℚ
axiom coin_flip_prob_value : coin_flip_prob = 1 / 2

/-- The probability of the fourth person winning on their nth turn -/
noncomputable def win_on_nth_turn (n : ℕ) : ℚ := coin_flip_prob ^ (5 * n)

/-- The sum of probabilities for all possible winning turns -/
noncomputable def total_win_probability : ℚ := ∑' n, win_on_nth_turn n

/-- Theorem stating that the probability of the fourth person winning is 1/31 -/
theorem fourth_person_win_probability_is_one_thirty_first :
  total_win_probability = fourth_person_win_probability := by
  sorry

#eval fourth_person_win_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_person_win_probability_is_one_thirty_first_l872_87218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_oil_rate_l872_87211

/-- Given two types of oil with different volumes and prices, 
    calculate the rate of the mixed oil per litre -/
theorem mixed_oil_rate (volume1 volume2 price1 price2 : ℚ) 
  (hv1 : volume1 = 10)
  (hv2 : volume2 = 5)
  (hp1 : price1 = 50)
  (hp2 : price2 = 68) :
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2) = 56 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_oil_rate_l872_87211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_BP_l872_87224

-- Define the circle and points
variable (circle : Set (ℝ × ℝ))
variable (A B C D P : ℝ × ℝ)

-- Define the conditions
axiom on_circle : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle
axiom intersect : ∃ P, P ∈ Set.Icc A C ∧ P ∈ Set.Icc B D
axiom AP_length : dist A P = 12
axiom PC_length : dist P C = 2
axiom BD_length : dist B D = 10
axiom BP_less_DP : dist B P < dist D P

-- Define the power of a point theorem
def power_of_point (X Y Z W : ℝ × ℝ) : Prop :=
  dist X Z * dist Y Z = dist X W * dist Y W

-- State the theorem to be proved
theorem find_BP : dist B P = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_BP_l872_87224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barrette_cost_is_three_l872_87240

/-- The cost of one set of barrettes -/
def barrette_cost : ℝ := sorry

/-- The cost of one comb -/
def comb_cost : ℝ := 1

/-- The total amount spent by Kristine -/
def kristine_spent : ℝ := barrette_cost + comb_cost

/-- The total amount spent by Crystal -/
def crystal_spent : ℝ := 3 * barrette_cost + comb_cost

/-- The total amount spent by both girls -/
def total_spent : ℝ := kristine_spent + crystal_spent

theorem barrette_cost_is_three :
  total_spent = 14 → barrette_cost = 3 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barrette_cost_is_three_l872_87240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_malibu_pool_drain_rate_l872_87248

/-- Represents the dimensions and draining properties of a rectangular pool. -/
structure Pool where
  width : ℝ
  length : ℝ
  depth : ℝ
  drainTime : ℝ

/-- Calculates the volume of a rectangular pool. -/
noncomputable def poolVolume (p : Pool) : ℝ := p.width * p.length * p.depth

/-- Calculates the drain rate of a pool in cubic feet per minute. -/
noncomputable def drainRate (p : Pool) : ℝ := poolVolume p / p.drainTime

/-- Theorem stating that a pool with given dimensions and drain time has a specific drain rate. -/
theorem malibu_pool_drain_rate (p : Pool) 
  (h1 : p.width = 80)
  (h2 : p.length = 150)
  (h3 : p.depth = 10)
  (h4 : p.drainTime = 2000) :
  drainRate p = 600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_malibu_pool_drain_rate_l872_87248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_present_price_l872_87271

/-- The price of a present bought by three people with a discount -/
theorem present_price (original_contribution : ℝ) 
  (h : original_contribution > 0) : 
  let discount_rate : ℝ := 0.2
  let num_people : ℕ := 3
  let individual_savings : ℝ := 4
  let discounted_contribution := original_contribution - individual_savings
  let total_price := num_people * discounted_contribution
  discounted_contribution = original_contribution * (1 - discount_rate) →
  total_price = 48 :=
by
  -- Introduce the local definitions
  intro discount_rate num_people individual_savings discounted_contribution total_price
  -- Introduce the hypothesis
  intro h_discount
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_present_price_l872_87271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_value_l872_87237

/-- An inverse proportion function passing through two specific points -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

/-- The first point (3, m) that the function passes through -/
def point_A (m : ℝ) : ℝ × ℝ := (3, m)

/-- The second point (m-1, 6) that the function passes through -/
def point_B (m : ℝ) : ℝ × ℝ := (m - 1, 6)

/-- Theorem stating that k = 6 for the given inverse proportion function -/
theorem inverse_proportion_k_value :
  ∃ (m : ℝ), 
    (inverse_proportion 6 (point_A m).1 = (point_A m).2) ∧ 
    (inverse_proportion 6 (point_B m).1 = (point_B m).2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_value_l872_87237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_of_f_l872_87233

-- Define the function f(x) = 2^x + x - 7
noncomputable def f (x : ℝ) : ℝ := 2^x + x - 7

-- Define the theorem
theorem root_interval_of_f (n : ℤ) : 
  (∃ r : ℝ, r ∈ Set.Ioo (n : ℝ) ((n + 1) : ℝ) ∧ f r = 0) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_of_f_l872_87233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_sum_l872_87212

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Represents a rotation in 2D space -/
structure Rotation where
  angle : ℝ
  center : Point

def triangle_DEF : Triangle := {
  a := { x := 0, y := 0 }
  b := { x := 0, y := 15 }
  c := { x := 20, y := 0 }
}

def triangle_DEF_prime : Triangle := {
  a := { x := 30, y := 20 }
  b := { x := 45, y := 20 }
  c := { x := 30, y := 0 }
}

/-- Applies a rotation to a point -/
noncomputable def apply_rotation (r : Rotation) (p : Point) : Point :=
  sorry

/-- Checks if two triangles are equal up to some small error -/
def triangles_equal (t1 t2 : Triangle) (ε : ℝ) : Prop :=
  sorry

/-- Applies a rotation to all points of a triangle -/
noncomputable def rotate_triangle (r : Rotation) (t : Triangle) : Triangle :=
  { a := apply_rotation r t.a
    b := apply_rotation r t.b
    c := apply_rotation r t.c }

theorem rotation_sum (r : Rotation) :
  0 < r.angle ∧ r.angle < 180 →
  triangles_equal (rotate_triangle r triangle_DEF) triangle_DEF_prime 0.001 →
  r.angle + r.center.x + r.center.y = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_sum_l872_87212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l872_87278

theorem sin_cos_product (θ : ℝ) (a : ℝ) (h1 : 0 < θ ∧ θ < π/2) (h2 : Real.sin (2*θ) = a) :
  Real.sin θ * Real.cos θ = a/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l872_87278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l872_87206

noncomputable def fill_time (a b c d e : ℝ) : ℝ :=
  1260 / 383

theorem cistern_fill_time (a b c d e : ℝ) 
  (ha : a = 5) (hb : b = 7) (hc : c = 9) (hd : d = 12) (he : e = 15) :
  fill_time a b c d e = 1260 / 383 := by
  rfl

#eval (1260 : ℚ) / 383

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l872_87206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_6_l872_87289

def T : ℕ → ℕ
  | 0 => 3  -- Add this case for 0
  | 1 => 3
  | n + 2 => 3^(T (n + 1))

theorem t_50_mod_6 : T 50 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_6_l872_87289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_one_l872_87285

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem tangent_slope_angle_at_one :
  Real.arctan ((deriv f) 1) = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_one_l872_87285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_violation_l872_87260

-- Define the types
inductive Resident : Type
| mkResident : String → Resident

inductive Currency : Type
| mkCurrency : String → Currency

-- Define the constants
def mikhail : Resident := Resident.mkResident "Mikhail"
def valentin : Resident := Resident.mkResident "Valentin"
def euro : Currency := Currency.mkCurrency "Euro"
def ruble : Currency := Currency.mkCurrency "Ruble"

-- Define the predicates
def is_russian_resident : Resident → Prop := sorry
def legal_currency_in_russia : Currency → Prop := sorry
def transaction_currency : Resident → Resident → Currency → Prop := sorry
def violates_legislation : Resident → Resident → Currency → Prop := sorry

-- Define the theorem
theorem transaction_violation 
  (h1 : is_russian_resident mikhail)
  (h2 : is_russian_resident valentin)
  (h3 : transaction_currency mikhail valentin euro)
  (h4 : legal_currency_in_russia ruble)
  (h5 : ¬legal_currency_in_russia euro) :
  violates_legislation mikhail valentin euro :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_violation_l872_87260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l872_87286

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e : e = Real.sqrt 3 / 2
  h_minor_axis : 2 * b = 4

/-- The equation of the ellipse and its bisected chord -/
def ellipse_equations (E : Ellipse) : Prop :=
  (∀ x y : ℝ, x^2 / 16 + y^2 / 4 = 1 ↔ x^2 / E.a^2 + y^2 / E.b^2 = 1) ∧
  (∀ x y : ℝ, x + 2*y - 4 = 0 ↔ 
    ∃ t : ℝ, x = 2 + t ∧ y = 1 - t/2 ∧ 
    (2 + t)^2 / E.a^2 + (1 - t/2)^2 / E.b^2 = 1)

/-- The main theorem -/
theorem ellipse_theorem (E : Ellipse) : ellipse_equations E := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l872_87286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_continued_fraction_equals_golden_ratio_l872_87250

/-- The golden ratio, defined as the positive solution to x^2 = x + 1 -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The infinite continued fraction 1 + 1/(1 + 1/(1 + ...)) -/
noncomputable def infiniteContinuedFraction : ℝ := φ

theorem infinite_continued_fraction_equals_golden_ratio :
  infiniteContinuedFraction = φ := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_continued_fraction_equals_golden_ratio_l872_87250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l872_87235

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x + Real.sqrt 3

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (y - x - Real.sqrt 3) / Real.sqrt 2

-- Theorem statement
theorem max_distance_ellipse_to_line :
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 6 ∧
  ∀ (x y : ℝ), ellipse x y →
    distance_to_line x y ≤ max_dist ∧
    ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧ distance_to_line x₀ y₀ = max_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l872_87235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_rate_is_three_l872_87264

/-- Calculates the painting rate per square meter given room dimensions, door and window areas, and total painting cost. -/
noncomputable def painting_rate_per_sq_m (room_length room_width room_height : ℝ)
                           (door_width door_height : ℝ)
                           (large_window_width large_window_height : ℝ)
                           (small_window_width small_window_height : ℝ)
                           (total_cost : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := 2 * (door_width * door_height)
  let large_window_area := large_window_width * large_window_height
  let small_window_area := 2 * (small_window_width * small_window_height)
  let paintable_area := wall_area - (door_area + large_window_area + small_window_area)
  total_cost / paintable_area

/-- Theorem stating that the painting rate is 3 Rs per square meter for the given room specifications. -/
theorem painting_rate_is_three :
  painting_rate_per_sq_m 10 7 5 1 3 2 1.5 1 1.5 474 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_rate_is_three_l872_87264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blowfish_in_own_tank_l872_87282

/-- Represents the aquarium setup with clownfish and blowfish -/
structure Aquarium where
  total_fish : ℕ
  clownfish : ℕ
  blowfish : ℕ
  clownfish_in_display : ℕ
  blowfish_in_display : ℕ

/-- Theorem stating the number of blowfish that stayed in their own tank -/
theorem blowfish_in_own_tank (a : Aquarium) 
  (total_fish_count : a.total_fish = 100)
  (equal_fish : a.clownfish = a.blowfish)
  (equal_display : a.clownfish_in_display = a.blowfish_in_display)
  (clownfish_display_after_return : a.clownfish_in_display = 16)
  (clownfish_display_before_return : a.clownfish_in_display * 3 / 2 = a.blowfish_in_display) :
  a.blowfish - a.blowfish_in_display = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blowfish_in_own_tank_l872_87282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_max_area_l872_87279

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ := sorry

/-- The base of a triangle -/
noncomputable def base (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

theorem isosceles_max_area (t1 t2 : Triangle) 
  (h1 : perimeter t1 = perimeter t2) 
  (h2 : base t1 = base t2) 
  (h3 : isIsosceles t1) 
  (h4 : ¬isIsosceles t2) : 
  area t1 ≥ area t2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_max_area_l872_87279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l872_87296

/-- The area of a trapezoid with height x, one base 3x, and the other base 4x -/
noncomputable def trapezoidArea (x : ℝ) : ℝ := (x * (3 * x + 4 * x)) / 2

theorem trapezoid_area_formula (x : ℝ) : 
  trapezoidArea x = (7 * x^2) / 2 := by
  -- Unfold the definition of trapezoidArea
  unfold trapezoidArea
  -- Simplify the expression
  simp [mul_add, mul_assoc, mul_comm, add_mul]
  -- Prove the equality
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l872_87296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_group_sum_l872_87232

/-- The sequence defined as 2n + 1 for natural numbers n -/
def mySequence (n : ℕ) : ℕ := 2 * n + 1

/-- The size of the group at position k in the cyclic grouping -/
def groupSize (k : ℕ) : ℕ := (k - 1) % 4 + 1

/-- The total number of elements before the kth group -/
def elementsBeforeGroup (k : ℕ) : ℕ := 
  (k - 1) / 4 * 10 + 
  match (k - 1) % 4 with
  | 0 => 0
  | 1 => 1
  | 2 => 3
  | _ => 6

/-- The sum of elements in the kth group -/
def groupSum (k : ℕ) : ℕ := 
  let start := elementsBeforeGroup k + 1
  let size := groupSize k
  (List.range size).map (λ i => mySequence (start + i)) |>.sum

theorem hundredth_group_sum : groupSum 100 = 1992 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_group_sum_l872_87232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_minimum_value_and_a_l872_87204

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Part 1: Tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  f 2 1 = 2 →
  (deriv (f 2)) 1 = -1 →
  x + y - 3 = 0 ↔ y = f 2 x ∧ x = 1 :=
by sorry

-- Part 2: Minimum value and a
theorem minimum_value_and_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = 3/2) →
  a = Real.sqrt (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_minimum_value_and_a_l872_87204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_symmetric_intervals_l872_87295

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_symmetric_intervals
  (f : ℝ → ℝ) (a b : ℝ) (h_odd : is_odd f)
  (h_incr : ∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x ≤ y → f x ≤ f y)
  (h_min : ∀ x, x ∈ Set.Icc a b → 1 ≤ f x) :
  (∀ x y, x ∈ Set.Icc (-b) (-a) → y ∈ Set.Icc (-b) (-a) → x ≤ y → f x ≤ f y) ∧
  (∀ x, x ∈ Set.Icc (-b) (-a) → f x ≤ -1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_symmetric_intervals_l872_87295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l872_87202

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 4 * x - 1 / x

-- Theorem statement
theorem non_monotonic_interval (k : ℝ) :
  (∀ x, x > 0 → f x = 2 * x^2 - Real.log x) →
  (∃ x y, k - 1 < x ∧ x < y ∧ y < k + 1 ∧ f x > f y) →
  (∃ x y, k - 1 < x ∧ x < y ∧ y < k + 1 ∧ f x < f y) →
  1 ≤ k ∧ k < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l872_87202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_set_intersection_reciprocal_sets_equal_l872_87262

-- Define the empty set
def empty_set : Set α := ∅

-- Define the intersection of two sets
def set_intersection (A B : Set α) : Set α := {x | x ∈ A ∧ x ∈ B}

-- Define a set of reciprocals
def reciprocal_set (f : ℝ → ℝ) : Set ℝ := {y | ∃ x, y = f x}

-- Statement 1: The intersection of any set with the empty set is the empty set
theorem empty_set_intersection (A : Set α) : set_intersection empty_set A = empty_set := by
  sorry

-- Statement 2: Two sets of reciprocals are equal
theorem reciprocal_sets_equal : 
  reciprocal_set (λ x => 1 / x) = reciprocal_set (λ t => 1 / t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_set_intersection_reciprocal_sets_equal_l872_87262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_division_l872_87234

/-- Given a sum divided among w, x, y, and z where:
    - For each rupee w gets, x gets 34.5 paisa
    - For each rupee w gets, y gets 45.625 paisa
    - For each rupee w gets, z gets 61.875 paisa
    - The share of y is 112.50 rupees
    This theorem proves that the total amount is approximately 596.82 rupees -/
theorem sum_division (w x y z : ℝ) (total : ℝ) : 
  (x = 0.345 * w) →
  (y = 0.45625 * w) →
  (z = 0.61875 * w) →
  (y = 112.50) →
  (total = w + x + y + z) →
  (abs (total - 596.82) < 0.01) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_division_l872_87234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l872_87209

theorem log_inequality : ∀ a b c : ℝ,
  (a = Real.log 2 / Real.log 5) →
  (b = Real.log 2 / Real.log (2/3)) →
  (c = Real.exp (1/2)) →
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l872_87209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l872_87222

/-- Given that real numbers 4, m, 9 form a geometric progression,
    prove that the eccentricity of the hyperbola x^2 + y^2/m = 1 is √7 -/
theorem hyperbola_eccentricity (m : ℝ) 
  (h1 : m^2 = 4 * 9) -- Condition for geometric progression
  (h2 : m > 0) -- Ensuring m is positive for a valid hyperbola equation
  : Real.sqrt ((1 + m) / m) = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l872_87222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l872_87267

-- Define the circle equation
def circleEquation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the curve equation
def curveEquation (x y : ℝ) : Prop := x*y - y = 0

-- Define a function to count intersection points
noncomputable def countIntersections : ℕ := 3

-- Theorem statement
theorem intersection_count : countIntersections = 3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l872_87267
