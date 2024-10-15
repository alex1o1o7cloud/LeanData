import Mathlib

namespace NUMINAMATH_CALUDE_well_volume_l1866_186680

/-- The volume of a circular cylinder with diameter 2 meters and height 14 meters is 14π cubic meters. -/
theorem well_volume :
  let diameter : ℝ := 2
  let height : ℝ := 14
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * height
  volume = 14 * π :=
by sorry

end NUMINAMATH_CALUDE_well_volume_l1866_186680


namespace NUMINAMATH_CALUDE_dress_designs_count_l1866_186654

/-- The number of available fabric colors -/
def num_colors : ℕ := 3

/-- The number of available patterns -/
def num_patterns : ℕ := 4

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_patterns

/-- Theorem stating that the total number of possible dress designs is 12 -/
theorem dress_designs_count : total_designs = 12 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l1866_186654


namespace NUMINAMATH_CALUDE_allocation_schemes_l1866_186638

theorem allocation_schemes (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 9) :
  (Nat.choose (n + k - 1) (k - 1)) = 165 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_l1866_186638


namespace NUMINAMATH_CALUDE_fair_coin_999th_flip_l1866_186662

/-- A fair coin is a coin that has equal probability of landing heads or tails. -/
def FairCoin : Type := Unit

/-- A sequence of coin flips. -/
def CoinFlips (n : ℕ) := Fin n → Bool

/-- The probability of an event occurring in a fair coin flip. -/
def prob (event : Bool → Prop) : ℚ := sorry

theorem fair_coin_999th_flip (c : FairCoin) (flips : CoinFlips 1000) :
  prob (λ result => result = true) = 1/2 := by sorry

end NUMINAMATH_CALUDE_fair_coin_999th_flip_l1866_186662


namespace NUMINAMATH_CALUDE_telescope_visual_range_increase_l1866_186697

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 90)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_increase_l1866_186697


namespace NUMINAMATH_CALUDE_smallest_difference_in_triangle_l1866_186642

theorem smallest_difference_in_triangle (PQ QR PR : ℕ) : 
  PQ + QR + PR = 2021 →
  PQ < QR →
  QR ≤ PR →
  PQ + QR > PR →
  PQ + PR > QR →
  QR + PR > PQ →
  ∃ (PQ' QR' PR' : ℕ), 
    PQ' + QR' + PR' = 2021 ∧
    PQ' < QR' ∧
    QR' ≤ PR' ∧
    PQ' + QR' > PR' ∧
    PQ' + PR' > QR' ∧
    QR' + PR' > PQ' ∧
    QR' - PQ' = 1 ∧
    ∀ (PQ'' QR'' PR'' : ℕ),
      PQ'' + QR'' + PR'' = 2021 →
      PQ'' < QR'' →
      QR'' ≤ PR'' →
      PQ'' + QR'' > PR'' →
      PQ'' + PR'' > QR'' →
      QR'' + PR'' > PQ'' →
      QR'' - PQ'' ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_in_triangle_l1866_186642


namespace NUMINAMATH_CALUDE_middle_number_proof_l1866_186641

theorem middle_number_proof (a b c d e : ℕ) : 
  ({a, b, c, d, e} : Finset ℕ) = {7, 8, 9, 10, 11} →
  a + b + c = 26 →
  c + d + e = 30 →
  c = 11 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1866_186641


namespace NUMINAMATH_CALUDE_range_of_a_l1866_186617

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1866_186617


namespace NUMINAMATH_CALUDE_angle_C_measure_side_c_length_l1866_186609

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition from the problem -/
def condition (t : Triangle) : Prop :=
  (t.a * Real.cos t.B + t.b * Real.cos t.A) / t.c = 2 * Real.cos t.C

theorem angle_C_measure (t : Triangle) (h : condition t) : t.C = π / 3 := by
  sorry

theorem side_c_length (t : Triangle) 
  (h1 : (1 / 2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3)
  (h2 : t.a + t.b = 6)
  (h3 : t.C = π / 3) : t.c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_side_c_length_l1866_186609


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1866_186685

/-- Given that i² = -1, prove that (3 - 2i) / (4 + 5i) = 2/41 - 23/41 * i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (3 - 2*i) / (4 + 5*i) = 2/41 - 23/41 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1866_186685


namespace NUMINAMATH_CALUDE_clothing_calculation_l1866_186676

/-- Calculates the remaining pieces of clothing after donations and disposal --/
def remaining_clothing (initial : ℕ) (first_donation : ℕ) (disposal : ℕ) : ℕ :=
  initial - (first_donation + 3 * first_donation) - disposal

/-- Theorem stating the remaining pieces of clothing --/
theorem clothing_calculation :
  remaining_clothing 100 5 15 = 65 := by
  sorry

end NUMINAMATH_CALUDE_clothing_calculation_l1866_186676


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l1866_186601

/-- Calculates the final price of a jacket after two successive price reductions -/
theorem jacket_price_reduction (initial_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  initial_price = 20 ∧ 
  first_reduction = 0.2 ∧ 
  second_reduction = 0.25 →
  initial_price * (1 - first_reduction) * (1 - second_reduction) = 12 :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l1866_186601


namespace NUMINAMATH_CALUDE_equation_solutions_l1866_186690

theorem equation_solutions :
  (∃ y₁ y₂ : ℝ, y₁ = 3 + 2 * Real.sqrt 2 ∧ y₂ = 3 - 2 * Real.sqrt 2 ∧
    ∀ y : ℝ, y^2 - 6*y + 1 = 0 ↔ (y = y₁ ∨ y = y₂)) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = 12 ∧
    ∀ x : ℝ, 2*(x-4)^2 = x^2 - 16 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1866_186690


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_450_l1866_186605

theorem distinct_prime_factors_of_450 : Nat.card (Nat.factors 450).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_450_l1866_186605


namespace NUMINAMATH_CALUDE_simplify_fraction_sum_l1866_186635

theorem simplify_fraction_sum (a b c d : ℕ) (h1 : a * d = b * c) (h2 : Nat.gcd a b = 1) :
  a + b = 11 → 75 * d = 200 * c :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_sum_l1866_186635


namespace NUMINAMATH_CALUDE_regular_polygon_with_45_degree_exterior_angle_has_8_sides_l1866_186684

/-- A regular polygon with an exterior angle of 45° has 8 sides. -/
theorem regular_polygon_with_45_degree_exterior_angle_has_8_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 0 →
    exterior_angle = 45 →
    (360 : ℝ) / exterior_angle = n →
    n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_45_degree_exterior_angle_has_8_sides_l1866_186684


namespace NUMINAMATH_CALUDE_wednesday_spending_multiple_l1866_186614

def monday_spending : ℝ := 60
def tuesday_spending : ℝ := 4 * monday_spending
def total_spending : ℝ := 600

theorem wednesday_spending_multiple : 
  ∃ x : ℝ, 
    monday_spending + tuesday_spending + x * monday_spending = total_spending ∧ 
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_spending_multiple_l1866_186614


namespace NUMINAMATH_CALUDE_line_point_value_l1866_186637

/-- Given a line with slope 2 passing through (3, 5) and (a, 7), prove a = 4 -/
theorem line_point_value (m : ℝ) (a : ℝ) : 
  m = 2 → -- The line has a slope of 2
  (5 : ℝ) - 5 = m * ((3 : ℝ) - 3) → -- The line passes through (3, 5)
  (7 : ℝ) - 5 = m * (a - 3) → -- The line passes through (a, 7)
  a = 4 := by sorry

end NUMINAMATH_CALUDE_line_point_value_l1866_186637


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1866_186668

/-- A hyperbola with equation x^2/4 - y^2/16 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/16 = 1

/-- Asymptotic lines with equations y = ±2x -/
def asymptotic_lines (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

/-- Theorem stating that the given hyperbola equation implies the asymptotic lines,
    but the asymptotic lines do not necessarily imply the specific hyperbola equation -/
theorem hyperbola_asymptotes :
  (∀ x y : ℝ, hyperbola x y → asymptotic_lines x y) ∧
  ¬(∀ x y : ℝ, asymptotic_lines x y → hyperbola x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1866_186668


namespace NUMINAMATH_CALUDE_chameleons_multiple_colors_l1866_186616

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The initial state of chameleons on the island -/
def initial_state : ChameleonState :=
  { red := 155, blue := 49, green := 96 }

/-- Defines the color change rule for chameleons -/
def color_change_rule (state : ChameleonState) : ChameleonState → Prop :=
  λ new_state =>
    (new_state.red + new_state.blue + new_state.green = state.red + state.blue + state.green) ∧
    (new_state.red - new_state.blue) % 3 = (state.red - state.blue) % 3 ∧
    (new_state.blue - new_state.green) % 3 = (state.blue - state.green) % 3 ∧
    (new_state.red - new_state.green) % 3 = (state.red - state.green) % 3

/-- Theorem stating that it's impossible for all chameleons to be the same color -/
theorem chameleons_multiple_colors (final_state : ChameleonState) :
  color_change_rule initial_state final_state →
  ¬(final_state.red = 0 ∧ final_state.blue = 0) ∧
  ¬(final_state.red = 0 ∧ final_state.green = 0) ∧
  ¬(final_state.blue = 0 ∧ final_state.green = 0) :=
by sorry

end NUMINAMATH_CALUDE_chameleons_multiple_colors_l1866_186616


namespace NUMINAMATH_CALUDE_solve_for_a_l1866_186686

theorem solve_for_a (a b : ℝ) (h1 : b/a = 4) (h2 : b = 24 - 4*a) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1866_186686


namespace NUMINAMATH_CALUDE_parabola_equation_l1866_186643

/-- Given a parabola y^2 = 2px with a point P(2, y_0) on it, and the distance from P to the directrix is 4,
    prove that p = 4 and the standard equation of the parabola is y^2 = 8x. -/
theorem parabola_equation (p : ℝ) (y_0 : ℝ) (h1 : p > 0) (h2 : y_0^2 = 2*p*2) (h3 : p/2 + 2 = 4) :
  p = 4 ∧ ∀ x y, y^2 = 8*x ↔ y^2 = 2*p*x := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1866_186643


namespace NUMINAMATH_CALUDE_marble_distribution_l1866_186615

theorem marble_distribution (x : ℚ) 
  (h1 : (5 * x + 2) + 2 * x + 4 * x = 88) : x = 86 / 11 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l1866_186615


namespace NUMINAMATH_CALUDE_points_form_circle_l1866_186623

theorem points_form_circle :
  ∀ (x y : ℝ), (∃ t : ℝ, x = Real.cos t ∧ y = Real.sin t) →
  x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_points_form_circle_l1866_186623


namespace NUMINAMATH_CALUDE_origin_midpoint_coordinates_l1866_186618

/-- Given two points A and B in a 2D Cartesian coordinate system, 
    this function returns true if the origin (0, 0) is the midpoint of AB. -/
def isOriginMidpoint (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 0

/-- Theorem stating that if the origin is the midpoint of AB and 
    A has coordinates (-1, 2), then B has coordinates (1, -2). -/
theorem origin_midpoint_coordinates :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (1, -2)
  isOriginMidpoint A B → B = (1, -2) := by
  sorry


end NUMINAMATH_CALUDE_origin_midpoint_coordinates_l1866_186618


namespace NUMINAMATH_CALUDE_pocket_knife_value_l1866_186653

def is_fair_division (n : ℕ) (knife_value : ℕ) : Prop :=
  let total_revenue := n * n
  let elder_share := (total_revenue / 20) * 10
  let younger_share := ((total_revenue / 20) * 10) + (total_revenue % 20) + knife_value
  elder_share = younger_share

theorem pocket_knife_value :
  ∃ (n : ℕ), 
    n > 0 ∧ 
    (n * n) % 20 = 6 ∧ 
    is_fair_division n 2 :=
by sorry

end NUMINAMATH_CALUDE_pocket_knife_value_l1866_186653


namespace NUMINAMATH_CALUDE_probability_red_or_white_l1866_186646

def total_marbles : ℕ := 50
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white :
  (red_marbles + white_marbles : ℚ) / total_marbles = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_l1866_186646


namespace NUMINAMATH_CALUDE_ratio_xy_system_l1866_186688

theorem ratio_xy_system (x y t : ℝ) 
  (eq1 : 2 * x + 5 * y = 6 * t) 
  (eq2 : 3 * x - y = t) : 
  x / y = 11 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_xy_system_l1866_186688


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_perpendicular_line_l1866_186600

/-- Represents a hyperbola in the Cartesian plane -/
structure Hyperbola where
  a : ℝ
  equation : ℝ → ℝ → Prop
  asymptote : ℝ → ℝ → Prop

/-- Represents a line in the Cartesian plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def Perpendicular (l1 l2 : Line) : Prop :=
  ∃ m1 m2 : ℝ, (∀ x y, l1.equation x y ↔ y = m1 * x + (0 : ℝ)) ∧ 
              (∀ x y, l2.equation x y ↔ y = m2 * x + (0 : ℝ)) ∧ 
              m1 * m2 = -1

/-- The main theorem -/
theorem hyperbola_asymptote_perpendicular_line (h : Hyperbola) (l : Line) : 
  h.a > 0 ∧ 
  (∀ x y, h.equation x y ↔ x^2 / h.a^2 - y^2 = 1) ∧
  (∀ x y, l.equation x y ↔ 2*x - y + 1 = 0) ∧
  (∃ la : Line, h.asymptote = la.equation ∧ Perpendicular la l) →
  h.a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_perpendicular_line_l1866_186600


namespace NUMINAMATH_CALUDE_mackenzie_bought_three_new_cds_l1866_186669

/-- Represents the price of a new CD -/
def new_cd_price : ℚ := 127.92 - 2 * 9.99

/-- Represents the number of new CDs Mackenzie bought -/
def mackenzie_new_cds : ℚ := (133.89 - 8 * 9.99) / (127.92 - 2 * 9.99) * 6

theorem mackenzie_bought_three_new_cds : 
  ⌊mackenzie_new_cds⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_mackenzie_bought_three_new_cds_l1866_186669


namespace NUMINAMATH_CALUDE_third_score_calculation_l1866_186648

/-- Given four scores where three are known and their average with an unknown fourth score is 76.6,
    prove that the unknown score must be 79.4. -/
theorem third_score_calculation (score1 score2 score4 : ℝ) (average : ℝ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 95 →
  average = 76.6 →
  ∃ score3 : ℝ, score3 = 79.4 ∧ (score1 + score2 + score3 + score4) / 4 = average :=
by sorry

end NUMINAMATH_CALUDE_third_score_calculation_l1866_186648


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l1866_186698

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l1866_186698


namespace NUMINAMATH_CALUDE_distance_difference_l1866_186658

-- Define the dimensions
def street_width : ℕ := 25
def block_length : ℕ := 450
def block_width : ℕ := 350
def alley_width : ℕ := 25

-- Define Sarah's path
def sarah_long_side : ℕ := block_length + alley_width
def sarah_short_side : ℕ := block_width

-- Define Sam's path
def sam_long_side : ℕ := block_length + 2 * street_width
def sam_short_side : ℕ := block_width + 2 * street_width

-- Calculate total distances
def sarah_total : ℕ := 2 * sarah_long_side + 2 * sarah_short_side
def sam_total : ℕ := 2 * sam_long_side + 2 * sam_short_side

-- Theorem to prove
theorem distance_difference :
  sam_total - sarah_total = 150 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1866_186658


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l1866_186647

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n + 1 < 0 ∧ n * (n + 1) = 2550 → n + (n + 1) = -101 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l1866_186647


namespace NUMINAMATH_CALUDE_fill_cistern_time_cistern_filling_problem_l1866_186659

/-- The time taken for two pipes to fill a cistern together -/
theorem fill_cistern_time (time_A time_B : ℝ) (h1 : time_A > 0) (h2 : time_B > 0) :
  let combined_rate := 1 / time_A + 1 / time_B
  combined_rate⁻¹ = (time_A * time_B) / (time_A + time_B) := by sorry

/-- Proof of the cistern filling problem -/
theorem cistern_filling_problem :
  let time_A : ℝ := 36  -- Time for Pipe A to fill the entire cistern
  let time_B : ℝ := 24  -- Time for Pipe B to fill the entire cistern
  let combined_time := (time_A * time_B) / (time_A + time_B)
  combined_time = 14.4 := by sorry

end NUMINAMATH_CALUDE_fill_cistern_time_cistern_filling_problem_l1866_186659


namespace NUMINAMATH_CALUDE_acid_mixture_proof_l1866_186657

-- Define the volumes and concentrations
def volume_60_percent : ℝ := 4
def concentration_60_percent : ℝ := 0.60
def volume_75_percent : ℝ := 16
def concentration_75_percent : ℝ := 0.75
def total_volume : ℝ := 20
def final_concentration : ℝ := 0.72

-- Theorem statement
theorem acid_mixture_proof :
  (volume_60_percent * concentration_60_percent + 
   volume_75_percent * concentration_75_percent) / total_volume = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_acid_mixture_proof_l1866_186657


namespace NUMINAMATH_CALUDE_residue_problem_l1866_186671

theorem residue_problem : (195 * 13 - 25 * 8 + 5) % 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_residue_problem_l1866_186671


namespace NUMINAMATH_CALUDE_mean_median_difference_is_03_l1866_186675

/-- Represents a frequency histogram bin -/
structure HistogramBin where
  lowerBound : ℝ
  upperBound : ℝ
  frequency : ℕ

/-- Calculates the median of a dataset represented by a frequency histogram -/
def calculateMedian (histogram : List HistogramBin) (totalStudents : ℕ) : ℝ :=
  sorry

/-- Calculates the mean of a dataset represented by a frequency histogram -/
def calculateMean (histogram : List HistogramBin) (totalStudents : ℕ) : ℝ :=
  sorry

theorem mean_median_difference_is_03 (histogram : List HistogramBin) : 
  let totalStudents := 20
  let h := [
    ⟨0, 1, 4⟩,
    ⟨2, 3, 2⟩,
    ⟨4, 5, 6⟩,
    ⟨6, 7, 3⟩,
    ⟨8, 9, 5⟩
  ]
  calculateMean h totalStudents - calculateMedian h totalStudents = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_is_03_l1866_186675


namespace NUMINAMATH_CALUDE_square_ABCD_l1866_186619

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop :=
  let AB := (q.B.x - q.A.x, q.B.y - q.A.y)
  let BC := (q.C.x - q.B.x, q.C.y - q.B.y)
  let CD := (q.D.x - q.C.x, q.D.y - q.C.y)
  let DA := (q.A.x - q.D.x, q.A.y - q.D.y)
  -- All sides have equal length
  AB.1^2 + AB.2^2 = BC.1^2 + BC.2^2 ∧
  BC.1^2 + BC.2^2 = CD.1^2 + CD.2^2 ∧
  CD.1^2 + CD.2^2 = DA.1^2 + DA.2^2 ∧
  -- Adjacent sides are perpendicular
  AB.1 * BC.1 + AB.2 * BC.2 = 0 ∧
  BC.1 * CD.1 + BC.2 * CD.2 = 0 ∧
  CD.1 * DA.1 + CD.2 * DA.2 = 0 ∧
  DA.1 * AB.1 + DA.2 * AB.2 = 0

theorem square_ABCD :
  let q := Quadrilateral.mk
    (Point.mk (-1) 3)
    (Point.mk 1 (-2))
    (Point.mk 6 0)
    (Point.mk 4 5)
  is_square q := by
  sorry

end NUMINAMATH_CALUDE_square_ABCD_l1866_186619


namespace NUMINAMATH_CALUDE_johns_remaining_money_l1866_186625

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Proves that John's remaining money after buying the ticket is 1725 dollars -/
theorem johns_remaining_money :
  let savings := base8_to_base10 5555
  let ticket_cost := 1200
  savings - ticket_cost = 1725 := by sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l1866_186625


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1866_186628

theorem inequality_system_solution (x : ℝ) :
  (x + 5 < 4) ∧ ((3 * x + 1) / 2 ≥ 2 * x - 1) → x < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1866_186628


namespace NUMINAMATH_CALUDE_probability_system_failure_correct_l1866_186687

/-- The probability of at least one component failing in a system of m identical components -/
def probability_system_failure (m : ℕ) (P : ℝ) : ℝ :=
  1 - (1 - P)^m

/-- Theorem: The probability of at least one component failing in a system of m identical components
    with individual failure probability P is 1-(1-P)^m -/
theorem probability_system_failure_correct (m : ℕ) (P : ℝ) 
    (h1 : 0 ≤ P) (h2 : P ≤ 1) :
  probability_system_failure m P = 1 - (1 - P)^m :=
by
  sorry

end NUMINAMATH_CALUDE_probability_system_failure_correct_l1866_186687


namespace NUMINAMATH_CALUDE_fraction_equality_l1866_186674

theorem fraction_equality (x y : ℝ) (h : x ≠ -y) : (-x + y) / (-x - y) = (x - y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1866_186674


namespace NUMINAMATH_CALUDE_correct_calculation_l1866_186631

theorem correct_calculation (x : ℤ) (h : x + 35 = 77) : x - 35 = 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1866_186631


namespace NUMINAMATH_CALUDE_remaining_marbles_l1866_186602

/-- Given Chris has 12 marbles and Ryan has 28 marbles, if they combine their marbles
    and each takes 1/4 of the total, the number of marbles remaining in the pile is 20. -/
theorem remaining_marbles (chris_marbles : ℕ) (ryan_marbles : ℕ) 
    (h1 : chris_marbles = 12) 
    (h2 : ryan_marbles = 28) : 
  let total_marbles := chris_marbles + ryan_marbles
  let taken_marbles := 2 * (total_marbles / 4)
  total_marbles - taken_marbles = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_marbles_l1866_186602


namespace NUMINAMATH_CALUDE_power_of_sixteen_five_fourths_l1866_186613

theorem power_of_sixteen_five_fourths : (16 : ℝ) ^ (5/4 : ℝ) = 32 := by sorry

end NUMINAMATH_CALUDE_power_of_sixteen_five_fourths_l1866_186613


namespace NUMINAMATH_CALUDE_odd_function_extension_l1866_186655

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_pos : ∀ x > 0, f x = x^2 + 1) :
  ∀ x < 0, f x = -x^2 - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1866_186655


namespace NUMINAMATH_CALUDE_log_identity_l1866_186699

theorem log_identity : Real.log 16 / Real.log 4 - (Real.log 3 / Real.log 2) * (Real.log 2 / Real.log 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l1866_186699


namespace NUMINAMATH_CALUDE_bailey_chew_toys_l1866_186696

theorem bailey_chew_toys (dog_treats rawhide_bones credit_cards items_per_charge : ℕ) 
  (h1 : dog_treats = 8)
  (h2 : rawhide_bones = 10)
  (h3 : credit_cards = 4)
  (h4 : items_per_charge = 5) :
  credit_cards * items_per_charge - (dog_treats + rawhide_bones) = 2 := by
  sorry

end NUMINAMATH_CALUDE_bailey_chew_toys_l1866_186696


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l1866_186612

/-- The measure of an interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_interior_angles : ℝ := (n - 2) * 180  -- sum of interior angles formula
  let interior_angle : ℝ := sum_interior_angles / n  -- each interior angle measure
  135

/-- Proof of the theorem -/
lemma prove_regular_octagon_interior_angle : 
  regular_octagon_interior_angle = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l1866_186612


namespace NUMINAMATH_CALUDE_fraction_power_cube_l1866_186673

theorem fraction_power_cube : (2 / 5 : ℚ) ^ 3 = 8 / 125 := by sorry

end NUMINAMATH_CALUDE_fraction_power_cube_l1866_186673


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1866_186693

theorem simplify_trig_expression :
  let θ : Real := 160 * π / 180  -- Convert 160° to radians
  (θ > π / 2) ∧ (θ < π) →  -- 160° is in the second quadrant
  1 / Real.sqrt (1 + Real.tan θ ^ 2) = -Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1866_186693


namespace NUMINAMATH_CALUDE_ratio_calculation_l1866_186645

theorem ratio_calculation (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) 
  (hw : w ≠ 0) : 
  x * z / (y * w) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l1866_186645


namespace NUMINAMATH_CALUDE_chocolate_division_l1866_186608

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) :
  total_chocolate = 48/5 →
  num_piles = 4 →
  total_chocolate / num_piles = 12/5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_division_l1866_186608


namespace NUMINAMATH_CALUDE_shaded_area_of_overlapping_sectors_l1866_186607

/-- The area of the shaded region formed by two overlapping sectors of a circle -/
theorem shaded_area_of_overlapping_sectors (r : ℝ) (θ : ℝ) (h1 : r = 15) (h2 : θ = 45) :
  let sector_area := θ / 360 * π * r^2
  let triangle_area := Real.sqrt 3 / 4 * r^2
  2 * (sector_area - triangle_area) = 56.25 * π - 112.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_overlapping_sectors_l1866_186607


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1866_186651

theorem complex_equation_sum (x y : ℝ) :
  (↑x + (↑y - 2) * I : ℂ) = 2 / (1 + I) →
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1866_186651


namespace NUMINAMATH_CALUDE_expected_draws_for_given_balls_l1866_186694

/-- Represents the number of balls of each color in the bag -/
structure BallCount where
  red : ℕ
  yellow : ℕ

/-- Calculates the expected number of balls drawn until two different colors are drawn -/
def expectedDraws (balls : BallCount) : ℚ :=
  sorry

/-- The theorem stating that the expected number of draws is 5/2 for the given ball configuration -/
theorem expected_draws_for_given_balls :
  expectedDraws { red := 3, yellow := 2 } = 5/2 := by sorry

end NUMINAMATH_CALUDE_expected_draws_for_given_balls_l1866_186694


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l1866_186611

/-- Given a circular seating arrangement where:
    - There is equal spacing between all positions
    - The 6th person is directly opposite the 16th person
    - One position is reserved for a teacher
    Prove that the total number of students (excluding the teacher) is 20. -/
theorem circular_seating_arrangement (n : ℕ) 
  (h1 : ∃ (teacher_pos : ℕ), teacher_pos ≤ n + 1) 
  (h2 : (6 + n/2) % (n + 1) = 16 % (n + 1)) : n = 20 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l1866_186611


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1866_186656

-- Define the line
def line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem line_not_in_second_quadrant :
  ¬ ∃ (x y : ℝ), line x y ∧ second_quadrant x y :=
sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1866_186656


namespace NUMINAMATH_CALUDE_power_inequality_l1866_186667

theorem power_inequality (a b : ℕ) (ha : a > 1) (hb : b > 2) :
  a^b + 1 ≥ b * (a + 1) ∧ (a^b + 1 = b * (a + 1) ↔ a = 2 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1866_186667


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_two_fixed_points_condition_l1866_186632

/-- Definition of a fixed point for a function f -/
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

/-- The given function f(x) = ax² + (b + 1)x + b - 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

/-- Theorem: The function f has fixed points at 3 and -1 when a = 1 and b = -2 -/
theorem fixed_points_for_specific_values :
  is_fixed_point (f 1 (-2)) 3 ∧ is_fixed_point (f 1 (-2)) (-1) :=
sorry

/-- Theorem: The function f always has two fixed points for any real b if and only if 0 < a < 1 -/
theorem two_fixed_points_condition (a : ℝ) :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) ↔
  (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_two_fixed_points_condition_l1866_186632


namespace NUMINAMATH_CALUDE_min_max_x_given_xy_eq_nx_plus_ny_l1866_186606

theorem min_max_x_given_xy_eq_nx_plus_ny (n x y : ℕ+) (h : x * y = n * x + n * y) :
  x ≥ n + 1 ∧ x ≤ n * (n + 1) :=
sorry

end NUMINAMATH_CALUDE_min_max_x_given_xy_eq_nx_plus_ny_l1866_186606


namespace NUMINAMATH_CALUDE_log_expression_equality_l1866_186622

theorem log_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - (5 : ℝ) ^ (Real.log 3 / Real.log 5) = -3 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l1866_186622


namespace NUMINAMATH_CALUDE_village_population_l1866_186620

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.15) = 3553 → P = 4400 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1866_186620


namespace NUMINAMATH_CALUDE_petyas_class_girls_count_l1866_186660

theorem petyas_class_girls_count :
  ∀ (x y : ℕ),
  x + y ≤ 40 →
  (2 : ℚ) / 3 * x + (1 : ℚ) / 7 * y = (1 : ℚ) / 3 * (x + y) →
  x = 12 :=
λ x y h1 h2 => by
  sorry

end NUMINAMATH_CALUDE_petyas_class_girls_count_l1866_186660


namespace NUMINAMATH_CALUDE_group_frequency_l1866_186670

theorem group_frequency (sample_capacity : ℕ) (group_frequency : ℚ) : 
  sample_capacity = 20 → group_frequency = 0.25 → 
  (sample_capacity : ℚ) * group_frequency = 5 := by
  sorry

end NUMINAMATH_CALUDE_group_frequency_l1866_186670


namespace NUMINAMATH_CALUDE_butter_problem_l1866_186603

theorem butter_problem (B : ℝ) : 
  (B / 2 + B / 5 + (B - B / 2 - B / 5) / 3 + 2 = B) → B = 10 :=
by sorry

end NUMINAMATH_CALUDE_butter_problem_l1866_186603


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1866_186633

theorem water_tank_capacity : ∀ (C : ℝ),
  (∃ (x : ℝ), x / C = 1 / 3 ∧ (x + 6) / C = 1 / 2) →
  C = 36 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1866_186633


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_difference_l1866_186663

theorem opposite_of_sqrt_difference : -(Real.sqrt 2 - Real.sqrt 3) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_difference_l1866_186663


namespace NUMINAMATH_CALUDE_decagon_triangles_l1866_186639

theorem decagon_triangles : ∀ n : ℕ, n = 10 → (n.choose 3) = 120 := by sorry

end NUMINAMATH_CALUDE_decagon_triangles_l1866_186639


namespace NUMINAMATH_CALUDE_max_product_of_digits_l1866_186627

theorem max_product_of_digits (A B : ℕ) : 
  (A ≤ 9) → 
  (B ≤ 9) → 
  (∃ (n : ℕ), A * 100000 + 2021 * 100 + B = 9 * n) →
  A * B ≤ 42 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_digits_l1866_186627


namespace NUMINAMATH_CALUDE_cylinder_height_equals_sphere_surface_area_l1866_186661

/-- Given a sphere of radius 6 cm and a right circular cylinder with equal height and diameter,
    if their surface areas are equal, then the height of the cylinder is 6√2 cm. -/
theorem cylinder_height_equals_sphere_surface_area (r h : ℝ) : 
  r = 6 →  -- radius of sphere is 6 cm
  h = 2 * r →  -- height of cylinder equals its diameter
  4 * Real.pi * r^2 = 2 * Real.pi * r * h →  -- surface areas are equal
  h = 6 * Real.sqrt 2 := by
  sorry

#check cylinder_height_equals_sphere_surface_area

end NUMINAMATH_CALUDE_cylinder_height_equals_sphere_surface_area_l1866_186661


namespace NUMINAMATH_CALUDE_master_bedroom_size_l1866_186650

theorem master_bedroom_size 
  (master_bath : ℝ) 
  (new_room : ℝ) 
  (h1 : master_bath = 150) 
  (h2 : new_room = 918) 
  (h3 : new_room = 2 * (master_bedroom + master_bath)) : 
  master_bedroom = 309 :=
by
  sorry

end NUMINAMATH_CALUDE_master_bedroom_size_l1866_186650


namespace NUMINAMATH_CALUDE_f_properties_l1866_186679

noncomputable def f (x : ℝ) := Real.log ((1 + x) / (1 - x))

theorem f_properties :
  ∃ (k : ℝ),
    (∀ x ∈ Set.Ioo 0 1, f x > k * (x + x^3 / 3)) ∧
    (∀ k' > k, ∃ x ∈ Set.Ioo 0 1, f x ≤ k' * (x + x^3 / 3)) ∧
    k = 2 ∧
    (∀ x ∈ Set.Ioo 0 1, f x > 2 * (x + x^3 / 3)) ∧
    (∀ h ∈ Set.Ioo 0 1, (f h - f 0) / h = 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1866_186679


namespace NUMINAMATH_CALUDE_translation_proof_l1866_186692

-- Define the original linear function
def original_function (x : ℝ) : ℝ := 3 * x - 1

-- Define the translation
def translation : ℝ := 3

-- Define the resulting function after translation
def translated_function (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem translation_proof :
  ∀ x : ℝ, translated_function x = original_function x + translation :=
by
  sorry

end NUMINAMATH_CALUDE_translation_proof_l1866_186692


namespace NUMINAMATH_CALUDE_distinct_dice_designs_count_l1866_186649

/-- Represents a dice design -/
structure DiceDesign where
  -- We don't need to explicitly define the structure,
  -- as we're only concerned with the count of distinct designs

/-- The number of ways to choose 2 numbers from 4 -/
def choose_two_from_four : Nat := 6

/-- The number of ways to arrange the chosen numbers on opposite faces -/
def arrangement_ways : Nat := 2

/-- The number of ways to color three pairs of opposite faces -/
def coloring_ways : Nat := 8

/-- The total number of distinct dice designs -/
def distinct_dice_designs : Nat := 
  (choose_two_from_four * arrangement_ways / 2) * coloring_ways

theorem distinct_dice_designs_count :
  distinct_dice_designs = 48 := by
  sorry

end NUMINAMATH_CALUDE_distinct_dice_designs_count_l1866_186649


namespace NUMINAMATH_CALUDE_wendys_laundry_l1866_186604

theorem wendys_laundry (machine_capacity : ℕ) (num_sweaters : ℕ) (num_loads : ℕ) :
  machine_capacity = 8 →
  num_sweaters = 33 →
  num_loads = 9 →
  num_loads * machine_capacity - num_sweaters = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_wendys_laundry_l1866_186604


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1866_186689

/-- Given two vectors a and b in ℝ², where a = (1,2) and b = (2,k),
    if 2a + b is perpendicular to a, then k = -6. -/
theorem perpendicular_vectors_k_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, k)
  (2 • a + b) • a = 0 → k = -6 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1866_186689


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1866_186691

def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem arithmetic_geometric_sequence_ratio :
  ∀ (a₁ a₂ b₁ b₂ b₃ : ℝ),
    is_arithmetic_sequence (-1) a₁ a₂ 8 →
    is_geometric_sequence (-1) b₁ b₂ b₃ (-4) →
    (a₁ * a₂) / b₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1866_186691


namespace NUMINAMATH_CALUDE_probability_of_same_color_l1866_186624

/-- Represents a 12-sided die with colored sides -/
structure ColoredDie :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (total_sides : ℕ)
  (side_sum : red + green + blue + yellow = total_sides)

/-- Calculates the probability of two identical dice showing the same color -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.red^2 + d.green^2 + d.blue^2 + d.yellow^2) / d.total_sides^2

/-- The specific 12-sided die described in the problem -/
def problem_die : ColoredDie :=
  { red := 3
    green := 4
    blue := 2
    yellow := 3
    total_sides := 12
    side_sum := by rfl }

theorem probability_of_same_color :
  same_color_probability problem_die = 19 / 72 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_same_color_l1866_186624


namespace NUMINAMATH_CALUDE_strategies_conversion_l1866_186664

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8^i)) 0

/-- The number of strategies in base 8 -/
def strategies_base8 : List Nat := [2, 3, 4]

theorem strategies_conversion :
  base8_to_base10 strategies_base8 = 282 := by
  sorry

end NUMINAMATH_CALUDE_strategies_conversion_l1866_186664


namespace NUMINAMATH_CALUDE_extreme_value_condition_l1866_186695

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - x - a * x

theorem extreme_value_condition (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≥ f a 1) ↔
  a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l1866_186695


namespace NUMINAMATH_CALUDE_inequalities_from_sum_positive_l1866_186665

theorem inequalities_from_sum_positive (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_from_sum_positive_l1866_186665


namespace NUMINAMATH_CALUDE_election_majority_l1866_186640

/-- Calculates the majority in an election --/
theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 7520 → 
  winning_percentage = 60 / 100 → 
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 1504 := by
  sorry


end NUMINAMATH_CALUDE_election_majority_l1866_186640


namespace NUMINAMATH_CALUDE_ericas_amount_l1866_186630

/-- The problem of calculating Erica's amount given the total and Sam's amount -/
theorem ericas_amount (total sam : ℚ) (h1 : total = 450.32) (h2 : sam = 325.67) :
  total - sam = 124.65 := by
  sorry

end NUMINAMATH_CALUDE_ericas_amount_l1866_186630


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l1866_186634

theorem simplify_sqrt_difference : 
  (Real.sqrt 528 / Real.sqrt 64) - (Real.sqrt 242 / Real.sqrt 121) = 1.461 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l1866_186634


namespace NUMINAMATH_CALUDE_hay_consumption_time_l1866_186678

/-- The number of weeks it takes for a group of animals to eat a given amount of hay -/
def time_to_eat_hay (goat_rate sheep_rate cow_rate : ℚ) (num_goats num_sheep num_cows : ℕ) (total_hay : ℚ) : ℚ :=
  total_hay / (goat_rate * num_goats + sheep_rate * num_sheep + cow_rate * num_cows)

/-- Theorem: Given the rates of hay consumption and number of animals, it takes 16 weeks to eat 30 cartloads of hay -/
theorem hay_consumption_time :
  let goat_rate : ℚ := 1 / 6
  let sheep_rate : ℚ := 1 / 8
  let cow_rate : ℚ := 1 / 3
  let num_goats : ℕ := 5
  let num_sheep : ℕ := 3
  let num_cows : ℕ := 2
  let total_hay : ℚ := 30
  time_to_eat_hay goat_rate sheep_rate cow_rate num_goats num_sheep num_cows total_hay = 16 := by
  sorry


end NUMINAMATH_CALUDE_hay_consumption_time_l1866_186678


namespace NUMINAMATH_CALUDE_catch_fraction_l1866_186682

theorem catch_fraction (joe_catches derek_catches tammy_catches : ℕ) : 
  joe_catches = 23 →
  derek_catches = 2 * joe_catches - 4 →
  tammy_catches = 30 →
  (tammy_catches : ℚ) / derek_catches = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_catch_fraction_l1866_186682


namespace NUMINAMATH_CALUDE_impossible_arrangement_l1866_186652

theorem impossible_arrangement : ¬ ∃ (seq : Fin 3972 → Fin 1986), 
  (∀ k : Fin 1986, (∃! i j : Fin 3972, seq i = k ∧ seq j = k ∧ i ≠ j)) ∧
  (∀ k : Fin 1986, ∀ i j : Fin 3972, 
    seq i = k → seq j = k → i ≠ j → 
    (j.val > i.val → j.val - i.val = k.val + 1)) :=
by sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l1866_186652


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1866_186621

/-- The problem statement -/
theorem tangent_line_problem (k : ℝ) (P : ℝ × ℝ) (A : ℝ × ℝ) :
  k > 0 →
  P.1 * k + P.2 + 4 = 0 →
  A.1^2 + A.2^2 - 2*A.2 = 0 →
  (∀ Q : ℝ × ℝ, Q.1^2 + Q.2^2 - 2*Q.2 = 0 → 
    (A.1 - P.1)^2 + (A.2 - P.2)^2 ≤ (Q.1 - P.1)^2 + (Q.2 - P.2)^2) →
  (A.1 - P.1)^2 + (A.2 - P.2)^2 = 4 →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l1866_186621


namespace NUMINAMATH_CALUDE_racecar_repair_cost_l1866_186636

/-- Proves that the original cost of fixing a racecar was $20,000 given specific conditions --/
theorem racecar_repair_cost 
  (discount_rate : Real) 
  (prize : Real) 
  (prize_keep_rate : Real) 
  (net_profit : Real) :
  discount_rate = 0.2 →
  prize = 70000 →
  prize_keep_rate = 0.9 →
  net_profit = 47000 →
  ∃ (original_cost : Real),
    original_cost = 20000 ∧
    prize * prize_keep_rate - original_cost * (1 - discount_rate) = net_profit :=
by
  sorry

end NUMINAMATH_CALUDE_racecar_repair_cost_l1866_186636


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1866_186629

theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n^2 - 19*n + 91 = k^2) ↔ (n = 9 ∨ n = 10) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1866_186629


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1866_186610

theorem inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | a * x^2 - (a + 2) * x + 2 ≥ 0} = {x : ℝ | 2/a ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1866_186610


namespace NUMINAMATH_CALUDE_traditionalist_fraction_l1866_186626

theorem traditionalist_fraction (num_provinces : ℕ) (num_progressives : ℕ) (num_traditionalists_per_province : ℕ) :
  num_provinces = 4 →
  num_traditionalists_per_province * 12 = num_progressives →
  (num_traditionalists_per_province * num_provinces) / (num_progressives + num_traditionalists_per_province * num_provinces) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_traditionalist_fraction_l1866_186626


namespace NUMINAMATH_CALUDE_period_of_cos_3x_l1866_186677

theorem period_of_cos_3x :
  let f : ℝ → ℝ := λ x => Real.cos (3 * x)
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, 0 < q ∧ q < p → ∃ x : ℝ, f (x + q) ≠ f x :=
by
  sorry

end NUMINAMATH_CALUDE_period_of_cos_3x_l1866_186677


namespace NUMINAMATH_CALUDE_square_circle_ratio_l1866_186644

theorem square_circle_ratio (s r : ℝ) (h : s > 0 ∧ r > 0) :
  s^2 / (π * r^2) = 250 / 196 →
  ∃ (a b c : ℕ), (a : ℝ) * Real.sqrt b / c = s / r ∧ a = 5 ∧ b = 10 ∧ c = 14 ∧ a + b + c = 29 :=
by sorry

end NUMINAMATH_CALUDE_square_circle_ratio_l1866_186644


namespace NUMINAMATH_CALUDE_remaining_segments_theorem_l1866_186681

/-- Represents the spiral pattern described in the problem -/
def spiral_pattern (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2) + n + 1

/-- The total length of the spiral in centimeters -/
def total_length : ℕ := 400

/-- The number of segments already drawn -/
def segments_drawn : ℕ := 7

/-- Calculates the total number of segments in the spiral -/
def total_segments (n : ℕ) : ℕ := 2 * n + 1

theorem remaining_segments_theorem :
  ∃ n : ℕ, 
    spiral_pattern n = total_length ∧ 
    total_segments n - segments_drawn = 32 :=
sorry

end NUMINAMATH_CALUDE_remaining_segments_theorem_l1866_186681


namespace NUMINAMATH_CALUDE_election_winner_votes_l1866_186683

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  winner_percentage = 54/100 →
  vote_difference = 288 →
  ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference →
  ↑total_votes * winner_percentage = 1944 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1866_186683


namespace NUMINAMATH_CALUDE_special_polyhedron_properties_l1866_186672

structure Polyhedron where
  convex : Bool
  flat_faces : Bool
  symmetry_planes : Nat
  vertices : Nat
  edges_per_vertex : Nat
  vertex_types : List (Nat × List Nat)

def special_polyhedron : Polyhedron :=
{
  convex := true,
  flat_faces := true,
  symmetry_planes := 2,
  vertices := 8,
  edges_per_vertex := 3,
  vertex_types := [
    (2, [1, 1, 1]),
    (4, [1, 1, 2]),
    (2, [2, 2, 3])
  ]
}

theorem special_polyhedron_properties (K : Polyhedron) 
  (h : K = special_polyhedron) : 
  ∃ (surface_area volume : ℝ), 
    surface_area = 13.86 ∧ 
    volume = 2.946 :=
sorry

end NUMINAMATH_CALUDE_special_polyhedron_properties_l1866_186672


namespace NUMINAMATH_CALUDE_circle_equation_from_parabola_focus_l1866_186666

/-- Given a parabola y^2 = 4x and a circle with its center at the focus of the parabola
    passing through the origin, the equation of the circle is x^2 + y^2 - 2x = 0 -/
theorem circle_equation_from_parabola_focus (x y : ℝ) :
  (y^2 = 4*x) →  -- Parabola equation
  (∃ (h k r : ℝ), (h = 1 ∧ k = 0) ∧  -- Focus at (1, 0)
    ((0 - h)^2 + (0 - k)^2 = r^2) ∧  -- Circle passes through origin
    ((x - h)^2 + (y - k)^2 = r^2)) →  -- General circle equation
  x^2 + y^2 - 2*x = 0 :=  -- Resulting circle equation
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_parabola_focus_l1866_186666
