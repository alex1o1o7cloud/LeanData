import Mathlib

namespace NUMINAMATH_CALUDE_cookies_per_bag_l2647_264777

theorem cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (baggies : ℕ) :
  chocolate_chip = 13 →
  oatmeal = 41 →
  baggies = 6 →
  (chocolate_chip + oatmeal) / baggies = 9 := by
sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l2647_264777


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2647_264776

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 80 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h1 : ArithmeticSequence a)
  (h2 : SumCondition a) :
  a 5 + (1/4) * a 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2647_264776


namespace NUMINAMATH_CALUDE_greater_root_of_quadratic_l2647_264741

theorem greater_root_of_quadratic (x : ℝ) :
  x^2 - 5*x - 36 = 0 → x ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greater_root_of_quadratic_l2647_264741


namespace NUMINAMATH_CALUDE_delta_composition_l2647_264746

-- Define the Delta operations
def rightDelta (x : ℤ) : ℤ := 9 - x
def leftDelta (x : ℤ) : ℤ := x - 9

-- State the theorem
theorem delta_composition : leftDelta (rightDelta 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_delta_composition_l2647_264746


namespace NUMINAMATH_CALUDE_function_uniqueness_l2647_264733

theorem function_uniqueness (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) = f n + 1)
  (h2 : ∃ k, f k = 1)
  (h3 : ∀ m, ∃ n, f n ≤ m) :
  ∀ n, f n = n + 1 := by
sorry

end NUMINAMATH_CALUDE_function_uniqueness_l2647_264733


namespace NUMINAMATH_CALUDE_point_A_coordinates_l2647_264757

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the translation operation
def translate (p : Point2D) (dx dy : ℝ) : Point2D :=
  { x := p.x + dx, y := p.y + dy }

theorem point_A_coordinates : 
  ∀ A : Point2D, 
  let B := translate (translate A 0 (-3)) 2 0
  B = Point2D.mk (-1) 5 → A = Point2D.mk (-3) 8 := by
sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l2647_264757


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l2647_264762

/-- Given that points (-3,-1) and (4,-6) are on opposite sides of the line 3x-2y-a=0,
    the range of values for a is (-7, 24). -/
theorem opposite_sides_line_range (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ = -3 ∧ y₁ = -1 ∧ x₂ = 4 ∧ y₂ = -6 ∧ 
    (3*x₁ - 2*y₁ - a) * (3*x₂ - 2*y₂ - a) < 0) ↔ 
  -7 < a ∧ a < 24 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l2647_264762


namespace NUMINAMATH_CALUDE_f_monotone_increasing_min_m_value_l2647_264748

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1/x) * log x

theorem f_monotone_increasing :
  StrictMono f := by sorry

theorem min_m_value (m : ℝ) :
  (∀ x > 0, (2 * f x - m) / (exp (m * x)) ≤ m) ↔ m ≥ 2/exp 1 := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_min_m_value_l2647_264748


namespace NUMINAMATH_CALUDE_mode_of_visual_acuity_l2647_264783

-- Define the visual acuity values and their frequencies
def visual_acuity : List ℝ := [4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
def frequencies : List ℕ := [2, 3, 6, 9, 12, 8, 5, 3]

-- Define a function to find the mode
def mode (values : List ℝ) (freqs : List ℕ) : ℝ :=
  let pairs := List.zip values freqs
  let maxFreq := List.foldl (fun acc (_, f) => max acc f) 0 pairs
  let modes := List.filter (fun (_, f) => f == maxFreq) pairs
  (List.head! modes).1

-- Theorem: The mode of visual acuity is 4.7
theorem mode_of_visual_acuity :
  mode visual_acuity frequencies = 4.7 :=
by sorry

end NUMINAMATH_CALUDE_mode_of_visual_acuity_l2647_264783


namespace NUMINAMATH_CALUDE_min_value_on_circle_l2647_264758

theorem min_value_on_circle (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l2647_264758


namespace NUMINAMATH_CALUDE_wire_ratio_l2647_264797

theorem wire_ratio (a b : ℝ) (h : a > 0) (k : b > 0) : 
  (a^2 / 16 = b^2 / (4 * Real.pi)) → a / b = 2 / Real.sqrt Real.pi := by
sorry

end NUMINAMATH_CALUDE_wire_ratio_l2647_264797


namespace NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_l2647_264726

-- Define the propositions
def p (m : ℝ) : Prop := m = -1
def q (m : ℝ) : Prop := ∀ (x y : ℝ), (x - 1 = 0) ∧ (x + m^2 * y = 0) → 
  (∃ (k : ℝ), k ≠ 0 ∧ (1 * k = m^2) ∨ (1 * m^2 = -k))

-- Theorem statement
theorem p_neither_sufficient_nor_necessary :
  (∃ m : ℝ, p m ∧ ¬q m) ∧ (∃ m : ℝ, q m ∧ ¬p m) :=
sorry

end NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_l2647_264726


namespace NUMINAMATH_CALUDE_sarah_candy_duration_l2647_264765

/-- The number of days Sarah's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (friends_candy : ℕ) 
  (traded_candy : ℕ) (given_away_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  let total_received := neighbors_candy + sister_candy + friends_candy
  let total_removed := traded_candy + given_away_candy
  let remaining_candy := total_received - total_removed
  remaining_candy / daily_consumption

/-- Theorem stating that Sarah's candy will last 9 days -/
theorem sarah_candy_duration : 
  candy_duration 66 15 20 10 5 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sarah_candy_duration_l2647_264765


namespace NUMINAMATH_CALUDE_triangle_area_l2647_264744

-- Define the plane region
def PlaneRegion (k : ℝ) := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 2 * p.1 ∧ k * p.1 - p.2 + 1 ≥ 0}

-- Define a right triangle
def IsRightTriangle (r : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a set in ℝ²
noncomputable def Area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem triangle_area (k : ℝ) :
  IsRightTriangle (PlaneRegion k) →
  Area (PlaneRegion k) = 1/5 ∨ Area (PlaneRegion k) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2647_264744


namespace NUMINAMATH_CALUDE_new_cards_for_500_l2647_264798

/-- Given a total number of cards, calculate the number of new cards received
    when trading one-fifth of the duplicate cards, where duplicates are one-fourth
    of the total. -/
def new_cards_received (total : ℕ) : ℕ :=
  (total / 4) / 5

/-- Theorem stating that given 500 total cards, the number of new cards
    received is 25. -/
theorem new_cards_for_500 : new_cards_received 500 = 25 := by
  sorry

end NUMINAMATH_CALUDE_new_cards_for_500_l2647_264798


namespace NUMINAMATH_CALUDE_complex_root_magnitude_l2647_264784

theorem complex_root_magnitude (z : ℂ) : z^2 + 2*z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_l2647_264784


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l2647_264737

theorem complex_fraction_equals_i (a b : ℝ) (h : Complex.I * (a + Complex.I * b) = Complex.I * (2 - Complex.I)) :
  (b + Complex.I * a) / (a - Complex.I * b) = Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l2647_264737


namespace NUMINAMATH_CALUDE_slope_product_of_triple_angle_and_slope_l2647_264775

/-- Given two non-horizontal lines with slopes m and n, where one line forms
    three times as large an angle with the horizontal as the other and has
    three times the slope, prove that mn = 9/4 -/
theorem slope_product_of_triple_angle_and_slope
  (m n : ℝ) -- slopes of the lines
  (h₁ : m ≠ 0) -- L₁ is not horizontal
  (h₂ : n ≠ 0) -- L₂ is not horizontal
  (h₃ : ∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) -- angle relation
  (h₄ : m = 3 * n) -- slope relation
  : m * n = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_product_of_triple_angle_and_slope_l2647_264775


namespace NUMINAMATH_CALUDE_sequence_relation_l2647_264779

/-- Given two sequences a and b, where a_n = n^2 and b_n are distinct positive integers,
    and for all n, the a_n-th term of b equals the b_n-th term of a,
    prove that (log(b 1 * b 4 * b 9 * b 16)) / (log(b 1 * b 2 * b 3 * b 4)) = 2 -/
theorem sequence_relation (b : ℕ+ → ℕ+) 
  (h_distinct : ∀ m n : ℕ+, m ≠ n → b m ≠ b n)
  (h_relation : ∀ n : ℕ+, b (n^2) = (b n)^2) :
  (Real.log ((b 1) * (b 4) * (b 9) * (b 16))) / (Real.log ((b 1) * (b 2) * (b 3) * (b 4))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_relation_l2647_264779


namespace NUMINAMATH_CALUDE_compression_force_l2647_264719

/-- Compression force calculation for cylindrical pillars -/
theorem compression_force (T H L : ℝ) : 
  T = 3 → H = 9 → L = (30 * T^5) / H^3 → L = 10 := by
  sorry

end NUMINAMATH_CALUDE_compression_force_l2647_264719


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2647_264718

theorem parallel_vectors_x_value (x : ℝ) 
  (h1 : x > 0) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (h2 : a = (8 + x/2, x)) 
  (h3 : b = (x + 1, 2)) 
  (h4 : ∃ (k : ℝ), a = k • b) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2647_264718


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2647_264792

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (n_Al n_Cl n_O : ℕ) (w_Al w_Cl w_O : ℝ) : ℝ :=
  n_Al * w_Al + n_Cl * w_Cl + n_O * w_O

/-- The molecular weight of a compound with 2 Al, 6 Cl, and 3 O atoms is 314.66 g/mol -/
theorem compound_molecular_weight :
  molecular_weight 2 6 3 26.98 35.45 16.00 = 314.66 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2647_264792


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2647_264773

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_perimeter_l2647_264773


namespace NUMINAMATH_CALUDE_area_of_extended_quadrilateral_l2647_264743

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (EF FG GH HE : ℝ)
  (area : ℝ)

-- Define the extended quadrilateral E'F'G'H'
structure ExtendedQuadrilateral :=
  (base : Quadrilateral)
  (EE' FF' GG' HH' : ℝ)

-- Define our specific quadrilateral
def EFGH : Quadrilateral :=
  { EF := 5
  , FG := 10
  , GH := 9
  , HE := 7
  , area := 12 }

-- Define our specific extended quadrilateral
def EFGH_extended : ExtendedQuadrilateral :=
  { base := EFGH
  , EE' := 7
  , FF' := 5
  , GG' := 10
  , HH' := 9 }

-- State the theorem
theorem area_of_extended_quadrilateral :
  (EFGH_extended.base.area + 
   EFGH_extended.base.EF * EFGH_extended.FF' +
   EFGH_extended.base.FG * EFGH_extended.GG' +
   EFGH_extended.base.GH * EFGH_extended.HH' +
   EFGH_extended.base.HE * EFGH_extended.EE') = 36 := by
  sorry

end NUMINAMATH_CALUDE_area_of_extended_quadrilateral_l2647_264743


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2647_264728

/-- Given an arithmetic sequence, prove that if the sum of the first four terms is 2l,
    the sum of the last four terms is 67, and the sum of the first n terms is 286,
    then the number of terms n is 26. -/
theorem arithmetic_sequence_problem (l : ℝ) (a d : ℝ) (n : ℕ) :
  (4 * a + 6 * d = 2 * l) →
  (4 * (a + (n - 1) * d) - 6 * d = 67) →
  (n * (2 * a + (n - 1) * d) / 2 = 286) →
  n = 26 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2647_264728


namespace NUMINAMATH_CALUDE_system_solution_l2647_264702

theorem system_solution (x y : ℝ) (eq1 : 2 * x + y = 6) (eq2 : x + 2 * y = 3) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2647_264702


namespace NUMINAMATH_CALUDE_probability_both_presidents_selected_l2647_264793

def club_sizes : List Nat := [6, 8, 9, 10]

def probability_both_presidents (n : Nat) : Rat :=
  (Nat.choose (n - 2) 2 : Rat) / (Nat.choose n 4 : Rat)

theorem probability_both_presidents_selected :
  (1 / 4 : Rat) * (club_sizes.map probability_both_presidents).sum = 119 / 700 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_presidents_selected_l2647_264793


namespace NUMINAMATH_CALUDE_grandmas_salad_ratio_l2647_264704

/-- Given the conditions of Grandma's salad, prove the ratio of pickles to cherry tomatoes -/
theorem grandmas_salad_ratio : 
  ∀ (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ),
    mushrooms = 3 →
    cherry_tomatoes = 2 * mushrooms →
    bacon_bits = 4 * pickles →
    red_bacon_bits * 3 = bacon_bits →
    red_bacon_bits = 32 →
    (pickles : ℚ) / cherry_tomatoes = 4 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_grandmas_salad_ratio_l2647_264704


namespace NUMINAMATH_CALUDE_corn_acreage_l2647_264754

theorem corn_acreage (total_land : ℕ) (ratio_beans : ℕ) (ratio_wheat : ℕ) (ratio_corn : ℕ) 
  (h1 : total_land = 1034)
  (h2 : ratio_beans = 5)
  (h3 : ratio_wheat = 2)
  (h4 : ratio_corn = 4) : 
  (total_land * ratio_corn) / (ratio_beans + ratio_wheat + ratio_corn) = 376 := by
  sorry

#eval (1034 * 4) / (5 + 2 + 4)

end NUMINAMATH_CALUDE_corn_acreage_l2647_264754


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l2647_264739

/-- An ellipse with center at the origin, one focus at (0,2), and a chord formed by
    the intersection with the line y=3x+7 whose midpoint has a y-coordinate of 1 --/
structure SpecialEllipse where
  /-- One focus of the ellipse --/
  focus : ℝ × ℝ
  /-- Slope of the intersecting line --/
  m : ℝ
  /-- y-intercept of the intersecting line --/
  b : ℝ
  /-- y-coordinate of the chord's midpoint --/
  midpoint_y : ℝ
  /-- Conditions for the special ellipse --/
  h1 : focus = (0, 2)
  h2 : m = 3
  h3 : b = 7
  h4 : midpoint_y = 1

/-- The equation of the ellipse --/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 12 = 1

/-- Theorem stating that the given special ellipse has the specified equation --/
theorem special_ellipse_equation (e : SpecialEllipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ellipse_equation p.1 p.2} ↔
    (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 12 = 1} :=
by sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l2647_264739


namespace NUMINAMATH_CALUDE_return_speed_calculation_l2647_264716

theorem return_speed_calculation (total_distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) 
  (h1 : total_distance = 300) 
  (h2 : outbound_speed = 75) 
  (h3 : average_speed = 50) :
  ∃ inbound_speed : ℝ, 
    inbound_speed = 37.5 ∧ 
    average_speed = total_distance / (total_distance / (2 * outbound_speed) + total_distance / (2 * inbound_speed)) := by
  sorry

end NUMINAMATH_CALUDE_return_speed_calculation_l2647_264716


namespace NUMINAMATH_CALUDE_complex_fourth_power_l2647_264736

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l2647_264736


namespace NUMINAMATH_CALUDE_f_of_five_equals_102_l2647_264786

/-- Given a function f(x) = 2x^2 + y where f(2) = 60, prove that f(5) = 102 -/
theorem f_of_five_equals_102 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 60) :
  f 5 = 102 := by
  sorry

end NUMINAMATH_CALUDE_f_of_five_equals_102_l2647_264786


namespace NUMINAMATH_CALUDE_digit_interchange_effect_l2647_264707

theorem digit_interchange_effect (n : ℕ) (p q : ℕ) : 
  n = 9 → 
  p > q → 
  p - q = 1 → 
  (10 * p + q) - (10 * q + p) = (n : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_digit_interchange_effect_l2647_264707


namespace NUMINAMATH_CALUDE_incorrect_vs_correct_calculation_l2647_264740

theorem incorrect_vs_correct_calculation (x : ℝ) (h : x - 3 + 49 = 66) : 
  (3 * x + 49) - 66 = 43 := by
sorry

end NUMINAMATH_CALUDE_incorrect_vs_correct_calculation_l2647_264740


namespace NUMINAMATH_CALUDE_defense_attorney_implication_l2647_264794

-- Define propositions
variable (P : Prop) -- P represents "the defendant is guilty"
variable (Q : Prop) -- Q represents "the defendant had an accomplice"

-- Theorem statement
theorem defense_attorney_implication : ¬(P → Q) → (P ∧ ¬Q) := by
  sorry

end NUMINAMATH_CALUDE_defense_attorney_implication_l2647_264794


namespace NUMINAMATH_CALUDE_value_of_x_l2647_264722

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2647_264722


namespace NUMINAMATH_CALUDE_sum_coefficients_x_minus_3y_to_20_l2647_264780

theorem sum_coefficients_x_minus_3y_to_20 :
  (fun x y => (x - 3 * y) ^ 20) 1 1 = 1048576 := by
  sorry

end NUMINAMATH_CALUDE_sum_coefficients_x_minus_3y_to_20_l2647_264780


namespace NUMINAMATH_CALUDE_angstadt_student_count_l2647_264749

/-- Given that:
  1. Half of Mr. Angstadt's students are enrolled in Statistics.
  2. 90% of the students in Statistics are seniors.
  3. There are 54 seniors enrolled in Statistics.
  Prove that Mr. Angstadt has 120 students throughout the school day. -/
theorem angstadt_student_count :
  ∀ (total_students stats_students seniors : ℕ),
  stats_students = total_students / 2 →
  seniors = (90 * stats_students) / 100 →
  seniors = 54 →
  total_students = 120 :=
by sorry

end NUMINAMATH_CALUDE_angstadt_student_count_l2647_264749


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2647_264712

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2647_264712


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l2647_264732

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_2 : a 2 = 3)
  (h_6 : a 6 = 7) :
  a 11 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l2647_264732


namespace NUMINAMATH_CALUDE_power_equality_l2647_264764

theorem power_equality : 32^4 * 4^6 = 2^32 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2647_264764


namespace NUMINAMATH_CALUDE_negation_of_all_ge_two_l2647_264747

theorem negation_of_all_ge_two :
  (¬ (∀ x : ℝ, x ≥ 2)) ↔ (∃ x₀ : ℝ, x₀ < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_ge_two_l2647_264747


namespace NUMINAMATH_CALUDE_isosceles_triangle_on_parabola_l2647_264745

/-- Given two points P and Q on the parabola y = -x^2 that form an isosceles triangle POQ with the origin O,
    prove that the distance between P and Q is twice the x-coordinate of P. -/
theorem isosceles_triangle_on_parabola (p : ℝ) :
  let P : ℝ × ℝ := (p, -p^2)
  let Q : ℝ × ℝ := (-p, -p^2)
  let O : ℝ × ℝ := (0, 0)
  (P.1^2 + P.2^2 = Q.1^2 + Q.2^2) →  -- PO = OQ (isosceles condition)
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * p  -- PQ = 2p
:= by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_on_parabola_l2647_264745


namespace NUMINAMATH_CALUDE_max_safe_sages_is_82_l2647_264703

/-- Represents a train with a given number of wagons. -/
structure Train :=
  (num_wagons : ℕ)

/-- Represents the journey details. -/
structure Journey :=
  (start_station : ℕ)
  (end_station : ℕ)
  (controller_start : ℕ)
  (controller_move_interval : ℕ)

/-- Represents the movement capabilities of sages. -/
structure SageMovement :=
  (max_move : ℕ)

/-- Represents the visibility range of sages. -/
structure SageVisibility :=
  (range : ℕ)

/-- Represents the maximum number of sages that can avoid controllers. -/
def max_safe_sages (t : Train) (j : Journey) (sm : SageMovement) (sv : SageVisibility) : ℕ :=
  82

/-- Theorem stating that 82 is the maximum number of sages that can avoid controllers. -/
theorem max_safe_sages_is_82 
  (t : Train) 
  (j : Journey) 
  (sm : SageMovement) 
  (sv : SageVisibility) : 
  max_safe_sages t j sm sv = 82 :=
by sorry

end NUMINAMATH_CALUDE_max_safe_sages_is_82_l2647_264703


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2647_264715

/-- Given two positive integers with specific LCM and HCF, prove one number given the other -/
theorem lcm_hcf_problem (A B : ℕ) (h1 : Nat.lcm A B = 7700) (h2 : Nat.gcd A B = 11) (h3 : B = 275) :
  A = 308 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2647_264715


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2647_264756

/-- Proves that mixing 100 mL of 10% alcohol solution with 300 mL of 30% alcohol solution
    results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 100
  let x_concentration : ℝ := 0.1
  let y_volume : ℝ := 300
  let y_concentration : ℝ := 0.3
  let target_concentration : ℝ := 0.25
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = target_concentration := by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2647_264756


namespace NUMINAMATH_CALUDE_dice_roll_probability_l2647_264720

/-- The probability of rolling a specific number on a standard six-sided die -/
def prob_single_roll : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a single die -/
def prob_not_one : ℚ := 5 / 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (numbers between 10 and 20 inclusive) -/
def favorable_outcomes : ℕ := 11

theorem dice_roll_probability : 
  (1 : ℚ) - prob_not_one * prob_not_one = favorable_outcomes / total_outcomes := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l2647_264720


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2647_264799

theorem sum_of_coefficients (x : ℝ) : 
  let p : ℝ → ℝ := λ x => 2 * (4 * x^8 - 5 * x^3 + 6) + 8 * (x^6 + 3 * x^4 - 4)
  p 1 = 10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2647_264799


namespace NUMINAMATH_CALUDE_largest_reciprocal_l2647_264701

theorem largest_reciprocal (a b c d e : ℝ) (ha : a = 1/4) (hb : b = 3/8) (hc : c = 1/2) (hd : d = 4) (he : e = 1000) :
  (1 / a > 1 / b) ∧ (1 / a > 1 / c) ∧ (1 / a > 1 / d) ∧ (1 / a > 1 / e) := by
  sorry

#check largest_reciprocal

end NUMINAMATH_CALUDE_largest_reciprocal_l2647_264701


namespace NUMINAMATH_CALUDE_point_two_units_from_negative_one_l2647_264711

theorem point_two_units_from_negative_one (x : ℝ) : 
  (|x - (-1)| = 2) ↔ (x = -3 ∨ x = 1) := by sorry

end NUMINAMATH_CALUDE_point_two_units_from_negative_one_l2647_264711


namespace NUMINAMATH_CALUDE_sequence_general_term_l2647_264767

theorem sequence_general_term (a : ℕ → ℚ) :
  a 1 = -1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = a (n + 1) - a n) →
  ∀ n : ℕ, n ≥ 1 → a n = -1 / n :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2647_264767


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l2647_264789

theorem sum_remainder_mod_nine (n : ℤ) : (8 - n + (n + 5)) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l2647_264789


namespace NUMINAMATH_CALUDE_final_selling_price_l2647_264778

/-- Calculate the final selling price of items with given conditions -/
theorem final_selling_price :
  let cycle_price : ℚ := 1400
  let helmet_price : ℚ := 400
  let safety_light_price : ℚ := 200
  let cycle_discount : ℚ := 0.1
  let helmet_discount : ℚ := 0.05
  let tax_rate : ℚ := 0.05
  let cycle_loss : ℚ := 0.12
  let helmet_profit : ℚ := 0.25
  let lock_price : ℚ := 300
  let transaction_fee : ℚ := 0.03

  let discounted_cycle := cycle_price * (1 - cycle_discount)
  let discounted_helmet := helmet_price * (1 - helmet_discount)
  let total_safety_lights := 2 * safety_light_price

  let total_before_tax := discounted_cycle + discounted_helmet + total_safety_lights
  let total_after_tax := total_before_tax * (1 + tax_rate)

  let selling_cycle := discounted_cycle * (1 - cycle_loss)
  let selling_helmet := discounted_helmet * (1 + helmet_profit)
  let selling_safety_lights := total_safety_lights

  let total_selling_before_fee := selling_cycle + selling_helmet + selling_safety_lights + lock_price
  let fee_amount := total_selling_before_fee * transaction_fee
  let total_selling_after_fee := total_selling_before_fee - fee_amount

  let final_price := ⌊total_selling_after_fee⌋

  final_price = 2215 := by sorry

end NUMINAMATH_CALUDE_final_selling_price_l2647_264778


namespace NUMINAMATH_CALUDE_no_integer_satisfies_inequality_l2647_264713

theorem no_integer_satisfies_inequality : 
  ¬ ∃ (n : ℤ), n > 1 ∧ (⌊Real.sqrt (n - 2) + 2 * Real.sqrt (n + 2)⌋ : ℤ) < ⌊Real.sqrt (9 * n + 6)⌋ := by
  sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_inequality_l2647_264713


namespace NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l2647_264788

theorem max_integer_a_for_real_roots : 
  ∀ a : ℤ, 
  (a ≠ 1) → 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 3 = 0) → 
  a ≤ 0 ∧ 
  ∀ b : ℤ, b > 0 → ¬(∃ x : ℝ, (b - 1) * x^2 - 2 * x + 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l2647_264788


namespace NUMINAMATH_CALUDE_expression_as_square_difference_l2647_264738

/-- The square difference formula for two real numbers -/
def square_difference (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- The expression (-x+y)(x+y) can be calculated using the square difference formula -/
theorem expression_as_square_difference (x y : ℝ) :
  ∃ (a b : ℝ), (-x + y) * (x + y) = square_difference a b :=
sorry

end NUMINAMATH_CALUDE_expression_as_square_difference_l2647_264738


namespace NUMINAMATH_CALUDE_emily_subtraction_l2647_264721

theorem emily_subtraction : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by sorry

end NUMINAMATH_CALUDE_emily_subtraction_l2647_264721


namespace NUMINAMATH_CALUDE_inequality_solution_l2647_264768

theorem inequality_solution (x : ℝ) : 
  (x / (x - 1) ≥ 2 * x) ↔ (1 < x ∧ x ≤ 3/2) ∨ (x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2647_264768


namespace NUMINAMATH_CALUDE_eggs_at_town_hall_l2647_264729

/-- Given the number of eggs found at different locations during an Easter egg hunt, 
    this theorem proves how many eggs were found at the town hall. -/
theorem eggs_at_town_hall 
  (total_eggs : ℕ)
  (club_house_eggs : ℕ)
  (park_eggs : ℕ)
  (h1 : total_eggs = 80)
  (h2 : club_house_eggs = 40)
  (h3 : park_eggs = 25) :
  total_eggs - (club_house_eggs + park_eggs) = 15 := by
  sorry

#check eggs_at_town_hall

end NUMINAMATH_CALUDE_eggs_at_town_hall_l2647_264729


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2647_264723

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), x = 2 + (35 / 100) * (1 / (1 - 1/100)^n)) →
  x = 233 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2647_264723


namespace NUMINAMATH_CALUDE_consumption_increase_l2647_264791

theorem consumption_increase (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let new_tax := 0.7 * T
  let new_revenue := 0.77 * (T * C)
  let new_consumption := C * (1 + 10/100)
  new_tax * new_consumption = new_revenue :=
sorry

end NUMINAMATH_CALUDE_consumption_increase_l2647_264791


namespace NUMINAMATH_CALUDE_solution_pairs_l2647_264795

theorem solution_pairs (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l2647_264795


namespace NUMINAMATH_CALUDE_total_units_is_531_l2647_264709

/-- A mixed-use development with various floor types and unit distributions -/
structure MixedUseDevelopment where
  total_floors : Nat
  regular_floors : Nat
  luxury_floors : Nat
  penthouse_floors : Nat
  commercial_floors : Nat
  other_floors : Nat
  regular_odd_units : Nat
  regular_even_units : Nat
  luxury_avg_units : Nat
  penthouse_units : Nat
  commercial_units : Nat
  amenities_uncounted_units : Nat
  other_uncounted_units : Nat

/-- Calculate the total number of units in the mixed-use development -/
def total_units (dev : MixedUseDevelopment) : Nat :=
  let regular_units := (dev.regular_floors / 2 + dev.regular_floors % 2) * dev.regular_odd_units +
                       (dev.regular_floors / 2) * dev.regular_even_units
  let luxury_units := dev.luxury_floors * dev.luxury_avg_units
  let penthouse_units := dev.penthouse_floors * dev.penthouse_units
  let commercial_units := dev.commercial_floors * dev.commercial_units
  let uncounted_units := dev.amenities_uncounted_units + dev.other_uncounted_units
  regular_units + luxury_units + penthouse_units + commercial_units + uncounted_units

/-- The mixed-use development described in the problem -/
def problem_development : MixedUseDevelopment where
  total_floors := 60
  regular_floors := 25
  luxury_floors := 20
  penthouse_floors := 10
  commercial_floors := 3
  other_floors := 2
  regular_odd_units := 14
  regular_even_units := 12
  luxury_avg_units := 8
  penthouse_units := 2
  commercial_units := 5
  amenities_uncounted_units := 4
  other_uncounted_units := 6

/-- Theorem stating that the total number of units in the problem development is 531 -/
theorem total_units_is_531 : total_units problem_development = 531 := by
  sorry


end NUMINAMATH_CALUDE_total_units_is_531_l2647_264709


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_81_l2647_264742

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_81_l2647_264742


namespace NUMINAMATH_CALUDE_crayons_erasers_difference_l2647_264700

/-- Given the initial number of crayons and erasers, and the remaining number of crayons,
    prove that the difference between remaining crayons and erasers is 353. -/
theorem crayons_erasers_difference (initial_crayons : ℕ) (initial_erasers : ℕ) (remaining_crayons : ℕ) 
    (h1 : initial_crayons = 531)
    (h2 : initial_erasers = 38)
    (h3 : remaining_crayons = 391) : 
  remaining_crayons - initial_erasers = 353 := by
  sorry

end NUMINAMATH_CALUDE_crayons_erasers_difference_l2647_264700


namespace NUMINAMATH_CALUDE_problem_G2_1_l2647_264796

theorem problem_G2_1 (a : ℚ) :
  137 / a = 0.1234234234235 → a = 1110 := by sorry

end NUMINAMATH_CALUDE_problem_G2_1_l2647_264796


namespace NUMINAMATH_CALUDE_square_side_length_l2647_264787

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 12) (h2 : side > 0) (h3 : area = side ^ 2) :
  side = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l2647_264787


namespace NUMINAMATH_CALUDE_diamonds_in_G10_num_diamonds_formula_num_diamonds_induction_l2647_264781

/-- Represents the number of diamonds in figure G_n -/
def num_diamonds (n : ℕ) : ℕ :=
  4 * n^2 + 5 * n - 8

theorem diamonds_in_G10 :
  num_diamonds 10 = 442 :=
by sorry

theorem num_diamonds_formula (n : ℕ) :
  n ≥ 1 →
  num_diamonds n =
    1 + -- initial diamond in G_1
    (4 * (n - 1) * n) + -- diamonds added to sides
    (8 * (n - 1)) -- diamonds added to corners
  :=
by sorry

theorem num_diamonds_induction (n : ℕ) :
  n ≥ 1 →
  num_diamonds n =
    (if n = 1 then 1
     else num_diamonds (n - 1) + 8 * (4 * (n - 1) + 1))
  :=
by sorry

end NUMINAMATH_CALUDE_diamonds_in_G10_num_diamonds_formula_num_diamonds_induction_l2647_264781


namespace NUMINAMATH_CALUDE_andrews_family_size_l2647_264790

/-- Given the conditions of Andrew's family mask usage, prove the number of family members excluding Andrew. -/
theorem andrews_family_size (total_masks : ℕ) (change_interval : ℕ) (total_days : ℕ) :
  total_masks = 100 →
  change_interval = 4 →
  total_days = 80 →
  ∃ (family_size : ℕ), family_size = 4 ∧ 
    (family_size + 1) * (total_days / change_interval) = total_masks :=
by sorry

end NUMINAMATH_CALUDE_andrews_family_size_l2647_264790


namespace NUMINAMATH_CALUDE_three_fractions_inequality_l2647_264751

theorem three_fractions_inequality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_inequality : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), k > 0 ∧ x = 2*k ∧ y = k ∧ z = k) :=
by sorry

end NUMINAMATH_CALUDE_three_fractions_inequality_l2647_264751


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2647_264755

theorem complex_equation_solution (z : ℂ) : 4 + 2 * Complex.I * z = 2 - 6 * Complex.I * z ↔ z = Complex.I / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2647_264755


namespace NUMINAMATH_CALUDE_wally_initial_tickets_l2647_264774

/-- Proves that Wally had 400 tickets initially given the conditions of the problem -/
theorem wally_initial_tickets : 
  ∀ (total : ℕ) (jensen finley : ℕ),
  (3 : ℚ) / 4 * total = jensen + finley →
  jensen * 11 = finley * 4 →
  finley = 220 →
  total = 400 :=
by sorry

end NUMINAMATH_CALUDE_wally_initial_tickets_l2647_264774


namespace NUMINAMATH_CALUDE_prosecutor_conclusion_l2647_264770

-- Define the types for guilt
inductive Guilt
| Guilty
| NotGuilty

-- Define the prosecutor's statements
def statement1 (X Y : Guilt) : Prop :=
  X = Guilt.NotGuilty ∨ Y = Guilt.Guilty

def statement2 (X : Guilt) : Prop :=
  X = Guilt.Guilty

-- Theorem to prove
theorem prosecutor_conclusion (X Y : Guilt) :
  statement1 X Y ∧ statement2 X →
  X = Guilt.Guilty ∧ Y = Guilt.Guilty :=
by
  sorry


end NUMINAMATH_CALUDE_prosecutor_conclusion_l2647_264770


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2647_264760

theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (first_term : a 1 = 5) 
  (fifth_term : a 5 = 2025) :
  a 3 = 225 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2647_264760


namespace NUMINAMATH_CALUDE_wood_measurement_correct_l2647_264771

/-- Represents the system of equations for the wood measurement problem from "The Mathematical Classic of Sunzi" --/
def wood_measurement_system (x y : ℝ) : Prop :=
  (x - y = 4.5) ∧ (y - (1/2) * x = 1)

/-- Theorem stating that the system of equations correctly represents the wood measurement problem --/
theorem wood_measurement_correct (x y : ℝ) :
  (x > y) ∧                         -- rope is longer than wood
  (x - y = 4.5) ∧                   -- 4.5 feet of rope left when measuring
  (y > (1/2) * x) ∧                 -- wood is longer than half the rope
  (y - (1/2) * x = 1) →             -- rope falls short by 1 foot when folded
  wood_measurement_system x y := by
  sorry


end NUMINAMATH_CALUDE_wood_measurement_correct_l2647_264771


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2647_264752

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a = 5 ∧ b = 12) ∨ (a = 5 ∧ c = 12) ∨ (b = 5 ∧ c = 12) →
  a^2 + b^2 = c^2 →
  c = 12 ∨ c = 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2647_264752


namespace NUMINAMATH_CALUDE_problem_solution_l2647_264724

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log x - x + 1

theorem problem_solution :
  (∃! a : ℝ, ∀ x > 0, f a x ≤ 0) ∧
  (∀ x ∈ Set.Ioo 0 (Real.pi / 2), Real.exp x * Real.sin x - x > f 1 x) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2647_264724


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l2647_264727

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l2647_264727


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l2647_264706

/-- The number of friends Alex has -/
def num_friends : ℕ := 15

/-- The initial number of coins Alex has -/
def initial_coins : ℕ := 100

/-- The minimum number of coins needed to distribute to friends -/
def min_coins_needed : ℕ := (num_friends * (num_friends + 1)) / 2

/-- The number of additional coins needed -/
def additional_coins_needed : ℕ := min_coins_needed - initial_coins

theorem alex_coin_distribution :
  additional_coins_needed = 20 := by sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l2647_264706


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l2647_264731

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - 3*x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 3

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = -2*x + 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l2647_264731


namespace NUMINAMATH_CALUDE_roots_of_unity_real_fifth_power_l2647_264766

theorem roots_of_unity_real_fifth_power :
  ∃ (S : Finset ℂ), 
    (S.card = 30) ∧ 
    (∀ z ∈ S, z^30 = 1) ∧
    (∃ (T : Finset ℂ), 
      (T ⊆ S) ∧ 
      (T.card = 10) ∧ 
      (∀ z ∈ T, ∃ (r : ℝ), z^5 = r) ∧
      (∀ z ∈ S \ T, ¬∃ (r : ℝ), z^5 = r)) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_unity_real_fifth_power_l2647_264766


namespace NUMINAMATH_CALUDE_marble_problem_l2647_264782

theorem marble_problem (b : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ)
  (h1 : angela = b)
  (h2 : brian = 3 * b)
  (h3 : caden = 4 * brian)
  (h4 : daryl = 6 * caden)
  (h5 : angela + brian + caden + daryl = 312) :
  b = 39 / 11 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l2647_264782


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2647_264725

theorem greatest_value_quadratic_inequality :
  ∃ (a : ℝ), a^2 - 10*a + 21 ≤ 0 ∧ ∀ (x : ℝ), x^2 - 10*x + 21 ≤ 0 → x ≤ a :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2647_264725


namespace NUMINAMATH_CALUDE_carpenter_tables_total_l2647_264734

/-- The number of tables made this month -/
def tables_this_month : ℕ := 10

/-- The difference in tables made between this month and last month -/
def difference : ℕ := 3

/-- The number of tables made last month -/
def tables_last_month : ℕ := tables_this_month - difference

/-- The total number of tables made over two months -/
def total_tables : ℕ := tables_this_month + tables_last_month

theorem carpenter_tables_total :
  total_tables = 17 := by sorry

end NUMINAMATH_CALUDE_carpenter_tables_total_l2647_264734


namespace NUMINAMATH_CALUDE_garden_perimeter_l2647_264753

-- Define the garden shape
structure Garden where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ

-- Define the conditions
def is_valid_garden (g : Garden) : Prop :=
  g.a + g.b + g.c = 3 ∧
  g.a ≥ 0 ∧ g.b ≥ 0 ∧ g.c ≥ 0 ∧ g.x ≥ 0

-- Calculate the perimeter
def perimeter (g : Garden) : ℝ :=
  3 + 5 + g.a + g.x + g.b + 4 + g.c + (4 + (5 - g.x))

-- Theorem statement
theorem garden_perimeter (g : Garden) (h : is_valid_garden g) : perimeter g = 24 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l2647_264753


namespace NUMINAMATH_CALUDE_unit_digit_product_l2647_264763

theorem unit_digit_product : ∃ n : ℕ, (5 + 1) * (5^3 + 1) * (5^6 + 1) * (5^12 + 1) ≡ 6 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_product_l2647_264763


namespace NUMINAMATH_CALUDE_equation_seven_solutions_l2647_264705

-- Define the equation
def equation (a x : ℝ) : Prop :=
  Real.sin (Real.sqrt (a^2 - x^2 - 2*x - 1)) = 0.5

-- Define the number of distinct solutions
def has_seven_distinct_solutions (a : ℝ) : Prop :=
  ∃ (s : Finset ℝ), s.card = 7 ∧ (∀ x ∈ s, equation a x) ∧
    (∀ x : ℝ, equation a x → x ∈ s)

-- State the theorem
theorem equation_seven_solutions :
  ∀ a : ℝ, has_seven_distinct_solutions a ↔ a = 17 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_seven_solutions_l2647_264705


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2647_264759

theorem simplify_fraction_product : 8 * (15 / 9) * (-45 / 40) = -1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2647_264759


namespace NUMINAMATH_CALUDE_total_distance_traveled_l2647_264730

/-- Calculates the total distance traveled given walking and running speeds and durations, with a break. -/
theorem total_distance_traveled
  (total_time : ℝ)
  (walking_time : ℝ)
  (walking_speed : ℝ)
  (running_time : ℝ)
  (running_speed : ℝ)
  (break_time : ℝ)
  (h1 : total_time = 2)
  (h2 : walking_time = 1)
  (h3 : walking_speed = 3.5)
  (h4 : running_time = 0.75)
  (h5 : running_speed = 8)
  (h6 : break_time = 0.25)
  (h7 : total_time = walking_time + running_time + break_time) :
  walking_time * walking_speed + running_time * running_speed = 9.5 := by
  sorry


end NUMINAMATH_CALUDE_total_distance_traveled_l2647_264730


namespace NUMINAMATH_CALUDE_fraction_equality_implies_division_l2647_264735

theorem fraction_equality_implies_division (A B C : ℕ) : 
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C →
  1 - 1 / (6 + 1 / (6 + 1 / 6)) = 1 / (A + 1 / (B + 1 / C)) →
  (A + B) / C = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_division_l2647_264735


namespace NUMINAMATH_CALUDE_rose_difference_is_34_l2647_264750

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := 58

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The difference in the number of red roses between Mrs. Santiago and Mrs. Garrett -/
def rose_difference : ℕ := santiago_roses - garrett_roses

theorem rose_difference_is_34 : rose_difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_rose_difference_is_34_l2647_264750


namespace NUMINAMATH_CALUDE_colored_points_theorem_l2647_264717

theorem colored_points_theorem (r b g : ℕ) (d_rb d_rg d_bg : ℝ) : 
  r + b + g = 15 →
  (r : ℝ) * (b : ℝ) * d_rb = 51 →
  (r : ℝ) * (g : ℝ) * d_rg = 39 →
  (b : ℝ) * (g : ℝ) * d_bg = 1 →
  d_rb > 0 →
  d_rg > 0 →
  d_bg > 0 →
  ((r = 13 ∧ b = 1 ∧ g = 1) ∨ (r = 8 ∧ b = 4 ∧ g = 3)) := by
sorry

end NUMINAMATH_CALUDE_colored_points_theorem_l2647_264717


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2647_264714

theorem arithmetic_sequence_third_term (a x : ℝ) : 
  a + (a + 2*x) = 6 → a + x = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2647_264714


namespace NUMINAMATH_CALUDE_min_value_theorem_l2647_264761

theorem min_value_theorem (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x + y = 2 → 1 / (x - 1) + 1 / y ≥ 1 / (a - 1) + 1 / b) ∧
  1 / (a - 1) + 1 / b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2647_264761


namespace NUMINAMATH_CALUDE_particle_motion_l2647_264708

/-- A particle moves under the influence of gravity and an additional constant acceleration. -/
theorem particle_motion
  (V₀ g a t V S : ℝ)
  (hV : V = g * t + a * t + V₀)
  (hS : S = (1/2) * (g + a) * t^2 + V₀ * t) :
  t = (2 * S) / (V + V₀) :=
sorry

end NUMINAMATH_CALUDE_particle_motion_l2647_264708


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l2647_264710

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x - 2 else Real.log x / Real.log a

-- State the theorem
theorem monotonic_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l2647_264710


namespace NUMINAMATH_CALUDE_problem_solution_l2647_264772

theorem problem_solution : let M := 2021 / 3
                           let N := M / 4
                           let Y := M + N
                           Y = 843 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2647_264772


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l2647_264785

theorem angle_in_second_quadrant : 
  let θ := (29 * Real.pi) / 6
  0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l2647_264785


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l2647_264769

/-- Given two points on an inverse proportion function, prove that y₁ < y₂ -/
theorem inverse_proportion_y_relationship (k : ℝ) (y₁ y₂ : ℝ) :
  (2 : ℝ) > 0 ∧ (3 : ℝ) > 0 ∧
  y₁ = (-k^2 - 1) / 2 ∧
  y₂ = (-k^2 - 1) / 3 →
  y₁ < y₂ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l2647_264769
