import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l1772_177261

theorem problem_solution (m n : ℝ) 
  (h1 : m = (Real.sqrt (n^2 - 4) + Real.sqrt (4 - n^2) + 4) / (n - 2))
  (h2 : n^2 - 4 ≥ 0)
  (h3 : 4 - n^2 ≥ 0)
  (h4 : n ≠ 2) :
  |m - 2*n| + Real.sqrt (8*m*n) = 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1772_177261


namespace NUMINAMATH_CALUDE_quadratic_roots_at_minimum_l1772_177264

/-- Given a quadratic function y = ax² + bx + c with a ≠ 0 and its lowest point at (1, -1),
    the roots of ax² + bx + c = -1 are both equal to 1. -/
theorem quadratic_roots_at_minimum (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≥ a * 1^2 + b * 1 + c) →
  (a * 1^2 + b * 1 + c = -1) →
  (∀ x, a * x^2 + b * x + c = -1 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_at_minimum_l1772_177264


namespace NUMINAMATH_CALUDE_jelly_bean_match_probability_l1772_177200

/-- Represents the distribution of jelly beans for a person -/
structure JellyBeanDistribution where
  green : ℕ
  blue : ℕ
  red : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeanDistribution.total (d : JellyBeanDistribution) : ℕ :=
  d.green + d.blue + d.red

/-- Lila's jelly bean distribution -/
def lila_beans : JellyBeanDistribution :=
  { green := 1, blue := 1, red := 1 }

/-- Max's jelly bean distribution -/
def max_beans : JellyBeanDistribution :=
  { green := 2, blue := 1, red := 3 }

/-- Calculates the probability of picking a specific color -/
def pick_probability (d : JellyBeanDistribution) (color : ℕ) : ℚ :=
  color / d.total

/-- Calculates the probability of both people picking the same color -/
def match_probability (d1 d2 : JellyBeanDistribution) : ℚ :=
  pick_probability d1 d1.green * pick_probability d2 d2.green +
  pick_probability d1 d1.blue * pick_probability d2 d2.blue +
  pick_probability d1 d1.red * pick_probability d2 d2.red

theorem jelly_bean_match_probability :
  match_probability lila_beans max_beans = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_match_probability_l1772_177200


namespace NUMINAMATH_CALUDE_system_solution_l1772_177216

theorem system_solution : ∃ (x y : ℝ), x + y = 0 ∧ 2*x + 3*y = 3 ∧ x = -3 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1772_177216


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1772_177234

theorem quadratic_roots_sum_product (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 201 →
  p*q + r*s = -28743/12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1772_177234


namespace NUMINAMATH_CALUDE_sonya_falls_l1772_177220

/-- The number of times each person fell while ice skating --/
structure FallCounts where
  steven : ℕ
  stephanie : ℕ
  sonya : ℕ
  sam : ℕ
  sophie : ℕ

/-- The conditions given in the problem --/
def carnival_conditions (fc : FallCounts) : Prop :=
  fc.steven = 3 ∧
  fc.stephanie = fc.steven + 13 ∧
  fc.sonya = fc.stephanie / 2 - 2 ∧
  fc.sam = 1 ∧
  fc.sophie = fc.sam + 4

/-- Theorem stating that Sonya fell 6 times --/
theorem sonya_falls (fc : FallCounts) (h : carnival_conditions fc) : fc.sonya = 6 := by
  sorry

end NUMINAMATH_CALUDE_sonya_falls_l1772_177220


namespace NUMINAMATH_CALUDE_quadratic_properties_l1772_177243

def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_properties (b c : ℝ) 
  (h1 : f b c 1 = 0) 
  (h2 : f b c 3 = 0) : 
  (f b c (-1) = 8) ∧ 
  (∀ x ∈ Set.Icc 2 4, f b c x ≤ 3) ∧
  (∃ x ∈ Set.Icc 2 4, f b c x = 3) ∧
  (∀ x ∈ Set.Icc 2 4, -1 ≤ f b c x) ∧
  (∃ x ∈ Set.Icc 2 4, f b c x = -1) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f b c x ≤ f b c y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1772_177243


namespace NUMINAMATH_CALUDE_no_infinite_sequence_sqrt_difference_l1772_177227

theorem no_infinite_sequence_sqrt_difference :
  ¬ (∃ (x : ℕ → ℝ), (∀ n, 0 < x n) ∧ 
    (∀ n, x (n + 2) = Real.sqrt (x (n + 1)) - Real.sqrt (x n))) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_sqrt_difference_l1772_177227


namespace NUMINAMATH_CALUDE_rice_container_problem_l1772_177211

theorem rice_container_problem (total_weight : ℚ) (container_weight : ℕ) 
  (h1 : total_weight = 33 / 4)
  (h2 : container_weight = 33)
  (h3 : (1 : ℚ) = 16 / 16) : 
  (total_weight * 16) / container_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_rice_container_problem_l1772_177211


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1772_177254

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_sum : a 4 + a 8 = -2) : 
  a 4^2 + 2 * a 6^2 + a 6 * a 10 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1772_177254


namespace NUMINAMATH_CALUDE_bowling_team_weight_l1772_177255

theorem bowling_team_weight (original_avg : ℝ) : 
  (7 * original_avg + 110 + 60) / 9 = 92 → original_avg = 94 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_weight_l1772_177255


namespace NUMINAMATH_CALUDE_min_connections_for_six_towns_l1772_177225

/-- The number of towns -/
def num_towns : ℕ := 6

/-- The formula for the number of connections in an undirected graph without loops -/
def connections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 6 towns, the minimum number of connections needed is 15 -/
theorem min_connections_for_six_towns :
  connections num_towns = 15 := by sorry

end NUMINAMATH_CALUDE_min_connections_for_six_towns_l1772_177225


namespace NUMINAMATH_CALUDE_factorization_equivalence_l1772_177275

variable (a x y : ℝ)

theorem factorization_equivalence : 
  (2*a*x^2 - 8*a*x*y + 8*a*y^2 = 2*a*(x - 2*y)^2) ∧ 
  (6*x*y^2 - 9*x^2*y - y^3 = -y*(3*x - y)^2) := by sorry

end NUMINAMATH_CALUDE_factorization_equivalence_l1772_177275


namespace NUMINAMATH_CALUDE_dice_product_div_eight_prob_l1772_177270

/-- Represents a standard 6-sided die --/
def Die : Type := Fin 6

/-- The probability space of rolling 8 dice --/
def DiceRoll : Type := Fin 8 → Die

/-- A function that determines if a number is divisible by 8 --/
def divisible_by_eight (n : ℕ) : Prop := n % 8 = 0

/-- The product of the numbers shown on the dice --/
def dice_product (roll : DiceRoll) : ℕ :=
  (List.range 8).foldl (λ acc i => acc * (roll i).val.succ) 1

/-- The event that the product of the dice roll is divisible by 8 --/
def event_divisible_by_eight (roll : DiceRoll) : Prop :=
  divisible_by_eight (dice_product roll)

/-- The probability measure on the dice roll space --/
axiom prob : (DiceRoll → Prop) → ℚ

/-- The probability of the event is well-defined --/
axiom prob_well_defined : ∀ (E : DiceRoll → Prop), 0 ≤ prob E ∧ prob E ≤ 1

theorem dice_product_div_eight_prob :
  prob event_divisible_by_eight = 199 / 256 := by
  sorry

end NUMINAMATH_CALUDE_dice_product_div_eight_prob_l1772_177270


namespace NUMINAMATH_CALUDE_arithmetic_sequence_zero_term_l1772_177245

/-- An arithmetic sequence with common difference d ≠ 0 -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h : d ≠ 0  -- d is non-zero
  seq : ∀ n : ℕ, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- The theorem statement -/
theorem arithmetic_sequence_zero_term
  (seq : ArithmeticSequence)
  (h : seq.a 3 + seq.a 9 = seq.a 10 - seq.a 8) :
  ∃! n : ℕ, seq.a n = 0 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_zero_term_l1772_177245


namespace NUMINAMATH_CALUDE_polynomial_value_equivalence_l1772_177271

theorem polynomial_value_equivalence (x : ℝ) : x^2 - (5/2)*x = 6 → 2*x^2 - 5*x + 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_equivalence_l1772_177271


namespace NUMINAMATH_CALUDE_school_club_members_l1772_177242

theorem school_club_members :
  ∃! n : ℕ,
    200 ≤ n ∧ n ≤ 300 ∧
    n % 6 = 3 ∧
    n % 8 = 5 ∧
    n % 9 = 7 ∧
    n = 269 := by sorry

end NUMINAMATH_CALUDE_school_club_members_l1772_177242


namespace NUMINAMATH_CALUDE_spice_difference_l1772_177288

def cinnamon : ℝ := 0.6666666666666666
def nutmeg : ℝ := 0.5
def ginger : ℝ := 0.4444444444444444

def total_difference : ℝ := |cinnamon - nutmeg| + |nutmeg - ginger| + |cinnamon - ginger|

theorem spice_difference : total_difference = 0.4444444444444444 := by sorry

end NUMINAMATH_CALUDE_spice_difference_l1772_177288


namespace NUMINAMATH_CALUDE_wire_radius_from_sphere_l1772_177204

/-- The radius of a wire's cross section when a sphere is melted and drawn into a wire -/
theorem wire_radius_from_sphere (r_sphere : ℝ) (l_wire : ℝ) (r_wire : ℝ) : 
  r_sphere = 12 →
  l_wire = 144 →
  (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_wire^2 * l_wire →
  r_wire = 4 := by
  sorry

#check wire_radius_from_sphere

end NUMINAMATH_CALUDE_wire_radius_from_sphere_l1772_177204


namespace NUMINAMATH_CALUDE_comparison_theorem_l1772_177256

theorem comparison_theorem :
  (-3 / 4 : ℚ) < -2 / 3 ∧ (3 : ℤ) > -|4| := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l1772_177256


namespace NUMINAMATH_CALUDE_enlarged_lawn_area_l1772_177223

theorem enlarged_lawn_area (initial_width : ℝ) (initial_area : ℝ) (new_width : ℝ) :
  initial_width = 8 →
  initial_area = 640 →
  new_width = 16 →
  let length : ℝ := initial_area / initial_width
  let new_area : ℝ := length * new_width
  new_area = 1280 := by
  sorry

end NUMINAMATH_CALUDE_enlarged_lawn_area_l1772_177223


namespace NUMINAMATH_CALUDE_external_angle_bisectors_collinear_l1772_177284

-- Define the basic structures
structure Point := (x : ℝ) (y : ℝ)
structure Line := (a : ℝ) (b : ℝ) (c : ℝ) -- ax + by + c = 0

-- Define the quadrilateral
structure Quadrilateral :=
  (A B C D : Point)
  (is_convex : Bool)

-- Define the intersection points of side extensions
def extension_intersections (q : Quadrilateral) : Point × Point := sorry

-- Define the external angle bisector
def external_angle_bisector (p1 p2 p3 : Point) : Line := sorry

-- Define collinearity
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Main theorem
theorem external_angle_bisectors_collinear (q : Quadrilateral) :
  let (P, Q) := extension_intersections q
  let AC_bisector := external_angle_bisector q.A q.C P
  let BD_bisector := external_angle_bisector q.B q.D Q
  let PQ_bisector := external_angle_bisector P Q q.A
  let I1 := sorry -- Intersection of AC_bisector and BD_bisector
  let I2 := sorry -- Intersection of BD_bisector and PQ_bisector
  let I3 := sorry -- Intersection of PQ_bisector and AC_bisector
  collinear I1 I2 I3 := by sorry

end NUMINAMATH_CALUDE_external_angle_bisectors_collinear_l1772_177284


namespace NUMINAMATH_CALUDE_exact_blue_marbles_probability_l1772_177276

def total_marbles : ℕ := 20
def blue_marbles : ℕ := 12
def red_marbles : ℕ := 8
def num_draws : ℕ := 8
def num_blue_draws : ℕ := 5

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem exact_blue_marbles_probability :
  (Nat.choose num_draws num_blue_draws : ℚ) *
  (prob_blue ^ num_blue_draws) *
  (prob_red ^ (num_draws - num_blue_draws)) =
  108864 / 390625 :=
by sorry

end NUMINAMATH_CALUDE_exact_blue_marbles_probability_l1772_177276


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1772_177214

theorem absolute_value_equation_solution :
  ∃ x : ℚ, |6 * x - 8| = 0 ∧ x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1772_177214


namespace NUMINAMATH_CALUDE_max_a_for_decreasing_cosine_minus_sine_l1772_177277

theorem max_a_for_decreasing_cosine_minus_sine :
  let f : ℝ → ℝ := λ x ↦ Real.cos x - Real.sin x
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ a → f y < f x) →
    a ≤ 3 * Real.pi / 4 ∧ 
    ∃ b : ℝ, b > 3 * Real.pi / 4 ∧ ¬(∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ b → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_decreasing_cosine_minus_sine_l1772_177277


namespace NUMINAMATH_CALUDE_sequence_termination_l1772_177230

def b : ℕ → ℚ
  | 0 => 41
  | 1 => 68
  | (k+2) => b k - 5 / b (k+1)

theorem sequence_termination :
  ∃ n : ℕ, n > 0 ∧ b n = 0 ∧ ∀ k < n, b k ≠ 0 ∧ b (k+1) = b (k-1) - 5 / b k :=
by
  use 559
  sorry

#eval b 559

end NUMINAMATH_CALUDE_sequence_termination_l1772_177230


namespace NUMINAMATH_CALUDE_cube_cutting_problem_l1772_177297

theorem cube_cutting_problem :
  ∃! n : ℕ, ∃ s : ℕ, n > s ∧ n^3 - s^3 = 152 :=
by sorry

end NUMINAMATH_CALUDE_cube_cutting_problem_l1772_177297


namespace NUMINAMATH_CALUDE_admission_cutoff_score_l1772_177269

theorem admission_cutoff_score (
  admitted_fraction : Real)
  (admitted_avg_diff : Real)
  (not_admitted_avg_diff : Real)
  (overall_avg : Real)
  (h1 : admitted_fraction = 2 / 5)
  (h2 : admitted_avg_diff = 15)
  (h3 : not_admitted_avg_diff = -20)
  (h4 : overall_avg = 90) :
  let cutoff_score := 
    (overall_avg - admitted_fraction * admitted_avg_diff - (1 - admitted_fraction) * not_admitted_avg_diff) /
    (admitted_fraction + (1 - admitted_fraction))
  cutoff_score = 96 := by
  sorry

end NUMINAMATH_CALUDE_admission_cutoff_score_l1772_177269


namespace NUMINAMATH_CALUDE_sum_10_with_7_dice_l1772_177267

/-- The number of ways to roll a sum of 10 with 7 fair 6-sided dice -/
def ways_to_roll_10_with_7_dice : ℕ :=
  Nat.choose 9 6

/-- The probability of rolling a sum of 10 with 7 fair 6-sided dice -/
def prob_sum_10_7_dice : ℚ :=
  ways_to_roll_10_with_7_dice / (6^7 : ℚ)

theorem sum_10_with_7_dice :
  ways_to_roll_10_with_7_dice = 84 ∧
  prob_sum_10_7_dice = 84 / (6^7 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sum_10_with_7_dice_l1772_177267


namespace NUMINAMATH_CALUDE_caleb_hamburger_cost_l1772_177292

/-- Represents the total cost of Caleb's hamburger purchase --/
def total_cost (single_price : ℚ) (double_price : ℚ) (total_burgers : ℕ) (double_burgers : ℕ) : ℚ :=
  single_price * (total_burgers - double_burgers) + double_price * double_burgers

/-- Theorem stating that Caleb's total spending on hamburgers is $74.50 --/
theorem caleb_hamburger_cost : 
  total_cost 1 (3/2) 50 49 = 149/2 := by
  sorry

end NUMINAMATH_CALUDE_caleb_hamburger_cost_l1772_177292


namespace NUMINAMATH_CALUDE_ellipse_tangent_circle_radius_l1772_177262

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    and a circle centered at one of its foci and tangent to the ellipse,
    prove that the radius of the circle is √((a^2 - b^2)/2). -/
theorem ellipse_tangent_circle_radius 
  (a b : ℝ) 
  (h_a : a = 6) 
  (h_b : b = 3) : 
  let c := Real.sqrt (a^2 - b^2)
  let r := Real.sqrt ((a^2 - b^2)/2)
  ∀ x y : ℝ,
  (x^2 / a^2 + y^2 / b^2 = 1) →
  ((x - c)^2 + y^2 = r^2) →
  r = Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_tangent_circle_radius_l1772_177262


namespace NUMINAMATH_CALUDE_stratified_sampling_count_l1772_177258

-- Define the total number of students and their gender distribution
def total_students : ℕ := 60
def female_students : ℕ := 24
def male_students : ℕ := 36

-- Define the number of students to be selected
def selected_students : ℕ := 20

-- Define the number of female and male students to be selected
def selected_female : ℕ := 8
def selected_male : ℕ := 12

-- Theorem statement
theorem stratified_sampling_count :
  (Nat.choose female_students selected_female) * (Nat.choose male_students selected_male) =
  (Nat.choose female_students selected_female) * (Nat.choose male_students selected_male) := by
  sorry

-- Ensure the conditions are met
axiom total_students_sum : female_students + male_students = total_students
axiom selected_students_sum : selected_female + selected_male = selected_students

end NUMINAMATH_CALUDE_stratified_sampling_count_l1772_177258


namespace NUMINAMATH_CALUDE_problem_solution_l1772_177281

theorem problem_solution (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1772_177281


namespace NUMINAMATH_CALUDE_column_of_1985_l1772_177205

/-- The column number (1-indexed) in which a given odd positive integer appears in the arrangement -/
def columnNumber (n : ℕ) : ℕ :=
  (n % 16 + 15) % 16 / 2 + 1

theorem column_of_1985 : columnNumber 1985 = 1 := by sorry

end NUMINAMATH_CALUDE_column_of_1985_l1772_177205


namespace NUMINAMATH_CALUDE_triangle_inequality_l1772_177294

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_range : 0 < α ∧ α < π
  cosine_rule : 2 * b * c * Real.cos α = b^2 + c^2 - a^2

-- State the theorem
theorem triangle_inequality (t : Triangle) : 
  (2 * t.b * t.c * Real.cos t.α) / (t.b + t.c) < t.b + t.c - t.a ∧ 
  t.b + t.c - t.a < (2 * t.b * t.c) / t.a :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1772_177294


namespace NUMINAMATH_CALUDE_gold_coin_problem_l1772_177217

theorem gold_coin_problem (c : ℕ+) (h1 : 8 * (c - 1) = 5 * c + 4) : 
  ∃ n : ℕ, n = 24 ∧ 8 * (c - 1) = n ∧ 5 * c + 4 = n :=
sorry

end NUMINAMATH_CALUDE_gold_coin_problem_l1772_177217


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l1772_177236

theorem imaginary_unit_sum (i : ℂ) (h : i^2 = -1) : 
  1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l1772_177236


namespace NUMINAMATH_CALUDE_alloy_ratio_proof_l1772_177285

/-- Proves that the ratio of lead to tin in alloy A is 2:3 given the specified conditions -/
theorem alloy_ratio_proof (alloy_A_weight : ℝ) (alloy_B_weight : ℝ) 
  (tin_copper_ratio_B : ℚ) (total_tin_new_alloy : ℝ) 
  (h1 : alloy_A_weight = 120)
  (h2 : alloy_B_weight = 180)
  (h3 : tin_copper_ratio_B = 3/5)
  (h4 : total_tin_new_alloy = 139.5) :
  ∃ (lead_A tin_A : ℝ), 
    lead_A + tin_A = alloy_A_weight ∧ 
    lead_A / tin_A = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_alloy_ratio_proof_l1772_177285


namespace NUMINAMATH_CALUDE_part_one_part_two_l1772_177224

-- Part 1
theorem part_one (x y : ℝ) : 
  y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 3 →
  x = 2 →
  x - y = -1 := by sorry

-- Part 2
theorem part_two (x : ℝ) :
  x = Real.sqrt 2 →
  (x / (x - 2)) / (2 + x - 4 / (2 - x)) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1772_177224


namespace NUMINAMATH_CALUDE_min_data_for_plan_y_effectiveness_l1772_177222

/-- Represents the cost in cents for Plan X given the data usage in MB -/
def cost_plan_x (data : ℕ) : ℕ := 20 * data

/-- Represents the cost in cents for Plan Y given the data usage in MB -/
def cost_plan_y (data : ℕ) : ℕ := 1500 + 10 * data

/-- Proves that 151 MB is the minimum amount of data that makes Plan Y more cost-effective -/
theorem min_data_for_plan_y_effectiveness : 
  ∀ d : ℕ, d ≥ 151 ↔ cost_plan_y d < cost_plan_x d :=
by sorry

end NUMINAMATH_CALUDE_min_data_for_plan_y_effectiveness_l1772_177222


namespace NUMINAMATH_CALUDE_discount_difference_l1772_177253

theorem discount_difference : 
  let original_bill : ℝ := 10000
  let single_discount_rate : ℝ := 0.4
  let first_successive_discount_rate : ℝ := 0.36
  let second_successive_discount_rate : ℝ := 0.04
  let single_discounted_amount : ℝ := original_bill * (1 - single_discount_rate)
  let successive_discounted_amount : ℝ := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)
  successive_discounted_amount - single_discounted_amount = 144 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l1772_177253


namespace NUMINAMATH_CALUDE_canned_food_bins_l1772_177208

theorem canned_food_bins (soup_bins vegetables_bins pasta_bins : Real) 
  (h1 : soup_bins = 0.12)
  (h2 : vegetables_bins = 0.12)
  (h3 : pasta_bins = 0.5) :
  soup_bins + vegetables_bins + pasta_bins = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_canned_food_bins_l1772_177208


namespace NUMINAMATH_CALUDE_product_b_sample_size_l1772_177259

/-- Calculates the number of items drawn from a specific product
    using stratified sampling method. -/
def stratifiedSample (totalItems : ℕ) (ratio : List ℕ) (sampleSize : ℕ) (productIndex : ℕ) : ℕ :=
  (sampleSize * (ratio.get! productIndex)) / (ratio.sum)

/-- Theorem: Given 1200 total items with ratio 3:4:5 for products A, B, and C,
    when drawing 60 items using stratified sampling,
    the number of items drawn from product B is 20. -/
theorem product_b_sample_size :
  let totalItems : ℕ := 1200
  let ratio : List ℕ := [3, 4, 5]
  let sampleSize : ℕ := 60
  let productBIndex : ℕ := 1
  stratifiedSample totalItems ratio sampleSize productBIndex = 20 := by
  sorry

end NUMINAMATH_CALUDE_product_b_sample_size_l1772_177259


namespace NUMINAMATH_CALUDE_min_value_of_f_l1772_177299

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 3) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y ∧ f a x = -29 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1772_177299


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_sally_pokemon_cards_proof_l1772_177266

theorem sally_pokemon_cards : ℕ → Prop :=
  fun x =>
    let sally_initial : ℕ := 27
    let dan_cards : ℕ := 41
    let difference : ℕ := 6
    sally_initial + x = dan_cards + difference →
    x = 20

-- The proof is omitted
theorem sally_pokemon_cards_proof : ∃ x, sally_pokemon_cards x :=
  sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_sally_pokemon_cards_proof_l1772_177266


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1772_177232

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) (h : x > 0) :
  (1.12 * L) * ((1 - 0.01 * x) * W) = 1.064 * (L * W) →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1772_177232


namespace NUMINAMATH_CALUDE_expression_evaluation_l1772_177268

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = a + 8)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 23 / 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1772_177268


namespace NUMINAMATH_CALUDE_factorial_ratio_50_48_l1772_177296

theorem factorial_ratio_50_48 : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_50_48_l1772_177296


namespace NUMINAMATH_CALUDE_spring_membership_decrease_l1772_177252

theorem spring_membership_decrease
  (fall_increase : Real)
  (total_decrease : Real)
  (h1 : fall_increase = 0.06)
  (h2 : total_decrease = 0.1414) :
  let fall_membership := 1 + fall_increase
  let spring_membership := 1 - total_decrease
  (fall_membership - spring_membership) / fall_membership = 0.19 := by
sorry

end NUMINAMATH_CALUDE_spring_membership_decrease_l1772_177252


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1772_177260

theorem arithmetic_geometric_mean_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c) / 3)^2 ≥ (a*b + b*c + c*a) / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1772_177260


namespace NUMINAMATH_CALUDE_bottles_maria_drank_l1772_177291

theorem bottles_maria_drank (initial bottles_bought bottles_remaining : ℕ) : 
  initial = 14 → bottles_bought = 45 → bottles_remaining = 51 → 
  initial + bottles_bought - bottles_remaining = 8 := by
sorry

end NUMINAMATH_CALUDE_bottles_maria_drank_l1772_177291


namespace NUMINAMATH_CALUDE_probability_of_three_hits_is_one_fifth_l1772_177231

/-- A set of random numbers -/
structure RandomSet :=
  (numbers : List Nat)

/-- Predicate to check if a number is a hit (1 to 6) -/
def isHit (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 6

/-- Count the number of hits in a set -/
def countHits (s : RandomSet) : Nat :=
  s.numbers.filter isHit |>.length

/-- The experiment data -/
def experimentData : List RandomSet := sorry

/-- The number of sets with exactly three hits -/
def setsWithThreeHits : Nat :=
  experimentData.filter (fun s => countHits s = 3) |>.length

/-- Total number of sets in the experiment -/
def totalSets : Nat := 20

theorem probability_of_three_hits_is_one_fifth :
  (setsWithThreeHits : ℚ) / totalSets = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_three_hits_is_one_fifth_l1772_177231


namespace NUMINAMATH_CALUDE_train_delay_l1772_177265

/-- Calculates the time difference in minutes for a train traveling a given distance at two different speeds -/
theorem train_delay (distance : ℝ) (speed1 speed2 : ℝ) :
  distance > 0 ∧ speed1 > 0 ∧ speed2 > 0 ∧ speed1 > speed2 →
  (distance / speed2 - distance / speed1) * 60 = 15 ∧
  distance = 70 ∧ speed1 = 40 ∧ speed2 = 35 :=
by sorry

end NUMINAMATH_CALUDE_train_delay_l1772_177265


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1772_177203

theorem simplify_trig_expression :
  (Real.sin (10 * π / 180) + Real.sin (20 * π / 180)) /
  (Real.cos (10 * π / 180) + Real.cos (20 * π / 180)) =
  Real.tan (15 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1772_177203


namespace NUMINAMATH_CALUDE_fireworks_cost_and_remaining_l1772_177278

def small_firework_cost : ℕ := 12
def large_firework_cost : ℕ := 25

def henry_small : ℕ := 3
def henry_large : ℕ := 2
def friend_small : ℕ := 4
def friend_large : ℕ := 1

def saved_fireworks : ℕ := 6
def used_saved_fireworks : ℕ := 3

theorem fireworks_cost_and_remaining :
  (let total_cost := (henry_small + friend_small) * small_firework_cost +
                     (henry_large + friend_large) * large_firework_cost
   let remaining_fireworks := henry_small + henry_large + friend_small + friend_large +
                              (saved_fireworks - used_saved_fireworks)
   (total_cost = 159) ∧ (remaining_fireworks = 13)) := by
  sorry

end NUMINAMATH_CALUDE_fireworks_cost_and_remaining_l1772_177278


namespace NUMINAMATH_CALUDE_parabola_single_intersection_l1772_177280

/-- A parabola with equation y = x^2 - x + k has only one intersection point with the x-axis if and only if k = 1/4 -/
theorem parabola_single_intersection (k : ℝ) : 
  (∃! x, x^2 - x + k = 0) ↔ k = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_single_intersection_l1772_177280


namespace NUMINAMATH_CALUDE_two_digit_number_representation_l1772_177282

/-- Represents a two-digit number -/
def two_digit_number (x y : ℕ) : ℕ := 10 * x + y

/-- The tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- The units digit of a two-digit number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_number_representation (x y : ℕ) (h1 : x < 10) (h2 : y < 10) :
  two_digit_number x y = 10 * x + y :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_representation_l1772_177282


namespace NUMINAMATH_CALUDE_simplify_expression_l1772_177235

theorem simplify_expression (n : ℕ) : (3^(n+4) - 3*(3^n)) / (3*(3^(n+3))) = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1772_177235


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l1772_177244

/-- Given a pet store with bird cages, prove the number of parakeets in each cage -/
theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 9)
  (h2 : parrots_per_cage = 2)
  (h3 : total_birds = 72) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 6 := by
  sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l1772_177244


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1772_177274

theorem unique_integer_solution : 
  ∀ x y : ℤ, x^2 - 2*x*y + 2*y^2 - 4*y^3 = 0 → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1772_177274


namespace NUMINAMATH_CALUDE_magnitude_of_4_minus_15i_l1772_177241

-- Define the complex number
def z : ℂ := 4 - 15 * Complex.I

-- State the theorem
theorem magnitude_of_4_minus_15i : Complex.abs z = Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_4_minus_15i_l1772_177241


namespace NUMINAMATH_CALUDE_least_addition_for_multiple_of_five_l1772_177238

theorem least_addition_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (879 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (879 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_multiple_of_five_l1772_177238


namespace NUMINAMATH_CALUDE_vector_orthogonality_l1772_177251

def a : ℝ × ℝ := (3, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)
def c : ℝ × ℝ := (0, 2)

theorem vector_orthogonality (x : ℝ) :
  a • (b x - c) = 0 → x = 4/3 := by sorry

end NUMINAMATH_CALUDE_vector_orthogonality_l1772_177251


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l1772_177218

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def digit_at (n : ℕ) (place : ℕ) : ℕ :=
  (n / (10 ^ place)) % 10

theorem four_digit_number_theorem (n : ℕ) :
  is_valid_number n ∧ 
  (digit_at n 0 + digit_at n 1 - 4 * digit_at n 3 = 1) ∧
  (digit_at n 0 + 10 * digit_at n 1 - 2 * digit_at n 2 = 14) →
  n = 1014 ∨ n = 2218 ∨ n = 1932 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l1772_177218


namespace NUMINAMATH_CALUDE_white_white_pairs_coincide_l1772_177249

/-- Represents a half of the geometric figure -/
structure Half where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the coinciding pairs when the halves are folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ

/-- The main theorem statement -/
theorem white_white_pairs_coincide 
  (half : Half) 
  (coinciding : CoincidingPairs) 
  (h1 : half.red = 4) 
  (h2 : half.blue = 7) 
  (h3 : half.white = 10) 
  (h4 : coinciding.red_red = 3) 
  (h5 : coinciding.blue_blue = 4) 
  (h6 : coinciding.red_white = 3) : 
  ∃ (white_white : ℕ), white_white = 7 ∧ 
    white_white = half.white - coinciding.red_white := by
  sorry

end NUMINAMATH_CALUDE_white_white_pairs_coincide_l1772_177249


namespace NUMINAMATH_CALUDE_expression_evaluation_l1772_177250

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -1
  (2*a - b)^2 + (a - b)*(a + b) - 5*a*(a - 2*b) = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1772_177250


namespace NUMINAMATH_CALUDE_circle_area_isosceles_triangle_l1772_177272

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_isosceles_triangle (a b c : ℝ) (h1 : a = 4) (h2 : b = 4) (h3 : c = 3) :
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = 16 * π :=
by sorry


end NUMINAMATH_CALUDE_circle_area_isosceles_triangle_l1772_177272


namespace NUMINAMATH_CALUDE_sin_2x_value_l1772_177229

theorem sin_2x_value (x : ℝ) (h : Real.tan (π - x) = 3) : Real.sin (2 * x) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_value_l1772_177229


namespace NUMINAMATH_CALUDE_sqrt_expressions_equality_l1772_177247

theorem sqrt_expressions_equality : 
  (Real.sqrt 8 - Real.sqrt (1/2) + Real.sqrt 18 = (9 * Real.sqrt 2) / 2) ∧ 
  ((Real.sqrt 2 + Real.sqrt 3)^2 - Real.sqrt 24 = 5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_equality_l1772_177247


namespace NUMINAMATH_CALUDE_jumping_contest_total_distance_l1772_177298

/-- Represents the distance jumped by an animal and the obstacle they cleared -/
structure JumpDistance where
  jump : ℕ
  obstacle : ℕ

/-- Calculates the total distance jumped including the obstacle -/
def totalDistance (jd : JumpDistance) : ℕ := jd.jump + jd.obstacle

theorem jumping_contest_total_distance 
  (grasshopper : JumpDistance)
  (frog : JumpDistance)
  (kangaroo : JumpDistance)
  (h1 : grasshopper.jump = 25 ∧ grasshopper.obstacle = 5)
  (h2 : frog.jump = grasshopper.jump + 15 ∧ frog.obstacle = 10)
  (h3 : kangaroo.jump = 2 * frog.jump ∧ kangaroo.obstacle = 15) :
  totalDistance grasshopper + totalDistance frog + totalDistance kangaroo = 175 := by
  sorry

#check jumping_contest_total_distance

end NUMINAMATH_CALUDE_jumping_contest_total_distance_l1772_177298


namespace NUMINAMATH_CALUDE_no_consistent_values_l1772_177287

theorem no_consistent_values : ¬∃ (A B C D : ℤ), 
  B = 59 ∧ 
  C = 27 ∧ 
  D = 31 ∧ 
  (4701 % A = 0) ∧ 
  A = B * C + D :=
sorry

end NUMINAMATH_CALUDE_no_consistent_values_l1772_177287


namespace NUMINAMATH_CALUDE_fraction_cube_two_thirds_l1772_177257

theorem fraction_cube_two_thirds : (2 / 3 : ℚ) ^ 3 = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cube_two_thirds_l1772_177257


namespace NUMINAMATH_CALUDE_factory_works_four_days_l1772_177295

/-- Represents a toy factory with weekly production and daily production rates. -/
structure ToyFactory where
  weekly_production : ℕ
  daily_production : ℕ

/-- Calculates the number of working days per week for a given toy factory. -/
def working_days (factory : ToyFactory) : ℕ :=
  factory.weekly_production / factory.daily_production

/-- Theorem: The toy factory works 4 days per week. -/
theorem factory_works_four_days :
  let factory : ToyFactory := { weekly_production := 6000, daily_production := 1500 }
  working_days factory = 4 := by
  sorry

#eval working_days { weekly_production := 6000, daily_production := 1500 }

end NUMINAMATH_CALUDE_factory_works_four_days_l1772_177295


namespace NUMINAMATH_CALUDE_power_of_power_l1772_177289

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l1772_177289


namespace NUMINAMATH_CALUDE_profit_percentage_l1772_177273

theorem profit_percentage (selling_price : ℝ) (cost_price : ℝ) (h : cost_price = 0.89 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (1 / 0.89 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l1772_177273


namespace NUMINAMATH_CALUDE_pizza_order_count_l1772_177263

theorem pizza_order_count (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 8) (h2 : total_slices = 168) :
  total_slices / slices_per_pizza = 21 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_count_l1772_177263


namespace NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l1772_177290

theorem shaded_area_of_concentric_circles 
  (outer_circle_area : ℝ)
  (inner_circle_radius : ℝ)
  (h1 : outer_circle_area = 81 * Real.pi)
  (h2 : inner_circle_radius = 4.5)
  : ∃ (shaded_area : ℝ), shaded_area = 54 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l1772_177290


namespace NUMINAMATH_CALUDE_mod_23_equivalence_l1772_177283

theorem mod_23_equivalence :
  ∃! n : ℕ, 0 ≤ n ∧ n < 23 ∧ 123456 % 23 = n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_mod_23_equivalence_l1772_177283


namespace NUMINAMATH_CALUDE_characterization_of_solutions_l1772_177248

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) * (f x - f y) = (x - y) * f x * f y

/-- The main theorem stating the form of functions satisfying the equation -/
theorem characterization_of_solutions :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ C : ℝ, C ≠ 0 ∧ ∀ x : ℝ, f x = C * x :=
by sorry

end NUMINAMATH_CALUDE_characterization_of_solutions_l1772_177248


namespace NUMINAMATH_CALUDE_max_b_value_l1772_177212

/-- The maximum value of b given the conditions -/
theorem max_b_value (a : ℝ) (f g : ℝ → ℝ) (h₁ : a > 0)
  (h₂ : ∀ x, f x = 6 * a^2 * Real.log x)
  (h₃ : ∀ x, g x = x^2 - 4*a*x - b)
  (h₄ : ∃ x₀, x₀ > 0 ∧ (deriv f x₀ = deriv g x₀) ∧ (f x₀ = g x₀)) :
  (∃ b : ℝ, ∀ b' : ℝ, b' ≤ b) ∧ (∀ b : ℝ, (∃ b' : ℝ, ∀ b'' : ℝ, b'' ≤ b') → b ≤ 1 / (3 * Real.exp 2)) :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l1772_177212


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1772_177207

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence : 
  let a₁ := (1 : ℚ) / 2
  let a₂ := (3 : ℚ) / 4
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 10 = (11 : ℚ) / 4 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1772_177207


namespace NUMINAMATH_CALUDE_estimated_y_at_25_l1772_177213

/-- Linear regression function -/
def linear_regression (x : ℝ) : ℝ := 0.5 * x - 0.81

/-- Theorem: The estimated value of y is 11.69 when x = 25 -/
theorem estimated_y_at_25 : linear_regression 25 = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_estimated_y_at_25_l1772_177213


namespace NUMINAMATH_CALUDE_wednesday_temperature_l1772_177226

/-- Given the high temperatures for three consecutive days (Monday, Tuesday, Wednesday),
    prove that Wednesday's temperature is 12°C under the given conditions. -/
theorem wednesday_temperature (M T W : ℤ) : 
  T = M + 4 →   -- Tuesday's temperature is 4°C warmer than Monday's
  W = M - 6 →   -- Wednesday's temperature is 6°C cooler than Monday's
  T = 22 →      -- Tuesday's temperature is 22°C
  W = 12 :=     -- Prove: Wednesday's temperature is 12°C
by sorry

end NUMINAMATH_CALUDE_wednesday_temperature_l1772_177226


namespace NUMINAMATH_CALUDE_power_function_sum_l1772_177233

/-- A function f is a power function if it has the form f(x) = cx^n + k where c ≠ 0 and n is a real number -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (c k n : ℝ), c ≠ 0 ∧ ∀ x, f x = c * x^n + k

/-- Given that f(x) = ax^(2a+1) - b + 1 is a power function, prove that a + b = 2 -/
theorem power_function_sum (a b : ℝ) :
  IsPowerFunction (fun x ↦ a * x^(2*a+1) - b + 1) → a + b = 2 := by
  sorry


end NUMINAMATH_CALUDE_power_function_sum_l1772_177233


namespace NUMINAMATH_CALUDE_price_increase_for_constant_revenue_l1772_177219

/-- Proves that a 25% price increase is necessary to maintain constant revenue when demand decreases by 20% --/
theorem price_increase_for_constant_revenue 
  (original_price original_demand : ℝ) 
  (new_demand : ℝ) 
  (h_demand_decrease : new_demand = 0.8 * original_demand) 
  (h_revenue_constant : original_price * original_demand = (original_price * (1 + 0.25)) * new_demand) :
  (original_price * (1 + 0.25) - original_price) / original_price = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_price_increase_for_constant_revenue_l1772_177219


namespace NUMINAMATH_CALUDE_chicken_farm_growth_l1772_177240

theorem chicken_farm_growth (initial_chickens : ℕ) (annual_increase : ℕ) (years : ℕ) 
  (h1 : initial_chickens = 550)
  (h2 : annual_increase = 150)
  (h3 : years = 9) :
  initial_chickens + years * annual_increase = 1900 :=
by sorry

end NUMINAMATH_CALUDE_chicken_farm_growth_l1772_177240


namespace NUMINAMATH_CALUDE_new_players_count_l1772_177221

theorem new_players_count (returning_players : ℕ) (groups : ℕ) (players_per_group : ℕ) :
  returning_players = 6 →
  groups = 9 →
  players_per_group = 6 →
  groups * players_per_group - returning_players = 48 := by
sorry

end NUMINAMATH_CALUDE_new_players_count_l1772_177221


namespace NUMINAMATH_CALUDE_canteen_distance_l1772_177293

theorem canteen_distance (a b c x : ℝ) : 
  a = 400 → 
  c = 600 → 
  a^2 + b^2 = c^2 → 
  x^2 = a^2 + (b - x)^2 → 
  x = 410 := by
sorry

end NUMINAMATH_CALUDE_canteen_distance_l1772_177293


namespace NUMINAMATH_CALUDE_real_solution_implies_a_eq_one_no_purely_imaginary_roots_l1772_177279

variable (a : ℝ)

/-- The complex polynomial z^2 - (a+i)z - (i+2) = 0 -/
def f (z : ℂ) : ℂ := z^2 - (a + Complex.I) * z - (Complex.I + 2)

theorem real_solution_implies_a_eq_one :
  (∃ x : ℝ, f a x = 0) → a = 1 := by sorry

theorem no_purely_imaginary_roots :
  ¬∃ y : ℝ, y ≠ 0 ∧ f a (Complex.I * y) = 0 := by sorry

end NUMINAMATH_CALUDE_real_solution_implies_a_eq_one_no_purely_imaginary_roots_l1772_177279


namespace NUMINAMATH_CALUDE_smallest_K_is_correct_l1772_177239

/-- The smallest positive integer K such that 8000 × K is a perfect square -/
def smallest_K : ℕ := 5

/-- A predicate that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem smallest_K_is_correct :
  (∀ k : ℕ, k > 0 → k < smallest_K → ¬ is_perfect_square (8000 * k)) ∧
  is_perfect_square (8000 * smallest_K) := by
  sorry

#check smallest_K_is_correct

end NUMINAMATH_CALUDE_smallest_K_is_correct_l1772_177239


namespace NUMINAMATH_CALUDE_sin_translation_l1772_177215

open Real

theorem sin_translation (t S : ℝ) (k : ℤ) : 
  (1 = sin (2 * t)) → 
  (S > 0) → 
  (1 = sin (2 * (t + S) - π / 3)) → 
  (t = π / 4 + k * π ∧ S ≥ π / 6) :=
sorry

end NUMINAMATH_CALUDE_sin_translation_l1772_177215


namespace NUMINAMATH_CALUDE_team_combinations_l1772_177209

-- Define the number of people and team size
def total_people : ℕ := 7
def team_size : ℕ := 4

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem team_combinations : combination total_people team_size = 35 := by
  sorry

end NUMINAMATH_CALUDE_team_combinations_l1772_177209


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_planes_l1772_177202

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_to_parallel_planes 
  (α β : Plane) (l : Line) 
  (h1 : perp l α) (h2 : para α β) : 
  perp l β := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_planes_l1772_177202


namespace NUMINAMATH_CALUDE_differential_equation_solution_l1772_177201

open Real

theorem differential_equation_solution (x : ℝ) (C : ℝ) :
  let y : ℝ → ℝ := λ x => cos x * (sin x + C)
  (deriv y) x + y x * tan x = cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l1772_177201


namespace NUMINAMATH_CALUDE_project_completion_l1772_177228

theorem project_completion (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (1 : ℚ) / a + (1 : ℚ) / b * 4 = 1) : 
  a + b = 9 ∨ a + b = 10 := by
sorry

end NUMINAMATH_CALUDE_project_completion_l1772_177228


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l1772_177206

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : (22 : ℝ) / 100 * total_students = (students_more_than_100 : ℝ))
  (h2 : (75 : ℝ) / 100 * (students_on_trip : ℝ) = (students_not_more_than_100 : ℝ))
  (h3 : students_on_trip = students_more_than_100 + students_not_more_than_100) :
  (students_on_trip : ℝ) / total_students = 88 / 100 :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l1772_177206


namespace NUMINAMATH_CALUDE_triangle_theorem_l1772_177246

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that if b cos C + c cos B = 2a cos A and AB · AC = √3 in a triangle ABC,
    then the measure of angle A is π/3 and the area of the triangle is 3/2. -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * t.a * Real.cos t.A)
  (h2 : t.a * t.c * Real.cos t.A = Real.sqrt 3) : 
  t.A = π / 3 ∧ (1 / 2 * t.a * t.c * Real.sin t.A = 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1772_177246


namespace NUMINAMATH_CALUDE_bicycle_problem_solution_l1772_177210

/-- Represents the bicycle sales and inventory problem over three days -/
def bicycle_problem (S1 S2 S3 B1 B2 B3 P1 P2 P3 C1 C2 C3 : ℕ) : Prop :=
  let sale_profit1 := S1 * P1
  let sale_profit2 := S2 * P2
  let sale_profit3 := S3 * P3
  let repair_cost1 := B1 * C1
  let repair_cost2 := B2 * C2
  let repair_cost3 := B3 * C3
  let net_profit1 := sale_profit1 - repair_cost1
  let net_profit2 := sale_profit2 - repair_cost2
  let net_profit3 := sale_profit3 - repair_cost3
  let total_net_profit := net_profit1 + net_profit2 + net_profit3
  let net_increase := (B1 - S1) + (B2 - S2) + (B3 - S3)
  (S1 = 10 ∧ S2 = 12 ∧ S3 = 9 ∧
   B1 = 15 ∧ B2 = 8 ∧ B3 = 11 ∧
   P1 = 250 ∧ P2 = 275 ∧ P3 = 260 ∧
   C1 = 100 ∧ C2 = 110 ∧ C3 = 120) →
  (total_net_profit = 4440 ∧ net_increase = 3)

theorem bicycle_problem_solution :
  ∀ S1 S2 S3 B1 B2 B3 P1 P2 P3 C1 C2 C3 : ℕ,
  bicycle_problem S1 S2 S3 B1 B2 B3 P1 P2 P3 C1 C2 C3 :=
sorry

end NUMINAMATH_CALUDE_bicycle_problem_solution_l1772_177210


namespace NUMINAMATH_CALUDE_smallest_in_A_l1772_177286

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the set A
def A : Set ℕ := {n | 11 ∣ sumOfDigits n ∧ 11 ∣ sumOfDigits (n + 1)}

-- State the theorem
theorem smallest_in_A : 
  2899999 ∈ A ∧ ∀ m ∈ A, m < 2899999 → m = 2899999 := by sorry

end NUMINAMATH_CALUDE_smallest_in_A_l1772_177286


namespace NUMINAMATH_CALUDE_modifiedLucas_50th_term_mod_5_l1772_177237

def modifiedLucas : ℕ → ℤ
  | 0 => 2
  | 1 => 5
  | n + 2 => modifiedLucas n + modifiedLucas (n + 1)

theorem modifiedLucas_50th_term_mod_5 :
  modifiedLucas 49 % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_modifiedLucas_50th_term_mod_5_l1772_177237
