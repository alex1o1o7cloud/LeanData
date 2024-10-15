import Mathlib

namespace NUMINAMATH_CALUDE_folded_rectangle_perimeter_l691_69178

/-- A rectangle ABCD with a fold from A to A' on CD creating a crease EF -/
structure FoldedRectangle where
  -- Length of AE
  ae : ℝ
  -- Length of EB
  eb : ℝ
  -- Length of CF
  cf : ℝ

/-- The perimeter of the folded rectangle -/
def perimeter (r : FoldedRectangle) : ℝ :=
  2 * (r.ae + r.eb + r.cf + (r.ae + r.eb - r.cf))

/-- Theorem stating that the perimeter of the specific folded rectangle is 82 -/
theorem folded_rectangle_perimeter :
  let r : FoldedRectangle := { ae := 3, eb := 15, cf := 8 }
  perimeter r = 82 := by sorry

end NUMINAMATH_CALUDE_folded_rectangle_perimeter_l691_69178


namespace NUMINAMATH_CALUDE_max_bracelet_earnings_l691_69176

theorem max_bracelet_earnings :
  let total_bracelets : ℕ := 235
  let bracelets_per_bag : ℕ := 10
  let price_per_bag : ℕ := 3000
  let full_bags : ℕ := total_bracelets / bracelets_per_bag
  let max_earnings : ℕ := full_bags * price_per_bag
  max_earnings = 69000 := by
  sorry

end NUMINAMATH_CALUDE_max_bracelet_earnings_l691_69176


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l691_69105

theorem arithmetic_mean_difference (p q r : ℝ) 
  (mean_pq : (p + q) / 2 = 10)
  (mean_qr : (q + r) / 2 = 22) :
  r - p = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l691_69105


namespace NUMINAMATH_CALUDE_pipe_C_rate_l691_69150

-- Define the rates of the pipes
def rate_A : ℚ := 1 / 60
def rate_B : ℚ := 1 / 80
def rate_combined : ℚ := 1 / 40

-- Define the rate of pipe C
def rate_C : ℚ := rate_A + rate_B - rate_combined

-- Theorem statement
theorem pipe_C_rate : rate_C = 1 / 240 := by
  sorry

end NUMINAMATH_CALUDE_pipe_C_rate_l691_69150


namespace NUMINAMATH_CALUDE_exponent_division_l691_69177

theorem exponent_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l691_69177


namespace NUMINAMATH_CALUDE_area_of_polygon15_l691_69182

/-- A 15-sided polygon on a 1 cm x 1 cm grid -/
def Polygon15 : List (ℤ × ℤ) :=
  [(1,3), (2,4), (2,5), (3,6), (4,6), (5,6), (6,5), (6,4), (5,3), (5,2), (4,1), (3,1), (2,2), (1,2), (1,3)]

/-- The area of a polygon given its vertices -/
def polygonArea (vertices : List (ℤ × ℤ)) : ℚ :=
  sorry

/-- Theorem stating that the area of the 15-sided polygon is 15 cm² -/
theorem area_of_polygon15 : polygonArea Polygon15 = 15 := by
  sorry

end NUMINAMATH_CALUDE_area_of_polygon15_l691_69182


namespace NUMINAMATH_CALUDE_med_school_acceptances_l691_69100

theorem med_school_acceptances 
  (total_researched : ℕ) 
  (applied_fraction : ℚ) 
  (accepted_fraction : ℚ) 
  (h1 : total_researched = 42)
  (h2 : applied_fraction = 1 / 3)
  (h3 : accepted_fraction = 1 / 2) :
  ↑⌊(total_researched : ℚ) * applied_fraction * accepted_fraction⌋ = 7 :=
by sorry

end NUMINAMATH_CALUDE_med_school_acceptances_l691_69100


namespace NUMINAMATH_CALUDE_saltwater_solution_bounds_l691_69152

theorem saltwater_solution_bounds :
  let solution_A : ℝ := 5  -- Concentration of solution A (%)
  let solution_B : ℝ := 8  -- Concentration of solution B (%)
  let solution_C : ℝ := 9  -- Concentration of solution C (%)
  let weight_A : ℝ := 60   -- Weight of solution A (g)
  let weight_B : ℝ := 60   -- Weight of solution B (g)
  let weight_C : ℝ := 47   -- Weight of solution C (g)
  let target_concentration : ℝ := 7  -- Target concentration (%)
  let target_weight : ℝ := 100       -- Target weight (g)

  ∀ x y z : ℝ,
    x + y + z = target_weight →
    solution_A * x + solution_B * y + solution_C * z = target_concentration * target_weight →
    0 ≤ x ∧ x ≤ weight_A →
    0 ≤ y ∧ y ≤ weight_B →
    0 ≤ z ∧ z ≤ weight_C →
    (∃ x_max : ℝ, x ≤ x_max ∧ x_max = 49) ∧
    (∃ x_min : ℝ, x_min ≤ x ∧ x_min = 35) :=
by sorry

end NUMINAMATH_CALUDE_saltwater_solution_bounds_l691_69152


namespace NUMINAMATH_CALUDE_log_inequality_solution_set_l691_69107

theorem log_inequality_solution_set :
  let f : ℝ → ℝ := fun x => Real.log (2 * x + 1) / Real.log (1/2)
  let S : Set ℝ := {x | f x ≥ Real.log 3 / Real.log (1/2)}
  S = Set.Ioc (-1/2) 1 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_solution_set_l691_69107


namespace NUMINAMATH_CALUDE_reaction_masses_l691_69144

-- Define the molar masses
def molar_mass_HCl : ℝ := 36.46
def molar_mass_AgNO3 : ℝ := 169.87
def molar_mass_AgCl : ℝ := 143.32

-- Define the number of moles of AgNO3
def moles_AgNO3 : ℝ := 3

-- Define the reaction stoichiometry
def stoichiometry : ℝ := 1

-- Theorem statement
theorem reaction_masses :
  let mass_HCl := moles_AgNO3 * molar_mass_HCl * stoichiometry
  let mass_AgNO3 := moles_AgNO3 * molar_mass_AgNO3
  let mass_AgCl := moles_AgNO3 * molar_mass_AgCl * stoichiometry
  (mass_HCl = 109.38) ∧ (mass_AgNO3 = 509.61) ∧ (mass_AgCl = 429.96) := by
  sorry

end NUMINAMATH_CALUDE_reaction_masses_l691_69144


namespace NUMINAMATH_CALUDE_total_books_l691_69156

theorem total_books (tim_books sam_books alice_books : ℕ) 
  (h1 : tim_books = 44)
  (h2 : sam_books = 52)
  (h3 : alice_books = 38) :
  tim_books + sam_books + alice_books = 134 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l691_69156


namespace NUMINAMATH_CALUDE_solution_characterization_l691_69186

def equation (x y z : ℝ) : Prop :=
  Real.sqrt (3^x * (5^y + 7^z)) + Real.sqrt (5^y * (7^z + 3^x)) + Real.sqrt (7^z * (3^x + 5^y)) = 
  Real.sqrt 2 * (3^x + 5^y + 7^z)

theorem solution_characterization (x y z : ℝ) :
  equation x y z → ∃ t : ℝ, x = t / Real.log 3 ∧ y = t / Real.log 5 ∧ z = t / Real.log 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l691_69186


namespace NUMINAMATH_CALUDE_largest_prime_diff_144_l691_69127

/-- Two natural numbers are considered different if they are not equal -/
def Different (a b : ℕ) : Prop := a ≠ b

/-- A natural number is prime if it's greater than 1 and its only positive divisors are 1 and itself -/
def IsPrime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d > 0 → d ∣ p → d = 1 ∨ d = p

/-- The statement that the largest possible difference between two different primes summing to 144 is 134 -/
theorem largest_prime_diff_144 : 
  ∃ (p q : ℕ), Different p q ∧ IsPrime p ∧ IsPrime q ∧ p + q = 144 ∧ 
  (∀ (r s : ℕ), Different r s → IsPrime r → IsPrime s → r + s = 144 → s - r ≤ 134) ∧
  q - p = 134 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_diff_144_l691_69127


namespace NUMINAMATH_CALUDE_decimal_point_shift_l691_69134

theorem decimal_point_shift (x : ℝ) : x - x / 10 = 37.35 → x = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_shift_l691_69134


namespace NUMINAMATH_CALUDE_number_of_divisors_of_fermat_like_expression_l691_69169

theorem number_of_divisors_of_fermat_like_expression : 
  ∃ (S : Finset Nat), 
    (∀ n ∈ S, n > 1 ∧ ∀ a : ℤ, (n : ℤ) ∣ (a^25 - a)) ∧ 
    (∀ n : Nat, n > 1 → (∀ a : ℤ, (n : ℤ) ∣ (a^25 - a)) → n ∈ S) ∧
    Finset.card S = 31 :=
sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_fermat_like_expression_l691_69169


namespace NUMINAMATH_CALUDE_last_day_third_quarter_common_year_l691_69194

/-- Represents a day in a month -/
structure DayInMonth where
  month : Nat
  day : Nat

/-- Definition of a common year -/
def isCommonYear (totalDays : Nat) : Prop := totalDays = 365

/-- Definition of the third quarter -/
def isInThirdQuarter (d : DayInMonth) : Prop :=
  d.month ∈ [7, 8, 9]

/-- The last day of the third quarter in a common year -/
theorem last_day_third_quarter_common_year (totalDays : Nat) 
  (h : isCommonYear totalDays) :
  ∃ (d : DayInMonth), 
    isInThirdQuarter d ∧ 
    d.month = 9 ∧ 
    d.day = 30 ∧ 
    (∀ (d' : DayInMonth), isInThirdQuarter d' → d'.month < d.month ∨ (d'.month = d.month ∧ d'.day ≤ d.day)) :=
sorry

end NUMINAMATH_CALUDE_last_day_third_quarter_common_year_l691_69194


namespace NUMINAMATH_CALUDE_entrance_exam_marks_l691_69148

/-- Proves that the number of marks awarded for each correct answer is 3 -/
theorem entrance_exam_marks : 
  ∀ (total_questions correct_answers total_marks : ℕ) 
    (wrong_answer_penalty : ℤ),
  total_questions = 70 →
  correct_answers = 27 →
  total_marks = 38 →
  wrong_answer_penalty = -1 →
  ∃ (marks_per_correct_answer : ℕ),
    marks_per_correct_answer * correct_answers + 
    wrong_answer_penalty * (total_questions - correct_answers) = total_marks ∧
    marks_per_correct_answer = 3 :=
by sorry

end NUMINAMATH_CALUDE_entrance_exam_marks_l691_69148


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_boys_neither_happy_nor_sad_is_6_l691_69114

theorem boys_neither_happy_nor_sad (total_children : Nat) (happy_children : Nat) (sad_children : Nat) 
  (neither_children : Nat) (total_boys : Nat) (total_girls : Nat) (happy_boys : Nat) (sad_girls : Nat) : Nat :=
  by
  -- Assumptions
  have h1 : total_children = 60 := by sorry
  have h2 : happy_children = 30 := by sorry
  have h3 : sad_children = 10 := by sorry
  have h4 : neither_children = 20 := by sorry
  have h5 : total_boys = 18 := by sorry
  have h6 : total_girls = 42 := by sorry
  have h7 : happy_boys = 6 := by sorry
  have h8 : sad_girls = 4 := by sorry
  
  -- Proof
  sorry

-- The theorem statement
theorem boys_neither_happy_nor_sad_is_6 : 
  boys_neither_happy_nor_sad 60 30 10 20 18 42 6 4 = 6 := by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_boys_neither_happy_nor_sad_is_6_l691_69114


namespace NUMINAMATH_CALUDE_fraction_comparison_l691_69141

theorem fraction_comparison : (22222222221 : ℚ) / 22222222223 > (33333333331 : ℚ) / 33333333334 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l691_69141


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_geq_3_l691_69164

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 6

theorem decreasing_function_implies_a_geq_3 :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 3 → f a x₁ > f a x₂) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_geq_3_l691_69164


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_l691_69116

theorem four_digit_numbers_count : 
  (Finset.range 9000).card = (Finset.Icc 1000 9999).card := by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_l691_69116


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_l691_69103

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the parabola
def parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x

-- Define the right focus of the ellipse
def right_focus (x y : ℝ) : Prop := ellipse x y ∧ x > 0 ∧ y = 0

-- Define the focus of the parabola
def parabola_focus (x y p : ℝ) : Prop := x = p / 2 ∧ y = 0

-- The main theorem
theorem parabola_ellipse_focus (p : ℝ) :
  p > 0 →
  (∃ x y, right_focus x y ∧ parabola_focus x y p) →
  p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_l691_69103


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l691_69142

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

-- Define the asymptote equation
def asymptote (m x y : ℝ) : Prop := y = m * x ∨ y = -m * x

-- Theorem statement
theorem hyperbola_asymptote_slope :
  ∃ m : ℝ, (∀ x y : ℝ, hyperbola x y → asymptote m x y) ∧ m = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l691_69142


namespace NUMINAMATH_CALUDE_matching_probability_is_four_fifteenths_l691_69185

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Alice's jelly bean distribution -/
def alice : JellyBeans := { green := 2, red := 2, blue := 1, yellow := 0 }

/-- Carl's jelly bean distribution -/
def carl : JellyBeans := { green := 3, red := 1, blue := 0, yellow := 2 }

/-- The probability of selecting matching colors -/
def matchingProbability (a c : JellyBeans) : ℚ :=
  (a.green * c.green + a.red * c.red) / (a.total * c.total)

theorem matching_probability_is_four_fifteenths :
  matchingProbability alice carl = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_is_four_fifteenths_l691_69185


namespace NUMINAMATH_CALUDE_closest_vertex_of_dilated_square_l691_69191

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  center : Point
  area : ℝ
  verticalSide : Bool

/-- Dilates a point from the origin by a given factor -/
def dilatePoint (p : Point) (factor : ℝ) : Point :=
  { x := p.x * factor, y := p.y * factor }

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Finds the vertex of a dilated square closest to the origin -/
def closestVertexToDilatedSquare (s : Square) (dilationFactor : ℝ) : Point :=
  sorry

theorem closest_vertex_of_dilated_square :
  let originalSquare : Square := {
    center := { x := 5, y := -3 },
    area := 16,
    verticalSide := true
  }
  let dilationFactor : ℝ := 3
  let closestVertex := closestVertexToDilatedSquare originalSquare dilationFactor
  closestVertex.x = 9 ∧ closestVertex.y = -3 := by
  sorry

end NUMINAMATH_CALUDE_closest_vertex_of_dilated_square_l691_69191


namespace NUMINAMATH_CALUDE_period_of_f_l691_69146

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

theorem period_of_f (f : ℝ → ℝ) (h : has_property f) : is_periodic f 4 := by
  sorry

end NUMINAMATH_CALUDE_period_of_f_l691_69146


namespace NUMINAMATH_CALUDE_sum_of_pairwise_ratios_geq_three_halves_l691_69160

theorem sum_of_pairwise_ratios_geq_three_halves 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_pairwise_ratios_geq_three_halves_l691_69160


namespace NUMINAMATH_CALUDE_distance_between_x_intercepts_l691_69101

-- Define the slopes and intersection point
def m1 : ℝ := 4
def m2 : ℝ := -2
def intersection : ℝ × ℝ := (8, 20)

-- Define the lines using point-slope form
def line1 (x : ℝ) : ℝ := m1 * (x - intersection.1) + intersection.2
def line2 (x : ℝ) : ℝ := m2 * (x - intersection.1) + intersection.2

-- Define x-intercepts
noncomputable def x_intercept1 : ℝ := (intersection.2 - m1 * intersection.1) / (-m1)
noncomputable def x_intercept2 : ℝ := (intersection.2 - m2 * intersection.1) / (-m2)

-- Theorem statement
theorem distance_between_x_intercepts :
  |x_intercept2 - x_intercept1| = 15 := by sorry

end NUMINAMATH_CALUDE_distance_between_x_intercepts_l691_69101


namespace NUMINAMATH_CALUDE_chess_go_problem_l691_69111

/-- Represents the prices and quantities of Chinese chess and Go sets -/
structure ChessGoSets where
  chess_price : ℝ
  go_price : ℝ
  total_sets : ℕ
  max_cost : ℝ

/-- Defines the conditions given in the problem -/
def problem_conditions (s : ChessGoSets) : Prop :=
  2 * s.chess_price + 3 * s.go_price = 140 ∧
  4 * s.chess_price + s.go_price = 130 ∧
  s.total_sets = 80 ∧
  s.max_cost = 2250

/-- Theorem stating the solution to the problem -/
theorem chess_go_problem (s : ChessGoSets) 
  (h : problem_conditions s) : 
  s.chess_price = 25 ∧ 
  s.go_price = 30 ∧ 
  (∀ m : ℕ, m * s.go_price + (s.total_sets - m) * s.chess_price ≤ s.max_cost → m ≤ 50) ∧
  (∀ a : ℝ, a > 0 → 
    (a < 10 → 0.9 * a * s.go_price < 0.7 * a * s.go_price + 60) ∧
    (a = 10 → 0.9 * a * s.go_price = 0.7 * a * s.go_price + 60) ∧
    (a > 10 → 0.9 * a * s.go_price > 0.7 * a * s.go_price + 60)) :=
by
  sorry

end NUMINAMATH_CALUDE_chess_go_problem_l691_69111


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l691_69137

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n, a n > 0) →
  geometric_sequence a q →
  (a 2 - a 1 = a 1 - (1/2) * a 3) →
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l691_69137


namespace NUMINAMATH_CALUDE_complement_of_25_36_l691_69197

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 180 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_25_36 :
  complement ⟨25, 36⟩ = ⟨154, 24⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_25_36_l691_69197


namespace NUMINAMATH_CALUDE_bees_after_seven_days_l691_69122

/-- Calculates the total number of bees in a hive after a given number of days -/
def total_bees_in_hive (initial_bees : ℕ) (hatch_rate : ℕ) (loss_rate : ℕ) (days : ℕ) : ℕ :=
  initial_bees + days * (hatch_rate - loss_rate) + 1

/-- Theorem stating the total number of bees in the hive after 7 days -/
theorem bees_after_seven_days :
  total_bees_in_hive 12500 3000 900 7 = 27201 := by
  sorry

#eval total_bees_in_hive 12500 3000 900 7

end NUMINAMATH_CALUDE_bees_after_seven_days_l691_69122


namespace NUMINAMATH_CALUDE_prob_green_is_0_15_l691_69171

/-- The probability of selecting a green jelly bean from a jar -/
def prob_green (prob_red prob_orange prob_blue prob_yellow : ℝ) : ℝ :=
  1 - (prob_red + prob_orange + prob_blue + prob_yellow)

/-- Theorem: The probability of selecting a green jelly bean is 0.15 -/
theorem prob_green_is_0_15 :
  prob_green 0.15 0.35 0.2 0.15 = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_is_0_15_l691_69171


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l691_69179

theorem quadratic_roots_relation (a b p q : ℝ) (r₁ r₂ : ℂ) : 
  (r₁ + r₂ = -a ∧ r₁ * r₂ = b) →  -- roots of x² + ax + b = 0
  (r₁^2 + r₂^2 = -p ∧ r₁^2 * r₂^2 = q) →  -- r₁² and r₂² are roots of x² + px + q = 0
  p = -a^2 + 2*b :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l691_69179


namespace NUMINAMATH_CALUDE_preimage_of_two_three_l691_69153

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Theorem statement
theorem preimage_of_two_three :
  ∃ (x y : ℝ), f (x, y) = (2, 3) ∧ x = 5/2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_two_three_l691_69153


namespace NUMINAMATH_CALUDE_right_triangle_area_l691_69139

/-- Given a right-angled triangle with perimeter 18 and sum of squares of side lengths 128, its area is 9. -/
theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 18 →
  a^2 + b^2 + c^2 = 128 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 9 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l691_69139


namespace NUMINAMATH_CALUDE_f_values_l691_69149

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt ((1 - Real.sin x) / (1 + Real.sin x)) - Real.sqrt ((1 + Real.sin x) / (1 - Real.sin x))) *
  (Real.sqrt ((1 - Real.cos x) / (1 + Real.cos x)) - Real.sqrt ((1 + Real.cos x) / (1 - Real.cos x)))

theorem f_values (x : ℝ) 
  (h1 : x ∈ Set.Ioo 0 (2 * Real.pi))
  (h2 : x ≠ Real.pi / 2 ∧ x ≠ Real.pi ∧ x ≠ 3 * Real.pi / 2) :
  (0 < x ∧ x < Real.pi / 2 ∨ Real.pi < x ∧ x < 3 * Real.pi / 2) → f x = 4 ∧
  (Real.pi / 2 < x ∧ x < Real.pi ∨ 3 * Real.pi / 2 < x ∧ x < 2 * Real.pi) → f x = -4 := by
  sorry

#check f_values

end NUMINAMATH_CALUDE_f_values_l691_69149


namespace NUMINAMATH_CALUDE_percentage_increase_l691_69192

/-- Given two positive real numbers a and b with a ratio of 4:5, 
    and x and m derived from a and b respectively, 
    prove that the percentage increase from a to x is 25% --/
theorem percentage_increase (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a / b = 4 / 5 →
  ∃ p, x = a * (1 + p / 100) →
  m = b * 0.6 →
  m / x = 0.6 →
  p = 25 := by
sorry


end NUMINAMATH_CALUDE_percentage_increase_l691_69192


namespace NUMINAMATH_CALUDE_cindy_calculation_l691_69181

theorem cindy_calculation (x : ℝ) : (x - 7) / 5 = 15 → (x - 5) / 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l691_69181


namespace NUMINAMATH_CALUDE_equation_solution_l691_69123

theorem equation_solution (x y : ℚ) : 
  (4 * x + 2 * y = 12) → 
  (2 * x + 4 * y = 16) → 
  (20 * x^2 + 24 * x * y + 20 * y^2 = 3280 / 9) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l691_69123


namespace NUMINAMATH_CALUDE_savings_calculation_l691_69113

def income_expenditure_ratio : Rat := 10 / 4
def income : ℕ := 19000
def tax_rate : Rat := 15 / 100
def long_term_investment_rate : Rat := 10 / 100
def short_term_investment_rate : Rat := 20 / 100

def calculate_savings (income_expenditure_ratio : Rat) (income : ℕ) (tax_rate : Rat) 
  (long_term_investment_rate : Rat) (short_term_investment_rate : Rat) : ℕ :=
  sorry

theorem savings_calculation :
  calculate_savings income_expenditure_ratio income tax_rate 
    long_term_investment_rate short_term_investment_rate = 11628 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l691_69113


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l691_69130

theorem expression_simplification_and_evaluation (x : ℤ) 
  (h1 : 3 * x + 7 > 1) (h2 : 2 * x - 1 < 5) :
  let expr := (x / (x - 1)) / ((x^2 - x) / (x^2 - 2*x + 1)) - (x + 2) / (x + 1)
  (expr = -1 / (x + 1)) ∧ 
  (expr = -1/3 ∨ expr = -1/2 ∨ expr = -1) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l691_69130


namespace NUMINAMATH_CALUDE_quadratic_max_condition_l691_69126

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 6*x - 7

-- Define the maximum value function
def y_max (t : ℝ) : ℝ := -(t-3)^2 + 2

-- Theorem statement
theorem quadratic_max_condition (t : ℝ) :
  (∀ x : ℝ, t ≤ x ∧ x ≤ t + 2 → f x ≤ y_max t) →
  (∃ x : ℝ, t ≤ x ∧ x ≤ t + 2 ∧ f x = y_max t) →
  t ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_condition_l691_69126


namespace NUMINAMATH_CALUDE_count_integers_correct_l691_69173

/-- Count of three-digit positive integers starting with 2 and greater than 217 -/
def count_integers : ℕ := 82

/-- The smallest three-digit integer starting with 2 and greater than 217 -/
def min_integer : ℕ := 218

/-- The largest three-digit integer starting with 2 -/
def max_integer : ℕ := 299

/-- Theorem stating that the count of integers is correct -/
theorem count_integers_correct :
  count_integers = max_integer - min_integer + 1 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_correct_l691_69173


namespace NUMINAMATH_CALUDE_quadratic_root_and_coefficient_l691_69193

def quadratic_polynomial (x : ℂ) : ℂ := 3 * x^2 - 24 * x + 60

theorem quadratic_root_and_coefficient : 
  (quadratic_polynomial (4 + 2*Complex.I) = 0) ∧ 
  (∃ (a b : ℝ), ∀ x, quadratic_polynomial x = 3 * x^2 + a * x + b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_and_coefficient_l691_69193


namespace NUMINAMATH_CALUDE_pentagon_area_greater_than_third_square_l691_69133

theorem pentagon_area_greater_than_third_square (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + (a*b)/4 + (Real.sqrt 3/4)*b^2 > ((a+b)^2)/3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_greater_than_third_square_l691_69133


namespace NUMINAMATH_CALUDE_number_calculation_l691_69109

theorem number_calculation (x : ℝ) (h : 0.3 * x = 108.0) : x = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l691_69109


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l691_69132

/-- Given a parabola and a hyperbola with specific properties, prove that the parameter 'a' of the hyperbola equals 1/4. -/
theorem parabola_hyperbola_intersection (p : ℝ) (m : ℝ) (a : ℝ) : 
  p > 0 → -- p is positive
  m^2 = 2*p -- point (1,m) is on the parabola y^2 = 2px
  → (1 - p/2)^2 + m^2 = 5^2 -- distance from (1,m) to focus (p/2, 0) is 5
  → ∃ (k : ℝ), k^2 * a = 1 ∧ k * m = 2 -- asymptote y = kx is perpendicular to AM (slope of AM is m/2)
  → a = 1/4 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l691_69132


namespace NUMINAMATH_CALUDE_distance_origin_to_point_l691_69180

/-- The distance between the origin (0, 0, 0) and the point (1, 2, 3) is √14 -/
theorem distance_origin_to_point :
  Real.sqrt ((1 : ℝ)^2 + 2^2 + 3^2) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_distance_origin_to_point_l691_69180


namespace NUMINAMATH_CALUDE_bella_roses_l691_69154

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of roses Bella received from her parents -/
def roses_from_parents : ℕ := 2

/-- The number of Bella's dancer friends -/
def number_of_friends : ℕ := 10

/-- The number of roses each friend gave to Bella -/
def roses_per_friend : ℕ := 2

/-- The total number of roses Bella received -/
def total_roses : ℕ := roses_from_parents * dozen + number_of_friends * roses_per_friend

theorem bella_roses : total_roses = 44 := by
  sorry

end NUMINAMATH_CALUDE_bella_roses_l691_69154


namespace NUMINAMATH_CALUDE_nate_search_speed_l691_69189

/-- The number of rows in Section G of the parking lot -/
def section_g_rows : ℕ := 15

/-- The number of cars per row in Section G -/
def section_g_cars_per_row : ℕ := 10

/-- The number of rows in Section H of the parking lot -/
def section_h_rows : ℕ := 20

/-- The number of cars per row in Section H -/
def section_h_cars_per_row : ℕ := 9

/-- The time Nate spent searching in minutes -/
def search_time : ℕ := 30

/-- The number of cars Nate can walk past per minute -/
def cars_per_minute : ℕ := 11

theorem nate_search_speed :
  (section_g_rows * section_g_cars_per_row + section_h_rows * section_h_cars_per_row) / search_time = cars_per_minute := by
  sorry

end NUMINAMATH_CALUDE_nate_search_speed_l691_69189


namespace NUMINAMATH_CALUDE_angle_D_value_l691_69115

-- Define the angles
def A : ℝ := 30
def B (D : ℝ) : ℝ := 2 * D
def C (D : ℝ) : ℝ := D + 40

-- Theorem statement
theorem angle_D_value :
  ∀ D : ℝ, A + B D + C D + D = 360 → D = 72.5 := by sorry

end NUMINAMATH_CALUDE_angle_D_value_l691_69115


namespace NUMINAMATH_CALUDE_savings_to_earnings_ratio_l691_69184

/-- Proves that the ratio of monthly savings to monthly earnings is 1/2 -/
theorem savings_to_earnings_ratio
  (monthly_earnings : ℕ)
  (vehicle_cost : ℕ)
  (saving_period : ℕ)
  (h1 : monthly_earnings = 4000)
  (h2 : vehicle_cost = 16000)
  (h3 : saving_period = 8) :
  (vehicle_cost / saving_period) / monthly_earnings = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_savings_to_earnings_ratio_l691_69184


namespace NUMINAMATH_CALUDE_special_linear_function_properties_l691_69162

/-- A linear function y = mx + c, where m is the slope and c is the y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- The linear function y = (2a-4)x + (3-b) -/
def specialLinearFunction (a b : ℝ) : LinearFunction where
  slope := 2*a - 4
  intercept := 3 - b

theorem special_linear_function_properties (a b : ℝ) :
  let f := specialLinearFunction a b
  (∃ k : ℝ, ∀ x, f.slope * x = k * x) ↔ (a ≠ 2 ∧ b = 3) ∧
  (f.slope < 0 ∧ f.intercept ≤ 0) ↔ (a < 2 ∧ b ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_special_linear_function_properties_l691_69162


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l691_69120

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l691_69120


namespace NUMINAMATH_CALUDE_parabola_intersection_l691_69131

/-- Parabola 1 function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 1

/-- Parabola 2 function -/
def g (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 1

/-- Theorem stating that (0, 1) and (-8, 233) are the only intersection points -/
theorem parabola_intersection :
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 0 ∧ y = 1) ∨ (x = -8 ∧ y = 233) := by
  sorry

#check parabola_intersection

end NUMINAMATH_CALUDE_parabola_intersection_l691_69131


namespace NUMINAMATH_CALUDE_fencing_required_l691_69166

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) (fencing : ℝ) : 
  area = 650 ∧ uncovered_side = 20 → fencing = 85 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l691_69166


namespace NUMINAMATH_CALUDE_existence_of_uncuttable_rectangle_l691_69175

/-- A rectangle with natural number side lengths -/
structure Rectangle where
  length : ℕ+
  width : ℕ+

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- A predicate that checks if two numbers are almost equal -/
def almost_equal (a b : ℕ) : Prop := a = b ∨ a = b + 1 ∨ a = b - 1

/-- A predicate that checks if a rectangle can be cut out from another rectangle -/
def can_cut_out (small big : Rectangle) : Prop :=
  small.length ≤ big.length ∧ small.width ≤ big.width ∨
  small.length ≤ big.width ∧ small.width ≤ big.length

theorem existence_of_uncuttable_rectangle :
  ∃ (r : Rectangle), ¬∃ (s : Rectangle), 
    can_cut_out s r ∧ almost_equal (area s) ((area r) / 2) :=
sorry

end NUMINAMATH_CALUDE_existence_of_uncuttable_rectangle_l691_69175


namespace NUMINAMATH_CALUDE_die_events_l691_69143

-- Define the sample space and events
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1}
def B : Set Nat := {2, 4, 6}
def C : Set Nat := {1, 2}
def D : Set Nat := {3, 4, 5, 6}
def E : Set Nat := {3, 6}

-- Theorem to prove the relationships and set operations
theorem die_events :
  (A ⊆ C) ∧
  (C ∪ D = Ω) ∧
  (E ⊆ D) ∧
  (Dᶜ = {1, 2}) ∧
  (Aᶜ ∩ C = {2}) ∧
  (Bᶜ ∪ C = {1, 2, 3}) ∧
  (Dᶜ ∪ Eᶜ = {1, 2, 4, 5}) :=
by sorry

end NUMINAMATH_CALUDE_die_events_l691_69143


namespace NUMINAMATH_CALUDE_perimeter_difference_l691_69174

/-- Represents a rectangle with length and height -/
structure Rectangle where
  length : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.height)

/-- Theorem stating the difference in perimeters of two rectangles -/
theorem perimeter_difference (inner outer : Rectangle) 
  (h1 : outer.length = 7)
  (h2 : outer.height = 5) :
  perimeter outer - perimeter inner = 24 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_difference_l691_69174


namespace NUMINAMATH_CALUDE_horner_method_f_neg_two_l691_69135

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem horner_method_f_neg_two :
  f (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_horner_method_f_neg_two_l691_69135


namespace NUMINAMATH_CALUDE_uncle_ben_eggs_l691_69112

theorem uncle_ben_eggs (total_chickens roosters non_laying_hens eggs_per_hen : ℕ) 
  (h1 : total_chickens = 440)
  (h2 : roosters = 39)
  (h3 : non_laying_hens = 15)
  (h4 : eggs_per_hen = 3) :
  total_chickens - roosters - non_laying_hens = 386 →
  (total_chickens - roosters - non_laying_hens) * eggs_per_hen = 1158 := by
  sorry

end NUMINAMATH_CALUDE_uncle_ben_eggs_l691_69112


namespace NUMINAMATH_CALUDE_closed_path_theorem_l691_69136

/-- A closed path on an m×n table satisfying specific conditions -/
structure ClosedPath (m n : ℕ) where
  -- Ensure m and n are at least 4
  m_ge_four : m ≥ 4
  n_ge_four : n ≥ 4
  -- A is the number of straight-forward vertices
  A : ℕ
  -- B is the number of squares with two opposite sides used
  B : ℕ
  -- C is the number of unused squares
  C : ℕ
  -- The path doesn't intersect itself
  no_self_intersection : True
  -- The path passes through all interior vertices
  passes_all_interior : True
  -- The path doesn't pass through outer vertices
  no_outer_vertices : True

/-- Theorem: For a closed path on an m×n table satisfying the given conditions,
    A = B - C + m + n - 1 -/
theorem closed_path_theorem (m n : ℕ) (path : ClosedPath m n) :
  path.A = path.B - path.C + m + n - 1 := by
  sorry

end NUMINAMATH_CALUDE_closed_path_theorem_l691_69136


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l691_69138

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/4 + 1/5, 1/4 + 1/6, 1/4 + 1/3, 1/4 + 1/8, 1/4 + 1/7]
  (∀ x ∈ sums, x ≤ (1/4 + 1/3)) ∧ (1/4 + 1/3 = 7/12) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l691_69138


namespace NUMINAMATH_CALUDE_polynomial_expansion_l691_69151

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 + 4 * x - 7) * (2 * x^4 - 3 * x^2 + 5) =
  6 * x^7 + 12 * x^5 - 9 * x^4 - 21 * x^3 - 11 * x + 35 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l691_69151


namespace NUMINAMATH_CALUDE_new_dwelling_points_order_l691_69165

open Real

-- Define the "new dwelling point" for each function
def α : ℝ := 1

-- β is implicitly defined by the equation ln(β+1) = 1/(β+1)
def β : ℝ := sorry

-- γ is implicitly defined by the equation cos γ = -sin γ, where γ ∈ (π/2, π)
noncomputable def γ : ℝ := sorry

axiom β_eq : log (β + 1) = 1 / (β + 1)
axiom γ_eq : cos γ = -sin γ
axiom γ_range : π / 2 < γ ∧ γ < π

-- Theorem statement
theorem new_dwelling_points_order : γ > α ∧ α > β := by sorry

end NUMINAMATH_CALUDE_new_dwelling_points_order_l691_69165


namespace NUMINAMATH_CALUDE_factorization_proof_l691_69102

theorem factorization_proof (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 3) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l691_69102


namespace NUMINAMATH_CALUDE_polynomial_value_constraint_l691_69183

theorem polynomial_value_constraint 
  (P : ℤ → ℤ) 
  (h_poly : ∀ x y : ℤ, (P x - P y) ∣ (x - y))
  (h_distinct : ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P a = 2 ∧ P b = 2 ∧ P c = 2) :
  ∀ x : ℤ, P x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_constraint_l691_69183


namespace NUMINAMATH_CALUDE_special_numbers_property_l691_69110

/-- Given a natural number, return the sum of its digits -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The list of 13 numbers that satisfy the conditions -/
def specialNumbers : List ℕ := [6, 15, 24, 33, 42, 51, 60, 105, 114, 123, 132, 141, 150]

theorem special_numbers_property :
  (∃ (nums : List ℕ),
    nums.length = 13 ∧
    nums.sum = 996 ∧
    nums.Nodup ∧
    ∀ (x y : ℕ), x ∈ nums → y ∈ nums → digitSum x = digitSum y) := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_property_l691_69110


namespace NUMINAMATH_CALUDE_remainder_two_power_200_minus_3_mod_7_l691_69125

theorem remainder_two_power_200_minus_3_mod_7 : 
  (2^200 - 3) % 7 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_two_power_200_minus_3_mod_7_l691_69125


namespace NUMINAMATH_CALUDE_bicycle_problem_l691_69155

/-- The time when two people traveling perpendicular to each other at different speeds are 100 miles apart -/
theorem bicycle_problem (jenny_speed mark_speed : ℝ) (h1 : jenny_speed = 10) (h2 : mark_speed = 15) :
  let t := (20 * Real.sqrt 13) / 13
  (t * jenny_speed) ^ 2 + (t * mark_speed) ^ 2 = 100 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_problem_l691_69155


namespace NUMINAMATH_CALUDE_find_q_l691_69167

theorem find_q (p q : ℝ) (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 3/2) (h4 : p*q = 9) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_q_l691_69167


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_inequality_l691_69118

theorem quadratic_no_real_roots_inequality (a b c : ℝ) :
  ((b + c) * x^2 + (a + c) * x + (a + b) = 0 → False) →
  4 * a * c - b^2 ≤ 3 * a * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_inequality_l691_69118


namespace NUMINAMATH_CALUDE_expected_zeroes_l691_69157

/-- Represents the probability of getting heads on the unfair coin. -/
def A : ℚ := 1/5

/-- Represents the length of the generated string. -/
def B : ℕ := 4

/-- Represents the expected number of zeroes in the string. -/
def C : ℚ := B/2

/-- Proves that the expected number of zeroes in the string is half its length,
    and that for the given probabilities, the string length is 4. -/
theorem expected_zeroes :
  C = B/2 ∧ A = 3*B/((B+1)*(B+2)*2) ∧ B = (4 - A*B)/(4*A) := by
  sorry

#eval C  -- Should output 2

end NUMINAMATH_CALUDE_expected_zeroes_l691_69157


namespace NUMINAMATH_CALUDE_fraction_of_fraction_fraction_of_three_fifths_is_two_fifteenths_l691_69170

theorem fraction_of_fraction (a b c d : ℚ) (h : a / b = c / d) :
  (c / d) / (a / b) = d / a :=
by sorry

theorem fraction_of_three_fifths_is_two_fifteenths :
  (2 / 15) / (3 / 5) = 2 / 9 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_fraction_of_three_fifths_is_two_fifteenths_l691_69170


namespace NUMINAMATH_CALUDE_range_of_m_l691_69195

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - x - y = 0

-- Define the point A
def A (m : ℝ) : ℝ × ℝ := (m, m)

-- Define the condition that any line through A intersects C
def intersects_C (m : ℝ) : Prop :=
  ∀ (k b : ℝ), ∃ (x y : ℝ), C x y ∧ y = k * x + b ∧ m * k + b = m

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, intersects_C m ↔ 0 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l691_69195


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l691_69163

theorem unique_integer_satisfying_conditions : ∃! (n : ℤ), n + 15 > 16 ∧ -3*n > -9 :=
  sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l691_69163


namespace NUMINAMATH_CALUDE_inequality_proof_l691_69117

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l691_69117


namespace NUMINAMATH_CALUDE_distance_between_points_l691_69188

def point1 : ℝ × ℝ := (3, -5)
def point2 : ℝ × ℝ := (-4, 4)

theorem distance_between_points :
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l691_69188


namespace NUMINAMATH_CALUDE_floor_sqrt_50_l691_69199

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_l691_69199


namespace NUMINAMATH_CALUDE_semicircle_pattern_area_l691_69108

/-- The area of the shaded region formed by semicircles in a foot-long pattern -/
theorem semicircle_pattern_area :
  let diameter : ℝ := 3  -- diameter of each semicircle in inches
  let pattern_length : ℝ := 12  -- length of the pattern in inches (1 foot)
  let num_semicircles : ℝ := pattern_length / diameter  -- number of semicircles in the pattern
  let semicircle_area : ℝ → ℝ := λ r => (π * r^2) / 2  -- area of a semicircle
  let total_area : ℝ := num_semicircles * semicircle_area (diameter / 2)
  total_area = (9/2) * π
  := by sorry

end NUMINAMATH_CALUDE_semicircle_pattern_area_l691_69108


namespace NUMINAMATH_CALUDE_circles_common_chord_common_chord_length_l691_69196

/-- Circle C₁ with equation x² + y² - 2x + 10y - 24 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 10*y - 24 = 0

/-- Circle C₂ with equation x² + y² + 2x + 2y - 8 = 0 -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 8 = 0

/-- The line on which the common chord of C₁ and C₂ lies -/
def common_chord_line (x y : ℝ) : Prop :=
  x - 6*y + 6 = 0

theorem circles_common_chord (x y : ℝ) :
  (C₁ x y ∧ C₂ x y) → common_chord_line x y :=
sorry

theorem common_chord_length : 
  ∃ (a b : ℝ), C₁ a b ∧ C₂ a b ∧ 
  ∃ (c d : ℝ), C₁ c d ∧ C₂ c d ∧ 
  ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 2 * 13^(1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_circles_common_chord_common_chord_length_l691_69196


namespace NUMINAMATH_CALUDE_inconsistent_means_l691_69121

theorem inconsistent_means : ¬ ∃ x : ℝ,
  (x + 42 + 78 + 104) / 4 = 62 ∧
  (48 + 62 + 98 + 124 + x) / 5 = 78 := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_means_l691_69121


namespace NUMINAMATH_CALUDE_max_xy_constraint_l691_69158

theorem max_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 8 * y = 65) :
  x * y ≤ 25 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 5 * x₀ + 8 * y₀ = 65 ∧ x₀ * y₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_constraint_l691_69158


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l691_69140

/-- Given that (m-2)x^|m| - bx - 1 = 0 is a quadratic equation in x, prove that m = -2 -/
theorem quadratic_equation_m_value (m b : ℝ) : 
  (∀ x, ∃ a c : ℝ, (m - 2) * x^(|m|) - b*x - 1 = a*x^2 + b*x + c) → 
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l691_69140


namespace NUMINAMATH_CALUDE_square_sum_implies_product_l691_69159

theorem square_sum_implies_product (x : ℝ) :
  Real.sqrt (10 + x) + Real.sqrt (15 - x) = 6 →
  (10 + x) * (15 - x) = 121 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_implies_product_l691_69159


namespace NUMINAMATH_CALUDE_percentage_of_girls_in_class_l691_69161

theorem percentage_of_girls_in_class (girls boys : ℕ) (h1 : girls = 10) (h2 : boys = 15) :
  (girls : ℚ) / ((girls : ℚ) + (boys : ℚ)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_in_class_l691_69161


namespace NUMINAMATH_CALUDE_multiple_compounds_with_same_oxygen_percentage_l691_69129

/-- Represents a chemical compound -/
structure Compound where
  elements : List String
  massPercentages : List Float
  deriving Repr

/-- Predicate to check if a compound has 57.14% oxygen -/
def hasCorrectOxygenPercentage (c : Compound) : Prop :=
  "O" ∈ c.elements ∧ 
  let oIndex := c.elements.indexOf "O"
  c.massPercentages[oIndex]! = 57.14

/-- Theorem stating that multiple compounds can have 57.14% oxygen -/
theorem multiple_compounds_with_same_oxygen_percentage :
  ∃ (c1 c2 : Compound), c1 ≠ c2 ∧ 
    hasCorrectOxygenPercentage c1 ∧ 
    hasCorrectOxygenPercentage c2 :=
sorry

end NUMINAMATH_CALUDE_multiple_compounds_with_same_oxygen_percentage_l691_69129


namespace NUMINAMATH_CALUDE_puzzle_solution_l691_69168

def concatenate (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

def special_sum (a b c : ℕ) : ℕ := 
  10000 * (a * b) + 100 * (a * c) + concatenate c b a

theorem puzzle_solution (h1 : special_sum 5 3 2 = 151022)
                        (h2 : special_sum 9 2 4 = 183652)
                        (h3 : special_sum 7 2 5 = 143547) :
  ∃ x, special_sum 7 2 x = 143547 ∧ x = 5 :=
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l691_69168


namespace NUMINAMATH_CALUDE_fraction_equality_l691_69106

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : a^2 + b^2 ≠ 0) (h4 : a^4 - 2*b^4 ≠ 0) 
  (h5 : a^2 * b^2 / (a^4 - 2*b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l691_69106


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l691_69190

/-- The number of ways to place n distinguishable balls into k distinguishable boxes -/
def placeBalls (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to place 5 distinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : placeBalls 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l691_69190


namespace NUMINAMATH_CALUDE_percent_difference_z_x_of_w_l691_69104

theorem percent_difference_z_x_of_w (y q w x z : ℝ) 
  (hw : w = 0.60 * q)
  (hq : q = 0.60 * y)
  (hz : z = 0.54 * y)
  (hx : x = 1.30 * w) :
  (z - x) / w = 0.20 := by
sorry

end NUMINAMATH_CALUDE_percent_difference_z_x_of_w_l691_69104


namespace NUMINAMATH_CALUDE_smallest_three_digit_product_l691_69187

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem smallest_three_digit_product :
  ∀ n x y : ℕ,
    n = x * y * (10 * x + y) →
    100 ≤ n →
    n < 1000 →
    is_prime x →
    is_prime y →
    is_prime (10 * x + y) →
    x < 10 →
    y < 10 →
    x % 2 = 0 →
    y % 2 = 1 →
    x ≠ y →
    x ≠ 10 * x + y →
    y ≠ 10 * x + y →
    n ≥ 138 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_product_l691_69187


namespace NUMINAMATH_CALUDE_ratio_michael_monica_l691_69119

-- Define the ages as real numbers
variable (patrick_age michael_age monica_age : ℝ)

-- Define the conditions
axiom ratio_patrick_michael : patrick_age / michael_age = 3 / 5
axiom sum_of_ages : patrick_age + michael_age + monica_age = 245
axiom age_difference : monica_age - patrick_age = 80

-- Theorem to prove
theorem ratio_michael_monica :
  michael_age / monica_age = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_ratio_michael_monica_l691_69119


namespace NUMINAMATH_CALUDE_coin_problem_l691_69147

theorem coin_problem (total_coins : ℕ) (total_value : ℕ) 
  (pennies nickels dimes quarters : ℕ) :
  total_coins = 11 →
  total_value = 165 →
  pennies ≥ 1 →
  nickels ≥ 1 →
  dimes ≥ 1 →
  quarters ≥ 1 →
  total_coins = pennies + nickels + dimes + quarters →
  total_value = pennies + 5 * nickels + 10 * dimes + 25 * quarters →
  quarters = 4 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l691_69147


namespace NUMINAMATH_CALUDE_group_size_proof_l691_69124

theorem group_size_proof (total_paise : ℕ) (h : total_paise = 4624) :
  ∃ n : ℕ, n * n = total_paise ∧ n = 68 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l691_69124


namespace NUMINAMATH_CALUDE_new_student_weight_l691_69172

theorem new_student_weight (n : ℕ) (w_avg : ℝ) (w_new_avg : ℝ) :
  n = 29 →
  w_avg = 28 →
  w_new_avg = 27.5 →
  (n : ℝ) * w_avg + (n + 1) * w_new_avg - n * w_avg = 13 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l691_69172


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l691_69128

theorem repeating_decimal_division (a b : ℚ) :
  a = 45 / 99 →
  b = 18 / 99 →
  a / b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l691_69128


namespace NUMINAMATH_CALUDE_max_triangle_area_l691_69198

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x + 1)^2 + y^2) + Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt (x^2 + (y - 1)^2) = 2 * Real.sqrt 2

-- Define the area of triangle F₁PF₂
def triangle_area (x y : ℝ) : ℝ :=
  abs (y) -- The base of the triangle is 2, so the area is |y|

-- Theorem statement
theorem max_triangle_area :
  ∀ x y : ℝ, C x y → triangle_area x y ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l691_69198


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l691_69145

/-- Given an ellipse with equation x²/16 + y²/9 = 1, 
    the length of the focal distance is 2√7 -/
theorem ellipse_focal_distance : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2/16 + y^2/9 = 1}
  ∃ c : ℝ, c = 2 * Real.sqrt 7 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ ellipse → 
      c = Real.sqrt ((x^2 + y^2) - 4 * Real.sqrt (x^2 * y^2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_focal_distance_l691_69145
