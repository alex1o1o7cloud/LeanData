import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_polynomials_exist_l3312_331272

/-- A quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of real roots of a quadratic polynomial -/
def num_real_roots (p : QuadraticPolynomial) : ℕ :=
  sorry

/-- The sum of two quadratic polynomials -/
def add (p q : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a + q.a, p.b + q.b, p.c + q.c⟩

theorem quadratic_polynomials_exist : ∃ (f g h : QuadraticPolynomial),
  (num_real_roots f = 2) ∧
  (num_real_roots g = 2) ∧
  (num_real_roots h = 2) ∧
  (num_real_roots (add f g) = 1) ∧
  (num_real_roots (add f h) = 1) ∧
  (num_real_roots (add g h) = 1) ∧
  (num_real_roots (add (add f g) h) = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_exist_l3312_331272


namespace NUMINAMATH_CALUDE_decision_box_two_exits_l3312_331265

-- Define the types of program blocks
inductive ProgramBlock
  | TerminationBox
  | InputOutputBox
  | ProcessingBox
  | DecisionBox

-- Define a function that returns the number of exit directions for each program block
def exitDirections (block : ProgramBlock) : Nat :=
  match block with
  | ProgramBlock.TerminationBox => 1
  | ProgramBlock.InputOutputBox => 1
  | ProgramBlock.ProcessingBox => 1
  | ProgramBlock.DecisionBox => 2

-- Theorem statement
theorem decision_box_two_exits :
  ∀ (block : ProgramBlock), exitDirections block = 2 ↔ block = ProgramBlock.DecisionBox :=
by sorry


end NUMINAMATH_CALUDE_decision_box_two_exits_l3312_331265


namespace NUMINAMATH_CALUDE_exam_score_problem_l3312_331259

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) :
  total_questions = 60 →
  correct_score = 4 →
  total_score = 120 →
  correct_answers = 36 →
  (total_questions - correct_answers) * (correct_score - (total_score - correct_answers * correct_score) / (total_questions - correct_answers)) = total_questions - correct_answers :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3312_331259


namespace NUMINAMATH_CALUDE_product_of_seven_and_sum_l3312_331281

theorem product_of_seven_and_sum (x : ℝ) : 27 - 7 = x * 5 → 7 * (x + 5) = 63 := by
  sorry

end NUMINAMATH_CALUDE_product_of_seven_and_sum_l3312_331281


namespace NUMINAMATH_CALUDE_left_handed_fraction_conference_l3312_331274

/-- Represents the fraction of left-handed participants for each country type -/
structure LeftHandedFractions where
  red : ℚ
  blue : ℚ
  green : ℚ
  yellow : ℚ

/-- Represents the ratio of participants from each country type -/
structure ParticipantRatio where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the fraction of left-handed participants given the ratio of participants
    and the fractions of left-handed participants for each country type -/
def leftHandedFraction (ratio : ParticipantRatio) (fractions : LeftHandedFractions) : ℚ :=
  (ratio.red * fractions.red + ratio.blue * fractions.blue +
   ratio.green * fractions.green + ratio.yellow * fractions.yellow) /
  (ratio.red + ratio.blue + ratio.green + ratio.yellow)

theorem left_handed_fraction_conference :
  let ratio : ParticipantRatio := ⟨10, 5, 3, 2⟩
  let fractions : LeftHandedFractions := ⟨37/100, 61/100, 26/100, 48/100⟩
  leftHandedFraction ratio fractions = 849/2000 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_fraction_conference_l3312_331274


namespace NUMINAMATH_CALUDE_part_one_part_two_l3312_331200

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f a x ≤ 4) → 
  -1 ≤ a ∧ a ≤ 2 :=
sorry

-- Part II
theorem part_two (a : ℝ) :
  (∃ x : ℝ, f a (x - a) - f a (x + a) ≤ 2 * a - 1) → 
  a ≥ 1/4 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3312_331200


namespace NUMINAMATH_CALUDE_uncolored_area_rectangle_with_circles_l3312_331221

/-- The uncolored area of a rectangle with four tangent circles --/
theorem uncolored_area_rectangle_with_circles (w h r : Real) 
  (hw : w = 30) 
  (hh : h = 50) 
  (hr : r = w / 4) 
  (circles_fit : 4 * r = w) 
  (circles_tangent : 2 * r = h / 2) : 
  w * h - 4 * Real.pi * r^2 = 1500 - 225 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_uncolored_area_rectangle_with_circles_l3312_331221


namespace NUMINAMATH_CALUDE_didi_fundraiser_total_l3312_331295

/-- Calculates the total amount raised by Didi for her local soup kitchen --/
theorem didi_fundraiser_total (num_cakes : ℕ) (slices_per_cake : ℕ) (price_per_slice : ℚ) 
  (donation1 : ℚ) (donation2 : ℚ) (donation3 : ℚ) (donation4 : ℚ) :
  num_cakes = 20 →
  slices_per_cake = 12 →
  price_per_slice = 1 →
  donation1 = 3/4 →
  donation2 = 1/2 →
  donation3 = 1/4 →
  donation4 = 1/10 →
  (num_cakes * slices_per_cake * price_per_slice) + 
  (num_cakes * slices_per_cake * (donation1 + donation2 + donation3 + donation4)) = 624 := by
sorry

end NUMINAMATH_CALUDE_didi_fundraiser_total_l3312_331295


namespace NUMINAMATH_CALUDE_six_year_olds_count_l3312_331211

/-- Represents the number of children in each age group -/
structure AgeGroups where
  three_year_olds : ℕ
  four_year_olds : ℕ
  five_year_olds : ℕ
  six_year_olds : ℕ

/-- Represents the Sunday school with its age groups and class information -/
structure SundaySchool where
  ages : AgeGroups
  avg_class_size : ℕ
  num_classes : ℕ

def SundaySchool.total_children (s : SundaySchool) : ℕ :=
  s.ages.three_year_olds + s.ages.four_year_olds + s.ages.five_year_olds + s.ages.six_year_olds

theorem six_year_olds_count (s : SundaySchool) 
  (h1 : s.ages.three_year_olds = 13)
  (h2 : s.ages.four_year_olds = 20)
  (h3 : s.ages.five_year_olds = 15)
  (h4 : s.avg_class_size = 35)
  (h5 : s.num_classes = 2)
  : s.ages.six_year_olds = 22 := by
  sorry

#check six_year_olds_count

end NUMINAMATH_CALUDE_six_year_olds_count_l3312_331211


namespace NUMINAMATH_CALUDE_m_range_l3312_331216

def p (x : ℝ) : Prop := x^2 - 2*x - 8 ≤ 0

def q (x m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) ≤ 0

theorem m_range (m : ℝ) :
  (m < 0) →
  (∀ x, p x → q x m) →
  m ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3312_331216


namespace NUMINAMATH_CALUDE_sqrt_three_subset_M_l3312_331299

def M : Set ℝ := {x | x ≤ 3}

theorem sqrt_three_subset_M : {Real.sqrt 3} ⊆ M := by sorry

end NUMINAMATH_CALUDE_sqrt_three_subset_M_l3312_331299


namespace NUMINAMATH_CALUDE_non_zero_terms_count_l3312_331232

/-- The expression to be expanded and simplified -/
def expression (x : ℝ) : ℝ := (x - 3) * (x^2 + 5*x + 8) + 2 * (x^3 + 3*x^2 - x - 4)

/-- The expanded and simplified form of the expression -/
def simplified_expression (x : ℝ) : ℝ := 3*x^3 + 8*x^2 - 9*x - 32

/-- Theorem stating that the number of non-zero terms in the simplified expression is 4 -/
theorem non_zero_terms_count : 
  (∃ (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0), 
    ∀ x, simplified_expression x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ (a b c d e : ℝ), ¬(∀ x, simplified_expression x = a*x^4 + b*x^3 + c*x^2 + d*x + e)) :=
sorry

end NUMINAMATH_CALUDE_non_zero_terms_count_l3312_331232


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l3312_331244

/-- The amount of material Cheryl used for her project -/
def material_used (material1 material2 leftover : ℚ) : ℚ :=
  material1 + material2 - leftover

/-- Theorem stating the total amount of material Cheryl used -/
theorem cheryl_material_usage :
  let material1 : ℚ := 4/9
  let material2 : ℚ := 2/3
  let leftover : ℚ := 8/18
  material_used material1 material2 leftover = 2/3 :=
by
  sorry

#check cheryl_material_usage

end NUMINAMATH_CALUDE_cheryl_material_usage_l3312_331244


namespace NUMINAMATH_CALUDE_jones_family_probability_l3312_331255

theorem jones_family_probability :
  let n : ℕ := 8  -- total number of children
  let k : ℕ := 4  -- number of sons (or daughters)
  let p : ℚ := 1/2  -- probability of a child being a son (or daughter)
  Nat.choose n k * p^k * (1-p)^(n-k) = 35/128 :=
by sorry

end NUMINAMATH_CALUDE_jones_family_probability_l3312_331255


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3312_331237

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3312_331237


namespace NUMINAMATH_CALUDE_line_properties_l3312_331225

def line_equation (x y : ℝ) : Prop := y = -x + 5

theorem line_properties :
  let angle_with_ox : ℝ := 135
  let intersection_point : ℝ × ℝ := (0, 5)
  let point_A : ℝ × ℝ := (2, 3)
  let point_B : ℝ × ℝ := (2, -3)
  (∀ x y, line_equation x y → 
    (Real.tan (angle_with_ox * π / 180) = -1 ∧ 
     line_equation (intersection_point.1) (intersection_point.2))) ∧
  line_equation point_A.1 point_A.2 ∧
  ¬line_equation point_B.1 point_B.2 :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l3312_331225


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3312_331290

/-- A line in the form y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Checks if a line has equal intercepts on the coordinate axes -/
def hasEqualIntercepts (l : Line) : Prop :=
  ∃ c : ℝ, c ≠ 0 ∧ l.k * c + l.b = -c ∧ l.b = c

/-- The specific line y = kx + 2k - 1 -/
def specificLine (k : ℝ) : Line :=
  { k := k, b := 2 * k - 1 }

/-- The condition k = -1 is sufficient but not necessary for the line to have equal intercepts -/
theorem sufficient_not_necessary :
  (∀ k : ℝ, k = -1 → hasEqualIntercepts (specificLine k)) ∧
  (∃ k : ℝ, k ≠ -1 ∧ hasEqualIntercepts (specificLine k)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3312_331290


namespace NUMINAMATH_CALUDE_millet_majority_on_sixth_day_l3312_331298

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  totalSeeds : ℚ
  milletSeeds : ℚ

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  let newTotalSeeds := state.totalSeeds + 2^(state.day - 1) / 2
  let newMilletSeeds := state.milletSeeds / 2 + 0.4 * 2^(state.day - 1) / 2
  { day := state.day + 1, totalSeeds := newTotalSeeds, milletSeeds := newMilletSeeds }

/-- The initial state of the feeder -/
def initialState : FeederState :=
  { day := 1, totalSeeds := 1/2, milletSeeds := 0.2 }

/-- Calculates the state of the feeder after n days -/
def stateAfterDays (n : Nat) : FeederState :=
  match n with
  | 0 => initialState
  | m + 1 => nextDay (stateAfterDays m)

/-- Theorem: On the 6th day, more than half of the seeds are millet -/
theorem millet_majority_on_sixth_day :
  let sixthDay := stateAfterDays 5
  sixthDay.milletSeeds > sixthDay.totalSeeds / 2 := by
  sorry

end NUMINAMATH_CALUDE_millet_majority_on_sixth_day_l3312_331298


namespace NUMINAMATH_CALUDE_negative_x_gt_1_is_inequality_l3312_331236

-- Define what an inequality is
def is_inequality (expr : Prop) : Prop :=
  ∃ (a b : ℝ), (expr = (a > b) ∨ expr = (a < b) ∨ expr = (a ≥ b) ∨ expr = (a ≤ b))

-- Theorem to prove
theorem negative_x_gt_1_is_inequality :
  is_inequality (-x > 1) :=
sorry

end NUMINAMATH_CALUDE_negative_x_gt_1_is_inequality_l3312_331236


namespace NUMINAMATH_CALUDE_digit_property_l3312_331262

theorem digit_property : ∃! (x : ℕ), x < 10 ∧ ∀ (a : ℕ), 10 * a + x = a + x + a * x := by
  sorry

end NUMINAMATH_CALUDE_digit_property_l3312_331262


namespace NUMINAMATH_CALUDE_thomas_score_l3312_331210

theorem thomas_score (n : ℕ) (avg_without_thomas avg_with_thomas thomas_score : ℚ) :
  n = 20 →
  avg_without_thomas = 78 →
  avg_with_thomas = 80 →
  (n - 1) * avg_without_thomas + thomas_score = n * avg_with_thomas →
  thomas_score = 118 := by
sorry

end NUMINAMATH_CALUDE_thomas_score_l3312_331210


namespace NUMINAMATH_CALUDE_expression_simplification_l3312_331213

theorem expression_simplification (x y z : ℝ) :
  (x - 3 * (y * z)) - ((x - 3 * y) * z) = -x * z := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3312_331213


namespace NUMINAMATH_CALUDE_coefficient_of_x5_in_expansion_l3312_331234

/-- The coefficient of x^5 in the binomial expansion of (x^2 - 2/x)^7 -/
def coefficientOfX5 : ℤ := -280

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_of_x5_in_expansion :
  coefficientOfX5 = (-2)^3 * binomial 7 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x5_in_expansion_l3312_331234


namespace NUMINAMATH_CALUDE_face_mask_profit_l3312_331205

/-- Calculate the total profit from selling face masks --/
theorem face_mask_profit (num_boxes : ℕ) (masks_per_box : ℕ) (total_cost : ℚ) (selling_price : ℚ) :
  num_boxes = 3 →
  masks_per_box = 20 →
  total_cost = 15 →
  selling_price = 1/2 →
  (num_boxes * masks_per_box : ℚ) * selling_price - total_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_face_mask_profit_l3312_331205


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_compare_expressions_l3312_331266

-- Problem 1
theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/(y+1) = 2) : 
  ∀ x' y' : ℝ, x' > 0 → y' > 0 → 1/x' + 2/(y'+1) = 2 → 2*x + y ≤ 2*x' + y' :=
sorry

-- Problem 2
theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : a + b = 1) : 
  8 - 1/a ≤ 1/b + 1/(a*b) :=
sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_compare_expressions_l3312_331266


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3312_331270

theorem unique_solution_trigonometric_equation :
  ∃! (x : ℝ), 0 < x ∧ x < 1 ∧ Real.sin (Real.arccos (Real.tan (Real.arcsin x))) = x :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3312_331270


namespace NUMINAMATH_CALUDE_bug_return_probability_l3312_331207

/-- Represents the probability of the bug being at its starting vertex after n moves -/
def P (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (1 - P (n - 1)) / 2

/-- The main theorem stating the probability of returning to the starting vertex on the 12th move -/
theorem bug_return_probability : P 12 = 683 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l3312_331207


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3312_331264

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, ax^2 + x + b > 0 ↔ 1 < x ∧ x < 2) →
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3312_331264


namespace NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l3312_331203

theorem equidistant_point_on_x_axis :
  ∃ x : ℝ,
    (x^2 + 4*x + 4 = x^2 + 16) ∧
    (∀ y : ℝ, y ≠ x → (y^2 + 4*y + 4 ≠ y^2 + 16)) →
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l3312_331203


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l3312_331249

theorem complex_magnitude_equation (t : ℝ) : 
  0 < t → t < 4 → Complex.abs (t + 3 * Complex.I * Real.sqrt 2) * Complex.abs (7 - 5 * Complex.I) = 35 * Real.sqrt 2 → 
  t = Real.sqrt (559 / 37) := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l3312_331249


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l3312_331293

theorem reciprocal_sum_theorem :
  (∀ (a b c : ℕ+), (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ ≠ 9/11) ∧
  (∀ (a b c : ℕ+), (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ > 41/42 →
    (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ ≥ 1) := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l3312_331293


namespace NUMINAMATH_CALUDE_a_lt_one_necessary_not_sufficient_for_a_squared_lt_one_l3312_331251

theorem a_lt_one_necessary_not_sufficient_for_a_squared_lt_one :
  (∀ a : ℝ, a^2 < 1 → a < 1) ∧
  (∃ a : ℝ, a < 1 ∧ a^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_a_lt_one_necessary_not_sufficient_for_a_squared_lt_one_l3312_331251


namespace NUMINAMATH_CALUDE_bread_slice_cost_l3312_331256

/-- Given the conditions of Tim's bread purchase, prove that each slice costs 40 cents. -/
theorem bread_slice_cost :
  let num_loaves : ℕ := 3
  let slices_per_loaf : ℕ := 20
  let payment : ℕ := 2 * 20
  let change : ℕ := 16
  let total_cost : ℕ := payment - change
  let total_slices : ℕ := num_loaves * slices_per_loaf
  let cost_per_slice : ℚ := total_cost / total_slices
  cost_per_slice * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bread_slice_cost_l3312_331256


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l3312_331267

def n : ℕ := 2^15 - 1

theorem greatest_prime_divisor_digit_sum :
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧
    (∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) ∧
    (Nat.digits 10 p).sum = 8) :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l3312_331267


namespace NUMINAMATH_CALUDE_scientific_notation_308000000_l3312_331279

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_308000000 :
  to_scientific_notation 308000000 = ScientificNotation.mk 3.08 8 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_308000000_l3312_331279


namespace NUMINAMATH_CALUDE_third_term_geometric_sequence_l3312_331271

theorem third_term_geometric_sequence
  (q : ℝ)
  (h_q_abs : |q| < 1)
  (h_sum : (a : ℕ → ℝ) → (∀ n, a (n + 1) = q * a n) → (∑' n, a n) = 8/5)
  (h_second_term : ∃ a : ℕ → ℝ, (∀ n, a (n + 1) = q * a n) ∧ a 1 = -1/2) :
  ∃ a : ℕ → ℝ, (∀ n, a (n + 1) = q * a n) ∧ a 1 = -1/2 ∧ a 2 = 1/8 :=
sorry

end NUMINAMATH_CALUDE_third_term_geometric_sequence_l3312_331271


namespace NUMINAMATH_CALUDE_average_height_is_12_l3312_331224

def plant_heights (h1 h2 h3 h4 : ℝ) : Prop :=
  h1 = 27 ∧ h3 = 9 ∧
  ((h2 = h1 / 3 ∨ h2 = h1 * 3) ∧
   (h3 = h2 / 3 ∨ h3 = h2 * 3) ∧
   (h4 = h3 / 3 ∨ h4 = h3 * 3))

theorem average_height_is_12 (h1 h2 h3 h4 : ℝ) :
  plant_heights h1 h2 h3 h4 → (h1 + h2 + h3 + h4) / 4 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_average_height_is_12_l3312_331224


namespace NUMINAMATH_CALUDE_pete_ran_least_l3312_331212

-- Define the set of runners
inductive Runner
| Phil
| Tom
| Pete
| Amal
| Sanjay

-- Define a function that maps each runner to their distance run
def distance : Runner → ℝ
| Runner.Phil => 4
| Runner.Tom => 6
| Runner.Pete => 2
| Runner.Amal => 8
| Runner.Sanjay => 7

-- Theorem: Pete ran the least distance
theorem pete_ran_least : ∀ r : Runner, distance Runner.Pete ≤ distance r :=
by sorry

end NUMINAMATH_CALUDE_pete_ran_least_l3312_331212


namespace NUMINAMATH_CALUDE_wellness_gym_ratio_l3312_331217

theorem wellness_gym_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) :
  (35 : ℝ) * f + 30 * m = 32 * (f + m) →
  (f : ℝ) / m = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_wellness_gym_ratio_l3312_331217


namespace NUMINAMATH_CALUDE_equal_numbers_product_l3312_331227

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 22 →
  b = 18 →
  c = 32 →
  d = e →
  d * e = 196 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l3312_331227


namespace NUMINAMATH_CALUDE_divide_six_a_squared_by_half_a_l3312_331233

theorem divide_six_a_squared_by_half_a (a : ℝ) (h : a ≠ 0) : 6 * a^2 / (a / 2) = 12 * a := by
  sorry

end NUMINAMATH_CALUDE_divide_six_a_squared_by_half_a_l3312_331233


namespace NUMINAMATH_CALUDE_all_transformed_points_in_S_l3312_331275

def S : Set ℂ := {z | -1 ≤ z.re ∧ z.re ≤ 1 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

theorem all_transformed_points_in_S : ∀ z ∈ S, (1/2 + 1/2*I)*z ∈ S := by
  sorry

end NUMINAMATH_CALUDE_all_transformed_points_in_S_l3312_331275


namespace NUMINAMATH_CALUDE_intersection_equality_l3312_331242

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x - 5 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x - 1 = 0}

-- State the theorem
theorem intersection_equality (a : ℝ) : A ∩ B a = B a → a = 0 ∨ a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l3312_331242


namespace NUMINAMATH_CALUDE_pears_picked_total_l3312_331202

/-- The number of pears Alyssa picked -/
def alyssa_pears : ℕ := 42

/-- The number of pears Nancy picked -/
def nancy_pears : ℕ := 17

/-- The total number of pears picked -/
def total_pears : ℕ := alyssa_pears + nancy_pears

theorem pears_picked_total : total_pears = 59 := by
  sorry

end NUMINAMATH_CALUDE_pears_picked_total_l3312_331202


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_6_or_8_l3312_331294

/-- The number of four-digit numbers -/
def total_four_digit_numbers : ℕ := 9000

/-- The number of digits that are not 6 or 8 for the first digit -/
def first_digit_choices : ℕ := 7

/-- The number of digits that are not 6 or 8 for the other digits -/
def other_digit_choices : ℕ := 8

/-- The number of four-digit numbers without 6 or 8 -/
def numbers_without_6_or_8 : ℕ := first_digit_choices * other_digit_choices * other_digit_choices * other_digit_choices

theorem four_digit_numbers_with_6_or_8 :
  total_four_digit_numbers - numbers_without_6_or_8 = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_6_or_8_l3312_331294


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3312_331246

/-- Given that the solution set of ax^2 + x + b > 0 with respect to x is (-1, 2), prove that a + b = 1 -/
theorem quadratic_inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + x + b > 0 ↔ -1 < x ∧ x < 2) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3312_331246


namespace NUMINAMATH_CALUDE_john_jury_duty_days_l3312_331253

/-- Calculates the total number of days spent on jury duty given the specified conditions. -/
def juryDutyDays (jurySelectionDays : ℕ) (trialMultiplier : ℕ) (deliberationFullDays : ℕ) (deliberationHoursPerDay : ℕ) : ℕ :=
  let trialDays := jurySelectionDays * trialMultiplier
  let deliberationHours := deliberationFullDays * deliberationHoursPerDay
  let deliberationDays := deliberationHours / 24
  jurySelectionDays + trialDays + deliberationDays

/-- Theorem stating that under the given conditions, John spends 14 days on jury duty. -/
theorem john_jury_duty_days :
  juryDutyDays 2 4 6 16 = 14 := by
  sorry

#eval juryDutyDays 2 4 6 16

end NUMINAMATH_CALUDE_john_jury_duty_days_l3312_331253


namespace NUMINAMATH_CALUDE_isosceles_triangles_angle_l3312_331257

/-- Isosceles triangle -/
structure IsoscelesTriangle (P Q R : ℝ × ℝ) :=
  (isosceles : dist P Q = dist Q R)

/-- Similar triangles -/
def SimilarTriangles (P Q R P' Q' R' : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    dist P Q = k * dist P' Q' ∧
    dist Q R = k * dist Q' R' ∧
    dist R P = k * dist R' P'

/-- Point lies on line segment -/
def OnSegment (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

/-- Point lies on line extension -/
def OnExtension (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ P = (1 - t) • A + t • B

/-- Perpendicular line segments -/
def Perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (R.1 - P.1) * (S.1 - Q.1) + (R.2 - P.2) * (S.2 - Q.2) = 0

/-- Angle measure -/
def AngleMeasure (P Q R : ℝ × ℝ) : ℝ :=
  sorry

theorem isosceles_triangles_angle (A B C A₁ B₁ C₁ : ℝ × ℝ) :
  IsoscelesTriangle A B C →
  IsoscelesTriangle A₁ B₁ C₁ →
  SimilarTriangles A B C A₁ B₁ C₁ →
  dist A C / dist A₁ C₁ = 5 / Real.sqrt 3 →
  OnSegment A₁ A C →
  OnSegment B₁ B C →
  OnExtension C₁ A B →
  Perpendicular A₁ B₁ B C →
  AngleMeasure A B C = 120 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_angle_l3312_331257


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3312_331296

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^4 - 18 * x^3 + 6 * x^2 - 4 * x + 30) % (2 * x - 4) = 30 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3312_331296


namespace NUMINAMATH_CALUDE_negation_equivalence_l3312_331282

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 8*x + 18 < 0) ↔ (∀ x : ℝ, x^2 - 8*x + 18 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3312_331282


namespace NUMINAMATH_CALUDE_total_crayons_l3312_331241

/-- The number of crayons each person has -/
structure CrayonCounts where
  wanda : ℕ
  dina : ℕ
  jacob : ℕ
  emma : ℕ
  xavier : ℕ
  hannah : ℕ

/-- The conditions of the problem -/
def crayon_problem (c : CrayonCounts) : Prop :=
  c.wanda = 62 ∧
  c.dina = 28 ∧
  c.jacob = c.dina - 2 ∧
  c.emma = 2 * c.wanda - 3 ∧
  c.xavier = ((c.jacob + c.dina) / 2) ^ 3 - 7 ∧
  c.hannah = (c.wanda + c.dina + c.jacob + c.emma + c.xavier) / 5

/-- The theorem to be proved -/
theorem total_crayons (c : CrayonCounts) : 
  crayon_problem c → c.wanda + c.dina + c.jacob + c.emma + c.xavier + c.hannah = 23895 := by
  sorry


end NUMINAMATH_CALUDE_total_crayons_l3312_331241


namespace NUMINAMATH_CALUDE_sons_age_l3312_331214

/-- Proves that given the conditions, the son's present age is 33 years. -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3312_331214


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l3312_331291

theorem unique_solution_sqrt_equation :
  ∀ x y : ℕ,
    x ≥ 1 →
    y ≥ 1 →
    y ≥ x →
    (Real.sqrt (2 * x) - 1) * (Real.sqrt (2 * y) - 1) = 1 →
    x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l3312_331291


namespace NUMINAMATH_CALUDE_mean_height_is_70_74_l3312_331223

def player_heights : List ℕ := [58, 59, 60, 61, 62, 63, 65, 65, 68, 70, 71, 74, 76, 78, 79, 81, 83, 85, 86]

def mean_height (heights : List ℕ) : ℚ :=
  (heights.sum : ℚ) / heights.length

theorem mean_height_is_70_74 :
  mean_height player_heights = 70.74 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_is_70_74_l3312_331223


namespace NUMINAMATH_CALUDE_barbara_candies_l3312_331268

/-- The number of candies Barbara bought -/
def candies_bought (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem barbara_candies :
  candies_bought 9 27 = 18 :=
by sorry

end NUMINAMATH_CALUDE_barbara_candies_l3312_331268


namespace NUMINAMATH_CALUDE_parabola_c_value_l3312_331252

theorem parabola_c_value (a b c : ℚ) :
  (∀ y : ℚ, -3 = a * 1^2 + b * 1 + c) →
  (∀ y : ℚ, -6 = a * 3^2 + b * 3 + c) →
  c = -15/4 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3312_331252


namespace NUMINAMATH_CALUDE_tetrahedron_similarity_counterexample_l3312_331273

/-- A tetrahedron with equilateral triangle base and three other sides --/
structure Tetrahedron :=
  (base : ℝ)
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Two triangular faces are similar --/
def similar_faces (t1 t2 : Tetrahedron) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ 
    ((t1.side1 = k * t2.side1 ∧ t1.side2 = k * t2.side2) ∨
     (t1.side1 = k * t2.side2 ∧ t1.side2 = k * t2.side3) ∨
     (t1.side1 = k * t2.side3 ∧ t1.side2 = k * t2.base) ∨
     (t1.side2 = k * t2.side1 ∧ t1.side3 = k * t2.side2) ∨
     (t1.side2 = k * t2.side2 ∧ t1.side3 = k * t2.side3) ∨
     (t1.side2 = k * t2.side3 ∧ t1.side3 = k * t2.base) ∨
     (t1.side3 = k * t2.side1 ∧ t1.base = k * t2.side2) ∨
     (t1.side3 = k * t2.side2 ∧ t1.base = k * t2.side3) ∨
     (t1.side3 = k * t2.side3 ∧ t1.base = k * t2.base))

/-- Two tetrahedrons are similar --/
def similar_tetrahedrons (t1 t2 : Tetrahedron) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ 
    t1.base = k * t2.base ∧
    t1.side1 = k * t2.side1 ∧
    t1.side2 = k * t2.side2 ∧
    t1.side3 = k * t2.side3

/-- The main theorem --/
theorem tetrahedron_similarity_counterexample :
  ∃ (t1 t2 : Tetrahedron),
    (∀ (f1 f2 : Tetrahedron → Tetrahedron → Prop),
      (f1 t1 t1 → f2 t1 t1 → f1 = f2) ∧
      (f1 t2 t2 → f2 t2 t2 → f1 = f2)) ∧
    (∀ (f1 : Tetrahedron → Tetrahedron → Prop),
      (f1 t1 t1 → ∃ (f2 : Tetrahedron → Tetrahedron → Prop), f2 t2 t2 ∧ f1 = f2)) ∧
    ¬(similar_tetrahedrons t1 t2) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_similarity_counterexample_l3312_331273


namespace NUMINAMATH_CALUDE_matrix_transformation_l3312_331285

theorem matrix_transformation (P Q : Matrix (Fin 3) (Fin 3) ℝ) : 
  P = !![3, 0, 0; 0, 0, 1; 0, 1, 0] → 
  (∀ a b c d e f g h i : ℝ, 
    Q = !![a, b, c; d, e, f; g, h, i] → 
    P * Q = !![3*a, 3*b, 3*c; g, h, i; d, e, f]) :=
by sorry

end NUMINAMATH_CALUDE_matrix_transformation_l3312_331285


namespace NUMINAMATH_CALUDE_max_cookies_andy_l3312_331238

/-- Represents the number of cookies eaten by each sibling -/
structure CookieDistribution where
  andy : ℕ
  alexa : ℕ
  john : ℕ

/-- Checks if a cookie distribution is valid according to the problem conditions -/
def isValidDistribution (d : CookieDistribution) : Prop :=
  d.alexa = 2 * d.andy + 2 ∧
  d.john = d.andy - 3 ∧
  d.andy + d.alexa + d.john = 30

/-- Theorem stating that the maximum number of cookies Andy can eat is 7 -/
theorem max_cookies_andy :
  ∀ d : CookieDistribution, isValidDistribution d → d.andy ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_andy_l3312_331238


namespace NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l3312_331280

theorem quadratic_reciprocal_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x * y = 1 ∧ x^2 - 2*(m+2)*x + m^2 - 4 = 0 ∧ y^2 - 2*(m+2)*y + m^2 - 4 = 0) 
  → m = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l3312_331280


namespace NUMINAMATH_CALUDE_candy_distribution_l3312_331208

theorem candy_distribution (total_candies : ℕ) (sour_percentage : ℚ) (num_people : ℕ) : 
  total_candies = 300 → 
  sour_percentage = 40 / 100 → 
  num_people = 3 → 
  (total_candies - (sour_percentage * total_candies).floor) / num_people = 60 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3312_331208


namespace NUMINAMATH_CALUDE_range_of_a_l3312_331204

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (a * x^2 - x + 1/(16*a))

def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → Real.sqrt (2*x + 1) < 1 + a*x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → a ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3312_331204


namespace NUMINAMATH_CALUDE_fraction_simplification_l3312_331263

theorem fraction_simplification :
  ((5^1004)^4 - (5^1002)^4) / ((5^1003)^4 - (5^1001)^4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3312_331263


namespace NUMINAMATH_CALUDE_trace_bag_count_is_five_l3312_331284

/-- The weight of one of Gordon's shopping bags in pounds -/
def gordon_bag1_weight : ℕ := 3

/-- The weight of the other of Gordon's shopping bags in pounds -/
def gordon_bag2_weight : ℕ := 7

/-- The weight of each of Trace's shopping bags in pounds -/
def trace_bag_weight : ℕ := 2

/-- The number of Trace's shopping bags -/
def trace_bag_count : ℕ := (gordon_bag1_weight + gordon_bag2_weight) / trace_bag_weight

theorem trace_bag_count_is_five : trace_bag_count = 5 := by
  sorry

#eval trace_bag_count

end NUMINAMATH_CALUDE_trace_bag_count_is_five_l3312_331284


namespace NUMINAMATH_CALUDE_largest_difference_l3312_331254

def U : ℕ := 3 * 2005^2006
def V : ℕ := 2005^2006
def W : ℕ := 2004 * 2005^2005
def X : ℕ := 3 * 2005^2005
def Y : ℕ := 2005^2005
def Z : ℕ := 2005^2004

theorem largest_difference : 
  (U - V > V - W) ∧ (U - V > W - X) ∧ (U - V > X - Y) ∧ (U - V > Y - Z) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l3312_331254


namespace NUMINAMATH_CALUDE_fish_in_pond_l3312_331235

theorem fish_in_pond (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) 
  (h1 : initial_tagged = 30)
  (h2 : second_catch = 50)
  (h3 : tagged_in_second = 2)
  (h4 : (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish) :
  total_fish = 750 :=
by sorry

#check fish_in_pond

end NUMINAMATH_CALUDE_fish_in_pond_l3312_331235


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l3312_331240

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 → b > 0 → -- Angles are positive
  a + b + 90 = 180 → -- Sum of angles in a triangle
  a / b = 7 / 2 → -- Ratio of acute angles
  (a = 70 ∧ b = 20) ∨ (a = 20 ∧ b = 70) := by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l3312_331240


namespace NUMINAMATH_CALUDE_smallest_gcd_yz_l3312_331269

theorem smallest_gcd_yz (x y z : ℕ+) (h1 : Nat.gcd x y = 210) (h2 : Nat.gcd x z = 770) :
  ∃ (y' z' : ℕ+), Nat.gcd x y' = 210 ∧ Nat.gcd x z' = 770 ∧ Nat.gcd y' z' = 10 ∧
  ∀ (y'' z'' : ℕ+), Nat.gcd x y'' = 210 → Nat.gcd x z'' = 770 → Nat.gcd y'' z'' ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_yz_l3312_331269


namespace NUMINAMATH_CALUDE_perfect_square_pair_iff_in_solution_set_l3312_331231

/-- A pair of integers (a, b) satisfies the perfect square property if
    a^2 + 4b and b^2 + 4a are both perfect squares. -/
def PerfectSquarePair (a b : ℤ) : Prop :=
  ∃ (m n : ℤ), a^2 + 4*b = m^2 ∧ b^2 + 4*a = n^2

/-- The set of solutions for the perfect square pair problem. -/
def SolutionSet : Set (ℤ × ℤ) :=
  {p | ∃ (k : ℤ), p = (k^2, 0) ∨ p = (0, k^2) ∨ p = (k, 1-k) ∨
                   p = (-6, -5) ∨ p = (-5, -6) ∨ p = (-4, -4)}

/-- The main theorem stating that a pair (a, b) satisfies the perfect square property
    if and only if it belongs to the solution set. -/
theorem perfect_square_pair_iff_in_solution_set (a b : ℤ) :
  PerfectSquarePair a b ↔ (a, b) ∈ SolutionSet := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_pair_iff_in_solution_set_l3312_331231


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3312_331292

theorem book_arrangement_count :
  let total_books : ℕ := 9
  let arabic_books : ℕ := 2
  let german_books : ℕ := 3
  let spanish_books : ℕ := 4
  let arabic_unit : ℕ := 1
  let spanish_unit : ℕ := 1
  let total_units : ℕ := arabic_unit + spanish_unit + german_books

  (total_books = arabic_books + german_books + spanish_books) →
  (Nat.factorial total_units * Nat.factorial arabic_books * Nat.factorial spanish_books = 5760) :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3312_331292


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3312_331250

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y * y = x * z

/-- The main theorem -/
theorem arithmetic_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence (a 1) (a 3) (a 9)) :
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3312_331250


namespace NUMINAMATH_CALUDE_ratio_of_x_to_y_l3312_331247

theorem ratio_of_x_to_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 9) (h4 : y = 0.5) :
  x / y = 36 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_to_y_l3312_331247


namespace NUMINAMATH_CALUDE_cube_sum_over_product_l3312_331297

theorem cube_sum_over_product (x y z : ℝ) :
  ((x - y)^3 + (y - z)^3 + (z - x)^3) / (15 * (x - y) * (y - z) * (z - x)) = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_l3312_331297


namespace NUMINAMATH_CALUDE_polynomial_ratio_l3312_331260

/-- Given a polynomial ax^4 + bx^3 + cx^2 + dx + e = 0 with roots 1, 2, 3, and 4,
    prove that c/e = 35/24 -/
theorem polynomial_ratio (a b c d e : ℝ) (h : ∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) :
  c / e = 35 / 24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_ratio_l3312_331260


namespace NUMINAMATH_CALUDE_coefficient_m5n3_in_expansion_l3312_331215

theorem coefficient_m5n3_in_expansion : ∀ m n : ℕ,
  (Nat.choose 8 5 : ℕ) = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_m5n3_in_expansion_l3312_331215


namespace NUMINAMATH_CALUDE_proposition_implications_l3312_331218

theorem proposition_implications (p q : Prop) 
  (h : ¬(¬p ∨ ¬q)) : (p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_implications_l3312_331218


namespace NUMINAMATH_CALUDE_largest_sum_is_1803_l3312_331287

/-- The set of digits to be used -/
def digits : Finset Nat := {1, 2, 3, 7, 8, 9}

/-- A function that computes the sum of two 3-digit numbers -/
def sum_3digit (a b c d e f : Nat) : Nat :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The theorem stating that 1803 is the largest possible sum -/
theorem largest_sum_is_1803 :
  ∀ a b c d e f : Nat,
    a ∈ digits → b ∈ digits → c ∈ digits →
    d ∈ digits → e ∈ digits → f ∈ digits →
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f →
    b ≠ c → b ≠ d → b ≠ e → b ≠ f →
    c ≠ d → c ≠ e → c ≠ f →
    d ≠ e → d ≠ f →
    e ≠ f →
    sum_3digit a b c d e f ≤ 1803 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_sum_is_1803_l3312_331287


namespace NUMINAMATH_CALUDE_digit_sum_equation_l3312_331220

/-- Given that a000 + a998 + a999 = 22997, prove that a = 7 -/
theorem digit_sum_equation (a : ℕ) : 
  a * 1000 + a * 998 + a * 999 = 22997 → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_equation_l3312_331220


namespace NUMINAMATH_CALUDE_horner_rule_v4_horner_rule_correct_l3312_331289

def horner_polynomial (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 20*x^3 - 8*x^2 + 35*x + 12

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 20
  v3 * x - 8

theorem horner_rule_v4 :
  horner_v4 (-2) = -16 :=
by sorry

theorem horner_rule_correct :
  horner_v4 (-2) = horner_polynomial (-2) :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v4_horner_rule_correct_l3312_331289


namespace NUMINAMATH_CALUDE_equilateral_triangle_hexagon_area_l3312_331283

theorem equilateral_triangle_hexagon_area (s t : ℝ) : 
  s > 0 → t > 0 → -- Ensure positive side lengths
  3 * s = 6 * t → -- Equal perimeters
  (s^2 * Real.sqrt 3) / 4 = 9 → -- Triangle area is 9
  (3 * t^2 * Real.sqrt 3) / 2 = 13.5 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_hexagon_area_l3312_331283


namespace NUMINAMATH_CALUDE_proportional_function_and_value_l3312_331239

/-- Given that y+3 is directly proportional to x and y=7 when x=2, prove:
    1. The function expression for y in terms of x
    2. The value of y when x = -1/2 -/
theorem proportional_function_and_value (y : ℝ → ℝ) (k : ℝ) 
    (h1 : ∀ x, y x + 3 = k * x)  -- y+3 is directly proportional to x
    (h2 : y 2 = 7)  -- when x=2, y=7
    : (∀ x, y x = 5*x - 3) ∧ (y (-1/2) = -11/2) := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_and_value_l3312_331239


namespace NUMINAMATH_CALUDE_probability_letter_in_mathematical_l3312_331276

def alphabet : Finset Char := sorry

def mathematical : String := "MATHEMATICAL"

theorem probability_letter_in_mathematical :
  let unique_letters := mathematical.toList.toFinset
  (unique_letters.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 := by sorry

end NUMINAMATH_CALUDE_probability_letter_in_mathematical_l3312_331276


namespace NUMINAMATH_CALUDE_max_intersections_l3312_331286

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane, represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The configuration of figures on the plane. -/
structure Configuration where
  circle : Circle
  lines : Fin 3 → Line

/-- The number of intersection points between a circle and a line. -/
def circleLineIntersections (c : Circle) (l : Line) : ℕ := sorry

/-- The number of intersection points between two lines. -/
def lineLineIntersections (l1 l2 : Line) : ℕ := sorry

/-- The total number of intersection points in a configuration. -/
def totalIntersections (config : Configuration) : ℕ := sorry

/-- The theorem stating that the maximum number of intersections is 9. -/
theorem max_intersections :
  ∃ (config : Configuration), totalIntersections config = 9 ∧
  ∀ (other : Configuration), totalIntersections other ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_l3312_331286


namespace NUMINAMATH_CALUDE_sweetest_sugar_water_l3312_331229

-- Define the initial sugar water concentration
def initial_concentration : ℚ := 25 / 125

-- Define Student A's final concentration (remains the same)
def concentration_A : ℚ := initial_concentration

-- Define Student B's added solution
def added_solution_B : ℚ := 20 / 50

-- Define Student C's added solution
def added_solution_C : ℚ := 2 / 5

-- Theorem statement
theorem sweetest_sugar_water :
  added_solution_C > concentration_A ∧
  added_solution_C > added_solution_B :=
sorry

end NUMINAMATH_CALUDE_sweetest_sugar_water_l3312_331229


namespace NUMINAMATH_CALUDE_vhs_trade_in_value_proof_l3312_331230

/-- The number of movies John has -/
def num_movies : ℕ := 100

/-- The cost of each DVD in dollars -/
def dvd_cost : ℚ := 10

/-- The total cost to replace all movies in dollars -/
def total_replacement_cost : ℚ := 800

/-- The trade-in value of each VHS in dollars -/
def vhs_trade_in_value : ℚ := 2

theorem vhs_trade_in_value_proof :
  vhs_trade_in_value * num_movies + total_replacement_cost = dvd_cost * num_movies :=
sorry

end NUMINAMATH_CALUDE_vhs_trade_in_value_proof_l3312_331230


namespace NUMINAMATH_CALUDE_perimeter_of_triangle_ABF2_l3312_331206

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 41 + y^2 / 25 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the chord AB
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Assume the chord AB passes through F1
axiom chord_through_F1 : A.1 = F1.1 ∧ A.2 = F1.2 ∨ B.1 = F1.1 ∧ B.2 = F1.2

-- Define the perimeter of triangle ABF2
def perimeter_ABF2 : ℝ := sorry

-- Theorem statement
theorem perimeter_of_triangle_ABF2 : perimeter_ABF2 = 4 * Real.sqrt 41 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_triangle_ABF2_l3312_331206


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_2005_l3312_331248

def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_m_divisible_by_2005 : 
  ∃ m : ℕ+, (3^100 * (m.val + 1) - 1) % 2005 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_2005_l3312_331248


namespace NUMINAMATH_CALUDE_S_intersect_T_l3312_331245

noncomputable def S : Set ℝ := {y | ∃ x, y = 2^x}
def T : Set ℝ := {x | Real.log (x - 1) < 0}

theorem S_intersect_T : S ∩ T = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_l3312_331245


namespace NUMINAMATH_CALUDE_units_digit_of_x_l3312_331228

def has_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem units_digit_of_x (p x : ℕ) (h1 : p * x = 32^10)
  (h2 : has_units_digit p 6) (h3 : x % 4 = 0) :
  has_units_digit x 1 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_x_l3312_331228


namespace NUMINAMATH_CALUDE_equal_volume_cans_l3312_331258

/-- Represents a cylindrical can with radius and height -/
structure Can where
  radius : ℝ
  height : ℝ

/-- Theorem stating the relation between two cans with equal volume -/
theorem equal_volume_cans (can1 can2 : Can) 
  (h_volume : can1.radius ^ 2 * can1.height = can2.radius ^ 2 * can2.height)
  (h_height : can2.height = 4 * can1.height)
  (h_narrow_radius : can1.radius = 10) :
  can2.radius = 20 := by
  sorry

end NUMINAMATH_CALUDE_equal_volume_cans_l3312_331258


namespace NUMINAMATH_CALUDE_area_of_region_l3312_331277

/-- The region defined by the inequality |4x - 20| + |3y - 6| ≤ 4 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |4 * p.1 - 20| + |3 * p.2 - 6| ≤ 4}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating that the area of the region is 8/3 -/
theorem area_of_region : area Region = 8/3 := by sorry

end NUMINAMATH_CALUDE_area_of_region_l3312_331277


namespace NUMINAMATH_CALUDE_alberts_number_l3312_331222

theorem alberts_number (a b c : ℚ) 
  (h1 : a = 2 * b + 1)
  (h2 : b = 2 * c + 1)
  (h3 : c = 2 * a + 2) :
  a = -11/7 := by
sorry

end NUMINAMATH_CALUDE_alberts_number_l3312_331222


namespace NUMINAMATH_CALUDE_sum_is_composite_l3312_331226

theorem sum_is_composite (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ (a : ℕ) + b + c + d = x * y :=
by
  sorry

end NUMINAMATH_CALUDE_sum_is_composite_l3312_331226


namespace NUMINAMATH_CALUDE_apple_picking_ratio_l3312_331201

/-- Represents the number of apples picked in each hour -/
structure ApplePicking where
  first_hour : ℕ
  second_hour : ℕ
  third_hour : ℕ

/-- Calculates the total number of apples picked -/
def total_apples (a : ApplePicking) : ℕ :=
  a.first_hour + a.second_hour + a.third_hour

/-- Theorem: The ratio of apples picked in the second hour to the first hour is 2:1 -/
theorem apple_picking_ratio (a : ApplePicking) :
  a.first_hour = 66 →
  a.third_hour = a.first_hour / 3 →
  total_apples a = 220 →
  a.second_hour = 2 * a.first_hour :=
by
  sorry


end NUMINAMATH_CALUDE_apple_picking_ratio_l3312_331201


namespace NUMINAMATH_CALUDE_four_consecutive_primes_sum_l3312_331288

theorem four_consecutive_primes_sum (A B : ℕ) : 
  (A > 0) → 
  (B > 0) → 
  (Nat.Prime A) → 
  (Nat.Prime B) → 
  (Nat.Prime (A - B)) → 
  (Nat.Prime (A + B)) → 
  (∃ p q r s : ℕ, 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧
    q = p + 2 ∧ r = q + 2 ∧ s = r + 2 ∧
    ((A = p ∧ B = q) ∨ (A = q ∧ B = p) ∨ (A = r ∧ B = p) ∨ (A = s ∧ B = p))) →
  p + q + r + s = 17 :=
sorry

end NUMINAMATH_CALUDE_four_consecutive_primes_sum_l3312_331288


namespace NUMINAMATH_CALUDE_derivative_f_minus_f4x_l3312_331243

/-- Given a function f where the derivative of f(x) - f(2x) at x = 1 is 5 and at x = 2 is 7,
    the derivative of f(x) - f(4x) at x = 1 is 19. -/
theorem derivative_f_minus_f4x (f : ℝ → ℝ) 
  (h1 : deriv (fun x ↦ f x - f (2 * x)) 1 = 5)
  (h2 : deriv (fun x ↦ f x - f (2 * x)) 2 = 7) :
  deriv (fun x ↦ f x - f (4 * x)) 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_minus_f4x_l3312_331243


namespace NUMINAMATH_CALUDE_sequence_increasing_iff_a_in_range_l3312_331261

def sequence_a (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 7 then (3 - a) * n - 3 else a^(n - 6)

theorem sequence_increasing_iff_a_in_range (a : ℝ) :
  (∀ n : ℕ, sequence_a a n ≤ sequence_a a (n + 1)) ↔ (9/4 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_sequence_increasing_iff_a_in_range_l3312_331261


namespace NUMINAMATH_CALUDE_village_population_l3312_331278

theorem village_population (population : ℕ) : 
  (90 : ℚ) / 100 * population = 8100 → population = 9000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3312_331278


namespace NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l3312_331209

theorem unique_solution_for_rational_equation :
  ∃! k : ℚ, ∀ x : ℚ, (x + 3) / (k * x + x - 3) = x ∧ k * x + x - 3 ≠ 0 → k = -7/3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l3312_331209


namespace NUMINAMATH_CALUDE_people_per_column_l3312_331219

theorem people_per_column (total_people : ℕ) 
  (h1 : total_people / 60 = 8) 
  (h2 : total_people % 16 = 0) : 
  total_people / 16 = 30 := by
sorry

end NUMINAMATH_CALUDE_people_per_column_l3312_331219
