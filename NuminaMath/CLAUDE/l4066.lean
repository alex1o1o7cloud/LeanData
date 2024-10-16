import Mathlib

namespace NUMINAMATH_CALUDE_salary_grade_increase_amount_l4066_406649

/-- Represents the salary grade of an employee -/
def SalaryGrade := {s : ℝ // 1 ≤ s ∧ s ≤ 5}

/-- Calculates the hourly wage based on salary grade and base increase -/
def hourlyWage (s : SalaryGrade) (x : ℝ) : ℝ :=
  7.50 + x * (s.val - 1)

/-- States that the difference in hourly wage between grade 5 and grade 1 is $1.25 -/
def wageDifference (x : ℝ) : Prop :=
  hourlyWage ⟨5, by norm_num⟩ x - hourlyWage ⟨1, by norm_num⟩ x = 1.25

theorem salary_grade_increase_amount :
  ∃ x : ℝ, wageDifference x ∧ x = 0.3125 := by sorry

end NUMINAMATH_CALUDE_salary_grade_increase_amount_l4066_406649


namespace NUMINAMATH_CALUDE_candy_ratio_in_bowl_l4066_406657

-- Define the properties of each bag
def bag1_total : ℕ := 27
def bag1_red_ratio : ℚ := 1/3

def bag2_total : ℕ := 36
def bag2_red_ratio : ℚ := 1/4

def bag3_total : ℕ := 45
def bag3_red_ratio : ℚ := 1/5

-- Define the theorem
theorem candy_ratio_in_bowl :
  let total_candies := bag1_total + bag2_total + bag3_total
  let total_red := bag1_total * bag1_red_ratio + bag2_total * bag2_red_ratio + bag3_total * bag3_red_ratio
  total_red / total_candies = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_in_bowl_l4066_406657


namespace NUMINAMATH_CALUDE_finn_bought_12_boxes_l4066_406643

/-- The cost of one package of index cards -/
def index_card_cost : ℚ := (55.40 - 15 * 1.85) / 7

/-- The number of boxes of paper clips Finn bought -/
def finn_paper_clips : ℚ := (61.70 - 10 * index_card_cost) / 1.85

theorem finn_bought_12_boxes :
  finn_paper_clips = 12 := by sorry

end NUMINAMATH_CALUDE_finn_bought_12_boxes_l4066_406643


namespace NUMINAMATH_CALUDE_range_of_x_when_proposition_false_l4066_406654

theorem range_of_x_when_proposition_false (x : ℝ) :
  x^2 - 5*x + 4 ≤ 0 → 1 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_when_proposition_false_l4066_406654


namespace NUMINAMATH_CALUDE_cos_225_degrees_l4066_406668

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l4066_406668


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_quadratic_equation_with_factoring_solutions_l4066_406613

theorem quadratic_equation_solutions :
  (∃ x : ℝ, x^2 - 5*x + 6 = 0) ↔ (∃ x : ℝ, x = 2 ∨ x = 3) :=
by sorry

theorem quadratic_equation_with_factoring_solutions :
  (∃ x : ℝ, (x - 2)^2 = 2*(x - 3)*(x - 2)) ↔ (∃ x : ℝ, x = 2 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_quadratic_equation_with_factoring_solutions_l4066_406613


namespace NUMINAMATH_CALUDE_vector_operation_proof_l4066_406693

def v1 : Fin 3 → ℝ := ![3, -2, 5]
def v2 : Fin 3 → ℝ := ![-1, 6, -3]

theorem vector_operation_proof :
  (2 : ℝ) • (v1 + v2) = ![4, 8, 4] := by sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l4066_406693


namespace NUMINAMATH_CALUDE_highest_frequency_last_3_groups_l4066_406686

/-- Represents the frequency distribution of a sample -/
structure FrequencyDistribution where
  total_sample : ℕ
  num_groups : ℕ
  cumulative_freq_7 : ℚ
  last_3_geometric : Bool
  common_ratio_gt_2 : Bool

/-- Theorem stating the highest frequency in the last 3 groups -/
theorem highest_frequency_last_3_groups
  (fd : FrequencyDistribution)
  (h1 : fd.total_sample = 100)
  (h2 : fd.num_groups = 10)
  (h3 : fd.cumulative_freq_7 = 79/100)
  (h4 : fd.last_3_geometric)
  (h5 : fd.common_ratio_gt_2) :
  ∃ (a r : ℕ),
    r > 2 ∧
    a + a * r + a * r^2 = 21 ∧
    (∀ x : ℕ, x ∈ [a, a * r, a * r^2] → x ≤ 16) ∧
    16 ∈ [a, a * r, a * r^2] :=
sorry

end NUMINAMATH_CALUDE_highest_frequency_last_3_groups_l4066_406686


namespace NUMINAMATH_CALUDE_function_characterization_l4066_406605

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property of the function
def SatisfiesProperty (f : RealFunction) : Prop :=
  ∀ x : ℝ, f x = f (x/2) + (x/2) * (deriv f x)

-- State the theorem
theorem function_characterization :
  ∀ f : RealFunction, SatisfiesProperty f →
  ∃ c b : ℝ, ∀ x : ℝ, f x = c * x + b :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l4066_406605


namespace NUMINAMATH_CALUDE_rectangle_max_area_max_area_achievable_l4066_406671

/-- Given a rectangle with perimeter 40 inches, its maximum area is 100 square inches. -/
theorem rectangle_max_area :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 40 →
  ∀ a : ℝ,
  (0 < a ∧ ∃ w h : ℝ, w > 0 ∧ h > 0 ∧ 2 * w + 2 * h = 40 ∧ a = w * h) →
  x * y ≥ a :=
by sorry

/-- The maximum area of 100 square inches is achievable. -/
theorem max_area_achievable :
  ∃ x y : ℝ,
  x > 0 ∧ y > 0 ∧
  2 * x + 2 * y = 40 ∧
  x * y = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_max_area_achievable_l4066_406671


namespace NUMINAMATH_CALUDE_power_of_power_l4066_406661

theorem power_of_power : (2^2)^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l4066_406661


namespace NUMINAMATH_CALUDE_cyclists_speeds_l4066_406682

/-- Represents the scenario of two cyclists riding towards each other -/
structure CyclistsScenario where
  x : ℝ  -- Speed of the first cyclist in km/h
  y : ℝ  -- Speed of the second cyclist in km/h
  AB : ℝ  -- Distance between the two starting points in km

/-- Condition 1: If the first cyclist starts 1 hour earlier and the second one starts half an hour later,
    they meet 18 minutes earlier than normal -/
def condition1 (s : CyclistsScenario) : Prop :=
  (s.AB / (s.x + s.y) + 1 - 18/60) * s.x + (s.AB / (s.x + s.y) - 1/2 - 18/60) * s.y = s.AB

/-- Condition 2: If the first cyclist starts half an hour later and the second one starts 1 hour earlier,
    the meeting point moves by 11.2 km (11200 meters) -/
def condition2 (s : CyclistsScenario) : Prop :=
  (s.AB - 1.5 * s.y) / (s.x + s.y) * s.x + 11.2 = s.AB / (s.x + s.y) * s.x

/-- Theorem stating that given the conditions, the speeds of the cyclists are 16 km/h and 14 km/h -/
theorem cyclists_speeds (s : CyclistsScenario) 
  (h1 : condition1 s) (h2 : condition2 s) : s.x = 16 ∧ s.y = 14 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_speeds_l4066_406682


namespace NUMINAMATH_CALUDE_tom_score_l4066_406658

/-- Calculates the score for regular enemies --/
def regularScore (kills : ℕ) : ℕ :=
  let baseScore := kills * 10
  if kills ≥ 200 then baseScore * 2
  else if kills ≥ 150 then baseScore + (baseScore * 3 / 4)
  else if kills ≥ 100 then baseScore + (baseScore / 2)
  else baseScore

/-- Calculates the score for elite enemies --/
def eliteScore (kills : ℕ) : ℕ :=
  let baseScore := kills * 25
  if kills ≥ 35 then baseScore + (baseScore * 7 / 10)
  else if kills ≥ 25 then baseScore + (baseScore / 2)
  else if kills ≥ 15 then baseScore + (baseScore * 3 / 10)
  else baseScore

/-- Calculates the score for boss enemies --/
def bossScore (kills : ℕ) : ℕ :=
  let baseScore := kills * 50
  if kills ≥ 10 then baseScore + (baseScore * 2 / 5)
  else if kills ≥ 5 then baseScore + (baseScore / 5)
  else baseScore

/-- Calculates the total score --/
def totalScore (regularKills eliteKills bossKills : ℕ) : ℕ :=
  regularScore regularKills + eliteScore eliteKills + bossScore bossKills

theorem tom_score : totalScore 160 20 8 = 3930 := by
  sorry

end NUMINAMATH_CALUDE_tom_score_l4066_406658


namespace NUMINAMATH_CALUDE_final_sum_theorem_l4066_406616

theorem final_sum_theorem (a b S : ℝ) (h : a + b = S) :
  3 * (a + 4) + 3 * (b + 4) = 3 * S + 24 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l4066_406616


namespace NUMINAMATH_CALUDE_stock_price_theorem_l4066_406621

def stock_price_evolution (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let first_year_price := initial_price * (1 + first_year_increase)
  let second_year_price := first_year_price * (1 - second_year_decrease)
  second_year_price

theorem stock_price_theorem :
  stock_price_evolution 100 1 0.25 = 150 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_theorem_l4066_406621


namespace NUMINAMATH_CALUDE_candy_bar_cost_l4066_406607

/-- The cost of candy bars purchased by Dan -/
def total_cost : ℚ := 6

/-- The number of candy bars Dan bought -/
def number_of_bars : ℕ := 2

/-- The cost of each candy bar -/
def cost_per_bar : ℚ := total_cost / number_of_bars

/-- Theorem stating that the cost of each candy bar is $3 -/
theorem candy_bar_cost : cost_per_bar = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l4066_406607


namespace NUMINAMATH_CALUDE_x_power_five_minus_reciprocal_l4066_406645

theorem x_power_five_minus_reciprocal (x : ℝ) (h : x + 1/x = Real.sqrt 2) :
  x^5 - 1/x^5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_power_five_minus_reciprocal_l4066_406645


namespace NUMINAMATH_CALUDE_highlighter_expense_proof_l4066_406620

def total_money : ℕ := 100
def sharpener_price : ℕ := 5
def notebook_price : ℕ := 5
def eraser_price : ℕ := 4
def sharpener_count : ℕ := 2
def notebook_count : ℕ := 4
def eraser_count : ℕ := 10

def heaven_expense : ℕ := sharpener_price * sharpener_count + notebook_price * notebook_count
def brother_eraser_expense : ℕ := eraser_price * eraser_count

theorem highlighter_expense_proof :
  total_money - (heaven_expense + brother_eraser_expense) = 30 := by sorry

end NUMINAMATH_CALUDE_highlighter_expense_proof_l4066_406620


namespace NUMINAMATH_CALUDE_seventh_term_is_ten_l4066_406684

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧ 
  (a 2 = 2) ∧ 
  (a 4 + a 5 = 12)

/-- Theorem stating that the 7th term of the arithmetic sequence is 10 -/
theorem seventh_term_is_ten (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_ten_l4066_406684


namespace NUMINAMATH_CALUDE_horners_rule_for_specific_polynomial_v3_value_at_3_l4066_406665

def horner_step (a : ℕ) (x v : ℕ) : ℕ := v * x + a

def horners_rule (coeffs : List ℕ) (x : ℕ) : ℕ :=
  coeffs.foldl (horner_step x) 0

theorem horners_rule_for_specific_polynomial (x : ℕ) :
  horners_rule [1, 1, 3, 2, 0, 1] x = x^5 + 2*x^3 + 3*x^2 + x + 1 := by sorry

theorem v3_value_at_3 :
  let coeffs := [1, 1, 3, 2, 0, 1]
  let x := 3
  let v0 := 1
  let v1 := horner_step 0 x v0
  let v2 := horner_step 2 x v1
  let v3 := horner_step 3 x v2
  v3 = 36 := by sorry

end NUMINAMATH_CALUDE_horners_rule_for_specific_polynomial_v3_value_at_3_l4066_406665


namespace NUMINAMATH_CALUDE_prime_square_difference_one_l4066_406609

theorem prime_square_difference_one (p q : ℕ) : 
  Prime p → Prime q → p^2 - 2*q^2 = 1 → (p = 3 ∧ q = 2) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_difference_one_l4066_406609


namespace NUMINAMATH_CALUDE_inequality_solution_l4066_406659

theorem inequality_solution (x : ℝ) : (x + 4) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l4066_406659


namespace NUMINAMATH_CALUDE_unique_n_exists_l4066_406625

theorem unique_n_exists : ∃! n : ℤ,
  50 < n ∧ n < 150 ∧
  7 ∣ n ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n % 4 = 1 ∧
  n = 105 := by
sorry

end NUMINAMATH_CALUDE_unique_n_exists_l4066_406625


namespace NUMINAMATH_CALUDE_circle_radius_with_chord_l4066_406610

/-- The radius of a circle given specific conditions --/
theorem circle_radius_with_chord (r : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    -- Line equation
    (A.1 - Real.sqrt 3 * A.2 + 8 = 0) ∧ 
    (B.1 - Real.sqrt 3 * B.2 + 8 = 0) ∧
    -- Circle equation
    (A.1^2 + A.2^2 = r^2) ∧ 
    (B.1^2 + B.2^2 = r^2) ∧
    -- Length of chord AB
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 36)) → 
  r = 5 := by
sorry


end NUMINAMATH_CALUDE_circle_radius_with_chord_l4066_406610


namespace NUMINAMATH_CALUDE_tangent_line_and_extreme_values_l4066_406687

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - 4

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem tangent_line_and_extreme_values :
  ∃ (a b : ℝ),
  (f a b 2 = -2) ∧
  (f' a b 2 = 1) ∧
  (a = -4 ∧ b = 5) ∧
  (∀ x : ℝ, f (-4) 5 x ≤ -2) ∧
  (f (-4) 5 1 = -2) ∧
  (∀ x : ℝ, f (-4) 5 x ≥ -58/27) ∧
  (f (-4) 5 (5/3) = -58/27) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_and_extreme_values_l4066_406687


namespace NUMINAMATH_CALUDE_expression_integer_iff_special_form_l4066_406690

def expression (n : ℤ) : ℝ :=
  (n + (n^2 + 1).sqrt)^(1/3) + (n - (n^2 + 1).sqrt)^(1/3)

theorem expression_integer_iff_special_form (n : ℤ) :
  ∃ (k : ℤ), k > 0 ∧ expression n = k ↔ ∃ (m : ℤ), m > 0 ∧ n = m * (m^2 + 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_expression_integer_iff_special_form_l4066_406690


namespace NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l4066_406642

theorem pigeonhole_on_permutation_sums (n : ℕ) : 
  ∀ (p : Fin (2*n) → Fin (2*n)), 
  ∃ (i j : Fin (2*n)), i ≠ j ∧ 
  ((p i).val + i.val + 1) % (2*n) = ((p j).val + j.val + 1) % (2*n) :=
sorry

end NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l4066_406642


namespace NUMINAMATH_CALUDE_min_x_minus_y_l4066_406601

theorem min_x_minus_y (x y : ℝ) (h1 : x > 0) (h2 : 0 > y) 
  (h3 : 1 / (x + 2) + 1 / (1 - y) = 1 / 6) : x - y ≥ 21 := by
  sorry

end NUMINAMATH_CALUDE_min_x_minus_y_l4066_406601


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l4066_406674

theorem rectangle_shorter_side 
  (perimeter : ℝ) 
  (area : ℝ) 
  (h_perimeter : perimeter = 60) 
  (h_area : area = 200) :
  ∃ (shorter_side longer_side : ℝ),
    shorter_side ≤ longer_side ∧
    2 * (shorter_side + longer_side) = perimeter ∧
    shorter_side * longer_side = area ∧
    shorter_side = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l4066_406674


namespace NUMINAMATH_CALUDE_correlation_significance_l4066_406681

-- Define r as a real number representing a correlation coefficient
variable (r : ℝ)

-- Define r_0.05 as the critical value for a 5% significance level
variable (r_0_05 : ℝ)

-- Define a function that represents the probability of an event
def event_probability (r : ℝ) (r_0_05 : ℝ) : Prop :=
  ∃ p : ℝ, p < 0.05 ∧ (|r| > r_0_05 ↔ p < 0.05)

-- Theorem stating the equivalence
theorem correlation_significance (r : ℝ) (r_0_05 : ℝ) :
  |r| > r_0_05 ↔ event_probability r r_0_05 :=
sorry

end NUMINAMATH_CALUDE_correlation_significance_l4066_406681


namespace NUMINAMATH_CALUDE_external_angle_ninety_degrees_l4066_406678

theorem external_angle_ninety_degrees (a b c : ℝ) (h1 : a = 40) (h2 : b = 50) 
  (h3 : a + b + c = 180) (x : ℝ) (h4 : x + c = 180) : x = 90 := by
  sorry

end NUMINAMATH_CALUDE_external_angle_ninety_degrees_l4066_406678


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l4066_406600

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  r : ℝ  -- radius of the sphere
  x : ℝ  -- width of the box
  y : ℝ  -- length of the box
  z : ℝ  -- height of the box

/-- Properties of the inscribed box -/
def InscribedBoxProperties (box : InscribedBox) : Prop :=
  box.x > 0 ∧ box.y > 0 ∧ box.z > 0 ∧  -- dimensions are positive
  box.z = 3 * box.x ∧  -- ratio between height and width is 1:3
  4 * (box.x + box.y + box.z) = 72 ∧  -- sum of edge lengths
  2 * (box.x * box.y + box.y * box.z + box.x * box.z) = 162 ∧  -- surface area
  4 * box.r^2 = box.x^2 + box.y^2 + box.z^2  -- inscribed in sphere

theorem inscribed_box_radius (box : InscribedBox) 
  (h : InscribedBoxProperties box) : box.r = 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l4066_406600


namespace NUMINAMATH_CALUDE_set_identities_l4066_406619

variable {α : Type*}
variable (A B C : Set α)

theorem set_identities :
  (A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)) ∧
  (A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end NUMINAMATH_CALUDE_set_identities_l4066_406619


namespace NUMINAMATH_CALUDE_susan_chairs_l4066_406608

/-- The number of chairs in Susan's house -/
def total_chairs : ℕ :=
  let red_chairs : ℕ := 5
  let yellow_chairs : ℕ := 4 * red_chairs
  let blue_chairs : ℕ := yellow_chairs - 2
  let green_chairs : ℕ := (red_chairs + blue_chairs) / 2
  red_chairs + yellow_chairs + blue_chairs + green_chairs

/-- Theorem stating the total number of chairs in Susan's house -/
theorem susan_chairs : total_chairs = 54 := by
  sorry

end NUMINAMATH_CALUDE_susan_chairs_l4066_406608


namespace NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l4066_406695

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℝ × ℝ := (-36, 26)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℝ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: -4y = 3x + 4 -/
def line2 (x y : ℝ) : Prop := -4 * y = 3 * x + 4

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℝ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l4066_406695


namespace NUMINAMATH_CALUDE_speed_conversion_l4066_406689

/-- Proves that a speed of 0.8 km/h, when expressed as a fraction in m/s with numerator 8, has a denominator of 36 -/
theorem speed_conversion (speed_kmh : ℚ) (speed_ms_num : ℕ) : 
  speed_kmh = 0.8 → speed_ms_num = 8 → 
  ∃ (speed_ms_den : ℕ), 
    (speed_kmh * 1000 / 3600 = speed_ms_num / speed_ms_den) ∧ 
    speed_ms_den = 36 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l4066_406689


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l4066_406604

theorem simplify_sqrt_difference (x : ℝ) (h : x ≤ 2) : 
  Real.sqrt (x^2 - 4*x + 4) - Real.sqrt (x^2 - 6*x + 9) = -1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l4066_406604


namespace NUMINAMATH_CALUDE_count_partitions_len_one_count_partitions_len_two_l4066_406634

/-- Represents a partition of the set {1, ..., n} into arithmetic progressions -/
def ArithmeticPartition (n : ℕ) := List (List ℕ)

/-- Checks if a partition is valid (all subsets are arithmetic progressions) -/
def IsValidPartition (n : ℕ) (p : ArithmeticPartition n) : Prop := sorry

/-- Counts the number of valid partitions with subsets of length at least 1 -/
def CountPartitionsLenOne (n : ℕ) : ℕ := sorry

/-- Counts the number of valid partitions with subsets of length at least 2 -/
def CountPartitionsLenTwo (n : ℕ) : ℕ := sorry

/-- Theorem: The number of valid partitions with subsets of length at least 1 -/
theorem count_partitions_len_one (n : ℕ) : 
  CountPartitionsLenOne n = 2^n - 2 := by sorry

/-- Theorem: The number of valid partitions with subsets of length at least 2 -/
theorem count_partitions_len_two (n : ℕ) : 
  CountPartitionsLenTwo n = 2^n - n * (n - 1) / 2 - 2 := by sorry

end NUMINAMATH_CALUDE_count_partitions_len_one_count_partitions_len_two_l4066_406634


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_not_always_supplementary_l4066_406631

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of corresponding angles
def corresponding_angles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the concept of supplementary angles
def supplementary (a1 a2 : Angle) : Prop := sorry

-- The theorem to be proven
theorem parallel_lines_corresponding_angles_not_always_supplementary :
  ¬ ∀ (l1 l2 : Line) (a1 a2 : Angle), 
    parallel l1 l2 → corresponding_angles a1 a2 l1 l2 → supplementary a1 a2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_not_always_supplementary_l4066_406631


namespace NUMINAMATH_CALUDE_first_player_can_force_draw_l4066_406603

/-- Represents the state of a square on the game board -/
inductive Square
| Empty : Square
| A : Square
| B : Square

/-- Represents the game board as a list of squares -/
def Board := List Square

/-- Checks if a given board contains the winning sequence ABA -/
def hasWinningSequence (board : Board) : Bool :=
  sorry

/-- Represents a player's move -/
structure Move where
  position : Nat
  letter : Square

/-- Applies a move to the board -/
def applyMove (board : Board) (move : Move) : Board :=
  sorry

/-- Checks if a move is valid on the given board -/
def isValidMove (board : Board) (move : Move) : Bool :=
  sorry

/-- Represents the game state -/
structure GameState where
  board : Board
  currentPlayer : Bool  -- True for first player, False for second player

/-- The main theorem stating that the first player can force a draw -/
theorem first_player_can_force_draw :
  ∃ (strategy : GameState → Move),
    ∀ (game : GameState),
      game.board.length = 14 →
      game.currentPlayer = true →
      ¬(hasWinningSequence (applyMove game.board (strategy game))) :=
sorry

end NUMINAMATH_CALUDE_first_player_can_force_draw_l4066_406603


namespace NUMINAMATH_CALUDE_base_of_term_l4066_406672

theorem base_of_term (x : ℝ) (k : ℝ) : 
  (1/2)^23 * (1/x)^k = 1/18^23 ∧ k = 11.5 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_of_term_l4066_406672


namespace NUMINAMATH_CALUDE_standard_ellipse_foci_l4066_406647

/-- Represents an ellipse with equation (x^2 / 10) + y^2 = 1 -/
structure StandardEllipse where
  equation : ∀ (x y : ℝ), (x^2 / 10) + y^2 = 1

/-- Represents the foci of an ellipse -/
structure EllipseFoci where
  x : ℝ
  y : ℝ

/-- Theorem: The foci of the standard ellipse are at (3, 0) and (-3, 0) -/
theorem standard_ellipse_foci (e : StandardEllipse) : 
  ∃ (f1 f2 : EllipseFoci), f1.x = 3 ∧ f1.y = 0 ∧ f2.x = -3 ∧ f2.y = 0 :=
sorry

end NUMINAMATH_CALUDE_standard_ellipse_foci_l4066_406647


namespace NUMINAMATH_CALUDE_system_solution_l4066_406627

/-- Given a system of equations, prove the solutions. -/
theorem system_solution (a b c : ℝ) :
  let eq1 := (y : ℝ) ^ 2 - z * x = a * (x + y + z) ^ 2
  let eq2 := x ^ 2 - y * z = b * (x + y + z) ^ 2
  let eq3 := z ^ 2 - x * y = c * (x + y + z) ^ 2
  (∃ s : ℝ,
    x = (2 * c - a - b + 1) * s ∧
    y = (2 * a - b - c + 1) * s ∧
    z = (2 * b - c - a + 1) * s ∧
    a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a = a + b + c) ∨
  (x = 0 ∧ y = 0 ∧ z = 0 ∧
    a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a ≠ a + b + c) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4066_406627


namespace NUMINAMATH_CALUDE_fraction_equality_and_sum_l4066_406656

theorem fraction_equality_and_sum : ∃! (α β : ℝ),
  (∀ x : ℝ, x ≠ -β → x ≠ -110.36 →
    (x - α) / (x + β) = (x^2 - 64*x + 1007) / (x^2 + 81*x - 3240)) ∧
  α + β = 146.483 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_and_sum_l4066_406656


namespace NUMINAMATH_CALUDE_colored_isosceles_triangle_l4066_406615

/-- A regular polygon with 4n + 1 vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (4 * n + 1) → ℝ × ℝ

/-- A coloring of 2n vertices in a (4n + 1)-gon -/
def Coloring (n : ℕ) := Fin (4 * n + 1) → Bool

/-- Three vertices form an isosceles triangle -/
def IsIsosceles (p : RegularPolygon n) (v1 v2 v3 : Fin (4 * n + 1)) : Prop :=
  let d12 := dist (p.vertices v1) (p.vertices v2)
  let d23 := dist (p.vertices v2) (p.vertices v3)
  let d31 := dist (p.vertices v3) (p.vertices v1)
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- Main theorem -/
theorem colored_isosceles_triangle (n : ℕ) (h : n ≥ 3) (p : RegularPolygon n) (c : Coloring n) :
  ∃ v1 v2 v3 : Fin (4 * n + 1), c v1 ∧ c v2 ∧ c v3 ∧ IsIsosceles p v1 v2 v3 :=
sorry


end NUMINAMATH_CALUDE_colored_isosceles_triangle_l4066_406615


namespace NUMINAMATH_CALUDE_pete_triple_age_of_son_l4066_406663

def pete_age : ℕ := 35
def son_age : ℕ := 9
def years_until_triple : ℕ := 4

theorem pete_triple_age_of_son :
  pete_age + years_until_triple = 3 * (son_age + years_until_triple) :=
by sorry

end NUMINAMATH_CALUDE_pete_triple_age_of_son_l4066_406663


namespace NUMINAMATH_CALUDE_license_plate_theorem_l4066_406660

/-- The number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The length of the license plate -/
def plate_length : ℕ := 5

/-- The number of letters at the start of the plate -/
def num_start_letters : ℕ := 2

/-- The number of digits at the end of the plate -/
def num_end_digits : ℕ := 3

/-- The number of ways to design a license plate with the given conditions -/
def license_plate_designs : ℕ :=
  num_letters * num_digits * (num_digits - 1)

theorem license_plate_theorem :
  license_plate_designs = 2340 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l4066_406660


namespace NUMINAMATH_CALUDE_contradiction_assumption_l4066_406614

theorem contradiction_assumption (x y : ℝ) (h : x > y) : 
  ¬(x^3 > y^3) ↔ x^3 ≤ y^3 := by
sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l4066_406614


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4066_406651

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4066_406651


namespace NUMINAMATH_CALUDE_unique_prime_pair_solution_l4066_406677

theorem unique_prime_pair_solution : 
  ∃! (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (7 * p * q^2 + p = q^3 + 43 * p^3 + 1) ∧ 
    p = 2 ∧ q = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_solution_l4066_406677


namespace NUMINAMATH_CALUDE_fraction_simplification_l4066_406673

theorem fraction_simplification (x : ℝ) 
  (h1 : x + 1 ≠ 0) (h2 : 2 + x ≠ 0) (h3 : 2 - x ≠ 0) (h4 : x = 0) : 
  (x^2 - 4*x + 4) / (x + 1) / ((3 / (x + 1)) - x + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4066_406673


namespace NUMINAMATH_CALUDE_min_a_for_parabola_l4066_406653

/-- Given a parabola y = ax^2 + bx + c with vertex at (1/4, -9/8), 
    where a > 0 and a + b + c is an integer, 
    the minimum possible value of a is 2/9 -/
theorem min_a_for_parabola (a b c : ℝ) : 
  a > 0 ∧ 
  (∃ k : ℤ, a + b + c = k) ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = a * (x - 1/4)^2 - 9/8) → 
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, (∃ k : ℤ, a' + b' + c' = k) ∧ 
      (∀ x : ℝ, a' * x^2 + b' * x + c' = a' * (x - 1/4)^2 - 9/8)) → 
    a' ≥ 2/9) ∧ 
  a = 2/9 := by
sorry

end NUMINAMATH_CALUDE_min_a_for_parabola_l4066_406653


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l4066_406632

theorem quadratic_inequality_solutions (a : ℝ) :
  (a = -1 → {x : ℝ | a * x^2 + 5 * x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6}) ∧
  ({x : ℝ | a * x^2 + 5 * x + 6 > 0} = {x : ℝ | x < -3 ∨ x > -2} → a = 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l4066_406632


namespace NUMINAMATH_CALUDE_circle_intersection_range_l4066_406635

theorem circle_intersection_range (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m ∧ x^2 + y^2 + 6*x - 8*y - 11 = 0) →
  1 ≤ m ∧ m ≤ 121 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l4066_406635


namespace NUMINAMATH_CALUDE_job_completion_time_l4066_406612

/-- Proves that the initial estimated time to finish the job is 8 days given the problem conditions. -/
theorem job_completion_time : 
  ∀ (initial_workers : ℕ) 
    (additional_workers : ℕ) 
    (days_before_joining : ℕ) 
    (days_after_joining : ℕ),
  initial_workers = 6 →
  additional_workers = 4 →
  days_before_joining = 3 →
  days_after_joining = 3 →
  ∃ (initial_estimate : ℕ),
    initial_estimate * initial_workers = 
      (initial_workers * days_before_joining + 
       (initial_workers + additional_workers) * days_after_joining) ∧
    initial_estimate = 8 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l4066_406612


namespace NUMINAMATH_CALUDE_intersection_range_l4066_406638

-- Define the semicircle
def semicircle (x y : ℝ) : Prop := x^2 + y^2 = 9 ∧ y ≥ 0

-- Define the line
def line (k x y : ℝ) : Prop := y = k*(x-3) + 4

-- Define the condition for two distinct solutions
def has_two_distinct_solutions (k : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    semicircle x₁ y₁ ∧ semicircle x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, has_two_distinct_solutions k ↔ 7/24 < k ∧ k ≤ 2/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l4066_406638


namespace NUMINAMATH_CALUDE_blueberry_picking_l4066_406624

/-- The total number of pints of blueberries picked by Annie, Kathryn, and Ben -/
def total_pints (annie kathryn ben : ℕ) : ℕ := annie + kathryn + ben

/-- Theorem stating the total number of pints picked given the conditions -/
theorem blueberry_picking :
  ∀ (annie kathryn ben : ℕ),
  annie = 8 →
  kathryn = annie + 2 →
  ben = kathryn - 3 →
  total_pints annie kathryn ben = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_blueberry_picking_l4066_406624


namespace NUMINAMATH_CALUDE_subset_intersection_bound_l4066_406617

theorem subset_intersection_bound (m n k : ℕ) (F : Fin k → Finset (Fin m)) :
  m ≥ n →
  n > 1 →
  (∀ i, (F i).card = n) →
  (∀ i j, i < j → (F i ∩ F j).card ≤ 1) →
  k ≤ m * (m - 1) / (n * (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_subset_intersection_bound_l4066_406617


namespace NUMINAMATH_CALUDE_initial_walking_speed_l4066_406667

/-- Proves that given a specific distance and time difference between two speeds,
    the initial speed is 11.25 kmph. -/
theorem initial_walking_speed 
  (distance : ℝ) 
  (time_diff : ℝ) 
  (faster_speed : ℝ) :
  distance = 9.999999999999998 →
  time_diff = 1/3 →
  faster_speed = 15 →
  ∃ (initial_speed : ℝ),
    distance / initial_speed - distance / faster_speed = time_diff ∧
    initial_speed = 11.25 := by
  sorry

#check initial_walking_speed

end NUMINAMATH_CALUDE_initial_walking_speed_l4066_406667


namespace NUMINAMATH_CALUDE_optimal_pricing_achieves_target_profit_l4066_406664

/-- Represents the pricing and sales model for a desk lamp in a shopping mall. -/
structure LampSalesModel where
  initial_purchase_price : ℝ
  initial_selling_price : ℝ
  initial_monthly_sales : ℝ
  price_sales_slope : ℝ
  target_monthly_profit : ℝ

/-- Calculates the monthly profit for a given selling price and number of lamps sold. -/
def monthly_profit (model : LampSalesModel) (selling_price : ℝ) (lamps_sold : ℝ) : ℝ :=
  (selling_price - model.initial_purchase_price) * lamps_sold

/-- Calculates the number of lamps sold based on the selling price. -/
def lamps_sold (model : LampSalesModel) (selling_price : ℝ) : ℝ :=
  model.initial_monthly_sales - model.price_sales_slope * (selling_price - model.initial_selling_price)

/-- Theorem stating that the optimal selling price and number of lamps achieve the target monthly profit. -/
theorem optimal_pricing_achieves_target_profit (model : LampSalesModel)
  (h_model : model = {
    initial_purchase_price := 30,
    initial_selling_price := 40,
    initial_monthly_sales := 600,
    price_sales_slope := 10,
    target_monthly_profit := 10000
  })
  (optimal_price : ℝ)
  (optimal_lamps : ℝ)
  (h_price : optimal_price = 50)
  (h_lamps : optimal_lamps = 500) :
  monthly_profit model optimal_price optimal_lamps = model.target_monthly_profit :=
sorry


end NUMINAMATH_CALUDE_optimal_pricing_achieves_target_profit_l4066_406664


namespace NUMINAMATH_CALUDE_function_characterization_l4066_406650

theorem function_characterization
  (f : ℕ → ℕ)
  (h : ∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 →
       f (a + b) = f a + f b + f c + f d) :
  ∀ n : ℕ, f n = n^2 * f 1 :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l4066_406650


namespace NUMINAMATH_CALUDE_labourer_savings_l4066_406639

/-- Calculates the amount saved by a labourer after clearing debt -/
def amount_saved (monthly_income : ℕ) (initial_expense : ℕ) (initial_months : ℕ) (reduced_expense : ℕ) (reduced_months : ℕ) : ℕ :=
  let initial_total_expense := initial_expense * initial_months
  let initial_total_income := monthly_income * initial_months
  let debt := if initial_total_expense > initial_total_income then initial_total_expense - initial_total_income else 0
  let reduced_total_expense := reduced_expense * reduced_months
  let reduced_total_income := monthly_income * reduced_months
  reduced_total_income - (reduced_total_expense + debt)

/-- Theorem stating the amount saved by the labourer -/
theorem labourer_savings :
  amount_saved 81 90 6 60 4 = 30 :=
by sorry

end NUMINAMATH_CALUDE_labourer_savings_l4066_406639


namespace NUMINAMATH_CALUDE_partner_q_investment_time_l4066_406626

/-- The investment time of partner q given the investment and profit ratios -/
theorem partner_q_investment_time
  (investment_ratio : ℚ) -- Ratio of p's investment to q's investment
  (profit_ratio : ℚ) -- Ratio of p's profit to q's profit
  (p_time : ℕ) -- Time p invested in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 10)
  (h3 : p_time = 8) :
  ∃ q_time : ℕ, q_time = 16 ∧ 
  profit_ratio * investment_ratio * q_time = p_time :=
by sorry

end NUMINAMATH_CALUDE_partner_q_investment_time_l4066_406626


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4066_406655

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4066_406655


namespace NUMINAMATH_CALUDE_minimum_distance_problem_l4066_406694

open Real

theorem minimum_distance_problem (a : ℝ) : 
  (∃ x₀ : ℝ, (x₀ - a)^2 + (log (3 * x₀) - 3 * a)^2 ≤ 1/10) → a = 1/30 := by
  sorry

end NUMINAMATH_CALUDE_minimum_distance_problem_l4066_406694


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l4066_406662

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < n → n % m ≠ 0

/-- The roots of a quadratic equation ax^2 + bx + c = 0 are given by (-b ± √(b^2 - 4ac)) / (2a) -/
def isRootOf (x : ℝ) (a b c : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem unique_k_for_prime_roots : ∃! k : ℕ, 
  ∃ p q : ℕ, 
    isPrime p ∧ 
    isPrime q ∧ 
    isRootOf p 1 (-63) k ∧ 
    isRootOf q 1 (-63) k :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l4066_406662


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l4066_406648

/-- The asymptotes of the hyperbola x²/16 - y²/9 = -1 are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/16 - y^2/9 = -1
  ∀ x y : ℝ, (∃ (ε : ℝ), ε > 0 ∧ ∀ δ : ℝ, δ > ε → h (δ * x) (δ * y)) →
    y = (3/4) * x ∨ y = -(3/4) * x :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l4066_406648


namespace NUMINAMATH_CALUDE_broken_flagpole_theorem_l4066_406628

/-- Represents a broken flagpole -/
structure BrokenFlagpole where
  initial_height : ℝ
  tip_height : ℝ
  break_point : ℝ

/-- The condition for a valid broken flagpole configuration -/
def is_valid_broken_flagpole (f : BrokenFlagpole) : Prop :=
  f.initial_height > 0 ∧
  f.tip_height > 0 ∧
  f.tip_height < f.initial_height ∧
  f.break_point > 0 ∧
  f.break_point < f.initial_height ∧
  (f.initial_height - f.break_point) * 2 = f.initial_height - f.tip_height

theorem broken_flagpole_theorem (f : BrokenFlagpole)
  (h_valid : is_valid_broken_flagpole f)
  (h_initial : f.initial_height = 12)
  (h_tip : f.tip_height = 2) :
  f.break_point = 7 := by
sorry

end NUMINAMATH_CALUDE_broken_flagpole_theorem_l4066_406628


namespace NUMINAMATH_CALUDE_minimal_withdrawals_l4066_406623

/-- Represents a withdrawal strategy -/
structure WithdrawalStrategy where
  red : ℕ
  blue : ℕ
  green : ℕ
  count : ℕ

/-- Represents the package of marbles -/
structure MarblePackage where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Checks if a withdrawal strategy is valid according to the constraints -/
def is_valid_strategy (s : WithdrawalStrategy) : Prop :=
  s.red ≤ 1 ∧ s.blue ≤ 2 ∧ s.red + s.blue + s.green ≤ 5

/-- Checks if a list of withdrawal strategies empties the package -/
def empties_package (p : MarblePackage) (strategies : List WithdrawalStrategy) : Prop :=
  strategies.foldl (fun acc s => 
    { red := acc.red - s.red * s.count
    , blue := acc.blue - s.blue * s.count
    , green := acc.green - s.green * s.count
    }) p = ⟨0, 0, 0⟩

/-- The main theorem stating the minimal number of withdrawals -/
theorem minimal_withdrawals (p : MarblePackage) 
  (h_red : p.red = 200) (h_blue : p.blue = 300) (h_green : p.green = 400) :
  ∃ (strategies : List WithdrawalStrategy),
    (∀ s ∈ strategies, is_valid_strategy s) ∧
    empties_package p strategies ∧
    (strategies.foldl (fun acc s => acc + s.count) 0 = 200) ∧
    (∀ (other_strategies : List WithdrawalStrategy),
      (∀ s ∈ other_strategies, is_valid_strategy s) →
      empties_package p other_strategies →
      strategies.foldl (fun acc s => acc + s.count) 0 ≤ 
      other_strategies.foldl (fun acc s => acc + s.count) 0) :=
sorry

end NUMINAMATH_CALUDE_minimal_withdrawals_l4066_406623


namespace NUMINAMATH_CALUDE_square_circle_ratio_l4066_406697

theorem square_circle_ratio : 
  let square_area : ℝ := 784
  let small_circle_circumference : ℝ := 8
  let larger_radius_ratio : ℝ := 7/3

  let square_side : ℝ := Real.sqrt square_area
  let small_circle_radius : ℝ := small_circle_circumference / (2 * Real.pi)
  let large_circle_radius : ℝ := larger_radius_ratio * small_circle_radius

  square_side / large_circle_radius = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_square_circle_ratio_l4066_406697


namespace NUMINAMATH_CALUDE_theater_construction_cost_ratio_l4066_406680

/-- Proves that the ratio of construction cost to land cost is 2:1 given the theater construction scenario --/
theorem theater_construction_cost_ratio :
  let cost_per_sqft : ℝ := 5
  let space_per_seat : ℝ := 12
  let num_seats : ℕ := 500
  let partner_share : ℝ := 0.4
  let tom_spent : ℝ := 54000

  let total_sqft : ℝ := space_per_seat * num_seats
  let land_cost : ℝ := total_sqft * cost_per_sqft
  let total_cost : ℝ := tom_spent / (1 - partner_share)
  let construction_cost : ℝ := total_cost - land_cost

  construction_cost / land_cost = 2 := by sorry

end NUMINAMATH_CALUDE_theater_construction_cost_ratio_l4066_406680


namespace NUMINAMATH_CALUDE_circle_radius_through_triangle_vertices_l4066_406640

theorem circle_radius_through_triangle_vertices (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let r := (max a (max b c)) / 2
  r = 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_through_triangle_vertices_l4066_406640


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l4066_406629

theorem vector_difference_magnitude : ∃ x : ℝ,
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![x, 4]
  (∃ k : ℝ, a = k • b) →
  Real.sqrt ((a 0 - b 0)^2 + (a 1 - b 1)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l4066_406629


namespace NUMINAMATH_CALUDE_square_area_l4066_406641

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2 + 5*x + 6

-- Define the horizontal line
def horizontal_line : ℝ := 10

-- Theorem statement
theorem square_area : ∃ (x₁ x₂ : ℝ), 
  parabola x₁ = horizontal_line ∧ 
  parabola x₂ = horizontal_line ∧ 
  (x₂ - x₁)^2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l4066_406641


namespace NUMINAMATH_CALUDE_total_tables_is_40_l4066_406666

/-- Represents the number of tables and seating capacity in a restaurant --/
structure Restaurant where
  new_tables : ℕ
  original_tables : ℕ
  new_table_capacity : ℕ
  original_table_capacity : ℕ
  total_seating_capacity : ℕ

/-- The conditions of the restaurant problem --/
def restaurant_conditions (r : Restaurant) : Prop :=
  r.new_table_capacity = 6 ∧
  r.original_table_capacity = 4 ∧
  r.total_seating_capacity = 212 ∧
  r.new_tables = r.original_tables + 12 ∧
  r.new_tables * r.new_table_capacity + r.original_tables * r.original_table_capacity = r.total_seating_capacity

/-- The theorem stating that the total number of tables is 40 --/
theorem total_tables_is_40 (r : Restaurant) (h : restaurant_conditions r) : 
  r.new_tables + r.original_tables = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_tables_is_40_l4066_406666


namespace NUMINAMATH_CALUDE_fraction_equals_373_l4066_406692

-- Define the factorization of x^4 + 324
def factor (x : ℤ) : ℤ × ℤ :=
  ((x * (x - 6) + 18), (x * (x + 6) + 18))

-- Define the numerator and denominator sequences
def num_seq : List ℤ := [10, 22, 34, 46, 58]
def den_seq : List ℤ := [4, 16, 28, 40, 52]

-- Define the fraction
def fraction : ℚ :=
  (num_seq.map (λ x => (factor x).1 * (factor x).2)).prod /
  (den_seq.map (λ x => (factor x).1 * (factor x).2)).prod

-- Theorem statement
theorem fraction_equals_373 : fraction = 373 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_373_l4066_406692


namespace NUMINAMATH_CALUDE_greatest_common_divisor_and_sum_of_digits_l4066_406622

def numbers : List Nat := [23115, 34365, 83197, 153589]

def differences (nums : List Nat) : List Nat :=
  List.map (λ (pair : Nat × Nat) => pair.2 - pair.1) (List.zip nums (List.tail nums))

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem greatest_common_divisor_and_sum_of_digits :
  let diffs := differences numbers
  let n := diffs.foldl Nat.gcd (diffs.head!)
  n = 1582 ∧ sumOfDigits n = 16 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_and_sum_of_digits_l4066_406622


namespace NUMINAMATH_CALUDE_root_product_expression_l4066_406679

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 2 = 0) → 
  (β^2 + p*β + 2 = 0) → 
  (γ^2 + q*γ + 3 = 0) → 
  (δ^2 + q*δ + 3 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 3*(q^2 - p^2) := by
sorry

end NUMINAMATH_CALUDE_root_product_expression_l4066_406679


namespace NUMINAMATH_CALUDE_cab_speed_reduction_l4066_406644

theorem cab_speed_reduction (usual_time : ℝ) (delay : ℝ) :
  usual_time = 75 ∧ delay = 15 →
  (usual_time / (usual_time + delay)) = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_cab_speed_reduction_l4066_406644


namespace NUMINAMATH_CALUDE_fishing_tournament_result_l4066_406685

def fishing_tournament (jacob_initial : ℕ) (alex_multiplier emily_multiplier : ℕ) 
  (alex_loss emily_loss : ℕ) : ℕ :=
  let alex_initial := alex_multiplier * jacob_initial
  let emily_initial := emily_multiplier * jacob_initial
  let alex_final := alex_initial - alex_loss
  let emily_final := emily_initial - emily_loss
  let target := max alex_final emily_final + 1
  target - jacob_initial

theorem fishing_tournament_result : 
  fishing_tournament 8 7 3 23 10 = 26 := by sorry

end NUMINAMATH_CALUDE_fishing_tournament_result_l4066_406685


namespace NUMINAMATH_CALUDE_sum_of_perimeters_equals_expected_l4066_406652

/-- Calculates the sum of perimeters of triangles formed by repeatedly
    connecting points 1/3 of the distance along each side of an initial
    equilateral triangle, for a given number of iterations. -/
def sumOfPerimeters (initialSideLength : ℚ) (iterations : ℕ) : ℚ :=
  let rec perimeter (sideLength : ℚ) (n : ℕ) : ℚ :=
    if n = 0 then 0
    else 3 * sideLength + perimeter (sideLength / 3) (n - 1)
  perimeter initialSideLength (iterations + 1)

/-- Theorem stating that the sum of perimeters of triangles formed by
    repeatedly connecting points 1/3 of the distance along each side of
    an initial equilateral triangle with side length 18 units, for 4
    iterations, is equal to 80 2/3 units. -/
theorem sum_of_perimeters_equals_expected :
  sumOfPerimeters 18 4 = 80 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_perimeters_equals_expected_l4066_406652


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_cubes_squared_l4066_406611

theorem largest_divisor_of_difference_of_cubes_squared (k : ℤ) : 
  ∃ (d : ℤ), d = 16 ∧ 
  d ∣ (((2*k+1)^3)^2 - ((2*k-1)^3)^2) ∧ 
  ∀ (n : ℤ), n > d → ¬(∀ (j : ℤ), n ∣ (((2*j+1)^3)^2 - ((2*j-1)^3)^2)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_cubes_squared_l4066_406611


namespace NUMINAMATH_CALUDE_initial_puppies_count_l4066_406636

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has left -/
def puppies_left : ℕ := 5

/-- The initial number of puppies Alyssa had -/
def initial_puppies : ℕ := puppies_given_away + puppies_left

theorem initial_puppies_count : initial_puppies = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l4066_406636


namespace NUMINAMATH_CALUDE_exactly_one_true_iff_or_and_not_and_l4066_406699

theorem exactly_one_true_iff_or_and_not_and (p q : Prop) :
  ((p ∨ q) ∧ ¬(p ∧ q)) ↔ (p ∨ q) ∧ ¬(p ↔ q) := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_true_iff_or_and_not_and_l4066_406699


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l4066_406669

theorem min_distance_to_origin (x y : ℝ) (h : 5 * x + 12 * y - 60 = 0) :
  ∃ (min : ℝ), min = 60 / 13 ∧ ∀ (a b : ℝ), 5 * a + 12 * b - 60 = 0 → min ≤ Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l4066_406669


namespace NUMINAMATH_CALUDE_decimal_25_to_binary_binary_to_decimal_25_l4066_406618

/-- Represents a binary digit (0 or 1) -/
inductive BinaryDigit
| zero : BinaryDigit
| one : BinaryDigit

/-- Represents a binary number as a list of binary digits -/
def BinaryNumber := List BinaryDigit

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : ℕ) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : ℕ :=
  sorry

theorem decimal_25_to_binary :
  decimalToBinary 25 = [BinaryDigit.one, BinaryDigit.one, BinaryDigit.zero, BinaryDigit.zero, BinaryDigit.one] :=
by sorry

theorem binary_to_decimal_25 :
  binaryToDecimal [BinaryDigit.one, BinaryDigit.one, BinaryDigit.zero, BinaryDigit.zero, BinaryDigit.one] = 25 :=
by sorry

end NUMINAMATH_CALUDE_decimal_25_to_binary_binary_to_decimal_25_l4066_406618


namespace NUMINAMATH_CALUDE_wilsons_theorem_l4066_406602

theorem wilsons_theorem (p : ℕ) (hp : Prime p) :
  (((Nat.factorial (p - 1)) : ℤ) % p = -1) ∧
  (p^2 ∣ ((Nat.factorial (p - 1)) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l4066_406602


namespace NUMINAMATH_CALUDE_length_ratio_theorem_l4066_406696

/-- Represents a three-stage rocket with cylindrical stages -/
structure ThreeStageRocket where
  l₁ : ℝ  -- Length of the first stage
  l₂ : ℝ  -- Length of the second stage
  l₃ : ℝ  -- Length of the third stage

/-- The conditions for the three-stage rocket -/
def RocketConditions (r : ThreeStageRocket) : Prop :=
  r.l₂ = (r.l₁ + r.l₃) / 2 ∧
  r.l₂^3 = (6 / 13) * (r.l₁^3 + r.l₃^3)

/-- The theorem stating the ratio of lengths of the first and third stages -/
theorem length_ratio_theorem (r : ThreeStageRocket) (h : RocketConditions r) :
  r.l₁ / r.l₃ = 7 / 5 := by
  sorry


end NUMINAMATH_CALUDE_length_ratio_theorem_l4066_406696


namespace NUMINAMATH_CALUDE_bricks_in_two_walls_l4066_406633

/-- The number of bricks used to build two walls with specified dimensions -/
theorem bricks_in_two_walls 
  (bricks_per_row : ℕ) 
  (rows_per_wall : ℕ) 
  (h1 : bricks_per_row = 30) 
  (h2 : rows_per_wall = 50) : 
  2 * (bricks_per_row * rows_per_wall) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_bricks_in_two_walls_l4066_406633


namespace NUMINAMATH_CALUDE_swimming_running_speed_ratio_l4066_406676

/-- Proves that the ratio of running speed to swimming speed is 4, given the specified conditions --/
theorem swimming_running_speed_ratio :
  ∀ (swimming_speed swimming_time running_time total_distance : ℝ),
  swimming_speed = 2 →
  swimming_time = 2 →
  running_time = swimming_time / 2 →
  total_distance = 12 →
  total_distance = swimming_speed * swimming_time + running_time * (total_distance - swimming_speed * swimming_time) / running_time →
  (total_distance - swimming_speed * swimming_time) / running_time / swimming_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_swimming_running_speed_ratio_l4066_406676


namespace NUMINAMATH_CALUDE_z_magnitude_l4066_406630

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_magnitude : 
  ((z - 2) * i = 1 + i) → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_z_magnitude_l4066_406630


namespace NUMINAMATH_CALUDE_wheel_radius_l4066_406646

theorem wheel_radius (total_distance : ℝ) (revolutions : ℕ) (h1 : total_distance = 798.2857142857142) (h2 : revolutions = 500) :
  ∃ (radius : ℝ), abs (radius - 0.254092376554174) < 0.000000000000001 :=
by
  sorry

end NUMINAMATH_CALUDE_wheel_radius_l4066_406646


namespace NUMINAMATH_CALUDE_positive_integer_triple_characterization_l4066_406698

theorem positive_integer_triple_characterization :
  ∀ (a b c : ℕ+),
    (a.val^2 = 2^b.val + c.val^4) →
    (a.val % 2 = 1 ∨ b.val % 2 = 1 ∨ c.val % 2 = 1) →
    (a.val % 2 = 0 ∨ b.val % 2 = 0) →
    (a.val % 2 = 0 ∨ c.val % 2 = 0) →
    (b.val % 2 = 0 ∨ c.val % 2 = 0) →
    ∃ (n : ℕ+), a.val = 3 * 2^(2*n.val) ∧ b.val = 4*n.val + 3 ∧ c.val = 2^n.val :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_triple_characterization_l4066_406698


namespace NUMINAMATH_CALUDE_largest_coeff_x5_implies_n10_l4066_406637

theorem largest_coeff_x5_implies_n10 (n : ℕ+) :
  (∀ k : ℕ, k ≠ 5 → Nat.choose n 5 ≥ Nat.choose n k) →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_largest_coeff_x5_implies_n10_l4066_406637


namespace NUMINAMATH_CALUDE_min_value_of_b_l4066_406688

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.log (x^2 - 2*a))^2

theorem min_value_of_b :
  ∃ (b : ℝ), (∀ (a : ℝ), ∃ (x₀ : ℝ), x₀ > 0 ∧ f x₀ a ≤ b) ∧
  (∀ (b' : ℝ), (∀ (a : ℝ), ∃ (x₀ : ℝ), x₀ > 0 ∧ f x₀ a ≤ b') → b ≤ b') ∧
  b = 4/5 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_b_l4066_406688


namespace NUMINAMATH_CALUDE_a_value_l4066_406683

-- Define the system of inequalities
def system (x a : ℝ) : Prop :=
  3 * x + a < 0 ∧ 2 * x + 7 > 4 * x - 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 0

-- Theorem statement
theorem a_value (a : ℝ) :
  (∀ x, system x a ↔ solution_set x) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l4066_406683


namespace NUMINAMATH_CALUDE_x_powers_sum_l4066_406606

theorem x_powers_sum (x : ℝ) (h : 47 = x^4 + 1/x^4) : 
  (x^2 + 1/x^2 = 7) ∧ (x^8 + 1/x^8 = -433) := by
  sorry

end NUMINAMATH_CALUDE_x_powers_sum_l4066_406606


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l4066_406670

theorem parabola_focus_distance (p : ℝ) (h1 : p > 0) : 
  let focus : ℝ × ℝ := (p / 2, 0)
  let distance_to_line (point : ℝ × ℝ) : ℝ := 
    |-(point.1) + point.2 - 1| / Real.sqrt 2
  distance_to_line focus = Real.sqrt 2 → p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l4066_406670


namespace NUMINAMATH_CALUDE_max_red_socks_l4066_406691

def is_valid_sock_distribution (r b g : ℕ) : Prop :=
  let t := r + b + g
  t ≤ 2500 ∧
  (r * (r - 1) + b * (b - 1) + g * (g - 1)) = (2 * t * (t - 1)) / 3

theorem max_red_socks :
  ∃ (r b g : ℕ),
    is_valid_sock_distribution r b g ∧
    r = 1625 ∧
    ∀ (r' b' g' : ℕ), is_valid_sock_distribution r' b' g' → r' ≤ r :=
by sorry

end NUMINAMATH_CALUDE_max_red_socks_l4066_406691


namespace NUMINAMATH_CALUDE_good_carrots_count_l4066_406675

theorem good_carrots_count (carol_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : bad_carrots = 7) :
  carol_carrots + mom_carrots - bad_carrots = 38 := by
  sorry

end NUMINAMATH_CALUDE_good_carrots_count_l4066_406675
