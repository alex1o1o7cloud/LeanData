import Mathlib

namespace NUMINAMATH_CALUDE_quadrilateral_with_equal_incircle_radii_is_rhombus_l3199_319960

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The intersection point of the diagonals of a quadrilateral -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- The radius of the incircle of a triangle -/
def incircleRadius (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is a rhombus -/
def isRhombus (q : Quadrilateral) : Prop :=
  sorry

theorem quadrilateral_with_equal_incircle_radii_is_rhombus
  (q : Quadrilateral)
  (h_convex : isConvex q)
  (O : Point)
  (h_O : O = diagonalIntersection q)
  (h_radii : incircleRadius q.A q.B O = incircleRadius q.B q.C O ∧
             incircleRadius q.B q.C O = incircleRadius q.C q.D O ∧
             incircleRadius q.C q.D O = incircleRadius q.D q.A O) :
  isRhombus q :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_with_equal_incircle_radii_is_rhombus_l3199_319960


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3199_319973

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 17) 
  (h2 : x * y = 6) : 
  x^2 + y^2 = 301 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3199_319973


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3199_319906

theorem yellow_balls_count (Y : ℕ) : 
  (Y : ℝ) / (Y + 2) * ((Y - 1) / (Y + 1)) = 1 / 2 → Y = 5 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3199_319906


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_A_intersect_B_B_subset_A_iff_m_range_l3199_319979

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

-- Part 1
theorem union_of_A_and_B : A ∪ B 4 = {x | -2 ≤ x ∧ x < 5} := by sorry

theorem complement_A_intersect_B : (Set.univ \ A) ∩ B 4 = {x | 4 < x ∧ x < 5} := by sorry

-- Part 2
theorem B_subset_A_iff_m_range : 
  ∀ m : ℝ, B m ⊆ A ↔ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_A_intersect_B_B_subset_A_iff_m_range_l3199_319979


namespace NUMINAMATH_CALUDE_range_of_a_l3199_319953

def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a : 
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  (∃ a : ℝ, (a < 0 ∨ (1/4 < a ∧ a < 4)) ∧ 
            ∀ b : ℝ, (0 ≤ b ∧ b ≤ 1/4) → b ≠ a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3199_319953


namespace NUMINAMATH_CALUDE_parabola_vertex_l3199_319954

/-- The equation of a parabola in the form 2y^2 + 8y + 3x + 7 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  2 * y^2 + 8 * y + 3 * x + 7 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  eq x y ∧ ∀ x' y', eq x' y' → y ≤ y'

theorem parabola_vertex :
  is_vertex (1/3) (-2) parabola_equation := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3199_319954


namespace NUMINAMATH_CALUDE_product_of_roots_l3199_319991

theorem product_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 6*x₁ + 8 = 0 → x₂^2 - 6*x₂ + 8 = 0 → x₁ * x₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3199_319991


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_theorem_l3199_319993

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 8

-- Define the line
def Line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the intersection points
def Intersects (k m : ℝ) (P Q : ℝ × ℝ) : Prop :=
  Line k m P.1 P.2 ∧ Line k m Q.1 Q.2 ∧
  Ellipse P.1 P.2 ∧ Ellipse Q.1 Q.2

-- Define the x-axis and y-axis intersection points
def AxisIntersections (k m : ℝ) (C D : ℝ × ℝ) : Prop :=
  C = (-m/k, 0) ∧ D = (0, m)

-- Define the trisection condition
def Trisection (O P Q C D : ℝ × ℝ) : Prop :=
  (D.1 - O.1, D.2 - O.2) = (1/3 * (P.1 - O.1), 1/3 * (P.2 - O.2)) + (2/3 * (Q.1 - O.1), 2/3 * (Q.2 - O.2)) ∧
  (C.1 - O.1, C.2 - O.2) = (1/3 * (Q.1 - O.1), 1/3 * (Q.2 - O.2)) + (2/3 * (P.1 - O.1), 2/3 * (P.2 - O.2))

theorem ellipse_line_intersection_theorem :
  ∃ (k m : ℝ) (P Q C D : ℝ × ℝ),
    Intersects k m P Q ∧
    AxisIntersections k m C D ∧
    Trisection (0, 0) P Q C D :=
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_theorem_l3199_319993


namespace NUMINAMATH_CALUDE_height_to_hypotenuse_not_always_half_l3199_319944

theorem height_to_hypotenuse_not_always_half : ∃ (a b c h : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ h > 0 ∧
  a^2 + b^2 = c^2 ∧  -- right triangle condition
  h ≠ c / 2 ∧        -- height is not half of hypotenuse
  h * c = a * b      -- height formula
  := by sorry

end NUMINAMATH_CALUDE_height_to_hypotenuse_not_always_half_l3199_319944


namespace NUMINAMATH_CALUDE_sixth_year_fee_l3199_319976

def membership_fee (initial_fee : ℕ) (annual_increase : ℕ) (year : ℕ) : ℕ :=
  initial_fee + (year - 1) * annual_increase

theorem sixth_year_fee :
  membership_fee 80 10 6 = 130 := by
  sorry

end NUMINAMATH_CALUDE_sixth_year_fee_l3199_319976


namespace NUMINAMATH_CALUDE_max_parts_5x5_grid_l3199_319995

/-- Represents a partition of a grid into parts with different areas -/
def GridPartition (n : ℕ) := List ℕ

/-- The sum of areas in a partition should equal the total grid area -/
def validPartition (g : ℕ) (p : GridPartition g) : Prop :=
  p.sum = g * g ∧ p.Nodup

/-- The maximum number of parts in a valid partition of a 5x5 grid -/
theorem max_parts_5x5_grid :
  (∃ (p : GridPartition 5), validPartition 5 p ∧ p.length = 6) ∧
  (∀ (p : GridPartition 5), validPartition 5 p → p.length ≤ 6) := by
  sorry

#check max_parts_5x5_grid

end NUMINAMATH_CALUDE_max_parts_5x5_grid_l3199_319995


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_characterization_l3199_319952

/-- A line passing through (1, 3) with equal x and y intercepts -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (1, 3) -/
  point_condition : 3 = m * 1 + b
  /-- The line has equal x and y intercepts -/
  equal_intercepts : b ≠ 0 → -b / m = b

/-- The equation of a line with equal intercepts passing through (1, 3) -/
def equal_intercept_line_equation (l : EqualInterceptLine) : Prop :=
  (l.m = 3 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 4)

/-- Theorem stating that a line with equal intercepts passing through (1, 3) 
    must have the equation 3x - y = 0 or x + y - 4 = 0 -/
theorem equal_intercept_line_equation_characterization (l : EqualInterceptLine) :
  equal_intercept_line_equation l := by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_characterization_l3199_319952


namespace NUMINAMATH_CALUDE_additional_earnings_calculation_l3199_319938

/-- Represents the financial data for a company's quarterly earnings and dividends. -/
structure CompanyFinancials where
  expectedEarnings : ℝ
  actualEarnings : ℝ
  additionalDividendRate : ℝ

/-- Calculates the additional earnings per share based on the company's financial data. -/
def additionalEarnings (cf : CompanyFinancials) : ℝ :=
  cf.actualEarnings - cf.expectedEarnings

/-- Theorem stating that the additional earnings per share is the difference between
    actual and expected earnings. -/
theorem additional_earnings_calculation (cf : CompanyFinancials) 
    (h1 : cf.expectedEarnings = 0.80)
    (h2 : cf.actualEarnings = 1.10)
    (h3 : cf.additionalDividendRate = 0.04) :
    additionalEarnings cf = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_additional_earnings_calculation_l3199_319938


namespace NUMINAMATH_CALUDE_same_club_probability_l3199_319941

theorem same_club_probability :
  let num_students : ℕ := 2
  let num_clubs : ℕ := 3
  let total_outcomes : ℕ := num_clubs ^ num_students
  let favorable_outcomes : ℕ := num_clubs
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_same_club_probability_l3199_319941


namespace NUMINAMATH_CALUDE_cubic_system_solution_l3199_319966

theorem cubic_system_solution (a b c : ℝ) : 
  a + b + c = 3 ∧ 
  a^2 + b^2 + c^2 = 35 ∧ 
  a^3 + b^3 + c^3 = 99 → 
  ({a, b, c} : Set ℝ) = {1, -3, 5} :=
sorry

end NUMINAMATH_CALUDE_cubic_system_solution_l3199_319966


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l3199_319955

theorem max_value_sum_of_roots (a b c : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 7 → 
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ 3 * Real.sqrt 23 ∧
  ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 7 ∧
    Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) = 3 * Real.sqrt 23 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l3199_319955


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l3199_319963

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x + y = 0

-- Define the circle passing through two points
def circle_passes_through (center_x center_y : ℝ) (point1_x point1_y point2_x point2_y : ℝ) : Prop :=
  (center_x - point1_x)^2 + (center_y - point1_y)^2 = (center_x - point2_x)^2 + (center_y - point2_y)^2

-- Theorem statement
theorem circle_radius_is_five :
  ∃ (center_x center_y : ℝ),
    line_equation center_x center_y ∧
    circle_passes_through center_x center_y 1 3 4 2 ∧
    ((center_x - 1)^2 + (center_y - 3)^2)^(1/2 : ℝ) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l3199_319963


namespace NUMINAMATH_CALUDE_units_digit_of_power_l3199_319942

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The base number -/
def base : ℕ := 5689

/-- The exponent -/
def exponent : ℕ := 439

theorem units_digit_of_power : units_digit (base ^ exponent) = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_l3199_319942


namespace NUMINAMATH_CALUDE_bags_filled_on_sunday_l3199_319928

/-- Given the total number of cans collected, cans per bag, and bags filled on Saturday,
    calculate the number of bags filled on Sunday. -/
theorem bags_filled_on_sunday
  (total_cans : ℕ)
  (cans_per_bag : ℕ)
  (bags_on_saturday : ℕ)
  (h1 : total_cans = 63)
  (h2 : cans_per_bag = 9)
  (h3 : bags_on_saturday = 3) :
  total_cans / cans_per_bag - bags_on_saturday = 4 := by
  sorry

end NUMINAMATH_CALUDE_bags_filled_on_sunday_l3199_319928


namespace NUMINAMATH_CALUDE_norris_balance_proof_l3199_319909

/-- Calculates the total savings with interest for Norris --/
def total_savings_with_interest (savings : List ℚ) (interest_rate : ℚ) : ℚ :=
  let base_savings := savings.sum
  let interest := 
    savings.take 4 -- Exclude January's savings from interest calculation
      |> List.scanl (λ acc x => acc + x) 0
      |> List.tail!
      |> List.map (λ x => x * interest_rate)
      |> List.sum
  base_savings + interest

/-- Calculates Norris's final balance --/
def norris_final_balance (savings : List ℚ) (interest_rate : ℚ) (loan_amount : ℚ) (repayment : ℚ) : ℚ :=
  total_savings_with_interest savings interest_rate + (loan_amount - repayment)

theorem norris_balance_proof (savings : List ℚ) (interest_rate : ℚ) (loan_amount : ℚ) (repayment : ℚ) :
  savings = [29, 25, 31, 35, 40] ∧ 
  interest_rate = 2 / 100 ∧
  loan_amount = 20 ∧
  repayment = 10 →
  norris_final_balance savings interest_rate loan_amount repayment = 175.76 := by
  sorry

end NUMINAMATH_CALUDE_norris_balance_proof_l3199_319909


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3199_319983

theorem gcd_of_powers_of_two : Nat.gcd (2^2016 - 1) (2^2008 - 1) = 2^8 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3199_319983


namespace NUMINAMATH_CALUDE_fifth_equation_is_correct_l3199_319964

-- Define the sequence of equations
def equation (n : ℕ) : Prop :=
  match n with
  | 1 => 2^1 * 1 = 2
  | 2 => 2^2 * 1 * 3 = 3 * 4
  | 3 => 2^3 * 1 * 3 * 5 = 4 * 5 * 6
  | 5 => 2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10
  | _ => True

-- Theorem statement
theorem fifth_equation_is_correct :
  equation 1 ∧ equation 2 ∧ equation 3 → equation 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_equation_is_correct_l3199_319964


namespace NUMINAMATH_CALUDE_probability_both_selected_l3199_319945

theorem probability_both_selected (prob_ram : ℚ) (prob_ravi : ℚ) 
  (h1 : prob_ram = 3/7) (h2 : prob_ravi = 1/5) : 
  prob_ram * prob_ravi = 3/35 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l3199_319945


namespace NUMINAMATH_CALUDE_cleaning_assignment_cases_l3199_319901

def number_of_people : ℕ := 6
def people_for_floor : ℕ := 2
def people_for_window : ℕ := 1

theorem cleaning_assignment_cases :
  (Nat.choose (number_of_people - 1) (people_for_floor - 1)) *
  (Nat.choose (number_of_people - people_for_floor) people_for_window) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_assignment_cases_l3199_319901


namespace NUMINAMATH_CALUDE_scientific_notation_1200_l3199_319932

theorem scientific_notation_1200 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1200 = a * (10 : ℝ) ^ n ∧ a = 1.2 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_1200_l3199_319932


namespace NUMINAMATH_CALUDE_third_roll_five_prob_l3199_319926

/-- Represents a six-sided die --/
inductive Die
| Fair
| Biased

/-- Probability of rolling a five for a given die --/
def prob_five (d : Die) : ℚ :=
  match d with
  | Die.Fair => 1/6
  | Die.Biased => 3/4

/-- Probability of rolling a non-five for a given die --/
def prob_not_five (d : Die) : ℚ :=
  match d with
  | Die.Fair => 5/6
  | Die.Biased => 1/4

/-- Probability of choosing each die initially --/
def initial_prob : ℚ := 1/2

/-- Theorem stating the probability of rolling a five on the third roll --/
theorem third_roll_five_prob :
  let p_fair := initial_prob * (prob_five Die.Fair)^2
  let p_biased := initial_prob * (prob_five Die.Biased)^2
  let p_fair_given_two_fives := p_fair / (p_fair + p_biased)
  let p_biased_given_two_fives := p_biased / (p_fair + p_biased)
  p_fair_given_two_fives * (prob_five Die.Fair) + 
  p_biased_given_two_fives * (prob_five Die.Biased) = 223/74 := by
  sorry

end NUMINAMATH_CALUDE_third_roll_five_prob_l3199_319926


namespace NUMINAMATH_CALUDE_donuts_for_class_l3199_319961

theorem donuts_for_class (total_students : ℕ) (donut_likers_percentage : ℚ) (donuts_per_student : ℕ) : 
  total_students = 30 →
  donut_likers_percentage = 4/5 →
  donuts_per_student = 2 →
  (↑total_students * donut_likers_percentage * ↑donuts_per_student) / 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_donuts_for_class_l3199_319961


namespace NUMINAMATH_CALUDE_second_char_lines_relation_l3199_319981

/-- Represents a character in a script with a certain number of lines. -/
structure Character where
  lines : ℕ

/-- Represents a script with three characters. -/
structure Script where
  char1 : Character
  char2 : Character
  char3 : Character
  first_has_more : char1.lines = char2.lines + 8
  third_has_two : char3.lines = 2
  first_has_twenty : char1.lines = 20

/-- The theorem stating the relationship between the lines of the second and third characters. -/
theorem second_char_lines_relation (script : Script) : 
  script.char2.lines = 3 * script.char3.lines + 6 := by
  sorry

end NUMINAMATH_CALUDE_second_char_lines_relation_l3199_319981


namespace NUMINAMATH_CALUDE_correct_calculation_l3199_319951

theorem correct_calculation (x : ℚ) (h : x / 6 = 12) : x * 7 = 504 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3199_319951


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l3199_319974

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l3199_319974


namespace NUMINAMATH_CALUDE_apple_difference_l3199_319912

def apple_contest (aaron bella claire daniel edward fiona george hannah : ℕ) : Prop :=
  aaron = 5 ∧ bella = 3 ∧ claire = 7 ∧ daniel = 2 ∧ edward = 4 ∧ fiona = 3 ∧ george = 1 ∧ hannah = 6 ∧
  claire ≥ aaron ∧ claire ≥ bella ∧ claire ≥ daniel ∧ claire ≥ edward ∧ claire ≥ fiona ∧ claire ≥ george ∧ claire ≥ hannah ∧
  aaron ≥ bella ∧ aaron ≥ daniel ∧ aaron ≥ edward ∧ aaron ≥ fiona ∧ aaron ≥ george ∧ aaron ≥ hannah ∧
  george ≤ aaron ∧ george ≤ bella ∧ george ≤ claire ∧ george ≤ daniel ∧ george ≤ edward ∧ george ≤ fiona ∧ george ≤ hannah

theorem apple_difference (aaron bella claire daniel edward fiona george hannah : ℕ) :
  apple_contest aaron bella claire daniel edward fiona george hannah →
  claire - george = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l3199_319912


namespace NUMINAMATH_CALUDE_man_son_age_difference_l3199_319982

/-- Given a man and his son, proves that the man is 20 years older than his son. -/
theorem man_son_age_difference (man_age son_age : ℕ) : 
  son_age = 18 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_difference_l3199_319982


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l3199_319998

theorem arithmetic_sequence_count : 
  ∀ (a₁ : ℕ) (aₙ : ℕ) (d : ℕ),
    a₁ = 2 → aₙ = 2010 → d = 4 →
    ∃ (n : ℕ), n = 503 ∧ aₙ = a₁ + (n - 1) * d :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l3199_319998


namespace NUMINAMATH_CALUDE_total_movies_count_l3199_319930

/-- The number of times Timothy and Theresa went to the movies in 2009 and 2010 -/
def total_movies (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ) : ℕ :=
  timothy_2009 + timothy_2010 + theresa_2009 + theresa_2010

/-- Theorem stating the total number of movies Timothy and Theresa saw in 2009 and 2010 -/
theorem total_movies_count : 
  ∀ (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ),
    timothy_2009 = 24 →
    timothy_2010 = timothy_2009 + 7 →
    theresa_2009 = timothy_2009 / 2 →
    theresa_2010 = 2 * timothy_2010 →
    total_movies timothy_2009 timothy_2010 theresa_2009 theresa_2010 = 129 := by
  sorry

#check total_movies_count

end NUMINAMATH_CALUDE_total_movies_count_l3199_319930


namespace NUMINAMATH_CALUDE_max_average_growth_rate_l3199_319922

theorem max_average_growth_rate
  (P₁ P₂ M : ℝ)
  (h_sum : P₁ + P₂ = M)
  (h_nonneg : 0 ≤ P₁ ∧ 0 ≤ P₂)
  (P : ℝ)
  (h_avg_growth : (1 + P)^2 = (1 + P₁) * (1 + P₂)) :
  P ≤ M / 2 :=
sorry

end NUMINAMATH_CALUDE_max_average_growth_rate_l3199_319922


namespace NUMINAMATH_CALUDE_hexagon_area_l3199_319950

-- Define the hexagon vertices
def hexagon_vertices : List (ℝ × ℝ) := [(0,0), (1,4), (3,4), (4,0), (3,-4), (1,-4)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem hexagon_area : polygon_area hexagon_vertices = 24 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_l3199_319950


namespace NUMINAMATH_CALUDE_cistern_width_l3199_319924

/-- Given a cistern with the following properties:
  * length: 10 meters
  * water depth: 1.35 meters
  * total wet surface area: 103.2 square meters
  Prove that the width of the cistern is 6 meters. -/
theorem cistern_width (length : ℝ) (water_depth : ℝ) (wet_surface_area : ℝ) :
  length = 10 →
  water_depth = 1.35 →
  wet_surface_area = 103.2 →
  ∃ (width : ℝ), 
    wet_surface_area = length * width + 2 * length * water_depth + 2 * width * water_depth ∧
    width = 6 :=
by sorry

end NUMINAMATH_CALUDE_cistern_width_l3199_319924


namespace NUMINAMATH_CALUDE_paintings_distribution_l3199_319947

/-- Given a total number of paintings, number of rooms, and paintings kept in a private study,
    calculate the number of paintings placed in each room. -/
def paintings_per_room (total : ℕ) (rooms : ℕ) (kept : ℕ) : ℕ :=
  (total - kept) / rooms

/-- Theorem stating that given 47 total paintings, 6 rooms, and 5 paintings kept in a private study,
    the number of paintings placed in each room is 7. -/
theorem paintings_distribution :
  paintings_per_room 47 6 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_paintings_distribution_l3199_319947


namespace NUMINAMATH_CALUDE_max_area_difference_l3199_319902

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem stating the maximum area difference between two rectangles -/
theorem max_area_difference :
  ∃ (r1 r2 : Rectangle),
    perimeter r1 = 200 ∧
    perimeter r2 = 200 ∧
    r2.width = 20 ∧
    ∀ (r3 r4 : Rectangle),
      perimeter r3 = 200 →
      perimeter r4 = 200 →
      r4.width = 20 →
      area r1 - area r2 ≥ area r3 - area r4 ∧
      area r1 - area r2 = 900 :=
sorry

end NUMINAMATH_CALUDE_max_area_difference_l3199_319902


namespace NUMINAMATH_CALUDE_kate_stickers_l3199_319929

/-- Given that the ratio of Kate's stickers to Jenna's stickers is 7:4 and Jenna has 12 stickers,
    prove that Kate has 21 stickers. -/
theorem kate_stickers (kate_stickers : ℕ) (jenna_stickers : ℕ) 
    (h1 : jenna_stickers = 12)
    (h2 : kate_stickers * 4 = jenna_stickers * 7) : 
  kate_stickers = 21 := by
  sorry

end NUMINAMATH_CALUDE_kate_stickers_l3199_319929


namespace NUMINAMATH_CALUDE_geometric_sum_n1_l3199_319904

theorem geometric_sum_n1 (a : ℝ) (h : a ≠ 1) :
  1 + a + a^2 + a^3 = (1 - a^4) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_n1_l3199_319904


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3199_319915

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 1365/4096 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3199_319915


namespace NUMINAMATH_CALUDE_elizabeth_borrowed_53_cents_l3199_319933

/-- The amount Elizabeth borrowed from her neighbor -/
def amount_borrowed : ℕ := by sorry

theorem elizabeth_borrowed_53_cents :
  let pencil_cost : ℕ := 600  -- in cents
  let elizabeth_has : ℕ := 500  -- in cents
  let needs_more : ℕ := 47  -- in cents
  amount_borrowed = pencil_cost - elizabeth_has - needs_more :=
by sorry

end NUMINAMATH_CALUDE_elizabeth_borrowed_53_cents_l3199_319933


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3199_319925

theorem polynomial_remainder_theorem (p : ℝ → ℝ) (hp1 : p 1 = 5) (hp3 : p 3 = 8) :
  ∃ (t : ℝ), ∃ (q : ℝ → ℝ), 
    ∀ x, p x = q x * ((x - 1) * (x - 3) * (x - 5)) + 
              (t * x^2 + (3 - 8*t)/2 * x + (7 + 6*t)/2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3199_319925


namespace NUMINAMATH_CALUDE_monotonic_quadratic_range_l3199_319984

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The function f is monotonic on the interval [-4, 4] -/
def is_monotonic_on_interval (a : ℝ) : Prop :=
  (∀ x y, -4 ≤ x ∧ x < y ∧ y ≤ 4 → f a x < f a y) ∨
  (∀ x y, -4 ≤ x ∧ x < y ∧ y ≤ 4 → f a x > f a y)

/-- If f(x) = x^2 + 2(a-1)x + 2 is monotonic on the interval [-4, 4], then a ≤ -3 or a ≥ 5 -/
theorem monotonic_quadratic_range (a : ℝ) : 
  is_monotonic_on_interval a → a ≤ -3 ∨ a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_range_l3199_319984


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l3199_319905

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^4 - 5 * X^3 + 3 * X^2 + 11 * X - 6
  let divisor : Polynomial ℚ := 5 * X^2 + 7
  let quotient : Polynomial ℚ := 2 * X^2 - X - 11/5
  (dividend : Polynomial ℚ).div divisor = quotient := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l3199_319905


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_hyperbola_l3199_319913

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - 3*x + 2 > 0) ↔ (x < 1 ∨ x > b)

-- Define the main theorem
theorem quadratic_inequality_and_hyperbola (a b : ℝ) :
  solution_set a b →
  (∀ x y : ℝ, x > 0 → y > 0 → a/x + b/y = 1 →
    (∀ k : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → a/x + b/y = 1 → 2*x + y ≥ k) → k ≤ 8)) →
  a = 1 ∧ b = 2 := by
sorry


end NUMINAMATH_CALUDE_quadratic_inequality_and_hyperbola_l3199_319913


namespace NUMINAMATH_CALUDE_base_equality_l3199_319923

/-- Given a positive integer b, converts the base-b number 101ᵦ to base 10 -/
def base_b_to_decimal (b : ℕ) : ℕ := b^2 + 1

/-- Converts 24₅ to base 10 -/
def base_5_to_decimal : ℕ := 2 * 5 + 4

/-- The theorem states that 4 is the unique positive integer b that satisfies 24₅ = 101ᵦ -/
theorem base_equality : ∃! (b : ℕ), b > 0 ∧ base_5_to_decimal = base_b_to_decimal b :=
sorry

end NUMINAMATH_CALUDE_base_equality_l3199_319923


namespace NUMINAMATH_CALUDE_alien_species_count_l3199_319994

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def alienSpeciesBase7 : List Nat := [5, 1, 2]

/-- The theorem stating that the base 7 number 215₇ is equal to 110 in base 10 --/
theorem alien_species_count : base7ToBase10 alienSpeciesBase7 = 110 := by
  sorry

end NUMINAMATH_CALUDE_alien_species_count_l3199_319994


namespace NUMINAMATH_CALUDE_triangle_2_3_4_l3199_319914

-- Define the triangle operation
def triangle (a b c : ℝ) : ℝ := b^3 - 5*a*c

-- Theorem statement
theorem triangle_2_3_4 : triangle 2 3 4 = -13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_2_3_4_l3199_319914


namespace NUMINAMATH_CALUDE_special_rectangle_area_l3199_319965

/-- Represents a rectangle with a diagonal of length y and length three times its width -/
structure SpecialRectangle where
  y : ℝ  -- diagonal length
  w : ℝ  -- width
  h : ℝ  -- height (length)
  h_eq : h = 3 * w  -- length is three times the width
  diag_eq : y^2 = h^2 + w^2  -- Pythagorean theorem for the diagonal

/-- The area of a SpecialRectangle is 3y^2/10 -/
theorem special_rectangle_area (rect : SpecialRectangle) :
  rect.w * rect.h = (3 * rect.y^2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_l3199_319965


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3199_319970

/-- Given a geometric sequence {a_n} where all terms are positive,
    if a_3 * a_5 + a_2 * a_10 + 2 * a_4 * a_6 = 100,
    then a_4 + a_6 = 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geo : ∀ n m : ℕ, a (n + m) = a n * (a 2) ^ (m - 1))
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_sum : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) :
  a 4 + a 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3199_319970


namespace NUMINAMATH_CALUDE_number_multiplied_by_six_l3199_319916

theorem number_multiplied_by_six (n : ℚ) : n / 11 = 2 → n * 6 = 132 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_six_l3199_319916


namespace NUMINAMATH_CALUDE_ellipse_focus_circle_radius_l3199_319990

/-- The radius of a circle centered at a focus of an ellipse and tangent to it -/
theorem ellipse_focus_circle_radius 
  (a b : ℝ) 
  (ha : a = 6) 
  (hb : b = 5) 
  (h_ellipse : a > b) 
  (h_positive : a > 0 ∧ b > 0) :
  let c := Real.sqrt (a^2 - b^2)
  let r := Real.sqrt ((a + c)^2 - a^2)
  r = Real.sqrt 705 / 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_circle_radius_l3199_319990


namespace NUMINAMATH_CALUDE_complement_union_and_intersection_l3199_319908

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 5}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 9}

-- Define the complement of a set in ℝ
def complement (S : Set ℝ) : Set ℝ := {x : ℝ | x ∉ S}

-- State the theorem
theorem complement_union_and_intersection :
  (complement (A ∪ B) = {x : ℝ | x ≤ 2 ∨ x ≥ 9}) ∧
  (complement (A ∩ B) = {x : ℝ | x < 3 ∨ x ≥ 5}) := by
  sorry

end NUMINAMATH_CALUDE_complement_union_and_intersection_l3199_319908


namespace NUMINAMATH_CALUDE_a_6_equals_8_l3199_319988

def S (n : ℕ+) : ℤ := n^2 - 3*n

theorem a_6_equals_8 : ∃ (a : ℕ+ → ℤ), a 6 = 8 ∧ ∀ n : ℕ+, S n - S (n-1) = a n :=
sorry

end NUMINAMATH_CALUDE_a_6_equals_8_l3199_319988


namespace NUMINAMATH_CALUDE_final_price_after_two_reductions_l3199_319948

/-- Given an original price and two identical percentage reductions, 
    calculate the final price after the reductions. -/
def final_price (original_price : ℝ) (reduction_percentage : ℝ) : ℝ :=
  original_price * (1 - reduction_percentage)^2

/-- Theorem stating that for a product with original price $100 and 
    two reductions of percentage m, the final price is 100(1-m)^2 -/
theorem final_price_after_two_reductions (m : ℝ) :
  final_price 100 m = 100 * (1 - m)^2 := by
  sorry

end NUMINAMATH_CALUDE_final_price_after_two_reductions_l3199_319948


namespace NUMINAMATH_CALUDE_negation_equivalence_l3199_319977

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3199_319977


namespace NUMINAMATH_CALUDE_sequence_properties_l3199_319989

/-- Given a sequence {a_n} with sum of first n terms S_n, satisfying a_n = 2S_n + 1 for n ∈ ℕ* -/
def a (n : ℕ) : ℤ := sorry

/-- Sum of first n terms of sequence {a_n} -/
def S (n : ℕ) : ℤ := sorry

/-- Sequence {b_n} defined as b_n = (2n-1) * a_n -/
def b (n : ℕ) : ℤ := sorry

/-- Sum of first n terms of sequence {b_n} -/
def T (n : ℕ) : ℤ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → a n = 2 * S n + 1) →
  (∀ n : ℕ, a n = (-1)^n) ∧
  (∀ n : ℕ, T n = (-1)^n * n) := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3199_319989


namespace NUMINAMATH_CALUDE_afternoon_rowing_count_l3199_319996

/-- The number of campers who went rowing in the afternoon -/
def afternoon_rowing (morning_rowing hiking total : ℕ) : ℕ :=
  total - (morning_rowing + hiking)

/-- Theorem stating that 26 campers went rowing in the afternoon -/
theorem afternoon_rowing_count :
  afternoon_rowing 41 4 71 = 26 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_rowing_count_l3199_319996


namespace NUMINAMATH_CALUDE_backpack_player_prices_l3199_319900

/-- Represents the prices and discounts for the backpack and portable music player problem -/
structure PriceInfo where
  backpack_price : ℕ
  player_price : ℕ
  renmin_discount : Rat
  carrefour_voucher : ℕ
  carrefour_voucher_threshold : ℕ
  budget : ℕ

/-- Calculates the total price at Renmin Department Store after discount -/
def renmin_total (info : PriceInfo) : Rat :=
  (info.backpack_price + info.player_price : Rat) * info.renmin_discount

/-- Calculates the total price at Carrefour after applying vouchers -/
def carrefour_total (info : PriceInfo) : ℕ :=
  info.player_price + info.backpack_price - 
    ((info.player_price + info.backpack_price) / info.carrefour_voucher_threshold) * info.carrefour_voucher

/-- The main theorem stating the correct prices and the more cost-effective store -/
theorem backpack_player_prices (info : PriceInfo) : 
  info.backpack_price = 92 ∧ 
  info.player_price = 360 ∧ 
  info.backpack_price + info.player_price = 452 ∧
  info.player_price = 4 * info.backpack_price - 8 ∧
  info.renmin_discount = 4/5 ∧
  info.carrefour_voucher = 30 ∧
  info.carrefour_voucher_threshold = 100 ∧
  info.budget = 400 →
  renmin_total info < carrefour_total info ∧
  renmin_total info ≤ info.budget ∧
  (carrefour_total info : Rat) ≤ info.budget := by
  sorry

end NUMINAMATH_CALUDE_backpack_player_prices_l3199_319900


namespace NUMINAMATH_CALUDE_find_number_l3199_319980

theorem find_number : ∃ n : ℤ, 695 - 329 = n - 254 ∧ n = 620 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3199_319980


namespace NUMINAMATH_CALUDE_average_age_increase_proof_l3199_319931

/-- The initial number of men in a group where replacing two men with two women increases the average age by 2 years -/
def initial_men_count : ℕ := 8

theorem average_age_increase_proof :
  let men_removed_age_sum := 20 + 28
  let women_added_age_sum := 32 + 32
  let age_difference := women_added_age_sum - men_removed_age_sum
  let average_age_increase := 2
  initial_men_count * average_age_increase = age_difference := by
  sorry

#check average_age_increase_proof

end NUMINAMATH_CALUDE_average_age_increase_proof_l3199_319931


namespace NUMINAMATH_CALUDE_xy_value_l3199_319910

theorem xy_value (x y : ℝ) (h : (x - 3)^2 + |y + 2| = 0) : x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3199_319910


namespace NUMINAMATH_CALUDE_magnitude_of_z_l3199_319946

/-- The magnitude of the complex number z = (1-i)/i is √2 -/
theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.abs ((1 - i) / i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l3199_319946


namespace NUMINAMATH_CALUDE_arithmetic_seq_first_term_arithmetic_seq_first_term_range_l3199_319967

/-- Arithmetic sequence with common difference -1 -/
def ArithmeticSeq (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => ArithmeticSeq a₁ n - 1

/-- Sum of first n terms of the arithmetic sequence -/
def SumSeq (a₁ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => SumSeq a₁ n + ArithmeticSeq a₁ (n + 1)

theorem arithmetic_seq_first_term (a₁ : ℝ) :
  SumSeq a₁ 5 = -5 → a₁ = 1 := by sorry

theorem arithmetic_seq_first_term_range (a₁ : ℝ) :
  (∀ n : ℕ, n > 0 → SumSeq a₁ n ≤ ArithmeticSeq a₁ n) → a₁ ≤ 0 := by sorry

end NUMINAMATH_CALUDE_arithmetic_seq_first_term_arithmetic_seq_first_term_range_l3199_319967


namespace NUMINAMATH_CALUDE_selection_theorem_l3199_319927

/-- Represents the number of students who can play only chess -/
def chess_only : ℕ := 2

/-- Represents the number of students who can play only Go -/
def go_only : ℕ := 3

/-- Represents the number of students who can play both chess and Go -/
def both : ℕ := 4

/-- Represents the total number of students -/
def total_students : ℕ := chess_only + go_only + both

/-- Calculates the number of ways to select two students for chess and Go competitions -/
def selection_ways : ℕ :=
  chess_only * go_only +  -- One from chess_only, one from go_only
  both * go_only +        -- One from both for chess, one from go_only
  chess_only * both +     -- One from chess_only, one from both for Go
  (both * (both - 1)) / 2 -- Two from both (combination)

/-- Theorem stating that the number of ways to select students is 32 -/
theorem selection_theorem : selection_ways = 32 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l3199_319927


namespace NUMINAMATH_CALUDE_sinusoid_amplitude_l3199_319968

/-- 
Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
if the function oscillates between 5 and -3, then the amplitude a is equal to 4.
-/
theorem sinusoid_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_sinusoid_amplitude_l3199_319968


namespace NUMINAMATH_CALUDE_area_in_three_triangles_l3199_319943

/-- Given a 6 by 8 rectangle with equilateral triangles on each side, 
    this function calculates the area of regions in exactly 3 of 4 triangles -/
def areaInThreeTriangles : ℝ := sorry

/-- The rectangle's width -/
def rectangleWidth : ℝ := 6

/-- The rectangle's length -/
def rectangleLength : ℝ := 8

/-- Theorem stating the area calculation -/
theorem area_in_three_triangles :
  areaInThreeTriangles = (288 - 154 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_area_in_three_triangles_l3199_319943


namespace NUMINAMATH_CALUDE_gcd_of_256_162_720_l3199_319962

theorem gcd_of_256_162_720 : Nat.gcd 256 (Nat.gcd 162 720) = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_162_720_l3199_319962


namespace NUMINAMATH_CALUDE_congruent_count_l3199_319907

theorem congruent_count : ∃ (n : ℕ), n = (Finset.filter (fun x => x > 0 ∧ x < 500 ∧ x % 9 = 4) (Finset.range 500)).card ∧ n = 56 := by
  sorry

end NUMINAMATH_CALUDE_congruent_count_l3199_319907


namespace NUMINAMATH_CALUDE_marathon_length_l3199_319969

/-- A marathon runner completes a race under specific conditions. -/
theorem marathon_length (initial_distance : ℝ) (initial_time : ℝ) (total_time : ℝ) 
  (pace_ratio : ℝ) (marathon_length : ℝ) : 
  initial_distance = 10 →
  initial_time = 1 →
  total_time = 3 →
  pace_ratio = 0.8 →
  marathon_length = initial_distance + 
    (total_time - initial_time) * (initial_distance / initial_time) * pace_ratio →
  marathon_length = 26 := by
  sorry

#check marathon_length

end NUMINAMATH_CALUDE_marathon_length_l3199_319969


namespace NUMINAMATH_CALUDE_solution_of_exponential_equation_l3199_319920

theorem solution_of_exponential_equation :
  {x : ℝ | (4 : ℝ) ^ (x^2 + 1) = 16} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_solution_of_exponential_equation_l3199_319920


namespace NUMINAMATH_CALUDE_economics_test_correct_answers_l3199_319921

theorem economics_test_correct_answers 
  (total_students : ℕ) 
  (correct_q1 : ℕ) 
  (correct_q2 : ℕ) 
  (not_taken : ℕ) 
  (h1 : total_students = 25) 
  (h2 : correct_q1 = 22) 
  (h3 : correct_q2 = 20) 
  (h4 : not_taken = 3) :
  (correct_q1 + correct_q2) - (total_students - not_taken) = 20 := by
sorry

end NUMINAMATH_CALUDE_economics_test_correct_answers_l3199_319921


namespace NUMINAMATH_CALUDE_one_third_of_360_l3199_319937

theorem one_third_of_360 : (360 : ℝ) * (1 / 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_360_l3199_319937


namespace NUMINAMATH_CALUDE_tan_70_cos_10_expression_l3199_319999

theorem tan_70_cos_10_expression : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_70_cos_10_expression_l3199_319999


namespace NUMINAMATH_CALUDE_equation_equivalence_product_l3199_319940

theorem equation_equivalence_product (a b x y : ℝ) (m n p q : ℤ) :
  (a^7 * x * y - a^6 * y - a^5 * x = a^3 * (b^4 - 1)) →
  ((a^m * x - a^n) * (a^p * y - a^q) = a^3 * b^4) →
  m * n * p * q = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_product_l3199_319940


namespace NUMINAMATH_CALUDE_age_problem_l3199_319997

/-- Proves that 10 years less than the average age of Mr. Bernard and Luke is 26 years. -/
theorem age_problem (luke_age : ℕ) (bernard_age : ℕ) : luke_age = 20 →
  bernard_age + 8 = 3 * luke_age →
  ((bernard_age + luke_age) / 2) - 10 = 26 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l3199_319997


namespace NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l3199_319935

theorem sphere_cylinder_equal_area (r : ℝ) : 
  (4 : ℝ) * Real.pi * r^2 = (2 : ℝ) * Real.pi * 4 * 8 → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l3199_319935


namespace NUMINAMATH_CALUDE_cos_18_degrees_l3199_319911

theorem cos_18_degrees :
  Real.cos (18 * π / 180) = Real.sqrt ((5 + Real.sqrt 5) / 8) := by
  sorry

end NUMINAMATH_CALUDE_cos_18_degrees_l3199_319911


namespace NUMINAMATH_CALUDE_unique_mod_equivalence_l3199_319987

theorem unique_mod_equivalence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_mod_equivalence_l3199_319987


namespace NUMINAMATH_CALUDE_five_letter_words_count_l3199_319958

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of five-letter words where the first and last letters are the same vowel,
    and the remaining three letters can be any letters from the alphabet -/
def num_words : ℕ := num_vowels * alphabet_size^3

theorem five_letter_words_count : num_words = 87880 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l3199_319958


namespace NUMINAMATH_CALUDE_jamie_quiz_performance_l3199_319957

theorem jamie_quiz_performance (y : ℕ) : 
  let total_questions : ℕ := 8 * y
  let missed_questions : ℕ := 2 * y
  let correct_questions : ℕ := total_questions - missed_questions
  (correct_questions : ℚ) / (total_questions : ℚ) * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_jamie_quiz_performance_l3199_319957


namespace NUMINAMATH_CALUDE_first_player_wins_98_max_n_first_player_wins_l3199_319919

/-- Represents the game board -/
def Board := Fin 1000 → Bool

/-- Represents a player's move -/
inductive Move
| Place (pos : Fin 1000) (num : Nat)
| Remove (start : Fin 1000) (len : Nat)

/-- Represents a player's strategy -/
def Strategy := Board → Move

/-- Applies a move to the board -/
def applyMove (b : Board) (m : Move) : Board :=
  sorry

/-- Checks if all tokens are placed in a row without gaps -/
def isWinningState (b : Board) : Prop :=
  sorry

/-- The game's rules and win condition -/
def gameRules (n : Nat) (s1 s2 : Strategy) : Prop :=
  sorry

/-- Theorem: First player can always win for n = 98 -/
theorem first_player_wins_98 :
  ∃ (s1 : Strategy), ∀ (s2 : Strategy), gameRules 98 s1 s2 → isWinningState (sorry : Board) :=
  sorry

/-- Theorem: 98 is the maximum n for which first player can always win -/
theorem max_n_first_player_wins :
  (∃ (s1 : Strategy), ∀ (s2 : Strategy), gameRules 98 s1 s2 → isWinningState (sorry : Board)) ∧
  (∀ n > 98, ∃ (s2 : Strategy), ∀ (s1 : Strategy), ¬(gameRules n s1 s2 → isWinningState (sorry : Board))) :=
  sorry

end NUMINAMATH_CALUDE_first_player_wins_98_max_n_first_player_wins_l3199_319919


namespace NUMINAMATH_CALUDE_regular_ngon_rotation_forms_regular_2ngon_l3199_319903

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (vertices : Fin n → V)
  (center : V)
  (is_regular : ∀ i j : Fin n, ‖vertices i - center‖ = ‖vertices j - center‖)

/-- Rotation of a vector about a point -/
def rotate (θ : ℝ) (center : V) (v : V) [NormedAddCommGroup V] [InnerProductSpace ℝ V] : V :=
  sorry

/-- The theorem statement -/
theorem regular_ngon_rotation_forms_regular_2ngon
  (n : ℕ) (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (ngon : RegularNGon n V) (θ : ℝ) :
  θ < 2 * Real.pi / n →
  (∃ (m : ℕ), θ = 2 * Real.pi / m) →
  ∃ (circle_center : V) (radius : ℝ),
    ∀ (i : Fin n),
      ‖circle_center - ngon.vertices i‖ = radius ∧
      ‖circle_center - rotate θ ngon.center (ngon.vertices i)‖ = radius :=
sorry

end NUMINAMATH_CALUDE_regular_ngon_rotation_forms_regular_2ngon_l3199_319903


namespace NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l3199_319934

-- Problem 1
theorem simplify_expression (a b : ℝ) : a + 2*b + 3*a - 2*b = 4*a := by sorry

-- Problem 2
theorem simplify_and_evaluate : (2*(2^2) - 3*2*1 + 8) - (5*2*1 - 4*(2^2) + 8) = 8 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l3199_319934


namespace NUMINAMATH_CALUDE_jason_percentage_more_than_zachary_l3199_319975

/-- Proves that Jason received 30% more money than Zachary from selling video games -/
theorem jason_percentage_more_than_zachary 
  (zachary_games : ℕ) 
  (zachary_price : ℚ) 
  (ryan_extra : ℚ) 
  (total_amount : ℚ) 
  (h1 : zachary_games = 40)
  (h2 : zachary_price = 5)
  (h3 : ryan_extra = 50)
  (h4 : total_amount = 770)
  (h5 : zachary_games * zachary_price + 2 * (zachary_games * zachary_price + ryan_extra) / 2 = total_amount) :
  (((zachary_games * zachary_price + ryan_extra) / 2 - zachary_games * zachary_price) / (zachary_games * zachary_price)) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_jason_percentage_more_than_zachary_l3199_319975


namespace NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l3199_319917

theorem zeroth_power_of_nonzero_rational (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l3199_319917


namespace NUMINAMATH_CALUDE_inverse_proposition_absolute_values_l3199_319971

theorem inverse_proposition_absolute_values (a b : ℝ) :
  (∀ x y : ℝ, x = y → |x| = |y|) →
  (∀ x y : ℝ, |x| = |y| → x = y) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_absolute_values_l3199_319971


namespace NUMINAMATH_CALUDE_softball_players_l3199_319949

/-- The number of softball players in a games hour -/
theorem softball_players (cricket hockey football total : ℕ) 
  (h1 : cricket = 10)
  (h2 : hockey = 12)
  (h3 : football = 16)
  (h4 : total = 51)
  (h5 : total = cricket + hockey + football + softball) : 
  softball = 13 := by
  sorry

end NUMINAMATH_CALUDE_softball_players_l3199_319949


namespace NUMINAMATH_CALUDE_equation_solution_l3199_319992

theorem equation_solution : ∃ x : ℝ, (24 - 4 = 3 + x) ∧ (x = 17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3199_319992


namespace NUMINAMATH_CALUDE_cake_muffin_mix_probability_l3199_319972

theorem cake_muffin_mix_probability (total_buyers : ℕ) (cake_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ)
  (h1 : total_buyers = 100)
  (h2 : cake_buyers = 50)
  (h3 : muffin_buyers = 40)
  (h4 : both_buyers = 19) :
  (total_buyers - (cake_buyers + muffin_buyers - both_buyers)) / total_buyers = 29 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_muffin_mix_probability_l3199_319972


namespace NUMINAMATH_CALUDE_next_multiple_remainder_l3199_319956

theorem next_multiple_remainder (N : ℕ) (h : N = 44 * 432) :
  (N + 432) % 39 = 12 := by
  sorry

end NUMINAMATH_CALUDE_next_multiple_remainder_l3199_319956


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3199_319986

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Partial sum of an arithmetic sequence -/
def partial_sum (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_properties
  (a : ℕ+ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : partial_sum a 6 > partial_sum a 7 ∧ partial_sum a 7 > partial_sum a 5) :
  (∃ d : ℝ, d < 0 ∧ ∀ n : ℕ+, a (n + 1) = a n + d) ∧
  partial_sum a 11 > 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3199_319986


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_sum_l3199_319978

theorem pascal_triangle_row20_sum : Nat.choose 20 4 + Nat.choose 20 5 = 20349 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_sum_l3199_319978


namespace NUMINAMATH_CALUDE_pushup_percentage_l3199_319936

def jumping_jacks : ℕ := 12
def pushups : ℕ := 8
def situps : ℕ := 20

def total_exercises : ℕ := jumping_jacks + pushups + situps

theorem pushup_percentage :
  (pushups : ℚ) / (total_exercises : ℚ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pushup_percentage_l3199_319936


namespace NUMINAMATH_CALUDE_pizza_topping_distribution_l3199_319918

/-- Pizza topping distribution problem -/
theorem pizza_topping_distribution 
  (pepperoni : ℕ) 
  (ham : ℕ) 
  (sausage : ℕ) 
  (slices : ℕ) :
  pepperoni = 30 →
  ham = 2 * pepperoni →
  sausage = pepperoni + 12 →
  slices = 6 →
  (pepperoni + ham + sausage) / slices = 22 := by
  sorry

#check pizza_topping_distribution

end NUMINAMATH_CALUDE_pizza_topping_distribution_l3199_319918


namespace NUMINAMATH_CALUDE_triangle_point_coordinates_l3199_319985

/-- Given a triangle ABC with the following properties:
  - A has coordinates (2, 8)
  - M has coordinates (4, 11) and is the midpoint of AB
  - L has coordinates (6, 6) and BL is the angle bisector of angle ABC
  Prove that the coordinates of point C are (6, 14) -/
theorem triangle_point_coordinates (A B C M L : ℝ × ℝ) : 
  A = (2, 8) →
  M = (4, 11) →
  L = (6, 6) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is midpoint of AB
  (L.1 - B.1) * (C.2 - B.2) = (L.2 - B.2) * (C.1 - B.1) →  -- BL is angle bisector
  C = (6, 14) := by sorry

end NUMINAMATH_CALUDE_triangle_point_coordinates_l3199_319985


namespace NUMINAMATH_CALUDE_stadium_entry_fee_l3199_319959

/-- Proves that the entry fee per person is $20 given the stadium conditions --/
theorem stadium_entry_fee (capacity : ℕ) (occupancy_ratio : ℚ) (fee_difference : ℕ) :
  capacity = 2000 →
  occupancy_ratio = 3/4 →
  fee_difference = 10000 →
  ∃ (fee : ℚ), fee = 20 ∧
    (capacity : ℚ) * fee - (capacity : ℚ) * occupancy_ratio * fee = fee_difference :=
by sorry

end NUMINAMATH_CALUDE_stadium_entry_fee_l3199_319959


namespace NUMINAMATH_CALUDE_inequality_solution_l3199_319939

theorem inequality_solution (x : ℝ) (h1 : x > 0) 
  (h2 : x * Real.sqrt (16 - x^2) + Real.sqrt (16*x - x^4) ≥ 16) : 
  x = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3199_319939
