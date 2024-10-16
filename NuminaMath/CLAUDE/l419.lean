import Mathlib

namespace NUMINAMATH_CALUDE_no_constant_term_in_expansion_l419_41914

theorem no_constant_term_in_expansion :
  ∀ k : ℕ, k ≤ 12 →
    (12 - k : ℚ) / 2 - 2 * k ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_constant_term_in_expansion_l419_41914


namespace NUMINAMATH_CALUDE_circle_center_from_intersection_l419_41950

/-- Given a parabola y = k x^2 and a circle x^2 - 2px + y^2 - 2qy = 0,
    if the abscissas of their intersection points are the roots of x^3 + ax + b = 0,
    then the center of the circle is (-b/2, (1-a)/2). -/
theorem circle_center_from_intersection (k a b : ℝ) :
  ∃ (p q : ℝ),
    (∀ x y : ℝ, y = k * x^2 ∧ x^2 - 2*p*x + y^2 - 2*q*y = 0 →
      x^3 + a*x + b = 0) →
    (p = b/2 ∧ q = (a-1)/2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_from_intersection_l419_41950


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_max_area_l419_41924

/-- Given a right triangle with legs of length a and hypotenuse of length c,
    the area of the triangle is maximized when the legs are equal. -/
theorem isosceles_right_triangle_max_area (a c : ℝ) (h1 : 0 < a) (h2 : 0 < c) :
  let area := (1/2) * a * (c^2 - a^2).sqrt
  ∀ b, 0 < b → b^2 + a^2 = c^2 → area ≥ (1/2) * a * b :=
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_max_area_l419_41924


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l419_41960

def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a : ℚ) (b : ℚ) :
  a = 3/4 → b = 1 → arithmetic_sequence a (b - a) 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l419_41960


namespace NUMINAMATH_CALUDE_marys_age_l419_41972

theorem marys_age (mary_age rahul_age : ℕ) 
  (h1 : rahul_age = mary_age + 30)
  (h2 : rahul_age + 20 = 2 * (mary_age + 20)) : 
  mary_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_marys_age_l419_41972


namespace NUMINAMATH_CALUDE_unique_c_for_quadratic_equation_l419_41981

theorem unique_c_for_quadratic_equation :
  ∃! (c : ℝ), c ≠ 0 ∧
    (∃! (b : ℝ), b > 0 ∧
      (∃! (x : ℝ), x^2 + (2*b + 2/b)*x + c = 0)) ∧
    c = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_for_quadratic_equation_l419_41981


namespace NUMINAMATH_CALUDE_units_digit_of_8129_power_1351_l419_41932

theorem units_digit_of_8129_power_1351 : 8129^1351 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_8129_power_1351_l419_41932


namespace NUMINAMATH_CALUDE_intersection_theorem_l419_41982

def setA : Set ℝ := {x | (x + 3) * (x - 1) ≤ 0}

def setB : Set ℝ := {x | ∃ y, y = Real.log (x^2 - x - 2)}

theorem intersection_theorem : 
  setA ∩ (setB.compl) = {x | -1 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l419_41982


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l419_41979

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |x^2 + a*x + 2| ≤ 4) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l419_41979


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l419_41954

theorem geometric_sequence_solution (x : ℝ) :
  (1 : ℝ) * x = x * 9 → x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l419_41954


namespace NUMINAMATH_CALUDE_joseph_investment_result_l419_41920

/-- Calculates the final amount in an investment account after a given number of years,
    with an initial investment, yearly interest rate, and monthly additional deposits. -/
def investment_calculation (initial_investment : ℝ) (interest_rate : ℝ) 
                           (monthly_deposit : ℝ) (years : ℕ) : ℝ :=
  sorry

/-- Theorem stating that the investment calculation for Joseph's scenario
    results in $3982 after two years. -/
theorem joseph_investment_result :
  investment_calculation 1000 0.10 100 2 = 3982 := by
  sorry

end NUMINAMATH_CALUDE_joseph_investment_result_l419_41920


namespace NUMINAMATH_CALUDE_valid_word_count_mod_2000_l419_41916

/-- Represents a letter in Zuminglish --/
inductive ZuminglishLetter
| M
| O
| P

/-- Represents whether a letter is a vowel or consonant --/
def isVowel : ZuminglishLetter → Bool
| ZuminglishLetter.O => true
| _ => false

/-- A Zuminglish word is a list of Zuminglish letters --/
def ZuminglishWord := List ZuminglishLetter

/-- Checks if a Zuminglish word is valid (no two O's are adjacent without at least two consonants in between) --/
def isValidWord : ZuminglishWord → Bool := sorry

/-- Counts the number of valid 12-letter Zuminglish words --/
def countValidWords : Nat := sorry

/-- The main theorem: The number of valid 12-letter Zuminglish words is congruent to 192 modulo 2000 --/
theorem valid_word_count_mod_2000 : countValidWords % 2000 = 192 := by sorry

end NUMINAMATH_CALUDE_valid_word_count_mod_2000_l419_41916


namespace NUMINAMATH_CALUDE_expression_evaluation_l419_41986

theorem expression_evaluation : -1^4 - (1 - 0.5) * (2 - (-3)^2) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l419_41986


namespace NUMINAMATH_CALUDE_manufacturing_sector_degrees_l419_41946

theorem manufacturing_sector_degrees (total_degrees : ℝ) (total_percent : ℝ) 
  (manufacturing_percent : ℝ) (h1 : total_degrees = 360) 
  (h2 : total_percent = 100) (h3 : manufacturing_percent = 60) : 
  (manufacturing_percent / total_percent) * total_degrees = 216 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_sector_degrees_l419_41946


namespace NUMINAMATH_CALUDE_tan_negative_225_degrees_l419_41917

theorem tan_negative_225_degrees : Real.tan (-(225 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_225_degrees_l419_41917


namespace NUMINAMATH_CALUDE_trigonometric_equality_l419_41925

theorem trigonometric_equality (θ α β γ x y z : ℝ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : Real.tan (θ + α) / x = Real.tan (θ + β) / y)
  (h8 : Real.tan (θ + β) / y = Real.tan (θ + γ) / z) : 
  (x + y) / (x - y) * Real.sin (α - β) ^ 2 + 
  (y + z) / (y - z) * Real.sin (β - γ) ^ 2 + 
  (z + x) / (z - x) * Real.sin (γ - α) ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l419_41925


namespace NUMINAMATH_CALUDE_patrick_pencil_purchase_l419_41905

/-- The number of pencils Patrick purchased -/
def num_pencils : ℕ := 60

/-- The ratio of cost price to selling price -/
def cost_to_sell_ratio : ℚ := 1.3333333333333333

/-- The number of pencils whose selling price equals the total loss -/
def loss_in_pencils : ℕ := 20

theorem patrick_pencil_purchase :
  num_pencils = 60 ∧
  (cost_to_sell_ratio - 1) * num_pencils = loss_in_pencils :=
sorry

end NUMINAMATH_CALUDE_patrick_pencil_purchase_l419_41905


namespace NUMINAMATH_CALUDE_inequality_proof_l419_41939

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l419_41939


namespace NUMINAMATH_CALUDE_six_digit_number_problem_l419_41983

theorem six_digit_number_problem : ∃! n : ℕ, 
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ k : ℕ, n = 200000 + k ∧ k < 100000 ∧
  10 * k + 2 = 3 * n ∧
  n = 285714 := by
sorry

end NUMINAMATH_CALUDE_six_digit_number_problem_l419_41983


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_neg_150_l419_41936

theorem largest_multiple_of_15_less_than_neg_150 :
  ∀ n : ℤ, n * 15 < -150 → n * 15 ≤ -165 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_neg_150_l419_41936


namespace NUMINAMATH_CALUDE_optimal_portfolio_l419_41975

/-- Represents an investment project with maximum profit and loss percentages -/
structure Project where
  max_profit : Real
  max_loss : Real

/-- Represents an investment portfolio with amounts invested in two projects -/
structure Portfolio where
  amount_a : Real
  amount_b : Real

def project_a : Project := { max_profit := 1.0, max_loss := 0.3 }
def project_b : Project := { max_profit := 0.5, max_loss := 0.1 }

def total_investment_limit : Real := 100000
def max_allowed_loss : Real := 18000

def portfolio_loss (p : Portfolio) : Real :=
  p.amount_a * project_a.max_loss + p.amount_b * project_b.max_loss

def portfolio_profit (p : Portfolio) : Real :=
  p.amount_a * project_a.max_profit + p.amount_b * project_b.max_profit

def is_valid_portfolio (p : Portfolio) : Prop :=
  p.amount_a ≥ 0 ∧ p.amount_b ≥ 0 ∧
  p.amount_a + p.amount_b ≤ total_investment_limit ∧
  portfolio_loss p ≤ max_allowed_loss

theorem optimal_portfolio :
  ∃ (p : Portfolio), is_valid_portfolio p ∧
    ∀ (q : Portfolio), is_valid_portfolio q → portfolio_profit q ≤ portfolio_profit p :=
  sorry

end NUMINAMATH_CALUDE_optimal_portfolio_l419_41975


namespace NUMINAMATH_CALUDE_expression_evaluation_l419_41947

theorem expression_evaluation : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l419_41947


namespace NUMINAMATH_CALUDE_jesse_carpet_need_l419_41940

/-- The additional carpet needed for Jesse's room -/
def additional_carpet_needed (room_length : ℝ) (room_width : ℝ) (existing_carpet : ℝ) : ℝ :=
  room_length * room_width - existing_carpet

/-- Theorem stating the additional carpet needed for Jesse's room -/
theorem jesse_carpet_need : 
  additional_carpet_needed 11 15 16 = 149 := by
  sorry

end NUMINAMATH_CALUDE_jesse_carpet_need_l419_41940


namespace NUMINAMATH_CALUDE_least_multiple_remainder_l419_41965

theorem least_multiple_remainder (m : ℕ) : 
  (m % 23 = 0) → 
  (m % 1821 = 710) → 
  (m = 3024) → 
  (m % 24 = 0) := by
sorry

end NUMINAMATH_CALUDE_least_multiple_remainder_l419_41965


namespace NUMINAMATH_CALUDE_average_rstp_l419_41970

theorem average_rstp (r s t u : ℝ) (h : (5 / 2) * (r + s + t + u) = 20) :
  (r + s + t + u) / 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_average_rstp_l419_41970


namespace NUMINAMATH_CALUDE_fraction_sum_rounded_l419_41935

theorem fraction_sum_rounded : 
  let sum := (3 : ℚ) / 20 + 7 / 200 + 8 / 2000 + 3 / 20000
  round (sum * 10000) / 10000 = (1892 : ℚ) / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_rounded_l419_41935


namespace NUMINAMATH_CALUDE_line_parameterization_l419_41993

/-- Given a line y = 2x + 5 parameterized as (x, y) = (r, -3) + t(5, k),
    prove that r = -4 and k = 10 -/
theorem line_parameterization (r k : ℝ) : 
  (∀ t x y : ℝ, x = r + 5*t ∧ y = -3 + k*t → y = 2*x + 5) →
  r = -4 ∧ k = 10 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l419_41993


namespace NUMINAMATH_CALUDE_macys_weekly_goal_l419_41911

/-- Macy's weekly running goal in miles -/
def weekly_goal : ℕ := 24

/-- Miles Macy runs per day -/
def miles_per_day : ℕ := 3

/-- Number of days Macy has run -/
def days_run : ℕ := 6

/-- Miles left to run after 6 days -/
def miles_left : ℕ := 6

/-- Theorem stating Macy's weekly running goal -/
theorem macys_weekly_goal : 
  weekly_goal = miles_per_day * days_run + miles_left := by sorry

end NUMINAMATH_CALUDE_macys_weekly_goal_l419_41911


namespace NUMINAMATH_CALUDE_exists_quadrilateral_equal_tangents_l419_41996

-- Define a quadrilateral as a structure with four angles
structure Quadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real

-- Define the property of having equal tangents for all angles
def hasEqualTangents (q : Quadrilateral) : Prop :=
  Real.tan q.α = Real.tan q.β ∧
  Real.tan q.β = Real.tan q.γ ∧
  Real.tan q.γ = Real.tan q.δ

-- Define the property of angles summing to 360°
def anglesSum360 (q : Quadrilateral) : Prop :=
  q.α + q.β + q.γ + q.δ = 360

-- Theorem stating the existence of a quadrilateral with equal tangents
theorem exists_quadrilateral_equal_tangents :
  ∃ q : Quadrilateral, anglesSum360 q ∧ hasEqualTangents q :=
sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_equal_tangents_l419_41996


namespace NUMINAMATH_CALUDE_equality_check_l419_41900

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  ((-2)^3 = -2^3) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-(-2) ≠ -|-2|) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l419_41900


namespace NUMINAMATH_CALUDE_unique_pair_satisfying_conditions_l419_41945

theorem unique_pair_satisfying_conditions :
  ∀ a b : ℕ+,
  a + b + (Nat.gcd a b)^2 = Nat.lcm a b ∧
  Nat.lcm a b = 2 * Nat.lcm (a - 1) b →
  a = 6 ∧ b = 15 := by
sorry

end NUMINAMATH_CALUDE_unique_pair_satisfying_conditions_l419_41945


namespace NUMINAMATH_CALUDE_sum_of_solutions_l419_41957

theorem sum_of_solutions (x : ℝ) : 
  (Real.sqrt x + Real.sqrt (9 / x) + Real.sqrt (x + 9 / x) = 7) → 
  (∃ y : ℝ, x^2 - (49/4) * x + 9 = 0 ∧ y^2 - (49/4) * y + 9 = 0 ∧ x + y = 49/4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l419_41957


namespace NUMINAMATH_CALUDE_doctor_visit_cost_is_400_l419_41969

/-- Represents Tom's medication and doctor visit expenses -/
structure MedicationExpenses where
  pills_per_day : ℕ
  doctor_visits_per_year : ℕ
  pill_cost : ℚ
  insurance_coverage : ℚ
  total_annual_cost : ℚ

/-- Calculates the cost of a single doctor visit -/
def doctor_visit_cost (e : MedicationExpenses) : ℚ :=
  let annual_pills := e.pills_per_day * 365
  let annual_pill_cost := annual_pills * e.pill_cost
  let patient_pill_cost := annual_pill_cost * (1 - e.insurance_coverage)
  let annual_doctor_cost := e.total_annual_cost - patient_pill_cost
  annual_doctor_cost / e.doctor_visits_per_year

/-- Theorem stating that Tom's doctor visit costs $400 -/
theorem doctor_visit_cost_is_400 (e : MedicationExpenses) 
  (h1 : e.pills_per_day = 2)
  (h2 : e.doctor_visits_per_year = 2)
  (h3 : e.pill_cost = 5)
  (h4 : e.insurance_coverage = 4/5)
  (h5 : e.total_annual_cost = 1530) :
  doctor_visit_cost e = 400 := by
  sorry

end NUMINAMATH_CALUDE_doctor_visit_cost_is_400_l419_41969


namespace NUMINAMATH_CALUDE_percentage_increase_theorem_l419_41980

theorem percentage_increase_theorem (initial_value : ℝ) 
  (first_increase_percent : ℝ) (second_increase_percent : ℝ) :
  let first_increase := initial_value * (1 + first_increase_percent / 100)
  let final_value := first_increase * (1 + second_increase_percent / 100)
  initial_value = 5000 ∧ first_increase_percent = 65 ∧ second_increase_percent = 45 →
  final_value = 11962.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_theorem_l419_41980


namespace NUMINAMATH_CALUDE_max_cubes_fit_l419_41906

def small_cube_edge : ℝ := 10.7
def large_cube_edge : ℝ := 100

theorem max_cubes_fit (small_cube_edge : ℝ) (large_cube_edge : ℝ) :
  small_cube_edge = 10.7 →
  large_cube_edge = 100 →
  ⌊(large_cube_edge ^ 3) / (small_cube_edge ^ 3)⌋ = 816 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_fit_l419_41906


namespace NUMINAMATH_CALUDE_lcm_one_to_five_l419_41971

theorem lcm_one_to_five : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 5))) = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_one_to_five_l419_41971


namespace NUMINAMATH_CALUDE_triangle_area_product_l419_41978

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (1/2) * (4/a) * (6/b) = 3 → 
  a * b = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_product_l419_41978


namespace NUMINAMATH_CALUDE_x_y_inequality_l419_41968

theorem x_y_inequality (x y : ℝ) 
  (h1 : x < 1) 
  (h2 : 1 < y) 
  (h3 : 2 * Real.log x + Real.log (1 - x) ≥ 3 * Real.log y + Real.log (y - 1)) :
  x^3 + y^3 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_y_inequality_l419_41968


namespace NUMINAMATH_CALUDE_ellipse_set_is_ellipse_l419_41964

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

-- Define points A and B
variable (A B : E)

-- Define the set of points P satisfying the condition
def ellipse_set (A B : E) : Set E :=
  {P : E | dist P A + dist P B = 2 * dist A B}

-- Theorem statement
theorem ellipse_set_is_ellipse (A B : E) (h : A ≠ B) :
  ∃ (C : E) (a b : ℝ), a > b ∧ b > 0 ∧
    ellipse_set A B = {P : E | (dist P C)^2 / a^2 + (dist P (C + (B - A)))^2 / b^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_ellipse_set_is_ellipse_l419_41964


namespace NUMINAMATH_CALUDE_area_of_second_square_l419_41902

/-- A right isosceles triangle with two inscribed squares -/
structure RightIsoscelesTriangleWithSquares where
  -- The side length of the triangle
  b : ℝ
  -- The side length of the first inscribed square (ADEF)
  a₁ : ℝ
  -- The side length of the second inscribed square (GHIJ)
  a : ℝ
  -- The first square is inscribed in the triangle
  h_a₁_inscribed : a₁ = b / 2
  -- The second square is inscribed in the triangle
  h_a_inscribed : a = (2 * b ^ 2) / (3 * b * Real.sqrt 2)

/-- The theorem statement -/
theorem area_of_second_square (t : RightIsoscelesTriangleWithSquares) 
    (h_area_first : t.a₁ ^ 2 = 2250) : 
    t.a ^ 2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_area_of_second_square_l419_41902


namespace NUMINAMATH_CALUDE_smallest_marble_count_l419_41901

theorem smallest_marble_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 7) % 10 = 0 ∧ 
  (n - 10) % 7 = 0 ∧
  n = 143 ∧
  ∀ (m : ℕ), (m ≥ 100 ∧ m ≤ 999) → (m + 7) % 10 = 0 → (m - 10) % 7 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l419_41901


namespace NUMINAMATH_CALUDE_existence_of_sequences_l419_41922

theorem existence_of_sequences : ∃ (u v : ℕ → ℕ) (k : ℕ+),
  (∀ n m : ℕ, n < m → u n < u m) ∧
  (∀ n m : ℕ, n < m → v n < v m) ∧
  (∀ n : ℕ, k * (u n * (u n + 1)) = v n ^ 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sequences_l419_41922


namespace NUMINAMATH_CALUDE_disjunction_true_l419_41997

theorem disjunction_true : 
  (∀ x : ℝ, x < 0 → 2^x > x) ∨ (∃ x : ℝ, x^2 + x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_disjunction_true_l419_41997


namespace NUMINAMATH_CALUDE_diamond_5_20_l419_41955

-- Define the diamond operation
noncomputable def diamond (x y : ℝ) : ℝ := sorry

-- Axioms for the diamond operation
axiom diamond_positive (x y : ℝ) : x > 0 → y > 0 → diamond x y > 0
axiom diamond_eq1 (x y : ℝ) : x > 0 → y > 0 → diamond (x * y) y = x * diamond y y
axiom diamond_eq2 (x : ℝ) : x > 0 → diamond (diamond x 2) x = diamond x 2
axiom diamond_2_2 : diamond 2 2 = 4

-- Theorem to prove
theorem diamond_5_20 : diamond 5 20 = 20 := by sorry

end NUMINAMATH_CALUDE_diamond_5_20_l419_41955


namespace NUMINAMATH_CALUDE_point_meeting_time_l419_41942

theorem point_meeting_time (b_initial c_initial b_speed c_speed : ℚ) (h1 : b_initial = -8)
  (h2 : c_initial = 16) (h3 : b_speed = 6) (h4 : c_speed = 2) :
  ∃ t : ℚ, t = 2 ∧ c_initial - b_initial - (b_speed + c_speed) * t = 8 :=
by sorry

end NUMINAMATH_CALUDE_point_meeting_time_l419_41942


namespace NUMINAMATH_CALUDE_exam_candidates_count_l419_41941

theorem exam_candidates_count :
  ∀ (x : ℕ),
  (x : ℝ) * 0.07 = (x : ℝ) * 0.06 + 82 →
  x = 8200 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_count_l419_41941


namespace NUMINAMATH_CALUDE_shaded_percentage_is_correct_l419_41995

/-- Represents a 7x7 grid with a checkered shading pattern and unshaded fourth row and column -/
structure CheckeredGrid :=
  (size : Nat)
  (is_seven_by_seven : size = 7)
  (checkered_pattern : Bool)
  (unshaded_fourth_row : Bool)
  (unshaded_fourth_column : Bool)

/-- Calculates the number of shaded squares in the CheckeredGrid -/
def shaded_squares (grid : CheckeredGrid) : Nat :=
  grid.size * grid.size - (grid.size + grid.size - 1)

/-- Calculates the total number of squares in the CheckeredGrid -/
def total_squares (grid : CheckeredGrid) : Nat :=
  grid.size * grid.size

/-- Theorem stating that the percentage of shaded squares is 36/49 -/
theorem shaded_percentage_is_correct (grid : CheckeredGrid) :
  (shaded_squares grid : ℚ) / (total_squares grid : ℚ) = 36 / 49 := by
  sorry

#eval (36 : ℚ) / 49  -- To show the approximate decimal value

end NUMINAMATH_CALUDE_shaded_percentage_is_correct_l419_41995


namespace NUMINAMATH_CALUDE_smallest_circle_passing_through_intersection_l419_41999

-- Define the line
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the smallest circle
def smallest_circle (x y : ℝ) : Prop := 5*x^2 + 5*y^2 + 6*x - 18*y - 1 = 0

-- Theorem statement
theorem smallest_circle_passing_through_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    line x1 y1 ∧ line x2 y2 ∧
    original_circle x1 y1 ∧ original_circle x2 y2 ∧
    (∀ (x y : ℝ), smallest_circle x y ↔ 
      ((x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2 ∧
       ∀ (c : ℝ → ℝ → Prop), (c x1 y1 ∧ c x2 y2) → 
         (∃ (xc yc r : ℝ), ∀ (x y : ℝ), c x y ↔ (x - xc)^2 + (y - yc)^2 = r^2) →
         (∃ (xs ys rs : ℝ), ∀ (x y : ℝ), smallest_circle x y ↔ (x - xs)^2 + (y - ys)^2 = rs^2 ∧ rs ≤ r))) :=
by sorry


end NUMINAMATH_CALUDE_smallest_circle_passing_through_intersection_l419_41999


namespace NUMINAMATH_CALUDE_snail_return_whole_hours_l419_41926

/-- Represents the snail's movement on a 2D plane -/
structure SnailMovement where
  speed : ℝ
  turnInterval : ℝ
  turnAngle : ℝ

/-- Represents the snail's position on a 2D plane -/
structure Position where
  x : ℝ
  y : ℝ

/-- Calculates the snail's position after a given time -/
def snailPosition (movement : SnailMovement) (time : ℝ) : Position :=
  sorry

/-- Theorem: The snail returns to its starting point only after a whole number of hours -/
theorem snail_return_whole_hours (movement : SnailMovement) 
    (h1 : movement.speed > 0)
    (h2 : movement.turnInterval = 1/4)
    (h3 : movement.turnAngle = π/2) :
  ∀ t : ℝ, snailPosition movement t = snailPosition movement 0 → ∃ n : ℕ, t = n :=
  sorry

end NUMINAMATH_CALUDE_snail_return_whole_hours_l419_41926


namespace NUMINAMATH_CALUDE_binary_110_equals_6_l419_41948

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Theorem statement
theorem binary_110_equals_6 :
  binary_to_decimal [false, true, true] = 6 := by
  sorry

end NUMINAMATH_CALUDE_binary_110_equals_6_l419_41948


namespace NUMINAMATH_CALUDE_point_coordinates_l419_41958

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ := |p.y|

/-- The distance of a point from the y-axis -/
def distanceFromYAxis (p : Point) : ℝ := |p.x|

/-- Determines if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y > 0

/-- Theorem: Given the conditions, the point M has coordinates (-2, 3) -/
theorem point_coordinates (M : Point)
    (h1 : distanceFromXAxis M = 3)
    (h2 : distanceFromYAxis M = 2)
    (h3 : isInSecondQuadrant M) :
    M = Point.mk (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l419_41958


namespace NUMINAMATH_CALUDE_A_union_complement_B_eq_A_l419_41907

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {1,3,5}
def B : Set Nat := {2,4}

theorem A_union_complement_B_eq_A : A ∪ (U \ B) = A := by
  sorry

end NUMINAMATH_CALUDE_A_union_complement_B_eq_A_l419_41907


namespace NUMINAMATH_CALUDE_chess_draw_probability_l419_41951

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.4)
  (h_not_lose : p_not_lose = 0.9) :
  p_not_lose - p_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l419_41951


namespace NUMINAMATH_CALUDE_age_difference_l419_41919

theorem age_difference (man_age son_age : ℕ) : 
  son_age = 24 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 26 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l419_41919


namespace NUMINAMATH_CALUDE_remainder_theorem_l419_41973

theorem remainder_theorem (x y z a b c d e : ℕ) : 
  0 < a ∧ a < 211 ∧ 
  0 < b ∧ b < 211 ∧ 
  0 < c ∧ c < 211 ∧ 
  0 < d ∧ d < 251 ∧ 
  0 < e ∧ e < 251 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  x % 211 = a ∧ 
  y % 211 = b ∧ 
  z % 211 = c ∧
  x % 251 = c ∧
  y % 251 = d ∧
  z % 251 = e →
  ∃! R, 0 ≤ R ∧ R < 211 * 251 ∧ (2 * x - y + 3 * z + 47) % (211 * 251) = R :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l419_41973


namespace NUMINAMATH_CALUDE_motorbike_distance_theorem_l419_41952

/-- Given two motorbikes traveling the same distance, with speeds of 60 km/h and 64 km/h
    respectively, and the slower bike taking 1 hour more than the faster bike,
    prove that the distance traveled is 960 kilometers. -/
theorem motorbike_distance_theorem (distance : ℝ) (time_slower : ℝ) (time_faster : ℝ) :
  (60 * time_slower = distance) →
  (64 * time_faster = distance) →
  (time_slower = time_faster + 1) →
  distance = 960 := by
  sorry

end NUMINAMATH_CALUDE_motorbike_distance_theorem_l419_41952


namespace NUMINAMATH_CALUDE_nancy_savings_l419_41913

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The number of coins in a dozen -/
def dozen : ℕ := 12

/-- The number of quarters Nancy has -/
def nancy_quarters : ℕ := 3 * dozen

/-- The number of dimes Nancy has -/
def nancy_dimes : ℕ := 2 * dozen

/-- The number of nickels Nancy has -/
def nancy_nickels : ℕ := 5 * dozen

/-- The total monetary value of Nancy's coins -/
def nancy_total : ℚ := 
  (nancy_quarters : ℚ) * quarter_value + 
  (nancy_dimes : ℚ) * dime_value + 
  (nancy_nickels : ℚ) * nickel_value

theorem nancy_savings : nancy_total = 14.40 := by
  sorry

end NUMINAMATH_CALUDE_nancy_savings_l419_41913


namespace NUMINAMATH_CALUDE_highway_problem_l419_41903

-- Define the speeds and distances
def yi_initial_speed : ℝ := 60
def speed_reduction_jia : ℝ := 0.4
def speed_reduction_yi : ℝ := 0.25
def time_jia_to_bing : ℝ := 9
def extra_distance_yi : ℝ := 50

-- Define the theorem
theorem highway_problem :
  ∃ (jia_initial_speed : ℝ) (distance_AD : ℝ),
    jia_initial_speed = 125 ∧ distance_AD = 1880 := by
  sorry


end NUMINAMATH_CALUDE_highway_problem_l419_41903


namespace NUMINAMATH_CALUDE_intersection_equals_N_l419_41944

def U := ℝ

def M : Set ℝ := {x : ℝ | x < 1}

def N : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_equals_N : M ∩ N = N := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_N_l419_41944


namespace NUMINAMATH_CALUDE_two_extreme_points_condition_l419_41908

open Real

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := x * log x - (a / 2) * x^2 - x + 1

-- Define the first derivative of f
noncomputable def f' (a x : ℝ) : ℝ := log x - a * x

-- Theorem statement
theorem two_extreme_points_condition (a : ℝ) :
  (∀ x > 0, ∃ y z, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    f' a x = 0 ∧ f' a y = 0 ∧ f' a z = 0) ↔
  (0 < a ∧ a < 1 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_two_extreme_points_condition_l419_41908


namespace NUMINAMATH_CALUDE_shaded_area_sum_l419_41927

/-- Represents the shaded area between a circle and an inscribed equilateral triangle --/
structure ShadedArea where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the shaded area for a given circle and inscribed equilateral triangle --/
def calculateShadedArea (sideLength : ℝ) : ShadedArea :=
  { a := 18.75,
    b := 21,
    c := 3 }

/-- Theorem stating the sum of a, b, and c for the given problem --/
theorem shaded_area_sum (sideLength : ℝ) :
  sideLength = 15 →
  let area := calculateShadedArea sideLength
  area.a + area.b + area.c = 42.75 := by
  sorry

#check shaded_area_sum

end NUMINAMATH_CALUDE_shaded_area_sum_l419_41927


namespace NUMINAMATH_CALUDE_N_composite_and_three_factors_l419_41976

def N (n : ℕ) : ℤ := n^4 - 90*n^2 - 91*n - 90

theorem N_composite_and_three_factors (n : ℕ) (h : n > 10) :
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N n = a * b) ∧
  (∃ (x y z : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ N n = x * y * z) :=
sorry

end NUMINAMATH_CALUDE_N_composite_and_three_factors_l419_41976


namespace NUMINAMATH_CALUDE_absolute_value_equation_l419_41930

theorem absolute_value_equation (x y : ℝ) :
  |2*x - Real.sqrt y| = 2*x + Real.sqrt y → y = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l419_41930


namespace NUMINAMATH_CALUDE_cool_parents_problem_l419_41933

theorem cool_parents_problem (U : Finset ℕ) (A B : Finset ℕ) 
  (h1 : Finset.card U = 40)
  (h2 : Finset.card A = 18)
  (h3 : Finset.card B = 20)
  (h4 : Finset.card (A ∩ B) = 11)
  (h5 : A ⊆ U)
  (h6 : B ⊆ U) :
  Finset.card (U \ (A ∪ B)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_cool_parents_problem_l419_41933


namespace NUMINAMATH_CALUDE_equation_solution_l419_41988

theorem equation_solution (x y : ℝ) :
  y^2 - 2*y = x^2 + 2*x ↔ y = x + 2 ∨ y = -x := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l419_41988


namespace NUMINAMATH_CALUDE_toms_weekly_fluid_intake_l419_41943

/-- Calculates the total fluid intake in ounces for a week given daily soda and water consumption --/
def weekly_fluid_intake (soda_cans : ℕ) (oz_per_can : ℕ) (water_oz : ℕ) : ℕ :=
  7 * (soda_cans * oz_per_can + water_oz)

/-- Theorem stating Tom's weekly fluid intake --/
theorem toms_weekly_fluid_intake :
  weekly_fluid_intake 5 12 64 = 868 := by
  sorry

end NUMINAMATH_CALUDE_toms_weekly_fluid_intake_l419_41943


namespace NUMINAMATH_CALUDE_propositions_are_false_l419_41928

-- Define a type for planes
def Plane : Type := Unit

-- Define a relation for "is in"
def is_in (α β : Plane) : Prop := sorry

-- Define a relation for "is parallel to"
def is_parallel (α β : Plane) : Prop := sorry

-- Define a type for points
def Point : Type := Unit

-- Define a property for three points being non-collinear
def non_collinear (p q r : Point) : Prop := sorry

-- Define a property for a point being on a plane
def on_plane (p : Point) (α : Plane) : Prop := sorry

-- Define a property for a point being equidistant from a plane
def equidistant_from_plane (p : Point) (β : Plane) : Prop := sorry

theorem propositions_are_false :
  (∃ α β γ : Plane, is_in α β ∧ is_in β γ ∧ ¬is_parallel α γ) ∧
  (∃ α β : Plane, ∃ p q r : Point,
    non_collinear p q r ∧
    on_plane p α ∧ on_plane q α ∧ on_plane r α ∧
    equidistant_from_plane p β ∧ equidistant_from_plane q β ∧ equidistant_from_plane r β ∧
    ¬is_parallel α β) :=
by sorry

end NUMINAMATH_CALUDE_propositions_are_false_l419_41928


namespace NUMINAMATH_CALUDE_inequality_solution_l419_41910

theorem inequality_solution (x : ℝ) : 
  (x / 4 ≤ 3 + x ∧ 3 + x ≤ -3 * (1 + x)) ↔ -4 ≤ x ∧ x ≤ -3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l419_41910


namespace NUMINAMATH_CALUDE_x_convergence_interval_l419_41998

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 8) / (x n + 7)

theorem x_convergence_interval :
  ∃ m : ℕ, 81 ≤ m ∧ m ≤ 242 ∧ x m ≤ 4 + 1 / (2^18) ∧
  ∀ k : ℕ, 0 < k ∧ k < 81 → x k > 4 + 1 / (2^18) := by
  sorry

end NUMINAMATH_CALUDE_x_convergence_interval_l419_41998


namespace NUMINAMATH_CALUDE_angle_equality_l419_41918

-- Define the types for points and angles
variable (Point Angle : Type)

-- Define the triangle ABC
variable (A B C : Point)

-- Define the points on the sides of the triangle
variable (P₁ P₂ Q₁ Q₂ R S M : Point)

-- Define the necessary geometric predicates
variable (lies_on : Point → Point → Point → Prop)
variable (is_midpoint : Point → Point → Point → Prop)
variable (angle : Point → Point → Point → Angle)
variable (length_eq : Point → Point → Point → Point → Prop)
variable (intersects : Point → Point → Point → Point → Point → Prop)
variable (on_circumcircle : Point → Point → Point → Point → Prop)
variable (inside_triangle : Point → Point → Point → Point → Prop)

-- State the theorem
theorem angle_equality 
  (h1 : lies_on P₁ A B) (h2 : lies_on P₂ A B) (h3 : lies_on P₂ B P₁)
  (h4 : length_eq A P₁ B P₂)
  (h5 : lies_on Q₁ B C) (h6 : lies_on Q₂ B C) (h7 : lies_on Q₂ B Q₁)
  (h8 : length_eq B Q₁ C Q₂)
  (h9 : intersects P₁ Q₂ P₂ Q₁ R)
  (h10 : on_circumcircle S P₁ P₂ R) (h11 : on_circumcircle S Q₁ Q₂ R)
  (h12 : inside_triangle S P₁ Q₁ R)
  (h13 : is_midpoint M A C) :
  angle P₁ R S = angle Q₁ R M :=
by sorry

end NUMINAMATH_CALUDE_angle_equality_l419_41918


namespace NUMINAMATH_CALUDE_negation_equivalence_l419_41949

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (a % 2 = 0 ∧ c % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l419_41949


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l419_41931

noncomputable def f (x : ℝ) : ℝ := -(Real.sqrt 3 / 3) * x^3 + 2

theorem tangent_slope_angle (x : ℝ) : 
  let f' := deriv f
  let slope := f' 1
  let angle := Real.arctan (-slope)
  x = 1 → angle = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l419_41931


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l419_41937

theorem coloring_book_shelves (initial_stock : ℕ) (acquired : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 2000 →
  acquired = 5000 →
  books_per_shelf = 2 →
  (initial_stock + acquired) / books_per_shelf = 3500 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l419_41937


namespace NUMINAMATH_CALUDE_max_product_of_sum_constrained_naturals_l419_41994

theorem max_product_of_sum_constrained_naturals
  (n k : ℕ) (h : k > 0) :
  let t : ℕ := n / k
  let r : ℕ := n % k
  ∃ (l : List ℕ),
    (l.length = k) ∧
    (l.sum = n) ∧
    (∀ (m : List ℕ), m.length = k → m.sum = n → l.prod ≥ m.prod) ∧
    (l.prod = (t + 1)^r * t^(k - r)) := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_sum_constrained_naturals_l419_41994


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l419_41953

theorem unique_two_digit_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (∃ q r : ℕ, n = q * (10 * (n % 10) + n / 10) + r ∧ q = 4 ∧ r = 3) ∧
  (∃ q r : ℕ, n = q * (n / 10 + n % 10) + r ∧ q = 8 ∧ r = 7) ∧
  n = 71 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l419_41953


namespace NUMINAMATH_CALUDE_flowerbed_count_l419_41912

theorem flowerbed_count (total_seeds : ℕ) (seeds_per_flowerbed : ℕ) (h1 : total_seeds = 45) (h2 : seeds_per_flowerbed = 5) :
  total_seeds / seeds_per_flowerbed = 9 := by
  sorry

end NUMINAMATH_CALUDE_flowerbed_count_l419_41912


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l419_41934

theorem half_angle_quadrant (α : Real) : 
  (π / 2 < α ∧ α < π) → 
  ((0 < (α / 2) ∧ (α / 2) < π / 2) ∨ (π < (α / 2) ∧ (α / 2) < 3 * π / 2)) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l419_41934


namespace NUMINAMATH_CALUDE_rahul_savings_l419_41962

/-- Proves that given the conditions on Rahul's savings, the total amount saved is 180,000 Rs. -/
theorem rahul_savings (nsc ppf : ℕ) : 
  (1 / 3 : ℚ) * nsc = (1 / 2 : ℚ) * ppf →
  ppf = 72000 →
  nsc + ppf = 180000 := by
sorry

end NUMINAMATH_CALUDE_rahul_savings_l419_41962


namespace NUMINAMATH_CALUDE_decreasing_function_condition_l419_41923

-- Define the function f(x)
def f (k x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

-- Define the derivative of f(x)
def f_derivative (k x : ℝ) : ℝ := 3 * k * x^2 + 6 * (k - 1) * x

-- Theorem statement
theorem decreasing_function_condition (k : ℝ) :
  (∀ x ∈ Set.Ioo 0 4, f_derivative k x ≤ 0) ↔ k ≤ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_condition_l419_41923


namespace NUMINAMATH_CALUDE_problem_solution_l419_41984

theorem problem_solution (x y : ℚ) 
  (h1 : x + y = 2/3)
  (h2 : x/y = 2/3) : 
  x - y = -2/15 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l419_41984


namespace NUMINAMATH_CALUDE_no_real_roots_l419_41990

theorem no_real_roots : ¬∃ x : ℝ, x + Real.sqrt (2 * x - 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l419_41990


namespace NUMINAMATH_CALUDE_some_number_calculation_l419_41909

theorem some_number_calculation : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 := by
  sorry

end NUMINAMATH_CALUDE_some_number_calculation_l419_41909


namespace NUMINAMATH_CALUDE_mean_temperature_l419_41989

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 0]

theorem mean_temperature :
  (List.sum temperatures : ℚ) / temperatures.length = -10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l419_41989


namespace NUMINAMATH_CALUDE_power_of_three_divides_a_l419_41966

def a : ℕ → ℤ
  | 0 => 3
  | n + 1 => (3 * a n ^ 2 + 1) / 2 - a n

theorem power_of_three_divides_a (k : ℕ) : 
  (3 ^ (k + 1) : ℤ) ∣ a (3 ^ k) := by sorry

end NUMINAMATH_CALUDE_power_of_three_divides_a_l419_41966


namespace NUMINAMATH_CALUDE_noodles_given_to_william_l419_41967

/-- The number of noodles Daniel initially had -/
def initial_noodles : ℕ := 66

/-- The number of noodles Daniel has now -/
def remaining_noodles : ℕ := 54

/-- The number of noodles Daniel gave to William -/
def noodles_given : ℕ := initial_noodles - remaining_noodles

theorem noodles_given_to_william : noodles_given = 12 := by
  sorry

end NUMINAMATH_CALUDE_noodles_given_to_william_l419_41967


namespace NUMINAMATH_CALUDE_subset_intersection_problem_l419_41938

theorem subset_intersection_problem (a : ℝ) :
  let A := { x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5 }
  let B := { x : ℝ | 3 ≤ x ∧ x ≤ 22 }
  (A ⊆ A ∩ B) ↔ (6 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_subset_intersection_problem_l419_41938


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_product_l419_41992

theorem geometric_sequence_sum_product (a b c : ℝ) : 
  (∃ q : ℝ, b = a * q ∧ c = b * q) →  -- geometric sequence condition
  a + b + c = 14 →                    -- sum condition
  a * b * c = 64 →                    -- product condition
  ((a = 8 ∧ b = 4 ∧ c = 2) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_product_l419_41992


namespace NUMINAMATH_CALUDE_central_square_area_l419_41977

theorem central_square_area (side_length : ℝ) (cut_distance : ℝ) :
  side_length = 15 →
  cut_distance = 4 →
  let central_square_side := cut_distance * Real.sqrt 2
  central_square_side ^ 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_central_square_area_l419_41977


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l419_41974

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : ∃ d : ℝ, a 2 - a 3 / 2 = a 3 / 2 - a 1) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l419_41974


namespace NUMINAMATH_CALUDE_grid_sum_equality_l419_41987

theorem grid_sum_equality (row1 row2 : List ℕ) (x : ℕ) :
  row1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1050] →
  row2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, x] →
  row1.sum = row2.sum →
  x = 950 := by
  sorry

end NUMINAMATH_CALUDE_grid_sum_equality_l419_41987


namespace NUMINAMATH_CALUDE_remainder_theorem_l419_41929

theorem remainder_theorem (r : ℝ) : (r^13 - r^5 + 1) % (r - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l419_41929


namespace NUMINAMATH_CALUDE_and_false_necessary_not_sufficient_for_or_false_l419_41991

theorem and_false_necessary_not_sufficient_for_or_false (p q : Prop) :
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ 
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_and_false_necessary_not_sufficient_for_or_false_l419_41991


namespace NUMINAMATH_CALUDE_two_approve_probability_l419_41961

/-- The probability of a voter approving the mayor's work -/
def approval_rate : ℝ := 0.6

/-- The number of voters randomly selected -/
def sample_size : ℕ := 4

/-- The number of approving voters we're interested in -/
def target_approvals : ℕ := 2

/-- The probability of exactly two out of four randomly selected voters approving the mayor's work -/
def prob_two_approve : ℝ := Nat.choose sample_size target_approvals * approval_rate ^ target_approvals * (1 - approval_rate) ^ (sample_size - target_approvals)

theorem two_approve_probability :
  prob_two_approve = 0.864 := by sorry

end NUMINAMATH_CALUDE_two_approve_probability_l419_41961


namespace NUMINAMATH_CALUDE_september_reading_goal_l419_41985

def total_pages_read (total_days : ℕ) (non_reading_days : ℕ) (special_day_pages : ℕ) (regular_daily_pages : ℕ) : ℕ :=
  let reading_days := total_days - non_reading_days
  let regular_reading_days := reading_days - 1
  regular_reading_days * regular_daily_pages + special_day_pages

theorem september_reading_goal :
  total_pages_read 30 4 100 20 = 600 := by
  sorry

end NUMINAMATH_CALUDE_september_reading_goal_l419_41985


namespace NUMINAMATH_CALUDE_lg_sum_equals_zero_l419_41915

-- Define lg as the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_zero : lg 5 + lg 0.2 = 0 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_zero_l419_41915


namespace NUMINAMATH_CALUDE_westward_notation_l419_41921

/-- Represents the direction on the runway -/
inductive Direction
  | East
  | West

/-- Represents a distance walked on the runway -/
structure Walk where
  distance : ℝ
  direction : Direction

/-- Converts a walk to its signed representation -/
def Walk.toSigned (w : Walk) : ℝ :=
  match w.direction with
  | Direction.East => w.distance
  | Direction.West => -w.distance

theorem westward_notation (d : ℝ) (h : d > 0) :
  let eastward := Walk.toSigned { distance := 8, direction := Direction.East }
  let westward := Walk.toSigned { distance := d, direction := Direction.West }
  eastward = 8 → westward = -d :=
by sorry

end NUMINAMATH_CALUDE_westward_notation_l419_41921


namespace NUMINAMATH_CALUDE_rational_root_l419_41963

theorem rational_root (x : ℝ) (hx : x ≠ 0) 
  (h1 : ∃ r : ℚ, x^5 = r) 
  (h2 : ∃ p : ℚ, 20*x + 19/x = p) : 
  ∃ q : ℚ, x = q := by
sorry

end NUMINAMATH_CALUDE_rational_root_l419_41963


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l419_41956

def arithmetic_sequence (a₁ a₂ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * (a₂ - a₁)

theorem thirtieth_term_of_sequence (a₁ a₂ : ℤ) (h₁ : a₁ = 3) (h₂ : a₂ = 7) :
  arithmetic_sequence a₁ a₂ 30 = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l419_41956


namespace NUMINAMATH_CALUDE_exponent_problem_l419_41904

theorem exponent_problem (x m n : ℝ) (hm : x^m = 5) (hn : x^n = 1/4) :
  x^(2*m - n) = 100 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l419_41904


namespace NUMINAMATH_CALUDE_shrimp_price_theorem_l419_41959

/-- The discounted price of a quarter-pound package of shrimp -/
def discounted_price : ℝ := 2.25

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.4

/-- The standard price per pound of shrimp -/
def standard_price : ℝ := 15

/-- Theorem stating that the standard price per pound of shrimp is $15 -/
theorem shrimp_price_theorem :
  standard_price = 15 ∧
  discounted_price = (1 - discount_rate) * (standard_price / 4) :=
by sorry

end NUMINAMATH_CALUDE_shrimp_price_theorem_l419_41959
