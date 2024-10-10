import Mathlib

namespace valid_numeral_count_l1280_128086

def is_single_digit_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def count_valid_numerals : ℕ :=
  let three_digit_count := 4 * 10 * 4
  let four_digit_count := 4 * 10 * 10 * 4
  three_digit_count + four_digit_count

theorem valid_numeral_count :
  count_valid_numerals = 1760 :=
sorry

end valid_numeral_count_l1280_128086


namespace monthly_savings_proof_l1280_128080

-- Define income tax rate
def income_tax_rate : ℚ := 13/100

-- Define salaries and pensions (in rubles)
def ivan_salary : ℕ := 55000
def vasilisa_salary : ℕ := 45000
def mother_salary : ℕ := 18000
def mother_pension : ℕ := 10000
def father_salary : ℕ := 20000
def son_state_scholarship : ℕ := 3000
def son_nonstate_scholarship : ℕ := 15000

-- Define monthly expenses (in rubles)
def monthly_expenses : ℕ := 74000

-- Function to calculate net income after tax
def net_income (gross_income : ℕ) : ℚ :=
  (gross_income : ℚ) * (1 - income_tax_rate)

-- Total monthly net income before 01.05.2018
def total_net_income_before_may : ℚ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  net_income mother_salary + net_income father_salary + son_state_scholarship

-- Total monthly net income from 01.05.2018 to 31.08.2018
def total_net_income_may_to_aug : ℚ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + son_state_scholarship

-- Total monthly net income from 01.09.2018 for 1 year
def total_net_income_from_sep : ℚ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + 
  son_state_scholarship + net_income son_nonstate_scholarship

-- Theorem to prove monthly savings for different periods
theorem monthly_savings_proof :
  (total_net_income_before_may - monthly_expenses = 49060) ∧
  (total_net_income_may_to_aug - monthly_expenses = 43400) ∧
  (total_net_income_from_sep - monthly_expenses = 56450) := by
  sorry


end monthly_savings_proof_l1280_128080


namespace cubic_factorization_l1280_128038

theorem cubic_factorization (a b c d e : ℝ) :
  (∀ x, 216 * x^3 - 27 = (a * x - b) * (c * x^2 + d * x - e)) →
  a + b + c + d + e = 72 := by
sorry

end cubic_factorization_l1280_128038


namespace average_of_numbers_l1280_128040

theorem average_of_numbers : 
  let numbers : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755, 755]
  (numbers.sum / numbers.length : ℚ) = 700 := by
sorry

end average_of_numbers_l1280_128040


namespace min_value_fraction_l1280_128001

theorem min_value_fraction (x : ℝ) (h : x > 6) :
  x^2 / (x - 6) ≥ 24 ∧ (x^2 / (x - 6) = 24 ↔ x = 12) := by
sorry

end min_value_fraction_l1280_128001


namespace emma_in_middle_l1280_128084

-- Define the friends
inductive Friend
| Allen
| Brian
| Chris
| Diana
| Emma

-- Define the car positions
inductive Position
| First
| Second
| Third
| Fourth
| Fifth

-- Define the seating arrangement
def Arrangement := Friend → Position

-- Define the conditions
def validArrangement (a : Arrangement) : Prop :=
  a Friend.Allen = Position.Second ∧
  a Friend.Diana = Position.First ∧
  (a Friend.Brian = Position.Fourth ∧ a Friend.Chris = Position.Fifth) ∨
  (a Friend.Brian = Position.Third ∧ a Friend.Chris = Position.Fourth) ∧
  (a Friend.Emma = Position.Third ∨ a Friend.Emma = Position.Fifth)

-- Theorem to prove
theorem emma_in_middle (a : Arrangement) :
  validArrangement a → a Friend.Emma = Position.Third :=
sorry

end emma_in_middle_l1280_128084


namespace abc_inequality_l1280_128017

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c + Real.sqrt (a^2 + b^2 + c^2))) / 
  ((a^2 + b^2 + c^2) * (a * b + b * c + a * c)) ≤ (3 + Real.sqrt 3) / 9 :=
sorry

end abc_inequality_l1280_128017


namespace ladder_slide_speed_l1280_128039

theorem ladder_slide_speed (x y : ℝ) (dx_dt : ℝ) (h1 : x^2 + y^2 = 5^2) 
  (h2 : x = 1.4) (h3 : dx_dt = 3) : 
  ∃ dy_dt : ℝ, 2*x*dx_dt + 2*y*dy_dt = 0 ∧ |dy_dt| = 0.875 := by
sorry

end ladder_slide_speed_l1280_128039


namespace study_method_is_algorithm_statements_are_not_algorithms_l1280_128034

/-- Represents a series of steps or instructions -/
structure Procedure where
  steps : List String

/-- Represents a statement or fact -/
structure Statement where
  content : String

/-- Definition of an algorithm -/
def is_algorithm (p : Procedure) : Prop :=
  p.steps.length > 0 ∧ ∀ s ∈ p.steps, s ≠ ""

theorem study_method_is_algorithm (study_method : Procedure)
  (h1 : study_method.steps = ["Preview before class", 
                              "Listen carefully and take good notes during class", 
                              "Review first and then do homework after class", 
                              "Do appropriate exercises"]) : 
  is_algorithm study_method := by sorry

theorem statements_are_not_algorithms (s : Statement) : 
  ¬ is_algorithm ⟨[s.content]⟩ := by sorry

#check study_method_is_algorithm
#check statements_are_not_algorithms

end study_method_is_algorithm_statements_are_not_algorithms_l1280_128034


namespace f_has_two_zeros_l1280_128049

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end f_has_two_zeros_l1280_128049


namespace geometric_sequence_sum_l1280_128020

/-- A sequence where each term is 1/3 of the previous term -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = (1 / 3) * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h : geometric_sequence a) (h1 : a 4 + a 5 = 4) : 
  a 2 + a 3 = 36 := by
  sorry

end geometric_sequence_sum_l1280_128020


namespace snack_distribution_solution_l1280_128061

/-- Represents the snack distribution problem for a kindergarten class. -/
structure SnackDistribution where
  pretzels : ℕ
  goldfish : ℕ
  suckers : ℕ
  kids : ℕ
  pretzel_popcorn_ratio : ℚ

/-- Calculates the number of items per snack type in each baggie. -/
def items_per_baggie (sd : SnackDistribution) : ℕ × ℕ × ℕ × ℕ :=
  let pretzels_per_baggie := sd.pretzels / sd.kids
  let goldfish_per_baggie := sd.goldfish / sd.kids
  let suckers_per_baggie := sd.suckers / sd.kids
  let popcorn_per_baggie := (sd.pretzel_popcorn_ratio * pretzels_per_baggie).ceil.toNat
  (pretzels_per_baggie, goldfish_per_baggie, suckers_per_baggie, popcorn_per_baggie)

/-- Calculates the total number of popcorn pieces needed. -/
def total_popcorn (sd : SnackDistribution) : ℕ :=
  let (_, _, _, popcorn_per_baggie) := items_per_baggie sd
  popcorn_per_baggie * sd.kids

/-- Calculates the total number of items in each baggie. -/
def total_items_per_baggie (sd : SnackDistribution) : ℕ :=
  let (p, g, s, c) := items_per_baggie sd
  p + g + s + c

/-- Theorem stating the solution to the snack distribution problem. -/
theorem snack_distribution_solution (sd : SnackDistribution) 
  (h1 : sd.pretzels = 64)
  (h2 : sd.goldfish = 4 * sd.pretzels)
  (h3 : sd.suckers = 32)
  (h4 : sd.kids = 23)
  (h5 : sd.pretzel_popcorn_ratio = 3/2) :
  total_popcorn sd = 69 ∧ total_items_per_baggie sd = 17 := by
  sorry


end snack_distribution_solution_l1280_128061


namespace circle_symmetry_l1280_128004

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

/-- Definition of the symmetry line -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Definition of symmetry between points with respect to the line -/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = y₁ + 1 ∧ y₂ = x₁ - 1

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

/-- Theorem stating that C₂ is symmetric to C₁ with respect to the given line -/
theorem circle_symmetry :
  ∀ x y : ℝ, C₂ x y ↔ ∃ x₁ y₁ : ℝ, C₁ x₁ y₁ ∧ symmetric_points x₁ y₁ x y :=
sorry

end circle_symmetry_l1280_128004


namespace yellow_marble_probability_l1280_128005

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ
  black : ℕ
  yellow : ℕ
  blue : ℕ

/-- The probability of drawing a yellow marble as the second marble -/
def second_yellow_probability (bagA bagB bagC : Bag) : ℚ :=
  let total_A := bagA.white + bagA.black
  let total_B := bagB.yellow + bagB.blue
  let total_C := bagC.yellow + bagC.blue
  let prob_white_A : ℚ := bagA.white / total_A
  let prob_black_A : ℚ := bagA.black / total_A
  let prob_yellow_B : ℚ := bagB.yellow / total_B
  let prob_yellow_C : ℚ := bagC.yellow / total_C
  prob_white_A * prob_yellow_B + prob_black_A * prob_yellow_C

/-- The main theorem stating the probability of drawing a yellow marble as the second marble -/
theorem yellow_marble_probability :
  let bagA : Bag := ⟨3, 4, 0, 0⟩
  let bagB : Bag := ⟨0, 0, 6, 4⟩
  let bagC : Bag := ⟨0, 0, 2, 5⟩
  second_yellow_probability bagA bagB bagC = 103 / 245 := by
  sorry

end yellow_marble_probability_l1280_128005


namespace negation_of_universal_proposition_l1280_128047

theorem negation_of_universal_proposition :
  (¬∀ x : ℝ, x ≥ 1 → Real.log x > 0) ↔ (∃ x : ℝ, x ≥ 1 ∧ Real.log x ≤ 0) := by
  sorry

end negation_of_universal_proposition_l1280_128047


namespace equation_one_solutions_equation_two_solution_l1280_128054

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  x * (x - 4) = 2 * x - 8 ↔ x = 4 ∨ x = 2 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (2 * x) / (2 * x - 3) - 4 / (2 * x + 3) = 1 ↔ x = 10.5 := by sorry

end equation_one_solutions_equation_two_solution_l1280_128054


namespace total_book_price_l1280_128042

theorem total_book_price (total_books : ℕ) (math_books : ℕ) (math_price : ℕ) (history_price : ℕ) :
  total_books = 90 →
  math_books = 54 →
  math_price = 4 →
  history_price = 5 →
  (math_books * math_price + (total_books - math_books) * history_price) = 396 :=
by sorry

end total_book_price_l1280_128042


namespace max_b_value_l1280_128000

/-- The volume of the box -/
def volume : ℕ := 360

/-- Theorem: Given a box with volume 360 cubic units and dimensions a, b, and c,
    where a, b, and c are integers satisfying 1 < c < b < a,
    the maximum possible value of b is 12. -/
theorem max_b_value (a b c : ℕ) 
  (h_volume : a * b * c = volume)
  (h_order : 1 < c ∧ c < b ∧ b < a) :
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = volume ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ b' = 12 :=
sorry

end max_b_value_l1280_128000


namespace octagon_sectors_area_l1280_128071

/-- The area of the region inside a regular octagon with side length 8 but outside
    the circular sectors with radius 4 centered at each vertex. -/
theorem octagon_sectors_area : 
  let side_length : ℝ := 8
  let sector_radius : ℝ := 4
  let octagon_area : ℝ := 8 * (1/2 * side_length^2 * Real.sqrt ((1 - Real.sqrt 2 / 2) / 2))
  let sectors_area : ℝ := 8 * (π * sector_radius^2 / 8)
  octagon_area - sectors_area = 256 * Real.sqrt ((1 - Real.sqrt 2 / 2) / 2) - 16 * π :=
by sorry

end octagon_sectors_area_l1280_128071


namespace solution_of_system_l1280_128041

theorem solution_of_system (x y : ℝ) : 
  (x^2 - x*y + y^2 = 49*(x - y) ∧ x^2 + x*y + y^2 = 76*(x + y)) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = 40 ∧ y = -24)) :=
by sorry

end solution_of_system_l1280_128041


namespace arithmetic_sequence_properties_l1280_128060

/-- An arithmetic sequence with given first term and 17th term -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 2
  term_17 : a 17 = 66

/-- The general formula for the nth term of the sequence -/
def general_formula (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  4 * n - 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = general_formula seq n) ∧
  ¬ ∃ n, seq.a n = 88 := by
  sorry


end arithmetic_sequence_properties_l1280_128060


namespace triangle_area_l1280_128002

/-- Given a triangle with perimeter 32 and inradius 2.5, its area is 40 -/
theorem triangle_area (p r a : ℝ) (h1 : p = 32) (h2 : r = 2.5) (h3 : a = p * r / 4) : a = 40 := by
  sorry

end triangle_area_l1280_128002


namespace root_implies_difference_of_fourth_powers_l1280_128021

theorem root_implies_difference_of_fourth_powers (a b : ℝ) :
  (∃ x, x^2 - 4*a^2*b^2*x = 4 ∧ x = (a^2 + b^2)^2) →
  (a^4 - b^4 = 2 ∨ a^4 - b^4 = -2) :=
by sorry

end root_implies_difference_of_fourth_powers_l1280_128021


namespace triangle_special_area_l1280_128082

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the angles form an arithmetic sequence and a, c, 4/√3 * b form a geometric sequence,
    then the area of the triangle is √3/2 * a² -/
theorem triangle_special_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  B = (A + C) / 2 →
  ∃ (q : ℝ), q > 0 ∧ c = q * a ∧ 4 / Real.sqrt 3 * b = q^2 * a →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 / 2 * a^2 := by
  sorry

end triangle_special_area_l1280_128082


namespace min_value_expression_l1280_128097

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^2 + b^2 + 2*a*b + 1 / (a + b)^2 ≥ 2 := by
  sorry

end min_value_expression_l1280_128097


namespace xy_range_l1280_128085

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2/x + 3*y + 4/y = 10) : 1 ≤ x*y ∧ x*y ≤ 8/3 := by
  sorry

end xy_range_l1280_128085


namespace no_real_m_for_equal_roots_l1280_128064

theorem no_real_m_for_equal_roots : 
  ¬∃ (m : ℝ), ∃ (x : ℝ), 
    (x * (x + 2) - (m + 3)) / ((x + 2) * (m + 2)) = x / m ∧
    ∀ (y : ℝ), (y * (y + 2) - (m + 3)) / ((y + 2) * (m + 2)) = y / m → y = x :=
by sorry

end no_real_m_for_equal_roots_l1280_128064


namespace simplify_nested_radicals_l1280_128098

theorem simplify_nested_radicals : 
  Real.sqrt (8 + 4 * Real.sqrt 3) + Real.sqrt (8 - 4 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end simplify_nested_radicals_l1280_128098


namespace school_survey_result_l1280_128081

/-- Calculates the number of girls in a school based on stratified sampling -/
def girlsInSchool (totalStudents sampleSize girlsInSample : ℕ) : ℕ :=
  (girlsInSample * totalStudents) / sampleSize

/-- Theorem stating that given the problem conditions, the number of girls in the school is 760 -/
theorem school_survey_result :
  let totalStudents : ℕ := 1600
  let sampleSize : ℕ := 200
  let girlsInSample : ℕ := 95
  girlsInSchool totalStudents sampleSize girlsInSample = 760 := by
  sorry

end school_survey_result_l1280_128081


namespace sum_of_reciprocals_l1280_128068

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) :
  1 / x + 1 / y = 4 := by
  sorry

end sum_of_reciprocals_l1280_128068


namespace discounted_price_theorem_l1280_128016

/-- The actual price of the good before discounts -/
def actual_price : ℝ := 9356.725146198829

/-- The first discount rate -/
def discount1 : ℝ := 0.20

/-- The second discount rate -/
def discount2 : ℝ := 0.10

/-- The third discount rate -/
def discount3 : ℝ := 0.05

/-- The final selling price after all discounts -/
def final_price : ℝ := 6400

/-- Theorem stating that applying the successive discounts to the actual price results in the final price -/
theorem discounted_price_theorem :
  actual_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = final_price := by
  sorry

end discounted_price_theorem_l1280_128016


namespace unique_solution_iff_a_eq_three_fourths_l1280_128003

/-- The equation (x - 3) / (ax - 2) = x has exactly one solution if and only if a = 3/4 -/
theorem unique_solution_iff_a_eq_three_fourths (a : ℝ) : 
  (∃! x : ℝ, (x - 3) / (a * x - 2) = x) ↔ a = 3/4 := by
  sorry

end unique_solution_iff_a_eq_three_fourths_l1280_128003


namespace sum_inequality_l1280_128037

theorem sum_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 1) : 
  (a / (1 - a^2)) + (b / (1 - b^2)) + (c / (1 - c^2)) ≥ (3 * Real.sqrt 3) / 2 := by
  sorry

end sum_inequality_l1280_128037


namespace max_ab_value_l1280_128010

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let f := fun x : ℝ => 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| → |h| < ε → f (1 + h) ≤ f 1) →
  (∀ c : ℝ, a * b ≤ c → c ≤ 9) :=
by sorry

end max_ab_value_l1280_128010


namespace mango_purchase_amount_l1280_128013

-- Define the variables
def grapes_kg : ℕ := 3
def grapes_rate : ℕ := 70
def mango_rate : ℕ := 55
def total_paid : ℕ := 705

-- Define the theorem
theorem mango_purchase_amount :
  ∃ (m : ℕ), 
    grapes_kg * grapes_rate + m * mango_rate = total_paid ∧ 
    m = 9 :=
by sorry

end mango_purchase_amount_l1280_128013


namespace count_four_digit_divisible_by_15_eq_600_l1280_128044

/-- The count of positive four-digit integers divisible by 15 -/
def count_four_digit_divisible_by_15 : ℕ :=
  (Finset.filter (fun n => n % 15 = 0) (Finset.range 9000)).card

/-- Theorem stating that the count of positive four-digit integers divisible by 15 is 600 -/
theorem count_four_digit_divisible_by_15_eq_600 :
  count_four_digit_divisible_by_15 = 600 := by
  sorry

end count_four_digit_divisible_by_15_eq_600_l1280_128044


namespace yellow_balls_count_l1280_128096

theorem yellow_balls_count (red_balls : ℕ) (yellow_balls : ℕ) 
  (h1 : red_balls = 1) 
  (h2 : (red_balls : ℚ) / (red_balls + yellow_balls) = 1 / 4) : 
  yellow_balls = 3 := by
  sorry

end yellow_balls_count_l1280_128096


namespace chessboard_rearrangement_l1280_128035

def knight_distance (a b : ℕ × ℕ) : ℕ :=
  sorry

def king_distance (a b : ℕ × ℕ) : ℕ :=
  sorry

def is_valid_rearrangement (N : ℕ) (f : (ℕ × ℕ) → (ℕ × ℕ)) : Prop :=
  ∀ a b : ℕ × ℕ, a.1 < N ∧ a.2 < N ∧ b.1 < N ∧ b.2 < N →
    knight_distance a b = 1 → king_distance (f a) (f b) = 1

theorem chessboard_rearrangement :
  (∃ f : (ℕ × ℕ) → (ℕ × ℕ), is_valid_rearrangement 3 f) ∧
  (¬ ∃ f : (ℕ × ℕ) → (ℕ × ℕ), is_valid_rearrangement 8 f) :=
sorry

end chessboard_rearrangement_l1280_128035


namespace unique_three_digit_numbers_l1280_128009

theorem unique_three_digit_numbers : ∃! (x y : ℕ), 
  100 ≤ x ∧ x ≤ 999 ∧ 
  100 ≤ y ∧ y ≤ 999 ∧ 
  1000 * x + y = 7 * x * y ∧
  x = 143 ∧ y = 143 := by
sorry

end unique_three_digit_numbers_l1280_128009


namespace complex_expression_simplification_l1280_128026

theorem complex_expression_simplification (x y : ℝ) : 
  (2 * x + 3 * Complex.I * y) * (2 * x - 3 * Complex.I * y) + 2 * x = 4 * x^2 + 2 * x - 9 * y^2 := by
  sorry

end complex_expression_simplification_l1280_128026


namespace bob_water_percentage_approx_36_percent_l1280_128087

/-- Represents a farmer with their crop acreages -/
structure Farmer where
  corn : ℝ
  cotton : ℝ
  beans : ℝ

/-- Calculates the total water usage for a farmer given water rates -/
def waterUsage (f : Farmer) (cornRate : ℝ) (cottonRate : ℝ) : ℝ :=
  f.corn * cornRate + f.cotton * cottonRate + f.beans * (2 * cornRate)

/-- The main theorem to prove -/
theorem bob_water_percentage_approx_36_percent 
  (bob : Farmer) 
  (brenda : Farmer)
  (bernie : Farmer)
  (cornRate : ℝ)
  (cottonRate : ℝ)
  (h_bob : bob = { corn := 3, cotton := 9, beans := 12 })
  (h_brenda : brenda = { corn := 6, cotton := 7, beans := 14 })
  (h_bernie : bernie = { corn := 2, cotton := 12, beans := 0 })
  (h_cornRate : cornRate = 20)
  (h_cottonRate : cottonRate = 80) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |waterUsage bob cornRate cottonRate / 
   (waterUsage bob cornRate cottonRate + 
    waterUsage brenda cornRate cottonRate + 
    waterUsage bernie cornRate cottonRate) - 0.36| < ε := by
  sorry

end bob_water_percentage_approx_36_percent_l1280_128087


namespace david_crunches_l1280_128023

theorem david_crunches (zachary_crunches : ℕ) (david_less : ℕ) 
  (h1 : zachary_crunches = 17)
  (h2 : david_less = 13) : 
  zachary_crunches - david_less = 4 := by
  sorry

end david_crunches_l1280_128023


namespace range_of_a_l1280_128011

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 1 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → a^2 - 3*a - x + 1 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(p a ∧ q a)) ∧ (¬¬(q a)) → a ∈ Set.Ico 1 2 :=
by sorry

end range_of_a_l1280_128011


namespace mabel_handled_90_transactions_l1280_128025

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := (110 * mabel_transactions) / 100
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := 80

-- State the theorem
theorem mabel_handled_90_transactions :
  mabel_transactions = 90 ∧
  anthony_transactions = (110 * mabel_transactions) / 100 ∧
  cal_transactions = (2 * anthony_transactions) / 3 ∧
  jade_transactions = cal_transactions + 14 ∧
  jade_transactions = 80 := by
  sorry


end mabel_handled_90_transactions_l1280_128025


namespace f_half_equals_half_l1280_128053

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 1 - 2 * x^2

-- State the theorem
theorem f_half_equals_half :
  (∀ x : ℝ, f (Real.sin x) = Real.cos (2 * x)) →
  f (1/2) = 1/2 := by
  sorry

end f_half_equals_half_l1280_128053


namespace quadratic_inequality_l1280_128074

theorem quadratic_inequality (x : ℝ) : x^2 - 34*x + 225 ≤ 9 ↔ 17 - Real.sqrt 73 ≤ x ∧ x ≤ 17 + Real.sqrt 73 := by
  sorry

end quadratic_inequality_l1280_128074


namespace incorrect_factorization_l1280_128075

theorem incorrect_factorization (x y : ℝ) : ¬(∀ x y : ℝ, x^2 + y^2 = (x + y)^2) :=
by sorry

end incorrect_factorization_l1280_128075


namespace difference_of_squares_l1280_128093

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end difference_of_squares_l1280_128093


namespace line_vector_at_5_l1280_128014

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_at_5 :
  (∀ t : ℝ, ∃ x y z : ℝ, line_vector t = (x, y, z)) →
  line_vector (-1) = (2, 6, 16) →
  line_vector 1 = (1, 3, 8) →
  line_vector 4 = (-2, -6, -16) →
  line_vector 5 = (-4, -12, -8) := by
  sorry

end line_vector_at_5_l1280_128014


namespace simplify_expression_l1280_128067

theorem simplify_expression : (0.3 * 0.2) / (0.4 * 0.5) - (0.1 * 0.6) = 0.24 := by
  sorry

end simplify_expression_l1280_128067


namespace sector_angle_l1280_128062

theorem sector_angle (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 2) :
  ∃ (α r : ℝ), α * r = arc_length ∧ (1/2) * α * r^2 = area ∧ α = 1 := by
  sorry

end sector_angle_l1280_128062


namespace quadratic_inequality_solution_l1280_128095

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + 7

-- Define the condition for the inequality
def inequality_condition (b : ℝ) : Prop :=
  ∀ x : ℝ, f b x < 0 ↔ (x < -2 ∨ x > 3)

-- Theorem statement
theorem quadratic_inequality_solution :
  ∃ b : ℝ, inequality_condition b ∧ b = 1 :=
sorry

end quadratic_inequality_solution_l1280_128095


namespace cubic_root_sum_l1280_128079

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - r - 1 = 0 → s^3 - s - 1 = 0 → t^3 - t - 1 = 0 → 
  (1 + r) / (1 - r) + (1 + s) / (1 - s) + (1 + t) / (1 - t) = -7 := by
sorry

end cubic_root_sum_l1280_128079


namespace zero_not_in_range_of_g_l1280_128027

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -1 then Int.ceil (1 / (x + 1))
  else if x < -1 then Int.floor (1 / (x + 1))
  else 0  -- This value doesn't matter as x = -1 is not in the domain

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -1 → g x ≠ 0 :=
by sorry

end zero_not_in_range_of_g_l1280_128027


namespace inequality_proof_l1280_128088

theorem inequality_proof (x y : ℝ) (h1 : x ≥ y) (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) ≥ 
  (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1)) ∧
  ((x = y ∨ y = 1) ↔ 
    (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) = 
    (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1))) :=
by sorry

end inequality_proof_l1280_128088


namespace sum_reciprocals_equals_one_plus_reciprocal_product_l1280_128083

theorem sum_reciprocals_equals_one_plus_reciprocal_product 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y + 1) : 
  1 / x + 1 / y = 1 + 1 / (x * y) := by
sorry

end sum_reciprocals_equals_one_plus_reciprocal_product_l1280_128083


namespace three_distinct_zeroes_l1280_128099

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then |2^x - 1| else 3 / (x - 1)

/-- The theorem stating the condition for three distinct zeroes -/
theorem three_distinct_zeroes (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) ↔ 0 < a ∧ a < 1 :=
sorry

end three_distinct_zeroes_l1280_128099


namespace ray_point_distance_product_l1280_128090

/-- Given two points on a ray from the origin, prove that the product of their distances
    equals the sum of the products of their coordinates. -/
theorem ray_point_distance_product (x₁ y₁ z₁ x₂ y₂ z₂ r₁ r₂ : ℝ) 
  (h₁ : r₁ = Real.sqrt (x₁^2 + y₁^2 + z₁^2))
  (h₂ : r₂ = Real.sqrt (x₂^2 + y₂^2 + z₂^2))
  (h_collinear : ∃ (t : ℝ), t > 0 ∧ x₂ = t * x₁ ∧ y₂ = t * y₁ ∧ z₂ = t * z₁) :
  r₁ * r₂ = x₁ * x₂ + y₁ * y₂ + z₁ * z₂ := by
  sorry

end ray_point_distance_product_l1280_128090


namespace chosen_number_proof_l1280_128078

theorem chosen_number_proof (x : ℝ) : (x / 9) - 100 = 10 → x = 990 := by
  sorry

end chosen_number_proof_l1280_128078


namespace degrees_to_radians_l1280_128052

theorem degrees_to_radians (π : Real) (h : π = 180) : 
  (240 : Real) * π / 180 = 4 * π / 3 := by sorry

end degrees_to_radians_l1280_128052


namespace fair_attendance_ratio_l1280_128029

theorem fair_attendance_ratio :
  let this_year : ℕ := 600
  let total_three_years : ℕ := 2800
  let next_year : ℕ := (total_three_years - this_year + 200) / 2
  let last_year : ℕ := next_year - 200
  (next_year : ℚ) / this_year = 2 :=
by sorry

end fair_attendance_ratio_l1280_128029


namespace farmer_goats_problem_l1280_128046

/-- Represents the number of additional goats needed to make half of the animals goats -/
def additional_goats (cows sheep initial_goats : ℕ) : ℕ :=
  let total := cows + sheep + initial_goats
  (total - 2 * initial_goats)

theorem farmer_goats_problem (cows sheep initial_goats : ℕ) 
  (h_cows : cows = 7)
  (h_sheep : sheep = 8)
  (h_initial_goats : initial_goats = 6) :
  additional_goats cows sheep initial_goats = 9 := by
  sorry

#eval additional_goats 7 8 6

end farmer_goats_problem_l1280_128046


namespace one_less_than_negative_one_l1280_128092

theorem one_less_than_negative_one : (-1 : ℤ) - 1 = -2 := by
  sorry

end one_less_than_negative_one_l1280_128092


namespace shortest_path_l1280_128072

/-- Represents an elevator in the building --/
inductive Elevator
| A | B | C | D | E | F | G | H | I | J

/-- Represents a floor in the building --/
inductive Floor
| First | Second

/-- Represents a location on a floor --/
structure Location where
  floor : Floor
  x : ℕ
  y : ℕ

/-- Defines the building layout --/
def building_layout : List (Elevator × Location) := sorry

/-- Defines the entrance location --/
def entrance : Location := sorry

/-- Defines the exit location --/
def exit : Location := sorry

/-- Determines if an elevator leads to a confined room --/
def is_confined (e : Elevator) : Bool := sorry

/-- Calculates the distance between two locations --/
def distance (l1 l2 : Location) : ℕ := sorry

/-- Determines if a path is valid (uses only non-confined elevators) --/
def is_valid_path (path : List Elevator) : Bool := sorry

/-- Calculates the total distance of a path --/
def path_distance (path : List Elevator) : ℕ := sorry

/-- The theorem to be proved --/
theorem shortest_path :
  let path := [Elevator.B, Elevator.J, Elevator.G]
  is_valid_path path ∧
  (∀ other_path, is_valid_path other_path → path_distance path ≤ path_distance other_path) :=
sorry

end shortest_path_l1280_128072


namespace equation_solution_l1280_128008

theorem equation_solution : ∃! x : ℝ, 4 * x - 3 = 5 * x + 2 := by
  sorry

end equation_solution_l1280_128008


namespace max_cookies_ella_l1280_128076

/-- Represents the recipe for cookies -/
structure Recipe where
  chocolate : Rat
  sugar : Rat
  eggs : Nat
  flour : Rat
  cookies : Nat

/-- Represents available ingredients -/
structure Ingredients where
  chocolate : Rat
  sugar : Rat
  eggs : Nat
  flour : Rat

/-- Calculates the maximum number of cookies that can be made -/
def maxCookies (recipe : Recipe) (ingredients : Ingredients) : Nat :=
  min
    (Nat.floor ((ingredients.chocolate / recipe.chocolate) * recipe.cookies))
    (min
      (Nat.floor ((ingredients.sugar / recipe.sugar) * recipe.cookies))
      (min
        ((ingredients.eggs / recipe.eggs) * recipe.cookies)
        (Nat.floor ((ingredients.flour / recipe.flour) * recipe.cookies))))

theorem max_cookies_ella :
  let recipe : Recipe := {
    chocolate := 1,
    sugar := 1/2,
    eggs := 1,
    flour := 1,
    cookies := 4
  }
  let ingredients : Ingredients := {
    chocolate := 4,
    sugar := 3,
    eggs := 6,
    flour := 10
  }
  maxCookies recipe ingredients = 16 := by
  sorry

#eval maxCookies
  { chocolate := 1, sugar := 1/2, eggs := 1, flour := 1, cookies := 4 }
  { chocolate := 4, sugar := 3, eggs := 6, flour := 10 }

end max_cookies_ella_l1280_128076


namespace complex_number_location_l1280_128048

theorem complex_number_location :
  ∀ z : ℂ, z / (1 + Complex.I) = Complex.I →
  (z.re < 0 ∧ z.im > 0) :=
by sorry

end complex_number_location_l1280_128048


namespace product_sum_and_reciprocals_geq_nine_l1280_128032

theorem product_sum_and_reciprocals_geq_nine (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end product_sum_and_reciprocals_geq_nine_l1280_128032


namespace point_N_coordinates_l1280_128028

/-- Given point M(5, -6) and vector a = (1, -2), if vector MN = -3 * vector a,
    then the coordinates of point N are (2, 0). -/
theorem point_N_coordinates :
  let M : ℝ × ℝ := (5, -6)
  let a : ℝ × ℝ := (1, -2)
  let N : ℝ × ℝ := (x, y)
  (x - M.1, y - M.2) = (-3 * a.1, -3 * a.2) →
  N = (2, 0) := by
sorry

end point_N_coordinates_l1280_128028


namespace bells_toll_together_l1280_128050

theorem bells_toll_together (a b c d : ℕ) 
  (ha : a = 5) (hb : b = 8) (hc : c = 11) (hd : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 := by
  sorry

end bells_toll_together_l1280_128050


namespace knickknack_weight_is_six_l1280_128043

-- Define the given conditions
def bookcase_max_weight : ℝ := 80
def hardcover_count : ℕ := 70
def hardcover_weight : ℝ := 0.5
def textbook_count : ℕ := 30
def textbook_weight : ℝ := 2
def knickknack_count : ℕ := 3
def weight_over_limit : ℝ := 33

-- Define the total weight of the collection
def total_weight : ℝ := bookcase_max_weight + weight_over_limit

-- Define the weight of hardcover books and textbooks
def books_weight : ℝ := hardcover_count * hardcover_weight + textbook_count * textbook_weight

-- Define the total weight of knick-knacks
def knickknacks_total_weight : ℝ := total_weight - books_weight

-- Theorem to prove
theorem knickknack_weight_is_six :
  knickknacks_total_weight / knickknack_count = 6 := by
  sorry

end knickknack_weight_is_six_l1280_128043


namespace alpha_value_l1280_128051

theorem alpha_value (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1))
  (h4 : Real.cos (Complex.arg α) = 1/2) :
  α = (-1 + Real.sqrt 33) / 4 + Complex.I * Real.sqrt (3 * ((-1 + Real.sqrt 33) / 4)^2) ∨
  α = (-1 - Real.sqrt 33) / 4 + Complex.I * Real.sqrt (3 * ((-1 - Real.sqrt 33) / 4)^2) :=
by sorry

end alpha_value_l1280_128051


namespace part_one_part_two_l1280_128030

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| - |x + 1|

-- Theorem for part I
theorem part_one : 
  ∀ x : ℝ, f (-1/2) x ≤ -1 ↔ x ≥ 1/4 := by sorry

-- Theorem for part II
theorem part_two :
  (∃ a : ℝ, ∀ x : ℝ, f a x ≤ 2*a) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≤ 2*a) → a ≥ 1/3) ∧
  (∀ x : ℝ, f (1/3) x ≤ 2*(1/3)) := by sorry

end part_one_part_two_l1280_128030


namespace vector_equality_l1280_128022

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def vector_b : Fin 2 → ℝ := ![1, -2]

theorem vector_equality (x : ℝ) : 
  ‖vector_a x + vector_b‖ = ‖vector_a x - vector_b‖ → x = 2 := by
sorry

end vector_equality_l1280_128022


namespace min_planks_for_color_condition_l1280_128033

/-- Represents a fence with colored planks. -/
structure Fence where
  n : ℕ                            -- number of planks
  colors : Fin n → Fin 100         -- color of each plank

/-- Checks if the fence satisfies the color condition. -/
def satisfiesColorCondition (f : Fence) : Prop :=
  ∀ (i j : Fin 100), i ≠ j →
    ∃ (p q : Fin f.n), p < q ∧ f.colors p = i ∧ f.colors q = j

/-- The theorem stating the minimum number of planks required. -/
theorem min_planks_for_color_condition :
  (∃ (f : Fence), satisfiesColorCondition f) →
  (∀ (f : Fence), satisfiesColorCondition f → f.n ≥ 199) ∧
  (∃ (f : Fence), f.n = 199 ∧ satisfiesColorCondition f) :=
sorry

end min_planks_for_color_condition_l1280_128033


namespace cubic_sum_theorem_l1280_128006

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (prod_sum_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c + 2*(a + b + c) = 672 := by
  sorry

end cubic_sum_theorem_l1280_128006


namespace rectangle_perimeters_l1280_128065

/-- The perimeter of a rectangle given its width and height. -/
def rectanglePerimeter (width height : ℝ) : ℝ := 2 * (width + height)

/-- The theorem stating the perimeters of the three rectangles formed from four photographs. -/
theorem rectangle_perimeters (photo_perimeter : ℝ) 
  (h1 : photo_perimeter = 20)
  (h2 : ∃ (w h : ℝ), rectanglePerimeter w h = photo_perimeter ∧ 
                      rectanglePerimeter (2*w) (2*h) = 40 ∧
                      rectanglePerimeter (4*w) h = 44 ∧
                      rectanglePerimeter w (4*h) = 56) :
  ∃ (p1 p2 p3 : ℝ), p1 = 40 ∧ p2 = 44 ∧ p3 = 56 ∧
    (p1 = 40 ∨ p1 = 44 ∨ p1 = 56) ∧
    (p2 = 40 ∨ p2 = 44 ∨ p2 = 56) ∧
    (p3 = 40 ∨ p3 = 44 ∨ p3 = 56) ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 := by
  sorry

end rectangle_perimeters_l1280_128065


namespace fiftycentchange_l1280_128091

/-- Represents the different types of U.S. coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Represents a combination of coins --/
def CoinCombination := List Coin

/-- Checks if a coin combination is valid for 50 cents --/
def isValidCombination (c : CoinCombination) : Bool := sorry

/-- Counts the number of quarters in a combination --/
def countQuarters (c : CoinCombination) : Nat := sorry

/-- Generates all valid coin combinations for 50 cents --/
def allCombinations : List CoinCombination := sorry

/-- The main theorem stating that there are 47 ways to make change for 50 cents --/
theorem fiftycentchange : 
  (allCombinations.filter (fun c => isValidCombination c ∧ countQuarters c ≤ 1)).length = 47 := by
  sorry

end fiftycentchange_l1280_128091


namespace sum_of_two_primes_odd_implies_one_is_two_l1280_128012

theorem sum_of_two_primes_odd_implies_one_is_two (p q : ℕ) :
  Prime p → Prime q → Odd (p + q) → (p = 2 ∨ q = 2) := by
  sorry

end sum_of_two_primes_odd_implies_one_is_two_l1280_128012


namespace max_min_product_l1280_128059

theorem max_min_product (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 12 → 
  a * b + b * c + c * a = 30 → 
  (min (a * b) (min (b * c) (c * a))) ≤ 2 :=
sorry

end max_min_product_l1280_128059


namespace divisibility_by_17_and_32_l1280_128045

theorem divisibility_by_17_and_32 (n : ℕ) (hn : n > 0) :
  (∃ k : ℤ, 5 * 3^(4*n + 1) + 2^(6*n + 1) = 17 * k) ∧
  (∃ m : ℤ, 5^2 * 7^(2*n + 1) + 3^(4*n) = 32 * m) := by
  sorry

end divisibility_by_17_and_32_l1280_128045


namespace unique_solution_quadratic_l1280_128019

theorem unique_solution_quadratic (q : ℝ) : 
  (q ≠ 0 ∧ ∀ x : ℝ, (q * x^2 - 18 * x + 8 = 0 → (∀ y : ℝ, q * y^2 - 18 * y + 8 = 0 → x = y))) ↔ 
  q = 81/8 := by
sorry

end unique_solution_quadratic_l1280_128019


namespace thirteen_seventh_mod_nine_l1280_128055

theorem thirteen_seventh_mod_nine (n : ℕ) : 
  13^7 % 9 = n ∧ 0 ≤ n ∧ n < 9 → n = 4 := by
  sorry

end thirteen_seventh_mod_nine_l1280_128055


namespace no_solution_exists_l1280_128018

theorem no_solution_exists : ¬ ∃ (m n : ℤ), 5 * m^2 - 6 * m * n + 7 * n^2 = 1985 := by
  sorry

end no_solution_exists_l1280_128018


namespace solution_property_l1280_128015

theorem solution_property (m n : ℝ) (hm : m ≠ 0) (h : m^2 + n*m - m = 0) : m + n = 1 := by
  sorry

end solution_property_l1280_128015


namespace rebecca_income_percentage_l1280_128036

def rebecca_income : ℕ := 15000
def jimmy_income : ℕ := 18000
def income_increase : ℕ := 7000

def new_rebecca_income : ℕ := rebecca_income + income_increase
def combined_income : ℕ := new_rebecca_income + jimmy_income

theorem rebecca_income_percentage :
  (new_rebecca_income : ℚ) / (combined_income : ℚ) = 55 / 100 := by sorry

end rebecca_income_percentage_l1280_128036


namespace percentage_difference_l1280_128007

theorem percentage_difference (x y : ℝ) (h : y = x * (1 + 0.53846153846153854)) :
  x = y * (1 - 0.35) := by
sorry

end percentage_difference_l1280_128007


namespace prime_even_intersection_l1280_128031

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def P : Set ℕ := {n : ℕ | isPrime n}
def Q : Set ℕ := {n : ℕ | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by sorry

end prime_even_intersection_l1280_128031


namespace quadratic_form_equivalence_l1280_128077

theorem quadratic_form_equivalence (x : ℝ) : x^2 + 4*x + 1 = (x + 2)^2 - 3 := by
  sorry

end quadratic_form_equivalence_l1280_128077


namespace max_goats_after_trading_l1280_128073

/-- Represents the trading system with coconuts, crabs, and goats -/
structure TradingSystem where
  coconuts_per_crab : ℕ
  crabs_per_goat : ℕ
  initial_coconuts : ℕ

/-- Calculates the number of goats obtained from trading coconuts -/
def goats_from_coconuts (ts : TradingSystem) : ℕ :=
  (ts.initial_coconuts / ts.coconuts_per_crab) / ts.crabs_per_goat

/-- Theorem stating that Max will have 19 goats after trading -/
theorem max_goats_after_trading :
  let ts : TradingSystem := {
    coconuts_per_crab := 3,
    crabs_per_goat := 6,
    initial_coconuts := 342
  }
  goats_from_coconuts ts = 19 := by
  sorry

end max_goats_after_trading_l1280_128073


namespace trajectory_intersection_fixed_point_l1280_128057

/-- The trajectory of a point equidistant from a fixed point and a fixed line -/
def Trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- A line not perpendicular to the x-axis -/
structure Line where
  t : ℝ
  m : ℝ
  h : t ≠ 0

/-- The condition that a line intersects the trajectory at two distinct points -/
def intersects_trajectory (l : Line) : Prop :=
  ∃ p q : ℝ × ℝ, p ≠ q ∧ p ∈ Trajectory ∧ q ∈ Trajectory ∧
    p.1 = l.t * p.2 + l.m ∧ q.1 = l.t * q.2 + l.m

/-- The condition that the x-axis is the angle bisector of ∠PBQ -/
def x_axis_bisects (l : Line) : Prop :=
  ∀ p q : ℝ × ℝ, p ≠ q → p ∈ Trajectory → q ∈ Trajectory →
    p.1 = l.t * p.2 + l.m → q.1 = l.t * q.2 + l.m →
    p.2 / (p.1 + 3) + q.2 / (q.1 + 3) = 0

/-- The main theorem -/
theorem trajectory_intersection_fixed_point :
  ∀ l : Line, intersects_trajectory l → x_axis_bisects l →
    l.m = 3 :=
sorry

end trajectory_intersection_fixed_point_l1280_128057


namespace cube_rotation_different_face_l1280_128058

-- Define a cube face
inductive CubeFace
  | Top
  | Bottom
  | Front
  | Back
  | Left
  | Right

-- Define a cube position
structure CubePosition :=
  (location : ℝ × ℝ × ℝ)
  (bottom_face : CubeFace)

-- Define a cube rotation
inductive CubeRotation
  | RollForward
  | RollBackward
  | RollLeft
  | RollRight

-- Define a function that applies a rotation to a cube position
def apply_rotation (pos : CubePosition) (rot : CubeRotation) : CubePosition :=
  sorry

-- Define the theorem
theorem cube_rotation_different_face :
  ∃ (initial_pos final_pos : CubePosition) (rotations : List CubeRotation),
    (initial_pos.location = final_pos.location) ∧
    (initial_pos.bottom_face ≠ final_pos.bottom_face) ∧
    (final_pos = rotations.foldl apply_rotation initial_pos) :=
  sorry

end cube_rotation_different_face_l1280_128058


namespace train_length_l1280_128066

/-- Given a train that crosses a signal post in 40 seconds and takes 2 minutes to cross a 1.8 kilometer
    long bridge at constant speed, the length of the train is 900 meters. -/
theorem train_length (signal_time : ℝ) (bridge_time : ℝ) (bridge_length : ℝ) :
  signal_time = 40 →
  bridge_time = 120 →
  bridge_length = 1800 →
  ∃ (train_length : ℝ) (speed : ℝ),
    train_length = speed * signal_time ∧
    train_length + bridge_length = speed * bridge_time ∧
    train_length = 900 :=
by sorry

end train_length_l1280_128066


namespace min_sum_nested_sqrt_l1280_128094

theorem min_sum_nested_sqrt (a b c : ℕ+) (k : ℕ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a : ℝ) * Real.sqrt ((b : ℝ) * Real.sqrt (c : ℝ)) = (k : ℝ)^2 →
  (∀ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z →
    (x : ℝ) * Real.sqrt ((y : ℝ) * Real.sqrt (z : ℝ)) = (k : ℝ)^2 →
    (x : ℕ) + (y : ℕ) + (z : ℕ) ≥ (a : ℕ) + (b : ℕ) + (c : ℕ)) →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 7 :=
by sorry

end min_sum_nested_sqrt_l1280_128094


namespace symmetry_about_x_axis_l1280_128056

/-- A point in the 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis for two points. -/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The theorem stating that (2, -3) is symmetric to (2, 3) with respect to the x-axis. -/
theorem symmetry_about_x_axis :
  symmetricAboutXAxis (Point.mk 2 3) (Point.mk 2 (-3)) := by
  sorry

end symmetry_about_x_axis_l1280_128056


namespace max_a_satisfies_equation_no_larger_a_satisfies_equation_max_a_is_maximum_l1280_128024

/-- The coefficient of x^4 in the expansion of (1-3x+ax^2)^8 --/
def coefficient_x4 (a : ℝ) : ℝ := 28 * a^2 + 2016 * a + 5670

/-- The equation that a must satisfy --/
def equation (a : ℝ) : Prop := coefficient_x4 a = 70

/-- The maximum value of a that satisfies the equation --/
noncomputable def max_a : ℝ := -36 + Real.sqrt 1096

theorem max_a_satisfies_equation : equation max_a :=
sorry

theorem no_larger_a_satisfies_equation :
  ∀ a : ℝ, a > max_a → ¬(equation a) :=
sorry

theorem max_a_is_maximum :
  ∃ (ε : ℝ), ε > 0 ∧ (∀ δ : ℝ, 0 < δ ∧ δ < ε → ¬(equation (max_a + δ))) :=
sorry

end max_a_satisfies_equation_no_larger_a_satisfies_equation_max_a_is_maximum_l1280_128024


namespace smallest_a_is_correct_l1280_128063

/-- A polynomial of the form x^3 - ax^2 + bx - 2310 with three positive integer roots -/
structure PolynomialWithThreeRoots where
  a : ℕ
  b : ℕ
  root1 : ℕ+
  root2 : ℕ+
  root3 : ℕ+
  is_root1 : (root1 : ℝ)^3 - a*(root1 : ℝ)^2 + b*(root1 : ℝ) - 2310 = 0
  is_root2 : (root2 : ℝ)^3 - a*(root2 : ℝ)^2 + b*(root2 : ℝ) - 2310 = 0
  is_root3 : (root3 : ℝ)^3 - a*(root3 : ℝ)^2 + b*(root3 : ℝ) - 2310 = 0

/-- The smallest possible value of a for a polynomial with three positive integer roots -/
def smallest_a (p : PolynomialWithThreeRoots) : ℕ := 78

theorem smallest_a_is_correct (p : PolynomialWithThreeRoots) :
  p.a ≥ smallest_a p :=
sorry

end smallest_a_is_correct_l1280_128063


namespace equation_solution_l1280_128069

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 - ⌊x₁⌋ = 2019) ∧ 
    (x₂^2 - ⌊x₂⌋ = 2019) ∧ 
    (x₁ = -Real.sqrt 1974) ∧ 
    (x₂ = Real.sqrt 2064) ∧ 
    (∀ (x : ℝ), x^2 - ⌊x⌋ = 2019 → x = x₁ ∨ x = x₂) :=
by
  sorry

#check equation_solution

end equation_solution_l1280_128069


namespace money_problem_l1280_128089

theorem money_problem (a b : ℝ) 
  (h1 : 4 * a + b = 68)
  (h2 : 2 * a - b < 16)
  (h3 : a + b > 22) :
  a < 14 ∧ b > 12 := by
  sorry

end money_problem_l1280_128089


namespace even_function_decreasing_interval_l1280_128070

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem even_function_decreasing_interval (k : ℝ) :
  (∀ x : ℝ, f k x = f k (-x)) →
  (∃ a : ℝ, ∀ x y : ℝ, x < y ∧ y ≤ 0 → f k x > f k y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f k x < f k y) :=
sorry

end even_function_decreasing_interval_l1280_128070
