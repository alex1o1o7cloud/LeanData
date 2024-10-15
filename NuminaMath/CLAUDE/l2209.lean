import Mathlib

namespace NUMINAMATH_CALUDE_lcm_problem_l2209_220963

-- Define the polynomials
def f (x : ℤ) : ℤ := 300 * x^4 + 425 * x^3 + 138 * x^2 - 17 * x - 6
def g (x : ℤ) : ℤ := 225 * x^4 - 109 * x^3 + 4

-- Define the LCM function for integers
def lcm_int (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

-- Define the LCM function for polynomials
noncomputable def lcm_poly (f g : ℤ → ℤ) : ℤ → ℤ := sorry

theorem lcm_problem :
  (lcm_int 4199 4641 5083 = 98141269893) ∧
  (lcm_poly f g = λ x => (225 * x^4 - 109 * x^3 + 4) * (4 * x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2209_220963


namespace NUMINAMATH_CALUDE_faye_coloring_books_l2209_220972

/-- Calculates the number of coloring books Faye bought -/
def coloring_books_bought (initial : ℕ) (given_away : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial - given_away)

theorem faye_coloring_books :
  coloring_books_bought 34 3 79 = 48 := by
  sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l2209_220972


namespace NUMINAMATH_CALUDE_log_problem_l2209_220969

theorem log_problem (y : ℝ) : y = (Real.log 4 / Real.log 16) ^ (Real.log 16 / Real.log 4) → Real.log y / Real.log 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2209_220969


namespace NUMINAMATH_CALUDE_bryans_bookshelves_l2209_220922

theorem bryans_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 34) (h2 : books_per_shelf = 17) :
  total_books / books_per_shelf = 2 :=
by sorry

end NUMINAMATH_CALUDE_bryans_bookshelves_l2209_220922


namespace NUMINAMATH_CALUDE_simplify_expression_l2209_220990

theorem simplify_expression (x : ℝ) :
  3 * x^3 + 4 * x^2 + 5 * x + 10 - (-6 + 3 * x^3 - 2 * x^2 + x) = 6 * x^2 + 4 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2209_220990


namespace NUMINAMATH_CALUDE_student_marks_average_l2209_220991

/-- Given a student's marks in mathematics, physics, and chemistry, 
    where the total marks in mathematics and physics is 50, 
    and the chemistry score is 20 marks more than physics, 
    prove that the average marks in mathematics and chemistry is 35. -/
theorem student_marks_average (m p c : ℕ) : 
  m + p = 50 → c = p + 20 → (m + c) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_average_l2209_220991


namespace NUMINAMATH_CALUDE_rajan_income_l2209_220908

/-- Represents the financial situation of two individuals --/
structure FinancialSituation where
  income_ratio : Rat
  expenditure_ratio : Rat
  savings : ℕ

/-- Calculates the income of the first person given a financial situation --/
def calculate_income (situation : FinancialSituation) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, Rajan's income is $7000 --/
theorem rajan_income (situation : FinancialSituation) 
  (h1 : situation.income_ratio = 7 / 6)
  (h2 : situation.expenditure_ratio = 6 / 5)
  (h3 : situation.savings = 1000) :
  calculate_income situation = 7000 :=
sorry

end NUMINAMATH_CALUDE_rajan_income_l2209_220908


namespace NUMINAMATH_CALUDE_function_maximum_implies_a_range_l2209_220983

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then -(x + 1) * Real.exp x else -2 * x - 1

theorem function_maximum_implies_a_range (a : ℝ) :
  (∃ (M : ℝ), ∀ (x : ℝ), f a x ≤ M) →
  a ≥ -(1/2) - 1/(2 * Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_function_maximum_implies_a_range_l2209_220983


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2209_220996

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 5, -3; 0, 3, -1; 7, -4, 2]
  Matrix.det A = 32 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2209_220996


namespace NUMINAMATH_CALUDE_inequality_proof_1_inequality_proof_2_l2209_220954

theorem inequality_proof_1 (x : ℝ) : 
  abs (x + 2) + abs (x - 2) > 6 ↔ x < -3 ∨ x > 3 := by sorry

theorem inequality_proof_2 (x : ℝ) : 
  abs (2*x - 1) - abs (x - 3) > 5 ↔ x < -7 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_1_inequality_proof_2_l2209_220954


namespace NUMINAMATH_CALUDE_root_difference_ratio_l2209_220995

/-- Given an equation x^4 - 7x - 3 = 0 with exactly two real roots a and b where a > b,
    the expression (a - b) / (a^4 - b^4) equals 1/7 -/
theorem root_difference_ratio (a b : ℝ) : 
  a > b → 
  a^4 - 7*a - 3 = 0 → 
  b^4 - 7*b - 3 = 0 → 
  (a - b) / (a^4 - b^4) = 1/7 := by
sorry

end NUMINAMATH_CALUDE_root_difference_ratio_l2209_220995


namespace NUMINAMATH_CALUDE_line_property_l2209_220981

/-- Given two points on a line, prove that m - 2b equals 21 --/
theorem line_property (x₁ y₁ x₂ y₂ m b : ℝ) 
  (h₁ : y₁ = m * x₁ + b) 
  (h₂ : y₂ = m * x₂ + b) 
  (h₃ : x₁ = 2) 
  (h₄ : y₁ = -3) 
  (h₅ : x₂ = 6) 
  (h₆ : y₂ = 9) : 
  m - 2 * b = 21 := by
  sorry

#check line_property

end NUMINAMATH_CALUDE_line_property_l2209_220981


namespace NUMINAMATH_CALUDE_selection_ways_l2209_220950

def group_size : ℕ := 8
def roles_to_fill : ℕ := 3

theorem selection_ways : (group_size.factorial) / ((group_size - roles_to_fill).factorial) = 336 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_l2209_220950


namespace NUMINAMATH_CALUDE_x_gt_1_necessary_not_sufficient_for_x_gt_2_l2209_220906

theorem x_gt_1_necessary_not_sufficient_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_1_necessary_not_sufficient_for_x_gt_2_l2209_220906


namespace NUMINAMATH_CALUDE_probability_all_heads_or_tails_proof_l2209_220986

/-- The probability of getting all heads or all tails when flipping six fair coins -/
def probability_all_heads_or_tails : ℚ := 1 / 32

/-- The number of fair coins being flipped -/
def num_coins : ℕ := 6

/-- A fair coin has two possible outcomes -/
def outcomes_per_coin : ℕ := 2

/-- The total number of possible outcomes when flipping the coins -/
def total_outcomes : ℕ := outcomes_per_coin ^ num_coins

/-- The number of favorable outcomes (all heads or all tails) -/
def favorable_outcomes : ℕ := 2

theorem probability_all_heads_or_tails_proof :
  probability_all_heads_or_tails = favorable_outcomes / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_probability_all_heads_or_tails_proof_l2209_220986


namespace NUMINAMATH_CALUDE_no_valid_y_exists_l2209_220980

theorem no_valid_y_exists : ¬∃ (y : ℝ), y^3 + y - 2 = 0 ∧ abs y < 1 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_y_exists_l2209_220980


namespace NUMINAMATH_CALUDE_sachin_age_is_28_l2209_220966

/-- The age of Sachin -/
def sachin_age : ℕ := sorry

/-- The age of Rahul -/
def rahul_age : ℕ := sorry

/-- Rahul is 8 years older than Sachin -/
axiom age_difference : rahul_age = sachin_age + 8

/-- The ratio of Sachin's age to Rahul's age is 7:9 -/
axiom age_ratio : (sachin_age : ℚ) / rahul_age = 7 / 9

/-- Sachin's age is 28 years -/
theorem sachin_age_is_28 : sachin_age = 28 := by sorry

end NUMINAMATH_CALUDE_sachin_age_is_28_l2209_220966


namespace NUMINAMATH_CALUDE_bart_earnings_l2209_220978

/-- Calculates the total earnings for Bart's survey work over five days --/
theorem bart_earnings (
  monday_rate : ℚ)
  (monday_questions : ℕ)
  (monday_surveys : ℕ)
  (tuesday_rate : ℚ)
  (tuesday_questions : ℕ)
  (tuesday_surveys : ℕ)
  (wednesday_rate : ℚ)
  (wednesday_questions : ℕ)
  (wednesday_surveys : ℕ)
  (thursday_rate : ℚ)
  (thursday_questions : ℕ)
  (thursday_surveys : ℕ)
  (friday_rate : ℚ)
  (friday_questions : ℕ)
  (friday_surveys : ℕ)
  (h1 : monday_rate = 20/100)
  (h2 : monday_questions = 10)
  (h3 : monday_surveys = 3)
  (h4 : tuesday_rate = 25/100)
  (h5 : tuesday_questions = 12)
  (h6 : tuesday_surveys = 4)
  (h7 : wednesday_rate = 10/100)
  (h8 : wednesday_questions = 15)
  (h9 : wednesday_surveys = 5)
  (h10 : thursday_rate = 15/100)
  (h11 : thursday_questions = 8)
  (h12 : thursday_surveys = 6)
  (h13 : friday_rate = 30/100)
  (h14 : friday_questions = 20)
  (h15 : friday_surveys = 2) :
  monday_rate * monday_questions * monday_surveys +
  tuesday_rate * tuesday_questions * tuesday_surveys +
  wednesday_rate * wednesday_questions * wednesday_surveys +
  thursday_rate * thursday_questions * thursday_surveys +
  friday_rate * friday_questions * friday_surveys = 447/10 := by
  sorry

end NUMINAMATH_CALUDE_bart_earnings_l2209_220978


namespace NUMINAMATH_CALUDE_bus_journey_distance_l2209_220947

theorem bus_journey_distance :
  ∀ (D s : ℝ),
    -- Original expected travel time
    (D / s = 2 + 1 + (3 * (D - 2 * s)) / (2 * s)) →
    -- Actual travel time (6 hours late)
    (2 + 1 + (3 * (D - 2 * s)) / (2 * s) = D / s + 6) →
    -- Travel time if delay occurred 120 miles further (4 hours late)
    ((2 * s + 120) / s + 1 + (3 * (D - 2 * s - 120)) / (2 * s) = D / s + 4) →
    -- Bus continues at 2/3 of original speed after delay
    (D - 2 * s) / ((2/3) * s) = (3 * (D - 2 * s)) / (2 * s) →
    D = 720 :=
by sorry

end NUMINAMATH_CALUDE_bus_journey_distance_l2209_220947


namespace NUMINAMATH_CALUDE_rectangle_width_equals_circle_area_l2209_220992

theorem rectangle_width_equals_circle_area (r : ℝ) (l w : ℝ) : 
  r = Real.sqrt 12 → 
  l = 3 * Real.sqrt 2 → 
  π * r^2 = l * w → 
  w = 2 * Real.sqrt 2 * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_equals_circle_area_l2209_220992


namespace NUMINAMATH_CALUDE_sum_of_pentagram_angles_l2209_220961

/-- A self-intersecting five-pointed star (pentagram) -/
structure Pentagram where
  vertices : Fin 5 → Point2
  is_self_intersecting : Bool

/-- The sum of angles at the vertices of a pentagram -/
def sum_of_vertex_angles (p : Pentagram) : ℝ := sorry

/-- Theorem: The sum of angles at the vertices of a self-intersecting pentagram is 180° -/
theorem sum_of_pentagram_angles (p : Pentagram) (h : p.is_self_intersecting = true) :
  sum_of_vertex_angles p = 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_pentagram_angles_l2209_220961


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l2209_220929

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that if a_2 and a_10 of an arithmetic sequence are roots of x^2 + 12x - 8 = 0, then a_6 = -6 -/
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2)^2 + 12*(a 2) - 8 = 0 →
  (a 10)^2 + 12*(a 10) - 8 = 0 →
  a 6 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l2209_220929


namespace NUMINAMATH_CALUDE_quadratic_function_properties_quadratic_function_root_range_l2209_220962

def f (q : ℝ) (x : ℝ) : ℝ := x^2 - 16*x + q + 3

theorem quadratic_function_properties (q : ℝ) :
  (∃ (min : ℝ), ∀ (x : ℝ), f q x ≥ min ∧ (∃ (x_min : ℝ), f q x_min = min) ∧ min = -60) →
  q = 1 :=
sorry

theorem quadratic_function_root_range (q : ℝ) :
  (∃ (x : ℝ), x ∈ Set.Icc (-1) 1 ∧ f q x = 0) →
  q ∈ Set.Icc (-20) 12 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_quadratic_function_root_range_l2209_220962


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l2209_220964

theorem min_value_of_fraction (x y : ℝ) 
  (hx : -3 ≤ x ∧ x ≤ 1) 
  (hy : -1 ≤ y ∧ y ≤ 3) 
  (hx_nonzero : x ≠ 0) : 
  (x + y) / x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l2209_220964


namespace NUMINAMATH_CALUDE_g_five_equals_one_l2209_220943

theorem g_five_equals_one (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x + y) = g x * g y) 
  (h2 : ∀ x : ℝ, g x ≠ 0) : 
  g 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_g_five_equals_one_l2209_220943


namespace NUMINAMATH_CALUDE_intersection_M_N_l2209_220973

def M : Set ℤ := {1, 2, 3, 4, 5, 6}

def N : Set ℤ := {x | -2 < x ∧ x < 5}

theorem intersection_M_N : M ∩ N = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2209_220973


namespace NUMINAMATH_CALUDE_colonization_combinations_eq_136_l2209_220959

/-- Represents the number of Earth-like planets -/
def earth_like : ℕ := 6

/-- Represents the number of Mars-like planets -/
def mars_like : ℕ := 6

/-- Represents the resource units required for an Earth-like planet -/
def earth_resource : ℕ := 3

/-- Represents the resource units required for a Mars-like planet -/
def mars_resource : ℕ := 1

/-- Represents the total available resource units -/
def total_resource : ℕ := 18

/-- Calculates the number of different combinations of planets that can be colonized -/
def colonization_combinations : ℕ := sorry

/-- Theorem stating that the number of different combinations of planets that can be colonized is 136 -/
theorem colonization_combinations_eq_136 : colonization_combinations = 136 := by sorry

end NUMINAMATH_CALUDE_colonization_combinations_eq_136_l2209_220959


namespace NUMINAMATH_CALUDE_third_side_is_three_l2209_220942

/-- Represents a triangle with two known side lengths and one unknown integer side length. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℕ

/-- The triangle inequality theorem for our specific triangle. -/
def triangle_inequality (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- The theorem stating that the third side of the triangle must be 3. -/
theorem third_side_is_three :
  ∀ t : Triangle,
    t.a = 3.14 →
    t.b = 0.67 →
    triangle_inequality t →
    t.c = 3 := by
  sorry

#check third_side_is_three

end NUMINAMATH_CALUDE_third_side_is_three_l2209_220942


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l2209_220914

theorem factor_implies_c_value (c : ℚ) :
  (∀ x : ℚ, (x + 7) ∣ (c * x^3 + 23 * x^2 - 3 * c * x + 45)) →
  c = 586 / 161 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l2209_220914


namespace NUMINAMATH_CALUDE_second_box_difference_l2209_220928

/-- Represents the amount of cereal in ounces for each box. -/
structure CerealBoxes where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Defines the properties of the cereal boxes based on the problem conditions. -/
def validCerealBoxes (boxes : CerealBoxes) : Prop :=
  boxes.first = 14 ∧
  boxes.second = boxes.first / 2 ∧
  boxes.second < boxes.third ∧
  boxes.first + boxes.second + boxes.third = 33

/-- Theorem stating that the difference between the third and second box is 5 ounces. -/
theorem second_box_difference (boxes : CerealBoxes) 
  (h : validCerealBoxes boxes) : boxes.third - boxes.second = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_box_difference_l2209_220928


namespace NUMINAMATH_CALUDE_lyra_remaining_budget_l2209_220934

/-- Calculates the remaining budget after Lyra's purchases --/
theorem lyra_remaining_budget (budget : ℝ) (chicken_price : ℝ) (beef_price : ℝ) (beef_weight : ℝ)
  (soup_price : ℝ) (soup_cans : ℕ) (milk_price : ℝ) (milk_discount : ℝ) :
  budget = 80 →
  chicken_price = 12 →
  beef_price = 3 →
  beef_weight = 4.5 →
  soup_price = 2 →
  soup_cans = 3 →
  milk_price = 4 →
  milk_discount = 0.1 →
  budget - (chicken_price + beef_price * beef_weight + 
    (soup_price * ↑soup_cans / 2) + milk_price * (1 - milk_discount)) = 47.9 := by
  sorry

#eval (80 : ℚ) - (12 + 3 * (9/2) + (2 * 3 / 2) + 4 * (1 - 1/10))

end NUMINAMATH_CALUDE_lyra_remaining_budget_l2209_220934


namespace NUMINAMATH_CALUDE_count_numbers_with_property_l2209_220944

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The reverse of a two-digit number. -/
def reverse (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

/-- The property that a number, when added to its reverse, sums to 144. -/
def hasProperty (n : ℕ) : Prop := n + reverse n = 144

/-- The main theorem stating that there are exactly 6 two-digit numbers satisfying the property. -/
theorem count_numbers_with_property : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, TwoDigitNumber n ∧ hasProperty n) ∧ Finset.card s = 6 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_property_l2209_220944


namespace NUMINAMATH_CALUDE_sin_double_angle_from_infinite_sum_l2209_220960

theorem sin_double_angle_from_infinite_sum (θ : ℝ) 
  (h : ∑' n, (Real.sin θ)^(2*n) = 4) : 
  Real.sin (2 * θ) = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_from_infinite_sum_l2209_220960


namespace NUMINAMATH_CALUDE_couples_in_club_is_three_l2209_220948

/-- Represents a book club with couples and single members -/
structure BookClub where
  weeksPerYear : ℕ
  ronPicksPerYear : ℕ
  singleMembers : ℕ

/-- Calculates the number of couples in the book club -/
def couplesInClub (club : BookClub) : ℕ :=
  (club.weeksPerYear - (2 * club.ronPicksPerYear + club.singleMembers * club.ronPicksPerYear)) / (2 * club.ronPicksPerYear)

/-- Theorem stating that the number of couples in the specified book club is 3 -/
theorem couples_in_club_is_three (club : BookClub) 
  (h1 : club.weeksPerYear = 52)
  (h2 : club.ronPicksPerYear = 4)
  (h3 : club.singleMembers = 5) : 
  couplesInClub club = 3 := by
  sorry

end NUMINAMATH_CALUDE_couples_in_club_is_three_l2209_220948


namespace NUMINAMATH_CALUDE_circle_area_equality_l2209_220999

theorem circle_area_equality (r₁ r₂ r : ℝ) (h₁ : r₁ = 24) (h₂ : r₂ = 35) :
  (π * r₂^2 - π * r₁^2 = π * r^2) → r = Real.sqrt 649 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equality_l2209_220999


namespace NUMINAMATH_CALUDE_sophia_rental_cost_l2209_220998

/-- Calculates the total cost of car rental given daily rate, per-mile rate, days rented, and miles driven -/
def total_rental_cost (daily_rate : ℚ) (per_mile_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + per_mile_rate * miles

/-- Proves that the total cost for Sophia's car rental is $275 -/
theorem sophia_rental_cost :
  total_rental_cost 30 0.25 5 500 = 275 := by
  sorry

end NUMINAMATH_CALUDE_sophia_rental_cost_l2209_220998


namespace NUMINAMATH_CALUDE_f_is_increasing_l2209_220941

def f (x : ℝ) := 2 * x + 1

theorem f_is_increasing : Monotone f := by sorry

end NUMINAMATH_CALUDE_f_is_increasing_l2209_220941


namespace NUMINAMATH_CALUDE_marys_tuesday_payment_l2209_220917

theorem marys_tuesday_payment 
  (credit_limit : ℕ) 
  (thursday_payment : ℕ) 
  (remaining_payment : ℕ) : 
  credit_limit - (thursday_payment + remaining_payment) = 15 :=
by
  sorry

#check marys_tuesday_payment 100 23 62

end NUMINAMATH_CALUDE_marys_tuesday_payment_l2209_220917


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l2209_220957

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((2 * x^2 + y^2) * (4 * x^2 + y^2)).sqrt) / (x * y) ≥ 3 :=
sorry

theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    (((2 * x₀^2 + y₀^2) * (4 * x₀^2 + y₀^2)).sqrt) / (x₀ * y₀) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l2209_220957


namespace NUMINAMATH_CALUDE_equation_roots_min_modulus_l2209_220994

noncomputable def find_a_b : ℝ × ℝ := sorry

theorem equation_roots (a b : ℝ) :
  find_a_b = (3, 3) :=
sorry

theorem min_modulus (a b : ℝ) (z : ℂ) :
  find_a_b = (a, b) →
  Complex.abs (z - (a + b * Complex.I)) = Complex.abs (2 / (1 + Complex.I)) →
  ∀ w : ℂ, Complex.abs (w - (a + b * Complex.I)) = Complex.abs (2 / (1 + Complex.I)) →
  Complex.abs z ≤ Complex.abs w →
  Complex.abs z = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_min_modulus_l2209_220994


namespace NUMINAMATH_CALUDE_expand_cubic_sum_simplify_complex_fraction_l2209_220968

-- Problem 1
theorem expand_cubic_sum (x y : ℝ) : (x + y) * (x^2 - x*y + y^2) = x^3 + y^3 := by
  sorry

-- Problem 2
theorem simplify_complex_fraction (a b c d : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (a^2 * b / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = -a^3 * b^3 / (8 * c * d^6) := by
  sorry

end NUMINAMATH_CALUDE_expand_cubic_sum_simplify_complex_fraction_l2209_220968


namespace NUMINAMATH_CALUDE_triangle_inradius_l2209_220925

/-- The inradius of a triangle with perimeter 36 and area 45 is 2.5 -/
theorem triangle_inradius (perimeter : ℝ) (area : ℝ) (inradius : ℝ) : 
  perimeter = 36 → area = 45 → inradius = area / (perimeter / 2) → inradius = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l2209_220925


namespace NUMINAMATH_CALUDE_rosa_bonheur_birthday_l2209_220946

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of leap years between two years -/
def leapYearCount (startYear endYear : Nat) : Nat :=
  let totalYears := endYear - startYear
  let potentialLeapYears := totalYears / 4
  potentialLeapYears - 1 -- Excluding 1900

/-- Calculates the day of the week given a starting day and number of days passed -/
def calculateDay (startDay : DayOfWeek) (daysPassed : Nat) : DayOfWeek :=
  match (daysPassed % 7) with
  | 0 => startDay
  | 1 => DayOfWeek.Sunday
  | 2 => DayOfWeek.Monday
  | 3 => DayOfWeek.Tuesday
  | 4 => DayOfWeek.Wednesday
  | 5 => DayOfWeek.Thursday
  | 6 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem rosa_bonheur_birthday 
  (anniversaryDay : DayOfWeek)
  (h : anniversaryDay = DayOfWeek.Wednesday) :
  calculateDay anniversaryDay 261 = DayOfWeek.Sunday := by
  sorry

#check rosa_bonheur_birthday

end NUMINAMATH_CALUDE_rosa_bonheur_birthday_l2209_220946


namespace NUMINAMATH_CALUDE_investment_rate_proof_l2209_220953

theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (target_income : ℝ) (available_rates : List ℝ) :
  total_investment = 12000 →
  first_investment = 5000 →
  second_investment = 4000 →
  first_rate = 0.03 →
  second_rate = 0.045 →
  target_income = 580 →
  available_rates = [0.05, 0.055, 0.06, 0.065, 0.07] →
  ∃ (optimal_rate : ℝ), 
    optimal_rate ∈ available_rates ∧
    optimal_rate = 0.07 ∧
    ∀ (rate : ℝ), rate ∈ available_rates →
      |((target_income - (first_investment * first_rate + second_investment * second_rate)) / 
        (total_investment - first_investment - second_investment)) - optimal_rate| ≤
      |((target_income - (first_investment * first_rate + second_investment * second_rate)) / 
        (total_investment - first_investment - second_investment)) - rate| :=
by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l2209_220953


namespace NUMINAMATH_CALUDE_unique_solution_is_twelve_l2209_220971

/-- Definition of the ♣ operation -/
def clubsuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating that 12 is the unique solution to A ♣ 7 = 76 -/
theorem unique_solution_is_twelve :
  ∃! A : ℝ, clubsuit A 7 = 76 ∧ A = 12 := by sorry

end NUMINAMATH_CALUDE_unique_solution_is_twelve_l2209_220971


namespace NUMINAMATH_CALUDE_f_is_generalized_distance_l2209_220916

-- Define the binary function f
def f (x y : ℝ) : ℝ := x^2 + y^2

-- State the theorem
theorem f_is_generalized_distance :
  (∀ x y : ℝ, f x y ≥ 0 ∧ (f x y = 0 ↔ x = 0 ∧ y = 0)) ∧ 
  (∀ x y : ℝ, f x y = f y x) ∧
  (∀ x y z : ℝ, f x y ≤ f x z + f z y) :=
sorry

end NUMINAMATH_CALUDE_f_is_generalized_distance_l2209_220916


namespace NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_complementary_l2209_220901

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
def DrawOutcome := Prod BallColor BallColor

/-- The set of all possible outcomes when drawing two balls -/
def SampleSpace : Set DrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def ExactlyOneBlack : Set DrawOutcome := sorry

/-- The event of drawing exactly two black balls -/
def ExactlyTwoBlack : Set DrawOutcome := sorry

/-- Two events are mutually exclusive if their intersection is empty -/
def MutuallyExclusive (A B : Set DrawOutcome) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if their union is the entire sample space -/
def Complementary (A B : Set DrawOutcome) : Prop :=
  A ∪ B = SampleSpace

theorem exactly_one_two_black_mutually_exclusive_not_complementary :
  MutuallyExclusive ExactlyOneBlack ExactlyTwoBlack ∧
  ¬Complementary ExactlyOneBlack ExactlyTwoBlack :=
sorry

end NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_complementary_l2209_220901


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2209_220937

theorem partial_fraction_sum_zero : 
  ∃ (A B C D E F : ℝ), 
    (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
      1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) = 
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
    A + B + C + D + E + F = 0 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2209_220937


namespace NUMINAMATH_CALUDE_couple_seating_arrangements_l2209_220905

/-- Represents a couple (a boy and a girl) -/
structure Couple :=
  (boy : Nat)
  (girl : Nat)

/-- Represents a seating arrangement on the bench -/
structure Arrangement :=
  (seat1 : Nat)
  (seat2 : Nat)
  (seat3 : Nat)
  (seat4 : Nat)

/-- Checks if a given arrangement is valid (each couple sits together) -/
def isValidArrangement (c1 c2 : Couple) (arr : Arrangement) : Prop :=
  (arr.seat1 = c1.boy ∧ arr.seat2 = c1.girl ∧ arr.seat3 = c2.boy ∧ arr.seat4 = c2.girl) ∨
  (arr.seat1 = c1.girl ∧ arr.seat2 = c1.boy ∧ arr.seat3 = c2.boy ∧ arr.seat4 = c2.girl) ∨
  (arr.seat1 = c1.boy ∧ arr.seat2 = c1.girl ∧ arr.seat3 = c2.girl ∧ arr.seat4 = c2.boy) ∨
  (arr.seat1 = c1.girl ∧ arr.seat2 = c1.boy ∧ arr.seat3 = c2.girl ∧ arr.seat4 = c2.boy) ∨
  (arr.seat1 = c2.boy ∧ arr.seat2 = c2.girl ∧ arr.seat3 = c1.boy ∧ arr.seat4 = c1.girl) ∨
  (arr.seat1 = c2.girl ∧ arr.seat2 = c2.boy ∧ arr.seat3 = c1.boy ∧ arr.seat4 = c1.girl) ∨
  (arr.seat1 = c2.boy ∧ arr.seat2 = c2.girl ∧ arr.seat3 = c1.girl ∧ arr.seat4 = c1.boy) ∨
  (arr.seat1 = c2.girl ∧ arr.seat2 = c2.boy ∧ arr.seat3 = c1.girl ∧ arr.seat4 = c1.boy)

/-- The main theorem: there are exactly 8 valid seating arrangements -/
theorem couple_seating_arrangements (c1 c2 : Couple) :
  ∃! (arrangements : Finset Arrangement), 
    (∀ arr ∈ arrangements, isValidArrangement c1 c2 arr) ∧
    arrangements.card = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_couple_seating_arrangements_l2209_220905


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l2209_220982

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l2209_220982


namespace NUMINAMATH_CALUDE_existence_of_large_solutions_l2209_220923

theorem existence_of_large_solutions :
  ∃ (x y z u v : ℕ), 
    x > 2000 ∧ y > 2000 ∧ z > 2000 ∧ u > 2000 ∧ v > 2000 ∧
    x^2 + y^2 + z^2 + u^2 + v^2 = x*y*z*u*v - 65 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_large_solutions_l2209_220923


namespace NUMINAMATH_CALUDE_dress_discount_problem_l2209_220967

theorem dress_discount_problem (P D : ℝ) : 
  P * (1 - D) * 1.25 = 71.4 →
  P - 71.4 = 5.25 →
  D = 0.255 := by
  sorry

end NUMINAMATH_CALUDE_dress_discount_problem_l2209_220967


namespace NUMINAMATH_CALUDE_cubic_identity_l2209_220904

theorem cubic_identity (a b : ℝ) : (a + b) * (a^2 - a*b + b^2) = a^3 + b^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l2209_220904


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2209_220932

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2209_220932


namespace NUMINAMATH_CALUDE_geometric_sequence_implies_b_eq_4_b_eq_4_not_sufficient_geometric_sequence_sufficient_not_necessary_l2209_220993

/-- A geometric sequence with first term 1, fifth term 16, and middle terms a, b, c -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ a = q ∧ b = q^2 ∧ c = q^3 ∧ 16 = q^4

/-- The statement that b = 4 is a necessary condition for the geometric sequence -/
theorem geometric_sequence_implies_b_eq_4 (a b c : ℝ) :
  is_geometric_sequence a b c → b = 4 :=
sorry

/-- The statement that b = 4 is not a sufficient condition for the geometric sequence -/
theorem b_eq_4_not_sufficient (a b c : ℝ) :
  b = 4 → ¬(∀ a c : ℝ, is_geometric_sequence a b c) :=
sorry

/-- The main theorem stating that the geometric sequence condition is sufficient but not necessary for b = 4 -/
theorem geometric_sequence_sufficient_not_necessary :
  (∃ a b c : ℝ, is_geometric_sequence a b c ∧ b = 4) ∧
  (∃ b : ℝ, b = 4 ∧ ¬(∀ a c : ℝ, is_geometric_sequence a b c)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_implies_b_eq_4_b_eq_4_not_sufficient_geometric_sequence_sufficient_not_necessary_l2209_220993


namespace NUMINAMATH_CALUDE_equation_C_most_suitable_l2209_220910

-- Define the equations
def equation_A : ℝ → Prop := λ x ↦ 2 * x^2 = 8
def equation_B : ℝ → Prop := λ x ↦ x * (x + 2) = x + 2
def equation_C : ℝ → Prop := λ x ↦ x^2 - 2*x = 3
def equation_D : ℝ → Prop := λ x ↦ 2 * x^2 + x - 1 = 0

-- Define a predicate for suitability for completing the square method
def suitable_for_completing_square (eq : ℝ → Prop) : Prop := sorry

-- Theorem stating that equation C is most suitable for completing the square
theorem equation_C_most_suitable :
  suitable_for_completing_square equation_C ∧
  (¬suitable_for_completing_square equation_A ∨
   ¬suitable_for_completing_square equation_B ∨
   ¬suitable_for_completing_square equation_D) :=
sorry

end NUMINAMATH_CALUDE_equation_C_most_suitable_l2209_220910


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2209_220988

/-- The quadratic equation kx^2 - 6x + 9 = 0 has real roots if and only if k ≤ 1 and k ≠ 0 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2209_220988


namespace NUMINAMATH_CALUDE_max_profit_at_150_l2209_220965

/-- Represents the total revenue function for the workshop --/
noncomputable def H (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 200 then 400 * x - x^2 else 40000

/-- Represents the total cost function for the workshop --/
def total_cost (x : ℝ) : ℝ := 7500 + 100 * x

/-- Represents the profit function for the workshop --/
noncomputable def profit (x : ℝ) : ℝ := H x - total_cost x

/-- Theorem stating the maximum profit and corresponding production volume --/
theorem max_profit_at_150 :
  (∃ (x : ℝ), ∀ (y : ℝ), profit y ≤ profit x) ∧
  (∀ (x : ℝ), profit x ≤ 15000) ∧
  profit 150 = 15000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_150_l2209_220965


namespace NUMINAMATH_CALUDE_find_m_value_l2209_220976

def A : Set ℝ := {-1, 1, 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem find_m_value : ∀ m : ℝ, B m ⊆ A → m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l2209_220976


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l2209_220985

theorem fibonacci_like_sequence (a : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h2 : a 11 = 157) :
  a 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l2209_220985


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2209_220984

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 5) * (Real.sqrt 8 / Real.sqrt 9) * (Real.sqrt 3 / Real.sqrt 7) = 
  (4 * Real.sqrt 105) / 105 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2209_220984


namespace NUMINAMATH_CALUDE_cube_root_of_sum_l2209_220933

theorem cube_root_of_sum (x y : ℝ) : 
  (Real.sqrt (x - 1) + (y + 2)^2 = 0) → 
  (x + y)^(1/3 : ℝ) = -1 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_sum_l2209_220933


namespace NUMINAMATH_CALUDE_volume_of_square_cross_section_cylinder_l2209_220931

/-- A cylinder with height 40 cm and a square cross-section when cut along the diameter of the base -/
structure SquareCrossSectionCylinder where
  height : ℝ
  height_eq : height = 40
  square_cross_section : Bool

/-- The volume of the cylinder in cubic decimeters -/
def cylinder_volume (c : SquareCrossSectionCylinder) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specified cylinder is 502.4 cubic decimeters -/
theorem volume_of_square_cross_section_cylinder :
  ∀ (c : SquareCrossSectionCylinder), cylinder_volume c = 502.4 :=
by sorry

end NUMINAMATH_CALUDE_volume_of_square_cross_section_cylinder_l2209_220931


namespace NUMINAMATH_CALUDE_bread_slices_eaten_for_breakfast_l2209_220902

theorem bread_slices_eaten_for_breakfast 
  (total_slices : ℕ) 
  (lunch_slices : ℕ) 
  (remaining_slices : ℕ) 
  (h1 : total_slices = 12)
  (h2 : lunch_slices = 2)
  (h3 : remaining_slices = 6) :
  (total_slices - (remaining_slices + lunch_slices)) / total_slices = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_eaten_for_breakfast_l2209_220902


namespace NUMINAMATH_CALUDE_booth_active_days_l2209_220955

/-- Represents the carnival snack booth scenario -/
def carnival_booth (days : ℕ) : Prop :=
  let popcorn_revenue := 50
  let cotton_candy_revenue := 3 * popcorn_revenue
  let daily_revenue := popcorn_revenue + cotton_candy_revenue
  let daily_rent := 30
  let ingredient_cost := 75
  let total_revenue := days * daily_revenue
  let total_rent := days * daily_rent
  let profit := total_revenue - total_rent - ingredient_cost
  profit = 895

/-- Theorem stating that the booth was active for 5 days -/
theorem booth_active_days : ∃ (d : ℕ), carnival_booth d ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_booth_active_days_l2209_220955


namespace NUMINAMATH_CALUDE_conference_handshakes_l2209_220915

theorem conference_handshakes (n : ℕ) (m : ℕ) : 
  n = 15 →  -- number of married couples
  m = 3 →   -- number of men who don't shake hands with each other
  (2 * n * (2 * n - 1) - 2 * n) / 2 - (m * (m - 1)) / 2 = 417 :=
by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l2209_220915


namespace NUMINAMATH_CALUDE_translation_theorem_l2209_220927

/-- Represents a point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

theorem translation_theorem :
  let M : Point := { x := -4, y := 3 }
  let M1 := translateHorizontal M (-3)
  let M2 := translateVertical M1 2
  M2 = { x := -7, y := 5 } := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l2209_220927


namespace NUMINAMATH_CALUDE_inverse_sum_product_l2209_220907

theorem inverse_sum_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 2 * x + y / 2 ≠ 0) :
  (2 * x + y / 2)⁻¹ * ((2 * x)⁻¹ + (y / 2)⁻¹) = (x * y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_product_l2209_220907


namespace NUMINAMATH_CALUDE_min_balls_to_draw_l2209_220977

theorem min_balls_to_draw (black white red : ℕ) (h1 : black = 10) (h2 : white = 9) (h3 : red = 8) :
  black + white + 1 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_l2209_220977


namespace NUMINAMATH_CALUDE_monomial_simplification_l2209_220920

theorem monomial_simplification (a : ℕ) (M : ℕ) (h1 : a = 100) (h2 : M = a) :
  (M : ℚ) / (a + 1 : ℚ) - 1 / ((a^2 : ℚ) + a) = 99 / 100 := by
  sorry

end NUMINAMATH_CALUDE_monomial_simplification_l2209_220920


namespace NUMINAMATH_CALUDE_finite_triples_satisfying_equation_l2209_220919

theorem finite_triples_satisfying_equation : 
  ∃ (S : Set (ℕ × ℕ × ℕ)), Finite S ∧ 
  ∀ (a b c : ℕ), (a * b * c = 2009 * (a + b + c) ∧ a > 0 ∧ b > 0 ∧ c > 0) ↔ (a, b, c) ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_triples_satisfying_equation_l2209_220919


namespace NUMINAMATH_CALUDE_training_hours_calculation_l2209_220921

/-- Given a person trains for a specific number of hours per day and a total number of days,
    calculate the total hours spent training. -/
def total_training_hours (hours_per_day : ℕ) (total_days : ℕ) : ℕ :=
  hours_per_day * total_days

/-- Theorem: A person training for 5 hours every day for 42 days spends 210 hours in total. -/
theorem training_hours_calculation :
  let hours_per_day : ℕ := 5
  let initial_days : ℕ := 30
  let additional_days : ℕ := 12
  let total_days : ℕ := initial_days + additional_days
  total_training_hours hours_per_day total_days = 210 := by
  sorry

#check training_hours_calculation

end NUMINAMATH_CALUDE_training_hours_calculation_l2209_220921


namespace NUMINAMATH_CALUDE_lightbulb_combinations_eq_seven_l2209_220997

/-- The number of ways to turn on at least one out of three lightbulbs -/
def lightbulb_combinations : ℕ :=
  -- Number of ways with one bulb on
  (3 : ℕ).choose 1 +
  -- Number of ways with two bulbs on
  (3 : ℕ).choose 2 +
  -- Number of ways with three bulbs on
  (3 : ℕ).choose 3

/-- Theorem stating that the number of ways to turn on at least one out of three lightbulbs is 7 -/
theorem lightbulb_combinations_eq_seven : lightbulb_combinations = 7 := by
  sorry

end NUMINAMATH_CALUDE_lightbulb_combinations_eq_seven_l2209_220997


namespace NUMINAMATH_CALUDE_max_value_theorem_l2209_220913

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  x + Real.sqrt (x * y) + (x * y * z) ^ (1/4) ≤ 7/6 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2209_220913


namespace NUMINAMATH_CALUDE_four_number_sequence_l2209_220987

theorem four_number_sequence (a b c d : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) → -- Geometric sequence condition
  a + b + c = 19 →
  (∃ q : ℝ, c = b + q ∧ d = c + q) → -- Arithmetic sequence condition
  b + c + d = 12 →
  ((a = 25 ∧ b = -10 ∧ c = 4 ∧ d = 18) ∨ (a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_four_number_sequence_l2209_220987


namespace NUMINAMATH_CALUDE_max_a_for_defined_f_l2209_220958

-- Define the function g(x) = |x-2| + |x-a|
def g (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- State the theorem
theorem max_a_for_defined_f :
  (∃ (a_max : ℝ), (∀ (a : ℝ), (∀ (x : ℝ), g a x ≥ 2 * a) → a ≤ a_max) ∧
                  (∀ (x : ℝ), g a_max x ≥ 2 * a_max) ∧
                  a_max = 2/3) :=
sorry

end NUMINAMATH_CALUDE_max_a_for_defined_f_l2209_220958


namespace NUMINAMATH_CALUDE_polynomial_identity_l2209_220912

theorem polynomial_identity (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (x - 1)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅ + a₇)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2209_220912


namespace NUMINAMATH_CALUDE_max_x_value_l2209_220989

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 7) (prod_sum_eq : x*y + x*z + y*z = 10) :
  x ≤ 3 ∧ ∃ (y' z' : ℝ), x = 3 ∧ y' + z' = 4 ∧ 3*y' + 3*z' + y'*z' = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l2209_220989


namespace NUMINAMATH_CALUDE_inversion_of_line_l2209_220938

/-- A circle in a plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- A line in a plane -/
structure Line :=
  (point : ℝ × ℝ)
  (direction : ℝ × ℝ)

/-- The result of inverting a line with respect to a circle -/
inductive InversionResult
  | SameLine : InversionResult
  | Circle : (ℝ × ℝ) → ℝ → InversionResult

/-- Inversion of a line with respect to a circle -/
def invert (l : Line) (c : Circle) : InversionResult :=
  sorry

/-- Theorem: The image of a line under inversion is either the line itself or a circle passing through the center of inversion -/
theorem inversion_of_line (l : Line) (c : Circle) :
  (invert l c = InversionResult.SameLine ∧ l.point = c.center) ∨
  (∃ center radius, invert l c = InversionResult.Circle center radius ∧ center = c.center) :=
sorry

end NUMINAMATH_CALUDE_inversion_of_line_l2209_220938


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2209_220940

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum1 : a 3 + a 6 = 11)
  (h_sum2 : a 5 + a 8 = 39) :
  ∃ d : ℝ, d = 7 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2209_220940


namespace NUMINAMATH_CALUDE_summer_camp_selection_probability_l2209_220936

theorem summer_camp_selection_probability :
  let total_students : ℕ := 9
  let male_students : ℕ := 5
  let female_students : ℕ := 4
  let selected_students : ℕ := 5
  let min_per_gender : ℕ := 2

  let total_combinations := Nat.choose total_students selected_students
  let valid_combinations := Nat.choose male_students min_per_gender * Nat.choose female_students (selected_students - min_per_gender) +
                            Nat.choose male_students (selected_students - min_per_gender) * Nat.choose female_students min_per_gender

  (valid_combinations : ℚ) / total_combinations = 50 / 63 :=
by sorry

end NUMINAMATH_CALUDE_summer_camp_selection_probability_l2209_220936


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l2209_220970

/-- Pentagon ABCDE with specified side lengths and relationships -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  AC : ℝ
  AD : ℝ
  AE : ℝ
  ab_eq_one : AB = 1
  bc_eq_one : BC = 1
  cd_eq_one : CD = 1
  de_eq_one : DE = 1
  ac_pythagoras : AC^2 = AB^2 + BC^2
  ad_pythagoras : AD^2 = AC^2 + CD^2
  ae_pythagoras : AE^2 = AD^2 + DE^2

/-- The perimeter of pentagon ABCDE is 6 -/
theorem pentagon_perimeter (p : Pentagon) : p.AB + p.BC + p.CD + p.DE + p.AE = 6 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_perimeter_l2209_220970


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l2209_220974

def product : ℕ := 95 * 97 * 99 * 101

theorem distinct_prime_factors_count :
  (Nat.factors product).toFinset.card = 6 :=
sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l2209_220974


namespace NUMINAMATH_CALUDE_gold_coin_puzzle_l2209_220909

theorem gold_coin_puzzle (n : ℕ) (c : ℕ) : 
  (∃ k : ℕ, n = 11 * (c - 3) ∧ k = c - 3) ∧ 
  n = 7 * c + 5 →
  n = 75 :=
by sorry

end NUMINAMATH_CALUDE_gold_coin_puzzle_l2209_220909


namespace NUMINAMATH_CALUDE_smallest_degree_is_five_l2209_220951

/-- The smallest degree of a polynomial p(x) such that (3x^5 - 5x^3 + 4x - 2) / p(x) has a horizontal asymptote -/
def smallest_degree_with_horizontal_asymptote : ℕ := by
  sorry

/-- The numerator of the rational function -/
def numerator (x : ℝ) : ℝ := 3*x^5 - 5*x^3 + 4*x - 2

/-- The rational function has a horizontal asymptote -/
def has_horizontal_asymptote (p : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x, |x| > M → |numerator x / p x - L| < ε

theorem smallest_degree_is_five :
  smallest_degree_with_horizontal_asymptote = 5 ∧
  ∃ (p : ℝ → ℝ), (∀ x, ∃ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ), p x = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) ∧
    has_horizontal_asymptote p ∧
    ∀ (q : ℝ → ℝ), (∀ x, ∃ (b₀ b₁ b₂ b₃ b₄ : ℝ), q x = b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
      ¬(has_horizontal_asymptote q) := by
  sorry

end NUMINAMATH_CALUDE_smallest_degree_is_five_l2209_220951


namespace NUMINAMATH_CALUDE_fruit_salad_cherries_l2209_220952

theorem fruit_salad_cherries (b r g c : ℕ) : 
  b + r + g + c = 580 →
  r = 2 * b →
  g = 3 * c →
  c = 3 * r →
  c = 129 := by
sorry

end NUMINAMATH_CALUDE_fruit_salad_cherries_l2209_220952


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l2209_220918

theorem dvd_pack_cost (total_cost : ℕ) (num_packs : ℕ) (cost_per_pack : ℕ) :
  total_cost = 2673 →
  num_packs = 33 →
  cost_per_pack = total_cost / num_packs →
  cost_per_pack = 81 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l2209_220918


namespace NUMINAMATH_CALUDE_perimeter_of_square_d_l2209_220900

/-- Given a square C with perimeter 32 cm and a square D with area equal to one-third the area of square C, 
    the perimeter of square D is (32√3)/3 cm. -/
theorem perimeter_of_square_d (C D : Real) : 
  (C = 32) →  -- perimeter of square C is 32 cm
  (D^2 = (C/4)^2 / 3) →  -- area of square D is one-third the area of square C
  (4 * D = 32 * Real.sqrt 3 / 3) := by  -- perimeter of square D is (32√3)/3 cm
sorry

end NUMINAMATH_CALUDE_perimeter_of_square_d_l2209_220900


namespace NUMINAMATH_CALUDE_infinite_solutions_implies_c_value_l2209_220975

/-- If infinitely many values of y satisfy the equation 3(5 + 2cy) = 15y + 15 + y^2, then c = 2.5 -/
theorem infinite_solutions_implies_c_value (c : ℝ) : 
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 15 * y + 15 + y^2) → c = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_implies_c_value_l2209_220975


namespace NUMINAMATH_CALUDE_system_solutions_l2209_220930

theorem system_solutions :
  ∀ x y : ℝ,
  (y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2) ∨ 
   (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2209_220930


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l2209_220903

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 6 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l2209_220903


namespace NUMINAMATH_CALUDE_expected_sales_after_price_change_l2209_220911

/-- Represents the relationship between price and sales of blenders -/
structure BlenderSales where
  price : ℝ
  units : ℝ

/-- The constant of proportionality for the inverse relationship -/
def k : ℝ := 15 * 500

/-- The inverse proportionality relationship between price and sales -/
def inverse_proportional (bs : BlenderSales) : Prop :=
  bs.price * bs.units = k

/-- The new price after discount -/
def new_price : ℝ := 1000 * (1 - 0.1)

/-- Theorem stating the expected sales under the new pricing scheme -/
theorem expected_sales_after_price_change 
  (initial : BlenderSales) 
  (h_initial : initial.price = 500 ∧ initial.units = 15) 
  (h_inverse : inverse_proportional initial) :
  ∃ (new : BlenderSales), 
    new.price = new_price ∧ 
    inverse_proportional new ∧ 
    (8 ≤ new.units ∧ new.units < 9) := by
  sorry

end NUMINAMATH_CALUDE_expected_sales_after_price_change_l2209_220911


namespace NUMINAMATH_CALUDE_gum_sharing_proof_l2209_220939

/-- The number of people sharing gum equally -/
def num_people (john_gum cole_gum aubrey_gum pieces_per_person : ℕ) : ℕ :=
  (john_gum + cole_gum + aubrey_gum) / pieces_per_person

/-- Proof that 3 people are sharing the gum -/
theorem gum_sharing_proof :
  num_people 54 45 0 33 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gum_sharing_proof_l2209_220939


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2209_220956

/-- Represents a 2D vector --/
structure Vector2D where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D --/
structure ParametricLine where
  origin : Vector2D
  direction : Vector2D

/-- The first line --/
def line1 : ParametricLine :=
  { origin := { x := 2, y := 3 },
    direction := { x := -1, y := 4 } }

/-- The second line --/
def line2 : ParametricLine :=
  { origin := { x := -1, y := 6 },
    direction := { x := 3, y := 5 } }

/-- The intersection point of two parametric lines --/
def intersection (l1 l2 : ParametricLine) : Vector2D :=
  { x := 28 / 17, y := 75 / 17 }

/-- Theorem stating that the intersection of line1 and line2 is (28/17, 75/17) --/
theorem intersection_of_lines :
  intersection line1 line2 = { x := 28 / 17, y := 75 / 17 } := by
  sorry

#check intersection_of_lines

end NUMINAMATH_CALUDE_intersection_of_lines_l2209_220956


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l2209_220935

theorem fourth_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (x^(1/4) : ℝ) - 15 / (8 - (x^(1/4) : ℝ))
  ∀ x : ℝ, f x = 0 ↔ x = 81 ∨ x = 625 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l2209_220935


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_k_value_l2209_220926

theorem quadratic_roots_imply_k_value (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 8 * x + k = 0 ↔ x = -2 + Real.sqrt 6 ∨ x = -2 - Real.sqrt 6) →
  k = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_k_value_l2209_220926


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l2209_220979

/-- The percentage of students who own cats in a school survey -/
theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 500) 
  (h2 : cat_owners = 75) : 
  (cat_owners : ℝ) / total_students * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l2209_220979


namespace NUMINAMATH_CALUDE_lcm_924_660_l2209_220924

theorem lcm_924_660 : Nat.lcm 924 660 = 4620 := by
  sorry

end NUMINAMATH_CALUDE_lcm_924_660_l2209_220924


namespace NUMINAMATH_CALUDE_empty_set_implies_a_range_l2209_220945

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + a * x + 1 = 0}

-- State the theorem
theorem empty_set_implies_a_range (a : ℝ) : 
  A a = ∅ → 0 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_empty_set_implies_a_range_l2209_220945


namespace NUMINAMATH_CALUDE_intersection_A_B_zero_range_of_m_l2209_220949

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

-- Theorem 1: Intersection of A and B when m = 0
theorem intersection_A_B_zero : A ∩ B 0 = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Range of m when q is necessary but not sufficient for p
theorem range_of_m (h : ∀ x, p x → q x m) : 
  m ≤ -2 ∨ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_zero_range_of_m_l2209_220949
