import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l627_62783

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, (x < -1/3 ∨ x > 1/2) ↔ a*x^2 + b*x + 2 < 0) → 
  a - b = -14 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l627_62783


namespace NUMINAMATH_CALUDE_inverse_proportion_in_first_third_quadrants_l627_62717

/-- An inverse proportion function -/
def InverseProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- A function whose graph lies in the first and third quadrants -/
def FirstThirdQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

theorem inverse_proportion_in_first_third_quadrants
  (f : ℝ → ℝ) (h1 : InverseProportion f) (h2 : FirstThirdQuadrants f) :
  ∃ k : ℝ, k > 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_in_first_third_quadrants_l627_62717


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l627_62794

/-- The orthocenter of a triangle is the point where all three altitudes intersect. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Given three points A, B, and C in 3D space, this theorem states that
    the orthocenter of the triangle formed by these points is (2, 3, 4). -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, -1, 2)
  let C : ℝ × ℝ × ℝ := (1, 6, 5)
  orthocenter A B C = (2, 3, 4) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l627_62794


namespace NUMINAMATH_CALUDE_banana_arrangements_l627_62720

def word_length : ℕ := 7

def identical_b_count : ℕ := 2

def distinct_letter_count : ℕ := 5

theorem banana_arrangements :
  (word_length.factorial / identical_b_count.factorial) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l627_62720


namespace NUMINAMATH_CALUDE_triangle_sequence_2009_position_l627_62747

def triangle_sequence (n : ℕ) : ℕ := n

def row_of_term (n : ℕ) : ℕ :=
  (n.sqrt : ℕ) + 1

def position_in_row (n : ℕ) : ℕ :=
  n - (row_of_term n - 1)^2

theorem triangle_sequence_2009_position :
  row_of_term 2009 = 45 ∧ position_in_row 2009 = 73 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sequence_2009_position_l627_62747


namespace NUMINAMATH_CALUDE_bamboo_problem_l627_62749

/-- 
Given a geometric sequence of 9 terms where the sum of the first 3 terms is 2 
and the sum of the last 3 terms is 128, the 5th term is equal to 32/7.
-/
theorem bamboo_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence
  a 1 + a 2 + a 3 = 2 →         -- sum of first 3 terms
  a 7 + a 8 + a 9 = 128 →       -- sum of last 3 terms
  a 5 = 32 / 7 := by
sorry

end NUMINAMATH_CALUDE_bamboo_problem_l627_62749


namespace NUMINAMATH_CALUDE_lottery_winning_numbers_l627_62715

/-- Calculates the number of winning numbers on each lottery ticket -/
theorem lottery_winning_numbers
  (num_tickets : ℕ)
  (winning_number_value : ℕ)
  (total_amount_won : ℕ)
  (h1 : num_tickets = 3)
  (h2 : winning_number_value = 20)
  (h3 : total_amount_won = 300)
  (h4 : total_amount_won % winning_number_value = 0)
  (h5 : (total_amount_won / winning_number_value) % num_tickets = 0) :
  total_amount_won / winning_number_value / num_tickets = 5 :=
by sorry

end NUMINAMATH_CALUDE_lottery_winning_numbers_l627_62715


namespace NUMINAMATH_CALUDE_quadratic_completion_l627_62734

theorem quadratic_completion (b : ℝ) (p : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 1 = (x + p)^2 - 7/4) → 
  b = Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_l627_62734


namespace NUMINAMATH_CALUDE_josh_marbles_count_l627_62726

theorem josh_marbles_count (initial_marbles found_marbles : ℕ) 
  (h1 : initial_marbles = 21)
  (h2 : found_marbles = 7) :
  initial_marbles + found_marbles = 28 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_count_l627_62726


namespace NUMINAMATH_CALUDE_solution_set_l627_62727

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := 2 * a - b + 3

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  otimes 0.5 x > -2 ∧ otimes (2 * x) 5 > 3 * x + 1

-- State the theorem
theorem solution_set :
  ∀ x : ℝ, inequality_system x ↔ 3 < x ∧ x < 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_l627_62727


namespace NUMINAMATH_CALUDE_lucas_chocolate_candy_l627_62787

/-- The number of pieces of chocolate candy Lucas makes for each student on Monday -/
def pieces_per_student : ℕ := 4

/-- The number of students not coming to class this upcoming Monday -/
def absent_students : ℕ := 3

/-- The number of pieces of chocolate candy Lucas will make this upcoming Monday -/
def upcoming_monday_pieces : ℕ := 28

/-- The number of pieces of chocolate candy Lucas made last Monday -/
def last_monday_pieces : ℕ := pieces_per_student * (upcoming_monday_pieces / pieces_per_student + absent_students)

theorem lucas_chocolate_candy : last_monday_pieces = 40 := by sorry

end NUMINAMATH_CALUDE_lucas_chocolate_candy_l627_62787


namespace NUMINAMATH_CALUDE_star_3_7_l627_62771

-- Define the star operation
def star (a b : ℕ) : ℕ := a^2 + 3*a*b + b^2

-- Theorem statement
theorem star_3_7 : star 3 7 = 121 := by
  sorry

end NUMINAMATH_CALUDE_star_3_7_l627_62771


namespace NUMINAMATH_CALUDE_chores_per_week_l627_62741

theorem chores_per_week 
  (cookie_price : ℕ) 
  (cookies_per_pack : ℕ) 
  (budget : ℕ) 
  (cookies_per_chore : ℕ) 
  (weeks : ℕ) 
  (h1 : cookie_price = 3)
  (h2 : cookies_per_pack = 24)
  (h3 : budget = 15)
  (h4 : cookies_per_chore = 3)
  (h5 : weeks = 10)
  : (budget / cookie_price) * cookies_per_pack / weeks / cookies_per_chore = 4 := by
  sorry

end NUMINAMATH_CALUDE_chores_per_week_l627_62741


namespace NUMINAMATH_CALUDE_beta_max_success_ratio_l627_62761

/-- Represents a contestant's scores in a two-day math contest -/
structure ContestScores where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ

/-- The maximum possible two-day success ratio for Beta -/
def beta_max_ratio : ℚ := 407 / 600

theorem beta_max_success_ratio 
  (alpha : ContestScores)
  (beta : ContestScores)
  (h1 : alpha.day1_score = 180 ∧ alpha.day1_total = 350)
  (h2 : alpha.day2_score = 170 ∧ alpha.day2_total = 250)
  (h3 : beta.day1_score > 0 ∧ beta.day2_score > 0)
  (h4 : beta.day1_total + beta.day2_total = 600)
  (h5 : (beta.day1_score : ℚ) / beta.day1_total < (alpha.day1_score : ℚ) / alpha.day1_total)
  (h6 : (beta.day2_score : ℚ) / beta.day2_total < (alpha.day2_score : ℚ) / alpha.day2_total)
  (h7 : (alpha.day1_score + alpha.day2_score : ℚ) / (alpha.day1_total + alpha.day2_total) = 7 / 12) :
  (∀ b : ContestScores, 
    b.day1_score > 0 ∧ b.day2_score > 0 →
    b.day1_total + b.day2_total = 600 →
    (b.day1_score : ℚ) / b.day1_total < (alpha.day1_score : ℚ) / alpha.day1_total →
    (b.day2_score : ℚ) / b.day2_total < (alpha.day2_score : ℚ) / alpha.day2_total →
    (b.day1_score + b.day2_score : ℚ) / (b.day1_total + b.day2_total) ≤ beta_max_ratio) :=
by
  sorry

end NUMINAMATH_CALUDE_beta_max_success_ratio_l627_62761


namespace NUMINAMATH_CALUDE_number_equation_solution_l627_62778

theorem number_equation_solution : 
  ∃ x : ℝ, (3034 - (x / 20.04) = 2984) ∧ (x = 1002) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l627_62778


namespace NUMINAMATH_CALUDE_pumpkin_price_theorem_l627_62776

-- Define the prices of seeds
def tomato_price : ℚ := 1.5
def chili_price : ℚ := 0.9

-- Define the total spent and the number of packets bought
def total_spent : ℚ := 18
def pumpkin_packets : ℕ := 3
def tomato_packets : ℕ := 4
def chili_packets : ℕ := 5

-- Define the theorem
theorem pumpkin_price_theorem :
  ∃ (pumpkin_price : ℚ),
    pumpkin_price * pumpkin_packets +
    tomato_price * tomato_packets +
    chili_price * chili_packets = total_spent ∧
    pumpkin_price = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_price_theorem_l627_62776


namespace NUMINAMATH_CALUDE_line_circle_intersection_l627_62728

theorem line_circle_intersection (a : ℝ) :
  ∃ (x y : ℝ), (a * x - y + 2 * a = 0) ∧ (x^2 + y^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l627_62728


namespace NUMINAMATH_CALUDE_truncation_result_l627_62799

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  edges : ℕ
  convex : Bool

/-- Represents a truncated convex polyhedron -/
structure TruncatedPolyhedron where
  original : ConvexPolyhedron
  vertices : ℕ
  edges : ℕ
  truncated : Bool

/-- Function that performs truncation on a convex polyhedron -/
def truncate (p : ConvexPolyhedron) : TruncatedPolyhedron :=
  { original := p
  , vertices := 2 * p.edges
  , edges := 3 * p.edges
  , truncated := true }

/-- Theorem stating the result of truncating a specific convex polyhedron -/
theorem truncation_result :
  ∀ (p : ConvexPolyhedron),
  p.edges = 100 →
  p.convex = true →
  let tp := truncate p
  tp.vertices = 200 ∧ tp.edges = 300 := by
  sorry

end NUMINAMATH_CALUDE_truncation_result_l627_62799


namespace NUMINAMATH_CALUDE_willam_tax_is_960_l627_62702

/-- Represents the farm tax scenario in Mr. Willam's village -/
structure FarmTax where
  -- Total taxable land in the village
  total_taxable_land : ℝ
  -- Tax rate per unit of taxable land
  tax_rate : ℝ
  -- Percentage of Mr. Willam's taxable land
  willam_land_percentage : ℝ

/-- Calculates Mr. Willam's tax payment -/
def willam_tax_payment (ft : FarmTax) : ℝ :=
  ft.total_taxable_land * ft.tax_rate * ft.willam_land_percentage

/-- Theorem stating that Mr. Willam's tax payment is $960 -/
theorem willam_tax_is_960 (ft : FarmTax) 
    (h1 : ft.tax_rate * ft.total_taxable_land = 3840)
    (h2 : ft.willam_land_percentage = 0.25) : 
  willam_tax_payment ft = 960 := by
  sorry


end NUMINAMATH_CALUDE_willam_tax_is_960_l627_62702


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l627_62755

theorem quadratic_inequality_solution_set (m : ℝ) : 
  m > 2 → ∀ x : ℝ, x^2 - 2*x + m > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l627_62755


namespace NUMINAMATH_CALUDE_bruce_payment_l627_62775

/-- The amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1000 to the shopkeeper -/
theorem bruce_payment : total_amount 8 70 8 55 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l627_62775


namespace NUMINAMATH_CALUDE_cd_cost_calculation_l627_62764

/-- The cost of the CD that Ibrahim wants to buy -/
def cd_cost : ℝ := 19

/-- The cost of the MP3 player -/
def mp3_cost : ℝ := 120

/-- Ibrahim's savings -/
def savings : ℝ := 55

/-- Money given by Ibrahim's father -/
def father_contribution : ℝ := 20

/-- The amount Ibrahim lacks after his savings and father's contribution -/
def amount_lacking : ℝ := 64

theorem cd_cost_calculation :
  cd_cost = mp3_cost + cd_cost - (savings + father_contribution) - amount_lacking :=
by sorry

end NUMINAMATH_CALUDE_cd_cost_calculation_l627_62764


namespace NUMINAMATH_CALUDE_probability_four_twos_l627_62763

def num_dice : ℕ := 12
def num_sides : ℕ := 8
def target_number : ℕ := 2
def num_success : ℕ := 4

theorem probability_four_twos : 
  (Nat.choose num_dice num_success : ℚ) * (1 / num_sides : ℚ)^num_success * ((num_sides - 1) / num_sides : ℚ)^(num_dice - num_success) = 
  495 * (1 / 4096 : ℚ) * (5764801 / 16777216 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_probability_four_twos_l627_62763


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l627_62701

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 5 (1/3) + 1 →
  x^3 - 3*x^2 + 3*x - 6 = 0 ∧ 
  (∃ (a b c : ℤ), x^3 - 3*x^2 + 3*x - 6 = x^3 + a*x^2 + b*x + c) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l627_62701


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l627_62756

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 6 →
  g = -2*a - c - e →
  (2*a + b*Complex.I) + (c + 2*d*Complex.I) + (e + f*Complex.I) + (g + 2*h*Complex.I) = 8*Complex.I →
  d + f + h = 3/2 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l627_62756


namespace NUMINAMATH_CALUDE_sunday_no_arguments_l627_62730

/-- Probability of a spouse arguing with their mother-in-law -/
def p_argue_with_mil : ℚ := 2/3

/-- Probability of siding with own mother in case of conflict -/
def p_side_with_mother : ℚ := 1/2

/-- Probability of no arguments between spouses on a Sunday -/
def p_no_arguments : ℚ := 4/9

theorem sunday_no_arguments : 
  p_no_arguments = 1 - (2 * p_argue_with_mil * p_side_with_mother - (p_argue_with_mil * p_side_with_mother)^2) := by
  sorry

end NUMINAMATH_CALUDE_sunday_no_arguments_l627_62730


namespace NUMINAMATH_CALUDE_ice_cream_line_count_l627_62769

theorem ice_cream_line_count (between : ℕ) (h : between = 5) : 
  between + 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_line_count_l627_62769


namespace NUMINAMATH_CALUDE_no_integer_solution_l627_62779

theorem no_integer_solution : ¬ ∃ (x : ℤ), x^2 * 7 = 2^14 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l627_62779


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l627_62786

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l627_62786


namespace NUMINAMATH_CALUDE_codecracker_combinations_l627_62737

/-- The number of colors available in the CodeCracker game -/
def num_colors : ℕ := 8

/-- The number of slots in a CodeCracker code -/
def num_slots : ℕ := 4

/-- Theorem stating the total number of possible codes in CodeCracker -/
theorem codecracker_combinations : (num_colors ^ num_slots : ℕ) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_codecracker_combinations_l627_62737


namespace NUMINAMATH_CALUDE_N_divisible_by_7_and_9_l627_62759

def N : ℕ := 1234567765432  -- This is the octal representation as a decimal number

theorem N_divisible_by_7_and_9 : 
  7 ∣ N ∧ 9 ∣ N :=
sorry

end NUMINAMATH_CALUDE_N_divisible_by_7_and_9_l627_62759


namespace NUMINAMATH_CALUDE_playground_width_l627_62791

theorem playground_width (area : ℝ) (length : ℝ) (h1 : area = 143.2) (h2 : length = 4) :
  area / length = 35.8 := by
  sorry

end NUMINAMATH_CALUDE_playground_width_l627_62791


namespace NUMINAMATH_CALUDE_chess_group_players_l627_62745

theorem chess_group_players (n : ℕ) : 
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (game : ℕ), game < n * (n - 1) / 2) →
  (∀ (game : ℕ), game < n * (n - 1) / 2 → ∃! (i j : ℕ), i < n ∧ j < n ∧ i ≠ j) →
  n * (n - 1) / 2 = 105 →
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_chess_group_players_l627_62745


namespace NUMINAMATH_CALUDE_husband_additional_payment_l627_62770

/-- Calculates the additional amount the husband needs to pay to split expenses equally for the house help -/
theorem husband_additional_payment (salary : ℝ) (medical_cost : ℝ) 
  (h1 : salary = 160)
  (h2 : medical_cost = 128)
  (h3 : salary ≥ medical_cost / 2) : 
  salary / 2 - medical_cost / 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_husband_additional_payment_l627_62770


namespace NUMINAMATH_CALUDE_tea_price_calculation_l627_62721

theorem tea_price_calculation (coffee_customers : ℕ) (tea_customers : ℕ) (coffee_price : ℚ) (total_revenue : ℚ) :
  coffee_customers = 7 →
  tea_customers = 8 →
  coffee_price = 5 →
  total_revenue = 67 →
  ∃ tea_price : ℚ, tea_price = 4 ∧ coffee_customers * coffee_price + tea_customers * tea_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_tea_price_calculation_l627_62721


namespace NUMINAMATH_CALUDE_equilateral_triangle_exists_l627_62705

-- Define a type for colors
inductive Color
| Black
| White

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define an equilateral triangle
structure EquilateralTriangle where
  p1 : Point
  p2 : Point
  p3 : Point
  eq_sides : distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1

-- Theorem statement
theorem equilateral_triangle_exists :
  ∃ (t : EquilateralTriangle),
    (distance t.p1 t.p2 = 1 ∨ distance t.p1 t.p2 = Real.sqrt 3) ∧
    (coloring t.p1 = coloring t.p2 ∧ coloring t.p2 = coloring t.p3) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_exists_l627_62705


namespace NUMINAMATH_CALUDE_chess_club_committee_probability_l627_62706

def total_members : ℕ := 27
def boys : ℕ := 15
def girls : ℕ := 12
def committee_size : ℕ := 5

theorem chess_club_committee_probability :
  let total_committees := Nat.choose total_members committee_size
  let all_boys_committees := Nat.choose boys committee_size
  let all_girls_committees := Nat.choose girls committee_size
  let favorable_committees := total_committees - (all_boys_committees + all_girls_committees)
  (favorable_committees : ℚ) / total_committees = 76935 / 80730 := by sorry

end NUMINAMATH_CALUDE_chess_club_committee_probability_l627_62706


namespace NUMINAMATH_CALUDE_snow_probability_l627_62739

theorem snow_probability (p1 p2 p3 : ℚ) (h1 : p1 = 1/4) (h2 : p2 = 1/2) (h3 : p3 = 1/3) :
  1 - (1 - p1)^2 * (1 - p2)^3 * (1 - p3)^2 = 31/32 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l627_62739


namespace NUMINAMATH_CALUDE_ellipse_m_value_l627_62710

/-- An ellipse with semi-major axis a, semi-minor axis b, and focal distance c. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_relation : a^2 - b^2 = c^2

/-- The given ellipse with a = 5 and left focus at (-4, 0). -/
def given_ellipse (m : ℝ) : Ellipse :=
  { a := 5
    b := m
    c := 4
    h_positive := by sorry
    h_relation := by sorry }

/-- Theorem stating that m = 3 for the given ellipse. -/
theorem ellipse_m_value :
  ∀ m > 0, (given_ellipse m).b = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l627_62710


namespace NUMINAMATH_CALUDE_polynomial_relation_l627_62711

theorem polynomial_relation (r : ℝ) : r^3 - 2*r + 1 = 0 → r^6 - 4*r^4 + 4*r^2 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_relation_l627_62711


namespace NUMINAMATH_CALUDE_opposite_sides_of_y_axis_l627_62785

/-- Given points A and B on opposite sides of the y-axis, with B on the right side,
    prove that the x-coordinate of A is negative. -/
theorem opposite_sides_of_y_axis (a : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (a, 1) ∧ B = (2, a) ∧ 
   (A.1 < 0 ∧ B.1 > 0) ∧ -- A and B are on opposite sides of the y-axis
   B.1 > 0) →              -- B is on the right side of the y-axis
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_opposite_sides_of_y_axis_l627_62785


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l627_62742

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l627_62742


namespace NUMINAMATH_CALUDE_min_value_and_range_l627_62735

theorem min_value_and_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b - a*b = 0 → x + 2*y ≤ a + 2*b) ∧ y > 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_range_l627_62735


namespace NUMINAMATH_CALUDE_total_garbage_accumulation_l627_62798

/-- Represents the garbage accumulation problem in Daniel's neighborhood --/
def garbage_accumulation (collection_days_per_week : ℕ) (kg_per_collection : ℝ) (weeks : ℕ) (reduction_factor : ℝ) : ℝ :=
  let week1_accumulation := collection_days_per_week * kg_per_collection
  let week2_accumulation := week1_accumulation * reduction_factor
  week1_accumulation + week2_accumulation

/-- Theorem stating the total garbage accumulated over two weeks --/
theorem total_garbage_accumulation :
  garbage_accumulation 3 200 2 0.5 = 900 := by
  sorry

#eval garbage_accumulation 3 200 2 0.5

end NUMINAMATH_CALUDE_total_garbage_accumulation_l627_62798


namespace NUMINAMATH_CALUDE_medium_supermarkets_sample_l627_62719

/-- Represents the number of supermarkets to be sampled -/
def sample_size : ℕ := 200

/-- Represents the number of large supermarkets -/
def large_supermarkets : ℕ := 200

/-- Represents the number of medium supermarkets -/
def medium_supermarkets : ℕ := 400

/-- Represents the number of small supermarkets -/
def small_supermarkets : ℕ := 1400

/-- Represents the total number of supermarkets -/
def total_supermarkets : ℕ := large_supermarkets + medium_supermarkets + small_supermarkets

/-- Theorem stating that the number of medium supermarkets to be sampled is 40 -/
theorem medium_supermarkets_sample :
  (sample_size : ℚ) * medium_supermarkets / total_supermarkets = 40 := by
  sorry

end NUMINAMATH_CALUDE_medium_supermarkets_sample_l627_62719


namespace NUMINAMATH_CALUDE_line_slope_45_degrees_l627_62790

theorem line_slope_45_degrees (m : ℝ) : 
  let P : ℝ × ℝ := (-2, m)
  let Q : ℝ × ℝ := (m, 4)
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_45_degrees_l627_62790


namespace NUMINAMATH_CALUDE_c_work_days_l627_62714

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 8
def work_rate_B : ℚ := 1 / 16
def work_rate_ABC : ℚ := 1 / 4

-- Define C's work rate as a function of x (days C needs to complete the work)
def work_rate_C (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem c_work_days :
  ∃ x : ℚ, x = 16 ∧ work_rate_A + work_rate_B + work_rate_C x = work_rate_ABC :=
sorry

end NUMINAMATH_CALUDE_c_work_days_l627_62714


namespace NUMINAMATH_CALUDE_wilsons_theorem_l627_62795

theorem wilsons_theorem (p : ℕ) (h : p > 1) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1) : ℤ) % p = p - 1 := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l627_62795


namespace NUMINAMATH_CALUDE_parallel_transitivity_parallel_planes_imply_parallel_line_perpendicular_implies_parallel_perpendicular_planes_imply_parallel_line_l627_62782

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line) (α β : Plane)

-- State the conditions
variable (h1 : ¬in_plane m α)
variable (h2 : ¬in_plane m β)
variable (h3 : ¬in_plane n α)
variable (h4 : ¬in_plane n β)

-- State the theorems to be proved
theorem parallel_transitivity 
  (h5 : parallel m n) (h6 : parallel_plane n α) : 
  parallel_plane m α := by sorry

theorem parallel_planes_imply_parallel_line 
  (h5 : parallel_plane m β) (h6 : parallel_planes α β) : 
  parallel_plane m α := by sorry

theorem perpendicular_implies_parallel 
  (h5 : perpendicular m n) (h6 : perpendicular_plane n α) : 
  parallel_plane m α := by sorry

theorem perpendicular_planes_imply_parallel_line 
  (h5 : perpendicular_plane m β) (h6 : perpendicular_planes α β) : 
  parallel_plane m α := by sorry

end NUMINAMATH_CALUDE_parallel_transitivity_parallel_planes_imply_parallel_line_perpendicular_implies_parallel_perpendicular_planes_imply_parallel_line_l627_62782


namespace NUMINAMATH_CALUDE_aron_dusting_time_l627_62765

/-- Represents the cleaning schedule and durations for Aron --/
structure CleaningSchedule where
  vacuum_duration : ℕ  -- Minutes spent vacuuming per day
  vacuum_frequency : ℕ  -- Days per week spent vacuuming
  dust_frequency : ℕ  -- Days per week spent dusting
  total_cleaning_time : ℕ  -- Total minutes spent cleaning per week

/-- Calculates the time spent dusting per day given a cleaning schedule --/
def dusting_time_per_day (schedule : CleaningSchedule) : ℕ :=
  let total_vacuum_time := schedule.vacuum_duration * schedule.vacuum_frequency
  let total_dust_time := schedule.total_cleaning_time - total_vacuum_time
  total_dust_time / schedule.dust_frequency

/-- Theorem stating that Aron spends 20 minutes dusting each day --/
theorem aron_dusting_time (schedule : CleaningSchedule) 
    (h1 : schedule.vacuum_duration = 30)
    (h2 : schedule.vacuum_frequency = 3)
    (h3 : schedule.dust_frequency = 2)
    (h4 : schedule.total_cleaning_time = 130) :
  dusting_time_per_day schedule = 20 := by
  sorry

end NUMINAMATH_CALUDE_aron_dusting_time_l627_62765


namespace NUMINAMATH_CALUDE_power_of_two_equality_l627_62703

theorem power_of_two_equality (n : ℕ) : 2^n = 2 * 16^2 * 64^3 → n = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l627_62703


namespace NUMINAMATH_CALUDE_new_savings_amount_l627_62723

def monthly_salary : ℕ := 6500
def initial_savings_rate : ℚ := 1/5
def expense_increase_rate : ℚ := 1/5

theorem new_savings_amount :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_expenses * expense_increase_rate
  let new_expenses := initial_expenses + expense_increase
  let new_savings := monthly_salary - new_expenses
  new_savings = 260 := by sorry

end NUMINAMATH_CALUDE_new_savings_amount_l627_62723


namespace NUMINAMATH_CALUDE_candidate_X_votes_and_result_l627_62707

-- Define the number of votes for each candidate
def votes_Z : ℕ := 25000
def votes_Y : ℕ := (3 * votes_Z) / 5
def votes_X : ℕ := (3 * votes_Y) / 2

-- Define the winning threshold
def winning_threshold : ℕ := 30000

-- Theorem to prove
theorem candidate_X_votes_and_result : 
  votes_X = 22500 ∧ votes_X < winning_threshold :=
by sorry

end NUMINAMATH_CALUDE_candidate_X_votes_and_result_l627_62707


namespace NUMINAMATH_CALUDE_gloin_tells_truth_l627_62796

/-- Represents the type of dwarf: either a knight or a liar -/
inductive DwarfType
  | Knight
  | Liar

/-- Represents a dwarf with their position and type -/
structure Dwarf :=
  (position : Nat)
  (type : DwarfType)

/-- The statement made by a dwarf -/
def statement (d : Dwarf) (line : List Dwarf) : Prop :=
  match d.position with
  | 10 => ∃ (right : Dwarf), right.position > d.position ∧ right.type = DwarfType.Knight
  | _ => ∃ (left : Dwarf), left.position < d.position ∧ left.type = DwarfType.Knight

/-- The main theorem -/
theorem gloin_tells_truth 
  (line : List Dwarf) 
  (h_count : line.length = 10)
  (h_knight : ∃ d ∈ line, d.type = DwarfType.Knight)
  (h_statements : ∀ d ∈ line, d.position ≠ 10 → 
    (d.type = DwarfType.Knight ↔ statement d line))
  (h_gloin : ∃ gloin ∈ line, gloin.position = 10)
  : ∃ gloin ∈ line, gloin.position = 10 ∧ gloin.type = DwarfType.Knight :=
sorry

end NUMINAMATH_CALUDE_gloin_tells_truth_l627_62796


namespace NUMINAMATH_CALUDE_train_distance_problem_l627_62768

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : d = 65) :
  let t := d / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d1 + d2 = 585 := by
sorry

end NUMINAMATH_CALUDE_train_distance_problem_l627_62768


namespace NUMINAMATH_CALUDE_complex_expression_equality_l627_62788

theorem complex_expression_equality (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  4 * (x^y * (7^y * 24^x)) / (x*y) + 5 * (x * (13^y * 15^x)) - 2 * (y * (6^x * 28^y)) + 7 * (x*y * (3^x * 19^y)) / (x+y) = 11948716.8 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l627_62788


namespace NUMINAMATH_CALUDE_sixth_term_value_l627_62725

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem sixth_term_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l627_62725


namespace NUMINAMATH_CALUDE_combine_squared_binomial_simplify_given_equation_solve_system_of_equations_l627_62760

-- Problem 1
theorem combine_squared_binomial (m n : ℝ) :
  3 * (m - n)^2 - 4 * (m - n)^2 + 3 * (m - n)^2 = 2 * (m - n)^2 :=
sorry

-- Problem 2
theorem simplify_given_equation (x y : ℝ) (h : x^2 + 2*y = 4) :
  3*x^2 + 6*y - 2 = 10 :=
sorry

-- Problem 3
theorem solve_system_of_equations (x y : ℝ) 
  (h1 : x^2 + x*y = 2) (h2 : 2*y^2 + 3*x*y = 5) :
  2*x^2 + 11*x*y + 6*y^2 = 19 :=
sorry

end NUMINAMATH_CALUDE_combine_squared_binomial_simplify_given_equation_solve_system_of_equations_l627_62760


namespace NUMINAMATH_CALUDE_min_value_expression_l627_62772

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  ∃ (min : ℝ), min = 60 ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' * b' * c' = 27 → 
    a'^2 + 6*a'*b' + 9*b'^2 + 3*c'^2 ≥ min) ∧
  (a^2 + 6*a*b + 9*b^2 + 3*c^2 = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l627_62772


namespace NUMINAMATH_CALUDE_f_is_odd_ellipse_y_axis_iff_l627_62754

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^2))

-- Theorem 1: f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

-- Define the ellipse equation
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ ∀ x y : ℝ, m * x^2 + n * y^2 = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1

-- Theorem 2: Necessary and sufficient condition for ellipse with foci on y-axis
theorem ellipse_y_axis_iff (m n : ℝ) : 
  is_ellipse_y_axis m n ↔ m > n ∧ n > 0 := by sorry

end NUMINAMATH_CALUDE_f_is_odd_ellipse_y_axis_iff_l627_62754


namespace NUMINAMATH_CALUDE_fraction_problem_l627_62700

theorem fraction_problem (x : ℚ) :
  (x / (2 * x + 11) = 3 / 4) → x = -33 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l627_62700


namespace NUMINAMATH_CALUDE_least_divisible_by_10_to_15_divided_by_26_l627_62774

theorem least_divisible_by_10_to_15_divided_by_26 :
  let j := Nat.lcm 10 (Nat.lcm 11 (Nat.lcm 12 (Nat.lcm 13 (Nat.lcm 14 15))))
  ∀ k : ℕ, (∀ i ∈ Finset.range 6, k % (i + 10) = 0) → k ≥ j
  → j / 26 = 2310 := by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_10_to_15_divided_by_26_l627_62774


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l627_62746

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  y * (x + y) + (x + y) * (x - y) = x^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) (h1 : m ≠ -1) (h2 : m^2 + 2*m + 1 ≠ 0) :
  ((2*m + 1) / (m + 1) + m - 1) / ((m + 2) / (m^2 + 2*m + 1)) = m^2 + m := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l627_62746


namespace NUMINAMATH_CALUDE_circle_center_sum_l627_62722

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 8*y + 13

/-- The center of a circle -/
def CircleCenter (h k : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 13) / 2

theorem circle_center_sum :
  ∀ h k : ℝ, CircleCenter h k CircleEquation → h + k = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l627_62722


namespace NUMINAMATH_CALUDE_square_division_perimeter_l627_62736

/-- Given a square with perimeter 160 units, when divided into two congruent rectangles
    horizontally and one of those rectangles is further divided into two congruent rectangles
    vertically, the perimeter of one of the smaller rectangles is 80 units. -/
theorem square_division_perimeter :
  ∀ (s : ℝ),
  s > 0 →
  4 * s = 160 →
  let horizontal_rectangle_width := s
  let horizontal_rectangle_height := s / 2
  let vertical_rectangle_width := s / 2
  let vertical_rectangle_height := s / 2
  2 * (vertical_rectangle_width + vertical_rectangle_height) = 80 :=
by
  sorry

#check square_division_perimeter

end NUMINAMATH_CALUDE_square_division_perimeter_l627_62736


namespace NUMINAMATH_CALUDE_percentage_problem_l627_62792

theorem percentage_problem : 
  ∀ x : ℝ, (120 : ℝ) = 1.5 * x → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l627_62792


namespace NUMINAMATH_CALUDE_pie_eating_contest_ratio_l627_62773

theorem pie_eating_contest_ratio (bill_pies sierra_pies adam_pies : ℕ) :
  adam_pies = bill_pies + 3 →
  sierra_pies = 12 →
  bill_pies + adam_pies + sierra_pies = 27 →
  sierra_pies / bill_pies = 2 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_ratio_l627_62773


namespace NUMINAMATH_CALUDE_two_digit_multiple_plus_two_l627_62752

theorem two_digit_multiple_plus_two : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 3 * 4 * 5 * k + 2 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_two_digit_multiple_plus_two_l627_62752


namespace NUMINAMATH_CALUDE_simplify_polynomial_l627_62732

theorem simplify_polynomial (x : ℝ) : 
  2 * x^2 * (4 * x^3 - 3 * x + 5) - 4 * (x^3 - x^2 + 3 * x - 8) = 
  8 * x^5 - 10 * x^3 + 14 * x^2 - 12 * x + 32 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l627_62732


namespace NUMINAMATH_CALUDE_palindrome_product_sum_l627_62789

/-- A positive three-digit palindrome is a number between 100 and 999 (inclusive) that reads the same forwards and backwards. -/
def IsPositiveThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ ((n / 10) % 10 = (n % 100) / 10)

/-- The theorem stating that if there exist two positive three-digit palindromes whose product is 445,545, then their sum is 1436. -/
theorem palindrome_product_sum : 
  ∃ (a b : ℕ), IsPositiveThreeDigitPalindrome a ∧ 
                IsPositiveThreeDigitPalindrome b ∧ 
                a * b = 445545 → 
                a + b = 1436 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_product_sum_l627_62789


namespace NUMINAMATH_CALUDE_exists_close_to_integer_l627_62753

theorem exists_close_to_integer (a : ℝ) (n : ℕ) (ha : a > 0) (hn : n > 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ ∃ m : ℤ, |k * a - m| ≤ 1 / n := by
  sorry

end NUMINAMATH_CALUDE_exists_close_to_integer_l627_62753


namespace NUMINAMATH_CALUDE_spherical_segment_volume_ratio_l627_62750

theorem spherical_segment_volume_ratio (α : ℝ) :
  let R : ℝ := 1  -- Assume unit sphere for simplicity
  let V_sphere : ℝ := (4 / 3) * Real.pi * R^3
  let H : ℝ := 2 * R * Real.sin (α / 4)^2
  let V_seg : ℝ := Real.pi * H^2 * (R - H / 3)
  V_seg / V_sphere = Real.sin (α / 4)^4 * (2 + Real.cos (α / 2)) :=
by sorry

end NUMINAMATH_CALUDE_spherical_segment_volume_ratio_l627_62750


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l627_62718

def A : Set ℝ := {x | 2 * x - 1 > 0}
def B : Set ℝ := {x | x * (x - 2) < 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1/2 < x ∧ x < 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l627_62718


namespace NUMINAMATH_CALUDE_road_division_l627_62744

theorem road_division (a b c : ℝ) : 
  a + b + c = 28 →
  a > 0 → b > 0 → c > 0 →
  a ≠ b → b ≠ c → a ≠ c →
  (a + b + c / 2) - a / 2 = 16 →
  b = 4 :=
by sorry

end NUMINAMATH_CALUDE_road_division_l627_62744


namespace NUMINAMATH_CALUDE_prob_same_suit_60_card_deck_l627_62748

/-- A deck of cards with a specified number of ranks and suits. -/
structure Deck :=
  (num_ranks : ℕ)
  (num_suits : ℕ)

/-- The probability of drawing two cards of the same suit from a deck. -/
def prob_same_suit (d : Deck) : ℚ :=
  if d.num_ranks * d.num_suits = 0 then 0
  else (d.num_ranks - 1) / (d.num_ranks * d.num_suits - 1)

/-- Theorem stating the probability of drawing two cards of the same suit
    from a 60-card deck with 15 ranks and 4 suits. -/
theorem prob_same_suit_60_card_deck :
  prob_same_suit ⟨15, 4⟩ = 14 / 59 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_suit_60_card_deck_l627_62748


namespace NUMINAMATH_CALUDE_inverse_proportion_point_relation_l627_62743

theorem inverse_proportion_point_relation :
  ∀ (y₁ y₂ y₃ : ℝ),
  y₁ = 3 / (-5) →
  y₂ = 3 / (-3) →
  y₃ = 3 / 2 →
  y₂ < y₁ ∧ y₁ < y₃ := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_relation_l627_62743


namespace NUMINAMATH_CALUDE_unique_solution_abc_l627_62767

/-- Represents a base-7 number with two digits --/
def Base7TwoDigit (a b : ℕ) : ℕ := 7 * a + b

/-- Represents a base-7 number with one digit --/
def Base7OneDigit (c : ℕ) : ℕ := c

/-- Represents a base-7 number with two digits, where the first digit is 'c' and the second is 0 --/
def Base7TwoDigitWithZero (c : ℕ) : ℕ := 7 * c

theorem unique_solution_abc (A B C : ℕ) :
  (0 < A ∧ A < 7) →
  (0 < B ∧ B < 7) →
  (0 < C ∧ C < 7) →
  Base7TwoDigit A B + Base7OneDigit C = Base7TwoDigitWithZero C →
  Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit C C →
  A = 3 ∧ B = 2 ∧ C = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_abc_l627_62767


namespace NUMINAMATH_CALUDE_kerosene_cost_l627_62777

/-- The cost of a pound of rice in dollars -/
def rice_cost : ℚ := 33 / 100

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

theorem kerosene_cost :
  ∀ (egg_cost : ℚ) (kerosene_half_liter_cost : ℚ),
    egg_cost * dozen = rice_cost →  -- A dozen eggs cost as much as a pound of rice
    kerosene_half_liter_cost = egg_cost * 8 →  -- A half-liter of kerosene costs as much as 8 eggs
    (2 * kerosene_half_liter_cost * cents_per_dollar : ℚ) = 44 :=
by sorry

end NUMINAMATH_CALUDE_kerosene_cost_l627_62777


namespace NUMINAMATH_CALUDE_cosine_tangent_equality_l627_62708

theorem cosine_tangent_equality : 4 * Real.cos (10 * π / 180) - Real.tan (80 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_tangent_equality_l627_62708


namespace NUMINAMATH_CALUDE_circle_area_ratio_l627_62704

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0) :
  (60 / 360 * (2 * Real.pi * C) = 40 / 360 * (2 * Real.pi * D)) →
  (Real.pi * C^2) / (Real.pi * D^2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l627_62704


namespace NUMINAMATH_CALUDE_equation_root_condition_l627_62712

theorem equation_root_condition (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = a * x + 1) ∧ 
  (∀ y : ℝ, y > 0 → |y| ≠ a * y + 1) → 
  a > -1 := by sorry

end NUMINAMATH_CALUDE_equation_root_condition_l627_62712


namespace NUMINAMATH_CALUDE_ratio_problem_l627_62740

theorem ratio_problem (c d : ℚ) : 
  (c / d = 5) → (c = 18 - 7 * d) → (d = 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l627_62740


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_l627_62731

/-- A polynomial of degree 2 with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Represents a factorization of a quadratic polynomial into two linear factors -/
structure Factorization where
  p : ℤ
  q : ℤ

/-- Checks if a factorization is valid for a given quadratic polynomial -/
def isValidFactorization (poly : QuadraticPolynomial) (fac : Factorization) : Prop :=
  poly.a = 1 ∧ poly.b = fac.p + fac.q ∧ poly.c = fac.p * fac.q

/-- Theorem stating that 259 is the smallest positive integer b for which
    x^2 + bx + 2008 can be factored into a product of two polynomials
    with integer coefficients -/
theorem smallest_factorizable_b :
  ∀ b : ℤ, b > 0 →
  (∃ fac : Factorization, isValidFactorization ⟨1, b, 2008⟩ fac) →
  b ≥ 259 :=
sorry

end NUMINAMATH_CALUDE_smallest_factorizable_b_l627_62731


namespace NUMINAMATH_CALUDE_least_11_heavy_three_digit_l627_62738

def is_11_heavy (n : ℕ) : Prop := n % 11 > 7

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_11_heavy_three_digit : 
  (∀ n : ℕ, is_three_digit n → is_11_heavy n → 107 ≤ n) ∧ 
  is_three_digit 107 ∧ 
  is_11_heavy 107 :=
sorry

end NUMINAMATH_CALUDE_least_11_heavy_three_digit_l627_62738


namespace NUMINAMATH_CALUDE_shortest_side_is_eight_l627_62716

/-- Represents a rectangular solid with sides in geometric progression -/
structure GeometricSolid where
  b : ℝ
  s : ℝ
  volume : ℝ
  surface_area : ℝ
  volume_eq : volume = b^3 / s
  surface_area_eq : surface_area = 2 * (b^2 / s + b^2 * s + b^2)

/-- The shortest side length of a geometric solid with given properties is 8 -/
theorem shortest_side_is_eight (solid : GeometricSolid)
  (h_volume : solid.volume = 512)
  (h_surface_area : solid.surface_area = 384) :
  min (solid.b / solid.s) (min solid.b (solid.b * solid.s)) = 8 := by
  sorry

#check shortest_side_is_eight

end NUMINAMATH_CALUDE_shortest_side_is_eight_l627_62716


namespace NUMINAMATH_CALUDE_log_cube_l627_62781

theorem log_cube (x : ℝ) (h : Real.log x / Real.log 3 = 5) : 
  Real.log (x^3) / Real.log 3 = 15 := by
sorry

end NUMINAMATH_CALUDE_log_cube_l627_62781


namespace NUMINAMATH_CALUDE_clearance_sale_earnings_l627_62713

/-- Calculates the total earnings from a clearance sale of winter jackets --/
theorem clearance_sale_earnings 
  (total_jackets : ℕ)
  (price_before_noon : ℚ)
  (price_after_noon : ℚ)
  (jackets_sold_after_noon : ℕ)
  (h1 : total_jackets = 214)
  (h2 : price_before_noon = 31.95)
  (h3 : price_after_noon = 18.95)
  (h4 : jackets_sold_after_noon = 133) :
  (total_jackets - jackets_sold_after_noon) * price_before_noon +
  jackets_sold_after_noon * price_after_noon = 5107.30 := by
  sorry


end NUMINAMATH_CALUDE_clearance_sale_earnings_l627_62713


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_max_area_divided_pen_l627_62766

/-- The maximum area of a rectangular pen given a fixed perimeter --/
theorem max_area_rectangular_pen (perimeter : ℝ) (area : ℝ) : 
  perimeter = 60 →
  area ≤ 225 ∧ 
  (∃ width height : ℝ, width > 0 ∧ height > 0 ∧ 2 * (width + height) = perimeter ∧ width * height = area) →
  (∀ width height : ℝ, width > 0 → height > 0 → 2 * (width + height) = perimeter → width * height ≤ 225) :=
by sorry

/-- The maximum area remains the same when divided into two equal sections --/
theorem max_area_divided_pen (perimeter : ℝ) (area : ℝ) (width height : ℝ) :
  perimeter = 60 →
  width > 0 →
  height > 0 →
  2 * (width + height) = perimeter →
  width * height = 225 →
  ∃ new_height : ℝ, new_height > 0 ∧ 2 * (width + new_height) = perimeter ∧ width * new_height = 225 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_max_area_divided_pen_l627_62766


namespace NUMINAMATH_CALUDE_min_tiles_cover_rect_l627_62733

/-- The side length of a square tile in inches -/
def tile_side : ℕ := 6

/-- The length of the rectangular region in feet -/
def rect_length : ℕ := 6

/-- The width of the rectangular region in feet -/
def rect_width : ℕ := 3

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The minimum number of tiles needed to cover the rectangular region -/
def min_tiles : ℕ := 72

theorem min_tiles_cover_rect : 
  (rect_length * inches_per_foot) * (rect_width * inches_per_foot) = 
  min_tiles * (tile_side * tile_side) :=
by sorry

end NUMINAMATH_CALUDE_min_tiles_cover_rect_l627_62733


namespace NUMINAMATH_CALUDE_remainder_theorem_l627_62758

theorem remainder_theorem (x y u v : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + 3 * u * y + 2) % y = (v + 2) % y := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l627_62758


namespace NUMINAMATH_CALUDE_obtuse_triangle_one_obtuse_angle_equilateral_triangle_60_degree_angles_l627_62729

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : (angles 0) + (angles 1) + (angles 2) = 180

-- Define an obtuse triangle
def ObtuseTriangle (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i > 90

-- Define an equilateral triangle
def EquilateralTriangle (t : Triangle) : Prop :=
  t.angles 0 = t.angles 1 ∧ t.angles 1 = t.angles 2

theorem obtuse_triangle_one_obtuse_angle (t : Triangle) (h : ObtuseTriangle t) :
  ∃! i : Fin 3, t.angles i > 90 :=
sorry

theorem equilateral_triangle_60_degree_angles (t : Triangle) (h : EquilateralTriangle t) :
  ∀ i : Fin 3, t.angles i = 60 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_one_obtuse_angle_equilateral_triangle_60_degree_angles_l627_62729


namespace NUMINAMATH_CALUDE_time_period_is_three_years_l627_62780

/-- Represents the simple interest calculation and conditions --/
def simple_interest_problem (t : ℝ) : Prop :=
  let initial_deposit : ℝ := 9000
  let final_amount : ℝ := 10200
  let higher_rate_amount : ℝ := 10740
  ∃ r : ℝ,
    -- Condition for the original interest rate
    initial_deposit * (1 + r * t / 100) = final_amount ∧
    -- Condition for the interest rate 2% higher
    initial_deposit * (1 + (r + 2) * t / 100) = higher_rate_amount

/-- The theorem stating that the time period is 3 years --/
theorem time_period_is_three_years :
  simple_interest_problem 3 := by sorry

end NUMINAMATH_CALUDE_time_period_is_three_years_l627_62780


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l627_62762

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ 4 + 2 * a^(x - 1)
  f 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l627_62762


namespace NUMINAMATH_CALUDE_triangle_area_implies_cd_one_l627_62751

theorem triangle_area_implies_cd_one (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h_line : ∀ x y, 2*c*x + 3*d*y = 12 → x ≥ 0 ∧ y ≥ 0)
  (h_area : (1/2) * (6/c) * (4/d) = 12) : c * d = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_implies_cd_one_l627_62751


namespace NUMINAMATH_CALUDE_inequality_proof_l627_62797

theorem inequality_proof (a b c : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n ≥ 2) (habc : a * b * c = 1) :
  (a / (b + c)^(1 / n : ℝ)) + (b / (c + a)^(1 / n : ℝ)) + (c / (a + b)^(1 / n : ℝ)) ≥ 3 / (2^(1 / n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l627_62797


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l627_62793

-- Define the set M
def M : Set ℝ := {0, 1, 2}

-- Define the set N
def N : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Theorem statement
theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l627_62793


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l627_62724

theorem arithmetic_sequence_length
  (a : ℤ)  -- First term
  (l : ℤ)  -- Last term
  (d : ℤ)  -- Common difference
  (h1 : a = -22)
  (h2 : l = 50)
  (h3 : d = 7)
  : (l - a) / d + 1 = 11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l627_62724


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l627_62757

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l627_62757


namespace NUMINAMATH_CALUDE_large_lemonade_price_l627_62709

/-- Represents the price and sales data for Tonya's lemonade stand --/
structure LemonadeStand where
  small_price : ℝ
  medium_price : ℝ
  large_price : ℝ
  total_sales : ℝ
  small_sales : ℝ
  medium_sales : ℝ
  large_cups_sold : ℕ

/-- Theorem stating that the price of a large cup of lemonade is $3 --/
theorem large_lemonade_price (stand : LemonadeStand)
  (h1 : stand.small_price = 1)
  (h2 : stand.medium_price = 2)
  (h3 : stand.total_sales = 50)
  (h4 : stand.small_sales = 11)
  (h5 : stand.medium_sales = 24)
  (h6 : stand.large_cups_sold = 5)
  (h7 : stand.total_sales = stand.small_sales + stand.medium_sales + stand.large_price * stand.large_cups_sold) :
  stand.large_price = 3 := by
  sorry


end NUMINAMATH_CALUDE_large_lemonade_price_l627_62709


namespace NUMINAMATH_CALUDE_team_leader_selection_l627_62784

theorem team_leader_selection (n : ℕ) (h : n = 5) : n * (n - 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_team_leader_selection_l627_62784
