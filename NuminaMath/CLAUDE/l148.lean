import Mathlib

namespace NUMINAMATH_CALUDE_second_batch_average_l148_14852

theorem second_batch_average (n1 n2 n3 : ℕ) (a1 a2 a3 overall_avg : ℝ) :
  n1 = 40 →
  n2 = 50 →
  n3 = 60 →
  a1 = 45 →
  a3 = 65 →
  overall_avg = 56.333333333333336 →
  (n1 * a1 + n2 * a2 + n3 * a3) / (n1 + n2 + n3) = overall_avg →
  a2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_second_batch_average_l148_14852


namespace NUMINAMATH_CALUDE_jerry_medical_bills_l148_14832

/-- The amount Jerry is claiming for medical bills -/
def medical_bills : ℝ := sorry

/-- Jerry's annual salary -/
def annual_salary : ℝ := 50000

/-- Number of years of lost salary -/
def years_of_lost_salary : ℕ := 30

/-- Total lost salary -/
def total_lost_salary : ℝ := annual_salary * years_of_lost_salary

/-- Punitive damages multiplier -/
def punitive_multiplier : ℕ := 3

/-- Percentage of claim Jerry receives -/
def claim_percentage : ℝ := 0.8

/-- Total amount Jerry receives -/
def total_received : ℝ := 5440000

/-- Theorem stating the amount of medical bills Jerry is claiming -/
theorem jerry_medical_bills :
  claim_percentage * (total_lost_salary + medical_bills + 
    punitive_multiplier * (total_lost_salary + medical_bills)) = total_received ∧
  medical_bills = 200000 := by sorry

end NUMINAMATH_CALUDE_jerry_medical_bills_l148_14832


namespace NUMINAMATH_CALUDE_inequality_range_l148_14828

theorem inequality_range (b : ℝ) : 
  (b > 0 ∧ ∃ y : ℝ, |y - 5| + 2 * |y - 2| > b) → 0 < b ∧ b < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l148_14828


namespace NUMINAMATH_CALUDE_state_fair_earnings_l148_14836

theorem state_fair_earnings :
  let ticket_price : ℚ := 5
  let food_price : ℚ := 8
  let ride_price : ℚ := 4
  let souvenir_price : ℚ := 15
  let total_ticket_sales : ℚ := 2520
  let num_attendees : ℚ := total_ticket_sales / ticket_price
  let food_buyers_ratio : ℚ := 2/3
  let ride_goers_ratio : ℚ := 1/4
  let souvenir_buyers_ratio : ℚ := 1/8
  let food_earnings : ℚ := num_attendees * food_buyers_ratio * food_price
  let ride_earnings : ℚ := num_attendees * ride_goers_ratio * ride_price
  let souvenir_earnings : ℚ := num_attendees * souvenir_buyers_ratio * souvenir_price
  let total_earnings : ℚ := total_ticket_sales + food_earnings + ride_earnings + souvenir_earnings
  total_earnings = 6657 := by sorry

end NUMINAMATH_CALUDE_state_fair_earnings_l148_14836


namespace NUMINAMATH_CALUDE_gre_exam_month_l148_14820

-- Define the months as an enumeration
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

def next_month : Month → Month
| Month.January => Month.February
| Month.February => Month.March
| Month.March => Month.April
| Month.April => Month.May
| Month.May => Month.June
| Month.June => Month.July
| Month.July => Month.August
| Month.August => Month.September
| Month.September => Month.October
| Month.October => Month.November
| Month.November => Month.December
| Month.December => Month.January

def months_later (start : Month) (n : Nat) : Month :=
  match n with
  | 0 => start
  | n + 1 => next_month (months_later start n)

theorem gre_exam_month (start_month : Month) (preparation_months : Nat) :
  start_month = Month.June ∧ preparation_months = 5 →
  months_later start_month preparation_months = Month.November :=
by sorry

end NUMINAMATH_CALUDE_gre_exam_month_l148_14820


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l148_14857

theorem complex_magnitude_product : 
  Complex.abs ((5 * Real.sqrt 2 - 5 * Complex.I) * (2 * Real.sqrt 3 + 4 * Complex.I)) = 10 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l148_14857


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l148_14812

theorem prime_factorization_sum (a b c : ℕ) : 
  2^a * 3^b * 7^c = 432 → a + b + c = 5 → 3*a + 2*b + 4*c = 18 := by
sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l148_14812


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l148_14884

def total_balls : ℕ := 6 + 5 + 2

def red_balls : ℕ := 6

theorem probability_two_red_balls :
  let prob_first_red : ℚ := red_balls / total_balls
  let prob_second_red : ℚ := (red_balls - 1) / (total_balls - 1)
  prob_first_red * prob_second_red = 5 / 26 := by sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l148_14884


namespace NUMINAMATH_CALUDE_inverse_proportion_points_l148_14851

/-- An inverse proportion function passing through (-2, 3) also passes through (2, -3) -/
theorem inverse_proportion_points : 
  ∀ f : ℝ → ℝ, 
  (∀ x ≠ 0, ∃ k, f x = k / x) →  -- f is an inverse proportion function
  f (-2) = 3 →                   -- f passes through (-2, 3)
  f 2 = -3                       -- f passes through (2, -3)
:= by sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_l148_14851


namespace NUMINAMATH_CALUDE_ratio_IJ_IF_is_14_13_l148_14871

/-- A structure representing the geometric configuration described in the problem -/
structure TriangleConfiguration where
  /-- Point F -/
  F : ℝ × ℝ
  /-- Point G -/
  G : ℝ × ℝ
  /-- Point H -/
  H : ℝ × ℝ
  /-- Point I -/
  I : ℝ × ℝ
  /-- Point J -/
  J : ℝ × ℝ
  /-- FGH is a right triangle with right angle at H -/
  FGH_right_at_H : (F.1 - H.1) * (G.1 - H.1) + (F.2 - H.2) * (G.2 - H.2) = 0
  /-- FG = 5 -/
  FG_length : (F.1 - G.1)^2 + (F.2 - G.2)^2 = 25
  /-- GH = 12 -/
  GH_length : (G.1 - H.1)^2 + (G.2 - H.2)^2 = 144
  /-- FHI is a right triangle with right angle at F -/
  FHI_right_at_F : (H.1 - F.1) * (I.1 - F.1) + (H.2 - F.2) * (I.2 - F.2) = 0
  /-- FI = 15 -/
  FI_length : (F.1 - I.1)^2 + (F.2 - I.2)^2 = 225
  /-- H and I are on opposite sides of FG -/
  H_I_opposite_sides : ((G.1 - F.1) * (H.2 - F.2) - (G.2 - F.2) * (H.1 - F.1)) *
                       ((G.1 - F.1) * (I.2 - F.2) - (G.2 - F.2) * (I.1 - F.1)) < 0
  /-- IJ is parallel to FG -/
  IJ_parallel_FG : (J.1 - I.1) * (G.2 - F.2) = (J.2 - I.2) * (G.1 - F.1)
  /-- J is on the extension of GH -/
  J_on_GH_extended : ∃ t : ℝ, J.1 = G.1 + t * (H.1 - G.1) ∧ J.2 = G.2 + t * (H.2 - G.2)

/-- The main theorem stating that the ratio IJ/IF is equal to 14/13 -/
theorem ratio_IJ_IF_is_14_13 (config : TriangleConfiguration) :
  let IJ := ((config.I.1 - config.J.1)^2 + (config.I.2 - config.J.2)^2).sqrt
  let IF := ((config.I.1 - config.F.1)^2 + (config.I.2 - config.F.2)^2).sqrt
  IJ / IF = 14 / 13 :=
sorry

end NUMINAMATH_CALUDE_ratio_IJ_IF_is_14_13_l148_14871


namespace NUMINAMATH_CALUDE_number_puzzle_l148_14888

theorem number_puzzle (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l148_14888


namespace NUMINAMATH_CALUDE_triangle_perimeter_l148_14863

-- Define the triangle sides
def a : ℝ := 10
def b : ℝ := 6
def c : ℝ := 7

-- Define the perimeter
def perimeter : ℝ := a + b + c

-- Theorem statement
theorem triangle_perimeter : perimeter = 23 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l148_14863


namespace NUMINAMATH_CALUDE_number_of_nickels_l148_14815

def quarter_value : Rat := 25 / 100
def dime_value : Rat := 10 / 100
def nickel_value : Rat := 5 / 100
def penny_value : Rat := 1 / 100

def num_quarters : Nat := 10
def num_dimes : Nat := 3
def num_pennies : Nat := 200
def total_amount : Rat := 5

theorem number_of_nickels : 
  ∃ (num_nickels : Nat), 
    (num_quarters : Nat) * quarter_value + 
    (num_dimes : Nat) * dime_value + 
    (num_nickels : Nat) * nickel_value + 
    (num_pennies : Nat) * penny_value = total_amount ∧ 
    num_nickels = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_nickels_l148_14815


namespace NUMINAMATH_CALUDE_candy_count_difference_l148_14837

/-- The number of candies Bryan has -/
def bryan_candies : ℕ := 50

/-- The number of candies Ben has -/
def ben_candies : ℕ := 20

/-- The difference in candy count between Bryan and Ben -/
def candy_difference : ℕ := bryan_candies - ben_candies

theorem candy_count_difference :
  candy_difference = 30 :=
by sorry

end NUMINAMATH_CALUDE_candy_count_difference_l148_14837


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l148_14860

theorem necessary_but_not_sufficient_condition 
  (A B C : Set α) 
  (hAnonempty : A.Nonempty) 
  (hBnonempty : B.Nonempty) 
  (hCnonempty : C.Nonempty) 
  (hUnion : A ∪ B = C) 
  (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ x, x ∈ C ∧ x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l148_14860


namespace NUMINAMATH_CALUDE_circles_intersection_l148_14879

def circle1_center : ℝ × ℝ := (0, 0)
def circle2_center : ℝ × ℝ := (-3, 4)
def circle2_radius : ℝ := 2

theorem circles_intersection (m : ℝ) :
  (∃ (x y : ℝ), (x - circle1_center.1)^2 + (y - circle1_center.2)^2 = m ∧
                (x - circle2_center.1)^2 + (y - circle2_center.2)^2 = circle2_radius^2) ↔
  9 < m ∧ m < 49 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersection_l148_14879


namespace NUMINAMATH_CALUDE_negative_two_less_than_negative_three_halves_l148_14885

theorem negative_two_less_than_negative_three_halves : -2 < -(3/2) := by
  sorry

end NUMINAMATH_CALUDE_negative_two_less_than_negative_three_halves_l148_14885


namespace NUMINAMATH_CALUDE_trigonometric_identity_l148_14808

theorem trigonometric_identity (α : ℝ) :
  Real.cos (3 / 2 * Real.pi + 4 * α) + Real.sin (3 * Real.pi - 8 * α) - Real.sin (4 * Real.pi - 12 * α) =
  4 * Real.cos (2 * α) * Real.cos (4 * α) * Real.sin (6 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l148_14808


namespace NUMINAMATH_CALUDE_number_ratio_problem_l148_14825

/-- Given three numbers satisfying specific conditions, prove their ratios -/
theorem number_ratio_problem (a b c : ℚ) : 
  a + b + c = 98 → 
  b = 30 → 
  c = (8/5) * b → 
  a / b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l148_14825


namespace NUMINAMATH_CALUDE_question_probabilities_l148_14878

def total_questions : ℕ := 5
def algebra_questions : ℕ := 2
def geometry_questions : ℕ := 3

theorem question_probabilities :
  let prob_algebra_then_geometry := (algebra_questions : ℚ) / total_questions * 
                                    (geometry_questions : ℚ) / (total_questions - 1)
  let prob_geometry_given_algebra := (geometry_questions : ℚ) / (total_questions - 1)
  prob_algebra_then_geometry = 3 / 10 ∧ prob_geometry_given_algebra = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_question_probabilities_l148_14878


namespace NUMINAMATH_CALUDE_equation_three_solutions_l148_14807

theorem equation_three_solutions :
  let f : ℝ → ℝ := λ x => (x^2 - 4) * (x^2 - 1) - (x^2 + 3*x + 2) * (x^2 - 8*x + 7)
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f x = 0 ∧ ∀ x, f x = 0 → x ∈ s :=
by sorry

end NUMINAMATH_CALUDE_equation_three_solutions_l148_14807


namespace NUMINAMATH_CALUDE_tiffany_found_two_bags_l148_14889

/-- The number of bags Tiffany found on the next day -/
def bags_found_next_day (bags_monday : ℕ) (total_bags : ℕ) : ℕ :=
  total_bags - bags_monday

/-- Theorem: Tiffany found 2 bags on the next day -/
theorem tiffany_found_two_bags :
  let bags_monday := 4
  let total_bags := 6
  bags_found_next_day bags_monday total_bags = 2 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_found_two_bags_l148_14889


namespace NUMINAMATH_CALUDE_price_of_zinc_l148_14800

/-- Given the price of copper, the total weight of brass, the selling price of brass,
    and the amount of copper used, calculate the price of zinc per pound. -/
theorem price_of_zinc 
  (price_copper : ℚ)
  (total_weight : ℚ)
  (selling_price : ℚ)
  (copper_used : ℚ)
  (h1 : price_copper = 65/100)
  (h2 : total_weight = 70)
  (h3 : selling_price = 45/100)
  (h4 : copper_used = 30)
  : ∃ (price_zinc : ℚ), price_zinc = 30/100 := by
  sorry

#check price_of_zinc

end NUMINAMATH_CALUDE_price_of_zinc_l148_14800


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l148_14843

/-- Given that x and y are inversely proportional, prove that when x + y = 60, x = 3y, 
    and x = -6, then y = -112.5 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) : 
  (x * y = k) →  -- x and y are inversely proportional
  (x + y = 60) →  -- sum condition
  (x = 3 * y) →  -- proportion condition
  (x = -6) →  -- given x value
  y = -112.5 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l148_14843


namespace NUMINAMATH_CALUDE_second_range_lower_limit_l148_14850

theorem second_range_lower_limit (x y : ℝ) 
  (h1 : 3 < x) (h2 : x < 8) (h3 : x > y) (h4 : x < 10) (h5 : x = 7) : 
  3 < y ∧ y ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_second_range_lower_limit_l148_14850


namespace NUMINAMATH_CALUDE_intersection_on_y_axis_l148_14897

/-- Given two lines in the xy-plane defined by equations 2x + 3y - k = 0 and x - ky + 12 = 0,
    if their intersection point lies on the y-axis, then k = 6 or k = -6. -/
theorem intersection_on_y_axis (k : ℝ) : 
  (∃ y : ℝ, 2 * 0 + 3 * y - k = 0 ∧ 0 - k * y + 12 = 0) →
  k = 6 ∨ k = -6 := by
sorry

end NUMINAMATH_CALUDE_intersection_on_y_axis_l148_14897


namespace NUMINAMATH_CALUDE_max_integer_difference_l148_14887

theorem max_integer_difference (x y : ℤ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) :
  (∀ (a b : ℤ), 5 < a ∧ a < 8 ∧ 8 < b ∧ b < 13 → y - x ≥ b - a) ∧ y - x ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_integer_difference_l148_14887


namespace NUMINAMATH_CALUDE_xiaoliang_draw_probability_l148_14853

/-- Represents the labels of balls in the box -/
inductive Label : Type
| one : Label
| two : Label
| three : Label
| four : Label

/-- The state of the box after Xiaoming's draw -/
structure BoxState :=
  (remaining_two : Nat)
  (remaining_three : Nat)
  (remaining_four : Nat)

/-- The initial state of the box -/
def initial_box : BoxState :=
  { remaining_two := 2
  , remaining_three := 1
  , remaining_four := 2 }

/-- The total number of balls remaining in the box -/
def total_remaining (box : BoxState) : Nat :=
  box.remaining_two + box.remaining_three + box.remaining_four

/-- The probability of drawing a ball with a specific label -/
def prob_draw (box : BoxState) (label : Label) : Rat :=
  match label with
  | Label.one => 0  -- No balls labeled 1 remaining
  | Label.two => box.remaining_two / (total_remaining box)
  | Label.three => box.remaining_three / (total_remaining box)
  | Label.four => box.remaining_four / (total_remaining box)

/-- The probability of drawing a ball matching Xiaoming's drawn balls -/
def prob_match_xiaoming (box : BoxState) : Rat :=
  prob_draw box Label.three

theorem xiaoliang_draw_probability :
  prob_match_xiaoming initial_box = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_xiaoliang_draw_probability_l148_14853


namespace NUMINAMATH_CALUDE_range_of_m_l148_14813

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -Real.sqrt (4 - p.2^2)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 6}

-- Define the condition for points P and Q
def existsPQ (m : ℝ) : Prop :=
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ C ∧ Q ∈ l ∧
    (P.1 - m, P.2) + (Q.1 - m, Q.2) = (0, 0)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, existsPQ m → 2 ≤ m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l148_14813


namespace NUMINAMATH_CALUDE_squarable_numbers_l148_14858

def isSquarable (n : ℕ) : Prop :=
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ (i : Fin n), ∃ (k : ℕ), (p i).val + i.val + 1 = k^2

theorem squarable_numbers : 
  (¬ isSquarable 7) ∧ 
  (isSquarable 9) ∧ 
  (¬ isSquarable 11) ∧ 
  (isSquarable 15) := by sorry

end NUMINAMATH_CALUDE_squarable_numbers_l148_14858


namespace NUMINAMATH_CALUDE_johns_donation_l148_14880

/-- Given 10 initial contributions, if a new donation causes the average
    contribution to increase by 80% to $90, then the new donation must be $490. -/
theorem johns_donation (initial_count : ℕ) (increase_percentage : ℚ) (new_average : ℚ) :
  initial_count = 10 →
  increase_percentage = 80 / 100 →
  new_average = 90 →
  let initial_average := new_average / (1 + increase_percentage)
  let initial_total := initial_count * initial_average
  let new_total := (initial_count + 1) * new_average
  new_total - initial_total = 490 := by
sorry

end NUMINAMATH_CALUDE_johns_donation_l148_14880


namespace NUMINAMATH_CALUDE_power_of_five_l148_14839

theorem power_of_five (m : ℕ) : 5^m = 5 * 25^4 * 625^3 → m = 21 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_l148_14839


namespace NUMINAMATH_CALUDE_hockey_league_face_count_l148_14875

/-- The number of times each team faces all other teams in a hockey league -/
def face_count (num_teams : ℕ) (total_games : ℕ) : ℕ :=
  total_games / (num_teams * (num_teams - 1) / 2)

/-- Theorem: In a hockey league with 18 teams, where each team faces all other teams
    the same number of times, and a total of 1530 games are played in the season,
    each team faces all the other teams 5 times. -/
theorem hockey_league_face_count :
  face_count 18 1530 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_face_count_l148_14875


namespace NUMINAMATH_CALUDE_right_triangle_special_case_l148_14873

theorem right_triangle_special_case (a b c : ℝ) :
  a > 0 →  -- AB is positive
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  c + b = 2*a →  -- Given condition
  b = 3/4 * a ∧ c = 5/4 * a := by sorry

end NUMINAMATH_CALUDE_right_triangle_special_case_l148_14873


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l148_14821

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (3 * a + 9 * b) % 63 = 45 ∧ (7 * a) % 63 = 1 ∧ (13 * b) % 63 = 1 :=
by sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l148_14821


namespace NUMINAMATH_CALUDE_restaurant_group_size_l148_14844

theorem restaurant_group_size (adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) (children : ℕ) : 
  adults = 2 →
  meal_cost = 3 →
  total_bill = 21 →
  children * meal_cost + adults * meal_cost = total_bill →
  children = 5 := by
sorry

end NUMINAMATH_CALUDE_restaurant_group_size_l148_14844


namespace NUMINAMATH_CALUDE_M_is_range_of_f_l148_14896

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x^2}

-- Define the function f(x) = x^2
def f : ℝ → ℝ := λ x ↦ x^2

-- Theorem statement
theorem M_is_range_of_f : M = Set.range f := by sorry

end NUMINAMATH_CALUDE_M_is_range_of_f_l148_14896


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l148_14898

-- Problem 1
theorem problem_1 : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) + Real.sqrt 6 / Real.sqrt 2 = 2 + Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x = Real.sqrt 2 - 2) : 
  ((1 / (x - 1) - 1 / (x + 1)) / ((x + 2) / (x^2 - 1))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l148_14898


namespace NUMINAMATH_CALUDE_equal_water_levels_l148_14809

/-- Represents a pool with initial height and drain time -/
structure Pool where
  initial_height : ℝ
  drain_time : ℝ

/-- The time when water levels in two pools become equal -/
def equal_level_time (pool_a pool_b : Pool) : ℝ :=
  1 -- The actual value we want to prove

theorem equal_water_levels (pool_a pool_b : Pool) :
  pool_b.initial_height = 1.5 * pool_a.initial_height →
  pool_a.drain_time = 2 →
  pool_b.drain_time = 1.5 →
  equal_level_time pool_a pool_b = 1 := by
  sorry

#check equal_water_levels

end NUMINAMATH_CALUDE_equal_water_levels_l148_14809


namespace NUMINAMATH_CALUDE_vector_collinearity_l148_14877

/-- Given vectors m, n, and k in ℝ², prove that if m - 2n is collinear with k, then t = 1 -/
theorem vector_collinearity (m n k : ℝ × ℝ) (t : ℝ) 
  (hm : m = (Real.sqrt 3, 1)) 
  (hn : n = (0, -1)) 
  (hk : k = (t, Real.sqrt 3)) 
  (hcol : ∃ (c : ℝ), c • (m - 2 • n) = k) : 
  t = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l148_14877


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l148_14874

theorem jacket_price_reduction (x : ℝ) : 
  (1 - x / 100) * (1 - 0.15) * (1 + 56.86274509803921 / 100) = 1 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l148_14874


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l148_14849

/-- Given three identical circles with circumference 48, where each circle touches the other two,
    and the arcs in the shaded region each subtend an angle of 90 degrees at the center of their
    respective circles, the perimeter of the shaded region is equal to 36. -/
theorem shaded_region_perimeter (circle_circumference : ℝ) (arc_angle : ℝ) : 
  circle_circumference = 48 → 
  arc_angle = 90 →
  (3 * (arc_angle / 360) * circle_circumference) = 36 := by
  sorry

#check shaded_region_perimeter

end NUMINAMATH_CALUDE_shaded_region_perimeter_l148_14849


namespace NUMINAMATH_CALUDE_elderly_in_sample_is_18_l148_14814

/-- Represents the distribution of employees in a company and their sampling --/
structure EmployeeSampling where
  total : ℕ
  young : ℕ
  elderly : ℕ
  sampledYoung : ℕ
  middleAged : ℕ := 2 * elderly
  youngRatio : ℚ := young / total
  elderlyRatio : ℚ := elderly / total

/-- The number of elderly employees in the sample given the conditions --/
def elderlyInSample (e : EmployeeSampling) : ℚ :=
  e.elderlyRatio * (e.sampledYoung / e.youngRatio)

/-- Theorem stating the number of elderly employees in the sample --/
theorem elderly_in_sample_is_18 (e : EmployeeSampling) 
    (h1 : e.total = 430)
    (h2 : e.young = 160)
    (h3 : e.sampledYoung = 32)
    (h4 : e.total = e.young + e.middleAged + e.elderly) :
  elderlyInSample e = 18 := by
  sorry

#eval elderlyInSample { total := 430, young := 160, elderly := 90, sampledYoung := 32 }

end NUMINAMATH_CALUDE_elderly_in_sample_is_18_l148_14814


namespace NUMINAMATH_CALUDE_negative_inequality_l148_14870

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l148_14870


namespace NUMINAMATH_CALUDE_parallel_implies_alternate_interior_angles_vertical_angles_are_equal_right_triangle_acute_angles_complementary_supplements_of_same_angle_are_equal_inverse_of_vertical_angles_false_others_true_l148_14841

-- Define the basic concepts
def Line : Type := sorry
def Angle : Type := sorry
def Triangle : Type := sorry

-- Define the properties
def parallel (l1 l2 : Line) : Prop := sorry
def alternateInteriorAngles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry
def verticalAngles (a1 a2 : Angle) : Prop := sorry
def rightTriangle (t : Triangle) : Prop := sorry
def acuteAngles (t : Triangle) (a1 a2 : Angle) : Prop := sorry
def complementaryAngles (a1 a2 : Angle) : Prop := sorry
def supplementaryAngles (a1 a2 : Angle) : Prop := sorry

-- Theorem A
theorem parallel_implies_alternate_interior_angles (l1 l2 : Line) (a1 a2 : Angle) :
  parallel l1 l2 → alternateInteriorAngles a1 a2 l1 l2 := sorry

-- Theorem B
theorem vertical_angles_are_equal (a1 a2 : Angle) :
  verticalAngles a1 a2 → a1 = a2 := sorry

-- Theorem C
theorem right_triangle_acute_angles_complementary (t : Triangle) (a1 a2 : Angle) :
  rightTriangle t → acuteAngles t a1 a2 → complementaryAngles a1 a2 := sorry

-- Theorem D
theorem supplements_of_same_angle_are_equal (a1 a2 a3 : Angle) :
  supplementaryAngles a1 a3 → supplementaryAngles a2 a3 → a1 = a2 := sorry

-- The main theorem: inverse of B is false, while inverses of A, C, and D are true
theorem inverse_of_vertical_angles_false_others_true :
  (∃ a1 a2 : Angle, a1 = a2 ∧ ¬verticalAngles a1 a2) ∧
  (∀ l1 l2 : Line, ∀ a1 a2 : Angle, alternateInteriorAngles a1 a2 l1 l2 → parallel l1 l2) ∧
  (∀ a1 a2 : Angle, complementaryAngles a1 a2 → ∃ t : Triangle, rightTriangle t ∧ acuteAngles t a1 a2) ∧
  (∀ a1 a2 a3 : Angle, a1 = a2 → supplementaryAngles a1 a3 → supplementaryAngles a2 a3) := sorry

end NUMINAMATH_CALUDE_parallel_implies_alternate_interior_angles_vertical_angles_are_equal_right_triangle_acute_angles_complementary_supplements_of_same_angle_are_equal_inverse_of_vertical_angles_false_others_true_l148_14841


namespace NUMINAMATH_CALUDE_negative_marks_for_wrong_answer_l148_14801

def total_questions : ℕ := 150
def correct_answers : ℕ := 120
def total_score : ℕ := 420
def correct_score : ℕ := 4

theorem negative_marks_for_wrong_answer :
  ∃ (x : ℚ), 
    (correct_score * correct_answers : ℚ) - 
    (x * (total_questions - correct_answers)) = total_score ∧
    x = 2 := by sorry

end NUMINAMATH_CALUDE_negative_marks_for_wrong_answer_l148_14801


namespace NUMINAMATH_CALUDE_smallest_start_number_for_2520_divisibility_l148_14823

theorem smallest_start_number_for_2520_divisibility : 
  ∃ (n : ℕ), n > 0 ∧ n ≤ 10 ∧ 
  (∀ (k : ℕ), n ≤ k ∧ k ≤ 10 → 2520 % k = 0) ∧
  (∀ (m : ℕ), m < n → ∃ (j : ℕ), m < j ∧ j ≤ 10 ∧ 2520 % j ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_start_number_for_2520_divisibility_l148_14823


namespace NUMINAMATH_CALUDE_bucket_fill_time_l148_14806

/-- The time taken to fill a bucket completely, given that two-thirds of it is filled in 90 seconds at a constant rate. -/
theorem bucket_fill_time (fill_rate : ℝ) (h1 : fill_rate > 0) : 
  (2 / 3 : ℝ) / fill_rate = 90 → 1 / fill_rate = 135 := by sorry

end NUMINAMATH_CALUDE_bucket_fill_time_l148_14806


namespace NUMINAMATH_CALUDE_fraction_sum_l148_14840

theorem fraction_sum : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l148_14840


namespace NUMINAMATH_CALUDE_apple_weight_is_quarter_pound_l148_14869

/-- The weight of a small apple in pounds -/
def apple_weight : ℝ := 0.25

/-- The cost of apples per pound in dollars -/
def cost_per_pound : ℝ := 2

/-- The total amount spent on apples in dollars -/
def total_spent : ℝ := 7

/-- The number of days the apples should last -/
def days : ℕ := 14

/-- Theorem stating that the weight of a small apple is 0.25 pounds -/
theorem apple_weight_is_quarter_pound :
  apple_weight = total_spent / (cost_per_pound * days) := by sorry

end NUMINAMATH_CALUDE_apple_weight_is_quarter_pound_l148_14869


namespace NUMINAMATH_CALUDE_solution_is_two_lines_l148_14854

-- Define the equation
def equation (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2 + 4*x

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {p | equation p.1 p.2}

-- Define the two lines
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}
def horizontal_line : Set (ℝ × ℝ) := {p | p.2 = 2}

-- Theorem statement
theorem solution_is_two_lines :
  solution_set = y_axis ∪ horizontal_line :=
sorry

end NUMINAMATH_CALUDE_solution_is_two_lines_l148_14854


namespace NUMINAMATH_CALUDE_secant_theorem_l148_14831

-- Define the basic geometric elements
variable (A B C M A₁ B₁ C₁ : ℝ × ℝ)

-- Define the triangle ABC
def is_triangle (A B C : ℝ × ℝ) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define that M is not on the sides or extensions of ABC
def M_not_on_triangle (A B C M : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, M ≠ A + t • (B - A) ∧ 
           M ≠ B + t • (C - B) ∧ 
           M ≠ C + t • (A - C)

-- Define the secant through M intersecting sides (or extensions) at A₁, B₁, C₁
def secant_intersects (A B C M A₁ B₁ C₁ : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ, 
    A₁ = A + t₁ • (B - A) ∧
    B₁ = B + t₂ • (C - B) ∧
    C₁ = C + t₃ • (A - C) ∧
    (∃ s₁ s₂ s₃ : ℝ, M = A₁ + s₁ • (B₁ - A₁) ∧
                     M = B₁ + s₂ • (C₁ - B₁) ∧
                     M = C₁ + s₃ • (A₁ - C₁))

-- Define oriented area function
noncomputable def oriented_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- Define oriented distance function
noncomputable def oriented_distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem secant_theorem (A B C M A₁ B₁ C₁ : ℝ × ℝ) 
  (h_triangle : is_triangle A B C)
  (h_M_not_on : M_not_on_triangle A B C M)
  (h_secant : secant_intersects A B C M A₁ B₁ C₁) :
  (oriented_area A B M) / (oriented_distance M C₁) + 
  (oriented_area B C M) / (oriented_distance M A₁) + 
  (oriented_area C A M) / (oriented_distance M B₁) = 0 := by sorry

end NUMINAMATH_CALUDE_secant_theorem_l148_14831


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l148_14893

theorem divisibility_by_seven (n : ℕ) : 7 ∣ (3^(12*n + 1) + 2^(6*n + 2)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l148_14893


namespace NUMINAMATH_CALUDE_february_has_greatest_difference_l148_14826

structure MonthData where
  drummers : ℕ
  bugle : ℕ
  cymbal : ℕ

def salesData : List MonthData := [
  ⟨5, 4, 3⟩, -- January
  ⟨6, 4, 2⟩, -- February
  ⟨4, 4, 4⟩, -- March
  ⟨2, 5, 3⟩, -- April
  ⟨3, 4, 5⟩  -- May
]

def percentageDifference (m : MonthData) : ℚ :=
  let max := max m.drummers (max m.bugle m.cymbal)
  let min := min m.drummers (min m.bugle m.cymbal)
  (max - min : ℚ) / min * 100

def februaryIndex : Fin 5 := 1

theorem february_has_greatest_difference :
  ∀ i : Fin 5, i ≠ februaryIndex →
    percentageDifference (salesData.get februaryIndex) ≥ percentageDifference (salesData.get i) :=
by sorry

end NUMINAMATH_CALUDE_february_has_greatest_difference_l148_14826


namespace NUMINAMATH_CALUDE_xyz_equals_27_l148_14846

theorem xyz_equals_27 
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = b * c * (x - 2))
  (eq_b : b = a * c * (y - 2))
  (eq_c : c = a * b * (z - 2))
  (sum_product : x * y + x * z + y * z = 10)
  (sum : x + y + z = 6) :
  x * y * z = 27 := by
  sorry


end NUMINAMATH_CALUDE_xyz_equals_27_l148_14846


namespace NUMINAMATH_CALUDE_min_S_proof_l148_14848

/-- The number of dice rolled -/
def n : ℕ := 333

/-- The target sum -/
def target_sum : ℕ := 1994

/-- The minimum value of S -/
def min_S : ℕ := 334

/-- The probability of obtaining a sum of k when rolling n standard dice -/
noncomputable def prob_sum (k : ℕ) : ℝ := sorry

theorem min_S_proof :
  (prob_sum target_sum > 0) ∧
  (prob_sum target_sum = prob_sum min_S) ∧
  (∀ S : ℕ, S < min_S → prob_sum target_sum ≠ prob_sum S) :=
sorry

end NUMINAMATH_CALUDE_min_S_proof_l148_14848


namespace NUMINAMATH_CALUDE_unique_digit_solution_l148_14833

/-- Represents a six-digit number as a list of digits -/
def SixDigitNumber := List Nat

/-- Converts a three-digit number to a six-digit number -/
def toSixDigit (n : Nat) : SixDigitNumber :=
  sorry

/-- Converts a list of digits to a natural number -/
def fromDigits (digits : List Nat) : Nat :=
  sorry

/-- Checks if all digits in a list are distinct -/
def allDistinct (digits : List Nat) : Prop :=
  sorry

/-- Theorem: Unique solution for the given digit equation system -/
theorem unique_digit_solution :
  ∃! (A B C D E F : Nat),
    A ∈ Finset.range 10 ∧
    B ∈ Finset.range 10 ∧
    C ∈ Finset.range 10 ∧
    D ∈ Finset.range 10 ∧
    E ∈ Finset.range 10 ∧
    F ∈ Finset.range 10 ∧
    allDistinct [A, B, C, D, E, F] ∧
    fromDigits [A, B, C] ^ 2 = fromDigits (toSixDigit (fromDigits [D, A, E, C, F, B])) ∧
    fromDigits [C, B, A] ^ 2 = fromDigits (toSixDigit (fromDigits [E, D, C, A, B, F])) ∧
    A = 3 ∧ B = 6 ∧ C = 4 ∧ D = 1 ∧ E = 2 ∧ F = 9 :=
  by sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l148_14833


namespace NUMINAMATH_CALUDE_age_problem_l148_14816

theorem age_problem (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  ∃ k : ℤ, (b - 1) / (a - 1) = k ∧ (b + 1) / (a + 1) = k + 1 →
  ∃ m : ℤ, (c - 1) / (b - 1) = m ∧ (c + 1) / (b + 1) = m + 1 →
  a + b + c ≤ 150 →
  a = 2 ∧ b = 7 ∧ c = 49 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l148_14816


namespace NUMINAMATH_CALUDE_smallest_multiple_of_7_greater_than_neg_50_l148_14819

theorem smallest_multiple_of_7_greater_than_neg_50 :
  ∀ n : ℤ, n > -50 ∧ n % 7 = 0 → n ≥ -49 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_7_greater_than_neg_50_l148_14819


namespace NUMINAMATH_CALUDE_largest_inscribed_pentagon_is_regular_l148_14881

/-- A pentagon inscribed in a circle of radius 1 --/
structure InscribedPentagon where
  /-- The vertices of the pentagon --/
  vertices : Fin 5 → ℝ × ℝ
  /-- All vertices lie on the unit circle --/
  on_circle : ∀ i, (vertices i).1^2 + (vertices i).2^2 = 1

/-- The area of an inscribed pentagon --/
def area (p : InscribedPentagon) : ℝ :=
  sorry

/-- A regular pentagon inscribed in a circle of radius 1 --/
def regular_pentagon : InscribedPentagon :=
  sorry

theorem largest_inscribed_pentagon_is_regular :
  ∀ p : InscribedPentagon, area p ≤ area regular_pentagon :=
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_pentagon_is_regular_l148_14881


namespace NUMINAMATH_CALUDE_ab_value_l148_14827

theorem ab_value (a b : ℝ) (h1 : a * Real.exp a = Real.exp 2) (h2 : Real.log (b / Real.exp 1) = Real.exp 3 / b) : a * b = Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l148_14827


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_l148_14886

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| - a

-- Theorem 1: Solution set for f(x) > x + 1 when a = 1
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x > x + 1} = {x : ℝ | x > 3 ∨ x < -1/3} :=
sorry

-- Theorem 2: Range of a for which ∃x : f(x) < 0.5 * f(x + 1)
theorem range_of_a :
  {a : ℝ | ∃ x, f a x < 0.5 * f a (x + 1)} = {a : ℝ | a > -2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_l148_14886


namespace NUMINAMATH_CALUDE_triangular_region_area_ratio_l148_14830

/-- Represents a square divided into a 6x6 grid -/
structure GridSquare where
  side_length : ℝ
  grid_size : ℕ := 6

/-- Represents the triangular region in the GridSquare -/
structure TriangularRegion (gs : GridSquare) where
  vertex1 : ℝ × ℝ  -- Midpoint of one side
  vertex2 : ℝ × ℝ  -- Diagonal corner of 4x4 block
  vertex3 : ℝ × ℝ  -- Midpoint of adjacent side

/-- Calculates the area of the GridSquare -/
def area_grid_square (gs : GridSquare) : ℝ :=
  gs.side_length ^ 2

/-- Calculates the area of the TriangularRegion -/
noncomputable def area_triangular_region (gs : GridSquare) (tr : TriangularRegion gs) : ℝ :=
  sorry  -- Actual calculation would go here

/-- The main theorem stating the ratio of areas -/
theorem triangular_region_area_ratio (gs : GridSquare) (tr : TriangularRegion gs) :
  area_triangular_region gs tr / area_grid_square gs = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_triangular_region_area_ratio_l148_14830


namespace NUMINAMATH_CALUDE_power_of_two_equality_l148_14895

theorem power_of_two_equality (x : ℤ) : (1 / 8 : ℚ) * (2 ^ 50) = 2 ^ x → x = 47 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l148_14895


namespace NUMINAMATH_CALUDE_custom_mul_equality_l148_14838

/-- Custom multiplication operation for real numbers -/
def custom_mul (a b : ℝ) : ℝ := (a - b^3)^2

/-- Theorem stating the equality for the given expression -/
theorem custom_mul_equality (x y : ℝ) :
  custom_mul ((x - y)^2) ((y^2 - x^2)^2) = ((x - y)^2 - (y^4 - 2*x^2*y^2 + x^4)^3)^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_equality_l148_14838


namespace NUMINAMATH_CALUDE_middle_school_math_club_payment_l148_14876

theorem middle_school_math_club_payment (A : Nat) : 
  A < 10 → (100 + 10 * A + 2) % 11 = 0 ↔ A = 3 := by
  sorry

end NUMINAMATH_CALUDE_middle_school_math_club_payment_l148_14876


namespace NUMINAMATH_CALUDE_range_of_r_l148_14847

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^4 + 6*x^2 + 9 - 2*x

-- State the theorem
theorem range_of_r :
  ∀ y : ℝ, y ≥ 9 ↔ ∃ x : ℝ, x ≥ 0 ∧ r x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_r_l148_14847


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l148_14867

theorem complex_fraction_equality : Complex.I * 4 / (Real.sqrt 3 + Complex.I) = 1 + Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l148_14867


namespace NUMINAMATH_CALUDE_luisa_apples_taken_l148_14864

/-- Proves that Luisa took out 2 apples from the bag -/
theorem luisa_apples_taken (initial_apples initial_oranges initial_mangoes : ℕ)
  (remaining_fruits : ℕ) :
  initial_apples = 7 →
  initial_oranges = 8 →
  initial_mangoes = 15 →
  remaining_fruits = 14 →
  ∃ (apples_taken : ℕ),
    apples_taken + 2 * apples_taken + (2 * initial_mangoes / 3) =
      initial_apples + initial_oranges + initial_mangoes - remaining_fruits ∧
    apples_taken = 2 :=
by sorry

end NUMINAMATH_CALUDE_luisa_apples_taken_l148_14864


namespace NUMINAMATH_CALUDE_f_non_monotonic_iff_l148_14862

/-- A piecewise function f(x) depending on parameters a and t -/
noncomputable def f (a t x : ℝ) : ℝ :=
  if x ≤ t then (4*a - 3)*x + 2*a - 4 else 2*x^3 - 6*x

/-- The theorem stating the condition for f to be non-monotonic for all t -/
theorem f_non_monotonic_iff (a : ℝ) :
  (∀ t : ℝ, ¬Monotone (f a t)) ↔ a ≤ 3/4 := by sorry

end NUMINAMATH_CALUDE_f_non_monotonic_iff_l148_14862


namespace NUMINAMATH_CALUDE_triangle_sides_from_heights_and_median_l148_14883

/-- Given a triangle with heights m₁ and m₂ corresponding to sides a and b respectively,
    and median k₃ corresponding to side c, prove that the sides a and b can be expressed as:
    a = m₂ / sin(γ) and b = m₁ / sin(γ), where γ is the angle opposite to side c. -/
theorem triangle_sides_from_heights_and_median 
  (m₁ m₂ k₃ : ℝ) (γ : ℝ) (hm₁ : m₁ > 0) (hm₂ : m₂ > 0) (hk₃ : k₃ > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ (a b : ℝ), a = m₂ / Real.sin γ ∧ b = m₁ / Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_from_heights_and_median_l148_14883


namespace NUMINAMATH_CALUDE_difference_is_895_l148_14829

/-- The smallest positive three-digit integer congruent to 7 (mod 13) -/
def m : ℕ := sorry

/-- The smallest positive four-digit integer congruent to 7 (mod 13) -/
def n : ℕ := sorry

/-- m is a three-digit number -/
axiom m_three_digit : 100 ≤ m ∧ m < 1000

/-- n is a four-digit number -/
axiom n_four_digit : 1000 ≤ n ∧ n < 10000

/-- m is congruent to 7 (mod 13) -/
axiom m_congruence : m % 13 = 7

/-- n is congruent to 7 (mod 13) -/
axiom n_congruence : n % 13 = 7

/-- m is the smallest such number -/
axiom m_smallest : ∀ k : ℕ, 100 ≤ k ∧ k < 1000 ∧ k % 13 = 7 → m ≤ k

/-- n is the smallest such number -/
axiom n_smallest : ∀ k : ℕ, 1000 ≤ k ∧ k < 10000 ∧ k % 13 = 7 → n ≤ k

theorem difference_is_895 : n - m = 895 := by sorry

end NUMINAMATH_CALUDE_difference_is_895_l148_14829


namespace NUMINAMATH_CALUDE_zeros_properties_l148_14872

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

-- State the theorem
theorem zeros_properties (k α β : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < 2)
  (h4 : f k α = 0) (h5 : f k β = 0) :
  (-7/2 < k ∧ k < -1) ∧ (1/α + 1/β < 4) := by
  sorry

end NUMINAMATH_CALUDE_zeros_properties_l148_14872


namespace NUMINAMATH_CALUDE_tangent_line_curve_l148_14892

-- Define the line equation
def line (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the curve equation
def curve (x y a : ℝ) : Prop := y = Real.log x + a

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x y : ℝ, line x y ∧ curve x y a ∧
    (∀ x' y' : ℝ, x' ≠ x → line x' y' → curve x' y' a → (y' - y) / (x' - x) ≠ 1 / x)

-- Theorem statement
theorem tangent_line_curve (a : ℝ) : is_tangent a → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_curve_l148_14892


namespace NUMINAMATH_CALUDE_area_of_region_is_5_25_l148_14802

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The region defined by the given inequalities -/
def Region : Set Point :=
  {p : Point | p.y > 3 * p.x ∧ p.y > 5 - 2 * p.x ∧ p.y < 6}

/-- The area of the region -/
noncomputable def areaOfRegion : ℝ := sorry

/-- Theorem stating that the area of the region is 5.25 square units -/
theorem area_of_region_is_5_25 : areaOfRegion = 5.25 := by sorry

end NUMINAMATH_CALUDE_area_of_region_is_5_25_l148_14802


namespace NUMINAMATH_CALUDE_condition_p_neither_sufficient_nor_necessary_for_q_l148_14868

theorem condition_p_neither_sufficient_nor_necessary_for_q :
  ¬(∀ x : ℝ, (1 / x ≤ 1) → (x^2 - 2*x ≥ 0)) ∧
  ¬(∀ x : ℝ, (x^2 - 2*x ≥ 0) → (1 / x ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_p_neither_sufficient_nor_necessary_for_q_l148_14868


namespace NUMINAMATH_CALUDE_smallest_cube_box_for_pyramid_l148_14842

theorem smallest_cube_box_for_pyramid (pyramid_height base_length base_width : ℝ) 
  (h_height : pyramid_height = 15)
  (h_base_length : base_length = 9)
  (h_base_width : base_width = 12) :
  let box_side := max pyramid_height (max base_length base_width)
  (box_side ^ 3 : ℝ) = 3375 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_box_for_pyramid_l148_14842


namespace NUMINAMATH_CALUDE_elle_practice_time_l148_14899

/-- The number of minutes Elle practices piano on a weekday -/
def weekday_practice : ℕ := 30

/-- The number of weekdays Elle practices piano -/
def weekdays : ℕ := 5

/-- The factor by which Elle's Saturday practice is longer than a weekday -/
def saturday_factor : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the total number of hours Elle spends practicing piano each week -/
def total_practice_hours : ℚ :=
  let weekday_total := weekday_practice * weekdays
  let saturday_practice := weekday_practice * saturday_factor
  let total_minutes := weekday_total + saturday_practice
  (total_minutes : ℚ) / minutes_per_hour

theorem elle_practice_time : total_practice_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_elle_practice_time_l148_14899


namespace NUMINAMATH_CALUDE_claire_photos_l148_14856

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 24) :
  claire = 12 := by
  sorry

end NUMINAMATH_CALUDE_claire_photos_l148_14856


namespace NUMINAMATH_CALUDE_rectangle_circle_square_area_l148_14890

theorem rectangle_circle_square_area :
  ∀ (rectangle_length rectangle_breadth rectangle_area circle_radius square_side : ℝ),
    rectangle_length = 5 * circle_radius →
    rectangle_breadth = 11 →
    rectangle_area = 220 →
    rectangle_area = rectangle_length * rectangle_breadth →
    circle_radius = square_side →
    square_side ^ 2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_square_area_l148_14890


namespace NUMINAMATH_CALUDE_ivan_apple_purchase_l148_14817

theorem ivan_apple_purchase (mini_pies : ℕ) (apples_per_mini_pie : ℚ) (leftover_apples : ℕ) 
  (h1 : mini_pies = 24)
  (h2 : apples_per_mini_pie = 1/2)
  (h3 : leftover_apples = 36) :
  (mini_pies : ℚ) * apples_per_mini_pie + leftover_apples = 48 := by
  sorry

end NUMINAMATH_CALUDE_ivan_apple_purchase_l148_14817


namespace NUMINAMATH_CALUDE_esme_school_non_pizza_eaters_l148_14811

/-- The number of teachers at Esme's school -/
def num_teachers : ℕ := 30

/-- The number of staff members at Esme's school -/
def num_staff : ℕ := 45

/-- The fraction of teachers who ate pizza -/
def teacher_pizza_fraction : ℚ := 2/3

/-- The fraction of staff members who ate pizza -/
def staff_pizza_fraction : ℚ := 4/5

/-- The total number of non-pizza eaters at Esme's school -/
def non_pizza_eaters : ℕ := 19

theorem esme_school_non_pizza_eaters :
  (num_teachers - (num_teachers : ℚ) * teacher_pizza_fraction).floor +
  (num_staff - (num_staff : ℚ) * staff_pizza_fraction).floor = non_pizza_eaters := by
  sorry

end NUMINAMATH_CALUDE_esme_school_non_pizza_eaters_l148_14811


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l148_14891

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (angle_a angle_b angle_c : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (positive_angles : 0 < angle_a ∧ 0 < angle_b ∧ 0 < angle_c)
  (angle_sum : angle_a + angle_b + angle_c = Real.pi)
  (law_of_sines : a / Real.sin angle_a = b / Real.sin angle_b)

-- State the theorem
theorem triangle_angle_relation (t : Triangle) 
  (h : t.angle_a = 3 * t.angle_b) :
  (t.a^2 - t.b^2) * (t.a - t.b) = t.b * t.c^2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_relation_l148_14891


namespace NUMINAMATH_CALUDE_P_equals_Q_l148_14882

def P : Set ℝ := {m | -1 < m ∧ m < 0}

def Q : Set ℝ := {m | ∀ x : ℝ, m*x^2 + 4*m*x - 4 < 0}

theorem P_equals_Q : P = Q := by sorry

end NUMINAMATH_CALUDE_P_equals_Q_l148_14882


namespace NUMINAMATH_CALUDE_at_least_two_black_balls_count_l148_14805

def total_white_balls : ℕ := 6
def total_black_balls : ℕ := 4
def balls_drawn : ℕ := 4

theorem at_least_two_black_balls_count :
  (Finset.sum (Finset.range 3) (λ i => 
    Nat.choose total_black_balls (i + 2) * Nat.choose total_white_balls (balls_drawn - (i + 2)))) = 115 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_black_balls_count_l148_14805


namespace NUMINAMATH_CALUDE_unique_four_letter_product_l148_14861

def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5
  | 'F' => 6 | 'G' => 7 | 'H' => 8 | 'I' => 9 | 'J' => 10
  | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14 | 'O' => 15
  | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25
  | 'Z' => 26
  | _ => 0

def list_product (s : String) : ℕ :=
  s.foldl (fun acc c => acc * letter_value c) 1

def is_valid_four_letter_string (s : String) : Prop :=
  s.length = 4 ∧ s.all (fun c => 'A' ≤ c ∧ c ≤ 'Z')

theorem unique_four_letter_product :
  ∀ s : String, is_valid_four_letter_string s →
    list_product s = list_product "TUVW" →
    s = "TUVW" := by sorry

#check unique_four_letter_product

end NUMINAMATH_CALUDE_unique_four_letter_product_l148_14861


namespace NUMINAMATH_CALUDE_inequality_range_l148_14859

theorem inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 - Real.log x / Real.log m < 0) ↔ 
  m ∈ Set.Icc (1/16) 1 ∧ m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l148_14859


namespace NUMINAMATH_CALUDE_angle_bisector_length_l148_14810

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 5 ∧ BC = 12 ∧ AC = 13

-- Define the angle bisector BE
def angle_bisector (A B C E : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let BE := Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2)
  let AE := Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2)
  let CE := Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2)
  AE / AB = CE / BC

-- Theorem statement
theorem angle_bisector_length 
  (A B C E : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : angle_bisector A B C E) :
  let BE := Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2)
  ∃ m : ℝ, BE = m * Real.sqrt 2 ∧ m = Real.sqrt 138 / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l148_14810


namespace NUMINAMATH_CALUDE_infinite_integers_satisfying_inequality_l148_14894

theorem infinite_integers_satisfying_inequality :
  ∃ (S : Set ℤ), (Set.Infinite S) ∧ 
  (∀ n ∈ S, (Real.sqrt (n + 1 : ℝ) ≤ Real.sqrt (3 * n + 2 : ℝ)) ∧ 
             (Real.sqrt (3 * n + 2 : ℝ) < Real.sqrt (4 * n - 1 : ℝ))) :=
sorry

end NUMINAMATH_CALUDE_infinite_integers_satisfying_inequality_l148_14894


namespace NUMINAMATH_CALUDE_derivative_y_l148_14803

noncomputable def y (x : ℝ) : ℝ := Real.cos x / x

theorem derivative_y (x : ℝ) (hx : x ≠ 0) :
  deriv y x = -((x * Real.sin x + Real.cos x) / x^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_y_l148_14803


namespace NUMINAMATH_CALUDE_work_completion_time_l148_14824

/-- 
Given a group of ladies that can complete a piece of work in 12 days,
prove that a group with twice as many ladies will complete half of the work in 3 days.
-/
theorem work_completion_time (num_ladies : ℕ) (total_work : ℝ) : 
  (num_ladies * 12 : ℝ) * total_work = 12 * total_work →
  ((2 * num_ladies : ℝ) * 3) * (total_work / 2) = 12 * (total_work / 2) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l148_14824


namespace NUMINAMATH_CALUDE_basketball_game_third_quarter_score_l148_14855

/-- Represents the points scored by a team in each quarter -/
structure TeamScore :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if a TeamScore follows a geometric sequence -/
def isGeometric (s : TeamScore) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if a TeamScore follows an arithmetic sequence -/
def isArithmetic (s : TeamScore) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a TeamScore -/
def totalScore (s : TeamScore) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

theorem basketball_game_third_quarter_score :
  ∀ (teamA teamB : TeamScore),
    teamA.q1 = teamB.q1 →                        -- Tied at the end of first quarter
    isGeometric teamA →                          -- Team A follows geometric sequence
    isArithmetic teamB →                         -- Team B follows arithmetic sequence
    totalScore teamA = totalScore teamB + 3 →    -- Team A wins by 3 points
    totalScore teamA ≤ 100 →                     -- Team A's total score ≤ 100
    totalScore teamB ≤ 100 →                     -- Team B's total score ≤ 100
    teamA.q3 + teamB.q3 = 60                     -- Total score in third quarter is 60
  := by sorry

end NUMINAMATH_CALUDE_basketball_game_third_quarter_score_l148_14855


namespace NUMINAMATH_CALUDE_exists_integer_root_polynomial_l148_14804

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Function to evaluate a quadratic polynomial at a given x -/
def evaluate (p : QuadraticPolynomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- Predicate to check if a quadratic polynomial has integer roots -/
def has_integer_roots (p : QuadraticPolynomial) : Prop :=
  ∃ (r₁ r₂ : ℤ), p.a * r₁^2 + p.b * r₁ + p.c = 0 ∧ p.a * r₂^2 + p.b * r₂ + p.c = 0

/-- The main theorem -/
theorem exists_integer_root_polynomial :
  ∃ (p : QuadraticPolynomial),
    p.a = 1 ∧
    (evaluate p (-1) ≤ evaluate ⟨1, 10, 20⟩ (-1) ∧ evaluate p (-1) ≥ evaluate ⟨1, 20, 10⟩ (-1)) ∧
    has_integer_roots p :=
by
  sorry

end NUMINAMATH_CALUDE_exists_integer_root_polynomial_l148_14804


namespace NUMINAMATH_CALUDE_special_triangle_properties_l148_14845

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Properties
  angle_sum : A + B + C = π
  side_angle_relation : (3 * b - c) * Real.cos A - a * Real.cos C = 0
  side_a_value : a = 2 * Real.sqrt 3
  area : 1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 2
  angle_product : Real.sin B * Real.sin C = 2 / 3

/-- Theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  Real.cos t.A = 1 / 3 ∧
  t.b = 3 ∧ t.c = 3 ∧
  Real.tan t.A + Real.tan t.B + Real.tan t.C = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l148_14845


namespace NUMINAMATH_CALUDE_product_of_roots_l148_14834

theorem product_of_roots (a b : ℝ) 
  (ha : a^2 - 4*a + 3 = 0) 
  (hb : b^2 - 4*b + 3 = 0) 
  (hab : a ≠ b) : 
  (a + 1) * (b + 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l148_14834


namespace NUMINAMATH_CALUDE_friend_product_sum_l148_14865

/-- A function representing the product of the first n positive integers -/
def productOfFirstN (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- A proposition stating that for any five natural numbers a, b, c, d, e,
    if the product of the first a numbers equals the sum of the products of
    the first b, c, d, and e numbers, then a must be either 3 or 4 -/
theorem friend_product_sum (a b c d e : ℕ) :
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e) →
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) →
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) →
  productOfFirstN a = productOfFirstN b + productOfFirstN c + productOfFirstN d + productOfFirstN e →
  a = 3 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_friend_product_sum_l148_14865


namespace NUMINAMATH_CALUDE_bracket_six_times_bracket_three_l148_14818

-- Define a function for the square bracket operation
def bracket (x : ℕ) : ℕ :=
  if x % 2 = 0 then
    x / 2 + 1
  else
    2 * x + 1

-- Theorem statement
theorem bracket_six_times_bracket_three : bracket 6 * bracket 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_bracket_six_times_bracket_three_l148_14818


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_2012_l148_14866

def units_digit_cycle : List Nat := [3, 9, 7, 1]

theorem units_digit_of_3_pow_2012 :
  (3^2012 : Nat) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_2012_l148_14866


namespace NUMINAMATH_CALUDE_total_tree_growth_l148_14835

/-- The growth rate of the first tree in meters per day -/
def tree1_growth_rate : ℝ := 1

/-- The growth rate of the second tree in meters per day -/
def tree2_growth_rate : ℝ := 2

/-- The growth rate of the third tree in meters per day -/
def tree3_growth_rate : ℝ := 2

/-- The growth rate of the fourth tree in meters per day -/
def tree4_growth_rate : ℝ := 3

/-- The number of days of growth -/
def days : ℕ := 4

/-- Theorem stating the total growth of four trees in 4 days -/
theorem total_tree_growth :
  tree1_growth_rate * days + tree2_growth_rate * days +
  tree3_growth_rate * days + tree4_growth_rate * days = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_tree_growth_l148_14835


namespace NUMINAMATH_CALUDE_incorrect_deduction_l148_14822

/-- Definition of an exponential function -/
def IsExponentialFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 1 ∧ ∀ x, f x = a^x

/-- Definition of a power function -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, α > 1 ∧ ∀ x, f x = x^α

/-- Definition of an increasing function -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem -/
theorem incorrect_deduction :
  (∀ f : ℝ → ℝ, IsExponentialFunction f → IsIncreasing f) →
  ¬(∀ f : ℝ → ℝ, IsPowerFunction f → IsIncreasing f) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_deduction_l148_14822
