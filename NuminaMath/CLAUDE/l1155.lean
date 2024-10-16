import Mathlib

namespace NUMINAMATH_CALUDE_five_letter_word_count_l1155_115594

/-- The number of letters in the alphabet -/
def alphabet_size : Nat := 26

/-- The number of vowels -/
def vowel_count : Nat := 5

/-- The number of five-letter words that begin and end with the same letter, 
    with the second letter always being a vowel -/
def word_count : Nat := alphabet_size * vowel_count * alphabet_size * alphabet_size

theorem five_letter_word_count : word_count = 87700 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_word_count_l1155_115594


namespace NUMINAMATH_CALUDE_ordered_pairs_count_l1155_115568

theorem ordered_pairs_count : 
  (∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 * p.2 + p.1 = p.2 + 92 ∧ p.1 > 0 ∧ p.2 > 0) 
    (Finset.product (Finset.range 93) (Finset.range 91))).card ∧ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l1155_115568


namespace NUMINAMATH_CALUDE_data_median_and_mode_l1155_115513

def data : List Int := [15, 17, 14, 10, 15, 17, 17, 16, 14, 12]

def median (l : List Int) : ℚ := sorry

def mode (l : List Int) : Int := sorry

theorem data_median_and_mode :
  median data = 14.5 ∧ mode data = 17 := by sorry

end NUMINAMATH_CALUDE_data_median_and_mode_l1155_115513


namespace NUMINAMATH_CALUDE_candy_cost_theorem_l1155_115572

/-- Calculates the cost of purchasing chocolate candies with a bulk discount -/
def calculate_candy_cost (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let boxes_needed := total_candies / candies_per_box
  let total_cost := boxes_needed * cost_per_box
  if total_candies > discount_threshold then
    total_cost * (1 - discount_rate)
  else
    total_cost

/-- The cost of purchasing 450 chocolate candies is $67.5 -/
theorem candy_cost_theorem :
  calculate_candy_cost 30 5 450 300 (1/10) = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_theorem_l1155_115572


namespace NUMINAMATH_CALUDE_austins_change_l1155_115550

/-- The amount of change Austin had left after buying robots --/
def change_left (num_robots : ℕ) (robot_cost tax initial_amount : ℚ) : ℚ :=
  initial_amount - (num_robots * robot_cost + tax)

/-- Theorem stating that Austin's change is $11.53 --/
theorem austins_change :
  change_left 7 8.75 7.22 80 = 11.53 := by
  sorry

end NUMINAMATH_CALUDE_austins_change_l1155_115550


namespace NUMINAMATH_CALUDE_theresa_work_hours_l1155_115501

/-- The average number of hours Theresa needs to work per week -/
def required_average : ℝ := 9

/-- The number of weeks Theresa needs to maintain the average -/
def total_weeks : ℕ := 7

/-- The hours Theresa worked in the first 6 weeks -/
def first_six_weeks : List ℝ := [10, 8, 9, 11, 6, 8]

/-- The sum of hours Theresa worked in the first 6 weeks -/
def sum_first_six : ℝ := first_six_weeks.sum

/-- The number of hours Theresa needs to work in the seventh week -/
def hours_seventh_week : ℝ := 11

theorem theresa_work_hours :
  (sum_first_six + hours_seventh_week) / total_weeks = required_average := by
  sorry

end NUMINAMATH_CALUDE_theresa_work_hours_l1155_115501


namespace NUMINAMATH_CALUDE_crow_percentage_among_non_pigeons_l1155_115558

theorem crow_percentage_among_non_pigeons (total_birds : ℝ) (crow_percentage : ℝ) (pigeon_percentage : ℝ)
  (h1 : crow_percentage = 40)
  (h2 : pigeon_percentage = 20)
  (h3 : 0 < total_birds) :
  (crow_percentage / (100 - pigeon_percentage)) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_crow_percentage_among_non_pigeons_l1155_115558


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1155_115544

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 1) :
  1 / a + 1 / b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1155_115544


namespace NUMINAMATH_CALUDE_max_value_on_triangle_vertices_l1155_115515

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a triangle in 2D space
structure Triangle where
  P : Point2D
  Q : Point2D
  R : Point2D

-- Define a linear function f(x, y) = ax + by + c
def linearFunction (a b c : ℝ) (p : Point2D) : ℝ :=
  a * p.x + b * p.y + c

-- Define a predicate to check if a point is in or on a triangle
def isInOrOnTriangle (t : Triangle) (p : Point2D) : Prop :=
  sorry -- The actual implementation is not needed for the theorem statement

-- Theorem statement
theorem max_value_on_triangle_vertices 
  (t : Triangle) (a b c : ℝ) (p : Point2D) 
  (h : isInOrOnTriangle t p) : 
  linearFunction a b c p ≤ max 
    (linearFunction a b c t.P) 
    (max (linearFunction a b c t.Q) (linearFunction a b c t.R)) := by
  sorry


end NUMINAMATH_CALUDE_max_value_on_triangle_vertices_l1155_115515


namespace NUMINAMATH_CALUDE_probability_receive_one_l1155_115542

/-- Probability of receiving a signal as 1 in a digital communication system with given error rates --/
theorem probability_receive_one (p_receive_zero_given_send_zero : ℝ)
                                (p_receive_one_given_send_zero : ℝ)
                                (p_receive_one_given_send_one : ℝ)
                                (p_receive_zero_given_send_one : ℝ)
                                (p_send_zero : ℝ)
                                (p_send_one : ℝ)
                                (h1 : p_receive_zero_given_send_zero = 0.9)
                                (h2 : p_receive_one_given_send_zero = 0.1)
                                (h3 : p_receive_one_given_send_one = 0.95)
                                (h4 : p_receive_zero_given_send_one = 0.05)
                                (h5 : p_send_zero = 0.5)
                                (h6 : p_send_one = 0.5) :
  p_send_zero * p_receive_one_given_send_zero + p_send_one * p_receive_one_given_send_one = 0.525 :=
by sorry

end NUMINAMATH_CALUDE_probability_receive_one_l1155_115542


namespace NUMINAMATH_CALUDE_product_mod_23_l1155_115578

theorem product_mod_23 :
  (2003 * 2004 * 2005 * 2006 * 2007 * 2008) % 23 = 3 := by sorry

end NUMINAMATH_CALUDE_product_mod_23_l1155_115578


namespace NUMINAMATH_CALUDE_condo_floors_count_l1155_115506

/-- Represents a condo development with regular and penthouse floors. -/
structure CondoDevelopment where
  regularUnitsPerFloor : ℕ
  penthouseUnitsPerFloor : ℕ
  penthouseFloors : ℕ
  totalUnits : ℕ

/-- Calculates the total number of floors in a condo development. -/
def totalFloors (c : CondoDevelopment) : ℕ :=
  let regularFloors := (c.totalUnits - c.penthouseFloors * c.penthouseUnitsPerFloor) / c.regularUnitsPerFloor
  regularFloors + c.penthouseFloors

/-- Theorem stating that a condo development with the given specifications has 23 floors. -/
theorem condo_floors_count (c : CondoDevelopment) 
    (h1 : c.regularUnitsPerFloor = 12)
    (h2 : c.penthouseUnitsPerFloor = 2)
    (h3 : c.penthouseFloors = 2)
    (h4 : c.totalUnits = 256) : 
  totalFloors c = 23 := by
  sorry

end NUMINAMATH_CALUDE_condo_floors_count_l1155_115506


namespace NUMINAMATH_CALUDE_points_in_different_half_spaces_l1155_115535

/-- A plane in 3D space defined by the equation ax + by + cz + d = 0 --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Determine if two points are on opposite sides of a plane --/
def oppositeHalfSpaces (plane : Plane) (p1 p2 : Point3D) : Prop :=
  (plane.a * p1.x + plane.b * p1.y + plane.c * p1.z + plane.d) *
  (plane.a * p2.x + plane.b * p2.y + plane.c * p2.z + plane.d) < 0

theorem points_in_different_half_spaces :
  let plane := Plane.mk 1 2 3 0
  let point1 := Point3D.mk 1 2 (-2)
  let point2 := Point3D.mk 2 1 (-1)
  oppositeHalfSpaces plane point1 point2 := by
  sorry


end NUMINAMATH_CALUDE_points_in_different_half_spaces_l1155_115535


namespace NUMINAMATH_CALUDE_triangle_count_is_36_l1155_115584

/-- A hexagon with diagonals and midpoint segments -/
structure HexagonWithDiagonalsAndMidpoints :=
  (vertices : Fin 6 → Point)
  (diagonals : List (Point × Point))
  (midpoint_segments : List (Point × Point))

/-- Count of triangles in the hexagon figure -/
def count_triangles (h : HexagonWithDiagonalsAndMidpoints) : ℕ :=
  sorry

/-- Theorem stating that the count of triangles is 36 -/
theorem triangle_count_is_36 (h : HexagonWithDiagonalsAndMidpoints) : 
  count_triangles h = 36 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_is_36_l1155_115584


namespace NUMINAMATH_CALUDE_xiaoqiang_father_annual_income_l1155_115575

def monthly_salary : ℕ := 4380
def months_in_year : ℕ := 12

theorem xiaoqiang_father_annual_income :
  monthly_salary * months_in_year = 52560 := by sorry

end NUMINAMATH_CALUDE_xiaoqiang_father_annual_income_l1155_115575


namespace NUMINAMATH_CALUDE_ellipse_problem_l1155_115557

/-- The ellipse problem -/
theorem ellipse_problem (a b : ℝ) (P : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧ a > b →  -- a and b are positive real numbers with a > b
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) →  -- P is on the ellipse
  ((P.1 + 1)^2 + P.2^2)^(1/2) - ((P.1 - 1)^2 + P.2^2)^(1/2) = a / 2 →  -- |PF₁| - |PF₂| = a/2
  (P.1 - 1) * (P.1 + 1) + P.2^2 = 0 →  -- PF₂ is perpendicular to F₁F₂
  (∃ (m : ℝ), (1 + m * P.2)^2 / 4 + P.2^2 / 3 = 1 ∧  -- equation of ellipse G
              (∃ (M N : ℝ × ℝ), M ≠ N ∧  -- M and N are distinct points
                (M.1^2 / 4 + M.2^2 / 3 = 1) ∧ (N.1^2 / 4 + N.2^2 / 3 = 1) ∧  -- M and N are on the ellipse
                (M.1 - 1 = m * M.2) ∧ (N.1 - 1 = m * N.2) ∧  -- M and N are on line l passing through F₂
                ((0 - M.2) * (N.1 - 1)) / ((0 - N.2) * (M.1 - 1)) = 2))  -- ratio of areas of triangles BF₂M and BF₂N is 2
  := by sorry


end NUMINAMATH_CALUDE_ellipse_problem_l1155_115557


namespace NUMINAMATH_CALUDE_fold_point_set_area_l1155_115525

/-- Triangle DEF with given side lengths and right angle -/
structure RightTriangle where
  de : ℝ
  df : ℝ
  angle_e_is_right : de^2 + ef^2 = df^2
  de_length : de = 24
  df_length : df = 48

/-- Set of fold points in the triangle -/
def FoldPointSet (t : RightTriangle) : Set (ℝ × ℝ) := sorry

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Main theorem: Area of fold point set -/
theorem fold_point_set_area (t : RightTriangle) :
  area (FoldPointSet t) = 156 * Real.pi - 144 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_fold_point_set_area_l1155_115525


namespace NUMINAMATH_CALUDE_weight_of_b_l1155_115555

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 43)
  (h2 : (a + b) / 2 = 48)
  (h3 : (b + c) / 2 = 42) :
  b = 51 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l1155_115555


namespace NUMINAMATH_CALUDE_interest_earned_proof_l1155_115560

def initial_investment : ℝ := 1200
def annual_interest_rate : ℝ := 0.12
def compounding_periods : ℕ := 4

def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

theorem interest_earned_proof :
  let final_amount := compound_interest initial_investment annual_interest_rate compounding_periods
  let total_interest := final_amount - initial_investment
  ∃ ε > 0, |total_interest - 688.22| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_earned_proof_l1155_115560


namespace NUMINAMATH_CALUDE_all_conditions_imply_right_triangle_l1155_115597

structure Triangle (A B C : ℝ) :=
  (angle_sum : A + B + C = 180)

def is_right_triangle (t : Triangle A B C) : Prop :=
  A = 90 ∨ B = 90 ∨ C = 90

theorem all_conditions_imply_right_triangle 
  (t : Triangle A B C) : 
  (A + B = C) ∨ 
  (∃ (k : ℝ), k > 0 ∧ A = k ∧ B = 2*k ∧ C = 3*k) ∨ 
  (A = 90 - B) ∨ 
  (A = B - C) → 
  is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_all_conditions_imply_right_triangle_l1155_115597


namespace NUMINAMATH_CALUDE_article_price_calculation_l1155_115505

theorem article_price_calculation (initial_price : ℝ) : 
  let after_first_discount := initial_price * 0.75
  let after_second_discount := after_first_discount * 0.85
  let after_increase := after_second_discount * 1.1
  let final_price := after_increase * 1.05
  final_price = 1226.25 → initial_price = 1843.75 := by
sorry

end NUMINAMATH_CALUDE_article_price_calculation_l1155_115505


namespace NUMINAMATH_CALUDE_probability_even_greater_than_10_l1155_115573

def ball_set : Finset ℕ := {1, 2, 3, 4, 5}

def is_valid_product (a b : ℕ) : Bool :=
  Even (a * b) ∧ a * b > 10

def valid_outcomes : Finset (ℕ × ℕ) :=
  ball_set.product ball_set

def favorable_outcomes : Finset (ℕ × ℕ) :=
  valid_outcomes.filter (fun p => is_valid_product p.1 p.2)

theorem probability_even_greater_than_10 :
  (favorable_outcomes.card : ℚ) / valid_outcomes.card = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_even_greater_than_10_l1155_115573


namespace NUMINAMATH_CALUDE_cycle_loss_percentage_l1155_115532

/-- Calculate the percentage of loss given the cost price and selling price -/
def percentage_loss (cost_price selling_price : ℕ) : ℚ :=
  (cost_price - selling_price : ℚ) / cost_price * 100

theorem cycle_loss_percentage :
  let cost_price := 2000
  let selling_price := 1800
  percentage_loss cost_price selling_price = 10 := by
sorry

end NUMINAMATH_CALUDE_cycle_loss_percentage_l1155_115532


namespace NUMINAMATH_CALUDE_ricardo_coin_value_difference_l1155_115590

/-- The total number of coins Ricardo has -/
def total_coins : ℕ := 3030

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculate the total value in cents given the number of pennies -/
def total_value (num_pennies : ℕ) : ℕ :=
  num_pennies * penny_value + (total_coins - num_pennies) * nickel_value

/-- The minimum number of pennies Ricardo can have -/
def min_pennies : ℕ := 1

/-- The maximum number of pennies Ricardo can have -/
def max_pennies : ℕ := total_coins - 1

theorem ricardo_coin_value_difference :
  (total_value min_pennies) - (total_value max_pennies) = 12112 := by
  sorry

end NUMINAMATH_CALUDE_ricardo_coin_value_difference_l1155_115590


namespace NUMINAMATH_CALUDE_probability_D_given_E_l1155_115574

-- Define the regions D and E
def region_D (x y : ℝ) : Prop := y ≤ 1 ∧ y ≥ x^2
def region_E (x y : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Define the areas of regions D and E
noncomputable def area_D : ℝ := 4/3
noncomputable def area_E : ℝ := 2

-- State the theorem
theorem probability_D_given_E : 
  (area_D / area_E) = 2/3 :=
sorry

end NUMINAMATH_CALUDE_probability_D_given_E_l1155_115574


namespace NUMINAMATH_CALUDE_blanch_breakfast_slices_l1155_115576

/-- The number of pizza slices Blanch ate for breakfast -/
def breakfast_slices : ℕ := sorry

/-- The total number of pizza slices Blanch started with -/
def total_slices : ℕ := 15

/-- The number of pizza slices Blanch ate for lunch -/
def lunch_slices : ℕ := 2

/-- The number of pizza slices Blanch ate as a snack -/
def snack_slices : ℕ := 2

/-- The number of pizza slices Blanch ate for dinner -/
def dinner_slices : ℕ := 5

/-- The number of pizza slices left at the end -/
def leftover_slices : ℕ := 2

/-- Theorem stating that Blanch ate 4 slices for breakfast -/
theorem blanch_breakfast_slices : 
  breakfast_slices = total_slices - (lunch_slices + snack_slices + dinner_slices + leftover_slices) :=
by sorry

end NUMINAMATH_CALUDE_blanch_breakfast_slices_l1155_115576


namespace NUMINAMATH_CALUDE_attendance_probability_additional_A_tickets_needed_l1155_115508

def total_students : ℕ := 50
def tickets_A : ℕ := 3
def tickets_B : ℕ := 7
def tickets_C : ℕ := 10

def total_tickets : ℕ := tickets_A + tickets_B + tickets_C

theorem attendance_probability :
  (total_tickets : ℚ) / total_students = 2 / 5 := by sorry

theorem additional_A_tickets_needed (x : ℕ) :
  (tickets_A + x : ℚ) / total_students = 1 / 5 → x = 7 := by sorry

end NUMINAMATH_CALUDE_attendance_probability_additional_A_tickets_needed_l1155_115508


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1155_115587

/-- Given an arithmetic sequence where the first term is 2/3 and the 17th term is 5/6,
    the 9th term is 3/4. -/
theorem arithmetic_sequence_ninth_term 
  (a : ℚ) 
  (seq : ℕ → ℚ) 
  (h1 : seq 1 = 2/3) 
  (h2 : seq 17 = 5/6) 
  (h3 : ∀ n : ℕ, seq (n + 1) - seq n = seq 2 - seq 1) : 
  seq 9 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1155_115587


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_l1155_115588

/-- Represents the daily rental business for canoes and kayaks. -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_count : ℕ
  kayak_count : ℕ

/-- The conditions of the rental business problem. -/
def rental_problem : RentalBusiness where
  canoe_price := 9
  kayak_price := 12
  canoe_count := 24  -- We know this from the solution, but it's derived from the conditions
  kayak_count := 18  -- We know this from the solution, but it's derived from the conditions

/-- The theorem stating the difference between canoes and kayaks rented. -/
theorem canoe_kayak_difference (b : RentalBusiness) 
  (h1 : b.canoe_price = 9)
  (h2 : b.kayak_price = 12)
  (h3 : 4 * b.kayak_count = 3 * b.canoe_count)
  (h4 : b.canoe_price * b.canoe_count + b.kayak_price * b.kayak_count = 432) :
  b.canoe_count - b.kayak_count = 6 := by
  sorry

#eval rental_problem.canoe_count - rental_problem.kayak_count

end NUMINAMATH_CALUDE_canoe_kayak_difference_l1155_115588


namespace NUMINAMATH_CALUDE_binomial_coefficient_product_l1155_115553

theorem binomial_coefficient_product (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_product_l1155_115553


namespace NUMINAMATH_CALUDE_tangent_line_condition_l1155_115596

/-- The curve function f(x) = x³ - 3ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem tangent_line_condition (a : ℝ) :
  (∀ b : ℝ, ¬∃ x : ℝ, f a x = -x + b ∧ f_derivative a x = -1) →
  a < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l1155_115596


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1155_115563

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 1 → x - 1 ≥ Real.log x) ↔ (∃ x : ℝ, x > 1 ∧ x - 1 < Real.log x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1155_115563


namespace NUMINAMATH_CALUDE_paulson_spending_percentage_l1155_115581

theorem paulson_spending_percentage 
  (income_increase : Real) 
  (expenditure_increase : Real) 
  (savings_increase : Real) : 
  income_increase = 0.20 → 
  expenditure_increase = 0.10 → 
  savings_increase = 0.50 → 
  ∃ (original_income : Real) (spending_percentage : Real),
    spending_percentage = 0.75 ∧ 
    original_income > 0 ∧
    (1 + income_increase) * original_income - 
    (1 + expenditure_increase) * spending_percentage * original_income = 
    (1 + savings_increase) * (original_income - spending_percentage * original_income) :=
by sorry

end NUMINAMATH_CALUDE_paulson_spending_percentage_l1155_115581


namespace NUMINAMATH_CALUDE_complex_vector_sum_l1155_115598

theorem complex_vector_sum (z₁ z₂ z₃ : ℂ) (x y : ℝ) 
  (h₁ : z₁ = -1 + I)
  (h₂ : z₂ = 1 + I)
  (h₃ : z₃ = 1 + 4*I)
  (h₄ : z₃ = x • z₁ + y • z₂) :
  x + y = 4 := by sorry

end NUMINAMATH_CALUDE_complex_vector_sum_l1155_115598


namespace NUMINAMATH_CALUDE_addition_to_reach_81_l1155_115517

theorem addition_to_reach_81 : 5 * 12 / (180 / 3) + 80 = 81 := by
  sorry

end NUMINAMATH_CALUDE_addition_to_reach_81_l1155_115517


namespace NUMINAMATH_CALUDE_roots_real_implies_ab_nonpositive_l1155_115541

/-- The polynomial x^4 + ax^3 + bx + c has all real roots -/
def has_all_real_roots (a b c : ℝ) : Prop :=
  ∀ x : ℂ, x^4 + a*x^3 + b*x + c = 0 → x.im = 0

/-- If all roots of the polynomial x^4 + ax^3 + bx + c are real numbers, then ab ≤ 0 -/
theorem roots_real_implies_ab_nonpositive (a b c : ℝ) :
  has_all_real_roots a b c → a * b ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_roots_real_implies_ab_nonpositive_l1155_115541


namespace NUMINAMATH_CALUDE_fraction_equality_l1155_115579

theorem fraction_equality (a b : ℝ) (h : b / a = 3 / 5) : (a - b) / a = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1155_115579


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1155_115599

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := 4 * x^2 + b * x + c

-- Theorem statement
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (f b c (-1) = -1 ∧ f b c 0 = 0) →
  (∃ x₁ x₂ : ℝ, f b c x₁ = 20 ∧ f b c x₂ = 20 ∧ f b c (x₁ + x₂) = 0) →
  (∀ x : ℝ, (x < -5/4 ∨ x > 0) → f b c x > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1155_115599


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l1155_115595

/-- The equation of the common chord of two intersecting circles -/
theorem common_chord_of_circles (x y : ℝ) : 
  (x^2 + y^2 = 10) ∧ ((x-1)^2 + (y-3)^2 = 10) → x + 3*y - 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l1155_115595


namespace NUMINAMATH_CALUDE_sample_size_example_l1155_115577

/-- Represents the sample size of a survey --/
def sample_size (population : ℕ) (selected : ℕ) : ℕ := selected

/-- Theorem stating that for a population of 300 students with 50 selected, the sample size is 50 --/
theorem sample_size_example : sample_size 300 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_example_l1155_115577


namespace NUMINAMATH_CALUDE_octal_minus_base9_equals_19559_l1155_115524

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem octal_minus_base9_equals_19559 : 
  let octal := [5, 4, 3, 2, 1]
  let base9 := [4, 3, 2, 1]
  base_to_decimal octal 8 - base_to_decimal base9 9 = 19559 := by
  sorry

end NUMINAMATH_CALUDE_octal_minus_base9_equals_19559_l1155_115524


namespace NUMINAMATH_CALUDE_donnas_truck_dryers_l1155_115534

/-- Calculates the number of dryers on Donna's truck given the weight constraints --/
theorem donnas_truck_dryers :
  let bridge_limit : ℕ := 20000
  let empty_truck_weight : ℕ := 12000
  let num_soda_crates : ℕ := 20
  let soda_crate_weight : ℕ := 50
  let dryer_weight : ℕ := 3000
  let loaded_truck_weight : ℕ := 24000
  let soda_weight : ℕ := num_soda_crates * soda_crate_weight
  let produce_weight : ℕ := 2 * soda_weight
  let truck_with_soda_produce : ℕ := empty_truck_weight + soda_weight + produce_weight
  let dryers_weight : ℕ := loaded_truck_weight - truck_with_soda_produce
  let num_dryers : ℕ := dryers_weight / dryer_weight
  num_dryers = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_donnas_truck_dryers_l1155_115534


namespace NUMINAMATH_CALUDE_equation_one_solution_l1155_115518

theorem equation_one_solution :
  ∃ x₁ x₂ : ℝ, (3 * x₁^2 - 9 = 0) ∧ (3 * x₂^2 - 9 = 0) ∧ (x₁ = Real.sqrt 3) ∧ (x₂ = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_one_solution_l1155_115518


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1155_115537

def quadratic_function (a m b x : ℝ) := a * x * (x - m) + b

theorem quadratic_function_properties
  (a m b : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_y_1_at_0 : quadratic_function a m b 0 = 1)
  (h_y_1_at_2 : quadratic_function a m b 2 = 1)
  (h_y_gt_4_at_3 : quadratic_function a m b 3 > 4)
  (k : ℝ)
  (h_passes_1_k : quadratic_function a m b 1 = k)
  (h_k_over_a : 0 < k / a ∧ k / a < 1) :
  m = 2 ∧ b = 1 ∧ a > 1 ∧ 1/2 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1155_115537


namespace NUMINAMATH_CALUDE_circle_equation_l1155_115556

/-- Theorem: The equation of a circle with center (1, -1) and radius 2 is (x-1)^2 + (y+1)^2 = 4 -/
theorem circle_equation (x y : ℝ) : 
  (∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (1, -1) ∧ 
    radius = 2 ∧ 
    ((x - center.1)^2 + (y - center.2)^2 = radius^2)) ↔ 
  ((x - 1)^2 + (y + 1)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1155_115556


namespace NUMINAMATH_CALUDE_train_length_calculation_l1155_115559

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : Real) (crossing_time : Real) (bridge_length : Real) :
  train_speed = 54 * (1000 / 3600) →
  crossing_time = 16.13204276991174 →
  bridge_length = 132 →
  train_speed * crossing_time - bridge_length = 109.9806415486761 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1155_115559


namespace NUMINAMATH_CALUDE_milk_replacement_problem_l1155_115585

theorem milk_replacement_problem (initial_volume : ℝ) (final_pure_milk : ℝ) :
  initial_volume = 45 ∧ final_pure_milk = 28.8 →
  ∃ (x : ℝ), x = 9 ∧ 
  (initial_volume - x) * (initial_volume - x) / initial_volume = final_pure_milk :=
by sorry

end NUMINAMATH_CALUDE_milk_replacement_problem_l1155_115585


namespace NUMINAMATH_CALUDE_min_max_sum_l1155_115589

theorem min_max_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) (h_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 300) : 
  (max (x₁ + x₂) (max (x₂ + x₃) (max (x₃ + x₄) (x₄ + x₅)))) ≥ 100 ∧ 
  ∃ (y₁ y₂ y₃ y₄ y₅ : ℝ), y₁ ≥ 0 ∧ y₂ ≥ 0 ∧ y₃ ≥ 0 ∧ y₄ ≥ 0 ∧ y₅ ≥ 0 ∧ 
  y₁ + y₂ + y₃ + y₄ + y₅ = 300 ∧ 
  max (y₁ + y₂) (max (y₂ + y₃) (max (y₃ + y₄) (y₄ + y₅))) = 100 :=
by sorry

end NUMINAMATH_CALUDE_min_max_sum_l1155_115589


namespace NUMINAMATH_CALUDE_carl_has_more_stamps_l1155_115564

/-- Given that Carl has 89 stamps and Kevin has 57 stamps, 
    prove that Carl has 32 more stamps than Kevin. -/
theorem carl_has_more_stamps (carl_stamps : ℕ) (kevin_stamps : ℕ) 
  (h1 : carl_stamps = 89) (h2 : kevin_stamps = 57) : 
  carl_stamps - kevin_stamps = 32 := by
  sorry

end NUMINAMATH_CALUDE_carl_has_more_stamps_l1155_115564


namespace NUMINAMATH_CALUDE_survey_total_is_120_l1155_115509

/-- Represents the survey results of parents' ratings on their children's online class experience -/
structure SurveyResults where
  total : ℕ
  excellent : ℕ
  verySatisfactory : ℕ
  satisfactory : ℕ
  needsImprovement : ℕ

/-- The conditions of the survey results -/
def surveyConditions (s : SurveyResults) : Prop :=
  s.excellent = (15 * s.total) / 100 ∧
  s.verySatisfactory = (60 * s.total) / 100 ∧
  s.satisfactory = (80 * (s.total - s.excellent - s.verySatisfactory)) / 100 ∧
  s.needsImprovement = s.total - s.excellent - s.verySatisfactory - s.satisfactory ∧
  s.needsImprovement = 6

/-- Theorem stating that the total number of parents who answered the survey is 120 -/
theorem survey_total_is_120 (s : SurveyResults) (h : surveyConditions s) : s.total = 120 := by
  sorry

end NUMINAMATH_CALUDE_survey_total_is_120_l1155_115509


namespace NUMINAMATH_CALUDE_vector_properties_l1155_115528

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, 1]
def c : Fin 2 → ℝ := ![3, -5]

theorem vector_properties :
  (∃ (k : ℝ), a + 2 • b = k • c) ∧
  ‖a + c‖ = 2 * ‖b‖ := by
sorry

end NUMINAMATH_CALUDE_vector_properties_l1155_115528


namespace NUMINAMATH_CALUDE_evaluate_expression_l1155_115562

theorem evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = 2) : y^2 * (y - 2*x) = 16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1155_115562


namespace NUMINAMATH_CALUDE_correct_factorization_l1155_115546

theorem correct_factorization (x y : ℝ) : x * (x - y) - y * (x - y) = (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1155_115546


namespace NUMINAMATH_CALUDE_min_distance_sum_to_points_l1155_115567

/-- The minimum distance sum from a point on the line x-y+1=0 to points (0,0) and (1,1) is 2 -/
theorem min_distance_sum_to_points : ∃ (min_dist : ℝ),
  min_dist = 2 ∧
  ∀ (P : ℝ × ℝ), 
    P.1 - P.2 + 1 = 0 → -- P is on the line x-y+1=0
    Real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) + Real.sqrt ((P.1 - 1)^2 + (P.2 - 1)^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_to_points_l1155_115567


namespace NUMINAMATH_CALUDE_typing_time_proof_l1155_115519

def original_speed : ℕ := 212
def speed_reduction : ℕ := 40
def document_length : ℕ := 3440

theorem typing_time_proof :
  let new_speed := original_speed - speed_reduction
  document_length / new_speed = 20 := by sorry

end NUMINAMATH_CALUDE_typing_time_proof_l1155_115519


namespace NUMINAMATH_CALUDE_rollo_guinea_pigs_food_l1155_115554

/-- The amount of food needed to feed all guinea pigs -/
def total_food (first_pig_food second_pig_food third_pig_food : ℕ) : ℕ :=
  first_pig_food + second_pig_food + third_pig_food

/-- Theorem stating the total amount of food needed for Rollo's guinea pigs -/
theorem rollo_guinea_pigs_food :
  ∃ (first_pig_food second_pig_food third_pig_food : ℕ),
    first_pig_food = 2 ∧
    second_pig_food = 2 * first_pig_food ∧
    third_pig_food = second_pig_food + 3 ∧
    total_food first_pig_food second_pig_food third_pig_food = 13 :=
by
  sorry

#check rollo_guinea_pigs_food

end NUMINAMATH_CALUDE_rollo_guinea_pigs_food_l1155_115554


namespace NUMINAMATH_CALUDE_smallest_multiple_square_l1155_115583

theorem smallest_multiple_square (a : ℕ) : 
  (∃ k : ℕ, a = 6 * k) ∧ 
  (∃ m : ℕ, a = 15 * m) ∧ 
  (∃ n : ℕ, a = n * n) ∧ 
  (∀ b : ℕ, b > 0 ∧ 
    (∃ k : ℕ, b = 6 * k) ∧ 
    (∃ m : ℕ, b = 15 * m) ∧ 
    (∃ n : ℕ, b = n * n) → 
    a ≤ b) → 
  a = 900 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_square_l1155_115583


namespace NUMINAMATH_CALUDE_lines_not_form_triangle_l1155_115512

/-- A line in the xy-plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The three lines in the problem -/
def l1 (m : ℝ) : Line := ⟨3, m, -1⟩
def l2 : Line := ⟨3, -2, -5⟩
def l3 : Line := ⟨6, 1, -5⟩

/-- Theorem stating the conditions under which the three lines cannot form a triangle -/
theorem lines_not_form_triangle (m : ℝ) : 
  (¬(∃ x y : ℝ, 3*x + m*y - 1 = 0 ∧ 3*x - 2*y - 5 = 0 ∧ 6*x + y - 5 = 0)) ↔ 
  (m = -2 ∨ m = 1/2) :=
sorry

end NUMINAMATH_CALUDE_lines_not_form_triangle_l1155_115512


namespace NUMINAMATH_CALUDE_unique_fifth_power_solution_l1155_115523

theorem unique_fifth_power_solution :
  ∀ x y : ℕ, x^5 = y^5 + 10*y^2 + 20*y + 1 → (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_fifth_power_solution_l1155_115523


namespace NUMINAMATH_CALUDE_remainder_problem_l1155_115502

theorem remainder_problem (R : ℕ) : 
  (29 = Nat.gcd (1255 - 8) (1490 - R)) →
  (1255 % 29 = 8) →
  (1490 % 29 = R) →
  R = 11 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1155_115502


namespace NUMINAMATH_CALUDE_puzzle_completion_time_l1155_115514

/-- Calculates the time to complete puzzles given the number of puzzles, pieces per puzzle, and completion rate. -/
def time_to_complete_puzzles (num_puzzles : ℕ) (pieces_per_puzzle : ℕ) (pieces_per_interval : ℕ) (interval_minutes : ℕ) : ℕ :=
  let total_pieces := num_puzzles * pieces_per_puzzle
  let pieces_per_minute := pieces_per_interval / interval_minutes
  total_pieces / pieces_per_minute

/-- Proves that completing 2 puzzles of 2000 pieces each at a rate of 100 pieces per 10 minutes takes 400 minutes. -/
theorem puzzle_completion_time :
  time_to_complete_puzzles 2 2000 100 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_completion_time_l1155_115514


namespace NUMINAMATH_CALUDE_constant_k_value_l1155_115592

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4) → k = -16 := by
  sorry

end NUMINAMATH_CALUDE_constant_k_value_l1155_115592


namespace NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l1155_115500

-- Part 1: Equation solution
theorem equation_solution :
  ∃ x : ℝ, (2 * x / (x - 2) + 3 / (2 - x) = 1) ∧ (x = 1) := by sorry

-- Part 2: System of inequalities solution
theorem inequalities_solution :
  ∃ x : ℝ, (2 * x - 1 ≥ 3 * (x - 1)) ∧
           ((5 - x) / 2 < x + 3) ∧
           (-1/3 < x) ∧ (x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l1155_115500


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1155_115536

theorem sqrt_equation_solution (x y : ℝ) :
  Real.sqrt (x^2 + y^2 - 1) = 1 - x - y ↔ (x = 1 ∧ y ≤ 0) ∨ (y = 1 ∧ x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1155_115536


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1155_115510

theorem units_digit_of_expression : 
  (30 * 32 * 34 * 36 * 38 * 40) / 2000 ≡ 6 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1155_115510


namespace NUMINAMATH_CALUDE_exactly_one_integer_satisfies_condition_l1155_115526

theorem exactly_one_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 20 - 5 * n ≥ 15 := by sorry

end NUMINAMATH_CALUDE_exactly_one_integer_satisfies_condition_l1155_115526


namespace NUMINAMATH_CALUDE_sophie_chocolates_l1155_115530

theorem sophie_chocolates :
  ∃ (x : ℕ), x ≥ 150 ∧ x % 15 = 7 ∧ ∀ (y : ℕ), y ≥ 150 ∧ y % 15 = 7 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_sophie_chocolates_l1155_115530


namespace NUMINAMATH_CALUDE_prism_with_27_edges_has_11_faces_l1155_115593

/-- A prism is a polyhedron with two congruent and parallel faces (called bases) 
    and whose other faces (called lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ
  lateral_faces : ℕ
  base_edges : ℕ

/-- The number of edges in a prism is equal to 3 times the number of lateral faces. -/
axiom prism_edge_count (p : Prism) : p.edges = 3 * p.lateral_faces

/-- The number of edges in each base of a prism is equal to the number of lateral faces. -/
axiom prism_base_edge_count (p : Prism) : p.base_edges = p.lateral_faces

/-- The total number of faces in a prism is equal to the number of lateral faces plus 2 (for the bases). -/
def total_faces (p : Prism) : ℕ := p.lateral_faces + 2

/-- Theorem: A prism with 27 edges has 11 faces. -/
theorem prism_with_27_edges_has_11_faces (p : Prism) (h : p.edges = 27) : total_faces p = 11 := by
  sorry


end NUMINAMATH_CALUDE_prism_with_27_edges_has_11_faces_l1155_115593


namespace NUMINAMATH_CALUDE_min_value_theorem_l1155_115511

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.sqrt (b / a) + Real.sqrt (a / b) - 2 = (Real.sqrt (a * b) - 4 * a * b) / (2 * a * b)) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 2 + 6 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → 
  Real.sqrt (y / x) + Real.sqrt (x / y) - 2 = (Real.sqrt (x * y) - 4 * x * y) / (2 * x * y) →
  1 / x + 2 / y ≥ min := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1155_115511


namespace NUMINAMATH_CALUDE_distance_B_to_x_axis_l1155_115527

def point_B : ℝ × ℝ := (2, -3)

def distance_to_x_axis (p : ℝ × ℝ) : ℝ := |p.2|

theorem distance_B_to_x_axis :
  distance_to_x_axis point_B = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_B_to_x_axis_l1155_115527


namespace NUMINAMATH_CALUDE_madmen_count_l1155_115591

/-- The number of madmen in a psychiatric hospital -/
def num_madmen : ℕ := 20

/-- The number of bites the chief doctor received -/
def chief_doctor_bites : ℕ := 100

/-- Theorem stating the number of madmen in the hospital -/
theorem madmen_count :
  num_madmen = 20 ∧
  (7 * num_madmen = 2 * num_madmen + chief_doctor_bites) :=
by sorry

end NUMINAMATH_CALUDE_madmen_count_l1155_115591


namespace NUMINAMATH_CALUDE_vessel_volume_ratio_l1155_115538

/-- Represents a vessel containing a mixture of milk and water -/
structure Vessel where
  milk : ℚ
  water : ℚ

/-- The ratio of milk to water in a vessel -/
def milkWaterRatio (v : Vessel) : ℚ := v.milk / v.water

/-- The total volume of a vessel -/
def volume (v : Vessel) : ℚ := v.milk + v.water

/-- Combines the contents of two vessels -/
def combineVessels (v1 v2 : Vessel) : Vessel :=
  { milk := v1.milk + v2.milk, water := v1.water + v2.water }

theorem vessel_volume_ratio (v1 v2 : Vessel) :
  milkWaterRatio v1 = 1/2 →
  milkWaterRatio v2 = 6/4 →
  milkWaterRatio (combineVessels v1 v2) = 1 →
  volume v1 / volume v2 = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_vessel_volume_ratio_l1155_115538


namespace NUMINAMATH_CALUDE_lines_parallel_iff_l1155_115580

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + 2 * a = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ (x y : ℝ), line1 a x y ↔ ∃ (k : ℝ), line2 a (x + k) (y + k)

-- Theorem statement
theorem lines_parallel_iff : ∀ (a : ℝ), parallel a ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_l1155_115580


namespace NUMINAMATH_CALUDE_cubic_root_product_l1155_115521

theorem cubic_root_product (u v w : ℝ) : 
  (u^3 - 15*u^2 + 13*u - 6 = 0) →
  (v^3 - 15*v^2 + 13*v - 6 = 0) →
  (w^3 - 15*w^2 + 13*w - 6 = 0) →
  (1 + u) * (1 + v) * (1 + w) = 35 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l1155_115521


namespace NUMINAMATH_CALUDE_tiffany_bags_total_l1155_115529

theorem tiffany_bags_total (monday_bags : ℕ) (next_day_bags : ℕ) 
  (h1 : monday_bags = 4) 
  (h2 : next_day_bags = 8) : 
  monday_bags + next_day_bags = 12 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_total_l1155_115529


namespace NUMINAMATH_CALUDE_least_five_digit_multiple_l1155_115522

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem least_five_digit_multiple : ∃ (n : ℕ),
  n = 21000 ∧
  n ≥ 10000 ∧ n < 100000 ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < n →
    ¬(is_divisible_by m 15 ∧
      is_divisible_by m 25 ∧
      is_divisible_by m 40 ∧
      is_divisible_by m 75 ∧
      is_divisible_by m 125 ∧
      is_divisible_by m 140)) ∧
  is_divisible_by n 15 ∧
  is_divisible_by n 25 ∧
  is_divisible_by n 40 ∧
  is_divisible_by n 75 ∧
  is_divisible_by n 125 ∧
  is_divisible_by n 140 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_multiple_l1155_115522


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l1155_115570

/-- A rectangle with length 10 and width 6 -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_eq : length = 10)
  (width_eq : width = 6)

/-- A circle passing through two vertices of the rectangle and tangent to the opposite side -/
structure CircleTangentToRectangle (rect : Rectangle) :=
  (radius : ℝ)
  (passes_through_vertices : Bool)
  (tangent_to_opposite_side : Bool)

/-- The theorem stating that the radius of the circle is 5 -/
theorem circle_radius_is_five (rect : Rectangle) (circle : CircleTangentToRectangle rect) :
  circle.radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l1155_115570


namespace NUMINAMATH_CALUDE_product_equals_square_l1155_115548

theorem product_equals_square : 500 * 2019 * 0.0505 * 20 = (2019 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l1155_115548


namespace NUMINAMATH_CALUDE_sum_of_k_values_l1155_115571

theorem sum_of_k_values : ∃ (S : Finset ℤ), 
  (∀ k ∈ S, ∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 9 = 0 ∧ 3 * y^2 - k * y + 9 = 0) ∧
  (∀ k : ℤ, (∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 9 = 0 ∧ 3 * y^2 - k * y + 9 = 0) → k ∈ S) ∧
  (S.sum id = 0) :=
sorry

end NUMINAMATH_CALUDE_sum_of_k_values_l1155_115571


namespace NUMINAMATH_CALUDE_problem_statement_l1155_115552

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = ({0, a^2, a+b} : Set ℝ) → a^2005 + b^2005 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1155_115552


namespace NUMINAMATH_CALUDE_b_completes_in_12_days_l1155_115547

/-- The number of days B takes to complete the remaining work after A works for 5 days -/
def days_B_completes_work (a_rate b_rate : ℚ) (a_days : ℕ) : ℚ :=
  (1 - a_rate * a_days) / b_rate

theorem b_completes_in_12_days :
  let a_rate : ℚ := 1 / 15
  let b_rate : ℚ := 1 / 18
  let a_days : ℕ := 5
  days_B_completes_work a_rate b_rate a_days = 12 := by
sorry

end NUMINAMATH_CALUDE_b_completes_in_12_days_l1155_115547


namespace NUMINAMATH_CALUDE_infinite_partition_numbers_l1155_115551

theorem infinite_partition_numbers : ∃ (f : ℕ → ℕ), Infinite {n : ℕ | ∃ k, n = f k ∧ n % 4 = 1 ∧ (3 * n * (3 * n + 1) / 2) % (6 * n) = 0} :=
sorry

end NUMINAMATH_CALUDE_infinite_partition_numbers_l1155_115551


namespace NUMINAMATH_CALUDE_jeff_scores_mean_l1155_115561

def jeff_scores : List ℝ := [89, 92, 88, 95, 91]

theorem jeff_scores_mean : (jeff_scores.sum / jeff_scores.length) = 91 := by
  sorry

end NUMINAMATH_CALUDE_jeff_scores_mean_l1155_115561


namespace NUMINAMATH_CALUDE_all_fruits_fallen_by_day_12_l1155_115586

/-- Represents the number of fruits that fall on a given day -/
def fruits_falling (day : ℕ) : ℕ :=
  if day ≤ 10 then day
  else (day - 10)

/-- Represents the total number of fruits that have fallen up to a given day -/
def total_fruits_fallen (day : ℕ) : ℕ :=
  if day ≤ 10 then day * (day + 1) / 2
  else 55 + (day - 10) * (day - 9) / 2

/-- The theorem stating that all fruits will have fallen by the end of the 12th day -/
theorem all_fruits_fallen_by_day_12 :
  total_fruits_fallen 12 = 58 ∧
  ∀ d : ℕ, d < 12 → total_fruits_fallen d < 58 := by
  sorry


end NUMINAMATH_CALUDE_all_fruits_fallen_by_day_12_l1155_115586


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1155_115549

theorem fractional_equation_solution_range (a x : ℝ) : 
  ((a + 2) / (x + 1) = 1) ∧ 
  (x ≤ 0) ∧ 
  (x + 1 ≠ 0) → 
  (a ≤ -1) ∧ (a ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1155_115549


namespace NUMINAMATH_CALUDE_consecutive_sum_product_l1155_115582

theorem consecutive_sum_product (n : ℕ) (h : n > 100) :
  ∃ (a b c : ℕ), (a > 1 ∧ b > 1 ∧ c > 1) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  ((n + (n + 1) + (n + 2) = a * b * c) ∨
   ((n + 1) + (n + 2) + (n + 3) = a * b * c)) := by
sorry


end NUMINAMATH_CALUDE_consecutive_sum_product_l1155_115582


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l1155_115545

theorem seed_germination_percentage
  (seeds_plot1 : ℕ)
  (seeds_plot2 : ℕ)
  (germination_rate_plot1 : ℚ)
  (germination_rate_plot2 : ℚ)
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 15 / 100)
  (h4 : germination_rate_plot2 = 35 / 100)
  : (((seeds_plot1 * germination_rate_plot1 + seeds_plot2 * germination_rate_plot2) / (seeds_plot1 + seeds_plot2)) : ℚ) = 23 / 100 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l1155_115545


namespace NUMINAMATH_CALUDE_sequence_properties_l1155_115531

/-- Sequence definition -/
def a (n : ℕ) (c : ℤ) : ℤ := -n^2 + 4*n + c

/-- Theorem stating the value of c and the minimum m for which a_m ≤ 0 -/
theorem sequence_properties :
  ∃ (c : ℤ),
    (a 3 c = 24) ∧
    (c = 21) ∧
    (∀ m : ℕ, m > 0 → (a m c ≤ 0 ↔ m ≥ 7)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1155_115531


namespace NUMINAMATH_CALUDE_let_go_to_catch_not_specific_analysis_l1155_115533

/-- Definition of "specific analysis of specific issues" methodology --/
def specific_analysis (methodology : String) : Prop :=
  methodology = "analyzing the particularity of contradictions under the guidance of the universality principle of contradictions"

/-- Set of idioms --/
def idioms : Finset String := 
  {"Prescribe the right medicine for the illness; Make clothes to fit the person",
   "Let go to catch; Attack the east while feigning the west",
   "Act according to the situation; Adapt to local conditions",
   "Teach according to aptitude; Differentiate instruction based on individual differences"}

/-- Predicate to check if an idiom reflects the methodology --/
def reflects_methodology (idiom : String) : Prop :=
  idiom ≠ "Let go to catch; Attack the east while feigning the west"

/-- Theorem stating that "Let go to catch; Attack the east while feigning the west" 
    does not reflect the methodology --/
theorem let_go_to_catch_not_specific_analysis :
  ∃ (idiom : String), idiom ∈ idioms ∧ ¬(reflects_methodology idiom) :=
by
  sorry

#check let_go_to_catch_not_specific_analysis

end NUMINAMATH_CALUDE_let_go_to_catch_not_specific_analysis_l1155_115533


namespace NUMINAMATH_CALUDE_inheritance_problem_l1155_115539

/-- The inheritance problem -/
theorem inheritance_problem (x : ℝ) 
  (h1 : 0.25 * x + 0.15 * x = 15000) : x = 37500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_problem_l1155_115539


namespace NUMINAMATH_CALUDE_right_triangle_area_l1155_115543

theorem right_triangle_area (h : ℝ) (α : ℝ) (A : ℝ) :
  h = 8 * Real.sqrt 2 →
  α = 45 * π / 180 →
  A = (h^2 / 4) →
  A = 32 :=
by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_right_triangle_area_l1155_115543


namespace NUMINAMATH_CALUDE_cindy_calculation_l1155_115566

theorem cindy_calculation (x : ℝ) : 
  ((x - 10) / 5 = 40) → ((x - 4) / 10 = 20.6) := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l1155_115566


namespace NUMINAMATH_CALUDE_special_nine_digit_numbers_exist_l1155_115516

/-- Represents a nine-digit number in the specified format -/
structure NineDigitNumber where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  b₁ : ℕ
  b₂ : ℕ
  b₃ : ℕ
  h₁ : a₁ ≠ 0
  h₂ : b₁ * 100 + b₂ * 10 + b₃ = 2 * (a₁ * 100 + a₂ * 10 + a₃)

/-- The value of the nine-digit number -/
def NineDigitNumber.value (n : NineDigitNumber) : ℕ :=
  n.a₁ * 100000000 + n.a₂ * 10000000 + n.a₃ * 1000000 +
  n.b₁ * 100000 + n.b₂ * 10000 + n.b₃ * 1000 +
  n.a₁ * 100 + n.a₂ * 10 + n.a₃

/-- Theorem stating the existence of the special nine-digit numbers -/
theorem special_nine_digit_numbers_exist : ∃ (n : NineDigitNumber),
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧
    n.value = (p₁ * p₂ * p₃ * p₄ * p₅)^2) ∧
  (n.value = 100200100 ∨ n.value = 225450225) :=
sorry

end NUMINAMATH_CALUDE_special_nine_digit_numbers_exist_l1155_115516


namespace NUMINAMATH_CALUDE_emily_sixth_score_l1155_115569

def emily_scores : List ℕ := [94, 97, 88, 90, 102]
def target_mean : ℚ := 95
def num_quizzes : ℕ := 6

theorem emily_sixth_score (sixth_score : ℕ) : 
  sixth_score = 99 →
  (emily_scores.sum + sixth_score) / num_quizzes = target_mean := by
sorry

end NUMINAMATH_CALUDE_emily_sixth_score_l1155_115569


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1155_115503

theorem pie_eating_contest (erik_pie frank_pie : ℝ) 
  (h_erik : erik_pie = 0.67) 
  (h_frank : frank_pie = 0.33) : 
  erik_pie - frank_pie = 0.34 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1155_115503


namespace NUMINAMATH_CALUDE_sin_cos_sum_greater_than_one_l1155_115504

theorem sin_cos_sum_greater_than_one (α : Real) (h : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin α + Real.cos α > 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_greater_than_one_l1155_115504


namespace NUMINAMATH_CALUDE_carols_rectangle_width_l1155_115540

/-- Given two rectangles with equal area, where one has a length of 5 inches
    and the other has dimensions of 2 inches by 60 inches,
    prove that the width of the first rectangle is 24 inches. -/
theorem carols_rectangle_width
  (length_carol : ℝ)
  (width_carol : ℝ)
  (length_jordan : ℝ)
  (width_jordan : ℝ)
  (h1 : length_carol = 5)
  (h2 : length_jordan = 2)
  (h3 : width_jordan = 60)
  (h4 : length_carol * width_carol = length_jordan * width_jordan) :
  width_carol = 24 :=
by sorry

end NUMINAMATH_CALUDE_carols_rectangle_width_l1155_115540


namespace NUMINAMATH_CALUDE_solve_system_l1155_115520

theorem solve_system (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1155_115520


namespace NUMINAMATH_CALUDE_ratio_bounds_l1155_115507

theorem ratio_bounds (A C : ℕ) (n : ℕ) :
  (10 ≤ A ∧ A ≤ 99) →
  (10 ≤ C ∧ C ≤ 99) →
  (100 * A + C) / (A + C) = n →
  11 ≤ n ∧ n ≤ 90 := by
sorry

end NUMINAMATH_CALUDE_ratio_bounds_l1155_115507


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l1155_115565

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 8) 
  (h2 : sum_first_two = 5) :
  ∃ a : ℝ, (a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6)) ∧ 
    (∃ r : ℝ, a / (1 - r) = S ∧ a + a * r = sum_first_two) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l1155_115565
