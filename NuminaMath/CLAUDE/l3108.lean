import Mathlib

namespace NUMINAMATH_CALUDE_addition_problems_l3108_310866

theorem addition_problems :
  (15 + (-8) + 4 + (-10) = 1) ∧
  ((-2) + (7 + 1/2) + 4.5 = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_addition_problems_l3108_310866


namespace NUMINAMATH_CALUDE_contractor_problem_l3108_310849

/-- Calculates the original number of days to complete a job given the original number of laborers,
    the number of absent laborers, and the number of days taken by the remaining laborers. -/
def original_completion_time (total_laborers : ℕ) (absent_laborers : ℕ) (actual_days : ℕ) : ℕ :=
  (total_laborers - absent_laborers) * actual_days / total_laborers

theorem contractor_problem (total_laborers absent_laborers actual_days : ℕ) 
  (h1 : total_laborers = 7)
  (h2 : absent_laborers = 3)
  (h3 : actual_days = 14) :
  original_completion_time total_laborers absent_laborers actual_days = 8 := by
  sorry

#eval original_completion_time 7 3 14

end NUMINAMATH_CALUDE_contractor_problem_l3108_310849


namespace NUMINAMATH_CALUDE_not_R_intersection_A_B_l3108_310874

def set_A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def set_B : Set ℝ := {x | x - 2 > 0}

theorem not_R_intersection_A_B :
  (set_A ∩ set_B)ᶜ = {x : ℝ | x ≤ 2 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_not_R_intersection_A_B_l3108_310874


namespace NUMINAMATH_CALUDE_transformed_area_is_450_l3108_310824

-- Define the transformation matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -8, 6]

-- Define the original area
def original_area : ℝ := 9

-- Theorem statement
theorem transformed_area_is_450 :
  let det := A.det
  let new_area := original_area * |det|
  new_area = 450 := by sorry

end NUMINAMATH_CALUDE_transformed_area_is_450_l3108_310824


namespace NUMINAMATH_CALUDE_allan_bought_three_balloons_l3108_310883

/-- The number of balloons Allan bought at the park -/
def balloons_bought_by_allan (allan_initial : ℕ) (jake_total : ℕ) (jake_difference : ℕ) : ℕ :=
  (jake_total - jake_difference) - allan_initial

/-- Theorem stating that Allan bought 3 balloons at the park -/
theorem allan_bought_three_balloons :
  balloons_bought_by_allan 2 6 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_allan_bought_three_balloons_l3108_310883


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l3108_310831

theorem perfect_square_quadratic (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 2*(k+1)*x + 4 = (x - a)^2) → (k = -3 ∨ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l3108_310831


namespace NUMINAMATH_CALUDE_days_to_pay_cash_register_l3108_310886

/-- Represents the financial data for Marie's bakery --/
structure BakeryFinances where
  cash_register_cost : ℕ
  bread_price : ℕ
  bread_quantity : ℕ
  cake_price : ℕ
  cake_quantity : ℕ
  daily_rent : ℕ
  daily_electricity : ℕ

/-- Calculates the number of days required to pay for the cash register --/
def days_to_pay (finances : BakeryFinances) : ℕ :=
  let daily_income := finances.bread_price * finances.bread_quantity + finances.cake_price * finances.cake_quantity
  let daily_expenses := finances.daily_rent + finances.daily_electricity
  let daily_profit := daily_income - daily_expenses
  finances.cash_register_cost / daily_profit

/-- Theorem stating that it takes 8 days to pay for the cash register --/
theorem days_to_pay_cash_register :
  let maries_finances : BakeryFinances := {
    cash_register_cost := 1040,
    bread_price := 2,
    bread_quantity := 40,
    cake_price := 12,
    cake_quantity := 6,
    daily_rent := 20,
    daily_electricity := 2
  }
  days_to_pay maries_finances = 8 := by
  sorry


end NUMINAMATH_CALUDE_days_to_pay_cash_register_l3108_310886


namespace NUMINAMATH_CALUDE_race_time_difference_l3108_310888

-- Define the race participants
structure Racer where
  name : String
  time : ℕ

-- Define the race conditions
def patrick : Racer := { name := "Patrick", time := 60 }
def amy : Racer := { name := "Amy", time := 36 }

-- Define Manu's time in terms of Amy's
def manu_time (amy : Racer) : ℕ := 2 * amy.time

-- Define the theorem
theorem race_time_difference (amy : Racer) (h : amy.time = 36) : 
  manu_time amy - patrick.time = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l3108_310888


namespace NUMINAMATH_CALUDE_cos_equality_exists_l3108_310885

theorem cos_equality_exists (n : ℤ) : ∃ n, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (315 * π / 180) ∧ n = 45 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_exists_l3108_310885


namespace NUMINAMATH_CALUDE_lottery_probability_l3108_310863

theorem lottery_probability : 
  (1 : ℚ) / 30 * (1 / 50 * 1 / 49 * 1 / 48 * 1 / 47 * 1 / 46) = 1 / 7627536000 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l3108_310863


namespace NUMINAMATH_CALUDE_equation_solution_l3108_310839

theorem equation_solution :
  ∃ x : ℝ, (((3 * x - 1) / (x + 4)) > 0) ∧ 
            (((x + 4) / (3 * x - 1)) > 0) ∧
            (Real.sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3 * x - 1)) = 0) ∧
            (x = 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3108_310839


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l3108_310898

theorem consecutive_pages_sum (x : ℕ) (h : x + (x + 1) = 185) : x = 92 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l3108_310898


namespace NUMINAMATH_CALUDE_max_sum_of_a_and_b_l3108_310848

theorem max_sum_of_a_and_b : ∀ a b : ℕ+,
  b > 2 →
  a^(b:ℕ) < 600 →
  a + b ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_a_and_b_l3108_310848


namespace NUMINAMATH_CALUDE_probability_at_least_one_event_l3108_310894

theorem probability_at_least_one_event (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/2) (h2 : p2 = 1/3) (h3 : p3 = 1/4) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_event_l3108_310894


namespace NUMINAMATH_CALUDE_perpendicular_vector_proof_l3108_310822

/-- Given two parallel lines with direction vector (5, 4), prove that the vector (v₁, v₂) 
    perpendicular to (5, 4) satisfying v₁ + v₂ = 7 is (-28, 35). -/
theorem perpendicular_vector_proof (v₁ v₂ : ℝ) : 
  (5 * 4 + 4 * (-5) = 0) →  -- Lines are parallel with direction vector (5, 4)
  (5 * v₁ + 4 * v₂ = 0) →   -- (v₁, v₂) is perpendicular to (5, 4)
  (v₁ + v₂ = 7) →           -- Sum of v₁ and v₂ is 7
  (v₁ = -28 ∧ v₂ = 35) :=   -- Conclusion: v₁ = -28 and v₂ = 35
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_vector_proof_l3108_310822


namespace NUMINAMATH_CALUDE_car_speed_proof_l3108_310844

/-- Proves that a car's speed is 600 km/h given the problem conditions -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v - 1 / 900) * 3600 = 2 ↔ v = 600 := by
  sorry

#check car_speed_proof

end NUMINAMATH_CALUDE_car_speed_proof_l3108_310844


namespace NUMINAMATH_CALUDE_circle_properties_l3108_310855

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*y + 5*m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

-- Theorem statement
theorem circle_properties :
  -- Part 1: Range of m
  (∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → (m < 1 ∨ m > 4)) ∧
  -- Part 2: Length of chord when m = -2
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ (-2) ∧
    circle_equation x₂ y₂ (-2) ∧
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 26) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3108_310855


namespace NUMINAMATH_CALUDE_certain_number_proof_l3108_310893

theorem certain_number_proof (x N : ℝ) 
  (h1 : 3 * x = (N - x) + 14) 
  (h2 : x = 10) : 
  N = 26 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3108_310893


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l3108_310842

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l3108_310842


namespace NUMINAMATH_CALUDE_no_integer_solution_for_trig_equation_l3108_310815

theorem no_integer_solution_for_trig_equation : 
  ¬ ∃ (a b : ℤ), Real.sqrt (4 - 3 * Real.sin (30 * π / 180)) = a + b * (1 / Real.sin (30 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_trig_equation_l3108_310815


namespace NUMINAMATH_CALUDE_expected_value_is_one_l3108_310897

/-- Represents the possible outcomes of rolling the die -/
inductive DieOutcome
| one
| two
| three
| four
| five
| six

/-- The probability of rolling each outcome -/
def prob (outcome : DieOutcome) : ℚ :=
  match outcome with
  | .one => 1/4
  | .two => 1/4
  | .three => 1/6
  | .four => 1/6
  | .five => 1/12
  | .six => 1/12

/-- The earnings associated with each outcome -/
def earnings (outcome : DieOutcome) : ℤ :=
  match outcome with
  | .one | .two => 4
  | .three | .four => -3
  | .five | .six => 0

/-- The expected value of earnings from one roll of the die -/
def expectedValue : ℚ :=
  (prob .one * earnings .one) +
  (prob .two * earnings .two) +
  (prob .three * earnings .three) +
  (prob .four * earnings .four) +
  (prob .five * earnings .five) +
  (prob .six * earnings .six)

/-- Theorem stating that the expected value of earnings is 1 -/
theorem expected_value_is_one : expectedValue = 1 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_one_l3108_310897


namespace NUMINAMATH_CALUDE_cube_sum_equals_110_l3108_310840

theorem cube_sum_equals_110 (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 6) : 
  x^3 + y^3 = 110 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_110_l3108_310840


namespace NUMINAMATH_CALUDE_pens_taken_after_second_month_pens_taken_after_second_month_is_41_l3108_310835

theorem pens_taken_after_second_month 
  (num_students : ℕ) 
  (red_pens_per_student : ℕ) 
  (black_pens_per_student : ℕ) 
  (pens_taken_first_month : ℕ) 
  (pens_per_student_after_split : ℕ) : ℕ :=
  let total_pens := num_students * (red_pens_per_student + black_pens_per_student)
  let pens_after_first_month := total_pens - pens_taken_first_month
  let pens_after_split := num_students * pens_per_student_after_split
  pens_after_first_month - pens_after_split

theorem pens_taken_after_second_month_is_41 :
  pens_taken_after_second_month 3 62 43 37 79 = 41 := by
  sorry

end NUMINAMATH_CALUDE_pens_taken_after_second_month_pens_taken_after_second_month_is_41_l3108_310835


namespace NUMINAMATH_CALUDE_overhead_percentage_problem_l3108_310851

/-- Calculates the percentage of cost for overhead given the purchase price, markup, and net profit. -/
def overhead_percentage (purchase_price markup net_profit : ℚ) : ℚ :=
  let overhead := markup - net_profit
  (overhead / purchase_price) * 100

/-- Theorem stating that given the specific values in the problem, the overhead percentage is 37.5% -/
theorem overhead_percentage_problem :
  let purchase_price : ℚ := 48
  let markup : ℚ := 30
  let net_profit : ℚ := 12
  overhead_percentage purchase_price markup net_profit = 37.5 := by
  sorry

#eval overhead_percentage 48 30 12

end NUMINAMATH_CALUDE_overhead_percentage_problem_l3108_310851


namespace NUMINAMATH_CALUDE_third_place_winnings_value_l3108_310852

/-- The amount of money in the pot -/
def pot_total : ℝ := 210

/-- The percentage of the pot that the third place winner receives -/
def third_place_percentage : ℝ := 0.15

/-- The amount of money the third place winner receives -/
def third_place_winnings : ℝ := pot_total * third_place_percentage

theorem third_place_winnings_value : third_place_winnings = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_third_place_winnings_value_l3108_310852


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l3108_310862

theorem sin_seven_pi_sixths : Real.sin (7 * Real.pi / 6) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l3108_310862


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3108_310837

theorem complex_equation_solution (a : ℂ) :
  a / (1 - Complex.I) = (1 + Complex.I) / Complex.I → a = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3108_310837


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l3108_310819

/-- Represents an isosceles trapezoid with given measurements -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculates the area of the isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the area of the specific trapezoid is approximately 318.93 -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 20,
    diagonal_length := 25,
    longer_base := 30
  }
  abs (trapezoid_area t - 318.93) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l3108_310819


namespace NUMINAMATH_CALUDE_proportion_problem_l3108_310872

theorem proportion_problem (y : ℝ) : (0.75 : ℝ) / 2 = y / 8 → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l3108_310872


namespace NUMINAMATH_CALUDE_seulgi_stack_higher_l3108_310809

/-- Represents the stack of boxes for each person -/
structure BoxStack where
  numBoxes : ℕ
  boxHeight : ℝ

/-- Calculates the total height of a stack of boxes -/
def totalHeight (stack : BoxStack) : ℝ :=
  stack.numBoxes * stack.boxHeight

theorem seulgi_stack_higher (hyunjeong seulgi : BoxStack)
  (h1 : hyunjeong.numBoxes = 15)
  (h2 : hyunjeong.boxHeight = 4.2)
  (h3 : seulgi.numBoxes = 20)
  (h4 : seulgi.boxHeight = 3.3) :
  totalHeight seulgi > totalHeight hyunjeong := by
  sorry

end NUMINAMATH_CALUDE_seulgi_stack_higher_l3108_310809


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3108_310816

theorem rectangle_max_area (a b : ℝ) (h : a > 0 ∧ b > 0) :
  2 * (a + b) = 60 → a * b ≤ 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3108_310816


namespace NUMINAMATH_CALUDE_sock_probability_theorem_l3108_310860

def gray_socks : ℕ := 12
def white_socks : ℕ := 10
def blue_socks : ℕ := 6

def total_socks : ℕ := gray_socks + white_socks + blue_socks

def probability_matching_or_different_colors : ℚ :=
  let total_combinations := Nat.choose total_socks 3
  let matching_gray := Nat.choose gray_socks 2 * (white_socks + blue_socks)
  let matching_white := Nat.choose white_socks 2 * (gray_socks + blue_socks)
  let matching_blue := Nat.choose blue_socks 2 * (gray_socks + white_socks)
  let all_different := gray_socks * white_socks * blue_socks
  let favorable_outcomes := matching_gray + matching_white + matching_blue + all_different
  (favorable_outcomes : ℚ) / total_combinations

theorem sock_probability_theorem :
  probability_matching_or_different_colors = 81 / 91 :=
by sorry

end NUMINAMATH_CALUDE_sock_probability_theorem_l3108_310860


namespace NUMINAMATH_CALUDE_triangle_median_inequality_l3108_310834

theorem triangle_median_inequality (a b c ma mb mc : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hma : ma > 0) (hmb : mb > 0) (hmc : mc > 0)
  (h_ma : 4 * ma^2 = 2 * (b^2 + c^2) - a^2)
  (h_mb : 4 * mb^2 = 2 * (c^2 + a^2) - b^2)
  (h_mc : 4 * mc^2 = 2 * (a^2 + b^2) - c^2) :
  ma^2 / a^2 + mb^2 / b^2 + mc^2 / c^2 ≥ 9/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_median_inequality_l3108_310834


namespace NUMINAMATH_CALUDE_complex_magnitude_l3108_310801

/-- Given a complex number z satisfying z(1+i) = 1-2i, prove that |z| = √10/2 -/
theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 1 - 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3108_310801


namespace NUMINAMATH_CALUDE_soap_brand_survey_l3108_310857

theorem soap_brand_survey (total : ℕ) (neither : ℕ) (only_w : ℕ) :
  total = 200 →
  neither = 80 →
  only_w = 60 →
  ∃ (both : ℕ),
    both * 4 = total - neither - only_w ∧
    both = 15 :=
by sorry

end NUMINAMATH_CALUDE_soap_brand_survey_l3108_310857


namespace NUMINAMATH_CALUDE_inequality_problem_l3108_310880

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > -a) 
  (h4 : c < d) (h5 : d < 0) : 
  (a / d + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) := by
sorry

end NUMINAMATH_CALUDE_inequality_problem_l3108_310880


namespace NUMINAMATH_CALUDE_main_theorem_l3108_310800

/-- The set of real numbers c > 0 for which exactly one of two statements is true --/
def C : Set ℝ := {c | c > 0 ∧ (c ≤ 1/2 ∨ c ≥ 1)}

/-- Statement p: The function y = c^x is monotonically decreasing on ℝ --/
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

/-- Statement q: The solution set of x + |x - 2c| > 1 is ℝ --/
def q (c : ℝ) : Prop := ∀ x : ℝ, x + |x - 2*c| > 1

/-- Main theorem: c is in set C if and only if exactly one of p(c) or q(c) is true --/
theorem main_theorem (c : ℝ) : c ∈ C ↔ (p c ∧ ¬q c) ∨ (¬p c ∧ q c) := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l3108_310800


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_tangent_to_circle_l3108_310821

theorem hyperbola_asymptotes_tangent_to_circle (m : ℝ) : 
  m > 0 → 
  (∃ (x y : ℝ), y^2 - x^2/m^2 = 1) →
  (∃ (x y : ℝ), x^2 + y^2 - 4*y + 3 = 0) →
  (∀ (x y : ℝ), (y^2 - x^2/m^2 = 0) → (x^2 + (y-2)^2 = 1)) →
  m = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_tangent_to_circle_l3108_310821


namespace NUMINAMATH_CALUDE_semicircle_rectangle_property_l3108_310873

-- Define the semicircle and its properties
structure Semicircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the rectangle
structure Rectangle where
  base : ℝ
  height : ℝ

-- Define a point on the semicircle
def PointOnSemicircle (s : Semicircle) := { p : ℝ × ℝ // (p.1 - s.center.1)^2 + (p.2 - s.center.2)^2 = s.radius^2 ∧ p.2 ≥ s.center.2 }

-- Theorem statement
theorem semicircle_rectangle_property
  (s : Semicircle)
  (r : Rectangle)
  (h_square : r.height = s.radius / Real.sqrt 2)  -- Height equals side of inscribed square
  (h_base : r.base = 2 * s.radius)  -- Base is diameter
  (M : PointOnSemicircle s)
  (E F : ℝ)  -- E and F are x-coordinates on the diameter
  (h_E : E ∈ Set.Icc s.center.1 (s.center.1 + s.radius))
  (h_F : F ∈ Set.Icc s.center.1 (s.center.1 + s.radius))
  : (F - s.center.1)^2 + (s.center.1 + 2*s.radius - E)^2 = (2*s.radius)^2 := by
  sorry


end NUMINAMATH_CALUDE_semicircle_rectangle_property_l3108_310873


namespace NUMINAMATH_CALUDE_min_sum_visible_faces_l3108_310896

/-- Represents a die in the 4x4x4 cube --/
structure Die where
  visible_faces : List Nat
  deriving Repr

/-- Represents the 4x4x4 cube made of dice --/
structure Cube where
  dice : List Die
  deriving Repr

/-- Checks if a die's opposite sides sum to 7 --/
def valid_die (d : Die) : Prop :=
  d.visible_faces.length ≤ 4 ∧ 
  ∀ i j, i + j = 5 → i < d.visible_faces.length → j < d.visible_faces.length → 
    d.visible_faces[i]! + d.visible_faces[j]! = 7

/-- Checks if the cube is valid (4x4x4 and made of 64 dice) --/
def valid_cube (c : Cube) : Prop :=
  c.dice.length = 64 ∧ ∀ d ∈ c.dice, valid_die d

/-- Calculates the sum of visible faces on the cube --/
def sum_visible_faces (c : Cube) : Nat :=
  c.dice.foldl (λ acc d => acc + d.visible_faces.foldl (λ sum face => sum + face) 0) 0

/-- Theorem: The smallest possible sum of visible faces on a valid 4x4x4 cube is 304 --/
theorem min_sum_visible_faces (c : Cube) (h : valid_cube c) : 
  ∃ (min_cube : Cube), valid_cube min_cube ∧ 
    sum_visible_faces min_cube = 304 ∧
    ∀ (other_cube : Cube), valid_cube other_cube → 
      sum_visible_faces other_cube ≥ sum_visible_faces min_cube := by
  sorry

end NUMINAMATH_CALUDE_min_sum_visible_faces_l3108_310896


namespace NUMINAMATH_CALUDE_chicken_to_beef_ratio_is_two_to_one_l3108_310892

/-- Represents the order of beef and chicken --/
structure FoodOrder where
  beef_pounds : ℕ
  beef_price_per_pound : ℕ
  chicken_price_per_pound : ℕ
  total_cost : ℕ

/-- Calculates the ratio of chicken to beef in the order --/
def chicken_to_beef_ratio (order : FoodOrder) : ℚ :=
  let beef_cost := order.beef_pounds * order.beef_price_per_pound
  let chicken_cost := order.total_cost - beef_cost
  let chicken_pounds := chicken_cost / order.chicken_price_per_pound
  chicken_pounds / order.beef_pounds

/-- Theorem stating that the ratio of chicken to beef is 2:1 for the given order --/
theorem chicken_to_beef_ratio_is_two_to_one (order : FoodOrder) 
  (h1 : order.beef_pounds = 1000)
  (h2 : order.beef_price_per_pound = 8)
  (h3 : order.chicken_price_per_pound = 3)
  (h4 : order.total_cost = 14000) : 
  chicken_to_beef_ratio order = 2 := by
  sorry

#eval chicken_to_beef_ratio { 
  beef_pounds := 1000, 
  beef_price_per_pound := 8, 
  chicken_price_per_pound := 3, 
  total_cost := 14000 
}

end NUMINAMATH_CALUDE_chicken_to_beef_ratio_is_two_to_one_l3108_310892


namespace NUMINAMATH_CALUDE_time_between_flashes_l3108_310858

/-- Represents the number of flashes in 3/4 of an hour -/
def flashes_per_three_quarters_hour : ℕ := 300

/-- Represents 3/4 of an hour in seconds -/
def three_quarters_hour_in_seconds : ℕ := 45 * 60

/-- Theorem stating that the time between flashes is 9 seconds -/
theorem time_between_flashes :
  three_quarters_hour_in_seconds / flashes_per_three_quarters_hour = 9 := by
  sorry

end NUMINAMATH_CALUDE_time_between_flashes_l3108_310858


namespace NUMINAMATH_CALUDE_house_worth_problem_l3108_310828

theorem house_worth_problem (initial_price final_price : ℝ) 
  (h1 : final_price = initial_price * 1.1 * 0.9)
  (h2 : final_price = 99000) : initial_price = 100000 := by
  sorry

end NUMINAMATH_CALUDE_house_worth_problem_l3108_310828


namespace NUMINAMATH_CALUDE_negative_i_fourth_power_l3108_310808

theorem negative_i_fourth_power (i : ℂ) (h : i^2 = -1) : (-i)^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_i_fourth_power_l3108_310808


namespace NUMINAMATH_CALUDE_find_number_l3108_310882

theorem find_number : ∃ n : ℕ, n + 3427 = 13200 ∧ n = 9773 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3108_310882


namespace NUMINAMATH_CALUDE_quadratic_one_zero_l3108_310867

/-- If a quadratic function f(x) = mx^2 - 2x + 3 has only one zero, then m = 0 or m = 1/3 -/
theorem quadratic_one_zero (m : ℝ) : 
  (∃! x, m * x^2 - 2 * x + 3 = 0) → (m = 0 ∨ m = 1/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_zero_l3108_310867


namespace NUMINAMATH_CALUDE_root_conditions_l3108_310890

/-- The equation x^4 + px^2 + q = 0 has real roots satisfying x₂/x₁ = x₃/x₂ = x₄/x₃ 
    if and only if p < 0 and q = p^2/4 -/
theorem root_conditions (p q : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₃ ≠ 0 ∧
    x₁^4 + p*x₁^2 + q = 0 ∧
    x₂^4 + p*x₂^2 + q = 0 ∧
    x₃^4 + p*x₃^2 + q = 0 ∧
    x₄^4 + p*x₄^2 + q = 0 ∧
    x₂/x₁ = x₃/x₂ ∧ x₃/x₂ = x₄/x₃) ↔
  (p < 0 ∧ q = p^2/4) :=
by sorry


end NUMINAMATH_CALUDE_root_conditions_l3108_310890


namespace NUMINAMATH_CALUDE_work_completion_time_l3108_310805

/-- The number of days it takes for person B to complete the work alone -/
def B_days : ℝ := 60

/-- The fraction of work completed by A and B together in 6 days -/
def work_completed : ℝ := 0.25

/-- The number of days A and B work together -/
def days_together : ℝ := 6

/-- The number of days it takes for person A to complete the work alone -/
def A_days : ℝ := 40

theorem work_completion_time :
  (1 / A_days + 1 / B_days) * days_together = work_completed :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3108_310805


namespace NUMINAMATH_CALUDE_coefficient_of_x_l3108_310802

/-- The coefficient of x in the expression 3(x - 4) + 4(7 - 2x^2 + 5x) - 8(2x - 1) is 7 -/
theorem coefficient_of_x (x : ℝ) : 
  let expr := 3*(x - 4) + 4*(7 - 2*x^2 + 5*x) - 8*(2*x - 1)
  ∃ (a b c : ℝ), expr = a*x^2 + 7*x + c :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l3108_310802


namespace NUMINAMATH_CALUDE_carpet_length_proof_l3108_310829

theorem carpet_length_proof (carpet_width : ℝ) (room_area : ℝ) (coverage_percentage : ℝ) :
  carpet_width = 4 →
  room_area = 180 →
  coverage_percentage = 0.20 →
  let carpet_area := room_area * coverage_percentage
  let carpet_length := carpet_area / carpet_width
  carpet_length = 9 := by sorry

end NUMINAMATH_CALUDE_carpet_length_proof_l3108_310829


namespace NUMINAMATH_CALUDE_bryden_receives_correct_amount_l3108_310820

/-- The face value of a state quarter in dollars -/
def face_value : ℚ := 1/4

/-- The number of quarters Bryden is selling -/
def num_quarters : ℕ := 6

/-- The percentage of face value offered by the collector -/
def offer_percentage : ℕ := 1500

/-- The amount Bryden will receive in dollars -/
def amount_received : ℚ := (offer_percentage : ℚ) / 100 * face_value * num_quarters

theorem bryden_receives_correct_amount : amount_received = 45/2 := by
  sorry

end NUMINAMATH_CALUDE_bryden_receives_correct_amount_l3108_310820


namespace NUMINAMATH_CALUDE_gcd_45_75_l3108_310826

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l3108_310826


namespace NUMINAMATH_CALUDE_flame_shooting_time_l3108_310899

theorem flame_shooting_time (firing_interval : ℝ) (flame_duration : ℝ) (total_time : ℝ) :
  firing_interval = 15 →
  flame_duration = 5 →
  total_time = 60 →
  (total_time / firing_interval) * flame_duration = 20 := by
  sorry

end NUMINAMATH_CALUDE_flame_shooting_time_l3108_310899


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l3108_310884

theorem difference_of_squares_example : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l3108_310884


namespace NUMINAMATH_CALUDE_shelf_theorem_l3108_310856

/-- Given two shelves, with the second twice as long as the first, and book thicknesses,
    prove the relation between the number of books on each shelf. -/
theorem shelf_theorem (A' H' S' M' E' : ℕ) (x y : ℝ) : 
  A' ≠ H' ∧ A' ≠ S' ∧ A' ≠ M' ∧ A' ≠ E' ∧ 
  H' ≠ S' ∧ H' ≠ M' ∧ H' ≠ E' ∧ 
  S' ≠ M' ∧ S' ≠ E' ∧ 
  M' ≠ E' ∧
  A' > 0 ∧ H' > 0 ∧ S' > 0 ∧ M' > 0 ∧ E' > 0 ∧
  y > x ∧ 
  A' * x + H' * y = S' * x + M' * y ∧ 
  E' * x = 2 * (A' * x + H' * y) →
  E' = (2 * A' * M' - 2 * S' * H') / (M' - H') :=
by sorry

end NUMINAMATH_CALUDE_shelf_theorem_l3108_310856


namespace NUMINAMATH_CALUDE_system_solution_l3108_310825

theorem system_solution : ∃ (x y : ℝ), 2 * x - y = 8 ∧ 3 * x + 2 * y = 5 := by
  use 3, -2
  sorry

end NUMINAMATH_CALUDE_system_solution_l3108_310825


namespace NUMINAMATH_CALUDE_balls_removed_by_other_students_l3108_310823

theorem balls_removed_by_other_students (tennis_balls soccer_balls baskets students_removed_8 remaining_balls : ℕ) 
  (h1 : tennis_balls = 15)
  (h2 : soccer_balls = 5)
  (h3 : baskets = 5)
  (h4 : students_removed_8 = 3)
  (h5 : remaining_balls = 56) : 
  ((baskets * (tennis_balls + soccer_balls)) - (students_removed_8 * 8) - remaining_balls) / 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_balls_removed_by_other_students_l3108_310823


namespace NUMINAMATH_CALUDE_factorial_divisibility_l3108_310879

theorem factorial_divisibility (p : ℕ) (h : Prime p) :
  ∃ k : ℕ, (p^2).factorial = k * (p.factorial ^ (p + 1)) :=
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l3108_310879


namespace NUMINAMATH_CALUDE_original_jellybean_count_l3108_310814

-- Define the daily reduction rate
def daily_reduction_rate : ℝ := 0.8

-- Define the function that calculates the remaining quantity after n days
def remaining_after_days (initial : ℝ) (days : ℕ) : ℝ :=
  initial * (daily_reduction_rate ^ days)

-- State the theorem
theorem original_jellybean_count :
  ∃ (initial : ℝ), remaining_after_days initial 2 = 32 ∧ initial = 50 := by
  sorry

end NUMINAMATH_CALUDE_original_jellybean_count_l3108_310814


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3108_310881

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  12 * x - 6 * y + 3 * y - 24 * x = -12 * x - 3 * y := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  3/2 * (a^2 * b - 2 * a * b^2) - 1/2 * (a * b^2 - 4 * a^2 * b) + 1/2 * a * b^2 =
  7/2 * a^2 * b - 3 * a * b^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3108_310881


namespace NUMINAMATH_CALUDE_parallel_vectors_tangent_l3108_310877

theorem parallel_vectors_tangent (θ : ℝ) (a b : ℝ × ℝ) : 
  a = (2, Real.sin θ) → 
  b = (1, Real.cos θ) → 
  (∃ (k : ℝ), a = k • b) → 
  Real.tan θ = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_tangent_l3108_310877


namespace NUMINAMATH_CALUDE_equation_solutions_l3108_310817

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 7 ∧ x2 = 2 - Real.sqrt 7 ∧
    x1^2 - 4*x1 - 3 = 0 ∧ x2^2 - 4*x2 - 3 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 4 ∧ x2 = 4/3 ∧
    (x1 + 1)^2 = (2*x1 - 3)^2 ∧ (x2 + 1)^2 = (2*x2 - 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3108_310817


namespace NUMINAMATH_CALUDE_greatest_integer_and_y_value_l3108_310864

theorem greatest_integer_and_y_value :
  (∃ x : ℤ, (∀ z : ℤ, 7 - 5*z > 22 → z ≤ x) ∧ 7 - 5*x > 22 ∧ x = -4) ∧
  (let x := -4; 2*x + 3 = -5) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_and_y_value_l3108_310864


namespace NUMINAMATH_CALUDE_cube_root_unity_product_l3108_310804

theorem cube_root_unity_product (w : ℂ) : w^3 = 1 → (1 - w + w^2) * (1 + w - w^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_product_l3108_310804


namespace NUMINAMATH_CALUDE_gift_spending_calculation_l3108_310853

/-- Given a total amount spent and an amount spent on giftwrapping and other expenses,
    calculate the amount spent on gifts. -/
def amount_spent_on_gifts (total_amount : ℚ) (giftwrapping_amount : ℚ) : ℚ :=
  total_amount - giftwrapping_amount

/-- Prove that the amount spent on gifts is $561.00, given the total amount
    spent is $700.00 and the amount spent on giftwrapping is $139.00. -/
theorem gift_spending_calculation :
  amount_spent_on_gifts 700 139 = 561 := by
  sorry

end NUMINAMATH_CALUDE_gift_spending_calculation_l3108_310853


namespace NUMINAMATH_CALUDE_square_side_ratio_l3108_310841

theorem square_side_ratio (area_ratio : ℚ) (h : area_ratio = 50 / 98) :
  ∃ (p q r : ℕ), 
    (Real.sqrt (area_ratio) = p * Real.sqrt q / r) ∧
    (p + q + r = 13) := by
  sorry

end NUMINAMATH_CALUDE_square_side_ratio_l3108_310841


namespace NUMINAMATH_CALUDE_absent_student_percentage_l3108_310836

theorem absent_student_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (absent_boys_fraction : ℚ)
  (absent_girls_fraction : ℚ)
  (h1 : total_students = 100)
  (h2 : boys = 50)
  (h3 : girls = 50)
  (h4 : boys + girls = total_students)
  (h5 : absent_boys_fraction = 1 / 5)
  (h6 : absent_girls_fraction = 1 / 4) :
  (↑boys * absent_boys_fraction + ↑girls * absent_girls_fraction) / ↑total_students = 225 / 1000 := by
  sorry

#check absent_student_percentage

end NUMINAMATH_CALUDE_absent_student_percentage_l3108_310836


namespace NUMINAMATH_CALUDE_vikki_earnings_insurance_deduction_l3108_310859

/-- Vikki's weekly earnings and deductions -/
def weekly_earnings_problem (hours_worked : ℕ) (hourly_rate : ℚ) (tax_rate : ℚ) 
  (union_dues : ℚ) (take_home_pay : ℚ) : Prop :=
  let gross_earnings := hours_worked * hourly_rate
  let tax_deduction := gross_earnings * tax_rate
  let after_tax_and_dues := gross_earnings - tax_deduction - union_dues
  let insurance_deduction := after_tax_and_dues - take_home_pay
  let insurance_percentage := insurance_deduction / gross_earnings * 100
  insurance_percentage = 5

theorem vikki_earnings_insurance_deduction :
  weekly_earnings_problem 42 10 (1/5) 5 310 :=
sorry

end NUMINAMATH_CALUDE_vikki_earnings_insurance_deduction_l3108_310859


namespace NUMINAMATH_CALUDE_plane_parallel_transitivity_l3108_310812

-- Define the concept of planes
variable (Plane : Type)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Theorem statement
theorem plane_parallel_transitivity (α β γ : Plane) :
  (∃ γ, parallel γ α ∧ parallel γ β) → parallel α β := by
  sorry

end NUMINAMATH_CALUDE_plane_parallel_transitivity_l3108_310812


namespace NUMINAMATH_CALUDE_linda_babysitting_hours_l3108_310850

/-- Linda's babysitting problem -/
theorem linda_babysitting_hours (babysitting_rate : ℚ) (application_fee : ℚ) (num_colleges : ℕ) :
  babysitting_rate = 10 →
  application_fee = 25 →
  num_colleges = 6 →
  (num_colleges * application_fee) / babysitting_rate = 15 :=
by sorry

end NUMINAMATH_CALUDE_linda_babysitting_hours_l3108_310850


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3108_310887

theorem sum_of_roots_quadratic (p q : ℝ) : 
  (p^2 - p - 1 = 0) → 
  (q^2 - q - 1 = 0) → 
  (p ≠ q) →
  (∃ x y : ℝ, x^2 - p*x + p*q = 0 ∧ y^2 - p*y + p*q = 0 ∧ x + y = (1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3108_310887


namespace NUMINAMATH_CALUDE_soda_price_theorem_l3108_310876

/-- Calculates the price of soda cans given specific discount conditions -/
def sodaPrice (regularPrice : ℝ) (caseDiscount : ℝ) (bulkDiscount : ℝ) (caseSize : ℕ) (numCans : ℕ) : ℝ :=
  let discountedPrice := regularPrice * (1 - caseDiscount)
  let fullCases := numCans / caseSize
  let remainingCans := numCans % caseSize
  let fullCasePrice := if fullCases ≥ 3
                       then (fullCases * caseSize * discountedPrice) * (1 - bulkDiscount)
                       else fullCases * caseSize * discountedPrice
  let remainingPrice := remainingCans * discountedPrice
  fullCasePrice + remainingPrice

/-- The price of 70 cans of soda under given discount conditions is $26.895 -/
theorem soda_price_theorem :
  sodaPrice 0.55 0.25 0.10 24 70 = 26.895 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_theorem_l3108_310876


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3108_310895

theorem fraction_sum_equality : 
  (3 : ℚ) / 12 + (6 : ℚ) / 120 + (9 : ℚ) / 1200 = (3075 : ℚ) / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3108_310895


namespace NUMINAMATH_CALUDE_min_a_squared_over_area_l3108_310810

theorem min_a_squared_over_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  b * Real.cos C + c * Real.cos B = 3 * a * Real.cos A →
  S = (1 / 2) * b * c * Real.sin A →
  a^2 / S ≥ 2 * Real.sqrt 2 := by
  sorry

#check min_a_squared_over_area

end NUMINAMATH_CALUDE_min_a_squared_over_area_l3108_310810


namespace NUMINAMATH_CALUDE_square_sum_value_l3108_310871

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l3108_310871


namespace NUMINAMATH_CALUDE_brians_breath_holding_l3108_310818

theorem brians_breath_holding (T : ℝ) : 
  T > 0 → (T * 2 * 2 * 1.5 = 60) → T = 10 := by
  sorry

end NUMINAMATH_CALUDE_brians_breath_holding_l3108_310818


namespace NUMINAMATH_CALUDE_kelly_games_left_l3108_310847

theorem kelly_games_left (initial_games give_away_games : ℕ) 
  (h1 : initial_games = 257)
  (h2 : give_away_games = 138) :
  initial_games - give_away_games = 119 :=
by sorry

end NUMINAMATH_CALUDE_kelly_games_left_l3108_310847


namespace NUMINAMATH_CALUDE_prob_no_green_3x3_value_main_result_l3108_310869

/-- Represents a 4x4 grid of colored squares -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 3x3 subgrid starting at (i, j) is all green -/
def has_green_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ x y, g (i + x) (j + y)

/-- The probability of not having a 3x3 green square in a 4x4 grid -/
def prob_no_green_3x3 : ℚ :=
  1 - (419 : ℚ) / 2^16

theorem prob_no_green_3x3_value :
  prob_no_green_3x3 = 65117 / 65536 :=
sorry

/-- The sum of the numerator and denominator of the probability -/
def sum_num_denom : ℕ := 65117 + 65536

theorem main_result :
  sum_num_denom = 130653 :=
sorry

end NUMINAMATH_CALUDE_prob_no_green_3x3_value_main_result_l3108_310869


namespace NUMINAMATH_CALUDE_circle_radius_is_one_l3108_310807

/-- The radius of a circle defined by the equation x^2 + y^2 - 2y = 0 is 1 -/
theorem circle_radius_is_one :
  let circle_eq := (fun x y : ℝ => x^2 + y^2 - 2*y = 0)
  ∃ (h k r : ℝ), r = 1 ∧ ∀ x y : ℝ, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_one_l3108_310807


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_zero_two_l3108_310843

def A : Set ℤ := {-2, 0, 2}

-- Define the absolute value function
def f (x : ℤ) : ℤ := abs x

-- Define B as the image of A under f
def B : Set ℤ := f '' A

-- State the theorem
theorem A_intersect_B_equals_zero_two : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_zero_two_l3108_310843


namespace NUMINAMATH_CALUDE_savings_ratio_is_three_fifths_l3108_310891

/-- Represents the savings scenario of Thomas and Joseph -/
structure SavingsScenario where
  thomas_monthly_savings : ℕ
  total_savings : ℕ
  saving_period_months : ℕ

/-- Calculates the ratio of Joseph's monthly savings to Thomas's monthly savings -/
def savings_ratio (scenario : SavingsScenario) : Rat :=
  let thomas_total := scenario.thomas_monthly_savings * scenario.saving_period_months
  let joseph_total := scenario.total_savings - thomas_total
  let joseph_monthly := joseph_total / scenario.saving_period_months
  joseph_monthly / scenario.thomas_monthly_savings

/-- The main theorem stating the ratio of Joseph's to Thomas's monthly savings -/
theorem savings_ratio_is_three_fifths (scenario : SavingsScenario)
  (h1 : scenario.thomas_monthly_savings = 40)
  (h2 : scenario.total_savings = 4608)
  (h3 : scenario.saving_period_months = 72) :
  savings_ratio scenario = 3 / 5 := by
  sorry

#eval savings_ratio { thomas_monthly_savings := 40, total_savings := 4608, saving_period_months := 72 }

end NUMINAMATH_CALUDE_savings_ratio_is_three_fifths_l3108_310891


namespace NUMINAMATH_CALUDE_f_max_value_l3108_310854

/-- The quadratic function f(y) = -9y^2 + 15y + 3 -/
def f (y : ℝ) := -9 * y^2 + 15 * y + 3

/-- The maximum value of f(y) is 6.25 -/
theorem f_max_value : ∃ (y : ℝ), f y = 6.25 ∧ ∀ (z : ℝ), f z ≤ 6.25 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l3108_310854


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3108_310806

theorem polynomial_factorization (x : ℝ) : 
  x^2 - 6*x + 9 - 49*x^4 = (-7*x^2 + x - 3) * (7*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3108_310806


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3108_310803

theorem line_circle_intersection (k : ℝ) :
  ∃ (x y : ℝ), y = k * (x - 1) ∧ x^2 + y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3108_310803


namespace NUMINAMATH_CALUDE_division_problem_l3108_310846

theorem division_problem (A : ℕ) (h : A % 7 = 3 ∧ A / 7 = 5) : A = 38 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3108_310846


namespace NUMINAMATH_CALUDE_weight_density_half_fluid_density_l3108_310833

/-- A spring-mass system submerged in a fluid -/
structure SpringMassSystem where
  /-- Spring constant -/
  k : ℝ
  /-- Mass of the weight -/
  m : ℝ
  /-- Acceleration due to gravity -/
  g : ℝ
  /-- Density of the fluid (kerosene) -/
  ρ_fluid : ℝ
  /-- Density of the weight material -/
  ρ_material : ℝ
  /-- Extension of the spring -/
  x : ℝ

/-- The theorem stating that the density of the weight material is half the density of the fluid -/
theorem weight_density_half_fluid_density (system : SpringMassSystem) 
  (h1 : system.k * system.x = system.m * system.g)  -- Force balance in air
  (h2 : system.m * system.g + system.k * system.x = system.ρ_fluid * system.g * (system.m / system.ρ_material))  -- Force balance in fluid
  (h3 : system.ρ_fluid > 0)  -- Fluid density is positive
  (h4 : system.m > 0)  -- Mass is positive
  (h5 : system.g > 0)  -- Gravity is positive
  : system.ρ_material = system.ρ_fluid / 2 := by
  sorry

#eval 800 / 2  -- Should output 400

end NUMINAMATH_CALUDE_weight_density_half_fluid_density_l3108_310833


namespace NUMINAMATH_CALUDE_mara_pink_crayons_percentage_l3108_310832

/-- The percentage of Mara's crayons that are pink -/
def mara_pink_percentage : ℝ := 10

theorem mara_pink_crayons_percentage 
  (mara_total : ℕ) 
  (luna_total : ℕ) 
  (luna_pink_percentage : ℝ) 
  (total_pink : ℕ) 
  (h1 : mara_total = 40)
  (h2 : luna_total = 50)
  (h3 : luna_pink_percentage = 20)
  (h4 : total_pink = 14)
  : mara_pink_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_mara_pink_crayons_percentage_l3108_310832


namespace NUMINAMATH_CALUDE_tangent_line_through_point_l3108_310811

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the tangent line equation
def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := 3 * x₀^2 * (x - x₀) + x₀^3

-- State the theorem
theorem tangent_line_through_point :
  ∃ (x₀ : ℝ), (tangent_line x₀ 1 = 1) ∧
  ((tangent_line x₀ x = 3*x - 2) ∨ (tangent_line x₀ x = 3/4*x + 1/4)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_through_point_l3108_310811


namespace NUMINAMATH_CALUDE_coefficient_x3y3_equals_15_l3108_310813

/-- The coefficient of x^3 * y^3 in the expansion of (x + y^2/x)(x + y)^5 -/
def coefficient_x3y3 (x y : ℝ) : ℝ :=
  let expanded := (x + y^2/x) * (x + y)^5
  sorry

theorem coefficient_x3y3_equals_15 :
  ∀ x y, x ≠ 0 → coefficient_x3y3 x y = 15 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y3_equals_15_l3108_310813


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3108_310830

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 2) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 2 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 2 ∧ x + y = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3108_310830


namespace NUMINAMATH_CALUDE_max_value_of_rational_function_l3108_310838

theorem max_value_of_rational_function : 
  (∀ x : ℝ, (5*x^2 + 10*x + 12) / (5*x^2 + 10*x + 2) ≤ 5) ∧ 
  (∀ ε > 0, ∃ x : ℝ, (5*x^2 + 10*x + 12) / (5*x^2 + 10*x + 2) > 5 - ε) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_rational_function_l3108_310838


namespace NUMINAMATH_CALUDE_sqrt_two_plus_x_l3108_310865

theorem sqrt_two_plus_x (x : ℝ) : x = Real.sqrt (2 + x) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_x_l3108_310865


namespace NUMINAMATH_CALUDE_defeat_crab_ways_l3108_310861

/-- Represents the number of claws on the giant enemy crab -/
def num_claws : ℕ := 2

/-- Represents the number of legs on the giant enemy crab -/
def num_legs : ℕ := 6

/-- Represents the minimum number of legs that must be cut before claws can be cut -/
def min_legs_before_claws : ℕ := 3

/-- The number of ways to defeat the giant enemy crab -/
def ways_to_defeat_crab : ℕ := num_legs.factorial * num_claws.factorial * (Nat.choose (num_legs + num_claws - min_legs_before_claws) num_claws)

/-- Theorem stating the number of ways to defeat the giant enemy crab -/
theorem defeat_crab_ways : ways_to_defeat_crab = 14400 := by
  sorry

end NUMINAMATH_CALUDE_defeat_crab_ways_l3108_310861


namespace NUMINAMATH_CALUDE_total_paths_count_l3108_310870

/-- Represents the number of paths between different types of points -/
structure PathCounts where
  redToBlue : Nat
  blueToGreen1 : Nat
  blueToGreen2 : Nat
  greenToOrange1 : Nat
  greenToOrange2 : Nat
  orange1ToB : Nat
  orange2ToB : Nat

/-- Calculates the total number of paths from A to B -/
def totalPaths (p : PathCounts) : Nat :=
  let blueToGreen := p.blueToGreen1 * 2 + p.blueToGreen2 * 2
  let greenToOrange := p.greenToOrange1 + p.greenToOrange2
  (p.redToBlue * blueToGreen * greenToOrange * p.orange1ToB) +
  (p.redToBlue * blueToGreen * greenToOrange * p.orange2ToB)

/-- The theorem stating the total number of paths from A to B -/
theorem total_paths_count (p : PathCounts) 
  (h1 : p.redToBlue = 14)
  (h2 : p.blueToGreen1 = 5)
  (h3 : p.blueToGreen2 = 7)
  (h4 : p.greenToOrange1 = 4)
  (h5 : p.greenToOrange2 = 3)
  (h6 : p.orange1ToB = 2)
  (h7 : p.orange2ToB = 8) :
  totalPaths p = 5376 := by
  sorry

end NUMINAMATH_CALUDE_total_paths_count_l3108_310870


namespace NUMINAMATH_CALUDE_first_player_wins_6x8_l3108_310868

/-- Represents a chocolate bar game -/
structure ChocolateGame where
  rows : ℕ
  cols : ℕ

/-- Calculates the total number of moves in a chocolate bar game -/
def totalMoves (game : ChocolateGame) : ℕ :=
  game.rows * game.cols - 1

/-- Determines if the first player wins the game -/
def firstPlayerWins (game : ChocolateGame) : Prop :=
  Odd (totalMoves game)

/-- Theorem: The first player wins in a 6x8 chocolate bar game -/
theorem first_player_wins_6x8 :
  firstPlayerWins ⟨6, 8⟩ := by sorry

end NUMINAMATH_CALUDE_first_player_wins_6x8_l3108_310868


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3108_310878

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3108_310878


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3108_310875

/-- An isosceles triangle with given altitude and perimeter has area 75 -/
theorem isosceles_triangle_area (b s : ℝ) : 
  b > 0 → s > 0 → 
  2 * s + 2 * b = 40 → -- perimeter condition
  b ^ 2 + 10 ^ 2 = s ^ 2 → -- Pythagorean theorem
  (2 * b) * 10 / 2 = 75 := by 
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l3108_310875


namespace NUMINAMATH_CALUDE_min_xy_value_l3108_310827

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 3 / (2 + x) + 3 / (2 + y) = 1) : 
  ∀ z, z = x * y → z ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l3108_310827


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_l3108_310845

theorem smallest_four_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 1000 ∧ n < 10000) ∧ 
    (75 * n) % 375 = 225 ∧
    (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → (75 * m) % 375 = 225 → m ≥ n) ∧
    n = 1003 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_l3108_310845


namespace NUMINAMATH_CALUDE_complement_of_union_l3108_310889

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2}

-- Define set N
def N : Finset Nat := {3, 4}

-- Theorem statement
theorem complement_of_union (u : Finset Nat) (m n : Finset Nat) 
  (hU : u = U) (hM : m = M) (hN : n = N) : 
  u \ (m ∪ n) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l3108_310889
