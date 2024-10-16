import Mathlib

namespace NUMINAMATH_CALUDE_husband_additional_payment_l3311_331166

/-- Calculates the additional amount the husband needs to pay to split expenses equally for the house help -/
theorem husband_additional_payment (salary : ℝ) (medical_cost : ℝ) 
  (h1 : salary = 160)
  (h2 : medical_cost = 128)
  (h3 : salary ≥ medical_cost / 2) : 
  salary / 2 - medical_cost / 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_husband_additional_payment_l3311_331166


namespace NUMINAMATH_CALUDE_yellow_leaves_count_l3311_331143

theorem yellow_leaves_count (thursday_leaves friday_leaves : ℕ) 
  (brown_percent green_percent : ℚ) :
  thursday_leaves = 12 →
  friday_leaves = 13 →
  brown_percent = 1/5 →
  green_percent = 1/5 →
  (thursday_leaves + friday_leaves : ℚ) * (1 - brown_percent - green_percent) = 15 :=
by sorry

end NUMINAMATH_CALUDE_yellow_leaves_count_l3311_331143


namespace NUMINAMATH_CALUDE_min_value_theorem_l3311_331150

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  2/x + 9/(1-2*x) ≥ 25 ∧ ∃ y, 0 < y ∧ y < 1/2 ∧ 2/y + 9/(1-2*y) = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3311_331150


namespace NUMINAMATH_CALUDE_revenue_percent_change_l3311_331174

/-- Calculates the percent change in revenue given initial conditions and tax changes -/
theorem revenue_percent_change 
  (initial_consumption : ℝ)
  (initial_tax_rate : ℝ)
  (tax_decrease_percent : ℝ)
  (consumption_increase_percent : ℝ)
  (additional_tax_decrease_percent : ℝ)
  (h1 : initial_consumption = 150)
  (h2 : tax_decrease_percent = 0.2)
  (h3 : consumption_increase_percent = 0.2)
  (h4 : additional_tax_decrease_percent = 0.02)
  (h5 : initial_consumption * (1 + consumption_increase_percent) < 200) :
  let new_consumption := initial_consumption * (1 + consumption_increase_percent)
  let new_tax_rate := initial_tax_rate * (1 - tax_decrease_percent - additional_tax_decrease_percent)
  let initial_revenue := initial_consumption * initial_tax_rate
  let new_revenue := new_consumption * new_tax_rate
  let percent_change := (new_revenue - initial_revenue) / initial_revenue * 100
  percent_change = -6.4 := by
sorry

end NUMINAMATH_CALUDE_revenue_percent_change_l3311_331174


namespace NUMINAMATH_CALUDE_sum_interior_angles_formula_sum_interior_angles_correct_l3311_331136

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2 : ℝ) * 180

theorem sum_interior_angles_formula (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2 : ℝ) * 180 :=
by
  sorry

/-- The sum of interior angles of a triangle -/
axiom triangle_sum : sum_interior_angles 3 = 180

/-- The sum of interior angles of a quadrilateral -/
axiom quadrilateral_sum : sum_interior_angles 4 = 360

theorem sum_interior_angles_correct (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2 : ℝ) * 180 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_formula_sum_interior_angles_correct_l3311_331136


namespace NUMINAMATH_CALUDE_function_inequality_l3311_331141

-- Define the function f(x) = ax - x^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

-- State the theorem
theorem function_inequality (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f a x₂ - f a x₁ > x₂ - x₁) →
  a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3311_331141


namespace NUMINAMATH_CALUDE_division_and_addition_l3311_331133

theorem division_and_addition : (12 / (1/6)) + 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l3311_331133


namespace NUMINAMATH_CALUDE_alphabet_proof_main_theorem_l3311_331147

/-- Represents an alphabet with letters containing dots and/or straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  line_only : ℕ
  dot_only : ℕ
  h_total : total = both + line_only + dot_only
  h_all_types : total > 0

/-- The specific alphabet described in the problem -/
def problem_alphabet : Alphabet where
  total := 40
  both := 9
  line_only := 24
  dot_only := 7
  h_total := by rfl
  h_all_types := by norm_num

/-- Theorem stating that the problem_alphabet satisfies the given conditions -/
theorem alphabet_proof : 
  ∃ (a : Alphabet), 
    a.total = 40 ∧ 
    a.both = 9 ∧ 
    a.line_only = 24 ∧ 
    a.dot_only = 7 :=
by
  use problem_alphabet
  simp [problem_alphabet]

/-- Main theorem to prove -/
theorem main_theorem (a : Alphabet) 
  (h1 : a.total = 40)
  (h2 : a.both = 9)
  (h3 : a.line_only = 24) :
  a.dot_only = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_alphabet_proof_main_theorem_l3311_331147


namespace NUMINAMATH_CALUDE_ball_selection_properties_l3311_331162

structure BallSelection where
  total_balls : Nat
  red_balls : Nat
  white_balls : Nat
  balls_drawn : Nat

def P (event : Set ℝ) : ℝ := sorry

def A (bs : BallSelection) : Set ℝ := sorry
def B (bs : BallSelection) : Set ℝ := sorry
def D (bs : BallSelection) : Set ℝ := sorry

theorem ball_selection_properties (bs : BallSelection) 
  (h1 : bs.total_balls = 4)
  (h2 : bs.red_balls = 2)
  (h3 : bs.white_balls = 2)
  (h4 : bs.balls_drawn = 2) :
  (P (A bs ∩ B bs) = P (A bs) * P (B bs)) ∧
  (P (A bs) + P (D bs) = 1) ∧
  (P (B bs ∩ D bs) = P (B bs) * P (D bs)) := by
  sorry

end NUMINAMATH_CALUDE_ball_selection_properties_l3311_331162


namespace NUMINAMATH_CALUDE_power_sum_l3311_331122

theorem power_sum (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l3311_331122


namespace NUMINAMATH_CALUDE_profit_calculation_l3311_331120

/-- Represents the profit distribution in a partnership --/
structure ProfitDistribution where
  mary_investment : ℝ
  mike_investment : ℝ
  total_profit : ℝ
  effort_share : ℝ
  investment_share : ℝ
  mary_extra : ℝ

/-- Theorem stating the profit calculation based on given conditions --/
theorem profit_calculation (pd : ProfitDistribution) 
  (h1 : pd.mary_investment = 600)
  (h2 : pd.mike_investment = 400)
  (h3 : pd.effort_share = 1/3)
  (h4 : pd.investment_share = 2/3)
  (h5 : pd.mary_extra = 1000)
  (h6 : pd.effort_share + pd.investment_share = 1) :
  pd.total_profit = 15000 := by
  sorry

#check profit_calculation

end NUMINAMATH_CALUDE_profit_calculation_l3311_331120


namespace NUMINAMATH_CALUDE_find_A_l3311_331139

theorem find_A : ∃ A : ℕ, ∃ B : ℕ, 
  (100 ≤ 600 + 10 * A + B) ∧ 
  (600 + 10 * A + B < 1000) ∧
  (600 + 10 * A + B - 41 = 591) ∧
  A = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l3311_331139


namespace NUMINAMATH_CALUDE_jessica_cut_four_orchids_l3311_331158

/-- The number of orchids Jessica cut from her garden -/
def orchids_cut (initial_orchids final_orchids : ℕ) : ℕ :=
  final_orchids - initial_orchids

/-- Theorem stating that Jessica cut 4 orchids -/
theorem jessica_cut_four_orchids :
  orchids_cut 3 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_four_orchids_l3311_331158


namespace NUMINAMATH_CALUDE_quadratic_radical_simplification_l3311_331161

theorem quadratic_radical_simplification :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x + y) = Real.sqrt x + Real.sqrt y → x = 0 ∨ y = 0) ∧
  (Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2) ∧
  (Real.sqrt (8 + 4 * Real.sqrt 3) = Real.sqrt 6 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_radical_simplification_l3311_331161


namespace NUMINAMATH_CALUDE_seats_per_bus_correct_l3311_331199

/-- Represents a school with classrooms, students, and buses for a field trip. -/
structure School where
  classrooms : ℕ
  students_per_classroom : ℕ
  seats_per_bus : ℕ

/-- Calculates the total number of students in the school. -/
def total_students (s : School) : ℕ :=
  s.classrooms * s.students_per_classroom

/-- Calculates the number of buses needed for the field trip. -/
def buses_needed (s : School) : ℕ :=
  (total_students s + s.seats_per_bus - 1) / s.seats_per_bus

/-- Theorem stating that for a school with 87 classrooms, 58 students per classroom,
    and buses with 29 seats each, the number of seats on each school bus is 29. -/
theorem seats_per_bus_correct (s : School) 
  (h1 : s.classrooms = 87)
  (h2 : s.students_per_classroom = 58)
  (h3 : s.seats_per_bus = 29) :
  s.seats_per_bus = 29 := by
  sorry

#eval buses_needed { classrooms := 87, students_per_classroom := 58, seats_per_bus := 29 }

end NUMINAMATH_CALUDE_seats_per_bus_correct_l3311_331199


namespace NUMINAMATH_CALUDE_roger_toy_purchase_l3311_331157

def max_toys_buyable (initial_amount : ℕ) (game_cost : ℕ) (toy_cost : ℕ) : ℕ :=
  (initial_amount - game_cost) / toy_cost

theorem roger_toy_purchase :
  max_toys_buyable 63 48 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_roger_toy_purchase_l3311_331157


namespace NUMINAMATH_CALUDE_area_triangle_ABC_in_special_cyclic_quadrilateral_l3311_331159

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Checks if a quadrilateral is cyclic (inscribed in a circle) -/
def isCyclic (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersectionPoint (A B C D : Point) : Point := sorry

/-- Theorem: Area of triangle ABC in a special cyclic quadrilateral -/
theorem area_triangle_ABC_in_special_cyclic_quadrilateral 
  (A B C D E : Point) (c : Circle) :
  isCyclic ⟨A, B, C, D⟩ c →
  E = intersectionPoint A C B D →
  A.x = D.x ∧ A.y = D.y →
  (C.x - E.x) / (E.x - D.x) = 3 / 2 ∧ (C.y - E.y) / (E.y - D.y) = 3 / 2 →
  triangleArea A B E = 8 →
  triangleArea A B C = 18 := by sorry

end NUMINAMATH_CALUDE_area_triangle_ABC_in_special_cyclic_quadrilateral_l3311_331159


namespace NUMINAMATH_CALUDE_x_4_sufficient_not_necessary_l3311_331106

def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)

def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

theorem x_4_sufficient_not_necessary :
  (∀ x : ℝ, x = 4 → magnitude_squared (vector_a x) = 25) ∧
  (∃ x : ℝ, x ≠ 4 ∧ magnitude_squared (vector_a x) = 25) := by
  sorry

end NUMINAMATH_CALUDE_x_4_sufficient_not_necessary_l3311_331106


namespace NUMINAMATH_CALUDE_circle_circumference_irrational_l3311_331181

theorem circle_circumference_irrational (d : ℚ) :
  Irrational (Real.pi * (d : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_irrational_l3311_331181


namespace NUMINAMATH_CALUDE_divisor_sum_ratio_l3311_331178

def N : ℕ := 48 * 49 * 75 * 343

def sum_of_divisors (n : ℕ) : ℕ := sorry

def sum_of_divisors_multiple_of_three (n : ℕ) : ℕ := sorry

def sum_of_divisors_not_multiple_of_three (n : ℕ) : ℕ := sorry

theorem divisor_sum_ratio :
  ∃ (a b : ℕ), 
    (sum_of_divisors_multiple_of_three N) * b = (sum_of_divisors_not_multiple_of_three N) * a ∧
    a ≠ 0 ∧ b ≠ 0 ∧ 
    (∀ (c d : ℕ), c ≠ 0 → d ≠ 0 → 
      (sum_of_divisors_multiple_of_three N) * d = (sum_of_divisors_not_multiple_of_three N) * c →
      a ≤ c ∧ b ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_divisor_sum_ratio_l3311_331178


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3311_331156

/-- A line passing through a point and perpendicular to another line -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  let point : ℝ × ℝ := (1, 2)
  let given_line (x y : ℝ) : Prop := x + 2*y - 1 = 0
  let perpendicular_line (x y : ℝ) : Prop := 2*x - y = 0
  (perpendicular_line x y ∧ (x, y) = point) →
  (∃ (m : ℝ), (y - 2) = m * (x - 1) ∧ m * (-1/2) = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3311_331156


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l3311_331160

/-- The volume of the tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (s : ℝ) (h : s = 8) : 
  let cube_volume := s^3
  let small_tetrahedron_volume := (1/3) * (1/2 * s^2) * s
  let purple_tetrahedron_volume := cube_volume - 4 * small_tetrahedron_volume
  purple_tetrahedron_volume = 512 - (1024/3) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l3311_331160


namespace NUMINAMATH_CALUDE_point_P_satisfies_conditions_l3311_331189

def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (-9, 2)
def P : ℝ × ℝ := (-3, -1)

def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def lies_on_line (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

theorem point_P_satisfies_conditions :
  lies_on_line A B P ∧ vector A P = (1/2 : ℝ) • vector A B := by sorry

end NUMINAMATH_CALUDE_point_P_satisfies_conditions_l3311_331189


namespace NUMINAMATH_CALUDE_triangle_area_is_five_l3311_331187

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 5

/-- The y-intercept of the line -/
def y_intercept : ℝ := -2

/-- The area of the triangle -/
def triangle_area : ℝ := 5

/-- Theorem: The area of the triangle formed by the line 2x - 5y - 10 = 0 and the coordinate axes is 5 -/
theorem triangle_area_is_five : 
  triangle_area = (1/2) * x_intercept * (-y_intercept) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_five_l3311_331187


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3311_331127

theorem complex_equation_solution (i : ℂ) (z : ℂ) 
  (h1 : i * i = -1)
  (h2 : z * (1 - i) = 3 + 2 * i) :
  z = 1/2 + 5/2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3311_331127


namespace NUMINAMATH_CALUDE_greater_than_negative_one_by_two_l3311_331192

theorem greater_than_negative_one_by_two : 
  (fun x => x > -1 ∧ x - (-1) = 2) 1 := by sorry

end NUMINAMATH_CALUDE_greater_than_negative_one_by_two_l3311_331192


namespace NUMINAMATH_CALUDE_partner_c_profit_share_l3311_331179

/-- Given the investment ratios of partners A, B, and C, and a total profit,
    calculate C's share of the profit. -/
theorem partner_c_profit_share 
  (a b c : ℝ) -- Investments of partners A, B, and C
  (total_profit : ℝ) -- Total profit
  (ha : a = 3 * b) -- A invests 3 times as much as B
  (hc : a = 2 / 3 * c) -- A invests 2/3 of what C invests
  : c / (a + b + c) * total_profit = 9 / 17 * total_profit :=
by sorry

end NUMINAMATH_CALUDE_partner_c_profit_share_l3311_331179


namespace NUMINAMATH_CALUDE_tank_filling_time_l3311_331132

theorem tank_filling_time (p q r s : ℚ) 
  (hp : p = 1/2) (hq : q = 1/4) (hr : r = 1/12) (hs : s = 1/6) :
  p + q + r + s = 1 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l3311_331132


namespace NUMINAMATH_CALUDE_zeroes_at_end_of_600_times_50_l3311_331121

theorem zeroes_at_end_of_600_times_50 : ∃ n : ℕ, 600 * 50 = n * 10000 ∧ n % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zeroes_at_end_of_600_times_50_l3311_331121


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l3311_331123

theorem final_sum_after_operations (S a b : ℝ) : 
  a + b = S → 3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l3311_331123


namespace NUMINAMATH_CALUDE_parallel_lines_coefficient_product_l3311_331184

/-- Two parallel lines with a specific distance between them -/
structure ParallelLines where
  a : ℝ
  b : ℝ
  l₁ : (x y : ℝ) → a * x + 2 * y + b = 0
  l₂ : (x y : ℝ) → (a - 1) * x + y + b = 0
  parallel : ∀ (x y : ℝ), a * x + 2 * y = (a - 1) * x + y
  distance : ∃ (k : ℝ), k * (b - 0) / Real.sqrt ((a - (a - 1))^2 + (2 - 1)^2) = Real.sqrt 2 / 2 ∧ k = 1 ∨ k = -1

/-- The product of coefficients a and b for parallel lines with specific distance -/
theorem parallel_lines_coefficient_product (pl : ParallelLines) : pl.a * pl.b = 4 ∨ pl.a * pl.b = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_coefficient_product_l3311_331184


namespace NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l3311_331138

theorem right_triangle_median_to_hypotenuse (DE DF : ℝ) :
  DE = 15 →
  DF = 20 →
  let EF := Real.sqrt (DE^2 + DF^2)
  let median := EF / 2
  median = 12.5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l3311_331138


namespace NUMINAMATH_CALUDE_speed_limit_calculation_l3311_331169

/-- Proves that given a distance of 150 miles traveled in 2 hours,
    and driving 15 mph above the speed limit, the speed limit is 60 mph. -/
theorem speed_limit_calculation (distance : ℝ) (time : ℝ) (speed_above_limit : ℝ) 
    (h1 : distance = 150)
    (h2 : time = 2)
    (h3 : speed_above_limit = 15) :
    distance / time - speed_above_limit = 60 := by
  sorry

end NUMINAMATH_CALUDE_speed_limit_calculation_l3311_331169


namespace NUMINAMATH_CALUDE_original_proposition_true_converse_false_l3311_331172

theorem original_proposition_true_converse_false :
  (∀ a b : ℝ, a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧
  (∃ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) ∧ a + b < 2) :=
by sorry

end NUMINAMATH_CALUDE_original_proposition_true_converse_false_l3311_331172


namespace NUMINAMATH_CALUDE_lottery_win_probability_l3311_331154

/-- Represents a lottery with a total number of tickets, cash prizes, and merchandise prizes. -/
structure Lottery where
  total_tickets : ℕ
  cash_prizes : ℕ
  merchandise_prizes : ℕ

/-- Calculates the probability of winning any prize in the lottery with a single ticket. -/
def win_probability (l : Lottery) : ℚ :=
  (l.cash_prizes + l.merchandise_prizes : ℚ) / l.total_tickets

/-- Theorem stating that the probability of winning any prize in the given lottery is 0.025. -/
theorem lottery_win_probability :
  let l : Lottery := ⟨1000, 5, 20⟩
  win_probability l = 25 / 1000 := by sorry

end NUMINAMATH_CALUDE_lottery_win_probability_l3311_331154


namespace NUMINAMATH_CALUDE_train_crossing_time_l3311_331113

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 50 →
  train_speed_kmh = 60 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3311_331113


namespace NUMINAMATH_CALUDE_star_3_7_l3311_331167

-- Define the star operation
def star (a b : ℕ) : ℕ := a^2 + 3*a*b + b^2

-- Theorem statement
theorem star_3_7 : star 3 7 = 121 := by
  sorry

end NUMINAMATH_CALUDE_star_3_7_l3311_331167


namespace NUMINAMATH_CALUDE_a_7_greater_than_3_l3311_331124

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sequence is monotonically increasing -/
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The theorem statement -/
theorem a_7_greater_than_3 (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : monotonically_increasing a) 
  (h3 : a 1 + a 10 = 6) : 
  a 7 > 3 := by
  sorry


end NUMINAMATH_CALUDE_a_7_greater_than_3_l3311_331124


namespace NUMINAMATH_CALUDE_investment_difference_l3311_331183

/-- Represents the final value of an investment given its initial value and growth factor -/
def final_value (initial : ℝ) (growth : ℝ) : ℝ := initial * growth

/-- Theorem: Given the initial investments and their changes in value, 
    the difference between Jackson's final investment value and 
    the combined final investment values of Brandon and Meagan is $850 -/
theorem investment_difference : 
  let jackson_initial := (500 : ℝ)
  let brandon_initial := (500 : ℝ)
  let meagan_initial := (700 : ℝ)
  let jackson_growth := (4 : ℝ)
  let brandon_growth := (0.2 : ℝ)
  let meagan_growth := (1.5 : ℝ)
  final_value jackson_initial jackson_growth - 
  (final_value brandon_initial brandon_growth + final_value meagan_initial meagan_growth) = 850 := by
  sorry


end NUMINAMATH_CALUDE_investment_difference_l3311_331183


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3311_331101

theorem product_of_three_numbers (a b c : ℚ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 2 * (b + c))
  (second_eq : b = 6 * c) : 
  a * b * c = 12000 / 49 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3311_331101


namespace NUMINAMATH_CALUDE_stratified_sampling_selection_l3311_331108

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of liberal arts students -/
def liberal_arts_students : ℕ := 5

/-- The number of science students -/
def science_students : ℕ := 10

/-- The number of liberal arts students to be selected -/
def selected_liberal_arts : ℕ := 2

/-- The number of science students to be selected -/
def selected_science : ℕ := 4

theorem stratified_sampling_selection :
  (binomial liberal_arts_students selected_liberal_arts) * 
  (binomial science_students selected_science) = 2100 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_selection_l3311_331108


namespace NUMINAMATH_CALUDE_pirate_theorem_l3311_331171

/-- Represents a pirate in the group -/
structure Pirate where
  id : Nat
  targets : Finset Nat

/-- Represents the group of pirates -/
def PirateGroup := Finset Pirate

/-- Counts the number of pirates killed in a given order -/
def countKilled (group : PirateGroup) (order : List Pirate) : Nat :=
  sorry

/-- Main theorem: If there exists an order where 28 pirates are killed,
    then in any other order, at least 10 pirates must be killed -/
theorem pirate_theorem (group : PirateGroup) :
  (∃ order : List Pirate, countKilled group order = 28) →
  (∀ order : List Pirate, countKilled group order ≥ 10) :=
by
  sorry


end NUMINAMATH_CALUDE_pirate_theorem_l3311_331171


namespace NUMINAMATH_CALUDE_winning_percentage_approx_l3311_331104

/-- Represents the votes received by each candidate in an election -/
structure ElectionResults where
  candidates : Fin 3 → ℕ
  candidate1_votes : candidates 0 = 3000
  candidate2_votes : candidates 1 = 5000
  candidate3_votes : candidates 2 = 20000

/-- Calculates the total number of votes in the election -/
def totalVotes (results : ElectionResults) : ℕ :=
  (results.candidates 0) + (results.candidates 1) + (results.candidates 2)

/-- Finds the maximum number of votes received by any candidate -/
def maxVotes (results : ElectionResults) : ℕ :=
  max (results.candidates 0) (max (results.candidates 1) (results.candidates 2))

/-- Calculates the percentage of votes received by the winning candidate -/
def winningPercentage (results : ElectionResults) : ℚ :=
  (maxVotes results : ℚ) / (totalVotes results : ℚ) * 100

/-- Theorem stating that the winning percentage is approximately 71.43% -/
theorem winning_percentage_approx (results : ElectionResults) :
  ∃ ε > 0, abs (winningPercentage results - 71.43) < ε :=
sorry

end NUMINAMATH_CALUDE_winning_percentage_approx_l3311_331104


namespace NUMINAMATH_CALUDE_different_city_probability_l3311_331170

theorem different_city_probability (pA_cityA pB_cityA : ℝ) 
  (h1 : 0 ≤ pA_cityA ∧ pA_cityA ≤ 1)
  (h2 : 0 ≤ pB_cityA ∧ pB_cityA ≤ 1)
  (h3 : pA_cityA = 0.6)
  (h4 : pB_cityA = 0.2) :
  (pA_cityA * (1 - pB_cityA)) + ((1 - pA_cityA) * pB_cityA) = 0.56 := by
sorry

end NUMINAMATH_CALUDE_different_city_probability_l3311_331170


namespace NUMINAMATH_CALUDE_least_integer_square_36_more_than_thrice_neg_six_satisfies_equation_least_integer_square_36_more_than_thrice_is_neg_six_l3311_331182

theorem least_integer_square_36_more_than_thrice (x : ℤ) : x^2 = 3*x + 36 → x ≥ -6 := by
  sorry

theorem neg_six_satisfies_equation : (-6 : ℤ)^2 = 3*(-6) + 36 := by
  sorry

theorem least_integer_square_36_more_than_thrice_is_neg_six :
  ∃ (x : ℤ), x^2 = 3*x + 36 ∧ ∀ (y : ℤ), y^2 = 3*y + 36 → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_least_integer_square_36_more_than_thrice_neg_six_satisfies_equation_least_integer_square_36_more_than_thrice_is_neg_six_l3311_331182


namespace NUMINAMATH_CALUDE_sunset_colors_l3311_331105

/-- Represents the number of colors in a quick shift -/
def quick_colors : ℕ := 5

/-- Represents the number of colors in a slow shift -/
def slow_colors : ℕ := 2

/-- Represents the duration of each shift in minutes -/
def shift_duration : ℕ := 10

/-- Represents the duration of a complete cycle (quick + slow) in minutes -/
def cycle_duration : ℕ := 2 * shift_duration

/-- Represents the duration of the sunset in minutes -/
def sunset_duration : ℕ := 2 * 60

/-- Represents the number of cycles in the sunset -/
def num_cycles : ℕ := sunset_duration / cycle_duration

/-- Represents the total number of colors in one cycle -/
def colors_per_cycle : ℕ := quick_colors + slow_colors

/-- Theorem stating that the total number of colors seen during the sunset is 42 -/
theorem sunset_colors : num_cycles * colors_per_cycle = 42 := by
  sorry

end NUMINAMATH_CALUDE_sunset_colors_l3311_331105


namespace NUMINAMATH_CALUDE_min_white_surface_fraction_l3311_331152

/-- Represents a cube with given edge length -/
structure Cube where
  edge_length : ℕ

/-- Represents the large cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cubes : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.edge_length^2

/-- Theorem: The minimum fraction of white surface area in the described cube configuration is 1/12 -/
theorem min_white_surface_fraction (lc : LargeCube) 
  (h1 : lc.edge_length = 4)
  (h2 : lc.small_cubes = 64)
  (h3 : lc.red_cubes = 48)
  (h4 : lc.white_cubes = 16) :
  ∃ (white_area : ℕ), 
    white_area ≤ lc.white_cubes ∧ 
    (white_area : ℚ) / (surface_area ⟨lc.edge_length⟩ : ℚ) = 1/12 ∧
    ∀ (other_white_area : ℕ), 
      other_white_area ≤ lc.white_cubes → 
      (other_white_area : ℚ) / (surface_area ⟨lc.edge_length⟩ : ℚ) ≥ 1/12 := by
  sorry

end NUMINAMATH_CALUDE_min_white_surface_fraction_l3311_331152


namespace NUMINAMATH_CALUDE_license_plate_combinations_license_plate_combinations_eq_187200_l3311_331163

theorem license_plate_combinations : ℕ :=
  let total_letters : ℕ := 26
  let letter_positions : ℕ := 4
  let digit_positions : ℕ := 3
  let repeated_letter_choices : ℕ := total_letters
  let non_repeated_letter_choices : ℕ := total_letters - 1
  let repeated_letter_arrangements : ℕ := Nat.choose letter_positions (letter_positions - 1)
  let first_digit_choices : ℕ := 10
  let second_digit_choices : ℕ := 9
  let third_digit_choices : ℕ := 8

  repeated_letter_choices * non_repeated_letter_choices * repeated_letter_arrangements *
  first_digit_choices * second_digit_choices * third_digit_choices

theorem license_plate_combinations_eq_187200 : license_plate_combinations = 187200 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_license_plate_combinations_eq_187200_l3311_331163


namespace NUMINAMATH_CALUDE_minimize_expression_l3311_331117

theorem minimize_expression (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) ≥ 3 ∧ (x + 4 / (x + 1) = 3 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_minimize_expression_l3311_331117


namespace NUMINAMATH_CALUDE_trays_per_trip_is_eight_l3311_331151

-- Define the problem parameters
def trays_table1 : ℕ := 27
def trays_table2 : ℕ := 5
def total_trips : ℕ := 4

-- Define the total number of trays
def total_trays : ℕ := trays_table1 + trays_table2

-- Theorem statement
theorem trays_per_trip_is_eight :
  total_trays / total_trips = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_trays_per_trip_is_eight_l3311_331151


namespace NUMINAMATH_CALUDE_officer_combinations_count_l3311_331145

def totalMembers : ℕ := 25
def officerPositions : ℕ := 3

def chooseOfficers (n m : ℕ) : ℕ := n * (n - 1) * (n - 2)

def officersCombinations : ℕ :=
  let withoutPairs := chooseOfficers (totalMembers - 4) officerPositions
  let withAliceBob := 3 * 2 * (totalMembers - 4)
  let withCharlesDiana := 3 * 2 * (totalMembers - 4)
  withoutPairs + withAliceBob + withCharlesDiana

theorem officer_combinations_count :
  officersCombinations = 8232 :=
sorry

end NUMINAMATH_CALUDE_officer_combinations_count_l3311_331145


namespace NUMINAMATH_CALUDE_solve_for_y_l3311_331142

theorem solve_for_y (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -6) : y = 29 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3311_331142


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3311_331149

theorem inequality_system_solution :
  ∃ (x y : ℝ), 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧
    (2 * x - 4 * y ≤ -3) ∧
    (x = -1/3) ∧ (y = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3311_331149


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l3311_331185

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (keeper_age_diff : ℕ) (team_avg_age : ℚ) : 
  team_size = 11 →
  captain_age = 27 →
  keeper_age_diff = 3 →
  team_avg_age = 24 →
  let keeper_age := captain_age + keeper_age_diff
  let total_age := team_avg_age * team_size
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + keeper_age)
  let remaining_avg_age := remaining_age / remaining_players
  remaining_avg_age = team_avg_age - 1 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l3311_331185


namespace NUMINAMATH_CALUDE_total_players_count_l3311_331140

/-- Represents the number of people playing kabaddi -/
def kabaddi_players : ℕ := 10

/-- Represents the number of people playing kho kho only -/
def kho_kho_only_players : ℕ := 35

/-- Represents the number of people playing both games -/
def both_games_players : ℕ := 5

/-- Calculates the total number of players -/
def total_players : ℕ := kabaddi_players - both_games_players + kho_kho_only_players + both_games_players

theorem total_players_count : total_players = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_players_count_l3311_331140


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3311_331130

theorem line_circle_intersection (a : ℝ) :
  ∃ (x y : ℝ), (a * x - y + 2 * a = 0) ∧ (x^2 + y^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3311_331130


namespace NUMINAMATH_CALUDE_solution_to_diophantine_equation_l3311_331197

theorem solution_to_diophantine_equation :
  ∀ x y z : ℕ+,
    x ≤ y ∧ y ≤ z →
    5 * (x * y + y * z + z * x) = 4 * x * y * z →
    ((x = 2 ∧ y = 5 ∧ z = 10) ∨ (x = 2 ∧ y = 4 ∧ z = 20)) :=
by sorry

end NUMINAMATH_CALUDE_solution_to_diophantine_equation_l3311_331197


namespace NUMINAMATH_CALUDE_town_population_l3311_331115

def present_population : ℝ → Prop :=
  λ p => (1 + 0.04) * p = 1289.6

theorem town_population : ∃ p : ℝ, present_population p ∧ p = 1240 :=
  sorry

end NUMINAMATH_CALUDE_town_population_l3311_331115


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3311_331168

theorem sqrt_sum_inequality (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) 
  (sum_eq : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3311_331168


namespace NUMINAMATH_CALUDE_mean_of_two_numbers_l3311_331116

def numbers : List ℕ := [1871, 1997, 2023, 2029, 2113, 2125, 2137]

def sum_of_all : ℕ := numbers.sum

def mean_of_five : ℕ := 2100

def sum_of_five : ℕ := 5 * mean_of_five

def sum_of_two : ℕ := sum_of_all - sum_of_five

theorem mean_of_two_numbers : (sum_of_two : ℚ) / 2 = 1397.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_two_numbers_l3311_331116


namespace NUMINAMATH_CALUDE_m_values_l3311_331173

def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3*m + 2}

theorem m_values (m : ℝ) (h : 2 ∈ A m) : m = 0 ∨ m = 2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_values_l3311_331173


namespace NUMINAMATH_CALUDE_box_volume_increase_l3311_331188

theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 4860)
  (hs : 2 * (l * w + w * h + l * h) = 1860)
  (he : 4 * (l + w + h) = 224) :
  (l + 2) * (w + 3) * (h + 1) = 5964 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l3311_331188


namespace NUMINAMATH_CALUDE_smallest_12_digit_with_all_digits_div_36_proof_l3311_331103

def is_12_digit (n : ℕ) : Prop := 10^11 ≤ n ∧ n < 10^12

def contains_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, n / 10^k % 10 = d

def smallest_12_digit_with_all_digits_div_36 : ℕ := 100023457896

theorem smallest_12_digit_with_all_digits_div_36_proof :
  (is_12_digit smallest_12_digit_with_all_digits_div_36) ∧
  (contains_all_digits smallest_12_digit_with_all_digits_div_36) ∧
  (smallest_12_digit_with_all_digits_div_36 % 36 = 0) ∧
  (∀ m : ℕ, m < smallest_12_digit_with_all_digits_div_36 →
    ¬(is_12_digit m ∧ contains_all_digits m ∧ m % 36 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_12_digit_with_all_digits_div_36_proof_l3311_331103


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3311_331190

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_sixth : a 6 = Nat.factorial 9)
  (h_ninth : a 9 = Nat.factorial 10) :
  a 1 = (Nat.factorial 9) / (10 * Real.rpow 10 (1/3)) := by
  sorry

#check geometric_sequence_first_term

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3311_331190


namespace NUMINAMATH_CALUDE_equation_solution_l3311_331175

theorem equation_solution (x : ℝ) : 
  (x^2 - 36) / 3 = (x^2 + 3*x + 9) / 6 ↔ x = 9 ∨ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3311_331175


namespace NUMINAMATH_CALUDE_stating_broken_flagpole_height_l3311_331191

/-- Represents a broken flagpole scenario -/
structure BrokenFlagpole where
  initial_height : ℝ
  distance_from_base : ℝ
  break_height : ℝ

/-- 
Theorem stating that for a flagpole of height 8 meters, if it breaks at a point x meters 
above the ground and the upper part touches the ground 3 meters away from the base, 
then x = √73 / 2.
-/
theorem broken_flagpole_height (f : BrokenFlagpole) 
    (h1 : f.initial_height = 8)
    (h2 : f.distance_from_base = 3) :
  f.break_height = Real.sqrt 73 / 2 := by
  sorry


end NUMINAMATH_CALUDE_stating_broken_flagpole_height_l3311_331191


namespace NUMINAMATH_CALUDE_gcf_of_60_and_75_l3311_331135

theorem gcf_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_and_75_l3311_331135


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l3311_331195

theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ((2 / a + 8 / b) ≥ 9) ∧
  (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 2 / a' + 8 / b' = 9 ∧ a' = 2 / 3 ∧ b' = 4 / 3) ∧
  (a^2 + b^2 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l3311_331195


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3311_331114

theorem fraction_sum_equality : (3 : ℚ) / 10 + 5 / 100 - 1 / 1000 = 349 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3311_331114


namespace NUMINAMATH_CALUDE_range_of_a_for_sqrt_function_l3311_331146

theorem range_of_a_for_sqrt_function (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = Real.sqrt (x^2 + 2*a*x + 1)) → 
  (∀ x, ∃ y, f x = y) →
  -1 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_sqrt_function_l3311_331146


namespace NUMINAMATH_CALUDE_value_of_b_l3311_331125

theorem value_of_b (b : ℚ) (h : b - b/4 = 5/2) : b = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l3311_331125


namespace NUMINAMATH_CALUDE_correct_statements_l3311_331165

-- Define the data sets
def dataSetA : List ℝ := sorry
def dataSetB : List ℝ := sorry
def dataSetC : List ℝ := [1, 2, 5, 5, 5, 3, 3]

-- Define variance function
def variance (data : List ℝ) : ℝ := sorry

-- Define median function
def median (data : List ℝ) : ℝ := sorry

-- Define mode function
def mode (data : List ℝ) : ℝ := sorry

-- Theorem to prove
theorem correct_statements :
  -- Statement A is incorrect
  ¬ (∀ (n : ℕ), n = 1000 → ∃ (h : ℕ), h = 500 ∧ h = n / 2) ∧
  -- Statement B is correct
  (variance dataSetA = 0.03 ∧ variance dataSetB = 0.1 → variance dataSetA < variance dataSetB) ∧
  -- Statement C is correct
  (median dataSetC = 3 ∧ mode dataSetC = 5) ∧
  -- Statement D is incorrect
  ¬ (∀ (population : Type) (property : population → Prop),
     (∀ x : population, property x) ↔ (∃ survey : population → Prop, ∀ x, survey x → property x))
  := by sorry

end NUMINAMATH_CALUDE_correct_statements_l3311_331165


namespace NUMINAMATH_CALUDE_picnic_cost_is_60_l3311_331180

/-- Calculates the total cost of a picnic basket given the number of people and item prices. -/
def picnic_cost (num_people : ℕ) (sandwich_price fruit_salad_price soda_price snack_price : ℕ) : ℕ :=
  let sandwich_cost := num_people * sandwich_price
  let fruit_salad_cost := num_people * fruit_salad_price
  let soda_cost := num_people * 2 * soda_price
  let snack_cost := 3 * snack_price
  sandwich_cost + fruit_salad_cost + soda_cost + snack_cost

/-- Theorem stating that the total cost of the picnic basket is $60. -/
theorem picnic_cost_is_60 :
  picnic_cost 4 5 3 2 4 = 60 := by
  sorry

#eval picnic_cost 4 5 3 2 4

end NUMINAMATH_CALUDE_picnic_cost_is_60_l3311_331180


namespace NUMINAMATH_CALUDE_zoom_download_time_l3311_331177

theorem zoom_download_time (total_time audio_glitch_time video_glitch_time : ℕ)
  (h_total : total_time = 82)
  (h_audio : audio_glitch_time = 2 * 4)
  (h_video : video_glitch_time = 6)
  (h_glitch_ratio : 2 * (audio_glitch_time + video_glitch_time) = total_time - (audio_glitch_time + video_glitch_time) - 40) :
  let mac_download_time := (total_time - (audio_glitch_time + video_glitch_time) - 2 * (audio_glitch_time + video_glitch_time)) / 4
  mac_download_time = 10 := by sorry

end NUMINAMATH_CALUDE_zoom_download_time_l3311_331177


namespace NUMINAMATH_CALUDE_kostya_bulbs_count_l3311_331126

/-- Function to calculate the number of bulbs after one round of planting -/
def plant_between (n : ℕ) : ℕ := 2 * n - 1

/-- Function to calculate the number of bulbs after three rounds of planting -/
def plant_three_rounds (n : ℕ) : ℕ := plant_between (plant_between (plant_between n))

/-- Theorem stating that if Kostya planted n bulbs and the final count after three rounds is 113, then n must be 15 -/
theorem kostya_bulbs_count : 
  ∀ n : ℕ, plant_three_rounds n = 113 → n = 15 := by
sorry

#eval plant_three_rounds 15  -- Should output 113

end NUMINAMATH_CALUDE_kostya_bulbs_count_l3311_331126


namespace NUMINAMATH_CALUDE_obtuse_triangle_one_obtuse_angle_equilateral_triangle_60_degree_angles_l3311_331131

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

end NUMINAMATH_CALUDE_obtuse_triangle_one_obtuse_angle_equilateral_triangle_60_degree_angles_l3311_331131


namespace NUMINAMATH_CALUDE_prob_sum_three_is_one_over_216_l3311_331107

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_die : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The target sum we're aiming for -/
def target_sum : ℕ := 3

/-- The probability of rolling a sum of 3 with three fair six-sided dice -/
def prob_sum_three : ℚ := prob_single_die ^ num_dice

theorem prob_sum_three_is_one_over_216 : 
  prob_sum_three = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_prob_sum_three_is_one_over_216_l3311_331107


namespace NUMINAMATH_CALUDE_pigeonhole_zodiac_signs_pigeonhole_birthday_weekday_l3311_331144

/-- The number of zodiac signs -/
def num_zodiac_signs : ℕ := 12

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The minimum number of employees needed to ensure at least two have the same zodiac sign -/
def min_employees_same_sign : ℕ := num_zodiac_signs + 1

/-- The minimum number of employees needed to ensure at least four have birthdays on the same day of the week -/
def min_employees_same_day : ℕ := days_in_week * 3 + 1

theorem pigeonhole_zodiac_signs :
  min_employees_same_sign = 13 :=
sorry

theorem pigeonhole_birthday_weekday :
  min_employees_same_day = 22 :=
sorry

end NUMINAMATH_CALUDE_pigeonhole_zodiac_signs_pigeonhole_birthday_weekday_l3311_331144


namespace NUMINAMATH_CALUDE_adas_original_seat_was_two_l3311_331148

/-- Represents the seats in the theater --/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends sitting in the theater --/
inductive Friend
| Ada
| Bea
| Ceci
| Dee
| Edie
| Faye

/-- Represents the state of the seating arrangement --/
structure SeatingArrangement where
  seats : Fin 6 → Option Friend

/-- Defines a valid initial seating arrangement --/
def validInitialArrangement (arr : SeatingArrangement) : Prop :=
  ∃ (emptySlot : Fin 6), 
    (∀ i : Fin 6, i ≠ emptySlot → arr.seats i ≠ none) ∧
    (arr.seats emptySlot = none) ∧
    (∃ (ada bea ceci dee edie faye : Fin 6), 
      ada ≠ bea ∧ ada ≠ ceci ∧ ada ≠ dee ∧ ada ≠ edie ∧ ada ≠ faye ∧
      bea ≠ ceci ∧ bea ≠ dee ∧ bea ≠ edie ∧ bea ≠ faye ∧
      ceci ≠ dee ∧ ceci ≠ edie ∧ ceci ≠ faye ∧
      dee ≠ edie ∧ dee ≠ faye ∧
      edie ≠ faye ∧
      arr.seats ada = some Friend.Ada ∧
      arr.seats bea = some Friend.Bea ∧
      arr.seats ceci = some Friend.Ceci ∧
      arr.seats dee = some Friend.Dee ∧
      arr.seats edie = some Friend.Edie ∧
      arr.seats faye = some Friend.Faye)

/-- Defines the final seating arrangement after movements --/
def finalArrangement (initial : SeatingArrangement) (final : SeatingArrangement) : Prop :=
  ∃ (bea bea' ceci ceci' dee dee' edie edie' : Fin 6),
    initial.seats bea = some Friend.Bea ∧
    initial.seats ceci = some Friend.Ceci ∧
    initial.seats dee = some Friend.Dee ∧
    initial.seats edie = some Friend.Edie ∧
    bea' = (bea + 3) % 6 ∧
    ceci' = (ceci + 2) % 6 ∧
    dee' ≠ dee ∧ edie' ≠ edie ∧
    final.seats bea' = some Friend.Bea ∧
    final.seats ceci' = some Friend.Ceci ∧
    final.seats dee' = some Friend.Dee ∧
    final.seats edie' = some Friend.Edie ∧
    (final.seats 0 = none ∨ final.seats 5 = none)

/-- Theorem: Ada's original seat was Seat 2 --/
theorem adas_original_seat_was_two 
  (initial final : SeatingArrangement)
  (h_initial : validInitialArrangement initial)
  (h_final : finalArrangement initial final) :
  ∃ (ada : Fin 6), initial.seats ada = some Friend.Ada ∧ ada = 1 := by
  sorry

end NUMINAMATH_CALUDE_adas_original_seat_was_two_l3311_331148


namespace NUMINAMATH_CALUDE_constant_sum_property_l3311_331176

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the ellipse -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 8 + p.y^2 / 4 = 1

/-- Definition of a line passing through a point -/
def Line (p : Point) (m : ℝ) :=
  {q : Point | q.y - p.y = m * (q.x - p.x)}

/-- The focus point of the ellipse -/
def F : Point := ⟨2, 0⟩

/-- Theorem stating the constant sum property -/
theorem constant_sum_property 
  (A B P : Point) 
  (hA : isOnEllipse A) 
  (hB : isOnEllipse B) 
  (hP : P.x = 0) 
  (hline : ∃ (m : ℝ), A ∈ Line F m ∧ B ∈ Line F m ∧ P ∈ Line F m)
  (m n : ℝ)
  (hm : (P.x - A.x, P.y - A.y) = m • (A.x - F.x, A.y - F.y))
  (hn : (P.x - B.x, P.y - B.y) = n • (B.x - F.x, B.y - F.y)) :
  m + n = -4 := by
    sorry


end NUMINAMATH_CALUDE_constant_sum_property_l3311_331176


namespace NUMINAMATH_CALUDE_inverse_proportion_product_l3311_331100

/-- Theorem: For points A(x₁, y₁) and B(x₂, y₂) on the graph of y = -3/x, 
    if x₁ * x₂ = 2, then y₁ * y₂ = 9/2 -/
theorem inverse_proportion_product (x₁ x₂ y₁ y₂ : ℝ) 
    (h1 : y₁ = -3 / x₁) 
    (h2 : y₂ = -3 / x₂) 
    (h3 : x₁ * x₂ = 2) : 
  y₁ * y₂ = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_product_l3311_331100


namespace NUMINAMATH_CALUDE_remainder_55_57_mod_7_l3311_331111

theorem remainder_55_57_mod_7 : (55 * 57) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_55_57_mod_7_l3311_331111


namespace NUMINAMATH_CALUDE_juan_birth_year_l3311_331198

def first_btc_year : ℕ := 1990
def btc_frequency : ℕ := 2
def juan_age_at_fifth_btc : ℕ := 14

def btc_year (n : ℕ) : ℕ := first_btc_year + (n - 1) * btc_frequency

theorem juan_birth_year :
  first_btc_year = 1990 →
  btc_frequency = 2 →
  juan_age_at_fifth_btc = 14 →
  btc_year 5 - juan_age_at_fifth_btc = 1984 :=
by
  sorry

end NUMINAMATH_CALUDE_juan_birth_year_l3311_331198


namespace NUMINAMATH_CALUDE_solution_set_l3311_331129

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := 2 * a - b + 3

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  otimes 0.5 x > -2 ∧ otimes (2 * x) 5 > 3 * x + 1

-- State the theorem
theorem solution_set :
  ∀ x : ℝ, inequality_system x ↔ 3 < x ∧ x < 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_l3311_331129


namespace NUMINAMATH_CALUDE_pen_cost_l3311_331118

theorem pen_cost (pen ink_refill pencil : ℝ) 
  (total_cost : pen + ink_refill + pencil = 2.35)
  (pen_ink_relation : pen = ink_refill + 1.50)
  (pencil_cost : pencil = 0.45) : 
  pen = 1.70 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l3311_331118


namespace NUMINAMATH_CALUDE_contradiction_method_correctness_l3311_331153

theorem contradiction_method_correctness :
  (∀ p q : ℝ, p^3 + q^3 = 2 → (¬(p + q ≤ 2) ↔ p + q > 2)) ∧
  (∀ a b : ℝ, |a| + |b| < 1 →
    (∃ x₁ x₂ : ℝ, x₁^2 + a*x₁ + b = 0 ∧ x₂^2 + a*x₂ + b = 0 ∧ x₁ ≠ x₂) →
    (∀ x : ℝ, x^2 + a*x + b = 0 → |x| < 1) ↔
    ¬(∃ x : ℝ, x^2 + a*x + b = 0 ∧ |x| ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_contradiction_method_correctness_l3311_331153


namespace NUMINAMATH_CALUDE_problem_statement_l3311_331137

theorem problem_statement (a b : ℝ) (h : a * b + b^2 = 12) :
  (a + b)^2 - (a + b) * (a - b) = 24 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3311_331137


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l3311_331155

theorem largest_integer_in_interval : ∃ x : ℤ, 
  (1 / 4 : ℚ) < (x : ℚ) / 9 ∧ 
  (x : ℚ) / 9 < (7 / 9 : ℚ) ∧ 
  ∀ y : ℤ, ((1 / 4 : ℚ) < (y : ℚ) / 9 ∧ (y : ℚ) / 9 < (7 / 9 : ℚ)) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l3311_331155


namespace NUMINAMATH_CALUDE_quadratic_function_condition_l3311_331102

theorem quadratic_function_condition (m : ℝ) : (|m - 2| = 2 ∧ m - 4 ≠ 0) ↔ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_condition_l3311_331102


namespace NUMINAMATH_CALUDE_orange_buckets_total_l3311_331186

theorem orange_buckets_total (x y : ℝ) : 
  x = 2 * 22.5 + 3 →
  y = x - 11.5 →
  22.5 + x + y = 107 := by
sorry

end NUMINAMATH_CALUDE_orange_buckets_total_l3311_331186


namespace NUMINAMATH_CALUDE_unique_solution_x_three_halves_l3311_331194

theorem unique_solution_x_three_halves :
  ∃! x : ℝ, ∀ y : ℝ, (y = (x^2 - 9) / (x - 3) ∧ y = 3*x) → x = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_x_three_halves_l3311_331194


namespace NUMINAMATH_CALUDE_lock_code_attempts_l3311_331193

theorem lock_code_attempts (num_digits : ℕ) (code_length : ℕ) : 
  num_digits = 5 → code_length = 3 → num_digits ^ code_length - 1 = 124 := by
  sorry

#eval 5^3 - 1  -- This should output 124

end NUMINAMATH_CALUDE_lock_code_attempts_l3311_331193


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3311_331128

-- Define a linear function f(x) = ax + b
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Theorem statement
theorem solution_set_of_inequality 
  (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f a b x < f a b y) 
  (h_intersect : f a b 2 = 0) :
  {x : ℝ | b * x^2 - a * x > 0} = {x : ℝ | -1/2 < x ∧ x < 0} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3311_331128


namespace NUMINAMATH_CALUDE_customized_bowling_ball_volume_l3311_331109

/-- The volume of a customized bowling ball -/
theorem customized_bowling_ball_volume :
  let ball_diameter : ℝ := 24
  let hole_depth : ℝ := 10
  let hole_diameters : List ℝ := [1.5, 1.5, 2, 2.5]
  let sphere_volume := (4 / 3) * π * (ball_diameter / 2) ^ 3
  let hole_volumes := hole_diameters.map (fun d => π * (d / 2) ^ 2 * hole_depth)
  sphere_volume - hole_volumes.sum = 2233.375 * π := by
  sorry

end NUMINAMATH_CALUDE_customized_bowling_ball_volume_l3311_331109


namespace NUMINAMATH_CALUDE_fraction_equality_implies_five_l3311_331196

theorem fraction_equality_implies_five (a b : ℕ) (h : (a + 30) / b = a / (b - 6)) : 
  (a + 30) / b = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_five_l3311_331196


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3311_331110

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3311_331110


namespace NUMINAMATH_CALUDE_scientific_notation_of_million_l3311_331164

theorem scientific_notation_of_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1000000 = a * (10 : ℝ) ^ n ∧ a = 1 ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_million_l3311_331164


namespace NUMINAMATH_CALUDE_container_volume_comparison_l3311_331119

theorem container_volume_comparison (x y : ℝ) (h : x ≠ y) :
  x^3 + y^3 > x^2*y + x*y^2 := by
  sorry

#check container_volume_comparison

end NUMINAMATH_CALUDE_container_volume_comparison_l3311_331119


namespace NUMINAMATH_CALUDE_triangle_inequality_l3311_331134

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b > c → b + c > a → a + c > b → 
  ¬(a = 5 ∧ b = 9 ∧ c = 4) :=
by
  sorry

#check triangle_inequality

end NUMINAMATH_CALUDE_triangle_inequality_l3311_331134


namespace NUMINAMATH_CALUDE_president_savings_l3311_331112

/-- Calculates the amount saved by the president for his reelection campaign --/
theorem president_savings (total_funds : ℝ) (friends_percentage : ℝ) (family_percentage : ℝ)
  (h1 : total_funds = 10000)
  (h2 : friends_percentage = 0.4)
  (h3 : family_percentage = 0.3) :
  total_funds - (friends_percentage * total_funds + family_percentage * (total_funds - friends_percentage * total_funds)) = 4200 :=
by sorry

end NUMINAMATH_CALUDE_president_savings_l3311_331112
