import Mathlib

namespace NUMINAMATH_CALUDE_triangle_angle_sine_inequality_l2441_244145

theorem triangle_angle_sine_inequality (α β γ : Real) 
  (h_triangle : α + β + γ = π) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) : 
  Real.sin (α/2 + β) + Real.sin (β/2 + γ) + Real.sin (γ/2 + α) > 
  Real.sin α + Real.sin β + Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_inequality_l2441_244145


namespace NUMINAMATH_CALUDE_catfish_weight_l2441_244183

theorem catfish_weight (trout_count : ℕ) (catfish_count : ℕ) (bluegill_count : ℕ)
  (trout_weight : ℝ) (bluegill_weight : ℝ) (total_weight : ℝ)
  (h1 : trout_count = 4)
  (h2 : catfish_count = 3)
  (h3 : bluegill_count = 5)
  (h4 : trout_weight = 2)
  (h5 : bluegill_weight = 2.5)
  (h6 : total_weight = 25)
  (h7 : total_weight = trout_count * trout_weight + catfish_count * (total_weight - trout_count * trout_weight - bluegill_count * bluegill_weight) / catfish_count + bluegill_count * bluegill_weight) :
  (total_weight - trout_count * trout_weight - bluegill_count * bluegill_weight) / catfish_count = 1.5 := by
sorry

end NUMINAMATH_CALUDE_catfish_weight_l2441_244183


namespace NUMINAMATH_CALUDE_probability_white_ball_l2441_244179

/-- The probability of drawing a white ball from a bag with red, white, and black balls -/
theorem probability_white_ball (red white black : ℕ) (h : red = 5 ∧ white = 2 ∧ black = 3) :
  (white : ℚ) / (red + white + black : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_ball_l2441_244179


namespace NUMINAMATH_CALUDE_simplify_expression_l2441_244137

theorem simplify_expression : (6^7 + 4^6) * (1^5 - (-1)^5)^10 = 290938368 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2441_244137


namespace NUMINAMATH_CALUDE_find_C_l2441_244126

theorem find_C (A B C : ℝ) 
  (h_diff1 : A ≠ B) (h_diff2 : A ≠ C) (h_diff3 : B ≠ C)
  (h1 : 3 * A - A = 10)
  (h2 : B + A = 12)
  (h3 : C - B = 6) : 
  C = 13 := by
sorry

end NUMINAMATH_CALUDE_find_C_l2441_244126


namespace NUMINAMATH_CALUDE_sum_possible_angles_l2441_244121

/-- An isosceles triangle with one angle of 80 degrees -/
structure IsoscelesTriangle80 where
  /-- The measure of one of the angles in degrees -/
  angle1 : ℝ
  /-- The measure of another angle in degrees -/
  angle2 : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True
  /-- One of the angles is 80 degrees -/
  has80Angle : angle1 = 80 ∨ angle2 = 80
  /-- The sum of all angles in a triangle is 180 degrees -/
  angleSum : angle1 + angle2 + (180 - angle1 - angle2) = 180

/-- The theorem to be proved -/
theorem sum_possible_angles (t : IsoscelesTriangle80) :
  ∃ (y1 y2 y3 : ℝ), (y1 = t.angle1 ∧ y1 ≠ 80) ∨ 
                    (y1 = t.angle2 ∧ y1 ≠ 80) ∨
                    (y1 = 180 - t.angle1 - t.angle2 ∧ y1 ≠ 80) ∧
                    (y2 = t.angle1 ∧ y2 ≠ 80) ∨ 
                    (y2 = t.angle2 ∧ y2 ≠ 80) ∨
                    (y2 = 180 - t.angle1 - t.angle2 ∧ y2 ≠ 80) ∧
                    (y3 = t.angle1 ∧ y3 ≠ 80) ∨ 
                    (y3 = t.angle2 ∧ y3 ≠ 80) ∨
                    (y3 = 180 - t.angle1 - t.angle2 ∧ y3 ≠ 80) ∧
                    y1 + y2 + y3 = 150 :=
  sorry

end NUMINAMATH_CALUDE_sum_possible_angles_l2441_244121


namespace NUMINAMATH_CALUDE_school_boys_count_l2441_244132

theorem school_boys_count (muslim_percent : ℝ) (hindu_percent : ℝ) (sikh_percent : ℝ) (other_count : ℕ) :
  muslim_percent = 44 →
  hindu_percent = 28 →
  sikh_percent = 10 →
  other_count = 54 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℝ) / (total : ℝ) * 100 = 100) ∧
    total = 300 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l2441_244132


namespace NUMINAMATH_CALUDE_rancher_cows_count_l2441_244157

theorem rancher_cows_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses →  -- The rancher raises 5 times as many cows as horses
  cows + horses = 168 →  -- The total number of animals is 168
  cows = 140 :=  -- Prove that the number of cows is 140
by sorry

end NUMINAMATH_CALUDE_rancher_cows_count_l2441_244157


namespace NUMINAMATH_CALUDE_ship_speed_in_still_water_l2441_244134

/-- Given a ship with downstream speed of 32 km/h and upstream speed of 28 km/h,
    prove that its speed in still water is 30 km/h. -/
theorem ship_speed_in_still_water 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : downstream_speed = 32)
  (h2 : upstream_speed = 28)
  (h3 : ∃ (ship_speed stream_speed : ℝ), 
    ship_speed > stream_speed ∧
    ship_speed + stream_speed = downstream_speed ∧
    ship_speed - stream_speed = upstream_speed) :
  ∃ (ship_speed : ℝ), ship_speed = 30 := by
sorry

end NUMINAMATH_CALUDE_ship_speed_in_still_water_l2441_244134


namespace NUMINAMATH_CALUDE_red_white_jelly_beans_in_fishbowl_l2441_244172

/-- The number of red jelly beans in one bag -/
def red_in_bag : ℕ := 24

/-- The number of white jelly beans in one bag -/
def white_in_bag : ℕ := 18

/-- The number of bags needed to fill the fishbowl -/
def bags_to_fill : ℕ := 3

/-- The total number of red and white jelly beans in the fishbowl -/
def total_red_white_in_bowl : ℕ := (red_in_bag + white_in_bag) * bags_to_fill

theorem red_white_jelly_beans_in_fishbowl :
  total_red_white_in_bowl = 126 :=
by sorry

end NUMINAMATH_CALUDE_red_white_jelly_beans_in_fishbowl_l2441_244172


namespace NUMINAMATH_CALUDE_direction_vector_valid_l2441_244186

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Converts a parametric equation to a line -/
def parametricToLine (x0 x1 y0 y1 : ℝ) : Line2D :=
  { a := y1 - y0, b := x0 - x1, c := x0 * y1 - x1 * y0 }

/-- Checks if a vector is parallel to a line -/
def isParallel (v : Vector2D) (l : Line2D) : Prop :=
  v.x * l.a + v.y * l.b = 0

/-- The given parametric equation of line l -/
def lineL : Line2D :=
  parametricToLine 1 3 2 1

/-- The proposed direction vector -/
def directionVector : Vector2D :=
  { x := -2, y := 1 }

theorem direction_vector_valid :
  isParallel directionVector lineL := by sorry

end NUMINAMATH_CALUDE_direction_vector_valid_l2441_244186


namespace NUMINAMATH_CALUDE_multiple_problem_l2441_244119

theorem multiple_problem (m : ℚ) : 38 + m * 43 = 124 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l2441_244119


namespace NUMINAMATH_CALUDE_apple_eating_time_l2441_244131

theorem apple_eating_time (apples_per_hour : ℕ) (total_apples : ℕ) (h1 : apples_per_hour = 5) (h2 : total_apples = 15) :
  total_apples / apples_per_hour = 3 := by
sorry

end NUMINAMATH_CALUDE_apple_eating_time_l2441_244131


namespace NUMINAMATH_CALUDE_geometry_algebra_properties_l2441_244103

-- Define supplementary angles
def supplementary (α β : Real) : Prop := α + β = 180

-- Define congruent angles
def congruent (α β : Real) : Prop := α = β

-- Define vertical angles
def vertical (α β : Real) : Prop := α = β

-- Define perpendicular lines
def perpendicular (l₁ l₂ : Line) : Prop := sorry

-- Define parallel lines
def parallel (l₁ l₂ : Line) : Prop := sorry

theorem geometry_algebra_properties :
  (∃ α β : Real, supplementary α β ∧ ¬congruent α β) ∧
  (∀ α β : Real, vertical α β → α = β) ∧
  ((-1 : Real)^(1/3) = -1) ∧
  (∀ l₁ l₂ l₃ : Line, perpendicular l₁ l₃ → perpendicular l₂ l₃ → parallel l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_geometry_algebra_properties_l2441_244103


namespace NUMINAMATH_CALUDE_pig_problem_l2441_244127

theorem pig_problem (x y : ℕ) : 
  (y - 100 = 100 * x) →  -- If each person contributes 100 coins, there's a surplus of 100
  (y = 90 * x) →         -- If each person contributes 90 coins, it's just enough
  (x = 10 ∧ y = 900) :=  -- Then the number of people is 10 and the price of the pig is 900
by sorry

end NUMINAMATH_CALUDE_pig_problem_l2441_244127


namespace NUMINAMATH_CALUDE_no_valid_flippy_numbers_l2441_244173

/-- A five-digit flippy number is a number of the form ababa or babab where a and b are distinct digits -/
def is_flippy_number (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧
  ((n = a * 10000 + b * 1000 + a * 100 + b * 10 + a) ∨
   (n = b * 10000 + a * 1000 + b * 100 + a * 10 + b))

/-- The sum of digits of a five-digit flippy number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

/-- There are no five-digit flippy numbers that are divisible by 11 and have a sum of digits divisible by 6 -/
theorem no_valid_flippy_numbers :
  ¬ ∃ n : ℕ, is_flippy_number n ∧ n % 11 = 0 ∧ (sum_of_digits n) % 6 = 0 := by
  sorry

#check no_valid_flippy_numbers

end NUMINAMATH_CALUDE_no_valid_flippy_numbers_l2441_244173


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2441_244101

/-- A cubic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (15/8) * x^3 + (5/4) * x^2 - (13/8) * x + 3

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q 0 = 3 ∧ q 1 = 5 ∧ q 2 = 13 ∧ q 3 = 41 := by
  sorry

#eval q 0
#eval q 1
#eval q 2
#eval q 3

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2441_244101


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2441_244185

theorem sum_of_a_and_b (a b c d : ℝ) 
  (h1 : a * c + b * d + b * c + a * d = 48)
  (h2 : c + d = 8) : 
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2441_244185


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2441_244122

theorem arithmetic_sequence_length (a₁ aₙ d : ℕ) (h : a₁ = 6) (h' : aₙ = 206) (h'' : d = 4) :
  (aₙ - a₁) / d + 1 = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2441_244122


namespace NUMINAMATH_CALUDE_product_of_square_roots_of_nine_l2441_244117

theorem product_of_square_roots_of_nine (a b : ℝ) : 
  a ^ 2 = 9 ∧ b ^ 2 = 9 ∧ a ≠ b → a * b = -9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_of_nine_l2441_244117


namespace NUMINAMATH_CALUDE_problem_solution_l2441_244124

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 10) 
  (h3 : x = 1) : 
  y = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2441_244124


namespace NUMINAMATH_CALUDE_harper_gift_cost_l2441_244136

/-- The total amount spent on teacher appreciation gifts --/
def total_gift_cost (son_teachers daughter_teachers gift_cost : ℕ) : ℕ :=
  (son_teachers + daughter_teachers) * gift_cost

/-- Theorem: Harper's total gift cost is $70 --/
theorem harper_gift_cost :
  total_gift_cost 3 4 10 = 70 := by
  sorry

end NUMINAMATH_CALUDE_harper_gift_cost_l2441_244136


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l2441_244115

theorem largest_n_with_unique_k : 
  (∃! (n : ℕ), n > 0 ∧ 
    (∃! (k : ℤ), (9:ℚ)/17 < (n:ℚ)/(n + k) ∧ (n:ℚ)/(n + k) < 8/15)) → 
  (∃! (n : ℕ), n = 72 ∧ n > 0 ∧ 
    (∃! (k : ℤ), (9:ℚ)/17 < (n:ℚ)/(n + k) ∧ (n:ℚ)/(n + k) < 8/15)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l2441_244115


namespace NUMINAMATH_CALUDE_lemonade_stand_total_profit_l2441_244138

/-- Calculates the profit for a single day of lemonade stand operation -/
def daily_profit (lemon_cost sugar_cost cup_cost extra_cost price_per_cup cups_sold : ℕ) : ℕ :=
  price_per_cup * cups_sold - (lemon_cost + sugar_cost + cup_cost + extra_cost)

/-- Represents the lemonade stand operation for three days -/
def lemonade_stand_profit : Prop :=
  let day1_profit := daily_profit 10 5 3 0 4 21
  let day2_profit := daily_profit 12 6 4 0 5 18
  let day3_profit := daily_profit 8 4 3 2 4 25
  day1_profit + day2_profit + day3_profit = 217

theorem lemonade_stand_total_profit : lemonade_stand_profit := by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_total_profit_l2441_244138


namespace NUMINAMATH_CALUDE_tangent_line_sqrt_x_at_one_l2441_244165

/-- The equation of the tangent line to y = √x at (1, 1) is x - 2y + 1 = 0 -/
theorem tangent_line_sqrt_x_at_one (x y : ℝ) : 
  (y = Real.sqrt x) → -- Given curve
  (x = 1 ∧ y = 1) → -- Point of tangency
  (x - 2*y + 1 = 0) -- Equation of tangent line
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_sqrt_x_at_one_l2441_244165


namespace NUMINAMATH_CALUDE_total_workers_is_214_l2441_244193

/-- Represents a workshop with its salary information -/
structure Workshop where
  avgSalary : ℕ
  techCount : ℕ
  techAvgSalary : ℕ
  otherSalary : ℕ

/-- Calculates the total number of workers in a workshop -/
def totalWorkers (w : Workshop) : ℕ :=
  let otherWorkers := (w.avgSalary * (w.techCount + 1) - w.techAvgSalary * w.techCount) / (w.avgSalary - w.otherSalary)
  w.techCount + otherWorkers

/-- The given workshops -/
def workshopA : Workshop := {
  avgSalary := 8000,
  techCount := 7,
  techAvgSalary := 20000,
  otherSalary := 6000
}

def workshopB : Workshop := {
  avgSalary := 9000,
  techCount := 10,
  techAvgSalary := 25000,
  otherSalary := 5000
}

def workshopC : Workshop := {
  avgSalary := 10000,
  techCount := 15,
  techAvgSalary := 30000,
  otherSalary := 7000
}

/-- The main theorem to prove -/
theorem total_workers_is_214 :
  totalWorkers workshopA + totalWorkers workshopB + totalWorkers workshopC = 214 := by
  sorry


end NUMINAMATH_CALUDE_total_workers_is_214_l2441_244193


namespace NUMINAMATH_CALUDE_not_all_angles_greater_than_60_l2441_244109

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : a + b + c = 180

-- Theorem statement
theorem not_all_angles_greater_than_60 (t : Triangle) : 
  ¬(t.a > 60 ∧ t.b > 60 ∧ t.c > 60) :=
by sorry

end NUMINAMATH_CALUDE_not_all_angles_greater_than_60_l2441_244109


namespace NUMINAMATH_CALUDE_jerome_money_ratio_l2441_244184

def jerome_money_problem (initial_money : ℕ) : Prop :=
  let meg_money : ℕ := 8
  let bianca_money : ℕ := 3 * meg_money
  let remaining_money : ℕ := 54
  initial_money = remaining_money + meg_money + bianca_money ∧
  (initial_money : ℚ) / remaining_money = 43 / 27

theorem jerome_money_ratio :
  ∃ (initial_money : ℕ), jerome_money_problem initial_money :=
sorry

end NUMINAMATH_CALUDE_jerome_money_ratio_l2441_244184


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2441_244167

theorem completing_square_quadratic (x : ℝ) : 
  (∃ c, (x^2 + 4*x + 2 = 0) ↔ ((x + 2)^2 = c)) → 
  (∃ c, ((x + 2)^2 = c) ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2441_244167


namespace NUMINAMATH_CALUDE_first_battery_was_voltaic_pile_l2441_244148

/-- Represents a battery -/
structure Battery where
  year : Nat
  creator : String
  components : List String

/-- The first recognized battery in the world -/
def first_battery : Battery :=
  { year := 1800,
    creator := "Alessandro Volta",
    components := ["different metals", "electrolyte"] }

/-- Theorem stating that the first recognized battery was the Voltaic pile -/
theorem first_battery_was_voltaic_pile :
  first_battery.year = 1800 ∧
  first_battery.creator = "Alessandro Volta" ∧
  first_battery.components = ["different metals", "electrolyte"] :=
by sorry

#check first_battery_was_voltaic_pile

end NUMINAMATH_CALUDE_first_battery_was_voltaic_pile_l2441_244148


namespace NUMINAMATH_CALUDE_distance_to_focus_is_4_l2441_244149

/-- The distance from a point on the parabola y^2 = 4x with x-coordinate 3 to its focus -/
def distance_to_focus (y : ℝ) : ℝ :=
  4

/-- A point P lies on the parabola y^2 = 4x -/
def on_parabola (x y : ℝ) : Prop :=
  y^2 = 4*x

theorem distance_to_focus_is_4 :
  ∀ y : ℝ, on_parabola 3 y → distance_to_focus y = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_is_4_l2441_244149


namespace NUMINAMATH_CALUDE_boat_license_count_l2441_244182

def boat_license_options : ℕ :=
  let letter_options := 3  -- A, M, or S
  let digit_options := 10  -- 0 to 9
  let digit_positions := 5
  letter_options * digit_options ^ digit_positions

theorem boat_license_count : boat_license_options = 300000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_count_l2441_244182


namespace NUMINAMATH_CALUDE_original_denominator_proof_l2441_244111

theorem original_denominator_proof (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 → (3 + 8 : ℚ) / (d + 8) = 1 / 3 → d = 25 := by
  sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l2441_244111


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2441_244197

theorem quadratic_root_relation (b c : ℝ) : 
  (∃ p q : ℝ, 2 * p^2 - 4 * p - 6 = 0 ∧ 2 * q^2 - 4 * q - 6 = 0 ∧
   ∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = p - 3 ∨ x = q - 3)) →
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2441_244197


namespace NUMINAMATH_CALUDE_xiaogang_shooting_probability_l2441_244155

theorem xiaogang_shooting_probability (total_shots : ℕ) (successful_shots : ℕ) 
  (h1 : total_shots = 50) 
  (h2 : successful_shots = 38) : 
  (successful_shots : ℚ) / (total_shots : ℚ) = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_xiaogang_shooting_probability_l2441_244155


namespace NUMINAMATH_CALUDE_h2so4_moles_l2441_244177

def reaction (nNaCl : ℝ) (nH2SO4 : ℝ) (nHCl : ℝ) (nNaHSO4 : ℝ) : Prop :=
  nNaCl = 2 ∧ nNaHSO4 = 2 ∧ nHCl = nNaHSO4 ∧ nH2SO4 = nNaCl

theorem h2so4_moles : ∃ (nH2SO4 : ℝ), reaction 2 nH2SO4 2 2 ∧ nH2SO4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_h2so4_moles_l2441_244177


namespace NUMINAMATH_CALUDE_problem_statement_l2441_244196

theorem problem_statement (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x + 2*y + 3*z = 1) : 
  (∃ (m : ℝ), m = 6 + 2*Real.sqrt 2 + 2*Real.sqrt 3 + 2*Real.sqrt 6 ∧ 
   (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + 2*b + 3*c = 1 → 
    1/a + 1/b + 1/c ≥ m) ∧
   1/x + 1/y + 1/z = m) ∧ 
  x^2 + y^2 + z^2 ≥ 1/14 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2441_244196


namespace NUMINAMATH_CALUDE_tangent_line_parallel_point_l2441_244170

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x + 1

theorem tangent_line_parallel_point (P₀ : ℝ × ℝ) : 
  P₀.1 = 1 ∧ P₀.2 = f P₀.1 ∧ f' P₀.1 = 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_point_l2441_244170


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2441_244188

theorem fractional_equation_solution : 
  ∃ x : ℝ, (3 - x) / (x - 4) + 1 / (4 - x) = 1 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2441_244188


namespace NUMINAMATH_CALUDE_percentage_difference_l2441_244198

-- Define the variables
variable (x y z : ℝ)

-- State the theorem
theorem percentage_difference (h1 : z = 400) (h2 : y = 1.2 * z) (h3 : x + y + z = 1480) :
  (x - y) / y = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2441_244198


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2441_244129

def set_A : Set ℝ := {x | (1 - x) * (1 + x) ≥ 0}

def set_B : Set ℝ := {y | ∃ x < 0, y = 2^x}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2441_244129


namespace NUMINAMATH_CALUDE_sum_neq_two_implies_both_neq_one_l2441_244108

theorem sum_neq_two_implies_both_neq_one (x y : ℝ) : x + y ≠ 2 → x ≠ 1 ∧ y ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_neq_two_implies_both_neq_one_l2441_244108


namespace NUMINAMATH_CALUDE_number_of_candidates_l2441_244159

/-- The number of ways to select a president and vice president -/
def selection_ways : ℕ := 30

/-- Theorem: Given 30 ways to select a president and vice president, 
    where the same person cannot be both, there are 6 candidates. -/
theorem number_of_candidates : 
  ∃ (n : ℕ), n > 0 ∧ n * (n - 1) = selection_ways := by
  sorry

end NUMINAMATH_CALUDE_number_of_candidates_l2441_244159


namespace NUMINAMATH_CALUDE_gcd_1213_1985_l2441_244141

theorem gcd_1213_1985 : 
  (¬ (1213 % 2 = 0)) → 
  (¬ (1213 % 3 = 0)) → 
  (¬ (1213 % 5 = 0)) → 
  (¬ (1985 % 2 = 0)) → 
  (¬ (1985 % 3 = 0)) → 
  (¬ (1985 % 5 = 0)) → 
  Nat.gcd 1213 1985 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1213_1985_l2441_244141


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2441_244181

theorem complex_magnitude_problem (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2441_244181


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l2441_244125

theorem binomial_coefficient_problem (m : ℤ) : 
  (Nat.choose 4 2 : ℤ) * m^2 = (Nat.choose 4 3 : ℤ) * m + 16 → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l2441_244125


namespace NUMINAMATH_CALUDE_arith_seq_mono_increasing_iff_a2_gt_a1_l2441_244147

/-- An arithmetic sequence -/
def ArithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def MonoIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- Theorem: For an arithmetic sequence, a_2 > a_1 is equivalent to the sequence being monotonically increasing -/
theorem arith_seq_mono_increasing_iff_a2_gt_a1 (a : ℕ → ℝ) (h : ArithmeticSeq a) :
  a 2 > a 1 ↔ MonoIncreasing a := by sorry

end NUMINAMATH_CALUDE_arith_seq_mono_increasing_iff_a2_gt_a1_l2441_244147


namespace NUMINAMATH_CALUDE_johns_break_time_l2441_244135

/-- Given the dancing times of John and James, prove that John's break was 1 hour long. -/
theorem johns_break_time (john_first_dance : ℝ) (john_second_dance : ℝ) 
  (james_dance_multiplier : ℝ) (total_dance_time : ℝ) :
  john_first_dance = 3 →
  john_second_dance = 5 →
  james_dance_multiplier = 1/3 →
  total_dance_time = 20 →
  ∃ (break_time : ℝ),
    total_dance_time = john_first_dance + john_second_dance + 
      ((john_first_dance + john_second_dance + break_time) + 
       james_dance_multiplier * (john_first_dance + john_second_dance + break_time)) ∧
    break_time = 1 := by
  sorry


end NUMINAMATH_CALUDE_johns_break_time_l2441_244135


namespace NUMINAMATH_CALUDE_polynomial_expansion_p_value_l2441_244194

/-- The value of p in the expansion of (x+y)^8 -/
theorem polynomial_expansion_p_value :
  ∀ (p q : ℝ),
  p > 0 →
  q > 0 →
  p + q = 1 →
  8 * p^7 * q = 28 * p^6 * q^2 →
  p = 7/9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_p_value_l2441_244194


namespace NUMINAMATH_CALUDE_billion_scientific_notation_l2441_244142

theorem billion_scientific_notation : 
  (1.1 * 10^9 : ℝ) = 1100000000 := by sorry

end NUMINAMATH_CALUDE_billion_scientific_notation_l2441_244142


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2441_244112

def game_sequence (n : ℕ) : ℕ := 16 * n + 700

theorem smallest_winning_number :
  ∃ (N : ℕ),
    N ≤ 999 ∧
    950 ≤ game_sequence N ∧
    game_sequence N ≤ 999 ∧
    ∀ (m : ℕ), m < N →
      (m ≤ 999 →
       (game_sequence m < 950 ∨ game_sequence m > 999)) ∧
    N = 16 :=
  sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2441_244112


namespace NUMINAMATH_CALUDE_maria_students_l2441_244120

/-- The number of students in Maria's high school -/
def M : ℕ := sorry

/-- The number of students in Jackson's high school -/
def J : ℕ := sorry

/-- Maria's high school has 4 times as many students as Jackson's high school -/
axiom maria_jackson_ratio : M = 4 * J

/-- The total number of students in both high schools is 3600 -/
axiom total_students : M + J = 3600

/-- Theorem: Maria's high school has 2880 students -/
theorem maria_students : M = 2880 := by sorry

end NUMINAMATH_CALUDE_maria_students_l2441_244120


namespace NUMINAMATH_CALUDE_xy_value_l2441_244144

theorem xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h1 : x^2 + y^2 = 3) (h2 : x^4 + y^4 = 15/8) : x * y = Real.sqrt 57 / 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2441_244144


namespace NUMINAMATH_CALUDE_trip_distance_l2441_244174

theorem trip_distance (total : ℝ) 
  (h1 : total > 0)
  (h2 : total / 4 + 30 + total / 10 + (total - (total / 4 + 30 + total / 10)) = total) : 
  total = 60 := by
sorry

end NUMINAMATH_CALUDE_trip_distance_l2441_244174


namespace NUMINAMATH_CALUDE_no_fraternity_member_is_club_member_l2441_244168

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (ClubMember : U → Prop)
variable (FraternityMember : U → Prop)
variable (Honest : U → Prop)

-- State the theorem
theorem no_fraternity_member_is_club_member
  (h1 : ∀ x, ClubMember x → Honest x)
  (h2 : ∃ x, Student x ∧ ¬Honest x)
  (h3 : ∀ x, Student x → FraternityMember x → ¬ClubMember x) :
  ∀ x, FraternityMember x → ¬ClubMember x :=
by
  sorry

end NUMINAMATH_CALUDE_no_fraternity_member_is_club_member_l2441_244168


namespace NUMINAMATH_CALUDE_power_function_through_sqrt2_l2441_244189

/-- A power function that passes through the point (2, √2) is equal to the square root function. -/
theorem power_function_through_sqrt2 (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x^α) →   -- f is a power function
  f 2 = Real.sqrt 2 →      -- f passes through (2, √2)
  ∀ x > 0, f x = Real.sqrt x := by
sorry

end NUMINAMATH_CALUDE_power_function_through_sqrt2_l2441_244189


namespace NUMINAMATH_CALUDE_root_condition_implies_m_range_l2441_244152

theorem root_condition_implies_m_range :
  ∀ m : ℝ,
  (∀ x : ℝ, x^2 - (3*m + 2)*x + 2*(m + 6) = 0 → x > 3) →
  m ≥ 2 ∧ m < 15/7 := by
sorry

end NUMINAMATH_CALUDE_root_condition_implies_m_range_l2441_244152


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l2441_244139

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_of_angle_between_vectors
  (c d : V)
  (h1 : ‖c‖ = 5)
  (h2 : ‖d‖ = 7)
  (h3 : ‖c + d‖ = 10) :
  inner c d / (‖c‖ * ‖d‖) = 13 / 35 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l2441_244139


namespace NUMINAMATH_CALUDE_probability_circle_or_square_l2441_244100

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of circles
def num_circles : ℕ := 3

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of triangles
def num_triangles : ℕ := 3

-- Theorem statement
theorem probability_circle_or_square :
  (num_circles + num_squares : ℚ) / total_figures = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_circle_or_square_l2441_244100


namespace NUMINAMATH_CALUDE_square_sum_identity_l2441_244153

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l2441_244153


namespace NUMINAMATH_CALUDE_chord_length_l2441_244160

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l2441_244160


namespace NUMINAMATH_CALUDE_m_value_min_value_l2441_244123

-- Define the solution set A
def A (m : ℤ) : Set ℝ := {x : ℝ | |x + 1| + |x - m| < 5}

-- Theorem 1
theorem m_value (m : ℤ) (h : 3 ∈ A m) : m = 3 := by sorry

-- Theorem 2
theorem min_value (a b c : ℝ) (h : a + 2*b + 2*c = 3) : 
  ∃ (min : ℝ), min = 1 ∧ a^2 + b^2 + c^2 ≥ min := by sorry

end NUMINAMATH_CALUDE_m_value_min_value_l2441_244123


namespace NUMINAMATH_CALUDE_different_subjects_count_l2441_244171

/-- The number of ways to choose 2 books from different subjects -/
def choose_different_subjects (chinese_books math_books english_books : ℕ) : ℕ :=
  chinese_books * math_books + chinese_books * english_books + math_books * english_books

/-- Theorem stating that there are 143 ways to choose 2 books from different subjects -/
theorem different_subjects_count :
  choose_different_subjects 9 7 5 = 143 := by
  sorry

end NUMINAMATH_CALUDE_different_subjects_count_l2441_244171


namespace NUMINAMATH_CALUDE_factorization_problems_l2441_244187

theorem factorization_problems (x y : ℝ) : 
  (7 * x^2 - 63 = 7 * (x + 3) * (x - 3)) ∧ 
  (x^3 + 6 * x^2 * y + 9 * x * y^2 = x * (x + 3 * y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2441_244187


namespace NUMINAMATH_CALUDE_average_income_proof_l2441_244192

def cab_driver_income : List ℝ := [200, 150, 750, 400, 500]

theorem average_income_proof :
  (List.sum cab_driver_income) / (List.length cab_driver_income) = 400 := by
  sorry

end NUMINAMATH_CALUDE_average_income_proof_l2441_244192


namespace NUMINAMATH_CALUDE_unique_solution_for_f_equals_two_l2441_244162

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x - 4
  else if x ≤ 2 then x^2 - 1
  else x/3 + 2

theorem unique_solution_for_f_equals_two :
  ∃! x : ℝ, f x = 2 ∧ x = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_f_equals_two_l2441_244162


namespace NUMINAMATH_CALUDE_range_of_a_l2441_244105

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| < 4}
def B : Set ℝ := {x | (x - 2) * (3 - x) > 0}

-- Define the proposition p and q
def p (a : ℝ) (x : ℝ) : Prop := x ∈ A a
def q (x : ℝ) : Prop := x ∈ B

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, ¬(p a x) → ¬(q x)) →
  a ∈ Set.Icc (-1 : ℝ) 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2441_244105


namespace NUMINAMATH_CALUDE_inequality_holds_l2441_244106

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > 0) :
  (1 / Real.log a) > (1 / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l2441_244106


namespace NUMINAMATH_CALUDE_chicken_problem_model_l2441_244180

/-- Represents the system of equations for the chicken buying problem -/
def chicken_equations (x y : ℕ) : Prop :=
  (8 * x - y = 3) ∧ (y - 7 * x = 4)

/-- Proves that the system of equations correctly models the given conditions -/
theorem chicken_problem_model (x y : ℕ) :
  (x > 0 ∧ y > 0) →
  (chicken_equations x y ↔
    (8 * x = y + 3 ∧ 7 * x + 4 = y)) :=
by sorry

end NUMINAMATH_CALUDE_chicken_problem_model_l2441_244180


namespace NUMINAMATH_CALUDE_factorial_ratio_l2441_244113

theorem factorial_ratio : Nat.factorial 15 / (Nat.factorial 6 * Nat.factorial 9) = 5005 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2441_244113


namespace NUMINAMATH_CALUDE_video_game_lives_calculation_l2441_244191

/-- Calculate the total number of lives for remaining players in a video game --/
theorem video_game_lives_calculation (initial_players : ℕ) (initial_lives : ℕ) 
  (quit_players : ℕ) (powerup_players : ℕ) (penalty_players : ℕ) 
  (powerup_lives : ℕ) (penalty_lives : ℕ) : 
  initial_players = 15 →
  initial_lives = 10 →
  quit_players = 5 →
  powerup_players = 4 →
  penalty_players = 6 →
  powerup_lives = 3 →
  penalty_lives = 2 →
  (initial_players - quit_players) * initial_lives + 
    powerup_players * powerup_lives - penalty_players * penalty_lives = 100 := by
  sorry


end NUMINAMATH_CALUDE_video_game_lives_calculation_l2441_244191


namespace NUMINAMATH_CALUDE_rectangle_to_square_cut_l2441_244178

theorem rectangle_to_square_cut (length width : ℝ) (h1 : length = 16) (h2 : width = 9) :
  ∃ (side : ℝ), side = 12 ∧ 
  2 * (side * side) = length * width ∧
  side ≤ length ∧ side ≤ width + (length - side) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_cut_l2441_244178


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l2441_244150

theorem binomial_expansion_example : 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 104060401 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l2441_244150


namespace NUMINAMATH_CALUDE_train_speed_is_25_l2441_244130

-- Define the train and its properties
structure Train :=
  (speed : ℝ)
  (length : ℝ)

-- Define the tunnels
def tunnel1_length : ℝ := 85
def tunnel2_length : ℝ := 160
def tunnel1_time : ℝ := 5
def tunnel2_time : ℝ := 8

-- Theorem statement
theorem train_speed_is_25 (t : Train) :
  (tunnel1_length + t.length) / tunnel1_time = t.speed →
  (tunnel2_length + t.length) / tunnel2_time = t.speed →
  t.speed = 25 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_is_25_l2441_244130


namespace NUMINAMATH_CALUDE_james_and_louise_ages_james_and_louise_ages_proof_l2441_244128

theorem james_and_louise_ages : ℕ → ℕ → Prop :=
  fun j l =>
    (j = l + 9) →                  -- James is nine years older than Louise
    (j + 7 = 3 * (l - 3)) →        -- Seven years from now, James will be three times as old as Louise was three years before now
    (j + l = 35)                   -- The sum of their current ages is 35

-- The proof of this theorem
theorem james_and_louise_ages_proof : ∃ j l : ℕ, james_and_louise_ages j l := by
  sorry

end NUMINAMATH_CALUDE_james_and_louise_ages_james_and_louise_ages_proof_l2441_244128


namespace NUMINAMATH_CALUDE_wall_bricks_count_l2441_244176

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 288

/-- Time taken by the first bricklayer to build the wall alone -/
def time1 : ℕ := 8

/-- Time taken by the second bricklayer to build the wall alone -/
def time2 : ℕ := 12

/-- Efficiency loss when working together (in bricks per hour) -/
def efficiency_loss : ℕ := 12

/-- Time taken by both bricklayers working together -/
def time_together : ℕ := 6

theorem wall_bricks_count :
  (time_together : ℚ) * ((total_bricks / time1 : ℚ) + (total_bricks / time2 : ℚ) - efficiency_loss) = total_bricks := by
  sorry

#check wall_bricks_count

end NUMINAMATH_CALUDE_wall_bricks_count_l2441_244176


namespace NUMINAMATH_CALUDE_difference_of_squares_l2441_244146

theorem difference_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2441_244146


namespace NUMINAMATH_CALUDE_sector_area_l2441_244107

/-- The area of a circular sector with a central angle of 150° and a radius of √3 is 5π/4 -/
theorem sector_area (α : Real) (r : Real) : 
  α = 150 * π / 180 →  -- Convert 150° to radians
  r = Real.sqrt 3 →
  (1 / 2) * α * r^2 = (5 * π) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2441_244107


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2441_244161

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2441_244161


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2441_244158

theorem inequality_solution_set : 
  {x : ℝ | 8*x^3 + 9*x^2 + 7*x < 6} = 
  {x : ℝ | (-6 < x ∧ x < -1/8) ∨ (-1/8 < x ∧ x < 1)} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2441_244158


namespace NUMINAMATH_CALUDE_min_sum_of_cubes_when_sum_is_eight_l2441_244116

theorem min_sum_of_cubes_when_sum_is_eight :
  ∀ x y : ℝ, x + y = 8 →
  x^3 + y^3 ≥ 4^3 + 4^3 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_cubes_when_sum_is_eight_l2441_244116


namespace NUMINAMATH_CALUDE_exponent_equality_l2441_244110

theorem exponent_equality (y x : ℕ) (h1 : 16 ^ y = 4 ^ x) (h2 : y = 7) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2441_244110


namespace NUMINAMATH_CALUDE_hidden_sum_is_55_l2441_244195

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The total number of dice -/
def num_dice : ℕ := 4

/-- The list of visible numbers on the dice -/
def visible_numbers : List ℕ := [1, 2, 3, 3, 4, 5, 5, 6]

/-- The sum of all numbers on all dice -/
def total_sum : ℕ := die_sum * num_dice

/-- The sum of visible numbers -/
def visible_sum : ℕ := visible_numbers.sum

theorem hidden_sum_is_55 : total_sum - visible_sum = 55 := by
  sorry

end NUMINAMATH_CALUDE_hidden_sum_is_55_l2441_244195


namespace NUMINAMATH_CALUDE_circular_arrangement_size_l2441_244190

/-- A circular arrangement of people with the property that the 7th person is directly opposite the 18th person -/
structure CircularArrangement where
  n : ℕ  -- Total number of people
  seventh_opposite_eighteenth : n ≥ 18 ∧ (18 - 7) * 2 + 2 = n

/-- The theorem stating that in a circular arrangement where the 7th person is directly opposite the 18th person, the total number of people is 24 -/
theorem circular_arrangement_size (c : CircularArrangement) : c.n = 24 := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_size_l2441_244190


namespace NUMINAMATH_CALUDE_line_transformation_l2441_244156

/-- Given a line with equation y = -3/4x + 5, prove that a new line M with one-third the slope
and three times the y-intercept has the equation y = -1/4x + 15. -/
theorem line_transformation (x y : ℝ) :
  (y = -3/4 * x + 5) →
  ∃ (M : ℝ → ℝ),
    (∀ x, M x = -1/4 * x + 15) ∧
    (∀ x, M x = 1/3 * (-3/4) * x + 3 * 5) :=
by sorry

end NUMINAMATH_CALUDE_line_transformation_l2441_244156


namespace NUMINAMATH_CALUDE_x_over_y_value_l2441_244164

theorem x_over_y_value (x y : ℝ) (h1 : x * y = 1) (h2 : x > 0) (h3 : y > 0) (h4 : y = 0.16666666666666666) :
  x / y = 36 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_value_l2441_244164


namespace NUMINAMATH_CALUDE_power_inequality_l2441_244133

theorem power_inequality (n : ℕ) (h : n > 1) : n ^ n > (n + 1) ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2441_244133


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l2441_244118

theorem smallest_lcm_with_gcd_five (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 →
  1000 ≤ l ∧ l < 10000 →
  Nat.gcd k l = 5 →
  ∀ m n : ℕ, 1000 ≤ m ∧ m < 10000 → 1000 ≤ n ∧ n < 10000 → Nat.gcd m n = 5 →
  Nat.lcm k l ≤ Nat.lcm m n →
  Nat.lcm k l = 203010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l2441_244118


namespace NUMINAMATH_CALUDE_distance_after_five_hours_l2441_244199

/-- The distance between two people walking in opposite directions -/
def distance_between (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 + speed2) * time

/-- Theorem: The distance between two people walking in opposite directions for 5 hours
    with speeds 5 km/hr and 10 km/hr is 75 km -/
theorem distance_after_five_hours :
  distance_between 5 10 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_five_hours_l2441_244199


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l2441_244102

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 →
  Nat.gcd a b = 55 →
  a = 210 →
  b = 605 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l2441_244102


namespace NUMINAMATH_CALUDE_least_common_denominator_l2441_244166

theorem least_common_denominator : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9)))))) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l2441_244166


namespace NUMINAMATH_CALUDE_min_distance_line_curve_l2441_244175

/-- The minimum distance between a point on the line y = 1 - x and a point on the curve y = -e^x is √2 -/
theorem min_distance_line_curve : 
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.2 = 1 - P.1) → 
    (Q.2 = -Real.exp Q.1) → 
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_line_curve_l2441_244175


namespace NUMINAMATH_CALUDE_fourth_to_first_class_ratio_l2441_244154

def num_classes : ℕ := 6
def students_first_class : ℕ := 20
def students_second_third_class : ℕ := 25
def students_fifth_sixth_class : ℕ := 28
def total_students : ℕ := 136

theorem fourth_to_first_class_ratio :
  ∃ (students_fourth_class : ℕ),
    students_first_class +
    2 * students_second_third_class +
    students_fourth_class +
    2 * students_fifth_sixth_class = total_students ∧
    students_fourth_class * 2 = students_first_class :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_to_first_class_ratio_l2441_244154


namespace NUMINAMATH_CALUDE_intersection_unique_l2441_244163

/-- The system of linear equations representing two lines -/
def line_system (x y : ℚ) : Prop :=
  8 * x - 5 * y = 40 ∧ 6 * x + 2 * y = 14

/-- The intersection point of the two lines -/
def intersection_point : ℚ × ℚ := (75/23, -64/23)

/-- Theorem stating that the intersection point is the unique solution to the system of equations -/
theorem intersection_unique :
  line_system intersection_point.1 intersection_point.2 ∧
  ∀ x y, line_system x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_unique_l2441_244163


namespace NUMINAMATH_CALUDE_parallel_lines_minimum_value_l2441_244143

theorem parallel_lines_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 2 * a + 3 * b ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_minimum_value_l2441_244143


namespace NUMINAMATH_CALUDE_second_expression_value_l2441_244151

theorem second_expression_value (a x : ℝ) (h1 : ((2 * a + 16) + x) / 2 = 69) (h2 : a = 26) : x = 70 := by
  sorry

end NUMINAMATH_CALUDE_second_expression_value_l2441_244151


namespace NUMINAMATH_CALUDE_selection_methods_count_l2441_244169

def n : ℕ := 10  -- Total number of college student village officials
def k : ℕ := 3   -- Number of individuals to be selected

def total_without_b : ℕ := Nat.choose (n - 1) k
def without_a_and_c : ℕ := Nat.choose (n - 3) k

theorem selection_methods_count : 
  total_without_b - without_a_and_c = 49 := by sorry

end NUMINAMATH_CALUDE_selection_methods_count_l2441_244169


namespace NUMINAMATH_CALUDE_strategy2_is_cheaper_l2441_244114

def original_price : ℝ := 12000

def strategy1_cost (price : ℝ) : ℝ :=
  price * (1 - 0.30) * (1 - 0.15) * (1 - 0.05)

def strategy2_cost (price : ℝ) : ℝ :=
  price * (1 - 0.45) * (1 - 0.10) * (1 - 0.10) + 150

theorem strategy2_is_cheaper :
  strategy2_cost original_price < strategy1_cost original_price :=
by sorry

end NUMINAMATH_CALUDE_strategy2_is_cheaper_l2441_244114


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l2441_244140

/-- A line in 2D space with a given slope and x-intercept. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ := l.slope * (-l.x_intercept) + 0

/-- Theorem stating that a line with slope -3 and x-intercept (3, 0) has y-intercept (0, 9). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := 3 }
  y_intercept l = 9 := by
  sorry


end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l2441_244140


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l2441_244104

-- Define the complex polynomial z^5 - z^3 + z
def f (z : ℂ) : ℂ := z^5 - z^3 + z

-- Define the property of being an nth root of unity
def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

-- State the theorem
theorem smallest_n_for_roots_of_unity : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), f z = 0 → is_nth_root_of_unity z n) ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    ∃ (w : ℂ), f w = 0 ∧ ¬(is_nth_root_of_unity w m)) ∧
  n = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l2441_244104
