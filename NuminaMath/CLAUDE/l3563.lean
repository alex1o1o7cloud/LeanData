import Mathlib

namespace NUMINAMATH_CALUDE_erdos_szekeres_theorem_l3563_356319

theorem erdos_szekeres_theorem (m n : ℕ) (seq : Fin (m * n + 1) → ℝ) :
  (∃ (subseq : Fin (m + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → subseq i < subseq j) ∧
    (∀ i j, i < j → seq (subseq i) ≤ seq (subseq j))) ∨
  (∃ (subseq : Fin (n + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → subseq i < subseq j) ∧
    (∀ i j, i < j → seq (subseq i) ≥ seq (subseq j))) :=
by sorry

end NUMINAMATH_CALUDE_erdos_szekeres_theorem_l3563_356319


namespace NUMINAMATH_CALUDE_total_paintable_area_is_876_l3563_356376

/-- The number of bedrooms in Isabella's house -/
def num_bedrooms : ℕ := 3

/-- The length of each bedroom in feet -/
def bedroom_length : ℕ := 12

/-- The width of each bedroom in feet -/
def bedroom_width : ℕ := 10

/-- The height of each bedroom in feet -/
def bedroom_height : ℕ := 8

/-- The area occupied by doorways and windows in each bedroom in square feet -/
def unpaintable_area : ℕ := 60

/-- The total area of walls to be painted in all bedrooms -/
def total_paintable_area : ℕ :=
  num_bedrooms * (
    2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height) - unpaintable_area
  )

/-- Theorem stating that the total area to be painted is 876 square feet -/
theorem total_paintable_area_is_876 : total_paintable_area = 876 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_is_876_l3563_356376


namespace NUMINAMATH_CALUDE_octal_127_equals_binary_1010111_l3563_356333

def octal_to_decimal (x : ℕ) : ℕ := 
  (x % 10) + 8 * ((x / 10) % 10) + 64 * (x / 100)

def decimal_to_binary (x : ℕ) : List ℕ :=
  if x = 0 then [0]
  else 
    let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux x []

theorem octal_127_equals_binary_1010111 : 
  decimal_to_binary (octal_to_decimal 127) = [1, 0, 1, 0, 1, 1, 1] := by
  sorry

#eval octal_to_decimal 127
#eval decimal_to_binary (octal_to_decimal 127)

end NUMINAMATH_CALUDE_octal_127_equals_binary_1010111_l3563_356333


namespace NUMINAMATH_CALUDE_james_recovery_time_l3563_356391

def initial_healing_time : ℝ := 4

def skin_graft_healing_time (t : ℝ) : ℝ := t * 1.5

def total_recovery_time (t : ℝ) : ℝ := t + skin_graft_healing_time t

theorem james_recovery_time :
  total_recovery_time initial_healing_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_recovery_time_l3563_356391


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3563_356338

-- Define the equations
def equation1 (x : ℝ) : Prop := 6 * x - 7 = 4 * x - 5
def equation2 (x : ℝ) : Prop := 4 / 3 - 8 * x = 3 - 11 / 2 * x

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -2/3 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3563_356338


namespace NUMINAMATH_CALUDE_range_of_x_when_f_leq_1_range_of_m_when_f_minus_g_geq_m_plus_1_l3563_356354

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := -|x + 1| + 4

-- Theorem 1: Range of x when f(x) ≤ 1
theorem range_of_x_when_f_leq_1 :
  {x : ℝ | f x ≤ 1} = Set.Icc 0 6 := by sorry

-- Theorem 2: Range of m when f(x) - g(x) ≥ m+1 for all x ∈ ℝ
theorem range_of_m_when_f_minus_g_geq_m_plus_1 :
  {m : ℝ | ∀ x, f x - g x ≥ m + 1} = Set.Iic (-3) := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_f_leq_1_range_of_m_when_f_minus_g_geq_m_plus_1_l3563_356354


namespace NUMINAMATH_CALUDE_max_value_expression_l3563_356326

theorem max_value_expression (x y : ℝ) : 
  (Real.sqrt (3 - Real.sqrt 2) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) * 
  (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≤ 9.5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3563_356326


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l3563_356372

theorem at_least_one_non_negative (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) 
  (h₁ : a₁ ≠ 0) (h₂ : a₂ ≠ 0) (h₃ : a₃ ≠ 0) (h₄ : a₄ ≠ 0) 
  (h₅ : a₅ ≠ 0) (h₆ : a₆ ≠ 0) (h₇ : a₇ ≠ 0) (h₈ : a₈ ≠ 0) : 
  max (a₁ * a₃ + a₂ * a₄) (max (a₁ * a₅ + a₂ * a₆) (max (a₁ * a₇ + a₂ * a₈) 
    (max (a₃ * a₅ + a₄ * a₆) (max (a₃ * a₇ + a₄ * a₈) (a₅ * a₇ + a₆ * a₈))))) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l3563_356372


namespace NUMINAMATH_CALUDE_increasing_function_m_range_l3563_356369

def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

theorem increasing_function_m_range :
  ∀ m : ℝ, (∀ x > 2, (∀ h > 0, f m (x + h) > f m x)) ↔ m < 5/2 :=
sorry

end NUMINAMATH_CALUDE_increasing_function_m_range_l3563_356369


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l3563_356392

/-- A number is 7-heavy if its remainder when divided by 7 is greater than 4 -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- A number is three-digit if it's between 100 and 999 inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_heavy : 
  is_three_digit 103 ∧ 
  is_7_heavy 103 ∧ 
  ∀ n : ℕ, is_three_digit n → is_7_heavy n → 103 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l3563_356392


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3563_356381

/-- A geometric sequence with a positive common ratio -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 6 = 8 * a 4 →
  a 2 = 2 →
  a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3563_356381


namespace NUMINAMATH_CALUDE_divisible_by_sum_of_digits_l3563_356341

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem divisible_by_sum_of_digits :
  ∀ n : ℕ, n ≤ 1988 →
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % (sum_of_digits k) = 0 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_sum_of_digits_l3563_356341


namespace NUMINAMATH_CALUDE_simplify_expression_1_evaluate_expression_2_l3563_356337

-- Expression 1
theorem simplify_expression_1 (a : ℝ) : 
  -2 * a^2 + 3 - (3 * a^2 - 6 * a + 1) + 3 = -5 * a^2 + 6 * a + 2 := by sorry

-- Expression 2
theorem evaluate_expression_2 (x y : ℝ) (hx : x = -2) (hy : y = -3) :
  (1/2) * x - 2 * (x - (1/3) * y^2) + (-3/2 * x + (1/3) * y^2) = 15 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_evaluate_expression_2_l3563_356337


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3563_356317

theorem right_triangle_perimeter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_area : (1/2) * a * b = 150) (h_leg : a = 30) : 
  a + b + c = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3563_356317


namespace NUMINAMATH_CALUDE_no_nonzero_integer_solution_l3563_356387

theorem no_nonzero_integer_solution :
  ∀ (a b c n : ℤ), 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_integer_solution_l3563_356387


namespace NUMINAMATH_CALUDE_car_repair_cost_johns_car_repair_cost_l3563_356384

/-- Calculates the total cost of car repairs given labor rate, hours worked, and part cost -/
theorem car_repair_cost (labor_rate : ℕ) (hours : ℕ) (part_cost : ℕ) : 
  labor_rate * hours + part_cost = 2400 :=
by
  sorry

/-- Proves the specific case of John's car repair cost -/
theorem johns_car_repair_cost : 
  75 * 16 + 1200 = 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_car_repair_cost_johns_car_repair_cost_l3563_356384


namespace NUMINAMATH_CALUDE_football_tournament_semifinal_probability_l3563_356374

theorem football_tournament_semifinal_probability :
  let num_teams : ℕ := 8
  let num_semifinal_pairs : ℕ := 2
  let prob_win_match : ℚ := 1 / 2
  
  -- Probability of team B being in the correct subgroup
  let prob_correct_subgroup : ℚ := num_semifinal_pairs / (num_teams - 1)
  
  -- Probability of both teams winning their matches to reach semifinals
  let prob_both_win : ℚ := prob_win_match * prob_win_match
  
  -- Total probability
  prob_correct_subgroup * prob_both_win = 1 / 14 :=
by sorry

end NUMINAMATH_CALUDE_football_tournament_semifinal_probability_l3563_356374


namespace NUMINAMATH_CALUDE_weight_of_b_l3563_356340

theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 47 →
  b = 39 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l3563_356340


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l3563_356360

theorem cubic_polynomial_root (a b : ℝ) : 
  (∃ (x : ℂ), x^3 + a*x^2 - x + b = 0 ∧ x = 2 - 3*I) → 
  (a = 7.5 ∧ b = -45.5) := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l3563_356360


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l3563_356388

/-- The average birth rate in people per two seconds -/
def average_birth_rate : ℝ := 5

/-- The death rate in people per two seconds -/
def death_rate : ℝ := 3

/-- The number of two-second intervals in a day -/
def intervals_per_day : ℝ := 43200

/-- The net population increase in one day -/
def net_increase_per_day : ℝ := 86400

theorem birth_rate_calculation :
  (average_birth_rate - death_rate) * intervals_per_day = net_increase_per_day :=
by sorry

end NUMINAMATH_CALUDE_birth_rate_calculation_l3563_356388


namespace NUMINAMATH_CALUDE_gold_alloy_composition_l3563_356344

/-- Proves that adding 12 ounces of pure gold to an alloy weighing 48 ounces
    that is 25% gold will result in an alloy that is 40% gold. -/
theorem gold_alloy_composition (initial_weight : ℝ) (initial_gold_percentage : ℝ) 
    (final_gold_percentage : ℝ) (added_gold : ℝ) : 
  initial_weight = 48 →
  initial_gold_percentage = 0.25 →
  final_gold_percentage = 0.40 →
  added_gold = 12 →
  (initial_weight * initial_gold_percentage + added_gold) / (initial_weight + added_gold) = final_gold_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_gold_alloy_composition_l3563_356344


namespace NUMINAMATH_CALUDE_circle_equation_equivalence_l3563_356315

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- The equation of a circle given its center and radius -/
def circleEquation (c : Circle) (p : Point2D) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The specific circle we're considering -/
def specificCircle : Circle :=
  { center := { x := 0, y := 1 },
    radius := 2 }

/-- The equation we want to prove represents our circle -/
def givenEquation (p : Point2D) : Prop :=
  p.x^2 + (p.y - 1)^2 = 4

theorem circle_equation_equivalence :
  ∀ p : Point2D, circleEquation specificCircle p ↔ givenEquation p := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_equivalence_l3563_356315


namespace NUMINAMATH_CALUDE_dana_friday_hours_l3563_356301

/-- Dana's hourly rate in dollars -/
def hourly_rate : ℕ := 13

/-- Hours worked on Saturday -/
def saturday_hours : ℕ := 10

/-- Hours worked on Sunday -/
def sunday_hours : ℕ := 3

/-- Total earnings for all three days in dollars -/
def total_earnings : ℕ := 286

/-- Calculates the number of hours worked on Friday -/
def friday_hours : ℕ :=
  (total_earnings - (hourly_rate * (saturday_hours + sunday_hours))) / hourly_rate

theorem dana_friday_hours :
  friday_hours = 9 := by
  sorry

end NUMINAMATH_CALUDE_dana_friday_hours_l3563_356301


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l3563_356312

/-- A circle tangent to the y-axis with center on the line x - 3y = 0 and passing through (6, 1) -/
structure TangentCircle where
  -- Center of the circle
  center : ℝ × ℝ
  -- Radius of the circle
  radius : ℝ
  -- The circle is tangent to the y-axis
  tangent_to_y_axis : center.1 = radius
  -- The center is on the line x - 3y = 0
  center_on_line : center.1 = 3 * center.2
  -- The circle passes through (6, 1)
  passes_through_point : (center.1 - 6)^2 + (center.2 - 1)^2 = radius^2

/-- The equation of the circle is either (x-3)² + (y-1)² = 9 or (x-111)² + (y-37)² = 111² -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y, (x - 3)^2 + (y - 1)^2 = 9) ∨
  (∀ x y, (x - 111)^2 + (y - 37)^2 = 111^2) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l3563_356312


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3563_356364

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the line
def line (x y t : ℝ) : Prop := x - 3*y + t = 0

-- Define point M
def point_M (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define the asymptotes
def asymptotes (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = k*x ∨ y = -k*x}

-- Theorem statement
theorem hyperbola_asymptotes 
  (a b t : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (ht : t ≠ 0) :
  ∃ (A B : ℝ × ℝ),
    (∀ x y, hyperbola a b x y → line x y t → (x, y) ∈ asymptotes (1/2)) ∧
    (A ∈ asymptotes (1/2) ∧ B ∈ asymptotes (1/2)) ∧
    (line A.1 A.2 t ∧ line B.1 B.2 t) ∧
    (dist (point_M t) A = dist (point_M t) B) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3563_356364


namespace NUMINAMATH_CALUDE_probability_blue_given_glass_l3563_356304

theorem probability_blue_given_glass (total_red : ℕ) (total_blue : ℕ)
  (red_glass : ℕ) (red_wooden : ℕ) (blue_glass : ℕ) (blue_wooden : ℕ)
  (h1 : total_red = red_glass + red_wooden)
  (h2 : total_blue = blue_glass + blue_wooden)
  (h3 : total_red = 5)
  (h4 : total_blue = 11)
  (h5 : red_glass = 2)
  (h6 : red_wooden = 3)
  (h7 : blue_glass = 4)
  (h8 : blue_wooden = 7) :
  (blue_glass : ℚ) / (red_glass + blue_glass) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_probability_blue_given_glass_l3563_356304


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3563_356327

theorem complex_number_quadrant (m : ℝ) (h1 : 1 < m) (h2 : m < 3/2) :
  let z : ℂ := (3 + I) - m * (2 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3563_356327


namespace NUMINAMATH_CALUDE_milk_sold_in_fl_oz_l3563_356343

def monday_morning_milk : ℕ := 150 * 250 + 40 * 300 + 50 * 350
def monday_evening_milk : ℕ := 50 * 400 + 25 * 500 + 25 * 450
def tuesday_morning_milk : ℕ := 24 * 300 + 18 * 350 + 18 * 400
def tuesday_evening_milk : ℕ := 50 * 450 + 70 * 500 + 80 * 550

def total_milk_bought : ℕ := monday_morning_milk + monday_evening_milk + tuesday_morning_milk + tuesday_evening_milk
def remaining_milk : ℕ := 84000
def ml_per_fl_oz : ℕ := 30

theorem milk_sold_in_fl_oz :
  (total_milk_bought - remaining_milk) / ml_per_fl_oz = 4215 := by sorry

end NUMINAMATH_CALUDE_milk_sold_in_fl_oz_l3563_356343


namespace NUMINAMATH_CALUDE_max_visible_cubes_12_l3563_356365

/-- Represents a cube formed by unit cubes --/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from any single point --/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  3 * cube.size^2 - 3 * (cube.size - 1) + 1

/-- Theorem stating that for a 12 × 12 × 12 cube, the maximum number of visible unit cubes is 400 --/
theorem max_visible_cubes_12 :
  max_visible_cubes { size := 12 } = 400 := by
  sorry

#eval max_visible_cubes { size := 12 }

end NUMINAMATH_CALUDE_max_visible_cubes_12_l3563_356365


namespace NUMINAMATH_CALUDE_imaginary_power_minus_fraction_l3563_356331

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_minus_fraction : i^7 - 2 / i = i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_minus_fraction_l3563_356331


namespace NUMINAMATH_CALUDE_intersection_perpendicular_l3563_356345

/-- The line y = x - 2 intersects the parabola y^2 = 2x at points A and B. 
    This theorem proves that OA ⊥ OB, where O is the origin (0, 0). -/
theorem intersection_perpendicular (A B : ℝ × ℝ) : 
  (∃ x y : ℝ, A = (x, y) ∧ y = x - 2 ∧ y^2 = 2*x) →
  (∃ x y : ℝ, B = (x, y) ∧ y = x - 2 ∧ y^2 = 2*x) →
  A ≠ B →
  let O : ℝ × ℝ := (0, 0)
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0 :=
by sorry


end NUMINAMATH_CALUDE_intersection_perpendicular_l3563_356345


namespace NUMINAMATH_CALUDE_stating_acid_solution_mixing_l3563_356325

/-- 
Given an initial acid solution and a replacement acid solution,
calculate the final acid concentration after replacing a portion of the initial solution.
-/
def final_acid_concentration (initial_concentration : ℝ) (replacement_concentration : ℝ) (replaced_fraction : ℝ) : ℝ :=
  (initial_concentration * (1 - replaced_fraction) + replacement_concentration * replaced_fraction) * 100

/-- 
Theorem stating that replacing half of a 50% acid solution with a 20% acid solution 
results in a 35% acid solution.
-/
theorem acid_solution_mixing :
  final_acid_concentration 0.5 0.2 0.5 = 35 := by
sorry

#eval final_acid_concentration 0.5 0.2 0.5

end NUMINAMATH_CALUDE_stating_acid_solution_mixing_l3563_356325


namespace NUMINAMATH_CALUDE_age_problem_l3563_356377

theorem age_problem (x : ℝ) : (1/2) * (8 * (x + 8) - 8 * (x - 8)) = x ↔ x = 64 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l3563_356377


namespace NUMINAMATH_CALUDE_negative_f_reflection_l3563_356303

-- Define a function f
variable (f : ℝ → ℝ)

-- Define reflection across x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Theorem: The graph of y = -f(x) is the reflection of y = f(x) across the x-axis
theorem negative_f_reflection (x : ℝ) : 
  reflect_x (x, f x) = (x, -f x) := by sorry

end NUMINAMATH_CALUDE_negative_f_reflection_l3563_356303


namespace NUMINAMATH_CALUDE_gift_contribution_max_l3563_356318

/-- Given a group of people contributing money, calculates the maximum possible contribution by a single person. -/
def max_contribution (n : ℕ) (total : ℚ) (min_contribution : ℚ) : ℚ :=
  total - (n - 1 : ℚ) * min_contribution

/-- Theorem stating the maximum possible contribution in the given scenario. -/
theorem gift_contribution_max (n : ℕ) (total : ℚ) (min_contribution : ℚ)
  (h_n : n = 10)
  (h_total : total = 20)
  (h_min : min_contribution = 1)
  (h_positive : ∀ i, i ≤ n → min_contribution ≤ (max_contribution n total min_contribution)) :
  max_contribution n total min_contribution = 11 :=
by sorry

end NUMINAMATH_CALUDE_gift_contribution_max_l3563_356318


namespace NUMINAMATH_CALUDE_inequality_proof_l3563_356367

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) :
  (a^5 - 1) / (a^4 - 1) * (b^5 - 1) / (b^4 - 1) > 25/64 * (a + 1) * (b + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3563_356367


namespace NUMINAMATH_CALUDE_two_common_tangents_range_l3563_356393

/-- Two circles in a 2D plane --/
structure TwoCircles where
  a : ℝ
  c1 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - 2)^2 + y^2 = 4
  c2 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - a)^2 + (y + 3)^2 = 9

/-- The condition for two circles to have exactly two common tangents --/
def has_two_common_tangents (circles : TwoCircles) : Prop :=
  1 < Real.sqrt ((circles.a - 2)^2 + 9) ∧ Real.sqrt ((circles.a - 2)^2 + 9) < 5

/-- Theorem stating the range of 'a' for which the circles have exactly two common tangents --/
theorem two_common_tangents_range (circles : TwoCircles) :
  has_two_common_tangents circles ↔ -2 < circles.a ∧ circles.a < 6 := by
  sorry


end NUMINAMATH_CALUDE_two_common_tangents_range_l3563_356393


namespace NUMINAMATH_CALUDE_rain_ratio_proof_l3563_356350

/-- Proves that the ratio of rain time on the third day to the second day is 2:1 -/
theorem rain_ratio_proof (first_day : ℕ) (second_day : ℕ) (total_time : ℕ) :
  first_day = 10 →
  second_day = first_day + 2 →
  total_time = 46 →
  ∃ (third_day : ℕ), 
    first_day + second_day + third_day = total_time ∧
    third_day = 2 * second_day :=
by
  sorry

end NUMINAMATH_CALUDE_rain_ratio_proof_l3563_356350


namespace NUMINAMATH_CALUDE_fish_tank_problem_l3563_356322

theorem fish_tank_problem (fish_taken_out fish_remaining : ℕ) 
  (h1 : fish_taken_out = 16) 
  (h2 : fish_remaining = 3) : 
  fish_taken_out + fish_remaining = 19 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l3563_356322


namespace NUMINAMATH_CALUDE_average_weight_is_15_l3563_356334

def regression_weight (age : ℕ) : ℝ := 2 * age + 7

def children_ages : List ℕ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

theorem average_weight_is_15 :
  (children_ages.map regression_weight).sum / children_ages.length = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_is_15_l3563_356334


namespace NUMINAMATH_CALUDE_symmetry_wrt_origin_l3563_356359

/-- Given a point P(3, 2) in the Cartesian coordinate system, 
    its symmetrical point P' with respect to the origin has coordinates (-3, -2). -/
theorem symmetry_wrt_origin :
  let P : ℝ × ℝ := (3, 2)
  let P' : ℝ × ℝ := (-P.1, -P.2)
  P' = (-3, -2) := by sorry

end NUMINAMATH_CALUDE_symmetry_wrt_origin_l3563_356359


namespace NUMINAMATH_CALUDE_dodecahedral_die_expected_value_l3563_356308

/-- A fair dodecahedral die with faces numbered 1 to 12 -/
def DodecahedralDie : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 12

/-- The expected value of rolling the dodecahedral die -/
def expected_value : ℚ := (DodecahedralDie.sum (fun i => prob i * (i + 1)))

/-- Theorem: The expected value of rolling a fair dodecahedral die is 6.5 -/
theorem dodecahedral_die_expected_value : expected_value = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedral_die_expected_value_l3563_356308


namespace NUMINAMATH_CALUDE_expand_product_l3563_356316

theorem expand_product (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x^2 + 52 * x + 84 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3563_356316


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l3563_356363

theorem lcm_gcd_product (a b : ℕ) (ha : a = 240) (hb : b = 360) :
  Nat.lcm a b * Nat.gcd a b = 17280 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l3563_356363


namespace NUMINAMATH_CALUDE_expand_expression_l3563_356351

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3563_356351


namespace NUMINAMATH_CALUDE_pauls_hourly_wage_l3563_356320

/-- Calculates the hourly wage given the number of hours worked, tax rate, expense rate, and remaining money. -/
def calculate_hourly_wage (hours_worked : ℕ) (tax_rate : ℚ) (expense_rate : ℚ) (remaining_money : ℚ) : ℚ :=
  remaining_money / ((1 - expense_rate) * ((1 - tax_rate) * hours_worked))

/-- Theorem stating that under the given conditions, the hourly wage is $12.50 -/
theorem pauls_hourly_wage :
  let hours_worked : ℕ := 40
  let tax_rate : ℚ := 1/5
  let expense_rate : ℚ := 3/20
  let remaining_money : ℚ := 340
  calculate_hourly_wage hours_worked tax_rate expense_rate remaining_money = 25/2 := by
  sorry


end NUMINAMATH_CALUDE_pauls_hourly_wage_l3563_356320


namespace NUMINAMATH_CALUDE_sum_of_ages_is_50_l3563_356309

/-- The sum of ages of 5 children born 2 years apart, where the eldest child is 14 years old -/
def sum_of_ages : ℕ → Prop
| n => ∃ (a b c d e : ℕ),
    a = 14 ∧
    b = a - 2 ∧
    c = b - 2 ∧
    d = c - 2 ∧
    e = d - 2 ∧
    n = a + b + c + d + e

/-- Theorem stating that the sum of ages is 50 years -/
theorem sum_of_ages_is_50 : sum_of_ages 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_50_l3563_356309


namespace NUMINAMATH_CALUDE_largest_integer_less_than_85_remainder_2_mod_6_l3563_356346

theorem largest_integer_less_than_85_remainder_2_mod_6 : 
  ∃ (n : ℤ), n < 85 ∧ n % 6 = 2 ∧ ∀ (m : ℤ), m < 85 ∧ m % 6 = 2 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_85_remainder_2_mod_6_l3563_356346


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l3563_356380

theorem cost_increase_percentage (initial_cost final_cost : ℝ) 
  (h1 : initial_cost = 75)
  (h2 : final_cost = 72)
  (h3 : ∃ x : ℝ, final_cost = (initial_cost + (x / 100) * initial_cost) * 0.8) :
  ∃ x : ℝ, x = 20 ∧ final_cost = (initial_cost + (x / 100) * initial_cost) * 0.8 := by
  sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l3563_356380


namespace NUMINAMATH_CALUDE_f_properties_l3563_356305

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x * Real.cos x - 5 * Real.sqrt 3 * (Real.cos x)^2 + 5/2 * Real.sqrt 3

theorem f_properties :
  let T := Real.pi
  ∀ (k : ℤ),
    (∀ (x : ℝ), f (x + T) = f x) ∧  -- f has period T
    (∀ (S : ℝ), S > 0 → (∀ (x : ℝ), f (x + S) = f x) → S ≥ T) ∧  -- T is the smallest positive period
    (∀ (x : ℝ), x ∈ Set.Icc (k * Real.pi - Real.pi/12) (k * Real.pi + 5 * Real.pi/12) → 
      ∀ (y : ℝ), y ∈ Set.Icc (k * Real.pi - Real.pi/12) (k * Real.pi + 5 * Real.pi/12) → 
        x ≤ y → f x ≤ f y) ∧  -- f is increasing on [kπ - π/12, kπ + 5π/12]
    (∀ (x : ℝ), x ∈ Set.Icc (k * Real.pi + 5 * Real.pi/12) (k * Real.pi + 11 * Real.pi/12) → 
      ∀ (y : ℝ), y ∈ Set.Icc (k * Real.pi + 5 * Real.pi/12) (k * Real.pi + 11 * Real.pi/12) → 
        x ≤ y → f x ≥ f y)  -- f is decreasing on [kπ + 5π/12, kπ + 11π/12]
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l3563_356305


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3563_356389

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 1 + a 2 = 20) →
  (a 3 + a 4 = 40) →
  (a 5 + a 6 = 80) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3563_356389


namespace NUMINAMATH_CALUDE_equal_numbers_product_l3563_356397

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 25 →
  c = 18 →
  d = e →
  d * e = 506.25 := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l3563_356397


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_2_mod_17_l3563_356307

theorem smallest_five_digit_congruent_to_2_mod_17 : ∃ (n : ℕ),
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit positive integer
  (n % 17 = 2) ∧              -- congruent to 2 modulo 17
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 2 → m ≥ n) ∧  -- smallest such integer
  n = 10013 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_2_mod_17_l3563_356307


namespace NUMINAMATH_CALUDE_mr_martin_coffee_cups_verify_mrs_martin_purchase_verify_mr_martin_purchase_l3563_356375

-- Define the cost of items
def bagel_cost : ℝ := 1.5
def coffee_cost : ℝ := 3.25

-- Define Mrs. Martin's purchase
def mrs_martin_coffee : ℕ := 3
def mrs_martin_bagels : ℕ := 2
def mrs_martin_total : ℝ := 12.75

-- Define Mr. Martin's purchase
def mr_martin_bagels : ℕ := 5
def mr_martin_total : ℝ := 14.00

-- Theorem to prove
theorem mr_martin_coffee_cups : ℕ := by
  -- The number of coffee cups Mr. Martin bought
  sorry

-- Verify Mrs. Martin's purchase
theorem verify_mrs_martin_purchase :
  mrs_martin_coffee * coffee_cost + mrs_martin_bagels * bagel_cost = mrs_martin_total := by
  sorry

-- Verify Mr. Martin's purchase
theorem verify_mr_martin_purchase :
  mr_martin_coffee_cups * coffee_cost + mr_martin_bagels * bagel_cost = mr_martin_total := by
  sorry

end NUMINAMATH_CALUDE_mr_martin_coffee_cups_verify_mrs_martin_purchase_verify_mr_martin_purchase_l3563_356375


namespace NUMINAMATH_CALUDE_technicians_sample_size_l3563_356302

/-- Represents the number of technicians to be included in a stratified sample -/
def technicians_in_sample (total_engineers : ℕ) (total_technicians : ℕ) (total_workers : ℕ) (sample_size : ℕ) : ℕ :=
  (total_technicians * sample_size) / (total_engineers + total_technicians + total_workers)

/-- Theorem stating that the number of technicians in the sample is 5 -/
theorem technicians_sample_size :
  technicians_in_sample 20 100 280 20 = 5 := by
  sorry

#eval technicians_in_sample 20 100 280 20

end NUMINAMATH_CALUDE_technicians_sample_size_l3563_356302


namespace NUMINAMATH_CALUDE_parallel_lines_from_parallel_planes_l3563_356310

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the intersection operation for planes
variable (planeIntersection : Plane → Plane → Line)

-- Define the subset relation for lines and planes
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_lines_from_parallel_planes 
  (m n : Line) (α β γ : Plane)
  (nonCoincidentLines : m ≠ n)
  (nonCoincidentPlanes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (planesParallel : parallelPlanes α β)
  (mIntersection : planeIntersection α γ = m)
  (nIntersection : planeIntersection β γ = n) :
  parallelLines m n :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_from_parallel_planes_l3563_356310


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l3563_356324

theorem complex_sum_magnitude (a b c : ℂ) : 
  Complex.abs a = 1 → 
  Complex.abs b = 1 → 
  Complex.abs c = 1 → 
  a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = -3 → 
  Complex.abs (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l3563_356324


namespace NUMINAMATH_CALUDE_nico_reading_proof_l3563_356335

/-- The number of pages Nico read on Monday -/
def pages_monday : ℕ := 39

/-- The number of pages Nico read on Tuesday -/
def pages_tuesday : ℕ := 12

/-- The total number of pages Nico read over three days -/
def total_pages : ℕ := 51

/-- The number of books Nico borrowed -/
def num_books : ℕ := 3

/-- The number of days Nico read -/
def num_days : ℕ := 3

theorem nico_reading_proof :
  pages_monday = total_pages - pages_tuesday ∧
  pages_monday + pages_tuesday ≤ total_pages ∧
  num_books = num_days := by sorry

end NUMINAMATH_CALUDE_nico_reading_proof_l3563_356335


namespace NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l3563_356395

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by sorry

end NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l3563_356395


namespace NUMINAMATH_CALUDE_simplified_win_ratio_l3563_356353

def chloe_wins : ℕ := 24
def max_wins : ℕ := 9

theorem simplified_win_ratio : 
  ∃ (a b : ℕ), a = 8 ∧ b = 3 ∧ chloe_wins * b = max_wins * a := by
  sorry

end NUMINAMATH_CALUDE_simplified_win_ratio_l3563_356353


namespace NUMINAMATH_CALUDE_one_cubic_yard_equals_27_cubic_feet_l3563_356330

-- Define the conversion rate between yards and feet
def yard_to_feet : ℝ := 3

-- Define a cubic yard in terms of cubic feet
def cubic_yard_to_cubic_feet : ℝ := yard_to_feet ^ 3

-- Theorem statement
theorem one_cubic_yard_equals_27_cubic_feet :
  cubic_yard_to_cubic_feet = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_cubic_yard_equals_27_cubic_feet_l3563_356330


namespace NUMINAMATH_CALUDE_toms_shirt_purchase_cost_l3563_356329

/-- The total cost of Tom's shirt purchase --/
def totalCost (numFandoms : ℕ) (shirtsPerFandom : ℕ) (originalPrice : ℚ) (discountPercentage : ℚ) (taxRate : ℚ) : ℚ :=
  let totalShirts := numFandoms * shirtsPerFandom
  let discountAmount := originalPrice * discountPercentage
  let discountedPrice := originalPrice - discountAmount
  let subtotal := totalShirts * discountedPrice
  let taxAmount := subtotal * taxRate
  subtotal + taxAmount

/-- Theorem stating that Tom's total cost is $264 --/
theorem toms_shirt_purchase_cost :
  totalCost 4 5 15 0.2 0.1 = 264 := by
  sorry

end NUMINAMATH_CALUDE_toms_shirt_purchase_cost_l3563_356329


namespace NUMINAMATH_CALUDE_total_carrots_l3563_356321

theorem total_carrots (sally_carrots fred_carrots : ℕ) :
  sally_carrots = 6 → fred_carrots = 4 → sally_carrots + fred_carrots = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_l3563_356321


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3563_356314

theorem quadratic_factorization (x : ℝ) : x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3563_356314


namespace NUMINAMATH_CALUDE_max_valid_ltrominos_eighteen_is_achievable_l3563_356382

-- Define the colors
inductive Color
  | Red
  | Green
  | Blue

-- Define the grid
def Grid := Fin 4 → Fin 4 → Color

-- Define an L-tromino
structure LTromino where
  x : Fin 4
  y : Fin 4
  orientation : Fin 4

-- Function to check if an L-tromino has one square of each color
def hasOneOfEachColor (g : Grid) (l : LTromino) : Bool := sorry

-- Function to count valid L-trominos in a grid
def countValidLTrominos (g : Grid) : Nat := sorry

-- Theorem statement
theorem max_valid_ltrominos (g : Grid) : 
  countValidLTrominos g ≤ 18 := sorry

-- Theorem stating that 18 is achievable
theorem eighteen_is_achievable : 
  ∃ g : Grid, countValidLTrominos g = 18 := sorry

end NUMINAMATH_CALUDE_max_valid_ltrominos_eighteen_is_achievable_l3563_356382


namespace NUMINAMATH_CALUDE_meter_to_step_conversion_l3563_356348

-- Define our units of measurement
variable (hops skips jumps steps meters : ℚ)

-- Define the relationships between units
variable (hop_skip_relation : 2 * hops = 3 * skips)
variable (jump_hop_relation : 4 * jumps = 6 * hops)
variable (jump_meter_relation : 5 * jumps = 20 * meters)
variable (skip_step_relation : 15 * skips = 10 * steps)

-- State the theorem
theorem meter_to_step_conversion :
  1 * meters = 3/8 * steps :=
sorry

end NUMINAMATH_CALUDE_meter_to_step_conversion_l3563_356348


namespace NUMINAMATH_CALUDE_intersection_point_l3563_356386

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y + 2 = 0

def C₂ (x y : ℝ) : Prop := y^2 = 8*x

-- State the theorem
theorem intersection_point :
  ∃! p : ℝ × ℝ, C₁ p.1 p.2 ∧ C₂ p.1 p.2 ∧ p = (2, -4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3563_356386


namespace NUMINAMATH_CALUDE_small_circle_area_l3563_356339

/-- Configuration of circles -/
structure CircleConfiguration where
  large_circle_area : ℝ
  small_circle_count : ℕ
  small_circles_inscribed : Prop

/-- Theorem: In a configuration where 6 small circles of equal radius are inscribed 
    in a large circle with an area of 120, the area of each small circle is 40 -/
theorem small_circle_area 
  (config : CircleConfiguration) 
  (h1 : config.large_circle_area = 120)
  (h2 : config.small_circle_count = 6)
  (h3 : config.small_circles_inscribed) :
  ∃ (small_circle_area : ℝ), small_circle_area = 40 ∧ 
    config.small_circle_count * small_circle_area = config.large_circle_area :=
by
  sorry


end NUMINAMATH_CALUDE_small_circle_area_l3563_356339


namespace NUMINAMATH_CALUDE_pat_calculation_l3563_356300

theorem pat_calculation (x : ℝ) : (x / 7 + 10 = 20) → (x * 7 - 10 = 480) := by
  sorry

end NUMINAMATH_CALUDE_pat_calculation_l3563_356300


namespace NUMINAMATH_CALUDE_assignment_effect_l3563_356385

/-- Represents the effect of the assignment statement M = M + 3 --/
theorem assignment_effect (M : ℤ) : 
  let M' := M + 3
  M' = M + 3 := by sorry

end NUMINAMATH_CALUDE_assignment_effect_l3563_356385


namespace NUMINAMATH_CALUDE_R_squared_eq_one_when_no_error_l3563_356362

/-- A structure representing a set of observations in a linear regression model. -/
structure LinearRegressionData (n : ℕ) where
  x : Fin n → ℝ
  y : Fin n → ℝ
  a : ℝ
  b : ℝ
  e : Fin n → ℝ

/-- The coefficient of determination (R-squared) for a linear regression model. -/
def R_squared (data : LinearRegressionData n) : ℝ :=
  sorry

/-- Theorem stating that if all error terms are zero, then R-squared equals 1. -/
theorem R_squared_eq_one_when_no_error (n : ℕ) (data : LinearRegressionData n)
  (h1 : ∀ i, data.y i = data.b * data.x i + data.a + data.e i)
  (h2 : ∀ i, data.e i = 0) :
  R_squared data = 1 :=
sorry

end NUMINAMATH_CALUDE_R_squared_eq_one_when_no_error_l3563_356362


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3563_356373

theorem complex_equation_solution : ∃ (a : ℝ) (b c : ℂ),
  a + b + c = 5 ∧
  a * b + b * c + c * a = 7 ∧
  a * b * c = 3 ∧
  (a = 1 ∨ a = 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3563_356373


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3563_356336

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3563_356336


namespace NUMINAMATH_CALUDE_line_xz_plane_intersection_l3563_356355

/-- The line passing through two points intersects the xz-plane at a specific point -/
theorem line_xz_plane_intersection (p₁ p₂ q : ℝ × ℝ × ℝ) : 
  p₁ = (2, 3, 5) → 
  p₂ = (4, 0, 9) → 
  (∃ t : ℝ, q = p₁ + t • (p₂ - p₁)) → 
  q.2 = 0 → 
  q = (4, 0, 9) := by
  sorry

#check line_xz_plane_intersection

end NUMINAMATH_CALUDE_line_xz_plane_intersection_l3563_356355


namespace NUMINAMATH_CALUDE_carnations_in_third_bouquet_l3563_356356

theorem carnations_in_third_bouquet 
  (total_bouquets : ℕ)
  (first_bouquet : ℕ)
  (second_bouquet : ℕ)
  (average_carnations : ℕ)
  (h1 : total_bouquets = 3)
  (h2 : first_bouquet = 9)
  (h3 : second_bouquet = 14)
  (h4 : average_carnations = 12) :
  average_carnations * total_bouquets - (first_bouquet + second_bouquet) = 13 :=
by
  sorry

#check carnations_in_third_bouquet

end NUMINAMATH_CALUDE_carnations_in_third_bouquet_l3563_356356


namespace NUMINAMATH_CALUDE_high_school_twelve_games_l3563_356349

/-- The number of teams in the "High School Twelve" basketball league -/
def num_teams : ℕ := 12

/-- The number of times each team plays every other team in the league -/
def games_per_pair : ℕ := 3

/-- The number of games each team plays against non-league teams -/
def non_league_games_per_team : ℕ := 6

/-- The total number of games in a season for the "High School Twelve" basketball league -/
def total_games : ℕ := (num_teams.choose 2) * games_per_pair + num_teams * non_league_games_per_team

theorem high_school_twelve_games :
  total_games = 270 := by sorry

end NUMINAMATH_CALUDE_high_school_twelve_games_l3563_356349


namespace NUMINAMATH_CALUDE_equation_solutions_l3563_356379

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1^2 - 5*x1 - 6 = 0 ∧ x2^2 - 5*x2 - 6 = 0 ∧ x1 = 6 ∧ x2 = -1) ∧
  (∃ y1 y2 : ℝ, (y1 + 1)*(y1 - 1) + y1*(y1 + 2) = 7 + 6*y1 ∧
                (y2 + 1)*(y2 - 1) + y2*(y2 + 2) = 7 + 6*y2 ∧
                y1 = Real.sqrt 5 + 1 ∧ y2 = 1 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3563_356379


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l3563_356394

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}
def N : Set ℝ := {x | 2*x - x^2 ≥ 0}

-- Define the open interval (1, 2]
def open_closed_interval : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem intersection_equals_interval : M ∩ N = open_closed_interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l3563_356394


namespace NUMINAMATH_CALUDE_complex_equation_implication_l3563_356399

theorem complex_equation_implication (x y : ℝ) : 
  Complex.I * Real.exp (-1) + 2 = y + x * Complex.I → x^3 + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implication_l3563_356399


namespace NUMINAMATH_CALUDE_angle_F_measure_l3563_356368

-- Define a triangle DEF
structure Triangle :=
  (D E F : ℝ)

-- Define the properties of the triangle
def validTriangle (t : Triangle) : Prop :=
  t.D > 0 ∧ t.E > 0 ∧ t.F > 0 ∧ t.D + t.E + t.F = 180

-- Theorem statement
theorem angle_F_measure (t : Triangle) 
  (h1 : validTriangle t) 
  (h2 : t.D = 3 * t.E) 
  (h3 : t.E = 18) : 
  t.F = 108 := by
  sorry

end NUMINAMATH_CALUDE_angle_F_measure_l3563_356368


namespace NUMINAMATH_CALUDE_perimeter_is_ten_x_l3563_356398

/-- The perimeter of a figure composed of rectangular segments -/
def perimeter_of_figure (x : ℝ) (hx : x ≠ 0) : ℝ :=
  let vertical_length1 := 3 * x
  let vertical_length2 := x
  let horizontal_length1 := 2 * x
  let horizontal_length2 := x
  vertical_length1 + vertical_length2 + horizontal_length1 + horizontal_length2 + 
  (3 * x - x) + (2 * x - x)

theorem perimeter_is_ten_x (x : ℝ) (hx : x ≠ 0) :
  perimeter_of_figure x hx = 10 * x := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_ten_x_l3563_356398


namespace NUMINAMATH_CALUDE_average_of_remaining_digits_l3563_356358

theorem average_of_remaining_digits 
  (total_digits : Nat) 
  (subset_digits : Nat)
  (total_average : ℚ) 
  (subset_average : ℚ) :
  total_digits = 9 →
  subset_digits = 4 →
  total_average = 18 →
  subset_average = 8 →
  (total_digits * total_average - subset_digits * subset_average) / (total_digits - subset_digits) = 26 :=
by sorry

end NUMINAMATH_CALUDE_average_of_remaining_digits_l3563_356358


namespace NUMINAMATH_CALUDE_committee_formation_count_l3563_356313

theorem committee_formation_count (n m k : ℕ) (hn : n = 10) (hm : m = 5) (hk : k = 1) :
  (Nat.choose (n - k) (m - k)) = 126 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l3563_356313


namespace NUMINAMATH_CALUDE_rectangle_width_is_five_l3563_356357

/-- A rectangle with specific properties -/
structure Rectangle where
  length : ℝ
  width : ℝ
  width_longer : width = length + 2
  perimeter : length * 2 + width * 2 = 16

/-- The width of the rectangle is 5 -/
theorem rectangle_width_is_five (r : Rectangle) : r.width = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_is_five_l3563_356357


namespace NUMINAMATH_CALUDE_second_train_start_time_l3563_356342

/-- The time when the trains meet, in hours after midnight -/
def meeting_time : ℝ := 12

/-- The time when the first train starts, in hours after midnight -/
def train1_start_time : ℝ := 7

/-- The speed of the first train in km/h -/
def train1_speed : ℝ := 20

/-- The speed of the second train in km/h -/
def train2_speed : ℝ := 25

/-- The distance between stations A and B in km -/
def total_distance : ℝ := 200

/-- The theorem stating that the second train must have started at 8 a.m. -/
theorem second_train_start_time :
  ∃ (train2_start_time : ℝ),
    train2_start_time = 8 ∧
    (meeting_time - train1_start_time) * train1_speed +
    (meeting_time - train2_start_time) * train2_speed = total_distance :=
by sorry

end NUMINAMATH_CALUDE_second_train_start_time_l3563_356342


namespace NUMINAMATH_CALUDE_monomial_replacement_four_terms_l3563_356370

/-- Given an expression (x^4 - 3)^2 + (x^3 + *)^2, where * is to be replaced by a monomial,
    prove that replacing * with (x^3 + 3x) results in an expression with exactly four terms
    after squaring and combining like terms. -/
theorem monomial_replacement_four_terms (x : ℝ) : 
  let original_expr := (x^4 - 3)^2 + (x^3 + (x^3 + 3*x))^2
  ∃ (a b c d : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    original_expr = a * x^n₁ + b * x^n₂ + c * x^n₃ + d * x^n₄ ∧
    n₁ > n₂ ∧ n₂ > n₃ ∧ n₃ > n₄ ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_monomial_replacement_four_terms_l3563_356370


namespace NUMINAMATH_CALUDE_units_digit_problem_l3563_356352

theorem units_digit_problem :
  ∃ n : ℕ, (15 + Real.sqrt 221)^19 + 3 * (15 + Real.sqrt 221)^83 = 10 * n + 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l3563_356352


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3563_356323

theorem sum_of_two_numbers (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = a + c ∨ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3563_356323


namespace NUMINAMATH_CALUDE_tournament_participants_l3563_356366

theorem tournament_participants : ∃ (n : ℕ), n > 0 ∧ 
  (n * (n - 1) / 2 : ℚ) = 90 + (n - 10) * (n - 11) ∧ 
  (∀ k : ℕ, k ≠ n → (k * (k - 1) / 2 : ℚ) ≠ 90 + (k - 10) * (k - 11)) := by
  sorry

end NUMINAMATH_CALUDE_tournament_participants_l3563_356366


namespace NUMINAMATH_CALUDE_a_sixth_bounds_l3563_356396

-- Define the condition
def condition (a : ℝ) : Prop := a^5 - a^3 + a = 2

-- State the theorem
theorem a_sixth_bounds {a : ℝ} (h : condition a) : 3 < a^6 ∧ a^6 < 4 := by
  sorry

end NUMINAMATH_CALUDE_a_sixth_bounds_l3563_356396


namespace NUMINAMATH_CALUDE_spending_ratio_l3563_356383

def monthly_allowance : ℚ := 12

def spending_scenario (first_week_spending : ℚ) : Prop :=
  let remaining_after_first_week := monthly_allowance - first_week_spending
  let second_week_spending := (1 / 4) * remaining_after_first_week
  monthly_allowance - first_week_spending - second_week_spending = 6

theorem spending_ratio :
  ∃ (first_week_spending : ℚ),
    spending_scenario first_week_spending ∧
    first_week_spending / monthly_allowance = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_spending_ratio_l3563_356383


namespace NUMINAMATH_CALUDE_inequality_theorem_l3563_356306

def inequality_solution (x : ℝ) : Prop :=
  x ≠ 4 ∧ (x^2 - 1) / ((x - 4)^2) ≥ 0

theorem inequality_theorem :
  {x : ℝ | inequality_solution x} = 
    Set.Iic (-1) ∪ Set.Icc 1 4 ∪ Set.Ioi 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3563_356306


namespace NUMINAMATH_CALUDE_longer_side_length_l3563_356378

/-- A rectangular plot with fence poles -/
structure FencedPlot where
  width : ℝ
  length : ℝ
  pole_distance : ℝ
  pole_count : ℕ

/-- The perimeter of a rectangle -/
def perimeter (plot : FencedPlot) : ℝ :=
  2 * (plot.width + plot.length)

/-- The total length of fencing -/
def fencing_length (plot : FencedPlot) : ℝ :=
  (plot.pole_count - 1 : ℝ) * plot.pole_distance

theorem longer_side_length (plot : FencedPlot) 
  (h1 : plot.width = 15)
  (h2 : plot.pole_distance = 5)
  (h3 : plot.pole_count = 26)
  (h4 : plot.width < plot.length)
  (h5 : perimeter plot = fencing_length plot) :
  plot.length = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_longer_side_length_l3563_356378


namespace NUMINAMATH_CALUDE_prism_height_l3563_356328

theorem prism_height (ab ac : ℝ) (volume : ℝ) (h1 : ab = ac) (h2 : ab = Real.sqrt 2) (h3 : volume = 3.0000000000000004) :
  let base_area := (1 / 2) * ab * ac
  let height := volume / base_area
  height = 3.0000000000000004 := by
sorry

end NUMINAMATH_CALUDE_prism_height_l3563_356328


namespace NUMINAMATH_CALUDE_books_returned_percentage_l3563_356347

-- Define the initial number of books
def initial_books : ℕ := 75

-- Define the final number of books
def final_books : ℕ := 63

-- Define the number of books loaned out (rounded to 40)
def loaned_books : ℕ := 40

-- Define the percentage of books returned
def percentage_returned : ℚ := 70

-- Theorem statement
theorem books_returned_percentage :
  (((initial_books - final_books : ℚ) / loaned_books) * 100 = percentage_returned) :=
sorry

end NUMINAMATH_CALUDE_books_returned_percentage_l3563_356347


namespace NUMINAMATH_CALUDE_fraction_relationship_l3563_356361

theorem fraction_relationship (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relationship_l3563_356361


namespace NUMINAMATH_CALUDE_ten_ways_to_form_ten_l3563_356332

/-- Represents the number of ways to form a given amount using $1 coins, $2 coins, and $5 bills -/
def ways_to_form (amount : ℕ) : ℕ :=
  (Finset.range (amount + 1)).sum (λ i =>
    (Finset.range ((amount - i) / 2 + 1)).sum (λ j =>
      if i + 2 * j + 5 * ((amount - i - 2 * j) / 5) = amount then 1 else 0
    )
  )

/-- The main theorem stating that there are exactly 10 ways to form $10 -/
theorem ten_ways_to_form_ten : ways_to_form 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_ways_to_form_ten_l3563_356332


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3563_356311

/-- The range of m for which the quadratic inequality mx^2 - mx + 1 < 0 has a non-empty solution set -/
theorem quadratic_inequality_solution_range :
  {m : ℝ | ∃ x, m * x^2 - m * x + 1 < 0} = {m | m < 0 ∨ m > 4} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3563_356311


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l3563_356390

theorem gcd_special_numbers : 
  Nat.gcd (2^2048 - 1) (2^2024 - 1) = 2^24 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l3563_356390


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l3563_356371

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits (n : ℕ) : ℕ := n / 100

def last_two_digits (n : ℕ) : ℕ := n % 100

def satisfies_conditions (n : ℕ) : Prop :=
  is_four_digit n ∧
  n % 3 = 0 ∧
  first_two_digits n - last_two_digits n = 11

def solution_set : Set ℕ := {1302, 1605, 1908, 2211, 2514, 2817, 3120, 3423, 3726, 4029, 4332, 4635, 4938, 5241, 5544, 5847, 6150, 6453, 6756, 7059, 7362, 7665, 7968, 8271, 8574, 8877, 9180, 9483, 9786, 10089, 10392, 10695, 10998}

theorem four_digit_number_theorem :
  {n : ℕ | satisfies_conditions n} = solution_set := by sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l3563_356371
