import Mathlib

namespace NUMINAMATH_CALUDE_square_side_length_l2264_226481

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 12) (h2 : side > 0) (h3 : area = side ^ 2) :
  side = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l2264_226481


namespace NUMINAMATH_CALUDE_rose_difference_is_34_l2264_226471

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := 58

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The difference in the number of red roses between Mrs. Santiago and Mrs. Garrett -/
def rose_difference : ℕ := santiago_roses - garrett_roses

theorem rose_difference_is_34 : rose_difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_rose_difference_is_34_l2264_226471


namespace NUMINAMATH_CALUDE_two_zeros_read_in_2006_06_l2264_226465

-- Define a function to count the number of zeros read in a number
def countZerosRead (n : ℝ) : ℕ := sorry

-- Define the given numbers
def num1 : ℝ := 200.06
def num2 : ℝ := 20.06
def num3 : ℝ := 2006.06

-- Theorem statement
theorem two_zeros_read_in_2006_06 :
  (countZerosRead num1 < 2) ∧
  (countZerosRead num2 < 2) ∧
  (countZerosRead num3 = 2) :=
sorry

end NUMINAMATH_CALUDE_two_zeros_read_in_2006_06_l2264_226465


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2264_226470

theorem rectangle_dimensions :
  ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  l = 2 * w →
  w * l = (1/2) * (2 * (w + l)) →
  w = (3/2) ∧ l = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2264_226470


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2264_226435

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_perimeter_l2264_226435


namespace NUMINAMATH_CALUDE_xyz_sum_product_sqrt_l2264_226421

theorem xyz_sum_product_sqrt (x y z : ℝ) 
  (h1 : y + z = 16)
  (h2 : z + x = 17)
  (h3 : x + y = 18) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 1831.78125 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_product_sqrt_l2264_226421


namespace NUMINAMATH_CALUDE_practice_hours_until_game_l2264_226424

/-- Calculates the total practice hours for a given number of weeks -/
def total_practice_hours (weeks : ℕ) : ℕ :=
  let weekday_hours := 3
  let weekday_count := 5
  let saturday_hours := 5
  let weekly_hours := weekday_hours * weekday_count + saturday_hours
  weekly_hours * weeks

/-- The number of weeks until the next game -/
def weeks_until_game : ℕ := 3

theorem practice_hours_until_game :
  total_practice_hours weeks_until_game = 60 := by
  sorry

end NUMINAMATH_CALUDE_practice_hours_until_game_l2264_226424


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l2264_226402

theorem gold_coin_distribution (x y : ℕ) (h1 : x + y = 16) (h2 : x ≠ y) :
  ∃ k : ℕ, x^2 - y^2 = k * (x - y) → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l2264_226402


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2264_226485

theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (first_term : a 1 = 5) 
  (fifth_term : a 5 = 2025) :
  a 3 = 225 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2264_226485


namespace NUMINAMATH_CALUDE_peach_pies_l2264_226438

theorem peach_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio peach_ratio : ℕ) : 
  total_pies = 36 →
  apple_ratio = 1 →
  blueberry_ratio = 4 →
  cherry_ratio = 3 →
  peach_ratio = 2 →
  (peach_ratio : ℚ) * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio + peach_ratio) = 8 := by
  sorry

#check peach_pies

end NUMINAMATH_CALUDE_peach_pies_l2264_226438


namespace NUMINAMATH_CALUDE_snack_spending_l2264_226414

/-- The total amount spent by Robert and Teddy on snacks for their friends -/
def total_spent (pizza_price : ℕ) (pizza_quantity : ℕ) (drink_price : ℕ) (robert_drink_quantity : ℕ) 
  (hamburger_price : ℕ) (hamburger_quantity : ℕ) (teddy_drink_quantity : ℕ) : ℕ :=
  pizza_price * pizza_quantity + 
  drink_price * (robert_drink_quantity + teddy_drink_quantity) + 
  hamburger_price * hamburger_quantity

/-- Theorem stating that Robert and Teddy spend $108 in total -/
theorem snack_spending : 
  total_spent 10 5 2 10 3 6 10 = 108 := by
  sorry

end NUMINAMATH_CALUDE_snack_spending_l2264_226414


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l2264_226442

/-- Given a compound where 3 moles weigh 528 grams, prove its molecular weight is 176 grams/mole. -/
theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 528)
  (h2 : num_moles = 3) :
  total_weight / num_moles = 176 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l2264_226442


namespace NUMINAMATH_CALUDE_lunch_break_duration_l2264_226497

structure PaintingScenario where
  paula_rate : ℝ
  assistants_rate : ℝ
  lunch_break : ℝ

def monday_work (s : PaintingScenario) : ℝ :=
  (9 - s.lunch_break) * (s.paula_rate + s.assistants_rate)

def tuesday_work (s : PaintingScenario) : ℝ :=
  (7 - s.lunch_break) * s.assistants_rate

def wednesday_work (s : PaintingScenario) : ℝ :=
  (10 - s.lunch_break) * s.paula_rate

theorem lunch_break_duration (s : PaintingScenario) :
  monday_work s = 0.6 →
  tuesday_work s = 0.3 →
  wednesday_work s = 0.1 →
  s.lunch_break = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l2264_226497


namespace NUMINAMATH_CALUDE_sequence_with_constant_triple_sum_l2264_226409

theorem sequence_with_constant_triple_sum :
  ∃! (a : Fin 8 → ℝ), 
    a 0 = 5 ∧ 
    a 7 = 8 ∧ 
    (∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 20) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_constant_triple_sum_l2264_226409


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l2264_226488

/-- 
Given an equation x^2 + y^2 + mx - 2y + 4 = 0 that represents a circle,
prove that m must be in the range (-∞, -2√3) ∪ (2√3, +∞).
-/
theorem circle_equation_m_range :
  ∀ m : ℝ, 
  (∃ x y : ℝ, x^2 + y^2 + m*x - 2*y + 4 = 0) →
  (m < -2 * Real.sqrt 3 ∨ m > 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l2264_226488


namespace NUMINAMATH_CALUDE_jerry_cut_eight_pine_trees_l2264_226407

/-- The number of logs produced by one pine tree -/
def logs_per_pine : ℕ := 80

/-- The number of logs produced by one maple tree -/
def logs_per_maple : ℕ := 60

/-- The number of logs produced by one walnut tree -/
def logs_per_walnut : ℕ := 100

/-- The number of maple trees Jerry cut -/
def maple_trees : ℕ := 3

/-- The number of walnut trees Jerry cut -/
def walnut_trees : ℕ := 4

/-- The total number of logs Jerry got -/
def total_logs : ℕ := 1220

/-- Theorem stating that Jerry cut 8 pine trees -/
theorem jerry_cut_eight_pine_trees :
  ∃ (pine_trees : ℕ), pine_trees * logs_per_pine + 
                      maple_trees * logs_per_maple + 
                      walnut_trees * logs_per_walnut = total_logs ∧ 
                      pine_trees = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerry_cut_eight_pine_trees_l2264_226407


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2264_226492

/-- The minimum value of (x - 2)^2 + (y - 2)^2 when (x, y) lies on the line x - y - 1 = 0 -/
theorem min_distance_to_line : 
  ∃ (min : ℝ), min = (1/2 : ℝ) ∧ 
  ∀ (x y : ℝ), x - y - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2264_226492


namespace NUMINAMATH_CALUDE_cost_per_cow_l2264_226444

/-- Calculates the cost per cow given Timothy's expenses --/
theorem cost_per_cow (land_acres : ℕ) (land_cost_per_acre : ℕ)
  (house_cost : ℕ) (num_cows : ℕ) (num_chickens : ℕ)
  (chicken_cost : ℕ) (solar_install_hours : ℕ)
  (solar_install_rate : ℕ) (solar_equipment_cost : ℕ)
  (total_cost : ℕ) :
  land_acres = 30 →
  land_cost_per_acre = 20 →
  house_cost = 120000 →
  num_cows = 20 →
  num_chickens = 100 →
  chicken_cost = 5 →
  solar_install_hours = 6 →
  solar_install_rate = 100 →
  solar_equipment_cost = 6000 →
  total_cost = 147700 →
  (total_cost - (land_acres * land_cost_per_acre + house_cost + 
    num_chickens * chicken_cost + 
    solar_install_hours * solar_install_rate + solar_equipment_cost)) / num_cows = 1000 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_cow_l2264_226444


namespace NUMINAMATH_CALUDE_test_score_calculation_l2264_226451

theorem test_score_calculation (total_questions : Nat) (score : Int) 
  (h1 : total_questions = 100)
  (h2 : score = 61) :
  ∃ (correct : Nat),
    correct ≤ total_questions ∧ 
    (correct : Int) - 2 * (total_questions - correct) = score ∧ 
    correct = 87 := by
  sorry

end NUMINAMATH_CALUDE_test_score_calculation_l2264_226451


namespace NUMINAMATH_CALUDE_crayons_erasers_difference_l2264_226417

/-- Given the initial number of crayons and erasers, and the remaining number of crayons,
    prove that the difference between remaining crayons and erasers is 353. -/
theorem crayons_erasers_difference (initial_crayons : ℕ) (initial_erasers : ℕ) (remaining_crayons : ℕ) 
    (h1 : initial_crayons = 531)
    (h2 : initial_erasers = 38)
    (h3 : remaining_crayons = 391) : 
  remaining_crayons - initial_erasers = 353 := by
  sorry

end NUMINAMATH_CALUDE_crayons_erasers_difference_l2264_226417


namespace NUMINAMATH_CALUDE_cubic_integer_root_l2264_226483

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a cubic polynomial at a point -/
def CubicPolynomial.eval (P : CubicPolynomial) (x : ℤ) : ℤ :=
  P.a * x^3 + P.b * x^2 + P.c * x + P.d

/-- The property that xP(x) = yP(y) for infinitely many integer pairs (x,y) with x ≠ y -/
def InfinitelyManySolutions (P : CubicPolynomial) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ n < |x| ∧ n < |y| ∧ x * P.eval x = y * P.eval y

theorem cubic_integer_root (P : CubicPolynomial) 
    (h : InfinitelyManySolutions P) : 
    ∃ k : ℤ, P.eval k = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_integer_root_l2264_226483


namespace NUMINAMATH_CALUDE_three_fractions_inequality_l2264_226472

theorem three_fractions_inequality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_inequality : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), k > 0 ∧ x = 2*k ∧ y = k ∧ z = k) :=
by sorry

end NUMINAMATH_CALUDE_three_fractions_inequality_l2264_226472


namespace NUMINAMATH_CALUDE_stamps_ratio_after_gift_l2264_226450

/-- Proves that given the initial conditions, the new ratio of Kaye's stamps to Alberto's stamps is 4:3 -/
theorem stamps_ratio_after_gift (x : ℕ) 
  (h1 : 5 * x - 12 = 3 * x + 12 + 32) : 
  (5 * x - 12) / (3 * x + 12) = 4 / 3 := by
  sorry

#check stamps_ratio_after_gift

end NUMINAMATH_CALUDE_stamps_ratio_after_gift_l2264_226450


namespace NUMINAMATH_CALUDE_max_element_of_S_l2264_226447

def S : Set ℚ := {x | ∃ (p q : ℕ), x = p / q ∧ q ≤ 2009 ∧ x < 1257 / 2009}

theorem max_element_of_S :
  ∃ (p₀ q₀ : ℕ), 
    (p₀ : ℚ) / q₀ ∈ S ∧ 
    (∀ (x : ℚ), x ∈ S → x ≤ (p₀ : ℚ) / q₀) ∧
    (Nat.gcd p₀ q₀ = 1) ∧
    p₀ = 229 ∧ 
    q₀ = 366 ∧ 
    p₀ + q₀ = 595 := by
  sorry

end NUMINAMATH_CALUDE_max_element_of_S_l2264_226447


namespace NUMINAMATH_CALUDE_circle_representation_l2264_226425

theorem circle_representation (a : ℝ) :
  ∃ h k r, ∀ x y : ℝ,
    x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1 = 0 ↔
    (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_representation_l2264_226425


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l2264_226420

/-- A polynomial with real coefficients -/
def g (p q r s : ℝ) (x : ℂ) : ℂ := x^4 + p*x^3 + q*x^2 + r*x + s

/-- Theorem: If g(3i) = 0 and g(1+2i) = 0, then p + q + r + s = 39 -/
theorem polynomial_root_sum (p q r s : ℝ) : 
  g p q r s (3*I) = 0 → g p q r s (1 + 2*I) = 0 → p + q + r + s = 39 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l2264_226420


namespace NUMINAMATH_CALUDE_custard_pie_pieces_l2264_226422

/-- Proves that the number of pieces a custard pie is cut into is 6, given the conditions of the bakery problem. -/
theorem custard_pie_pieces : ℕ :=
  let pumpkin_pieces : ℕ := 8
  let pumpkin_price : ℕ := 5
  let custard_price : ℕ := 6
  let pumpkin_pies_sold : ℕ := 4
  let custard_pies_sold : ℕ := 5
  let total_revenue : ℕ := 340

  have h1 : pumpkin_pieces * pumpkin_price * pumpkin_pies_sold + custard_price * custard_pies_sold * custard_pie_pieces = total_revenue := by sorry

  custard_pie_pieces
where
  custard_pie_pieces : ℕ := 6

#check custard_pie_pieces

end NUMINAMATH_CALUDE_custard_pie_pieces_l2264_226422


namespace NUMINAMATH_CALUDE_percentage_problem_l2264_226469

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 36 → P = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2264_226469


namespace NUMINAMATH_CALUDE_youngest_child_age_exists_l2264_226437

/-- Represents the ages of the four children -/
structure ChildrenAges where
  twin_age : ℕ
  child1_age : ℕ
  child2_age : ℕ

/-- Calculates the total bill for the dinner -/
def calculate_bill (ages : ChildrenAges) : ℚ :=
  (30 * (25 : ℚ) / 100) + 
  ((2 * ages.twin_age + ages.child1_age + ages.child2_age) * (55 : ℚ) / 100)

theorem youngest_child_age_exists : ∃ (ages : ChildrenAges), 
  calculate_bill ages = 1510 / 100 ∧
  ages.twin_age ≠ ages.child1_age ∧
  ages.twin_age ≠ ages.child2_age ∧
  ages.child1_age ≠ ages.child2_age ∧
  min ages.twin_age (min ages.child1_age ages.child2_age) = 1 := by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_exists_l2264_226437


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_one_or_neg_one_third_l2264_226467

-- Define the two lines
def line1 (r : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (-2 + r, 5 - 3*k*r, k*r)
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (2*t, 2 + 2*t, -2*t)

-- Define coplanarity
def coplanar (l1 l2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b c d : ℝ), ∀ (t s : ℝ),
    a * (l1 t).1 + b * (l1 t).2.1 + c * (l1 t).2.2 + d =
    a * (l2 s).1 + b * (l2 s).2.1 + c * (l2 s).2.2 + d

-- Theorem statement
theorem lines_coplanar_iff_k_eq_neg_one_or_neg_one_third :
  ∀ k : ℝ, coplanar (line1 · k) line2 ↔ k = -1 ∨ k = -1/3 :=
sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_one_or_neg_one_third_l2264_226467


namespace NUMINAMATH_CALUDE_fourth_power_of_nested_root_l2264_226401

theorem fourth_power_of_nested_root : 
  let x := Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 4))
  x^4 = 9 + 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_of_nested_root_l2264_226401


namespace NUMINAMATH_CALUDE_cosine_value_for_given_point_l2264_226430

theorem cosine_value_for_given_point :
  ∀ α : Real,
  let P : Real × Real := (2 * Real.cos (120 * π / 180), Real.sqrt 2 * Real.sin (225 * π / 180))
  (Real.cos α = P.1 / Real.sqrt (P.1^2 + P.2^2) ∧
   Real.sin α = P.2 / Real.sqrt (P.1^2 + P.2^2)) →
  Real.cos α = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_for_given_point_l2264_226430


namespace NUMINAMATH_CALUDE_maria_initial_money_l2264_226461

/-- The amount of money Maria had when she left the fair -/
def money_left : ℕ := 16

/-- The difference between the amount of money Maria had when she got to the fair and when she left -/
def money_difference : ℕ := 71

/-- The amount of money Maria had when she got to the fair -/
def money_initial : ℕ := money_left + money_difference

theorem maria_initial_money : money_initial = 87 := by
  sorry

end NUMINAMATH_CALUDE_maria_initial_money_l2264_226461


namespace NUMINAMATH_CALUDE_theater_attendance_l2264_226494

theorem theater_attendance 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (total_attendance : ℕ) 
  (total_revenue : ℚ) 
  (h1 : adult_price = 60 / 100)
  (h2 : child_price = 25 / 100)
  (h3 : total_attendance = 280)
  (h4 : total_revenue = 140) :
  ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adult_price * adults + child_price * children = total_revenue ∧
    children = 80 := by
sorry

end NUMINAMATH_CALUDE_theater_attendance_l2264_226494


namespace NUMINAMATH_CALUDE_abcd_multiplication_l2264_226410

theorem abcd_multiplication (A B C D : ℕ) : 
  A < 10 → B < 10 → C < 10 → D < 10 →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (1000 * A + 100 * B + 10 * C + D) * 2 = 10000 * A + 1000 * B + 100 * C + 10 * D →
  A + B = 1 := by
sorry

end NUMINAMATH_CALUDE_abcd_multiplication_l2264_226410


namespace NUMINAMATH_CALUDE_min_value_theorem_l2264_226486

theorem min_value_theorem (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x + y = 2 → 1 / (x - 1) + 1 / y ≥ 1 / (a - 1) + 1 / b) ∧
  1 / (a - 1) + 1 / b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2264_226486


namespace NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l2264_226480

theorem conditional_probability_rain_given_east_wind 
  (p_east_wind : ℝ) 
  (p_east_wind_and_rain : ℝ) 
  (h1 : p_east_wind = 8/30) 
  (h2 : p_east_wind_and_rain = 7/30) : 
  p_east_wind_and_rain / p_east_wind = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l2264_226480


namespace NUMINAMATH_CALUDE_inverse_proportion_difference_positive_l2264_226490

theorem inverse_proportion_difference_positive 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : x₁ < x₂) 
  (h4 : x₂ < 0) : 
  y₁ - y₂ > 0 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_difference_positive_l2264_226490


namespace NUMINAMATH_CALUDE_tournament_score_sum_l2264_226474

/-- A round-robin tournament with three players -/
structure Tournament :=
  (players : Fin 3 → ℕ)

/-- The scoring system for the tournament -/
def score (result : ℕ) : ℕ :=
  match result with
  | 0 => 2  -- win
  | 1 => 1  -- draw
  | _ => 0  -- loss

/-- The theorem stating that the sum of all players' scores is always 6 -/
theorem tournament_score_sum (t : Tournament) : 
  (t.players 0) + (t.players 1) + (t.players 2) = 6 :=
sorry

end NUMINAMATH_CALUDE_tournament_score_sum_l2264_226474


namespace NUMINAMATH_CALUDE_min_value_product_l2264_226460

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x + 3 * y) * (y + 3 * z) * (x * z + 2) ≥ 96 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 3 * y₀) * (y₀ + 3 * z₀) * (x₀ * z₀ + 2) = 96 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l2264_226460


namespace NUMINAMATH_CALUDE_volume_between_concentric_spheres_l2264_226476

theorem volume_between_concentric_spheres :
  let r₁ : ℝ := 4  -- radius of smaller sphere
  let r₂ : ℝ := 7  -- radius of larger sphere
  let V : ℝ := (4 / 3) * Real.pi * (r₂^3 - r₁^3)  -- volume between spheres
  V = 372 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_volume_between_concentric_spheres_l2264_226476


namespace NUMINAMATH_CALUDE_ginas_college_expenses_l2264_226405

/-- Calculates the total college expenses for Gina -/
def total_college_expenses (credits : ℕ) (cost_per_credit : ℕ) (num_textbooks : ℕ) (cost_per_textbook : ℕ) (facilities_fee : ℕ) : ℕ :=
  credits * cost_per_credit + num_textbooks * cost_per_textbook + facilities_fee

/-- Proves that Gina's total college expenses are $7100 -/
theorem ginas_college_expenses :
  total_college_expenses 14 450 5 120 200 = 7100 := by
  sorry

end NUMINAMATH_CALUDE_ginas_college_expenses_l2264_226405


namespace NUMINAMATH_CALUDE_complex_problem_l2264_226499

def complex_equation (z : ℂ) : Prop := (1 + 2*Complex.I) * z = 3 - 4*Complex.I

def third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_problem (z : ℂ) (h : complex_equation z) :
  z = -1 - 2*Complex.I ∧ third_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_complex_problem_l2264_226499


namespace NUMINAMATH_CALUDE_cricket_bat_profit_l2264_226468

/-- Calculates the profit amount for a cricket bat sale -/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 850 ∧ profit_percentage = 36 →
  (selling_price - selling_price / (1 + profit_percentage / 100)) = 225 := by
sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_l2264_226468


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2264_226478

theorem simplify_and_evaluate :
  let x : ℚ := -1
  let y : ℚ := 1
  let expr := (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y)
  expr = -x^2 + 3*y^2 ∧ expr = 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2264_226478


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2264_226434

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - I) / (1 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2264_226434


namespace NUMINAMATH_CALUDE_particle_probability_theorem_l2264_226496

/-- Probability of hitting (0,0) first when starting from (x,y) -/
noncomputable def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

theorem particle_probability_theorem :
  ∃ (p : ℕ), p > 0 ∧ ¬(3 ∣ p) ∧ P 3 5 = p / 3^7 := by sorry

end NUMINAMATH_CALUDE_particle_probability_theorem_l2264_226496


namespace NUMINAMATH_CALUDE_currency_exchange_problem_l2264_226440

def exchange_rate : ℚ := 9 / 6

def spent_amount : ℕ := 45

theorem currency_exchange_problem (d : ℕ) :
  (d : ℚ) * exchange_rate - spent_amount = d →
  (d / 10 + d % 10 : ℕ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_currency_exchange_problem_l2264_226440


namespace NUMINAMATH_CALUDE_triangle_ratio_l2264_226458

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points N, D, E, F
variable (N D E F : ℝ × ℝ)

-- Define the conditions
variable (h1 : N = ((A.1 + C.1)/2, (A.2 + C.2)/2))  -- N is midpoint of AC
variable (h2 : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 100) -- AB = 10
variable (h3 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 324) -- BC = 18
variable (h4 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1-t) * C.1, t * B.2 + (1-t) * C.2)) -- D on BC
variable (h5 : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (s * A.1 + (1-s) * B.1, s * A.2 + (1-s) * B.2)) -- E on AB
variable (h6 : ∃ r u : ℝ, F = (r * D.1 + (1-r) * E.1, r * D.2 + (1-r) * E.2) ∧
                          F = (u * A.1 + (1-u) * N.1, u * A.2 + (1-u) * N.2)) -- F is intersection of DE and AN
variable (h7 : (D.1 - B.1)^2 + (D.2 - B.2)^2 = 9 * ((E.1 - B.1)^2 + (E.2 - B.2)^2)) -- BD = 3BE

-- Theorem statement
theorem triangle_ratio :
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 1/9 * ((F.1 - E.1)^2 + (F.2 - E.2)^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l2264_226458


namespace NUMINAMATH_CALUDE_smallest_n_for_factors_l2264_226452

theorem smallest_n_for_factors (k : ℕ) : 
  (∀ m : ℕ, m > 0 → (5^2 ∣ m * 2^k * 6^2 * 7^3) → (3^3 ∣ m * 2^k * 6^2 * 7^3) → m ≥ 75) ∧
  (5^2 ∣ 75 * 2^k * 6^2 * 7^3) ∧
  (3^3 ∣ 75 * 2^k * 6^2 * 7^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_factors_l2264_226452


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_value_l2264_226408

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y = 0

-- Theorem statement
theorem hyperbola_asymptote_implies_a_value :
  ∀ a : ℝ, (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) →
  a = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_value_l2264_226408


namespace NUMINAMATH_CALUDE_negative_difference_l2264_226446

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l2264_226446


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2264_226416

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + 2*x - 3 < 0 ↔ -3 < x ∧ x < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2264_226416


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2264_226491

theorem least_positive_integer_congruence : ∃! x : ℕ+, 
  (x : ℤ) + 3701 ≡ 1580 [ZMOD 15] ∧ 
  (x : ℤ) ≡ 7 [ZMOD 9] ∧
  ∀ y : ℕ+, ((y : ℤ) + 3701 ≡ 1580 [ZMOD 15] ∧ (y : ℤ) ≡ 7 [ZMOD 9]) → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2264_226491


namespace NUMINAMATH_CALUDE_discount_percentage_is_correct_l2264_226436

/-- Calculates the discount percentage given the purchase price, selling price for 10% profit, and additional costs --/
def calculate_discount_percentage (purchase_price selling_price_for_profit transport_cost installation_cost : ℚ) : ℚ :=
  let labelled_price := selling_price_for_profit / 1.1
  let discount_amount := labelled_price - purchase_price
  (discount_amount / labelled_price) * 100

/-- Theorem stating that the discount percentage is equal to (500/23)% given the problem conditions --/
theorem discount_percentage_is_correct :
  let purchase_price : ℚ := 13500
  let selling_price_for_profit : ℚ := 18975
  let transport_cost : ℚ := 125
  let installation_cost : ℚ := 250
  calculate_discount_percentage purchase_price selling_price_for_profit transport_cost installation_cost = 500 / 23 := by
  sorry

#eval calculate_discount_percentage 13500 18975 125 250

end NUMINAMATH_CALUDE_discount_percentage_is_correct_l2264_226436


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2264_226473

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a = 5 ∧ b = 12) ∨ (a = 5 ∧ c = 12) ∨ (b = 5 ∧ c = 12) →
  a^2 + b^2 = c^2 →
  c = 12 ∨ c = 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2264_226473


namespace NUMINAMATH_CALUDE_space_divided_by_five_spheres_l2264_226484

/-- Maximum number of regions a sphere can be divided by n circles -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => a (n + 1) + 2 * (n + 1)

/-- Maximum number of regions space can be divided by n spheres -/
def b : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => b (n + 1) + a (n + 1)

theorem space_divided_by_five_spheres :
  b 5 = 22 := by sorry

end NUMINAMATH_CALUDE_space_divided_by_five_spheres_l2264_226484


namespace NUMINAMATH_CALUDE_difference_of_squares_l2264_226403

theorem difference_of_squares (m : ℝ) : m^2 - 16 = (m + 4) * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2264_226403


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_five_l2264_226455

theorem fraction_zero_implies_x_negative_five (x : ℝ) :
  (x + 5) / (x - 2) = 0 → x = -5 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_five_l2264_226455


namespace NUMINAMATH_CALUDE_f_of_three_equals_six_l2264_226443

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f(3) = 6 -/
theorem f_of_three_equals_six (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : f 1 = 4)
  (h2 : f 2 = 9)
  (h3 : ∀ x, f x = a * x + b * x + 3) :
  f 3 = 6 := by
sorry

end NUMINAMATH_CALUDE_f_of_three_equals_six_l2264_226443


namespace NUMINAMATH_CALUDE_trapezoid_triangle_area_l2264_226439

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  -- Area of the trapezoid
  area : ℝ
  -- Condition that one base is twice the other
  base_ratio : Bool
  -- Point of intersection of diagonals
  O : Point
  -- Midpoint of base AD
  P : Point
  -- Points where BP and CP intersect the diagonals
  M : Point
  N : Point

/-- The area of triangle MON in a trapezoid with specific properties -/
def area_MON (t : Trapezoid) : Set ℝ :=
  {45/4, 36/5}

/-- Theorem stating the area of triangle MON in a trapezoid with given properties -/
theorem trapezoid_triangle_area (t : Trapezoid) 
  (h1 : t.area = 405) : 
  (area_MON t).Nonempty ∧ (∀ x ∈ area_MON t, x = 45/4 ∨ x = 36/5) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_triangle_area_l2264_226439


namespace NUMINAMATH_CALUDE_point_two_units_from_negative_one_l2264_226400

theorem point_two_units_from_negative_one (x : ℝ) : 
  (|x - (-1)| = 2) ↔ (x = -3 ∨ x = 1) := by sorry

end NUMINAMATH_CALUDE_point_two_units_from_negative_one_l2264_226400


namespace NUMINAMATH_CALUDE_green_ribbons_count_l2264_226475

theorem green_ribbons_count (total : ℕ) 
  (h_red : (1 : ℚ) / 4 * total = total / 4)
  (h_blue : (3 : ℚ) / 8 * total = 3 * total / 8)
  (h_green : (1 : ℚ) / 8 * total = total / 8)
  (h_white : total - (total / 4 + 3 * total / 8 + total / 8) = 36) :
  total / 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_green_ribbons_count_l2264_226475


namespace NUMINAMATH_CALUDE_correct_propositions_l2264_226449

/-- A structure representing a plane with lines -/
structure Plane where
  /-- The type of lines in the plane -/
  Line : Type
  /-- Perpendicularity relation between lines -/
  perp : Line → Line → Prop
  /-- Parallelism relation between lines -/
  parallel : Line → Line → Prop

/-- The main theorem stating the two correct propositions -/
theorem correct_propositions (P : Plane) 
  (a b c α β γ : P.Line) : 
  (P.perp a α ∧ P.perp b β ∧ P.perp α β → P.perp a b) ∧
  (P.parallel α β ∧ P.parallel β γ ∧ P.perp a α → P.perp a γ) := by
  sorry


end NUMINAMATH_CALUDE_correct_propositions_l2264_226449


namespace NUMINAMATH_CALUDE_zinc_copper_ratio_theorem_l2264_226413

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents a mixture of zinc and copper -/
structure Mixture where
  total_weight : ℝ
  zinc_weight : ℝ

/-- Calculates the ratio of zinc to copper in a mixture -/
def zinc_copper_ratio (m : Mixture) : Ratio :=
  sorry

/-- The given mixture of zinc and copper -/
def given_mixture : Mixture :=
  { total_weight := 74
    zinc_weight := 33.3 }

/-- Theorem stating the correct ratio of zinc to copper in the given mixture -/
theorem zinc_copper_ratio_theorem :
  zinc_copper_ratio given_mixture = Ratio.mk 333 407 :=
  sorry

end NUMINAMATH_CALUDE_zinc_copper_ratio_theorem_l2264_226413


namespace NUMINAMATH_CALUDE_expression_evaluation_l2264_226453

theorem expression_evaluation : (2000^2 : ℝ) / (402^2 - 398^2) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2264_226453


namespace NUMINAMATH_CALUDE_cd_price_difference_l2264_226487

theorem cd_price_difference (album_price book_price : ℝ) (h1 : album_price = 20) (h2 : book_price = 18) : 
  let cd_price := book_price - 4
  (album_price - cd_price) / album_price * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_cd_price_difference_l2264_226487


namespace NUMINAMATH_CALUDE_intersection_point_l2264_226493

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := (x^2 - 8*x + 12) / (2*x - 6)
def g (x b c d e : ℝ) : ℝ := (b*x^2 + c*x + d) / (x - e)

-- State the theorem
theorem intersection_point (b c d e : ℝ) :
  -- Conditions
  (∀ x, (2*x - 6 = 0 ↔ x - e = 0)) →  -- Same vertical asymptote
  (∃ k, ∀ x, g x b c d e = -2*x - 4 + k / (x - e)) →  -- Oblique asymptote of g
  (f (-3) = g (-3) b c d e) →  -- Intersection at x = -3
  -- Conclusion
  (∃ x y, x ≠ -3 ∧ f x = g x b c d e ∧ x = 14 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l2264_226493


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l2264_226428

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_non_factor_product (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  is_factor a 48 → 
  is_factor b 48 → 
  ¬(is_factor (a * b) 48) → 
  (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → is_factor x 48 → is_factor y 48 → 
    ¬(is_factor (x * y) 48) → a * b ≤ x * y) → 
  a * b = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l2264_226428


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2264_226412

theorem p_necessary_not_sufficient_for_q : 
  (∀ x : ℝ, |x - 1| < 2 → x + 1 ≥ 0) ∧ 
  (∃ x : ℝ, x + 1 ≥ 0 ∧ |x - 1| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2264_226412


namespace NUMINAMATH_CALUDE_polygon_diagonals_l2264_226498

/-- A polygon with interior angle sum of 1800° has 9 diagonals from any vertex -/
theorem polygon_diagonals (n : ℕ) : 
  (n - 2) * 180 = 1800 → n - 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l2264_226498


namespace NUMINAMATH_CALUDE_sum_10_is_negative_15_l2264_226404

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * a 1 + (n * (n - 1) / 2 : ℝ) * (a 2 - a 1)
  S_3 : S 3 = 6
  S_6 : S 6 = 3

/-- The sum of the first 10 terms is -15 -/
theorem sum_10_is_negative_15 (seq : ArithmeticSequence) : seq.S 10 = -15 := by
  sorry

end NUMINAMATH_CALUDE_sum_10_is_negative_15_l2264_226404


namespace NUMINAMATH_CALUDE_company_female_managers_l2264_226489

/-- Represents the number of female managers in a company -/
def female_managers (total_employees : ℕ) (female_employees : ℕ) (male_employees : ℕ) : ℕ :=
  (2 * female_employees) / 5

theorem company_female_managers :
  let total_employees := female_employees + male_employees
  let female_employees := 625
  let total_managers := (2 * total_employees) / 5
  let male_managers := (2 * male_employees) / 5
  female_managers total_employees female_employees male_employees = 250 :=
by
  sorry

#check company_female_managers

end NUMINAMATH_CALUDE_company_female_managers_l2264_226489


namespace NUMINAMATH_CALUDE_friend_payment_ratio_l2264_226432

def james_meal : ℚ := 16
def friend_meal : ℚ := 14
def tip_percentage : ℚ := 20 / 100
def james_total_paid : ℚ := 21

def total_bill : ℚ := james_meal + friend_meal
def tip : ℚ := total_bill * tip_percentage
def total_bill_with_tip : ℚ := total_bill + tip
def james_share : ℚ := james_total_paid - tip
def friend_payment : ℚ := total_bill - james_share

theorem friend_payment_ratio :
  friend_payment / total_bill_with_tip = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_friend_payment_ratio_l2264_226432


namespace NUMINAMATH_CALUDE_annes_bowling_score_l2264_226431

theorem annes_bowling_score (annes_score bob_score : ℕ) : 
  annes_score = bob_score + 50 →
  (annes_score + bob_score) / 2 = 150 →
  annes_score = 175 := by
sorry

end NUMINAMATH_CALUDE_annes_bowling_score_l2264_226431


namespace NUMINAMATH_CALUDE_seating_arrangements_5_total_arrangements_l2264_226423

/-- Defines the number of seating arrangements for n people -/
def seating_arrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => seating_arrangements (n + 1) + seating_arrangements n

/-- Theorem stating that the number of seating arrangements for 5 people is 8 -/
theorem seating_arrangements_5 : seating_arrangements 5 = 8 := by sorry

/-- Theorem stating that the total number of arrangements for two independent groups of 5 is 64 -/
theorem total_arrangements : seating_arrangements 5 * seating_arrangements 5 = 64 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_5_total_arrangements_l2264_226423


namespace NUMINAMATH_CALUDE_tangent_and_inequality_imply_m_range_l2264_226429

open Real

noncomputable def f (x : ℝ) : ℝ := x / (Real.exp x)

theorem tangent_and_inequality_imply_m_range :
  (∀ x ∈ Set.Ioo (1/2) (3/2), f x < 1 / (m + 6*x - 3*x^2)) →
  m ∈ Set.Icc (-9/4) (ℯ - 3) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_inequality_imply_m_range_l2264_226429


namespace NUMINAMATH_CALUDE_smaller_integer_problem_l2264_226456

theorem smaller_integer_problem (a b : ℕ+) : 
  (a : ℕ) + 8 = (b : ℕ) → a * b = 80 → (a : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_integer_problem_l2264_226456


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2264_226459

/-- Given a line 3x - 4y + k = 0, if the sum of its x-intercept and y-intercept is 2, then k = -24 -/
theorem line_intercepts_sum (k : ℝ) : 
  (∃ x y : ℝ, 3*x - 4*y + k = 0 ∧ x + y = 2) → k = -24 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2264_226459


namespace NUMINAMATH_CALUDE_optimal_garden_dimensions_l2264_226433

/-- Represents a rectangular garden with one side along a house wall. -/
structure Garden where
  width : ℝ  -- Width of the garden (perpendicular to the house)
  length : ℝ  -- Length of the garden (parallel to the house)

/-- Calculates the area of a rectangular garden. -/
def Garden.area (g : Garden) : ℝ := g.width * g.length

/-- Calculates the cost of fencing for three sides of the garden. -/
def Garden.fenceCost (g : Garden) : ℝ := 10 * (g.length + 2 * g.width)

/-- Theorem stating the optimal dimensions of the garden. -/
theorem optimal_garden_dimensions (houseLength : ℝ) (totalFenceCost : ℝ) :
  houseLength = 300 → totalFenceCost = 2000 →
  ∃ (g : Garden),
    g.fenceCost = totalFenceCost ∧
    g.length = 100 ∧
    ∀ (g' : Garden), g'.fenceCost = totalFenceCost → g.area ≥ g'.area :=
sorry

end NUMINAMATH_CALUDE_optimal_garden_dimensions_l2264_226433


namespace NUMINAMATH_CALUDE_max_non_managers_l2264_226418

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 9 →
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 41 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l2264_226418


namespace NUMINAMATH_CALUDE_stability_comparison_l2264_226495

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines when one athlete's performance is more stable than another's -/
def more_stable (a b : Athlete) : Prop :=
  a.average_score = b.average_score ∧ a.variance < b.variance

theorem stability_comparison (a b : Athlete) 
  (h : a.variance < b.variance) (h_avg : a.average_score = b.average_score) : 
  more_stable a b := by
  sorry

#check stability_comparison

end NUMINAMATH_CALUDE_stability_comparison_l2264_226495


namespace NUMINAMATH_CALUDE_speech_competition_arrangements_l2264_226415

/-- The number of students in the speech competition -/
def num_students : ℕ := 6

/-- The total number of arrangements where B and C are adjacent -/
def total_arrangements_bc_adjacent : ℕ := 240

/-- The number of arrangements where A is first or last, and B and C are adjacent -/
def arrangements_a_first_or_last : ℕ := 96

/-- The number of valid arrangements for the speech competition -/
def valid_arrangements : ℕ := total_arrangements_bc_adjacent - arrangements_a_first_or_last

theorem speech_competition_arrangements :
  valid_arrangements = 144 :=
sorry

end NUMINAMATH_CALUDE_speech_competition_arrangements_l2264_226415


namespace NUMINAMATH_CALUDE_employee_age_when_hired_l2264_226477

theorem employee_age_when_hired (age_when_hired : ℕ) (years_worked : ℕ) : 
  age_when_hired + years_worked = 70 →
  years_worked = 19 →
  age_when_hired = 51 := by
  sorry

end NUMINAMATH_CALUDE_employee_age_when_hired_l2264_226477


namespace NUMINAMATH_CALUDE_paper_towel_savings_l2264_226426

/-- Calculates the percent savings per roll when buying a package of rolls compared to individual rolls -/
def percent_savings_per_roll (package_price : ℚ) (package_size : ℕ) (individual_price : ℚ) : ℚ :=
  let package_price_per_roll := package_price / package_size
  let savings_per_roll := individual_price - package_price_per_roll
  (savings_per_roll / individual_price) * 100

/-- Theorem: The percent savings per roll for a 12-roll package priced at $9 compared to
    buying 12 rolls individually at $1 each is 25% -/
theorem paper_towel_savings :
  percent_savings_per_roll 9 12 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_paper_towel_savings_l2264_226426


namespace NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l2264_226479

-- System 1
theorem system_1_solution :
  ∃ (x y : ℝ), 2 * x + 3 * y = 9 ∧ x = 2 * y + 1 ∧ x = 3 ∧ y = 1 := by sorry

-- System 2
theorem system_2_solution :
  ∃ (x y : ℝ), 2 * x - y = 6 ∧ 3 * x + 2 * y = 2 ∧ x = 2 ∧ y = -2 := by sorry

end NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l2264_226479


namespace NUMINAMATH_CALUDE_nonSimilar500PointedStars_l2264_226466

/-- The number of non-similar regular n-pointed stars -/
def nonSimilarStars (n : ℕ) : ℕ :=
  (n.totient - 2) / 2

/-- Theorem: The number of non-similar regular 500-pointed stars is 99 -/
theorem nonSimilar500PointedStars : nonSimilarStars 500 = 99 := by
  sorry

#eval nonSimilarStars 500  -- This should evaluate to 99

end NUMINAMATH_CALUDE_nonSimilar500PointedStars_l2264_226466


namespace NUMINAMATH_CALUDE_geometric_region_equivalence_l2264_226457

theorem geometric_region_equivalence (x y : ℝ) :
  (x^2 + y^2 - 4 ≥ 0 ∧ x^2 - 1 ≥ 0 ∧ y^2 - 1 ≥ 0) ↔
  ((x^2 + y^2 ≥ 4) ∧ (x ≤ -1 ∨ x ≥ 1) ∧ (y ≤ -1 ∨ y ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_region_equivalence_l2264_226457


namespace NUMINAMATH_CALUDE_kevins_age_l2264_226463

theorem kevins_age (vanessa_age : ℕ) (future_years : ℕ) (ratio : ℕ) :
  vanessa_age = 2 →
  future_years = 5 →
  ratio = 3 →
  ∃ kevin_age : ℕ, kevin_age + future_years = ratio * (vanessa_age + future_years) ∧ kevin_age = 16 :=
by sorry

end NUMINAMATH_CALUDE_kevins_age_l2264_226463


namespace NUMINAMATH_CALUDE_evaluate_expression_l2264_226427

theorem evaluate_expression (x y z w : ℚ) 
  (hx : x = 1/4)
  (hy : y = 1/3)
  (hz : z = -2)
  (hw : w = 3) :
  x^3 * y^2 * z^2 * w = 1/48 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2264_226427


namespace NUMINAMATH_CALUDE_custom_op_solution_l2264_226454

/-- Custom operation for integers -/
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if y12 = 110 using the custom operation, then y = 11 -/
theorem custom_op_solution :
  ∀ y : ℤ, customOp y 12 = 110 → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l2264_226454


namespace NUMINAMATH_CALUDE_handshake_problem_l2264_226406

theorem handshake_problem (a b : ℕ) : 
  a + b = 20 →
  (a * (a - 1)) / 2 + (b * (b - 1)) / 2 = 106 →
  a * b = 84 := by
sorry

end NUMINAMATH_CALUDE_handshake_problem_l2264_226406


namespace NUMINAMATH_CALUDE_total_books_l2264_226448

theorem total_books (x : ℚ) : ℚ := by
  -- Betty's books
  let betty_books := x

  -- Sister's books: x + (1/4)x
  let sister_books := x + (1/4) * x

  -- Cousin's books: 2 * (x + (1/4)x)
  let cousin_books := 2 * (x + (1/4) * x)

  -- Total books
  let total := betty_books + sister_books + cousin_books

  -- Prove that total = (19/4)x
  sorry

end NUMINAMATH_CALUDE_total_books_l2264_226448


namespace NUMINAMATH_CALUDE_linear_independence_of_polynomial_basis_l2264_226445

theorem linear_independence_of_polynomial_basis :
  ∀ (α₁ α₂ α₃ α₄ : ℝ),
  (∀ x : ℝ, α₁ + α₂ * x + α₃ * x^2 + α₄ * x^3 = 0) →
  (α₁ = 0 ∧ α₂ = 0 ∧ α₃ = 0 ∧ α₄ = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_independence_of_polynomial_basis_l2264_226445


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2264_226482

/-- Two arithmetic sequences and their partial sums -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n / T n = (7 * n + 2) / (n + 3)

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) (S T : ℕ → ℚ) 
  (h : arithmetic_sequences a b S T) : 
  (a 2 + a 20) / (b 7 + b 15) = 149 / 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2264_226482


namespace NUMINAMATH_CALUDE_division_problem_l2264_226462

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 161)
  (h2 : quotient = 10)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 16 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2264_226462


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2264_226441

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem sufficient_not_necessary
  (f : ℝ → ℝ) (h : OddFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧
  (∃ x₁ x₂ : ℝ, f x₁ + f x₂ = 0 ∧ x₁ + x₂ ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2264_226441


namespace NUMINAMATH_CALUDE_circle_equation_with_radius_3_l2264_226411

theorem circle_equation_with_radius_3 (c : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + c = 0 ↔ (x - 4)^2 + (y + 5)^2 = 3^2) → 
  c = 32 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_with_radius_3_l2264_226411


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2264_226419

theorem max_gcd_13n_plus_4_8n_plus_3 (n : ℕ+) : 
  (Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 7) ∧ 
  (∃ m : ℕ+, Nat.gcd (13 * m + 4) (8 * m + 3) = 7) := by
sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2264_226419


namespace NUMINAMATH_CALUDE_sticker_collection_total_l2264_226464

/-- The number of stickers Karl has -/
def karl_stickers : ℕ := 25

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := karl_stickers + 20

/-- The number of stickers Ben has -/
def ben_stickers : ℕ := ryan_stickers - 10

/-- The total number of stickers placed in the book -/
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem sticker_collection_total :
  total_stickers = 105 := by sorry

end NUMINAMATH_CALUDE_sticker_collection_total_l2264_226464
