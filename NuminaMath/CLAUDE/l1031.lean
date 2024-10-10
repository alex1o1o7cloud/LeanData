import Mathlib

namespace geometric_sequence_sum_l1031_103157

/-- Given a geometric sequence with sum of first n terms Sn = k · 3^n + 1, k = -1 -/
theorem geometric_sequence_sum (n : ℕ) (k : ℝ) :
  (∀ n, ∃ Sn : ℝ, Sn = k * 3^n + 1) →
  (∃ a : ℕ → ℝ, ∀ i j, i < j → a i * a j = (a i)^2) →
  k = -1 :=
sorry

end geometric_sequence_sum_l1031_103157


namespace focal_length_of_hyperbola_C_l1031_103159

-- Define the hyperbola C
def hyperbola_C (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

-- Define the asymptote of C
def asymptote_C (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

-- State the theorem
theorem focal_length_of_hyperbola_C (m : ℝ) (h1 : m > 0) :
  (∀ x y : ℝ, hyperbola_C m x y ↔ asymptote_C m x y) →
  2 * Real.sqrt (m + m) = 4 := by sorry

end focal_length_of_hyperbola_C_l1031_103159


namespace sector_perimeter_and_area_l1031_103125

/-- Given a circular sector with radius 6 cm and central angle π/4 radians,
    prove that its perimeter is 12 + 3π/2 cm and its area is 9π/2 cm². -/
theorem sector_perimeter_and_area :
  let r : ℝ := 6
  let θ : ℝ := π / 4
  let perimeter : ℝ := 2 * r + r * θ
  let area : ℝ := (1 / 2) * r^2 * θ
  perimeter = 12 + 3 * π / 2 ∧ area = 9 * π / 2 := by
  sorry


end sector_perimeter_and_area_l1031_103125


namespace percentage_of_360_is_120_l1031_103151

theorem percentage_of_360_is_120 : 
  (120 : ℝ) / 360 * 100 = 100 / 3 :=
sorry

end percentage_of_360_is_120_l1031_103151


namespace sum_of_specific_terms_l1031_103123

theorem sum_of_specific_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2 + 2*n) :
  a 3 + a 4 + a 5 + a 6 = 40 := by
  sorry

end sum_of_specific_terms_l1031_103123


namespace bus_capacity_problem_l1031_103141

/-- 
Given a bus with capacity 200 people, prove that if it carries x fraction of its capacity 
on the first trip and 4/5 of its capacity on the return trip, and the total number of 
people on both trips is 310, then x = 3/4.
-/
theorem bus_capacity_problem (x : ℚ) : 
  (200 * x + 200 * (4/5) = 310) → x = 3/4 := by
  sorry

end bus_capacity_problem_l1031_103141


namespace complex_equality_implies_sum_zero_l1031_103156

theorem complex_equality_implies_sum_zero (z : ℂ) (x y : ℝ) :
  Complex.abs (z + 1) = Complex.abs (z - Complex.I) →
  z = Complex.mk x y →
  x + y = 0 := by
sorry

end complex_equality_implies_sum_zero_l1031_103156


namespace smallest_d_is_four_l1031_103191

def is_valid_pair (c d : ℕ+) : Prop :=
  (c : ℤ) - (d : ℤ) = 8 ∧ 
  Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16

theorem smallest_d_is_four :
  ∀ c d : ℕ+, is_valid_pair c d → d ≥ 4 ∧ ∃ c' : ℕ+, is_valid_pair c' 4 :=
by sorry

end smallest_d_is_four_l1031_103191


namespace thursday_miles_proof_l1031_103146

/-- The number of miles flown on Tuesday each week -/
def tuesday_miles : ℕ := 1134

/-- The total number of miles flown over 3 weeks -/
def total_miles : ℕ := 7827

/-- The number of weeks the pilot flies -/
def num_weeks : ℕ := 3

/-- The number of miles flown on Thursday each week -/
def thursday_miles : ℕ := (total_miles - num_weeks * tuesday_miles) / num_weeks

theorem thursday_miles_proof :
  thursday_miles = 1475 :=
by sorry

end thursday_miles_proof_l1031_103146


namespace expression_simplification_and_evaluation_l1031_103137

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ,
    5 - 2*x ≥ 1 →
    x + 3 > 0 →
    x + 1 ≠ 0 →
    (2 + x) * (2 - x) ≠ 0 →
    (x^2 - 4*x + 4) / (x + 1) / ((3 / (x + 1)) - x + 1) = (2 - x) / (2 + x) ∧
    (2 - 0) / (2 + 0) = 1 :=
by sorry

end expression_simplification_and_evaluation_l1031_103137


namespace average_multiples_of_10_l1031_103154

/-- The average of multiples of 10 from 10 to 500 inclusive is 255 -/
theorem average_multiples_of_10 : 
  let first := 10
  let last := 500
  let step := 10
  (first + last) / 2 = 255 := by sorry

end average_multiples_of_10_l1031_103154


namespace fixed_points_of_f_composition_l1031_103189

def f (x : ℝ) : ℝ := x^2 - 4*x

theorem fixed_points_of_f_composition (x : ℝ) : 
  f (f x) = f x ↔ x ∈ ({-1, 0, 4, 5} : Set ℝ) :=
sorry

end fixed_points_of_f_composition_l1031_103189


namespace no_rational_roots_l1031_103149

theorem no_rational_roots (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  ∀ (x : ℚ), x^2 + 2*p*x + 2*q ≠ 0 := by
  sorry

end no_rational_roots_l1031_103149


namespace claire_gift_card_balance_l1031_103161

/-- Calculates the remaining balance on Claire's gift card after a week of purchases. -/
def remaining_balance (gift_card_value : ℚ) (latte_cost : ℚ) (croissant_cost : ℚ) 
  (days : ℕ) (cookie_cost : ℚ) (num_cookies : ℕ) : ℚ :=
  gift_card_value - 
  ((latte_cost + croissant_cost) * days + cookie_cost * num_cookies)

/-- Proves that Claire will have $43.00 left on her gift card after a week of purchases. -/
theorem claire_gift_card_balance : 
  remaining_balance 100 3.75 3.50 7 1.25 5 = 43 := by
  sorry

end claire_gift_card_balance_l1031_103161


namespace no_special_multiple_l1031_103120

/-- Calculates the digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- Generates a repunit with m ones -/
def repunit (m : ℕ) : ℕ :=
  if m = 0 then 0 else (10^m - 1) / 9

/-- The main theorem -/
theorem no_special_multiple :
  ¬ ∃ (n m : ℕ), 
    (∃ k : ℕ, n = k * (10 * 94)) ∧
    (n % repunit m = 0) ∧
    (digit_sum n < m) :=
sorry

end no_special_multiple_l1031_103120


namespace routes_2x2_grid_proof_l1031_103158

/-- The number of routes on a 2x2 grid from top-left to bottom-right -/
def routes_2x2_grid : ℕ := 6

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem routes_2x2_grid_proof :
  routes_2x2_grid = choose 4 2 :=
by sorry

end routes_2x2_grid_proof_l1031_103158


namespace expected_pine_saplings_in_sample_l1031_103194

/-- Given a forestry farm with the following characteristics:
  * total_saplings: The total number of saplings
  * pine_saplings: The number of pine saplings
  * sample_size: The size of the sample to be drawn
  
  This theorem proves that the expected number of pine saplings in the sample
  is equal to (pine_saplings / total_saplings) * sample_size. -/
theorem expected_pine_saplings_in_sample
  (total_saplings : ℕ)
  (pine_saplings : ℕ)
  (sample_size : ℕ)
  (h1 : total_saplings = 3000)
  (h2 : pine_saplings = 400)
  (h3 : sample_size = 150)
  : (pine_saplings : ℚ) / total_saplings * sample_size = 20 := by
  sorry

end expected_pine_saplings_in_sample_l1031_103194


namespace range_of_a_l1031_103193

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (a * x^2 - x + (1/4) * a)
def q (a : ℝ) : Prop := ∀ x > 0, 3^x - 9^x < a

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → 0 ≤ a ∧ a ≤ 1 :=
by sorry

end range_of_a_l1031_103193


namespace slope_of_line_l1031_103185

theorem slope_of_line (x y : ℝ) : 4 * y = -6 * x + 12 → (y - 3) = (-3/2) * (x - 0) := by
  sorry

end slope_of_line_l1031_103185


namespace thomas_order_total_correct_l1031_103172

/-- Calculates the total bill for an international order including shipping and import taxes -/
def calculate_total_bill (clothes_cost accessories_cost : ℝ)
  (clothes_shipping_rate accessories_shipping_rate : ℝ)
  (clothes_tax_rate accessories_tax_rate : ℝ) : ℝ :=
  let clothes_shipping := clothes_cost * clothes_shipping_rate
  let accessories_shipping := accessories_cost * accessories_shipping_rate
  let clothes_tax := clothes_cost * clothes_tax_rate
  let accessories_tax := accessories_cost * accessories_tax_rate
  clothes_cost + accessories_cost + clothes_shipping + accessories_shipping + clothes_tax + accessories_tax

/-- Thomas's international order total matches the calculated amount -/
theorem thomas_order_total_correct :
  calculate_total_bill 85 36 0.3 0.15 0.1 0.05 = 162.20 := by
  sorry

end thomas_order_total_correct_l1031_103172


namespace train_speed_train_speed_problem_l1031_103168

/-- The speed of a train given its length, time to pass a person, and the person's speed in the opposite direction. -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph

/-- Proof that a train 165 meters long passing a man in 9 seconds, with the man running at 6 kmph in the opposite direction, has a speed of 60 kmph. -/
theorem train_speed_problem : train_speed 165 9 6 = 60 := by
  sorry

end train_speed_train_speed_problem_l1031_103168


namespace min_bailing_rate_solution_l1031_103171

/-- Represents the problem of determining the minimum bailing rate for a boat --/
def MinBailingRateProblem (distance_to_shore : ℝ) (rowing_speed : ℝ) (water_intake_rate : ℝ) (max_water_capacity : ℝ) : Prop :=
  let time_to_shore : ℝ := distance_to_shore / rowing_speed
  let total_water_intake : ℝ := water_intake_rate * time_to_shore
  let excess_water : ℝ := total_water_intake - max_water_capacity
  let min_bailing_rate : ℝ := excess_water / time_to_shore
  min_bailing_rate = 2

/-- The theorem stating the minimum bailing rate for the given problem --/
theorem min_bailing_rate_solution :
  MinBailingRateProblem 0.5 6 12 50 := by
  sorry


end min_bailing_rate_solution_l1031_103171


namespace income_data_mean_difference_l1031_103153

/-- Represents the income data for a group of families -/
structure IncomeData where
  num_families : ℕ
  min_income : ℕ
  max_income : ℕ
  incorrect_max_income : ℕ

/-- Calculates the difference between the mean of incorrect data and actual data -/
def mean_difference (data : IncomeData) : ℚ :=
  (data.incorrect_max_income - data.max_income : ℚ) / data.num_families

/-- Theorem stating the difference between means for the given problem -/
theorem income_data_mean_difference :
  ∀ (data : IncomeData),
  data.num_families = 800 →
  data.min_income = 10000 →
  data.max_income = 120000 →
  data.incorrect_max_income = 1200000 →
  mean_difference data = 1350 := by
  sorry

#eval mean_difference {
  num_families := 800,
  min_income := 10000,
  max_income := 120000,
  incorrect_max_income := 1200000
}

end income_data_mean_difference_l1031_103153


namespace cafe_benches_theorem_l1031_103177

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 5^2 + tens * 5^1 + ones * 5^0

/-- Calculates the number of benches needed given a total number of people and people per bench -/
def benchesNeeded (totalPeople : Nat) (peoplePerBench : Nat) : Nat :=
  (totalPeople + peoplePerBench - 1) / peoplePerBench

theorem cafe_benches_theorem (cafeCapacity : Nat) (peoplePerBench : Nat) :
  cafeCapacity = 310 ∧ peoplePerBench = 3 →
  benchesNeeded (base5ToBase10 cafeCapacity) peoplePerBench = 27 := by
  sorry

#eval benchesNeeded (base5ToBase10 310) 3

end cafe_benches_theorem_l1031_103177


namespace lcm_gcd_product_15_45_l1031_103174

theorem lcm_gcd_product_15_45 : Nat.lcm 15 45 * Nat.gcd 15 45 = 675 := by
  sorry

end lcm_gcd_product_15_45_l1031_103174


namespace double_negation_and_abs_value_l1031_103133

theorem double_negation_and_abs_value : 
  (-(-2) = 2) ∧ (-(abs (-2)) = -2) := by sorry

end double_negation_and_abs_value_l1031_103133


namespace roots_quadratic_equation_l1031_103135

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 - a - 2013 = 0) → 
  (b^2 - b - 2013 = 0) → 
  (a ≠ b) →
  (a^2 + 2*a + 3*b - 2 = 2014) := by
  sorry

end roots_quadratic_equation_l1031_103135


namespace max_ratio_squared_l1031_103176

theorem max_ratio_squared (a b x y : ℝ) : 
  0 < a → 0 < b → a ≥ b →
  0 ≤ x → x < a →
  0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 →
  b^2 + x^2 = (a - x)^2 + (b - y)^2 →
  a^2 + b^2 = x^2 + b^2 →
  (∀ a' b' : ℝ, 0 < a' → 0 < b' → a' ≥ b' → (a' / b')^2 ≤ 4/3) ∧
  (∃ a' b' : ℝ, 0 < a' → 0 < b' → a' ≥ b' → (a' / b')^2 = 4/3) :=
by sorry

end max_ratio_squared_l1031_103176


namespace balloon_fraction_after_tripling_l1031_103187

theorem balloon_fraction_after_tripling (total : ℝ) (h : total > 0) :
  let yellow_initial := (2/3) * total
  let green_initial := total - yellow_initial
  let green_after := 3 * green_initial
  let total_after := yellow_initial + green_after
  green_after / total_after = 3/5 := by
sorry

end balloon_fraction_after_tripling_l1031_103187


namespace specific_ellipse_area_l1031_103113

/-- An ellipse with given properties --/
structure Ellipse where
  major_axis_endpoint1 : ℝ × ℝ
  major_axis_endpoint2 : ℝ × ℝ
  point_on_ellipse : ℝ × ℝ

/-- The area of an ellipse with the given properties --/
def ellipse_area (e : Ellipse) : ℝ := sorry

/-- Theorem stating that the area of the specific ellipse is 50π --/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_endpoint1 := (-5, 2),
    major_axis_endpoint2 := (15, 2),
    point_on_ellipse := (11, 6)
  }
  ellipse_area e = 50 * Real.pi := by sorry

end specific_ellipse_area_l1031_103113


namespace statement_1_statement_4_main_theorem_l1031_103165

-- Statement ①
theorem statement_1 : ∀ x y : ℝ, (x + y ≠ 3) → (x ≠ 2 ∨ y ≠ 1) := by sorry

-- Statement ④
theorem statement_4 : (¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1) := by sorry

-- Main theorem combining both statements
theorem main_theorem : (∀ x y : ℝ, (x + y ≠ 3) → (x ≠ 2 ∨ y ≠ 1)) ∧
                       ((¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1)) := by sorry

end statement_1_statement_4_main_theorem_l1031_103165


namespace shoe_discount_percentage_l1031_103103

def original_price : ℝ := 62.50 + 3.75
def amount_saved : ℝ := 3.75
def amount_spent : ℝ := 62.50

theorem shoe_discount_percentage : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |((amount_saved / original_price) * 100 - 6)| < ε :=
sorry

end shoe_discount_percentage_l1031_103103


namespace max_volume_triangular_cone_l1031_103198

/-- A quadrilateral cone with a square base -/
structure QuadrilateralCone where
  /-- Side length of the square base -/
  baseSideLength : ℝ
  /-- Sum of distances from apex to two adjacent vertices of the base -/
  sumOfDistances : ℝ

/-- Theorem: Maximum volume of triangular cone (A-BCM) -/
theorem max_volume_triangular_cone (cone : QuadrilateralCone) 
  (h1 : cone.baseSideLength = 6)
  (h2 : cone.sumOfDistances = 10) : 
  ∃ (v : ℝ), v = 24 ∧ ∀ (volume : ℝ), volume ≤ v :=
by
  sorry

end max_volume_triangular_cone_l1031_103198


namespace tangent_line_curve_range_l1031_103186

theorem tangent_line_curve_range (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, y = x - a ∧ y = Real.log (x + b) ∧ 
    (∀ x' y' : ℝ, y' = x' - a → y' ≤ Real.log (x' + b))) →
  (∀ z : ℝ, z ∈ Set.Ioo 0 (1/2) ↔ ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 
    (∃ x y : ℝ, y = x - a' ∧ y = Real.log (x + b') ∧ 
      (∀ x' y' : ℝ, y' = x' - a' → y' ≤ Real.log (x' + b'))) ∧
    z = a'^2 / (2 + b')) :=
by sorry

end tangent_line_curve_range_l1031_103186


namespace y_intercept_of_line_with_slope_3_and_x_intercept_4_l1031_103197

/-- A line is defined by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate of the point where the line crosses the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.slope * (-l.point.1) + l.point.2

theorem y_intercept_of_line_with_slope_3_and_x_intercept_4 :
  let l : Line := { slope := 3, point := (4, 0) }
  y_intercept l = -12 := by sorry

end y_intercept_of_line_with_slope_3_and_x_intercept_4_l1031_103197


namespace sammy_bottle_caps_l1031_103132

theorem sammy_bottle_caps :
  ∀ (billie janine sammy : ℕ),
    billie = 2 →
    janine = 3 * billie →
    sammy = janine + 2 →
    sammy = 8 := by
  sorry

end sammy_bottle_caps_l1031_103132


namespace power_three_2023_mod_seven_l1031_103179

theorem power_three_2023_mod_seven : 3^2023 % 7 = 3 := by
  sorry

end power_three_2023_mod_seven_l1031_103179


namespace min_x_prime_factorization_l1031_103180

theorem min_x_prime_factorization (x y : ℕ+) (h : 13 * x^4 = 29 * y^12) :
  ∃ (a b c d : ℕ), 
    (x = (29^3 : ℕ+) * (13^3 : ℕ+)) ∧
    (∀ z : ℕ+, 13 * z^4 = 29 * y^12 → x ≤ z) ∧
    (Nat.Prime a ∧ Nat.Prime b) ∧
    (x = a^c * b^d) ∧
    (a + b + c + d = 48) := by
  sorry

end min_x_prime_factorization_l1031_103180


namespace tangent_line_and_minimum_value_l1031_103108

noncomputable section

-- Define the function f
def f (x a b : ℝ) : ℝ := Real.exp x * (x^2 - (a + 2) * x + b)

-- Define the derivative of f
def f' (x a b : ℝ) : ℝ := Real.exp x * (x^2 - a * x + b - (a + 2))

theorem tangent_line_and_minimum_value (a b : ℝ) :
  (f' 0 a b = -2 * a^2) →
  (b = a + 2 - 2 * a^2) ∧
  (∀ a < 0, ∃ M ≥ 2, ∀ x > 0, f x a b < M) :=
by sorry

end

end tangent_line_and_minimum_value_l1031_103108


namespace intersection_line_equation_l1031_103148

-- Define the circle (x-1)^2 + y^2 = 1
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point P
def P : ℝ × ℝ := (2, 3)

-- Define point C (center of circle1)
def C : ℝ × ℝ := (1, 0)

-- Define the circle with diameter PC
def circle2 (x y : ℝ) : Prop := (x - (P.1 + C.1)/2)^2 + (y - (P.2 + C.2)/2)^2 = ((P.1 - C.1)^2 + (P.2 - C.2)^2) / 4

-- Theorem statement
theorem intersection_line_equation : 
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → x + 3*y - 2 = 0 := by sorry

end intersection_line_equation_l1031_103148


namespace number_property_l1031_103167

theorem number_property : ∃! x : ℝ, x - 18 = 3 * (86 - x) :=
  sorry

end number_property_l1031_103167


namespace inequality_and_equality_conditions_l1031_103170

theorem inequality_and_equality_conditions (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) :
  ((1 + a)^2 / (1 + b) ≤ 1 + a^2 / b ↔ b < -1 ∨ b > 0) ∧
  ((1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b ∧ b ≠ -1 ∧ b ≠ 0) :=
by sorry

end inequality_and_equality_conditions_l1031_103170


namespace total_cost_calculation_l1031_103175

/-- The total cost of remaining balloons for Sam and Mary -/
def total_cost (s a m c : ℝ) : ℝ :=
  ((s - a) + m) * c

/-- Theorem stating the total cost of remaining balloons for Sam and Mary -/
theorem total_cost_calculation (s a m c : ℝ) 
  (hs : s = 6) (ha : a = 5) (hm : m = 7) (hc : c = 9) : 
  total_cost s a m c = 72 := by
  sorry

#eval total_cost 6 5 7 9

end total_cost_calculation_l1031_103175


namespace firecracker_sales_profit_l1031_103112

/-- Electronic firecracker sales model -/
structure FirecrackerSales where
  cost : ℝ
  price : ℝ
  volume : ℝ
  profit : ℝ
  h1 : cost = 80
  h2 : 80 ≤ price ∧ price ≤ 160
  h3 : volume = -2 * price + 320
  h4 : profit = (price - cost) * volume

/-- Theorem about firecracker sales profit -/
theorem firecracker_sales_profit (model : FirecrackerSales) :
  -- 1. Profit function
  model.profit = -2 * model.price^2 + 480 * model.price - 25600 ∧
  -- 2. Maximum profit
  (∃ max_profit : ℝ, max_profit = 3200 ∧
    ∀ p, 80 ≤ p ∧ p ≤ 160 → 
      -2 * p^2 + 480 * p - 25600 ≤ max_profit) ∧
  (∃ max_price : ℝ, max_price = 120 ∧
    -2 * max_price^2 + 480 * max_price - 25600 = 3200) ∧
  -- 3. Profit of 2400 at lower price
  (∃ lower_price : ℝ, lower_price = 100 ∧
    -2 * lower_price^2 + 480 * lower_price - 25600 = 2400 ∧
    ∀ p, 80 ≤ p ∧ p ≤ 160 ∧ p ≠ lower_price ∧
      -2 * p^2 + 480 * p - 25600 = 2400 → p > lower_price) := by
  sorry

end firecracker_sales_profit_l1031_103112


namespace paving_cost_l1031_103140

/-- The cost of paving a rectangular floor given its dimensions and rate per square meter. -/
theorem paving_cost (length width rate : ℝ) : length = 5.5 → width = 4 → rate = 700 → length * width * rate = 15400 := by
  sorry

end paving_cost_l1031_103140


namespace class_composition_solution_l1031_103196

/-- Represents the class composition problem --/
structure ClassComposition where
  total_students : ℕ
  girls : ℕ
  boys : ℕ

/-- Checks if the given class composition satisfies the problem conditions --/
def satisfies_conditions (c : ClassComposition) : Prop :=
  c.total_students = c.girls + c.boys ∧
  c.girls * 2 = c.boys * 3 ∧
  (c.total_students * 2 - 150 = c.girls * 5)

/-- The theorem stating the solution to the class composition problem --/
theorem class_composition_solution :
  ∃ c : ClassComposition, c.total_students = 300 ∧ c.girls = 180 ∧ c.boys = 120 ∧
  satisfies_conditions c := by
  sorry

#check class_composition_solution

end class_composition_solution_l1031_103196


namespace sally_picked_42_peaches_l1031_103122

/-- The number of peaches Sally picked at the orchard -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

/-- Theorem: Sally picked 42 peaches at the orchard -/
theorem sally_picked_42_peaches (initial final : ℕ) 
  (h1 : initial = 13) 
  (h2 : final = 55) : 
  peaches_picked initial final = 42 := by
  sorry

end sally_picked_42_peaches_l1031_103122


namespace smallest_number_proof_l1031_103105

def smallest_number : ℕ := 910314816600

theorem smallest_number_proof :
  (∀ i ∈ Finset.range 28, smallest_number % (i + 1) = 0) ∧
  smallest_number % 29 ≠ 0 ∧
  smallest_number % 30 ≠ 0 ∧
  (∀ n < smallest_number, 
    (∀ i ∈ Finset.range 28, n % (i + 1) = 0) →
    (n % 29 = 0 ∨ n % 30 = 0)) :=
by sorry

end smallest_number_proof_l1031_103105


namespace units_digit_of_sum_of_powers_l1031_103114

theorem units_digit_of_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (42^4 + 24^4) % 10 = n ∧ n = 2 := by
  sorry

end units_digit_of_sum_of_powers_l1031_103114


namespace parking_space_per_car_l1031_103142

/-- Calculates the area required to park one car given the dimensions of a parking lot,
    the percentage of usable area, and the total number of cars that can be parked. -/
theorem parking_space_per_car
  (length width : ℝ)
  (usable_percentage : ℝ)
  (total_cars : ℕ)
  (h1 : length = 400)
  (h2 : width = 500)
  (h3 : usable_percentage = 0.8)
  (h4 : total_cars = 16000) :
  (length * width * usable_percentage) / total_cars = 10 := by
  sorry

#check parking_space_per_car

end parking_space_per_car_l1031_103142


namespace student_allowance_proof_l1031_103110

def weekly_allowance : ℝ := 3.00

theorem student_allowance_proof :
  let arcade_spend := (2 : ℝ) / 5 * weekly_allowance
  let remaining_after_arcade := weekly_allowance - arcade_spend
  let toy_store_spend := (1 : ℝ) / 3 * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spend
  remaining_after_toy_store = 1.20
  →
  weekly_allowance = 3.00 := by
  sorry

end student_allowance_proof_l1031_103110


namespace intersection_distance_product_l1031_103106

/-- Given a line passing through (0, 1) that intersects y = x^2 at A and B,
    the product of the absolute values of x-coordinates of A and B is 1 -/
theorem intersection_distance_product (k : ℝ) : 
  let line := fun x => k * x + 1
  let parabola := fun x => x^2
  let roots := {x : ℝ | parabola x = line x}
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ a ≠ b ∧ |a| * |b| = 1 := by
  sorry

end intersection_distance_product_l1031_103106


namespace quadratic_equation_roots_l1031_103169

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + 3*x₁ - 2 = 0) ∧ (x₂^2 + 3*x₂ - 2 = 0) := by
  sorry

end quadratic_equation_roots_l1031_103169


namespace carina_coffee_amount_l1031_103109

/-- Calculates the total amount of coffee Carina has given the number of 10-ounce packages -/
def total_coffee (num_ten_oz_packages : ℕ) : ℕ :=
  let num_five_oz_packages := num_ten_oz_packages + 2
  let oz_from_ten := 10 * num_ten_oz_packages
  let oz_from_five := 5 * num_five_oz_packages
  oz_from_ten + oz_from_five

/-- Proves that Carina has 115 ounces of coffee in total -/
theorem carina_coffee_amount : total_coffee 7 = 115 := by
  sorry

end carina_coffee_amount_l1031_103109


namespace least_divisible_by_five_smallest_primes_l1031_103116

def five_smallest_primes : List Nat := [2, 3, 5, 7, 11]

def product_of_primes : Nat := five_smallest_primes.prod

theorem least_divisible_by_five_smallest_primes :
  (∀ n : Nat, n > 0 ∧ (∀ p ∈ five_smallest_primes, n % p = 0) → n ≥ product_of_primes) ∧
  (∀ p ∈ five_smallest_primes, product_of_primes % p = 0) :=
sorry

end least_divisible_by_five_smallest_primes_l1031_103116


namespace min_power_congruence_l1031_103150

theorem min_power_congruence :
  ∃ (m n : ℕ), 
    n > m ∧ 
    m ≥ 1 ∧ 
    42^n % 100 = 42^m % 100 ∧
    (∀ (m' n' : ℕ), n' > m' ∧ m' ≥ 1 ∧ 42^n' % 100 = 42^m' % 100 → m + n ≤ m' + n') ∧
    m = 2 ∧
    n = 22 :=
by sorry

end min_power_congruence_l1031_103150


namespace quadratic_equation_from_sum_and_sum_of_squares_l1031_103134

theorem quadratic_equation_from_sum_and_sum_of_squares 
  (x₁ x₂ : ℝ) 
  (h_sum : x₁ + x₂ = 3) 
  (h_sum_squares : x₁^2 + x₂^2 = 5) :
  ∀ x : ℝ, x^2 - 3*x + 2 = 0 ↔ (x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_equation_from_sum_and_sum_of_squares_l1031_103134


namespace intersection_of_M_and_N_l1031_103107

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 1 2 := by sorry

end intersection_of_M_and_N_l1031_103107


namespace f_derivative_l1031_103130

noncomputable def f (x : ℝ) := Real.sin x + 3^x

theorem f_derivative (x : ℝ) : 
  deriv f x = Real.cos x + 3^x * Real.log 3 := by sorry

end f_derivative_l1031_103130


namespace population_after_five_years_l1031_103147

/-- Represents the yearly change in organization population -/
def yearly_change (b : ℝ) : ℝ := 2.7 * b - 8.5

/-- Calculates the population after n years -/
def population_after_years (initial_population : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => initial_population
  | n + 1 => yearly_change (population_after_years initial_population n)

/-- Theorem stating the population after 5 years -/
theorem population_after_five_years :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |population_after_years 25 5 - 2875| < ε :=
sorry

end population_after_five_years_l1031_103147


namespace intersection_empty_iff_a_values_l1031_103111

def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - a) = 0}

def B : Set ℝ := {x | (x - 2) * (x - 3) = 0}

theorem intersection_empty_iff_a_values (a : ℝ) :
  A a ∩ B = ∅ ↔ a = 1 ∨ a = 4 ∨ a = 6 := by
  sorry

end intersection_empty_iff_a_values_l1031_103111


namespace chess_team_boys_count_l1031_103115

theorem chess_team_boys_count :
  ∀ (total_members : ℕ) (total_attendees : ℕ) (boys : ℕ) (girls : ℕ),
  total_members = 30 →
  total_attendees = 20 →
  total_members = boys + girls →
  total_attendees = boys + (girls / 3) →
  boys = 15 := by
sorry

end chess_team_boys_count_l1031_103115


namespace lines_perp_to_plane_are_parallel_l1031_103160

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perp m α) 
  (h3 : perp n α) : 
  parallel m n :=
sorry

end lines_perp_to_plane_are_parallel_l1031_103160


namespace actual_length_is_320_l1031_103104

/-- Blueprint scale factor -/
def scale_factor : ℝ := 20

/-- Measured length on the blueprint in cm -/
def measured_length : ℝ := 16

/-- Actual length of the part in cm -/
def actual_length : ℝ := measured_length * scale_factor

/-- Theorem stating that the actual length is 320cm -/
theorem actual_length_is_320 : actual_length = 320 := by
  sorry

end actual_length_is_320_l1031_103104


namespace staircase_perimeter_l1031_103166

/-- Given a rectangle with a staircase-shaped region removed, 
    if the remaining area is 104 square feet, 
    then the perimeter of the remaining region is 52.4 feet. -/
theorem staircase_perimeter (width height : ℝ) (area remaining_area : ℝ) : 
  width = 10 →
  area = width * height →
  remaining_area = area - 40 →
  remaining_area = 104 →
  width + height + 3 + 5 + 20 = 52.4 :=
by sorry

end staircase_perimeter_l1031_103166


namespace diagonal_intersection_theorem_l1031_103139

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem diagonal_intersection_theorem (ABCD : Quadrilateral) (E : Point) :
  isConvex ABCD →
  distance ABCD.A ABCD.B = 9 →
  distance ABCD.C ABCD.D = 12 →
  distance ABCD.A ABCD.C = 14 →
  E = lineIntersection ABCD.A ABCD.C ABCD.B ABCD.D →
  triangleArea ABCD.A E ABCD.D = triangleArea ABCD.B E ABCD.C →
  distance ABCD.A E = 6 := by
  sorry

end diagonal_intersection_theorem_l1031_103139


namespace rug_coverage_area_l1031_103199

/-- Given three rugs with specified overlapping areas, calculate the total floor area covered. -/
theorem rug_coverage_area (total_rug_area double_layer triple_layer : ℝ) 
  (h1 : total_rug_area = 212)
  (h2 : double_layer = 24)
  (h3 : triple_layer = 24) :
  total_rug_area - double_layer - 2 * triple_layer = 140 :=
by sorry

end rug_coverage_area_l1031_103199


namespace isosceles_triangle_perimeter_l1031_103136

/-- A triangle with side lengths a, b, and c is isosceles if at least two of its sides are equal -/
def IsIsosceles (a b c : ℝ) : Prop := a = b ∨ b = c ∨ a = c

/-- The perimeter of a triangle with side lengths a, b, and c -/
def Perimeter (a b c : ℝ) : ℝ := a + b + c

theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  IsIsosceles a b c →
  (a = 5 ∧ b = 8) ∨ (a = 8 ∧ b = 5) ∨ (b = 5 ∧ c = 8) ∨ (b = 8 ∧ c = 5) ∨ (a = 5 ∧ c = 8) ∨ (a = 8 ∧ c = 5) →
  Perimeter a b c = 18 ∨ Perimeter a b c = 21 :=
by sorry

end isosceles_triangle_perimeter_l1031_103136


namespace cube_surface_area_l1031_103124

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 512 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 384 := by
  sorry

end cube_surface_area_l1031_103124


namespace counterexample_exists_l1031_103181

theorem counterexample_exists : ∃ (a b : ℝ), a < b ∧ a^2 ≥ b^2 := by
  sorry

end counterexample_exists_l1031_103181


namespace log_36_in_terms_of_a_b_l1031_103127

theorem log_36_in_terms_of_a_b (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  Real.log 36 = 2 * a + 2 * b := by
  sorry

end log_36_in_terms_of_a_b_l1031_103127


namespace faye_money_left_l1031_103129

def initial_money : ℕ := 20
def mother_multiplier : ℕ := 2
def cupcake_price : ℚ := 3/2
def cupcake_quantity : ℕ := 10
def cookie_box_price : ℕ := 3
def cookie_box_quantity : ℕ := 5

theorem faye_money_left :
  let total_money := initial_money + mother_multiplier * initial_money
  let spent_money := cupcake_price * cupcake_quantity + cookie_box_price * cookie_box_quantity
  total_money - spent_money = 30 := by
sorry

end faye_money_left_l1031_103129


namespace arithmetic_expression_value_l1031_103131

theorem arithmetic_expression_value :
  ∀ (A B C : Nat),
    A ≠ B → A ≠ C → B ≠ C →
    A < 10 → B < 10 → C < 10 →
    3 * C % 10 = C →
    (2 * B + 1) % 10 = B →
    300 + 10 * B + C = 395 :=
by
  sorry

end arithmetic_expression_value_l1031_103131


namespace stock_percentage_l1031_103188

/-- The percentage of a stock given certain conditions -/
theorem stock_percentage (income : ℝ) (investment : ℝ) (percentage : ℝ) : 
  income = 1000 →
  investment = 10000 →
  income = (percentage * investment) / 100 →
  percentage = 10 := by
sorry

end stock_percentage_l1031_103188


namespace specific_doctor_selection_mixed_team_selection_l1031_103163

-- Define the number of doctors
def total_doctors : ℕ := 20
def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8

-- Define the number of doctors to be selected
def team_size : ℕ := 5

-- Theorem for part (1)
theorem specific_doctor_selection :
  Nat.choose (total_doctors - 2) (team_size - 1) = 3060 := by sorry

-- Theorem for part (2)
theorem mixed_team_selection :
  Nat.choose total_doctors team_size - 
  Nat.choose internal_medicine_doctors team_size - 
  Nat.choose surgeons team_size = 14656 := by sorry

end specific_doctor_selection_mixed_team_selection_l1031_103163


namespace least_tablets_for_given_box_l1031_103192

/-- The least number of tablets to extract from a box containing two types of medicine
    to ensure at least two tablets of each kind are among the extracted. -/
def least_tablets_to_extract (tablets_a tablets_b : ℕ) : ℕ :=
  max ((tablets_a - 1) + 2) ((tablets_b - 1) + 2)

/-- Theorem: Given a box with 10 tablets of medicine A and 13 tablets of medicine B,
    the least number of tablets that should be taken to ensure at least two tablets
    of each kind are among the extracted is 12. -/
theorem least_tablets_for_given_box :
  least_tablets_to_extract 10 13 = 12 := by
  sorry

end least_tablets_for_given_box_l1031_103192


namespace birch_count_is_87_l1031_103128

def is_valid_tree_arrangement (total_trees : ℕ) (birch_count : ℕ) : Prop :=
  ∃ (lime_count : ℕ),
    -- Total number of trees is 130
    total_trees = 130 ∧
    -- Sum of birches and limes is the total number of trees
    birch_count + lime_count = total_trees ∧
    -- There is at least one birch and one lime
    birch_count > 0 ∧ lime_count > 0 ∧
    -- The number of limes is equal to the number of groups of two birches plus one lime
    lime_count = (birch_count - 1) / 2 ∧
    -- There is exactly one group of three consecutive birches
    (birch_count - 1) % 2 = 1

theorem birch_count_is_87 :
  ∃ (birch_count : ℕ), is_valid_tree_arrangement 130 birch_count ∧ birch_count = 87 :=
sorry

end birch_count_is_87_l1031_103128


namespace five_digit_multiple_of_6_l1031_103183

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

theorem five_digit_multiple_of_6 (d : ℕ) :
  d < 10 →
  is_multiple_of_6 (47690 + d) →
  d = 4 ∨ d = 8 := by
  sorry

end five_digit_multiple_of_6_l1031_103183


namespace gcd_of_256_180_600_l1031_103195

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 4 := by
  sorry

end gcd_of_256_180_600_l1031_103195


namespace basketball_team_math_enrollment_l1031_103100

theorem basketball_team_math_enrollment (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 25 →
  physics_players = 12 →
  both_subjects = 5 →
  (∃ (math_players : ℕ), math_players = total_players - physics_players + both_subjects ∧ math_players = 18) :=
by
  sorry

end basketball_team_math_enrollment_l1031_103100


namespace f_is_even_and_decreasing_l1031_103143

def f (x : ℝ) : ℝ := -x^2 + 1

theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end f_is_even_and_decreasing_l1031_103143


namespace smallest_k_value_l1031_103121

theorem smallest_k_value (p q r s k : ℕ+) : 
  (p + 2*q + 3*r + 4*s = k) →
  (4*p = 3*q) →
  (4*p = 2*r) →
  (4*p = s) →
  (∀ p' q' r' s' k' : ℕ+, 
    (p' + 2*q' + 3*r' + 4*s' = k') →
    (4*p' = 3*q') →
    (4*p' = 2*r') →
    (4*p' = s') →
    k ≤ k') →
  k = 77 := by
sorry

end smallest_k_value_l1031_103121


namespace part_one_part_two_l1031_103119

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(m+1)*x + m^2 - 5 = 0}

-- Theorem for part (1)
theorem part_one (a : ℝ) : A ∪ B a = A → a = 2 ∨ a = 3 := by sorry

-- Theorem for part (2)
theorem part_two (m : ℝ) : A ∩ C m = C m → m ≤ -3 := by sorry

end part_one_part_two_l1031_103119


namespace extreme_point_and_tangent_lines_l1031_103164

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - a^2*x

-- State the theorem
theorem extreme_point_and_tangent_lines :
  -- Given conditions
  ∃ (a : ℝ), (∃ (x : ℝ), x = 1 ∧ (∀ (h : ℝ), h ≠ 0 → (f a (x + h) - f a x) / h > 0 ∨ (f a (x + h) - f a x) / h < 0)) →
  -- Conclusions
  (∃ (x : ℝ), f a x = -5 ∧ ∀ (y : ℝ), f a y ≥ -5) ∧
  (f 1 0 = 0 ∧ ∃ (m₁ m₂ : ℝ), m₁ = -1 ∧ m₂ = -5/4 ∧
    ∀ (x : ℝ), (f 1 x = m₁ * x ∨ f 1 x = m₂ * x) → 
      ∀ (y : ℝ), y = m₁ * x ∨ y = m₂ * x → f 1 y = y) :=
by sorry

end extreme_point_and_tangent_lines_l1031_103164


namespace inequality_proof_l1031_103155

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b) + (b / Real.sqrt a) ≥ Real.sqrt a + Real.sqrt b ∧
  (a + b = 1 → (1/a) + (1/b) + (1/(a*b)) ≥ 8) :=
by sorry

end inequality_proof_l1031_103155


namespace account_balance_first_year_l1031_103173

/-- Proves that the account balance at the end of the first year is correct -/
theorem account_balance_first_year 
  (initial_deposit : ℝ) 
  (interest_first_year : ℝ) 
  (balance_first_year : ℝ) :
  initial_deposit = 1000 →
  interest_first_year = 100 →
  balance_first_year = initial_deposit + interest_first_year →
  balance_first_year = 1100 := by
  sorry

#check account_balance_first_year

end account_balance_first_year_l1031_103173


namespace population_growth_rate_l1031_103178

/-- Proves that given a population of 10,000 that grows to 12,100 in 2 years
    with a constant annual growth rate, the annual percentage increase is 10%. -/
theorem population_growth_rate (initial_population : ℕ) (final_population : ℕ) 
  (years : ℕ) (growth_rate : ℝ) :
  initial_population = 10000 →
  final_population = 12100 →
  years = 2 →
  final_population = initial_population * (1 + growth_rate) ^ years →
  growth_rate = 0.1 := by
  sorry

end population_growth_rate_l1031_103178


namespace right_triangle_area_l1031_103145

theorem right_triangle_area (h : ℝ) (angle : ℝ) : 
  h = 13 → angle = 45 → 
  let area := (1/2) * (h / Real.sqrt 2) * (h / Real.sqrt 2)
  area = 84.5 := by sorry

end right_triangle_area_l1031_103145


namespace nancy_books_l1031_103138

theorem nancy_books (alyssa_books : ℕ) (nancy_multiplier : ℕ) : 
  alyssa_books = 36 → nancy_multiplier = 7 → alyssa_books * nancy_multiplier = 252 := by
  sorry

end nancy_books_l1031_103138


namespace terminal_side_in_fourth_quadrant_l1031_103144

def angle_in_fourth_quadrant (α : Real) : Prop :=
  -2 * Real.pi < α ∧ α < -3 * Real.pi / 2

theorem terminal_side_in_fourth_quadrant :
  angle_in_fourth_quadrant (-5) :=
sorry

end terminal_side_in_fourth_quadrant_l1031_103144


namespace total_pay_calculation_l1031_103152

/-- Calculate the total pay for a worker given their regular and overtime hours -/
theorem total_pay_calculation (regular_rate : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) :
  let overtime_rate := 2 * regular_rate
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  let total_pay := regular_pay + overtime_pay
  (regular_rate = 3 ∧ regular_hours = 40 ∧ overtime_hours = 10) →
  total_pay = 180 := by
sorry


end total_pay_calculation_l1031_103152


namespace expression_value_l1031_103118

theorem expression_value (x y z : ℤ) (hx : x = -3) (hy : y = 5) (hz : z = -4) :
  x^2 + y^2 - z^2 + 2*x*y = -12 := by
  sorry

end expression_value_l1031_103118


namespace infinitely_many_coprime_binomial_quotients_l1031_103182

/-- Given positive integers k, l, and m, there exist infinitely many positive integers n
    such that (n choose k) / m is a positive integer coprime with m. -/
theorem infinitely_many_coprime_binomial_quotients
  (k l m : ℕ+) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S,
    ∃ (q : ℕ+), Nat.choose n k.val = q.val * m.val ∧ Nat.Coprime q.val m.val :=
sorry

end infinitely_many_coprime_binomial_quotients_l1031_103182


namespace letters_with_both_dot_and_line_l1031_103117

/-- Represents the number of letters in the alphabet -/
def total_letters : ℕ := 40

/-- Represents the number of letters with only a straight line -/
def straight_line_only : ℕ := 24

/-- Represents the number of letters with only a dot -/
def dot_only : ℕ := 7

/-- Represents the number of letters with both a dot and a straight line -/
def both : ℕ := total_letters - straight_line_only - dot_only

theorem letters_with_both_dot_and_line :
  both = 9 :=
sorry

end letters_with_both_dot_and_line_l1031_103117


namespace whistle_cost_l1031_103101

theorem whistle_cost (total_cost yoyo_cost : ℕ) (h1 : total_cost = 38) (h2 : yoyo_cost = 24) :
  total_cost - yoyo_cost = 14 := by
  sorry

end whistle_cost_l1031_103101


namespace mitchell_gum_chewing_l1031_103190

theorem mitchell_gum_chewing (packets : ℕ) (pieces_per_packet : ℕ) (not_chewed : ℕ) :
  packets = 8 →
  pieces_per_packet = 7 →
  not_chewed = 2 →
  packets * pieces_per_packet - not_chewed = 54 := by
  sorry

end mitchell_gum_chewing_l1031_103190


namespace system_solution_exists_l1031_103102

theorem system_solution_exists : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  (4 * x₁ - 3 * y₁ = -3 ∧ 8 * x₁ + 5 * y₁ = 11 + x₁^2) ∧
  (4 * x₂ - 3 * y₂ = -3 ∧ 8 * x₂ + 5 * y₂ = 11 + x₂^2) ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by sorry

end system_solution_exists_l1031_103102


namespace four_points_cyclic_l1031_103126

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the necessary geometric relations
variable (collinear : Point → Point → Point → Prop)
variable (orthocenter : Point → Point → Point → Point)
variable (lies_on : Point → Line → Prop)
variable (concurrent : Line → Line → Line → Prop)
variable (cyclic : Point → Point → Point → Point → Prop)

-- Define the theorem
theorem four_points_cyclic
  (A B C D P Q R : Point)
  (AP BQ CR : Line)
  (h1 : ¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A C D ∧ ¬ collinear B C D)
  (h2 : orthocenter B C D ≠ D)
  (h3 : P = orthocenter B C D)
  (h4 : Q = orthocenter C A D)
  (h5 : R = orthocenter A B D)
  (h6 : lies_on A AP ∧ lies_on P AP)
  (h7 : lies_on B BQ ∧ lies_on Q BQ)
  (h8 : lies_on C CR ∧ lies_on R CR)
  (h9 : AP ≠ BQ ∧ BQ ≠ CR ∧ CR ≠ AP)
  (h10 : concurrent AP BQ CR)
  : cyclic A B C D :=
sorry

end four_points_cyclic_l1031_103126


namespace table_tennis_probabilities_l1031_103162

def num_players : ℕ := 6
def num_players_A : ℕ := 3
def num_players_B : ℕ := 1
def num_players_C : ℕ := 2

def probability_at_least_one_C : ℚ := 3/5
def probability_same_association : ℚ := 4/15

theorem table_tennis_probabilities :
  (num_players = num_players_A + num_players_B + num_players_C) →
  (probability_at_least_one_C = 3/5) ∧
  (probability_same_association = 4/15) :=
by sorry

end table_tennis_probabilities_l1031_103162


namespace intersection_points_count_l1031_103184

/-- The number of intersection points between y = |3x + 6| and y = -|4x - 3| -/
theorem intersection_points_count : ∃! p : ℝ × ℝ, 
  (|3 * p.1 + 6| = p.2) ∧ (-|4 * p.1 - 3| = p.2) := by sorry

end intersection_points_count_l1031_103184
