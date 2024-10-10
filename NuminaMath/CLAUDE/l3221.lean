import Mathlib

namespace unique_x_value_l3221_322178

def A (x : ℝ) : Set ℝ := {2, 0, x}
def B (x : ℝ) : Set ℝ := {2, x^2}

theorem unique_x_value : 
  ∀ x : ℝ, (A x ∩ B x = B x) → x = 1 := by
  sorry

end unique_x_value_l3221_322178


namespace fraction_product_l3221_322180

theorem fraction_product : (1 / 4 : ℚ) * (2 / 5 : ℚ) * (3 / 6 : ℚ) = 1 / 20 := by
  sorry

end fraction_product_l3221_322180


namespace smallest_positive_angle_with_same_terminal_side_l3221_322173

theorem smallest_positive_angle_with_same_terminal_side (angle : ℝ) : 
  angle = 1000 →
  (∃ (k : ℤ), angle = 280 + 360 * k) →
  (∀ (x : ℝ), 0 ≤ x ∧ x < 360 ∧ (∃ (m : ℤ), angle = x + 360 * m) → x ≥ 280) :=
by sorry

end smallest_positive_angle_with_same_terminal_side_l3221_322173


namespace pencil_remainder_l3221_322113

theorem pencil_remainder (a b : ℕ) 
  (ha : a % 8 = 5) 
  (hb : b % 8 = 6) : 
  (a + b) % 8 = 3 := by
sorry

end pencil_remainder_l3221_322113


namespace sum_of_max_min_a_is_zero_l3221_322101

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - a*x - 20*a^2

-- Define the condition that the difference between any two solutions does not exceed 9
def solution_difference_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, f a x < 0 → f a y < 0 → |x - y| ≤ 9

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ :=
  {a : ℝ | solution_difference_condition a}

-- State the theorem
theorem sum_of_max_min_a_is_zero :
  ∃ (a_min a_max : ℝ), 
    a_min ∈ valid_a_set ∧ 
    a_max ∈ valid_a_set ∧ 
    (∀ a ∈ valid_a_set, a_min ≤ a ∧ a ≤ a_max) ∧
    a_min + a_max = 0 := by
  sorry

end sum_of_max_min_a_is_zero_l3221_322101


namespace sum_interior_angles_regular_polygon_l3221_322138

/-- 
For a regular polygon where each exterior angle is 40°, 
the sum of the interior angles is 1260°.
-/
theorem sum_interior_angles_regular_polygon (n : ℕ) : 
  (360 / n = 40) → (n - 2) * 180 = 1260 := by
  sorry

end sum_interior_angles_regular_polygon_l3221_322138


namespace gcd_of_powers_of_three_l3221_322114

theorem gcd_of_powers_of_three : Nat.gcd (3^1001 - 1) (3^1010 - 1) = 19682 := by
  sorry

end gcd_of_powers_of_three_l3221_322114


namespace train_journey_times_l3221_322168

/-- Proves that given the conditions of two trains running late, their usual journey times are both 2 hours -/
theorem train_journey_times (speed_ratio_A speed_ratio_B : ℚ) (delay_A delay_B : ℚ) 
  (h1 : speed_ratio_A = 4/5)
  (h2 : speed_ratio_B = 3/4)
  (h3 : delay_A = 1/2)  -- 30 minutes in hours
  (h4 : delay_B = 2/3)  -- 40 minutes in hours
  : ∃ (T_A T_B : ℚ), T_A = 2 ∧ T_B = 2 ∧ 
    (1/speed_ratio_A) * T_A = T_A + delay_A ∧
    (1/speed_ratio_B) * T_B = T_B + delay_B :=
by sorry


end train_journey_times_l3221_322168


namespace gcd_168_486_l3221_322140

theorem gcd_168_486 : Nat.gcd 168 486 = 6 := by
  sorry

end gcd_168_486_l3221_322140


namespace compute_expression_l3221_322162

theorem compute_expression : 3 * 3^4 + 9^60 / 9^59 - 27^3 = -19431 := by
  sorry

end compute_expression_l3221_322162


namespace profit_percentage_calculation_l3221_322112

theorem profit_percentage_calculation (selling_price cost_price : ℝ) : 
  selling_price = 600 → 
  cost_price = 480 → 
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end profit_percentage_calculation_l3221_322112


namespace price_per_shirt_is_35_l3221_322106

/-- Calculates the price per shirt given the following parameters:
    * num_employees: number of employees
    * shirts_per_employee: number of shirts made per employee per day
    * hours_per_shift: number of hours in a shift
    * hourly_wage: hourly wage per employee
    * per_shirt_wage: additional wage per shirt made
    * nonemployee_expenses: daily nonemployee expenses
    * daily_profit: target daily profit
-/
def price_per_shirt (
  num_employees : ℕ
) (shirts_per_employee : ℕ
) (hours_per_shift : ℕ
) (hourly_wage : ℚ
) (per_shirt_wage : ℚ
) (nonemployee_expenses : ℚ
) (daily_profit : ℚ
) : ℚ :=
  let total_shirts := num_employees * shirts_per_employee
  let total_wages := num_employees * hours_per_shift * hourly_wage + total_shirts * per_shirt_wage
  let total_expenses := total_wages + nonemployee_expenses
  let total_revenue := daily_profit + total_expenses
  total_revenue / total_shirts

theorem price_per_shirt_is_35 :
  price_per_shirt 20 20 8 12 5 1000 9080 = 35 := by
  sorry

#eval price_per_shirt 20 20 8 12 5 1000 9080

end price_per_shirt_is_35_l3221_322106


namespace time_addition_sum_l3221_322194

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the resulting time -/
def addTime (start : Time) (dHours dMinutes dSeconds : Nat) : Time :=
  sorry

/-- Converts a 24-hour time to 12-hour format -/
def to12Hour (t : Time) : Time :=
  sorry

theorem time_addition_sum (startTime : Time) :
  let endTime := to12Hour (addTime startTime 145 50 15)
  endTime.hours + endTime.minutes + endTime.seconds = 69 := by
  sorry

end time_addition_sum_l3221_322194


namespace rectangular_box_volume_l3221_322163

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 30)
  (area2 : w * h = 18)
  (area3 : l * h = 15) :
  l * w * h = 90 := by
sorry

end rectangular_box_volume_l3221_322163


namespace integer_solutions_of_equation_l3221_322129

theorem integer_solutions_of_equation : 
  ∀ x y : ℤ, x^2 - x*y - 6*y^2 + 2*x + 19*y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
sorry

end integer_solutions_of_equation_l3221_322129


namespace square_area_from_perimeter_l3221_322171

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 40 → area = (perimeter / 4)^2 → area = 100 := by
  sorry

end square_area_from_perimeter_l3221_322171


namespace two_digit_integers_count_l3221_322121

def digits : List ℕ := [2, 3, 4, 7]
def tens_digits : List ℕ := [2, 3]
def units_digits : List ℕ := [4, 7]

theorem two_digit_integers_count : 
  (List.length tens_digits) * (List.length units_digits) = 4 := by
  sorry

end two_digit_integers_count_l3221_322121


namespace rectangle_area_rectangle_area_is_100_l3221_322135

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_100 :
  rectangle_area 625 10 = 100 := by
  sorry

end rectangle_area_rectangle_area_is_100_l3221_322135


namespace quadratic_roots_condition_l3221_322125

/-- 
For a quadratic equation x^2 - 3x + c to have roots in the form x = (3 ± √(2c-3)) / 2, 
c must equal 2.
-/
theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ ∃ s : ℝ, s^2 = 2*c - 3 ∧ x = (3 + s) / 2 ∨ x = (3 - s) / 2) →
  c = 2 :=
by sorry

end quadratic_roots_condition_l3221_322125


namespace tan_3x_increasing_interval_l3221_322157

theorem tan_3x_increasing_interval (m : ℝ) : 
  (∀ x₁ x₂, m < x₁ ∧ x₁ < x₂ ∧ x₂ < π/6 → Real.tan (3*x₁) < Real.tan (3*x₂)) → 
  m ∈ Set.Icc (-π/6) (π/6) := by
sorry

end tan_3x_increasing_interval_l3221_322157


namespace no_triple_squares_l3221_322189

theorem no_triple_squares (n : ℕ+) : 
  ¬(∃ (a b c : ℕ), (2 * n.val^2 + 1 = a^2) ∧ (3 * n.val^2 + 1 = b^2) ∧ (6 * n.val^2 + 1 = c^2)) :=
by sorry

end no_triple_squares_l3221_322189


namespace square_difference_equality_l3221_322181

theorem square_difference_equality : (2 + 3)^2 - (2^2 + 3^2) = 12 := by
  sorry

end square_difference_equality_l3221_322181


namespace tangent_line_equation_l3221_322142

noncomputable def f (x : ℝ) : ℝ := x - Real.cos x

theorem tangent_line_equation :
  let p : ℝ × ℝ := (π / 2, π / 2)
  let m : ℝ := 1 + Real.sin (π / 2)
  let tangent_eq (x y : ℝ) : Prop := 2 * x - y - π / 2 = 0
  tangent_eq (p.1) (p.2) ∧
  ∀ x y : ℝ, tangent_eq x y ↔ y - p.2 = m * (x - p.1) :=
by sorry

end tangent_line_equation_l3221_322142


namespace replacement_sugar_percentage_l3221_322104

/-- Represents a sugar solution with a given weight and sugar percentage -/
structure SugarSolution where
  weight : ℝ
  sugarPercentage : ℝ

/-- Calculates the amount of sugar in a solution -/
def sugarAmount (solution : SugarSolution) : ℝ :=
  solution.weight * solution.sugarPercentage

theorem replacement_sugar_percentage
  (original : SugarSolution)
  (replacement : SugarSolution)
  (final : SugarSolution)
  (h1 : original.sugarPercentage = 0.10)
  (h2 : final.sugarPercentage = 0.14)
  (h3 : final.weight = original.weight)
  (h4 : replacement.weight = original.weight / 4)
  (h5 : sugarAmount final = sugarAmount original - sugarAmount original / 4 + sugarAmount replacement) :
  replacement.sugarPercentage = 0.26 := by
sorry

end replacement_sugar_percentage_l3221_322104


namespace functional_equation_implies_constant_l3221_322119

/-- A function from ℤ² to [0,1] satisfying the given functional equation -/
def FunctionalEquation (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y : ℤ, 0 ≤ f (x, y) ∧ f (x, y) ≤ 1 ∧ 
  f (x, y) = (f (x - 1, y) + f (x, y - 1)) / 2

/-- Theorem stating that any function satisfying the functional equation must be constant -/
theorem functional_equation_implies_constant 
  (f : ℤ × ℤ → ℝ) 
  (h : FunctionalEquation f) : 
  ∃ c : ℝ, c ∈ Set.Icc 0 1 ∧ ∀ x y : ℤ, f (x, y) = c :=
sorry

end functional_equation_implies_constant_l3221_322119


namespace square_root_of_four_l3221_322190

theorem square_root_of_four : 
  {x : ℝ | x ^ 2 = 4} = {2, -2} := by sorry

end square_root_of_four_l3221_322190


namespace line_satisfies_conditions_l3221_322136

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point bisects a line segment --/
def bisectsSegment (p : Point) (l : Line) : Prop :=
  ∃ (p1 p2 : Point), pointOnLine p1 l ∧ pointOnLine p2 l ∧ 
    p.x = (p1.x + p2.x) / 2 ∧ p.y = (p1.y + p2.y) / 2

/-- Check if a line lies between two other lines --/
def linesBetween (l : Line) (l1 l2 : Line) : Prop :=
  ∀ (p : Point), pointOnLine p l → 
    (l1.a * p.x + l1.b * p.y + l1.c) * (l2.a * p.x + l2.b * p.y + l2.c) ≤ 0

theorem line_satisfies_conditions : 
  let P : Point := ⟨3, 0⟩
  let L : Line := ⟨8, -1, -24⟩
  let L1 : Line := ⟨2, -1, -2⟩
  let L2 : Line := ⟨1, 1, 3⟩
  pointOnLine P L ∧ 
  bisectsSegment P L ∧
  linesBetween L L1 L2 :=
by sorry

end line_satisfies_conditions_l3221_322136


namespace stream_speed_l3221_322150

/-- Proves that given a boat with a speed of 22 km/hr in still water,
    traveling 189 km downstream in 7 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 22 →
  downstream_distance = 189 →
  downstream_time = 7 →
  ∃ stream_speed : ℝ,
    stream_speed = 5 ∧
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry


end stream_speed_l3221_322150


namespace dog_food_insufficient_l3221_322177

/-- Proves that the amount of dog food remaining after two weeks is negative -/
theorem dog_food_insufficient (num_dogs : ℕ) (food_per_meal : ℚ) (meals_per_day : ℕ) 
  (initial_food : ℚ) (days : ℕ) :
  num_dogs = 5 →
  food_per_meal = 3/4 →
  meals_per_day = 3 →
  initial_food = 45 →
  days = 14 →
  initial_food - (num_dogs * food_per_meal * meals_per_day * days) < 0 :=
by sorry

end dog_food_insufficient_l3221_322177


namespace quadratic_roots_relation_l3221_322158

theorem quadratic_roots_relation (c d : ℚ) : 
  (∃ r s : ℚ, r + s = 3/5 ∧ r * s = -8/5) →
  (∃ p q : ℚ, p + q = -c ∧ p * q = d ∧ p = r - 3 ∧ q = s - 3) →
  d = 28/5 := by
sorry

end quadratic_roots_relation_l3221_322158


namespace trigonometric_identities_l3221_322147

theorem trigonometric_identities :
  (Real.tan (25 * π / 180) + Real.tan (20 * π / 180) + Real.tan (25 * π / 180) * Real.tan (20 * π / 180) = 1) ∧
  (1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.cos (10 * π / 180) = 4) := by
  sorry

end trigonometric_identities_l3221_322147


namespace vegetarians_count_l3221_322197

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  onlyVeg : ℕ
  onlyNonVeg : ℕ
  bothVegAndNonVeg : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def totalVegetarians (fd : FamilyDiet) : ℕ :=
  fd.onlyVeg + fd.bothVegAndNonVeg

/-- Theorem stating that the number of vegetarians in the family is 28 -/
theorem vegetarians_count (fd : FamilyDiet) 
  (h1 : fd.onlyVeg = 16)
  (h2 : fd.onlyNonVeg = 9)
  (h3 : fd.bothVegAndNonVeg = 12) :
  totalVegetarians fd = 28 := by
  sorry

end vegetarians_count_l3221_322197


namespace find_m_value_l3221_322126

theorem find_m_value (α : Real) (m : Real) :
  let P : Real × Real := (-8 * m, -6 * Real.sin (30 * π / 180))
  (∃ (r : Real), r > 0 ∧ P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α) →
  Real.cos α = -4/5 →
  m = 1/2 := by
sorry

end find_m_value_l3221_322126


namespace sunflower_cost_l3221_322111

theorem sunflower_cost
  (num_roses : ℕ)
  (num_sunflowers : ℕ)
  (cost_per_rose : ℚ)
  (total_cost : ℚ)
  (h1 : num_roses = 24)
  (h2 : num_sunflowers = 3)
  (h3 : cost_per_rose = 3/2)
  (h4 : total_cost = 45) :
  (total_cost - num_roses * cost_per_rose) / num_sunflowers = 3 := by
sorry

end sunflower_cost_l3221_322111


namespace intersection_value_l3221_322196

-- Define the complex plane
variable (z : ℂ)

-- Define the first equation |z - 3| = 3|z + 3|
def equation1 (z : ℂ) : Prop := Complex.abs (z - 3) = 3 * Complex.abs (z + 3)

-- Define the second equation |z| = k
def equation2 (z : ℂ) (k : ℝ) : Prop := Complex.abs z = k

-- Define the condition of intersection at exactly one point
def single_intersection (k : ℝ) : Prop :=
  ∃! z, equation1 z ∧ equation2 z k

-- The theorem to prove
theorem intersection_value :
  ∃! k, k > 0 ∧ single_intersection k ∧ k = 4.5 :=
sorry

end intersection_value_l3221_322196


namespace worker_savings_fraction_l3221_322179

theorem worker_savings_fraction (P : ℝ) (S : ℝ) (h1 : P > 0) (h2 : 0 ≤ S ∧ S ≤ 1) 
  (h3 : 12 * S * P = 2 * (1 - S) * P) : S = 1 / 7 := by
  sorry

end worker_savings_fraction_l3221_322179


namespace original_number_is_seventeen_l3221_322153

theorem original_number_is_seventeen : 
  ∀ x : ℕ, 
  (∀ y : ℕ, y < 6 → ¬(23 ∣ (x + y))) → 
  (23 ∣ (x + 6)) → 
  x = 17 := by
sorry

end original_number_is_seventeen_l3221_322153


namespace acute_triangle_tangent_difference_range_l3221_322195

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b² - a² = ac, then 1 < 1/tan(A) - 1/tan(B) < 2√3/3 -/
theorem acute_triangle_tangent_difference_range 
  (A B C : ℝ) (a b c : ℝ) 
  (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sides : b^2 - a^2 = a*c) :
  1 < 1 / Real.tan A - 1 / Real.tan B ∧ 
  1 / Real.tan A - 1 / Real.tan B < 2 * Real.sqrt 3 / 3 := by
  sorry

end acute_triangle_tangent_difference_range_l3221_322195


namespace zoo_cost_theorem_l3221_322182

def zoo_cost (goat_price : ℚ) (goat_count : ℕ) (llama_price_factor : ℚ) 
              (kangaroo_price_factor : ℚ) (kangaroo_multiple : ℕ) 
              (discount_rate : ℚ) : ℚ :=
  let llama_count := 2 * goat_count
  let kangaroo_count := kangaroo_multiple * 5
  let llama_price := goat_price * (1 + llama_price_factor)
  let kangaroo_price := llama_price * (1 - kangaroo_price_factor)
  let goat_cost := goat_price * goat_count
  let llama_cost := llama_price * llama_count
  let kangaroo_cost := kangaroo_price * kangaroo_count
  let total_cost := goat_cost + llama_cost + kangaroo_cost
  let discounted_cost := total_cost * (1 - discount_rate)
  discounted_cost

theorem zoo_cost_theorem : 
  zoo_cost 400 3 (1/2) (1/4) 2 (1/10) = 8850 := by sorry

end zoo_cost_theorem_l3221_322182


namespace line_l_passes_through_fixed_point_chord_length_y_axis_shortest_chord_equation_l3221_322160

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 16

-- Define the line l
def line_l (x y a : ℝ) : Prop := x - a * y + 3 * a - 2 = 0

-- Statement A
theorem line_l_passes_through_fixed_point (a : ℝ) :
  ∃ x y, line_l x y a ∧ x = 2 ∧ y = 3 :=
sorry

-- Statement B
theorem chord_length_y_axis :
  ∃ y₁ y₂, circle_C 0 y₁ ∧ circle_C 0 y₂ ∧ y₂ - y₁ = 2 * Real.sqrt 15 :=
sorry

-- Statement D
theorem shortest_chord_equation (a : ℝ) :
  (∀ x y, line_l x y a → circle_C x y → 
    ∀ x' y', line_l x' y' a → circle_C x' y' → 
      (x - x')^2 + (y - y')^2 ≤ (x - (-1))^2 + (y - 1)^2) →
  ∃ k, a = -3/2 ∧ k * (3 * x + 2 * y - 12) = x - a * y + 3 * a - 2 :=
sorry

end line_l_passes_through_fixed_point_chord_length_y_axis_shortest_chord_equation_l3221_322160


namespace magazine_cost_lynne_magazine_cost_l3221_322159

/-- The cost of each magazine given Lynne's purchase details -/
theorem magazine_cost (cat_books : ℕ) (solar_books : ℕ) (magazines : ℕ) 
  (book_price : ℕ) (total_spent : ℕ) : ℕ :=
  let total_books := cat_books + solar_books
  let book_cost := total_books * book_price
  let magazine_total_cost := total_spent - book_cost
  magazine_total_cost / magazines

/-- Proof that each magazine costs $4 given Lynne's purchase details -/
theorem lynne_magazine_cost : 
  magazine_cost 7 2 3 7 75 = 4 := by
  sorry

end magazine_cost_lynne_magazine_cost_l3221_322159


namespace remainder_theorem_l3221_322107

theorem remainder_theorem : ∃ q : ℕ, 
  2^206 + 206 = q * (2^103 + 2^53 + 1) + 205 := by
  sorry

end remainder_theorem_l3221_322107


namespace correct_categorization_l3221_322193

def numbers : List ℚ := [15, -3/8, 0, 0.15, -30, -12.8, 22/5, 20]

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n
def is_fraction (q : ℚ) : Prop := ¬(is_integer q)
def is_positive_integer (q : ℚ) : Prop := is_integer q ∧ q > 0
def is_negative_fraction (q : ℚ) : Prop := is_fraction q ∧ q < 0
def is_non_negative (q : ℚ) : Prop := q ≥ 0

def integer_set : Set ℚ := {q ∈ numbers | is_integer q}
def fraction_set : Set ℚ := {q ∈ numbers | is_fraction q}
def positive_integer_set : Set ℚ := {q ∈ numbers | is_positive_integer q}
def negative_fraction_set : Set ℚ := {q ∈ numbers | is_negative_fraction q}
def non_negative_set : Set ℚ := {q ∈ numbers | is_non_negative q}

theorem correct_categorization :
  integer_set = {15, 0, -30, 20} ∧
  fraction_set = {-3/8, 0.15, -12.8, 22/5} ∧
  positive_integer_set = {15, 20} ∧
  negative_fraction_set = {-3/8, -12.8} ∧
  non_negative_set = {15, 0, 0.15, 22/5, 20} := by
  sorry

end correct_categorization_l3221_322193


namespace hyperbola_equation_l3221_322151

theorem hyperbola_equation (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) :
  f1 = (0, 5) →
  f2 = (0, -5) →
  p = (2, 3 * Real.sqrt 5 / 2) →
  ∃ (a b : ℝ),
    a^2 = 9 ∧
    b^2 = 16 ∧
    ∀ (x y : ℝ),
      (y^2 / a^2) - (x^2 / b^2) = 1 ↔
      (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 4 * a^2 ∧
      (p.1 - f1.1)^2 + (p.2 - f1.2)^2 - ((p.1 - f2.1)^2 + (p.2 - f2.2)^2) = 4 * a^2 :=
by sorry

end hyperbola_equation_l3221_322151


namespace inequality_proof_l3221_322127

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/b)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end inequality_proof_l3221_322127


namespace pair_conditions_l3221_322130

def satisfies_conditions (a b : ℚ) : Prop :=
  a * b = 24 ∧ a + b > 0

theorem pair_conditions :
  ¬(satisfies_conditions (-6) (-4)) ∧
  (satisfies_conditions 3 8) ∧
  ¬(satisfies_conditions (-3/2) (-16)) ∧
  (satisfies_conditions 2 12) ∧
  (satisfies_conditions (4/3) 18) :=
by sorry

end pair_conditions_l3221_322130


namespace quadratic_function_properties_l3221_322166

/-- A quadratic function that intersects the x-axis at (0,0) and (-2,0) and has a minimum value of -1 -/
def f (x : ℝ) : ℝ := x^2 + 2*x

theorem quadratic_function_properties :
  (f 0 = 0) ∧
  (f (-2) = 0) ∧
  (∃ x₀, ∀ x, f x ≥ f x₀) ∧
  (∃ x₀, f x₀ = -1) ∧
  (∀ x, f x = x^2 + 2*x) := by
  sorry

end quadratic_function_properties_l3221_322166


namespace geometric_sequence_property_l3221_322133

/-- A geometric sequence with its sum sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  geom : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)  -- Geometric property
  sum : ∀ n : ℕ, S n = (a 0 * (1 - (a 1 / a 0)^n)) / (1 - (a 1 / a 0))  -- Sum formula

/-- Theorem: If S_4 / S_2 = 3 for a geometric sequence, then 2a_2 - a_4 = 0 -/
theorem geometric_sequence_property (seq : GeometricSequence) 
  (h : seq.S 4 / seq.S 2 = 3) : 2 * seq.a 2 - seq.a 4 = 0 := by
  sorry


end geometric_sequence_property_l3221_322133


namespace right_triangle_hypotenuse_l3221_322186

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by
  sorry

end right_triangle_hypotenuse_l3221_322186


namespace tangent_line_at_point_one_zero_l3221_322110

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_point_one_zero :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = x - 1 :=
by sorry

end tangent_line_at_point_one_zero_l3221_322110


namespace collinear_dots_probability_l3221_322148

/-- The number of dots in each row or column of the grid -/
def grid_size : ℕ := 5

/-- The total number of dots in the grid -/
def total_dots : ℕ := grid_size * grid_size

/-- The number of dots to be selected -/
def selected_dots : ℕ := 4

/-- The number of sets of collinear dots -/
def collinear_sets : ℕ := 14

/-- The total number of ways to choose 4 dots out of 25 -/
def total_combinations : ℕ := Nat.choose total_dots selected_dots

/-- The probability of selecting 4 collinear dots -/
def collinear_probability : ℚ := collinear_sets / total_combinations

theorem collinear_dots_probability :
  collinear_probability = 7 / 6325 := by sorry

end collinear_dots_probability_l3221_322148


namespace puppies_adopted_theorem_l3221_322122

/-- The number of puppies adopted each day from a shelter -/
def puppies_adopted_per_day (initial_puppies additional_puppies adoption_days : ℕ) : ℕ :=
  (initial_puppies + additional_puppies) / adoption_days

/-- Theorem stating the number of puppies adopted each day -/
theorem puppies_adopted_theorem (initial_puppies additional_puppies adoption_days : ℕ) 
  (h1 : initial_puppies = 5)
  (h2 : additional_puppies = 35)
  (h3 : adoption_days = 5) :
  puppies_adopted_per_day initial_puppies additional_puppies adoption_days = 8 := by
  sorry

end puppies_adopted_theorem_l3221_322122


namespace prob_three_odd_dice_l3221_322164

def num_dice : ℕ := 5
def num_odd : ℕ := 3

theorem prob_three_odd_dice :
  (num_dice.choose num_odd : ℚ) * (1 / 2) ^ num_dice = 5 / 16 := by
  sorry

end prob_three_odd_dice_l3221_322164


namespace arg_cube_equals_pi_l3221_322187

theorem arg_cube_equals_pi (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 3)
  (h₂ : Complex.abs z₂ = 5)
  (h₃ : Complex.abs (z₁ + z₂) = 7) :
  (Complex.arg (z₁ - z₂))^3 = π := by
  sorry

end arg_cube_equals_pi_l3221_322187


namespace sequence_bound_l3221_322170

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i, 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j, i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end sequence_bound_l3221_322170


namespace unique_solution_for_euler_equation_l3221_322199

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- The statement to prove -/
theorem unique_solution_for_euler_equation :
  ∀ a n : ℕ, a ≠ 0 ∧ n ≠ 0 → (φ (a^n + n) = 2^n) → (a = 2 ∧ n = 1) :=
sorry

end unique_solution_for_euler_equation_l3221_322199


namespace equation_holds_l3221_322155

theorem equation_holds (a b : ℝ) : a^2 - b^2 - (-2*b^2) = a^2 + b^2 := by
  sorry

end equation_holds_l3221_322155


namespace unique_intersection_points_l3221_322118

/-- The set of values k for which |z - 2| = 3|z + 2| intersects |z| = k in exactly one point -/
def intersection_points : Set ℝ :=
  {1.5, 4.5, 5.5}

/-- Predicate to check if a complex number satisfies |z - 2| = 3|z + 2| -/
def satisfies_equation (z : ℂ) : Prop :=
  Complex.abs (z - 2) = 3 * Complex.abs (z + 2)

/-- Predicate to check if a complex number has magnitude k -/
def has_magnitude (z : ℂ) (k : ℝ) : Prop :=
  Complex.abs z = k

/-- The theorem stating that the intersection_points set contains all values of k
    for which |z - 2| = 3|z + 2| intersects |z| = k in exactly one point -/
theorem unique_intersection_points :
  ∀ k : ℝ, (∃! z : ℂ, satisfies_equation z ∧ has_magnitude z k) ↔ k ∈ intersection_points :=
by sorry

end unique_intersection_points_l3221_322118


namespace inequality_proof_l3221_322174

theorem inequality_proof (x a : ℝ) (h : x > a ∧ a > 0) : x^2 > x*a ∧ x*a > a^2 := by
  sorry

end inequality_proof_l3221_322174


namespace unique_rectangle_dimensions_l3221_322184

theorem unique_rectangle_dimensions : 
  ∃! (a b : ℕ), 
    b > a ∧ 
    a > 0 ∧ 
    b > 0 ∧
    (a - 4) * (b - 4) = 2 * (a * b) / 3 := by
  sorry

end unique_rectangle_dimensions_l3221_322184


namespace nikitas_claim_incorrect_l3221_322169

theorem nikitas_claim_incorrect : ¬∃ (x y : ℕ), 5 * (x - y) = 49 := by
  sorry

end nikitas_claim_incorrect_l3221_322169


namespace additional_cars_needed_l3221_322152

def current_cars : ℕ := 37
def cars_per_row : ℕ := 8

theorem additional_cars_needed : 
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (current_cars + n) % cars_per_row = 0 ∧
    ∀ (m : ℕ), m < n → (current_cars + m) % cars_per_row ≠ 0 :=
by
  sorry

end additional_cars_needed_l3221_322152


namespace fifth_term_of_geometric_sequence_l3221_322154

-- Define a positive geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ (n : ℕ), a (n + 1) = a n * r

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_condition1 : a 1 * a 8 = 4 * a 5)
  (h_condition2 : (a 4 + 2 * a 6) / 2 = 18) :
  a 5 = 16 := by
  sorry

end fifth_term_of_geometric_sequence_l3221_322154


namespace least_product_of_distinct_primes_greater_than_10_l3221_322134

theorem least_product_of_distinct_primes_greater_than_10 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧
    p > 10 ∧ q > 10 ∧
    p ≠ q ∧
    p * q = 143 ∧
    ∀ a b : ℕ, a.Prime → b.Prime → a > 10 → b > 10 → a ≠ b → a * b ≥ 143 :=
by sorry

end least_product_of_distinct_primes_greater_than_10_l3221_322134


namespace polygon_25_sides_diagonals_l3221_322109

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem polygon_25_sides_diagonals : num_diagonals 25 = 275 := by
  sorry

end polygon_25_sides_diagonals_l3221_322109


namespace standard_equation_min_area_OPQ_l3221_322139

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the right focus F
def right_focus (a b : ℝ) : Prop := a^2 - b^2 = 1

-- Define the perpendicular condition
def perpendicular_condition (a b : ℝ) : Prop := b = 1

-- Theorem for the standard equation of the ellipse
theorem standard_equation (a b : ℝ) 
  (h1 : ellipse x y a b) 
  (h2 : right_focus a b) 
  (h3 : perpendicular_condition a b) : 
  x^2 / 2 + y^2 = 1 := by sorry

-- Define the triangle OPQ
def triangle_OPQ (x y m : ℝ) : Prop := 
  x^2 / 2 + y^2 = 1 ∧ 
  ∃ (P : ℝ × ℝ), P.2 = 2 ∧ 
  (P.1 * y = 2 * x ∨ (P.1 = 0 ∧ x = Real.sqrt 2))

-- Theorem for the minimum area of triangle OPQ
theorem min_area_OPQ (x y m : ℝ) 
  (h : triangle_OPQ x y m) : 
  ∃ (S : ℝ), S ≥ 1 ∧ 
  (∀ (S' : ℝ), triangle_OPQ x y m → S' ≥ S) := by sorry

end standard_equation_min_area_OPQ_l3221_322139


namespace shifted_quadratic_roots_l3221_322102

theorem shifted_quadratic_roots
  (b c : ℝ)
  (h1 : ∃ x1 x2 : ℝ, x1 = 2 ∧ x2 = -3 ∧ ∀ x, x^2 + b*x + c = 0 ↔ x = x1 ∨ x = x2) :
  ∃ y1 y2 : ℝ, y1 = 6 ∧ y2 = 1 ∧ ∀ x, (x-4)^2 + b*(x-4) + c = 0 ↔ x = y1 ∨ x = y2 :=
sorry

end shifted_quadratic_roots_l3221_322102


namespace problem_solution_l3221_322137

-- Custom operation
def star (x y : ℕ) : ℕ := x * y + 1

-- Prime number function
def nth_prime (n : ℕ) : ℕ := sorry

-- Product function
def product_to_n (n : ℕ) : ℚ := sorry

-- Area of inscribed square
def inscribed_square_area (r : ℝ) : ℝ := sorry

theorem problem_solution :
  (star (star 2 4) 2 = 19) ∧
  (nth_prime 8 = 19) ∧
  (product_to_n 50 = 1 / 50) ∧
  (inscribed_square_area 10 = 200) :=
by sorry

end problem_solution_l3221_322137


namespace halfway_distance_theorem_l3221_322103

def errand_distances : List ℕ := [10, 15, 5]

theorem halfway_distance_theorem (distances : List ℕ) (h : distances = errand_distances) :
  (distances.sum / 2 : ℕ) = 15 := by sorry

end halfway_distance_theorem_l3221_322103


namespace grid_filling_exists_l3221_322131

/-- A function representing the grid filling -/
def GridFilling (n : ℕ) := Fin n → Fin n → Fin (2*n - 1)

/-- Predicate to check if a number is a power of 2 -/
def IsPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- Predicate to check if the grid filling is valid -/
def IsValidFilling (n : ℕ) (f : GridFilling n) : Prop :=
  (∀ k : Fin n, ∀ i j : Fin n, i ≠ j → f k i ≠ f k j) ∧
  (∀ k : Fin n, ∀ i j : Fin n, i ≠ j → f i k ≠ f j k)

theorem grid_filling_exists (n : ℕ) (h : IsPowerOfTwo n) :
  ∃ f : GridFilling n, IsValidFilling n f :=
sorry

end grid_filling_exists_l3221_322131


namespace steve_initial_berries_l3221_322145

/-- Proves that Steve started with 21 berries given the conditions of the problem -/
theorem steve_initial_berries :
  ∀ (stacy_initial steve_initial : ℕ),
    stacy_initial = 32 →
    steve_initial + 4 = stacy_initial - 7 →
    steve_initial = 21 :=
by
  sorry

end steve_initial_berries_l3221_322145


namespace price_adjustment_l3221_322165

theorem price_adjustment (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let increased_price := original_price * (1 + 30 / 100)
  let decrease_factor := 3 / 13
  increased_price * (1 - decrease_factor) = original_price :=
by sorry

end price_adjustment_l3221_322165


namespace sum_of_valid_numbers_l3221_322185

def digits : List ℕ := [1, 3, 5, 7]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  ∃ (a b c d : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  n = 1000 * a + 100 * b + 10 * c + d

def valid_numbers : List ℕ := sorry

theorem sum_of_valid_numbers :
  (List.length valid_numbers = 24) ∧
  (List.sum valid_numbers = 106656) :=
sorry

end sum_of_valid_numbers_l3221_322185


namespace negative_difference_l3221_322192

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end negative_difference_l3221_322192


namespace dresses_total_l3221_322117

/-- The total number of dresses for Emily, Melissa, and Debora -/
def total_dresses (emily_dresses melissa_dresses debora_dresses : ℕ) : ℕ :=
  emily_dresses + melissa_dresses + debora_dresses

/-- Theorem stating the total number of dresses given the conditions -/
theorem dresses_total (emily_dresses : ℕ) 
  (h1 : emily_dresses = 16)
  (h2 : ∃ (melissa_dresses : ℕ), melissa_dresses = emily_dresses / 2)
  (h3 : ∃ (debora_dresses : ℕ), debora_dresses = emily_dresses / 2 + 12) :
  ∃ (total : ℕ), total = total_dresses emily_dresses (emily_dresses / 2) (emily_dresses / 2 + 12) ∧ total = 44 :=
by
  sorry

end dresses_total_l3221_322117


namespace retail_price_increase_l3221_322141

theorem retail_price_increase (manufacturing_cost : ℝ) (retailer_price : ℝ) (customer_price : ℝ)
  (h1 : customer_price = retailer_price * 1.3)
  (h2 : customer_price = manufacturing_cost * 1.82) :
  (retailer_price - manufacturing_cost) / manufacturing_cost = 0.4 := by
sorry

end retail_price_increase_l3221_322141


namespace books_minus_figures_equals_two_l3221_322124

/-- The number of books on Jerry's shelf -/
def initial_books : ℕ := 7

/-- The initial number of action figures on Jerry's shelf -/
def initial_action_figures : ℕ := 3

/-- The number of action figures Jerry added later -/
def added_action_figures : ℕ := 2

/-- The total number of action figures after addition -/
def total_action_figures : ℕ := initial_action_figures + added_action_figures

theorem books_minus_figures_equals_two :
  initial_books - total_action_figures = 2 := by
  sorry

end books_minus_figures_equals_two_l3221_322124


namespace inverse_variation_problem_l3221_322156

/-- Given that 5y varies inversely as the square of x, and y = 4 when x = 2, 
    prove that y = 1 when x = 4 -/
theorem inverse_variation_problem (k : ℝ) :
  (∀ x y : ℝ, x ≠ 0 → 5 * y = k / (x ^ 2)) →
  (5 * 4 = k / (2 ^ 2)) →
  ∃ y : ℝ, 5 * y = k / (4 ^ 2) ∧ y = 1 :=
by sorry

end inverse_variation_problem_l3221_322156


namespace only_satisfying_sets_l3221_322172

/-- A set of four real numbers satisfying the given condition -/
def SatisfyingSet (a b c d : ℝ) : Prop :=
  a + b*c*d = 2 ∧ b + a*c*d = 2 ∧ c + a*b*d = 2 ∧ d + a*b*c = 2

/-- The theorem stating the only satisfying sets -/
theorem only_satisfying_sets :
  ∀ a b c d : ℝ, SatisfyingSet a b c d ↔ 
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
    (a = -1 ∧ b = -1 ∧ c = -1 ∧ d = 3) ∨
    (a = -1 ∧ b = -1 ∧ c = 3 ∧ d = -1) ∨
    (a = -1 ∧ b = 3 ∧ c = -1 ∧ d = -1) ∨
    (a = 3 ∧ b = -1 ∧ c = -1 ∧ d = -1) :=
by sorry

end only_satisfying_sets_l3221_322172


namespace photo_arrangement_probability_photo_arrangement_probability_is_one_twentieth_l3221_322167

/-- The probability that in a group of six students with distinct heights,
    arranged in two rows of three each, every student in the back row
    is taller than every student in the front row. -/
theorem photo_arrangement_probability : ℚ :=
  let n_students : ℕ := 6
  let n_per_row : ℕ := 3
  let total_arrangements : ℕ := n_students.factorial
  let favorable_arrangements : ℕ := (n_per_row.factorial) * (n_per_row.factorial)
  favorable_arrangements / total_arrangements

/-- Proof that the probability is 1/20 -/
theorem photo_arrangement_probability_is_one_twentieth :
  photo_arrangement_probability = 1 / 20 := by
  sorry

end photo_arrangement_probability_photo_arrangement_probability_is_one_twentieth_l3221_322167


namespace total_balls_theorem_l3221_322123

/-- The number of balls of wool used for a single item -/
def balls_per_item : String → ℕ
  | "scarf" => 3
  | "sweater" => 4
  | "hat" => 2
  | "mittens" => 1
  | _ => 0

/-- The number of items made by Aaron -/
def aaron_items : String → ℕ
  | "scarf" => 10
  | "sweater" => 5
  | "hat" => 6
  | _ => 0

/-- The number of items made by Enid -/
def enid_items : String → ℕ
  | "sweater" => 8
  | "hat" => 12
  | "mittens" => 4
  | _ => 0

/-- The total number of balls of wool used by both Enid and Aaron -/
def total_balls_used : ℕ := 
  (aaron_items "scarf" * balls_per_item "scarf") +
  (aaron_items "sweater" * balls_per_item "sweater") +
  (aaron_items "hat" * balls_per_item "hat") +
  (enid_items "sweater" * balls_per_item "sweater") +
  (enid_items "hat" * balls_per_item "hat") +
  (enid_items "mittens" * balls_per_item "mittens")

theorem total_balls_theorem : total_balls_used = 122 := by
  sorry

end total_balls_theorem_l3221_322123


namespace julia_balls_count_l3221_322198

/-- The number of balls Julia bought -/
def total_balls (red_packs yellow_packs green_packs balls_per_pack : ℕ) : ℕ :=
  (red_packs + yellow_packs + green_packs) * balls_per_pack

/-- Proof that Julia bought 399 balls -/
theorem julia_balls_count :
  total_balls 3 10 8 19 = 399 := by
  sorry

end julia_balls_count_l3221_322198


namespace tracys_candies_l3221_322105

theorem tracys_candies (x : ℕ) : 
  (x % 3 = 0) →  -- x is divisible by 3
  (x % 2 = 0) →  -- x is divisible by 2
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 5) →  -- Tracy's brother took between 1 and 5 candies
  (x / 2 - 30 - k = 3) →  -- Tracy was left with 3 candies after all events
  x = 72 :=
by sorry

#check tracys_candies

end tracys_candies_l3221_322105


namespace marble_problem_l3221_322183

theorem marble_problem (a b : ℚ) : 
  let brian := 3 * a
  let caden := 4 * brian
  let daryl := 6 * caden
  b = 6 →
  a + (brian - b) + caden + daryl = 240 →
  a = 123 / 44 := by
sorry

end marble_problem_l3221_322183


namespace min_episodes_watched_l3221_322128

/-- Represents the number of episodes aired on each day of the week -/
def weekly_schedule : List Nat := [0, 1, 1, 1, 1, 2, 2]

/-- The total number of episodes in the TV series -/
def total_episodes : Nat := 60

/-- The duration of Xiaogao's trip in days -/
def trip_duration : Nat := 17

/-- Calculates the maximum number of episodes that can be aired during the trip -/
def max_episodes_during_trip (schedule : List Nat) (duration : Nat) : Nat :=
  sorry

/-- Theorem: The minimum number of episodes Xiaogao can watch is 39 -/
theorem min_episodes_watched : 
  total_episodes - max_episodes_during_trip weekly_schedule trip_duration = 39 := by
  sorry

end min_episodes_watched_l3221_322128


namespace train_lengths_l3221_322149

/-- Theorem: Train Lengths
Given:
- A bridge of length 800 meters
- Train A takes 45 seconds to cross the bridge
- Train B takes 40 seconds to cross the bridge
- Train A takes 15 seconds to pass a lamp post
- Train B takes 10 seconds to pass a lamp post

Prove that the length of Train A is 400 meters and the length of Train B is 800/3 meters.
-/
theorem train_lengths (bridge_length : ℝ) (time_A_bridge time_B_bridge time_A_post time_B_post : ℝ)
  (h1 : bridge_length = 800)
  (h2 : time_A_bridge = 45)
  (h3 : time_B_bridge = 40)
  (h4 : time_A_post = 15)
  (h5 : time_B_post = 10) :
  ∃ (length_A length_B : ℝ),
    length_A = 400 ∧ length_B = 800 / 3 ∧
    length_A + bridge_length = (length_A / time_A_post) * time_A_bridge ∧
    length_B + bridge_length = (length_B / time_B_post) * time_B_bridge :=
by
  sorry

end train_lengths_l3221_322149


namespace linear_equation_solution_l3221_322132

theorem linear_equation_solution :
  ∃! x : ℝ, 8 * x = 2 * x - 6 :=
by
  use -1
  constructor
  · -- Prove that -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check linear_equation_solution

end linear_equation_solution_l3221_322132


namespace divisibility_equivalence_l3221_322176

theorem divisibility_equivalence (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) := by
  sorry

end divisibility_equivalence_l3221_322176


namespace operation_is_multiplication_l3221_322144

theorem operation_is_multiplication : 
  ((0.137 + 0.098)^2 - (0.137 - 0.098)^2) / (0.137 * 0.098) = 4 := by
  sorry

end operation_is_multiplication_l3221_322144


namespace quadratic_inequality_range_l3221_322120

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - 3*a*x + 9 < 0) ↔ -2 ≤ a ∧ a ≤ 2 := by
  sorry

end quadratic_inequality_range_l3221_322120


namespace quadratic_discriminant_l3221_322175

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 9x + 4 is 1 -/
theorem quadratic_discriminant : discriminant 5 (-9) 4 = 1 := by
  sorry

end quadratic_discriminant_l3221_322175


namespace resort_tips_multiple_l3221_322188

theorem resort_tips_multiple (total_months : Nat) (special_month_fraction : Real) 
  (h1 : total_months = 7)
  (h2 : special_month_fraction = 0.5)
  (average_other_months : Real)
  (special_month_tips : Real)
  (h3 : special_month_tips = special_month_fraction * (average_other_months * (total_months - 1) + special_month_tips))
  (h4 : ∃ (m : Real), special_month_tips = m * average_other_months) :
  ∃ (m : Real), special_month_tips = 6 * average_other_months :=
by sorry

end resort_tips_multiple_l3221_322188


namespace commute_days_calculation_l3221_322191

theorem commute_days_calculation (x : ℕ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : ∃ a b c : ℕ, 
    a + b + c = x ∧  -- Total days
    b + c = 6 ∧      -- Bus to work
    a + c = 18 ∧     -- Bus from work
    a + b = 14) :    -- Train commutes
  x = 19 := by
sorry

end commute_days_calculation_l3221_322191


namespace podium_height_l3221_322143

/-- The height of the podium given two configurations of books -/
theorem podium_height (l w : ℝ) (h : ℝ) : 
  l + h - w = 40 → w + h - l = 34 → h = 37 := by sorry

end podium_height_l3221_322143


namespace roots_of_polynomial_l3221_322116

theorem roots_of_polynomial (x : ℝ) :
  (x^2 - 5*x + 6)*(x - 3)*(2*x - 8) = 0 ↔ x = 2 ∨ x = 3 ∨ x = 4 := by
  sorry

end roots_of_polynomial_l3221_322116


namespace vacation_duration_l3221_322146

theorem vacation_duration (plane_cost hotel_cost_per_day total_cost : ℕ) 
  (h1 : plane_cost = 48)
  (h2 : hotel_cost_per_day = 24)
  (h3 : total_cost = 120) :
  ∃ d : ℕ, d = 3 ∧ plane_cost + hotel_cost_per_day * d = total_cost := by
  sorry

end vacation_duration_l3221_322146


namespace toothpicks_in_specific_grid_l3221_322108

/-- Calculates the number of toothpicks in a modified grid -/
def toothpicks_in_modified_grid (length width corner_size : ℕ) : ℕ :=
  let vertical_lines := length + 1
  let horizontal_lines := width + 1
  let corner_lines := corner_size + 1
  let total_without_corner := vertical_lines * width + horizontal_lines * length
  let corner_toothpicks := corner_lines * corner_size * 2
  total_without_corner - corner_toothpicks

/-- Theorem stating the number of toothpicks in the specific grid described in the problem -/
theorem toothpicks_in_specific_grid :
  toothpicks_in_modified_grid 70 45 5 = 6295 :=
by sorry

end toothpicks_in_specific_grid_l3221_322108


namespace expression_factorization_l3221_322161

theorem expression_factorization (x : ℝ) : 
  (16 * x^7 + 36 * x^4 - 9) - (4 * x^7 - 6 * x^4 - 9) = 6 * x^4 * (2 * x^3 + 7) := by
  sorry

end expression_factorization_l3221_322161


namespace sequence_sum_l3221_322115

theorem sequence_sum (A B C D E F G H I J : ℝ) : 
  D = 8 →
  A + B + C + D = 45 →
  B + C + D + E = 45 →
  C + D + E + F = 45 →
  D + E + F + G = 45 →
  E + F + G + H = 45 →
  F + G + H + I = 45 →
  G + H + I + J = 45 →
  A + J = 0 := by
sorry

end sequence_sum_l3221_322115


namespace unique_solution_g_equals_g_inv_l3221_322100

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - 5

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

-- Theorem statement
theorem unique_solution_g_equals_g_inv :
  ∃! x : ℝ, g x = g_inv x :=
sorry

end unique_solution_g_equals_g_inv_l3221_322100
