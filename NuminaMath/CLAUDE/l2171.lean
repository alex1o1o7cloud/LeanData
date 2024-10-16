import Mathlib

namespace NUMINAMATH_CALUDE_three_digit_ending_l2171_217135

theorem three_digit_ending (N : ℕ) (h1 : N > 0) (h2 : N % 1000 = N^2 % 1000) 
  (h3 : N % 1000 ≥ 100) : N % 1000 = 127 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_ending_l2171_217135


namespace NUMINAMATH_CALUDE_stratified_sampling_equal_probability_l2171_217165

/-- Represents a stratified sampling setup -/
structure StratifiedSampling where
  population : Type
  strata : Type
  num_layers : ℕ
  stratification : population → strata

/-- The probability of an individual being sampled in stratified sampling -/
def sample_probability (ss : StratifiedSampling) (individual : ss.population) : ℝ :=
  sorry

/-- Theorem stating that the sample probability is independent of the number of layers and stratification -/
theorem stratified_sampling_equal_probability (ss : StratifiedSampling) 
  (individual1 individual2 : ss.population) :
  sample_probability ss individual1 = sample_probability ss individual2 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_equal_probability_l2171_217165


namespace NUMINAMATH_CALUDE_problem_solution_l2171_217132

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - |x - 2|

-- State the theorem
theorem problem_solution (m : ℝ) (a b c x y z : ℝ) 
  (h1 : ∀ x, f m (x + 1) ≥ 0 ↔ 0 ≤ x ∧ x ≤ 1)
  (h2 : x^2 + y^2 + z^2 = a^2 + b^2 + c^2)
  (h3 : x^2 + y^2 + z^2 = m) :
  m = 1 ∧ a*x + b*y + c*z ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2171_217132


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2171_217102

theorem angle_between_vectors (a b : ℝ × ℝ) :
  (∀ x y : ℝ, a.1 * x + a.2 * y = 1) →  -- a is a unit vector
  b = (2, 2 * Real.sqrt 3) →           -- b = (2, 2√3)
  a.1 * (2 * a.1 + b.1) + a.2 * (2 * a.2 + b.2) = 0 →  -- a ⟂ (2a + b)
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2171_217102


namespace NUMINAMATH_CALUDE_at_least_one_zero_negation_l2171_217181

theorem at_least_one_zero_negation (a b : ℝ) :
  ¬(a = 0 ∨ b = 0) ↔ (a ≠ 0 ∧ b ≠ 0) := by sorry

end NUMINAMATH_CALUDE_at_least_one_zero_negation_l2171_217181


namespace NUMINAMATH_CALUDE_carrot_to_lettuce_ratio_l2171_217151

def lettuce_calories : ℕ := 50
def dressing_calories : ℕ := 210
def pizza_crust_calories : ℕ := 600
def pizza_cheese_calories : ℕ := 400
def total_consumed_calories : ℕ := 330

def pizza_total_calories : ℕ := pizza_crust_calories + (pizza_crust_calories / 3) + pizza_cheese_calories

def salad_calories (carrot_calories : ℕ) : ℕ := lettuce_calories + carrot_calories + dressing_calories

theorem carrot_to_lettuce_ratio :
  ∃ (carrot_calories : ℕ),
    (salad_calories carrot_calories / 4 + pizza_total_calories / 5 = total_consumed_calories) ∧
    (carrot_calories / lettuce_calories = 2) := by
  sorry

end NUMINAMATH_CALUDE_carrot_to_lettuce_ratio_l2171_217151


namespace NUMINAMATH_CALUDE_octagon_area_in_circle_l2171_217161

theorem octagon_area_in_circle (r : ℝ) (h : r = 2.5) : 
  let octagon_area := 8 * (r^2 * Real.sin (π/8) * Real.cos (π/8))
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |octagon_area - 17.672| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_octagon_area_in_circle_l2171_217161


namespace NUMINAMATH_CALUDE_grid_paths_l2171_217152

theorem grid_paths (total_steps : ℕ) (right_steps : ℕ) (up_steps : ℕ) 
  (h1 : total_steps = right_steps + up_steps)
  (h2 : total_steps = 10)
  (h3 : right_steps = 6)
  (h4 : up_steps = 4) :
  Nat.choose total_steps up_steps = 210 := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_l2171_217152


namespace NUMINAMATH_CALUDE_gnuff_tutoring_time_l2171_217120

/-- Proves that given Gnuff's tutoring rates and total amount paid, the number of minutes tutored is 18 --/
theorem gnuff_tutoring_time (flat_rate per_minute_rate total_paid : ℕ) : 
  flat_rate = 20 → 
  per_minute_rate = 7 → 
  total_paid = 146 → 
  (total_paid - flat_rate) / per_minute_rate = 18 := by
sorry

end NUMINAMATH_CALUDE_gnuff_tutoring_time_l2171_217120


namespace NUMINAMATH_CALUDE_average_pen_price_is_correct_l2171_217118

/-- The average price of a pen before discount given the following conditions:
  * 30 pens and 75 pencils were purchased
  * The total cost after a 10% discount is $510
  * The average price of a pencil before discount is $2.00
-/
def averagePenPrice (numPens : ℕ) (numPencils : ℕ) (totalCostAfterDiscount : ℚ) 
  (pencilPrice : ℚ) (discountRate : ℚ) : ℚ :=
  let totalCostBeforeDiscount : ℚ := totalCostAfterDiscount / (1 - discountRate)
  let totalPencilCost : ℚ := numPencils * pencilPrice
  let totalPenCost : ℚ := totalCostBeforeDiscount - totalPencilCost
  totalPenCost / numPens

theorem average_pen_price_is_correct : 
  averagePenPrice 30 75 510 2 (1/10) = 13.89 := by
  sorry

end NUMINAMATH_CALUDE_average_pen_price_is_correct_l2171_217118


namespace NUMINAMATH_CALUDE_jerome_car_ratio_l2171_217124

/-- Represents the number of toy cars Jerome has at different times --/
structure ToyCars where
  initial : ℕ
  bought_last_month : ℕ
  total_now : ℕ

/-- Calculates the number of toy cars bought this month --/
def cars_bought_this_month (tc : ToyCars) : ℕ :=
  tc.total_now - tc.initial - tc.bought_last_month

/-- Represents a ratio as a pair of natural numbers --/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of cars bought this month to last month --/
def buying_ratio (tc : ToyCars) : Ratio :=
  { numerator := cars_bought_this_month tc,
    denominator := tc.bought_last_month }

/-- Theorem stating that the ratio of cars bought this month to last month is 2:1 --/
theorem jerome_car_ratio : 
  ∀ tc : ToyCars, 
  tc.initial = 25 ∧ tc.bought_last_month = 5 ∧ tc.total_now = 40 →
  buying_ratio tc = { numerator := 2, denominator := 1 } := by
  sorry

end NUMINAMATH_CALUDE_jerome_car_ratio_l2171_217124


namespace NUMINAMATH_CALUDE_problem_solution_l2171_217117

theorem problem_solution (a b : ℚ) 
  (h1 : 3 * a + 4 * b = 0) 
  (h2 : a = 2 * b - 3) : 
  9 * a - 6 * b = -81 / 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2171_217117


namespace NUMINAMATH_CALUDE_expression_evaluation_l2171_217167

theorem expression_evaluation : (4 * 5 * 6) * (1/4 + 1/5 - 1/10) = 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2171_217167


namespace NUMINAMATH_CALUDE_quadratic_vertex_l2171_217145

/-- The quadratic function f(x) = 2(x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex -/
def vertex_y : ℝ := 1

/-- Theorem: The vertex of the quadratic function f(x) = 2(x-3)^2 + 1 is at (3,1) -/
theorem quadratic_vertex : 
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l2171_217145


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l2171_217105

theorem smallest_factor_for_perfect_square (n : ℕ) (h : n = 31360) : 
  (∃ (y : ℕ), y > 0 ∧ ∃ (k : ℕ), n * y = k^2) ∧ 
  (∀ (z : ℕ), z > 0 → z < 623 → ¬∃ (k : ℕ), n * z = k^2) ∧
  (∃ (k : ℕ), n * 623 = k^2) := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l2171_217105


namespace NUMINAMATH_CALUDE_f_simplification_symmetry_condition_g_maximum_condition_l2171_217160

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2) * Real.cos (x / 2) - 2 * Real.sqrt 3 * Real.sin (x / 2) ^ 2 + Real.sqrt 3

noncomputable def g (x : ℝ) : ℝ := f x + Real.sin x

theorem f_simplification (x : ℝ) : f x = 2 * Real.sin (x + π / 3) := by sorry

theorem symmetry_condition (φ : ℝ) :
  (∃ k : ℤ, π / 3 + φ + π / 3 = k * π) → φ = π / 3 := by sorry

theorem g_maximum_condition (θ : ℝ) :
  (∀ x : ℝ, g x ≤ g θ) → Real.cos θ = Real.sqrt 3 / Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_f_simplification_symmetry_condition_g_maximum_condition_l2171_217160


namespace NUMINAMATH_CALUDE_race_distance_l2171_217199

/-- The race problem -/
theorem race_distance (t_a t_b : ℝ) (d_diff : ℝ) (h1 : t_a = 20) (h2 : t_b = 25) (h3 : d_diff = 16) :
  ∃ d : ℝ, d > 0 ∧ d / t_a * t_b = d + d_diff := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l2171_217199


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2171_217185

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2171_217185


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l2171_217137

/-- The number of ways to choose a starting lineup from a volleyball team. -/
def starting_lineup_count (total_players : ℕ) (lineup_size : ℕ) (captain_count : ℕ) : ℕ :=
  total_players * (Nat.choose (total_players - 1) (lineup_size - 1))

/-- Theorem: The number of ways to choose a starting lineup of 8 players
    (including one captain) from a team of 18 players is 350,064. -/
theorem volleyball_lineup_count :
  starting_lineup_count 18 8 1 = 350064 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l2171_217137


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_main_theorem_l2171_217153

def numerator : ℕ := 22 * 23 * 24 * 25 * 26 * 27
def denominator : ℕ := 2000

theorem units_digit_of_fraction (n d : ℕ) (h : d ≠ 0) : 
  (n / d) % 10 = ((n % (d * 10)) / d) % 10 :=
sorry

theorem main_theorem : (numerator / denominator) % 10 = 8 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_main_theorem_l2171_217153


namespace NUMINAMATH_CALUDE_american_car_production_l2171_217163

/-- The number of cars American carmakers produce each year -/
def total_cars : ℕ := 5650000

/-- The number of car suppliers -/
def num_suppliers : ℕ := 5

/-- The number of cars the first supplier receives -/
def first_supplier : ℕ := 1000000

/-- The number of cars the second supplier receives -/
def second_supplier : ℕ := first_supplier + 500000

/-- The number of cars the third supplier receives -/
def third_supplier : ℕ := first_supplier + second_supplier

/-- The number of cars the fourth supplier receives -/
def fourth_supplier : ℕ := 325000

/-- The number of cars the fifth supplier receives -/
def fifth_supplier : ℕ := fourth_supplier

theorem american_car_production :
  total_cars = first_supplier + second_supplier + third_supplier + fourth_supplier + fifth_supplier :=
by sorry

end NUMINAMATH_CALUDE_american_car_production_l2171_217163


namespace NUMINAMATH_CALUDE_sin_675_degrees_l2171_217140

theorem sin_675_degrees : Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_675_degrees_l2171_217140


namespace NUMINAMATH_CALUDE_one_prime_in_alternating_series_l2171_217189

/-- The nth number in the alternating 1-0 series -/
def A (n : ℕ) : ℕ := 
  (10^(2*n) - 1) / 99

/-- The series of alternating 1-0 numbers -/
def alternating_series : Set ℕ :=
  {x | ∃ n : ℕ, x = A n}

/-- Theorem: There is exactly one prime number in the alternating 1-0 series -/
theorem one_prime_in_alternating_series : 
  ∃! p, p ∈ alternating_series ∧ Nat.Prime p :=
sorry

end NUMINAMATH_CALUDE_one_prime_in_alternating_series_l2171_217189


namespace NUMINAMATH_CALUDE_sea_turtle_count_sea_turtle_count_proof_l2171_217115

theorem sea_turtle_count : ℕ → Prop :=
  fun total_turtles =>
    (total_turtles : ℚ) * (1 : ℚ) / (3 : ℚ) + (28 : ℚ) = total_turtles ∧
    total_turtles = 42

-- Proof
theorem sea_turtle_count_proof : sea_turtle_count 42 := by
  sorry

end NUMINAMATH_CALUDE_sea_turtle_count_sea_turtle_count_proof_l2171_217115


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l2171_217110

theorem alcohol_mixture_problem (A W : ℝ) :
  A / W = 2 / 5 →
  A / (W + 10) = 2 / 7 →
  A = 10 := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l2171_217110


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l2171_217164

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem stating that f is monotonically decreasing on (-∞, 1]
theorem f_monotone_decreasing :
  MonotoneOn f (Set.Iic 1) := by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l2171_217164


namespace NUMINAMATH_CALUDE_average_of_five_quantities_l2171_217121

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 33) : 
  (q1 + q2 + q3 + q4 + q5) / 5 = 15.6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_quantities_l2171_217121


namespace NUMINAMATH_CALUDE_ratio_equality_l2171_217113

/-- Sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2018 / (n + 1) * a (n + 1) + a n

/-- Sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2020 / (n + 1) * b (n + 1) + b n

/-- Theorem stating the equality of the ratio of specific terms in sequences a and b -/
theorem ratio_equality : a 1010 / 1010 = b 1009 / 1009 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2171_217113


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l2171_217198

theorem unique_solution_factorial_equation :
  ∃! (n : ℕ), n * n.factorial + 2 * n.factorial = 5040 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l2171_217198


namespace NUMINAMATH_CALUDE_circles_properties_l2171_217186

-- Define the circles O and M
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 1 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_M A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_M B.1 B.2 ∧
  A ≠ B

-- Define the theorem
theorem circles_properties 
  (A B : ℝ × ℝ) 
  (h : intersection_points A B) :
  (∃ (T1 T2 : ℝ × ℝ), T1 ≠ T2 ∧ 
    (∀ x y, circle_O x y → (x - T1.1) * T1.1 + (y - T1.2) * T1.2 = 0) ∧
    (∀ x y, circle_M x y → (x - T1.1) * T1.1 + (y - T1.2) * T1.2 = 0) ∧
    (∀ x y, circle_O x y → (x - T2.1) * T2.1 + (y - T2.2) * T2.2 = 0) ∧
    (∀ x y, circle_M x y → (x - T2.1) * T2.1 + (y - T2.2) * T2.2 = 0)) ∧
  (∀ x y, circle_O x y ↔ circle_M (2*A.1 - x) (2*A.2 - y)) ∧
  (∃ E F : ℝ × ℝ, circle_O E.1 E.2 ∧ circle_M F.1 F.2 ∧
    ∀ E' F' : ℝ × ℝ, circle_O E'.1 E'.2 → circle_M F'.1 F'.2 →
      (E.1 - F.1)^2 + (E.2 - F.2)^2 ≥ (E'.1 - F'.1)^2 + (E'.2 - F'.2)^2) ∧
  (∃ E F : ℝ × ℝ, circle_O E.1 E.2 ∧ circle_M F.1 F.2 ∧
    (E.1 - F.1)^2 + (E.2 - F.2)^2 = (4 + Real.sqrt 5)^2) :=
sorry

end NUMINAMATH_CALUDE_circles_properties_l2171_217186


namespace NUMINAMATH_CALUDE_sector_central_angle_l2171_217179

theorem sector_central_angle (area : Real) (radius : Real) (central_angle : Real) :
  area = 3 * Real.pi / 8 →
  radius = 1 →
  area = 1 / 2 * central_angle * radius ^ 2 →
  central_angle = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2171_217179


namespace NUMINAMATH_CALUDE_unique_triple_solution_l2171_217126

theorem unique_triple_solution (x y p : ℕ+) (h_prime : Nat.Prime p) 
  (h_p : p = x^2 + 1) (h_y : 2*p^2 = y^2 + 1) : 
  (x = 2 ∧ y = 7 ∧ p = 5) := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l2171_217126


namespace NUMINAMATH_CALUDE_chocolates_in_box_l2171_217197

/-- Represents the dimensions of a cuboid -/
structure Dimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def volume (d : Dimensions) : ℝ :=
  d.width * d.length * d.height

/-- The dimensions of the box -/
def box_dimensions : Dimensions :=
  { width := 30, length := 20, height := 5 }

/-- The dimensions of a single chocolate -/
def chocolate_dimensions : Dimensions :=
  { width := 6, length := 4, height := 1 }

/-- Theorem stating that the number of chocolates in the box is 125 -/
theorem chocolates_in_box :
  (volume box_dimensions) / (volume chocolate_dimensions) = 125 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_in_box_l2171_217197


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2171_217182

/-- Given that (m+1)x^(m^2+1) - 2x - 5 = 0 is a quadratic equation in x 
    and m + 1 ≠ 0, prove that m = 1 -/
theorem quadratic_equation_m_value (m : ℝ) : 
  (∃ a b c : ℝ, ∀ x : ℝ, (m + 1) * x^(m^2 + 1) - 2*x - 5 = a*x^2 + b*x + c) ∧ 
  (m + 1 ≠ 0) → 
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2171_217182


namespace NUMINAMATH_CALUDE_unique_successful_arrangement_l2171_217190

/-- Represents a cell in the table -/
inductive Cell
| One
| NegOne

/-- Represents a square table -/
def Table (n : ℕ) := Fin (2^n - 1) → Fin (2^n - 1) → Cell

/-- Checks if two cells are neighbors -/
def is_neighbor (n : ℕ) (i j i' j' : Fin (2^n - 1)) : Prop :=
  (i = i' ∧ (j.val + 1 = j'.val ∨ j.val = j'.val + 1)) ∨
  (j = j' ∧ (i.val + 1 = i'.val ∨ i.val = i'.val + 1))

/-- Checks if a table is a successful arrangement -/
def is_successful (n : ℕ) (t : Table n) : Prop :=
  ∀ i j, t i j = Cell.One ↔ 
    ∀ i' j', is_neighbor n i j i' j' → t i' j' = Cell.One

/-- The main theorem -/
theorem unique_successful_arrangement (n : ℕ) :
  ∃! t : Table n, is_successful n t ∧ (∀ i j, t i j = Cell.One) :=
sorry

end NUMINAMATH_CALUDE_unique_successful_arrangement_l2171_217190


namespace NUMINAMATH_CALUDE_rectangle_breadth_unchanged_l2171_217171

theorem rectangle_breadth_unchanged 
  (L B : ℝ) 
  (h1 : L > 0) 
  (h2 : B > 0) 
  (new_L : ℝ) 
  (h3 : new_L = L / 2) 
  (new_A : ℝ) 
  (h4 : new_A = L * B / 2) :
  ∃ (new_B : ℝ), new_A = new_L * new_B ∧ new_B = B := by
sorry

end NUMINAMATH_CALUDE_rectangle_breadth_unchanged_l2171_217171


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l2171_217166

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: Area of the fourth rectangle in a divided large rectangle -/
theorem fourth_rectangle_area
  (a b : ℝ)
  (r1 r2 r3 r4 : Rectangle)
  (h1 : r1.width = 2*a ∧ r1.height = b)
  (h2 : r2.width = 3*a ∧ r2.height = b)
  (h3 : r3.width = 2*a ∧ r3.height = 2*b)
  (h4 : r4.width = 3*a ∧ r4.height = 2*b)
  (area1 : area r1 = 2*a*b)
  (area2 : area r2 = 6*a*b)
  (area3 : area r3 = 4*a*b) :
  area r4 = 6*a*b :=
sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l2171_217166


namespace NUMINAMATH_CALUDE_fishing_tomorrow_count_l2171_217125

/-- Represents the fishing patterns in a coastal village -/
structure FishingVillage where
  daily : Nat        -- Number of people fishing daily
  everyOtherDay : Nat -- Number of people fishing every other day
  everyThreeDays : Nat -- Number of people fishing every three days
  yesterdayCount : Nat -- Number of people who fished yesterday
  todayCount : Nat     -- Number of people fishing today

/-- Calculates the number of people who will fish tomorrow -/
def tomorrowFishers (village : FishingVillage) : Nat :=
  sorry

/-- Theorem stating that given the specific fishing patterns and counts for yesterday and today,
    15 people will fish tomorrow -/
theorem fishing_tomorrow_count (village : FishingVillage) 
  (h1 : village.daily = 7)
  (h2 : village.everyOtherDay = 8)
  (h3 : village.everyThreeDays = 3)
  (h4 : village.yesterdayCount = 12)
  (h5 : village.todayCount = 10) :
  tomorrowFishers village = 15 := by
  sorry

end NUMINAMATH_CALUDE_fishing_tomorrow_count_l2171_217125


namespace NUMINAMATH_CALUDE_smallest_binary_divisible_by_225_l2171_217177

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_divisible_by_225 :
  ∃ (n : ℕ), is_binary_number n ∧ 225 ∣ n ∧
  ∀ (m : ℕ), is_binary_number m → 225 ∣ m → n ≤ m :=
by
  -- The proof would go here
  sorry

#eval (11111111100 : ℕ).digits 10  -- To verify the number in base 10
#eval 11111111100 % 225  -- To verify divisibility by 225

end NUMINAMATH_CALUDE_smallest_binary_divisible_by_225_l2171_217177


namespace NUMINAMATH_CALUDE_race_finish_time_difference_l2171_217111

theorem race_finish_time_difference :
  ∀ (total_runners : ℕ) 
    (fast_runners : ℕ) 
    (slow_runners : ℕ) 
    (fast_time : ℝ) 
    (total_time : ℝ),
  total_runners = fast_runners + slow_runners →
  total_runners = 8 →
  fast_runners = 5 →
  fast_time = 8 →
  total_time = 70 →
  ∃ (slow_time : ℝ),
    total_time = fast_runners * fast_time + slow_runners * slow_time ∧
    slow_time - fast_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_race_finish_time_difference_l2171_217111


namespace NUMINAMATH_CALUDE_oil_price_reduction_percentage_l2171_217141

/-- Proves that the percentage reduction in oil price is 25% given the specified conditions --/
theorem oil_price_reduction_percentage (original_price reduced_price : ℚ) : 
  reduced_price = 50 →
  (1000 / reduced_price) - (1000 / original_price) = 5 →
  (original_price - reduced_price) / original_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_percentage_l2171_217141


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2171_217162

theorem unique_quadratic_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + a * x + 1 = 0) ↔ a = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2171_217162


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2171_217174

theorem imaginary_part_of_z (z : ℂ) : 
  (z * (1 + Complex.I) * Complex.I^3) / (1 - Complex.I) = 1 - Complex.I →
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2171_217174


namespace NUMINAMATH_CALUDE_line_parameterization_l2171_217119

/-- Given a line y = -2x + 7 parameterized by (x, y) = (p, 3) + t(6, l),
    prove that p = 2 and l = -12 -/
theorem line_parameterization (x y p l t : ℝ) : 
  (y = -2 * x + 7) →
  (x = p + 6 * t ∧ y = 3 + l * t) →
  (p = 2 ∧ l = -12) := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l2171_217119


namespace NUMINAMATH_CALUDE_total_dolls_count_l2171_217170

theorem total_dolls_count (big_box_capacity : ℕ) (small_box_capacity : ℕ) 
                          (big_box_count : ℕ) (small_box_count : ℕ) 
                          (h1 : big_box_capacity = 7)
                          (h2 : small_box_capacity = 4)
                          (h3 : big_box_count = 5)
                          (h4 : small_box_count = 9) :
  big_box_capacity * big_box_count + small_box_capacity * small_box_count = 71 :=
by sorry

end NUMINAMATH_CALUDE_total_dolls_count_l2171_217170


namespace NUMINAMATH_CALUDE_book_purchase_ratio_l2171_217114

/-- The number of people who purchased both books A and B -/
def both : ℕ := 500

/-- The number of people who purchased only book A -/
def only_A : ℕ := 1000

/-- The number of people who purchased only book B -/
def only_B : ℕ := both / 2

/-- The total number of people who purchased book A -/
def total_A : ℕ := only_A + both

/-- The total number of people who purchased book B -/
def total_B : ℕ := only_B + both

/-- The ratio of people who purchased book A to those who purchased book B is 2:1 -/
theorem book_purchase_ratio : total_A / total_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_ratio_l2171_217114


namespace NUMINAMATH_CALUDE_complex_angle_proof_l2171_217144

theorem complex_angle_proof (z : ℂ) : z = -1 - Real.sqrt 3 * I → ∃ r θ : ℝ, z = r * Complex.exp (θ * I) ∧ θ = (4 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_angle_proof_l2171_217144


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l2171_217191

theorem ratio_equation_solution (c d : ℚ) 
  (h1 : c / d = 4)
  (h2 : c = 15 - 3 * d) : 
  d = 15 / 7 := by sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l2171_217191


namespace NUMINAMATH_CALUDE_camera_filter_savings_percentage_l2171_217142

theorem camera_filter_savings_percentage : 
  let kit_price : ℚ := 144.20
  let filter_prices : List ℚ := [21.75, 21.75, 18.60, 18.60, 23.80, 29.35, 29.35]
  let total_individual_price : ℚ := filter_prices.sum
  let savings : ℚ := total_individual_price - kit_price
  let savings_percentage : ℚ := (savings / total_individual_price) * 100
  savings_percentage = 11.64 := by sorry

end NUMINAMATH_CALUDE_camera_filter_savings_percentage_l2171_217142


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_of_ones_l2171_217136

/-- Given a natural number n, construct a number consisting of n ones -/
def numberWithOnes (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

theorem sum_of_digits_of_square_of_ones (n : ℕ) :
  sumOfDigits ((numberWithOnes n)^2) = n^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_of_ones_l2171_217136


namespace NUMINAMATH_CALUDE_ball_in_cylinder_l2171_217138

/-- Given a horizontal cylindrical measuring cup with base radius √3 cm and a solid ball
    of radius R cm that is submerged and causes the water level to rise exactly R cm,
    prove that R = 3/2 cm. -/
theorem ball_in_cylinder (R : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * R^3 = Real.pi * 3 * R → R = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ball_in_cylinder_l2171_217138


namespace NUMINAMATH_CALUDE_seed_mixture_problem_l2171_217143

/-- Proves that in a mixture of seed mixtures X and Y, where X is 40% ryegrass
    and Y is 25% ryegrass, if the final mixture contains 35% ryegrass,
    then the percentage of X in the final mixture is 200/3. -/
theorem seed_mixture_problem (x y : ℝ) :
  x + y = 100 →  -- x and y represent percentages of X and Y in the final mixture
  0.40 * x + 0.25 * y = 35 →  -- The final mixture contains 35% ryegrass
  x = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_seed_mixture_problem_l2171_217143


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2171_217175

-- Define the polynomial g(x)
def g (p q r s : ℝ) (x : ℂ) : ℂ := x^4 + p*x^3 + q*x^2 + r*x + s

-- State the theorem
theorem sum_of_coefficients (p q r s : ℝ) :
  (g p q r s (3*I) = 0) →
  (g p q r s (1 + 3*I) = 0) →
  p + q + r + s = 89 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2171_217175


namespace NUMINAMATH_CALUDE_baking_dish_recipe_book_ratio_l2171_217194

/-- The cost of Liz's purchases -/
def total_cost : ℚ := 40

/-- The cost of the recipe book -/
def recipe_book_cost : ℚ := 6

/-- The cost of each ingredient -/
def ingredient_cost : ℚ := 3

/-- The number of ingredients purchased -/
def num_ingredients : ℕ := 5

/-- The additional cost of the apron compared to the recipe book -/
def apron_extra_cost : ℚ := 1

/-- The ratio of the baking dish cost to the recipe book cost -/
def baking_dish_to_recipe_book_ratio : ℚ := 2

theorem baking_dish_recipe_book_ratio :
  (total_cost - (recipe_book_cost + (recipe_book_cost + apron_extra_cost) + 
   (ingredient_cost * num_ingredients))) / recipe_book_cost = baking_dish_to_recipe_book_ratio := by
  sorry

end NUMINAMATH_CALUDE_baking_dish_recipe_book_ratio_l2171_217194


namespace NUMINAMATH_CALUDE_english_to_maths_ratio_l2171_217149

/-- Represents the marks obtained in different subjects -/
structure Marks where
  english : ℕ
  science : ℕ
  maths : ℕ

/-- Represents the ratio between two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of English to Maths marks -/
theorem english_to_maths_ratio (m : Marks) : 
  m.science = 17 ∧ 
  m.english = 3 * m.science ∧ 
  m.english + m.science + m.maths = 170 → 
  ∃ r : Ratio, r.numerator = 1 ∧ r.denominator = 2 ∧ 
    r.numerator * m.maths = r.denominator * m.english :=
by sorry

end NUMINAMATH_CALUDE_english_to_maths_ratio_l2171_217149


namespace NUMINAMATH_CALUDE_unique_pair_sum_28_l2171_217130

theorem unique_pair_sum_28 :
  ∃! (a b : ℕ), a ≠ b ∧ a > 11 ∧ b > 11 ∧ a + b = 28 ∧
  (Even a ∨ Even b) ∧
  (∀ (c d : ℕ), c ≠ d ∧ c > 11 ∧ d > 11 ∧ c + d = 28 ∧ (Even c ∨ Even d) → (c = a ∧ d = b) ∨ (c = b ∧ d = a)) ∧
  a = 12 ∧ b = 16 :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_sum_28_l2171_217130


namespace NUMINAMATH_CALUDE_pyramid_volume_no_conditional_l2171_217176

/-- Algorithm to calculate triangle area from three side lengths -/
def triangle_area (a b c : ℝ) : ℝ := sorry

/-- Algorithm to calculate line slope from two points' coordinates -/
def line_slope (x1 y1 x2 y2 : ℝ) : ℝ := sorry

/-- Algorithm to calculate common logarithm of a number -/
noncomputable def common_log (x : ℝ) : ℝ := sorry

/-- Algorithm to calculate pyramid volume from base area and height -/
def pyramid_volume (base_area height : ℝ) : ℝ := sorry

/-- Predicate to check if an algorithm contains conditional statements -/
def has_conditional {α β : Type} (f : α → β) : Prop := sorry

theorem pyramid_volume_no_conditional :
  ¬ has_conditional pyramid_volume ∧
  has_conditional triangle_area ∧
  has_conditional line_slope ∧
  has_conditional common_log :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_no_conditional_l2171_217176


namespace NUMINAMATH_CALUDE_dividend_calculation_l2171_217128

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h1 : remainder = 6)
  (h2 : divisor = 5 * quotient)
  (h3 : divisor = 3 * remainder + 2) :
  divisor * quotient + remainder = 86 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2171_217128


namespace NUMINAMATH_CALUDE_class_visual_conditions_most_comprehensive_l2171_217169

/-- Represents a survey option -/
inductive SurveyOption
| LightTubes
| ClassVisualConditions
| NationwideExerciseTime
| FoodPigmentContent

/-- Defines characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  geographical_spread : Bool
  data_collection_feasibility : Bool

/-- Defines what makes a survey comprehensive -/
def is_comprehensive (s : SurveyCharacteristics) : Prop :=
  s.population_size ≤ 100 ∧ ¬s.geographical_spread ∧ s.data_collection_feasibility

/-- Assigns characteristics to each survey option -/
def survey_characteristics : SurveyOption → SurveyCharacteristics
| SurveyOption.LightTubes => ⟨50, false, false⟩
| SurveyOption.ClassVisualConditions => ⟨30, false, true⟩
| SurveyOption.NationwideExerciseTime => ⟨1000000, true, false⟩
| SurveyOption.FoodPigmentContent => ⟨500, true, false⟩

/-- Theorem stating that investigating visual conditions of a class is the most suitable for a comprehensive survey -/
theorem class_visual_conditions_most_comprehensive :
  ∀ (s : SurveyOption), s ≠ SurveyOption.ClassVisualConditions →
  is_comprehensive (survey_characteristics SurveyOption.ClassVisualConditions) ∧
  ¬(is_comprehensive (survey_characteristics s)) :=
by sorry

end NUMINAMATH_CALUDE_class_visual_conditions_most_comprehensive_l2171_217169


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l2171_217134

theorem least_three_digit_multiple_of_11 : ∃ (n : ℕ), n = 110 ∧ 
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 11 ∣ m → n ≤ m) ∧ 
  n ≥ 100 ∧ n < 1000 ∧ 11 ∣ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l2171_217134


namespace NUMINAMATH_CALUDE_infinitely_many_consecutive_sums_of_squares_l2171_217116

/-- A function that checks if a number is the sum of two squares --/
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

/-- The main theorem stating that there are infinitely many n satisfying the condition --/
theorem infinitely_many_consecutive_sums_of_squares :
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ 
    isSumOfTwoSquares n ∧
    isSumOfTwoSquares (n + 1) ∧
    isSumOfTwoSquares (n + 2) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_consecutive_sums_of_squares_l2171_217116


namespace NUMINAMATH_CALUDE_swim_time_ratio_l2171_217108

/-- The ratio of time taken to swim upstream to downstream -/
theorem swim_time_ratio (v_m : ℝ) (v_s : ℝ) (h1 : v_m = 4.5) (h2 : v_s = 1.5) :
  (v_m + v_s) / (v_m - v_s) = 2 := by
  sorry

#check swim_time_ratio

end NUMINAMATH_CALUDE_swim_time_ratio_l2171_217108


namespace NUMINAMATH_CALUDE_square_perimeter_l2171_217101

theorem square_perimeter (s : ℝ) (h : s > 0) :
  (5 * s / 2 = 40) → (4 * s = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2171_217101


namespace NUMINAMATH_CALUDE_police_force_female_officers_l2171_217159

theorem police_force_female_officers :
  ∀ (total_female : ℕ) (first_shift_total : ℕ) (first_shift_female_percent : ℚ),
    first_shift_total = 204 →
    first_shift_female_percent = 17 / 100 →
    (first_shift_total / 2 : ℚ) = first_shift_female_percent * total_female →
    total_female = 600 := by
  sorry

end NUMINAMATH_CALUDE_police_force_female_officers_l2171_217159


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2171_217157

-- Define the function type
def ContinuousFunction (α : Type*) := α → ℝ

-- State the theorem
theorem functional_equation_solution
  (f : ContinuousFunction ℝ)
  (h_cont : Continuous f)
  (h_domain : ∀ x : ℝ, x > 0 → f x ≠ 0)
  (h_eq : ∀ x y : ℝ, x > 0 → y > 0 →
    f (x + 1/x) + f (y + 1/y) = f (x + 1/y) + f (y + 1/x)) :
  ∃ c d : ℝ, ∀ x : ℝ, x > 0 → f x = c * x + d :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2171_217157


namespace NUMINAMATH_CALUDE_triangle_cosine_values_l2171_217180

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a - c = (√6/6)b and sin B = √6 sin C, then cos A = √6/4 and cos(2A - π/6) = (√15 - √3)/8 -/
theorem triangle_cosine_values (a b c A B C : ℝ) 
  (h1 : a - c = (Real.sqrt 6 / 6) * b)
  (h2 : Real.sin B = Real.sqrt 6 * Real.sin C) :
  Real.cos A = Real.sqrt 6 / 4 ∧ 
  Real.cos (2 * A - π / 6) = (Real.sqrt 15 - Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_values_l2171_217180


namespace NUMINAMATH_CALUDE_central_cell_value_l2171_217150

theorem central_cell_value (n : ℕ) (h1 : n = 29) :
  let total_sum := n * (n * (n + 1) / 2)
  let above_diagonal_sum := 3 * ((total_sum - n * (n + 1) / 2) / 2)
  let below_diagonal_sum := (total_sum - n * (n + 1) / 2) / 2
  let diagonal_sum := total_sum - above_diagonal_sum - below_diagonal_sum
  above_diagonal_sum = 3 * below_diagonal_sum →
  diagonal_sum / n = 15 := by
  sorry

#check central_cell_value

end NUMINAMATH_CALUDE_central_cell_value_l2171_217150


namespace NUMINAMATH_CALUDE_triangle_angle_c_l2171_217148

theorem triangle_angle_c (A B : ℝ) (hA : 3 * Real.sin A + 4 * Real.cos B = 6) 
  (hB : 4 * Real.sin B + 3 * Real.cos A = 1) : 
  ∃ C : ℝ, C = π / 6 ∧ A + B + C = π := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l2171_217148


namespace NUMINAMATH_CALUDE_median_of_special_sequence_l2171_217147

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem median_of_special_sequence :
  let N : ℕ := sequence_sum 150
  let median_position : ℕ := (N + 1) / 2
  ∃ (k : ℕ), k = 106 ∧ 
    sequence_sum (k - 1) < median_position ∧ 
    median_position ≤ sequence_sum k :=
by sorry

end NUMINAMATH_CALUDE_median_of_special_sequence_l2171_217147


namespace NUMINAMATH_CALUDE_tangent_circles_count_l2171_217127

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Determines if a circle is tangent to two other circles -/
def is_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  are_externally_tangent c c1 ∧ are_externally_tangent c c2

/-- The main theorem to be proven -/
theorem tangent_circles_count 
  (O1 O2 : Circle) 
  (h_tangent : are_externally_tangent O1 O2) 
  (h_radius1 : O1.radius = 2) 
  (h_radius2 : O2.radius = 4) : 
  ∃! (s : Finset Circle), 
    Finset.card s = 5 ∧ 
    ∀ c ∈ s, c.radius = 6 ∧ is_tangent_to_both c O1 O2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l2171_217127


namespace NUMINAMATH_CALUDE_complex_set_characterization_l2171_217156

theorem complex_set_characterization (z : ℂ) :
  (z - 1)^2 = Complex.abs (z - 1)^2 ↔ z.im = 0 :=
sorry

end NUMINAMATH_CALUDE_complex_set_characterization_l2171_217156


namespace NUMINAMATH_CALUDE_max_d_value_l2171_217188

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ 707340 + 10 * d + 4 * e ≥ 100000 ∧ 707340 + 10 * d + 4 * e < 1000000

def is_multiple_of_34 (d e : ℕ) : Prop :=
  (707340 + 10 * d + 4 * e) % 34 = 0

theorem max_d_value (d e : ℕ) :
  is_valid_number d e → is_multiple_of_34 d e → d ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l2171_217188


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l2171_217154

/-- A linear function with slope k and y-intercept b -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

/-- Predicate for a point (x, y) being in quadrant I -/
def InQuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Predicate for a point (x, y) being in quadrant II -/
def InQuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Predicate for a point (x, y) being in quadrant IV -/
def InQuadrantIV (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Theorem stating that the graph of y = 2x + 1 passes through quadrants I, II, and IV -/
theorem linear_function_quadrants :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (y₁ = LinearFunction 2 1 x₁) ∧ InQuadrantI x₁ y₁ ∧
    (y₂ = LinearFunction 2 1 x₂) ∧ InQuadrantII x₂ y₂ ∧
    (y₃ = LinearFunction 2 1 x₃) ∧ InQuadrantIV x₃ y₃ :=
by
  sorry


end NUMINAMATH_CALUDE_linear_function_quadrants_l2171_217154


namespace NUMINAMATH_CALUDE_susan_coins_value_l2171_217184

theorem susan_coins_value :
  ∀ (n d : ℕ),
  n + d = 30 →
  5 * n + 10 * d + 90 = 10 * n + 5 * d →
  5 * n + 10 * d = 180 := by
sorry

end NUMINAMATH_CALUDE_susan_coins_value_l2171_217184


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l2171_217133

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 25)
  (h2 : failed_english = 48)
  (h3 : failed_both = 27) :
  100 - (failed_hindi + failed_english - failed_both) = 54 := by
sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l2171_217133


namespace NUMINAMATH_CALUDE_system_solution_l2171_217112

theorem system_solution (a b : ℝ) : 
  (a * 1 - b * 2 = -1) → 
  (a * 1 + b * 2 = 7) → 
  3 * a - 4 * b = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2171_217112


namespace NUMINAMATH_CALUDE_casey_nail_coats_l2171_217187

/-- The time it takes to apply and dry one coat of nail polish -/
def coat_time : ℕ := 20 + 20

/-- The total time spent on decorating nails -/
def total_time : ℕ := 120

/-- The number of coats applied to each nail -/
def num_coats : ℕ := total_time / coat_time

theorem casey_nail_coats : num_coats = 3 := by
  sorry

end NUMINAMATH_CALUDE_casey_nail_coats_l2171_217187


namespace NUMINAMATH_CALUDE_quadrilateral_properties_l2171_217173

/-- A quadrilateral with coordinates of its four vertices -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The diagonals of a quadrilateral -/
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((q.A.1 - q.C.1, q.A.2 - q.C.2), (q.B.1 - q.D.1, q.B.2 - q.D.2))

/-- Check if the diagonals of a quadrilateral are equal -/
def equal_diagonals (q : Quadrilateral) : Prop :=
  let (d1, d2) := diagonals q
  d1.1^2 + d1.2^2 = d2.1^2 + d2.2^2

/-- Check if the diagonals of a quadrilateral bisect each other -/
def diagonals_bisect (q : Quadrilateral) : Prop :=
  let mid1 := ((q.A.1 + q.C.1) / 2, (q.A.2 + q.C.2) / 2)
  let mid2 := ((q.B.1 + q.D.1) / 2, (q.B.2 + q.D.2) / 2)
  mid1 = mid2

/-- Check if the diagonals of a quadrilateral are perpendicular -/
def perpendicular_diagonals (q : Quadrilateral) : Prop :=
  let (d1, d2) := diagonals q
  d1.1 * d2.1 + d1.2 * d2.2 = 0

/-- Check if all sides of a quadrilateral are equal -/
def equal_sides (q : Quadrilateral) : Prop :=
  let side1 := (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2
  let side2 := (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2
  let side3 := (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2
  let side4 := (q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2
  side1 = side2 ∧ side2 = side3 ∧ side3 = side4

/-- A quadrilateral is a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop :=
  equal_diagonals q ∧ diagonals_bisect q

/-- A quadrilateral is a rhombus -/
def is_rhombus (q : Quadrilateral) : Prop :=
  equal_sides q

/-- A quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop :=
  equal_diagonals q ∧ perpendicular_diagonals q

theorem quadrilateral_properties :
  (∀ q : Quadrilateral, equal_diagonals q ∧ diagonals_bisect q → is_rectangle q) ∧
  ¬(∀ q : Quadrilateral, perpendicular_diagonals q → is_rhombus q) ∧
  (∀ q : Quadrilateral, equal_diagonals q ∧ perpendicular_diagonals q → is_square q) ∧
  (∀ q : Quadrilateral, equal_sides q → is_rhombus q) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_l2171_217173


namespace NUMINAMATH_CALUDE_yue_bao_scientific_notation_l2171_217109

theorem yue_bao_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    (1853 * 1000000000 : ℝ) = a * (10 : ℝ) ^ n ∧
    a = 1.853 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_yue_bao_scientific_notation_l2171_217109


namespace NUMINAMATH_CALUDE_fine_payment_l2171_217195

theorem fine_payment (F : ℚ) 
  (hF : F > 0)
  (hJoe : F / 4 + 3 + F / 3 - 3 + F / 2 - 4 = F) : 
  F / 2 - 4 = 5 * F / 12 := by
  sorry

end NUMINAMATH_CALUDE_fine_payment_l2171_217195


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2171_217172

theorem absolute_value_inequality (x a : ℝ) (ha : a > 0) :
  (|x - 3| + |x - 4| + |x - 5| < a) ↔ (a > 4) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2171_217172


namespace NUMINAMATH_CALUDE_sum_lent_is_350_l2171_217103

/-- Proves that the sum lent is 350 Rs. given the specified conditions --/
theorem sum_lent_is_350 (P : ℚ) : 
  (∀ (I : ℚ), I = P * (4 : ℚ) * (8 : ℚ) / 100) →  -- Simple interest formula
  (∀ (I : ℚ), I = P - 238) →                      -- Interest is 238 less than principal
  P = 350 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_is_350_l2171_217103


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2171_217107

/-- Given two points are symmetric with respect to the origin -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  B.1 = -A.1 ∧ B.2 = -A.2

theorem symmetric_point_coordinates :
  let A : ℝ × ℝ := (-2, 1)
  let B : ℝ × ℝ := (2, -1)
  symmetric_wrt_origin A B → B = (2, -1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2171_217107


namespace NUMINAMATH_CALUDE_plain_cookies_sold_l2171_217196

-- Define the types for our variables
def chocolate_chip_price : ℚ := 125 / 100
def plain_price : ℚ := 75 / 100
def total_boxes : ℕ := 1585
def total_value : ℚ := 158625 / 100

-- Define the theorem
theorem plain_cookies_sold :
  ∃ (c p : ℕ),
    c + p = total_boxes ∧
    c * chocolate_chip_price + p * plain_price = total_value ∧
    p = 790 := by
  sorry


end NUMINAMATH_CALUDE_plain_cookies_sold_l2171_217196


namespace NUMINAMATH_CALUDE_supermarket_spending_l2171_217146

theorem supermarket_spending (total : ℚ) : 
  (1/5 : ℚ) * total + (1/3 : ℚ) * total + (1/10 : ℚ) * total + 11 = total → 
  total = 30 :=
by sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2171_217146


namespace NUMINAMATH_CALUDE_imaginary_unit_power_2016_l2171_217123

theorem imaginary_unit_power_2016 (i : ℂ) (h : i^2 = -1) : i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_2016_l2171_217123


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l2171_217183

/-- Given two internally tangent circles C₁ and C₂ with equations x² + y² + 2ax + a² - 4 = 0 and 
    x² + y² - 2by + b² - 1 = 0 respectively, where a, b ∈ ℝ and ab ≠ 0, 
    the minimum value of 1/a² + 1/b² is 9 -/
theorem min_value_sum_reciprocal_squares (a b : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y + b^2 - 1 = 0) →
  a ≠ 0 →
  b ≠ 0 →
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 ≠ 0 ∨ x^2 + y^2 - 2*b*y + b^2 - 1 ≠ 0 ∨ 
    (x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y + b^2 - 1 = 0)) →
  (1 / a^2 + 1 / b^2) ≥ 9 :=
by sorry

#check min_value_sum_reciprocal_squares

end NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l2171_217183


namespace NUMINAMATH_CALUDE_task_completion_probability_l2171_217139

theorem task_completion_probability 
  (task1_prob : ℝ) 
  (task1_not_task2_prob : ℝ) 
  (h1 : task1_prob = 2/3) 
  (h2 : task1_not_task2_prob = 4/15) 
  (h_independent : task1_not_task2_prob = task1_prob * (1 - task2_prob)) : 
  task2_prob = 3/5 :=
by
  sorry

#check task_completion_probability

end NUMINAMATH_CALUDE_task_completion_probability_l2171_217139


namespace NUMINAMATH_CALUDE_min_value_theorem_l2171_217131

/-- A geometric sequence with positive terms satisfying the given conditions -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r > 0, ∀ n, a (n + 1) = r * a n) ∧
  a 3 = a 2 + 2 * a 1

/-- The existence of terms satisfying the product condition -/
def ExistTerms (a : ℕ → ℝ) : Prop :=
  ∃ m n : ℕ, a m * a n = 64 * (a 1)^2

/-- The theorem statement -/
theorem min_value_theorem (a : ℕ → ℝ) 
  (h1 : GeometricSequence a) 
  (h2 : ExistTerms a) : 
  ∀ m n : ℕ, a m * a n = 64 * (a 1)^2 → 1 / m + 9 / n ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2171_217131


namespace NUMINAMATH_CALUDE_diana_bottle_caps_l2171_217158

theorem diana_bottle_caps (initial final eaten : ℕ) : 
  final = 61 → eaten = 4 → initial = final + eaten := by sorry

end NUMINAMATH_CALUDE_diana_bottle_caps_l2171_217158


namespace NUMINAMATH_CALUDE_horse_speed_l2171_217122

theorem horse_speed (field_area : ℝ) (run_time : ℝ) (horse_speed : ℝ) : 
  field_area = 576 →
  run_time = 8 →
  horse_speed = (4 * Real.sqrt field_area) / run_time →
  horse_speed = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_horse_speed_l2171_217122


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l2171_217155

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, 
    Nat.Prime p ∧ 
    p ∣ (36^2 + 45^2) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (36^2 + 45^2) → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l2171_217155


namespace NUMINAMATH_CALUDE_sally_coin_problem_l2171_217104

/-- Represents the number and value of coins in Sally's bank -/
structure CoinBank where
  pennies : ℕ
  nickels : ℕ
  pennyValue : ℕ
  nickelValue : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (bank : CoinBank) : ℕ :=
  bank.pennies * bank.pennyValue + bank.nickels * bank.nickelValue

/-- Represents gifts of nickels -/
structure NickelGift where
  fromDad : ℕ
  fromMom : ℕ

theorem sally_coin_problem (initialBank : CoinBank) (gift : NickelGift) :
  initialBank.pennies = 8 ∧
  initialBank.nickels = 7 ∧
  initialBank.pennyValue = 1 ∧
  initialBank.nickelValue = 5 ∧
  gift.fromDad = 9 ∧
  gift.fromMom = 2 →
  let finalBank : CoinBank := {
    pennies := initialBank.pennies,
    nickels := initialBank.nickels + gift.fromDad + gift.fromMom,
    pennyValue := initialBank.pennyValue,
    nickelValue := initialBank.nickelValue
  }
  finalBank.nickels = 18 ∧ totalValue finalBank = 98 := by
  sorry

end NUMINAMATH_CALUDE_sally_coin_problem_l2171_217104


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2171_217129

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x - x*y + 6*y = 0) :
  ∀ z w : ℝ, z > 0 ∧ w > 0 ∧ 2*z - z*w + 6*w = 0 → x + y ≤ z + w ∧ x + y = 8 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2171_217129


namespace NUMINAMATH_CALUDE_common_term_implies_fermat_number_l2171_217193

/-- Definition of the second-order arithmetic sequence -/
def a (n : ℕ) (k : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

/-- Definition of Fermat numbers -/
def fermat (m : ℕ) : ℕ :=
  2^(2^m) + 1

/-- Theorem stating that if k satisfies the condition, it must be a Fermat number -/
theorem common_term_implies_fermat_number (k : ℕ) (h1 : k > 2) :
  (∃ n m : ℕ, a n k = fermat m) → (∃ m : ℕ, k = fermat m) :=
sorry

end NUMINAMATH_CALUDE_common_term_implies_fermat_number_l2171_217193


namespace NUMINAMATH_CALUDE_percentage_of_4_to_50_percentage_of_4_to_50_proof_l2171_217178

theorem percentage_of_4_to_50 : ℝ → Prop :=
  fun x => (4 / 50 * 100 = x) → x = 8

-- The proof goes here
theorem percentage_of_4_to_50_proof : percentage_of_4_to_50 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_4_to_50_percentage_of_4_to_50_proof_l2171_217178


namespace NUMINAMATH_CALUDE_cylinder_cone_base_radii_equal_l2171_217106

/-- Given a cylinder and a cone with the same height and base radius, 
    if the ratio of their volumes is 3, then their base radii are equal -/
theorem cylinder_cone_base_radii_equal 
  (h : ℝ) -- height of both cylinder and cone
  (r_cylinder : ℝ) -- radius of cylinder base
  (r_cone : ℝ) -- radius of cone base
  (h_positive : h > 0)
  (r_cylinder_positive : r_cylinder > 0)
  (r_cone_positive : r_cone > 0)
  (same_radius : r_cylinder = r_cone)
  (volume_ratio : π * r_cylinder^2 * h / ((1/3) * π * r_cone^2 * h) = 3) :
  r_cylinder = r_cone :=
sorry

end NUMINAMATH_CALUDE_cylinder_cone_base_radii_equal_l2171_217106


namespace NUMINAMATH_CALUDE_odd_prime_sum_of_squares_l2171_217100

theorem odd_prime_sum_of_squares (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (x y m : ℤ), 1 + x^2 + y^2 = m * p ∧ 0 < m ∧ m < p := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_sum_of_squares_l2171_217100


namespace NUMINAMATH_CALUDE_derivative_at_one_implies_a_value_l2171_217192

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a) * log x

theorem derivative_at_one_implies_a_value (a : ℝ) :
  (deriv (f a)) 1 = -2 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_implies_a_value_l2171_217192


namespace NUMINAMATH_CALUDE_original_savings_calculation_l2171_217168

theorem original_savings_calculation (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) :
  furniture_fraction = 3 / 4 →
  tv_cost = 200 →
  (1 - furniture_fraction) * savings = tv_cost →
  savings = 800 := by
sorry

end NUMINAMATH_CALUDE_original_savings_calculation_l2171_217168
