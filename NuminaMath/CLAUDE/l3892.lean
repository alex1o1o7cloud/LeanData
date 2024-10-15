import Mathlib

namespace NUMINAMATH_CALUDE_initial_average_weight_l3892_389255

/-- Proves that the initially calculated average weight was 58.4 kg given the conditions of the problem. -/
theorem initial_average_weight (class_size : ℕ) (misread_weight : ℝ) (correct_weight : ℝ) (correct_average : ℝ) :
  class_size = 20 →
  misread_weight = 56 →
  correct_weight = 65 →
  correct_average = 58.85 →
  ∃ (initial_average : ℝ),
    initial_average * class_size + (correct_weight - misread_weight) = correct_average * class_size ∧
    initial_average = 58.4 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_weight_l3892_389255


namespace NUMINAMATH_CALUDE_camdens_dogs_legs_count_l3892_389273

theorem camdens_dogs_legs_count :
  ∀ (justin_dogs : ℕ) (rico_dogs : ℕ) (camden_dogs : ℕ),
    justin_dogs = 14 →
    rico_dogs = justin_dogs + 10 →
    camden_dogs = (3 * rico_dogs) / 4 →
    camden_dogs * 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_camdens_dogs_legs_count_l3892_389273


namespace NUMINAMATH_CALUDE_cosine_B_in_special_triangle_l3892_389267

theorem cosine_B_in_special_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 →  -- acute angle A
  0 < B ∧ B < π/2 →  -- acute angle B
  0 < C ∧ C < π/2 →  -- acute angle C
  a = (2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) / Real.sin A →  -- side-angle relation
  b = (2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) / Real.sin B →  -- side-angle relation
  c = (2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) / Real.sin C →  -- side-angle relation
  a = Real.sqrt 7 →
  b = 3 →
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3 →
  Real.cos B = Real.sqrt 7 / 14 := by
sorry

end NUMINAMATH_CALUDE_cosine_B_in_special_triangle_l3892_389267


namespace NUMINAMATH_CALUDE_childrens_bikes_count_l3892_389288

theorem childrens_bikes_count (regular_bikes : ℕ) (regular_wheels : ℕ) (childrens_wheels : ℕ) (total_wheels : ℕ) :
  regular_bikes = 7 →
  regular_wheels = 2 →
  childrens_wheels = 4 →
  total_wheels = 58 →
  regular_bikes * regular_wheels + childrens_wheels * (total_wheels - regular_bikes * regular_wheels) / childrens_wheels = 11 :=
by
  sorry

#check childrens_bikes_count

end NUMINAMATH_CALUDE_childrens_bikes_count_l3892_389288


namespace NUMINAMATH_CALUDE_point_on_circle_l3892_389257

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def arc_length (θ : ℝ) : ℝ := θ

theorem point_on_circle (start_x start_y end_x end_y θ : ℝ) : 
  unit_circle start_x start_y →
  unit_circle end_x end_y →
  arc_length θ = π/3 →
  start_x = 1 →
  start_y = 0 →
  end_x = 1/2 →
  end_y = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_point_on_circle_l3892_389257


namespace NUMINAMATH_CALUDE_microwave_sales_calculation_toaster_sales_calculation_l3892_389207

/-- Represents the relationship between number of items sold and their cost --/
structure SalesCostRelation where
  items : ℕ  -- number of items sold
  cost : ℕ   -- cost of each item in dollars
  constant : ℕ -- the constant of proportionality

/-- Given a SalesCostRelation and a new cost, calculate the new number of items --/
def calculate_new_sales (scr : SalesCostRelation) (new_cost : ℕ) : ℚ :=
  scr.constant / new_cost

theorem microwave_sales_calculation 
  (microwave_initial : SalesCostRelation)
  (h_microwave_initial : microwave_initial.items = 10 ∧ microwave_initial.cost = 400)
  (h_microwave_constant : microwave_initial.constant = microwave_initial.items * microwave_initial.cost) :
  calculate_new_sales microwave_initial 800 = 5 := by sorry

theorem toaster_sales_calculation
  (toaster_initial : SalesCostRelation)
  (h_toaster_initial : toaster_initial.items = 6 ∧ toaster_initial.cost = 600)
  (h_toaster_constant : toaster_initial.constant = toaster_initial.items * toaster_initial.cost) :
  Int.floor (calculate_new_sales toaster_initial 1000) = 4 := by sorry

end NUMINAMATH_CALUDE_microwave_sales_calculation_toaster_sales_calculation_l3892_389207


namespace NUMINAMATH_CALUDE_smallest_m_divisibility_l3892_389237

theorem smallest_m_divisibility (n : ℕ) (h_odd : Odd n) :
  (∃ (m : ℕ), m > 0 ∧ ∀ (k : ℕ), k > 0 → k < m →
    ¬(262417 ∣ (529^n + k * 132^n))) ∧
  (262417 ∣ (529^n + 1 * 132^n)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_divisibility_l3892_389237


namespace NUMINAMATH_CALUDE_peter_money_brought_l3892_389241

/-- The amount of money Peter brought to the store -/
def money_brought : ℚ := 2

/-- The cost of soda per ounce -/
def soda_cost_per_ounce : ℚ := 1/4

/-- The amount of soda Peter bought in ounces -/
def soda_amount : ℚ := 6

/-- The amount of money Peter left with -/
def money_left : ℚ := 1/2

/-- Proves that the amount of money Peter brought is correct -/
theorem peter_money_brought :
  money_brought = soda_cost_per_ounce * soda_amount + money_left :=
by sorry

end NUMINAMATH_CALUDE_peter_money_brought_l3892_389241


namespace NUMINAMATH_CALUDE_rhombus_adjacent_sides_equal_but_not_all_parallelograms_l3892_389244

-- Define a parallelogram
class Parallelogram :=
  (sides : Fin 4 → ℝ)
  (opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3)

-- Define a rhombus as a special case of parallelogram
class Rhombus extends Parallelogram :=
  (all_sides_equal : ∀ i j : Fin 4, sides i = sides j)

-- Theorem statement
theorem rhombus_adjacent_sides_equal_but_not_all_parallelograms 
  (r : Rhombus) (p : Parallelogram) : 
  (∀ i : Fin 4, r.sides i = r.sides ((i + 1) % 4)) ∧ 
  ¬(∀ (p : Parallelogram), ∀ i : Fin 4, p.sides i = p.sides ((i + 1) % 4)) :=
sorry

end NUMINAMATH_CALUDE_rhombus_adjacent_sides_equal_but_not_all_parallelograms_l3892_389244


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3892_389247

/-- The constant term in the expansion of x(1 - 1/√x)^5 is 10 -/
theorem constant_term_expansion (x : ℝ) (x_pos : x > 0) :
  ∃ (f : ℝ → ℝ), (∀ y, y ≠ 0 → f y = y * (1 - 1 / Real.sqrt y)^5) ∧
  (∃ c, ∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - x| → |y - x| < δ → |f y - (10 + c * (y - x))| < ε * |y - x|) :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3892_389247


namespace NUMINAMATH_CALUDE_min_slope_tangent_line_l3892_389233

variable (x y : ℝ)

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 3

-- Theorem statement
theorem min_slope_tangent_line :
  ∃ (x₀ : ℝ), 
    (∀ x, f' x ≥ f' x₀) ∧ 
    (3*x - y + 1 = 0 ↔ y = f' x₀ * (x - x₀) + f x₀) :=
sorry

end NUMINAMATH_CALUDE_min_slope_tangent_line_l3892_389233


namespace NUMINAMATH_CALUDE_kerosene_cost_l3892_389268

/-- The cost of kerosene in a market with given price relationships -/
theorem kerosene_cost (rice_pound_cost : ℝ) (h1 : rice_pound_cost = 0.24) :
  let dozen_eggs_cost := rice_pound_cost
  let half_liter_kerosene_cost := dozen_eggs_cost / 2
  let liter_kerosene_cost := 2 * half_liter_kerosene_cost
  let cents_per_dollar := 100
  ⌊liter_kerosene_cost * cents_per_dollar⌋ = 24 := by sorry

end NUMINAMATH_CALUDE_kerosene_cost_l3892_389268


namespace NUMINAMATH_CALUDE_course_duration_l3892_389260

theorem course_duration (total_hours : ℕ) (class_hours_1 : ℕ) (class_hours_2 : ℕ) (homework_hours : ℕ) :
  total_hours = 336 →
  class_hours_1 = 3 →
  class_hours_2 = 4 →
  homework_hours = 4 →
  (2 * class_hours_1 + class_hours_2 + homework_hours) * 24 = total_hours :=
by
  sorry

end NUMINAMATH_CALUDE_course_duration_l3892_389260


namespace NUMINAMATH_CALUDE_stating_number_of_people_in_first_group_l3892_389246

/-- Represents the amount of work one person can do in one day -/
def work_per_person_per_day : ℝ := 1

/-- Represents the number of days given in the problem -/
def days : ℕ := 3

/-- Represents the number of people in the second group -/
def people_second_group : ℕ := 6

/-- Represents the amount of work done by the first group -/
def work_first_group : ℕ := 3

/-- Represents the amount of work done by the second group -/
def work_second_group : ℕ := 6

/-- 
Theorem stating that the number of people in the first group is 3,
given the conditions from the problem.
-/
theorem number_of_people_in_first_group : 
  ∃ (p : ℕ), 
    p * days * work_per_person_per_day = work_first_group ∧
    people_second_group * days * work_per_person_per_day = work_second_group ∧
    p = 3 := by
  sorry

end NUMINAMATH_CALUDE_stating_number_of_people_in_first_group_l3892_389246


namespace NUMINAMATH_CALUDE_number_difference_l3892_389250

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 25220)
  (div_12 : 12 ∣ a)
  (relation : b = a / 100) : 
  a - b = 24750 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l3892_389250


namespace NUMINAMATH_CALUDE_zoo_lion_cubs_l3892_389205

theorem zoo_lion_cubs (initial_animals : ℕ) (gorillas_sent : ℕ) (hippo_adopted : ℕ) (giraffes_adopted : ℕ) 
  (rhinos_added : ℕ) (crocodiles_added : ℕ) (final_animals : ℕ) :
  initial_animals = 150 →
  gorillas_sent = 12 →
  hippo_adopted = 1 →
  giraffes_adopted = 8 →
  rhinos_added = 4 →
  crocodiles_added = 5 →
  final_animals = 260 →
  ∃ (cubs : ℕ), 
    final_animals = initial_animals - gorillas_sent + hippo_adopted + giraffes_adopted + 
                    rhinos_added + crocodiles_added + cubs + 3 * cubs ∧
    cubs = 26 :=
by sorry

end NUMINAMATH_CALUDE_zoo_lion_cubs_l3892_389205


namespace NUMINAMATH_CALUDE_hotel_breakfast_probability_l3892_389249

def num_guests : ℕ := 3
def num_roll_types : ℕ := 4
def total_rolls : ℕ := 12
def rolls_per_guest : ℕ := 4

def probability_one_of_each : ℚ :=
  (9 : ℚ) / 55 * (8 : ℚ) / 35 * (1 : ℚ) / 1

theorem hotel_breakfast_probability :
  probability_one_of_each = (72 : ℚ) / 1925 :=
by sorry

end NUMINAMATH_CALUDE_hotel_breakfast_probability_l3892_389249


namespace NUMINAMATH_CALUDE_cory_fruit_order_l3892_389248

def fruit_arrangement (a o b g : ℕ) : ℕ :=
  Nat.factorial 9 / (Nat.factorial a * Nat.factorial o * Nat.factorial b * Nat.factorial g)

theorem cory_fruit_order : fruit_arrangement 3 3 2 1 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_cory_fruit_order_l3892_389248


namespace NUMINAMATH_CALUDE_degree_of_composed_product_l3892_389229

/-- Given polynomials p and q with degrees 3 and 4 respectively,
    prove that the degree of p(x^4) * q(x^5) is 32 -/
theorem degree_of_composed_product (p q : Polynomial ℝ) 
  (hp : Polynomial.degree p = 3) (hq : Polynomial.degree q = 4) :
  Polynomial.degree (p.comp (Polynomial.X ^ 4) * q.comp (Polynomial.X ^ 5)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_composed_product_l3892_389229


namespace NUMINAMATH_CALUDE_remainder_when_divided_by_x_plus_one_l3892_389245

def q (x : ℝ) : ℝ := 2*x^4 - 3*x^3 + 4*x^2 - 5*x + 6

theorem remainder_when_divided_by_x_plus_one :
  ∃ p : ℝ → ℝ, q = fun x ↦ (x + 1) * p x + 20 :=
sorry

end NUMINAMATH_CALUDE_remainder_when_divided_by_x_plus_one_l3892_389245


namespace NUMINAMATH_CALUDE_smallest_number_with_million_divisors_l3892_389278

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_million_divisors (N : ℕ) :
  divisor_count N = 1000000 →
  N ≥ 2^9 * (3 * 5 * 7 * 11 * 13)^4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_million_divisors_l3892_389278


namespace NUMINAMATH_CALUDE_inequality_solution_set_quadratic_inequality_solution_set_l3892_389234

-- Part 1
theorem inequality_solution_set (x : ℝ) :
  (2 - x) / (x + 3) > 0 ↔ x ∈ Set.Ioo (-3 : ℝ) 2 :=
sorry

-- Part 2
theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_quadratic_inequality_solution_set_l3892_389234


namespace NUMINAMATH_CALUDE_range_of_m_l3892_389218

def A : Set ℝ := {x | (x - 4) / (x + 3) ≤ 0}

def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 1}

theorem range_of_m : 
  ∀ m : ℝ, (A ∩ B m = B m) ↔ m ∈ Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_range_of_m_l3892_389218


namespace NUMINAMATH_CALUDE_existence_of_integers_l3892_389214

theorem existence_of_integers : ∃ (m n p q : ℕ+), 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q ∧
  (m : ℝ) + n = p + q ∧
  Real.sqrt (m : ℝ) + (n : ℝ) ^ (1/4) = Real.sqrt (p : ℝ) + (q : ℝ) ^ (1/3) ∧
  Real.sqrt (m : ℝ) + (n : ℝ) ^ (1/4) > 2004 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l3892_389214


namespace NUMINAMATH_CALUDE_min_value_theorem_l3892_389256

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  2 * x^2 + 24 * x + 128 / x^3 ≥ 168 ∧
  (2 * x^2 + 24 * x + 128 / x^3 = 168 ↔ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3892_389256


namespace NUMINAMATH_CALUDE_parabola_directrix_l3892_389227

/-- The equation of a parabola -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 - 6 * x + 1

/-- The equation of the directrix -/
def directrix (y : ℝ) : Prop := y = -25/12

/-- Theorem: The directrix of the given parabola is y = -25/12 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p q : ℝ, parabola p q → (p - x)^2 + (q - y)^2 = (q - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3892_389227


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3892_389239

/-- The problem setup for the interest rate calculation --/
structure InterestProblem where
  total_sum : ℝ
  second_part : ℝ
  first_part : ℝ
  first_rate : ℝ
  first_time : ℝ
  second_time : ℝ
  second_rate : ℝ

/-- The interest rate calculation theorem --/
theorem interest_rate_calculation (p : InterestProblem)
  (h1 : p.total_sum = 2769)
  (h2 : p.second_part = 1704)
  (h3 : p.first_part = p.total_sum - p.second_part)
  (h4 : p.first_rate = 3 / 100)
  (h5 : p.first_time = 8)
  (h6 : p.second_time = 3)
  (h7 : p.first_part * p.first_rate * p.first_time = p.second_part * p.second_rate * p.second_time) :
  p.second_rate = 5 / 100 := by
  sorry


end NUMINAMATH_CALUDE_interest_rate_calculation_l3892_389239


namespace NUMINAMATH_CALUDE_rectangle_area_l3892_389276

/-- Given a rectangle with length L and width W, if 2L + W = 34 and L + 2W = 38, then the area of the rectangle is 140. -/
theorem rectangle_area (L W : ℝ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) :
  L * W = 140 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3892_389276


namespace NUMINAMATH_CALUDE_valid_triangle_divisions_l3892_389290

/-- Represents a division of an equilateral triangle into smaller triangles -/
structure TriangleDivision where
  n : ℕ  -- number of smaller triangles
  k : ℕ  -- number of identical polygons

/-- Predicate to check if a division is valid -/
def is_valid_division (d : TriangleDivision) : Prop :=
  d.n = 36 ∧ d.k ∣ d.n ∧ 
  (d.k = 1 ∨ d.k = 3 ∨ d.k = 4 ∨ d.k = 9 ∨ d.k = 12 ∨ d.k = 36)

/-- Theorem stating the valid divisions of the triangle -/
theorem valid_triangle_divisions :
  ∀ d : TriangleDivision, is_valid_division d ↔ 
    (d.k = 1 ∨ d.k = 3 ∨ d.k = 4 ∨ d.k = 9 ∨ d.k = 12 ∨ d.k = 36) :=
by sorry

end NUMINAMATH_CALUDE_valid_triangle_divisions_l3892_389290


namespace NUMINAMATH_CALUDE_floor_square_minus_square_floor_l3892_389293

theorem floor_square_minus_square_floor (x : ℝ) : x = 13.7 → ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 18 := by
  sorry

end NUMINAMATH_CALUDE_floor_square_minus_square_floor_l3892_389293


namespace NUMINAMATH_CALUDE_grunters_win_all_games_l3892_389289

/-- The probability of the Grunters winning a single game -/
def p : ℚ := 3/4

/-- The number of games in the series -/
def n : ℕ := 6

/-- The theorem stating the probability of the Grunters winning all games -/
theorem grunters_win_all_games : p ^ n = 729/4096 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_all_games_l3892_389289


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l3892_389219

theorem sum_remainder_mod_11 : (99001 + 99002 + 99003 + 99004 + 99005 + 99006) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l3892_389219


namespace NUMINAMATH_CALUDE_x_equals_two_l3892_389236

theorem x_equals_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 6 * x^3 + 12 * x * y^2 = 3 * x^2 * y + 3 * x^4) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_two_l3892_389236


namespace NUMINAMATH_CALUDE_total_appetizers_l3892_389284

def hotdogs : ℕ := 60
def cheese_pops : ℕ := 40
def chicken_nuggets : ℕ := 80
def mini_quiches : ℕ := 100
def stuffed_mushrooms : ℕ := 50

theorem total_appetizers : 
  hotdogs + cheese_pops + chicken_nuggets + mini_quiches + stuffed_mushrooms = 330 := by
  sorry

end NUMINAMATH_CALUDE_total_appetizers_l3892_389284


namespace NUMINAMATH_CALUDE_expected_winnings_is_five_thirds_l3892_389292

/-- A coin with three possible outcomes -/
inductive CoinOutcome
  | Heads
  | Tails
  | Edge

/-- The probability of each outcome -/
def probability (outcome : CoinOutcome) : ℚ :=
  match outcome with
  | .Heads => 1/3
  | .Tails => 1/2
  | .Edge => 1/6

/-- The payoff for each outcome in dollars -/
def payoff (outcome : CoinOutcome) : ℤ :=
  match outcome with
  | .Heads => 2
  | .Tails => 4
  | .Edge => -6

/-- The expected winnings from flipping the coin -/
def expectedWinnings : ℚ :=
  (probability CoinOutcome.Heads * payoff CoinOutcome.Heads) +
  (probability CoinOutcome.Tails * payoff CoinOutcome.Tails) +
  (probability CoinOutcome.Edge * payoff CoinOutcome.Edge)

theorem expected_winnings_is_five_thirds :
  expectedWinnings = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_winnings_is_five_thirds_l3892_389292


namespace NUMINAMATH_CALUDE_expression_equality_l3892_389243

theorem expression_equality : 2 * Real.sqrt 3 * (3/2)^(1/3) * 12^(1/6) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l3892_389243


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l3892_389200

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  z₁ = 1 + I →
  (z₁.re = z₂.re ∧ z₁.im = -z₂.im) →
  z₁ * z₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l3892_389200


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3892_389298

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 770 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (eq1 : a + b = 17)
  (eq2 : a * b + c + d = 98)
  (eq3 : a * d + b * c = 176)
  (eq4 : c * d = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 770 ∧ ∃ a b c d, a^2 + b^2 + c^2 + d^2 = 770 := by
  sorry

#check max_sum_of_squares

end NUMINAMATH_CALUDE_max_sum_of_squares_l3892_389298


namespace NUMINAMATH_CALUDE_L_shape_sum_implies_all_ones_l3892_389230

/-- A type representing a 2015x2015 matrix of real numbers -/
def Matrix2015 := Fin 2015 → Fin 2015 → ℝ

/-- The L-shape property: sum of any three numbers in an L-shape is 3 -/
def has_L_shape_property (M : Matrix2015) : Prop :=
  ∀ i j k l : Fin 2015, 
    ((i = k ∧ j ≠ l) ∨ (i ≠ k ∧ j = l)) → 
    M i j + M k j + M k l = 3

/-- All elements in the matrix are 1 -/
def all_ones (M : Matrix2015) : Prop :=
  ∀ i j : Fin 2015, M i j = 1

/-- Main theorem: If a 2015x2015 matrix has the L-shape property, then all its elements are 1 -/
theorem L_shape_sum_implies_all_ones (M : Matrix2015) :
  has_L_shape_property M → all_ones M := by
  sorry


end NUMINAMATH_CALUDE_L_shape_sum_implies_all_ones_l3892_389230


namespace NUMINAMATH_CALUDE_sufficient_fabric_l3892_389281

/-- Represents the dimensions of a rectangular piece of fabric -/
structure FabricDimensions where
  length : ℕ
  width : ℕ

/-- Checks if a piece of fabric can be cut into at least n smaller pieces -/
def canCutInto (fabric : FabricDimensions) (piece : FabricDimensions) (n : ℕ) : Prop :=
  ∃ (l w : ℕ), 
    l * piece.length ≤ fabric.length ∧ 
    w * piece.width ≤ fabric.width ∧ 
    l * w ≥ n

theorem sufficient_fabric : 
  let fabric := FabricDimensions.mk 140 75
  let dress := FabricDimensions.mk 45 26
  canCutInto fabric dress 8 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_fabric_l3892_389281


namespace NUMINAMATH_CALUDE_integer_as_sum_diff_squares_l3892_389216

theorem integer_as_sum_diff_squares (n : ℤ) :
  ∃ (a b c : ℤ), n = a^2 + b^2 - c^2 := by
  sorry

end NUMINAMATH_CALUDE_integer_as_sum_diff_squares_l3892_389216


namespace NUMINAMATH_CALUDE_jacket_price_restoration_l3892_389224

theorem jacket_price_restoration (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_first_reduction := initial_price * (1 - 0.25)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.15)
  let required_increase := (initial_price / price_after_second_reduction) - 1
  abs (required_increase - 0.5686) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_restoration_l3892_389224


namespace NUMINAMATH_CALUDE_intersection_when_a_is_neg_two_intersection_equals_A_iff_l3892_389264

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x + a < 0}

-- Theorem 1: When a = -2, A ∩ B = {x | 1/2 ≤ x < 2}
theorem intersection_when_a_is_neg_two :
  A ∩ B (-2) = {x : ℝ | 1/2 ≤ x ∧ x < 2} := by sorry

-- Theorem 2: A ∩ B = A if and only if a < -3
theorem intersection_equals_A_iff (a : ℝ) :
  A ∩ B a = A ↔ a < -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_neg_two_intersection_equals_A_iff_l3892_389264


namespace NUMINAMATH_CALUDE_travel_time_ngapara_to_zipra_l3892_389271

/-- Proves that the time taken to travel from Ngapara to Zipra is 60 hours -/
theorem travel_time_ngapara_to_zipra (time_ngapara_zipra : ℝ) 
  (h1 : 0.8 * time_ngapara_zipra + time_ngapara_zipra = 108) : 
  time_ngapara_zipra = 60 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_ngapara_to_zipra_l3892_389271


namespace NUMINAMATH_CALUDE_divisibility_by_3804_l3892_389270

theorem divisibility_by_3804 (n : ℕ+) :
  ∃ k : ℤ, (n.val^3 - n.val : ℤ) * (5^(8*n.val+4) + 3^(4*n.val+2)) = 3804 * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_3804_l3892_389270


namespace NUMINAMATH_CALUDE_noah_yearly_call_cost_l3892_389202

/-- The total cost of Noah's calls to his Grammy for a year -/
def total_cost (calls_per_week : ℕ) (minutes_per_call : ℕ) (cost_per_minute : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (calls_per_week * minutes_per_call * weeks_per_year : ℕ) * cost_per_minute

/-- Theorem stating that Noah's yearly call cost to his Grammy is $78 -/
theorem noah_yearly_call_cost :
  total_cost 1 30 (5/100) 52 = 78 := by
  sorry

end NUMINAMATH_CALUDE_noah_yearly_call_cost_l3892_389202


namespace NUMINAMATH_CALUDE_lucy_cake_packs_l3892_389291

/-- Represents the number of packs of cookies Lucy bought -/
def cookie_packs : ℕ := 23

/-- Represents the total number of grocery packs Lucy bought -/
def total_packs : ℕ := 27

/-- Represents the number of cake packs Lucy bought -/
def cake_packs : ℕ := total_packs - cookie_packs

/-- Proves that the number of cake packs Lucy bought is equal to 4 -/
theorem lucy_cake_packs : cake_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_lucy_cake_packs_l3892_389291


namespace NUMINAMATH_CALUDE_ceiling_times_self_210_l3892_389217

theorem ceiling_times_self_210 : ∃ x : ℝ, ⌈x⌉ * x = 210 ∧ x = 14 := by sorry

end NUMINAMATH_CALUDE_ceiling_times_self_210_l3892_389217


namespace NUMINAMATH_CALUDE_billy_weight_l3892_389282

/-- Given the weights of Billy, Brad, and Carl, prove that Billy weighs 159 pounds. -/
theorem billy_weight (billy brad carl : ℕ) 
  (h1 : billy = brad + 9)
  (h2 : brad = carl + 5)
  (h3 : carl = 145) : 
  billy = 159 := by
  sorry

end NUMINAMATH_CALUDE_billy_weight_l3892_389282


namespace NUMINAMATH_CALUDE_intersection_equality_l3892_389222

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem intersection_equality (a : ℝ) : A ∩ B a = B a → a = 0 ∨ a = -1 ∨ a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l3892_389222


namespace NUMINAMATH_CALUDE_inequality_range_l3892_389209

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2*x + 1 - a^2 < 0) ↔ 
  (a > 3 ∨ a < -3) := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l3892_389209


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3892_389203

theorem arithmetic_evaluation : 23 - |(-6)| - 23 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3892_389203


namespace NUMINAMATH_CALUDE_special_line_equation_l3892_389225

/-- A line passing through (5,2) with y-intercept twice the x-intercept -/
structure SpecialLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (5,2)
  point_condition : 2 = m * 5 + b
  -- The y-intercept is twice the x-intercept
  intercept_condition : b = -2 * (b / m)

/-- The equation of the special line is either 2x+y-12=0 or 2x-5y=0 -/
theorem special_line_equation (l : SpecialLine) :
  (2 * l.m + 1 ≠ 0 ∧ 2 * l.m * l.b + l.b = 12) ∨
  (l.m = 2/5 ∧ l.b = 0) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l3892_389225


namespace NUMINAMATH_CALUDE_power_decomposition_l3892_389213

/-- Sum of the first k odd numbers -/
def sum_odd (k : ℕ) : ℕ := k^2

/-- The nth odd number -/
def nth_odd (n : ℕ) : ℕ := 2*n - 1

theorem power_decomposition (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (n^2 = sum_odd 10) →
  (nth_odd ((m-1)^2 + 1) = 21) →
  m + n = 15 := by sorry

end NUMINAMATH_CALUDE_power_decomposition_l3892_389213


namespace NUMINAMATH_CALUDE_percentage_of_non_roses_l3892_389228

theorem percentage_of_non_roses (roses tulips daisies : ℕ) : 
  roses = 25 → tulips = 40 → daisies = 35 → 
  (tulips + daisies : ℚ) / (roses + tulips + daisies) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_non_roses_l3892_389228


namespace NUMINAMATH_CALUDE_replaced_person_age_l3892_389201

/-- Given a group of 10 persons, if replacing one person with a 16-year-old
    decreases the average age by 3 years, then the replaced person was 46 years old. -/
theorem replaced_person_age (group_size : ℕ) (avg_decrease : ℝ) (new_person_age : ℕ) :
  group_size = 10 →
  avg_decrease = 3 →
  new_person_age = 16 →
  (group_size : ℝ) * avg_decrease + new_person_age = 46 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_age_l3892_389201


namespace NUMINAMATH_CALUDE_white_space_area_is_31_l3892_389280

/-- Represents the dimensions of a rectangular board -/
structure Board :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the area of a board -/
def boardArea (b : Board) : ℕ := b.width * b.height

/-- Represents the area covered by each letter -/
structure LetterAreas :=
  (C : ℕ)
  (O : ℕ)
  (D : ℕ)
  (E : ℕ)

/-- Calculates the total area covered by all letters -/
def totalLetterArea (l : LetterAreas) : ℕ := l.C + l.O + l.D + l.E

/-- The main theorem stating the white space area -/
theorem white_space_area_is_31 (board : Board) (letters : LetterAreas) : 
  board.width = 4 ∧ board.height = 18 ∧ 
  letters.C = 8 ∧ letters.O = 10 ∧ letters.D = 10 ∧ letters.E = 13 →
  boardArea board - totalLetterArea letters = 31 := by
  sorry


end NUMINAMATH_CALUDE_white_space_area_is_31_l3892_389280


namespace NUMINAMATH_CALUDE_sum_three_digit_numbers_eq_255744_l3892_389206

/-- The sum of all three-digit natural numbers with digits ranging from 1 to 8 -/
def sum_three_digit_numbers : ℕ :=
  let digit_sum : ℕ := (8 * 9) / 2  -- Sum of digits from 1 to 8
  let digit_count : ℕ := 8 * 8      -- Number of times each digit appears in each place
  let place_sum : ℕ := digit_sum * digit_count
  place_sum * 111

theorem sum_three_digit_numbers_eq_255744 :
  sum_three_digit_numbers = 255744 := by
  sorry

end NUMINAMATH_CALUDE_sum_three_digit_numbers_eq_255744_l3892_389206


namespace NUMINAMATH_CALUDE_total_balls_count_l3892_389274

-- Define the number of boxes
def num_boxes : ℕ := 3

-- Define the number of balls in each box
def balls_per_box : ℕ := 5

-- Theorem to prove
theorem total_balls_count : num_boxes * balls_per_box = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_count_l3892_389274


namespace NUMINAMATH_CALUDE_expression_value_l3892_389258

theorem expression_value (a b : ℝ) (h : a - 2*b = 3) : 2*a - 4*b - 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3892_389258


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l3892_389220

theorem percentage_of_sikh_boys 
  (total_boys : ℕ) 
  (muslim_percentage : ℚ) 
  (hindu_percentage : ℚ) 
  (other_boys : ℕ) 
  (h1 : total_boys = 300)
  (h2 : muslim_percentage = 44 / 100)
  (h3 : hindu_percentage = 28 / 100)
  (h4 : other_boys = 54) :
  (total_boys - (muslim_percentage * total_boys + hindu_percentage * total_boys + other_boys)) / total_boys = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l3892_389220


namespace NUMINAMATH_CALUDE_log_x3y2_equals_2_l3892_389275

theorem log_x3y2_equals_2 
  (x y : ℝ) 
  (h1 : Real.log (x^2 * y^5) = 2) 
  (h2 : Real.log (x^3 * y^2) = 2) : 
  Real.log (x^3 * y^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_log_x3y2_equals_2_l3892_389275


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3892_389297

theorem quadratic_root_property (m : ℝ) : 
  m^2 - 3*m + 1 = 0 → 2*m^2 - 6*m - 2024 = -2026 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3892_389297


namespace NUMINAMATH_CALUDE_cookies_in_box_l3892_389223

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The weight capacity of the box in pounds -/
def box_capacity : ℕ := 40

/-- The weight of each cookie in ounces -/
def cookie_weight : ℕ := 2

/-- Proves that the number of cookies that can fit in the box is 320 -/
theorem cookies_in_box : 
  (box_capacity * ounces_per_pound) / cookie_weight = 320 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_box_l3892_389223


namespace NUMINAMATH_CALUDE_cherry_pie_degrees_l3892_389215

/-- The number of students in Richelle's class -/
def total_students : ℕ := 36

/-- The number of students who prefer chocolate pie -/
def chocolate_preference : ℕ := 12

/-- The number of students who prefer apple pie -/
def apple_preference : ℕ := 8

/-- The number of students who prefer blueberry pie -/
def blueberry_preference : ℕ := 6

/-- The number of students who prefer cherry pie -/
def cherry_preference : ℕ := (total_students - (chocolate_preference + apple_preference + blueberry_preference)) / 2

theorem cherry_pie_degrees : 
  (cherry_preference : ℚ) / total_students * 360 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_degrees_l3892_389215


namespace NUMINAMATH_CALUDE_simplify_exponential_fraction_l3892_389272

theorem simplify_exponential_fraction (n : ℕ) :
  (3^(n+4) - 3*(3^n)) / (3*(3^(n+3))) = 26 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponential_fraction_l3892_389272


namespace NUMINAMATH_CALUDE_complex_subtraction_problem_l3892_389231

theorem complex_subtraction_problem :
  (4 : ℂ) - 3*I - ((5 : ℂ) - 12*I) = -1 + 9*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_problem_l3892_389231


namespace NUMINAMATH_CALUDE_probability_not_pulling_prize_l3892_389299

/-- Given odds of 5:6 for pulling a prize, prove that the probability of not pulling the prize is 6/11 -/
theorem probability_not_pulling_prize (favorable_outcomes unfavorable_outcomes : ℕ) 
  (h : favorable_outcomes = 5 ∧ unfavorable_outcomes = 6) : 
  (unfavorable_outcomes : ℚ) / (favorable_outcomes + unfavorable_outcomes) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_pulling_prize_l3892_389299


namespace NUMINAMATH_CALUDE_x_positive_iff_sum_geq_two_l3892_389279

theorem x_positive_iff_sum_geq_two (x : ℝ) : x > 0 ↔ x + 1/x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_x_positive_iff_sum_geq_two_l3892_389279


namespace NUMINAMATH_CALUDE_coefficient_value_l3892_389262

-- Define the polynomial Q(x)
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + d*x - 8

-- State the theorem
theorem coefficient_value (d : ℝ) :
  (∀ x, (x + 2 : ℝ) ∣ Q d x) → d = -2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_value_l3892_389262


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3892_389253

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3892_389253


namespace NUMINAMATH_CALUDE_limit_sin_squared_minus_tan_squared_over_x_fourth_l3892_389296

open Real

theorem limit_sin_squared_minus_tan_squared_over_x_fourth : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((sin x)^2 - (tan x)^2) / x^4 + 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_sin_squared_minus_tan_squared_over_x_fourth_l3892_389296


namespace NUMINAMATH_CALUDE_no_obtuse_tetrahedron_l3892_389238

/-- Definition of a tetrahedron with obtuse angles -/
def ObtuseTetrahedron :=
  {t : Set (ℝ × ℝ × ℝ) | 
    (∃ v₁ v₂ v₃ v₄, t = {v₁, v₂, v₃, v₄}) ∧ 
    (∀ v ∈ t, ∀ α β γ, 
      (α + β + γ = 360) ∧ 
      (90 < α) ∧ (α < 180) ∧ 
      (90 < β) ∧ (β < 180) ∧ 
      (90 < γ) ∧ (γ < 180))}

/-- Theorem stating that an obtuse tetrahedron does not exist -/
theorem no_obtuse_tetrahedron : ¬ ∃ t : Set (ℝ × ℝ × ℝ), t ∈ ObtuseTetrahedron := by
  sorry

end NUMINAMATH_CALUDE_no_obtuse_tetrahedron_l3892_389238


namespace NUMINAMATH_CALUDE_geometry_theorem_l3892_389295

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_theorem 
  (a b : Line) (α β : Plane) 
  (h : perpendicular b α) :
  (parallel_line_plane a α → perpendicular_lines a b) ∧
  (perpendicular b β → parallel_planes α β) := by
  sorry

end NUMINAMATH_CALUDE_geometry_theorem_l3892_389295


namespace NUMINAMATH_CALUDE_expression_value_l3892_389210

theorem expression_value : (10 : ℝ) * 0.5 * 3 / (1/6) = 90 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3892_389210


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l3892_389235

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The problem statement -/
theorem bouncing_ball_distance :
  let initialHeight : ℝ := 80
  let reboundRatio : ℝ := 2/3
  let bounces : ℕ := 3
  totalDistance initialHeight reboundRatio bounces = 257.78 := by
  sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l3892_389235


namespace NUMINAMATH_CALUDE_sequence_condition_l3892_389242

/-- A sequence is monotonically increasing if each term is greater than the previous one. -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The general term of the sequence a_n = n^2 + bn -/
def a (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem sequence_condition (b : ℝ) :
  MonotonicallyIncreasing (a · b) → b > -3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_condition_l3892_389242


namespace NUMINAMATH_CALUDE_find_number_l3892_389251

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 69 :=
by sorry

end NUMINAMATH_CALUDE_find_number_l3892_389251


namespace NUMINAMATH_CALUDE_wire_cutting_l3892_389254

/-- Given a wire cut into two pieces, where the shorter piece is 2/5th of the longer piece
    and is 17.14285714285714 cm long, prove that the total length of the wire before cutting is 60 cm. -/
theorem wire_cutting (shorter_piece : ℝ) (longer_piece : ℝ) :
  shorter_piece = 17.14285714285714 →
  shorter_piece = (2 / 5) * longer_piece →
  shorter_piece + longer_piece = 60 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l3892_389254


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3892_389261

theorem inequality_solution_set (x : ℝ) : 
  (x - 1)^2023 - 2^2023 * x^2023 ≤ x + 1 ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3892_389261


namespace NUMINAMATH_CALUDE_propositions_p_and_q_l3892_389285

theorem propositions_p_and_q : 
  (∀ x : ℝ, (2 : ℝ)^x < (3 : ℝ)^x) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_l3892_389285


namespace NUMINAMATH_CALUDE_sign_distribution_of_products_l3892_389221

theorem sign_distribution_of_products (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ((-a*b > 0 ∧ a*c < 0 ∧ b*d < 0 ∧ c*d < 0) ∨
   (-a*b < 0 ∧ a*c > 0 ∧ b*d > 0 ∧ c*d > 0) ∨
   (-a*b < 0 ∧ a*c < 0 ∧ b*d > 0 ∧ c*d > 0) ∨
   (-a*b < 0 ∧ a*c > 0 ∧ b*d < 0 ∧ c*d > 0) ∨
   (-a*b < 0 ∧ a*c > 0 ∧ b*d > 0 ∧ c*d < 0)) := by
  sorry

end NUMINAMATH_CALUDE_sign_distribution_of_products_l3892_389221


namespace NUMINAMATH_CALUDE_min_sum_squares_l3892_389226

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (min : ℝ), min = 10 ∧ x^2 + y^2 + z^2 ≥ min ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^3 + y₀^3 + z₀^3 - 3*x₀*y₀*z₀ = 8 ∧ x₀^2 + y₀^2 + z₀^2 = min :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3892_389226


namespace NUMINAMATH_CALUDE_rent_increase_problem_l3892_389294

/-- Proves that given the conditions of the rent increase scenario, 
    the original rent of the friend whose rent was increased was $1250 -/
theorem rent_increase_problem (num_friends : ℕ) (initial_avg : ℝ) 
  (increase_percent : ℝ) (new_avg : ℝ) : 
  num_friends = 4 →
  initial_avg = 800 →
  increase_percent = 0.16 →
  new_avg = 850 →
  ∃ (original_rent : ℝ), 
    original_rent * (1 + increase_percent) + 
    (num_friends - 1 : ℝ) * initial_avg = 
    num_friends * new_avg ∧ 
    original_rent = 1250 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l3892_389294


namespace NUMINAMATH_CALUDE_condition_analysis_l3892_389240

theorem condition_analysis (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l3892_389240


namespace NUMINAMATH_CALUDE_function_has_zero_l3892_389252

theorem function_has_zero (m : ℝ) : ∃ x : ℝ, x^3 + 5*m*x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_has_zero_l3892_389252


namespace NUMINAMATH_CALUDE_jane_age_problem_l3892_389263

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem jane_age_problem :
  ∃ j : ℕ, j > 0 ∧ is_perfect_square (j - 2) ∧ is_perfect_cube (j + 2) ∧
  ∀ k : ℕ, k > 0 → is_perfect_square (k - 2) → is_perfect_cube (k + 2) → j ≤ k :=
sorry

end NUMINAMATH_CALUDE_jane_age_problem_l3892_389263


namespace NUMINAMATH_CALUDE_coin_combination_difference_l3892_389269

def coin_values : List ℕ := [10, 20, 50]
def target_amount : ℕ := 45

def valid_combination (coins : List ℕ) : Prop :=
  coins.all (λ c => c ∈ coin_values) ∧ coins.sum = target_amount

def num_coins (coins : List ℕ) : ℕ := coins.length

theorem coin_combination_difference :
  ∃ (min_coins max_coins : List ℕ),
    valid_combination min_coins ∧
    valid_combination max_coins ∧
    (∀ coins, valid_combination coins → num_coins min_coins ≤ num_coins coins) ∧
    (∀ coins, valid_combination coins → num_coins coins ≤ num_coins max_coins) ∧
    num_coins max_coins - num_coins min_coins = 0 :=
sorry

end NUMINAMATH_CALUDE_coin_combination_difference_l3892_389269


namespace NUMINAMATH_CALUDE_S_100_equals_10100_l3892_389208

/-- The number of integers in the solution set for x^2 - x < 2nx -/
def a (n : ℕ+) : ℕ := 2 * n

/-- The sum of the first n terms of the sequence {a_n} -/
def S (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem stating that S_100 equals 10100 -/
theorem S_100_equals_10100 : S 100 = 10100 := by sorry

end NUMINAMATH_CALUDE_S_100_equals_10100_l3892_389208


namespace NUMINAMATH_CALUDE_complex_imaginary_condition_l3892_389277

theorem complex_imaginary_condition (m : ℝ) :
  (∃ z : ℂ, z = Complex.mk (3*m - 2) (m - 1) ∧ z.re = 0) → m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_imaginary_condition_l3892_389277


namespace NUMINAMATH_CALUDE_discount_calculation_l3892_389283

/-- Given a bill amount and a discount for double the time, calculate the discount for the original time. -/
theorem discount_calculation (bill_amount : ℝ) (double_time_discount : ℝ) 
  (h1 : bill_amount = 110) 
  (h2 : double_time_discount = 18.33) : 
  ∃ (original_discount : ℝ), original_discount = 9.165 ∧ 
  original_discount = double_time_discount / 2 := by
  sorry

#check discount_calculation

end NUMINAMATH_CALUDE_discount_calculation_l3892_389283


namespace NUMINAMATH_CALUDE_cofactor_sum_l3892_389286

/-- The algebraic cofactor function of the element 7 in the given determinant -/
def f (x a : ℝ) : ℝ := -x^2 - a*x + 2

/-- The theorem stating that if the solution set of f(x) > 0 is (-1, b), then a + b = 1 -/
theorem cofactor_sum (a b : ℝ) : 
  (∀ x, f x a > 0 ↔ -1 < x ∧ x < b) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_cofactor_sum_l3892_389286


namespace NUMINAMATH_CALUDE_factorization_proof_l3892_389211

variables (a x y : ℝ)

theorem factorization_proof :
  (ax^2 - 7*a*x + 6*a = a*(x-6)*(x-1)) ∧
  (x*y^2 - 9*x = x*(y+3)*(y-3)) ∧
  (1 - x^2 + 2*x*y - y^2 = (1+x-y)*(1-x+y)) ∧
  (8*(x^2 - 2*y^2) - x*(7*x+y) + x*y = (x+4*y)*(x-4*y)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_proof_l3892_389211


namespace NUMINAMATH_CALUDE_system_solution_range_l3892_389232

/-- Given a system of equations 2x + y = 1 + 4a and x + 2y = 2 - a,
    if x + y > 0, then a > -1 -/
theorem system_solution_range (x y a : ℝ) 
  (eq1 : 2 * x + y = 1 + 4 * a)
  (eq2 : x + 2 * y = 2 - a)
  (h : x + y > 0) : 
  a > -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_range_l3892_389232


namespace NUMINAMATH_CALUDE_domino_arrangement_theorem_l3892_389266

/-- Represents a domino piece -/
structure Domino :=
  (first : Nat)
  (second : Nat)
  (h1 : first ≤ 6)
  (h2 : second ≤ 6)

/-- Represents a set of dominoes -/
def DominoSet := List Domino

/-- Represents a square frame made of dominoes -/
structure Frame :=
  (dominoes : List Domino)
  (side_sum : Nat)

/-- The total number of points in a standard set of dominoes minus doubles 3, 4, 5, and 6 -/
def total_points : Nat := 132

/-- The number of frames to be formed -/
def num_frames : Nat := 3

/-- Theorem: It's possible to arrange 24 dominoes into 3 square frames with equal side sums -/
theorem domino_arrangement_theorem (dominoes : DominoSet) 
  (h1 : dominoes.length = 24)
  (h2 : (dominoes.map (λ d => d.first + d.second)).sum = total_points) :
  ∃ (frames : List Frame), 
    frames.length = num_frames ∧ 
    (∀ f ∈ frames, f.dominoes.length = 8 ∧ 
      (f.dominoes.map (λ d => d.first + d.second)).sum = total_points / num_frames) ∧
    (∀ f ∈ frames, ∀ side : List Domino, side.length = 3 → 
      (side.map (λ d => d.first + d.second)).sum = f.side_sum) := by
  sorry


end NUMINAMATH_CALUDE_domino_arrangement_theorem_l3892_389266


namespace NUMINAMATH_CALUDE_events_A_D_independent_l3892_389212

structure Ball :=
  (label : Nat)

def Ω : Type := Ball × Ball

def P : Set Ω → ℝ := sorry

def A : Set Ω := {ω : Ω | ω.fst.label = 1}
def D : Set Ω := {ω : Ω | ω.fst.label + ω.snd.label = 7}

theorem events_A_D_independent :
  P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_events_A_D_independent_l3892_389212


namespace NUMINAMATH_CALUDE_sum_of_data_l3892_389287

theorem sum_of_data (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a = 12) :
  a + b + c = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_data_l3892_389287


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_specific_geometric_series_ratio_l3892_389265

/-- For an infinite geometric series with first term a and sum S, 
    the common ratio r is given by r = 1 - (a / S) -/
theorem infinite_geometric_series_ratio 
  (a : ℝ) (S : ℝ) (h1 : a > 0) (h2 : S > a) :
  let r := 1 - (a / S)
  (∀ n : ℕ, a * r^n = a * (1 - a/S)^n) ∧ 
  (∑' n, a * r^n = S) →
  r = (S - a) / S :=
sorry

/-- The common ratio of an infinite geometric series with 
    first term 520 and sum 3250 is 273/325 -/
theorem specific_geometric_series_ratio :
  let a : ℝ := 520
  let S : ℝ := 3250
  let r := 1 - (a / S)
  (∀ n : ℕ, a * r^n = 520 * (1 - 520/3250)^n) ∧ 
  (∑' n, a * r^n = 3250) →
  r = 273 / 325 :=
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_specific_geometric_series_ratio_l3892_389265


namespace NUMINAMATH_CALUDE_time_puzzle_l3892_389259

theorem time_puzzle : ∃ x : ℝ, 
  0 ≤ x ∧ x ≤ 24 ∧ 
  (x / 4) + ((24 - x) / 2) = x ∧ 
  x = 9.6 := by
sorry

end NUMINAMATH_CALUDE_time_puzzle_l3892_389259


namespace NUMINAMATH_CALUDE_kevin_kangaroo_hops_l3892_389204

/-- The sum of a geometric series with first term 1/4, common ratio 3/4, and 6 terms -/
def geometric_sum : ℚ :=
  let a : ℚ := 1/4
  let r : ℚ := 3/4
  let n : ℕ := 6
  a * (1 - r^n) / (1 - r)

/-- Theorem stating that the geometric sum equals 3367/4096 -/
theorem kevin_kangaroo_hops : geometric_sum = 3367/4096 := by
  sorry

end NUMINAMATH_CALUDE_kevin_kangaroo_hops_l3892_389204
