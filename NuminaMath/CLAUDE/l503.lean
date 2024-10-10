import Mathlib

namespace max_value_expression_l503_50385

theorem max_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * (x + y + z)^2) / ((x + y)^2 * (y + z)^2) ≤ (1 : ℝ) / 4 := by
  sorry

end max_value_expression_l503_50385


namespace problem_statement_l503_50337

theorem problem_statement (x : ℝ) (h : x = Real.sqrt 2) : 
  (x + 2)^2 - 4*x*(x + 1) = -2 := by
  sorry

end problem_statement_l503_50337


namespace tangent_curves_a_value_l503_50364

theorem tangent_curves_a_value (a : ℝ) : 
  let f (x : ℝ) := x + Real.log x
  let g (x : ℝ) := a * x^2 + (a + 2) * x + 1
  let f' (x : ℝ) := 1 + 1 / x
  let g' (x : ℝ) := 2 * a * x + (a + 2)
  (f 1 = g 1) ∧ 
  (f' 1 = g' 1) ∧ 
  (∀ x ≠ 1, f x ≠ g x) →
  a = 8 := by
sorry

end tangent_curves_a_value_l503_50364


namespace gp_common_ratio_l503_50328

/-- Geometric progression properties -/
structure GeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  n : ℕ  -- number of terms
  last : ℝ  -- last term
  sum : ℝ  -- sum of terms

/-- Theorem: Common ratio of a specific geometric progression -/
theorem gp_common_ratio 
  (gp : GeometricProgression) 
  (h1 : gp.a = 9)
  (h2 : gp.last = 1/3)
  (h3 : gp.sum = 40/3) :
  gp.r = 1/3 := by
  sorry

end gp_common_ratio_l503_50328


namespace abs_a_gt_abs_b_l503_50361

theorem abs_a_gt_abs_b (a b : ℝ) (ha : a > 0) (hb : b < 0) (hab : a + b > 0) : |a| > |b| := by
  sorry

end abs_a_gt_abs_b_l503_50361


namespace difference_from_averages_l503_50399

theorem difference_from_averages (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50) 
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 := by
sorry

end difference_from_averages_l503_50399


namespace girls_in_college_l503_50345

theorem girls_in_college (total_students : ℕ) (boys_ratio girls_ratio : ℕ) : 
  total_students = 546 →
  boys_ratio = 8 →
  girls_ratio = 5 →
  ∃ (num_girls : ℕ), num_girls = 210 ∧ 
    boys_ratio * num_girls + girls_ratio * num_girls = girls_ratio * total_students :=
by
  sorry

end girls_in_college_l503_50345


namespace adult_ticket_cost_l503_50362

theorem adult_ticket_cost (num_children num_adults : ℕ) (child_ticket_cost total_cost : ℚ) 
  (h1 : num_children = 6)
  (h2 : num_adults = 10)
  (h3 : child_ticket_cost = 10)
  (h4 : total_cost = 220)
  : (total_cost - num_children * child_ticket_cost) / num_adults = 16 := by
  sorry

end adult_ticket_cost_l503_50362


namespace bus_capacity_l503_50303

theorem bus_capacity (rows : ℕ) (sections_per_row : ℕ) (students_per_section : ℕ) :
  rows = 13 →
  sections_per_row = 2 →
  students_per_section = 2 →
  rows * sections_per_row * students_per_section = 52 := by
  sorry

end bus_capacity_l503_50303


namespace log_relationship_l503_50329

theorem log_relationship : ∀ (a b : ℝ), 
  a = Real.log 135 / Real.log 4 → 
  b = Real.log 45 / Real.log 2 → 
  a = b / 2 := by
sorry

end log_relationship_l503_50329


namespace existence_of_A_for_any_E_l503_50393

/-- Property P: A sequence is a permutation of {1, 2, ..., n} -/
def has_property_P (A : List ℕ) : Prop :=
  A.length ≥ 2 ∧ A.Nodup ∧ ∀ i, i ∈ A → i ∈ Finset.range A.length

/-- T(A) sequence definition -/
def T (A : List ℕ) : List ℕ :=
  List.zipWith (fun a b => if a < b then 1 else 0) A A.tail

theorem existence_of_A_for_any_E (n : ℕ) (E : List ℕ) 
    (h_n : n ≥ 2) 
    (h_E_length : E.length = n - 1) 
    (h_E_elements : ∀ e ∈ E, e = 0 ∨ e = 1) :
    ∃ A : List ℕ, has_property_P A ∧ T A = E :=
  sorry

end existence_of_A_for_any_E_l503_50393


namespace min_alpha_value_l503_50309

/-- Definition of α-level quasi-periodic function -/
def is_alpha_quasi_periodic (f : ℝ → ℝ) (D : Set ℝ) (α : ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x ∈ D, α * f x = f (x + T)

/-- The function f on the domain [1,+∞) -/
noncomputable def f : ℝ → ℝ
| x => if 1 ≤ x ∧ x < 2 then 2^x * (2*x + 1) else 0  -- We define f only for [1,2) as given

/-- Theorem statement -/
theorem min_alpha_value :
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y) →  -- Monotonically increasing
  (∀ α, is_alpha_quasi_periodic f (Set.Ici 1) α → α ≥ 10/3) ∧
  (is_alpha_quasi_periodic f (Set.Ici 1) (10/3)) :=
by sorry

end min_alpha_value_l503_50309


namespace complex_triangle_problem_l503_50378

theorem complex_triangle_problem (x y z : ℂ) 
  (eq1 : x^2 + y^2 + z^2 = x*y + y*z + z*x)
  (eq2 : Complex.abs (x + y + z) = 21)
  (eq3 : Complex.abs (x - y) = 2 * Real.sqrt 3)
  (eq4 : Complex.abs x = 3 * Real.sqrt 3) :
  Complex.abs y^2 + Complex.abs z^2 = 132 := by
  sorry

end complex_triangle_problem_l503_50378


namespace sphere_triangle_distance_l503_50304

/-- The distance from the center of a sphere to the plane of a tangent triangle -/
theorem sphere_triangle_distance (r : ℝ) (a b c : ℝ) (h_sphere : r = 10) 
  (h_triangle : a = 18 ∧ b = 18 ∧ c = 30) (h_tangent : True) : 
  ∃ d : ℝ, d = (10 * Real.sqrt 37) / 33 ∧ 
  d^2 + ((a + b + c) / 2 * (2 * a * b) / (a + b + c))^2 = r^2 := by
  sorry

end sphere_triangle_distance_l503_50304


namespace at_least_one_greater_than_point_seven_l503_50358

theorem at_least_one_greater_than_point_seven (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x > 0.7 ∨ y > 0.7 ∨ (1 / (x + y)) > 0.7 := by
  sorry

end at_least_one_greater_than_point_seven_l503_50358


namespace pentagon_extension_l503_50336

/-- Given a pentagon ABCDE with extended sides, prove the relation between A and A', B', C', D', E' -/
theorem pentagon_extension (A B C D E A' B' C' D' E' : ℝ × ℝ) 
  (h1 : B = (A + A') / 2)
  (h2 : C = (B + B') / 2)
  (h3 : D = (C + C') / 2)
  (h4 : E = (D + D') / 2)
  (h5 : A = (E + E') / 2) :
  A = (1/32 : ℝ) • A' + (1/16 : ℝ) • B' + (1/8 : ℝ) • C' + (1/4 : ℝ) • D' + (1/2 : ℝ) • E' :=
by sorry

end pentagon_extension_l503_50336


namespace trigonometric_identity_l503_50308

theorem trigonometric_identity : 
  Real.sin (40 * π / 180) * Real.sin (10 * π / 180) + 
  Real.cos (40 * π / 180) * Real.sin (80 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end trigonometric_identity_l503_50308


namespace wall_ratio_l503_50305

/-- Proves that for a rectangular wall with given dimensions, the ratio of length to height is 7:1 -/
theorem wall_ratio (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  volume = l * w * h →
  w = 4 →
  volume = 16128 →
  l / h = 7 := by
  sorry

end wall_ratio_l503_50305


namespace unique_solution_on_sphere_l503_50310

theorem unique_solution_on_sphere (x y : ℝ) :
  (x - 8)^2 + (y - 9)^2 + (x - y)^2 = 1/3 →
  x = 8 + 1/3 ∧ y = 8 + 2/3 := by
  sorry

end unique_solution_on_sphere_l503_50310


namespace coefficient_x_fifth_power_l503_50334

theorem coefficient_x_fifth_power (x : ℝ) : 
  (Finset.range 10).sum (λ k => (Nat.choose 9 k : ℝ) * x^(9 - k) * (3 * Real.sqrt 2)^k) = 
  40824 * x^5 + (Finset.range 10).sum (λ k => if k ≠ 4 then (Nat.choose 9 k : ℝ) * x^(9 - k) * (3 * Real.sqrt 2)^k else 0) := by
sorry

end coefficient_x_fifth_power_l503_50334


namespace crazy_silly_school_series_l503_50320

/-- The number of different books in the 'crazy silly school' series -/
def num_books : ℕ := sorry

/-- The number of different movies in the 'crazy silly school' series -/
def num_movies : ℕ := 11

/-- The number of books you have read -/
def books_read : ℕ := 13

/-- The number of movies you have watched -/
def movies_watched : ℕ := 12

theorem crazy_silly_school_series :
  (books_read = movies_watched + 1) →
  (num_books = 12) :=
by sorry

end crazy_silly_school_series_l503_50320


namespace units_digit_of_power_difference_l503_50349

theorem units_digit_of_power_difference : ∃ n : ℕ, (5^2019 - 3^2019) % 10 = 8 := by sorry

end units_digit_of_power_difference_l503_50349


namespace intersection_point_congruences_l503_50327

/-- Proves that (15, 8) is the unique intersection point of two congruences modulo 20 -/
theorem intersection_point_congruences : ∃! (x y : ℕ), 
  x < 20 ∧ 
  y < 20 ∧ 
  (7 * x + 3) % 20 = y ∧ 
  (13 * x + 18) % 20 = y :=
sorry

end intersection_point_congruences_l503_50327


namespace compare_expressions_l503_50330

theorem compare_expressions (a : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) := by
  sorry

end compare_expressions_l503_50330


namespace cell_population_after_9_days_l503_50397

/-- Represents the growth and mortality of a cell population over time -/
def cell_population (initial_cells : ℕ) (growth_rate : ℚ) (mortality_rate : ℚ) (cycles : ℕ) : ℕ :=
  sorry

/-- Theorem stating the cell population after 9 days -/
theorem cell_population_after_9_days :
  cell_population 5 2 (9/10) 3 = 28 :=
sorry

end cell_population_after_9_days_l503_50397


namespace fred_initial_cards_l503_50352

/-- Given that Fred gave away 18 cards, found 40 new cards, and ended up with 48 cards,
    prove that he must have started with 26 cards. -/
theorem fred_initial_cards :
  ∀ (initial_cards given_away new_cards final_cards : ℕ),
    given_away = 18 →
    new_cards = 40 →
    final_cards = 48 →
    initial_cards - given_away + new_cards = final_cards →
    initial_cards = 26 := by
  sorry

end fred_initial_cards_l503_50352


namespace zero_lt_m_lt_one_necessary_not_sufficient_l503_50302

-- Define the quadratic equation
def quadratic_eq (m : ℝ) (x : ℝ) : Prop := x^2 + x + m^2 - 1 = 0

-- Define the condition for two real roots with different signs
def has_two_real_roots_diff_signs (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₂ < 0 ∧ quadratic_eq m x₁ ∧ quadratic_eq m x₂

-- Theorem stating that 0 < m < 1 is a necessary but not sufficient condition
theorem zero_lt_m_lt_one_necessary_not_sufficient :
  (∀ m : ℝ, has_two_real_roots_diff_signs m → 0 < m ∧ m < 1) ∧
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ ¬has_two_real_roots_diff_signs m) :=
sorry

end zero_lt_m_lt_one_necessary_not_sufficient_l503_50302


namespace union_equals_reals_subset_condition_l503_50383

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a < x ∧ x < 3 + a}
def B : Set ℝ := {x | x ≤ -1 ∨ x ≥ 1}

-- Theorem 1: A ∪ B = ℝ iff -2 ≤ a ≤ -1
theorem union_equals_reals (a : ℝ) : A a ∪ B = Set.univ ↔ -2 ≤ a ∧ a ≤ -1 := by
  sorry

-- Theorem 2: A ⊆ B iff a ≤ -4 or a ≥ 1
theorem subset_condition (a : ℝ) : A a ⊆ B ↔ a ≤ -4 ∨ a ≥ 1 := by
  sorry

end union_equals_reals_subset_condition_l503_50383


namespace probability_is_34_39_l503_50395

-- Define the total number of students and enrollments
def total_students : ℕ := 40
def french_enrollment : ℕ := 28
def spanish_enrollment : ℕ := 26
def german_enrollment : ℕ := 15
def french_spanish : ℕ := 10
def french_german : ℕ := 6
def spanish_german : ℕ := 8
def all_three : ℕ := 3

-- Define the function to calculate the probability
def probability_different_classes : ℚ := by sorry

-- Theorem statement
theorem probability_is_34_39 : 
  probability_different_classes = 34 / 39 := by sorry

end probability_is_34_39_l503_50395


namespace ellipse_k_range_l503_50316

/-- Represents an ellipse equation with parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (b^2) + y^2 / (a^2) = 1 ↔ x^2 + k * y^2 = 2

/-- Foci are on the y-axis if the equation is in the form x^2/b^2 + y^2/a^2 = 1 with a > b -/
def foci_on_y_axis (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (b^2) + y^2 / (a^2) = 1 ↔ x^2 + k * y^2 = 2

/-- The main theorem stating the range of k -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ∧ foci_on_y_axis k → 0 < k ∧ k < 1 :=
sorry

end ellipse_k_range_l503_50316


namespace symmetric_polynomial_square_factor_l503_50382

/-- A polynomial in two variables that is symmetric in its arguments -/
def SymmetricPolynomial (R : Type) [CommRing R] :=
  {p : R → R → R // ∀ x y, p x y = p y x}

theorem symmetric_polynomial_square_factor
  {R : Type} [CommRing R] (p : SymmetricPolynomial R)
  (h : ∃ q : R → R → R, ∀ x y, p.val x y = (x - y) * q x y) :
  ∃ r : R → R → R, ∀ x y, p.val x y = (x - y)^2 * r x y := by
  sorry

end symmetric_polynomial_square_factor_l503_50382


namespace geometric_sequence_m_range_l503_50347

theorem geometric_sequence_m_range 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (m : ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = q * a n)
  (h_q_range : q > Real.rpow 5 (1/3) ∧ q < 2)
  (h_equation : m * a 6 * a 7 = a 8 ^ 2 - 2 * a 4 * a 9) :
  m > 3 ∧ m < 6 := by
sorry

end geometric_sequence_m_range_l503_50347


namespace escalator_ride_time_main_escalator_theorem_l503_50381

/-- Represents the time it takes Leo to ride an escalator in different scenarios -/
structure EscalatorRide where
  stationary_walk : ℝ  -- Time to walk down stationary escalator
  moving_walk : ℝ      -- Time to walk down moving escalator
  no_walk : ℝ          -- Time to ride without walking (to be proven)

/-- Theorem stating that given the conditions, the time to ride without walking is 48 seconds -/
theorem escalator_ride_time (ride : EscalatorRide) 
  (h1 : ride.stationary_walk = 80)
  (h2 : ride.moving_walk = 30) : 
  ride.no_walk = 48 := by
  sorry

/-- Main theorem combining all conditions and the result -/
theorem main_escalator_theorem : 
  ∃ (ride : EscalatorRide), ride.stationary_walk = 80 ∧ ride.moving_walk = 30 ∧ ride.no_walk = 48 := by
  sorry

end escalator_ride_time_main_escalator_theorem_l503_50381


namespace hotel_price_difference_l503_50312

-- Define the charges for single rooms at hotels P, R, and G
def single_room_P (r g : ℝ) : ℝ := 0.45 * r
def single_room_P' (r g : ℝ) : ℝ := 0.90 * g

-- Define the charges for double rooms at hotels P, R, and G
def double_room_P (r g : ℝ) : ℝ := 0.70 * r
def double_room_P' (r g : ℝ) : ℝ := 0.80 * g

-- Define the charges for suites at hotels P, R, and G
def suite_P (r g : ℝ) : ℝ := 0.60 * r
def suite_P' (r g : ℝ) : ℝ := 0.85 * g

theorem hotel_price_difference (r_single g_single r_double g_double : ℝ) :
  single_room_P r_single g_single = single_room_P' r_single g_single ∧
  double_room_P r_double g_double = double_room_P' r_double g_double →
  (r_single / g_single - 1) * 100 - (r_double / g_double - 1) * 100 = 85.71 :=
by sorry

end hotel_price_difference_l503_50312


namespace arithmetic_sequence_sum_l503_50389

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  S : ℕ → ℝ
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Theorem: For an arithmetic sequence where S_17 = 17/2, a_3 + a_15 = 1 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
    (h : seq.S 17 = 17 / 2) : seq.a 3 + seq.a 15 = 1 := by
  sorry

end arithmetic_sequence_sum_l503_50389


namespace wendy_total_profit_l503_50300

/-- Represents a fruit sale --/
structure FruitSale where
  price : Float
  quantity : Nat
  profit_margin : Float
  discount : Float

/-- Represents a day's sales --/
structure DaySales where
  morning_apples : FruitSale
  morning_oranges : FruitSale
  morning_bananas : FruitSale
  afternoon_apples : FruitSale
  afternoon_oranges : FruitSale
  afternoon_bananas : FruitSale

/-- Represents unsold fruits --/
structure UnsoldFruits where
  banana_quantity : Nat
  banana_price : Float
  banana_discount : Float
  banana_profit_margin : Float
  orange_quantity : Nat
  orange_price : Float
  orange_discount : Float
  orange_profit_margin : Float

/-- Calculate profit for a single fruit sale --/
def calculate_profit (sale : FruitSale) : Float :=
  sale.price * sale.quantity.toFloat * (1 - sale.discount) * sale.profit_margin

/-- Calculate total profit for a day --/
def calculate_day_profit (day : DaySales) : Float :=
  calculate_profit day.morning_apples +
  calculate_profit day.morning_oranges +
  calculate_profit day.morning_bananas +
  calculate_profit day.afternoon_apples +
  calculate_profit day.afternoon_oranges +
  calculate_profit day.afternoon_bananas

/-- Calculate profit from unsold fruits --/
def calculate_unsold_profit (unsold : UnsoldFruits) : Float :=
  unsold.banana_quantity.toFloat * unsold.banana_price * (1 - unsold.banana_discount) * unsold.banana_profit_margin +
  unsold.orange_quantity.toFloat * unsold.orange_price * (1 - unsold.orange_discount) * unsold.orange_profit_margin

/-- Main theorem: Wendy's total profit for the week --/
theorem wendy_total_profit (day1 day2 : DaySales) (unsold : UnsoldFruits) :
  calculate_day_profit day1 + calculate_day_profit day2 + calculate_unsold_profit unsold = 84.07 := by
  sorry

end wendy_total_profit_l503_50300


namespace fraction_subtraction_l503_50396

theorem fraction_subtraction : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end fraction_subtraction_l503_50396


namespace sugar_consumption_reduction_l503_50394

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 6)
  (h2 : new_price = 7.5)
  (h3 : initial_price > 0 ∧ new_price > 0) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 20 := by
  sorry

end sugar_consumption_reduction_l503_50394


namespace smallest_n_for_inequality_l503_50375

-- Define a function to represent the power tower of 2's
def powerTower (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 ^ (powerTower n)

-- Define the right-hand side of the inequality
def rightHandSide : ℕ := 3^(3^(3^3))

-- Theorem statement
theorem smallest_n_for_inequality :
  (∀ k < 6, powerTower k ≤ rightHandSide) ∧
  (powerTower 6 > rightHandSide) := by
  sorry

end smallest_n_for_inequality_l503_50375


namespace solution_when_k_gt_neg_one_no_solution_when_k_eq_neg_one_solution_when_k_lt_neg_one_k_upper_bound_l503_50387

/-- The function f(x) defined in the problem -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (1 - k)*x + 2 - k

/-- Theorem stating the solution for f(x) < 2 when k > -1 -/
theorem solution_when_k_gt_neg_one (k : ℝ) (x : ℝ) (h : k > -1) :
  f k x < 2 ↔ -1 < x ∧ x < k :=
sorry

/-- Theorem stating there's no solution for f(x) < 2 when k = -1 -/
theorem no_solution_when_k_eq_neg_one (x : ℝ) :
  ¬(f (-1) x < 2) :=
sorry

/-- Theorem stating the solution for f(x) < 2 when k < -1 -/
theorem solution_when_k_lt_neg_one (k : ℝ) (x : ℝ) (h : k < -1) :
  f k x < 2 ↔ k < x ∧ x < -1 :=
sorry

/-- Theorem stating the upper bound of k when f(n) + 11 ≥ 0 for all natural numbers n -/
theorem k_upper_bound (k : ℝ) (h : ∀ (n : ℕ), f k n + 11 ≥ 0) :
  k ≤ 25/4 :=
sorry

end solution_when_k_gt_neg_one_no_solution_when_k_eq_neg_one_solution_when_k_lt_neg_one_k_upper_bound_l503_50387


namespace sum_of_quotient_dividend_divisor_l503_50372

theorem sum_of_quotient_dividend_divisor (N : ℕ) (h : N = 40) : 
  (N / 2) + N + 2 = 62 := by
  sorry

end sum_of_quotient_dividend_divisor_l503_50372


namespace position_of_2007_l503_50398

/-- Represents the position of a number in the table -/
structure Position where
  row : ℕ
  column : ℕ

/-- The arrangement of positive odd numbers in 5 columns -/
def arrangement (n : ℕ) : Position :=
  let cycle := (n - 1) / 8
  let position := (n - 1) % 8
  match position with
  | 0 => ⟨cycle * 2 + 1, 2⟩
  | 1 => ⟨cycle * 2 + 1, 3⟩
  | 2 => ⟨cycle * 2 + 1, 4⟩
  | 3 => ⟨cycle * 2 + 1, 5⟩
  | 4 => ⟨cycle * 2 + 2, 1⟩
  | 5 => ⟨cycle * 2 + 2, 2⟩
  | 6 => ⟨cycle * 2 + 2, 3⟩
  | 7 => ⟨cycle * 2 + 2, 4⟩
  | _ => ⟨0, 0⟩  -- This case should never occur

theorem position_of_2007 : arrangement 2007 = ⟨251, 5⟩ := by
  sorry

end position_of_2007_l503_50398


namespace emily_flower_spending_l503_50391

def flower_price : ℝ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2
def discount_threshold : ℕ := 3
def discount_rate : ℝ := 0.2

def total_flowers : ℕ := roses_bought + daisies_bought

def apply_discount (price : ℝ) : ℝ :=
  if total_flowers > discount_threshold then
    price * (1 - discount_rate)
  else
    price

theorem emily_flower_spending :
  apply_discount (flower_price * (roses_bought + daisies_bought : ℝ)) = 9.60 := by
  sorry

end emily_flower_spending_l503_50391


namespace broadcast_end_date_prove_broadcast_end_date_l503_50332

/-- Represents a date with year, month, and day. -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Represents a day of the week. -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the broadcasting schedule. -/
structure BroadcastSchedule where
  wednesday : Nat
  friday : Nat
  saturday : Nat
  sunday : Nat

/-- Calculates the end date of the broadcast. -/
def calculateEndDate (startDate : Date) (totalEpisodes : Nat) (schedule : BroadcastSchedule) : Date :=
  sorry

/-- Determines the day of the week for a given date. -/
def getDayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Main theorem to prove -/
theorem broadcast_end_date (startDate : Date) (totalEpisodes : Nat) (schedule : BroadcastSchedule) :
  let endDate := calculateEndDate startDate totalEpisodes schedule
  endDate.year = 2016 ∧ endDate.month = 5 ∧ endDate.day = 29 ∧
  getDayOfWeek endDate = DayOfWeek.Sunday :=
by
  sorry

/-- Initial conditions -/
def initialDate : Date := { year := 2015, month := 12, day := 26 }
def episodeCount : Nat := 135
def broadcastSchedule : BroadcastSchedule := { wednesday := 1, friday := 1, saturday := 2, sunday := 2 }

/-- Proof of the main theorem with initial conditions -/
theorem prove_broadcast_end_date :
  let endDate := calculateEndDate initialDate episodeCount broadcastSchedule
  endDate.year = 2016 ∧ endDate.month = 5 ∧ endDate.day = 29 ∧
  getDayOfWeek endDate = DayOfWeek.Sunday :=
by
  sorry

end broadcast_end_date_prove_broadcast_end_date_l503_50332


namespace height_difference_climbing_l503_50306

/-- Proves that the difference in height climbed between two people with different climbing rates over a given time is equal to the product of the time and the difference in their climbing rates. -/
theorem height_difference_climbing (matt_rate jason_rate : ℝ) (time : ℝ) 
  (h1 : matt_rate = 6)
  (h2 : jason_rate = 12)
  (h3 : time = 7) :
  jason_rate * time - matt_rate * time = (jason_rate - matt_rate) * time :=
by sorry

/-- Calculates the actual height difference between Jason and Matt after 7 minutes of climbing. -/
def actual_height_difference (matt_rate jason_rate : ℝ) (time : ℝ) 
  (h1 : matt_rate = 6)
  (h2 : jason_rate = 12)
  (h3 : time = 7) : ℝ :=
jason_rate * time - matt_rate * time

#eval actual_height_difference 6 12 7 rfl rfl rfl

end height_difference_climbing_l503_50306


namespace two_solutions_exist_l503_50314

def A (x : ℝ) : Set ℝ := {0, 1, 2, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem two_solutions_exist :
  ∃! (s : Set ℝ), (∃ (x₁ x₂ : ℝ), s = {x₁, x₂} ∧ 
    ∀ (x : ℝ), (A x ∪ B x = A x) ↔ (x ∈ s)) ∧ 
    (∀ (x : ℝ), x ∈ s → x^2 = 2) :=
sorry

end two_solutions_exist_l503_50314


namespace divisibility_by_twelve_l503_50311

theorem divisibility_by_twelve (n : Nat) : n < 10 → (516 * 10 + n) % 12 = 0 ↔ n = 0 ∨ n = 4 := by
  sorry

end divisibility_by_twelve_l503_50311


namespace unique_solution_for_equation_l503_50315

theorem unique_solution_for_equation :
  ∃! n : ℚ, (1 : ℚ) / (n + 2) + (2 : ℚ) / (n + 2) + (n + 1) / (n + 2) = 3 ∧ n = -1 := by
  sorry

end unique_solution_for_equation_l503_50315


namespace unique_common_difference_l503_50313

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  d : ℝ  -- common difference
  n : ℕ  -- number of terms
  third_term_is_7 : a + 2 * d = 7
  last_term_is_37 : a + (n - 1) * d = 37
  sum_is_198 : n * (2 * a + (n - 1) * d) / 2 = 198

/-- Theorem stating the existence and uniqueness of the common difference -/
theorem unique_common_difference (seq : ArithmeticSequence) : 
  ∃! d : ℝ, seq.d = d := by sorry

end unique_common_difference_l503_50313


namespace abs_ratio_greater_than_one_l503_50326

theorem abs_ratio_greater_than_one {a b : ℝ} (h1 : a < b) (h2 : b < 0) : |a| / |b| > 1 := by
  sorry

end abs_ratio_greater_than_one_l503_50326


namespace perfect_square_trinomial_l503_50319

theorem perfect_square_trinomial (a k : ℝ) : 
  (∃ b : ℝ, a^2 - k*a + 25 = (a - b)^2) → k = 10 ∨ k = -10 := by
  sorry

end perfect_square_trinomial_l503_50319


namespace day_of_week_proof_l503_50384

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Calculate the number of days between two dates -/
def daysBetween (d1 d2 : Date) : Int :=
  sorry

/-- Get the day of the week for a given date -/
def getDayOfWeek (d : Date) (knownDate : Date) (knownDay : DayOfWeek) : DayOfWeek :=
  sorry

theorem day_of_week_proof :
  let knownDate := Date.mk 1998 4 10
  let knownDay := DayOfWeek.Friday
  let date1 := Date.mk 1918 7 6
  let date2 := Date.mk 2018 6 6
  (getDayOfWeek date1 knownDate knownDay = DayOfWeek.Saturday) ∧
  (getDayOfWeek date2 knownDate knownDay = DayOfWeek.Tuesday) := by
  sorry

end day_of_week_proof_l503_50384


namespace common_difference_of_arithmetic_sequence_l503_50390

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence 
  (a : ℕ → ℤ) (h : arithmetic_sequence a) (h5 : a 5 = 3) (h6 : a 6 = -2) : 
  ∃ d : ℤ, d = -5 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end common_difference_of_arithmetic_sequence_l503_50390


namespace students_remaining_after_four_stops_l503_50318

theorem students_remaining_after_four_stops :
  let initial_students : ℕ := 60
  let stops : ℕ := 4
  let fraction_remaining : ℚ := 2 / 3
  let final_students := initial_students * fraction_remaining ^ stops
  final_students = 20 := by
  sorry

end students_remaining_after_four_stops_l503_50318


namespace no_special_arrangement_exists_l503_50379

theorem no_special_arrangement_exists : ¬ ∃ (p : Fin 20 → Fin 20), Function.Bijective p ∧
  ∀ (i j : Fin 20), i.val % 10 = j.val % 10 → i ≠ j →
    |p i - p j| - 1 = i.val % 10 := by
  sorry

end no_special_arrangement_exists_l503_50379


namespace largest_four_digit_divisible_by_35_l503_50370

theorem largest_four_digit_divisible_by_35 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 35 = 0 → n ≤ 9975 :=
by sorry

end largest_four_digit_divisible_by_35_l503_50370


namespace min_sum_of_product_1800_l503_50353

theorem min_sum_of_product_1800 (a b c : ℕ+) (h : a * b * c = 1800) :
  (∀ x y z : ℕ+, x * y * z = 1800 → a + b + c ≤ x + y + z) ∧ a + b + c = 64 := by
  sorry

end min_sum_of_product_1800_l503_50353


namespace range_of_m_l503_50359

/-- Given the conditions:
    1. p: |4-x| ≤ 6
    2. q: x^2 - 2x + 1 ≤ 0 (m > 0)
    3. p is not a necessary but not sufficient condition for q
    
    Prove that the range of values for the real number m is m ≥ 9. -/
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |4 - x| ≤ 6 → (x^2 - 2*x + 1 ≤ 0 ∧ m > 0)) →
  (∃ x : ℝ, |4 - x| ≤ 6 ∧ (x^2 - 2*x + 1 > 0 ∨ m ≤ 0)) →
  (∀ x : ℝ, (x^2 - 2*x + 1 ≤ 0 ∧ m > 0) → |4 - x| ≤ 6) →
  m ≥ 9 := by
  sorry

end range_of_m_l503_50359


namespace multiples_of_four_l503_50392

/-- Given a natural number n, if there are exactly 25 multiples of 4
    between n and 108 (inclusive), then n = 12. -/
theorem multiples_of_four (n : ℕ) : 
  (∃ (l : List ℕ), l.length = 25 ∧ 
    (∀ x ∈ l, x % 4 = 0 ∧ n ≤ x ∧ x ≤ 108) ∧
    (∀ y, n ≤ y ∧ y ≤ 108 ∧ y % 4 = 0 → y ∈ l)) →
  n = 12 := by
  sorry

end multiples_of_four_l503_50392


namespace magnitude_of_b_l503_50354

/-- Given vectors a and b in ℝ², prove that |b| = √2 under the given conditions -/
theorem magnitude_of_b (a b : ℝ × ℝ) : 
  a = (-Real.sqrt 3, 1) →
  (a.1 + 2 * b.1) * a.1 + (a.2 + 2 * b.2) * a.2 = 0 →
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2 = 0 →
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2 := by
  sorry


end magnitude_of_b_l503_50354


namespace sum_of_cubes_l503_50360

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 12) : x^3 + y^3 = 935 := by
  sorry

end sum_of_cubes_l503_50360


namespace max_m_over_n_l503_50365

open Real

noncomputable def f (m n x : ℝ) : ℝ := Real.exp (-x) + (n * x) / (m * x + n)

theorem max_m_over_n (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, x ≥ 0 → f m n x ≥ 1) ∧ f m n 0 = 1 →
  m / n ≤ (1 : ℝ) / 2 :=
by sorry

end max_m_over_n_l503_50365


namespace probability_4_club_2_is_1_663_l503_50340

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of 4s in a standard deck -/
def NumberOf4s : ℕ := 4

/-- Number of clubs in a standard deck -/
def NumberOfClubs : ℕ := 13

/-- Number of 2s in a standard deck -/
def NumberOf2s : ℕ := 4

/-- Probability of drawing a 4 as the first card, a club as the second card, 
    and a 2 as the third card from a standard 52-card deck -/
def probability_4_club_2 : ℚ :=
  (NumberOf4s : ℚ) / StandardDeck *
  NumberOfClubs / (StandardDeck - 1) *
  NumberOf2s / (StandardDeck - 2)

theorem probability_4_club_2_is_1_663 : 
  probability_4_club_2 = 1 / 663 := by
  sorry

end probability_4_club_2_is_1_663_l503_50340


namespace max_prime_factors_b_l503_50355

theorem max_prime_factors_b (a b : ℕ+) 
  (h_gcd : (Nat.gcd a b).factors.length = 5)
  (h_lcm : (Nat.lcm a b).factors.length = 20)
  (h_fewer : (b.val.factors.length : ℕ) < a.val.factors.length) :
  b.val.factors.length ≤ 12 := by
  sorry

end max_prime_factors_b_l503_50355


namespace unique_number_l503_50346

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k, n = 13*k

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ k, n = k^2

theorem unique_number : 
  ∃! n : ℕ, 
    is_two_digit n ∧ 
    is_odd n ∧ 
    is_multiple_of_13 n ∧ 
    is_perfect_square (sum_of_digits n) ∧ 
    n = 13 := by sorry

end unique_number_l503_50346


namespace kamals_biology_marks_l503_50307

def english_marks : ℕ := 76
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def total_subjects : ℕ := 5
def average_marks : ℕ := 75

theorem kamals_biology_marks :
  ∃ (biology_marks : ℕ),
    biology_marks = total_subjects * average_marks - (english_marks + mathematics_marks + physics_marks + chemistry_marks) :=
by sorry

end kamals_biology_marks_l503_50307


namespace total_volume_of_cubes_l503_50380

def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume (carl_cubes : ℕ) (kate_cubes : ℕ) (carl_side_length : ℕ) (kate_side_length : ℕ) : ℕ :=
  carl_cubes * cube_volume carl_side_length + kate_cubes * cube_volume kate_side_length

theorem total_volume_of_cubes : total_volume 4 3 3 4 = 300 := by
  sorry

end total_volume_of_cubes_l503_50380


namespace map_distance_conversion_l503_50348

/-- Given a map scale where 312 inches represents 136 km,
    prove that 34 inches on the map corresponds to approximately 14.82 km in actual distance. -/
theorem map_distance_conversion (map_distance : ℝ) (actual_distance : ℝ) (ram_map_distance : ℝ)
  (h1 : map_distance = 312)
  (h2 : actual_distance = 136)
  (h3 : ram_map_distance = 34) :
  ∃ (ε : ℝ), ε > 0 ∧ abs ((actual_distance / map_distance) * ram_map_distance - 14.82) < ε :=
sorry

end map_distance_conversion_l503_50348


namespace unique_three_digit_number_l503_50377

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_ones : ones < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Checks if all digits in a three-digit number are the same -/
def allDigitsSame (n : Nat) : Prop :=
  (n / 100 = (n / 10) % 10) ∧ ((n / 10) % 10 = n % 10)

theorem unique_three_digit_number :
  ∃! (n : ThreeDigitNumber),
    (n.hundreds + n.ones = 5) ∧
    (n.tens = 3) ∧
    (n.hundreds ≠ n.tens) ∧
    (n.tens ≠ n.ones) ∧
    (n.hundreds ≠ n.ones) ∧
    allDigitsSame (n.toNat + 124) ∧
    n.toNat = 431 := by
  sorry

end unique_three_digit_number_l503_50377


namespace henrys_deductions_l503_50366

/-- Henry's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- State tax rate as a decimal -/
def tax_rate : ℚ := 21 / 1000

/-- Fixed community fee in dollars per hour -/
def community_fee : ℚ := 1 / 2

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℚ := 100

/-- Calculate the total deductions in cents -/
def total_deductions : ℚ :=
  hourly_wage * tax_rate * dollars_to_cents + community_fee * dollars_to_cents

theorem henrys_deductions :
  total_deductions = 102.5 := by sorry

end henrys_deductions_l503_50366


namespace floor_sqrt_17_squared_l503_50368

theorem floor_sqrt_17_squared : ⌊Real.sqrt 17⌋^2 = 16 := by
  sorry

end floor_sqrt_17_squared_l503_50368


namespace number_factorization_l503_50373

theorem number_factorization (n : ℤ) : 
  (∃ x y : ℤ, n = x * y ∧ y - x = 6 ∧ x^4 + y^4 = 272) → n = -8 := by
  sorry

end number_factorization_l503_50373


namespace lewis_weekly_earnings_l503_50344

/-- Lewis's earnings during harvest -/
def harvest_earnings : ℕ := 178

/-- Duration of harvest in weeks -/
def harvest_duration : ℕ := 89

/-- Lewis's weekly earnings during harvest -/
def weekly_earnings : ℚ := harvest_earnings / harvest_duration

theorem lewis_weekly_earnings :
  weekly_earnings = 2 :=
sorry

end lewis_weekly_earnings_l503_50344


namespace percentage_difference_l503_50341

theorem percentage_difference : (70 / 100 * 100) - (60 / 100 * 80) = 22 := by
  sorry

end percentage_difference_l503_50341


namespace derivative_problems_l503_50369

open Real

theorem derivative_problems :
  (∀ x : ℝ, x ≠ 0 → deriv (λ x => x * (1 + 2/x + 2/x^2)) x = 1 - 2/x^2) ∧
  (∀ x : ℝ, deriv (λ x => x^4 - 3*x^2 - 5*x + 6) x = 4*x^3 - 6*x - 5) := by
  sorry

end derivative_problems_l503_50369


namespace circular_road_width_l503_50343

theorem circular_road_width 
  (inner_radius outer_radius : ℝ) 
  (h1 : 2 * Real.pi * inner_radius + 2 * Real.pi * outer_radius = 88) 
  (h2 : inner_radius = (1/3) * outer_radius) : 
  outer_radius - inner_radius = 22 / Real.pi := by
sorry

end circular_road_width_l503_50343


namespace pirate_treasure_sum_l503_50374

-- Define a function to convert from base 8 to base 10
def base8ToBase10 (n : Nat) : Nat :=
  let digits := n.digits 8
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Define the theorem
theorem pirate_treasure_sum :
  let silk := 5267
  let stones := 6712
  let spices := 327
  base8ToBase10 silk + base8ToBase10 stones + base8ToBase10 spices = 6488 := by
  sorry


end pirate_treasure_sum_l503_50374


namespace lcm_fraction_even_l503_50325

theorem lcm_fraction_even (n : ℕ) : 
  (n > 0) → (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    n = (Nat.lcm x y + Nat.lcm y z) / Nat.lcm x z) ↔ Even n :=
sorry

end lcm_fraction_even_l503_50325


namespace steak_weight_problem_l503_50338

theorem steak_weight_problem (original_weight : ℝ) : 
  (0.8 * (0.5 * original_weight) = 12) → original_weight = 30 := by
  sorry

end steak_weight_problem_l503_50338


namespace point_not_on_line_l503_50386

theorem point_not_on_line (m k : ℝ) (h1 : m * k > 0) :
  ¬(∃ (x y : ℝ), x = 2000 ∧ y = 0 ∧ y = m * x + k) :=
by sorry

end point_not_on_line_l503_50386


namespace perpendicular_lines_and_circle_l503_50367

-- Define the lines and circle
def l₁ (a x y : ℝ) : Prop := a * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def C (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 8*y + 21 = 0

-- State the theorem
theorem perpendicular_lines_and_circle 
  (a : ℝ) -- Coefficient of x in l₁
  (h_perp : a * 2 + 4 = 0) -- Perpendicularity condition
  : 
  -- Part 1: Intersection point
  (∃ x y : ℝ, l₁ a x y ∧ l₂ x y ∧ x = -1 ∧ y = 0) ∧ 
  -- Part 2: No common points between l₁ and C
  (∀ x y : ℝ, ¬(l₁ a x y ∧ C x y)) :=
by sorry

end perpendicular_lines_and_circle_l503_50367


namespace simplify_trig_expression_l503_50371

theorem simplify_trig_expression :
  7 * 8 * (Real.sin (10 * π / 180) + Real.sin (20 * π / 180)) /
  (Real.cos (10 * π / 180) + Real.cos (20 * π / 180)) =
  Real.tan (15 * π / 180) := by sorry

end simplify_trig_expression_l503_50371


namespace lcm_36_100_l503_50321

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end lcm_36_100_l503_50321


namespace function_symmetry_l503_50323

/-- The function f(x) = (1-x)/(1+x) is symmetric about the line y = x -/
theorem function_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (1 - x) / (1 + x)
  f (f x) = x :=
sorry

end function_symmetry_l503_50323


namespace square_sum_from_sum_and_product_l503_50351

theorem square_sum_from_sum_and_product (x y : ℝ) :
  x + y = 5 → x * y = 6 → x^2 + y^2 = 13 := by sorry

end square_sum_from_sum_and_product_l503_50351


namespace prop_P_implies_t_range_prop_P_sufficient_not_necessary_implies_a_range_l503_50317

/-- Represents the curve equation -/
def curve_equation (x y t : ℝ) : Prop :=
  x^2 / (4 - t) + y^2 / (t - 1) = 1

/-- Predicate for the curve being an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (t : ℝ) : Prop :=
  ∃ x y : ℝ, curve_equation x y t

/-- The inequality involving t and a -/
def inequality (t a : ℝ) : Prop :=
  t^2 - (a + 3) * t + (a + 2) < 0

/-- Proposition P implies the range of t -/
theorem prop_P_implies_t_range :
  ∀ t : ℝ, is_ellipse_x_foci t → 1 < t ∧ t < 5/2 :=
sorry

/-- Proposition P is sufficient but not necessary for Q implies the range of a -/
theorem prop_P_sufficient_not_necessary_implies_a_range :
  (∀ t a : ℝ, is_ellipse_x_foci t → inequality t a) ∧
  (∃ t a : ℝ, inequality t a ∧ ¬is_ellipse_x_foci t) →
  ∀ a : ℝ, a > 1/2 :=
sorry

end prop_P_implies_t_range_prop_P_sufficient_not_necessary_implies_a_range_l503_50317


namespace range_of_a_l503_50376

theorem range_of_a (a : ℝ) : 
  (∀ x θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 2) → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
sorry

end range_of_a_l503_50376


namespace existence_and_uniqueness_of_t_l503_50333

def f (x : ℝ) := -x - 4

theorem existence_and_uniqueness_of_t :
  ∃! t : ℝ,
    (∀ x : ℝ, f x = -x - 4) ∧
    (f (-6) = 2 ∧ f 2 = -6) ∧
    (∀ k : ℝ, k > 0 → ∀ x : ℝ, f (x + k) < f x) ∧
    ({x : ℝ | |f (x - t) + 2| < 4} = Set.Ioo (-4 : ℝ) 4) :=
by sorry

#check existence_and_uniqueness_of_t

end existence_and_uniqueness_of_t_l503_50333


namespace scooter_depreciation_l503_50342

theorem scooter_depreciation (initial_value : ℝ) : 
  (initial_value * (3/4)^5 = 9492.1875) → initial_value = 40000 := by
  sorry

end scooter_depreciation_l503_50342


namespace evaluate_expression_l503_50324

theorem evaluate_expression : 2 - (-3) - 4 * (-5) - 6 - (-7) - 8 * (-9) + 10 = 108 := by
  sorry

end evaluate_expression_l503_50324


namespace percentage_of_a_l503_50322

theorem percentage_of_a (a b c : ℝ) (P : ℝ) : 
  (P / 100) * a = 8 →
  0.08 * b = 2 →
  c = b / a →
  P = 100 := by
sorry

end percentage_of_a_l503_50322


namespace exponent_multiplication_l503_50388

theorem exponent_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end exponent_multiplication_l503_50388


namespace hyperbola_distance_property_l503_50357

/-- A point on a hyperbola with specific distance properties -/
structure HyperbolaPoint where
  P : ℝ × ℝ
  on_hyperbola : (P.1^2 / 4) - P.2^2 = 1
  distance_to_right_focus : Real.sqrt ((P.1 - Real.sqrt 5)^2 + P.2^2) = 5

/-- The theorem stating the distance property of the hyperbola point -/
theorem hyperbola_distance_property (hp : HyperbolaPoint) :
  let d := Real.sqrt ((hp.P.1 + Real.sqrt 5)^2 + hp.P.2^2)
  d = 1 ∨ d = 9 := by
  sorry

end hyperbola_distance_property_l503_50357


namespace allowance_increase_l503_50339

/-- The base amount of Kathleen's middle school allowance -/
def base_amount : ℝ := 8

/-- Kathleen's middle school allowance -/
def middle_school_allowance (x : ℝ) : ℝ := x + 2

/-- Kathleen's senior year allowance -/
def senior_year_allowance (x : ℝ) : ℝ := 5 + 2 * (x + 2)

/-- The percentage increase in Kathleen's weekly allowance -/
def percentage_increase : ℝ := 150

theorem allowance_increase (x : ℝ) :
  x = base_amount ↔
  (1 + percentage_increase / 100) * middle_school_allowance x = senior_year_allowance x :=
sorry

end allowance_increase_l503_50339


namespace hare_wins_l503_50331

/-- Race parameters --/
def race_duration : ℕ := 60
def hare_speed : ℕ := 10
def hare_run_time : ℕ := 30
def hare_nap_time : ℕ := 30
def tortoise_delay : ℕ := 10
def tortoise_speed : ℕ := 4

/-- Calculate distance covered by the hare --/
def hare_distance : ℕ := hare_speed * hare_run_time

/-- Calculate distance covered by the tortoise --/
def tortoise_distance : ℕ := tortoise_speed * (race_duration - tortoise_delay)

/-- Theorem stating that the hare wins the race --/
theorem hare_wins : hare_distance > tortoise_distance := by
  sorry

end hare_wins_l503_50331


namespace second_smallest_odd_is_three_l503_50356

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

def second_smallest_odd : ℕ → Prop
| 3 => ∃ (x : ℕ), (is_odd x ∧ in_range x ∧ x < 3) ∧
                  ∀ (y : ℕ), (is_odd y ∧ in_range y ∧ y ≠ x ∧ y ≠ 3) → y > 3
| _ => False

theorem second_smallest_odd_is_three : second_smallest_odd 3 := by
  sorry

end second_smallest_odd_is_three_l503_50356


namespace negation_of_proposition_l503_50350

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end negation_of_proposition_l503_50350


namespace fraction_sum_inequality_l503_50335

theorem fraction_sum_inequality (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  (a / (b*c + 1)) + (b / (a*c + 1)) + (c / (a*b + 1)) ≤ 2 := by
  sorry

end fraction_sum_inequality_l503_50335


namespace max_value_implies_a_l503_50363

def f (x : ℝ) := x^2 - 2*x + 1

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 2), f x ≤ 4) ∧
  (∃ x ∈ Set.Icc a (a + 2), f x = 4) →
  a = 1 ∨ a = -1 := by
sorry

end max_value_implies_a_l503_50363


namespace squat_rack_cost_squat_rack_cost_proof_l503_50301

/-- The cost of a squat rack, given that the barbell costs 1/10 as much and the total is $2750 -/
theorem squat_rack_cost : ℝ → ℝ → Prop :=
  fun (squat_rack_cost barbell_cost : ℝ) =>
    barbell_cost = squat_rack_cost / 10 ∧
    squat_rack_cost + barbell_cost = 2750 →
    squat_rack_cost = 2500

/-- Proof of the squat rack cost theorem -/
theorem squat_rack_cost_proof : squat_rack_cost 2500 250 := by
  sorry

end squat_rack_cost_squat_rack_cost_proof_l503_50301
