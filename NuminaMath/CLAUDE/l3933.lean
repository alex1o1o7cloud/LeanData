import Mathlib

namespace NUMINAMATH_CALUDE_main_theorem_l3933_393386

/-- The set of natural numbers with an odd number of 1s in their binary representation up to 2^n - 1 -/
def A (n : ℕ) : Finset ℕ :=
  sorry

/-- The set of natural numbers with an even number of 1s in their binary representation up to 2^n - 1 -/
def B (n : ℕ) : Finset ℕ :=
  sorry

/-- The difference between the sum of nth powers of numbers in A and B -/
def S (n : ℕ) : ℤ :=
  (A n).sum (fun x => x^n) - (B n).sum (fun x => x^n)

/-- The main theorem stating the closed form of S(n) -/
theorem main_theorem (n : ℕ) : S n = (-1)^(n-1) * (n.factorial : ℤ) * 2^(n*(n-1)/2) :=
  sorry

end NUMINAMATH_CALUDE_main_theorem_l3933_393386


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l3933_393301

/-- A police emergency number is a positive integer that ends with 133 in decimal representation -/
def IsPoliceEmergencyNumber (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7 -/
theorem police_emergency_number_prime_divisor (n : ℕ) (h : IsPoliceEmergencyNumber n) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
  sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l3933_393301


namespace NUMINAMATH_CALUDE_sugar_solution_sweetness_l3933_393355

theorem sugar_solution_sweetness (a b m : ℝ) 
  (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : a / b < (a + m) / (b + m) := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_sweetness_l3933_393355


namespace NUMINAMATH_CALUDE_range_of_x_l3933_393323

theorem range_of_x (x : ℝ) : (x^2 - 2*x - 3 ≥ 0) ∧ ¬(|1 - x/2| < 1) ↔ x ≤ -1 ∨ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l3933_393323


namespace NUMINAMATH_CALUDE_line_perpendicular_implies_planes_perpendicular_l3933_393382

-- Define the structure for a plane
structure Plane :=
  (points : Set Point)

-- Define the structure for a line
structure Line :=
  (points : Set Point)

-- Define the perpendicular relation between a line and a plane
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Define the contained relation between a line and a plane
def contained (l : Line) (p : Plane) : Prop := sorry

-- Define the perpendicular relation between two planes
def perpendicularPlanes (p1 p2 : Plane) : Prop := sorry

-- Theorem statement
theorem line_perpendicular_implies_planes_perpendicular 
  (α β : Plane) (m : Line) 
  (h_distinct : α ≠ β) 
  (h_perp : perpendicular m β) 
  (h_contained : contained m α) : 
  perpendicularPlanes α β := by sorry

end NUMINAMATH_CALUDE_line_perpendicular_implies_planes_perpendicular_l3933_393382


namespace NUMINAMATH_CALUDE_tan_half_sum_l3933_393329

theorem tan_half_sum (p q : Real) 
  (h1 : Real.cos p + Real.cos q = 1/3)
  (h2 : Real.sin p + Real.sin q = 4/9) : 
  Real.tan ((p + q) / 2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_l3933_393329


namespace NUMINAMATH_CALUDE_power_two_ge_product_l3933_393359

theorem power_two_ge_product (m n : ℕ) : 2^(m+n-2) ≥ m*n := by
  sorry

end NUMINAMATH_CALUDE_power_two_ge_product_l3933_393359


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l3933_393310

-- Define the quadrilateral AMOL
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the lengths of sides
def AM (q : Quadrilateral) : ℝ := 10
def MO (q : Quadrilateral) : ℝ := 11
def OL (q : Quadrilateral) : ℝ := 12

-- Define the condition for perpendicular bisectors
def perpendicular_bisectors_condition (q : Quadrilateral) : Prop :=
  ∃ E : ℝ × ℝ, 
    E = ((q.A.1 + q.C.1) / 2, (q.A.2 + q.C.2) / 2) ∧
    (E.1 - q.A.1) * (q.B.1 - q.A.1) + (E.2 - q.A.2) * (q.B.2 - q.A.2) = 0 ∧
    (E.1 - q.C.1) * (q.D.1 - q.C.1) + (E.2 - q.C.2) * (q.D.2 - q.C.2) = 0

-- State the theorem
theorem quadrilateral_side_length (q : Quadrilateral) :
  AM q = 10 ∧ MO q = 11 ∧ OL q = 12 ∧ perpendicular_bisectors_condition q →
  Real.sqrt ((q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2) = Real.sqrt 77 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_side_length_l3933_393310


namespace NUMINAMATH_CALUDE_mini_football_betting_strategy_l3933_393312

theorem mini_football_betting_strategy :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧
    x₁ + x₂ + x₃ + x₄ = 1 ∧
    3 * x₁ ≥ 1 ∧
    4 * x₂ ≥ 1 ∧
    5 * x₃ ≥ 1 ∧
    8 * x₄ ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_mini_football_betting_strategy_l3933_393312


namespace NUMINAMATH_CALUDE_hannahs_farm_animals_l3933_393372

/-- The total number of animals on Hannah's farm -/
def total_animals (num_pigs : ℕ) : ℕ :=
  let num_cows := 2 * num_pigs - 3
  let num_goats := num_cows + 6
  num_pigs + num_cows + num_goats

/-- Theorem stating the total number of animals on Hannah's farm -/
theorem hannahs_farm_animals :
  total_animals 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_farm_animals_l3933_393372


namespace NUMINAMATH_CALUDE_meeting_percentage_is_42_percent_l3933_393381

def work_day_hours : ℕ := 10
def lunch_break_minutes : ℕ := 30
def first_meeting_minutes : ℕ := 60

def work_day_minutes : ℕ := work_day_hours * 60
def effective_work_minutes : ℕ := work_day_minutes - lunch_break_minutes
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (effective_work_minutes : ℚ) * 100

theorem meeting_percentage_is_42_percent : 
  ⌊meeting_percentage⌋ = 42 :=
sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_42_percent_l3933_393381


namespace NUMINAMATH_CALUDE_exponent_division_l3933_393320

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3933_393320


namespace NUMINAMATH_CALUDE_max_xy_value_l3933_393380

theorem max_xy_value (a b c x y : ℝ) :
  a * x + b * y + 2 * c = 0 →
  c ≠ 0 →
  a * b - c^2 ≥ 0 →
  x * y ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l3933_393380


namespace NUMINAMATH_CALUDE_cube_space_diagonal_l3933_393393

theorem cube_space_diagonal (surface_area : ℝ) (h : surface_area = 64) :
  let side_length := Real.sqrt (surface_area / 6)
  let space_diagonal := side_length * Real.sqrt 3
  space_diagonal = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_space_diagonal_l3933_393393


namespace NUMINAMATH_CALUDE_contrapositive_statement_1_contrapositive_statement_2_negation_statement_3_sufficient_condition_statement_4_l3933_393352

-- Statement 1
theorem contrapositive_statement_1 :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
sorry

-- Statement 2
theorem contrapositive_statement_2 :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

-- Statement 3
theorem negation_statement_3 :
  ¬(∃ x₀ : ℝ, x₀ > 1 ∧ x₀^2 - 2*x₀ - 3 = 0) ↔
  (∀ x : ℝ, x > 1 → x^2 - 2*x - 3 ≠ 0) :=
sorry

-- Statement 4
theorem sufficient_condition_statement_4 (a : ℝ) :
  (∀ x : ℝ, -2 < x ∧ x < -1 → (x + a)*(x + 1) < 0) →
  a > 2 :=
sorry

end NUMINAMATH_CALUDE_contrapositive_statement_1_contrapositive_statement_2_negation_statement_3_sufficient_condition_statement_4_l3933_393352


namespace NUMINAMATH_CALUDE_odd_divisor_of_4a_squared_minus_1_l3933_393398

theorem odd_divisor_of_4a_squared_minus_1 (n : ℤ) (h : Odd n) :
  ∃ a : ℤ, n ∣ (4 * a^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_divisor_of_4a_squared_minus_1_l3933_393398


namespace NUMINAMATH_CALUDE_shopping_cart_deletion_l3933_393361

theorem shopping_cart_deletion (initial_items final_items : ℕ) 
  (h1 : initial_items = 18) 
  (h2 : final_items = 8) : 
  initial_items - final_items = 10 := by
  sorry

end NUMINAMATH_CALUDE_shopping_cart_deletion_l3933_393361


namespace NUMINAMATH_CALUDE_rectangular_box_height_l3933_393363

/-- Proves that the height of a rectangular box is 2 cm, given its volume, length, and width. -/
theorem rectangular_box_height (volume : ℝ) (length width : ℝ) (h1 : volume = 144) (h2 : length = 12) (h3 : width = 6) :
  volume = length * width * 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_height_l3933_393363


namespace NUMINAMATH_CALUDE_average_math_score_l3933_393319

def june_score : ℝ := 94.5
def patty_score : ℝ := 87.5
def josh_score : ℝ := 99.75
def henry_score : ℝ := 95.5
def lucy_score : ℝ := 91
def mark_score : ℝ := 97.25

def num_children : ℕ := 6

theorem average_math_score :
  (june_score + patty_score + josh_score + henry_score + lucy_score + mark_score) / num_children = 94.25 := by
  sorry

end NUMINAMATH_CALUDE_average_math_score_l3933_393319


namespace NUMINAMATH_CALUDE_divisors_of_8n_cubed_l3933_393317

def is_product_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p * q

def count_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisors_of_8n_cubed (n : ℕ) 
  (h1 : is_product_of_two_primes n)
  (h2 : count_divisors n = 22)
  (h3 : Odd n) :
  count_divisors (8 * n^3) = 496 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_8n_cubed_l3933_393317


namespace NUMINAMATH_CALUDE_complex_simplification_l3933_393364

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  5 * (1 + i^3) / ((2 + i) * (2 - i)) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l3933_393364


namespace NUMINAMATH_CALUDE_square_roots_combination_l3933_393309

theorem square_roots_combination : ∃ (a : ℚ), a * Real.sqrt 2 = Real.sqrt 8 ∧
  (∀ (b : ℚ), b * Real.sqrt 3 ≠ Real.sqrt 6) ∧
  (∀ (c : ℚ), c * Real.sqrt 2 ≠ Real.sqrt 12) ∧
  (∀ (d : ℚ), d * Real.sqrt 12 ≠ Real.sqrt 18) := by
  sorry

end NUMINAMATH_CALUDE_square_roots_combination_l3933_393309


namespace NUMINAMATH_CALUDE_equal_perimeters_l3933_393399

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

-- Define the quadrilateral ABCD
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def D : Point := sorry

-- Define the inscribed circle and its center I
def inscribedCircle : Circle := sorry
def I : Point := inscribedCircle.center

-- Define the circumcircle ω of triangle ACI
def ω : Circle := sorry

-- Define points X, Y, Z, T on ω
def X : Point := sorry
def Y : Point := sorry
def Z : Point := sorry
def T : Point := sorry

-- Define a function to calculate the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a function to calculate the perimeter of a quadrilateral
def perimeter (p q r s : Point) : ℝ :=
  distance p q + distance q r + distance r s + distance s p

-- State the theorem
theorem equal_perimeters :
  perimeter A D T X = perimeter C D Y Z := by sorry

end NUMINAMATH_CALUDE_equal_perimeters_l3933_393399


namespace NUMINAMATH_CALUDE_reading_time_calculation_l3933_393332

def total_homework_time : ℕ := 60
def math_time : ℕ := 15
def spelling_time : ℕ := 18

theorem reading_time_calculation :
  total_homework_time - (math_time + spelling_time) = 27 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l3933_393332


namespace NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l3933_393344

theorem max_integer_a_for_real_roots : 
  ∀ a : ℤ, 
    a ≠ 1 →
    (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 3 = 0) →
    a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l3933_393344


namespace NUMINAMATH_CALUDE_hyperbola_focus_l3933_393328

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  2 * x^2 - y^2 + 8 * x - 6 * y - 8 = 0

/-- Definition of a focus for this hyperbola -/
def is_focus (x y : ℝ) : Prop :=
  ∃ (sign : ℝ), sign = 1 ∨ sign = -1 ∧
  x = -2 + sign * Real.sqrt 10.5 ∧
  y = -3

/-- Theorem stating that (-2 + √10.5, -3) is a focus of the hyperbola -/
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ is_focus x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l3933_393328


namespace NUMINAMATH_CALUDE_no_pentagon_decagon_tiling_l3933_393305

/-- The interior angle of a regular pentagon in degrees -/
def pentagon_angle : ℝ := 108

/-- The interior angle of a regular decagon in degrees -/
def decagon_angle : ℝ := 144

/-- The sum of angles at a vertex in a tiling -/
def vertex_angle_sum : ℝ := 360

/-- Theorem stating the impossibility of tiling with regular pentagons and decagons -/
theorem no_pentagon_decagon_tiling : 
  ¬ ∃ (p d : ℕ), p * pentagon_angle + d * decagon_angle = vertex_angle_sum :=
sorry

end NUMINAMATH_CALUDE_no_pentagon_decagon_tiling_l3933_393305


namespace NUMINAMATH_CALUDE_part_one_part_two_l3933_393345

-- Define the & operation
def ampersand (a b : ℚ) : ℚ := b^2 - a*b

-- Part 1
theorem part_one : ampersand (2/3) (-1/2) = 7/12 := by sorry

-- Part 2
theorem part_two (x y : ℚ) (h : |x + 1| + (y - 3)^2 = 0) : 
  ampersand x y = 12 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3933_393345


namespace NUMINAMATH_CALUDE_figure_area_proof_l3933_393373

theorem figure_area_proof (r1_height r1_width r2_height r2_width r3_height r3_width r4_height r4_width : ℕ) 
  (h1 : r1_height = 6 ∧ r1_width = 5)
  (h2 : r2_height = 3 ∧ r2_width = 5)
  (h3 : r3_height = 3 ∧ r3_width = 10)
  (h4 : r4_height = 8 ∧ r4_width = 2) :
  r1_height * r1_width + r2_height * r2_width + r3_height * r3_width + r4_height * r4_width = 91 := by
  sorry

#check figure_area_proof

end NUMINAMATH_CALUDE_figure_area_proof_l3933_393373


namespace NUMINAMATH_CALUDE_remainder_problem_l3933_393304

theorem remainder_problem (d r : ℤ) : 
  d > 1 → 
  1122 % d = r → 
  1540 % d = r → 
  2455 % d = r → 
  d - r = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3933_393304


namespace NUMINAMATH_CALUDE_computer_pricing_l3933_393307

/-- Given a computer's cost and selling prices, prove the relationship between different profit percentages. -/
theorem computer_pricing (C : ℝ) : 
  (1.5 * C = 2678.57) → (1.4 * C = 2500.00) := by sorry

end NUMINAMATH_CALUDE_computer_pricing_l3933_393307


namespace NUMINAMATH_CALUDE_walter_works_five_days_l3933_393347

/-- Calculates the number of days Walter works per week given his hourly rate, daily hours, allocation percentage, and allocated amount for school. -/
def calculate_work_days (hourly_rate : ℚ) (daily_hours : ℚ) (allocation_percentage : ℚ) (school_allocation : ℚ) : ℚ :=
  let daily_earnings := hourly_rate * daily_hours
  let weekly_earnings := school_allocation / allocation_percentage
  weekly_earnings / daily_earnings

/-- Theorem stating that Walter works 5 days a week given the specified conditions. -/
theorem walter_works_five_days 
  (hourly_rate : ℚ) 
  (daily_hours : ℚ) 
  (allocation_percentage : ℚ) 
  (school_allocation : ℚ) 
  (h1 : hourly_rate = 5)
  (h2 : daily_hours = 4)
  (h3 : allocation_percentage = 3/4)
  (h4 : school_allocation = 75) :
  calculate_work_days hourly_rate daily_hours allocation_percentage school_allocation = 5 := by
  sorry

end NUMINAMATH_CALUDE_walter_works_five_days_l3933_393347


namespace NUMINAMATH_CALUDE_num_available_sandwiches_l3933_393337

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different kinds of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different kinds of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents whether turkey is available. -/
def turkey_available : Prop := True

/-- Represents whether salami is available. -/
def salami_available : Prop := True

/-- Represents whether Swiss cheese is available. -/
def swiss_cheese_available : Prop := True

/-- Represents whether rye bread is available. -/
def rye_bread_available : Prop := True

/-- Represents the number of sandwiches with turkey/Swiss cheese combination. -/
def turkey_swiss_combinations : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread/salami combination. -/
def rye_salami_combinations : ℕ := num_cheeses

/-- Calculates the total number of possible sandwich combinations. -/
def total_combinations : ℕ := num_breads * num_meats * num_cheeses

/-- Theorem stating the number of different sandwiches a customer can order. -/
theorem num_available_sandwiches : 
  total_combinations - turkey_swiss_combinations - rye_salami_combinations = 199 := by
  sorry

end NUMINAMATH_CALUDE_num_available_sandwiches_l3933_393337


namespace NUMINAMATH_CALUDE_opposite_absolute_values_imply_y_power_x_l3933_393367

theorem opposite_absolute_values_imply_y_power_x (x y : ℝ) : 
  |2*y - 3| + |5*x - 10| = 0 → y^x = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_absolute_values_imply_y_power_x_l3933_393367


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l3933_393387

/-- Given four positive terms in an arithmetic sequence with their product equal to 256,
    the smallest possible value of the second term is 4. -/
theorem smallest_b_in_arithmetic_sequence (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →  -- all terms are positive
  ∃ (r : ℝ), a = b - r ∧ c = b + r ∧ d = b + 2*r →  -- arithmetic sequence
  a * b * c * d = 256 →  -- product is 256
  b ≥ 4 ∧ ∃ (r : ℝ), 4 - r > 0 ∧ 4 * (4 - r) * (4 + r) * (4 + 2*r) = 256 :=  -- b ≥ 4 and there exists a valid r for b = 4
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l3933_393387


namespace NUMINAMATH_CALUDE_count_valid_digits_l3933_393314

theorem count_valid_digits : 
  let is_valid (A : ℕ) := 0 ≤ A ∧ A ≤ 9 ∧ 571 * 10 + A < 5716
  (Finset.filter is_valid (Finset.range 10)).card = 6 := by
sorry

end NUMINAMATH_CALUDE_count_valid_digits_l3933_393314


namespace NUMINAMATH_CALUDE_number_ratio_problem_l3933_393384

theorem number_ratio_problem (x y z : ℝ) : 
  x + y + z = 110 →
  y = 30 →
  z = (1/3) * x →
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l3933_393384


namespace NUMINAMATH_CALUDE_ratio_proof_l3933_393368

theorem ratio_proof (a b : ℝ) (h : (a - 3*b) / (2*a - b) = 0.14285714285714285) : 
  a/b = 4 := by sorry

end NUMINAMATH_CALUDE_ratio_proof_l3933_393368


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l3933_393308

theorem floor_equation_solutions (n : ℤ) : 
  (⌊n^2 / 3⌋ : ℤ) - (⌊n / 3⌋ : ℤ)^2 = 3 ↔ n = -8 ∨ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l3933_393308


namespace NUMINAMATH_CALUDE_system_inconsistent_l3933_393394

-- Define the coefficient matrix A
def A : Matrix (Fin 4) (Fin 5) ℚ :=
  !![1, 2, -1, 3, -1;
     2, -1, 3, 1, -1;
     1, -1, 1, 2, 0;
     4, 0, 3, 6, -2]

-- Define the augmented matrix Â
def A_hat : Matrix (Fin 4) (Fin 6) ℚ :=
  !![1, 2, -1, 3, -1, 0;
     2, -1, 3, 1, -1, -1;
     1, -1, 1, 2, 0, 2;
     4, 0, 3, 6, -2, 5]

-- Theorem statement
theorem system_inconsistent :
  Matrix.rank A < Matrix.rank A_hat :=
sorry

end NUMINAMATH_CALUDE_system_inconsistent_l3933_393394


namespace NUMINAMATH_CALUDE_face_moisturizer_cost_l3933_393370

/-- Proves that the cost of each face moisturizer is $50 given the problem conditions -/
theorem face_moisturizer_cost (tanya_face_moisturizer_cost : ℝ) 
  (h1 : 2 * (2 * tanya_face_moisturizer_cost + 4 * 60) = 2 * tanya_face_moisturizer_cost + 4 * 60 + 1020) : 
  tanya_face_moisturizer_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_face_moisturizer_cost_l3933_393370


namespace NUMINAMATH_CALUDE_no_valid_triangle_difference_l3933_393343

theorem no_valid_triangle_difference (n : ℕ) : 
  ((n + 3) * (n + 4)) / 2 - (n * (n + 1)) / 2 ≠ 111 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_triangle_difference_l3933_393343


namespace NUMINAMATH_CALUDE_greeting_card_exchange_l3933_393371

theorem greeting_card_exchange (n : ℕ) (h : n * (n - 1) = 90) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_greeting_card_exchange_l3933_393371


namespace NUMINAMATH_CALUDE_parabola_roots_difference_l3933_393334

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_roots_difference (a b c : ℝ) :
  (∃ (y : ℝ → ℝ), ∀ x, y x = parabola a b c x) →
  (∃ h k, ∀ x, parabola a b c x = a * (x - h)^2 + k) →
  (parabola a b c 2 = -4) →
  (parabola a b c 4 = 12) →
  (∃ m n, m > n ∧ parabola a b c m = 0 ∧ parabola a b c n = 0) →
  (∃ m n, m > n ∧ parabola a b c m = 0 ∧ parabola a b c n = 0 ∧ m - n = 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_roots_difference_l3933_393334


namespace NUMINAMATH_CALUDE_christian_age_when_brian_is_40_l3933_393385

/-- Represents a person's age --/
structure Age where
  current : ℕ
  future : ℕ

/-- Represents the ages of Christian and Brian --/
structure AgeRelation where
  christian : Age
  brian : Age
  yearsUntilFuture : ℕ

/-- The conditions of the problem --/
def problemConditions (ages : AgeRelation) : Prop :=
  ages.christian.current = 2 * ages.brian.current ∧
  ages.brian.future = 40 ∧
  ages.christian.future = 72 ∧
  ages.christian.future = ages.christian.current + ages.yearsUntilFuture ∧
  ages.brian.future = ages.brian.current + ages.yearsUntilFuture

/-- The theorem to prove --/
theorem christian_age_when_brian_is_40 (ages : AgeRelation) :
  problemConditions ages → ages.christian.future = 72 := by
  sorry


end NUMINAMATH_CALUDE_christian_age_when_brian_is_40_l3933_393385


namespace NUMINAMATH_CALUDE_both_hit_probability_l3933_393350

/-- The probability of person A hitting the target -/
def prob_A : ℚ := 8 / 10

/-- The probability of person B hitting the target -/
def prob_B : ℚ := 7 / 10

/-- The theorem stating that the probability of both A and B hitting the target
    is equal to the product of their individual probabilities -/
theorem both_hit_probability :
  (prob_A * prob_B : ℚ) = 14 / 25 := by sorry

end NUMINAMATH_CALUDE_both_hit_probability_l3933_393350


namespace NUMINAMATH_CALUDE_product_of_divisors_36_l3933_393330

theorem product_of_divisors_36 (n : Nat) (h : n = 36) :
  (Finset.prod (Finset.filter (· ∣ n) (Finset.range (n + 1))) id) = 10077696 := by
  sorry

end NUMINAMATH_CALUDE_product_of_divisors_36_l3933_393330


namespace NUMINAMATH_CALUDE_revenue_decrease_l3933_393346

theorem revenue_decrease (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax := 0.7 * T
  let new_consumption := 1.2 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (original_revenue - new_revenue) / original_revenue = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_l3933_393346


namespace NUMINAMATH_CALUDE_ratio_equals_one_l3933_393335

theorem ratio_equals_one (a b c : ℝ) 
  (eq1 : 2*a + 13*b + 3*c = 90)
  (eq2 : 3*a + 9*b + c = 72) :
  (3*b + c) / (a + 2*b) = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_equals_one_l3933_393335


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l3933_393354

/-- Proves that given 48 pieces of junk mail and 8 houses, each house will receive 6 pieces of junk mail. -/
theorem junk_mail_distribution (total_mail : ℕ) (num_houses : ℕ) (h1 : total_mail = 48) (h2 : num_houses = 8) :
  total_mail / num_houses = 6 := by
sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l3933_393354


namespace NUMINAMATH_CALUDE_square_area_relation_l3933_393349

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I := 2*a + 3*b
  let area_I := (diagonal_I^2) / 2
  let area_II := 3 * area_I
  area_II = (3 * (2*a + 3*b)^2) / 2 := by sorry

end NUMINAMATH_CALUDE_square_area_relation_l3933_393349


namespace NUMINAMATH_CALUDE_clothes_percentage_is_25_percent_l3933_393375

def monthly_income : ℝ := 90000
def household_percentage : ℝ := 0.50
def medicine_percentage : ℝ := 0.15
def savings : ℝ := 9000

theorem clothes_percentage_is_25_percent :
  let clothes_expense := monthly_income - (household_percentage * monthly_income + medicine_percentage * monthly_income + savings)
  clothes_expense / monthly_income = 0.25 := by
sorry

end NUMINAMATH_CALUDE_clothes_percentage_is_25_percent_l3933_393375


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3933_393313

theorem unique_four_digit_number : ∃! n : ℕ, 
  1000 ≤ n ∧ n < 10000 ∧ 
  n % 131 = 112 ∧ 
  n % 132 = 98 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3933_393313


namespace NUMINAMATH_CALUDE_sets_A_B_properties_l3933_393322

theorem sets_A_B_properties (p q : ℝ) (h : p * q ≠ 0) :
  (∀ x₀ : ℝ, 9^x₀ + p * 3^x₀ + q = 0 → q * 9^(-x₀) + p * 3^(-x₀) + 1 = 0) ∧
  (∃ p q : ℝ, 
    (∃ x : ℝ, 9^x + p * 3^x + q = 0 ∧ q * 9^x + p * 3^x + 1 = 0) ∧
    (∀ x : ℝ, x ≠ 1 → 9^x + p * 3^x + q = 0 → q * 9^x + p * 3^x + 1 ≠ 0) ∧
    (9^1 + p * 3^1 + q = 0 ∧ q * 9^1 + p * 3^1 + 1 = 0) ∧
    p = -4 ∧ q = 3) :=
by sorry

end NUMINAMATH_CALUDE_sets_A_B_properties_l3933_393322


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l3933_393341

theorem triangle_side_ratio (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : b + c ≤ 2 * a) (h5 : c + a ≤ 2 * b) (h6 : a < b + c) (h7 : b < c + a) :
  2 / 3 < b / a ∧ b / a < 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l3933_393341


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l3933_393338

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem parallel_line_y_intercept
  (b : Line)
  (given_line : Line)
  (p : Point) :
  parallel b given_line →
  given_line.slope = -3 →
  given_line.intercept = 6 →
  p.x = 3 →
  p.y = -1 →
  pointOnLine p b →
  b.intercept = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l3933_393338


namespace NUMINAMATH_CALUDE_complex_power_problem_l3933_393339

theorem complex_power_problem (z : ℂ) : 
  (1 + z) / (1 - z) = Complex.I → z^2023 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l3933_393339


namespace NUMINAMATH_CALUDE_subset_of_A_l3933_393388

def A : Set ℕ := {x | x ≤ 4}

theorem subset_of_A : {3} ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_subset_of_A_l3933_393388


namespace NUMINAMATH_CALUDE_perpendicular_iff_a_eq_one_l3933_393351

-- Define the lines
def line1 (x y : ℝ) : Prop := x + y = 0
def line2 (a x y : ℝ) : Prop := x - a * y = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, line1 x1 y1 ∧ line2 a x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    (x2 - x1) * (y2 - y1) = 0

-- State the theorem
theorem perpendicular_iff_a_eq_one :
  ∀ a : ℝ, perpendicular a ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_iff_a_eq_one_l3933_393351


namespace NUMINAMATH_CALUDE_integral_2x_plus_exp_x_l3933_393316

open Real MeasureTheory Interval

theorem integral_2x_plus_exp_x : ∫ x in (-1)..(1), (2 * x + Real.exp x) = Real.exp 1 - Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_integral_2x_plus_exp_x_l3933_393316


namespace NUMINAMATH_CALUDE_polygon_sides_l3933_393306

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1440 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3933_393306


namespace NUMINAMATH_CALUDE_fraction_sum_l3933_393311

theorem fraction_sum (x y : ℚ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3933_393311


namespace NUMINAMATH_CALUDE_alexander_exhibition_problem_l3933_393302

/-- The number of pictures at each new gallery -/
def pictures_per_new_gallery (
  original_pictures : ℕ
  ) (new_galleries : ℕ
  ) (pencils_per_picture : ℕ
  ) (pencils_for_signing : ℕ
  ) (total_pencils : ℕ
  ) : ℕ :=
  let total_exhibitions := new_galleries + 1
  let pencils_for_drawing := total_pencils - (total_exhibitions * pencils_for_signing)
  let total_pictures := pencils_for_drawing / pencils_per_picture
  let new_gallery_pictures := total_pictures - original_pictures
  new_gallery_pictures / new_galleries

theorem alexander_exhibition_problem :
  pictures_per_new_gallery 9 5 4 2 88 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alexander_exhibition_problem_l3933_393302


namespace NUMINAMATH_CALUDE_lipschitz_arithmetic_is_translation_l3933_393379

/-- A function f : ℝ → ℝ satisfying the given conditions -/
def LipschitzArithmeticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|) ∧
  (∀ x : ℝ, ∃ d : ℝ, ∀ n : ℕ, (f^[n]) x = x + n • d)

/-- The main theorem -/
theorem lipschitz_arithmetic_is_translation
  (f : ℝ → ℝ) (h : LipschitzArithmeticFunction f) :
  ∃ a : ℝ, ∀ x : ℝ, f x = x + a := by
  sorry

end NUMINAMATH_CALUDE_lipschitz_arithmetic_is_translation_l3933_393379


namespace NUMINAMATH_CALUDE_original_number_proof_l3933_393321

theorem original_number_proof : ∃ x : ℤ, (x + 24) % 27 = 0 ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3933_393321


namespace NUMINAMATH_CALUDE_scientific_notation_600000_l3933_393356

theorem scientific_notation_600000 : 600000 = 6 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_600000_l3933_393356


namespace NUMINAMATH_CALUDE_grocery_payment_proof_l3933_393342

def grocery_cost (soup_cans bread_loaves cereal_boxes milk_gallons apples cookie_bags olive_oil : ℕ)
  (soup_price bread_price cereal_price milk_price apple_price cookie_price oil_price : ℕ) : ℕ :=
  soup_cans * soup_price + bread_loaves * bread_price + cereal_boxes * cereal_price +
  milk_gallons * milk_price + apples * apple_price + cookie_bags * cookie_price + olive_oil * oil_price

def min_bills_needed (total_cost bill_value : ℕ) : ℕ :=
  (total_cost + bill_value - 1) / bill_value

theorem grocery_payment_proof :
  let total_cost := grocery_cost 6 3 4 2 7 5 1 2 5 3 4 1 3 8
  min_bills_needed total_cost 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_grocery_payment_proof_l3933_393342


namespace NUMINAMATH_CALUDE_system_solution_and_equality_l3933_393358

theorem system_solution_and_equality (a b c : ℝ) (h : a * b * c ≠ 0) :
  ∃! (x y z : ℝ),
    (b * z + c * y = a ∧ c * x + a * z = b ∧ a * y + b * x = c) ∧
    (x = (b^2 + c^2 - a^2) / (2 * b * c) ∧
     y = (c^2 + a^2 - b^2) / (2 * a * c) ∧
     z = (a^2 + b^2 - c^2) / (2 * a * b)) ∧
    ((1 - x^2) / a^2 = (1 - y^2) / b^2 ∧ (1 - y^2) / b^2 = (1 - z^2) / c^2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_and_equality_l3933_393358


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3933_393331

/-- The inclination angle of the line x*sin(π/7) + y*cos(π/7) = 0 is 6π/7 -/
theorem line_inclination_angle : 
  let line_eq := fun (x y : ℝ) => x * Real.sin (π / 7) + y * Real.cos (π / 7) = 0
  ∃ (α : ℝ), α = 6 * π / 7 ∧ 
    (∀ (x y : ℝ), line_eq x y → 
      Real.tan α = - (Real.sin (π / 7) / Real.cos (π / 7))) :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3933_393331


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l3933_393383

theorem complex_fraction_calculation : 
  (((11 + 1/9 - (3 + 2/5) * (1 + 2/17)) - (8 + 2/5) / 3.6) / (2 + 6/25)) = 20/9 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l3933_393383


namespace NUMINAMATH_CALUDE_vector_at_t_4_l3933_393300

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point : ℝ → ℝ × ℝ × ℝ

/-- The given line satisfies the conditions -/
def given_line : ParameterizedLine :=
  { point := sorry }

theorem vector_at_t_4 (line : ParameterizedLine) 
  (h1 : line.point (-2) = (2, 6, 16)) 
  (h2 : line.point 1 = (-1, -5, -10)) :
  line.point 4 = (0, -4, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_t_4_l3933_393300


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3933_393390

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 6) / (x - 1) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3933_393390


namespace NUMINAMATH_CALUDE_correct_apple_count_l3933_393333

/-- Represents the types of apples Aria needs to buy. -/
structure AppleCount where
  red : ℕ
  granny : ℕ
  golden : ℕ

/-- Calculates the total number of apples Aria needs to buy for two weeks. -/
def totalApplesForTwoWeeks (normalDays weekDays : ℕ) (normalMix specialMix : AppleCount) : AppleCount :=
  { red := normalDays * normalMix.red + weekDays * specialMix.red,
    granny := normalDays * normalMix.granny + weekDays * specialMix.granny,
    golden := (normalDays + weekDays) * normalMix.golden }

/-- Theorem stating the correct number of apples Aria needs to buy for two weeks. -/
theorem correct_apple_count :
  let normalDays : ℕ := 10
  let weekDays : ℕ := 4
  let normalMix : AppleCount := { red := 1, granny := 2, golden := 1 }
  let specialMix : AppleCount := { red := 2, granny := 1, golden := 1 }
  let result := totalApplesForTwoWeeks normalDays weekDays normalMix specialMix
  result.red = 18 ∧ result.granny = 24 ∧ result.golden = 14 := by sorry

end NUMINAMATH_CALUDE_correct_apple_count_l3933_393333


namespace NUMINAMATH_CALUDE_sequence_sum_l3933_393325

theorem sequence_sum (P Q R S T U V : ℝ) : 
  S = 7 ∧ 
  P + Q + R = 21 ∧ 
  Q + R + S = 21 ∧ 
  R + S + T = 21 ∧ 
  S + T + U = 21 ∧ 
  T + U + V = 21 → 
  P + V = 14 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3933_393325


namespace NUMINAMATH_CALUDE_negative_square_two_l3933_393348

theorem negative_square_two : -2^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_two_l3933_393348


namespace NUMINAMATH_CALUDE_max_quarters_exact_solution_l3933_393326

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- Tony's total money in dollars -/
def total_money : ℚ := 490 / 100

/-- 
  Given that Tony has the same number of quarters and dimes, and his total money is $4.90,
  prove that the maximum number of quarters he can have is 14.
-/
theorem max_quarters : 
  ∀ q : ℕ, 
  (q : ℚ) * (quarter_value + dime_value) ≤ total_money → 
  q ≤ 14 :=
by sorry

/-- Prove that 14 quarters and 14 dimes exactly equal $4.90 -/
theorem exact_solution : 
  (14 : ℚ) * (quarter_value + dime_value) = total_money :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_exact_solution_l3933_393326


namespace NUMINAMATH_CALUDE_triangle_inradius_l3933_393324

/-- Given a triangle with perimeter 28 cm and area 28 cm², prove that its inradius is 2 cm -/
theorem triangle_inradius (p A r : ℝ) (h1 : p = 28) (h2 : A = 28) (h3 : A = r * p / 2) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l3933_393324


namespace NUMINAMATH_CALUDE_family_eating_habits_l3933_393376

theorem family_eating_habits (only_veg only_nonveg total_veg : ℕ) 
  (h1 : only_veg = 13)
  (h2 : only_nonveg = 8)
  (h3 : total_veg = 19) :
  total_veg - only_veg = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_family_eating_habits_l3933_393376


namespace NUMINAMATH_CALUDE_two_face_painted_count_l3933_393397

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  painted_faces : ℕ := 6

/-- Represents a painted cube cut into unit cubes -/
structure CutPaintedCube (n : ℕ) extends PaintedCube n

/-- The number of unit cubes with at least two painted faces in a cut painted cube -/
def num_two_face_painted (c : CutPaintedCube 4) : ℕ := 32

theorem two_face_painted_count (c : CutPaintedCube 4) : 
  num_two_face_painted c = 32 := by sorry

end NUMINAMATH_CALUDE_two_face_painted_count_l3933_393397


namespace NUMINAMATH_CALUDE_probability_first_odd_given_two_odd_one_even_l3933_393357

/-- Represents the outcome of drawing a ball -/
inductive BallOutcome
  | Odd
  | Even

/-- Represents the result of drawing three balls -/
structure ThreeBallDraw where
  first : BallOutcome
  second : BallOutcome
  third : BallOutcome

def is_valid_draw (draw : ThreeBallDraw) : Prop :=
  (draw.first = BallOutcome.Odd ∧ draw.second = BallOutcome.Odd ∧ draw.third = BallOutcome.Even) ∨
  (draw.first = BallOutcome.Odd ∧ draw.second = BallOutcome.Even ∧ draw.third = BallOutcome.Odd)

def probability_first_odd (total_balls : ℕ) (odd_balls : ℕ) : ℚ :=
  (odd_balls : ℚ) / (total_balls : ℚ)

theorem probability_first_odd_given_two_odd_one_even 
  (total_balls : ℕ) (odd_balls : ℕ) (h1 : total_balls = 100) (h2 : odd_balls = 50) :
  probability_first_odd total_balls odd_balls = 1/4 :=
sorry

end NUMINAMATH_CALUDE_probability_first_odd_given_two_odd_one_even_l3933_393357


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l3933_393365

/-- Circle C₁ with center (4, 0) and radius 3 -/
def C₁ : Set (ℝ × ℝ) := {p | (p.1 - 4)^2 + p.2^2 = 9}

/-- Circle C₂ with center (0, 3) and radius 2 -/
def C₂ : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 3)^2 = 4}

/-- The center of C₁ -/
def center₁ : ℝ × ℝ := (4, 0)

/-- The center of C₂ -/
def center₂ : ℝ × ℝ := (0, 3)

/-- The radius of C₁ -/
def radius₁ : ℝ := 3

/-- The radius of C₂ -/
def radius₂ : ℝ := 2

/-- Theorem: C₁ and C₂ are externally tangent -/
theorem circles_externally_tangent :
  (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 = (radius₁ + radius₂)^2 :=
by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l3933_393365


namespace NUMINAMATH_CALUDE_clarinet_cost_is_125_l3933_393392

def initial_savings : ℕ := 10
def price_per_book : ℕ := 5
def total_books_sold : ℕ := 25

def clarinet_cost : ℕ := total_books_sold * price_per_book

theorem clarinet_cost_is_125 : clarinet_cost = 125 := by
  sorry

end NUMINAMATH_CALUDE_clarinet_cost_is_125_l3933_393392


namespace NUMINAMATH_CALUDE_correct_height_order_l3933_393340

-- Define the friends
inductive Friend : Type
| David : Friend
| Emma : Friend
| Fiona : Friend

-- Define the height comparison relation
def taller_than : Friend → Friend → Prop := sorry

-- Define the conditions
axiom different_heights :
  ∀ (a b : Friend), a ≠ b → (taller_than a b ∨ taller_than b a)

axiom transitive :
  ∀ (a b c : Friend), taller_than a b → taller_than b c → taller_than a c

axiom asymmetric :
  ∀ (a b : Friend), taller_than a b → ¬taller_than b a

axiom exactly_one_true :
  (¬(taller_than Friend.Fiona Friend.Emma) ∧
   ¬(taller_than Friend.David Friend.Fiona) ∧
   taller_than Friend.David Friend.Emma) ∨
  (taller_than Friend.Fiona Friend.David ∧
   taller_than Friend.Fiona Friend.Emma) ∨
  (¬(taller_than Friend.David Friend.Emma) ∧
   ¬(taller_than Friend.David Friend.Fiona))

-- Theorem to prove
theorem correct_height_order :
  taller_than Friend.David Friend.Emma ∧
  taller_than Friend.Emma Friend.Fiona ∧
  taller_than Friend.David Friend.Fiona :=
sorry

end NUMINAMATH_CALUDE_correct_height_order_l3933_393340


namespace NUMINAMATH_CALUDE_problem_solution_l3933_393303

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 ≠ 0

def q (m : ℝ) : Prop := m > 0

-- Define the condition that either p or q is true, but not both
def condition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- Define the set of m values that satisfy the condition
def solution_set : Set ℝ := {m | condition m}

-- Theorem statement
theorem problem_solution : 
  solution_set = {m | m ∈ Set.Ioo (-2) 0 ∪ Set.Ioi 2} :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3933_393303


namespace NUMINAMATH_CALUDE_greatest_integer_no_real_roots_l3933_393369

theorem greatest_integer_no_real_roots (a : ℤ) : 
  (∀ x : ℝ, x^2 + a*x + 15 ≠ 0) → a ≤ 7 ∧ ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 15 ≠ 0) → b ≤ 7 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_no_real_roots_l3933_393369


namespace NUMINAMATH_CALUDE_real_part_of_complex_difference_times_i_l3933_393315

theorem real_part_of_complex_difference_times_i :
  let z₁ : ℂ := 4 + 29 * Complex.I
  let z₂ : ℂ := 6 + 9 * Complex.I
  (z₁ - z₂) * Complex.I |>.re = 20 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_complex_difference_times_i_l3933_393315


namespace NUMINAMATH_CALUDE_marks_siblings_count_l3933_393353

/-- The number of Mark's siblings given the egg distribution problem -/
def marks_siblings : ℕ :=
  let total_eggs : ℕ := 24  -- two dozen eggs
  let eggs_per_person : ℕ := 6
  let total_people : ℕ := total_eggs / eggs_per_person
  total_people - 1

theorem marks_siblings_count : marks_siblings = 3 := by
  sorry

end NUMINAMATH_CALUDE_marks_siblings_count_l3933_393353


namespace NUMINAMATH_CALUDE_subtraction_result_l3933_393362

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  valid : hundreds < 10 ∧ tens < 10 ∧ units < 10

/-- The result of subtracting two three-digit numbers -/
def subtract (a b : ThreeDigitNumber) : ThreeDigitNumber :=
  sorry

theorem subtraction_result 
  (a b : ThreeDigitNumber)
  (h_units : a.units = b.units + 6)
  (h_result_units : (subtract a b).units = 5)
  (h_result_tens : (subtract a b).tens = 9)
  (h_no_borrow : a.tens ≥ b.tens) :
  (subtract a b).hundreds = 4 :=
sorry

end NUMINAMATH_CALUDE_subtraction_result_l3933_393362


namespace NUMINAMATH_CALUDE_distance_between_A_and_B_l3933_393378

def point_A : Fin 3 → ℝ := ![1, 2, 3]
def point_B : Fin 3 → ℝ := ![-1, 3, -2]

theorem distance_between_A_and_B : 
  Real.sqrt (((point_B 0 - point_A 0)^2 + (point_B 1 - point_A 1)^2 + (point_B 2 - point_A 2)^2 : ℝ)) = Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_A_and_B_l3933_393378


namespace NUMINAMATH_CALUDE_rhombus_diagonals_not_necessarily_equal_l3933_393374

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The diagonals of a rhombus -/
def rhombus_diagonals (r : Rhombus) : ℝ × ℝ := sorry

/-- Theorem: The diagonals of a rhombus are not necessarily equal -/
theorem rhombus_diagonals_not_necessarily_equal :
  ∃ r : Rhombus, (rhombus_diagonals r).1 ≠ (rhombus_diagonals r).2 :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_not_necessarily_equal_l3933_393374


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l3933_393336

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem stating that the loss percentage for a radio with cost price 1500 and selling price 1110 is 26% -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 1500
  let selling_price : ℚ := 1110
  loss_percentage cost_price selling_price = 26 := by
  sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l3933_393336


namespace NUMINAMATH_CALUDE_ellipse_properties_l3933_393395

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Properties of the ellipse and related points -/
structure EllipseProperties (E : Ellipse) where
  O : Point
  A : Point
  B : Point
  M : Point
  C : Point
  N : Point
  h_O : O.x = 0 ∧ O.y = 0
  h_A : A.x = E.a ∧ A.y = 0
  h_B : B.x = 0 ∧ B.y = E.b
  h_M : M.x = 2 * E.a / 3 ∧ M.y = E.b / 3
  h_OM_slope : (M.y - O.y) / (M.x - O.x) = Real.sqrt 5 / 10
  h_C : C.x = -E.a ∧ C.y = 0
  h_N : N.x = (B.x + C.x) / 2 ∧ N.y = (B.y + C.y) / 2
  h_symmetric : ∃ (S : Point), S.y = 13 / 2 ∧
    (S.x - N.x) * (E.a / E.b + E.b / E.a) = S.y + N.y

/-- The main theorem to prove -/
theorem ellipse_properties (E : Ellipse) (props : EllipseProperties E) :
  (Real.sqrt (E.a^2 - E.b^2) / E.a = 2 * Real.sqrt 5 / 5) ∧
  (E.a^2 = 45 ∧ E.b^2 = 9) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3933_393395


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3933_393389

-- Problem 1
theorem problem_1 : 2013^2 - 2012 * 2014 = 1 := by sorry

-- Problem 2
theorem problem_2 (m n : ℤ) : ((m - n)^6 / (n - m)^4) * (m - n)^3 = (m - n)^5 := by sorry

-- Problem 3
theorem problem_3 (a b c : ℝ) : (a - 2*b + 3*c) * (a - 2*b - 3*c) = a^2 - 4*a*b + 4*b^2 - 9*c^2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3933_393389


namespace NUMINAMATH_CALUDE_square_roots_problem_l3933_393377

theorem square_roots_problem (n : ℝ) (a : ℝ) (h1 : n > 0) 
  (h2 : (a + 3) ^ 2 = n) (h3 : (2 * a + 3) ^ 2 = n) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3933_393377


namespace NUMINAMATH_CALUDE_cake_recipe_proof_l3933_393327

/-- Represents the amounts of ingredients in cups -/
structure Recipe :=
  (flour : ℚ)
  (sugar : ℚ)
  (cocoa : ℚ)
  (milk : ℚ)

def original_recipe : Recipe :=
  { flour := 3/4
  , sugar := 2/3
  , cocoa := 1/3
  , milk := 1/2 }

def doubled_recipe : Recipe :=
  { flour := 2 * original_recipe.flour
  , sugar := 2 * original_recipe.sugar
  , cocoa := 2 * original_recipe.cocoa
  , milk := 2 * original_recipe.milk }

def already_added : Recipe :=
  { flour := 1/2
  , sugar := 1/4
  , cocoa := 0
  , milk := 0 }

def additional_needed : Recipe :=
  { flour := doubled_recipe.flour - already_added.flour
  , sugar := doubled_recipe.sugar - already_added.sugar
  , cocoa := doubled_recipe.cocoa - already_added.cocoa
  , milk := doubled_recipe.milk - already_added.milk }

theorem cake_recipe_proof :
  additional_needed.flour = 1 ∧
  additional_needed.sugar = 13/12 ∧
  additional_needed.cocoa = 2/3 ∧
  additional_needed.milk = 1 :=
sorry

end NUMINAMATH_CALUDE_cake_recipe_proof_l3933_393327


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l3933_393366

theorem subset_implies_m_equals_one (m : ℝ) : 
  let A : Set ℝ := {-1, 2, 2*m-1}
  let B : Set ℝ := {2, m^2}
  B ⊆ A → m = 1 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l3933_393366


namespace NUMINAMATH_CALUDE_distance_between_points_l3933_393396

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 4)
  let p2 : ℝ × ℝ := (-6, -1)
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 106 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3933_393396


namespace NUMINAMATH_CALUDE_tom_typing_speed_l3933_393318

theorem tom_typing_speed (words_per_page : ℕ) (pages_typed : ℕ) (minutes_taken : ℕ) :
  words_per_page = 450 →
  pages_typed = 10 →
  minutes_taken = 50 →
  (words_per_page * pages_typed) / minutes_taken = 90 := by
  sorry

end NUMINAMATH_CALUDE_tom_typing_speed_l3933_393318


namespace NUMINAMATH_CALUDE_max_rectangle_area_l3933_393360

theorem max_rectangle_area (perimeter : ℕ) (min_diff : ℕ) : perimeter = 160 → min_diff = 10 → ∃ (length width : ℕ), 
  length + width = perimeter / 2 ∧ 
  length ≥ width + min_diff ∧
  ∀ (l w : ℕ), l + w = perimeter / 2 → l ≥ w + min_diff → l * w ≤ length * width ∧
  length * width = 1575 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l3933_393360


namespace NUMINAMATH_CALUDE_banana_arrangements_l3933_393391

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

theorem banana_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3933_393391
