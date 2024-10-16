import Mathlib

namespace NUMINAMATH_CALUDE_smallest_positive_integer_l548_54809

theorem smallest_positive_integer : ∀ n : ℕ, n > 0 → n ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_l548_54809


namespace NUMINAMATH_CALUDE_c_prime_coordinates_l548_54898

/-- Triangle ABC with vertices A(1,2), B(2,1), and C(3,2) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Similar triangle A'B'C' with similarity ratio 2 and origin as center of similarity -/
def similarTriangle (t : Triangle) : Triangle :=
  { A := (2 * t.A.1, 2 * t.A.2),
    B := (2 * t.B.1, 2 * t.B.2),
    C := (2 * t.C.1, 2 * t.C.2) }

/-- The original triangle ABC -/
def ABC : Triangle :=
  { A := (1, 2),
    B := (2, 1),
    C := (3, 2) }

/-- Theorem stating that C' has coordinates (6,4) or (-6,-4) -/
theorem c_prime_coordinates :
  let t' := similarTriangle ABC
  (t'.C = (6, 4) ∨ t'.C = (-6, -4)) :=
sorry

end NUMINAMATH_CALUDE_c_prime_coordinates_l548_54898


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l548_54834

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 10 = 3 →
  a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l548_54834


namespace NUMINAMATH_CALUDE_science_fiction_books_l548_54819

/-- Represents the number of books in the science fiction section of a library. -/
def num_books : ℕ := 3824 / 478

/-- Theorem stating that the number of books in the science fiction section is 8. -/
theorem science_fiction_books : num_books = 8 := by
  sorry

end NUMINAMATH_CALUDE_science_fiction_books_l548_54819


namespace NUMINAMATH_CALUDE_cornmeal_mixture_proof_l548_54860

/-- Proves that mixing 40 pounds of cornmeal with soybean meal results in a 280 lb mixture
    that is 13% protein, given that soybean meal is 14% protein and cornmeal is 7% protein. -/
theorem cornmeal_mixture_proof (total_weight : ℝ) (soybean_protein : ℝ) (cornmeal_protein : ℝ)
    (desired_protein : ℝ) (cornmeal_weight : ℝ) :
  total_weight = 280 →
  soybean_protein = 0.14 →
  cornmeal_protein = 0.07 →
  desired_protein = 0.13 →
  cornmeal_weight = 40 →
  let soybean_weight := total_weight - cornmeal_weight
  (soybean_protein * soybean_weight + cornmeal_protein * cornmeal_weight) / total_weight = desired_protein :=
by sorry

end NUMINAMATH_CALUDE_cornmeal_mixture_proof_l548_54860


namespace NUMINAMATH_CALUDE_show_revenue_l548_54841

/-- Calculates the total revenue for two shows given the attendance of the first show,
    the multiplier for the second show's attendance, and the ticket price. -/
def totalRevenue (firstShowAttendance : ℕ) (secondShowMultiplier : ℕ) (ticketPrice : ℕ) : ℕ :=
  (firstShowAttendance + secondShowMultiplier * firstShowAttendance) * ticketPrice

/-- Theorem stating that the total revenue for both shows is $20,000 -/
theorem show_revenue : totalRevenue 200 3 25 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_show_revenue_l548_54841


namespace NUMINAMATH_CALUDE_no_four_digit_reverse_diff_1008_l548_54803

theorem no_four_digit_reverse_diff_1008 : 
  ¬ ∃ (a b c d : ℕ), 
    (1000 ≤ 1000 * a + 100 * b + 10 * c + d) ∧ 
    (1000 * a + 100 * b + 10 * c + d < 10000) ∧
    ((1000 * a + 100 * b + 10 * c + d) - (1000 * d + 100 * c + 10 * b + a) = 1008) :=
by sorry

end NUMINAMATH_CALUDE_no_four_digit_reverse_diff_1008_l548_54803


namespace NUMINAMATH_CALUDE_fraction_inequality_l548_54815

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  c / (a^2) > c / (b^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l548_54815


namespace NUMINAMATH_CALUDE_equation_solution_iff_common_root_l548_54820

theorem equation_solution_iff_common_root
  (a : ℝ) (f g h : ℝ → ℝ) 
  (ha : a > 1)
  (h_sum_nonneg : ∀ x, f x + g x + h x ≥ 0) :
  (∃ x, a^(f x) + a^(g x) + a^(h x) = 3) ↔ 
  (∃ x, f x = 0 ∧ g x = 0 ∧ h x = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_iff_common_root_l548_54820


namespace NUMINAMATH_CALUDE_zoo_ticket_price_l548_54806

theorem zoo_ticket_price :
  let monday_children : ℕ := 7
  let monday_adults : ℕ := 5
  let tuesday_children : ℕ := 4
  let tuesday_adults : ℕ := 2
  let child_ticket_price : ℕ := 3
  let total_revenue : ℕ := 61
  ∃ (adult_ticket_price : ℕ),
    (monday_children * child_ticket_price + monday_adults * adult_ticket_price) +
    (tuesday_children * child_ticket_price + tuesday_adults * adult_ticket_price) = total_revenue ∧
    adult_ticket_price = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_price_l548_54806


namespace NUMINAMATH_CALUDE_A_specific_value_l548_54847

def A : ℕ → ℕ
  | 0 => 1
  | n + 1 => A (n / 2023) + A (n / 2023^2) + A (n / 2023^3)

theorem A_specific_value : A (2023^(3^2) + 20) = 653 := by
  sorry

end NUMINAMATH_CALUDE_A_specific_value_l548_54847


namespace NUMINAMATH_CALUDE_sequence_property_l548_54883

def sequence_a (n : ℕ) : ℚ :=
  3 / (15 * n - 14)

theorem sequence_property (a : ℕ → ℚ) (h1 : a 1 = 3) 
  (h2 : ∀ n : ℕ, n > 0 → 1 / a (n + 1) - 1 / a n = 5) :
  ∀ n : ℕ, n > 0 → a n = sequence_a n := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l548_54883


namespace NUMINAMATH_CALUDE_root_sum_eighth_power_l548_54811

theorem root_sum_eighth_power (r s : ℝ) : 
  r^2 - r * Real.sqrt 5 + 1 = 0 ∧ 
  s^2 - s * Real.sqrt 5 + 1 = 0 → 
  r^8 + s^8 = 47 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_eighth_power_l548_54811


namespace NUMINAMATH_CALUDE_path_area_and_cost_calculation_l548_54864

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path -/
def construction_cost (area cost_per_sqm : ℝ) : ℝ :=
  area * cost_per_sqm

theorem path_area_and_cost_calculation 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm : ℝ) 
  (h1 : field_length = 75) 
  (h2 : field_width = 55) 
  (h3 : path_width = 2.5) 
  (h4 : cost_per_sqm = 7) : 
  path_area field_length field_width path_width = 675 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_sqm = 4725 :=
by
  sorry

end NUMINAMATH_CALUDE_path_area_and_cost_calculation_l548_54864


namespace NUMINAMATH_CALUDE_rational_function_value_l548_54828

structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c

def has_asymptotes (f : RationalFunction) (x₁ x₂ : ℝ) : Prop :=
  f.q x₁ = 0 ∧ f.q x₂ = 0

def passes_through (f : RationalFunction) (x y : ℝ) : Prop :=
  f.q x ≠ 0 ∧ f.p x / f.q x = y

theorem rational_function_value (f : RationalFunction) :
  has_asymptotes f (-4) 1 →
  passes_through f 0 0 →
  passes_through f 2 (-2) →
  f.p 3 / f.q 3 = -9/7 := by
    sorry

end NUMINAMATH_CALUDE_rational_function_value_l548_54828


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l548_54872

/-- Given two vectors a and b in ℝ², prove that under certain conditions, 
    the magnitude of 2a + 3b equals √91. -/
theorem vector_magnitude_proof (a b : ℝ × ℝ) : 
  a = (4, -3) → 
  ‖b‖ = 3 → 
  (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) = -1/2 → 
  ‖2 • a + 3 • b‖ = Real.sqrt 91 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l548_54872


namespace NUMINAMATH_CALUDE_symmetric_solution_l548_54816

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  2 * x^2 + 3 * x * y + y^2 = 70 ∧ 6 * x^2 + x * y - y^2 = 50

/-- Given solution -/
def x₁ : ℝ := 3
def y₁ : ℝ := 4

/-- Theorem stating that if (x₁, y₁) is a solution, then (-x₁, -y₁) is also a solution -/
theorem symmetric_solution :
  system x₁ y₁ → system (-x₁) (-y₁) := by sorry

end NUMINAMATH_CALUDE_symmetric_solution_l548_54816


namespace NUMINAMATH_CALUDE_inequality_solution_set_l548_54897

theorem inequality_solution_set :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l548_54897


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l548_54880

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a * r^5 / (1 - r)) / (a / (1 - r)) = 1 / 81 → r = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l548_54880


namespace NUMINAMATH_CALUDE_factor_x10_minus_1024_l548_54885

theorem factor_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x-2)*(x+2)*(x^4 + 2*x^3 + 4*x^2 + 8*x + 16)*(x^4 - 2*x^3 + 4*x^2 - 8*x + 16) := by
  sorry

end NUMINAMATH_CALUDE_factor_x10_minus_1024_l548_54885


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l548_54827

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  (∀ y, -49 ≤ y ∧ y ≤ 49 → Real.sqrt (49 + y) + Real.sqrt (49 - y) ≤ Real.sqrt (49 + x) + Real.sqrt (49 - x)) →
  Real.sqrt (49 + x) + Real.sqrt (49 - x) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l548_54827


namespace NUMINAMATH_CALUDE_first_month_sale_is_2500_l548_54861

/-- Calculates the sale in the first month given the sales in other months and the average -/
def first_month_sale (second_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (average : ℕ) : ℕ :=
  4 * average - (second_month + third_month + fourth_month)

/-- Proves that the sale in the first month is 2500 given the conditions -/
theorem first_month_sale_is_2500 :
  first_month_sale 4000 3540 1520 2890 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_is_2500_l548_54861


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l548_54889

def a : ℕ := 3547
def b : ℕ := 12739
def c : ℕ := 21329
def r : ℕ := 17

theorem greatest_divisor_with_remainder (d : ℕ) : d > 0 → d.gcd (a - r) = d → d.gcd (b - r) = d → d.gcd (c - r) = d → 
  (∀ k : ℕ, k > d → (k.gcd (a - r) ≠ k ∨ k.gcd (b - r) ≠ k ∨ k.gcd (c - r) ≠ k)) → 
  (∀ n : ℕ, n > 0 → (a % n = r ∧ b % n = r ∧ c % n = r) → n ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l548_54889


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l548_54865

/-- Represents an ellipse with semi-major axis 'a' and semi-minor axis 2 -/
structure Ellipse where
  a : ℝ
  h_a : a > 2

/-- Represents a line with equation y = x - 2 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 2}

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- A focus of the ellipse -/
def focus (e : Ellipse) : ℝ × ℝ :=
  sorry

theorem ellipse_eccentricity (e : Ellipse) :
  focus e ∈ Line →
  eccentricity e = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l548_54865


namespace NUMINAMATH_CALUDE_negation_equivalence_l548_54867

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l548_54867


namespace NUMINAMATH_CALUDE_problem_statement_l548_54891

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.exp x = 0.1

-- Define the perpendicularity condition for two lines
def perpendicular (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x - a * y = 0) → (2 * x + a * y - 1 = 0) → 
  (1 / a) * (-2 / a) = -1

-- Define proposition q
def q : Prop := ∀ a : ℝ, perpendicular a ↔ a = Real.sqrt 2

-- The theorem to be proved
theorem problem_statement : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l548_54891


namespace NUMINAMATH_CALUDE_car_b_speed_l548_54845

/-- Proves that given the initial conditions and final state, Car B's speed is 50 mph -/
theorem car_b_speed (initial_distance : ℝ) (car_a_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 30 →
  car_a_speed = 58 →
  time = 4.75 →
  final_distance = 8 →
  (car_a_speed * time - initial_distance - final_distance) / time = 50 := by
sorry

end NUMINAMATH_CALUDE_car_b_speed_l548_54845


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l548_54839

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (5217 * 10 + N) % 6 = 0 ∧
  ∀ (M : ℕ), M ≤ 9 → (5217 * 10 + M) % 6 = 0 → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l548_54839


namespace NUMINAMATH_CALUDE_zebra_catches_tiger_l548_54807

/-- The time it takes for a zebra to catch a tiger given their speeds and the tiger's head start -/
theorem zebra_catches_tiger (zebra_speed tiger_speed : ℝ) (head_start : ℝ) : 
  zebra_speed = 55 →
  tiger_speed = 30 →
  head_start = 5 →
  (head_start * tiger_speed) / (zebra_speed - tiger_speed) = 6 := by
  sorry

end NUMINAMATH_CALUDE_zebra_catches_tiger_l548_54807


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l548_54892

theorem system_of_inequalities_solution (x : ℝ) :
  (x^2 > x + 2 ∧ 4*x^2 ≤ 4*x + 15) ↔ 
  (x ∈ Set.Icc (-3/2) (-1) ∪ Set.Ioc 2 (5/2)) :=
sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l548_54892


namespace NUMINAMATH_CALUDE_sid_computer_accessories_cost_l548_54873

/-- Calculates the amount spent on computer accessories given the initial amount,
    snack cost, and remaining amount after purchases. -/
def computer_accessories_cost (initial_amount : ℕ) (snack_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  initial_amount - snack_cost - remaining_amount

/-- Proves that Sid spent $12 on computer accessories given the problem conditions. -/
theorem sid_computer_accessories_cost :
  let initial_amount : ℕ := 48
  let snack_cost : ℕ := 8
  let remaining_amount : ℕ := (initial_amount / 2) + 4
  computer_accessories_cost initial_amount snack_cost remaining_amount = 12 := by
  sorry

#eval computer_accessories_cost 48 8 28

end NUMINAMATH_CALUDE_sid_computer_accessories_cost_l548_54873


namespace NUMINAMATH_CALUDE_rectangles_in_4x5_grid_l548_54843

/-- The number of rectangles in a horizontal strip of width n -/
def horizontalRectangles (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of rectangles in a vertical strip of height m -/
def verticalRectangles (m : ℕ) : ℕ := m * (m + 1) / 2

/-- The total number of rectangles in an m×n grid -/
def totalRectangles (m n : ℕ) : ℕ :=
  m * horizontalRectangles n + n * verticalRectangles m - m * n

theorem rectangles_in_4x5_grid :
  totalRectangles 4 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_4x5_grid_l548_54843


namespace NUMINAMATH_CALUDE_battleship_detectors_l548_54850

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a ship with width and height -/
structure Ship :=
  (width : ℕ)
  (height : ℕ)

/-- Function to calculate the minimum number of detectors required -/
def min_detectors (g : Grid) (s : Ship) : ℕ :=
  ((g.width - 1) / 3) * 2 + ((g.width - 1) % 3) + 1

/-- Theorem stating the minimum number of detectors for the Battleship problem -/
theorem battleship_detectors :
  let grid : Grid := ⟨203, 1⟩
  let ship : Ship := ⟨2, 1⟩
  min_detectors grid ship = 134 := by
  sorry

#check battleship_detectors

end NUMINAMATH_CALUDE_battleship_detectors_l548_54850


namespace NUMINAMATH_CALUDE_greatest_root_of_f_l548_54836

def f (x : ℝ) : ℝ := 16 * x^6 - 15 * x^4 + 4 * x^2 - 1

theorem greatest_root_of_f :
  ∃ (r : ℝ), f r = 0 ∧ r = 1 ∧ ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_f_l548_54836


namespace NUMINAMATH_CALUDE_largest_equal_sum_digits_l548_54802

/-- The sum of decimal digits of a natural number -/
def sumDecimalDigits (n : ℕ) : ℕ := sorry

/-- The sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 503 is the largest number less than 1000 
    with equal sum of decimal and binary digits -/
theorem largest_equal_sum_digits : 
  ∀ n : ℕ, n < 1000 → n > 503 → 
    sumDecimalDigits n ≠ sumBinaryDigits n :=
by sorry

end NUMINAMATH_CALUDE_largest_equal_sum_digits_l548_54802


namespace NUMINAMATH_CALUDE_minutes_to_date_time_correct_l548_54814

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Converts minutes to a DateTime structure -/
def minutesToDateTime (startDateTime : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting date and time -/
def startDateTime : DateTime :=
  { year := 2015, month := 1, day := 1, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 3050

/-- The expected result date and time -/
def expectedDateTime : DateTime :=
  { year := 2015, month := 1, day := 3, hour := 2, minute := 50 }

theorem minutes_to_date_time_correct :
  minutesToDateTime startDateTime minutesToAdd = expectedDateTime :=
  sorry

end NUMINAMATH_CALUDE_minutes_to_date_time_correct_l548_54814


namespace NUMINAMATH_CALUDE_no_real_solution_for_inequality_l548_54840

theorem no_real_solution_for_inequality :
  ¬∃ (x : ℝ), 3 * x^2 + 9 * x ≤ -12 := by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_inequality_l548_54840


namespace NUMINAMATH_CALUDE_power_product_rule_l548_54875

theorem power_product_rule (a : ℝ) : a^3 * a^5 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l548_54875


namespace NUMINAMATH_CALUDE_sin_30_plus_cos_60_quadratic_equation_solutions_l548_54858

-- Problem 1
theorem sin_30_plus_cos_60 : Real.sin (π / 6) + Real.cos (π / 3) = 1 := by sorry

-- Problem 2
theorem quadratic_equation_solutions (x : ℝ) : 
  x^2 - 4*x = 12 ↔ x = 6 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_sin_30_plus_cos_60_quadratic_equation_solutions_l548_54858


namespace NUMINAMATH_CALUDE_muffin_cost_l548_54882

theorem muffin_cost (paid : ℕ) (change : ℕ) (num_muffins : ℕ) : 
  paid = 20 ∧ change = 11 ∧ num_muffins = 12 → 
  (paid - change) * 100 / num_muffins = 75 := by
  sorry

end NUMINAMATH_CALUDE_muffin_cost_l548_54882


namespace NUMINAMATH_CALUDE_power_fraction_evaluation_l548_54812

theorem power_fraction_evaluation :
  ((2^1010)^2 - (2^1008)^2) / ((2^1009)^2 - (2^1007)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_evaluation_l548_54812


namespace NUMINAMATH_CALUDE_popsicle_stick_sum_l548_54894

theorem popsicle_stick_sum : 
  ∀ (gino ana sam speaker : ℕ),
    gino = 63 →
    ana = 128 →
    sam = 75 →
    speaker = 50 →
    gino + ana + sam + speaker = 316 :=
by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_sum_l548_54894


namespace NUMINAMATH_CALUDE_mono_increasing_implies_g_neg_one_lt_g_one_l548_54871

-- Define a monotonically increasing function
def MonotonicallyIncreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → g x < g y

-- Theorem statement
theorem mono_increasing_implies_g_neg_one_lt_g_one
    (g : ℝ → ℝ) (h : MonotonicallyIncreasing g) :
    g (-1) < g 1 := by
  sorry

end NUMINAMATH_CALUDE_mono_increasing_implies_g_neg_one_lt_g_one_l548_54871


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l548_54849

-- Equation 1: x^2 + 4x - 1 = 0
theorem equation_one_solutions (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ x = Real.sqrt 5 - 2 ∨ x = -Real.sqrt 5 - 2 := by sorry

-- Equation 2: (x-1)^2 = 3(x-1)
theorem equation_two_solutions (x : ℝ) :
  (x - 1)^2 = 3*(x - 1) ↔ x = 1 ∨ x = 4 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l548_54849


namespace NUMINAMATH_CALUDE_fraction_simplification_l548_54856

theorem fraction_simplification : 
  (3 - 6 + 12 - 24 + 48 - 96) / (6 - 12 + 24 - 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l548_54856


namespace NUMINAMATH_CALUDE_sin_alpha_plus_beta_equals_one_l548_54805

theorem sin_alpha_plus_beta_equals_one 
  (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1) 
  (h2 : Real.cos α + Real.sin β = Real.sqrt 3) : 
  Real.sin (α + β) = 1 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_beta_equals_one_l548_54805


namespace NUMINAMATH_CALUDE_order_exponentials_l548_54825

theorem order_exponentials : 4^9 < 6^7 ∧ 6^7 < 3^13 := by
  sorry

end NUMINAMATH_CALUDE_order_exponentials_l548_54825


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l548_54890

/-- Given a geometric sequence where S_n = 48 and S_2n = 60, prove that S_3n = 63 -/
theorem geometric_sequence_sum (S : ℕ → ℝ) (n : ℕ) 
  (h1 : S n = 48) 
  (h2 : S (2 * n) = 60) 
  (h_geometric : ∀ k : ℕ, S (k + 1) / S k = S (k + 2) / S (k + 1)) :
  S (3 * n) = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l548_54890


namespace NUMINAMATH_CALUDE_fifteenth_thirtyseventh_415th_digit_l548_54869

/-- The decimal representation of 15/37 has a repeating sequence of '405'. -/
def decimal_rep : ℚ → List ℕ := sorry

/-- The nth digit after the decimal point in the decimal representation of a rational number. -/
def nth_digit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- The 415th digit after the decimal point in the decimal representation of 15/37 is 4. -/
theorem fifteenth_thirtyseventh_415th_digit :
  nth_digit (15 / 37) 415 = 4 := by sorry

end NUMINAMATH_CALUDE_fifteenth_thirtyseventh_415th_digit_l548_54869


namespace NUMINAMATH_CALUDE_plan_a_is_lowest_l548_54899

/-- Represents a payment plan with monthly payment, duration, and interest rate -/
structure PaymentPlan where
  monthly_payment : ℝ
  duration : ℕ
  interest_rate : ℝ

/-- Calculates the total repayment amount for a given payment plan -/
def total_repayment (plan : PaymentPlan) : ℝ :=
  let principal := plan.monthly_payment * plan.duration
  principal + principal * plan.interest_rate

/-- The three payment plans available to Aaron -/
def plan_a : PaymentPlan := ⟨100, 12, 0.1⟩
def plan_b : PaymentPlan := ⟨90, 15, 0.08⟩
def plan_c : PaymentPlan := ⟨80, 18, 0.06⟩

/-- Theorem stating that Plan A has the lowest total repayment amount -/
theorem plan_a_is_lowest :
  total_repayment plan_a < total_repayment plan_b ∧
  total_repayment plan_a < total_repayment plan_c :=
by sorry

end NUMINAMATH_CALUDE_plan_a_is_lowest_l548_54899


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_11_mod_14_l548_54817

theorem least_five_digit_congruent_to_11_mod_14 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  n % 14 = 11 ∧               -- congruent to 11 (mod 14)
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 14 = 11 → m ≥ n) ∧  -- least such number
  n = 10007 :=                -- the answer is 10007
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_11_mod_14_l548_54817


namespace NUMINAMATH_CALUDE_quadratic_monotone_increasing_l548_54844

/-- A quadratic function f(x) = x^2 + 2ax + b is monotonically increasing
    on the interval [-1, +∞) if and only if a ≥ 1 -/
theorem quadratic_monotone_increasing (a b : ℝ) :
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ → x₁^2 + 2*a*x₁ + b < x₂^2 + 2*a*x₂ + b) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotone_increasing_l548_54844


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l548_54863

theorem line_tangent_to_circle (b : ℝ) : 
  (∀ x y : ℝ, x - y + b = 0 → (x^2 + y^2 = 25 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → 
      ((x' - y' + b ≠ 0) ∨ (x'^2 + y'^2 ≠ 25)))) → 
  b = 5 * Real.sqrt 2 ∨ b = -5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l548_54863


namespace NUMINAMATH_CALUDE_positive_expression_l548_54823

theorem positive_expression (x : ℝ) (h : x > 0) : x^2 + π*x + (15*π/2)*Real.sin x > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l548_54823


namespace NUMINAMATH_CALUDE_equality_multiplication_l548_54854

theorem equality_multiplication (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end NUMINAMATH_CALUDE_equality_multiplication_l548_54854


namespace NUMINAMATH_CALUDE_class_size_is_37_l548_54833

/-- Represents the number of students in a class with specific age distribution. -/
def number_of_students (common_age : ℕ) (total_age_sum : ℕ) : ℕ :=
  (total_age_sum + 3) / common_age

/-- Theorem stating the number of students in the class is 37. -/
theorem class_size_is_37 :
  ∃ (common_age : ℕ),
    common_age > 0 ∧
    number_of_students common_age 330 = 37 ∧
    330 = 7 * (common_age - 1) + 2 * (common_age + 2) + (37 - 9) * common_age :=
sorry

end NUMINAMATH_CALUDE_class_size_is_37_l548_54833


namespace NUMINAMATH_CALUDE_three_power_greater_than_n_plus_two_times_two_power_l548_54826

theorem three_power_greater_than_n_plus_two_times_two_power (n : ℕ) (h : n > 2) :
  3^n > (n + 2) * 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_three_power_greater_than_n_plus_two_times_two_power_l548_54826


namespace NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_interval_l548_54884

/-- A cubic equation x^3 + px + q = 0 has three distinct roots in (-2, 4) if and only if
    its coefficients p and q satisfy the given conditions. -/
theorem cubic_three_distinct_roots_in_interval
  (p q : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    -2 < x₁ ∧ x₁ < 4 ∧ -2 < x₂ ∧ x₂ < 4 ∧ -2 < x₃ ∧ x₃ < 4 ∧
    x₁^3 + p*x₁ + q = 0 ∧ x₂^3 + p*x₂ + q = 0 ∧ x₃^3 + p*x₃ + q = 0) ↔
  (4*p^3 + 27*q^2 < 0 ∧ -4*p - 64 < q ∧ q < 2*p + 8) :=
sorry

end NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_interval_l548_54884


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l548_54808

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + k^2 - 3 > 0) ↔ (k > 2 ∨ k < -2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l548_54808


namespace NUMINAMATH_CALUDE_fewer_heads_probability_l548_54822

/-- The number of coins being flipped -/
def n : ℕ := 8

/-- The probability of getting the same number of heads and tails -/
def p_equal : ℚ := (n.choose (n / 2)) / 2^n

/-- The probability of getting fewer heads than tails -/
def p_fewer_heads : ℚ := (1 - p_equal) / 2

theorem fewer_heads_probability :
  p_fewer_heads = 93 / 256 := by sorry

end NUMINAMATH_CALUDE_fewer_heads_probability_l548_54822


namespace NUMINAMATH_CALUDE_hose_flow_rate_l548_54878

/-- Given a pool that takes 50 hours to fill, water costs 1 cent for 10 gallons,
    and it costs 5 dollars to fill the pool, the hose runs at a rate of 100 gallons per hour. -/
theorem hose_flow_rate (fill_time : ℕ) (water_cost : ℚ) (fill_cost : ℕ) :
  fill_time = 50 →
  water_cost = 1 / 10 →
  fill_cost = 5 →
  (fill_cost * 100 : ℚ) / (water_cost * fill_time) = 100 := by
  sorry

end NUMINAMATH_CALUDE_hose_flow_rate_l548_54878


namespace NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l548_54832

def f (x : ℝ) : ℝ := -x^2 + 3

theorem quadratic_function_satisfies_conditions :
  (∀ x y : ℝ, x < y → f x > f y) ∧ 
  f 0 = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l548_54832


namespace NUMINAMATH_CALUDE_proposition_q_undetermined_l548_54893

theorem proposition_q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) ∧ ¬(q ∧ ¬q) := by
sorry

end NUMINAMATH_CALUDE_proposition_q_undetermined_l548_54893


namespace NUMINAMATH_CALUDE_cadence_total_earnings_l548_54870

/-- Calculates the total earnings of Cadence from two companies given the specified conditions. -/
theorem cadence_total_earnings :
  let old_company_years : ℚ := 3.5
  let old_company_monthly_salary : ℚ := 5000
  let old_company_bonus_rate : ℚ := 0.5
  let new_company_years : ℕ := 4
  let new_company_salary_raise : ℚ := 0.2
  let new_company_bonus_rate : ℚ := 1
  let third_year_deduction_rate : ℚ := 0.02

  let old_company_salary := old_company_years * 12 * old_company_monthly_salary
  let old_company_bonus := (old_company_years.floor * old_company_bonus_rate * old_company_monthly_salary) +
                           (old_company_years - old_company_years.floor) * old_company_bonus_rate * old_company_monthly_salary
  let new_company_monthly_salary := old_company_monthly_salary * (1 + new_company_salary_raise)
  let new_company_salary := new_company_years * 12 * new_company_monthly_salary
  let new_company_bonus := new_company_years * new_company_bonus_rate * new_company_monthly_salary
  let third_year_deduction := third_year_deduction_rate * 12 * new_company_monthly_salary

  let total_earnings := old_company_salary + old_company_bonus + new_company_salary + new_company_bonus - third_year_deduction

  total_earnings = 529310 := by
    sorry

end NUMINAMATH_CALUDE_cadence_total_earnings_l548_54870


namespace NUMINAMATH_CALUDE_interest_rate_difference_l548_54895

/-- Proves that for a sum of $700 at simple interest for 4 years, 
    if a higher rate fetches $56 more interest, 
    then the difference between the higher rate and the original rate is 2 percentage points. -/
theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (original_rate : ℝ) 
  (higher_rate : ℝ) 
  (h1 : principal = 700) 
  (h2 : time = 4) 
  (h3 : higher_rate * principal * time / 100 = original_rate * principal * time / 100 + 56) : 
  higher_rate - original_rate = 2 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l548_54895


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l548_54842

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l548_54842


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_plus_one_l548_54877

theorem smallest_three_digit_multiple_plus_one : ∃ (n : ℕ), 
  (n = 421) ∧ 
  (100 ≤ n) ∧ 
  (n < 1000) ∧ 
  (∃ (k : ℕ), n = k * 3 + 1) ∧
  (∃ (k : ℕ), n = k * 4 + 1) ∧
  (∃ (k : ℕ), n = k * 5 + 1) ∧
  (∃ (k : ℕ), n = k * 6 + 1) ∧
  (∃ (k : ℕ), n = k * 7 + 1) ∧
  (∀ (m : ℕ), 
    (100 ≤ m) ∧ 
    (m < n) → 
    ¬((∃ (k : ℕ), m = k * 3 + 1) ∧
      (∃ (k : ℕ), m = k * 4 + 1) ∧
      (∃ (k : ℕ), m = k * 5 + 1) ∧
      (∃ (k : ℕ), m = k * 6 + 1) ∧
      (∃ (k : ℕ), m = k * 7 + 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_plus_one_l548_54877


namespace NUMINAMATH_CALUDE_inequality_proof_l548_54857

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ineq : 1/a + 1/b + 1/c ≥ a + b + c) :
  a + b + c ≥ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l548_54857


namespace NUMINAMATH_CALUDE_intersecting_lines_angles_l548_54821

-- Define a structure for a line
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a structure for an angle
structure Angle :=
  (measure : ℝ)

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function for alternate interior angles
def alternate_interior_angles (l1 l2 : Line) (t : Line) : Angle × Angle :=
  sorry

-- Define a function for corresponding angles
def corresponding_angles (l1 l2 : Line) (t : Line) : Angle × Angle :=
  sorry

-- Define a function for consecutive interior angles
def consecutive_interior_angles (l1 l2 : Line) (t : Line) : Angle × Angle :=
  sorry

-- Main theorem
theorem intersecting_lines_angles (l1 l2 t : Line) 
  (h : ¬ are_parallel l1 l2) : 
  ∃ (a1 a2 : Angle), 
    (alternate_interior_angles l1 l2 t = (a1, a2) ∧ a1.measure ≠ a2.measure) ∨
    (corresponding_angles l1 l2 t = (a1, a2) ∧ a1.measure ≠ a2.measure) ∨
    (consecutive_interior_angles l1 l2 t = (a1, a2) ∧ a1.measure + a2.measure ≠ 180) :=
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_angles_l548_54821


namespace NUMINAMATH_CALUDE_garden_tilling_time_l548_54837

/-- Calculates the time required to till a rectangular plot -/
def tillingTime (width : ℕ) (length : ℕ) (swathWidth : ℕ) (tillRate : ℚ) : ℚ :=
  let rows := width / swathWidth
  let totalDistance := rows * length
  let totalSeconds := totalDistance * tillRate
  totalSeconds / 60

theorem garden_tilling_time :
  tillingTime 110 120 2 2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_garden_tilling_time_l548_54837


namespace NUMINAMATH_CALUDE_third_term_expansion_l548_54813

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of the third term in (3b+2a)^6
def third_term_coefficient : ℕ := binomial 6 2 * 3^4 * 2^2

-- Theorem statement
theorem third_term_expansion :
  third_term_coefficient = 4860 ∧ binomial 6 2 = 15 := by sorry

end NUMINAMATH_CALUDE_third_term_expansion_l548_54813


namespace NUMINAMATH_CALUDE_abcdef_hex_bits_l548_54896

def hex_to_decimal (hex : String) : ℕ :=
  -- Convert hexadecimal string to decimal
  sorry

def bits_required (n : ℕ) : ℕ :=
  -- Calculate the number of bits required to represent n
  sorry

theorem abcdef_hex_bits :
  bits_required (hex_to_decimal "ABCDEF") = 24 := by
  sorry

end NUMINAMATH_CALUDE_abcdef_hex_bits_l548_54896


namespace NUMINAMATH_CALUDE_solutions_equation1_solution_equation2_l548_54859

-- Define the equations
def equation1 (x : ℝ) : Prop := (x - 2)^2 = 36
def equation2 (x : ℝ) : Prop := (2*x - 1)^3 = -125

-- Statement for the first equation
theorem solutions_equation1 : 
  (∃ x : ℝ, equation1 x) ∧ 
  (∀ x : ℝ, equation1 x ↔ (x = 8 ∨ x = -4)) :=
sorry

-- Statement for the second equation
theorem solution_equation2 : 
  (∃ x : ℝ, equation2 x) ∧
  (∀ x : ℝ, equation2 x ↔ x = -2) :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solution_equation2_l548_54859


namespace NUMINAMATH_CALUDE_share_price_increase_l548_54818

theorem share_price_increase (initial_price : ℝ) (q1_increase : ℝ) (q2_increase : ℝ) :
  q1_increase = 0.25 →
  q2_increase = 0.44 →
  ((initial_price * (1 + q1_increase) * (1 + q2_increase) - initial_price) / initial_price) = 0.80 :=
by sorry

end NUMINAMATH_CALUDE_share_price_increase_l548_54818


namespace NUMINAMATH_CALUDE_cover_ways_2x13_l548_54866

/-- The number of ways to cover a 2 × n rectangular board with 1 × 2 tiles -/
def cover_ways : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| (n + 3) => cover_ways (n + 2) + cover_ways (n + 1)

/-- Tiles of size 1 × 2 -/
structure Tile :=
  (width : ℕ := 1)
  (height : ℕ := 2)

/-- A 2 × 13 rectangular board -/
structure Board :=
  (width : ℕ := 2)
  (height : ℕ := 13)

/-- Theorem: The number of ways to cover a 2 × 13 board with 1 × 2 tiles is 377 -/
theorem cover_ways_2x13 : cover_ways 13 = 377 := by
  sorry

end NUMINAMATH_CALUDE_cover_ways_2x13_l548_54866


namespace NUMINAMATH_CALUDE_current_tariff_calculation_specific_case_calculation_l548_54835

/-- Calculates the current actual tariff after two successive reductions -/
def current_tariff (S : ℝ) : ℝ := (1 - 0.4) * (1 - 0.3) * S

/-- Theorem stating the current actual tariff calculation -/
theorem current_tariff_calculation (S : ℝ) : 
  current_tariff S = (1 - 0.4) * (1 - 0.3) * S := by sorry

/-- Theorem for the specific case when S = 1000 -/
theorem specific_case_calculation : 
  current_tariff 1000 = 420 := by sorry

end NUMINAMATH_CALUDE_current_tariff_calculation_specific_case_calculation_l548_54835


namespace NUMINAMATH_CALUDE_alyssa_total_games_l548_54852

/-- The total number of soccer games Alyssa will attend over three years -/
def total_games (this_year last_year next_year : ℕ) : ℕ :=
  this_year + last_year + next_year

/-- Proof that Alyssa will attend 39 soccer games in total -/
theorem alyssa_total_games :
  total_games 11 13 15 = 39 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_total_games_l548_54852


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l548_54810

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l548_54810


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l548_54830

theorem quadratic_inequality_solution (a b : ℝ) 
  (h1 : (1 : ℝ) / 3 * 1 = -1 / a) 
  (h2 : (1 : ℝ) / 3 + 1 = -b / a) 
  (h3 : a < 0) : 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l548_54830


namespace NUMINAMATH_CALUDE_total_rackets_packed_l548_54888

/-- Proves that given the conditions of racket packaging, the total number of rackets is 100 -/
theorem total_rackets_packed (total_cartons : ℕ) (three_racket_cartons : ℕ) 
  (h1 : total_cartons = 38)
  (h2 : three_racket_cartons = 24) :
  3 * three_racket_cartons + 2 * (total_cartons - three_racket_cartons) = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_rackets_packed_l548_54888


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l548_54876

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x + 1

theorem derivative_f_at_zero :
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l548_54876


namespace NUMINAMATH_CALUDE_new_student_weight_l548_54804

/-- Given a group of students and their weights, calculates the weight of a new student
    that changes the average weight of the group. -/
theorem new_student_weight
  (n : ℕ) -- number of students before new admission
  (w : ℝ) -- average weight before new admission
  (new_w : ℝ) -- new average weight after admission
  (h1 : n = 29) -- there are 29 students initially
  (h2 : w = 28) -- the initial average weight is 28 kg
  (h3 : new_w = 27.4) -- the new average weight is 27.4 kg
  : (n + 1) * new_w - n * w = 10 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l548_54804


namespace NUMINAMATH_CALUDE_train_length_l548_54874

/-- Proves that a train traveling at 45 km/hr crossing a 255 m bridge in 30 seconds has a length of 120 m -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_speed * 1000 / 3600) * crossing_time - bridge_length = 120 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l548_54874


namespace NUMINAMATH_CALUDE_simplify_sqrt_one_minus_sin_two_alpha_l548_54848

theorem simplify_sqrt_one_minus_sin_two_alpha (α : Real) 
  (h : π / 4 < α ∧ α < π / 2) : 
  Real.sqrt (1 - Real.sin (2 * α)) = Real.sin α - Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_one_minus_sin_two_alpha_l548_54848


namespace NUMINAMATH_CALUDE_smallest_number_divisible_plus_one_l548_54829

theorem smallest_number_divisible_plus_one (n : ℕ) : n = 1038239 ↔ 
  (∀ m : ℕ, m < n → ¬((m + 1) % 618 = 0 ∧ (m + 1) % 3648 = 0 ∧ (m + 1) % 60 = 0)) ∧
  ((n + 1) % 618 = 0 ∧ (n + 1) % 3648 = 0 ∧ (n + 1) % 60 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_plus_one_l548_54829


namespace NUMINAMATH_CALUDE_klinker_daughter_age_l548_54824

/-- Proves that given Mr. Klinker is 35 years old and in 15 years he will be twice as old as his daughter, his daughter's current age is 10 years. -/
theorem klinker_daughter_age (klinker_age : ℕ) (daughter_age : ℕ) : 
  klinker_age = 35 →
  klinker_age + 15 = 2 * (daughter_age + 15) →
  daughter_age = 10 := by
sorry

end NUMINAMATH_CALUDE_klinker_daughter_age_l548_54824


namespace NUMINAMATH_CALUDE_sin_sum_from_sin_cos_sums_l548_54887

theorem sin_sum_from_sin_cos_sums (x y : Real) 
  (h1 : Real.sin x + Real.sin y = Real.sqrt 2 / 2)
  (h2 : Real.cos x + Real.cos y = Real.sqrt 6 / 2) :
  Real.sin (x + y) = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_from_sin_cos_sums_l548_54887


namespace NUMINAMATH_CALUDE_count_integers_with_2_and_3_l548_54801

def count_integers_with_digits (lower_bound upper_bound : ℕ) (digit1 digit2 : ℕ) : ℕ :=
  sorry

theorem count_integers_with_2_and_3 :
  count_integers_with_digits 1000 2000 2 3 = 108 :=
sorry

end NUMINAMATH_CALUDE_count_integers_with_2_and_3_l548_54801


namespace NUMINAMATH_CALUDE_minimum_trees_l548_54879

theorem minimum_trees (L : ℕ) (X : ℕ) : 
  (∀ n < L, ¬ ∃ m : ℕ, (0.13 : ℝ) * n < m ∧ m < (0.14 : ℝ) * n) →
  ((0.13 : ℝ) * L < X ∧ X < (0.14 : ℝ) * L) →
  L = 15 := by
sorry

end NUMINAMATH_CALUDE_minimum_trees_l548_54879


namespace NUMINAMATH_CALUDE_inequality_equivalence_l548_54851

theorem inequality_equivalence (x : ℝ) :
  -1 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l548_54851


namespace NUMINAMATH_CALUDE_unique_solution_equation_l548_54862

theorem unique_solution_equation :
  ∃! (a b : ℝ), 2 * (a^2 + 1) * (b^2 + 1) = (a + 1)^2 * (a * b + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l548_54862


namespace NUMINAMATH_CALUDE_fruit_pie_theorem_l548_54831

/-- Represents the quantities of fruits used in pie making -/
structure FruitQuantities where
  apples : ℕ
  peaches : ℕ
  pears : ℕ
  plums : ℕ

/-- The ratio of fruits used per apple in pie making -/
structure FruitRatio where
  peaches_per_apple : ℕ
  pears_per_apple : ℕ
  plums_per_apple : ℕ

/-- Calculate the quantities of fruits used given the number of apples and the ratio -/
def calculate_used_fruits (apples_used : ℕ) (ratio : FruitRatio) : FruitQuantities :=
  { apples := apples_used,
    peaches := apples_used * ratio.peaches_per_apple,
    pears := apples_used * ratio.pears_per_apple,
    plums := apples_used * ratio.plums_per_apple }

theorem fruit_pie_theorem (initial_apples initial_peaches initial_pears initial_plums : ℕ)
                          (ratio : FruitRatio)
                          (apples_left : ℕ) :
  initial_apples = 40 →
  initial_peaches = 54 →
  initial_pears = 60 →
  initial_plums = 48 →
  ratio.peaches_per_apple = 2 →
  ratio.pears_per_apple = 3 →
  ratio.plums_per_apple = 4 →
  apples_left = 39 →
  calculate_used_fruits (initial_apples - apples_left) ratio =
    { apples := 1, peaches := 2, pears := 3, plums := 4 } :=
by sorry


end NUMINAMATH_CALUDE_fruit_pie_theorem_l548_54831


namespace NUMINAMATH_CALUDE_max_pieces_of_cake_l548_54886

/-- The size of the cake in inches -/
def cake_size : ℕ := 16

/-- The size of each piece in inches -/
def piece_size : ℕ := 4

/-- The area of the cake in square inches -/
def cake_area : ℕ := cake_size * cake_size

/-- The area of each piece in square inches -/
def piece_area : ℕ := piece_size * piece_size

/-- The maximum number of pieces that can be cut from the cake -/
def max_pieces : ℕ := cake_area / piece_area

theorem max_pieces_of_cake :
  max_pieces = 16 :=
sorry

end NUMINAMATH_CALUDE_max_pieces_of_cake_l548_54886


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l548_54800

theorem scientific_notation_proof : 
  ∃ (a : ℝ) (n : ℤ), 680000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 6.8 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l548_54800


namespace NUMINAMATH_CALUDE_log_roots_sum_l548_54838

theorem log_roots_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 2 * (Real.log a)^2 + 4 * (Real.log a) + 1 = 0 ∧ 
       2 * (Real.log b)^2 + 4 * (Real.log b) + 1 = 0) : 
  (Real.log a)^2 + Real.log (a^2) + a * b = Real.exp (-2) - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_log_roots_sum_l548_54838


namespace NUMINAMATH_CALUDE_man_son_age_difference_l548_54868

/-- Represents the age difference between a man and his son -/
def ageDifference (manAge sonAge : ℕ) : ℕ := manAge - sonAge

/-- Theorem stating the age difference between a man and his son -/
theorem man_son_age_difference :
  ∀ (manAge sonAge : ℕ),
    sonAge = 18 →
    manAge + 2 = 2 * (sonAge + 2) →
    ageDifference manAge sonAge = 20 := by
  sorry


end NUMINAMATH_CALUDE_man_son_age_difference_l548_54868


namespace NUMINAMATH_CALUDE_average_weight_b_c_l548_54855

theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 42 →
  b = 35 →
  (b + c) / 2 = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l548_54855


namespace NUMINAMATH_CALUDE_flash_fraction_is_one_l548_54846

/-- The fraction of an hour it takes for a light to flash 120 times, given that it flashes every 30 seconds -/
def flash_fraction : ℚ :=
  let flash_interval : ℚ := 30 / 3600  -- 30 seconds as a fraction of an hour
  let total_flashes : ℕ := 120
  total_flashes * flash_interval

theorem flash_fraction_is_one : flash_fraction = 1 := by
  sorry

end NUMINAMATH_CALUDE_flash_fraction_is_one_l548_54846


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l548_54853

/-- Given a set {6, 13, 18, 4, x} where 10 is the arithmetic mean, prove that x = 9 -/
theorem arithmetic_mean_problem (x : ℝ) : 
  (6 + 13 + 18 + 4 + x) / 5 = 10 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l548_54853


namespace NUMINAMATH_CALUDE_area_not_perfect_square_l548_54881

/-- A primitive Pythagorean triple -/
structure PrimitivePythagoreanTriple where
  a : ℕ
  b : ℕ
  c : ℕ
  isPrimitive : Nat.gcd a b = 1
  isPythagorean : a^2 + b^2 = c^2

/-- The area of a right triangle with legs a and b is not a perfect square -/
theorem area_not_perfect_square (t : PrimitivePythagoreanTriple) :
  ¬ ∃ (n : ℕ), (t.a * t.b) / 2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_area_not_perfect_square_l548_54881
