import Mathlib

namespace NUMINAMATH_CALUDE_wrong_quotient_calculation_l2757_275724

theorem wrong_quotient_calculation (dividend : ℕ) (correct_divisor wrong_divisor correct_quotient : ℕ) : 
  dividend = correct_divisor * correct_quotient →
  correct_divisor = 21 →
  correct_quotient = 24 →
  wrong_divisor = 12 →
  dividend / wrong_divisor = 42 :=
by sorry

end NUMINAMATH_CALUDE_wrong_quotient_calculation_l2757_275724


namespace NUMINAMATH_CALUDE_gcd_lcm_45_150_l2757_275712

theorem gcd_lcm_45_150 : Nat.gcd 45 150 = 15 ∧ Nat.lcm 45 150 = 450 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_45_150_l2757_275712


namespace NUMINAMATH_CALUDE_potato_wedges_count_l2757_275740

/-- The number of wedges one potato can be cut into -/
def wedges_per_potato : ℕ := sorry

/-- The total number of potatoes harvested -/
def total_potatoes : ℕ := 67

/-- The number of potatoes cut into wedges -/
def wedge_potatoes : ℕ := 13

/-- The number of potato chips one potato can make -/
def chips_per_potato : ℕ := 20

/-- The difference between the number of potato chips and wedges -/
def chip_wedge_difference : ℕ := 436

theorem potato_wedges_count :
  wedges_per_potato = 8 ∧
  (total_potatoes - wedge_potatoes) / 2 * chips_per_potato - wedge_potatoes * wedges_per_potato = chip_wedge_difference :=
by sorry

end NUMINAMATH_CALUDE_potato_wedges_count_l2757_275740


namespace NUMINAMATH_CALUDE_constant_triangle_sum_l2757_275788

/-- Given a rectangle ABCD with width 'a' and height 'b', and a line 'r' parallel to AB
    intersecting diagonal AC at point (x₀, y₀), the sum of the areas of the two triangles
    formed by 'r' is constant and equal to (a*b)/2, regardless of the position of 'r'. -/
theorem constant_triangle_sum (a b x₀ : ℝ) (ha : a > 0) (hb : b > 0) (hx : 0 ≤ x₀ ∧ x₀ ≤ a) :
  let y₀ := (b / a) * x₀
  let area₁ := (1 / 2) * b * x₀
  let area₂ := (1 / 2) * b * (a - x₀)
  area₁ + area₂ = (a * b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_constant_triangle_sum_l2757_275788


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l2757_275766

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 :=
sorry

theorem min_value_reciprocal_sum_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 4 / y = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l2757_275766


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2757_275708

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2757_275708


namespace NUMINAMATH_CALUDE_platform_length_l2757_275787

/-- Given a train and platform with specific properties, prove the length of the platform. -/
theorem platform_length (train_length : ℝ) (time_cross_platform : ℝ) (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 40)
  (h3 : time_cross_pole = 18) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 367 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l2757_275787


namespace NUMINAMATH_CALUDE_set_B_equals_l2757_275781

def A : Set ℝ := {x | x^2 ≤ 4}

def B : Set ℕ := {x | x > 0 ∧ (x - 1 : ℝ) ∈ A}

theorem set_B_equals : B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_set_B_equals_l2757_275781


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_5_l2757_275761

theorem x_plus_2y_equals_5 (x y : ℝ) 
  (eq1 : 2 * x + y = 6) 
  (eq2 : (x + y) / 3 = 1.222222222222222) : 
  x + 2 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_5_l2757_275761


namespace NUMINAMATH_CALUDE_curve_E_perpendicular_points_sum_inverse_squares_l2757_275731

-- Define the curve E
def curve_E (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the property of perpendicular vectors
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem curve_E_perpendicular_points_sum_inverse_squares (x₁ y₁ x₂ y₂ : ℝ) :
  curve_E x₁ y₁ → curve_E x₂ y₂ → perpendicular x₁ y₁ x₂ y₂ →
  1 / (x₁^2 + y₁^2) + 1 / (x₂^2 + y₂^2) = 7 / 12 :=
by sorry

end NUMINAMATH_CALUDE_curve_E_perpendicular_points_sum_inverse_squares_l2757_275731


namespace NUMINAMATH_CALUDE_chicken_count_l2757_275789

theorem chicken_count (east : ℕ) (west : ℕ) : 
  east = 40 → 
  (east : ℚ) + west * (1 - 1/4 - 1/3) = (1/2 : ℚ) * (east + west) → 
  east + west = 280 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l2757_275789


namespace NUMINAMATH_CALUDE_units_digit_of_31_cubed_plus_13_cubed_l2757_275736

theorem units_digit_of_31_cubed_plus_13_cubed : ∃ n : ℕ, 31^3 + 13^3 = 10 * n + 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_31_cubed_plus_13_cubed_l2757_275736


namespace NUMINAMATH_CALUDE_lunch_break_duration_l2757_275743

-- Define the painting rates and lunch break duration
structure PaintingScenario where
  joseph_rate : ℝ
  helpers_rate : ℝ
  lunch_break : ℝ

-- Define the conditions from the problem
def monday_condition (s : PaintingScenario) : Prop :=
  (8 - s.lunch_break) * (s.joseph_rate + s.helpers_rate) = 0.6

def tuesday_condition (s : PaintingScenario) : Prop :=
  (5 - s.lunch_break) * s.helpers_rate = 0.3

def wednesday_condition (s : PaintingScenario) : Prop :=
  (6 - s.lunch_break) * s.joseph_rate = 0.1

-- Theorem stating that the lunch break is 45 minutes
theorem lunch_break_duration :
  ∃ (s : PaintingScenario),
    monday_condition s ∧
    tuesday_condition s ∧
    wednesday_condition s ∧
    s.lunch_break = 0.75 := by sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l2757_275743


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l2757_275723

theorem complex_square_one_plus_i : (1 + Complex.I) ^ 2 = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l2757_275723


namespace NUMINAMATH_CALUDE_binomial_10_2_l2757_275703

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_2_l2757_275703


namespace NUMINAMATH_CALUDE_no_perfect_squares_l2757_275702

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- State the theorem
theorem no_perfect_squares : 
  ¬(is_perfect_square (factorial 100 * factorial 101)) ∧
  ¬(is_perfect_square (factorial 100 * factorial 102)) ∧
  ¬(is_perfect_square (factorial 101 * factorial 102)) ∧
  ¬(is_perfect_square (factorial 101 * factorial 103)) ∧
  ¬(is_perfect_square (factorial 102 * factorial 103)) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l2757_275702


namespace NUMINAMATH_CALUDE_log_3125_base_5_between_consecutive_integers_l2757_275711

theorem log_3125_base_5_between_consecutive_integers :
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 3125 / Real.log 5 ∧ Real.log 3125 / Real.log 5 < (d : ℝ) ∧ c + d = 10 := by
  sorry

end NUMINAMATH_CALUDE_log_3125_base_5_between_consecutive_integers_l2757_275711


namespace NUMINAMATH_CALUDE_sum_a_b_equals_three_l2757_275710

theorem sum_a_b_equals_three (a b : ℝ) (h : |a - 4| + (b + 1)^2 = 0) : a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_three_l2757_275710


namespace NUMINAMATH_CALUDE_dividend_calculation_l2757_275705

theorem dividend_calculation (divisor quotient remainder : ℕ) :
  divisor = 15 →
  quotient = 8 →
  remainder = 5 →
  (divisor * quotient + remainder) = 125 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2757_275705


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2757_275791

theorem complex_power_magnitude : 
  Complex.abs ((2/5 : ℂ) + (7/5 : ℂ) * Complex.I) ^ 8 = 7890481/390625 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2757_275791


namespace NUMINAMATH_CALUDE_roots_of_equation_l2757_275797

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => 3*x*(x-1) - 2*(x-1)
  (f 1 = 0 ∧ f (2/3) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 2/3) := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2757_275797


namespace NUMINAMATH_CALUDE_integer_points_in_triangle_l2757_275774

def count_points (n : ℕ) : ℕ :=
  (n + 1) * (n + 2) / 2

theorem integer_points_in_triangle : count_points 7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_integer_points_in_triangle_l2757_275774


namespace NUMINAMATH_CALUDE_new_average_after_exclusion_l2757_275733

/-- Theorem: New average after excluding students with low marks -/
theorem new_average_after_exclusion
  (total_students : ℕ)
  (original_average : ℚ)
  (excluded_students : ℕ)
  (excluded_average : ℚ)
  (h1 : total_students = 33)
  (h2 : original_average = 90)
  (h3 : excluded_students = 3)
  (h4 : excluded_average = 40) :
  let remaining_students := total_students - excluded_students
  let total_marks := total_students * original_average
  let excluded_marks := excluded_students * excluded_average
  let remaining_marks := total_marks - excluded_marks
  (remaining_marks / remaining_students : ℚ) = 95 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_exclusion_l2757_275733


namespace NUMINAMATH_CALUDE_problem_statement_l2757_275718

theorem problem_statement (x y a b c : ℝ) : 
  (x = -y) → 
  (a * b = 1) → 
  (|c| = 2) → 
  ((((x + y) / 2)^2023) - ((-a * b)^2023) + c^3 = 9 ∨ 
   (((x + y) / 2)^2023) - ((-a * b)^2023) + c^3 = -7) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2757_275718


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_two_l2757_275734

-- Define the polynomial
def P (a b x : ℝ) : ℝ := a * (x^3 - x^2 + 3*x) + b * (2*x^2 + x) + x^3 - 5

-- State the theorem
theorem polynomial_value_at_negative_two 
  (a b : ℝ) 
  (h : P a b 2 = -17) : 
  P a b (-2) = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_two_l2757_275734


namespace NUMINAMATH_CALUDE_xy_equals_ten_l2757_275771

theorem xy_equals_ten (x y : ℝ) (h : x * (x + y) = x^2 + 10) : x * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_ten_l2757_275771


namespace NUMINAMATH_CALUDE_pizza_slices_count_l2757_275729

-- Define the number of pizzas
def num_pizzas : Nat := 4

-- Define the number of slices for each type of pizza
def slices_first_two : Nat := 8
def slices_third : Nat := 10
def slices_fourth : Nat := 12

-- Define the total number of slices
def total_slices : Nat := 2 * slices_first_two + slices_third + slices_fourth

-- Theorem to prove
theorem pizza_slices_count : total_slices = 38 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_count_l2757_275729


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l2757_275759

theorem cryptarithmetic_puzzle (F I V E G H T : ℕ) : 
  (F = 8) →
  (V % 2 = 1) →
  (100 * F + 10 * I + V + 100 * F + 10 * I + V = 10000 * E + 1000 * I + 100 * G + 10 * H + T) →
  (F ≠ I ∧ F ≠ V ∧ F ≠ E ∧ F ≠ G ∧ F ≠ H ∧ F ≠ T ∧
   I ≠ V ∧ I ≠ E ∧ I ≠ G ∧ I ≠ H ∧ I ≠ T ∧
   V ≠ E ∧ V ≠ G ∧ V ≠ H ∧ V ≠ T ∧
   E ≠ G ∧ E ≠ H ∧ E ≠ T ∧
   G ≠ H ∧ G ≠ T ∧
   H ≠ T) →
  (F < 10 ∧ I < 10 ∧ V < 10 ∧ E < 10 ∧ G < 10 ∧ H < 10 ∧ T < 10) →
  I = 2 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l2757_275759


namespace NUMINAMATH_CALUDE_remainder_theorem_l2757_275732

theorem remainder_theorem (x y u v : ℕ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x = u * y + v) (h4 : v < y) :
  (x + 4 * u * y) % y = v := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2757_275732


namespace NUMINAMATH_CALUDE_andre_gave_23_flowers_l2757_275730

/-- The number of flowers Rosa initially had -/
def initial_flowers : ℕ := 67

/-- The number of flowers Rosa has now -/
def final_flowers : ℕ := 90

/-- The number of flowers Andre gave to Rosa -/
def andre_flowers : ℕ := final_flowers - initial_flowers

theorem andre_gave_23_flowers : andre_flowers = 23 := by
  sorry

end NUMINAMATH_CALUDE_andre_gave_23_flowers_l2757_275730


namespace NUMINAMATH_CALUDE_abc_inequality_and_reciprocal_sum_l2757_275716

theorem abc_inequality_and_reciprocal_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 2) : 
  a * b * c ≤ 8 / 27 ∧ 1 / a + 1 / b + 1 / c ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_and_reciprocal_sum_l2757_275716


namespace NUMINAMATH_CALUDE_contractor_absence_l2757_275715

theorem contractor_absence (total_days : ℕ) (daily_pay : ℚ) (daily_fine : ℚ) (total_received : ℚ) :
  total_days = 30 ∧ 
  daily_pay = 25 ∧ 
  daily_fine = 7.5 ∧ 
  total_received = 555 →
  ∃ (days_worked days_absent : ℕ),
    days_worked + days_absent = total_days ∧
    daily_pay * days_worked - daily_fine * days_absent = total_received ∧
    days_absent = 6 :=
by sorry

end NUMINAMATH_CALUDE_contractor_absence_l2757_275715


namespace NUMINAMATH_CALUDE_biology_quiz_probability_l2757_275763

theorem biology_quiz_probability : 
  let n : ℕ := 6  -- number of guessed questions
  let k : ℕ := 4  -- number of possible answers per question
  let p : ℚ := 1 / k  -- probability of guessing correctly on a single question
  1 - (1 - p) ^ n = 3367 / 4096 :=
by
  sorry

end NUMINAMATH_CALUDE_biology_quiz_probability_l2757_275763


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2757_275772

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a + b)^2 + (b + c)^2 + (c + a)^2 = 2*(a + b + c) + 6*a*b*c) :
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ |2*(a + b + c) - 6*a*b*c| := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2757_275772


namespace NUMINAMATH_CALUDE_rayden_extra_birds_l2757_275700

def lily_ducks : ℕ := 20
def lily_geese : ℕ := 10

def rayden_ducks : ℕ := 3 * lily_ducks
def rayden_geese : ℕ := 4 * lily_geese

theorem rayden_extra_birds : 
  (rayden_ducks + rayden_geese) - (lily_ducks + lily_geese) = 70 := by
  sorry

end NUMINAMATH_CALUDE_rayden_extra_birds_l2757_275700


namespace NUMINAMATH_CALUDE_intersection_range_l2757_275748

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0 ∧ y > 0

-- Define the line
def line (k x y : ℝ) : Prop := y = k*(x + 2)

-- Define the intersection condition
def intersects (k : ℝ) : Prop :=
  ∃ x y, curve x y ∧ line k x y

-- State the theorem
theorem intersection_range :
  ∀ k, intersects k ↔ k > 0 ∧ k ≤ 3/4 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l2757_275748


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2757_275762

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 2*x^4 - 5*x^3 + 9 = 
  (x - 2) * (x^5 + 2*x^4 + 6*x^3 + 7*x^2 + 14*x + 28) + R :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2757_275762


namespace NUMINAMATH_CALUDE_square_sum_equality_l2757_275796

theorem square_sum_equality (n : ℤ) : n + n + n + n = 4 * n := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l2757_275796


namespace NUMINAMATH_CALUDE_family_probability_l2757_275755

theorem family_probability : 
  let p_boy : ℝ := 1/2
  let p_girl : ℝ := 1/2
  let num_children : ℕ := 4
  let p_all_boys : ℝ := p_boy ^ num_children
  let p_all_girls : ℝ := p_girl ^ num_children
  let p_at_least_one_of_each : ℝ := 1 - p_all_boys - p_all_girls
  p_at_least_one_of_each = 7/8 := by
sorry

end NUMINAMATH_CALUDE_family_probability_l2757_275755


namespace NUMINAMATH_CALUDE_production_decrease_l2757_275739

theorem production_decrease (x : ℝ) : 
  (1 - x / 100) * (1 - x / 100) = 0.49 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_production_decrease_l2757_275739


namespace NUMINAMATH_CALUDE_curve_tangent_l2757_275738

/-- Given a curve C defined by x = √2 cos(φ) and y = sin(φ), prove that for a point M on C,
    if the angle between OM and the positive x-axis is π/3, then tan(φ) = √6. -/
theorem curve_tangent (φ : ℝ) : 
  let M : ℝ × ℝ := (Real.sqrt 2 * Real.cos φ, Real.sin φ)
  (M.2 / M.1 = Real.tan (π / 3)) → Real.tan φ = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_l2757_275738


namespace NUMINAMATH_CALUDE_max_candies_eaten_l2757_275713

theorem max_candies_eaten (n : ℕ) (h : n = 28) : 
  (n * (n - 1)) / 2 = 378 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_eaten_l2757_275713


namespace NUMINAMATH_CALUDE_determinant_zero_l2757_275799

theorem determinant_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![1, Real.sin (a - b), Real.sin a],
    ![Real.sin (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_determinant_zero_l2757_275799


namespace NUMINAMATH_CALUDE_sandwich_combinations_l2757_275785

theorem sandwich_combinations (n_toppings : Nat) (n_sauces : Nat) :
  n_toppings = 7 → n_sauces = 3 →
  (Nat.choose n_toppings 2) * (Nat.choose n_sauces 1) = 63 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l2757_275785


namespace NUMINAMATH_CALUDE_min_fraction_sum_l2757_275767

/-- The set of digits to choose from -/
def digits : Finset ℕ := {0, 1, 5, 6, 7, 8, 9}

/-- The proposition that four natural numbers are distinct digits from our set -/
def are_distinct_digits (w x y z : ℕ) : Prop :=
  w ∈ digits ∧ x ∈ digits ∧ y ∈ digits ∧ z ∈ digits ∧
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z

/-- The theorem stating the minimum value of the sum -/
theorem min_fraction_sum :
  ∃ (w x y z : ℕ), are_distinct_digits w x y z ∧ x ≠ 0 ∧ z ≠ 0 ∧
  (∀ (w' x' y' z' : ℕ), are_distinct_digits w' x' y' z' ∧ x' ≠ 0 ∧ z' ≠ 0 →
    (w : ℚ) / x + (y : ℚ) / z ≤ (w' : ℚ) / x' + (y' : ℚ) / z') ∧
  (w : ℚ) / x + (y : ℚ) / z = 1 / 8 :=
sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l2757_275767


namespace NUMINAMATH_CALUDE_cube_decomposition_smallest_term_l2757_275721

theorem cube_decomposition_smallest_term (m : ℕ) (h1 : m ≥ 2) : 
  (m^2 - m + 1 = 73) → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_decomposition_smallest_term_l2757_275721


namespace NUMINAMATH_CALUDE_investment_dividend_calculation_l2757_275775

/-- Calculates the dividend received from an investment in shares with premium and dividend rates -/
def calculate_dividend (investment : ℚ) (share_value : ℚ) (premium_rate : ℚ) (dividend_rate : ℚ) : ℚ :=
  let share_cost := share_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := share_value * dividend_rate
  num_shares * dividend_per_share

/-- Theorem stating that the given investment yields the correct dividend -/
theorem investment_dividend_calculation :
  calculate_dividend 14400 100 (20/100) (7/100) = 840 := by
  sorry

#eval calculate_dividend 14400 100 (20/100) (7/100)

end NUMINAMATH_CALUDE_investment_dividend_calculation_l2757_275775


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2757_275746

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - 3*a = 0 ∧ x = -2) →
  (∃ y : ℝ, y^2 - a*y - 3*a = 0 ∧ y = 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2757_275746


namespace NUMINAMATH_CALUDE_max_value_theorem_l2757_275704

open Real

noncomputable def e : ℝ := Real.exp 1

theorem max_value_theorem (a b : ℝ) :
  (∀ x : ℝ, (e - a) * (Real.exp x) + x + b + 1 ≤ 0) →
  (b + 1) / a ≤ 1 / e :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2757_275704


namespace NUMINAMATH_CALUDE_factory_defect_rate_l2757_275764

theorem factory_defect_rate (total_output : ℝ) : 
  let machine_a_output := 0.4 * total_output
  let machine_b_output := 0.6 * total_output
  let machine_a_defect_rate := 9 / 1000
  let total_defect_rate := 0.0156
  ∃ (machine_b_defect_rate : ℝ),
    0.4 * machine_a_defect_rate + 0.6 * machine_b_defect_rate = total_defect_rate ∧
    1 / machine_b_defect_rate = 50 := by
  sorry

end NUMINAMATH_CALUDE_factory_defect_rate_l2757_275764


namespace NUMINAMATH_CALUDE_tan_pi_4_minus_theta_l2757_275793

theorem tan_pi_4_minus_theta (θ : Real) 
  (h1 : θ > -π/2 ∧ θ < 0) 
  (h2 : Real.cos (2*θ) - 3*Real.sin (θ - π/2) = 1) : 
  Real.tan (π/4 - θ) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_4_minus_theta_l2757_275793


namespace NUMINAMATH_CALUDE_lynne_magazines_l2757_275777

def num_books : ℕ := 9
def book_cost : ℕ := 7
def magazine_cost : ℕ := 4
def total_spent : ℕ := 75

theorem lynne_magazines :
  ∃ (num_magazines : ℕ),
    num_magazines * magazine_cost + num_books * book_cost = total_spent ∧
    num_magazines = 3 := by
  sorry

end NUMINAMATH_CALUDE_lynne_magazines_l2757_275777


namespace NUMINAMATH_CALUDE_stuffed_animals_problem_l2757_275773

theorem stuffed_animals_problem (M : ℕ) : 
  (M + 2*M + (2*M + 5) = 175) → M = 34 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_problem_l2757_275773


namespace NUMINAMATH_CALUDE_tyler_aquariums_l2757_275794

-- Define the given conditions
def animals_per_aquarium : ℕ := 64
def total_animals : ℕ := 512

-- State the theorem
theorem tyler_aquariums : 
  total_animals / animals_per_aquarium = 8 := by
  sorry

end NUMINAMATH_CALUDE_tyler_aquariums_l2757_275794


namespace NUMINAMATH_CALUDE_complex_subtraction_l2757_275780

theorem complex_subtraction (a b : ℂ) (h1 : a = 6 - 3*I) (h2 : b = 2 + 4*I) : 
  a - 3*b = -15*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2757_275780


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2757_275795

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := x ≥ 0

-- Define the concept of simplest quadratic radical
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ 
  ∀ y : ℝ, y > 1 → ¬(∃ z : ℝ, x = y * z^2)

-- Define the set of given expressions
def GivenExpressions : Set ℝ := {8, 1/3, 6, 0.1}

-- Theorem statement
theorem simplest_quadratic_radical :
  ∀ x ∈ GivenExpressions, 
    SimplestQuadraticRadical (Real.sqrt x) → x = 6 :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2757_275795


namespace NUMINAMATH_CALUDE_certain_number_value_l2757_275719

theorem certain_number_value (x y z : ℝ) 
  (h1 : 0.5 * x = y + z) 
  (h2 : x - 2 * y = 40) : 
  z = 20 := by sorry

end NUMINAMATH_CALUDE_certain_number_value_l2757_275719


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2757_275728

/-- A geometric sequence with given first and fourth terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 1)
  (h_fourth : a 4 = 27) :
  a 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2757_275728


namespace NUMINAMATH_CALUDE_parabola_above_line_l2757_275735

/-- Given non-zero real numbers a, b, and c, if the parabola y = ax^2 + bx + c is positioned
    above the line y = cx, then the parabola y = cx^2 - bx + a is positioned above
    the line y = cx - b. -/
theorem parabola_above_line (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_above : ∀ x, a * x^2 + b * x + c > c * x) :
  ∀ x, c * x^2 - b * x + a > c * x - b :=
by sorry

end NUMINAMATH_CALUDE_parabola_above_line_l2757_275735


namespace NUMINAMATH_CALUDE_no_simple_algebraic_solution_l2757_275750

variable (g V₀ a S V t : ℝ)

def velocity_equation := V = g * t + V₀

def displacement_equation := S = (1/2) * g * t^2 + V₀ * t + (1/3) * a * t^3

theorem no_simple_algebraic_solution :
  ∀ g V₀ a S V t : ℝ,
  velocity_equation g V₀ V t →
  displacement_equation g V₀ a S t →
  ¬∃ f : ℝ → ℝ → ℝ → ℝ → ℝ, t = f S g V₀ a :=
by sorry

end NUMINAMATH_CALUDE_no_simple_algebraic_solution_l2757_275750


namespace NUMINAMATH_CALUDE_regular_polygon_18_degree_exterior_angles_l2757_275714

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides -/
theorem regular_polygon_18_degree_exterior_angles (n : ℕ) : 
  (n > 0) → (360 / n = 18) → n = 20 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_18_degree_exterior_angles_l2757_275714


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2757_275744

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  -- Define the properties of the unit circle and 45° angle
  have unit_circle : ∀ θ, Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1 := by sorry
  have symmetry_45 : Real.sin (π / 4) = Real.cos (π / 4) := by sorry

  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2757_275744


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l2757_275720

def line (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 1

def opposite_sides (a : ℝ) : Prop :=
  line a 1 1 > 0 ∧ line a 0 (-2) < 0

def range_of_a : Set ℝ := { a | a < -2 ∨ a > 1/2 }

theorem range_of_a_theorem :
  ∀ a : ℝ, opposite_sides a ↔ a ∈ range_of_a := by sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l2757_275720


namespace NUMINAMATH_CALUDE_jade_transactions_jade_transactions_proof_l2757_275757

/-- Proves that Jade handled 80 transactions given the specified conditions. -/
theorem jade_transactions : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mabel_transactions anthony_transactions cal_transactions jade_transactions =>
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    cal_transactions = anthony_transactions * 2 / 3 →
    jade_transactions = cal_transactions + 14 →
    jade_transactions = 80

/-- Proof of the theorem -/
theorem jade_transactions_proof : jade_transactions 90 99 66 80 := by
  sorry

end NUMINAMATH_CALUDE_jade_transactions_jade_transactions_proof_l2757_275757


namespace NUMINAMATH_CALUDE_contrapositive_implies_range_l2757_275725

/-- The condition p for a real number x given a real number a -/
def p (a : ℝ) (x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- The condition q for a real number x -/
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

/-- The range of values for a -/
def a_range (a : ℝ) : Prop := a ≥ 2 ∨ a ≤ -4

theorem contrapositive_implies_range (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, ¬q x → ¬p a x) → a_range a :=
sorry

end NUMINAMATH_CALUDE_contrapositive_implies_range_l2757_275725


namespace NUMINAMATH_CALUDE_nancy_chip_distribution_l2757_275737

/-- The number of tortilla chips Nancy initially had -/
def initial_chips : ℕ := 22

/-- The number of tortilla chips Nancy gave to her brother -/
def brother_chips : ℕ := 7

/-- The number of tortilla chips Nancy kept for herself -/
def nancy_chips : ℕ := 10

/-- The number of tortilla chips Nancy gave to her sister -/
def sister_chips : ℕ := initial_chips - brother_chips - nancy_chips

theorem nancy_chip_distribution :
  sister_chips = 5 := by sorry

end NUMINAMATH_CALUDE_nancy_chip_distribution_l2757_275737


namespace NUMINAMATH_CALUDE_power_calculation_l2757_275717

theorem power_calculation : (8^5 / 8^2) * 2^10 / 2^3 = 65536 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2757_275717


namespace NUMINAMATH_CALUDE_problem_solution_l2757_275765

open Real

-- Define proposition p
def p : Prop := ∀ x : ℝ, (2:ℝ)^x + (1:ℝ) / (2:ℝ)^x > 2

-- Define proposition q
def q : Prop := ∃ x : ℝ, x ∈ Set.Icc 0 (π/2) ∧ sin x + cos x = 1/2

-- Theorem statement
theorem problem_solution : ¬p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2757_275765


namespace NUMINAMATH_CALUDE_soccer_sideline_time_l2757_275753

/-- Given a soccer game duration and a player's playing times, calculate the time spent on the sideline -/
theorem soccer_sideline_time (game_duration playing_time1 playing_time2 : ℕ) 
  (h1 : game_duration = 90)
  (h2 : playing_time1 = 20)
  (h3 : playing_time2 = 35) :
  game_duration - (playing_time1 + playing_time2) = 35 := by
  sorry

end NUMINAMATH_CALUDE_soccer_sideline_time_l2757_275753


namespace NUMINAMATH_CALUDE_abs_sum_inequality_positive_reals_inequality_l2757_275768

-- Problem 1
theorem abs_sum_inequality (x : ℝ) :
  |x - 1| + |x + 1| ≤ 4 ↔ x ∈ Set.Icc (-2 : ℝ) 2 := by sorry

-- Problem 2
theorem positive_reals_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ a + b + c := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_positive_reals_inequality_l2757_275768


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2757_275779

theorem simplify_complex_fraction :
  let x := (1 : ℝ) / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 6 - 2)))
  x = (3 * Real.sqrt 5 + 2 * Real.sqrt 6 + 2) / 29 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2757_275779


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2757_275709

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = -x^2 + 4}
def N : Set ℝ := {x | x > 0}

-- State the theorem
theorem set_intersection_theorem :
  M ∩ N = Set.Ioo 0 4 := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2757_275709


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l2757_275758

theorem least_positive_angle_theorem (θ : Real) : 
  (θ > 0 ∧ Real.cos (5 * π / 180) = Real.sin (25 * π / 180) + Real.sin θ) →
  θ = 35 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l2757_275758


namespace NUMINAMATH_CALUDE_max_points_for_top_teams_is_76_l2757_275776

/-- Represents a soccer league with given parameters -/
structure SoccerLeague where
  numTeams : Nat
  gamesAgainstEachTeam : Nat
  pointsForWin : Nat
  pointsForDraw : Nat
  pointsForLoss : Nat

/-- Calculates the maximum possible points for each of the top three teams in the league -/
def maxPointsForTopTeams (league : SoccerLeague) : Nat :=
  sorry

/-- Theorem stating the maximum points for top teams in the specific league configuration -/
theorem max_points_for_top_teams_is_76 :
  let league : SoccerLeague := {
    numTeams := 9
    gamesAgainstEachTeam := 4
    pointsForWin := 3
    pointsForDraw := 1
    pointsForLoss := 0
  }
  maxPointsForTopTeams league = 76 := by sorry

end NUMINAMATH_CALUDE_max_points_for_top_teams_is_76_l2757_275776


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2757_275756

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2 * |x + a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x > 1} = Set.Ioo (-2) (-2/3) := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ x ∈ Set.Icc 2 3, (∀ a : ℝ, f a x > 0) → a ∈ Set.Ioo (-5/2) (-2) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2757_275756


namespace NUMINAMATH_CALUDE_five_travelers_three_rooms_l2757_275749

/-- The number of ways to arrange travelers into guest rooms -/
def arrange_travelers (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- There are at least 1 traveler in each room -/
axiom at_least_one (n k : ℕ) : arrange_travelers n k > 0 → k ≤ n

theorem five_travelers_three_rooms :
  arrange_travelers 5 3 = 150 :=
sorry

end NUMINAMATH_CALUDE_five_travelers_three_rooms_l2757_275749


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2757_275751

theorem square_sum_given_sum_and_product (a b : ℝ) : 
  a + b = 6 → a * b = 3 → a^2 + b^2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2757_275751


namespace NUMINAMATH_CALUDE_circle_radius_three_inches_l2757_275752

theorem circle_radius_three_inches (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_three_inches_l2757_275752


namespace NUMINAMATH_CALUDE_car_dealership_problem_l2757_275742

theorem car_dealership_problem (initial_cars : ℕ) (initial_silver_percent : ℚ)
  (new_shipment : ℕ) (final_silver_percent : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percent = 1/5)
  (h3 : new_shipment = 80)
  (h4 : final_silver_percent = 3/10) :
  (1 - (final_silver_percent * (initial_cars + new_shipment) - initial_silver_percent * initial_cars) / new_shipment) = 13/20 := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_problem_l2757_275742


namespace NUMINAMATH_CALUDE_line_through_first_third_quadrants_positive_slope_l2757_275760

/-- A line passing through the first and third quadrants has a positive slope -/
theorem line_through_first_third_quadrants_positive_slope (k : ℝ) 
  (h1 : k ≠ 0) 
  (h2 : ∀ (x y : ℝ), y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) : 
  k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_first_third_quadrants_positive_slope_l2757_275760


namespace NUMINAMATH_CALUDE_smallest_add_to_multiple_of_five_l2757_275792

theorem smallest_add_to_multiple_of_five : ∃ (n : ℕ), n > 0 ∧ (729 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (729 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_add_to_multiple_of_five_l2757_275792


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2757_275722

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 1 - 5 * Complex.I) : 
  z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2757_275722


namespace NUMINAMATH_CALUDE_tangent_circle_to_sphere_reasoning_l2757_275701

/-- Represents the type of reasoning used in geometric analogies --/
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical
  | Other

/-- Represents a geometric property in 2D --/
structure Property2D where
  statement : String

/-- Represents a geometric property in 3D --/
structure Property3D where
  statement : String

/-- The property of tangent lines to circles in 2D --/
def tangentLineCircle : Property2D :=
  { statement := "When a line is tangent to a circle, the line connecting the center of the circle to the tangent point is perpendicular to the line" }

/-- The property of tangent planes to spheres in 3D --/
def tangentPlaneSphere : Property3D :=
  { statement := "When a plane is tangent to a sphere, the line connecting the center of the sphere to the tangent point is perpendicular to the plane" }

/-- The theorem stating that the reasoning used to extend the 2D property to 3D is analogical --/
theorem tangent_circle_to_sphere_reasoning :
  (∃ (p2d : Property2D) (p3d : Property3D), p2d = tangentLineCircle ∧ p3d = tangentPlaneSphere) →
  (∃ (r : ReasoningType), r = ReasoningType.Analogical) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_to_sphere_reasoning_l2757_275701


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l2757_275754

-- Define the asymptotes of the hyperbola
def asymptote_slope : ℝ := 2

-- Define the ellipse that shares foci with the hyperbola
def ellipse_equation (x y : ℝ) : Prop := x^2 / 49 + y^2 / 24 = 1

-- Define the standard form of a hyperbola
def hyperbola_equation (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_standard_form :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), y = asymptote_slope * x ∨ y = -asymptote_slope * x) →
  (∀ (x y : ℝ), ellipse_equation x y ↔ 
    ∃ (x' y' : ℝ), hyperbola_equation a b x' y' ∧ 
    (x - x')^2 + (y - y')^2 = 0) →
  a^2 = 5 ∧ b^2 = 20 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l2757_275754


namespace NUMINAMATH_CALUDE_square_of_102_l2757_275786

theorem square_of_102 : 102 * 102 = 10404 := by
  sorry

end NUMINAMATH_CALUDE_square_of_102_l2757_275786


namespace NUMINAMATH_CALUDE_tony_education_timeline_l2757_275706

theorem tony_education_timeline (total_years graduate_years : ℕ) 
  (h1 : total_years = 14)
  (h2 : graduate_years = 2) :
  ∃ (science_degree_years : ℕ), 
    science_degree_years * 3 + graduate_years = total_years ∧ 
    science_degree_years = 4 := by
  sorry

end NUMINAMATH_CALUDE_tony_education_timeline_l2757_275706


namespace NUMINAMATH_CALUDE_solve_equation_l2757_275741

theorem solve_equation (x : ℝ) : 1 - 2 / (1 + x) = 1 / (1 + x) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2757_275741


namespace NUMINAMATH_CALUDE_players_sold_is_two_l2757_275782

/-- Represents the financial transactions of a football club --/
def football_club_transactions 
  (initial_balance : ℚ) 
  (selling_price : ℚ) 
  (buying_price : ℚ) 
  (players_bought : ℕ) 
  (final_balance : ℚ) : Prop :=
  ∃ (players_sold : ℕ), 
    initial_balance + (selling_price * players_sold) - (buying_price * players_bought) = final_balance

/-- Theorem stating that the number of players sold is 2 --/
theorem players_sold_is_two : 
  football_club_transactions 100 10 15 4 60 → 
  ∃ (players_sold : ℕ), players_sold = 2 := by
  sorry

end NUMINAMATH_CALUDE_players_sold_is_two_l2757_275782


namespace NUMINAMATH_CALUDE_pancakes_remaining_l2757_275790

theorem pancakes_remaining (total : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) : 
  total = 21 → bobby_ate = 5 → dog_ate = 7 → total - (bobby_ate + dog_ate) = 9 := by
  sorry

end NUMINAMATH_CALUDE_pancakes_remaining_l2757_275790


namespace NUMINAMATH_CALUDE_lab_capacity_l2757_275783

/-- Represents a chemistry lab with work-stations for students -/
structure ChemistryLab where
  total_stations : ℕ
  two_student_stations : ℕ
  three_student_stations : ℕ
  station_sum : total_stations = two_student_stations + three_student_stations

/-- Calculates the total number of students that can use the lab at one time -/
def total_students (lab : ChemistryLab) : ℕ :=
  2 * lab.two_student_stations + 3 * lab.three_student_stations

/-- Theorem stating the number of students that can use the lab at one time -/
theorem lab_capacity (lab : ChemistryLab) 
    (h1 : lab.total_stations = 16)
    (h2 : lab.two_student_stations = 10) :
  total_students lab = 38 := by
  sorry

#eval total_students { total_stations := 16, two_student_stations := 10, three_student_stations := 6, station_sum := rfl }

end NUMINAMATH_CALUDE_lab_capacity_l2757_275783


namespace NUMINAMATH_CALUDE_binomial_320_320_l2757_275778

theorem binomial_320_320 : Nat.choose 320 320 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_320_320_l2757_275778


namespace NUMINAMATH_CALUDE_root_of_cubic_equation_l2757_275798

theorem root_of_cubic_equation :
  let x : ℝ := Real.sin (π / 14)
  (0 < x ∧ x < Real.pi / 13) ∧
  8 * x^3 - 4 * x^2 - 4 * x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_of_cubic_equation_l2757_275798


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2757_275769

theorem parallelogram_side_length 
  (s : ℝ) 
  (side1 : ℝ) 
  (side2 : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h : side1 = 3 * s) 
  (h' : side2 = s) 
  (h'' : angle = π / 3) 
  (h''' : area = 9 * Real.sqrt 3) 
  (h'''' : area = side1 * side2 * Real.sin angle) : 
  s = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2757_275769


namespace NUMINAMATH_CALUDE_min_a_for_inequality_l2757_275707

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + 2*a*x + 1 ≥ 0) ↔ a ≥ -5/4 := by
  sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_l2757_275707


namespace NUMINAMATH_CALUDE_log_half_condition_l2757_275784

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Theorem stating that x < 1 is a necessary but not sufficient condition for log_half(x) > 0
theorem log_half_condition (x : ℝ) :
  (log_half x > 0 → x < 1) ∧ ¬(x < 1 → log_half x > 0) :=
by sorry

end NUMINAMATH_CALUDE_log_half_condition_l2757_275784


namespace NUMINAMATH_CALUDE_rhombus_y_coord_rhombus_x_coord_range_l2757_275726

/-- A rhombus ABCD with specific properties -/
structure Rhombus where
  /-- The y-coordinate of point B -/
  b : ℝ
  /-- The x-coordinate of point D -/
  x : ℝ
  /-- The y-coordinate of point D -/
  y : ℝ
  /-- B is on the negative half of the y-axis -/
  h_b_neg : b < 0
  /-- ABCD is a rhombus -/
  h_is_rhombus : True  -- This is a placeholder, as we can't directly represent "is_rhombus" in this simple structure
  /-- The intersection of diagonals M falls on the x-axis -/
  h_diag_intersect : True  -- This is a placeholder for the diagonal intersection condition

/-- Theorem: In the given rhombus, y = -b - 1 -/
theorem rhombus_y_coord (r : Rhombus) : r.y = -r.b - 1 := by
  sorry

/-- The x-coordinate of point D can be any real number -/
theorem rhombus_x_coord_range (r : Rhombus) : r.x ∈ Set.univ := by
  sorry

end NUMINAMATH_CALUDE_rhombus_y_coord_rhombus_x_coord_range_l2757_275726


namespace NUMINAMATH_CALUDE_investment_income_percentage_l2757_275745

/-- Proves that the total annual income from two investments is 6% of the total invested amount -/
theorem investment_income_percentage : ∀ (investment1 investment2 rate1 rate2 : ℝ),
  investment1 = 2400 →
  investment2 = 2399.9999999999995 →
  rate1 = 0.04 →
  rate2 = 0.08 →
  let total_investment := investment1 + investment2
  let total_income := investment1 * rate1 + investment2 * rate2
  (total_income / total_investment) * 100 = 6 := by sorry

end NUMINAMATH_CALUDE_investment_income_percentage_l2757_275745


namespace NUMINAMATH_CALUDE_college_entrance_exam_score_l2757_275770

theorem college_entrance_exam_score (total_questions unanswered_questions answered_questions correct_answers incorrect_answers : ℕ)
  (raw_score : ℚ) :
  total_questions = 85 →
  unanswered_questions = 3 →
  answered_questions = 82 →
  answered_questions = correct_answers + incorrect_answers →
  raw_score = 67 →
  raw_score = correct_answers - 0.25 * incorrect_answers →
  correct_answers = 70 := by
sorry

end NUMINAMATH_CALUDE_college_entrance_exam_score_l2757_275770


namespace NUMINAMATH_CALUDE_book_probabilities_l2757_275727

/-- Represents the book collection with given properties -/
structure BookCollection where
  total : ℕ
  liberal_arts : ℕ
  hardcover : ℕ
  softcover_science : ℕ
  total_eq : total = 100
  liberal_arts_eq : liberal_arts = 40
  hardcover_eq : hardcover = 70
  softcover_science_eq : softcover_science = 20

/-- Calculates the probability of selecting a liberal arts hardcover book -/
def prob_liberal_arts_hardcover (bc : BookCollection) : ℚ :=
  (bc.hardcover - bc.softcover_science : ℚ) / bc.total

/-- Calculates the probability of selecting a liberal arts book then a hardcover book -/
def prob_liberal_arts_then_hardcover (bc : BookCollection) : ℚ :=
  (bc.liberal_arts : ℚ) / bc.total * (bc.hardcover : ℚ) / bc.total

/-- Main theorem stating the probabilities -/
theorem book_probabilities (bc : BookCollection) :
    prob_liberal_arts_hardcover bc = 3/10 ∧
    prob_liberal_arts_then_hardcover bc = 28/100 := by
  sorry

end NUMINAMATH_CALUDE_book_probabilities_l2757_275727


namespace NUMINAMATH_CALUDE_total_price_is_23_l2757_275747

/-- The price of cucumbers in dollars per kilogram -/
def cucumber_price : ℝ := 5

/-- The price of tomatoes in dollars per kilogram -/
def tomato_price : ℝ := cucumber_price * (1 - 0.2)

/-- The total price of tomatoes and cucumbers -/
def total_price : ℝ := 2 * tomato_price + 3 * cucumber_price

theorem total_price_is_23 : total_price = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_price_is_23_l2757_275747
