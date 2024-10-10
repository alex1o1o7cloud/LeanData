import Mathlib

namespace arithmetic_sequence_constant_sum_l965_96556

theorem arithmetic_sequence_constant_sum (a₁ d : ℝ) :
  let a : ℕ → ℝ := λ n => a₁ + (n - 1) * d
  let S : ℕ → ℝ := λ n => n * (2 * a₁ + (n - 1) * d) / 2
  (∀ a₁' d', a₁' + (1 + 7 + 10) * d' = a₁ + (1 + 7 + 10) * d) →
  (∀ a₁' d', S 13 = 13 * (2 * a₁' + 12 * d') / 2) :=
by sorry

end arithmetic_sequence_constant_sum_l965_96556


namespace probability_of_white_ball_l965_96509

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.3 →
  p_black = 0.5 →
  p_red + p_black + p_white = 1 →
  p_white = 0.2 := by
sorry

end probability_of_white_ball_l965_96509


namespace zoo_lion_cubs_l965_96540

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

end zoo_lion_cubs_l965_96540


namespace neg_f_is_reflection_about_x_axis_l965_96550

/-- A function representing the original graph -/
def f : ℝ → ℝ := sorry

/-- The negation of function f -/
def neg_f (x : ℝ) : ℝ := -f x

/-- Theorem stating that neg_f is a reflection of f about the x-axis -/
theorem neg_f_is_reflection_about_x_axis :
  ∀ x y : ℝ, f x = y ↔ neg_f x = -y :=
sorry

end neg_f_is_reflection_about_x_axis_l965_96550


namespace sum_three_digit_numbers_eq_255744_l965_96519

/-- The sum of all three-digit natural numbers with digits ranging from 1 to 8 -/
def sum_three_digit_numbers : ℕ :=
  let digit_sum : ℕ := (8 * 9) / 2  -- Sum of digits from 1 to 8
  let digit_count : ℕ := 8 * 8      -- Number of times each digit appears in each place
  let place_sum : ℕ := digit_sum * digit_count
  place_sum * 111

theorem sum_three_digit_numbers_eq_255744 :
  sum_three_digit_numbers = 255744 := by
  sorry

end sum_three_digit_numbers_eq_255744_l965_96519


namespace special_line_equation_l965_96560

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

end special_line_equation_l965_96560


namespace exterior_angle_hexagon_octagon_exterior_angle_hexagon_octagon_is_105_l965_96569

/-- The measure of an exterior angle formed by a regular hexagon and a regular octagon sharing a common side -/
theorem exterior_angle_hexagon_octagon : ℝ :=
  let hexagon_interior_angle := (180 * (6 - 2) / 6 : ℝ)
  let octagon_interior_angle := (180 * (8 - 2) / 8 : ℝ)
  360 - (hexagon_interior_angle + octagon_interior_angle)

/-- The exterior angle formed by a regular hexagon and a regular octagon sharing a common side is 105 degrees -/
theorem exterior_angle_hexagon_octagon_is_105 :
  exterior_angle_hexagon_octagon = 105 := by
  sorry

end exterior_angle_hexagon_octagon_exterior_angle_hexagon_octagon_is_105_l965_96569


namespace events_A_D_independent_l965_96564

structure Ball :=
  (label : Nat)

def Ω : Type := Ball × Ball

def P : Set Ω → ℝ := sorry

def A : Set Ω := {ω : Ω | ω.fst.label = 1}
def D : Set Ω := {ω : Ω | ω.fst.label + ω.snd.label = 7}

theorem events_A_D_independent :
  P (A ∩ D) = P A * P D := by sorry

end events_A_D_independent_l965_96564


namespace complex_subtraction_problem_l965_96581

theorem complex_subtraction_problem :
  (4 : ℂ) - 3*I - ((5 : ℂ) - 12*I) = -1 + 9*I := by
  sorry

end complex_subtraction_problem_l965_96581


namespace power_decomposition_l965_96565

/-- Sum of the first k odd numbers -/
def sum_odd (k : ℕ) : ℕ := k^2

/-- The nth odd number -/
def nth_odd (n : ℕ) : ℕ := 2*n - 1

theorem power_decomposition (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (n^2 = sum_odd 10) →
  (nth_odd ((m-1)^2 + 1) = 21) →
  m + n = 15 := by sorry

end power_decomposition_l965_96565


namespace expected_disease_cases_l965_96586

theorem expected_disease_cases (total_sample : ℕ) (disease_proportion : ℚ) :
  total_sample = 300 →
  disease_proportion = 1 / 4 →
  (total_sample : ℚ) * disease_proportion = 75 := by
  sorry

end expected_disease_cases_l965_96586


namespace symmetric_complex_product_l965_96538

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  z₁ = 1 + I →
  (z₁.re = z₂.re ∧ z₁.im = -z₂.im) →
  z₁ * z₂ = 2 := by
  sorry

end symmetric_complex_product_l965_96538


namespace inequality_solution_set_quadratic_inequality_solution_set_l965_96578

-- Part 1
theorem inequality_solution_set (x : ℝ) :
  (2 - x) / (x + 3) > 0 ↔ x ∈ Set.Ioo (-3 : ℝ) 2 :=
sorry

-- Part 2
theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2 : ℝ) 2 :=
sorry

end inequality_solution_set_quadratic_inequality_solution_set_l965_96578


namespace cory_fruit_order_l965_96516

def fruit_arrangement (a o b g : ℕ) : ℕ :=
  Nat.factorial 9 / (Nat.factorial a * Nat.factorial o * Nat.factorial b * Nat.factorial g)

theorem cory_fruit_order : fruit_arrangement 3 3 2 1 = 5040 := by
  sorry

end cory_fruit_order_l965_96516


namespace point_on_circle_l965_96526

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

end point_on_circle_l965_96526


namespace wax_left_after_detailing_l965_96548

/-- The amount of wax needed to detail Kellan's car in ounces -/
def car_wax : ℕ := 3

/-- The amount of wax needed to detail Kellan's SUV in ounces -/
def suv_wax : ℕ := 4

/-- The amount of wax in the bottle Kellan bought in ounces -/
def bottle_wax : ℕ := 11

/-- The amount of wax Kellan spilled in ounces -/
def spilled_wax : ℕ := 2

/-- Theorem stating the amount of wax left after detailing both vehicles -/
theorem wax_left_after_detailing : 
  bottle_wax - spilled_wax - (car_wax + suv_wax) = 2 := by
  sorry

end wax_left_after_detailing_l965_96548


namespace min_sum_squares_l965_96514

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (min : ℝ), min = 10 ∧ x^2 + y^2 + z^2 ≥ min ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^3 + y₀^3 + z₀^3 - 3*x₀*y₀*z₀ = 8 ∧ x₀^2 + y₀^2 + z₀^2 = min :=
by
  sorry

end min_sum_squares_l965_96514


namespace arithmetic_evaluation_l965_96503

theorem arithmetic_evaluation : 23 - |(-6)| - 23 = -6 := by
  sorry

end arithmetic_evaluation_l965_96503


namespace range_of_m_l965_96577

def A : Set ℝ := {x | (x - 4) / (x + 3) ≤ 0}

def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 1}

theorem range_of_m : 
  ∀ m : ℝ, (A ∩ B m = B m) ↔ m ∈ Set.Ici (-1) := by sorry

end range_of_m_l965_96577


namespace wire_cutting_l965_96541

/-- Given a wire cut into two pieces, where the shorter piece is 2/5th of the longer piece
    and is 17.14285714285714 cm long, prove that the total length of the wire before cutting is 60 cm. -/
theorem wire_cutting (shorter_piece : ℝ) (longer_piece : ℝ) :
  shorter_piece = 17.14285714285714 →
  shorter_piece = (2 / 5) * longer_piece →
  shorter_piece + longer_piece = 60 := by
  sorry

end wire_cutting_l965_96541


namespace rhombus_adjacent_sides_equal_but_not_all_parallelograms_l965_96563

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

end rhombus_adjacent_sides_equal_but_not_all_parallelograms_l965_96563


namespace intersection_circle_origin_implies_a_plusminus_one_no_symmetric_intersection_l965_96504

/-- The line equation y = ax + 1 -/
def line_equation (a x y : ℝ) : Prop := y = a * x + 1

/-- The hyperbola equation 3x^2 - y^2 = 1 -/
def hyperbola_equation (x y : ℝ) : Prop := 3 * x^2 - y^2 = 1

/-- Two points A(x₁, y₁) and B(x₂, y₂) are the intersection of the line and hyperbola -/
def intersection_points (a x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_equation a x₁ y₁ ∧ hyperbola_equation x₁ y₁ ∧
  line_equation a x₂ y₂ ∧ hyperbola_equation x₂ y₂

/-- The circle with diameter AB passes through the origin -/
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

/-- Two points are symmetric about the line y = (1/2)x -/
def symmetric_about_line (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ + y₂) / 2 = (1 / 2) * ((x₁ + x₂) / 2) ∧
  (y₁ - y₂) / (x₁ - x₂) = -2

theorem intersection_circle_origin_implies_a_plusminus_one :
  ∀ (a x₁ y₁ x₂ y₂ : ℝ),
  intersection_points a x₁ y₁ x₂ y₂ →
  circle_through_origin x₁ y₁ x₂ y₂ →
  a = 1 ∨ a = -1 :=
sorry

theorem no_symmetric_intersection :
  ¬ ∃ (a x₁ y₁ x₂ y₂ : ℝ),
  intersection_points a x₁ y₁ x₂ y₂ ∧
  symmetric_about_line x₁ y₁ x₂ y₂ :=
sorry

end intersection_circle_origin_implies_a_plusminus_one_no_symmetric_intersection_l965_96504


namespace abigail_savings_l965_96593

def monthly_savings : ℕ := 4000
def months_in_year : ℕ := 12

theorem abigail_savings : monthly_savings * months_in_year = 48000 := by
  sorry

end abigail_savings_l965_96593


namespace min_value_theorem_l965_96543

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  2 * x^2 + 24 * x + 128 / x^3 ≥ 168 ∧
  (2 * x^2 + 24 * x + 128 / x^3 = 168 ↔ x = 4) :=
by sorry

end min_value_theorem_l965_96543


namespace b_age_is_39_l965_96566

/-- Represents a person's age --/
structure Age where
  value : ℕ

/-- Represents the ages of three people A, B, and C --/
structure AgeGroup where
  a : Age
  b : Age
  c : Age

/-- Checks if the given ages satisfy the conditions of the problem --/
def satisfiesConditions (ages : AgeGroup) : Prop :=
  (ages.a.value + 10 = 2 * (ages.b.value - 10)) ∧
  (ages.a.value = ages.b.value + 9) ∧
  (ages.c.value = ages.a.value + 4)

/-- Theorem stating that if the conditions are satisfied, B's age is 39 --/
theorem b_age_is_39 (ages : AgeGroup) :
  satisfiesConditions ages → ages.b.value = 39 := by
  sorry

#check b_age_is_39

end b_age_is_39_l965_96566


namespace inequality_solution_set_l965_96501

theorem inequality_solution_set (x : ℝ) : 
  (x - 1)^2023 - 2^2023 * x^2023 ≤ x + 1 ↔ x ≥ -1 := by sorry

end inequality_solution_set_l965_96501


namespace gcd_8_factorial_6_factorial_squared_l965_96551

theorem gcd_8_factorial_6_factorial_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_8_factorial_6_factorial_squared_l965_96551


namespace time_puzzle_l965_96574

theorem time_puzzle : ∃ x : ℝ, 
  0 ≤ x ∧ x ≤ 24 ∧ 
  (x / 4) + ((24 - x) / 2) = x ∧ 
  x = 9.6 := by
sorry

end time_puzzle_l965_96574


namespace tailoring_cost_l965_96557

theorem tailoring_cost (num_shirts num_pants : ℕ) (shirt_time : ℝ) (hourly_rate : ℝ) :
  num_shirts = 10 →
  num_pants = 12 →
  shirt_time = 1.5 →
  hourly_rate = 30 →
  (num_shirts * shirt_time + num_pants * (2 * shirt_time)) * hourly_rate = 1530 := by
  sorry

end tailoring_cost_l965_96557


namespace tims_sleep_hours_l965_96521

/-- Proves that Tim slept 6 hours each day for the first 2 days given the conditions -/
theorem tims_sleep_hours (x : ℝ) : 
  (2 * x + 2 * 10 = 32) → x = 6 := by sorry

end tims_sleep_hours_l965_96521


namespace sum_of_four_variables_l965_96531

theorem sum_of_four_variables (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 250)
  (h2 : a*b + b*c + c*a + a*d + b*d + c*d = 3) :
  a + b + c + d = 16 := by
sorry

end sum_of_four_variables_l965_96531


namespace L_shape_sum_implies_all_ones_l965_96580

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


end L_shape_sum_implies_all_ones_l965_96580


namespace max_school_leaders_l965_96596

/-- Represents the number of years in a period -/
def period : ℕ := 10

/-- Represents the length of a principal's term in years -/
def principal_term : ℕ := 3

/-- Represents the length of an assistant principal's term in years -/
def assistant_principal_term : ℕ := 2

/-- Calculates the maximum number of individuals serving in a role given the period and term length -/
def max_individuals (period : ℕ) (term : ℕ) : ℕ :=
  (period + term - 1) / term

/-- Theorem stating the maximum number of principals and assistant principals over the given period -/
theorem max_school_leaders :
  max_individuals period principal_term + max_individuals period assistant_principal_term = 9 := by
  sorry

end max_school_leaders_l965_96596


namespace square_minus_product_plus_square_l965_96530

theorem square_minus_product_plus_square : 7^2 - 4*5 + 2^2 = 33 := by
  sorry

end square_minus_product_plus_square_l965_96530


namespace course_duration_l965_96500

theorem course_duration (total_hours : ℕ) (class_hours_1 : ℕ) (class_hours_2 : ℕ) (homework_hours : ℕ) :
  total_hours = 336 →
  class_hours_1 = 3 →
  class_hours_2 = 4 →
  homework_hours = 4 →
  (2 * class_hours_1 + class_hours_2 + homework_hours) * 24 = total_hours :=
by
  sorry

end course_duration_l965_96500


namespace remainder_when_divided_by_x_plus_one_l965_96517

def q (x : ℝ) : ℝ := 2*x^4 - 3*x^3 + 4*x^2 - 5*x + 6

theorem remainder_when_divided_by_x_plus_one :
  ∃ p : ℝ → ℝ, q = fun x ↦ (x + 1) * p x + 20 :=
sorry

end remainder_when_divided_by_x_plus_one_l965_96517


namespace smallest_m_divisibility_l965_96533

theorem smallest_m_divisibility (n : ℕ) (h_odd : Odd n) :
  (∃ (m : ℕ), m > 0 ∧ ∀ (k : ℕ), k > 0 → k < m →
    ¬(262417 ∣ (529^n + k * 132^n))) ∧
  (262417 ∣ (529^n + 1 * 132^n)) := by
  sorry

end smallest_m_divisibility_l965_96533


namespace S_100_equals_10100_l965_96572

/-- The number of integers in the solution set for x^2 - x < 2nx -/
def a (n : ℕ+) : ℕ := 2 * n

/-- The sum of the first n terms of the sequence {a_n} -/
def S (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem stating that S_100 equals 10100 -/
theorem S_100_equals_10100 : S 100 = 10100 := by sorry

end S_100_equals_10100_l965_96572


namespace inequality_range_l965_96588

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2*x + 1 - a^2 < 0) ↔ 
  (a > 3 ∨ a < -3) := by
sorry

end inequality_range_l965_96588


namespace component_scrap_probability_l965_96507

/-- The probability of a component passing the first inspection -/
def p_pass_first : ℝ := 0.8

/-- The probability of a component passing the second inspection, given it failed the first -/
def p_pass_second : ℝ := 0.9

/-- The probability of a component being scrapped -/
def p_scrapped : ℝ := (1 - p_pass_first) * (1 - p_pass_second)

theorem component_scrap_probability :
  p_scrapped = 0.02 :=
sorry

end component_scrap_probability_l965_96507


namespace parabola_directrix_l965_96511

/-- The equation of a parabola -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 - 6 * x + 1

/-- The equation of the directrix -/
def directrix (y : ℝ) : Prop := y = -25/12

/-- Theorem: The directrix of the given parabola is y = -25/12 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p q : ℝ, parabola p q → (p - x)^2 + (q - y)^2 = (q - d)^2) :=
sorry

end parabola_directrix_l965_96511


namespace expression_equality_l965_96562

theorem expression_equality : 2 * Real.sqrt 3 * (3/2)^(1/3) * 12^(1/6) = 6 := by sorry

end expression_equality_l965_96562


namespace group_size_after_new_member_l965_96567

theorem group_size_after_new_member (n : ℕ) : 
  (n * 14 = n * 14) →  -- Initial average age is 14
  (n * 14 + 32 = (n + 1) * 15) →  -- New average age is 15 after adding a 32-year-old
  n = 17 := by
sorry

end group_size_after_new_member_l965_96567


namespace sum_remainder_mod_11_l965_96599

theorem sum_remainder_mod_11 : (99001 + 99002 + 99003 + 99004 + 99005 + 99006) % 11 = 5 := by
  sorry

end sum_remainder_mod_11_l965_96599


namespace peter_money_brought_l965_96592

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

end peter_money_brought_l965_96592


namespace unique_solution_l965_96549

theorem unique_solution (a m n : ℕ+) (h : Real.sqrt (a^2 - 4 * Real.sqrt 2) = Real.sqrt m - Real.sqrt n) :
  m = 8 ∧ n = 1 ∧ a = 3 := by
  sorry

end unique_solution_l965_96549


namespace intersection_equality_l965_96546

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem intersection_equality (a : ℝ) : A ∩ B a = B a → a = 0 ∨ a = -1 ∨ a = 1/3 := by
  sorry

end intersection_equality_l965_96546


namespace condition_analysis_l965_96591

theorem condition_analysis (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end condition_analysis_l965_96591


namespace shells_weight_calculation_l965_96528

/-- Given an initial weight of shells and an additional weight of shells,
    calculate the total weight of shells. -/
def total_weight (initial_weight additional_weight : ℕ) : ℕ :=
  initial_weight + additional_weight

/-- Theorem: The total weight of shells is 17 pounds when
    the initial weight is 5 pounds and the additional weight is 12 pounds. -/
theorem shells_weight_calculation :
  total_weight 5 12 = 17 := by
  sorry

end shells_weight_calculation_l965_96528


namespace range_of_a_l965_96520

-- Define the conditions
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- Define the set A (solution set for p)
def A : Set ℝ := {x | p x}

-- Define the set B (solution set for q)
def B (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (A ⊆ B a) ∧ (A ≠ B a)) →
  (∃ a_min a_max : ℝ, a_min = 0 ∧ a_max = 1/2 ∧ ∀ a : ℝ, a_min ≤ a ∧ a ≤ a_max) :=
sorry

end range_of_a_l965_96520


namespace stating_number_of_people_in_first_group_l965_96518

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

end stating_number_of_people_in_first_group_l965_96518


namespace largest_value_l965_96506

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > max (1/2) (max (a^2 + b^2) (2*a*b)) := by
  sorry

end largest_value_l965_96506


namespace jacket_price_restoration_l965_96559

theorem jacket_price_restoration (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_first_reduction := initial_price * (1 - 0.25)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.15)
  let required_increase := (initial_price / price_after_second_reduction) - 1
  abs (required_increase - 0.5686) < 0.0001 := by
  sorry

end jacket_price_restoration_l965_96559


namespace initial_average_weight_l965_96542

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

end initial_average_weight_l965_96542


namespace percentage_of_sikh_boys_l965_96568

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

end percentage_of_sikh_boys_l965_96568


namespace todd_initial_gum_l965_96532

/-- Todd's initial amount of gum -/
def initial_gum : ℕ := sorry

/-- Amount of gum Steve gave Todd -/
def steve_gum : ℕ := 16

/-- Todd's final amount of gum -/
def final_gum : ℕ := 54

/-- Theorem stating that Todd's initial amount of gum is 38 pieces -/
theorem todd_initial_gum : initial_gum = 54 - 16 := by sorry

end todd_initial_gum_l965_96532


namespace negative_expression_l965_96529

theorem negative_expression : 
  (|(-1)| - |(-7)| < 0) ∧ 
  (|(-7)| + |(-1)| ≥ 0) ∧ 
  (|(-7)| - (-1) ≥ 0) ∧ 
  (|(-1)| - (-7) ≥ 0) := by
  sorry

end negative_expression_l965_96529


namespace existence_of_integers_l965_96575

theorem existence_of_integers : ∃ (m n p q : ℕ+), 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q ∧
  (m : ℝ) + n = p + q ∧
  Real.sqrt (m : ℝ) + (n : ℝ) ^ (1/4) = Real.sqrt (p : ℝ) + (q : ℝ) ^ (1/3) ∧
  Real.sqrt (m : ℝ) + (n : ℝ) ^ (1/4) > 2004 := by
  sorry

end existence_of_integers_l965_96575


namespace cherry_pie_degrees_l965_96576

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

end cherry_pie_degrees_l965_96576


namespace circle_area_through_points_l965_96508

/-- The area of a circle with center P(5, -2) passing through point Q(-7, 6) is 208π -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (5, -2)
  let Q : ℝ × ℝ := (-7, 6)
  let r := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  π * r^2 = 208 * π := by sorry

end circle_area_through_points_l965_96508


namespace water_volume_is_16_l965_96522

/-- Represents a cubical water tank -/
structure CubicalTank where
  side_length : ℝ
  water_level : ℝ
  capacity_ratio : ℝ

/-- Calculates the volume of water in a cubical tank -/
def water_volume (tank : CubicalTank) : ℝ :=
  tank.water_level * tank.side_length * tank.side_length

/-- Theorem: The volume of water in the specified cubical tank is 16 cubic feet -/
theorem water_volume_is_16 (tank : CubicalTank) 
  (h1 : tank.water_level = 1)
  (h2 : tank.capacity_ratio = 0.25)
  (h3 : tank.water_level = tank.capacity_ratio * tank.side_length) :
  water_volume tank = 16 := by
  sorry

end water_volume_is_16_l965_96522


namespace replaced_person_age_l965_96595

/-- Given a group of 10 persons, if replacing one person with a 16-year-old
    decreases the average age by 3 years, then the replaced person was 46 years old. -/
theorem replaced_person_age (group_size : ℕ) (avg_decrease : ℝ) (new_person_age : ℕ) :
  group_size = 10 →
  avg_decrease = 3 →
  new_person_age = 16 →
  (group_size : ℝ) * avg_decrease + new_person_age = 46 :=
by sorry

end replaced_person_age_l965_96595


namespace sequence_condition_l965_96561

/-- A sequence is monotonically increasing if each term is greater than the previous one. -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The general term of the sequence a_n = n^2 + bn -/
def a (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem sequence_condition (b : ℝ) :
  MonotonicallyIncreasing (a · b) → b > -3 := by
  sorry

end sequence_condition_l965_96561


namespace polynomial_factorization_l965_96534

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 := by
  sorry

end polynomial_factorization_l965_96534


namespace constant_term_expansion_l965_96515

/-- The constant term in the expansion of x(1 - 1/√x)^5 is 10 -/
theorem constant_term_expansion (x : ℝ) (x_pos : x > 0) :
  ∃ (f : ℝ → ℝ), (∀ y, y ≠ 0 → f y = y * (1 - 1 / Real.sqrt y)^5) ∧
  (∃ c, ∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - x| → |y - x| < δ → |f y - (10 + c * (y - x))| < ε * |y - x|) :=
sorry

end constant_term_expansion_l965_96515


namespace valentines_day_treats_cost_l965_96583

/-- The cost of Valentine's Day treats for two dogs -/
def total_cost (heart_biscuit_cost puppy_boots_cost : ℕ) : ℕ :=
  let dog_a_cost := 5 * heart_biscuit_cost + puppy_boots_cost
  let dog_b_cost := 7 * heart_biscuit_cost + 2 * puppy_boots_cost
  dog_a_cost + dog_b_cost

/-- Theorem stating the total cost of Valentine's Day treats for two dogs -/
theorem valentines_day_treats_cost :
  total_cost 2 15 = 69 := by
  sorry

end valentines_day_treats_cost_l965_96583


namespace divisible_by_five_l965_96582

theorem divisible_by_five (B : Nat) : B < 10 → (647 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by
  sorry

end divisible_by_five_l965_96582


namespace sign_distribution_of_products_l965_96513

theorem sign_distribution_of_products (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ((-a*b > 0 ∧ a*c < 0 ∧ b*d < 0 ∧ c*d < 0) ∨
   (-a*b < 0 ∧ a*c > 0 ∧ b*d > 0 ∧ c*d > 0) ∨
   (-a*b < 0 ∧ a*c < 0 ∧ b*d > 0 ∧ c*d > 0) ∨
   (-a*b < 0 ∧ a*c > 0 ∧ b*d < 0 ∧ c*d > 0) ∨
   (-a*b < 0 ∧ a*c > 0 ∧ b*d > 0 ∧ c*d < 0)) := by
  sorry

end sign_distribution_of_products_l965_96513


namespace horner_rule_v₃_l965_96598

/-- Horner's Rule for a polynomial of degree 6 -/
def horner_rule (a₀ a₁ a₂ a₃ a₄ a₅ a₆ x : ℝ) : ℝ :=
  ((((((a₆ * x + a₅) * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀)

/-- The third intermediate value in Horner's Rule calculation -/
def v₃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ x : ℝ) : ℝ :=
  (((a₆ * x + a₅) * x + a₄) * x + a₃)

theorem horner_rule_v₃ :
  v₃ 64 (-192) 240 (-160) 60 (-12) 1 2 = -80 :=
sorry

end horner_rule_v₃_l965_96598


namespace function_has_zero_l965_96535

theorem function_has_zero (m : ℝ) : ∃ x : ℝ, x^3 + 5*m*x - 2 = 0 := by
  sorry

end function_has_zero_l965_96535


namespace integer_as_sum_diff_squares_l965_96524

theorem integer_as_sum_diff_squares (n : ℤ) :
  ∃ (a b c : ℤ), n = a^2 + b^2 - c^2 := by
  sorry

end integer_as_sum_diff_squares_l965_96524


namespace cookies_in_box_l965_96547

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

end cookies_in_box_l965_96547


namespace factorization_proof_l965_96590

variables (a x y : ℝ)

theorem factorization_proof :
  (ax^2 - 7*a*x + 6*a = a*(x-6)*(x-1)) ∧
  (x*y^2 - 9*x = x*(y+3)*(y-3)) ∧
  (1 - x^2 + 2*x*y - y^2 = (1+x-y)*(1-x+y)) ∧
  (8*(x^2 - 2*y^2) - x*(7*x+y) + x*y = (x+4*y)*(x-4*y)) :=
by sorry

end factorization_proof_l965_96590


namespace number_difference_l965_96554

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 25220)
  (div_12 : 12 ∣ a)
  (relation : b = a / 100) : 
  a - b = 24750 := by
sorry

end number_difference_l965_96554


namespace max_single_color_coins_l965_96558

/-- Represents the state of coins -/
structure CoinState where
  red : Nat
  yellow : Nat
  blue : Nat

/-- Represents a coin exchange -/
inductive Exchange
  | RedYellowToBlue
  | RedBlueToYellow
  | YellowBlueToRed

/-- Applies an exchange to a coin state -/
def applyExchange (state : CoinState) (exchange : Exchange) : CoinState :=
  match exchange with
  | Exchange.RedYellowToBlue => 
      { red := state.red - 1, yellow := state.yellow - 1, blue := state.blue + 1 }
  | Exchange.RedBlueToYellow => 
      { red := state.red - 1, yellow := state.yellow + 1, blue := state.blue - 1 }
  | Exchange.YellowBlueToRed => 
      { red := state.red + 1, yellow := state.yellow - 1, blue := state.blue - 1 }

/-- Checks if all coins are of the same color -/
def isSingleColor (state : CoinState) : Bool :=
  (state.red = 0 && state.blue = 0) || 
  (state.red = 0 && state.yellow = 0) || 
  (state.yellow = 0 && state.blue = 0)

/-- Counts the total number of coins -/
def totalCoins (state : CoinState) : Nat :=
  state.red + state.yellow + state.blue

/-- The main theorem to prove -/
theorem max_single_color_coins :
  ∃ (finalState : CoinState) (exchanges : List Exchange), 
    let initialState := { red := 3, yellow := 4, blue := 5 : CoinState }
    finalState = exchanges.foldl applyExchange initialState ∧
    isSingleColor finalState ∧
    totalCoins finalState = 7 ∧
    finalState.yellow = 7 ∧
    ∀ (otherState : CoinState) (otherExchanges : List Exchange),
      otherState = otherExchanges.foldl applyExchange initialState →
      isSingleColor otherState →
      totalCoins otherState ≤ totalCoins finalState :=
by
  sorry


end max_single_color_coins_l965_96558


namespace interest_rate_calculation_l965_96510

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


end interest_rate_calculation_l965_96510


namespace degree_of_composed_product_l965_96579

/-- Given polynomials p and q with degrees 3 and 4 respectively,
    prove that the degree of p(x^4) * q(x^5) is 32 -/
theorem degree_of_composed_product (p q : Polynomial ℝ) 
  (hp : Polynomial.degree p = 3) (hq : Polynomial.degree q = 4) :
  Polynomial.degree (p.comp (Polynomial.X ^ 4) * q.comp (Polynomial.X ^ 5)) = 32 := by
  sorry

end degree_of_composed_product_l965_96579


namespace no_obtuse_tetrahedron_l965_96570

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

end no_obtuse_tetrahedron_l965_96570


namespace kevin_kangaroo_hops_l965_96539

/-- The sum of a geometric series with first term 1/4, common ratio 3/4, and 6 terms -/
def geometric_sum : ℚ :=
  let a : ℚ := 1/4
  let r : ℚ := 3/4
  let n : ℕ := 6
  a * (1 - r^n) / (1 - r)

/-- Theorem stating that the geometric sum equals 3367/4096 -/
theorem kevin_kangaroo_hops : geometric_sum = 3367/4096 := by
  sorry

end kevin_kangaroo_hops_l965_96539


namespace angle_from_point_l965_96523

/-- Given a point A with coordinates (sin 23°, -cos 23°) on the terminal side of angle α,
    where 0° < α < 360°, prove that α = 293°. -/
theorem angle_from_point (α : Real) : 
  0 < α ∧ α < 360 ∧ 
  (∃ (A : ℝ × ℝ), A.1 = Real.sin (23 * π / 180) ∧ A.2 = -Real.cos (23 * π / 180) ∧ 
    A.1 = Real.cos α ∧ A.2 = Real.sin α) →
  α = 293 * π / 180 := by
sorry

end angle_from_point_l965_96523


namespace intersection_complement_equals_set_l965_96505

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 4}

-- Define set N
def N : Set Nat := {1, 3, 5}

-- Theorem statement
theorem intersection_complement_equals_set :
  N ∩ (U \ M) = {3, 5} := by sorry

end intersection_complement_equals_set_l965_96505


namespace quadratic_properties_l965_96537

/-- A quadratic function with vertex at (1, -4) and axis of symmetry at x = 1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) :
  (∀ x, f a b c x = a * (x - 1)^2 - 4) →
  (2 * a + b = 0) ∧
  (f a b c (-1) = 0 ∧ f a b c 3 = 0) ∧
  (∀ m, f a b c (m - 1) < f a b c m → m > 3/2) :=
sorry

end quadratic_properties_l965_96537


namespace quadratic_function_domain_range_l965_96512

theorem quadratic_function_domain_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 m, ∃ y ∈ Set.Icc (-6) (-2), y = x^2 - 4*x - 2) ∧
  (∀ y ∈ Set.Icc (-6) (-2), ∃ x ∈ Set.Icc 0 m, y = x^2 - 4*x - 2) →
  m ∈ Set.Icc 2 4 := by
  sorry

end quadratic_function_domain_range_l965_96512


namespace noah_yearly_call_cost_l965_96502

/-- The total cost of Noah's calls to his Grammy for a year -/
def total_cost (calls_per_week : ℕ) (minutes_per_call : ℕ) (cost_per_minute : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (calls_per_week * minutes_per_call * weeks_per_year : ℕ) * cost_per_minute

/-- Theorem stating that Noah's yearly call cost to his Grammy is $78 -/
theorem noah_yearly_call_cost :
  total_cost 1 30 (5/100) 52 = 78 := by
  sorry

end noah_yearly_call_cost_l965_96502


namespace max_value_fraction_l965_96552

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2*x + y)) + (y / (x + 2*y)) ≤ 2/3 ∧ 
  ((x / (2*x + y)) + (y / (x + 2*y)) = 2/3 ↔ 2*x + y = x + 2*y) :=
by sorry

end max_value_fraction_l965_96552


namespace coefficient_value_l965_96545

-- Define the polynomial Q(x)
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + d*x - 8

-- State the theorem
theorem coefficient_value (d : ℝ) :
  (∀ x, (x + 2 : ℝ) ∣ Q d x) → d = -2 := by
  sorry

end coefficient_value_l965_96545


namespace find_number_l965_96555

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 69 :=
by sorry

end find_number_l965_96555


namespace bouncing_ball_distance_l965_96544

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

end bouncing_ball_distance_l965_96544


namespace expression_value_l965_96589

theorem expression_value : (10 : ℝ) * 0.5 * 3 / (1/6) = 90 := by
  sorry

end expression_value_l965_96589


namespace all_rules_correct_l965_96536

/-- Custom addition operation -/
def oplus (a b : ℝ) : ℝ := a + b + 1

/-- Custom subtraction operation -/
def ominus (a b : ℝ) : ℝ := a - b - 1

/-- Theorem stating the correctness of all three rules -/
theorem all_rules_correct (a b c : ℝ) : 
  (oplus a b = oplus b a) ∧ 
  (oplus a (oplus b c) = oplus (oplus a b) c) ∧ 
  (ominus a (oplus b c) = ominus (ominus a b) c) :=
sorry

end all_rules_correct_l965_96536


namespace complex_number_theorem_l965_96597

theorem complex_number_theorem (m : ℝ) : 
  let z : ℂ := m + (m^2 - 1) * Complex.I
  m = -1 := by
sorry

end complex_number_theorem_l965_96597


namespace ceiling_times_self_210_l965_96525

theorem ceiling_times_self_210 : ∃ x : ℝ, ⌈x⌉ * x = 210 ∧ x = 14 := by sorry

end ceiling_times_self_210_l965_96525


namespace microwave_sales_calculation_toaster_sales_calculation_l965_96571

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

end microwave_sales_calculation_toaster_sales_calculation_l965_96571


namespace power_seven_150_mod_12_l965_96527

theorem power_seven_150_mod_12 : 7^150 ≡ 1 [ZMOD 12] := by sorry

end power_seven_150_mod_12_l965_96527


namespace sick_animals_count_l965_96587

/-- The number of chickens at Stacy's farm -/
def num_chickens : ℕ := 26

/-- The number of piglets at Stacy's farm -/
def num_piglets : ℕ := 40

/-- The number of goats at Stacy's farm -/
def num_goats : ℕ := 34

/-- The fraction of animals that get sick -/
def sick_fraction : ℚ := 1/2

/-- The total number of sick animals -/
def total_sick_animals : ℕ := (num_chickens + num_piglets + num_goats) / 2

theorem sick_animals_count : total_sick_animals = 50 := by
  sorry

end sick_animals_count_l965_96587


namespace expression_value_l965_96573

theorem expression_value (a b : ℝ) (h : a - 2*b = 3) : 2*a - 4*b - 5 = 1 := by
  sorry

end expression_value_l965_96573


namespace hotel_breakfast_probability_l965_96553

def num_guests : ℕ := 3
def num_roll_types : ℕ := 4
def total_rolls : ℕ := 12
def rolls_per_guest : ℕ := 4

def probability_one_of_each : ℚ :=
  (9 : ℚ) / 55 * (8 : ℚ) / 35 * (1 : ℚ) / 1

theorem hotel_breakfast_probability :
  probability_one_of_each = (72 : ℚ) / 1925 :=
by sorry

end hotel_breakfast_probability_l965_96553


namespace whack_a_mole_tickets_value_l965_96585

/-- The number of tickets Ned won playing 'skee ball' -/
def skee_ball_tickets : ℕ := 19

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 9

/-- The number of candies Ned could buy -/
def candies_bought : ℕ := 5

/-- The number of tickets Ned won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := candy_cost * candies_bought - skee_ball_tickets

theorem whack_a_mole_tickets_value : whack_a_mole_tickets = 26 := by
  sorry

end whack_a_mole_tickets_value_l965_96585


namespace percentage_of_non_roses_l965_96594

theorem percentage_of_non_roses (roses tulips daisies : ℕ) : 
  roses = 25 → tulips = 40 → daisies = 35 → 
  (tulips + daisies : ℚ) / (roses + tulips + daisies) * 100 = 75 := by
  sorry

end percentage_of_non_roses_l965_96594


namespace congruence_problem_l965_96584

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 33 [ZMOD 60]) (h2 : b ≡ 85 [ZMOD 60]) :
  ∃! n : ℤ, 200 ≤ n ∧ n ≤ 251 ∧ a - b ≡ n [ZMOD 60] ∧ n = 248 := by
  sorry

end congruence_problem_l965_96584
