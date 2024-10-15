import Mathlib

namespace NUMINAMATH_CALUDE_four_digit_divisible_by_fourteen_l4074_407473

theorem four_digit_divisible_by_fourteen (n : Nat) : 
  n < 10 ∧ 945 * n < 10000 ∧ 945 * n ≥ 1000 ∧ (945 * n) % 14 = 0 → n = 8 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_fourteen_l4074_407473


namespace NUMINAMATH_CALUDE_common_tangent_sum_l4074_407481

-- Define the parabolas
def Q₁ (x y : ℝ) : Prop := y = x^2 + 53/50
def Q₂ (x y : ℝ) : Prop := x = y^2 + 91/8

-- Define the common tangent line
def M (p q r : ℕ) (x y : ℝ) : Prop := p * x + q * y = r

-- Main theorem
theorem common_tangent_sum (p q r : ℕ) :
  (p > 0) →
  (q > 0) →
  (r > 0) →
  (Nat.gcd p q = 1) →
  (Nat.gcd p r = 1) →
  (Nat.gcd q r = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    Q₁ x₁ y₁ ∧ 
    Q₂ x₂ y₂ ∧ 
    M p q r x₁ y₁ ∧ 
    M p q r x₂ y₂ ∧
    (∃ (m : ℚ), q = m * p)) →
  p + q + r = 9 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l4074_407481


namespace NUMINAMATH_CALUDE_remainder_equivalence_l4074_407458

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem: The remainder when dividing a number by 3 or 9 is the same as
    the remainder when dividing the sum of its digits by 3 or 9 -/
theorem remainder_equivalence (n : ℕ) :
  (n % 3 = sum_of_digits n % 3) ∧ (n % 9 = sum_of_digits n % 9) :=
sorry

end NUMINAMATH_CALUDE_remainder_equivalence_l4074_407458


namespace NUMINAMATH_CALUDE_counterexample_exists_l4074_407496

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 3)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4074_407496


namespace NUMINAMATH_CALUDE_day_of_week_theorem_l4074_407446

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  number : ℕ

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek := sorry

/-- Checks if a year is a leap year -/
def isLeapYear (y : Year) : Bool := sorry

/-- The main theorem to prove -/
theorem day_of_week_theorem (N : Year) :
  dayOfWeek N 250 = DayOfWeek.Wednesday →
  dayOfWeek (Year.mk (N.number + 1)) 150 = DayOfWeek.Wednesday →
  dayOfWeek (Year.mk (N.number - 1)) 50 = DayOfWeek.Monday := by
  sorry

end NUMINAMATH_CALUDE_day_of_week_theorem_l4074_407446


namespace NUMINAMATH_CALUDE_student_transfer_fraction_l4074_407408

theorem student_transfer_fraction (initial_students new_students final_students : ℕ) : 
  initial_students = 160 →
  new_students = 20 →
  final_students = 120 →
  (initial_students + new_students - final_students) / (initial_students + new_students) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_student_transfer_fraction_l4074_407408


namespace NUMINAMATH_CALUDE_inverse_of_A_l4074_407493

def A : Matrix (Fin 2) (Fin 2) ℚ := !![5, -3; 3, -2]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![2, -3; 3, -5]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l4074_407493


namespace NUMINAMATH_CALUDE_ice_cream_sundae_combinations_l4074_407421

/-- The number of different kinds of ice cream. -/
def n : ℕ := 8

/-- The number of scoops in a sundae. -/
def k : ℕ := 2

/-- The number of unique two-scoop sundaes with different ice cream flavors. -/
def different_flavors : ℕ := n.choose k

/-- The number of unique two-scoop sundaes with identical ice cream flavors. -/
def identical_flavors : ℕ := n

/-- The total number of unique two-scoop sundaes. -/
def total_sundaes : ℕ := different_flavors + identical_flavors

theorem ice_cream_sundae_combinations :
  total_sundaes = 36 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundae_combinations_l4074_407421


namespace NUMINAMATH_CALUDE_nine_appears_once_l4074_407488

def multiply_987654321_by_9 : ℕ := 987654321 * 9

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  n.digits 10
    |>.filter (· = d)
    |>.length

theorem nine_appears_once :
  count_digit multiply_987654321_by_9 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_nine_appears_once_l4074_407488


namespace NUMINAMATH_CALUDE_original_earnings_before_raise_l4074_407428

theorem original_earnings_before_raise (new_earnings : ℝ) (percent_increase : ℝ) 
  (h1 : new_earnings = 80)
  (h2 : percent_increase = 60) :
  let original_earnings := new_earnings / (1 + percent_increase / 100)
  original_earnings = 50 := by
sorry

end NUMINAMATH_CALUDE_original_earnings_before_raise_l4074_407428


namespace NUMINAMATH_CALUDE_escalator_steps_l4074_407404

theorem escalator_steps (n : ℕ) : 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 6 = 5 ∧ 
  n % 7 = 6 ∧ 
  n % 20 = 19 ∧ 
  n < 1000 → 
  n = 839 :=
by sorry

end NUMINAMATH_CALUDE_escalator_steps_l4074_407404


namespace NUMINAMATH_CALUDE_train_length_l4074_407476

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 9 → speed_kmh * (5/18) * time_s = 225 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l4074_407476


namespace NUMINAMATH_CALUDE_sandy_puppies_given_to_friends_l4074_407495

/-- Given the initial number of puppies and the number of puppies left,
    calculate the number of puppies given to friends. -/
def puppies_given_to_friends (initial_puppies left_puppies : ℕ) : ℕ :=
  initial_puppies - left_puppies

/-- Theorem stating that for Sandy's specific case, the number of puppies
    given to friends is 4. -/
theorem sandy_puppies_given_to_friends :
  puppies_given_to_friends 8 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sandy_puppies_given_to_friends_l4074_407495


namespace NUMINAMATH_CALUDE_snail_track_time_equivalence_l4074_407416

theorem snail_track_time_equivalence (clockwise_time : Real) (counterclockwise_time : Real) : 
  clockwise_time = 1.5 → counterclockwise_time = 90 → clockwise_time * 60 = counterclockwise_time :=
by
  sorry

end NUMINAMATH_CALUDE_snail_track_time_equivalence_l4074_407416


namespace NUMINAMATH_CALUDE_pizza_class_size_l4074_407440

/-- Proves that the number of students in a class is 68, given the pizza ordering scenario. -/
theorem pizza_class_size :
  let pizza_slices : ℕ := 18  -- Number of slices in a large pizza
  let total_pizzas : ℕ := 6   -- Total number of pizzas ordered
  let cheese_leftover : ℕ := 8  -- Number of cheese slices leftover
  let onion_leftover : ℕ := 4   -- Number of onion slices leftover
  let cheese_per_student : ℕ := 2  -- Number of cheese slices per student
  let onion_per_student : ℕ := 1   -- Number of onion slices per student

  let total_slices : ℕ := pizza_slices * total_pizzas
  let used_cheese : ℕ := total_slices - cheese_leftover
  let used_onion : ℕ := total_slices - onion_leftover

  (∃ (num_students : ℕ),
    num_students * cheese_per_student = used_cheese ∧
    num_students * onion_per_student = used_onion) →
  (∃! (num_students : ℕ), num_students = 68) :=
by sorry

end NUMINAMATH_CALUDE_pizza_class_size_l4074_407440


namespace NUMINAMATH_CALUDE_four_intersection_points_range_l4074_407454

/-- Parabola C: x^2 = 4y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

/-- Circle M: x^2 + (y-4)^2 = r^2 -/
def circle_M (x y r : ℝ) : Prop := x^2 + (y-4)^2 = r^2

/-- The number of intersection points between C and M -/
noncomputable def intersection_count (r : ℝ) : ℕ := sorry

theorem four_intersection_points_range (r : ℝ) :
  r > 0 ∧ intersection_count r = 4 → 2 * Real.sqrt 3 < r ∧ r < 4 := by sorry

end NUMINAMATH_CALUDE_four_intersection_points_range_l4074_407454


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_75_plus_6_l4074_407430

-- Define the function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the function to get the units digit of 7^n
def unitsDigitOf7Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case should never occur

-- Theorem statement
theorem units_digit_of_7_power_75_plus_6 :
  unitsDigit (unitsDigitOf7Power 75 + 6) = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_75_plus_6_l4074_407430


namespace NUMINAMATH_CALUDE_original_average_proof_l4074_407489

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℝ) : 
  n = 30 → 
  new_avg = 90 → 
  new_avg = 2 * original_avg → 
  original_avg = 45 := by
sorry

end NUMINAMATH_CALUDE_original_average_proof_l4074_407489


namespace NUMINAMATH_CALUDE_bus_profit_properties_l4074_407442

/-- Represents the daily profit of a bus given the number of passengers -/
def daily_profit (x : ℕ) : ℤ :=
  2 * x - 600

theorem bus_profit_properties :
  let min_passengers_no_loss := 300
  let profit_500_passengers := daily_profit 500
  let relationship (x : ℕ) := daily_profit x = 2 * x - 600
  (∀ x : ℕ, x ≥ min_passengers_no_loss → daily_profit x ≥ 0) ∧
  (profit_500_passengers = 400) ∧
  (∀ x : ℕ, relationship x) :=
by sorry

end NUMINAMATH_CALUDE_bus_profit_properties_l4074_407442


namespace NUMINAMATH_CALUDE_flag_problem_l4074_407406

theorem flag_problem (x : ℝ) : 
  (8 * 5 : ℝ) + (x * 7) + (5 * 5) = 15 * 9 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_flag_problem_l4074_407406


namespace NUMINAMATH_CALUDE_box_value_proof_l4074_407455

theorem box_value_proof : ∃ x : ℝ, (1 + 1.1 + 1.11 + x = 4.44) ∧ (x = 1.23) := by
  sorry

end NUMINAMATH_CALUDE_box_value_proof_l4074_407455


namespace NUMINAMATH_CALUDE_evaluate_expression_l4074_407462

theorem evaluate_expression : 2003^3 - 2002 * 2003^2 - 2002^2 * 2003 + 2002^3 = 4005 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4074_407462


namespace NUMINAMATH_CALUDE_distinct_positive_numbers_properties_l4074_407467

theorem distinct_positive_numbers_properties (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  (a > b ∨ a < b ∨ a = b) ∧ 
  (a ≠ c ∧ b ≠ c ∧ a ≠ b) := by
  sorry

end NUMINAMATH_CALUDE_distinct_positive_numbers_properties_l4074_407467


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4074_407410

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Define the intersection set
def intersection_set : Set (ℝ × ℝ) := {(0, 0), (1, 1)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4074_407410


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l4074_407429

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_negative_2023 :
  reciprocal (-2023) = -1/2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l4074_407429


namespace NUMINAMATH_CALUDE_bake_four_pans_l4074_407499

/-- The number of pans of cookies that can be baked in a given time -/
def pans_of_cookies (total_time minutes_per_pan : ℕ) : ℕ :=
  total_time / minutes_per_pan

/-- Proof that 4 pans of cookies can be baked in 28 minutes when each pan takes 7 minutes -/
theorem bake_four_pans : pans_of_cookies 28 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bake_four_pans_l4074_407499


namespace NUMINAMATH_CALUDE_cube_inequality_iff_l4074_407453

theorem cube_inequality_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_iff_l4074_407453


namespace NUMINAMATH_CALUDE_modified_square_boundary_length_l4074_407468

/-- The boundary length of a modified square figure --/
theorem modified_square_boundary_length :
  ∀ (square_area : ℝ) (num_segments : ℕ),
    square_area = 100 →
    num_segments = 4 →
    ∃ (boundary_length : ℝ),
      boundary_length = 5 * Real.pi + 10 := by
  sorry

end NUMINAMATH_CALUDE_modified_square_boundary_length_l4074_407468


namespace NUMINAMATH_CALUDE_jenny_wedding_budget_l4074_407456

/-- Calculates the total catering budget for a wedding --/
def totalCateringBudget (totalGuests : ℕ) (steakMultiplier : ℕ) (steakCost chickenCost : ℚ) : ℚ :=
  let chickenGuests := totalGuests / (steakMultiplier + 1)
  let steakGuests := totalGuests - chickenGuests
  steakGuests * steakCost + chickenGuests * chickenCost

/-- Proves that the total catering budget for Jenny's wedding is $1860 --/
theorem jenny_wedding_budget :
  totalCateringBudget 80 3 25 18 = 1860 := by
  sorry

end NUMINAMATH_CALUDE_jenny_wedding_budget_l4074_407456


namespace NUMINAMATH_CALUDE_broken_calculator_multiplication_l4074_407407

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem broken_calculator_multiplication :
  ∀ a b : ℕ, is_two_digit a → is_two_digit b →
  (a * b = 1001 ∨ a * b = 1100) ↔ 
  ((a = 11 ∧ b = 91) ∨ (a = 91 ∧ b = 11) ∨
   (a = 13 ∧ b = 77) ∨ (a = 77 ∧ b = 13) ∨
   (a = 25 ∧ b = 44) ∨ (a = 44 ∧ b = 25)) :=
by sorry

end NUMINAMATH_CALUDE_broken_calculator_multiplication_l4074_407407


namespace NUMINAMATH_CALUDE_sum_of_integers_30_to_50_l4074_407443

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_of_integers_30_to_50 (x y : ℕ) :
  x = sum_of_integers 30 50 →
  y = count_even_integers 30 50 →
  x + y = 851 →
  x = 840 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_30_to_50_l4074_407443


namespace NUMINAMATH_CALUDE_gaochun_temperature_difference_l4074_407425

def temperature_difference (low high : Int) : Int :=
  high - low

theorem gaochun_temperature_difference :
  let low : Int := -2
  let high : Int := 9
  temperature_difference low high = 11 := by
  sorry

end NUMINAMATH_CALUDE_gaochun_temperature_difference_l4074_407425


namespace NUMINAMATH_CALUDE_right_angle_points_iff_h_squared_leq_ac_l4074_407414

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  a : ℝ  -- length of one base
  c : ℝ  -- length of the other base
  h : ℝ  -- altitude
  a_pos : 0 < a  -- a is positive
  c_pos : 0 < c  -- c is positive
  h_pos : 0 < h  -- h is positive

/-- The existence of points P on the axis of symmetry where both legs subtend right angles -/
def exists_right_angle_points (t : IsoscelesTrapezoid) : Prop :=
  ∃ P : ℝ × ℝ, P.1 = 0 ∧ 0 ≤ P.2 ∧ P.2 ≤ t.h ∧ 
    (P.2^2 + (t.a/2)^2 = (P.2^2 + ((t.a - t.c)/2)^2 + t.h^2)/4) ∧
    ((t.h - P.2)^2 + (t.c/2)^2 = (P.2^2 + ((t.a - t.c)/2)^2 + t.h^2)/4)

/-- The main theorem -/
theorem right_angle_points_iff_h_squared_leq_ac (t : IsoscelesTrapezoid) :
  exists_right_angle_points t ↔ t.h^2 ≤ t.a * t.c :=
sorry

end NUMINAMATH_CALUDE_right_angle_points_iff_h_squared_leq_ac_l4074_407414


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4074_407466

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4074_407466


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l4074_407444

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l: x = my + 4 -/
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 4

/-- Point on parabola C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : parabola_C x y

/-- Points M and N are perpendicular from origin -/
def perpendicular_from_origin (M N : PointOnC) : Prop :=
  M.x * N.x + M.y * N.y = 0

theorem line_passes_through_fixed_point (m : ℝ) 
  (M N : PointOnC) (h_distinct : M ≠ N) 
  (h_on_l : line_l m M.x M.y ∧ line_l m N.x N.y)
  (h_perp : perpendicular_from_origin M N) :
  line_l m 4 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l4074_407444


namespace NUMINAMATH_CALUDE_exists_unprovable_by_induction_l4074_407448

-- Define a proposition that represents a mathematical statement
def MathStatement : Type := Prop

-- Define a function that represents the ability to prove a statement by induction
def ProvableByInduction (s : MathStatement) : Prop := sorry

-- Theorem: There exists a true mathematical statement that cannot be proven by induction
theorem exists_unprovable_by_induction : 
  ∃ (s : MathStatement), s ∧ ¬(ProvableByInduction s) := by sorry

end NUMINAMATH_CALUDE_exists_unprovable_by_induction_l4074_407448


namespace NUMINAMATH_CALUDE_integral_even_function_l4074_407474

/-- Given that f(x) = ax^2 + (a-2)x + a^2 is an even function, 
    prove that the integral of (x^2 + x + √(4 - x^2)) from -a to a equals 16/3 + 2π -/
theorem integral_even_function (a : ℝ) 
  (h : ∀ x, a*x^2 + (a-2)*x + a^2 = a*(-x)^2 + (a-2)*(-x) + a^2) :
  ∫ x in (-a)..a, (x^2 + x + Real.sqrt (4 - x^2)) = 16/3 + 2*Real.pi := by
  sorry

end NUMINAMATH_CALUDE_integral_even_function_l4074_407474


namespace NUMINAMATH_CALUDE_min_dot_product_of_tangents_l4074_407417

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_M (x y θ : ℝ) : Prop := (x - 5 * Real.cos θ)^2 + (y - 5 * Real.sin θ)^2 = 1

-- Define a point on circle M
def point_on_M (P : ℝ × ℝ) (θ : ℝ) : Prop := circle_M P.1 P.2 θ

-- Define tangent lines from P to circle O
def tangent_to_O (P E : ℝ × ℝ) : Prop := 
  circle_O E.1 E.2 ∧ (P.1 - E.1) * E.1 + (P.2 - E.2) * E.2 = 0

-- Statement of the theorem
theorem min_dot_product_of_tangents :
  ∀ (P : ℝ × ℝ) (θ : ℝ),
  point_on_M P θ →
  ∃ (E F : ℝ × ℝ),
  tangent_to_O P E ∧ tangent_to_O P F →
  (∀ (E' F' : ℝ × ℝ), tangent_to_O P E' ∧ tangent_to_O P F' →
    ((P.1 - E.1) * (P.1 - F.1) + (P.2 - E.2) * (P.2 - F.2)) ≤
    ((P.1 - E'.1) * (P.1 - F'.1) + (P.2 - E'.2) * (P.2 - F'.2))) ∧
  ((P.1 - E.1) * (P.1 - F.1) + (P.2 - E.2) * (P.2 - F.2)) = 6 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_of_tangents_l4074_407417


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4074_407413

theorem quadratic_two_distinct_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁^2 - 4*x₁ + 2 = 0 ∧ x₂^2 - 4*x₂ + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4074_407413


namespace NUMINAMATH_CALUDE_constant_term_is_70_l4074_407470

/-- 
Given a natural number n, this function represents the coefficient 
of the r-th term in the expansion of (x + 1/x)^(2n)
-/
def binomialCoeff (n : ℕ) (r : ℕ) : ℕ := Nat.choose (2 * n) r

/-- 
This theorem states that if the coefficients of the fourth and sixth terms 
in the expansion of (x + 1/x)^(2n) are equal, then the constant term 
in the expansion is 70
-/
theorem constant_term_is_70 (n : ℕ) 
  (h : binomialCoeff n 3 = binomialCoeff n 5) : 
  binomialCoeff n 4 = 70 := by
  sorry


end NUMINAMATH_CALUDE_constant_term_is_70_l4074_407470


namespace NUMINAMATH_CALUDE_a_51_value_l4074_407450

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) - a n = 2

theorem a_51_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 51 = 101 := by
  sorry

end NUMINAMATH_CALUDE_a_51_value_l4074_407450


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l4074_407459

-- Define the quadrant type
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define the function to determine the quadrant of an angle
def angle_quadrant (θ : Real) : Quadrant :=
  sorry

-- Theorem statement
theorem angle_in_second_quadrant (θ : Real) 
  (h1 : Real.sin θ > Real.cos θ) 
  (h2 : Real.tan θ < 0) : 
  angle_quadrant θ = Quadrant.second :=
sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l4074_407459


namespace NUMINAMATH_CALUDE_total_marbles_l4074_407490

theorem total_marbles (x : ℝ) : (4*x + 2) + 2*x + (3*x + 1) = 9*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l4074_407490


namespace NUMINAMATH_CALUDE_common_tangent_parabola_log_l4074_407401

theorem common_tangent_parabola_log (a : ℝ) : 
  (∃ x₁ x₂ y : ℝ, 
    y = a * x₁^2 ∧ 
    y = Real.log x₂ ∧ 
    2 * a * x₁ = 2 ∧ 
    1 / x₂ = 2) → 
  a = 1 / Real.log (2 * Real.exp 1) := by
sorry

end NUMINAMATH_CALUDE_common_tangent_parabola_log_l4074_407401


namespace NUMINAMATH_CALUDE_function_transformation_l4074_407405

theorem function_transformation (f : ℝ → ℝ) : 
  (∀ x, f (2 * x - 1) = x^2 - x) → 
  (∀ x, f x = (1/4) * (x^2 - 1)) := by
sorry

end NUMINAMATH_CALUDE_function_transformation_l4074_407405


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l4074_407463

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ

/-- The y-intercept of the tangent line to two circles -/
def yIntercept (line : TangentLine) : ℝ := sorry

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (7, 0), radius := 2 }
  let line : TangentLine := {
    circle1 := c1,
    circle2 := c2,
    tangentPoint1 := sorry,  -- Exact point not given, but in first quadrant
    tangentPoint2 := sorry   -- Exact point not given, but in first quadrant
  }
  yIntercept line = 4.5 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l4074_407463


namespace NUMINAMATH_CALUDE_student_number_problem_l4074_407434

theorem student_number_problem (x : ℤ) : x = 60 ↔ 4 * x - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l4074_407434


namespace NUMINAMATH_CALUDE_product_of_integers_l4074_407497

theorem product_of_integers (a b : ℕ+) 
  (h1 : (a : ℚ) / (b : ℚ) = 12)
  (h2 : a + b = 144) :
  (a : ℚ) * (b : ℚ) = 248832 / 169 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l4074_407497


namespace NUMINAMATH_CALUDE_fraction_subtraction_theorem_l4074_407426

theorem fraction_subtraction_theorem : 
  (3 + 6 + 9) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 6 + 9) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_theorem_l4074_407426


namespace NUMINAMATH_CALUDE_horse_cow_pricing_system_l4074_407451

theorem horse_cow_pricing_system (x y : ℝ) :
  (4 * x + 6 * y = 48 ∧ 3 * x + 5 * y = 38) ↔
  (∃ (horse_price cow_price : ℝ),
    horse_price = x ∧
    cow_price = y ∧
    4 * horse_price + 6 * cow_price = 48 ∧
    3 * horse_price + 5 * cow_price = 38) :=
by sorry

end NUMINAMATH_CALUDE_horse_cow_pricing_system_l4074_407451


namespace NUMINAMATH_CALUDE_middle_term_of_five_term_arithmetic_sequence_l4074_407487

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem middle_term_of_five_term_arithmetic_sequence 
  (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) 
  (h_first : a 1 = 21) (h_last : a 5 = 53) : 
  a 3 = 37 := by
sorry

end NUMINAMATH_CALUDE_middle_term_of_five_term_arithmetic_sequence_l4074_407487


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l4074_407441

def repeating_decimal_to_fraction (d : ℚ) : ℚ := d

theorem repeating_decimal_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : Nat.gcd a b = 1) (h4 : repeating_decimal_to_fraction (35/99 : ℚ) = a / b) : 
  a + b = 134 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l4074_407441


namespace NUMINAMATH_CALUDE_water_fraction_in_mixture_l4074_407424

theorem water_fraction_in_mixture (alcohol_to_water_ratio : ℚ) :
  alcohol_to_water_ratio = 2/3 →
  (water_volume / (water_volume + alcohol_volume) = 3/5) :=
by
  sorry

end NUMINAMATH_CALUDE_water_fraction_in_mixture_l4074_407424


namespace NUMINAMATH_CALUDE_prob_A_wins_one_round_prob_at_least_one_wins_l4074_407415

/-- Probability of A winning exactly one round in a two-round competition -/
theorem prob_A_wins_one_round 
  (p_A1 : ℚ) -- Probability of A winning first round
  (p_A2 : ℚ) -- Probability of A winning second round
  (h_p_A1 : p_A1 = 4/5)
  (h_p_A2 : p_A2 = 2/3) :
  p_A1 * (1 - p_A2) + (1 - p_A1) * p_A2 = 2/5 := by sorry

/-- Probability of at least one of A and B winning a two-round competition -/
theorem prob_at_least_one_wins
  (p_A1 p_A2 p_B1 p_B2 : ℚ) -- Probabilities of A and B winning each round
  (h_p_A1 : p_A1 = 4/5)
  (h_p_A2 : p_A2 = 2/3)
  (h_p_B1 : p_B1 = 3/5)
  (h_p_B2 : p_B2 = 3/4) :
  1 - (1 - p_A1 * p_A2) * (1 - p_B1 * p_B2) = 223/300 := by sorry

end NUMINAMATH_CALUDE_prob_A_wins_one_round_prob_at_least_one_wins_l4074_407415


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l4074_407433

def M : Set ℝ := {x | x^2 + 2*x - 3 = 0}
def N : Set ℝ := {-1, 2, 3}

theorem union_of_M_and_N : M ∪ N = {-1, 1, 2, -3, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l4074_407433


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l4074_407422

theorem quadratic_completing_square (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l4074_407422


namespace NUMINAMATH_CALUDE_tShirts_per_package_example_l4074_407480

/-- Given a total number of white t-shirts and a number of packages,
    calculate the number of t-shirts per package. -/
def tShirtsPerPackage (total : ℕ) (packages : ℕ) : ℕ :=
  total / packages

/-- Theorem: Given 70 white t-shirts in 14 packages,
    prove that each package contains 5 t-shirts. -/
theorem tShirts_per_package_example :
  tShirtsPerPackage 70 14 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tShirts_per_package_example_l4074_407480


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l4074_407464

theorem sum_of_squares_and_products (a b c : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → 
  a^2 + b^2 + c^2 = 48 → 
  a*b + b*c + c*a = 24 → 
  a + b + c = 4 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l4074_407464


namespace NUMINAMATH_CALUDE_roots_sum_zero_l4074_407431

/-- Given two quadratic trinomials with specific properties, prove their product's roots sum to 0 -/
theorem roots_sum_zero (a b : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₃ ≠ x₄ ∧ 
    (∀ x : ℝ, x^2 + a*x + b = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    (∀ x : ℝ, x^2 + b*x + a = 0 ↔ (x = x₃ ∨ x = x₄))) →
  (∃ y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₂ ≠ y₃ ∧
    (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + b*x + a) = 0 ↔ (x = y₁ ∨ x = y₂ ∨ x = y₃))) →
  (∃ z₁ z₂ z₃ : ℝ, 
    (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + b*x + a) = 0 ↔ (x = z₁ ∨ x = z₂ ∨ x = z₃)) ∧
    z₁ + z₂ + z₃ = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_zero_l4074_407431


namespace NUMINAMATH_CALUDE_bill_selling_price_l4074_407439

theorem bill_selling_price (P : ℝ) 
  (h1 : P + 0.1 * P = 1.1 * P)  -- Original selling price
  (h2 : 0.9 * P + 0.3 * (0.9 * P) = 1.17 * P)  -- New selling price with 30% profit
  (h3 : 1.17 * P = 1.1 * P + 42)  -- Equation relating the two selling prices
  : 1.1 * P = 660 := by
  sorry

end NUMINAMATH_CALUDE_bill_selling_price_l4074_407439


namespace NUMINAMATH_CALUDE_paint_cube_cost_l4074_407403

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem paint_cube_cost
  (paint_cost_per_kg : ℝ)
  (paint_coverage_per_kg : ℝ)
  (cube_side_length : ℝ)
  (h1 : paint_cost_per_kg = 60)
  (h2 : paint_coverage_per_kg = 20)
  (h3 : cube_side_length = 10) :
  cube_side_length ^ 2 * 6 / paint_coverage_per_kg * paint_cost_per_kg = 1800 :=
by sorry

end NUMINAMATH_CALUDE_paint_cube_cost_l4074_407403


namespace NUMINAMATH_CALUDE_original_speed_theorem_l4074_407460

def distance : ℝ := 160
def speed_increase : ℝ := 0.25
def time_saved : ℝ := 0.4

theorem original_speed_theorem (original_speed : ℝ) 
  (h1 : original_speed > 0) 
  (h2 : distance / original_speed - distance / (original_speed * (1 + speed_increase)) = time_saved) : 
  original_speed = 80 := by
sorry

end NUMINAMATH_CALUDE_original_speed_theorem_l4074_407460


namespace NUMINAMATH_CALUDE_book_cost_price_l4074_407452

theorem book_cost_price (selling_price profit_percentage : ℝ)
  (h1 : profit_percentage = 0.10)
  (h2 : selling_price = (1 + profit_percentage) * 2800)
  (h3 : selling_price + 140 = (1 + 0.15) * 2800) :
  2800 = (selling_price - (1 + profit_percentage) * 2800) / profit_percentage :=
by sorry

end NUMINAMATH_CALUDE_book_cost_price_l4074_407452


namespace NUMINAMATH_CALUDE_parabola_directrix_l4074_407483

/-- Represents a parabola in the form y = ax^2 -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ := fun x => a * x^2

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  fun y => ∃ k, y = -k ∧ p.a = 1 / (4 * k)

theorem parabola_directrix (p : Parabola) (h : p.a = 1/4) :
  directrix p = fun y => y = -1 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4074_407483


namespace NUMINAMATH_CALUDE_cereal_serving_size_l4074_407475

/-- Represents the number of cups of cereal in a box -/
def total_cups : ℕ := 18

/-- Represents the number of servings in a box -/
def total_servings : ℕ := 9

/-- Represents the number of cups per serving -/
def cups_per_serving : ℚ := total_cups / total_servings

theorem cereal_serving_size : cups_per_serving = 2 := by
  sorry

end NUMINAMATH_CALUDE_cereal_serving_size_l4074_407475


namespace NUMINAMATH_CALUDE_video_votes_theorem_l4074_407494

theorem video_votes_theorem (total_votes : ℕ) (score : ℤ) (like_percentage : ℚ) : 
  like_percentage = 3/4 ∧ 
  score = 120 ∧ 
  (like_percentage * total_votes : ℚ).num * 2 - total_votes = score → 
  total_votes = 240 := by
sorry

end NUMINAMATH_CALUDE_video_votes_theorem_l4074_407494


namespace NUMINAMATH_CALUDE_milk_for_six_cookies_l4074_407418

/-- The number of cookies that can be baked with 1 gallon of milk -/
def cookies_per_gallon : ℕ := 24

/-- The number of quarts in a gallon -/
def quarts_per_gallon : ℕ := 4

/-- The number of cookies we want to bake -/
def target_cookies : ℕ := 6

/-- Calculate the amount of milk in quarts needed to bake a given number of cookies -/
def milk_needed (cookies : ℕ) : ℚ :=
  (cookies : ℚ) * (quarts_per_gallon : ℚ) / (cookies_per_gallon : ℚ)

/-- Theorem: The amount of milk needed to bake 6 cookies is 1 quart -/
theorem milk_for_six_cookies :
  milk_needed target_cookies = 1 := by
  sorry

end NUMINAMATH_CALUDE_milk_for_six_cookies_l4074_407418


namespace NUMINAMATH_CALUDE_polynomial_roots_and_factorization_l4074_407492

theorem polynomial_roots_and_factorization (m : ℤ) :
  (∀ x : ℤ, 2 * x^4 + m * x^2 + 8 = 0 → (∃ y : ℤ, x = y)) →
  (m = -10 ∧
   ∀ x : ℝ, 2 * x^4 + m * x^2 + 8 = 2 * (x + 1) * (x - 1) * (x + 2) * (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_and_factorization_l4074_407492


namespace NUMINAMATH_CALUDE_remainder_problem_l4074_407484

theorem remainder_problem (x y : ℤ) 
  (h1 : x % 82 = 5)
  (h2 : (x + y^2) % 41 = 0) :
  (x + y^3 + 7) % 61 = 45 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4074_407484


namespace NUMINAMATH_CALUDE_students_per_row_l4074_407485

theorem students_per_row (S R x : ℕ) : 
  S = x * R + 6 →  -- First scenario
  S = 12 * (R - 3) →  -- Second scenario
  S = 6 * R →  -- Third condition
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_students_per_row_l4074_407485


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4074_407478

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a-1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4074_407478


namespace NUMINAMATH_CALUDE_largest_package_size_l4074_407412

theorem largest_package_size (ming catherine alex : ℕ) 
  (h_ming : ming = 36) 
  (h_catherine : catherine = 60) 
  (h_alex : alex = 48) : 
  Nat.gcd ming (Nat.gcd catherine alex) = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l4074_407412


namespace NUMINAMATH_CALUDE_skyler_song_composition_l4074_407437

/-- Represents the success levels of songs --/
inductive SuccessLevel
  | ExtremelySuccessful
  | Successful
  | ModeratelySuccessful
  | LessSuccessful
  | Unreleased

/-- Represents Skyler's song composition --/
structure SongComposition where
  hitSongs : Nat
  top100Songs : Nat
  unreleasedSongs : Nat
  duetsTop20 : Nat
  duetsBelow200 : Nat
  soundtracksExtremely : Nat
  soundtracksModerate : Nat
  soundtracksLukewarm : Nat
  internationalGlobal : Nat
  internationalRegional : Nat
  internationalOverlooked : Nat

/-- Calculates the total number of songs --/
def totalSongs (composition : SongComposition) : Nat :=
  composition.hitSongs + composition.top100Songs + composition.unreleasedSongs +
  composition.duetsTop20 + composition.duetsBelow200 +
  composition.soundtracksExtremely + composition.soundtracksModerate + composition.soundtracksLukewarm +
  composition.internationalGlobal + composition.internationalRegional + composition.internationalOverlooked

/-- Calculates the number of songs for each success level --/
def songsBySuccessLevel (composition : SongComposition) : SuccessLevel → Nat
  | SuccessLevel.ExtremelySuccessful => composition.hitSongs + composition.internationalGlobal
  | SuccessLevel.Successful => composition.top100Songs + composition.duetsTop20 + composition.soundtracksExtremely
  | SuccessLevel.ModeratelySuccessful => composition.soundtracksModerate + composition.internationalRegional
  | SuccessLevel.LessSuccessful => composition.soundtracksLukewarm + composition.internationalOverlooked + composition.duetsBelow200
  | SuccessLevel.Unreleased => composition.unreleasedSongs

/-- Theorem stating the total number of songs and their success level breakdown --/
theorem skyler_song_composition :
  ∃ (composition : SongComposition),
    composition.hitSongs = 25 ∧
    composition.top100Songs = composition.hitSongs + 10 ∧
    composition.unreleasedSongs = composition.hitSongs - 5 ∧
    composition.duetsTop20 = 6 ∧
    composition.duetsBelow200 = 6 ∧
    composition.soundtracksExtremely = 3 ∧
    composition.soundtracksModerate = 8 ∧
    composition.soundtracksLukewarm = 7 ∧
    composition.internationalGlobal = 1 ∧
    composition.internationalRegional = 7 ∧
    composition.internationalOverlooked = 14 ∧
    totalSongs composition = 132 ∧
    songsBySuccessLevel composition SuccessLevel.ExtremelySuccessful = 26 ∧
    songsBySuccessLevel composition SuccessLevel.Successful = 44 ∧
    songsBySuccessLevel composition SuccessLevel.ModeratelySuccessful = 15 ∧
    songsBySuccessLevel composition SuccessLevel.LessSuccessful = 27 ∧
    songsBySuccessLevel composition SuccessLevel.Unreleased = 20 := by
  sorry

end NUMINAMATH_CALUDE_skyler_song_composition_l4074_407437


namespace NUMINAMATH_CALUDE_more_girls_than_boys_in_class_l4074_407423

theorem more_girls_than_boys_in_class (num_students : ℕ) (num_teachers : ℕ) 
  (h_students : num_students = 42)
  (h_teachers : num_teachers = 6)
  (h_ratio : ∃ (x : ℕ), num_students = 7 * x ∧ 3 * x = num_boys ∧ 4 * x = num_girls) :
  num_girls - num_boys = 6 :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_in_class_l4074_407423


namespace NUMINAMATH_CALUDE_i_power_2016_l4074_407477

-- Define i as a complex number
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem i_power_2016 : i^2016 = 1 :=
  -- Assume the given conditions
  have h1 : i^1 = i := by sorry
  have h2 : i^2 = -1 := by sorry
  have h3 : i^3 = -i := by sorry
  have h4 : i^4 = 1 := by sorry
  have h5 : i^5 = i := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_i_power_2016_l4074_407477


namespace NUMINAMATH_CALUDE_song_guessing_game_theorem_l4074_407436

/-- The Song Guessing Game -/
structure SongGuessingGame where
  /-- Probability of correctly guessing a song from group A -/
  probA : ℝ
  /-- Probability of correctly guessing a song from group B -/
  probB : ℝ
  /-- Number of songs played from group A -/
  numA : ℕ
  /-- Number of songs played from group B -/
  numB : ℕ
  /-- Points earned for correctly guessing a song from group A -/
  pointsA : ℕ
  /-- Points earned for correctly guessing a song from group B -/
  pointsB : ℕ

/-- The probability of guessing at least 2 song titles correctly -/
def probAtLeastTwo (game : SongGuessingGame) : ℝ := sorry

/-- The expectation of the total score -/
def expectedScore (game : SongGuessingGame) : ℝ := sorry

/-- Main theorem about the Song Guessing Game -/
theorem song_guessing_game_theorem (game : SongGuessingGame) 
  (h1 : game.probA = 2/3)
  (h2 : game.probB = 1/2)
  (h3 : game.numA = 2)
  (h4 : game.numB = 2)
  (h5 : game.pointsA = 1)
  (h6 : game.pointsB = 2) :
  probAtLeastTwo game = 29/36 ∧ expectedScore game = 10/3 := by sorry

end NUMINAMATH_CALUDE_song_guessing_game_theorem_l4074_407436


namespace NUMINAMATH_CALUDE_floor_of_negative_decimal_l4074_407402

theorem floor_of_negative_decimal (x : ℝ) : x = -3.7 → ⌊x⌋ = -4 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_negative_decimal_l4074_407402


namespace NUMINAMATH_CALUDE_diagonals_from_vertex_l4074_407469

/-- For a polygon with interior angles summing to 540°, 
    the number of diagonals that can be drawn from one vertex is 2. -/
theorem diagonals_from_vertex (n : ℕ) : 
  (n - 2) * 180 = 540 → (n - 3 : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_from_vertex_l4074_407469


namespace NUMINAMATH_CALUDE_peach_count_l4074_407419

/-- Calculates the total number of peaches after picking -/
def total_peaches (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem: The total number of peaches is the sum of initial and picked peaches -/
theorem peach_count (initial picked : ℕ) :
  total_peaches initial picked = initial + picked := by
  sorry

end NUMINAMATH_CALUDE_peach_count_l4074_407419


namespace NUMINAMATH_CALUDE_tan_value_for_given_point_l4074_407486

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_value_for_given_point (θ : Real) (h : ∃ (r : Real), r * (Real.cos θ) = -Real.sqrt 3 / 2 ∧ r * (Real.sin θ) = 1 / 2) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_for_given_point_l4074_407486


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4074_407479

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (1/x₁ + 1/x₂ = -3) →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4074_407479


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l4074_407432

theorem football_team_right_handed_players (total_players : ℕ) (throwers : ℕ) :
  total_players = 70 →
  throwers = 37 →
  (total_players - throwers) % 3 = 0 →
  59 = throwers + (total_players - throwers) * 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l4074_407432


namespace NUMINAMATH_CALUDE_probability_same_group_l4074_407471

/-- The probability that two specific cards (5 and 14) are in the same group
    when drawing 4 cards from 20, where groups are determined by card value. -/
theorem probability_same_group : 
  let total_cards : ℕ := 20
  let cards_drawn : ℕ := 4
  let remaining_cards : ℕ := total_cards - cards_drawn + 2  -- +2 because 5 and 14 are known
  let favorable_outcomes : ℕ := (remaining_cards - 14 + 1) * (remaining_cards - 14) + 
                                (5 - 1) * (5 - 2)
  let total_outcomes : ℕ := remaining_cards * (remaining_cards - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 51 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_group_l4074_407471


namespace NUMINAMATH_CALUDE_pretzel_problem_l4074_407498

theorem pretzel_problem (barry_pretzels shelly_pretzels angie_pretzels : ℕ) : 
  barry_pretzels = 12 →
  shelly_pretzels = barry_pretzels / 2 →
  angie_pretzels = 3 * shelly_pretzels →
  angie_pretzels = 18 := by
  sorry

end NUMINAMATH_CALUDE_pretzel_problem_l4074_407498


namespace NUMINAMATH_CALUDE_binomial_sum_equals_120_l4074_407449

theorem binomial_sum_equals_120 : 
  Nat.choose 8 2 + Nat.choose 8 3 + Nat.choose 9 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_120_l4074_407449


namespace NUMINAMATH_CALUDE_lizzie_wins_iff_composite_l4074_407445

/-- The game state represents the numbers on the blackboard -/
def GameState := List ℚ

/-- A move in the game is selecting a subset of numbers and replacing them with their average -/
def Move := List ℕ

/-- Represents whether a number is composite -/
def IsComposite (n : ℕ) : Prop := ∃ k, 1 < k ∧ k < n ∧ k ∣ n

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if all numbers in the game state are equal -/
def allEqual (state : GameState) : Prop :=
  sorry

/-- Represents a winning strategy for Lizzie -/
def WinningStrategy (n : ℕ) : Prop :=
  ∀ initialState : GameState, 
    initialState.length = n →
    ∃ moveSequence : List Move, 
      allEqual (moveSequence.foldl applyMove initialState)

theorem lizzie_wins_iff_composite (n : ℕ) (h : n ≥ 3) :
  WinningStrategy n ↔ IsComposite n :=
sorry

end NUMINAMATH_CALUDE_lizzie_wins_iff_composite_l4074_407445


namespace NUMINAMATH_CALUDE_product_of_reciprocal_minus_one_geq_eight_l4074_407411

theorem product_of_reciprocal_minus_one_geq_eight (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reciprocal_minus_one_geq_eight_l4074_407411


namespace NUMINAMATH_CALUDE_extremum_condition_increasing_interval_two_roots_condition_l4074_407409

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + 6 * x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * a * x + 6

theorem extremum_condition (a : ℝ) : f' a 3 = 0 := by sorry

theorem increasing_interval (a : ℝ) :
  (∀ m : ℝ, (∀ x ∈ Set.Ioo m (m + 2), f' a x > 0) ↔ m ∈ Set.Iic 0 ∪ Set.Ici 3) := by sorry

theorem two_roots_condition (a : ℝ) :
  (∀ m : ℝ, (∃ x y : ℝ, x ∈ Set.Icc 1 3 ∧ y ∈ Set.Icc 1 3 ∧ x ≠ y ∧ f a x + m = 0 ∧ f a y + m = 0) ↔
  m ∈ Set.Ioo (-14/3) (-9/2)) := by sorry

end NUMINAMATH_CALUDE_extremum_condition_increasing_interval_two_roots_condition_l4074_407409


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l4074_407447

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that in a geometric sequence of positive numbers where the third term is 16 and the seventh term is 2, the fifth term is 2. -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : IsGeometricSequence a)
  (h_third_term : a 3 = 16)
  (h_seventh_term : a 7 = 2) :
  a 5 = 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l4074_407447


namespace NUMINAMATH_CALUDE_james_earnings_ratio_l4074_407465

theorem james_earnings_ratio :
  ∀ (february_earnings : ℕ),
    4000 + february_earnings + (february_earnings - 2000) = 18000 →
    february_earnings / 4000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_earnings_ratio_l4074_407465


namespace NUMINAMATH_CALUDE_total_games_in_season_l4074_407400

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  games_per_pair * (n * (n - 1) / 2)

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264 -/
theorem total_games_in_season :
  hockey_league_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_season_l4074_407400


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4074_407438

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) :
  (1 - i)^2 / z = 1 + i → z = -1 - i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4074_407438


namespace NUMINAMATH_CALUDE_perpendicular_slope_l4074_407435

/-- The slope of a line perpendicular to the line containing points (2, -3) and (-4, -8) is -6/5 -/
theorem perpendicular_slope : 
  let p₁ : ℚ × ℚ := (2, -3)
  let p₂ : ℚ × ℚ := (-4, -8)
  let m : ℚ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  (-1 / m) = -6/5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l4074_407435


namespace NUMINAMATH_CALUDE_max_revenue_l4074_407427

def price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def revenue (t : ℕ) : ℝ :=
  price t * sales_volume t

theorem max_revenue :
  ∃ (t : ℕ), t = 25 ∧ revenue t = 1125 ∧
  ∀ (s : ℕ), 0 < s ∧ s ≤ 30 → revenue s ≤ revenue t := by
  sorry

end NUMINAMATH_CALUDE_max_revenue_l4074_407427


namespace NUMINAMATH_CALUDE_smallest_add_subtract_for_perfect_square_l4074_407461

theorem smallest_add_subtract_for_perfect_square (n m : ℕ) : 
  (∀ k : ℕ, k < 470 → ¬∃ i : ℕ, 92555 + k = i^2) ∧
  (∃ i : ℕ, 92555 + 470 = i^2) ∧
  (∀ j : ℕ, j < 139 → ¬∃ i : ℕ, 92555 - j = i^2) ∧
  (∃ i : ℕ, 92555 - 139 = i^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_add_subtract_for_perfect_square_l4074_407461


namespace NUMINAMATH_CALUDE_automotive_test_time_l4074_407491

/-- Proves that given a car driven the same distance three times at speeds of 4, 5, and 6 miles per hour respectively, and a total distance of 180 miles, the total time taken is 37 hours. -/
theorem automotive_test_time (total_distance : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_distance = 180 ∧ 
  speed1 = 4 ∧ 
  speed2 = 5 ∧ 
  speed3 = 6 → 
  (total_distance / (3 * speed1) + total_distance / (3 * speed2) + total_distance / (3 * speed3)) = 37 := by
  sorry

#check automotive_test_time

end NUMINAMATH_CALUDE_automotive_test_time_l4074_407491


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4074_407420

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3)*(8*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4074_407420


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l4074_407482

/-- Convert a binary number (represented as a list of 0s and 1s) to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.foldr (fun bit acc => 2 * acc + bit) 0

/-- Convert a decimal number to its binary representation -/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 2) ((m % 2) :: acc)
  aux n []

theorem binary_arithmetic_equality :
  let a := [1, 0, 0, 1, 1, 0]  -- 100110₂
  let b := [1, 0, 0, 1]        -- 1001₂
  let c := [1, 1, 0]           -- 110₂
  let d := [1, 1]              -- 11₂
  let result := [1, 0, 1, 1, 1, 1, 0]  -- 1011110₂
  (binary_to_decimal a + binary_to_decimal b) * binary_to_decimal c / binary_to_decimal d =
  binary_to_decimal result := by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l4074_407482


namespace NUMINAMATH_CALUDE_base_prime_182_l4074_407472

/-- Represents a number in base prime notation --/
def BasePrime : Type := List Nat

/-- Converts a natural number to its base prime representation --/
def toBasePrime (n : Nat) : BasePrime :=
  sorry

/-- Theorem: The base prime representation of 182 is [1, 0, 0, 1, 0, 1] --/
theorem base_prime_182 : toBasePrime 182 = [1, 0, 0, 1, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_182_l4074_407472


namespace NUMINAMATH_CALUDE_triangle_theorem_l4074_407457

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - Real.sqrt 2 * t.b * t.c = t.a^2)
  (h2 : t.c / t.b = 2 * Real.sqrt 2) : 
  t.A = π/4 ∧ Real.tan t.B = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l4074_407457
