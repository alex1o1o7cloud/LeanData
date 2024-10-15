import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_y_l396_39660

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l396_39660


namespace NUMINAMATH_CALUDE_equation_solution_l396_39689

theorem equation_solution : ∃! x : ℚ, x + 2/3 = 7/15 + 1/5 - x/2 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l396_39689


namespace NUMINAMATH_CALUDE_divisibility_theorem_l396_39613

theorem divisibility_theorem (a b c d u : ℤ) 
  (h1 : u ∣ (a * c)) 
  (h2 : u ∣ (b * c + a * d)) 
  (h3 : u ∣ (b * d)) : 
  (u ∣ (b * c)) ∧ (u ∣ (a * d)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l396_39613


namespace NUMINAMATH_CALUDE_larger_circle_tangent_to_line_and_axes_l396_39679

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

/-- Checks if a circle is tangent to a line ax + by = c -/
def isTangentToLine (circle : Circle) (a b c : ℝ) : Prop :=
  let (x, y) := circle.center
  |a * x + b * y - c| / Real.sqrt (a^2 + b^2) = circle.radius

/-- Checks if a circle is tangent to both coordinate axes -/
def isTangentToAxes (circle : Circle) : Prop :=
  circle.center.1 = circle.radius ∧ circle.center.2 = circle.radius

/-- The theorem to be proved -/
theorem larger_circle_tangent_to_line_and_axes :
  ∃ (circle : Circle),
    circle.center = (5/2, 5/2) ∧
    circle.radius = 5/2 ∧
    isInFirstQuadrant circle.center ∧
    isTangentToLine circle 3 4 5 ∧
    isTangentToAxes circle ∧
    (∀ (other : Circle),
      isInFirstQuadrant other.center →
      isTangentToLine other 3 4 5 →
      isTangentToAxes other →
      other.radius ≤ circle.radius) :=
  sorry

end NUMINAMATH_CALUDE_larger_circle_tangent_to_line_and_axes_l396_39679


namespace NUMINAMATH_CALUDE_no_integer_solution_l396_39683

theorem no_integer_solution : ¬∃ (m n : ℤ), m^2 + 1954 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l396_39683


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_is_2_root_6_l396_39678

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ :=
  sorry

/-- The specific quadrilateral from the problem -/
def problem_quadrilateral : Quadrilateral :=
  ⟨13, 10, 8, 11⟩

theorem largest_inscribed_circle_radius_is_2_root_6 :
  largest_inscribed_circle_radius problem_quadrilateral = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_is_2_root_6_l396_39678


namespace NUMINAMATH_CALUDE_expression_equals_four_l396_39633

theorem expression_equals_four :
  (8 : ℝ) ^ (1/3) + (1/3)⁻¹ - 2 * Real.cos (30 * π / 180) + |1 - Real.sqrt 3| = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_four_l396_39633


namespace NUMINAMATH_CALUDE_barbara_butcher_cost_l396_39618

/-- The cost of Barbara's purchase at the butcher's --/
def butcher_cost (steak_weight : ℝ) (steak_price : ℝ) (chicken_weight : ℝ) (chicken_price : ℝ) : ℝ :=
  steak_weight * steak_price + chicken_weight * chicken_price

/-- Theorem: Barbara's total cost at the butcher's is $79.50 --/
theorem barbara_butcher_cost :
  butcher_cost 4.5 15 1.5 8 = 79.5 := by
  sorry

end NUMINAMATH_CALUDE_barbara_butcher_cost_l396_39618


namespace NUMINAMATH_CALUDE_solution_of_system_l396_39623

/-- Given a system of equations, prove that the solutions are (2, 1) and (2/5, -1/5) -/
theorem solution_of_system :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ = 2 ∧ y₁ = 1) ∧
    (x₂ = 2/5 ∧ y₂ = -1/5) ∧
    (∀ x y : ℝ,
      (5 * x * (1 + 1 / (x^2 + y^2)) = 12 ∧
       5 * y * (1 - 1 / (x^2 + y^2)) = 4) ↔
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l396_39623


namespace NUMINAMATH_CALUDE_color_film_fraction_l396_39632

theorem color_film_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw := 40 * x
  let total_color := 4 * y
  let selected_bw := (y / x) * (40 * x) / 100
  let selected_color := total_color
  (selected_color) / (selected_bw + selected_color) = 10 / 11 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l396_39632


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l396_39653

/-- For a rectangle with sum of length and width equal to 28 meters, the perimeter is 56 meters. -/
theorem rectangle_perimeter (l w : ℝ) (h : l + w = 28) : 2 * (l + w) = 56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l396_39653


namespace NUMINAMATH_CALUDE_square_root_of_nine_three_is_square_root_of_nine_l396_39601

theorem square_root_of_nine (x : ℝ) : x ^ 2 = 9 → x = 3 ∨ x = -3 := by
  sorry

theorem three_is_square_root_of_nine : ∃ x : ℝ, x ^ 2 = 9 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_three_is_square_root_of_nine_l396_39601


namespace NUMINAMATH_CALUDE_x_power_plus_inverse_l396_39630

theorem x_power_plus_inverse (θ : ℝ) (x : ℂ) (n : ℤ) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : x + 1/x = 2 * Real.cos θ) :
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_x_power_plus_inverse_l396_39630


namespace NUMINAMATH_CALUDE_phase_shift_sin_5x_minus_pi_half_l396_39667

/-- The phase shift of the function y = sin(5x - π/2) is π/10 to the right or -π/10 to the left -/
theorem phase_shift_sin_5x_minus_pi_half :
  let f : ℝ → ℝ := λ x => Real.sin (5 * x - π / 2)
  ∃ φ : ℝ, (φ = π / 10 ∨ φ = -π / 10) ∧
    ∀ x : ℝ, f x = Real.sin (5 * (x - φ)) :=
by sorry

end NUMINAMATH_CALUDE_phase_shift_sin_5x_minus_pi_half_l396_39667


namespace NUMINAMATH_CALUDE_train_length_calculation_l396_39635

theorem train_length_calculation (platform_crossing_time platform_length signal_crossing_time : ℝ) 
  (h1 : platform_crossing_time = 27)
  (h2 : platform_length = 150.00000000000006)
  (h3 : signal_crossing_time = 18) :
  ∃ train_length : ℝ, train_length = 300.0000000000001 ∧
    platform_crossing_time * (train_length / signal_crossing_time) = train_length + platform_length :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l396_39635


namespace NUMINAMATH_CALUDE_interval_bound_l396_39645

-- Define the functions
def f (x : ℝ) := x^4 - 2*x^2
def g (x : ℝ) := 4*x^2 - 8
def h (t x : ℝ) := 4*(t^3 - t)*x - 3*t^4 + 2*t^2

-- State the theorem
theorem interval_bound 
  (t : ℝ) 
  (ht : 0 < |t| ∧ |t| ≤ Real.sqrt 2) 
  (m n : ℝ) 
  (hmn : m ≤ n ∧ Set.Icc m n ⊆ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)) 
  (h_inequality : ∀ x ∈ Set.Icc m n, f x ≥ h t x ∧ h t x ≥ g x) : 
  n - m ≤ Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_interval_bound_l396_39645


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l396_39629

theorem consecutive_even_numbers_sum (a b c : ℤ) : 
  (∃ k : ℤ, b = 2 * k) →  -- b is even
  (a = b - 2) →           -- a is the previous even number
  (c = b + 2) →           -- c is the next even number
  (a + b = 18) →          -- sum of first and second
  (a + c = 22) →          -- sum of first and third
  (b + c = 28) →          -- sum of second and third
  b = 11 :=               -- middle number is 11
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l396_39629


namespace NUMINAMATH_CALUDE_letter_150_is_Z_l396_39692

def repeating_sequence : ℕ → Char
  | n => let idx := n % 3
         if idx = 0 then 'Z'
         else if idx = 1 then 'X'
         else 'Y'

theorem letter_150_is_Z : repeating_sequence 150 = 'Z' := by
  sorry

end NUMINAMATH_CALUDE_letter_150_is_Z_l396_39692


namespace NUMINAMATH_CALUDE_average_problem_l396_39691

theorem average_problem (x : ℝ) : (15 + 25 + x + 30) / 4 = 23 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l396_39691


namespace NUMINAMATH_CALUDE_tangent_line_at_P_tangent_line_not_at_P_l396_39657

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Theorem for the first part
theorem tangent_line_at_P :
  ∃ (l : ℝ → ℝ), (l 1 = -2) ∧ 
  (∀ x : ℝ, l x = -2) ∧
  (∀ x : ℝ, x ≠ 1 → (l x - f x) / (x - 1) ≤ (l 1 - f 1) / (1 - 1)) :=
sorry

-- Theorem for the second part
theorem tangent_line_not_at_P :
  ∃ (l : ℝ → ℝ), (l 1 = -2) ∧ 
  (∀ x : ℝ, 9*x + 4*(l x) - 1 = 0) ∧
  (∃ x₀ : ℝ, x₀ ≠ 1 ∧ 
    (∀ x : ℝ, x ≠ x₀ → (l x - f x) / (x - x₀) ≤ (l x₀ - f x₀) / (x₀ - x₀))) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_tangent_line_not_at_P_l396_39657


namespace NUMINAMATH_CALUDE_range_of_m_l396_39608

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b)
  (h : ∀ (a b : ℝ), a > 0 → b > 0 → a * b = 2 * a + b → a + 2 * b ≥ m^2 - 8 * m) :
  -1 ≤ m ∧ m ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l396_39608


namespace NUMINAMATH_CALUDE_max_intersection_length_l396_39634

noncomputable section

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit
def M : Point := Unit.unit
def N : Point := Unit.unit
def A : Point := Unit.unit
def B : Point := Unit.unit
def C : Point := Unit.unit

-- Define the diameter and its length
def diameter (c : Circle) : ℝ := 2

-- Define the property that MN is a diameter
def is_diameter (c : Circle) (m n : Point) : Prop := True

-- Define A as the midpoint of the semicircular arc
def is_midpoint_arc (c : Circle) (m n a : Point) : Prop := True

-- Define the length of MB
def length_MB : ℝ := 4/7

-- Define C as a point on the other semicircular arc
def on_other_arc (c : Circle) (m n c : Point) : Prop := True

-- Define the intersections of MN with AC and BC
def intersection_AC_MN (c : Circle) (m n a c : Point) : Point := Unit.unit
def intersection_BC_MN (c : Circle) (m n b c : Point) : Point := Unit.unit

-- Define the length of the line segment formed by the intersections
def length_intersections (p q : Point) : ℝ := 0

-- Theorem statement
theorem max_intersection_length (c : Circle) :
  is_diameter c M N →
  is_midpoint_arc c M N A →
  length_MB = 4/7 →
  on_other_arc c M N C →
  ∃ (d : ℝ), d = 10 - 7 * Real.sqrt 3 ∧
    ∀ (V W : Point),
      V = intersection_AC_MN c M N A C →
      W = intersection_BC_MN c M N B C →
      length_intersections V W ≤ d :=
sorry

end

end NUMINAMATH_CALUDE_max_intersection_length_l396_39634


namespace NUMINAMATH_CALUDE_first_to_light_is_match_l396_39685

/-- Represents items that can be lit --/
inductive LightableItem
  | Match
  | Candle
  | KeroseneLamp
  | Stove

/-- Represents the state of a room --/
structure Room where
  isDark : Bool
  hasMatch : Bool
  items : List LightableItem

/-- Determines the first item that must be lit in a given room --/
def firstItemToLight (room : Room) : LightableItem := by sorry

/-- Theorem: The first item to light in a dark room with a match and other lightable items is the match itself --/
theorem first_to_light_is_match (room : Room) 
  (h1 : room.isDark = true) 
  (h2 : room.hasMatch = true) 
  (h3 : LightableItem.Candle ∈ room.items) 
  (h4 : LightableItem.KeroseneLamp ∈ room.items) 
  (h5 : LightableItem.Stove ∈ room.items) : 
  firstItemToLight room = LightableItem.Match := by sorry

end NUMINAMATH_CALUDE_first_to_light_is_match_l396_39685


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l396_39690

/-- Given two rectangles with equal areas, where one has length 5 and width 24,
    and the other has width 10, prove that the length of the second rectangle is 12. -/
theorem equal_area_rectangles (l₁ w₁ w₂ : ℝ) (h₁ : l₁ = 5) (h₂ : w₁ = 24) (h₃ : w₂ = 10) :
  let a₁ := l₁ * w₁
  let l₂ := a₁ / w₂
  l₂ = 12 := by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l396_39690


namespace NUMINAMATH_CALUDE_battery_difference_is_thirteen_l396_39688

/-- The number of batteries Tom used in his flashlights -/
def flashlight_batteries : ℕ := 2

/-- The number of batteries Tom used in his toys -/
def toy_batteries : ℕ := 15

/-- The number of batteries Tom used in his controllers -/
def controller_batteries : ℕ := 2

/-- The difference between the number of batteries in Tom's toys and flashlights -/
def battery_difference : ℕ := toy_batteries - flashlight_batteries

theorem battery_difference_is_thirteen : battery_difference = 13 := by
  sorry

end NUMINAMATH_CALUDE_battery_difference_is_thirteen_l396_39688


namespace NUMINAMATH_CALUDE_dorchester_puppies_washed_l396_39650

/-- Calculates the number of puppies washed given the total earnings, base pay, and rate per puppy -/
def puppies_washed (total_earnings base_pay rate_per_puppy : ℚ) : ℚ :=
  (total_earnings - base_pay) / rate_per_puppy

/-- Proves that Dorchester washed 16 puppies on Wednesday -/
theorem dorchester_puppies_washed :
  puppies_washed 76 40 (9/4) = 16 := by
  sorry

end NUMINAMATH_CALUDE_dorchester_puppies_washed_l396_39650


namespace NUMINAMATH_CALUDE_fish_in_each_bowl_l396_39648

theorem fish_in_each_bowl (total_bowls : ℕ) (total_fish : ℕ) (h1 : total_bowls = 261) (h2 : total_fish = 6003) :
  total_fish / total_bowls = 23 := by
  sorry

end NUMINAMATH_CALUDE_fish_in_each_bowl_l396_39648


namespace NUMINAMATH_CALUDE_solution_set_1_correct_solution_set_2_correct_l396_39642

-- Define the solution set for the first inequality
def solution_set_1 : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the solution set for the second inequality based on the value of a
def solution_set_2 (a : ℝ) : Set ℝ :=
  if a = -2 then Set.univ
  else if a > -2 then {x | x ≤ -2 ∨ x ≥ a}
  else {x | x ≤ a ∨ x ≥ -2}

-- Theorem for the first inequality
theorem solution_set_1_correct :
  ∀ x : ℝ, x ∈ solution_set_1 ↔ (2 * x) / (x + 1) < 1 :=
sorry

-- Theorem for the second inequality
theorem solution_set_2_correct :
  ∀ a x : ℝ, x ∈ solution_set_2 a ↔ x^2 + (2 - a) * x - 2 * a ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_1_correct_solution_set_2_correct_l396_39642


namespace NUMINAMATH_CALUDE_count_switchable_positions_l396_39636

/-- Represents the number of revolutions a clock hand makes in one hour -/
def revolutions_per_hour (is_minute_hand : Bool) : ℚ :=
  if is_minute_hand then 1 else 1/12

/-- Represents a valid clock position -/
def is_valid_position (hour_pos : ℚ) (minute_pos : ℚ) : Prop :=
  0 ≤ hour_pos ∧ hour_pos < 1 ∧ 0 ≤ minute_pos ∧ minute_pos < 1

/-- Represents a clock position that remains valid when hands are switched -/
def is_switchable_position (t : ℚ) : Prop :=
  is_valid_position (t * revolutions_per_hour false) (t * revolutions_per_hour true) ∧
  is_valid_position (t * revolutions_per_hour true) (t * revolutions_per_hour false)

/-- The main theorem stating the number of switchable positions -/
theorem count_switchable_positions :
  (∃ (S : Finset ℚ), (∀ t ∈ S, is_switchable_position t) ∧ S.card = 143) :=
sorry

end NUMINAMATH_CALUDE_count_switchable_positions_l396_39636


namespace NUMINAMATH_CALUDE_second_derivative_value_l396_39651

def f (q : ℝ) : ℝ := 3 * q - 3

theorem second_derivative_value (q : ℝ) : f (f q) = 210 → q = 74 / 3 := by
  sorry

end NUMINAMATH_CALUDE_second_derivative_value_l396_39651


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l396_39698

theorem fraction_to_decimal : (5 : ℚ) / 40 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l396_39698


namespace NUMINAMATH_CALUDE_third_and_fourth_terms_equal_21_l396_39647

def a (n : ℕ) : ℤ := -n^2 + 7*n + 9

theorem third_and_fourth_terms_equal_21 : a 3 = 21 ∧ a 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_third_and_fourth_terms_equal_21_l396_39647


namespace NUMINAMATH_CALUDE_max_difference_l396_39652

theorem max_difference (a b : ℝ) : 
  a < 0 → 
  (∀ x, a < x ∧ x < b → (x^2 + 2017*a)*(x + 2016*b) ≥ 0) → 
  b - a ≤ 2017 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_l396_39652


namespace NUMINAMATH_CALUDE_exists_N_average_twelve_l396_39627

theorem exists_N_average_twelve : ∃ N : ℝ, 11 < N ∧ N < 19 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_exists_N_average_twelve_l396_39627


namespace NUMINAMATH_CALUDE_two_digit_number_divisible_by_55_l396_39658

theorem two_digit_number_divisible_by_55 (a b : ℕ) : 
  (a ≥ 1 ∧ a ≤ 9) →  -- 'a' is a single digit (tens place)
  (b ≥ 0 ∧ b ≤ 9) →  -- 'b' is a single digit (units place)
  (10 * a + b) % 55 = 0 →  -- number is divisible by 55
  (∀ (x y : ℕ), (x ≥ 1 ∧ x ≤ 9) → (y ≥ 0 ∧ y ≤ 9) → (10 * x + y) % 55 = 0 → x * y ≤ 15) →  -- greatest possible value of b × a is 15
  10 * a + b = 65 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_divisible_by_55_l396_39658


namespace NUMINAMATH_CALUDE_difference_multiplier_proof_l396_39686

theorem difference_multiplier_proof : ∃ x : ℕ, 
  let sum := 555 + 445
  let difference := 555 - 445
  220040 = sum * (x * difference) + 40 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_multiplier_proof_l396_39686


namespace NUMINAMATH_CALUDE_q_investment_value_l396_39693

/-- Represents the investment and profit division of two business partners -/
structure BusinessInvestment where
  p_investment : ℝ
  q_investment : ℝ
  profit_ratio : ℝ × ℝ

/-- Given the conditions of the problem, prove that q's investment is 45000 -/
theorem q_investment_value (b : BusinessInvestment) 
  (h1 : b.p_investment = 30000)
  (h2 : b.profit_ratio = (2, 3)) :
  b.q_investment = 45000 := by
  sorry

end NUMINAMATH_CALUDE_q_investment_value_l396_39693


namespace NUMINAMATH_CALUDE_unknown_number_is_ten_l396_39699

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem unknown_number_is_ten :
  ∀ n : ℝ, euro 8 (euro n 5) = 640 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_is_ten_l396_39699


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l396_39606

/-- Calculates the average speed of a cyclist's trip given two segments with different speeds -/
theorem cyclist_average_speed (d1 d2 v1 v2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 11) (h3 : v1 = 11) (h4 : v2 = 8) :
  (d1 + d2) / ((d1 / v1) + (d2 / v2)) = 1664 / 185 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_average_speed_l396_39606


namespace NUMINAMATH_CALUDE_reading_time_calculation_l396_39659

theorem reading_time_calculation (pages_per_hour_1 pages_per_hour_2 pages_per_hour_3 : ℕ)
  (total_pages : ℕ) (h1 : pages_per_hour_1 = 21) (h2 : pages_per_hour_2 = 30)
  (h3 : pages_per_hour_3 = 45) (h4 : total_pages = 128) :
  let total_time := (3 * total_pages) / (pages_per_hour_1 + pages_per_hour_2 + pages_per_hour_3)
  total_time = 4 := by
sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l396_39659


namespace NUMINAMATH_CALUDE_square_of_two_power_minus_twice_l396_39665

theorem square_of_two_power_minus_twice (N : ℕ+) :
  (∃ k : ℕ, 2^N.val - 2 * N.val = k^2) ↔ N = 1 ∨ N = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_power_minus_twice_l396_39665


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l396_39609

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l396_39609


namespace NUMINAMATH_CALUDE_expand_product_l396_39638

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l396_39638


namespace NUMINAMATH_CALUDE_squares_after_six_steps_l396_39697

/-- The number of squares after n steps, given an initial configuration of 5 squares 
    and each step adds 3 squares -/
def num_squares (n : ℕ) : ℕ := 5 + 3 * n

/-- Theorem stating that after 6 steps, there are 23 squares -/
theorem squares_after_six_steps : num_squares 6 = 23 := by
  sorry

end NUMINAMATH_CALUDE_squares_after_six_steps_l396_39697


namespace NUMINAMATH_CALUDE_sodium_bicarbonate_required_l396_39655

-- Define the chemical reaction
structure Reaction where
  NaHCO₃ : ℕ
  HCl : ℕ
  NaCl : ℕ
  H₂O : ℕ
  CO₂ : ℕ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.NaHCO₃ = r.HCl ∧ r.NaHCO₃ = r.NaCl ∧ r.NaHCO₃ = r.H₂O ∧ r.NaHCO₃ = r.CO₂

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.HCl = 3 ∧ r.H₂O = 3 ∧ r.CO₂ = 3 ∧ r.NaCl = 3

-- Theorem to prove
theorem sodium_bicarbonate_required (r : Reaction) 
  (h1 : balanced_equation r) (h2 : given_conditions r) : 
  r.NaHCO₃ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sodium_bicarbonate_required_l396_39655


namespace NUMINAMATH_CALUDE_vector_problem_l396_39602

/-- Given two vectors a and b in ℝ², proves that if a is collinear with b and their dot product is -10, then b is equal to (-4, 2) -/
theorem vector_problem (a b : ℝ × ℝ) : 
  a = (2, -1) → 
  (∃ k : ℝ, b = k • a) → 
  a.1 * b.1 + a.2 * b.2 = -10 → 
  b = (-4, 2) := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l396_39602


namespace NUMINAMATH_CALUDE_liam_needed_one_more_correct_answer_l396_39612

/-- Represents the number of questions Liam answered correctly in each category -/
structure CorrectAnswers where
  programming : ℕ
  dataStructures : ℕ
  algorithms : ℕ

/-- Calculates the total number of correct answers -/
def totalCorrect (answers : CorrectAnswers) : ℕ :=
  answers.programming + answers.dataStructures + answers.algorithms

/-- Represents the examination structure and Liam's performance -/
structure Examination where
  totalQuestions : ℕ
  programmingQuestions : ℕ
  dataStructuresQuestions : ℕ
  algorithmsQuestions : ℕ
  passingPercentage : ℚ
  correctAnswers : CorrectAnswers

/-- Theorem stating that Liam needed 1 more correct answer to pass -/
theorem liam_needed_one_more_correct_answer (exam : Examination)
  (h1 : exam.totalQuestions = 50)
  (h2 : exam.programmingQuestions = 15)
  (h3 : exam.dataStructuresQuestions = 20)
  (h4 : exam.algorithmsQuestions = 15)
  (h5 : exam.passingPercentage = 65 / 100)
  (h6 : exam.correctAnswers.programming = 12)
  (h7 : exam.correctAnswers.dataStructures = 10)
  (h8 : exam.correctAnswers.algorithms = 10) :
  ⌈exam.totalQuestions * exam.passingPercentage⌉ - totalCorrect exam.correctAnswers = 1 := by
  sorry


end NUMINAMATH_CALUDE_liam_needed_one_more_correct_answer_l396_39612


namespace NUMINAMATH_CALUDE_geometric_sequence_special_term_l396_39668

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The 14th term of a geometric sequence -/
def a_14 (a : ℕ → ℝ) : ℝ := a 14

/-- The 4th term of a geometric sequence -/
def a_4 (a : ℕ → ℝ) : ℝ := a 4

/-- The 24th term of a geometric sequence -/
def a_24 (a : ℕ → ℝ) : ℝ := a 24

/-- Theorem: In a geometric sequence, if a_4 and a_24 are roots of 3x^2 - 2014x + 9 = 0, then a_14 = √3 -/
theorem geometric_sequence_special_term (a : ℕ → ℝ) :
  geometric_sequence a →
  (3 * (a_4 a)^2 - 2014 * (a_4 a) + 9 = 0) →
  (3 * (a_24 a)^2 - 2014 * (a_24 a) + 9 = 0) →
  a_14 a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_term_l396_39668


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l396_39616

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + Real.log 4 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + Real.log 4 / Real.log 8 + 1) +
  1 / (1 + (Real.log 5 / Real.log 15 + Real.log 3 / Real.log 15)) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l396_39616


namespace NUMINAMATH_CALUDE_summer_work_hours_adjustment_l396_39619

theorem summer_work_hours_adjustment 
  (initial_weeks : ℕ) 
  (initial_hours_per_week : ℝ) 
  (unavailable_weeks : ℕ) 
  (adjusted_hours_per_week : ℝ) :
  initial_weeks > unavailable_weeks →
  initial_weeks * initial_hours_per_week = 
    (initial_weeks - unavailable_weeks) * adjusted_hours_per_week →
  adjusted_hours_per_week = initial_hours_per_week * (initial_weeks / (initial_weeks - unavailable_weeks)) :=
by
  sorry

#eval (31.25 : Float)

end NUMINAMATH_CALUDE_summer_work_hours_adjustment_l396_39619


namespace NUMINAMATH_CALUDE_triangle_inequality_l396_39676

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = π) :
  let f (x : ℝ) := 1 - Real.sqrt (Real.sqrt 3 * Real.tan (x / 2)) + Real.sqrt 3 * Real.tan (x / 2)
  (f A * f B) + (f B * f C) + (f C * f A) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l396_39676


namespace NUMINAMATH_CALUDE_cube_root_of_four_solution_l396_39695

theorem cube_root_of_four_solution (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_four_solution_l396_39695


namespace NUMINAMATH_CALUDE_staircase_steps_l396_39696

/-- The number of toothpicks in a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 3

/-- The proposition that 8 steps result in 490 toothpicks -/
theorem staircase_steps : toothpicks 8 = 490 := by
  sorry

#check staircase_steps

end NUMINAMATH_CALUDE_staircase_steps_l396_39696


namespace NUMINAMATH_CALUDE_gcd_987654_876543_l396_39605

theorem gcd_987654_876543 : Nat.gcd 987654 876543 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_987654_876543_l396_39605


namespace NUMINAMATH_CALUDE_five_people_seven_chairs_l396_39611

/-- The number of ways to arrange n people in k chairs, where the first person
    cannot sit in m specific chairs. -/
def seating_arrangements (n k m : ℕ) : ℕ :=
  (k - m) * (k - 1).factorial / (k - n).factorial

/-- The problem statement -/
theorem five_people_seven_chairs : seating_arrangements 5 7 2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_five_people_seven_chairs_l396_39611


namespace NUMINAMATH_CALUDE_smallest_b_value_l396_39662

theorem smallest_b_value (a b : ℕ+) 
  (h1 : a.val - b.val = 10)
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ x : ℕ+, 2 ≤ x.val → x.val < b.val → False :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l396_39662


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l396_39670

theorem simplify_fraction_product : 8 * (15 / 9) * (-21 / 35) = -8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l396_39670


namespace NUMINAMATH_CALUDE_sum_a_d_equals_one_l396_39684

theorem sum_a_d_equals_one 
  (a b c d : ℤ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_one_l396_39684


namespace NUMINAMATH_CALUDE_complex_product_polar_form_l396_39610

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the problem statement
theorem complex_product_polar_form :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  (4 * cis (25 * π / 180)) * (-3 * cis (48 * π / 180)) = r * cis θ ∧
  r = 12 ∧ θ = 253 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_polar_form_l396_39610


namespace NUMINAMATH_CALUDE_sum_10_to_16_l396_39604

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q
  sum_2_4 : a 2 + a 4 = 32
  sum_6_8 : a 6 + a 8 = 16

/-- The sum of the 10th, 12th, 14th, and 16th terms equals 12 -/
theorem sum_10_to_16 (seq : GeometricSequence) :
  seq.a 10 + seq.a 12 + seq.a 14 + seq.a 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_10_to_16_l396_39604


namespace NUMINAMATH_CALUDE_sqrt_inequality_l396_39639

theorem sqrt_inequality (x : ℝ) :
  (3 - x ≥ 0) → (x + 1 ≥ 0) →
  (Real.sqrt (3 - x) - Real.sqrt (x + 1) > 1/2 ↔ -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l396_39639


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l396_39663

theorem fraction_equation_solution : 
  ∃ x : ℚ, (3/4 * 60 - x * 60 + 63 = 12) ∧ (x = 8/5) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l396_39663


namespace NUMINAMATH_CALUDE_intersection_size_lower_bound_l396_39677

theorem intersection_size_lower_bound 
  (n k : ℕ) 
  (A : Fin (k + 1) → Finset (Fin (4 * n))) 
  (h1 : ∀ i, (A i).card = 2 * n) :
  ∃ i j, i ≠ j ∧ (A i ∩ A j).card ≥ n - n / k := by
  sorry

end NUMINAMATH_CALUDE_intersection_size_lower_bound_l396_39677


namespace NUMINAMATH_CALUDE_dice_probability_l396_39656

/-- The number of sides on each die -/
def sides : ℕ := 15

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The threshold for "low" numbers -/
def low_threshold : ℕ := 10

/-- The number of low outcomes on a single die -/
def low_outcomes : ℕ := low_threshold - 1

/-- The number of high outcomes on a single die -/
def high_outcomes : ℕ := sides - low_outcomes

/-- The probability of rolling a low number on a single die -/
def prob_low : ℚ := low_outcomes / sides

/-- The probability of rolling a high number on a single die -/
def prob_high : ℚ := high_outcomes / sides

/-- The number of ways to choose 3 dice out of 5 -/
def ways_to_choose : ℕ := (num_dice.choose 3)

theorem dice_probability : 
  (ways_to_choose : ℚ) * prob_low^3 * prob_high^2 = 216/625 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l396_39656


namespace NUMINAMATH_CALUDE_sum_odd_integers_l396_39607

theorem sum_odd_integers (n : ℕ) (h : n * (n + 1) = 4970) : n^2 = 4900 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_l396_39607


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l396_39603

/-- The equation of a line passing through (1,1) and tangent to the circle x^2 - 2x + y^2 = 0 is y = 1 -/
theorem tangent_line_to_circle (x y : ℝ) : 
  (∃ k : ℝ, y - 1 = k * (x - 1)) ∧ 
  (x^2 - 2*x + y^2 = 0 → (x - 1)^2 + (y - 0)^2 = 1) →
  y = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l396_39603


namespace NUMINAMATH_CALUDE_amc_12_score_problem_l396_39615

theorem amc_12_score_problem (total_problems : Nat) (attempted_problems : Nat) 
  (correct_points : Nat) (incorrect_points : Nat) (unanswered_points : Nat) 
  (unanswered_count : Nat) (min_score : Nat) :
  total_problems = 30 →
  attempted_problems = 26 →
  correct_points = 7 →
  incorrect_points = 0 →
  unanswered_points = 1 →
  unanswered_count = 4 →
  min_score = 150 →
  ∃ (correct_count : Nat), 
    correct_count * correct_points + 
    (attempted_problems - correct_count) * incorrect_points + 
    unanswered_count * unanswered_points ≥ min_score ∧
    correct_count = 21 ∧
    ∀ (x : Nat), x < 21 → 
      x * correct_points + 
      (attempted_problems - x) * incorrect_points + 
      unanswered_count * unanswered_points < min_score :=
by sorry

end NUMINAMATH_CALUDE_amc_12_score_problem_l396_39615


namespace NUMINAMATH_CALUDE_medical_team_probability_l396_39666

theorem medical_team_probability (male_doctors female_doctors team_size : ℕ) 
  (h1 : male_doctors = 6)
  (h2 : female_doctors = 3)
  (h3 : team_size = 5) : 
  (1 - (Nat.choose male_doctors team_size : ℚ) / (Nat.choose (male_doctors + female_doctors) team_size)) = 60/63 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_probability_l396_39666


namespace NUMINAMATH_CALUDE_geometric_progression_fifth_term_l396_39622

theorem geometric_progression_fifth_term 
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁ = 2^(1/4 : ℝ))
  (h₂ : a₂ = 2^(1/5 : ℝ))
  (h₃ : a₃ = 2^(1/6 : ℝ))
  (h₄ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) :
  ∃ a₅ : ℝ, a₅ = 2^(11/60 : ℝ) ∧ 
    ∃ a₄ : ℝ, a₄ = a₃ * (a₂ / a₁) ∧ a₅ = a₄ * (a₂ / a₁) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_fifth_term_l396_39622


namespace NUMINAMATH_CALUDE_figure_rearrangeable_to_square_l396_39673

/-- A figure on graph paper can be rearranged into a square if and only if 
    its area (in unit squares) is a perfect square. -/
theorem figure_rearrangeable_to_square (n : ℕ) : 
  (∃ (k : ℕ), n = k^2) ↔ (∃ (s : ℕ), s^2 = n) :=
sorry

end NUMINAMATH_CALUDE_figure_rearrangeable_to_square_l396_39673


namespace NUMINAMATH_CALUDE_mary_clothing_expense_l396_39620

-- Define the costs of the shirt and jacket
def shirt_cost : Real := 13.04
def jacket_cost : Real := 12.27

-- Define the total cost
def total_cost : Real := shirt_cost + jacket_cost

-- Theorem statement
theorem mary_clothing_expense : total_cost = 25.31 := by
  sorry

end NUMINAMATH_CALUDE_mary_clothing_expense_l396_39620


namespace NUMINAMATH_CALUDE_existence_of_distinct_pairs_l396_39643

theorem existence_of_distinct_pairs 
  (S T : Type) [Finite S] [Finite T] 
  (U : Set (S × T)) 
  (h1 : ∀ s : S, ∃ t : T, (s, t) ∉ U) 
  (h2 : ∀ t : T, ∃ s : S, (s, t) ∈ U) :
  ∃ (s₁ s₂ : S) (t₁ t₂ : T), 
    s₁ ≠ s₂ ∧ t₁ ≠ t₂ ∧ 
    (s₁, t₁) ∈ U ∧ (s₂, t₂) ∈ U ∧ 
    (s₁, t₂) ∉ U ∧ (s₂, t₁) ∉ U :=
by sorry

end NUMINAMATH_CALUDE_existence_of_distinct_pairs_l396_39643


namespace NUMINAMATH_CALUDE_class_average_problem_l396_39644

theorem class_average_problem (n₁ n₂ : ℕ) (avg₂ avg_combined : ℚ) :
  n₁ = 30 →
  n₂ = 50 →
  avg₂ = 60 →
  avg_combined = 52.5 →
  ∃ avg₁ : ℚ, avg₁ = 40 ∧ (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = avg_combined :=
by sorry

end NUMINAMATH_CALUDE_class_average_problem_l396_39644


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_condition_l396_39694

theorem quadratic_complex_roots_condition (a : ℝ) :
  (∃ x : ℂ, x^2 + a*x + 1 = 0 ∧ x.im ≠ 0) →
  a < 2 ∧
  ¬(a < 2 → ∃ x : ℂ, x^2 + a*x + 1 = 0 ∧ x.im ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_condition_l396_39694


namespace NUMINAMATH_CALUDE_abigail_report_words_l396_39640

/-- Represents Abigail's report writing scenario -/
structure ReportWriting where
  typing_speed : ℕ  -- words per 30 minutes
  words_written : ℕ
  time_needed : ℕ  -- in minutes

/-- Calculates the total number of words in the report -/
def total_words (r : ReportWriting) : ℕ :=
  r.words_written + r.typing_speed * r.time_needed / 30

/-- Theorem stating that the total words in Abigail's report is 1000 -/
theorem abigail_report_words :
  ∃ (r : ReportWriting), r.typing_speed = 300 ∧ r.words_written = 200 ∧ r.time_needed = 80 ∧ total_words r = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_abigail_report_words_l396_39640


namespace NUMINAMATH_CALUDE_mary_screws_on_hand_l396_39621

def screws_needed (sections : ℕ) (screws_per_section : ℕ) : ℕ :=
  sections * screws_per_section

theorem mary_screws_on_hand 
  (sections : ℕ) 
  (screws_per_section : ℕ) 
  (buy_ratio : ℕ) 
  (h1 : sections = 4) 
  (h2 : screws_per_section = 6) 
  (h3 : buy_ratio = 2) :
  ∃ (initial_screws : ℕ), 
    initial_screws + buy_ratio * initial_screws = screws_needed sections screws_per_section ∧ 
    initial_screws = 8 :=
by sorry

end NUMINAMATH_CALUDE_mary_screws_on_hand_l396_39621


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l396_39687

theorem arithmetic_geometric_sequence_ratio (d : ℝ) (h : d ≠ 0) :
  let a : ℕ → ℝ := λ n => d * n.pred
  (a 1 * a 9 = (a 3)^2) →
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5/8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l396_39687


namespace NUMINAMATH_CALUDE_divisibility_implies_inequality_l396_39626

theorem divisibility_implies_inequality (k m : ℕ+) (h1 : k > m) 
  (h2 : (k^3 - m^3) ∣ (k * m * (k^2 - m^2))) : 
  (k - m)^3 > 3 * k * m := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_inequality_l396_39626


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l396_39614

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are parallel
def linesParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- The main theorem
theorem parallel_line_through_point :
  let A : Point2D := ⟨-1, 0⟩
  let l1 : Line2D := ⟨2, -1, 1⟩
  let l2 : Line2D := ⟨2, -1, 2⟩
  pointOnLine A l2 ∧ linesParallel l1 l2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l396_39614


namespace NUMINAMATH_CALUDE_subtract_point_five_from_forty_seven_point_two_l396_39674

theorem subtract_point_five_from_forty_seven_point_two : 47.2 - 0.5 = 46.7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_point_five_from_forty_seven_point_two_l396_39674


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l396_39661

-- Define the function f(x) = x³ + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- State the theorem
theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  -- The proof would go here, but we're using sorry as instructed
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l396_39661


namespace NUMINAMATH_CALUDE_chessboard_paradox_l396_39680

/-- Represents a part of the chessboard -/
structure ChessboardPart where
  cells : ℕ
  deriving Repr

/-- Represents the chessboard -/
structure Chessboard where
  parts : List ChessboardPart
  totalCells : ℕ
  deriving Repr

/-- Function to rearrange parts of the chessboard -/
def rearrange (c : Chessboard) : Chessboard :=
  c -- Placeholder for rearrangement logic

theorem chessboard_paradox (c : Chessboard) 
  (h1 : c.parts.length = 4)
  (h2 : c.totalCells = 64) :
  (rearrange c).totalCells = 64 :=
sorry

end NUMINAMATH_CALUDE_chessboard_paradox_l396_39680


namespace NUMINAMATH_CALUDE_calculate_expression_l396_39617

theorem calculate_expression : -5 + 2 * (-3) + (-12) / (-2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l396_39617


namespace NUMINAMATH_CALUDE_no_right_angle_in_sequence_l396_39625

/-- Represents a triangle with three angles -/
structure Triangle where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { angleA := t.angleA, angleB := t.angleB, angleC := t.angleC }

/-- The original triangle ABC -/
def originalTriangle : Triangle :=
  { angleA := 59, angleB := 61, angleC := 60 }

/-- Generates the nth triangle in the sequence -/
def nthTriangle (n : ℕ) : Triangle :=
  match n with
  | 0 => originalTriangle
  | n+1 => nextTriangle (nthTriangle n)

theorem no_right_angle_in_sequence :
  ∀ n : ℕ, (nthTriangle n).angleA ≠ 90 ∧ (nthTriangle n).angleB ≠ 90 ∧ (nthTriangle n).angleC ≠ 90 :=
sorry

end NUMINAMATH_CALUDE_no_right_angle_in_sequence_l396_39625


namespace NUMINAMATH_CALUDE_lavinia_son_katie_daughter_age_ratio_l396_39664

/-- Proves that the ratio of Lavinia's son's age to Katie's daughter's age is 2:1 given the specified conditions. -/
theorem lavinia_son_katie_daughter_age_ratio :
  ∀ (katie_daughter_age : ℕ) 
    (lavinia_daughter_age : ℕ) 
    (lavinia_son_age : ℕ),
  katie_daughter_age = 12 →
  lavinia_daughter_age = katie_daughter_age - 10 →
  lavinia_son_age = lavinia_daughter_age + 22 →
  ∃ (k : ℕ), k * katie_daughter_age = lavinia_son_age →
  lavinia_son_age / katie_daughter_age = 2 :=
by sorry

end NUMINAMATH_CALUDE_lavinia_son_katie_daughter_age_ratio_l396_39664


namespace NUMINAMATH_CALUDE_train_passengers_l396_39649

theorem train_passengers (P : ℕ) : 
  (((P - P / 3 + 280) / 2 + 12) = 248) → P = 288 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l396_39649


namespace NUMINAMATH_CALUDE_car_travel_time_l396_39631

theorem car_travel_time (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) 
  (h1 : distance = 270)
  (h2 : new_speed = 30)
  (h3 : time_ratio = 3/2)
  : ∃ (initial_time : ℝ), 
    initial_time = 6 ∧ 
    distance = new_speed * (initial_time * time_ratio) :=
by sorry

end NUMINAMATH_CALUDE_car_travel_time_l396_39631


namespace NUMINAMATH_CALUDE_cubic_root_quadratic_coefficient_l396_39681

theorem cubic_root_quadratic_coefficient 
  (A B C : ℝ) 
  (r s : ℝ) 
  (h1 : A ≠ 0)
  (h2 : A * r^2 + B * r + C = 0)
  (h3 : A * s^2 + B * s + C = 0) :
  ∃ (p q : ℝ), r^3^2 + p * r^3 + q = 0 ∧ s^3^2 + p * s^3 + q = 0 ∧ 
  p = (B^3 - 3*A*B*C + 2*A*C^2) / A^3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_quadratic_coefficient_l396_39681


namespace NUMINAMATH_CALUDE_winning_scores_count_l396_39675

/-- Represents a cross country meet with specific rules --/
structure CrossCountryMeet where
  runners_per_team : Nat
  total_runners : Nat
  min_score : Nat
  max_score : Nat

/-- Calculates the total score of all runners --/
def total_meet_score (meet : CrossCountryMeet) : Nat :=
  meet.total_runners * (meet.total_runners + 1) / 2

/-- Defines a valid cross country meet with given parameters --/
def valid_meet : CrossCountryMeet :=
  { runners_per_team := 6
  , total_runners := 12
  , min_score := 21
  , max_score := 38 }

/-- Theorem stating the number of possible winning scores --/
theorem winning_scores_count (meet : CrossCountryMeet) 
  (h1 : meet = valid_meet) 
  (h2 : meet.total_runners = 2 * meet.runners_per_team) 
  (h3 : total_meet_score meet = 78) : 
  (meet.max_score - meet.min_score + 1 : Nat) = 18 := by
  sorry

end NUMINAMATH_CALUDE_winning_scores_count_l396_39675


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_zero_is_exists_lt_zero_l396_39628

theorem negation_of_forall_geq_zero_is_exists_lt_zero :
  (¬ ∀ x : ℝ, x^2 + x ≥ 0) ↔ (∃ x : ℝ, x^2 + x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_zero_is_exists_lt_zero_l396_39628


namespace NUMINAMATH_CALUDE_projectile_height_l396_39600

theorem projectile_height (t : ℝ) : 
  (∃ t₀ : ℝ, t₀ > 0 ∧ -4.9 * t₀^2 + 30 * t₀ = 35 ∧ 
   ∀ t' : ℝ, t' > 0 ∧ -4.9 * t'^2 + 30 * t' = 35 → t₀ ≤ t') → 
  t = 10/7 := by
sorry

end NUMINAMATH_CALUDE_projectile_height_l396_39600


namespace NUMINAMATH_CALUDE_homework_problem_count_l396_39669

theorem homework_problem_count (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) : 
  math_pages = 2 → reading_pages = 4 → problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 := by
sorry

end NUMINAMATH_CALUDE_homework_problem_count_l396_39669


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l396_39637

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {2, 3, 4}

-- Define set B
def B : Set Nat := {4, 5}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = A ∩ (U \ B) := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l396_39637


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l396_39671

theorem polynomial_product_expansion :
  let p₁ : Polynomial ℝ := 5 * X^2 + 3 * X - 4
  let p₂ : Polynomial ℝ := 6 * X^3 + 2 * X^2 - X + 7
  p₁ * p₂ = 30 * X^5 + 28 * X^4 - 23 * X^3 + 24 * X^2 + 25 * X - 28 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l396_39671


namespace NUMINAMATH_CALUDE_sqrt_10_plus_2_range_l396_39672

theorem sqrt_10_plus_2_range : 5 < Real.sqrt 10 + 2 ∧ Real.sqrt 10 + 2 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_10_plus_2_range_l396_39672


namespace NUMINAMATH_CALUDE_hugo_tom_box_folding_l396_39682

/-- The number of small boxes Hugo and Tom fold together -/
def small_boxes : ℕ := 4200

/-- The time it takes Hugo to fold a small box (in seconds) -/
def hugo_small_time : ℕ := 3

/-- The time it takes Tom to fold a small or medium box (in seconds) -/
def tom_box_time : ℕ := 4

/-- The total time Hugo and Tom spend folding boxes (in seconds) -/
def total_time : ℕ := 7200

/-- The number of medium boxes Hugo and Tom fold together -/
def medium_boxes : ℕ := 1800

theorem hugo_tom_box_folding :
  small_boxes = (total_time / hugo_small_time) + (total_time / tom_box_time) :=
sorry

end NUMINAMATH_CALUDE_hugo_tom_box_folding_l396_39682


namespace NUMINAMATH_CALUDE_jelly_beans_initial_amount_l396_39654

theorem jelly_beans_initial_amount :
  ∀ (initial_amount eaten_amount : ℕ) 
    (num_piles pile_weight : ℕ),
  eaten_amount = 6 →
  num_piles = 3 →
  pile_weight = 10 →
  initial_amount = eaten_amount + num_piles * pile_weight →
  initial_amount = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_jelly_beans_initial_amount_l396_39654


namespace NUMINAMATH_CALUDE_product_equality_l396_39624

theorem product_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (a * b + b * c + a * c) * ((a * b)⁻¹ + (b * c)⁻¹ + (a * c)⁻¹) = 
  (a * b + b * c + a * c)^2 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l396_39624


namespace NUMINAMATH_CALUDE_even_function_implies_k_equals_one_l396_39641

/-- A function f is even if f(-x) = f(x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = kx^2 + (k-1)x + 3 --/
def f (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + (k - 1) * x + 3

/-- If f(x) = kx^2 + (k-1)x + 3 is an even function, then k = 1 --/
theorem even_function_implies_k_equals_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_k_equals_one_l396_39641


namespace NUMINAMATH_CALUDE_floor_sqrt_24_squared_l396_39646

theorem floor_sqrt_24_squared : ⌊Real.sqrt 24⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_24_squared_l396_39646
