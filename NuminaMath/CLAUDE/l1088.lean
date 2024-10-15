import Mathlib

namespace NUMINAMATH_CALUDE_g_composition_sqrt3_l1088_108877

noncomputable def g (b c : ℝ) (x : ℝ) : ℝ := b * x + c * x^3 - Real.sqrt 3

theorem g_composition_sqrt3 (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  g b c (g b c (Real.sqrt 3)) = -Real.sqrt 3 → b = 0 ∧ c = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_sqrt3_l1088_108877


namespace NUMINAMATH_CALUDE_jane_inspection_fraction_l1088_108881

theorem jane_inspection_fraction :
  ∀ (P : ℝ) (J : ℝ),
    P > 0 →
    J > 0 →
    J < 1 →
    0.005 * (1 - J) * P + 0.008 * J * P = 0.0075 * P →
    J = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_jane_inspection_fraction_l1088_108881


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1088_108873

/-- Given a geometric sequence {a_n} where a₁ = x, a₂ = x-1, and a₃ = 2x-2,
    prove that the general term is a_n = -2^(n-1) -/
theorem geometric_sequence_general_term (x : ℝ) (a : ℕ → ℝ) (h1 : a 1 = x) (h2 : a 2 = x - 1) (h3 : a 3 = 2*x - 2) :
  ∀ n : ℕ, a n = -2^(n-1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1088_108873


namespace NUMINAMATH_CALUDE_chocolate_box_theorem_l1088_108834

/-- Represents a box of chocolates -/
structure ChocolateBox where
  initial_count : ℕ
  rows : ℕ
  columns : ℕ

/-- The state of the box after each rearrangement -/
inductive BoxState
  | Initial
  | AfterFirstRearrange
  | AfterSecondRearrange
  | Final

/-- Function to calculate the number of chocolates at each state -/
def chocolates_at_state (box : ChocolateBox) (state : BoxState) : ℕ :=
  match state with
  | BoxState.Initial => box.initial_count
  | BoxState.AfterFirstRearrange => 3 * box.columns - 1
  | BoxState.AfterSecondRearrange => 5 * box.rows - 1
  | BoxState.Final => box.initial_count / 3

theorem chocolate_box_theorem (box : ChocolateBox) :
  chocolates_at_state box BoxState.Initial = 60 ∧
  chocolates_at_state box BoxState.Initial - chocolates_at_state box BoxState.AfterFirstRearrange = 25 :=
by sorry


end NUMINAMATH_CALUDE_chocolate_box_theorem_l1088_108834


namespace NUMINAMATH_CALUDE_mean_of_four_numbers_l1088_108874

theorem mean_of_four_numbers (a b c d : ℚ) (h : a + b + c + d = 3/4) : 
  (a + b + c + d) / 4 = 3/16 := by
sorry

end NUMINAMATH_CALUDE_mean_of_four_numbers_l1088_108874


namespace NUMINAMATH_CALUDE_complex_modulus_product_range_l1088_108865

theorem complex_modulus_product_range (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs (z₁ + z₂) = 4)
  (h₂ : Complex.abs (z₁ - z₂) = 3) :
  7/4 ≤ Complex.abs (z₁ * z₂) ∧ Complex.abs (z₁ * z₂) ≤ 25/4 :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_product_range_l1088_108865


namespace NUMINAMATH_CALUDE_cos_negative_1500_degrees_l1088_108888

theorem cos_negative_1500_degrees : Real.cos ((-1500 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_1500_degrees_l1088_108888


namespace NUMINAMATH_CALUDE_error_percentage_squared_vs_multiplied_by_eight_l1088_108859

theorem error_percentage_squared_vs_multiplied_by_eight (x : ℝ) (h : x > 0) :
  let correct_result := 8 * x
  let incorrect_result := x ^ 2
  let error := |incorrect_result - correct_result|
  let error_percentage := error / correct_result * 100
  error_percentage = |x - 8| / 8 * 100 := by sorry

end NUMINAMATH_CALUDE_error_percentage_squared_vs_multiplied_by_eight_l1088_108859


namespace NUMINAMATH_CALUDE_julio_bonus_l1088_108847

/-- Calculates Julio's bonus given his commission rate, customer numbers, salary, and total earnings -/
def calculate_bonus (commission_rate : ℕ) (customers_week1 : ℕ) (salary : ℕ) (total_earnings : ℕ) : ℕ :=
  let customers_week2 := 2 * customers_week1
  let customers_week3 := 3 * customers_week1
  let total_customers := customers_week1 + customers_week2 + customers_week3
  let total_commission := commission_rate * total_customers
  total_earnings - salary - total_commission

/-- Theorem stating that Julio's bonus is $50 given the problem conditions -/
theorem julio_bonus :
  calculate_bonus 1 35 500 760 = 50 := by
  sorry

end NUMINAMATH_CALUDE_julio_bonus_l1088_108847


namespace NUMINAMATH_CALUDE_pants_cost_is_correct_l1088_108846

/-- The cost of pants given total payment, cost of shirt, and change received -/
def cost_of_pants (total_payment shirt_cost change : ℚ) : ℚ :=
  total_payment - shirt_cost - change

/-- Theorem stating the cost of pants is $9.24 given the problem conditions -/
theorem pants_cost_is_correct :
  let total_payment : ℚ := 20
  let shirt_cost : ℚ := 8.25
  let change : ℚ := 2.51
  cost_of_pants total_payment shirt_cost change = 9.24 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_is_correct_l1088_108846


namespace NUMINAMATH_CALUDE_complement_union_A_B_complement_A_inter_B_l1088_108887

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- State the theorems to be proved
theorem complement_union_A_B : 
  (Set.univ : Set ℝ) \ (A ∪ B) = {x | x ≤ 2 ∨ x ≥ 10} := by sorry

theorem complement_A_inter_B : 
  ((Set.univ : Set ℝ) \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_complement_A_inter_B_l1088_108887


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l1088_108810

theorem easter_egg_hunt (total_eggs : ℕ) (club_house_eggs : ℕ) (town_hall_eggs : ℕ) 
  (h1 : total_eggs = 80)
  (h2 : club_house_eggs = 40)
  (h3 : town_hall_eggs = 15) :
  ∃ park_eggs : ℕ, 
    park_eggs = total_eggs - club_house_eggs - town_hall_eggs ∧ 
    park_eggs = 25 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l1088_108810


namespace NUMINAMATH_CALUDE_positive_root_of_cubic_l1088_108882

theorem positive_root_of_cubic (x : ℝ) :
  x = 2 + Real.sqrt 2 →
  x^3 - 4*x^2 + x - 2*Real.sqrt 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_positive_root_of_cubic_l1088_108882


namespace NUMINAMATH_CALUDE_cat_finishes_food_on_day_l1088_108837

/-- Represents the days of the week -/
inductive Day : Type
| monday : Day
| tuesday : Day
| wednesday : Day
| thursday : Day
| friday : Day
| saturday : Day
| sunday : Day

/-- Calculates the number of days since Monday -/
def daysSinceMonday (d : Day) : ℕ :=
  match d with
  | Day.monday => 0
  | Day.tuesday => 1
  | Day.wednesday => 2
  | Day.thursday => 3
  | Day.friday => 4
  | Day.saturday => 5
  | Day.sunday => 6

/-- The amount of food the cat eats in the morning (in cans) -/
def morningMeal : ℚ := 2/5

/-- The amount of food the cat eats in the evening (in cans) -/
def eveningMeal : ℚ := 1/6

/-- The total number of cans in the box -/
def totalCans : ℕ := 10

/-- The day on which the cat finishes all the food -/
def finishDay : Day := Day.saturday

/-- Theorem stating that the cat finishes all the food on the specified day -/
theorem cat_finishes_food_on_day :
  (morningMeal + eveningMeal) * (daysSinceMonday finishDay + 1 : ℚ) > totalCans ∧
  (morningMeal + eveningMeal) * (daysSinceMonday finishDay : ℚ) ≤ totalCans :=
by sorry


end NUMINAMATH_CALUDE_cat_finishes_food_on_day_l1088_108837


namespace NUMINAMATH_CALUDE_line_perp_plane_sufficient_condition_l1088_108892

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- Theorem statement
theorem line_perp_plane_sufficient_condition 
  (m n : Line) (α : Plane) :
  para m n → perp n α → perp m α :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_sufficient_condition_l1088_108892


namespace NUMINAMATH_CALUDE_zero_sequence_arithmetic_not_geometric_l1088_108896

-- Define the sequence
def a : ℕ → ℝ
  | _ => 0

-- Theorem statement
theorem zero_sequence_arithmetic_not_geometric :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧ 
  ¬(∀ n m : ℕ, a n ≠ 0 → a (n + 1) / a n = a (m + 1) / a m) :=
by sorry

end NUMINAMATH_CALUDE_zero_sequence_arithmetic_not_geometric_l1088_108896


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_14_l1088_108869

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_14 :
  last_two_digits (sum_factorials 14) = last_two_digits 409113 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_14_l1088_108869


namespace NUMINAMATH_CALUDE_quadratic_equation_with_opposite_roots_l1088_108860

theorem quadratic_equation_with_opposite_roots (x y : ℝ) :
  x^2 - 6*x + 9 = -|y - 1| →
  ∃ (a b c : ℝ), a ≠ 0 ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 ∧
  a = 1 ∧ b = -4 ∧ c = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_opposite_roots_l1088_108860


namespace NUMINAMATH_CALUDE_remainder_relationship_l1088_108897

theorem remainder_relationship (M M' N D S S' s s' : ℕ) : 
  M > M' →
  M % D = S →
  M' % D = S' →
  (M^2 * M') % D = s →
  N^2 % D = s' →
  (∃ M M' N D S S' s s' : ℕ, s = s') ∧
  (∃ M M' N D S S' s s' : ℕ, s < s') :=
by sorry

end NUMINAMATH_CALUDE_remainder_relationship_l1088_108897


namespace NUMINAMATH_CALUDE_scientific_notation_43000000_l1088_108868

theorem scientific_notation_43000000 :
  (43000000 : ℝ) = 4.3 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_43000000_l1088_108868


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_on_unit_interval_l1088_108806

/-- The minimum value of the maximum absolute value of a quadratic function on [-1, 1] -/
theorem min_max_abs_quadratic_on_unit_interval :
  ∃ (F : ℝ), F = 1/2 ∧ 
  (∀ (a b : ℝ) (f : ℝ → ℝ), 
    (∀ x, f x = x^2 + a*x + b) → 
    (∀ x, |x| ≤ 1 → |f x| ≤ F) ∧
    (∃ a b : ℝ, ∃ x, |x| ≤ 1 ∧ |f x| = F)) :=
sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_on_unit_interval_l1088_108806


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1088_108876

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Defines a line in 2D space using the equation y = mx + b -/
structure Line where
  m : ℚ  -- slope
  b : ℚ  -- y-intercept

def line1 : Line := { m := 3, b := 0 }
def line2 : Line := { m := -7, b := 5 }

theorem intersection_of_lines (l1 l2 : Line) : 
  ∃! p : IntersectionPoint, 
    p.y = l1.m * p.x + l1.b ∧ 
    p.y = l2.m * p.x + l2.b := by
  sorry

#check intersection_of_lines line1 line2

end NUMINAMATH_CALUDE_intersection_of_lines_l1088_108876


namespace NUMINAMATH_CALUDE_book_selection_l1088_108875

theorem book_selection (n m k : ℕ) (h1 : n = 7) (h2 : m = 5) (h3 : k = 3) :
  (Nat.choose (n - 2) k) = (Nat.choose m k) :=
by sorry

end NUMINAMATH_CALUDE_book_selection_l1088_108875


namespace NUMINAMATH_CALUDE_andy_lateness_l1088_108844

structure TravelDelay where
  normalTime : Nat
  redLights : Nat
  redLightDelay : Nat
  constructionDelay : Nat
  detourDelay : Nat
  storeDelay : Nat
  trafficDelay : Nat
  departureTime : Nat
  schoolStartTime : Nat

def calculateLateness (delay : TravelDelay) : Nat :=
  let totalDelay := delay.normalTime +
                    delay.redLights * delay.redLightDelay +
                    delay.constructionDelay +
                    delay.detourDelay +
                    delay.storeDelay +
                    delay.trafficDelay
  let arrivalTime := delay.departureTime + totalDelay
  if arrivalTime > delay.schoolStartTime then
    arrivalTime - delay.schoolStartTime
  else
    0

theorem andy_lateness (delay : TravelDelay)
  (h1 : delay.normalTime = 30)
  (h2 : delay.redLights = 4)
  (h3 : delay.redLightDelay = 3)
  (h4 : delay.constructionDelay = 10)
  (h5 : delay.detourDelay = 7)
  (h6 : delay.storeDelay = 5)
  (h7 : delay.trafficDelay = 15)
  (h8 : delay.departureTime = 435)  -- 7:15 AM in minutes since midnight
  (h9 : delay.schoolStartTime = 480)  -- 8:00 AM in minutes since midnight
  : calculateLateness delay = 34 := by
  sorry


end NUMINAMATH_CALUDE_andy_lateness_l1088_108844


namespace NUMINAMATH_CALUDE_power_of_ten_plus_one_divisibility_not_always_divisible_by_nine_l1088_108883

theorem power_of_ten_plus_one_divisibility (n : ℕ) :
  (9 ∣ 10^n + 1) → (9 ∣ 10^(n+1) + 1) :=
by sorry

theorem not_always_divisible_by_nine :
  ∃ n : ℕ, ¬(9 ∣ 10^n + 1) :=
by sorry

end NUMINAMATH_CALUDE_power_of_ten_plus_one_divisibility_not_always_divisible_by_nine_l1088_108883


namespace NUMINAMATH_CALUDE_inequality_proof_l1088_108841

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (1 + a^2) + Real.sqrt (1 + b^2) ≥ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1088_108841


namespace NUMINAMATH_CALUDE_marks_speed_l1088_108893

/-- Given a distance of 24 miles and a time of 4 hours, the speed is 6 miles per hour. -/
theorem marks_speed (distance : ℝ) (time : ℝ) (h1 : distance = 24) (h2 : time = 4) :
  distance / time = 6 := by
  sorry

end NUMINAMATH_CALUDE_marks_speed_l1088_108893


namespace NUMINAMATH_CALUDE_monotone_increasing_implies_m_leq_half_m_leq_half_not_implies_monotone_increasing_l1088_108804

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x

-- Define the property of f being monotonically increasing on (0, +∞)
def is_monotone_increasing_on_positive (m : ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f m x < f m y

-- Theorem stating that if f is monotonically increasing on (0, +∞), then m ≤ 1/2
theorem monotone_increasing_implies_m_leq_half (m : ℝ) :
  is_monotone_increasing_on_positive m → m ≤ 1/2 :=
sorry

-- Theorem stating that m ≤ 1/2 does not necessarily imply f is monotonically increasing on (0, +∞)
theorem m_leq_half_not_implies_monotone_increasing :
  ∃ m, m ≤ 1/2 ∧ ¬is_monotone_increasing_on_positive m :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_implies_m_leq_half_m_leq_half_not_implies_monotone_increasing_l1088_108804


namespace NUMINAMATH_CALUDE_equality_from_fraction_equality_l1088_108845

theorem equality_from_fraction_equality (a b c d : ℝ) :
  (a + b) / (c + d) = (b + c) / (a + d) ∧ 
  (a + b) / (c + d) ≠ -1 →
  a = c :=
by sorry

end NUMINAMATH_CALUDE_equality_from_fraction_equality_l1088_108845


namespace NUMINAMATH_CALUDE_complex_number_properties_l1088_108890

/-- Given a complex number z where z + 1/z is real, this theorem proves:
    1. The value of z that minimizes |z + 2 - i|
    2. The minimum value of |z + 2 - i|
    3. u = (1 - z) / (1 + z) is purely imaginary -/
theorem complex_number_properties (z : ℂ) 
    (h : (z + z⁻¹).im = 0) : 
    ∃ (min_z : ℂ) (min_val : ℝ),
    (min_z = -2 * Real.sqrt 5 / 5 + (Real.sqrt 5 / 5) * Complex.I) ∧
    (min_val = Real.sqrt 5 - 1) ∧
    (∀ w : ℂ, Complex.abs (w + 2 - Complex.I) ≥ min_val) ∧
    (Complex.abs (min_z + 2 - Complex.I) = min_val) ∧
    ((1 - z) / (1 + z)).re = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1088_108890


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1088_108850

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1088_108850


namespace NUMINAMATH_CALUDE_canoe_current_speed_l1088_108813

/-- Represents the speed of a canoe in still water and the speed of the current. -/
structure CanoeSpeedData where
  canoe_speed : ℝ
  current_speed : ℝ

/-- Calculates the effective speed of a canoe given the canoe's speed in still water and the current speed. -/
def effective_speed (upstream : Bool) (data : CanoeSpeedData) : ℝ :=
  if upstream then data.canoe_speed - data.current_speed else data.canoe_speed + data.current_speed

/-- Theorem stating that given the conditions of the canoe problem, the speed of the current is 7 miles per hour. -/
theorem canoe_current_speed : 
  ∀ (data : CanoeSpeedData),
    (effective_speed true data) * 6 = 12 →
    (effective_speed false data) * 0.75 = 12 →
    data.current_speed = 7 := by
  sorry


end NUMINAMATH_CALUDE_canoe_current_speed_l1088_108813


namespace NUMINAMATH_CALUDE_root_product_property_l1088_108856

theorem root_product_property (a b : ℂ) : 
  (a^4 + a^3 - 1 = 0) → (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_root_product_property_l1088_108856


namespace NUMINAMATH_CALUDE_mean_of_combined_sets_l1088_108880

theorem mean_of_combined_sets :
  ∀ (set1 set2 : List ℝ),
    set1.length = 7 →
    set2.length = 8 →
    (set1.sum / set1.length : ℝ) = 15 →
    (set2.sum / set2.length : ℝ) = 20 →
    ((set1 ++ set2).sum / (set1 ++ set2).length : ℝ) = 17.67 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_combined_sets_l1088_108880


namespace NUMINAMATH_CALUDE_midpoint_property_implies_linear_l1088_108822

/-- A function satisfying the midpoint property -/
def HasMidpointProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x + y) / 2) = (f x + f y) / 2

/-- The main theorem: continuous functions with the midpoint property are linear -/
theorem midpoint_property_implies_linear
  (f : ℝ → ℝ) (hf : Continuous f) (hm : HasMidpointProperty f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_midpoint_property_implies_linear_l1088_108822


namespace NUMINAMATH_CALUDE_group_size_l1088_108858

theorem group_size (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 3 ∧ old_weight = 65 ∧ new_weight = 89 →
  (new_weight - old_weight) / average_increase = 8 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l1088_108858


namespace NUMINAMATH_CALUDE_right_triangle_minimum_side_l1088_108838

theorem right_triangle_minimum_side : ∃ (s : ℕ), 
  (s ≥ 25) ∧ 
  (∀ (t : ℕ), t < 25 → ¬(7^2 + 24^2 = t^2)) ∧
  (7^2 + 24^2 = s^2) ∧
  (7 + 24 > s) ∧ (24 + s > 7) ∧ (7 + s > 24) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_minimum_side_l1088_108838


namespace NUMINAMATH_CALUDE_coin_sum_theorem_l1088_108840

def coin_values : List Nat := [5, 10, 25, 50]

def is_valid_sum (n : Nat) : Prop :=
  ∃ (a b c d e : Nat), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧
    a + b + c + d + e = n

theorem coin_sum_theorem :
  ¬(is_valid_sum 40) ∧ 
  (is_valid_sum 65) ∧ 
  (is_valid_sum 85) ∧ 
  (is_valid_sum 105) ∧ 
  (is_valid_sum 130) := by
  sorry

end NUMINAMATH_CALUDE_coin_sum_theorem_l1088_108840


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l1088_108811

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l1088_108811


namespace NUMINAMATH_CALUDE_unknown_number_is_six_l1088_108832

theorem unknown_number_is_six : ∃ x : ℚ, (2 / 3) * x + 6 = 10 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_is_six_l1088_108832


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l1088_108808

theorem ratio_of_percentages (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hR : R = 0.6 * P)
  (hN : N = 0.5 * R) :
  M / N = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l1088_108808


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l1088_108899

theorem mean_equality_implies_z_value :
  let mean1 := (8 + 12 + 24) / 3
  let mean2 := (16 + z) / 2
  mean1 = mean2 → z = 40 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l1088_108899


namespace NUMINAMATH_CALUDE_triangle_area_and_square_coverage_l1088_108895

/-- Given a triangle with side lengths 9, 40, and 41, prove its area and the fraction it covers of a square with side length 41. -/
theorem triangle_area_and_square_coverage :
  ∃ (triangle_area : ℝ) (square_area : ℝ) (coverage_fraction : ℚ),
    triangle_area = 180 ∧
    square_area = 41 ^ 2 ∧
    coverage_fraction = 180 / 1681 ∧
    (9 : ℝ) ^ 2 + 40 ^ 2 = 41 ^ 2 ∧
    triangle_area = (1 / 2 : ℝ) * 9 * 40 ∧
    coverage_fraction = triangle_area / square_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_and_square_coverage_l1088_108895


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_l1088_108812

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | -1 < x ∧ x < 3} := by sorry

-- Theorem for ∁ℝA
theorem complement_A : (Set.univ : Set ℝ) \ A = {x | x ≤ -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_l1088_108812


namespace NUMINAMATH_CALUDE_jenny_max_earnings_l1088_108848

def neighborhood_A_homes : ℕ := 10
def neighborhood_A_boxes_per_home : ℕ := 2
def neighborhood_B_homes : ℕ := 5
def neighborhood_B_boxes_per_home : ℕ := 5
def price_per_box : ℕ := 2

def total_boxes_A : ℕ := neighborhood_A_homes * neighborhood_A_boxes_per_home
def total_boxes_B : ℕ := neighborhood_B_homes * neighborhood_B_boxes_per_home

def max_earnings : ℕ := max total_boxes_A total_boxes_B * price_per_box

theorem jenny_max_earnings :
  max_earnings = 50 := by
  sorry

end NUMINAMATH_CALUDE_jenny_max_earnings_l1088_108848


namespace NUMINAMATH_CALUDE_fraction_order_l1088_108878

theorem fraction_order : 
  (22 : ℚ) / 19 < (18 : ℚ) / 15 ∧ 
  (18 : ℚ) / 15 < (21 : ℚ) / 17 ∧ 
  (21 : ℚ) / 17 < (20 : ℚ) / 16 := by
sorry

end NUMINAMATH_CALUDE_fraction_order_l1088_108878


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l1088_108867

-- Define the quadratic equations
def quadratic1 (a x : ℝ) : ℝ := a * x^2 + (a + 1) * x - 2
def quadratic2 (a x : ℝ) : ℝ := (1 - a) * x^2 + (a + 1) * x - 2

-- Define the conditions for real solutions
def realSolutions1 (a : ℝ) : Prop :=
  a < -5 - 2 * Real.sqrt 6 ∨ (2 * Real.sqrt 6 - 5 < a ∧ a < 0) ∨ a > 0

def realSolutions2 (a : ℝ) : Prop :=
  a < 1 ∨ (1 < a ∧ a < 3) ∨ a > 3

-- Theorem statement
theorem quadratic_real_solutions :
  ∀ a : ℝ,
    (∃ x : ℝ, quadratic1 a x = 0) ↔ realSolutions1 a ∧
    (∃ x : ℝ, quadratic2 a x = 0) ↔ realSolutions2 a :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l1088_108867


namespace NUMINAMATH_CALUDE_fraction_comparison_l1088_108802

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a / b < c / d) 
  (h2 : b > d) 
  (h3 : d > 0) : 
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1088_108802


namespace NUMINAMATH_CALUDE_no_solution_iff_a_in_range_l1088_108894

/-- The equation has no solutions if and only if a is in the specified range -/
theorem no_solution_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, 5*|x - 4*a| + |x - a^2| + 4*x - 4*a ≠ 0) ↔ 
  (a < -8 ∨ a > 0) := by
sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_in_range_l1088_108894


namespace NUMINAMATH_CALUDE_transport_cost_calculation_l1088_108830

/-- The transport cost for Ramesh's refrigerator purchase --/
def transport_cost : ℕ := by sorry

/-- The labelled price of the refrigerator before discount --/
def labelled_price : ℕ := by sorry

/-- The discounted price Ramesh paid for the refrigerator --/
def discounted_price : ℕ := 17500

/-- The installation cost --/
def installation_cost : ℕ := 250

/-- The selling price to earn 10% profit without discount --/
def selling_price : ℕ := 24475

/-- The discount rate applied to the labelled price --/
def discount_rate : ℚ := 1/5

/-- The profit rate desired if no discount was offered --/
def profit_rate : ℚ := 1/10

theorem transport_cost_calculation :
  discounted_price = labelled_price * (1 - discount_rate) ∧
  selling_price = labelled_price * (1 + profit_rate) ∧
  transport_cost + discounted_price + installation_cost = selling_price ∧
  transport_cost = 6725 := by sorry

end NUMINAMATH_CALUDE_transport_cost_calculation_l1088_108830


namespace NUMINAMATH_CALUDE_solution_set_equals_interval_l1088_108803

-- Define the ⊕ operation
def circle_plus (a b : ℝ) : ℝ := a * b + 2 * a + b

-- Define the set of x satisfying the inequality
def solution_set : Set ℝ := {x | circle_plus x (x - 2) < 0}

-- Theorem statement
theorem solution_set_equals_interval :
  solution_set = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_interval_l1088_108803


namespace NUMINAMATH_CALUDE_expression_evaluation_l1088_108814

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(y+1) * y^x) / (y^(x+1) * x^y) = x / y := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1088_108814


namespace NUMINAMATH_CALUDE_min_intersection_points_l1088_108854

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of circles -/
structure CircleConfiguration where
  n : ℕ+
  circles : Fin (4 * n) → Circle
  same_radius : ∀ i j, (circles i).radius = (circles j).radius
  no_tangent : ∀ i j, i ≠ j → (circles i).center ≠ (circles j).center ∨ 
               dist (circles i).center (circles j).center ≠ (circles i).radius + (circles j).radius
  intersect_at_least_three : ∀ i, ∃ j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
                             dist (circles i).center (circles j).center < (circles i).radius + (circles j).radius ∧
                             dist (circles i).center (circles k).center < (circles i).radius + (circles k).radius ∧
                             dist (circles i).center (circles l).center < (circles i).radius + (circles l).radius

/-- The number of intersection points in a circle configuration -/
def num_intersection_points (config : CircleConfiguration) : ℕ :=
  sorry

/-- The main theorem: the minimum number of intersection points is 4n -/
theorem min_intersection_points (config : CircleConfiguration) :
  num_intersection_points config ≥ 4 * config.n :=
sorry

end NUMINAMATH_CALUDE_min_intersection_points_l1088_108854


namespace NUMINAMATH_CALUDE_smallest_share_is_five_l1088_108891

/-- Represents the distribution of coins among three children --/
structure CoinDistribution where
  one_franc : ℕ
  five_franc : ℕ
  fifty_cent : ℕ

/-- Checks if the distribution satisfies the problem conditions --/
def valid_distribution (d : CoinDistribution) : Prop :=
  d.one_franc + 5 * d.five_franc + (d.fifty_cent : ℚ) / 2 = 100 ∧
  d.fifty_cent = d.one_franc / 9

/-- Calculates the smallest share among the three children --/
def smallest_share (d : CoinDistribution) : ℚ :=
  min (min (d.one_franc : ℚ) (5 * d.five_franc : ℚ)) ((d.fifty_cent : ℚ) / 2)

/-- Theorem stating the smallest possible share is 5 francs --/
theorem smallest_share_is_five :
  ∀ d : CoinDistribution, valid_distribution d → smallest_share d = 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_share_is_five_l1088_108891


namespace NUMINAMATH_CALUDE_gym_class_group_sizes_l1088_108853

/-- Given a gym class with two groups of students, prove that if the total number of students is 71 and one group has 37 students, then the other group must have 34 students. -/
theorem gym_class_group_sizes (total_students : ℕ) (group1_size : ℕ) (group2_size : ℕ) 
  (h1 : total_students = 71)
  (h2 : group2_size = 37)
  (h3 : total_students = group1_size + group2_size) :
  group1_size = 34 := by
  sorry

end NUMINAMATH_CALUDE_gym_class_group_sizes_l1088_108853


namespace NUMINAMATH_CALUDE_circle_center_sum_l1088_108839

/-- Given a circle with equation x^2 + y^2 - 10x + 4y = -40, 
    the sum of the x and y coordinates of its center is 3. -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 - 10*x + 4*y = -40 → x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1088_108839


namespace NUMINAMATH_CALUDE_quadratic_equation_has_two_distinct_real_roots_l1088_108855

/-- The quadratic equation x^2 - 3x - 1 = 0 has two distinct real roots -/
theorem quadratic_equation_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ - 1 = 0 ∧ x₂^2 - 3*x₂ - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_has_two_distinct_real_roots_l1088_108855


namespace NUMINAMATH_CALUDE_number_relationship_l1088_108823

theorem number_relationship : 
  let a : ℝ := -0.3
  let b : ℝ := (0.3:ℝ)^2
  let c : ℝ := 2^(0.3:ℝ)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_number_relationship_l1088_108823


namespace NUMINAMATH_CALUDE_charity_race_fundraising_l1088_108866

theorem charity_race_fundraising (total_students : ℕ) (group1_students : ℕ) (group1_amount : ℕ) (group2_amount : ℕ) :
  total_students = 30 →
  group1_students = 10 →
  group1_amount = 20 →
  group2_amount = 30 →
  (group1_students * group1_amount) + ((total_students - group1_students) * group2_amount) = 800 :=
by sorry

end NUMINAMATH_CALUDE_charity_race_fundraising_l1088_108866


namespace NUMINAMATH_CALUDE_ticket_distribution_proof_l1088_108805

theorem ticket_distribution_proof (total_tickets : ℕ) (total_amount : ℚ) 
  (price_15 price_10 price_5_5 : ℚ) :
  total_tickets = 22 →
  total_amount = 229 →
  price_15 = 15 →
  price_10 = 10 →
  price_5_5 = (11 : ℚ) / 2 →
  ∃! (x y z : ℕ), 
    x + y + z = total_tickets ∧ 
    price_15 * x + price_10 * y + price_5_5 * z = total_amount ∧
    x = 9 ∧ y = 5 ∧ z = 8 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_proof_l1088_108805


namespace NUMINAMATH_CALUDE_bus_related_time_trip_time_breakdown_l1088_108833

/-- Represents the duration of Luke's trip to London in minutes -/
def total_trip_time : ℕ := 525

/-- Represents the wait time for the first bus in minutes -/
def first_bus_wait : ℕ := 25

/-- Represents the duration of the first bus ride in minutes -/
def first_bus_ride : ℕ := 40

/-- Represents the wait time for the second bus in minutes -/
def second_bus_wait : ℕ := 15

/-- Represents the duration of the second bus ride in minutes -/
def second_bus_ride : ℕ := 10

/-- Represents the walk time to the train station in minutes -/
def walk_time : ℕ := 15

/-- Represents the wait time for the train in minutes -/
def train_wait : ℕ := 2 * walk_time

/-- Represents the duration of the train ride in minutes -/
def train_ride : ℕ := 360

/-- Proves that the total bus-related time is 90 minutes -/
theorem bus_related_time :
  first_bus_wait + first_bus_ride + second_bus_wait + second_bus_ride = 90 :=
by sorry

/-- Proves that the sum of all components equals the total trip time -/
theorem trip_time_breakdown :
  first_bus_wait + first_bus_ride + second_bus_wait + second_bus_ride +
  walk_time + train_wait + train_ride = total_trip_time :=
by sorry

end NUMINAMATH_CALUDE_bus_related_time_trip_time_breakdown_l1088_108833


namespace NUMINAMATH_CALUDE_goats_in_field_l1088_108849

theorem goats_in_field (total : Nat) (cows : Nat) (sheep : Nat) (chickens : Nat) 
  (h1 : total = 900)
  (h2 : cows = 250)
  (h3 : sheep = 310)
  (h4 : chickens = 180) :
  total - (cows + sheep + chickens) = 160 := by
  sorry

end NUMINAMATH_CALUDE_goats_in_field_l1088_108849


namespace NUMINAMATH_CALUDE_stating_smallest_n_with_constant_term_l1088_108884

/-- 
Given a positive integer n and the expression (4x^3 + 1/x^2)^n,
this function returns true if there exists a constant term in the expansion,
and false otherwise.
-/
def has_constant_term (n : ℕ+) : Prop :=
  ∃ r : ℕ, r ≤ n ∧ 3 * n = 5 * r

/-- 
Theorem stating that 5 is the smallest positive integer n 
for which there exists a constant term in the expansion of (4x^3 + 1/x^2)^n.
-/
theorem smallest_n_with_constant_term : 
  (∀ m : ℕ+, m < 5 → ¬has_constant_term m) ∧ has_constant_term 5 :=
sorry

end NUMINAMATH_CALUDE_stating_smallest_n_with_constant_term_l1088_108884


namespace NUMINAMATH_CALUDE_original_function_derivation_l1088_108870

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Rotates a linear function 180° around the origin -/
def rotate180 (f : LinearFunction) : LinearFunction :=
  { slope := -f.slope, intercept := -f.intercept }

/-- Translates a linear function horizontally -/
def translateLeft (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + f.slope * units }

/-- Checks if a linear function passes through two points -/
def passesThrough (f : LinearFunction) (x1 y1 x2 y2 : ℝ) : Prop :=
  f.slope * x1 + f.intercept = y1 ∧ f.slope * x2 + f.intercept = y2

theorem original_function_derivation (k b : ℝ) :
  let f := LinearFunction.mk k b
  let rotated := rotate180 f
  let translated := translateLeft rotated 2
  passesThrough translated (-4) 0 0 2 →
  k = 1/2 ∧ b = -1 := by sorry

end NUMINAMATH_CALUDE_original_function_derivation_l1088_108870


namespace NUMINAMATH_CALUDE_max_grain_mass_l1088_108862

/-- The maximum mass of grain that can be loaded onto a rectangular platform -/
theorem max_grain_mass (length width : Real) (max_angle : Real) (density : Real) :
  length = 10 ∧ 
  width = 5 ∧ 
  max_angle = π / 4 ∧ 
  density = 1200 →
  ∃ (mass : Real),
    mass = 175000 ∧ 
    mass = density * (length * width * (width / 2) / 2 + length * width * (width / 4))
    := by sorry

end NUMINAMATH_CALUDE_max_grain_mass_l1088_108862


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1088_108835

theorem parallel_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, -3]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1088_108835


namespace NUMINAMATH_CALUDE_rats_to_chihuahuas_ratio_l1088_108826

theorem rats_to_chihuahuas_ratio : 
  ∀ (total : ℕ) (rats : ℕ) (chihuahuas : ℕ),
  total = 70 →
  rats = 60 →
  chihuahuas = total - rats →
  ∃ (k : ℕ), rats = k * chihuahuas →
  (rats : ℚ) / chihuahuas = 6 / 1 := by
sorry

end NUMINAMATH_CALUDE_rats_to_chihuahuas_ratio_l1088_108826


namespace NUMINAMATH_CALUDE_cube_root_27_minus_2_l1088_108829

theorem cube_root_27_minus_2 : (27 : ℝ) ^ (1/3) - 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_minus_2_l1088_108829


namespace NUMINAMATH_CALUDE_stating_systematic_sampling_theorem_l1088_108864

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  first_sample : ℕ

/-- 
  Given a systematic sampling scheme and a group number,
  returns the number drawn from that group
-/
def number_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_sample + (group - 1) * (s.population_size / s.sample_size)

/-- 
  Theorem stating that if the number drawn from the 13th group is 101
  in a systematic sampling of 20 from 160, then the number drawn from
  the 3rd group is 21
-/
theorem systematic_sampling_theorem :
  ∀ (s : SystematicSampling),
    s.population_size = 160 →
    s.sample_size = 20 →
    number_in_group s 13 = 101 →
    number_in_group s 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_stating_systematic_sampling_theorem_l1088_108864


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l1088_108851

theorem sqrt_sum_equals_eight : 
  Real.sqrt (18 - 8 * Real.sqrt 2) + Real.sqrt (18 + 8 * Real.sqrt 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l1088_108851


namespace NUMINAMATH_CALUDE_cube_surface_area_l1088_108817

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1728 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 864 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1088_108817


namespace NUMINAMATH_CALUDE_weight_of_b_l1088_108863

/-- Given three weights a, b, and c, prove that b = 33 under the given conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 44 →
  b = 33 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l1088_108863


namespace NUMINAMATH_CALUDE_circle_properties_l1088_108857

/-- A circle with center on the y-axis, radius 1, and passing through (1, 2) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 1

theorem circle_properties :
  ∃ (b : ℝ), 
    (∀ x y : ℝ, circle_equation x y ↔ x^2 + (y - b)^2 = 1) ∧ 
    (0, b) = (0, 2) ∧
    (∀ x y : ℝ, circle_equation x y → (x - 0)^2 + (y - b)^2 = 1) ∧
    circle_equation 1 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1088_108857


namespace NUMINAMATH_CALUDE_average_age_when_youngest_born_l1088_108801

/-- Given a group of people, their average age, and the age of the youngest person,
    calculate the average age of the group when the youngest was born. -/
theorem average_age_when_youngest_born 
  (n : ℕ) -- Total number of people
  (avg : ℝ) -- Current average age
  (youngest : ℝ) -- Age of the youngest person
  (h1 : n = 7) -- There are 7 people
  (h2 : avg = 30) -- The current average age is 30
  (h3 : youngest = 3) -- The youngest person is 3 years old
  : (n * avg - youngest) / (n - 1) = 34.5 := by
  sorry

end NUMINAMATH_CALUDE_average_age_when_youngest_born_l1088_108801


namespace NUMINAMATH_CALUDE_greg_bike_rotations_l1088_108836

/-- Calculates the additional wheel rotations needed to reach a goal distance -/
def additional_rotations_needed (rotations_per_block : ℕ) (goal_blocks : ℕ) (current_rotations : ℕ) : ℕ :=
  rotations_per_block * goal_blocks - current_rotations

theorem greg_bike_rotations :
  let rotations_per_block : ℕ := 200
  let goal_blocks : ℕ := 8
  let current_rotations : ℕ := 600
  additional_rotations_needed rotations_per_block goal_blocks current_rotations = 1000 := by
  sorry

end NUMINAMATH_CALUDE_greg_bike_rotations_l1088_108836


namespace NUMINAMATH_CALUDE_marias_painting_price_l1088_108831

/-- The selling price of Maria's painting --/
def selling_price (brush_cost canvas_cost paint_cost_per_liter paint_liters earnings : ℕ) : ℕ :=
  brush_cost + canvas_cost + paint_cost_per_liter * paint_liters + earnings

/-- Theorem stating the selling price of Maria's painting --/
theorem marias_painting_price :
  selling_price 20 (3 * 20) 8 5 80 = 200 := by
  sorry

end NUMINAMATH_CALUDE_marias_painting_price_l1088_108831


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l1088_108885

/-- Calculates the weighted average profit percentage for cricket bat sales -/
theorem cricket_bat_profit_percentage :
  let bat_a_quantity : ℕ := 5
  let bat_a_cost : ℚ := 900
  let bat_a_profit : ℚ := 300
  let bat_b_quantity : ℕ := 8
  let bat_b_cost : ℚ := 1200
  let bat_b_profit : ℚ := 400
  let bat_c_quantity : ℕ := 3
  let bat_c_cost : ℚ := 1500
  let bat_c_profit : ℚ := 500

  let total_cost : ℚ := bat_a_quantity * bat_a_cost + bat_b_quantity * bat_b_cost + bat_c_quantity * bat_c_cost
  let total_profit : ℚ := bat_a_quantity * bat_a_profit + bat_b_quantity * bat_b_profit + bat_c_quantity * bat_c_profit

  let weighted_avg_profit_percentage : ℚ := (total_profit / total_cost) * 100

  weighted_avg_profit_percentage = 100/3 := by sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l1088_108885


namespace NUMINAMATH_CALUDE_orchids_cut_correct_l1088_108886

/-- The number of red orchids Sally cut from her garden -/
def orchids_cut (initial_red : ℕ) (final_red : ℕ) : ℕ :=
  final_red - initial_red

/-- Theorem stating that the number of orchids Sally cut is the difference between final and initial red orchids -/
theorem orchids_cut_correct (initial_red initial_white final_red : ℕ) 
  (h1 : initial_red = 9)
  (h2 : initial_white = 3)
  (h3 : final_red = 15) :
  orchids_cut initial_red final_red = 6 := by
  sorry

#eval orchids_cut 9 15

end NUMINAMATH_CALUDE_orchids_cut_correct_l1088_108886


namespace NUMINAMATH_CALUDE_full_price_revenue_l1088_108820

def total_tickets : ℕ := 180
def total_revenue : ℕ := 2400

def ticket_revenue (full_price : ℕ) (num_full_price : ℕ) : Prop :=
  ∃ (half_price : ℕ),
    half_price = full_price / 2 ∧
    num_full_price + (total_tickets - num_full_price) = total_tickets ∧
    num_full_price * full_price + (total_tickets - num_full_price) * half_price = total_revenue

theorem full_price_revenue : 
  ∃ (full_price : ℕ) (num_full_price : ℕ), 
    ticket_revenue full_price num_full_price ∧ 
    full_price * num_full_price = 300 :=
by sorry

end NUMINAMATH_CALUDE_full_price_revenue_l1088_108820


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l1088_108828

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The y-coordinate of a point on a line given its x-coordinate -/
def Line.y_at (l : Line) (x : ℝ) : ℝ :=
  l.slope * x + l.y_intercept

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line_through_point 
  (l : Line) (x₀ y₀ : ℝ) : 
  parallel l { slope := -3, y_intercept := 6 } →
  l.y_at x₀ = y₀ →
  x₀ = 3 →
  y₀ = -2 →
  l.y_intercept = 7 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l1088_108828


namespace NUMINAMATH_CALUDE_f_monotone_increasing_interval_l1088_108816

-- Define the function f(x) = x^2 + 2x + 3
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem about the monotonically increasing interval of f
theorem f_monotone_increasing_interval :
  ∃ (a : ℝ), a = -1 ∧
  ∀ (x y : ℝ), x > a → y > x → f y > f x :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_interval_l1088_108816


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1088_108815

/-- 
Given an equilateral triangle with area 100√3 cm², 
prove that its perimeter is 60 cm.
-/
theorem equilateral_triangle_perimeter (A : ℝ) (p : ℝ) : 
  A = 100 * Real.sqrt 3 → p = 60 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1088_108815


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l1088_108889

theorem inequality_and_minimum_value 
  (m n : ℝ) 
  (h_diff : m ≠ n) 
  (h_pos_m : m > 0) 
  (h_pos_n : n > 0) 
  (x y : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) :
  (m^2 / x + n^2 / y > (m + n)^2 / (x + y)) ∧
  (∃ (min_val : ℝ) (min_x : ℝ), 
    min_val = 64 ∧ 
    min_x = 1/8 ∧ 
    (∀ x, 0 < x ∧ x < 1/5 → 5/x + 9/(1-5*x) ≥ min_val) ∧
    (5/min_x + 9/(1-5*min_x) = min_val)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l1088_108889


namespace NUMINAMATH_CALUDE_y_greater_than_one_l1088_108821

theorem y_greater_than_one (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 := by
  sorry

end NUMINAMATH_CALUDE_y_greater_than_one_l1088_108821


namespace NUMINAMATH_CALUDE_parallelogram_iff_midpoints_l1088_108871

-- Define the points
variable (A B C D P Q E F : ℝ × ℝ)

-- Define the conditions
def is_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def on_diagonal (P Q B D : ℝ × ℝ) : Prop := sorry

def point_order (B P Q D : ℝ × ℝ) : Prop := sorry

def equal_segments (B P Q D : ℝ × ℝ) : Prop := sorry

def line_intersection (A P B C E : ℝ × ℝ) : Prop := sorry

def line_intersection' (A Q C D F : ℝ × ℝ) : Prop := sorry

def is_parallelogram (A B C D : ℝ × ℝ) : Prop := sorry

def is_midpoint (E B C : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem parallelogram_iff_midpoints
  (h1 : is_quadrilateral A B C D)
  (h2 : on_diagonal P Q B D)
  (h3 : point_order B P Q D)
  (h4 : equal_segments B P Q D)
  (h5 : line_intersection A P B C E)
  (h6 : line_intersection' A Q C D F) :
  is_parallelogram A B C D ↔ (is_midpoint E B C ∧ is_midpoint F C D) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_iff_midpoints_l1088_108871


namespace NUMINAMATH_CALUDE_percentage_problem_l1088_108872

theorem percentage_problem (x : ℝ) (h : 160 = 320 / 100 * x) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1088_108872


namespace NUMINAMATH_CALUDE_range_of_k_l1088_108800

-- Define the equation
def equation (x y k : ℝ) : Prop := x + y - 6 * Real.sqrt (x + y) + 3 * k = 0

-- Define the condition that the equation represents only one line
def represents_one_line (k : ℝ) : Prop :=
  ∀ x y : ℝ, equation x y k → ∃! (x' y' : ℝ), equation x' y' k ∧ x' = x ∧ y' = y

-- Theorem statement
theorem range_of_k (k : ℝ) :
  represents_one_line k ↔ k = 3 ∨ k < 0 := by sorry

end NUMINAMATH_CALUDE_range_of_k_l1088_108800


namespace NUMINAMATH_CALUDE_bennys_working_hours_l1088_108852

/-- Calculates the total working hours given hours per day and number of days worked -/
def totalWorkingHours (hoursPerDay : ℕ) (daysWorked : ℕ) : ℕ :=
  hoursPerDay * daysWorked

/-- Proves that Benny's total working hours is 18 given the conditions -/
theorem bennys_working_hours :
  let hoursPerDay : ℕ := 3
  let daysWorked : ℕ := 6
  totalWorkingHours hoursPerDay daysWorked = 18 := by
  sorry

end NUMINAMATH_CALUDE_bennys_working_hours_l1088_108852


namespace NUMINAMATH_CALUDE_average_income_calculation_l1088_108809

theorem average_income_calculation (total_customers : ℕ) 
  (wealthy_customers : ℕ) (other_customers : ℕ) 
  (wealthy_avg_income : ℝ) (other_avg_income : ℝ) :
  total_customers = wealthy_customers + other_customers →
  wealthy_customers = 10 →
  other_customers = 40 →
  wealthy_avg_income = 55000 →
  other_avg_income = 42500 →
  (wealthy_customers * wealthy_avg_income + other_customers * other_avg_income) / total_customers = 45000 :=
by sorry

end NUMINAMATH_CALUDE_average_income_calculation_l1088_108809


namespace NUMINAMATH_CALUDE_least_months_to_triple_l1088_108879

def interest_rate : ℝ := 1.05

theorem least_months_to_triple (n : ℕ) : (∀ m : ℕ, m < n → interest_rate ^ m ≤ 3) ∧ interest_rate ^ n > 3 ↔ n = 23 := by
  sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l1088_108879


namespace NUMINAMATH_CALUDE_triangle_problem_l1088_108843

theorem triangle_problem (a b c A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  Real.sqrt 3 * b * Real.sin A - a * Real.cos B - 2 * a = 0 ∧
  b = Real.sqrt 7 ∧
  1/2 * a * b * Real.sin C = Real.sqrt 3 / 2 →
  B = 2 * π / 3 ∧ a + b + c = 3 + Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1088_108843


namespace NUMINAMATH_CALUDE_stratified_sampling_most_reasonable_l1088_108819

/-- Represents the different grade levels in the study -/
inductive GradeLevel
  | Three
  | Six
  | Nine

/-- Represents different sampling methods -/
inductive SamplingMethod
  | LotDrawing
  | Systematic
  | Stratified
  | RandomNumber

/-- Represents the study of visual acuity across different grade levels -/
structure VisualAcuityStudy where
  gradeLevels : List GradeLevel
  sampleProportion : ℝ
  samplingMethod : SamplingMethod

/-- Checks if a sampling method is the most reasonable for a given study -/
def isMostReasonable (study : VisualAcuityStudy) (method : SamplingMethod) : Prop :=
  method = study.samplingMethod ∧
  ∀ otherMethod : SamplingMethod, otherMethod ≠ method → 
    (study.samplingMethod = otherMethod → False)

/-- The main theorem stating that stratified sampling is the most reasonable method for the visual acuity study -/
theorem stratified_sampling_most_reasonable (study : VisualAcuityStudy) :
  study.gradeLevels = [GradeLevel.Three, GradeLevel.Six, GradeLevel.Nine] →
  0 < study.sampleProportion ∧ study.sampleProportion ≤ 1 →
  isMostReasonable study SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_reasonable_l1088_108819


namespace NUMINAMATH_CALUDE_evaluate_expression_l1088_108898

theorem evaluate_expression : (10^8 / (2.5 * 10^5)) * 3 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1088_108898


namespace NUMINAMATH_CALUDE_range_of_c_l1088_108827

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 1 / a + 1 / b = 1) (h2 : 1 / (a + b) + 1 / c = 1) :
  1 < c ∧ c ≤ 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l1088_108827


namespace NUMINAMATH_CALUDE_isabel_photo_distribution_l1088_108807

/-- Given a total number of pictures and a number of albums, 
    calculate the number of pictures in each album assuming equal distribution. -/
def picturesPerAlbum (totalPictures : ℕ) (numAlbums : ℕ) : ℕ :=
  totalPictures / numAlbums

/-- Theorem stating that given 6 pictures divided into 3 albums, 
    each album contains 2 pictures. -/
theorem isabel_photo_distribution :
  let phonePhotos := 2
  let cameraPhotos := 4
  let totalPhotos := phonePhotos + cameraPhotos
  let numAlbums := 3
  picturesPerAlbum totalPhotos numAlbums = 2 := by
  sorry

end NUMINAMATH_CALUDE_isabel_photo_distribution_l1088_108807


namespace NUMINAMATH_CALUDE_double_reflection_result_l1088_108825

/-- Reflect a point about the line y=x -/
def reflectYEqualsX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Reflect a point about the line y=-x -/
def reflectYEqualsNegX (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- The final position after two reflections -/
def finalPosition (p : ℝ × ℝ) : ℝ × ℝ :=
  reflectYEqualsNegX (reflectYEqualsX p)

theorem double_reflection_result :
  finalPosition (3, -7) = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_result_l1088_108825


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l1088_108861

-- Define the polynomials
def p (x : ℝ) := x^2 + 2
def q (x : ℝ) := 3*x^3 + 5*x^2 + 2
def r (x : ℝ) := x^4 - 3*x^3 + 2*x^2

-- Define the expression
def expression (x : ℝ) := p x * q x - 2 * r x

-- Theorem statement
theorem nonzero_terms_count : 
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
  ∀ x, expression x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e :=
sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l1088_108861


namespace NUMINAMATH_CALUDE_original_number_proof_l1088_108824

theorem original_number_proof (x : ℝ) (h : 1 + 1/x = 7/3) : x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1088_108824


namespace NUMINAMATH_CALUDE_david_work_rate_l1088_108818

/-- The number of days it takes John to complete the work -/
def john_days : ℝ := 9

/-- The number of days it takes David and John together to complete the work -/
def combined_days : ℝ := 3.2142857142857144

/-- The number of days it takes David to complete the work alone -/
def david_days : ℝ := 5

/-- Theorem stating that given John's work rate and the combined work rate of David and John,
    David's individual work rate can be determined -/
theorem david_work_rate (ε : ℝ) (h_ε : ε > 0) :
  ∃ (d : ℝ), abs (d - david_days) < ε ∧
  1 / d + 1 / john_days = 1 / combined_days :=
sorry

end NUMINAMATH_CALUDE_david_work_rate_l1088_108818


namespace NUMINAMATH_CALUDE_positive_numbers_l1088_108842

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l1088_108842
