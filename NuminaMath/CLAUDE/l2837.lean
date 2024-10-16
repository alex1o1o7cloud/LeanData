import Mathlib

namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l2837_283755

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, 3^x + 5^y + 14 = z! ↔ (x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l2837_283755


namespace NUMINAMATH_CALUDE_base_10_255_equals_base_4_3333_l2837_283762

/-- Converts a list of digits in base 4 to a natural number in base 10 -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Theorem: 255 in base 10 is equal to 3333 in base 4 -/
theorem base_10_255_equals_base_4_3333 :
  255 = base4ToNat [3, 3, 3, 3] := by
  sorry

end NUMINAMATH_CALUDE_base_10_255_equals_base_4_3333_l2837_283762


namespace NUMINAMATH_CALUDE_minimum_value_implies_m_l2837_283797

noncomputable def f (x m : ℝ) : ℝ := 2 * x * Real.log (2 * x - 1) - Real.log (2 * x - 1) - m * x + Real.exp (-1)

theorem minimum_value_implies_m (h : ∀ x ∈ Set.Icc 1 (3/2), f x m ≥ -4 + Real.exp (-1)) 
  (h_min : ∃ x ∈ Set.Icc 1 (3/2), f x m = -4 + Real.exp (-1)) : 
  m = 4/3 * Real.log 2 + 8/3 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_m_l2837_283797


namespace NUMINAMATH_CALUDE_rate_of_increase_comparison_l2837_283786

theorem rate_of_increase_comparison (x : ℝ) :
  let f (x : ℝ) := 1000 * x
  let g (x : ℝ) := x^2 / 1000
  (0 < x ∧ x < 500000) → (deriv f x > deriv g x) ∧
  (x > 500000) → (deriv f x < deriv g x) := by
  sorry

end NUMINAMATH_CALUDE_rate_of_increase_comparison_l2837_283786


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_8_l2837_283765

def binary_number : ℕ := 0b1110101101101

theorem remainder_of_binary_div_8 :
  binary_number % 8 = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_8_l2837_283765


namespace NUMINAMATH_CALUDE_product_sum_inequality_l2837_283700

theorem product_sum_inequality (a b c x y z : ℝ) 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a * x + b * y + c * z ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l2837_283700


namespace NUMINAMATH_CALUDE_initial_speed_proof_l2837_283799

theorem initial_speed_proof (total_distance : ℝ) (first_duration : ℝ) (second_speed : ℝ) (second_duration : ℝ) (remaining_distance : ℝ) :
  total_distance = 600 →
  first_duration = 3 →
  second_speed = 80 →
  second_duration = 4 →
  remaining_distance = 130 →
  ∃ initial_speed : ℝ,
    initial_speed * first_duration + second_speed * second_duration = total_distance - remaining_distance ∧
    initial_speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_proof_l2837_283799


namespace NUMINAMATH_CALUDE_shopping_trip_theorem_l2837_283706

/-- Shopping Trip Theorem -/
theorem shopping_trip_theorem (initial_amount : ℕ) (shoe_cost : ℕ) :
  initial_amount = 158 →
  shoe_cost = 45 →
  let bag_cost := shoe_cost - 17
  let lunch_cost := bag_cost / 4
  let total_spent := shoe_cost + bag_cost + lunch_cost
  initial_amount - total_spent = 78 := by
sorry

end NUMINAMATH_CALUDE_shopping_trip_theorem_l2837_283706


namespace NUMINAMATH_CALUDE_dads_first_half_speed_is_28_l2837_283783

/-- The speed of Jake's dad during the first half of the journey to the water park -/
def dads_first_half_speed : ℝ := by sorry

/-- The total journey time for Jake's dad in hours -/
def total_journey_time : ℝ := 0.5

/-- Jake's biking speed in miles per hour -/
def jake_bike_speed : ℝ := 11

/-- Time it takes Jake to bike to the water park in hours -/
def jake_bike_time : ℝ := 2

/-- Jake's dad's speed during the second half of the journey in miles per hour -/
def dads_second_half_speed : ℝ := 60

theorem dads_first_half_speed_is_28 :
  dads_first_half_speed = 28 := by sorry

end NUMINAMATH_CALUDE_dads_first_half_speed_is_28_l2837_283783


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l2837_283760

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧ (∃ x : ℝ, x > 0 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l2837_283760


namespace NUMINAMATH_CALUDE_warren_guests_proof_l2837_283732

/-- The number of tables Warren has -/
def num_tables : ℝ := 252.0

/-- The number of guests each table can hold -/
def guests_per_table : ℝ := 4.0

/-- The total number of guests Warren can accommodate -/
def total_guests : ℝ := num_tables * guests_per_table

theorem warren_guests_proof : total_guests = 1008.0 := by
  sorry

end NUMINAMATH_CALUDE_warren_guests_proof_l2837_283732


namespace NUMINAMATH_CALUDE_rectangle_increase_l2837_283788

/-- Proves that for a rectangle with length increased by 10% and area increased by 37.5%,
    the breadth must be increased by 25% -/
theorem rectangle_increase (L B : ℝ) (h_pos_L : L > 0) (h_pos_B : B > 0) : 
  ∃ p : ℝ, 
    (1.1 * L) * (B * (1 + p / 100)) = 1.375 * (L * B) ∧ 
    p = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_increase_l2837_283788


namespace NUMINAMATH_CALUDE_balloon_count_l2837_283724

/-- Represents the balloon shooting game with two levels. -/
structure BalloonGame where
  /-- The number of balloons missed in the first level -/
  missed_first_level : ℕ
  /-- The total number of balloons in each level -/
  total_balloons : ℕ

/-- The conditions of the balloon shooting game -/
def game_conditions (game : BalloonGame) : Prop :=
  let hit_first_level := 4 * game.missed_first_level + 2
  let hit_second_level := hit_first_level + 8
  hit_second_level = 6 * game.missed_first_level ∧
  game.total_balloons = hit_first_level + game.missed_first_level

/-- The theorem stating that the number of balloons in each level is 147 -/
theorem balloon_count (game : BalloonGame) 
  (h : game_conditions game) : game.total_balloons = 147 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l2837_283724


namespace NUMINAMATH_CALUDE_relationship_between_D_and_A_l2837_283758

theorem relationship_between_D_and_A (A B C D : Prop) 
  (h1 : A → B)
  (h2 : ¬(B → A))
  (h3 : B → C)
  (h4 : ¬(C → B))
  (h5 : D ↔ C) :
  (D → A) ∧ ¬(A → D) :=
sorry

end NUMINAMATH_CALUDE_relationship_between_D_and_A_l2837_283758


namespace NUMINAMATH_CALUDE_laura_drives_234_miles_per_week_l2837_283791

/-- Calculates the total miles driven per week based on Laura's travel habits -/
def total_miles_per_week (school_round_trip : ℕ) (supermarket_extra : ℕ) (gym_distance : ℕ) (friend_distance : ℕ) : ℕ :=
  let school_miles := school_round_trip * 5
  let supermarket_miles := (school_round_trip + 2 * supermarket_extra) * 2
  let gym_miles := 2 * gym_distance * 3
  let friend_miles := 2 * friend_distance
  school_miles + supermarket_miles + gym_miles + friend_miles

/-- Theorem stating that Laura drives 234 miles per week -/
theorem laura_drives_234_miles_per_week :
  total_miles_per_week 20 10 5 12 = 234 := by
  sorry

end NUMINAMATH_CALUDE_laura_drives_234_miles_per_week_l2837_283791


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2837_283771

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  ∃ (M : ℝ), M = 6 * Real.sqrt 5 ∧ 
  Real.sqrt (3 * x + 4) + Real.sqrt (3 * y + 4) + Real.sqrt (3 * z + 4) ≤ M ∧
  ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 7 ∧
    Real.sqrt (3 * x' + 4) + Real.sqrt (3 * y' + 4) + Real.sqrt (3 * z' + 4) = M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2837_283771


namespace NUMINAMATH_CALUDE_roots_sum_absolute_value_l2837_283753

theorem roots_sum_absolute_value (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + x + m = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    |x₁| + |x₂| = 3) →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_absolute_value_l2837_283753


namespace NUMINAMATH_CALUDE_no_five_circle_arrangement_l2837_283769

-- Define a structure for a point in a plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is the circumcenter of a triangle
def isCircumcenter (p : Point2D) (a b c : Point2D) : Prop :=
  (p.x - a.x)^2 + (p.y - a.y)^2 = (p.x - b.x)^2 + (p.y - b.y)^2 ∧
  (p.x - b.x)^2 + (p.y - b.y)^2 = (p.x - c.x)^2 + (p.y - c.y)^2

-- Theorem statement
theorem no_five_circle_arrangement :
  ¬ ∃ (p₁ p₂ p₃ p₄ p₅ : Point2D),
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    (isCircumcenter p₁ p₂ p₃ p₄ ∨ isCircumcenter p₁ p₂ p₃ p₅ ∨ isCircumcenter p₁ p₂ p₄ p₅ ∨ isCircumcenter p₁ p₃ p₄ p₅) ∧
    (isCircumcenter p₂ p₁ p₃ p₄ ∨ isCircumcenter p₂ p₁ p₃ p₅ ∨ isCircumcenter p₂ p₁ p₄ p₅ ∨ isCircumcenter p₂ p₃ p₄ p₅) ∧
    (isCircumcenter p₃ p₁ p₂ p₄ ∨ isCircumcenter p₃ p₁ p₂ p₅ ∨ isCircumcenter p₃ p₁ p₄ p₅ ∨ isCircumcenter p₃ p₂ p₄ p₅) ∧
    (isCircumcenter p₄ p₁ p₂ p₃ ∨ isCircumcenter p₄ p₁ p₂ p₅ ∨ isCircumcenter p₄ p₁ p₃ p₅ ∨ isCircumcenter p₄ p₂ p₃ p₅) ∧
    (isCircumcenter p₅ p₁ p₂ p₃ ∨ isCircumcenter p₅ p₁ p₂ p₄ ∨ isCircumcenter p₅ p₁ p₃ p₄ ∨ isCircumcenter p₅ p₂ p₃ p₄) :=
by
  sorry


end NUMINAMATH_CALUDE_no_five_circle_arrangement_l2837_283769


namespace NUMINAMATH_CALUDE_events_independent_l2837_283767

-- Define the sample space
def Ω : Type := (Bool × Bool)

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define events A, B, and C
def A : Set Ω := {ω | ω.1 = true}
def B : Set Ω := {ω | ω.2 = true}
def C : Set Ω := {ω | ω.1 = ω.2}

-- State the theorem
theorem events_independent :
  (P (A ∩ B) = P A * P B) ∧
  (P (B ∩ C) = P B * P C) ∧
  (P (A ∩ C) = P A * P C) := by sorry

end NUMINAMATH_CALUDE_events_independent_l2837_283767


namespace NUMINAMATH_CALUDE_divisible_by_three_divisible_by_nine_l2837_283790

/-- Represents the decimal digits of a non-negative integer -/
def DecimalDigits : Type := List Nat

/-- Returns the sum of digits in a DecimalDigits representation -/
def sum_of_digits (digits : DecimalDigits) : Nat :=
  digits.sum

/-- Converts a non-negative integer to its DecimalDigits representation -/
def to_decimal_digits (n : Nat) : DecimalDigits :=
  sorry

/-- Converts a DecimalDigits representation back to the original number -/
def from_decimal_digits (digits : DecimalDigits) : Nat :=
  sorry

/-- Theorem: A number is divisible by 3 iff the sum of its digits is divisible by 3 -/
theorem divisible_by_three (n : Nat) :
  n % 3 = 0 ↔ (sum_of_digits (to_decimal_digits n)) % 3 = 0 :=
  sorry

/-- Theorem: A number is divisible by 9 iff the sum of its digits is divisible by 9 -/
theorem divisible_by_nine (n : Nat) :
  n % 9 = 0 ↔ (sum_of_digits (to_decimal_digits n)) % 9 = 0 :=
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_divisible_by_nine_l2837_283790


namespace NUMINAMATH_CALUDE_area_ratio_is_one_l2837_283778

/-- Theorem: The ratio of the areas of rectangles M and N is 1 -/
theorem area_ratio_is_one (a b x y : ℝ) : a > 0 → b > 0 → x > 0 → y > 0 → 
  b * x + a * y = a * b → (x * y) / ((a - x) * (b - y)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_one_l2837_283778


namespace NUMINAMATH_CALUDE_product_of_mixed_numbers_l2837_283756

theorem product_of_mixed_numbers :
  let a : Rat := 2 + 1/6
  let b : Rat := 3 + 2/9
  a * b = 377/54 := by
  sorry

end NUMINAMATH_CALUDE_product_of_mixed_numbers_l2837_283756


namespace NUMINAMATH_CALUDE_square_all_digits_odd_iff_one_or_three_l2837_283722

/-- A function that returns true if all digits in the decimal representation of a natural number are odd -/
def allDigitsOdd (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d % 2 = 1

/-- Theorem stating that for a positive integer n, all digits in n^2 are odd if and only if n = 1 or n = 3 -/
theorem square_all_digits_odd_iff_one_or_three (n : ℕ) (hn : n > 0) :
  allDigitsOdd (n^2) ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_all_digits_odd_iff_one_or_three_l2837_283722


namespace NUMINAMATH_CALUDE_number_division_proof_l2837_283716

theorem number_division_proof (x : ℝ) : 4 * x = 166.08 → x / 4 = 10.38 := by
  sorry

end NUMINAMATH_CALUDE_number_division_proof_l2837_283716


namespace NUMINAMATH_CALUDE_circle_extrema_l2837_283754

theorem circle_extrema (x y : ℝ) (h : (x - 3)^2 + (y - 3)^2 = 6) :
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ y₀ / x₀ = 3 + 2 * Real.sqrt 2 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → y₁ / x₁ ≤ 3 + 2 * Real.sqrt 2) ∧
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ y₀ / x₀ = 3 - 2 * Real.sqrt 2 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → y₁ / x₁ ≥ 3 - 2 * Real.sqrt 2) ∧
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ Real.sqrt ((x₀ - 2)^2 + y₀^2) = Real.sqrt 10 + Real.sqrt 6 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → Real.sqrt ((x₁ - 2)^2 + y₁^2) ≤ Real.sqrt 10 + Real.sqrt 6) ∧
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 3)^2 = 6 ∧ Real.sqrt ((x₀ - 2)^2 + y₀^2) = Real.sqrt 10 - Real.sqrt 6 ∧
    ∀ (x₁ y₁ : ℝ), (x₁ - 3)^2 + (y₁ - 3)^2 = 6 → Real.sqrt ((x₁ - 2)^2 + y₁^2) ≥ Real.sqrt 10 - Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_circle_extrema_l2837_283754


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l2837_283773

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x^2 + 3/x ≥ (3/2) * Real.rpow 18 (1/3) :=
by sorry

theorem min_value_achievable :
  ∃ x > 0, x^2 + 3/x = (3/2) * Real.rpow 18 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l2837_283773


namespace NUMINAMATH_CALUDE_regions_99_lines_l2837_283796

/-- The number of regions created by lines in a plane -/
def num_regions (num_lines : ℕ) (all_parallel : Bool) (all_intersect_one_point : Bool) : ℕ :=
  if all_parallel then
    num_lines + 1
  else if all_intersect_one_point then
    2 * num_lines
  else
    0  -- This case is not used in our theorem, but included for completeness

/-- Theorem stating the possible number of regions created by 99 lines in a plane -/
theorem regions_99_lines :
  ∀ (n : ℕ), n < 199 →
  (∃ (all_parallel all_intersect_one_point : Bool),
    num_regions 99 all_parallel all_intersect_one_point = n) →
  (n = 100 ∨ n = 198) :=
by
  sorry

#check regions_99_lines

end NUMINAMATH_CALUDE_regions_99_lines_l2837_283796


namespace NUMINAMATH_CALUDE_netPopulationIncreaseIs345600_l2837_283795

/-- Calculates the net population increase in one day given birth and death rates -/
def netPopulationIncreaseInOneDay (birthRate : ℕ) (deathRate : ℕ) : ℕ :=
  let netIncreasePerTwoSeconds := birthRate - deathRate
  let netIncreasePerSecond := netIncreasePerTwoSeconds / 2
  let secondsInDay : ℕ := 24 * 60 * 60
  netIncreasePerSecond * secondsInDay

/-- Theorem stating that the net population increase in one day is 345,600 given the specified birth and death rates -/
theorem netPopulationIncreaseIs345600 :
  netPopulationIncreaseInOneDay 10 2 = 345600 := by
  sorry

#eval netPopulationIncreaseInOneDay 10 2

end NUMINAMATH_CALUDE_netPopulationIncreaseIs345600_l2837_283795


namespace NUMINAMATH_CALUDE_hohyeon_taller_than_seulgi_l2837_283745

/-- Seulgi's height in centimeters -/
def seulgi_height : ℕ := 159

/-- Hohyeon's height in centimeters -/
def hohyeon_height : ℕ := 162

/-- Theorem stating that Hohyeon is taller than Seulgi -/
theorem hohyeon_taller_than_seulgi : hohyeon_height > seulgi_height := by
  sorry

end NUMINAMATH_CALUDE_hohyeon_taller_than_seulgi_l2837_283745


namespace NUMINAMATH_CALUDE_rachel_homework_pages_l2837_283735

/-- The total number of pages for math and biology homework -/
def total_math_biology_pages (math_pages biology_pages : ℕ) : ℕ :=
  math_pages + biology_pages

/-- Theorem: Given Rachel has 8 pages of math homework and 3 pages of biology homework,
    the total number of pages for math and biology homework is 11. -/
theorem rachel_homework_pages : total_math_biology_pages 8 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_pages_l2837_283735


namespace NUMINAMATH_CALUDE_tangent_line_proof_l2837_283798

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * x

/-- The line equation: 2x - y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_proof :
  (∃ (x₀ : ℝ), f x₀ = x₀ ∧ line_equation x₀ (f x₀)) ∧  -- The line passes through a point on f(x)
  line_equation 1 1 ∧  -- The line passes through (1,1)
  (∀ (x : ℝ), f' x = (2 : ℝ)) →  -- The derivative of f(x) is 2
  ∃ (x₀ : ℝ), f x₀ = x₀ ∧ line_equation x₀ (f x₀) ∧ f' x₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l2837_283798


namespace NUMINAMATH_CALUDE_max_a_when_f_has_minimum_l2837_283740

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

/-- Proposition: If f(x) has a minimum value, then the maximum value of a is 1 -/
theorem max_a_when_f_has_minimum (a : ℝ) :
  (∃ m : ℝ, ∀ x : ℝ, f a x ≥ m) → a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_a_when_f_has_minimum_l2837_283740


namespace NUMINAMATH_CALUDE_air_quality_probabilities_l2837_283730

def prob_grade_A : ℝ := 0.8
def prob_grade_B : ℝ := 0.1
def prob_grade_C : ℝ := 0.1

def prob_satisfactory (p_A p_B p_C : ℝ) : ℝ :=
  p_A * p_A + 2 * p_A * (1 - p_A)

def prob_two_out_of_three (p : ℝ) : ℝ :=
  3 * p * p * (1 - p)

theorem air_quality_probabilities :
  prob_satisfactory prob_grade_A prob_grade_B prob_grade_C = 0.96 ∧
  prob_two_out_of_three (prob_satisfactory prob_grade_A prob_grade_B prob_grade_C) = 0.110592 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probabilities_l2837_283730


namespace NUMINAMATH_CALUDE_eliana_refills_l2837_283712

theorem eliana_refills (total_spent : ℕ) (cost_per_refill : ℕ) (h1 : total_spent = 63) (h2 : cost_per_refill = 21) :
  total_spent / cost_per_refill = 3 := by
  sorry

end NUMINAMATH_CALUDE_eliana_refills_l2837_283712


namespace NUMINAMATH_CALUDE_max_lateral_area_inscribed_cylinder_l2837_283733

/-- The maximum lateral surface area of a cylinder inscribed in a sphere -/
theorem max_lateral_area_inscribed_cylinder (r : ℝ) (h : r > 0) :
  ∃ (cylinder_area : ℝ),
    (∀ (inscribed_cylinder_area : ℝ), inscribed_cylinder_area ≤ cylinder_area) ∧
    cylinder_area = 2 * Real.pi * r^2 :=
sorry

end NUMINAMATH_CALUDE_max_lateral_area_inscribed_cylinder_l2837_283733


namespace NUMINAMATH_CALUDE_quarters_percentage_l2837_283725

theorem quarters_percentage (num_dimes : ℕ) (num_quarters : ℕ) : num_dimes = 70 → num_quarters = 30 → 
  (num_quarters * 25 : ℚ) / ((num_dimes * 10 + num_quarters * 25) : ℚ) * 100 = 51724 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_quarters_percentage_l2837_283725


namespace NUMINAMATH_CALUDE_mikes_remaining_books_l2837_283746

theorem mikes_remaining_books (initial_books sold_books : ℕ) :
  initial_books = 51 →
  sold_books = 45 →
  initial_books - sold_books = 6 :=
by sorry

end NUMINAMATH_CALUDE_mikes_remaining_books_l2837_283746


namespace NUMINAMATH_CALUDE_inequality_abc_l2837_283751

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_abc_l2837_283751


namespace NUMINAMATH_CALUDE_price_adjustment_l2837_283703

theorem price_adjustment (a : ℝ) : 
  let price_after_reductions := a * (1 - 0.1) * (1 - 0.1)
  let final_price := price_after_reductions * (1 + 0.2)
  final_price = 0.972 * a :=
by sorry

end NUMINAMATH_CALUDE_price_adjustment_l2837_283703


namespace NUMINAMATH_CALUDE_good_number_characterization_twenty_nine_is_good_good_numbers_up_to_nine_correct_product_of_good_numbers_is_good_l2837_283715

def is_good_number (n : ℤ) : Prop :=
  ∃ x y : ℤ, n = x^2 + 2*x*y + 2*y^2

theorem good_number_characterization (n : ℤ) :
  is_good_number n ↔ ∃ a b : ℤ, n = a^2 + b^2 :=
sorry

theorem twenty_nine_is_good : is_good_number 29 :=
sorry

def good_numbers_up_to_nine : List ℤ := [1, 2, 4, 5, 8, 9]

theorem good_numbers_up_to_nine_correct :
  ∀ n : ℤ, n ∈ good_numbers_up_to_nine ↔ (1 ≤ n ∧ n ≤ 9 ∧ is_good_number n) :=
sorry

theorem product_of_good_numbers_is_good (m n : ℤ) :
  is_good_number m → is_good_number n → is_good_number (m * n) :=
sorry

end NUMINAMATH_CALUDE_good_number_characterization_twenty_nine_is_good_good_numbers_up_to_nine_correct_product_of_good_numbers_is_good_l2837_283715


namespace NUMINAMATH_CALUDE_max_value_of_f_l2837_283742

def f (x a : ℝ) : ℝ := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2837_283742


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2837_283709

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 3| = |x + 5| := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2837_283709


namespace NUMINAMATH_CALUDE_equation_solution_l2837_283792

theorem equation_solution : 
  ∃ x : ℝ, ((x * 5) / 2.5) - (8 * 2.25) = 5.5 ∧ x = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2837_283792


namespace NUMINAMATH_CALUDE_typists_productivity_l2837_283738

/-- Given that 20 typists can type 42 letters in 20 minutes, 
    prove that 30 typists can type 189 letters in 1 hour at the same rate. -/
theorem typists_productivity (typists_20 : ℕ) (letters_20 : ℕ) (minutes_20 : ℕ) 
  (typists_30 : ℕ) (minutes_60 : ℕ) :
  typists_20 = 20 →
  letters_20 = 42 →
  minutes_20 = 20 →
  typists_30 = 30 →
  minutes_60 = 60 →
  (typists_30 : ℚ) * (letters_20 : ℚ) / (typists_20 : ℚ) * (minutes_60 : ℚ) / (minutes_20 : ℚ) = 189 :=
by sorry

end NUMINAMATH_CALUDE_typists_productivity_l2837_283738


namespace NUMINAMATH_CALUDE_sum_difference_remainder_mod_two_l2837_283747

theorem sum_difference_remainder_mod_two : 
  let n := 100
  let sum_remainder_one := (Finset.range n).sum (fun i => if i % 2 = 1 then i + 1 else 0)
  let sum_remainder_zero := (Finset.range n).sum (fun i => if i % 2 = 0 then i + 1 else 0)
  sum_remainder_zero - sum_remainder_one = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_difference_remainder_mod_two_l2837_283747


namespace NUMINAMATH_CALUDE_geometry_perpendicular_parallel_l2837_283782

open Set

structure Geometry3D where
  Point : Type
  Line : Type
  Plane : Type
  on_plane : Point → Plane → Prop
  perp_line_plane : Line → Plane → Prop
  perp_plane_plane : Plane → Plane → Prop
  parallel_line_plane : Line → Plane → Prop
  parallel_plane_plane : Plane → Plane → Prop
  line_through : Point → Line → Prop
  plane_through : Point → Plane → Prop

variable (G : Geometry3D)

theorem geometry_perpendicular_parallel 
  (P : G.Point) (π : G.Plane) (h : ¬ G.on_plane P π) :
  (∃! l : G.Line, G.line_through P l ∧ G.perp_line_plane l π) ∧
  (∃ S : Set G.Plane, Infinite S ∧ ∀ σ ∈ S, G.plane_through P σ ∧ G.perp_plane_plane σ π) ∧
  (∃ S : Set G.Line, Infinite S ∧ ∀ l ∈ S, G.line_through P l ∧ G.parallel_line_plane l π) ∧
  (∃! σ : G.Plane, G.plane_through P σ ∧ G.parallel_plane_plane σ π) :=
sorry

end NUMINAMATH_CALUDE_geometry_perpendicular_parallel_l2837_283782


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2837_283718

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.I * z = z + a * Complex.I)
  (h2 : Complex.abs z = Real.sqrt 2)
  (h3 : a > 0) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2837_283718


namespace NUMINAMATH_CALUDE_integral_reciprocal_plus_x_l2837_283710

theorem integral_reciprocal_plus_x : ∫ x in (2 : ℝ)..4, (1 / x + x) = Real.log 2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_plus_x_l2837_283710


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2837_283719

theorem smallest_n_congruence : ∃! n : ℕ+, 
  (∀ m : ℕ+, 5 * m ≡ 1846 [ZMOD 26] → n ≤ m) ∧ 
  (5 * n ≡ 1846 [ZMOD 26]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2837_283719


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l2837_283780

theorem price_decrease_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 900)
  (h2 : new_price = 684) :
  (original_price - new_price) / original_price * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l2837_283780


namespace NUMINAMATH_CALUDE_college_ratio_theorem_l2837_283726

/-- Represents the ratio of boys to girls in a college -/
structure CollegeRatio where
  boys : ℕ
  girls : ℕ

/-- Given the total number of students and the number of girls, calculate the ratio of boys to girls -/
def calculateRatio (totalStudents : ℕ) (numGirls : ℕ) : CollegeRatio :=
  { boys := totalStudents - numGirls,
    girls := numGirls }

/-- Theorem stating that for a college with 240 total students and 140 girls, the ratio of boys to girls is 5:7 -/
theorem college_ratio_theorem :
  let ratio := calculateRatio 240 140
  ratio.boys = 5 ∧ ratio.girls = 7 := by
  sorry


end NUMINAMATH_CALUDE_college_ratio_theorem_l2837_283726


namespace NUMINAMATH_CALUDE_joes_test_scores_l2837_283750

theorem joes_test_scores (initial_avg : ℝ) (lowest_score : ℝ) (new_avg : ℝ) :
  initial_avg = 70 →
  lowest_score = 55 →
  new_avg = 75 →
  ∃ n : ℕ, n > 1 ∧
    (n : ℝ) * initial_avg - lowest_score = (n - 1 : ℝ) * new_avg ∧
    n = 4 :=
by sorry

end NUMINAMATH_CALUDE_joes_test_scores_l2837_283750


namespace NUMINAMATH_CALUDE_percentage_calculation_l2837_283713

theorem percentage_calculation : 
  let initial_value : ℝ := 180
  let percentage : ℝ := 1/3
  let divisor : ℝ := 6
  (initial_value * (percentage / 100)) / divisor = 0.1 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2837_283713


namespace NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l2837_283702

theorem smallest_solution_absolute_value_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y ∧
  x = -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l2837_283702


namespace NUMINAMATH_CALUDE_exists_self_power_congruence_l2837_283757

theorem exists_self_power_congruence : ∃ N : ℕ, 
  (10^2000 ≤ N) ∧ (N < 10^2001) ∧ (N ≡ N^2001 [ZMOD 10^2001]) := by
  sorry

end NUMINAMATH_CALUDE_exists_self_power_congruence_l2837_283757


namespace NUMINAMATH_CALUDE_aaron_age_l2837_283761

/-- Proves that Aaron is 16 years old given the conditions of the problem -/
theorem aaron_age : 
  ∀ (aaron_age henry_age sister_age : ℕ),
  sister_age = 3 * aaron_age →
  henry_age = 4 * sister_age →
  henry_age + sister_age = 240 →
  aaron_age = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_aaron_age_l2837_283761


namespace NUMINAMATH_CALUDE_withdrawal_theorem_l2837_283776

/-- Calculates the number of bills received when withdrawing from two banks -/
def number_of_bills (withdrawal_per_bank : ℕ) (bill_denomination : ℕ) : ℕ :=
  (2 * withdrawal_per_bank) / bill_denomination

/-- Theorem: Withdrawing $300 from each of two banks in $20 bills results in 30 bills -/
theorem withdrawal_theorem : number_of_bills 300 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_withdrawal_theorem_l2837_283776


namespace NUMINAMATH_CALUDE_second_number_proof_l2837_283741

theorem second_number_proof (x : ℕ) : 
  (∃ k m : ℕ, 1657 = 127 * k + 6 ∧ x = 127 * m + 5 ∧ 
   ∀ d : ℕ, d > 127 → (1657 % d ≠ 6 ∨ x % d ≠ 5)) → 
  x = 1529 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l2837_283741


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2837_283720

theorem complex_equation_solution (z : ℂ) : z * (1 + 2*I) = 3 + I → z = 1 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2837_283720


namespace NUMINAMATH_CALUDE_log_product_equals_five_l2837_283739

theorem log_product_equals_five :
  (Real.log 4 / Real.log 2) * (Real.log 8 / Real.log 4) *
  (Real.log 16 / Real.log 8) * (Real.log 32 / Real.log 16) = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_five_l2837_283739


namespace NUMINAMATH_CALUDE_sum_one_to_fortyfive_base6_l2837_283787

/-- Represents a number in base 6 --/
def Base6 := ℕ

/-- Converts a natural number to its base 6 representation --/
def to_base6 (n : ℕ) : Base6 := sorry

/-- Converts a base 6 number to its natural number representation --/
def from_base6 (b : Base6) : ℕ := sorry

/-- Adds two base 6 numbers --/
def add_base6 (a b : Base6) : Base6 := sorry

/-- Multiplies two base 6 numbers --/
def mul_base6 (a b : Base6) : Base6 := sorry

/-- Divides a base 6 number by 2 --/
def div2_base6 (a : Base6) : Base6 := sorry

/-- Calculates the sum of an arithmetic sequence in base 6 --/
def sum_arithmetic_base6 (first last : Base6) : Base6 :=
  let n := add_base6 (from_base6 last) (to_base6 1)
  div2_base6 (mul_base6 n (add_base6 first last))

/-- The main theorem to be proved --/
theorem sum_one_to_fortyfive_base6 :
  sum_arithmetic_base6 (to_base6 1) (to_base6 45) = to_base6 2003 := by sorry

end NUMINAMATH_CALUDE_sum_one_to_fortyfive_base6_l2837_283787


namespace NUMINAMATH_CALUDE_smallest_b_value_l2837_283736

theorem smallest_b_value : ∃ (b : ℝ), b > 0 ∧
  (∀ (x : ℝ), x > 0 →
    (9 * Real.sqrt ((3 * x)^2 + 2^2) - 6 * x^2 - 4) / (Real.sqrt (4 + 6 * x^2) + 5) = 3 →
    b ≤ x) ∧
  (9 * Real.sqrt ((3 * b)^2 + 2^2) - 6 * b^2 - 4) / (Real.sqrt (4 + 6 * b^2) + 5) = 3 ∧
  b = Real.sqrt (11 / 30) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2837_283736


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l2837_283701

/-- The lateral area of a cylinder with base diameter and height both equal to 4 cm is 16π cm². -/
theorem cylinder_lateral_area (π : ℝ) (h : π > 0) : 
  let d : ℝ := 4 -- diameter
  let h : ℝ := 4 -- height
  let r : ℝ := d / 2 -- radius
  let lateral_area : ℝ := 2 * π * r * h
  lateral_area = 16 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l2837_283701


namespace NUMINAMATH_CALUDE_tom_fruit_purchase_total_l2837_283784

/-- Calculate the total amount Tom paid for fruits with discount and tax --/
theorem tom_fruit_purchase_total : 
  let apple_cost : ℝ := 8 * 70
  let mango_cost : ℝ := 9 * 90
  let grape_cost : ℝ := 5 * 150
  let total_before_discount : ℝ := apple_cost + mango_cost + grape_cost
  let discount_rate : ℝ := 0.10
  let tax_rate : ℝ := 0.05
  let discounted_amount : ℝ := total_before_discount * (1 - discount_rate)
  let final_amount : ℝ := discounted_amount * (1 + tax_rate)
  final_amount = 2003.4 := by sorry

end NUMINAMATH_CALUDE_tom_fruit_purchase_total_l2837_283784


namespace NUMINAMATH_CALUDE_library_interval_proof_l2837_283728

def dance_interval : ℕ := 6
def karate_interval : ℕ := 12
def next_common_day : ℕ := 36

theorem library_interval_proof (x : ℕ) 
  (h1 : x > 0)
  (h2 : x ∣ next_common_day)
  (h3 : x ≠ dance_interval)
  (h4 : x ≠ karate_interval)
  (h5 : ∀ y : ℕ, y > 0 → y ∣ next_common_day → y ≠ dance_interval → y ≠ karate_interval → y ≤ x) :
  x = 18 := by sorry

end NUMINAMATH_CALUDE_library_interval_proof_l2837_283728


namespace NUMINAMATH_CALUDE_problem_solution_l2837_283749

theorem problem_solution (a b c d e : ℝ) 
  (h1 : a^2 + b^2 + c^2 + e = d + Real.sqrt (a + b + c - d + 3*e))
  (h2 : e = 2) : 
  d = 21/4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2837_283749


namespace NUMINAMATH_CALUDE_remaining_sausage_meat_l2837_283793

/-- Calculates the remaining sausage meat in ounces after some links are eaten -/
theorem remaining_sausage_meat 
  (total_pounds : ℕ) 
  (total_links : ℕ) 
  (eaten_links : ℕ) 
  (h1 : total_pounds = 10) 
  (h2 : total_links = 40) 
  (h3 : eaten_links = 12) : 
  (total_pounds * 16 - (total_pounds * 16 / total_links) * eaten_links : ℕ) = 112 := by
  sorry

#check remaining_sausage_meat

end NUMINAMATH_CALUDE_remaining_sausage_meat_l2837_283793


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2837_283708

/-- Given a function f(x) = 2a^(x-b) + 1 where a > 0 and a ≠ 1, 
    if f(2) = 3, then b = 2 -/
theorem fixed_point_exponential_function (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  (∀ x, 2 * a^(x - b) + 1 = 3) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2837_283708


namespace NUMINAMATH_CALUDE_water_level_rise_l2837_283717

-- Define the cube's edge length
def cube_edge : ℝ := 16

-- Define the vessel's base dimensions
def vessel_length : ℝ := 20
def vessel_width : ℝ := 15

-- Define the volume of the cube
def cube_volume : ℝ := cube_edge ^ 3

-- Define the area of the vessel's base
def vessel_base_area : ℝ := vessel_length * vessel_width

-- Theorem statement
theorem water_level_rise :
  (cube_volume / vessel_base_area) = (cube_edge ^ 3) / (vessel_length * vessel_width) :=
by sorry

end NUMINAMATH_CALUDE_water_level_rise_l2837_283717


namespace NUMINAMATH_CALUDE_range_of_a_l2837_283748

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, Monotone (fun x => (3 - 2*a)^x)

-- Define the theorem
theorem range_of_a (a : ℝ) : p a ∧ ¬(q a) → a ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2837_283748


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2837_283775

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ Real.sqrt ((7 * x) / 3) = x :=
  ⟨7/3, by sorry⟩

end NUMINAMATH_CALUDE_unique_positive_solution_l2837_283775


namespace NUMINAMATH_CALUDE_product_of_integers_l2837_283770

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 20)
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x * y = 99 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l2837_283770


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2837_283729

theorem least_five_digit_square_cube : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  (∃ a : ℕ, n = a^2) ∧ 
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, m < n → ¬(m ≥ 10000 ∧ m < 100000 ∧ (∃ x : ℕ, m = x^2) ∧ (∃ y : ℕ, m = y^3))) ∧
  n = 15625 := by
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2837_283729


namespace NUMINAMATH_CALUDE_minimum_value_curve_exponent_l2837_283794

theorem minimum_value_curve_exponent (m n : ℝ) (a : ℝ) : 
  m > 0 → n > 0 → m + n = 1 → 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (1/x) + (16/y) ≥ (1/m) + (16/n)) →
  ((m/5)^a = n/4) →
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_minimum_value_curve_exponent_l2837_283794


namespace NUMINAMATH_CALUDE_min_divisions_is_48_l2837_283764

/-- Represents a cell division strategy -/
structure DivisionStrategy where
  div42 : ℕ  -- number of divisions resulting in 42 cells
  div44 : ℕ  -- number of divisions resulting in 44 cells

/-- The number of cells after applying a division strategy -/
def resultingCells (s : DivisionStrategy) : ℕ :=
  1 + 41 * s.div42 + 43 * s.div44

/-- A division strategy is valid if it results in exactly 1993 cells -/
def isValidStrategy (s : DivisionStrategy) : Prop :=
  resultingCells s = 1993

/-- The total number of divisions in a strategy -/
def totalDivisions (s : DivisionStrategy) : ℕ :=
  s.div42 + s.div44

/-- There exists a valid division strategy -/
axiom exists_valid_strategy : ∃ s : DivisionStrategy, isValidStrategy s

/-- The minimum number of divisions needed is 48 -/
theorem min_divisions_is_48 :
  ∃ s : DivisionStrategy, isValidStrategy s ∧
    totalDivisions s = 48 ∧
    ∀ t : DivisionStrategy, isValidStrategy t → totalDivisions s ≤ totalDivisions t :=
  sorry

end NUMINAMATH_CALUDE_min_divisions_is_48_l2837_283764


namespace NUMINAMATH_CALUDE_work_schedule_lcm_l2837_283766

theorem work_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_work_schedule_lcm_l2837_283766


namespace NUMINAMATH_CALUDE_sixth_sample_is_98_l2837_283737

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) (k : ℕ) : ℕ :=
  start + (k - 1) * (total / sampleSize)

theorem sixth_sample_is_98 :
  systematicSample 900 50 8 6 = 98 := by
  sorry

end NUMINAMATH_CALUDE_sixth_sample_is_98_l2837_283737


namespace NUMINAMATH_CALUDE_average_weight_increase_l2837_283743

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 35 + 75
  let new_average := new_total / 8
  new_average - initial_average = 5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2837_283743


namespace NUMINAMATH_CALUDE_expression_evaluation_l2837_283781

theorem expression_evaluation :
  let x : ℚ := 1/2
  (x - 3)^2 + (x + 3)*(x - 3) + 2*x*(2 - x) = -1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2837_283781


namespace NUMINAMATH_CALUDE_problem_statement_l2837_283768

theorem problem_statement (x y : ℝ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2837_283768


namespace NUMINAMATH_CALUDE_smallest_circle_radius_l2837_283772

/-- A regular hexagon with side length 2 units -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- A circle in the context of our problem -/
structure Circle :=
  (center : Fin 6)  -- Vertex of the hexagon (0 to 5)
  (radius : ℝ)

/-- Three circles touching each other externally -/
def touching_circles (h : RegularHexagon) (c₁ c₂ c₃ : Circle) : Prop :=
  (c₁.center = 0 ∧ c₂.center = 1 ∧ c₃.center = 2) ∧  -- Centers at A, B, C
  (c₁.radius + c₂.radius = h.side_length) ∧
  (c₁.radius + c₃.radius = h.side_length * Real.sqrt 3) ∧
  (c₂.radius + c₃.radius = h.side_length)

theorem smallest_circle_radius 
  (h : RegularHexagon) 
  (c₁ c₂ c₃ : Circle) 
  (touch : touching_circles h c₁ c₂ c₃) :
  min c₁.radius (min c₂.radius c₃.radius) = 2 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_radius_l2837_283772


namespace NUMINAMATH_CALUDE_square_side_length_for_unit_area_l2837_283727

theorem square_side_length_for_unit_area (s : ℝ) :
  s > 0 → s * s = 1 → s = 1 := by sorry

end NUMINAMATH_CALUDE_square_side_length_for_unit_area_l2837_283727


namespace NUMINAMATH_CALUDE_triangle_side_equations_l2837_283752

theorem triangle_side_equations (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ¬(∃ x y z : ℝ, x^2 - 2*b*x + 2*a*c = 0 ∧ y^2 - 2*c*y + 2*a*b = 0 ∧ z^2 - 2*a*z + 2*b*c = 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_equations_l2837_283752


namespace NUMINAMATH_CALUDE_regular_ngon_construction_l2837_283734

/-- Theorem about the construction of points on the extensions of a regular n-gon's sides -/
theorem regular_ngon_construction (n : ℕ) (a : ℝ) (h_n : n ≥ 5) :
  let α : ℝ := π - (2 * π) / n
  ∀ (x : ℕ → ℝ), 
    (∀ k, x k = (a + x ((k + 1) % n)) * Real.cos α) →
    ∀ k, x k = (a * Real.cos α) / (1 - Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_regular_ngon_construction_l2837_283734


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_gt_one_min_cone_cylinder_volume_ratio_l2837_283744

/-- Represents a cone with an inscribed sphere and circumscribed cylinder -/
structure ConeWithSphereAndCylinder where
  R : ℝ  -- Base radius of the cone
  h : ℝ  -- Height of the cone
  r : ℝ  -- Radius of the inscribed sphere
  cone_volume : ℝ  -- Volume of the cone
  cylinder_volume : ℝ  -- Volume of the cylinder

/-- The ratio of cone volume to cylinder volume is always greater than 1 -/
theorem cone_cylinder_volume_ratio_gt_one (c : ConeWithSphereAndCylinder) :
  c.cone_volume / c.cylinder_volume > 1 :=
sorry

/-- The minimum ratio of cone volume to cylinder volume is 4/3 -/
theorem min_cone_cylinder_volume_ratio (c : ConeWithSphereAndCylinder) :
  ∃ (k : ℝ), k = 4/3 ∧ c.cone_volume / c.cylinder_volume ≥ k :=
sorry

end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_gt_one_min_cone_cylinder_volume_ratio_l2837_283744


namespace NUMINAMATH_CALUDE_vector_addition_l2837_283704

/-- Given two vectors a and b in ℝ², prove that 2b + 3a equals (6,1) -/
theorem vector_addition (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (0, -1)) :
  2 • b + 3 • a = (6, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l2837_283704


namespace NUMINAMATH_CALUDE_old_refrigerator_cost_proof_l2837_283731

/-- The daily cost of Kurt's new refrigerator in dollars -/
def new_refrigerator_cost : ℝ := 0.45

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The amount Kurt saves in a month with his new refrigerator in dollars -/
def monthly_savings : ℝ := 12

/-- The daily cost of Kurt's old refrigerator in dollars -/
def old_refrigerator_cost : ℝ := 0.85

theorem old_refrigerator_cost_proof : 
  old_refrigerator_cost * days_in_month - new_refrigerator_cost * days_in_month = monthly_savings :=
sorry

end NUMINAMATH_CALUDE_old_refrigerator_cost_proof_l2837_283731


namespace NUMINAMATH_CALUDE_min_c_plus_d_is_15_l2837_283707

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

theorem min_c_plus_d_is_15 :
  ∀ (A B C D : Digit),
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A.val + B.val : ℕ) ≠ 0 →
    (C.val + D.val : ℕ) ≠ 0 →
    (A.val + B.val : ℕ) < (C.val + D.val) →
    (C.val + D.val) % (A.val + B.val) = 0 →
    ∀ (E F G H : Digit),
      E ≠ F → E ≠ G → E ≠ H → F ≠ G → F ≠ H → G ≠ H →
      (E.val + F.val : ℕ) ≠ 0 →
      (G.val + H.val : ℕ) ≠ 0 →
      (E.val + F.val : ℕ) < (G.val + H.val) →
      (G.val + H.val) % (E.val + F.val) = 0 →
      (A.val + B.val : ℕ) / (C.val + D.val : ℕ) ≤ (E.val + F.val : ℕ) / (G.val + H.val : ℕ) →
      (C.val + D.val : ℕ) ≤ 15 :=
by sorry

#check min_c_plus_d_is_15

end NUMINAMATH_CALUDE_min_c_plus_d_is_15_l2837_283707


namespace NUMINAMATH_CALUDE_function_identity_l2837_283777

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : 
  ∀ n : ℕ+, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2837_283777


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l2837_283759

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Theorem for the first question
theorem union_condition (a : ℝ) : A ∪ B a = B a → a = 1 := by sorry

-- Theorem for the second question
theorem intersection_condition (a : ℝ) : A ∩ B a = B a → a ≤ -1 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l2837_283759


namespace NUMINAMATH_CALUDE_vacation_class_ratio_l2837_283789

theorem vacation_class_ratio :
  ∀ (grant_vacations : ℕ) (kelvin_classes : ℕ),
    kelvin_classes = 90 →
    grant_vacations + kelvin_classes = 450 →
    (grant_vacations : ℚ) / kelvin_classes = 4 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_class_ratio_l2837_283789


namespace NUMINAMATH_CALUDE_processing_time_theorem_l2837_283714

/-- Calculates the total processing time in hours for a set of pictures --/
def total_processing_time (tree_count : ℕ) (flower_count : ℕ) (grass_count : ℕ) 
  (tree_time : ℚ) (flower_time : ℚ) (grass_time : ℚ) : ℚ :=
  ((tree_count : ℚ) * tree_time + (flower_count : ℚ) * flower_time + (grass_count : ℚ) * grass_time) / 60

/-- Theorem stating the total processing time for the given set of pictures --/
theorem processing_time_theorem : 
  total_processing_time 320 400 240 (3/2) (5/2) 1 = 860/30 := by
  sorry

end NUMINAMATH_CALUDE_processing_time_theorem_l2837_283714


namespace NUMINAMATH_CALUDE_count_D_eq_3_is_13_l2837_283785

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D_eq_3 : ℕ := sorry

theorem count_D_eq_3_is_13 : count_D_eq_3 = 13 := by sorry

end NUMINAMATH_CALUDE_count_D_eq_3_is_13_l2837_283785


namespace NUMINAMATH_CALUDE_pentagon_area_half_octagon_l2837_283774

/-- Regular octagon with vertices labeled CHILDREN -/
structure RegularOctagon :=
  (vertices : Fin 8 → ℝ × ℝ)
  (is_regular : sorry)

/-- Pentagon formed by 5 consecutive vertices of the octagon -/
def Pentagon (o : RegularOctagon) : Set (ℝ × ℝ) :=
  {p | ∃ i : Fin 5, p = o.vertices i}

/-- Area of a shape in ℝ² -/
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

theorem pentagon_area_half_octagon (o : RegularOctagon) 
  (h : area {p | ∃ i : Fin 8, p = o.vertices i} = 1) : 
  area (Pentagon o) = 1/2 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_half_octagon_l2837_283774


namespace NUMINAMATH_CALUDE_hyperbola_from_circle_intersection_l2837_283723

/-- Circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 9 = 0

/-- Points A and B on y-axis -/
def points_on_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ B.1 = 0 ∧ circle_eq A.1 A.2 ∧ circle_eq B.1 B.2

/-- Points A and B trisect focal distance -/
def trisect_focal_distance (A B : ℝ × ℝ) (c : ℝ) : Prop :=
  abs (A.2 - B.2) = 2 * c / 3

/-- Standard hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := y^2/9 - x^2/72 = 1

/-- Main theorem -/
theorem hyperbola_from_circle_intersection :
  ∀ (A B : ℝ × ℝ) (c : ℝ),
  points_on_y_axis A B →
  trisect_focal_distance A B c →
  ∀ (x y : ℝ), hyperbola_eq x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_from_circle_intersection_l2837_283723


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2837_283763

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 / 4 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = 2 * Real.sqrt 3 / 3 * x ∨ y = -2 * Real.sqrt 3 / 3 * x

/-- Theorem stating that the given asymptote equations are correct for the given hyperbola -/
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2837_283763


namespace NUMINAMATH_CALUDE_kosher_meals_count_l2837_283721

/-- Calculates the number of kosher meals given the total number of clients,
    number of vegan meals, number of both vegan and kosher meals,
    and number of meals that are neither vegan nor kosher. -/
def kosher_meals (total : ℕ) (vegan : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  total - neither - (vegan - both)

/-- Proves that the number of clients needing kosher meals is 8,
    given the specific conditions from the problem. -/
theorem kosher_meals_count :
  kosher_meals 30 7 3 18 = 8 := by
  sorry

end NUMINAMATH_CALUDE_kosher_meals_count_l2837_283721


namespace NUMINAMATH_CALUDE_range_f_minus_g_theorem_l2837_283705

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the range of f + g
def range_f_plus_g (f g : ℝ → ℝ) : Set ℝ := Set.range (λ x ↦ f x + g x)

-- Define the range of f - g
def range_f_minus_g (f g : ℝ → ℝ) : Set ℝ := Set.range (λ x ↦ f x - g x)

-- State the theorem
theorem range_f_minus_g_theorem (hf : is_odd f) (hg : is_even g) 
  (h_range : range_f_plus_g f g = Set.Icc 1 3) : 
  range_f_minus_g f g = Set.Ioc (-3) (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_f_minus_g_theorem_l2837_283705


namespace NUMINAMATH_CALUDE_smallest_base_for_145_l2837_283779

theorem smallest_base_for_145 :
  ∀ b : ℕ, b ≥ 2 →
    (∀ n : ℕ, n ≥ 2 ∧ n < b → n^2 ≤ 145 ∧ 145 < n^3) →
    b = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_145_l2837_283779


namespace NUMINAMATH_CALUDE_intersection_complement_A_and_B_range_of_a_for_C_subset_A_l2837_283711

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x < 0}
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}

-- Define the set C
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 1}

-- Statement 1
theorem intersection_complement_A_and_B :
  (Set.compl A ∩ B) = {x : ℝ | x ≥ 0} := by sorry

-- Statement 2
theorem range_of_a_for_C_subset_A :
  {a : ℝ | C a ⊆ A} = {a : ℝ | a ≤ -1/2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_A_and_B_range_of_a_for_C_subset_A_l2837_283711
