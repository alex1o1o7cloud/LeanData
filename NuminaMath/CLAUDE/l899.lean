import Mathlib

namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2s_l899_89974

-- Define the distance function
def S (t : ℝ) : ℝ := 2 * (1 - t)^2

-- Define the instantaneous velocity function (derivative of S)
def v (t : ℝ) : ℝ := -4 * (1 - t)

-- Theorem statement
theorem instantaneous_velocity_at_2s :
  v 2 = 4 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2s_l899_89974


namespace NUMINAMATH_CALUDE_not_valid_prism_diagonals_5_6_9_not_valid_prism_diagonals_7_8_11_l899_89929

/-- A function to check if three positive real numbers can represent the lengths of external diagonals of a right regular prism -/
def is_valid_prism_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 > c^2 ∧
  b^2 + c^2 > a^2 ∧
  c^2 + a^2 > b^2

/-- Theorem stating that {5,6,9} cannot be the lengths of external diagonals of a right regular prism -/
theorem not_valid_prism_diagonals_5_6_9 :
  ¬ is_valid_prism_diagonals 5 6 9 :=
sorry

/-- Theorem stating that {7,8,11} cannot be the lengths of external diagonals of a right regular prism -/
theorem not_valid_prism_diagonals_7_8_11 :
  ¬ is_valid_prism_diagonals 7 8 11 :=
sorry

end NUMINAMATH_CALUDE_not_valid_prism_diagonals_5_6_9_not_valid_prism_diagonals_7_8_11_l899_89929


namespace NUMINAMATH_CALUDE_invalid_inequality_l899_89994

theorem invalid_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬(1 / (a - b) > 1 / a) := by
  sorry

end NUMINAMATH_CALUDE_invalid_inequality_l899_89994


namespace NUMINAMATH_CALUDE_nikitas_claim_incorrect_l899_89904

theorem nikitas_claim_incorrect : ¬∃ (x y : ℕ), 5 * (x - y) = 49 := by
  sorry

end NUMINAMATH_CALUDE_nikitas_claim_incorrect_l899_89904


namespace NUMINAMATH_CALUDE_base4_sum_234_73_l899_89964

/-- Converts a number from base 4 to base 10 -/
def base4_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10_to_base4 (n : ℕ) : ℕ := sorry

/-- The sum of two numbers in base 4 -/
def base4_sum (a b : ℕ) : ℕ :=
  base10_to_base4 (base4_to_base10 a + base4_to_base10 b)

theorem base4_sum_234_73 : base4_sum 234 73 = 10303 := by sorry

end NUMINAMATH_CALUDE_base4_sum_234_73_l899_89964


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l899_89966

theorem ice_cream_sundaes (n : ℕ) (k : ℕ) : n = 8 ∧ k = 2 → Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l899_89966


namespace NUMINAMATH_CALUDE_triangle_square_ratio_l899_89924

/-- A triangle in a 2D plane --/
structure Triangle :=
  (a b c : ℝ)
  (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (hineq : a + b > c ∧ b + c > a ∧ c + a > b)

/-- The side length of the largest square inscribed in a triangle --/
noncomputable def maxInscribedSquareSide (t : Triangle) : ℝ := sorry

/-- The side length of the smallest square circumscribed around a triangle --/
noncomputable def minCircumscribedSquareSide (t : Triangle) : ℝ := sorry

/-- A triangle is right-angled if one of its angles is 90 degrees --/
def isRightTriangle (t : Triangle) : Prop := sorry

theorem triangle_square_ratio (t : Triangle) :
  minCircumscribedSquareSide t / maxInscribedSquareSide t ≥ 2 ∧
  (minCircumscribedSquareSide t / maxInscribedSquareSide t = 2 ↔ isRightTriangle t) :=
sorry

end NUMINAMATH_CALUDE_triangle_square_ratio_l899_89924


namespace NUMINAMATH_CALUDE_limit_between_exponentials_l899_89920

theorem limit_between_exponentials (a : ℝ) (ha : a > 0) :
  Real.exp a < (Real.exp (a + 1)) / (Real.exp 1 - 1) ∧
  (Real.exp (a + 1)) / (Real.exp 1 - 1) < Real.exp (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_limit_between_exponentials_l899_89920


namespace NUMINAMATH_CALUDE_one_element_condition_at_most_one_element_condition_l899_89970

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

-- Theorem 1
theorem one_element_condition (a : ℝ) :
  (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = 1) := by sorry

-- Theorem 2
theorem at_most_one_element_condition (a : ℝ) :
  (∃ x, x ∈ A a → ∀ y, y ∈ A a → x = y) ↔ (a ≥ 1 ∨ a = 0) := by sorry

end NUMINAMATH_CALUDE_one_element_condition_at_most_one_element_condition_l899_89970


namespace NUMINAMATH_CALUDE_corn_acreage_l899_89968

theorem corn_acreage (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : beans_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) : 
  (total_land * corn_ratio) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l899_89968


namespace NUMINAMATH_CALUDE_julia_balls_count_l899_89940

/-- The number of balls Julia bought -/
def total_balls (red_packs yellow_packs green_packs balls_per_pack : ℕ) : ℕ :=
  (red_packs + yellow_packs + green_packs) * balls_per_pack

/-- Proof that Julia bought 399 balls -/
theorem julia_balls_count :
  total_balls 3 10 8 19 = 399 := by
  sorry

end NUMINAMATH_CALUDE_julia_balls_count_l899_89940


namespace NUMINAMATH_CALUDE_x_plus_x_squared_l899_89990

theorem x_plus_x_squared (x : ℕ) (h : x = 3) : x + (x * x) = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_x_squared_l899_89990


namespace NUMINAMATH_CALUDE_ab_power_2022_l899_89983

theorem ab_power_2022 (a b : ℝ) (h : |3*a + 1| + (b - 3)^2 = 0) : (a*b)^2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_power_2022_l899_89983


namespace NUMINAMATH_CALUDE_exercise_minutes_proof_l899_89913

def javier_minutes : ℕ := 50
def javier_days : ℕ := 10

def sanda_minutes_1 : ℕ := 90
def sanda_days_1 : ℕ := 3
def sanda_minutes_2 : ℕ := 75
def sanda_days_2 : ℕ := 2
def sanda_minutes_3 : ℕ := 45
def sanda_days_3 : ℕ := 4

def total_exercise_minutes : ℕ := 1100

theorem exercise_minutes_proof :
  (javier_minutes * javier_days) +
  (sanda_minutes_1 * sanda_days_1) +
  (sanda_minutes_2 * sanda_days_2) +
  (sanda_minutes_3 * sanda_days_3) = total_exercise_minutes :=
by sorry

end NUMINAMATH_CALUDE_exercise_minutes_proof_l899_89913


namespace NUMINAMATH_CALUDE_smallest_q_for_decimal_sequence_l899_89907

theorem smallest_q_for_decimal_sequence (p q : ℕ+) : 
  (p : ℚ) / q = 0.123456789 → q ≥ 10989019 := by sorry

end NUMINAMATH_CALUDE_smallest_q_for_decimal_sequence_l899_89907


namespace NUMINAMATH_CALUDE_equation_satisfied_at_eight_l899_89951

def f (x : ℝ) : ℝ := 2 * x - 3

theorem equation_satisfied_at_eight :
  ∃ x : ℝ, 2 * (f x) - 21 = f (x - 4) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_at_eight_l899_89951


namespace NUMINAMATH_CALUDE_ellipse_and_midpoint_trajectory_l899_89972

/-- Definition of the ellipse -/
def Ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the midpoint trajectory -/
def MidpointTrajectory (x y : ℝ) : Prop := (x - 1/2)^2 + 4 * (y - 1/4)^2 = 1

/-- Theorem: The standard equation of the ellipse and the midpoint trajectory -/
theorem ellipse_and_midpoint_trajectory :
  (∀ x y, Ellipse x y ↔ x^2 / 4 + y^2 = 1) ∧
  (∀ x₀ y₀ x y, Ellipse x₀ y₀ → x = (x₀ + 1) / 2 ∧ y = (y₀ + 1/2) / 2 → MidpointTrajectory x y) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_midpoint_trajectory_l899_89972


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l899_89943

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := m + 1 + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l899_89943


namespace NUMINAMATH_CALUDE_circle_area_and_diameter_l899_89988

/-- For a circle with circumference 36 cm, prove its area and diameter -/
theorem circle_area_and_diameter (C : ℝ) (h : C = 36) :
  ∃ (A d : ℝ),
    A = 324 / Real.pi ∧
    d = 36 / Real.pi ∧
    C = Real.pi * d ∧
    A = Real.pi * (d / 2)^2 := by
sorry


end NUMINAMATH_CALUDE_circle_area_and_diameter_l899_89988


namespace NUMINAMATH_CALUDE_british_flag_theorem_expected_value_zero_l899_89918

/-- A rectangle in a 2D plane -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: For any rectangle ABCD and any point P, AP^2 + CP^2 - BP^2 - DP^2 = 0 -/
theorem british_flag_theorem (rect : Rectangle) (P : ℝ × ℝ) :
  distanceSquared rect.A P + distanceSquared rect.C P
  = distanceSquared rect.B P + distanceSquared rect.D P := by
  sorry

/-- Corollary: The expected value of AP^2 + CP^2 - BP^2 - DP^2 is always 0 -/
theorem expected_value_zero (rect : Rectangle) :
  ∃ E : ℝ, E = 0 ∧ ∀ P : ℝ × ℝ,
    E = distanceSquared rect.A P + distanceSquared rect.C P
      - distanceSquared rect.B P - distanceSquared rect.D P := by
  sorry

end NUMINAMATH_CALUDE_british_flag_theorem_expected_value_zero_l899_89918


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l899_89980

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 + 3 * x) ↔ x ≥ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l899_89980


namespace NUMINAMATH_CALUDE_game_result_l899_89981

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 9
  else if n % 2 = 0 then 3
  else 1

def allie_rolls : List ℕ := [6, 3, 4]
def betty_rolls : List ℕ := [1, 2, 5, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : 
  total_points allie_rolls * total_points betty_rolls = 294 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l899_89981


namespace NUMINAMATH_CALUDE_only_satisfying_sets_l899_89901

/-- A set of four real numbers satisfying the given condition -/
def SatisfyingSet (a b c d : ℝ) : Prop :=
  a + b*c*d = 2 ∧ b + a*c*d = 2 ∧ c + a*b*d = 2 ∧ d + a*b*c = 2

/-- The theorem stating the only satisfying sets -/
theorem only_satisfying_sets :
  ∀ a b c d : ℝ, SatisfyingSet a b c d ↔ 
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
    (a = -1 ∧ b = -1 ∧ c = -1 ∧ d = 3) ∨
    (a = -1 ∧ b = -1 ∧ c = 3 ∧ d = -1) ∨
    (a = -1 ∧ b = 3 ∧ c = -1 ∧ d = -1) ∨
    (a = 3 ∧ b = -1 ∧ c = -1 ∧ d = -1) :=
by sorry

end NUMINAMATH_CALUDE_only_satisfying_sets_l899_89901


namespace NUMINAMATH_CALUDE_max_angle_cone_from_semicircle_l899_89950

/-- The maximum angle between generatrices of a cone formed by a semicircle -/
theorem max_angle_cone_from_semicircle :
  ∀ (r : ℝ),
  r > 0 →
  let semicircle_arc_length := r * Real.pi
  let base_circumference := 2 * r * Real.pi / 2
  semicircle_arc_length = base_circumference →
  ∃ (θ : ℝ),
  θ = 60 * (Real.pi / 180) ∧
  ∀ (α : ℝ),
  (α ≥ 0 ∧ α ≤ θ) →
  ∃ (g₁ g₂ : ℝ × ℝ),
  (g₁.1 - g₂.1)^2 + (g₁.2 - g₂.2)^2 ≤ r^2 ∧
  Real.arccos ((g₁.1 * g₂.1 + g₁.2 * g₂.2) / r^2) = α :=
by sorry

end NUMINAMATH_CALUDE_max_angle_cone_from_semicircle_l899_89950


namespace NUMINAMATH_CALUDE_manager_average_salary_l899_89987

/-- Represents the salary distribution in Plutarch Enterprises -/
structure SalaryDistribution where
  total_employees : ℝ
  marketer_ratio : ℝ
  engineer_ratio : ℝ
  marketer_salary : ℝ
  engineer_salary : ℝ
  average_salary : ℝ

/-- Theorem stating the average salary of managers in Plutarch Enterprises -/
theorem manager_average_salary (sd : SalaryDistribution) 
  (h1 : sd.marketer_ratio = 0.7)
  (h2 : sd.engineer_ratio = 0.1)
  (h3 : sd.marketer_salary = 50000)
  (h4 : sd.engineer_salary = 80000)
  (h5 : sd.average_salary = 80000) :
  (sd.average_salary * sd.total_employees - 
   (sd.marketer_ratio * sd.marketer_salary * sd.total_employees + 
    sd.engineer_ratio * sd.engineer_salary * sd.total_employees)) / 
   ((1 - sd.marketer_ratio - sd.engineer_ratio) * sd.total_employees) = 185000 := by
  sorry

end NUMINAMATH_CALUDE_manager_average_salary_l899_89987


namespace NUMINAMATH_CALUDE_claire_gerbils_l899_89911

/-- Represents the number of gerbils Claire has -/
def num_gerbils : ℕ := 60

/-- Represents the number of hamsters Claire has -/
def num_hamsters : ℕ := 30

/-- The total number of pets Claire has -/
def total_pets : ℕ := 90

/-- The total number of male pets Claire has -/
def total_male_pets : ℕ := 25

theorem claire_gerbils :
  (num_gerbils + num_hamsters = total_pets) ∧
  (num_gerbils / 4 + num_hamsters / 3 = total_male_pets) →
  num_gerbils = 60 := by
  sorry

end NUMINAMATH_CALUDE_claire_gerbils_l899_89911


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l899_89971

/-- The area of wrapping paper required for a rectangular box -/
def wrapping_paper_area (l w h : ℝ) : ℝ := l * w + 2 * l * h + 2 * w * h + 4 * h^2

/-- Theorem stating the area of wrapping paper required for a rectangular box -/
theorem wrapping_paper_area_theorem (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  let box_volume := l * w * h
  let paper_length := l + 2 * h
  let paper_width := w + 2 * h
  let paper_area := paper_length * paper_width
  paper_area = wrapping_paper_area l w h :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l899_89971


namespace NUMINAMATH_CALUDE_sequence_increasing_l899_89953

theorem sequence_increasing (n : ℕ+) : 
  let a : ℕ+ → ℚ := fun k => k / (k + 2)
  a n < a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_l899_89953


namespace NUMINAMATH_CALUDE_sticker_distribution_l899_89976

theorem sticker_distribution (n k : ℕ) (hn : n = 10) (hk : k = 5) :
  Nat.choose (n + k - 1) (k - 1) = 1001 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l899_89976


namespace NUMINAMATH_CALUDE_extreme_point_of_f_l899_89977

/-- The function f(x) = 2x^2 - 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4 * x

theorem extreme_point_of_f :
  ∃! x : ℝ, ∀ y : ℝ, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_extreme_point_of_f_l899_89977


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l899_89932

theorem simple_interest_rate_for_doubling (principal : ℝ) (h : principal > 0) : 
  ∃ (rate : ℝ), 
    (rate > 0) ∧ 
    (rate < 100) ∧
    (principal * (1 + rate / 100 * 25) = 2 * principal) ∧
    (rate = 4) := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l899_89932


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_l899_89945

theorem mean_equality_implies_y_value : ∃ y : ℝ,
  (4 + 7 + 11 + 14) / 4 = (10 + y + 5) / 3 ∧ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_l899_89945


namespace NUMINAMATH_CALUDE_share_of_y_l899_89947

/-- The share of y in a sum divided among x, y, and z, where for each rupee x gets,
    y gets 45 paisa and z gets 50 paisa, and the total amount is Rs. 78. -/
theorem share_of_y (x y z : ℝ) : 
  x + y + z = 78 →  -- Total amount condition
  y = 0.45 * x →    -- Relationship between y and x
  z = 0.5 * x →     -- Relationship between z and x
  y = 18 :=         -- Share of y
by sorry

end NUMINAMATH_CALUDE_share_of_y_l899_89947


namespace NUMINAMATH_CALUDE_mias_gift_spending_l899_89937

theorem mias_gift_spending (total_spending : ℕ) (num_siblings : ℕ) (parent_gift : ℕ) (num_parents : ℕ) 
  (h1 : total_spending = 150)
  (h2 : num_siblings = 3)
  (h3 : parent_gift = 30)
  (h4 : num_parents = 2) :
  (total_spending - num_parents * parent_gift) / num_siblings = 30 := by
  sorry

end NUMINAMATH_CALUDE_mias_gift_spending_l899_89937


namespace NUMINAMATH_CALUDE_parabola_c_is_negative_eighteen_l899_89998

/-- A parabola passing through two given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  pass_through_point1 : 2 * 2^2 + b * 2 + c = 6
  pass_through_point2 : 2 * (-3)^2 + b * (-3) + c = -24

/-- The value of c for the parabola -/
def parabola_c_value (p : Parabola) : ℝ := -18

/-- Theorem stating that the value of c for the parabola is -18 -/
theorem parabola_c_is_negative_eighteen (p : Parabola) : 
  parabola_c_value p = p.c := by sorry

end NUMINAMATH_CALUDE_parabola_c_is_negative_eighteen_l899_89998


namespace NUMINAMATH_CALUDE_exists_function_satisfying_condition_l899_89956

theorem exists_function_satisfying_condition : 
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = |x + 1| := by
  sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_condition_l899_89956


namespace NUMINAMATH_CALUDE_andrew_flooring_planks_l899_89952

/-- The number of wooden planks Andrew bought for his flooring project -/
def total_planks : ℕ := 91

/-- The number of planks used in Andrew's bedroom -/
def bedroom_planks : ℕ := 8

/-- The number of planks used in the living room -/
def living_room_planks : ℕ := 20

/-- The number of planks used in the kitchen -/
def kitchen_planks : ℕ := 11

/-- The number of planks used in the dining room -/
def dining_room_planks : ℕ := 13

/-- The number of planks used in the guest bedroom -/
def guest_bedroom_planks : ℕ := bedroom_planks - 2

/-- The number of planks used in each hallway -/
def hallway_planks : ℕ := 4

/-- The number of planks used in the study -/
def study_planks : ℕ := guest_bedroom_planks + 3

/-- The number of planks ruined in each bedroom -/
def bedroom_ruined_planks : ℕ := 3

/-- The number of planks ruined in the living room -/
def living_room_ruined_planks : ℕ := 2

/-- The number of planks ruined in the study -/
def study_ruined_planks : ℕ := 1

/-- The number of leftover planks -/
def leftover_planks : ℕ := 7

/-- The number of hallways -/
def number_of_hallways : ℕ := 2

theorem andrew_flooring_planks :
  total_planks = 
    bedroom_planks + bedroom_ruined_planks +
    living_room_planks + living_room_ruined_planks +
    kitchen_planks +
    dining_room_planks +
    guest_bedroom_planks + bedroom_ruined_planks +
    (hallway_planks * number_of_hallways) +
    study_planks + study_ruined_planks +
    leftover_planks :=
by sorry

end NUMINAMATH_CALUDE_andrew_flooring_planks_l899_89952


namespace NUMINAMATH_CALUDE_ram_price_calculation_ram_price_theorem_l899_89928

theorem ram_price_calculation (initial_price : ℝ) 
  (increase_percentage : ℝ) (decrease_percentage : ℝ) : ℝ :=
  let increased_price := initial_price * (1 + increase_percentage)
  let final_price := increased_price * (1 - decrease_percentage)
  final_price

theorem ram_price_theorem : 
  ram_price_calculation 50 0.3 0.2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ram_price_calculation_ram_price_theorem_l899_89928


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l899_89961

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 140) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 21) : 
  2 * (a*b + b*c + c*a) = 784 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l899_89961


namespace NUMINAMATH_CALUDE_select_and_arrange_theorem_l899_89957

/-- The number of ways to select k items from n items -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange k items -/
def permutation (k : ℕ) : ℕ := Nat.factorial k

/-- The total number of people -/
def total_people : ℕ := 9

/-- The number of people to be selected and arranged -/
def selected_people : ℕ := 3

/-- The number of ways to select and arrange people -/
def ways_to_select_and_arrange : ℕ := combination total_people selected_people * permutation selected_people

theorem select_and_arrange_theorem : ways_to_select_and_arrange = 504 := by
  sorry

end NUMINAMATH_CALUDE_select_and_arrange_theorem_l899_89957


namespace NUMINAMATH_CALUDE_acute_angles_inequality_l899_89997

theorem acute_angles_inequality (α β : Real) 
  (h_α_acute : 0 < α ∧ α < π / 2) 
  (h_β_acute : 0 < β ∧ β < π / 2) : 
  Real.sin α ^ 3 * Real.cos β ^ 3 + Real.sin α ^ 3 * Real.sin β ^ 3 + Real.cos α ^ 3 ≥ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_inequality_l899_89997


namespace NUMINAMATH_CALUDE_smallest_percent_increase_l899_89982

-- Define the values for the relevant questions
def value : Nat → ℕ
| 1 => 100
| 2 => 300
| 3 => 400
| 4 => 700
| 12 => 180000
| 13 => 360000
| 14 => 720000
| 15 => 1440000
| _ => 0  -- Default case, not used in our problem

-- Define the percent increase function
def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

-- Theorem statement
theorem smallest_percent_increase :
  let increase_1_2 := percent_increase (value 1) (value 2)
  let increase_2_3 := percent_increase (value 2) (value 3)
  let increase_3_4 := percent_increase (value 3) (value 4)
  let increase_12_13 := percent_increase (value 12) (value 13)
  let increase_14_15 := percent_increase (value 14) (value 15)
  increase_2_3 < increase_1_2 ∧
  increase_2_3 < increase_3_4 ∧
  increase_2_3 < increase_12_13 ∧
  increase_2_3 < increase_14_15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_percent_increase_l899_89982


namespace NUMINAMATH_CALUDE_expression_evaluation_l899_89954

theorem expression_evaluation : (255^2 - 231^2 - (231^2 - 207^2)) / 24 = 48 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l899_89954


namespace NUMINAMATH_CALUDE_chess_playoff_orders_l899_89923

/-- Represents the structure of a chess playoff tournament --/
structure ChessPlayoff where
  numPlayers : Nat
  numMatches : Nat
  firstMatchPlayers : Fin 3 × Fin 3
  secondMatchPlayer : Fin 3

/-- Calculates the number of possible prize orders in a chess playoff tournament --/
def numPossibleOrders (tournament : ChessPlayoff) : Nat :=
  2^tournament.numMatches

/-- Theorem stating that the number of possible prize orders in the given tournament structure is 4 --/
theorem chess_playoff_orders (tournament : ChessPlayoff) 
  (h1 : tournament.numPlayers = 3)
  (h2 : tournament.numMatches = 2)
  (h3 : tournament.firstMatchPlayers = (⟨2, by norm_num⟩, ⟨1, by norm_num⟩))
  (h4 : tournament.secondMatchPlayer = ⟨0, by norm_num⟩) :
  numPossibleOrders tournament = 4 := by
  sorry


end NUMINAMATH_CALUDE_chess_playoff_orders_l899_89923


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_thirds_l899_89991

-- Define the polynomial
def f (x : ℝ) : ℝ := (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7)

-- Theorem statement
theorem sum_of_roots_equals_fourteen_thirds :
  ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = 14/3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_thirds_l899_89991


namespace NUMINAMATH_CALUDE_sum_lent_calculation_l899_89967

/-- Calculates the sum lent given the interest rate, time period, and interest amount -/
theorem sum_lent_calculation (interest_rate : ℚ) (years : ℕ) (interest_difference : ℚ) : 
  interest_rate = 5 / 100 →
  years = 8 →
  interest_difference = 360 →
  (1 - years * interest_rate) * 600 = interest_difference :=
by
  sorry

end NUMINAMATH_CALUDE_sum_lent_calculation_l899_89967


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_y_value_l899_89934

theorem sqrt_equality_implies_y_value (y : ℝ) :
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 8 → y = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_y_value_l899_89934


namespace NUMINAMATH_CALUDE_cross_in_square_l899_89912

theorem cross_in_square (s : ℝ) (h : s > 0) : 
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → s = 36 := by
  sorry

end NUMINAMATH_CALUDE_cross_in_square_l899_89912


namespace NUMINAMATH_CALUDE_vegetarians_count_l899_89939

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  onlyVeg : ℕ
  onlyNonVeg : ℕ
  bothVegAndNonVeg : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def totalVegetarians (fd : FamilyDiet) : ℕ :=
  fd.onlyVeg + fd.bothVegAndNonVeg

/-- Theorem stating that the number of vegetarians in the family is 28 -/
theorem vegetarians_count (fd : FamilyDiet) 
  (h1 : fd.onlyVeg = 16)
  (h2 : fd.onlyNonVeg = 9)
  (h3 : fd.bothVegAndNonVeg = 12) :
  totalVegetarians fd = 28 := by
  sorry

end NUMINAMATH_CALUDE_vegetarians_count_l899_89939


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l899_89989

theorem quadratic_expression_value (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 8) 
  (eq2 : 2 * x + 3 * y = 11) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 2041 / 25 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l899_89989


namespace NUMINAMATH_CALUDE_zoo_cost_theorem_l899_89936

def zoo_cost (goat_price : ℚ) (goat_count : ℕ) (llama_price_factor : ℚ) 
              (kangaroo_price_factor : ℚ) (kangaroo_multiple : ℕ) 
              (discount_rate : ℚ) : ℚ :=
  let llama_count := 2 * goat_count
  let kangaroo_count := kangaroo_multiple * 5
  let llama_price := goat_price * (1 + llama_price_factor)
  let kangaroo_price := llama_price * (1 - kangaroo_price_factor)
  let goat_cost := goat_price * goat_count
  let llama_cost := llama_price * llama_count
  let kangaroo_cost := kangaroo_price * kangaroo_count
  let total_cost := goat_cost + llama_cost + kangaroo_cost
  let discounted_cost := total_cost * (1 - discount_rate)
  discounted_cost

theorem zoo_cost_theorem : 
  zoo_cost 400 3 (1/2) (1/4) 2 (1/10) = 8850 := by sorry

end NUMINAMATH_CALUDE_zoo_cost_theorem_l899_89936


namespace NUMINAMATH_CALUDE_tomato_price_equation_l899_89900

/-- The original price per pound of tomatoes -/
def P : ℝ := sorry

/-- The selling price per pound of remaining tomatoes -/
def selling_price : ℝ := 0.968888888888889

/-- The proportion of tomatoes that were not ruined -/
def remaining_proportion : ℝ := 0.9

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.09

theorem tomato_price_equation : 
  (1 + profit_percentage) * P = selling_price * remaining_proportion := by sorry

end NUMINAMATH_CALUDE_tomato_price_equation_l899_89900


namespace NUMINAMATH_CALUDE_square_difference_equality_l899_89935

theorem square_difference_equality : (2 + 3)^2 - (2^2 + 3^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l899_89935


namespace NUMINAMATH_CALUDE_geography_book_count_l899_89999

/-- Given a shelf of books with specific counts, calculate the number of geography books. -/
theorem geography_book_count (total : ℕ) (history : ℕ) (math : ℕ) 
  (h_total : total = 100)
  (h_history : history = 32)
  (h_math : math = 43) :
  total - history - math = 25 := by
  sorry

end NUMINAMATH_CALUDE_geography_book_count_l899_89999


namespace NUMINAMATH_CALUDE_julies_earnings_l899_89973

/-- Calculates Julie's earnings for landscaping services --/
def calculate_earnings (
  lawn_rate : ℚ)
  (weed_rate : ℚ)
  (prune_rate : ℚ)
  (mulch_rate : ℚ)
  (lawn_hours_sept : ℚ)
  (weed_hours_sept : ℚ)
  (prune_hours_sept : ℚ)
  (mulch_hours_sept : ℚ) : ℚ :=
  let sept_earnings := 
    lawn_rate * lawn_hours_sept +
    weed_rate * weed_hours_sept +
    prune_rate * prune_hours_sept +
    mulch_rate * mulch_hours_sept
  let oct_earnings := 
    lawn_rate * (lawn_hours_sept * 1.5) +
    weed_rate * (weed_hours_sept * 1.5) +
    prune_rate * (prune_hours_sept * 1.5) +
    mulch_rate * (mulch_hours_sept * 1.5)
  sept_earnings + oct_earnings

/-- Theorem: Julie's total earnings for September and October --/
theorem julies_earnings : 
  calculate_earnings 4 8 10 12 25 3 10 5 = 710 := by
  sorry

end NUMINAMATH_CALUDE_julies_earnings_l899_89973


namespace NUMINAMATH_CALUDE_dividend_calculation_l899_89942

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h_remainder : remainder = 19)
  (h_quotient : quotient = 61)
  (h_divisor : divisor = 8) :
  divisor * quotient + remainder = 507 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l899_89942


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l899_89925

theorem geometric_sequence_product (x y z : ℝ) : 
  1 < x ∧ x < y ∧ y < z ∧ z < 4 →
  (∃ r : ℝ, r > 0 ∧ x = r * 1 ∧ y = r * x ∧ z = r * y ∧ 4 = r * z) →
  1 * x * y * z * 4 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l899_89925


namespace NUMINAMATH_CALUDE_factor_statements_l899_89906

theorem factor_statements : 
  (∃ n : ℤ, 30 = 5 * n) ∧ (∃ m : ℤ, 180 = 9 * m) := by sorry

end NUMINAMATH_CALUDE_factor_statements_l899_89906


namespace NUMINAMATH_CALUDE_inverse_sum_product_l899_89992

theorem inverse_sum_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : 3*x + y/3 + z ≠ 0) : 
  (3*x + y/3 + z)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹ + z⁻¹) = (x*y*z)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_product_l899_89992


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l899_89995

/-- A point P(x, y) is in the second quadrant if and only if x < 0 and y > 0 -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P is a - 2 -/
def x_coordinate (a : ℝ) : ℝ := a - 2

/-- The y-coordinate of point P is 2 -/
def y_coordinate : ℝ := 2

/-- Theorem: For a point P(a-2, 2) to be in the second quadrant, a must be less than 2 -/
theorem point_in_second_quadrant (a : ℝ) : 
  second_quadrant (x_coordinate a) y_coordinate ↔ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l899_89995


namespace NUMINAMATH_CALUDE_range_of_a_l899_89908

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≥ 0

-- Define the set A corresponding to ¬p
def A : Set ℝ := {x | x < -2 ∨ x > 10}

-- Define the set B corresponding to q
def B (a : ℝ) : Set ℝ := {x | x ≤ 1 - a ∨ x ≥ 1 + a}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (a > 0 ∧ A ⊆ B a ∧ A ≠ B a) → (0 < a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l899_89908


namespace NUMINAMATH_CALUDE_min_tan_angle_l899_89921

/-- The set of complex numbers with nonnegative real and imaginary parts -/
def S : Set ℂ :=
  {z : ℂ | z.re ≥ 0 ∧ z.im ≥ 0}

/-- The condition |z^2 + 2| ≤ |z| -/
def satisfiesCondition (z : ℂ) : Prop :=
  Complex.abs (z^2 + 2) ≤ Complex.abs z

/-- The angle between a complex number and the real axis -/
noncomputable def angle (z : ℂ) : ℝ :=
  Real.arctan (z.im / z.re)

/-- The main theorem -/
theorem min_tan_angle :
  ∃ (min_tan : ℝ), min_tan = Real.sqrt 7 ∧
  ∀ z ∈ S, satisfiesCondition z →
  Real.tan (angle z) ≥ min_tan :=
sorry

end NUMINAMATH_CALUDE_min_tan_angle_l899_89921


namespace NUMINAMATH_CALUDE_sarah_candy_count_l899_89986

/-- The number of candy pieces Sarah received for Halloween -/
def total_candy : ℕ := sorry

/-- The number of candy pieces Sarah ate -/
def eaten_candy : ℕ := 36

/-- The number of piles Sarah made with the remaining candy -/
def number_of_piles : ℕ := 8

/-- The number of candy pieces in each pile -/
def pieces_per_pile : ℕ := 9

/-- Theorem stating that the total number of candy pieces Sarah received is 108 -/
theorem sarah_candy_count : total_candy = 108 := by
  sorry

end NUMINAMATH_CALUDE_sarah_candy_count_l899_89986


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l899_89914

theorem rectangle_dimension_change (L B : ℝ) (p : ℝ) 
  (h1 : L > 0) (h2 : B > 0) (h3 : p > 0) :
  (L * (1 + p)) * (B * 0.75) = L * B * 1.05 → p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l899_89914


namespace NUMINAMATH_CALUDE_children_with_cats_l899_89949

/-- Represents the number of children in each category -/
structure KindergartenPets where
  total : ℕ
  onlyDogs : ℕ
  bothPets : ℕ
  onlyCats : ℕ

/-- The conditions of the kindergarten pet situation -/
def kindergartenConditions : KindergartenPets where
  total := 30
  onlyDogs := 18
  bothPets := 6
  onlyCats := 30 - 18 - 6

theorem children_with_cats (k : KindergartenPets) 
  (h1 : k.total = 30)
  (h2 : k.onlyDogs = 18)
  (h3 : k.bothPets = 6)
  (h4 : k.total = k.onlyDogs + k.onlyCats + k.bothPets) :
  k.onlyCats + k.bothPets = 12 := by
  sorry

#eval kindergartenConditions.onlyCats + kindergartenConditions.bothPets

end NUMINAMATH_CALUDE_children_with_cats_l899_89949


namespace NUMINAMATH_CALUDE_unique_x_value_l899_89909

def A (x : ℝ) : Set ℝ := {2, 0, x}
def B (x : ℝ) : Set ℝ := {2, x^2}

theorem unique_x_value : 
  ∀ x : ℝ, (A x ∩ B x = B x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_value_l899_89909


namespace NUMINAMATH_CALUDE_complex_power_difference_l899_89993

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference : i^2 = -1 → (1 + i)^18 - (1 - i)^18 = 1024 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l899_89993


namespace NUMINAMATH_CALUDE_equality_holds_l899_89931

-- Define the property P for the function f
def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f (x + y)| ≥ |f x + f y|

-- State the theorem
theorem equality_holds (f : ℝ → ℝ) (h : satisfies_inequality f) :
  ∀ x y : ℝ, |f (x + y)| = |f x + f y| := by
  sorry

end NUMINAMATH_CALUDE_equality_holds_l899_89931


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_minimum_is_four_l899_89946

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |a*x - 5|

-- Part 1
theorem solution_set_when_a_is_one :
  let a : ℝ := 1
  {x : ℝ | f a x ≥ 9} = {x : ℝ | x ≤ -1 ∨ x > 5} := by sorry

-- Part 2
theorem a_value_when_minimum_is_four :
  ∃ (a : ℝ), 0 < a ∧ a < 5 ∧ 
  (∀ x : ℝ, f a x ≥ 4) ∧
  (∃ x : ℝ, f a x = 4) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_minimum_is_four_l899_89946


namespace NUMINAMATH_CALUDE_circle_equation_chord_length_implies_k_min_distance_l899_89984

-- Define the circle
def circle_center : ℝ × ℝ := (3, -2)
def circle_radius : ℝ := 5

-- Define points A and B
def point_A : ℝ × ℝ := (-1, 1)
def point_B : ℝ × ℝ := (-2, -2)

-- Define the line l that the circle center lies on
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the chord line
def chord_line (k x y : ℝ) : Prop := k * x - y + 5 = 0

-- Define the line for minimum distance
def line_min_dist (x y : ℝ) : Prop := x - y + 5 = 0

-- Theorem statements
theorem circle_equation : 
  ∀ x y : ℝ, (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 ↔
  (x - 3)^2 + (y + 2)^2 = 25 := by sorry

theorem chord_length_implies_k :
  ∃ k : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ,
    chord_line k x₁ y₁ ∧ chord_line k x₂ y₂ ∧
    (x₁ - circle_center.1)^2 + (y₁ - circle_center.2)^2 = circle_radius^2 ∧
    (x₂ - circle_center.1)^2 + (y₂ - circle_center.2)^2 = circle_radius^2 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 64) →
  k = -20/21 := by sorry

theorem min_distance :
  ∀ P Q : ℝ × ℝ,
  ((P.1 - circle_center.1)^2 + (P.2 - circle_center.2)^2 = circle_radius^2) →
  line_min_dist Q.1 Q.2 →
  ∃ d : ℝ, d ≥ 5 * Real.sqrt 2 - 5 ∧
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ≥ d^2 := by sorry

end NUMINAMATH_CALUDE_circle_equation_chord_length_implies_k_min_distance_l899_89984


namespace NUMINAMATH_CALUDE_sum_of_valid_numbers_l899_89933

def digits : List ℕ := [1, 3, 5, 7]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  ∃ (a b c d : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  n = 1000 * a + 100 * b + 10 * c + d

def valid_numbers : List ℕ := sorry

theorem sum_of_valid_numbers :
  (List.length valid_numbers = 24) ∧
  (List.sum valid_numbers = 106656) :=
sorry

end NUMINAMATH_CALUDE_sum_of_valid_numbers_l899_89933


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l899_89985

theorem tan_alpha_plus_pi_third (α β : Real) 
  (h1 : Real.tan (α + β + π/6) = 1/2) 
  (h2 : Real.tan (β - π/6) = -1/3) : 
  Real.tan (α + π/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l899_89985


namespace NUMINAMATH_CALUDE_perfect_square_from_sqrt_l899_89965

theorem perfect_square_from_sqrt (n : ℤ) :
  ∃ (m : ℤ), m = 2 + 2 * Real.sqrt (28 * n^2 + 1) → ∃ (k : ℤ), m = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_from_sqrt_l899_89965


namespace NUMINAMATH_CALUDE_airport_distance_airport_distance_proof_l899_89927

theorem airport_distance : ℝ → Prop :=
  fun d : ℝ =>
    let initial_speed : ℝ := 45
    let speed_increase : ℝ := 20
    let late_time : ℝ := 0.75  -- 45 minutes in hours
    let early_time : ℝ := 0.25  -- 15 minutes in hours
    let t : ℝ := (d / initial_speed) - late_time  -- Time if he continued at initial speed
    
    (d = initial_speed * (t + late_time)) ∧
    (d - initial_speed = (initial_speed + speed_increase) * (t - early_time)) →
    d = 61.875

-- The proof would go here
theorem airport_distance_proof : airport_distance 61.875 := by
  sorry

end NUMINAMATH_CALUDE_airport_distance_airport_distance_proof_l899_89927


namespace NUMINAMATH_CALUDE_bus_stop_time_l899_89959

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages. -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 50) 
  (h2 : speed_with_stops = 43) : ℝ :=
by
  -- The proof goes here
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l899_89959


namespace NUMINAMATH_CALUDE_cubic_transformation_1993_l899_89903

/-- Cubic transformation: sum of cubes of digits --/
def cubicTransform (n : ℕ) : ℕ := sorry

/-- Sequence of cubic transformations starting from n --/
def cubicSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | i + 1 => cubicTransform (cubicSequence n i)

/-- Predicate for sequence alternating between two values --/
def alternatesBetween (seq : ℕ → ℕ) (a b : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i ≥ k, seq i = a ∧ seq (i + 1) = b ∨ seq i = b ∧ seq (i + 1) = a

theorem cubic_transformation_1993 :
  alternatesBetween (cubicSequence 1993) 1459 919 := by sorry

end NUMINAMATH_CALUDE_cubic_transformation_1993_l899_89903


namespace NUMINAMATH_CALUDE_large_pizza_cost_l899_89948

/-- Represents the cost and size of a pizza --/
structure Pizza where
  side_length : ℝ
  cost : ℝ

/-- Calculates the area of a square pizza --/
def pizza_area (p : Pizza) : ℝ := p.side_length ^ 2

theorem large_pizza_cost : ∃ (large_pizza : Pizza),
  let small_pizza := Pizza.mk 12 10
  let total_budget := 60
  let separate_purchase_area := 2 * (total_budget / small_pizza.cost * pizza_area small_pizza)
  large_pizza.side_length = 18 ∧
  large_pizza.cost = 21.6 ∧
  (total_budget / large_pizza.cost * pizza_area large_pizza) = separate_purchase_area + 36 := by
  sorry

end NUMINAMATH_CALUDE_large_pizza_cost_l899_89948


namespace NUMINAMATH_CALUDE_randys_trip_l899_89975

theorem randys_trip (x : ℚ) 
  (h1 : x / 4 + 30 + x / 6 = x) : x = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_randys_trip_l899_89975


namespace NUMINAMATH_CALUDE_sand_remaining_l899_89916

/-- Calculates the remaining amount of sand in a truck after transit -/
theorem sand_remaining (initial_sand lost_sand : ℝ) :
  initial_sand ≥ 0 →
  lost_sand ≥ 0 →
  lost_sand ≤ initial_sand →
  initial_sand - lost_sand = initial_sand - lost_sand :=
by
  sorry

#check sand_remaining 4.1 2.4

end NUMINAMATH_CALUDE_sand_remaining_l899_89916


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l899_89941

theorem smallest_solution_of_equation (x : ℝ) :
  (3 * x^2 + 33 * x - 90 = x * (x + 18)) →
  x ≥ -10.5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l899_89941


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l899_89958

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 36 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l899_89958


namespace NUMINAMATH_CALUDE_ed_conch_shells_ed_conch_shells_eq_8_l899_89963

theorem ed_conch_shells (initial_shells : ℕ) (ed_limpet : ℕ) (ed_oyster : ℕ) (jacob_extra : ℕ) (total_shells : ℕ) : ℕ :=
  let ed_known := ed_limpet + ed_oyster
  let jacob_shells := ed_known + jacob_extra
  let known_shells := initial_shells + ed_known + jacob_shells
  total_shells - known_shells

theorem ed_conch_shells_eq_8 : 
  ed_conch_shells 2 7 2 2 30 = 8 := by sorry

end NUMINAMATH_CALUDE_ed_conch_shells_ed_conch_shells_eq_8_l899_89963


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l899_89917

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l899_89917


namespace NUMINAMATH_CALUDE_bracelet_arrangement_l899_89996

/-- The number of unique arrangements of beads on a bracelet -/
def uniqueArrangements (n : ℕ) : ℕ := sorry

/-- Two specific beads are always adjacent -/
def adjacentBeads : Prop := sorry

/-- Rotations and reflections of the same arrangement are considered identical -/
def symmetryEquivalence : Prop := sorry

theorem bracelet_arrangement :
  uniqueArrangements 8 = 720 ∧ adjacentBeads ∧ symmetryEquivalence :=
sorry

end NUMINAMATH_CALUDE_bracelet_arrangement_l899_89996


namespace NUMINAMATH_CALUDE_boat_capacity_l899_89919

theorem boat_capacity (trips_per_day : ℕ) (total_people : ℕ) (total_days : ℕ) :
  trips_per_day = 4 →
  total_people = 96 →
  total_days = 2 →
  total_people / (trips_per_day * total_days) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_capacity_l899_89919


namespace NUMINAMATH_CALUDE_sequence_bound_l899_89905

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i, 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j, i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bound_l899_89905


namespace NUMINAMATH_CALUDE_sum_inequality_l899_89902

theorem sum_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) : a + b ≤ 3 * c := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l899_89902


namespace NUMINAMATH_CALUDE_parabola_area_l899_89930

-- Define the two parabolas
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 8 - x^2

-- Define the region
def R : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem parabola_area : 
  (∫ (x : ℝ) in R, g x - f x) = 64/3 := by sorry

end NUMINAMATH_CALUDE_parabola_area_l899_89930


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l899_89969

theorem consecutive_pages_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20736 → n + (n + 1) = 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l899_89969


namespace NUMINAMATH_CALUDE_no_integer_satisfies_condition_l899_89915

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_integer_satisfies_condition : 
  ¬ ∃ n : ℕ+, (n : ℕ) % sum_of_digits n = 0 → sum_of_digits (n * sum_of_digits n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_condition_l899_89915


namespace NUMINAMATH_CALUDE_hayden_evening_snack_l899_89979

/-- Calculates the amount of nuts in one serving given the bag cost, weight, coupon value, and cost per serving after coupon. -/
def nuts_per_serving (bag_cost : ℚ) (bag_weight : ℚ) (coupon : ℚ) (serving_cost : ℚ) : ℚ :=
  let cost_after_coupon := bag_cost - coupon
  let num_servings := cost_after_coupon / serving_cost
  bag_weight / num_servings

/-- Theorem stating that under the given conditions, the amount of nuts in one serving is 1 oz. -/
theorem hayden_evening_snack :
  nuts_per_serving 25 40 5 (1/2) = 1 := by sorry

end NUMINAMATH_CALUDE_hayden_evening_snack_l899_89979


namespace NUMINAMATH_CALUDE_syrup_volume_proof_l899_89938

/-- Calculates the final volume of syrup in cups -/
def final_syrup_volume (original_volume : ℚ) (reduction_factor : ℚ) (added_sugar : ℚ) (cups_per_quart : ℚ) : ℚ :=
  original_volume * cups_per_quart * reduction_factor + added_sugar

theorem syrup_volume_proof (original_volume : ℚ) (reduction_factor : ℚ) (added_sugar : ℚ) (cups_per_quart : ℚ)
  (h1 : original_volume = 6)
  (h2 : reduction_factor = 1 / 12)
  (h3 : added_sugar = 1)
  (h4 : cups_per_quart = 4) :
  final_syrup_volume original_volume reduction_factor added_sugar cups_per_quart = 3 := by
  sorry

#eval final_syrup_volume 6 (1/12) 1 4

end NUMINAMATH_CALUDE_syrup_volume_proof_l899_89938


namespace NUMINAMATH_CALUDE_total_widgets_sold_is_360_l899_89960

/-- The sum of an arithmetic sequence with first term 3, common difference 3, and 15 terms -/
def widget_sales_sum : ℕ :=
  let first_term := 3
  let common_difference := 3
  let num_days := 15
  (num_days * (2 * first_term + (num_days - 1) * common_difference)) / 2

/-- Theorem stating that the total number of widgets sold is 360 -/
theorem total_widgets_sold_is_360 : widget_sales_sum = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_widgets_sold_is_360_l899_89960


namespace NUMINAMATH_CALUDE_square_area_on_circle_and_tangent_l899_89910

/-- Given a circle with radius 5 and a square with two vertices on the circle
    and two vertices on a tangent to the circle, the area of the square is 64. -/
theorem square_area_on_circle_and_tangent :
  ∀ (circle : ℝ → ℝ → Prop) (square : ℝ → ℝ → Prop) (r : ℝ),
  (r = 5) →  -- The radius of the circle is 5
  (∃ (A B C D : ℝ × ℝ),
    -- Two vertices of the square lie on the circle
    circle A.1 A.2 ∧ circle C.1 C.2 ∧
    -- The other two vertices lie on a tangent to the circle
    (∃ (t : ℝ → ℝ → Prop), t B.1 B.2 ∧ t D.1 D.2) ∧
    -- A, B, C, D form a square
    square A.1 A.2 ∧ square B.1 B.2 ∧ square C.1 C.2 ∧ square D.1 D.2) →
  (∃ (area : ℝ), area = 64) :=
by sorry

end NUMINAMATH_CALUDE_square_area_on_circle_and_tangent_l899_89910


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l899_89955

/-- The function f(x) = x^3 - x + a --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x + a

/-- The condition a^2 - a = 0 --/
def condition (a : ℝ) : Prop := a^2 - a = 0

/-- f is an increasing function --/
def is_increasing (a : ℝ) : Prop := ∀ x y, x < y → f a x < f a y

theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ a, condition a → is_increasing a) ∧
  ¬(∀ a, is_increasing a → condition a) :=
sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l899_89955


namespace NUMINAMATH_CALUDE_abs_three_implies_plus_minus_three_l899_89978

theorem abs_three_implies_plus_minus_three (a : ℝ) : 
  |a| = 3 → (a = 3 ∨ a = -3) := by sorry

end NUMINAMATH_CALUDE_abs_three_implies_plus_minus_three_l899_89978


namespace NUMINAMATH_CALUDE_distance_between_points_l899_89922

theorem distance_between_points :
  let A : ℝ × ℝ := (8, -5)
  let B : ℝ × ℝ := (0, 10)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 17 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l899_89922


namespace NUMINAMATH_CALUDE_distance_between_stations_l899_89962

/-- The distance between two stations given the conditions of two trains meeting --/
theorem distance_between_stations
  (speed_train1 : ℝ)
  (speed_train2 : ℝ)
  (extra_distance : ℝ)
  (h1 : speed_train1 = 20)
  (h2 : speed_train2 = 25)
  (h3 : extra_distance = 70)
  (h4 : speed_train1 > 0)
  (h5 : speed_train2 > 0) :
  ∃ (time : ℝ),
    time > 0 ∧
    speed_train1 * time + speed_train2 * time = speed_train1 * time + extra_distance ∧
    speed_train1 * time + speed_train2 * time = 630 :=
by sorry


end NUMINAMATH_CALUDE_distance_between_stations_l899_89962


namespace NUMINAMATH_CALUDE_females_dont_listen_l899_89944

/-- A structure representing the survey results -/
structure SurveyResults where
  total_listen : Nat
  males_listen : Nat
  total_dont_listen : Nat
  total_respondents : Nat
  males_listen_le_total_listen : males_listen ≤ total_listen
  total_respondents_eq : total_respondents = total_listen + total_dont_listen

/-- The theorem stating the number of females who don't listen to the radio station -/
theorem females_dont_listen (survey : SurveyResults)
  (h_total_listen : survey.total_listen = 200)
  (h_males_listen : survey.males_listen = 75)
  (h_total_dont_listen : survey.total_dont_listen = 180)
  (h_total_respondents : survey.total_respondents = 380) :
  survey.total_dont_listen - (survey.total_respondents - survey.total_listen) = 180 := by
  sorry


end NUMINAMATH_CALUDE_females_dont_listen_l899_89944


namespace NUMINAMATH_CALUDE_abs_negative_two_thirds_equals_two_thirds_l899_89926

theorem abs_negative_two_thirds_equals_two_thirds : 
  |(-2 : ℚ) / 3| = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_abs_negative_two_thirds_equals_two_thirds_l899_89926
