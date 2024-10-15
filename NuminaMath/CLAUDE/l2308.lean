import Mathlib

namespace NUMINAMATH_CALUDE_triangle_perimeter_l2308_230814

theorem triangle_perimeter (x : ℕ+) : 
  (1 < x) ∧ (x < 5) ∧ (1 + x > 4) ∧ (x + 4 > 1) ∧ (4 + 1 > x) → 
  1 + x + 4 = 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2308_230814


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l2308_230819

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 2*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l2308_230819


namespace NUMINAMATH_CALUDE_crouton_calories_l2308_230815

def salad_calories : ℕ := 350
def lettuce_calories : ℕ := 30
def cucumber_calories : ℕ := 80
def num_croutons : ℕ := 12

theorem crouton_calories : 
  (salad_calories - lettuce_calories - cucumber_calories) / num_croutons = 20 := by
  sorry

end NUMINAMATH_CALUDE_crouton_calories_l2308_230815


namespace NUMINAMATH_CALUDE_complex_modulus_sum_difference_l2308_230811

theorem complex_modulus_sum_difference :
  let z₁ : ℂ := 3 - 5*I
  let z₂ : ℂ := 3 + 5*I
  let z₃ : ℂ := -2 + 6*I
  Complex.abs z₁ + Complex.abs z₂ - Real.sqrt (Complex.abs z₃) = 2 * Real.sqrt 34 - Real.sqrt (2 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_sum_difference_l2308_230811


namespace NUMINAMATH_CALUDE_cuboid_height_calculation_l2308_230809

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: A cuboid with length 4 cm, breadth 6 cm, and surface area 120 cm² has a height of 3.6 cm -/
theorem cuboid_height_calculation (h : ℝ) :
  cuboidSurfaceArea 4 6 h = 120 → h = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_calculation_l2308_230809


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l2308_230889

def A : Set ℝ := {x | x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem intersection_implies_a_equals_one (a : ℝ) :
  A ∩ B a = {1} → a = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l2308_230889


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2308_230890

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2308_230890


namespace NUMINAMATH_CALUDE_abs_inequality_l2308_230854

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l2308_230854


namespace NUMINAMATH_CALUDE_f_2_eq_125_l2308_230892

/-- Horner's method for evaluating polynomials -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 + 3x^4 + 2x^3 - 4x + 5 -/
def f (x : ℝ) : ℝ :=
  horner [5, -4, 0, 2, 3, 2] x

theorem f_2_eq_125 : f 2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_f_2_eq_125_l2308_230892


namespace NUMINAMATH_CALUDE_ab_plus_one_neq_a_plus_b_l2308_230880

theorem ab_plus_one_neq_a_plus_b (a b : ℝ) : ab + 1 ≠ a + b ↔ a ≠ 1 ∧ b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_one_neq_a_plus_b_l2308_230880


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2308_230883

/-- Proves that in a triangle ABC, if angle C is triple angle B and angle B is 18°, then angle A is 108° -/
theorem triangle_angle_calculation (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- Angles are positive
  A + B + C = 180 → -- Sum of angles in a triangle
  C = 3 * B → -- Angle C is triple angle B
  B = 18 → -- Angle B is 18°
  A = 108 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2308_230883


namespace NUMINAMATH_CALUDE_quadrilateral_angle_combinations_l2308_230812

/-- Represents the type of an angle in a quadrilateral -/
inductive AngleType
| Acute
| Right
| Obtuse

/-- Represents a combination of angles in a quadrilateral -/
structure AngleCombination :=
  (acute : Nat)
  (right : Nat)
  (obtuse : Nat)

/-- A convex quadrilateral has exactly four angles -/
def total_angles : Nat := 4

/-- The sum of interior angles in a quadrilateral is 360 degrees -/
def angle_sum : Nat := 360

/-- Theorem: The only possible combinations of internal angles in a convex quadrilateral
    are the seven combinations listed. -/
theorem quadrilateral_angle_combinations :
  ∃ (valid_combinations : List AngleCombination),
    (valid_combinations.length = 7) ∧
    (∀ combo : AngleCombination,
      (combo.acute + combo.right + combo.obtuse = total_angles) →
      (combo.right * 90 + combo.acute * 89 + combo.obtuse * 91 ≤ angle_sum) →
      (combo.right * 90 + combo.acute * 1 + combo.obtuse * 91 ≥ angle_sum) →
      (combo ∈ valid_combinations)) ∧
    (∀ combo : AngleCombination,
      combo ∈ valid_combinations →
      (combo.acute + combo.right + combo.obtuse = total_angles) ∧
      (combo.right * 90 + combo.acute * 89 + combo.obtuse * 91 ≤ angle_sum) ∧
      (combo.right * 90 + combo.acute * 1 + combo.obtuse * 91 ≥ angle_sum)) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_combinations_l2308_230812


namespace NUMINAMATH_CALUDE_paint_remaining_is_three_eighths_l2308_230831

/-- The fraction of paint remaining after three days of usage --/
def paint_remaining (initial_amount : ℚ) : ℚ :=
  let day1_remaining := initial_amount / 2
  let day2_remaining := day1_remaining * 3/4
  let day3_remaining := day2_remaining / 2
  day3_remaining / initial_amount

/-- Theorem stating that the fraction of paint remaining after three days is 3/8 --/
theorem paint_remaining_is_three_eighths (initial_amount : ℚ) :
  paint_remaining initial_amount = 3/8 := by
  sorry

#eval paint_remaining 2  -- To check the result

end NUMINAMATH_CALUDE_paint_remaining_is_three_eighths_l2308_230831


namespace NUMINAMATH_CALUDE_negation_of_cube_odd_l2308_230869

theorem negation_of_cube_odd (P : ℕ → Prop) :
  (¬ ∀ x : ℕ, Odd x → Odd (x^3)) ↔ (∃ x : ℕ, Odd x ∧ Even (x^3)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_cube_odd_l2308_230869


namespace NUMINAMATH_CALUDE_first_grade_allocation_l2308_230885

theorem first_grade_allocation (total : ℕ) (ratio_first : ℕ) (ratio_second : ℕ) (ratio_third : ℕ) 
  (h_total : total = 160)
  (h_ratio : ratio_first = 6 ∧ ratio_second = 5 ∧ ratio_third = 5) :
  (total * ratio_first) / (ratio_first + ratio_second + ratio_third) = 60 := by
  sorry

end NUMINAMATH_CALUDE_first_grade_allocation_l2308_230885


namespace NUMINAMATH_CALUDE_ruble_payment_l2308_230884

theorem ruble_payment (n : ℤ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end NUMINAMATH_CALUDE_ruble_payment_l2308_230884


namespace NUMINAMATH_CALUDE_geometric_progression_special_ratio_l2308_230842

/-- A geometric progression where each term is positive and any term is equal to the sum of the next three following terms has a common ratio that satisfies r³ + r² + r - 1 = 0. -/
theorem geometric_progression_special_ratio :
  ∀ (a : ℝ) (r : ℝ),
  (a > 0) →  -- First term is positive
  (r > 0) →  -- Common ratio is positive
  (∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) →  -- Any term equals sum of next three
  r^3 + r^2 + r - 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_special_ratio_l2308_230842


namespace NUMINAMATH_CALUDE_spider_movement_limit_l2308_230896

/-- Represents the spider's position and movement on the wall --/
structure SpiderPosition :=
  (height : ℝ)  -- Current height of the spider
  (day : ℕ)     -- Current day

/-- Defines the daily movement of the spider --/
def daily_movement (sp : SpiderPosition) : SpiderPosition :=
  ⟨sp.height + 2, sp.day + 1⟩

/-- Checks if the spider can be moved up 3 feet --/
def can_move_up (sp : SpiderPosition) (wall_height : ℝ) : Prop :=
  sp.height + 3 ≤ wall_height

/-- Theorem: Tony runs out of room after 8 days --/
theorem spider_movement_limit :
  ∀ (wall_height : ℝ) (initial_height : ℝ),
  wall_height = 18 → initial_height = 3 →
  ∃ (n : ℕ), n = 8 ∧
  ¬(can_move_up (n.iterate daily_movement ⟨initial_height, 0⟩) wall_height) ∧
  ∀ (m : ℕ), m < n →
  can_move_up (m.iterate daily_movement ⟨initial_height, 0⟩) wall_height :=
by sorry

end NUMINAMATH_CALUDE_spider_movement_limit_l2308_230896


namespace NUMINAMATH_CALUDE_expansion_simplification_l2308_230813

theorem expansion_simplification (a b : ℝ) : (a + b) * (3 * a - b) - b * (a - b) = 3 * a^2 + a * b := by
  sorry

end NUMINAMATH_CALUDE_expansion_simplification_l2308_230813


namespace NUMINAMATH_CALUDE_min_value_sum_of_inverses_l2308_230857

theorem min_value_sum_of_inverses (x y z p q r : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0)
  (sum_eq_10 : x + y + z + p + q + r = 10) :
  1/x + 9/y + 4/z + 25/p + 16/q + 36/r ≥ 441/10 ∧ 
  ∃ (x' y' z' p' q' r' : ℝ), 
    x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ p' > 0 ∧ q' > 0 ∧ r' > 0 ∧
    x' + y' + z' + p' + q' + r' = 10 ∧
    1/x' + 9/y' + 4/z' + 25/p' + 16/q' + 36/r' = 441/10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_inverses_l2308_230857


namespace NUMINAMATH_CALUDE_line_translation_proof_l2308_230804

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translateLine (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - dy }

/-- The original line y = 4x -/
def originalLine : Line :=
  { slope := 4, intercept := 0 }

/-- The amount of downward translation -/
def translationAmount : ℝ := 5

theorem line_translation_proof :
  translateLine originalLine translationAmount = { slope := 4, intercept := -5 } := by
  sorry

end NUMINAMATH_CALUDE_line_translation_proof_l2308_230804


namespace NUMINAMATH_CALUDE_sum_of_abc_l2308_230886

theorem sum_of_abc (a b c : ℕ+) 
  (h1 : a.val * b.val + c.val = 47)
  (h2 : b.val * c.val + a.val = 47)
  (h3 : a.val * c.val + b.val = 47) :
  a.val + b.val + c.val = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l2308_230886


namespace NUMINAMATH_CALUDE_speed_of_car_C_l2308_230861

/-- Proves that given the conditions of the problem, the speed of car C is 26 km/h --/
theorem speed_of_car_C (v_A v_B : ℝ) (t_A t_B t_C : ℝ) :
  v_A = 24 →
  v_B = 20 →
  t_A = 5 / 60 →
  t_B = 10 / 60 →
  t_C = 12 / 60 →
  v_A * t_A = v_B * t_B →
  ∃ (v_C : ℝ), v_C * t_C = v_A * t_A ∧ v_C = 26 :=
by sorry

#check speed_of_car_C

end NUMINAMATH_CALUDE_speed_of_car_C_l2308_230861


namespace NUMINAMATH_CALUDE_sqrt_two_equality_l2308_230853

theorem sqrt_two_equality : (2 : ℝ) / Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_equality_l2308_230853


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2308_230847

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 1 → 1 / a + 1 / b ≤ 1 / x + 1 / y) ∧
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2308_230847


namespace NUMINAMATH_CALUDE_count_equals_60_l2308_230829

/-- A function that generates all 5-digit numbers composed of digits 1, 2, 3, 4, and 5 without repetition -/
def generate_numbers : List Nat := sorry

/-- A function that checks if a number is greater than 23145 and less than 43521 -/
def is_in_range (n : Nat) : Bool := 23145 < n && n < 43521

/-- The count of numbers in the specified range -/
def count_in_range : Nat :=
  (generate_numbers.filter is_in_range).length

theorem count_equals_60 : count_in_range = 60 := by sorry

end NUMINAMATH_CALUDE_count_equals_60_l2308_230829


namespace NUMINAMATH_CALUDE_c_investment_is_2000_l2308_230852

/-- Represents a partnership investment and profit distribution --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Theorem stating that under given conditions, C's investment is 2000 --/
theorem c_investment_is_2000 (p : Partnership) 
  (h1 : p.a_investment = 8000)
  (h2 : p.b_investment = 4000)
  (h3 : p.total_profit = 252000)
  (h4 : p.c_profit = 36000)
  (h5 : p.c_profit * (p.a_investment + p.b_investment + p.c_investment) = 
        p.c_investment * p.total_profit) : 
  p.c_investment = 2000 := by
  sorry

#check c_investment_is_2000

end NUMINAMATH_CALUDE_c_investment_is_2000_l2308_230852


namespace NUMINAMATH_CALUDE_berry_ratio_l2308_230897

/-- Given the distribution of berries among Stacy, Steve, and Sylar, 
    prove that the ratio of Stacy's berries to Steve's berries is 4:1 -/
theorem berry_ratio (total berries_stacy berries_steve berries_sylar : ℕ) :
  total = 1100 →
  berries_stacy = 800 →
  berries_steve = 2 * berries_sylar →
  total = berries_stacy + berries_steve + berries_sylar →
  berries_stacy / berries_steve = 4 := by
  sorry

#check berry_ratio

end NUMINAMATH_CALUDE_berry_ratio_l2308_230897


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2308_230891

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 3 = 3 →  -- given condition
  a 2 ^ 2 = a 1 * a 4 →  -- geometric sequence condition
  a 5 = 5 ∨ a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2308_230891


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2308_230865

/-- Given a moving straight line ax + by + c - 2 = 0 where a > 0, c > 0,
    that always passes through point (1, m), and the maximum distance
    from (4, 0) to the line is 3, the minimum value of 1/(2a) + 2/c is 9/4. -/
theorem min_value_of_expression (a b c m : ℝ) : 
  a > 0 → c > 0 → 
  (∀ x y, a * x + b * y + c - 2 = 0 → x = 1 → y = m) →
  (∃ x y, a * x + b * y + c - 2 = 0 ∧ 
    Real.sqrt ((x - 4)^2 + y^2) = 3) →
  (∀ x y, a * x + b * y + c - 2 = 0 → 
    Real.sqrt ((x - 4)^2 + y^2) ≤ 3) →
  (1 / (2 * a) + 2 / c) ≥ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2308_230865


namespace NUMINAMATH_CALUDE_no_solution_to_double_inequality_l2308_230810

theorem no_solution_to_double_inequality :
  ¬ ∃ x : ℝ, (4 * x - 3 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_double_inequality_l2308_230810


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2308_230846

theorem geometric_series_sum (x : ℝ) :
  (|x| < 1) →
  (∑' n, x^n = 4) →
  x = 3/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2308_230846


namespace NUMINAMATH_CALUDE_james_marbles_left_james_final_marbles_l2308_230803

/-- Represents the number of marbles in each bag -/
structure Bags where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  g : Nat

/-- Calculates the total number of marbles in all bags -/
def totalMarbles (bags : Bags) : Nat :=
  bags.a + bags.b + bags.c + bags.d + bags.e + bags.f + bags.g

/-- Represents James' marble collection -/
structure MarbleCollection where
  initialTotal : Nat
  bags : Bags
  forgottenBag : Nat

/-- Theorem stating that James will have 20 marbles left -/
theorem james_marbles_left (collection : MarbleCollection) : Nat :=
  if collection.initialTotal = 28 ∧
     collection.bags.a = 4 ∧
     collection.bags.b = 3 ∧
     collection.bags.c = 5 ∧
     collection.bags.d = 2 * collection.bags.c - 1 ∧
     collection.bags.e = collection.bags.a / 2 ∧
     collection.bags.f = 3 ∧
     collection.bags.g = collection.bags.e ∧
     collection.forgottenBag = 4 ∧
     totalMarbles collection.bags = collection.initialTotal
  then
    collection.initialTotal - (collection.bags.d + collection.bags.f) + collection.forgottenBag
  else
    0

/-- Main theorem to prove -/
theorem james_final_marbles (collection : MarbleCollection) :
  james_marbles_left collection = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_marbles_left_james_final_marbles_l2308_230803


namespace NUMINAMATH_CALUDE_max_value_expression_l2308_230856

theorem max_value_expression (n : ℕ) (h : n = 15000) :
  let factorization := 2^3 * 3 * 5^4
  ∃ (x y : ℕ), 
    (2*x - y = 0 ∨ 3*x - y = 0) ∧ 
    (x ∣ n) ∧
    ∀ (x' y' : ℕ), (2*x' - y' = 0 ∨ 3*x' - y' = 0) ∧ (x' ∣ n) → 
      2*x + 3*y ≥ 2*x' + 3*y' ∧
      2*x + 3*y = 60000 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2308_230856


namespace NUMINAMATH_CALUDE_lcm_of_3_5_7_18_l2308_230826

theorem lcm_of_3_5_7_18 : Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 18)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_3_5_7_18_l2308_230826


namespace NUMINAMATH_CALUDE_bus_driver_hours_l2308_230807

/-- Represents the compensation structure and work hours of a bus driver --/
structure BusDriver where
  regularRate : ℝ
  regularHours : ℝ
  overtimeRate : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total hours worked by a bus driver --/
def totalHours (driver : BusDriver) : ℝ :=
  driver.regularHours + driver.overtimeHours

/-- Theorem stating the conditions and the result to be proved --/
theorem bus_driver_hours (driver : BusDriver) 
  (h1 : driver.regularRate = 16)
  (h2 : driver.regularHours = 40)
  (h3 : driver.overtimeRate = driver.regularRate * 1.75)
  (h4 : driver.totalCompensation = 976)
  (h5 : driver.totalCompensation = driver.regularRate * driver.regularHours + 
                                   driver.overtimeRate * driver.overtimeHours) :
  totalHours driver = 52 := by
  sorry

#eval 40 + 12 -- Expected output: 52

end NUMINAMATH_CALUDE_bus_driver_hours_l2308_230807


namespace NUMINAMATH_CALUDE_machine_output_2023_l2308_230877

/-- A function that computes the output of Ava's machine for a four-digit number -/
def machine_output (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a * b + c * d

/-- Theorem stating that the machine output for 2023 is 6 -/
theorem machine_output_2023 : machine_output 2023 = 6 := by
  sorry

end NUMINAMATH_CALUDE_machine_output_2023_l2308_230877


namespace NUMINAMATH_CALUDE_steel_bar_length_l2308_230838

/-- Given three types of steel bars A, B, and C with lengths x, y, and z respectively,
    prove that the total length of 1 bar of type A, 2 bars of type B, and 3 bars of type C
    is x + 2y + 3z, given the conditions. -/
theorem steel_bar_length (x y z : ℝ) 
  (h1 : 2 * x + y + 3 * z = 23) 
  (h2 : x + 4 * y + 5 * z = 36) : 
  x + 2 * y + 3 * z = (7 * x + 14 * y + 21 * z) / 7 := by
  sorry

end NUMINAMATH_CALUDE_steel_bar_length_l2308_230838


namespace NUMINAMATH_CALUDE_quadratic_integer_solutions_l2308_230839

theorem quadratic_integer_solutions (p q : ℝ) : 
  p + q = 1998 ∧ 
  (∃ a b : ℤ, ∀ x : ℝ, x^2 + p*x + q = 0 ↔ x = a ∨ x = b) →
  (p = 1998 ∧ q = 0) ∨ (p = -2002 ∧ q = 4000) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_solutions_l2308_230839


namespace NUMINAMATH_CALUDE_factorization_proof_l2308_230801

theorem factorization_proof (b : ℝ) : 2 * b^2 - 8 * b + 8 = 2 * (b - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2308_230801


namespace NUMINAMATH_CALUDE_cos_x_value_l2308_230845

theorem cos_x_value (x : Real) (h1 : Real.tan (x + Real.pi / 4) = 2) 
  (h2 : x ∈ Set.Icc (Real.pi) (3 * Real.pi / 2)) : 
  Real.cos x = - (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_value_l2308_230845


namespace NUMINAMATH_CALUDE_dans_apples_l2308_230873

theorem dans_apples (benny_apples total_apples : ℕ) 
  (h1 : benny_apples = 2)
  (h2 : total_apples = 11) :
  total_apples - benny_apples = 9 :=
by sorry

end NUMINAMATH_CALUDE_dans_apples_l2308_230873


namespace NUMINAMATH_CALUDE_sum_of_sequences_l2308_230874

theorem sum_of_sequences : (2+12+22+32+42) + (10+20+30+40+50) = 260 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l2308_230874


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l2308_230833

theorem no_solutions_for_equation : ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (2 / a + 2 / b = 1 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l2308_230833


namespace NUMINAMATH_CALUDE_symmetry_properties_l2308_230893

noncomputable def f (a b c p q x : ℝ) : ℝ := (a * 4^x + b * 2^x + c) / (p * 2^x + q)

theorem symmetry_properties (a b c p q : ℝ) :
  (p = 0 ∧ q ≠ 0 ∧ a^2 + b^2 ≠ 0 →
    ¬(∃t, ∀x, f a b c p q (t + x) = f a b c p q (t - x)) ∧
    ¬(∃t, ∀x, f a b c p q (t + x) + f a b c p q (t - x) = 2 * f a b c p q t)) ∧
  (p ≠ 0 ∧ q = 0 ∧ a * c ≠ 0 →
    (∃t, ∀x, f a b c p q (t + x) = f a b c p q (t - x)) ∨
    (∃t, ∀x, f a b c p q (t + x) + f a b c p q (t - x) = 2 * f a b c p q t)) ∧
  (p * q ≠ 0 ∧ a = 0 ∧ b^2 + c^2 ≠ 0 →
    ∃t, ∀x, f a b c p q (t + x) + f a b c p q (t - x) = 2 * f a b c p q t) ∧
  (p * q ≠ 0 ∧ a ≠ 0 →
    ¬(∃t, ∀x, f a b c p q (t + x) = f a b c p q (t - x))) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_properties_l2308_230893


namespace NUMINAMATH_CALUDE_subtraction_in_third_quadrant_l2308_230802

/-- Given complex numbers z₁ and z₂, prove that z₁ - z₂ is in the third quadrant -/
theorem subtraction_in_third_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = -2 + I) 
  (h₂ : z₂ = 1 + 2*I) : 
  let z := z₁ - z₂
  (z.re < 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_in_third_quadrant_l2308_230802


namespace NUMINAMATH_CALUDE_paige_folders_l2308_230875

theorem paige_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 27 →
  deleted_files = 9 →
  files_per_folder = 6 →
  (initial_files - deleted_files) / files_per_folder = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_paige_folders_l2308_230875


namespace NUMINAMATH_CALUDE_trapezoid_parallel_sides_l2308_230840

/-- Trapezoid properties and parallel sides calculation -/
theorem trapezoid_parallel_sides 
  (t : ℝ) 
  (m : ℝ) 
  (n : ℝ) 
  (E : ℝ) 
  (h_t : t = 204) 
  (h_m : m = 14) 
  (h_n : n = 2) 
  (h_E : E = 59 + 29/60 + 23/3600) : 
  ∃ (a c : ℝ), 
    a - c = m ∧ 
    t = (a + c) / 2 * (2 * t / (a + c)) ∧ 
    a = 24 ∧ 
    c = 10 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_parallel_sides_l2308_230840


namespace NUMINAMATH_CALUDE_bug_on_square_probability_l2308_230834

/-- Represents the probability of the bug being at its starting vertex after n moves -/
def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 1 - Q n

/-- The bug's movement on a square -/
theorem bug_on_square_probability : Q 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_bug_on_square_probability_l2308_230834


namespace NUMINAMATH_CALUDE_package_weight_sum_l2308_230871

theorem package_weight_sum (x y z : ℝ) 
  (h1 : x + y = 112)
  (h2 : y + z = 118)
  (h3 : z + x = 120) :
  x + y + z = 175 := by
sorry

end NUMINAMATH_CALUDE_package_weight_sum_l2308_230871


namespace NUMINAMATH_CALUDE_paper_boutique_sales_l2308_230824

theorem paper_boutique_sales (notebook_sales : ℝ) (marker_sales : ℝ) (stapler_sales : ℝ)
  (h1 : notebook_sales = 25)
  (h2 : marker_sales = 40)
  (h3 : stapler_sales = 15)
  (h4 : notebook_sales + marker_sales + stapler_sales + (100 - notebook_sales - marker_sales - stapler_sales) = 100) :
  100 - notebook_sales - marker_sales = 35 := by
sorry

end NUMINAMATH_CALUDE_paper_boutique_sales_l2308_230824


namespace NUMINAMATH_CALUDE_line_equation_l2308_230881

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- A line passes through the origin if its y-intercept is 0 -/
def passes_through_origin (l : Line) : Prop :=
  l.y_intercept = 0

/-- A line has equal x and y intercepts if y_intercept = -slope * y_intercept -/
def equal_intercepts (l : Line) : Prop :=
  l.y_intercept = -l.slope * l.y_intercept

/-- The main theorem -/
theorem line_equation (l m : Line) :
  passes_through_origin l →
  parallel l m →
  equal_intercepts m →
  l.slope = -1 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2308_230881


namespace NUMINAMATH_CALUDE_circle_radius_exists_l2308_230800

theorem circle_radius_exists : ∃ r : ℝ, r > 0 ∧ π * r^2 + 2 * r - 2 * π * r = 12 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_exists_l2308_230800


namespace NUMINAMATH_CALUDE_x_14_plus_inverse_l2308_230805

theorem x_14_plus_inverse (x : ℂ) (h : x^2 + x + 1 = 0) : x^14 + 1/x^14 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_14_plus_inverse_l2308_230805


namespace NUMINAMATH_CALUDE_max_planes_from_points_l2308_230851

/-- The number of points in space -/
def num_points : ℕ := 15

/-- A function that calculates the number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The theorem stating the maximum number of planes determined by the points -/
theorem max_planes_from_points :
  choose num_points 3 = 455 := by sorry

end NUMINAMATH_CALUDE_max_planes_from_points_l2308_230851


namespace NUMINAMATH_CALUDE_problem_solution_l2308_230830

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m ≥ 0

def proposition_q (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (m - 4) + y^2 / (6 - m) = 1 ∧ 
  ((m - 4) * (6 - m) < 0)

theorem problem_solution (m : ℝ) :
  (¬ proposition_p m ↔ m < 1) ∧
  (¬(proposition_p m ∧ proposition_q m) ∧ (proposition_p m ∨ proposition_q m) ↔ 
    m < 1 ∨ (4 ≤ m ∧ m ≤ 6)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2308_230830


namespace NUMINAMATH_CALUDE_train_passing_jogger_l2308_230844

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger (jogger_speed train_speed : ℝ) (initial_distance train_length : ℝ) : 
  jogger_speed = 9 →
  train_speed = 45 →
  initial_distance = 360 →
  train_length = 180 →
  (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 54 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_jogger_l2308_230844


namespace NUMINAMATH_CALUDE_alpha_beta_difference_bounds_l2308_230867

theorem alpha_beta_difference_bounds (α β : ℝ) (h : -1 < α ∧ α < β ∧ β < 1) :
  -2 < α - β ∧ α - β < 0 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_difference_bounds_l2308_230867


namespace NUMINAMATH_CALUDE_odd_factors_of_x_squared_plus_one_l2308_230849

theorem odd_factors_of_x_squared_plus_one (x : ℤ) (d : ℤ) :
  d > 0 → Odd d → (x^2 + 1) % d = 0 → ∃ h : ℤ, d = 4*h + 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_factors_of_x_squared_plus_one_l2308_230849


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2308_230866

theorem isosceles_triangle_base_length 
  (equilateral_perimeter : ℝ) 
  (isosceles_perimeter : ℝ) 
  (h_equilateral : equilateral_perimeter = 60) 
  (h_isosceles : isosceles_perimeter = 50) 
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) : 
  isosceles_base = 10 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2308_230866


namespace NUMINAMATH_CALUDE_large_lemonhead_doll_cost_l2308_230817

/-- The cost of a large lemonhead doll satisfies the given conditions -/
theorem large_lemonhead_doll_cost :
  ∃ (L : ℝ), 
    (L > 0) ∧ 
    (350 / (L - 2) = 350 / L + 20) ∧ 
    (L = 7) := by
  sorry

end NUMINAMATH_CALUDE_large_lemonhead_doll_cost_l2308_230817


namespace NUMINAMATH_CALUDE_adam_bought_seven_boxes_l2308_230835

/-- The number of boxes Adam gave away -/
def boxes_given_away : ℕ := 7

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 6

/-- The number of pieces Adam still has -/
def remaining_pieces : ℕ := 36

/-- The number of boxes Adam bought initially -/
def initial_boxes : ℕ := 7

theorem adam_bought_seven_boxes :
  initial_boxes * pieces_per_box = boxes_given_away * pieces_per_box + remaining_pieces :=
by sorry

end NUMINAMATH_CALUDE_adam_bought_seven_boxes_l2308_230835


namespace NUMINAMATH_CALUDE_extremum_implies_a_eq_neg_two_l2308_230855

/-- The function f(x) = a ln x + x^2 has an extremum at x = 1 -/
def has_extremum_at_one (a : ℝ) : Prop :=
  let f := fun (x : ℝ) => a * Real.log x + x^2
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f x ≤ f 1 ∨ f x ≥ f 1

/-- If f(x) = a ln x + x^2 has an extremum at x = 1, then a = -2 -/
theorem extremum_implies_a_eq_neg_two (a : ℝ) :
  has_extremum_at_one a → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_extremum_implies_a_eq_neg_two_l2308_230855


namespace NUMINAMATH_CALUDE_min_value_M_l2308_230888

theorem min_value_M (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let M := (((a / (b + c)) ^ (1/4)) + ((b / (c + a)) ^ (1/4)) + ((c / (a + b)) ^ (1/4)) +
            ((b + c) / a) ^ (1/2) + ((a + c) / b) ^ (1/2) + ((a + b) / c) ^ (1/2))
  M ≥ 3 * Real.sqrt 2 + (3 * (8 ^ (1/4))) / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_M_l2308_230888


namespace NUMINAMATH_CALUDE_second_division_remainder_l2308_230882

theorem second_division_remainder (n : ℕ) : 
  n % 68 = 0 ∧ n / 68 = 269 → n % 18291 = 1 :=
by sorry

end NUMINAMATH_CALUDE_second_division_remainder_l2308_230882


namespace NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l2308_230841

theorem fgh_supermarkets_in_us (total : ℕ) (difference : ℕ) : 
  total = 84 → difference = 10 → (total / 2 + difference / 2) = 47 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l2308_230841


namespace NUMINAMATH_CALUDE_torch_relay_probability_l2308_230832

/-- The number of torchbearers -/
def n : ℕ := 5

/-- The number of torchbearers to be selected -/
def k : ℕ := 2

/-- The total number of ways to select k torchbearers from n torchbearers -/
def total_combinations : ℕ := n.choose k

/-- The number of ways to select k consecutive torchbearers from n torchbearers -/
def consecutive_combinations : ℕ := n - k + 1

/-- The probability of selecting consecutive torchbearers -/
def probability : ℚ := consecutive_combinations / total_combinations

theorem torch_relay_probability : probability = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_torch_relay_probability_l2308_230832


namespace NUMINAMATH_CALUDE_age_difference_l2308_230828

theorem age_difference : ∃ (a b : ℕ), 
  (a ≤ 9 ∧ b ≤ 9) ∧ 
  (10 * a + b + 5 = 2 * (10 * b + a + 5)) ∧
  ((10 * a + b) - (10 * b + a) = 18) := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2308_230828


namespace NUMINAMATH_CALUDE_new_average_after_removing_scores_l2308_230876

theorem new_average_after_removing_scores (n : ℕ) (original_avg : ℚ) (score1 score2 : ℕ) :
  n = 60 →
  original_avg = 82 →
  score1 = 95 →
  score2 = 97 →
  let total_sum := n * original_avg
  let remaining_sum := total_sum - (score1 + score2)
  let new_avg := remaining_sum / (n - 2)
  new_avg = 81.52 := by sorry

end NUMINAMATH_CALUDE_new_average_after_removing_scores_l2308_230876


namespace NUMINAMATH_CALUDE_average_salary_problem_l2308_230895

theorem average_salary_problem (salary_raj roshan : ℕ) : 
  (salary_raj + roshan) / 2 = 4000 →
  ((salary_raj + roshan + 7000) : ℚ) / 3 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_problem_l2308_230895


namespace NUMINAMATH_CALUDE_solution_difference_l2308_230806

theorem solution_difference (p q : ℝ) : 
  ((3 * p - 9) / (p^2 + 3*p - 18) = p + 3) →
  ((3 * q - 9) / (q^2 + 3*q - 18) = q + 3) →
  p ≠ q →
  p > q →
  p - q = 2 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2308_230806


namespace NUMINAMATH_CALUDE_total_interest_calculation_l2308_230870

/-- Calculate total interest over 10 years with principal trebling after 5 years -/
theorem total_interest_calculation (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 600) : 
  P * R * 5 / 100 + 3 * P * R * 5 / 100 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l2308_230870


namespace NUMINAMATH_CALUDE_prob_at_least_one_defective_l2308_230864

/-- The probability of drawing a defective box from each large box -/
def p_defective : ℝ := 0.01

/-- The probability of drawing a non-defective box from each large box -/
def p_non_defective : ℝ := 1 - p_defective

/-- The number of boxes drawn -/
def n : ℕ := 3

theorem prob_at_least_one_defective :
  1 - p_non_defective ^ n = 1 - 0.99 ^ 3 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_defective_l2308_230864


namespace NUMINAMATH_CALUDE_distinct_digit_numbers_count_l2308_230862

/-- A function that counts the number of integers between 1000 and 9999 with four distinct digits -/
def count_distinct_digit_numbers : ℕ :=
  9 * 9 * 8 * 7

/-- The theorem stating that the count of integers between 1000 and 9999 with four distinct digits is 4536 -/
theorem distinct_digit_numbers_count :
  count_distinct_digit_numbers = 4536 := by
  sorry

end NUMINAMATH_CALUDE_distinct_digit_numbers_count_l2308_230862


namespace NUMINAMATH_CALUDE_correct_regression_sequence_l2308_230843

-- Define the steps of linear regression analysis
inductive RegressionStep
  | InterpretEquation
  | CollectData
  | CalculateEquation
  | CalculateCorrelation
  | DrawScatterPlot

-- Define a type for sequences of regression steps
def RegressionSequence := List RegressionStep

-- Define the correct sequence
def correctSequence : RegressionSequence :=
  [RegressionStep.CollectData,
   RegressionStep.DrawScatterPlot,
   RegressionStep.CalculateCorrelation,
   RegressionStep.CalculateEquation,
   RegressionStep.InterpretEquation]

-- Theorem stating that the defined sequence is correct
theorem correct_regression_sequence :
  correctSequence = [RegressionStep.CollectData,
                     RegressionStep.DrawScatterPlot,
                     RegressionStep.CalculateCorrelation,
                     RegressionStep.CalculateEquation,
                     RegressionStep.InterpretEquation] :=
by sorry

end NUMINAMATH_CALUDE_correct_regression_sequence_l2308_230843


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l2308_230860

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l2308_230860


namespace NUMINAMATH_CALUDE_earning_amount_l2308_230823

/-- Represents the earning and spending pattern over 60 days -/
def pattern_result (E : ℚ) : ℚ :=
  30 * (E - 15)

/-- Proves that the earning amount must be 17 given the conditions -/
theorem earning_amount : ∃ E : ℚ, pattern_result E = 60 ∧ E = 17 := by
  sorry

end NUMINAMATH_CALUDE_earning_amount_l2308_230823


namespace NUMINAMATH_CALUDE_eggs_to_market_l2308_230808

/-- Represents the number of dozens of eggs collected on each collection day -/
def eggs_collected_per_day : ℕ := 8

/-- Represents the number of collection days per week -/
def collection_days_per_week : ℕ := 2

/-- Represents the number of dozens of eggs delivered to the mall -/
def eggs_to_mall : ℕ := 5

/-- Represents the number of dozens of eggs used for pie -/
def eggs_for_pie : ℕ := 4

/-- Represents the number of dozens of eggs donated to charity -/
def eggs_to_charity : ℕ := 4

/-- Represents the total number of dozens of eggs collected in a week -/
def total_eggs_collected : ℕ := eggs_collected_per_day * collection_days_per_week

/-- Represents the total number of dozens of eggs used or given away -/
def total_eggs_used : ℕ := eggs_to_mall + eggs_for_pie + eggs_to_charity

/-- Proves that the number of dozens of eggs delivered to the market is 3 -/
theorem eggs_to_market : total_eggs_collected - total_eggs_used = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_to_market_l2308_230808


namespace NUMINAMATH_CALUDE_paperclip_capacity_l2308_230818

theorem paperclip_capacity (small_volume small_capacity large_volume efficiency : ℝ) 
  (h1 : small_volume = 12)
  (h2 : small_capacity = 40)
  (h3 : large_volume = 60)
  (h4 : efficiency = 0.8)
  : (large_volume * efficiency * small_capacity) / small_volume = 160 := by
  sorry

end NUMINAMATH_CALUDE_paperclip_capacity_l2308_230818


namespace NUMINAMATH_CALUDE_right_triangle_inequality_right_triangle_inequality_optimal_l2308_230821

theorem right_triangle_inequality (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (side_order : a ≤ b ∧ b < c) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3 * Real.sqrt 2) * a * b * c :=
by sorry

theorem right_triangle_inequality_optimal (k : ℝ) 
  (h : ∀ (a b c : ℝ), a^2 + b^2 = c^2 → a ≤ b → b < c → 
    a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ k * a * b * c) :
  k ≤ 2 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_right_triangle_inequality_optimal_l2308_230821


namespace NUMINAMATH_CALUDE_joes_total_lift_weight_l2308_230859

-- Define the weights of the lifts
def first_lift : ℕ := 400
def second_lift : ℕ := 2 * first_lift - 300

-- Define the total weight
def total_weight : ℕ := first_lift + second_lift

-- Theorem statement
theorem joes_total_lift_weight : total_weight = 900 := by
  sorry

end NUMINAMATH_CALUDE_joes_total_lift_weight_l2308_230859


namespace NUMINAMATH_CALUDE_min_value_of_a_l2308_230894

theorem min_value_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 9 * x₁ - (4 + a) * 3 * x₁ + 4 = 0 ∧ 
                 9 * x₂ - (4 + a) * 3 * x₂ + 4 = 0) →
  a ≥ 0 ∧ ∀ ε > 0, ∃ a' : ℝ, a' < ε ∧ 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 9 * x₁ - (4 + a') * 3 * x₁ + 4 = 0 ∧ 
                  9 * x₂ - (4 + a') * 3 * x₂ + 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2308_230894


namespace NUMINAMATH_CALUDE_attic_junk_percentage_l2308_230837

theorem attic_junk_percentage :
  ∀ (total useful heirlooms junk : ℕ),
    useful = (20 : ℕ) * total / 100 →
    heirlooms = (10 : ℕ) * total / 100 →
    useful = 8 →
    junk = 28 →
    total = useful + heirlooms + junk →
    (junk : ℚ) / (total : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_attic_junk_percentage_l2308_230837


namespace NUMINAMATH_CALUDE_f_discontinuities_l2308_230848

noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then
    if x ≠ -2 then (x^2 + 7*x + 10) / (x^2 - 4)
    else 11/4
  else 4*x - 3

theorem f_discontinuities :
  (∃ (L : ℝ), ContinuousAt (fun x => if x ≠ -2 then f x else L) (-2)) ∧
  (¬ ContinuousAt f 2) := by
  sorry

end NUMINAMATH_CALUDE_f_discontinuities_l2308_230848


namespace NUMINAMATH_CALUDE_simplify_expression_l2308_230820

theorem simplify_expression (y : ℝ) : 7 * y + 8 - 3 * y + 16 = 4 * y + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2308_230820


namespace NUMINAMATH_CALUDE_function_range_theorem_l2308_230898

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_range_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ≠ 0, is_odd_function f) →
  (∀ x ≠ 0, f (x + 5/2) * f x = 1) →
  f (-1) > 1 →
  f 2016 = (a + 3) / (a - 3) →
  0 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_function_range_theorem_l2308_230898


namespace NUMINAMATH_CALUDE_dance_event_women_count_l2308_230850

/-- Proves that given the conditions of the dance event, the number of women who attended is 12 -/
theorem dance_event_women_count :
  ∀ (num_men : ℕ) (men_partners : ℕ) (women_partners : ℕ),
    num_men = 9 →
    men_partners = 4 →
    women_partners = 3 →
    ∃ (num_women : ℕ),
      num_women * women_partners = num_men * men_partners ∧
      num_women = 12 := by
  sorry

end NUMINAMATH_CALUDE_dance_event_women_count_l2308_230850


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2308_230878

theorem product_sum_theorem (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 120 →
  a + b + c + d + e = 27 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2308_230878


namespace NUMINAMATH_CALUDE_solution_set_of_even_monotonic_function_l2308_230858

def f (a b x : ℝ) := (x - 2) * (a * x + b)

theorem solution_set_of_even_monotonic_function 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_increasing : ∀ x y, 0 < x → x < y → f a b x < f a b y) :
  {x : ℝ | f a b (2 - x) > 0} = {x : ℝ | x < 0 ∨ x > 4} := by
sorry


end NUMINAMATH_CALUDE_solution_set_of_even_monotonic_function_l2308_230858


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2308_230836

theorem purely_imaginary_complex_number (x : ℝ) :
  let z : ℂ := (x^2 - 1) + (x + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2308_230836


namespace NUMINAMATH_CALUDE_tan_half_implies_expression_eight_l2308_230825

theorem tan_half_implies_expression_eight (x : ℝ) (h : Real.tan x = 1 / 2) :
  (2 * Real.sin x + 3 * Real.cos x) / (Real.cos x - Real.sin x) = 8 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_implies_expression_eight_l2308_230825


namespace NUMINAMATH_CALUDE_unifying_sqrt_plus_m_range_l2308_230863

/-- A function is unifying on [a,b] if it's monotonic and maps [a,b] onto itself --/
def IsUnifying (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  Monotone f ∧ 
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

/-- The theorem stating the range of m for which f(x) = √(x+1) + m is a unifying function --/
theorem unifying_sqrt_plus_m_range :
  ∃ a b : ℝ, a < b ∧ 
  (∃ m : ℝ, IsUnifying (fun x ↦ Real.sqrt (x + 1) + m) a b) ↔ 
  m ∈ Set.Ioo (-5/4) (-1) ∪ {-1} :=
sorry

end NUMINAMATH_CALUDE_unifying_sqrt_plus_m_range_l2308_230863


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l2308_230827

theorem largest_integer_inequality (y : ℤ) : (y / 4 : ℚ) + 3 / 7 < 7 / 4 ↔ y ≤ 5 := by
  sorry

#check largest_integer_inequality

end NUMINAMATH_CALUDE_largest_integer_inequality_l2308_230827


namespace NUMINAMATH_CALUDE_parabola_focus_l2308_230872

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := y = 2 * x^2 + 4 * x + 5

-- Define the focus of a parabola
def is_focus (x y : ℝ) (f : ℝ × ℝ) : Prop :=
  f.1 = x ∧ f.2 = y

-- Theorem statement
theorem parabola_focus :
  ∃ (f : ℝ × ℝ), is_focus (-1) (25/8) f ∧
  ∀ (x y : ℝ), parabola_equation x y →
  is_focus x y f :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l2308_230872


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2308_230822

/-- The line equation passing through a fixed point -/
def line_equation (k x y : ℝ) : Prop :=
  k * x + (1 - k) * y - 3 = 0

/-- Theorem stating that the line passes through (3, 3) for all k -/
theorem fixed_point_on_line :
  ∀ (k : ℝ), line_equation k 3 3 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2308_230822


namespace NUMINAMATH_CALUDE_proposition_implication_l2308_230899

theorem proposition_implication (m : ℝ) : 
  (∀ x, -2 ≤ x ∧ x ≤ 10 → 1 - m ≤ x ∧ x ≤ 1 + m) ∧ 
  (∃ x, 1 - m ≤ x ∧ x ≤ 1 + m ∧ (x < -2 ∨ x > 10)) ∧
  (m > 0) →
  m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_proposition_implication_l2308_230899


namespace NUMINAMATH_CALUDE_root_expression_value_l2308_230887

theorem root_expression_value (r s : ℝ) : 
  (3 * r^2 + 4 * r - 18 = 0) →
  (3 * s^2 + 4 * s - 18 = 0) →
  r ≠ s →
  (3 * r^3 - 3 * s^3) / (r - s) = 70/3 := by
sorry

end NUMINAMATH_CALUDE_root_expression_value_l2308_230887


namespace NUMINAMATH_CALUDE_student_multiplication_factor_l2308_230816

theorem student_multiplication_factor : ∃ (x : ℚ), 121 * x - 138 = 104 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_factor_l2308_230816


namespace NUMINAMATH_CALUDE_unique_outstanding_defeats_all_l2308_230879

/-- Represents a tournament with n participants. -/
structure Tournament (n : ℕ) where
  -- n ≥ 3
  n_ge_three : n ≥ 3
  -- Defeat relation
  defeats : Fin n → Fin n → Prop
  -- Every match has a definite winner
  winner_exists : ∀ i j : Fin n, i ≠ j → (defeats i j ∨ defeats j i) ∧ ¬(defeats i j ∧ defeats j i)

/-- Definition of an outstanding participant -/
def is_outstanding (t : Tournament n) (a : Fin n) : Prop :=
  ∀ b : Fin n, b ≠ a → t.defeats a b ∨ ∃ c : Fin n, t.defeats c b ∧ t.defeats a c

/-- The main theorem -/
theorem unique_outstanding_defeats_all (t : Tournament n) (a : Fin n) :
  (∀ b : Fin n, b ≠ a → is_outstanding t b → b = a) →
  is_outstanding t a →
  ∀ b : Fin n, b ≠ a → t.defeats a b :=
by sorry

end NUMINAMATH_CALUDE_unique_outstanding_defeats_all_l2308_230879


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_cubes_l2308_230868

theorem roots_sum_reciprocal_cubes (r s : ℝ) : 
  (3 * r^2 + 5 * r + 2 = 0) → 
  (3 * s^2 + 5 * s + 2 = 0) → 
  (r ≠ s) →
  (1 / r^3 + 1 / s^3 = -27 / 35) :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_cubes_l2308_230868
