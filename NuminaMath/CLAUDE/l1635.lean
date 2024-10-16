import Mathlib

namespace NUMINAMATH_CALUDE_blue_red_ratio_13_l1635_163559

/-- Represents the ratio of blue to red face areas in a cube cutting problem -/
def blue_to_red_ratio (n : ℕ) : ℚ :=
  (6 * n^3 - 6 * n^2) / (6 * n^2)

/-- Theorem stating that for a cube of side length 13, the ratio of blue to red face areas is 12 -/
theorem blue_red_ratio_13 : blue_to_red_ratio 13 = 12 := by
  sorry

#eval blue_to_red_ratio 13

end NUMINAMATH_CALUDE_blue_red_ratio_13_l1635_163559


namespace NUMINAMATH_CALUDE_real_equal_roots_l1635_163505

theorem real_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - k * x + x + 8 = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - k * y + y + 8 = 0 → y = x) ↔ 
  (k = 9 ∨ k = -7) :=
sorry

end NUMINAMATH_CALUDE_real_equal_roots_l1635_163505


namespace NUMINAMATH_CALUDE_expression_simplification_l1635_163503

theorem expression_simplification (x y z : ℝ) 
  (h_pos : 0 < z ∧ z < y ∧ y < x) : 
  (x^z * y^x * z^y) / (z^z * y^y * x^x) = (x/z)^(z-y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1635_163503


namespace NUMINAMATH_CALUDE_rectangular_prism_properties_l1635_163529

/-- A rectangular prism with dimensions 12, 16, and 21 inches has a diagonal length of 29 inches
    and a surface area of 1560 square inches. -/
theorem rectangular_prism_properties :
  let a : ℝ := 12
  let b : ℝ := 16
  let c : ℝ := 21
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let surface_area := 2 * (a*b + b*c + c*a)
  diagonal = 29 ∧ surface_area = 1560 := by
  sorry

#check rectangular_prism_properties

end NUMINAMATH_CALUDE_rectangular_prism_properties_l1635_163529


namespace NUMINAMATH_CALUDE_hotdogs_served_today_l1635_163553

/-- The number of hot dogs served during lunch today -/
def lunch_hotdogs : ℕ := 9

/-- The number of hot dogs served during dinner today -/
def dinner_hotdogs : ℕ := 2

/-- The total number of hot dogs served today -/
def total_hotdogs : ℕ := lunch_hotdogs + dinner_hotdogs

theorem hotdogs_served_today : total_hotdogs = 11 := by
  sorry

end NUMINAMATH_CALUDE_hotdogs_served_today_l1635_163553


namespace NUMINAMATH_CALUDE_garden_flowers_equality_l1635_163562

/-- Given a garden with white and red flowers, calculate the number of additional red flowers needed to make their quantities equal. -/
def additional_red_flowers (white : ℕ) (red : ℕ) : ℕ :=
  if white > red then white - red else 0

/-- Theorem: In a garden with 555 white flowers and 347 red flowers, 208 additional red flowers are needed to make their quantities equal. -/
theorem garden_flowers_equality : additional_red_flowers 555 347 = 208 := by
  sorry

end NUMINAMATH_CALUDE_garden_flowers_equality_l1635_163562


namespace NUMINAMATH_CALUDE_unique_equal_expression_l1635_163509

theorem unique_equal_expression (x : ℝ) (h : x > 0) :
  (x^(x+1) + x^(x+1) = 2*x^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ x^(2*x+2)) ∧
  (x^(x+1) + x^(x+1) ≠ (2*x)^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ (2*x)^(2*x+2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_equal_expression_l1635_163509


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1635_163555

-- Define the custom operation ⊛
def circledAst (a b : ℕ) : ℕ := sorry

-- Properties of ⊛
axiom circledAst_self (a : ℕ) : circledAst a a = a
axiom circledAst_zero (a : ℕ) : circledAst a 0 = 2 * a
axiom circledAst_add (a b c d : ℕ) : 
  (circledAst a b) + (circledAst c d) = circledAst (a + c) (b + d)

-- Theorems to prove
theorem problem_1 : circledAst (2 + 3) (0 + 3) = 7 := by sorry

theorem problem_2 : circledAst 1024 48 = 2000 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1635_163555


namespace NUMINAMATH_CALUDE_sequence_squared_l1635_163517

theorem sequence_squared (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) > a n) 
  (h3 : ∀ n, (a (n + 1))^2 + (a n)^2 + 1 = 2 * ((a (n + 1)) * (a n) + a (n + 1) + a n)) :
  ∀ n, a n = n^2 := by sorry

end NUMINAMATH_CALUDE_sequence_squared_l1635_163517


namespace NUMINAMATH_CALUDE_composite_function_ratio_l1635_163550

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem
theorem composite_function_ratio :
  (f (g (f 2))) / (g (f (g 2))) = 41 / 7 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_ratio_l1635_163550


namespace NUMINAMATH_CALUDE_kamal_present_age_l1635_163564

/-- Represents the present age of Kamal -/
def kamal_age : ℕ := sorry

/-- Represents the present age of Kamal's son -/
def son_age : ℕ := sorry

/-- Kamal was 4 times as old as his son 8 years ago -/
axiom condition1 : kamal_age - 8 = 4 * (son_age - 8)

/-- After 8 years, Kamal will be twice as old as his son -/
axiom condition2 : kamal_age + 8 = 2 * (son_age + 8)

/-- Theorem stating that Kamal's present age is 40 years -/
theorem kamal_present_age : kamal_age = 40 := by sorry

end NUMINAMATH_CALUDE_kamal_present_age_l1635_163564


namespace NUMINAMATH_CALUDE_exists_right_triangle_with_perpendicular_medians_l1635_163525

/-- A right-angled triangle with one given leg and perpendicular medians to the other two sides -/
structure RightTriangleWithPerpendicularMedians where
  /-- The length of the given leg -/
  a : ℝ
  /-- The length of the second leg -/
  b : ℝ
  /-- The length of the hypotenuse -/
  c : ℝ
  /-- The given leg is positive -/
  a_pos : 0 < a
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagoras : a^2 + b^2 = c^2
  /-- The medians to the other two sides are perpendicular -/
  medians_perpendicular : (2*c^2 + 2*b^2 - a^2) * (2*c^2 + 2*a^2 - b^2) = 9*a^2*b^2

/-- There exists a right-angled triangle with one given leg and perpendicular medians to the other two sides -/
theorem exists_right_triangle_with_perpendicular_medians (a : ℝ) (ha : 0 < a) : 
  ∃ t : RightTriangleWithPerpendicularMedians, t.a = a :=
sorry

end NUMINAMATH_CALUDE_exists_right_triangle_with_perpendicular_medians_l1635_163525


namespace NUMINAMATH_CALUDE_range_of_a_l1635_163528

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 65

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, f a x > 0) → -16 < a ∧ a < 16 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1635_163528


namespace NUMINAMATH_CALUDE_custom_op_7_neg3_custom_op_not_commutative_l1635_163587

-- Define the custom operation ※
def custom_op (a b : ℤ) : ℤ := (a + 2) * 2 - b

-- Theorem 1: 7 ※ (-3) = 21
theorem custom_op_7_neg3 : custom_op 7 (-3) = 21 := by sorry

-- Theorem 2: 7 ※ (-3) ≠ (-3) ※ 7
theorem custom_op_not_commutative : custom_op 7 (-3) ≠ custom_op (-3) 7 := by sorry

end NUMINAMATH_CALUDE_custom_op_7_neg3_custom_op_not_commutative_l1635_163587


namespace NUMINAMATH_CALUDE_program_cost_calculation_l1635_163536

-- Define constants
def millisecond_to_second : Real := 0.001
def minute_to_millisecond : Nat := 60000
def os_overhead_cost : Real := 1.07
def computer_time_cost_per_ms : Real := 0.023
def data_tape_cost : Real := 5.35
def memory_cost_per_mb : Real := 0.15
def electricity_cost_per_kwh : Real := 0.02
def program_runtime_minutes : Nat := 45
def program_memory_gb : Real := 3.5
def program_electricity_kwh : Real := 2
def gb_to_mb : Nat := 1024

-- Define the theorem
theorem program_cost_calculation :
  let total_milliseconds := program_runtime_minutes * minute_to_millisecond
  let computer_time_cost := total_milliseconds * computer_time_cost_per_ms
  let memory_usage_mb := program_memory_gb * gb_to_mb
  let memory_cost := memory_usage_mb * memory_cost_per_mb
  let electricity_cost := program_electricity_kwh * electricity_cost_per_kwh
  let total_cost := os_overhead_cost + computer_time_cost + data_tape_cost + memory_cost + electricity_cost
  total_cost = 62644.06 := by
  sorry

end NUMINAMATH_CALUDE_program_cost_calculation_l1635_163536


namespace NUMINAMATH_CALUDE_unique_intersecting_line_l1635_163599

/-- A parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 8 * p.1}

/-- The point M with coordinates (2, 4) -/
def M : ℝ × ℝ := (2, 4)

/-- A line that passes through point M and intersects the parabola at exactly one point -/
def UniqueLine (l : Set (ℝ × ℝ)) : Prop :=
  M ∈ l ∧ (∃! p, p ∈ l ∩ Parabola)

/-- The theorem stating that there is exactly one unique line passing through M
    that intersects the parabola at exactly one point -/
theorem unique_intersecting_line :
  ∃! l : Set (ℝ × ℝ), UniqueLine l :=
sorry

end NUMINAMATH_CALUDE_unique_intersecting_line_l1635_163599


namespace NUMINAMATH_CALUDE_company_sugar_usage_l1635_163502

/-- The amount of sugar (in grams) used by a chocolate company in two minutes -/
def sugar_used_in_two_minutes (sugar_per_bar : ℝ) (bars_per_minute : ℝ) : ℝ :=
  2 * (sugar_per_bar * bars_per_minute)

/-- Theorem stating that the company uses 108 grams of sugar in two minutes -/
theorem company_sugar_usage :
  sugar_used_in_two_minutes 1.5 36 = 108 := by
  sorry

end NUMINAMATH_CALUDE_company_sugar_usage_l1635_163502


namespace NUMINAMATH_CALUDE_salary_after_adjustments_l1635_163561

-- Define the initial salary
def initial_salary : ℝ := 3000

-- Define the raise percentage
def raise_percentage : ℝ := 0.15

-- Define the reduction percentage
def reduction_percentage : ℝ := 0.10

-- Theorem to prove
theorem salary_after_adjustments :
  initial_salary * (1 + raise_percentage) * (1 - reduction_percentage) = 3105 := by
  sorry

end NUMINAMATH_CALUDE_salary_after_adjustments_l1635_163561


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_quadratic_distinct_positive_integer_roots_l1635_163568

/-- The quadratic equation mx^2 - (m+2)x + 2 = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 - (m + 2) * x + 2 = 0

theorem quadratic_always_real_roots :
  ∀ m : ℝ, ∃ x : ℝ, quadratic_equation m x :=
sorry

theorem quadratic_distinct_positive_integer_roots :
  ∀ m : ℤ, (∃ x y : ℤ, x ≠ y ∧ 0 < x ∧ 0 < y ∧ quadratic_equation (m : ℝ) (x : ℝ) ∧ quadratic_equation (m : ℝ) (y : ℝ)) ↔ m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_quadratic_distinct_positive_integer_roots_l1635_163568


namespace NUMINAMATH_CALUDE_range_of_m_l1635_163577

-- Define the necessary condition p
def p (x m : ℝ) : Prop := (x - m)^2 > 3*(x - m)

-- Define the condition q
def q (x : ℝ) : Prop := x^2 - 3*x - 4 ≤ 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ (∃ x, p x m ∧ ¬q x)

-- Theorem statement
theorem range_of_m :
  {m : ℝ | necessary_but_not_sufficient m} = {m | m < -4 ∨ m > 4} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1635_163577


namespace NUMINAMATH_CALUDE_baron_munchausen_claim_false_l1635_163589

theorem baron_munchausen_claim_false : ∃ n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ ¬∃ m : ℕ, 0 ≤ m ∧ m ≤ 99 ∧ (100 * n + m)^2 = 100 * n + m := by
  sorry

end NUMINAMATH_CALUDE_baron_munchausen_claim_false_l1635_163589


namespace NUMINAMATH_CALUDE_rectangle_with_tangent_circle_l1635_163569

theorem rectangle_with_tangent_circle 
  (r : ℝ) 
  (h1 : r = 6) 
  (A_circle : ℝ) 
  (h2 : A_circle = π * r^2) 
  (A_rectangle : ℝ) 
  (h3 : A_rectangle = 3 * A_circle) 
  (shorter_side : ℝ) 
  (h4 : shorter_side = 2 * r) 
  (longer_side : ℝ) 
  (h5 : A_rectangle = shorter_side * longer_side) : 
  longer_side = 9 * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_with_tangent_circle_l1635_163569


namespace NUMINAMATH_CALUDE_total_different_books_l1635_163516

/-- The number of different books read by three people given their individual book counts and shared book information. -/
def differentBooksRead (tonyBooks deanBooks breannaBooks tonyDeanShared allShared : ℕ) : ℕ :=
  tonyBooks + deanBooks + breannaBooks - tonyDeanShared - 2 * allShared

/-- Theorem stating that Tony, Dean, and Breanna read 47 different books in total. -/
theorem total_different_books : 
  differentBooksRead 23 12 17 3 1 = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_different_books_l1635_163516


namespace NUMINAMATH_CALUDE_horizontal_distance_calculation_l1635_163548

/-- Given a vertical climb and a ratio of vertical to horizontal movement,
    calculate the horizontal distance traveled. -/
theorem horizontal_distance_calculation
  (vertical_climb : ℝ)
  (vertical_ratio : ℝ)
  (horizontal_ratio : ℝ)
  (h_positive : vertical_climb > 0)
  (h_ratio_positive : vertical_ratio > 0 ∧ horizontal_ratio > 0)
  (h_climb : vertical_climb = 1350)
  (h_ratio : vertical_ratio / horizontal_ratio = 1 / 2) :
  vertical_climb * horizontal_ratio / vertical_ratio = 2700 := by
  sorry

end NUMINAMATH_CALUDE_horizontal_distance_calculation_l1635_163548


namespace NUMINAMATH_CALUDE_faster_train_speed_l1635_163512

/-- The speed of the faster train given two trains moving in opposite directions --/
theorem faster_train_speed
  (slow_speed : ℝ)
  (length_slow : ℝ)
  (length_fast : ℝ)
  (crossing_time : ℝ)
  (h_slow_speed : slow_speed = 60)
  (h_length_slow : length_slow = 1.10)
  (h_length_fast : length_fast = 0.9)
  (h_crossing_time : crossing_time = 47.99999999999999 / 3600) :
  ∃ (fast_speed : ℝ), fast_speed = 90 ∧
    (fast_speed + slow_speed) * crossing_time = length_slow + length_fast :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l1635_163512


namespace NUMINAMATH_CALUDE_first_divisor_problem_l1635_163515

theorem first_divisor_problem (x : ℕ) : x = 31 ↔ 
  x > 9 ∧ 
  x < 282 ∧
  282 % x = 3 ∧
  282 % 9 = 3 ∧
  279 % x = 0 ∧
  ∀ y : ℕ, y > 9 ∧ y < x → (282 % y ≠ 3 ∨ 279 % y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l1635_163515


namespace NUMINAMATH_CALUDE_sister_chromatid_separation_in_second_division_sister_chromatid_separation_not_in_other_stages_l1635_163537

/-- Represents the stages of meiosis --/
inductive MeiosisStage
  | Interphase
  | TetradFormation
  | FirstDivision
  | SecondDivision

/-- Represents the events that occur during meiosis --/
inductive MeiosisEvent
  | ChromosomeReplication
  | HomologousPairing
  | ChromatidSeparation

/-- Defines the characteristics of each meiosis stage --/
def stageCharacteristics : MeiosisStage → List MeiosisEvent
  | MeiosisStage.Interphase => [MeiosisEvent.ChromosomeReplication]
  | MeiosisStage.TetradFormation => [MeiosisEvent.HomologousPairing]
  | MeiosisStage.FirstDivision => []
  | MeiosisStage.SecondDivision => [MeiosisEvent.ChromatidSeparation]

/-- Theorem: Sister chromatid separation occurs during the second meiotic division --/
theorem sister_chromatid_separation_in_second_division :
  MeiosisEvent.ChromatidSeparation ∈ stageCharacteristics MeiosisStage.SecondDivision :=
by sorry

/-- Corollary: Sister chromatid separation does not occur in other stages --/
theorem sister_chromatid_separation_not_in_other_stages :
  ∀ stage, stage ≠ MeiosisStage.SecondDivision →
    MeiosisEvent.ChromatidSeparation ∉ stageCharacteristics stage :=
by sorry

end NUMINAMATH_CALUDE_sister_chromatid_separation_in_second_division_sister_chromatid_separation_not_in_other_stages_l1635_163537


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l1635_163594

def numbers : List ℕ := [18, 27, 45]

theorem arithmetic_mean_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 30 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l1635_163594


namespace NUMINAMATH_CALUDE_suraya_kayla_difference_l1635_163583

/-- The number of apples picked by each person --/
structure ApplePickers where
  suraya : ℕ
  caleb : ℕ
  kayla : ℕ

/-- The conditions of the apple-picking scenario --/
def apple_picking_scenario (p : ApplePickers) : Prop :=
  p.suraya = p.caleb + 12 ∧
  p.caleb + 5 = p.kayla ∧
  p.kayla = 20

/-- The theorem stating the difference between Suraya's and Kayla's apple count --/
theorem suraya_kayla_difference (p : ApplePickers) 
  (h : apple_picking_scenario p) : p.suraya - p.kayla = 7 := by
  sorry

end NUMINAMATH_CALUDE_suraya_kayla_difference_l1635_163583


namespace NUMINAMATH_CALUDE_tan_alpha_value_implies_expression_value_l1635_163597

theorem tan_alpha_value_implies_expression_value (α : Real) 
  (h : Real.tan α = -1/2) : 
  (Real.sin (2 * α) + 2 * Real.cos (2 * α)) / 
  (4 * Real.cos (2 * α) - 4 * Real.sin (2 * α)) = 1/14 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_implies_expression_value_l1635_163597


namespace NUMINAMATH_CALUDE_sum_equals_932_l1635_163596

-- Define the value of a number in a given base
def value_in_base (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

-- Define the given numbers
def num1 : ℕ := value_in_base [3, 5, 1] 7
def num2 : ℕ := value_in_base [13, 12, 4] 13

-- Theorem to prove
theorem sum_equals_932 : num1 + num2 = 932 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_932_l1635_163596


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1635_163531

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1  -- Arithmetic property
  h_sum : ∀ n, S n = n * (a 1 + a n) / 2  -- Sum formula

/-- Theorem: For an arithmetic sequence with S₁ = 1 and S₄/S₂ = 4, S₆/S₄ = 9/4 -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence)
    (h1 : seq.S 1 = 1)
    (h2 : seq.S 4 / seq.S 2 = 4) :
  seq.S 6 / seq.S 4 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1635_163531


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1635_163551

theorem inequality_equivalence (p : ℝ) : 
  (∀ q : ℝ, q > 0 → p + q ≠ 0 → (3 * (p * q^2 + 2 * p^2 * q + 2 * q^2 + 5 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 
  (0 ≤ p ∧ p ≤ 7.275) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1635_163551


namespace NUMINAMATH_CALUDE_f_problem_l1635_163598

noncomputable section

variable (f : ℝ → ℝ)

axiom f_increasing : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom f_domain : ∀ x, 0 < x → ∃ y, f x = y
axiom f_property : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y
axiom f_6 : f 6 = 1

theorem f_problem :
  (f 1 = 0) ∧
  (∀ x, 0 < x → (f (x + 3) - f (1 / x) < 2 ↔ 0 < x ∧ x < (-3 + 3 * Real.sqrt 17) / 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_f_problem_l1635_163598


namespace NUMINAMATH_CALUDE_sprint_team_total_distance_l1635_163560

theorem sprint_team_total_distance (team_size : ℕ) (distance_per_person : ℝ) :
  team_size = 250 →
  distance_per_person = 7.5 →
  team_size * distance_per_person = 1875 := by
sorry

end NUMINAMATH_CALUDE_sprint_team_total_distance_l1635_163560


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1635_163501

/-- Given that Jacob is 24 years old now and Tony will be 18 years old in 6 years,
    prove that the ratio of Tony's current age to Jacob's current age is 1:2. -/
theorem age_ratio_proof (jacob_age : ℕ) (tony_future_age : ℕ) (years_until_future : ℕ) :
  jacob_age = 24 →
  tony_future_age = 18 →
  years_until_future = 6 →
  (tony_future_age - years_until_future) * 2 = jacob_age := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1635_163501


namespace NUMINAMATH_CALUDE_volleyball_tournament_wins_l1635_163539

theorem volleyball_tournament_wins (n : ℕ) (h_n : n = 73) :
  ∀ (p m : ℕ) (x : ℕ) (h_x : 0 < x ∧ x < n),
  x * p + (n - x) * m = n * (n - 1) / 2 →
  p = m :=
by sorry

end NUMINAMATH_CALUDE_volleyball_tournament_wins_l1635_163539


namespace NUMINAMATH_CALUDE_consecutive_math_majors_probability_l1635_163547

/-- The number of people sitting around the table -/
def total_people : ℕ := 11

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 3

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 3

/-- The probability of all math majors sitting consecutively -/
def prob_consecutive_math_majors : ℚ := 1 / 4320

theorem consecutive_math_majors_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := (total_people - math_majors + 1) * Nat.factorial math_majors
  Rat.cast favorable_arrangements / Rat.cast total_arrangements = prob_consecutive_math_majors := by
  sorry

end NUMINAMATH_CALUDE_consecutive_math_majors_probability_l1635_163547


namespace NUMINAMATH_CALUDE_four_students_arrangement_l1635_163520

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange four students in a line
    with three students standing together. -/
def arrangements_with_restriction : ℕ :=
  permutations 2 * permutations 3

theorem four_students_arrangement :
  permutations 4 - arrangements_with_restriction = 12 := by
  sorry

end NUMINAMATH_CALUDE_four_students_arrangement_l1635_163520


namespace NUMINAMATH_CALUDE_meetings_count_l1635_163582

/-- Represents the movement of an individual between two points -/
structure Movement where
  speed : ℝ
  journeys : ℕ

/-- Calculates the number of meetings between two individuals -/
def calculate_meetings (a b : Movement) : ℕ :=
  sorry

theorem meetings_count :
  let a : Movement := { speed := 1, journeys := 2015 }
  let b : Movement := { speed := 2, journeys := 4029 }
  (calculate_meetings a b) = 6044 := by
  sorry

end NUMINAMATH_CALUDE_meetings_count_l1635_163582


namespace NUMINAMATH_CALUDE_ball_max_height_l1635_163595

/-- The height function of a ball thrown upwards -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 20

/-- The maximum height achieved by the ball -/
theorem ball_max_height : ∃ (max : ℝ), ∀ (t : ℝ), h t ≤ max ∧ ∃ (t_max : ℝ), h t_max = max :=
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l1635_163595


namespace NUMINAMATH_CALUDE_larger_integer_problem_l1635_163500

theorem larger_integer_problem (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) (h4 : y - x = 8) (h5 : x * y = 272) : y = 17 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l1635_163500


namespace NUMINAMATH_CALUDE_gcd_459_357_l1635_163592

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1635_163592


namespace NUMINAMATH_CALUDE_largest_valid_number_l1635_163576

def is_valid (n : ℕ) : Prop :=
  n % 10 ≠ 0 ∧
  ∀ (a b : ℕ), a < 10 → b < 10 →
    ∃ (x y z : ℕ), n = x * 100 + a * 10 + b + y * 10 + z ∧
    n % (x * 10 + y + z) = 0

theorem largest_valid_number : 
  is_valid 9999 ∧ 
  ∀ m : ℕ, m > 9999 → ¬(is_valid m) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l1635_163576


namespace NUMINAMATH_CALUDE_largest_solution_is_57_98_l1635_163540

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

/-- The equation from the problem -/
def equation (x : ℝ) : Prop := (floor x : ℝ) = 8 + 50 * (frac x)

/-- The theorem statement -/
theorem largest_solution_is_57_98 :
  ∃ (x : ℝ), equation x ∧ (∀ (y : ℝ), equation y → y ≤ x) ∧ x = 57.98 :=
sorry

end NUMINAMATH_CALUDE_largest_solution_is_57_98_l1635_163540


namespace NUMINAMATH_CALUDE_sandwich_theorem_l1635_163588

/-- The number of sandwiches Samson ate on different days and meals --/
def sandwich_count : Prop :=
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let monday_total := monday_lunch + monday_dinner
  let tuesday_total := tuesday_lunch + tuesday_dinner
  let wednesday_total := wednesday_lunch + wednesday_dinner
  wednesday_total - (monday_total + tuesday_total) = 5

theorem sandwich_theorem : sandwich_count := by
  sorry

end NUMINAMATH_CALUDE_sandwich_theorem_l1635_163588


namespace NUMINAMATH_CALUDE_maxwell_current_age_l1635_163566

/-- Maxwell's current age --/
def maxwell_age : ℕ := 6

/-- Maxwell's sister's current age --/
def sister_age : ℕ := 2

/-- Years into the future when Maxwell will be twice his sister's age --/
def years_future : ℕ := 2

theorem maxwell_current_age :
  maxwell_age = 6 ∧
  sister_age = 2 ∧
  maxwell_age + years_future = 2 * (sister_age + years_future) :=
by sorry

end NUMINAMATH_CALUDE_maxwell_current_age_l1635_163566


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1635_163508

theorem smallest_number_proof (a b c : ℚ) : 
  b = 4 * a →
  c = 2 * b →
  (a + b + c) / 3 = 78 →
  a = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1635_163508


namespace NUMINAMATH_CALUDE_sam_remaining_yellow_marbles_l1635_163518

/-- The number of yellow marbles Sam has after Joan took some -/
def remaining_yellow_marbles (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Proof that Sam has 61 yellow marbles after Joan took 25 -/
theorem sam_remaining_yellow_marbles :
  remaining_yellow_marbles 86 25 = 61 := by
  sorry

end NUMINAMATH_CALUDE_sam_remaining_yellow_marbles_l1635_163518


namespace NUMINAMATH_CALUDE_total_apples_l1635_163570

/-- Represents the number of apples in a pack -/
def apples_per_pack : ℕ := 4

/-- Represents the number of packs bought -/
def packs_bought : ℕ := 2

/-- Theorem stating that buying 2 packs of 4 apples each results in 8 apples total -/
theorem total_apples : apples_per_pack * packs_bought = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l1635_163570


namespace NUMINAMATH_CALUDE_jennas_age_l1635_163514

/-- Given that Jenna is 5 years older than Darius, their ages sum to 21, and Darius is 8 years old,
    prove that Jenna is 13 years old. -/
theorem jennas_age (jenna_age darius_age : ℕ) 
  (h1 : jenna_age = darius_age + 5)
  (h2 : jenna_age + darius_age = 21)
  (h3 : darius_age = 8) :
  jenna_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_jennas_age_l1635_163514


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l1635_163541

theorem coefficient_x_squared (n : ℕ) : 
  (2 : ℤ) * 4 * (Nat.choose 6 2) - 2 * (Nat.choose 6 1) + 1 = 109 := by
  sorry

#check coefficient_x_squared

end NUMINAMATH_CALUDE_coefficient_x_squared_l1635_163541


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l1635_163521

theorem smallest_right_triangle_area :
  ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 4 ∨ b = 4 ∨ c = 4) →
  (a = 6 ∨ b = 6 ∨ c = 6) →
  a^2 + b^2 = c^2 →
  (1/2 * a * b) ≥ 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l1635_163521


namespace NUMINAMATH_CALUDE_binary_1010_is_10_l1635_163519

/-- Converts a binary number represented as a list of bits (0s and 1s) to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

theorem binary_1010_is_10 : binary_to_decimal [0, 1, 0, 1] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_is_10_l1635_163519


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1635_163579

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 7 ≤ 2) ↔ (x ≤ 3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1635_163579


namespace NUMINAMATH_CALUDE_segment_length_theorem_solvability_condition_l1635_163593

/-- Two mutually tangent circles with radii r₁ and r₂ -/
structure TangentCircles where
  r₁ : ℝ
  r₂ : ℝ
  r₁_pos : r₁ > 0
  r₂_pos : r₂ > 0

/-- A line intersecting two circles in four points, creating three equal segments -/
structure IntersectingLine (tc : TangentCircles) where
  d : ℝ
  d_pos : d > 0
  intersects_circles : True  -- This is a placeholder for the intersection property

/-- The main theorem relating the segment length to the radii -/
theorem segment_length_theorem (tc : TangentCircles) (l : IntersectingLine tc) :
    l.d^2 = (1/12) * (14*tc.r₁*tc.r₂ - tc.r₁^2 - tc.r₂^2) := by sorry

/-- The solvability condition for the problem -/
theorem solvability_condition (tc : TangentCircles) :
    (∃ l : IntersectingLine tc, True) ↔ 
    (7 - 4*Real.sqrt 3 ≤ tc.r₁ / tc.r₂ ∧ tc.r₁ / tc.r₂ ≤ 7 + 4*Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_segment_length_theorem_solvability_condition_l1635_163593


namespace NUMINAMATH_CALUDE_incorrect_calculations_l1635_163558

theorem incorrect_calculations : 
  (¬ (4237 * 27925 = 118275855)) ∧ 
  (¬ (42971064 / 8264 = 5201)) ∧ 
  (¬ (1965^2 = 3761225)) ∧ 
  (¬ (371293^(1/5) = 23)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculations_l1635_163558


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1635_163542

theorem quadratic_equation_roots (p : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + p * x = 2) ∧ 
  (3 * (-1)^2 + p * (-1) = 2) →
  (3 * (2/3)^2 + p * (2/3) = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1635_163542


namespace NUMINAMATH_CALUDE_pizza_slice_cost_l1635_163526

theorem pizza_slice_cost (num_pizzas : ℕ) (slices_per_pizza : ℕ) (total_cost : ℚ) :
  num_pizzas = 3 →
  slices_per_pizza = 12 →
  total_cost = 72 →
  (5 : ℚ) * (total_cost / (↑num_pizzas * ↑slices_per_pizza)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_cost_l1635_163526


namespace NUMINAMATH_CALUDE_percentage_to_pass_l1635_163565

/-- Given a test with maximum marks and a student's performance, 
    calculate the percentage needed to pass the test. -/
theorem percentage_to_pass (max_marks student_marks shortfall : ℕ) :
  max_marks = 400 →
  student_marks = 80 →
  shortfall = 40 →
  (((student_marks + shortfall) : ℚ) / max_marks) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l1635_163565


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1635_163522

/-- A complex number z is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Definition of the complex number z in terms of m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 2) (m^2 - 3*m - 2)

/-- m = -1 is a sufficient but not necessary condition for z to be purely imaginary -/
theorem sufficient_not_necessary_condition :
  (isPurelyImaginary (z (-1))) ∧
  (∃ m : ℝ, m ≠ -1 ∧ isPurelyImaginary (z m)) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1635_163522


namespace NUMINAMATH_CALUDE_square_formation_possible_l1635_163527

theorem square_formation_possible (figure_area : ℕ) (h : figure_area = 4) :
  ∃ (n : ℕ), n > 0 ∧ (n * n) % figure_area = 0 :=
sorry

end NUMINAMATH_CALUDE_square_formation_possible_l1635_163527


namespace NUMINAMATH_CALUDE_vector_angle_difference_l1635_163563

theorem vector_angle_difference (α β : Real) (a b : Fin 2 → Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (ha : a = λ i => if i = 0 then Real.cos α else Real.sin α)
  (hb : b = λ i => if i = 0 then Real.cos β else Real.sin β)
  (h_eq : ‖(2 : Real) • a + b‖ = ‖a - (2 : Real) • b‖) :
  β - α = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_difference_l1635_163563


namespace NUMINAMATH_CALUDE_complex_number_problem_l1635_163510

theorem complex_number_problem (z : ℂ) : 
  (∃ (b : ℝ), z = b * I) → 
  (∃ (c : ℝ), (z + 2)^2 + 8 * I = c * I) → 
  z = 2 * I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1635_163510


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l1635_163554

/-- The polynomial with coefficients p and q -/
def polynomial (p q : ℚ) (x : ℚ) : ℚ :=
  p * x^4 + q * x^3 + 20 * x^2 - 10 * x + 15

/-- The factor of the polynomial -/
def factor (x : ℚ) : ℚ :=
  5 * x^2 - 3 * x + 3

theorem polynomial_factor_implies_coefficients (p q : ℚ) :
  (∃ (a b : ℚ), ∀ x, polynomial p q x = factor x * (a * x^2 + b * x + 5)) →
  p = 0 ∧ q = 25/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l1635_163554


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l1635_163572

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) :
  (∀ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 7 → ⌊b - a⌋ ≤ 2) ∧
  (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 7 ∧ ⌊b - a⌋ = 2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l1635_163572


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1635_163507

/-- An isosceles triangle with perimeter 53 and base 11 has equal sides of length 21 -/
theorem isosceles_triangle_side_length : 
  ∀ (x : ℝ), 
  x > 0 → -- Ensure positive side length
  x + x + 11 = 53 → -- Perimeter condition
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1635_163507


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l1635_163532

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l1635_163532


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l1635_163552

def N : Matrix (Fin 2) (Fin 2) ℝ := !![1, 4; 1, 1]

theorem matrix_equation_proof :
  N^3 - 3 • (N^2) + 4 • N = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l1635_163552


namespace NUMINAMATH_CALUDE_no_integer_solution_a_squared_minus_3b_squared_equals_8_l1635_163538

theorem no_integer_solution_a_squared_minus_3b_squared_equals_8 :
  ¬ ∃ (a b : ℤ), a^2 - 3*b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_a_squared_minus_3b_squared_equals_8_l1635_163538


namespace NUMINAMATH_CALUDE_maryann_rescue_l1635_163535

/-- The number of friends Maryann needs to rescue -/
def rescue_problem (cheap_time expensive_time total_time : ℕ) : Prop :=
  let time_per_friend := cheap_time + expensive_time
  ∃ (num_friends : ℕ), num_friends * time_per_friend = total_time

theorem maryann_rescue :
  rescue_problem 6 8 42 → ∃ (num_friends : ℕ), num_friends = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_maryann_rescue_l1635_163535


namespace NUMINAMATH_CALUDE_safari_animal_ratio_l1635_163545

theorem safari_animal_ratio :
  let antelopes : ℕ := 80
  let rabbits : ℕ := antelopes + 34
  let hyenas : ℕ := antelopes + rabbits - 42
  let wild_dogs : ℕ := hyenas + 50
  let total_animals : ℕ := 605
  let leopards : ℕ := total_animals - (antelopes + rabbits + hyenas + wild_dogs)
  leopards * 2 = rabbits :=
by sorry

end NUMINAMATH_CALUDE_safari_animal_ratio_l1635_163545


namespace NUMINAMATH_CALUDE_train_speed_l1635_163574

/-- The speed of a train given its length, time to pass a man, and the man's speed in the opposite direction -/
theorem train_speed (train_length : Real) (passing_time : Real) (man_speed : Real) :
  train_length = 240 ∧ 
  passing_time = 13.090909090909092 ∧ 
  man_speed = 6 →
  (train_length / 1000) / (passing_time / 3600) - man_speed = 60 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1635_163574


namespace NUMINAMATH_CALUDE_quadratic_sum_l1635_163578

/-- A quadratic function f(x) = ax^2 + bx + c with vertex (3, -2) and f(0) = 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 3)^2 - 2) →  -- vertex form
  QuadraticFunction a b c 0 = 0 →                         -- passes through (0, 0)
  a + b + c = -10/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1635_163578


namespace NUMINAMATH_CALUDE_intersection_false_necessary_not_sufficient_for_union_false_l1635_163533

theorem intersection_false_necessary_not_sufficient_for_union_false (P Q : Prop) :
  (¬(P ∨ Q) → ¬(P ∧ Q)) ∧ (∃ (P Q : Prop), ¬(P ∧ Q) ∧ (P ∨ Q)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_false_necessary_not_sufficient_for_union_false_l1635_163533


namespace NUMINAMATH_CALUDE_can_measure_fifteen_minutes_l1635_163571

/-- Represents an hourglass with a specific duration. -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of measuring time with two hourglasses. -/
structure MeasurementState where
  time : ℕ
  hg7 : ℕ
  hg11 : ℕ

/-- Defines a single step in the measurement process. -/
inductive MeasurementStep
  | FlipHg7
  | FlipHg11
  | Wait

/-- Applies a measurement step to the current state. -/
def applyStep (state : MeasurementState) (step : MeasurementStep) : MeasurementState :=
  sorry

/-- Checks if the given sequence of steps results in exactly 15 minutes. -/
def measuresFifteenMinutes (steps : List MeasurementStep) : Prop :=
  sorry

/-- Theorem stating that it's possible to measure 15 minutes with 7 and 11-minute hourglasses. -/
theorem can_measure_fifteen_minutes :
  ∃ (steps : List MeasurementStep), measuresFifteenMinutes steps :=
  sorry

end NUMINAMATH_CALUDE_can_measure_fifteen_minutes_l1635_163571


namespace NUMINAMATH_CALUDE_associate_professor_items_l1635_163590

def CommitteeMeeting (associate_count : ℕ) (assistant_count : ℕ) 
  (total_pencils : ℕ) (total_charts : ℕ) : Prop :=
  associate_count + assistant_count = 9 ∧
  assistant_count = 11 ∧
  2 * assistant_count = 16 ∧
  total_pencils = 11 ∧
  total_charts = 16

theorem associate_professor_items :
  ∃! (associate_count : ℕ), CommitteeMeeting associate_count (9 - associate_count) 11 16 ∧
  associate_count = 1 ∧
  11 = 9 - associate_count ∧
  16 = 2 * (9 - associate_count) :=
sorry

end NUMINAMATH_CALUDE_associate_professor_items_l1635_163590


namespace NUMINAMATH_CALUDE_larans_weekly_profit_l1635_163575

/-- Represents Laran's poster business --/
structure PosterBusiness where
  total_posters_per_day : ℕ
  large_posters_per_day : ℕ
  large_poster_price : ℕ
  large_poster_cost : ℕ
  small_poster_price : ℕ
  small_poster_cost : ℕ

/-- Calculates the weekly profit for the poster business --/
def weekly_profit (business : PosterBusiness) : ℕ :=
  let small_posters_per_day := business.total_posters_per_day - business.large_posters_per_day
  let large_poster_profit := business.large_poster_price - business.large_poster_cost
  let small_poster_profit := business.small_poster_price - business.small_poster_cost
  let daily_profit := business.large_posters_per_day * large_poster_profit + small_posters_per_day * small_poster_profit
  5 * daily_profit

/-- Laran's poster business setup --/
def larans_business : PosterBusiness :=
  { total_posters_per_day := 5
  , large_posters_per_day := 2
  , large_poster_price := 10
  , large_poster_cost := 5
  , small_poster_price := 6
  , small_poster_cost := 3 }

/-- Theorem stating that Laran's weekly profit is $95 --/
theorem larans_weekly_profit :
  weekly_profit larans_business = 95 := by
  sorry


end NUMINAMATH_CALUDE_larans_weekly_profit_l1635_163575


namespace NUMINAMATH_CALUDE_ribbon_length_for_circular_sign_l1635_163580

/-- Given a circular region with area 616 square inches, using π ≈ 22/7,
    and adding 10% extra to the circumference, prove that the amount of
    ribbon needed (rounded up to the nearest inch) is 97 inches. -/
theorem ribbon_length_for_circular_sign :
  let area : ℝ := 616
  let π_approx : ℝ := 22 / 7
  let radius : ℝ := Real.sqrt (area / π_approx)
  let circumference : ℝ := 2 * π_approx * radius
  let extra_ribbon : ℝ := 0.1 * circumference
  let total_ribbon : ℝ := circumference + extra_ribbon
  ⌈total_ribbon⌉ = 97 := by
sorry

end NUMINAMATH_CALUDE_ribbon_length_for_circular_sign_l1635_163580


namespace NUMINAMATH_CALUDE_all_dihedral_angles_equal_all_polyhedral_angles_equal_l1635_163544

/-- A nearly regular polyhedron -/
structure NearlyRegularPolyhedron where
  /-- The polyhedron has a high degree of symmetry -/
  high_symmetry : Prop
  /-- Each face is a regular polygon -/
  regular_faces : Prop
  /-- Faces are arranged symmetrically around each vertex -/
  symmetric_face_arrangement : Prop
  /-- The polyhedron has vertex-transitivity property -/
  vertex_transitivity : Prop

/-- Dihedral angle of a polyhedron -/
def dihedral_angle (P : NearlyRegularPolyhedron) : Type := sorry

/-- Polyhedral angle of a polyhedron -/
def polyhedral_angle (P : NearlyRegularPolyhedron) : Type := sorry

/-- Theorem stating that all dihedral angles of a nearly regular polyhedron are equal -/
theorem all_dihedral_angles_equal (P : NearlyRegularPolyhedron) :
  ∀ a b : dihedral_angle P, a = b :=
sorry

/-- Theorem stating that all polyhedral angles of a nearly regular polyhedron are equal -/
theorem all_polyhedral_angles_equal (P : NearlyRegularPolyhedron) :
  ∀ a b : polyhedral_angle P, a = b :=
sorry

end NUMINAMATH_CALUDE_all_dihedral_angles_equal_all_polyhedral_angles_equal_l1635_163544


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l1635_163567

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → 
  ∃ k : ℕ, x = 4 * k → 
  x^2 < 500 → 
  ∀ y : ℕ, (y > 0 ∧ ∃ m : ℕ, y = 4 * m ∧ y^2 < 500) → y ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l1635_163567


namespace NUMINAMATH_CALUDE_employee_device_distribution_l1635_163511

theorem employee_device_distribution (E : ℝ) (E_pos : E > 0) : 
  let cell_phone := (2/3 : ℝ) * E
  let pager := (2/5 : ℝ) * E
  let both := (0.4 : ℝ) * E
  let neither := E - (cell_phone + pager - both)
  neither = (1/3 : ℝ) * E := by
sorry

end NUMINAMATH_CALUDE_employee_device_distribution_l1635_163511


namespace NUMINAMATH_CALUDE_infection_model_properties_l1635_163543

/-- Represents the infection spread model -/
structure InfectionModel where
  initialInfected : ℕ := 1
  totalAfterTwoRounds : ℕ := 64
  averageInfectionRate : ℕ
  thirdRoundInfections : ℕ

/-- Theorem stating the properties of the infection model -/
theorem infection_model_properties (model : InfectionModel) :
  model.initialInfected = 1 ∧
  model.totalAfterTwoRounds = 64 →
  model.averageInfectionRate = 7 ∧
  model.thirdRoundInfections = 448 := by
  sorry

#check infection_model_properties

end NUMINAMATH_CALUDE_infection_model_properties_l1635_163543


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1635_163523

theorem arithmetic_sequence_problem (a₁ a₂ a₃ : ℚ) (x : ℚ) 
  (h1 : a₁ = 1/3)
  (h2 : a₂ = 2*x)
  (h3 : a₃ = x + 4)
  (h_arithmetic : a₃ - a₂ = a₂ - a₁) :
  x = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1635_163523


namespace NUMINAMATH_CALUDE_max_b_value_l1635_163513

theorem max_b_value (a b c : ℕ) : 
  a * b * c = 240 →
  1 < c →
  c < b →
  b < a →
  b ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l1635_163513


namespace NUMINAMATH_CALUDE_nikanor_lost_second_match_l1635_163585

/-- Represents a player in the knock-out table tennis game -/
inductive Player : Type
| Nikanor : Player
| Philemon : Player
| Agathon : Player

/-- Represents the state of the game after each match -/
structure GameState :=
  (matches_played : Nat)
  (nikanor_matches : Nat)
  (philemon_matches : Nat)
  (agathon_matches : Nat)
  (last_loser : Player)

/-- The rules of the knock-out table tennis game -/
def game_rules (state : GameState) : Prop :=
  state.matches_played = (state.nikanor_matches + state.philemon_matches + state.agathon_matches) / 2 ∧
  state.nikanor_matches + state.philemon_matches + state.agathon_matches = state.matches_played * 2 ∧
  state.nikanor_matches ≤ state.matches_played ∧
  state.philemon_matches ≤ state.matches_played ∧
  state.agathon_matches ≤ state.matches_played

/-- The final state of the game -/
def final_state : GameState :=
  { matches_played := 21
  , nikanor_matches := 10
  , philemon_matches := 15
  , agathon_matches := 17
  , last_loser := Player.Nikanor }

/-- Theorem stating that Nikanor lost the second match -/
theorem nikanor_lost_second_match :
  game_rules final_state →
  final_state.last_loser = Player.Nikanor :=
by sorry

end NUMINAMATH_CALUDE_nikanor_lost_second_match_l1635_163585


namespace NUMINAMATH_CALUDE_pattern_proof_l1635_163586

theorem pattern_proof (a : ℕ) : 4 * a * (a + 1) + 1 = (2 * a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_pattern_proof_l1635_163586


namespace NUMINAMATH_CALUDE_carla_food_bank_theorem_l1635_163584

/-- Represents the food bank scenario with Carla --/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day1_restock : ℕ
  day2_people : ℕ
  day2_cans_per_person : ℕ
  day2_restock : ℕ

/-- Calculates the total number of cans given away --/
def total_cans_given_away (fb : FoodBank) : ℕ :=
  fb.day1_people * fb.day1_cans_per_person + fb.day2_people * fb.day2_cans_per_person

/-- Theorem stating that the total cans given away is 2500 --/
theorem carla_food_bank_theorem (fb : FoodBank) 
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day1_restock = 1500)
  (h5 : fb.day2_people = 1000)
  (h6 : fb.day2_cans_per_person = 2)
  (h7 : fb.day2_restock = 3000) :
  total_cans_given_away fb = 2500 := by
  sorry

end NUMINAMATH_CALUDE_carla_food_bank_theorem_l1635_163584


namespace NUMINAMATH_CALUDE_simplify_fraction_l1635_163524

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1635_163524


namespace NUMINAMATH_CALUDE_square_sum_product_l1635_163530

theorem square_sum_product (x y : ℝ) (hx : x = Real.sqrt 5 + Real.sqrt 3) (hy : y = Real.sqrt 5 - Real.sqrt 3) :
  x^2 + x*y + y^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l1635_163530


namespace NUMINAMATH_CALUDE_school_sections_l1635_163556

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) : 
  (boys / (Nat.gcd boys girls)) + (girls / (Nat.gcd boys girls)) = 29 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_l1635_163556


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1635_163581

/-- Configuration of semicircles and inscribed circle -/
structure SemicircleConfig where
  R : ℝ  -- Radius of larger semicircle
  r : ℝ  -- Radius of smaller semicircle
  x : ℝ  -- Radius of inscribed circle

/-- The inscribed circle is tangent to both semicircles and the diameter -/
def is_tangent (config : SemicircleConfig) : Prop :=
  ∃ (O O₁ O₂ : ℝ × ℝ),
    let d := config.R - config.x
    let h := Real.sqrt (d^2 - config.x^2)
    (config.R + config.r)^2 = d^2 + (config.r + config.x)^2 ∧
    h^2 + config.x^2 = config.R^2 ∧
    h^2 + (config.r + config.x)^2 = (config.R + config.r)^2

theorem inscribed_circle_radius
  (config : SemicircleConfig)
  (h₁ : config.R = 18)
  (h₂ : config.r = 9)
  (h₃ : is_tangent config) :
  config.x = 8 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1635_163581


namespace NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l1635_163549

/-- The area of a triangle with base 8 cm and height 10 cm is 40 square centimeters. -/
theorem triangle_area_with_given_base_and_height :
  let base : ℝ := 8
  let height : ℝ := 10
  let area : ℝ := (1 / 2) * base * height
  area = 40 := by sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l1635_163549


namespace NUMINAMATH_CALUDE_dove_flag_dimensions_l1635_163504

/-- Represents the shape of a dove on a square grid -/
structure DoveShape where
  area : ℝ
  perimeter_type : List String
  grid_type : String

/-- Represents the dimensions of a rectangular flag -/
structure FlagDimensions where
  length : ℝ
  height : ℝ

/-- Theorem: Given a dove shape with area 192 cm² on a square grid, 
    the flag dimensions are 24 cm × 16 cm -/
theorem dove_flag_dimensions 
  (dove : DoveShape) 
  (h1 : dove.area = 192) 
  (h2 : dove.perimeter_type = ["quarter-circle", "straight line"])
  (h3 : dove.grid_type = "square") :
  ∃ (flag : FlagDimensions), flag.length = 24 ∧ flag.height = 16 :=
by sorry

end NUMINAMATH_CALUDE_dove_flag_dimensions_l1635_163504


namespace NUMINAMATH_CALUDE_pan_division_l1635_163506

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan of chocolate cake -/
def pan : Dimensions := ⟨24, 30⟩

/-- Represents a piece of the chocolate cake -/
def piece : Dimensions := ⟨3, 2⟩

/-- Theorem stating that the pan can be divided into exactly 120 pieces -/
theorem pan_division :
  (area pan) / (area piece) = 120 := by sorry

end NUMINAMATH_CALUDE_pan_division_l1635_163506


namespace NUMINAMATH_CALUDE_julio_earnings_l1635_163557

/-- Julio's earnings calculation --/
theorem julio_earnings (commission_per_customer : ℕ) (first_week_customers : ℕ) 
  (salary : ℕ) (bonus : ℕ) : 
  commission_per_customer = 1 →
  first_week_customers = 35 →
  salary = 500 →
  bonus = 50 →
  (commission_per_customer * (first_week_customers + 2 * first_week_customers + 3 * first_week_customers) + 
   salary + bonus) = 760 := by
  sorry

#check julio_earnings

end NUMINAMATH_CALUDE_julio_earnings_l1635_163557


namespace NUMINAMATH_CALUDE_internal_resistance_of_current_source_l1635_163573

/-- Given an electric circuit with resistors R₁ and R₂, and a current source
    with internal resistance r, prove that r = 30 Ω when R₁ = 10 Ω, R₂ = 30 Ω,
    and the current ratio I₂/I₁ = 1.5 when the polarity is reversed. -/
theorem internal_resistance_of_current_source
  (R₁ R₂ r : ℝ)
  (h₁ : R₁ = 10)
  (h₂ : R₂ = 30)
  (h₃ : (R₁ + r) / (R₂ + r) = 1.5) :
  r = 30 := by
  sorry

#check internal_resistance_of_current_source

end NUMINAMATH_CALUDE_internal_resistance_of_current_source_l1635_163573


namespace NUMINAMATH_CALUDE_balls_removed_l1635_163546

def initial_balls : ℕ := 8
def current_balls : ℕ := 6

theorem balls_removed : initial_balls - current_balls = 2 := by
  sorry

end NUMINAMATH_CALUDE_balls_removed_l1635_163546


namespace NUMINAMATH_CALUDE_museum_tickets_l1635_163534

/-- Calculates the maximum number of tickets that can be purchased given a regular price, 
    discount price, discount threshold, and budget. -/
def maxTickets (regularPrice discountPrice discountThreshold budget : ℕ) : ℕ :=
  let fullPriceTickets := min discountThreshold (budget / regularPrice)
  let remainingBudget := budget - fullPriceTickets * regularPrice
  let discountTickets := remainingBudget / discountPrice
  fullPriceTickets + discountTickets

/-- Theorem stating that given the specific conditions of the problem, 
    the maximum number of tickets that can be purchased is 15. -/
theorem museum_tickets : maxTickets 11 8 10 150 = 15 := by
  sorry

end NUMINAMATH_CALUDE_museum_tickets_l1635_163534


namespace NUMINAMATH_CALUDE_first_plane_speed_calculation_l1635_163591

/-- The speed of the first plane in kilometers per hour -/
def first_plane_speed : ℝ := 110

/-- The speed of the second plane in kilometers per hour -/
def second_plane_speed : ℝ := 90

/-- The time taken for the planes to be 800 km apart in hours -/
def time : ℝ := 4.84848484848

/-- The distance between the planes after the given time in kilometers -/
def distance : ℝ := 800

theorem first_plane_speed_calculation :
  (first_plane_speed + second_plane_speed) * time = distance := by
  sorry

end NUMINAMATH_CALUDE_first_plane_speed_calculation_l1635_163591
