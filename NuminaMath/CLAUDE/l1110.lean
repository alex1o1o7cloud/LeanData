import Mathlib

namespace NUMINAMATH_CALUDE_hua_method_uses_golden_ratio_l1110_111080

/-- Represents the mathematical concepts that could be used in optimization methods -/
inductive OptimizationConcept
  | GoldenRatio
  | Mean
  | Mode
  | Median

/-- Represents Hua Luogeng's optimal selection method -/
def HuaOptimalSelectionMethod : Type := OptimizationConcept

/-- The concept used in Hua Luogeng's optimal selection method -/
def concept_used : HuaOptimalSelectionMethod := OptimizationConcept.GoldenRatio

/-- Theorem stating that the concept used in Hua Luogeng's optimal selection method is the golden ratio -/
theorem hua_method_uses_golden_ratio :
  concept_used = OptimizationConcept.GoldenRatio :=
by sorry

end NUMINAMATH_CALUDE_hua_method_uses_golden_ratio_l1110_111080


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_4_sqrt_5_l1110_111066

theorem sqrt_sum_equals_4_sqrt_5 : 
  Real.sqrt (24 - 8 * Real.sqrt 2) + Real.sqrt (24 + 8 * Real.sqrt 2) = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_4_sqrt_5_l1110_111066


namespace NUMINAMATH_CALUDE_wily_person_exists_l1110_111037

inductive PersonType
  | Knight
  | Liar
  | Wily

structure Person where
  type : PersonType
  statement : Prop

def is_truthful (p : Person) : Prop :=
  match p.type with
  | PersonType.Knight => p.statement
  | PersonType.Liar => ¬p.statement
  | PersonType.Wily => True

theorem wily_person_exists (people : Fin 3 → Person)
  (h1 : (people 0).statement = ∃ i, (people i).type = PersonType.Liar)
  (h2 : (people 1).statement = ∀ i j, i ≠ j → ((people i).type = PersonType.Liar ∨ (people j).type = PersonType.Liar))
  (h3 : (people 2).statement = ∀ i, (people i).type = PersonType.Liar)
  : ∃ i, (people i).type = PersonType.Wily :=
by
  sorry

end NUMINAMATH_CALUDE_wily_person_exists_l1110_111037


namespace NUMINAMATH_CALUDE_roots_less_than_one_l1110_111098

theorem roots_less_than_one (a b : ℝ) 
  (h1 : |a| + |b| < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x : ℝ, x^2 + a*x + b = 0 → |x| < 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_less_than_one_l1110_111098


namespace NUMINAMATH_CALUDE_distance_between_ports_l1110_111072

/-- The distance between ports A and B in kilometers -/
def distance_AB : ℝ := 40

/-- The speed of the ship in still water in km/h -/
def ship_speed : ℝ := 26

/-- The speed of the river current in km/h -/
def current_speed : ℝ := 6

/-- The number of round trips made by the ship -/
def round_trips : ℕ := 4

/-- The total time taken for all round trips in hours -/
def total_time : ℝ := 13

theorem distance_between_ports :
  let downstream_speed := ship_speed + current_speed
  let upstream_speed := ship_speed - current_speed
  let time_per_round_trip := total_time / round_trips
  let downstream_time := (upstream_speed * time_per_round_trip) / (downstream_speed + upstream_speed)
  distance_AB = downstream_speed * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_distance_between_ports_l1110_111072


namespace NUMINAMATH_CALUDE_intersection_area_is_400_l1110_111008

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  P : Point3D
  Q : Point3D
  R : Point3D

def cube : Cube := {
  A := { x := 0, y := 0, z := 0 },
  B := { x := 20, y := 0, z := 0 },
  C := { x := 20, y := 0, z := 20 },
  D := { x := 20, y := 20, z := 20 }
}

def plane : Plane := {
  P := { x := 3, y := 0, z := 0 },
  Q := { x := 20, y := 0, z := 8 },
  R := { x := 20, y := 12, z := 20 }
}

/-- Calculate the area of the polygon formed by the intersection of the plane and the cube -/
def intersectionArea (c : Cube) (p : Plane) : ℝ :=
  sorry

theorem intersection_area_is_400 : intersectionArea cube plane = 400 := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_is_400_l1110_111008


namespace NUMINAMATH_CALUDE_bill_left_with_411_l1110_111054

/-- Calculates the amount of money Bill is left with after all transactions and expenses -/
def billsRemainingMoney : ℝ :=
  let merchantA_sale := 8 * 9
  let merchantB_sale := 15 * 11
  let sheriff_fine := 80
  let merchantC_sale := 25 * 8
  let protection_cost := 30
  let passerby_sale := 12 * 7
  
  let total_earnings := merchantA_sale + merchantB_sale + merchantC_sale + passerby_sale
  let total_expenses := sheriff_fine + protection_cost
  
  total_earnings - total_expenses

/-- Theorem stating that Bill is left with $411 after all transactions and expenses -/
theorem bill_left_with_411 : billsRemainingMoney = 411 := by
  sorry

end NUMINAMATH_CALUDE_bill_left_with_411_l1110_111054


namespace NUMINAMATH_CALUDE_custom_mul_five_three_l1110_111025

/-- Custom multiplication operation -/
def custom_mul (a b : ℤ) : ℤ := a^2 - a*b + b^2

/-- Theorem stating that 5*3 = 19 under the custom multiplication -/
theorem custom_mul_five_three : custom_mul 5 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_five_three_l1110_111025


namespace NUMINAMATH_CALUDE_quadratic_function_comparison_l1110_111004

/-- Proves that for points A(x₁, y₁) and B(x₂, y₂) on the graph of y = (x - 1)² + 1, 
    if x₁ > x₂ > 1, then y₁ > y₂. -/
theorem quadratic_function_comparison (x₁ x₂ y₁ y₂ : ℝ) 
    (h1 : y₁ = (x₁ - 1)^2 + 1)
    (h2 : y₂ = (x₂ - 1)^2 + 1)
    (h3 : x₁ > x₂)
    (h4 : x₂ > 1) : 
  y₁ > y₂ := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_comparison_l1110_111004


namespace NUMINAMATH_CALUDE_special_line_properties_l1110_111060

/-- A line passing through (2,3) with x-intercept twice the y-intercept -/
def special_line (x y : ℝ) : Prop := x + 2*y - 8 = 0

theorem special_line_properties :
  (special_line 2 3) ∧ 
  (∃ (a : ℝ), a ≠ 0 ∧ special_line (2*a) 0 ∧ special_line 0 a) :=
by sorry

end NUMINAMATH_CALUDE_special_line_properties_l1110_111060


namespace NUMINAMATH_CALUDE_jester_count_l1110_111006

theorem jester_count (total_legs total_heads : ℕ) 
  (jester_legs jester_heads elephant_legs elephant_heads : ℕ) : 
  total_legs = 50 → 
  total_heads = 18 → 
  jester_legs = 3 → 
  jester_heads = 1 → 
  elephant_legs = 4 → 
  elephant_heads = 1 → 
  ∃ (num_jesters num_elephants : ℕ), 
    num_jesters * jester_legs + num_elephants * elephant_legs = total_legs ∧
    num_jesters * jester_heads + num_elephants * elephant_heads = total_heads ∧
    num_jesters = 22 :=
by sorry

end NUMINAMATH_CALUDE_jester_count_l1110_111006


namespace NUMINAMATH_CALUDE_road_length_for_given_conditions_l1110_111018

/-- Calculates the length of a road given the number of trees, space between trees, and space occupied by each tree. -/
def road_length (num_trees : ℕ) (space_between : ℕ) (tree_space : ℕ) : ℕ :=
  (num_trees * tree_space) + ((num_trees - 1) * space_between)

/-- Theorem stating that for 11 trees, with 14 feet between each tree, and each tree taking 1 foot of space, the road length is 151 feet. -/
theorem road_length_for_given_conditions :
  road_length 11 14 1 = 151 := by
  sorry

end NUMINAMATH_CALUDE_road_length_for_given_conditions_l1110_111018


namespace NUMINAMATH_CALUDE_incorrect_statement_l1110_111034

theorem incorrect_statement : ¬(
  (∀ x : ℝ, x ∈ [0, 1] → Real.exp x ≥ 1) ∧
  (∃ x : ℝ, x^2 + x + 1 < 0)
) := by sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1110_111034


namespace NUMINAMATH_CALUDE_angle_measure_in_pentagon_l1110_111096

structure Pentagon where
  F : ℝ
  G : ℝ
  H : ℝ
  I : ℝ
  J : ℝ

def is_convex_pentagon (p : Pentagon) : Prop :=
  p.F + p.G + p.H + p.I + p.J = 540

theorem angle_measure_in_pentagon (p : Pentagon) 
  (convex : is_convex_pentagon p)
  (fgh_congruent : p.F = p.G ∧ p.G = p.H)
  (ij_congruent : p.I = p.J)
  (f_less_than_i : p.F + 80 = p.I) :
  p.I = 156 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_pentagon_l1110_111096


namespace NUMINAMATH_CALUDE_sum_of_max_min_f_l1110_111049

/-- Given a > 0, prove that the sum of the maximum and minimum values of the function
f(x) = (2009^(x+1) + 2007) / (2009^x + 1) + sin x on the interval [-a, a] is equal to 4016. -/
theorem sum_of_max_min_f (a : ℝ) (h : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ (2009^(x+1) + 2007) / (2009^x + 1) + Real.sin x
  (⨆ x ∈ Set.Icc (-a) a, f x) + (⨅ x ∈ Set.Icc (-a) a, f x) = 4016 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_f_l1110_111049


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_solution_set_all_reals_implies_a_range_l1110_111015

-- Part 1
theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, ax^2 - 2*a*x + 3 > 0 ↔ -1 < x ∧ x < 3) →
  a = -1 :=
sorry

-- Part 2
theorem solution_set_all_reals_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ax^2 - 2*a*x + 3 > 0) →
  0 ≤ a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_solution_set_all_reals_implies_a_range_l1110_111015


namespace NUMINAMATH_CALUDE_largest_unreachable_score_l1110_111057

/-- 
Given that:
1. Easy questions earn 3 points.
2. Harder questions earn 7 points.
3. Scores are achieved by combinations of these point values.

Prove that 11 is the largest integer that cannot be expressed as a linear combination of 3 and 7 
with non-negative integer coefficients.
-/
theorem largest_unreachable_score : 
  ∀ n : ℕ, n > 11 → ∃ x y : ℕ, n = 3 * x + 7 * y :=
by sorry

end NUMINAMATH_CALUDE_largest_unreachable_score_l1110_111057


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_exists_185_is_greatest_l1110_111033

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 200 ∧ Nat.gcd n 30 = 5 → n ≤ 185 :=
by
  sorry

theorem exists_185 : 185 < 200 ∧ Nat.gcd 185 30 = 5 :=
by
  sorry

theorem is_greatest : ∀ m : ℕ, m < 200 ∧ Nat.gcd m 30 = 5 → m ≤ 185 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_exists_185_is_greatest_l1110_111033


namespace NUMINAMATH_CALUDE_min_value_of_xy_l1110_111044

theorem min_value_of_xy (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h_geom : (Real.log x) * (Real.log y) = 1/4) : 
  ∀ z, x * y ≥ z → z ≤ Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_xy_l1110_111044


namespace NUMINAMATH_CALUDE_solution_count_theorem_l1110_111059

/-- The number of solutions to the equation 2x + 3y + z + x^2 = n for positive integers x, y, z -/
def num_solutions (n : ℕ+) : ℕ := 
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    2 * t.1 + 3 * t.2.1 + t.2.2 + t.1 * t.1 = n.val ∧ 
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0) (Finset.product (Finset.range n.val) (Finset.product (Finset.range n.val) (Finset.range n.val)))).card

theorem solution_count_theorem (n : ℕ+) : 
  num_solutions n = 25 → n = 32 ∨ n = 33 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_theorem_l1110_111059


namespace NUMINAMATH_CALUDE_equation_proof_l1110_111027

theorem equation_proof : 300 * 2 + (12 + 4) * 1 / 8 = 602 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1110_111027


namespace NUMINAMATH_CALUDE_inequality_holds_l1110_111091

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1110_111091


namespace NUMINAMATH_CALUDE_apple_lovers_l1110_111079

structure FruitPreferences where
  total : ℕ
  apple : ℕ
  orange : ℕ
  mango : ℕ
  banana : ℕ
  grapes : ℕ
  orange_mango_not_apple : ℕ
  mango_apple_not_orange : ℕ
  all_three : ℕ
  banana_grapes_only : ℕ
  apple_banana_grapes_not_others : ℕ

def room : FruitPreferences := {
  total := 60,
  apple := 40,
  orange := 17,
  mango := 23,
  banana := 12,
  grapes := 9,
  orange_mango_not_apple := 7,
  mango_apple_not_orange := 10,
  all_three := 4,
  banana_grapes_only := 6,
  apple_banana_grapes_not_others := 3
}

theorem apple_lovers (pref : FruitPreferences) : pref.apple = 40 :=
  sorry

end NUMINAMATH_CALUDE_apple_lovers_l1110_111079


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1110_111073

theorem absolute_value_inequality (x : ℝ) : 
  |x + 1| > 3 ↔ x ∈ Set.Iio (-4) ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1110_111073


namespace NUMINAMATH_CALUDE_min_value_of_function_l1110_111043

theorem min_value_of_function (x : ℝ) (h : x > 0) : 4 * x + 1 / x^2 ≥ 5 ∧ ∃ y > 0, 4 * y + 1 / y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1110_111043


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_product_12_and_smallest_sum_l1110_111003

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem greatest_two_digit_with_product_12_and_smallest_sum :
  ∃ (n : ℕ), is_two_digit n ∧ 
             digit_product n = 12 ∧
             (∀ m : ℕ, is_two_digit m → digit_product m = 12 → digit_sum m ≥ digit_sum n) ∧
             (∀ k : ℕ, is_two_digit k → digit_product k = 12 → digit_sum k = digit_sum n → k ≤ n) ∧
             n = 43 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_product_12_and_smallest_sum_l1110_111003


namespace NUMINAMATH_CALUDE_emily_cleaning_time_l1110_111000

/-- Represents the cleaning time distribution among four people -/
structure CleaningTime where
  total : ℝ
  lillyAndFiona : ℝ
  jack : ℝ
  emily : ℝ

/-- Theorem stating Emily's cleaning time in minutes -/
theorem emily_cleaning_time (ct : CleaningTime) : 
  ct.total = 8 ∧ 
  ct.lillyAndFiona = 1/4 * ct.total ∧ 
  ct.jack = 1/3 * ct.total ∧ 
  ct.emily = ct.total - ct.lillyAndFiona - ct.jack → 
  ct.emily * 60 = 200 := by
  sorry

#check emily_cleaning_time

end NUMINAMATH_CALUDE_emily_cleaning_time_l1110_111000


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l1110_111092

theorem gain_percent_calculation (C S : ℝ) (h : C > 0) :
  50 * C = 20 * S → (S - C) / C * 100 = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l1110_111092


namespace NUMINAMATH_CALUDE_circle_intersection_chord_length_l1110_111062

/-- A circle in the xy-plane -/
structure Circle where
  a : ℝ
  equation : ℝ → ℝ → Prop :=
    fun x y ↦ x^2 + y^2 + 2*x - 2*y + a = 0

/-- A line in the xy-plane -/
def Line : ℝ → ℝ → Prop :=
  fun x y ↦ x + y + 2 = 0

/-- The length of a chord formed by the intersection of a circle and a line -/
def ChordLength (c : Circle) : ℝ :=
  4 -- Given in the problem

/-- The main theorem -/
theorem circle_intersection_chord_length (c : Circle) :
  (∀ x y, Line x y → c.equation x y) →
  ChordLength c = 4 →
  c.a = -4 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_chord_length_l1110_111062


namespace NUMINAMATH_CALUDE_system_solution_l1110_111094

theorem system_solution (a : ℚ) :
  (∃! x y : ℚ, 2*x + 3*y = 5 ∧ x - y = 2 ∧ x + 4*y = a) ↔ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1110_111094


namespace NUMINAMATH_CALUDE_winter_sales_is_seven_million_l1110_111097

/-- The number of pizzas sold in millions for each season --/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The percentage of pizzas sold in fall --/
def fall_percentage : ℝ := 0.20

/-- The given seasonal sales data --/
def given_sales : SeasonalSales where
  spring := 6
  summer := 7
  fall := fall_percentage * (6 + 7 + fall_percentage * (6 + 7 + 5 + 7) + 7)
  winter := 7

/-- Theorem stating that the winter sales is 7 million pizzas --/
theorem winter_sales_is_seven_million (s : SeasonalSales) :
  s.spring = 6 →
  s.summer = 7 →
  s.fall = fall_percentage * (s.spring + s.summer + s.fall + s.winter) →
  s.winter = 7 := by
  sorry

#eval given_sales.winter

end NUMINAMATH_CALUDE_winter_sales_is_seven_million_l1110_111097


namespace NUMINAMATH_CALUDE_honey_jar_theorem_l1110_111063

def initial_honey : ℝ := 1.2499999999999998
def draw_percentage : ℝ := 0.20
def num_iterations : ℕ := 4

def honey_left (initial : ℝ) (draw : ℝ) (iterations : ℕ) : ℝ :=
  initial * (1 - draw) ^ iterations

theorem honey_jar_theorem :
  honey_left initial_honey draw_percentage num_iterations = 0.512 := by
  sorry

end NUMINAMATH_CALUDE_honey_jar_theorem_l1110_111063


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1110_111084

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - m - 2) (m + 1)
  (z.re = 0 ∧ z.im ≠ 0) → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1110_111084


namespace NUMINAMATH_CALUDE_lonely_island_turtles_l1110_111042

theorem lonely_island_turtles : 
  ∀ (happy_island lonely_island : ℕ),
  happy_island = 60 →
  happy_island = 2 * lonely_island + 10 →
  lonely_island = 25 := by
sorry

end NUMINAMATH_CALUDE_lonely_island_turtles_l1110_111042


namespace NUMINAMATH_CALUDE_charts_brought_is_eleven_l1110_111090

/-- The number of charts brought to a committee meeting --/
def charts_brought (associate_profs assistant_profs : ℕ) : ℕ :=
  associate_profs + 2 * assistant_profs

/-- Proof that 11 charts were brought to the meeting --/
theorem charts_brought_is_eleven :
  ∃ (associate_profs assistant_profs : ℕ),
    associate_profs + assistant_profs = 7 ∧
    2 * associate_profs + assistant_profs = 10 ∧
    charts_brought associate_profs assistant_profs = 11 :=
by
  sorry

#check charts_brought_is_eleven

end NUMINAMATH_CALUDE_charts_brought_is_eleven_l1110_111090


namespace NUMINAMATH_CALUDE_square_minus_one_divisible_by_three_l1110_111050

theorem square_minus_one_divisible_by_three (n : ℕ) (h : ¬ 3 ∣ n) : 3 ∣ (n^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_square_minus_one_divisible_by_three_l1110_111050


namespace NUMINAMATH_CALUDE_sum_of_percentages_l1110_111077

theorem sum_of_percentages (X Y Z : ℝ) : 
  X = 0.2 * 50 →
  40 = 0.2 * Y →
  40 = (Z / 100) * 50 →
  X + Y + Z = 290 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_percentages_l1110_111077


namespace NUMINAMATH_CALUDE_range_of_slope_intersecting_line_l1110_111053

/-- Given two points P and Q, and a line l that intersects the extension of PQ,
    prove the range of values for the slope of l. -/
theorem range_of_slope_intersecting_line (P Q : ℝ × ℝ) (m : ℝ) : 
  P = (-1, 1) →
  Q = (2, 2) →
  ∃ (x y : ℝ), x + m * y + m = 0 ∧ 
    (∃ (t : ℝ), x = -1 + 3 * t ∧ y = 1 + t) →
  -3 < m ∧ m < -2/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_slope_intersecting_line_l1110_111053


namespace NUMINAMATH_CALUDE_sector_area_l1110_111021

theorem sector_area (θ : Real) (s : Real) (A : Real) :
  θ = 2 ∧ s = 4 → A = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1110_111021


namespace NUMINAMATH_CALUDE_prob_four_ones_twelve_dice_l1110_111011

def n : ℕ := 12  -- total number of dice
def k : ℕ := 4   -- number of dice showing 1
def s : ℕ := 6   -- number of sides on each die

-- Probability of rolling exactly k ones out of n dice
def prob_exactly_k_ones : ℚ :=
  (Nat.choose n k : ℚ) * (1 / s) ^ k * ((s - 1) / s) ^ (n - k)

theorem prob_four_ones_twelve_dice :
  prob_exactly_k_ones = 495 * 390625 / 2176782336 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_ones_twelve_dice_l1110_111011


namespace NUMINAMATH_CALUDE_student_fail_marks_l1110_111038

theorem student_fail_marks (total_marks passing_percentage student_marks : ℕ) 
  (h1 : total_marks = 700)
  (h2 : passing_percentage = 33)
  (h3 : student_marks = 175) :
  (total_marks * passing_percentage / 100 : ℕ) - student_marks = 56 :=
by sorry

end NUMINAMATH_CALUDE_student_fail_marks_l1110_111038


namespace NUMINAMATH_CALUDE_expression_value_l1110_111039

theorem expression_value : (2.502 + 0.064)^2 - (2.502 - 0.064)^2 / (2.502 * 0.064) = 4.002 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1110_111039


namespace NUMINAMATH_CALUDE_quadratic_three_axis_intersections_l1110_111082

theorem quadratic_three_axis_intersections (k : ℝ) :
  (∃ x₁ x₂ y : ℝ, x₁ ≠ x₂ ∧ 
    (k * x₁^2 - 4 * x₁ - 3 = 0) ∧ 
    (k * x₂^2 - 4 * x₂ - 3 = 0) ∧ 
    (k * 0^2 - 4 * 0 - 3 = y)) ↔ 
  (k > -4/3 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_three_axis_intersections_l1110_111082


namespace NUMINAMATH_CALUDE_work_completion_time_l1110_111081

-- Define the work rates
def work_rate_a_and_b : ℚ := 1 / 6
def work_rate_a : ℚ := 1 / 11
def work_rate_c : ℚ := 1 / 13

-- Define the theorem
theorem work_completion_time :
  let work_rate_b : ℚ := work_rate_a_and_b - work_rate_a
  let work_rate_abc : ℚ := work_rate_a + work_rate_b + work_rate_c
  (1 : ℚ) / work_rate_abc = 858 / 209 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1110_111081


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l1110_111047

theorem imaginary_unit_sum : ∃ i : ℂ, i * i = -1 ∧ i + i^2 + i^3 = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l1110_111047


namespace NUMINAMATH_CALUDE_smallest_multiple_twenty_five_satisfies_smallest_x_is_25_l1110_111046

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 625 ∣ (450 * x) → x ≥ 25 := by
  sorry

theorem twenty_five_satisfies : 625 ∣ (450 * 25) := by
  sorry

theorem smallest_x_is_25 : ∃ x : ℕ, x > 0 ∧ 625 ∣ (450 * x) ∧ ∀ y : ℕ, (y > 0 ∧ 625 ∣ (450 * y)) → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_twenty_five_satisfies_smallest_x_is_25_l1110_111046


namespace NUMINAMATH_CALUDE_problem_solution_l1110_111058

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (a x : ℝ) : ℝ := 1/2 * x^2 - 4*a*x + a * log x + a + 1/2

/-- The function g(x) as defined in the problem -/
noncomputable def g (a x : ℝ) : ℝ := f a x + 2*a

/-- The derivative of g(x) -/
noncomputable def g' (a x : ℝ) : ℝ := x - 4*a + a/x

theorem problem_solution (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    g' a x₁ = 0 ∧ g' a x₂ = 0 ∧
    g a x₁ + g a x₂ ≥ g' a (x₁ * x₂)) →
  1/4 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1110_111058


namespace NUMINAMATH_CALUDE_expected_sum_of_marbles_l1110_111083

-- Define the set of marbles
def marbles : Finset ℕ := Finset.range 7

-- Define the function to calculate the sum of two marbles
def marbleSum (pair : Finset ℕ) : ℕ := Finset.sum pair id

-- Define the set of all possible pairs of marbles
def marblePairs : Finset (Finset ℕ) := marbles.powerset.filter (fun s => s.card = 2)

-- Statement of the theorem
theorem expected_sum_of_marbles :
  (Finset.sum marblePairs marbleSum) / marblePairs.card = 52 / 7 := by
sorry

end NUMINAMATH_CALUDE_expected_sum_of_marbles_l1110_111083


namespace NUMINAMATH_CALUDE_square_root_of_product_l1110_111036

theorem square_root_of_product : Real.sqrt ((90 + 6) * (90 - 6)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_product_l1110_111036


namespace NUMINAMATH_CALUDE_power_multiplication_l1110_111029

theorem power_multiplication (x : ℝ) : x^5 * x^9 = x^14 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l1110_111029


namespace NUMINAMATH_CALUDE_boundary_slopes_sum_l1110_111076

/-- Parabola P with equation y = x^2 + 4x + 4 -/
def P : ℝ → ℝ := λ x => x^2 + 4*x + 4

/-- Point Q -/
def Q : ℝ × ℝ := (10, 16)

/-- Function to determine if a line with slope m through Q intersects P -/
def intersects (m : ℝ) : Prop :=
  ∃ x : ℝ, P x = Q.2 + m * (x - Q.1)

/-- The lower boundary slope -/
noncomputable def r : ℝ := -24 - 16 * Real.sqrt 2

/-- The upper boundary slope -/
noncomputable def s : ℝ := -24 + 16 * Real.sqrt 2

/-- Theorem stating that r + s = -48 -/
theorem boundary_slopes_sum : r + s = -48 := by sorry

end NUMINAMATH_CALUDE_boundary_slopes_sum_l1110_111076


namespace NUMINAMATH_CALUDE_pascal_triangle_element_l1110_111016

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : ℕ := 31

/-- The position of the number we're looking for (1-indexed) -/
def target_position : ℕ := 25

/-- The row number in Pascal's triangle (0-indexed) -/
def row_number : ℕ := row_length - 1

/-- The column number in Pascal's triangle (0-indexed) -/
def column_number : ℕ := target_position - 1

theorem pascal_triangle_element :
  Nat.choose row_number column_number = 593775 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_element_l1110_111016


namespace NUMINAMATH_CALUDE_max_profit_price_l1110_111061

/-- Represents the sales volume as a function of unit price -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the profit as a function of unit price -/
def profit (x : ℝ) : ℝ := (x - 20) * (sales_volume x)

/-- Theorem: The unit price that maximizes profit is 35 yuan -/
theorem max_profit_price : 
  ∃ (x : ℝ), x = 35 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_price_l1110_111061


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1110_111031

noncomputable def f (x a : ℝ) : ℝ := (1 + Real.cos (2 * x)) * 1 + 1 * (Real.sqrt 3 * Real.sin (2 * x) + a)

theorem max_value_implies_a_equals_one (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = 4) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1110_111031


namespace NUMINAMATH_CALUDE_hotel_revenue_l1110_111017

theorem hotel_revenue
  (total_rooms : ℕ)
  (single_room_cost double_room_cost : ℕ)
  (single_rooms_booked : ℕ)
  (h_total : total_rooms = 260)
  (h_single_cost : single_room_cost = 35)
  (h_double_cost : double_room_cost = 60)
  (h_single_booked : single_rooms_booked = 64) :
  single_room_cost * single_rooms_booked +
  double_room_cost * (total_rooms - single_rooms_booked) = 14000 := by
sorry

end NUMINAMATH_CALUDE_hotel_revenue_l1110_111017


namespace NUMINAMATH_CALUDE_fair_distribution_theorem_l1110_111009

/-- Represents the state of the game -/
structure GameState where
  total_cards : ℕ
  a_points : ℕ
  b_points : ℕ
  win_points : ℕ
  a_win_prob : ℚ
  b_win_prob : ℚ

/-- Calculates the probability of A winning the game from the current state -/
def prob_a_wins (state : GameState) : ℚ :=
  sorry

/-- Calculates the fair number of cards for each player -/
def fair_distribution (state : GameState) : ℕ × ℕ :=
  sorry

/-- Theorem stating the fair distribution of cards -/
theorem fair_distribution_theorem (state : GameState) 
  (h1 : state.total_cards = 12)
  (h2 : state.a_points = 2)
  (h3 : state.b_points = 1)
  (h4 : state.win_points = 3)
  (h5 : state.a_win_prob = 1/2)
  (h6 : state.b_win_prob = 1/2) :
  fair_distribution state = (9, 3) :=
sorry

end NUMINAMATH_CALUDE_fair_distribution_theorem_l1110_111009


namespace NUMINAMATH_CALUDE_other_number_proof_l1110_111030

/-- Given two positive integers with specific HCF and LCM, prove that if one is 24, the other is 156 -/
theorem other_number_proof (A B : ℕ+) : 
  Nat.gcd A B = 12 →
  Nat.lcm A B = 312 →
  A = 24 →
  B = 156 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l1110_111030


namespace NUMINAMATH_CALUDE_garden_area_proof_l1110_111071

theorem garden_area_proof (total_posts : ℕ) (post_spacing : ℕ) :
  total_posts = 24 →
  post_spacing = 6 →
  ∃ (short_posts long_posts : ℕ),
    short_posts + long_posts = total_posts / 2 ∧
    long_posts = 3 * short_posts ∧
    (short_posts - 1) * post_spacing * (long_posts - 1) * post_spacing = 576 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_proof_l1110_111071


namespace NUMINAMATH_CALUDE_high_correlation_implies_r_close_to_one_l1110_111005

-- Define a type for variables
def Variable : Type := ℝ

-- Define a correlation coefficient
def correlation_coefficient (x y : Variable) : ℝ := sorry

-- Define what it means for the degree of linear correlation to be very high
def high_linear_correlation (x y : Variable) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ abs (correlation_coefficient x y) > 1 - ε

-- The theorem to prove
theorem high_correlation_implies_r_close_to_one (x y : Variable) :
  high_linear_correlation x y → ∃ (δ : ℝ), δ > 0 ∧ δ < 0.1 ∧ abs (correlation_coefficient x y) > 1 - δ :=
sorry

end NUMINAMATH_CALUDE_high_correlation_implies_r_close_to_one_l1110_111005


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1110_111026

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics -/
structure Survey where
  total_population : ℕ
  strata : List ℕ
  sample_size : ℕ

/-- Determines the appropriate sampling method for a given survey -/
def appropriate_sampling_method (s : Survey) : SamplingMethod :=
  if s.strata.length > 1 then SamplingMethod.Stratified else SamplingMethod.Random

/-- The two surveys from the problem -/
def survey1 : Survey :=
  { total_population := 500
  , strata := [125, 280, 95]
  , sample_size := 100 }

def survey2 : Survey :=
  { total_population := 12
  , strata := [12]
  , sample_size := 3 }

/-- Theorem stating the correct sampling methods for the given surveys -/
theorem correct_sampling_methods :
  (appropriate_sampling_method survey1 = SamplingMethod.Stratified) ∧
  (appropriate_sampling_method survey2 = SamplingMethod.Random) := by
  sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1110_111026


namespace NUMINAMATH_CALUDE_yoongi_has_second_largest_number_l1110_111048

/-- Represents a student with their assigned number -/
structure Student where
  name : String
  number : Nat

/-- Checks if a student has the second largest number among a list of students -/
def hasSecondLargestNumber (s : Student) (students : List Student) : Prop :=
  ∃ (larger smaller : Student),
    larger ∈ students ∧
    smaller ∈ students ∧
    s ∈ students ∧
    larger.number > s.number ∧
    s.number > smaller.number ∧
    ∀ (other : Student), other ∈ students → other.number ≤ larger.number

theorem yoongi_has_second_largest_number :
  let yoongi := Student.mk "Yoongi" 7
  let jungkook := Student.mk "Jungkook" 6
  let yuna := Student.mk "Yuna" 9
  let students := [yoongi, jungkook, yuna]
  hasSecondLargestNumber yoongi students := by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_second_largest_number_l1110_111048


namespace NUMINAMATH_CALUDE_tangent_line_and_minimum_value_l1110_111064

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem tangent_line_and_minimum_value (a : ℝ) :
  (∀ x, x > 0 → f a x = a * x^2 - (a + 2) * x + Real.log x) →
  (a = 1 → ∀ y, y = -2 ↔ y = f 1 1 ∧ (∀ h, h ≠ 0 → (f 1 (1 + h) - f 1 1) / h = 0)) ∧
  (a > 0 → (∀ x, x ∈ Set.Icc 1 (Real.exp 1) → f a x ≥ -2) ∧ 
           (∃ x, x ∈ Set.Icc 1 (Real.exp 1) ∧ f a x = -2) →
           a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_minimum_value_l1110_111064


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_implication_l1110_111051

theorem sufficient_not_necessary_implication (p q : Prop) :
  (p → q) ∧ ¬(q → p) → (¬q → ¬p) ∧ ¬(¬p → ¬q) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_implication_l1110_111051


namespace NUMINAMATH_CALUDE_garden_volume_l1110_111045

/-- Calculates the volume of a rectangular prism -/
def rectangularPrismVolume (length width height : ℝ) : ℝ :=
  length * width * height

/-- Theorem: The volume of a rectangular prism garden with dimensions 12 m, 5 m, and 3 m is 180 cubic meters -/
theorem garden_volume :
  rectangularPrismVolume 12 5 3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_garden_volume_l1110_111045


namespace NUMINAMATH_CALUDE_running_yardage_l1110_111002

/-- The star running back's total yardage -/
def total_yardage : ℕ := 150

/-- The star running back's passing yardage -/
def passing_yardage : ℕ := 60

/-- Theorem: The star running back's running yardage is 90 yards -/
theorem running_yardage : total_yardage - passing_yardage = 90 := by
  sorry

end NUMINAMATH_CALUDE_running_yardage_l1110_111002


namespace NUMINAMATH_CALUDE_quadrilateral_identity_l1110_111010

/-- A quadrilateral with sides a, b, c, d, diagonals e, f, and g being the length of the segment
    connecting the midpoints of the diagonals. -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  g : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  e_pos : 0 < e
  f_pos : 0 < f
  g_pos : 0 < g

/-- The theorem stating that for any quadrilateral, the sum of squares of its sides equals
    the sum of squares of its diagonals plus four times the square of the length of the segment
    connecting the midpoints of the diagonals. -/
theorem quadrilateral_identity (q : Quadrilateral) :
  q.a^2 + q.b^2 + q.c^2 + q.d^2 = q.e^2 + q.f^2 + 4 * q.g^2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_identity_l1110_111010


namespace NUMINAMATH_CALUDE_symmetry_implies_axis_l1110_111019

/-- A function that is symmetric about x = 2 -/
def SymmetricAboutTwo (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (4 - x)

/-- The axis of symmetry for a function -/
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

/-- If f(x) = f(4-x) for all x, then x = 2 is the axis of symmetry of f -/
theorem symmetry_implies_axis (f : ℝ → ℝ) (h : SymmetricAboutTwo f) :
    AxisOfSymmetry f 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_axis_l1110_111019


namespace NUMINAMATH_CALUDE_four_digit_with_five_or_seven_l1110_111007

theorem four_digit_with_five_or_seven (total_four_digit : Nat) (without_five_or_seven : Nat) :
  total_four_digit = 9000 →
  without_five_or_seven = 3584 →
  total_four_digit - without_five_or_seven = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_with_five_or_seven_l1110_111007


namespace NUMINAMATH_CALUDE_equal_intercepts_iff_specific_equation_not_in_second_quadrant_iff_a_leq_neg_one_l1110_111089

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

-- Define the condition for equal intercepts
def equal_intercepts (a : ℝ) : Prop := ∃ (k : ℝ), line_equation a k 0 ∧ line_equation a 0 k

-- Define the condition for not passing through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop := ∀ (x y : ℝ), line_equation a x y → (x > 0 → y ≤ 0)

-- Theorem 1: Equal intercepts condition
theorem equal_intercepts_iff_specific_equation :
  ∀ (a : ℝ), equal_intercepts a ↔ (∀ (x y : ℝ), x + y + 4 = 0 ↔ line_equation a x y) :=
sorry

-- Theorem 2: Not passing through second quadrant condition
theorem not_in_second_quadrant_iff_a_leq_neg_one :
  ∀ (a : ℝ), not_in_second_quadrant a ↔ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_equal_intercepts_iff_specific_equation_not_in_second_quadrant_iff_a_leq_neg_one_l1110_111089


namespace NUMINAMATH_CALUDE_max_perimeter_rectangle_l1110_111070

/-- Represents a rectangular enclosure -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The perimeter of the rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: Maximum perimeter of a rectangle with given constraints -/
theorem max_perimeter_rectangle : 
  ∃ (r : Rectangle), 
    area r = 8000 ∧ 
    r.width ≥ 50 ∧
    ∀ (r' : Rectangle), area r' = 8000 ∧ r'.width ≥ 50 → perimeter r' ≤ perimeter r ∧
    r.length = 100 ∧ 
    r.width = 80 ∧ 
    perimeter r = 360 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_rectangle_l1110_111070


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1110_111086

theorem quadratic_transformation (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (h k r : ℝ) (hr : r ≠ 0), ∀ x : ℝ,
    a * x^2 + b * x + c = r^2 * ((x / r - h)^2) + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1110_111086


namespace NUMINAMATH_CALUDE_pie_chart_percentage_central_angle_relation_l1110_111040

/-- Represents a part of a pie chart -/
structure PieChartPart where
  percentage : ℝ
  centralAngle : ℝ

/-- Theorem stating the relationship between percentage and central angle in a pie chart -/
theorem pie_chart_percentage_central_angle_relation (part : PieChartPart) :
  part.percentage = part.centralAngle / 360 := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_percentage_central_angle_relation_l1110_111040


namespace NUMINAMATH_CALUDE_right_triangle_side_lengths_l1110_111013

/-- A right-angled triangle with given incircle and circumcircle radii -/
structure RightTriangle where
  -- The radius of the incircle
  inradius : ℝ
  -- The radius of the circumcircle
  circumradius : ℝ
  -- The lengths of the three sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- Conditions
  inradius_positive : 0 < inradius
  circumradius_positive : 0 < circumradius
  right_angle : a ^ 2 + b ^ 2 = c ^ 2
  incircle_condition : a + b - c = 2 * inradius
  circumcircle_condition : c = 2 * circumradius

/-- The main theorem stating the side lengths of the triangle -/
theorem right_triangle_side_lengths (t : RightTriangle) 
    (h1 : t.inradius = 8)
    (h2 : t.circumradius = 41) :
    (t.a = 18 ∧ t.b = 80 ∧ t.c = 82) ∨ (t.a = 80 ∧ t.b = 18 ∧ t.c = 82) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_lengths_l1110_111013


namespace NUMINAMATH_CALUDE_min_prime_angle_sum_90_l1110_111087

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem min_prime_angle_sum_90 :
  ∀ x y : ℕ,
    isPrime x →
    isPrime y →
    x + y = 90 →
    y ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_min_prime_angle_sum_90_l1110_111087


namespace NUMINAMATH_CALUDE_sixteen_greater_than_thirtytwo_l1110_111028

/-- Represents a domino placement on a board -/
structure DominoPlacement (n : ℕ) where
  placements : Fin n → Fin 8 × Fin 8 × Bool
  no_overlap : ∀ i j, i ≠ j → placements i ≠ placements j

/-- The number of ways to place n dominoes on an 8x8 board -/
def num_placements (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of 16-domino placements is greater than 32-domino placements -/
theorem sixteen_greater_than_thirtytwo :
  num_placements 16 > num_placements 32 := by sorry

end NUMINAMATH_CALUDE_sixteen_greater_than_thirtytwo_l1110_111028


namespace NUMINAMATH_CALUDE_indeterminate_neutral_eight_year_boys_l1110_111056

structure Classroom where
  total_children : Nat
  happy_children : Nat
  sad_children : Nat
  neutral_children : Nat
  total_boys : Nat
  total_girls : Nat
  happy_boys : Nat
  happy_girls : Nat
  sad_boys : Nat
  sad_girls : Nat
  age_seven_total : Nat
  age_seven_boys : Nat
  age_seven_girls : Nat
  age_eight_total : Nat
  age_eight_boys : Nat
  age_eight_girls : Nat
  age_nine_total : Nat
  age_nine_boys : Nat
  age_nine_girls : Nat

def classroom : Classroom := {
  total_children := 60,
  happy_children := 30,
  sad_children := 10,
  neutral_children := 20,
  total_boys := 16,
  total_girls := 44,
  happy_boys := 6,
  happy_girls := 12,
  sad_boys := 6,
  sad_girls := 4,
  age_seven_total := 20,
  age_seven_boys := 8,
  age_seven_girls := 12,
  age_eight_total := 25,
  age_eight_boys := 5,
  age_eight_girls := 20,
  age_nine_total := 15,
  age_nine_boys := 3,
  age_nine_girls := 12
}

theorem indeterminate_neutral_eight_year_boys (c : Classroom) : 
  c = classroom → 
  ¬∃ (n : Nat), n = c.age_eight_boys - (number_of_happy_eight_year_boys + number_of_sad_eight_year_boys) :=
by sorry

end NUMINAMATH_CALUDE_indeterminate_neutral_eight_year_boys_l1110_111056


namespace NUMINAMATH_CALUDE_function_evaluation_l1110_111001

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x) : f 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_evaluation_l1110_111001


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l1110_111093

/-- The surface area of the circumscribed sphere of a rectangular solid with face diagonals √3, √5, and 2 -/
theorem circumscribed_sphere_surface_area (a b c : ℝ) : 
  a^2 + b^2 = 3 → b^2 + c^2 = 5 → c^2 + a^2 = 4 → 
  4 * Real.pi * ((a^2 + b^2 + c^2) / 4) = 6 * Real.pi := by
  sorry

#check circumscribed_sphere_surface_area

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l1110_111093


namespace NUMINAMATH_CALUDE_minimal_area_parallelepiped_l1110_111023

/-- A right parallelepiped with integer side lengths -/
structure RightParallelepiped where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- The volume of a right parallelepiped -/
def volume (p : RightParallelepiped) : ℕ :=
  p.a * p.b * p.c

/-- The surface area of a right parallelepiped -/
def surfaceArea (p : RightParallelepiped) : ℕ :=
  2 * (p.a * p.b + p.b * p.c + p.c * p.a)

/-- The set of all right parallelepipeds with volume > 1000 -/
def validParallelepipeds : Set RightParallelepiped :=
  {p : RightParallelepiped | volume p > 1000}

theorem minimal_area_parallelepiped :
  ∃ (p : RightParallelepiped),
    p ∈ validParallelepipeds ∧
    p.a = 7 ∧ p.b = 12 ∧ p.c = 12 ∧
    ∀ (q : RightParallelepiped),
      q ∈ validParallelepipeds →
      surfaceArea p ≤ surfaceArea q :=
sorry

end NUMINAMATH_CALUDE_minimal_area_parallelepiped_l1110_111023


namespace NUMINAMATH_CALUDE_andrew_worked_300_days_l1110_111032

/-- Represents the company's vacation policy and Andrew's vacation usage --/
structure VacationData where
  /-- The number of work days required to earn one vacation day --/
  work_days_per_vacation_day : ℕ
  /-- Vacation days taken in March --/
  march_vacation : ℕ
  /-- Vacation days taken in September --/
  september_vacation : ℕ
  /-- Remaining vacation days --/
  remaining_vacation : ℕ

/-- Calculates the total number of days worked given the vacation data --/
def days_worked (data : VacationData) : ℕ :=
  sorry

/-- Theorem stating that given the specific vacation data, Andrew worked 300 days --/
theorem andrew_worked_300_days : 
  let data : VacationData := {
    work_days_per_vacation_day := 10,
    march_vacation := 5,
    september_vacation := 10,
    remaining_vacation := 15
  }
  days_worked data = 300 := by
  sorry

end NUMINAMATH_CALUDE_andrew_worked_300_days_l1110_111032


namespace NUMINAMATH_CALUDE_square_root_of_25_l1110_111095

theorem square_root_of_25 : (Real.sqrt 25) ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_25_l1110_111095


namespace NUMINAMATH_CALUDE_problem_solution_l1110_111024

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m|

-- State the theorem
theorem problem_solution :
  -- Given conditions
  (∀ x : ℝ, f x 2 ≤ 3 ↔ x ∈ Set.Icc (-1) 5) →
  -- Part I: m = 2
  (∃ m : ℝ, ∀ x : ℝ, f x m ≤ 3 ↔ x ∈ Set.Icc (-1) 5) ∧ 
  -- Part II: Minimum value of a² + b² + c² is 2/3
  (∀ a b c : ℝ, a - 2*b + c = 2 → a^2 + b^2 + c^2 ≥ 2/3) ∧
  (∃ a b c : ℝ, a - 2*b + c = 2 ∧ a^2 + b^2 + c^2 = 2/3) :=
by sorry


end NUMINAMATH_CALUDE_problem_solution_l1110_111024


namespace NUMINAMATH_CALUDE_investment_percentage_problem_l1110_111041

theorem investment_percentage_problem (x y : ℝ) (P : ℝ) : 
  x + y = 2000 →
  y = 600 →
  0.1 * x - (P / 100) * y = 92 →
  P = 8 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_problem_l1110_111041


namespace NUMINAMATH_CALUDE_greatest_possible_award_l1110_111014

theorem greatest_possible_award (total_prize : ℕ) (num_winners : ℕ) (min_award : ℕ) :
  total_prize = 600 →
  num_winners = 15 →
  min_award = 15 →
  (2 : ℚ) / 5 * total_prize = (3 : ℚ) / 5 * num_winners * min_award →
  ∃ (max_award : ℕ), max_award = 390 ∧
    max_award ≤ total_prize ∧
    max_award ≥ min_award ∧
    ∃ (other_awards : List ℕ),
      other_awards.length = num_winners - 1 ∧
      (∀ x ∈ other_awards, min_award ≤ x) ∧
      max_award + other_awards.sum = total_prize :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_award_l1110_111014


namespace NUMINAMATH_CALUDE_cuboid_volume_l1110_111075

theorem cuboid_volume (a b c : ℝ) : 
  (a^2 + b^2 + c^2 = 16) →  -- space diagonal length is 4
  (a / 4 = 1/2) →           -- edge a forms 60° angle with diagonal
  (b / 4 = 1/2) →           -- edge b forms 60° angle with diagonal
  (c / 4 = 1/2) →           -- edge c forms 60° angle with diagonal
  (a * b * c = 8) :=        -- volume is 8
by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l1110_111075


namespace NUMINAMATH_CALUDE_inverse_g_at_neg43_l1110_111055

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem
theorem inverse_g_at_neg43 :
  Function.invFun g (-43) = -2 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_at_neg43_l1110_111055


namespace NUMINAMATH_CALUDE_hyperbola_intersection_midpoint_l1110_111020

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 4*x - y - 7 = 0

-- Theorem statement
theorem hyperbola_intersection_midpoint :
  ∃ (A B : ℝ × ℝ),
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    line_equation A.1 A.2 ∧
    line_equation B.1 B.2 ∧
    line_equation P.1 P.2 ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_midpoint_l1110_111020


namespace NUMINAMATH_CALUDE_orange_harvest_days_l1110_111052

def sacks_per_day : ℕ := 4
def total_sacks : ℕ := 56

theorem orange_harvest_days : 
  total_sacks / sacks_per_day = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_days_l1110_111052


namespace NUMINAMATH_CALUDE_lee_makes_27_cookies_l1110_111078

/-- Given that Lee can make 18 cookies with 2 cups of flour, 
    this function calculates how many cookies he can make with any amount of flour. -/
def cookies_from_flour (cups : ℚ) : ℚ :=
  18 * cups / 2

/-- Theorem stating that Lee can make 27 cookies with 3 cups of flour. -/
theorem lee_makes_27_cookies : cookies_from_flour 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_lee_makes_27_cookies_l1110_111078


namespace NUMINAMATH_CALUDE_set_formation_criterion_l1110_111022

-- Define a type for objects
variable {α : Type}

-- Define a predicate for well-defined and specific objects
variable (is_well_defined : α → Prop)

-- Define a predicate for collections that can form sets
def can_form_set (collection : Set α) : Prop :=
  ∀ x ∈ collection, is_well_defined x

-- Theorem statement
theorem set_formation_criterion (collection : Set α) :
  can_form_set is_well_defined collection ↔ ∀ x ∈ collection, is_well_defined x :=
sorry

end NUMINAMATH_CALUDE_set_formation_criterion_l1110_111022


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_a_range_l1110_111074

/-- The piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

/-- The theorem stating the range of a for which f is strictly decreasing -/
theorem f_strictly_decreasing_a_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, (f a x₁ - f a x₂) * (x₁ - x₂) < 0) ↔ 0 < a ∧ a ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_f_strictly_decreasing_a_range_l1110_111074


namespace NUMINAMATH_CALUDE_petya_vasya_game_l1110_111099

theorem petya_vasya_game (k : ℚ) : 
  ∃ (a b c : ℚ), ∃ (x y : ℚ), 
    x^3 + a*x^2 + b*x + c = 0 ∧ 
    y^3 + a*y^2 + b*y + c = 0 ∧ 
    y - x = 2014 :=
by sorry

end NUMINAMATH_CALUDE_petya_vasya_game_l1110_111099


namespace NUMINAMATH_CALUDE_cube_rotation_theorem_l1110_111035

/-- Represents the orientation of a picture on the top face of a cube -/
inductive PictureOrientation
| Original
| Rotated90
| Rotated180

/-- Represents a cube with a picture on its top face -/
structure Cube :=
  (orientation : PictureOrientation)

/-- Represents the action of rolling a cube across its edges -/
def roll (c : Cube) : Cube :=
  sorry

/-- Represents a sequence of rolls that returns the cube to its original position -/
def rollSequence (c : Cube) : Cube :=
  sorry

theorem cube_rotation_theorem (c : Cube) :
  (∃ (seq : Cube → Cube), seq c = Cube.mk PictureOrientation.Rotated180) ∧
  (∀ (seq : Cube → Cube), seq c ≠ Cube.mk PictureOrientation.Rotated90) :=
sorry

end NUMINAMATH_CALUDE_cube_rotation_theorem_l1110_111035


namespace NUMINAMATH_CALUDE_coaching_start_date_l1110_111069

/-- Represents a date in a year -/
structure Date :=
  (month : Nat)
  (day : Nat)

/-- Calculates the number of days from the start of the year to a given date in a non-leap year -/
def daysFromYearStart (d : Date) : Nat :=
  sorry

/-- Calculates the date that is a given number of days before another date in a non-leap year -/
def dateBeforeDays (d : Date) (days : Nat) : Date :=
  sorry

theorem coaching_start_date :
  let end_date : Date := ⟨9, 4⟩  -- September 4
  let coaching_duration : Nat := 245
  let start_date := dateBeforeDays end_date coaching_duration
  start_date = ⟨1, 2⟩  -- January 2
  :=
sorry

end NUMINAMATH_CALUDE_coaching_start_date_l1110_111069


namespace NUMINAMATH_CALUDE_function_existence_l1110_111085

theorem function_existence (k : ℤ) (hk : k ≠ 0) :
  ∃ f : ℤ → ℤ, ∀ a b : ℤ, k * (f (a + b)) + f (a * b) = f a * f b + k :=
by sorry

end NUMINAMATH_CALUDE_function_existence_l1110_111085


namespace NUMINAMATH_CALUDE_complementary_angles_of_same_angle_are_equal_l1110_111088

/-- Two angles are complementary if their sum is 90 degrees -/
def Complementary (α β : ℝ) : Prop := α + β = 90

/-- An angle is the complement of another if together they form 90 degrees -/
def IsComplement (α β : ℝ) : Prop := Complementary α β

theorem complementary_angles_of_same_angle_are_equal 
  (θ α β : ℝ) (h1 : IsComplement θ α) (h2 : IsComplement θ β) : α = β := by
  sorry

#check complementary_angles_of_same_angle_are_equal

end NUMINAMATH_CALUDE_complementary_angles_of_same_angle_are_equal_l1110_111088


namespace NUMINAMATH_CALUDE_lee_quiz_probability_l1110_111065

theorem lee_quiz_probability (p : ℚ) (h : p = 5/8) :
  1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_lee_quiz_probability_l1110_111065


namespace NUMINAMATH_CALUDE_batsman_highest_score_l1110_111068

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (h1 : average = 60)
  (h2 : score_difference = 180)
  (h3 : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    (highest_score : ℚ) - lowest_score = score_difference ∧
    (highest_score + lowest_score : ℚ) = 
      total_innings * average - (total_innings - 2) * average_excluding_extremes ∧
    highest_score = 194 := by
  sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l1110_111068


namespace NUMINAMATH_CALUDE_correct_proposition_l1110_111012

-- Define proposition P
def P : Prop := ∀ x : ℝ, x^2 ≥ 0

-- Define proposition Q
def Q : Prop := ∃ x : ℚ, x^2 ≠ 3

-- Theorem to prove
theorem correct_proposition : P ∨ (¬Q) := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l1110_111012


namespace NUMINAMATH_CALUDE_february_production_l1110_111067

/-- Represents the monthly carrot cake production sequence -/
def carrotCakeSequence : ℕ → ℕ
| 0 => 19  -- October (0-indexed)
| n + 1 => carrotCakeSequence n + 2

/-- Theorem stating that the 5th term (February) of the sequence is 27 -/
theorem february_production : carrotCakeSequence 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_february_production_l1110_111067
