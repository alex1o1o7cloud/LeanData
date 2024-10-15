import Mathlib

namespace NUMINAMATH_CALUDE_exam_failure_percentage_l3051_305178

theorem exam_failure_percentage 
  (total_candidates : ℕ) 
  (hindi_failure_rate : ℚ)
  (both_failure_rate : ℚ)
  (english_only_pass : ℕ) :
  total_candidates = 3000 →
  hindi_failure_rate = 36/100 →
  both_failure_rate = 15/100 →
  english_only_pass = 630 →
  ∃ (english_failure_rate : ℚ),
    english_failure_rate = 85/100 ∧
    english_only_pass = total_candidates * ((1 - english_failure_rate) + (hindi_failure_rate - both_failure_rate)) :=
by sorry

end NUMINAMATH_CALUDE_exam_failure_percentage_l3051_305178


namespace NUMINAMATH_CALUDE_joseph_decks_l3051_305164

/-- The number of complete decks given a total number of cards and cards per deck -/
def number_of_decks (total_cards : ℕ) (cards_per_deck : ℕ) : ℕ :=
  total_cards / cards_per_deck

/-- Proof that Joseph has 4 complete decks of cards -/
theorem joseph_decks :
  number_of_decks 208 52 = 4 := by
  sorry

end NUMINAMATH_CALUDE_joseph_decks_l3051_305164


namespace NUMINAMATH_CALUDE_factorization_2x_squared_minus_4x_l3051_305197

theorem factorization_2x_squared_minus_4x (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2x_squared_minus_4x_l3051_305197


namespace NUMINAMATH_CALUDE_chocolates_in_boxes_l3051_305130

theorem chocolates_in_boxes (total_chocolates : ℕ) (filled_boxes : ℕ) (loose_chocolates : ℕ) (friend_chocolates : ℕ) (box_capacity : ℕ) : 
  total_chocolates = 50 →
  filled_boxes = 3 →
  loose_chocolates = 5 →
  friend_chocolates = 25 →
  box_capacity = 15 →
  (total_chocolates - loose_chocolates) / filled_boxes = box_capacity →
  (loose_chocolates + friend_chocolates) / box_capacity = 2 := by
sorry

end NUMINAMATH_CALUDE_chocolates_in_boxes_l3051_305130


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3051_305191

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 9 = 16 →
  a 2 * a 5 * a 8 = 64 := by
  sorry

#check geometric_sequence_product

end NUMINAMATH_CALUDE_geometric_sequence_product_l3051_305191


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l3051_305104

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The axis of symmetry for a function f is a vertical line x = a such that
    f(a + x) = f(a - x) for all x in the domain of f -/
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) :
  IsEven (fun x ↦ f (x + 1)) → AxisOfSymmetry f 1 := by sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l3051_305104


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l3051_305145

/-- Given a rectangular pen with perimeter 60 feet and one side length at least 15 feet,
    the maximum possible area is 225 square feet. -/
theorem max_area_rectangular_pen :
  ∀ (x y : ℝ),
    x > 0 ∧ y > 0 →
    x + y = 30 →
    (x ≥ 15 ∨ y ≥ 15) →
    x * y ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l3051_305145


namespace NUMINAMATH_CALUDE_max_value_cos_squared_minus_sin_l3051_305175

open Real

theorem max_value_cos_squared_minus_sin (x : ℝ) : 
  ∃ (M : ℝ), M = (5 : ℝ) / 4 ∧ ∀ x, cos x ^ 2 - sin x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_cos_squared_minus_sin_l3051_305175


namespace NUMINAMATH_CALUDE_average_rainfall_leap_year_february_l3051_305183

/-- Calculates the average rainfall per hour in February of a leap year -/
theorem average_rainfall_leap_year_february (total_rainfall : ℝ) :
  total_rainfall = 420 →
  (35 : ℝ) / 58 = total_rainfall / (29 * 24) := by
  sorry

end NUMINAMATH_CALUDE_average_rainfall_leap_year_february_l3051_305183


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l3051_305194

/-- A geometric sequence with sum S_n = 3 · 2^n + k -/
def geometric_sequence (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) (k : ℝ) : Prop :=
  ∀ n : ℕ+, S n = 3 * 2^(n : ℝ) + k

theorem geometric_sequence_constant (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  geometric_sequence a S (-3) →
  (∀ n : ℕ+, a n = S n - S (n - 1)) →
  a 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l3051_305194


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3051_305143

theorem sin_2alpha_value (α : Real) (h : Real.sin α - Real.cos α = 1/5) : 
  Real.sin (2 * α) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3051_305143


namespace NUMINAMATH_CALUDE_bouquet_39_roses_cost_l3051_305181

/-- Represents the cost of a bouquet of roses -/
structure BouquetCost where
  baseCost : ℝ
  additionalCostPerRose : ℝ

/-- Calculates the total cost of a bouquet given the number of roses -/
def totalCost (bc : BouquetCost) (numRoses : ℕ) : ℝ :=
  bc.baseCost + bc.additionalCostPerRose * numRoses

/-- Theorem: Given the conditions, a bouquet of 39 roses costs $58.75 -/
theorem bouquet_39_roses_cost
  (bc : BouquetCost)
  (h1 : bc.baseCost = 10)
  (h2 : totalCost bc 12 = 25) :
  totalCost bc 39 = 58.75 := by
  sorry

#check bouquet_39_roses_cost

end NUMINAMATH_CALUDE_bouquet_39_roses_cost_l3051_305181


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3051_305120

theorem imaginary_part_of_z (z : ℂ) (h : 1 + (1 + 2 * z) * Complex.I = 0) :
  z.im = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3051_305120


namespace NUMINAMATH_CALUDE_inverse_of_AB_l3051_305105

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 0; 0, 2]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![1, 1/2; 0, 1]

theorem inverse_of_AB :
  (A * B)⁻¹ = !![1, -1; 0, 1/2] := by sorry

end NUMINAMATH_CALUDE_inverse_of_AB_l3051_305105


namespace NUMINAMATH_CALUDE_lucy_doll_collection_l3051_305167

/-- Represents Lucy's doll collection problem -/
theorem lucy_doll_collection (X : ℕ) (Z : ℕ) : 
  (X : ℚ) * (1 + 1/5) = X + 5 → -- 20% increase after adding 5 dolls
  Z = (X + 5 + (X + 5) / 10 : ℚ).floor → -- 10% more dolls from updated collection
  X = 25 ∧ Z = 33 := by
  sorry

end NUMINAMATH_CALUDE_lucy_doll_collection_l3051_305167


namespace NUMINAMATH_CALUDE_debugging_time_l3051_305148

theorem debugging_time (total_hours : ℝ) (flow_chart_frac : ℝ) (coding_frac : ℝ) (meeting_frac : ℝ)
  (h1 : total_hours = 192)
  (h2 : flow_chart_frac = 3 / 10)
  (h3 : coding_frac = 3 / 8)
  (h4 : meeting_frac = 1 / 5)
  (h5 : flow_chart_frac + coding_frac + meeting_frac < 1) :
  total_hours - (flow_chart_frac + coding_frac + meeting_frac) * total_hours = 24 := by
  sorry

end NUMINAMATH_CALUDE_debugging_time_l3051_305148


namespace NUMINAMATH_CALUDE_sphere_surface_area_in_dihedral_angle_l3051_305135

/-- The surface area of a part of a sphere inside a dihedral angle -/
theorem sphere_surface_area_in_dihedral_angle 
  (R a α : ℝ) 
  (h_positive_R : R > 0)
  (h_positive_a : a > 0)
  (h_a_lt_R : a < R)
  (h_angle_range : 0 < α ∧ α < π) :
  let surface_area := 
    2 * R^2 * Real.arccos ((R * Real.cos α) / Real.sqrt (R^2 - a^2 * Real.sin α^2)) - 
    2 * R * a * Real.sin α * Real.arccos ((a * Real.cos α) / Real.sqrt (R^2 - a^2 * Real.sin α^2))
  surface_area > 0 ∧ surface_area < 4 * π * R^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_in_dihedral_angle_l3051_305135


namespace NUMINAMATH_CALUDE_unique_box_filling_l3051_305195

/-- Represents a rectangular parallelepiped with integer dimensions -/
structure Brick where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a brick -/
def Brick.volume (b : Brick) : ℕ := b.length * b.width * b.height

/-- The box to be filled -/
def box : Brick := ⟨10, 11, 14⟩

/-- The first type of brick -/
def brickA : Brick := ⟨2, 5, 8⟩

/-- The second type of brick -/
def brickB : Brick := ⟨2, 3, 7⟩

/-- Theorem stating that the only way to fill the box is with 14 bricks of type A and 10 of type B -/
theorem unique_box_filling :
  ∀ (x y : ℕ), 
    x * brickA.volume + y * brickB.volume = box.volume → 
    (x = 14 ∧ y = 10) := by sorry

end NUMINAMATH_CALUDE_unique_box_filling_l3051_305195


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l3051_305182

/-- Tim's daily task count -/
def daily_tasks : ℕ := 100

/-- Tim's working days per week -/
def working_days : ℕ := 6

/-- Number of tasks paying $1.2 each -/
def tasks_1_2 : ℕ := 40

/-- Number of tasks paying $1.5 each -/
def tasks_1_5 : ℕ := 30

/-- Number of tasks paying $2 each -/
def tasks_2 : ℕ := 30

/-- Payment rate for the first group of tasks -/
def rate_1_2 : ℚ := 1.2

/-- Payment rate for the second group of tasks -/
def rate_1_5 : ℚ := 1.5

/-- Payment rate for the third group of tasks -/
def rate_2 : ℚ := 2

/-- Tim's weekly earnings -/
def weekly_earnings : ℚ := 918

theorem tim_weekly_earnings :
  daily_tasks = tasks_1_2 + tasks_1_5 + tasks_2 →
  working_days * (tasks_1_2 * rate_1_2 + tasks_1_5 * rate_1_5 + tasks_2 * rate_2) = weekly_earnings :=
by sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l3051_305182


namespace NUMINAMATH_CALUDE_pen_ratio_is_one_l3051_305170

theorem pen_ratio_is_one (initial_pens : ℕ) (mike_pens : ℕ) (sharon_pens : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 25)
  (h2 : mike_pens = 22)
  (h3 : sharon_pens = 19)
  (h4 : final_pens = 75) :
  (final_pens + sharon_pens - (initial_pens + mike_pens)) / (initial_pens + mike_pens) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_ratio_is_one_l3051_305170


namespace NUMINAMATH_CALUDE_rectangle_area_error_percent_l3051_305144

/-- Given a rectangle with sides measured with errors, calculate the error percent in the area --/
theorem rectangle_area_error_percent (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let measured_length := 1.05 * L
  let measured_width := 0.96 * W
  let actual_area := L * W
  let measured_area := measured_length * measured_width
  let error := measured_area - actual_area
  let error_percent := (error / actual_area) * 100
  error_percent = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percent_l3051_305144


namespace NUMINAMATH_CALUDE_union_A_complement_B_l3051_305157

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- Theorem statement
theorem union_A_complement_B : A ∪ (U \ B) = {x | x < 2} := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_l3051_305157


namespace NUMINAMATH_CALUDE_complex_power_sum_l3051_305152

/-- If z is a complex number satisfying z + 1/z = 2 cos 5°, then z^1500 + 1/z^1500 = 1 -/
theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1500 + 1/z^1500 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3051_305152


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_3_l3051_305123

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

theorem parallel_lines_imply_a_equals_3 :
  ∀ a : ℝ,
  let l1 : Line := ⟨a, 2, 3*a⟩
  let l2 : Line := ⟨3, a-1, a-7⟩
  parallel l1 l2 → a = 3 := by
  sorry

#check parallel_lines_imply_a_equals_3

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_3_l3051_305123


namespace NUMINAMATH_CALUDE_number_satisfying_equation_l3051_305101

theorem number_satisfying_equation : ∃! x : ℚ, x + 72 = 2 * x / (2/3) := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_equation_l3051_305101


namespace NUMINAMATH_CALUDE_gumball_machine_total_l3051_305146

/-- Represents the number of gumballs of each color in a gumball machine. -/
structure GumballMachine where
  red : ℕ
  green : ℕ
  blue : ℕ
  yellow : ℕ
  orange : ℕ

/-- Represents the conditions of the gumball machine problem. -/
def gumball_machine_conditions (m : GumballMachine) : Prop :=
  m.blue = m.red / 2 ∧
  m.green = 4 * m.blue ∧
  m.yellow = (7 * m.blue) / 2 ∧
  m.orange = (2 * (m.red + m.blue)) / 3 ∧
  m.red = (3 * m.yellow) / 2 ∧
  m.yellow = 24

/-- The theorem stating that a gumball machine satisfying the given conditions has 186 gumballs. -/
theorem gumball_machine_total (m : GumballMachine) 
  (h : gumball_machine_conditions m) : 
  m.red + m.green + m.blue + m.yellow + m.orange = 186 := by
  sorry


end NUMINAMATH_CALUDE_gumball_machine_total_l3051_305146


namespace NUMINAMATH_CALUDE_rectangle_formations_l3051_305133

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of horizontal lines -/
def horizontal_lines : ℕ := 5

/-- The number of vertical lines -/
def vertical_lines : ℕ := 5

/-- The number of horizontal lines needed to form a rectangle -/
def horizontal_lines_needed : ℕ := 2

/-- The number of vertical lines needed to form a rectangle -/
def vertical_lines_needed : ℕ := 2

/-- The theorem stating the number of ways to form a rectangle -/
theorem rectangle_formations :
  (choose horizontal_lines horizontal_lines_needed) *
  (choose vertical_lines vertical_lines_needed) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formations_l3051_305133


namespace NUMINAMATH_CALUDE_probability_no_shaded_l3051_305140

/-- Represents a rectangle in the 2 by 1001 grid --/
structure Rectangle where
  left : Nat
  right : Nat
  top : Nat
  bottom : Nat

/-- The total number of possible rectangles in the grid --/
def total_rectangles : Nat := 501501

/-- The number of rectangles containing at least one shaded square --/
def shaded_rectangles : Nat := 252002

/-- Checks if a rectangle contains a shaded square --/
def contains_shaded (r : Rectangle) : Prop :=
  (r.left = 1 ∧ r.right ≥ 1) ∨ 
  (r.left ≤ 501 ∧ r.right ≥ 501) ∨ 
  (r.left ≤ 1001 ∧ r.right = 1001)

/-- The main theorem stating the probability of choosing a rectangle without a shaded square --/
theorem probability_no_shaded : 
  (total_rectangles - shaded_rectangles) / total_rectangles = 249499 / 501501 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_shaded_l3051_305140


namespace NUMINAMATH_CALUDE_negative_f_reflection_l3051_305109

-- Define a function f
variable (f : ℝ → ℝ)

-- Define reflection across x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Theorem: The graph of y = -f(x) is the reflection of y = f(x) across the x-axis
theorem negative_f_reflection (x : ℝ) : 
  reflect_x (x, f x) = (x, -f x) := by sorry

end NUMINAMATH_CALUDE_negative_f_reflection_l3051_305109


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3051_305137

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₀ + a₂ + a₄ = 41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3051_305137


namespace NUMINAMATH_CALUDE_sarah_initial_money_l3051_305124

def toy_car_price : ℕ := 11
def toy_car_quantity : ℕ := 2
def scarf_price : ℕ := 10
def beanie_price : ℕ := 14
def remaining_money : ℕ := 7

theorem sarah_initial_money :
  ∃ (initial_money : ℕ),
    initial_money = 
      remaining_money + beanie_price + scarf_price + (toy_car_price * toy_car_quantity) ∧
    initial_money = 53 := by
  sorry

end NUMINAMATH_CALUDE_sarah_initial_money_l3051_305124


namespace NUMINAMATH_CALUDE_converse_proposition_l3051_305186

theorem converse_proposition : ∀ x : ℝ, (1 / (x - 1) ≥ 3) → (x ≤ 4 / 3) := by sorry

end NUMINAMATH_CALUDE_converse_proposition_l3051_305186


namespace NUMINAMATH_CALUDE_min_value_inequality_l3051_305121

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : a + 2*b + 3*c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3051_305121


namespace NUMINAMATH_CALUDE_parabola_directrix_l3051_305125

/-- The directrix of the parabola x = -1/4 * y^2 is x = 1 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), x = -(1/4) * y^2 → 
  ∃ (d : ℝ), d = 1 ∧ 
  ∀ (p : ℝ × ℝ), p.1 = -(1/4) * p.2^2 → 
  (p.1 - d)^2 = (p.1 - (-d))^2 + p.2^2 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3051_305125


namespace NUMINAMATH_CALUDE_power_function_m_values_l3051_305188

/-- A function is a power function if it's of the form f(x) = ax^n, where a ≠ 0 and n is a real number -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The given function f(x) = (m^2 - m - 1)x^3 -/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ (m^2 - m - 1) * x^3

/-- Theorem: If f(x) = (m^2 - m - 1)x^3 is a power function, then m = -1 or m = 2 -/
theorem power_function_m_values (m : ℝ) : IsPowerFunction (f m) → m = -1 ∨ m = 2 := by
  sorry


end NUMINAMATH_CALUDE_power_function_m_values_l3051_305188


namespace NUMINAMATH_CALUDE_lcm_factor_is_one_l3051_305184

/-- Given two positive integers with specific properties, prove that a certain factor of their LCM is 1. -/
theorem lcm_factor_is_one (A B : ℕ+) (X : ℕ) 
  (hcf : Nat.gcd A B = 10)
  (a_val : A = 150)
  (lcm_fact : Nat.lcm A B = 10 * X * 15) : X = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_is_one_l3051_305184


namespace NUMINAMATH_CALUDE_equation_simplification_l3051_305169

theorem equation_simplification :
  120 + (150 / 10) + (35 * 9) - 300 - (420 / 7) + 2^3 = 98 := by
  sorry

end NUMINAMATH_CALUDE_equation_simplification_l3051_305169


namespace NUMINAMATH_CALUDE_bella_needs_twelve_beads_l3051_305180

/-- Given the number of friends, beads per bracelet, and beads on hand,
    calculate the number of additional beads needed. -/
def additional_beads_needed (friends : ℕ) (beads_per_bracelet : ℕ) (beads_on_hand : ℕ) : ℕ :=
  max 0 (friends * beads_per_bracelet - beads_on_hand)

/-- Proof that Bella needs 12 more beads to make bracelets for her friends. -/
theorem bella_needs_twelve_beads :
  additional_beads_needed 6 8 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_bella_needs_twelve_beads_l3051_305180


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3051_305168

theorem inequality_system_solution (x : ℝ) : 
  (x - 2 < 0 ∧ 5 * x + 1 > 2 * (x - 1)) ↔ -1/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3051_305168


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3051_305177

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem geometric_sequence_property
  (b : ℕ → ℝ)
  (h_geometric : geometric_sequence b)
  (h_b1 : b 1 = 1)
  (s t : ℕ)
  (h_distinct : s ≠ t)
  (h_positive : s > 0 ∧ t > 0) :
  (b t) ^ (s - 1) / (b s) ^ (t - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3051_305177


namespace NUMINAMATH_CALUDE_money_division_l3051_305111

theorem money_division (total : ℕ) (p q r : ℕ) : 
  p + q + r = total ∧ 
  3 * p = 7 * q ∧ 
  7 * q = 12 * r ∧ 
  r - q = 3500 → 
  q - p = 2800 := by sorry

end NUMINAMATH_CALUDE_money_division_l3051_305111


namespace NUMINAMATH_CALUDE_bus_driver_rate_l3051_305176

/-- Represents the bus driver's compensation structure and work details -/
structure BusDriverCompensation where
  regularHours : ℕ := 40
  totalHours : ℕ
  overtimeMultiplier : ℚ
  totalCompensation : ℚ

/-- Calculates the regular hourly rate given the compensation structure -/
def calculateRegularRate (bdc : BusDriverCompensation) : ℚ :=
  let overtimeHours := bdc.totalHours - bdc.regularHours
  bdc.totalCompensation / (bdc.regularHours + overtimeHours * bdc.overtimeMultiplier)

/-- Theorem stating that the bus driver's regular rate is $16 per hour -/
theorem bus_driver_rate : 
  let bdc : BusDriverCompensation := {
    totalHours := 65,
    overtimeMultiplier := 1.75,
    totalCompensation := 1340
  }
  calculateRegularRate bdc = 16 := by sorry

end NUMINAMATH_CALUDE_bus_driver_rate_l3051_305176


namespace NUMINAMATH_CALUDE_toy_cost_l3051_305193

/-- The cost of each toy given Paul's savings and allowance -/
theorem toy_cost (initial_savings : ℕ) (allowance : ℕ) (num_toys : ℕ) 
  (h1 : initial_savings = 3)
  (h2 : allowance = 7)
  (h3 : num_toys = 2)
  (h4 : num_toys > 0) :
  (initial_savings + allowance) / num_toys = 5 := by
  sorry


end NUMINAMATH_CALUDE_toy_cost_l3051_305193


namespace NUMINAMATH_CALUDE_two_days_saved_l3051_305118

/-- Represents the work scenario with original and additional workers --/
structure WorkScenario where
  originalMen : ℕ
  originalDays : ℕ
  additionalMen : ℕ
  totalWork : ℕ

/-- Calculates the number of days saved when additional workers join --/
def daysSaved (w : WorkScenario) : ℕ :=
  w.originalDays - (w.totalWork / (w.originalMen + w.additionalMen))

/-- Theorem stating that in the given scenario, 2 days are saved --/
theorem two_days_saved (w : WorkScenario) 
  (h1 : w.originalMen = 30)
  (h2 : w.originalDays = 8)
  (h3 : w.additionalMen = 10)
  (h4 : w.totalWork = w.originalMen * w.originalDays) :
  daysSaved w = 2 := by
  sorry

#eval daysSaved { originalMen := 30, originalDays := 8, additionalMen := 10, totalWork := 240 }

end NUMINAMATH_CALUDE_two_days_saved_l3051_305118


namespace NUMINAMATH_CALUDE_rectangle_area_l3051_305192

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 1225)
  (h2 : rectangle_breadth = 10)
  : ∃ (circle_radius : ℝ) (rectangle_length : ℝ),
    circle_radius ^ 2 = square_area ∧
    rectangle_length = (2 / 5) * circle_radius ∧
    rectangle_length * rectangle_breadth = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3051_305192


namespace NUMINAMATH_CALUDE_right_triangle_legs_l3051_305158

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of the first segment of the hypotenuse -/
  a : ℝ
  /-- Length of the second segment of the hypotenuse -/
  b : ℝ
  /-- The first leg of the triangle -/
  leg1 : ℝ
  /-- The second leg of the triangle -/
  leg2 : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The first segment plus radius equals the first leg -/
  h1 : a + r = leg1
  /-- The second segment plus radius equals the second leg -/
  h2 : b + r = leg2
  /-- The Pythagorean theorem holds -/
  pythagoras : leg1^2 + leg2^2 = (a + b)^2

/-- The main theorem -/
theorem right_triangle_legs (t : RightTriangleWithInscribedCircle)
  (ha : t.a = 5) (hb : t.b = 12) : t.leg1 = 8 ∧ t.leg2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l3051_305158


namespace NUMINAMATH_CALUDE_ball_purchase_theorem_l3051_305185

/-- Represents the cost and quantity of balls in two purchases -/
structure BallPurchase where
  soccer_price : ℝ
  volleyball_price : ℝ
  soccer_quantity1 : ℕ
  volleyball_quantity1 : ℕ
  total_cost1 : ℝ
  total_quantity2 : ℕ
  soccer_price_increase : ℝ
  volleyball_price_decrease : ℝ
  total_cost2_ratio : ℝ

/-- Theorem stating the prices of balls and the quantity of volleyballs in the second purchase -/
theorem ball_purchase_theorem (bp : BallPurchase)
  (h1 : bp.soccer_quantity1 * bp.soccer_price + bp.volleyball_quantity1 * bp.volleyball_price = bp.total_cost1)
  (h2 : bp.soccer_price = bp.volleyball_price + 30)
  (h3 : bp.soccer_quantity1 = 40)
  (h4 : bp.volleyball_quantity1 = 30)
  (h5 : bp.total_cost1 = 4000)
  (h6 : bp.total_quantity2 = 50)
  (h7 : bp.soccer_price_increase = 0.1)
  (h8 : bp.volleyball_price_decrease = 0.1)
  (h9 : bp.total_cost2_ratio = 0.86) :
  bp.soccer_price = 70 ∧ bp.volleyball_price = 40 ∧
  ∃ m : ℕ, m = 10 ∧ 
    (bp.total_quantity2 - m) * (bp.soccer_price * (1 + bp.soccer_price_increase)) +
    m * (bp.volleyball_price * (1 - bp.volleyball_price_decrease)) =
    bp.total_cost1 * bp.total_cost2_ratio :=
by sorry

end NUMINAMATH_CALUDE_ball_purchase_theorem_l3051_305185


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l3051_305163

theorem lcm_from_hcf_and_product (x y : ℕ+) : 
  Nat.gcd x y = 12 → x * y = 2460 → Nat.lcm x y = 205 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l3051_305163


namespace NUMINAMATH_CALUDE_initial_fee_equals_65_l3051_305187

/-- The initial fee of the first car rental plan -/
def initial_fee : ℝ := 65

/-- The cost per mile for the first plan -/
def cost_per_mile_plan1 : ℝ := 0.40

/-- The cost per mile for the second plan -/
def cost_per_mile_plan2 : ℝ := 0.60

/-- The number of miles driven -/
def miles_driven : ℝ := 325

/-- Theorem stating that the initial fee makes both plans cost the same for the given miles -/
theorem initial_fee_equals_65 :
  initial_fee + cost_per_mile_plan1 * miles_driven = cost_per_mile_plan2 * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_initial_fee_equals_65_l3051_305187


namespace NUMINAMATH_CALUDE_parallel_intersecting_lines_c_is_zero_l3051_305141

/-- Two lines that are parallel and intersect at a specific point -/
structure ParallelIntersectingLines where
  a : ℝ
  b : ℝ
  c : ℝ
  parallel : a / 2 = -2 / b
  intersect_x : 2 * a - 2 * (-4) = c
  intersect_y : 2 * 2 + b * (-4) = c

/-- The theorem stating that for such lines, c must be 0 -/
theorem parallel_intersecting_lines_c_is_zero (lines : ParallelIntersectingLines) : lines.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_intersecting_lines_c_is_zero_l3051_305141


namespace NUMINAMATH_CALUDE_sum_and_simplification_l3051_305112

theorem sum_and_simplification : 
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (7 : ℚ) / 8 + (11 : ℚ) / 12 = (n : ℚ) / d ∧ 
  (∀ (k : ℕ), k > 1 → ¬(k ∣ n ∧ k ∣ d)) :=
by sorry

end NUMINAMATH_CALUDE_sum_and_simplification_l3051_305112


namespace NUMINAMATH_CALUDE_inequality_proof_l3051_305113

theorem inequality_proof (a : ℝ) (h : a > 0) : 
  Real.sqrt (a + 1/a) - Real.sqrt 2 ≥ Real.sqrt a + 1/(Real.sqrt a) - 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3051_305113


namespace NUMINAMATH_CALUDE_bakery_combinations_l3051_305153

/-- The number of ways to distribute n items among k categories, 
    with at least m items in each of the first two categories -/
def distribute (n k m : ℕ) : ℕ :=
  -- We don't provide the implementation, just the type signature
  sorry

/-- The specific case for the bakery problem -/
theorem bakery_combinations : distribute 8 5 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_bakery_combinations_l3051_305153


namespace NUMINAMATH_CALUDE_min_value_and_max_product_l3051_305147

def f (x : ℝ) : ℝ := 2 * abs (x + 1) - abs (x - 1)

theorem min_value_and_max_product :
  (∃ k : ℝ, ∀ x : ℝ, f x ≥ k ∧ ∃ x₀ : ℝ, f x₀ = k) ∧
  (∀ a b c : ℝ, a^2 + c^2 + b^2/2 = 2 → b*(a+c) ≤ 2) ∧
  (∃ a b c : ℝ, a^2 + c^2 + b^2/2 = 2 ∧ b*(a+c) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_max_product_l3051_305147


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l3051_305165

def A : ℂ := 3 - 4 * Complex.I
def M : ℂ := -3 + 2 * Complex.I
def S : ℂ := 2 * Complex.I
def P : ℂ := -1

theorem complex_arithmetic_result : A - M + S + P = 5 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l3051_305165


namespace NUMINAMATH_CALUDE_initial_men_count_l3051_305199

/-- Represents the initial number of men -/
def initialMen : ℕ := 200

/-- Represents the initial food duration in days -/
def initialDuration : ℕ := 20

/-- Represents the number of days after which some men leave -/
def daysBeforeLeaving : ℕ := 15

/-- Represents the number of men who leave -/
def menWhoLeave : ℕ := 100

/-- Represents the remaining food duration after some men leave -/
def remainingDuration : ℕ := 10

theorem initial_men_count :
  initialMen * daysBeforeLeaving = (initialMen - menWhoLeave) * remainingDuration ∧
  initialMen * initialDuration = initialMen * daysBeforeLeaving + (initialMen - menWhoLeave) * remainingDuration :=
by sorry

end NUMINAMATH_CALUDE_initial_men_count_l3051_305199


namespace NUMINAMATH_CALUDE_expression_evaluation_l3051_305127

theorem expression_evaluation (m n : ℤ) (h1 : m = 2) (h2 : n = 1) : 
  (2 * m^2 - 3 * m * n + 8) - (5 * m * n - 4 * m^2 + 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3051_305127


namespace NUMINAMATH_CALUDE_inverse_g_at_19_128_l3051_305149

noncomputable def g (x : ℝ) : ℝ := (x^7 - 1) / 4

theorem inverse_g_at_19_128 :
  g⁻¹ (19/128) = (51/32)^(1/7) := by
sorry

end NUMINAMATH_CALUDE_inverse_g_at_19_128_l3051_305149


namespace NUMINAMATH_CALUDE_technicians_sample_size_l3051_305108

/-- Represents the number of technicians to be included in a stratified sample -/
def technicians_in_sample (total_engineers : ℕ) (total_technicians : ℕ) (total_workers : ℕ) (sample_size : ℕ) : ℕ :=
  (total_technicians * sample_size) / (total_engineers + total_technicians + total_workers)

/-- Theorem stating that the number of technicians in the sample is 5 -/
theorem technicians_sample_size :
  technicians_in_sample 20 100 280 20 = 5 := by
  sorry

#eval technicians_in_sample 20 100 280 20

end NUMINAMATH_CALUDE_technicians_sample_size_l3051_305108


namespace NUMINAMATH_CALUDE_point_and_tangent_line_l3051_305151

def f (a t x : ℝ) : ℝ := x^3 + a*x
def g (b c t x : ℝ) : ℝ := b*x^2 + c
def h (a b c t x : ℝ) : ℝ := f a t x - g b c t x

theorem point_and_tangent_line (t : ℝ) (h_t : t ≠ 0) :
  ∃ (a b c : ℝ),
    (f a t t = 0) ∧
    (g b c t t = 0) ∧
    (∀ x, (deriv (f a t)) x = (deriv (g b c t)) x) ∧
    (∀ x ∈ Set.Ioo (-1) 3, StrictMonoOn (h a b c t) (Set.Ioo (-1) 3)) →
    (a = -t^2 ∧ b = t ∧ c = -t^3 ∧ (t ≤ -9 ∨ t ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_point_and_tangent_line_l3051_305151


namespace NUMINAMATH_CALUDE_valid_number_count_l3051_305160

/-- Represents a valid seven-digit number configuration --/
structure ValidNumber :=
  (digits : Fin 7 → Fin 7)
  (injective : Function.Injective digits)
  (no_6_7_at_ends : digits 0 ≠ 5 ∧ digits 0 ≠ 6 ∧ digits 6 ≠ 5 ∧ digits 6 ≠ 6)
  (one_adjacent_six : ∃ i, (digits i = 0 ∧ digits (i+1) = 5) ∨ (digits i = 5 ∧ digits (i+1) = 0))

/-- The number of valid seven-digit numbers --/
def count_valid_numbers : ℕ := sorry

/-- Theorem stating the count of valid numbers --/
theorem valid_number_count : count_valid_numbers = 768 := by sorry

end NUMINAMATH_CALUDE_valid_number_count_l3051_305160


namespace NUMINAMATH_CALUDE_angle_triple_complement_l3051_305172

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l3051_305172


namespace NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l3051_305196

def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_55_divisors :
  ∃ (n : ℕ), num_divisors n = 55 ∧ 
  (∀ m : ℕ, num_divisors m = 55 → n ≤ m) ∧
  n = 3^4 * 2^10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l3051_305196


namespace NUMINAMATH_CALUDE_leas_purchases_total_cost_l3051_305155

/-- The total cost of Léa's purchases is $28, given that she bought one book for $16, 
    three binders for $2 each, and six notebooks for $1 each. -/
theorem leas_purchases_total_cost : 
  let book_cost : ℕ := 16
  let binder_cost : ℕ := 2
  let notebook_cost : ℕ := 1
  let num_binders : ℕ := 3
  let num_notebooks : ℕ := 6
  book_cost + num_binders * binder_cost + num_notebooks * notebook_cost = 28 :=
by sorry

end NUMINAMATH_CALUDE_leas_purchases_total_cost_l3051_305155


namespace NUMINAMATH_CALUDE_min_sum_three_integers_l3051_305162

theorem min_sum_three_integers (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∃ (k₁ k₂ k₃ : ℕ), 
    (1 / a + 1 / b : ℚ) = k₁ * (1 / c : ℚ) ∧
    (1 / a + 1 / c : ℚ) = k₂ * (1 / b : ℚ) ∧
    (1 / b + 1 / c : ℚ) = k₃ * (1 / a : ℚ)) →
  a + b + c ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_min_sum_three_integers_l3051_305162


namespace NUMINAMATH_CALUDE_unique_four_digit_prime_product_l3051_305171

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem unique_four_digit_prime_product :
  ∃! n : ℕ,
    1000 ≤ n ∧ n ≤ 9999 ∧
    ∃ (p q r s : ℕ),
      is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
      p < q ∧ q < r ∧
      n = p * q * r ∧
      p + q = r - q ∧
      p + q + r = s^2 ∧
      n = 2015 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_prime_product_l3051_305171


namespace NUMINAMATH_CALUDE_two_lines_perpendicular_to_plane_are_parallel_two_planes_perpendicular_to_line_are_parallel_l3051_305166

-- Define the basic geometric objects
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the geometric relationships
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular_plane : Plane → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Statement ②
theorem two_lines_perpendicular_to_plane_are_parallel 
  (p : Plane) (l1 l2 : Line) :
  perpendicular l1 p → perpendicular l2 p → parallel l1 l2 := by sorry

-- Statement ③
theorem two_planes_perpendicular_to_line_are_parallel 
  (l : Line) (p1 p2 : Plane) :
  perpendicular_plane p1 l → perpendicular_plane p2 l → parallel_plane p1 p2 := by sorry

end NUMINAMATH_CALUDE_two_lines_perpendicular_to_plane_are_parallel_two_planes_perpendicular_to_line_are_parallel_l3051_305166


namespace NUMINAMATH_CALUDE_intersection_X_complement_Y_l3051_305142

def U : Set ℝ := Set.univ

def X : Set ℝ := {x | x^2 - x = 0}

def Y : Set ℝ := {x | x^2 + x = 0}

theorem intersection_X_complement_Y : X ∩ (U \ Y) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_X_complement_Y_l3051_305142


namespace NUMINAMATH_CALUDE_temperature_difference_l3051_305138

theorem temperature_difference (highest lowest : Int) 
  (h1 : highest = 11) 
  (h2 : lowest = -11) : 
  highest - lowest = 22 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3051_305138


namespace NUMINAMATH_CALUDE_cube_of_negative_l3051_305156

theorem cube_of_negative (x : ℝ) : (-x)^3 = -x^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_l3051_305156


namespace NUMINAMATH_CALUDE_tenfold_largest_two_digit_l3051_305174

def largest_two_digit_number : ℕ := 99

theorem tenfold_largest_two_digit : 10 * largest_two_digit_number = 990 := by
  sorry

end NUMINAMATH_CALUDE_tenfold_largest_two_digit_l3051_305174


namespace NUMINAMATH_CALUDE_simplify_fraction_l3051_305110

theorem simplify_fraction : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3051_305110


namespace NUMINAMATH_CALUDE_calculate_expression_l3051_305159

theorem calculate_expression : 101 * 102^2 - 101 * 98^2 = 80800 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3051_305159


namespace NUMINAMATH_CALUDE_files_per_folder_l3051_305114

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) :
  initial_files = 93 →
  deleted_files = 21 →
  num_folders = 9 →
  (initial_files - deleted_files) / num_folders = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_files_per_folder_l3051_305114


namespace NUMINAMATH_CALUDE_remaining_macaroons_weight_l3051_305189

def macaroon_problem (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) (bags_eaten : ℕ) : ℕ :=
  let total_weight := total_macaroons * weight_per_macaroon
  let macaroons_per_bag := total_macaroons / num_bags
  let weight_per_bag := macaroons_per_bag * weight_per_macaroon
  total_weight - (bags_eaten * weight_per_bag)

theorem remaining_macaroons_weight :
  macaroon_problem 12 5 4 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_macaroons_weight_l3051_305189


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3051_305116

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, (x < -1 → (x < -1 ∨ x > 1)) ∧ ¬((x < -1 ∨ x > 1) → x < -1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3051_305116


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3051_305103

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 / (x^2 - 1) = 1 / (x - 1) - 1 / (x + 1)) ∧
  (2*x / (x^2 - 1) = 1 / (x - 1) + 1 / (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3051_305103


namespace NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l3051_305131

/-- Represents the fishing schedule in a coastal village -/
structure FishingSchedule where
  daily : ℕ              -- Number of people fishing daily
  everyOtherDay : ℕ      -- Number of people fishing every other day
  everyThreeDay : ℕ      -- Number of people fishing every three days
  yesterday : ℕ          -- Number of people who fished yesterday
  today : ℕ              -- Number of people fishing today

/-- Calculates the number of people who will fish tomorrow given a FishingSchedule -/
def tomorrowFishers (schedule : FishingSchedule) : ℕ :=
  schedule.daily +
  schedule.everyThreeDay +
  (schedule.everyOtherDay - (schedule.yesterday - schedule.daily))

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow 
  (schedule : FishingSchedule)
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterday = 12)
  (h5 : schedule.today = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

#eval tomorrowFishers { daily := 7, everyOtherDay := 8, everyThreeDay := 3, yesterday := 12, today := 10 }

end NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l3051_305131


namespace NUMINAMATH_CALUDE_points_per_round_l3051_305102

theorem points_per_round (total_rounds : ℕ) (total_points : ℕ) 
  (h1 : total_rounds = 177)
  (h2 : total_points = 8142) : 
  total_points / total_rounds = 46 := by
  sorry

end NUMINAMATH_CALUDE_points_per_round_l3051_305102


namespace NUMINAMATH_CALUDE_fraction_always_defined_l3051_305119

theorem fraction_always_defined (x : ℝ) : (x^2 + 2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_fraction_always_defined_l3051_305119


namespace NUMINAMATH_CALUDE_at_least_thirty_percent_have_all_colors_l3051_305132

/-- Represents the distribution of flags among children -/
structure FlagDistribution where
  total_children : ℕ
  blue_percentage : ℚ
  red_percentage : ℚ
  green_percentage : ℚ

/-- Conditions for the flag distribution problem -/
def valid_distribution (d : FlagDistribution) : Prop :=
  d.blue_percentage = 55 / 100 ∧
  d.red_percentage = 45 / 100 ∧
  d.green_percentage = 30 / 100 ∧
  (d.total_children * 3) % 2 = 0 ∧
  d.blue_percentage + d.red_percentage + d.green_percentage ≥ 1

/-- The main theorem stating that at least 30% of children have all three colors -/
theorem at_least_thirty_percent_have_all_colors (d : FlagDistribution) 
  (h : valid_distribution d) : 
  ∃ (all_colors_percentage : ℚ), 
    all_colors_percentage ≥ 30 / 100 ∧ 
    all_colors_percentage ≤ d.blue_percentage ∧
    all_colors_percentage ≤ d.red_percentage ∧
    all_colors_percentage ≤ d.green_percentage :=
sorry

end NUMINAMATH_CALUDE_at_least_thirty_percent_have_all_colors_l3051_305132


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_200_100_l3051_305128

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Predicate to check if a number is prime -/
def isPrime (p : ℕ) : Prop := sorry

/-- The largest 2-digit prime factor of (200 choose 100) -/
def largestTwoDigitPrimeFactor : ℕ := 61

theorem largest_two_digit_prime_factor_of_binomial_200_100 :
  ∀ p : ℕ, 
    10 ≤ p → p < 100 → isPrime p → 
    p ∣ binomial 200 100 →
    p ≤ largestTwoDigitPrimeFactor ∧
    isPrime largestTwoDigitPrimeFactor ∧
    largestTwoDigitPrimeFactor ∣ binomial 200 100 := by
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_200_100_l3051_305128


namespace NUMINAMATH_CALUDE_largest_non_representable_l3051_305115

def is_representable (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), n = 15 * x + 18 * y + 20 * z

theorem largest_non_representable : 
  (∀ m > 97, is_representable m) ∧ ¬(is_representable 97) :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_l3051_305115


namespace NUMINAMATH_CALUDE_dana_friday_hours_l3051_305107

/-- Dana's hourly rate in dollars -/
def hourly_rate : ℕ := 13

/-- Hours worked on Saturday -/
def saturday_hours : ℕ := 10

/-- Hours worked on Sunday -/
def sunday_hours : ℕ := 3

/-- Total earnings for all three days in dollars -/
def total_earnings : ℕ := 286

/-- Calculates the number of hours worked on Friday -/
def friday_hours : ℕ :=
  (total_earnings - (hourly_rate * (saturday_hours + sunday_hours))) / hourly_rate

theorem dana_friday_hours :
  friday_hours = 9 := by
  sorry

end NUMINAMATH_CALUDE_dana_friday_hours_l3051_305107


namespace NUMINAMATH_CALUDE_tony_winnings_l3051_305179

/-- Calculates the winnings for a single lottery ticket -/
def ticket_winnings (winning_numbers : ℕ) : ℕ :=
  if winning_numbers ≤ 2 then
    15 * winning_numbers
  else
    30 + 20 * (winning_numbers - 2)

/-- Represents Tony's lottery tickets and calculates total winnings -/
def total_winnings : ℕ :=
  ticket_winnings 3 + ticket_winnings 5 + ticket_winnings 2 + ticket_winnings 4

/-- Theorem stating that Tony's total winnings are $240 -/
theorem tony_winnings : total_winnings = 240 := by
  sorry

end NUMINAMATH_CALUDE_tony_winnings_l3051_305179


namespace NUMINAMATH_CALUDE_power_set_of_S_l3051_305136

def S : Set ℕ := {0, 1}

theorem power_set_of_S :
  𝒫 S = {∅, {0}, {1}, {0, 1}} := by
  sorry

end NUMINAMATH_CALUDE_power_set_of_S_l3051_305136


namespace NUMINAMATH_CALUDE_digit_equation_solution_l3051_305161

theorem digit_equation_solution :
  ∀ (A M C : ℕ),
    A ≤ 9 → M ≤ 9 → C ≤ 9 →
    (100 * A + 10 * M + C) * (2 * (A + M + C + 1)) = 4010 →
    A = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l3051_305161


namespace NUMINAMATH_CALUDE_lemonade_sale_duration_l3051_305198

/-- 
Given that Stanley sells 4 cups of lemonade per hour and Carl sells 7 cups per hour,
prove that they sold lemonade for 3 hours if Carl sold 9 more cups than Stanley.
-/
theorem lemonade_sale_duration : ∃ h : ℕ, h > 0 ∧ 7 * h = 4 * h + 9 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_sale_duration_l3051_305198


namespace NUMINAMATH_CALUDE_cube_property_l3051_305154

theorem cube_property : ∃! (n : ℕ), n > 0 ∧ ∃ (k : ℕ), n^3 + 2*n^2 + 9*n + 8 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_property_l3051_305154


namespace NUMINAMATH_CALUDE_selene_and_tanya_spend_16_l3051_305122

/-- Represents the prices of items in the school canteen -/
structure CanteenPrices where
  sandwich : ℕ
  hamburger : ℕ
  hotdog : ℕ
  fruitJuice : ℕ

/-- Represents an order in the canteen -/
structure Order where
  sandwiches : ℕ
  hamburgers : ℕ
  hotdogs : ℕ
  fruitJuices : ℕ

/-- Calculates the total cost of an order given the prices -/
def orderCost (prices : CanteenPrices) (order : Order) : ℕ :=
  prices.sandwich * order.sandwiches +
  prices.hamburger * order.hamburgers +
  prices.hotdog * order.hotdogs +
  prices.fruitJuice * order.fruitJuices

/-- The main theorem stating that Selene and Tanya spend $16 together -/
theorem selene_and_tanya_spend_16 (prices : CanteenPrices) 
    (seleneOrder : Order) (tanyaOrder : Order) : 
    prices.sandwich = 2 → 
    prices.hamburger = 2 → 
    prices.hotdog = 1 → 
    prices.fruitJuice = 2 → 
    seleneOrder.sandwiches = 3 → 
    seleneOrder.fruitJuices = 1 → 
    seleneOrder.hamburgers = 0 → 
    seleneOrder.hotdogs = 0 → 
    tanyaOrder.hamburgers = 2 → 
    tanyaOrder.fruitJuices = 2 → 
    tanyaOrder.sandwiches = 0 → 
    tanyaOrder.hotdogs = 0 → 
    orderCost prices seleneOrder + orderCost prices tanyaOrder = 16 := by
  sorry

end NUMINAMATH_CALUDE_selene_and_tanya_spend_16_l3051_305122


namespace NUMINAMATH_CALUDE_shanmukham_purchase_l3051_305134

/-- Calculates the final amount to pay for goods given the original price, rebate percentage, and sales tax percentage. -/
def finalAmount (originalPrice rebatePercentage salesTaxPercentage : ℚ) : ℚ :=
  let priceAfterRebate := originalPrice * (1 - rebatePercentage / 100)
  let salesTax := priceAfterRebate * (salesTaxPercentage / 100)
  priceAfterRebate + salesTax

/-- Theorem stating that given the specific conditions, the final amount to pay is 6876.10 -/
theorem shanmukham_purchase :
  finalAmount 6650 6 10 = 6876.1 := by
  sorry

end NUMINAMATH_CALUDE_shanmukham_purchase_l3051_305134


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3051_305173

theorem gcd_of_specific_numbers : Nat.gcd 333333333 666666666 = 333333333 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3051_305173


namespace NUMINAMATH_CALUDE_acid_mixture_concentration_exists_l3051_305139

theorem acid_mixture_concentration_exists :
  ∃! P : ℝ, ∃ a w : ℝ,
    a > 0 ∧ w > 0 ∧
    (a / (a + w + 2)) * 100 = 30 ∧
    ((a + 1) / (a + w + 3)) * 100 = 40 ∧
    (a / (a + w)) * 100 = P ∧
    (P = 50 ∨ P = 52 ∨ P = 55 ∨ P = 57 ∨ P = 60) :=
by sorry

end NUMINAMATH_CALUDE_acid_mixture_concentration_exists_l3051_305139


namespace NUMINAMATH_CALUDE_book_cost_problem_l3051_305150

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) 
  (h1 : total_cost = 420)
  (h2 : loss_percent = 0.15)
  (h3 : gain_percent = 0.19)
  (h4 : ∃ (sell_price : ℝ), 
    sell_price = (1 - loss_percent) * (total_cost - x) ∧ 
    sell_price = (1 + gain_percent) * x) : 
  ∃ (x : ℝ), x = 245 ∧ x + (total_cost - x) = total_cost := by
sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3051_305150


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3051_305117

theorem min_value_sum_squares (x y z : ℝ) (h : x - 2*y - 3*z = 4) :
  ∃ (m : ℝ), m = 8/7 ∧ (∀ x y z : ℝ, x - 2*y - 3*z = 4 → x^2 + y^2 + z^2 ≥ m) ∧
  (∃ x y z : ℝ, x - 2*y - 3*z = 4 ∧ x^2 + y^2 + z^2 = m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3051_305117


namespace NUMINAMATH_CALUDE_sum_of_square_root_differences_l3051_305126

theorem sum_of_square_root_differences (S : ℝ) : 
  S = 1 / (4 - Real.sqrt 9) - 1 / (Real.sqrt 9 - Real.sqrt 8) + 
      1 / (Real.sqrt 8 - Real.sqrt 7) - 1 / (Real.sqrt 7 - Real.sqrt 6) + 
      1 / (Real.sqrt 6 - 3) → 
  S = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_square_root_differences_l3051_305126


namespace NUMINAMATH_CALUDE_green_balls_count_l3051_305100

theorem green_balls_count (red : ℕ) (blue : ℕ) (prob : ℚ) (green : ℕ) : 
  red = 3 → 
  blue = 2 → 
  prob = 1/12 → 
  (red : ℚ)/(red + blue + green : ℚ) * ((red - 1 : ℚ)/(red + blue + green - 1 : ℚ)) = prob → 
  green = 4 :=
by sorry

end NUMINAMATH_CALUDE_green_balls_count_l3051_305100


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3051_305129

theorem complex_fraction_simplification :
  (5 + 12 * Complex.I) / (2 - 3 * Complex.I) = -2 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3051_305129


namespace NUMINAMATH_CALUDE_handshake_count_l3051_305106

theorem handshake_count (n : ℕ) (h : n = 7) : (n * (n - 1)) / 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l3051_305106


namespace NUMINAMATH_CALUDE_sum_bounds_l3051_305190

theorem sum_bounds (a b c d e : ℝ) :
  0 < (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) ∧
  (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) < 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_bounds_l3051_305190
