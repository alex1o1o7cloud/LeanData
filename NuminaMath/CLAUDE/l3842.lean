import Mathlib

namespace NUMINAMATH_CALUDE_no_integer_solutions_to_equation_l3842_384298

theorem no_integer_solutions_to_equation :
  ¬∃ (w x y z : ℤ), (5 : ℝ)^w + (5 : ℝ)^x = (7 : ℝ)^y + (7 : ℝ)^z :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_to_equation_l3842_384298


namespace NUMINAMATH_CALUDE_mean_temperature_is_87_l3842_384222

def temperatures : List ℝ := [84, 86, 85, 87, 89, 90, 88]

theorem mean_temperature_is_87 :
  (temperatures.sum / temperatures.length : ℝ) = 87 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_87_l3842_384222


namespace NUMINAMATH_CALUDE_sports_club_members_l3842_384237

theorem sports_club_members (B T Both Neither : ℕ) 
  (hB : B = 17)
  (hT : T = 19)
  (hBoth : Both = 11)
  (hNeither : Neither = 2) :
  B + T - Both + Neither = 27 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l3842_384237


namespace NUMINAMATH_CALUDE_fraction_difference_equals_specific_fraction_l3842_384201

theorem fraction_difference_equals_specific_fraction : 
  (3^2 + 5^2 + 7^2) / (2^2 + 4^2 + 6^2) - (2^2 + 4^2 + 6^2) / (3^2 + 5^2 + 7^2) = 3753 / 4656 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_specific_fraction_l3842_384201


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3842_384247

theorem complex_modulus_problem (z : ℂ) (h : z * (2 - Complex.I) = 1 + 7 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3842_384247


namespace NUMINAMATH_CALUDE_three_digit_difference_times_second_largest_l3842_384245

def largest_three_digit (a b c : Nat) : Nat :=
  100 * max a (max b c) + 10 * max (min (max a b) (max b c)) (min a (min b c)) + min a (min b c)

def smallest_three_digit (a b c : Nat) : Nat :=
  100 * min a (min b c) + 10 * min (max (min a b) (min b c)) (max a (max b c)) + max a (max b c)

def second_largest_three_digit (a b c : Nat) : Nat :=
  let max_digit := max a (max b c)
  let min_digit := min a (min b c)
  let mid_digit := a + b + c - max_digit - min_digit
  100 * max_digit + 10 * mid_digit + min_digit

theorem three_digit_difference_times_second_largest (a b c : Nat) 
  (ha : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (hd : a ∈ [2, 5, 8] ∧ b ∈ [2, 5, 8] ∧ c ∈ [2, 5, 8]) : 
  (largest_three_digit a b c - smallest_three_digit a b c) * second_largest_three_digit a b c = 490050 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_difference_times_second_largest_l3842_384245


namespace NUMINAMATH_CALUDE_projections_of_P_l3842_384207

def P : ℝ × ℝ × ℝ := (2, 3, 4)

def projection_planes : List (ℝ × ℝ × ℝ) := [(2, 3, 0), (0, 3, 4), (2, 0, 4)]
def projection_axes : List (ℝ × ℝ × ℝ) := [(2, 0, 0), (0, 3, 0), (0, 0, 4)]

theorem projections_of_P :
  (projection_planes = [(2, 3, 0), (0, 3, 4), (2, 0, 4)]) ∧
  (projection_axes = [(2, 0, 0), (0, 3, 0), (0, 0, 4)]) := by
  sorry

end NUMINAMATH_CALUDE_projections_of_P_l3842_384207


namespace NUMINAMATH_CALUDE_even_function_implies_k_equals_one_l3842_384286

/-- A function f is even if f(-x) = f(x) for all x in its domain. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = kx^2 + (k-1)x + 2. -/
def f (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + (k - 1) * x + 2

/-- If f(x) = kx^2 + (k-1)x + 2 is an even function, then k = 1. -/
theorem even_function_implies_k_equals_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_k_equals_one_l3842_384286


namespace NUMINAMATH_CALUDE_total_inches_paved_before_today_l3842_384233

/-- Represents a road section with its length and completion percentage -/
structure RoadSection where
  length : ℝ
  percentComplete : ℝ

/-- Calculates the total inches repaved before today given three road sections and additional inches repaved today -/
def totalInchesPavedBeforeToday (sectionA sectionB sectionC : RoadSection) (additionalInches : ℝ) : ℝ :=
  sectionA.length * sectionA.percentComplete +
  sectionB.length * sectionB.percentComplete +
  sectionC.length * sectionC.percentComplete

/-- Theorem stating that the total inches repaved before today is 6900 -/
theorem total_inches_paved_before_today :
  let sectionA : RoadSection := { length := 4000, percentComplete := 0.7 }
  let sectionB : RoadSection := { length := 3500, percentComplete := 0.6 }
  let sectionC : RoadSection := { length := 2500, percentComplete := 0.8 }
  let additionalInches : ℝ := 950
  totalInchesPavedBeforeToday sectionA sectionB sectionC additionalInches = 6900 := by
  sorry

end NUMINAMATH_CALUDE_total_inches_paved_before_today_l3842_384233


namespace NUMINAMATH_CALUDE_f_x1_gt_f_x2_l3842_384202

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Axioms based on the given conditions
axiom f_symmetry (x : ℝ) : f (2 - x) = f x

axiom f_derivative_condition (x : ℝ) (h : x ≠ 1) : f' x / (x - 1) < 0

axiom x1_x2_sum (x₁ x₂ : ℝ) : x₁ + x₂ > 2

axiom x1_lt_x2 (x₁ x₂ : ℝ) : x₁ < x₂

-- The theorem to be proved
theorem f_x1_gt_f_x2 (x₁ x₂ : ℝ) : f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_x1_gt_f_x2_l3842_384202


namespace NUMINAMATH_CALUDE_modulus_of_z_l3842_384238

theorem modulus_of_z (z : ℂ) (h : z * (Complex.I + 1) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3842_384238


namespace NUMINAMATH_CALUDE_solution_in_quadrant_I_l3842_384265

theorem solution_in_quadrant_I (c : ℝ) : 
  (∃ x y : ℝ, x - 2*y = 4 ∧ 2*c*x + y = 5 ∧ x > 0 ∧ y > 0) ↔ 
  (-1/4 < c ∧ c < 5/8) :=
sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_I_l3842_384265


namespace NUMINAMATH_CALUDE_pool_filling_time_l3842_384205

theorem pool_filling_time (t1 t2 t_combined : ℝ) : 
  t1 = 8 → t_combined = 4.8 → 1/t1 + 1/t2 = 1/t_combined → t2 = 12 := by
sorry

end NUMINAMATH_CALUDE_pool_filling_time_l3842_384205


namespace NUMINAMATH_CALUDE_min_value_of_g_l3842_384290

theorem min_value_of_g (φ : Real) (h1 : 0 < φ) (h2 : φ < π) : 
  let f := fun x => Real.sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)
  let g := fun x => f (x - 3 * π / 4)
  (∀ y, f (π / 12 - y) = f (π / 12 + y)) →
  (∃ x ∈ Set.Icc (-π / 4) (π / 6), g x = -1) ∧
  (∀ x ∈ Set.Icc (-π / 4) (π / 6), g x ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_g_l3842_384290


namespace NUMINAMATH_CALUDE_eighteen_tons_equals_18000kg_l3842_384267

-- Define the conversion factor between tons and kilograms
def tons_to_kg (t : ℝ) : ℝ := 1000 * t

-- Theorem statement
theorem eighteen_tons_equals_18000kg : tons_to_kg 18 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_tons_equals_18000kg_l3842_384267


namespace NUMINAMATH_CALUDE_divisible_by_3_or_5_count_l3842_384258

def count_divisible (n : Nat) : Nat :=
  (n / 3) + (n / 5) - (n / 15)

theorem divisible_by_3_or_5_count : count_divisible 46 = 21 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_3_or_5_count_l3842_384258


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3842_384285

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3842_384285


namespace NUMINAMATH_CALUDE_apple_ratio_problem_l3842_384241

theorem apple_ratio_problem (green_apples red_apples : ℕ) : 
  (green_apples : ℚ) / red_apples = 5 / 3 → 
  green_apples = 15 → 
  red_apples = 9 := by
sorry

end NUMINAMATH_CALUDE_apple_ratio_problem_l3842_384241


namespace NUMINAMATH_CALUDE_sandwich_count_l3842_384261

def num_meats : ℕ := 12
def num_cheeses : ℕ := 8
def num_toppings : ℕ := 5

def sandwich_combinations : ℕ := num_meats * (num_cheeses.choose 2) * num_toppings

theorem sandwich_count : sandwich_combinations = 1680 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_count_l3842_384261


namespace NUMINAMATH_CALUDE_part_one_part_two_l3842_384214

-- Part 1
theorem part_one : Real.sqrt 9 + 2 * Real.sin (30 * π / 180) - 1 = 3 := by sorry

-- Part 2
theorem part_two : 
  ∀ x : ℝ, (2*x - 3)^2 = 2*(2*x - 3) ↔ x = 3/2 ∨ x = 5/2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3842_384214


namespace NUMINAMATH_CALUDE_parabola_tangent_angle_sine_l3842_384262

/-- Given a parabola x^2 = 4y with focus F(0, 1), and a point A on the parabola where the tangent line has slope 2, 
    prove that the sine of the angle between AF and the tangent line at A is √5/5. -/
theorem parabola_tangent_angle_sine (A : ℝ × ℝ) : 
  let (x, y) := A
  (x^2 = 4*y) →                   -- A is on the parabola
  ((1/2)*x = 2) →                 -- Slope of tangent at A is 2
  let F := (0, 1)                 -- Focus of the parabola
  let slope_AF := (y - 1) / (x - 0)
  let tan_theta := |((1/2)*x - slope_AF) / (1 + (1/2)*x * slope_AF)|
  Real.sqrt (tan_theta^2 / (1 + tan_theta^2)) = Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_angle_sine_l3842_384262


namespace NUMINAMATH_CALUDE_project_completion_theorem_l3842_384277

/-- The number of days it takes to complete the project -/
def project_completion_time (a_time b_time : ℝ) (a_quit_before : ℝ) : ℝ :=
  let a_rate := 1 / a_time
  let b_rate := 1 / b_time
  let combined_rate := a_rate + b_rate
  15

/-- Theorem stating that the project will be completed in 15 days -/
theorem project_completion_theorem :
  project_completion_time 10 30 10 = 15 := by
  sorry

#eval project_completion_time 10 30 10

end NUMINAMATH_CALUDE_project_completion_theorem_l3842_384277


namespace NUMINAMATH_CALUDE_solution_set_f_leq_3x_plus_4_range_of_m_for_f_geq_m_all_reals_l3842_384234

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- Theorem for the solution set of f(x) ≤ 3x + 4
theorem solution_set_f_leq_3x_plus_4 :
  {x : ℝ | f x ≤ 3 * x + 4} = {x : ℝ | x ≥ 0} :=
sorry

-- Theorem for the range of m
theorem range_of_m_for_f_geq_m_all_reals (m : ℝ) :
  ({x : ℝ | f x ≥ m} = Set.univ) ↔ m ∈ Set.Iic 4 :=
sorry

#check solution_set_f_leq_3x_plus_4
#check range_of_m_for_f_geq_m_all_reals

end NUMINAMATH_CALUDE_solution_set_f_leq_3x_plus_4_range_of_m_for_f_geq_m_all_reals_l3842_384234


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3842_384289

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote of C
def asymptote (x y : ℝ) : Prop :=
  y = (Real.sqrt 5 / 2) * x

-- Define the ellipse that shares a focus with C
def ellipse (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y, asymptote x y → hyperbola a b x y) ∧
  (∃ x y, ellipse x y ∧ hyperbola a b x y) →
  a^2 = 4 ∧ b^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3842_384289


namespace NUMINAMATH_CALUDE_total_profit_is_60000_l3842_384203

/-- Calculates the total profit of a partnership given the investments and one partner's share of the profit -/
def calculate_total_profit (a_investment b_investment c_investment : ℕ) (c_profit : ℕ) : ℕ :=
  let total_parts := a_investment / 9000 + b_investment / 9000 + c_investment / 9000
  let c_parts := c_investment / 9000
  let profit_per_part := c_profit / c_parts
  total_parts * profit_per_part

/-- Proves that the total profit is $60,000 given the specific investments and c's profit share -/
theorem total_profit_is_60000 :
  calculate_total_profit 45000 63000 72000 24000 = 60000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_60000_l3842_384203


namespace NUMINAMATH_CALUDE_described_loop_is_while_loop_l3842_384251

/-- Represents a generic loop structure -/
structure LoopStructure :=
  (condition_evaluation : Bool)
  (execution_order : Bool)

/-- Defines a While loop structure -/
def is_while_loop (loop : LoopStructure) : Prop :=
  loop.condition_evaluation ∧ loop.execution_order

/-- Theorem stating that the described loop structure is a While loop -/
theorem described_loop_is_while_loop :
  ∀ (loop : LoopStructure),
  loop.condition_evaluation = true ∧
  loop.execution_order = true →
  is_while_loop loop :=
by
  sorry

#check described_loop_is_while_loop

end NUMINAMATH_CALUDE_described_loop_is_while_loop_l3842_384251


namespace NUMINAMATH_CALUDE_invertible_function_problem_l3842_384275

theorem invertible_function_problem (g : ℝ → ℝ) (c : ℝ) 
  (h_invertible : Function.Bijective g)
  (h_gc : g c = 3)
  (h_g3 : g 3 = 5) :
  c - 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_invertible_function_problem_l3842_384275


namespace NUMINAMATH_CALUDE_bike_shop_profit_is_3000_l3842_384228

/-- Calculates the profit of a bike shop given various parameters. -/
def bike_shop_profit (tire_repair_charge : ℕ) (tire_repair_cost : ℕ) (tire_repairs : ℕ)
                     (complex_repair_charge : ℕ) (complex_repair_cost : ℕ) (complex_repairs : ℕ)
                     (retail_profit : ℕ) (fixed_expenses : ℕ) : ℕ :=
  (tire_repairs * (tire_repair_charge - tire_repair_cost)) +
  (complex_repairs * (complex_repair_charge - complex_repair_cost)) +
  retail_profit - fixed_expenses

/-- Theorem stating that the bike shop's profit is $3000 under given conditions. -/
theorem bike_shop_profit_is_3000 :
  bike_shop_profit 20 5 300 300 50 2 2000 4000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_bike_shop_profit_is_3000_l3842_384228


namespace NUMINAMATH_CALUDE_egg_difference_l3842_384206

/-- Given that Megan bought 2 dozen eggs, 3 eggs broke, and twice as many cracked,
    prove that the difference between the eggs in perfect condition and those that are cracked is 9. -/
theorem egg_difference (total : ℕ) (broken : ℕ) (cracked : ℕ) :
  total = 2 * 12 →
  broken = 3 →
  cracked = 2 * broken →
  total - broken - cracked - cracked = 9 :=
by sorry

end NUMINAMATH_CALUDE_egg_difference_l3842_384206


namespace NUMINAMATH_CALUDE_min_value_sum_product_l3842_384218

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l3842_384218


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3842_384271

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧ (∀ m : ℕ, m < n → ∃ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3842_384271


namespace NUMINAMATH_CALUDE_highlight_film_average_time_l3842_384279

/-- Given the footage times for 5 players, prove that the average time each player gets in the highlight film is 2 minutes -/
theorem highlight_film_average_time (point_guard shooting_guard small_forward power_forward center : ℕ) 
  (h1 : point_guard = 130)
  (h2 : shooting_guard = 145)
  (h3 : small_forward = 85)
  (h4 : power_forward = 60)
  (h5 : center = 180) :
  (point_guard + shooting_guard + small_forward + power_forward + center) / (5 * 60) = 2 := by
  sorry

end NUMINAMATH_CALUDE_highlight_film_average_time_l3842_384279


namespace NUMINAMATH_CALUDE_tricolor_triangles_odd_l3842_384248

/-- Represents the color of a point -/
inductive Color
| Red
| Yellow
| Blue

/-- Represents a point in the triangle -/
structure Point where
  color : Color

/-- Represents a triangle ABC with m interior points -/
structure ColoredTriangle where
  m : ℕ
  A : Point
  B : Point
  C : Point
  interior_points : Fin m → Point

/-- A function that counts the number of triangles with vertices of all different colors -/
def count_tricolor_triangles (t : ColoredTriangle) : ℕ := sorry

/-- The main theorem stating that the number of triangles with vertices of all different colors is odd -/
theorem tricolor_triangles_odd (t : ColoredTriangle) 
  (h1 : t.A.color = Color.Red)
  (h2 : t.B.color = Color.Yellow)
  (h3 : t.C.color = Color.Blue) :
  Odd (count_tricolor_triangles t) := by sorry

end NUMINAMATH_CALUDE_tricolor_triangles_odd_l3842_384248


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3842_384263

theorem arithmetic_calculations :
  (2 / 19 * 8 / 25 + 17 / 25 / (19 / 2) = 2 / 19) ∧
  (1 / 4 * 125 * 1 / 25 * 8 = 10) ∧
  ((1 / 3 + 1 / 4) / (1 / 2 - 1 / 3) = 7 / 2) ∧
  ((1 / 6 + 1 / 8) * 24 * 1 / 9 = 7 / 9) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3842_384263


namespace NUMINAMATH_CALUDE_max_y_over_x_l3842_384291

theorem max_y_over_x (x y : ℝ) (h : x^2 + y^2 - 6*x - 6*y + 12 = 0) :
  ∃ (max : ℝ), max = 3 + 2 * Real.sqrt 2 ∧ 
    ∀ (x' y' : ℝ), x'^2 + y'^2 - 6*x' - 6*y' + 12 = 0 → y' / x' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_y_over_x_l3842_384291


namespace NUMINAMATH_CALUDE_parallel_tangent_implies_a_le_one_l3842_384246

open Real

/-- The function f(x) = ln x + (1/2)x^2 + ax has a tangent line parallel to 3x - y = 0 for some x > 0 -/
def has_parallel_tangent (a : ℝ) : Prop :=
  ∃ x > 0, (1 / x) + x + a = 3

/-- Theorem: If f(x) has a tangent line parallel to 3x - y = 0, then a ≤ 1 -/
theorem parallel_tangent_implies_a_le_one (a : ℝ) (h : has_parallel_tangent a) : a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangent_implies_a_le_one_l3842_384246


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3842_384256

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) (d : ℤ) (h : arithmetic_sequence a d) :
  (a 5 = -1 ∧ a 8 = 2 → a 1 = -5 ∧ d = 1) ∧
  (a 1 + a 6 = 12 ∧ a 4 = 7 → a 9 = 17) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3842_384256


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3842_384295

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → (6 * s^2 = 54) → s^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3842_384295


namespace NUMINAMATH_CALUDE_one_is_hilbert_number_h_hilbert_formula_larger_h_hilbert_number_l3842_384242

-- Definition of a Hilbert number
def is_hilbert_number (p : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p = x^2 + y^2 - x*y

-- Definition of an H Hilbert number
def is_h_hilbert_number (p : ℕ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ p = (2*n - 1)^2 + (2*n + 1)^2 - (2*n - 1)*(2*n + 1)

-- Theorem statements
theorem one_is_hilbert_number : is_hilbert_number 1 := by sorry

theorem h_hilbert_formula (n : ℕ) (h : n > 0) : 
  is_h_hilbert_number (4*n^2 + 3) := by sorry

theorem larger_h_hilbert_number (m n : ℕ) (hm : m > 0) (hn : n > 0) (h_diff : 4*n^2 + 3 - (4*m^2 + 3) = 48) :
  4*n^2 + 3 = 67 := by sorry

end NUMINAMATH_CALUDE_one_is_hilbert_number_h_hilbert_formula_larger_h_hilbert_number_l3842_384242


namespace NUMINAMATH_CALUDE_functional_eq_solution_l3842_384293

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = max (f (x + y)) (f x * f y)

/-- The main theorem stating that any function satisfying the functional equation
    must be constant with values between 0 and 1, inclusive -/
theorem functional_eq_solution (f : ℝ → ℝ) (h : SatisfiesFunctionalEq f) :
    ∃ c : ℝ, (0 ≤ c ∧ c ≤ 1) ∧ (∀ x : ℝ, f x = c) :=
  sorry

end NUMINAMATH_CALUDE_functional_eq_solution_l3842_384293


namespace NUMINAMATH_CALUDE_no_fixed_points_l3842_384225

/-- Definition of a fixed point for a function f -/
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = x

/-- The specific function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ :=
  x^2 + 1

/-- Theorem: f(x) = x^2 + 1 has no fixed points -/
theorem no_fixed_points : ¬∃ x : ℝ, is_fixed_point f x := by
  sorry

end NUMINAMATH_CALUDE_no_fixed_points_l3842_384225


namespace NUMINAMATH_CALUDE_berry_temperature_theorem_l3842_384287

theorem berry_temperature_theorem (temps : List Float) (avg : Float) : 
  temps.length = 6 ∧ 
  temps = [99.1, 98.2, 98.7, 99.3, 99, 98.9] ∧ 
  avg = 99 →
  (temps.sum + 99.8) / 7 = avg :=
by sorry

end NUMINAMATH_CALUDE_berry_temperature_theorem_l3842_384287


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l3842_384273

/-- The number of points on the circle -/
def num_points : ℕ := 8

/-- The number of chords to be selected -/
def num_selected_chords : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := num_points.choose 2

/-- The number of ways to select the chords -/
def ways_to_select_chords : ℕ := total_chords.choose num_selected_chords

/-- The number of convex quadrilaterals that can be formed -/
def convex_quadrilaterals : ℕ := num_points.choose 4

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := convex_quadrilaterals / ways_to_select_chords

theorem convex_quadrilateral_probability : probability = 2 / 585 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l3842_384273


namespace NUMINAMATH_CALUDE_total_path_length_is_5pi_sqrt34_l3842_384208

/-- Rectangle ABCD with given dimensions -/
structure Rectangle where
  AB : ℝ
  BC : ℝ

/-- Rotation parameters -/
structure RotationParams where
  firstAngle : ℝ  -- in radians
  secondAngle : ℝ  -- in radians

/-- Calculate the total path length of point A during rotations -/
def totalPathLength (rect : Rectangle) (rotParams : RotationParams) : ℝ :=
  sorry

/-- Theorem: The total path length of point A is 5π × √34 -/
theorem total_path_length_is_5pi_sqrt34 (rect : Rectangle) (rotParams : RotationParams) :
  rect.AB = 3 → rect.BC = 5 → rotParams.firstAngle = π → rotParams.secondAngle = 3 * π / 2 →
  totalPathLength rect rotParams = 5 * π * Real.sqrt 34 :=
sorry

end NUMINAMATH_CALUDE_total_path_length_is_5pi_sqrt34_l3842_384208


namespace NUMINAMATH_CALUDE_work_completion_time_l3842_384253

-- Define the work rates for each person
def amit_rate : ℚ := 1 / 15
def ananthu_rate : ℚ := 1 / 90
def chandra_rate : ℚ := 1 / 45

-- Define the number of days each person worked alone
def amit_solo_days : ℕ := 3
def ananthu_solo_days : ℕ := 6

-- Define the combined work rate of all three people
def combined_rate : ℚ := amit_rate + ananthu_rate + chandra_rate

-- Theorem statement
theorem work_completion_time : 
  let work_done_solo := amit_rate * amit_solo_days + ananthu_rate * ananthu_solo_days
  let remaining_work := 1 - work_done_solo
  let days_together := (remaining_work / combined_rate).ceil
  amit_solo_days + ananthu_solo_days + days_together = 17 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3842_384253


namespace NUMINAMATH_CALUDE_sum_first_100_base6_l3842_384294

/-- Represents a number in base 6 --/
def Base6 := Nat

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : Nat) : Base6 := sorry

/-- Adds two numbers in base 6 --/
def addBase6 (a b : Base6) : Base6 := sorry

/-- Multiplies two numbers in base 6 --/
def mulBase6 (a b : Base6) : Base6 := sorry

/-- Divides two numbers in base 6 --/
def divBase6 (a b : Base6) : Base6 := sorry

/-- Computes the sum of the first n (in base 6) natural numbers in base 6 --/
def sumFirstNBase6 (n : Base6) : Base6 := sorry

theorem sum_first_100_base6 :
  sumFirstNBase6 (toBase6 100) = toBase6 7222 := by sorry

end NUMINAMATH_CALUDE_sum_first_100_base6_l3842_384294


namespace NUMINAMATH_CALUDE_prank_combinations_l3842_384260

theorem prank_combinations (monday tuesday wednesday thursday friday saturday sunday : ℕ) :
  monday = 1 →
  tuesday = 2 →
  wednesday = 6 →
  thursday = 5 →
  friday = 0 →
  saturday = 2 →
  sunday = 1 →
  monday * tuesday * wednesday * thursday * friday * saturday * sunday = 0 := by
sorry

end NUMINAMATH_CALUDE_prank_combinations_l3842_384260


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3842_384255

theorem quadratic_equation_roots (k : ℝ) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 3 * x₁^2 + k * x₁ = 5 ∧ 3 * x₂^2 + k * x₂ = 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3842_384255


namespace NUMINAMATH_CALUDE_count_grid_paths_l3842_384249

/-- The number of paths from (0,0) to (m,n) on a grid, moving only right or up -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- Theorem: The number of distinct paths from the bottom-left corner to the top-right corner
    of an m × n grid, moving only upward or to the right, is equal to (m+n choose m) -/
theorem count_grid_paths (m n : ℕ) : 
  grid_paths m n = Nat.choose (m + n) m := by sorry

end NUMINAMATH_CALUDE_count_grid_paths_l3842_384249


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3842_384240

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 5, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![2, 6; -1, 3]

theorem matrix_multiplication_result :
  A * B = !![7, 15; 8, 36] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3842_384240


namespace NUMINAMATH_CALUDE_certain_number_proof_l3842_384220

theorem certain_number_proof (x : ℝ) : x * 2.13 = 0.3408 → x = 0.1600 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3842_384220


namespace NUMINAMATH_CALUDE_similar_triangle_scaling_l3842_384272

theorem similar_triangle_scaling (base1 height1 base2 : ℝ) (height2 : ℝ) : 
  base1 = 12 → height1 = 6 → base2 = 8 → 
  (base1 / height1 = base2 / height2) → 
  height2 = 4 := by sorry

end NUMINAMATH_CALUDE_similar_triangle_scaling_l3842_384272


namespace NUMINAMATH_CALUDE_intersection_angle_l3842_384250

-- Define the lines
def line1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0
def line2 (x : ℝ) : Prop := x + 5 = 0

-- Define the angle between the lines
def angle_between_lines : ℝ := 30

-- Theorem statement
theorem intersection_angle :
  ∃ (x y : ℝ), line1 x y ∧ line2 x → angle_between_lines = 30 := by sorry

end NUMINAMATH_CALUDE_intersection_angle_l3842_384250


namespace NUMINAMATH_CALUDE_min_value_a_l3842_384292

theorem min_value_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ (x y : ℝ), x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 25) : 
  a ≥ 16 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) * (1/x + (16 - ε)/y) < 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l3842_384292


namespace NUMINAMATH_CALUDE_biloca_path_theorem_l3842_384269

/-- Represents the dimensions and paths of ants on a tiled floor -/
structure AntPaths where
  diagonal_length : ℝ
  tile_width : ℝ
  tile_length : ℝ
  pipoca_path : ℝ
  tonica_path : ℝ
  cotinha_path : ℝ

/-- Calculates the length of Biloca's path -/
def biloca_path_length (ap : AntPaths) : ℝ :=
  3 * ap.diagonal_length + 4 * ap.tile_width + 2 * ap.tile_length

/-- Theorem stating the length of Biloca's path -/
theorem biloca_path_theorem (ap : AntPaths) 
  (h1 : ap.pipoca_path = 5 * ap.diagonal_length)
  (h2 : ap.pipoca_path = 25)
  (h3 : ap.tonica_path = 5 * ap.diagonal_length + 4 * ap.tile_width)
  (h4 : ap.tonica_path = 37)
  (h5 : ap.cotinha_path = 5 * ap.tile_length + 4 * ap.tile_width)
  (h6 : ap.cotinha_path = 32) :
  biloca_path_length ap = 35 := by
  sorry


end NUMINAMATH_CALUDE_biloca_path_theorem_l3842_384269


namespace NUMINAMATH_CALUDE_arrangement_count_l3842_384224

def number_of_arrangements (black red blue : ℕ) : ℕ :=
  Nat.factorial (black + red + blue) / (Nat.factorial black * Nat.factorial red * Nat.factorial blue)

theorem arrangement_count :
  number_of_arrangements 2 3 4 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3842_384224


namespace NUMINAMATH_CALUDE_total_payment_example_l3842_384209

/-- Calculates the total amount paid for a meal including sales tax and tip -/
def total_payment (meal_cost : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  meal_cost * (1 + sales_tax_rate + tip_rate)

/-- Theorem: The total payment for a $100 meal with 4% sales tax and 6% tip is $110 -/
theorem total_payment_example : total_payment 100 0.04 0.06 = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_example_l3842_384209


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_circles_l3842_384296

theorem smallest_distance_between_complex_circles
  (z w : ℂ)
  (hz : Complex.abs (z - (2 + 2*Complex.I)) = 2)
  (hw : Complex.abs (w + (3 + 5*Complex.I)) = 4) :
  Complex.abs (z - w) ≥ Real.sqrt 74 - 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_circles_l3842_384296


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l3842_384299

theorem smallest_n_for_sqrt_difference : ∃ n : ℕ+, (∀ m : ℕ+, m < n → Real.sqrt m.val - Real.sqrt (m.val - 1) ≥ 0.1) ∧ (Real.sqrt n.val - Real.sqrt (n.val - 1) < 0.1) ∧ n = 26 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_l3842_384299


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3842_384264

theorem triangle_third_side_length 
  (a b x : ℕ) 
  (ha : a = 1) 
  (hb : b = 5) 
  (hx : x > 0) :
  (a + b > x ∧ a + x > b ∧ b + x > a) → x = 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3842_384264


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_domain_eq_reals_l3842_384274

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

theorem domain_eq_reals : Set.range (fun x => |x|) = Set.range (fun x => Real.sqrt (x^2)) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_domain_eq_reals_l3842_384274


namespace NUMINAMATH_CALUDE_gnomon_magic_diagonal_sums_equal_l3842_384280

/-- Represents a 3x3 square --/
def Square := Matrix (Fin 3) (Fin 3) ℝ

/-- Checks if a 3x3 square is gnomon-magic --/
def is_gnomon_magic (s : Square) : Prop :=
  let sum1 := s 1 1 + s 1 2 + s 2 1 + s 2 2
  let sum2 := s 1 2 + s 1 3 + s 2 2 + s 2 3
  let sum3 := s 2 1 + s 2 2 + s 3 1 + s 3 2
  let sum4 := s 2 2 + s 2 3 + s 3 2 + s 3 3
  sum1 = sum2 ∧ sum2 = sum3 ∧ sum3 = sum4

/-- Calculates the sum of the main diagonal --/
def main_diagonal_sum (s : Square) : ℝ :=
  s 1 1 + s 2 2 + s 3 3

/-- Calculates the sum of the anti-diagonal --/
def anti_diagonal_sum (s : Square) : ℝ :=
  s 1 3 + s 2 2 + s 3 1

/-- Theorem: In a 3x3 gnomon-magic square, the sums of numbers along the two diagonals are equal --/
theorem gnomon_magic_diagonal_sums_equal (s : Square) (h : is_gnomon_magic s) :
  main_diagonal_sum s = anti_diagonal_sum s := by
  sorry

end NUMINAMATH_CALUDE_gnomon_magic_diagonal_sums_equal_l3842_384280


namespace NUMINAMATH_CALUDE_certain_number_proof_l3842_384226

theorem certain_number_proof : ∃ n : ℕ, n - 999 = 9001 ∧ n = 10000 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3842_384226


namespace NUMINAMATH_CALUDE_max_value_of_g_l3842_384232

def g (x : ℝ) := 4 * x - x^4

theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3842_384232


namespace NUMINAMATH_CALUDE_counterexample_existence_l3842_384229

theorem counterexample_existence : ∃ (n : ℕ), ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n + 2)) ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_existence_l3842_384229


namespace NUMINAMATH_CALUDE_lucky_in_thirteen_l3842_384270

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- A number is lucky if the sum of its digits is divisible by 7 -/
def is_lucky (n : ℕ) : Prop :=
  sum_of_digits n % 7 = 0

/-- Main theorem: Any sequence of 13 consecutive natural numbers contains a lucky number -/
theorem lucky_in_thirteen (start : ℕ) : ∃ k : ℕ, k ∈ Finset.range 13 ∧ is_lucky (start + k) := by
  sorry

end NUMINAMATH_CALUDE_lucky_in_thirteen_l3842_384270


namespace NUMINAMATH_CALUDE_cube_sum_positive_l3842_384227

theorem cube_sum_positive (x y z : ℝ) (h1 : x < y) (h2 : y < z) :
  (x - y)^3 + (y - z)^3 + (z - x)^3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_positive_l3842_384227


namespace NUMINAMATH_CALUDE_log_xy_equals_three_fourths_l3842_384200

-- Define x and y as positive real numbers
variable (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Define the given conditions
def condition1 : Prop := Real.log (x^2 * y^4) = 2
def condition2 : Prop := Real.log (x^3 * y^2) = 2

-- State the theorem
theorem log_xy_equals_three_fourths 
  (h1 : condition1 x y) (h2 : condition2 x y) : 
  Real.log (x * y) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_equals_three_fourths_l3842_384200


namespace NUMINAMATH_CALUDE_triangle_inradius_l3842_384252

theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : p = 40) 
  (h2 : A = 50) 
  (h3 : A = r * p / 2) : r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l3842_384252


namespace NUMINAMATH_CALUDE_audrey_dream_fraction_l3842_384215

theorem audrey_dream_fraction (total_sleep : ℝ) (not_dreaming : ℝ) 
  (h1 : total_sleep = 10)
  (h2 : not_dreaming = 6) :
  (total_sleep - not_dreaming) / total_sleep = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_audrey_dream_fraction_l3842_384215


namespace NUMINAMATH_CALUDE_angle2_value_l3842_384204

-- Define the angles
variable (angle1 angle2 angle3 : ℝ)

-- Define the conditions
def complementary (a b : ℝ) : Prop := a + b = 90
def supplementary (a b : ℝ) : Prop := a + b = 180

-- State the theorem
theorem angle2_value (h1 : complementary angle1 angle2)
                     (h2 : supplementary angle1 angle3)
                     (h3 : angle3 = 125) :
  angle2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_angle2_value_l3842_384204


namespace NUMINAMATH_CALUDE_ezekiel_hike_third_day_l3842_384268

/-- Represents a three-day hike --/
structure ThreeDayHike where
  total_distance : ℕ
  day1_distance : ℕ
  day2_distance : ℕ

/-- Calculates the distance covered on the third day of a three-day hike --/
def third_day_distance (hike : ThreeDayHike) : ℕ :=
  hike.total_distance - (hike.day1_distance + hike.day2_distance)

/-- Theorem stating that for the given hike parameters, the third day distance is 22 km --/
theorem ezekiel_hike_third_day :
  let hike : ThreeDayHike := {
    total_distance := 50,
    day1_distance := 10,
    day2_distance := 18
  }
  third_day_distance hike = 22 := by
  sorry


end NUMINAMATH_CALUDE_ezekiel_hike_third_day_l3842_384268


namespace NUMINAMATH_CALUDE_unique_solution_l3842_384219

theorem unique_solution : ∃! (n : ℕ), n > 0 ∧ 5^29 * 4^15 = 2 * n^29 :=
by
  use 10
  constructor
  · sorry -- Proof that 10 satisfies the equation
  · sorry -- Proof of uniqueness

#check unique_solution

end NUMINAMATH_CALUDE_unique_solution_l3842_384219


namespace NUMINAMATH_CALUDE_brenda_mice_problem_l3842_384281

theorem brenda_mice_problem (total_mice : ℕ) : 
  (∃ (given_to_robbie sold_to_store sold_as_feeder remaining : ℕ),
    given_to_robbie = total_mice / 6 ∧
    sold_to_store = 3 * given_to_robbie ∧
    sold_as_feeder = (total_mice - given_to_robbie - sold_to_store) / 2 ∧
    remaining = total_mice - given_to_robbie - sold_to_store - sold_as_feeder ∧
    remaining = 4 ∧
    total_mice % 3 = 0) →
  total_mice / 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_brenda_mice_problem_l3842_384281


namespace NUMINAMATH_CALUDE_cos_495_degrees_l3842_384235

theorem cos_495_degrees : Real.cos (495 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_495_degrees_l3842_384235


namespace NUMINAMATH_CALUDE_dress_price_calculation_l3842_384266

/-- Given a dress with an original price, discount rate, and tax rate, 
    calculate the total selling price after discount and tax. -/
def totalSellingPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let salePrice := originalPrice * (1 - discountRate)
  let taxAmount := salePrice * taxRate
  salePrice + taxAmount

/-- Theorem stating that for a dress with original price $80, 25% discount, 
    and 10% tax, the total selling price is $66. -/
theorem dress_price_calculation :
  totalSellingPrice 80 0.25 0.10 = 66 := by
  sorry

#eval totalSellingPrice 80 0.25 0.10

end NUMINAMATH_CALUDE_dress_price_calculation_l3842_384266


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_plus_constant_l3842_384278

theorem no_solution_absolute_value_plus_constant :
  ∀ x : ℝ, ¬(|5*x| + 7 = 0) :=
sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_plus_constant_l3842_384278


namespace NUMINAMATH_CALUDE_income_comparison_l3842_384236

theorem income_comparison (A B : ℝ) (h : B = A * (1 + 1/3)) : 
  A = B * (1 - 1/4) := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l3842_384236


namespace NUMINAMATH_CALUDE_armands_guessing_game_l3842_384212

theorem armands_guessing_game (x : ℕ) : x = 33 ↔ 3 * x = 2 * 51 - 3 := by
  sorry

end NUMINAMATH_CALUDE_armands_guessing_game_l3842_384212


namespace NUMINAMATH_CALUDE_magical_stack_with_89_fixed_has_266_cards_l3842_384217

/-- Represents a stack of cards -/
structure CardStack :=
  (n : ℕ)
  (is_magical : Bool)
  (card_89_position : ℕ)

/-- Checks if a card stack is magical and card 89 retains its position -/
def is_magical_with_89_fixed (stack : CardStack) : Prop :=
  stack.is_magical ∧ stack.card_89_position = 89

/-- Theorem: A magical stack where card 89 retains its position has 266 cards -/
theorem magical_stack_with_89_fixed_has_266_cards (stack : CardStack) :
  is_magical_with_89_fixed stack → 2 * stack.n = 266 := by
  sorry

#check magical_stack_with_89_fixed_has_266_cards

end NUMINAMATH_CALUDE_magical_stack_with_89_fixed_has_266_cards_l3842_384217


namespace NUMINAMATH_CALUDE_six_boxes_consecutive_green_balls_l3842_384210

/-- The number of ways to fill n boxes with red or green balls, such that at least one box
    contains a green ball and the boxes containing green balls are consecutively numbered. -/
def consecutiveGreenBalls (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

/-- Theorem stating that for 6 boxes, there are 21 ways to fill them under the given conditions. -/
theorem six_boxes_consecutive_green_balls :
  consecutiveGreenBalls 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_six_boxes_consecutive_green_balls_l3842_384210


namespace NUMINAMATH_CALUDE_petyas_friends_l3842_384223

theorem petyas_friends (total_stickers : ℕ) : 
  (∃ (friends : ℕ), total_stickers = 5 * friends + 8 ∧ total_stickers = 6 * friends - 11) →
  (∃ (friends : ℕ), friends = 19 ∧ total_stickers = 5 * friends + 8 ∧ total_stickers = 6 * friends - 11) :=
by sorry

end NUMINAMATH_CALUDE_petyas_friends_l3842_384223


namespace NUMINAMATH_CALUDE_perfume_usage_fraction_l3842_384259

/-- The fraction of perfume used in a cylindrical bottle -/
theorem perfume_usage_fraction 
  (r : ℝ) -- radius of the cylinder base
  (h : ℝ) -- height of the cylinder
  (v_remaining : ℝ) -- volume of remaining perfume in liters
  (hr : r = 7) -- given radius
  (hh : h = 10) -- given height
  (hv : v_remaining = 0.45) -- given remaining volume
  : (π * r^2 * h / 1000 - v_remaining) / (π * r^2 * h / 1000) = (49 * π - 45) / (49 * π) :=
by sorry

end NUMINAMATH_CALUDE_perfume_usage_fraction_l3842_384259


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3842_384213

theorem smallest_sum_of_reciprocals (a b : ℕ+) : 
  a ≠ b → 
  (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 → 
  (∀ c d : ℕ+, c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → (a : ℕ) + (b : ℕ) ≤ (c : ℕ) + (d : ℕ)) →
  (a : ℕ) + (b : ℕ) = 49 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3842_384213


namespace NUMINAMATH_CALUDE_battery_mass_problem_l3842_384243

theorem battery_mass_problem (x y : ℝ) 
  (eq1 : 2 * x + 2 * y = 72)
  (eq2 : 3 * x + 2 * y = 96) :
  x = 24 := by
sorry

end NUMINAMATH_CALUDE_battery_mass_problem_l3842_384243


namespace NUMINAMATH_CALUDE_existence_of_ones_divisible_by_2019_l3842_384254

theorem existence_of_ones_divisible_by_2019 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, k > 0 ∧ (10^n - 1) / 9 = k * 2019) :=
sorry

end NUMINAMATH_CALUDE_existence_of_ones_divisible_by_2019_l3842_384254


namespace NUMINAMATH_CALUDE_savings_percentage_l3842_384231

/-- Represents the financial situation of a man over two years --/
structure FinancialSituation where
  income_year1 : ℝ
  savings_year1 : ℝ
  income_year2 : ℝ
  savings_year2 : ℝ

/-- The financial situation satisfies the given conditions --/
def satisfies_conditions (fs : FinancialSituation) : Prop :=
  fs.income_year1 > 0 ∧
  fs.savings_year1 > 0 ∧
  fs.income_year2 = 1.5 * fs.income_year1 ∧
  fs.savings_year2 = 2 * fs.savings_year1 ∧
  (fs.income_year1 - fs.savings_year1) + (fs.income_year2 - fs.savings_year2) = 2 * (fs.income_year1 - fs.savings_year1)

/-- The theorem stating that the man saved 50% of his income in the first year --/
theorem savings_percentage (fs : FinancialSituation) (h : satisfies_conditions fs) :
  fs.savings_year1 / fs.income_year1 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l3842_384231


namespace NUMINAMATH_CALUDE_letters_in_mailboxes_l3842_384284

/-- The number of ways to distribute n items into k categories -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of letters -/
def num_letters : ℕ := 4

/-- The number of mailboxes -/
def num_mailboxes : ℕ := 3

/-- Theorem: The number of ways to put 4 letters into 3 mailboxes is 81 -/
theorem letters_in_mailboxes :
  distribute num_letters num_mailboxes = 81 := by sorry

end NUMINAMATH_CALUDE_letters_in_mailboxes_l3842_384284


namespace NUMINAMATH_CALUDE_power_product_simplification_l3842_384244

theorem power_product_simplification :
  (-4/5 : ℚ)^2022 * (5/4 : ℚ)^2021 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l3842_384244


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3842_384239

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_negative_2023 :
  reciprocal (-2023) = -1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3842_384239


namespace NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l3842_384216

-- Define the new operation
def star_op (a b : ℝ) : ℝ := a^2 - a*b + b

-- Theorem statement
theorem equation_has_two_distinct_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ star_op x₁ 3 = 5 ∧ star_op x₂ 3 = 5 :=
by sorry

end NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l3842_384216


namespace NUMINAMATH_CALUDE_f_range_l3842_384221

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem f_range : ∀ x : ℝ, f x = -3 * π / 4 ∨ f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l3842_384221


namespace NUMINAMATH_CALUDE_spider_dressing_combinations_l3842_384283

/-- The number of legs of the spider -/
def num_legs : ℕ := 10

/-- The number of socks per leg -/
def socks_per_leg : ℕ := 2

/-- The number of shoes per leg -/
def shoes_per_leg : ℕ := 1

/-- The total number of items to wear -/
def total_items : ℕ := num_legs * (socks_per_leg + shoes_per_leg)

/-- The number of ways to arrange socks on one leg -/
def sock_arrangements_per_leg : ℕ := 2  -- 2! = 2

theorem spider_dressing_combinations :
  (Nat.choose total_items num_legs) * (sock_arrangements_per_leg ^ num_legs) =
  (Nat.factorial total_items) / (Nat.factorial num_legs * Nat.factorial (total_items - num_legs)) * 1024 :=
by sorry

end NUMINAMATH_CALUDE_spider_dressing_combinations_l3842_384283


namespace NUMINAMATH_CALUDE_range_of_f_l3842_384288

-- Define the function f
def f (x : ℝ) : ℝ := |x + 10| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-15) 25 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3842_384288


namespace NUMINAMATH_CALUDE_min_hexagon_area_l3842_384230

/-- Represents a configuration of two intersecting triangles -/
structure IntersectingTriangles where
  /-- The number of finite disjoint regions formed -/
  regions : Nat
  /-- The number of triangular regions -/
  triangular_regions : Nat
  /-- The area of each triangular region -/
  triangle_area : ℝ
  /-- The area of the hexagonal region -/
  hexagon_area : ℝ

/-- Theorem stating the minimum possible area of the hexagonal region -/
theorem min_hexagon_area (config : IntersectingTriangles) :
  config.regions = 7 →
  config.triangular_regions = 6 →
  config.triangle_area = 1 →
  config.hexagon_area ≥ 6 := by
  sorry

#check min_hexagon_area

end NUMINAMATH_CALUDE_min_hexagon_area_l3842_384230


namespace NUMINAMATH_CALUDE_julians_comic_frames_l3842_384297

/-- The number of frames on each page of Julian's comic book -/
def frames_per_page : ℕ := 11

/-- The number of pages in Julian's comic book -/
def total_pages : ℕ := 13

/-- The total number of frames in Julian's comic book -/
def total_frames : ℕ := frames_per_page * total_pages

theorem julians_comic_frames :
  total_frames = 143 := by
  sorry

end NUMINAMATH_CALUDE_julians_comic_frames_l3842_384297


namespace NUMINAMATH_CALUDE_common_root_inequality_l3842_384257

theorem common_root_inequality (a b t : ℝ) (ha : a > 0) (hb : b > 0) (ht : t > 1)
  (eq1 : t^2 + a*t - 100 = 0) (eq2 : t^2 - 200*t + b = 0) : b - a > 100 := by
  sorry

end NUMINAMATH_CALUDE_common_root_inequality_l3842_384257


namespace NUMINAMATH_CALUDE_car_discount_proof_l3842_384211

/-- Given a car's original price and trading conditions, prove the discount percentage. -/
theorem car_discount_proof (P : ℝ) (P_b P_s : ℝ) (h1 : P > 0) (h2 : P_s = 1.60 * P_b) (h3 : P_s = 1.52 * P) : 
  ∃ D : ℝ, D = 0.05 ∧ P_b = P * (1 - D) := by
sorry

end NUMINAMATH_CALUDE_car_discount_proof_l3842_384211


namespace NUMINAMATH_CALUDE_rotten_apples_l3842_384282

/-- Given a problem about apples in crates and boxes, prove the number of rotten apples. -/
theorem rotten_apples (apples_per_crate : ℕ) (num_crates : ℕ) (apples_per_box : ℕ) (num_boxes : ℕ)
  (h1 : apples_per_crate = 42)
  (h2 : num_crates = 12)
  (h3 : apples_per_box = 10)
  (h4 : num_boxes = 50) :
  apples_per_crate * num_crates - apples_per_box * num_boxes = 4 := by
  sorry

#check rotten_apples

end NUMINAMATH_CALUDE_rotten_apples_l3842_384282


namespace NUMINAMATH_CALUDE_clubsuit_equality_theorem_l3842_384276

-- Define the clubsuit operation
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def equality_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | clubsuit p.1 p.2 = clubsuit p.2 p.1}

-- Define the set of points on x-axis, y-axis, y = x, and y = -x
def target_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

theorem clubsuit_equality_theorem : equality_set = target_set := by
  sorry


end NUMINAMATH_CALUDE_clubsuit_equality_theorem_l3842_384276
