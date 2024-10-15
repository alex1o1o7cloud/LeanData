import Mathlib

namespace NUMINAMATH_CALUDE_polar_equation_of_line_l1044_104401

/-- The polar equation of a line passing through (5,0) and perpendicular to α = π/4 -/
theorem polar_equation_of_line (ρ θ : ℝ) : 
  (∃ (x y : ℝ), x = 5 ∧ y = 0 ∧ ρ * (Real.cos θ) = x ∧ ρ * (Real.sin θ) = y) →
  (∀ (α : ℝ), α = π/4 → (Real.tan α) * (Real.tan (α + π/2)) = -1) →
  ρ * Real.sin (π/4 + θ) = 5 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_polar_equation_of_line_l1044_104401


namespace NUMINAMATH_CALUDE_fraction_always_defined_l1044_104432

theorem fraction_always_defined (y : ℝ) : (y^2 + 1) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_fraction_always_defined_l1044_104432


namespace NUMINAMATH_CALUDE_polygon_sides_l1044_104460

theorem polygon_sides (sum_interior_angles sum_exterior_angles : ℕ) : 
  sum_interior_angles - sum_exterior_angles = 720 →
  sum_exterior_angles = 360 →
  (∃ n : ℕ, sum_interior_angles = (n - 2) * 180 ∧ n = 8) :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l1044_104460


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1044_104474

theorem solve_exponential_equation (y : ℝ) : 5^(2*y) = Real.sqrt 125 → y = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1044_104474


namespace NUMINAMATH_CALUDE_binomial_10_choose_5_l1044_104422

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_5_l1044_104422


namespace NUMINAMATH_CALUDE_problem_solution_l1044_104456

theorem problem_solution (x : ℝ) : 
  x + Real.sqrt (x^2 - 1) + 1 / (x - Real.sqrt (x^2 - 1)) = 15 →
  x^3 + Real.sqrt (x^6 - 1) + 1 / (x^3 + Real.sqrt (x^6 - 1)) = 3970049 / 36000 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1044_104456


namespace NUMINAMATH_CALUDE_min_value_expression_l1044_104452

theorem min_value_expression (x : ℝ) (h : 0 ≤ x ∧ x < 4) :
  ∃ (min : ℝ), min = Real.sqrt 5 ∧
  ∀ y, 0 ≤ y ∧ y < 4 → (y^2 + 2*y + 6) / (2*y + 2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1044_104452


namespace NUMINAMATH_CALUDE_mall_sales_growth_rate_l1044_104445

theorem mall_sales_growth_rate :
  let initial_sales := 1000000  -- January sales in yuan
  let feb_decrease := 0.1       -- 10% decrease in February
  let april_sales := 1296000    -- April sales in yuan
  let growth_rate := 0.2        -- 20% growth rate to be proven
  (initial_sales * (1 - feb_decrease) * (1 + growth_rate)^2 = april_sales) := by
  sorry

end NUMINAMATH_CALUDE_mall_sales_growth_rate_l1044_104445


namespace NUMINAMATH_CALUDE_figure_perimeter_is_33_l1044_104416

/-- The perimeter of a figure composed of a 4x4 square with a 2x1 rectangle protruding from one side -/
def figurePerimeter (unitSquareSideLength : ℝ) : ℝ :=
  let largeSquareSide := 4 * unitSquareSideLength
  let rectangleWidth := 2 * unitSquareSideLength
  let rectangleHeight := unitSquareSideLength
  2 * largeSquareSide + rectangleWidth + rectangleHeight

theorem figure_perimeter_is_33 :
  figurePerimeter 2 = 33 := by
  sorry


end NUMINAMATH_CALUDE_figure_perimeter_is_33_l1044_104416


namespace NUMINAMATH_CALUDE_even_numbers_set_builder_notation_l1044_104446

-- Define the set of even numbers
def EvenNumbers : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

-- State the theorem
theorem even_numbers_set_builder_notation : 
  EvenNumbers = {x : ℤ | ∃ n : ℤ, x = 2 * n} := by sorry

end NUMINAMATH_CALUDE_even_numbers_set_builder_notation_l1044_104446


namespace NUMINAMATH_CALUDE_wednesday_production_l1044_104499

/-- The number of clay pots Nancy created on each day of the week --/
structure ClayPotProduction where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- The conditions of Nancy's clay pot production --/
def nancysProduction : ClayPotProduction where
  monday := 12
  tuesday := 12 * 2
  wednesday := 50 - (12 + 12 * 2)

/-- Theorem stating that Nancy created 14 clay pots on Wednesday --/
theorem wednesday_production : nancysProduction.wednesday = 14 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_production_l1044_104499


namespace NUMINAMATH_CALUDE_vodka_mixture_profit_l1044_104403

/-- Represents the profit percentage of a mixture of two vodkas -/
def mixtureProfitPercentage (profit1 profit2 : ℚ) (increase1 increase2 : ℚ) : ℚ :=
  (profit1 * increase1 + profit2 * increase2) / 2

theorem vodka_mixture_profit :
  let initialProfit1 : ℚ := 10 / 100
  let initialProfit2 : ℚ := 40 / 100
  let increase1 : ℚ := 4 / 3
  let increase2 : ℚ := 5 / 3
  mixtureProfitPercentage initialProfit1 initialProfit2 increase1 increase2 = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_vodka_mixture_profit_l1044_104403


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1044_104441

theorem smallest_solution_abs_equation :
  ∃ x : ℝ, x * |x| = 3 * x - 2 ∧
  ∀ y : ℝ, y * |y| = 3 * y - 2 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1044_104441


namespace NUMINAMATH_CALUDE_rectangle_y_coordinate_sum_l1044_104415

/-- Given a rectangle with opposite vertices (5,22) and (12,-3),
    the sum of the y-coordinates of the other two vertices is 19. -/
theorem rectangle_y_coordinate_sum :
  let v1 : ℝ × ℝ := (5, 22)
  let v2 : ℝ × ℝ := (12, -3)
  let mid_y : ℝ := (v1.2 + v2.2) / 2
  19 = 2 * mid_y := by sorry

end NUMINAMATH_CALUDE_rectangle_y_coordinate_sum_l1044_104415


namespace NUMINAMATH_CALUDE_inequality_proof_l1044_104408

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1044_104408


namespace NUMINAMATH_CALUDE_find_a_l1044_104463

def A : Set ℝ := {x | 1 < x ∧ x < 7}
def B (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 5}

theorem find_a : ∃ a : ℝ, A ∩ B a = {x | 3 < x ∧ x < 7} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1044_104463


namespace NUMINAMATH_CALUDE_roots_sum_ln_abs_l1044_104406

theorem roots_sum_ln_abs (m : ℝ) (x₁ x₂ : ℝ) :
  (Real.log (|x₁ - 2|) = m) ∧ (Real.log (|x₂ - 2|) = m) →
  x₁ + x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_ln_abs_l1044_104406


namespace NUMINAMATH_CALUDE_fresh_grapes_weight_calculation_l1044_104471

/-- The weight of dried grapes in kilograms -/
def dried_grapes_weight : ℝ := 66.67

/-- The fraction of water in fresh grapes by weight -/
def fresh_water_fraction : ℝ := 0.75

/-- The fraction of water in dried grapes by weight -/
def dried_water_fraction : ℝ := 0.25

/-- The weight of fresh grapes in kilograms -/
def fresh_grapes_weight : ℝ := 200.01

theorem fresh_grapes_weight_calculation :
  fresh_grapes_weight = dried_grapes_weight * (1 - dried_water_fraction) / (1 - fresh_water_fraction) :=
by sorry

end NUMINAMATH_CALUDE_fresh_grapes_weight_calculation_l1044_104471


namespace NUMINAMATH_CALUDE_banana_pear_weight_equivalence_l1044_104420

/-- Given that 9 bananas weigh the same as 6 pears, prove that 36 bananas weigh the same as 24 pears. -/
theorem banana_pear_weight_equivalence (banana_weight pear_weight : ℝ) 
  (h : 9 * banana_weight = 6 * pear_weight) :
  36 * banana_weight = 24 * pear_weight := by
  sorry

end NUMINAMATH_CALUDE_banana_pear_weight_equivalence_l1044_104420


namespace NUMINAMATH_CALUDE_original_number_proof_l1044_104473

theorem original_number_proof :
  ∃ x : ℝ, x * 1.1 = 550 ∧ x = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1044_104473


namespace NUMINAMATH_CALUDE_window_wood_strip_width_l1044_104426

/-- Represents the dimensions of a glass piece in centimeters -/
structure GlassDimensions where
  width : ℝ
  height : ℝ

/-- Represents the configuration of a window -/
structure WindowConfig where
  glassDimensions : GlassDimensions
  woodStripWidth : ℝ

/-- Calculates the total area of glass in the window -/
def totalGlassArea (config : WindowConfig) : ℝ :=
  4 * config.glassDimensions.width * config.glassDimensions.height

/-- Calculates the total area of the window -/
def totalWindowArea (config : WindowConfig) : ℝ :=
  (2 * config.glassDimensions.width + 3 * config.woodStripWidth) *
  (2 * config.glassDimensions.height + 3 * config.woodStripWidth)

/-- Theorem: If the total area of glass equals the total area of wood,
    then the wood strip width is 20/3 cm -/
theorem window_wood_strip_width
  (config : WindowConfig)
  (h1 : config.glassDimensions.width = 30)
  (h2 : config.glassDimensions.height = 20)
  (h3 : totalGlassArea config = totalWindowArea config - totalGlassArea config) :
  config.woodStripWidth = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_window_wood_strip_width_l1044_104426


namespace NUMINAMATH_CALUDE_paper_flipping_difference_l1044_104402

theorem paper_flipping_difference (Y G : ℕ) : 
  Y - 152 = G + 152 + 346 → Y - G = 650 := by sorry

end NUMINAMATH_CALUDE_paper_flipping_difference_l1044_104402


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_over_four_l1044_104424

theorem tan_alpha_minus_pi_over_four (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan β = 1 / 3) :
  Real.tan (α - π / 4) = - 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_over_four_l1044_104424


namespace NUMINAMATH_CALUDE_right_triangle_legs_l1044_104423

theorem right_triangle_legs (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive lengths
  c = 25 →                 -- Hypotenuse is 25 cm
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  b / a = 4 / 3 →          -- Ratio of legs is 4:3
  a = 15 ∧ b = 20 :=       -- Legs are 15 cm and 20 cm
by sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l1044_104423


namespace NUMINAMATH_CALUDE_smallest_n_for_less_than_one_percent_probability_l1044_104484

def double_factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n+2 => (n+2) * double_factorial n

def townspeople_win_probability (n : ℕ) : ℚ :=
  (n.factorial : ℚ) / (double_factorial (2*n+1) : ℚ)

theorem smallest_n_for_less_than_one_percent_probability :
  ∀ k : ℕ, k < 6 → townspeople_win_probability k ≥ 1/100 ∧
  townspeople_win_probability 6 < 1/100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_less_than_one_percent_probability_l1044_104484


namespace NUMINAMATH_CALUDE_ordering_abc_l1044_104433

noncomputable def a : ℝ := 0.98 + Real.sin 0.01
noncomputable def b : ℝ := Real.exp (-0.01)
noncomputable def c : ℝ := (Real.log 2022) / (Real.log 2023) / (Real.log 2021) / (Real.log 2023)

theorem ordering_abc : c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l1044_104433


namespace NUMINAMATH_CALUDE_min_cost_floppy_cd_l1044_104439

/-- The minimum cost of 3 floppy disks and 9 CDs given price constraints -/
theorem min_cost_floppy_cd (x y : ℝ) 
  (h1 : 4 * x + 5 * y ≥ 20) 
  (h2 : 6 * x + 3 * y ≤ 24) : 
  ∃ (m : ℝ), m = 3 * x + 9 * y ∧ m ≥ 22 ∧ ∀ (n : ℝ), n = 3 * x + 9 * y → n ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_cost_floppy_cd_l1044_104439


namespace NUMINAMATH_CALUDE_recurrence_sequence_uniqueness_l1044_104494

/-- A sequence of natural numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

/-- A bounded sequence of natural numbers -/
def BoundedSequence (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n, a n ≤ M

theorem recurrence_sequence_uniqueness (a : ℕ → ℕ) 
  (h_recurrence : RecurrenceSequence a) (h_bounded : BoundedSequence a) :
  ∀ n, a n = 2 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_uniqueness_l1044_104494


namespace NUMINAMATH_CALUDE_statement_equivalence_l1044_104458

theorem statement_equivalence (x y : ℝ) :
  ((x - 1) * (y + 2) ≠ 0 → x ≠ 1 ∧ y ≠ -2) ↔
  (x = 1 ∨ y = -2 → (x - 1) * (y + 2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l1044_104458


namespace NUMINAMATH_CALUDE_trajectory_equation_l1044_104404

/-- 
Given a point P in the plane, if its distance to the line y=-3 is equal to 
its distance to the point (0,3), then the equation of its trajectory is x^2 = 12y.
-/
theorem trajectory_equation (P : ℝ × ℝ) : 
  (∀ (x y : ℝ), P = (x, y) → |y + 3| = ((x - 0)^2 + (y - 3)^2).sqrt) →
  (∃ (x y : ℝ), P = (x, y) ∧ x^2 = 12*y) :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1044_104404


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1044_104443

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a4 : a 4 = 8) :
  a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1044_104443


namespace NUMINAMATH_CALUDE_sqrt_sum_square_product_l1044_104451

theorem sqrt_sum_square_product (x : ℝ) :
  Real.sqrt (9 + x) + Real.sqrt (25 - x) = 10 →
  (9 + x) * (25 - x) = 1089 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_square_product_l1044_104451


namespace NUMINAMATH_CALUDE_same_solution_implies_k_equals_two_l1044_104469

theorem same_solution_implies_k_equals_two (k : ℝ) :
  (∃ x : ℝ, (2 * x - 1) / 3 = 5 ∧ k * x - 1 = 15) →
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_same_solution_implies_k_equals_two_l1044_104469


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1044_104481

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1044_104481


namespace NUMINAMATH_CALUDE_final_student_count_l1044_104488

def initial_students : ℕ := 31
def students_left : ℕ := 5
def new_students : ℕ := 11

theorem final_student_count : 
  initial_students - students_left + new_students = 37 := by
  sorry

end NUMINAMATH_CALUDE_final_student_count_l1044_104488


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l1044_104447

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^12 + i^17 + i^22 + i^27 + i^32 + i^37 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l1044_104447


namespace NUMINAMATH_CALUDE_yellow_pairs_count_l1044_104437

theorem yellow_pairs_count (blue_count : ℕ) (yellow_count : ℕ) (total_count : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_count = 63 →
  yellow_count = 69 →
  total_count = blue_count + yellow_count →
  total_pairs = 66 →
  blue_blue_pairs = 27 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 30 ∧ 
    yellow_yellow_pairs = (yellow_count - (total_pairs - blue_blue_pairs - (yellow_count - blue_count) / 2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_yellow_pairs_count_l1044_104437


namespace NUMINAMATH_CALUDE_maria_earnings_l1044_104444

/-- Calculates the total earnings of a flower saleswoman over three days --/
def flower_sales_earnings (tulip_price rose_price : ℚ) 
  (day1_tulips day1_roses : ℕ) 
  (day2_multiplier : ℚ) 
  (day3_tulip_percentage : ℚ) 
  (day3_roses : ℕ) : ℚ :=
  let day1_earnings := tulip_price * day1_tulips + rose_price * day1_roses
  let day2_earnings := day2_multiplier * day1_earnings
  let day3_tulips := day3_tulip_percentage * (day2_multiplier * day1_tulips)
  let day3_earnings := tulip_price * day3_tulips + rose_price * day3_roses
  day1_earnings + day2_earnings + day3_earnings

/-- Theorem stating that Maria's total earnings over three days is $420 --/
theorem maria_earnings : 
  flower_sales_earnings 2 3 30 20 2 (1/10) 16 = 420 := by
  sorry

end NUMINAMATH_CALUDE_maria_earnings_l1044_104444


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1044_104438

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_eq : a 3 * a 9 = 2 * (a 5)^2)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1044_104438


namespace NUMINAMATH_CALUDE_garden_area_increase_l1044_104418

theorem garden_area_increase :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter := 2 * (rect_length + rect_width)
  let square_side := rect_perimeter / 4
  let rect_area := rect_length * rect_width
  let square_area := square_side * square_side
  square_area - rect_area = 400 := by
sorry

end NUMINAMATH_CALUDE_garden_area_increase_l1044_104418


namespace NUMINAMATH_CALUDE_steven_name_day_l1044_104477

def wordsOnDay (n : ℕ) : ℕ :=
  2 * (n / 2) + 4 * ((n - 1) / 2)

theorem steven_name_day (n : ℕ) : wordsOnDay n = 44 ↔ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_steven_name_day_l1044_104477


namespace NUMINAMATH_CALUDE_antons_winning_numbers_infinite_l1044_104414

theorem antons_winning_numbers_infinite :
  ∃ (f : ℕ → ℕ), Function.Injective f ∧
  ∀ (x : ℕ), 
    let n := f x
    (¬ ∃ (m : ℕ), n = m ^ 2) ∧ 
    ∃ (k : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4)) = k ^ 2 :=
sorry

end NUMINAMATH_CALUDE_antons_winning_numbers_infinite_l1044_104414


namespace NUMINAMATH_CALUDE_erasers_per_friend_l1044_104485

/-- Given 9306 erasers shared among 99 friends, prove that each friend receives 94 erasers. -/
theorem erasers_per_friend :
  let total_erasers : ℕ := 9306
  let num_friends : ℕ := 99
  let erasers_per_friend : ℕ := total_erasers / num_friends
  erasers_per_friend = 94 := by sorry

end NUMINAMATH_CALUDE_erasers_per_friend_l1044_104485


namespace NUMINAMATH_CALUDE_range_of_a_l1044_104491

def f (x a : ℝ) : ℝ := |x - a| + x + 5

theorem range_of_a (a : ℝ) : (∀ x, f x a ≥ 8) ↔ |a + 5| ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1044_104491


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l1044_104450

open Complex

theorem smallest_distance_between_complex_points (z w : ℂ) 
  (hz : abs (z + 2 + 4*I) = 2)
  (hw : abs (w - 6 - 7*I) = 4) :
  ∃ (d : ℝ), d = Real.sqrt 185 - 6 ∧ ∀ (z' w' : ℂ), 
    abs (z' + 2 + 4*I) = 2 → abs (w' - 6 - 7*I) = 4 → 
    abs (z' - w') ≥ d ∧ ∃ (z'' w'' : ℂ), abs (z'' - w'') = d :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l1044_104450


namespace NUMINAMATH_CALUDE_nancy_age_proof_l1044_104468

/-- Nancy's age in years -/
def nancy_age : ℕ := 5

/-- Nancy's grandmother's age in years -/
def grandmother_age : ℕ := 10 * nancy_age

/-- Age difference between Nancy's grandmother and Nancy at Nancy's birth -/
def age_difference : ℕ := 45

theorem nancy_age_proof :
  nancy_age = 5 ∧
  grandmother_age = 10 * nancy_age ∧
  grandmother_age - nancy_age = age_difference :=
by sorry

end NUMINAMATH_CALUDE_nancy_age_proof_l1044_104468


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1044_104495

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : IsArithmeticSequence a) 
  (h_eq : 4 * a 3 + a 11 - 3 * a 5 = 10) : 
  a 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1044_104495


namespace NUMINAMATH_CALUDE_product_of_fractions_l1044_104462

theorem product_of_fractions : 
  (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1044_104462


namespace NUMINAMATH_CALUDE_projection_onto_xoy_plane_l1044_104436

/-- Given a space orthogonal coordinate system Oxyz, prove that the projection
    of point P(1, 2, 3) onto the xOy plane has coordinates (1, 2, 0). -/
theorem projection_onto_xoy_plane :
  let P : ℝ × ℝ × ℝ := (1, 2, 3)
  let xoy_plane : Set (ℝ × ℝ × ℝ) := {v | v.2.2 = 0}
  let projection (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, p.2.1, 0)
  projection P ∈ xoy_plane ∧ projection P = (1, 2, 0) := by
  sorry

end NUMINAMATH_CALUDE_projection_onto_xoy_plane_l1044_104436


namespace NUMINAMATH_CALUDE_clinton_school_earnings_l1044_104464

/-- Represents the total compensation for all students -/
def total_compensation : ℝ := 1456

/-- Represents the number of students from Arlington school -/
def arlington_students : ℕ := 8

/-- Represents the number of days Arlington students worked -/
def arlington_days : ℕ := 4

/-- Represents the number of students from Bradford school -/
def bradford_students : ℕ := 6

/-- Represents the number of days Bradford students worked -/
def bradford_days : ℕ := 7

/-- Represents the number of students from Clinton school -/
def clinton_students : ℕ := 7

/-- Represents the number of days Clinton students worked -/
def clinton_days : ℕ := 8

/-- Theorem stating that the total earnings for Clinton school students is 627.20 dollars -/
theorem clinton_school_earnings :
  let total_student_days := arlington_students * arlington_days + bradford_students * bradford_days + clinton_students * clinton_days
  let daily_wage := total_compensation / total_student_days
  clinton_students * clinton_days * daily_wage = 627.2 := by
  sorry

end NUMINAMATH_CALUDE_clinton_school_earnings_l1044_104464


namespace NUMINAMATH_CALUDE_parallelepiped_to_cube_l1044_104428

/-- A rectangular parallelepiped with edges 8, 8, and 27 has the same volume as a cube with side length 12 -/
theorem parallelepiped_to_cube : 
  let parallelepiped_volume := 8 * 8 * 27
  let cube_volume := 12 * 12 * 12
  parallelepiped_volume = cube_volume := by
  sorry

#eval 8 * 8 * 27
#eval 12 * 12 * 12

end NUMINAMATH_CALUDE_parallelepiped_to_cube_l1044_104428


namespace NUMINAMATH_CALUDE_product_remainder_l1044_104442

theorem product_remainder (N : ℕ) : 
  (1274 * 1275 * N * 1285) % 12 = 6 → N % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1044_104442


namespace NUMINAMATH_CALUDE_parabola_focus_x_coordinate_l1044_104467

/-- Given a parabola y^2 = 2px (p > 0) and a point A(1, 2) on this parabola,
    if the distance from point A to point B(x, 0) is equal to its distance to the line x = -1,
    then x = 1. -/
theorem parabola_focus_x_coordinate (p : ℝ) (x : ℝ) :
  p > 0 →
  (2 : ℝ)^2 = 2 * p * 1 →
  (∀ y : ℝ, y^2 = 2 * p * x ↔ (y = 2 ∧ x = 1)) →
  (x - 1)^2 + 2^2 = (1 - (-1))^2 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_x_coordinate_l1044_104467


namespace NUMINAMATH_CALUDE_second_investment_rate_l1044_104496

/-- Calculates the interest rate of the second investment given the total investment,
    the amount and rate of the first investment, and the total amount after interest. -/
theorem second_investment_rate (total_investment : ℝ) (first_investment : ℝ) 
  (first_rate : ℝ) (total_after_interest : ℝ) :
  total_investment = 1000 →
  first_investment = 200 →
  first_rate = 0.03 →
  total_after_interest = 1046 →
  (total_after_interest - total_investment - (first_investment * first_rate)) / 
  (total_investment - first_investment) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_rate_l1044_104496


namespace NUMINAMATH_CALUDE_total_ways_to_choose_courses_l1044_104465

-- Define the number of courses of each type
def num_courses_A : ℕ := 4
def num_courses_B : ℕ := 2

-- Define the total number of courses to be chosen
def total_courses_to_choose : ℕ := 4

-- Define the function to calculate the number of ways to choose courses
def num_ways_to_choose : ℕ := 
  (num_courses_B.choose 1 * num_courses_A.choose 3) +
  (num_courses_B.choose 2 * num_courses_A.choose 2)

-- Theorem statement
theorem total_ways_to_choose_courses : num_ways_to_choose = 14 := by
  sorry


end NUMINAMATH_CALUDE_total_ways_to_choose_courses_l1044_104465


namespace NUMINAMATH_CALUDE_min_output_avoids_losses_l1044_104413

/-- The profit function for a company's product -/
def profit_function (x : ℝ) : ℝ := 0.1 * x - 150

/-- The minimum output to avoid losses -/
def min_output : ℝ := 1500

theorem min_output_avoids_losses :
  ∀ x : ℝ, x ≥ min_output → profit_function x ≥ 0 ∧
  ∀ y : ℝ, y < min_output → ∃ z : ℝ, z ≥ y ∧ profit_function z < 0 :=
by sorry

end NUMINAMATH_CALUDE_min_output_avoids_losses_l1044_104413


namespace NUMINAMATH_CALUDE_january_more_expensive_l1044_104409

/-- Represents the cost of purchasing screws and bolts in different months -/
structure CostComparison where
  january_screws_per_dollar : ℕ
  january_bolts_per_dollar : ℕ
  february_set_screws : ℕ
  february_set_bolts : ℕ
  february_set_price : ℕ
  tractor_screws : ℕ
  tractor_bolts : ℕ

/-- Calculates the cost of purchasing screws and bolts for a tractor in January -/
def january_cost (c : CostComparison) : ℚ :=
  (c.tractor_screws : ℚ) / c.january_screws_per_dollar + (c.tractor_bolts : ℚ) / c.january_bolts_per_dollar

/-- Calculates the cost of purchasing screws and bolts for a tractor in February -/
def february_cost (c : CostComparison) : ℚ :=
  (c.february_set_price : ℚ) * (max (c.tractor_screws / c.february_set_screws) (c.tractor_bolts / c.february_set_bolts))

/-- Theorem stating that the cost in January is higher than in February -/
theorem january_more_expensive (c : CostComparison) 
    (h1 : c.january_screws_per_dollar = 40)
    (h2 : c.january_bolts_per_dollar = 60)
    (h3 : c.february_set_screws = 25)
    (h4 : c.february_set_bolts = 25)
    (h5 : c.february_set_price = 1)
    (h6 : c.tractor_screws = 600)
    (h7 : c.tractor_bolts = 600) :
  january_cost c > february_cost c := by
  sorry

end NUMINAMATH_CALUDE_january_more_expensive_l1044_104409


namespace NUMINAMATH_CALUDE_triangle_special_sequence_equilateral_l1044_104486

/-- A triangle with angles forming an arithmetic sequence and reciprocals of side lengths forming an arithmetic sequence is equilateral. -/
theorem triangle_special_sequence_equilateral (A B C : ℝ) (a b c : ℝ) :
  -- Angles form an arithmetic sequence
  ∃ (d : ℝ), (B = A + d ∧ C = B + d) →
  -- Reciprocals of side lengths form an arithmetic sequence
  ∃ (k : ℝ), (1/b = 1/a + k ∧ 1/c = 1/b + k) →
  -- Angles sum to 180°
  A + B + C = 180 →
  -- Side lengths are positive
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Conclusion: The triangle is equilateral
  A = 60 ∧ B = 60 ∧ C = 60 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_sequence_equilateral_l1044_104486


namespace NUMINAMATH_CALUDE_four_propositions_l1044_104448

theorem four_propositions :
  (∀ a b : ℝ, |a + b| - 2 * |a| ≤ |a - b|) ∧
  (∀ a b : ℝ, |a - b| < 1 → |a| < |b| + 1) ∧
  (∀ x y : ℝ, |x| < 2 ∧ |y| > 3 → |x / y| < 2/3) ∧
  (∀ A B : ℝ, A > 0 ∧ B > 0 → Real.log ((|A| + |B|) / 2) ≥ (Real.log |A| + Real.log |B|) / 2) :=
by sorry

end NUMINAMATH_CALUDE_four_propositions_l1044_104448


namespace NUMINAMATH_CALUDE_closest_multiple_l1044_104429

def target : ℕ := 2500
def divisor : ℕ := 18

-- Define a function to calculate the distance between two natural numbers
def distance (a b : ℕ) : ℕ := max a b - min a b

-- Define a function to check if a number is a multiple of another
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement of the theorem
theorem closest_multiple :
  ∀ n : ℕ, is_multiple n divisor →
    distance n target ≥ distance 2502 target :=
sorry

end NUMINAMATH_CALUDE_closest_multiple_l1044_104429


namespace NUMINAMATH_CALUDE_seven_mult_three_equals_sixteen_l1044_104407

-- Define the custom operation *
def custom_mult (a b : ℤ) : ℤ := 4*a + 3*b - a*b

-- State the theorem
theorem seven_mult_three_equals_sixteen : custom_mult 7 3 = 16 := by sorry

end NUMINAMATH_CALUDE_seven_mult_three_equals_sixteen_l1044_104407


namespace NUMINAMATH_CALUDE_sum_of_double_root_k_values_l1044_104493

/-- The quadratic equation we're working with -/
def quadratic (k x : ℝ) : ℝ := x^2 + 2*k*x + 7*k - 10

/-- The discriminant of our quadratic equation -/
def discriminant (k : ℝ) : ℝ := (2*k)^2 - 4*(7*k - 10)

/-- A value of k for which the quadratic equation has exactly one solution -/
def is_double_root (k : ℝ) : Prop := discriminant k = 0

/-- The theorem stating that the sum of the values of k for which
    the quadratic equation has exactly one solution is 7 -/
theorem sum_of_double_root_k_values :
  ∃ k₁ k₂ : ℝ, k₁ ≠ k₂ ∧ is_double_root k₁ ∧ is_double_root k₂ ∧ k₁ + k₂ = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_double_root_k_values_l1044_104493


namespace NUMINAMATH_CALUDE_function_inequality_l1044_104483

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 2) = -f x)
  (h2 : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 → f x₁ < f x₂)
  (h3 : ∀ x : ℝ, f (x + 2) = f (-x + 2)) :
  f 4.5 < f 7 ∧ f 7 < f 6.5 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l1044_104483


namespace NUMINAMATH_CALUDE_task_completion_ways_l1044_104421

theorem task_completion_ways (method1_count method2_count : ℕ) :
  method1_count + method2_count = 
  (number_of_ways_to_choose_person : ℕ) :=
by sorry

#check task_completion_ways 5 4

end NUMINAMATH_CALUDE_task_completion_ways_l1044_104421


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l1044_104476

theorem complex_power_magnitude : Complex.abs ((2 : ℂ) + Complex.I) ^ 8 = 625 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l1044_104476


namespace NUMINAMATH_CALUDE_max_sum_exp_l1044_104475

theorem max_sum_exp (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 ≤ 4) :
  ∃ (M : ℝ), M = 4 * Real.exp 1 ∧ ∀ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 ≤ 4 →
    Real.exp x + Real.exp y + Real.exp z + Real.exp w ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_exp_l1044_104475


namespace NUMINAMATH_CALUDE_iceland_visitors_iceland_visitor_count_l1044_104472

theorem iceland_visitors (total : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 50)
  (h2 : norway = 23)
  (h3 : both = 21)
  (h4 : neither = 23) :
  total = (total - norway - neither + both) + norway - both + neither :=
by sorry

theorem iceland_visitor_count : 
  ∃ (iceland : ℕ), iceland = 50 - 23 - 23 + 21 ∧ iceland = 25 :=
by sorry

end NUMINAMATH_CALUDE_iceland_visitors_iceland_visitor_count_l1044_104472


namespace NUMINAMATH_CALUDE_motorcycle_trip_time_difference_specific_motorcycle_problem_l1044_104455

/-- Given a motorcycle traveling at a constant speed, prove the time difference between two trips -/
theorem motorcycle_trip_time_difference (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ) : 
  speed > 0 → 
  distance1 > 0 →
  distance2 > 0 →
  distance1 > distance2 →
  (distance1 / speed - distance2 / speed) * 60 = (distance1 - distance2) / speed * 60 := by
  sorry

/-- Specific instance of the theorem for the given problem -/
theorem specific_motorcycle_problem : 
  (400 / 40 - 360 / 40) * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_trip_time_difference_specific_motorcycle_problem_l1044_104455


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1044_104417

theorem smallest_number_proof (x y z : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  (x + y + z) / 3 = 165 →
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1044_104417


namespace NUMINAMATH_CALUDE_percentage_boys_soccer_l1044_104427

def total_students : ℕ := 420
def boys : ℕ := 312
def soccer_players : ℕ := 250
def girls_not_playing : ℕ := 53

theorem percentage_boys_soccer : 
  (boys - (total_students - soccer_players - girls_not_playing)) / soccer_players * 100 = 78 := by
  sorry

end NUMINAMATH_CALUDE_percentage_boys_soccer_l1044_104427


namespace NUMINAMATH_CALUDE_negation_of_implication_l1044_104425

theorem negation_of_implication (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 = 0 ∧ (x ≠ 0 ∨ y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1044_104425


namespace NUMINAMATH_CALUDE_count_distinct_sums_of_special_fractions_l1044_104479

def IsSpecialFraction (a b : ℕ+) : Prop := a + b = 17

def SumOfSpecialFractions (n : ℕ) : Prop :=
  ∃ (a₁ b₁ a₂ b₂ : ℕ+),
    IsSpecialFraction a₁ b₁ ∧
    IsSpecialFraction a₂ b₂ ∧
    n = (a₁ : ℚ) / b₁ + (a₂ : ℚ) / b₂

theorem count_distinct_sums_of_special_fractions :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, SumOfSpecialFractions n) ∧
    (∀ n, SumOfSpecialFractions n → n ∈ s) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_sums_of_special_fractions_l1044_104479


namespace NUMINAMATH_CALUDE_remainder_problem_l1044_104405

theorem remainder_problem (d : ℕ) (a b : ℕ) (h1 : d > 0) (h2 : d ≤ a ∧ d ≤ b) 
  (h3 : ∀ k > d, k ∣ a ∨ k ∣ b) (h4 : b % d = 5) : a % d = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1044_104405


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1044_104482

theorem trigonometric_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1044_104482


namespace NUMINAMATH_CALUDE_carnation_bouquet_problem_l1044_104478

/-- Given 3 bouquets of carnations with known quantities in the first and third bouquets,
    and a known average, prove the quantity in the second bouquet. -/
theorem carnation_bouquet_problem (b1 b3 avg : ℕ) (h1 : b1 = 9) (h3 : b3 = 13) (havg : avg = 12) :
  ∃ b2 : ℕ, (b1 + b2 + b3) / 3 = avg ∧ b2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_carnation_bouquet_problem_l1044_104478


namespace NUMINAMATH_CALUDE_garden_bug_problem_l1044_104430

theorem garden_bug_problem (initial_plants : ℕ) (day1_eaten : ℕ) (day3_eaten : ℕ) : 
  initial_plants = 30 →
  day1_eaten = 20 →
  day3_eaten = 1 →
  initial_plants - day1_eaten - (initial_plants - day1_eaten) / 2 - day3_eaten = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_bug_problem_l1044_104430


namespace NUMINAMATH_CALUDE_base_eight_representation_l1044_104497

theorem base_eight_representation (a b c d e f : ℕ) 
  (h1 : 208208 = 8^5 * a + 8^4 * b + 8^3 * c + 8^2 * d + 8 * e + f)
  (h2 : a ≤ 7 ∧ b ≤ 7 ∧ c ≤ 7 ∧ d ≤ 7 ∧ e ≤ 7 ∧ f ≤ 7) :
  a * b * c + d * e * f = 72 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_representation_l1044_104497


namespace NUMINAMATH_CALUDE_peters_correct_percentage_l1044_104440

theorem peters_correct_percentage (y : ℕ) :
  let total_questions : ℕ := 7 * y
  let missed_questions : ℕ := 2 * y
  let correct_questions : ℕ := total_questions - missed_questions
  (correct_questions : ℚ) / (total_questions : ℚ) * 100 = 500 / 7 :=
by sorry

end NUMINAMATH_CALUDE_peters_correct_percentage_l1044_104440


namespace NUMINAMATH_CALUDE_smallest_number_in_arithmetic_sequence_l1044_104461

theorem smallest_number_in_arithmetic_sequence (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 29 →
  b = 30 →
  c = b + 5 →
  a < b ∧ b < c →
  a = 22 := by sorry

end NUMINAMATH_CALUDE_smallest_number_in_arithmetic_sequence_l1044_104461


namespace NUMINAMATH_CALUDE_female_managers_count_l1044_104457

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  female_employees : ℕ
  total_managers : ℕ
  male_managers : ℕ

/-- Conditions for the company -/
def company_conditions (c : Company) : Prop :=
  c.female_employees = 500 ∧
  c.total_managers = (2 * c.total_employees) / 5 ∧
  c.male_managers = (2 * (c.total_employees - c.female_employees)) / 5

/-- The theorem to be proved -/
theorem female_managers_count (c : Company) 
  (h : company_conditions c) : 
  c.total_managers - c.male_managers = 200 := by
  sorry

end NUMINAMATH_CALUDE_female_managers_count_l1044_104457


namespace NUMINAMATH_CALUDE_percentage_difference_l1044_104454

theorem percentage_difference (A B y : ℝ) : 
  A > 0 → B > A → B = A * (1 + y / 100) → y = 100 * (B - A) / A := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1044_104454


namespace NUMINAMATH_CALUDE_joan_football_games_l1044_104470

/-- The number of football games Joan attended this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan attended last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem joan_football_games :
  total_games = 13 := by sorry

end NUMINAMATH_CALUDE_joan_football_games_l1044_104470


namespace NUMINAMATH_CALUDE_existence_of_digit_in_power_of_two_l1044_104412

theorem existence_of_digit_in_power_of_two (k d : ℕ) (h1 : k > 1) (h2 : d < 9) :
  ∃ n : ℕ, (2^n : ℕ) % 10^k = d := by
  sorry

end NUMINAMATH_CALUDE_existence_of_digit_in_power_of_two_l1044_104412


namespace NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l1044_104466

theorem quadratic_reciprocal_roots (a b c : ℤ) (x₁ x₂ : ℚ) : 
  a ≠ 0 →
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ * x₂ = 1) →
  (x₁ + x₂ = 4) →
  (∃ n m : ℤ, x₁ = n ∧ x₂ = m) →
  (c = a ∧ b = -4*a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l1044_104466


namespace NUMINAMATH_CALUDE_parabola_focus_l1044_104431

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = 9 * x^2 + 6 * x - 2

/-- The focus of a parabola -/
def is_focus (f : ℝ × ℝ) (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a h k : ℝ), 
    (∀ x y, eq x y ↔ y = a * (x - h)^2 + k) ∧
    f = (h, k + 1 / (4 * a))

/-- Theorem: The focus of the parabola y = 9x^2 + 6x - 2 is (-1/3, -107/36) -/
theorem parabola_focus :
  is_focus (-1/3, -107/36) parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1044_104431


namespace NUMINAMATH_CALUDE_total_tissues_l1044_104449

def group1 : ℕ := 15
def group2 : ℕ := 20
def group3 : ℕ := 18
def group4 : ℕ := 22
def group5 : ℕ := 25
def tissues_per_box : ℕ := 70

theorem total_tissues : 
  (group1 + group2 + group3 + group4 + group5) * tissues_per_box = 7000 := by
  sorry

end NUMINAMATH_CALUDE_total_tissues_l1044_104449


namespace NUMINAMATH_CALUDE_mystery_book_price_l1044_104487

theorem mystery_book_price (biography_price : ℝ) (total_discount : ℝ) 
  (biography_quantity : ℕ) (mystery_quantity : ℕ) (total_discount_rate : ℝ) 
  (mystery_discount_rate : ℝ) :
  biography_price = 20 →
  total_discount = 19 →
  biography_quantity = 5 →
  mystery_quantity = 3 →
  total_discount_rate = 0.43 →
  mystery_discount_rate = 0.375 →
  ∃ (mystery_price : ℝ),
    mystery_price * mystery_quantity * mystery_discount_rate + 
    biography_price * biography_quantity * (total_discount_rate - mystery_discount_rate) = 
    total_discount ∧
    mystery_price = 12 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_price_l1044_104487


namespace NUMINAMATH_CALUDE_expression_evaluation_l1044_104490

theorem expression_evaluation : 2 - (-3) - 4 + (-5) - 6 + 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1044_104490


namespace NUMINAMATH_CALUDE_decimal_expansion_of_one_forty_ninth_l1044_104480

/-- The repeating sequence in the decimal expansion of 1/49 -/
def repeating_sequence : List Nat :=
  [0, 2, 0, 4, 0, 8, 1, 6, 3, 2, 6, 5, 3, 0, 6, 1, 2, 2, 4, 4, 8, 9, 7, 9,
   5, 9, 1, 8, 3, 6, 7, 3, 4, 6, 9, 3, 8, 7, 7, 5, 5, 1]

/-- The length of the repeating sequence -/
def sequence_length : Nat := 42

/-- Theorem stating that the decimal expansion of 1/49 has the given repeating sequence -/
theorem decimal_expansion_of_one_forty_ninth :
  ∃ (n : Nat), (1 : ℚ) / 49 = (n : ℚ) / (10^sequence_length - 1) ∧
  repeating_sequence = (n.digits 10).reverse.take sequence_length :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_of_one_forty_ninth_l1044_104480


namespace NUMINAMATH_CALUDE_square_tens_seven_units_six_l1044_104434

theorem square_tens_seven_units_six (n : ℤ) : 
  (n^2 % 100 ≥ 70) ∧ (n^2 % 100 < 80) → n^2 % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_tens_seven_units_six_l1044_104434


namespace NUMINAMATH_CALUDE_boy_scout_interest_l1044_104419

/-- Represents the simple interest calculation for a Boy Scout Troop's account --/
theorem boy_scout_interest (final_balance : ℝ) (rate : ℝ) (time : ℝ) (interest : ℝ) : 
  final_balance = 310.45 →
  rate = 0.06 →
  time = 0.25 →
  interest = final_balance - (final_balance / (1 + rate * time)) →
  interest = 4.54 := by
sorry

end NUMINAMATH_CALUDE_boy_scout_interest_l1044_104419


namespace NUMINAMATH_CALUDE_problem_solution_l1044_104453

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 4 * x

theorem problem_solution :
  -- Part 1
  (∀ x : ℝ, f 2 x ≥ 2 * x + 1 ↔ x ∈ Set.Ici (-1)) ∧
  -- Part 2
  (∀ a : ℝ, a > 0 →
    (∀ x : ℝ, x ∈ Set.Ioi (-2) → f a (2 * x) > 7 * x + a^2 - 3) →
    a ∈ Set.Ioo 0 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1044_104453


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1044_104489

-- Define the quadratic function f(x)
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_function_theorem (a b : ℝ) :
  (∀ x, (f (f a b x + 2*x) a b) / (f a b x) = x^2 + 2023*x + 2040) →
  a = 2021 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1044_104489


namespace NUMINAMATH_CALUDE_angle_difference_range_l1044_104459

theorem angle_difference_range (α β : Real) (h1 : -π/2 < α) (h2 : α < β) (h3 : β < π/2) :
  ∃ (x : Real), -π < x ∧ x < 0 ∧ x = α - β :=
sorry

end NUMINAMATH_CALUDE_angle_difference_range_l1044_104459


namespace NUMINAMATH_CALUDE_amoeba_growth_5_days_l1044_104400

def amoeba_population (initial_population : ℕ) (split_rate : ℕ) (days : ℕ) : ℕ :=
  initial_population * split_rate ^ days

theorem amoeba_growth_5_days :
  amoeba_population 1 3 5 = 243 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_growth_5_days_l1044_104400


namespace NUMINAMATH_CALUDE_find_number_l1044_104435

theorem find_number (x : ℚ) : ((x / 9) - 13) / 7 - 8 = 13 → x = 1440 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1044_104435


namespace NUMINAMATH_CALUDE_sqrt_meaningful_value_l1044_104410

theorem sqrt_meaningful_value (x : ℝ) : 
  (x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 3) → 
  (x - 2 ≥ 0 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_value_l1044_104410


namespace NUMINAMATH_CALUDE_red_white_red_probability_l1044_104411

/-- The probability of drawing a red marble, then a white marble, and then a red marble
    from a bag containing 4 red marbles and 6 white marbles, without replacement. -/
theorem red_white_red_probability (total_marbles : Nat) (red_marbles : Nat) (white_marbles : Nat)
    (h1 : total_marbles = red_marbles + white_marbles)
    (h2 : red_marbles = 4)
    (h3 : white_marbles = 6) :
    (red_marbles : ℚ) / total_marbles *
    (white_marbles : ℚ) / (total_marbles - 1) *
    (red_marbles - 1 : ℚ) / (total_marbles - 2) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_red_white_red_probability_l1044_104411


namespace NUMINAMATH_CALUDE_y1_value_l1044_104492

theorem y1_value (y1 y2 y3 : ℝ) 
  (h1 : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h2 : (1 - y1)^2 + 2*(y1 - y2)^2 + 2*(y2 - y3)^2 + y3^2 = 1/2) :
  y1 = (2*Real.sqrt 2 - 1) / (2*Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_y1_value_l1044_104492


namespace NUMINAMATH_CALUDE_promotions_recipients_l1044_104498

def stadium_capacity : ℕ := 4500
def cap_interval : ℕ := 90
def shirt_interval : ℕ := 45
def sunglasses_interval : ℕ := 60

theorem promotions_recipients : 
  (∀ n : ℕ, n ≤ stadium_capacity → 
    (n % cap_interval = 0 ∧ n % shirt_interval = 0 ∧ n % sunglasses_interval = 0) ↔ 
    n % 180 = 0) →
  (stadium_capacity / 180 = 25) :=
by sorry

end NUMINAMATH_CALUDE_promotions_recipients_l1044_104498
