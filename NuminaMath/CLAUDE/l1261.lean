import Mathlib

namespace NUMINAMATH_CALUDE_g_zero_iff_a_eq_four_thirds_l1261_126144

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x - 4

-- State the theorem
theorem g_zero_iff_a_eq_four_thirds :
  ∀ a : ℝ, g a = 0 ↔ a = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_g_zero_iff_a_eq_four_thirds_l1261_126144


namespace NUMINAMATH_CALUDE_a_time_is_ten_l1261_126101

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 80 meters long
  a.speed * a.time = 80 ∧
  b.speed * b.time = 80 ∧
  -- A beats B by 56 meters or 7 seconds
  a.speed * (a.time + 7) = 136 ∧
  b.time = a.time + 7

/-- Theorem stating A's time is 10 seconds -/
theorem a_time_is_ten (a b : Runner) (h : Race a b) : a.time = 10 :=
  sorry

end NUMINAMATH_CALUDE_a_time_is_ten_l1261_126101


namespace NUMINAMATH_CALUDE_function_has_extrema_l1261_126100

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + (2*a + 1)*x

-- State the theorem
theorem function_has_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
   (∀ x, f a x ≥ f a x₁) ∧ 
   (∀ x, f a x ≤ f a x₂)) ↔ 
  (a > 1 ∨ a < -1/3) :=
sorry

end NUMINAMATH_CALUDE_function_has_extrema_l1261_126100


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1261_126125

theorem pure_imaginary_condition (m : ℝ) : 
  (Complex.I * (m^2 - 1) = m^2 + m - 2 + Complex.I * (m^2 - 1)) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1261_126125


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1261_126172

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 2 * x^2 - 4 * x - 1
  ∃ x1 x2 : ℝ, x1 = (2 + Real.sqrt 6) / 2 ∧ 
              x2 = (2 - Real.sqrt 6) / 2 ∧ 
              f x1 = 0 ∧ f x2 = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1261_126172


namespace NUMINAMATH_CALUDE_g_difference_l1261_126169

def g (n : ℕ) : ℚ := (1/4) * n * (n+1) * (n+2) * (n+3)

theorem g_difference (s : ℕ) : g s - g (s-1) = s * (s+1) * (s+2) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l1261_126169


namespace NUMINAMATH_CALUDE_coefficient_is_40_l1261_126166

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x³y³ in the expansion of (x+y)(2x-y)⁵
def coefficient_x3y3 : ℤ :=
  2^2 * (-1)^3 * binomial 5 3 + 2^3 * binomial 5 2

-- Theorem statement
theorem coefficient_is_40 : coefficient_x3y3 = 40 := by sorry

end NUMINAMATH_CALUDE_coefficient_is_40_l1261_126166


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1261_126160

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2, x)
  let b : ℝ × ℝ := (1, 2)
  parallel a b → x = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1261_126160


namespace NUMINAMATH_CALUDE_perpendicular_line_with_same_intercept_l1261_126138

/-- Given a line l with equation x/3 - y/4 = 1, 
    prove that the line with equation 3x + 4y + 16 = 0 
    has the same y-intercept as l and is perpendicular to l -/
theorem perpendicular_line_with_same_intercept 
  (x y : ℝ) (l : x / 3 - y / 4 = 1) :
  ∃ (m b : ℝ), 
    (-- Same y-intercept condition
     b = -4) ∧ 
    (-- Perpendicular condition
     m * (4 / 3) = -1) ∧
    (-- Equation of the new line
     3 * x + 4 * y + 16 = 0) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_with_same_intercept_l1261_126138


namespace NUMINAMATH_CALUDE_original_number_proof_l1261_126187

theorem original_number_proof (x : ℝ) : 
  (x * 1.125 - x * 0.75 = 30) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1261_126187


namespace NUMINAMATH_CALUDE_cube_increase_theorem_l1261_126129

theorem cube_increase_theorem :
  let s : ℝ := 1  -- Initial side length (can be any positive real number)
  let s' : ℝ := 1.2 * s  -- New side length after 20% increase
  let A : ℝ := 6 * s^2  -- Initial surface area
  let V : ℝ := s^3  -- Initial volume
  let A' : ℝ := 6 * s'^2  -- New surface area
  let V' : ℝ := s'^3  -- New volume
  let x : ℝ := (A' - A) / A * 100  -- Percentage increase in surface area
  let y : ℝ := (V' - V) / V * 100  -- Percentage increase in volume
  5 * (y - x) = 144 := by sorry

end NUMINAMATH_CALUDE_cube_increase_theorem_l1261_126129


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l1261_126137

theorem quadratic_real_solutions (x y : ℝ) :
  (9 * y^2 + 6 * x * y + 2 * x + 10 = 0) ↔ (x ≤ -10/3 ∨ x ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l1261_126137


namespace NUMINAMATH_CALUDE_five_rooks_on_five_by_five_board_l1261_126102

/-- The number of ways to place n distinct rooks on an n×n chess board
    such that no two rooks share the same row or column -/
def rook_placements (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem: The number of ways to place 5 distinct rooks on a 5×5 chess board,
    such that no two rooks share the same row or column, is equal to 5! (120) -/
theorem five_rooks_on_five_by_five_board :
  rook_placements 5 = 120 := by
  sorry

#eval rook_placements 5  -- Should output 120

end NUMINAMATH_CALUDE_five_rooks_on_five_by_five_board_l1261_126102


namespace NUMINAMATH_CALUDE_distance_between_points_l1261_126140

theorem distance_between_points : ∀ (x1 y1 x2 y2 : ℝ), 
  x1 = 0 ∧ y1 = 6 ∧ x2 = 8 ∧ y2 = 0 → 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1261_126140


namespace NUMINAMATH_CALUDE_elisa_target_amount_l1261_126118

/-- Elisa's target amount problem -/
theorem elisa_target_amount (current_amount additional_amount : ℕ) 
  (h1 : current_amount = 37)
  (h2 : additional_amount = 16) :
  current_amount + additional_amount = 53 := by
  sorry

end NUMINAMATH_CALUDE_elisa_target_amount_l1261_126118


namespace NUMINAMATH_CALUDE_carla_cooking_time_l1261_126109

/-- Represents the cooking time for each item in minutes -/
structure CookingTime where
  waffle : ℕ
  steak : ℕ
  chili : ℕ

/-- Represents the number of items to be cooked -/
structure CookingItems where
  waffle : ℕ
  steak : ℕ
  chili : ℕ

/-- Calculates the total cooking time given the cooking times and items to be cooked -/
def totalCookingTime (time : CookingTime) (items : CookingItems) : ℕ :=
  time.waffle * items.waffle + time.steak * items.steak + time.chili * items.chili

/-- Theorem stating that Carla's total cooking time is 100 minutes -/
theorem carla_cooking_time :
  let time := CookingTime.mk 10 6 20
  let items := CookingItems.mk 3 5 2
  totalCookingTime time items = 100 := by sorry

end NUMINAMATH_CALUDE_carla_cooking_time_l1261_126109


namespace NUMINAMATH_CALUDE_remainder_theorem_l1261_126152

def f (x : ℝ) : ℝ := 5*x^7 - 3*x^6 - 8*x^5 + 3*x^3 + 5*x^2 - 20

def g (x : ℝ) : ℝ := 3*x - 9

theorem remainder_theorem :
  ∃ q : ℝ → ℝ, f = fun x ↦ g x * q x + 6910 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1261_126152


namespace NUMINAMATH_CALUDE_not_q_is_false_l1261_126184

theorem not_q_is_false (p q : Prop) (hp : ¬p) (hq : q) : ¬(¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_q_is_false_l1261_126184


namespace NUMINAMATH_CALUDE_existence_of_binomial_solution_l1261_126162

theorem existence_of_binomial_solution (a b : ℕ+) :
  ∃ (x y : ℕ+), Nat.choose (x + y) 2 = a * x + b * y := by
  sorry

end NUMINAMATH_CALUDE_existence_of_binomial_solution_l1261_126162


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1261_126124

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h1 : a 3 - 3 * a 2 = 2) 
  (h2 : 5 * a 4 = (12 * a 3 + 2 * a 5) / 2) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1261_126124


namespace NUMINAMATH_CALUDE_total_earnings_1200_l1261_126168

/-- Represents the prices for services of a car model -/
structure ModelPrices where
  oil_change : ℕ
  repair : ℕ
  car_wash : ℕ

/-- Represents the number of services performed for a car model -/
structure ModelServices where
  oil_changes : ℕ
  repairs : ℕ
  car_washes : ℕ

/-- Calculates the total earnings for a single car model -/
def modelEarnings (prices : ModelPrices) (services : ModelServices) : ℕ :=
  prices.oil_change * services.oil_changes +
  prices.repair * services.repairs +
  prices.car_wash * services.car_washes

/-- Theorem stating that the total earnings for the day is $1200 -/
theorem total_earnings_1200 
  (prices_A : ModelPrices)
  (prices_B : ModelPrices)
  (prices_C : ModelPrices)
  (services_A : ModelServices)
  (services_B : ModelServices)
  (services_C : ModelServices)
  (h1 : prices_A = ⟨20, 30, 5⟩)
  (h2 : prices_B = ⟨25, 40, 8⟩)
  (h3 : prices_C = ⟨30, 50, 10⟩)
  (h4 : services_A = ⟨5, 10, 15⟩)
  (h5 : services_B = ⟨3, 4, 10⟩)
  (h6 : services_C = ⟨2, 6, 5⟩) :
  modelEarnings prices_A services_A + 
  modelEarnings prices_B services_B + 
  modelEarnings prices_C services_C = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_1200_l1261_126168


namespace NUMINAMATH_CALUDE_broken_flagpole_l1261_126176

theorem broken_flagpole (h : ℝ) (d : ℝ) (x : ℝ) : 
  h = 6 → d = 2 → x * x + d * d = (h - x) * (h - x) → x = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_broken_flagpole_l1261_126176


namespace NUMINAMATH_CALUDE_randy_biscuits_l1261_126159

/-- The number of biscuits Randy is left with after receiving and losing some -/
def biscuits_left (initial : ℕ) (father_gift : ℕ) (mother_gift : ℕ) (brother_ate : ℕ) : ℕ :=
  initial + father_gift + mother_gift - brother_ate

/-- Theorem stating that Randy is left with 40 biscuits -/
theorem randy_biscuits :
  biscuits_left 32 13 15 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_randy_biscuits_l1261_126159


namespace NUMINAMATH_CALUDE_triangle_side_length_l1261_126192

theorem triangle_side_length (y : ℝ) :
  y > 0 →  -- y is positive
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧  -- a and b are positive
    a = 10 ∧  -- shorter leg is 10
    a^2 + b^2 = y^2 ∧  -- Pythagorean theorem
    b = a * Real.sqrt 3 →  -- ratio of sides in a 30-60-90 triangle
  y = 10 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1261_126192


namespace NUMINAMATH_CALUDE_ryan_chinese_time_l1261_126177

/-- The time Ryan spends on learning English and Chinese daily -/
def total_time : ℝ := 3

/-- The time Ryan spends on learning English daily -/
def english_time : ℝ := 2

/-- The time Ryan spends on learning Chinese daily -/
def chinese_time : ℝ := total_time - english_time

theorem ryan_chinese_time : chinese_time = 1 := by sorry

end NUMINAMATH_CALUDE_ryan_chinese_time_l1261_126177


namespace NUMINAMATH_CALUDE_program_output_l1261_126121

def S : ℕ → ℚ
  | 0 => 2
  | n + 1 => 1 / (1 - S n)

theorem program_output : S 2017 = -1 := by
  sorry

end NUMINAMATH_CALUDE_program_output_l1261_126121


namespace NUMINAMATH_CALUDE_direct_proportion_function_m_l1261_126183

theorem direct_proportion_function_m (m : ℝ) : 
  (m^2 - 3 = 1 ∧ m + 2 ≠ 0) ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_direct_proportion_function_m_l1261_126183


namespace NUMINAMATH_CALUDE_unique_modular_equivalence_l1261_126175

theorem unique_modular_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -2050 [ZMOD 13] ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_equivalence_l1261_126175


namespace NUMINAMATH_CALUDE_inequalities_proof_l1261_126195

theorem inequalities_proof :
  (∀ a b c : ℝ, a^2 + b^2 + c^2 ≥ a*b + a*c + b*c) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1261_126195


namespace NUMINAMATH_CALUDE_initial_green_papayas_l1261_126167

/-- The number of green papayas that turned yellow on Friday -/
def friday_yellow : ℕ := 2

/-- The number of green papayas that turned yellow on Sunday -/
def sunday_yellow : ℕ := 2 * friday_yellow

/-- The number of green papayas left on the tree -/
def remaining_green : ℕ := 8

/-- The initial number of green papayas on the tree -/
def initial_green : ℕ := remaining_green + friday_yellow + sunday_yellow

theorem initial_green_papayas : initial_green = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_green_papayas_l1261_126167


namespace NUMINAMATH_CALUDE_age_problem_l1261_126106

/-- The age problem -/
theorem age_problem (sebastian_age : ℕ) (sister_age : ℕ) (father_age : ℕ) : 
  sebastian_age = 40 →
  sister_age = sebastian_age - 10 →
  (sebastian_age - 5) + (sister_age - 5) = 3 * (father_age - 5) / 4 →
  father_age = 90 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1261_126106


namespace NUMINAMATH_CALUDE_intersection_M_N_l1261_126186

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | |x| > 2}

theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1261_126186


namespace NUMINAMATH_CALUDE_ellipse_foci_product_l1261_126131

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  P.1^2 / 16 + P.2^2 / 12 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define the dot product condition
def satisfies_dot_product (P : ℝ × ℝ) : Prop :=
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 9

-- Theorem statement
theorem ellipse_foci_product (P : ℝ × ℝ) :
  is_on_ellipse P → satisfies_dot_product P →
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  (Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_product_l1261_126131


namespace NUMINAMATH_CALUDE_fraction_relation_l1261_126135

theorem fraction_relation (w x y z : ℝ) 
  (h1 : x / y = 5)
  (h2 : y / z = 1 / 2)
  (h3 : z / w = 7)
  (hw : w ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : 
  w / x = 2 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l1261_126135


namespace NUMINAMATH_CALUDE_division_problem_l1261_126170

theorem division_problem : ∃ x : ℝ, 550 - (104 / x) = 545 ∧ x = 20.8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1261_126170


namespace NUMINAMATH_CALUDE_glucose_solution_volume_l1261_126128

/-- The volume of glucose solution containing 15 grams of glucose -/
def volume_15g : ℝ := 100

/-- The volume of glucose solution used in the given condition -/
def volume_given : ℝ := 65

/-- The mass of glucose in the given volume -/
def mass_given : ℝ := 9.75

/-- The target mass of glucose -/
def mass_target : ℝ := 15

theorem glucose_solution_volume :
  (mass_given / volume_given) * volume_15g = mass_target :=
by sorry

end NUMINAMATH_CALUDE_glucose_solution_volume_l1261_126128


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l1261_126105

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^7 ways to put 7 distinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : ways_to_put_balls_in_boxes 7 3 = 3^7 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l1261_126105


namespace NUMINAMATH_CALUDE_misplaced_sheets_count_l1261_126197

/-- Represents a booklet of printed notes -/
structure Booklet where
  total_pages : ℕ
  total_sheets : ℕ
  misplaced_sheets : ℕ
  avg_remaining : ℝ

/-- The theorem stating the number of misplaced sheets -/
theorem misplaced_sheets_count (b : Booklet) 
  (h1 : b.total_pages = 60)
  (h2 : b.total_sheets = 30)
  (h3 : b.avg_remaining = 21) :
  b.misplaced_sheets = 15 := by
  sorry

#check misplaced_sheets_count

end NUMINAMATH_CALUDE_misplaced_sheets_count_l1261_126197


namespace NUMINAMATH_CALUDE_intersection_slope_l1261_126123

/-- Given two lines that intersect at a point, find the slope of one line -/
theorem intersection_slope (m : ℝ) : 
  (∃ (x y : ℝ), y = 2*x + 3 ∧ y = m*x + 1 ∧ x = 1 ∧ y = 5) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l1261_126123


namespace NUMINAMATH_CALUDE_calculation_proof_l1261_126193

theorem calculation_proof : 
  (3.2 - 2.95) / (0.25 * 2 + 1/4) + (2 * 0.3) / (2.3 - (1 + 2/5)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1261_126193


namespace NUMINAMATH_CALUDE_correct_understanding_of_philosophy_l1261_126149

-- Define the characteristics of philosophy
def originatesFromLife (p : Type) : Prop := sorry
def affectsLife (p : Type) : Prop := sorry
def formsSpontaneously (p : Type) : Prop := sorry
def summarizesKnowledge (p : Type) : Prop := sorry

-- Define Yu Wujin's statement
def yuWujinStatement (p : Type) : Prop := sorry

-- Theorem to prove
theorem correct_understanding_of_philosophy (p : Type) :
  yuWujinStatement p ↔ (originatesFromLife p ∧ affectsLife p) :=
sorry

end NUMINAMATH_CALUDE_correct_understanding_of_philosophy_l1261_126149


namespace NUMINAMATH_CALUDE_division_and_addition_l1261_126178

theorem division_and_addition : -4 + 6 / (-2) = -7 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l1261_126178


namespace NUMINAMATH_CALUDE_factorization_equality_l1261_126111

theorem factorization_equality (x : ℝ) : 5*x*(x+2) + 9*(x+2) = (x+2)*(5*x+9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1261_126111


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l1261_126191

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -(x + 2)^2 + 6

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- Theorem: The intersection of the parabola and y-axis is at (0, 2) -/
theorem parabola_y_axis_intersection :
  ∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ y_axis p.1 ∧ p = (0, 2) := by
sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l1261_126191


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l1261_126199

/-- Triangle ABC with vertices A(2,0), B(8,0), and C(5,5) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨2, 0⟩, ⟨8, 0⟩, ⟨5, 5⟩}

/-- The circle circumscribing triangle ABC -/
def circumcircle : Set (ℝ × ℝ) :=
  sorry

/-- A square with side length 5 -/
def square_PQRS : Set (ℝ × ℝ) :=
  sorry

/-- The radius of the circumcircle -/
def radius (circle : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Two vertices of the square lie on the sides of the triangle -/
axiom square_vertices_on_triangle :
  ∃ (P Q : ℝ × ℝ), P ∈ square_PQRS ∧ Q ∈ square_PQRS ∧
    (P.1 - 2) / 3 = P.2 / 5 ∧ (Q.1 - 5) / 3 = -Q.2 / 5

/-- The other two vertices of the square lie on the circumcircle -/
axiom square_vertices_on_circle :
  ∃ (R S : ℝ × ℝ), R ∈ square_PQRS ∧ S ∈ square_PQRS ∧
    R ∈ circumcircle ∧ S ∈ circumcircle

/-- The side length of the square is 5 -/
axiom square_side_length :
  ∀ (X Y : ℝ × ℝ), X ∈ square_PQRS → Y ∈ square_PQRS →
    X ≠ Y → (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 25

theorem circle_radius_is_five :
  radius circumcircle = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l1261_126199


namespace NUMINAMATH_CALUDE_time_per_furniture_piece_l1261_126119

theorem time_per_furniture_piece (chairs tables total_time : ℕ) : 
  chairs = 7 → tables = 3 → total_time = 40 → (chairs + tables) * 4 = total_time := by
  sorry

end NUMINAMATH_CALUDE_time_per_furniture_piece_l1261_126119


namespace NUMINAMATH_CALUDE_tadpole_fish_difference_l1261_126116

def initial_fish : ℕ := 50
def tadpole_ratio : ℕ := 3
def fish_caught : ℕ := 7
def tadpole_development_ratio : ℚ := 1/2

theorem tadpole_fish_difference : 
  (tadpole_ratio * initial_fish) * tadpole_development_ratio - (initial_fish - fish_caught) = 32 := by
  sorry

end NUMINAMATH_CALUDE_tadpole_fish_difference_l1261_126116


namespace NUMINAMATH_CALUDE_initial_peaches_calculation_l1261_126148

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked : ℕ := 42

/-- The total number of peaches Sally has now -/
def total_peaches_now : ℕ := 55

/-- The initial number of peaches at Sally's roadside fruit dish -/
def initial_peaches : ℕ := total_peaches_now - peaches_picked

theorem initial_peaches_calculation :
  initial_peaches = total_peaches_now - peaches_picked :=
by sorry

end NUMINAMATH_CALUDE_initial_peaches_calculation_l1261_126148


namespace NUMINAMATH_CALUDE_abs_value_inequality_l1261_126126

theorem abs_value_inequality (x : ℝ) : 2 ≤ |x - 3| ∧ |x - 3| ≤ 4 ↔ (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_abs_value_inequality_l1261_126126


namespace NUMINAMATH_CALUDE_calculate_expression_l1261_126127

theorem calculate_expression : 500 * 996 * 0.0996 * 20 + 5000 = 997016 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1261_126127


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1261_126130

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1261_126130


namespace NUMINAMATH_CALUDE_complex_power_sum_l1261_126114

theorem complex_power_sum (w : ℂ) (hw : w^2 - w + 1 = 0) : 
  w^103 + w^104 + w^105 + w^106 + w^107 = -1 := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1261_126114


namespace NUMINAMATH_CALUDE_bug_position_after_1995_jumps_l1261_126189

/-- Represents the five points on the circle -/
inductive Point
| one
| two
| three
| four
| five

/-- Determines if a point is odd -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true

/-- Moves the bug according to the rules -/
def move (p : Point) : Point :=
  match p with
  | Point.one => Point.two
  | Point.two => Point.five
  | Point.three => Point.four
  | Point.four => Point.two
  | Point.five => Point.one

/-- Performs n jumps starting from a given point -/
def jump (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => jump (move start) n

theorem bug_position_after_1995_jumps :
  jump Point.three 1995 = Point.one := by sorry

end NUMINAMATH_CALUDE_bug_position_after_1995_jumps_l1261_126189


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1261_126146

/-- Given the expression 5(x^3 - 3x^2 + 4) - 8(2x^3 - x^2 - 2), 
    the sum of the squares of its coefficients when fully simplified is 1466. -/
theorem sum_of_squared_coefficients : 
  let expr := fun (x : ℝ) => 5 * (x^3 - 3*x^2 + 4) - 8 * (2*x^3 - x^2 - 2)
  let simplified := fun (x : ℝ) => -11*x^3 - 7*x^2 + 36
  (∀ x, expr x = simplified x) → 
  (-11)^2 + (-7)^2 + 36^2 = 1466 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1261_126146


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1261_126185

/-- Given a man's speed with the current and the speed of the current, 
    calculates the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  2 * speed_with_current - 3 * speed_of_current

/-- Theorem stating that given the specific speeds in the problem, 
    the man's speed against the current is 11.2 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 18 3.4 = 11.2 := by
  sorry

#eval speed_against_current 18 3.4

end NUMINAMATH_CALUDE_mans_speed_against_current_l1261_126185


namespace NUMINAMATH_CALUDE_square_area_equals_perimeter_l1261_126107

theorem square_area_equals_perimeter (s : ℝ) (h : s > 0) : s^2 = 4*s → s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equals_perimeter_l1261_126107


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l1261_126122

theorem least_subtrahend_for_divisibility (n : ℕ) (a b c : ℕ) (h_n : n = 157632) (h_a : a = 12) (h_b : b = 18) (h_c : c = 24) :
  ∃ (k : ℕ), k = 24 ∧
  (∀ (m : ℕ), m < k → ¬(∃ (q : ℕ), n - m = q * a ∧ n - m = q * b ∧ n - m = q * c)) ∧
  (∃ (q : ℕ), n - k = q * a ∧ n - k = q * b ∧ n - k = q * c) :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l1261_126122


namespace NUMINAMATH_CALUDE_circle_C_properties_l1261_126151

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 2)^2 = 9

-- Define the line L where the center of C lies
def line_L (x y : ℝ) : Prop :=
  x + 2*y + 1 = 0

-- Define the line that potentially intersects C
def intersecting_line (a x y : ℝ) : Prop :=
  a*x - y + 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, 0)

theorem circle_C_properties :
  -- The circle C passes through M(0,-2) and N(3,1)
  circle_C 0 (-2) ∧ circle_C 3 1 ∧
  -- The center of C lies on line L
  ∃ (cx cy : ℝ), line_L cx cy ∧ ∀ (x y : ℝ), circle_C x y ↔ (x - cx)^2 + (y - cy)^2 = 9 ∧
  -- There's no real a such that the line ax-y+1=0 intersects C at two points
  -- and is perpendicularly bisected by the line through P
  ¬ ∃ (a : ℝ), 
    (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
                          intersecting_line a x₁ y₁ ∧ intersecting_line a x₂ y₂) ∧
    (∃ (mx my : ℝ), circle_C mx my ∧ 
                    (mx - point_P.1) * (x₂ - x₁) + (my - point_P.2) * (y₂ - y₁) = 0 ∧
                    2 * mx = x₁ + x₂ ∧ 2 * my = y₁ + y₂) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l1261_126151


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1261_126196

theorem quadratic_expression_value (x : ℝ) (h : 3 * x^2 + 5 * x + 1 = 0) :
  (x + 2)^2 + x * (2 * x + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1261_126196


namespace NUMINAMATH_CALUDE_inequality_proof_l1261_126141

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) ≥ 12) ∧
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1261_126141


namespace NUMINAMATH_CALUDE_problem_statement_l1261_126173

theorem problem_statement (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : (a + b)^2012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1261_126173


namespace NUMINAMATH_CALUDE_dot_product_specific_vectors_l1261_126117

theorem dot_product_specific_vectors (α : ℝ) : 
  let a : ℝ × ℝ := (Real.cos α, Real.sin α)
  let b : ℝ × ℝ := (Real.cos (π/3 + α), Real.sin (π/3 + α))
  (a.1 * b.1 + a.2 * b.2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_dot_product_specific_vectors_l1261_126117


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_property_l1261_126155

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

-- State the theorem
theorem isosceles_right_triangle_property 
  (a : ℝ) (x₁ x₂ : ℝ) (t : ℝ) : 
  x₁ < x₂ →
  f a x₁ = 0 →
  f a x₂ = 0 →
  (∃ x₀, x₁ < x₀ ∧ x₀ < x₂ ∧ 
    (x₂ - x₁) / 2 = -f a x₀ ∧
    (x₂ - x₀) = (x₀ - x₁)) →
  Real.sqrt ((x₂ - 1) / (x₁ - 1)) = t →
  a * t - (a + t) = 1 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_property_l1261_126155


namespace NUMINAMATH_CALUDE_problem_solution_l1261_126147

def set_A (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a^2 + 1}
def set_B (x : ℝ) : Set ℝ := {0, 1, x}

theorem problem_solution :
  (∀ a : ℝ, -3 ∈ set_A a → a = 0 ∨ a = -1) ∧
  (∀ x : ℝ, x^2 ∈ set_B x ∧ x ≠ 0 ∧ x ≠ 1 → x = -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1261_126147


namespace NUMINAMATH_CALUDE_m_less_than_n_l1261_126181

/-- Represents a quadratic function f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines N based on the coefficients of a quadratic function -/
def N (f : QuadraticFunction) : ℝ :=
  |f.a + f.b + f.c| + |2*f.a - f.b|

/-- Defines M based on the coefficients of a quadratic function -/
def M (f : QuadraticFunction) : ℝ :=
  |f.a - f.b + f.c| + |2*f.a + f.b|

/-- Theorem stating that M < N for a quadratic function satisfying certain conditions -/
theorem m_less_than_n (f : QuadraticFunction)
  (h1 : f.a + f.b + f.c < 0)
  (h2 : f.a - f.b + f.c > 0)
  (h3 : f.a > 0)
  (h4 : -f.b / (2 * f.a) > 1) :
  M f < N f := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_n_l1261_126181


namespace NUMINAMATH_CALUDE_min_sum_squares_l1261_126104

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  ∃ (m : ℝ), m = (1:ℝ)/14 ∧ x^2 + y^2 + z^2 ≥ m ∧ 
  (x^2 + y^2 + z^2 = m ↔ x = (1:ℝ)/14 ∧ y = (1:ℝ)/7 ∧ z = (3:ℝ)/14) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1261_126104


namespace NUMINAMATH_CALUDE_robin_gum_packages_l1261_126198

theorem robin_gum_packages :
  ∀ (packages : ℕ),
  (7 * packages + 6 = 41) →
  packages = 5 := by
sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l1261_126198


namespace NUMINAMATH_CALUDE_geometry_theorems_l1261_126120

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Axioms for the properties of parallel and perpendicular
axiom parallel_planes_transitive : 
  ∀ (α β γ : Plane), parallel_planes α β → parallel_planes α γ → parallel_planes β γ

axiom perpendicular_parallel_planes : 
  ∀ (m : Line) (α β : Plane), 
    perpendicular_line_plane m α → parallel_line_plane m β → perpendicular_planes α β

-- Theorem to prove
theorem geometry_theorems :
  (∀ (α β γ : Plane), parallel_planes α β → parallel_planes α γ → parallel_planes β γ) ∧
  (∀ (m : Line) (α β : Plane), 
    perpendicular_line_plane m α → parallel_line_plane m β → perpendicular_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorems_l1261_126120


namespace NUMINAMATH_CALUDE_equation_solution_l1261_126194

theorem equation_solution : 
  ∃ x : ℚ, x - (3 : ℚ) / 4 = (5 : ℚ) / 12 - (1 : ℚ) / 3 ∧ x = (5 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1261_126194


namespace NUMINAMATH_CALUDE_integer_divisibility_problem_l1261_126174

theorem integer_divisibility_problem (a b : ℤ) :
  (a^6 + 1) ∣ (b^11 - 2023*b^3 + 40*b) →
  (a^4 - 1) ∣ (b^10 - 2023*b^2 - 41) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_divisibility_problem_l1261_126174


namespace NUMINAMATH_CALUDE_polyhedron_face_edges_divisible_by_three_l1261_126180

-- Define a polyhedron
structure Polyhedron where
  faces : Set Face
  edges : Set Edge
  vertices : Set Vertex

-- Define a face
structure Face where
  edges : Set Edge

-- Define an edge
structure Edge where
  vertices : Fin 2 → Vertex

-- Define a vertex
structure Vertex where

-- Define a color
inductive Color
  | White
  | Black

-- Define a coloring function
def coloring (p : Polyhedron) : Face → Color := sorry

-- Define the number of edges for a face
def numEdges (f : Face) : Nat := sorry

-- Define adjacency for faces
def adjacent (f1 f2 : Face) : Prop := sorry

-- Theorem statement
theorem polyhedron_face_edges_divisible_by_three 
  (p : Polyhedron) 
  (h1 : ∀ f1 f2 : Face, f1 ∈ p.faces → f2 ∈ p.faces → adjacent f1 f2 → coloring p f1 ≠ coloring p f2)
  (h2 : ∃ f : Face, f ∈ p.faces ∧ ∀ f' : Face, f' ∈ p.faces → f' ≠ f → (numEdges f') % 3 = 0) :
  ∀ f : Face, f ∈ p.faces → (numEdges f) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_face_edges_divisible_by_three_l1261_126180


namespace NUMINAMATH_CALUDE_amy_muffins_l1261_126154

def muffins_series (n : ℕ) : ℕ := n * (n + 1) / 2

theorem amy_muffins :
  let days : ℕ := 5
  let start_muffins : ℕ := 1
  let leftover_muffins : ℕ := 7
  let total_brought := muffins_series days
  total_brought + leftover_muffins = 22 :=
by sorry

end NUMINAMATH_CALUDE_amy_muffins_l1261_126154


namespace NUMINAMATH_CALUDE_bus_problem_l1261_126143

/-- The number of people who got off the bus -/
def people_got_off (initial : ℕ) (final : ℕ) : ℕ := initial - final

/-- Theorem stating that 47 people got off the bus -/
theorem bus_problem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 90) 
  (h2 : final = 43) : 
  people_got_off initial final = 47 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l1261_126143


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1261_126145

theorem geometric_sequence_problem (a : ℕ → ℚ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 5 = (5 : ℚ) / 3 →                   -- 5th term equals constant term of expansion
  a 3 * a 7 = (25 : ℚ) / 9 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1261_126145


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l1261_126134

theorem right_triangle_hypotenuse_and_perimeter :
  ∀ (a b h p : ℝ),
  a = 24 →
  b = 32 →
  h^2 = a^2 + b^2 →
  p = a + b + h →
  h = 40 ∧ p = 96 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l1261_126134


namespace NUMINAMATH_CALUDE_sphere_identical_views_other_bodies_different_views_l1261_126150

-- Define the geometric bodies
inductive GeometricBody
  | Cylinder
  | Cone
  | Sphere
  | TriangularPyramid

-- Define a function to check if a geometric body has identical views
def hasIdenticalViews (body : GeometricBody) : Prop :=
  match body with
  | GeometricBody.Sphere => true
  | _ => false

-- Theorem stating that only a sphere has identical views
theorem sphere_identical_views :
  ∀ (body : GeometricBody),
    hasIdenticalViews body ↔ body = GeometricBody.Sphere :=
by sorry

-- Prove that other geometric bodies do not have identical views
theorem other_bodies_different_views :
  ¬(hasIdenticalViews GeometricBody.Cylinder) ∧
  ¬(hasIdenticalViews GeometricBody.Cone) ∧
  ¬(hasIdenticalViews GeometricBody.TriangularPyramid) :=
by sorry

end NUMINAMATH_CALUDE_sphere_identical_views_other_bodies_different_views_l1261_126150


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l1261_126190

theorem hexagon_angle_measure (N U M B E R S : ℝ) : 
  -- Hexagon condition
  N + U + M + B + E + R + S = 720 →
  -- Congruent angles
  N = M →
  B = R →
  -- Supplementary angles
  U + S = 180 →
  -- Conclusion
  E = 180 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l1261_126190


namespace NUMINAMATH_CALUDE_evaluate_expression_l1261_126133

theorem evaluate_expression : (728 * 728) - (727 * 729) = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1261_126133


namespace NUMINAMATH_CALUDE_inequality_subtraction_l1261_126112

theorem inequality_subtraction (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_subtraction_l1261_126112


namespace NUMINAMATH_CALUDE_min_filtration_layers_l1261_126165

theorem min_filtration_layers (a : ℝ) (ha : a > 0) : 
  (∃ n : ℕ, n ≥ 5 ∧ a * (4/5)^n ≤ (1/3) * a ∧ ∀ m : ℕ, m < 5 → a * (4/5)^m > (1/3) * a) :=
sorry

end NUMINAMATH_CALUDE_min_filtration_layers_l1261_126165


namespace NUMINAMATH_CALUDE_stating_dieRollSumWays_l1261_126142

/-- Represents the number of faces on a standard die -/
def diefaces : ℕ := 6

/-- Represents the number of times the die is rolled -/
def numrolls : ℕ := 6

/-- Represents the target sum we're aiming for -/
def targetsum : ℕ := 21

/-- 
Calculates the number of ways to roll a fair six-sided die 'numrolls' times 
such that the sum of the outcomes is 'targetsum'
-/
def numWaysToSum (diefaces numrolls targetsum : ℕ) : ℕ := sorry

/-- 
Theorem stating that the number of ways to roll a fair six-sided die six times 
such that the sum of the outcomes is 21 is equal to 15504
-/
theorem dieRollSumWays : numWaysToSum diefaces numrolls targetsum = 15504 := by sorry

end NUMINAMATH_CALUDE_stating_dieRollSumWays_l1261_126142


namespace NUMINAMATH_CALUDE_triangle_expression_simplification_l1261_126163

/-- Given a triangle with side lengths x, y, and z, prove that |x+y-z|-2|y-x-z| = -x + 3y - 3z -/
theorem triangle_expression_simplification
  (x y z : ℝ)
  (hxy : x + y > z)
  (hyz : y + z > x)
  (hxz : x + z > y) :
  |x + y - z| - 2 * |y - x - z| = -x + 3*y - 3*z := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_simplification_l1261_126163


namespace NUMINAMATH_CALUDE_james_barrels_l1261_126188

/-- The number of barrels James has -/
def number_of_barrels : ℕ := 3

/-- The capacity of a cask in gallons -/
def cask_capacity : ℕ := 20

/-- The capacity of a barrel in gallons -/
def barrel_capacity : ℕ := 2 * cask_capacity + 3

/-- The total storage capacity in gallons -/
def total_capacity : ℕ := 172

/-- Proof that James has 3 barrels -/
theorem james_barrels :
  number_of_barrels * barrel_capacity + cask_capacity = total_capacity :=
by sorry

end NUMINAMATH_CALUDE_james_barrels_l1261_126188


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1261_126171

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by
  sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) ↔ (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1261_126171


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1261_126182

theorem arithmetic_calculation : (10 - 9 + 8) * 7 + 6 - 5 * (4 - 3 + 2) - 1 = 53 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1261_126182


namespace NUMINAMATH_CALUDE_top_pyramid_volume_calculation_l1261_126136

/-- A right square pyramid with given dimensions -/
structure RightSquarePyramid where
  base_edge : ℝ
  slant_edge : ℝ

/-- The volume of the top portion of a right square pyramid cut by a plane parallel to its base -/
def top_pyramid_volume (p : RightSquarePyramid) (cut_ratio : ℝ) : ℝ :=
  sorry

/-- The main theorem stating the volume of the top portion of the cut pyramid -/
theorem top_pyramid_volume_calculation (p : RightSquarePyramid) 
  (h_base : p.base_edge = 10 * Real.sqrt 2)
  (h_slant : p.slant_edge = 12)
  (h_cut_ratio : cut_ratio = 1/4) :
  top_pyramid_volume p cut_ratio = 84.375 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_top_pyramid_volume_calculation_l1261_126136


namespace NUMINAMATH_CALUDE_three_numbers_average_l1261_126161

theorem three_numbers_average (a b c : ℝ) 
  (h1 : a + (b + c)/2 = 65)
  (h2 : b + (a + c)/2 = 69)
  (h3 : c + (a + b)/2 = 76) :
  (a + b + c)/3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_average_l1261_126161


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1261_126164

theorem quadratic_form_sum (a b c : ℝ) : 
  (∀ x, 8 * x^2 - 48 * x - 320 = a * (x + b)^2 + c) → 
  a + b + c = -387 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1261_126164


namespace NUMINAMATH_CALUDE_sum_of_ages_l1261_126103

/-- Given a son who is 27 years old and a woman whose age is three years more
    than twice her son's age, prove that the sum of their ages is 84 years. -/
theorem sum_of_ages (son_age : ℕ) (woman_age : ℕ) : son_age = 27 →
  woman_age = 2 * son_age + 3 → son_age + woman_age = 84 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1261_126103


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_l1261_126110

theorem smallest_solution_quartic (x : ℝ) : 
  (x^4 - 50*x^2 + 625 = 0) → (∃ y : ℝ, y^4 - 50*y^2 + 625 = 0 ∧ y ≤ x) → x ≥ -5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_l1261_126110


namespace NUMINAMATH_CALUDE_parabola_b_value_l1261_126115

/-- A parabola passing through three given points has a specific b value -/
theorem parabola_b_value (b c : ℚ) :
  ((-1)^2 + b*(-1) + c = -11) →
  (3^2 + b*3 + c = 17) →
  (2^2 + b*2 + c = 6) →
  b = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l1261_126115


namespace NUMINAMATH_CALUDE_no_four_naturals_exist_l1261_126179

theorem no_four_naturals_exist : ¬∃ (a b c d : ℕ), 
  a + b + c + d = 2^100 ∧ a * b * c * d = 17^100 := by
  sorry

end NUMINAMATH_CALUDE_no_four_naturals_exist_l1261_126179


namespace NUMINAMATH_CALUDE_jorges_total_goals_l1261_126157

/-- The total number of goals Jorge scored over two seasons -/
def total_goals (last_season_goals this_season_goals : ℕ) : ℕ :=
  last_season_goals + this_season_goals

/-- Theorem stating that Jorge's total goals over two seasons is 343 -/
theorem jorges_total_goals :
  total_goals 156 187 = 343 := by
  sorry

end NUMINAMATH_CALUDE_jorges_total_goals_l1261_126157


namespace NUMINAMATH_CALUDE_thermal_equilibrium_problem_l1261_126108

/-- Represents the thermal equilibrium in a system of water and metal bars -/
structure ThermalSystem where
  initialWaterTemp : ℝ
  initialBarTemp : ℝ
  firstEquilibriumTemp : ℝ
  finalEquilibriumTemp : ℝ

/-- The thermal equilibrium problem -/
theorem thermal_equilibrium_problem (system : ThermalSystem)
  (h1 : system.initialWaterTemp = 100)
  (h2 : system.initialBarTemp = 20)
  (h3 : system.firstEquilibriumTemp = 80)
  : system.finalEquilibriumTemp = 68 := by
  sorry

end NUMINAMATH_CALUDE_thermal_equilibrium_problem_l1261_126108


namespace NUMINAMATH_CALUDE_four_identical_differences_l1261_126113

theorem four_identical_differences (S : Finset ℕ) : 
  S.card = 20 → (∀ n ∈ S, n < 70) → 
  ∃ (d : ℕ) (a b c d e f g h : ℕ), 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S ∧
    a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ g ≠ h ∧
    b - a = d - c ∧ d - c = f - e ∧ f - e = h - g ∧ h - g = d :=
by sorry

end NUMINAMATH_CALUDE_four_identical_differences_l1261_126113


namespace NUMINAMATH_CALUDE_election_votes_total_l1261_126156

theorem election_votes_total (winning_percentage : ℚ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 60 / 100 →
  majority = 1300 →
  winning_percentage * total_votes - (1 - winning_percentage) * total_votes = majority →
  total_votes = 6500 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_total_l1261_126156


namespace NUMINAMATH_CALUDE_no_positive_integers_satisfying_equation_l1261_126132

theorem no_positive_integers_satisfying_equation : 
  ¬∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 2) = n * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integers_satisfying_equation_l1261_126132


namespace NUMINAMATH_CALUDE_binary_operation_equality_l1261_126153

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its binary representation as a list of bits. -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec to_bits (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
    to_bits n

/-- The first binary number in the problem: 110110₂ -/
def num1 : List Bool := [true, true, false, true, true, false]

/-- The second binary number in the problem: 101010₂ -/
def num2 : List Bool := [true, false, true, false, true, false]

/-- The divisor in the problem: 100₂ -/
def divisor : List Bool := [true, false, false]

/-- The expected result: 111001101100₂ -/
def expected_result : List Bool := [true, true, true, false, false, true, true, false, true, true, false, false]

/-- Theorem stating the equality of the binary operation and the expected result -/
theorem binary_operation_equality :
  nat_to_binary ((binary_to_nat num1 * binary_to_nat num2) / binary_to_nat divisor) = expected_result :=
sorry

end NUMINAMATH_CALUDE_binary_operation_equality_l1261_126153


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1261_126158

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 72) (h2 : Nat.gcd a c = 240) :
  (∃ (x y z : ℕ+), x = a ∧ y = b ∧ z = c ∧ Nat.gcd y z = 24) ∧
  (∀ (p q : ℕ+), Nat.gcd p q < 24 → ¬(∃ (r : ℕ+), Nat.gcd r p = 72 ∧ Nat.gcd r q = 240)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1261_126158


namespace NUMINAMATH_CALUDE_complex_modulus_product_l1261_126139

theorem complex_modulus_product : Complex.abs ((10 - 6*I) * (7 + 24*I)) = 25 * Real.sqrt 136 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l1261_126139
