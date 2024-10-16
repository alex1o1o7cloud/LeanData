import Mathlib

namespace NUMINAMATH_CALUDE_max_fraction_two_digit_nums_l1760_176000

theorem max_fraction_two_digit_nums (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17 := by
sorry

end NUMINAMATH_CALUDE_max_fraction_two_digit_nums_l1760_176000


namespace NUMINAMATH_CALUDE_tree_height_proof_l1760_176037

/-- The initial height of a tree that grows 0.5 feet per year for 6 years and is 1/6 taller
    at the end of the 6th year compared to the 4th year. -/
def initial_tree_height : ℝ :=
  let growth_rate : ℝ := 0.5
  let years : ℕ := 6
  let h : ℝ := 4  -- Initial height to be proved
  h

theorem tree_height_proof (h : ℝ) (growth_rate : ℝ) (years : ℕ) 
    (h_growth : growth_rate = 0.5)
    (h_years : years = 6)
    (h_ratio : h + years * growth_rate = (h + 4 * growth_rate) * (1 + 1/6)) :
  h = initial_tree_height :=
sorry

#check tree_height_proof

end NUMINAMATH_CALUDE_tree_height_proof_l1760_176037


namespace NUMINAMATH_CALUDE_chloe_shoes_altered_l1760_176041

/-- Given the cost per shoe and total cost, calculate the number of pairs of shoes to be altered. -/
def shoesAltered (costPerShoe : ℕ) (totalCost : ℕ) : ℕ :=
  (totalCost / costPerShoe) / 2

/-- Theorem: Given the specific costs, prove that Chloe wants to get 14 pairs of shoes altered. -/
theorem chloe_shoes_altered :
  shoesAltered 37 1036 = 14 := by
  sorry

end NUMINAMATH_CALUDE_chloe_shoes_altered_l1760_176041


namespace NUMINAMATH_CALUDE_buffy_breath_holding_time_l1760_176019

/-- Represents the breath-holding times of Kelly, Brittany, and Buffy in seconds -/
structure BreathHoldingTimes where
  kelly : ℕ
  brittany : ℕ
  buffy : ℕ

/-- The breath-holding contest results -/
def contest : BreathHoldingTimes :=
  { kelly := 3 * 60,  -- Kelly's time in seconds
    brittany := 3 * 60 - 20,  -- Brittany's time is 20 seconds less than Kelly's
    buffy := (3 * 60 - 20) - 40  -- Buffy's time is 40 seconds less than Brittany's
  }

/-- Theorem stating that Buffy held her breath for 120 seconds -/
theorem buffy_breath_holding_time :
  contest.buffy = 120 := by
  sorry

end NUMINAMATH_CALUDE_buffy_breath_holding_time_l1760_176019


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l1760_176074

theorem min_value_of_expression (x y : ℝ) : 
  (x^2 * y^2 - 1)^2 + (x^2 + y^2)^2 ≥ 1 := by
  sorry

theorem lower_bound_achievable : 
  ∃ x y : ℝ, (x^2 * y^2 - 1)^2 + (x^2 + y^2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l1760_176074


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l1760_176091

theorem opposite_of_negative_two_thirds : 
  -(-(2/3 : ℚ)) = 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l1760_176091


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1760_176036

theorem negation_of_proposition (a : ℝ) : 
  ¬(a ≠ 0 → a^2 > 0) ↔ (a = 0 → a^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1760_176036


namespace NUMINAMATH_CALUDE_unique_solution_system_l1760_176055

theorem unique_solution_system :
  ∃! (a b c d : ℝ),
    (a * b + c + d = 3) ∧
    (b * c + d + a = 5) ∧
    (c * d + a + b = 2) ∧
    (d * a + b + c = 6) ∧
    (a = 2) ∧ (b = 0) ∧ (c = 0) ∧ (d = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1760_176055


namespace NUMINAMATH_CALUDE_cube_of_negative_a_b_squared_l1760_176002

theorem cube_of_negative_a_b_squared (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_a_b_squared_l1760_176002


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_149_l1760_176004

theorem first_nonzero_digit_after_decimal_1_149 : ∃ (n : ℕ) (d : ℕ),
  (1 : ℚ) / 149 = (n : ℚ) / 10^(d + 1) + (7 : ℚ) / 10^(d + 2) + (r : ℚ)
  ∧ 0 ≤ r
  ∧ r < 1 / 10^(d + 2)
  ∧ n < 10^(d + 1) :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_149_l1760_176004


namespace NUMINAMATH_CALUDE_song_book_cost_l1760_176063

/-- The cost of the song book given the total amount spent and the cost of the trumpet -/
theorem song_book_cost (total_spent : ℚ) (trumpet_cost : ℚ) (h1 : total_spent = 151) (h2 : trumpet_cost = 145.16) :
  total_spent - trumpet_cost = 5.84 := by
  sorry

end NUMINAMATH_CALUDE_song_book_cost_l1760_176063


namespace NUMINAMATH_CALUDE_smallest_percentage_correct_l1760_176071

theorem smallest_percentage_correct (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.9) 
  (h2 : p2 = 0.8) 
  (h3 : p3 = 0.7) : 
  (1 - ((1 - p1) + (1 - p2) + (1 - p3))) ≥ 0.4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_percentage_correct_l1760_176071


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l1760_176031

theorem gcd_factorial_eight_and_factorial_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 1920 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l1760_176031


namespace NUMINAMATH_CALUDE_line_transformation_l1760_176073

open Matrix

/-- The matrix representing the linear transformation -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 0, 1]

/-- The original line equation: x + y + 2 = 0 -/
def original_line (x y : ℝ) : Prop := x + y + 2 = 0

/-- The transformed line equation: x + 2y + 2 = 0 -/
def transformed_line (x y : ℝ) : Prop := x + 2*y + 2 = 0

/-- Theorem stating that the linear transformation maps the original line to the transformed line -/
theorem line_transformation :
  ∀ (x y : ℝ), original_line x y → 
  ∃ (x' y' : ℝ), M.mulVec ![x', y'] = ![x, y] ∧ transformed_line x' y' := by
sorry

end NUMINAMATH_CALUDE_line_transformation_l1760_176073


namespace NUMINAMATH_CALUDE_right_triangle_case1_right_triangle_case2_l1760_176023

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- length of BC
  b : ℝ  -- length of AC
  c : ℝ  -- length of AB
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

-- Theorem for the first scenario
theorem right_triangle_case1 (t : RightTriangle) (h1 : t.a = 7) (h2 : t.b = 24) : t.c = 25 := by
  sorry

-- Theorem for the second scenario
theorem right_triangle_case2 (t : RightTriangle) (h1 : t.a = 12) (h2 : t.c = 13) : t.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_case1_right_triangle_case2_l1760_176023


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l1760_176093

/-- Given a hyperbola with equation x²/m + y²/(2m-1) = 1, prove that the range of m is 0 < m < 1/2 -/
theorem hyperbola_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2/m + y^2/(2*m-1) = 1) → 0 < m ∧ m < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l1760_176093


namespace NUMINAMATH_CALUDE_smallest_coprime_to_210_l1760_176096

theorem smallest_coprime_to_210 : 
  ∃ (x : ℕ), x > 1 ∧ Nat.gcd x 210 = 1 ∧ ∀ (y : ℕ), y > 1 ∧ y < x → Nat.gcd y 210 ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_coprime_to_210_l1760_176096


namespace NUMINAMATH_CALUDE_det_A_eq_22_l1760_176053

def A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -5; -4, 6]

theorem det_A_eq_22 : A.det = 22 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_22_l1760_176053


namespace NUMINAMATH_CALUDE_expression_evaluation_l1760_176018

theorem expression_evaluation : (3 * 5 * 6) * (1/3 + 1/5 + 1/6) = 63 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1760_176018


namespace NUMINAMATH_CALUDE_max_a_value_l1760_176079

theorem max_a_value (x y : ℝ) (hx : 0 < x ∧ x ≤ 2) (hy : 0 < y ∧ y ≤ 2) (hxy : x * y = 2) :
  (∀ a : ℝ, (∀ x y : ℝ, 0 < x ∧ x ≤ 2 → 0 < y ∧ y ≤ 2 → x * y = 2 → 
    6 - 2*x - y ≥ a*(2 - x)*(4 - y)) → a ≤ 1) ∧ 
  (∃ a : ℝ, a = 1 ∧ ∀ x y : ℝ, 0 < x ∧ x ≤ 2 → 0 < y ∧ y ≤ 2 → x * y = 2 → 
    6 - 2*x - y ≥ a*(2 - x)*(4 - y)) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l1760_176079


namespace NUMINAMATH_CALUDE_sqrt_sum_approximation_l1760_176054

theorem sqrt_sum_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |((Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 1.00) / (Real.sqrt 0.49)) - 2.6507| < ε :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_approximation_l1760_176054


namespace NUMINAMATH_CALUDE_least_six_digit_binary_l1760_176062

theorem least_six_digit_binary : ∃ n : ℕ, n = 32 ∧ 
  (∀ m : ℕ, m < n → (Nat.log 2 m).succ < 6) ∧
  (Nat.log 2 n).succ = 6 :=
sorry

end NUMINAMATH_CALUDE_least_six_digit_binary_l1760_176062


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_l1760_176067

def square_of_1222222221 : ℕ := 1493822537037038241

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_square :
  sum_of_digits square_of_1222222221 = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_l1760_176067


namespace NUMINAMATH_CALUDE_increasing_g_implies_m_bound_l1760_176033

open Real

theorem increasing_g_implies_m_bound (m : ℝ) :
  (∀ x > 2, Monotone (fun x => (x - m) * (exp x - x) - exp x + x^2 + x)) →
  m ≤ (2 * exp 2 + 1) / (exp 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_increasing_g_implies_m_bound_l1760_176033


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l1760_176081

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpoint_octagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of the midpoint octagon is 1/4 of the original octagon's area -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpoint_octagon o) = (1/4) * area o :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l1760_176081


namespace NUMINAMATH_CALUDE_jos_number_l1760_176035

theorem jos_number (n k l : ℕ) : 
  0 < n ∧ n < 150 ∧ n = 9 * k - 2 ∧ n = 8 * l - 4 →
  n ≤ 132 ∧ (∃ (k' l' : ℕ), 132 = 9 * k' - 2 ∧ 132 = 8 * l' - 4) :=
by sorry

end NUMINAMATH_CALUDE_jos_number_l1760_176035


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l1760_176001

theorem isosceles_right_triangle_roots (a b : ℂ) : 
  a ^ 2 = 2 * b ∧ b ≠ 0 ↔ 
  ∃ (x₁ x₂ : ℂ), x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
    (∀ x, x ^ 2 + a * x + b = 0 ↔ x = x₁ ∨ x = x₂) ∧
    (x₂ / x₁ = Complex.I ∨ x₂ / x₁ = -Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l1760_176001


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l1760_176044

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem largest_digit_divisible_by_6 :
  ∀ N : ℕ, N ≤ 9 →
    (is_divisible_by_6 (45670 + N) → N ≤ 8) ∧
    is_divisible_by_6 (45670 + 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l1760_176044


namespace NUMINAMATH_CALUDE_trigonometric_equalities_l1760_176046

theorem trigonometric_equalities
  (α β γ a b c : ℝ)
  (h_alpha : 0 < α ∧ α < π)
  (h_beta : 0 < β ∧ β < π)
  (h_gamma : 0 < γ ∧ γ < π)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_c : c > 0)
  (h_b_eq : b = (c * (Real.cos α + Real.cos β * Real.cos γ)) / (Real.sin γ)^2)
  (h_a_eq : a = (c * (Real.cos β + Real.cos α * Real.cos γ)) / (Real.sin γ)^2)
  (h_identity : 1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0) :
  (Real.cos α + Real.cos β * Real.cos γ = Real.sin α * Real.sin β) ∧
  (Real.cos β + Real.cos α * Real.cos γ = Real.sin α * Real.sin γ) ∧
  (Real.cos γ + Real.cos α * Real.cos β = Real.sin β * Real.sin γ) ∧
  (a * Real.sin γ = c * Real.sin α) ∧
  (b * Real.sin γ = c * Real.sin β) ∧
  (c * Real.sin α = a * Real.sin γ) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equalities_l1760_176046


namespace NUMINAMATH_CALUDE_empire_state_elephant_equivalence_l1760_176043

/-- The number of Empire State Buildings equivalent to 1 billion elephants -/
def empire_state_buildings : ℕ := 25000

/-- The number of elephants equivalent to the total weight of Empire State Buildings -/
def elephants : ℕ := 1000000000

/-- The number of elephants equivalent to one Empire State Building -/
def elephants_per_building : ℕ := elephants / empire_state_buildings

theorem empire_state_elephant_equivalence :
  elephants_per_building = 40000 :=
sorry

end NUMINAMATH_CALUDE_empire_state_elephant_equivalence_l1760_176043


namespace NUMINAMATH_CALUDE_least_product_for_divisibility_least_product_for_cross_divisibility_l1760_176080

theorem least_product_for_divisibility (a b : ℕ+) :
  (∃ k : ℕ, a.val^a.val * b.val^b.val = 2000 * k) →
  (∀ c d : ℕ+, (∃ l : ℕ, c.val^c.val * d.val^d.val = 2000 * l) → c.val * d.val ≥ 10) ∧
  (∃ m n : ℕ+, m.val * n.val = 10 ∧ ∃ p : ℕ, m.val^m.val * n.val^n.val = 2000 * p) :=
sorry

theorem least_product_for_cross_divisibility (a b : ℕ+) :
  (∃ k : ℕ, a.val^b.val * b.val^a.val = 2000 * k) →
  (∀ c d : ℕ+, (∃ l : ℕ, c.val^d.val * d.val^c.val = 2000 * l) → c.val * d.val ≥ 20) ∧
  (∃ m n : ℕ+, m.val * n.val = 20 ∧ ∃ p : ℕ, m.val^n.val * n.val^m.val = 2000 * p) :=
sorry

end NUMINAMATH_CALUDE_least_product_for_divisibility_least_product_for_cross_divisibility_l1760_176080


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1760_176006

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1760_176006


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l1760_176011

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

-- Define monotonicity on an interval
def MonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

-- State the theorem
theorem monotonic_quadratic_function (a : ℝ) :
  MonotonicOn (f a) 1 2 → a ≤ -2 ∨ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l1760_176011


namespace NUMINAMATH_CALUDE_distance_AB_is_300_l1760_176038

/-- The distance between points A and B in meters -/
def distance_AB : ℝ := 300

/-- The speed ratio of Person A to Person B -/
def speed_ratio : ℝ := 2

/-- The distance Person B is from point B when Person A reaches B -/
def distance_B_to_B : ℝ := 100

/-- The distance from B where Person A and B meet when A returns -/
def meeting_distance : ℝ := 60

/-- Theorem stating the distance between A and B is 300 meters -/
theorem distance_AB_is_300 :
  distance_AB = 300 ∧
  speed_ratio = 2 ∧
  distance_B_to_B = 100 ∧
  meeting_distance = 60 →
  distance_AB = 300 := by
  sorry

#check distance_AB_is_300

end NUMINAMATH_CALUDE_distance_AB_is_300_l1760_176038


namespace NUMINAMATH_CALUDE_final_alcohol_percentage_l1760_176072

def initial_volume : ℝ := 40
def initial_alcohol_percentage : ℝ := 5
def added_alcohol : ℝ := 5.5
def added_water : ℝ := 4.5

theorem final_alcohol_percentage :
  let initial_alcohol := initial_volume * (initial_alcohol_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  (final_alcohol / final_volume) * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_final_alcohol_percentage_l1760_176072


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l1760_176092

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Pay per task in dollars -/
def pay_per_task : ℚ := 1.2

/-- Number of working days per week -/
def working_days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * pay_per_task * working_days_per_week

theorem tim_weekly_earnings : weekly_earnings = 720 := by sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l1760_176092


namespace NUMINAMATH_CALUDE_fraction_equality_l1760_176051

theorem fraction_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1760_176051


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l1760_176017

theorem square_rectangle_area_relation :
  ∃ (x₁ x₂ : ℝ),
    (x₁ - 5) * (x₁ + 6) = 3 * (x₁ - 4)^2 ∧
    (x₂ - 5) * (x₂ + 6) = 3 * (x₂ - 4)^2 ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l1760_176017


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1760_176049

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) (h_pos_L : L > 0) (h_pos_W : W > 0) :
  (1.16 * L) * (W * (1 - x / 100)) = 1.102 * (L * W) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1760_176049


namespace NUMINAMATH_CALUDE_sugar_sacks_weight_difference_l1760_176005

theorem sugar_sacks_weight_difference (x y : ℝ) : 
  x + y = 40 →
  x - 1 = 0.6 * (y + 1) →
  |x - y| = 8 := by
sorry

end NUMINAMATH_CALUDE_sugar_sacks_weight_difference_l1760_176005


namespace NUMINAMATH_CALUDE_cube_difference_l1760_176084

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : 
  a^3 - b^3 = 108 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l1760_176084


namespace NUMINAMATH_CALUDE_angle_value_proof_l1760_176094

theorem angle_value_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.cos (α + β) = Real.sin (α - β)) : α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_proof_l1760_176094


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l1760_176029

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = x / Real.sqrt (x + 2)) ↔ x > -2 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l1760_176029


namespace NUMINAMATH_CALUDE_compare_negative_decimals_l1760_176077

theorem compare_negative_decimals : -3.3 < -3.14 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_decimals_l1760_176077


namespace NUMINAMATH_CALUDE_angle_in_first_quadrant_l1760_176022

-- Define the angle in degrees and minutes
def angle : ℚ := -999 * 360 / 360 - 30 / 60

-- Function to normalize an angle to the range [0, 360)
def normalize_angle (θ : ℚ) : ℚ :=
  θ - 360 * ⌊θ / 360⌋

-- Define the first quadrant
def is_first_quadrant (θ : ℚ) : Prop :=
  0 < normalize_angle θ ∧ normalize_angle θ < 90

-- Theorem statement
theorem angle_in_first_quadrant : is_first_quadrant angle := by
  sorry

end NUMINAMATH_CALUDE_angle_in_first_quadrant_l1760_176022


namespace NUMINAMATH_CALUDE_custom_deck_combination_l1760_176028

-- Define the number of suits
def num_suits : ℕ := 4

-- Define the number of cards per suit
def cards_per_suit : ℕ := 12

-- Define the number of face cards per suit
def face_cards_per_suit : ℕ := 3

-- Define the total number of cards in the deck
def total_cards : ℕ := num_suits * cards_per_suit

-- Theorem statement
theorem custom_deck_combination : 
  (Nat.choose num_suits 3) * 3 * face_cards_per_suit * cards_per_suit * cards_per_suit = 5184 := by
  sorry

end NUMINAMATH_CALUDE_custom_deck_combination_l1760_176028


namespace NUMINAMATH_CALUDE_initial_mangoes_l1760_176065

/-- Given a bag of fruits with the following conditions:
    - Initially contains 7 apples, 8 oranges, and M mangoes
    - 2 apples are removed
    - 4 oranges are removed (twice the number of apples removed)
    - 2/3 of the mangoes are removed
    - 14 fruits remain in the bag
    Prove that the initial number of mangoes (M) is 15 -/
theorem initial_mangoes (M : ℕ) : 
  (7 - 2) + (8 - 4) + (M - (2 * M / 3)) = 14 → M = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_mangoes_l1760_176065


namespace NUMINAMATH_CALUDE_fried_chicken_dinner_pieces_l1760_176089

/-- Represents the number of pieces of chicken in a family-size Fried Chicken Dinner -/
def fried_chicken_pieces : ℕ := sorry

/-- The number of pieces of chicken in a Chicken Pasta order -/
def chicken_pasta_pieces : ℕ := 2

/-- The number of pieces of chicken in a Barbecue Chicken order -/
def barbecue_chicken_pieces : ℕ := 3

/-- The number of Fried Chicken Dinner orders -/
def fried_chicken_orders : ℕ := 2

/-- The number of Chicken Pasta orders -/
def chicken_pasta_orders : ℕ := 6

/-- The number of Barbecue Chicken orders -/
def barbecue_chicken_orders : ℕ := 3

/-- The total number of pieces of chicken needed for all orders -/
def total_chicken_pieces : ℕ := 37

theorem fried_chicken_dinner_pieces :
  fried_chicken_pieces * fried_chicken_orders +
  chicken_pasta_pieces * chicken_pasta_orders +
  barbecue_chicken_pieces * barbecue_chicken_orders =
  total_chicken_pieces ∧
  fried_chicken_pieces = 8 := by sorry

end NUMINAMATH_CALUDE_fried_chicken_dinner_pieces_l1760_176089


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l1760_176069

/-- Represents the distance between two cities on a map and in reality. -/
structure MapDistance where
  /-- Distance on the map in centimeters -/
  map_distance : ℝ
  /-- Map scale: 1 cm on map represents this many km in reality -/
  scale : ℝ

/-- Calculates the real-world distance in meters given a MapDistance -/
def real_distance (d : MapDistance) : ℝ :=
  d.map_distance * d.scale * 1000

/-- The distance between Stockholm and Uppsala -/
def stockholm_uppsala : MapDistance :=
  { map_distance := 55
  , scale := 30 }

/-- Theorem stating that the distance between Stockholm and Uppsala is 1650000 meters -/
theorem stockholm_uppsala_distance :
  real_distance stockholm_uppsala = 1650000 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l1760_176069


namespace NUMINAMATH_CALUDE_john_toy_store_spending_l1760_176075

def weekly_allowance : ℚ := 240/100

theorem john_toy_store_spending (arcade_fraction : ℚ) (candy_store_amount : ℚ) 
  (h1 : arcade_fraction = 3/5)
  (h2 : candy_store_amount = 64/100) :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction)
  let toy_store_amount := remaining_after_arcade - candy_store_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by sorry

end NUMINAMATH_CALUDE_john_toy_store_spending_l1760_176075


namespace NUMINAMATH_CALUDE_product_remainder_remainder_proof_l1760_176026

theorem product_remainder (a b m : ℕ) (h : m > 0) : (a * b) % m = ((a % m) * (b % m)) % m := by sorry

theorem remainder_proof : (1023 * 999999) % 139 = 32 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_product_remainder_remainder_proof_l1760_176026


namespace NUMINAMATH_CALUDE_industrial_lubricants_budget_percentage_l1760_176003

theorem industrial_lubricants_budget_percentage
  (total_degrees : ℝ)
  (microphotonics_percent : ℝ)
  (home_electronics_percent : ℝ)
  (food_additives_percent : ℝ)
  (genetically_modified_microorganisms_percent : ℝ)
  (basic_astrophysics_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : microphotonics_percent = 14)
  (h3 : home_electronics_percent = 19)
  (h4 : food_additives_percent = 10)
  (h5 : genetically_modified_microorganisms_percent = 24)
  (h6 : basic_astrophysics_degrees = 90) :
  let total_percent : ℝ := 100
  let basic_astrophysics_percent : ℝ := (basic_astrophysics_degrees / total_degrees) * total_percent
  let known_sectors_percent : ℝ := microphotonics_percent + home_electronics_percent + 
                                   food_additives_percent + genetically_modified_microorganisms_percent
  let industrial_lubricants_percent : ℝ := total_percent - known_sectors_percent - basic_astrophysics_percent
  industrial_lubricants_percent = 8 := by
  sorry

end NUMINAMATH_CALUDE_industrial_lubricants_budget_percentage_l1760_176003


namespace NUMINAMATH_CALUDE_smallest_in_S_l1760_176057

def S : Set ℝ := {1, -2, -1.7, 0, Real.pi}

theorem smallest_in_S : ∀ x ∈ S, -2 ≤ x := by sorry

end NUMINAMATH_CALUDE_smallest_in_S_l1760_176057


namespace NUMINAMATH_CALUDE_composite_expression_l1760_176059

theorem composite_expression (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_expression_l1760_176059


namespace NUMINAMATH_CALUDE_swimming_improvement_l1760_176087

/-- Represents John's swimming performance -/
structure SwimmingPerformance where
  laps : ℕ
  time : ℕ

/-- Calculates the lap time in minutes per lap -/
def lapTime (performance : SwimmingPerformance) : ℚ :=
  performance.time / performance.laps

theorem swimming_improvement 
  (initial : SwimmingPerformance) 
  (final : SwimmingPerformance) 
  (h1 : initial.laps = 15) 
  (h2 : initial.time = 35) 
  (h3 : final.laps = 18) 
  (h4 : final.time = 33) : 
  lapTime initial - lapTime final = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_swimming_improvement_l1760_176087


namespace NUMINAMATH_CALUDE_fixed_stable_points_equality_l1760_176056

def f (a x : ℝ) : ℝ := a * x^2 - 1

def isFixedPoint (a x : ℝ) : Prop := f a x = x

def isStablePoint (a x : ℝ) : Prop := f a (f a x) = x

def fixedPointSet (a : ℝ) : Set ℝ := {x | isFixedPoint a x}

def stablePointSet (a : ℝ) : Set ℝ := {x | isStablePoint a x}

theorem fixed_stable_points_equality (a : ℝ) :
  (fixedPointSet a = stablePointSet a ∧ (fixedPointSet a).Nonempty) ↔ -1/4 ≤ a ∧ a ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_stable_points_equality_l1760_176056


namespace NUMINAMATH_CALUDE_office_employees_l1760_176027

/-- Proves that the total number of employees in an office is 2200 given specific conditions -/
theorem office_employees (total : ℕ) (male_ratio : ℚ) (old_male_ratio : ℚ) (young_males : ℕ) 
  (h1 : male_ratio = 2/5)
  (h2 : old_male_ratio = 3/10)
  (h3 : young_males = 616)
  (h4 : ↑young_males = (1 - old_male_ratio) * (male_ratio * ↑total)) : 
  total = 2200 := by
sorry

end NUMINAMATH_CALUDE_office_employees_l1760_176027


namespace NUMINAMATH_CALUDE_factorization_equality_l1760_176083

theorem factorization_equality (x : ℝ) : 45 * x^2 + 135 * x = 45 * x * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1760_176083


namespace NUMINAMATH_CALUDE_line_equation_from_triangle_l1760_176021

/-- Given a line passing through (-a, 0), (b, 0), and (0, h), where the area of the triangle
    formed in the second quadrant is T, prove that the equation of this line is
    2Tx - (b+a)^2y + 2T(b+a) = 0 -/
theorem line_equation_from_triangle (a b h T : ℝ) :
  (∃ (line : ℝ → ℝ → Prop),
    line (-a) 0 ∧
    line b 0 ∧
    line 0 h ∧
    (1/2 : ℝ) * (b + a) * h = T) →
  (∃ (line : ℝ → ℝ → Prop),
    ∀ x y, line x y ↔ 2 * T * x - (b + a)^2 * y + 2 * T * (b + a) = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_triangle_l1760_176021


namespace NUMINAMATH_CALUDE_total_collection_l1760_176050

/-- A group of students collecting money -/
structure StudentGroup where
  members : ℕ
  contribution_per_member : ℕ
  total_paise : ℕ

/-- Conversion rate from paise to rupees -/
def paise_to_rupees (paise : ℕ) : ℚ :=
  (paise : ℚ) / 100

/-- Theorem stating the total amount collected by the group -/
theorem total_collection (group : StudentGroup) 
    (h1 : group.members = 54)
    (h2 : group.contribution_per_member = group.members)
    (h3 : group.total_paise = group.members * group.contribution_per_member) : 
    paise_to_rupees group.total_paise = 29.16 := by
  sorry

end NUMINAMATH_CALUDE_total_collection_l1760_176050


namespace NUMINAMATH_CALUDE_stratified_sampling_l1760_176099

theorem stratified_sampling (total_students : ℕ) (class1_students : ℕ) (class2_students : ℕ) 
  (sample_size : ℕ) (h1 : total_students = class1_students + class2_students) 
  (h2 : total_students = 96) (h3 : class1_students = 54) (h4 : class2_students = 42) 
  (h5 : sample_size = 16) :
  (class1_students * sample_size / total_students : ℚ) = 9 ∧
  (class2_students * sample_size / total_students : ℚ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l1760_176099


namespace NUMINAMATH_CALUDE_total_song_requests_l1760_176068

/-- Represents the total number of song requests --/
def T : ℕ := 30

/-- Theorem stating that the total number of song requests is 30 --/
theorem total_song_requests :
  T = 30 ∧
  T = (1/2 : ℚ) * T + (1/6 : ℚ) * T + 5 + 2 + 1 + 2 :=
by sorry

end NUMINAMATH_CALUDE_total_song_requests_l1760_176068


namespace NUMINAMATH_CALUDE_generating_function_value_l1760_176024

/-- The generating function of two linear functions -/
def generating_function (m n x : ℝ) : ℝ := m * (x + 1) + n * (2 * x)

/-- Theorem: The generating function equals 2 when x = 1 and m + n = 1 -/
theorem generating_function_value : 
  ∀ m n : ℝ, m + n = 1 → generating_function m n 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_generating_function_value_l1760_176024


namespace NUMINAMATH_CALUDE_term_2500_mod_7_l1760_176040

/-- Defines the sequence where the (2n)th positive integer appears n times
    and the (2n-1)th positive integer appears n+1 times -/
def sequence_term (k : ℕ) : ℕ := sorry

/-- The 2500th term of the sequence -/
def term_2500 : ℕ := sequence_term 2500

theorem term_2500_mod_7 : term_2500 % 7 = 1 := by sorry

end NUMINAMATH_CALUDE_term_2500_mod_7_l1760_176040


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1760_176012

-- Define the polynomial
def P (x r s t : ℝ) : ℝ := x^4 + r*x^2 + s*x + t

-- State the theorem
theorem root_sum_theorem (b r s t : ℝ) :
  (P (b - 6) r s t = 0) ∧ 
  (P (b - 5) r s t = 0) ∧ 
  (P (b - 4) r s t = 0) →
  r + t = -61 := by
  sorry


end NUMINAMATH_CALUDE_root_sum_theorem_l1760_176012


namespace NUMINAMATH_CALUDE_rectangle_division_possible_l1760_176030

theorem rectangle_division_possible : ∃ (w1 h1 w2 h2 w3 h3 : ℕ+), 
  (w1 * h1 : ℕ) + (w2 * h2 : ℕ) + (w3 * h3 : ℕ) = 100 * 70 ∧
  (w1 : ℕ) ≤ 100 ∧ (h1 : ℕ) ≤ 70 ∧
  (w2 : ℕ) ≤ 100 ∧ (h2 : ℕ) ≤ 70 ∧
  (w3 : ℕ) ≤ 100 ∧ (h3 : ℕ) ≤ 70 ∧
  2 * (w1 * h1 : ℕ) = (w2 * h2 : ℕ) ∧
  2 * (w2 * h2 : ℕ) = (w3 * h3 : ℕ) := by
  sorry

#check rectangle_division_possible

end NUMINAMATH_CALUDE_rectangle_division_possible_l1760_176030


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1760_176076

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}
def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

-- State the theorem
theorem circle_intersection_range (r : ℝ) :
  r > 0 ∧ M ∩ N r = N r ↔ r ∈ Set.Ioo 0 (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1760_176076


namespace NUMINAMATH_CALUDE_fruit_supply_theorem_l1760_176020

/-- Represents the weekly fruit requirements for a bakery -/
structure BakeryRequirement where
  strawberries : ℕ
  blueberries : ℕ
  raspberries : ℕ

/-- Calculates the total number of sacks needed for a given fruit over 10 weeks -/
def totalSacksFor10Weeks (weeklyRequirements : List BakeryRequirement) (getFruit : BakeryRequirement → ℕ) : ℕ :=
  10 * (weeklyRequirements.map getFruit).sum

/-- The list of weekly requirements for all bakeries -/
def allBakeries : List BakeryRequirement := [
  ⟨2, 3, 5⟩,
  ⟨4, 2, 8⟩,
  ⟨12, 10, 7⟩,
  ⟨8, 4, 3⟩,
  ⟨15, 6, 12⟩,
  ⟨5, 9, 11⟩
]

theorem fruit_supply_theorem :
  totalSacksFor10Weeks allBakeries (·.strawberries) = 460 ∧
  totalSacksFor10Weeks allBakeries (·.blueberries) = 340 ∧
  totalSacksFor10Weeks allBakeries (·.raspberries) = 460 := by
  sorry

end NUMINAMATH_CALUDE_fruit_supply_theorem_l1760_176020


namespace NUMINAMATH_CALUDE_equation_solution_l1760_176015

theorem equation_solution :
  ∃ x : ℚ, (x - 55) / 3 = (2 - 3*x + x^2) / 4 ∧ (x = 20/3 ∨ x = -11) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1760_176015


namespace NUMINAMATH_CALUDE_no_factors_l1760_176082

def p (x : ℝ) : ℝ := x^4 - 4*x^2 + 16

def factor1 (x : ℝ) : ℝ := x^2 - 4
def factor2 (x : ℝ) : ℝ := x + 2
def factor3 (x : ℝ) : ℝ := x^2 + 4*x + 4
def factor4 (x : ℝ) : ℝ := x^2 + 1

theorem no_factors :
  (∃ (x : ℝ), p x ≠ 0 ∧ factor1 x = 0) ∧
  (∃ (x : ℝ), p x ≠ 0 ∧ factor2 x = 0) ∧
  (∃ (x : ℝ), p x ≠ 0 ∧ factor3 x = 0) ∧
  (∃ (x : ℝ), p x ≠ 0 ∧ factor4 x = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_factors_l1760_176082


namespace NUMINAMATH_CALUDE_quadratic_function_increasing_condition_l1760_176016

/-- Given a quadratic function y = x^2 - 2mx + 5, if y increases as x increases when x > -1, then m ≤ -1 -/
theorem quadratic_function_increasing_condition (m : ℝ) : 
  (∀ x > -1, ∀ y > x, (y^2 - 2*m*y + 5) > (x^2 - 2*m*x + 5)) → m ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_increasing_condition_l1760_176016


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l1760_176039

theorem polygon_interior_angles (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 1260 → (n - 2) * 180 = sum_angles → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l1760_176039


namespace NUMINAMATH_CALUDE_singer_total_hours_l1760_176034

/-- Given a singer who works on songs with the following parameters:
    - Hours worked per day on one song
    - Days required to complete one song
    - Number of songs to complete
    Calculate the total number of hours taken to complete all songs -/
def total_hours (hours_per_day : ℕ) (days_per_song : ℕ) (num_songs : ℕ) : ℕ :=
  hours_per_day * days_per_song * num_songs

/-- Theorem stating that for a singer working 10 hours a day, 
    with each song taking 10 days to complete, 
    the total time to complete 3 songs is 300 hours -/
theorem singer_total_hours : total_hours 10 10 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_singer_total_hours_l1760_176034


namespace NUMINAMATH_CALUDE_function_power_id_implies_bijective_l1760_176008

variable {X : Type*}

def compose_n_times {X : Type*} (f : X → X) : ℕ → (X → X)
  | 0 => id
  | n + 1 => f ∘ (compose_n_times f n)

theorem function_power_id_implies_bijective
  (f : X → X) (k : ℕ) (hk : k > 0) (h : compose_n_times f k = id) :
  Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_function_power_id_implies_bijective_l1760_176008


namespace NUMINAMATH_CALUDE_external_tangent_y_intercept_l1760_176064

/-- Given two circles with centers and radii as specified, 
    prove that their common external tangent with positive slope 
    has a y-intercept of 740/171 -/
theorem external_tangent_y_intercept :
  let c1 : ℝ × ℝ := (1, 3)  -- Center of circle 1
  let r1 : ℝ := 3           -- Radius of circle 1
  let c2 : ℝ × ℝ := (15, 8) -- Center of circle 2
  let r2 : ℝ := 10          -- Radius of circle 2
  let m : ℝ := (140 : ℝ) / 171 -- Slope of the tangent line (positive)
  let b : ℝ := (740 : ℝ) / 171 -- y-intercept to be proved
  let tangent_line (x : ℝ) := m * x + b -- Equation of the tangent line
  (∀ x y : ℝ, (x - c1.1)^2 + (y - c1.2)^2 = r1^2 → 
    (tangent_line x - y)^2 ≥ (r1 * m)^2) ∧ 
  (∀ x y : ℝ, (x - c2.1)^2 + (y - c2.2)^2 = r2^2 → 
    (tangent_line x - y)^2 ≥ (r2 * m)^2) ∧
  (∃ x1 y1 x2 y2 : ℝ, 
    (x1 - c1.1)^2 + (y1 - c1.2)^2 = r1^2 ∧
    (x2 - c2.1)^2 + (y2 - c2.2)^2 = r2^2 ∧
    tangent_line x1 = y1 ∧ tangent_line x2 = y2) :=
by sorry

end NUMINAMATH_CALUDE_external_tangent_y_intercept_l1760_176064


namespace NUMINAMATH_CALUDE_fence_perimeter_l1760_176090

/-- The outer perimeter of a square fence given the number of posts, post width, and gap between posts. -/
def outerPerimeter (numPosts : ℕ) (postWidth : ℚ) (gapWidth : ℕ) : ℚ :=
  let postsPerSide : ℕ := numPosts / 4
  let numGaps : ℕ := postsPerSide - 1
  let gapLength : ℚ := numGaps * gapWidth
  let postLength : ℚ := postsPerSide * postWidth
  let sideLength : ℚ := gapLength + postLength
  4 * sideLength

/-- Theorem stating that the outer perimeter of the fence is 274 feet. -/
theorem fence_perimeter :
  outerPerimeter 36 (1/2) 8 = 274 := by sorry

end NUMINAMATH_CALUDE_fence_perimeter_l1760_176090


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l1760_176014

def parabola (y : ℝ) : ℝ := 2 * y^2 - 6 * y + 3

theorem parabola_intercepts_sum :
  ∃ (a b c : ℝ),
    (parabola 0 = a) ∧
    (parabola b = 0) ∧
    (parabola c = 0) ∧
    (b ≠ c) ∧
    (a + b + c = 6) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l1760_176014


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1760_176009

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 4/9) (h2 : x - y = 2/9) : x^2 - y^2 = 8/81 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1760_176009


namespace NUMINAMATH_CALUDE_a_minus_b_bounds_l1760_176010

theorem a_minus_b_bounds (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := by sorry

end NUMINAMATH_CALUDE_a_minus_b_bounds_l1760_176010


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_l1760_176032

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_b_terms :
  ∃ (m : ℕ), m = 3 ∧ 
  (∀ (n : ℕ), n ≥ 1 → Nat.gcd (b n) (b (n + 1)) ≤ m) ∧
  (∃ (k : ℕ), k ≥ 1 ∧ Nat.gcd (b k) (b (k + 1)) = m) :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_l1760_176032


namespace NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l1760_176042

theorem sum_and_ratio_implies_difference (x y : ℚ) 
  (h1 : x + y = 540) 
  (h2 : x / y = 7 / 8) : 
  y - x = 36 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l1760_176042


namespace NUMINAMATH_CALUDE_absolute_value_of_one_plus_i_squared_l1760_176078

theorem absolute_value_of_one_plus_i_squared : Complex.abs ((1 : ℂ) + Complex.I) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_one_plus_i_squared_l1760_176078


namespace NUMINAMATH_CALUDE_place_three_after_correct_l1760_176095

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ :=
  10 * n.tens + n.units

/-- The result of placing 3 after a two-digit number -/
def place_three_after (n : TwoDigitNumber) : ℕ :=
  100 * n.tens + 10 * n.units + 3

theorem place_three_after_correct (n : TwoDigitNumber) :
  place_three_after n = 100 * n.tens + 10 * n.units + 3 := by
  sorry

end NUMINAMATH_CALUDE_place_three_after_correct_l1760_176095


namespace NUMINAMATH_CALUDE_inverse_inequality_iff_inequality_l1760_176097

theorem inverse_inequality_iff_inequality (a b : ℝ) (h : a * b > 0) :
  (1 / a < 1 / b) ↔ (a > b) := by sorry

end NUMINAMATH_CALUDE_inverse_inequality_iff_inequality_l1760_176097


namespace NUMINAMATH_CALUDE_total_precious_stones_l1760_176045

theorem total_precious_stones (agate olivine sapphire diamond amethyst ruby : ℕ) : 
  agate = 25 →
  olivine = agate + 5 →
  sapphire = 2 * olivine →
  diamond = olivine + 11 →
  amethyst = sapphire + diamond →
  ruby = diamond + 7 →
  agate + olivine + sapphire + diamond + amethyst + ruby = 305 := by
  sorry

end NUMINAMATH_CALUDE_total_precious_stones_l1760_176045


namespace NUMINAMATH_CALUDE_fraction_decrease_l1760_176025

theorem fraction_decrease (m n : ℝ) (h : m ≠ 0 ∧ n ≠ 0) : 
  (3*m + 3*n) / ((3*m) * (3*n)) = (1/3) * ((m + n) / (m * n)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decrease_l1760_176025


namespace NUMINAMATH_CALUDE_odd_function_property_l1760_176070

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g as a function from ℝ to ℝ
def g (x : ℝ) : ℝ := f x + 9

-- Theorem statement
theorem odd_function_property (hf_odd : ∀ x, f (-x) = -f x) (hg_value : g (-2) = 3) : f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1760_176070


namespace NUMINAMATH_CALUDE_town_population_l1760_176058

theorem town_population (total_population : ℕ) 
  (h1 : total_population < 6000)
  (h2 : ∃ (boys girls : ℕ), girls = (11 * boys) / 10 ∧ boys + girls = total_population * 10 / 21)
  (h3 : ∃ (women men : ℕ), men = (23 * women) / 20 ∧ women + men = total_population * 20 / 43)
  (h4 : ∃ (children adults : ℕ), children = (6 * adults) / 5 ∧ children + adults = total_population) :
  total_population = 3311 := by
sorry

end NUMINAMATH_CALUDE_town_population_l1760_176058


namespace NUMINAMATH_CALUDE_g_geq_f_implies_t_leq_one_l1760_176088

noncomputable section

open Real

-- Define the functions f and g
def f (x : ℝ) : ℝ := exp x - x * log x
def g (t : ℝ) (x : ℝ) : ℝ := exp x - t * x^2 + x

-- State the theorem
theorem g_geq_f_implies_t_leq_one (t : ℝ) :
  (∀ x > 0, g t x ≥ f x) → t ≤ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_g_geq_f_implies_t_leq_one_l1760_176088


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l1760_176066

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k : ℕ, a k > 0) →  -- Positive sequence
  (∃ q : ℝ, q > 0 ∧ ∀ k : ℕ, a (k + 1) = q * a k) →  -- Geometric sequence
  a 2018 = a 2017 + 2 * a 2016 →  -- Given condition
  (a m * a n = 16 * (a 1)^2) →  -- Derived from √(a_m * a_n) = 4a_1
  (∀ i j : ℕ, i > 0 ∧ j > 0 ∧ a i * a j = 16 * (a 1)^2 → 1/i + 5/j ≥ 7/4) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l1760_176066


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l1760_176085

/-- Calculates the actual profit percentage for a retailer given a markup and discount rate -/
def actualProfitPercentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let markedPrice := 1 + markup
  let sellingPrice := markedPrice * (1 - discount)
  let profit := sellingPrice - 1
  profit * 100

/-- Theorem stating that the actual profit percentage is 5% for a 40% markup and 25% discount -/
theorem retailer_profit_percentage :
  actualProfitPercentage 0.4 0.25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l1760_176085


namespace NUMINAMATH_CALUDE_least_multiple_with_digit_sum_l1760_176061

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem least_multiple_with_digit_sum (N : ℕ) : N = 779 ↔ 
  (∀ m : ℕ, m < N → (m % 19 = 0 → sum_of_digits m ≠ 23)) ∧ 
  (N % 19 = 0) ∧ 
  (sum_of_digits N = 23) :=
sorry

end NUMINAMATH_CALUDE_least_multiple_with_digit_sum_l1760_176061


namespace NUMINAMATH_CALUDE_inequality_proof_l1760_176007

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x * y + y * z + z * x ≤ 1) : 
  (x + 1/x) * (y + 1/y) * (z + 1/z) ≥ 8 * (x + y) * (y + z) * (z + x) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1760_176007


namespace NUMINAMATH_CALUDE_equation_solution_l1760_176086

theorem equation_solution (x : ℝ) (h1 : x^2 + x ≠ 0) (h2 : x + 1 ≠ 0) :
  (2 - 1 / (x^2 + x) = (2*x + 1) / (x + 1)) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1760_176086


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l1760_176098

theorem circle_tangent_to_x_axis (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*x + 2*b*y + b^2 = 0 → 
   ∃ p : ℝ × ℝ, p.1^2 + p.2^2 = 0 ∧ p.2 = 0) → 
  b = 2 ∨ b = -2 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l1760_176098


namespace NUMINAMATH_CALUDE_shaded_squares_in_six_by_six_grid_l1760_176013

/-- Represents a grid with a given size and number of unshaded squares per row -/
structure Grid where
  size : Nat
  unshadedPerRow : Nat

/-- Calculates the number of shaded squares in the grid -/
def shadedSquares (g : Grid) : Nat :=
  g.size * (g.size - g.unshadedPerRow)

theorem shaded_squares_in_six_by_six_grid :
  ∀ (g : Grid), g.size = 6 → g.unshadedPerRow = 1 → shadedSquares g = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_squares_in_six_by_six_grid_l1760_176013


namespace NUMINAMATH_CALUDE_angle_alpha_trig_l1760_176052

theorem angle_alpha_trig (α : Real) (m : Real) :
  m ≠ 0 →
  (∃ (x y : Real), x = -Real.sqrt 3 ∧ y = m ∧ x^2 + y^2 = (Real.cos α)^2 + (Real.sin α)^2) →
  Real.sin α = (Real.sqrt 2 / 4) * m →
  (m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  Real.cos α = -Real.sqrt 6 / 4 ∧
  ((m > 0 → Real.tan α = -Real.sqrt 15 / 3) ∧
   (m < 0 → Real.tan α = Real.sqrt 15 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_angle_alpha_trig_l1760_176052


namespace NUMINAMATH_CALUDE_square_of_nines_l1760_176048

theorem square_of_nines (n : ℕ) (h : n = 999999) : n^2 = (n + 1) * (n - 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_of_nines_l1760_176048


namespace NUMINAMATH_CALUDE_relay_race_distance_l1760_176047

/-- Proves that in a 5-member relay team where one member runs twice the distance of others,
    and the total race distance is 18 km, each of the other members runs 3 km. -/
theorem relay_race_distance (team_size : ℕ) (ralph_multiplier : ℕ) (total_distance : ℝ) :
  team_size = 5 →
  ralph_multiplier = 2 →
  total_distance = 18 →
  ∃ (other_distance : ℝ),
    other_distance = 3 ∧
    (team_size - 1) * other_distance + ralph_multiplier * other_distance = total_distance :=
by sorry

end NUMINAMATH_CALUDE_relay_race_distance_l1760_176047


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1760_176060

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  a 1 + a 2 = 4 →
  d = 2 →
  a 7 + a 8 = 28 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1760_176060
