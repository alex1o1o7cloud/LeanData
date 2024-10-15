import Mathlib

namespace NUMINAMATH_CALUDE_task_completion_probability_l101_10138

theorem task_completion_probability
  (p_task2 : ℝ)
  (p_task1_not_task2 : ℝ)
  (h1 : p_task2 = 3 / 5)
  (h2 : p_task1_not_task2 = 0.15)
  (h_independent : True)  -- Representing the independence of tasks
  : ∃ (p_task1 : ℝ), p_task1 = 0.375 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_probability_l101_10138


namespace NUMINAMATH_CALUDE_inequality_proof_l101_10177

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l101_10177


namespace NUMINAMATH_CALUDE_class_size_l101_10143

theorem class_size (football : ℕ) (long_tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : football = 26)
  (h2 : long_tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 9) :
  football + long_tennis - both + neither = 38 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l101_10143


namespace NUMINAMATH_CALUDE_overlap_area_is_three_quarters_l101_10157

/-- Represents a point on a 3x3 grid --/
structure GridPoint where
  x : Fin 3
  y : Fin 3

/-- Represents a triangle on the grid --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The first triangle connecting top left, middle right, and bottom center --/
def triangle1 : GridTriangle :=
  { p1 := ⟨0, 0⟩, p2 := ⟨2, 1⟩, p3 := ⟨1, 2⟩ }

/-- The second triangle connecting top right, middle left, and bottom center --/
def triangle2 : GridTriangle :=
  { p1 := ⟨2, 0⟩, p2 := ⟨0, 1⟩, p3 := ⟨1, 2⟩ }

/-- Calculates the area of overlap between two triangles on the grid --/
def areaOfOverlap (t1 t2 : GridTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of overlap between the two specific triangles is 0.75 --/
theorem overlap_area_is_three_quarters :
  areaOfOverlap triangle1 triangle2 = 0.75 := by sorry

end NUMINAMATH_CALUDE_overlap_area_is_three_quarters_l101_10157


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_at_m_one_l101_10130

theorem min_sum_squares (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + x₂^2 = (x₁ + x₂)^2 - 2*x₁*x₂ →
  (∃ D : ℝ, D ≥ 0 ∧ D = (m + 3)^2) →
  x₁ + x₂ = -(m + 1) →
  x₁ * x₂ = 2*m - 2 →
  x₁^2 + x₂^2 ≥ 4 :=
by sorry

theorem min_sum_squares_at_m_one (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + x₂^2 = (x₁ + x₂)^2 - 2*x₁*x₂ →
  (∃ D : ℝ, D ≥ 0 ∧ D = (m + 3)^2) →
  x₁ + x₂ = -(m + 1) →
  x₁ * x₂ = 2*m - 2 →
  m = 1 →
  x₁^2 + x₂^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_at_m_one_l101_10130


namespace NUMINAMATH_CALUDE_red_marble_fraction_l101_10137

theorem red_marble_fraction (total : ℝ) (h_total_pos : total > 0) : 
  let blue := (2/3) * total
  let red := (1/3) * total
  let new_blue := 3 * blue
  let new_total := new_blue + red
  red / new_total = 1/7 := by
sorry

end NUMINAMATH_CALUDE_red_marble_fraction_l101_10137


namespace NUMINAMATH_CALUDE_no_real_roots_l101_10105

theorem no_real_roots : ∀ x : ℝ, x^2 - x + 9 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l101_10105


namespace NUMINAMATH_CALUDE_calculation_1_calculation_2_calculation_3_calculation_4_calculation_5_calculation_6_l101_10195

theorem calculation_1 : 238 + 45 * 5 = 463 := by sorry

theorem calculation_2 : 65 * 4 - 128 = 132 := by sorry

theorem calculation_3 : 900 - 108 * 4 = 468 := by sorry

theorem calculation_4 : 369 + (512 - 215) = 666 := by sorry

theorem calculation_5 : 758 - 58 * 9 = 236 := by sorry

theorem calculation_6 : 105 * (81 / 9 - 3) = 630 := by sorry

end NUMINAMATH_CALUDE_calculation_1_calculation_2_calculation_3_calculation_4_calculation_5_calculation_6_l101_10195


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l101_10118

theorem greatest_integer_fraction_inequality :
  ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l101_10118


namespace NUMINAMATH_CALUDE_train_speed_problem_l101_10150

theorem train_speed_problem (total_distance : ℝ) (speed_increase : ℝ) (distance_difference : ℝ) (time_difference : ℝ) :
  total_distance = 103 ∧ 
  speed_increase = 4 ∧ 
  distance_difference = 23 ∧ 
  time_difference = 1/4 →
  ∃ (initial_speed : ℝ) (initial_time : ℝ),
    initial_speed = 80 ∧
    initial_speed * initial_time + (initial_speed * initial_time + distance_difference) = total_distance ∧
    (initial_speed + speed_increase) * (initial_time + time_difference) = initial_speed * initial_time + distance_difference :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l101_10150


namespace NUMINAMATH_CALUDE_smallest_norm_w_l101_10180

theorem smallest_norm_w (w : ℝ × ℝ) (h : ‖w + (4, 2)‖ = 10) :
  ∃ (w_min : ℝ × ℝ), (∀ w' : ℝ × ℝ, ‖w' + (4, 2)‖ = 10 → ‖w_min‖ ≤ ‖w'‖) ∧ ‖w_min‖ = 10 - 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_norm_w_l101_10180


namespace NUMINAMATH_CALUDE_peters_pizza_fraction_l101_10113

theorem peters_pizza_fraction (total_slices : ℕ) (peters_solo_slices : ℕ) (shared_slices : ℕ) :
  total_slices = 16 →
  peters_solo_slices = 3 →
  shared_slices = 2 →
  (peters_solo_slices : ℚ) / total_slices + (shared_slices : ℚ) / (2 * total_slices) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_peters_pizza_fraction_l101_10113


namespace NUMINAMATH_CALUDE_badminton_players_l101_10142

theorem badminton_players (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 28)
  (h2 : tennis = 19)
  (h3 : neither = 2)
  (h4 : both = 10) :
  ∃ badminton : ℕ, badminton = 17 ∧ 
    total = tennis + badminton - both + neither :=
by sorry

end NUMINAMATH_CALUDE_badminton_players_l101_10142


namespace NUMINAMATH_CALUDE_min_champion_wins_l101_10171

theorem min_champion_wins (n : ℕ) (h : n = 10 ∨ n = 11) :
  let min_wins := (n / 2 : ℚ).ceil.toNat + 1
  ∀ k : ℕ, (∀ i : ℕ, i < n → i ≠ k → (n - 1).choose 2 ≤ k + i * (k - 1)) →
    min_wins ≤ k := by
  sorry

end NUMINAMATH_CALUDE_min_champion_wins_l101_10171


namespace NUMINAMATH_CALUDE_max_value_of_expression_l101_10100

theorem max_value_of_expression (n : ℤ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  3 * (500 - n) ≤ 1200 ∧ ∃ (m : ℤ), 100 ≤ m ∧ m ≤ 999 ∧ 3 * (500 - m) = 1200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l101_10100


namespace NUMINAMATH_CALUDE_friendly_angle_values_l101_10133

-- Define a friendly triangle
def is_friendly_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ (a = 2*b ∨ b = 2*c ∨ c = 2*a)

-- Theorem statement
theorem friendly_angle_values :
  ∀ a b c : ℝ,
  is_friendly_triangle a b c →
  (a = 42 ∨ b = 42 ∨ c = 42) →
  (a = 42 ∨ a = 84 ∨ a = 92 ∨
   b = 42 ∨ b = 84 ∨ b = 92 ∨
   c = 42 ∨ c = 84 ∨ c = 92) :=
by sorry

end NUMINAMATH_CALUDE_friendly_angle_values_l101_10133


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l101_10144

theorem complex_magnitude_problem (z : ℂ) (h : (1 - z) / (1 + z) = Complex.I) :
  Complex.abs (1 + z) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l101_10144


namespace NUMINAMATH_CALUDE_sum_equals_fourteen_thousand_minus_m_l101_10103

theorem sum_equals_fourteen_thousand_minus_m (M : ℕ) : 
  1989 + 1991 + 1993 + 1995 + 1997 + 1999 + 2001 = 14000 - M → M = 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_fourteen_thousand_minus_m_l101_10103


namespace NUMINAMATH_CALUDE_congruence_solution_l101_10183

theorem congruence_solution (n : ℕ) : n ≡ 40 [ZMOD 43] ↔ 11 * n ≡ 10 [ZMOD 43] ∧ n ≤ 42 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l101_10183


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l101_10165

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l101_10165


namespace NUMINAMATH_CALUDE_paper_folding_cutting_perimeter_ratio_l101_10174

theorem paper_folding_cutting_perimeter_ratio :
  let original_length : ℝ := 10
  let original_width : ℝ := 8
  let folded_length : ℝ := original_length / 2
  let folded_width : ℝ := original_width
  let small_rectangle_length : ℝ := folded_length
  let small_rectangle_width : ℝ := folded_width / 2
  let large_rectangle_length : ℝ := folded_length
  let large_rectangle_width : ℝ := folded_width
  let small_rectangle_perimeter : ℝ := 2 * (small_rectangle_length + small_rectangle_width)
  let large_rectangle_perimeter : ℝ := 2 * (large_rectangle_length + large_rectangle_width)
  small_rectangle_perimeter / large_rectangle_perimeter = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_cutting_perimeter_ratio_l101_10174


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l101_10134

theorem smallest_positive_solution (x : ℝ) : 
  (x^4 - 40*x^2 + 400 = 0 ∧ x > 0 ∧ ∀ y > 0, y^4 - 40*y^2 + 400 = 0 → x ≤ y) → 
  x = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l101_10134


namespace NUMINAMATH_CALUDE_phil_initial_books_l101_10153

def initial_book_count (pages_per_book : ℕ) (books_lost : ℕ) (pages_left : ℕ) : ℕ :=
  (pages_left / pages_per_book) + books_lost

theorem phil_initial_books :
  initial_book_count 100 2 800 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_phil_initial_books_l101_10153


namespace NUMINAMATH_CALUDE_framing_needed_photo_framing_proof_l101_10132

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered photo -/
theorem framing_needed (original_width original_height : ℕ) 
  (enlargement_factor : ℕ) (border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (final_width + final_height)
  let perimeter_feet := (perimeter_inches + 11) / 12  -- Rounding up to nearest foot
  perimeter_feet

/-- Proves that a 5x7 inch photo, quadrupled and with 3-inch border, requires 10 feet of framing -/
theorem photo_framing_proof :
  framing_needed 5 7 4 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_framing_needed_photo_framing_proof_l101_10132


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l101_10124

/-- The volume of a cylinder formed by rotating a rectangle about its lengthwise axis -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (length_positive : 0 < length) (width_positive : 0 < width) :
  let radius : ℝ := width / 2
  let height : ℝ := length
  let volume : ℝ := π * radius^2 * height
  (length = 16 ∧ width = 8) → volume = 256 * π := by
  sorry

#check cylinder_volume_from_rectangle

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l101_10124


namespace NUMINAMATH_CALUDE_symmetric_complex_division_l101_10126

/-- Two complex numbers are symmetric with respect to y = x if their real and imaginary parts are swapped -/
def symmetric_wrt_y_eq_x (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

/-- The main theorem -/
theorem symmetric_complex_division (z₁ z₂ : ℂ) 
  (h_sym : symmetric_wrt_y_eq_x z₁ z₂) (h_z₁ : z₁ = 1 + 2*I) : 
  z₁ / z₂ = 4/5 + 3/5*I :=
sorry

end NUMINAMATH_CALUDE_symmetric_complex_division_l101_10126


namespace NUMINAMATH_CALUDE_gcd_45_75_l101_10152

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l101_10152


namespace NUMINAMATH_CALUDE_line_perp_plane_sufficient_condition_l101_10102

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- Theorem statement
theorem line_perp_plane_sufficient_condition 
  (m n : Line) (α : Plane) :
  para m n → perp n α → perp m α := by
  sorry

end NUMINAMATH_CALUDE_line_perp_plane_sufficient_condition_l101_10102


namespace NUMINAMATH_CALUDE_community_pantry_fraction_l101_10148

theorem community_pantry_fraction (total_donation : ℚ) 
  (crisis_fund_fraction : ℚ) (livelihood_fraction : ℚ) (contingency_amount : ℚ) :
  total_donation = 240 →
  crisis_fund_fraction = 1/2 →
  livelihood_fraction = 1/4 →
  contingency_amount = 30 →
  (total_donation - crisis_fund_fraction * total_donation - 
   livelihood_fraction * (total_donation - crisis_fund_fraction * total_donation) - 
   contingency_amount) / total_donation = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_community_pantry_fraction_l101_10148


namespace NUMINAMATH_CALUDE_line_separate_from_circle_l101_10111

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of a point being inside a circle -/
def inside_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

/-- Definition of a line being separate from a circle -/
def separate_from_circle (l : Line) (c : Circle) : Prop :=
  let d := |l.a * c.center.1 + l.b * c.center.2 + l.c| / Real.sqrt (l.a^2 + l.b^2)
  d > c.radius

/-- Main theorem -/
theorem line_separate_from_circle 
  (a : ℝ) 
  (h_a : a > 0) 
  (M : ℝ × ℝ) 
  (h_M : inside_circle M ⟨⟨0, 0⟩, a, h_a⟩) 
  (h_M_not_center : M ≠ (0, 0)) :
  separate_from_circle ⟨M.1, M.2, -a^2⟩ ⟨⟨0, 0⟩, a, h_a⟩ :=
sorry

end NUMINAMATH_CALUDE_line_separate_from_circle_l101_10111


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_proof_l101_10121

theorem arithmetic_sequence_sum_proof : 
  let n : ℕ := 10
  let a : ℕ := 70
  let d : ℕ := 3
  let l : ℕ := 97
  3 * (n / 2 * (a + l)) = 2505 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_proof_l101_10121


namespace NUMINAMATH_CALUDE_marbles_left_l101_10194

def initial_marbles : ℕ := 143
def marbles_given : ℕ := 73

theorem marbles_left : initial_marbles - marbles_given = 70 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_l101_10194


namespace NUMINAMATH_CALUDE_min_square_area_is_49_l101_10184

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circle with diameter -/
structure Circle where
  diameter : ℝ

/-- Calculates the minimum square side length to contain given shapes -/
def minSquareSideLength (rect1 rect2 : Rectangle) (circle : Circle) : ℝ :=
  sorry

/-- Theorem: The minimum area of the square containing the given shapes is 49 -/
theorem min_square_area_is_49 : 
  let rect1 : Rectangle := ⟨2, 4⟩
  let rect2 : Rectangle := ⟨3, 5⟩
  let circle : Circle := ⟨3⟩
  (minSquareSideLength rect1 rect2 circle) ^ 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_min_square_area_is_49_l101_10184


namespace NUMINAMATH_CALUDE_coffee_stock_proof_l101_10109

/-- Represents the initial stock of coffee in pounds -/
def initial_stock : ℝ := 400

/-- Represents the percentage of decaffeinated coffee in the initial stock -/
def initial_decaf_percent : ℝ := 0.25

/-- Represents the additional coffee purchase in pounds -/
def additional_purchase : ℝ := 100

/-- Represents the percentage of decaffeinated coffee in the additional purchase -/
def additional_decaf_percent : ℝ := 0.60

/-- Represents the final percentage of decaffeinated coffee in the total stock -/
def final_decaf_percent : ℝ := 0.32

theorem coffee_stock_proof :
  initial_stock * initial_decaf_percent + additional_purchase * additional_decaf_percent =
  final_decaf_percent * (initial_stock + additional_purchase) :=
by sorry

end NUMINAMATH_CALUDE_coffee_stock_proof_l101_10109


namespace NUMINAMATH_CALUDE_wrong_observation_value_l101_10116

theorem wrong_observation_value (n : ℕ) (original_mean new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 30)
  (h3 : new_mean = 30.5) :
  ∃ (wrong_value correct_value : ℝ),
    (n : ℝ) * original_mean = (n - 1 : ℝ) * original_mean + wrong_value ∧
    (n : ℝ) * new_mean = (n - 1 : ℝ) * original_mean + correct_value ∧
    wrong_value = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l101_10116


namespace NUMINAMATH_CALUDE_lock_settings_count_l101_10135

/-- The number of digits on each dial of the lock -/
def numDigits : ℕ := 10

/-- The number of dials on the lock -/
def numDials : ℕ := 4

/-- The set of all possible digits -/
def digitSet : Finset ℕ := Finset.range numDigits

/-- The set of valid first digits (excluding zero) -/
def validFirstDigits : Finset ℕ := digitSet.filter (λ x => x ≠ 0)

/-- The number of different settings possible for the lock -/
def numSettings : ℕ := validFirstDigits.card * (numDigits - 1) * (numDigits - 2) * (numDigits - 3)

theorem lock_settings_count :
  numSettings = 4536 :=
sorry

end NUMINAMATH_CALUDE_lock_settings_count_l101_10135


namespace NUMINAMATH_CALUDE_milk_price_proof_l101_10170

/-- The original price of milk before discount -/
def milk_original_price : ℝ := 10

/-- Lily's initial budget -/
def initial_budget : ℝ := 60

/-- Cost of celery -/
def celery_cost : ℝ := 5

/-- Original price of cereal -/
def cereal_original_price : ℝ := 12

/-- Discount rate for cereal -/
def cereal_discount_rate : ℝ := 0.5

/-- Cost of bread -/
def bread_cost : ℝ := 8

/-- Discount rate for milk -/
def milk_discount_rate : ℝ := 0.1

/-- Cost of one potato -/
def potato_unit_cost : ℝ := 1

/-- Number of potatoes bought -/
def potato_quantity : ℕ := 6

/-- Amount left after buying all items -/
def amount_left : ℝ := 26

theorem milk_price_proof :
  let cereal_cost := cereal_original_price * (1 - cereal_discount_rate)
  let potato_cost := potato_unit_cost * potato_quantity
  let other_items_cost := celery_cost + cereal_cost + bread_cost + potato_cost
  let total_spent := initial_budget - amount_left
  let milk_discounted_price := total_spent - other_items_cost
  milk_original_price = milk_discounted_price / (1 - milk_discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_milk_price_proof_l101_10170


namespace NUMINAMATH_CALUDE_specific_qiandu_surface_area_l101_10162

/-- Represents a right-angled triangular prism ("堑堵") with an isosceles right-angled triangle base -/
structure QianDu where
  hypotenuse : ℝ
  height : ℝ

/-- Calculates the surface area of a QianDu -/
def surface_area (qd : QianDu) : ℝ := sorry

/-- Theorem stating the surface area of a specific QianDu -/
theorem specific_qiandu_surface_area :
  ∃ (qd : QianDu), qd.hypotenuse = 2 ∧ qd.height = 2 ∧ surface_area qd = 4 * Real.sqrt 2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_specific_qiandu_surface_area_l101_10162


namespace NUMINAMATH_CALUDE_constant_term_implies_a_value_l101_10140

/-- 
Given that the constant term in the expansion of (x + a/x)(2x-1)^5 is 30, 
prove that a = 3.
-/
theorem constant_term_implies_a_value (a : ℝ) : 
  (∃ (f : ℝ → ℝ), 
    (∀ x, f x = (x + a/x) * (2*x - 1)^5) ∧ 
    (∃ c, ∀ x, f x = c + x * (f x - c) ∧ c = 30)) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_constant_term_implies_a_value_l101_10140


namespace NUMINAMATH_CALUDE_equation_solution_l101_10125

theorem equation_solution : ∃ x : ℝ, (2 / x = 3 / (x + 1)) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l101_10125


namespace NUMINAMATH_CALUDE_shaded_area_equals_unshaded_triangle_area_l101_10149

/-- The area of the shaded region in a rectangular grid with an unshaded right triangle -/
theorem shaded_area_equals_unshaded_triangle_area (width height : ℝ) :
  width = 14 ∧ height = 5 →
  let grid_area := width * height
  let triangle_area := (1 / 2) * width * height
  let shaded_area := grid_area - triangle_area
  shaded_area = triangle_area := by sorry

end NUMINAMATH_CALUDE_shaded_area_equals_unshaded_triangle_area_l101_10149


namespace NUMINAMATH_CALUDE_soccer_team_wins_solution_l101_10188

def soccer_team_wins (total_games wins losses draws : ℕ) : Prop :=
  total_games = wins + losses + draws ∧
  losses = 2 ∧
  3 * wins + draws = 46

theorem soccer_team_wins_solution :
  ∃ (wins losses draws : ℕ),
    soccer_team_wins 20 wins losses draws ∧ wins = 14 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_wins_solution_l101_10188


namespace NUMINAMATH_CALUDE_four_isosceles_triangles_l101_10136

-- Define a point in 2D space
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a triangle by its three vertices
structure Triangle :=
  (v1 : Point)
  (v2 : Point)
  (v3 : Point)

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := squaredDistance t.v1 t.v2
  let d2 := squaredDistance t.v2 t.v3
  let d3 := squaredDistance t.v3 t.v1
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the 5 triangles
def triangle1 : Triangle := ⟨⟨1, 5⟩, ⟨3, 5⟩, ⟨2, 3⟩⟩
def triangle2 : Triangle := ⟨⟨4, 3⟩, ⟨4, 5⟩, ⟨6, 3⟩⟩
def triangle3 : Triangle := ⟨⟨1, 2⟩, ⟨4, 3⟩, ⟨7, 2⟩⟩
def triangle4 : Triangle := ⟨⟨5, 1⟩, ⟨4, 3⟩, ⟨6, 1⟩⟩
def triangle5 : Triangle := ⟨⟨3, 1⟩, ⟨4, 3⟩, ⟨5, 1⟩⟩

-- Theorem to prove
theorem four_isosceles_triangles :
  (isIsosceles triangle1) ∧
  (isIsosceles triangle2) ∧
  (isIsosceles triangle3) ∧
  (¬ isIsosceles triangle4) ∧
  (isIsosceles triangle5) :=
sorry

end NUMINAMATH_CALUDE_four_isosceles_triangles_l101_10136


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l101_10164

/-- Two lines intersecting the y-axis at the same non-zero point -/
structure IntersectingLines where
  b : ℝ
  s : ℝ
  t : ℝ
  hb : b ≠ 0
  h1 : 0 = (5/2) * s + b
  h2 : 0 = (7/3) * t + b

/-- The ratio of x-intercepts is 14/15 -/
theorem x_intercept_ratio (l : IntersectingLines) : l.s / l.t = 14 / 15 := by
  sorry

#check x_intercept_ratio

end NUMINAMATH_CALUDE_x_intercept_ratio_l101_10164


namespace NUMINAMATH_CALUDE_minimum_value_and_inequality_l101_10173

def f (x m : ℝ) : ℝ := |x - m| + |x + 1|

theorem minimum_value_and_inequality {m a b c : ℝ} (h_min : ∀ x, f x m ≥ 4) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + 2*b + 3*c = m) :
  (m = -5 ∨ m = 3) ∧ (1/a + 1/(2*b) + 1/(3*c) ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_and_inequality_l101_10173


namespace NUMINAMATH_CALUDE_percentage_decrease_of_b_l101_10175

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 → 
  b > 0 → 
  a / b = 4 / 5 → 
  x = a * 1.25 → 
  m = b * (1 - p / 100) → 
  m / x = 0.6 → 
  p = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_decrease_of_b_l101_10175


namespace NUMINAMATH_CALUDE_expression_evaluation_l101_10181

theorem expression_evaluation : 101^3 + 3*(101^2) + 3*101 + 1 = 1061208 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l101_10181


namespace NUMINAMATH_CALUDE_expenses_calculation_l101_10189

/-- Represents the revenue allocation ratio -/
structure RevenueRatio :=
  (employee_salaries : ℕ)
  (stock_purchases : ℕ)
  (rent : ℕ)
  (marketing_costs : ℕ)

/-- Calculates the total amount spent on employee salaries, rent, and marketing costs -/
def calculate_expenses (revenue : ℕ) (ratio : RevenueRatio) : ℕ :=
  let total_ratio := ratio.employee_salaries + ratio.stock_purchases + ratio.rent + ratio.marketing_costs
  let unit_value := revenue / total_ratio
  (ratio.employee_salaries + ratio.rent + ratio.marketing_costs) * unit_value

/-- Theorem stating that the calculated expenses for the given revenue and ratio equal $7,800 -/
theorem expenses_calculation (revenue : ℕ) (ratio : RevenueRatio) :
  revenue = 10800 ∧ 
  ratio = { employee_salaries := 3, stock_purchases := 5, rent := 2, marketing_costs := 8 } →
  calculate_expenses revenue ratio = 7800 :=
by sorry

end NUMINAMATH_CALUDE_expenses_calculation_l101_10189


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l101_10168

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  q > 0 →  -- positive common ratio
  a 1 = 2 →  -- given condition
  4 * a 2 * a 8 = (a 4) ^ 2 →  -- given condition
  a 3 = (1 : ℝ) / 2 := by  -- conclusion to prove
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l101_10168


namespace NUMINAMATH_CALUDE_harvard_applicants_l101_10179

/-- The number of students who choose to attend Harvard University -/
def students_attending : ℕ := 900

/-- The acceptance rate for Harvard University applicants -/
def acceptance_rate : ℚ := 5 / 100

/-- The percentage of accepted students who choose to attend Harvard University -/
def attendance_rate : ℚ := 90 / 100

/-- The number of students who applied to Harvard University -/
def applicants : ℕ := 20000

theorem harvard_applicants :
  (↑students_attending : ℚ) = (↑applicants : ℚ) * acceptance_rate * attendance_rate := by
  sorry

end NUMINAMATH_CALUDE_harvard_applicants_l101_10179


namespace NUMINAMATH_CALUDE_down_payment_proof_l101_10187

-- Define the number of people
def num_people : ℕ := 3

-- Define the individual payment amount
def individual_payment : ℚ := 1166.67

-- Function to round to nearest dollar
def round_to_dollar (x : ℚ) : ℕ := 
  (x + 0.5).floor.toNat

-- Define the total down payment
def total_down_payment : ℕ := num_people * round_to_dollar individual_payment

-- Theorem statement
theorem down_payment_proof : total_down_payment = 3501 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_proof_l101_10187


namespace NUMINAMATH_CALUDE_no_power_of_three_l101_10161

theorem no_power_of_three (a b : ℕ+) : ¬∃ k : ℕ, (15 * a + b) * (a + 15 * b) = 3^k := by
  sorry

end NUMINAMATH_CALUDE_no_power_of_three_l101_10161


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l101_10101

/-- The probability of selecting a matching pair of shoes from a box with 7 pairs -/
theorem matching_shoes_probability (n : ℕ) (total : ℕ) (pairs : ℕ) : 
  n = 7 → total = 2 * n → pairs = n → 
  (pairs : ℚ) / (total.choose 2 : ℚ) = 1 / 13 := by
  sorry

#check matching_shoes_probability

end NUMINAMATH_CALUDE_matching_shoes_probability_l101_10101


namespace NUMINAMATH_CALUDE_cupboard_cost_price_correct_l101_10192

/-- The cost price of a cupboard satisfying the given conditions -/
def cupboard_cost_price : ℝ :=
  let below_cost_percentage : ℝ := 0.12
  let profit_percentage : ℝ := 0.12
  let additional_amount : ℝ := 1650
  
  -- Define the selling price as a function of the cost price
  let selling_price (cost : ℝ) : ℝ := cost * (1 - below_cost_percentage)
  
  -- Define the new selling price (with profit) as a function of the cost price
  let new_selling_price (cost : ℝ) : ℝ := cost * (1 + profit_percentage)
  
  -- The cost price that satisfies the conditions
  6875

/-- Theorem stating that the calculated cost price satisfies the given conditions -/
theorem cupboard_cost_price_correct : 
  let cost := cupboard_cost_price
  let below_cost_percentage : ℝ := 0.12
  let profit_percentage : ℝ := 0.12
  let additional_amount : ℝ := 1650
  let selling_price := cost * (1 - below_cost_percentage)
  let new_selling_price := cost * (1 + profit_percentage)
  (new_selling_price - selling_price = additional_amount) ∧ 
  (cost = 6875) :=
by sorry

#eval cupboard_cost_price

end NUMINAMATH_CALUDE_cupboard_cost_price_correct_l101_10192


namespace NUMINAMATH_CALUDE_square_side_length_l101_10120

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) : 
  rectangle_length = 9 →
  rectangle_width = 16 →
  square_side * square_side = rectangle_length * rectangle_width →
  square_side = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l101_10120


namespace NUMINAMATH_CALUDE_odd_integers_sum_13_to_45_l101_10190

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  (a₁ + aₙ) * n / 2

theorem odd_integers_sum_13_to_45 :
  arithmetic_sum 13 45 2 = 493 := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_sum_13_to_45_l101_10190


namespace NUMINAMATH_CALUDE_tan_two_beta_l101_10131

theorem tan_two_beta (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2) 
  (h2 : Real.tan (α - β) = 3) : 
  Real.tan (2 * β) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_beta_l101_10131


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l101_10199

theorem fraction_ratio_equality : ∃ (x y : ℚ), 
  (x / y) / (7 / 15) = ((5 / 3) / ((2 / 3) - (1 / 4))) / ((1 / 3 + 1 / 6) / (1 / 2 - 1 / 3)) ∧
  x / y = 28 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l101_10199


namespace NUMINAMATH_CALUDE_long_distance_call_cost_per_minute_l101_10146

/-- Calculates the cost per minute of a long distance call given the initial card value,
    call duration, and remaining credit. -/
def cost_per_minute (initial_value : ℚ) (call_duration : ℚ) (remaining_credit : ℚ) : ℚ :=
  (initial_value - remaining_credit) / call_duration

/-- Proves that the cost per minute for long distance calls is $0.16 given the specified conditions. -/
theorem long_distance_call_cost_per_minute :
  let initial_value : ℚ := 30
  let call_duration : ℚ := 22
  let remaining_credit : ℚ := 26.48
  cost_per_minute initial_value call_duration remaining_credit = 0.16 := by
  sorry

#eval cost_per_minute 30 22 26.48

end NUMINAMATH_CALUDE_long_distance_call_cost_per_minute_l101_10146


namespace NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l101_10115

theorem at_least_one_fraction_less_than_two (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l101_10115


namespace NUMINAMATH_CALUDE_additional_marbles_needed_l101_10154

/-- The number of friends James has -/
def num_friends : ℕ := 15

/-- The initial number of marbles James has -/
def initial_marbles : ℕ := 80

/-- The function to calculate the sum of first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the number of additional marbles needed -/
theorem additional_marbles_needed : 
  sum_first_n num_friends - initial_marbles = 40 := by
  sorry

end NUMINAMATH_CALUDE_additional_marbles_needed_l101_10154


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l101_10155

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l101_10155


namespace NUMINAMATH_CALUDE_cow_to_horse_ratio_l101_10159

def total_animals : ℕ := 168
def num_cows : ℕ := 140

theorem cow_to_horse_ratio :
  let num_horses := total_animals - num_cows
  num_cows / num_horses = 5 := by
sorry

end NUMINAMATH_CALUDE_cow_to_horse_ratio_l101_10159


namespace NUMINAMATH_CALUDE_remove_five_blocks_count_l101_10186

/-- Represents the number of exposed blocks after removing n blocks -/
def E (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 * n + 1

/-- Represents the number of blocks in the k-th layer from the top -/
def blocks_in_layer (k : ℕ) : ℕ := 4^(k-1)

/-- The total number of ways to remove 5 blocks from the stack -/
def remove_five_blocks : ℕ := 
  (E 0) * (E 1) * (E 2) * (E 3) * (E 4) - (E 0) * (blocks_in_layer 2) * (blocks_in_layer 2) * (blocks_in_layer 2) * (blocks_in_layer 2)

theorem remove_five_blocks_count : remove_five_blocks = 3384 := by
  sorry

end NUMINAMATH_CALUDE_remove_five_blocks_count_l101_10186


namespace NUMINAMATH_CALUDE_coordinates_of_C_l101_10160

-- Define the points
def A : ℝ × ℝ := (11, 9)
def B : ℝ × ℝ := (2, -3)
def D : ℝ × ℝ := (-1, 3)

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  -- AB = AC
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  -- D is on BC
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) ∧
  -- AD is perpendicular to BC
  (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0

-- Theorem statement
theorem coordinates_of_C :
  ∃ C : ℝ × ℝ, triangle_ABC C ∧ C = (-4, 9) := by sorry

end NUMINAMATH_CALUDE_coordinates_of_C_l101_10160


namespace NUMINAMATH_CALUDE_precision_of_0_598_l101_10128

/-- Represents the precision of a decimal number -/
inductive Precision
  | Whole
  | Tenth
  | Hundredth
  | Thousandth
  | TenThousandth
  deriving Repr

/-- Determines the precision of an approximate number -/
def precision (x : Float) : Precision :=
  match x.toString.split (· == '.') with
  | [_, decimal] =>
    match decimal.length with
    | 1 => Precision.Tenth
    | 2 => Precision.Hundredth
    | 3 => Precision.Thousandth
    | 4 => Precision.TenThousandth
    | _ => Precision.Whole
  | _ => Precision.Whole

theorem precision_of_0_598 :
  precision 0.598 = Precision.Thousandth := by
  sorry

end NUMINAMATH_CALUDE_precision_of_0_598_l101_10128


namespace NUMINAMATH_CALUDE_simplify_expression_l101_10196

theorem simplify_expression (a : ℝ) (h1 : a ≠ -3) (h2 : a ≠ 1) :
  (1 - 4 / (a + 3)) / ((a^2 - 2*a + 1) / (2*a + 6)) = 2 / (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l101_10196


namespace NUMINAMATH_CALUDE_speedster_convertibles_l101_10172

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) :
  4 * speedsters = 3 * total →
  5 * convertibles = 3 * speedsters →
  total - speedsters = 30 →
  convertibles = 54 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertibles_l101_10172


namespace NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l101_10167

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + 5*x

-- Theorem for part I
theorem solution_set_part_I :
  {x : ℝ | |x + 1| + 5*x ≤ 5*x + 3} = Set.Icc (-4) 2 := by sorry

-- Theorem for part II
theorem range_of_a_part_II :
  ∀ a : ℝ, (∀ x ≥ -1, f a x ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) := by sorry

end NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l101_10167


namespace NUMINAMATH_CALUDE_train_crossing_time_l101_10127

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 120 → 
  train_speed_kmh = 48 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 9 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l101_10127


namespace NUMINAMATH_CALUDE_sum_simplification_l101_10129

theorem sum_simplification :
  (296 + 297 + 298 + 299 + 1 + 2 + 3 + 4 = 1200) ∧
  (457 + 458 + 459 + 460 + 461 + 462 + 463 = 3220) := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l101_10129


namespace NUMINAMATH_CALUDE_disjunction_and_negation_implication_l101_10185

theorem disjunction_and_negation_implication (p q : Prop) :
  (p ∨ q) → ¬p → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_and_negation_implication_l101_10185


namespace NUMINAMATH_CALUDE_situps_problem_l101_10176

/-- Situps problem -/
theorem situps_problem (diana_rate hani_rate total_situps : ℕ) 
  (h1 : hani_rate = diana_rate + 3)
  (h2 : diana_rate = 4)
  (h3 : total_situps = 110) :
  diana_rate * (total_situps / (diana_rate + hani_rate)) = 40 := by
  sorry

#check situps_problem

end NUMINAMATH_CALUDE_situps_problem_l101_10176


namespace NUMINAMATH_CALUDE_janes_numbers_l101_10139

def is_valid_number (n : ℕ) : Prop :=
  n % 180 = 0 ∧ n % 42 = 0 ∧ 500 < n ∧ n < 4000

theorem janes_numbers :
  {n : ℕ | is_valid_number n} = {1260, 2520, 3780} :=
sorry

end NUMINAMATH_CALUDE_janes_numbers_l101_10139


namespace NUMINAMATH_CALUDE_intersection_rectangular_prisms_cubes_l101_10158

-- Define the set of all rectangular prisms
def rectangular_prisms : Set (ℝ × ℝ × ℝ) := {p | ∃ l w h, p = (l, w, h) ∧ l > 0 ∧ w > 0 ∧ h > 0}

-- Define the set of all cubes
def cubes : Set (ℝ × ℝ × ℝ) := {c | ∃ s, c = (s, s, s) ∧ s > 0}

-- Theorem statement
theorem intersection_rectangular_prisms_cubes :
  rectangular_prisms ∩ cubes = cubes :=
by sorry

end NUMINAMATH_CALUDE_intersection_rectangular_prisms_cubes_l101_10158


namespace NUMINAMATH_CALUDE_probability_not_snow_l101_10169

theorem probability_not_snow (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snow_l101_10169


namespace NUMINAMATH_CALUDE_face_value_calculation_l101_10110

/-- Given a banker's discount and true discount, calculate the face value (sum due) -/
def calculate_face_value (bankers_discount true_discount : ℚ) : ℚ :=
  (bankers_discount * true_discount) / (bankers_discount - true_discount)

/-- Theorem stating that given a banker's discount of 144 and a true discount of 120, the face value is 840 -/
theorem face_value_calculation (bankers_discount true_discount : ℚ) 
  (h1 : bankers_discount = 144)
  (h2 : true_discount = 120) :
  calculate_face_value bankers_discount true_discount = 840 := by
sorry

end NUMINAMATH_CALUDE_face_value_calculation_l101_10110


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l101_10122

/-- Proves that the expansion of (3y-2)*(5y^12+3y^11+5y^10+3y^9) equals 15y^13 - y^12 + 9y^11 - y^10 + 6y^9 for all real y. -/
theorem polynomial_expansion_equality (y : ℝ) : 
  (3*y - 2) * (5*y^12 + 3*y^11 + 5*y^10 + 3*y^9) = 
  15*y^13 - y^12 + 9*y^11 - y^10 + 6*y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l101_10122


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l101_10193

theorem sum_of_reciprocals_positive (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * (c + d) + d * c * (a + b) + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l101_10193


namespace NUMINAMATH_CALUDE_doras_stickers_solve_l101_10108

/-- The number of packs of stickers Dora gets -/
def doras_stickers (allowance : ℕ) (card_cost : ℕ) (sticker_box_cost : ℕ) : ℕ :=
  let total_money := 2 * allowance
  let remaining_money := total_money - card_cost
  let boxes_bought := remaining_money / sticker_box_cost
  boxes_bought / 2

theorem doras_stickers_solve :
  doras_stickers 9 10 2 = 2 := by
  sorry

#eval doras_stickers 9 10 2

end NUMINAMATH_CALUDE_doras_stickers_solve_l101_10108


namespace NUMINAMATH_CALUDE_unique_even_square_Q_l101_10106

/-- Definition of the polynomial Q --/
def Q (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 3*x + 25

/-- Predicate for x being even --/
def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k

/-- Theorem stating that there exists exactly one even integer x such that Q(x) is a perfect square --/
theorem unique_even_square_Q : ∃! x : ℤ, is_even x ∧ ∃ y : ℤ, Q x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_even_square_Q_l101_10106


namespace NUMINAMATH_CALUDE_jackson_inbox_problem_l101_10156

theorem jackson_inbox_problem (initial_deleted : ℕ) (initial_received : ℕ)
  (subsequent_deleted : ℕ) (subsequent_received : ℕ) (final_count : ℕ)
  (h1 : initial_deleted = 50)
  (h2 : initial_received = 15)
  (h3 : subsequent_deleted = 20)
  (h4 : subsequent_received = 5)
  (h5 : final_count = 30) :
  final_count - (initial_received + subsequent_received) = 10 := by
  sorry

end NUMINAMATH_CALUDE_jackson_inbox_problem_l101_10156


namespace NUMINAMATH_CALUDE_worm_gnawed_pages_in_four_volumes_l101_10112

/-- Represents a book volume with a specific number of pages -/
structure Volume :=
  (pages : ℕ)

/-- Represents a bookshelf with a list of volumes -/
structure Bookshelf :=
  (volumes : List Volume)

/-- Calculates the number of pages a worm gnaws through in a bookshelf -/
def wormGnawedPages (shelf : Bookshelf) : ℕ :=
  match shelf.volumes with
  | [] => 0
  | [_] => 0
  | v1 :: vs :: tail => 
    (vs.pages + (match tail with
                 | [_] => 0
                 | v3 :: _ => v3.pages
                 | _ => 0))

/-- Theorem stating the number of pages gnawed by the worm -/
theorem worm_gnawed_pages_in_four_volumes : 
  ∀ (shelf : Bookshelf),
    shelf.volumes.length = 4 →
    (∀ v ∈ shelf.volumes, v.pages = 200) →
    wormGnawedPages shelf = 400 := by
  sorry


end NUMINAMATH_CALUDE_worm_gnawed_pages_in_four_volumes_l101_10112


namespace NUMINAMATH_CALUDE_function_properties_l101_10141

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x * Real.exp 1

-- State the theorem
theorem function_properties :
  -- Part 1: The constant a is 2
  (∃ a : ℝ, (deriv (f a)) 0 = -1 ∧ a = 2) ∧
  -- Part 2: For x > 0, x^2 < e^x
  (∀ x : ℝ, x > 0 → x^2 < Real.exp x) ∧
  -- Part 3: For any positive c, there exists x₀ such that for x ∈ (x₀, +∞), x^2 < ce^x
  (∀ c : ℝ, c > 0 → ∃ x₀ : ℝ, ∀ x : ℝ, x > x₀ → x^2 < c * Real.exp x) :=
by sorry

end

end NUMINAMATH_CALUDE_function_properties_l101_10141


namespace NUMINAMATH_CALUDE_counterexample_exists_l101_10166

theorem counterexample_exists : ∃ a : ℝ, (abs a > 2) ∧ (a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l101_10166


namespace NUMINAMATH_CALUDE_chandler_saves_for_bike_l101_10123

/-- The number of weeks needed for Chandler to save enough money to buy the mountain bike -/
def weeks_to_save : ℕ → Prop :=
  λ w => 
    let bike_cost : ℕ := 600
    let birthday_money : ℕ := 60 + 40 + 20
    let weekly_earnings : ℕ := 20
    let weekly_expenses : ℕ := 4
    let weekly_savings : ℕ := weekly_earnings - weekly_expenses
    birthday_money + w * weekly_savings = bike_cost

theorem chandler_saves_for_bike : weeks_to_save 30 := by
  sorry

end NUMINAMATH_CALUDE_chandler_saves_for_bike_l101_10123


namespace NUMINAMATH_CALUDE_count_special_numbers_l101_10117

theorem count_special_numbers : ∃ (S : Finset Nat),
  (∀ n ∈ S, n < 500 ∧ n % 5 = 0 ∧ n % 10 ≠ 0 ∧ n % 15 ≠ 0) ∧
  (∀ n < 500, n % 5 = 0 ∧ n % 10 ≠ 0 ∧ n % 15 ≠ 0 → n ∈ S) ∧
  S.card = 33 :=
by
  sorry

#check count_special_numbers

end NUMINAMATH_CALUDE_count_special_numbers_l101_10117


namespace NUMINAMATH_CALUDE_rhombus_area_l101_10197

/-- The area of a rhombus with longer diagonal 30 units and angle 60° between diagonals is 225√3 square units -/
theorem rhombus_area (d₁ : ℝ) (θ : ℝ) (h₁ : d₁ = 30) (h₂ : θ = Real.pi / 3) :
  let d₂ := d₁ * Real.sin θ
  d₁ * d₂ / 2 = 225 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l101_10197


namespace NUMINAMATH_CALUDE_sum_of_cubes_equality_l101_10198

def original_equation (x : ℝ) : Prop :=
  x * Real.rpow x (1/3) + 4*x - 9 * Real.rpow x (1/3) + 2 = 0

def transformed_equation (y : ℝ) : Prop :=
  y^4 + 4*y^3 - 9*y + 2 = 0

def roots_original : Set ℝ :=
  {x : ℝ | original_equation x ∧ x ≥ 0}

def roots_transformed : Set ℝ :=
  {y : ℝ | transformed_equation y ∧ y ≥ 0}

theorem sum_of_cubes_equality :
  ∀ (x₁ x₂ x₃ x₄ : ℝ) (y₁ y₂ y₃ y₄ : ℝ),
  roots_original = {x₁, x₂, x₃, x₄} →
  roots_transformed = {y₁, y₂, y₃, y₄} →
  x₁^3 + x₂^3 + x₃^3 + x₄^3 = y₁^9 + y₂^9 + y₃^9 + y₄^9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equality_l101_10198


namespace NUMINAMATH_CALUDE_log_23_between_consecutive_integers_sum_l101_10147

theorem log_23_between_consecutive_integers_sum : ∃ (c d : ℤ), 
  c + 1 = d ∧ 
  (c : ℝ) < Real.log 23 / Real.log 10 ∧ 
  Real.log 23 / Real.log 10 < (d : ℝ) ∧ 
  c + d = 3 := by sorry

end NUMINAMATH_CALUDE_log_23_between_consecutive_integers_sum_l101_10147


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_less_than_one_l101_10163

theorem inequality_solution_implies_m_less_than_one :
  (∃ x : ℝ, |x + 2| - |x + 3| > m) → m < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_less_than_one_l101_10163


namespace NUMINAMATH_CALUDE_max_tshirts_purchased_l101_10151

def tshirt_cost : ℚ := 915 / 100
def total_spent : ℚ := 201

theorem max_tshirts_purchased : 
  ⌊total_spent / tshirt_cost⌋ = 21 := by sorry

end NUMINAMATH_CALUDE_max_tshirts_purchased_l101_10151


namespace NUMINAMATH_CALUDE_correct_evolution_process_l101_10178

-- Define the types of population growth models
inductive PopulationGrowthModel
| Primitive
| Traditional
| Modern

-- Define the characteristics of each model
structure ModelCharacteristics where
  productiveForces : ℕ
  disasterResistance : ℕ
  birthRate : ℕ
  deathRate : ℕ
  economicLevel : ℕ
  socialSecurity : ℕ

-- Define the evolution process
def evolutionProcess : List PopulationGrowthModel :=
  [PopulationGrowthModel.Primitive, PopulationGrowthModel.Traditional, PopulationGrowthModel.Modern]

-- Define the characteristics for each model
def primitiveCharacteristics : ModelCharacteristics :=
  { productiveForces := 1, disasterResistance := 1, birthRate := 3, deathRate := 3,
    economicLevel := 1, socialSecurity := 1 }

def traditionalCharacteristics : ModelCharacteristics :=
  { productiveForces := 2, disasterResistance := 2, birthRate := 3, deathRate := 1,
    economicLevel := 2, socialSecurity := 2 }

def modernCharacteristics : ModelCharacteristics :=
  { productiveForces := 3, disasterResistance := 3, birthRate := 1, deathRate := 1,
    economicLevel := 3, socialSecurity := 3 }

-- Theorem stating that the evolution process is correct
theorem correct_evolution_process :
  evolutionProcess = [PopulationGrowthModel.Primitive, PopulationGrowthModel.Traditional, PopulationGrowthModel.Modern] :=
by sorry

end NUMINAMATH_CALUDE_correct_evolution_process_l101_10178


namespace NUMINAMATH_CALUDE_intersection_points_on_line_l101_10114

-- Define the system of equations
def system (t x y : ℝ) : Prop :=
  (3 * x - 2 * y = 8 * t - 5) ∧
  (2 * x + 3 * y = 6 * t + 9) ∧
  (x + y = 2 * t + 1)

-- Theorem statement
theorem intersection_points_on_line :
  ∀ (t x y : ℝ), system t x y → y = -1/6 * x + 8/5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_line_l101_10114


namespace NUMINAMATH_CALUDE_train_length_l101_10119

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 * 1000 / 3600 →
  crossing_time = 16.7986561075114 →
  bridge_length = 170 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.0000000001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l101_10119


namespace NUMINAMATH_CALUDE_container_volume_increase_l101_10104

/-- Given a cylindrical container with volume V = πr²h that holds 3 gallons,
    prove that a new container with triple the radius and double the height holds 54 gallons. -/
theorem container_volume_increase (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 3 → π * (3*r)^2 * (2*h) = 54 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_increase_l101_10104


namespace NUMINAMATH_CALUDE_weight_of_larger_square_l101_10182

/-- Represents the properties of a square metal piece -/
structure MetalSquare where
  side : ℝ  -- side length in inches
  weight : ℝ  -- weight in ounces

/-- Theorem stating the relationship between two metal squares -/
theorem weight_of_larger_square 
  (small : MetalSquare) 
  (large : MetalSquare) 
  (h1 : small.side = 4) 
  (h2 : small.weight = 16) 
  (h3 : large.side = 6) 
  (h_uniform : ∀ (s1 s2 : MetalSquare), s1.weight / (s1.side ^ 2) = s2.weight / (s2.side ^ 2)) :
  large.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_larger_square_l101_10182


namespace NUMINAMATH_CALUDE_opposite_pairs_l101_10145

theorem opposite_pairs : 
  ((-3)^2 = -(-3^2)) ∧ 
  ((-3)^2 ≠ -(3^2)) ∧ 
  ((-2)^3 ≠ -(-2^3)) ∧ 
  (|-2|^3 ≠ -(|-2^3|)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l101_10145


namespace NUMINAMATH_CALUDE_thirteen_thousand_one_hundred_twenty_one_obtainable_twelve_thousand_one_hundred_thirty_one_not_obtainable_l101_10191

/-- The set of numbers that can be written on the blackboard -/
inductive BoardNumber : ℕ → Prop where
  | one : BoardNumber 1
  | two : BoardNumber 2
  | add (m n : ℕ) : BoardNumber m → BoardNumber n → BoardNumber (m + n + m * n)

/-- A number is obtainable if it's in the set of BoardNumbers -/
def Obtainable (n : ℕ) : Prop := BoardNumber n

theorem thirteen_thousand_one_hundred_twenty_one_obtainable :
  Obtainable 13121 :=
sorry

theorem twelve_thousand_one_hundred_thirty_one_not_obtainable :
  ¬ Obtainable 12131 :=
sorry

end NUMINAMATH_CALUDE_thirteen_thousand_one_hundred_twenty_one_obtainable_twelve_thousand_one_hundred_thirty_one_not_obtainable_l101_10191


namespace NUMINAMATH_CALUDE_three_hour_therapy_charge_l101_10107

def therapy_charge (first_hour_rate : ℕ) (additional_hour_rate : ℕ) (hours : ℕ) : ℕ :=
  first_hour_rate + (hours - 1) * additional_hour_rate

theorem three_hour_therapy_charge :
  ∀ (first_hour_rate additional_hour_rate : ℕ),
    first_hour_rate = additional_hour_rate + 20 →
    therapy_charge first_hour_rate additional_hour_rate 5 = 300 →
    therapy_charge first_hour_rate additional_hour_rate 3 = 188 :=
by
  sorry

#check three_hour_therapy_charge

end NUMINAMATH_CALUDE_three_hour_therapy_charge_l101_10107
